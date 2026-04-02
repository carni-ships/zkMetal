// Subproduct tree GPU kernels for fast multi-point polynomial evaluation
// Build subproduct tree bottom-up, then remainder tree top-down.

#include "../fields/bn254_fr.metal"

// --- Build subproduct tree ---

// Level 0 → 1: Create degree-2 products from pairs of linear factors.
// Input: n points. Output: n/2 degree-2 polys (3 coefficients each).
// (x - p_{2i})(x - p_{2i+1}) = p_{2i}*p_{2i+1} - (p_{2i}+p_{2i+1})*x + x^2
kernel void tree_build_linear_pairs(
    device const Fr* points          [[buffer(0)]],
    device Fr* output                [[buffer(1)]],   // n/2 × 3 coefficients
    constant uint& n                 [[buffer(2)]],   // number of points (must be even)
    uint gid                         [[thread_position_in_grid]]
) {
    uint pair_idx = gid;
    if (pair_idx >= n / 2) return;

    Fr p0 = points[2 * pair_idx];
    Fr p1 = points[2 * pair_idx + 1];

    // (x - p0)(x - p1) = p0*p1 - (p0+p1)*x + x^2
    Fr c0 = fr_mul(p0, p1);
    Fr c1 = fr_sub(fr_zero(), fr_add(p0, p1));  // -(p0+p1)
    Fr c2 = fr_one();

    uint base = pair_idx * 3;
    output[base]     = c0;
    output[base + 1] = c1;
    output[base + 2] = c2;
}

// Generic schoolbook multiply for tree build at a given level.
// Multiplies count pairs of degree-d polynomials → degree-2d polynomials.
// Each thread computes one output coefficient of one output polynomial.
// Input layout: count polys of (d+1) coefficients each, contiguous.
// Output layout: count polys of (2d+1) coefficients each, contiguous.
kernel void tree_build_schoolbook(
    device const Fr* left            [[buffer(0)]],   // count polys, (d+1) coeffs each
    device const Fr* right           [[buffer(1)]],   // count polys, (d+1) coeffs each
    device Fr* output                [[buffer(2)]],   // count polys, (2d+1) coeffs each
    constant uint& d_plus_1          [[buffer(3)]],   // input poly size = d+1
    constant uint& out_size          [[buffer(4)]],   // output poly size = 2d+1
    constant uint& count             [[buffer(5)]],   // number of multiplications
    uint gid                         [[thread_position_in_grid]]
) {
    uint poly_idx = gid / out_size;
    uint coeff_idx = gid % out_size;
    if (poly_idx >= count) return;

    uint left_base = poly_idx * d_plus_1;
    uint right_base = poly_idx * d_plus_1;

    // c[k] = sum_{i+j=k, 0<=i<d+1, 0<=j<d+1} a[i] * b[j]
    Fr sum = fr_zero();
    uint start = (coeff_idx >= d_plus_1) ? (coeff_idx - d_plus_1 + 1) : 0;
    uint end = min(coeff_idx, d_plus_1 - 1);
    for (uint i = start; i <= end; i++) {
        sum = fr_add(sum, fr_mul(left[left_base + i], right[right_base + coeff_idx - i]));
    }
    output[poly_idx * out_size + coeff_idx] = sum;
}

// --- Remainder tree ---

// Polynomial reverse: rev(f) for degree d = f[d], f[d-1], ..., f[0]
// Reverses count polynomials in-place.
kernel void poly_reverse_batch(
    device Fr* polys                 [[buffer(0)]],   // count polys, poly_size coeffs each
    constant uint& poly_size         [[buffer(1)]],   // number of coefficients per poly
    constant uint& count             [[buffer(2)]],
    uint gid                         [[thread_position_in_grid]]
) {
    uint poly_idx = gid / (poly_size / 2);
    uint i = gid % (poly_size / 2);
    if (poly_idx >= count) return;

    uint base = poly_idx * poly_size;
    uint j = poly_size - 1 - i;
    Fr tmp = polys[base + i];
    polys[base + i] = polys[base + j];
    polys[base + j] = tmp;
}

// Truncate and multiply: out[i] = a[i] * b[i] for i < trunc_size, per polynomial.
// Used for Newton iteration: h * (2 - f*h) mod x^k
kernel void poly_mul_truncate(
    device const Fr* a               [[buffer(0)]],
    device const Fr* b               [[buffer(1)]],
    device Fr* out                   [[buffer(2)]],
    constant uint& size              [[buffer(3)]],   // total element count
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    out[gid] = fr_mul(a[gid], b[gid]);
}

// Compute 2 - f (element-wise) for batched polynomials.
// For each poly of size poly_stride: out[0] = 2 - f[0], out[i] = -f[i] for i > 0.
// n = total element count (count * poly_stride).
kernel void poly_two_minus(
    device const Fr* f               [[buffer(0)]],
    device Fr* out                   [[buffer(1)]],
    constant uint& n                 [[buffer(2)]],
    constant uint& poly_stride       [[buffer(3)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    if (gid % poly_stride == 0) {
        // Constant term of this polynomial: 2 - f[0]
        Fr two = fr_add(fr_one(), fr_one());
        out[gid] = fr_sub(two, f[gid]);
    } else {
        out[gid] = fr_sub(fr_zero(), f[gid]);
    }
}

// Remainder computation: r = f - q * g for batched polynomials.
// f has degree < 2d, g has degree d, q has degree < d.
// output r has degree < d.
// This kernel computes: out[i] = f[i] - (q*g)[i] for the first d coefficients.
// The q*g product is precomputed and passed in.
kernel void poly_remainder_sub(
    device const Fr* f               [[buffer(0)]],
    device const Fr* qg              [[buffer(1)]],   // q * g product
    device Fr* out                   [[buffer(2)]],
    constant uint& n                 [[buffer(3)]],   // number of elements to process
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[gid] = fr_sub(f[gid], qg[gid]);
}

// Schoolbook polynomial remainder for small degrees.
// Computes r = f mod g where deg(f) < 2*deg(g).
// Each thread computes the remainder for one polynomial pair.
// f: dividend, size 2d. g: divisor, size d+1 (monic, leading coeff = 1).
// Result: stored in first d elements of output.
kernel void poly_remainder_schoolbook(
    device const Fr* f_polys         [[buffer(0)]],   // count polys, 2*d coeffs each
    device const Fr* g_polys         [[buffer(1)]],   // count polys, (d+1) coeffs each
    device Fr* out                   [[buffer(2)]],   // count polys, d coeffs each
    constant uint& f_size            [[buffer(3)]],   // 2*d
    constant uint& g_size            [[buffer(4)]],   // d+1
    constant uint& out_size          [[buffer(5)]],   // d
    constant uint& count             [[buffer(6)]],
    uint gid                         [[thread_position_in_grid]]
) {
    // One thread per polynomial — sequential long division
    if (gid >= count) return;

    uint f_base = gid * f_size;
    uint g_base = gid * g_size;
    uint o_base = gid * out_size;

    // Copy f to output workspace (we'll modify it in-place)
    // Use a local buffer. f_size <= 2*out_size.
    // Since g is monic (leading coeff = 1), division step is just subtraction.
    // Work in device memory directly.

    // Copy f into output area (we only need the first f_size coefficients)
    // Actually, f has f_size coeffs, output has out_size coeffs.
    // We do long division from the top.

    // r = f (copy all coefficients into a temporary)
    // For degree d-1 quotient: d iterations
    // Each iteration: r[deg] gives quotient coeff, subtract g * r[deg] * x^(deg - d)

    // Since f_size can be up to 2*1024 = 2048 which is too large for thread-local,
    // we work in the output buffer + an extra workspace trick.
    // For simplicity, limit this kernel to small sizes (f_size <= 256).

    // Work array: use device output buffer as workspace
    // We need space for f_size elements. output has out_size = f_size/2 elements.
    // So we need extra space. For now, write f to output extended area...
    // Actually, let's just work with two reads from f.

    // Simple approach: synthetic division for monic divisor
    // Process from highest degree downward
    // Note: g is monic, so g[d] = 1 (not stored in g_polys which has d+1 elements including the leading 1)

    // Load f into threadgroup scratch? No, one thread per poly.
    // Just iterate. f_size is small (≤ some limit).

    // Copy f to output (first out_size elements = lower half of f)
    for (uint i = 0; i < out_size; i++) {
        out[o_base + i] = f_polys[f_base + i];
    }

    // Long division: for k = f_size-1 down to out_size (= d)
    // At step k: the leading coefficient is f[k] (or modified f[k])
    // Subtract: f[j] -= f[k] * g[j - (k - d)] for j in [k-d, k)
    // Since g is monic (g[d]=1), the leading coeff IS f[k] (no division needed)
    // After subtraction, f[k] becomes 0.

    // We need the upper half of f too. Read them from f_polys.
    // Process from k = f_size-1 downto out_size
    // But we only have the lower half in out. Upper half we read from f_polys.
    // Modified values need to propagate. This is tricky with split storage.

    // Simpler: just use the fact that f_size is small and do it with a local array.
    // For f_size up to 128 (64 × 2), that's 128 Fr = 4KB. Might be OK for GPU registers.

    // Actually with 128 × 8 uint32 = 1024 registers... too many.
    // Let's just limit to very small sizes and use device memory.

    // For larger sizes, use the NTT-based approach in Swift.
    // This kernel is for the bottom levels of the remainder tree only.

    // Pragmatic approach: use two arrays in device memory
    // 'out' stores the remainder (modified in-place), sized out_size
    // 'f_polys' has the original upper half that we read

    // Actually, let's keep it simple and correct:
    // Create local work array sized f_size (stack allocation)
    // This limits the kernel to f_size ≤ ~64 (2KB stack)
    // For larger sizes, use NTT-based remainder in Swift.

    // For now, limit to f_size ≤ 64 (handled in Swift dispatch logic)
    Fr work[128]; // max f_size = 128 (degree 64 × 2)
    for (uint i = 0; i < f_size; i++) {
        work[i] = f_polys[f_base + i];
    }

    uint d = out_size; // degree of g = d, g has d+1 coefficients
    for (uint k = f_size - 1; k >= d; k--) {
        Fr lead = work[k];
        // Subtract lead * g[j] from work[k - d + j] for j = 0..d-1
        // (g[d] = 1, so work[k] -= lead * 1 = lead → becomes 0)
        for (uint j = 0; j < d; j++) {
            work[k - d + j] = fr_sub(work[k - d + j], fr_mul(lead, g_polys[g_base + j]));
        }
        // work[k] = 0 (implicit, from monic leading term)
        if (k == 0) break; // prevent underflow since k is uint
    }

    // Copy lower d coefficients to output
    for (uint i = 0; i < out_size; i++) {
        out[o_base + i] = work[i];
    }
}
