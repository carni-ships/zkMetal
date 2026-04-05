// Multi-point polynomial evaluation GPU kernels
//
// Three evaluation modes:
//   1. Horner: one thread per evaluation point (single poly, many points)
//   2. Batch Horner: one thread per (poly, point) pair (many polys at one point)
//   3. Cross Horner: one thread per (poly, point) in M x N matrix
//
// Subproduct tree kernels for O(n log^2 n) fast multi-point evaluation:
//   - Build subproduct tree bottom-up from linear factors
//   - Remainder tree top-down via polynomial division
//
// Supports BN254 Fr (256-bit Montgomery) and BabyBear (32-bit Barrett).

#include "../fields/bn254_fr.metal"
#include "../fields/babybear.metal"

// ============================================================
// Horner evaluation: single poly at many points
// ============================================================
// One thread per evaluation point. Each thread runs Horner's method
// independently for the full polynomial at its assigned point.

kernel void mpe_horner_bn254(
    device const Fr* coeffs        [[buffer(0)]],   // polynomial coefficients (ascending degree)
    device const Fr* points        [[buffer(1)]],   // evaluation points
    device Fr* results             [[buffer(2)]],   // output evaluations
    constant uint& degree          [[buffer(3)]],   // number of coefficients
    constant uint& num_points      [[buffer(4)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= num_points) return;

    Fr x = points[gid];
    Fr result = coeffs[degree - 1];
    for (uint i = degree - 1; i > 0; i--) {
        result = fr_add(fr_mul(result, x), coeffs[i - 1]);
    }
    results[gid] = result;
}

kernel void mpe_horner_babybear(
    device const Bb* coeffs        [[buffer(0)]],
    device const Bb* points        [[buffer(1)]],
    device Bb* results             [[buffer(2)]],
    constant uint& degree          [[buffer(3)]],
    constant uint& num_points      [[buffer(4)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= num_points) return;

    Bb x = points[gid];
    Bb result = coeffs[degree - 1];
    for (uint i = degree - 1; i > 0; i--) {
        result = bb_add(bb_mul(result, x), coeffs[i - 1]);
    }
    results[gid] = result;
}

// ============================================================
// Batch Horner: many polys at one point
// ============================================================
// One thread per polynomial. All polys must have the same degree.
// Evaluates poly[gid] at the single given point.

kernel void mpe_batch_horner_bn254(
    device const Fr* polys         [[buffer(0)]],   // M * degree coefficients packed
    device const Fr* point         [[buffer(1)]],   // single evaluation point
    device Fr* results             [[buffer(2)]],   // M output evaluations
    constant uint& degree          [[buffer(3)]],   // coefficients per polynomial
    constant uint& num_polys       [[buffer(4)]],   // M
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= num_polys) return;

    Fr x = point[0];
    device const Fr* coeffs = polys + gid * degree;

    Fr result = coeffs[degree - 1];
    for (uint i = degree - 1; i > 0; i--) {
        result = fr_add(fr_mul(result, x), coeffs[i - 1]);
    }
    results[gid] = result;
}

kernel void mpe_batch_horner_babybear(
    device const Bb* polys         [[buffer(0)]],
    device const Bb* point         [[buffer(1)]],
    device Bb* results             [[buffer(2)]],
    constant uint& degree          [[buffer(3)]],
    constant uint& num_polys       [[buffer(4)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= num_polys) return;

    Bb x = point[0];
    device const Bb* coeffs = polys + gid * degree;

    Bb result = coeffs[degree - 1];
    for (uint i = degree - 1; i > 0; i--) {
        result = bb_add(bb_mul(result, x), coeffs[i - 1]);
    }
    results[gid] = result;
}

// ============================================================
// Cross Horner: M polys at N points (full M x N matrix)
// ============================================================
// Total threads = M * N. Thread gid handles (poly_idx, point_idx).
// Output layout: results[poly_idx * num_points + point_idx].

kernel void mpe_cross_horner_bn254(
    device const Fr* polys         [[buffer(0)]],   // M * degree coefficients
    device const Fr* points        [[buffer(1)]],   // N evaluation points
    device Fr* results             [[buffer(2)]],   // M * N output evaluations
    constant uint& degree          [[buffer(3)]],   // coefficients per polynomial
    constant uint& num_points      [[buffer(4)]],   // N
    constant uint& num_polys       [[buffer(5)]],   // M
    uint gid                       [[thread_position_in_grid]]
) {
    uint total = num_polys * num_points;
    if (gid >= total) return;

    uint poly_idx = gid / num_points;
    uint point_idx = gid % num_points;

    device const Fr* coeffs = polys + poly_idx * degree;
    Fr x = points[point_idx];

    Fr result = coeffs[degree - 1];
    for (uint i = degree - 1; i > 0; i--) {
        result = fr_add(fr_mul(result, x), coeffs[i - 1]);
    }
    results[poly_idx * num_points + point_idx] = result;
}

kernel void mpe_cross_horner_babybear(
    device const Bb* polys         [[buffer(0)]],
    device const Bb* points        [[buffer(1)]],
    device Bb* results             [[buffer(2)]],
    constant uint& degree          [[buffer(3)]],
    constant uint& num_points      [[buffer(4)]],
    constant uint& num_polys       [[buffer(5)]],
    uint gid                       [[thread_position_in_grid]]
) {
    uint total = num_polys * num_points;
    if (gid >= total) return;

    uint poly_idx = gid / num_points;
    uint point_idx = gid % num_points;

    device const Bb* coeffs = polys + poly_idx * degree;
    Bb x = points[point_idx];

    Bb result = coeffs[degree - 1];
    for (uint i = degree - 1; i > 0; i--) {
        result = bb_add(bb_mul(result, x), coeffs[i - 1]);
    }
    results[poly_idx * num_points + point_idx] = result;
}

// ============================================================
// Subproduct tree — build linear factors (bottom level)
// ============================================================
// Input: n points. Output: n/2 degree-2 polynomials (3 coefficients each).
// (x - p_{2i}) * (x - p_{2i+1}) = p_{2i}*p_{2i+1} - (p_{2i}+p_{2i+1})*x + x^2

kernel void mpe_tree_linear_bn254(
    device const Fr* points          [[buffer(0)]],
    device Fr* output                [[buffer(1)]],   // n/2 * 3 coefficients
    constant uint& n                 [[buffer(2)]],   // number of points (must be even)
    uint gid                         [[thread_position_in_grid]]
) {
    uint pair_idx = gid;
    if (pair_idx >= n / 2) return;

    Fr p0 = points[2 * pair_idx];
    Fr p1 = points[2 * pair_idx + 1];

    // (x - p0)(x - p1) = p0*p1 - (p0+p1)*x + x^2
    Fr c0 = fr_mul(p0, p1);
    Fr c1 = fr_sub(fr_zero(), fr_add(p0, p1));
    Fr c2 = fr_one();

    uint base = pair_idx * 3;
    output[base]     = c0;
    output[base + 1] = c1;
    output[base + 2] = c2;
}

kernel void mpe_tree_linear_babybear(
    device const Bb* points          [[buffer(0)]],
    device Bb* output                [[buffer(1)]],
    constant uint& n                 [[buffer(2)]],
    uint gid                         [[thread_position_in_grid]]
) {
    uint pair_idx = gid;
    if (pair_idx >= n / 2) return;

    Bb p0 = points[2 * pair_idx];
    Bb p1 = points[2 * pair_idx + 1];

    Bb c0 = bb_mul(p0, p1);
    Bb c1 = bb_sub(Bb{0}, bb_add(p0, p1));
    Bb c2 = Bb{1};

    uint base = pair_idx * 3;
    output[base]     = c0;
    output[base + 1] = c1;
    output[base + 2] = c2;
}

// ============================================================
// Subproduct tree — schoolbook multiply level (generic)
// ============================================================
// Multiplies count pairs of degree-d polynomials into degree-2d polynomials.
// Each thread computes one output coefficient of one output polynomial.

kernel void mpe_tree_multiply_bn254(
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

    Fr sum = fr_zero();
    uint start = (coeff_idx >= d_plus_1) ? (coeff_idx - d_plus_1 + 1) : 0;
    uint end = min(coeff_idx, d_plus_1 - 1);
    for (uint i = start; i <= end; i++) {
        sum = fr_add(sum, fr_mul(left[left_base + i], right[right_base + coeff_idx - i]));
    }
    output[poly_idx * out_size + coeff_idx] = sum;
}

kernel void mpe_tree_multiply_babybear(
    device const Bb* left            [[buffer(0)]],
    device const Bb* right           [[buffer(1)]],
    device Bb* output                [[buffer(2)]],
    constant uint& d_plus_1          [[buffer(3)]],
    constant uint& out_size          [[buffer(4)]],
    constant uint& count             [[buffer(5)]],
    uint gid                         [[thread_position_in_grid]]
) {
    uint poly_idx = gid / out_size;
    uint coeff_idx = gid % out_size;
    if (poly_idx >= count) return;

    uint left_base = poly_idx * d_plus_1;
    uint right_base = poly_idx * d_plus_1;

    Bb sum = Bb{0};
    uint start = (coeff_idx >= d_plus_1) ? (coeff_idx - d_plus_1 + 1) : 0;
    uint end = min(coeff_idx, d_plus_1 - 1);
    for (uint i = start; i <= end; i++) {
        sum = bb_add(sum, bb_mul(left[left_base + i], right[right_base + coeff_idx - i]));
    }
    output[poly_idx * out_size + coeff_idx] = sum;
}

// ============================================================
// Subproduct tree — remainder (schoolbook, small degree)
// ============================================================
// Computes r = f mod g where deg(f) < 2*deg(g), g is monic.
// One thread per polynomial pair. For bottom levels of remainder tree.

kernel void mpe_remainder_bn254(
    device const Fr* f_polys         [[buffer(0)]],   // count polys, f_size coeffs each
    device const Fr* g_polys         [[buffer(1)]],   // count polys, g_size coeffs each
    device Fr* out                   [[buffer(2)]],   // count polys, out_size coeffs each
    constant uint& f_size            [[buffer(3)]],   // 2*d
    constant uint& g_size            [[buffer(4)]],   // d+1 (monic, leading coeff = 1)
    constant uint& out_size          [[buffer(5)]],   // d
    constant uint& count             [[buffer(6)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    uint f_base = gid * f_size;
    uint g_base = gid * g_size;
    uint o_base = gid * out_size;

    // Work array: limited to f_size <= 128
    Fr work[128];
    for (uint i = 0; i < f_size && i < 128; i++) {
        work[i] = f_polys[f_base + i];
    }

    uint d = out_size;
    for (uint k = f_size - 1; k >= d; k--) {
        Fr lead = work[k];
        for (uint j = 0; j < d; j++) {
            work[k - d + j] = fr_sub(work[k - d + j], fr_mul(lead, g_polys[g_base + j]));
        }
        if (k == 0) break;
    }

    for (uint i = 0; i < out_size; i++) {
        out[o_base + i] = work[i];
    }
}

kernel void mpe_remainder_babybear(
    device const Bb* f_polys         [[buffer(0)]],
    device const Bb* g_polys         [[buffer(1)]],
    device Bb* out                   [[buffer(2)]],
    constant uint& f_size            [[buffer(3)]],
    constant uint& g_size            [[buffer(4)]],
    constant uint& out_size          [[buffer(5)]],
    constant uint& count             [[buffer(6)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    uint f_base = gid * f_size;
    uint g_base = gid * g_size;
    uint o_base = gid * out_size;

    Bb work[128];
    for (uint i = 0; i < f_size && i < 128; i++) {
        work[i] = f_polys[f_base + i];
    }

    uint d = out_size;
    for (uint k = f_size - 1; k >= d; k--) {
        Bb lead = work[k];
        for (uint j = 0; j < d; j++) {
            work[k - d + j] = bb_sub(work[k - d + j], bb_mul(lead, g_polys[g_base + j]));
        }
        if (k == 0) break;
    }

    for (uint i = 0; i < out_size; i++) {
        out[o_base + i] = work[i];
    }
}
