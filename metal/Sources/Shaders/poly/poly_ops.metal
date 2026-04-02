// Polynomial operation GPU kernels for BN254 Fr
// Element-wise operations on coefficient arrays

#include "../fields/bn254_fr.metal"

// Element-wise polynomial addition: c[i] = a[i] + b[i]
kernel void poly_add(
    device const Fr* a             [[buffer(0)]],
    device const Fr* b             [[buffer(1)]],
    device Fr* c                   [[buffer(2)]],
    constant uint& n               [[buffer(3)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    c[gid] = fr_add(a[gid], b[gid]);
}

// Element-wise polynomial subtraction: c[i] = a[i] - b[i]
kernel void poly_sub(
    device const Fr* a             [[buffer(0)]],
    device const Fr* b             [[buffer(1)]],
    device Fr* c                   [[buffer(2)]],
    constant uint& n               [[buffer(3)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    c[gid] = fr_sub(a[gid], b[gid]);
}

// Element-wise (Hadamard) product: c[i] = a[i] * b[i]
// Used after NTT: poly_mul = iNTT(hadamard(NTT(a), NTT(b)))
kernel void poly_hadamard(
    device const Fr* a             [[buffer(0)]],
    device const Fr* b             [[buffer(1)]],
    device Fr* c                   [[buffer(2)]],
    constant uint& n               [[buffer(3)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    c[gid] = fr_mul(a[gid], b[gid]);
}

// Scalar multiplication: b[i] = a[i] * scalar
kernel void poly_scalar_mul(
    device const Fr* a             [[buffer(0)]],
    device Fr* b                   [[buffer(1)]],
    constant Fr* scalar            [[buffer(2)]],   // constant broadcast
    constant uint& n               [[buffer(3)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    b[gid] = fr_mul(a[gid], scalar[0]);
}

// Evaluate polynomial at a single point using Horner's method
// Each threadgroup reduces a chunk, final reduction done on CPU or separate pass.
// For single-point eval, a sequential approach is more appropriate.
// This kernel evaluates polynomial at `point` using parallel prefix.
// result = sum(coeffs[i] * point^i)
// We precompute powers of point and do a dot product.
kernel void poly_eval_powers(
    device const Fr* coeffs        [[buffer(0)]],
    device const Fr* powers        [[buffer(1)]],   // powers[i] = point^i
    device Fr* partial_sums        [[buffer(2)]],   // one partial sum per threadgroup
    constant uint& n               [[buffer(3)]],
    uint gid                       [[thread_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    // Each thread computes coeffs[gid] * powers[gid]
    threadgroup Fr shared_sum[256];

    Fr val = fr_zero();
    if (gid < n) {
        val = fr_mul(coeffs[gid], powers[gid]);
    }
    shared_sum[tid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction within threadgroup
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] = fr_add(shared_sum[tid], shared_sum[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        partial_sums[tgid] = shared_sum[0];
    }
}

// Batch polynomial evaluation: evaluate one polynomial at many points (Horner per thread).
// Each GPU thread evaluates the polynomial at one point independently.
// This is compute-bound (O(n*d) field multiplications), not memory-bound.
kernel void poly_eval_horner(
    device const Fr* coeffs        [[buffer(0)]],
    device const Fr* points        [[buffer(1)]],
    device Fr* results             [[buffer(2)]],
    constant uint& degree          [[buffer(3)]],   // polynomial degree + 1
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

// Chunked Horner: split polynomial into K chunks, K threads per point.
// Increases parallelism from num_points to num_points * POLY_CHUNKS threads.
// Each thread evaluates a sub-polynomial, then thread 0 per point combines results.
#define POLY_CHUNKS 16

kernel void poly_eval_horner_chunked(
    device const Fr* coeffs        [[buffer(0)]],
    device const Fr* points        [[buffer(1)]],
    device Fr* results             [[buffer(2)]],
    constant uint& degree          [[buffer(3)]],
    constant uint& num_points      [[buffer(4)]],
    uint gid                       [[thread_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]]
) {
    uint point_idx = gid / POLY_CHUNKS;
    uint chunk_idx = gid % POLY_CHUNKS;

    if (point_idx >= num_points) return;

    Fr x = points[point_idx];
    uint chunk_size = degree / POLY_CHUNKS;
    uint start = chunk_idx * chunk_size;
    uint end = (chunk_idx == POLY_CHUNKS - 1) ? degree : start + chunk_size;

    // Evaluate sub-polynomial at x using Horner
    Fr partial = coeffs[end - 1];
    for (uint i = end - 1; i > start; i--) {
        partial = fr_add(fr_mul(partial, x), coeffs[i - 1]);
    }

    // Store partial result in shared memory
    threadgroup Fr partials[256];
    partials[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 per point: compute x^chunk_size and combine with Horner
    if (chunk_idx == 0) {
        // x^chunk_size via repeated squaring
        Fr xs = fr_one();
        Fr base = x;
        uint s = chunk_size;
        while (s > 0) {
            if (s & 1) xs = fr_mul(xs, base);
            base = fr_sqr(base);
            s >>= 1;
        }

        // Outer Horner: p(x) = partial[0] + x^s*(partial[1] + x^s*(...))
        uint base_tid = (tid / POLY_CHUNKS) * POLY_CHUNKS;
        Fr result = partials[base_tid + POLY_CHUNKS - 1];
        for (int k = POLY_CHUNKS - 2; k >= 0; k--) {
            result = fr_add(fr_mul(result, xs), partials[base_tid + uint(k)]);
        }
        results[point_idx] = result;
    }
}

// Divide polynomial by (x - root) in-place (synthetic division)
// Assuming poly is divisible by (x - root)
// coeffs are in ascending order: c0 + c1*x + c2*x^2 + ...
// Result has degree - 1 coefficients stored in coeffs[0..degree-1]
kernel void poly_div_linear(
    device Fr* coeffs              [[buffer(0)]],
    device const Fr* root          [[buffer(1)]],   // the root value
    constant uint& degree          [[buffer(2)]],   // original degree + 1
    uint gid                       [[thread_position_in_grid]]
) {
    // Synthetic division is inherently sequential, so this kernel
    // does it for multiple polynomials in parallel (one per thread).
    // For single polynomial, just call with 1 thread.
    // This is a placeholder — real parallel divide needs different approach.
    if (gid != 0) return;

    Fr r = root[0];
    // Process from highest degree down
    // new[n-2] = old[n-1]
    // new[i] = old[i+1] + r * new[i+1]
    Fr carry = coeffs[degree - 1];
    coeffs[degree - 1] = fr_zero();
    for (uint i = degree - 2; ; i--) {
        Fr temp = coeffs[i];
        coeffs[i] = carry;
        carry = fr_add(temp, fr_mul(r, carry));  // Should be zero for last
        if (i == 0) break;
    }
}

// Batch inverse using Montgomery's trick.
// Each threadgroup inverts a chunk of up to BATCH_INV_CHUNK elements
// using only 1 Fermat inverse + 3*(chunk-1) multiplications.
// Input: n elements in a[]. Output: a[i]^(-1) written to out[].
#define BATCH_INV_CHUNK 512

kernel void batch_inverse(
    device const Fr* a             [[buffer(0)]],
    device Fr* out                 [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]]
) {
    // Only thread 0 per threadgroup does the work (Montgomery's trick is sequential)
    if (tid != 0) return;

    uint base = tgid * BATCH_INV_CHUNK;
    uint chunk = min(uint(BATCH_INV_CHUNK), n - base);
    if (chunk == 0) return;

    // Phase 1: Build prefix products in output buffer
    out[base] = a[base];
    for (uint i = 1; i < chunk; i++) {
        out[base + i] = fr_mul(out[base + i - 1], a[base + i]);
    }

    // Phase 2: Invert the total product
    Fr inv = fr_inv(out[base + chunk - 1]);

    // Phase 3: Backward sweep
    for (uint i = chunk - 1; i > 0; i--) {
        Fr ai = a[base + i];
        out[base + i] = fr_mul(inv, out[base + i - 1]);
        inv = fr_mul(inv, ai);
    }
    out[base] = inv;
}
