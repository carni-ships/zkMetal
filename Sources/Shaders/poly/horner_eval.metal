// GPU Horner polynomial evaluation kernels
//
// Evaluate p(x) = a_0 + a_1*x + ... + a_n*x^n at multiple points simultaneously.
// Each thread evaluates at one point using Horner's method:
//   result = ((a_n * x + a_{n-1}) * x + ...) * x + a_0
//
// Supports BN254 Fr (256-bit Montgomery).

#include "../fields/bn254_fr.metal"

// ============================================================
// Single polynomial at many points
// ============================================================
// One thread per evaluation point. Each thread runs Horner's method
// for the full polynomial at its assigned point.

kernel void horner_eval_bn254(
    device const Fr* coeffs        [[buffer(0)]],   // polynomial coefficients (ascending degree)
    device const Fr* points        [[buffer(1)]],   // evaluation points
    device Fr* results             [[buffer(2)]],   // output evaluations
    constant uint& degree          [[buffer(3)]],   // number of coefficients
    constant uint& num_points      [[buffer(4)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= num_points) return;

    Fr x = points[gid];

    // Horner's method: start from highest-degree coefficient
    Fr result = coeffs[degree - 1];
    for (uint i = degree - 1; i > 0; i--) {
        result = fr_add(fr_mul(result, x), coeffs[i - 1]);
    }
    results[gid] = result;
}

// ============================================================
// Batch: evaluate M polynomials at a single point
// ============================================================
// One thread per polynomial. All polys must have the same degree.

kernel void horner_eval_batch_bn254(
    device const Fr* polys         [[buffer(0)]],   // M * degree coefficients packed contiguously
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

// ============================================================
// Multi: evaluate single polynomial at many points, chunked
// ============================================================
// Same as horner_eval_bn254 but with threadgroup-local coefficient caching
// for better memory access patterns on large polynomials.
// Each threadgroup loads coefficients into threadgroup memory once,
// then all threads in the group evaluate using the cached coefficients.

// Maximum degree for threadgroup caching (32 KB / 32 bytes per Fr = 1024)
#define HORNER_TG_MAX_DEGREE 512

kernel void horner_eval_cached_bn254(
    device const Fr* coeffs        [[buffer(0)]],
    device const Fr* points        [[buffer(1)]],
    device Fr* results             [[buffer(2)]],
    constant uint& degree          [[buffer(3)]],
    constant uint& num_points      [[buffer(4)]],
    uint gid                       [[thread_position_in_grid]],
    uint lid                       [[thread_position_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    if (gid >= num_points) return;

    // For polynomials that fit in threadgroup memory, cache coefficients
    threadgroup Fr tg_coeffs[HORNER_TG_MAX_DEGREE];

    if (degree <= HORNER_TG_MAX_DEGREE) {
        // Cooperatively load coefficients into threadgroup memory
        for (uint i = lid; i < degree; i += tg_size) {
            tg_coeffs[i] = coeffs[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        Fr x = points[gid];
        Fr result = tg_coeffs[degree - 1];
        for (uint i = degree - 1; i > 0; i--) {
            result = fr_add(fr_mul(result, x), tg_coeffs[i - 1]);
        }
        results[gid] = result;
    } else {
        // Fallback: read from device memory directly
        Fr x = points[gid];
        Fr result = coeffs[degree - 1];
        for (uint i = degree - 1; i > 0; i--) {
            result = fr_add(fr_mul(result, x), coeffs[i - 1]);
        }
        results[gid] = result;
    }
}
