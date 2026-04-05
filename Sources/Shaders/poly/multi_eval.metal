// Multi-point polynomial evaluation GPU kernels
// Evaluates polynomials at many points simultaneously using Horner's method.
// Critical for KZG openings, FRI, and STARK provers.
//
// Three field variants:
//   - BN254 Fr (8x uint32 Montgomery, 256-bit)
//   - BabyBear (single uint32, Barrett reduction)
//   - Goldilocks (uint64, special reduction)
//
// Single-poly kernels: each thread evaluates the polynomial at one point.
// Batch kernels: each thread evaluates one (poly, point) pair.

#include "../fields/bn254_fr.metal"
#include "../fields/babybear.metal"
#include "../fields/goldilocks.metal"

// ============================================================
// BN254 Fr — Single polynomial, many points
// ============================================================

kernel void poly_eval_bn254(
    device const Fr* coeffs        [[buffer(0)]],   // polynomial coefficients (ascending degree)
    device const Fr* points        [[buffer(1)]],   // evaluation points
    device Fr* results             [[buffer(2)]],   // output evaluations
    constant uint& degree          [[buffer(3)]],   // number of coefficients
    constant uint& num_points      [[buffer(4)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= num_points) return;

    Fr x = points[gid];
    // Horner: result = c[d-1]*x + c[d-2], then *x + c[d-3], ...
    Fr result = coeffs[degree - 1];
    for (uint i = degree - 1; i > 0; i--) {
        result = fr_add(fr_mul(result, x), coeffs[i - 1]);
    }
    results[gid] = result;
}

// ============================================================
// BabyBear — Single polynomial, many points
// ============================================================

kernel void poly_eval_babybear(
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
// Goldilocks — Single polynomial, many points
// ============================================================

kernel void poly_eval_goldilocks(
    device const Gl* coeffs        [[buffer(0)]],
    device const Gl* points        [[buffer(1)]],
    device Gl* results             [[buffer(2)]],
    constant uint& degree          [[buffer(3)]],
    constant uint& num_points      [[buffer(4)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= num_points) return;

    Gl x = points[gid];
    Gl result = coeffs[degree - 1];
    for (uint i = degree - 1; i > 0; i--) {
        result = gl_add(gl_mul(result, x), coeffs[i - 1]);
    }
    results[gid] = result;
}

// ============================================================
// BN254 Fr — Batch: M polynomials at N points
// ============================================================
// Layout: polys are packed contiguously, each of length `degree`.
//   polys[poly_idx * degree + coeff_idx]
// Output: results[poly_idx * num_points + point_idx]
// Each thread handles one (poly, point) pair.
// Total threads = num_polys * num_points.

kernel void poly_eval_batch_bn254(
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

// ============================================================
// BabyBear — Batch: M polynomials at N points
// ============================================================

kernel void poly_eval_batch_babybear(
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
// Goldilocks — Batch: M polynomials at N points
// ============================================================

kernel void poly_eval_batch_goldilocks(
    device const Gl* polys         [[buffer(0)]],
    device const Gl* points        [[buffer(1)]],
    device Gl* results             [[buffer(2)]],
    constant uint& degree          [[buffer(3)]],
    constant uint& num_points      [[buffer(4)]],
    constant uint& num_polys       [[buffer(5)]],
    uint gid                       [[thread_position_in_grid]]
) {
    uint total = num_polys * num_points;
    if (gid >= total) return;

    uint poly_idx = gid / num_points;
    uint point_idx = gid % num_points;

    device const Gl* coeffs = polys + poly_idx * degree;
    Gl x = points[point_idx];

    Gl result = coeffs[degree - 1];
    for (uint i = degree - 1; i > 0; i--) {
        result = gl_add(gl_mul(result, x), coeffs[i - 1]);
    }
    results[poly_idx * num_points + point_idx] = result;
}
