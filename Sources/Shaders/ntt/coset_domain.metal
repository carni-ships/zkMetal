// Coset Domain GPU kernels
// Operations: coset shift/unshift, vanishing polynomial evaluation, divide by vanishing
// Supports BN254 Fr and BabyBear fields.
//
// Coset shift: multiply each element by g^i where g is the coset generator.
// Vanishing polynomial: Z_H(x) = x^n - 1
// Divide by vanishing: out[i] = evals[i] / Z_H(coset_point[i])

#include "../fields/bn254_fr.metal"
#include "../fields/babybear.metal"

// ============================================================
// Coset shift: data[i] *= g^i (in-place)
// powers[i] = g^i precomputed on CPU
// ============================================================

kernel void coset_shift_bn254(
    device Fr* data                  [[buffer(0)]],
    device const Fr* powers          [[buffer(1)]],   // powers[i] = g^i
    constant uint& n                 [[buffer(2)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    data[gid] = fr_mul(data[gid], powers[gid]);
}

kernel void coset_shift_babybear(
    device Bb* data                  [[buffer(0)]],
    device const Bb* powers          [[buffer(1)]],
    constant uint& n                 [[buffer(2)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    data[gid] = bb_mul(data[gid], powers[gid]);
}

// ============================================================
// Coset unshift: data[i] *= g^(-i) (in-place, the inverse operation)
// inv_powers[i] = g^(-i) precomputed on CPU
// ============================================================

kernel void coset_unshift_bn254(
    device Fr* data                  [[buffer(0)]],
    device const Fr* inv_powers      [[buffer(1)]],   // inv_powers[i] = g^(-i)
    constant uint& n                 [[buffer(2)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    data[gid] = fr_mul(data[gid], inv_powers[gid]);
}

kernel void coset_unshift_babybear(
    device Bb* data                  [[buffer(0)]],
    device const Bb* inv_powers      [[buffer(1)]],
    constant uint& n                 [[buffer(2)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    data[gid] = bb_mul(data[gid], inv_powers[gid]);
}

// ============================================================
// Vanishing polynomial evaluation: Z_H(x) = x^n - 1
// Each thread evaluates Z_H at one point.
// points[i] is a field element; n is the domain size (power of 2).
// out[i] = points[i]^domain_size - 1
// ============================================================

kernel void vanishing_poly_eval_bn254(
    device const Fr* points          [[buffer(0)]],
    device Fr* out                   [[buffer(1)]],
    constant uint& num_points        [[buffer(2)]],
    constant uint& log_domain_size   [[buffer(3)]],   // domain_size = 1 << log_domain_size
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= num_points) return;
    // Compute x^(2^log_domain_size) by repeated squaring log_domain_size times
    Fr x = points[gid];
    for (uint i = 0; i < log_domain_size; i++) {
        x = fr_mul(x, x);
    }
    // x now holds points[gid]^domain_size; subtract 1
    out[gid] = fr_sub(x, fr_one());
}

kernel void vanishing_poly_eval_babybear(
    device const Bb* points          [[buffer(0)]],
    device Bb* out                   [[buffer(1)]],
    constant uint& num_points        [[buffer(2)]],
    constant uint& log_domain_size   [[buffer(3)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= num_points) return;
    Bb x = points[gid];
    for (uint i = 0; i < log_domain_size; i++) {
        x = bb_mul(x, x);
    }
    out[gid] = bb_sub(x, bb_one());
}

// ============================================================
// Divide by vanishing polynomial on coset:
// Given evaluations evals[i] over coset domain {g*omega^i},
// compute out[i] = evals[i] / Z_H(g*omega^i).
//
// Z_H(g*omega^i) = (g*omega^i)^n - 1 = g^n * (omega^i)^n - 1 = g^n * 1 - 1 = g^n - 1
// (since omega^n = 1). So Z_H is constant on the coset: zh_val = g^n - 1.
//
// We precompute zh_inv = 1/(g^n - 1) on CPU and just multiply.
// ============================================================

kernel void divide_by_vanishing_bn254(
    device const Fr* evals           [[buffer(0)]],
    device Fr* out                   [[buffer(1)]],
    device const Fr* zh_inv          [[buffer(2)]],   // single element: 1/(g^n - 1)
    constant uint& n                 [[buffer(3)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[gid] = fr_mul(evals[gid], zh_inv[0]);
}

kernel void divide_by_vanishing_babybear(
    device const Bb* evals           [[buffer(0)]],
    device Bb* out                   [[buffer(1)]],
    device const Bb* zh_inv          [[buffer(2)]],
    constant uint& n                 [[buffer(3)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[gid] = bb_mul(evals[gid], zh_inv[0]);
}
