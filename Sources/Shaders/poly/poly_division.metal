// Polynomial division GPU kernels
//
// Three operations:
// 1. Division by vanishing polynomial Z_H(x) = x^n - 1 in evaluation (coset) domain
// 2. Division by linear factor (X - a) via synthetic division
// 3. Batch division by multiple linear factors
//
// Supports BN254 Fr (256-bit Montgomery) and BabyBear (32-bit Barrett).

#include "../fields/bn254_fr.metal"
#include "../fields/babybear.metal"

// ============================================================
// Division by vanishing polynomial: BN254 Fr
// ============================================================
//
// Given evaluations of a constraint polynomial over a coset domain
// {g * omega^0, g * omega^1, ..., g * omega^(N-1)} where omega is
// a primitive N-th root of unity and g is the coset generator:
//
// Z_H(g * omega^i) = (g * omega^i)^n - 1
//
// For a subgroup of size n dividing N, Z_H(x) = x^n - 1.
// The quotient is: out[i] = evals[i] / Z_H(coset_point_i)
//
// Phase 1: Precompute Z_H inverses using batch inverse (Montgomery's trick).
// Phase 2: Element-wise multiply evals by Z_H inverses.
//
// This kernel handles Phase 2 (element-wise multiply).
// Z_H inverse precomputation done via batch_inverse kernel or CPU.

// Precompute Z_H values: zh_vals[i] = (g * omega^i)^n - 1
// g_powers[i] = g * omega^i (precomputed coset points)
// n = subgroup size (the exponent)
kernel void poly_zh_eval_bn254(
    device const Fr* g_powers          [[buffer(0)]],  // coset points: g*omega^i
    device Fr* zh_vals                 [[buffer(1)]],  // output: Z_H(coset_point_i)
    constant uint& domain_size         [[buffer(2)]],  // N (evaluation domain size)
    constant uint& subgroup_log        [[buffer(3)]],  // log2(n) where n = subgroup size
    uint gid                           [[thread_position_in_grid]]
) {
    if (gid >= domain_size) return;

    // Compute g_powers[gid]^n via repeated squaring where n = 2^subgroup_log
    Fr x = g_powers[gid];
    uint log_n = subgroup_log;
    for (uint i = 0; i < log_n; i++) {
        x = fr_sqr(x);
    }
    // x = (g * omega^gid)^n, now subtract 1
    zh_vals[gid] = fr_sub(x, fr_one());
}

// Element-wise division: out[i] = evals[i] * zh_inv[i]
kernel void poly_div_by_vanishing_bn254(
    device const Fr* evals             [[buffer(0)]],  // constraint poly evaluations
    device const Fr* zh_inv            [[buffer(1)]],  // precomputed Z_H inverses
    device Fr* out                     [[buffer(2)]],  // quotient evaluations
    constant uint& n                   [[buffer(3)]],  // domain size
    uint gid                           [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[gid] = fr_mul(evals[gid], zh_inv[gid]);
}

// ============================================================
// Division by vanishing polynomial: BabyBear
// ============================================================

kernel void poly_zh_eval_babybear(
    device const Bb* g_powers          [[buffer(0)]],
    device Bb* zh_vals                 [[buffer(1)]],
    constant uint& domain_size         [[buffer(2)]],
    constant uint& subgroup_log        [[buffer(3)]],
    uint gid                           [[thread_position_in_grid]]
) {
    if (gid >= domain_size) return;

    Bb x = g_powers[gid];
    uint log_n = subgroup_log;
    for (uint i = 0; i < log_n; i++) {
        x = bb_sqr(x);
    }
    zh_vals[gid] = bb_sub(x, bb_one());
}

kernel void poly_div_by_vanishing_babybear(
    device const Bb* evals             [[buffer(0)]],
    device const Bb* zh_inv            [[buffer(1)]],
    device Bb* out                     [[buffer(2)]],
    constant uint& n                   [[buffer(3)]],
    uint gid                           [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[gid] = bb_mul(evals[gid], zh_inv[gid]);
}

// ============================================================
// Division by linear factor (X - root): BN254 Fr
// ============================================================
//
// Synthetic division of polynomial f(X) by (X - root) in coefficient form.
// f(X) = (X - root) * q(X) + remainder
//
// Sequential per polynomial, but we run M polynomials in parallel.
// Each threadgroup handles one polynomial.
// coeffs[poly_idx * stride + i] for i in [0, degree)
// Output: quotient of degree-1 coefficients + 1 remainder element.

kernel void poly_div_by_linear_bn254(
    device const Fr* coeffs            [[buffer(0)]],  // M polynomials packed, each of length `degree`
    device Fr* quotients               [[buffer(1)]],  // M quotients packed, each of length `degree - 1`
    device Fr* remainders              [[buffer(2)]],  // M remainders
    device const Fr* roots             [[buffer(3)]],  // M root values
    constant uint& degree              [[buffer(4)]],  // degree (number of coefficients)
    constant uint& num_polys           [[buffer(5)]],
    uint tgid                          [[threadgroup_position_in_grid]],
    uint tid                           [[thread_index_in_threadgroup]]
) {
    if (tid != 0 || tgid >= num_polys) return;

    uint poly_idx = tgid;
    Fr root = roots[poly_idx];
    uint in_base = poly_idx * degree;
    uint out_base = poly_idx * (degree - 1);

    // Synthetic division: process from highest degree down
    // q[n-2] = c[n-1]
    // q[k] = c[k+1] + root * q[k+1]
    Fr carry = coeffs[in_base + degree - 1];
    quotients[out_base + degree - 2] = carry;

    for (uint k = degree - 2; k > 0; k--) {
        carry = fr_add(coeffs[in_base + k], fr_mul(root, carry));
        quotients[out_base + k - 1] = carry;
    }

    // Remainder = c[0] + root * q[0]
    remainders[poly_idx] = fr_add(coeffs[in_base], fr_mul(root, carry));
}

// ============================================================
// Division by linear factor (X - root): BabyBear
// ============================================================

kernel void poly_div_by_linear_babybear(
    device const Bb* coeffs            [[buffer(0)]],
    device Bb* quotients               [[buffer(1)]],
    device Bb* remainders              [[buffer(2)]],
    device const Bb* roots             [[buffer(3)]],
    constant uint& degree              [[buffer(4)]],
    constant uint& num_polys           [[buffer(5)]],
    uint tgid                          [[threadgroup_position_in_grid]],
    uint tid                           [[thread_index_in_threadgroup]]
) {
    if (tid != 0 || tgid >= num_polys) return;

    uint poly_idx = tgid;
    Bb root = roots[poly_idx];
    uint in_base = poly_idx * degree;
    uint out_base = poly_idx * (degree - 1);

    Bb carry = coeffs[in_base + degree - 1];
    quotients[out_base + degree - 2] = carry;

    for (uint k = degree - 2; k > 0; k--) {
        carry = bb_add(coeffs[in_base + k], bb_mul(root, carry));
        quotients[out_base + k - 1] = carry;
    }

    remainders[poly_idx] = bb_add(coeffs[in_base], bb_mul(root, carry));
}

// ============================================================
// Batch division by multiple linear factors: BN254 Fr
// ============================================================
//
// Divide one polynomial f(X) by multiple roots [r_0, r_1, ..., r_{M-1}].
// For each root r_j: f(X) = (X - r_j) * q_j(X) + rem_j
// Each threadgroup handles one root (parallel across roots).

kernel void poly_batch_div_bn254(
    device const Fr* coeffs            [[buffer(0)]],  // single polynomial, length `degree`
    device Fr* quotients               [[buffer(1)]],  // M quotients, each of length `degree - 1`
    device Fr* remainders              [[buffer(2)]],  // M remainders
    device const Fr* roots             [[buffer(3)]],  // M roots
    constant uint& degree              [[buffer(4)]],
    constant uint& num_roots           [[buffer(5)]],
    uint tgid                          [[threadgroup_position_in_grid]],
    uint tid                           [[thread_index_in_threadgroup]]
) {
    if (tid != 0 || tgid >= num_roots) return;

    uint root_idx = tgid;
    Fr root = roots[root_idx];
    uint out_base = root_idx * (degree - 1);

    Fr carry = coeffs[degree - 1];
    quotients[out_base + degree - 2] = carry;

    for (uint k = degree - 2; k > 0; k--) {
        carry = fr_add(coeffs[k], fr_mul(root, carry));
        quotients[out_base + k - 1] = carry;
    }

    remainders[root_idx] = fr_add(coeffs[0], fr_mul(root, carry));
}

// ============================================================
// Batch division by multiple linear factors: BabyBear
// ============================================================

kernel void poly_batch_div_babybear(
    device const Bb* coeffs            [[buffer(0)]],
    device Bb* quotients               [[buffer(1)]],
    device Bb* remainders              [[buffer(2)]],
    device const Bb* roots             [[buffer(3)]],
    constant uint& degree              [[buffer(4)]],
    constant uint& num_roots           [[buffer(5)]],
    uint tgid                          [[threadgroup_position_in_grid]],
    uint tid                           [[thread_index_in_threadgroup]]
) {
    if (tid != 0 || tgid >= num_roots) return;

    uint root_idx = tgid;
    Bb root = roots[root_idx];
    uint out_base = root_idx * (degree - 1);

    Bb carry = coeffs[degree - 1];
    quotients[out_base + degree - 2] = carry;

    for (uint k = degree - 2; k > 0; k--) {
        carry = bb_add(coeffs[k], bb_mul(root, carry));
        quotients[out_base + k - 1] = carry;
    }

    remainders[root_idx] = bb_add(coeffs[0], bb_mul(root, carry));
}

// ============================================================
// Batch inverse via Montgomery's trick (for Z_H inverse precomputation)
// ============================================================
// Used internally by the vanishing polynomial division pipeline.
// Each threadgroup inverts a chunk of elements using 1 Fermat inverse.

#define VH_BATCH_INV_CHUNK_FR 256

kernel void poly_div_batch_inverse_bn254(
    device const Fr* a                 [[buffer(0)]],
    device Fr* out                     [[buffer(1)]],
    constant uint& n                   [[buffer(2)]],
    uint tid                           [[thread_index_in_threadgroup]],
    uint tgid                          [[threadgroup_position_in_grid]]
) {
    if (tid != 0) return;

    uint base = tgid * VH_BATCH_INV_CHUNK_FR;
    uint chunk = min(uint(VH_BATCH_INV_CHUNK_FR), n - base);
    if (chunk == 0) return;

    // Phase 1: Build prefix products
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

#define VH_BATCH_INV_CHUNK_BB 1024

kernel void poly_div_batch_inverse_babybear(
    device const Bb* a                 [[buffer(0)]],
    device Bb* out                     [[buffer(1)]],
    constant uint& n                   [[buffer(2)]],
    uint tid                           [[thread_index_in_threadgroup]],
    uint tgid                          [[threadgroup_position_in_grid]]
) {
    if (tid != 0) return;

    uint base = tgid * VH_BATCH_INV_CHUNK_BB;
    uint chunk = min(uint(VH_BATCH_INV_CHUNK_BB), n - base);
    if (chunk == 0) return;

    out[base] = a[base];
    for (uint i = 1; i < chunk; i++) {
        out[base + i] = bb_mul(out[base + i - 1], a[base + i]);
    }

    Bb inv = bb_inv(out[base + chunk - 1]);

    for (uint i = chunk - 1; i > 0; i--) {
        Bb ai = a[base + i];
        out[base + i] = bb_mul(inv, out[base + i - 1]);
        inv = bb_mul(inv, ai);
    }
    out[base] = inv;
}
