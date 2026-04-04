// Circle FRI fold kernels for Mersenne31 field
// Unlike standard FRI over multiplicative subgroups, Circle FRI folds over the circle group:
//   - First fold: y-coordinate (twin-coset decomposition) — pairs (x,y) and (x,-y)
//   - Subsequent folds: x-coordinate squaring map — x -> 2x^2 - 1
//
// Fold formula:
//   g[i] = (f[i] + f[i+half])/2 + alpha * (f[i] - f[i+half]) / (2 * twiddle[i])
// where twiddle = y (first fold) or x (subsequent folds).
// The /2 is absorbed into the twiddle precomputation: we store 1/(2*twiddle).

#include "../fields/mersenne31.metal"

// Precomputed inverse of 2 mod p = (p+1)/2
constant uint M31_INV2 = 1073741824u;  // (2^31 - 1 + 1) / 2 = 2^30

// Circle FRI first fold: uses y-coordinate twiddles (twin-coset decomposition).
// Pairs f[i] with f[i + half] where these correspond to (x, y) and (x, -y).
// inv_2y[i] = 1/(2*y_i) precomputed on CPU.
//
// folded[i] = (f[i] + f[i + half]) / 2 + alpha * (f[i] - f[i + half]) * inv_2y[i]
kernel void circle_fri_fold_first(
    device const M31* evals         [[buffer(0)]],
    device M31* folded              [[buffer(1)]],
    device const M31* inv_2y        [[buffer(2)]],  // 1/(2*y_i) for i in [0, n/2)
    constant M31* alpha             [[buffer(3)]],   // random challenge
    constant uint& n                [[buffer(4)]],   // current domain size (must be even)
    uint gid                        [[thread_position_in_grid]]
) {
    uint half_n = n >> 1;
    if (gid >= half_n) return;

    M31 a = evals[gid];
    M31 b = evals[gid + half_n];

    // sum = (a + b) / 2
    M31 sum_raw = m31_add(a, b);
    M31 half_sum = m31_mul(sum_raw, M31{M31_INV2});

    // diff_term = alpha * (a - b) * inv_2y[i]
    M31 diff = m31_sub(a, b);
    M31 alpha_val = alpha[0];
    M31 alpha_diff = m31_mul(alpha_val, diff);
    M31 diff_term = m31_mul(alpha_diff, inv_2y[gid]);

    folded[gid] = m31_add(half_sum, diff_term);
}

// Circle FRI subsequent fold: uses x-coordinate twiddles.
// After the first fold projects from the circle to the x-axis,
// subsequent folds use the squaring map x -> 2x^2 - 1.
// Pairs f[i] with f[i + half] where these share the same x after squaring.
// inv_2x[i] = 1/(2*x_i) precomputed on CPU.
//
// folded[i] = (f[i] + f[i + half]) / 2 + alpha * (f[i] - f[i + half]) * inv_2x[i]
kernel void circle_fri_fold(
    device const M31* evals         [[buffer(0)]],
    device M31* folded              [[buffer(1)]],
    device const M31* inv_2x        [[buffer(2)]],  // 1/(2*x_i) for i in [0, n/2)
    constant M31* alpha             [[buffer(3)]],   // random challenge
    constant uint& n                [[buffer(4)]],   // current domain size (must be even)
    uint gid                        [[thread_position_in_grid]]
) {
    uint half_n = n >> 1;
    if (gid >= half_n) return;

    M31 a = evals[gid];
    M31 b = evals[gid + half_n];

    // sum = (a + b) / 2
    M31 sum_raw = m31_add(a, b);
    M31 half_sum = m31_mul(sum_raw, M31{M31_INV2});

    // diff_term = alpha * (a - b) * inv_2x[i]
    M31 diff = m31_sub(a, b);
    M31 alpha_val = alpha[0];
    M31 alpha_diff = m31_mul(alpha_val, diff);
    M31 diff_term = m31_mul(alpha_diff, inv_2x[gid]);

    folded[gid] = m31_add(half_sum, diff_term);
}

// Fused 2-round Circle FRI fold: applies first fold (y-twiddle) and one x-fold
// in a single kernel. Reads 4 elements, writes 1. Eliminates intermediate buffer.
//
// Round 1 (y-fold, size n -> n/2):
//   f1[i]        = (e[i] + e[i+n/2])/2 + alpha0 * (e[i] - e[i+n/2]) * inv_2y[i]
//   f1[i+n/4]    = (e[i+n/4] + e[i+3n/4])/2 + alpha0 * (e[i+n/4] - e[i+3n/4]) * inv_2y[i+n/4]
// Round 2 (x-fold, size n/2 -> n/4):
//   folded[i]    = (f1[i] + f1[i+n/4])/2 + alpha1 * (f1[i] - f1[i+n/4]) * inv_2x_r2[i]
kernel void circle_fri_fold_fused2(
    device const M31* evals         [[buffer(0)]],
    device M31* folded              [[buffer(1)]],
    device const M31* inv_2y        [[buffer(2)]],  // 1/(2*y_i) for round 1
    device const M31* inv_2x        [[buffer(3)]],  // 1/(2*x_i) for round 2 (after y-fold domain)
    constant M31* alpha0            [[buffer(4)]],   // challenge for y-fold
    constant M31* alpha1            [[buffer(5)]],   // challenge for x-fold
    constant uint& n                [[buffer(6)]],   // original domain size
    uint gid                        [[thread_position_in_grid]]
) {
    uint quarter = n >> 2;
    if (gid >= quarter) return;

    uint half_n = n >> 1;

    // Read 4 elements
    M31 e0 = evals[gid];
    M31 e1 = evals[gid + quarter];
    M31 e2 = evals[gid + half_n];
    M31 e3 = evals[gid + half_n + quarter];

    M31 a0 = alpha0[0];
    M31 inv2 = M31{M31_INV2};

    // Round 1: y-fold
    // f1_lo = (e0 + e2)/2 + alpha0 * (e0 - e2) * inv_2y[gid]
    M31 sum02 = m31_mul(m31_add(e0, e2), inv2);
    M31 diff02 = m31_sub(e0, e2);
    M31 f1_lo = m31_add(sum02, m31_mul(m31_mul(a0, diff02), inv_2y[gid]));

    // f1_hi = (e1 + e3)/2 + alpha0 * (e1 - e3) * inv_2y[gid + quarter]
    M31 sum13 = m31_mul(m31_add(e1, e3), inv2);
    M31 diff13 = m31_sub(e1, e3);
    M31 f1_hi = m31_add(sum13, m31_mul(m31_mul(a0, diff13), inv_2y[gid + quarter]));

    // Round 2: x-fold
    M31 a1 = alpha1[0];
    M31 sum_r2 = m31_mul(m31_add(f1_lo, f1_hi), inv2);
    M31 diff_r2 = m31_sub(f1_lo, f1_hi);
    folded[gid] = m31_add(sum_r2, m31_mul(m31_mul(a1, diff_r2), inv_2x[gid]));
}
