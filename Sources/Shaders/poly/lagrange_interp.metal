// GPU-accelerated Lagrange interpolation kernels for BN254 Fr
//
// Given n evaluation points {(x_i, y_i)}, recover polynomial coefficients.
//
// Strategy:
//   1. Compute barycentric weights w_i = 1 / prod_{j != i}(x_i - x_j)  [GPU parallel]
//   2. Compute scaled weights: s_i = y_i * w_i                          [GPU parallel]
//   3. Build coefficients via accumulating Lagrange basis polynomials     [CPU, O(n^2)]
//
// The GPU kernels accelerate the O(n^2) weight computation which is the bottleneck
// for moderate n (the basis accumulation is also O(n^2) but has serial dependencies).
//
// For large n on roots-of-unity domains, use NTT-based interpolation instead.

#include "../fields/bn254_fr.metal"

// ============================================================
// Kernel 1: Compute denominator products for barycentric weights
// Each thread i computes: denom[i] = prod_{j != i} (x_i - x_j)
// ============================================================

kernel void lagrange_denom_bn254(
    device const Fr* points        [[buffer(0)]],   // x_0, x_1, ..., x_{n-1}
    device Fr* denoms              [[buffer(1)]],   // output: denom[i]
    constant uint& n               [[buffer(2)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    Fr xi = points[gid];
    Fr prod = fr_one();

    for (uint j = 0; j < n; j++) {
        if (j == gid) continue;
        Fr diff = fr_sub(xi, points[j]);
        prod = fr_mul(prod, diff);
    }

    denoms[gid] = prod;
}

// ============================================================
// Kernel 2: Compute scaled barycentric weights
// scaled[i] = y_i * inv_denom[i]
// where inv_denom has already been batch-inverted on CPU
// ============================================================

kernel void lagrange_scale_bn254(
    device const Fr* values        [[buffer(0)]],   // y_0, y_1, ..., y_{n-1}
    device const Fr* inv_denoms    [[buffer(1)]],   // 1/denom[i] (batch-inverted)
    device Fr* scaled              [[buffer(2)]],   // output: y_i / denom_i
    constant uint& n               [[buffer(3)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    scaled[gid] = fr_mul(values[gid], inv_denoms[gid]);
}

// ============================================================
// Kernel 3: Evaluate interpolated polynomial at a single point z
// Uses barycentric formula:
//   p(z) = [sum_i s_i / (z - x_i)] / [sum_i w_i / (z - x_i)]
// where s_i = y_i * w_i
//
// Each thread handles one term; results are reduced on CPU.
// ============================================================

kernel void lagrange_eval_bn254(
    device const Fr* points        [[buffer(0)]],   // x_i
    device const Fr* weights       [[buffer(1)]],   // barycentric weights w_i
    device const Fr* values        [[buffer(2)]],   // y_i
    device Fr* numer_terms         [[buffer(3)]],   // output: y_i * w_i / (z - x_i)
    device Fr* denom_terms         [[buffer(4)]],   // output: w_i / (z - x_i)
    constant Fr& z                 [[buffer(5)]],   // evaluation point
    constant uint& n               [[buffer(6)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    Fr diff = fr_sub(z, points[gid]);
    Fr inv_diff = fr_inv(diff);
    Fr w_over_diff = fr_mul(weights[gid], inv_diff);

    numer_terms[gid] = fr_mul(values[gid], w_over_diff);
    denom_terms[gid] = w_over_diff;
}
