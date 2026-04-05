// FRI query phase kernels for Metal GPU
// Supports BN254 Fr (256-bit), BabyBear (32-bit), and Mersenne-31 (32-bit) fields.
//
// Two kernel types:
// 1. fri_fold_layer_*: fold one FRI layer — for each coset pair (f(x), f(-x)),
//    compute folded = (f(x) + f(-x))/2 + alpha * (f(x) - f(-x))/(2x)
//    using precomputed inv_twiddles (omega^{-i}).
//    The /2 is absorbed (same convention as fri_kernels.metal).
//
// 2. fri_batch_query_*: gather evaluations at query positions in parallel.

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// BN254 Fr field (256-bit Montgomery)
// ============================================================================

#include "../fields/bn254_fr.metal"

// FRI fold one layer for BN254 Fr.
// For each coset pair i, computes:
//   folded[i] = (evals[i] + evals[i + half]) + alpha * inv_twiddles[i] * (evals[i] - evals[i + half])
// The /2 is absorbed into subsequent rounds or the final check.
kernel void fri_fold_layer_bn254(
    device const Fr* evals          [[buffer(0)]],
    device Fr* folded               [[buffer(1)]],
    device const Fr* inv_twiddles   [[buffer(2)]],
    constant Fr* alpha              [[buffer(3)]],
    constant uint& n                [[buffer(4)]],
    uint gid                        [[thread_position_in_grid]]
) {
    uint half_n = n >> 1;
    if (gid >= half_n) return;

    Fr a = evals[gid];
    Fr b = evals[gid + half_n];
    Fr sum = fr_add(a, b);
    Fr diff = fr_sub(a, b);

    Fr alpha_val = alpha[0];
    Fr w_inv = inv_twiddles[gid];
    Fr alpha_w = fr_mul(alpha_val, w_inv);
    Fr term = fr_mul(alpha_w, diff);

    folded[gid] = fr_add(sum, term);
}

// Batch query: gather evaluations at specified indices for BN254 Fr.
// Each thread reads one query index and writes the evaluation at that position.
kernel void fri_batch_query_bn254(
    device const Fr* evals          [[buffer(0)]],
    device const uint* indices      [[buffer(1)]],
    device Fr* results              [[buffer(2)]],
    constant uint& num_queries      [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid >= num_queries) return;
    uint idx = indices[gid];
    results[gid] = evals[idx];
}

// ============================================================================
// BabyBear field (32-bit)
// ============================================================================

// Inline BabyBear field ops to avoid duplicate symbol issues with bn254_fr
// (both files define metal_stdlib usage; we include bb directly)

constant uint FQ_BB_P = 0x78000001u;  // 2013265921
constant uint FQ_BB_MU = 2290649223u;

struct FqBb {
    uint v;
};

FqBb fqbb_add(FqBb a, FqBb b) {
    uint sum = a.v + b.v;
    return FqBb{sum >= FQ_BB_P ? sum - FQ_BB_P : sum};
}

FqBb fqbb_sub(FqBb a, FqBb b) {
    if (a.v >= b.v) return FqBb{a.v - b.v};
    return FqBb{a.v + FQ_BB_P - b.v};
}

FqBb fqbb_mul(FqBb a, FqBb b) {
    ulong prod = ulong(a.v) * ulong(b.v);
    uint prod_lo = uint(prod);
    uint prod_hi = uint(prod >> 32);
    ulong t1 = ulong(prod_lo) * ulong(FQ_BB_MU);
    ulong t2 = ulong(prod_hi) * ulong(FQ_BB_MU);
    uint q = uint((t2 + (t1 >> 32)) >> 30);
    uint r = uint(prod - ulong(q) * ulong(FQ_BB_P));
    return FqBb{r >= FQ_BB_P ? r - FQ_BB_P : r};
}

// FRI fold one layer for BabyBear.
kernel void fri_fold_layer_babybear(
    device const FqBb* evals        [[buffer(0)]],
    device FqBb* folded             [[buffer(1)]],
    device const FqBb* inv_twiddles [[buffer(2)]],
    constant FqBb* alpha            [[buffer(3)]],
    constant uint& n                [[buffer(4)]],
    uint gid                        [[thread_position_in_grid]]
) {
    uint half_n = n >> 1;
    if (gid >= half_n) return;

    FqBb a = evals[gid];
    FqBb b = evals[gid + half_n];
    FqBb sum = fqbb_add(a, b);
    FqBb diff = fqbb_sub(a, b);

    FqBb alpha_val = alpha[0];
    FqBb w_inv = inv_twiddles[gid];
    FqBb alpha_w = fqbb_mul(alpha_val, w_inv);
    FqBb term = fqbb_mul(alpha_w, diff);

    folded[gid] = fqbb_add(sum, term);
}

// Batch query for BabyBear.
kernel void fri_batch_query_babybear(
    device const FqBb* evals        [[buffer(0)]],
    device const uint* indices      [[buffer(1)]],
    device FqBb* results            [[buffer(2)]],
    constant uint& num_queries      [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid >= num_queries) return;
    uint idx = indices[gid];
    results[gid] = evals[idx];
}

// ============================================================================
// Mersenne-31 field (32-bit)
// ============================================================================

constant uint FQ_M31_P = 0x7FFFFFFFu;  // 2147483647

struct FqM31 {
    uint v;
};

FqM31 fqm31_add(FqM31 a, FqM31 b) {
    uint s = a.v + b.v;
    uint r = (s & FQ_M31_P) + (s >> 31);
    return FqM31{r == FQ_M31_P ? 0u : r};
}

FqM31 fqm31_sub(FqM31 a, FqM31 b) {
    if (a.v >= b.v) return FqM31{a.v - b.v};
    return FqM31{a.v + FQ_M31_P - b.v};
}

FqM31 fqm31_mul(FqM31 a, FqM31 b) {
    ulong prod = ulong(a.v) * ulong(b.v);
    uint lo = uint(prod & ulong(FQ_M31_P));
    uint hi = uint(prod >> 31);
    uint s = lo + hi;
    uint r = (s & FQ_M31_P) + (s >> 31);
    return FqM31{r == FQ_M31_P ? 0u : r};
}

// FRI fold one layer for Mersenne-31.
kernel void fri_fold_layer_m31(
    device const FqM31* evals       [[buffer(0)]],
    device FqM31* folded            [[buffer(1)]],
    device const FqM31* inv_twiddles [[buffer(2)]],
    constant FqM31* alpha           [[buffer(3)]],
    constant uint& n                [[buffer(4)]],
    uint gid                        [[thread_position_in_grid]]
) {
    uint half_n = n >> 1;
    if (gid >= half_n) return;

    FqM31 a = evals[gid];
    FqM31 b = evals[gid + half_n];
    FqM31 sum = fqm31_add(a, b);
    FqM31 diff = fqm31_sub(a, b);

    FqM31 alpha_val = alpha[0];
    FqM31 w_inv = inv_twiddles[gid];
    FqM31 alpha_w = fqm31_mul(alpha_val, w_inv);
    FqM31 term = fqm31_mul(alpha_w, diff);

    folded[gid] = fqm31_add(sum, term);
}

// Batch query for Mersenne-31.
kernel void fri_batch_query_m31(
    device const FqM31* evals       [[buffer(0)]],
    device const uint* indices      [[buffer(1)]],
    device FqM31* results           [[buffer(2)]],
    constant uint& num_queries      [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid >= num_queries) return;
    uint idx = indices[gid];
    results[gid] = evals[idx];
}
