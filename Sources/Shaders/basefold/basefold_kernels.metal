// Basefold polynomial commitment — GPU kernels
// Multilinear folding: new[j] = old[j] + alpha * (old[j + half] - old[j])
// This is the same as sumcheck reduce but placed here for module independence.

#include <metal_stdlib>
using namespace metal;

// Fr type and arithmetic are prepended at compile time from bn254_fr.metal

// Single-round basefold fold: reduce 2^n evaluations to 2^(n-1)
// Layout: evals[0..half) = "low" half, evals[half..n) = "high" half
// Fold: out[j] = evals[j] + alpha * (evals[j + half] - evals[j])
kernel void basefold_fold(
    device const Fr* evals          [[buffer(0)]],
    device Fr* output               [[buffer(1)]],
    constant Fr& alpha              [[buffer(2)]],
    constant uint& half_n           [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid >= half_n) return;

    Fr a = evals[gid];              // f(..., x_i=0, ...)
    Fr b = evals[gid + half_n];     // f(..., x_i=1, ...)

    // out = a + alpha * (b - a) = (1 - alpha) * a + alpha * b
    Fr diff = fr_sub(b, a);
    Fr r_diff = fr_mul(alpha, diff);
    output[gid] = fr_add(a, r_diff);
}

// Fused 2-round basefold fold: reduce 2^n to 2^(n-2) in one dispatch
// Round 1: fold with alpha0 (n -> n/2)
// Round 2: fold with alpha1 (n/2 -> n/4)
kernel void basefold_fold_fused2(
    device const Fr* evals          [[buffer(0)]],
    device Fr* output               [[buffer(1)]],
    constant Fr& alpha0             [[buffer(2)]],
    constant Fr& alpha1             [[buffer(3)]],
    constant uint& quarter_n        [[buffer(4)]],
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid >= quarter_n) return;

    uint half_n = quarter_n * 2;
    uint n = quarter_n * 4;

    // Read 4 elements: [gid], [gid + quarter], [gid + half], [gid + 3*quarter]
    Fr a = evals[gid];
    Fr b = evals[gid + quarter_n];
    Fr c = evals[gid + half_n];
    Fr d = evals[gid + half_n + quarter_n];

    // Round 1: fold pairs (a,c) and (b,d) with alpha0
    // mid0 = a + alpha0 * (c - a)
    // mid1 = b + alpha0 * (d - b)
    Fr mid0 = fr_add(a, fr_mul(alpha0, fr_sub(c, a)));
    Fr mid1 = fr_add(b, fr_mul(alpha0, fr_sub(d, b)));

    // Round 2: fold (mid0, mid1) with alpha1
    // out = mid0 + alpha1 * (mid1 - mid0)
    output[gid] = fr_add(mid0, fr_mul(alpha1, fr_sub(mid1, mid0)));
}
