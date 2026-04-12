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

// Fused 4-round basefold fold: reduce 2^n to 2^(n-4) in one dispatch
// Reads 16 elements (for n=16): pairs are (0,8),(1,9),(2,10),(3,11),(4,12),(5,13),(6,14),(7,15)
// After 4 rounds: 16 -> 8 -> 4 -> 2 -> 1 elements
kernel void basefold_fold_fused4(
    device const Fr* evals          [[buffer(0)]],
    device Fr* output               [[buffer(1)]],  // size n/16
    constant Fr& alpha0             [[buffer(2)]],
    constant Fr& alpha1             [[buffer(3)]],
    constant Fr& alpha2             [[buffer(4)]],
    constant Fr& alpha3             [[buffer(5)]],
    constant uint& sixteenth_n      [[buffer(6)]],  // n / 16
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid >= sixteenth_n) return;

    uint n = sixteenth_n * 16;
    uint half_n = n >> 1;
    uint quarter_n = n >> 2;
    uint eighth_n = n >> 3;

    // Read 16 elements: low 8 (indices 0..7) and high 8 (indices 8..15)
    Fr e0  = evals[gid];
    Fr e1  = evals[gid + eighth_n];
    Fr e2  = evals[gid + quarter_n];
    Fr e3  = evals[gid + quarter_n + eighth_n];
    Fr e4  = evals[gid + half_n];
    Fr e5  = evals[gid + half_n + eighth_n];
    Fr e6  = evals[gid + half_n + quarter_n];
    Fr e7  = evals[gid + half_n + quarter_n + eighth_n];
    Fr e8  = evals[gid + 8 * sixteenth_n];
    Fr e9  = evals[gid + 8 * sixteenth_n + eighth_n];
    Fr e10 = evals[gid + 8 * sixteenth_n + quarter_n];
    Fr e11 = evals[gid + 8 * sixteenth_n + quarter_n + eighth_n];
    Fr e12 = evals[gid + 8 * sixteenth_n + half_n];
    Fr e13 = evals[gid + 8 * sixteenth_n + half_n + eighth_n];
    Fr e14 = evals[gid + 8 * sixteenth_n + half_n + quarter_n];
    Fr e15 = evals[gid + 8 * sixteenth_n + half_n + quarter_n + eighth_n];

    // Round 1: 16 -> 8 (stride = 8)
    Fr m0  = fr_add(e0,  fr_mul(alpha0, fr_sub(e8,  e0)));
    Fr m1  = fr_add(e1,  fr_mul(alpha0, fr_sub(e9,  e1)));
    Fr m2  = fr_add(e2,  fr_mul(alpha0, fr_sub(e10, e2)));
    Fr m3  = fr_add(e3,  fr_mul(alpha0, fr_sub(e11, e3)));
    Fr m4  = fr_add(e4,  fr_mul(alpha0, fr_sub(e12, e4)));
    Fr m5  = fr_add(e5,  fr_mul(alpha0, fr_sub(e13, e5)));
    Fr m6  = fr_add(e6,  fr_mul(alpha0, fr_sub(e14, e6)));
    Fr m7  = fr_add(e7,  fr_mul(alpha0, fr_sub(e15, e7)));

    // Round 2: 8 -> 4 (stride = 4)
    Fr r0 = fr_add(m0, fr_mul(alpha1, fr_sub(m4, m0)));
    Fr r1 = fr_add(m1, fr_mul(alpha1, fr_sub(m5, m1)));
    Fr r2 = fr_add(m2, fr_mul(alpha1, fr_sub(m6, m2)));
    Fr r3 = fr_add(m3, fr_mul(alpha1, fr_sub(m7, m3)));

    // Round 3: 4 -> 2 (stride = 2)
    Fr s0 = fr_add(r0, fr_mul(alpha2, fr_sub(r2, r0)));
    Fr s1 = fr_add(r1, fr_mul(alpha2, fr_sub(r3, r1)));

    // Round 4: 2 -> 1 (stride = 1)
    output[gid] = fr_add(s0, fr_mul(alpha3, fr_sub(s1, s0)));
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

// Reed-Solomon linear extrapolation: extend evaluations by blowup factor 2.
// For each pair (f(0,...), f(1,...)) with indices (i, i+halfN):
//   extended[i]        = 2*f(1,...) - f(0,...)           i.e. f(2,...)
//   extended[i+halfN]  = 2*extended[i] - f(1,...)        i.e. f(3,...)
// Input:  evals[0..n), where n = 2*halfN
// Output: extended[0..n), the second half of the RS-encoded vector
kernel void basefold_rs_extend(
    device const Fr* evals           [[buffer(0)]],
    device Fr* extended              [[buffer(1)]],
    constant Fr& two                 [[buffer(2)]],
    constant uint& half_n            [[buffer(3)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= half_n) return;

    Fr f0 = evals[gid];                // f(0,...)
    Fr f1 = evals[gid + half_n];       // f(1,...)

    // f(2,...) = 2*f(1,...) - f(0,...)
    Fr ext0 = fr_sub(fr_mul(two, f1), f0);
    // f(3,...) = 2*f(2,...) - f(1,...)
    Fr ext1 = fr_sub(fr_mul(two, ext0), f1);

    extended[gid] = ext0;
    extended[gid + half_n] = ext1;
}
