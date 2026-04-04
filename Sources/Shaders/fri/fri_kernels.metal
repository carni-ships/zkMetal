// FRI (Fast Reed-Solomon IOP) kernels for Metal GPU
// Supports BN254 Fr field for compatibility with existing NTT infrastructure
// Core operation: FRI fold — reduce polynomial degree by 2x each round

#include "../fields/bn254_fr.metal"

// FRI fold: given evals[0..n-1] on domain omega^{0..n-1},
// compute folded evals of half the size using random challenge beta.
// folded[i] = (evals[i] + evals[i + n/2]) / 2 + beta * (evals[i] - evals[i + n/2]) / (2 * omega^i)
// Equivalently: folded[i] = (1 + beta/omega^i) * evals[i] / 2 + (1 - beta/omega^i) * evals[i + n/2] / 2
//
// Simplified form using twiddles:
// folded[i] = (evals[i] + evals[i + half]) + beta * inv_twiddle[i] * (evals[i] - evals[i + half])
// (where the /2 is absorbed into the next round or final check)
kernel void fri_fold(
    device const Fr* evals          [[buffer(0)]],
    device Fr* folded               [[buffer(1)]],
    device const Fr* inv_twiddles   [[buffer(2)]],  // omega^{-i} for i in [0, n/2)
    constant Fr* beta               [[buffer(3)]],   // random challenge (constant broadcast)
    constant uint& n                [[buffer(4)]],   // current domain size (must be even)
    uint gid                        [[thread_position_in_grid]]
) {
    uint half_n = n >> 1;
    if (gid >= half_n) return;

    Fr a = evals[gid];
    Fr b = evals[gid + half_n];
    Fr sum = fr_add(a, b);
    Fr diff = fr_sub(a, b);

    Fr beta_val = beta[0];
    Fr w_inv = inv_twiddles[gid];
    Fr beta_w = fr_mul(beta_val, w_inv);
    Fr term = fr_mul(beta_w, diff);

    folded[gid] = fr_add(sum, term);
}

// Fused 2-round FRI fold: applies two consecutive fold rounds in one kernel.
// Reads 4 elements, computes both folds in registers, writes 1 element.
// Eliminates intermediate N/2 buffer and halves dispatch count.
// Twiddle reuse: round 2 uses inv_twiddles[2*j] since omega_{N/2}^{-j} = omega_N^{-2j}.
kernel void fri_fold_fused2(
    device const Fr* evals          [[buffer(0)]],  // size n
    device Fr* folded               [[buffer(1)]],  // size n/4
    device const Fr* inv_twiddles   [[buffer(2)]],  // omega_n^{-i} for i in [0, n)
    constant Fr* beta0              [[buffer(3)]],  // challenge for first fold (constant broadcast)
    constant Fr* beta1              [[buffer(4)]],  // challenge for second fold (constant broadcast)
    constant uint& n                [[buffer(5)]],  // current domain size (must be >= 4)
    uint gid                        [[thread_position_in_grid]]
) {
    uint quarter = n >> 2;
    if (gid >= quarter) return;

    uint half_n = n >> 1;

    // Read 4 elements needed for both fold rounds
    Fr a0 = evals[gid];
    Fr a1 = evals[gid + quarter];
    Fr a2 = evals[gid + half_n];
    Fr a3 = evals[gid + half_n + quarter];

    Fr b0 = beta0[0];

    // Round 1 (size n → n/2):
    // f1[gid]         = (a0 + a2) + beta0 * tw[gid] * (a0 - a2)
    // f1[gid+quarter] = (a1 + a3) + beta0 * tw[gid+quarter] * (a1 - a3)
    Fr tw_lo = inv_twiddles[gid];
    Fr tw_hi = inv_twiddles[gid + quarter];

    Fr sum02 = fr_add(a0, a2);
    Fr diff02 = fr_sub(a0, a2);
    Fr f1_lo = fr_add(sum02, fr_mul(fr_mul(b0, tw_lo), diff02));

    Fr sum13 = fr_add(a1, a3);
    Fr diff13 = fr_sub(a1, a3);
    Fr f1_hi = fr_add(sum13, fr_mul(fr_mul(b0, tw_hi), diff13));

    // Round 2 (size n/2 → n/4):
    // folded[gid] = (f1_lo + f1_hi) + beta1 * tw2[gid] * (f1_lo - f1_hi)
    // tw2[gid] = omega_{n/2}^{-gid} = omega_n^{-2*gid} = inv_twiddles[2*gid]
    Fr b1 = beta1[0];
    Fr tw2 = inv_twiddles[2 * gid];

    Fr sum_f = fr_add(f1_lo, f1_hi);
    Fr diff_f = fr_sub(f1_lo, f1_hi);
    folded[gid] = fr_add(sum_f, fr_mul(fr_mul(b1, tw2), diff_f));
}

// Fused 4-round FRI fold: applies four consecutive fold rounds in one kernel.
// Reads 16 elements, computes all 4 folds in registers, writes 1 element.
// Reduces dispatch count by 4x compared to single-fold.
// Twiddle relation: omega_{N/2^k}^{-j} = omega_N^{-j*2^k}
kernel void fri_fold_fused4(
    device const Fr* evals          [[buffer(0)]],  // size n
    device Fr* folded               [[buffer(1)]],  // size n/16
    device const Fr* inv_twiddles   [[buffer(2)]],  // omega_n^{-i} for i in [0, n)
    constant Fr* betas              [[buffer(3)]],  // 4 challenges (contiguous)
    constant uint& n                [[buffer(4)]],  // current domain size (must be >= 16)
    uint gid                        [[thread_position_in_grid]]
) {
    uint sixteenth = n >> 4;
    if (gid >= sixteenth) return;

    // Load 16 input elements with stride n/16
    Fr d[16];
    for (uint k = 0; k < 16; k++) {
        d[k] = evals[gid + k * sixteenth];
    }

    // Round 1 (n → n/2): fold 8 pairs at stride 8*sixteenth = n/2
    // tw[i] = inv_twiddles[gid + i*sixteenth] for i = 0..7
    Fr b0 = betas[0];
    for (uint k = 0; k < 8; k++) {
        Fr a = d[k];
        Fr b = d[k + 8];
        Fr tw = inv_twiddles[gid + k * sixteenth];
        Fr s = fr_add(a, b);
        Fr df = fr_sub(a, b);
        d[k] = fr_add(s, fr_mul(fr_mul(b0, tw), df));
    }

    // Round 2 (n/2 → n/4): fold 4 pairs
    // fold2 position j uses omega_{n/2}^{-j} = omega_n^{-2j} = inv_twiddles[2j]
    // After round1, d[k] represents fold1 at position gid + k*(n/16)
    // twiddle for pair k: inv_twiddles[2*(gid + k*sixteenth)]
    Fr b1 = betas[1];
    for (uint k = 0; k < 4; k++) {
        Fr a = d[k];
        Fr b = d[k + 4];
        Fr tw = inv_twiddles[2 * gid + k * sixteenth * 2];
        Fr s = fr_add(a, b);
        Fr df = fr_sub(a, b);
        d[k] = fr_add(s, fr_mul(fr_mul(b1, tw), df));
    }

    // Round 3 (n/4 → n/8): fold 2 pairs
    // fold3 position j uses omega_{n/4}^{-j} = omega_n^{-4j} = inv_twiddles[4j]
    // After round2, d[k] represents fold2 at position gid + k*(n/16)
    // twiddle for pair k: inv_twiddles[4*(gid + k*sixteenth)]
    Fr b2 = betas[2];
    for (uint k = 0; k < 2; k++) {
        Fr a = d[k];
        Fr b = d[k + 2];
        Fr tw = inv_twiddles[4 * gid + k * sixteenth * 4];
        Fr s = fr_add(a, b);
        Fr df = fr_sub(a, b);
        d[k] = fr_add(s, fr_mul(fr_mul(b2, tw), df));
    }

    // Round 4 (n/8 → n/16): fold 1 pair
    // tw = omega_n^{-8j} = inv_twiddles[8 * gid]
    Fr b3 = betas[3];
    {
        Fr a = d[0];
        Fr b = d[1];
        Fr tw = inv_twiddles[8 * gid];
        Fr s = fr_add(a, b);
        Fr df = fr_sub(a, b);
        folded[gid] = fr_add(s, fr_mul(fr_mul(b3, tw), df));
    }
}

// FRI fold-by-4: reduce polynomial degree by 4x in one round.
// Uses 4th-root-of-unity decomposition:
//   f(x) = f0(x^4) + x*f1(x^4) + x^2*f2(x^4) + x^3*f3(x^4)
//   f_folded(y) = f0(y) + r*f1(y) + r^2*f2(y) + r^3*f3(y)
//
// Thread i reads evals at indices i, i+N/4, i+N/2, i+3N/4
// and produces one output using challenge r and inverse twiddle factors.
//
// The recovery formulas from evaluations at x, w*x, w^2*x, w^3*x:
//   f0(x^4) = (e0 + e1 + e2 + e3) / 4
//   f1(x^4) = (e0 - e1 + e2 - e3) * inv_x / 4    [using w^2 = -1 for quartic root]
//   ... but this requires working with 4th roots in the evaluation domain.
//
// For our domain layout (omega^i for i in [0,N)), the 4 coset elements for index i are:
//   evals[i], evals[i + N/4], evals[i + N/2], evals[i + 3N/4]
// which correspond to omega^i, omega^{i+N/4}, omega^{i+N/2}, omega^{i+3N/4}
// The stride N/4 means omega^{N/4} is a primitive 4th root of unity.
//
// Using inv_twiddles (omega^{-i}):
//   inv_x = inv_twiddles[i]
//   inv_x2 = inv_twiddles[2*i]  (or fr_mul(inv_x, inv_x))
//   inv_x3 = inv_twiddles[3*i]  (or fr_mul(inv_x2, inv_x))
kernel void fri_fold_by4(
    device const Fr* evals          [[buffer(0)]],
    device Fr* folded               [[buffer(1)]],
    device const Fr* inv_twiddles   [[buffer(2)]],  // omega_N^{-i} for i in [0, N)
    constant Fr* challenge          [[buffer(3)]],   // random challenge r
    constant uint& n                [[buffer(4)]],   // current domain size (must be divisible by 4)
    uint gid                        [[thread_position_in_grid]]
) {
    uint quarter = n >> 2;
    if (gid >= quarter) return;

    // Read 4 evaluations at stride N/4
    Fr e0 = evals[gid];
    Fr e1 = evals[gid + quarter];
    Fr e2 = evals[gid + 2 * quarter];
    Fr e3 = evals[gid + 3 * quarter];

    // Compute sums/differences (butterfly-like)
    // s02 = e0 + e2,  d02 = e0 - e2
    // s13 = e1 + e3,  d13 = e1 - e3
    Fr s02 = fr_add(e0, e2);
    Fr d02 = fr_sub(e0, e2);
    Fr s13 = fr_add(e1, e3);
    Fr d13 = fr_sub(e1, e3);

    // f0_coeff = s02 + s13 = e0 + e1 + e2 + e3  (sum of all)
    // f2_coeff = s02 - s13 = e0 - e1 + e2 - e3  (alternating sum)
    // f1_coeff = d02 + d13 = e0 + e1 - e2 - e3  ... wait
    // Actually for our domain layout:
    // omega^{N/4} is a primitive 4th root, call it w4.
    // evals at omega^i, omega^{i+N/4}, omega^{i+N/2}, omega^{i+3N/4}
    // = f(x), f(w4*x), f(w4^2*x), f(w4^3*x) where x = omega^i, w4^2 = omega^{N/2} = -1
    //
    // So: f(x) = f0(x^4) + x*f1(x^4) + x^2*f2(x^4) + x^3*f3(x^4)
    // f(w4*x) = f0 + w4*x*f1 + w4^2*x^2*f2 + w4^3*x^3*f3
    //         = f0 + w4*x*f1 - x^2*f2 - w4*x^3*f3
    // f(w4^2*x) = f0 - x*f1 + x^2*f2 - x^3*f3
    // f(w4^3*x) = f0 - w4*x*f1 - x^2*f2 + w4*x^3*f3
    //
    // Summing:
    // e0 + e2 = 2*f0 + 2*x^2*f2  =>  s02 = 2*(f0 + x^2*f2)
    // e0 - e2 = 2*x*f1 + 2*x^3*f3  =>  d02 = 2*x*(f1 + x^2*f3)
    // e1 + e3 = 2*f0 - 2*x^2*f2  =>  s13 = 2*(f0 - x^2*f2)
    // e1 - e3 = 2*w4*x*f1 - 2*w4*x^3*f3 = 2*w4*x*(f1 - x^2*f3)
    //         =>  d13 = 2*w4*x*(f1 - x^2*f3)
    //
    // Therefore:
    // f0 = (s02 + s13) / 4
    // x^2*f2 = (s02 - s13) / 4  =>  f2 = (s02 - s13) / (4*x^2)
    // x*(f1 + x^2*f3) = d02/2
    // w4*x*(f1 - x^2*f3) = d13/2
    // => x*f1 = (d02/2 + d13/(2*w4)) / 2 = (d02 + d13*w4_inv) / 4
    //    f1 = (d02 + d13*w4_inv) / (4*x)
    // x^3*f3 = (d02/2 - d13/(2*w4)) / 2 = (d02 - d13*w4_inv) / 4
    //    f3 = (d02 - d13*w4_inv) / (4*x^3)
    //
    // folded = f0 + r*f1 + r^2*f2 + r^3*f3
    //
    // To avoid computing 1/4 explicitly (it gets absorbed or we can multiply at the end),
    // we compute 4*folded and then multiply by inv4 at the end:
    // 4*folded = (s02 + s13)
    //          + r * (d02 + d13*w4_inv) * inv_x
    //          + r^2 * (s02 - s13) * inv_x2
    //          + r^3 * (d02 - d13*w4_inv) * inv_x3

    Fr r_val = challenge[0];

    // Compute w4_inv: w4 = omega^{N/4}, so w4_inv = omega^{-N/4} = inv_twiddles[N/4]
    // But N/4 = quarter, and inv_twiddles has size N, so inv_twiddles[quarter] is valid.
    Fr w4_inv = inv_twiddles[quarter];

    // d13_w4inv = d13 * w4_inv
    Fr d13_w4inv = fr_mul(d13, w4_inv);

    // The 4 "numerator" terms (before dividing by x powers):
    Fr t0 = fr_add(s02, s13);                // 4*f0
    Fr t1 = fr_add(d02, d13_w4inv);          // 4*x*f1
    Fr t2 = fr_sub(s02, s13);                // 4*x^2*f2
    Fr t3 = fr_sub(d02, d13_w4inv);          // 4*x^3*f3

    // Multiply by inverse twiddles to remove x powers
    Fr inv_x = inv_twiddles[gid];
    Fr inv_x2 = fr_mul(inv_x, inv_x);
    Fr inv_x3 = fr_mul(inv_x2, inv_x);

    t1 = fr_mul(t1, inv_x);
    t2 = fr_mul(t2, inv_x2);
    t3 = fr_mul(t3, inv_x3);

    // Combine with challenge: folded*4 = t0 + r*t1 + r^2*t2 + r^3*t3
    // Use Horner: ((r*t3 + t2)*r + t1)*r + t0
    Fr r2 = fr_mul(r_val, r_val);
    Fr r3 = fr_mul(r2, r_val);

    Fr result = fr_add(t0, fr_mul(r_val, t1));
    result = fr_add(result, fr_mul(r2, t2));
    result = fr_add(result, fr_mul(r3, t3));

    // Multiply by inv4 to get the actual folded value
    // inv4 = inverse of 4 in Fr = (r+1)/4 mod r
    // For BN254 Fr: 4^{-1} mod r
    // r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
    // inv4 = (r+1)/4 = 5472060717959818805561601436314318772137091100104008585924551046643952123905
    // In 8x32-bit limbs (little-endian Montgomery form):
    // We compute it as fr_mul(result, INV4) where INV4 is a constant.
    // INV4 in standard form: 0x0c19139c_b84c680a_6e14116d_a0601950_0a0d9a5e_ee5c5386_50f5a7b6_c4000001
    // But we need Montgomery form. Easiest: use fr_inv on the Montgomery form of 4.
    // Actually: in the fold-by-2 kernel, there's no /2 either — it's absorbed.
    // We follow the same convention: absorb the /4 into subsequent rounds or final check.
    // The fold-by-2 comment says: "where the /2 is absorbed into the next round or final check"
    // So we do the same: skip the /4 division.

    folded[gid] = result;
}

// Batch FRI query: given a set of query positions, extract evaluations
// from the polynomial at those positions (and their paired positions)
kernel void fri_query_extract(
    device const Fr* evals          [[buffer(0)]],
    device const uint* query_indices [[buffer(1)]],
    device Fr* query_evals          [[buffer(2)]],  // output: pairs (eval[i], eval[i + half])
    constant uint& n                [[buffer(3)]],
    constant uint& num_queries      [[buffer(4)]],
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid >= num_queries) return;

    uint half_n = n >> 1;
    uint idx = query_indices[gid];
    uint paired_idx = (idx < half_n) ? idx + half_n : idx - half_n;

    query_evals[gid * 2] = evals[idx];
    query_evals[gid * 2 + 1] = evals[paired_idx];
}

// Coset LDE (Low-Degree Extension): shift evaluations from domain to coset
// coset_evals[i] = evals[i] * shift^i
// Used to evaluate on a coset of the original domain for FRI
kernel void fri_coset_shift(
    device Fr* evals                [[buffer(0)]],
    device const Fr* shift_powers   [[buffer(1)]],  // shift^0, shift^1, ..., shift^{n-1}
    constant uint& n                [[buffer(2)]],
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    evals[gid] = fr_mul(evals[gid], shift_powers[gid]);
}

// Coset unshift: inverse of coset_shift
kernel void fri_coset_unshift(
    device Fr* evals                [[buffer(0)]],
    device const Fr* inv_shift_powers [[buffer(1)]],
    constant uint& n                [[buffer(2)]],
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    evals[gid] = fr_mul(evals[gid], inv_shift_powers[gid]);
}
