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
