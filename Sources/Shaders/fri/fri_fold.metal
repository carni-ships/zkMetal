// FRI fold kernel — dedicated GPU-accelerated FRI folding for STARK provers
//
// Given evaluations on a domain of size n, folds them using a random challenge
// to produce evaluations on a half-sized domain. This is the core operation
// in the FRI commit phase.
//
// Formula: result[i] = (evals[i] + evals[i + n/2]) + challenge * (evals[i] - evals[i + n/2]) * domain_inv[i]
//
// where domain_inv[i] = 1 / domain[i] are precomputed domain element inverses.
// This formulation is equivalent to the twiddle-based fold but works with
// explicit domain representations (cosets, shifted domains, etc.) common in
// STARK provers like Plonky3 and Stwo.
//
// Supports BN254 Fr (256-bit Montgomery form).

#include "../fields/bn254_fr.metal"

// Single FRI fold round: n elements -> n/2 elements
// For each i in [0, n/2):
//   result[i] = (evals[i] + evals[i + n/2]) + challenge * (evals[i] - evals[i + n/2]) * domain_inv[i]
kernel void fri_fold_kernel(
    device const Fr* evals          [[buffer(0)]],
    device Fr* result               [[buffer(1)]],
    device const Fr* domain_inv     [[buffer(2)]],  // precomputed 1/domain[i] for i in [0, n/2)
    constant Fr* challenge          [[buffer(3)]],
    constant uint& n                [[buffer(4)]],   // current domain size (must be even)
    uint gid                        [[thread_position_in_grid]]
) {
    uint half_n = n >> 1;
    if (gid >= half_n) return;

    Fr a = evals[gid];
    Fr b = evals[gid + half_n];
    Fr sum = fr_add(a, b);
    Fr diff = fr_sub(a, b);

    Fr c = challenge[0];
    Fr d_inv = domain_inv[gid];
    // challenge * diff * domain_inv[i]
    Fr term = fr_mul(c, fr_mul(diff, d_inv));

    result[gid] = fr_add(sum, term);
}

// Fused 2-round FRI fold: applies two consecutive fold rounds in one kernel.
// Reads 4 elements, computes both folds in registers, writes 1 element.
// Eliminates intermediate n/2 buffer and halves dispatch count.
//
// Round 1: fold n -> n/2 using challenge0 and domain_inv (size n/2)
// Round 2: fold n/2 -> n/4 using challenge1 and domain_inv2 (size n/4)
//
// domain_inv2[j] is the inverse of the round-2 domain element at position j.
// For standard multiplicative domains, domain_inv2[j] = domain_inv[2*j]
// (since the squared domain has elements domain[j]^2, and we store their inverses).
kernel void fri_fold_fused2_kernel(
    device const Fr* evals          [[buffer(0)]],  // size n
    device Fr* result               [[buffer(1)]],  // size n/4
    device const Fr* domain_inv     [[buffer(2)]],  // round-1 domain inverses, size n/2
    device const Fr* domain_inv2    [[buffer(3)]],  // round-2 domain inverses, size n/4
    constant Fr* challenge0         [[buffer(4)]],
    constant Fr* challenge1         [[buffer(5)]],
    constant uint& n                [[buffer(6)]],   // current domain size (must be >= 4)
    uint gid                        [[thread_position_in_grid]]
) {
    uint quarter = n >> 2;
    if (gid >= quarter) return;

    uint half_n = n >> 1;

    // Read 4 elements for the two fold rounds
    Fr a0 = evals[gid];
    Fr a1 = evals[gid + quarter];
    Fr a2 = evals[gid + half_n];
    Fr a3 = evals[gid + half_n + quarter];

    Fr c0 = challenge0[0];

    // Round 1: fold pairs (a0,a2) and (a1,a3) with stride half_n
    Fr sum02 = fr_add(a0, a2);
    Fr diff02 = fr_sub(a0, a2);
    Fr dinv_lo = domain_inv[gid];
    Fr f1_lo = fr_add(sum02, fr_mul(c0, fr_mul(diff02, dinv_lo)));

    Fr sum13 = fr_add(a1, a3);
    Fr diff13 = fr_sub(a1, a3);
    Fr dinv_hi = domain_inv[gid + quarter];
    Fr f1_hi = fr_add(sum13, fr_mul(c0, fr_mul(diff13, dinv_hi)));

    // Round 2: fold (f1_lo, f1_hi) with round-2 domain inverses
    Fr c1 = challenge1[0];
    Fr dinv2 = domain_inv2[gid];

    Fr sum_f = fr_add(f1_lo, f1_hi);
    Fr diff_f = fr_sub(f1_lo, f1_hi);
    result[gid] = fr_add(sum_f, fr_mul(c1, fr_mul(diff_f, dinv2)));
}

// GPU batch domain inverse: compute 1/domain[i] for each element.
// Uses the field inversion via Fermat's little theorem: a^{-1} = a^{p-2} mod p.
// This is compute-intensive but embarrassingly parallel on GPU.
kernel void fri_domain_inverse_kernel(
    device const Fr* domain         [[buffer(0)]],
    device Fr* domain_inv           [[buffer(1)]],
    constant uint& count            [[buffer(2)]],
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    domain_inv[gid] = fr_inv(domain[gid]);
}
