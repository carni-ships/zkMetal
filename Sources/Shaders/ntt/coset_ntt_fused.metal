// Fused Coset-Shift + NTT Butterfly GPU kernels for STARK provers
//
// Key optimization: instead of a separate coset-shift pass followed by NTT,
// we fuse the shift multiplication into the first butterfly stage.
// For DIT NTT stage 0: a[i] and a[i+1] are paired. Before the butterfly,
// multiply each element by the coset shift power: a[i] *= shift^(bitrev(i)).
//
// This saves one full GPU pass over the data (bandwidth-bound on large arrays).
//
// Also provides a fused last-iNTT-stage + coset-unshift for cosetINTT.

#include "../fields/bn254_fr.metal"
#include "../fields/babybear.metal"

// ============================================================
// Fused coset-shift + first butterfly stage (DIT, stage 0)
// For forward coset NTT: pre-multiply data[i] by shift_powers[i]
// then do the stage-0 butterfly: pairs at (2k, 2k+1).
// shift_powers[i] = cosetShift^(bitrev(i, logN)) precomputed on CPU.
//
// This kernel replaces: coset_shift dispatch + ntt_butterfly(stage=0).
// ============================================================

kernel void coset_shift_butterfly_fr(
    device Fr* data                    [[buffer(0)]],
    device const Fr* twiddles          [[buffer(1)]],
    device const Fr* shift_powers      [[buffer(2)]],   // shift^(bitrev(i)), length N
    constant uint& n                   [[buffer(3)]],
    uint gid                           [[thread_position_in_grid]]
) {
    uint num_butterflies = n >> 1;
    if (gid >= num_butterflies) return;

    // Stage 0: half_block = 1, block_size = 2
    uint i = gid * 2;
    uint j = i + 1;

    // Fused: apply coset shift before butterfly
    Fr a = fr_mul(data[i], shift_powers[i]);
    Fr b = fr_mul(data[j], shift_powers[j]);

    // Stage 0 butterfly: twiddle_idx = (gid % 1) * (n / 2) = 0
    // So twiddle is always 1 for stage 0
    data[i] = fr_add(a, b);
    data[j] = fr_sub(a, b);
}

kernel void coset_shift_butterfly_bb(
    device Bb* data                    [[buffer(0)]],
    device const Bb* twiddles          [[buffer(1)]],
    device const Bb* shift_powers      [[buffer(2)]],
    constant uint& n                   [[buffer(3)]],
    uint gid                           [[thread_position_in_grid]]
) {
    uint num_butterflies = n >> 1;
    if (gid >= num_butterflies) return;

    uint i = gid * 2;
    uint j = i + 1;

    Bb a = bb_mul(data[i], shift_powers[i]);
    Bb b = bb_mul(data[j], shift_powers[j]);

    data[i] = bb_add(a, b);
    data[j] = bb_sub(a, b);
}

// ============================================================
// Fused last iNTT stage + coset unshift (DIF)
// For inverse coset NTT: after the last iNTT butterfly stage,
// multiply by inv_shift_powers and scale by 1/N.
// Combines: intt_butterfly(last stage) + coset_unshift + scale.
//
// DIF last stage (stage 0): pairs at (2k, 2k+1), twiddle = 1
//   a' = a + b
//   b' = a - b
// Then multiply by inv_shift_powers[i] * (1/N).
// ============================================================

kernel void intt_unshift_scale_fr(
    device Fr* data                    [[buffer(0)]],
    device const Fr* twiddles_inv      [[buffer(1)]],
    device const Fr* inv_shift_powers  [[buffer(2)]],   // shift^(-bitrev(i)) * (1/N)
    constant uint& n                   [[buffer(3)]],
    uint gid                           [[thread_position_in_grid]]
) {
    uint num_butterflies = n >> 1;
    if (gid >= num_butterflies) return;

    uint i = gid * 2;
    uint j = i + 1;

    Fr a = data[i];
    Fr b = data[j];

    // Last DIF stage: twiddle = 1
    Fr sum  = fr_add(a, b);
    Fr diff = fr_sub(a, b);

    // Fused: unshift + scale
    data[i] = fr_mul(sum,  inv_shift_powers[i]);
    data[j] = fr_mul(diff, inv_shift_powers[j]);
}

kernel void intt_unshift_scale_bb(
    device Bb* data                    [[buffer(0)]],
    device const Bb* twiddles_inv      [[buffer(1)]],
    device const Bb* inv_shift_powers  [[buffer(2)]],
    constant uint& n                   [[buffer(3)]],
    uint gid                           [[thread_position_in_grid]]
) {
    uint num_butterflies = n >> 1;
    if (gid >= num_butterflies) return;

    uint i = gid * 2;
    uint j = i + 1;

    Bb a = data[i];
    Bb b = data[j];

    Bb sum  = bb_add(a, b);
    Bb diff = bb_sub(a, b);

    data[i] = bb_mul(sum,  inv_shift_powers[i]);
    data[j] = bb_mul(diff, inv_shift_powers[j]);
}

// ============================================================
// Standalone coset shift for coefficient-form input.
// Multiplies coeffs[i] by shift^i. Used when not fusing with butterfly.
// ============================================================

kernel void coset_shift_powers_fr(
    device Fr* data                    [[buffer(0)]],
    device const Fr* powers            [[buffer(1)]],   // powers[i] = shift^i
    constant uint& n                   [[buffer(2)]],
    uint gid                           [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    data[gid] = fr_mul(data[gid], powers[gid]);
}

kernel void coset_shift_powers_bb(
    device Bb* data                    [[buffer(0)]],
    device const Bb* powers            [[buffer(1)]],
    constant uint& n                   [[buffer(2)]],
    uint gid                           [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    data[gid] = bb_mul(data[gid], powers[gid]);
}

// ============================================================
// Coset unshift: multiply data[i] by shift^(-i).
// ============================================================

kernel void coset_unshift_powers_fr(
    device Fr* data                    [[buffer(0)]],
    device const Fr* inv_powers        [[buffer(1)]],   // inv_powers[i] = shift^(-i)
    constant uint& n                   [[buffer(2)]],
    uint gid                           [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    data[gid] = fr_mul(data[gid], inv_powers[gid]);
}

kernel void coset_unshift_powers_bb(
    device Bb* data                    [[buffer(0)]],
    device const Bb* inv_powers        [[buffer(1)]],
    constant uint& n                   [[buffer(2)]],
    uint gid                           [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    data[gid] = bb_mul(data[gid], inv_powers[gid]);
}
