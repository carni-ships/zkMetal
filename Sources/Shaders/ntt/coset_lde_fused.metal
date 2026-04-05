// Fused Coset LDE GPU kernels
// Combines zero-pad + coset-shift into single dispatch with custom coset generator.
// Also includes batch variants that process multiple columns in one dispatch.
// Supports BN254 Fr and BabyBear fields.

#include "../fields/bn254_fr.metal"
#include "../fields/babybear.metal"

// ============================================================
// Fused zero-pad + coset shift: single kernel for LDE pipeline.
// Copies N coefficients, zero-pads to M, multiplies by g^i.
// powers[i] = cosetShift^i (precomputed on CPU).
// ============================================================

kernel void lde_zero_pad_coset_shift_fr(
    device const Fr* input           [[buffer(0)]],
    device Fr* output                [[buffer(1)]],
    device const Fr* powers          [[buffer(2)]],   // powers[i] = g^i, length M
    constant uint& n_orig            [[buffer(3)]],
    constant uint& n_extended        [[buffer(4)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= n_extended) return;
    Fr val = (gid < n_orig) ? input[gid] : fr_zero();
    output[gid] = fr_mul(val, powers[gid]);
}

kernel void lde_zero_pad_coset_shift_bb(
    device const Bb* input           [[buffer(0)]],
    device Bb* output                [[buffer(1)]],
    device const Bb* powers          [[buffer(2)]],
    constant uint& n_orig            [[buffer(3)]],
    constant uint& n_extended        [[buffer(4)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= n_extended) return;
    Bb val = (gid < n_orig) ? input[gid] : bb_zero();
    output[gid] = bb_mul(val, powers[gid]);
}

// ============================================================
// Plain zero-pad (no coset shift): for coefficient-form LDE
// where the coset shift is applied separately or is identity.
// ============================================================

kernel void lde_zero_pad_fr(
    device const Fr* input           [[buffer(0)]],
    device Fr* output                [[buffer(1)]],
    constant uint& n_orig            [[buffer(2)]],
    constant uint& n_extended        [[buffer(3)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= n_extended) return;
    output[gid] = (gid < n_orig) ? input[gid] : fr_zero();
}

kernel void lde_zero_pad_bb(
    device const Bb* input           [[buffer(0)]],
    device Bb* output                [[buffer(1)]],
    constant uint& n_orig            [[buffer(2)]],
    constant uint& n_extended        [[buffer(3)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= n_extended) return;
    output[gid] = (gid < n_orig) ? input[gid] : bb_zero();
}

// ============================================================
// Batch fused zero-pad + coset shift: process numCols columns
// packed contiguously in a single dispatch.
// Layout: column c at input[c * n_orig .. (c+1)*n_orig]
//         column c at output[c * n_extended .. (c+1)*n_extended]
// powers[] is shared across all columns (length = n_extended).
// ============================================================

kernel void lde_batch_zero_pad_coset_shift_fr(
    device const Fr* input           [[buffer(0)]],
    device Fr* output                [[buffer(1)]],
    device const Fr* powers          [[buffer(2)]],
    constant uint& n_orig            [[buffer(3)]],
    constant uint& n_extended        [[buffer(4)]],
    constant uint& num_cols          [[buffer(5)]],
    uint gid                         [[thread_position_in_grid]]
) {
    uint total = n_extended * num_cols;
    if (gid >= total) return;
    uint col = gid / n_extended;
    uint idx = gid % n_extended;
    uint in_offset = col * n_orig;
    Fr val = (idx < n_orig) ? input[in_offset + idx] : fr_zero();
    output[gid] = fr_mul(val, powers[idx]);
}

kernel void lde_batch_zero_pad_coset_shift_bb(
    device const Bb* input           [[buffer(0)]],
    device Bb* output                [[buffer(1)]],
    device const Bb* powers          [[buffer(2)]],
    constant uint& n_orig            [[buffer(3)]],
    constant uint& n_extended        [[buffer(4)]],
    constant uint& num_cols          [[buffer(5)]],
    uint gid                         [[thread_position_in_grid]]
) {
    uint total = n_extended * num_cols;
    if (gid >= total) return;
    uint col = gid / n_extended;
    uint idx = gid % n_extended;
    uint in_offset = col * n_orig;
    Bb val = (idx < n_orig) ? input[in_offset + idx] : bb_zero();
    output[gid] = bb_mul(val, powers[idx]);
}

// ============================================================
// In-place coset shift: data[i] *= powers[i]
// Used when coefficients are already in the output buffer.
// ============================================================

kernel void lde_coset_shift_inplace_fr(
    device Fr* data                  [[buffer(0)]],
    device const Fr* powers          [[buffer(1)]],
    constant uint& n                 [[buffer(2)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    data[gid] = fr_mul(data[gid], powers[gid]);
}

kernel void lde_coset_shift_inplace_bb(
    device Bb* data                  [[buffer(0)]],
    device const Bb* powers          [[buffer(1)]],
    constant uint& n                 [[buffer(2)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    data[gid] = bb_mul(data[gid], powers[gid]);
}
