// Coset LDE (Low-Degree Extension) GPU kernels
// Operations: zero-pad coefficients to extended domain, apply coset shift (multiply by g^i)
// Supports BabyBear, Goldilocks, BN254 Fr, and Mersenne31 fields.

#include "../fields/babybear.metal"
#include "../fields/goldilocks.metal"
#include "../fields/bn254_fr.metal"
#include "../fields/mersenne31.metal"

// ============================================================
// Zero-pad kernel: copy N coefficients into M-element buffer, zero the rest
// Works for any field with uint-sized or ulong-sized elements.
// ============================================================

// Zero-pad for BabyBear (4 bytes per element)
kernel void zero_pad_bb(
    device const Bb* input           [[buffer(0)]],
    device Bb* output                [[buffer(1)]],
    constant uint& n_orig            [[buffer(2)]],   // original size
    constant uint& n_extended        [[buffer(3)]],   // extended size
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= n_extended) return;
    output[gid] = (gid < n_orig) ? input[gid] : bb_zero();
}

// Zero-pad for Goldilocks (8 bytes per element)
kernel void zero_pad_gl(
    device const Gl* input           [[buffer(0)]],
    device Gl* output                [[buffer(1)]],
    constant uint& n_orig            [[buffer(2)]],
    constant uint& n_extended        [[buffer(3)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= n_extended) return;
    output[gid] = (gid < n_orig) ? input[gid] : gl_zero();
}

// Zero-pad for BN254 Fr (32 bytes per element)
kernel void zero_pad_fr(
    device const Fr* input           [[buffer(0)]],
    device Fr* output                [[buffer(1)]],
    constant uint& n_orig            [[buffer(2)]],
    constant uint& n_extended        [[buffer(3)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= n_extended) return;
    output[gid] = (gid < n_orig) ? input[gid] : fr_zero();
}

// Zero-pad for Mersenne31 (4 bytes per element)
kernel void zero_pad_m31(
    device const M31* input          [[buffer(0)]],
    device M31* output               [[buffer(1)]],
    constant uint& n_orig            [[buffer(2)]],
    constant uint& n_extended        [[buffer(3)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= n_extended) return;
    output[gid] = (gid < n_orig) ? input[gid] : m31_zero();
}

// ============================================================
// Coset shift kernels: multiply coefficient[i] by g^i
// where g is the coset generator (precomputed powers passed in buffer)
// ============================================================

// Coset shift for BabyBear
// powers[i] = g^i mod p, precomputed on CPU
kernel void coset_shift_bb(
    device Bb* data                  [[buffer(0)]],
    device const Bb* powers          [[buffer(1)]],   // powers[i] = g^i
    constant uint& n                 [[buffer(2)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    data[gid] = bb_mul(data[gid], powers[gid]);
}

// Coset shift for Goldilocks
kernel void coset_shift_gl(
    device Gl* data                  [[buffer(0)]],
    device const Gl* powers          [[buffer(1)]],
    constant uint& n                 [[buffer(2)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    data[gid] = gl_mul(data[gid], powers[gid]);
}

// Coset shift for BN254 Fr
kernel void coset_shift_fr(
    device Fr* data                  [[buffer(0)]],
    device const Fr* powers          [[buffer(1)]],
    constant uint& n                 [[buffer(2)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    data[gid] = fr_mul(data[gid], powers[gid]);
}

// Coset shift for Mersenne31
kernel void coset_shift_m31(
    device M31* data                 [[buffer(0)]],
    device const M31* powers         [[buffer(1)]],
    constant uint& n                 [[buffer(2)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    data[gid] = m31_mul(data[gid], powers[gid]);
}

// ============================================================
// Fused zero-pad + coset shift: copy N elements, zero-pad to M,
// then multiply by g^i. Single kernel to avoid extra memory barrier.
// ============================================================

kernel void zero_pad_coset_shift_bb(
    device const Bb* input           [[buffer(0)]],
    device Bb* output                [[buffer(1)]],
    device const Bb* powers          [[buffer(2)]],   // powers[i] = g^i, length M
    constant uint& n_orig            [[buffer(3)]],
    constant uint& n_extended        [[buffer(4)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= n_extended) return;
    Bb val = (gid < n_orig) ? input[gid] : bb_zero();
    output[gid] = bb_mul(val, powers[gid]);
}

kernel void zero_pad_coset_shift_gl(
    device const Gl* input           [[buffer(0)]],
    device Gl* output                [[buffer(1)]],
    device const Gl* powers          [[buffer(2)]],
    constant uint& n_orig            [[buffer(3)]],
    constant uint& n_extended        [[buffer(4)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= n_extended) return;
    Gl val = (gid < n_orig) ? input[gid] : gl_zero();
    output[gid] = gl_mul(val, powers[gid]);
}

kernel void zero_pad_coset_shift_fr(
    device const Fr* input           [[buffer(0)]],
    device Fr* output                [[buffer(1)]],
    device const Fr* powers          [[buffer(2)]],
    constant uint& n_orig            [[buffer(3)]],
    constant uint& n_extended        [[buffer(4)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= n_extended) return;
    Fr val = (gid < n_orig) ? input[gid] : fr_zero();
    output[gid] = fr_mul(val, powers[gid]);
}

kernel void zero_pad_coset_shift_m31(
    device const M31* input          [[buffer(0)]],
    device M31* output               [[buffer(1)]],
    device const M31* powers         [[buffer(2)]],
    constant uint& n_orig            [[buffer(3)]],
    constant uint& n_extended        [[buffer(4)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= n_extended) return;
    M31 val = (gid < n_orig) ? input[gid] : m31_zero();
    output[gid] = m31_mul(val, powers[gid]);
}

// ============================================================
// Batch coset shift: process multiple columns packed contiguously.
// Layout: column c, element i is at data[c * stride + i].
// ============================================================

kernel void batch_coset_shift_bb(
    device Bb* data                  [[buffer(0)]],
    device const Bb* powers          [[buffer(1)]],   // shared powers (length = stride)
    constant uint& stride            [[buffer(2)]],   // elements per column
    constant uint& num_cols          [[buffer(3)]],
    uint gid                         [[thread_position_in_grid]]
) {
    uint total = stride * num_cols;
    if (gid >= total) return;
    uint idx_in_col = gid % stride;
    data[gid] = bb_mul(data[gid], powers[idx_in_col]);
}

kernel void batch_coset_shift_gl(
    device Gl* data                  [[buffer(0)]],
    device const Gl* powers          [[buffer(1)]],
    constant uint& stride            [[buffer(2)]],
    constant uint& num_cols          [[buffer(3)]],
    uint gid                         [[thread_position_in_grid]]
) {
    uint total = stride * num_cols;
    if (gid >= total) return;
    uint idx_in_col = gid % stride;
    data[gid] = gl_mul(data[gid], powers[idx_in_col]);
}
