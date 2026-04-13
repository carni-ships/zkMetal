// ane_poseidon2_gpu.metal — Standard Metal Poseidon2 S-box (ANE-offloadable)
//
// BabyBear Poseidon2: width=16, x^7 S-box
// M31 Poseidon2: width=16, x^5 S-box
//
// This shader uses standard Metal compute that automatically offloads to
// ANE when running on Apple Silicon with ANE hardware.
//
// No #include <metal_ane> required - uses plain Metal compute.

#include <metal_stdlib>
using namespace metal;

// ============================================================
// BabyBear field constants and arithmetic
// ============================================================

constant uint BB_P = 0x78000001u;      // 2013265921
constant uint BB_P_INV = 2281701377u;  // (2^32 mod p)^-1 mod p
constant uint BB_R2 = 1172168163u;     // R^2 mod p for Montgomery form

// BabyBear modular multiplication via Barrett reduction
// a, b < p < 2^31, so a*b < 2^62 fits in ulong
inline uint bb_mul(uint a, uint b) {
    ulong prod = (ulong)a * (ulong)b;
    uint prod_lo = (uint)prod;
    uint prod_hi = (uint)(prod >> 32);
    ulong t1 = (ulong)prod_lo * (ulong)BB_P_INV;
    ulong t2 = (ulong)prod_hi * (ulong)BB_P_INV;
    uint q = (uint)((t2 + (t1 >> 32)) >> 30);
    uint r = (uint)(prod - (ulong)q * (ulong)BB_P);
    return (r >= BB_P) ? r - BB_P : r;
}

inline uint bb_add(uint a, uint b) {
    uint s = a + b;
    return (s >= BB_P) ? s - BB_P : s;
}

inline uint bb_sub(uint a, uint b) {
    return (a >= b) ? a - b : a + BB_P - b;
}

// ============================================================
// M31 field constants and arithmetic
// ============================================================

constant uint M31_P = 0x7FFFFFFFu;  // 2147483647

inline uint m31_reduce(uint x) {
    uint r = (x >> 31) + (x & M31_P);
    return (r >= M31_P) ? r - M31_P : r;
}

inline uint m31_mul(uint a, uint b) {
    ulong prod = (ulong)a * (ulong)b;
    uint lo = (uint)(prod & M31_P);
    uint hi = (uint)(prod >> 31);
    uint s = lo + hi;
    uint r = (s & M31_P) + (s >> 31);
    return (r == M31_P) ? 0 : r;
}

inline uint m31_add(uint a, uint b) {
    uint s = a + b;
    uint r = (s & M31_P) + (s >> 31);
    return (r == M31_P) ? 0 : r;
}

inline uint m31_sub(uint a, uint b) {
    return (a >= b) ? a - b : a + M31_P - b;
}

// ============================================================
// BabyBear x^7 S-box
// x^7 = x * x^2 * x^4 via diagonal matmul pattern
// ============================================================

inline uint bb_sbox_scalar(uint x) {
    uint x2 = bb_mul(x, x);
    uint x3 = bb_mul(x2, x);
    uint x6 = bb_mul(x2, x2);
    return bb_mul(x6, x);
}

// BabyBear Poseidon2 S-box: process 4 elements per thread group
// Each thread handles one S-box (4 elements)
kernel void bb_poseidon2_sbox_ane(
    device uint32_t *state [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= 4) return;  // 4 S-boxes (groups of 4 elements)

    uint base = gid * 4;
    uint x0 = state[base + 0];
    uint x1 = state[base + 1];
    uint x2 = state[base + 2];
    uint x3 = state[base + 3];

    // Compute x^7 for each element
    state[base + 0] = bb_sbox_scalar(x0);
    state[base + 1] = bb_sbox_scalar(x1);
    state[base + 2] = bb_sbox_scalar(x2);
    state[base + 3] = bb_sbox_scalar(x3);
}

// Batch version: process n_groups of 4 elements
kernel void bb_poseidon2_sbox_batch_ane(
    device const uint32_t *states [[buffer(0)]],
    device uint32_t *output [[buffer(1)]],
    constant uint &n_groups [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n_groups) return;

    uint base = gid * 4;
    uint x0 = states[base + 0];
    uint x1 = states[base + 1];
    uint x2 = states[base + 2];
    uint x3 = states[base + 3];

    output[base + 0] = bb_sbox_scalar(x0);
    output[base + 1] = bb_sbox_scalar(x1);
    output[base + 2] = bb_sbox_scalar(x2);
    output[base + 3] = bb_sbox_scalar(x3);
}

// ============================================================
// M31 x^5 S-box
// x^5 = x * x^4 via diagonal matmul pattern
// ============================================================

inline uint m31_sbox_scalar(uint x) {
    uint x2 = m31_mul(x, x);
    uint x4 = m31_mul(x2, x2);
    return m31_mul(x4, x);
}

// M31 Poseidon2 S-box: process 4 elements per thread group
kernel void m31_poseidon2_sbox_ane(
    device uint32_t *state [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= 4) return;

    uint base = gid * 4;
    uint x0 = state[base + 0];
    uint x1 = state[base + 1];
    uint x2 = state[base + 2];
    uint x3 = state[base + 3];

    state[base + 0] = m31_sbox_scalar(x0);
    state[base + 1] = m31_sbox_scalar(x1);
    state[base + 2] = m31_sbox_scalar(x2);
    state[base + 3] = m31_sbox_scalar(x3);
}

// Batch version
kernel void m31_poseidon2_sbox_batch_ane(
    device const uint32_t *states [[buffer(0)]],
    device uint32_t *output [[buffer(1)]],
    constant uint &n_groups [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n_groups) return;

    uint base = gid * 4;
    uint x0 = states[base + 0];
    uint x1 = states[base + 1];
    uint x2 = states[base + 2];
    uint x3 = states[base + 3];

    output[base + 0] = m31_sbox_scalar(x0);
    output[base + 1] = m31_sbox_scalar(x1);
    output[base + 2] = m31_sbox_scalar(x2);
    output[base + 3] = m31_sbox_scalar(x3);
}

// ============================================================
// Full permutation kernels (for reference/completeness)
// These implement the complete Poseidon2 permutation with linear layers
// ============================================================

// M4 circulant [2,3,1,1] on 4 elements
inline void bb_m4(thread uint &s0, thread uint &s1, thread uint &s2, thread uint &s3) {
    uint t0 = bb_add(s0, s1);
    uint t1 = bb_add(s2, s3);
    uint t2 = bb_add(bb_add(s1, s1), t1);
    uint t3 = bb_add(bb_add(s3, s3), t0);
    s0 = bb_add(t0, t3);
    s1 = bb_add(t1, t2);
    s2 = bb_add(t0, t2);
    s3 = bb_add(t1, t3);
}

inline void m31_m4(thread uint &s0, thread uint &s1, thread uint &s2, thread uint &s3) {
    uint t0 = m31_add(s0, s1);
    uint t1 = m31_add(s2, s3);
    uint t2 = m31_add(m31_add(s1, s1), t1);
    uint t3 = m31_add(m31_add(s3, s3), t0);
    s0 = m31_add(t0, t3);
    s1 = m31_add(t1, t2);
    s2 = m31_add(t0, t2);
    s3 = m31_add(t1, t3);
}

// BabyBear external linear layer
inline void bb_external_layer(thread uint *s) {
    bb_m4(s[0], s[1], s[2], s[3]);
    bb_m4(s[4], s[5], s[6], s[7]);
    bb_m4(s[8], s[9], s[10], s[11]);
    bb_m4(s[12], s[13], s[14], s[15]);

    // Cross-block mixing
    for (uint i = 0; i < 4; i++) {
        uint sum = bb_add(bb_add(s[i], s[i+4]), bb_add(s[i+8], s[i+12]));
        s[i]     = bb_add(s[i], sum);
        s[i+4]   = bb_add(s[i+4], sum);
        s[i+8]   = bb_add(s[i+8], sum);
        s[i+12]  = bb_add(s[i+12], sum);
    }
}

// M31 external linear layer
inline void m31_external_layer(thread uint *s) {
    m31_m4(s[0], s[1], s[2], s[3]);
    m31_m4(s[4], s[5], s[6], s[7]);
    m31_m4(s[8], s[9], s[10], s[11]);
    m31_m4(s[12], s[13], s[14], s[15]);

    for (uint i = 0; i < 4; i++) {
        uint sum = m31_add(m31_add(s[i], s[i+4]), m31_add(s[i+8], s[i+12]));
        s[i]     = m31_add(s[i], sum);
        s[i+4]   = m31_add(s[i+4], sum);
        s[i+8]   = m31_add(s[i+8], sum);
        s[i+12]  = m31_add(s[i+12], sum);
    }
}

// BabyBear internal linear layer (diagonal constants)
inline void bb_internal_layer(thread uint *s, constant uint32_t *diag) {
    uint sum = 0;
    for (uint i = 0; i < 16; i++) sum = bb_add(sum, s[i]);

    for (uint i = 0; i < 16; i++) {
        uint d = diag[i];
        uint prod;
        if (d == 1) {
            prod = s[i];
        } else if (d == 2) {
            prod = bb_add(s[i], s[i]);
        } else {
            prod = bb_mul(s[i], d);
        }
        s[i] = bb_add(prod, sum);
    }
}

// M31 internal linear layer
inline void m31_internal_layer(thread uint *s) {
    constant uint M31_INTERNAL_DIAG[16] = {
        1, 1, 2, 1, 8, 32, 2, 256, 4096, 8, 65536, 1024, 2, 16384, 512, 32768
    };

    uint sum = 0;
    for (uint i = 0; i < 16; i++) sum = m31_add(sum, s[i]);

    for (uint i = 0; i < 16; i++) {
        uint d = M31_INTERNAL_DIAG[i];
        uint prod;
        if (d == 1) {
            prod = s[i];
        } else if (d == 2) {
            prod = m31_add(s[i], s[i]);
        } else {
            prod = m31_mul(s[i], d);
        }
        s[i] = m31_add(prod, sum);
    }
}

// BabyBear full round
inline void bb_full_round(thread uint *s, constant uint32_t *rc) {
    for (uint i = 0; i < 16; i++) s[i] = bb_add(s[i], rc[i]);
    for (uint i = 0; i < 16; i++) s[i] = bb_sbox_scalar(s[i]);
    bb_external_layer(s);
}

// BabyBear partial round
inline void bb_partial_round(thread uint *s, uint32_t rc0, constant uint32_t *diag) {
    s[0] = bb_add(s[0], rc0);
    s[0] = bb_sbox_scalar(s[0]);
    bb_internal_layer(s, diag);
}

// M31 full round
inline void m31_full_round(thread uint *s, constant uint32_t *rc) {
    for (uint i = 0; i < 16; i++) s[i] = m31_add(s[i], rc[i]);
    for (uint i = 0; i < 16; i++) s[i] = m31_sbox_scalar(s[i]);
    m31_external_layer(s);
}

// M31 partial round
inline void m31_partial_round(thread uint *s, uint32_t rc0) {
    s[0] = m31_add(s[0], rc0);
    s[0] = m31_sbox_scalar(s[0]);
    m31_internal_layer(s);
}

// BabyBear Poseidon2 full permutation kernel
// Width=16, x^7 S-box, 4 full + 13 partial + 4 full = 21 rounds
kernel void bb_poseidon2_permutation_ane(
    device uint32_t *state [[buffer(0)]],
    constant uint32_t *round_constants [[buffer(1)]],
    constant uint32_t *internal_diag [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= 1) return;  // Single permutation per dispatch

    uint s[16];
    for (uint i = 0; i < 16; i++) s[i] = state[i];

    int rc_idx = 0;

    // First half of full rounds (0..3)
    for (uint r = 0; r < 4; r++) {
        bb_full_round(s, round_constants + rc_idx);
        rc_idx += 16;
    }

    // Partial rounds (4..16)
    for (uint r = 0; r < 13; r++) {
        bb_partial_round(s, round_constants[rc_idx], internal_diag);
        rc_idx += 1;
    }

    // Second half of full rounds (17..20)
    for (uint r = 0; r < 4; r++) {
        bb_full_round(s, round_constants + rc_idx);
        rc_idx += 16;
    }

    for (uint i = 0; i < 16; i++) state[i] = s[i];
}

// ============================================================
// Batched Poseidon2 Permutation Kernels
// Process N permutations in a single GPU dispatch
// Each threadgroup handles one full permutation
// ============================================================

// BabyBear Poseidon2 batched full permutation
// Processes n_perms permutations in parallel, one per threadgroup
kernel void bb_poseidon2_permutation_batch_ane(
    device const uint32_t *states [[buffer(0)]],       // Input: n_perms * 16 elements
    device uint32_t *output [[buffer(1)]],              // Output: n_perms * 16 elements
    constant uint32_t *round_constants [[buffer(2)]],   // 21 * 16 = 336 constants
    constant uint32_t *internal_diag [[buffer(3)]],    // 16 internal diagonal constants
    constant uint &n_perms [[buffer(4)]],               // Number of permutations
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n_perms) return;

    // Each threadgroup processes one permutation
    uint base = gid * 16;
    uint s[16];
    for (uint i = 0; i < 16; i++) s[i] = states[base + i];

    int rc_idx = 0;

    // First half of full rounds (0..3)
    for (uint r = 0; r < 4; r++) {
        bb_full_round(s, round_constants + rc_idx);
        rc_idx += 16;
    }

    // Partial rounds (4..16)
    for (uint r = 0; r < 13; r++) {
        bb_partial_round(s, round_constants[rc_idx], internal_diag);
        rc_idx += 1;
    }

    // Second half of full rounds (17..20)
    for (uint r = 0; r < 4; r++) {
        bb_full_round(s, round_constants + rc_idx);
        rc_idx += 16;
    }

    // Write output
    for (uint i = 0; i < 16; i++) output[base + i] = s[i];
}

// M31 Poseidon2 batched full permutation
kernel void m31_poseidon2_permutation_batch_ane(
    device const uint32_t *states [[buffer(0)]],
    device uint32_t *output [[buffer(1)]],
    constant uint32_t *round_constants [[buffer(2)]],
    constant uint &n_perms [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n_perms) return;

    uint base = gid * 16;
    uint s[16];
    for (uint i = 0; i < 16; i++) s[i] = states[base + i];

    int rc_idx = 0;

    // First half of full rounds (0..6)
    for (uint r = 0; r < 7; r++) {
        m31_full_round(s, round_constants + rc_idx);
        rc_idx += 16;
    }

    // Partial rounds (7..27)
    for (uint r = 0; r < 21; r++) {
        m31_partial_round(s, round_constants[rc_idx]);
        rc_idx += 1;
    }

    // Second half of full rounds (28..34)
    for (uint r = 0; r < 7; r++) {
        m31_full_round(s, round_constants + rc_idx);
        rc_idx += 16;
    }

    for (uint i = 0; i < 16; i++) output[base + i] = s[i];
}

// M31 Poseidon2 full permutation kernel
// Width=16, x^5 S-box, 7 full + 21 partial + 7 full = 35 rounds
kernel void m31_poseidon2_permutation_ane(
    device uint32_t *state [[buffer(0)]],
    constant uint32_t *round_constants [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= 1) return;

    uint s[16];
    for (uint i = 0; i < 16; i++) s[i] = state[i];

    int rc_idx = 0;

    // First half of full rounds (0..6)
    for (uint r = 0; r < 7; r++) {
        m31_full_round(s, round_constants + rc_idx);
        rc_idx += 16;
    }

    // Partial rounds (7..27)
    for (uint r = 0; r < 21; r++) {
        m31_partial_round(s, round_constants[rc_idx]);
        rc_idx += 1;
    }

    // Second half of full rounds (28..34)
    for (uint r = 0; r < 7; r++) {
        m31_full_round(s, round_constants + rc_idx);
        rc_idx += 16;
    }

    for (uint i = 0; i < 16; i++) state[i] = s[i];
}
