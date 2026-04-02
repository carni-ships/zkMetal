// Keccak-256 GPU kernel — batch hashing
// Each thread computes one Keccak-256 hash of a fixed-size input.
// Implements the Keccak-f[1600] permutation with 24 rounds.

#include <metal_stdlib>
using namespace metal;

// Keccak-f[1600] round constants
constant ulong KECCAK_RC[24] = {
    0x0000000000000001UL, 0x0000000000008082UL,
    0x800000000000808aUL, 0x8000000080008000UL,
    0x000000000000808bUL, 0x0000000080000001UL,
    0x8000000080008081UL, 0x8000000000008009UL,
    0x000000000000008aUL, 0x0000000000000088UL,
    0x0000000080008009UL, 0x000000008000000aUL,
    0x000000008000808bUL, 0x800000000000008bUL,
    0x8000000000008089UL, 0x8000000000008003UL,
    0x8000000000008002UL, 0x8000000000000080UL,
    0x000000000000800aUL, 0x800000008000000aUL,
    0x8000000080008081UL, 0x8000000000008080UL,
    0x0000000080000001UL, 0x8000000080008008UL,
};

// Rotation offsets for Keccak
constant uint KECCAK_ROT[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
};

// Pi permutation indices
constant uint KECCAK_PI[25] = {
     0, 10, 20,  5, 15,
    16,  1, 11, 21,  6,
     7, 17,  2, 12, 22,
    23,  8, 18,  3, 13,
    14, 24,  9, 19,  4,
};

ulong rotl64(ulong x, uint n) {
    return (x << n) | (x >> (64 - n));
}

// Keccak-f[1600] permutation on 25 x 64-bit state
// Optimized: in-place Rho+Pi using Pi's 24-cycle (eliminates 25-element tmp array),
// row-by-row Chi (5 temps instead of 25). Peak register pressure: ~35 ulongs vs ~60.
void keccak_f1600(thread ulong state[25]) {
    for (uint round = 0; round < 24; round++) {
        // Theta — fully unrolled for explicit register usage
        ulong C0 = state[0] ^ state[5] ^ state[10] ^ state[15] ^ state[20];
        ulong C1 = state[1] ^ state[6] ^ state[11] ^ state[16] ^ state[21];
        ulong C2 = state[2] ^ state[7] ^ state[12] ^ state[17] ^ state[22];
        ulong C3 = state[3] ^ state[8] ^ state[13] ^ state[18] ^ state[23];
        ulong C4 = state[4] ^ state[9] ^ state[14] ^ state[19] ^ state[24];

        ulong D0 = C4 ^ rotl64(C1, 1);
        ulong D1 = C0 ^ rotl64(C2, 1);
        ulong D2 = C1 ^ rotl64(C3, 1);
        ulong D3 = C2 ^ rotl64(C4, 1);
        ulong D4 = C3 ^ rotl64(C0, 1);

        state[0]  ^= D0; state[5]  ^= D0; state[10] ^= D0; state[15] ^= D0; state[20] ^= D0;
        state[1]  ^= D1; state[6]  ^= D1; state[11] ^= D1; state[16] ^= D1; state[21] ^= D1;
        state[2]  ^= D2; state[7]  ^= D2; state[12] ^= D2; state[17] ^= D2; state[22] ^= D2;
        state[3]  ^= D3; state[8]  ^= D3; state[13] ^= D3; state[18] ^= D3; state[23] ^= D3;
        state[4]  ^= D4; state[9]  ^= D4; state[14] ^= D4; state[19] ^= D4; state[24] ^= D4;

        // Rho + Pi — in-place using Pi's single 24-cycle
        // Reverse cycle: each source position hasn't been written yet
        // Cycle: 1→10→7→11→17→18→3→5→16→8→21→24→4→15→23→19→13→12→2→20→14→22→9→6→(1)
        ulong temp = state[1];
        state[1]  = rotl64(state[6],  44);
        state[6]  = rotl64(state[9],  20);
        state[9]  = rotl64(state[22], 61);
        state[22] = rotl64(state[14], 39);
        state[14] = rotl64(state[20], 18);
        state[20] = rotl64(state[2],  62);
        state[2]  = rotl64(state[12], 43);
        state[12] = rotl64(state[13], 25);
        state[13] = rotl64(state[19],  8);
        state[19] = rotl64(state[23], 56);
        state[23] = rotl64(state[15], 41);
        state[15] = rotl64(state[4],  27);
        state[4]  = rotl64(state[24], 14);
        state[24] = rotl64(state[21],  2);
        state[21] = rotl64(state[8],  55);
        state[8]  = rotl64(state[16], 45);
        state[16] = rotl64(state[5],  36);
        state[5]  = rotl64(state[3],  28);
        state[3]  = rotl64(state[18], 21);
        state[18] = rotl64(state[17], 15);
        state[17] = rotl64(state[11], 10);
        state[11] = rotl64(state[7],   6);
        state[7]  = rotl64(state[10],  3);
        state[10] = rotl64(temp,       1);

        // Chi — row by row, 5 temps per row (in-place, no 25-element array)
        #pragma unroll
        for (uint y = 0; y < 5; y++) {
            ulong t0 = state[y*5], t1 = state[y*5+1], t2 = state[y*5+2],
                  t3 = state[y*5+3], t4 = state[y*5+4];
            state[y*5]   = t0 ^ (~t1 & t2);
            state[y*5+1] = t1 ^ (~t2 & t3);
            state[y*5+2] = t2 ^ (~t3 & t4);
            state[y*5+3] = t3 ^ (~t4 & t0);
            state[y*5+4] = t4 ^ (~t0 & t1);
        }

        // Iota
        state[0] ^= KECCAK_RC[round];
    }
}

// Keccak-256 hash of a 64-byte input (two 32-byte leaves for Merkle)
// rate = 1088 bits = 136 bytes for Keccak-256
// Absorb 64 bytes, pad, squeeze 32 bytes
kernel void keccak256_hash_64(
    device const uchar* input      [[buffer(0)]],
    device uchar* output           [[buffer(1)]],
    constant uint& count           [[buffer(2)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    // Initialize state to zero
    ulong state[25];
    for (uint i = 0; i < 25; i++) state[i] = 0;

    // Absorb 64 bytes of input (8 lanes of 8 bytes each)
    device const ulong* in64 = (device const ulong*)(input + gid * 64);
    for (uint i = 0; i < 8; i++) {
        state[i] ^= in64[i];
    }

    // Keccak padding: 0x01 at byte 64 (lane 8, byte 0), 0x80 at byte 135 (lane 16, byte 7)
    // For Keccak-256 (rate=136 bytes), domain separator is 0x01
    state[8] ^= 0x01UL;
    state[16] ^= 0x8000000000000000UL;

    // Permute
    keccak_f1600(state);

    // Squeeze 32 bytes (4 lanes)
    device ulong* out64 = (device ulong*)(output + gid * 32);
    for (uint i = 0; i < 4; i++) {
        out64[i] = state[i];
    }
}

// Inline Keccak-256 hash of two 32-byte inputs → 32-byte output (for Merkle)
void keccak256_hash_pair(threadgroup ulong* left, threadgroup ulong* right,
                          threadgroup ulong* out) {
    ulong state[25];
    for (uint i = 0; i < 25; i++) state[i] = 0;
    // Absorb 64 bytes: left (32 bytes = 4 ulongs) + right (32 bytes = 4 ulongs)
    for (uint i = 0; i < 4; i++) state[i] = left[i];
    for (uint i = 0; i < 4; i++) state[4 + i] = right[i];
    state[8] ^= 0x01UL;
    state[16] ^= 0x8000000000000000UL;
    keccak_f1600(state);
    for (uint i = 0; i < 4; i++) out[i] = state[i];
}

// Fused multi-level Keccak Merkle tree: each threadgroup processes a subtree
// Shared memory: 1024 * 4 ulongs = 32KB
#define KECCAK_SUBTREE_SIZE 1024

kernel void keccak256_merkle_fused(
    device const uchar* leaves    [[buffer(0)]],
    device uchar* roots           [[buffer(1)]],
    constant uint& num_levels     [[buffer(2)]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tgid                     [[threadgroup_position_in_grid]],
    uint tg_size                  [[threads_per_threadgroup]]
) {
    // Each node is 4 ulongs = 32 bytes
    threadgroup ulong shared_data[KECCAK_SUBTREE_SIZE * 4];

    // Load leaves (32 bytes each = 4 ulongs each)
    uint leaf_base = tgid * KECCAK_SUBTREE_SIZE;
    device const ulong* leaves64 = (device const ulong*)leaves;
    uint src_lo = (leaf_base + tid) * 4;
    uint src_hi = (leaf_base + tid + tg_size) * 4;
    uint dst_lo = tid * 4;
    uint dst_hi = (tid + tg_size) * 4;
    for (uint i = 0; i < 4; i++) shared_data[dst_lo + i] = leaves64[src_lo + i];
    for (uint i = 0; i < 4; i++) shared_data[dst_hi + i] = leaves64[src_hi + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint active = KECCAK_SUBTREE_SIZE;
    for (uint level = 0; level < num_levels; level++) {
        uint pairs = active >> 1;
        if (tid < pairs) {
            keccak256_hash_pair(&shared_data[tid * 2 * 4],
                                &shared_data[(tid * 2 + 1) * 4],
                                &shared_data[tid * 4]);
        }
        active = pairs;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        device ulong* out64 = (device ulong*)(roots + tgid * 32);
        for (uint i = 0; i < 4; i++) out64[i] = shared_data[i];
    }
}

// Keccak-256 hash of a 32-byte input (single field element or leaf)
kernel void keccak256_hash_32(
    device const uchar* input      [[buffer(0)]],
    device uchar* output           [[buffer(1)]],
    constant uint& count           [[buffer(2)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    ulong state[25];
    for (uint i = 0; i < 25; i++) state[i] = 0;

    device const ulong* in64 = (device const ulong*)(input + gid * 32);
    for (uint i = 0; i < 4; i++) {
        state[i] ^= in64[i];
    }

    state[4] ^= 0x01UL;
    state[16] ^= 0x8000000000000000UL;

    keccak_f1600(state);

    device ulong* out64 = (device ulong*)(output + gid * 32);
    for (uint i = 0; i < 4; i++) {
        out64[i] = state[i];
    }
}
