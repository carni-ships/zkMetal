// Keccak-f1600 permutation with ARM NEON intrinsics
// Implements Keccak-256 (rate=136, capacity=64, output=32 bytes)
//
// NEON strategy:
//   - θ step: vectorized column parity XOR using uint64x2_t pairs
//   - χ step: vbicq_u64 for ~a[i+1] & a[i+2] (AND-NOT)
//   - ρ/π: scalar rotations (fixed per-lane, hard to vectorize)
//   - ι: scalar XOR of round constant into lane 0

#include "NeonFieldOps.h"
#include <arm_neon.h>
#include <string.h>
#include <stdint.h>

// Round constants for Keccak-f1600
static const uint64_t RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL,
};

// ρ rotation offsets indexed by lane position [x + 5*y]
static const int ROT[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14
};

// π permutation: new[i] = old[PI[i]]
static const int PI[25] = {
     0, 10, 20,  5, 15,
    16,  1, 11, 21,  6,
     7, 17,  2, 12, 22,
    23,  8, 18,  3, 13,
    14, 24,  9, 19,  4
};

static inline uint64_t rotl64(uint64_t x, int n) {
    if (n == 0) return x;
    return (x << n) | (x >> (64 - n));
}

// Pure scalar reference for correctness
static void keccak_f1600_scalar(uint64_t state[25]) {
    uint64_t s[25];
    memcpy(s, state, 200);

    for (int round = 0; round < 24; round++) {
        // θ
        uint64_t C[5];
        for (int x = 0; x < 5; x++)
            C[x] = s[x] ^ s[x+5] ^ s[x+10] ^ s[x+15] ^ s[x+20];
        uint64_t D[5];
        for (int x = 0; x < 5; x++)
            D[x] = C[(x+4) % 5] ^ rotl64(C[(x+1) % 5], 1);
        for (int i = 0; i < 25; i++)
            s[i] ^= D[i % 5];

        // ρ + π
        uint64_t t[25];
        for (int i = 0; i < 25; i++)
            t[PI[i]] = rotl64(s[i], ROT[i]);

        // χ
        for (int y = 0; y < 25; y += 5) {
            for (int x = 0; x < 5; x++)
                s[y + x] = t[y + x] ^ (~t[y + (x+1)%5] & t[y + (x+2)%5]);
        }

        // ι
        s[0] ^= RC[round];
    }

    memcpy(state, s, 200);
}

void keccak_f1600_neon(uint64_t state[25]) {
    // Work with local variables for all 25 lanes
    uint64_t s[25];
    memcpy(s, state, 200);

    for (int round = 0; round < 24; round++) {
        // === θ step ===
        // Column parities: C[x] = s[x] ^ s[x+5] ^ s[x+10] ^ s[x+15] ^ s[x+20]
        // Use NEON to XOR pairs
        uint64x2_t c01_a = vld1q_u64(&s[0]);   // s[0], s[1]
        uint64x2_t c01_b = vld1q_u64(&s[5]);   // s[5], s[6]
        uint64x2_t c01_c = vld1q_u64(&s[10]);  // s[10], s[11]
        uint64x2_t c01_d = vld1q_u64(&s[15]);  // s[15], s[16]
        uint64x2_t c01_e = vld1q_u64(&s[20]);  // s[20], s[21]
        uint64x2_t c01 = veorq_u64(veorq_u64(veorq_u64(c01_a, c01_b), veorq_u64(c01_c, c01_d)), c01_e);

        uint64x2_t c23_a = vld1q_u64(&s[2]);
        uint64x2_t c23_b = vld1q_u64(&s[7]);
        uint64x2_t c23_c = vld1q_u64(&s[12]);
        uint64x2_t c23_d = vld1q_u64(&s[17]);
        uint64x2_t c23_e = vld1q_u64(&s[22]);
        uint64x2_t c23 = veorq_u64(veorq_u64(veorq_u64(c23_a, c23_b), veorq_u64(c23_c, c23_d)), c23_e);

        uint64_t c4 = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

        uint64_t C[5];
        vst1q_u64(&C[0], c01);
        vst1q_u64(&C[2], c23);
        C[4] = c4;

        // D[x] = C[x-1] ^ rotl64(C[x+1], 1)
        uint64_t D[5];
        D[0] = C[4] ^ rotl64(C[1], 1);
        D[1] = C[0] ^ rotl64(C[2], 1);
        D[2] = C[1] ^ rotl64(C[3], 1);
        D[3] = C[2] ^ rotl64(C[4], 1);
        D[4] = C[3] ^ rotl64(C[0], 1);

        // XOR D into all lanes — vectorized in pairs
        uint64x2_t d01 = vld1q_u64(&D[0]);
        uint64x2_t d23 = vld1q_u64(&D[2]);

        for (int y = 0; y < 25; y += 5) {
            uint64x2_t v01 = veorq_u64(vld1q_u64(&s[y]), d01);
            vst1q_u64(&s[y], v01);
            uint64x2_t v23 = veorq_u64(vld1q_u64(&s[y + 2]), d23);
            vst1q_u64(&s[y + 2], v23);
            s[y + 4] ^= D[4];
        }

        // === ρ and π steps (combined) ===
        uint64_t t[25];
        for (int i = 0; i < 25; i++) {
            t[PI[i]] = rotl64(s[i], ROT[i]);
        }

        // === χ step ===
        // For each row of 5: s[x] = t[x] ^ (~t[x+1] & t[x+2])
        // Use vbicq_u64(a, b) = a & ~b, so ~t[x+1] & t[x+2] = vbicq(t[x+2], t[x+1])
        for (int y = 0; y < 25; y += 5) {
            // Process lanes 0,1 with NEON
            uint64x2_t t01 = vld1q_u64(&t[y]);
            uint64x2_t t12 = vld1q_u64(&t[y + 1]);
            uint64x2_t t23 = vld1q_u64(&t[y + 2]);
            // chi01 = t01 ^ vbic(t23, t12) = t[0]^(~t[1]&t[2]), t[1]^(~t[2]&t[3])
            uint64x2_t chi01 = veorq_u64(t01, vbicq_u64(t23, t12));
            vst1q_u64(&s[y], chi01);

            // Process lanes 2,3 with NEON
            uint64x2_t t34 = vld1q_u64(&t[y + 3]);
            // Need t[4],t[0] for lanes 2,3
            uint64_t t4_arr[2] = { t[y + 4], t[y] };
            uint64x2_t t40 = vld1q_u64(t4_arr);
            uint64x2_t chi23 = veorq_u64(t23, vbicq_u64(t40, t34));
            vst1q_u64(&s[y + 2], chi23);

            // Lane 4 scalar
            s[y + 4] = t[y + 4] ^ (~t[y] & t[y + 1]);
        }

        // === ι step ===
        s[0] ^= RC[round];
    }

    memcpy(state, s, 200);
}

void keccak256_hash_neon(const uint8_t *input, size_t len, uint8_t output[32]) {
    uint64_t state[25];
    memset(state, 0, 200);

    const int rate = 136; // bytes
    size_t offset = 0;

    // Absorb full blocks
    while (offset + rate <= len) {
        const uint64_t *block = (const uint64_t *)(input + offset);
        // rate / 8 = 17 lanes to XOR
        // Vectorized XOR in pairs (8 pairs = 16 lanes) + 1 scalar
        for (int i = 0; i < 16; i += 2) {
            uint64x2_t sv = vld1q_u64(&state[i]);
            uint64x2_t bv = vld1q_u64(&block[i]);
            vst1q_u64(&state[i], veorq_u64(sv, bv));
        }
        state[16] ^= block[16];

        keccak_f1600_neon(state);
        offset += rate;
    }

    // Absorb remaining bytes + padding
    // Copy remaining into a zeroed rate-sized buffer
    uint8_t buf[136];
    memset(buf, 0, 136);
    size_t remaining = len - offset;
    if (remaining > 0) {
        memcpy(buf, input + offset, remaining);
    }

    // Keccak padding: 0x01 after data, 0x80 at end of rate block
    buf[remaining] = 0x01;
    buf[rate - 1] |= 0x80;

    // XOR padded block into state
    const uint64_t *padBlock = (const uint64_t *)buf;
    for (int i = 0; i < 16; i += 2) {
        uint64x2_t sv = vld1q_u64(&state[i]);
        uint64x2_t bv = vld1q_u64(&padBlock[i]);
        vst1q_u64(&state[i], veorq_u64(sv, bv));
    }
    state[16] ^= padBlock[16];

    keccak_f1600_neon(state);

    // Squeeze: first 32 bytes = first 4 lanes
    memcpy(output, state, 32);
}

void keccak256_hash_pair_neon(const uint8_t a[32], const uint8_t b[32], uint8_t output[32]) {
    // 64 bytes input < 136 byte rate, so single block + padding
    uint64_t state[25];
    memset(state, 0, 200);

    // XOR first 32 bytes (4 lanes) from a
    const uint64_t *a64 = (const uint64_t *)a;
    const uint64_t *b64 = (const uint64_t *)b;

    // Load a (lanes 0-3) and b (lanes 4-7) using NEON
    uint64x2_t a01 = vld1q_u64(&a64[0]);
    uint64x2_t a23 = vld1q_u64(&a64[2]);
    uint64x2_t b01 = vld1q_u64(&b64[0]);
    uint64x2_t b23 = vld1q_u64(&b64[2]);
    vst1q_u64(&state[0], a01);
    vst1q_u64(&state[2], a23);
    vst1q_u64(&state[4], b01);
    vst1q_u64(&state[6], b23);

    // Padding: byte 64 = 0x01, byte 135 = 0x80
    // Byte 64 is in lane 8 (64/8=8), byte offset 0
    state[8] = 0x01;
    // Byte 135 is in lane 16 (135/8=16), byte offset 7 → 0x80 << 56
    state[16] = 0x8000000000000000ULL;

    keccak_f1600_neon(state);

    // Output first 32 bytes
    memcpy(output, state, 32);
}

void keccak256_batch_hash_pairs_neon(const uint8_t *inputs, uint8_t *outputs, size_t n) {
    // inputs: n pairs of 32-byte values = n * 64 bytes
    // outputs: n * 32 bytes
    for (size_t i = 0; i < n; i++) {
        keccak256_hash_pair_neon(
            inputs + i * 64,
            inputs + i * 64 + 32,
            outputs + i * 32
        );
    }
}
