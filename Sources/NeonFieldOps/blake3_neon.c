// Blake3 NEON-optimized compression for ARM64
// Uses uint32x4_t for row-wise vectorization of the 4x4 state matrix.
// Column rounds operate element-wise; diagonal rounds use vextq_u32 rotation.

#include <arm_neon.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

// Blake3 IV (same as Blake2s)
static const uint32_t BLAKE3_IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};

// Message permutation
static const uint8_t MSG_PERM[16] = {
    2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8
};

// Flags
#define BLAKE3_PARENT 4

// Rotate right by 16: swap high/low 16-bit halves
static inline uint32x4_t rotr16(uint32x4_t x) {
    return vreinterpretq_u32_u16(vrev32q_u16(vreinterpretq_u16_u32(x)));
}

// Rotate right by 12
static inline uint32x4_t rotr12(uint32x4_t x) {
    return vsriq_n_u32(vshlq_n_u32(x, 20), x, 12);
}

// Rotate right by 8
static inline uint32x4_t rotr8(uint32x4_t x) {
    // Use TBL for byte-level shuffle: rotate each 32-bit lane right by 1 byte
    static const uint8_t tbl[16] = {
        1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12
    };
    uint8x16_t idx = vld1q_u8(tbl);
    return vreinterpretq_u32_u8(vqtbl1q_u8(vreinterpretq_u8_u32(x), idx));
}

// Rotate right by 7
static inline uint32x4_t rotr7(uint32x4_t x) {
    return vsriq_n_u32(vshlq_n_u32(x, 25), x, 7);
}

// G function on all 4 columns simultaneously
// Operates on row vectors: row0=a, row1=b, row2=c, row3=d
static inline void g_cols(uint32x4_t *row0, uint32x4_t *row1,
                          uint32x4_t *row2, uint32x4_t *row3,
                          uint32x4_t mx, uint32x4_t my) {
    *row0 = vaddq_u32(vaddq_u32(*row0, *row1), mx);
    *row3 = rotr16(veorq_u32(*row3, *row0));
    *row2 = vaddq_u32(*row2, *row3);
    *row1 = rotr12(veorq_u32(*row1, *row2));
    *row0 = vaddq_u32(vaddq_u32(*row0, *row1), my);
    *row3 = rotr8(veorq_u32(*row3, *row0));
    *row2 = vaddq_u32(*row2, *row3);
    *row1 = rotr7(veorq_u32(*row1, *row2));
}

// Blake3 compress: 7 rounds of column + diagonal mixing
// cv: 8-word chaining value
// msg: 16-word message block
// counter: 64-bit counter (lo, hi)
// block_len: number of bytes in the block
// flags: Blake3 flags
// out: 16-word output state (first 8 words are the hash)
static void blake3_compress_neon(const uint32_t cv[8], uint32_t msg[16],
                                 uint32_t counter_lo, uint32_t counter_hi,
                                 uint32_t block_len, uint32_t flags,
                                 uint32_t out[16]) {
    // Load state rows
    uint32x4_t row0 = vld1q_u32(&cv[0]);
    uint32x4_t row1 = vld1q_u32(&cv[4]);
    uint32x4_t row2 = vld1q_u32(BLAKE3_IV);
    uint32_t row3_arr[4] = { counter_lo, counter_hi, block_len, flags };
    uint32x4_t row3 = vld1q_u32(row3_arr);

    for (int round = 0; round < 7; round++) {
        // Load message words for column round
        // Columns: G(0,4,8,12,m0,m1), G(1,5,9,13,m2,m3), G(2,6,10,14,m4,m5), G(3,7,11,15,m6,m7)
        uint32_t mx_arr[4] = { msg[0], msg[2], msg[4], msg[6] };
        uint32_t my_arr[4] = { msg[1], msg[3], msg[5], msg[7] };
        uint32x4_t mx = vld1q_u32(mx_arr);
        uint32x4_t my = vld1q_u32(my_arr);

        // Column round
        g_cols(&row0, &row1, &row2, &row3, mx, my);

        // Diagonal round: rotate rows for diagonal access
        // row1 shifts left by 1: (1,2,3,0) -> maps col 0->1, 1->2, 2->3, 3->0
        row1 = vextq_u32(row1, row1, 1);
        row2 = vextq_u32(row2, row2, 2);
        row3 = vextq_u32(row3, row3, 3);

        // Diagonals: G(0,5,10,15,m8,m9), G(1,6,11,12,m10,m11), G(2,7,8,13,m12,m13), G(3,4,9,14,m14,m15)
        uint32_t dx_arr[4] = { msg[8], msg[10], msg[12], msg[14] };
        uint32_t dy_arr[4] = { msg[9], msg[11], msg[13], msg[15] };
        mx = vld1q_u32(dx_arr);
        my = vld1q_u32(dy_arr);

        g_cols(&row0, &row1, &row2, &row3, mx, my);

        // Un-rotate rows
        row1 = vextq_u32(row1, row1, 3);
        row2 = vextq_u32(row2, row2, 2);
        row3 = vextq_u32(row3, row3, 1);

        // Permute message for next round (skip after last round)
        if (round < 6) {
            uint32_t tmp[16];
            for (int i = 0; i < 16; i++) {
                tmp[i] = msg[MSG_PERM[i]];
            }
            memcpy(msg, tmp, 64);
        }
    }

    // Finalize: XOR top and bottom halves
    uint32x4_t cv0 = vld1q_u32(&cv[0]);
    uint32x4_t cv1 = vld1q_u32(&cv[4]);
    row0 = veorq_u32(row0, row2);
    row1 = veorq_u32(row1, row3);
    row2 = veorq_u32(row2, cv0);
    row3 = veorq_u32(row3, cv1);

    vst1q_u32(&out[0], row0);
    vst1q_u32(&out[4], row1);
    vst1q_u32(&out[8], row2);
    vst1q_u32(&out[12], row3);
}

// Hash a single parent node: left(32B) || right(32B) -> output(32B)
void blake3_hash_pair_neon(const uint8_t left[32], const uint8_t right[32],
                           uint8_t output[32]) {
    // Build 16-word message from left || right
    uint32_t msg[16];
    memcpy(&msg[0], left, 32);
    memcpy(&msg[8], right, 32);

    uint32_t state[16];
    blake3_compress_neon(BLAKE3_IV, msg, 0, 0, 64, BLAKE3_PARENT, state);

    // Output first 8 words = 32 bytes
    memcpy(output, state, 32);
}

// Batch hash n parent pairs
// inputs: n * 64 bytes (n pairs of left||right, each 32 bytes)
// outputs: n * 32 bytes
void blake3_batch_hash_pairs_neon(const uint8_t *inputs, uint8_t *outputs, size_t n) {
    for (size_t i = 0; i < n; i++) {
        blake3_hash_pair_neon(inputs + i * 64, inputs + i * 64 + 32,
                              outputs + i * 32);
    }
}
