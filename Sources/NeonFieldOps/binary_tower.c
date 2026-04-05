// Binary tower field arithmetic using ARM64 PMULL (carry-less multiply) intrinsics
// GF(2) → GF(2^8) → GF(2^16) → GF(2^32) → GF(2^64) → GF(2^128)
//
// Tower construction: GF(2^{2k}) = GF(2^k)[X] / (X^2 + X + alpha_k)
// Addition = XOR at every level (free)
// Multiplication uses PMULL for 64-bit carry-less multiply, Karatsuba for 128-bit
//
// GF(2^128) reduction polynomial: x^128 + x^7 + x^2 + x + 1 (AES-GCM standard)

#include "NeonFieldOps.h"
#include <arm_neon.h>
#include <string.h>
#include <stdlib.h>

// ============================================================
// GF(2^8): Lookup table multiply with AES polynomial x^8+x^4+x^3+x+1 (0x11B)
// ============================================================

// Log/exp tables for GF(2^8) with generator 3
static uint8_t gf8_log_table[256];
static uint8_t gf8_exp_table[512];
static uint8_t gf8_inv_table[256];
static int gf8_tables_initialized = 0;

// Bit-serial GF(2^8) multiply (used to build tables)
static inline uint8_t gf8_mul_slow(uint8_t a, uint8_t b) {
    uint8_t result = 0;
    for (int i = 0; i < 8; i++) {
        if (b & 1) result ^= a;
        uint8_t carry = a >> 7;
        a = (a << 1) ^ (carry ? 0x1B : 0);  // reduce by x^8+x^4+x^3+x+1
        b >>= 1;
    }
    return result;
}

static void gf8_init_tables(void) {
    if (gf8_tables_initialized) return;

    // Build exp/log tables with generator 3
    uint8_t x = 1;
    for (int i = 0; i < 255; i++) {
        gf8_exp_table[i] = x;
        gf8_exp_table[i + 255] = x;  // wraparound
        gf8_log_table[x] = (uint8_t)i;
        x = gf8_mul_slow(x, 3);
    }
    gf8_log_table[0] = 0;  // convention

    // Build inverse table: inv[a] = a^254
    gf8_inv_table[0] = 0;
    for (int i = 1; i < 256; i++) {
        int log_val = gf8_log_table[i];
        // a^(-1) = a^254 = exp(254 * log(a) mod 255)
        int inv_log = (255 - log_val) % 255;
        gf8_inv_table[i] = gf8_exp_table[inv_log];
    }

    gf8_tables_initialized = 1;
}

// ============================================================
// Public GF(2^8) API
// ============================================================

void bt_gf8_init(void) {
    gf8_init_tables();
}

uint8_t bt_gf8_mul(uint8_t a, uint8_t b) {
    if (a == 0 || b == 0) return 0;
    int log_sum = (int)gf8_log_table[a] + (int)gf8_log_table[b];
    return gf8_exp_table[log_sum];  // table is doubled for wraparound
}

uint8_t bt_gf8_inv(uint8_t a) {
    return gf8_inv_table[a];
}

uint8_t bt_gf8_sqr(uint8_t a) {
    return bt_gf8_mul(a, a);
}

// ============================================================
// GF(2^64): Direct PMULL carry-less multiply
// ============================================================
//
// vmull_p64 computes 64x64 → 128-bit carry-less product.
// We then reduce by a chosen irreducible polynomial for GF(2^64).
//
// For the binary tower, GF(2^64) uses the polynomial:
//   x^64 + x^4 + x^3 + x + 1  (pentanomial)
// Reduction constant for bits [64..127]: reduce by XORing shifted copies.

// Carry-less multiply 64x64 → 128 bits (lo, hi)
static inline void clmul64(uint64_t a, uint64_t b, uint64_t *lo, uint64_t *hi) {
    // ARM NEON PMULL: polynomial multiply long
    poly64_t pa = (poly64_t)a;
    poly64_t pb = (poly64_t)b;
    poly128_t result = vmull_p64(pa, pb);
    // Extract lo/hi from poly128_t via vreinterpret
    uint64x2_t r = vreinterpretq_u64_p128(result);
    *lo = vgetq_lane_u64(r, 0);
    *hi = vgetq_lane_u64(r, 1);
}

// Reduce 128-bit carry-less product to GF(2^64)
// Irreducible: x^64 + x^4 + x^3 + x + 1
// For each bit i in [64..127], replace x^i with x^{i-64} * (x^4 + x^3 + x + 1)
// = XOR hi shifted by 0, then by (shift amounts for the reduction polynomial low bits)
static inline uint64_t gf64_reduce(uint64_t lo, uint64_t hi) {
    // Reduction: x^64 = x^4 + x^3 + x + 1
    // For the high 64 bits, we need:
    // lo ^= hi ^ (hi << 1) ^ (hi << 3) ^ (hi << 4)
    // But shifts can overflow — bits shifted out of hi<<k go to position 64+k
    // So we need a second round of reduction for those overflow bits.

    uint64_t t;

    // First round: fold hi into lo
    t = hi;
    lo ^= (t << 1) ^ (t << 3) ^ (t << 4) ^ t;
    // Overflow bits from shifts: bits that shifted past bit 63
    uint64_t overflow = (t >> 63) ^ (t >> 61) ^ (t >> 60);
    // Second round: reduce overflow (at most a few bits set, no further overflow)
    lo ^= overflow ^ (overflow << 1) ^ (overflow << 3) ^ (overflow << 4);

    return lo;
}

uint64_t bt_gf64_mul(uint64_t a, uint64_t b) {
    uint64_t lo, hi;
    clmul64(a, b, &lo, &hi);
    return gf64_reduce(lo, hi);
}

uint64_t bt_gf64_sqr(uint64_t a) {
    return bt_gf64_mul(a, a);
}

// Inverse via Itoh-Tsujii: a^(-1) = a^(2^64 - 2) in GF(2^64)
// Using addition chain for the exponent 2^64 - 2
uint64_t bt_gf64_inv(uint64_t a) {
    if (a == 0) return 0;

    // Compute a^(2^64-2) using repeated squaring chains
    // 2^64 - 2 = 2 * (2^63 - 1)
    // Strategy: compute a^(2^k - 1) for increasing k, then square once

    // r1 = a^(2^1 - 1) = a
    uint64_t r1 = a;
    // r2 = a^(2^2 - 1) = a^3
    uint64_t r2 = bt_gf64_mul(bt_gf64_sqr(r1), r1);
    // r4 = a^(2^4 - 1) = (r2^(2^2)) * r2
    uint64_t r4 = bt_gf64_mul(bt_gf64_sqr(bt_gf64_sqr(r2)), r2);
    // r8 = a^(2^8 - 1) = (r4^(2^4)) * r4
    uint64_t t = r4;
    for (int i = 0; i < 4; i++) t = bt_gf64_sqr(t);
    uint64_t r8 = bt_gf64_mul(t, r4);
    // r16 = a^(2^16 - 1)
    t = r8;
    for (int i = 0; i < 8; i++) t = bt_gf64_sqr(t);
    uint64_t r16 = bt_gf64_mul(t, r8);
    // r32 = a^(2^32 - 1)
    t = r16;
    for (int i = 0; i < 16; i++) t = bt_gf64_sqr(t);
    uint64_t r32 = bt_gf64_mul(t, r16);
    // r64 = a^(2^64 - 1)
    t = r32;
    for (int i = 0; i < 32; i++) t = bt_gf64_sqr(t);
    uint64_t r64 = bt_gf64_mul(t, r32);

    // a^(2^64 - 2) = a^(2^64 - 1) / a = r64 * a^(-1)... no, directly:
    // a^(2^64 - 2) = (a^(2^64 - 1)) * a^(-1)
    // But that's circular. Instead: a^(2^64-2) = a^(2*(2^63-1)) = (a^(2^63-1))^2
    // r63 = r64 * inv(a)... no.
    // Better: a^(2^64 - 2) = a^(2^64 - 1) * a^(-1)
    // Since a * a^(2^64-2) = a^(2^64-1) = 1 in GF(2^64)*
    // So we want r64 / a = r64 * a^(-1). But we're computing a^(-1)!
    // The trick: a^(2^64-2) = (a^(2^63-1))^2
    // r63 = a^(2^63-1): square r32 31 times then mul by r31...
    // Simpler approach: just compute r63 directly.

    // Actually: 2^64-2 in binary is 111...110 (63 ones followed by a zero)
    // So a^(2^64-2) = product of a^(2^i) for i=1..63
    // = (a^2) * (a^4) * ... * (a^(2^63))
    // = (a * a^2 * a^4 * ... * a^(2^62)) * a^(2^63) / a
    // Easier: a^(2^64-2) = (a^(2^63-1))^2
    // r63 = a^(2^63-1)
    // But building r63 needs r32, r31... Let's just do:
    // We have r32 = a^(2^32-1)
    // r63 = (r32)^(2^31) * a^(2^31-1)
    // r31 = (r16)^(2^15) * r15...
    // This is getting complicated. Let's just use a simpler approach.

    // Simple: a^(-1) = a^(2^64-2). We already have r64 = a^(2^64-1).
    // Since r64 = a^(2^64-1) and a * a^(2^64-2) = a^(2^64-1),
    // if a^(2^64-1) = 1 (which it is for nonzero a in GF(2^64)),
    // then a * a^(2^64-2) = 1, so a^(2^64-2) = a^(-1).
    // But we computed r64 by multiplying chains, so r64 should equal 1.
    // That doesn't directly give us a^(-1).

    // Correct Itoh-Tsujii approach:
    // a^(-1) = a^(2^64 - 2)
    // Build using: a^(2^k - 1) via the chain, then square the result of a^(2^63 - 1)
    // a^(2^63 - 1): need to build this

    // r32 = a^(2^32 - 1) [have this]
    // need r63: a^(2^63-1) = (a^(2^32-1))^(2^31) * a^(2^31-1)
    // need r31: a^(2^31-1) = (a^(2^16-1))^(2^15) * a^(2^15-1)
    // need r15: a^(2^15-1) = (a^(2^8-1))^(2^7) * a^(2^7-1)
    // need r7:  a^(2^7-1) = (a^(2^4-1))^(2^3) * a^(2^3-1)
    // need r3:  a^(2^3-1) = (a^(2^2-1))^(2^1) * a^(2^1-1) = r2^2 * r1 = a^6 * a = a^7
    // Wait, r3 = a^(2^3-1) = a^7. r2 = a^3. r2^(2^1) = a^(3*2) = a^6. a^6 * a = a^7. Yes.

    uint64_t r3 = bt_gf64_mul(bt_gf64_sqr(r2), r1);

    // r7 = (r4)^(2^3) * r3
    t = r4;
    for (int i = 0; i < 3; i++) t = bt_gf64_sqr(t);
    uint64_t r7 = bt_gf64_mul(t, r3);

    // r15 = (r8)^(2^7) * r7
    t = r8;
    for (int i = 0; i < 7; i++) t = bt_gf64_sqr(t);
    uint64_t r15 = bt_gf64_mul(t, r7);

    // r31 = (r16)^(2^15) * r15
    t = r16;
    for (int i = 0; i < 15; i++) t = bt_gf64_sqr(t);
    uint64_t r31 = bt_gf64_mul(t, r15);

    // r63 = (r32)^(2^31) * r31
    t = r32;
    for (int i = 0; i < 31; i++) t = bt_gf64_sqr(t);
    uint64_t r63 = bt_gf64_mul(t, r31);

    // a^(2^64-2) = r63^2
    return bt_gf64_sqr(r63);
}

// ============================================================
// GF(2^128): Karatsuba over GF(2^64) with PMULL
// ============================================================
//
// We use the standard polynomial representation with
// irreducible: x^128 + x^7 + x^2 + x + 1 (AES-GCM polynomial)
//
// An element is represented as (lo, hi) where value = lo + hi * x^64
// This is NOT the tower extension — this is the "flat" polynomial
// representation used by AES-GCM, which has better hardware support.
//
// Alternatively, for Binius we can use the tower form. We provide both.

// --- Flat GF(2^128) with AES-GCM polynomial ---

// 128-bit carry-less multiply using Karatsuba with 3 PMULL instructions
static inline void clmul128(uint64_t a_lo, uint64_t a_hi,
                              uint64_t b_lo, uint64_t b_hi,
                              uint64_t *r0, uint64_t *r1,
                              uint64_t *r2, uint64_t *r3) {
    uint64_t lo_lo, lo_hi, hi_lo, hi_hi, mid_lo, mid_hi;

    clmul64(a_lo, b_lo, &lo_lo, &lo_hi);
    clmul64(a_hi, b_hi, &hi_lo, &hi_hi);
    clmul64(a_lo ^ a_hi, b_lo ^ b_hi, &mid_lo, &mid_hi);

    // Karatsuba recombination:
    // result = lo + (mid + lo + hi) * x^64 + hi * x^128
    // where lo = a_lo*b_lo (128 bit), hi = a_hi*b_hi (128 bit)
    // mid = (a_lo+a_hi)*(b_lo+b_hi) (128 bit)
    uint64_t mid_adj_lo = mid_lo ^ lo_lo ^ hi_lo;
    uint64_t mid_adj_hi = mid_hi ^ lo_hi ^ hi_hi;

    *r0 = lo_lo;
    *r1 = lo_hi ^ mid_adj_lo;
    *r2 = hi_lo ^ mid_adj_hi;
    *r3 = hi_hi;
}

// Barrett reduction of 256-bit product to GF(2^128)
// Irreducible: x^128 + x^7 + x^2 + x + 1
// For bits in [128..255], reduce by XOR with shifted copies of the low part
static inline void gf128_reduce(uint64_t r0, uint64_t r1, uint64_t r2, uint64_t r3,
                                  uint64_t *out_lo, uint64_t *out_hi) {
    // Reduce r3 (bits [192..255])
    // x^128 = x^7 + x^2 + x + 1
    // For a bit at position 128+i, replace with bits at positions i+7, i+2, i+1, i
    // r3 represents bits [192..255], so bit j in r3 = bit (j+192) of the product
    // After reduction: goes to positions j+192-128+{0,1,2,7} = j+64+{0,1,2,7}
    // These land in r1 (bits [64..127]) and r2 (bits [128..191])

    // Fold r3 into r1/r2
    r1 ^= (r3 << 7) ^ (r3 << 2) ^ (r3 << 1) ^ r3;
    r2 ^= (r3 >> 57) ^ (r3 >> 62) ^ (r3 >> 63);
    // r3 is now consumed

    // Fold r2 (bits [128..191]) into r0/r1
    r0 ^= (r2 << 7) ^ (r2 << 2) ^ (r2 << 1) ^ r2;
    r1 ^= (r2 >> 57) ^ (r2 >> 62) ^ (r2 >> 63);
    // r2 is now consumed

    *out_lo = r0;
    *out_hi = r1;
}

void bt_gf128_mul(const uint64_t a[2], const uint64_t b[2], uint64_t r[2]) {
    uint64_t r0, r1, r2, r3;
    clmul128(a[0], a[1], b[0], b[1], &r0, &r1, &r2, &r3);
    gf128_reduce(r0, r1, r2, r3, &r[0], &r[1]);
}

void bt_gf128_sqr(const uint64_t a[2], uint64_t r[2]) {
    bt_gf128_mul(a, a, r);
}

void bt_gf128_add(const uint64_t a[2], const uint64_t b[2], uint64_t r[2]) {
    r[0] = a[0] ^ b[0];
    r[1] = a[1] ^ b[1];
}

// Inverse via Itoh-Tsujii in GF(2^128)
// a^(-1) = a^(2^128 - 2)
void bt_gf128_inv(const uint64_t a[2], uint64_t r[2]) {
    uint64_t t[2], r1[2], r2[2], r4[2], r8[2], r16[2], r32[2], r64[2];

    // r1 = a
    r1[0] = a[0]; r1[1] = a[1];

    // r2 = a^(2^2-1) = a^3 = a^2 * a
    bt_gf128_sqr(r1, t);
    bt_gf128_mul(t, r1, r2);

    // r4 = a^(2^4-1) = (r2)^(2^2) * r2
    memcpy(t, r2, 16);
    for (int i = 0; i < 2; i++) bt_gf128_sqr(t, t);
    bt_gf128_mul(t, r2, r4);

    // r8
    memcpy(t, r4, 16);
    for (int i = 0; i < 4; i++) bt_gf128_sqr(t, t);
    bt_gf128_mul(t, r4, r8);

    // r16
    memcpy(t, r8, 16);
    for (int i = 0; i < 8; i++) bt_gf128_sqr(t, t);
    bt_gf128_mul(t, r8, r16);

    // r32
    memcpy(t, r16, 16);
    for (int i = 0; i < 16; i++) bt_gf128_sqr(t, t);
    bt_gf128_mul(t, r16, r32);

    // r64
    memcpy(t, r32, 16);
    for (int i = 0; i < 32; i++) bt_gf128_sqr(t, t);
    bt_gf128_mul(t, r32, r64);

    // r127 = a^(2^127-1) = (r64)^(2^63) * r63
    // We need r63: build r3, r7, r15, r31, r63
    uint64_t r3[2], r7[2], r15[2], r31[2], r63[2];

    // r3
    bt_gf128_sqr(r2, t);
    bt_gf128_mul(t, r1, r3);

    // r7 = (r4)^(2^3) * r3
    memcpy(t, r4, 16);
    for (int i = 0; i < 3; i++) bt_gf128_sqr(t, t);
    bt_gf128_mul(t, r3, r7);

    // r15
    memcpy(t, r8, 16);
    for (int i = 0; i < 7; i++) bt_gf128_sqr(t, t);
    bt_gf128_mul(t, r7, r15);

    // r31
    memcpy(t, r16, 16);
    for (int i = 0; i < 15; i++) bt_gf128_sqr(t, t);
    bt_gf128_mul(t, r15, r31);

    // r63
    memcpy(t, r32, 16);
    for (int i = 0; i < 31; i++) bt_gf128_sqr(t, t);
    bt_gf128_mul(t, r31, r63);

    // r127
    uint64_t r127[2];
    memcpy(t, r64, 16);
    for (int i = 0; i < 63; i++) bt_gf128_sqr(t, t);
    bt_gf128_mul(t, r63, r127);

    // a^(2^128-2) = (r127)^2
    bt_gf128_sqr(r127, r);
}

// ============================================================
// Batch operations for vectorized processing
// ============================================================

void bt_gf64_batch_mul(const uint64_t *a, const uint64_t *b, uint64_t *out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = bt_gf64_mul(a[i], b[i]);
    }
}

void bt_gf64_batch_add(const uint64_t *a, const uint64_t *b, uint64_t *out, int n) {
    // Process 2 elements at a time using NEON XOR
    int i = 0;
    for (; i + 1 < n; i += 2) {
        uint64x2_t va = vld1q_u64(a + i);
        uint64x2_t vb = vld1q_u64(b + i);
        vst1q_u64(out + i, veorq_u64(va, vb));
    }
    for (; i < n; i++) {
        out[i] = a[i] ^ b[i];
    }
}

void bt_gf64_batch_sqr(const uint64_t *a, uint64_t *out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = bt_gf64_sqr(a[i]);
    }
}

void bt_gf128_batch_mul(const uint64_t *a, const uint64_t *b, uint64_t *out, int n) {
    for (int i = 0; i < n; i++) {
        bt_gf128_mul(a + i * 2, b + i * 2, out + i * 2);
    }
}

void bt_gf128_batch_add(const uint64_t *a, const uint64_t *b, uint64_t *out, int n) {
    // Each element is 2 x uint64_t = 128 bits, XOR with NEON
    for (int i = 0; i < n; i++) {
        uint64x2_t va = vld1q_u64(a + i * 2);
        uint64x2_t vb = vld1q_u64(b + i * 2);
        vst1q_u64(out + i * 2, veorq_u64(va, vb));
    }
}

// ============================================================
// Tower-form operations: GF(2^128) as tower over GF(2^64)
// ============================================================
//
// GF(2^128) = GF(2^64)[X] / (X^2 + X + delta)
// Element (lo, hi) represents lo + hi*X
// Multiply: (a0 + a1*X)(b0 + b1*X) mod (X^2+X+delta)
//   = (a0*b0 + a1*b1*delta) + ((a0+a1)*(b0+b1) + a0*b0)*X
// Uses 3 GF(2^64) multiplications (Karatsuba)

// Tower delta constant — must be chosen so X^2+X+delta is irreducible over GF(2^64)
// Using delta = 2 (same as Swift BinaryField128.DELTA)
#define BT_TOWER_DELTA 0x0000000000000002ULL

void bt_tower128_mul(const uint64_t a[2], const uint64_t b[2], uint64_t r[2]) {
    uint64_t a0b0 = bt_gf64_mul(a[0], b[0]);
    uint64_t a1b1 = bt_gf64_mul(a[1], b[1]);
    uint64_t cross = bt_gf64_mul(a[0] ^ a[1], b[0] ^ b[1]);
    uint64_t a1b1_delta = bt_gf64_mul(a1b1, BT_TOWER_DELTA);

    r[0] = a0b0 ^ a1b1_delta;
    r[1] = cross ^ a0b0;
}

void bt_tower128_sqr(const uint64_t a[2], uint64_t r[2]) {
    bt_tower128_mul(a, a, r);
}

void bt_tower128_add(const uint64_t a[2], const uint64_t b[2], uint64_t r[2]) {
    r[0] = a[0] ^ b[0];
    r[1] = a[1] ^ b[1];
}

void bt_tower128_inv(const uint64_t a[2], uint64_t r[2]) {
    // Norm N = a0^2 + a0*a1 + a1^2*delta
    uint64_t a0sq = bt_gf64_sqr(a[0]);
    uint64_t a0a1 = bt_gf64_mul(a[0], a[1]);
    uint64_t a1sq_delta = bt_gf64_mul(bt_gf64_sqr(a[1]), BT_TOWER_DELTA);
    uint64_t norm = a0sq ^ a0a1 ^ a1sq_delta;
    uint64_t norm_inv = bt_gf64_inv(norm);

    r[0] = bt_gf64_mul(a[0] ^ a[1], norm_inv);
    r[1] = bt_gf64_mul(a[1], norm_inv);
}

// Batch tower128 multiply
void bt_tower128_batch_mul(const uint64_t *a, const uint64_t *b, uint64_t *out, int n) {
    for (int i = 0; i < n; i++) {
        bt_tower128_mul(a + i * 2, b + i * 2, out + i * 2);
    }
}

// ============================================================
// GF(2^32) / GF(2^16) accelerated via PMULL (promote to 64-bit clmul)
// ============================================================

// GF(2^32) multiply: use PMULL on 32-bit values (zero-extended to 64-bit)
// Irreducible: x^32 + x^7 + x^3 + x^2 + 1 (CRC-32C polynomial)
uint32_t bt_gf32_mul(uint32_t a, uint32_t b) {
    uint64_t lo, hi;
    clmul64((uint64_t)a, (uint64_t)b, &lo, &hi);
    // Product is at most 62 bits (32+32-2), so hi has at most 0 or very few bits
    // Actually: 32x32 clmul gives a 63-bit result (bits 0..62)
    // hi contains bits [64..62]... no. clmul64 of 32-bit values:
    // max degree is 31+31=62, so result fits in 63 bits → lo only (hi has bits [64..])
    // Actually lo has bits [0..63], hi has bits [64..127].
    // For 32x32, max result bit is 62, so it fits entirely in lo.

    // Reduce by x^32 + x^7 + x^3 + x^2 + 1
    // The result is in lo, bits [0..62]. Need to reduce bits [32..62].
    uint64_t r = lo;
    // Fold high bits: for bit i >= 32, replace x^i with x^{i-32} * (x^7 + x^3 + x^2 + 1)
    uint64_t high = r >> 32;
    r ^= (high << 7) ^ (high << 3) ^ (high << 2) ^ high;
    // After folding, new bits might appear at [32..38], fold again
    high = r >> 32;
    r ^= (high << 7) ^ (high << 3) ^ (high << 2) ^ high;
    // One more round to be safe (bits could cascade)
    high = r >> 32;
    r ^= (high << 7) ^ (high << 3) ^ (high << 2) ^ high;

    return (uint32_t)(r & 0xFFFFFFFF);
}

uint32_t bt_gf32_sqr(uint32_t a) {
    return bt_gf32_mul(a, a);
}

// GF(2^16) multiply via PMULL
uint16_t bt_gf16_mul(uint16_t a, uint16_t b) {
    uint64_t lo, hi;
    clmul64((uint64_t)a, (uint64_t)b, &lo, &hi);
    // 16x16 clmul: max bit 30, fits in lo

    // Reduce by x^16 + x^5 + x^3 + x + 1 (standard irreducible for GF(2^16))
    uint64_t r = lo;
    uint64_t high = r >> 16;
    r ^= (high << 5) ^ (high << 3) ^ (high << 1) ^ high;
    high = r >> 16;
    r ^= (high << 5) ^ (high << 3) ^ (high << 1) ^ high;

    return (uint16_t)(r & 0xFFFF);
}

uint16_t bt_gf16_sqr(uint16_t a) {
    return bt_gf16_mul(a, a);
}

// GF(2^16) inverse via Itoh-Tsujii: a^(-1) = a^(2^16-2)
uint16_t bt_gf16_inv(uint16_t a) {
    if (a == 0) return 0;

    // Build a^(2^k - 1) for k = 1,2,4,8,15
    // r1 = a
    uint16_t r1 = a;
    // r2 = a^(2^2-1) = a^2 * a = a^3
    uint16_t r2 = bt_gf16_mul(bt_gf16_sqr(r1), r1);
    // r4 = a^(2^4-1) = (r2^(2^2)) * r2
    uint16_t t = r2;
    for (int i = 0; i < 2; i++) t = bt_gf16_sqr(t);
    uint16_t r4 = bt_gf16_mul(t, r2);
    // r8 = a^(2^8-1)
    t = r4;
    for (int i = 0; i < 4; i++) t = bt_gf16_sqr(t);
    uint16_t r8 = bt_gf16_mul(t, r4);
    // r3 = a^(2^3-1) = (r2^2) * r1
    uint16_t r3 = bt_gf16_mul(bt_gf16_sqr(r2), r1);
    // r7 = (r4^(2^3)) * r3
    t = r4;
    for (int i = 0; i < 3; i++) t = bt_gf16_sqr(t);
    uint16_t r7 = bt_gf16_mul(t, r3);
    // r15 = (r8^(2^7)) * r7
    t = r8;
    for (int i = 0; i < 7; i++) t = bt_gf16_sqr(t);
    uint16_t r15 = bt_gf16_mul(t, r7);
    // a^(2^16-2) = r15^2
    return bt_gf16_sqr(r15);
}

// GF(2^32) inverse via Itoh-Tsujii: a^(-1) = a^(2^32-2)
uint32_t bt_gf32_inv(uint32_t a) {
    if (a == 0) return 0;

    uint32_t r1 = a;
    uint32_t r2 = bt_gf32_mul(bt_gf32_sqr(r1), r1);
    uint32_t t = r2;
    for (int i = 0; i < 2; i++) t = bt_gf32_sqr(t);
    uint32_t r4 = bt_gf32_mul(t, r2);
    t = r4;
    for (int i = 0; i < 4; i++) t = bt_gf32_sqr(t);
    uint32_t r8 = bt_gf32_mul(t, r4);
    t = r8;
    for (int i = 0; i < 8; i++) t = bt_gf32_sqr(t);
    uint32_t r16 = bt_gf32_mul(t, r8);
    // Build r3, r7, r15, r31
    uint32_t r3 = bt_gf32_mul(bt_gf32_sqr(r2), r1);
    t = r4;
    for (int i = 0; i < 3; i++) t = bt_gf32_sqr(t);
    uint32_t r7 = bt_gf32_mul(t, r3);
    t = r8;
    for (int i = 0; i < 7; i++) t = bt_gf32_sqr(t);
    uint32_t r15 = bt_gf32_mul(t, r7);
    t = r16;
    for (int i = 0; i < 15; i++) t = bt_gf32_sqr(t);
    uint32_t r31 = bt_gf32_mul(t, r15);
    // a^(2^32-2) = r31^2
    return bt_gf32_sqr(r31);
}
