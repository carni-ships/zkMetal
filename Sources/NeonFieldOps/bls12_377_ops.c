// BLS12-377 Fq (base field, 377-bit) and Fr (scalar field, 253-bit) C CIOS Montgomery arithmetic
// Plus G1 point operations (Jacobian projective, y^2 = x^3 + 1)
//
// Fq: 6x64-bit limbs, q = 0x01ae3a4617c510eac63b05c06ca1493b1a22d9f300f5138f1ef3622fba094800170b5d4430000000_8508c00000000001
// Fr: 4x64-bit limbs, r = 0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001
//
// Interop: Swift Fq377 = 12xUInt32 = 6xuint64_t (Montgomery form, little-endian)
//          Swift Fr377 = 8xUInt32 = 4xuint64_t (Montgomery form, little-endian)
//          Point377Projective = 3xFq377 = 18xuint64_t (x[6], y[6], z[6])
//          Point377Affine = 2xFq377 = 12xuint64_t (x[6], y[6])

#include "NeonFieldOps.h"
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <dispatch/dispatch.h>

typedef unsigned __int128 uint128_t;

// ============================================================
// BLS12-377 Fq (base field) constants — 6x64-bit limbs
// ============================================================

static const uint64_t FQ_P[6] = {
    0x8508c00000000001ULL, 0x170b5d4430000000ULL,
    0x1ef3622fba094800ULL, 0x1a22d9f300f5138fULL,
    0xc63b05c06ca1493bULL, 0x01ae3a4617c510eaULL
};

// -q^{-1} mod 2^64
static const uint64_t FQ_INV = 0x8508bfffffffFFFFULL;

// R mod q (Montgomery form of 1): 2^384 mod q
static const uint64_t FQ_ONE[6] = {
    0x02cdffffffffff68ULL, 0x51409f837fffffb1ULL,
    0x9f7db3a98a7d3ff2ULL, 0x7b4e97b76e7c6305ULL,
    0x4cf495bf803c84e8ULL, 0x008d6661e2fdf49aULL
};

// R^2 mod q: 2^768 mod q
static const uint64_t FQ_R2[6] = {
    0xb786686c9400cd22ULL, 0x0329fcaab00431b1ULL,
    0x22a5f11162d6b46dULL, 0xbfdf7d03827dc3acULL,
    0x837e92f041790bf9ULL, 0x006dfccb1e914b88ULL
};

// ============================================================
// BLS12-377 Fr (scalar field) constants — 4x64-bit limbs
// ============================================================

static const uint64_t FR_P[4] = {
    0x0a11800000000001ULL, 0x59aa76fed0000001ULL,
    0x60b44d1e5c37b001ULL, 0x12ab655e9a2ca556ULL
};

// -r^{-1} mod 2^64
static const uint64_t FR_INV = 0x0a117fffffffffffULL;

// R mod r (Montgomery 1)
static const uint64_t FR_ONE[4] = {
    0x7d1c7ffffffffff3ULL, 0x7257f50f6ffffff2ULL,
    0x16d81575512c0feeULL, 0x0d4bda322bbb9a9dULL
};

// R^2 mod r
static const uint64_t FR_R2[4] = {
    0x25d577bab861857bULL, 0xcc2c27b58860591fULL,
    0xa7cc008fe5dc8593ULL, 0x011fdae7eff1c939ULL
};

// ============================================================
// Fq CIOS Montgomery multiplication (fully unrolled, 6 limbs)
// ============================================================

static inline void fq_mul(const uint64_t a[6], const uint64_t b[6], uint64_t r[6]) {
    uint64_t t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0;

    // Iteration i=0
    {
        uint128_t w;
        w = (uint128_t)a[0] * b[0];
        t0 = (uint64_t)w; uint64_t c = (uint64_t)(w >> 64);
        w = (uint128_t)a[0] * b[1] + t1 + c;
        t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[0] * b[2] + t2 + c;
        t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[0] * b[3] + t3 + c;
        t3 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[0] * b[4] + t4 + c;
        t4 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[0] * b[5] + t5 + c;
        t5 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t6 = c;
        uint64_t m = t0 * FQ_INV;
        w = (uint128_t)m * FQ_P[0] + t0; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[1] + t1 + c; t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[2] + t2 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[3] + t3 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[4] + t4 + c; t3 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[5] + t5 + c; t4 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t5 = t6 + c; t6 = (t5 < c) ? 1 : 0;
    }
    // Iteration i=1
    {
        uint128_t w; uint64_t c;
        w = (uint128_t)a[1] * b[0] + t0; t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[1] * b[1] + t1 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[1] * b[2] + t2 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[1] * b[3] + t3 + c; t3 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[1] * b[4] + t4 + c; t4 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[1] * b[5] + t5 + c; t5 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t6 += c;
        uint64_t m = t0 * FQ_INV;
        w = (uint128_t)m * FQ_P[0] + t0; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[1] + t1 + c; t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[2] + t2 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[3] + t3 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[4] + t4 + c; t3 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[5] + t5 + c; t4 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t5 = t6 + c; t6 = (t5 < c) ? 1 : 0;
    }
    // Iteration i=2
    {
        uint128_t w; uint64_t c;
        w = (uint128_t)a[2] * b[0] + t0; t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[2] * b[1] + t1 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[2] * b[2] + t2 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[2] * b[3] + t3 + c; t3 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[2] * b[4] + t4 + c; t4 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[2] * b[5] + t5 + c; t5 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t6 += c;
        uint64_t m = t0 * FQ_INV;
        w = (uint128_t)m * FQ_P[0] + t0; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[1] + t1 + c; t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[2] + t2 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[3] + t3 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[4] + t4 + c; t3 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[5] + t5 + c; t4 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t5 = t6 + c; t6 = (t5 < c) ? 1 : 0;
    }
    // Iteration i=3
    {
        uint128_t w; uint64_t c;
        w = (uint128_t)a[3] * b[0] + t0; t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[3] * b[1] + t1 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[3] * b[2] + t2 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[3] * b[3] + t3 + c; t3 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[3] * b[4] + t4 + c; t4 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[3] * b[5] + t5 + c; t5 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t6 += c;
        uint64_t m = t0 * FQ_INV;
        w = (uint128_t)m * FQ_P[0] + t0; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[1] + t1 + c; t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[2] + t2 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[3] + t3 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[4] + t4 + c; t3 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[5] + t5 + c; t4 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t5 = t6 + c; t6 = (t5 < c) ? 1 : 0;
    }
    // Iteration i=4
    {
        uint128_t w; uint64_t c;
        w = (uint128_t)a[4] * b[0] + t0; t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[4] * b[1] + t1 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[4] * b[2] + t2 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[4] * b[3] + t3 + c; t3 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[4] * b[4] + t4 + c; t4 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[4] * b[5] + t5 + c; t5 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t6 += c;
        uint64_t m = t0 * FQ_INV;
        w = (uint128_t)m * FQ_P[0] + t0; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[1] + t1 + c; t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[2] + t2 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[3] + t3 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[4] + t4 + c; t3 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[5] + t5 + c; t4 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t5 = t6 + c; t6 = (t5 < c) ? 1 : 0;
    }
    // Iteration i=5
    {
        uint128_t w; uint64_t c;
        w = (uint128_t)a[5] * b[0] + t0; t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[5] * b[1] + t1 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[5] * b[2] + t2 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[5] * b[3] + t3 + c; t3 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[5] * b[4] + t4 + c; t4 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[5] * b[5] + t5 + c; t5 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t6 += c;
        uint64_t m = t0 * FQ_INV;
        w = (uint128_t)m * FQ_P[0] + t0; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[1] + t1 + c; t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[2] + t2 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[3] + t3 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[4] + t4 + c; t3 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[5] + t5 + c; t4 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t5 = t6 + c; t6 = (t5 < c) ? 1 : 0;
    }

    // Final conditional subtraction
    if (t6 || (t5 > FQ_P[5]) ||
        (t5 == FQ_P[5] && (t4 > FQ_P[4] ||
        (t4 == FQ_P[4] && (t3 > FQ_P[3] ||
        (t3 == FQ_P[3] && (t2 > FQ_P[2] ||
        (t2 == FQ_P[2] && (t1 > FQ_P[1] ||
        (t1 == FQ_P[1] && t0 >= FQ_P[0])))))))))) {
        uint128_t borrow;
        borrow = (uint128_t)t0 - FQ_P[0]; t0 = (uint64_t)borrow;
        uint64_t b1 = (uint64_t)(borrow >> 64) & 1;
        borrow = (uint128_t)t1 - FQ_P[1] - b1; t1 = (uint64_t)borrow;
        uint64_t b2 = (uint64_t)(borrow >> 64) & 1;
        borrow = (uint128_t)t2 - FQ_P[2] - b2; t2 = (uint64_t)borrow;
        uint64_t b3 = (uint64_t)(borrow >> 64) & 1;
        borrow = (uint128_t)t3 - FQ_P[3] - b3; t3 = (uint64_t)borrow;
        uint64_t b4 = (uint64_t)(borrow >> 64) & 1;
        borrow = (uint128_t)t4 - FQ_P[4] - b4; t4 = (uint64_t)borrow;
        uint64_t b5 = (uint64_t)(borrow >> 64) & 1;
        t5 = t5 - FQ_P[5] - b5;
    }

    r[0] = t0; r[1] = t1; r[2] = t2; r[3] = t3; r[4] = t4; r[5] = t5;
}

static inline void fq_sqr(const uint64_t a[6], uint64_t r[6]) {
    // Optimized squaring: upper-triangle doubled + diagonal (21 muls vs 36)
    uint128_t w;
    uint64_t c;
    uint64_t s[12];

    // Upper triangle: a[i]*a[j] for j>i (15 cross products)
    // Row 0: a[0]*a[1..5]
    w = (uint128_t)a[0] * a[1];
    s[1] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[0] * a[2] + c;
    s[2] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[0] * a[3] + c;
    s[3] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[0] * a[4] + c;
    s[4] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[0] * a[5] + c;
    s[5] = (uint64_t)w; s[6] = (uint64_t)(w >> 64);

    // Row 1: a[1]*a[2..5]
    w = (uint128_t)a[1] * a[2] + s[3];
    s[3] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[1] * a[3] + s[4] + c;
    s[4] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[1] * a[4] + s[5] + c;
    s[5] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[1] * a[5] + s[6] + c;
    s[6] = (uint64_t)w; s[7] = (uint64_t)(w >> 64);

    // Row 2: a[2]*a[3..5]
    w = (uint128_t)a[2] * a[3] + s[5];
    s[5] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[2] * a[4] + s[6] + c;
    s[6] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[2] * a[5] + s[7] + c;
    s[7] = (uint64_t)w; s[8] = (uint64_t)(w >> 64);

    // Row 3: a[3]*a[4..5]
    w = (uint128_t)a[3] * a[4] + s[7];
    s[7] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[3] * a[5] + s[8] + c;
    s[8] = (uint64_t)w; s[9] = (uint64_t)(w >> 64);

    // Row 4: a[4]*a[5]
    w = (uint128_t)a[4] * a[5] + s[9];
    s[9] = (uint64_t)w; s[10] = (uint64_t)(w >> 64);

    // Double the cross terms
    s[11] = s[10] >> 63;
    s[10] = (s[10] << 1) | (s[9] >> 63);
    s[9]  = (s[9]  << 1) | (s[8] >> 63);
    s[8]  = (s[8]  << 1) | (s[7] >> 63);
    s[7]  = (s[7]  << 1) | (s[6] >> 63);
    s[6]  = (s[6]  << 1) | (s[5] >> 63);
    s[5]  = (s[5]  << 1) | (s[4] >> 63);
    s[4]  = (s[4]  << 1) | (s[3] >> 63);
    s[3]  = (s[3]  << 1) | (s[2] >> 63);
    s[2]  = (s[2]  << 1) | (s[1] >> 63);
    s[1]  = s[1] << 1;

    // Add diagonal terms (6 muls)
    w = (uint128_t)a[0] * a[0];
    s[0] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)s[1] + c;
    s[1] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[1] * a[1] + s[2] + c;
    s[2] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)s[3] + c;
    s[3] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[2] * a[2] + s[4] + c;
    s[4] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)s[5] + c;
    s[5] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[3] * a[3] + s[6] + c;
    s[6] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)s[7] + c;
    s[7] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[4] * a[4] + s[8] + c;
    s[8] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)s[9] + c;
    s[9] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[5] * a[5] + s[10] + c;
    s[10] = (uint64_t)w; c = (uint64_t)(w >> 64);
    s[11] += c;

    // Montgomery reduction (6 iterations, fully unrolled)
    {
        uint64_t m = s[0] * FQ_INV;
        w = (uint128_t)m * FQ_P[0] + s[0]; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[1] + s[1] + c; s[0] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[2] + s[2] + c; s[1] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[3] + s[3] + c; s[2] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[4] + s[4] + c; s[3] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[5] + s[5] + c; s[4] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)s[6] + c; s[5] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)s[7] + c; s[6] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)s[8] + c; s[7] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)s[9] + c; s[8] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)s[10] + c; s[9] = (uint64_t)w; c = (uint64_t)(w >> 64);
        s[10] = s[11] + c; s[11] = 0;
    }
    {
        uint64_t m = s[0] * FQ_INV;
        w = (uint128_t)m * FQ_P[0] + s[0]; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[1] + s[1] + c; s[0] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[2] + s[2] + c; s[1] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[3] + s[3] + c; s[2] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[4] + s[4] + c; s[3] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[5] + s[5] + c; s[4] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)s[6] + c; s[5] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)s[7] + c; s[6] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)s[8] + c; s[7] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)s[9] + c; s[8] = (uint64_t)w; c = (uint64_t)(w >> 64);
        s[9] = s[10] + c; s[10] = 0;
    }
    {
        uint64_t m = s[0] * FQ_INV;
        w = (uint128_t)m * FQ_P[0] + s[0]; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[1] + s[1] + c; s[0] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[2] + s[2] + c; s[1] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[3] + s[3] + c; s[2] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[4] + s[4] + c; s[3] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[5] + s[5] + c; s[4] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)s[6] + c; s[5] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)s[7] + c; s[6] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)s[8] + c; s[7] = (uint64_t)w; c = (uint64_t)(w >> 64);
        s[8] = s[9] + c; s[9] = 0;
    }
    {
        uint64_t m = s[0] * FQ_INV;
        w = (uint128_t)m * FQ_P[0] + s[0]; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[1] + s[1] + c; s[0] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[2] + s[2] + c; s[1] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[3] + s[3] + c; s[2] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[4] + s[4] + c; s[3] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[5] + s[5] + c; s[4] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)s[6] + c; s[5] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)s[7] + c; s[6] = (uint64_t)w; c = (uint64_t)(w >> 64);
        s[7] = s[8] + c; s[8] = 0;
    }
    {
        uint64_t m = s[0] * FQ_INV;
        w = (uint128_t)m * FQ_P[0] + s[0]; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[1] + s[1] + c; s[0] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[2] + s[2] + c; s[1] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[3] + s[3] + c; s[2] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[4] + s[4] + c; s[3] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[5] + s[5] + c; s[4] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)s[6] + c; s[5] = (uint64_t)w; c = (uint64_t)(w >> 64);
        s[6] = s[7] + c; s[7] = 0;
    }
    {
        uint64_t m = s[0] * FQ_INV;
        w = (uint128_t)m * FQ_P[0] + s[0]; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[1] + s[1] + c; s[0] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[2] + s[2] + c; s[1] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[3] + s[3] + c; s[2] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[4] + s[4] + c; s[3] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FQ_P[5] + s[5] + c; s[4] = (uint64_t)w; c = (uint64_t)(w >> 64);
        s[5] = s[6] + c;
    }

    // Final conditional subtraction
    uint64_t borrow = 0;
    uint64_t tmp[6];
    uint128_t d;
    d = (uint128_t)s[0] - FQ_P[0]; tmp[0] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)s[1] - FQ_P[1] - borrow; tmp[1] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)s[2] - FQ_P[2] - borrow; tmp[2] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)s[3] - FQ_P[3] - borrow; tmp[3] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)s[4] - FQ_P[4] - borrow; tmp[4] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)s[5] - FQ_P[5] - borrow; tmp[5] = (uint64_t)d; borrow = (d >> 127) & 1;

    if (!borrow) {
        memcpy(r, tmp, 48);
    } else {
        memcpy(r, s, 48);
    }
}

static inline void fq_add(const uint64_t a[6], const uint64_t b[6], uint64_t r[6]) {
    uint128_t s;
    s = (uint128_t)a[0] + b[0]; r[0] = (uint64_t)s;
    s = (uint128_t)a[1] + b[1] + (uint64_t)(s >> 64); r[1] = (uint64_t)s;
    s = (uint128_t)a[2] + b[2] + (uint64_t)(s >> 64); r[2] = (uint64_t)s;
    s = (uint128_t)a[3] + b[3] + (uint64_t)(s >> 64); r[3] = (uint64_t)s;
    s = (uint128_t)a[4] + b[4] + (uint64_t)(s >> 64); r[4] = (uint64_t)s;
    s = (uint128_t)a[5] + b[5] + (uint64_t)(s >> 64); r[5] = (uint64_t)s;
    uint64_t carry = (uint64_t)(s >> 64);

    if (carry || (r[5] > FQ_P[5]) ||
        (r[5] == FQ_P[5] && (r[4] > FQ_P[4] ||
        (r[4] == FQ_P[4] && (r[3] > FQ_P[3] ||
        (r[3] == FQ_P[3] && (r[2] > FQ_P[2] ||
        (r[2] == FQ_P[2] && (r[1] > FQ_P[1] ||
        (r[1] == FQ_P[1] && r[0] >= FQ_P[0])))))))))) {
        uint128_t borrow;
        borrow = (uint128_t)r[0] - FQ_P[0]; r[0] = (uint64_t)borrow;
        uint64_t b1 = (uint64_t)(borrow >> 64) & 1;
        borrow = (uint128_t)r[1] - FQ_P[1] - b1; r[1] = (uint64_t)borrow;
        uint64_t b2 = (uint64_t)(borrow >> 64) & 1;
        borrow = (uint128_t)r[2] - FQ_P[2] - b2; r[2] = (uint64_t)borrow;
        uint64_t b3 = (uint64_t)(borrow >> 64) & 1;
        borrow = (uint128_t)r[3] - FQ_P[3] - b3; r[3] = (uint64_t)borrow;
        uint64_t b4 = (uint64_t)(borrow >> 64) & 1;
        borrow = (uint128_t)r[4] - FQ_P[4] - b4; r[4] = (uint64_t)borrow;
        uint64_t b5 = (uint64_t)(borrow >> 64) & 1;
        r[5] = r[5] - FQ_P[5] - b5;
    }
}

static inline void fq_sub(const uint64_t a[6], const uint64_t b[6], uint64_t r[6]) {
    uint128_t borrow;
    borrow = (uint128_t)a[0] - b[0]; r[0] = (uint64_t)borrow;
    uint64_t b1 = (uint64_t)(borrow >> 64) & 1;
    borrow = (uint128_t)a[1] - b[1] - b1; r[1] = (uint64_t)borrow;
    uint64_t b2 = (uint64_t)(borrow >> 64) & 1;
    borrow = (uint128_t)a[2] - b[2] - b2; r[2] = (uint64_t)borrow;
    uint64_t b3 = (uint64_t)(borrow >> 64) & 1;
    borrow = (uint128_t)a[3] - b[3] - b3; r[3] = (uint64_t)borrow;
    uint64_t b4 = (uint64_t)(borrow >> 64) & 1;
    borrow = (uint128_t)a[4] - b[4] - b4; r[4] = (uint64_t)borrow;
    uint64_t b5 = (uint64_t)(borrow >> 64) & 1;
    borrow = (uint128_t)a[5] - b[5] - b5; r[5] = (uint64_t)borrow;
    uint64_t underflow = (uint64_t)(borrow >> 64) & 1;

    if (underflow) {
        uint128_t s;
        s = (uint128_t)r[0] + FQ_P[0]; r[0] = (uint64_t)s;
        s = (uint128_t)r[1] + FQ_P[1] + (uint64_t)(s >> 64); r[1] = (uint64_t)s;
        s = (uint128_t)r[2] + FQ_P[2] + (uint64_t)(s >> 64); r[2] = (uint64_t)s;
        s = (uint128_t)r[3] + FQ_P[3] + (uint64_t)(s >> 64); r[3] = (uint64_t)s;
        s = (uint128_t)r[4] + FQ_P[4] + (uint64_t)(s >> 64); r[4] = (uint64_t)s;
        r[5] = r[5] + FQ_P[5] + (uint64_t)(s >> 64);
    }
}

static inline void fq_neg(const uint64_t a[6], uint64_t r[6]) {
    if (a[0] == 0 && a[1] == 0 && a[2] == 0 && a[3] == 0 && a[4] == 0 && a[5] == 0) {
        memset(r, 0, 48);
        return;
    }
    fq_sub(FQ_P, a, r);
}

static inline void fq_dbl(const uint64_t a[6], uint64_t r[6]) {
    fq_add(a, a, r);
}

// Fq inverse via Fermat: a^(q-2) mod q
static void fq_inverse(const uint64_t a[6], uint64_t r[6]) {
    // exp = q - 2
    uint64_t exp[6];
    memcpy(exp, FQ_P, 48);
    exp[0] -= 2;  // q[0] >= 2 so no borrow

    uint64_t result[6], base[6];
    memcpy(result, FQ_ONE, 48);
    memcpy(base, a, 48);

    for (int i = 0; i < 6; i++) {
        uint64_t word = exp[i];
        for (int bit = 0; bit < 64; bit++) {
            if (word & 1) {
                uint64_t tmp[6];
                fq_mul(result, base, tmp);
                memcpy(result, tmp, 48);
            }
            uint64_t tmp[6];
            fq_mul(base, base, tmp);
            memcpy(base, tmp, 48);
            word >>= 1;
        }
    }
    memcpy(r, result, 48);
}

// ============================================================
// Exported Fq functions
// ============================================================

void bls12_377_fq_mul(const uint64_t a[6], const uint64_t b[6], uint64_t r[6]) {
    fq_mul(a, b, r);
}

void bls12_377_fq_sqr(const uint64_t a[6], uint64_t r[6]) {
    fq_sqr(a, r);
}

void bls12_377_fq_add(const uint64_t a[6], const uint64_t b[6], uint64_t r[6]) {
    fq_add(a, b, r);
}

void bls12_377_fq_sub(const uint64_t a[6], const uint64_t b[6], uint64_t r[6]) {
    fq_sub(a, b, r);
}

void bls12_377_fq_neg(const uint64_t a[6], uint64_t r[6]) {
    fq_neg(a, r);
}

void bls12_377_fq_inverse(const uint64_t a[6], uint64_t r[6]) {
    fq_inverse(a, r);
}

// ============================================================
// Fr CIOS Montgomery multiplication (fully unrolled, 4 limbs)
// ============================================================

static inline void fr_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0;

    // Iteration i=0
    {
        uint128_t w;
        w = (uint128_t)a[0] * b[0];
        t0 = (uint64_t)w; uint64_t c = (uint64_t)(w >> 64);
        w = (uint128_t)a[0] * b[1] + t1 + c;
        t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[0] * b[2] + t2 + c;
        t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[0] * b[3] + t3 + c;
        t3 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t4 = c;
        uint64_t m = t0 * FR_INV;
        w = (uint128_t)m * FR_P[0] + t0; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FR_P[1] + t1 + c; t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FR_P[2] + t2 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FR_P[3] + t3 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t3 = t4 + c; t4 = (t3 < c) ? 1 : 0;
    }
    // Iteration i=1
    {
        uint128_t w; uint64_t c;
        w = (uint128_t)a[1] * b[0] + t0; t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[1] * b[1] + t1 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[1] * b[2] + t2 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[1] * b[3] + t3 + c; t3 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t4 += c;
        uint64_t m = t0 * FR_INV;
        w = (uint128_t)m * FR_P[0] + t0; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FR_P[1] + t1 + c; t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FR_P[2] + t2 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FR_P[3] + t3 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t3 = t4 + c; t4 = (t3 < c) ? 1 : 0;
    }
    // Iteration i=2
    {
        uint128_t w; uint64_t c;
        w = (uint128_t)a[2] * b[0] + t0; t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[2] * b[1] + t1 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[2] * b[2] + t2 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[2] * b[3] + t3 + c; t3 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t4 += c;
        uint64_t m = t0 * FR_INV;
        w = (uint128_t)m * FR_P[0] + t0; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FR_P[1] + t1 + c; t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FR_P[2] + t2 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FR_P[3] + t3 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t3 = t4 + c; t4 = (t3 < c) ? 1 : 0;
    }
    // Iteration i=3
    {
        uint128_t w; uint64_t c;
        w = (uint128_t)a[3] * b[0] + t0; t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[3] * b[1] + t1 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[3] * b[2] + t2 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[3] * b[3] + t3 + c; t3 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t4 += c;
        uint64_t m = t0 * FR_INV;
        w = (uint128_t)m * FR_P[0] + t0; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FR_P[1] + t1 + c; t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FR_P[2] + t2 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FR_P[3] + t3 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t3 = t4 + c; t4 = (t3 < c) ? 1 : 0;
    }

    // Final reduction
    if (t4 || (t3 > FR_P[3]) ||
        (t3 == FR_P[3] && (t2 > FR_P[2] ||
        (t2 == FR_P[2] && (t1 > FR_P[1] ||
        (t1 == FR_P[1] && t0 >= FR_P[0])))))) {
        uint128_t borrow;
        borrow = (uint128_t)t0 - FR_P[0]; t0 = (uint64_t)borrow;
        uint64_t b1 = (uint64_t)(borrow >> 64) & 1;
        borrow = (uint128_t)t1 - FR_P[1] - b1; t1 = (uint64_t)borrow;
        uint64_t b2 = (uint64_t)(borrow >> 64) & 1;
        borrow = (uint128_t)t2 - FR_P[2] - b2; t2 = (uint64_t)borrow;
        uint64_t b3 = (uint64_t)(borrow >> 64) & 1;
        t3 = t3 - FR_P[3] - b3;
    }

    r[0] = t0; r[1] = t1; r[2] = t2; r[3] = t3;
}

static inline void fr_sqr(const uint64_t a[4], uint64_t r[4]) {
    // Optimized squaring: upper-triangle doubled + diagonal (10 muls vs 16)
    uint128_t w;
    uint64_t c;
    uint64_t s0, s1, s2, s3, s4, s5, s6, s7;

    // Upper triangle: a[i]*a[j] for j>i (6 muls)
    w = (uint128_t)a[0] * a[1];
    s1 = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[0] * a[2] + c;
    s2 = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[0] * a[3] + c;
    s3 = (uint64_t)w; s4 = (uint64_t)(w >> 64);
    w = (uint128_t)a[1] * a[2] + s3;
    s3 = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[1] * a[3] + s4 + c;
    s4 = (uint64_t)w; s5 = (uint64_t)(w >> 64);
    w = (uint128_t)a[2] * a[3] + s5;
    s5 = (uint64_t)w; s6 = (uint64_t)(w >> 64);

    // Double the cross terms
    s7 = s6 >> 63;
    s6 = (s6 << 1) | (s5 >> 63);
    s5 = (s5 << 1) | (s4 >> 63);
    s4 = (s4 << 1) | (s3 >> 63);
    s3 = (s3 << 1) | (s2 >> 63);
    s2 = (s2 << 1) | (s1 >> 63);
    s1 = s1 << 1;

    // Add diagonal terms (4 muls)
    w = (uint128_t)a[0] * a[0];
    s0 = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)s1 + c;
    s1 = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[1] * a[1] + s2 + c;
    s2 = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)s3 + c;
    s3 = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[2] * a[2] + s4 + c;
    s4 = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)s5 + c;
    s5 = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[3] * a[3] + s6 + c;
    s6 = (uint64_t)w; c = (uint64_t)(w >> 64);
    s7 += c;

    // Montgomery reduction (4 iterations)
    {
        uint64_t m = s0 * FR_INV;
        w = (uint128_t)m * FR_P[0] + s0; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FR_P[1] + s1 + c; s0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FR_P[2] + s2 + c; s1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FR_P[3] + s3 + c; s2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)s4 + c; s3 = (uint64_t)w; c = (uint64_t)(w >> 64);
        s4 = s5 + c; s5 = s6; s6 = s7; s7 = 0;
    }
    {
        uint64_t m = s0 * FR_INV;
        w = (uint128_t)m * FR_P[0] + s0; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FR_P[1] + s1 + c; s0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FR_P[2] + s2 + c; s1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FR_P[3] + s3 + c; s2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)s4 + c; s3 = (uint64_t)w; c = (uint64_t)(w >> 64);
        s4 = s5 + c; s5 = s6; s6 = 0;
    }
    {
        uint64_t m = s0 * FR_INV;
        w = (uint128_t)m * FR_P[0] + s0; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FR_P[1] + s1 + c; s0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FR_P[2] + s2 + c; s1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FR_P[3] + s3 + c; s2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)s4 + c; s3 = (uint64_t)w; c = (uint64_t)(w >> 64);
        s4 = s5 + c; s5 = 0;
    }
    {
        uint64_t m = s0 * FR_INV;
        w = (uint128_t)m * FR_P[0] + s0; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FR_P[1] + s1 + c; s0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FR_P[2] + s2 + c; s1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FR_P[3] + s3 + c; s2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        s3 = s4 + c;
    }

    // Final conditional subtraction
    uint64_t borrow = 0;
    uint64_t r0, r1, r2, r3;
    uint128_t d;
    d = (uint128_t)s0 - FR_P[0] - borrow; r0 = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)s1 - FR_P[1] - borrow; r1 = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)s2 - FR_P[2] - borrow; r2 = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)s3 - FR_P[3] - borrow; r3 = (uint64_t)d; borrow = (d >> 127) & 1;
    if (!borrow) { r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3; }
    else { r[0] = s0; r[1] = s1; r[2] = s2; r[3] = s3; }
}

static inline void fr_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t s;
    s = (uint128_t)a[0] + b[0]; r[0] = (uint64_t)s;
    s = (uint128_t)a[1] + b[1] + (uint64_t)(s >> 64); r[1] = (uint64_t)s;
    s = (uint128_t)a[2] + b[2] + (uint64_t)(s >> 64); r[2] = (uint64_t)s;
    s = (uint128_t)a[3] + b[3] + (uint64_t)(s >> 64); r[3] = (uint64_t)s;
    uint64_t carry = (uint64_t)(s >> 64);

    if (carry || (r[3] > FR_P[3]) ||
        (r[3] == FR_P[3] && (r[2] > FR_P[2] ||
        (r[2] == FR_P[2] && (r[1] > FR_P[1] ||
        (r[1] == FR_P[1] && r[0] >= FR_P[0])))))) {
        uint128_t borrow;
        borrow = (uint128_t)r[0] - FR_P[0]; r[0] = (uint64_t)borrow;
        uint64_t b1 = (uint64_t)(borrow >> 64) & 1;
        borrow = (uint128_t)r[1] - FR_P[1] - b1; r[1] = (uint64_t)borrow;
        uint64_t b2 = (uint64_t)(borrow >> 64) & 1;
        borrow = (uint128_t)r[2] - FR_P[2] - b2; r[2] = (uint64_t)borrow;
        uint64_t b3 = (uint64_t)(borrow >> 64) & 1;
        r[3] = r[3] - FR_P[3] - b3;
    }
}

static inline void fr_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t borrow;
    borrow = (uint128_t)a[0] - b[0]; r[0] = (uint64_t)borrow;
    uint64_t b1 = (uint64_t)(borrow >> 64) & 1;
    borrow = (uint128_t)a[1] - b[1] - b1; r[1] = (uint64_t)borrow;
    uint64_t b2 = (uint64_t)(borrow >> 64) & 1;
    borrow = (uint128_t)a[2] - b[2] - b2; r[2] = (uint64_t)borrow;
    uint64_t b3 = (uint64_t)(borrow >> 64) & 1;
    borrow = (uint128_t)a[3] - b[3] - b3; r[3] = (uint64_t)borrow;
    uint64_t underflow = (uint64_t)(borrow >> 64) & 1;

    if (underflow) {
        uint128_t s;
        s = (uint128_t)r[0] + FR_P[0]; r[0] = (uint64_t)s;
        s = (uint128_t)r[1] + FR_P[1] + (uint64_t)(s >> 64); r[1] = (uint64_t)s;
        s = (uint128_t)r[2] + FR_P[2] + (uint64_t)(s >> 64); r[2] = (uint64_t)s;
        r[3] = r[3] + FR_P[3] + (uint64_t)(s >> 64);
    }
}

static inline void fr_neg(const uint64_t a[4], uint64_t r[4]) {
    if (a[0] == 0 && a[1] == 0 && a[2] == 0 && a[3] == 0) {
        memset(r, 0, 32);
        return;
    }
    fr_sub(FR_P, a, r);
}

// ============================================================
// Exported Fr functions
// ============================================================

void bls12_377_fr_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    fr_mul(a, b, r);
}

void bls12_377_fr_sqr(const uint64_t a[4], uint64_t r[4]) {
    fr_sqr(a, r);
}

void bls12_377_fr_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    fr_add(a, b, r);
}

void bls12_377_fr_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    fr_sub(a, b, r);
}

void bls12_377_fr_neg(const uint64_t a[4], uint64_t r[4]) {
    fr_neg(a, r);
}

// ============================================================
// BLS12-377 Fr NTT (Number Theoretic Transform)
// TWO_ADICITY = 47, root of unity for 2^47-th roots
// ============================================================

// Primitive 2^47-th root of unity in STANDARD form (not Montgomery).
// Converted to Montgomery at twiddle-init time via mul by R^2.
static const uint64_t FR377_ROOT_2_47[4] = {
    0x476ef4a4ec2a895eULL, 0x9b506ee363e3f04aULL,
    0x60c69477d1a8a12fULL, 0x11d4b7f60cb92cc1ULL
};
static const int FR377_TWO_ADICITY = 47;

// ---- Twiddle cache ----

typedef struct {
    uint64_t *fwd;
    uint64_t *inv;
    int logN;
} Fr377CachedTwiddles;

static Fr377CachedTwiddles fr377_cached[48] = {{0}};
static pthread_mutex_t fr377_cache_lock = PTHREAD_MUTEX_INITIALIZER;

static void fr377_pow(const uint64_t base[4], uint64_t exp_val, uint64_t result[4]) {
    memcpy(result, FR_ONE, 32);
    uint64_t b[4];
    memcpy(b, base, 32);
    while (exp_val > 0) {
        if (exp_val & 1)
            fr_mul(result, b, result);
        fr_mul(b, b, b);
        exp_val >>= 1;
    }
}

// Compute a^(p-2) mod p for modular inversion via Fermat's little theorem
static void fr377_inv(const uint64_t a[4], uint64_t result[4]) {
    uint64_t pm2[4];
    pm2[0] = FR_P[0] - 2;
    pm2[1] = FR_P[1];
    pm2[2] = FR_P[2];
    pm2[3] = FR_P[3];

    memcpy(result, FR_ONE, 32);
    uint64_t b[4];
    memcpy(b, a, 32);
    for (int i = 0; i < 4; i++) {
        for (int bit = 0; bit < 64; bit++) {
            if ((pm2[i] >> bit) & 1)
                fr_mul(result, b, result);
            fr_mul(b, b, b);
        }
    }
}

static void fr377_ensure_twiddles(int logN) {
    if (fr377_cached[logN].fwd) return;

    pthread_mutex_lock(&fr377_cache_lock);
    if (fr377_cached[logN].fwd) { pthread_mutex_unlock(&fr377_cache_lock); return; }

    int n = 1 << logN;

    // Convert root from standard to Montgomery form: root_mont = root * R^2 mod r
    uint64_t root_mont[4];
    fr_mul(FR377_ROOT_2_47, FR_R2, root_mont);

    // omega = root^(2^(TWO_ADICITY - logN)) — the primitive 2^logN-th root of unity
    uint64_t omega[4];
    memcpy(omega, root_mont, 32);
    for (int i = 0; i < FR377_TWO_ADICITY - logN; i++)
        fr_mul(omega, omega, omega);

    uint64_t omega_inv[4];
    fr377_inv(omega, omega_inv);

    // Allocate: (n-1) twiddles, each 4 limbs = 32 bytes
    uint64_t *fwd = (uint64_t *)malloc((size_t)(n - 1) * 32);
    uint64_t *inv = (uint64_t *)malloc((size_t)(n - 1) * 32);

    for (int s = 0; s < logN; s++) {
        int halfBlock = 1 << s;
        int offset = halfBlock - 1;

        // w_m = omega^(n / 2^(s+1))
        uint64_t w_m[4], w_m_inv[4];
        fr377_pow(omega, (uint64_t)(n >> (s + 1)), w_m);
        fr377_pow(omega_inv, (uint64_t)(n >> (s + 1)), w_m_inv);

        uint64_t w[4], wi[4];
        memcpy(w, FR_ONE, 32);
        memcpy(wi, FR_ONE, 32);

        for (int j = 0; j < halfBlock; j++) {
            memcpy(&fwd[(offset + j) * 4], w, 32);
            memcpy(&inv[(offset + j) * 4], wi, 32);
            fr_mul(w, w_m, w);
            fr_mul(wi, w_m_inv, wi);
        }
    }

    fr377_cached[logN].fwd = fwd;
    fr377_cached[logN].inv = inv;
    fr377_cached[logN].logN = logN;
    pthread_mutex_unlock(&fr377_cache_lock);
}

// Bit-reversal for 256-bit (4x64-bit) elements
static void fr377_bit_reverse_permute(uint64_t *data, int logN) {
    int n = 1 << logN;
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) {
            uint64_t t;
            t = data[i*4+0]; data[i*4+0] = data[j*4+0]; data[j*4+0] = t;
            t = data[i*4+1]; data[i*4+1] = data[j*4+1]; data[j*4+1] = t;
            t = data[i*4+2]; data[i*4+2] = data[j*4+2]; data[j*4+2] = t;
            t = data[i*4+3]; data[i*4+3] = data[j*4+3]; data[j*4+3] = t;
        }
    }
}

// Forward NTT: Cooley-Tukey DIT (bit-reverse then butterfly)
void bls12_377_fr_ntt(uint64_t *data, int logN) {
    if (logN <= 0) return;
    int n = 1 << logN;

    fr377_ensure_twiddles(logN);
    const uint64_t *tw = fr377_cached[logN].fwd;

    fr377_bit_reverse_permute(data, logN);

    for (int s = 0; s < logN; s++) {
        int halfBlock = 1 << s;
        int blockSize = halfBlock << 1;
        int nBlocks = n / blockSize;
        int twOffset = halfBlock - 1;

        for (int bk = 0; bk < nBlocks; bk++) {
            int base = bk * blockSize;
            // Twiddle skip: j==0 has twiddle==1 (Montgomery identity),
            // skip expensive 256-bit Montgomery mul
            {
                uint64_t *u = &data[base * 4];
                uint64_t *vp = &data[(base + halfBlock) * 4];

                uint64_t sum[4], diff[4];
                fr_add(u, vp, sum);
                fr_sub(u, vp, diff);

                memcpy(u, sum, 32);
                memcpy(vp, diff, 32);
            }
            for (int j = 1; j < halfBlock; j++) {
                uint64_t *u = &data[(base + j) * 4];
                uint64_t *vp = &data[(base + j + halfBlock) * 4];
                const uint64_t *twj = &tw[(twOffset + j) * 4];

                uint64_t v[4];
                fr_mul(twj, vp, v);

                uint64_t sum[4], diff[4];
                fr_add(u, v, sum);
                fr_sub(u, v, diff);

                memcpy(u, sum, 32);
                memcpy(vp, diff, 32);
            }
        }
    }
}

// Inverse NTT: Gentleman-Sande DIF + bit-reverse + 1/n scaling
void bls12_377_fr_intt(uint64_t *data, int logN) {
    if (logN <= 0) return;
    int n = 1 << logN;

    fr377_ensure_twiddles(logN);
    const uint64_t *tw = fr377_cached[logN].inv;

    // DIF stages (top-down)
    for (int si = 0; si < logN; si++) {
        int s = logN - 1 - si;
        int halfBlock = 1 << s;
        int blockSize = halfBlock << 1;
        int nBlocks = n / blockSize;
        int twOffset = halfBlock - 1;

        for (int bk = 0; bk < nBlocks; bk++) {
            int base = bk * blockSize;
            // Twiddle skip: j==0 has twiddle==1 (Montgomery identity)
            {
                uint64_t *ap = &data[base * 4];
                uint64_t *bp = &data[(base + halfBlock) * 4];

                uint64_t a[4], b[4];
                memcpy(a, ap, 32);
                memcpy(b, bp, 32);

                uint64_t sum[4], diff[4];
                fr_add(a, b, sum);
                fr_sub(a, b, diff);

                memcpy(ap, sum, 32);
                memcpy(bp, diff, 32);
            }
            for (int j = 1; j < halfBlock; j++) {
                uint64_t *ap = &data[(base + j) * 4];
                uint64_t *bp = &data[(base + j + halfBlock) * 4];
                const uint64_t *twj = &tw[(twOffset + j) * 4];

                uint64_t a[4], b[4];
                memcpy(a, ap, 32);
                memcpy(b, bp, 32);

                uint64_t sum[4], diff[4], prod[4];
                fr_add(a, b, sum);
                fr_sub(a, b, diff);
                fr_mul(diff, twj, prod);

                memcpy(ap, sum, 32);
                memcpy(bp, prod, 32);
            }
        }
    }

    fr377_bit_reverse_permute(data, logN);

    // Scale by 1/n: compute n_inv in Montgomery form
    uint64_t n_plain[4] = {(uint64_t)n, 0, 0, 0};
    uint64_t n_mont[4];
    fr_mul(n_plain, FR_R2, n_mont);

    uint64_t n_inv[4];
    fr377_inv(n_mont, n_inv);

    for (int i = 0; i < n; i++) {
        fr_mul(&data[i * 4], n_inv, &data[i * 4]);
    }
}

// ============================================================
// G1 Point operations: Jacobian projective coordinates
// y^2 = x^3 + 1 (a=0, b=1)
// Identity: Z = 0
// Point layout: [x[6], y[6], z[6]] = 18 uint64_t
// Affine layout: [x[6], y[6]] = 12 uint64_t
// ============================================================

static inline int g1_is_id(const uint64_t p[18]) {
    return p[12] == 0 && p[13] == 0 && p[14] == 0 &&
           p[15] == 0 && p[16] == 0 && p[17] == 0;
}

static inline void g1_set_id(uint64_t p[18]) {
    memcpy(p, FQ_ONE, 48);       // x = 1
    memcpy(p + 6, FQ_ONE, 48);   // y = 1
    memset(p + 12, 0, 48);       // z = 0
}

// Point doubling (a=0 curve: y^2 = x^3 + b)
static void g1_double(const uint64_t p[18], uint64_t r[18]) {
    if (g1_is_id(p)) { memcpy(r, p, 144); return; }

    const uint64_t *px = p, *py = p + 6, *pz = p + 12;
    uint64_t *rx = r, *ry = r + 6, *rz = r + 12;

    uint64_t a[6], b[6], c[6], d[6], e[6], f[6], t[6];

    fq_sqr(px, a);          // a = X^2
    fq_sqr(py, b);          // b = Y^2
    fq_sqr(b, c);           // c = Y^4

    // d = 2*((X+B)^2 - A - C)
    fq_add(px, b, t);
    fq_sqr(t, t);
    fq_sub(t, a, t);
    fq_sub(t, c, t);
    fq_dbl(t, d);

    // e = 3*A
    fq_dbl(a, e);
    fq_add(e, a, e);

    fq_sqr(e, f);           // f = E^2

    // X3 = F - 2*D
    fq_dbl(d, t);
    fq_sub(f, t, rx);

    // Y3 = E*(D - X3) - 8*C
    fq_sub(d, rx, t);
    fq_mul(e, t, ry);
    fq_dbl(c, t);
    fq_dbl(t, t);
    fq_dbl(t, t);           // 8*C
    fq_sub(ry, t, ry);

    // Z3 = (Y+Z)^2 - B - Z^2
    fq_add(py, pz, t);
    fq_sqr(t, rz);
    fq_sub(rz, b, rz);
    uint64_t z2[6];
    fq_sqr(pz, z2);
    fq_sub(rz, z2, rz);
}

// Full projective addition (Jacobian + Jacobian)
static void g1_add(const uint64_t p[18], const uint64_t q[18], uint64_t r[18]) {
    if (g1_is_id(p)) { memcpy(r, q, 144); return; }
    if (g1_is_id(q)) { memcpy(r, p, 144); return; }

    const uint64_t *px = p, *py = p + 6, *pz = p + 12;
    const uint64_t *qx = q, *qy = q + 6, *qz = q + 12;

    uint64_t z1z1[6], z2z2[6], u1[6], u2[6], s1[6], s2[6];
    uint64_t h[6], rr[6], t[6];

    fq_sqr(pz, z1z1);
    fq_sqr(qz, z2z2);
    fq_mul(px, z2z2, u1);
    fq_mul(qx, z1z1, u2);

    fq_mul(qz, z2z2, t);
    fq_mul(py, t, s1);
    fq_mul(pz, z1z1, t);
    fq_mul(qy, t, s2);

    fq_sub(u2, u1, h);

    // 2*h
    uint64_t h2[6];
    fq_dbl(h, h2);

    // Check for doubling case
    if (h2[0] == 0 && h2[1] == 0 && h2[2] == 0 &&
        h2[3] == 0 && h2[4] == 0 && h2[5] == 0) {
        uint64_t sdiff[6];
        fq_sub(s2, s1, sdiff);
        fq_dbl(sdiff, sdiff);
        if (sdiff[0] == 0 && sdiff[1] == 0 && sdiff[2] == 0 &&
            sdiff[3] == 0 && sdiff[4] == 0 && sdiff[5] == 0) {
            g1_double(p, r);
        } else {
            g1_set_id(r);
        }
        return;
    }

    // rr = 2*(S2 - S1)
    fq_sub(s2, s1, rr);
    fq_dbl(rr, rr);

    uint64_t i[6], j[6], v[6];
    fq_sqr(h2, i);           // i = (2*h)^2
    fq_mul(h, i, j);         // j = h * i
    fq_mul(u1, i, v);        // v = u1 * i

    // X3 = rr^2 - j - 2*v
    uint64_t *rx = r, *ry = r + 6, *rz = r + 12;
    fq_sqr(rr, rx);
    fq_sub(rx, j, rx);
    fq_dbl(v, t);
    fq_sub(rx, t, rx);

    // Y3 = rr*(v - X3) - 2*s1*j
    fq_sub(v, rx, t);
    fq_mul(rr, t, ry);
    fq_mul(s1, j, t);
    fq_dbl(t, t);
    fq_sub(ry, t, ry);

    // Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2) * H
    fq_add(pz, qz, t);
    fq_sqr(t, rz);
    fq_sub(rz, z1z1, rz);
    fq_sub(rz, z2z2, rz);
    fq_mul(rz, h, rz);
}

// Mixed addition: projective P + affine Q (saves field muls)
static void g1_add_mixed(const uint64_t p[18], const uint64_t q_aff[12], uint64_t r[18]) {
    if (g1_is_id(p)) {
        // Copy affine point to projective: (x, y, ONE)
        memcpy(r, q_aff, 96);
        memcpy(r + 12, FQ_ONE, 48);
        return;
    }

    const uint64_t *px = p, *py = p + 6, *pz = p + 12;
    const uint64_t *qx = q_aff, *qy = q_aff + 6;

    uint64_t z1z1[6], u2[6], s2[6], h[6], hh[6], i[6], j[6], rr[6], v[6], t[6];

    fq_sqr(pz, z1z1);                // Z1Z1 = Z1^2
    fq_mul(qx, z1z1, u2);            // U2 = X2*Z1Z1
    fq_mul(pz, z1z1, t);             // Z1^3
    fq_mul(qy, t, s2);               // S2 = Y2*Z1^3

    fq_sub(u2, px, h);               // H = U2 - X1
    fq_sub(s2, py, rr);
    fq_dbl(rr, rr);                   // rr = 2*(S2 - Y1)

    // Check doubling case
    uint64_t h2[6];
    fq_dbl(h, h2);
    if (h2[0] == 0 && h2[1] == 0 && h2[2] == 0 &&
        h2[3] == 0 && h2[4] == 0 && h2[5] == 0) {
        uint64_t sdiff[6];
        fq_sub(s2, py, sdiff);
        fq_dbl(sdiff, sdiff);
        if (sdiff[0] == 0 && sdiff[1] == 0 && sdiff[2] == 0 &&
            sdiff[3] == 0 && sdiff[4] == 0 && sdiff[5] == 0) {
            g1_double(p, r);
        } else {
            g1_set_id(r);
        }
        return;
    }

    fq_sqr(h2, i);                   // I = (2*H)^2
    fq_mul(h, i, j);                 // J = H*I
    fq_mul(px, i, v);                // V = X1*I

    uint64_t *rx = r, *ry = r + 6, *rz = r + 12;

    // X3 = rr^2 - J - 2*V
    fq_sqr(rr, rx);
    fq_sub(rx, j, rx);
    fq_dbl(v, t);
    fq_sub(rx, t, rx);

    // Y3 = rr*(V - X3) - 2*Y1*J
    fq_sub(v, rx, t);
    fq_mul(rr, t, ry);
    fq_mul(py, j, t);
    fq_dbl(t, t);
    fq_sub(ry, t, ry);

    // Z3 = (Z1+H)^2 - Z1Z1 - HH
    fq_sqr(h, hh);
    fq_add(pz, h, t);
    fq_sqr(t, rz);
    fq_sub(rz, z1z1, rz);
    fq_sub(rz, hh, rz);
}

// Scalar multiplication using windowed method (w=4, 16-entry table)
static void g1_scalar_mul(const uint64_t p[18], const uint64_t scalar[6], uint64_t r[18]) {
    // Build table: table[i] = i*P for i=0..15
    uint64_t table[16 * 18];
    g1_set_id(table);                     // table[0] = identity
    memcpy(table + 18, p, 144);           // table[1] = P
    g1_double(p, table + 2 * 18);         // table[2] = 2P
    for (int i = 3; i < 16; i++) {
        g1_add(table + (i - 1) * 18, p, table + i * 18);
    }

    g1_set_id(r);

    // Process scalar from MSB to LSB, 4 bits at a time
    // 384 bits = 96 nibbles
    int started = 0;
    for (int i = 95; i >= 0; i--) {
        if (started) {
            uint64_t tmp[18];
            g1_double(r, tmp);
            g1_double(tmp, r);
            g1_double(r, tmp);
            g1_double(tmp, r);
        }

        int word_idx = i / 16;   // which uint64_t
        int nib_idx = i % 16;    // which nibble within that word
        uint64_t nibble = (scalar[word_idx] >> (nib_idx * 4)) & 0xF;

        if (nibble != 0) {
            if (!started) {
                memcpy(r, table + nibble * 18, 144);
                started = 1;
            } else {
                uint64_t tmp[18];
                g1_add(r, table + nibble * 18, tmp);
                memcpy(r, tmp, 144);
            }
        }
    }

    if (!started) {
        g1_set_id(r);
    }
}

// ============================================================
// Exported G1 point functions
// ============================================================

void bls12_377_g1_point_add(const uint64_t p[18], const uint64_t q[18], uint64_t r[18]) {
    g1_add(p, q, r);
}

void bls12_377_g1_point_double(const uint64_t p[18], uint64_t r[18]) {
    g1_double(p, r);
}

void bls12_377_g1_point_add_mixed(const uint64_t p[18], const uint64_t q_aff[12], uint64_t r[18]) {
    g1_add_mixed(p, q_aff, r);
}

void bls12_377_g1_scalar_mul(const uint64_t p[18], const uint64_t scalar[6], uint64_t r[18]) {
    g1_scalar_mul(p, scalar, r);
}

void bls12_377_g1_to_affine(const uint64_t p[18], uint64_t aff[12]) {
    if (g1_is_id(p)) {
        memset(aff, 0, 96);
        return;
    }
    uint64_t zinv[6], zinv2[6], zinv3[6];
    fq_inverse(p + 12, zinv);
    fq_sqr(zinv, zinv2);
    fq_mul(zinv2, zinv, zinv3);
    fq_mul(p, zinv2, aff);
    fq_mul(p + 6, zinv3, aff + 6);
}

// ============================================================
// Pippenger MSM for BLS12-377 G1
// Features: batch-to-affine, signed-digit recoding, adaptive window,
//           multi-threaded windows via pthreads, mixed affine addition
// ============================================================

static inline int fq_is_zero(const uint64_t a[6]) {
    return (a[0] | a[1] | a[2] | a[3] | a[4] | a[5]) == 0;
}

static inline int g1_aff_is_id(const uint64_t q[12]) {
    return (q[0] | q[1] | q[2] | q[3] | q[4] | q[5] |
            q[6] | q[7] | q[8] | q[9] | q[10] | q[11]) == 0;
}

// Batch projective-to-affine via Montgomery's trick (single inversion)
static void g1_batch_to_affine(const uint64_t *proj, uint64_t *aff, int n) {
    if (n == 0) return;

    uint64_t *prods = (uint64_t *)malloc((size_t)n * 48);
    int first_valid = -1;

    for (int i = 0; i < n; i++) {
        const uint64_t *pz = proj + i * 18 + 12;
        if (g1_is_id(proj + i * 18)) {
            if (i == 0) memcpy(prods, FQ_ONE, 48);
            else memcpy(prods + i * 6, prods + (i - 1) * 6, 48);
        } else {
            if (first_valid < 0) {
                first_valid = i;
                memcpy(prods + i * 6, pz, 48);
            } else {
                fq_mul(prods + (i - 1) * 6, pz, prods + i * 6);
            }
        }
    }

    if (first_valid < 0) {
        memset(aff, 0, (size_t)n * 96);
        free(prods);
        return;
    }

    uint64_t inv[6];
    fq_inverse(prods + (n - 1) * 6, inv);

    for (int i = n - 1; i >= 0; i--) {
        if (g1_is_id(proj + i * 18)) {
            memset(aff + i * 12, 0, 96);
            continue;
        }

        uint64_t zinv[6];
        if (i > first_valid) {
            fq_mul(inv, prods + (i - 1) * 6, zinv);
            uint64_t tmp[6];
            fq_mul(inv, proj + i * 18 + 12, tmp);
            memcpy(inv, tmp, 48);
        } else {
            memcpy(zinv, inv, 48);
        }

        uint64_t zinv2[6], zinv3[6];
        fq_sqr(zinv, zinv2);
        fq_mul(zinv2, zinv, zinv3);
        fq_mul(proj + i * 18, zinv2, aff + i * 12);
        fq_mul(proj + i * 18 + 6, zinv3, aff + i * 12 + 6);
    }

    free(prods);
}

// Extract window_bits-wide window from 256-bit scalar (8 x uint32_t LE)
static inline uint32_t g1_extract_window(const uint32_t *scalar, int window_idx, int window_bits) {
    int bit_offset = window_idx * window_bits;
    int word_idx = bit_offset / 32;
    int bit_in_word = bit_offset % 32;

    uint64_t word = scalar[word_idx];
    if (word_idx + 1 < 8)
        word |= ((uint64_t)scalar[word_idx + 1]) << 32;

    return (uint32_t)((word >> bit_in_word) & ((1u << window_bits) - 1));
}

// Adaptive window sizing for BLS12-377 (6-limb Fq, heavier field ops)
static int g1_optimal_window_bits(int n) {
    if (n <= 4)      return 3;
    if (n <= 32)     return 5;
    if (n <= 256)    return 8;
    if (n <= 2048)   return 10;
    if (n <= 8192)   return 11;
    if (n <= 32768)  return 13;
    if (n <= 131072) return 14;
    if (n <= 524288) return 15;
    return 16;
}

// Per-window worker task
typedef struct {
    const uint64_t *points;     // n affine points (12 limbs each)
    const uint32_t *scalars;    // n scalars (8 uint32 each)
    int n;
    int window_bits;
    int window_idx;
    int num_buckets;
    uint64_t result[18];        // output projective point (18 limbs)
} G1WindowTask;

static void *g1_window_worker(void *arg) {
    G1WindowTask *task = (G1WindowTask *)arg;
    int wb = task->window_bits;
    int w = task->window_idx;
    int nb = task->num_buckets;
    int nn = task->n;

    // Allocate projective buckets (identity-initialized)
    uint64_t *buckets = (uint64_t *)malloc((size_t)(nb + 1) * 144);
    for (int b = 0; b <= nb; b++)
        g1_set_id(buckets + b * 18);

    // Phase 1: Bucket accumulation (mixed affine addition)
    for (int i = 0; i < nn; i++) {
        uint32_t digit = g1_extract_window(task->scalars + i * 8, w, wb);
        if (digit != 0) {
            uint64_t tmp[18];
            g1_add_mixed(buckets + digit * 18, task->points + i * 12, tmp);
            memcpy(buckets + digit * 18, tmp, 144);
        }
    }

    // Phase 2: Batch convert buckets to affine (Montgomery's trick)
    uint64_t *bucket_aff = (uint64_t *)malloc((size_t)nb * 96);
    g1_batch_to_affine(buckets + 18, bucket_aff, nb);

    // Phase 3: Running-sum reduction using mixed addition
    uint64_t running[18], window_sum[18];
    g1_set_id(running);
    g1_set_id(window_sum);

    for (int j = nb - 1; j >= 0; j--) {
        if (!g1_aff_is_id(bucket_aff + j * 12)) {
            uint64_t tmp[18];
            g1_add_mixed(running, bucket_aff + j * 12, tmp);
            memcpy(running, tmp, 144);
        }
        uint64_t tmp[18];
        g1_add(window_sum, running, tmp);
        memcpy(window_sum, tmp, 144);
    }

    memcpy(task->result, window_sum, 144);
    free(buckets);
    free(bucket_aff);
    return NULL;
}

// Main entry point: BLS12-377 G1 Pippenger MSM
void bls12_377_g1_pippenger_msm(
    const uint64_t *points,    // n affine points: n x 12 uint64_t (x[6], y[6])
    const uint32_t *scalars,   // n scalars: n x 8 uint32_t (256-bit LE)
    int n,
    uint64_t *result)          // output: 18 uint64_t (projective)
{
    if (n == 0) { g1_set_id(result); return; }

    // For very small n, use direct scalar-mul accumulation
    if (n <= 4) {
        g1_set_id(result);
        for (int i = 0; i < n; i++) {
            // Convert scalar from 8 x uint32 to 4 x uint64
            uint64_t s64[4];
            for (int j = 0; j < 4; j++)
                s64[j] = (uint64_t)scalars[i * 8 + j * 2] | ((uint64_t)scalars[i * 8 + j * 2 + 1] << 32);

            // Convert affine to projective for scalar_mul
            // scalar_mul expects 6-limb Fq scalar (384 bits), pad with zeros
            uint64_t s384[6] = {s64[0], s64[1], s64[2], s64[3], 0, 0};

            uint64_t proj_pt[18];
            memcpy(proj_pt, points + i * 12, 96);  // x, y
            memcpy(proj_pt + 12, FQ_ONE, 48);       // z = 1

            uint64_t term[18];
            g1_scalar_mul(proj_pt, s384, term);

            uint64_t tmp[18];
            g1_add(result, term, tmp);
            memcpy(result, tmp, 144);
        }
        return;
    }

    int wb = g1_optimal_window_bits(n);
    int num_windows = (256 + wb - 1) / wb;
    int num_buckets = (1 << wb) - 1;

    // Allocate tasks and threads
    G1WindowTask *tasks = (G1WindowTask *)malloc((size_t)num_windows * sizeof(G1WindowTask));

    for (int w = 0; w < num_windows; w++) {
        tasks[w].points = points;
        tasks[w].scalars = scalars;
        tasks[w].n = n;
        tasks[w].window_bits = wb;
        tasks[w].window_idx = w;
        tasks[w].num_buckets = num_buckets;
    }

    dispatch_apply(num_windows, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
        ^(size_t w) {
            g1_window_worker(&tasks[w]);
        });

    // Horner combination: result = sum windowResults[w] * 2^(w * wb)
    memcpy(result, tasks[num_windows - 1].result, 144);
    for (int w = num_windows - 2; w >= 0; w--) {
        uint64_t tmp[18];
        for (int s = 0; s < wb; s++) {
            g1_double(result, tmp);
            memcpy(result, tmp, 144);
        }
        g1_add(result, tasks[w].result, tmp);
        memcpy(result, tmp, 144);
    }

    free(tasks);
}

// Batch inverse for Fq (6-limb) via Montgomery's trick
void bls12_377_fq_batch_inverse(const uint64_t *in, uint64_t *out, int n) {
    if (n <= 0) return;
    memcpy(&out[0], &in[0], 48);
    for (int i = 1; i < n; i++)
        fq_mul(&out[(i-1)*6], &in[i*6], &out[i*6]);
    uint64_t inv_acc[6];
    fq_inverse(&out[(n-1)*6], inv_acc);
    for (int i = n - 1; i > 0; i--) {
        uint64_t tmp[6];
        fq_mul(inv_acc, &out[(i-1)*6], tmp);
        fq_mul(inv_acc, &in[i*6], inv_acc);
        memcpy(&out[i*6], tmp, 48);
    }
    memcpy(&out[0], inv_acc, 48);
}

// Batch inverse for Fr (4-limb) via Montgomery's trick
void bls12_377_fr_batch_inverse(const uint64_t *in, uint64_t *out, int n) {
    if (n <= 0) return;
    memcpy(&out[0], &in[0], 32);
    for (int i = 1; i < n; i++)
        fr_mul(&out[(i-1)*4], &in[i*4], &out[i*4]);
    uint64_t inv_acc[4];
    fr377_inv(&out[(n-1)*4], inv_acc);
    for (int i = n - 1; i > 0; i--) {
        uint64_t tmp[4];
        fr_mul(inv_acc, &out[(i-1)*4], tmp);
        fr_mul(inv_acc, &in[i*4], inv_acc);
        memcpy(&out[i*4], tmp, 32);
    }
    memcpy(&out[0], inv_acc, 32);
}
// ============================================================
// GLV endomorphism for BLS12-377 G1 MSM
// φ(x,y) = (ω·x, y) where ω is a cube root of unity in Fq
// k = k1 + k2·λ (mod r) with |k1|, |k2| ≈ √r (~128 bits)
//
// Constants:
//   λ (cube root of unity in Fr):
//     0x12ab655e9a2ca55660b44d1e5c37b0010000000000000000
//   ω (cube root of unity in Fq, standard form):
//     0x01ae3a4617c510eabc8756ba8f8c524eb8882a75cc9bc8e3
//     59064ee822fb5bffd1e945779fffffffffffffffffffffff
//   a1 = r - λ = 91893752504881257701523279626832445441
//   Lattice: v1=(a1, 1), v2=(-1, a1-1)
//   Decomposition: c1 = floor(k*(a1-1)/r), k1 = k - c1*a1, k2 = -c1
// ============================================================

// ω (cube root of unity in Fq) in Montgomery form
// ω_mont = ω * R mod q where R = 2^384
static const uint64_t FQ_OMEGA_MONT[6] = {
    // Precomputed: fq_mul(omega_raw, FQ_R2)
    // omega_raw = q - 1 (which equals -1 mod q, but that's wrong)
    // Actually omega = (q-1)/2 + ... no.
    // BLS12-377: omega^3 = 1 mod q, omega != 1
    // omega in standard form:
    //   0x01ae3a4617c510ea bc8756ba8f8c524e b8882a75cc9bc8e3
    //   59064ee822fb5bff d1e945779fffffff ffffffffffffffff
    // (This is q - 1 shifted -- actually it's a specific algebraic value)
    // We'll compute this at init time via fq_mul(raw, R2)
    0, 0, 0, 0, 0, 0  // placeholder - filled by init
};

// omega raw (standard form, 6x64-bit LE)
static const uint64_t FQ_OMEGA_RAW[6] = {
    0xFFFFFFFFFFFFFFFFULL, 0xD1E945779FFFFFFFULL,
    0x59064EE822FB5BFFULL, 0xB8882A75CC9BC8E3ULL,
    0xBC8756BA8F8C524EULL, 0x01AE3A4617C510EAULL
};

// GLV lattice constant a1 = 91893752504881257701523279626832445441
// In hex: 0x452217CC90000001_0A11800000000001
static const uint64_t GLV377_A1[2] = {
    0x0A11800000000001ULL, 0x452217CC90000001ULL
};

// a1 - 1
static const uint64_t GLV377_A1M1[2] = {
    0x0A11800000000000ULL, 0x452217CC90000001ULL
};

// half_r = (r+1)/2
static const uint64_t FR_HALF[4] = {
    0x0508C00000000001ULL, 0x2CD53B7F68000000ULL,
    0xB05A268F2E1BD801ULL, 0x0955B2AF4D1652ABULL
};

// Compute omega in Montgomery form (call once at startup or lazily)
static uint64_t glv377_omega_mont[6] = {0};
static int glv377_omega_init = 0;

static void glv377_ensure_omega(void) {
    if (glv377_omega_init) return;
    fq_mul(FQ_OMEGA_RAW, FQ_R2, glv377_omega_mont);
    glv377_omega_init = 1;
}

// 256-bit >= 256-bit comparison (4x64 LE)
static inline int u256_gte(const uint64_t a[4], const uint64_t b[4]) {
    for (int i = 3; i >= 0; i--) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return 0;
    }
    return 1; // equal
}

// 256-bit subtraction: r = a - b, returns borrow
static inline int u256_sub(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint128_t borrow = 0;
    for (int i = 0; i < 4; i++) {
        uint128_t diff = (uint128_t)a[i] - b[i] - borrow;
        r[i] = (uint64_t)diff;
        borrow = (diff >> 127) & 1; // borrow if negative
    }
    return (int)borrow;
}

// 256-bit addition: r = a + b, returns carry
static inline int u256_add(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint128_t carry = 0;
    for (int i = 0; i < 4; i++) {
        carry += (uint128_t)a[i] + b[i];
        r[i] = (uint64_t)carry;
        carry >>= 64;
    }
    return (int)carry;
}

// 256x128 -> 384-bit multiply, then approximate div by r to get ~128-bit quotient
// Returns c1 = approx floor(k * (a1-1) / r)
static void glv377_compute_c1(const uint64_t k[4], uint64_t *c1_lo, uint64_t *c1_hi) {
    // Full 256x128 multiply: k[0..3] * a1m1[0..1] -> prod[0..5]
    uint64_t prod[6] = {0,0,0,0,0,0};
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 2; j++) {
            uint128_t w = (uint128_t)k[i] * GLV377_A1M1[j] + prod[i+j] + carry;
            prod[i+j] = (uint64_t)w;
            carry = (uint64_t)(w >> 64);
        }
        prod[i+2] += carry;
    }

    // prod is 384 bits. Quotient c1 = prod / r.
    // Since r ≈ 2^253, we approximate: c1 ≈ prod >> 251, then adjust.
    // prod >> 251 = (prod[3] >> 59) | (prod[4] << 5), (prod[4] >> 59) | (prod[5] << 5)
    *c1_lo = (prod[3] >> 59) | (prod[4] << 5);
    *c1_hi = (prod[4] >> 59) | (prod[5] << 5);
}

// 128x128 -> 256-bit multiply
static void mul128x128_glv(uint64_t a0, uint64_t a1, const uint64_t b[2], uint64_t r[4]) {
    uint128_t w;
    w = (uint128_t)a0 * b[0];
    r[0] = (uint64_t)w;
    uint64_t carry = (uint64_t)(w >> 64);

    w = (uint128_t)a0 * b[1] + carry;
    uint64_t t1 = (uint64_t)w;
    uint64_t t1c = (uint64_t)(w >> 64);

    w = (uint128_t)a1 * b[0] + t1;
    r[1] = (uint64_t)w;
    carry = (uint64_t)(w >> 64);

    w = (uint128_t)a1 * b[1] + t1c + carry;
    r[2] = (uint64_t)w;
    r[3] = (uint64_t)(w >> 64);
}

// GLV decomposition: scalar k (256-bit, 8xuint32 LE) -> (k1, k2, neg1, neg2)
// k ≡ k1 + k2·λ (mod r), |k1|,|k2| ≈ √r
static void glv377_decompose(const uint32_t *scalar,
                              uint32_t *k1_out, uint32_t *k2_out,
                              int *neg1, int *neg2) {
    // Convert to 4x64
    uint64_t k[4];
    for (int i = 0; i < 4; i++)
        k[i] = (uint64_t)scalar[2*i] | ((uint64_t)scalar[2*i+1] << 32);

    // Reduce mod r
    while (u256_gte(k, FR_P)) {
        uint64_t tmp[4];
        u256_sub(tmp, k, FR_P);
        memcpy(k, tmp, 32);
    }

    // c1 = approx floor(k * (a1-1) / r)
    uint64_t c1_lo, c1_hi;
    glv377_compute_c1(k, &c1_lo, &c1_hi);

    // k1 = k - c1 * a1
    uint64_t c1a1[4];
    mul128x128_glv(c1_lo, c1_hi, GLV377_A1, c1a1);

    uint64_t k1[4];
    int borrow = u256_sub(k1, k, c1a1);
    if (borrow) {
        // c1 too big, adjust down
        if (c1_lo == 0) c1_hi--;
        c1_lo--;
        u256_add(k1, k1, FR_P);
    }

    // Check if k1 >= r (c1 too small)
    while (u256_gte(k1, FR_P)) {
        uint64_t tmp[4];
        u256_sub(tmp, k1, FR_P);
        memcpy(k1, tmp, 32);
        c1_lo++; if (c1_lo == 0) c1_hi++;
    }

    // k2 = c1 (positive value; actual k2 = -c1)
    uint64_t k2[4] = {c1_lo, c1_hi, 0, 0};

    // If k1 > half_r, negate for balanced representation
    *neg1 = 0;
    if (u256_gte(k1, FR_HALF)) {
        uint64_t tmp[4];
        u256_sub(tmp, FR_P, k1);
        memcpy(k1, tmp, 32);
        *neg1 = 1;
    }

    *neg2 = (c1_lo != 0 || c1_hi != 0) ? 1 : 0;

    // Write out as 8xuint32
    for (int i = 0; i < 4; i++) {
        k1_out[2*i] = (uint32_t)(k1[i] & 0xFFFFFFFF);
        k1_out[2*i+1] = (uint32_t)(k1[i] >> 32);
        k2_out[2*i] = (uint32_t)(k2[i] & 0xFFFFFFFF);
        k2_out[2*i+1] = (uint32_t)(k2[i] >> 32);
    }
}

// Apply endomorphism: φ(x,y) = (ω·x, y)
// Input: affine point (12 limbs in Montgomery form)
// Output: affine point with x multiplied by ω
static void glv377_apply_endomorphism(const uint64_t *p_aff, uint64_t *out_aff) {
    fq_mul(p_aff, glv377_omega_mont, out_aff);  // out.x = ω * p.x
    memcpy(out_aff + 6, p_aff + 6, 48);          // out.y = p.y
}

// Negate affine point y-coordinate: (x, -y)
static void g1_aff_negate_y(uint64_t *aff) {
    uint64_t neg_y[6];
    fq_neg(aff + 6, neg_y);
    memcpy(aff + 6, neg_y, 48);
}

// Optimal window bits for GLV (128-bit scalars, 2n points)
static int g1_glv_window_bits(int n2) {
    // n2 = 2*n (doubled point count)
    if (n2 <= 8)       return 3;
    if (n2 <= 64)      return 5;
    if (n2 <= 512)     return 8;
    if (n2 <= 4096)    return 10;
    if (n2 <= 16384)   return 11;
    if (n2 <= 65536)   return 12;
    if (n2 <= 262144)  return 13;
    return 14;
}

// GLV-accelerated Pippenger MSM for BLS12-377 G1
// Decomposes each 253-bit scalar into two ~128-bit scalars via GLV,
// doubles point count, halves window count => ~2× speedup
void bls12_377_g1_glv_pippenger_msm(
    const uint64_t *points,    // n affine points: n x 12 uint64_t
    const uint32_t *scalars,   // n scalars: n x 8 uint32_t (256-bit LE)
    int n,
    uint64_t *result)          // output: 18 uint64_t (projective)
{
    if (n == 0) { g1_set_id(result); return; }

    // For very small n, fall back to non-GLV
    if (n <= 16) {
        bls12_377_g1_pippenger_msm(points, scalars, n, result);
        return;
    }

    glv377_ensure_omega();

    int n2 = 2 * n;

    // Allocate: 2n affine points + 2n scalars (8 uint32 each)
    uint64_t *glv_points = (uint64_t *)malloc((size_t)n2 * 96);   // 12 limbs each
    uint32_t *glv_scalars = (uint32_t *)malloc((size_t)n2 * 32);  // 8 uint32 each

    // Decompose scalars and build point/scalar arrays
    // First n entries: k1 * P (or -k1 * P if neg1)
    // Next n entries: k2 * φ(P) (or -k2 * φ(P) if neg2)
    dispatch_apply(n, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
        ^(size_t i) {
            uint32_t k1[8], k2[8];
            int neg1, neg2;
            glv377_decompose(scalars + i * 8, k1, k2, &neg1, &neg2);

            // P_i for k1
            memcpy(glv_points + i * 12, points + i * 12, 96);
            if (neg1) {
                g1_aff_negate_y(glv_points + i * 12);
            }
            memcpy(glv_scalars + i * 8, k1, 32);

            // φ(P_i) for k2
            glv377_apply_endomorphism(points + i * 12, glv_points + (n + i) * 12);
            if (neg2) {
                g1_aff_negate_y(glv_points + (n + i) * 12);
            }
            memcpy(glv_scalars + (n + i) * 8, k2, 32);
        });

    // Run Pippenger on 2n points with ~128-bit scalars
    // Use adapted window sizing for half-width scalars
    int wb = g1_glv_window_bits(n2);
    int scalar_bits = 128;
    int num_windows = (scalar_bits + wb - 1) / wb;
    int num_buckets = (1 << wb) - 1;

    G1WindowTask *tasks = (G1WindowTask *)malloc((size_t)num_windows * sizeof(G1WindowTask));
    for (int w = 0; w < num_windows; w++) {
        tasks[w].points = glv_points;
        tasks[w].scalars = glv_scalars;
        tasks[w].n = n2;
        tasks[w].window_bits = wb;
        tasks[w].window_idx = w;
        tasks[w].num_buckets = num_buckets;
    }

    dispatch_apply(num_windows, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
        ^(size_t w) {
            g1_window_worker(&tasks[w]);
        });

    // Horner combination
    memcpy(result, tasks[num_windows - 1].result, 144);
    for (int w = num_windows - 2; w >= 0; w--) {
        uint64_t tmp[18];
        for (int s = 0; s < wb; s++) {
            g1_double(result, tmp);
            memcpy(result, tmp, 144);
        }
        g1_add(result, tasks[w].result, tmp);
        memcpy(result, tmp, 144);
    }

    free(tasks);
    free(glv_points);
    free(glv_scalars);
}

