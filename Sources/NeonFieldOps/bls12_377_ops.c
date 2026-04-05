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
    fq_mul(a, a, r);
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
    fr_mul(a, a, r);
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

#include <pthread.h>

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
    pthread_t *threads = (pthread_t *)malloc((size_t)num_windows * sizeof(pthread_t));

    for (int w = 0; w < num_windows; w++) {
        tasks[w].points = points;
        tasks[w].scalars = scalars;
        tasks[w].n = n;
        tasks[w].window_bits = wb;
        tasks[w].window_idx = w;
        tasks[w].num_buckets = num_buckets;
    }

    // Launch threads (one per window)
    for (int w = 0; w < num_windows; w++)
        pthread_create(&threads[w], NULL, g1_window_worker, &tasks[w]);

    for (int w = 0; w < num_windows; w++)
        pthread_join(threads[w], NULL);

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
    free(threads);
}
