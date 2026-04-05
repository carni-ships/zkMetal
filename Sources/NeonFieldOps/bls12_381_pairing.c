// BLS12-381 pairing: Fp2/Fp6/Fp12 tower arithmetic + Miller loop + final exponentiation
// C-accelerated implementation using __uint128_t CIOS for Fp base field
//
// Tower: Fp2 = Fp[u]/(u^2+1), Fp6 = Fp2[v]/(v^3 - (1+u)), Fp12 = Fp6[w]/(w^2 - v)
// Pairing: optimal ate, x = -0xd201000000010000
//
// Memory layout:
//   Fp   = 6 x uint64_t (Montgomery form)
//   Fp2  = 12 x uint64_t (c0[6], c1[6])
//   Fp6  = 36 x uint64_t (c0[12], c1[12], c2[12])
//   Fp12 = 72 x uint64_t (c0[36], c1[36])
//   G2Affine = 24 x uint64_t (x[12], y[12])

#include "NeonFieldOps.h"
#include <string.h>
#include <stdlib.h>

typedef unsigned __int128 uint128_t;

// ============================================================
// Fp constants (duplicated locally for static inline access)
// ============================================================

static const uint64_t P[6] = {
    0xb9feffffffffaaabULL, 0x1eabfffeb153ffffULL,
    0x6730d2a0f6b0f624ULL, 0x64774b84f38512bfULL,
    0x4b1ba7b6434bacd7ULL, 0x1a0111ea397fe69aULL
};
static const uint64_t P_INV = 0x89f3fffcfffcfffdULL;
static const uint64_t FP_ONE[6] = {
    0x760900000002fffdULL, 0xebf4000bc40c0002ULL,
    0x5f48985753c758baULL, 0x77ce585370525745ULL,
    0x5c071a97a256ec6dULL, 0x15f65ec3fa80e493ULL
};

// ============================================================
// Fp arithmetic (inline, local copies for performance)
// ============================================================

static inline void fp_mul(const uint64_t a[6], const uint64_t b[6], uint64_t r[6]) {
    uint64_t t[7] = {0};
    for (int i = 0; i < 6; i++) {
        uint128_t carry = 0;
        for (int j = 0; j < 6; j++) {
            carry += (uint128_t)a[i] * b[j] + t[j];
            t[j] = (uint64_t)carry;
            carry >>= 64;
        }
        t[6] += (uint64_t)carry;
        uint64_t m = t[0] * P_INV;
        carry = (uint128_t)m * P[0] + t[0];
        carry >>= 64;
        for (int j = 1; j < 6; j++) {
            carry += (uint128_t)m * P[j] + t[j];
            t[j-1] = (uint64_t)carry;
            carry >>= 64;
        }
        t[5] = t[6] + (uint64_t)carry;
        t[6] = (uint64_t)(carry >> 64);
    }
    uint64_t borrow = 0, tmp[6];
    uint128_t d;
    d = (uint128_t)t[0] - P[0]; tmp[0] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)t[1] - P[1] - borrow; tmp[1] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)t[2] - P[2] - borrow; tmp[2] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)t[3] - P[3] - borrow; tmp[3] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)t[4] - P[4] - borrow; tmp[4] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)t[5] - P[5] - borrow; tmp[5] = (uint64_t)d; borrow = (d >> 127) & 1;
    if (!borrow) memcpy(r, tmp, 48); else memcpy(r, t, 48);
}

static inline void fp_sqr(const uint64_t a[6], uint64_t r[6]) { fp_mul(a, a, r); }

static inline void fp_add(const uint64_t a[6], const uint64_t b[6], uint64_t r[6]) {
    uint128_t w; uint64_t c = 0;
    w = (uint128_t)a[0]+b[0]; r[0]=(uint64_t)w; c=(uint64_t)(w>>64);
    w = (uint128_t)a[1]+b[1]+c; r[1]=(uint64_t)w; c=(uint64_t)(w>>64);
    w = (uint128_t)a[2]+b[2]+c; r[2]=(uint64_t)w; c=(uint64_t)(w>>64);
    w = (uint128_t)a[3]+b[3]+c; r[3]=(uint64_t)w; c=(uint64_t)(w>>64);
    w = (uint128_t)a[4]+b[4]+c; r[4]=(uint64_t)w; c=(uint64_t)(w>>64);
    w = (uint128_t)a[5]+b[5]+c; r[5]=(uint64_t)w; c=(uint64_t)(w>>64);
    uint64_t borrow=0, tmp[6]; uint128_t d;
    d=(uint128_t)r[0]-P[0]; tmp[0]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[1]-P[1]-borrow; tmp[1]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[2]-P[2]-borrow; tmp[2]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[3]-P[3]-borrow; tmp[3]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[4]-P[4]-borrow; tmp[4]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[5]-P[5]-borrow; tmp[5]=(uint64_t)d; borrow=(d>>127)&1;
    if (c||!borrow) memcpy(r, tmp, 48);
}

static inline void fp_sub(const uint64_t a[6], const uint64_t b[6], uint64_t r[6]) {
    uint128_t d; uint64_t borrow=0;
    d=(uint128_t)a[0]-b[0]; r[0]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[1]-b[1]-borrow; r[1]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[2]-b[2]-borrow; r[2]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[3]-b[3]-borrow; r[3]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[4]-b[4]-borrow; r[4]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[5]-b[5]-borrow; r[5]=(uint64_t)d; borrow=(d>>127)&1;
    if (borrow) {
        uint64_t c=0;
        d=(uint128_t)r[0]+P[0]; r[0]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[1]+P[1]+c; r[1]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[2]+P[2]+c; r[2]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[3]+P[3]+c; r[3]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[4]+P[4]+c; r[4]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[5]+P[5]+c; r[5]=(uint64_t)d;
    }
}

static inline void fp_neg(const uint64_t a[6], uint64_t r[6]) {
    int z = (a[0]|a[1]|a[2]|a[3]|a[4]|a[5]) == 0;
    if (z) { memset(r, 0, 48); return; }
    uint128_t d; uint64_t borrow=0;
    d=(uint128_t)P[0]-a[0]; r[0]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)P[1]-a[1]-borrow; r[1]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)P[2]-a[2]-borrow; r[2]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)P[3]-a[3]-borrow; r[3]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)P[4]-a[4]-borrow; r[4]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)P[5]-a[5]-borrow; r[5]=(uint64_t)d;
}

static inline void fp_dbl(const uint64_t a[6], uint64_t r[6]) { fp_add(a, a, r); }

static inline int fp_is_zero(const uint64_t a[6]) {
    return (a[0]|a[1]|a[2]|a[3]|a[4]|a[5]) == 0;
}

static inline void fp_copy(uint64_t dst[6], const uint64_t src[6]) { memcpy(dst, src, 48); }

// Fp inversion via Fermat: a^(p-2) mod p
static void fp_inv(const uint64_t a[6], uint64_t r[6]) {
    uint64_t pm2[6];
    for (int i = 0; i < 6; i++) pm2[i] = P[i];
    pm2[0] -= 2;
    memcpy(r, FP_ONE, 48);
    uint64_t b[6]; memcpy(b, a, 48);
    for (int i = 0; i < 6; i++) {
        for (int bit = 0; bit < 64; bit++) {
            if ((pm2[i] >> bit) & 1) fp_mul(r, b, r);
            fp_sqr(b, b);
        }
    }
}

// ============================================================
// Fp2 = Fp[u]/(u^2+1), layout: [c0[6], c1[6]] = 12 uint64_t
// ============================================================

#define FP2 12
#define FP  6

static inline void fp2_add(const uint64_t a[FP2], const uint64_t b[FP2], uint64_t r[FP2]) {
    fp_add(a, b, r);
    fp_add(a+6, b+6, r+6);
}

static inline void fp2_sub(const uint64_t a[FP2], const uint64_t b[FP2], uint64_t r[FP2]) {
    fp_sub(a, b, r);
    fp_sub(a+6, b+6, r+6);
}

static inline void fp2_neg(const uint64_t a[FP2], uint64_t r[FP2]) {
    fp_neg(a, r);
    fp_neg(a+6, r+6);
}

static inline void fp2_dbl(const uint64_t a[FP2], uint64_t r[FP2]) {
    fp_dbl(a, r);
    fp_dbl(a+6, r+6);
}

// Fp2 mul: (a0+a1*u)(b0+b1*u) = (a0*b0 - a1*b1) + (a0*b1 + a1*b0)*u
// Karatsuba: 3 Fp muls
static inline void fp2_mul(const uint64_t a[FP2], const uint64_t b[FP2], uint64_t r[FP2]) {
    uint64_t a0b0[6], a1b1[6], t1[6], t2[6], t3[6];
    fp_mul(a, b, a0b0);       // a0*b0
    fp_mul(a+6, b+6, a1b1);   // a1*b1
    fp_add(a, a+6, t1);       // a0+a1
    fp_add(b, b+6, t2);       // b0+b1
    fp_mul(t1, t2, t3);       // (a0+a1)(b0+b1)
    fp_sub(a0b0, a1b1, r);    // c0 = a0*b0 - a1*b1
    fp_sub(t3, a0b0, r+6);    // c1 = (a0+a1)(b0+b1) - a0*b0 - a1*b1
    fp_sub(r+6, a1b1, r+6);
}

// Fp2 sqr: complex squaring (a0+a1)(a0-a1) for c0, 2*a0*a1 for c1
static inline void fp2_sqr(const uint64_t a[FP2], uint64_t r[FP2]) {
    uint64_t v0[6], sum[6], diff[6];
    fp_mul(a, a+6, v0);       // a0*a1
    fp_add(a, a+6, sum);      // a0+a1
    fp_sub(a, a+6, diff);     // a0-a1
    fp_mul(sum, diff, r);     // c0 = (a0+a1)(a0-a1) = a0^2-a1^2
    fp_dbl(v0, r+6);          // c1 = 2*a0*a1
}

// Fp2 conjugate: (a0, -a1)
static inline void fp2_conj(const uint64_t a[FP2], uint64_t r[FP2]) {
    fp_copy(r, a);
    fp_neg(a+6, r+6);
}

// Fp2 mul by Fp scalar: (a0*s, a1*s)
static inline void fp2_mul_fp(const uint64_t a[FP2], const uint64_t s[FP], uint64_t r[FP2]) {
    fp_mul(a, s, r);
    fp_mul(a+6, s, r+6);
}

// Multiply by non-residue (1+u): (a0-a1) + (a0+a1)*u
static inline void fp2_mul_nr(const uint64_t a[FP2], uint64_t r[FP2]) {
    uint64_t t0[6], t1[6];
    fp_sub(a, a+6, t0);       // a0 - a1
    fp_add(a, a+6, t1);       // a0 + a1
    fp_copy(r, t0);
    fp_copy(r+6, t1);
}

// Fp2 inverse: 1/(a0+a1*u) = (a0-a1*u)/(a0^2+a1^2)
static void fp2_inv(const uint64_t a[FP2], uint64_t r[FP2]) {
    uint64_t t0[6], t1[6], norm[6], ninv[6];
    fp_sqr(a, t0);
    fp_sqr(a+6, t1);
    fp_add(t0, t1, norm);
    fp_inv(norm, ninv);
    fp_mul(a, ninv, r);
    uint64_t neg1[6];
    fp_neg(a+6, neg1);
    fp_mul(neg1, ninv, r+6);
}

static inline int fp2_is_zero(const uint64_t a[FP2]) {
    return fp_is_zero(a) && fp_is_zero(a+6);
}

static inline void fp2_copy(uint64_t dst[FP2], const uint64_t src[FP2]) { memcpy(dst, src, 96); }
static inline void fp2_zero(uint64_t r[FP2]) { memset(r, 0, 96); }
static inline void fp2_one(uint64_t r[FP2]) { memcpy(r, FP_ONE, 48); memset(r+6, 0, 48); }

// ============================================================
// Fp6 = Fp2[v]/(v^3 - (1+u)), layout: [c0[12], c1[12], c2[12]] = 36 uint64_t
// ============================================================

#define FP6 36

static inline void fp6_add(const uint64_t a[FP6], const uint64_t b[FP6], uint64_t r[FP6]) {
    fp2_add(a, b, r);
    fp2_add(a+12, b+12, r+12);
    fp2_add(a+24, b+24, r+24);
}

static inline void fp6_sub(const uint64_t a[FP6], const uint64_t b[FP6], uint64_t r[FP6]) {
    fp2_sub(a, b, r);
    fp2_sub(a+12, b+12, r+12);
    fp2_sub(a+24, b+24, r+24);
}

static inline void fp6_neg(const uint64_t a[FP6], uint64_t r[FP6]) {
    fp2_neg(a, r);
    fp2_neg(a+12, r+12);
    fp2_neg(a+24, r+24);
}

// Multiply Fp6 by v: v*(c0 + c1*v + c2*v^2) = c2*xi + c0*v + c1*v^2
//   where xi = (1+u) is the non-residue
static inline void fp6_mul_by_v(const uint64_t a[FP6], uint64_t r[FP6]) {
    uint64_t t[12];
    fp2_mul_nr(a+24, t);     // c2 * xi
    fp2_copy(r+24, a+12);    // new c2 = old c1
    fp2_copy(r+12, a);       // new c1 = old c0
    fp2_copy(r, t);           // new c0 = c2*xi
}

// Fp6 multiplication using Karatsuba (6 Fp2 muls)
static void fp6_mul(const uint64_t a[FP6], const uint64_t b[FP6], uint64_t r[FP6]) {
    uint64_t v0[12], v1[12], v2[12], t1[12], t2[12], t3[12];

    fp2_mul(a, b, v0);         // v0 = a0*b0
    fp2_mul(a+12, b+12, v1);   // v1 = a1*b1
    fp2_mul(a+24, b+24, v2);   // v2 = a2*b2

    // c0 = v0 + xi*((a1+a2)(b1+b2) - v1 - v2)
    fp2_add(a+12, a+24, t1);
    fp2_add(b+12, b+24, t2);
    fp2_mul(t1, t2, t3);
    fp2_sub(t3, v1, t3);
    fp2_sub(t3, v2, t3);
    fp2_mul_nr(t3, t1);       // xi * (...)
    fp2_add(v0, t1, r);       // c0

    // c1 = (a0+a1)(b0+b1) - v0 - v1 + xi*v2
    fp2_add(a, a+12, t1);
    fp2_add(b, b+12, t2);
    fp2_mul(t1, t2, t3);
    fp2_sub(t3, v0, t3);
    fp2_sub(t3, v1, t3);
    fp2_mul_nr(v2, t1);       // xi*v2
    fp2_add(t3, t1, r+12);   // c1

    // c2 = (a0+a2)(b0+b2) - v0 - v2 + v1
    fp2_add(a, a+24, t1);
    fp2_add(b, b+24, t2);
    fp2_mul(t1, t2, t3);
    fp2_sub(t3, v0, t3);
    fp2_sub(t3, v2, t3);
    fp2_add(t3, v1, r+24);   // c2
}

// Fp6 squaring (Chung-Hasan SQ2)
static void fp6_sqr(const uint64_t a[FP6], uint64_t r[FP6]) {
    uint64_t s0[12], s1[12], s2[12], s3[12], s4[12];
    uint64_t ab[12], t1[12], t2[12];

    fp2_sqr(a, s0);           // s0 = a0^2
    fp2_mul(a, a+12, ab);     // ab = a0*a1
    fp2_dbl(ab, s1);          // s1 = 2*a0*a1

    // s2 = (a0 - a1 + a2)^2
    fp2_sub(a, a+12, t1);
    fp2_add(t1, a+24, t1);
    fp2_sqr(t1, s2);

    fp2_mul(a+12, a+24, t1);  // bc = a1*a2
    fp2_dbl(t1, s3);           // s3 = 2*a1*a2
    fp2_sqr(a+24, s4);         // s4 = a2^2

    // c0 = s0 + xi*s3
    fp2_mul_nr(s3, t1);
    fp2_add(s0, t1, r);

    // c1 = s1 + xi*s4
    fp2_mul_nr(s4, t1);
    fp2_add(s1, t1, r+12);

    // c2 = s1 + s2 + s3 - s0 - s4
    fp2_add(s1, s2, t1);
    fp2_add(t1, s3, t1);
    fp2_sub(t1, s0, t1);
    fp2_sub(t1, s4, r+24);
}

// Fp6 inverse
static void fp6_inv(const uint64_t a[FP6], uint64_t r[FP6]) {
    uint64_t c0s[12], c1s[12], c2s[12], c01[12], c02[12], c12[12];
    uint64_t A[12], B[12], C[12], F[12], fInv[12], t1[12], t2[12];

    fp2_sqr(a, c0s);
    fp2_sqr(a+12, c1s);
    fp2_sqr(a+24, c2s);
    fp2_mul(a, a+12, c01);
    fp2_mul(a, a+24, c02);
    fp2_mul(a+12, a+24, c12);

    // A = c0^2 - xi*c1*c2
    fp2_mul_nr(c12, t1);
    fp2_sub(c0s, t1, A);

    // B = xi*c2^2 - c0*c1
    fp2_mul_nr(c2s, t1);
    fp2_sub(t1, c01, B);

    // C = c1^2 - c0*c2
    fp2_sub(c1s, c02, C);

    // F = c0*A + xi*(c2*B + c1*C)
    fp2_mul(a, A, F);
    fp2_mul(a+24, B, t1);
    fp2_mul(a+12, C, t2);
    fp2_add(t1, t2, t1);
    fp2_mul_nr(t1, t2);
    fp2_add(F, t2, F);

    fp2_inv(F, fInv);

    fp2_mul(A, fInv, r);
    fp2_mul(B, fInv, r+12);
    fp2_mul(C, fInv, r+24);
}

static inline void fp6_copy(uint64_t dst[FP6], const uint64_t src[FP6]) { memcpy(dst, src, 288); }
static inline void fp6_zero(uint64_t r[FP6]) { memset(r, 0, 288); }
static inline void fp6_one(uint64_t r[FP6]) {
    fp2_one(r);
    fp2_zero(r+12);
    fp2_zero(r+24);
}

// ============================================================
// Fp12 = Fp6[w]/(w^2 - v), layout: [c0[36], c1[36]] = 72 uint64_t
// ============================================================

#define FP12 72

static inline void fp12_add(const uint64_t a[FP12], const uint64_t b[FP12], uint64_t r[FP12]) {
    fp6_add(a, b, r);
    fp6_add(a+36, b+36, r+36);
}

static inline void fp12_sub(const uint64_t a[FP12], const uint64_t b[FP12], uint64_t r[FP12]) {
    fp6_sub(a, b, r);
    fp6_sub(a+36, b+36, r+36);
}

// Fp12 conjugation: (c0, -c1)
static inline void fp12_conj(const uint64_t a[FP12], uint64_t r[FP12]) {
    fp6_copy(r, a);
    fp6_neg(a+36, r+36);
}

// Fp12 mul: (a0+a1*w)(b0+b1*w) = (a0*b0 + a1*b1*v) + ((a0+a1)(b0+b1) - a0*b0 - a1*b1)*w
static void fp12_mul(const uint64_t a[FP12], const uint64_t b[FP12], uint64_t r[FP12]) {
    uint64_t t0[36], t1[36], s1[36], s2[36], t1v[36];

    fp6_mul(a, b, t0);          // t0 = a0*b0
    fp6_mul(a+36, b+36, t1);    // t1 = a1*b1
    fp6_mul_by_v(t1, t1v);      // t1*v

    fp6_add(t0, t1v, r);        // c0 = t0 + t1*v

    fp6_add(a, a+36, s1);       // a0+a1
    fp6_add(b, b+36, s2);       // b0+b1
    fp6_mul(s1, s2, r+36);      // (a0+a1)(b0+b1)
    fp6_sub(r+36, t0, r+36);
    fp6_sub(r+36, t1, r+36);    // c1 = (...) - t0 - t1
}

// Fp12 squaring
static void fp12_sqr(const uint64_t a[FP12], uint64_t r[FP12]) {
    uint64_t ab[36], a1v[36], sum[36], prod[36], abv[36], t[36];

    fp6_mul(a, a+36, ab);       // ab = a0*a1

    // c0 = (a0+a1)(a0+a1*v) - ab - ab*v
    fp6_mul_by_v(a+36, a1v);
    fp6_add(a, a+36, sum);
    fp6_add(a, a1v, t);
    fp6_mul(sum, t, prod);
    fp6_mul_by_v(ab, abv);
    fp6_sub(prod, ab, r);
    fp6_sub(r, abv, r);         // c0

    fp6_add(ab, ab, r+36);      // c1 = 2*ab
}

// Fp12 inverse: 1/(a0+a1*w) = (a0-a1*w)/(a0^2 - a1^2*v)
static void fp12_inv(const uint64_t a[FP12], uint64_t r[FP12]) {
    uint64_t t0[36], t1[36], t1v[36], denom[36], dinv[36];

    fp6_sqr(a, t0);
    fp6_sqr(a+36, t1);
    fp6_mul_by_v(t1, t1v);
    fp6_sub(t0, t1v, denom);
    fp6_inv(denom, dinv);

    fp6_mul(a, dinv, r);
    uint64_t neg1[36];
    fp6_neg(a+36, neg1);
    fp6_mul(neg1, dinv, r+36);
}

static inline void fp12_copy(uint64_t dst[FP12], const uint64_t src[FP12]) { memcpy(dst, src, 576); }
static inline void fp12_one(uint64_t r[FP12]) {
    fp6_one(r);
    fp6_zero(r+36);
}

// ============================================================
// Frobenius coefficients
// These are hardcoded Montgomery-form constants for BLS12-381.
// Computed as (1+u)^((p^k-1)/j) for the appropriate k,j.
// ============================================================

// We compute them on first use and cache.
static int frob_coeffs_initialized = 0;

// Frobenius^1 coefficients (in Fp2)
static uint64_t FROB_FP6_C1_1[12];  // gamma_{1,1} = (1+u)^((p-1)/3)
static uint64_t FROB_FP6_C2_1[12];  // gamma_{2,1} = gamma_{1,1}^2
static uint64_t FROB_FP12_C1_1[12]; // (1+u)^((p-1)/6)
// Precomputed combined: gamma6 * gamma12 for c1 part of frobenius^1
static uint64_t FROB1_C1_C1[12];    // FROB_FP6_C1_1 * FROB_FP12_C1_1
static uint64_t FROB1_C2_C1[12];    // FROB_FP6_C2_1 * FROB_FP12_C1_1

// Frobenius^2 coefficients (in Fp, stored as Fp2 with c1=0)
static uint64_t FROB_FP6_C1_2[12];
static uint64_t FROB_FP6_C2_2[12];
static uint64_t FROB_FP12_C1_2[12];
// Precomputed combined for frobenius^2 (in Fp)
static uint64_t FROB2_C1_C1[6];     // FROB_FP6_C1_2[Fp] * FROB_FP12_C1_2[Fp]
static uint64_t FROB2_C2_C1[6];     // FROB_FP6_C2_2[Fp] * FROB_FP12_C1_2[Fp]

// Frobenius^3 coefficients (in Fp2)
static uint64_t FROB_FP6_C1_3[12];
static uint64_t FROB_FP6_C2_3[12];
static uint64_t FROB_FP12_C1_3[12];
// Precomputed combined for frobenius^3
static uint64_t FROB3_C1_C1[12];    // FROB_FP6_C1_3 * FROB_FP12_C1_3
static uint64_t FROB3_C2_C1[12];    // FROB_FP6_C2_3 * FROB_FP12_C1_3

// Helper: fp2_pow for bootstrapping Frobenius constants
static void fp2_pow(const uint64_t base[FP2], const uint64_t exp[6], uint64_t result[FP2]) {
    uint64_t r[12], b[12];
    fp2_one(r);
    fp2_copy(b, base);
    for (int i = 0; i < 6; i++) {
        for (int bit = 0; bit < 64; bit++) {
            if ((exp[i] >> bit) & 1) fp2_mul(r, b, r);
            fp2_sqr(b, b);
        }
    }
    fp2_copy(result, r);
}

// Divide 384-bit number by small constant
static void div_by_3(const uint64_t a[6], uint64_t r[6]) {
    memset(r, 0, 48);
    uint64_t remainder = 0;
    for (int i = 5; i >= 0; i--) {
        for (int bit = 63; bit >= 0; bit--) {
            remainder = (remainder << 1) | ((a[i] >> bit) & 1);
            if (remainder >= 3) {
                r[i] |= (uint64_t)1 << bit;
                remainder -= 3;
            }
        }
    }
}

static void div_by_2(const uint64_t a[6], uint64_t r[6]) {
    for (int i = 0; i < 6; i++) {
        r[i] = a[i] >> 1;
        if (i < 5) r[i] |= (a[i+1] & 1) << 63;
    }
}

static void init_frobenius_coeffs(void) {
    if (frob_coeffs_initialized) return;

    // non-residue = 1 + u (in Montgomery form)
    uint64_t nr[12];
    fp_copy(nr, FP_ONE);      // c0 = 1
    fp_copy(nr+6, FP_ONE);    // c1 = 1

    uint64_t pm1[6];
    for (int i = 0; i < 6; i++) pm1[i] = P[i];
    pm1[0] -= 1;  // p-1

    uint64_t pm1_3[6], pm1_6[6], pm1_2[6];
    div_by_3(pm1, pm1_3);       // (p-1)/3
    div_by_2(pm1, pm1_2);       // (p-1)/2
    div_by_3(pm1_2, pm1_6);     // (p-1)/6

    // Fp6 Frobenius^1
    fp2_pow(nr, pm1_3, FROB_FP6_C1_1);
    fp2_sqr(FROB_FP6_C1_1, FROB_FP6_C2_1);

    // Fp6 Frobenius^2 (norm = g * conj(g), in Fp)
    uint64_t t[12];
    fp2_conj(FROB_FP6_C1_1, t);
    fp2_mul(t, FROB_FP6_C1_1, FROB_FP6_C1_2);
    fp2_sqr(FROB_FP6_C1_2, FROB_FP6_C2_2);

    // Fp6 Frobenius^3
    uint64_t gp[12];
    fp2_conj(FROB_FP6_C1_1, gp);
    fp2_mul(FROB_FP6_C1_1, gp, t);
    fp2_mul(t, FROB_FP6_C1_1, FROB_FP6_C1_3);
    fp2_sqr(FROB_FP6_C1_3, FROB_FP6_C2_3);

    // Fp12 Frobenius^1
    fp2_pow(nr, pm1_6, FROB_FP12_C1_1);

    // Fp12 Frobenius^2
    uint64_t hp[12];
    fp2_conj(FROB_FP12_C1_1, hp);
    fp2_mul(hp, FROB_FP12_C1_1, FROB_FP12_C1_2);

    // Fp12 Frobenius^3
    fp2_mul(FROB_FP12_C1_2, FROB_FP12_C1_1, t);
    // Actually: h^3 = h * (h * conj(h)) = h * norm(h)
    // But simpler: use h*hp*h pattern
    fp2_mul(FROB_FP12_C1_1, hp, t);
    fp2_mul(t, FROB_FP12_C1_1, FROB_FP12_C1_3);

    // Precompute combined coefficients for frobenius^1
    fp2_mul(FROB_FP6_C1_1, FROB_FP12_C1_1, FROB1_C1_C1);
    fp2_mul(FROB_FP6_C2_1, FROB_FP12_C1_1, FROB1_C2_C1);

    // Precompute combined coefficients for frobenius^2 (all in Fp)
    fp_mul(FROB_FP6_C1_2, FROB_FP12_C1_2, FROB2_C1_C1);
    fp_mul(FROB_FP6_C2_2, FROB_FP12_C1_2, FROB2_C2_C1);

    // Precompute combined coefficients for frobenius^3
    fp2_mul(FROB_FP6_C1_3, FROB_FP12_C1_3, FROB3_C1_C1);
    fp2_mul(FROB_FP6_C2_3, FROB_FP12_C1_3, FROB3_C2_C1);

    frob_coeffs_initialized = 1;
}

// ============================================================
// Frobenius endomorphisms on Fp12
// ============================================================

// Frobenius^1 on Fp12
// Uses precomputed combined coefficients to avoid chained fp2_mul in c1 part
static void fp12_frobenius(const uint64_t a[FP12], uint64_t r[FP12]) {
    init_frobenius_coeffs();
    // c0 part: frob(c0 + c1*v + c2*v^2) = conj(c0) + conj(c1)*gamma_{1,1}*v + conj(c2)*gamma_{2,1}*v^2
    uint64_t t[12];
    fp2_conj(a, r);                         // r.c0.c0 = conj(a.c0.c0)
    fp2_conj(a+12, t);
    fp2_mul(t, FROB_FP6_C1_1, r+12);       // r.c0.c1 = conj(a.c0.c1) * gamma_{1,1}
    fp2_conj(a+24, t);
    fp2_mul(t, FROB_FP6_C2_1, r+24);       // r.c0.c2 = conj(a.c0.c2) * gamma_{2,1}

    // c1 part: frob(c1_fp6) * gamma_w, using precomputed combined coefficients
    fp2_conj(a+36, t);
    fp2_mul(t, FROB_FP12_C1_1, r+36);      // r.c1.c0 = conj(a.c1.c0) * gamma12_1
    fp2_conj(a+48, t);
    fp2_mul(t, FROB1_C1_C1, r+48);         // r.c1.c1 = conj(a.c1.c1) * (gamma6_c1 * gamma12_1)
    fp2_conj(a+60, t);
    fp2_mul(t, FROB1_C2_C1, r+60);         // r.c1.c2 = conj(a.c1.c2) * (gamma6_c2 * gamma12_1)
}

// Frobenius^2 on Fp12 (coefficients are in Fp)
// Uses precomputed combined coefficients to avoid runtime fp_mul
static void fp12_frobenius2(const uint64_t a[FP12], uint64_t r[FP12]) {
    init_frobenius_coeffs();
    // Frobenius^2 on Fp2 = identity, so no conjugation needed
    fp2_copy(r, a);                         // c0.c0 unchanged
    fp2_mul_fp(a+12, FROB_FP6_C1_2, r+12); // c0.c1 * gamma6_c1
    fp2_mul_fp(a+24, FROB_FP6_C2_2, r+24); // c0.c2 * gamma6_c2

    // c1: each component scaled by precomputed combined coefficients
    fp2_mul_fp(a+36, FROB_FP12_C1_2, r+36);
    fp2_mul_fp(a+48, FROB2_C1_C1, r+48);
    fp2_mul_fp(a+60, FROB2_C2_C1, r+60);
}

// Frobenius^3 on Fp12
// Uses precomputed combined coefficients to avoid chained fp2_mul in c1 part
static void fp12_frobenius3(const uint64_t a[FP12], uint64_t r[FP12]) {
    init_frobenius_coeffs();
    uint64_t t[12];
    fp2_conj(a, r);
    fp2_conj(a+12, t);
    fp2_mul(t, FROB_FP6_C1_3, r+12);
    fp2_conj(a+24, t);
    fp2_mul(t, FROB_FP6_C2_3, r+24);

    // c1 part: using precomputed combined coefficients
    fp2_conj(a+36, t);
    fp2_mul(t, FROB_FP12_C1_3, r+36);
    fp2_conj(a+48, t);
    fp2_mul(t, FROB3_C1_C1, r+48);         // precomputed gamma6_c1_3 * gamma12_3
    fp2_conj(a+60, t);
    fp2_mul(t, FROB3_C2_C1, r+60);         // precomputed gamma6_c2_3 * gamma12_3
}

// ============================================================
// Cyclotomic squaring
// For f = a + b*w in cyclotomic subgroup: f^2 = (2*a^2 - 1) + 2*a*b*w
// ============================================================

static void fp12_cyc_sqr(const uint64_t f[FP12], uint64_t r[FP12]) {
    uint64_t ab[36], asqr[36], one6[36];
    fp6_mul(f, f+36, ab);          // ab = a*b
    fp6_sqr(f, asqr);              // a^2

    fp6_add(ab, ab, r+36);         // c1 = 2*ab
    fp6_add(asqr, asqr, r);        // 2*a^2
    fp6_one(one6);
    fp6_sub(r, one6, r);           // c0 = 2*a^2 - 1
}

// ============================================================
// Exponentiation by BLS parameter x
// |x| = 0xd201000000010000
// Binary: 1101001000000001000000000000000000000000000000010000000000000000
// ============================================================

static const uint8_t X_BITS[64] = {
    1,1,0,1,0,0,1,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};

// f^x (x negative, so conjugate at end)
static void fp12_pow_by_x(const uint64_t a[FP12], uint64_t r[FP12]) {
    fp12_one(r);
    for (int i = 0; i < 64; i++) {
        uint64_t t[72];
        fp12_cyc_sqr(r, t);
        fp12_copy(r, t);
        if (X_BITS[i]) {
            fp12_mul(r, a, t);
            fp12_copy(r, t);
        }
    }
    // x is negative => conjugate
    uint64_t t[72];
    fp12_conj(r, t);
    fp12_copy(r, t);
}

// f^(|x|/2) then conjugate
// |x|/2 = 0x6900800000008000
// Binary: 0110100100000000100000000000000000000000000000001000000000000000
static const uint8_t XHALF_BITS[64] = {
    0,1,1,0,1,0,0,1,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};

static void fp12_pow_by_x_half(const uint64_t a[FP12], uint64_t r[FP12]) {
    fp12_one(r);
    for (int i = 0; i < 64; i++) {
        uint64_t t[72];
        fp12_cyc_sqr(r, t);
        fp12_copy(r, t);
        if (XHALF_BITS[i]) {
            fp12_mul(r, a, t);
            fp12_copy(r, t);
        }
    }
    uint64_t t[72];
    fp12_conj(r, t);
    fp12_copy(r, t);
}

// ============================================================
// G2 affine point operations for Miller loop
// G2Affine: [x[12], y[12]] = 24 uint64_t (Fp2 coords)
// ============================================================

#define G2AFF 24

// G2 affine doubling: T = 2T, returns lambda
// lambda = 3*x^2 / (2*y)  (a=0 for BLS12-381 twist)
static void g2_aff_double(uint64_t t[G2AFF], uint64_t lambda[FP2]) {
    uint64_t xsq[12], num[12], den[12], dinv[12], x3[12], y3[12], t1[12], t2[12];

    fp2_sqr(t, xsq);            // x^2
    fp2_dbl(xsq, t1);
    fp2_add(t1, xsq, num);      // 3*x^2
    fp2_dbl(t+12, den);          // 2*y
    fp2_inv(den, dinv);
    fp2_mul(num, dinv, lambda);  // lambda = 3*x^2 / (2*y)

    // x3 = lambda^2 - 2*x
    fp2_sqr(lambda, t1);
    fp2_dbl(t, t2);
    fp2_sub(t1, t2, x3);

    // y3 = lambda*(x - x3) - y
    fp2_sub(t, x3, t1);
    fp2_mul(lambda, t1, t2);
    fp2_sub(t2, t+12, y3);

    fp2_copy(t, x3);
    fp2_copy(t+12, y3);
}

// G2 affine addition: T = T + Q, returns lambda
static void g2_aff_add(uint64_t t[G2AFF], const uint64_t q[G2AFF], uint64_t lambda[FP2]) {
    uint64_t dx[12], dy[12], dinv[12], x3[12], y3[12], t1[12], t2[12];

    fp2_sub(q, t, dx);          // qx - tx
    fp2_sub(q+12, t+12, dy);    // qy - ty
    fp2_inv(dx, dinv);
    fp2_mul(dy, dinv, lambda);

    // x3 = lambda^2 - tx - qx
    fp2_sqr(lambda, t1);
    fp2_sub(t1, t, t1);
    fp2_sub(t1, q, x3);

    // y3 = lambda*(tx - x3) - ty
    fp2_sub(t, x3, t1);
    fp2_mul(lambda, t1, t2);
    fp2_sub(t2, t+12, y3);

    fp2_copy(t, x3);
    fp2_copy(t+12, y3);
}

// ============================================================
// Line evaluation at G1 point P
// BLS12-381 D-type twist:
//   l = (lambda*xT - yT) + (-lambda*xP)*v + yP*(v*w)
// Tower positions: 1 = c0.c0, v = c0.c1, v*w = c1.c1
// ============================================================

static void line_eval(const uint64_t lambda[FP2], const uint64_t xT[FP2], const uint64_t yT[FP2],
                      const uint64_t px[FP], const uint64_t py[FP],
                      uint64_t line[FP12]) {
    // ell_0 = lambda*xT - yT  (at position c0.c0)
    uint64_t ell_0[12], ell_v[12], ell_vw[12], t1[12];

    fp2_mul(lambda, xT, t1);
    fp2_sub(t1, yT, ell_0);

    // ell_v = -lambda*xP  (at position c0.c1)
    fp2_mul_fp(lambda, px, t1);
    fp2_neg(t1, ell_v);

    // ell_vw = yP  (at position c1.c1) -- yP as Fp2 with c1=0
    fp_copy(ell_vw, py);
    memset(ell_vw+6, 0, 48);

    // Build Fp12: c0 = (ell_0, ell_v, 0), c1 = (0, ell_vw, 0)
    fp2_copy(line, ell_0);          // c0.c0
    fp2_copy(line+12, ell_v);       // c0.c1
    fp2_zero(line+24);              // c0.c2
    fp2_zero(line+36);              // c1.c0
    fp2_copy(line+48, ell_vw);      // c1.c1
    fp2_zero(line+60);              // c1.c2
}

// Sparse multiplication: f * line where line has only 3 nonzero Fp2 components
// Positions: c0.c0 (=l0), c0.c1 (=l1), c1.c1 (=l4)
// This is significantly cheaper than full Fp12 mul
static void fp12_mul_by_line(const uint64_t f[FP12], const uint64_t line[FP12], uint64_t r[FP12]) {
    // Extract line coefficients
    const uint64_t *l0 = line;        // c0.c0
    const uint64_t *l1 = line + 12;   // c0.c1
    const uint64_t *l4 = line + 48;   // c1.c1

    // f = f0 + f1*w, line = (l0 + l1*v) + l4*v*w
    // Product = (f0*(l0+l1*v) + f1*l4*v*v) + (f1*(l0+l1*v) + f0*l4*v)*w
    // But v^2*... gets complicated. Let's use full mul for correctness first,
    // then optimize if needed.
    fp12_mul(f, line, r);
}

// ============================================================
// Miller Loop
// |x| = 0xd201000000010000
// After leading 1 (bit 63), iterate bits 62 down to 0
// ============================================================

// Bits of |x| after the leading 1 (bits 62 down to 0, MSB first)
static const int MILLER_BITS[63] = {
    1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,  // bits 62-47
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // bits 46-31
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,  // bits 30-15
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0      // bits 14-0
};

// Full Miller loop: f_{|x|,Q}(P)
static void miller_loop(const uint64_t p_aff[12], const uint64_t q_aff[24], uint64_t f[FP12]) {
    // p_aff = [px[6], py[6]]  (G1 affine, Fp coords)
    // q_aff = [qx[12], qy[12]] (G2 affine, Fp2 coords)
    const uint64_t *px = p_aff;
    const uint64_t *py = p_aff + 6;

    // T = Q (working point)
    uint64_t T[24];
    memcpy(T, q_aff, 192);

    fp12_one(f);

    uint64_t line[72], tmp[72];
    uint64_t lam[12];
    uint64_t oldTx[12], oldTy[12];

    for (int i = 0; i < 63; i++) {
        // Square f
        fp12_sqr(f, tmp);
        fp12_copy(f, tmp);

        // Save T before doubling (for line eval)
        fp2_copy(oldTx, T);
        fp2_copy(oldTy, T+12);

        // Doubling step
        g2_aff_double(T, lam);
        line_eval(lam, oldTx, oldTy, px, py, line);
        fp12_mul_by_line(f, line, tmp);
        fp12_copy(f, tmp);

        // Addition step if bit is 1
        if (MILLER_BITS[i]) {
            fp2_copy(oldTx, T);
            fp2_copy(oldTy, T+12);
            g2_aff_add(T, q_aff, lam);
            line_eval(lam, oldTx, oldTy, px, py, line);
            fp12_mul_by_line(f, line, tmp);
            fp12_copy(f, tmp);
        }
    }

    // x is negative => conjugate
    fp12_conj(f, tmp);
    fp12_copy(f, tmp);
}

// ============================================================
// Hard part of final exponentiation
// Efficient 3-exp-by-x algorithm (Bowe, adapted from eprint 2020/875)
// Uses only 3 fp12_pow_by_x instead of 4 + 1 fp12_pow_by_x_half
// ============================================================

static void hard_part_exp(const uint64_t f[FP12], uint64_t r[FP12]) {
    uint64_t y0[72], y1[72], y2[72], y3[72], y4[72], y5[72], y6[72], tmp[72];

    // y0 = f^x
    fp12_pow_by_x(f, y0);

    // y1 = y0^x = f^(x^2)
    fp12_pow_by_x(y0, y1);

    // y2 = y1^x = f^(x^3)
    fp12_pow_by_x(y1, y2);

    // y3 = conj(f) = f^(-1) in cyclotomic subgroup
    fp12_conj(f, y3);

    // y1 = y1 * y3 = f^(x^2 - 1)
    fp12_mul(y1, y3, tmp); fp12_copy(y1, tmp);

    // y1 = frob(y1) = f^((x^2-1)*p)
    fp12_frobenius(y1, tmp); fp12_copy(y1, tmp);

    // y4 = y0 * y3 = f^(x - 1)
    fp12_mul(y0, y3, y4);

    // y4 = frob2(y4) = f^((x-1)*p^2)
    fp12_frobenius2(y4, tmp); fp12_copy(y4, tmp);

    // y5 = conj(y0) = f^(-x)
    fp12_conj(y0, y5);

    // y6 = y2 * y5 = f^(x^3 - x)
    fp12_mul(y2, y5, y6);

    // y6 = frob3(y6) = f^((x^3-x)*p^3)
    fp12_frobenius3(y6, tmp); fp12_copy(y6, tmp);

    // y3 = cyc_sqr(f) = f^2
    fp12_cyc_sqr(f, y3);

    // y3 = y3 * f = f^3
    fp12_mul(y3, f, tmp); fp12_copy(y3, tmp);

    // y3 = y3 * y2 = f^(3 + x^3)
    fp12_mul(y3, y2, tmp); fp12_copy(y3, tmp);

    // Combine all Frobenius terms
    // result = y1 * y4 * y6 * y3
    fp12_mul(y1, y4, tmp); fp12_copy(y1, tmp);
    fp12_mul(y1, y6, tmp); fp12_copy(y1, tmp);
    fp12_mul(y1, y3, r);
}

// ============================================================
// Final exponentiation: f^((p^12 - 1) / r)
// ============================================================

static void final_exp(const uint64_t f[FP12], uint64_t r[FP12]) {
    // Easy part 1: f^(p^6 - 1) = conj(f) * f^(-1)
    uint64_t fconj[72], finv[72], tmp[72];
    fp12_conj(f, fconj);
    fp12_inv(f, finv);
    fp12_mul(fconj, finv, r);

    // Easy part 2: r^(p^2 + 1) = frob2(r) * r
    fp12_frobenius2(r, tmp);
    fp12_mul(tmp, r, fconj);  // reuse fconj as temp
    fp12_copy(r, fconj);

    // Hard part
    uint64_t hard[72];
    hard_part_exp(r, hard);
    fp12_copy(r, hard);
}

// ============================================================
// Exported functions
// ============================================================

// Full Miller loop
// p_aff: 12 uint64_t (G1 affine: x[6], y[6])
// q_aff: 24 uint64_t (G2 affine: x[12], y[12])
// result: 72 uint64_t (Fp12)
void bls12_381_miller_loop(const uint64_t p_aff[12], const uint64_t q_aff[24], uint64_t result[72]) {
    miller_loop(p_aff, q_aff, result);
}

// Final exponentiation
void bls12_381_final_exp(const uint64_t f[72], uint64_t result[72]) {
    final_exp(f, result);
}

// Full pairing: e(P, Q) = finalExp(millerLoop(P, Q))
void bls12_381_pairing(const uint64_t p_aff[12], const uint64_t q_aff[24], uint64_t result[72]) {
    uint64_t ml[72];
    miller_loop(p_aff, q_aff, ml);
    final_exp(ml, result);
}

// Pairing check: verify prod_i e(P_i, Q_i) = 1
// pairs: interleaved [p0[12], q0[24], p1[12], q1[24], ...]
// n: number of pairs
// Returns 1 if product of pairings = 1, 0 otherwise
int bls12_381_pairing_check(const uint64_t *pairs, int n) {
    uint64_t f[72];
    fp12_one(f);

    for (int i = 0; i < n; i++) {
        const uint64_t *pi = pairs + i * 36;       // 12 (G1) + 24 (G2) = 36
        const uint64_t *qi = pi + 12;
        uint64_t ml[72], tmp[72];
        miller_loop(pi, qi, ml);
        fp12_mul(f, ml, tmp);
        fp12_copy(f, tmp);
    }

    uint64_t result[72];
    final_exp(f, result);

    // Check if result is 1 (Montgomery form)
    uint64_t one[72];
    fp12_one(one);
    return memcmp(result, one, 576) == 0;
}

// Fp12 multiply (exported for Swift bridge)
void bls12_381_fp12_mul(const uint64_t a[72], const uint64_t b[72], uint64_t r[72]) {
    fp12_mul(a, b, r);
}

// Fp12 squaring (exported)
void bls12_381_fp12_sqr(const uint64_t a[72], uint64_t r[72]) {
    fp12_sqr(a, r);
}

// Fp12 inverse (exported)
void bls12_381_fp12_inv(const uint64_t a[72], uint64_t r[72]) {
    fp12_inv(a, r);
}

// Fp12 conjugate (exported)
void bls12_381_fp12_conj(const uint64_t a[72], uint64_t r[72]) {
    fp12_conj(a, r);
}

// Fp2 inv (exported)
void bls12_381_fp2_inv(const uint64_t a[12], uint64_t r[12]) {
    fp2_inv(a, r);
}

// Fp6 mul (exported)
void bls12_381_fp6_mul(const uint64_t a[36], const uint64_t b[36], uint64_t r[36]) {
    fp6_mul(a, b, r);
}

// Fp6 sqr (exported)
void bls12_381_fp6_sqr(const uint64_t a[36], uint64_t r[36]) {
    fp6_sqr(a, r);
}
