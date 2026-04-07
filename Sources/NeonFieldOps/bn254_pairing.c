// BN254 pairing: Fp2/Fp6/Fp12 tower arithmetic + Miller loop + final exponentiation
// C-accelerated implementation using __uint128_t CIOS for Fp base field
//
// Tower: Fp2 = Fp[u]/(u^2+1), Fp6 = Fp2[v]/(v^3 - xi) where xi=9+u, Fp12 = Fp6[w]/(w^2 - v)
// Pairing: optimal ate, loop parameter = 6x+2, x = 4965661367071055936
//
// Memory layout:
//   Fp   = 4 x uint64_t (Montgomery form, 256-bit)
//   Fp2  = 8 x uint64_t (c0[4], c1[4])
//   Fp6  = 24 x uint64_t (c0[8], c1[8], c2[8])
//   Fp12 = 48 x uint64_t (c0[24], c1[24])
//   G1Affine = 8 x uint64_t (x[4], y[4])
//   G2Affine = 16 x uint64_t (x[8], y[8])

#include "NeonFieldOps.h"
#include <string.h>
#include <stdlib.h>

typedef unsigned __int128 uint128_t;

// ============================================================
// Fp constants for BN254
// p = 21888242871839275222246405745257275088696311157297823662689037894645226208583
// ============================================================

static const uint64_t P[4] = {
    0x3c208c16d87cfd47ULL, 0x97816a916871ca8dULL,
    0xb85045b68181585dULL, 0x30644e72e131a029ULL
};
static const uint64_t P_INV = 0x87d20782e4866389ULL;  // -p^{-1} mod 2^64
static const uint64_t FP_ONE[4] = {  // R mod p (Montgomery form of 1)
    0xd35d438dc58f0d9dULL, 0x0a78eb28f5c70b3dULL,
    0x666ea36f7879462cULL, 0x0e0a77c19a07df2fULL
};

// Montgomery form of 9
static const uint64_t FP_NINE[4] = {
    0xf60647ce410d7ff7ULL, 0x2f3d6f4dd31bd011ULL,
    0x2943337e3940c6d1ULL, 0x1d9598e8a7e39857ULL
};

// ============================================================
// Fp arithmetic (CIOS Montgomery, 4 x 64-bit limbs)
// ============================================================

static inline void fp_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0;
    for (int i = 0; i < 4; i++) {
        uint128_t w; uint64_t c;
        w = (uint128_t)a[i] * b[0] + t0;
        t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[i] * b[1] + t1 + c;
        t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[i] * b[2] + t2 + c;
        t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[i] * b[3] + t3 + c;
        t3 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t4 += c;
        uint64_t m = t0 * P_INV;
        w = (uint128_t)m * P[0] + t0; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * P[1] + t1 + c; t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * P[2] + t2 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * P[3] + t3 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t3 = t4 + c; t4 = 0;
    }
    uint64_t borrow = 0, tmp[4]; uint128_t d;
    d = (uint128_t)t0 - P[0]; tmp[0] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)t1 - P[1] - borrow; tmp[1] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)t2 - P[2] - borrow; tmp[2] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)t3 - P[3] - borrow; tmp[3] = (uint64_t)d; borrow = (d >> 127) & 1;
    if (!borrow) memcpy(r, tmp, 32); else { r[0]=t0; r[1]=t1; r[2]=t2; r[3]=t3; }
}

static inline void fp_sqr(const uint64_t a[4], uint64_t r[4]) { fp_mul(a, a, r); }

static inline void fp_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t w; uint64_t c = 0;
    w = (uint128_t)a[0]+b[0]; r[0]=(uint64_t)w; c=(uint64_t)(w>>64);
    w = (uint128_t)a[1]+b[1]+c; r[1]=(uint64_t)w; c=(uint64_t)(w>>64);
    w = (uint128_t)a[2]+b[2]+c; r[2]=(uint64_t)w; c=(uint64_t)(w>>64);
    w = (uint128_t)a[3]+b[3]+c; r[3]=(uint64_t)w; c=(uint64_t)(w>>64);
    uint64_t borrow=0, tmp[4]; uint128_t d;
    d=(uint128_t)r[0]-P[0]; tmp[0]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[1]-P[1]-borrow; tmp[1]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[2]-P[2]-borrow; tmp[2]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[3]-P[3]-borrow; tmp[3]=(uint64_t)d; borrow=(d>>127)&1;
    if (c||!borrow) memcpy(r, tmp, 32);
}

static inline void fp_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t d; uint64_t borrow=0;
    d=(uint128_t)a[0]-b[0]; r[0]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[1]-b[1]-borrow; r[1]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[2]-b[2]-borrow; r[2]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[3]-b[3]-borrow; r[3]=(uint64_t)d; borrow=(d>>127)&1;
    if (borrow) {
        uint64_t c=0; uint128_t w;
        w=(uint128_t)r[0]+P[0]; r[0]=(uint64_t)w; c=(uint64_t)(w>>64);
        w=(uint128_t)r[1]+P[1]+c; r[1]=(uint64_t)w; c=(uint64_t)(w>>64);
        w=(uint128_t)r[2]+P[2]+c; r[2]=(uint64_t)w; c=(uint64_t)(w>>64);
        w=(uint128_t)r[3]+P[3]+c; r[3]=(uint64_t)w;
    }
}

static inline void fp_neg(const uint64_t a[4], uint64_t r[4]) {
    if ((a[0]|a[1]|a[2]|a[3]) == 0) { memset(r, 0, 32); return; }
    uint128_t d; uint64_t borrow=0;
    d=(uint128_t)P[0]-a[0]; r[0]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)P[1]-a[1]-borrow; r[1]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)P[2]-a[2]-borrow; r[2]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)P[3]-a[3]-borrow; r[3]=(uint64_t)d;
}

static inline void fp_dbl(const uint64_t a[4], uint64_t r[4]) { fp_add(a, a, r); }
static inline int fp_is_zero(const uint64_t a[4]) { return (a[0]|a[1]|a[2]|a[3]) == 0; }
static inline void fp_copy(uint64_t dst[4], const uint64_t src[4]) { memcpy(dst, src, 32); }

// Fp inversion via Fermat: a^(p-2) mod p
static void fp_inv(const uint64_t a[4], uint64_t r[4]) {
    uint64_t pm2[4] = { P[0]-2, P[1], P[2], P[3] };
    memcpy(r, FP_ONE, 32);
    uint64_t b[4]; memcpy(b, a, 32);
    for (int i = 0; i < 4; i++) {
        for (int bit = 0; bit < 64; bit++) {
            if ((pm2[i] >> bit) & 1) fp_mul(r, b, r);
            fp_sqr(b, b);
        }
    }
}

// ============================================================
// Fp2 = Fp[u]/(u^2+1), layout: [c0[4], c1[4]] = 8 uint64_t
// ============================================================

#define FP2 8
#define FP  4

static inline void fp2_add(const uint64_t a[FP2], const uint64_t b[FP2], uint64_t r[FP2]) {
    fp_add(a, b, r); fp_add(a+4, b+4, r+4);
}
static inline void fp2_sub(const uint64_t a[FP2], const uint64_t b[FP2], uint64_t r[FP2]) {
    fp_sub(a, b, r); fp_sub(a+4, b+4, r+4);
}
static inline void fp2_neg(const uint64_t a[FP2], uint64_t r[FP2]) {
    fp_neg(a, r); fp_neg(a+4, r+4);
}
static inline void fp2_dbl(const uint64_t a[FP2], uint64_t r[FP2]) {
    fp_dbl(a, r); fp_dbl(a+4, r+4);
}

// Fp2 mul: Karatsuba (3 Fp muls)
static inline void fp2_mul(const uint64_t a[FP2], const uint64_t b[FP2], uint64_t r[FP2]) {
    uint64_t a0b0[4], a1b1[4], t1[4], t2[4], t3[4];
    fp_mul(a, b, a0b0);       // a0*b0
    fp_mul(a+4, b+4, a1b1);   // a1*b1
    fp_add(a, a+4, t1);       // a0+a1
    fp_add(b, b+4, t2);       // b0+b1
    fp_mul(t1, t2, t3);       // (a0+a1)(b0+b1)
    fp_sub(a0b0, a1b1, r);    // c0 = a0*b0 - a1*b1
    fp_sub(t3, a0b0, r+4);
    fp_sub(r+4, a1b1, r+4);   // c1 = cross - a0*b0 - a1*b1
}

// Fp2 sqr: complex squaring
static inline void fp2_sqr(const uint64_t a[FP2], uint64_t r[FP2]) {
    uint64_t v0[4], sum[4], diff[4];
    fp_mul(a, a+4, v0);
    fp_add(a, a+4, sum);
    fp_sub(a, a+4, diff);
    fp_mul(sum, diff, r);     // c0 = (a0+a1)(a0-a1)
    fp_dbl(v0, r+4);          // c1 = 2*a0*a1
}

static inline void fp2_conj(const uint64_t a[FP2], uint64_t r[FP2]) {
    fp_copy(r, a); fp_neg(a+4, r+4);
}

static inline void fp2_mul_fp(const uint64_t a[FP2], const uint64_t s[FP], uint64_t r[FP2]) {
    fp_mul(a, s, r); fp_mul(a+4, s, r+4);
}

// Multiply by non-residue xi = 9+u: (a0+a1*u)(9+u) = (9*a0-a1) + (a0+9*a1)*u
static inline void fp2_mul_nr(const uint64_t a[FP2], uint64_t r[FP2]) {
    uint64_t t0[4], t1[4];
    fp_mul(a, FP_NINE, t0);     // 9*a0
    fp_sub(t0, a+4, r);         // c0 = 9*a0 - a1
    fp_mul(a+4, FP_NINE, t1);   // 9*a1
    fp_add(a, t1, r+4);         // c1 = a0 + 9*a1
}

static void fp2_inv(const uint64_t a[FP2], uint64_t r[FP2]) {
    uint64_t t0[4], t1[4], norm[4], ninv[4], neg1[4];
    fp_sqr(a, t0);
    fp_sqr(a+4, t1);
    fp_add(t0, t1, norm);       // norm = a0^2 + a1^2 (since u^2 = -1)
    fp_inv(norm, ninv);
    fp_mul(a, ninv, r);
    fp_neg(a+4, neg1);
    fp_mul(neg1, ninv, r+4);
}

static inline int fp2_is_zero(const uint64_t a[FP2]) {
    return fp_is_zero(a) && fp_is_zero(a+4);
}
static inline void fp2_copy(uint64_t dst[FP2], const uint64_t src[FP2]) { memcpy(dst, src, 64); }
static inline void fp2_zero(uint64_t r[FP2]) { memset(r, 0, 64); }
static inline void fp2_one(uint64_t r[FP2]) { memcpy(r, FP_ONE, 32); memset(r+4, 0, 32); }

// ============================================================
// Fp6 = Fp2[v]/(v^3 - xi), layout: [c0[8], c1[8], c2[8]] = 24 uint64_t
// ============================================================

#define FP6 24

static inline void fp6_add(const uint64_t a[FP6], const uint64_t b[FP6], uint64_t r[FP6]) {
    fp2_add(a, b, r); fp2_add(a+8, b+8, r+8); fp2_add(a+16, b+16, r+16);
}
static inline void fp6_sub(const uint64_t a[FP6], const uint64_t b[FP6], uint64_t r[FP6]) {
    fp2_sub(a, b, r); fp2_sub(a+8, b+8, r+8); fp2_sub(a+16, b+16, r+16);
}
static inline void fp6_neg(const uint64_t a[FP6], uint64_t r[FP6]) {
    fp2_neg(a, r); fp2_neg(a+8, r+8); fp2_neg(a+16, r+16);
}

// Multiply Fp6 by v: v*(c0 + c1*v + c2*v^2) = c2*xi + c0*v + c1*v^2
static inline void fp6_mul_by_v(const uint64_t a[FP6], uint64_t r[FP6]) {
    uint64_t t[8];
    fp2_mul_nr(a+16, t);     // c2 * xi
    fp2_copy(r+16, a+8);     // new c2 = old c1
    fp2_copy(r+8, a);        // new c1 = old c0
    fp2_copy(r, t);           // new c0 = c2*xi
}

// Fp6 multiplication (Karatsuba, 6 Fp2 muls)
static void fp6_mul(const uint64_t a[FP6], const uint64_t b[FP6], uint64_t r[FP6]) {
    uint64_t v0[8], v1[8], v2[8], t1[8], t2[8], t3[8];
    fp2_mul(a, b, v0);
    fp2_mul(a+8, b+8, v1);
    fp2_mul(a+16, b+16, v2);

    // c0 = v0 + xi*((a1+a2)(b1+b2) - v1 - v2)
    fp2_add(a+8, a+16, t1);
    fp2_add(b+8, b+16, t2);
    fp2_mul(t1, t2, t3);
    fp2_sub(t3, v1, t3);
    fp2_sub(t3, v2, t3);
    fp2_mul_nr(t3, t1);
    fp2_add(v0, t1, r);

    // c1 = (a0+a1)(b0+b1) - v0 - v1 + xi*v2
    fp2_add(a, a+8, t1);
    fp2_add(b, b+8, t2);
    fp2_mul(t1, t2, t3);
    fp2_sub(t3, v0, t3);
    fp2_sub(t3, v1, t3);
    fp2_mul_nr(v2, t1);
    fp2_add(t3, t1, r+8);

    // c2 = (a0+a2)(b0+b2) - v0 - v2 + v1
    fp2_add(a, a+16, t1);
    fp2_add(b, b+16, t2);
    fp2_mul(t1, t2, t3);
    fp2_sub(t3, v0, t3);
    fp2_sub(t3, v2, t3);
    fp2_add(t3, v1, r+16);
}

// Fp6 squaring (Chung-Hasan SQ2)
static void fp6_sqr(const uint64_t a[FP6], uint64_t r[FP6]) {
    uint64_t s0[8], s1[8], s2[8], s3[8], s4[8], ab[8], t1[8];

    fp2_sqr(a, s0);
    fp2_mul(a, a+8, ab);
    fp2_dbl(ab, s1);
    fp2_sub(a, a+8, t1); fp2_add(t1, a+16, t1); fp2_sqr(t1, s2);
    fp2_mul(a+8, a+16, t1); fp2_dbl(t1, s3);
    fp2_sqr(a+16, s4);

    fp2_mul_nr(s3, t1); fp2_add(s0, t1, r);
    fp2_mul_nr(s4, t1); fp2_add(s1, t1, r+8);
    fp2_add(s1, s2, t1); fp2_add(t1, s3, t1);
    fp2_sub(t1, s0, t1); fp2_sub(t1, s4, r+16);
}

// Fp6 inverse
static void fp6_inv(const uint64_t a[FP6], uint64_t r[FP6]) {
    uint64_t c0s[8], c1s[8], c2s[8], c01[8], c02[8], c12[8];
    uint64_t A[8], B[8], C[8], F[8], fInv[8], t1[8], t2[8];

    fp2_sqr(a, c0s); fp2_sqr(a+8, c1s); fp2_sqr(a+16, c2s);
    fp2_mul(a, a+8, c01); fp2_mul(a, a+16, c02); fp2_mul(a+8, a+16, c12);

    fp2_mul_nr(c12, t1); fp2_sub(c0s, t1, A);
    fp2_mul_nr(c2s, t1); fp2_sub(t1, c01, B);
    fp2_sub(c1s, c02, C);

    fp2_mul(a, A, F);
    fp2_mul(a+16, B, t1); fp2_mul(a+8, C, t2);
    fp2_add(t1, t2, t1); fp2_mul_nr(t1, t2);
    fp2_add(F, t2, F);

    fp2_inv(F, fInv);
    fp2_mul(A, fInv, r); fp2_mul(B, fInv, r+8); fp2_mul(C, fInv, r+16);
}

static inline void fp6_copy(uint64_t dst[FP6], const uint64_t src[FP6]) { memcpy(dst, src, 192); }
static inline void fp6_zero(uint64_t r[FP6]) { memset(r, 0, 192); }
static inline void fp6_one(uint64_t r[FP6]) { fp2_one(r); fp2_zero(r+8); fp2_zero(r+16); }

// ============================================================
// Fp12 = Fp6[w]/(w^2 - v), layout: [c0[24], c1[24]] = 48 uint64_t
// ============================================================

#define FP12 48

static inline void fp12_add(const uint64_t a[FP12], const uint64_t b[FP12], uint64_t r[FP12]) {
    fp6_add(a, b, r); fp6_add(a+24, b+24, r+24);
}
static inline void fp12_sub(const uint64_t a[FP12], const uint64_t b[FP12], uint64_t r[FP12]) {
    fp6_sub(a, b, r); fp6_sub(a+24, b+24, r+24);
}
static inline void fp12_conj(const uint64_t a[FP12], uint64_t r[FP12]) {
    fp6_copy(r, a); fp6_neg(a+24, r+24);
}

// Fp12 mul: (a0+a1*w)(b0+b1*w) = (a0*b0 + a1*b1*v) + ((a0+a1)(b0+b1) - a0*b0 - a1*b1)*w
static void fp12_mul(const uint64_t a[FP12], const uint64_t b[FP12], uint64_t r[FP12]) {
    uint64_t t0[24], t1[24], s1[24], s2[24], t1v[24];
    fp6_mul(a, b, t0);
    fp6_mul(a+24, b+24, t1);
    fp6_mul_by_v(t1, t1v);
    fp6_add(t0, t1v, r);
    fp6_add(a, a+24, s1);
    fp6_add(b, b+24, s2);
    fp6_mul(s1, s2, r+24);
    fp6_sub(r+24, t0, r+24);
    fp6_sub(r+24, t1, r+24);
}

// Fp12 squaring
static void fp12_sqr(const uint64_t a[FP12], uint64_t r[FP12]) {
    uint64_t ab[24], a1v[24], sum[24], prod[24], abv[24], t[24];
    fp6_mul(a, a+24, ab);
    fp6_mul_by_v(a+24, a1v);
    fp6_add(a, a+24, sum);
    fp6_add(a, a1v, t);
    fp6_mul(sum, t, prod);
    fp6_mul_by_v(ab, abv);
    fp6_sub(prod, ab, r);
    fp6_sub(r, abv, r);
    fp6_add(ab, ab, r+24);
}

// Fp12 inverse
static void fp12_inv(const uint64_t a[FP12], uint64_t r[FP12]) {
    uint64_t t0[24], t1[24], t1v[24], denom[24], dinv[24], neg1[24];
    fp6_sqr(a, t0); fp6_sqr(a+24, t1);
    fp6_mul_by_v(t1, t1v);
    fp6_sub(t0, t1v, denom);
    fp6_inv(denom, dinv);
    fp6_mul(a, dinv, r);
    fp6_neg(a+24, neg1);
    fp6_mul(neg1, dinv, r+24);
}

static inline void fp12_copy(uint64_t dst[FP12], const uint64_t src[FP12]) { memcpy(dst, src, 384); }
static inline void fp12_one(uint64_t r[FP12]) { fp6_one(r); fp6_zero(r+24); }

// ============================================================
// Frobenius coefficients for BN254
// Hardcoded Montgomery form constants (ALG6 optimization)
// All intermediate products precomputed to eliminate runtime fp2_sqr/fp2_mul
// ============================================================

// gamma_{1,1} = xi^((p-1)/6)
static const uint64_t GAMMA_1_1[8] = {
    0xaf9ba69633144907ULL, 0xca6b1d7387afb78aULL,
    0x11bded5ef08a2087ULL, 0x02f34d751a1f3a7cULL,
    0xa222ae234c492d72ULL, 0xd00f02a4565de15bULL,
    0xdc2ff3a253dfc926ULL, 0x10a75716b3899551ULL
};

// gamma_{1,2} = xi^((p-1)/3)
static const uint64_t GAMMA_1_2[8] = {
    0xb5773b104563ab30ULL, 0x347f91c8a9aa6454ULL,
    0x7a007127242e0991ULL, 0x1956bcd8118214ecULL,
    0x6e849f1ea0aa4757ULL, 0xaa1c7b6d89f89141ULL,
    0xb6e713cdfae0ca3aULL, 0x26694fbb4e82ebc3ULL
};

// gamma_{1,3} = xi^((p-1)/2)
static const uint64_t GAMMA_1_3[8] = {
    0xe4bbdd0c2936b629ULL, 0xbb30f162e133bacbULL,
    0x31a9d1b6f9645366ULL, 0x253570bea500f8ddULL,
    0xa1d77ce45ffe77c7ULL, 0x07affd117826d1dbULL,
    0x6d16bd27bb7edc6bULL, 0x2c87200285defeccULL
};

// Precomputed: gamma_{1,2}^2
static const uint64_t GAMMA_1_2_SQ[8] = {
    0x7361d77f843abe92ULL, 0xa5bb2bd3273411fbULL,
    0x9c941f314b3e2399ULL, 0x15df9cddbb9fd3ecULL,
    0x5dddfd154bd8c949ULL, 0x62cb29a5a4445b60ULL,
    0x37bc870a0c7dd2b9ULL, 0x24830a9d3171f0fdULL
};

// Precomputed: gamma_{1,1} * gamma_{1,2}^2
static const uint64_t GAMMA_1_1_MUL_1_2_SQ[8] = {
    0xc970692f41690fe7ULL, 0xe240342127694b0bULL,
    0x32bee66b83c459e8ULL, 0x12aabced0ab08841ULL,
    0x0d485d2340aebfa9ULL, 0x05193418ab2fcc57ULL,
    0xd3b0a40b8a4910f5ULL, 0x2f21ebb535d2925aULL
};

// gamma_{2,1} = xi^((p^2-1)/6) -- in Fp (c1=0)
static const uint64_t GAMMA_2_1[4] = {
    0xca8d800500fa1bf2ULL, 0xf0c5d61468b39769ULL,
    0x0e201271ad0d4418ULL, 0x04290f65bad856e6ULL
};

// gamma_{2,2} = xi^((p^2-1)/3)
static const uint64_t GAMMA_2_2[4] = {
    0x3350c88e13e80b9cULL, 0x7dce557cdb5e56b9ULL,
    0x6001b4b8b615564aULL, 0x2682e617020217e0ULL
};

// gamma_{2,3} = xi^((p^2-1)/2)
static const uint64_t GAMMA_2_3[4] = {
    0x68c3488912edefaaULL, 0x8d087f6872aabf4fULL,
    0x51e1a24709081231ULL, 0x2259d6b14729c0faULL
};

// Precomputed: gamma_{2,2}^2
static const uint64_t GAMMA_2_2_SQ[4] = {
    0x71930c11d782e155ULL, 0xa6bb947cffbe3323ULL,
    0xaa303344d4741444ULL, 0x2c3b3f0d26594943ULL
};

// Precomputed: gamma_{2,1} * gamma_{2,2}^2
static const uint64_t GAMMA_2_1_MUL_2_2_SQ[4] = {
    0x08cfc388c494f1abULL, 0x19b315148d1373d4ULL,
    0x584e90fdcb6c0213ULL, 0x09e1685bdf2f8849ULL
};

// Frobenius^3 precomputed constants (direct, not chained frob2+frob)
// gamma_{3,1} = gamma_{1,1} * gamma_{2,1}
static const uint64_t GAMMA_3_1[8] = {
    0x365316184e46d97dULL, 0x0af7129ed4c96d9fULL,
    0x659da72fca1009b5ULL, 0x08116d8983a20d23ULL,
    0xb1df4af7c39c1939ULL, 0x3d9f02878a73bf7fULL,
    0x9b2220928caf0ae0ULL, 0x26684515eff054a6ULL
};
// gamma_{3,2} = gamma_{1,2} * gamma_{2,2}
static const uint64_t GAMMA_3_2[8] = {
    0xc9af22f716ad6badULL, 0xb311782a4aa662b2ULL,
    0x19eeaf64e248c7f4ULL, 0x20273e77e3439f82ULL,
    0xacc02860f7ce93acULL, 0x3933d5817ba76b4cULL,
    0x69e6188b446c8467ULL, 0x0a46036d4417cc55ULL
};
// gamma_{3,3} = gamma_{1,3} * gamma_{2,3}
static const uint64_t GAMMA_3_3[8] = {
    0x5764af0aaf46471eULL, 0xdc50792e873e0fc1ULL,
    0x86a673ff881d04f6ULL, 0x0b2eddb43c30a74cULL,
    0x9a490f32787e8580ULL, 0x8fd16d7ff04af8b1ULL,
    0x4b39888ec6027bf2ULL, 0x03dd2e705b52a15dULL
};
// Precomputed: gamma_{3,2}^2
static const uint64_t GAMMA_3_2_SQ[8] = {
    0x448a93a57b6762dfULL, 0xbfd62df528fdeadfULL,
    0xd858f5d00e9bd47aULL, 0x06b03d4d3476ec58ULL,
    0x2b19daf4bcc936d1ULL, 0xa1a54e7a56f4299fULL,
    0xb533eee05adeaef1ULL, 0x170c812b84dda0b2ULL
};
// Precomputed: gamma_{3,1} * gamma_{3,2}^2
static const uint64_t GAMMA_3_1_MUL_3_2_SQ[8] = {
    0xe0bc4b2275cf559fULL, 0xc238b945c154e60fULL,
    0x803982a5929a7d5eULL, 0x15ce052df7e4a37eULL,
    0x2d28efbdbf3799a7ULL, 0x9b097e3c1ad60773ULL,
    0x982d4113af4a535bULL, 0x24e18991e3056063ULL
};

// ============================================================
// Frobenius endomorphisms on Fp12 (ALG6: all constants precomputed)
// ============================================================

// Frobenius^1: uses precomputed GAMMA_1_2_SQ and GAMMA_1_1_MUL_1_2_SQ
static void fp12_frobenius(const uint64_t a[FP12], uint64_t r[FP12]) {
    uint64_t t[8];
    // c0 part
    fp2_conj(a, r);                              // c0.c0 = conj(a.c0.c0)
    fp2_conj(a+8, t);
    fp2_mul(t, GAMMA_1_2, r+8);                 // c0.c1
    fp2_conj(a+16, t);
    fp2_mul(t, GAMMA_1_2_SQ, r+16);             // c0.c2

    // c1 part
    fp2_conj(a+24, t);
    fp2_mul(t, GAMMA_1_1, r+24);                // c1.c0
    fp2_conj(a+32, t);
    fp2_mul(t, GAMMA_1_3, r+32);                // c1.c1
    fp2_conj(a+40, t);
    fp2_mul(t, GAMMA_1_1_MUL_1_2_SQ, r+40);    // c1.c2
}

// Frobenius^2: uses precomputed GAMMA_2_2_SQ and GAMMA_2_1_MUL_2_2_SQ
static void fp12_frobenius2(const uint64_t a[FP12], uint64_t r[FP12]) {
    fp2_copy(r, a);                              // c0.c0 unchanged
    fp2_mul_fp(a+8, GAMMA_2_2, r+8);            // c0.c1
    fp2_mul_fp(a+16, GAMMA_2_2_SQ, r+16);       // c0.c2
    fp2_mul_fp(a+24, GAMMA_2_1, r+24);          // c1.c0
    fp2_mul_fp(a+32, GAMMA_2_3, r+32);          // c1.c1
    fp2_mul_fp(a+40, GAMMA_2_1_MUL_2_2_SQ, r+40); // c1.c2
}

// Frobenius^3: direct computation (not chained frob2+frob), saves ~50% work
static void fp12_frobenius3(const uint64_t a[FP12], uint64_t r[FP12]) {
    uint64_t t[8];
    // p^3 is odd, so Fp2 Frobenius = conjugation
    fp2_conj(a, r);                              // c0.c0
    fp2_conj(a+8, t);
    fp2_mul(t, GAMMA_3_2, r+8);                 // c0.c1
    fp2_conj(a+16, t);
    fp2_mul(t, GAMMA_3_2_SQ, r+16);             // c0.c2
    fp2_conj(a+24, t);
    fp2_mul(t, GAMMA_3_1, r+24);                // c1.c0
    fp2_conj(a+32, t);
    fp2_mul(t, GAMMA_3_3, r+32);                // c1.c1
    fp2_conj(a+40, t);
    fp2_mul(t, GAMMA_3_1_MUL_3_2_SQ, r+40);    // c1.c2
}

// ============================================================
// G2 affine point operations for Miller loop
// G2Affine: [x[8], y[8]] = 16 uint64_t (Fp2 coords)
// ============================================================

#define G2AFF 16

// G2 affine doubling: T = 2T, returns lambda
static void g2_aff_double(uint64_t t[G2AFF], uint64_t lambda[FP2]) {
    uint64_t xsq[8], num[8], den[8], dinv[8], x3[8], y3[8], t1[8], t2[8];
    fp2_sqr(t, xsq);
    fp2_dbl(xsq, t1);
    fp2_add(t1, xsq, num);       // 3*x^2
    fp2_dbl(t+8, den);           // 2*y
    fp2_inv(den, dinv);
    fp2_mul(num, dinv, lambda);
    fp2_sqr(lambda, t1);
    fp2_dbl(t, t2);
    fp2_sub(t1, t2, x3);         // x3 = lambda^2 - 2*x
    fp2_sub(t, x3, t1);
    fp2_mul(lambda, t1, t2);
    fp2_sub(t2, t+8, y3);        // y3 = lambda*(x - x3) - y
    fp2_copy(t, x3);
    fp2_copy(t+8, y3);
}

// G2 affine addition: T = T + Q, returns lambda
static void g2_aff_add(uint64_t t[G2AFF], const uint64_t q[G2AFF], uint64_t lambda[FP2]) {
    uint64_t dx[8], dy[8], dinv[8], x3[8], y3[8], t1[8], t2[8];
    fp2_sub(q, t, dx);
    fp2_sub(q+8, t+8, dy);
    fp2_inv(dx, dinv);
    fp2_mul(dy, dinv, lambda);
    fp2_sqr(lambda, t1);
    fp2_sub(t1, t, t1);
    fp2_sub(t1, q, x3);
    fp2_sub(t, x3, t1);
    fp2_mul(lambda, t1, t2);
    fp2_sub(t2, t+8, y3);
    fp2_copy(t, x3);
    fp2_copy(t+8, y3);
}

// ============================================================
// Line evaluation at G1 point P
// BN254 M-type twist:
//   yP at c0.c0, -lam*xP at c1.c0, (lam*xT - yT) at c1.c1
// Positions: c0.c0 (=l0), c1.c0 (=l3), c1.c1 (=l4)
// ============================================================

static void line_eval(const uint64_t lambda[FP2], const uint64_t xT[FP2], const uint64_t yT[FP2],
                      const uint64_t px[FP], const uint64_t py[FP],
                      uint64_t line[FP12]) {
    memset(line, 0, 384);

    // ell_0 = yP (as Fp2 with c1=0) at c0.c0
    fp_copy(line, py);
    // c0.c0.c1 is already 0

    // ell_3 = -lam*xP at c1.c0
    uint64_t t1[8];
    fp2_mul_fp(lambda, px, t1);
    fp2_neg(t1, line+24);

    // ell_4 = lam*xT - yT at c1.c1
    uint64_t t2[8];
    fp2_mul(lambda, xT, t2);
    fp2_sub(t2, yT, line+32);
}

// ============================================================
// Miller Loop
// NAF of 6x+2, x = 4965661367071055936 = 0x44E992B44A6909F1
// ============================================================

// NAF representation of 6x+2 (MSB first, 66 entries)
static const int8_t SIX_X_PLUS_2_NAF[66] = {
     1,  0, -1,  0,  1,  0,  0,  0, -1,  0, -1,  0,  0,  0, -1,  0,
     1,  0, -1,  0,  0, -1,  0,  0,  0,  0,  0,  1,  0,  0, -1,  0,
     1,  0,  0, -1,  0,  0,  0,  0, -1,  0,  1,  0,  0,  0, -1,  0,
    -1,  0,  0,  1,  0,  0,  0, -1,  0,  0, -1,  0,  1,  0,  1,  0,
     0,  0
};

// Frobenius correction constants for Q1, Q2
// gamma_{1,2} for Q1.x conjugation, gamma_{1,3} for Q1.y conjugation
// gamma_{2,2} for Q2.x (Fp scalar), gamma_{2,3} for Q2.y (Fp scalar)

static void miller_loop(const uint64_t p_aff[8], const uint64_t q_aff[16], uint64_t f[FP12]) {
    const uint64_t *px = p_aff;
    const uint64_t *py = p_aff + 4;

    uint64_t T[16];
    memcpy(T, q_aff, 128);

    // negQ
    uint64_t negQ[16];
    memcpy(negQ, q_aff, 64);        // x
    fp2_neg(q_aff + 8, negQ + 8);   // -y

    fp12_one(f);

    uint64_t line[48], tmp[48];
    uint64_t lam[8];
    uint64_t oldTx[8], oldTy[8];

    for (int i = 1; i < 66; i++) {
        fp12_sqr(f, tmp);
        fp12_copy(f, tmp);

        fp2_copy(oldTx, T);
        fp2_copy(oldTy, T+8);
        g2_aff_double(T, lam);
        line_eval(lam, oldTx, oldTy, px, py, line);
        fp12_mul(f, line, tmp);
        fp12_copy(f, tmp);

        if (SIX_X_PLUS_2_NAF[i] == 1) {
            fp2_copy(oldTx, T);
            fp2_copy(oldTy, T+8);
            g2_aff_add(T, q_aff, lam);
            line_eval(lam, oldTx, oldTy, px, py, line);
            fp12_mul(f, line, tmp);
            fp12_copy(f, tmp);
        } else if (SIX_X_PLUS_2_NAF[i] == -1) {
            fp2_copy(oldTx, T);
            fp2_copy(oldTy, T+8);
            g2_aff_add(T, negQ, lam);
            line_eval(lam, oldTx, oldTy, px, py, line);
            fp12_mul(f, line, tmp);
            fp12_copy(f, tmp);
        }
    }

    // Frobenius correction: Q1 = pi(Q), Q2 = -pi^2(Q)
    // Q1.x = conj(Q.x) * gamma_{1,2}
    // Q1.y = conj(Q.y) * gamma_{1,3}
    uint64_t q1[16];
    fp2_conj(q_aff, q1);
    fp2_mul(q1, GAMMA_1_2, q1);    // Q1.x (in-place)
    fp2_conj(q_aff + 8, q1 + 8);
    fp2_mul(q1 + 8, GAMMA_1_3, q1 + 8); // Q1.y

    fp2_copy(oldTx, T); fp2_copy(oldTy, T+8);
    g2_aff_add(T, q1, lam);
    line_eval(lam, oldTx, oldTy, px, py, line);
    fp12_mul(f, line, tmp);
    fp12_copy(f, tmp);

    // Q2.x = Q.x * gamma_{2,2}
    // Q2.y = -Q.y * gamma_{2,3}
    uint64_t q2[16];
    fp2_mul_fp(q_aff, GAMMA_2_2, q2);
    fp2_mul_fp(q_aff + 8, GAMMA_2_3, q2 + 8);
    fp2_neg(q2 + 8, q2 + 8);

    fp2_copy(oldTx, T); fp2_copy(oldTy, T+8);
    g2_aff_add(T, q2, lam);
    line_eval(lam, oldTx, oldTy, px, py, line);
    fp12_mul(f, line, tmp);
    fp12_copy(f, tmp);
}

// ============================================================
// Granger-Scott cyclotomic squaring for BN254
// For f = c0 + c1*w in GΦ₆ where w²=v and norm(f) = c0² - v*c1² = 1:
//   c0_new = 1 + 2*v*c1²
//   c1_new = 2*c0*c1
// ============================================================

static void fp12_cyc_sqr(const uint64_t f[FP12], uint64_t r[FP12]) {
    uint64_t c1sq[24], vc1sq[24], c0c1[24];

    fp6_sqr(f+24, c1sq);              // c1²
    fp6_mul_by_v(c1sq, vc1sq);        // v·c1²
    fp6_add(vc1sq, vc1sq, r);         // 2·v·c1² → r[0..23] = new c0
    fp_add(r, FP_ONE, r);             // +1 (add 1 to c0.c0.c0)

    fp6_mul(f, f+24, c0c1);           // c0·c1
    fp6_add(c0c1, c0c1, r+24);       // 2·c0·c1 → r[24..47] = new c1
}

// ============================================================
// Final Exponentiation
// f^((p^12 - 1) / r) = easy part * hard part
// ============================================================

// f^x using cyclotomic squaring (for elements in cyclotomic subgroup)
// x = 4965661367071055936 = 0x44E992B44A6909F1
static void fp12_pow_by_x(const uint64_t a[FP12], uint64_t r[FP12]) {
    const uint64_t x = 0x44E992B44A6909F1ULL;
    fp12_copy(r, a);
    uint64_t tmp[48];
    for (int i = 61; i >= 0; i--) {
        fp12_cyc_sqr(r, tmp);    // cyclotomic squaring (faster than generic)
        fp12_copy(r, tmp);
        if ((x >> i) & 1) {
            fp12_mul(r, a, tmp);
            fp12_copy(r, tmp);
        }
    }
}

// f^n for small n, using cyclotomic squaring
static void fp12_pow_small(const uint64_t a[FP12], int n, uint64_t r[FP12]) {
    if (n == 0) { fp12_one(r); return; }
    if (n == 1) { fp12_copy(r, a); return; }
    fp12_one(r);
    uint64_t base[48], tmp[48];
    fp12_copy(base, a);
    int k = n;
    while (k > 0) {
        if (k & 1) { fp12_mul(r, base, tmp); fp12_copy(r, tmp); }
        fp12_cyc_sqr(base, tmp); fp12_copy(base, tmp);
        k >>= 1;
    }
}

static void final_exp(const uint64_t f_in[FP12], uint64_t r[FP12]) {
    // Easy part: f^(p^6 - 1) * (p^2 + 1)
    uint64_t fconj[48], finv[48], tmp[48], result[48];
    fp12_conj(f_in, fconj);
    fp12_inv(f_in, finv);
    fp12_mul(fconj, finv, result);           // f^(p^6 - 1)

    fp12_frobenius2(result, tmp);
    fp12_mul(tmp, result, fconj);            // * (p^2 + 1), reuse fconj
    fp12_copy(result, fconj);

    // Hard part: result^((p^4 - p^2 + 1) / r)
    // Decomposition:
    //   lambda_0 = -(2 + 18x + 30x^2 + 36x^3)
    //   lambda_1 = 1 - 12x - 18x^2 - 36x^3
    //   lambda_2 = 1 + 6x^2
    //   lambda_3 = 1
    // hard = result^lambda_0 * frob(result)^lambda_1 * frob2(result)^lambda_2 * frob3(result)

    uint64_t a_val[48], a2[48], a3[48];
    fp12_pow_by_x(result, a_val);           // f^x
    fp12_pow_by_x(a_val, a2);              // f^(x^2)
    fp12_pow_by_x(a2, a3);                 // f^(x^3)

    // t0 = conj(f^2 * a^18 * a2^30 * a3^36)
    uint64_t f2[48], a18[48], a2_30[48], a3_36[48], t0[48];
    fp12_cyc_sqr(result, f2);
    fp12_pow_small(a_val, 18, a18);
    fp12_pow_small(a2, 30, a2_30);
    fp12_pow_small(a3, 36, a3_36);
    uint64_t tmp2[48];
    fp12_mul(f2, a18, tmp);
    fp12_mul(a2_30, a3_36, tmp2);
    fp12_mul(tmp, tmp2, t0);
    fp12_conj(t0, tmp);
    fp12_copy(t0, tmp);

    // t1 = frob(f * conj(a^12 * a2^18 * a3^36))
    uint64_t a12[48], a2_18[48], inner1[48], t1[48];
    fp12_pow_small(a_val, 12, a12);
    fp12_pow_small(a2, 18, a2_18);
    fp12_mul(a12, a2_18, tmp);
    fp12_mul(tmp, a3_36, tmp2);
    fp12_copy(tmp, tmp2);
    fp12_conj(tmp, inner1);
    fp12_mul(result, inner1, tmp);
    fp12_frobenius(tmp, t1);

    // t2 = frob2(f * a2^6)
    uint64_t a2_6[48], t2[48];
    fp12_pow_small(a2, 6, a2_6);
    fp12_mul(result, a2_6, tmp);
    fp12_frobenius2(tmp, t2);

    // t3 = frob3(f)
    uint64_t t3[48];
    fp12_frobenius3(result, t3);

    // result = t0 * t1 * t2 * t3
    fp12_mul(t0, t1, tmp);
    fp12_mul(t2, t3, t0);
    fp12_mul(tmp, t0, r);
}

// ============================================================
// Exported functions
// ============================================================

void bn254_miller_loop(const uint64_t p_aff[8], const uint64_t q_aff[16], uint64_t result[48]) {
    miller_loop(p_aff, q_aff, result);
}

void bn254_final_exp(const uint64_t f[48], uint64_t result[48]) {
    final_exp(f, result);
}

void bn254_pairing(const uint64_t p_aff[8], const uint64_t q_aff[16], uint64_t result[48]) {
    uint64_t ml[48];
    miller_loop(p_aff, q_aff, ml);
    final_exp(ml, result);
}

// Pairing check: verify prod_i e(P_i, Q_i) = 1
// pairs: interleaved [p0[8], q0[16], p1[8], q1[16], ...]
// n: number of pairs
// Returns 1 if product of pairings = 1, 0 otherwise
int bn254_pairing_check(const uint64_t *pairs, int n) {
    uint64_t f[48], ml[48], tmp[48];
    fp12_one(f);

    for (int i = 0; i < n; i++) {
        const uint64_t *pi = pairs + i * 24;   // 8 (G1) + 16 (G2) = 24
        const uint64_t *qi = pi + 8;
        miller_loop(pi, qi, ml);
        fp12_mul(f, ml, tmp);
        fp12_copy(f, tmp);
    }

    uint64_t result[48];
    final_exp(f, result);

    uint64_t one[48];
    fp12_one(one);
    return memcmp(result, one, 384) == 0;
}
