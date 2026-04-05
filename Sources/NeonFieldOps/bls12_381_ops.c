// BLS12-381 field arithmetic and G1 point operations — optimized C implementation
// CIOS Montgomery multiplication with __uint128_t for ARM64
//
// G1 curve: y^2 = x^3 + 4 (a=0, b=4)
// Fr: scalar field, 255-bit, 4×64-bit limbs
// Fp: base field, 381-bit, 6×64-bit limbs
//
// Interop: Swift Fp381 = 12×UInt32 = 6×uint64_t (Montgomery form)
//          Fr381 = 8×UInt32 = 4×uint64_t (Montgomery form)
//          G1Projective381 = 3×Fp381 = 18×uint64_t (x[6], y[6], z[6])

#include "NeonFieldOps.h"
#include <string.h>
#include <stdlib.h>
#include <pthread.h>

typedef unsigned __int128 uint128_t;

// ============================================================
// BLS12-381 Fr (scalar field) constants
// p = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
// ============================================================

static const uint64_t FR_P[4] = {
    0xffffffff00000001ULL, 0x53bda402fffe5bfeULL,
    0x3339d80809a1d805ULL, 0x73eda753299d7d48ULL
};
static const uint64_t FR_INV = 0xfffffffeffffffffULL;  // -p^{-1} mod 2^64
static const uint64_t FR_ONE[4] = {  // R mod p (Montgomery form of 1)
    0x00000001fffffffeULL, 0x5884b7fa00034802ULL,
    0x998c4fefecbc4ff5ULL, 0x1824b159acc5056fULL
};

// ============================================================
// CIOS Montgomery multiplication for BLS12-381 Fr (4-limb)
// ============================================================

static inline void fr_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t t0=0,t1=0,t2=0,t3=0,t4=0;
    #define FR_ITER(I) { \
        uint128_t w; uint64_t c; \
        w=(uint128_t)a[I]*b[0]+t0; t0=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[1]+t1+c; t1=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[2]+t2+c; t2=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[3]+t3+c; t3=(uint64_t)w; c=(uint64_t)(w>>64); \
        t4+=c; \
        uint64_t m=t0*FR_INV; \
        w=(uint128_t)m*FR_P[0]+t0; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*FR_P[1]+t1+c; t0=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*FR_P[2]+t2+c; t1=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*FR_P[3]+t3+c; t2=(uint64_t)w; c=(uint64_t)(w>>64); \
        t3=t4+c; t4=0; \
    }
    FR_ITER(0) FR_ITER(1) FR_ITER(2) FR_ITER(3)
    #undef FR_ITER
    uint64_t borrow=0; uint64_t r0,r1,r2,r3; uint128_t d;
    d=(uint128_t)t0-FR_P[0]-borrow; r0=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t1-FR_P[1]-borrow; r1=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t2-FR_P[2]-borrow; r2=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t3-FR_P[3]-borrow; r3=(uint64_t)d; borrow=(d>>127)&1;
    if(!borrow){r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;}
    else{r[0]=t0;r[1]=t1;r[2]=t2;r[3]=t3;}
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
    uint64_t borrow=0; uint64_t r0,r1,r2,r3; uint128_t d;
    d=(uint128_t)s0-FR_P[0]-borrow; r0=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)s1-FR_P[1]-borrow; r1=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)s2-FR_P[2]-borrow; r2=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)s3-FR_P[3]-borrow; r3=(uint64_t)d; borrow=(d>>127)&1;
    if(!borrow){r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;}
    else{r[0]=s0;r[1]=s1;r[2]=s2;r[3]=s3;}
}

static inline void fr_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t w; uint64_t c=0;
    w=(uint128_t)a[0]+b[0]; r[0]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[1]+b[1]+c; r[1]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[2]+b[2]+c; r[2]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[3]+b[3]+c; r[3]=(uint64_t)w; c=(uint64_t)(w>>64);
    uint64_t borrow=0; uint64_t r0,r1,r2,r3; uint128_t d;
    d=(uint128_t)r[0]-FR_P[0]; r0=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[1]-FR_P[1]-borrow; r1=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[2]-FR_P[2]-borrow; r2=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[3]-FR_P[3]-borrow; r3=(uint64_t)d; borrow=(d>>127)&1;
    if(c||!borrow){r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;}
}

static inline void fr_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t d; uint64_t borrow=0;
    d=(uint128_t)a[0]-b[0]; r[0]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[1]-b[1]-borrow; r[1]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[2]-b[2]-borrow; r[2]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[3]-b[3]-borrow; r[3]=(uint64_t)d; borrow=(d>>127)&1;
    if(borrow){
        uint64_t c=0;
        d=(uint128_t)r[0]+FR_P[0]; r[0]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[1]+FR_P[1]+c; r[1]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[2]+FR_P[2]+c; r[2]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[3]+FR_P[3]+c; r[3]=(uint64_t)d;
    }
}

static inline int fr_is_zero(const uint64_t a[4]) {
    return (a[0] | a[1] | a[2] | a[3]) == 0;
}

static inline void fr_neg(const uint64_t a[4], uint64_t r[4]) {
    if (fr_is_zero(a)) { memset(r, 0, 32); return; }
    uint128_t d; uint64_t borrow=0;
    d=(uint128_t)FR_P[0]-a[0]; r[0]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)FR_P[1]-a[1]-borrow; r[1]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)FR_P[2]-a[2]-borrow; r[2]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)FR_P[3]-a[3]-borrow; r[3]=(uint64_t)d;
}

// ============================================================
// Exported Fr operations
// ============================================================

void bls12_381_fr_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) { fr_mul(a, b, r); }
void bls12_381_fr_sqr(const uint64_t a[4], uint64_t r[4]) { fr_sqr(a, r); }
void bls12_381_fr_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) { fr_add(a, b, r); }
void bls12_381_fr_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) { fr_sub(a, b, r); }
void bls12_381_fr_neg(const uint64_t a[4], uint64_t r[4]) { fr_neg(a, r); }

// ============================================================
// BLS12-381 Fp (base field) constants
// p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
// ============================================================

static const uint64_t FP381_P[6] = {
    0xb9feffffffffaaabULL, 0x1eabfffeb153ffffULL,
    0x6730d2a0f6b0f624ULL, 0x64774b84f38512bfULL,
    0x4b1ba7b6434bacd7ULL, 0x1a0111ea397fe69aULL
};
static const uint64_t FP381_INV = 0x89f3fffcfffcfffdULL;  // -p^{-1} mod 2^64
static const uint64_t FP381_ONE[6] = {  // R mod p (Montgomery form of 1)
    0x760900000002fffdULL, 0xebf4000bc40c0002ULL,
    0x5f48985753c758baULL, 0x77ce585370525745ULL,
    0x5c071a97a256ec6dULL, 0x15f65ec3fa80e493ULL
};

// ============================================================
// CIOS Montgomery multiplication for BLS12-381 Fp (6-limb)
// ============================================================

static inline void fp_mul(const uint64_t a[6], const uint64_t b[6], uint64_t r[6]) {
    uint64_t t[7] = {0};

    for (int i = 0; i < 6; i++) {
        // Multiply: t += a[i] * b
        uint128_t carry = 0;
        for (int j = 0; j < 6; j++) {
            carry += (uint128_t)a[i] * b[j] + t[j];
            t[j] = (uint64_t)carry;
            carry >>= 64;
        }
        t[6] += (uint64_t)carry;

        // Reduce: t += m * p, then shift right by 64
        uint64_t m = t[0] * FP381_INV;
        carry = (uint128_t)m * FP381_P[0] + t[0];
        carry >>= 64;
        for (int j = 1; j < 6; j++) {
            carry += (uint128_t)m * FP381_P[j] + t[j];
            t[j-1] = (uint64_t)carry;
            carry >>= 64;
        }
        t[5] = t[6] + (uint64_t)carry;
        t[6] = (uint64_t)(carry >> 64);
    }

    // Final conditional subtraction
    uint64_t borrow = 0;
    uint64_t tmp[6];
    uint128_t d;
    d = (uint128_t)t[0] - FP381_P[0]; tmp[0] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)t[1] - FP381_P[1] - borrow; tmp[1] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)t[2] - FP381_P[2] - borrow; tmp[2] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)t[3] - FP381_P[3] - borrow; tmp[3] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)t[4] - FP381_P[4] - borrow; tmp[4] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)t[5] - FP381_P[5] - borrow; tmp[5] = (uint64_t)d; borrow = (d >> 127) & 1;

    if (!borrow) {
        memcpy(r, tmp, 48);
    } else {
        memcpy(r, t, 48);
    }
}

static inline void fp_sqr(const uint64_t a[6], uint64_t r[6]) {
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
        uint64_t m = s[0] * FP381_INV;
        w = (uint128_t)m * FP381_P[0] + s[0]; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP381_P[1] + s[1] + c; s[0] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP381_P[2] + s[2] + c; s[1] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP381_P[3] + s[3] + c; s[2] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP381_P[4] + s[4] + c; s[3] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP381_P[5] + s[5] + c; s[4] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)s[6] + c; s[5] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)s[7] + c; s[6] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)s[8] + c; s[7] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)s[9] + c; s[8] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)s[10] + c; s[9] = (uint64_t)w; c = (uint64_t)(w >> 64);
        s[10] = s[11] + c; s[11] = 0;
    }
    {
        uint64_t m = s[0] * FP381_INV;
        w = (uint128_t)m * FP381_P[0] + s[0]; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP381_P[1] + s[1] + c; s[0] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP381_P[2] + s[2] + c; s[1] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP381_P[3] + s[3] + c; s[2] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP381_P[4] + s[4] + c; s[3] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP381_P[5] + s[5] + c; s[4] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)s[6] + c; s[5] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)s[7] + c; s[6] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)s[8] + c; s[7] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)s[9] + c; s[8] = (uint64_t)w; c = (uint64_t)(w >> 64);
        s[9] = s[10] + c; s[10] = 0;
    }
    {
        uint64_t m = s[0] * FP381_INV;
        w = (uint128_t)m * FP381_P[0] + s[0]; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP381_P[1] + s[1] + c; s[0] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP381_P[2] + s[2] + c; s[1] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP381_P[3] + s[3] + c; s[2] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP381_P[4] + s[4] + c; s[3] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP381_P[5] + s[5] + c; s[4] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)s[6] + c; s[5] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)s[7] + c; s[6] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)s[8] + c; s[7] = (uint64_t)w; c = (uint64_t)(w >> 64);
        s[8] = s[9] + c; s[9] = 0;
    }
    {
        uint64_t m = s[0] * FP381_INV;
        w = (uint128_t)m * FP381_P[0] + s[0]; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP381_P[1] + s[1] + c; s[0] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP381_P[2] + s[2] + c; s[1] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP381_P[3] + s[3] + c; s[2] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP381_P[4] + s[4] + c; s[3] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP381_P[5] + s[5] + c; s[4] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)s[6] + c; s[5] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)s[7] + c; s[6] = (uint64_t)w; c = (uint64_t)(w >> 64);
        s[7] = s[8] + c; s[8] = 0;
    }
    {
        uint64_t m = s[0] * FP381_INV;
        w = (uint128_t)m * FP381_P[0] + s[0]; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP381_P[1] + s[1] + c; s[0] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP381_P[2] + s[2] + c; s[1] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP381_P[3] + s[3] + c; s[2] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP381_P[4] + s[4] + c; s[3] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP381_P[5] + s[5] + c; s[4] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)s[6] + c; s[5] = (uint64_t)w; c = (uint64_t)(w >> 64);
        s[6] = s[7] + c; s[7] = 0;
    }
    {
        uint64_t m = s[0] * FP381_INV;
        w = (uint128_t)m * FP381_P[0] + s[0]; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP381_P[1] + s[1] + c; s[0] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP381_P[2] + s[2] + c; s[1] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP381_P[3] + s[3] + c; s[2] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP381_P[4] + s[4] + c; s[3] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP381_P[5] + s[5] + c; s[4] = (uint64_t)w; c = (uint64_t)(w >> 64);
        s[5] = s[6] + c;
    }

    // Final conditional subtraction
    uint64_t borrow = 0;
    uint64_t tmp[6];
    uint128_t d;
    d = (uint128_t)s[0] - FP381_P[0]; tmp[0] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)s[1] - FP381_P[1] - borrow; tmp[1] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)s[2] - FP381_P[2] - borrow; tmp[2] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)s[3] - FP381_P[3] - borrow; tmp[3] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)s[4] - FP381_P[4] - borrow; tmp[4] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)s[5] - FP381_P[5] - borrow; tmp[5] = (uint64_t)d; borrow = (d >> 127) & 1;

    if (!borrow) {
        memcpy(r, tmp, 48);
    } else {
        memcpy(r, s, 48);
    }
}

static inline void fp_add(const uint64_t a[6], const uint64_t b[6], uint64_t r[6]) {
    uint128_t w; uint64_t c = 0;
    w = (uint128_t)a[0] + b[0]; r[0] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[1] + b[1] + c; r[1] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[2] + b[2] + c; r[2] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[3] + b[3] + c; r[3] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[4] + b[4] + c; r[4] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[5] + b[5] + c; r[5] = (uint64_t)w; c = (uint64_t)(w >> 64);

    // Conditional subtraction
    uint64_t borrow = 0; uint64_t tmp[6]; uint128_t d;
    d = (uint128_t)r[0] - FP381_P[0]; tmp[0] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)r[1] - FP381_P[1] - borrow; tmp[1] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)r[2] - FP381_P[2] - borrow; tmp[2] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)r[3] - FP381_P[3] - borrow; tmp[3] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)r[4] - FP381_P[4] - borrow; tmp[4] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)r[5] - FP381_P[5] - borrow; tmp[5] = (uint64_t)d; borrow = (d >> 127) & 1;
    if (c || !borrow) { memcpy(r, tmp, 48); }
}

static inline void fp_sub(const uint64_t a[6], const uint64_t b[6], uint64_t r[6]) {
    uint128_t d; uint64_t borrow = 0;
    d = (uint128_t)a[0] - b[0]; r[0] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)a[1] - b[1] - borrow; r[1] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)a[2] - b[2] - borrow; r[2] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)a[3] - b[3] - borrow; r[3] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)a[4] - b[4] - borrow; r[4] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)a[5] - b[5] - borrow; r[5] = (uint64_t)d; borrow = (d >> 127) & 1;
    if (borrow) {
        uint64_t c = 0;
        d = (uint128_t)r[0] + FP381_P[0]; r[0] = (uint64_t)d; c = (uint64_t)(d >> 64);
        d = (uint128_t)r[1] + FP381_P[1] + c; r[1] = (uint64_t)d; c = (uint64_t)(d >> 64);
        d = (uint128_t)r[2] + FP381_P[2] + c; r[2] = (uint64_t)d; c = (uint64_t)(d >> 64);
        d = (uint128_t)r[3] + FP381_P[3] + c; r[3] = (uint64_t)d; c = (uint64_t)(d >> 64);
        d = (uint128_t)r[4] + FP381_P[4] + c; r[4] = (uint64_t)d; c = (uint64_t)(d >> 64);
        d = (uint128_t)r[5] + FP381_P[5] + c; r[5] = (uint64_t)d;
    }
}

static inline void fp_dbl(const uint64_t a[6], uint64_t r[6]) {
    fp_add(a, a, r);
}

static inline int fp_is_zero(const uint64_t a[6]) {
    return (a[0] | a[1] | a[2] | a[3] | a[4] | a[5]) == 0;
}

static inline void fp_neg(const uint64_t a[6], uint64_t r[6]) {
    if (fp_is_zero(a)) { memset(r, 0, 48); return; }
    uint128_t d; uint64_t borrow = 0;
    d = (uint128_t)FP381_P[0] - a[0]; r[0] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)FP381_P[1] - a[1] - borrow; r[1] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)FP381_P[2] - a[2] - borrow; r[2] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)FP381_P[3] - a[3] - borrow; r[3] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)FP381_P[4] - a[4] - borrow; r[4] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)FP381_P[5] - a[5] - borrow; r[5] = (uint64_t)d;
}

// ============================================================
// Exported Fp operations
// ============================================================

void bls12_381_fp_mul(const uint64_t a[6], const uint64_t b[6], uint64_t r[6]) { fp_mul(a, b, r); }
void bls12_381_fp_sqr(const uint64_t a[6], uint64_t r[6]) { fp_sqr(a, r); }
void bls12_381_fp_add(const uint64_t a[6], const uint64_t b[6], uint64_t r[6]) { fp_add(a, b, r); }
void bls12_381_fp_sub(const uint64_t a[6], const uint64_t b[6], uint64_t r[6]) { fp_sub(a, b, r); }
void bls12_381_fp_neg(const uint64_t a[6], uint64_t r[6]) { fp_neg(a, r); }

// ============================================================
// BLS12-381 G1 Jacobian projective point operations
// Layout: [x0..x5, y0..y5, z0..z5] = 18 uint64_t
// Identity: Z = 0
// Curve: y^2 = x^3 + 4 (a=0)
// ============================================================

static inline void g1_set_id(uint64_t p[18]) {
    memcpy(p, FP381_ONE, 48);       // x = 1
    memcpy(p + 6, FP381_ONE, 48);   // y = 1
    memset(p + 12, 0, 48);          // z = 0
}

static inline int g1_is_id(const uint64_t p[18]) {
    return fp_is_zero(p + 12);
}

// Point doubling for a=0 curve (y^2 = x^3 + 4)
static void g1_dbl(const uint64_t p[18], uint64_t r[18]) {
    if (g1_is_id(p)) { memcpy(r, p, 144); return; }

    const uint64_t *px = p, *py = p + 6, *pz = p + 12;
    uint64_t a[6], b[6], c[6], d[6], e[6], f[6], t1[6], t2[6];

    fp_sqr(px, a);             // a = x^2
    fp_sqr(py, b);             // b = y^2
    fp_sqr(b, c);              // c = y^4

    // d = 2((x+b)^2 - a - c)
    fp_add(px, b, t1);
    fp_sqr(t1, t1);
    fp_sub(t1, a, t1);
    fp_sub(t1, c, t1);
    fp_dbl(t1, d);

    // e = 3a = 3*x^2
    fp_dbl(a, t1);
    fp_add(t1, a, e);

    fp_sqr(e, f);              // f = e^2

    // x3 = f - 2d
    fp_dbl(d, t1);
    fp_sub(f, t1, r);

    // y3 = e(d - x3) - 8c
    fp_sub(d, r, t1);
    fp_mul(e, t1, t2);
    fp_dbl(c, t1);
    fp_dbl(t1, t1);
    fp_dbl(t1, t1);            // 8c
    fp_sub(t2, t1, r + 6);

    // z3 = (y+z)^2 - b - z^2
    fp_add(py, pz, t1);
    fp_sqr(t1, t1);
    fp_sub(t1, b, t1);
    fp_sqr(pz, t2);
    fp_sub(t1, t2, r + 12);
}

// Full projective addition
static void g1_add(const uint64_t p[18], const uint64_t q[18], uint64_t r[18]) {
    if (g1_is_id(p)) { memcpy(r, q, 144); return; }
    if (g1_is_id(q)) { memcpy(r, p, 144); return; }

    const uint64_t *px = p, *py = p+6, *pz = p+12;
    const uint64_t *qx = q, *qy = q+6, *qz = q+12;

    uint64_t z1z1[6], z2z2[6], u1[6], u2[6], s1[6], s2[6];
    uint64_t h[6], rr[6], ii[6], j[6], vv[6], t1[6];

    fp_sqr(pz, z1z1);
    fp_sqr(qz, z2z2);
    fp_mul(px, z2z2, u1);
    fp_mul(qx, z1z1, u2);
    fp_mul(qz, z2z2, t1);
    fp_mul(py, t1, s1);
    fp_mul(pz, z1z1, t1);
    fp_mul(qy, t1, s2);

    fp_sub(u2, u1, h);
    fp_sub(s2, s1, t1);
    fp_dbl(t1, rr);

    if (fp_is_zero(h)) {
        if (fp_is_zero(rr)) { g1_dbl(p, r); return; }
        g1_set_id(r); return;
    }

    fp_dbl(h, t1);
    fp_sqr(t1, ii);
    fp_mul(h, ii, j);
    fp_mul(u1, ii, vv);

    // x3 = r^2 - j - 2v
    fp_sqr(rr, r);
    fp_sub(r, j, r);
    fp_dbl(vv, t1);
    fp_sub(r, t1, r);

    // y3 = r(v - x3) - 2*s1*j
    fp_sub(vv, r, t1);
    fp_mul(rr, t1, r + 6);
    fp_mul(s1, j, t1);
    fp_dbl(t1, t1);
    fp_sub(r + 6, t1, r + 6);

    // z3 = ((z1+z2)^2 - z1z1 - z2z2) * h
    fp_add(pz, qz, t1);
    fp_sqr(t1, t1);
    fp_sub(t1, z1z1, t1);
    fp_sub(t1, z2z2, t1);
    fp_mul(t1, h, r + 12);
}

// Mixed addition: projective P + affine Q (Z_Q = 1)
static void g1_add_mixed(const uint64_t p[18], const uint64_t q_aff[12], uint64_t r[18]) {
    if (g1_is_id(p)) {
        memcpy(r, q_aff, 96);
        memcpy(r + 12, FP381_ONE, 48);
        return;
    }

    const uint64_t *px = p, *py = p+6, *pz = p+12;
    const uint64_t *qx = q_aff, *qy = q_aff+6;

    uint64_t z1z1[6], u2[6], s2[6], h[6], hh[6], rr[6], t1[6];

    fp_sqr(pz, z1z1);
    fp_mul(qx, z1z1, u2);       // u2 = qx * z1^2
    fp_mul(pz, z1z1, t1);
    fp_mul(qy, t1, s2);         // s2 = qy * z1^3

    fp_sub(u2, px, h);          // h = u2 - px
    fp_sub(s2, py, t1);         // t1 = s2 - py
    fp_dbl(t1, rr);

    if (fp_is_zero(h)) {
        if (fp_is_zero(rr)) { g1_dbl(p, r); return; }
        g1_set_id(r); return;
    }

    fp_sqr(h, hh);
    uint64_t j[6], vv[6];
    fp_mul(h, hh, j);
    fp_mul(px, hh, vv);

    // x3 = r^2 - j - 2v
    fp_sqr(rr, r);
    fp_sub(r, j, r);
    fp_dbl(vv, t1);
    fp_sub(r, t1, r);

    // y3 = r(v - x3) - 2*s1*j
    fp_sub(vv, r, t1);
    fp_mul(rr, t1, r + 6);
    fp_mul(py, j, t1);
    fp_dbl(t1, t1);
    fp_sub(r + 6, t1, r + 6);

    // z3 = (z1 + h)^2 - z1z1 - hh
    fp_add(pz, h, t1);
    fp_sqr(t1, t1);
    fp_sub(t1, z1z1, t1);
    fp_sub(t1, hh, r + 12);
}

// ============================================================
// G1 scalar multiplication using windowed method (w=4)
// scalar: 4 x uint64_t in non-Montgomery integer form
// ============================================================

void bls12_381_g1_scalar_mul(const uint64_t p[18], const uint64_t scalar[4], uint64_t r[18]) {
    uint64_t table[16 * 18];

    g1_set_id(table);
    memcpy(table + 18, p, 144);
    for (int i = 2; i < 16; i++) {
        g1_add(table + (i - 1) * 18, p, table + i * 18);
    }

    uint8_t nibbles[64];
    for (int i = 0; i < 4; i++) {
        uint64_t word = scalar[i];
        for (int j = 0; j < 16; j++) {
            nibbles[i * 16 + j] = (uint8_t)(word & 0xF);
            word >>= 4;
        }
    }

    int top = 63;
    while (top >= 0 && nibbles[top] == 0) top--;

    if (top < 0) {
        g1_set_id(r);
        return;
    }

    uint64_t result[18];
    memcpy(result, table + nibbles[top] * 18, 144);

    for (int i = top - 1; i >= 0; i--) {
        uint64_t tmp[18];
        g1_dbl(result, tmp); memcpy(result, tmp, 144);
        g1_dbl(result, tmp); memcpy(result, tmp, 144);
        g1_dbl(result, tmp); memcpy(result, tmp, 144);
        g1_dbl(result, tmp); memcpy(result, tmp, 144);
        if (nibbles[i]) {
            g1_add(result, table + nibbles[i] * 18, tmp);
            memcpy(result, tmp, 144);
        }
    }

    memcpy(r, result, 144);
}

// ============================================================
// Exported G1 point operations
// ============================================================

void bls12_381_g1_point_add(const uint64_t p[18], const uint64_t q[18], uint64_t r[18]) {
    g1_add(p, q, r);
}

void bls12_381_g1_point_double(const uint64_t p[18], uint64_t r[18]) {
    g1_dbl(p, r);
}

void bls12_381_g1_point_add_mixed(const uint64_t p[18], const uint64_t q_aff[12], uint64_t r[18]) {
    g1_add_mixed(p, q_aff, r);
}

// ============================================================
// BLS12-381 G1 Pippenger MSM
// ============================================================

static void fp_inv(const uint64_t a[6], uint64_t result[6]) {
    uint64_t pm2[6];
    pm2[0] = FP381_P[0] - 2;
    pm2[1] = FP381_P[1];
    pm2[2] = FP381_P[2];
    pm2[3] = FP381_P[3];
    pm2[4] = FP381_P[4];
    pm2[5] = FP381_P[5];
    memcpy(result, FP381_ONE, 48);
    uint64_t b[6];
    memcpy(b, a, 48);
    for (int i = 0; i < 6; i++) {
        for (int bit = 0; bit < 64; bit++) {
            if ((pm2[i] >> bit) & 1)
                fp_mul(result, b, result);
            fp_sqr(b, b);
        }
    }
}

static inline int g1_aff_is_id(const uint64_t q[12]) {
    return (q[0] | q[1] | q[2] | q[3] | q[4] | q[5] |
            q[6] | q[7] | q[8] | q[9] | q[10] | q[11]) == 0;
}

static void g1_batch_to_affine(const uint64_t *proj, uint64_t *aff, int n) {
    if (n == 0) return;
    uint64_t *prods = (uint64_t *)malloc((size_t)n * 48);
    int first_valid = -1;
    for (int i = 0; i < n; i++) {
        if (g1_is_id(proj + i * 18)) {
            if (i == 0) memcpy(prods, FP381_ONE, 48);
            else memcpy(prods + i * 6, prods + (i-1) * 6, 48);
        } else {
            if (first_valid < 0) {
                first_valid = i;
                memcpy(prods + i * 6, proj + i * 18 + 12, 48);
            } else {
                fp_mul(prods + (i-1) * 6, proj + i * 18 + 12, prods + i * 6);
            }
        }
    }
    if (first_valid < 0) {
        memset(aff, 0, (size_t)n * 96);
        free(prods);
        return;
    }
    uint64_t inv[6];
    fp_inv(prods + (n-1) * 6, inv);
    for (int i = n - 1; i >= 0; i--) {
        if (g1_is_id(proj + i * 18)) {
            memset(aff + i * 12, 0, 96);
            continue;
        }
        uint64_t zinv[6];
        if (i > first_valid) {
            fp_mul(inv, prods + (i-1) * 6, zinv);
            fp_mul(inv, proj + i * 18 + 12, inv);
        } else {
            memcpy(zinv, inv, 48);
        }
        uint64_t zinv2[6], zinv3[6];
        fp_sqr(zinv, zinv2);
        fp_mul(zinv2, zinv, zinv3);
        fp_mul(proj + i * 18, zinv2, aff + i * 12);
        fp_mul(proj + i * 18 + 6, zinv3, aff + i * 12 + 6);
    }
    free(prods);
}

static inline uint32_t g1_extract_window(const uint32_t *scalar, int window_idx, int window_bits) {
    int bit_offset = window_idx * window_bits;
    int word_idx = bit_offset / 32;
    int bit_in_word = bit_offset % 32;
    uint64_t word = scalar[word_idx];
    if (word_idx + 1 < 8) word |= ((uint64_t)scalar[word_idx + 1]) << 32;
    return (uint32_t)((word >> bit_in_word) & ((1u << window_bits) - 1));
}

static int g1_optimal_window_bits(int n) {
    if (n <= 4) return 3;
    if (n <= 32) return 5;
    if (n <= 256) return 8;
    if (n <= 2048) return 10;
    if (n <= 8192) return 11;
    if (n <= 32768) return 13;
    if (n <= 131072) return 14;
    if (n <= 524288) return 15;
    return 16;
}

typedef struct {
    const uint64_t *points;
    const uint32_t *scalars;
    int n, window_bits, window_idx, num_buckets;
    uint64_t result[18];
} G1WindowTask;

static void *g1_window_worker(void *arg) {
    G1WindowTask *task = (G1WindowTask *)arg;
    int wb = task->window_bits, w = task->window_idx;
    int nb = task->num_buckets, nn = task->n;
    uint64_t *buckets = (uint64_t *)malloc((size_t)(nb + 1) * 144);
    for (int b = 0; b <= nb; b++) g1_set_id(buckets + b * 18);
    for (int i = 0; i < nn; i++) {
        uint32_t digit = g1_extract_window(task->scalars + i * 8, w, wb);
        if (digit != 0) {
            uint64_t tmp[18];
            g1_add_mixed(buckets + digit * 18, task->points + i * 12, tmp);
            memcpy(buckets + digit * 18, tmp, 144);
        }
    }
    uint64_t *bucket_aff = (uint64_t *)malloc((size_t)nb * 96);
    g1_batch_to_affine(buckets + 18, bucket_aff, nb);
    uint64_t running[18], window_sum[18];
    g1_set_id(running); g1_set_id(window_sum);
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
    free(buckets); free(bucket_aff);
    return NULL;
}

void bls12_381_g1_pippenger_msm(const uint64_t *points, const uint32_t *scalars,
                                 int n, uint64_t *result) {
    if (n == 0) { g1_set_id(result); return; }
    int wb = g1_optimal_window_bits(n);
    int num_windows = (256 + wb - 1) / wb;
    int num_buckets = (1 << wb) - 1;
    G1WindowTask *tasks = (G1WindowTask *)malloc((size_t)num_windows * sizeof(G1WindowTask));
    pthread_t *threads = (pthread_t *)malloc((size_t)num_windows * sizeof(pthread_t));
    for (int w = 0; w < num_windows; w++) {
        tasks[w].points = points; tasks[w].scalars = scalars;
        tasks[w].n = n; tasks[w].window_bits = wb;
        tasks[w].window_idx = w; tasks[w].num_buckets = num_buckets;
    }
    for (int w = 0; w < num_windows; w++)
        pthread_create(&threads[w], NULL, g1_window_worker, &tasks[w]);
    for (int w = 0; w < num_windows; w++)
        pthread_join(threads[w], NULL);
    memcpy(result, tasks[num_windows - 1].result, 144);
    for (int w = num_windows - 2; w >= 0; w--) {
        uint64_t tmp[18];
        for (int s = 0; s < wb; s++) { g1_dbl(result, tmp); memcpy(result, tmp, 144); }
        g1_add(result, tasks[w].result, tmp);
        memcpy(result, tmp, 144);
    }
    free(tasks); free(threads);
}

// ============================================================
// BLS12-381 Fp2 = Fp[u]/(u^2+1) tower arithmetic
// Layout: [c0[6], c1[6]] = 12 uint64_t
// Element represents c0 + c1*u
// ============================================================

static inline void fp2_add(const uint64_t a[12], const uint64_t b[12], uint64_t r[12]) {
    fp_add(a, b, r);
    fp_add(a + 6, b + 6, r + 6);
}

static inline void fp2_sub(const uint64_t a[12], const uint64_t b[12], uint64_t r[12]) {
    fp_sub(a, b, r);
    fp_sub(a + 6, b + 6, r + 6);
}

static inline void fp2_neg(const uint64_t a[12], uint64_t r[12]) {
    fp_neg(a, r);
    fp_neg(a + 6, r + 6);
}

static inline void fp2_dbl(const uint64_t a[12], uint64_t r[12]) {
    fp_dbl(a, r);
    fp_dbl(a + 6, r + 6);
}

static inline int fp2_is_zero(const uint64_t a[12]) {
    return fp_is_zero(a) && fp_is_zero(a + 6);
}

// Conjugate: (a0, -a1)
static inline void fp2_conj(const uint64_t a[12], uint64_t r[12]) {
    memcpy(r, a, 48);
    fp_neg(a + 6, r + 6);
}

// Karatsuba multiplication: 3 Fp muls
// c0 = a0*b0 - a1*b1
// c1 = (a0+a1)(b0+b1) - a0*b0 - a1*b1
static inline void fp2_mul(const uint64_t a[12], const uint64_t b[12], uint64_t r[12]) {
    uint64_t a0b0[6], a1b1[6], t1[6], t2[6], t3[6];

    fp_mul(a, b, a0b0);          // a0*b0
    fp_mul(a + 6, b + 6, a1b1);  // a1*b1

    fp_add(a, a + 6, t1);        // a0 + a1
    fp_add(b, b + 6, t2);        // b0 + b1
    fp_mul(t1, t2, t3);          // (a0+a1)(b0+b1)

    fp_sub(a0b0, a1b1, r);       // c0 = a0*b0 - a1*b1
    fp_sub(t3, a0b0, r + 6);     // c1 = t3 - a0*b0 - a1*b1
    fp_sub(r + 6, a1b1, r + 6);
}

// Optimized squaring: 2 Fp muls
// c0 = (a0+a1)(a0-a1) = a0^2 - a1^2
// c1 = 2*a0*a1
static inline void fp2_sqr(const uint64_t a[12], uint64_t r[12]) {
    uint64_t v0[6], t1[6], t2[6];

    fp_mul(a, a + 6, v0);       // a0*a1
    fp_add(a, a + 6, t1);       // a0 + a1
    fp_sub(a, a + 6, t2);       // a0 - a1
    fp_mul(t1, t2, r);          // c0 = (a0+a1)(a0-a1)
    fp_dbl(v0, r + 6);          // c1 = 2*a0*a1
}

// Multiply by non-residue (1+u) for Fp6 tower:
// (a0 + a1*u)(1 + u) = (a0 - a1) + (a0 + a1)*u
static inline void fp2_mul_by_nonresidue(const uint64_t a[12], uint64_t r[12]) {
    uint64_t t0[6], t1[6];
    fp_sub(a, a + 6, t0);      // a0 - a1
    fp_add(a, a + 6, t1);      // a0 + a1
    memcpy(r, t0, 48);
    memcpy(r + 6, t1, 48);
}

// Multiply Fp2 element by Fp scalar
static inline void fp2_mul_by_fp(const uint64_t a[12], const uint64_t b[6], uint64_t r[12]) {
    fp_mul(a, b, r);
    fp_mul(a + 6, b, r + 6);
}

// ============================================================
// Exported Fp2 operations
// ============================================================

void bls12_381_fp2_add(const uint64_t a[12], const uint64_t b[12], uint64_t r[12]) { fp2_add(a, b, r); }
void bls12_381_fp2_sub(const uint64_t a[12], const uint64_t b[12], uint64_t r[12]) { fp2_sub(a, b, r); }
void bls12_381_fp2_neg(const uint64_t a[12], uint64_t r[12]) { fp2_neg(a, r); }
void bls12_381_fp2_mul(const uint64_t a[12], const uint64_t b[12], uint64_t r[12]) { fp2_mul(a, b, r); }
void bls12_381_fp2_sqr(const uint64_t a[12], uint64_t r[12]) { fp2_sqr(a, r); }
void bls12_381_fp2_conj(const uint64_t a[12], uint64_t r[12]) { fp2_conj(a, r); }
void bls12_381_fp2_mul_by_nonresidue(const uint64_t a[12], uint64_t r[12]) { fp2_mul_by_nonresidue(a, r); }

// ============================================================
// BLS12-381 G2 Jacobian projective point operations over Fp2
// Layout: [x0..x11, y0..y11, z0..z11] = 36 uint64_t
// Identity: Z = 0
// Curve: y^2 = x^3 + 4(1+u) (a=0, b'=4(1+u))
// ============================================================

static inline void g2_set_id(uint64_t p[36]) {
    // x = 1 (Fp2 one = (Fp_one, 0))
    memcpy(p, FP381_ONE, 48);
    memset(p + 6, 0, 48);
    // y = 1
    memcpy(p + 12, FP381_ONE, 48);
    memset(p + 18, 0, 48);
    // z = 0
    memset(p + 24, 0, 96);
}

static inline int g2_is_id(const uint64_t p[36]) {
    return fp2_is_zero(p + 24);
}

// Point doubling for a=0 curve over Fp2 (same algorithm as G1)
static void g2_dbl(const uint64_t p[36], uint64_t r[36]) {
    if (g2_is_id(p)) { memcpy(r, p, 288); return; }

    const uint64_t *px = p, *py = p + 12, *pz = p + 24;
    uint64_t a[12], b[12], c[12], d[12], e[12], f[12], t1[12], t2[12];

    fp2_sqr(px, a);             // a = x^2
    fp2_sqr(py, b);             // b = y^2
    fp2_sqr(b, c);              // c = y^4

    // d = 2((x+b)^2 - a - c)
    fp2_add(px, b, t1);
    fp2_sqr(t1, t1);
    fp2_sub(t1, a, t1);
    fp2_sub(t1, c, t1);
    fp2_dbl(t1, d);

    // e = 3a = 3*x^2
    fp2_dbl(a, t1);
    fp2_add(t1, a, e);

    fp2_sqr(e, f);              // f = e^2

    // x3 = f - 2d
    fp2_dbl(d, t1);
    fp2_sub(f, t1, r);

    // y3 = e(d - x3) - 8c
    fp2_sub(d, r, t1);
    fp2_mul(e, t1, t2);
    fp2_dbl(c, t1);
    fp2_dbl(t1, t1);
    fp2_dbl(t1, t1);            // 8c
    fp2_sub(t2, t1, r + 12);

    // z3 = (y+z)^2 - b - z^2
    fp2_add(py, pz, t1);
    fp2_sqr(t1, t1);
    fp2_sub(t1, b, t1);
    fp2_sqr(pz, t2);
    fp2_sub(t1, t2, r + 24);
}

// Full projective addition over Fp2
static void g2_add(const uint64_t p[36], const uint64_t q[36], uint64_t r[36]) {
    if (g2_is_id(p)) { memcpy(r, q, 288); return; }
    if (g2_is_id(q)) { memcpy(r, p, 288); return; }

    const uint64_t *px = p, *py = p+12, *pz = p+24;
    const uint64_t *qx = q, *qy = q+12, *qz = q+24;

    uint64_t z1z1[12], z2z2[12], u1[12], u2[12], s1[12], s2[12];
    uint64_t h[12], rr[12], ii[12], j[12], vv[12], t1[12];

    fp2_sqr(pz, z1z1);
    fp2_sqr(qz, z2z2);
    fp2_mul(px, z2z2, u1);
    fp2_mul(qx, z1z1, u2);
    fp2_mul(qz, z2z2, t1);
    fp2_mul(py, t1, s1);
    fp2_mul(pz, z1z1, t1);
    fp2_mul(qy, t1, s2);

    fp2_sub(u2, u1, h);
    fp2_sub(s2, s1, t1);
    fp2_dbl(t1, rr);

    if (fp2_is_zero(h)) {
        if (fp2_is_zero(rr)) { g2_dbl(p, r); return; }
        g2_set_id(r); return;
    }

    fp2_dbl(h, t1);
    fp2_sqr(t1, ii);
    fp2_mul(h, ii, j);
    fp2_mul(u1, ii, vv);

    // x3 = r^2 - j - 2v
    fp2_sqr(rr, r);
    fp2_sub(r, j, r);
    fp2_dbl(vv, t1);
    fp2_sub(r, t1, r);

    // y3 = r(v - x3) - 2*s1*j
    fp2_sub(vv, r, t1);
    fp2_mul(rr, t1, r + 12);
    fp2_mul(s1, j, t1);
    fp2_dbl(t1, t1);
    fp2_sub(r + 12, t1, r + 12);

    // z3 = ((z1+z2)^2 - z1z1 - z2z2) * h
    fp2_add(pz, qz, t1);
    fp2_sqr(t1, t1);
    fp2_sub(t1, z1z1, t1);
    fp2_sub(t1, z2z2, t1);
    fp2_mul(t1, h, r + 24);
}

// Mixed addition: projective P + affine Q (Z_Q = 1) over Fp2
static void g2_add_mixed(const uint64_t p[36], const uint64_t q_aff[24], uint64_t r[36]) {
    if (g2_is_id(p)) {
        memcpy(r, q_aff, 192);
        // z = Fp2(1) = (Fp_one, 0)
        memcpy(r + 24, FP381_ONE, 48);
        memset(r + 30, 0, 48);
        return;
    }

    const uint64_t *px = p, *py = p+12, *pz = p+24;
    const uint64_t *qx = q_aff, *qy = q_aff+12;

    uint64_t z1z1[12], u2[12], s2[12], h[12], hh[12], rr[12], t1[12];

    fp2_sqr(pz, z1z1);
    fp2_mul(qx, z1z1, u2);         // u2 = qx * z1^2
    fp2_mul(pz, z1z1, t1);
    fp2_mul(qy, t1, s2);           // s2 = qy * z1^3

    fp2_sub(u2, px, h);            // h = u2 - px
    fp2_sub(s2, py, t1);           // t1 = s2 - py
    fp2_dbl(t1, rr);

    if (fp2_is_zero(h)) {
        if (fp2_is_zero(rr)) { g2_dbl(p, r); return; }
        g2_set_id(r); return;
    }

    fp2_sqr(h, hh);
    uint64_t j[12], vv[12];
    fp2_mul(h, hh, j);
    fp2_mul(px, hh, vv);

    // x3 = r^2 - j - 2v
    fp2_sqr(rr, r);
    fp2_sub(r, j, r);
    fp2_dbl(vv, t1);
    fp2_sub(r, t1, r);

    // y3 = r(v - x3) - 2*s1*j
    fp2_sub(vv, r, t1);
    fp2_mul(rr, t1, r + 12);
    fp2_mul(py, j, t1);
    fp2_dbl(t1, t1);
    fp2_sub(r + 12, t1, r + 12);

    // z3 = (z1 + h)^2 - z1z1 - hh
    fp2_add(pz, h, t1);
    fp2_sqr(t1, t1);
    fp2_sub(t1, z1z1, t1);
    fp2_sub(t1, hh, r + 24);
}

// ============================================================
// G2 scalar multiplication using windowed method (w=4)
// scalar: 4 x uint64_t in non-Montgomery integer form
// ============================================================

void bls12_381_g2_scalar_mul(const uint64_t p[36], const uint64_t scalar[4], uint64_t r[36]) {
    uint64_t table[16 * 36];

    g2_set_id(table);
    memcpy(table + 36, p, 288);
    for (int i = 2; i < 16; i++) {
        g2_add(table + (i - 1) * 36, p, table + i * 36);
    }

    uint8_t nibbles[64];
    for (int i = 0; i < 4; i++) {
        uint64_t word = scalar[i];
        for (int j = 0; j < 16; j++) {
            nibbles[i * 16 + j] = (uint8_t)(word & 0xF);
            word >>= 4;
        }
    }

    int top = 63;
    while (top >= 0 && nibbles[top] == 0) top--;

    if (top < 0) {
        g2_set_id(r);
        return;
    }

    uint64_t result[36];
    memcpy(result, table + nibbles[top] * 36, 288);

    for (int i = top - 1; i >= 0; i--) {
        uint64_t tmp[36];
        g2_dbl(result, tmp); memcpy(result, tmp, 288);
        g2_dbl(result, tmp); memcpy(result, tmp, 288);
        g2_dbl(result, tmp); memcpy(result, tmp, 288);
        g2_dbl(result, tmp); memcpy(result, tmp, 288);
        if (nibbles[i]) {
            g2_add(result, table + nibbles[i] * 36, tmp);
            memcpy(result, tmp, 288);
        }
    }

    memcpy(r, result, 288);
}

// ============================================================
// Exported G2 point operations
// ============================================================

void bls12_381_g2_point_add(const uint64_t p[36], const uint64_t q[36], uint64_t r[36]) {
    g2_add(p, q, r);
}

void bls12_381_g2_point_double(const uint64_t p[36], uint64_t r[36]) {
    g2_dbl(p, r);
}

void bls12_381_g2_point_add_mixed(const uint64_t p[36], const uint64_t q_aff[24], uint64_t r[36]) {
    g2_add_mixed(p, q_aff, r);
}

// ============================================================
// Exported Fp operations: inv, sqrt
// ============================================================

void bls12_381_fp_inv_ext(const uint64_t a[6], uint64_t r[6]) {
    fp_inv(a, r);
}

// Fp sqrt via a^((p+1)/4). Returns 1 if sqrt exists, 0 otherwise.
int bls12_381_fp_sqrt(const uint64_t a[6], uint64_t r[6]) {
    // Compute (p+1)/4
    uint64_t exp[6];
    for (int i = 0; i < 6; i++) exp[i] = FP381_P[i];
    // p+1
    uint64_t carry = 0;
    for (int i = 0; i < 6; i++) {
        uint64_t sum = exp[i] + (i == 0 ? 1 : 0) + carry;
        carry = (sum < exp[i]) || (i == 0 && sum == 0) ? 1 : 0;
        exp[i] = sum;
    }
    // >>2
    for (int i = 0; i < 5; i++) {
        exp[i] = (exp[i] >> 2) | (exp[i+1] << 62);
    }
    exp[5] >>= 2;

    // Square-and-multiply
    memcpy(r, FP381_ONE, 48);
    uint64_t base[6];
    memcpy(base, a, 48);
    for (int i = 0; i < 6; i++) {
        uint64_t word = exp[i];
        for (int bit = 0; bit < 64; bit++) {
            if ((word >> bit) & 1) fp_mul(r, base, r);
            fp_sqr(base, base);
        }
    }

    // Verify: r^2 == a?
    uint64_t check[6];
    fp_sqr(r, check);
    for (int i = 0; i < 6; i++) {
        if (check[i] != a[i]) return 0;
    }
    return 1;
}

// ============================================================
// G2 wide scalar multiplication (for cofactor clearing, up to 640 bits)
// scalar: n_limbs x uint64_t
// ============================================================

void bls12_381_g2_scalar_mul_wide(const uint64_t p[36], const uint64_t *scalar,
                                   int n_limbs, uint64_t r[36]) {
    uint64_t table[16 * 36];

    g2_set_id(table);
    memcpy(table + 36, p, 288);
    for (int i = 2; i < 16; i++) {
        g2_add(table + (i - 1) * 36, p, table + i * 36);
    }

    int total_nibbles = n_limbs * 16;
    uint8_t *nibbles = (uint8_t *)calloc(total_nibbles, 1);
    for (int i = 0; i < n_limbs; i++) {
        uint64_t word = scalar[i];
        for (int j = 0; j < 16; j++) {
            nibbles[i * 16 + j] = (uint8_t)(word & 0xF);
            word >>= 4;
        }
    }

    int top = total_nibbles - 1;
    while (top >= 0 && nibbles[top] == 0) top--;

    if (top < 0) {
        g2_set_id(r);
        free(nibbles);
        return;
    }

    uint64_t result[36];
    memcpy(result, table + nibbles[top] * 36, 288);

    for (int i = top - 1; i >= 0; i--) {
        uint64_t tmp[36];
        g2_dbl(result, tmp); memcpy(result, tmp, 288);
        g2_dbl(result, tmp); memcpy(result, tmp, 288);
        g2_dbl(result, tmp); memcpy(result, tmp, 288);
        g2_dbl(result, tmp); memcpy(result, tmp, 288);
        if (nibbles[i]) {
            g2_add(result, table + nibbles[i] * 36, tmp);
            memcpy(result, tmp, 288);
        }
    }

    memcpy(r, result, 288);
    free(nibbles);
}
