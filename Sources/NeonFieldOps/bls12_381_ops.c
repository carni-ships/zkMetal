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
    fr_mul(a, a, r);
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
    fp_mul(a, a, r);
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
