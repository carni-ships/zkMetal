// Stark252 field CIOS Montgomery arithmetic (CPU-side).
// p = 2^251 + 17 * 2^192 + 1
//   = 0x0800000000000011000000000000000000000000000000000000000000000001
// 4 x uint64_t limbs, little-endian Montgomery form.

#include "NeonFieldOps.h"
#include <string.h>

typedef unsigned __int128 uint128_t;

// ============================================================
// Stark252 Fp constants
// ============================================================

static const uint64_t SK_P[4] = {
    0x0000000000000001ULL, 0x0000000000000000ULL,
    0x0000000000000000ULL, 0x0800000000000011ULL
};

// -p^{-1} mod 2^64
// p mod 2^64 = 1, so p^{-1} mod 2^64 = 1, and -p^{-1} = 2^64 - 1
static const uint64_t SK_INV = 0xFFFFFFFFFFFFFFFFULL;

// R = 2^256 mod p (Montgomery form of 1)
static const uint64_t SK_ONE[4] = {
    0xFFFFFFFFFFFFFFE1ULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0x07FFFFFFFFFFFDF0ULL
};

// R^2 mod p
static const uint64_t SK_R2[4] = {
    0xFFFFFD737E000401ULL, 0x00000001330FFFFFULL,
    0xFFFFFFFFFF6F8000ULL, 0x07FFD4AB5E008810ULL
};

// ============================================================
// CIOS Montgomery multiplication: r = a * b * R^{-1} mod p
// ============================================================

void stark252_fp_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t t0=0,t1=0,t2=0,t3=0,t4=0;
    #define SK_ITER(I) { \
        uint128_t w; uint64_t c; \
        w=(uint128_t)a[I]*b[0]+t0; t0=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[1]+t1+c; t1=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[2]+t2+c; t2=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[3]+t3+c; t3=(uint64_t)w; c=(uint64_t)(w>>64); \
        t4+=c; \
        uint64_t m=t0*SK_INV; \
        w=(uint128_t)m*SK_P[0]+t0; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*SK_P[1]+t1+c; t0=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*SK_P[2]+t2+c; t1=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*SK_P[3]+t3+c; t2=(uint64_t)w; c=(uint64_t)(w>>64); \
        t3=t4+c; t4=0; \
    }
    SK_ITER(0) SK_ITER(1) SK_ITER(2) SK_ITER(3)
    #undef SK_ITER

    // Conditional subtraction
    uint64_t borrow=0; uint64_t r0,r1,r2,r3; uint128_t d;
    d=(uint128_t)t0-SK_P[0]-borrow; r0=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t1-SK_P[1]-borrow; r1=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t2-SK_P[2]-borrow; r2=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t3-SK_P[3]-borrow; r3=(uint64_t)d; borrow=(d>>127)&1;
    if(!borrow){r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;}
    else{r[0]=t0;r[1]=t1;r[2]=t2;r[3]=t3;}
}

// ============================================================
// CIOS Montgomery squaring (reuses mul for now)
// ============================================================

void stark252_fp_sqr(const uint64_t a[4], uint64_t r[4]) {
    stark252_fp_mul(a, a, r);
}

// ============================================================
// Modular addition: r = (a + b) mod p
// ============================================================

void stark252_fp_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t w; uint64_t c=0;
    w=(uint128_t)a[0]+b[0]; r[0]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[1]+b[1]+c; r[1]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[2]+b[2]+c; r[2]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[3]+b[3]+c; r[3]=(uint64_t)w; c=(uint64_t)(w>>64);
    // Conditional subtraction of p
    uint64_t borrow=0; uint64_t r0,r1,r2,r3; uint128_t d;
    d=(uint128_t)r[0]-SK_P[0]; r0=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[1]-SK_P[1]-borrow; r1=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[2]-SK_P[2]-borrow; r2=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[3]-SK_P[3]-borrow; r3=(uint64_t)d; borrow=(d>>127)&1;
    if(c||!borrow){r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;}
}

// ============================================================
// Modular subtraction: r = (a - b) mod p
// ============================================================

void stark252_fp_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t d; uint64_t borrow=0;
    d=(uint128_t)a[0]-b[0]; r[0]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[1]-b[1]-borrow; r[1]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[2]-b[2]-borrow; r[2]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[3]-b[3]-borrow; r[3]=(uint64_t)d; borrow=(d>>127)&1;
    if(borrow){
        uint64_t c=0;
        d=(uint128_t)r[0]+SK_P[0]; r[0]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[1]+SK_P[1]+c; r[1]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[2]+SK_P[2]+c; r[2]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[3]+SK_P[3]+c; r[3]=(uint64_t)d;
    }
}

// ============================================================
// Modular negation: r = (-a) mod p
// ============================================================

void stark252_fp_neg(const uint64_t a[4], uint64_t r[4]) {
    if((a[0] | a[1] | a[2] | a[3]) == 0) {
        r[0]=0; r[1]=0; r[2]=0; r[3]=0;
        return;
    }
    uint128_t d; uint64_t borrow=0;
    d=(uint128_t)SK_P[0]-a[0]; r[0]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)SK_P[1]-a[1]-borrow; r[1]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)SK_P[2]-a[2]-borrow; r[2]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)SK_P[3]-a[3]-borrow; r[3]=(uint64_t)d;
}
