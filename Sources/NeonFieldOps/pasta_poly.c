// Pasta (Pallas + Vesta) polynomial operations in C.
// Pallas Fr = Vesta Fp (scalar field of Pallas = base field of Vesta)
// Vesta Fr = Pallas Fp (scalar field of Vesta = base field of Pallas)
// All values in Montgomery form (4 x uint64_t).

#include "NeonFieldOps.h"
#include <string.h>
#include <stdlib.h>

typedef unsigned __int128 uint128_t;

// ============================================================
// Pallas Fr constants (= Vesta Fp modulus)
// p = 0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001
// ============================================================

static const uint64_t PFR_P[4] = {
    0x8c46eb2100000001ULL, 0x224698fc0994a8ddULL,
    0x0000000000000000ULL, 0x4000000000000000ULL
};
static const uint64_t PFR_INV = 0x8c46eb20ffffffffULL;  // -p^{-1} mod 2^64
static const uint64_t PFR_ONE[4] = {  // R mod p (Montgomery form of 1)
    0x5b2b3e9cfffffffdULL, 0x992c350be3420567ULL,
    0xffffffffffffffffULL, 0x3fffffffffffffffULL
};

// ============================================================
// Vesta Fr constants (= Pallas Fp modulus)
// p = 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001
// ============================================================

static const uint64_t VFR_P[4] = {
    0x992d30ed00000001ULL, 0x224698fc094cf91bULL,
    0x0000000000000000ULL, 0x4000000000000000ULL
};
static const uint64_t VFR_INV = 0x992d30ecffffffffULL;  // -p^{-1} mod 2^64
static const uint64_t VFR_ONE[4] = {  // R mod p (Montgomery form of 1)
    0x34786d38fffffffdULL, 0x992c350be41914adULL,
    0xffffffffffffffffULL, 0x3fffffffffffffffULL
};

// ============================================================
// CIOS Montgomery multiplication for Pallas Fr (= Vesta Fp field)
// ============================================================

static inline void pfr_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t t0=0,t1=0,t2=0,t3=0,t4=0;
    #define PFR_ITER(I) { \
        uint128_t w; uint64_t c; \
        w=(uint128_t)a[I]*b[0]+t0; t0=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[1]+t1+c; t1=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[2]+t2+c; t2=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[3]+t3+c; t3=(uint64_t)w; c=(uint64_t)(w>>64); \
        t4+=c; \
        uint64_t m=t0*PFR_INV; \
        w=(uint128_t)m*PFR_P[0]+t0; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*PFR_P[1]+t1+c; t0=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*PFR_P[2]+t2+c; t1=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*PFR_P[3]+t3+c; t2=(uint64_t)w; c=(uint64_t)(w>>64); \
        t3=t4+c; t4=0; \
    }
    PFR_ITER(0) PFR_ITER(1) PFR_ITER(2) PFR_ITER(3)
    #undef PFR_ITER
    uint64_t borrow=0; uint64_t r0,r1,r2,r3; uint128_t d;
    d=(uint128_t)t0-PFR_P[0]-borrow; r0=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t1-PFR_P[1]-borrow; r1=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t2-PFR_P[2]-borrow; r2=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t3-PFR_P[3]-borrow; r3=(uint64_t)d; borrow=(d>>127)&1;
    if(!borrow){r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;}
    else{r[0]=t0;r[1]=t1;r[2]=t2;r[3]=t3;}
}

static inline void pfr_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t w; uint64_t c=0;
    w=(uint128_t)a[0]+b[0]; r[0]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[1]+b[1]+c; r[1]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[2]+b[2]+c; r[2]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[3]+b[3]+c; r[3]=(uint64_t)w; c=(uint64_t)(w>>64);
    uint64_t borrow=0; uint64_t r0,r1,r2,r3; uint128_t d;
    d=(uint128_t)r[0]-PFR_P[0]; r0=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[1]-PFR_P[1]-borrow; r1=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[2]-PFR_P[2]-borrow; r2=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[3]-PFR_P[3]-borrow; r3=(uint64_t)d; borrow=(d>>127)&1;
    if(c||!borrow){r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;}
}

static inline void pfr_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t d; uint64_t borrow=0;
    d=(uint128_t)a[0]-b[0]; r[0]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[1]-b[1]-borrow; r[1]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[2]-b[2]-borrow; r[2]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[3]-b[3]-borrow; r[3]=(uint64_t)d; borrow=(d>>127)&1;
    if(borrow){
        uint64_t c2=0;
        d=(uint128_t)r[0]+PFR_P[0]; r[0]=(uint64_t)d; c2=(uint64_t)(d>>64);
        d=(uint128_t)r[1]+PFR_P[1]+c2; r[1]=(uint64_t)d; c2=(uint64_t)(d>>64);
        d=(uint128_t)r[2]+PFR_P[2]+c2; r[2]=(uint64_t)d; c2=(uint64_t)(d>>64);
        d=(uint128_t)r[3]+PFR_P[3]+c2; r[3]=(uint64_t)d;
    }
}

// ============================================================
// CIOS Montgomery multiplication for Vesta Fr (= Pallas Fp field)
// ============================================================

static inline void vfr_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t t0=0,t1=0,t2=0,t3=0,t4=0;
    #define VFR_ITER(I) { \
        uint128_t w; uint64_t c; \
        w=(uint128_t)a[I]*b[0]+t0; t0=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[1]+t1+c; t1=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[2]+t2+c; t2=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[3]+t3+c; t3=(uint64_t)w; c=(uint64_t)(w>>64); \
        t4+=c; \
        uint64_t m=t0*VFR_INV; \
        w=(uint128_t)m*VFR_P[0]+t0; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*VFR_P[1]+t1+c; t0=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*VFR_P[2]+t2+c; t1=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*VFR_P[3]+t3+c; t2=(uint64_t)w; c=(uint64_t)(w>>64); \
        t3=t4+c; t4=0; \
    }
    VFR_ITER(0) VFR_ITER(1) VFR_ITER(2) VFR_ITER(3)
    #undef VFR_ITER
    uint64_t borrow=0; uint64_t r0,r1,r2,r3; uint128_t d;
    d=(uint128_t)t0-VFR_P[0]-borrow; r0=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t1-VFR_P[1]-borrow; r1=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t2-VFR_P[2]-borrow; r2=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t3-VFR_P[3]-borrow; r3=(uint64_t)d; borrow=(d>>127)&1;
    if(!borrow){r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;}
    else{r[0]=t0;r[1]=t1;r[2]=t2;r[3]=t3;}
}

static inline void vfr_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t w; uint64_t c=0;
    w=(uint128_t)a[0]+b[0]; r[0]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[1]+b[1]+c; r[1]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[2]+b[2]+c; r[2]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[3]+b[3]+c; r[3]=(uint64_t)w; c=(uint64_t)(w>>64);
    uint64_t borrow=0; uint64_t r0,r1,r2,r3; uint128_t d;
    d=(uint128_t)r[0]-VFR_P[0]; r0=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[1]-VFR_P[1]-borrow; r1=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[2]-VFR_P[2]-borrow; r2=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[3]-VFR_P[3]-borrow; r3=(uint64_t)d; borrow=(d>>127)&1;
    if(c||!borrow){r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;}
}

static inline void vfr_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t d; uint64_t borrow=0;
    d=(uint128_t)a[0]-b[0]; r[0]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[1]-b[1]-borrow; r[1]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[2]-b[2]-borrow; r[2]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[3]-b[3]-borrow; r[3]=(uint64_t)d; borrow=(d>>127)&1;
    if(borrow){
        uint64_t c2=0;
        d=(uint128_t)r[0]+VFR_P[0]; r[0]=(uint64_t)d; c2=(uint64_t)(d>>64);
        d=(uint128_t)r[1]+VFR_P[1]+c2; r[1]=(uint64_t)d; c2=(uint64_t)(d>>64);
        d=(uint128_t)r[2]+VFR_P[2]+c2; r[2]=(uint64_t)d; c2=(uint64_t)(d>>64);
        d=(uint128_t)r[3]+VFR_P[3]+c2; r[3]=(uint64_t)d;
    }
}

// ============================================================
// Pallas Fr polynomial operations
// ============================================================

/// Horner evaluation: result = coeffs[0] + coeffs[1]*z + coeffs[2]*z^2 + ...
void pallas_fr_horner_eval(const uint64_t *coeffs, int n, const uint64_t z[4],
                            uint64_t result[4]) {
    if (n == 0) { memset(result, 0, 32); return; }
    memcpy(result, coeffs + (n - 1) * 4, 32);
    for (int i = n - 2; i >= 0; i--) {
        uint64_t tmp[4];
        pfr_mul(result, z, tmp);
        pfr_add(tmp, coeffs + i * 4, result);
    }
}

/// Synthetic division: quotient = (p(x) - p(z)) / (x - z)
/// q[n-2] = coeffs[n-1], q[i] = coeffs[i+1] + z * q[i+1]
void pallas_fr_synthetic_div(const uint64_t *coeffs, const uint64_t z[4],
                              int n, uint64_t *quotient) {
    if (n < 2) return;
    memcpy(quotient + (n - 2) * 4, coeffs + (n - 1) * 4, 32);
    for (int i = n - 3; i >= 0; i--) {
        uint64_t tmp[4];
        pfr_mul(z, quotient + (i + 1) * 4, tmp);
        pfr_add(coeffs + (i + 1) * 4, tmp, quotient + i * 4);
    }
}

/// Fused evaluation + synthetic division in one pass.
void pallas_fr_eval_and_div(const uint64_t *coeffs, int n, const uint64_t z[4],
                             uint64_t eval_out[4], uint64_t *quotient) {
    if (n == 0) { memset(eval_out, 0, 32); return; }
    if (n == 1) { memcpy(eval_out, coeffs, 32); return; }
    memcpy(quotient + (n - 2) * 4, coeffs + (n - 1) * 4, 32);
    for (int i = n - 3; i >= 0; i--) {
        uint64_t tmp[4];
        pfr_mul(z, quotient + (i + 1) * 4, tmp);
        pfr_add(coeffs + (i + 1) * 4, tmp, quotient + i * 4);
    }
    uint64_t tmp[4];
    pfr_mul(z, quotient, tmp);
    pfr_add(coeffs, tmp, eval_out);
}

/// Batch scalar multiply: data[i] *= scalar
void pallas_fr_batch_mul_scalar(uint64_t *data, const uint64_t scalar[4], int n) {
    for (int i = 0; i < n; i++) {
        uint64_t tmp[4];
        pfr_mul(data + i * 4, scalar, tmp);
        memcpy(data + i * 4, tmp, 32);
    }
}

/// Batch add: result[i] = a[i] + b[i]
void pallas_fr_batch_add(const uint64_t *a, const uint64_t *b,
                           uint64_t *result, int n) {
    for (int i = 0; i < n; i++) {
        pfr_add(a + i * 4, b + i * 4, result + i * 4);
    }
}

/// Batch sub: result[i] = a[i] - b[i]
void pallas_fr_batch_sub(const uint64_t *a, const uint64_t *b,
                           uint64_t *result, int n) {
    for (int i = 0; i < n; i++) {
        pfr_sub(a + i * 4, b + i * 4, result + i * 4);
    }
}

/// Inner product: result = sum(a[i] * b[i])
void pallas_fr_inner_product(const uint64_t *a, const uint64_t *b,
                              int n, uint64_t result[4]) {
    uint64_t acc[4] = {0, 0, 0, 0};
    for (int i = 0; i < n; i++) {
        uint64_t prod[4];
        pfr_mul(a + i * 4, b + i * 4, prod);
        pfr_add(acc, prod, acc);
    }
    memcpy(result, acc, 32);
}

// ============================================================
// Vesta Fr polynomial operations
// ============================================================

/// Horner evaluation: result = coeffs[0] + coeffs[1]*z + coeffs[2]*z^2 + ...
void vesta_fr_horner_eval(const uint64_t *coeffs, int n, const uint64_t z[4],
                           uint64_t result[4]) {
    if (n == 0) { memset(result, 0, 32); return; }
    memcpy(result, coeffs + (n - 1) * 4, 32);
    for (int i = n - 2; i >= 0; i--) {
        uint64_t tmp[4];
        vfr_mul(result, z, tmp);
        vfr_add(tmp, coeffs + i * 4, result);
    }
}

/// Synthetic division: quotient = (p(x) - p(z)) / (x - z)
void vesta_fr_synthetic_div(const uint64_t *coeffs, const uint64_t z[4],
                             int n, uint64_t *quotient) {
    if (n < 2) return;
    memcpy(quotient + (n - 2) * 4, coeffs + (n - 1) * 4, 32);
    for (int i = n - 3; i >= 0; i--) {
        uint64_t tmp[4];
        vfr_mul(z, quotient + (i + 1) * 4, tmp);
        vfr_add(coeffs + (i + 1) * 4, tmp, quotient + i * 4);
    }
}

/// Fused evaluation + synthetic division in one pass.
void vesta_fr_eval_and_div(const uint64_t *coeffs, int n, const uint64_t z[4],
                            uint64_t eval_out[4], uint64_t *quotient) {
    if (n == 0) { memset(eval_out, 0, 32); return; }
    if (n == 1) { memcpy(eval_out, coeffs, 32); return; }
    memcpy(quotient + (n - 2) * 4, coeffs + (n - 1) * 4, 32);
    for (int i = n - 3; i >= 0; i--) {
        uint64_t tmp[4];
        vfr_mul(z, quotient + (i + 1) * 4, tmp);
        vfr_add(coeffs + (i + 1) * 4, tmp, quotient + i * 4);
    }
    uint64_t tmp[4];
    vfr_mul(z, quotient, tmp);
    vfr_add(coeffs, tmp, eval_out);
}

/// Batch scalar multiply: data[i] *= scalar
void vesta_fr_batch_mul_scalar(uint64_t *data, const uint64_t scalar[4], int n) {
    for (int i = 0; i < n; i++) {
        uint64_t tmp[4];
        vfr_mul(data + i * 4, scalar, tmp);
        memcpy(data + i * 4, tmp, 32);
    }
}

/// Batch add: result[i] = a[i] + b[i]
void vesta_fr_batch_add(const uint64_t *a, const uint64_t *b,
                          uint64_t *result, int n) {
    for (int i = 0; i < n; i++) {
        vfr_add(a + i * 4, b + i * 4, result + i * 4);
    }
}

/// Batch sub: result[i] = a[i] - b[i]
void vesta_fr_batch_sub(const uint64_t *a, const uint64_t *b,
                          uint64_t *result, int n) {
    for (int i = 0; i < n; i++) {
        vfr_sub(a + i * 4, b + i * 4, result + i * 4);
    }
}

/// Inner product: result = sum(a[i] * b[i])
void vesta_fr_inner_product(const uint64_t *a, const uint64_t *b,
                             int n, uint64_t result[4]) {
    uint64_t acc[4] = {0, 0, 0, 0};
    for (int i = 0; i < n; i++) {
        uint64_t prod[4];
        vfr_mul(a + i * 4, b + i * 4, prod);
        vfr_add(acc, prod, acc);
    }
    memcpy(result, acc, 32);
}
