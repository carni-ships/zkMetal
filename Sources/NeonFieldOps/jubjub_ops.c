// Jubjub twisted Edwards curve point operations in C with CIOS Montgomery arithmetic.
// Curve: a*x^2 + y^2 = 1 + d*x^2*y^2 over BLS12-381 Fr
// a = -1, d = -(10240/10241) mod p
// Extended coordinates: (X, Y, Z, T) where x = X/Z, y = Y/Z, T = XY/Z
// Each coordinate is 4 x uint64_t in BLS12-381 Fr Montgomery form.

#include "NeonFieldOps.h"
#include <string.h>

typedef unsigned __int128 uint128_t;

// ============================================================
// BLS12-381 Fr field constants (Jubjub base field)
// p = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
// ============================================================

static const uint64_t JJB_P[4] = {
    0xffffffff00000001ULL, 0x53bda402fffe5bfeULL,
    0x3339d80809a1d805ULL, 0x73eda753299d7d48ULL
};
static const uint64_t JJB_INV = 0xfffffffeffffffffULL;  // -p^{-1} mod 2^64
static const uint64_t JJB_ONE[4] = {  // R mod p (Montgomery form of 1)
    0x00000001fffffffeULL, 0x5884b7fa00034802ULL,
    0x998c4fefecbc4ff5ULL, 0x1824b159acc5056fULL
};
static const uint64_t JJB_ZERO[4] = {0, 0, 0, 0};

// R^2 mod p (for converting integers to Montgomery form)
static const uint64_t JJB_R2[4] = {
    0xc999e990f3f29c6dULL, 0x2b6cedcb87925c23ULL,
    0x05d314967254398fULL, 0x0748d9d99f59ff11ULL
};

// ============================================================
// CIOS Montgomery multiplication for BLS12-381 Fr
// ============================================================

static inline void jjb_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t t0=0,t1=0,t2=0,t3=0,t4=0;
    #define JJB_ITER(I) { \
        uint128_t w; uint64_t c; \
        w=(uint128_t)a[I]*b[0]+t0; t0=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[1]+t1+c; t1=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[2]+t2+c; t2=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[3]+t3+c; t3=(uint64_t)w; c=(uint64_t)(w>>64); \
        t4+=c; \
        uint64_t m=t0*JJB_INV; \
        w=(uint128_t)m*JJB_P[0]+t0; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*JJB_P[1]+t1+c; t0=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*JJB_P[2]+t2+c; t1=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*JJB_P[3]+t3+c; t2=(uint64_t)w; c=(uint64_t)(w>>64); \
        t3=t4+c; t4=0; \
    }
    JJB_ITER(0) JJB_ITER(1) JJB_ITER(2) JJB_ITER(3)
    #undef JJB_ITER
    uint64_t borrow=0; uint64_t r0,r1,r2,r3; uint128_t d;
    d=(uint128_t)t0-JJB_P[0]-borrow; r0=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t1-JJB_P[1]-borrow; r1=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t2-JJB_P[2]-borrow; r2=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t3-JJB_P[3]-borrow; r3=(uint64_t)d; borrow=(d>>127)&1;
    if(!borrow){r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;}
    else{r[0]=t0;r[1]=t1;r[2]=t2;r[3]=t3;}
}

static inline void jjb_sqr(const uint64_t a[4], uint64_t r[4]) {
    jjb_mul(a, a, r);
}

static inline void jjb_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t w; uint64_t c=0;
    w=(uint128_t)a[0]+b[0]; r[0]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[1]+b[1]+c; r[1]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[2]+b[2]+c; r[2]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[3]+b[3]+c; r[3]=(uint64_t)w; c=(uint64_t)(w>>64);
    uint64_t borrow=0; uint64_t r0,r1,r2,r3; uint128_t d;
    d=(uint128_t)r[0]-JJB_P[0]; r0=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[1]-JJB_P[1]-borrow; r1=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[2]-JJB_P[2]-borrow; r2=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[3]-JJB_P[3]-borrow; r3=(uint64_t)d; borrow=(d>>127)&1;
    if(c||!borrow){r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;}
}

static inline void jjb_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t d; uint64_t borrow=0;
    d=(uint128_t)a[0]-b[0]; r[0]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[1]-b[1]-borrow; r[1]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[2]-b[2]-borrow; r[2]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[3]-b[3]-borrow; r[3]=(uint64_t)d; borrow=(d>>127)&1;
    if(borrow){
        uint64_t c=0;
        d=(uint128_t)r[0]+JJB_P[0]; r[0]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[1]+JJB_P[1]+c; r[1]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[2]+JJB_P[2]+c; r[2]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[3]+JJB_P[3]+c; r[3]=(uint64_t)d;
    }
}

static inline void jjb_neg(const uint64_t a[4], uint64_t r[4]) {
    if ((a[0] | a[1] | a[2] | a[3]) == 0) {
        r[0] = r[1] = r[2] = r[3] = 0;
        return;
    }
    uint128_t d; uint64_t borrow = 0;
    d = (uint128_t)JJB_P[0] - a[0]; r[0] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)JJB_P[1] - a[1] - borrow; r[1] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)JJB_P[2] - a[2] - borrow; r[2] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)JJB_P[3] - a[3] - borrow; r[3] = (uint64_t)d;
}

static inline void jjb_dbl(const uint64_t a[4], uint64_t r[4]) {
    jjb_add(a, a, r);
}

static inline int jjb_is_zero(const uint64_t a[4]) {
    return (a[0] | a[1] | a[2] | a[3]) == 0;
}

// Convert integer (4 x uint64_t) to Montgomery form
static void jjb_to_mont(const uint64_t a[4], uint64_t r[4]) {
    jjb_mul(a, JJB_R2, r);
}

// ============================================================
// Jubjub curve constants (computed at first use)
// ============================================================

// a = -1 in Montgomery form (= p - R mod p)
// d = -(10240/10241) mod p in Montgomery form
static uint64_t JJB_A_MONT[4];
static uint64_t JJB_D_MONT[4];
static int jjb_constants_initialized = 0;

static void jjb_init_constants(void) {
    if (jjb_constants_initialized) return;

    // a = -1: negate Montgomery 1
    jjb_neg(JJB_ONE, JJB_A_MONT);

    // d raw value from Swift: 19257038036680949359750312669786877991949435402254120286184196891950884077233
    // These are the standard-form limbs; convert to Montgomery via mul by R^2
    uint64_t d_raw[4] = {
        0x01065fd6d6343eb1ULL, 0x292d7f6d37579d26ULL,
        0xf5fd9207e6bd7fd4ULL, 0x2a9318e74bfa2b48ULL
    };
    jjb_to_mont(d_raw, JJB_D_MONT);

    jjb_constants_initialized = 1;
}

// ============================================================
// Extended twisted Edwards point operations
// Layout: [X0..X3, Y0..Y3, Z0..Z3, T0..T3] = 16 uint64_t
// Identity: (0, 1, 1, 0)
// ============================================================

static inline void jjb_pt_set_id(uint64_t p[16]) {
    memcpy(p, JJB_ZERO, 32);       // X = 0
    memcpy(p + 4, JJB_ONE, 32);    // Y = 1
    memcpy(p + 8, JJB_ONE, 32);    // Z = 1
    memcpy(p + 12, JJB_ZERO, 32);  // T = 0
}

static inline int jjb_pt_is_id(const uint64_t p[16]) {
    // Identity: X = 0 and Y = Z
    return jjb_is_zero(p) &&
           p[4] == p[8] && p[5] == p[9] && p[6] == p[10] && p[7] == p[11];
}

// Point addition using extended coordinates (unified formula)
// For a*x^2 + y^2 = 1 + d*x^2*y^2 with a = -1:
// A = X1*X2, B = Y1*Y2, C = d*T1*T2, D = Z1*Z2
// E = (X1+Y1)*(X2+Y2) - A - B
// F = D - C, G = D + C
// H = B - a*A = B + A  (since a = -1)
// X3 = E*F, Y3 = G*H, T3 = E*H, Z3 = F*G
static void jjb_pt_add(const uint64_t p[16], const uint64_t q[16], uint64_t r[16]) {
    jjb_init_constants();

    const uint64_t *px = p, *py = p+4, *pz = p+8, *pt = p+12;
    const uint64_t *qx = q, *qy = q+4, *qz = q+8, *qt = q+12;

    uint64_t aa[4], bb[4], cc[4], dd[4], ee[4], ff[4], gg[4], hh[4];
    uint64_t t1[4], t2[4], t3[4];

    jjb_mul(px, qx, aa);           // A = X1*X2
    jjb_mul(py, qy, bb);           // B = Y1*Y2
    jjb_mul(pt, qt, t1);
    jjb_mul(t1, JJB_D_MONT, cc);   // C = d*T1*T2
    jjb_mul(pz, qz, dd);           // D = Z1*Z2

    jjb_add(px, py, t1);           // X1+Y1
    jjb_add(qx, qy, t2);           // X2+Y2
    jjb_mul(t1, t2, t3);           // (X1+Y1)*(X2+Y2)
    jjb_add(aa, bb, t1);           // A+B
    jjb_sub(t3, t1, ee);           // E = (X1+Y1)*(X2+Y2) - A - B

    jjb_sub(dd, cc, ff);           // F = D - C
    jjb_add(dd, cc, gg);           // G = D + C
    // H = B - a*A = B - (-1)*A = B + A
    jjb_add(bb, aa, hh);           // H = B + A

    jjb_mul(ee, ff, r);            // X3 = E*F
    jjb_mul(gg, hh, r + 4);        // Y3 = G*H
    jjb_mul(ff, gg, r + 8);        // Z3 = F*G
    jjb_mul(ee, hh, r + 12);       // T3 = E*H
}

// Point doubling (more efficient than generic add)
// A = X^2, B = Y^2, C = 2*Z^2
// D = a*A = -A  (since a = -1)
// E = (X+Y)^2 - A - B
// G = D + B = B - A, F = G - C, H = D - B = -A - B
// X3 = E*F, Y3 = G*H, T3 = E*H, Z3 = F*G
static void jjb_pt_dbl(const uint64_t p[16], uint64_t r[16]) {
    jjb_init_constants();

    const uint64_t *px = p, *py = p+4, *pz = p+8;

    uint64_t aa[4], bb[4], cc[4], dd[4], ee[4], ff[4], gg[4], hh[4];
    uint64_t t1[4], t2[4];

    jjb_sqr(px, aa);               // A = X^2
    jjb_sqr(py, bb);               // B = Y^2
    jjb_sqr(pz, t1);
    jjb_dbl(t1, cc);               // C = 2*Z^2
    jjb_neg(aa, dd);               // D = a*A = -A  (since a = -1)

    jjb_add(px, py, t1);
    jjb_sqr(t1, t2);
    jjb_add(aa, bb, t1);
    jjb_sub(t2, t1, ee);           // E = (X+Y)^2 - A - B

    jjb_add(dd, bb, gg);           // G = D + B = B - A
    jjb_sub(gg, cc, ff);           // F = G - C
    jjb_sub(dd, bb, hh);           // H = D - B = -A - B

    jjb_mul(ee, ff, r);            // X3 = E*F
    jjb_mul(gg, hh, r + 4);        // Y3 = G*H
    jjb_mul(ff, gg, r + 8);        // Z3 = F*G
    jjb_mul(ee, hh, r + 12);       // T3 = E*H
}

// ============================================================
// Scalar multiplication using windowed method (w=4)
// scalar: 4 x uint64_t in non-Montgomery integer form (little-endian)
// ============================================================

void jubjub_scalar_mul(const uint64_t p[16], const uint64_t scalar[4], uint64_t r[16]) {
    // Precompute table[0..15] = [identity, P, 2P, ..., 15P]
    uint64_t table[16 * 16];  // 16 extended points

    // table[0] = identity
    jjb_pt_set_id(table);
    // table[1] = P
    memcpy(table + 16, p, 128);
    // table[i] = table[i-1] + P for i = 2..15
    for (int i = 2; i < 16; i++) {
        jjb_pt_add(table + (i - 1) * 16, p, table + i * 16);
    }

    // Extract nibbles from scalar (MSB first): 256 bits = 64 nibbles
    uint8_t nibbles[64];
    for (int i = 0; i < 4; i++) {
        uint64_t word = scalar[i];
        for (int j = 0; j < 16; j++) {
            nibbles[i * 16 + j] = (uint8_t)(word & 0xF);
            word >>= 4;
        }
    }

    // Find highest non-zero nibble
    int top = 63;
    while (top >= 0 && nibbles[top] == 0) top--;

    if (top < 0) {
        jjb_pt_set_id(r);
        return;
    }

    // Start with table[top nibble]
    uint64_t result[16];
    memcpy(result, table + nibbles[top] * 16, 128);

    // Process remaining nibbles from MSB to LSB
    for (int i = top - 1; i >= 0; i--) {
        uint64_t tmp[16];
        jjb_pt_dbl(result, tmp); memcpy(result, tmp, 128);
        jjb_pt_dbl(result, tmp); memcpy(result, tmp, 128);
        jjb_pt_dbl(result, tmp); memcpy(result, tmp, 128);
        jjb_pt_dbl(result, tmp); memcpy(result, tmp, 128);
        if (nibbles[i]) {
            jjb_pt_add(result, table + nibbles[i] * 16, tmp);
            memcpy(result, tmp, 128);
        }
    }

    memcpy(r, result, 128);
}

// Exported point addition
void jubjub_point_add(const uint64_t p[16], const uint64_t q[16], uint64_t r[16]) {
    jjb_pt_add(p, q, r);
}

// Exported point doubling
void jubjub_point_double(const uint64_t p[16], uint64_t r[16]) {
    jjb_pt_dbl(p, r);
}
