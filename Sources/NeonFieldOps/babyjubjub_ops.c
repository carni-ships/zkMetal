// BabyJubjub twisted Edwards curve point operations in C with CIOS Montgomery arithmetic.
// Curve: a*x^2 + y^2 = 1 + d*x^2*y^2 over BN254 Fr
// a = 168700, d = 168696
// Extended coordinates: (X, Y, Z, T) where x = X/Z, y = Y/Z, T = XY/Z
// Each coordinate is 4 x uint64_t in BN254 Fr Montgomery form.

#include "NeonFieldOps.h"
#include <string.h>

typedef unsigned __int128 uint128_t;

// ============================================================
// BN254 Fr field constants (BabyJubjub base field)
// ============================================================

static const uint64_t BJJ_P[4] = {
    0x43e1f593f0000001ULL, 0x2833e84879b97091ULL,
    0xb85045b68181585dULL, 0x30644e72e131a029ULL
};
static const uint64_t BJJ_INV = 0xc2e1f593efffffffULL;  // -p^{-1} mod 2^64
static const uint64_t BJJ_ONE[4] = {  // R mod p (Montgomery form of 1)
    0xac96341c4ffffffbULL, 0x36fc76959f60cd29ULL,
    0x666ea36f7879462eULL, 0x0e0a77c19a07df2fULL
};
static const uint64_t BJJ_ZERO[4] = {0, 0, 0, 0};

// R^2 mod p (for converting integers to Montgomery form)
static const uint64_t BJJ_R2[4] = {
    0x1bb8e645ae216da7ULL, 0x53fe3ab1e35c59e3ULL,
    0x8c49833d53bb8085ULL, 0x0216d0b17f4e44a5ULL
};

// ============================================================
// CIOS Montgomery multiplication for BN254 Fr
// ============================================================

static inline void bjj_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t t0=0,t1=0,t2=0,t3=0,t4=0;
    #define BJJ_ITER(I) { \
        uint128_t w; uint64_t c; \
        w=(uint128_t)a[I]*b[0]+t0; t0=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[1]+t1+c; t1=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[2]+t2+c; t2=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[3]+t3+c; t3=(uint64_t)w; c=(uint64_t)(w>>64); \
        t4+=c; \
        uint64_t m=t0*BJJ_INV; \
        w=(uint128_t)m*BJJ_P[0]+t0; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*BJJ_P[1]+t1+c; t0=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*BJJ_P[2]+t2+c; t1=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*BJJ_P[3]+t3+c; t2=(uint64_t)w; c=(uint64_t)(w>>64); \
        t3=t4+c; t4=0; \
    }
    BJJ_ITER(0) BJJ_ITER(1) BJJ_ITER(2) BJJ_ITER(3)
    #undef BJJ_ITER
    uint64_t borrow=0; uint64_t r0,r1,r2,r3; uint128_t d;
    d=(uint128_t)t0-BJJ_P[0]-borrow; r0=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t1-BJJ_P[1]-borrow; r1=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t2-BJJ_P[2]-borrow; r2=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t3-BJJ_P[3]-borrow; r3=(uint64_t)d; borrow=(d>>127)&1;
    if(!borrow){r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;}
    else{r[0]=t0;r[1]=t1;r[2]=t2;r[3]=t3;}
}

static inline void bjj_sqr(const uint64_t a[4], uint64_t r[4]) {
    bjj_mul(a, a, r);
}

static inline void bjj_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t w; uint64_t c=0;
    w=(uint128_t)a[0]+b[0]; r[0]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[1]+b[1]+c; r[1]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[2]+b[2]+c; r[2]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[3]+b[3]+c; r[3]=(uint64_t)w; c=(uint64_t)(w>>64);
    uint64_t borrow=0; uint64_t r0,r1,r2,r3; uint128_t d;
    d=(uint128_t)r[0]-BJJ_P[0]; r0=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[1]-BJJ_P[1]-borrow; r1=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[2]-BJJ_P[2]-borrow; r2=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[3]-BJJ_P[3]-borrow; r3=(uint64_t)d; borrow=(d>>127)&1;
    if(c||!borrow){r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;}
}

static inline void bjj_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t d; uint64_t borrow=0;
    d=(uint128_t)a[0]-b[0]; r[0]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[1]-b[1]-borrow; r[1]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[2]-b[2]-borrow; r[2]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[3]-b[3]-borrow; r[3]=(uint64_t)d; borrow=(d>>127)&1;
    if(borrow){
        uint64_t c=0;
        d=(uint128_t)r[0]+BJJ_P[0]; r[0]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[1]+BJJ_P[1]+c; r[1]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[2]+BJJ_P[2]+c; r[2]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[3]+BJJ_P[3]+c; r[3]=(uint64_t)d;
    }
}

static inline void bjj_neg(const uint64_t a[4], uint64_t r[4]) {
    if ((a[0] | a[1] | a[2] | a[3]) == 0) {
        r[0] = r[1] = r[2] = r[3] = 0;
        return;
    }
    uint128_t d; uint64_t borrow = 0;
    d = (uint128_t)BJJ_P[0] - a[0]; r[0] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)BJJ_P[1] - a[1] - borrow; r[1] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)BJJ_P[2] - a[2] - borrow; r[2] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)BJJ_P[3] - a[3] - borrow; r[3] = (uint64_t)d;
}

static inline void bjj_dbl(const uint64_t a[4], uint64_t r[4]) {
    bjj_add(a, a, r);
}

static inline int bjj_is_zero(const uint64_t a[4]) {
    return (a[0] | a[1] | a[2] | a[3]) == 0;
}

// Convert integer (4 x uint64_t) to Montgomery form
static void bjj_to_mont(const uint64_t a[4], uint64_t r[4]) {
    bjj_mul(a, BJJ_R2, r);
}

// ============================================================
// BabyJubjub curve constants (computed at first use)
// ============================================================

// a = 168700 in Montgomery form
// d = 168696 in Montgomery form
// These are computed once on first call.
static uint64_t BJJ_A_MONT[4];
static uint64_t BJJ_D_MONT[4];
static int bjj_constants_initialized = 0;

static void bjj_init_constants(void) {
    if (bjj_constants_initialized) return;
    uint64_t a_raw[4] = {168700, 0, 0, 0};
    uint64_t d_raw[4] = {168696, 0, 0, 0};
    bjj_to_mont(a_raw, BJJ_A_MONT);
    bjj_to_mont(d_raw, BJJ_D_MONT);
    bjj_constants_initialized = 1;
}

// ============================================================
// Extended twisted Edwards point operations
// Layout: [X0..X3, Y0..Y3, Z0..Z3, T0..T3] = 16 uint64_t
// Identity: (0, 1, 1, 0)
// ============================================================

static inline void bjj_pt_set_id(uint64_t p[16]) {
    memcpy(p, BJJ_ZERO, 32);       // X = 0
    memcpy(p + 4, BJJ_ONE, 32);    // Y = 1
    memcpy(p + 8, BJJ_ONE, 32);    // Z = 1
    memcpy(p + 12, BJJ_ZERO, 32);  // T = 0
}

static inline int bjj_pt_is_id(const uint64_t p[16]) {
    // Identity: X = 0 and Y = Z
    return bjj_is_zero(p) &&
           p[4] == p[8] && p[5] == p[9] && p[6] == p[10] && p[7] == p[11];
}

// Point addition using extended coordinates (unified formula)
// For a*x^2 + y^2 = 1 + d*x^2*y^2 with general a:
// A = X1*X2, B = Y1*Y2, C = d*T1*T2, D = Z1*Z2
// E = (X1+Y1)*(X2+Y2) - A - B
// F = D - C, G = D + C
// H = B - a*A
// X3 = E*F, Y3 = G*H, T3 = E*H, Z3 = F*G
static void bjj_pt_add(const uint64_t p[16], const uint64_t q[16], uint64_t r[16]) {
    bjj_init_constants();

    const uint64_t *px = p, *py = p+4, *pz = p+8, *pt = p+12;
    const uint64_t *qx = q, *qy = q+4, *qz = q+8, *qt = q+12;

    uint64_t aa[4], bb[4], cc[4], dd[4], ee[4], ff[4], gg[4], hh[4];
    uint64_t t1[4], t2[4], t3[4];

    bjj_mul(px, qx, aa);           // A = X1*X2
    bjj_mul(py, qy, bb);           // B = Y1*Y2
    bjj_mul(pt, qt, t1);
    bjj_mul(t1, BJJ_D_MONT, cc);   // C = d*T1*T2
    bjj_mul(pz, qz, dd);           // D = Z1*Z2

    bjj_add(px, py, t1);           // X1+Y1
    bjj_add(qx, qy, t2);           // X2+Y2
    bjj_mul(t1, t2, t3);           // (X1+Y1)*(X2+Y2)
    bjj_add(aa, bb, t1);           // A+B
    bjj_sub(t3, t1, ee);           // E = (X1+Y1)*(X2+Y2) - A - B

    bjj_sub(dd, cc, ff);           // F = D - C
    bjj_add(dd, cc, gg);           // G = D + C
    bjj_mul(BJJ_A_MONT, aa, t1);   // a*A
    bjj_sub(bb, t1, hh);           // H = B - a*A

    bjj_mul(ee, ff, r);            // X3 = E*F
    bjj_mul(gg, hh, r + 4);        // Y3 = G*H
    bjj_mul(ff, gg, r + 8);        // Z3 = F*G
    bjj_mul(ee, hh, r + 12);       // T3 = E*H
}

// Point doubling (more efficient than generic add)
// A = X^2, B = Y^2, C = 2*Z^2
// D = a*A
// E = (X+Y)^2 - A - B
// G = D + B, F = G - C, H = D - B
// X3 = E*F, Y3 = G*H, T3 = E*H, Z3 = F*G
static void bjj_pt_dbl(const uint64_t p[16], uint64_t r[16]) {
    bjj_init_constants();

    const uint64_t *px = p, *py = p+4, *pz = p+8;

    uint64_t aa[4], bb[4], cc[4], dd[4], ee[4], ff[4], gg[4], hh[4];
    uint64_t t1[4], t2[4];

    bjj_sqr(px, aa);               // A = X^2
    bjj_sqr(py, bb);               // B = Y^2
    bjj_sqr(pz, t1);
    bjj_dbl(t1, cc);               // C = 2*Z^2
    bjj_mul(BJJ_A_MONT, aa, dd);   // D = a*A

    bjj_add(px, py, t1);
    bjj_sqr(t1, t2);
    bjj_add(aa, bb, t1);
    bjj_sub(t2, t1, ee);           // E = (X+Y)^2 - A - B

    bjj_add(dd, bb, gg);           // G = D + B
    bjj_sub(gg, cc, ff);           // F = G - C
    bjj_sub(dd, bb, hh);           // H = D - B

    bjj_mul(ee, ff, r);            // X3 = E*F
    bjj_mul(gg, hh, r + 4);        // Y3 = G*H
    bjj_mul(ff, gg, r + 8);        // Z3 = F*G
    bjj_mul(ee, hh, r + 12);       // T3 = E*H
}

// ============================================================
// Scalar multiplication using windowed method (w=4)
// scalar: 4 x uint64_t in non-Montgomery integer form (little-endian)
// ============================================================

void babyjubjub_scalar_mul(const uint64_t p[16], const uint64_t scalar[4], uint64_t r[16]) {
    // Precompute table[0..15] = [identity, P, 2P, ..., 15P]
    uint64_t table[16 * 16];  // 16 extended points

    // table[0] = identity
    bjj_pt_set_id(table);
    // table[1] = P
    memcpy(table + 16, p, 128);
    // table[i] = table[i-1] + P for i = 2..15
    for (int i = 2; i < 16; i++) {
        bjj_pt_add(table + (i - 1) * 16, p, table + i * 16);
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
        bjj_pt_set_id(r);
        return;
    }

    // Start with table[top nibble]
    uint64_t result[16];
    memcpy(result, table + nibbles[top] * 16, 128);

    // Process remaining nibbles from MSB to LSB
    for (int i = top - 1; i >= 0; i--) {
        uint64_t tmp[16];
        bjj_pt_dbl(result, tmp); memcpy(result, tmp, 128);
        bjj_pt_dbl(result, tmp); memcpy(result, tmp, 128);
        bjj_pt_dbl(result, tmp); memcpy(result, tmp, 128);
        bjj_pt_dbl(result, tmp); memcpy(result, tmp, 128);
        if (nibbles[i]) {
            bjj_pt_add(result, table + nibbles[i] * 16, tmp);
            memcpy(result, tmp, 128);
        }
    }

    memcpy(r, result, 128);
}

// Exported point addition
void babyjubjub_point_add(const uint64_t p[16], const uint64_t q[16], uint64_t r[16]) {
    bjj_pt_add(p, q, r);
}

// Exported point doubling
void babyjubjub_point_double(const uint64_t p[16], uint64_t r[16]) {
    bjj_pt_dbl(p, r);
}
