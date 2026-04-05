// Grumpkin curve point operations in C with CIOS Montgomery arithmetic.
// Grumpkin: y^2 = x^3 - 17 over BN254 Fr (a=0, b=-17).
// Points in Jacobian projective coordinates: (X, Y, Z) -> affine (X/Z^2, Y/Z^3).
// Each coordinate is 4 x uint64_t in BN254 Fr Montgomery form.

#include "NeonFieldOps.h"
#include <string.h>

typedef unsigned __int128 uint128_t;

// ============================================================
// BN254 Fr field constants (Grumpkin base field)
// ============================================================

static const uint64_t GK_P[4] = {
    0x43e1f593f0000001ULL, 0x2833e84879b97091ULL,
    0xb85045b68181585dULL, 0x30644e72e131a029ULL
};
static const uint64_t GK_INV = 0xc2e1f593efffffffULL;  // -p^{-1} mod 2^64
static const uint64_t GK_ONE[4] = {  // R mod p (Montgomery form of 1)
    0xac96341c4ffffffbULL, 0x36fc76959f60cd29ULL,
    0x666ea36f7879462eULL, 0x0e0a77c19a07df2fULL
};

// ============================================================
// CIOS Montgomery multiplication for BN254 Fr
// ============================================================

static inline void gk_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t t0=0,t1=0,t2=0,t3=0,t4=0;
    #define GK_ITER(I) { \
        uint128_t w; uint64_t c; \
        w=(uint128_t)a[I]*b[0]+t0; t0=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[1]+t1+c; t1=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[2]+t2+c; t2=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[3]+t3+c; t3=(uint64_t)w; c=(uint64_t)(w>>64); \
        t4+=c; \
        uint64_t m=t0*GK_INV; \
        w=(uint128_t)m*GK_P[0]+t0; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*GK_P[1]+t1+c; t0=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*GK_P[2]+t2+c; t1=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*GK_P[3]+t3+c; t2=(uint64_t)w; c=(uint64_t)(w>>64); \
        t3=t4+c; t4=0; \
    }
    GK_ITER(0) GK_ITER(1) GK_ITER(2) GK_ITER(3)
    #undef GK_ITER
    uint64_t borrow=0; uint64_t r0,r1,r2,r3; uint128_t d;
    d=(uint128_t)t0-GK_P[0]-borrow; r0=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t1-GK_P[1]-borrow; r1=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t2-GK_P[2]-borrow; r2=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t3-GK_P[3]-borrow; r3=(uint64_t)d; borrow=(d>>127)&1;
    if(!borrow){r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;}
    else{r[0]=t0;r[1]=t1;r[2]=t2;r[3]=t3;}
}

static inline void gk_sqr(const uint64_t a[4], uint64_t r[4]) {
    gk_mul(a, a, r);
}

static inline void gk_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t w; uint64_t c=0;
    w=(uint128_t)a[0]+b[0]; r[0]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[1]+b[1]+c; r[1]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[2]+b[2]+c; r[2]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[3]+b[3]+c; r[3]=(uint64_t)w; c=(uint64_t)(w>>64);
    uint64_t borrow=0; uint64_t r0,r1,r2,r3; uint128_t d;
    d=(uint128_t)r[0]-GK_P[0]; r0=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[1]-GK_P[1]-borrow; r1=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[2]-GK_P[2]-borrow; r2=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[3]-GK_P[3]-borrow; r3=(uint64_t)d; borrow=(d>>127)&1;
    if(c||!borrow){r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;}
}

static inline void gk_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t d; uint64_t borrow=0;
    d=(uint128_t)a[0]-b[0]; r[0]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[1]-b[1]-borrow; r[1]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[2]-b[2]-borrow; r[2]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[3]-b[3]-borrow; r[3]=(uint64_t)d; borrow=(d>>127)&1;
    if(borrow){
        uint64_t c=0;
        d=(uint128_t)r[0]+GK_P[0]; r[0]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[1]+GK_P[1]+c; r[1]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[2]+GK_P[2]+c; r[2]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[3]+GK_P[3]+c; r[3]=(uint64_t)d;
    }
}

static inline void gk_dbl(const uint64_t a[4], uint64_t r[4]) {
    gk_add(a, a, r);
}

static inline int gk_is_zero(const uint64_t a[4]) {
    return (a[0] | a[1] | a[2] | a[3]) == 0;
}

// ============================================================
// Jacobian projective point operations for Grumpkin (a=0)
// Layout: [x0..x3, y0..y3, z0..z3] = 12 uint64_t
// Identity: Z = 0
// ============================================================

static inline void gk_pt_set_id(uint64_t p[12]) {
    memcpy(p, GK_ONE, 32);       // x = 1
    memcpy(p + 4, GK_ONE, 32);   // y = 1
    memset(p + 8, 0, 32);        // z = 0
}

static inline int gk_pt_is_id(const uint64_t p[12]) {
    return gk_is_zero(p + 8);
}

// Point doubling for a=0 curve (y^2 = x^3 + b)
// Same formula as BN254 G1 doubling since a=0.
static void gk_pt_dbl(const uint64_t p[12], uint64_t r[12]) {
    if (gk_pt_is_id(p)) { memcpy(r, p, 96); return; }

    const uint64_t *px = p, *py = p + 4, *pz = p + 8;
    uint64_t a[4], b[4], c[4], d[4], e[4], f[4], t1[4], t2[4];

    gk_sqr(px, a);             // a = x^2
    gk_sqr(py, b);             // b = y^2
    gk_sqr(b, c);              // c = y^4

    // d = 2((x+b)^2 - a - c)
    gk_add(px, b, t1);
    gk_sqr(t1, t1);
    gk_sub(t1, a, t1);
    gk_sub(t1, c, t1);
    gk_dbl(t1, d);

    // e = 3a = 3*x^2
    gk_dbl(a, t1);
    gk_add(t1, a, e);

    gk_sqr(e, f);              // f = e^2

    // x3 = f - 2d
    gk_dbl(d, t1);
    gk_sub(f, t1, r);

    // y3 = e(d - x3) - 8c
    gk_sub(d, r, t1);
    gk_mul(e, t1, t2);
    gk_dbl(c, t1);
    gk_dbl(t1, t1);
    gk_dbl(t1, t1);            // 8c
    gk_sub(t2, t1, r + 4);

    // z3 = (y+z)^2 - b - z^2
    gk_add(py, pz, t1);
    gk_sqr(t1, t1);
    gk_sub(t1, b, t1);
    gk_sqr(pz, t2);
    gk_sub(t1, t2, r + 8);
}

// Full projective addition
static void gk_pt_add(const uint64_t p[12], const uint64_t q[12], uint64_t r[12]) {
    if (gk_pt_is_id(p)) { memcpy(r, q, 96); return; }
    if (gk_pt_is_id(q)) { memcpy(r, p, 96); return; }

    const uint64_t *px = p, *py = p+4, *pz = p+8;
    const uint64_t *qx = q, *qy = q+4, *qz = q+8;

    uint64_t z1z1[4], z2z2[4], u1[4], u2[4], s1[4], s2[4];
    uint64_t h[4], rr[4], ii[4], j[4], vv[4], t1[4];

    gk_sqr(pz, z1z1);
    gk_sqr(qz, z2z2);
    gk_mul(px, z2z2, u1);
    gk_mul(qx, z1z1, u2);
    gk_mul(qz, z2z2, t1);
    gk_mul(py, t1, s1);
    gk_mul(pz, z1z1, t1);
    gk_mul(qy, t1, s2);

    gk_sub(u2, u1, h);
    gk_sub(s2, s1, t1);
    gk_dbl(t1, rr);

    if (gk_is_zero(h)) {
        if (gk_is_zero(rr)) { gk_pt_dbl(p, r); return; }
        gk_pt_set_id(r); return;
    }

    gk_dbl(h, t1);
    gk_sqr(t1, ii);
    gk_mul(h, ii, j);
    gk_mul(u1, ii, vv);

    // x3 = r^2 - j - 2v
    gk_sqr(rr, r);
    gk_sub(r, j, r);
    gk_dbl(vv, t1);
    gk_sub(r, t1, r);

    // y3 = r(v - x3) - 2*s1*j
    gk_sub(vv, r, t1);
    gk_mul(rr, t1, r + 4);
    gk_mul(s1, j, t1);
    gk_dbl(t1, t1);
    gk_sub(r + 4, t1, r + 4);

    // z3 = ((z1+z2)^2 - z1z1 - z2z2) * h
    gk_add(pz, qz, t1);
    gk_sqr(t1, t1);
    gk_sub(t1, z1z1, t1);
    gk_sub(t1, z2z2, t1);
    gk_mul(t1, h, r + 8);
}

// Scalar multiplication using windowed method (w=4)
// scalar: 4 x uint64_t in non-Montgomery integer form (little-endian)
void grumpkin_scalar_mul(const uint64_t p[12], const uint64_t scalar[4], uint64_t r[12]) {
    // Windowed scalar multiplication, w=4 (process 4 bits at a time)
    uint64_t table[16 * 12];

    // table[0] = identity
    gk_pt_set_id(table);
    // table[1] = P
    memcpy(table + 12, p, 96);
    // table[i] = table[i-1] + P for i = 2..15
    for (int i = 2; i < 16; i++) {
        gk_pt_add(table + (i - 1) * 12, p, table + i * 12);
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
        gk_pt_set_id(r);
        return;
    }

    // Start with table[top nibble]
    uint64_t result[12];
    memcpy(result, table + nibbles[top] * 12, 96);

    // Process remaining nibbles from MSB to LSB
    for (int i = top - 1; i >= 0; i--) {
        uint64_t tmp[12];
        gk_pt_dbl(result, tmp); memcpy(result, tmp, 96);
        gk_pt_dbl(result, tmp); memcpy(result, tmp, 96);
        gk_pt_dbl(result, tmp); memcpy(result, tmp, 96);
        gk_pt_dbl(result, tmp); memcpy(result, tmp, 96);
        if (nibbles[i]) {
            gk_pt_add(result, table + nibbles[i] * 12, tmp);
            memcpy(result, tmp, 96);
        }
    }

    memcpy(r, result, 96);
}

// Exported point addition
void grumpkin_point_add(const uint64_t p[12], const uint64_t q[12], uint64_t r[12]) {
    gk_pt_add(p, q, r);
}

// Exported point doubling
void grumpkin_point_double(const uint64_t p[12], uint64_t r[12]) {
    gk_pt_dbl(p, r);
}
