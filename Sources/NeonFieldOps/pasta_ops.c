// Pasta (Pallas + Vesta) curve operations in C with CIOS Montgomery arithmetic.
// Pallas: y^2 = x^3 + 5 over Pallas Fp
// Vesta:  y^2 = x^3 + 5 over Vesta Fp
// Cycle property: Pallas Fp = Vesta Fr, Vesta Fp = Pallas Fr.
// Points in Jacobian projective coordinates: (X, Y, Z) -> affine (X/Z^2, Y/Z^3).
// Each coordinate is 4 x uint64_t in Montgomery form.

#include "NeonFieldOps.h"
#include <string.h>
#include <stdlib.h>
#include <dispatch/dispatch.h>

typedef unsigned __int128 uint128_t;

// ============================================================
// Pallas Fp constants
// p = 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001
// ============================================================

static const uint64_t PA_P[4] = {
    0x992d30ed00000001ULL, 0x224698fc094cf91bULL,
    0x0000000000000000ULL, 0x4000000000000000ULL
};
static const uint64_t PA_INV = 0x992d30ecffffffffULL;  // -p^{-1} mod 2^64
static const uint64_t PA_ONE[4] = {  // R mod p (Montgomery form of 1)
    0x34786d38fffffffdULL, 0x992c350be41914adULL,
    0xffffffffffffffffULL, 0x3fffffffffffffffULL
};
static const uint64_t PA_R2[4] = {  // R^2 mod p
    0x8c78ecb30000000fULL, 0xd7d30dbd8b0de0e7ULL,
    0x7797a99bc3c95d18ULL, 0x096d41af7b9cb714ULL
};

// ============================================================
// Vesta Fp constants
// p = 0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001
// ============================================================

static const uint64_t VE_P[4] = {
    0x8c46eb2100000001ULL, 0x224698fc0994a8ddULL,
    0x0000000000000000ULL, 0x4000000000000000ULL
};
static const uint64_t VE_INV = 0x8c46eb20ffffffffULL;  // -p^{-1} mod 2^64
static const uint64_t VE_ONE[4] = {  // R mod p (Montgomery form of 1)
    0x5b2b3e9cfffffffdULL, 0x992c350be3420567ULL,
    0xffffffffffffffffULL, 0x3fffffffffffffffULL
};
static const uint64_t VE_R2[4] = {  // R^2 mod p
    0xfc9678ff0000000fULL, 0x67bb433d891a16e3ULL,
    0x7fae231004ccf590ULL, 0x096d41af7ccfdaa9ULL
};

// ============================================================
// CIOS Montgomery multiplication for Pallas Fp
// ============================================================

static inline void pa_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t t0=0,t1=0,t2=0,t3=0,t4=0;
    #define PA_ITER(I) { \
        uint128_t w; uint64_t c; \
        w=(uint128_t)a[I]*b[0]+t0; t0=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[1]+t1+c; t1=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[2]+t2+c; t2=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[3]+t3+c; t3=(uint64_t)w; c=(uint64_t)(w>>64); \
        t4+=c; \
        uint64_t m=t0*PA_INV; \
        w=(uint128_t)m*PA_P[0]+t0; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*PA_P[1]+t1+c; t0=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*PA_P[2]+t2+c; t1=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*PA_P[3]+t3+c; t2=(uint64_t)w; c=(uint64_t)(w>>64); \
        t3=t4+c; t4=0; \
    }
    PA_ITER(0) PA_ITER(1) PA_ITER(2) PA_ITER(3)
    #undef PA_ITER
    uint64_t borrow=0; uint64_t r0,r1,r2,r3; uint128_t d;
    d=(uint128_t)t0-PA_P[0]-borrow; r0=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t1-PA_P[1]-borrow; r1=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t2-PA_P[2]-borrow; r2=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t3-PA_P[3]-borrow; r3=(uint64_t)d; borrow=(d>>127)&1;
    if(!borrow){r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;}
    else{r[0]=t0;r[1]=t1;r[2]=t2;r[3]=t3;}
}

static inline void pa_sqr(const uint64_t a[4], uint64_t r[4]) {
    pa_mul(a, a, r);
}

static inline void pa_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t w; uint64_t c=0;
    w=(uint128_t)a[0]+b[0]; r[0]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[1]+b[1]+c; r[1]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[2]+b[2]+c; r[2]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[3]+b[3]+c; r[3]=(uint64_t)w; c=(uint64_t)(w>>64);
    uint64_t borrow=0; uint64_t r0,r1,r2,r3; uint128_t d;
    d=(uint128_t)r[0]-PA_P[0]; r0=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[1]-PA_P[1]-borrow; r1=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[2]-PA_P[2]-borrow; r2=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[3]-PA_P[3]-borrow; r3=(uint64_t)d; borrow=(d>>127)&1;
    if(c||!borrow){r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;}
}

static inline void pa_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t d; uint64_t borrow=0;
    d=(uint128_t)a[0]-b[0]; r[0]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[1]-b[1]-borrow; r[1]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[2]-b[2]-borrow; r[2]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[3]-b[3]-borrow; r[3]=(uint64_t)d; borrow=(d>>127)&1;
    if(borrow){
        uint64_t c=0;
        d=(uint128_t)r[0]+PA_P[0]; r[0]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[1]+PA_P[1]+c; r[1]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[2]+PA_P[2]+c; r[2]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[3]+PA_P[3]+c; r[3]=(uint64_t)d;
    }
}

static inline void pa_dbl(const uint64_t a[4], uint64_t r[4]) {
    pa_add(a, a, r);
}

static inline int pa_is_zero(const uint64_t a[4]) {
    return (a[0] | a[1] | a[2] | a[3]) == 0;
}

static inline void pa_neg(const uint64_t a[4], uint64_t r[4]) {
    if (pa_is_zero(a)) { memset(r, 0, 32); return; }
    uint128_t d; uint64_t borrow=0;
    d=(uint128_t)PA_P[0]-a[0]; r[0]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)PA_P[1]-a[1]-borrow; r[1]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)PA_P[2]-a[2]-borrow; r[2]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)PA_P[3]-a[3]-borrow; r[3]=(uint64_t)d;
}

// ============================================================
// CIOS Montgomery multiplication for Vesta Fp
// ============================================================

static inline void ve_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t t0=0,t1=0,t2=0,t3=0,t4=0;
    #define VE_ITER(I) { \
        uint128_t w; uint64_t c; \
        w=(uint128_t)a[I]*b[0]+t0; t0=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[1]+t1+c; t1=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[2]+t2+c; t2=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[3]+t3+c; t3=(uint64_t)w; c=(uint64_t)(w>>64); \
        t4+=c; \
        uint64_t m=t0*VE_INV; \
        w=(uint128_t)m*VE_P[0]+t0; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*VE_P[1]+t1+c; t0=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*VE_P[2]+t2+c; t1=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*VE_P[3]+t3+c; t2=(uint64_t)w; c=(uint64_t)(w>>64); \
        t3=t4+c; t4=0; \
    }
    VE_ITER(0) VE_ITER(1) VE_ITER(2) VE_ITER(3)
    #undef VE_ITER
    uint64_t borrow=0; uint64_t r0,r1,r2,r3; uint128_t d;
    d=(uint128_t)t0-VE_P[0]-borrow; r0=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t1-VE_P[1]-borrow; r1=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t2-VE_P[2]-borrow; r2=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t3-VE_P[3]-borrow; r3=(uint64_t)d; borrow=(d>>127)&1;
    if(!borrow){r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;}
    else{r[0]=t0;r[1]=t1;r[2]=t2;r[3]=t3;}
}

static inline void ve_sqr(const uint64_t a[4], uint64_t r[4]) {
    ve_mul(a, a, r);
}

static inline void ve_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t w; uint64_t c=0;
    w=(uint128_t)a[0]+b[0]; r[0]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[1]+b[1]+c; r[1]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[2]+b[2]+c; r[2]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[3]+b[3]+c; r[3]=(uint64_t)w; c=(uint64_t)(w>>64);
    uint64_t borrow=0; uint64_t r0,r1,r2,r3; uint128_t d;
    d=(uint128_t)r[0]-VE_P[0]; r0=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[1]-VE_P[1]-borrow; r1=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[2]-VE_P[2]-borrow; r2=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[3]-VE_P[3]-borrow; r3=(uint64_t)d; borrow=(d>>127)&1;
    if(c||!borrow){r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;}
}

static inline void ve_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t d; uint64_t borrow=0;
    d=(uint128_t)a[0]-b[0]; r[0]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[1]-b[1]-borrow; r[1]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[2]-b[2]-borrow; r[2]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[3]-b[3]-borrow; r[3]=(uint64_t)d; borrow=(d>>127)&1;
    if(borrow){
        uint64_t c=0;
        d=(uint128_t)r[0]+VE_P[0]; r[0]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[1]+VE_P[1]+c; r[1]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[2]+VE_P[2]+c; r[2]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[3]+VE_P[3]+c; r[3]=(uint64_t)d;
    }
}

static inline void ve_dbl(const uint64_t a[4], uint64_t r[4]) {
    ve_add(a, a, r);
}

static inline int ve_is_zero(const uint64_t a[4]) {
    return (a[0] | a[1] | a[2] | a[3]) == 0;
}

static inline void ve_neg(const uint64_t a[4], uint64_t r[4]) {
    if (ve_is_zero(a)) { memset(r, 0, 32); return; }
    uint128_t d; uint64_t borrow=0;
    d=(uint128_t)VE_P[0]-a[0]; r[0]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)VE_P[1]-a[1]-borrow; r[1]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)VE_P[2]-a[2]-borrow; r[2]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)VE_P[3]-a[3]-borrow; r[3]=(uint64_t)d;
}

// ============================================================
// Exported field operations
// ============================================================

void pallas_fp_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) { pa_mul(a, b, r); }
void pallas_fp_sqr(const uint64_t a[4], uint64_t r[4]) { pa_sqr(a, r); }
void pallas_fp_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) { pa_add(a, b, r); }
void pallas_fp_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) { pa_sub(a, b, r); }
void pallas_fp_neg(const uint64_t a[4], uint64_t r[4]) { pa_neg(a, r); }

void vesta_fp_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) { ve_mul(a, b, r); }
void vesta_fp_sqr(const uint64_t a[4], uint64_t r[4]) { ve_sqr(a, r); }
void vesta_fp_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) { ve_add(a, b, r); }
void vesta_fp_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) { ve_sub(a, b, r); }
void vesta_fp_neg(const uint64_t a[4], uint64_t r[4]) { ve_neg(a, r); }

// ============================================================
// Pallas Jacobian projective point operations (a=0, b=5)
// Layout: [x0..x3, y0..y3, z0..z3] = 12 uint64_t
// Identity: Z = 0
// ============================================================

static inline void pa_pt_set_id(uint64_t p[12]) {
    memcpy(p, PA_ONE, 32);       // x = 1
    memcpy(p + 4, PA_ONE, 32);   // y = 1
    memset(p + 8, 0, 32);        // z = 0
}

static inline int pa_pt_is_id(const uint64_t p[12]) {
    return pa_is_zero(p + 8);
}

// Point doubling for a=0 curve (y^2 = x^3 + 5)
static void pa_pt_dbl(const uint64_t p[12], uint64_t r[12]) {
    if (pa_pt_is_id(p)) { memcpy(r, p, 96); return; }

    const uint64_t *px = p, *py = p + 4, *pz = p + 8;
    uint64_t a[4], b[4], c[4], d[4], e[4], f[4], t1[4], t2[4];

    pa_sqr(px, a);             // a = x^2
    pa_sqr(py, b);             // b = y^2
    pa_sqr(b, c);              // c = y^4

    // d = 2((x+b)^2 - a - c)
    pa_add(px, b, t1);
    pa_sqr(t1, t1);
    pa_sub(t1, a, t1);
    pa_sub(t1, c, t1);
    pa_dbl(t1, d);

    // e = 3a = 3*x^2
    pa_dbl(a, t1);
    pa_add(t1, a, e);

    pa_sqr(e, f);              // f = e^2

    // x3 = f - 2d
    pa_dbl(d, t1);
    pa_sub(f, t1, r);

    // y3 = e(d - x3) - 8c
    pa_sub(d, r, t1);
    pa_mul(e, t1, t2);
    pa_dbl(c, t1);
    pa_dbl(t1, t1);
    pa_dbl(t1, t1);            // 8c
    pa_sub(t2, t1, r + 4);

    // z3 = (y+z)^2 - b - z^2
    pa_add(py, pz, t1);
    pa_sqr(t1, t1);
    pa_sub(t1, b, t1);
    pa_sqr(pz, t2);
    pa_sub(t1, t2, r + 8);
}

// Full projective addition
static void pa_pt_add(const uint64_t p[12], const uint64_t q[12], uint64_t r[12]) {
    if (pa_pt_is_id(p)) { memcpy(r, q, 96); return; }
    if (pa_pt_is_id(q)) { memcpy(r, p, 96); return; }

    const uint64_t *px = p, *py = p+4, *pz = p+8;
    const uint64_t *qx = q, *qy = q+4, *qz = q+8;

    uint64_t z1z1[4], z2z2[4], u1[4], u2[4], s1[4], s2[4];
    uint64_t h[4], rr[4], ii[4], j[4], vv[4], t1[4];

    pa_sqr(pz, z1z1);
    pa_sqr(qz, z2z2);
    pa_mul(px, z2z2, u1);
    pa_mul(qx, z1z1, u2);
    pa_mul(qz, z2z2, t1);
    pa_mul(py, t1, s1);
    pa_mul(pz, z1z1, t1);
    pa_mul(qy, t1, s2);

    pa_sub(u2, u1, h);
    pa_sub(s2, s1, t1);
    pa_dbl(t1, rr);

    if (pa_is_zero(h)) {
        if (pa_is_zero(rr)) { pa_pt_dbl(p, r); return; }
        pa_pt_set_id(r); return;
    }

    pa_dbl(h, t1);
    pa_sqr(t1, ii);
    pa_mul(h, ii, j);
    pa_mul(u1, ii, vv);

    // x3 = r^2 - j - 2v
    pa_sqr(rr, r);
    pa_sub(r, j, r);
    pa_dbl(vv, t1);
    pa_sub(r, t1, r);

    // y3 = r(v - x3) - 2*s1*j
    pa_sub(vv, r, t1);
    pa_mul(rr, t1, r + 4);
    pa_mul(s1, j, t1);
    pa_dbl(t1, t1);
    pa_sub(r + 4, t1, r + 4);

    // z3 = ((z1+z2)^2 - z1z1 - z2z2) * h
    pa_add(pz, qz, t1);
    pa_sqr(t1, t1);
    pa_sub(t1, z1z1, t1);
    pa_sub(t1, z2z2, t1);
    pa_mul(t1, h, r + 8);
}

// Mixed addition: projective P + affine Q (Z_Q = 1)
static void pa_pt_add_mixed(const uint64_t p[12], const uint64_t q_aff[8], uint64_t r[12]) {
    if (pa_pt_is_id(p)) {
        memcpy(r, q_aff, 64);
        memcpy(r + 8, PA_ONE, 32);
        return;
    }

    const uint64_t *px = p, *py = p+4, *pz = p+8;
    const uint64_t *qx = q_aff, *qy = q_aff+4;

    uint64_t z1z1[4], u2[4], s2[4], h[4], hh[4], rr[4], t1[4];

    pa_sqr(pz, z1z1);
    pa_mul(qx, z1z1, u2);       // u2 = qx * z1^2
    pa_mul(pz, z1z1, t1);
    pa_mul(qy, t1, s2);         // s2 = qy * z1^3

    pa_sub(u2, px, h);          // h = u2 - px (u1 = px since z_q=1)
    pa_sub(s2, py, t1);         // t1 = s2 - py (s1 = py since z_q=1)
    pa_dbl(t1, rr);

    if (pa_is_zero(h)) {
        if (pa_is_zero(rr)) { pa_pt_dbl(p, r); return; }
        pa_pt_set_id(r); return;
    }

    pa_sqr(h, hh);
    uint64_t j[4], vv[4];
    pa_mul(h, hh, j);
    pa_mul(px, hh, vv);

    // x3 = r^2 - j - 2v
    pa_sqr(rr, r);
    pa_sub(r, j, r);
    pa_dbl(vv, t1);
    pa_sub(r, t1, r);

    // y3 = r(v - x3) - 2*s1*j
    pa_sub(vv, r, t1);
    pa_mul(rr, t1, r + 4);
    pa_mul(py, j, t1);
    pa_dbl(t1, t1);
    pa_sub(r + 4, t1, r + 4);

    // z3 = (z1 + h)^2 - z1z1 - hh
    pa_add(pz, h, t1);
    pa_sqr(t1, t1);
    pa_sub(t1, z1z1, t1);
    pa_sub(t1, hh, r + 8);
}

// ============================================================
// Vesta Jacobian projective point operations (a=0, b=5)
// ============================================================

static inline void ve_pt_set_id(uint64_t p[12]) {
    memcpy(p, VE_ONE, 32);
    memcpy(p + 4, VE_ONE, 32);
    memset(p + 8, 0, 32);
}

static inline int ve_pt_is_id(const uint64_t p[12]) {
    return ve_is_zero(p + 8);
}

static void ve_pt_dbl(const uint64_t p[12], uint64_t r[12]) {
    if (ve_pt_is_id(p)) { memcpy(r, p, 96); return; }

    const uint64_t *px = p, *py = p + 4, *pz = p + 8;
    uint64_t a[4], b[4], c[4], d[4], e[4], f[4], t1[4], t2[4];

    ve_sqr(px, a);
    ve_sqr(py, b);
    ve_sqr(b, c);

    ve_add(px, b, t1);
    ve_sqr(t1, t1);
    ve_sub(t1, a, t1);
    ve_sub(t1, c, t1);
    ve_dbl(t1, d);

    ve_dbl(a, t1);
    ve_add(t1, a, e);

    ve_sqr(e, f);

    ve_dbl(d, t1);
    ve_sub(f, t1, r);

    ve_sub(d, r, t1);
    ve_mul(e, t1, t2);
    ve_dbl(c, t1);
    ve_dbl(t1, t1);
    ve_dbl(t1, t1);
    ve_sub(t2, t1, r + 4);

    ve_add(py, pz, t1);
    ve_sqr(t1, t1);
    ve_sub(t1, b, t1);
    ve_sqr(pz, t2);
    ve_sub(t1, t2, r + 8);
}

static void ve_pt_add(const uint64_t p[12], const uint64_t q[12], uint64_t r[12]) {
    if (ve_pt_is_id(p)) { memcpy(r, q, 96); return; }
    if (ve_pt_is_id(q)) { memcpy(r, p, 96); return; }

    const uint64_t *px = p, *py = p+4, *pz = p+8;
    const uint64_t *qx = q, *qy = q+4, *qz = q+8;

    uint64_t z1z1[4], z2z2[4], u1[4], u2[4], s1[4], s2[4];
    uint64_t h[4], rr[4], ii[4], j[4], vv[4], t1[4];

    ve_sqr(pz, z1z1);
    ve_sqr(qz, z2z2);
    ve_mul(px, z2z2, u1);
    ve_mul(qx, z1z1, u2);
    ve_mul(qz, z2z2, t1);
    ve_mul(py, t1, s1);
    ve_mul(pz, z1z1, t1);
    ve_mul(qy, t1, s2);

    ve_sub(u2, u1, h);
    ve_sub(s2, s1, t1);
    ve_dbl(t1, rr);

    if (ve_is_zero(h)) {
        if (ve_is_zero(rr)) { ve_pt_dbl(p, r); return; }
        ve_pt_set_id(r); return;
    }

    ve_dbl(h, t1);
    ve_sqr(t1, ii);
    ve_mul(h, ii, j);
    ve_mul(u1, ii, vv);

    ve_sqr(rr, r);
    ve_sub(r, j, r);
    ve_dbl(vv, t1);
    ve_sub(r, t1, r);

    ve_sub(vv, r, t1);
    ve_mul(rr, t1, r + 4);
    ve_mul(s1, j, t1);
    ve_dbl(t1, t1);
    ve_sub(r + 4, t1, r + 4);

    ve_add(pz, qz, t1);
    ve_sqr(t1, t1);
    ve_sub(t1, z1z1, t1);
    ve_sub(t1, z2z2, t1);
    ve_mul(t1, h, r + 8);
}

static void ve_pt_add_mixed(const uint64_t p[12], const uint64_t q_aff[8], uint64_t r[12]) {
    if (ve_pt_is_id(p)) {
        memcpy(r, q_aff, 64);
        memcpy(r + 8, VE_ONE, 32);
        return;
    }

    const uint64_t *px = p, *py = p+4, *pz = p+8;
    const uint64_t *qx = q_aff, *qy = q_aff+4;

    uint64_t z1z1[4], u2[4], s2[4], h[4], hh[4], rr[4], t1[4];

    ve_sqr(pz, z1z1);
    ve_mul(qx, z1z1, u2);
    ve_mul(pz, z1z1, t1);
    ve_mul(qy, t1, s2);

    ve_sub(u2, px, h);
    ve_sub(s2, py, t1);
    ve_dbl(t1, rr);

    if (ve_is_zero(h)) {
        if (ve_is_zero(rr)) { ve_pt_dbl(p, r); return; }
        ve_pt_set_id(r); return;
    }

    ve_sqr(h, hh);
    uint64_t j[4], vv[4];
    ve_mul(h, hh, j);
    ve_mul(px, hh, vv);

    ve_sqr(rr, r);
    ve_sub(r, j, r);
    ve_dbl(vv, t1);
    ve_sub(r, t1, r);

    ve_sub(vv, r, t1);
    ve_mul(rr, t1, r + 4);
    ve_mul(py, j, t1);
    ve_dbl(t1, t1);
    ve_sub(r + 4, t1, r + 4);

    ve_add(pz, h, t1);
    ve_sqr(t1, t1);
    ve_sub(t1, z1z1, t1);
    ve_sub(t1, hh, r + 8);
}

// ============================================================
// Pallas scalar multiplication using windowed method (w=4)
// scalar: 4 x uint64_t in non-Montgomery integer form
// ============================================================

void pallas_scalar_mul(const uint64_t p[12], const uint64_t scalar[4], uint64_t r[12]) {
    uint64_t table[16 * 12];

    pa_pt_set_id(table);
    memcpy(table + 12, p, 96);
    for (int i = 2; i < 16; i++) {
        pa_pt_add(table + (i - 1) * 12, p, table + i * 12);
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
        pa_pt_set_id(r);
        return;
    }

    uint64_t result[12];
    memcpy(result, table + nibbles[top] * 12, 96);

    for (int i = top - 1; i >= 0; i--) {
        uint64_t tmp[12];
        pa_pt_dbl(result, tmp); memcpy(result, tmp, 96);
        pa_pt_dbl(result, tmp); memcpy(result, tmp, 96);
        pa_pt_dbl(result, tmp); memcpy(result, tmp, 96);
        pa_pt_dbl(result, tmp); memcpy(result, tmp, 96);
        if (nibbles[i]) {
            pa_pt_add(result, table + nibbles[i] * 12, tmp);
            memcpy(result, tmp, 96);
        }
    }

    memcpy(r, result, 96);
}

// ============================================================
// Vesta scalar multiplication using windowed method (w=4)
// ============================================================

void vesta_scalar_mul(const uint64_t p[12], const uint64_t scalar[4], uint64_t r[12]) {
    uint64_t table[16 * 12];

    ve_pt_set_id(table);
    memcpy(table + 12, p, 96);
    for (int i = 2; i < 16; i++) {
        ve_pt_add(table + (i - 1) * 12, p, table + i * 12);
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
        ve_pt_set_id(r);
        return;
    }

    uint64_t result[12];
    memcpy(result, table + nibbles[top] * 12, 96);

    for (int i = top - 1; i >= 0; i--) {
        uint64_t tmp[12];
        ve_pt_dbl(result, tmp); memcpy(result, tmp, 96);
        ve_pt_dbl(result, tmp); memcpy(result, tmp, 96);
        ve_pt_dbl(result, tmp); memcpy(result, tmp, 96);
        ve_pt_dbl(result, tmp); memcpy(result, tmp, 96);
        if (nibbles[i]) {
            ve_pt_add(result, table + nibbles[i] * 12, tmp);
            memcpy(result, tmp, 96);
        }
    }

    memcpy(r, result, 96);
}

// ============================================================
// Exported point operations
// ============================================================

void pallas_point_add(const uint64_t p[12], const uint64_t q[12], uint64_t r[12]) {
    pa_pt_add(p, q, r);
}

void pallas_point_double(const uint64_t p[12], uint64_t r[12]) {
    pa_pt_dbl(p, r);
}

void pallas_point_add_mixed(const uint64_t p[12], const uint64_t q_aff[8], uint64_t r[12]) {
    pa_pt_add_mixed(p, q_aff, r);
}

void vesta_point_add(const uint64_t p[12], const uint64_t q[12], uint64_t r[12]) {
    ve_pt_add(p, q, r);
}

void vesta_point_double(const uint64_t p[12], uint64_t r[12]) {
    ve_pt_dbl(p, r);
}

void vesta_point_add_mixed(const uint64_t p[12], const uint64_t q_aff[8], uint64_t r[12]) {
    ve_pt_add_mixed(p, q_aff, r);
}

// ============================================================
// Field inversion via Fermat: a^(p-2) mod p
// ============================================================

static void pa_inv(const uint64_t a[4], uint64_t result[4]) {
    uint64_t pm2[4];
    pm2[0] = PA_P[0] - 2; pm2[1] = PA_P[1]; pm2[2] = PA_P[2]; pm2[3] = PA_P[3];
    memcpy(result, PA_ONE, 32);
    uint64_t b[4]; memcpy(b, a, 32);
    for (int i = 0; i < 4; i++) {
        for (int bit = 0; bit < 64; bit++) {
            if ((pm2[i] >> bit) & 1) pa_mul(result, b, result);
            pa_sqr(b, b);
        }
    }
}

static void ve_inv(const uint64_t a[4], uint64_t result[4]) {
    uint64_t pm2[4];
    pm2[0] = VE_P[0] - 2; pm2[1] = VE_P[1]; pm2[2] = VE_P[2]; pm2[3] = VE_P[3];
    memcpy(result, VE_ONE, 32);
    uint64_t b[4]; memcpy(b, a, 32);
    for (int i = 0; i < 4; i++) {
        for (int bit = 0; bit < 64; bit++) {
            if ((pm2[i] >> bit) & 1) ve_mul(result, b, result);
            ve_sqr(b, b);
        }
    }
}

// ============================================================
// Pasta Pippenger MSM — macro-generated for both Pallas and Vesta
// Follows the exact pattern of bn254_pippenger_msm in bn254_msm.c:
//   1. Adaptive window sizing
//   2. Bucket accumulation with mixed affine addition
//   3. Batch-to-affine via Montgomery's trick (single field inversion per window)
//   4. Running-sum reduction with mixed affine addition
//   5. Multi-threaded windows via pthreads
//   6. Horner combination
// ============================================================

#define PASTA_MSM_IMPL(PREFIX, FP_MUL, FP_SQR, FP_ADD, FP_SUB, FP_INV_FN, \
                       FP_IS_ZERO, FP_ONE_CONST, \
                       PT_SET_ID, PT_IS_ID, PT_DBL, PT_ADD, PT_ADD_MIXED) \
\
static void PREFIX##_batch_to_affine(const uint64_t *proj, uint64_t *aff, int n) { \
    if (n == 0) return; \
    uint64_t *prods = (uint64_t *)malloc((size_t)n * 32); \
    int first_valid = -1; \
    for (int i = 0; i < n; i++) { \
        if (PT_IS_ID(proj + i * 12)) { \
            if (i == 0) memcpy(prods, FP_ONE_CONST, 32); \
            else memcpy(prods + i * 4, prods + (i-1) * 4, 32); \
        } else { \
            if (first_valid < 0) { \
                first_valid = i; \
                memcpy(prods + i * 4, proj + i * 12 + 8, 32); \
            } else { \
                FP_MUL(prods + (i-1) * 4, proj + i * 12 + 8, prods + i * 4); \
            } \
        } \
    } \
    if (first_valid < 0) { \
        memset(aff, 0, (size_t)n * 64); \
        free(prods); return; \
    } \
    uint64_t inv[4]; \
    FP_INV_FN(prods + (n-1) * 4, inv); \
    for (int i = n - 1; i >= 0; i--) { \
        if (PT_IS_ID(proj + i * 12)) { \
            memset(aff + i * 8, 0, 64); continue; \
        } \
        uint64_t zinv[4]; \
        if (i > first_valid) { \
            FP_MUL(inv, prods + (i-1) * 4, zinv); \
            FP_MUL(inv, proj + i * 12 + 8, inv); \
        } else { \
            memcpy(zinv, inv, 32); \
        } \
        uint64_t zinv2[4], zinv3[4]; \
        FP_SQR(zinv, zinv2); \
        FP_MUL(zinv2, zinv, zinv3); \
        FP_MUL(proj + i * 12, zinv2, aff + i * 8); \
        FP_MUL(proj + i * 12 + 4, zinv3, aff + i * 8 + 4); \
    } \
    free(prods); \
} \
\
static inline uint32_t PREFIX##_extract_window(const uint32_t *scalar, int window_idx, int window_bits) { \
    int bit_offset = window_idx * window_bits; \
    int word_idx = bit_offset / 32; \
    int bit_in_word = bit_offset % 32; \
    uint64_t word = scalar[word_idx]; \
    if (word_idx + 1 < 8) word |= ((uint64_t)scalar[word_idx + 1]) << 32; \
    return (uint32_t)((word >> bit_in_word) & ((1u << window_bits) - 1)); \
} \
\
typedef struct { \
    const uint64_t *points; \
    const uint32_t *scalars; \
    int n; \
    int window_bits; \
    int window_idx; \
    int num_buckets; \
    uint64_t result[12]; \
} PREFIX##_WindowTask; \
\
static void *PREFIX##_window_worker(void *arg) { \
    PREFIX##_WindowTask *task = (PREFIX##_WindowTask *)arg; \
    int wb = task->window_bits; \
    int w = task->window_idx; \
    int nb = task->num_buckets; \
    int nn = task->n; \
    uint64_t *buckets = (uint64_t *)malloc((size_t)(nb + 1) * 96); \
    for (int b = 0; b <= nb; b++) PT_SET_ID(buckets + b * 12); \
    for (int i = 0; i < nn; i++) { \
        uint32_t digit = PREFIX##_extract_window(task->scalars + i * 8, w, wb); \
        if (digit != 0) { \
            uint64_t tmp[12]; \
            PT_ADD_MIXED(buckets + digit * 12, task->points + i * 8, tmp); \
            memcpy(buckets + digit * 12, tmp, 96); \
        } \
    } \
    uint64_t *bucket_aff = (uint64_t *)malloc((size_t)nb * 64); \
    PREFIX##_batch_to_affine(buckets + 12, bucket_aff, nb); \
    uint64_t running[12], window_sum[12]; \
    PT_SET_ID(running); \
    PT_SET_ID(window_sum); \
    for (int j = nb - 1; j >= 0; j--) { \
        if (!(bucket_aff[j*8]==0 && bucket_aff[j*8+1]==0 && \
              bucket_aff[j*8+2]==0 && bucket_aff[j*8+3]==0 && \
              bucket_aff[j*8+4]==0 && bucket_aff[j*8+5]==0 && \
              bucket_aff[j*8+6]==0 && bucket_aff[j*8+7]==0)) { \
            uint64_t tmp[12]; \
            PT_ADD_MIXED(running, bucket_aff + j * 8, tmp); \
            memcpy(running, tmp, 96); \
        } \
        uint64_t tmp[12]; \
        PT_ADD(window_sum, running, tmp); \
        memcpy(window_sum, tmp, 96); \
    } \
    memcpy(task->result, window_sum, 96); \
    free(buckets); \
    free(bucket_aff); \
    return NULL; \
}

static int pasta_optimal_window_bits(int n) {
    if (n <= 4)     return 3;
    if (n <= 32)    return 5;
    if (n <= 256)   return 8;
    if (n <= 2048)  return 10;
    if (n <= 8192)  return 11;
    if (n <= 32768) return 13;
    if (n <= 131072) return 14;
    if (n <= 524288) return 15;
    return 16;
}

// Instantiate Pallas MSM helpers
PASTA_MSM_IMPL(pa, pa_mul, pa_sqr, pa_add, pa_sub, pa_inv,
               pa_is_zero, PA_ONE,
               pa_pt_set_id, pa_pt_is_id, pa_pt_dbl, pa_pt_add, pa_pt_add_mixed)

// Instantiate Vesta MSM helpers
PASTA_MSM_IMPL(ve, ve_mul, ve_sqr, ve_add, ve_sub, ve_inv,
               ve_is_zero, VE_ONE,
               ve_pt_set_id, ve_pt_is_id, ve_pt_dbl, ve_pt_add, ve_pt_add_mixed)

#define PASTA_PIPPENGER(FUNC_NAME, PREFIX, PT_SET_ID, PT_DBL, PT_ADD) \
void FUNC_NAME( \
    const uint64_t *points, const uint32_t *scalars, \
    int n, uint64_t *result) \
{ \
    if (n == 0) { PT_SET_ID(result); return; } \
    int wb = pasta_optimal_window_bits(n); \
    int num_windows = (256 + wb - 1) / wb; \
    int num_buckets = (1 << wb) - 1; \
    PREFIX##_WindowTask *tasks = (PREFIX##_WindowTask *)malloc( \
        (size_t)num_windows * sizeof(PREFIX##_WindowTask)); \
    for (int w = 0; w < num_windows; w++) { \
        tasks[w].points = points; \
        tasks[w].scalars = scalars; \
        tasks[w].n = n; \
        tasks[w].window_bits = wb; \
        tasks[w].window_idx = w; \
        tasks[w].num_buckets = num_buckets; \
    } \
    dispatch_apply(num_windows, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), \
        ^(size_t w) { \
            PREFIX##_window_worker(&tasks[w]); \
        }); \
    memcpy(result, tasks[num_windows - 1].result, 96); \
    for (int w = num_windows - 2; w >= 0; w--) { \
        uint64_t tmp[12]; \
        for (int s = 0; s < wb; s++) { \
            PT_DBL(result, tmp); memcpy(result, tmp, 96); \
        } \
        PT_ADD(result, tasks[w].result, tmp); \
        memcpy(result, tmp, 96); \
    } \
    free(tasks); \
}

PASTA_PIPPENGER(pallas_pippenger_msm, pa, pa_pt_set_id, pa_pt_dbl, pa_pt_add)
PASTA_PIPPENGER(vesta_pippenger_msm, ve, ve_pt_set_id, ve_pt_dbl, ve_pt_add)

// Batch inverse via Montgomery's trick
void pallas_fp_batch_inverse(const uint64_t *in, uint64_t *out, int n) {
    if (n <= 0) return;
    memcpy(&out[0], &in[0], 32);
    for (int i = 1; i < n; i++)
        pa_mul(&out[(i-1)*4], &in[i*4], &out[i*4]);
    uint64_t inv_acc[4];
    pa_inv(&out[(n-1)*4], inv_acc);
    for (int i = n - 1; i > 0; i--) {
        uint64_t tmp[4];
        pa_mul(inv_acc, &out[(i-1)*4], tmp);
        pa_mul(inv_acc, &in[i*4], inv_acc);
        memcpy(&out[i*4], tmp, 32);
    }
    memcpy(&out[0], inv_acc, 32);
}

void vesta_fp_batch_inverse(const uint64_t *in, uint64_t *out, int n) {
    if (n <= 0) return;
    memcpy(&out[0], &in[0], 32);
    for (int i = 1; i < n; i++)
        ve_mul(&out[(i-1)*4], &in[i*4], &out[i*4]);
    uint64_t inv_acc[4];
    ve_inv(&out[(n-1)*4], inv_acc);
    for (int i = n - 1; i > 0; i--) {
        uint64_t tmp[4];
        ve_mul(inv_acc, &out[(i-1)*4], tmp);
        ve_mul(inv_acc, &in[i*4], inv_acc);
        memcpy(&out[i*4], tmp, 32);
    }
    memcpy(&out[0], inv_acc, 32);
}

// ============================================================
// Pasta endo-combine kernel (affine batch g1 + g2.scale(scalar))
// Uses signed-digit window with 2-bit digits, 64 iterations.
// endo_coeff multiplies g2.x before point operations.
// Result[i] = g1[i] + g2[i].scale(scalar) in affine coordinates.
// ============================================================

// Helper: projective addition with mixed addition (proj P + affine Q).
// Returns 1 if result is identity, 0 otherwise.
static int pa_proj_add_aff(const uint64_t projP[12], const uint64_t affQ[8], uint64_t result[12]) {
    if (pa_pt_is_id(projP)) {
        memcpy(result, affQ, 64);
        memcpy(result + 8, PA_ONE, 32);
        return 0;
    }
    if (ve_is_zero(affQ)) {
        memcpy(result, projP, 96);
        return 0;
    }
    // Mixed addition: projective P + affine Q (Z_Q = 1)
    const uint64_t *px = projP, *py = projP+4, *pz = projP+8;
    const uint64_t *qx = affQ, *qy = affQ+4;

    uint64_t z1z1[4], u2[4], s2[4], h[4], hh[4], rr[4], t1[4];
    uint64_t j[4], vv[4];

    pa_sqr(pz, z1z1);
    pa_mul(qx, z1z1, u2);
    pa_mul(pz, z1z1, t1);
    pa_mul(qy, t1, s2);

    pa_sub(u2, px, h);
    pa_sub(s2, py, t1);
    pa_dbl(t1, rr);

    if (pa_is_zero(h)) {
        if (pa_is_zero(rr)) {
            pa_pt_dbl(projP, result);
            return 0;
        }
        return 1; // identity
    }

    pa_sqr(h, hh);
    pa_mul(h, hh, j);
    pa_mul(px, hh, vv);

    pa_sqr(rr, result);
    pa_sub(result, j, result);
    pa_dbl(vv, t1);
    pa_sub(result, t1, result);

    pa_sub(vv, result, t1);
    pa_mul(rr, t1, result + 4);
    pa_mul(py, j, t1);
    pa_dbl(t1, t1);
    pa_sub(result + 4, t1, result + 4);

    pa_add(pz, h, t1);
    pa_sqr(t1, t1);
    pa_sub(t1, z1z1, t1);
    pa_sub(t1, hh, result + 8);
    return 0;
}

static void pa_proj_dbl(const uint64_t p[12], uint64_t r[12]) {
    pa_pt_dbl(p, r);
}

// Convert affine (8 u64) to projective (12 u64) with Z=1.
static void pa_aff_to_proj(const uint64_t aff[8], uint64_t proj[12]) {
    memcpy(proj, aff, 64);
    memcpy(proj + 8, PA_ONE, 32);
}

// Negate y coordinate of affine point (in-place, Montgomery form).
static void pa_aff_neg_inplace(uint64_t aff[8]) {
    pa_neg(aff + 4, aff + 4);
}

// Multiply x coordinate of affine point by scalar (in-place).
static void pa_aff_scale_x(uint64_t aff[8], const uint64_t scalar[4]) {
    uint64_t new_x[4];
    pa_mul(aff, scalar, new_x);
    memcpy(aff, new_x, 32);
}

// Add two affine points (result may alias either input).
static void pa_aff_add(const uint64_t a[8], const uint64_t b[8], uint64_t r[8]) {
    uint64_t t[12];
    pa_aff_to_proj(a, t);
    pa_proj_add_aff(t, b, t);
    memcpy(r, t, 64);
}

// Convert projective point to affine (Z=1 assumed, or handle Z=0 -> identity).
static void pa_proj_to_aff(const uint64_t proj[12], uint64_t aff[8]) {
    if (pa_pt_is_id(proj)) {
        memset(aff, 0, 64);
        return;
    }
    // affine = proj / Z: x' = X / Z^2, y' = Y / Z^3
    uint64_t z_inv[4], z_inv2[4], z_inv3[4];
    pa_inv(proj + 8, z_inv);
    pa_sqr(z_inv, z_inv2);
    pa_mul(z_inv2, z_inv, z_inv3);
    pa_mul(proj, z_inv2, aff);
    pa_mul(proj + 4, z_inv3, aff + 4);
}

// Signed-digit endo-combine for Pallas.
// g1, g2: count affine points (8 u64 each: x[4], y[4]).
// endo_coeff: 4 u64 Montgomery form.
// scalars: count * 8 u64 (128-bit little-endian, standard integer form).
// result: count affine points (8 u64 each).
void batch_pallas_endo_combine(
    const uint64_t *g1_x, const uint64_t *g1_y,
    const uint64_t *g2_x, const uint64_t *g2_y,
    const uint64_t *endo_coeff,
    const uint64_t *scalars,
    uint32_t count,
    uint64_t *result_x, uint64_t *result_y)
{
    if (count == 0) return;

    for (uint32_t i = 0; i < count; i++) {
        const uint64_t *g2_ax = g2_x + i * 4;
        const uint64_t *g2_ay = g2_y + i * 4;
        const uint64_t *sc = scalars + i * 8;

        // Build table entries in projective form:
        // t0 = identity (already zeroed)
        // t1 = g2 (projective)
        // t2 = endo(g2) — multiply x by endo_coeff
        // t3 = g2 + endo(g2) — add in affine then convert

        uint64_t t1[12], t2[12], t3[12], t4[12];
        uint64_t acc[12], tmp_s[12], tmp_acc[12];

        // t1 = g2 (projective)
        pa_aff_to_proj(g2_ax, t1);

        // t2 = endo(g2): multiply x by endo_coeff, keep y
        memcpy(t2, g2_ax, 32);
        pa_mul(t2, endo_coeff, t2);
        memcpy(t2 + 4, g2_ay, 32);
        memcpy(t2 + 8, PA_ONE, 32);

        // t3 = g2 + endo(g2) in affine
        uint64_t t3_aff[8];
        pa_aff_add(g2_ax, t2, t3_aff);
        pa_aff_to_proj(t3_aff, t3);

        // acc = t3 + t3 = 2 * (g2 + endo(g2)) = 2 * (phi(g2) + g2)
        pa_proj_dbl(t3, acc);

        // Process 64 iterations of 2 bits each (128-bit scalar).
        // Each iteration: acc = acc + s; acc = acc + acc, where s depends on scalar bits.
        for (int iter = 63; iter >= 0; iter--) {
            // Extract 2 bits at positions 2*iter and 2*iter+1
            int word_idx = (2 * iter) / 64;
            int bit_idx = (2 * iter) % 64;

            uint64_t w = sc[word_idx];
            if (bit_idx + 1 < 64) {
                w |= sc[word_idx + 1] << 64;
            }
            uint64_t bit0 = (w >> (2 * iter)) & 1;
            uint64_t bit1 = (w >> (2 * iter + 1)) & 1;

            // s = g2 (affine form in tmp_s, then projective in tmp_s)
            uint64_t tmp_s_aff[8];
            memcpy(tmp_s_aff, g2_ax, 32);
            memcpy(tmp_s_aff + 4, g2_ay, 32);

            // Conditionally negate s
            if (bit0) {
                pa_aff_neg_inplace(tmp_s_aff);
            }

            // Conditionally apply endomorphism to s
            if (bit1) {
                pa_aff_scale_x(tmp_s_aff, endo_coeff);
            }

            // acc = acc + s (projective + affine)
            pa_proj_add_aff(acc, tmp_s_aff, tmp_acc);
            memcpy(acc, tmp_acc, 96);

            // acc = acc + acc = 2 * acc
            pa_proj_dbl(acc, tmp_acc);
            memcpy(acc, tmp_acc, 96);
        }

        // acc = acc + g1[i] (projective + affine)
        uint64_t g1_aff[8];
        memcpy(g1_aff, g1_x + i * 4, 32);
        memcpy(g1_aff + 4, g1_y + i * 4, 32);
        pa_proj_add_aff(acc, g1_aff, acc);

        // Convert to affine
        uint64_t res_aff[8];
        pa_proj_to_aff(acc, res_aff);
        memcpy(result_x + i * 4, res_aff, 32);
        memcpy(result_y + i * 4, res_aff + 4, 32);
    }
}

// Same helper functions for Vesta
static int ve_proj_add_aff(const uint64_t projP[12], const uint64_t affQ[8], uint64_t result[12]) {
    if (ve_pt_is_id(projP)) {
        memcpy(result, affQ, 64);
        memcpy(result + 8, VE_ONE, 32);
        return 0;
    }
    if (ve_is_zero(affQ)) {
        memcpy(result, projP, 96);
        return 0;
    }
    const uint64_t *px = projP, *py = projP+4, *pz = projP+8;
    const uint64_t *qx = affQ, *qy = affQ+4;

    uint64_t z1z1[4], u2[4], s2[4], h[4], hh[4], rr[4], t1[4];
    uint64_t j[4], vv[4];

    ve_sqr(pz, z1z1);
    ve_mul(qx, z1z1, u2);
    ve_mul(pz, z1z1, t1);
    ve_mul(qy, t1, s2);

    ve_sub(u2, px, h);
    ve_sub(s2, py, t1);
    ve_dbl(t1, rr);

    if (ve_is_zero(h)) {
        if (ve_is_zero(rr)) {
            ve_pt_dbl(projP, result);
            return 0;
        }
        return 1;
    }

    ve_sqr(h, hh);
    ve_mul(h, hh, j);
    ve_mul(px, hh, vv);

    ve_sqr(rr, result);
    ve_sub(result, j, result);
    ve_dbl(vv, t1);
    ve_sub(result, t1, result);

    ve_sub(vv, result, t1);
    ve_mul(rr, t1, result + 4);
    ve_mul(py, j, t1);
    ve_dbl(t1, t1);
    ve_sub(result + 4, t1, result + 4);

    ve_add(pz, h, t1);
    ve_sqr(t1, t1);
    ve_sub(t1, z1z1, t1);
    ve_sub(t1, hh, result + 8);
    return 0;
}

static void ve_proj_dbl(const uint64_t p[12], uint64_t r[12]) {
    ve_pt_dbl(p, r);
}

static void ve_aff_to_proj(const uint64_t aff[8], uint64_t proj[12]) {
    memcpy(proj, aff, 64);
    memcpy(proj + 8, VE_ONE, 32);
}

static void ve_aff_neg_inplace(uint64_t aff[8]) {
    ve_neg(aff + 4, aff + 4);
}

static void ve_aff_scale_x(uint64_t aff[8], const uint64_t scalar[4]) {
    uint64_t new_x[4];
    ve_mul(aff, scalar, new_x);
    memcpy(aff, new_x, 32);
}

static void ve_aff_add(const uint64_t a[8], const uint64_t b[8], uint64_t r[8]) {
    uint64_t t[12];
    ve_aff_to_proj(a, t);
    ve_proj_add_aff(t, b, t);
    memcpy(r, t, 64);
}

static void ve_proj_to_aff(const uint64_t proj[12], uint64_t aff[8]) {
    if (ve_pt_is_id(proj)) {
        memset(aff, 0, 64);
        return;
    }
    uint64_t z_inv[4], z_inv2[4], z_inv3[4];
    ve_inv(proj + 8, z_inv);
    ve_sqr(z_inv, z_inv2);
    ve_mul(z_inv2, z_inv, z_inv3);
    ve_mul(proj, z_inv2, aff);
    ve_mul(proj + 4, z_inv3, aff + 4);
}

void batch_vesta_endo_combine(
    const uint64_t *g1_x, const uint64_t *g1_y,
    const uint64_t *g2_x, const uint64_t *g2_y,
    const uint64_t *endo_coeff,
    const uint64_t *scalars,
    uint32_t count,
    uint64_t *result_x, uint64_t *result_y)
{
    if (count == 0) return;

    for (uint32_t i = 0; i < count; i++) {
        const uint64_t *g2_ax = g2_x + i * 4;
        const uint64_t *g2_ay = g2_y + i * 4;
        const uint64_t *sc = scalars + i * 8;

        uint64_t t1[12], t2[12], t3[12], t4[12];
        uint64_t acc[12], tmp_s[12], tmp_acc[12];

        // t1 = g2
        ve_aff_to_proj(g2_ax, t1);

        // t2 = endo(g2)
        memcpy(t2, g2_ax, 32);
        ve_mul(t2, endo_coeff, t2);
        memcpy(t2 + 4, g2_ay, 32);
        memcpy(t2 + 8, VE_ONE, 32);

        // t3 = g2 + endo(g2)
        uint64_t t3_aff[8];
        ve_aff_add(g2_ax, t2, t3_aff);
        ve_aff_to_proj(t3_aff, t3);

        // acc = 2 * (g2 + endo(g2))
        ve_proj_dbl(t3, acc);

        for (int iter = 63; iter >= 0; iter--) {
            int word_idx = (2 * iter) / 64;
            int bit_idx = (2 * iter) % 64;

            uint64_t w = sc[word_idx];
            if (bit_idx + 1 < 64) {
                w |= sc[word_idx + 1] << 64;
            }
            uint64_t bit0 = (w >> (2 * iter)) & 1;
            uint64_t bit1 = (w >> (2 * iter + 1)) & 1;

            uint64_t tmp_s_aff[8];
            memcpy(tmp_s_aff, g2_ax, 32);
            memcpy(tmp_s_aff + 4, g2_ay, 32);

            if (bit0) {
                ve_aff_neg_inplace(tmp_s_aff);
            }

            if (bit1) {
                ve_aff_scale_x(tmp_s_aff, endo_coeff);
            }

            ve_proj_add_aff(acc, tmp_s_aff, tmp_acc);
            memcpy(acc, tmp_acc, 96);

            ve_proj_dbl(acc, tmp_acc);
            memcpy(acc, tmp_acc, 96);
        }

        uint64_t g1_aff[8];
        memcpy(g1_aff, g1_x + i * 4, 32);
        memcpy(g1_aff + 4, g1_y + i * 4, 32);
        ve_proj_add_aff(acc, g1_aff, acc);

        uint64_t res_aff[8];
        ve_proj_to_aff(acc, res_aff);
        memcpy(result_x + i * 4, res_aff, 32);
        memcpy(result_y + i * 4, res_aff + 4, 32);
    }
}
