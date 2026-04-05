// Pasta (Pallas + Vesta) curve operations in C with CIOS Montgomery arithmetic.
// Pallas: y^2 = x^3 + 5 over Pallas Fp
// Vesta:  y^2 = x^3 + 5 over Vesta Fp
// Cycle property: Pallas Fp = Vesta Fr, Vesta Fp = Pallas Fr.
// Points in Jacobian projective coordinates: (X, Y, Z) -> affine (X/Z^2, Y/Z^3).
// Each coordinate is 4 x uint64_t in Montgomery form.

#include "NeonFieldOps.h"
#include <string.h>

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
