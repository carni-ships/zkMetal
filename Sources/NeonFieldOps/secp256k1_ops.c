// secp256k1 field arithmetic and point operations — optimized C implementation
// CIOS Montgomery multiplication with __uint128_t for ARM64
//
// Curve: y^2 = x^3 + 7 (a=0, b=7)
// Base field p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
//             = 2^256 - 2^32 - 977
//
// Interop: Swift SecpFp = 8×UInt32 = 4×uint64_t (Montgomery form)
//          SecpPointProjective = 3×SecpFp = 12×uint64_t (x[4], y[4], z[4])
//          Scalars: 4×uint64_t limbs (little-endian, non-Montgomery integer)

#include "NeonFieldOps.h"
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

typedef unsigned __int128 uint128_t;

// ============================================================
// secp256k1 Fp constants
// ============================================================

static const uint64_t SECP_P[4] = {
    0xfffffffefffffc2fULL, 0xffffffffffffffffULL,
    0xffffffffffffffffULL, 0xffffffffffffffffULL
};
static const uint64_t SECP_INV = 0xd838091dd2253531ULL;  // -p^{-1} mod 2^64
static const uint64_t SECP_ONE[4] = {  // R mod p (Montgomery form of 1)
    0x00000001000003d1ULL, 0x0000000000000000ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL
};

// ============================================================
// CIOS Montgomery multiplication (fully unrolled, 4 limbs)
// ============================================================

static inline void secp_fp_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0;

    // Iteration i=0
    {
        uint128_t w;
        w = (uint128_t)a[0] * b[0];
        t0 = (uint64_t)w; uint64_t c = (uint64_t)(w >> 64);
        w = (uint128_t)a[0] * b[1] + t1 + c;
        t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[0] * b[2] + t2 + c;
        t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[0] * b[3] + t3 + c;
        t3 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t4 = c;
        uint64_t m = t0 * SECP_INV;
        w = (uint128_t)m * SECP_P[0] + t0; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * SECP_P[1] + t1 + c; t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * SECP_P[2] + t2 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * SECP_P[3] + t3 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t3 = t4 + c; t4 = (t3 < c) ? 1 : 0;
    }
    // Iteration i=1
    {
        uint128_t w; uint64_t c;
        w = (uint128_t)a[1] * b[0] + t0; t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[1] * b[1] + t1 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[1] * b[2] + t2 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[1] * b[3] + t3 + c; t3 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t4 += c;
        uint64_t m = t0 * SECP_INV;
        w = (uint128_t)m * SECP_P[0] + t0; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * SECP_P[1] + t1 + c; t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * SECP_P[2] + t2 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * SECP_P[3] + t3 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t3 = t4 + c; t4 = (t3 < c) ? 1 : 0;
    }
    // Iteration i=2
    {
        uint128_t w; uint64_t c;
        w = (uint128_t)a[2] * b[0] + t0; t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[2] * b[1] + t1 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[2] * b[2] + t2 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[2] * b[3] + t3 + c; t3 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t4 += c;
        uint64_t m = t0 * SECP_INV;
        w = (uint128_t)m * SECP_P[0] + t0; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * SECP_P[1] + t1 + c; t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * SECP_P[2] + t2 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * SECP_P[3] + t3 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t3 = t4 + c; t4 = (t3 < c) ? 1 : 0;
    }
    // Iteration i=3
    {
        uint128_t w; uint64_t c;
        w = (uint128_t)a[3] * b[0] + t0; t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[3] * b[1] + t1 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[3] * b[2] + t2 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[3] * b[3] + t3 + c; t3 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t4 += c;
        uint64_t m = t0 * SECP_INV;
        w = (uint128_t)m * SECP_P[0] + t0; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * SECP_P[1] + t1 + c; t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * SECP_P[2] + t2 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * SECP_P[3] + t3 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t3 = t4 + c; t4 = (t3 < c) ? 1 : 0;
    }

    // Final reduction
    if (t4 || (t3 > SECP_P[3]) ||
        (t3 == SECP_P[3] && (t2 > SECP_P[2] ||
        (t2 == SECP_P[2] && (t1 > SECP_P[1] ||
        (t1 == SECP_P[1] && t0 >= SECP_P[0])))))) {
        uint128_t borrow;
        borrow = (uint128_t)t0 - SECP_P[0]; t0 = (uint64_t)borrow;
        uint64_t b1 = (uint64_t)(borrow >> 64) & 1;
        borrow = (uint128_t)t1 - SECP_P[1] - b1; t1 = (uint64_t)borrow;
        uint64_t b2 = (uint64_t)(borrow >> 64) & 1;
        borrow = (uint128_t)t2 - SECP_P[2] - b2; t2 = (uint64_t)borrow;
        uint64_t b3 = (uint64_t)(borrow >> 64) & 1;
        t3 = t3 - SECP_P[3] - b3;
    }

    r[0] = t0; r[1] = t1; r[2] = t2; r[3] = t3;
}

static inline void secp_fp_sqr(const uint64_t a[4], uint64_t r[4]) {
    secp_fp_mul(a, a, r);
}

static inline void secp_fp_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t s;
    s = (uint128_t)a[0] + b[0]; r[0] = (uint64_t)s;
    s = (uint128_t)a[1] + b[1] + (uint64_t)(s >> 64); r[1] = (uint64_t)s;
    s = (uint128_t)a[2] + b[2] + (uint64_t)(s >> 64); r[2] = (uint64_t)s;
    s = (uint128_t)a[3] + b[3] + (uint64_t)(s >> 64); r[3] = (uint64_t)s;
    uint64_t carry = (uint64_t)(s >> 64);

    if (carry || (r[3] > SECP_P[3]) ||
        (r[3] == SECP_P[3] && (r[2] > SECP_P[2] ||
        (r[2] == SECP_P[2] && (r[1] > SECP_P[1] ||
        (r[1] == SECP_P[1] && r[0] >= SECP_P[0])))))) {
        uint128_t borrow;
        borrow = (uint128_t)r[0] - SECP_P[0]; r[0] = (uint64_t)borrow;
        uint64_t b1 = (uint64_t)(borrow >> 64) & 1;
        borrow = (uint128_t)r[1] - SECP_P[1] - b1; r[1] = (uint64_t)borrow;
        uint64_t b2 = (uint64_t)(borrow >> 64) & 1;
        borrow = (uint128_t)r[2] - SECP_P[2] - b2; r[2] = (uint64_t)borrow;
        uint64_t b3 = (uint64_t)(borrow >> 64) & 1;
        r[3] = r[3] - SECP_P[3] - b3;
    }
}

static inline void secp_fp_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t borrow;
    borrow = (uint128_t)a[0] - b[0]; r[0] = (uint64_t)borrow;
    uint64_t b1 = (uint64_t)(borrow >> 64) & 1;
    borrow = (uint128_t)a[1] - b[1] - b1; r[1] = (uint64_t)borrow;
    uint64_t b2 = (uint64_t)(borrow >> 64) & 1;
    borrow = (uint128_t)a[2] - b[2] - b2; r[2] = (uint64_t)borrow;
    uint64_t b3 = (uint64_t)(borrow >> 64) & 1;
    borrow = (uint128_t)a[3] - b[3] - b3; r[3] = (uint64_t)borrow;
    uint64_t underflow = (uint64_t)(borrow >> 64) & 1;

    if (underflow) {
        uint128_t s;
        s = (uint128_t)r[0] + SECP_P[0]; r[0] = (uint64_t)s;
        s = (uint128_t)r[1] + SECP_P[1] + (uint64_t)(s >> 64); r[1] = (uint64_t)s;
        s = (uint128_t)r[2] + SECP_P[2] + (uint64_t)(s >> 64); r[2] = (uint64_t)s;
        r[3] = r[3] + SECP_P[3] + (uint64_t)(s >> 64);
    }
}

static inline void secp_fp_dbl(const uint64_t a[4], uint64_t r[4]) {
    secp_fp_add(a, a, r);
}

// ============================================================
// Point operations: Jacobian projective coordinates
// y^2 = x^3 + 7 (a=0)
// Identity: Z = 0
// ============================================================

static inline int secp_pt_is_id(const uint64_t p[12]) {
    return p[8] == 0 && p[9] == 0 && p[10] == 0 && p[11] == 0;
}

static inline void secp_pt_set_id(uint64_t p[12]) {
    memcpy(p, SECP_ONE, 32);      // x = 1
    memcpy(p + 4, SECP_ONE, 32);  // y = 1
    memset(p + 8, 0, 32);         // z = 0
}

// Point doubling (a=0 curve: y^2 = x^3 + b)
// Using standard Jacobian doubling formula for a=0
static void secp_pt_dbl(const uint64_t p[12], uint64_t r[12]) {
    if (secp_pt_is_id(p)) { memcpy(r, p, 96); return; }

    const uint64_t *px = p, *py = p + 4, *pz = p + 8;
    uint64_t *rx = r, *ry = r + 4, *rz = r + 8;

    uint64_t a[4], b[4], c[4], d[4], e[4], f[4], t[4];

    secp_fp_sqr(px, a);          // a = X^2
    secp_fp_sqr(py, b);          // b = Y^2
    secp_fp_sqr(b, c);           // c = Y^4

    // d = 2*((X+B)^2 - A - C)
    secp_fp_add(px, b, t);
    secp_fp_sqr(t, t);
    secp_fp_sub(t, a, t);
    secp_fp_sub(t, c, t);
    secp_fp_dbl(t, d);

    // e = 3*A
    secp_fp_dbl(a, e);
    secp_fp_add(e, a, e);

    secp_fp_sqr(e, f);           // f = E^2

    // X3 = F - 2*D
    secp_fp_dbl(d, t);
    secp_fp_sub(f, t, rx);

    // Y3 = E*(D - X3) - 8*C
    secp_fp_sub(d, rx, t);
    secp_fp_mul(e, t, ry);
    secp_fp_dbl(c, t);
    secp_fp_dbl(t, t);
    secp_fp_dbl(t, t);           // 8*C
    secp_fp_sub(ry, t, ry);

    // Z3 = (Y+Z)^2 - B - Z^2
    secp_fp_add(py, pz, t);
    secp_fp_sqr(t, rz);
    secp_fp_sub(rz, b, rz);
    uint64_t z2[4];
    secp_fp_sqr(pz, z2);
    secp_fp_sub(rz, z2, rz);
}

// Full projective addition (Jacobian + Jacobian)
static void secp_pt_add(const uint64_t p[12], const uint64_t q[12], uint64_t r[12]) {
    if (secp_pt_is_id(p)) { memcpy(r, q, 96); return; }
    if (secp_pt_is_id(q)) { memcpy(r, p, 96); return; }

    const uint64_t *px = p, *py = p + 4, *pz = p + 8;
    const uint64_t *qx = q, *qy = q + 4, *qz = q + 8;

    uint64_t z1z1[4], z2z2[4], u1[4], u2[4], s1[4], s2[4];
    uint64_t h[4], rr[4], t[4];

    secp_fp_sqr(pz, z1z1);
    secp_fp_sqr(qz, z2z2);
    secp_fp_mul(px, z2z2, u1);
    secp_fp_mul(qx, z1z1, u2);

    secp_fp_mul(qz, z2z2, t);
    secp_fp_mul(py, t, s1);
    secp_fp_mul(pz, z1z1, t);
    secp_fp_mul(qy, t, s2);

    secp_fp_sub(u2, u1, h);

    // 2*h
    uint64_t h2[4];
    secp_fp_dbl(h, h2);

    // Check for doubling case
    if (h2[0] == 0 && h2[1] == 0 && h2[2] == 0 && h2[3] == 0) {
        // h == 0, check if s1 == s2
        uint64_t sdiff[4];
        secp_fp_sub(s2, s1, sdiff);
        secp_fp_dbl(sdiff, sdiff);
        if (sdiff[0] == 0 && sdiff[1] == 0 && sdiff[2] == 0 && sdiff[3] == 0) {
            secp_pt_dbl(p, r);
        } else {
            secp_pt_set_id(r);
        }
        return;
    }

    // rr = 2*(S2 - S1)
    secp_fp_sub(s2, s1, rr);
    secp_fp_dbl(rr, rr);

    uint64_t i[4], j[4], v[4];
    secp_fp_sqr(h2, i);           // i = (2*h)^2
    secp_fp_mul(h, i, j);         // j = h * i
    secp_fp_mul(u1, i, v);        // v = u1 * i

    // X3 = rr^2 - j - 2*v
    uint64_t *rx = r, *ry = r + 4, *rz = r + 8;
    secp_fp_sqr(rr, rx);
    secp_fp_sub(rx, j, rx);
    secp_fp_dbl(v, t);
    secp_fp_sub(rx, t, rx);

    // Y3 = rr*(v - X3) - 2*s1*j
    secp_fp_sub(v, rx, t);
    secp_fp_mul(rr, t, ry);
    secp_fp_mul(s1, j, t);
    secp_fp_dbl(t, t);
    secp_fp_sub(ry, t, ry);

    // Z3 = 2 * Z1 * Z2 * h = ((Z1+Z2)^2 - Z1Z1 - Z2Z2) * h
    // Simplified: Z3 = 2*Z1*Z2*h
    secp_fp_mul(pz, qz, t);
    secp_fp_dbl(t, t);
    secp_fp_mul(t, h, rz);
}

// ============================================================
// Scalar multiplication: double-and-add
// scalar: 4×uint64_t (little-endian, non-Montgomery integer)
// ============================================================

static void secp_pt_scalar_mul(const uint64_t p[12], const uint64_t scalar[4], uint64_t r[12]) {
    uint64_t result[12], base[12];
    secp_pt_set_id(result);
    memcpy(base, p, 96);

    for (int i = 0; i < 4; i++) {
        uint64_t word = scalar[i];
        for (int bit = 0; bit < 64; bit++) {
            if (word & 1) {
                uint64_t tmp[12];
                secp_pt_add(result, base, tmp);
                memcpy(result, tmp, 96);
            }
            uint64_t tmp2[12];
            secp_pt_dbl(base, tmp2);
            memcpy(base, tmp2, 96);
            word >>= 1;
        }
    }
    memcpy(r, result, 96);
}

// ============================================================
// Field inversion via Fermat's little theorem: a^(p-2)
// ============================================================

static void secp_fp_inv(const uint64_t a[4], uint64_t r[4]) {
    // p-2 for secp256k1
    uint64_t exp[4] = {
        SECP_P[0] - 2, SECP_P[1], SECP_P[2], SECP_P[3]
    };

    uint64_t result[4], base[4];
    memcpy(result, SECP_ONE, 32);
    memcpy(base, a, 32);

    for (int i = 0; i < 4; i++) {
        uint64_t word = exp[i];
        for (int bit = 0; bit < 64; bit++) {
            if (word & 1) {
                uint64_t tmp[4];
                secp_fp_mul(result, base, tmp);
                memcpy(result, tmp, 32);
            }
            uint64_t tmp2[4];
            secp_fp_sqr(base, tmp2);
            memcpy(base, tmp2, 32);
            word >>= 1;
        }
    }
    memcpy(r, result, 32);
}

// Convert projective to affine
static void secp_pt_to_affine(const uint64_t p[12], uint64_t ax[4], uint64_t ay[4]) {
    if (secp_pt_is_id(p)) {
        memset(ax, 0, 32);
        memset(ay, 0, 32);
        return;
    }
    uint64_t zinv[4], zinv2[4], zinv3[4];
    secp_fp_inv(p + 8, zinv);
    secp_fp_sqr(zinv, zinv2);
    secp_fp_mul(zinv2, zinv, zinv3);
    secp_fp_mul(p, zinv2, ax);
    secp_fp_mul(p + 4, zinv3, ay);
}

// ============================================================
// Mixed affine addition: projective P + affine Q (Z_Q = 1)
// Saves 2 muls + 1 sqr vs full projective add
// ============================================================

static inline int secp_aff_is_id(const uint64_t q[8]) {
    return (q[0] | q[1] | q[2] | q[3] | q[4] | q[5] | q[6] | q[7]) == 0;
}

static void secp_pt_add_mixed(const uint64_t p[12], const uint64_t q_aff[8], uint64_t r[12]) {
    if (secp_aff_is_id(q_aff)) {
        memcpy(r, p, 96);
        return;
    }
    if (secp_pt_is_id(p)) {
        memcpy(r, q_aff, 64);        // x, y from affine
        memcpy(r + 8, SECP_ONE, 32); // z = 1
        return;
    }

    const uint64_t *px = p, *py = p+4, *pz = p+8;
    const uint64_t *qx = q_aff, *qy = q_aff + 4;

    uint64_t z1z1[4], u2[4], s2[4], h[4], rr[4];
    uint64_t ii[4], j[4], vv[4], t1[4], hh[4];

    secp_fp_sqr(pz, z1z1);
    secp_fp_mul(qx, z1z1, u2);         // u2 = qx * z1²
    secp_fp_mul(pz, z1z1, t1);
    secp_fp_mul(qy, t1, s2);           // s2 = qy * z1³

    secp_fp_sub(u2, px, h);            // h = u2 - px (u1 = px)
    secp_fp_sub(s2, py, t1);
    secp_fp_dbl(t1, rr);               // r = 2(s2 - s1)

    if (h[0] == 0 && h[1] == 0 && h[2] == 0 && h[3] == 0) {
        if (rr[0] == 0 && rr[1] == 0 && rr[2] == 0 && rr[3] == 0) {
            secp_pt_dbl(p, r); return;
        }
        secp_pt_set_id(r); return;
    }

    secp_fp_dbl(h, t1);
    secp_fp_sqr(t1, ii);
    secp_fp_mul(h, ii, j);
    secp_fp_mul(px, ii, vv);

    // x3 = r² - j - 2v
    secp_fp_sqr(rr, r);
    secp_fp_sub(r, j, r);
    secp_fp_dbl(vv, t1);
    secp_fp_sub(r, t1, r);

    // y3 = r(v - x3) - 2*s1*j
    secp_fp_sub(vv, r, t1);
    secp_fp_mul(rr, t1, r + 4);
    secp_fp_mul(py, j, t1);
    secp_fp_dbl(t1, t1);
    secp_fp_sub(r + 4, t1, r + 4);

    // z3 = (z1 + h)² - z1z1 - h²
    secp_fp_add(pz, h, t1);
    secp_fp_sqr(t1, t1);
    secp_fp_sub(t1, z1z1, t1);
    secp_fp_sqr(h, hh);
    secp_fp_sub(t1, hh, r + 8);
}

// ============================================================
// Batch projective-to-affine (Montgomery's trick)
// ============================================================

static void secp_batch_to_affine(const uint64_t *proj, uint64_t *aff, int n) {
    if (n == 0) return;

    uint64_t *prods = (uint64_t *)malloc((size_t)n * 32);
    int first_valid = -1;

    for (int i = 0; i < n; i++) {
        if (secp_pt_is_id(proj + i * 12)) {
            if (i == 0) memcpy(prods, SECP_ONE, 32);
            else memcpy(prods + i * 4, prods + (i-1) * 4, 32);
        } else {
            if (first_valid < 0) {
                first_valid = i;
                memcpy(prods + i * 4, proj + i * 12 + 8, 32);  // Z_i
            } else {
                secp_fp_mul(prods + (i-1) * 4, proj + i * 12 + 8, prods + i * 4);
            }
        }
    }

    if (first_valid < 0) {
        memset(aff, 0, (size_t)n * 64);
        free(prods);
        return;
    }

    uint64_t inv[4];
    secp_fp_inv(prods + (n-1) * 4, inv);

    for (int i = n - 1; i >= 0; i--) {
        if (secp_pt_is_id(proj + i * 12)) {
            memset(aff + i * 8, 0, 64);
            continue;
        }

        uint64_t zinv[4];
        if (i > first_valid) {
            secp_fp_mul(inv, prods + (i-1) * 4, zinv);
            secp_fp_mul(inv, proj + i * 12 + 8, inv);
        } else {
            memcpy(zinv, inv, 32);
        }

        uint64_t zinv2[4], zinv3[4];
        secp_fp_sqr(zinv, zinv2);
        secp_fp_mul(zinv2, zinv, zinv3);
        secp_fp_mul(proj + i * 12, zinv2, aff + i * 8);
        secp_fp_mul(proj + i * 12 + 4, zinv3, aff + i * 8 + 4);
    }

    free(prods);
}

// ============================================================
// Pippenger MSM helpers
// ============================================================

static inline uint32_t secp_extract_window(const uint32_t *scalar, int window_idx, int window_bits) {
    int bit_offset = window_idx * window_bits;
    int word_idx = bit_offset / 32;
    int bit_in_word = bit_offset % 32;

    uint64_t word = scalar[word_idx];
    if (word_idx + 1 < 8)
        word |= ((uint64_t)scalar[word_idx + 1]) << 32;

    return (uint32_t)((word >> bit_in_word) & ((1u << window_bits) - 1));
}

static int secp_optimal_window_bits(int n) {
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

// Per-window worker
typedef struct {
    const uint64_t *points;     // n affine points (8 limbs each)
    const uint32_t *scalars;    // n scalars (8 uint32 each)
    int n;
    int window_bits;
    int window_idx;
    int num_buckets;
    uint64_t result[12];
} SecpWindowTask;

static void *secp_window_worker(void *arg) {
    SecpWindowTask *task = (SecpWindowTask *)arg;
    int wb = task->window_bits;
    int w = task->window_idx;
    int nb = task->num_buckets;
    int nn = task->n;

    // Allocate projective buckets
    uint64_t *buckets = (uint64_t *)malloc((size_t)(nb + 1) * 96);
    for (int b = 0; b <= nb; b++)
        secp_pt_set_id(buckets + b * 12);

    // Phase 1: Bucket accumulation (mixed affine addition)
    for (int i = 0; i < nn; i++) {
        uint32_t digit = secp_extract_window(task->scalars + i * 8, w, wb);
        if (digit != 0) {
            uint64_t tmp[12];
            secp_pt_add_mixed(buckets + digit * 12, task->points + i * 8, tmp);
            memcpy(buckets + digit * 12, tmp, 96);
        }
    }

    // Phase 2: Batch convert buckets to affine
    uint64_t *bucket_aff = (uint64_t *)malloc((size_t)nb * 64);
    secp_batch_to_affine(buckets + 12, bucket_aff, nb);

    // Phase 3: Running-sum reduction using mixed addition
    uint64_t running[12], window_sum[12];
    secp_pt_set_id(running);
    secp_pt_set_id(window_sum);

    for (int j = nb - 1; j >= 0; j--) {
        if (!(bucket_aff[j*8] == 0 && bucket_aff[j*8+1] == 0 &&
              bucket_aff[j*8+2] == 0 && bucket_aff[j*8+3] == 0 &&
              bucket_aff[j*8+4] == 0 && bucket_aff[j*8+5] == 0 &&
              bucket_aff[j*8+6] == 0 && bucket_aff[j*8+7] == 0)) {
            uint64_t tmp[12];
            secp_pt_add_mixed(running, bucket_aff + j * 8, tmp);
            memcpy(running, tmp, 96);
        }
        uint64_t tmp[12];
        secp_pt_add(window_sum, running, tmp);
        memcpy(window_sum, tmp, 96);
    }

    memcpy(task->result, window_sum, 96);
    free(buckets);
    free(bucket_aff);
    return NULL;
}

// ============================================================
// Exported functions
// ============================================================

void secp256k1_point_scalar_mul(const uint64_t p[12], const uint64_t scalar[4], uint64_t r[12]) {
    secp_pt_scalar_mul(p, scalar, r);
}

void secp256k1_point_add(const uint64_t p[12], const uint64_t q[12], uint64_t r[12]) {
    secp_pt_add(p, q, r);
}

// Shamir's trick: compute s1*P1 + s2*P2 in a single double-and-add scan.
// Precomputes P1, P2, P1+P2 and scans both scalars MSB-to-LSB simultaneously.
// ~25% faster than two separate scalar muls + add.
void secp256k1_shamir_double_mul(const uint64_t p1[12], const uint64_t s1[4],
                                  const uint64_t p2[12], const uint64_t s2[4],
                                  uint64_t r[12]) {
    // Precompute table: table[0]=identity, table[1]=P1, table[2]=P2, table[3]=P1+P2
    uint64_t table[4][12];
    secp_pt_set_id(table[0]);
    memcpy(table[1], p1, 96);
    memcpy(table[2], p2, 96);
    secp_pt_add(p1, p2, table[3]);

    // Find highest set bit across both scalars
    int top_bit = -1;
    for (int i = 3; i >= 0; i--) {
        uint64_t combined = s1[i] | s2[i];
        if (combined != 0) {
            int bit_in_word = 63 - __builtin_clzll(combined);
            top_bit = i * 64 + bit_in_word;
            break;
        }
    }

    if (top_bit < 0) {
        secp_pt_set_id(r);
        return;
    }

    uint64_t result[12];
    secp_pt_set_id(result);

    for (int bit = top_bit; bit >= 0; bit--) {
        // Double
        uint64_t tmp[12];
        secp_pt_dbl(result, tmp);
        memcpy(result, tmp, 96);

        // Get bit from each scalar
        int word = bit / 64;
        int pos = bit % 64;
        int b1 = (s1[word] >> pos) & 1;
        int b2 = (s2[word] >> pos) & 1;
        int idx = b1 | (b2 << 1);

        if (idx != 0) {
            secp_pt_add(result, table[idx], tmp);
            memcpy(result, tmp, 96);
        }
    }

    memcpy(r, result, 96);
}

void secp256k1_point_to_affine(const uint64_t p[12], uint64_t ax[4], uint64_t ay[4]) {
    secp_pt_to_affine(p, ax, ay);
}

void secp256k1_pippenger_msm(const uint64_t *points, const uint32_t *scalars,
                              int n, uint64_t *result) {
    if (n == 0) {
        secp_pt_set_id(result);
        return;
    }
    if (n == 1) {
        // Single point: just scalar mul
        // Convert affine to projective
        uint64_t proj[12];
        memcpy(proj, points, 64);
        memcpy(proj + 8, SECP_ONE, 32);
        // Convert 8×uint32 scalar to 4×uint64
        uint64_t sc64[4];
        for (int i = 0; i < 4; i++)
            sc64[i] = (uint64_t)scalars[2*i] | ((uint64_t)scalars[2*i+1] << 32);
        secp_pt_scalar_mul(proj, sc64, result);
        return;
    }

    int wb = secp_optimal_window_bits(n);
    int nw = (256 + wb - 1) / wb;
    int nb = (1 << wb) - 1;

    // Launch window workers in parallel
    SecpWindowTask *tasks = (SecpWindowTask *)malloc((size_t)nw * sizeof(SecpWindowTask));
    pthread_t *threads = (pthread_t *)malloc((size_t)nw * sizeof(pthread_t));

    for (int w = 0; w < nw; w++) {
        tasks[w].points = points;
        tasks[w].scalars = scalars;
        tasks[w].n = n;
        tasks[w].window_bits = wb;
        tasks[w].window_idx = w;
        tasks[w].num_buckets = nb;
    }

    // Limit threads to avoid oversubscription
    int max_threads = 8;
    if (nw <= max_threads) {
        for (int w = 0; w < nw; w++)
            pthread_create(&threads[w], NULL, secp_window_worker, &tasks[w]);
        for (int w = 0; w < nw; w++)
            pthread_join(threads[w], NULL);
    } else {
        // Run in batches
        for (int batch_start = 0; batch_start < nw; batch_start += max_threads) {
            int batch_end = batch_start + max_threads;
            if (batch_end > nw) batch_end = nw;
            for (int w = batch_start; w < batch_end; w++)
                pthread_create(&threads[w], NULL, secp_window_worker, &tasks[w]);
            for (int w = batch_start; w < batch_end; w++)
                pthread_join(threads[w], NULL);
        }
    }

    // Combine windows using Horner's method
    memcpy(result, tasks[nw - 1].result, 96);
    for (int w = nw - 2; w >= 0; w--) {
        for (int b = 0; b < wb; b++) {
            uint64_t tmp[12];
            secp_pt_dbl(result, tmp);
            memcpy(result, tmp, 96);
        }
        uint64_t tmp[12];
        secp_pt_add(result, tasks[w].result, tmp);
        memcpy(result, tmp, 96);
    }

    free(tasks);
    free(threads);
}

// ============================================================
// secp256k1 Fr (scalar field) CIOS Montgomery arithmetic
// n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
// ============================================================

static const uint64_t SECP_N[4] = {
    0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
};
static const uint64_t SECP_FR_INV = 0x4B0DFF665588B13FULL;
static const uint64_t SECP_FR_ONE[4] = {
    0x402DA1732FC9BEBFULL, 0x4551231950B75FC4ULL,
    0x0000000000000001ULL, 0x0000000000000000ULL
};
static const uint64_t SECP_FR_R2[4] = {
    0x896CF21467D7D140ULL, 0x741496C20E7CF878ULL,
    0xE697F5E45BCD07C6ULL, 0x9D671CD581C69BC5ULL
};

static inline void secp_fr_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0;
#define SECP_FR_ITER(I) do { \
        uint128_t w; uint64_t c; \
        w = (uint128_t)a[I] * b[0] + t0; t0 = (uint64_t)w; c = (uint64_t)(w >> 64); \
        w = (uint128_t)a[I] * b[1] + t1 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64); \
        w = (uint128_t)a[I] * b[2] + t2 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64); \
        w = (uint128_t)a[I] * b[3] + t3 + c; t3 = (uint64_t)w; c = (uint64_t)(w >> 64); \
        t4 += c; \
        uint64_t m = t0 * SECP_FR_INV; \
        w = (uint128_t)m * SECP_N[0] + t0; c = (uint64_t)(w >> 64); \
        w = (uint128_t)m * SECP_N[1] + t1 + c; t0 = (uint64_t)w; c = (uint64_t)(w >> 64); \
        w = (uint128_t)m * SECP_N[2] + t2 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64); \
        w = (uint128_t)m * SECP_N[3] + t3 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64); \
        t3 = t4 + c; t4 = (t3 < c) ? 1 : 0; \
    } while(0)
    SECP_FR_ITER(0); SECP_FR_ITER(1); SECP_FR_ITER(2); SECP_FR_ITER(3);
#undef SECP_FR_ITER
    if (t4 || (t3 > SECP_N[3]) ||
        (t3 == SECP_N[3] && (t2 > SECP_N[2] ||
        (t2 == SECP_N[2] && (t1 > SECP_N[1] ||
        (t1 == SECP_N[1] && t0 >= SECP_N[0])))))) {
        uint128_t borrow;
        borrow = (uint128_t)t0 - SECP_N[0]; t0 = (uint64_t)borrow;
        uint64_t b1 = (uint64_t)(borrow >> 64) & 1;
        borrow = (uint128_t)t1 - SECP_N[1] - b1; t1 = (uint64_t)borrow;
        uint64_t b2 = (uint64_t)(borrow >> 64) & 1;
        borrow = (uint128_t)t2 - SECP_N[2] - b2; t2 = (uint64_t)borrow;
        uint64_t b3 = (uint64_t)(borrow >> 64) & 1;
        t3 = t3 - SECP_N[3] - b3;
    }
    r[0] = t0; r[1] = t1; r[2] = t2; r[3] = t3;
}

static inline void secp_fr_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t s;
    s = (uint128_t)a[0] + b[0]; r[0] = (uint64_t)s;
    s = (uint128_t)a[1] + b[1] + (uint64_t)(s >> 64); r[1] = (uint64_t)s;
    s = (uint128_t)a[2] + b[2] + (uint64_t)(s >> 64); r[2] = (uint64_t)s;
    s = (uint128_t)a[3] + b[3] + (uint64_t)(s >> 64); r[3] = (uint64_t)s;
    uint64_t carry = (uint64_t)(s >> 64);
    if (carry || (r[3] > SECP_N[3]) ||
        (r[3] == SECP_N[3] && (r[2] > SECP_N[2] ||
        (r[2] == SECP_N[2] && (r[1] > SECP_N[1] ||
        (r[1] == SECP_N[1] && r[0] >= SECP_N[0])))))) {
        uint128_t borrow;
        borrow = (uint128_t)r[0] - SECP_N[0]; r[0] = (uint64_t)borrow;
        uint64_t b1 = (uint64_t)(borrow >> 64) & 1;
        borrow = (uint128_t)r[1] - SECP_N[1] - b1; r[1] = (uint64_t)borrow;
        uint64_t b2 = (uint64_t)(borrow >> 64) & 1;
        borrow = (uint128_t)r[2] - SECP_N[2] - b2; r[2] = (uint64_t)borrow;
        uint64_t b3 = (uint64_t)(borrow >> 64) & 1;
        r[3] = r[3] - SECP_N[3] - b3;
    }
}

static inline void secp_fr_neg(const uint64_t a[4], uint64_t r[4]) {
    if (a[0] == 0 && a[1] == 0 && a[2] == 0 && a[3] == 0) {
        r[0] = 0; r[1] = 0; r[2] = 0; r[3] = 0; return;
    }
    uint128_t borrow;
    borrow = (uint128_t)SECP_N[0] - a[0]; r[0] = (uint64_t)borrow;
    uint64_t b1 = (uint64_t)(borrow >> 64) & 1;
    borrow = (uint128_t)SECP_N[1] - a[1] - b1; r[1] = (uint64_t)borrow;
    uint64_t b2 = (uint64_t)(borrow >> 64) & 1;
    borrow = (uint128_t)SECP_N[2] - a[2] - b2; r[2] = (uint64_t)borrow;
    uint64_t b3 = (uint64_t)(borrow >> 64) & 1;
    r[3] = SECP_N[3] - a[3] - b3;
}

static inline void secp_fr_from_mont(const uint64_t a[4], uint64_t r[4]) {
    const uint64_t one[4] = {1, 0, 0, 0};
    secp_fr_mul(a, one, r);
}

static inline void secp_fr_to_mont(const uint64_t a[4], uint64_t r[4]) {
    secp_fr_mul(a, SECP_FR_R2, r);
}

static void secp_fr_inv(const uint64_t a[4], uint64_t r[4]) {
    uint64_t exp[4] = { SECP_N[0] - 2, SECP_N[1], SECP_N[2], SECP_N[3] };
    uint64_t result[4], base[4];
    memcpy(result, SECP_FR_ONE, 32);
    memcpy(base, a, 32);
    for (int i = 0; i < 4; i++) {
        uint64_t word = exp[i];
        for (int bit = 0; bit < 64; bit++) {
            if (word & 1) { uint64_t t[4]; secp_fr_mul(result, base, t); memcpy(result, t, 32); }
            uint64_t t2[4]; secp_fr_mul(base, base, t2); memcpy(base, t2, 32);
            word >>= 1;
        }
    }
    memcpy(r, result, 32);
}

static void secp_fr_batch_inv(const uint64_t *a, int n, uint64_t *out) {
    if (n == 0) return;
    uint64_t *prods = (uint64_t *)malloc(n * 32);
    memcpy(prods, a, 32);
    for (int i = 1; i < n; i++)
        secp_fr_mul(prods + (i-1)*4, a + i*4, prods + i*4);
    uint64_t inv[4];
    secp_fr_inv(prods + (n-1)*4, inv);
    for (int i = n-1; i >= 1; i--) {
        secp_fr_mul(inv, prods + (i-1)*4, out + i*4);
        uint64_t tmp[4]; secp_fr_mul(inv, a + i*4, tmp); memcpy(inv, tmp, 32);
    }
    memcpy(out, inv, 32);
    free(prods);
}

// R^2 mod p for secp256k1 Fp
static const uint64_t SECP_FP_R2[4] = {
    0x000007a2000e90a1ULL, 0x0000000000000001ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL
};

int secp256k1_ecdsa_batch_prepare(
    const uint64_t *sigs, const uint64_t *pubkeys, const uint8_t *recov,
    int n, uint64_t *out_points, uint32_t *out_scalars)
{
    if (n == 0) return 0;

    uint64_t *s_vals = (uint64_t *)malloc(n * 32);
    for (int i = 0; i < n; i++)
        memcpy(s_vals + i*4, sigs + i*12 + 4, 32);
    uint64_t *s_invs = (uint64_t *)malloc(n * 32);
    secp_fr_batch_inv(s_vals, n, s_invs);
    free(s_vals);

    uint64_t rng = 0xDEADBEEF12345678ULL;
    rng ^= (uint64_t)(uintptr_t)&rng;
    uint64_t *weights = (uint64_t *)malloc(n * 32);
    for (int i = 0; i < n; i++) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        uint64_t w0 = rng;
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        uint64_t w1 = rng;
        uint64_t raw[4] = {w0, w1, 0, 0};
        secp_fr_to_mont(raw, weights + i*4);
    }

    uint64_t gScalar[4] = {0, 0, 0, 0};

    static const uint64_t GEN_X_RAW[4] = {
        0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL,
        0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL
    };
    static const uint64_t GEN_Y_RAW[4] = {
        0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL,
        0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL
    };
    uint64_t genX[4], genY[4];
    secp_fp_mul(GEN_X_RAW, SECP_FP_R2, genX);
    secp_fp_mul(GEN_Y_RAW, SECP_FP_R2, genY);

    uint64_t seven_raw[4] = {7, 0, 0, 0};
    uint64_t seven_mont[4];
    secp_fp_mul(seven_raw, SECP_FP_R2, seven_mont);

    uint64_t exp_sqrt[4];
    {
        uint128_t s = (uint128_t)SECP_P[0] + 1;
        exp_sqrt[0] = (uint64_t)s;
        s = (uint128_t)SECP_P[1] + (s >> 64);
        exp_sqrt[1] = (uint64_t)s;
        s = (uint128_t)SECP_P[2] + (s >> 64);
        exp_sqrt[2] = (uint64_t)s;
        s = (uint128_t)SECP_P[3] + (s >> 64);
        exp_sqrt[3] = (uint64_t)s;
        for (int j = 0; j < 3; j++)
            exp_sqrt[j] = (exp_sqrt[j] >> 2) | (exp_sqrt[j+1] << 62);
        exp_sqrt[3] >>= 2;
    }

    int outIdx = 0;
    for (int i = 0; i < n; i++) {
        const uint64_t *ri = sigs + i*12;
        const uint64_t *zi = sigs + i*12 + 8;
        const uint64_t *sInv = s_invs + i*4;
        const uint64_t *wi = weights + i*4;

        uint64_t u1[4], u2[4];
        secp_fr_mul(zi, sInv, u1);
        secp_fr_mul(ri, sInv, u2);

        uint64_t wu1[4], tmp[4];
        secp_fr_mul(wi, u1, wu1);
        secp_fr_add(gScalar, wu1, tmp);
        memcpy(gScalar, tmp, 32);

        memcpy(out_points + outIdx*8, pubkeys + i*8, 64);
        uint64_t wu2[4], wu2_int[4];
        secp_fr_mul(wi, u2, wu2);
        secp_fr_from_mont(wu2, wu2_int);
        for (int j = 0; j < 4; j++) {
            out_scalars[outIdx*8 + j*2] = (uint32_t)(wu2_int[j] & 0xFFFFFFFF);
            out_scalars[outIdx*8 + j*2 + 1] = (uint32_t)(wu2_int[j] >> 32);
        }
        outIdx++;

        uint64_t r_int[4];
        secp_fr_from_mont(ri, r_int);
        uint64_t x_fp[4];
        secp_fp_mul(r_int, SECP_FP_R2, x_fp);

        uint64_t x2[4], x3[4], rhs[4];
        secp_fp_sqr(x_fp, x2);
        secp_fp_mul(x2, x_fp, x3);
        secp_fp_add(x3, seven_mont, rhs);

        uint64_t y_fp[4], base_s[4];
        memcpy(y_fp, SECP_ONE, 32);
        memcpy(base_s, rhs, 32);
        for (int j = 0; j < 4; j++) {
            uint64_t word = exp_sqrt[j];
            for (int bit = 0; bit < 64; bit++) {
                if (word & 1) { uint64_t t[4]; secp_fp_mul(y_fp, base_s, t); memcpy(y_fp, t, 32); }
                uint64_t t2[4]; secp_fp_sqr(base_s, t2); memcpy(base_s, t2, 32);
                word >>= 1;
            }
        }

        uint64_t check[4];
        secp_fp_sqr(y_fp, check);
        if (check[0] != rhs[0] || check[1] != rhs[1] ||
            check[2] != rhs[2] || check[3] != rhs[3]) {
            free(s_invs); free(weights); return -1;
        }

        uint64_t y_int[4];
        const uint64_t one_raw[4] = {1, 0, 0, 0};
        secp_fp_mul(y_fp, one_raw, y_int);
        uint8_t yParity = (uint8_t)(y_int[0] & 1);
        uint8_t wantParity = recov ? recov[i] : 0;
        if (yParity != wantParity) {
            uint64_t zero[4] = {0, 0, 0, 0};
            secp_fp_sub(zero, y_fp, y_fp);
        }

        memcpy(out_points + outIdx*8, x_fp, 32);
        memcpy(out_points + outIdx*8 + 4, y_fp, 32);

        uint64_t negW[4], negW_int[4];
        secp_fr_neg(wi, negW);
        secp_fr_from_mont(negW, negW_int);
        for (int j = 0; j < 4; j++) {
            out_scalars[outIdx*8 + j*2] = (uint32_t)(negW_int[j] & 0xFFFFFFFF);
            out_scalars[outIdx*8 + j*2 + 1] = (uint32_t)(negW_int[j] >> 32);
        }
        outIdx++;
    }

    memcpy(out_points + outIdx*8, genX, 32);
    memcpy(out_points + outIdx*8 + 4, genY, 32);
    uint64_t gScalar_int[4];
    secp_fr_from_mont(gScalar, gScalar_int);
    for (int j = 0; j < 4; j++) {
        out_scalars[outIdx*8 + j*2] = (uint32_t)(gScalar_int[j] & 0xFFFFFFFF);
        out_scalars[outIdx*8 + j*2 + 1] = (uint32_t)(gScalar_int[j] >> 32);
    }

    free(s_invs); free(weights);
    return 0;
}

void secp256k1_fr_inverse(const uint64_t a[4], uint64_t r[4]) { secp_fr_inv(a, r); }
void secp256k1_fr_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) { secp_fr_mul(a, b, r); }
void secp256k1_fr_batch_inverse(const uint64_t *a, int n, uint64_t *out) { secp_fr_batch_inv(a, n, out); }
