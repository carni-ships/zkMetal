// BN254 G1 Pippenger MSM — optimized C implementation
// Features:
//   1. CIOS Montgomery field ops (4×64-bit limbs, __uint128_t)
//   2. Mixed affine addition (projective + affine, saves 4 muls/add)
//   3. Batch-to-affine via Montgomery's trick (1 inversion per window)
//   4. Multi-threaded windows (GCD dispatch_apply)
//   5. Adaptive window sizing
//
// Interop: Swift PointAffine = (Fp x, Fp y) where Fp = 8×UInt32
//          In 64-bit view: 4×uint64_t per Fp, 8×uint64_t per PointAffine
//          Scalars: 8×uint32_t per scalar (little-endian limbs)

#include "NeonFieldOps.h"
#include <stdlib.h>
#include <string.h>
#include <dispatch/dispatch.h>

typedef unsigned __int128 uint128_t;

// ============================================================
// BN254 Fp (base field) constants
// ============================================================

static const uint64_t FP_P[4] = {
    0x3c208c16d87cfd47ULL, 0x97816a916871ca8dULL,
    0xb85045b68181585dULL, 0x30644e72e131a029ULL
};
static const uint64_t FP_INV = 0x87d20782e4866389ULL;  // -p^{-1} mod 2^64
static const uint64_t FP_ONE[4] = {  // R mod p (Montgomery form of 1)
    0xd35d438dc58f0d9dULL, 0x0a78eb28f5c70b3dULL,
    0x666ea36f7879462cULL, 0x0e0a77c19a07df2fULL
};

// ============================================================
// CIOS Montgomery multiplication (fully unrolled, 4 limbs)
// ============================================================

static inline void fp_mul(const uint64_t a[4], const uint64_t b[4], uint64_t result[4]) {
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
        t4 += c;
        uint64_t m = t0 * FP_INV;
        w = (uint128_t)m * FP_P[0] + t0;
        c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP_P[1] + t1 + c;
        t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP_P[2] + t2 + c;
        t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP_P[3] + t3 + c;
        t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t3 = t4 + c; t4 = 0;
    }
    // Iteration i=1
    {
        uint128_t w; uint64_t c;
        w = (uint128_t)a[1] * b[0] + t0;
        t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[1] * b[1] + t1 + c;
        t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[1] * b[2] + t2 + c;
        t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[1] * b[3] + t3 + c;
        t3 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t4 += c;
        uint64_t m = t0 * FP_INV;
        w = (uint128_t)m * FP_P[0] + t0;
        c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP_P[1] + t1 + c;
        t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP_P[2] + t2 + c;
        t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP_P[3] + t3 + c;
        t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t3 = t4 + c; t4 = 0;
    }
    // Iteration i=2
    {
        uint128_t w; uint64_t c;
        w = (uint128_t)a[2] * b[0] + t0;
        t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[2] * b[1] + t1 + c;
        t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[2] * b[2] + t2 + c;
        t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[2] * b[3] + t3 + c;
        t3 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t4 += c;
        uint64_t m = t0 * FP_INV;
        w = (uint128_t)m * FP_P[0] + t0;
        c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP_P[1] + t1 + c;
        t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP_P[2] + t2 + c;
        t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP_P[3] + t3 + c;
        t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t3 = t4 + c; t4 = 0;
    }
    // Iteration i=3
    {
        uint128_t w; uint64_t c;
        w = (uint128_t)a[3] * b[0] + t0;
        t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[3] * b[1] + t1 + c;
        t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[3] * b[2] + t2 + c;
        t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[3] * b[3] + t3 + c;
        t3 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t4 += c;
        uint64_t m = t0 * FP_INV;
        w = (uint128_t)m * FP_P[0] + t0;
        c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP_P[1] + t1 + c;
        t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP_P[2] + t2 + c;
        t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP_P[3] + t3 + c;
        t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t3 = t4 + c; t4 = 0;
    }

    // Final conditional subtraction
    uint64_t borrow = 0;
    uint64_t r0, r1, r2, r3;
    uint128_t d;
    d = (uint128_t)t0 - FP_P[0] - borrow;
    r0 = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)t1 - FP_P[1] - borrow;
    r1 = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)t2 - FP_P[2] - borrow;
    r2 = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)t3 - FP_P[3] - borrow;
    r3 = (uint64_t)d; borrow = (d >> 127) & 1;

    if (!borrow) {
        result[0] = r0; result[1] = r1;
        result[2] = r2; result[3] = r3;
    } else {
        result[0] = t0; result[1] = t1;
        result[2] = t2; result[3] = t3;
    }
}

static inline void fp_sqr(const uint64_t a[4], uint64_t r[4]) {
    // Optimized squaring: compute full 8-limb a^2 via upper-triangle
    // doubling + diagonal (10 muls vs 16), then Montgomery reduce.
    uint128_t w;
    uint64_t c;
    uint64_t s0, s1, s2, s3, s4, s5, s6, s7;

    // Step 1: Upper triangle products a[i]*a[j] for j>i
    // Row 0: a[0]*a[1..3]
    w = (uint128_t)a[0] * a[1];
    s1 = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[0] * a[2] + c;
    s2 = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[0] * a[3] + c;
    s3 = (uint64_t)w; s4 = (uint64_t)(w >> 64);

    // Row 1: a[1]*a[2..3]
    w = (uint128_t)a[1] * a[2] + s3;
    s3 = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[1] * a[3] + s4 + c;
    s4 = (uint64_t)w; s5 = (uint64_t)(w >> 64);

    // Row 2: a[2]*a[3]
    w = (uint128_t)a[2] * a[3] + s5;
    s5 = (uint64_t)w; s6 = (uint64_t)(w >> 64);

    // Step 2: Double the upper triangle (shift left by 1 bit)
    s7 = s6 >> 63;
    s6 = (s6 << 1) | (s5 >> 63);
    s5 = (s5 << 1) | (s4 >> 63);
    s4 = (s4 << 1) | (s3 >> 63);
    s3 = (s3 << 1) | (s2 >> 63);
    s2 = (s2 << 1) | (s1 >> 63);
    s1 = s1 << 1;

    // Step 3: Add diagonal a[i]*a[i]
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

    // Step 4: Montgomery reduction (4 iterations)
    // Iteration 0
    {
        uint64_t m = s0 * FP_INV;
        w = (uint128_t)m * FP_P[0] + s0; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP_P[1] + s1 + c; s0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP_P[2] + s2 + c; s1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP_P[3] + s3 + c; s2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)s4 + c; s3 = (uint64_t)w; c = (uint64_t)(w >> 64);
        s4 = s5 + c; s5 = s6; s6 = s7; s7 = 0;
    }
    // Iteration 1
    {
        uint64_t m = s0 * FP_INV;
        w = (uint128_t)m * FP_P[0] + s0; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP_P[1] + s1 + c; s0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP_P[2] + s2 + c; s1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP_P[3] + s3 + c; s2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)s4 + c; s3 = (uint64_t)w; c = (uint64_t)(w >> 64);
        s4 = s5 + c; s5 = s6; s6 = 0;
    }
    // Iteration 2
    {
        uint64_t m = s0 * FP_INV;
        w = (uint128_t)m * FP_P[0] + s0; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP_P[1] + s1 + c; s0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP_P[2] + s2 + c; s1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP_P[3] + s3 + c; s2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)s4 + c; s3 = (uint64_t)w; c = (uint64_t)(w >> 64);
        s4 = s5 + c; s5 = 0;
    }
    // Iteration 3
    {
        uint64_t m = s0 * FP_INV;
        w = (uint128_t)m * FP_P[0] + s0; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP_P[1] + s1 + c; s0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP_P[2] + s2 + c; s1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FP_P[3] + s3 + c; s2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        s3 = s4 + c;
    }

    // Final conditional subtraction
    uint64_t borrow = 0;
    uint64_t r0, r1, r2, r3;
    uint128_t d;
    d = (uint128_t)s0 - FP_P[0] - borrow;
    r0 = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)s1 - FP_P[1] - borrow;
    r1 = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)s2 - FP_P[2] - borrow;
    r2 = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)s3 - FP_P[3] - borrow;
    r3 = (uint64_t)d; borrow = (d >> 127) & 1;

    if (!borrow) {
        r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
    } else {
        r[0] = s0; r[1] = s1; r[2] = s2; r[3] = s3;
    }
}

static inline void fp_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t w;
    uint64_t c = 0;
    w = (uint128_t)a[0] + b[0];
    r[0] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[1] + b[1] + c;
    r[1] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[2] + b[2] + c;
    r[2] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[3] + b[3] + c;
    r[3] = (uint64_t)w; c = (uint64_t)(w >> 64);

    uint64_t borrow = 0;
    uint64_t r0, r1, r2, r3;
    uint128_t d;
    d = (uint128_t)r[0] - FP_P[0];
    r0 = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)r[1] - FP_P[1] - borrow;
    r1 = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)r[2] - FP_P[2] - borrow;
    r2 = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)r[3] - FP_P[3] - borrow;
    r3 = (uint64_t)d; borrow = (d >> 127) & 1;

    if (c || !borrow) {
        r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
    }
}

static inline void fp_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t d;
    uint64_t borrow = 0;
    d = (uint128_t)a[0] - b[0];
    r[0] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)a[1] - b[1] - borrow;
    r[1] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)a[2] - b[2] - borrow;
    r[2] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)a[3] - b[3] - borrow;
    r[3] = (uint64_t)d; borrow = (d >> 127) & 1;

    if (borrow) {
        uint64_t c = 0;
        uint128_t w;
        w = (uint128_t)r[0] + FP_P[0];
        r[0] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)r[1] + FP_P[1] + c;
        r[1] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)r[2] + FP_P[2] + c;
        r[2] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)r[3] + FP_P[3] + c;
        r[3] = (uint64_t)w;
    }
}

static inline void fp_dbl(const uint64_t a[4], uint64_t r[4]) {
    fp_add(a, a, r);
}

static inline int fp_is_zero(const uint64_t a[4]) {
    return (a[0] | a[1] | a[2] | a[3]) == 0;
}

static inline void fp_copy(uint64_t dst[4], const uint64_t src[4]) {
    dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = src[3];
}

// Fp inversion via Fermat: a^(p-2) mod p
static void fp_inv(const uint64_t a[4], uint64_t result[4]) {
    uint64_t pm2[4];
    pm2[0] = FP_P[0] - 2;
    pm2[1] = FP_P[1];
    pm2[2] = FP_P[2];
    pm2[3] = FP_P[3];

    memcpy(result, FP_ONE, 32);
    uint64_t b[4];
    memcpy(b, a, 32);
    for (int i = 0; i < 4; i++) {
        for (int bit = 0; bit < 64; bit++) {
            if ((pm2[i] >> bit) & 1)
                fp_mul(result, b, result);
            fp_sqr(b, b);
        }
    }
}

// ============================================================
// Jacobian projective point operations (y² = x³ + 3)
// Layout: [x0..x3, y0..y3, z0..z3] = 12 uint64_t = 96 bytes
// Identity: Z = 0
// ============================================================

static inline void pt_set_id(uint64_t p[12]) {
    memcpy(p, FP_ONE, 32);       // x = 1
    memcpy(p + 4, FP_ONE, 32);   // y = 1
    memset(p + 8, 0, 32);        // z = 0
}

static inline int pt_is_id(const uint64_t p[12]) {
    return fp_is_zero(p + 8);
}

// Point doubling (a = 0 curve)
static void pt_dbl(const uint64_t p[12], uint64_t r[12]) {
    if (pt_is_id(p)) { memcpy(r, p, 96); return; }

    const uint64_t *px = p, *py = p + 4, *pz = p + 8;
    uint64_t a[4], b[4], c[4], d[4], e[4], f[4], t1[4], t2[4];

    fp_sqr(px, a);             // a = x²
    fp_sqr(py, b);             // b = y²
    fp_sqr(b, c);              // c = y⁴

    // d = 2((x+b)² - a - c)
    fp_add(px, b, t1);
    fp_sqr(t1, t1);
    fp_sub(t1, a, t1);
    fp_sub(t1, c, t1);
    fp_dbl(t1, d);

    // e = 3a
    fp_dbl(a, t1);
    fp_add(t1, a, e);

    fp_sqr(e, f);              // f = e²

    // x3 = f - 2d
    fp_dbl(d, t1);
    fp_sub(f, t1, r);

    // y3 = e(d - x3) - 8c
    fp_sub(d, r, t1);
    fp_mul(e, t1, t2);
    fp_dbl(c, t1);
    fp_dbl(t1, t1);
    fp_dbl(t1, t1);            // 8c
    fp_sub(t2, t1, r + 4);

    // z3 = (y+z)² - b - z²
    fp_add(py, pz, t1);
    fp_sqr(t1, t1);
    fp_sub(t1, b, t1);
    fp_sqr(pz, t2);
    fp_sub(t1, t2, r + 8);
}

// Full projective addition
static void pt_add(const uint64_t p[12], const uint64_t q[12], uint64_t r[12]) {
    if (pt_is_id(p)) { memcpy(r, q, 96); return; }
    if (pt_is_id(q)) { memcpy(r, p, 96); return; }

    const uint64_t *px = p, *py = p+4, *pz = p+8;
    const uint64_t *qx = q, *qy = q+4, *qz = q+8;

    uint64_t z1z1[4], z2z2[4], u1[4], u2[4], s1[4], s2[4];
    uint64_t h[4], rr[4], ii[4], j[4], vv[4], t1[4];

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
        if (fp_is_zero(rr)) { pt_dbl(p, r); return; }
        pt_set_id(r); return;
    }

    fp_dbl(h, t1);
    fp_sqr(t1, ii);
    fp_mul(h, ii, j);
    fp_mul(u1, ii, vv);

    // x3 = r² - j - 2v
    fp_sqr(rr, r);
    fp_sub(r, j, r);
    fp_dbl(vv, t1);
    fp_sub(r, t1, r);

    // y3 = r(v - x3) - 2*s1*j
    fp_sub(vv, r, t1);
    fp_mul(rr, t1, r + 4);
    fp_mul(s1, j, t1);
    fp_dbl(t1, t1);
    fp_sub(r + 4, t1, r + 4);

    // z3 = ((z1+z2)² - z1z1 - z2z2) * h
    fp_add(pz, qz, t1);
    fp_sqr(t1, t1);
    fp_sub(t1, z1z1, t1);
    fp_sub(t1, z2z2, t1);
    fp_mul(t1, h, r + 8);
}

// Check if an affine point is identity (represented as (0,0))
static inline int aff_is_id(const uint64_t q[8]) {
    return (q[0] | q[1] | q[2] | q[3] | q[4] | q[5] | q[6] | q[7]) == 0;
}

// Mixed addition: projective P + affine Q (Z_Q = 1)
// Saves 2 muls + 1 sqr vs full projective add
static void pt_add_mixed(const uint64_t p[12], const uint64_t q_aff[8], uint64_t r[12]) {
    if (aff_is_id(q_aff)) {
        memcpy(r, p, 96);
        return;
    }
    if (pt_is_id(p)) {
        memcpy(r, q_aff, 64);        // x, y from affine
        memcpy(r + 8, FP_ONE, 32);   // z = 1
        return;
    }

    const uint64_t *px = p, *py = p+4, *pz = p+8;
    const uint64_t *qx = q_aff, *qy = q_aff + 4;

    uint64_t z1z1[4], u2[4], s2[4], h[4], rr[4];
    uint64_t ii[4], j[4], vv[4], t1[4], hh[4];

    fp_sqr(pz, z1z1);
    fp_mul(qx, z1z1, u2);         // u2 = qx * z1²
    fp_mul(pz, z1z1, t1);
    fp_mul(qy, t1, s2);           // s2 = qy * z1³

    fp_sub(u2, px, h);            // h = u2 - px (u1 = px)
    fp_sub(s2, py, t1);
    fp_dbl(t1, rr);               // r = 2(s2 - s1)

    if (fp_is_zero(h)) {
        if (fp_is_zero(rr)) { pt_dbl(p, r); return; }
        pt_set_id(r); return;
    }

    fp_dbl(h, t1);
    fp_sqr(t1, ii);
    fp_mul(h, ii, j);
    fp_mul(px, ii, vv);

    // x3 = r² - j - 2v
    fp_sqr(rr, r);
    fp_sub(r, j, r);
    fp_dbl(vv, t1);
    fp_sub(r, t1, r);

    // y3 = r(v - x3) - 2*s1*j
    fp_sub(vv, r, t1);
    fp_mul(rr, t1, r + 4);
    fp_mul(py, j, t1);
    fp_dbl(t1, t1);
    fp_sub(r + 4, t1, r + 4);

    // z3 = (z1 + h)² - z1z1 - h²
    fp_add(pz, h, t1);
    fp_sqr(t1, t1);
    fp_sub(t1, z1z1, t1);
    fp_sqr(h, hh);
    fp_sub(t1, hh, r + 8);
}

// ============================================================
// Batch projective-to-affine (Montgomery's trick)
// Single inversion for all non-identity points
// ============================================================

// Convert n projective points to affine in-place.
// proj: array of n×12 uint64_t (projective)
// aff: output array of n×8 uint64_t (affine)
// Identity points get (0,0) in affine output.
static void batch_to_affine(const uint64_t *proj, uint64_t *aff, int n) {
    if (n == 0) return;

    // Compute cumulative products of Z values
    uint64_t *prods = (uint64_t *)malloc((size_t)n * 32);
    int first_valid = -1;

    for (int i = 0; i < n; i++) {
        if (pt_is_id(proj + i * 12)) {
            if (i == 0) memcpy(prods, FP_ONE, 32);
            else memcpy(prods + i * 4, prods + (i-1) * 4, 32);
        } else {
            if (first_valid < 0) {
                first_valid = i;
                memcpy(prods + i * 4, proj + i * 12 + 8, 32);  // Z_i
            } else {
                fp_mul(prods + (i-1) * 4, proj + i * 12 + 8, prods + i * 4);
            }
        }
    }

    if (first_valid < 0) {
        // All identity
        memset(aff, 0, (size_t)n * 64);
        free(prods);
        return;
    }

    // Invert the product
    uint64_t inv[4];
    fp_inv(prods + (n-1) * 4, inv);

    // Back-propagate
    for (int i = n - 1; i >= 0; i--) {
        if (pt_is_id(proj + i * 12)) {
            memset(aff + i * 8, 0, 64);
            continue;
        }

        uint64_t zinv[4];
        if (i > first_valid) {
            fp_mul(inv, prods + (i-1) * 4, zinv);
            fp_mul(inv, proj + i * 12 + 8, inv);  // inv *= z_i
        } else {
            fp_copy(zinv, inv);
        }

        uint64_t zinv2[4], zinv3[4];
        fp_sqr(zinv, zinv2);
        fp_mul(zinv2, zinv, zinv3);
        fp_mul(proj + i * 12, zinv2, aff + i * 8);       // x_aff = X * Z^{-2}
        fp_mul(proj + i * 12 + 4, zinv3, aff + i * 8 + 4); // y_aff = Y * Z^{-3}
    }

    free(prods);
}

// ============================================================
// Scalar window extraction
// ============================================================

static inline uint32_t extract_window(const uint32_t *scalar, int window_idx, int window_bits) {
    int bit_offset = window_idx * window_bits;
    int word_idx = bit_offset / 32;
    int bit_in_word = bit_offset % 32;

    uint64_t word = scalar[word_idx];
    if (word_idx + 1 < 8)
        word |= ((uint64_t)scalar[word_idx + 1]) << 32;

    return (uint32_t)((word >> bit_in_word) & ((1u << window_bits) - 1));
}

// ============================================================
// Adaptive window sizing
// ============================================================

static int optimal_window_bits(int n) {
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

// ============================================================
// Per-window worker
// ============================================================

typedef struct {
    const uint64_t *points;     // n affine points (8 limbs each)
    const uint32_t *scalars;    // n scalars (8 uint32 each)
    int n;
    int window_bits;
    int window_idx;
    int num_buckets;
    uint64_t result[12];        // output projective point
} WindowTask;

static void *window_worker(void *arg) {
    WindowTask *task = (WindowTask *)arg;
    int wb = task->window_bits;
    int w = task->window_idx;
    int nb = task->num_buckets;
    int nn = task->n;

    // Allocate projective buckets (identity-initialized)
    uint64_t *buckets = (uint64_t *)malloc((size_t)(nb + 1) * 96);
    for (int b = 0; b <= nb; b++)
        pt_set_id(buckets + b * 12);

    // Phase 1: Bucket accumulation (mixed affine addition)
    for (int i = 0; i < nn; i++) {
        uint32_t digit = extract_window(task->scalars + i * 8, w, wb);
        if (digit != 0) {
            uint64_t tmp[12];
            pt_add_mixed(buckets + digit * 12, task->points + i * 8, tmp);
            memcpy(buckets + digit * 12, tmp, 96);
        }
    }

    // Phase 2: Batch convert buckets to affine (Montgomery's trick)
    // Only convert buckets 1..nb (skip bucket 0)
    uint64_t *bucket_aff = (uint64_t *)malloc((size_t)nb * 64);
    batch_to_affine(buckets + 12, bucket_aff, nb);

    // Phase 3: Running-sum reduction using mixed addition
    uint64_t running[12], window_sum[12];
    pt_set_id(running);
    pt_set_id(window_sum);

    for (int j = nb - 1; j >= 0; j--) {
        // bucket_aff[j] corresponds to original bucket j+1
        if (!(bucket_aff[j*8] == 0 && bucket_aff[j*8+1] == 0 &&
              bucket_aff[j*8+2] == 0 && bucket_aff[j*8+3] == 0 &&
              bucket_aff[j*8+4] == 0 && bucket_aff[j*8+5] == 0 &&
              bucket_aff[j*8+6] == 0 && bucket_aff[j*8+7] == 0)) {
            uint64_t tmp[12];
            pt_add_mixed(running, bucket_aff + j * 8, tmp);
            memcpy(running, tmp, 96);
        }
        uint64_t tmp[12];
        pt_add(window_sum, running, tmp);
        memcpy(window_sum, tmp, 96);
    }

    memcpy(task->result, window_sum, 96);
    free(buckets);
    free(bucket_aff);
    return NULL;
}

// ============================================================
// BN254 Fr (scalar field) constants — for inner product / fold
// ============================================================

static const uint64_t FR_P[4] = {
    0x43e1f593f0000001ULL, 0x2833e84879b97091ULL,
    0xb85045b68181585dULL, 0x30644e72e131a029ULL
};
static const uint64_t FR_INV = 0xc2e1f593efffffffULL;
static const uint64_t FR_ONE[4] = {  // R mod r (Montgomery form of 1)
    0xac96341c4ffffffbULL, 0x36fc76959f60cd29ULL,
    0x666ea36f7879462eULL, 0x0e0a77c19a07df2fULL
};

static inline void fr_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    // Reuse fp_mul structure with Fr constants
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
    // Optimized squaring for Fr: upper-triangle doubled + diagonal (10 muls vs 16)
    uint128_t w;
    uint64_t c;
    uint64_t s0, s1, s2, s3, s4, s5, s6, s7;

    // Upper triangle: a[i]*a[j] for j>i
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

    // Double
    s7 = s6 >> 63;
    s6 = (s6 << 1) | (s5 >> 63);
    s5 = (s5 << 1) | (s4 >> 63);
    s4 = (s4 << 1) | (s3 >> 63);
    s3 = (s3 << 1) | (s2 >> 63);
    s2 = (s2 << 1) | (s1 >> 63);
    s1 = s1 << 1;

    // Add diagonal
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

static void fr_inv(const uint64_t a[4], uint64_t result[4]) {
    uint64_t pm2[4];
    pm2[0] = FR_P[0] - 2; pm2[1] = FR_P[1]; pm2[2] = FR_P[2]; pm2[3] = FR_P[3];
    memcpy(result, FR_ONE, 32);
    uint64_t b[4]; memcpy(b, a, 32);
    for (int i = 0; i < 4; i++) {
        for (int bit = 0; bit < 64; bit++) {
            if ((pm2[i] >> bit) & 1)
                fr_mul(result, b, result);
            fr_sqr(b, b);
        }
    }
}

// ============================================================
// Exported C functions for IPA / verification hot paths
// ============================================================

// Point scalar multiplication (exported)
// p: projective point (12 uint64), scalar: 8 uint32 (non-Montgomery integer)
void bn254_point_scalar_mul(const uint64_t p[12], const uint32_t scalar[8], uint64_t r[12]);

// Fr inner product: result = sum(a[i] * b[i]) for i=0..n-1
// a, b: arrays of n Fr elements (4 uint64 each, Montgomery form)
// result: single Fr element (4 uint64)
void bn254_fr_inner_product(const uint64_t *a, const uint64_t *b, int n, uint64_t result[4]);

// Fr vector fold: out[i] = a[i]*x + b[i]*xInv for i=0..n-1
// a, b: Fr arrays (4 limbs each), x, xInv: Fr elements
void bn254_fr_vector_fold(const uint64_t *a, const uint64_t *b,
                           const uint64_t x[4], const uint64_t xInv[4],
                           int n, uint64_t *out);

// ============================================================
// Scalar multiplication: P * scalar (double-and-add)
// scalar: 8 × uint32_t limbs (little-endian, non-Montgomery integer)
// ============================================================

static void pt_scalar_mul(const uint64_t p[12], const uint32_t scalar[8], uint64_t r[12]) {
    // Windowed scalar multiplication, w=4 (process 4 bits at a time)
    // Precompute table[0..15] = [identity, P, 2P, ..., 15P]
    // Then scan scalar from MSB: for each nibble, double 4 times then add table[nibble]
    // Cost: 15 adds (precomp) + 252 doubles + ~64 adds ≈ 331 ops vs 256 doubles + ~128 adds = 384 ops

    uint64_t table[16 * 12];  // 16 projective points

    // table[0] = identity
    pt_set_id(table);
    // table[1] = P
    memcpy(table + 12, p, 96);
    // table[i] = table[i-1] + P for i = 2..15
    for (int i = 2; i < 16; i++) {
        pt_add(table + (i - 1) * 12, p, table + i * 12);
    }

    // Extract nibbles from scalar (MSB first): 256 bits = 64 nibbles
    // scalar is 8 x uint32_t, little-endian limbs
    // Nibble 63 is the highest 4 bits of scalar[7], nibble 0 is lowest 4 bits of scalar[0]
    uint8_t nibbles[64];
    for (int i = 0; i < 8; i++) {
        uint32_t word = scalar[i];
        for (int j = 0; j < 8; j++) {
            nibbles[i * 8 + j] = (uint8_t)(word & 0xF);
            word >>= 4;
        }
    }

    // Find highest non-zero nibble
    int top = 63;
    while (top >= 0 && nibbles[top] == 0) top--;

    if (top < 0) {
        pt_set_id(r);
        return;
    }

    // Start with table[top nibble]
    uint64_t result[12];
    memcpy(result, table + nibbles[top] * 12, 96);

    // Process remaining nibbles from MSB to LSB
    for (int i = top - 1; i >= 0; i--) {
        // Double 4 times
        uint64_t tmp[12];
        pt_dbl(result, tmp); memcpy(result, tmp, 96);
        pt_dbl(result, tmp); memcpy(result, tmp, 96);
        pt_dbl(result, tmp); memcpy(result, tmp, 96);
        pt_dbl(result, tmp); memcpy(result, tmp, 96);
        // Add table[nibble] if non-zero
        if (nibbles[i]) {
            pt_add(result, table + nibbles[i] * 12, tmp);
            memcpy(result, tmp, 96);
        }
    }

    memcpy(r, result, 96);
}

// Batch fold generators: result[i] = scalar_mul(GL[i], xInv) + scalar_mul(GR[i], x)
// GL, GR: projective points (12 uint64 each)
// x, xInv: 8 × uint32_t scalars (non-Montgomery, integer form)
// Uses GCD dispatch_apply for parallelism.
typedef struct {
    const uint64_t *GL;
    const uint64_t *GR;
    const uint32_t *x;
    const uint32_t *xInv;
    uint64_t *result;
    int start;
    int end;
} FoldChunk;

// Straus double-scalar multiplication: computes s1*P1 + s2*P2 with shared doublings.
// Uses w=4 windowed method on both scalars simultaneously.
// Cost: ~252 doublings + ~64 adds (vs ~504 doublings + ~128 adds for two separate scalar muls)
static void pt_double_scalar_mul(
    const uint64_t P1[12], const uint32_t s1[8],
    const uint64_t P2[12], const uint32_t s2[8],
    uint64_t r[12])
{
    // Precompute tables: table1[i] = i*P1, table2[i] = i*P2 for i in 0..15
    uint64_t table1[16 * 12], table2[16 * 12];

    pt_set_id(table1);
    memcpy(table1 + 12, P1, 96);
    for (int i = 2; i < 16; i++)
        pt_add(table1 + (i - 1) * 12, P1, table1 + i * 12);

    pt_set_id(table2);
    memcpy(table2 + 12, P2, 96);
    for (int i = 2; i < 16; i++)
        pt_add(table2 + (i - 1) * 12, P2, table2 + i * 12);

    // Extract nibbles for both scalars
    uint8_t nib1[64], nib2[64];
    for (int i = 0; i < 8; i++) {
        uint32_t w1 = s1[i], w2 = s2[i];
        for (int j = 0; j < 8; j++) {
            nib1[i * 8 + j] = (uint8_t)(w1 & 0xF); w1 >>= 4;
            nib2[i * 8 + j] = (uint8_t)(w2 & 0xF); w2 >>= 4;
        }
    }

    // Find highest non-zero nibble across both scalars
    int top = 63;
    while (top >= 0 && nib1[top] == 0 && nib2[top] == 0) top--;

    if (top < 0) { pt_set_id(r); return; }

    // Initialize with table lookups at top nibble
    uint64_t result[12];
    pt_set_id(result);
    if (nib1[top]) {
        memcpy(result, table1 + nib1[top] * 12, 96);
    }
    if (nib2[top]) {
        uint64_t tmp[12];
        pt_add(result, table2 + nib2[top] * 12, tmp);
        memcpy(result, tmp, 96);
    }

    // Process remaining nibbles
    for (int i = top - 1; i >= 0; i--) {
        uint64_t tmp[12];
        pt_dbl(result, tmp); memcpy(result, tmp, 96);
        pt_dbl(result, tmp); memcpy(result, tmp, 96);
        pt_dbl(result, tmp); memcpy(result, tmp, 96);
        pt_dbl(result, tmp); memcpy(result, tmp, 96);
        if (nib1[i]) {
            pt_add(result, table1 + nib1[i] * 12, tmp);
            memcpy(result, tmp, 96);
        }
        if (nib2[i]) {
            pt_add(result, table2 + nib2[i] * 12, tmp);
            memcpy(result, tmp, 96);
        }
    }

    memcpy(r, result, 96);
}

void bn254_fold_generators(
    const uint64_t *GL,         // halfLen projective points (12 uint64 each)
    const uint64_t *GR,         // halfLen projective points (12 uint64 each)
    const uint32_t *x,          // scalar (8 × uint32_t)
    const uint32_t *xInv,       // scalar (8 × uint32_t)
    int halfLen,
    uint64_t *result)           // halfLen projective points output
{
    if (halfLen <= 0) return;

    int nChunks = 8;
    if (halfLen < nChunks) nChunks = halfLen;
    int chunkSize = (halfLen + nChunks - 1) / nChunks;
    int total = halfLen;

    dispatch_apply(nChunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
        ^(size_t idx) {
            int start = (int)idx * chunkSize;
            int end = start + chunkSize;
            if (end > total) end = total;
            for (int i = start; i < end; i++) {
                pt_double_scalar_mul(
                    GL + i * 12, xInv,
                    GR + i * 12, x,
                    result + i * 12);
            }
        });
}

// ============================================================
// CPU single-window reduce for cooperative GPU/CPU MSM
// ============================================================

void bn254_cpu_window_reduce(
    const uint64_t *points,
    const uint32_t *sorted_indices,
    const uint32_t *offsets,
    const uint32_t *counts,
    int n_buckets,
    uint64_t *result)
{
    uint64_t *buckets = (uint64_t *)malloc((size_t)n_buckets * 96);
    for (int b = 0; b < n_buckets; b++)
        pt_set_id(buckets + b * 12);

    static const uint64_t FP_ZERO_LOCAL[4] = {0, 0, 0, 0};

    for (int b = 1; b < n_buckets; b++) {
        uint32_t off = offsets[b];
        uint32_t cnt = counts[b];
        for (uint32_t j = 0; j < cnt; j++) {
            uint32_t raw_idx = sorted_indices[off + j];
            uint32_t pt_idx = raw_idx & 0x7FFFFFFF;
            int negate = (raw_idx >> 31) & 1;
            if (negate) {
                uint64_t neg_aff[8];
                memcpy(neg_aff, points + pt_idx * 8, 32);
                fp_sub(FP_ZERO_LOCAL, points + pt_idx * 8 + 4, neg_aff + 4);
                uint64_t tmp[12];
                pt_add_mixed(buckets + b * 12, neg_aff, tmp);
                memcpy(buckets + b * 12, tmp, 96);
            } else {
                uint64_t tmp[12];
                pt_add_mixed(buckets + b * 12, points + pt_idx * 8, tmp);
                memcpy(buckets + b * 12, tmp, 96);
            }
        }
    }

    int active = n_buckets - 1;
    uint64_t *bucket_aff = (uint64_t *)malloc((size_t)active * 64);
    batch_to_affine(buckets + 12, bucket_aff, active);

    uint64_t running[12], window_sum[12];
    pt_set_id(running);
    pt_set_id(window_sum);

    for (int j = active - 1; j >= 0; j--) {
        if (!(bucket_aff[j*8] == 0 && bucket_aff[j*8+1] == 0 &&
              bucket_aff[j*8+2] == 0 && bucket_aff[j*8+3] == 0 &&
              bucket_aff[j*8+4] == 0 && bucket_aff[j*8+5] == 0 &&
              bucket_aff[j*8+6] == 0 && bucket_aff[j*8+7] == 0)) {
            uint64_t tmp[12];
            pt_add_mixed(running, bucket_aff + j * 8, tmp);
            memcpy(running, tmp, 96);
        }
        uint64_t tmp[12];
        pt_add(window_sum, running, tmp);
        memcpy(window_sum, tmp, 96);
    }

    memcpy(result, window_sum, 96);
    free(buckets);
    free(bucket_aff);
}

// ============================================================
// Main entry point: Pippenger MSM
// ============================================================

void bn254_pippenger_msm(
    const uint64_t *points,    // n affine points: n × 8 uint64_t
    const uint32_t *scalars,   // n scalars: n × 8 uint32_t
    int n,
    uint64_t *result)          // output: 12 uint64_t (projective)
{
    if (n == 0) { pt_set_id(result); return; }

    int wb = optimal_window_bits(n);
    int num_windows = (256 + wb - 1) / wb;
    int num_buckets = (1 << wb) - 1;

    // Allocate tasks
    WindowTask *tasks = (WindowTask *)malloc((size_t)num_windows * sizeof(WindowTask));

    for (int w = 0; w < num_windows; w++) {
        tasks[w].points = points;
        tasks[w].scalars = scalars;
        tasks[w].n = n;
        tasks[w].window_bits = wb;
        tasks[w].window_idx = w;
        tasks[w].num_buckets = num_buckets;
    }

    dispatch_apply(num_windows, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
        ^(size_t w) {
            window_worker(&tasks[w]);
        });

    // Horner combination: result = Σ windowResults[w] × 2^(w × wb)
    memcpy(result, tasks[num_windows - 1].result, 96);
    for (int w = num_windows - 2; w >= 0; w--) {
        uint64_t tmp[12];
        for (int s = 0; s < wb; s++) {
            pt_dbl(result, tmp);
            memcpy(result, tmp, 96);
        }
        pt_add(result, tasks[w].result, tmp);
        memcpy(result, tmp, 96);
    }

    free(tasks);
}

// ============================================================
// MSM from projective points (small n, direct scalar-mul accumulation)
// Avoids batchToAffine overhead for small IPA rounds.
// ============================================================

typedef struct {
    const uint64_t *points;
    const uint32_t *scalars;
    uint64_t partial[12];  // block-local partial sum
    int start;
    int end;
} MSMProjChunk;

void bn254_msm_projective(
    const uint64_t *points,    // n projective points (12 uint64_t each)
    const uint32_t *scalars,   // n scalars (8 uint32_t each)
    int n,
    uint64_t *result)          // output: 12 uint64_t (projective)
{
    if (n == 0) { pt_set_id(result); return; }

    // Single-threaded for very small n
    if (n <= 4) {
        pt_set_id(result);
        for (int i = 0; i < n; i++) {
            uint64_t term[12], tmp[12];
            pt_scalar_mul(points + i * 12, scalars + i * 8, term);
            pt_add(result, term, tmp);
            memcpy(result, tmp, 96);
        }
        return;
    }

    // Multi-threaded for larger n
    int nChunks = 8;
    if (n < nChunks) nChunks = n;

    MSMProjChunk *chunks = (MSMProjChunk *)malloc(nChunks * sizeof(MSMProjChunk));
    int chunkSize = (n + nChunks - 1) / nChunks;

    for (int t = 0; t < nChunks; t++) {
        chunks[t].points = points;
        chunks[t].scalars = scalars;
        chunks[t].start = t * chunkSize;
        chunks[t].end = (t + 1) * chunkSize;
        if (chunks[t].end > n) chunks[t].end = n;
    }

    dispatch_apply(nChunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
        ^(size_t idx) {
            MSMProjChunk *c = &chunks[idx];
            pt_set_id(c->partial);
            for (int i = c->start; i < c->end; i++) {
                uint64_t term[12], tmp[12];
                pt_scalar_mul(c->points + i * 12, c->scalars + i * 8, term);
                pt_add(c->partial, term, tmp);
                memcpy(c->partial, tmp, 96);
            }
        });

    // Combine partial sums
    memcpy(result, chunks[0].partial, 96);
    for (int t = 1; t < nChunks; t++) {
        uint64_t tmp[12];
        pt_add(result, chunks[t].partial, tmp);
        memcpy(result, tmp, 96);
    }
    free(chunks);
}

// ============================================================
// Dual MSM: compute two projective MSMs with a shared thread pool.
// Eliminates double thread-creation overhead for IPA L/R.
// ============================================================

typedef struct {
    const uint64_t *points;
    const uint32_t *scalars;
    uint64_t partial[12];
    int start, end;
} DualMSMChunk;

void bn254_dual_msm_projective(
    const uint64_t *points1, const uint32_t *scalars1, int n1,
    const uint64_t *points2, const uint32_t *scalars2, int n2,
    uint64_t result1[12], uint64_t result2[12])
{
    if (n1 + n2 <= 4) {
        bn254_msm_projective(points1, scalars1, n1, result1);
        bn254_msm_projective(points2, scalars2, n2, result2);
        return;
    }
    int total = n1 + n2;
    int t1 = (n1 * 8 + total - 1) / total;
    if (t1 < 1) t1 = 1; if (t1 > 7) t1 = 7;
    int t2 = 8 - t1;
    if (n1 < t1) t1 = n1; if (n2 < t2) t2 = n2;
    int nBlocks = t1 + t2;
    DualMSMChunk *chunks = (DualMSMChunk *)malloc(16 * sizeof(DualMSMChunk));

    int cs1 = (n1 + t1 - 1) / t1;
    for (int t = 0; t < t1; t++) {
        chunks[t].points = points1; chunks[t].scalars = scalars1;
        chunks[t].start = t * cs1;
        chunks[t].end = (t+1)*cs1 > n1 ? n1 : (t+1)*cs1;
    }
    int cs2 = (n2 + t2 - 1) / t2;
    for (int t = 0; t < t2; t++) {
        chunks[t1+t].points = points2; chunks[t1+t].scalars = scalars2;
        chunks[t1+t].start = t * cs2;
        chunks[t1+t].end = (t+1)*cs2 > n2 ? n2 : (t+1)*cs2;
    }

    dispatch_apply(nBlocks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
        ^(size_t idx) {
            DualMSMChunk *c = &chunks[idx];
            pt_set_id(c->partial);
            for (int i = c->start; i < c->end; i++) {
                uint64_t term[12], tmp[12];
                pt_scalar_mul(c->points + i * 12, c->scalars + i * 8, term);
                pt_add(c->partial, term, tmp);
                memcpy(c->partial, tmp, 96);
            }
        });

    pt_set_id(result1);
    for (int t = 0; t < t1; t++) {
        uint64_t tmp[12];
        pt_add(result1, chunks[t].partial, tmp);
        memcpy(result1, tmp, 96);
    }
    pt_set_id(result2);
    for (int t = 0; t < t2; t++) {
        uint64_t tmp[12];
        pt_add(result2, chunks[t1+t].partial, tmp);
        memcpy(result2, tmp, 96);
    }
    free(chunks);
}

// ============================================================
// Exported utility functions
// ============================================================

void bn254_point_scalar_mul(const uint64_t p[12], const uint32_t scalar[8], uint64_t r[12]) {
    pt_scalar_mul(p, scalar, r);
}

void bn254_point_add_mixed(const uint64_t p[12], const uint64_t q_aff[8], uint64_t r[12]) {
    pt_add_mixed(p, q_aff, r);
}

void bn254_point_add(const uint64_t p[12], const uint64_t q[12], uint64_t r[12]) {
    pt_add(p, q, r);
}

void bn254_fr_inverse(const uint64_t a[4], uint64_t r[4]) {
    fr_inv(a, r);
}

void bn254_fr_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    fr_mul(a, b, r);
}

void bn254_fr_sqr(const uint64_t a[4], uint64_t r[4]) {
    fr_sqr(a, r);
}

void bn254_fp_sqr(const uint64_t a[4], uint64_t r[4]) {
    fp_sqr(a, r);
}

void bn254_fr_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    fr_add(a, b, r);
}

void bn254_fr_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    fr_sub(a, b, r);
}

void bn254_fr_neg(const uint64_t a[4], uint64_t r[4]) {
    uint64_t zero[4] = {0,0,0,0};
    fr_sub(zero, a, r);
}

void bn254_fr_pow(const uint64_t a[4], uint64_t exp, uint64_t r[4]) {
    memcpy(r, FR_ONE, 32);
    uint64_t base[4]; memcpy(base, a, 32);
    while (exp > 0) {
        if (exp & 1) fr_mul(r, base, r);
        fr_sqr(base, base);
        exp >>= 1;
    }
}

int bn254_fr_eq(const uint64_t a[4], const uint64_t b[4]) {
    return a[0]==b[0] && a[1]==b[1] && a[2]==b[2] && a[3]==b[3];
}

// Helper: negate a projective point (negate Y coordinate in Fp)
static inline void pt_neg(const uint64_t p[12], uint64_t out[12]) {
    memcpy(out, p, 32);       // X unchanged
    static const uint64_t FP_ZERO[4] = {0, 0, 0, 0};
    fp_sub(FP_ZERO, p + 4, out + 4);  // Y = -Y mod P
    memcpy(out + 8, p + 8, 32);  // Z unchanged
}

// Helper: convert Fr (Montgomery) to uint32 scalar for point_scalar_mul
static inline void fr_to_scalar(const uint64_t mont[4], uint32_t scalar[8]) {
    uint64_t raw_one[4] = {1, 0, 0, 0};
    uint64_t integer[4];
    fr_mul(mont, raw_one, integer);
    for (int j = 0; j < 4; j++) {
        scalar[j*2] = (uint32_t)(integer[j]);
        scalar[j*2+1] = (uint32_t)(integer[j] >> 32);
    }
}

// Batch KZG verification using C field and point ops.
// Optimized: 2 scalar muls per tuple + 1 final, instead of 4 per tuple.
// Accumulates: accumC = Σ rho^i * C_i, accumW = Σ rho^i * (s-z_i) * W_i,
//              accumEval = Σ rho^i * eval_i (Fr scalar)
// Final check: accumC + (-accumEval)*G - accumW == identity
int bn254_batch_kzg_verify(const uint64_t *srsG1, const uint64_t *srsSecret,
                           const uint64_t *commitments, const uint64_t *points,
                           const uint64_t *evaluations, const uint64_t *witnesses,
                           const uint64_t *batchChallenge, int n) {
    // srsG1 is affine (8 uint64), convert to projective
    uint64_t g1[12];
    memcpy(g1, srsG1, 64);  // x, y
    memcpy(g1+8, FP_ONE, 32); // z = 1 (in Fp Montgomery form)

    uint64_t accumC[12]; memset(accumC, 0, 96);  // identity
    uint64_t accumW[12]; memset(accumW, 0, 96);  // identity
    uint64_t accumEval[4]; memset(accumEval, 0, 32); // zero Fr

    uint64_t rho[4]; memcpy(rho, FR_ONE, 32); // rho = 1

    for (int i = 0; i < n; i++) {
        const uint64_t *commit = commitments + i * 12;
        const uint64_t *pt = points + i * 4;
        const uint64_t *eval_val = evaluations + i * 4;
        const uint64_t *witness = witnesses + i * 12;

        // rhoC = rho * C_i
        uint32_t rho_scalar[8];
        fr_to_scalar(rho, rho_scalar);
        uint64_t rhoC[12];
        bn254_point_scalar_mul(commit, rho_scalar, rhoC);

        // accumC += rhoC
        uint64_t newAccumC[12];
        pt_add(accumC, rhoC, newAccumC);
        memcpy(accumC, newAccumC, 96);

        // accumEval += rho * eval
        uint64_t rhoEval[4];
        fr_mul(rho, eval_val, rhoEval);
        fr_add(accumEval, rhoEval, accumEval);

        // rho_sz = rho * (s - z_i)
        uint64_t sMinusZ[4];
        fr_sub(srsSecret, pt, sMinusZ);
        uint64_t rho_sz[4];
        fr_mul(rho, sMinusZ, rho_sz);

        // rho_sz_W = rho_sz * W_i
        uint32_t rho_sz_scalar[8];
        fr_to_scalar(rho_sz, rho_sz_scalar);
        uint64_t rho_sz_W[12];
        bn254_point_scalar_mul(witness, rho_sz_scalar, rho_sz_W);

        // accumW += rho_sz_W
        uint64_t newAccumW[12];
        pt_add(accumW, rho_sz_W, newAccumW);
        memcpy(accumW, newAccumW, 96);

        // rho *= batchChallenge
        uint64_t newRho[4];
        fr_mul(rho, batchChallenge, newRho);
        memcpy(rho, newRho, 32);
    }

    // Final: check accumC - accumEval*G - accumW == identity
    // Compute neg_accumEval = -accumEval
    uint64_t neg_accumEval[4];
    bn254_fr_neg(accumEval, neg_accumEval);
    uint32_t neg_eval_scalar[8];
    fr_to_scalar(neg_accumEval, neg_eval_scalar);

    // evalG = neg_accumEval * G
    uint64_t evalG[12];
    bn254_point_scalar_mul(g1, neg_eval_scalar, evalG);

    // result = accumC + evalG - accumW
    uint64_t tmp[12];
    pt_add(accumC, evalG, tmp);

    // Negate accumW (negate Y in Fp)
    uint64_t neg_accumW[12];
    pt_neg(accumW, neg_accumW);

    uint64_t result[12];
    pt_add(tmp, neg_accumW, result);

    // Check if result is identity (z == 0 in projective)
    return (result[8]==0 && result[9]==0 && result[10]==0 && result[11]==0) ? 1 : 0;
}

void bn254_batch_to_affine(const uint64_t *proj, uint64_t *aff, int n) {
    batch_to_affine(proj, aff, n);
}

void bn254_projective_to_affine(const uint64_t p[12], uint64_t affine[8]) {
    if (fp_is_zero(p + 8)) {
        // Identity point: return (0, 0)
        memset(affine, 0, 64);
        return;
    }
    uint64_t zinv[4], zinv2[4], zinv3[4];
    fp_inv(p + 8, zinv);
    fp_sqr(zinv, zinv2);
    fp_mul(zinv2, zinv, zinv3);
    fp_mul(p, zinv2, affine);        // x_aff = X / Z^2
    fp_mul(p + 4, zinv3, affine + 4); // y_aff = Y / Z^3
}

void bn254_fr_inner_product(const uint64_t *a, const uint64_t *b, int n, uint64_t result[4]) {
    uint64_t acc[4] = {0, 0, 0, 0};
    for (int i = 0; i < n; i++) {
        uint64_t prod[4];
        fr_mul(a + i * 4, b + i * 4, prod);
        fr_add(acc, prod, acc);
    }
    memcpy(result, acc, 32);
}

void bn254_fr_vector_fold(const uint64_t *a, const uint64_t *b,
                           const uint64_t x[4], const uint64_t xInv[4],
                           int n, uint64_t *out) {
    for (int i = 0; i < n; i++) {
        uint64_t ax[4], bxi[4];
        fr_mul(a + i * 4, x, ax);
        fr_mul(b + i * 4, xInv, bxi);
        fr_add(ax, bxi, out + i * 4);
    }
}

void bn254_fr_batch_to_limbs(const uint64_t *mont, uint32_t *limbs, int n) {
    // Convert n Fr elements from Montgomery form to integer uint32 limbs.
    static const uint64_t ONE[4] = {1, 0, 0, 0};
    for (int i = 0; i < n; i++) {
        uint64_t r[4];
        fr_mul(mont + i * 4, ONE, r);
        limbs[i * 8 + 0] = (uint32_t)(r[0]);
        limbs[i * 8 + 1] = (uint32_t)(r[0] >> 32);
        limbs[i * 8 + 2] = (uint32_t)(r[1]);
        limbs[i * 8 + 3] = (uint32_t)(r[1] >> 32);
        limbs[i * 8 + 4] = (uint32_t)(r[2]);
        limbs[i * 8 + 5] = (uint32_t)(r[2] >> 32);
        limbs[i * 8 + 6] = (uint32_t)(r[3]);
        limbs[i * 8 + 7] = (uint32_t)(r[3] >> 32);
    }
}

void bn254_fr_synthetic_div(const uint64_t *coeffs, const uint64_t z[4],
                             int n, uint64_t *quotient) {
    // coeffs: polynomial coefficients, n elements (4 uint64 each, Montgomery form)
    // z: evaluation point (4 uint64, Montgomery form)
    // quotient: output n-1 elements
    // q[n-2] = coeffs[n-1]
    // q[i] = coeffs[i+1] + z * q[i+1]  for i = n-3..0
    if (n < 2) return;
    memcpy(quotient + (n - 2) * 4, coeffs + (n - 1) * 4, 32);
    for (int i = n - 3; i >= 0; i--) {
        uint64_t tmp[4];
        fr_mul(z, quotient + (i + 1) * 4, tmp);
        fr_add(coeffs + (i + 1) * 4, tmp, quotient + i * 4);
    }
}

// Horner evaluation: result = coeffs[0] + coeffs[1]*z + coeffs[2]*z^2 + ...
// coeffs: n elements in Montgomery form (4 uint64 each)
// z: evaluation point (Montgomery form)
// result: single Fr output (Montgomery form)
void bn254_fr_horner_eval(const uint64_t *coeffs, int n, const uint64_t z[4],
                           uint64_t result[4]) {
    if (n == 0) { memset(result, 0, 32); return; }
    // Start from highest degree: result = coeffs[n-1]
    memcpy(result, coeffs + (n - 1) * 4, 32);
    // result = result * z + coeffs[i] for i = n-2..0
    for (int i = n - 2; i >= 0; i--) {
        uint64_t tmp[4];
        fr_mul(result, z, tmp);
        fr_add(tmp, coeffs + i * 4, result);
    }
}

// Fused evaluation + synthetic division: computes both p(z) and q(x) = (p(x)-p(z))/(x-z)
// in a single pass. The synthetic division already computes q, and p(z) = coeffs[0] + z*q[0].
// coeffs: n elements (Montgomery form), z: evaluation point
// eval_out: p(z), quotient: n-1 elements
void bn254_fr_eval_and_div(const uint64_t *coeffs, int n, const uint64_t z[4],
                            uint64_t eval_out[4], uint64_t *quotient) {
    if (n == 0) { memset(eval_out, 0, 32); return; }
    if (n == 1) { memcpy(eval_out, coeffs, 32); return; }
    // Synthetic division: q[n-2] = coeffs[n-1], q[i] = coeffs[i+1] + z*q[i+1]
    memcpy(quotient + (n - 2) * 4, coeffs + (n - 1) * 4, 32);
    for (int i = n - 3; i >= 0; i--) {
        uint64_t tmp[4];
        fr_mul(z, quotient + (i + 1) * 4, tmp);
        fr_add(coeffs + (i + 1) * 4, tmp, quotient + i * 4);
    }
    // p(z) = coeffs[0] + z * q[0]
    uint64_t tmp[4];
    fr_mul(z, quotient, tmp);
    fr_add(coeffs, tmp, eval_out);
}

// Batch inverse using Montgomery's trick: O(3n) muls + 1 inversion.
// out[i] = a[i]^(-1) for i=0..n-1.
void bn254_fr_batch_inverse(const uint64_t *a, int n, uint64_t *out) {
    if (n == 0) return;
    if (n == 1) {
        fr_inv(a, out);
        return;
    }
    // Phase 1: prefix products. out[i] = a[0]*a[1]*...*a[i]
    memcpy(out, a, 32);
    for (int i = 1; i < n; i++) {
        fr_mul(out + (i - 1) * 4, a + i * 4, out + i * 4);
    }
    // Phase 2: invert the total product
    uint64_t inv[4];
    fr_inv(out + (n - 1) * 4, inv);
    // Phase 3: back-propagate inverses
    for (int i = n - 1; i >= 1; i--) {
        // out[i] = inv * prefix[i-1]
        uint64_t tmp[4];
        fr_mul(inv, out + (i - 1) * 4, tmp);
        // update inv = inv * a[i]
        uint64_t new_inv[4];
        fr_mul(inv, a + i * 4, new_inv);
        memcpy(inv, new_inv, 32);
        memcpy(out + i * 4, tmp, 32);
    }
    memcpy(out, inv, 32);
}

// Full sumcheck for a single multilinear polynomial.
// Input: evals[0..2^numVars], pre-derived challenges[0..numVars].
// Output: rounds[round][0..2] = (S(0), S(1), S(2)) stored as round*3*4 uint64_t,
//         finalEval[4] = final evaluation.
// Parallel: uses threads for large round sizes.
typedef struct {
    const uint64_t *evals;
    int halfN;
    uint64_t s0[4], s1[4], s2[4]; // partial sums for this thread
    int start, end;
} SumcheckRoundChunk;

// Fused round-poly + reduce worker: computes S(0), S(1), S(2) partial sums
// AND writes reduced output in a single pass (one memory traversal).
typedef struct {
    uint64_t *buf;        // mutable: writes reduced values to first half
    const uint64_t *challenge;
    int halfN;
    uint64_t s0[4], s1[4], s2[4];
    int start, end;
} SumcheckFusedChunk;

static void sumcheck_fused_worker(SumcheckFusedChunk *c) {
    uint64_t s0[4] = {0,0,0,0};
    uint64_t s1[4] = {0,0,0,0};
    uint64_t s2[4] = {0,0,0,0};
    uint64_t *buf = c->buf;
    const uint64_t *ch = c->challenge;
    int halfN = c->halfN;
    for (int i = c->start; i < c->end; i++) {
        uint64_t *a = buf + i * 4;
        uint64_t *b = buf + (halfN + i) * 4;
        // Round poly: accumulate S(0), S(1), S(2)
        uint64_t tmp[4];
        fr_add(s0, a, tmp); memcpy(s0, tmp, 32);
        fr_add(s1, b, tmp); memcpy(s1, tmp, 32);
        uint64_t twoB[4]; fr_add(b, b, twoB);
        uint64_t f2[4]; fr_sub(twoB, a, f2);
        fr_add(s2, f2, tmp); memcpy(s2, tmp, 32);
        // Reduce: a = a + ch * (b - a)
        uint64_t diff[4]; fr_sub(b, a, diff);
        uint64_t rd[4]; fr_mul(ch, diff, rd);
        uint64_t res[4]; fr_add(a, rd, res);
        memcpy(a, res, 32);
    }
    memcpy(c->s0, s0, 32);
    memcpy(c->s1, s1, 32);
    memcpy(c->s2, s2, 32);
}

static void *sumcheck_round_worker(void *arg) {
    SumcheckRoundChunk *c = (SumcheckRoundChunk *)arg;
    uint64_t s0[4] = {0,0,0,0};
    uint64_t s1[4] = {0,0,0,0};
    uint64_t s2[4] = {0,0,0,0};
    const uint64_t *ev = c->evals;
    int halfN = c->halfN;
    for (int i = c->start; i < c->end; i++) {
        const uint64_t *a = ev + i * 4;
        const uint64_t *b = ev + (halfN + i) * 4;
        uint64_t tmp[4];
        fr_add(s0, a, tmp); memcpy(s0, tmp, 32);
        fr_add(s1, b, tmp); memcpy(s1, tmp, 32);
        uint64_t twoB[4];
        fr_add(b, b, twoB);
        uint64_t f2[4];
        fr_sub(twoB, a, f2);
        fr_add(s2, f2, tmp); memcpy(s2, tmp, 32);
    }
    memcpy(c->s0, s0, 32);
    memcpy(c->s1, s1, 32);
    memcpy(c->s2, s2, 32);
    return NULL;
}

void bn254_fr_full_sumcheck(const uint64_t *evals, int numVars,
                             const uint64_t *challenges,
                             uint64_t *rounds, uint64_t *finalEval) {
    int n = 1 << numVars;
    uint64_t *buf = (uint64_t *)malloc(n * 32);
    memcpy(buf, evals, n * 32);

    // Pre-allocate thread chunks once, reused across all rounds
    const int maxThreads = 8;
    SumcheckFusedChunk *chunks = (SumcheckFusedChunk *)malloc(maxThreads * sizeof(SumcheckFusedChunk));

    for (int round = 0; round < numVars; round++) {
        int halfN = n / 2;
        uint64_t *rout = rounds + round * 12; // 3 Fr per round

        const uint64_t *ch = challenges + round * 4;
        if (halfN >= 8192) {
            // Fused parallel round-poly + reduce (single memory traversal)
            int nT = 8;
            if (halfN / 1024 < nT) nT = halfN / 1024;
            if (nT < 1) nT = 1;
            int perT = (halfN + nT - 1) / nT;
            for (int t = 0; t < nT; t++) {
                chunks[t].buf = buf;
                chunks[t].challenge = ch;
                chunks[t].halfN = halfN;
                chunks[t].start = t * perT;
                chunks[t].end = (t + 1) * perT;
                if (chunks[t].end > halfN) chunks[t].end = halfN;
            }
            dispatch_apply(nT, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                ^(size_t idx) {
                    sumcheck_fused_worker(&chunks[idx]);
                });
            // Reduce partial sums from all threads
            memcpy(rout, chunks[0].s0, 32);
            memcpy(rout + 4, chunks[0].s1, 32);
            memcpy(rout + 8, chunks[0].s2, 32);
            for (int t = 1; t < nT; t++) {
                uint64_t tmp[4];
                fr_add(rout, chunks[t].s0, tmp); memcpy(rout, tmp, 32);
                fr_add(rout + 4, chunks[t].s1, tmp); memcpy(rout + 4, tmp, 32);
                fr_add(rout + 8, chunks[t].s2, tmp); memcpy(rout + 8, tmp, 32);
            }
        } else {
            // Single-threaded fused round-poly + reduce
            uint64_t s0[4] = {0,0,0,0};
            uint64_t s1[4] = {0,0,0,0};
            uint64_t s2[4] = {0,0,0,0};
            for (int i = 0; i < halfN; i++) {
                uint64_t *a = buf + i * 4;
                uint64_t *b = buf + (halfN + i) * 4;
                uint64_t tmp[4];
                fr_add(s0, a, tmp); memcpy(s0, tmp, 32);
                fr_add(s1, b, tmp); memcpy(s1, tmp, 32);
                uint64_t twoB[4]; fr_add(b, b, twoB);
                uint64_t f2[4]; fr_sub(twoB, a, f2);
                fr_add(s2, f2, tmp); memcpy(s2, tmp, 32);
                // Reduce in same pass
                uint64_t diff[4]; fr_sub(b, a, diff);
                uint64_t rd[4]; fr_mul(ch, diff, rd);
                uint64_t res[4]; fr_add(a, rd, res);
                memcpy(a, res, 32);
            }
            memcpy(rout, s0, 32);
            memcpy(rout + 4, s1, 32);
            memcpy(rout + 8, s2, 32);
        }
        n = halfN;
    }
    memcpy(finalEval, buf, 32);
    free(chunks);
    free(buf);
}

// Evaluate multilinear extension at a point.
// evals: 2^numVars Fr elements. point: numVars Fr elements.
// Returns single Fr in result.
// Uses bn254_fr_vector_fold for the inner loop (same operation pattern).
// Stack-based for numVars <= 6 (64 elements = 2KB), heap for larger.
void bn254_fr_mle_eval(const uint64_t *evals, int numVars,
                        const uint64_t *point, uint64_t result[4]) {
    int n = 1 << numVars;

    // Stack buffer for small sizes (up to 2^6 = 64 elements = 2KB)
    uint64_t stack_buf[256]; // 64 Fr elements * 4 uint64 each
    uint64_t *buf;
    if (numVars <= 6) {
        buf = stack_buf;
    } else {
        buf = (uint64_t *)malloc(n * 32);
    }
    memcpy(buf, evals, n * 32);

    for (int v = 0; v < numVars; v++) {
        int half = n >> 1;
        const uint64_t *r = point + v * 4;
        uint64_t one_minus_r[4];
        fr_sub(FR_ONE, r, one_minus_r);
        // buf[i] = (1-r)*buf[i] + r*buf[half+i]
        // This is vector_fold(a=buf, b=buf+half, x=(1-r), xInv=r, n=half)
        bn254_fr_vector_fold(buf, buf + half * 4, one_minus_r, r, half, buf);
        n = half;
    }
    memcpy(result, buf, 32);
    if (numVars > 6) free(buf);
}

// Fused sparse matvec + MLE eval: computes MLE(M*z)(point) without
// materializing the M*z vector as a Swift array.
// CSR format: rowPtr[rows+1], colIdx[nnz], values[nnz*4 uint64].
// z: n Fr elements. point: numVars Fr elements.
// padM: 1 << numVars (padded row count, must be >= rows).
// result: single Fr element.
void bn254_sparse_matvec_mle(
    const int *rowPtr, const int *colIdx, const uint64_t *values,
    int rows, const uint64_t *z,
    const uint64_t *point, int numVars, int padM,
    uint64_t result[4])
{
    // Stack buffer for padded matvec result (up to 64 rows)
    uint64_t stack_mv[256]; // 64 Fr elements
    uint64_t *mv;
    if (padM <= 64) {
        mv = stack_mv;
    } else {
        mv = (uint64_t *)malloc(padM * 32);
    }

    // Compute M*z
    for (int i = 0; i < rows; i++) {
        uint64_t acc[4] = {0,0,0,0};
        for (int k = rowPtr[i]; k < rowPtr[i+1]; k++) {
            uint64_t tmp[4];
            fr_mul(values + k*4, z + colIdx[k]*4, tmp);
            fr_add(acc, tmp, acc);
        }
        memcpy(mv + i*4, acc, 32);
    }
    // Zero-pad
    for (int i = rows; i < padM; i++) {
        memset(mv + i*4, 0, 32);
    }

    // MLE eval (reuse stack buffer — we modify in-place)
    int n = padM;
    for (int v = 0; v < numVars; v++) {
        int half = n >> 1;
        const uint64_t *r = point + v * 4;
        uint64_t one_minus_r[4];
        fr_sub(FR_ONE, r, one_minus_r);
        bn254_fr_vector_fold(mv, mv + half * 4, one_minus_r, r, half, mv);
        n = half;
    }
    memcpy(result, mv, 32);
    if (padM > 64) free(mv);
}

// Fused: gather subtable values by index, add beta, batch inverse.
// out[i] = 1/(beta + subtable[indices[i]])
// Single-chunk batch inverse on a contiguous range [base, base+chunk) of out[].
// Assumes out[] already contains values to invert; writes inverses in-place.
static void batch_inverse_chunk(uint64_t *out, int base, int chunk) {
    if (chunk == 0) return;
    if (chunk == 1) { fr_inv(out + base * 4, out + base * 4); return; }
    uint64_t *prefix = (uint64_t *)malloc(chunk * 32);
    memcpy(prefix, out + base * 4, 32);
    for (int i = 1; i < chunk; i++) {
        fr_mul(prefix + (i - 1) * 4, out + (base + i) * 4, prefix + i * 4);
    }
    uint64_t inv[4];
    fr_inv(prefix + (chunk - 1) * 4, inv);
    for (int i = chunk - 1; i > 0; i--) {
        uint64_t tmp[4];
        fr_mul(inv, prefix + (i - 1) * 4, tmp);
        uint64_t new_inv[4];
        fr_mul(inv, out + (base + i) * 4, new_inv);
        memcpy(inv, new_inv, 32);
        memcpy(out + (base + i) * 4, tmp, 32);
    }
    memcpy(out + base * 4, inv, 32);
    free(prefix);
}

typedef struct {
    uint64_t *out;
    int base;
    int chunk;
} BatchInvChunk;

void bn254_fr_inverse_evals_indexed(const uint64_t beta[4], const uint64_t *subtable,
                                     const int *indices, int n, uint64_t *out) {
    if (n == 0) return;
    // Phase 1: gather + beta-add in one pass
    for (int i = 0; i < n; i++) {
        fr_add(beta, subtable + indices[i] * 4, out + i * 4);
    }
    if (n == 1) {
        fr_inv(out, out);
        return;
    }
    // Phase 2: parallel chunked batch inverse
    int nChunks = 10;
    if (n < nChunks * 1024) nChunks = (n + 1023) / 1024;
    if (nChunks < 1) nChunks = 1;
    int perChunk = (n + nChunks - 1) / nChunks;
    int total = n;

    dispatch_apply(nChunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
        ^(size_t idx) {
            int base = (int)idx * perChunk;
            int end = base + perChunk;
            if (end > total) end = total;
            batch_inverse_chunk(out, base, end - base);
        });
}

// Compute beta+values and then batch inverse: out[i] = 1/(beta + values[i])
void bn254_fr_inverse_evals(const uint64_t beta[4], const uint64_t *values,
                             int n, uint64_t *out) {
    if (n == 0) return;
    for (int i = 0; i < n; i++) {
        fr_add(beta, values + i * 4, out + i * 4);
    }
    if (n == 1) {
        fr_inv(out, out);
        return;
    }
    // Parallel chunked batch inverse for large n
    int nChunks = 10;
    if (n < nChunks * 1024) nChunks = (n + 1023) / 1024;
    if (nChunks < 1) nChunks = 1;
    int perChunk = (n + nChunks - 1) / nChunks;
    int total = n;

    dispatch_apply(nChunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
        ^(size_t idx) {
            int base = (int)idx * perChunk;
            int end = base + perChunk;
            if (end > total) end = total;
            batch_inverse_chunk(out, base, end - base);
        });
}

// Compute weighted inverse evals: out[j] = weights[j] / (beta + values[j])
void bn254_fr_weighted_inverse_evals(const uint64_t beta[4], const uint64_t *values,
                                      const uint64_t *weights, int n, uint64_t *out) {
    bn254_fr_inverse_evals(beta, values, n, out);
    for (int i = 0; i < n; i++) {
        uint64_t tmp[4];
        fr_mul(out + i * 4, weights + i * 4, tmp);
        memcpy(out + i * 4, tmp, 32);
    }
}

// Fused: compute MLE(1/(beta + subtable[indices[x]]))(point) in a single pass.
// Avoids materializing the full inverse array, better cache behavior.
// Returns result = sum_x eq(x, point) / (beta + subtable[indices[x]])
void bn254_fr_inverse_mle_eval(const uint64_t beta[4], const uint64_t *subtable,
                                const int *indices, int n, int numVars,
                                const uint64_t *point, uint64_t result[4]) {
    if (n == 0) { memset(result, 0, 32); return; }

    // Step 1: Build eq polynomial eq[x] = prod_i ((1-r_i)(1-x_i) + r_i*x_i)
    // Start with eq[0] = 1, then for each variable, double the table size
    uint64_t *eq = (uint64_t *)malloc(n * 32);
    memcpy(eq, FR_ONE, 32);
    int cur_size = 1;
    for (int v = 0; v < numVars; v++) {
        const uint64_t *r = point + v * 4;
        uint64_t one_minus_r[4];
        fr_sub(FR_ONE, r, one_minus_r);
        // Expand: eq[i + cur_size] = eq[i] * r, eq[i] = eq[i] * (1-r)
        for (int i = cur_size - 1; i >= 0; i--) {
            fr_mul(eq + i * 4, r, eq + (i + cur_size) * 4);
            uint64_t tmp[4];
            fr_mul(eq + i * 4, one_minus_r, tmp);
            memcpy(eq + i * 4, tmp, 32);
        }
        cur_size *= 2;
    }

    // Step 2: Compute d[i] = beta + subtable[indices[i]]
    uint64_t *d = (uint64_t *)malloc(n * 32);
    for (int i = 0; i < n; i++) {
        fr_add(beta, subtable + indices[i] * 4, d + i * 4);
    }

    // Step 3: Batch inverse with fused inner product during back-propagation
    // 3a: prefix products
    uint64_t *prefix = (uint64_t *)malloc(n * 32);
    memcpy(prefix, d, 32);
    for (int i = 1; i < n; i++) {
        fr_mul(prefix + (i - 1) * 4, d + i * 4, prefix + i * 4);
    }

    // 3b: single Fermat inversion
    uint64_t inv[4];
    fr_inv(prefix + (n - 1) * 4, inv);

    // 3c: back-propagation + accumulation
    uint64_t acc[4] = {0, 0, 0, 0};
    for (int i = n - 1; i >= 1; i--) {
        // inv_i = inv * prefix[i-1]
        uint64_t inv_i[4];
        fr_mul(inv, prefix + (i - 1) * 4, inv_i);
        // update inv = inv * d[i]
        uint64_t new_inv[4];
        fr_mul(inv, d + i * 4, new_inv);
        memcpy(inv, new_inv, 32);
        // Accumulate: acc += eq[i] * inv_i
        uint64_t prod[4];
        fr_mul(eq + i * 4, inv_i, prod);
        fr_add(acc, prod, acc);
    }
    // i=0: inv_0 = inv (the updated inv is d[0]^{-1} after all updates)
    uint64_t prod0[4];
    fr_mul(eq, inv, prod0);
    fr_add(acc, prod0, acc);

    memcpy(result, acc, 32);
    free(eq);
    free(d);
    free(prefix);
}

// ============================================================
// Basefold: single-round fold on CPU.
// result[i] = evals[i] + alpha * (evals[i + halfN] - evals[i])
// evals: 2*halfN Fr elements (4 uint64 each, Montgomery form).
// result: halfN Fr elements output.
// alpha: single Fr element (4 uint64, Montgomery form).
// Multi-threaded via GCD for halfN >= 4096.
// ============================================================
#include <dispatch/dispatch.h>

void bn254_fr_basefold_fold(const uint64_t *evals, uint64_t *result,
                             const uint64_t *alpha, uint32_t halfN) {
    if (halfN >= 4096) {
        int nBlocks = 8;
        if (halfN < 8192) nBlocks = 4;
        int chunk = (int)halfN / nBlocks;
        int hn = (int)halfN;
        dispatch_apply(nBlocks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
            ^(size_t t) {
                int s = (int)t * chunk;
                int e = ((int)t == nBlocks - 1) ? hn : s + chunk;
                for (int j = s; j < e; j++) {
                    const uint64_t *a = evals + j * 4;
                    const uint64_t *b = evals + (j + hn) * 4;
                    uint64_t diff[4], rd[4];
                    fr_sub(b, a, diff);
                    fr_mul(alpha, diff, rd);
                    fr_add(a, rd, result + j * 4);
                }
            });
    } else {
        for (uint32_t j = 0; j < halfN; j++) {
            const uint64_t *a = evals + j * 4;
            const uint64_t *b = evals + (j + halfN) * 4;
            uint64_t diff[4], rd[4];
            fr_sub(b, a, diff);
            fr_mul(alpha, diff, rd);
            fr_add(a, rd, result + j * 4);
        }
    }
}

// ============================================================
// Basefold: fold all rounds on CPU, storing each intermediate layer.
// out_layers: flat buffer that will receive layer0 (n/2 elements),
//   layer1 (n/4 elements), ..., layerK (1 element).
// Total elements written = n/2 + n/4 + ... + 1 = n - 1.
// evals: n Fr elements (4 uint64 each, Montgomery form).
// point: numVars Fr elements (challenges).
// numVars: log2(n).
// Multi-threaded for large rounds via dispatch_apply.
// ============================================================

void bn254_fr_basefold_fold_all(const uint64_t *evals, int numVars,
                                 const uint64_t *point, uint64_t *out_layers) {
    int n = 1 << numVars;

    // Working buffer: starts as copy of evals, folded in-place each round
    uint64_t *buf = (uint64_t *)malloc(n * 32);
    memcpy(buf, evals, n * 32);

    uint64_t *outPtr = out_layers;

    for (int v = 0; v < numVars; v++) {
        int halfN = n >> 1;
        const uint64_t *alpha = point + v * 4;

        if (halfN >= 4096) {
            // Multi-threaded fold via GCD
            int nBlocks = 8;
            if (halfN < 8192) nBlocks = 4;
            int chunk = halfN / nBlocks;
            uint64_t *srcBuf = buf;
            uint64_t *dstBuf = outPtr;
            int hn = halfN;
            dispatch_apply(nBlocks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                ^(size_t t) {
                    int s = (int)t * chunk;
                    int e = ((int)t == nBlocks - 1) ? hn : s + chunk;
                    for (int j = s; j < e; j++) {
                        const uint64_t *a = srcBuf + j * 4;
                        const uint64_t *b = srcBuf + (j + hn) * 4;
                        uint64_t diff[4], rd[4], res[4];
                        fr_sub(b, a, diff);
                        fr_mul(alpha, diff, rd);
                        fr_add(a, rd, res);
                        memcpy(dstBuf + j * 4, res, 32);
                    }
                });
        } else {
            // Single-threaded fold
            for (int j = 0; j < halfN; j++) {
                const uint64_t *a = buf + j * 4;
                const uint64_t *b = buf + (j + halfN) * 4;
                uint64_t diff[4], rd[4], res[4];
                fr_sub(b, a, diff);
                fr_mul(alpha, diff, rd);
                fr_add(a, rd, res);
                memcpy(outPtr + j * 4, res, 32);
            }
        }

        // Copy output back to buf for next round
        memcpy(buf, outPtr, halfN * 32);
        outPtr += halfN * 4;  // advance output pointer
        n = halfN;
    }

    free(buf);
}

// ============================================================
// BGMW fixed-base scalar multiplication
// Precomputes tables of [0, G, 2G, ..., (2^w-1)G] for each
// generator at each window position. Runtime: additions only.
// ============================================================

typedef struct {
    const uint64_t *gens;
    uint64_t *proj;
    int start, end;
    int window_bits, num_windows, table_size, entries_per_gen;
} bgmw_pre_args_t;

static void *bgmw_pre_worker(void *arg) {
    bgmw_pre_args_t *a = (bgmw_pre_args_t *)arg;
    for (int i = a->start; i < a->end; i++) {
        const uint64_t *gaff = a->gens + i * 8;
        int base_off = i * a->entries_per_gen;

        // Convert affine generator to projective
        uint64_t window_base[12];
        memcpy(window_base, gaff, 64);       // x, y from affine
        memcpy(window_base + 8, FP_ONE, 32); // z = 1

        for (int win = 0; win < a->num_windows; win++) {
            int win_off = base_off + win * a->table_size;

            // table[win][0] = 1 * window_base
            memcpy(a->proj + (size_t)win_off * 12, window_base, 96);

            // table[win][d-1] = d * window_base (incremental add)
            uint64_t acc[12];
            memcpy(acc, window_base, 96);
            for (int d = 1; d < a->table_size; d++) {
                uint64_t result[12];
                pt_add(acc, window_base, result);
                memcpy(acc, result, 96);
                memcpy(a->proj + (size_t)(win_off + d) * 12, acc, 96);
            }

            // Advance window_base by 2^w doublings
            if (win < a->num_windows - 1) {
                for (int b = 0; b < a->window_bits; b++) {
                    uint64_t dbl[12];
                    pt_dbl(window_base, dbl);
                    memcpy(window_base, dbl, 96);
                }
            }
        }
    }
    return NULL;
}

void bgmw_precompute(const uint64_t *generators_affine, int n,
                     int window_bits, uint64_t *table_out) {
    const int num_windows = (256 + window_bits - 1) / window_bits;
    const int table_size = (1 << window_bits) - 1;
    const int entries_per_gen = num_windows * table_size;
    const int total = n * entries_per_gen;

    uint64_t *proj = (uint64_t *)malloc((size_t)total * 96);
    if (!proj) return;

    int num_blocks = n < 8 ? n : 8;
    int chunk = (n + num_blocks - 1) / num_blocks;
    bgmw_pre_args_t *args = (bgmw_pre_args_t *)malloc(num_blocks * sizeof(bgmw_pre_args_t));

    for (int t = 0; t < num_blocks; t++) {
        args[t].gens = generators_affine;
        args[t].proj = proj;
        args[t].start = t * chunk;
        args[t].end = (t + 1) * chunk < n ? (t + 1) * chunk : n;
        args[t].window_bits = window_bits;
        args[t].num_windows = num_windows;
        args[t].table_size = table_size;
        args[t].entries_per_gen = entries_per_gen;
    }
    dispatch_apply(num_blocks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
        ^(size_t idx) {
            bgmw_pre_worker(&args[idx]);
        });
    free(args);

    batch_to_affine(proj, table_out, total);
    free(proj);
}

typedef struct {
    const uint64_t *table;
    const uint32_t *scalars;
    uint64_t partial[12];
    int start, end;
    int window_bits, num_windows, table_size, entries_per_gen;
    uint32_t mask;
} bgmw_msm_args_t;

static void *bgmw_msm_worker(void *arg) {
    bgmw_msm_args_t *a = (bgmw_msm_args_t *)arg;
    uint64_t acc[12];
    pt_set_id(acc);

    for (int i = a->start; i < a->end; i++) {
        int scalar_base = i * 8;
        int gen_table_base = i * a->entries_per_gen;
        int bit_offset = 0;

        for (int win = 0; win < a->num_windows; win++) {
            int limb_idx = bit_offset / 32;
            int bit_idx = bit_offset % 32;

            uint32_t digit;
            if (limb_idx < 8) {
                digit = a->scalars[scalar_base + limb_idx] >> bit_idx;
                if (bit_idx + a->window_bits > 32 && limb_idx + 1 < 8) {
                    digit |= a->scalars[scalar_base + limb_idx + 1] << (32 - bit_idx);
                }
                digit &= a->mask;
            } else {
                digit = 0;
            }

            if (digit != 0) {
                int entry_idx = gen_table_base + win * a->table_size + (int)(digit - 1);
                const uint64_t *entry_ptr = a->table + (size_t)entry_idx * 8;
                uint64_t tmp[12];
                pt_add_mixed(acc, entry_ptr, tmp);
                memcpy(acc, tmp, 96);
            }

            bit_offset += a->window_bits;
        }
    }
    memcpy(a->partial, acc, 96);
    return NULL;
}

void bgmw_msm(const uint64_t *table, int n, int window_bits,
              const uint32_t *scalars, uint64_t *result) {
    const int num_windows = (256 + window_bits - 1) / window_bits;
    const int table_size = (1 << window_bits) - 1;
    const int entries_per_gen = num_windows * table_size;
    const uint32_t mask = (uint32_t)((1 << window_bits) - 1);

    int num_blocks = n < 8 ? n : 8;
    int chunk = (n + num_blocks - 1) / num_blocks;
    bgmw_msm_args_t *args = (bgmw_msm_args_t *)malloc(num_blocks * sizeof(bgmw_msm_args_t));

    for (int t = 0; t < num_blocks; t++) {
        args[t].table = table;
        args[t].scalars = scalars;
        args[t].start = t * chunk;
        args[t].end = (t + 1) * chunk < n ? (t + 1) * chunk : n;
        args[t].window_bits = window_bits;
        args[t].num_windows = num_windows;
        args[t].table_size = table_size;
        args[t].entries_per_gen = entries_per_gen;
        args[t].mask = mask;
        pt_set_id(args[t].partial);
    }
    dispatch_apply(num_blocks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
        ^(size_t idx) {
            bgmw_msm_worker(&args[idx]);
        });

    memcpy(result, args[0].partial, 96);
    for (int t = 1; t < num_blocks; t++) {
        if (!pt_is_id(args[t].partial)) {
            uint64_t tmp[12];
            pt_add(result, args[t].partial, tmp);
            memcpy(result, tmp, 96);
        }
    }
    free(args);
}

// ============================================================
// Exported Fp operations: inv, sqrt
// ============================================================

void bn254_fp_inv(const uint64_t a[4], uint64_t r[4]) {
    fp_inv(a, r);
}

// Fp sqrt via a^((p+1)/4). Returns 1 if sqrt exists, 0 otherwise.
// BN254 Fp has p ≡ 3 mod 4.
int bn254_fp_sqrt(const uint64_t a[4], uint64_t r[4]) {
    // Compute (p+1)/4
    uint64_t exp[4];
    for (int i = 0; i < 4; i++) exp[i] = FP_P[i];
    // p+1
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t sum = exp[i] + (i == 0 ? 1 : 0) + carry;
        carry = (sum < exp[i]) || (i == 0 && sum == 0) ? 1 : 0;
        exp[i] = sum;
    }
    // >>2  (divide by 4)
    for (int i = 0; i < 3; i++) {
        exp[i] = (exp[i] >> 2) | (exp[i+1] << 62);
    }
    exp[3] >>= 2;

    // Square-and-multiply
    memcpy(r, FP_ONE, 32);
    uint64_t base[4];
    memcpy(base, a, 32);
    for (int i = 0; i < 4; i++) {
        uint64_t word = exp[i];
        for (int bit = 0; bit < 64; bit++) {
            if ((word >> bit) & 1) fp_mul(r, base, r);
            fp_sqr(base, base);
        }
    }

    // Verify: r^2 == a?
    uint64_t check[4];
    fp_sqr(r, check);
    for (int i = 0; i < 4; i++) {
        if (check[i] != a[i]) return 0;
    }
    return 1;
}

// ============================================================
// IPA fused round: compute L, R, fold a, b, G in one C call.
// Eliminates Swift array copies, repeated C call overhead, and
// thread spawning per operation.
// ============================================================

