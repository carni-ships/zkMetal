// BN254 G1 Pippenger MSM — optimized C implementation
// Features:
//   1. CIOS Montgomery field ops (4×64-bit limbs, __uint128_t)
//   2. Mixed affine addition (projective + affine, saves 4 muls/add)
//   3. Batch-to-affine via Montgomery's trick (1 inversion per window)
//   4. Multi-threaded windows (pthreads)
//   5. Adaptive window sizing
//
// Interop: Swift PointAffine = (Fp x, Fp y) where Fp = 8×UInt32
//          In 64-bit view: 4×uint64_t per Fp, 8×uint64_t per PointAffine
//          Scalars: 8×uint32_t per scalar (little-endian limbs)

#include "NeonFieldOps.h"
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

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
    fp_mul(a, a, r);
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
            fp_mul(b, b, b);
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

// Mixed addition: projective P + affine Q (Z_Q = 1)
// Saves 2 muls + 1 sqr vs full projective add
static void pt_add_mixed(const uint64_t p[12], const uint64_t q_aff[8], uint64_t r[12]) {
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
    uint64_t result[12], base[12];
    pt_set_id(result);
    memcpy(base, p, 96);

    for (int i = 0; i < 8; i++) {
        uint32_t word = scalar[i];
        for (int bit = 0; bit < 32; bit++) {
            if (word & 1) {
                uint64_t tmp[12];
                pt_add(result, base, tmp);
                memcpy(result, tmp, 96);
            }
            uint64_t tmp[12];
            pt_dbl(base, tmp);
            memcpy(base, tmp, 96);
            word >>= 1;
        }
    }
    memcpy(r, result, 96);
}

// Batch fold generators: result[i] = scalar_mul(GL[i], xInv) + scalar_mul(GR[i], x)
// GL, GR: projective points (12 uint64 each)
// x, xInv: 8 × uint32_t scalars (non-Montgomery, integer form)
// Uses pthreads for parallelism.
typedef struct {
    const uint64_t *GL;
    const uint64_t *GR;
    const uint32_t *x;
    const uint32_t *xInv;
    uint64_t *result;
    int start;
    int end;
} FoldChunk;

static void *fold_worker(void *arg) {
    FoldChunk *c = (FoldChunk *)arg;
    for (int i = c->start; i < c->end; i++) {
        uint64_t gL_scaled[12], gR_scaled[12], sum[12];
        pt_scalar_mul(c->GL + i * 12, c->xInv, gL_scaled);
        pt_scalar_mul(c->GR + i * 12, c->x, gR_scaled);
        pt_add(gL_scaled, gR_scaled, sum);
        memcpy(c->result + i * 12, sum, 96);
    }
    return NULL;
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

    int nThreads = 8;  // cap threads
    if (halfLen < nThreads) nThreads = halfLen;

    pthread_t threads[8];
    FoldChunk chunks[8];
    int chunkSize = (halfLen + nThreads - 1) / nThreads;

    for (int t = 0; t < nThreads; t++) {
        chunks[t].GL = GL;
        chunks[t].GR = GR;
        chunks[t].x = x;
        chunks[t].xInv = xInv;
        chunks[t].result = result;
        chunks[t].start = t * chunkSize;
        chunks[t].end = (t + 1) * chunkSize;
        if (chunks[t].end > halfLen) chunks[t].end = halfLen;
        pthread_create(&threads[t], NULL, fold_worker, &chunks[t]);
    }
    for (int t = 0; t < nThreads; t++)
        pthread_join(threads[t], NULL);
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

    // Allocate tasks and threads
    WindowTask *tasks = (WindowTask *)malloc((size_t)num_windows * sizeof(WindowTask));
    pthread_t *threads = (pthread_t *)malloc((size_t)num_windows * sizeof(pthread_t));

    for (int w = 0; w < num_windows; w++) {
        tasks[w].points = points;
        tasks[w].scalars = scalars;
        tasks[w].n = n;
        tasks[w].window_bits = wb;
        tasks[w].window_idx = w;
        tasks[w].num_buckets = num_buckets;
    }

    // Launch threads (one per window)
    for (int w = 0; w < num_windows; w++)
        pthread_create(&threads[w], NULL, window_worker, &tasks[w]);

    for (int w = 0; w < num_windows; w++)
        pthread_join(threads[w], NULL);

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
    free(threads);
}

// ============================================================
// Exported utility functions
// ============================================================

void bn254_point_scalar_mul(const uint64_t p[12], const uint32_t scalar[8], uint64_t r[12]) {
    pt_scalar_mul(p, scalar, r);
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
