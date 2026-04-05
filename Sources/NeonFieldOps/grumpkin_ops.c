// Grumpkin curve point operations in C with CIOS Montgomery arithmetic.
// Grumpkin: y^2 = x^3 - 17 over BN254 Fr (a=0, b=-17).
// Points in Jacobian projective coordinates: (X, Y, Z) -> affine (X/Z^2, Y/Z^3).
// Each coordinate is 4 x uint64_t in BN254 Fr Montgomery form.

#include "NeonFieldOps.h"
#include <string.h>
#include <stdlib.h>
#include <pthread.h>

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

static inline void gk_copy(uint64_t dst[4], const uint64_t src[4]) {
    dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = src[3];
}

// Fp inversion via Fermat: a^(p-2) mod p
static void gk_inv(const uint64_t a[4], uint64_t result[4]) {
    uint64_t pm2[4];
    pm2[0] = GK_P[0] - 2;
    pm2[1] = GK_P[1];
    pm2[2] = GK_P[2];
    pm2[3] = GK_P[3];

    memcpy(result, GK_ONE, 32);
    uint64_t b[4];
    memcpy(b, a, 32);
    for (int i = 0; i < 4; i++) {
        for (int bit = 0; bit < 64; bit++) {
            if ((pm2[i] >> bit) & 1)
                gk_mul(result, b, result);
            gk_sqr(b, b);
        }
    }
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

// Check if an affine point is identity (represented as (0,0))
static inline int gk_aff_is_id(const uint64_t q[8]) {
    return (q[0] | q[1] | q[2] | q[3] | q[4] | q[5] | q[6] | q[7]) == 0;
}

// Mixed addition: projective P + affine Q (Z_Q = 1)
// Saves 2 muls + 1 sqr vs full projective add
static void gk_pt_add_mixed(const uint64_t p[12], const uint64_t q_aff[8], uint64_t r[12]) {
    if (gk_aff_is_id(q_aff)) {
        memcpy(r, p, 96);
        return;
    }
    if (gk_pt_is_id(p)) {
        memcpy(r, q_aff, 64);        // x, y from affine
        memcpy(r + 8, GK_ONE, 32);   // z = 1
        return;
    }

    const uint64_t *px = p, *py = p+4, *pz = p+8;
    const uint64_t *qx = q_aff, *qy = q_aff + 4;

    uint64_t z1z1[4], u2[4], s2[4], h[4], rr[4];
    uint64_t ii[4], j[4], vv[4], t1[4], hh[4];

    gk_sqr(pz, z1z1);
    gk_mul(qx, z1z1, u2);         // u2 = qx * z1^2
    gk_mul(pz, z1z1, t1);
    gk_mul(qy, t1, s2);           // s2 = qy * z1^3

    gk_sub(u2, px, h);            // h = u2 - px (u1 = px)
    gk_sub(s2, py, t1);
    gk_dbl(t1, rr);               // r = 2(s2 - s1)

    if (gk_is_zero(h)) {
        if (gk_is_zero(rr)) { gk_pt_dbl(p, r); return; }
        gk_pt_set_id(r); return;
    }

    gk_dbl(h, t1);
    gk_sqr(t1, ii);
    gk_mul(h, ii, j);
    gk_mul(px, ii, vv);

    // x3 = r^2 - j - 2v
    gk_sqr(rr, r);
    gk_sub(r, j, r);
    gk_dbl(vv, t1);
    gk_sub(r, t1, r);

    // y3 = r(v - x3) - 2*s1*j
    gk_sub(vv, r, t1);
    gk_mul(rr, t1, r + 4);
    gk_mul(py, j, t1);
    gk_dbl(t1, t1);
    gk_sub(r + 4, t1, r + 4);

    // z3 = (z1 + h)^2 - z1z1 - h^2
    gk_add(pz, h, t1);
    gk_sqr(t1, t1);
    gk_sub(t1, z1z1, t1);
    gk_sqr(h, hh);
    gk_sub(t1, hh, r + 8);
}

// Exported mixed addition
void grumpkin_point_add_mixed(const uint64_t p[12], const uint64_t q_aff[8], uint64_t r[12]) {
    gk_pt_add_mixed(p, q_aff, r);
}

// ============================================================
// Batch projective-to-affine (Montgomery's trick)
// Single inversion for all non-identity points
// ============================================================

static void gk_batch_to_affine(const uint64_t *proj, uint64_t *aff, int n) {
    if (n == 0) return;

    uint64_t *prods = (uint64_t *)malloc((size_t)n * 32);
    int first_valid = -1;

    for (int i = 0; i < n; i++) {
        if (gk_pt_is_id(proj + i * 12)) {
            if (i == 0) memcpy(prods, GK_ONE, 32);
            else memcpy(prods + i * 4, prods + (i-1) * 4, 32);
        } else {
            if (first_valid < 0) {
                first_valid = i;
                memcpy(prods + i * 4, proj + i * 12 + 8, 32);  // Z_i
            } else {
                gk_mul(prods + (i-1) * 4, proj + i * 12 + 8, prods + i * 4);
            }
        }
    }

    if (first_valid < 0) {
        memset(aff, 0, (size_t)n * 64);
        free(prods);
        return;
    }

    uint64_t inv[4];
    gk_inv(prods + (n-1) * 4, inv);

    for (int i = n - 1; i >= 0; i--) {
        if (gk_pt_is_id(proj + i * 12)) {
            memset(aff + i * 8, 0, 64);
            continue;
        }

        uint64_t zinv[4];
        if (i > first_valid) {
            gk_mul(inv, prods + (i-1) * 4, zinv);
            gk_mul(inv, proj + i * 12 + 8, inv);
        } else {
            gk_copy(zinv, inv);
        }

        uint64_t zinv2[4], zinv3[4];
        gk_sqr(zinv, zinv2);
        gk_mul(zinv2, zinv, zinv3);
        gk_mul(proj + i * 12, zinv2, aff + i * 8);
        gk_mul(proj + i * 12 + 4, zinv3, aff + i * 8 + 4);
    }

    free(prods);
}

// ============================================================
// Scalar window extraction
// ============================================================

static inline uint32_t gk_extract_window(const uint32_t *scalar, int window_idx, int window_bits) {
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

static int gk_optimal_window_bits(int n) {
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
// Per-window worker for Grumpkin Pippenger MSM
// ============================================================

typedef struct {
    const uint64_t *points;     // n affine points (8 limbs each)
    const uint32_t *scalars;    // n scalars (8 uint32 each)
    int n;
    int window_bits;
    int window_idx;
    int num_buckets;
    uint64_t result[12];        // output projective point
} GkWindowTask;

static void *gk_window_worker(void *arg) {
    GkWindowTask *task = (GkWindowTask *)arg;
    int wb = task->window_bits;
    int w = task->window_idx;
    int nb = task->num_buckets;
    int nn = task->n;

    // Allocate projective buckets (identity-initialized)
    uint64_t *buckets = (uint64_t *)malloc((size_t)(nb + 1) * 96);
    for (int b = 0; b <= nb; b++)
        gk_pt_set_id(buckets + b * 12);

    // Phase 1: Bucket accumulation (mixed affine addition)
    for (int i = 0; i < nn; i++) {
        uint32_t digit = gk_extract_window(task->scalars + i * 8, w, wb);
        if (digit != 0) {
            uint64_t tmp[12];
            gk_pt_add_mixed(buckets + digit * 12, task->points + i * 8, tmp);
            memcpy(buckets + digit * 12, tmp, 96);
        }
    }

    // Phase 2: Batch convert buckets to affine (Montgomery's trick)
    uint64_t *bucket_aff = (uint64_t *)malloc((size_t)nb * 64);
    gk_batch_to_affine(buckets + 12, bucket_aff, nb);

    // Phase 3: Running-sum reduction using mixed addition
    uint64_t running[12], window_sum[12];
    gk_pt_set_id(running);
    gk_pt_set_id(window_sum);

    for (int j = nb - 1; j >= 0; j--) {
        if (!gk_aff_is_id(bucket_aff + j * 8)) {
            uint64_t tmp[12];
            gk_pt_add_mixed(running, bucket_aff + j * 8, tmp);
            memcpy(running, tmp, 96);
        }
        uint64_t tmp[12];
        gk_pt_add(window_sum, running, tmp);
        memcpy(window_sum, tmp, 96);
    }

    memcpy(task->result, window_sum, 96);
    free(buckets);
    free(bucket_aff);
    return NULL;
}

// ============================================================
// Grumpkin Pippenger MSM
// ============================================================

void grumpkin_pippenger_msm(
    const uint64_t *points,    // n affine points: n x 8 uint64_t (BN254 Fr Montgomery form)
    const uint32_t *scalars,   // n scalars: n x 8 uint32_t (little-endian 256-bit integer)
    int n,
    uint64_t *result)          // output: 12 uint64_t (projective)
{
    if (n == 0) { gk_pt_set_id(result); return; }

    int wb = gk_optimal_window_bits(n);
    int num_windows = (256 + wb - 1) / wb;
    int num_buckets = (1 << wb) - 1;

    GkWindowTask *tasks = (GkWindowTask *)malloc((size_t)num_windows * sizeof(GkWindowTask));
    pthread_t *threads = (pthread_t *)malloc((size_t)num_windows * sizeof(pthread_t));

    for (int w = 0; w < num_windows; w++) {
        tasks[w].points = points;
        tasks[w].scalars = scalars;
        tasks[w].n = n;
        tasks[w].window_bits = wb;
        tasks[w].window_idx = w;
        tasks[w].num_buckets = num_buckets;
    }

    for (int w = 0; w < num_windows; w++)
        pthread_create(&threads[w], NULL, gk_window_worker, &tasks[w]);

    for (int w = 0; w < num_windows; w++)
        pthread_join(threads[w], NULL);

    // Horner combination: result = sum windowResults[w] * 2^(w * wb)
    memcpy(result, tasks[num_windows - 1].result, 96);
    for (int w = num_windows - 2; w >= 0; w--) {
        uint64_t tmp[12];
        for (int s = 0; s < wb; s++) {
            gk_pt_dbl(result, tmp);
            memcpy(result, tmp, 96);
        }
        gk_pt_add(result, tasks[w].result, tmp);
        memcpy(result, tmp, 96);
    }

    free(tasks);
    free(threads);
}
