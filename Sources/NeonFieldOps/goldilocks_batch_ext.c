// Extended Goldilocks NEON batch operations
// p = 2^64 - 2^32 + 1 (Goldilocks prime)
//
// goldilocks_neon.c already has batch add/sub/mul and NTT.
// This file adds: inner product, batch scale, linear combination,
// and Goldilocks extension field (Fp2) operations.

#include "NeonFieldOps.h"
#include <arm_neon.h>
#include <string.h>

// ============================================================
// Constants
// ============================================================

#define GL_P    0xFFFFFFFF00000001ULL
#define GL_EPS  0xFFFFFFFFULL

typedef unsigned __int128 uint128_t;

// ============================================================
// Scalar Goldilocks arithmetic (duplicated for self-containment)
// ============================================================

static inline uint64_t glx_reduce128(uint64_t hi, uint64_t lo) {
    uint64_t hi_lo = hi & 0xFFFFFFFFULL;
    uint64_t hi_hi = hi >> 32;
    uint64_t hi_lo_eps = hi_lo * GL_EPS;
    uint64_t t1;
    unsigned c1 = __builtin_add_overflow(lo, hi_lo_eps, &t1);
    uint64_t t2;
    unsigned b2 = __builtin_sub_overflow(t1, hi_hi, &t2);
    uint64_t r = t2;
    if (c1) r += GL_EPS;
    if (b2) r += GL_P;
    if (r >= GL_P) r -= GL_P;
    return r;
}

static inline uint64_t glx_mul(uint64_t a, uint64_t b) {
    uint128_t prod = (uint128_t)a * b;
    return glx_reduce128((uint64_t)(prod >> 64), (uint64_t)prod);
}

static inline uint64_t glx_add(uint64_t a, uint64_t b) {
    uint64_t s;
    unsigned carry = __builtin_add_overflow(a, b, &s);
    if (carry) {
        s += GL_EPS;
        if (s < GL_EPS) s += GL_EPS;
    }
    return s >= GL_P ? s - GL_P : s;
}

static inline uint64_t glx_sub(uint64_t a, uint64_t b) {
    if (a >= b) return a - b;
    return a + GL_P - b;
}

// ============================================================
// NEON vectorized add/sub (2-wide, from goldilocks_neon.c)
// ============================================================

static inline uint64x2_t glx_p_vec(void) { return vdupq_n_u64(GL_P); }
static inline uint64x2_t glx_eps_vec(void) { return vdupq_n_u64(GL_EPS); }

static inline uint64x2_t glx_add_vec(uint64x2_t a, uint64x2_t b) {
    uint64x2_t p = glx_p_vec();
    uint64x2_t eps = glx_eps_vec();
    uint64x2_t sum = vaddq_u64(a, b);
    uint64x2_t carry_mask = vcltq_u64(sum, a);
    sum = vaddq_u64(sum, vandq_u64(eps, carry_mask));
    uint64x2_t carry2_mask = vandq_u64(carry_mask, vcltq_u64(sum, eps));
    sum = vaddq_u64(sum, vandq_u64(eps, carry2_mask));
    uint64x2_t ge_mask = vcgeq_u64(sum, p);
    sum = vsubq_u64(sum, vandq_u64(p, ge_mask));
    return sum;
}

static inline uint64x2_t glx_sub_vec(uint64x2_t a, uint64x2_t b) {
    uint64x2_t p = glx_p_vec();
    uint64x2_t diff = vsubq_u64(a, b);
    uint64x2_t borrow_mask = vcltq_u64(a, b);
    diff = vaddq_u64(diff, vandq_u64(p, borrow_mask));
    return diff;
}

// ============================================================
// Batch scale: out[i] = a[i] * scalar
// ============================================================

void gl_batch_scale_neon(const uint64_t *a, uint64_t scalar, uint64_t *out, int n) {
    int i = 0;
    for (; i + 3 < n; i += 4) {
        uint128_t p0 = (uint128_t)a[i]   * scalar;
        uint128_t p1 = (uint128_t)a[i+1] * scalar;
        uint128_t p2 = (uint128_t)a[i+2] * scalar;
        uint128_t p3 = (uint128_t)a[i+3] * scalar;
        out[i]   = glx_reduce128((uint64_t)(p0 >> 64), (uint64_t)p0);
        out[i+1] = glx_reduce128((uint64_t)(p1 >> 64), (uint64_t)p1);
        out[i+2] = glx_reduce128((uint64_t)(p2 >> 64), (uint64_t)p2);
        out[i+3] = glx_reduce128((uint64_t)(p3 >> 64), (uint64_t)p3);
    }
    for (; i < n; i++)
        out[i] = glx_mul(a[i], scalar);
}

// ============================================================
// Inner product: result = sum(a[i] * b[i]) mod p
// ============================================================

uint64_t gl_inner_product_neon(const uint64_t *a, const uint64_t *b, int n) {
    uint64_t acc = 0;
    int i = 0;
    // Use interleaved scalar muls for ILP
    for (; i + 3 < n; i += 4) {
        uint64_t p0 = glx_mul(a[i],   b[i]);
        uint64_t p1 = glx_mul(a[i+1], b[i+1]);
        uint64_t p2 = glx_mul(a[i+2], b[i+2]);
        uint64_t p3 = glx_mul(a[i+3], b[i+3]);
        // Accumulate with NEON add
        uint64x2_t va = {p0, p1};
        uint64x2_t vb = {p2, p3};
        uint64x2_t vsum = glx_add_vec(va, vb);
        uint64x2_t vacc = {acc, 0};
        vacc = glx_add_vec(vacc, vsum);
        acc = glx_add(vgetq_lane_u64(vacc, 0), vgetq_lane_u64(vacc, 1));
    }
    for (; i < n; i++)
        acc = glx_add(acc, glx_mul(a[i], b[i]));
    return acc;
}

// ============================================================
// Linear combination: out[i] = alpha * a[i] + beta * b[i]
// ============================================================

void gl_linear_combine_neon(const uint64_t *a, const uint64_t *b,
                             uint64_t alpha, uint64_t beta,
                             uint64_t *out, int n) {
    int i = 0;
    for (; i + 1 < n; i += 2) {
        uint64_t ta0 = glx_mul(a[i],   alpha);
        uint64_t ta1 = glx_mul(a[i+1], alpha);
        uint64_t tb0 = glx_mul(b[i],   beta);
        uint64_t tb1 = glx_mul(b[i+1], beta);
        uint64x2_t va = {ta0, ta1};
        uint64x2_t vb = {tb0, tb1};
        vst1q_u64(out + i, glx_add_vec(va, vb));
    }
    if (i < n)
        out[i] = glx_add(glx_mul(a[i], alpha), glx_mul(b[i], beta));
}

// ============================================================
// Batch negate: out[i] = -a[i] mod p
// ============================================================

void gl_batch_neg_neon(const uint64_t *a, uint64_t *out, int n) {
    int i = 0;
    uint64x2_t p_vec = glx_p_vec();
    uint64x2_t zero = vdupq_n_u64(0);
    for (; i + 1 < n; i += 2) {
        uint64x2_t va = vld1q_u64(a + i);
        uint64x2_t neg = vsubq_u64(p_vec, va);
        // If a == 0, result should be 0
        uint64x2_t is_zero = vceqq_u64(va, zero);
        neg = vbicq_u64(neg, is_zero);
        vst1q_u64(out + i, neg);
    }
    if (i < n)
        out[i] = a[i] == 0 ? 0 : GL_P - a[i];
}

// ============================================================
// Goldilocks Fp2 = Fp[w]/(w^2 - 7)
// Element: a0 + a1*w, stored as [a0, a1] (2 x uint64_t)
// ============================================================

void gl_ext2_add_neon(const uint64_t a[2], const uint64_t b[2], uint64_t r[2]) {
    uint64x2_t va = vld1q_u64(a);
    uint64x2_t vb = vld1q_u64(b);
    vst1q_u64(r, glx_add_vec(va, vb));
}

void gl_ext2_sub_neon(const uint64_t a[2], const uint64_t b[2], uint64_t r[2]) {
    uint64x2_t va = vld1q_u64(a);
    uint64x2_t vb = vld1q_u64(b);
    vst1q_u64(r, glx_sub_vec(va, vb));
}

// Mul in Fp2 = Fp[w]/(w^2 - 7):
// (a0 + a1*w)(b0 + b1*w) = (a0*b0 + 7*a1*b1) + (a0*b1 + a1*b0)*w
void gl_ext2_mul_neon(const uint64_t a[2], const uint64_t b[2], uint64_t r[2]) {
    uint64_t a0b0 = glx_mul(a[0], b[0]);
    uint64_t a1b1 = glx_mul(a[1], b[1]);
    uint64_t a0b1 = glx_mul(a[0], b[1]);
    uint64_t a1b0 = glx_mul(a[1], b[0]);

    r[0] = glx_add(a0b0, glx_mul(7, a1b1));
    r[1] = glx_add(a0b1, a1b0);
}

// Square in Fp2: (a0+a1*w)^2 = (a0^2 + 7*a1^2) + 2*a0*a1*w
void gl_ext2_sqr_neon(const uint64_t a[2], uint64_t r[2]) {
    uint64_t a0sq = glx_mul(a[0], a[0]);
    uint64_t a1sq = glx_mul(a[1], a[1]);
    uint64_t a0a1 = glx_mul(a[0], a[1]);

    r[0] = glx_add(a0sq, glx_mul(7, a1sq));
    r[1] = glx_add(a0a1, a0a1);  // 2 * a0*a1
}

// Batch Fp2 mul: out[i] = a[i] * b[i] in Fp2
void gl_ext2_batch_mul_neon(const uint64_t *a, const uint64_t *b, uint64_t *out, int n) {
    for (int i = 0; i < n; i++) {
        gl_ext2_mul_neon(a + 2*i, b + 2*i, out + 2*i);
    }
}
