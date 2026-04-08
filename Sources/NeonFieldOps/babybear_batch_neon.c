// BabyBear NEON batch field operations (beyond NTT)
// p = 0x78000001 = 2013265921 (31-bit prime)
//
// Extends babybear_ntt.c with general-purpose batch operations:
// - batch add/sub/mul/neg/scale
// - inner product
// - linear combination
// All use the Plonky3-style Montgomery multiply (7 NEON instructions per 4 muls).

#include "NeonFieldOps.h"
#include <arm_neon.h>
#include <string.h>

// ============================================================
// Constants (must match babybear_ntt.c)
// ============================================================

#define BB_P            2013265921u
#define BB_P_S          2013265921
#define BB_2P           4026531842u
#define BB_R2_MOD_P     1172168163u   // R^2 mod p (R = 2^32)
#define BB_P_INV        2281701377u   // p^{-1} mod 2^32
#define BB_P_INV_S      ((int32_t)BB_P_INV)

// ============================================================
// Scalar Montgomery arithmetic
// ============================================================

static inline uint32_t bb_monty_reduce64(uint64_t x) {
    uint32_t lo = (uint32_t)x;
    uint32_t q = lo * BB_P_INV;
    int64_t t = (int64_t)x - (int64_t)q * (int64_t)BB_P;
    int32_t r = (int32_t)(t >> 32);
    return r < 0 ? (uint32_t)(r + BB_P_S) : (uint32_t)r;
}

static inline uint32_t bb_monty_mul(uint32_t a, uint32_t b) {
    return bb_monty_reduce64((uint64_t)a * (uint64_t)b);
}

static inline uint32_t bb_monty_add(uint32_t a, uint32_t b) {
    uint32_t s = a + b;
    return s >= BB_P ? s - BB_P : s;
}

static inline uint32_t bb_monty_sub(uint32_t a, uint32_t b) {
    return a >= b ? a - b : a + BB_P - b;
}

static inline uint32_t bb_to_monty(uint32_t a) {
    return bb_monty_mul(a, BB_R2_MOD_P);
}

static inline uint32_t bb_from_monty(uint32_t a) {
    return bb_monty_reduce64((uint64_t)a);
}

// ============================================================
// NEON 4-wide Montgomery multiply (Plonky3 technique)
// ============================================================

static inline int32x4_t bb_monty_mul_neon(int32x4_t a, int32x4_t b,
                                           int32x4_t p_inv_v, int32x4_t p_vec) {
    int32x4_t prod_lo = vmulq_s32(a, b);
    int32x4_t q = vmulq_s32(prod_lo, p_inv_v);
    int32x4_t ab_hi = vqdmulhq_s32(a, b);
    int32x4_t qp_hi = vqdmulhq_s32(q, p_vec);
    int32x4_t r = vhsubq_s32(ab_hi, qp_hi);
    int32x4_t mask = vshrq_n_s32(r, 31);
    return vaddq_s32(r, vandq_s32(mask, p_vec));
}

static inline int32x4_t bb_monty_add_neon(int32x4_t a, int32x4_t b, int32x4_t p_vec) {
    int32x4_t s = vaddq_s32(a, b);
    int32x4_t reduced = vsubq_s32(s, p_vec);
    int32x4_t mask = vshrq_n_s32(reduced, 31);
    return vbslq_s32(vreinterpretq_u32_s32(mask), s, reduced);
}

static inline int32x4_t bb_monty_sub_neon(int32x4_t a, int32x4_t b, int32x4_t p_vec) {
    int32x4_t diff = vsubq_s32(a, b);
    int32x4_t mask = vshrq_n_s32(diff, 31);
    return vaddq_s32(diff, vandq_s32(mask, p_vec));
}

// ============================================================
// Batch operations (elements in Montgomery form)
// ============================================================

void bb_batch_add_neon(const uint32_t *a, const uint32_t *b, uint32_t *out, int n) {
    int32x4_t pv = vdupq_n_s32(BB_P_S);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        int32x4_t va = vld1q_s32((const int32_t *)(a + i));
        int32x4_t vb = vld1q_s32((const int32_t *)(b + i));
        vst1q_s32((int32_t *)(out + i), bb_monty_add_neon(va, vb, pv));
    }
    for (; i < n; i++)
        out[i] = bb_monty_add(a[i], b[i]);
}

void bb_batch_sub_neon(const uint32_t *a, const uint32_t *b, uint32_t *out, int n) {
    int32x4_t pv = vdupq_n_s32(BB_P_S);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        int32x4_t va = vld1q_s32((const int32_t *)(a + i));
        int32x4_t vb = vld1q_s32((const int32_t *)(b + i));
        vst1q_s32((int32_t *)(out + i), bb_monty_sub_neon(va, vb, pv));
    }
    for (; i < n; i++)
        out[i] = bb_monty_sub(a[i], b[i]);
}

void bb_batch_mul_neon(const uint32_t *a, const uint32_t *b, uint32_t *out, int n) {
    int32x4_t p_inv = vdupq_n_s32(BB_P_INV_S);
    int32x4_t pv = vdupq_n_s32(BB_P_S);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        int32x4_t va = vld1q_s32((const int32_t *)(a + i));
        int32x4_t vb = vld1q_s32((const int32_t *)(b + i));
        vst1q_s32((int32_t *)(out + i), bb_monty_mul_neon(va, vb, p_inv, pv));
    }
    for (; i < n; i++)
        out[i] = bb_monty_mul(a[i], b[i]);
}

void bb_batch_neg_neon(const uint32_t *a, uint32_t *out, int n) {
    int32x4_t pv = vdupq_n_s32(BB_P_S);
    int32x4_t zero = vdupq_n_s32(0);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        int32x4_t va = vld1q_s32((const int32_t *)(a + i));
        // -a = p - a if a != 0, else 0
        int32x4_t neg = vsubq_s32(pv, va);
        uint32x4_t is_zero = vceqq_s32(va, zero);
        neg = vbicq_s32(neg, vreinterpretq_s32_u32(is_zero));
        vst1q_s32((int32_t *)(out + i), neg);
    }
    for (; i < n; i++)
        out[i] = a[i] == 0 ? 0 : BB_P - a[i];
}

void bb_batch_mul_scalar_neon(const uint32_t *a, uint32_t scalar, uint32_t *out, int n) {
    int32x4_t p_inv = vdupq_n_s32(BB_P_INV_S);
    int32x4_t pv = vdupq_n_s32(BB_P_S);
    int32x4_t s_vec = vdupq_n_s32((int32_t)scalar);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        int32x4_t va = vld1q_s32((const int32_t *)(a + i));
        vst1q_s32((int32_t *)(out + i), bb_monty_mul_neon(va, s_vec, p_inv, pv));
    }
    for (; i < n; i++)
        out[i] = bb_monty_mul(a[i], scalar);
}

// Inner product: sum(a[i] * b[i]) in Montgomery form
// Accumulate in 64-bit to avoid overflow, reduce periodically
uint32_t bb_inner_product_neon(const uint32_t *a, const uint32_t *b, int n) {
    int32x4_t p_inv = vdupq_n_s32(BB_P_INV_S);
    int32x4_t pv = vdupq_n_s32(BB_P_S);

    // Products are in [0, p), accumulate in scalar (64-bit)
    uint64_t acc = 0;
    int i = 0;
    for (; i + 3 < n; i += 4) {
        int32x4_t va = vld1q_s32((const int32_t *)(a + i));
        int32x4_t vb = vld1q_s32((const int32_t *)(b + i));
        int32x4_t prod = bb_monty_mul_neon(va, vb, p_inv, pv);
        // Horizontal accumulate (treat as unsigned)
        uint32x4_t uprod = vreinterpretq_u32_s32(prod);
        acc += (uint64_t)vgetq_lane_u32(uprod, 0);
        acc += (uint64_t)vgetq_lane_u32(uprod, 1);
        acc += (uint64_t)vgetq_lane_u32(uprod, 2);
        acc += (uint64_t)vgetq_lane_u32(uprod, 3);

        // Reduce when approaching overflow (~every 2^32 / p ~ 2 iterations)
        if ((i & 0x7FF) == 0 && i > 0) {
            acc = acc % BB_P;
        }
    }
    for (; i < n; i++)
        acc += (uint64_t)bb_monty_mul(a[i], b[i]);

    return (uint32_t)(acc % BB_P);
}

// Linear combination: out[i] = alpha * a[i] + beta * b[i]
// All values in Montgomery form
void bb_linear_combine_neon(const uint32_t *a, const uint32_t *b,
                             uint32_t alpha, uint32_t beta,
                             uint32_t *out, int n) {
    int32x4_t p_inv = vdupq_n_s32(BB_P_INV_S);
    int32x4_t pv = vdupq_n_s32(BB_P_S);
    int32x4_t alpha_v = vdupq_n_s32((int32_t)alpha);
    int32x4_t beta_v = vdupq_n_s32((int32_t)beta);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        int32x4_t va = vld1q_s32((const int32_t *)(a + i));
        int32x4_t vb = vld1q_s32((const int32_t *)(b + i));
        int32x4_t ta = bb_monty_mul_neon(va, alpha_v, p_inv, pv);
        int32x4_t tb = bb_monty_mul_neon(vb, beta_v, p_inv, pv);
        vst1q_s32((int32_t *)(out + i), bb_monty_add_neon(ta, tb, pv));
    }
    for (; i < n; i++) {
        uint32_t ta = bb_monty_mul(a[i], alpha);
        uint32_t tb = bb_monty_mul(b[i], beta);
        out[i] = bb_monty_add(ta, tb);
    }
}

// Convert batch to/from Montgomery form
void bb_batch_to_monty_neon(const uint32_t *in, uint32_t *out, int n) {
    int32x4_t p_inv = vdupq_n_s32(BB_P_INV_S);
    int32x4_t pv = vdupq_n_s32(BB_P_S);
    int32x4_t r2_vec = vdupq_n_s32((int32_t)BB_R2_MOD_P);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        int32x4_t v = vld1q_s32((const int32_t *)(in + i));
        vst1q_s32((int32_t *)(out + i), bb_monty_mul_neon(v, r2_vec, p_inv, pv));
    }
    for (; i < n; i++)
        out[i] = bb_to_monty(in[i]);
}

void bb_batch_from_monty_neon(const uint32_t *in, uint32_t *out, int n) {
    int32x4_t p_inv = vdupq_n_s32(BB_P_INV_S);
    int32x4_t pv = vdupq_n_s32(BB_P_S);
    int32x4_t ones = vdupq_n_s32(1);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        int32x4_t v = vld1q_s32((const int32_t *)(in + i));
        vst1q_s32((int32_t *)(out + i), bb_monty_mul_neon(v, ones, p_inv, pv));
    }
    for (; i < n; i++)
        out[i] = bb_from_monty(in[i]);
}

// ============================================================
// Standard-form (non-Montgomery) scalar arithmetic
// For use with Swift's Bb which stores values in standard form.
// ============================================================

static inline uint32_t bb_std_mul(uint32_t a, uint32_t b) {
    return (uint32_t)((uint64_t)a * (uint64_t)b % BB_P);
}

static inline uint32_t bb_std_add(uint32_t a, uint32_t b) {
    uint32_t s = a + b;
    return s >= BB_P ? s - BB_P : s;
}

static inline uint32_t bb_std_sub(uint32_t a, uint32_t b) {
    return a >= b ? a - b : a + BB_P - b;
}

static inline uint32_t bb_std_pow(uint32_t base, uint32_t exp) {
    uint32_t result = 1;
    while (exp > 0) {
        if (exp & 1) result = bb_std_mul(result, base);
        base = bb_std_mul(base, base);
        exp >>= 1;
    }
    return result;
}

static inline uint32_t bb_std_inv(uint32_t a) {
    return bb_std_pow(a, BB_P - 2);
}

// ============================================================
// Batch inversion (standard form) via Montgomery's trick
// out[i] = 1/a[i] mod p. Zero elements get zero output.
// ============================================================

void bb_batch_inverse(const uint32_t *a, uint32_t *out, int n) {
    if (n == 0) return;
    if (n == 1) { out[0] = a[0] == 0 ? 0 : bb_std_inv(a[0]); return; }

    // Use out[] as scratch for prefix products (safe since we overwrite in backward pass)
    out[0] = a[0] == 0 ? 1 : a[0];
    for (int i = 1; i < n; i++) {
        out[i] = a[i] == 0 ? out[i - 1] : bb_std_mul(out[i - 1], a[i]);
    }

    uint32_t inv = bb_std_inv(out[n - 1]);

    for (int i = n - 1; i > 0; i--) {
        if (a[i] == 0) {
            out[i] = 0;
        } else {
            out[i] = bb_std_mul(inv, out[i - 1]);
            inv = bb_std_mul(inv, a[i]);
        }
    }
    out[0] = a[0] == 0 ? 0 : inv;
}

// ============================================================
// FRI fold (standard form):
// out[i] = (f[i]+f[i+half])*inv2 + beta*(f[i]-f[i+half])*invDenom[i]
// ============================================================

void bb_fri_fold(const uint32_t *f, const uint32_t *invDenom,
                 uint32_t inv2, uint32_t beta,
                 uint32_t *out, int half) {
    for (int i = 0; i < half; i++) {
        uint32_t f0 = f[i], f1 = f[i + half];
        uint32_t even = bb_std_mul(bb_std_add(f0, f1), inv2);
        uint32_t odd = bb_std_mul(bb_std_sub(f0, f1), invDenom[i]);
        out[i] = bb_std_add(even, bb_std_mul(beta, odd));
    }
}

// ============================================================
// Vanishing polynomial (standard form):
// out[i] = base * gen^i - one
// ============================================================

void bb_vanishing_poly(uint32_t base, uint32_t gen, uint32_t one_val,
                       uint32_t *out, int n) {
    uint32_t acc = base;
    for (int i = 0; i < n; i++) {
        out[i] = bb_std_sub(acc, one_val);
        acc = bb_std_mul(acc, gen);
    }
}
