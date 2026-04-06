// NEON-accelerated NTT for lattice-based cryptography (Kyber/Dilithium)
//
// Kyber:     q = 3329    (12-bit prime, n=256)
// Dilithium: q = 8380417 (23-bit prime, n=256)
//
// Two acceleration strategies:
//   1. Single-polynomial: vectorized butterfly — 4 adjacent butterflies per NEON op
//   2. Batch-4: 4 independent NTTs in 4 NEON lanes (interleaved layout)
//
// Barrett reduction avoids division:
//   reduce(x) = x - floor(x * m >> s) * q
// For Kyber:     m = 5039,     s = 24  (handles products up to 3328^2 = ~11M < 2^24)
// For Dilithium: m = 33587228, s = 48  (handles products up to 8380416^2 < 2^47)

#include "NeonFieldOps.h"
#include <arm_neon.h>
#include <stdlib.h>
#include <string.h>

// ============================================================
// Kyber constants (q = 3329)
// ============================================================

#define KYBER_Q         3329u
#define KYBER_BARRETT_M 5039u      // floor(2^24 / 3329)
#define KYBER_BARRETT_S 24

// Primitive 256th root of unity for Kyber: zeta = 17
// 17^128 = -1 (mod 3329), so 17 has order 256 in Z_3329*.
#define KYBER_ZETA      17u

// ============================================================
// Dilithium constants (q = 8380417)
// ============================================================

#define DIL_Q           8380417u
// Barrett: m = floor(2^48 / 8380417) = 33587228
#define DIL_BARRETT_M   33587228ULL
#define DIL_BARRETT_S   48

// Primitive 256th root of unity for Dilithium: 3073009
// (as defined in existing LatticeFields.swift: ZETA = 3073009)
#define DIL_ZETA        3073009u

// ============================================================
// Scalar Barrett reduction — Kyber
// ============================================================

static inline uint32_t kyber_reduce(uint32_t x) {
    // x < 3329^2 ~ 11.08M < 2^24
    uint32_t t = (uint32_t)(((uint64_t)x * KYBER_BARRETT_M) >> KYBER_BARRETT_S);
    int32_t r = (int32_t)x - (int32_t)(t * KYBER_Q);
    if (r >= (int32_t)KYBER_Q) r -= (int32_t)KYBER_Q;
    if (r < 0) r += (int32_t)KYBER_Q;
    return (uint32_t)r;
}

static inline uint32_t kyber_mul(uint32_t a, uint32_t b) {
    return kyber_reduce(a * b);
}

static inline uint32_t kyber_add(uint32_t a, uint32_t b) {
    uint32_t s = a + b;
    return s >= KYBER_Q ? s - KYBER_Q : s;
}

static inline uint32_t kyber_sub(uint32_t a, uint32_t b) {
    return a >= b ? a - b : a + KYBER_Q - b;
}

// ============================================================
// Scalar Barrett reduction — Dilithium
// ============================================================

static inline uint32_t dil_reduce(uint64_t x) {
    // x < 8380417^2 ~ 7.02 * 10^13 < 2^47
    // Barrett: t = floor(x * M / 2^48), r = x - t*q
    uint64_t t = (uint64_t)((unsigned __int128)x * DIL_BARRETT_M >> DIL_BARRETT_S);
    int64_t r = (int64_t)x - (int64_t)(t * (uint64_t)DIL_Q);
    if (r >= (int64_t)DIL_Q) r -= (int64_t)DIL_Q;
    if (r < 0) r += (int64_t)DIL_Q;
    return (uint32_t)r;
}

static inline uint32_t dil_mul(uint32_t a, uint32_t b) {
    return dil_reduce((uint64_t)a * (uint64_t)b);
}

static inline uint32_t dil_add(uint32_t a, uint32_t b) {
    uint32_t s = a + b;
    return s >= DIL_Q ? s - DIL_Q : s;
}

static inline uint32_t dil_sub(uint32_t a, uint32_t b) {
    return a >= b ? a - b : a + DIL_Q - b;
}

// ============================================================
// Scalar modular exponentiation
// ============================================================

static inline uint32_t kyber_pow(uint32_t base, uint32_t exp) {
    uint32_t result = 1;
    uint32_t b = base % KYBER_Q;
    while (exp > 0) {
        if (exp & 1) result = kyber_mul(result, b);
        b = kyber_mul(b, b);
        exp >>= 1;
    }
    return result;
}

static inline uint32_t dil_pow(uint32_t base, uint32_t exp) {
    uint32_t result = 1;
    uint32_t b = base % DIL_Q;
    while (exp > 0) {
        if (exp & 1) result = dil_mul(result, b);
        b = dil_mul(b, b);
        exp >>= 1;
    }
    return result;
}

// ============================================================
// NEON 4-wide Barrett reduction — Kyber (32-bit lanes)
// ============================================================
// Input: 4 products in uint32x4_t, each < 3329^2 ~ 11M
// Output: 4 reduced values in [0, q)

static inline uint32x4_t kyber_barrett_neon(uint32x4_t x) {
    uint32x4_t m_vec = vdupq_n_u32(KYBER_BARRETT_M);
    uint32x4_t q_vec = vdupq_n_u32(KYBER_Q);

    // Compute x*m via widening multiply (2 lanes at a time -> uint64)
    uint32x2_t x_lo = vget_low_u32(x);
    uint32x2_t x_hi = vget_high_u32(x);
    uint32x2_t m_lo = vget_low_u32(m_vec);

    uint64x2_t prod_lo = vmull_u32(x_lo, m_lo);
    uint64x2_t prod_hi = vmull_u32(x_hi, vget_high_u32(m_vec));

    // t = prod >> 24
    uint32x2_t t_lo = vmovn_u64(vshrq_n_u64(prod_lo, KYBER_BARRETT_S));
    uint32x2_t t_hi = vmovn_u64(vshrq_n_u64(prod_hi, KYBER_BARRETT_S));
    uint32x4_t t = vcombine_u32(t_lo, t_hi);

    // r = x - t * q
    uint32x4_t tq = vmulq_u32(t, q_vec);
    uint32x4_t r = vsubq_u32(x, tq);

    // Conditional subtract: if r >= q, r -= q
    uint32x4_t mask = vcgeq_u32(r, q_vec);
    r = vsubq_u32(r, vandq_u32(mask, q_vec));

    return r;
}

// NEON 4-wide modular multiply for Kyber
static inline uint32x4_t kyber_mul_neon(uint32x4_t a, uint32x4_t b) {
    // Product fits in 32 bits since a,b < 3329 and 3329*3329 = 11082241 < 2^24
    uint32x4_t prod = vmulq_u32(a, b);
    return kyber_barrett_neon(prod);
}

// NEON 4-wide modular add for Kyber
static inline uint32x4_t kyber_add_neon(uint32x4_t a, uint32x4_t b) {
    uint32x4_t q_vec = vdupq_n_u32(KYBER_Q);
    uint32x4_t s = vaddq_u32(a, b);
    uint32x4_t mask = vcgeq_u32(s, q_vec);
    return vsubq_u32(s, vandq_u32(mask, q_vec));
}

// NEON 4-wide modular sub for Kyber
static inline uint32x4_t kyber_sub_neon(uint32x4_t a, uint32x4_t b) {
    uint32x4_t q_vec = vdupq_n_u32(KYBER_Q);
    uint32x4_t diff = vsubq_u32(a, b);
    uint32x4_t wrap = vaddq_u32(diff, q_vec);
    uint32x4_t mask = vcgeq_u32(a, b);
    return vbslq_u32(mask, diff, wrap);
}

// ============================================================
// NEON butterfly for Kyber — Cooley-Tukey (DIT)
// ============================================================

static inline void kyber_ct_butterfly_neon(uint32x4_t *u, uint32x4_t *v, uint32x4_t tw) {
    uint32x4_t t = kyber_mul_neon(tw, *v);
    uint32x4_t new_u = kyber_add_neon(*u, t);
    uint32x4_t new_v = kyber_sub_neon(*u, t);
    *u = new_u;
    *v = new_v;
}

// Gentleman-Sande (DIF) butterfly for INTT:
//   sum = u + v
//   diff = (v - u) * tw   (note: v - u, matching reference Kyber INTT)
static inline void kyber_gs_butterfly_neon(uint32x4_t *u, uint32x4_t *v, uint32x4_t tw) {
    uint32x4_t sum = kyber_add_neon(*u, *v);
    uint32x4_t diff = kyber_sub_neon(*v, *u);
    *u = sum;
    *v = kyber_mul_neon(diff, tw);
}

// ============================================================
// Bit-reverse for 8-bit value, then shift to get 7-bit reversal
// ============================================================

static inline uint8_t bitrev7(uint8_t x) {
    uint8_t v = x;
    v = ((v & 0x55) << 1) | ((v >> 1) & 0x55);
    v = ((v & 0x33) << 2) | ((v >> 2) & 0x33);
    v = ((v & 0x0F) << 4) | ((v >> 4) & 0x0F);
    return v >> 1;
}

// ============================================================
// Kyber twiddle table (128 entries, bit-reversed powers of zeta)
// ============================================================

static uint32_t kyber_twiddles[128];
static int kyber_twiddles_init = 0;

static void kyber_init_twiddles(void) {
    if (kyber_twiddles_init) return;
    uint32_t powers[256];
    powers[0] = 1;
    for (int i = 1; i < 256; i++)
        powers[i] = kyber_mul(powers[i-1], KYBER_ZETA);

    for (int i = 0; i < 128; i++)
        kyber_twiddles[i] = powers[bitrev7((uint8_t)i)];

    kyber_twiddles_init = 1;
}

// ============================================================
// Kyber forward NTT (single polynomial, NEON-accelerated)
// ============================================================

void lattice_ntt_kyber_neon(uint32_t *poly) {
    kyber_init_twiddles();

    int k = 1;
    int len = 128;
    while (len >= 2) {
        int start = 0;
        while (start < 256) {
            uint32_t tw = kyber_twiddles[k];
            k++;

            if (len >= 4) {
                uint32x4_t tw_vec = vdupq_n_u32(tw);
                int j = start;
                for (; j + 3 < start + len; j += 4) {
                    uint32x4_t u = vld1q_u32(&poly[j]);
                    uint32x4_t v = vld1q_u32(&poly[j + len]);
                    kyber_ct_butterfly_neon(&u, &v, tw_vec);
                    vst1q_u32(&poly[j], u);
                    vst1q_u32(&poly[j + len], v);
                }
                for (; j < start + len; j++) {
                    uint32_t t = kyber_mul(tw, poly[j + len]);
                    uint32_t u = poly[j];
                    poly[j] = kyber_add(u, t);
                    poly[j + len] = kyber_sub(u, t);
                }
            } else {
                for (int j = start; j < start + len; j++) {
                    uint32_t t = kyber_mul(tw, poly[j + len]);
                    uint32_t u = poly[j];
                    poly[j] = kyber_add(u, t);
                    poly[j + len] = kyber_sub(u, t);
                }
            }
            start += 2 * len;
        }
        len >>= 1;
    }
}

// ============================================================
// Kyber inverse NTT (single polynomial, NEON-accelerated)
// ============================================================

void lattice_intt_kyber_neon(uint32_t *poly) {
    kyber_init_twiddles();

    int k = 127;
    int len = 2;
    while (len <= 128) {
        int start = 0;
        while (start < 256) {
            uint32_t tw = kyber_twiddles[k];
            k--;

            if (len >= 4) {
                uint32x4_t tw_vec = vdupq_n_u32(tw);
                int j = start;
                for (; j + 3 < start + len; j += 4) {
                    uint32x4_t u = vld1q_u32(&poly[j]);
                    uint32x4_t v = vld1q_u32(&poly[j + len]);
                    kyber_gs_butterfly_neon(&u, &v, tw_vec);
                    vst1q_u32(&poly[j], u);
                    vst1q_u32(&poly[j + len], v);
                }
                for (; j < start + len; j++) {
                    uint32_t t = poly[j];
                    uint32_t s = kyber_add(t, poly[j + len]);
                    uint32_t d = kyber_mul(tw, kyber_sub(poly[j + len], t));
                    poly[j] = s;
                    poly[j + len] = d;
                }
            } else {
                for (int j = start; j < start + len; j++) {
                    uint32_t t = poly[j];
                    poly[j] = kyber_add(t, poly[j + len]);
                    poly[j + len] = kyber_mul(tw, kyber_sub(poly[j + len], t));
                }
            }
            start += 2 * len;
        }
        len <<= 1;
    }

    // Scale by 1/128 mod q
    uint32_t inv128 = kyber_pow(128, KYBER_Q - 2);
    uint32x4_t inv_vec = vdupq_n_u32(inv128);
    int i = 0;
    for (; i + 3 < 256; i += 4) {
        uint32x4_t v = vld1q_u32(&poly[i]);
        v = kyber_mul_neon(v, inv_vec);
        vst1q_u32(&poly[i], v);
    }
    for (; i < 256; i++)
        poly[i] = kyber_mul(poly[i], inv128);
}

// ============================================================
// Dilithium twiddle table
// ============================================================

static uint32_t dil_twiddles[128];
static int dil_twiddles_init = 0;

static void dil_init_twiddles(void) {
    if (dil_twiddles_init) return;
    uint32_t powers[256];
    powers[0] = 1;
    for (int i = 1; i < 256; i++)
        powers[i] = dil_mul(powers[i-1], DIL_ZETA);

    for (int i = 0; i < 128; i++)
        dil_twiddles[i] = powers[bitrev7((uint8_t)i)];

    dil_twiddles_init = 1;
}

// ============================================================
// Dilithium NEON operations
// Products need 64-bit intermediates (23-bit * 23-bit = 46 bits).
// Use vmull_u32 for 2 lanes at a time, then Barrett reduce.
// ============================================================

static inline uint32x4_t dil_mul_neon(uint32x4_t a, uint32x4_t b) {
    // Process as two pairs via vmull_u32 -> uint64x2
    uint32x2_t a_lo = vget_low_u32(a);
    uint32x2_t a_hi = vget_high_u32(a);
    uint32x2_t b_lo = vget_low_u32(b);
    uint32x2_t b_hi = vget_high_u32(b);

    uint64x2_t prod0 = vmull_u32(a_lo, b_lo);
    uint64x2_t prod1 = vmull_u32(a_hi, b_hi);

    // Barrett reduce each 64-bit product via 128-bit multiply
    uint64_t p0 = vgetq_lane_u64(prod0, 0);
    uint64_t p1 = vgetq_lane_u64(prod0, 1);
    uint64_t p2 = vgetq_lane_u64(prod1, 0);
    uint64_t p3 = vgetq_lane_u64(prod1, 1);

    uint32_t r0 = dil_reduce(p0);
    uint32_t r1 = dil_reduce(p1);
    uint32_t r2 = dil_reduce(p2);
    uint32_t r3 = dil_reduce(p3);

    uint32_t arr[4] = {r0, r1, r2, r3};
    return vld1q_u32(arr);
}

// Dilithium NEON add/sub
static inline uint32x4_t dil_add_neon(uint32x4_t a, uint32x4_t b) {
    uint32x4_t q_vec = vdupq_n_u32(DIL_Q);
    uint32x4_t s = vaddq_u32(a, b);
    uint32x4_t mask = vcgeq_u32(s, q_vec);
    return vsubq_u32(s, vandq_u32(mask, q_vec));
}

static inline uint32x4_t dil_sub_neon(uint32x4_t a, uint32x4_t b) {
    uint32x4_t q_vec = vdupq_n_u32(DIL_Q);
    uint32x4_t diff = vsubq_u32(a, b);
    uint32x4_t wrap = vaddq_u32(diff, q_vec);
    uint32x4_t mask = vcgeq_u32(a, b);
    return vbslq_u32(mask, diff, wrap);
}

// Dilithium CT butterfly (forward NTT)
static inline void dil_ct_butterfly_neon(uint32x4_t *u, uint32x4_t *v, uint32x4_t tw) {
    uint32x4_t t = dil_mul_neon(tw, *v);
    uint32x4_t new_u = dil_add_neon(*u, t);
    uint32x4_t new_v = dil_sub_neon(*u, t);
    *u = new_u;
    *v = new_v;
}

// Dilithium GS butterfly (inverse NTT)
// diff = v - u (matching reference: poly[j+len] - t where t = poly[j])
static inline void dil_gs_butterfly_neon(uint32x4_t *u, uint32x4_t *v, uint32x4_t tw) {
    uint32x4_t sum = dil_add_neon(*u, *v);
    uint32x4_t diff = dil_sub_neon(*v, *u);
    *u = sum;
    *v = dil_mul_neon(diff, tw);
}

// ============================================================
// Dilithium forward NTT (single polynomial, NEON-accelerated)
// ============================================================

void lattice_ntt_dilithium_neon(uint32_t *poly) {
    dil_init_twiddles();

    int k = 1;
    int len = 128;
    while (len >= 2) {
        int start = 0;
        while (start < 256) {
            uint32_t tw = dil_twiddles[k];
            k++;

            if (len >= 4) {
                uint32x4_t tw_vec = vdupq_n_u32(tw);
                int j = start;
                for (; j + 3 < start + len; j += 4) {
                    uint32x4_t u = vld1q_u32(&poly[j]);
                    uint32x4_t v = vld1q_u32(&poly[j + len]);
                    dil_ct_butterfly_neon(&u, &v, tw_vec);
                    vst1q_u32(&poly[j], u);
                    vst1q_u32(&poly[j + len], v);
                }
                for (; j < start + len; j++) {
                    uint32_t t = dil_mul(tw, poly[j + len]);
                    uint32_t u = poly[j];
                    poly[j] = dil_add(u, t);
                    poly[j + len] = dil_sub(u, t);
                }
            } else {
                for (int j = start; j < start + len; j++) {
                    uint32_t t = dil_mul(tw, poly[j + len]);
                    uint32_t u = poly[j];
                    poly[j] = dil_add(u, t);
                    poly[j + len] = dil_sub(u, t);
                }
            }
            start += 2 * len;
        }
        len >>= 1;
    }
}

// ============================================================
// Dilithium inverse NTT (single polynomial, NEON-accelerated)
// ============================================================

void lattice_intt_dilithium_neon(uint32_t *poly) {
    dil_init_twiddles();

    int k = 127;
    int len = 2;
    while (len <= 128) {
        int start = 0;
        while (start < 256) {
            uint32_t tw = dil_twiddles[k];
            k--;

            if (len >= 4) {
                uint32x4_t tw_vec = vdupq_n_u32(tw);
                int j = start;
                for (; j + 3 < start + len; j += 4) {
                    uint32x4_t u = vld1q_u32(&poly[j]);
                    uint32x4_t v = vld1q_u32(&poly[j + len]);
                    dil_gs_butterfly_neon(&u, &v, tw_vec);
                    vst1q_u32(&poly[j], u);
                    vst1q_u32(&poly[j + len], v);
                }
                for (; j < start + len; j++) {
                    uint32_t t = poly[j];
                    poly[j] = dil_add(t, poly[j + len]);
                    poly[j + len] = dil_mul(tw, dil_sub(poly[j + len], t));
                }
            } else {
                for (int j = start; j < start + len; j++) {
                    uint32_t t = poly[j];
                    poly[j] = dil_add(t, poly[j + len]);
                    poly[j + len] = dil_mul(tw, dil_sub(poly[j + len], t));
                }
            }
            start += 2 * len;
        }
        len <<= 1;
    }

    // Scale by 1/128 mod q
    uint32_t inv128 = dil_pow(128, DIL_Q - 2);
    uint32x4_t inv_vec = vdupq_n_u32(inv128);
    int i = 0;
    for (; i + 3 < 256; i += 4) {
        uint32x4_t v = vld1q_u32(&poly[i]);
        v = dil_mul_neon(v, inv_vec);
        vst1q_u32(&poly[i], v);
    }
    for (; i < 256; i++)
        poly[i] = dil_mul(poly[i], inv128);
}

// ============================================================
// Batch-4 NTT: 4 independent NTTs in 4 NEON lanes
// Layout: interleaved — poly[i][j] stored at buf[j*4 + i]
// Each NEON load grabs coefficients from 4 different polynomials.
// ============================================================

void lattice_ntt_kyber_batch4(uint32_t *polys) {
    kyber_init_twiddles();

    int k = 1;
    int len = 128;
    while (len >= 2) {
        int start = 0;
        while (start < 256) {
            uint32_t tw = kyber_twiddles[k];
            k++;
            uint32x4_t tw_vec = vdupq_n_u32(tw);

            for (int j = start; j < start + len; j++) {
                uint32x4_t u = vld1q_u32(&polys[j * 4]);
                uint32x4_t v = vld1q_u32(&polys[(j + len) * 4]);
                kyber_ct_butterfly_neon(&u, &v, tw_vec);
                vst1q_u32(&polys[j * 4], u);
                vst1q_u32(&polys[(j + len) * 4], v);
            }
            start += 2 * len;
        }
        len >>= 1;
    }
}

void lattice_intt_kyber_batch4(uint32_t *polys) {
    kyber_init_twiddles();

    int k = 127;
    int len = 2;
    while (len <= 128) {
        int start = 0;
        while (start < 256) {
            uint32_t tw = kyber_twiddles[k];
            k--;
            uint32x4_t tw_vec = vdupq_n_u32(tw);

            for (int j = start; j < start + len; j++) {
                uint32x4_t u = vld1q_u32(&polys[j * 4]);
                uint32x4_t v = vld1q_u32(&polys[(j + len) * 4]);
                kyber_gs_butterfly_neon(&u, &v, tw_vec);
                vst1q_u32(&polys[j * 4], u);
                vst1q_u32(&polys[(j + len) * 4], v);
            }
            start += 2 * len;
        }
        len <<= 1;
    }

    uint32_t inv128 = kyber_pow(128, KYBER_Q - 2);
    uint32x4_t inv_vec = vdupq_n_u32(inv128);
    for (int j = 0; j < 256; j++) {
        uint32x4_t v = vld1q_u32(&polys[j * 4]);
        v = kyber_mul_neon(v, inv_vec);
        vst1q_u32(&polys[j * 4], v);
    }
}

void lattice_ntt_dilithium_batch4(uint32_t *polys) {
    dil_init_twiddles();

    int k = 1;
    int len = 128;
    while (len >= 2) {
        int start = 0;
        while (start < 256) {
            uint32_t tw = dil_twiddles[k];
            k++;
            uint32x4_t tw_vec = vdupq_n_u32(tw);

            for (int j = start; j < start + len; j++) {
                uint32x4_t u = vld1q_u32(&polys[j * 4]);
                uint32x4_t v = vld1q_u32(&polys[(j + len) * 4]);
                dil_ct_butterfly_neon(&u, &v, tw_vec);
                vst1q_u32(&polys[j * 4], u);
                vst1q_u32(&polys[(j + len) * 4], v);
            }
            start += 2 * len;
        }
        len >>= 1;
    }
}

void lattice_intt_dilithium_batch4(uint32_t *polys) {
    dil_init_twiddles();

    int k = 127;
    int len = 2;
    while (len <= 128) {
        int start = 0;
        while (start < 256) {
            uint32_t tw = dil_twiddles[k];
            k--;
            uint32x4_t tw_vec = vdupq_n_u32(tw);

            for (int j = start; j < start + len; j++) {
                uint32x4_t u = vld1q_u32(&polys[j * 4]);
                uint32x4_t v = vld1q_u32(&polys[(j + len) * 4]);
                dil_gs_butterfly_neon(&u, &v, tw_vec);
                vst1q_u32(&polys[j * 4], u);
                vst1q_u32(&polys[(j + len) * 4], v);
            }
            start += 2 * len;
        }
        len <<= 1;
    }

    uint32_t inv128 = dil_pow(128, DIL_Q - 2);
    uint32x4_t inv_vec = vdupq_n_u32(inv128);
    for (int j = 0; j < 256; j++) {
        uint32x4_t v = vld1q_u32(&polys[j * 4]);
        v = dil_mul_neon(v, inv_vec);
        vst1q_u32(&polys[j * 4], v);
    }
}

// ============================================================
// Helper: interleave/deinterleave for batch-4 API
// ============================================================

void lattice_interleave4(const uint32_t *p0, const uint32_t *p1,
                         const uint32_t *p2, const uint32_t *p3,
                         uint32_t *out) {
    for (int j = 0; j < 256; j++) {
        out[j * 4 + 0] = p0[j];
        out[j * 4 + 1] = p1[j];
        out[j * 4 + 2] = p2[j];
        out[j * 4 + 3] = p3[j];
    }
}

void lattice_deinterleave4(const uint32_t *interleaved,
                            uint32_t *p0, uint32_t *p1,
                            uint32_t *p2, uint32_t *p3) {
    for (int j = 0; j < 256; j++) {
        p0[j] = interleaved[j * 4 + 0];
        p1[j] = interleaved[j * 4 + 1];
        p2[j] = interleaved[j * 4 + 2];
        p3[j] = interleaved[j * 4 + 3];
    }
}
