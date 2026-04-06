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

// ============================================================
// Signed-coefficient APIs for standard PQC representations
// Kyber: int16_t coefficients in [0, q) (FIPS 203 canonical)
// Dilithium: int32_t coefficients in [0, q) (FIPS 204 canonical)
// These use vmull_s16 / vmull_s32 for signed NEON multiply.
// ============================================================

// --- Signed scalar helpers ---

static inline int16_t kyber_reduce_s(int32_t x) {
    // Barrett reduction for signed input, result in [0, q)
    int32_t t = (int32_t)(((int64_t)x * (int64_t)KYBER_BARRETT_M) >> KYBER_BARRETT_S);
    int32_t r = x - t * (int32_t)KYBER_Q;
    if (r >= (int32_t)KYBER_Q) r -= (int32_t)KYBER_Q;
    if (r < 0) r += (int32_t)KYBER_Q;
    return (int16_t)r;
}

static inline int16_t kyber_mul_s(int16_t a, int16_t b) {
    return kyber_reduce_s((int32_t)a * (int32_t)b);
}

static inline int16_t kyber_add_s(int16_t a, int16_t b) {
    int32_t s = (int32_t)a + (int32_t)b;
    if (s >= (int32_t)KYBER_Q) s -= (int32_t)KYBER_Q;
    if (s < 0) s += (int32_t)KYBER_Q;
    return (int16_t)s;
}

static inline int16_t kyber_sub_s(int16_t a, int16_t b) {
    int32_t d = (int32_t)a - (int32_t)b;
    if (d < 0) d += (int32_t)KYBER_Q;
    if (d >= (int32_t)KYBER_Q) d -= (int32_t)KYBER_Q;
    return (int16_t)d;
}

static inline int16_t kyber_pow_s(int16_t base, uint32_t exp) {
    int32_t result = 1;
    int32_t b = (int32_t)base;
    if (b < 0) b += KYBER_Q;
    while (exp > 0) {
        if (exp & 1) result = (int32_t)kyber_reduce_s(result * b);
        b = (int32_t)kyber_reduce_s(b * b);
        exp >>= 1;
    }
    return (int16_t)result;
}

static inline int32_t dil_reduce_s(int64_t x) {
    int64_t t = (x * (int64_t)DIL_BARRETT_M) >> DIL_BARRETT_S;
    int32_t r = (int32_t)(x - t * (int64_t)DIL_Q);
    if (r >= (int32_t)DIL_Q) r -= (int32_t)DIL_Q;
    if (r < 0) r += (int32_t)DIL_Q;
    return r;
}

static inline int32_t dil_mul_s(int32_t a, int32_t b) {
    return dil_reduce_s((int64_t)a * (int64_t)b);
}

static inline int32_t dil_add_s(int32_t a, int32_t b) {
    int64_t s = (int64_t)a + (int64_t)b;
    if (s >= (int64_t)DIL_Q) s -= (int64_t)DIL_Q;
    if (s < 0) s += (int64_t)DIL_Q;
    return (int32_t)s;
}

static inline int32_t dil_sub_s(int32_t a, int32_t b) {
    int64_t d = (int64_t)a - (int64_t)b;
    if (d < 0) d += (int64_t)DIL_Q;
    if (d >= (int64_t)DIL_Q) d -= (int64_t)DIL_Q;
    return (int32_t)d;
}

static inline int32_t dil_pow_s(int32_t base, uint32_t exp) {
    int64_t result = 1;
    int64_t b = (int64_t)base;
    if (b < 0) b += DIL_Q;
    while (exp > 0) {
        if (exp & 1) result = (int64_t)dil_reduce_s(result * b);
        b = (int64_t)dil_reduce_s(b * b);
        exp >>= 1;
    }
    return (int32_t)result;
}

// --- Signed twiddle tables (int16_t for Kyber, int32_t for Dilithium) ---

static int16_t kyber_twiddles_s16[128];
static int32_t dil_twiddles_s32[128];
static int kyber_twiddles_s16_init = 0;
static int dil_twiddles_s32_init = 0;

static void kyber_init_twiddles_s16(void) {
    if (kyber_twiddles_s16_init) return;
    int16_t powers[256];
    powers[0] = 1;
    for (int i = 1; i < 256; i++)
        powers[i] = kyber_reduce_s((int32_t)powers[i-1] * (int32_t)KYBER_ZETA);
    for (int i = 0; i < 128; i++)
        kyber_twiddles_s16[i] = powers[bitrev7((uint8_t)i)];
    __sync_synchronize();
    kyber_twiddles_s16_init = 1;
}

static void dil_init_twiddles_s32(void) {
    if (dil_twiddles_s32_init) return;
    int32_t powers[256];
    powers[0] = 1;
    for (int i = 1; i < 256; i++)
        powers[i] = dil_reduce_s((int64_t)powers[i-1] * (int64_t)DIL_ZETA);
    for (int i = 0; i < 128; i++)
        dil_twiddles_s32[i] = powers[bitrev7((uint8_t)i)];
    __sync_synchronize();
    dil_twiddles_s32_init = 1;
}

// ============================================================
// NEON vectorized Barrett for signed Kyber: int32x4_t -> int16x4_t
// Input: 4 signed 32-bit products. Output: 4 reduced int16 in [0, q).
// ============================================================

#ifdef __ARM_NEON
static inline int16x4_t kyber_barrett_neon_s(int32x4_t a) {
    // t = (a * BARRETT_M) >> 24, using 64-bit widening multiply
    int32x4_t m_vec = vdupq_n_s32((int32_t)KYBER_BARRETT_M);
    int64x2_t prod_lo = vmull_s32(vget_low_s32(a), vget_low_s32(m_vec));
    int64x2_t prod_hi = vmull_s32(vget_high_s32(a), vget_high_s32(m_vec));
    int32x2_t t_lo = vmovn_s64(vshrq_n_s64(prod_lo, KYBER_BARRETT_S));
    int32x2_t t_hi = vmovn_s64(vshrq_n_s64(prod_hi, KYBER_BARRETT_S));
    int32x4_t t = vcombine_s32(t_lo, t_hi);
    // r = a - t * q
    int32x4_t q_vec = vdupq_n_s32((int32_t)KYBER_Q);
    int32x4_t r = vsubq_s32(a, vmulq_s32(t, q_vec));
    // Conditional: if r >= q, r -= q; if r < 0, r += q
    uint32x4_t ge_q = vcgeq_s32(r, q_vec);
    uint32x4_t lt_0 = vcltq_s32(r, vdupq_n_s32(0));
    r = vsubq_s32(r, vandq_s32(vreinterpretq_s32_u32(ge_q), q_vec));
    r = vaddq_s32(r, vandq_s32(vreinterpretq_s32_u32(lt_0), q_vec));
    return vmovn_s32(r);
}
#endif

// ============================================================
// kyber_ntt_neon: Signed int16_t Kyber NTT with NEON vmull_s16
// Cooley-Tukey DIT, in-place, n = 2^logN (max 256)
// ============================================================

void kyber_ntt_neon(int16_t *data, int logN) {
    kyber_init_twiddles_s16();
    int n = 1 << logN;
    if (n > 256) n = 256;

    int k = 1;
    int len = n / 2;

#ifdef __ARM_NEON
    while (len >= 4) {
        for (int start = 0; start < n; start += 2 * len) {
            int16_t tw = kyber_twiddles_s16[k];
            k++;
            int j;
            for (j = start; j + 3 < start + len; j += 4) {
                int16x4_t a = vld1_s16(&data[j]);
                int16x4_t b = vld1_s16(&data[j + len]);
                // t = tw * b mod q using vmull_s16 -> int32x4_t
                int32x4_t prod = vmull_s16(vdup_n_s16(tw), b);
                int16x4_t t = kyber_barrett_neon_s(prod);
                // data[j] = (a + t) mod q
                int32x4_t sum = vaddl_s16(a, t);
                int32x4_t q_vec = vdupq_n_s32((int32_t)KYBER_Q);
                uint32x4_t ge_q = vcgeq_s32(sum, q_vec);
                sum = vsubq_s32(sum, vandq_s32(vreinterpretq_s32_u32(ge_q), q_vec));
                // data[j+len] = (a - t) mod q
                int32x4_t diff = vsubl_s16(a, t);
                uint32x4_t lt_0 = vcltq_s32(diff, vdupq_n_s32(0));
                diff = vaddq_s32(diff, vandq_s32(vreinterpretq_s32_u32(lt_0), q_vec));
                vst1_s16(&data[j], vmovn_s32(sum));
                vst1_s16(&data[j + len], vmovn_s32(diff));
            }
            for (; j < start + len; j++) {
                int16_t t = kyber_mul_s(tw, data[j + len]);
                int16_t u = data[j];
                data[j] = kyber_add_s(u, t);
                data[j + len] = kyber_sub_s(u, t);
            }
        }
        len >>= 1;
    }
    while (len >= 2) {
        for (int start = 0; start < n; start += 2 * len) {
            int16_t tw = kyber_twiddles_s16[k];
            k++;
            for (int j = start; j < start + len; j++) {
                int16_t t = kyber_mul_s(tw, data[j + len]);
                int16_t u = data[j];
                data[j] = kyber_add_s(u, t);
                data[j + len] = kyber_sub_s(u, t);
            }
        }
        len >>= 1;
    }
#else
    while (len >= 2) {
        for (int start = 0; start < n; start += 2 * len) {
            int16_t tw = kyber_twiddles_s16[k];
            k++;
            for (int j = start; j < start + len; j++) {
                int16_t t = kyber_mul_s(tw, data[j + len]);
                int16_t u = data[j];
                data[j] = kyber_add_s(u, t);
                data[j + len] = kyber_sub_s(u, t);
            }
        }
        len >>= 1;
    }
#endif
}

// ============================================================
// kyber_intt_neon: Signed int16_t Kyber INTT with NEON vmull_s16
// Gentleman-Sande DIF + 1/(n/2) scaling
// ============================================================

void kyber_intt_neon(int16_t *data, int logN) {
    kyber_init_twiddles_s16();
    int n = 1 << logN;
    if (n > 256) n = 256;

    int16_t inv_n = kyber_pow_s((int16_t)(n / 2), KYBER_Q - 2);

    int k = 127;
    int len = 2;

#ifdef __ARM_NEON
    // Scalar for small len
    while (len < 4 && len <= n / 2) {
        for (int start = 0; start < n; start += 2 * len) {
            int16_t tw = kyber_twiddles_s16[k];
            k--;
            for (int j = start; j < start + len; j++) {
                int16_t t_val = data[j];
                data[j] = kyber_add_s(t_val, data[j + len]);
                data[j + len] = kyber_mul_s(tw, kyber_sub_s(data[j + len], t_val));
            }
        }
        len <<= 1;
    }

    // NEON path for len >= 4
    while (len <= n / 2) {
        for (int start = 0; start < n; start += 2 * len) {
            int16_t tw = kyber_twiddles_s16[k];
            k--;
            int j;
            for (j = start; j + 3 < start + len; j += 4) {
                int16x4_t a = vld1_s16(&data[j]);
                int16x4_t b = vld1_s16(&data[j + len]);
                // sum = (a + b) mod q
                int32x4_t sum = vaddl_s16(a, b);
                int32x4_t q_vec = vdupq_n_s32((int32_t)KYBER_Q);
                uint32x4_t ge_q = vcgeq_s32(sum, q_vec);
                sum = vsubq_s32(sum, vandq_s32(vreinterpretq_s32_u32(ge_q), q_vec));
                // diff = (b - a) mod q
                int32x4_t diff = vsubl_s16(b, a);
                uint32x4_t lt_0 = vcltq_s32(diff, vdupq_n_s32(0));
                diff = vaddq_s32(diff, vandq_s32(vreinterpretq_s32_u32(lt_0), q_vec));
                // data[j+len] = tw * diff mod q
                int16x4_t diff16 = vmovn_s32(diff);
                int32x4_t prod = vmull_s16(vdup_n_s16(tw), diff16);
                int16x4_t reduced = kyber_barrett_neon_s(prod);
                vst1_s16(&data[j], vmovn_s32(sum));
                vst1_s16(&data[j + len], reduced);
            }
            for (; j < start + len; j++) {
                int16_t t_val = data[j];
                data[j] = kyber_add_s(t_val, data[j + len]);
                data[j + len] = kyber_mul_s(tw, kyber_sub_s(data[j + len], t_val));
            }
        }
        len <<= 1;
    }

    // Scale by inv_n using vmull_s16
    for (int i = 0; i + 3 < n; i += 4) {
        int16x4_t vals = vld1_s16(&data[i]);
        int32x4_t prod = vmull_s16(vdup_n_s16(inv_n), vals);
        int16x4_t reduced = kyber_barrett_neon_s(prod);
        vst1_s16(&data[i], reduced);
    }
    for (int i = (n & ~3); i < n; i++) {
        data[i] = kyber_mul_s(inv_n, data[i]);
    }
#else
    while (len <= n / 2) {
        for (int start = 0; start < n; start += 2 * len) {
            int16_t tw = kyber_twiddles_s16[k];
            k--;
            for (int j = start; j < start + len; j++) {
                int16_t t_val = data[j];
                data[j] = kyber_add_s(t_val, data[j + len]);
                data[j + len] = kyber_mul_s(tw, kyber_sub_s(data[j + len], t_val));
            }
        }
        len <<= 1;
    }
    for (int i = 0; i < n; i++) {
        data[i] = kyber_mul_s(inv_n, data[i]);
    }
#endif
}

// ============================================================
// dilithium_ntt_neon: Signed int32_t Dilithium NTT with NEON vmull_s32
// Cooley-Tukey DIT, in-place, n = 2^logN (max 256)
// ============================================================

void dilithium_ntt_neon(int32_t *data, int logN) {
    dil_init_twiddles_s32();
    int n = 1 << logN;
    if (n > 256) n = 256;

    int k = 1;
    int len = n / 2;

#ifdef __ARM_NEON
    while (len >= 4) {
        for (int start = 0; start < n; start += 2 * len) {
            int32_t tw = dil_twiddles_s32[k];
            k++;
            int j;
            for (j = start; j + 3 < start + len; j += 4) {
                int32x4_t a = vld1q_s32(&data[j]);
                int32x4_t b = vld1q_s32(&data[j + len]);
                // t = tw * b mod q using vmull_s32 -> int64x2_t (2 at a time)
                int32x2_t tw_v = vdup_n_s32(tw);
                int64x2_t prod_lo = vmull_s32(tw_v, vget_low_s32(b));
                int64x2_t prod_hi = vmull_s32(tw_v, vget_high_s32(b));
                // Barrett reduce each 64-bit product
                int64_t p0 = vgetq_lane_s64(prod_lo, 0);
                int64_t p1 = vgetq_lane_s64(prod_lo, 1);
                int64_t p2 = vgetq_lane_s64(prod_hi, 0);
                int64_t p3 = vgetq_lane_s64(prod_hi, 1);
                int32_t r0 = dil_reduce_s(p0);
                int32_t r1 = dil_reduce_s(p1);
                int32_t r2 = dil_reduce_s(p2);
                int32_t r3 = dil_reduce_s(p3);
                int32_t tarr[4] = {r0, r1, r2, r3};
                int32x4_t t = vld1q_s32(tarr);

                // data[j] = (a + t) mod q
                int32x4_t sum = vaddq_s32(a, t);
                int32x4_t q_vec = vdupq_n_s32((int32_t)DIL_Q);
                uint32x4_t ge_q = vcgeq_s32(sum, q_vec);
                sum = vsubq_s32(sum, vandq_s32(vreinterpretq_s32_u32(ge_q), q_vec));
                // data[j+len] = (a - t) mod q
                int32x4_t diff = vsubq_s32(a, t);
                uint32x4_t lt_0 = vcltq_s32(diff, vdupq_n_s32(0));
                diff = vaddq_s32(diff, vandq_s32(vreinterpretq_s32_u32(lt_0), q_vec));

                vst1q_s32(&data[j], sum);
                vst1q_s32(&data[j + len], diff);
            }
            for (; j < start + len; j++) {
                int32_t t = dil_mul_s(tw, data[j + len]);
                int32_t u = data[j];
                data[j] = dil_add_s(u, t);
                data[j + len] = dil_sub_s(u, t);
            }
        }
        len >>= 1;
    }
    while (len >= 2) {
        for (int start = 0; start < n; start += 2 * len) {
            int32_t tw = dil_twiddles_s32[k];
            k++;
            for (int j = start; j < start + len; j++) {
                int32_t t = dil_mul_s(tw, data[j + len]);
                int32_t u = data[j];
                data[j] = dil_add_s(u, t);
                data[j + len] = dil_sub_s(u, t);
            }
        }
        len >>= 1;
    }
#else
    while (len >= 2) {
        for (int start = 0; start < n; start += 2 * len) {
            int32_t tw = dil_twiddles_s32[k];
            k++;
            for (int j = start; j < start + len; j++) {
                int32_t t = dil_mul_s(tw, data[j + len]);
                int32_t u = data[j];
                data[j] = dil_add_s(u, t);
                data[j + len] = dil_sub_s(u, t);
            }
        }
        len >>= 1;
    }
#endif
}

// ============================================================
// dilithium_intt_neon: Signed int32_t Dilithium INTT with NEON vmull_s32
// Gentleman-Sande DIF + 1/(n/2) scaling
// ============================================================

void dilithium_intt_neon(int32_t *data, int logN) {
    dil_init_twiddles_s32();
    int n = 1 << logN;
    if (n > 256) n = 256;

    int32_t inv_n = dil_pow_s((int32_t)(n / 2), DIL_Q - 2);

    int k = 127;
    int len = 2;

#ifdef __ARM_NEON
    // Scalar for small len
    while (len < 4 && len <= n / 2) {
        for (int start = 0; start < n; start += 2 * len) {
            int32_t tw = dil_twiddles_s32[k];
            k--;
            for (int j = start; j < start + len; j++) {
                int32_t t_val = data[j];
                data[j] = dil_add_s(t_val, data[j + len]);
                data[j + len] = dil_mul_s(tw, dil_sub_s(data[j + len], t_val));
            }
        }
        len <<= 1;
    }

    // NEON path for len >= 4
    while (len <= n / 2) {
        for (int start = 0; start < n; start += 2 * len) {
            int32_t tw = dil_twiddles_s32[k];
            k--;
            int j;
            for (j = start; j + 3 < start + len; j += 4) {
                int32x4_t a = vld1q_s32(&data[j]);
                int32x4_t b = vld1q_s32(&data[j + len]);
                // sum = (a + b) mod q
                int32x4_t sum = vaddq_s32(a, b);
                int32x4_t q_vec = vdupq_n_s32((int32_t)DIL_Q);
                uint32x4_t ge_q = vcgeq_s32(sum, q_vec);
                sum = vsubq_s32(sum, vandq_s32(vreinterpretq_s32_u32(ge_q), q_vec));
                // diff = (b - a) mod q
                int32x4_t diff = vsubq_s32(b, a);
                uint32x4_t lt_0 = vcltq_s32(diff, vdupq_n_s32(0));
                diff = vaddq_s32(diff, vandq_s32(vreinterpretq_s32_u32(lt_0), q_vec));
                // data[j+len] = tw * diff mod q via vmull_s32
                int32x2_t tw_v = vdup_n_s32(tw);
                int64x2_t p_lo = vmull_s32(tw_v, vget_low_s32(diff));
                int64x2_t p_hi = vmull_s32(tw_v, vget_high_s32(diff));
                int64_t p0 = vgetq_lane_s64(p_lo, 0);
                int64_t p1 = vgetq_lane_s64(p_lo, 1);
                int64_t p2 = vgetq_lane_s64(p_hi, 0);
                int64_t p3 = vgetq_lane_s64(p_hi, 1);
                int32_t rarr[4] = { dil_reduce_s(p0), dil_reduce_s(p1),
                                    dil_reduce_s(p2), dil_reduce_s(p3) };
                int32x4_t reduced = vld1q_s32(rarr);
                vst1q_s32(&data[j], sum);
                vst1q_s32(&data[j + len], reduced);
            }
            for (; j < start + len; j++) {
                int32_t t_val = data[j];
                data[j] = dil_add_s(t_val, data[j + len]);
                data[j + len] = dil_mul_s(tw, dil_sub_s(data[j + len], t_val));
            }
        }
        len <<= 1;
    }

    // Scale by inv_n using vmull_s32
    int32x2_t inv_v = vdup_n_s32(inv_n);
    for (int i = 0; i + 3 < n; i += 4) {
        int32x4_t vals = vld1q_s32(&data[i]);
        int64x2_t p_lo = vmull_s32(inv_v, vget_low_s32(vals));
        int64x2_t p_hi = vmull_s32(inv_v, vget_high_s32(vals));
        int64_t v0 = vgetq_lane_s64(p_lo, 0);
        int64_t v1 = vgetq_lane_s64(p_lo, 1);
        int64_t v2 = vgetq_lane_s64(p_hi, 0);
        int64_t v3 = vgetq_lane_s64(p_hi, 1);
        int32_t rarr[4] = { dil_reduce_s(v0), dil_reduce_s(v1),
                            dil_reduce_s(v2), dil_reduce_s(v3) };
        vst1q_s32(&data[i], vld1q_s32(rarr));
    }
    for (int i = (n & ~3); i < n; i++) {
        data[i] = dil_mul_s(inv_n, data[i]);
    }
#else
    while (len <= n / 2) {
        for (int start = 0; start < n; start += 2 * len) {
            int32_t tw = dil_twiddles_s32[k];
            k--;
            for (int j = start; j < start + len; j++) {
                int32_t t_val = data[j];
                data[j] = dil_add_s(t_val, data[j + len]);
                data[j + len] = dil_mul_s(tw, dil_sub_s(data[j + len], t_val));
            }
        }
        len <<= 1;
    }
    for (int i = 0; i < n; i++) {
        data[i] = dil_mul_s(inv_n, data[i]);
    }
#endif
}
