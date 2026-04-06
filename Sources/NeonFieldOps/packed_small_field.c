// Packed 4-wide BabyBear / M31 NEON arithmetic
//
// Packs 4 field elements into a single NEON uint32x4_t register for
// vectorized add/sub/mul. Gives 2-3x throughput vs scalar for batch ops.
//
// BabyBear: p = 2013265921 = 15 * 2^27 + 1  (31-bit prime)
// M31:      p = 2^31 - 1 = 2147483647        (Mersenne prime)
//
// Multiplication uses the Plonky3-style Montgomery technique for BabyBear
// (7 NEON instructions per 4 muls) and shift+mask for M31.

#include "NeonFieldOps.h"
#include <arm_neon.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

// ============================================================
// BabyBear constants
// ============================================================

#define BB_P          2013265921u
#define BB_P_S        ((int32_t)BB_P)
#define BB_P_INV      2281701377u   // p^{-1} mod 2^32
#define BB_P_INV_S    ((int32_t)BB_P_INV)
#define BB_R2_MOD_P   1172168163u   // R^2 mod p (R = 2^32)

// ============================================================
// M31 constants
// ============================================================

#define M31_P         0x7FFFFFFFu   // 2^31 - 1

// ============================================================
// Packed BabyBear: 4-wide NEON arithmetic
// ============================================================

// Montgomery multiply: 4 BabyBear muls in 7 NEON instructions (Plonky3 technique).
// Inputs/outputs in Montgomery form, values in [0, p).
static inline uint32x4_t packed_bb_mul_impl(uint32x4_t a, uint32x4_t b,
                                             int32x4_t p_inv_v, int32x4_t p_vec) {
    int32x4_t sa = vreinterpretq_s32_u32(a);
    int32x4_t sb = vreinterpretq_s32_u32(b);
    // prod_lo = low 32 bits of a*b
    int32x4_t prod_lo = vmulq_s32(sa, sb);
    // q = prod_lo * p_inv mod 2^32
    int32x4_t q = vmulq_s32(prod_lo, p_inv_v);
    // ab_hi = high 32 bits of a*b (doubled via vqdmulh)
    int32x4_t ab_hi = vqdmulhq_s32(sa, sb);
    // qp_hi = high 32 bits of q*p (doubled via vqdmulh)
    int32x4_t qp_hi = vqdmulhq_s32(q, p_vec);
    // r = (ab_hi - qp_hi) / 2  (halving subtract undoes the doubling)
    int32x4_t r = vhsubq_s32(ab_hi, qp_hi);
    // Conditional add p if r < 0
    int32x4_t mask = vshrq_n_s32(r, 31);
    r = vaddq_s32(r, vandq_s32(mask, p_vec));
    return vreinterpretq_u32_s32(r);
}

uint32x4_t packed_bb_add(uint32x4_t a, uint32x4_t b) {
    uint32x4_t p_vec = vdupq_n_u32(BB_P);
    uint32x4_t s = vaddq_u32(a, b);
    // If s >= p, subtract p. Use unsigned saturating subtract to detect.
    uint32x4_t reduced = vsubq_u32(s, p_vec);
    // s >= p iff reduced didn't underflow. Compare: s >= p means reduced is valid.
    uint32x4_t ge_mask = vcgeq_u32(s, p_vec);
    return vbslq_u32(ge_mask, reduced, s);
}

uint32x4_t packed_bb_sub(uint32x4_t a, uint32x4_t b) {
    uint32x4_t p_vec = vdupq_n_u32(BB_P);
    // If a >= b: a - b. Else: a + p - b.
    uint32x4_t diff = vsubq_u32(a, b);
    uint32x4_t wrapped = vaddq_u32(diff, p_vec);
    uint32x4_t ge_mask = vcgeq_u32(a, b);
    return vbslq_u32(ge_mask, diff, wrapped);
}

uint32x4_t packed_bb_mul(uint32x4_t a, uint32x4_t b) {
    int32x4_t p_inv_v = vdupq_n_s32(BB_P_INV_S);
    int32x4_t p_vec = vdupq_n_s32(BB_P_S);
    return packed_bb_mul_impl(a, b, p_inv_v, p_vec);
}

// Barrett reduction of 4 products from 64-bit to 32-bit BabyBear.
// lo = products[0..1] as uint64x2_t, hi = products[2..3] as uint64x2_t.
// Uses Barrett: q = (x * M) >> 62, r = x - q*P, conditional subtract.
uint32x4_t packed_bb_reduce(uint64x2_t lo, uint64x2_t hi) {
    // Barrett constant M = ceil(2^62 / P)
    // P = 2013265921, 2^62 / P = 2284886667.58..., ceil = 2284886668
    const uint64_t M_CONST = 2284886668ULL;
    const uint64_t P64 = (uint64_t)BB_P;
    uint64x2_t m_vec = vdupq_n_u64(M_CONST);
    uint64x2_t p64_vec = vdupq_n_u64(P64);

    // Process lo pair
    // For NEON we need to do the Barrett manually per-lane since
    // there's no 64x64->128 NEON multiply. Use scalar extraction.
    uint64_t v0 = vgetq_lane_u64(lo, 0);
    uint64_t v1 = vgetq_lane_u64(lo, 1);
    uint64_t v2 = vgetq_lane_u64(hi, 0);
    uint64_t v3 = vgetq_lane_u64(hi, 1);

    // Barrett reduce each: q = (v * M) >> 62 using __uint128_t
    __uint128_t p0 = (__uint128_t)v0 * M_CONST;
    __uint128_t p1 = (__uint128_t)v1 * M_CONST;
    __uint128_t p2 = (__uint128_t)v2 * M_CONST;
    __uint128_t p3 = (__uint128_t)v3 * M_CONST;

    uint64_t q0 = (uint64_t)(p0 >> 62);
    uint64_t q1 = (uint64_t)(p1 >> 62);
    uint64_t q2 = (uint64_t)(p2 >> 62);
    uint64_t q3 = (uint64_t)(p3 >> 62);

    uint32_t r0 = (uint32_t)(v0 - q0 * P64);
    uint32_t r1 = (uint32_t)(v1 - q1 * P64);
    uint32_t r2 = (uint32_t)(v2 - q2 * P64);
    uint32_t r3 = (uint32_t)(v3 - q3 * P64);

    // Conditional subtract if r >= P
    if (r0 >= BB_P) r0 -= BB_P;
    if (r1 >= BB_P) r1 -= BB_P;
    if (r2 >= BB_P) r2 -= BB_P;
    if (r3 >= BB_P) r3 -= BB_P;

    uint32_t arr[4] = { r0, r1, r2, r3 };
    return vld1q_u32(arr);
}

// ============================================================
// Packed M31: 4-wide NEON arithmetic
// ============================================================

uint32x4_t packed_m31_add(uint32x4_t a, uint32x4_t b) {
    uint32x4_t p_vec = vdupq_n_u32(M31_P);
    uint32x4_t s = vaddq_u32(a, b);
    // M31 reduce: r = (s >> 31) + (s & p)
    uint32x4_t hi = vshrq_n_u32(s, 31);
    uint32x4_t lo = vandq_u32(s, p_vec);
    uint32x4_t r = vaddq_u32(hi, lo);
    // Conditional subtract if r >= p
    uint32x4_t ge_mask = vcgeq_u32(r, p_vec);
    return vsubq_u32(r, vandq_u32(ge_mask, p_vec));
}

uint32x4_t packed_m31_sub(uint32x4_t a, uint32x4_t b) {
    uint32x4_t p_vec = vdupq_n_u32(M31_P);
    // If a >= b: a - b. Else: a + p - b.
    uint32x4_t diff = vsubq_u32(a, b);
    uint32x4_t wrapped = vaddq_u32(diff, p_vec);
    uint32x4_t ge_mask = vcgeq_u32(a, b);
    return vbslq_u32(ge_mask, diff, wrapped);
}

uint32x4_t packed_m31_mul(uint32x4_t a, uint32x4_t b) {
    uint32x4_t p_vec = vdupq_n_u32(M31_P);

    // vmull_u32 processes 2 elements at a time -> 64-bit products
    // Low pair: elements [0,1]
    uint64x2_t prod_lo = vmull_u32(vget_low_u32(a), vget_low_u32(b));
    // High pair: elements [2,3]
    uint64x2_t prod_hi = vmull_u32(vget_high_u32(a), vget_high_u32(b));

    // M31 reduce each 64-bit product: r = (prod & 0x7FFFFFFF) + (prod >> 31)
    uint64x2_t mask64 = vdupq_n_u64(M31_P);

    uint64x2_t lo_lo = vandq_u64(prod_lo, mask64);
    uint64x2_t lo_hi = vshrq_n_u64(prod_lo, 31);
    uint64x2_t r_lo = vaddq_u64(lo_lo, lo_hi);

    uint64x2_t hi_lo = vandq_u64(prod_hi, mask64);
    uint64x2_t hi_hi = vshrq_n_u64(prod_hi, 31);
    uint64x2_t r_hi = vaddq_u64(hi_lo, hi_hi);

    // Narrow back to 32-bit (results fit in 32 bits after reduction)
    uint32x2_t n_lo = vmovn_u64(r_lo);
    uint32x2_t n_hi = vmovn_u64(r_hi);
    uint32x4_t r = vcombine_u32(n_lo, n_hi);

    // Final conditional subtract if r >= p
    uint32x4_t ge_mask = vcgeq_u32(r, p_vec);
    return vsubq_u32(r, vandq_u32(ge_mask, p_vec));
}

// ============================================================
// BabyBear batch operations (callable from Swift)
// ============================================================

void packed_bb_dot_product(const uint32_t *a, const uint32_t *b, int n, uint32_t *result) {
    int32x4_t p_inv_v = vdupq_n_s32(BB_P_INV_S);
    int32x4_t p_vec_s = vdupq_n_s32(BB_P_S);
    uint32x4_t p_vec = vdupq_n_u32(BB_P);

    // Accumulate in Montgomery form
    uint32x4_t acc = vdupq_n_u32(0);
    int i = 0;

    for (; i + 3 < n; i += 4) {
        uint32x4_t va = vld1q_u32(a + i);
        uint32x4_t vb = vld1q_u32(b + i);
        uint32x4_t prod = packed_bb_mul_impl(va, vb, p_inv_v, p_vec_s);
        // Accumulate with mod-add
        uint32x4_t s = vaddq_u32(acc, prod);
        uint32x4_t ge_mask = vcgeq_u32(s, p_vec);
        acc = vsubq_u32(s, vandq_u32(ge_mask, p_vec));
    }

    // Horizontal reduction: sum the 4 lanes
    // Extract lanes and reduce scalar
    uint32_t lanes[4];
    vst1q_u32(lanes, acc);
    uint64_t sum = (uint64_t)lanes[0] + lanes[1];
    sum += (uint64_t)lanes[2] + lanes[3];
    uint32_t r = (uint32_t)(sum % BB_P);

    // Handle tail elements
    for (; i < n; i++) {
        uint64_t prod = (uint64_t)a[i] * b[i];
        // Montgomery reduce
        uint32_t lo = (uint32_t)prod;
        uint32_t q = lo * BB_P_INV;
        int64_t t = (int64_t)prod - (int64_t)q * (int64_t)BB_P;
        int32_t rr = (int32_t)(t >> 32);
        uint32_t val = rr < 0 ? (uint32_t)(rr + BB_P_S) : (uint32_t)rr;
        uint64_t s = (uint64_t)r + val;
        r = s >= BB_P ? (uint32_t)(s - BB_P) : (uint32_t)s;
    }

    *result = r;
}

void packed_bb_vector_add(const uint32_t *a, const uint32_t *b, uint32_t *out, int n) {
    uint32x4_t p_vec = vdupq_n_u32(BB_P);
    int i = 0;

    for (; i + 3 < n; i += 4) {
        uint32x4_t va = vld1q_u32(a + i);
        uint32x4_t vb = vld1q_u32(b + i);
        uint32x4_t s = vaddq_u32(va, vb);
        uint32x4_t ge_mask = vcgeq_u32(s, p_vec);
        vst1q_u32(out + i, vsubq_u32(s, vandq_u32(ge_mask, p_vec)));
    }

    // Scalar tail
    for (; i < n; i++) {
        uint32_t s = a[i] + b[i];
        out[i] = s >= BB_P ? s - BB_P : s;
    }
}

void packed_m31_dot_product(const uint32_t *a, const uint32_t *b, int n, uint32_t *result) {
    uint32x4_t p_vec = vdupq_n_u32(M31_P);

    // Use 64-bit accumulators to avoid overflow during summation.
    // Each product is at most (2^31-2)^2 < 2^62, and we add up to 4 of those
    // per iteration in 64-bit, then reduce periodically.
    uint64x2_t acc_lo = vdupq_n_u64(0);
    uint64x2_t acc_hi = vdupq_n_u64(0);
    int i = 0;
    int count = 0;

    for (; i + 3 < n; i += 4) {
        uint32x4_t va = vld1q_u32(a + i);
        uint32x4_t vb = vld1q_u32(b + i);

        // 32x32 -> 64 multiply
        uint64x2_t prod_lo = vmull_u32(vget_low_u32(va), vget_low_u32(vb));
        uint64x2_t prod_hi = vmull_u32(vget_high_u32(va), vget_high_u32(vb));

        acc_lo = vaddq_u64(acc_lo, prod_lo);
        acc_hi = vaddq_u64(acc_hi, prod_hi);
        count++;

        // Reduce every 4 iterations to prevent 64-bit overflow
        // Max product ~2^62, accumulate 4 -> ~2^64, so reduce at 3 to be safe
        if (count >= 3) {
            // M31 reduce the accumulators
            uint64x2_t mask64 = vdupq_n_u64(M31_P);

            uint64x2_t r_lo = vaddq_u64(vandq_u64(acc_lo, mask64), vshrq_n_u64(acc_lo, 31));
            uint64x2_t r_hi = vaddq_u64(vandq_u64(acc_hi, mask64), vshrq_n_u64(acc_hi, 31));

            acc_lo = r_lo;
            acc_hi = r_hi;
            count = 0;
        }
    }

    // Final M31 reduce accumulators
    uint64x2_t mask64 = vdupq_n_u64(M31_P);
    uint64x2_t r_lo = vaddq_u64(vandq_u64(acc_lo, mask64), vshrq_n_u64(acc_lo, 31));
    uint64x2_t r_hi = vaddq_u64(vandq_u64(acc_hi, mask64), vshrq_n_u64(acc_hi, 31));

    // Sum all 4 lanes
    uint64_t s0 = vgetq_lane_u64(r_lo, 0);
    uint64_t s1 = vgetq_lane_u64(r_lo, 1);
    uint64_t s2 = vgetq_lane_u64(r_hi, 0);
    uint64_t s3 = vgetq_lane_u64(r_hi, 1);

    uint64_t total = s0 + s1 + s2 + s3;

    // Full M31 reduce
    uint32_t r = (uint32_t)((total & M31_P) + (total >> 31));
    if (r >= M31_P) r -= M31_P;

    // Handle tail elements
    for (; i < n; i++) {
        uint64_t prod = (uint64_t)a[i] * b[i];
        uint32_t lo = (uint32_t)(prod & M31_P);
        uint32_t hi = (uint32_t)(prod >> 31);
        uint32_t val = lo + hi;
        if (val >= M31_P) val -= M31_P;
        uint32_t s = r + val;
        if (s >= M31_P) s -= M31_P;
        r = s;
    }

    *result = r;
}

// ============================================================
// BabyBear NTT butterfly (vectorized)
// ============================================================

void packed_bb_ntt_butterfly(uint32_t *a_arr, uint32_t *b_arr,
                              const uint32_t *tw, int n) {
    int32x4_t p_inv_v = vdupq_n_s32(BB_P_INV_S);
    int32x4_t p_vec_s = vdupq_n_s32(BB_P_S);
    uint32x4_t p_vec = vdupq_n_u32(BB_P);
    int i = 0;

    for (; i + 3 < n; i += 4) {
        uint32x4_t va = vld1q_u32(a_arr + i);
        uint32x4_t vb = vld1q_u32(b_arr + i);
        uint32x4_t vt = vld1q_u32(tw + i);

        // t = b * twiddle (Montgomery mul)
        uint32x4_t tb = packed_bb_mul_impl(vb, vt, p_inv_v, p_vec_s);

        // a' = a + t (mod p)
        uint32x4_t sum = vaddq_u32(va, tb);
        uint32x4_t ge_sum = vcgeq_u32(sum, p_vec);
        uint32x4_t new_a = vsubq_u32(sum, vandq_u32(ge_sum, p_vec));

        // b' = a - t (mod p)
        uint32x4_t diff = vsubq_u32(va, tb);
        uint32x4_t lt_mask = vcltq_u32(va, tb);
        uint32x4_t new_b = vaddq_u32(diff, vandq_u32(lt_mask, p_vec));

        vst1q_u32(a_arr + i, new_a);
        vst1q_u32(b_arr + i, new_b);
    }

    // Scalar tail
    for (; i < n; i++) {
        // Montgomery mul: b[i] * tw[i]
        uint64_t prod = (uint64_t)b_arr[i] * tw[i];
        uint32_t lo = (uint32_t)prod;
        uint32_t q = lo * BB_P_INV;
        int64_t t = (int64_t)prod - (int64_t)q * (int64_t)BB_P;
        int32_t rr = (int32_t)(t >> 32);
        uint32_t tb = rr < 0 ? (uint32_t)(rr + BB_P_S) : (uint32_t)rr;

        uint32_t old_a = a_arr[i];
        uint32_t sum = old_a + tb;
        a_arr[i] = sum >= BB_P ? sum - BB_P : sum;
        b_arr[i] = old_a >= tb ? old_a - tb : old_a + BB_P - tb;
    }
}

// ============================================================
// BabyBear DIT butterfly: (u, v) -> (u + tw*v, u - tw*v)
// Data in Montgomery form; twiddles has halfBlock entries.
// ============================================================

void packed_bb_butterfly_dit(uint32_t *data, int halfBlock, int nBlocks,
                              const uint32_t *twiddles) {
    int32x4_t p_inv = vdupq_n_s32(BB_P_INV_S);
    int32x4_t p_vec = vdupq_n_s32(BB_P_S);

    for (int bk = 0; bk < nBlocks; bk++) {
        int base = bk * (halfBlock << 1);
        int j = 0;
        for (; j + 3 < halfBlock; j += 4) {
            int32x4_t tw_vec = vld1q_s32((const int32_t *)&twiddles[j]);
            int32x4_t u = vld1q_s32((const int32_t *)&data[base + j]);
            int32x4_t v_raw = vld1q_s32((const int32_t *)&data[base + j + halfBlock]);

            // tw * v (Montgomery)
            int32x4_t prod_lo = vmulq_s32(tw_vec, v_raw);
            int32x4_t q = vmulq_s32(prod_lo, p_inv);
            int32x4_t ab_hi = vqdmulhq_s32(tw_vec, v_raw);
            int32x4_t qp_hi = vqdmulhq_s32(q, p_vec);
            int32x4_t v = vhsubq_s32(ab_hi, qp_hi);
            int32x4_t mask = vshrq_n_s32(v, 31);
            v = vaddq_s32(v, vandq_s32(mask, p_vec));

            // u + v mod p
            int32x4_t sum = vaddq_s32(u, v);
            int32x4_t s_red = vsubq_s32(sum, p_vec);
            int32x4_t s_mask = vshrq_n_s32(s_red, 31);
            sum = vbslq_s32(vreinterpretq_u32_s32(s_mask), sum, s_red);

            // u - v mod p
            int32x4_t diff = vsubq_s32(u, v);
            int32x4_t d_mask = vshrq_n_s32(diff, 31);
            diff = vaddq_s32(diff, vandq_s32(d_mask, p_vec));

            vst1q_s32((int32_t *)&data[base + j], sum);
            vst1q_s32((int32_t *)&data[base + j + halfBlock], diff);
        }
        // Scalar tail
        for (; j < halfBlock; j++) {
            uint32_t u_s = data[base + j];
            uint64_t prod = (uint64_t)twiddles[j] * data[base + j + halfBlock];
            uint32_t lo = (uint32_t)prod;
            uint32_t qq = lo * BB_P_INV;
            int64_t t = (int64_t)prod - (int64_t)qq * (int64_t)BB_P;
            int32_t rr = (int32_t)(t >> 32);
            uint32_t v_s = rr < 0 ? (uint32_t)(rr + BB_P_S) : (uint32_t)rr;
            uint32_t s = u_s + v_s;
            data[base + j] = s >= BB_P ? s - BB_P : s;
            data[base + j + halfBlock] = u_s >= v_s ? u_s - v_s : u_s + BB_P - v_s;
        }
    }
}

// ============================================================
// BabyBear DIF butterfly: (a, b) -> (a + b, (a - b)*tw)
// ============================================================

void packed_bb_butterfly_dif(uint32_t *data, int halfBlock, int nBlocks,
                              const uint32_t *twiddles) {
    int32x4_t p_inv = vdupq_n_s32(BB_P_INV_S);
    int32x4_t p_vec = vdupq_n_s32(BB_P_S);

    for (int bk = 0; bk < nBlocks; bk++) {
        int base = bk * (halfBlock << 1);
        int j = 0;
        for (; j + 3 < halfBlock; j += 4) {
            int32x4_t a = vld1q_s32((const int32_t *)&data[base + j]);
            int32x4_t b = vld1q_s32((const int32_t *)&data[base + j + halfBlock]);
            int32x4_t tw_vec = vld1q_s32((const int32_t *)&twiddles[j]);

            // sum = a + b mod p
            int32x4_t sum = vaddq_s32(a, b);
            int32x4_t s_red = vsubq_s32(sum, p_vec);
            int32x4_t s_mask = vshrq_n_s32(s_red, 31);
            sum = vbslq_s32(vreinterpretq_u32_s32(s_mask), sum, s_red);

            // diff = a - b mod p
            int32x4_t diff = vsubq_s32(a, b);
            int32x4_t d_mask = vshrq_n_s32(diff, 31);
            diff = vaddq_s32(diff, vandq_s32(d_mask, p_vec));

            // diff * tw (Montgomery)
            int32x4_t prod_lo = vmulq_s32(diff, tw_vec);
            int32x4_t q = vmulq_s32(prod_lo, p_inv);
            int32x4_t ab_hi = vqdmulhq_s32(diff, tw_vec);
            int32x4_t qp_hi = vqdmulhq_s32(q, p_vec);
            int32x4_t r = vhsubq_s32(ab_hi, qp_hi);
            int32x4_t mask = vshrq_n_s32(r, 31);
            r = vaddq_s32(r, vandq_s32(mask, p_vec));

            vst1q_s32((int32_t *)&data[base + j], sum);
            vst1q_s32((int32_t *)&data[base + j + halfBlock], r);
        }
        for (; j < halfBlock; j++) {
            uint32_t av = data[base + j];
            uint32_t bv = data[base + j + halfBlock];
            uint32_t s = av + bv;
            data[base + j] = s >= BB_P ? s - BB_P : s;
            uint32_t d = av >= bv ? av - bv : av + BB_P - bv;
            uint64_t prod = (uint64_t)d * twiddles[j];
            uint32_t lo = (uint32_t)prod;
            uint32_t qq = lo * BB_P_INV;
            int64_t t = (int64_t)prod - (int64_t)qq * (int64_t)BB_P;
            int32_t rr = (int32_t)(t >> 32);
            data[base + j + halfBlock] = rr < 0 ? (uint32_t)(rr + BB_P_S) : (uint32_t)rr;
        }
    }
}

// ============================================================
// M31 packed butterflies
// ============================================================

void packed_m31_butterfly_dit(uint32_t *data, int halfBlock, int nBlocks,
                              const uint32_t *twiddles) {
    for (int bk = 0; bk < nBlocks; bk++) {
        int base = bk * (halfBlock << 1);
        int j = 0;
        for (; j + 3 < halfBlock; j += 4) {
            uint32x4_t u = vld1q_u32(data + base + j);
            uint32x4_t v = vld1q_u32(data + base + j + halfBlock);
            uint32x4_t tw = vld1q_u32(twiddles + j);
            uint32x4_t bw = packed_m31_mul(v, tw);
            vst1q_u32(data + base + j, packed_m31_add(u, bw));
            vst1q_u32(data + base + j + halfBlock, packed_m31_sub(u, bw));
        }
        for (; j < halfBlock; j++) {
            uint32_t u = data[base + j];
            uint64_t prod = (uint64_t)data[base + j + halfBlock] * twiddles[j];
            uint32_t lo = (uint32_t)(prod & M31_P);
            uint32_t hi = (uint32_t)(prod >> 31);
            uint32_t r = lo + hi;
            r = (r >> 31) + (r & M31_P);
            if (r >= M31_P) r -= M31_P;
            uint32_t s = u + r;
            if (s >= M31_P) s -= M31_P;
            data[base + j] = s;
            data[base + j + halfBlock] = u >= r ? u - r : u + M31_P - r;
        }
    }
}

void packed_m31_butterfly_dif(uint32_t *data, int halfBlock, int nBlocks,
                              const uint32_t *twiddles) {
    for (int bk = 0; bk < nBlocks; bk++) {
        int base = bk * (halfBlock << 1);
        int j = 0;
        for (; j + 3 < halfBlock; j += 4) {
            uint32x4_t a = vld1q_u32(data + base + j);
            uint32x4_t b = vld1q_u32(data + base + j + halfBlock);
            uint32x4_t tw = vld1q_u32(twiddles + j);
            vst1q_u32(data + base + j, packed_m31_add(a, b));
            vst1q_u32(data + base + j + halfBlock, packed_m31_mul(packed_m31_sub(a, b), tw));
        }
        for (; j < halfBlock; j++) {
            uint32_t a = data[base + j];
            uint32_t b = data[base + j + halfBlock];
            uint32_t s = a + b;
            if (s >= M31_P) s -= M31_P;
            data[base + j] = s;
            uint32_t d = a >= b ? a - b : a + M31_P - b;
            uint64_t prod = (uint64_t)d * twiddles[j];
            uint32_t lo = (uint32_t)(prod & M31_P);
            uint32_t hi = (uint32_t)(prod >> 31);
            uint32_t r = lo + hi;
            r = (r >> 31) + (r & M31_P);
            if (r >= M31_P) r -= M31_P;
            data[base + j + halfBlock] = r;
        }
    }
}

// ============================================================
// BabyBear packed NTT — scalar helpers for twiddle generation
// ============================================================

#define BB_ROOT_2_27  440564289u    // primitive 2^27-th root of unity

static inline uint32_t psf_plain_mul(uint32_t a, uint32_t b) {
    return (uint32_t)((uint64_t)a * b % BB_P);
}

static inline uint32_t psf_plain_pow(uint32_t base, uint32_t exp) {
    uint32_t r = 1, b = base;
    while (exp > 0) {
        if (exp & 1) r = psf_plain_mul(r, b);
        b = psf_plain_mul(b, b);
        exp >>= 1;
    }
    return r;
}

static inline uint32_t psf_monty_reduce64(uint64_t x) {
    uint32_t lo = (uint32_t)x;
    uint32_t q = lo * BB_P_INV;
    int64_t t = (int64_t)x - (int64_t)q * (int64_t)BB_P;
    int32_t r = (int32_t)(t >> 32);
    return r < 0 ? (uint32_t)(r + BB_P_S) : (uint32_t)r;
}

static inline uint32_t psf_to_monty(uint32_t a) {
    return psf_monty_reduce64((uint64_t)a * BB_R2_MOD_P);
}

static inline uint32_t psf_monty_mul(uint32_t a, uint32_t b) {
    return psf_monty_reduce64((uint64_t)a * (uint64_t)b);
}

static inline uint32_t psf_monty_pow(uint32_t base, uint32_t exp) {
    uint32_t result = psf_to_monty(1);
    uint32_t b = base;
    while (exp > 0) {
        if (exp & 1) result = psf_monty_mul(result, b);
        b = psf_monty_mul(b, b);
        exp >>= 1;
    }
    return result;
}

// ============================================================
// BabyBear packed NTT — twiddle cache + full NTT
// ============================================================

typedef struct {
    uint32_t *fwd;
    uint32_t *inv;
    int logN;
} PsfTwiddles;

static PsfTwiddles psf_tw_cache[28] = {{0}};
static pthread_mutex_t psf_tw_lock = PTHREAD_MUTEX_INITIALIZER;

static void psf_ensure_twiddles(int logN) {
    if (psf_tw_cache[logN].fwd) return;

    pthread_mutex_lock(&psf_tw_lock);
    if (psf_tw_cache[logN].fwd) { pthread_mutex_unlock(&psf_tw_lock); return; }

    int n = 1 << logN;

    uint32_t omega_plain = BB_ROOT_2_27;
    for (int i = 0; i < 27 - logN; i++)
        omega_plain = psf_plain_mul(omega_plain, omega_plain);
    uint32_t omega_mont = psf_to_monty(omega_plain);

    uint32_t omega_inv_plain = psf_plain_pow(omega_plain, BB_P - 2);
    uint32_t omega_inv_mont = psf_to_monty(omega_inv_plain);

    uint32_t *fwd = (uint32_t *)malloc((size_t)(n - 1) * sizeof(uint32_t));
    uint32_t *inv = (uint32_t *)malloc((size_t)(n - 1) * sizeof(uint32_t));

    uint32_t one_mont = psf_to_monty(1);

    for (int s = 0; s < logN; s++) {
        int halfBlock = 1 << s;
        int offset = halfBlock - 1;
        uint32_t w_m = psf_monty_pow(omega_mont, (uint32_t)(n >> (s + 1)));
        uint32_t w_m_inv = psf_monty_pow(omega_inv_mont, (uint32_t)(n >> (s + 1)));
        uint32_t w = one_mont, wi = one_mont;
        for (int j = 0; j < halfBlock; j++) {
            fwd[offset + j] = w;
            inv[offset + j] = wi;
            w = psf_monty_mul(w, w_m);
            wi = psf_monty_mul(wi, w_m_inv);
        }
    }

    psf_tw_cache[logN].fwd = fwd;
    psf_tw_cache[logN].inv = inv;
    psf_tw_cache[logN].logN = logN;
    pthread_mutex_unlock(&psf_tw_lock);
}

static void psf_bit_reverse(uint32_t *data, int logN) {
    int n = 1 << logN;
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) {
            uint32_t tmp = data[i];
            data[i] = data[j];
            data[j] = tmp;
        }
    }
}

// Helper: NEON Montgomery multiply and store
static inline void psf_neon_monty_mul_store(const int32_t *src, int32x4_t mul_vec,
                                             int32x4_t p_inv, int32x4_t p_vec,
                                             int32_t *dst) {
    int32x4_t v = vld1q_s32(src);
    int32x4_t prod_lo = vmulq_s32(v, mul_vec);
    int32x4_t q = vmulq_s32(prod_lo, p_inv);
    int32x4_t ab_hi = vqdmulhq_s32(v, mul_vec);
    int32x4_t qp_hi = vqdmulhq_s32(q, p_vec);
    int32x4_t r = vhsubq_s32(ab_hi, qp_hi);
    int32x4_t mask = vshrq_n_s32(r, 31);
    r = vaddq_s32(r, vandq_s32(mask, p_vec));
    vst1q_s32(dst, r);
}

void packed_bb_ntt(uint32_t *data, int logN) {
    if (logN <= 0) return;
    int n = 1 << logN;

    psf_ensure_twiddles(logN);
    const uint32_t *tw = psf_tw_cache[logN].fwd;

    // Convert to Montgomery form
    {
        int32x4_t r2_vec = vdupq_n_s32((int32_t)BB_R2_MOD_P);
        int32x4_t p_inv = vdupq_n_s32(BB_P_INV_S);
        int32x4_t p_vec = vdupq_n_s32(BB_P_S);
        int i = 0;
        for (; i + 3 < n; i += 4)
            psf_neon_monty_mul_store((const int32_t *)&data[i], r2_vec, p_inv, p_vec, (int32_t *)&data[i]);
        for (; i < n; i++)
            data[i] = psf_monty_reduce64((uint64_t)data[i] * BB_R2_MOD_P);
    }

    psf_bit_reverse(data, logN);

    // Butterfly stages (Cooley-Tukey DIT)
    for (int s = 0; s < logN; s++) {
        int halfBlock = 1 << s;
        int blockSize = halfBlock << 1;
        int nBlocks = n / blockSize;
        int twOffset = halfBlock - 1;

        if (halfBlock >= 4) {
            packed_bb_butterfly_dit(data, halfBlock, nBlocks, tw + twOffset);
        } else {
            for (int bk = 0; bk < nBlocks; bk++) {
                int base = bk * blockSize;
                for (int j = 0; j < halfBlock; j++) {
                    uint32_t u = data[base + j];
                    uint32_t v = psf_monty_mul(tw[twOffset + j], data[base + j + halfBlock]);
                    uint32_t sum = u + v;
                    data[base + j] = sum >= BB_P ? sum - BB_P : sum;
                    data[base + j + halfBlock] = u >= v ? u - v : u + BB_P - v;
                }
            }
        }
    }

    // Convert from Montgomery form
    {
        int32x4_t ones = vdupq_n_s32(1);
        int32x4_t p_inv = vdupq_n_s32(BB_P_INV_S);
        int32x4_t p_vec = vdupq_n_s32(BB_P_S);
        int i = 0;
        for (; i + 3 < n; i += 4)
            psf_neon_monty_mul_store((const int32_t *)&data[i], ones, p_inv, p_vec, (int32_t *)&data[i]);
        for (; i < n; i++)
            data[i] = psf_monty_reduce64((uint64_t)data[i]);
    }
}

void packed_bb_intt(uint32_t *data, int logN) {
    if (logN <= 0) return;
    int n = 1 << logN;

    psf_ensure_twiddles(logN);
    const uint32_t *tw = psf_tw_cache[logN].inv;

    // Convert to Montgomery form
    {
        int32x4_t r2_vec = vdupq_n_s32((int32_t)BB_R2_MOD_P);
        int32x4_t p_inv = vdupq_n_s32(BB_P_INV_S);
        int32x4_t p_vec = vdupq_n_s32(BB_P_S);
        int i = 0;
        for (; i + 3 < n; i += 4)
            psf_neon_monty_mul_store((const int32_t *)&data[i], r2_vec, p_inv, p_vec, (int32_t *)&data[i]);
        for (; i < n; i++)
            data[i] = psf_monty_reduce64((uint64_t)data[i] * BB_R2_MOD_P);
    }

    // DIF stages (Gentleman-Sande, top-down)
    for (int si = 0; si < logN; si++) {
        int s = logN - 1 - si;
        int halfBlock = 1 << s;
        int blockSize = halfBlock << 1;
        int nBlocks = n / blockSize;
        int twOffset = halfBlock - 1;

        if (halfBlock >= 4) {
            packed_bb_butterfly_dif(data, halfBlock, nBlocks, tw + twOffset);
        } else {
            for (int bk = 0; bk < nBlocks; bk++) {
                int base = bk * blockSize;
                for (int j = 0; j < halfBlock; j++) {
                    uint32_t a = data[base + j];
                    uint32_t b = data[base + j + halfBlock];
                    uint32_t sum = a + b;
                    data[base + j] = sum >= BB_P ? sum - BB_P : sum;
                    uint32_t d = a >= b ? a - b : a + BB_P - b;
                    data[base + j + halfBlock] = psf_monty_mul(d, tw[twOffset + j]);
                }
            }
        }
    }

    psf_bit_reverse(data, logN);

    // Scale by 1/n and convert from Montgomery form
    {
        uint32_t n_inv_plain = psf_plain_pow((uint32_t)n, BB_P - 2);
        int32x4_t n_inv_vec = vdupq_n_s32((int32_t)n_inv_plain);
        int32x4_t p_inv = vdupq_n_s32(BB_P_INV_S);
        int32x4_t p_vec = vdupq_n_s32(BB_P_S);
        int i = 0;
        for (; i + 3 < n; i += 4)
            psf_neon_monty_mul_store((const int32_t *)&data[i], n_inv_vec, p_inv, p_vec, (int32_t *)&data[i]);
        for (; i < n; i++)
            data[i] = psf_monty_reduce64((uint64_t)data[i] * n_inv_plain);
    }
}
