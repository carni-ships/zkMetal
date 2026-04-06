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
#include <string.h>

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
