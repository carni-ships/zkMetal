// Batch Poseidon2 over small fields using NEON
//
// Implements Poseidon2 permutation for:
// - BabyBear (p = 0x78000001, width=16, Montgomery form)
// - M31 (p = 2^31 - 1, width=16)
// - Goldilocks (p = 2^64 - 2^32 + 1, width=12)
//
// Uses NEON SIMD for the external/internal matrix multiplications and S-box.
// BabyBear/M31: 4 elements per NEON register (32-bit lanes)
// Goldilocks: 2 elements per NEON register (64-bit lanes)

#include "NeonFieldOps.h"
#include <arm_neon.h>
#include <string.h>

// ============================================================
// BabyBear Poseidon2 constants and helpers
// ============================================================

#define BB_P        2013265921u
#define BB_P_S      2013265921
#define BB_P_INV    2281701377u
#define BB_P_INV_S  ((int32_t)BB_P_INV)
#define BB_R2       1172168163u  // R^2 mod p

#define BB_WIDTH    16           // Poseidon2 width for BabyBear
#define BB_SBOX_DEG 7           // x^7 S-box

static inline int32x4_t bb_p2_mul(int32x4_t a, int32x4_t b,
                                    int32x4_t p_inv, int32x4_t pv) {
    int32x4_t prod_lo = vmulq_s32(a, b);
    int32x4_t q = vmulq_s32(prod_lo, p_inv);
    int32x4_t ab_hi = vqdmulhq_s32(a, b);
    int32x4_t qp_hi = vqdmulhq_s32(q, pv);
    int32x4_t r = vhsubq_s32(ab_hi, qp_hi);
    int32x4_t mask = vshrq_n_s32(r, 31);
    return vaddq_s32(r, vandq_s32(mask, pv));
}

static inline int32x4_t bb_p2_add(int32x4_t a, int32x4_t b, int32x4_t pv) {
    int32x4_t s = vaddq_s32(a, b);
    int32x4_t reduced = vsubq_s32(s, pv);
    int32x4_t mask = vshrq_n_s32(reduced, 31);
    return vbslq_s32(vreinterpretq_u32_s32(mask), s, reduced);
}

static inline int32x4_t bb_p2_sub(int32x4_t a, int32x4_t b, int32x4_t pv) {
    int32x4_t diff = vsubq_s32(a, b);
    int32x4_t mask = vshrq_n_s32(diff, 31);
    return vaddq_s32(diff, vandq_s32(mask, pv));
}

// S-box: x^7 = x * (x^2)^3 = x * x^2 * (x^2)^2
static inline int32x4_t bb_p2_sbox(int32x4_t x, int32x4_t p_inv, int32x4_t pv) {
    int32x4_t x2 = bb_p2_mul(x, x, p_inv, pv);
    int32x4_t x3 = bb_p2_mul(x2, x, p_inv, pv);
    int32x4_t x4 = bb_p2_mul(x2, x2, p_inv, pv);
    int32x4_t x7 = bb_p2_mul(x3, x4, p_inv, pv);
    return x7;
}

// Scalar BabyBear Montgomery ops for round constants setup
static inline uint32_t bb_p2_monty_reduce64(uint64_t x) {
    uint32_t lo = (uint32_t)x;
    uint32_t q = lo * BB_P_INV;
    int64_t t = (int64_t)x - (int64_t)q * (int64_t)BB_P;
    int32_t r = (int32_t)(t >> 32);
    return r < 0 ? (uint32_t)(r + BB_P_S) : (uint32_t)r;
}

static inline uint32_t bb_p2_to_monty(uint32_t a) {
    return bb_p2_monty_reduce64((uint64_t)a * BB_R2);
}

static inline uint32_t bb_p2_scalar_mul(uint32_t a, uint32_t b) {
    return bb_p2_monty_reduce64((uint64_t)a * (uint64_t)b);
}

static inline uint32_t bb_p2_scalar_add(uint32_t a, uint32_t b) {
    uint32_t s = a + b;
    return s >= BB_P ? s - BB_P : s;
}

// ============================================================
// BabyBear Poseidon2 external matrix (M4 Circ(2,3,1,1) x4 + diag)
// Plonky3-compatible: width=16, state = 4 groups of 4 elements
// ============================================================

// M4 circulant matrix [2,3,1,1]: applied to each 4-element group
static inline void bb_p2_m4(int32x4_t *s, int32x4_t p_inv, int32x4_t pv) {
    // Input: s[0..3] = 4 uint32 in NEON register
    // M4: [2 3 1 1; 1 2 3 1; 1 1 2 3; 3 1 1 2]
    // Compute t = sum of all 4, then result[i] = t + s[i] + diagonal_extra*s[...]
    // Actually: row0 = 2*s0 + 3*s1 + s2 + s3 = sum + s0 + 2*s1
    // Let's just compute directly.
    int32_t v[4];
    vst1q_s32(v, *s);

    // Use scalar because M4 is small and NEON shuffle is awkward for circulant
    int32_t t = v[0]; // accumulate sum
    for (int j = 1; j < 4; j++) {
        int32_t sv = v[j];
        int32_t sum = t + sv;
        t = sum >= BB_P_S ? sum - BB_P_S : sum;
    }
    // t = sum of all 4 elements (mod p, approximately -- lazy)

    // row_i = t + s[i] + s[(i+1)%4]  (for circulant [2,3,1,1])
    // Actually M4 circ(2,3,1,1):
    //   out[0] = 2*s0 + 3*s1 + s2 + s3
    //   out[1] = s0 + 2*s1 + 3*s2 + s3
    //   out[2] = s0 + s1 + 2*s2 + 3*s3
    //   out[3] = 3*s0 + s1 + s2 + 2*s3
    // = sum + s_i + 2*s_{(i+1)%4}  ... no.
    // = (s0+s1+s2+s3) + s_i + 2*s_{(i+1)%4}
    // Check: out[0] = sum + s0 + 2*s1 = (s0+s1+s2+s3) + s0 + 2*s1 = 2s0 + 3s1 + s2 + s3. Yes!

    uint32_t uv[4] = {(uint32_t)v[0], (uint32_t)v[1], (uint32_t)v[2], (uint32_t)v[3]};
    uint32_t usum = (uint32_t)t;
    uint32_t out[4];
    for (int i = 0; i < 4; i++) {
        uint32_t extra = bb_p2_scalar_add(uv[i], bb_p2_scalar_add(uv[(i+1)%4], uv[(i+1)%4]));
        out[i] = bb_p2_scalar_add(usum, extra);
    }
    *s = vld1q_s32((const int32_t *)out);
}

// External matrix: apply M4 to each group of 4, then add sums across groups
// This is the Poseidon2 external "diffusion" layer for width 16
static void bb_p2_external_matrix(uint32_t state[16]) {
    int32x4_t pv = vdupq_n_s32(BB_P_S);
    int32x4_t p_inv = vdupq_n_s32(BB_P_INV_S);

    // Load 4 groups
    int32x4_t g0 = vld1q_s32((const int32_t *)(state));
    int32x4_t g1 = vld1q_s32((const int32_t *)(state + 4));
    int32x4_t g2 = vld1q_s32((const int32_t *)(state + 8));
    int32x4_t g3 = vld1q_s32((const int32_t *)(state + 12));

    // Apply M4 to each group
    bb_p2_m4(&g0, p_inv, pv);
    bb_p2_m4(&g1, p_inv, pv);
    bb_p2_m4(&g2, p_inv, pv);
    bb_p2_m4(&g3, p_inv, pv);

    // Cross-group diffusion: for each position i in [0,4):
    //   sum_i = g0[i] + g1[i] + g2[i] + g3[i]
    //   g_j[i] += sum_i  for all j
    // This doubles each element's own contribution but that's the Poseidon2 spec.
    int32x4_t sum = bb_p2_add(bb_p2_add(g0, g1, pv), bb_p2_add(g2, g3, pv), pv);
    g0 = bb_p2_add(g0, sum, pv);
    g1 = bb_p2_add(g1, sum, pv);
    g2 = bb_p2_add(g2, sum, pv);
    g3 = bb_p2_add(g3, sum, pv);

    vst1q_s32((int32_t *)(state), g0);
    vst1q_s32((int32_t *)(state + 4), g1);
    vst1q_s32((int32_t *)(state + 8), g2);
    vst1q_s32((int32_t *)(state + 12), g3);
}

// Internal matrix: apply diag multiply + sum for Poseidon2 internal rounds
// For width 16, internal = diagonal * element + sum_of_all
static void bb_p2_internal_matrix(uint32_t state[16], const uint32_t diag[16]) {
    int32x4_t pv = vdupq_n_s32(BB_P_S);
    int32x4_t p_inv = vdupq_n_s32(BB_P_INV_S);

    // Compute sum of all elements
    int32x4_t s0 = vld1q_s32((const int32_t *)state);
    int32x4_t s1 = vld1q_s32((const int32_t *)(state + 4));
    int32x4_t s2 = vld1q_s32((const int32_t *)(state + 8));
    int32x4_t s3 = vld1q_s32((const int32_t *)(state + 12));

    // Horizontal sum using NEON pairwise
    int32x4_t t01 = bb_p2_add(s0, s1, pv);
    int32x4_t t23 = bb_p2_add(s2, s3, pv);
    int32x4_t t = bb_p2_add(t01, t23, pv);
    // Reduce to scalar
    int32x2_t t_lo = vget_low_s32(t);
    int32x2_t t_hi = vget_high_s32(t);
    int32x2_t t2 = vadd_s32(t_lo, t_hi);
    uint32_t sum_val = (uint32_t)(vget_lane_s32(t2, 0) + vget_lane_s32(t2, 1));
    sum_val = sum_val >= BB_P ? sum_val - BB_P : sum_val;
    if (sum_val >= BB_P) sum_val -= BB_P;
    int32x4_t sum_vec = vdupq_n_s32((int32_t)sum_val);

    // Apply: state[i] = diag[i] * state[i] + sum
    int32x4_t d0 = vld1q_s32((const int32_t *)diag);
    int32x4_t d1 = vld1q_s32((const int32_t *)(diag + 4));
    int32x4_t d2 = vld1q_s32((const int32_t *)(diag + 8));
    int32x4_t d3 = vld1q_s32((const int32_t *)(diag + 12));

    s0 = bb_p2_add(bb_p2_mul(d0, s0, p_inv, pv), sum_vec, pv);
    s1 = bb_p2_add(bb_p2_mul(d1, s1, p_inv, pv), sum_vec, pv);
    s2 = bb_p2_add(bb_p2_mul(d2, s2, p_inv, pv), sum_vec, pv);
    s3 = bb_p2_add(bb_p2_mul(d3, s3, p_inv, pv), sum_vec, pv);

    vst1q_s32((int32_t *)state, s0);
    vst1q_s32((int32_t *)(state + 4), s1);
    vst1q_s32((int32_t *)(state + 8), s2);
    vst1q_s32((int32_t *)(state + 12), s3);
}

// Full external round: add round constants, S-box on all, matrix
static void bb_p2_full_round(uint32_t state[16], const uint32_t rc[16]) {
    int32x4_t pv = vdupq_n_s32(BB_P_S);
    int32x4_t p_inv = vdupq_n_s32(BB_P_INV_S);

    for (int g = 0; g < 4; g++) {
        int32x4_t s = vld1q_s32((const int32_t *)(state + g*4));
        int32x4_t c = vld1q_s32((const int32_t *)(rc + g*4));
        s = bb_p2_add(s, c, pv);
        s = bb_p2_sbox(s, p_inv, pv);
        vst1q_s32((int32_t *)(state + g*4), s);
    }
    bb_p2_external_matrix(state);
}

// Partial round: add constant to state[0] only, S-box on state[0] only, internal matrix
static void bb_p2_partial_round(uint32_t state[16], uint32_t rc0,
                                 const uint32_t diag[16]) {
    state[0] = bb_p2_scalar_add(state[0], rc0);
    // S-box on state[0] only: x^7
    uint32_t x = state[0];
    uint32_t x2 = bb_p2_scalar_mul(x, x);
    uint32_t x3 = bb_p2_scalar_mul(x2, x);
    uint32_t x4 = bb_p2_scalar_mul(x2, x2);
    state[0] = bb_p2_scalar_mul(x3, x4);

    bb_p2_internal_matrix(state, diag);
}

// BabyBear Poseidon2 permutation (width=16, x^7 S-box)
// round_constants: (RF_FULL * 16 + RP_PARTIAL * 1) uint32_t values in Montgomery form
// internal_diag: 16 diagonal constants in Montgomery form
// RF_FULL = 8 (4 full rounds before + 4 after), RP_PARTIAL = 13
void bb_poseidon2_permutation_neon(uint32_t state[16],
                                    const uint32_t *round_constants,
                                    const uint32_t internal_diag[16],
                                    int num_full_rounds,
                                    int num_partial_rounds) {
    int half_full = num_full_rounds / 2;
    int rc_idx = 0;

    // First half of full rounds
    for (int r = 0; r < half_full; r++) {
        bb_p2_full_round(state, round_constants + rc_idx);
        rc_idx += 16;
    }

    // Partial rounds
    for (int r = 0; r < num_partial_rounds; r++) {
        bb_p2_partial_round(state, round_constants[rc_idx], internal_diag);
        rc_idx += 1;
    }

    // Second half of full rounds
    for (int r = 0; r < half_full; r++) {
        bb_p2_full_round(state, round_constants + rc_idx);
        rc_idx += 16;
    }
}

// ============================================================
// M31 Poseidon2 (width=16, x^5 S-box)
// ============================================================

#define M31_P 0x7FFFFFFFu

static inline uint32_t m31_p2_reduce(uint32_t x) {
    uint32_t r = (x >> 31) + (x & M31_P);
    return r >= M31_P ? r - M31_P : r;
}

static inline uint32_t m31_p2_add(uint32_t a, uint32_t b) {
    return m31_p2_reduce(a + b);
}

static inline uint32_t m31_p2_mul(uint32_t a, uint32_t b) {
    uint64_t prod = (uint64_t)a * b;
    uint32_t lo = (uint32_t)(prod & M31_P);
    uint32_t hi = (uint32_t)(prod >> 31);
    return m31_p2_reduce(lo + hi);
}

static inline uint32x4_t m31_p2_reduce_neon(uint32x4_t x) {
    uint32x4_t p_vec = vdupq_n_u32(M31_P);
    uint32x4_t hi = vshrq_n_u32(x, 31);
    uint32x4_t lo = vandq_u32(x, p_vec);
    uint32x4_t r = vaddq_u32(hi, lo);
    uint32x4_t ge_mask = vcgeq_u32(r, p_vec);
    return vsubq_u32(r, vandq_u32(ge_mask, p_vec));
}

static inline uint32x4_t m31_p2_add_neon(uint32x4_t a, uint32x4_t b) {
    return m31_p2_reduce_neon(vaddq_u32(a, b));
}

static inline uint32x4_t m31_p2_sub_neon(uint32x4_t a, uint32x4_t b) {
    uint32x4_t p_vec = vdupq_n_u32(M31_P);
    uint32x4_t diff = vsubq_u32(a, b);
    uint32x4_t borrow = vcltq_u32(a, b);
    return vaddq_u32(diff, vandq_u32(borrow, p_vec));
}

static inline uint32x4_t m31_p2_mul_neon(uint32x4_t a, uint32x4_t b) {
    uint32x4_t p_vec = vdupq_n_u32(M31_P);
    uint32x2_t a_lo = vget_low_u32(a), a_hi = vget_high_u32(a);
    uint32x2_t b_lo = vget_low_u32(b), b_hi = vget_high_u32(b);
    uint64x2_t prod_lo = vmull_u32(a_lo, b_lo);
    uint64x2_t prod_hi = vmull_u32(a_hi, b_hi);
    uint64x2_t mask31 = vdupq_n_u64(M31_P);
    uint64x2_t lo_lo = vandq_u64(prod_lo, mask31);
    uint64x2_t lo_hi = vshrq_n_u64(prod_lo, 31);
    uint64x2_t hi_lo = vandq_u64(prod_hi, mask31);
    uint64x2_t hi_hi = vshrq_n_u64(prod_hi, 31);
    uint32x2_t r_lo = vmovn_u64(vaddq_u64(lo_lo, lo_hi));
    uint32x2_t r_hi = vmovn_u64(vaddq_u64(hi_lo, hi_hi));
    uint32x4_t r = vcombine_u32(r_lo, r_hi);
    uint32x4_t hi_bits = vshrq_n_u32(r, 31);
    uint32x4_t lo_bits = vandq_u32(r, p_vec);
    r = vaddq_u32(hi_bits, lo_bits);
    uint32x4_t ge_mask = vcgeq_u32(r, p_vec);
    return vsubq_u32(r, vandq_u32(ge_mask, p_vec));
}

// M31 S-box: x^5
static inline uint32x4_t m31_p2_sbox(uint32x4_t x) {
    uint32x4_t x2 = m31_p2_mul_neon(x, x);
    uint32x4_t x4 = m31_p2_mul_neon(x2, x2);
    return m31_p2_mul_neon(x4, x);
}

// M31 external matrix (same structure as BabyBear)
static void m31_p2_external_matrix(uint32_t state[16]) {
    // Apply M4 circ(2,3,1,1) to each group of 4
    for (int g = 0; g < 4; g++) {
        uint32_t *s = state + g * 4;
        uint32_t sum = m31_p2_add(m31_p2_add(s[0], s[1]), m31_p2_add(s[2], s[3]));
        uint32_t out[4];
        for (int i = 0; i < 4; i++) {
            uint32_t extra = m31_p2_add(s[i], m31_p2_add(s[(i+1)%4], s[(i+1)%4]));
            out[i] = m31_p2_add(sum, extra);
        }
        memcpy(s, out, 16);
    }

    // Cross-group: sum_i = s0[i]+s1[i]+s2[i]+s3[i], s_j[i] += sum_i
    uint32x4_t g0 = vld1q_u32(state);
    uint32x4_t g1 = vld1q_u32(state + 4);
    uint32x4_t g2 = vld1q_u32(state + 8);
    uint32x4_t g3 = vld1q_u32(state + 12);
    uint32x4_t sum = m31_p2_add_neon(m31_p2_add_neon(g0, g1), m31_p2_add_neon(g2, g3));
    vst1q_u32(state,      m31_p2_add_neon(g0, sum));
    vst1q_u32(state + 4,  m31_p2_add_neon(g1, sum));
    vst1q_u32(state + 8,  m31_p2_add_neon(g2, sum));
    vst1q_u32(state + 12, m31_p2_add_neon(g3, sum));
}

static void m31_p2_internal_matrix(uint32_t state[16], const uint32_t diag[16]) {
    // sum of all
    uint32_t sum = 0;
    for (int i = 0; i < 16; i++)
        sum = m31_p2_add(sum, state[i]);

    // state[i] = diag[i] * state[i] + sum
    for (int g = 0; g < 4; g++) {
        uint32x4_t s = vld1q_u32(state + g*4);
        uint32x4_t d = vld1q_u32(diag + g*4);
        uint32x4_t prod = m31_p2_mul_neon(s, d);
        uint32x4_t sum_v = vdupq_n_u32(sum);
        vst1q_u32(state + g*4, m31_p2_add_neon(prod, sum_v));
    }
}

void m31_poseidon2_permutation_neon(uint32_t state[16],
                                     const uint32_t *round_constants,
                                     const uint32_t internal_diag[16],
                                     int num_full_rounds,
                                     int num_partial_rounds) {
    int half_full = num_full_rounds / 2;
    int rc_idx = 0;

    // First half full rounds
    for (int r = 0; r < half_full; r++) {
        // Add round constants + S-box on all
        for (int g = 0; g < 4; g++) {
            uint32x4_t s = vld1q_u32(state + g*4);
            uint32x4_t c = vld1q_u32(round_constants + rc_idx + g*4);
            s = m31_p2_add_neon(s, c);
            s = m31_p2_sbox(s);
            vst1q_u32(state + g*4, s);
        }
        rc_idx += 16;
        m31_p2_external_matrix(state);
    }

    // Partial rounds
    for (int r = 0; r < num_partial_rounds; r++) {
        state[0] = m31_p2_add(state[0], round_constants[rc_idx]);
        rc_idx++;
        // S-box on state[0] only
        uint32_t x = state[0];
        uint32_t x2 = m31_p2_mul(x, x);
        uint32_t x4 = m31_p2_mul(x2, x2);
        state[0] = m31_p2_mul(x4, x);
        m31_p2_internal_matrix(state, internal_diag);
    }

    // Second half full rounds
    for (int r = 0; r < half_full; r++) {
        for (int g = 0; g < 4; g++) {
            uint32x4_t s = vld1q_u32(state + g*4);
            uint32x4_t c = vld1q_u32(round_constants + rc_idx + g*4);
            s = m31_p2_add_neon(s, c);
            s = m31_p2_sbox(s);
            vst1q_u32(state + g*4, s);
        }
        rc_idx += 16;
        m31_p2_external_matrix(state);
    }
}

// ============================================================
// Goldilocks Poseidon2 (width=12, x^7 S-box)
// ============================================================

#define GL_P_VAL 0xFFFFFFFF00000001ULL
#define GL_EPS_VAL 0xFFFFFFFFULL

typedef unsigned __int128 uint128_t;

static inline uint64_t gl_p2_reduce128(uint64_t hi, uint64_t lo) {
    uint64_t hi_lo = hi & 0xFFFFFFFFULL;
    uint64_t hi_hi = hi >> 32;
    uint64_t hi_lo_eps = hi_lo * GL_EPS_VAL;
    uint64_t t1;
    unsigned c1 = __builtin_add_overflow(lo, hi_lo_eps, &t1);
    uint64_t t2;
    unsigned b2 = __builtin_sub_overflow(t1, hi_hi, &t2);
    uint64_t r = t2;
    if (c1) r += GL_EPS_VAL;
    if (b2) r += GL_P_VAL;
    if (r >= GL_P_VAL) r -= GL_P_VAL;
    return r;
}

static inline uint64_t gl_p2_mul(uint64_t a, uint64_t b) {
    uint128_t prod = (uint128_t)a * b;
    return gl_p2_reduce128((uint64_t)(prod >> 64), (uint64_t)prod);
}

static inline uint64_t gl_p2_add(uint64_t a, uint64_t b) {
    uint64_t s;
    unsigned carry = __builtin_add_overflow(a, b, &s);
    if (carry) {
        s += GL_EPS_VAL;
        if (s < GL_EPS_VAL) s += GL_EPS_VAL;
    }
    return s >= GL_P_VAL ? s - GL_P_VAL : s;
}

static inline uint64_t gl_p2_sub(uint64_t a, uint64_t b) {
    return a >= b ? a - b : a + GL_P_VAL - b;
}

// S-box: x^7
static inline uint64_t gl_p2_sbox(uint64_t x) {
    uint64_t x2 = gl_p2_mul(x, x);
    uint64_t x3 = gl_p2_mul(x2, x);
    uint64_t x4 = gl_p2_mul(x2, x2);
    return gl_p2_mul(x3, x4);
}

// NEON 2-wide add for Goldilocks
static inline uint64x2_t gl_p2_add_vec(uint64x2_t a, uint64x2_t b) {
    uint64x2_t p = vdupq_n_u64(GL_P_VAL);
    uint64x2_t eps = vdupq_n_u64(GL_EPS_VAL);
    uint64x2_t sum = vaddq_u64(a, b);
    uint64x2_t carry_mask = vcltq_u64(sum, a);
    sum = vaddq_u64(sum, vandq_u64(eps, carry_mask));
    uint64x2_t carry2_mask = vandq_u64(carry_mask, vcltq_u64(sum, eps));
    sum = vaddq_u64(sum, vandq_u64(eps, carry2_mask));
    uint64x2_t ge_mask = vcgeq_u64(sum, p);
    return vsubq_u64(sum, vandq_u64(p, ge_mask));
}

// Goldilocks Poseidon2 external matrix (width=12, 3 groups of 4)
// Actually Goldilocks Poseidon2 is typically width=8 or width=12.
// We'll implement width=12 with M4 circ(2,3,1,1) on 3 groups of 4.
static void gl_p2_external_matrix(uint64_t state[12]) {
    // Apply M4 to each group of 4
    for (int g = 0; g < 3; g++) {
        uint64_t *s = state + g * 4;
        uint64_t sum = gl_p2_add(gl_p2_add(s[0], s[1]), gl_p2_add(s[2], s[3]));
        uint64_t out[4];
        for (int i = 0; i < 4; i++) {
            uint64_t extra = gl_p2_add(s[i], gl_p2_add(s[(i+1)%4], s[(i+1)%4]));
            out[i] = gl_p2_add(sum, extra);
        }
        memcpy(s, out, 32);
    }

    // Cross-group using NEON
    for (int i = 0; i < 4; i += 2) {
        uint64x2_t g0 = vld1q_u64(state + i);
        uint64x2_t g1 = vld1q_u64(state + 4 + i);
        uint64x2_t g2 = vld1q_u64(state + 8 + i);
        uint64x2_t sum = gl_p2_add_vec(gl_p2_add_vec(g0, g1), g2);
        vst1q_u64(state + i,     gl_p2_add_vec(g0, sum));
        vst1q_u64(state + 4 + i, gl_p2_add_vec(g1, sum));
        vst1q_u64(state + 8 + i, gl_p2_add_vec(g2, sum));
    }
}

static void gl_p2_internal_matrix(uint64_t state[12], const uint64_t diag[12]) {
    uint64_t sum = 0;
    for (int i = 0; i < 12; i++)
        sum = gl_p2_add(sum, state[i]);
    for (int i = 0; i < 12; i++)
        state[i] = gl_p2_add(gl_p2_mul(diag[i], state[i]), sum);
}

void gl_poseidon2_permutation_neon(uint64_t state[12],
                                    const uint64_t *round_constants,
                                    const uint64_t internal_diag[12],
                                    int num_full_rounds,
                                    int num_partial_rounds) {
    int half_full = num_full_rounds / 2;
    int rc_idx = 0;

    // First half full rounds
    for (int r = 0; r < half_full; r++) {
        for (int i = 0; i < 12; i++)
            state[i] = gl_p2_sbox(gl_p2_add(state[i], round_constants[rc_idx + i]));
        rc_idx += 12;
        gl_p2_external_matrix(state);
    }

    // Partial rounds
    for (int r = 0; r < num_partial_rounds; r++) {
        state[0] = gl_p2_sbox(gl_p2_add(state[0], round_constants[rc_idx]));
        rc_idx++;
        gl_p2_internal_matrix(state, internal_diag);
    }

    // Second half full rounds
    for (int r = 0; r < half_full; r++) {
        for (int i = 0; i < 12; i++)
            state[i] = gl_p2_sbox(gl_p2_add(state[i], round_constants[rc_idx + i]));
        rc_idx += 12;
        gl_p2_external_matrix(state);
    }
}
