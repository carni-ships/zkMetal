// M31 (Mersenne-31) NEON batch field operations
// p = 2^31 - 1 = 0x7FFFFFFF = 2147483647
//
// M31 is a single-word prime that fits perfectly in 32-bit NEON lanes.
// Reduction: for x in [0, 2^32), reduce to [0, p) via x mod (2^31 - 1)
//   = (x >> 31) + (x & p), with one conditional subtract if >= p.
//
// Barrett reduction for mul: product fits in 62 bits (31x31),
// so we use vmull_u32 -> 64-bit, then reduce.

#include "NeonFieldOps.h"
#include <arm_neon.h>
#include <string.h>

// ============================================================
// Constants
// ============================================================

#define M31_P          0x7FFFFFFFu  // 2^31 - 1
#define M31_P_S        ((int32_t)M31_P)

// ============================================================
// Scalar M31 arithmetic
// ============================================================

static inline uint32_t m31_reduce(uint32_t x) {
    // x in [0, 2^32): reduce mod (2^31-1)
    uint32_t r = (x >> 31) + (x & M31_P);
    return r >= M31_P ? r - M31_P : r;
}

static inline uint32_t m31_add(uint32_t a, uint32_t b) {
    return m31_reduce(a + b);
}

static inline uint32_t m31_sub(uint32_t a, uint32_t b) {
    // a - b mod p: if a >= b, a-b; else a + p - b
    return a >= b ? a - b : a + M31_P - b;
}

static inline uint32_t m31_mul(uint32_t a, uint32_t b) {
    uint64_t prod = (uint64_t)a * b;
    // prod < 2^62, reduce: hi31 + lo31
    uint32_t lo = (uint32_t)(prod & M31_P);
    uint32_t hi = (uint32_t)(prod >> 31);
    return m31_reduce(lo + hi);
}

static inline uint32_t m31_neg(uint32_t a) {
    return a == 0 ? 0 : M31_P - a;
}

static inline uint32_t m31_pow(uint32_t base, uint32_t exp) {
    uint32_t r = 1, b = base;
    while (exp > 0) {
        if (exp & 1) r = m31_mul(r, b);
        b = m31_mul(b, b);
        exp >>= 1;
    }
    return r;
}

static inline uint32_t m31_inv(uint32_t a) {
    return m31_pow(a, M31_P - 2);
}

// ============================================================
// NEON 4-wide M31 arithmetic
// ============================================================

// Reduce: x in [0, 2^32) -> [0, p)
// r = (x >> 31) + (x & p); if r >= p then r -= p
static inline uint32x4_t m31_reduce_neon(uint32x4_t x) {
    uint32x4_t p_vec = vdupq_n_u32(M31_P);
    uint32x4_t hi = vshrq_n_u32(x, 31);
    uint32x4_t lo = vandq_u32(x, p_vec);
    uint32x4_t r = vaddq_u32(hi, lo);
    // Conditional subtract: if r >= p, r -= p
    uint32x4_t ge_mask = vcgeq_u32(r, p_vec);
    r = vsubq_u32(r, vandq_u32(ge_mask, p_vec));
    return r;
}

// Add: (a + b) mod p, both inputs in [0, p)
static inline uint32x4_t m31_add_neon(uint32x4_t a, uint32x4_t b) {
    return m31_reduce_neon(vaddq_u32(a, b));
}

// Sub: (a - b) mod p, both inputs in [0, p)
static inline uint32x4_t m31_sub_neon(uint32x4_t a, uint32x4_t b) {
    uint32x4_t p_vec = vdupq_n_u32(M31_P);
    uint32x4_t diff = vsubq_u32(a, b);
    // If a < b (borrow), add p. Detect via: if diff > a (unsigned wraparound)
    uint32x4_t borrow = vcltq_u32(a, b);
    return vaddq_u32(diff, vandq_u32(borrow, p_vec));
}

// Neg: -a mod p = (a == 0) ? 0 : p - a
static inline uint32x4_t m31_neg_neon(uint32x4_t a) {
    uint32x4_t p_vec = vdupq_n_u32(M31_P);
    uint32x4_t r = vsubq_u32(p_vec, a);
    // If a == 0, result should be 0, not p
    uint32x4_t zero_mask = vceqq_u32(a, vdupq_n_u32(0));
    return vbicq_u32(r, zero_mask);  // r & ~zero_mask
}

// Multiply 4 pairs using widening mul + Mersenne reduction
// vmull produces 2x uint64x2_t from 2x uint32x2_t, so we split into lo/hi halves
static inline uint32x4_t m31_mul_neon(uint32x4_t a, uint32x4_t b) {
    uint32x4_t p_vec = vdupq_n_u32(M31_P);

    // Low 2 elements
    uint32x2_t a_lo = vget_low_u32(a);
    uint32x2_t b_lo = vget_low_u32(b);
    uint64x2_t prod_lo = vmull_u32(a_lo, b_lo);  // 2 products, each 64-bit

    // High 2 elements
    uint32x2_t a_hi = vget_high_u32(a);
    uint32x2_t b_hi = vget_high_u32(b);
    uint64x2_t prod_hi = vmull_u32(a_hi, b_hi);

    // Mersenne reduction: r = (prod >> 31) + (prod & p)
    // Extract lower 31 bits and upper bits
    uint64x2_t mask31 = vdupq_n_u64(M31_P);

    uint64x2_t lo_lo = vandq_u64(prod_lo, mask31);
    uint64x2_t lo_hi = vshrq_n_u64(prod_lo, 31);
    uint64x2_t hi_lo = vandq_u64(prod_hi, mask31);
    uint64x2_t hi_hi = vshrq_n_u64(prod_hi, 31);

    // Sum in 64-bit, then narrow to 32-bit
    uint64x2_t sum_lo = vaddq_u64(lo_lo, lo_hi);
    uint64x2_t sum_hi = vaddq_u64(hi_lo, hi_hi);

    // Narrow back to 32-bit (values fit in 32 bits: max ~2^32)
    uint32x2_t r_lo = vmovn_u64(sum_lo);
    uint32x2_t r_hi = vmovn_u64(sum_hi);
    uint32x4_t r = vcombine_u32(r_lo, r_hi);

    // Final Mersenne reduction: if r >= p, subtract p
    // Need another pass since sum could be up to ~2*p
    uint32x4_t hi_bits = vshrq_n_u32(r, 31);
    uint32x4_t lo_bits = vandq_u32(r, p_vec);
    r = vaddq_u32(hi_bits, lo_bits);
    uint32x4_t ge_mask = vcgeq_u32(r, p_vec);
    r = vsubq_u32(r, vandq_u32(ge_mask, p_vec));
    return r;
}

// ============================================================
// Batch operations
// ============================================================

void m31_batch_add_neon(const uint32_t *a, const uint32_t *b, uint32_t *out, int n) {
    int i = 0;
    for (; i + 3 < n; i += 4) {
        uint32x4_t va = vld1q_u32(a + i);
        uint32x4_t vb = vld1q_u32(b + i);
        vst1q_u32(out + i, m31_add_neon(va, vb));
    }
    for (; i < n; i++)
        out[i] = m31_add(a[i], b[i]);
}

void m31_batch_sub_neon(const uint32_t *a, const uint32_t *b, uint32_t *out, int n) {
    int i = 0;
    for (; i + 3 < n; i += 4) {
        uint32x4_t va = vld1q_u32(a + i);
        uint32x4_t vb = vld1q_u32(b + i);
        vst1q_u32(out + i, m31_sub_neon(va, vb));
    }
    for (; i < n; i++)
        out[i] = m31_sub(a[i], b[i]);
}

void m31_batch_mul_neon(const uint32_t *a, const uint32_t *b, uint32_t *out, int n) {
    int i = 0;
    for (; i + 3 < n; i += 4) {
        uint32x4_t va = vld1q_u32(a + i);
        uint32x4_t vb = vld1q_u32(b + i);
        vst1q_u32(out + i, m31_mul_neon(va, vb));
    }
    for (; i < n; i++)
        out[i] = m31_mul(a[i], b[i]);
}

void m31_batch_neg_neon(const uint32_t *a, uint32_t *out, int n) {
    int i = 0;
    for (; i + 3 < n; i += 4) {
        uint32x4_t va = vld1q_u32(a + i);
        vst1q_u32(out + i, m31_neg_neon(va));
    }
    for (; i < n; i++)
        out[i] = m31_neg(a[i]);
}

void m31_batch_mul_scalar_neon(const uint32_t *a, uint32_t scalar, uint32_t *out, int n) {
    uint32x4_t s_vec = vdupq_n_u32(scalar);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        uint32x4_t va = vld1q_u32(a + i);
        vst1q_u32(out + i, m31_mul_neon(va, s_vec));
    }
    for (; i < n; i++)
        out[i] = m31_mul(a[i], scalar);
}

uint32_t m31_inner_product_neon(const uint32_t *a, const uint32_t *b, int n) {
    // Accumulate in 64-bit to avoid overflow, then reduce periodically
    uint64x2_t acc_lo = vdupq_n_u64(0);
    uint64x2_t acc_hi = vdupq_n_u64(0);
    int i = 0;

    // Each product is at most (2^31-1)^2 < 2^62.
    // After Mersenne reduction, each term < 2^31.
    // We can accumulate ~2^33 terms in 64-bit without overflow.
    // For safety, reduce every 2^30 iterations.
    int chunk = 0;
    for (; i + 3 < n; i += 4) {
        uint32x4_t va = vld1q_u32(a + i);
        uint32x4_t vb = vld1q_u32(b + i);
        uint32x4_t prod = m31_mul_neon(va, vb);  // each in [0, p)

        // Widen and accumulate
        acc_lo = vaddq_u64(acc_lo, vmovl_u32(vget_low_u32(prod)));
        acc_hi = vaddq_u64(acc_hi, vmovl_u32(vget_high_u32(prod)));

        chunk += 4;
        if (chunk >= (1 << 30)) {
            // Reduce accumulators mod p
            uint64_t s = vgetq_lane_u64(acc_lo, 0) + vgetq_lane_u64(acc_lo, 1)
                       + vgetq_lane_u64(acc_hi, 0) + vgetq_lane_u64(acc_hi, 1);
            // Mersenne reduce 64-bit value
            uint32_t lo_part = (uint32_t)(s & M31_P);
            uint32_t hi_part = (uint32_t)(s >> 31);
            s = (uint64_t)m31_reduce(lo_part + hi_part);
            acc_lo = vdupq_n_u64(s);
            acc_hi = vdupq_n_u64(0);
            chunk = 0;
        }
    }

    // Horizontal sum
    uint64_t sum = vgetq_lane_u64(acc_lo, 0) + vgetq_lane_u64(acc_lo, 1)
                 + vgetq_lane_u64(acc_hi, 0) + vgetq_lane_u64(acc_hi, 1);

    // Scalar tail
    for (; i < n; i++)
        sum += (uint64_t)m31_mul(a[i], b[i]);

    // Final Mersenne reduction of 64-bit sum
    // Could be up to ~n * (2^31) which fits in 64-bit for n < 2^33
    while (sum >= M31_P) {
        uint32_t lo_part = (uint32_t)(sum & M31_P);
        uint32_t hi_part = (uint32_t)(sum >> 31);
        sum = (uint64_t)lo_part + hi_part;
    }
    return (uint32_t)sum;
}

// Batch dot product: for n vectors of length len
void m31_batch_dot_product_neon(const uint32_t *a, const uint32_t *b,
                                 int vecLen, int nVecs, uint32_t *results) {
    for (int v = 0; v < nVecs; v++) {
        results[v] = m31_inner_product_neon(a + v * vecLen, b + v * vecLen, vecLen);
    }
}

// ============================================================
// M31 Extension field: Fp3 = M31[x]/(x^3 - 5)
// Representation: [c0, c1, c2] where element = c0 + c1*x + c2*x^2
// ============================================================

void m31_ext3_add_neon(const uint32_t a[3], const uint32_t b[3], uint32_t r[3]) {
    r[0] = m31_add(a[0], b[0]);
    r[1] = m31_add(a[1], b[1]);
    r[2] = m31_add(a[2], b[2]);
}

void m31_ext3_sub_neon(const uint32_t a[3], const uint32_t b[3], uint32_t r[3]) {
    r[0] = m31_sub(a[0], b[0]);
    r[1] = m31_sub(a[1], b[1]);
    r[2] = m31_sub(a[2], b[2]);
}

// Multiply in Fp3 = M31[x]/(x^3 - 5):
// (a0+a1*x+a2*x^2)(b0+b1*x+b2*x^2) mod (x^3-5)
// x^3 = 5, so:
// c0 = a0*b0 + 5*(a1*b2 + a2*b1)
// c1 = a0*b1 + a1*b0 + 5*a2*b2
// c2 = a0*b2 + a1*b1 + a2*b0
void m31_ext3_mul_neon(const uint32_t a[3], const uint32_t b[3], uint32_t r[3]) {
    uint32_t a0b0 = m31_mul(a[0], b[0]);
    uint32_t a0b1 = m31_mul(a[0], b[1]);
    uint32_t a0b2 = m31_mul(a[0], b[2]);
    uint32_t a1b0 = m31_mul(a[1], b[0]);
    uint32_t a1b1 = m31_mul(a[1], b[1]);
    uint32_t a1b2 = m31_mul(a[1], b[2]);
    uint32_t a2b0 = m31_mul(a[2], b[0]);
    uint32_t a2b1 = m31_mul(a[2], b[1]);
    uint32_t a2b2 = m31_mul(a[2], b[2]);

    // x^3 = 5
    r[0] = m31_add(a0b0, m31_mul(5, m31_add(a1b2, a2b1)));
    r[1] = m31_add(m31_add(a0b1, a1b0), m31_mul(5, a2b2));
    r[2] = m31_add(m31_add(a0b2, a1b1), a2b0);
}

// ============================================================
// M31 NTT (radix-2 Cooley-Tukey, NEON-accelerated)
// p = 2^31 - 1 has multiplicative group of order 2^31 - 2 = 2 * (2^30 - 1)
// Two-adicity: only 1 (since p-1 = 2 * odd). So standard radix-2 NTT
// is limited to size 2. For larger NTTs over M31, use Circle NTT or
// extension field NTT. Here we provide a size-limited NTT for completeness
// and the building blocks (twiddle butterfly) for circle/mixed-radix variants.
// ============================================================

// Butterfly: (a, b) -> (a + b*w, a - b*w)
void m31_butterfly_neon(uint32_t *data, int halfBlock, int nBlocks,
                        const uint32_t *twiddles) {
    for (int bk = 0; bk < nBlocks; bk++) {
        int base = bk * (halfBlock << 1);
        int j = 0;
        for (; j + 3 < halfBlock; j += 4) {
            uint32x4_t va = vld1q_u32(data + base + j);
            uint32x4_t vb = vld1q_u32(data + base + j + halfBlock);
            uint32x4_t vw = vld1q_u32(twiddles + j);
            uint32x4_t bw = m31_mul_neon(vb, vw);
            vst1q_u32(data + base + j, m31_add_neon(va, bw));
            vst1q_u32(data + base + j + halfBlock, m31_sub_neon(va, bw));
        }
        for (; j < halfBlock; j++) {
            uint32_t u = data[base + j];
            uint32_t bw = m31_mul(data[base + j + halfBlock], twiddles[j]);
            data[base + j] = m31_add(u, bw);
            data[base + j + halfBlock] = m31_sub(u, bw);
        }
    }
}

// DIF butterfly: (a, b) -> (a + b, (a - b)*w)
void m31_butterfly_dif_neon(uint32_t *data, int halfBlock, int nBlocks,
                            const uint32_t *twiddles) {
    for (int bk = 0; bk < nBlocks; bk++) {
        int base = bk * (halfBlock << 1);
        int j = 0;
        for (; j + 3 < halfBlock; j += 4) {
            uint32x4_t va = vld1q_u32(data + base + j);
            uint32x4_t vb = vld1q_u32(data + base + j + halfBlock);
            uint32x4_t vw = vld1q_u32(twiddles + j);
            uint32x4_t sum = m31_add_neon(va, vb);
            uint32x4_t diff = m31_sub_neon(va, vb);
            vst1q_u32(data + base + j, sum);
            vst1q_u32(data + base + j + halfBlock, m31_mul_neon(diff, vw));
        }
        for (; j < halfBlock; j++) {
            uint32_t a = data[base + j];
            uint32_t b = data[base + j + halfBlock];
            data[base + j] = m31_add(a, b);
            data[base + j + halfBlock] = m31_mul(m31_sub(a, b), twiddles[j]);
        }
    }
}
