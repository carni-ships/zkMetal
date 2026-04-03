// BabyBear NTT with ARM NEON intrinsics
// p = 0x78000001 = 2013265921 (31-bit prime, 2^31 - 2^27 + 1)
// Uses Barrett reduction for modular multiplication (no 64-bit division).
// Processes 4 butterflies simultaneously via uint32x4_t NEON vectors.

#include "NeonFieldOps.h"
#include <arm_neon.h>
#include <stdlib.h>
#include <string.h>

// ============================================================
// BabyBear constants
// ============================================================

#define BB_P          2013265921u   // 0x78000001
#define BB_P2         4026531842u   // 2 * P (for lazy reduction)
#define BB_ROOT_2_27  440564289u    // primitive 2^27-th root of unity
#define BB_GENERATOR  31u

// Barrett reduction: r = a*b mod p
// m = floor(2^62 / p), shift = 62
// q = (uint64_t)(prod * m) >> 62, r = prod - q * p, if r >= p: r -= p
#define BARRETT_M     2290649223ull  // floor(2^62 / p)
#define BARRETT_SHIFT 62

// ============================================================
// Scalar field arithmetic
// ============================================================

static inline uint32_t bb_add(uint32_t a, uint32_t b) {
    uint32_t s = a + b;
    return s >= BB_P ? s - BB_P : s;
}

static inline uint32_t bb_sub(uint32_t a, uint32_t b) {
    return a >= b ? a - b : a + BB_P - b;
}

static inline uint32_t bb_mul(uint32_t a, uint32_t b) {
    uint64_t prod = (uint64_t)a * (uint64_t)b;
    uint64_t q = (unsigned __int128)prod * BARRETT_M >> BARRETT_SHIFT;
    uint32_t r = (uint32_t)(prod - q * BB_P);
    return r >= BB_P ? r - BB_P : r;
}

static inline uint32_t bb_pow(uint32_t base, uint32_t exp) {
    uint32_t result = 1;
    uint32_t b = base;
    while (exp > 0) {
        if (exp & 1) result = bb_mul(result, b);
        b = bb_mul(b, b);
        exp >>= 1;
    }
    return result;
}

static inline uint32_t bb_inv(uint32_t a) {
    return bb_pow(a, BB_P - 2);
}

// ============================================================
// NEON 4-wide field arithmetic
// ============================================================

// NEON Barrett multiply: 4 lanes of (a * b) mod p
// Each a, b < p < 2^31. Product < p^2 < 2^62. Fits in uint64.
// We process as two pairs via vmull_u32 (2 lanes each -> 64-bit result).
//
// Barrett reduction: q = floor(prod * M / 2^62), r = prod - q*p
// where M = floor(2^62 / p) = 2290649223.
// Since prod < 2^62 and M < 2^32, prod*M < 2^94.
// We split prod into (prod_hi32, prod_lo32) and compute:
//   lo_m = prod_lo32 * M  (uint64)
//   hi_m = prod_hi32 * M  (uint64)
//   q = (hi_m + (lo_m >> 32)) >> 30
// This gives exact q or q-1, corrected by up to two conditional subtracts.
static inline uint32x4_t bb_mul_neon(uint32x4_t a, uint32x4_t b) {
    uint32x2_t m_vec = vdup_n_u32((uint32_t)BARRETT_M);
    uint32x4_t pv = vdupq_n_u32(BB_P);

    // --- Lanes 0,1 ---
    uint64x2_t prod_01 = vmull_u32(vget_low_u32(a), vget_low_u32(b));
    uint32x2_t prod_01_lo = vmovn_u64(prod_01);          // low 32 bits
    uint32x2_t prod_01_hi = vshrn_n_u64(prod_01, 32);    // high 32 bits

    uint64x2_t lo_m_01 = vmull_u32(prod_01_lo, m_vec);   // prod_lo * M
    uint64x2_t hi_m_01 = vmull_u32(prod_01_hi, m_vec);   // prod_hi * M
    // q = (hi_m + (lo_m >> 32)) >> 30
    uint64x2_t sum_01 = vaddq_u64(hi_m_01, vshrq_n_u64(lo_m_01, 32));
    uint32x2_t q_01 = vshrn_n_u64(sum_01, 30);

    // --- Lanes 2,3 ---
    uint64x2_t prod_23 = vmull_u32(vget_high_u32(a), vget_high_u32(b));
    uint32x2_t prod_23_lo = vmovn_u64(prod_23);
    uint32x2_t prod_23_hi = vshrn_n_u64(prod_23, 32);

    uint64x2_t lo_m_23 = vmull_u32(prod_23_lo, m_vec);
    uint64x2_t hi_m_23 = vmull_u32(prod_23_hi, m_vec);
    uint64x2_t sum_23 = vaddq_u64(hi_m_23, vshrq_n_u64(lo_m_23, 32));
    uint32x2_t q_23 = vshrn_n_u64(sum_23, 30);

    // Combine q and prod_low32 into 4-wide vectors
    uint32x4_t q = vcombine_u32(q_01, q_23);
    uint32x4_t prod_low32 = vcombine_u32(prod_01_lo, prod_23_lo);

    // r = prod_low32 - q * p (mod 2^32, wrapping is fine since r < 2p < 2^32)
    uint32x4_t qp = vmulq_u32(q, pv);
    uint32x4_t r = vsubq_u32(prod_low32, qp);

    // Conditional subtract: if r >= p, r -= p (up to twice for off-by-one)
    uint32x4_t mask = vcgeq_u32(r, pv);
    r = vsubq_u32(r, vandq_u32(mask, pv));
    mask = vcgeq_u32(r, pv);
    r = vsubq_u32(r, vandq_u32(mask, pv));

    return r;
}

// NEON modular add: (a + b) mod p, inputs in [0, p)
static inline uint32x4_t bb_add_neon(uint32x4_t a, uint32x4_t b) {
    uint32x4_t s = vaddq_u32(a, b);
    uint32x4_t pv = vdupq_n_u32(BB_P);
    uint32x4_t mask = vcgeq_u32(s, pv);
    return vsubq_u32(s, vandq_u32(mask, pv));
}

// NEON modular sub: (a - b) mod p, inputs in [0, p)
static inline uint32x4_t bb_sub_neon(uint32x4_t a, uint32x4_t b) {
    uint32x4_t pv = vdupq_n_u32(BB_P);
    // If a >= b, result = a - b. Else result = a + p - b.
    uint32x4_t diff = vsubq_u32(a, b);
    uint32x4_t mask = vcltq_u32(a, b);  // lanes where a < b
    return vaddq_u32(diff, vandq_u32(mask, pv));
}

// Multiply all 4 lanes by the same scalar twiddle factor
static inline uint32x4_t bb_mul_scalar_neon(uint32x4_t a, uint32_t tw) {
    return bb_mul_neon(a, vdupq_n_u32(tw));
}

// ============================================================
// Bit-reversal permutation
// ============================================================

static inline uint32_t bit_reverse(uint32_t x, int logN) {
    uint32_t r = 0;
    for (int i = 0; i < logN; i++) {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    return r;
}

static void bit_reverse_permute(uint32_t *data, int logN) {
    int n = 1 << logN;
    for (int i = 0; i < n; i++) {
        int j = (int)bit_reverse((uint32_t)i, logN);
        if (i < j) {
            uint32_t tmp = data[i];
            data[i] = data[j];
            data[j] = tmp;
        }
    }
}

// ============================================================
// Twiddle factor precomputation
// ============================================================

// Compute primitive 2^logN-th root of unity
static uint32_t bb_root_of_unity(int logN) {
    uint32_t omega = BB_ROOT_2_27;
    for (int i = 0; i < 27 - logN; i++) {
        omega = bb_mul(omega, omega);
    }
    return omega;
}

// Precompute twiddle factors for all stages.
// Layout: for stage s (0-indexed), halfBlock = 1 << s, twiddles start at offset halfBlock.
// twiddles[halfBlock + j] = omega_blockSize^j for j in [0, halfBlock)
// where omega_blockSize = omega_N^(N / blockSize)
// Total storage: N/2 entries (plus entry 0 unused, entry 1 = 1).
// Actually store flat: twiddles[0..N/2) where for stage s, entry at j has
//   twiddle = omega_N^(j * (N / (2 * halfBlock)))   using bit-reversal indexing...
//
// Simpler: store per-stage twiddles.
// For Cooley-Tukey DIT stage s: halfBlock = 1<<s, blockSize = 2*halfBlock
//   w_m = omega_N^(N/blockSize) = omega_N^(N >> (s+1))
//   For butterfly j in [0, halfBlock): twiddle = w_m^j
// We precompute a flat table: twiddles_flat[offset_s + j] for each stage s.
// offset_s = sum_{i=0}^{s-1} (1 << i) = (1 << s) - 1
// So offset_s = halfBlock - 1, and we store halfBlock twiddles per stage.
// Total = sum_{s=0}^{logN-1} (1<<s) = (1<<logN) - 1 = N - 1.

typedef struct {
    uint32_t *twiddles;  // N-1 entries for forward, N-1 for inverse
    int logN;
} TwiddleTable;

static TwiddleTable *precompute_twiddles(int logN) {
    int n = 1 << logN;
    TwiddleTable *table = (TwiddleTable *)malloc(sizeof(TwiddleTable));
    table->logN = logN;
    table->twiddles = (uint32_t *)malloc((size_t)(n - 1) * sizeof(uint32_t));

    uint32_t omega = bb_root_of_unity(logN);

    for (int s = 0; s < logN; s++) {
        int halfBlock = 1 << s;
        int offset = halfBlock - 1;
        // w_m = omega^(n / (2 * halfBlock)) = omega^(n >> (s+1))
        uint32_t w_m = bb_pow(omega, (uint32_t)(n >> (s + 1)));
        uint32_t w = 1;
        for (int j = 0; j < halfBlock; j++) {
            table->twiddles[offset + j] = w;
            w = bb_mul(w, w_m);
        }
    }

    return table;
}

static TwiddleTable *precompute_inv_twiddles(int logN) {
    int n = 1 << logN;
    TwiddleTable *table = (TwiddleTable *)malloc(sizeof(TwiddleTable));
    table->logN = logN;
    table->twiddles = (uint32_t *)malloc((size_t)(n - 1) * sizeof(uint32_t));

    uint32_t omega = bb_root_of_unity(logN);
    uint32_t omega_inv = bb_inv(omega);

    for (int s = 0; s < logN; s++) {
        int halfBlock = 1 << s;
        int offset = halfBlock - 1;
        uint32_t w_m = bb_pow(omega_inv, (uint32_t)(n >> (s + 1)));
        uint32_t w = 1;
        for (int j = 0; j < halfBlock; j++) {
            table->twiddles[offset + j] = w;
            w = bb_mul(w, w_m);
        }
    }

    return table;
}

static void free_twiddle_table(TwiddleTable *table) {
    if (table) {
        free(table->twiddles);
        free(table);
    }
}

// ============================================================
// Forward NTT — Cooley-Tukey DIT
// ============================================================

void babybear_ntt_neon(uint32_t *data, int logN) {
    if (logN <= 0) return;
    int n = 1 << logN;

    // Bit-reversal permutation
    bit_reverse_permute(data, logN);

    // Precompute twiddle factors
    TwiddleTable *tw = precompute_twiddles(logN);

    // Butterfly stages
    for (int s = 0; s < logN; s++) {
        int halfBlock = 1 << s;
        int blockSize = halfBlock << 1;
        int nBlocks = n / blockSize;
        int twOffset = halfBlock - 1;

        if (halfBlock >= 4) {
            // NEON path: process 4 butterflies at a time
            for (int bk = 0; bk < nBlocks; bk++) {
                int base = bk * blockSize;
                int j = 0;
                // NEON: 4 butterflies per iteration
                for (; j + 3 < halfBlock; j += 4) {
                    uint32x4_t tw_vec = vld1q_u32(&tw->twiddles[twOffset + j]);
                    uint32x4_t u = vld1q_u32(&data[base + j]);
                    uint32x4_t v_raw = vld1q_u32(&data[base + j + halfBlock]);
                    uint32x4_t v = bb_mul_neon(tw_vec, v_raw);
                    vst1q_u32(&data[base + j], bb_add_neon(u, v));
                    vst1q_u32(&data[base + j + halfBlock], bb_sub_neon(u, v));
                }
                // Scalar tail
                for (; j < halfBlock; j++) {
                    uint32_t u = data[base + j];
                    uint32_t v = bb_mul(tw->twiddles[twOffset + j], data[base + j + halfBlock]);
                    data[base + j] = bb_add(u, v);
                    data[base + j + halfBlock] = bb_sub(u, v);
                }
            }
        } else {
            // Small stages (halfBlock < 4): scalar path
            for (int bk = 0; bk < nBlocks; bk++) {
                int base = bk * blockSize;
                for (int j = 0; j < halfBlock; j++) {
                    uint32_t u = data[base + j];
                    uint32_t v = bb_mul(tw->twiddles[twOffset + j], data[base + j + halfBlock]);
                    data[base + j] = bb_add(u, v);
                    data[base + j + halfBlock] = bb_sub(u, v);
                }
            }
        }
    }

    free_twiddle_table(tw);
}

// ============================================================
// Inverse NTT — Gentleman-Sande DIF
// ============================================================

void babybear_intt_neon(uint32_t *data, int logN) {
    if (logN <= 0) return;
    int n = 1 << logN;

    // Precompute inverse twiddle factors
    TwiddleTable *tw = precompute_inv_twiddles(logN);

    // DIF stages (top-down: from stage logN-1 down to 0)
    for (int si = 0; si < logN; si++) {
        int s = logN - 1 - si;
        int halfBlock = 1 << s;
        int blockSize = halfBlock << 1;
        int nBlocks = n / blockSize;
        int twOffset = halfBlock - 1;

        if (halfBlock >= 4) {
            for (int bk = 0; bk < nBlocks; bk++) {
                int base = bk * blockSize;
                int j = 0;
                for (; j + 3 < halfBlock; j += 4) {
                    uint32x4_t tw_vec = vld1q_u32(&tw->twiddles[twOffset + j]);
                    uint32x4_t a = vld1q_u32(&data[base + j]);
                    uint32x4_t b = vld1q_u32(&data[base + j + halfBlock]);
                    vst1q_u32(&data[base + j], bb_add_neon(a, b));
                    uint32x4_t diff = bb_sub_neon(a, b);
                    vst1q_u32(&data[base + j + halfBlock], bb_mul_neon(diff, tw_vec));
                }
                for (; j < halfBlock; j++) {
                    uint32_t a = data[base + j];
                    uint32_t b = data[base + j + halfBlock];
                    data[base + j] = bb_add(a, b);
                    data[base + j + halfBlock] = bb_mul(bb_sub(a, b), tw->twiddles[twOffset + j]);
                }
            }
        } else {
            for (int bk = 0; bk < nBlocks; bk++) {
                int base = bk * blockSize;
                for (int j = 0; j < halfBlock; j++) {
                    uint32_t a = data[base + j];
                    uint32_t b = data[base + j + halfBlock];
                    data[base + j] = bb_add(a, b);
                    data[base + j + halfBlock] = bb_mul(bb_sub(a, b), tw->twiddles[twOffset + j]);
                }
            }
        }
    }

    free_twiddle_table(tw);

    // Bit-reversal permutation
    bit_reverse_permute(data, logN);

    // Scale by 1/n
    uint32_t n_inv = bb_inv((uint32_t)n);
    uint32x4_t n_inv_vec = vdupq_n_u32(n_inv);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        uint32x4_t v = vld1q_u32(&data[i]);
        vst1q_u32(&data[i], bb_mul_neon(v, n_inv_vec));
    }
    for (; i < n; i++) {
        data[i] = bb_mul(data[i], n_inv);
    }
}
