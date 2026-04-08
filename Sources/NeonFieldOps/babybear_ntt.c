// BabyBear NTT with ARM NEON intrinsics — Montgomery form
// p = 0x78000001 = 2013265921 (31-bit prime, 2^31 - 2^27 + 1)
//
// Uses Montgomery multiplication via vqdmulhq_s32 + vhsubq_s32 (Plonky3 technique):
//   ~7 NEON instructions per 4-element modular multiply (vs ~20 for Barrett).
// Twiddle tables cached across calls. Lazy [0,2p) reduction for add/sub.

#include "NeonFieldOps.h"
#include <arm_neon.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <dispatch/dispatch.h>

#define BB_NTT_PAR_THRESHOLD 4096

// ============================================================
// Constants
// ============================================================

#define BB_P            2013265921u   // 0x78000001
#define BB_P_S          2013265921    // signed
#define BB_2P           4026531842u   // 2*P (lazy reduction bound)
#define BB_ROOT_2_27    440564289u    // primitive 2^27-th root of unity

// Montgomery R = 2^32
// R mod p = 2^32 mod p = 268435454
#define MONTY_R_MOD_P   268435454u    // 2^32 mod p
// R^2 mod p — precomputed: (2^32)^2 mod p
#define MONTY_R2_MOD_P  1172168163u   // 2^64 mod p

// p^{-1} mod 2^32: Montgomery reduction constant (Plonky3 subtraction variant)
// p * P_INV ≡ 1 (mod 2^32)
#define P_INV           2281701377u   // p^{-1} mod 2^32
#define P_INV_S         ((int32_t)P_INV)

// ============================================================
// Scalar Montgomery arithmetic
// ============================================================

static inline uint32_t monty_reduce64(uint64_t x) {
    uint32_t lo = (uint32_t)x;
    uint32_t q = lo * P_INV;
    int64_t t = (int64_t)x - (int64_t)q * (int64_t)BB_P;
    int32_t r = (int32_t)(t >> 32);
    return r < 0 ? (uint32_t)(r + BB_P_S) : (uint32_t)r;
}

static inline uint32_t monty_mul(uint32_t a, uint32_t b) {
    return monty_reduce64((uint64_t)a * (uint64_t)b);
}

static inline uint32_t to_monty(uint32_t a) {
    return monty_mul(a, MONTY_R2_MOD_P);
}

static inline uint32_t from_monty(uint32_t a) {
    return monty_reduce64((uint64_t)a);
}

static inline uint32_t monty_add(uint32_t a, uint32_t b) {
    uint32_t s = a + b;
    return s >= BB_P ? s - BB_P : s;
}

static inline uint32_t monty_sub(uint32_t a, uint32_t b) {
    return a >= b ? a - b : a + BB_P - b;
}

static inline uint32_t monty_pow(uint32_t base, uint32_t exp) {
    uint32_t result = to_monty(1);  // 1 in Montgomery form = R mod p
    uint32_t b = base;  // already in Montgomery form
    while (exp > 0) {
        if (exp & 1) result = monty_mul(result, b);
        b = monty_mul(b, b);
        exp >>= 1;
    }
    return result;
}

static inline uint32_t monty_inv(uint32_t a) {
    return monty_pow(a, BB_P - 2);
}

// Plain (non-Montgomery) pow for root computation
static inline uint32_t plain_mul(uint32_t a, uint32_t b) {
    return (uint32_t)((uint64_t)a * b % BB_P);
}

static inline uint32_t plain_pow(uint32_t base, uint32_t exp) {
    uint32_t result = 1;
    uint32_t b = base;
    while (exp > 0) {
        if (exp & 1) result = plain_mul(result, b);
        b = plain_mul(b, b);
        exp >>= 1;
    }
    return result;
}

// ============================================================
// NEON 4-wide Montgomery multiply
// ============================================================
//
// Plonky3 technique: uses vqdmulhq_s32 (doubling saturating multiply-high)
// which computes floor(2*a*b / 2^32) = floor(a*b / 2^31).
//
// Montgomery reduction with R=2^32 (subtraction variant):
//   prod_lo = (a*b) mod 2^32           [vmulq_s32]
//   q = prod_lo * p^{-1} mod 2^32      [vmulq_s32] — positive inverse!
//   prod_hi31 = floor(a*b / 2^31)      [vqdmulhq_s32]
//   qp_hi31 = floor(q*p / 2^31)        [vqdmulhq_s32]
//   result = (prod_hi31 - qp_hi31) / 2 [vhsubq_s32]  — halving sub gives /2^32
//   if result < 0: result += p
// Since q*p ≡ a*b (mod 2^32), (a*b - q*p) is divisible by 2^32.
//
// Total: 7 NEON instructions for 4 multiplies.

static inline int32x4_t monty_mul_neon(int32x4_t a, int32x4_t b,
                                        int32x4_t p_inv_v, int32x4_t p_vec) {
    int32x4_t prod_lo = vmulq_s32(a, b);
    int32x4_t q = vmulq_s32(prod_lo, p_inv_v);
    int32x4_t ab_hi = vqdmulhq_s32(a, b);
    int32x4_t qp_hi = vqdmulhq_s32(q, p_vec);
    int32x4_t r = vhsubq_s32(ab_hi, qp_hi);
    // Conditional add p if r < 0
    int32x4_t mask = vshrq_n_s32(r, 31);
    return vaddq_s32(r, vandq_s32(mask, p_vec));
}

// Lazy modular add: result in [0, 2p). No conditional branch.
static inline int32x4_t monty_add_lazy_neon(int32x4_t a, int32x4_t b) {
    return vaddq_s32(a, b);  // caller ensures a+b < 2p (both in [0,p))
}

// Full modular add: result in [0, p).
static inline int32x4_t monty_add_neon(int32x4_t a, int32x4_t b, int32x4_t p_vec) {
    int32x4_t s = vaddq_s32(a, b);
    // if s >= p: s -= p.  Since s < 2p for inputs in [0,p), one subtract suffices.
    int32x4_t reduced = vsubq_s32(s, p_vec);
    // Keep reduced if >= 0, else keep s
    int32x4_t mask = vshrq_n_s32(reduced, 31);  // all 1s if reduced < 0
    return vbslq_s32(vreinterpretq_u32_s32(mask), s, reduced);
}

// Full modular sub: result in [0, p).
static inline int32x4_t monty_sub_neon(int32x4_t a, int32x4_t b, int32x4_t p_vec) {
    int32x4_t diff = vsubq_s32(a, b);
    // if diff < 0: diff += p
    int32x4_t mask = vshrq_n_s32(diff, 31);
    return vaddq_s32(diff, vandq_s32(mask, p_vec));
}

// ============================================================
// Bit-reversal permutation
// ============================================================

static void bit_reverse_permute(uint32_t *data, int logN) {
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

// ============================================================
// Cached twiddle tables (Montgomery form)
// ============================================================

typedef struct {
    uint32_t *fwd;   // forward twiddles, N-1 entries
    uint32_t *inv;   // inverse twiddles, N-1 entries
    int logN;
} CachedTwiddles;

// Simple cache: one table per logN (up to 27)
static CachedTwiddles cached[28] = {{0}};
static pthread_mutex_t cache_lock = PTHREAD_MUTEX_INITIALIZER;

static void ensure_twiddles(int logN) {
    if (cached[logN].fwd) return;  // already computed

    pthread_mutex_lock(&cache_lock);
    if (cached[logN].fwd) { pthread_mutex_unlock(&cache_lock); return; }

    int n = 1 << logN;

    // Compute root of unity in plain form, then convert to Montgomery
    uint32_t omega_plain = BB_ROOT_2_27;
    for (int i = 0; i < 27 - logN; i++)
        omega_plain = plain_mul(omega_plain, omega_plain);
    uint32_t omega = to_monty(omega_plain);

    uint32_t omega_inv_plain = plain_pow(omega_plain, BB_P - 2);
    uint32_t omega_inv = to_monty(omega_inv_plain);

    uint32_t *fwd = (uint32_t *)malloc((size_t)(n - 1) * sizeof(uint32_t));
    uint32_t *inv = (uint32_t *)malloc((size_t)(n - 1) * sizeof(uint32_t));

    uint32_t one_mont = to_monty(1);

    for (int s = 0; s < logN; s++) {
        int halfBlock = 1 << s;
        int offset = halfBlock - 1;
        uint32_t w_m = monty_pow(omega, (uint32_t)(n >> (s + 1)));
        uint32_t w_m_inv = monty_pow(omega_inv, (uint32_t)(n >> (s + 1)));
        uint32_t w = one_mont;
        uint32_t wi = one_mont;
        for (int j = 0; j < halfBlock; j++) {
            fwd[offset + j] = w;
            inv[offset + j] = wi;
            w = monty_mul(w, w_m);
            wi = monty_mul(wi, w_m_inv);
        }
    }

    cached[logN].fwd = fwd;
    cached[logN].inv = inv;
    cached[logN].logN = logN;
    pthread_mutex_unlock(&cache_lock);
}

// ============================================================
// Forward NTT — Cooley-Tukey DIT (Montgomery form)
// ============================================================

void babybear_ntt_neon(uint32_t *data, int logN) {
    if (logN <= 0) return;
    int n = 1 << logN;

    ensure_twiddles(logN);
    const uint32_t *tw = cached[logN].fwd;

    // Convert input to Montgomery form
    {
        int32x4_t r2_vec = vdupq_n_s32((int32_t)MONTY_R2_MOD_P);
        int32x4_t p_inv = vdupq_n_s32(P_INV_S);
        int32x4_t pv = vdupq_n_s32(BB_P_S);
        int i = 0;
        for (; i + 3 < n; i += 4) {
            int32x4_t v = vld1q_s32((const int32_t *)&data[i]);
            v = monty_mul_neon(v, r2_vec, p_inv, pv);
            vst1q_s32((int32_t *)&data[i], v);
        }
        for (; i < n; i++)
            data[i] = to_monty(data[i]);
    }

    // Bit-reversal permutation
    bit_reverse_permute(data, logN);

    // Butterfly stages
    int32x4_t p_inv = vdupq_n_s32(P_INV_S);
    int32x4_t pv = vdupq_n_s32(BB_P_S);

    for (int s = 0; s < logN; s++) {
        int halfBlock = 1 << s;
        int blockSize = halfBlock << 1;
        int nBlocks = n / blockSize;
        int twOffset = halfBlock - 1;

        if (halfBlock >= 4) {
            if (nBlocks >= BB_NTT_PAR_THRESHOLD / halfBlock && nBlocks >= 8) {
                // Parallel NEON path
                int nChunks = 8;
                if (nBlocks < nChunks * 4) nChunks = 4;
                int chunk = nBlocks / nChunks;
                uint32_t *sd = data;
                const uint32_t *st = tw;
                int shb = halfBlock, sbs = blockSize, sto = twOffset;
                dispatch_apply(nChunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                    ^(size_t idx) {
                        int32x4_t lp_inv = vdupq_n_s32(P_INV_S);
                        int32x4_t lpv = vdupq_n_s32(BB_P_S);
                        int start = (int)idx * chunk;
                        int end = ((int)idx == nChunks - 1) ? nBlocks : start + chunk;
                        for (int bk = start; bk < end; bk++) {
                            int base = bk * sbs;
                            int j = 0;
                            for (; j + 3 < shb; j += 4) {
                                int32x4_t tw_vec = vld1q_s32((const int32_t *)&st[sto + j]);
                                int32x4_t u = vld1q_s32((const int32_t *)&sd[base + j]);
                                int32x4_t v_raw = vld1q_s32((const int32_t *)&sd[base + j + shb]);
                                int32x4_t v = monty_mul_neon(tw_vec, v_raw, lp_inv, lpv);
                                vst1q_s32((int32_t *)&sd[base + j], monty_add_neon(u, v, lpv));
                                vst1q_s32((int32_t *)&sd[base + j + shb], monty_sub_neon(u, v, lpv));
                            }
                            for (; j < shb; j++) {
                                uint32_t u = sd[base + j];
                                uint32_t v = monty_mul(st[sto + j], sd[base + j + shb]);
                                sd[base + j] = monty_add(u, v);
                                sd[base + j + shb] = monty_sub(u, v);
                            }
                        }
                    });
            } else {
                // Serial NEON path
                for (int bk = 0; bk < nBlocks; bk++) {
                    int base = bk * blockSize;
                    int j = 0;
                    for (; j + 3 < halfBlock; j += 4) {
                        int32x4_t tw_vec = vld1q_s32((const int32_t *)&tw[twOffset + j]);
                        int32x4_t u = vld1q_s32((const int32_t *)&data[base + j]);
                        int32x4_t v_raw = vld1q_s32((const int32_t *)&data[base + j + halfBlock]);
                        int32x4_t v = monty_mul_neon(tw_vec, v_raw, p_inv, pv);
                        vst1q_s32((int32_t *)&data[base + j], monty_add_neon(u, v, pv));
                        vst1q_s32((int32_t *)&data[base + j + halfBlock], monty_sub_neon(u, v, pv));
                    }
                    for (; j < halfBlock; j++) {
                        uint32_t u = data[base + j];
                        uint32_t v = monty_mul(tw[twOffset + j], data[base + j + halfBlock]);
                        data[base + j] = monty_add(u, v);
                        data[base + j + halfBlock] = monty_sub(u, v);
                    }
                }
            }
        } else {
            // Small stages: scalar
            for (int bk = 0; bk < nBlocks; bk++) {
                int base = bk * blockSize;
                // Twiddle skip: j==0 always has twiddle==1
                {
                    uint32_t u = data[base];
                    uint32_t v_raw = data[base + halfBlock];
                    data[base] = monty_add(u, v_raw);
                    data[base + halfBlock] = monty_sub(u, v_raw);
                }
                for (int j = 1; j < halfBlock; j++) {
                    uint32_t u = data[base + j];
                    uint32_t v = monty_mul(tw[twOffset + j], data[base + j + halfBlock]);
                    data[base + j] = monty_add(u, v);
                    data[base + j + halfBlock] = monty_sub(u, v);
                }
            }
        }
    }

    // Convert back from Montgomery form
    if (n >= BB_NTT_PAR_THRESHOLD) {
        uint32_t *sd = data;
        int sn = n;
        int nChunks = 8;
        int chunk = (n + nChunks - 1) / nChunks;
        chunk = (chunk + 3) & ~3;  // align to 4
        dispatch_apply(nChunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
            ^(size_t idx) {
                int32x4_t lp_inv = vdupq_n_s32(P_INV_S);
                int32x4_t lpv = vdupq_n_s32(BB_P_S);
                int32x4_t ones = vdupq_n_s32(1);
                int start = (int)idx * chunk;
                int end = start + chunk;
                if (end > sn) end = sn;
                int i = start;
                for (; i + 3 < end; i += 4) {
                    int32x4_t v = vld1q_s32((const int32_t *)&sd[i]);
                    v = monty_mul_neon(v, ones, lp_inv, lpv);
                    vst1q_s32((int32_t *)&sd[i], v);
                }
                for (; i < end; i++)
                    sd[i] = from_monty(sd[i]);
            });
    } else {
        int i = 0;
        int32x4_t ones = vdupq_n_s32(1);
        for (; i + 3 < n; i += 4) {
            int32x4_t v = vld1q_s32((const int32_t *)&data[i]);
            v = monty_mul_neon(v, ones, p_inv, pv);
            vst1q_s32((int32_t *)&data[i], v);
        }
        for (; i < n; i++)
            data[i] = from_monty(data[i]);
    }
}

// ============================================================
// Inverse NTT — Gentleman-Sande DIF (Montgomery form)
// ============================================================

void babybear_intt_neon(uint32_t *data, int logN) {
    if (logN <= 0) return;
    int n = 1 << logN;

    ensure_twiddles(logN);
    const uint32_t *tw = cached[logN].inv;

    // Convert to Montgomery form
    {
        int32x4_t r2_vec = vdupq_n_s32((int32_t)MONTY_R2_MOD_P);
        int32x4_t p_inv = vdupq_n_s32(P_INV_S);
        int32x4_t pv = vdupq_n_s32(BB_P_S);
        int i = 0;
        for (; i + 3 < n; i += 4) {
            int32x4_t v = vld1q_s32((const int32_t *)&data[i]);
            v = monty_mul_neon(v, r2_vec, p_inv, pv);
            vst1q_s32((int32_t *)&data[i], v);
        }
        for (; i < n; i++)
            data[i] = to_monty(data[i]);
    }

    int32x4_t p_inv = vdupq_n_s32(P_INV_S);
    int32x4_t pv = vdupq_n_s32(BB_P_S);

    // DIF stages (top-down)
    for (int si = 0; si < logN; si++) {
        int s = logN - 1 - si;
        int halfBlock = 1 << s;
        int blockSize = halfBlock << 1;
        int nBlocks = n / blockSize;
        int twOffset = halfBlock - 1;

        if (halfBlock >= 4) {
            if (nBlocks >= BB_NTT_PAR_THRESHOLD / halfBlock && nBlocks >= 8) {
                int nChunks = 8;
                if (nBlocks < nChunks * 4) nChunks = 4;
                int chunk = nBlocks / nChunks;
                uint32_t *sd = data;
                const uint32_t *st = tw;
                int shb = halfBlock, sbs = blockSize, sto = twOffset;
                dispatch_apply(nChunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                    ^(size_t idx) {
                        int32x4_t lp_inv = vdupq_n_s32(P_INV_S);
                        int32x4_t lpv = vdupq_n_s32(BB_P_S);
                        int start = (int)idx * chunk;
                        int end = ((int)idx == nChunks - 1) ? nBlocks : start + chunk;
                        for (int bk = start; bk < end; bk++) {
                            int base = bk * sbs;
                            int j = 0;
                            for (; j + 3 < shb; j += 4) {
                                int32x4_t tw_vec = vld1q_s32((const int32_t *)&st[sto + j]);
                                int32x4_t a = vld1q_s32((const int32_t *)&sd[base + j]);
                                int32x4_t b = vld1q_s32((const int32_t *)&sd[base + j + shb]);
                                int32x4_t sum = monty_add_neon(a, b, lpv);
                                int32x4_t diff = monty_sub_neon(a, b, lpv);
                                vst1q_s32((int32_t *)&sd[base + j], sum);
                                vst1q_s32((int32_t *)&sd[base + j + shb],
                                          monty_mul_neon(diff, tw_vec, lp_inv, lpv));
                            }
                            for (; j < shb; j++) {
                                uint32_t a = sd[base + j];
                                uint32_t b = sd[base + j + shb];
                                sd[base + j] = monty_add(a, b);
                                sd[base + j + shb] = monty_mul(monty_sub(a, b), st[sto + j]);
                            }
                        }
                    });
            } else {
                for (int bk = 0; bk < nBlocks; bk++) {
                    int base = bk * blockSize;
                    int j = 0;
                    for (; j + 3 < halfBlock; j += 4) {
                        int32x4_t tw_vec = vld1q_s32((const int32_t *)&tw[twOffset + j]);
                        int32x4_t a = vld1q_s32((const int32_t *)&data[base + j]);
                        int32x4_t b = vld1q_s32((const int32_t *)&data[base + j + halfBlock]);
                        int32x4_t sum = monty_add_neon(a, b, pv);
                        int32x4_t diff = monty_sub_neon(a, b, pv);
                        vst1q_s32((int32_t *)&data[base + j], sum);
                        vst1q_s32((int32_t *)&data[base + j + halfBlock],
                                  monty_mul_neon(diff, tw_vec, p_inv, pv));
                    }
                    for (; j < halfBlock; j++) {
                        uint32_t a = data[base + j];
                        uint32_t b = data[base + j + halfBlock];
                        data[base + j] = monty_add(a, b);
                        data[base + j + halfBlock] = monty_mul(monty_sub(a, b), tw[twOffset + j]);
                    }
                }
            }
        } else {
            for (int bk = 0; bk < nBlocks; bk++) {
                int base = bk * blockSize;
                // Twiddle skip: j==0 has twiddle==1
                {
                    uint32_t a = data[base];
                    uint32_t b = data[base + halfBlock];
                    data[base] = monty_add(a, b);
                    data[base + halfBlock] = monty_sub(a, b);
                }
                for (int j = 1; j < halfBlock; j++) {
                    uint32_t a = data[base + j];
                    uint32_t b = data[base + j + halfBlock];
                    data[base + j] = monty_add(a, b);
                    data[base + j + halfBlock] = monty_mul(monty_sub(a, b), tw[twOffset + j]);
                }
            }
        }
    }

    // Bit-reversal permutation
    bit_reverse_permute(data, logN);

    // Scale by 1/n and convert from Montgomery form in one step:
    // from_monty(monty_mul(x, n_inv_mont)) = x * n_inv_mont * R^{-1} * R^{-1} ... no, that's wrong.
    // Just: from_monty(monty_mul(x, n_inv_mont)) where n_inv_mont = to_monty(n^{-1} mod p)
    // monty_mul gives x * n_inv_mont * R^{-1}, then from_monty gives * R^{-1} again.
    // That's one R^{-1} too many.
    //
    // Correct: monty_mul(x, n_inv_mont) gives (x * n_inv * R) * R^{-1} = x * n_inv (in Mont form).
    // Then from_monty gives x * n_inv (in plain form). Two steps.
    //
    // Or: single step — monty_mul(x, plain_n_inv) where plain_n_inv is NOT in Montgomery form:
    //   = x * plain_n_inv * R^{-1}. Then from_monty gives x * plain_n_inv * R^{-2} — wrong.
    //
    // Simplest correct: fuse scale + from_monty by multiplying by n_inv (plain, not Montgomery).
    // monty_reduce64(x * n_inv_plain) = x * n_inv_plain * R^{-1}.
    // But x is in Montgomery form = x_real * R, so:
    // result = x_real * R * n_inv_plain * R^{-1} = x_real * n_inv_plain. Correct!
    {
        uint32_t n_inv_plain = plain_pow(n, BB_P - 2);  // n^{-1} mod p (plain form)
        int32x4_t n_inv_vec = vdupq_n_s32((int32_t)n_inv_plain);
        int i = 0;
        for (; i + 3 < n; i += 4) {
            int32x4_t v = vld1q_s32((const int32_t *)&data[i]);
            v = monty_mul_neon(v, n_inv_vec, p_inv, pv);
            vst1q_s32((int32_t *)&data[i], v);
        }
        for (; i < n; i++)
            data[i] = monty_reduce64((uint64_t)data[i] * n_inv_plain);
    }
}
