// NEON-optimized Goldilocks field operations
// p = 2^64 - 2^32 + 1 = 0xFFFFFFFF00000001
//
// Uses ARM NEON uint64x2_t for 2-wide vectorized add/sub.
// For mul: uses scalar interleaved pairs (ARM64 mul+umulh) since
// NEON lacks 64x64->128 multiply.

#include "NeonFieldOps.h"
#include <arm_neon.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

// ============================================================
// Constants
// ============================================================

#define GL_P          0xFFFFFFFF00000001ULL
#define GL_EPS        0xFFFFFFFFULL           // 2^32 - 1
#define GL_ROOT_2_32  1753635133440165772ULL
#define GL_TWO_ADICITY 32

typedef unsigned __int128 uint128_t;

// NEON constant vectors
static inline uint64x2_t gl_p_vec(void) { return vdupq_n_u64(GL_P); }
static inline uint64x2_t gl_eps_vec(void) { return vdupq_n_u64(GL_EPS); }

// ============================================================
// Scalar field arithmetic (same as goldilocks_ntt.c, inlined here)
// ============================================================

static inline uint64_t gl_reduce128(uint64_t hi, uint64_t lo) {
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

static inline uint64_t gl_mul_scalar(uint64_t a, uint64_t b) {
    uint128_t prod = (uint128_t)a * b;
    return gl_reduce128((uint64_t)(prod >> 64), (uint64_t)prod);
}

static inline uint64_t gl_sqr_scalar(uint64_t a) {
    return gl_mul_scalar(a, a);
}

static inline uint64_t gl_add_scalar(uint64_t a, uint64_t b) {
    uint64_t s;
    unsigned carry = __builtin_add_overflow(a, b, &s);
    if (carry) {
        s += GL_EPS;
        if (s < GL_EPS) s += GL_EPS;
    }
    return s >= GL_P ? s - GL_P : s;
}

static inline uint64_t gl_sub_scalar(uint64_t a, uint64_t b) {
    if (a >= b) return a - b;
    return a + GL_P - b;
}

static inline uint64_t gl_pow_scalar(uint64_t base, uint64_t exp) {
    uint64_t result = 1;
    uint64_t b = base;
    while (exp > 0) {
        if (exp & 1) result = gl_mul_scalar(result, b);
        b = gl_sqr_scalar(b);
        exp >>= 1;
    }
    return result;
}

static inline uint64_t gl_inv_scalar(uint64_t a) {
    return gl_pow_scalar(a, GL_P - 2);
}

// ============================================================
// NEON vectorized add/sub (2-wide)
// ============================================================

// Branchless: result = a + b mod p
// If a + b overflows 64 bits, add eps (since 2^64 ≡ eps mod p)
// Then if result >= p, subtract p
static inline uint64x2_t gl_add_vec(uint64x2_t a, uint64x2_t b) {
    uint64x2_t p = gl_p_vec();
    uint64x2_t eps = gl_eps_vec();

    uint64x2_t sum = vaddq_u64(a, b);
    // Detect carry: if sum < a, there was overflow
    uint64x2_t carry_mask = vcltq_u64(sum, a);
    // carry_mask is all-1s where carry happened; eps & mask = eps or 0
    sum = vaddq_u64(sum, vandq_u64(eps, carry_mask));

    // Handle double overflow from adding eps (extremely rare but correct)
    // If sum < eps after adding eps, we overflowed again
    uint64x2_t carry2_mask = vandq_u64(carry_mask, vcltq_u64(sum, eps));
    sum = vaddq_u64(sum, vandq_u64(eps, carry2_mask));

    // Conditional subtract p if sum >= p
    uint64x2_t ge_mask = vcgeq_u64(sum, p);
    sum = vsubq_u64(sum, vandq_u64(p, ge_mask));
    return sum;
}

// Branchless: result = a - b mod p
// If a < b (borrow), add p
static inline uint64x2_t gl_sub_vec(uint64x2_t a, uint64x2_t b) {
    uint64x2_t p = gl_p_vec();
    uint64x2_t diff = vsubq_u64(a, b);
    // Detect borrow: if a < b
    uint64x2_t borrow_mask = vcltq_u64(a, b);
    diff = vaddq_u64(diff, vandq_u64(p, borrow_mask));
    return diff;
}

// ============================================================
// Batch add/sub using NEON
// ============================================================

void gl_batch_add_neon(const uint64_t *a, const uint64_t *b, uint64_t *out, int n) {
    int i = 0;
    for (; i + 1 < n; i += 2) {
        uint64x2_t va = vld1q_u64(a + i);
        uint64x2_t vb = vld1q_u64(b + i);
        uint64x2_t vr = gl_add_vec(va, vb);
        vst1q_u64(out + i, vr);
    }
    // Tail element
    if (i < n) {
        out[i] = gl_add_scalar(a[i], b[i]);
    }
}

void gl_batch_sub_neon(const uint64_t *a, const uint64_t *b, uint64_t *out, int n) {
    int i = 0;
    for (; i + 1 < n; i += 2) {
        uint64x2_t va = vld1q_u64(a + i);
        uint64x2_t vb = vld1q_u64(b + i);
        uint64x2_t vr = gl_sub_vec(va, vb);
        vst1q_u64(out + i, vr);
    }
    if (i < n) {
        out[i] = gl_sub_scalar(a[i], b[i]);
    }
}

// ============================================================
// Batch multiply using interleaved scalar (ILP)
// ============================================================

void gl_batch_mul_neon(const uint64_t *a, const uint64_t *b, uint64_t *out, int n) {
    // Use interleaved scalar muls for ILP — ARM64 has 2 multiply units
    int i = 0;
    for (; i + 3 < n; i += 4) {
        // Issue 4 multiplies to saturate pipelines
        uint128_t p0 = (uint128_t)a[i]   * b[i];
        uint128_t p1 = (uint128_t)a[i+1] * b[i+1];
        uint128_t p2 = (uint128_t)a[i+2] * b[i+2];
        uint128_t p3 = (uint128_t)a[i+3] * b[i+3];
        out[i]   = gl_reduce128((uint64_t)(p0 >> 64), (uint64_t)p0);
        out[i+1] = gl_reduce128((uint64_t)(p1 >> 64), (uint64_t)p1);
        out[i+2] = gl_reduce128((uint64_t)(p2 >> 64), (uint64_t)p2);
        out[i+3] = gl_reduce128((uint64_t)(p3 >> 64), (uint64_t)p3);
    }
    for (; i < n; i++) {
        out[i] = gl_mul_scalar(a[i], b[i]);
    }
}

// ============================================================
// NEON vectorized butterfly
// ============================================================

// Butterfly: out_top = a + b*w, out_bot = a - b*w
// The multiply b*w is scalar (no NEON 64x64->128), but add/sub use NEON
static inline void gl_butterfly_neon_pair(
    uint64_t *top0, uint64_t *top1,
    uint64_t *bot0, uint64_t *bot1,
    uint64_t a0, uint64_t a1,
    uint64_t b0, uint64_t b1,
    uint64_t w0, uint64_t w1)
{
    // Scalar muls (interleaved for ILP)
    uint64_t bw0 = gl_mul_scalar(b0, w0);
    uint64_t bw1 = gl_mul_scalar(b1, w1);

    // NEON add/sub
    uint64x2_t va = {a0, a1};
    uint64x2_t vbw = {bw0, bw1};
    uint64x2_t vsum = gl_add_vec(va, vbw);
    uint64x2_t vdif = gl_sub_vec(va, vbw);

    *top0 = vgetq_lane_u64(vsum, 0);
    *top1 = vgetq_lane_u64(vsum, 1);
    *bot0 = vgetq_lane_u64(vdif, 0);
    *bot1 = vgetq_lane_u64(vdif, 1);
}

// ============================================================
// Batch butterfly for NTT stages
// ============================================================

void gl_batch_butterfly_neon(uint64_t *data, const uint64_t *twiddles, int halfBlock, int nBlocks) {
    for (int bk = 0; bk < nBlocks; bk++) {
        int base = bk * (halfBlock << 1);
        // Twiddle skip: j==0 always has twiddle==1, skip expensive 64-bit mul
        {
            uint64_t a0 = data[base];
            uint64_t b0 = data[base + halfBlock];
            data[base] = gl_add_scalar(a0, b0);
            data[base + halfBlock] = gl_sub_scalar(a0, b0);
        }
        int j = 1;
        // Process pairs of butterflies with NEON add/sub
        for (; j + 1 < halfBlock; j += 2) {
            uint64_t a0 = data[base + j];
            uint64_t a1 = data[base + j + 1];
            uint64_t b0 = data[base + j + halfBlock];
            uint64_t b1 = data[base + j + 1 + halfBlock];

            // Scalar twiddle multiplies (interleaved for ILP)
            uint64_t bw0 = gl_mul_scalar(b0, twiddles[j]);
            uint64_t bw1 = gl_mul_scalar(b1, twiddles[j + 1]);

            // NEON vectorized add/sub
            uint64x2_t va  = {a0, a1};
            uint64x2_t vbw = {bw0, bw1};
            uint64x2_t vsum = gl_add_vec(va, vbw);
            uint64x2_t vdif = gl_sub_vec(va, vbw);

            vst1q_u64(data + base + j, vsum);
            vst1q_u64(data + base + j + halfBlock, vdif);
        }
        // Tail
        if (j < halfBlock) {
            uint64_t u = data[base + j];
            uint64_t v = gl_mul_scalar(data[base + j + halfBlock], twiddles[j]);
            data[base + j] = gl_add_scalar(u, v);
            data[base + j + halfBlock] = gl_sub_scalar(u, v);
        }
    }
}

// DIF butterfly: out_top = a + b, out_bot = (a - b) * w
void gl_batch_butterfly_dif_neon(uint64_t *data, const uint64_t *twiddles, int halfBlock, int nBlocks) {
    for (int bk = 0; bk < nBlocks; bk++) {
        int base = bk * (halfBlock << 1);
        // Twiddle skip: j==0 has twiddle==1, diff * 1 = diff (skip mul)
        {
            uint64_t a0 = data[base];
            uint64_t b0 = data[base + halfBlock];
            data[base] = gl_add_scalar(a0, b0);
            data[base + halfBlock] = gl_sub_scalar(a0, b0);
        }
        int j = 1;
        for (; j + 1 < halfBlock; j += 2) {
            uint64x2_t va = vld1q_u64(data + base + j);
            uint64x2_t vb = vld1q_u64(data + base + j + halfBlock);
            uint64x2_t vsum = gl_add_vec(va, vb);
            uint64x2_t vdif = gl_sub_vec(va, vb);

            vst1q_u64(data + base + j, vsum);

            // Scalar twiddle multiply on the difference
            uint64_t d0 = vgetq_lane_u64(vdif, 0);
            uint64_t d1 = vgetq_lane_u64(vdif, 1);
            uint64_t r0 = gl_mul_scalar(d0, twiddles[j]);
            uint64_t r1 = gl_mul_scalar(d1, twiddles[j + 1]);
            uint64x2_t vr = {r0, r1};
            vst1q_u64(data + base + j + halfBlock, vr);
        }
        if (j < halfBlock) {
            uint64_t a = data[base + j];
            uint64_t b = data[base + j + halfBlock];
            data[base + j] = gl_add_scalar(a, b);
            data[base + j + halfBlock] = gl_mul_scalar(gl_sub_scalar(a, b), twiddles[j]);
        }
    }
}

// ============================================================
// Bit-reversal permutation
// ============================================================

static void bit_reverse_permute64(uint64_t *data, int logN) {
    int n = 1 << logN;
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) {
            uint64_t tmp = data[i];
            data[i] = data[j];
            data[j] = tmp;
        }
    }
}

// ============================================================
// Cached twiddle tables (shared with scalar NTT)
// ============================================================

typedef struct {
    uint64_t *fwd;
    uint64_t *inv;
    int logN;
} GlNeonCachedTwiddles;

static GlNeonCachedTwiddles gl_neon_cached[33] = {{0}};
static pthread_mutex_t gl_neon_cache_lock = PTHREAD_MUTEX_INITIALIZER;

static void gl_neon_ensure_twiddles(int logN) {
    if (gl_neon_cached[logN].fwd) return;

    pthread_mutex_lock(&gl_neon_cache_lock);
    if (gl_neon_cached[logN].fwd) { pthread_mutex_unlock(&gl_neon_cache_lock); return; }

    int n = 1 << logN;

    uint64_t omega = GL_ROOT_2_32;
    for (int i = 0; i < GL_TWO_ADICITY - logN; i++)
        omega = gl_sqr_scalar(omega);

    uint64_t omega_inv = gl_inv_scalar(omega);

    uint64_t *fwd = (uint64_t *)malloc((size_t)(n - 1) * sizeof(uint64_t));
    uint64_t *inv = (uint64_t *)malloc((size_t)(n - 1) * sizeof(uint64_t));

    for (int s = 0; s < logN; s++) {
        int halfBlock = 1 << s;
        int offset = halfBlock - 1;
        uint64_t w_m = gl_pow_scalar(omega, (uint64_t)(n >> (s + 1)));
        uint64_t w_m_inv = gl_pow_scalar(omega_inv, (uint64_t)(n >> (s + 1)));
        uint64_t w = 1;
        uint64_t wi = 1;
        for (int j = 0; j < halfBlock; j++) {
            fwd[offset + j] = w;
            inv[offset + j] = wi;
            w = gl_mul_scalar(w, w_m);
            wi = gl_mul_scalar(wi, w_m_inv);
        }
    }

    gl_neon_cached[logN].fwd = fwd;
    gl_neon_cached[logN].inv = inv;
    gl_neon_cached[logN].logN = logN;
    pthread_mutex_unlock(&gl_neon_cache_lock);
}

// ============================================================
// Forward NTT — Cooley-Tukey DIT with NEON butterflies
// ============================================================

void goldilocks_ntt_neon(uint64_t *data, int logN) {
    if (logN <= 0) return;
    int n = 1 << logN;

    gl_neon_ensure_twiddles(logN);
    const uint64_t *tw = gl_neon_cached[logN].fwd;

    bit_reverse_permute64(data, logN);

    for (int s = 0; s < logN; s++) {
        int halfBlock = 1 << s;
        int blockSize = halfBlock << 1;
        int nBlocks = n / blockSize;
        int twOffset = halfBlock - 1;

        gl_batch_butterfly_neon(data, tw + twOffset, halfBlock, nBlocks);
    }
}

// ============================================================
// Inverse NTT — Gentleman-Sande DIF with NEON butterflies
// ============================================================

void goldilocks_intt_neon(uint64_t *data, int logN) {
    if (logN <= 0) return;
    int n = 1 << logN;

    gl_neon_ensure_twiddles(logN);
    const uint64_t *tw = gl_neon_cached[logN].inv;

    // DIF stages (top-down)
    for (int si = 0; si < logN; si++) {
        int s = logN - 1 - si;
        int halfBlock = 1 << s;
        int blockSize = halfBlock << 1;
        int nBlocks = n / blockSize;
        int twOffset = halfBlock - 1;

        gl_batch_butterfly_dif_neon(data, tw + twOffset, halfBlock, nBlocks);
    }

    bit_reverse_permute64(data, logN);

    // Scale by 1/n
    uint64_t n_inv = gl_inv_scalar((uint64_t)n);
    int i = 0;
    // Batch multiply by n_inv using interleaved scalar
    for (; i + 3 < n; i += 4) {
        uint128_t p0 = (uint128_t)data[i]   * n_inv;
        uint128_t p1 = (uint128_t)data[i+1] * n_inv;
        uint128_t p2 = (uint128_t)data[i+2] * n_inv;
        uint128_t p3 = (uint128_t)data[i+3] * n_inv;
        data[i]   = gl_reduce128((uint64_t)(p0 >> 64), (uint64_t)p0);
        data[i+1] = gl_reduce128((uint64_t)(p1 >> 64), (uint64_t)p1);
        data[i+2] = gl_reduce128((uint64_t)(p2 >> 64), (uint64_t)p2);
        data[i+3] = gl_reduce128((uint64_t)(p3 >> 64), (uint64_t)p3);
    }
    for (; i < n; i++) {
        data[i] = gl_mul_scalar(data[i], n_inv);
    }
}
