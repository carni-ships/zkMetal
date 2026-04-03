// Goldilocks NTT with optimized ARM64 scalar arithmetic
// p = 2^64 - 2^32 + 1 = 0xFFFFFFFF00000001
//
// Uses __uint128_t for 64x64->128 multiply (compiles to ARM64 mul+umulh pair).
// Branchless reduction exploiting the special prime structure.
// Cached twiddle tables. Fully unrolled carry chains.

#include "NeonFieldOps.h"
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

// ============================================================
// Constants
// ============================================================

#define GL_P          0xFFFFFFFF00000001ULL
#define GL_EPS        0xFFFFFFFFULL           // 2^32 - 1
#define GL_ROOT_2_32  1753635133440165772ULL   // primitive 2^32-th root of unity
#define GL_TWO_ADICITY 32

typedef unsigned __int128 uint128_t;

// ============================================================
// Branchless field arithmetic
// ============================================================

// Reduce a 128-bit product mod p = 2^64 - 2^32 + 1.
// Uses the identity: 2^64 ≡ 2^32 - 1 (mod p).
// So (hi:lo) = hi * 2^64 + lo ≡ hi * (2^32 - 1) + lo (mod p).
//
// Split hi = hi_hi * 2^32 + hi_lo:
//   hi * eps = hi_lo * eps + hi_hi * eps * 2^32
// But eps * 2^32 = 2^64 - 2^32 ≡ eps - 2^32 + eps = ... gets circular.
//
// Direct approach: t = lo + hi_lo * eps - hi_hi, with overflow/underflow fixup.
static inline uint64_t gl_reduce128(uint64_t hi, uint64_t lo) {
    uint64_t hi_lo = hi & 0xFFFFFFFFULL;
    uint64_t hi_hi = hi >> 32;

    // t1 = lo + hi_lo * eps (may overflow 64 bits by at most eps)
    uint64_t hi_lo_eps = hi_lo * GL_EPS;  // 32×32 = 64 bits, no overflow
    uint64_t t1;
    unsigned c1 = __builtin_add_overflow(lo, hi_lo_eps, &t1);

    // t2 = t1 - hi_hi (may underflow)
    uint64_t t2;
    unsigned b2 = __builtin_sub_overflow(t1, hi_hi, &t2);

    // Adjust: overflow means +2^64 ≡ +eps; underflow means -2^64 ≡ -eps, but we add p
    uint64_t r = t2;
    if (c1) r += GL_EPS;   // carry: added 2^64 worth, which ≡ eps (mod p)
    if (b2) r += GL_P;     // borrow: went negative, add p
    if (r >= GL_P) r -= GL_P;
    return r;
}

static inline uint64_t gl_mul(uint64_t a, uint64_t b) {
    uint128_t prod = (uint128_t)a * b;
    return gl_reduce128((uint64_t)(prod >> 64), (uint64_t)prod);
}

static inline uint64_t gl_sqr(uint64_t a) {
    return gl_mul(a, a);
}

static inline uint64_t gl_add(uint64_t a, uint64_t b) {
    uint64_t s;
    unsigned carry = __builtin_add_overflow(a, b, &s);
    // If carry, s wrapped around; add eps (since 2^64 ≡ eps mod p)
    if (carry) {
        s += GL_EPS;
        // After adding eps, could still be >= p (very rare)
        if (s < GL_EPS) s += GL_EPS;  // second wrap (only if s+eps overflowed)
    }
    return s >= GL_P ? s - GL_P : s;
}

static inline uint64_t gl_sub(uint64_t a, uint64_t b) {
    if (a >= b) return a - b;
    return a + GL_P - b;
}

static inline uint64_t gl_pow(uint64_t base, uint64_t exp) {
    uint64_t result = 1;
    uint64_t b = base;
    while (exp > 0) {
        if (exp & 1) result = gl_mul(result, b);
        b = gl_sqr(b);
        exp >>= 1;
    }
    return result;
}

static inline uint64_t gl_inv(uint64_t a) {
    return gl_pow(a, GL_P - 2);
}

// ============================================================
// Bit-reversal permutation (64-bit elements)
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
// Cached twiddle tables
// ============================================================

typedef struct {
    uint64_t *fwd;   // forward twiddles
    uint64_t *inv;   // inverse twiddles
    int logN;
} GlCachedTwiddles;

static GlCachedTwiddles gl_cached[33] = {{0}};
static pthread_mutex_t gl_cache_lock = PTHREAD_MUTEX_INITIALIZER;

static void gl_ensure_twiddles(int logN) {
    if (gl_cached[logN].fwd) return;

    pthread_mutex_lock(&gl_cache_lock);
    if (gl_cached[logN].fwd) { pthread_mutex_unlock(&gl_cache_lock); return; }

    int n = 1 << logN;

    // Compute root of unity: omega = ROOT_2_32^(2^(32-logN))
    uint64_t omega = GL_ROOT_2_32;
    for (int i = 0; i < GL_TWO_ADICITY - logN; i++)
        omega = gl_sqr(omega);

    uint64_t omega_inv = gl_inv(omega);

    uint64_t *fwd = (uint64_t *)malloc((size_t)(n - 1) * sizeof(uint64_t));
    uint64_t *inv = (uint64_t *)malloc((size_t)(n - 1) * sizeof(uint64_t));

    for (int s = 0; s < logN; s++) {
        int halfBlock = 1 << s;
        int offset = halfBlock - 1;
        uint64_t w_m = gl_pow(omega, (uint64_t)(n >> (s + 1)));
        uint64_t w_m_inv = gl_pow(omega_inv, (uint64_t)(n >> (s + 1)));
        uint64_t w = 1;
        uint64_t wi = 1;
        for (int j = 0; j < halfBlock; j++) {
            fwd[offset + j] = w;
            inv[offset + j] = wi;
            w = gl_mul(w, w_m);
            wi = gl_mul(wi, w_m_inv);
        }
    }

    gl_cached[logN].fwd = fwd;
    gl_cached[logN].inv = inv;
    gl_cached[logN].logN = logN;
    pthread_mutex_unlock(&gl_cache_lock);
}

// ============================================================
// Forward NTT — Cooley-Tukey DIT
// ============================================================

void goldilocks_ntt(uint64_t *data, int logN) {
    if (logN <= 0) return;
    int n = 1 << logN;

    gl_ensure_twiddles(logN);
    const uint64_t *tw = gl_cached[logN].fwd;

    bit_reverse_permute64(data, logN);

    for (int s = 0; s < logN; s++) {
        int halfBlock = 1 << s;
        int blockSize = halfBlock << 1;
        int nBlocks = n / blockSize;
        int twOffset = halfBlock - 1;

        for (int bk = 0; bk < nBlocks; bk++) {
            int base = bk * blockSize;
            for (int j = 0; j < halfBlock; j++) {
                uint64_t u = data[base + j];
                uint64_t v = gl_mul(tw[twOffset + j], data[base + j + halfBlock]);
                data[base + j] = gl_add(u, v);
                data[base + j + halfBlock] = gl_sub(u, v);
            }
        }
    }
}

// ============================================================
// Inverse NTT — Gentleman-Sande DIF
// ============================================================

void goldilocks_intt(uint64_t *data, int logN) {
    if (logN <= 0) return;
    int n = 1 << logN;

    gl_ensure_twiddles(logN);
    const uint64_t *tw = gl_cached[logN].inv;

    // DIF stages (top-down)
    for (int si = 0; si < logN; si++) {
        int s = logN - 1 - si;
        int halfBlock = 1 << s;
        int blockSize = halfBlock << 1;
        int nBlocks = n / blockSize;
        int twOffset = halfBlock - 1;

        for (int bk = 0; bk < nBlocks; bk++) {
            int base = bk * blockSize;
            for (int j = 0; j < halfBlock; j++) {
                uint64_t a = data[base + j];
                uint64_t b = data[base + j + halfBlock];
                data[base + j] = gl_add(a, b);
                data[base + j + halfBlock] = gl_mul(gl_sub(a, b), tw[twOffset + j]);
            }
        }
    }

    bit_reverse_permute64(data, logN);

    // Scale by 1/n
    uint64_t n_inv = gl_inv((uint64_t)n);
    for (int i = 0; i < n; i++) {
        data[i] = gl_mul(data[i], n_inv);
    }
}
