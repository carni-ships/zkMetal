// Optimized 256-bit Montgomery multiplication for ARM64
// Fully unrolled CIOS (Coarsely Integrated Operand Scanning) with __uint128_t.
//
// Supports: BN254 Fr/Fp, BLS12-377 Fr, secp256k1 Fp
// Compiles to pipelined ARM64 mul+umulh pairs with adcs carry chains.

#include "NeonFieldOps.h"
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <dispatch/dispatch.h>

typedef unsigned __int128 uint128_t;

// ============================================================
// Generic 4-limb CIOS Montgomery multiply
// ============================================================
//
// Computes: result = (a * b * R^{-1}) mod p
// Where R = 2^256, a and b are in Montgomery form.
//
// CIOS: for each limb of a, multiply-accumulate with all limbs of b,
// then immediately reduce by one limb using Montgomery's trick.

static inline void mont_mul_4limb(
    const uint64_t a[4], const uint64_t b[4],
    const uint64_t p[4], uint64_t inv,
    uint64_t result[4])
{
    // 5-limb accumulator
    uint64_t t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0;

    // Iteration i=0: multiply a[0] * b[j], accumulate into t
    {
        uint128_t w;
        w = (uint128_t)a[0] * b[0];
        t0 = (uint64_t)w; uint64_t c = (uint64_t)(w >> 64);
        w = (uint128_t)a[0] * b[1] + t1 + c;
        t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[0] * b[2] + t2 + c;
        t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[0] * b[3] + t3 + c;
        t3 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t4 += c;

        // Montgomery reduction for t0
        uint64_t m = t0 * inv;
        w = (uint128_t)m * p[0] + t0;
        c = (uint64_t)(w >> 64);
        w = (uint128_t)m * p[1] + t1 + c;
        t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * p[2] + t2 + c;
        t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * p[3] + t3 + c;
        t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t3 = t4 + c;
        t4 = 0;
    }

    // Iteration i=1
    {
        uint128_t w;
        uint64_t c;
        w = (uint128_t)a[1] * b[0] + t0;
        t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[1] * b[1] + t1 + c;
        t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[1] * b[2] + t2 + c;
        t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[1] * b[3] + t3 + c;
        t3 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t4 += c;

        uint64_t m = t0 * inv;
        w = (uint128_t)m * p[0] + t0;
        c = (uint64_t)(w >> 64);
        w = (uint128_t)m * p[1] + t1 + c;
        t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * p[2] + t2 + c;
        t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * p[3] + t3 + c;
        t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t3 = t4 + c;
        t4 = 0;
    }

    // Iteration i=2
    {
        uint128_t w;
        uint64_t c;
        w = (uint128_t)a[2] * b[0] + t0;
        t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[2] * b[1] + t1 + c;
        t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[2] * b[2] + t2 + c;
        t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[2] * b[3] + t3 + c;
        t3 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t4 += c;

        uint64_t m = t0 * inv;
        w = (uint128_t)m * p[0] + t0;
        c = (uint64_t)(w >> 64);
        w = (uint128_t)m * p[1] + t1 + c;
        t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * p[2] + t2 + c;
        t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * p[3] + t3 + c;
        t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t3 = t4 + c;
        t4 = 0;
    }

    // Iteration i=3
    {
        uint128_t w;
        uint64_t c;
        w = (uint128_t)a[3] * b[0] + t0;
        t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[3] * b[1] + t1 + c;
        t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[3] * b[2] + t2 + c;
        t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[3] * b[3] + t3 + c;
        t3 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t4 += c;

        uint64_t m = t0 * inv;
        w = (uint128_t)m * p[0] + t0;
        c = (uint64_t)(w >> 64);
        w = (uint128_t)m * p[1] + t1 + c;
        t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * p[2] + t2 + c;
        t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * p[3] + t3 + c;
        t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t3 = t4 + c;
        t4 = 0;
    }

    // Final conditional subtraction
    // If result >= p, subtract p
    uint64_t borrow;
    uint64_t r0, r1, r2, r3;

    borrow = 0;
    uint128_t d;
    d = (uint128_t)t0 - p[0] - borrow;
    r0 = (uint64_t)d;
    borrow = (d >> 127) & 1;  // borrow bit
    d = (uint128_t)t1 - p[1] - borrow;
    r1 = (uint64_t)d;
    borrow = (d >> 127) & 1;
    d = (uint128_t)t2 - p[2] - borrow;
    r2 = (uint64_t)d;
    borrow = (d >> 127) & 1;
    d = (uint128_t)t3 - p[3] - borrow;
    r3 = (uint64_t)d;
    borrow = (d >> 127) & 1;

    // If no borrow, use subtracted result; otherwise keep original
    if (!borrow) {
        result[0] = r0; result[1] = r1;
        result[2] = r2; result[3] = r3;
    } else {
        result[0] = t0; result[1] = t1;
        result[2] = t2; result[3] = t3;
    }
}

// 4-limb add: result = (a + b) mod p
static inline void mont_add_4limb(
    const uint64_t a[4], const uint64_t b[4],
    const uint64_t p[4], uint64_t result[4])
{
    uint128_t w;
    uint64_t c = 0;
    w = (uint128_t)a[0] + b[0];
    result[0] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[1] + b[1] + c;
    result[1] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[2] + b[2] + c;
    result[2] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[3] + b[3] + c;
    result[3] = (uint64_t)w; c = (uint64_t)(w >> 64);

    // Conditional subtract p if result >= p
    uint64_t borrow = 0;
    uint64_t r0, r1, r2, r3;
    uint128_t d;
    d = (uint128_t)result[0] - p[0];
    r0 = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)result[1] - p[1] - borrow;
    r1 = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)result[2] - p[2] - borrow;
    r2 = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)result[3] - p[3] - borrow;
    r3 = (uint64_t)d; borrow = (d >> 127) & 1;

    if (c || !borrow) {
        result[0] = r0; result[1] = r1;
        result[2] = r2; result[3] = r3;
    }
}

// 4-limb sub: result = (a - b) mod p
static inline void mont_sub_4limb(
    const uint64_t a[4], const uint64_t b[4],
    const uint64_t p[4], uint64_t result[4])
{
    uint128_t d;
    uint64_t borrow = 0;
    d = (uint128_t)a[0] - b[0];
    result[0] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)a[1] - b[1] - borrow;
    result[1] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)a[2] - b[2] - borrow;
    result[2] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)a[3] - b[3] - borrow;
    result[3] = (uint64_t)d; borrow = (d >> 127) & 1;

    // If borrow, add p
    if (borrow) {
        uint64_t c = 0;
        uint128_t w;
        w = (uint128_t)result[0] + p[0];
        result[0] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)result[1] + p[1] + c;
        result[1] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)result[2] + p[2] + c;
        result[2] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)result[3] + p[3] + c;
        result[3] = (uint64_t)w;
    }
}

// ============================================================
// BN254 Fr NTT
// ============================================================

static const uint64_t BN254_FR_P[4] = {
    0x43e1f593f0000001ULL, 0x2833e84879b97091ULL,
    0xb85045b68181585dULL, 0x30644e72e131a029ULL
};
static const uint64_t BN254_FR_INV = 0xc2e1f593efffffffULL;

// Root of unity: stored in Montgomery form (4 limbs)
static const uint64_t BN254_FR_ROOT_2_28[4] = {
    0x636e735580d13d9cULL, 0xa22bf3742445ffd6ULL,
    0x56452ac01eb203d8ULL, 0x1860ef942963f9e7ULL
};
static const uint64_t BN254_FR_TWO_ADICITY = 28;

// Montgomery form of 1 (R mod p)
static const uint64_t BN254_FR_ONE[4] = {
    0xac96341c4ffffffbULL, 0x36fc76959f60cd29ULL,
    0x666ea36f7879462eULL, 0x0e0a77c19a07df2fULL
};

// ============================================================
// Twiddle cache for BN254 Fr NTT
// ============================================================

typedef struct {
    uint64_t *fwd;   // forward twiddles (4 limbs each)
    uint64_t *inv;   // inverse twiddles (4 limbs each)
    int logN;
} FrCachedTwiddles;

static FrCachedTwiddles fr_cached[29] = {{0}};
static pthread_mutex_t fr_cache_lock = PTHREAD_MUTEX_INITIALIZER;

static void fr_pow(const uint64_t base[4], uint64_t exp_val,
                   uint64_t result[4]) {
    memcpy(result, BN254_FR_ONE, 32);
    uint64_t b[4];
    memcpy(b, base, 32);
    while (exp_val > 0) {
        if (exp_val & 1)
            mont_mul_4limb(result, b, BN254_FR_P, BN254_FR_INV, result);
        mont_mul_4limb(b, b, BN254_FR_P, BN254_FR_INV, b);
        exp_val >>= 1;
    }
}

// Compute a^(p-2) mod p for inversion
static void fr_inv(const uint64_t a[4], uint64_t result[4]) {
    // Use p - 2 as exponent (need full 256-bit exp)
    // p - 2 in limbs: subtract 2 from P
    uint64_t pm2[4];
    pm2[0] = BN254_FR_P[0] - 2;
    pm2[1] = BN254_FR_P[1];
    pm2[2] = BN254_FR_P[2];
    pm2[3] = BN254_FR_P[3];

    memcpy(result, BN254_FR_ONE, 32);
    uint64_t b[4];
    memcpy(b, a, 32);
    for (int i = 0; i < 4; i++) {
        for (int bit = 0; bit < 64; bit++) {
            if ((pm2[i] >> bit) & 1)
                mont_mul_4limb(result, b, BN254_FR_P, BN254_FR_INV, result);
            mont_mul_4limb(b, b, BN254_FR_P, BN254_FR_INV, b);
        }
    }
}

static void fr_ensure_twiddles(int logN) {
    if (fr_cached[logN].fwd) return;

    pthread_mutex_lock(&fr_cache_lock);
    if (fr_cached[logN].fwd) { pthread_mutex_unlock(&fr_cache_lock); return; }

    int n = 1 << logN;

    // omega = ROOT^(2^(TWO_ADICITY - logN))
    uint64_t omega[4];
    memcpy(omega, BN254_FR_ROOT_2_28, 32);
    for (int i = 0; i < (int)BN254_FR_TWO_ADICITY - logN; i++)
        mont_mul_4limb(omega, omega, BN254_FR_P, BN254_FR_INV, omega);

    uint64_t omega_inv[4];
    fr_inv(omega, omega_inv);

    // Allocate: (n-1) twiddles, each 4 limbs = 32 bytes
    uint64_t *fwd = (uint64_t *)malloc((size_t)(n - 1) * 32);
    uint64_t *inv = (uint64_t *)malloc((size_t)(n - 1) * 32);

    for (int s = 0; s < logN; s++) {
        int halfBlock = 1 << s;
        int offset = halfBlock - 1;

        // w_m = omega^(n / 2^(s+1))
        uint64_t w_m[4], w_m_inv[4];
        fr_pow(omega, (uint64_t)(n >> (s + 1)), w_m);
        fr_pow(omega_inv, (uint64_t)(n >> (s + 1)), w_m_inv);

        uint64_t w[4], wi[4];
        memcpy(w, BN254_FR_ONE, 32);
        memcpy(wi, BN254_FR_ONE, 32);

        for (int j = 0; j < halfBlock; j++) {
            memcpy(&fwd[(offset + j) * 4], w, 32);
            memcpy(&inv[(offset + j) * 4], wi, 32);
            mont_mul_4limb(w, w_m, BN254_FR_P, BN254_FR_INV, w);
            mont_mul_4limb(wi, w_m_inv, BN254_FR_P, BN254_FR_INV, wi);
        }
    }

    fr_cached[logN].fwd = fwd;
    fr_cached[logN].inv = inv;
    fr_cached[logN].logN = logN;
    pthread_mutex_unlock(&fr_cache_lock);
}

// Bit-reversal for 256-bit (4×64-bit) elements
static void bit_reverse_permute256(uint64_t *data, int logN) {
    int n = 1 << logN;
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) {
            // Swap 4 limbs
            uint64_t t0 = data[i*4+0]; data[i*4+0] = data[j*4+0]; data[j*4+0] = t0;
            uint64_t t1 = data[i*4+1]; data[i*4+1] = data[j*4+1]; data[j*4+1] = t1;
            uint64_t t2 = data[i*4+2]; data[i*4+2] = data[j*4+2]; data[j*4+2] = t2;
            uint64_t t3 = data[i*4+3]; data[i*4+3] = data[j*4+3]; data[j*4+3] = t3;
        }
    }
}

// Forward NTT for BN254 Fr
// data: array of n elements, each 4×64-bit limbs in Montgomery form
// Single NTT butterfly block (DIT): process one block at base offset
static inline void ntt_dit_block(uint64_t *data, const uint64_t *tw,
                                  int base, int halfBlock, int twOffset)
{
    // j==0: twiddle==1, skip Montgomery mul
    {
        uint64_t *u = &data[base * 4];
        uint64_t *vp = &data[(base + halfBlock) * 4];
        uint64_t sum[4], diff[4];
        mont_add_4limb(u, vp, BN254_FR_P, sum);
        mont_sub_4limb(u, vp, BN254_FR_P, diff);
        memcpy(u, sum, 32);
        memcpy(vp, diff, 32);
    }
    for (int j = 1; j < halfBlock; j++) {
        uint64_t *u = &data[(base + j) * 4];
        uint64_t *vp = &data[(base + j + halfBlock) * 4];
        const uint64_t *twj = &tw[(twOffset + j) * 4];
        uint64_t v[4];
        mont_mul_4limb(twj, vp, BN254_FR_P, BN254_FR_INV, v);
        uint64_t sum[4], diff[4];
        mont_add_4limb(u, v, BN254_FR_P, sum);
        mont_sub_4limb(u, v, BN254_FR_P, diff);
        memcpy(u, sum, 32);
        memcpy(vp, diff, 32);
    }
}

#define NTT_PARALLEL_THRESHOLD 4096

void bn254_fr_ntt(uint64_t *data, int logN) {
    if (logN <= 0) return;
    int n = 1 << logN;

    fr_ensure_twiddles(logN);
    const uint64_t *tw = fr_cached[logN].fwd;

    bit_reverse_permute256(data, logN);

    for (int s = 0; s < logN; s++) {
        int halfBlock = 1 << s;
        int blockSize = halfBlock << 1;
        int nBlocks = n / blockSize;
        int twOffset = halfBlock - 1;

        if (n >= NTT_PARALLEL_THRESHOLD && nBlocks >= 4) {
            // Parallel: dispatch blocks across threads
            int nb = nBlocks, bs = blockSize, hb = halfBlock, two = twOffset;
            uint64_t *d = data;
            const uint64_t *t = tw;
            dispatch_apply(nb, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                ^(size_t idx) {
                    int base = (int)idx * bs;
                    ntt_dit_block(d, t, base, hb, two);
                });
        } else {
            for (int bk = 0; bk < nBlocks; bk++) {
                int base = bk * blockSize;
                ntt_dit_block(data, tw, base, halfBlock, twOffset);
            }
        }
    }
}

// Single INTT butterfly block (DIF): process one block at base offset
static inline void intt_dif_block(uint64_t *data, const uint64_t *tw,
                                   int base, int halfBlock, int twOffset)
{
    // j==0: twiddle==1, skip Montgomery mul
    {
        uint64_t *ap = &data[base * 4];
        uint64_t *bp = &data[(base + halfBlock) * 4];
        uint64_t a[4], b[4];
        memcpy(a, ap, 32); memcpy(b, bp, 32);
        uint64_t sum[4], diff[4];
        mont_add_4limb(a, b, BN254_FR_P, sum);
        mont_sub_4limb(a, b, BN254_FR_P, diff);
        memcpy(ap, sum, 32); memcpy(bp, diff, 32);
    }
    for (int j = 1; j < halfBlock; j++) {
        uint64_t *ap = &data[(base + j) * 4];
        uint64_t *bp = &data[(base + j + halfBlock) * 4];
        const uint64_t *twj = &tw[(twOffset + j) * 4];
        uint64_t a[4], b[4];
        memcpy(a, ap, 32); memcpy(b, bp, 32);
        uint64_t sum[4], diff[4], prod[4];
        mont_add_4limb(a, b, BN254_FR_P, sum);
        mont_sub_4limb(a, b, BN254_FR_P, diff);
        mont_mul_4limb(diff, twj, BN254_FR_P, BN254_FR_INV, prod);
        memcpy(ap, sum, 32); memcpy(bp, prod, 32);
    }
}

// Inverse NTT for BN254 Fr (DIF + bit-reverse + scale)
void bn254_fr_intt(uint64_t *data, int logN) {
    if (logN <= 0) return;
    int n = 1 << logN;

    fr_ensure_twiddles(logN);
    const uint64_t *tw = fr_cached[logN].inv;

    // DIF stages (top-down)
    for (int si = 0; si < logN; si++) {
        int s = logN - 1 - si;
        int halfBlock = 1 << s;
        int blockSize = halfBlock << 1;
        int nBlocks = n / blockSize;
        int twOffset = halfBlock - 1;

        if (n >= NTT_PARALLEL_THRESHOLD && nBlocks >= 4) {
            int nb = nBlocks, bs = blockSize, hb = halfBlock, two = twOffset;
            uint64_t *d = data;
            const uint64_t *t = tw;
            dispatch_apply(nb, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                ^(size_t idx) {
                    int base = (int)idx * bs;
                    intt_dif_block(d, t, base, hb, two);
                });
        } else {
            for (int bk = 0; bk < nBlocks; bk++) {
                int base = bk * blockSize;
                intt_dif_block(data, tw, base, halfBlock, twOffset);
            }
        }
    }

    bit_reverse_permute256(data, logN);

    // Scale by 1/n using batch_mul_scalar for parallel execution
    uint64_t n_mont[4];
    uint64_t n_plain[4] = {(uint64_t)n, 0, 0, 0};
    static const uint64_t BN254_FR_R2[4] = {
        0x1bb8e645ae216da7ULL, 0x53fe3ab1e35c59e3ULL,
        0x8c49833d53bb8085ULL, 0x0216d0b17f4e44a5ULL
    };
    mont_mul_4limb(n_plain, BN254_FR_R2, BN254_FR_P, BN254_FR_INV, n_mont);

    uint64_t n_inv[4];
    fr_inv(n_mont, n_inv);

    bn254_fr_batch_mul_scalar(data, n_inv, data, n);
}

// ============================================================
// Exported C wrapper for benchmarking (matches mont_mul_asm signature)
// ============================================================
void mont_mul_c(uint64_t *result, const uint64_t *a, const uint64_t *b) {
    mont_mul_4limb(a, b, BN254_FR_P, BN254_FR_INV, result);
}

// C batch multiply: data[i] *= multiplier (for fair comparison with ASM batch)
void mont_mul_batch_c(uint64_t *data, const uint64_t *multiplier, int n) {
    for (int i = 0; i < n; i++) {
        mont_mul_4limb(data + i * 4, multiplier, BN254_FR_P, BN254_FR_INV, data + i * 4);
    }
}

// C pair batch multiply: result[i] = a[i] * b[i]
void mont_mul_pair_batch_c(uint64_t *result, const uint64_t *a, const uint64_t *b, int n) {
    for (int i = 0; i < n; i++) {
        mont_mul_4limb(a + i * 4, b + i * 4, BN254_FR_P, BN254_FR_INV, result + i * 4);
    }
}

// Test function to verify assembly correctness from C
int mont_mul_asm_test(void) {
    uint64_t one[4] = {
        0xac96341c4ffffffbULL, 0x36fc76959f60cd29ULL,
        0x666ea36f7879462eULL, 0x0e0a77c19a07df2fULL
    };
    uint64_t asm_result[4] = {0};
    uint64_t c_result[4] = {0};

    // extern declared in header
    mont_mul_asm(asm_result, one, one);
    mont_mul_c(c_result, one, one);

    for (int i = 0; i < 4; i++) {
        if (asm_result[i] != c_result[i]) return -1;
    }
    return 0;
}

// ============================================================
// GKR-specific accelerated operations
// ============================================================

// Fr element is 4 x uint64_t = 32 bytes
#define FR_LIMBS 4
#define FR_BYTES 32

static inline int fr_is_zero(const uint64_t a[4]) {
    return (a[0] | a[1] | a[2] | a[3]) == 0;
}

static inline void fr_copy(uint64_t dst[4], const uint64_t src[4]) {
    dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = src[3];
}

static inline void fr_zero(uint64_t dst[4]) {
    dst[0] = 0; dst[1] = 0; dst[2] = 0; dst[3] = 0;
}

/// Compute eq polynomial evaluations: eq[i] for i in {0,1}^n given point r.
/// eq[0] = 1, then for each variable r_i, split: eq[2j+1] = eq[j]*r_i, eq[2j] = eq[j]*(1-r_i).
/// point: n Fr elements (4 uint64 each). eq: output 2^n Fr elements.
void gkr_eq_poly(const uint64_t *point, int n, uint64_t *eq) {
    static const uint64_t ZERO[4] = {0,0,0,0};
    int size = 1 << n;
    // Initialize eq[0] = 1 (Montgomery)
    memset(eq, 0, (size_t)size * FR_BYTES);
    memcpy(eq, BN254_FR_ONE, FR_BYTES);

    for (int i = 0; i < n; i++) {
        int half = 1 << i;
        const uint64_t *ri = point + i * FR_LIMBS;
        uint64_t oneMinusRi[4];
        mont_sub_4limb(BN254_FR_ONE, ri, BN254_FR_P, oneMinusRi);

        // Process in reverse to avoid overwriting
        for (int j = half - 1; j >= 0; j--) {
            uint64_t *ej = eq + j * FR_LIMBS;
            uint64_t *e2j1 = eq + (2*j+1) * FR_LIMBS;
            uint64_t *e2j = eq + (2*j) * FR_LIMBS;
            uint64_t tmp[4];
            mont_mul_4limb(ej, ri, BN254_FR_P, BN254_FR_INV, e2j1);
            mont_mul_4limb(ej, oneMinusRi, BN254_FR_P, BN254_FR_INV, tmp);
            fr_copy(e2j, tmp);
        }
    }
}

/// In-place MLE fold: v[j] = (1-challenge)*v[j] + challenge*v[j+half] for j in 0..<half.
/// v: array of at least 2*half Fr elements. half: number of output elements.
void gkr_mle_fold(uint64_t *v, int half, const uint64_t challenge[4]) {
    uint64_t oneMinusC[4];
    mont_sub_4limb(BN254_FR_ONE, challenge, BN254_FR_P, oneMinusC);

    for (int j = 0; j < half; j++) {
        uint64_t *vj = v + j * FR_LIMBS;
        const uint64_t *vjh = v + (j + half) * FR_LIMBS;
        uint64_t t1[4], t2[4];
        mont_mul_4limb(oneMinusC, vj, BN254_FR_P, BN254_FR_INV, t1);
        mont_mul_4limb(challenge, vjh, BN254_FR_P, BN254_FR_INV, t2);
        mont_add_4limb(t1, t2, BN254_FR_P, vj);
    }
}

/// Sparse wiring entry for GKR sumcheck: (index, addCoeff[4], mulCoeff[4])
/// Layout: [idx as uint64_t, addCoeff[4], mulCoeff[4]] = 9 uint64_t per entry
#define WENTRY_STRIDE 9
#define WENTRY_IDX(base, i) ((base)[(i)*WENTRY_STRIDE])
#define WENTRY_ADD(base, i) ((base) + (i)*WENTRY_STRIDE + 1)
#define WENTRY_MUL(base, i) ((base) + (i)*WENTRY_STRIDE + 5)

/// Build sparse wiring from gate list with eq polynomial evaluations.
/// Gates: array of (gateType, leftInput, rightInput) as 3 x int32 per gate.
/// eqVals: eq polynomial evaluations (numGates Fr elements).
/// weight: Fr element to scale eq values.
/// wiring: output sparse wiring dict as flat array.
/// Returns number of nonzero entries written to wiring.
/// wiringDict: working buffer of 9 uint64_t per possible index.
///
/// This builds the sparse wiring for one (rPoint, weight) pair and merges into
/// the provided accumulator.
void gkr_accumulate_wiring(
    const int32_t *gates,   // numGates * 3: [type, left, right] per gate
    int numGates,
    const uint64_t *eqVals, // numGates Fr elements
    const uint64_t weight[4],
    int inSize,
    // Accumulator: hashtable stored as flat array indexed by xyIdx
    // Each slot: [valid, addCoeff[4], mulCoeff[4]] = 9 uint64_t
    uint64_t *accum,
    int accumCapacity,
    int32_t *nonzeroIndices,  // track which indices are populated
    int *numNonzero)
{
    static const uint64_t ZERO[4] = {0,0,0,0};

    for (int g = 0; g < numGates; g++) {
        const uint64_t *eqVal = eqVals + g * FR_LIMBS;
        if (fr_is_zero(eqVal)) continue;

        uint64_t coeff[4];
        mont_mul_4limb(weight, eqVal, BN254_FR_P, BN254_FR_INV, coeff);

        int gateType = gates[g * 3];
        int left = gates[g * 3 + 1];
        int right = gates[g * 3 + 2];
        int xyIdx = left * inSize + right;

        uint64_t *slot = accum + xyIdx * WENTRY_STRIDE;
        if (slot[0] == 0) {
            // New entry
            slot[0] = 1;
            memset(slot + 1, 0, 4 * FR_BYTES);  // clear add+mul
            nonzeroIndices[*numNonzero] = (int32_t)xyIdx;
            (*numNonzero)++;
        }

        if (gateType == 0) { // add
            uint64_t tmp[4];
            mont_add_4limb(slot + 1, coeff, BN254_FR_P, tmp);
            memcpy(slot + 1, tmp, FR_BYTES);
        } else { // mul
            uint64_t tmp[4];
            mont_add_4limb(slot + 5, coeff, BN254_FR_P, tmp);
            memcpy(slot + 5, tmp, FR_BYTES);
        }
    }
}

// Comparison function for qsort of int32_t
static int cmp_int32(const void *a, const void *b) {
    int32_t va = *(const int32_t *)a;
    int32_t vb = *(const int32_t *)b;
    return (va > vb) - (va < vb);
}

/// GKR sumcheck X-phase round evaluation.
/// Processes sorted wiring entries and computes s0, s1, s2.
/// wiring: flat array of entries (WENTRY_STRIDE uint64_t each, sorted by idx).
/// numEntries: number of wiring entries.
/// vx: MLE x-evaluations (vxSize elements).
/// vy: MLE y-evaluations (vySize elements).
/// nIn: log2 of input layer size.
/// halfSize: currentTableSize / 2.
/// s0, s1, s2: output accumulators (initialized to zero by caller).
void gkr_sumcheck_round_x(
    const uint64_t *wiring, int numEntries,
    const uint64_t *vx, int vxSize,
    const uint64_t *vy, int vySize,
    int nIn, int halfSize,
    uint64_t s0[4], uint64_t s1[4], uint64_t s2[4])
{
    static const uint64_t ZERO[4] = {0,0,0,0};
    int yMask = vySize - 1;
    int vxHalf = vxSize / 2;

    // Find split position (first entry with idx >= halfSize)
    int splitPos = 0;
    {
        int lo = 0, hi = numEntries;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if ((int64_t)WENTRY_IDX(wiring, mid) < halfSize) lo = mid + 1;
            else hi = mid;
        }
        splitPos = lo;
    }

    int li = 0, hi_ptr = splitPos;

    while (li < splitPos || hi_ptr < numEntries) {
        int64_t lowIdx = li < splitPos ? (int64_t)WENTRY_IDX(wiring, li) : INT64_MAX;
        int64_t highIdx = hi_ptr < numEntries ? (int64_t)WENTRY_IDX(wiring, hi_ptr) - halfSize : INT64_MAX;

        const uint64_t *a0, *m0, *a1, *m1;
        int mergedIdx;

        if (lowIdx <= highIdx) {
            mergedIdx = (int)lowIdx;
            a0 = WENTRY_ADD(wiring, li);
            m0 = WENTRY_MUL(wiring, li);
            li++;
            if (lowIdx == highIdx) {
                a1 = WENTRY_ADD(wiring, hi_ptr);
                m1 = WENTRY_MUL(wiring, hi_ptr);
                hi_ptr++;
            } else {
                a1 = ZERO; m1 = ZERO;
            }
        } else {
            mergedIdx = (int)highIdx;
            a0 = ZERO; m0 = ZERO;
            a1 = WENTRY_ADD(wiring, hi_ptr);
            m1 = WENTRY_MUL(wiring, hi_ptr);
            hi_ptr++;
        }

        int yIdx = mergedIdx & yMask;
        int xIdx = mergedIdx >> nIn;
        const uint64_t *vx0 = (xIdx < vxHalf) ? vx + xIdx * FR_LIMBS : ZERO;
        const uint64_t *vx1 = (xIdx + vxHalf < vxSize) ? vx + (xIdx + vxHalf) * FR_LIMBS : ZERO;
        const uint64_t *vyVal = (yIdx < vySize) ? vy + yIdx * FR_LIMBS : ZERO;

        // Contribution per entry pair:
        //   g(t) = (a_t + m_t*vy)*vx_t + a_t*vy
        // where a_t, m_t are interpolated add/mul coefficients at t
        // Optimize: skip multiplications for zero coefficients
        uint64_t c0[4], c1[4], a0vy[4], a1vy[4];
        int a0z = fr_is_zero(a0), m0z = fr_is_zero(m0);
        int a1z = fr_is_zero(a1), m1z = fr_is_zero(m1);

        // c0 = a0 + m0*vy
        if (m0z) { fr_copy(c0, a0); }
        else if (a0z) { mont_mul_4limb(m0, vyVal, BN254_FR_P, BN254_FR_INV, c0); }
        else { uint64_t t[4]; mont_mul_4limb(m0, vyVal, BN254_FR_P, BN254_FR_INV, t); mont_add_4limb(a0, t, BN254_FR_P, c0); }

        // c1 = a1 + m1*vy
        if (m1z) { fr_copy(c1, a1); }
        else if (a1z) { mont_mul_4limb(m1, vyVal, BN254_FR_P, BN254_FR_INV, c1); }
        else { uint64_t t[4]; mont_mul_4limb(m1, vyVal, BN254_FR_P, BN254_FR_INV, t); mont_add_4limb(a1, t, BN254_FR_P, c1); }

        // a0vy = a0*vy, a1vy = a1*vy
        if (a0z) { fr_zero(a0vy); } else { mont_mul_4limb(a0, vyVal, BN254_FR_P, BN254_FR_INV, a0vy); }
        if (a1z) { fr_zero(a1vy); } else { mont_mul_4limb(a1, vyVal, BN254_FR_P, BN254_FR_INV, a1vy); }

        // g0 = c0*vx0 + a0vy
        uint64_t g0[4], g1[4];
        if (fr_is_zero(c0) && a0z) { /* g0 = 0, skip */ }
        else {
            mont_mul_4limb(c0, vx0, BN254_FR_P, BN254_FR_INV, g0);
            if (!a0z) { mont_add_4limb(g0, a0vy, BN254_FR_P, g0); }
            mont_add_4limb(s0, g0, BN254_FR_P, s0);
        }

        // g1 = c1*vx1 + a1vy
        if (fr_is_zero(c1) && a1z) { /* g1 = 0, skip */ }
        else {
            mont_mul_4limb(c1, vx1, BN254_FR_P, BN254_FR_INV, g1);
            if (!a1z) { mont_add_4limb(g1, a1vy, BN254_FR_P, g1); }
            mont_add_4limb(s1, g1, BN254_FR_P, s1);
        }

        // g2: c2=2c1-c0, a2vy=2*a1vy-a0vy, vx2=2vx1-vx0
        uint64_t c2[4], a2vy[4], vx2[4], g2[4];
        mont_add_4limb(c1, c1, BN254_FR_P, c2);
        mont_sub_4limb(c2, c0, BN254_FR_P, c2);
        mont_add_4limb(a1vy, a1vy, BN254_FR_P, a2vy);
        mont_sub_4limb(a2vy, a0vy, BN254_FR_P, a2vy);
        mont_add_4limb(vx1, vx1, BN254_FR_P, vx2);
        mont_sub_4limb(vx2, vx0, BN254_FR_P, vx2);
        mont_mul_4limb(c2, vx2, BN254_FR_P, BN254_FR_INV, g2);
        mont_add_4limb(g2, a2vy, BN254_FR_P, g2);
        mont_add_4limb(s2, g2, BN254_FR_P, s2);
    }
}

/// GKR sumcheck Y-phase round evaluation.
void gkr_sumcheck_round_y(
    const uint64_t *wiring, int numEntries,
    const uint64_t vxScalar[4],
    const uint64_t *vy, int vySize,
    int halfSize,
    uint64_t s0[4], uint64_t s1[4], uint64_t s2[4])
{
    static const uint64_t ZERO[4] = {0,0,0,0};
    int vyHalf = vySize / 2;

    int splitPos = 0;
    {
        int lo = 0, hi2 = numEntries;
        while (lo < hi2) {
            int mid = (lo + hi2) / 2;
            if ((int64_t)WENTRY_IDX(wiring, mid) < halfSize) lo = mid + 1;
            else hi2 = mid;
        }
        splitPos = lo;
    }

    int li = 0, hi_ptr = splitPos;

    while (li < splitPos || hi_ptr < numEntries) {
        int64_t lowIdx = li < splitPos ? (int64_t)WENTRY_IDX(wiring, li) : INT64_MAX;
        int64_t highIdx = hi_ptr < numEntries ? (int64_t)WENTRY_IDX(wiring, hi_ptr) - halfSize : INT64_MAX;

        const uint64_t *a0, *m0, *a1, *m1;
        int mergedIdx;

        if (lowIdx <= highIdx) {
            mergedIdx = (int)lowIdx;
            a0 = WENTRY_ADD(wiring, li);
            m0 = WENTRY_MUL(wiring, li);
            li++;
            if (lowIdx == highIdx) {
                a1 = WENTRY_ADD(wiring, hi_ptr);
                m1 = WENTRY_MUL(wiring, hi_ptr);
                hi_ptr++;
            } else {
                a1 = ZERO; m1 = ZERO;
            }
        } else {
            mergedIdx = (int)highIdx;
            a0 = ZERO; m0 = ZERO;
            a1 = WENTRY_ADD(wiring, hi_ptr);
            m1 = WENTRY_MUL(wiring, hi_ptr);
            hi_ptr++;
        }

        const uint64_t *vy0 = (mergedIdx < vyHalf) ? vy + mergedIdx * FR_LIMBS : ZERO;
        const uint64_t *vy1 = (mergedIdx + vyHalf < vySize) ? vy + (mergedIdx + vyHalf) * FR_LIMBS : ZERO;

        // Optimize: skip multiplications for zero coefficients
        uint64_t c0[4], c1[4], a0vx[4], a1vx[4];
        int a0z = fr_is_zero(a0), m0z = fr_is_zero(m0);
        int a1z = fr_is_zero(a1), m1z = fr_is_zero(m1);

        if (m0z) { fr_copy(c0, a0); }
        else if (a0z) { mont_mul_4limb(m0, vxScalar, BN254_FR_P, BN254_FR_INV, c0); }
        else { uint64_t t[4]; mont_mul_4limb(m0, vxScalar, BN254_FR_P, BN254_FR_INV, t); mont_add_4limb(a0, t, BN254_FR_P, c0); }

        if (m1z) { fr_copy(c1, a1); }
        else if (a1z) { mont_mul_4limb(m1, vxScalar, BN254_FR_P, BN254_FR_INV, c1); }
        else { uint64_t t[4]; mont_mul_4limb(m1, vxScalar, BN254_FR_P, BN254_FR_INV, t); mont_add_4limb(a1, t, BN254_FR_P, c1); }

        if (a0z) { fr_zero(a0vx); } else { mont_mul_4limb(a0, vxScalar, BN254_FR_P, BN254_FR_INV, a0vx); }
        if (a1z) { fr_zero(a1vx); } else { mont_mul_4limb(a1, vxScalar, BN254_FR_P, BN254_FR_INV, a1vx); }

        uint64_t g0[4], g1[4];
        if (fr_is_zero(c0) && a0z) { /* skip */ }
        else {
            mont_mul_4limb(c0, vy0, BN254_FR_P, BN254_FR_INV, g0);
            if (!a0z) { mont_add_4limb(g0, a0vx, BN254_FR_P, g0); }
            mont_add_4limb(s0, g0, BN254_FR_P, s0);
        }

        if (fr_is_zero(c1) && a1z) { /* skip */ }
        else {
            mont_mul_4limb(c1, vy1, BN254_FR_P, BN254_FR_INV, g1);
            if (!a1z) { mont_add_4limb(g1, a1vx, BN254_FR_P, g1); }
            mont_add_4limb(s1, g1, BN254_FR_P, s1);
        }

        uint64_t c2[4], a2vx[4], vy2[4], g2[4];
        mont_add_4limb(c1, c1, BN254_FR_P, c2);
        mont_sub_4limb(c2, c0, BN254_FR_P, c2);
        mont_add_4limb(a1vx, a1vx, BN254_FR_P, a2vx);
        mont_sub_4limb(a2vx, a0vx, BN254_FR_P, a2vx);
        mont_add_4limb(vy1, vy1, BN254_FR_P, vy2);
        mont_sub_4limb(vy2, vy0, BN254_FR_P, vy2);
        mont_mul_4limb(c2, vy2, BN254_FR_P, BN254_FR_INV, g2);
        mont_add_4limb(g2, a2vx, BN254_FR_P, g2);
        mont_add_4limb(s2, g2, BN254_FR_P, s2);
    }
}

/// Reduce sparse wiring after a sumcheck round.
/// Merges low/high halves using challenge value.
/// Input wiring: numEntries entries (WENTRY_STRIDE each, sorted by idx).
/// Output wiring: written to outWiring, returns number of output entries.
/// halfSize: the split threshold.
int gkr_wiring_reduce(
    const uint64_t *wiring, int numEntries,
    const uint64_t challenge[4],
    int halfSize,
    uint64_t *outWiring)
{
    uint64_t oneMinusC[4];
    mont_sub_4limb(BN254_FR_ONE, challenge, BN254_FR_P, oneMinusC);

    int splitPos = 0;
    {
        int lo = 0, hi2 = numEntries;
        while (lo < hi2) {
            int mid = (lo + hi2) / 2;
            if ((int64_t)WENTRY_IDX(wiring, mid) < halfSize) lo = mid + 1;
            else hi2 = mid;
        }
        splitPos = lo;
    }

    int li = 0, hi_ptr = splitPos;
    int outCount = 0;

    while (li < splitPos || hi_ptr < numEntries) {
        int64_t lowIdx = li < splitPos ? (int64_t)WENTRY_IDX(wiring, li) : INT64_MAX;
        int64_t highIdx = hi_ptr < numEntries ? (int64_t)WENTRY_IDX(wiring, hi_ptr) - halfSize : INT64_MAX;

        uint64_t *out = outWiring + outCount * WENTRY_STRIDE;

        if (lowIdx < highIdx) {
            out[0] = (uint64_t)lowIdx;
            const uint64_t *aCoeff = WENTRY_ADD(wiring, li);
            const uint64_t *mCoeff = WENTRY_MUL(wiring, li);
            // Skip zero coefficients to save mont_mul
            if (fr_is_zero(aCoeff)) { fr_zero(out + 1); }
            else { mont_mul_4limb(oneMinusC, aCoeff, BN254_FR_P, BN254_FR_INV, out + 1); }
            if (fr_is_zero(mCoeff)) { fr_zero(out + 5); }
            else { mont_mul_4limb(oneMinusC, mCoeff, BN254_FR_P, BN254_FR_INV, out + 5); }
            li++;
        } else if (highIdx < lowIdx) {
            out[0] = (uint64_t)highIdx;
            const uint64_t *aCoeff = WENTRY_ADD(wiring, hi_ptr);
            const uint64_t *mCoeff = WENTRY_MUL(wiring, hi_ptr);
            if (fr_is_zero(aCoeff)) { fr_zero(out + 1); }
            else { mont_mul_4limb(challenge, aCoeff, BN254_FR_P, BN254_FR_INV, out + 1); }
            if (fr_is_zero(mCoeff)) { fr_zero(out + 5); }
            else { mont_mul_4limb(challenge, mCoeff, BN254_FR_P, BN254_FR_INV, out + 5); }
            hi_ptr++;
        } else {
            out[0] = (uint64_t)lowIdx;
            const uint64_t *aLo = WENTRY_ADD(wiring, li);
            const uint64_t *aHi = WENTRY_ADD(wiring, hi_ptr);
            const uint64_t *mLo = WENTRY_MUL(wiring, li);
            const uint64_t *mHi = WENTRY_MUL(wiring, hi_ptr);
            uint64_t t1[4], t2[4];
            // Add coefficients
            int aLoZ = fr_is_zero(aLo), aHiZ = fr_is_zero(aHi);
            if (aLoZ && aHiZ) {
                fr_zero(out + 1);
            } else if (aLoZ) {
                mont_mul_4limb(challenge, aHi, BN254_FR_P, BN254_FR_INV, out + 1);
            } else if (aHiZ) {
                mont_mul_4limb(oneMinusC, aLo, BN254_FR_P, BN254_FR_INV, out + 1);
            } else {
                mont_mul_4limb(oneMinusC, aLo, BN254_FR_P, BN254_FR_INV, t1);
                mont_mul_4limb(challenge, aHi, BN254_FR_P, BN254_FR_INV, t2);
                mont_add_4limb(t1, t2, BN254_FR_P, out + 1);
            }
            // Mul coefficients
            int mLoZ = fr_is_zero(mLo), mHiZ = fr_is_zero(mHi);
            if (mLoZ && mHiZ) {
                fr_zero(out + 5);
            } else if (mLoZ) {
                mont_mul_4limb(challenge, mHi, BN254_FR_P, BN254_FR_INV, out + 5);
            } else if (mHiZ) {
                mont_mul_4limb(oneMinusC, mLo, BN254_FR_P, BN254_FR_INV, out + 5);
            } else {
                mont_mul_4limb(oneMinusC, mLo, BN254_FR_P, BN254_FR_INV, t1);
                mont_mul_4limb(challenge, mHi, BN254_FR_P, BN254_FR_INV, t2);
                mont_add_4limb(t1, t2, BN254_FR_P, out + 5);
            }
            li++;
            hi_ptr++;
        }
        outCount++;
    }
    return outCount;
}

/// Complete GKR sumcheck for one layer.
/// This fuses wiring construction, all rounds, and MLE reduction into one C call.
///
/// gates: numGates * 3 int32_t [type, left, right].
/// rPoints: numRPoints * (point[nOut * 4 uint64] + weight[4 uint64]).
/// prevEvals: 2^nIn Fr elements (MLE of previous layer).
/// nOut, nIn: log2 sizes.
/// challenges_out: output 2*nIn Fr elements (random challenges, filled by caller after each round).
/// msgs_out: output 2*nIn * 3 Fr elements (s0, s1, s2 per round).
/// curVx_out, curVy_out: output folded MLE values.
///
/// Returns through pointers. The transcript interaction must still happen in Swift
/// between rounds, so we expose a per-round stepping interface instead.

/// Single-round step for GKR sumcheck.
/// Call this for each of the 2*nIn rounds.
/// wiring/numEntries: current sparse wiring (WENTRY_STRIDE layout).
/// curVx/curVy: current MLE folds.
/// round: 0..2*nIn-1.
/// nIn: log2 input size.
/// currentTableSize: starts at 2^(2*nIn), halves each round.
/// s0,s1,s2: output round polynomial values.
void gkr_sumcheck_step(
    const uint64_t *wiring, int numEntries,
    const uint64_t *curVx, int vxSize,
    const uint64_t *curVy, int vySize,
    int round, int nIn, int currentTableSize,
    uint64_t s0[4], uint64_t s1[4], uint64_t s2[4])
{
    int halfSize = currentTableSize / 2;
    int isXPhase = (round < nIn);

    fr_zero(s0);
    fr_zero(s1);
    fr_zero(s2);

    if (isXPhase) {
        gkr_sumcheck_round_x(wiring, numEntries, curVx, vxSize, curVy, vySize,
                              nIn, halfSize, s0, s1, s2);
    } else {
        // Y-phase: vx is a single scalar
        const uint64_t *vxScalar = (vxSize > 0) ? curVx : (const uint64_t[]){0,0,0,0};
        gkr_sumcheck_round_y(wiring, numEntries, vxScalar, curVy, vySize,
                              halfSize, s0, s1, s2);
    }
}
