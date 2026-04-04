// Optimized 256-bit Montgomery multiplication for ARM64
// Fully unrolled CIOS (Coarsely Integrated Operand Scanning) with __uint128_t.
//
// Supports: BN254 Fr/Fp, BLS12-377 Fr, secp256k1 Fp
// Compiles to pipelined ARM64 mul+umulh pairs with adcs carry chains.

#include "NeonFieldOps.h"
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

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

        for (int bk = 0; bk < nBlocks; bk++) {
            int base = bk * blockSize;
            for (int j = 0; j < halfBlock; j++) {
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

        for (int bk = 0; bk < nBlocks; bk++) {
            int base = bk * blockSize;
            for (int j = 0; j < halfBlock; j++) {
                uint64_t *ap = &data[(base + j) * 4];
                uint64_t *bp = &data[(base + j + halfBlock) * 4];
                const uint64_t *twj = &tw[(twOffset + j) * 4];

                uint64_t a[4], b[4];
                memcpy(a, ap, 32);
                memcpy(b, bp, 32);

                uint64_t sum[4], diff[4], prod[4];
                mont_add_4limb(a, b, BN254_FR_P, sum);
                mont_sub_4limb(a, b, BN254_FR_P, diff);
                mont_mul_4limb(diff, twj, BN254_FR_P, BN254_FR_INV, prod);

                memcpy(ap, sum, 32);
                memcpy(bp, prod, 32);
            }
        }
    }

    bit_reverse_permute256(data, logN);

    // Scale by 1/n (in Montgomery form)
    // Compute n_inv = n^{-1} mod p in Montgomery form
    // First, convert n to Montgomery form: n_mont = n * R mod p
    // Then invert: n_inv_mont = n_mont^{-1} in Montgomery form
    // Simpler: compute frFromInt(n) then invert
    uint64_t n_mont[4];
    memcpy(n_mont, BN254_FR_ONE, 32);
    // Multiply ONE by n to get n in Montgomery form
    // ONE is already R mod p, so n * ONE = n * R mod p... no.
    // Actually n_mont = mont_mul(n_plain, R2) where n_plain is just n.
    // But we need n as 4 limbs first.
    uint64_t n_plain[4] = {(uint64_t)n, 0, 0, 0};
    // R^2 mod p for BN254 Fr
    static const uint64_t BN254_FR_R2[4] = {
        0x1bb8e645ae216da7ULL, 0x53fe3ab1e35c59e3ULL,
        0x8c49833d53bb8085ULL, 0x0216d0b17f4e44a5ULL
    };
    mont_mul_4limb(n_plain, BN254_FR_R2, BN254_FR_P, BN254_FR_INV, n_mont);

    uint64_t n_inv[4];
    fr_inv(n_mont, n_inv);

    for (int i = 0; i < n; i++) {
        mont_mul_4limb(&data[i * 4], n_inv, BN254_FR_P, BN254_FR_INV, &data[i * 4]);
    }
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
