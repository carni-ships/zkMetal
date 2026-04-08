// Batch field operations for BN254 Fr using optimized C with branchless logic.
//
// The 256-bit carry chain limits NEON vectorization for individual additions,
// so the main wins come from:
// - Loop unrolling for ILP across independent elements
// - Branchless conditional subtract/add using masks
// - Prefetch hints for large batch sizes
// - __uint128_t for clean carry propagation

#include "NeonFieldOps.h"
#include <stdlib.h>
#include <string.h>
#include <dispatch/dispatch.h>

typedef unsigned __int128 uint128_t;

#define BATCH_THREAD_THRESHOLD 4096
#define MAX_THREADS 8

static const uint64_t FR_P[4] = {
    0x43e1f593f0000001ULL, 0x2833e84879b97091ULL,
    0xb85045b68181585dULL, 0x30644e72e131a029ULL
};

// Branchless modular add: result = (a + b) mod p
static inline void fr_add_branchless(
    const uint64_t a[4], const uint64_t b[4], uint64_t result[4])
{
    // Step 1: a + b with carry
    uint128_t w;
    uint64_t c = 0;
    w = (uint128_t)a[0] + b[0];
    uint64_t s0 = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[1] + b[1] + c;
    uint64_t s1 = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[2] + b[2] + c;
    uint64_t s2 = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[3] + b[3] + c;
    uint64_t s3 = (uint64_t)w; c = (uint64_t)(w >> 64);

    // Step 2: s - p with borrow
    uint64_t borrow = 0;
    uint128_t d;
    d = (uint128_t)s0 - FR_P[0];
    uint64_t r0 = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)s1 - FR_P[1] - borrow;
    uint64_t r1 = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)s2 - FR_P[2] - borrow;
    uint64_t r2 = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)s3 - FR_P[3] - borrow;
    uint64_t r3 = (uint64_t)d; borrow = (d >> 127) & 1;

    // Branchless select: if carry from add OR no borrow from sub, use reduced
    // mask = 0xFFFFFFFFFFFFFFFF if we should use reduced (r), 0 otherwise
    uint64_t use_reduced = c | (borrow ^ 1);  // 1 if should subtract
    uint64_t mask = (uint64_t)(-(int64_t)use_reduced);  // 0 or 0xFFFF...

    result[0] = (r0 & mask) | (s0 & ~mask);
    result[1] = (r1 & mask) | (s1 & ~mask);
    result[2] = (r2 & mask) | (s2 & ~mask);
    result[3] = (r3 & mask) | (s3 & ~mask);
}

// Branchless modular sub: result = (a - b) mod p
static inline void fr_sub_branchless(
    const uint64_t a[4], const uint64_t b[4], uint64_t result[4])
{
    // Step 1: a - b with borrow
    uint128_t d;
    uint64_t borrow = 0;
    d = (uint128_t)a[0] - b[0];
    uint64_t s0 = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)a[1] - b[1] - borrow;
    uint64_t s1 = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)a[2] - b[2] - borrow;
    uint64_t s2 = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)a[3] - b[3] - borrow;
    uint64_t s3 = (uint64_t)d; borrow = (d >> 127) & 1;

    // Step 2: s + p (for correction)
    uint64_t c = 0;
    uint128_t w;
    w = (uint128_t)s0 + FR_P[0];
    uint64_t r0 = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)s1 + FR_P[1] + c;
    uint64_t r1 = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)s2 + FR_P[2] + c;
    uint64_t r2 = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)s3 + FR_P[3] + c;
    uint64_t r3 = (uint64_t)w;

    // Branchless select: if borrow, use corrected (r); else use s
    uint64_t mask = (uint64_t)(-(int64_t)borrow);  // 0 or 0xFFFF...

    result[0] = (r0 & mask) | (s0 & ~mask);
    result[1] = (r1 & mask) | (s1 & ~mask);
    result[2] = (r2 & mask) | (s2 & ~mask);
    result[3] = (r3 & mask) | (s3 & ~mask);
}

// Branchless negate: result = p - a (or 0 if a == 0)
static inline void fr_neg_branchless(
    const uint64_t a[4], uint64_t result[4])
{
    // p - a
    uint128_t d;
    uint64_t borrow = 0;
    d = (uint128_t)FR_P[0] - a[0];
    uint64_t r0 = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)FR_P[1] - a[1] - borrow;
    uint64_t r1 = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)FR_P[2] - a[2] - borrow;
    uint64_t r2 = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)FR_P[3] - a[3] - borrow;
    uint64_t r3 = (uint64_t)d;

    // If a == 0, result should be 0 (neg of zero is zero)
    uint64_t nonzero = a[0] | a[1] | a[2] | a[3];
    uint64_t mask = (uint64_t)(-(int64_t)(nonzero != 0));

    result[0] = r0 & mask;
    result[1] = r1 & mask;
    result[2] = r2 & mask;
    result[3] = r3 & mask;
}

// ============================================================
// Montgomery multiply (reuse for batch_mul_scalar)
// ============================================================
static const uint64_t FR_INV = 0xc2e1f593efffffffULL;

static inline void fr_mont_mul(
    const uint64_t a[4], const uint64_t b[4], uint64_t result[4])
{
    uint64_t t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0;

    // Iteration i=0
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
        uint64_t m = t0 * FR_INV;
        w = (uint128_t)m * FR_P[0] + t0; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FR_P[1] + t1 + c; t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FR_P[2] + t2 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FR_P[3] + t3 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t3 = t4 + c; t4 = 0;
    }
    // Iteration i=1
    {
        uint128_t w; uint64_t c;
        w = (uint128_t)a[1] * b[0] + t0; t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[1] * b[1] + t1 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[1] * b[2] + t2 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[1] * b[3] + t3 + c; t3 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t4 += c;
        uint64_t m = t0 * FR_INV;
        w = (uint128_t)m * FR_P[0] + t0; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FR_P[1] + t1 + c; t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FR_P[2] + t2 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FR_P[3] + t3 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t3 = t4 + c; t4 = 0;
    }
    // Iteration i=2
    {
        uint128_t w; uint64_t c;
        w = (uint128_t)a[2] * b[0] + t0; t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[2] * b[1] + t1 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[2] * b[2] + t2 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[2] * b[3] + t3 + c; t3 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t4 += c;
        uint64_t m = t0 * FR_INV;
        w = (uint128_t)m * FR_P[0] + t0; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FR_P[1] + t1 + c; t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FR_P[2] + t2 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FR_P[3] + t3 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t3 = t4 + c; t4 = 0;
    }
    // Iteration i=3
    {
        uint128_t w; uint64_t c;
        w = (uint128_t)a[3] * b[0] + t0; t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[3] * b[1] + t1 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[3] * b[2] + t2 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)a[3] * b[3] + t3 + c; t3 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t4 += c;
        uint64_t m = t0 * FR_INV;
        w = (uint128_t)m * FR_P[0] + t0; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FR_P[1] + t1 + c; t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FR_P[2] + t2 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * FR_P[3] + t3 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
        t3 = t4 + c; t4 = 0;
    }

    // Final conditional subtraction (branchless)
    uint64_t borrow = 0;
    uint128_t d;
    d = (uint128_t)t0 - FR_P[0]; uint64_t r0 = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)t1 - FR_P[1] - borrow; uint64_t r1 = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)t2 - FR_P[2] - borrow; uint64_t r2 = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)t3 - FR_P[3] - borrow; uint64_t r3 = (uint64_t)d; borrow = (d >> 127) & 1;

    uint64_t mask = (uint64_t)(-(int64_t)(borrow ^ 1));
    result[0] = (r0 & mask) | (t0 & ~mask);
    result[1] = (r1 & mask) | (t1 & ~mask);
    result[2] = (r2 & mask) | (t2 & ~mask);
    result[3] = (r3 & mask) | (t3 & ~mask);
}

// ============================================================
// Batch operations — 2x unrolled with prefetch
// ============================================================

void bn254_fr_batch_add_neon(uint64_t *result, const uint64_t *a,
                              const uint64_t *b, int n)
{
    int i = 0;
    // 2x unrolled loop
    for (; i + 1 < n; i += 2) {
        // Prefetch next pair
        if (i + 3 < n) {
            __builtin_prefetch(&a[(i + 2) * 4], 0, 1);
            __builtin_prefetch(&b[(i + 2) * 4], 0, 1);
            __builtin_prefetch(&a[(i + 3) * 4], 0, 1);
            __builtin_prefetch(&b[(i + 3) * 4], 0, 1);
        }
        fr_add_branchless(&a[i * 4], &b[i * 4], &result[i * 4]);
        fr_add_branchless(&a[(i + 1) * 4], &b[(i + 1) * 4], &result[(i + 1) * 4]);
    }
    // Handle remainder
    if (i < n) {
        fr_add_branchless(&a[i * 4], &b[i * 4], &result[i * 4]);
    }
}

void bn254_fr_batch_sub_neon(uint64_t *result, const uint64_t *a,
                              const uint64_t *b, int n)
{
    int i = 0;
    for (; i + 1 < n; i += 2) {
        if (i + 3 < n) {
            __builtin_prefetch(&a[(i + 2) * 4], 0, 1);
            __builtin_prefetch(&b[(i + 2) * 4], 0, 1);
            __builtin_prefetch(&a[(i + 3) * 4], 0, 1);
            __builtin_prefetch(&b[(i + 3) * 4], 0, 1);
        }
        fr_sub_branchless(&a[i * 4], &b[i * 4], &result[i * 4]);
        fr_sub_branchless(&a[(i + 1) * 4], &b[(i + 1) * 4], &result[(i + 1) * 4]);
    }
    if (i < n) {
        fr_sub_branchless(&a[i * 4], &b[i * 4], &result[i * 4]);
    }
}

void bn254_fr_batch_neg_neon(uint64_t *result, const uint64_t *a, int n)
{
    int i = 0;
    for (; i + 1 < n; i += 2) {
        if (i + 3 < n) {
            __builtin_prefetch(&a[(i + 2) * 4], 0, 1);
            __builtin_prefetch(&a[(i + 3) * 4], 0, 1);
        }
        fr_neg_branchless(&a[i * 4], &result[i * 4]);
        fr_neg_branchless(&a[(i + 1) * 4], &result[(i + 1) * 4]);
    }
    if (i < n) {
        fr_neg_branchless(&a[i * 4], &result[i * 4]);
    }
}

void bn254_fr_batch_mul_scalar_neon(uint64_t *result, const uint64_t *a,
                                     const uint64_t *scalar, int n)
{
    int i = 0;
    for (; i + 1 < n; i += 2) {
        if (i + 3 < n) {
            __builtin_prefetch(&a[(i + 2) * 4], 0, 1);
            __builtin_prefetch(&a[(i + 3) * 4], 0, 1);
        }
        fr_mont_mul(&a[i * 4], scalar, &result[i * 4]);
        fr_mont_mul(&a[(i + 1) * 4], scalar, &result[(i + 1) * 4]);
    }
    if (i < n) {
        fr_mont_mul(&a[i * 4], scalar, &result[i * 4]);
    }
}

/// Batch scalar-plus-vector: result[i] = scalar + a[i] for i in 0..n-1
void bn254_fr_batch_add_scalar_neon(uint64_t *result, const uint64_t *scalar,
                                     const uint64_t *a, int n)
{
    int i = 0;
    for (; i + 1 < n; i += 2) {
        if (i + 3 < n) {
            __builtin_prefetch(&a[(i + 2) * 4], 0, 1);
            __builtin_prefetch(&a[(i + 3) * 4], 0, 1);
        }
        fr_add_branchless(scalar, &a[i * 4], &result[i * 4]);
        fr_add_branchless(scalar, &a[(i + 1) * 4], &result[(i + 1) * 4]);
    }
    if (i < n) {
        fr_add_branchless(scalar, &a[i * 4], &result[i * 4]);
    }
}

/// Batch scalar-minus-vector: result[i] = scalar - a[i] for i in 0..n-1
void bn254_fr_batch_scalar_sub_neon(uint64_t *result, const uint64_t *scalar,
                                     const uint64_t *a, int n)
{
    for (int i = 0; i < n; i++) {
        if (i + 2 < n) __builtin_prefetch(&a[(i + 2) * 4], 0, 1);
        fr_sub_branchless(scalar, &a[i * 4], &result[i * 4]);
    }
}

/// Batch element-wise multiply: result[i] = a[i] * b[i] for i in 0..n-1
void bn254_fr_batch_mul_neon(uint64_t *result, const uint64_t *a,
                              const uint64_t *b, int n)
{
    int i = 0;
    for (; i + 1 < n; i += 2) {
        if (i + 3 < n) {
            __builtin_prefetch(&a[(i + 2) * 4], 0, 1);
            __builtin_prefetch(&b[(i + 2) * 4], 0, 1);
        }
        fr_mont_mul(&a[i * 4], &b[i * 4], &result[i * 4]);
        fr_mont_mul(&a[(i + 1) * 4], &b[(i + 1) * 4], &result[(i + 1) * 4]);
    }
    if (i < n) {
        fr_mont_mul(&a[i * 4], &b[i * 4], &result[i * 4]);
    }
}

/// Fused scalar-multiply-accumulate: result[i] += scalar * a[i] for i in 0..n-1
void bn254_fr_batch_mac_neon(uint64_t *result, const uint64_t *a,
                              const uint64_t *scalar, int n)
{
    uint64_t tmp[4];
    for (int i = 0; i < n; i++) {
        if (i + 2 < n) __builtin_prefetch(&a[(i + 2) * 4], 0, 1);
        fr_mont_mul(&a[i * 4], scalar, tmp);
        fr_add_branchless(tmp, &result[i * 4], &result[i * 4]);
    }
}

/// Sumcheck reduce: result[i] = a[i] + challenge * (b[i] - a[i]) for i in 0..halfN-1
/// where a = evals[0..halfN-1], b = evals[halfN..n-1]
void bn254_fr_sumcheck_reduce(const uint64_t *evals, const uint64_t *challenge,
                               uint64_t *result, int halfN)
{
    uint64_t diff[4], scaled[4];
    for (int i = 0; i < halfN; i++) {
        if (i + 2 < halfN) {
            __builtin_prefetch(&evals[(i + 2) * 4], 0, 1);
            __builtin_prefetch(&evals[(halfN + i + 2) * 4], 0, 1);
        }
        const uint64_t *ai = &evals[i * 4];
        const uint64_t *bi = &evals[(halfN + i) * 4];
        fr_sub_branchless(bi, ai, diff);
        fr_mont_mul(challenge, diff, scaled);
        fr_add_branchless(ai, scaled, &result[i * 4]);
    }
}

/// IPA fold: result[i] = v[i] + challenge * v[halfN+i]
void bn254_fr_ipa_fold(const uint64_t *v, const uint64_t *challenge,
                        uint64_t *result, int halfN)
{
    uint64_t scaled[4];
    for (int i = 0; i < halfN; i++) {
        if (i + 2 < halfN) {
            __builtin_prefetch(&v[(i + 2) * 4], 0, 1);
            __builtin_prefetch(&v[(halfN + i + 2) * 4], 0, 1);
        }
        const uint64_t *lo = &v[i * 4];
        const uint64_t *hi = &v[(halfN + i) * 4];
        fr_mont_mul(challenge, hi, scaled);
        fr_add_branchless(lo, scaled, &result[i * 4]);
    }
}

/// FRI fold: result[i] = (a[i]+b[i]) + challenge*(a[i]-b[i])*invTwiddles[i]
void bn254_fr_fri_fold(const uint64_t *evals, const uint64_t *challenge,
                        const uint64_t *invTwiddles, uint64_t *result, int half)
{
    uint64_t sum[4], diff[4], prod[4], term[4];
    for (int i = 0; i < half; i++) {
        if (i + 2 < half) {
            __builtin_prefetch(&evals[(i + 2) * 4], 0, 1);
            __builtin_prefetch(&evals[(half + i + 2) * 4], 0, 1);
            __builtin_prefetch(&invTwiddles[(i + 2) * 4], 0, 1);
        }
        const uint64_t *a = &evals[i * 4];
        const uint64_t *b = &evals[(half + i) * 4];
        fr_add_branchless(a, b, sum);
        fr_sub_branchless(a, b, diff);
        fr_mont_mul(diff, &invTwiddles[i * 4], prod);
        fr_mont_mul(challenge, prod, term);
        fr_add_branchless(sum, term, &result[i * 4]);
    }
}

/// Batch FMA: result[i] = result[i] * scalar + other[i]
void bn254_fr_batch_fma_scalar(uint64_t *result, const uint64_t *scalar,
                                const uint64_t *other, int n)
{
    if (n >= BATCH_THREAD_THRESHOLD) {
        int nChunks = MAX_THREADS;
        if (n < nChunks * 1024) nChunks = 4;
        int chunkSize = n / nChunks;
        int total = n;
        uint64_t *sr = result;
        const uint64_t *ss = scalar, *so = other;
        dispatch_apply(nChunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
            ^(size_t idx) {
                int start = (int)idx * chunkSize;
                int end = ((int)idx == nChunks - 1) ? total : start + chunkSize;
                uint64_t tmp[4];
                for (int i = start; i < end; i++) {
                    if (i + 2 < end) {
                        __builtin_prefetch(&sr[(i + 2) * 4], 1, 1);
                        __builtin_prefetch(&so[(i + 2) * 4], 0, 1);
                    }
                    fr_mont_mul(&sr[i * 4], ss, tmp);
                    fr_add_branchless(tmp, &so[i * 4], &sr[i * 4]);
                }
            });
        return;
    }
    uint64_t tmp[4];
    for (int i = 0; i < n; i++) {
        if (i + 2 < n) {
            __builtin_prefetch(&result[(i + 2) * 4], 1, 1);
            __builtin_prefetch(&other[(i + 2) * 4], 0, 1);
        }
        fr_mont_mul(&result[i * 4], scalar, tmp);
        fr_add_branchless(tmp, &other[i * 4], &result[i * 4]);
    }
}

/// Batch subtract scalar from array: result[i] = a[i] - scalar for i in 0..n-1
void bn254_fr_batch_sub_scalar(uint64_t *result, const uint64_t *a,
                                const uint64_t *scalar, int n)
{
    for (int i = 0; i < n; i++) {
        if (i + 2 < n) __builtin_prefetch(&a[(i + 2) * 4], 0, 1);
        fr_sub_branchless(&a[i * 4], scalar, &result[i * 4]);
    }
}

/// AXPY: result[i] += scalar * x[i] for i in 0..n-1
void bn254_fr_batch_axpy(uint64_t *result, const uint64_t *scalar,
                          const uint64_t *x, int n)
{
    uint64_t tmp[4];
    for (int i = 0; i < n; i++) {
        if (i + 2 < n) {
            __builtin_prefetch(&result[(i + 2) * 4], 1, 1);
            __builtin_prefetch(&x[(i + 2) * 4], 0, 1);
        }
        fr_mont_mul(scalar, &x[i * 4], tmp);
        fr_add_branchless(&result[i * 4], tmp, &result[i * 4]);
    }
}

/// Interleaved fold: result[i] = evals[2i] + challenge*(evals[2i+1]-evals[2i])
static void fold_interleaved_range(const uint64_t *evals, const uint64_t *challenge,
                                    uint64_t *result, int start, int end)
{
    uint64_t diff[4], scaled[4];
    for (int i = start; i < end; i++) {
        if (i + 2 < end) {
            __builtin_prefetch(&evals[(2 * (i + 2)) * 4], 0, 1);
        }
        const uint64_t *lo = &evals[(2 * i) * 4];
        const uint64_t *hi = &evals[(2 * i + 1) * 4];
        fr_sub_branchless(hi, lo, diff);
        fr_mont_mul(challenge, diff, scaled);
        fr_add_branchless(lo, scaled, &result[i * 4]);
    }
}

void bn254_fr_fold_interleaved(const uint64_t *evals, const uint64_t *challenge,
                                uint64_t *result, int halfN)
{
    if (halfN >= BATCH_THREAD_THRESHOLD) {
        int nChunks = MAX_THREADS;
        if (halfN < nChunks * 1024) nChunks = 4;
        int chunkSize = halfN / nChunks;
        int total = halfN;
        const uint64_t *se = evals, *sc = challenge;
        uint64_t *sr = result;
        dispatch_apply(nChunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                       ^(size_t idx) {
            int s = (int)idx * chunkSize;
            int e = ((int)idx == nChunks - 1) ? total : s + chunkSize;
            fold_interleaved_range(se, sc, sr, s, e);
        });
    } else {
        fold_interleaved_range(evals, challenge, result, 0, halfN);
    }
}

/// ZM fold interleaved: result[i] = evals[2i] + challenge * evals[2i+1]
/// For Zeromorph/Gemini where the formula is lo + c*hi (NOT lo + c*(hi-lo)).
static void fold_zm_interleaved_range(const uint64_t *evals, const uint64_t *challenge,
                                       uint64_t *result, int start, int end)
{
    uint64_t scaled[4];
    for (int i = start; i < end; i++) {
        if (i + 2 < end) {
            __builtin_prefetch(&evals[(2 * (i + 2)) * 4], 0, 1);
        }
        const uint64_t *lo = &evals[(2 * i) * 4];
        const uint64_t *hi = &evals[(2 * i + 1) * 4];
        fr_mont_mul(challenge, hi, scaled);
        fr_add_branchless(lo, scaled, &result[i * 4]);
    }
}

void bn254_fr_fold_zm_interleaved(const uint64_t *evals, const uint64_t *challenge,
                                   uint64_t *result, int halfN)
{
    if (halfN >= BATCH_THREAD_THRESHOLD) {
        int nChunks = MAX_THREADS;
        if (halfN < nChunks * 1024) nChunks = 4;
        int chunkSize = halfN / nChunks;
        int total = halfN;
        const uint64_t *se = evals, *sc = challenge;
        uint64_t *sr = result;
        dispatch_apply(nChunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                       ^(size_t idx) {
            int s = (int)idx * chunkSize;
            int e = ((int)idx == nChunks - 1) ? total : s + chunkSize;
            fold_zm_interleaved_range(se, sc, sr, s, e);
        });
    } else {
        fold_zm_interleaved_range(evals, challenge, result, 0, halfN);
    }
}

/// Fold halves: result[i] = arr[i] + challenge * (arr[i + halfN] - arr[i])
/// For non-interleaved layout where first half = arr[0..halfN-1],
/// second half = arr[halfN..2*halfN-1].
void bn254_fr_fold_halves(const uint64_t *arr, const uint64_t *challenge,
                           uint64_t *result, int halfN)
{
    if (halfN >= BATCH_THREAD_THRESHOLD) {
        int nChunks = MAX_THREADS;
        if (halfN < nChunks * 1024) nChunks = 4;
        int chunkSize = halfN / nChunks;
        int total = halfN;
        const uint64_t *sa = arr, *sc = challenge;
        uint64_t *sr = result;
        dispatch_apply(nChunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
            ^(size_t idx) {
                int start = (int)idx * chunkSize;
                int end = ((int)idx == nChunks - 1) ? total : start + chunkSize;
                uint64_t diff[4], scaled[4];
                for (int i = start; i < end; i++) {
                    if (i + 2 < end) {
                        __builtin_prefetch(&sa[(i + 2) * 4], 0, 1);
                        __builtin_prefetch(&sa[(i + 2 + total) * 4], 0, 1);
                    }
                    fr_sub_branchless(&sa[(i + total) * 4], &sa[i * 4], diff);
                    fr_mont_mul(sc, diff, scaled);
                    fr_add_branchless(&sa[i * 4], scaled, &sr[i * 4]);
                }
            });
        return;
    }
    uint64_t diff[4], scaled[4];
    for (int i = 0; i < halfN; i++) {
        if (i + 2 < halfN) {
            __builtin_prefetch(&arr[(i + 2) * 4], 0, 1);
            __builtin_prefetch(&arr[(i + 2 + halfN) * 4], 0, 1);
        }
        fr_sub_branchless(&arr[(i + halfN) * 4], &arr[i * 4], diff);
        fr_mont_mul(challenge, diff, scaled);
        fr_add_branchless(&arr[i * 4], scaled, &result[i * 4]);
    }
}

// ============================================================================
// In-place fold variants: write results back to the beginning of the input
// buffer, eliminating temporary array allocations. Safe because position i
// only reads from positions >= 2i (interleaved) or i and i+halfN (halves).
// ============================================================================

/// In-place interleaved fold: data[i] = data[2i] + challenge*(data[2i+1]-data[2i])
static void fold_interleaved_inplace_range(uint64_t *data, const uint64_t *challenge,
                                            int start, int end)
{
    uint64_t diff[4], scaled[4];
    for (int i = start; i < end; i++) {
        if (i + 2 < end) {
            __builtin_prefetch(&data[(2 * (i + 2)) * 4], 0, 1);
        }
        const uint64_t *lo = &data[(2 * i) * 4];
        const uint64_t *hi = &data[(2 * i + 1) * 4];
        fr_sub_branchless(hi, lo, diff);
        fr_mont_mul(challenge, diff, scaled);
        fr_add_branchless(lo, scaled, &data[i * 4]);
    }
}

void bn254_fr_fold_interleaved_inplace(uint64_t *data, const uint64_t *challenge, int halfN)
{
    if (halfN >= BATCH_THREAD_THRESHOLD) {
        int nChunks = MAX_THREADS;
        if (halfN < nChunks * 1024) nChunks = 4;
        int chunkSize = halfN / nChunks;
        int total = halfN;
        uint64_t *sd = data;
        const uint64_t *sc = challenge;
        dispatch_apply(nChunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                       ^(size_t idx) {
            int s = (int)idx * chunkSize;
            int e = ((int)idx == nChunks - 1) ? total : s + chunkSize;
            fold_interleaved_inplace_range(sd, sc, s, e);
        });
    } else {
        fold_interleaved_inplace_range(data, challenge, 0, halfN);
    }
}

/// In-place ZM fold: data[i] = data[2i] + challenge * data[2i+1]
static void fold_zm_interleaved_inplace_range(uint64_t *data, const uint64_t *challenge,
                                               int start, int end)
{
    uint64_t scaled[4];
    for (int i = start; i < end; i++) {
        if (i + 2 < end) {
            __builtin_prefetch(&data[(2 * (i + 2)) * 4], 0, 1);
        }
        const uint64_t *lo = &data[(2 * i) * 4];
        const uint64_t *hi = &data[(2 * i + 1) * 4];
        fr_mont_mul(challenge, hi, scaled);
        fr_add_branchless(lo, scaled, &data[i * 4]);
    }
}

void bn254_fr_fold_zm_interleaved_inplace(uint64_t *data, const uint64_t *challenge, int halfN)
{
    if (halfN >= BATCH_THREAD_THRESHOLD) {
        int nChunks = MAX_THREADS;
        if (halfN < nChunks * 1024) nChunks = 4;
        int chunkSize = halfN / nChunks;
        int total = halfN;
        uint64_t *sd = data;
        const uint64_t *sc = challenge;
        dispatch_apply(nChunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                       ^(size_t idx) {
            int s = (int)idx * chunkSize;
            int e = ((int)idx == nChunks - 1) ? total : s + chunkSize;
            fold_zm_interleaved_inplace_range(sd, sc, s, e);
        });
    } else {
        fold_zm_interleaved_inplace_range(data, challenge, 0, halfN);
    }
}

/// In-place fold halves: data[i] = data[i] + challenge * (data[i+halfN] - data[i])
/// Safe because we read data[i] and data[i+halfN] before overwriting data[i],
/// and the second half (data[halfN..]) is never written.
void bn254_fr_fold_halves_inplace(uint64_t *data, const uint64_t *challenge, int halfN)
{
    if (halfN >= BATCH_THREAD_THRESHOLD) {
        int nChunks = MAX_THREADS;
        if (halfN < nChunks * 1024) nChunks = 4;
        int chunkSize = halfN / nChunks;
        int total = halfN;
        uint64_t *sd = data;
        const uint64_t *sc = challenge;
        dispatch_apply(nChunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
            ^(size_t idx) {
                int start = (int)idx * chunkSize;
                int end = ((int)idx == nChunks - 1) ? total : start + chunkSize;
                uint64_t diff[4], scaled[4];
                for (int i = start; i < end; i++) {
                    if (i + 2 < end) {
                        __builtin_prefetch(&sd[(i + 2) * 4], 0, 1);
                        __builtin_prefetch(&sd[(i + 2 + total) * 4], 0, 1);
                    }
                    fr_sub_branchless(&sd[(i + total) * 4], &sd[i * 4], diff);
                    fr_mont_mul(sc, diff, scaled);
                    fr_add_branchless(&sd[i * 4], scaled, &sd[i * 4]);
                }
            });
        return;
    }
    uint64_t diff[4], scaled[4];
    for (int i = 0; i < halfN; i++) {
        if (i + 2 < halfN) {
            __builtin_prefetch(&data[(i + 2) * 4], 0, 1);
            __builtin_prefetch(&data[(i + 2 + halfN) * 4], 0, 1);
        }
        fr_sub_branchless(&data[(i + halfN) * 4], &data[i * 4], diff);
        fr_mont_mul(challenge, diff, scaled);
        fr_add_branchless(&data[i * 4], scaled, &data[i * 4]);
    }
}

/// In-place sumcheck reduce: data[i] = data[i] + challenge * (data[i+halfN] - data[i])
/// Same as fold_halves_inplace (identical formula).
void bn254_fr_sumcheck_reduce_inplace(uint64_t *data, const uint64_t *challenge, int halfN)
{
    bn254_fr_fold_halves_inplace(data, challenge, halfN);
}

/// In-place IPA fold: data[i] = data[i] + challenge * data[i+halfN]
void bn254_fr_ipa_fold_inplace(uint64_t *data, const uint64_t *challenge, int halfN)
{
    if (halfN >= BATCH_THREAD_THRESHOLD) {
        int nChunks = MAX_THREADS;
        if (halfN < nChunks * 1024) nChunks = 4;
        int chunkSize = halfN / nChunks;
        int total = halfN;
        uint64_t *sd = data;
        const uint64_t *sc = challenge;
        dispatch_apply(nChunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
            ^(size_t idx) {
                int start = (int)idx * chunkSize;
                int end = ((int)idx == nChunks - 1) ? total : start + chunkSize;
                uint64_t scaled[4];
                for (int i = start; i < end; i++) {
                    if (i + 2 < end) {
                        __builtin_prefetch(&sd[(i + 2) * 4], 0, 1);
                        __builtin_prefetch(&sd[(i + 2 + total) * 4], 0, 1);
                    }
                    fr_mont_mul(sc, &sd[(i + total) * 4], scaled);
                    fr_add_branchless(&sd[i * 4], scaled, &sd[i * 4]);
                }
            });
        return;
    }
    uint64_t scaled[4];
    for (int i = 0; i < halfN; i++) {
        if (i + 2 < halfN) {
            __builtin_prefetch(&data[(i + 2) * 4], 0, 1);
            __builtin_prefetch(&data[(i + 2 + halfN) * 4], 0, 1);
        }
        fr_mont_mul(challenge, &data[(i + halfN) * 4], scaled);
        fr_add_branchless(&data[i * 4], scaled, &data[i * 4]);
    }
}

/// In-place FRI fold: data[i] = (data[i]+data[i+half]) + challenge*(data[i]-data[i+half])*invTwiddles[i]
void bn254_fr_fri_fold_inplace(uint64_t *data, const uint64_t *challenge,
                                const uint64_t *invTwiddles, int half)
{
    uint64_t sum[4], diff[4], prod[4], term[4];
    for (int i = 0; i < half; i++) {
        if (i + 2 < half) {
            __builtin_prefetch(&data[(i + 2) * 4], 0, 1);
            __builtin_prefetch(&data[(half + i + 2) * 4], 0, 1);
            __builtin_prefetch(&invTwiddles[(i + 2) * 4], 0, 1);
        }
        const uint64_t *a = &data[i * 4];
        const uint64_t *b = &data[(half + i) * 4];
        fr_add_branchless(a, b, sum);
        fr_sub_branchless(a, b, diff);
        fr_mont_mul(diff, &invTwiddles[i * 4], prod);
        fr_mont_mul(challenge, prod, term);
        fr_add_branchless(sum, term, &data[i * 4]);
    }
}

/// Spartan degree-2 sumcheck round:
///   s0 = sum_{i<h} w[i]*z[i]
///   s1 = sum_{i<h} w[i+h]*z[i+h]
///   s2 = sum_{i<h} (2*w[i+h]-w[i])*(2*z[i+h]-z[i])
void bn254_fr_spartan_sumcheck_deg2(const uint64_t *w, const uint64_t *z,
                                     int halfN,
                                     uint64_t *s0, uint64_t *s1, uint64_t *s2)
{
    uint64_t acc0[4] = {0}, acc1[4] = {0}, acc2[4] = {0};
    uint64_t prod[4], w2[4], z2[4], dw[4], dz[4];

    for (int i = 0; i < halfN; i++) {
        if (i + 2 < halfN) {
            __builtin_prefetch(&w[(i + 2) * 4], 0, 1);
            __builtin_prefetch(&w[(halfN + i + 2) * 4], 0, 1);
            __builtin_prefetch(&z[(i + 2) * 4], 0, 1);
            __builtin_prefetch(&z[(halfN + i + 2) * 4], 0, 1);
        }
        const uint64_t *w0 = &w[i * 4];
        const uint64_t *w1 = &w[(halfN + i) * 4];
        const uint64_t *z0 = &z[i * 4];
        const uint64_t *z1 = &z[(halfN + i) * 4];

        // s0 += w0 * z0
        fr_mont_mul(w0, z0, prod);
        fr_add_branchless(acc0, prod, acc0);

        // s1 += w1 * z1
        fr_mont_mul(w1, z1, prod);
        fr_add_branchless(acc1, prod, acc1);

        // s2 += (2*w1 - w0) * (2*z1 - z0)
        fr_add_branchless(w1, w1, dw);
        fr_sub_branchless(dw, w0, w2);
        fr_add_branchless(z1, z1, dz);
        fr_sub_branchless(dz, z0, z2);
        fr_mont_mul(w2, z2, prod);
        fr_add_branchless(acc2, prod, acc2);
    }

    memcpy(s0, acc0, 32);
    memcpy(s1, acc1, 32);
    memcpy(s2, acc2, 32);
}

/// Spartan degree-3 sumcheck round for F(x) = eq(tau,x) * (Az(x)*Bz(x) - Cz(x))
void bn254_fr_spartan_sumcheck_deg3(const uint64_t *eq, const uint64_t *az,
                                     const uint64_t *bz, const uint64_t *cz,
                                     int halfN,
                                     uint64_t *s0, uint64_t *s1,
                                     uint64_t *s2, uint64_t *s3)
{
    uint64_t acc0[4] = {0}, acc1[4] = {0}, acc2[4] = {0}, acc3[4] = {0};
    uint64_t ab[4], f[4], tmp[4];
    uint64_t eq2[4], a2[4], b2[4], c2[4];
    uint64_t eq3[4], a3[4], b3[4], c3[4];
    uint64_t d[4];

    for (int i = 0; i < halfN; i++) {
        if (i + 2 < halfN) {
            __builtin_prefetch(&eq[(i + 2) * 4], 0, 1);
            __builtin_prefetch(&eq[(halfN + i + 2) * 4], 0, 1);
            __builtin_prefetch(&az[(i + 2) * 4], 0, 1);
            __builtin_prefetch(&az[(halfN + i + 2) * 4], 0, 1);
        }
        const uint64_t *e0 = &eq[i * 4];
        const uint64_t *e1 = &eq[(halfN + i) * 4];
        const uint64_t *a0 = &az[i * 4];
        const uint64_t *a1 = &az[(halfN + i) * 4];
        const uint64_t *b0 = &bz[i * 4];
        const uint64_t *b1 = &bz[(halfN + i) * 4];
        const uint64_t *c0 = &cz[i * 4];
        const uint64_t *c1 = &cz[(halfN + i) * 4];

        // t=0: eq0 * (a0*b0 - c0)
        fr_mont_mul(a0, b0, ab);
        fr_sub_branchless(ab, c0, tmp);
        fr_mont_mul(e0, tmp, f);
        fr_add_branchless(acc0, f, acc0);

        // t=1: eq1 * (a1*b1 - c1)
        fr_mont_mul(a1, b1, ab);
        fr_sub_branchless(ab, c1, tmp);
        fr_mont_mul(e1, tmp, f);
        fr_add_branchless(acc1, f, acc1);

        // t=2: linear interpolation at 2
        fr_add_branchless(e1, e1, d); fr_sub_branchless(d, e0, eq2);
        fr_add_branchless(a1, a1, d); fr_sub_branchless(d, a0, a2);
        fr_add_branchless(b1, b1, d); fr_sub_branchless(d, b0, b2);
        fr_add_branchless(c1, c1, d); fr_sub_branchless(d, c0, c2);
        fr_mont_mul(a2, b2, ab);
        fr_sub_branchless(ab, c2, tmp);
        fr_mont_mul(eq2, tmp, f);
        fr_add_branchless(acc2, f, acc2);

        // t=3: linear interpolation at 3
        fr_sub_branchless(e1, e0, d); fr_add_branchless(eq2, d, eq3);
        fr_sub_branchless(a1, a0, d); fr_add_branchless(a2, d, a3);
        fr_sub_branchless(b1, b0, d); fr_add_branchless(b2, d, b3);
        fr_sub_branchless(c1, c0, d); fr_add_branchless(c2, d, c3);
        fr_mont_mul(a3, b3, ab);
        fr_sub_branchless(ab, c3, tmp);
        fr_mont_mul(eq3, tmp, f);
        fr_add_branchless(acc3, f, acc3);
    }

    memcpy(s0, acc0, 32);
    memcpy(s1, acc1, 32);
    memcpy(s2, acc2, 32);
    memcpy(s3, acc3, 32);
}

// ============================================================
// Multi-threaded batch operations for large arrays
// ============================================================

static void batch_parallel(uint64_t *result, const uint64_t *a,
                            const uint64_t *b, const uint64_t *scalar,
                            int n, int op)
{
    if (n < BATCH_THREAD_THRESHOLD) {
        // Single-threaded path
        switch (op) {
        case 0: bn254_fr_batch_add_neon(result, a, b, n); break;
        case 1: bn254_fr_batch_sub_neon(result, a, b, n); break;
        case 2: bn254_fr_batch_neg_neon(result, a, n); break;
        case 3: bn254_fr_batch_mul_scalar_neon(result, a, scalar, n); break;
        case 4: bn254_fr_batch_mul_neon(result, a, b, n); break;
        case 5: bn254_fr_batch_add_scalar_neon(result, scalar, a, n); break;
        }
        return;
    }

    int nChunks = MAX_THREADS;
    if (n < nChunks * 1024) nChunks = 4;
    int chunkSize = n / nChunks;
    int total = n;

    dispatch_apply(nChunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
        ^(size_t idx) {
            int start = (int)idx * chunkSize;
            int end = ((int)idx == nChunks - 1) ? total : start + chunkSize;
            int count = end - start;
            uint64_t *res = result + start * 4;
            const uint64_t *ap = a + start * 4;
            switch (op) {
            case 0: {
                const uint64_t *bp = b + start * 4;
                for (int i = 0; i < count; i++)
                    fr_add_branchless(&ap[i * 4], &bp[i * 4], &res[i * 4]);
                break;
            }
            case 1: {
                const uint64_t *bp = b + start * 4;
                for (int i = 0; i < count; i++)
                    fr_sub_branchless(&ap[i * 4], &bp[i * 4], &res[i * 4]);
                break;
            }
            case 2:
                for (int i = 0; i < count; i++)
                    fr_neg_branchless(&ap[i * 4], &res[i * 4]);
                break;
            case 3:
                for (int i = 0; i < count; i++)
                    fr_mont_mul(&ap[i * 4], scalar, &res[i * 4]);
                break;
            case 4: {
                const uint64_t *bp = b + start * 4;
                for (int i = 0; i < count; i++)
                    fr_mont_mul(&ap[i * 4], &bp[i * 4], &res[i * 4]);
                break;
            }
            case 5:
                for (int i = 0; i < count; i++)
                    fr_add_branchless(scalar, &ap[i * 4], &res[i * 4]);
                break;
            }
        });
}

// Parallel wrappers
void bn254_fr_batch_add_parallel(uint64_t *result, const uint64_t *a,
                                  const uint64_t *b, int n)
{
    batch_parallel(result, a, b, NULL, n, 0);
}

void bn254_fr_batch_sub_parallel(uint64_t *result, const uint64_t *a,
                                  const uint64_t *b, int n)
{
    batch_parallel(result, a, b, NULL, n, 1);
}

void bn254_fr_batch_neg_parallel(uint64_t *result, const uint64_t *a, int n)
{
    batch_parallel(result, a, NULL, NULL, n, 2);
}

void bn254_fr_batch_mul_scalar_parallel(uint64_t *result, const uint64_t *a,
                                         const uint64_t *scalar, int n)
{
    batch_parallel(result, a, NULL, scalar, n, 3);
}

void bn254_fr_batch_mul_parallel(uint64_t *result, const uint64_t *a,
                                  const uint64_t *b, int n)
{
    batch_parallel(result, a, b, NULL, n, 4);
}

void bn254_fr_batch_add_scalar_parallel(uint64_t *result, const uint64_t *a,
                                         const uint64_t *scalar, int n)
{
    batch_parallel(result, a, NULL, scalar, n, 5);
}

// ============================================================
// Vector sum: result = sum(a[i]) for i=0..n-1
// Uses 4-way accumulation to improve ILP, then tree-reduce.
// ============================================================
void bn254_fr_vector_sum(const uint64_t *a, int n, uint64_t result[4])
{
    if (n == 0) {
        result[0] = result[1] = result[2] = result[3] = 0;
        return;
    }
    // 4-way parallel accumulators for instruction-level parallelism
    uint64_t acc0[4] = {0, 0, 0, 0};
    uint64_t acc1[4] = {0, 0, 0, 0};
    uint64_t acc2[4] = {0, 0, 0, 0};
    uint64_t acc3[4] = {0, 0, 0, 0};

    int i = 0;
    for (; i + 3 < n; i += 4) {
        if (i + 7 < n) {
            __builtin_prefetch(&a[(i + 4) * 4], 0, 1);
            __builtin_prefetch(&a[(i + 6) * 4], 0, 1);
        }
        fr_add_branchless(acc0, &a[i * 4], acc0);
        fr_add_branchless(acc1, &a[(i + 1) * 4], acc1);
        fr_add_branchless(acc2, &a[(i + 2) * 4], acc2);
        fr_add_branchless(acc3, &a[(i + 3) * 4], acc3);
    }
    for (; i < n; i++) {
        fr_add_branchless(acc0, &a[i * 4], acc0);
    }
    // Reduce accumulators
    fr_add_branchless(acc0, acc1, acc0);
    fr_add_branchless(acc2, acc3, acc2);
    fr_add_branchless(acc0, acc2, result);
}

// ============================================================
// Batch beta+value: result[i] = beta + values[indices[i]]
// Fuses the gather + field add in one pass for cache efficiency.
// ============================================================
void bn254_fr_batch_beta_add(const uint64_t *beta, const uint64_t *values,
                              const int *indices, int m, uint64_t *result)
{
    int i = 0;
    for (; i + 1 < m; i += 2) {
        if (i + 3 < m) {
            __builtin_prefetch(&values[indices[i + 2] * 4], 0, 1);
            __builtin_prefetch(&values[indices[i + 3] * 4], 0, 1);
        }
        fr_add_branchless(beta, &values[indices[i] * 4], &result[i * 4]);
        fr_add_branchless(beta, &values[indices[i + 1] * 4], &result[(i + 1) * 4]);
    }
    if (i < m) {
        fr_add_branchless(beta, &values[indices[i] * 4], &result[i * 4]);
    }
}

// ============================================================
// Batch range-check decomposition: extracts chunk indices from
// Montgomery-form Fr elements without per-element heap allocation.
// For each lookup[i], reduces from Montgomery form to integer,
// then extracts (numChunks) indices of (bitsPerChunk) bits each.
// Output: indices[k*m + i] = chunk k's index for lookup i.
// ============================================================

// Montgomery reduction: multiply by 1 (raw) to convert from Montgomery form
static inline uint64_t fr_to_uint64(const uint64_t a[4])
{
    // Multiply a * [1, 0, 0, 0] using CIOS Montgomery multiplication
    // Since b = [1,0,0,0], only a[j]*b[0]=a[j]*1 is nonzero
    uint64_t t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0;

    // i=0: multiply by b[0]=1
    t0 = a[0]; t1 = a[1]; t2 = a[2]; t3 = a[3]; t4 = 0;
    uint64_t m = t0 * FR_INV;
    uint128_t w; uint64_t c;
    w = (uint128_t)m * FR_P[0] + t0; c = (uint64_t)(w >> 64);
    w = (uint128_t)m * FR_P[1] + t1 + c; t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)m * FR_P[2] + t2 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)m * FR_P[3] + t3 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
    t3 = t4 + c; t4 = 0;

    // i=1: multiply by b[1]=0, just reduce
    m = t0 * FR_INV;
    w = (uint128_t)m * FR_P[0] + t0; c = (uint64_t)(w >> 64);
    w = (uint128_t)m * FR_P[1] + t1 + c; t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)m * FR_P[2] + t2 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)m * FR_P[3] + t3 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
    t3 = t4 + c; t4 = 0;

    // i=2
    m = t0 * FR_INV;
    w = (uint128_t)m * FR_P[0] + t0; c = (uint64_t)(w >> 64);
    w = (uint128_t)m * FR_P[1] + t1 + c; t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)m * FR_P[2] + t2 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)m * FR_P[3] + t3 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
    t3 = t4 + c; t4 = 0;

    // i=3
    m = t0 * FR_INV;
    w = (uint128_t)m * FR_P[0] + t0; c = (uint64_t)(w >> 64);
    w = (uint128_t)m * FR_P[1] + t1 + c; t0 = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)m * FR_P[2] + t2 + c; t1 = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)m * FR_P[3] + t3 + c; t2 = (uint64_t)w; c = (uint64_t)(w >> 64);
    t3 = t4 + c;

    // Final reduction
    uint128_t d;
    uint64_t borrow = 0;
    d = (uint128_t)t0 - FR_P[0]; uint64_t r0 = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)t1 - FR_P[1] - borrow; borrow = (d >> 127) & 1;
    d = (uint128_t)t2 - FR_P[2] - borrow; borrow = (d >> 127) & 1;
    d = (uint128_t)t3 - FR_P[3] - borrow; borrow = (d >> 127) & 1;
    if (!borrow) return r0;  // reduced value
    return t0;
}

void bn254_fr_batch_decompose(const uint64_t *lookups, int m,
                               int numChunks, int bitsPerChunk,
                               int *indices)
{
    uint64_t chunkMask = ((uint64_t)1 << bitsPerChunk) - 1;
    if (m < 8192) {
        // Small: single-threaded
        for (int i = 0; i < m; i++) {
            if (i + 4 < m) {
                __builtin_prefetch(&lookups[(i + 4) * 4], 0, 1);
            }
            uint64_t v = fr_to_uint64(&lookups[i * 4]);
            for (int k = 0; k < numChunks; k++) {
                indices[k * m + i] = (int)((v >> (k * bitsPerChunk)) & chunkMask);
            }
        }
        return;
    }
    int nChunks = 8;
    if (nChunks > m / 1024) nChunks = m / 1024;
    if (nChunks < 1) nChunks = 1;
    int perChunk = (m + nChunks - 1) / nChunks;
    int total = m;

    dispatch_apply(nChunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
        ^(size_t idx) {
            int start = (int)idx * perChunk;
            int end = start + perChunk;
            if (end > total) end = total;
            for (int i = start; i < end; i++) {
                if (i + 4 < end) {
                    __builtin_prefetch(&lookups[(i + 4) * 4], 0, 1);
                }
                uint64_t v = fr_to_uint64(&lookups[i * 4]);
                for (int k = 0; k < numChunks; k++) {
                    indices[k * total + i] = (int)((v >> (k * bitsPerChunk)) & chunkMask);
                }
            }
        });
}

// ============================================================
// Fused pointwise mul-sub: result[i] = a[i]*b[i] - c[i]
// Used by Groth16 computeH for p(x) = a(x)*b(x) - c(x)
// ============================================================

static void fr_pointwise_mul_sub_range(
    uint64_t *result, const uint64_t *a, const uint64_t *b,
    const uint64_t *c, int start, int end)
{
    for (int i = start; i < end; i++) {
        uint64_t tmp[4];
        fr_mont_mul(&a[i * 4], &b[i * 4], tmp);
        fr_sub_branchless(tmp, &c[i * 4], &result[i * 4]);
    }
}

void bn254_fr_pointwise_mul_sub(const uint64_t *a, const uint64_t *b,
                                 const uint64_t *c, uint64_t *result, int n)
{
    if (n < BATCH_THREAD_THRESHOLD) {
        fr_pointwise_mul_sub_range(result, a, b, c, 0, n);
        return;
    }

    int nChunks = MAX_THREADS;
    if (n < nChunks * 1024) nChunks = 4;
    int chunkSize = n / nChunks;
    int total = n;

    dispatch_apply(nChunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
        ^(size_t idx) {
            int start = (int)idx * chunkSize;
            int end = ((int)idx == nChunks - 1) ? total : start + chunkSize;
            fr_pointwise_mul_sub_range(result, a, b, c, start, end);
        });
}

// ============================================================
// Coefficient division by vanishing polynomial Z_H(x) = x^n - 1
// Given polynomial p of degree 2n-1, computes h = p / Z_H
// where Z_H(x) = x^n - 1, so long division yields:
//   for i from 2n-1 downto n: h[i-n] = rem[i]; rem[i-n] += rem[i]
// ============================================================

void bn254_fr_coeff_div_vanishing(const uint64_t *pCoeffs, int domainN,
                                   uint64_t *hCoeffs)
{
    int bigN = domainN * 2;

    // Copy p into working buffer
    uint64_t *rem = (uint64_t *)malloc(bigN * 4 * sizeof(uint64_t));
    memcpy(rem, pCoeffs, bigN * 4 * sizeof(uint64_t));

    // Zero output
    memset(hCoeffs, 0, domainN * 4 * sizeof(uint64_t));

    // Long division: for i from bigN-1 down to domainN
    for (int i = bigN - 1; i >= domainN; i--) {
        // h[i - domainN] = rem[i]
        memcpy(&hCoeffs[(i - domainN) * 4], &rem[i * 4], 4 * sizeof(uint64_t));
        // rem[i - domainN] += rem[i]
        fr_add_branchless(&rem[(i - domainN) * 4], &rem[i * 4],
                          &rem[(i - domainN) * 4]);
    }

    free(rem);
}

// Linear combine: result[i] = running[i] + rho * new_vals[i]
// Used by HyperNova witness folding.

void bn254_fr_linear_combine(const uint64_t *running, const uint64_t *new_vals,
                              const uint64_t rho[4], uint64_t *result, int count) {
    if (count < 4096) {
        uint64_t tmp[4];
        for (int i = 0; i < count; i++) {
            fr_mont_mul(rho, &new_vals[i * 4], tmp);
            fr_add_branchless(&running[i * 4], tmp, &result[i * 4]);
        }
        return;
    }
    int nChunks = 8;
    if (count < nChunks * 512) nChunks = (count + 511) / 512;
    int chunk = (count + nChunks - 1) / nChunks;
    int total = count;

    dispatch_apply(nChunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
        ^(size_t idx) {
            int start = (int)idx * chunk;
            int end = start + chunk;
            if (end > total) end = total;
            uint64_t tmp[4];
            for (int i = start; i < end; i++) {
                fr_mont_mul(rho, &new_vals[i * 4], tmp);
                fr_add_branchless(&running[i * 4], tmp, &result[i * 4]);
            }
        });
}

// ============================================================
// Pairwise multiply: result[i] = a[2i] * a[2i+1], i=0..half-1
// Used for binary multiplication tree in grand product GKR.
// ============================================================
void bn254_fr_batch_mul_adjacent(uint64_t *result, const uint64_t *a, int half)
{
    int i = 0;
    for (; i + 1 < half; i += 2) {
        if (i + 3 < half) {
            __builtin_prefetch(&a[(2 * (i + 2)) * 4], 0, 1);
            __builtin_prefetch(&a[(2 * (i + 3)) * 4], 0, 1);
        }
        fr_mont_mul(&a[(2 * i) * 4], &a[(2 * i + 1) * 4], &result[i * 4]);
        fr_mont_mul(&a[(2 * (i + 1)) * 4], &a[(2 * (i + 1) + 1) * 4], &result[(i + 1) * 4]);
    }
    if (i < half) {
        fr_mont_mul(&a[(2 * i) * 4], &a[(2 * i + 1) * 4], &result[i * 4]);
    }
}

// ============================================================
// Grand product degree-3 sumcheck round.
// Given eq[0..2*half-1], left[0..2*half-1], right[0..2*half-1],
// computes s0,s1,s2,s3 where:
//   s_t = sum_j eq_t(j) * left_t(j) * right_t(j)
// with linear extrapolation at t=0,1,2,3.
// ============================================================
void bn254_fr_gp_sumcheck_round(
    const uint64_t *eq, const uint64_t *left, const uint64_t *right,
    int half, uint64_t s0[4], uint64_t s1[4], uint64_t s2[4], uint64_t s3[4])
{
    uint64_t acc0[4] = {0,0,0,0};
    uint64_t acc1[4] = {0,0,0,0};
    uint64_t acc2[4] = {0,0,0,0};
    uint64_t acc3[4] = {0,0,0,0};

    // Constants: 2 and 3 in Montgomery form (computed from R mod p = Montgomery(1))
    static const uint64_t ONE_MONT[4] = {0xac96341c4ffffffb, 0x36fc76959f60cd29, 0x666ea36f7879462e, 0x0e0a77c19a07df2f};
    uint64_t TWO[4], THREE[4];
    fr_add_branchless(ONE_MONT, ONE_MONT, TWO);
    fr_add_branchless(TWO, ONE_MONT, THREE);

    for (int j = 0; j < half; j++) {
        const uint64_t *eq0 = &eq[j * 4];
        const uint64_t *l0  = &left[j * 4];
        const uint64_t *r0  = &right[j * 4];
        const uint64_t *eq1 = &eq[(j + half) * 4];
        const uint64_t *l1  = &left[(j + half) * 4];
        const uint64_t *r1  = &right[(j + half) * 4];

        // s0 += eq0 * l0 * r0
        uint64_t tmp[4], prod[4];
        fr_mont_mul(l0, r0, tmp);
        fr_mont_mul(eq0, tmp, prod);
        fr_add_branchless(acc0, prod, acc0);

        // s1 += eq1 * l1 * r1
        fr_mont_mul(l1, r1, tmp);
        fr_mont_mul(eq1, tmp, prod);
        fr_add_branchless(acc1, prod, acc1);

        // t=2: f2 = 2*f1 - f0 for each of eq, left, right
        uint64_t eq2[4], l2[4], r2[4], dbl[4];
        fr_add_branchless(eq1, eq1, dbl); fr_sub_branchless(dbl, eq0, eq2);
        fr_add_branchless(l1, l1, dbl);   fr_sub_branchless(dbl, l0, l2);
        fr_add_branchless(r1, r1, dbl);   fr_sub_branchless(dbl, r0, r2);
        fr_mont_mul(l2, r2, tmp);
        fr_mont_mul(eq2, tmp, prod);
        fr_add_branchless(acc2, prod, acc2);

        // t=3: f3 = 3*f1 - 2*f0
        uint64_t eq3[4], l3[4], r3[4], t3a[4], t3b[4];
        fr_mont_mul(THREE, eq1, t3a); fr_mont_mul(TWO, eq0, t3b); fr_sub_branchless(t3a, t3b, eq3);
        fr_mont_mul(THREE, l1, t3a);  fr_mont_mul(TWO, l0, t3b);  fr_sub_branchless(t3a, t3b, l3);
        fr_mont_mul(THREE, r1, t3a);  fr_mont_mul(TWO, r0, t3b);  fr_sub_branchless(t3a, t3b, r3);
        fr_mont_mul(l3, r3, tmp);
        fr_mont_mul(eq3, tmp, prod);
        fr_add_branchless(acc3, prod, acc3);
    }

    // Final reduce to [0, p)
    for (int k = 0; k < 4; k++) { s0[k] = acc0[k]; s1[k] = acc1[k]; s2[k] = acc2[k]; s3[k] = acc3[k]; }
}

// ============================================================
// Tensor sumcheck degree-2 round polynomial.
// Given interleaved evals[2*i] = lo, evals[2*i+1] = hi for i in 0..half-1,
// computes s0 = sum(lo), s1 = sum(hi), s2 = sum(2*hi - lo).
// ============================================================
void bn254_fr_tensor_sumcheck_round(
    const uint64_t *evals, int half,
    uint64_t s0[4], uint64_t s1[4], uint64_t s2[4])
{
    uint64_t acc0[4] = {0,0,0,0};
    uint64_t acc1[4] = {0,0,0,0};
    uint64_t acc2[4] = {0,0,0,0};

    for (int i = 0; i < half; i++) {
        if (i + 2 < half) {
            __builtin_prefetch(&evals[(2 * (i + 2)) * 4], 0, 1);
            __builtin_prefetch(&evals[(2 * (i + 2) + 1) * 4], 0, 1);
        }
        const uint64_t *lo = &evals[(2 * i) * 4];
        const uint64_t *hi = &evals[(2 * i + 1) * 4];
        fr_add_branchless(acc0, lo, acc0);
        fr_add_branchless(acc1, hi, acc1);
        // at2 = 2*hi - lo
        uint64_t dbl[4], at2[4];
        fr_add_branchless(hi, hi, dbl);
        fr_sub_branchless(dbl, lo, at2);
        fr_add_branchless(acc2, at2, acc2);
    }
    for (int k = 0; k < 4; k++) { s0[k] = acc0[k]; s1[k] = acc1[k]; s2[k] = acc2[k]; }
}

// ============================================================
// Multiply by successive powers: result[i] = a[i] * base^i
// Used for coset shift in NTT: shifted[i] = coeffs[i] * g^i
// ============================================================
void bn254_fr_batch_mul_powers(uint64_t *result, const uint64_t *a,
                               const uint64_t *base, int n)
{
    // current = base^0 = 1 (Montgomery form)
    uint64_t current[4] = {
        0xac96341c4ffffffbULL, 0x36fc76959f60cd29ULL,
        0x666ea36f7879462eULL, 0x0e0a77c19a07df2fULL
    };
    const uint64_t *aPtr = a;
    uint64_t *rPtr = result;
    for (int i = 0; i < n; i++) {
        fr_mont_mul(aPtr, current, rPtr);
        uint64_t next[4];
        fr_mont_mul(current, base, next);
        current[0] = next[0]; current[1] = next[1];
        current[2] = next[2]; current[3] = next[3];
        aPtr += 4;
        rPtr += 4;
    }
}

/// Batch element-wise multiply: result[i] = a[i] * b[i]
void bn254_fr_batch_mul(const uint64_t *a, const uint64_t *b,
                         uint64_t *result, int n)
{
    if (n >= BATCH_THREAD_THRESHOLD) {
        bn254_fr_batch_mul_parallel((uint64_t *)result, a, b, n);
        return;
    }
    for (int i = 0; i < n; i++) {
        if (i + 2 < n) {
            __builtin_prefetch(&a[(i + 2) * 4], 0, 1);
            __builtin_prefetch(&b[(i + 2) * 4], 0, 1);
        }
        fr_mont_mul(&a[i * 4], &b[i * 4], &result[i * 4]);
    }
}

/// Prefix product: result[0] = 1, result[i] = result[i-1] * a[i-1]
void bn254_fr_prefix_product(const uint64_t *a, uint64_t *result, int n)
{
    if (n <= 0) return;
    // result[0] = Montgomery(1)
    static const uint64_t ONE[4] = {
        0xac96341c4ffffffbULL, 0x36fc76959f60cd29ULL,
        0x666ea36f7879462eULL, 0x0e0a77c19a07df2fULL
    };
    memcpy(result, ONE, 32);
    for (int i = 1; i < n; i++) {
        fr_mont_mul(&result[(i - 1) * 4], &a[(i - 1) * 4], &result[i * 4]);
    }
}

/// Batch linear combine: result[i] = a[i] + scalar * b[i]
void bn254_fr_batch_linear_combine(const uint64_t *a, const uint64_t *scalar,
                                    const uint64_t *b, uint64_t *result, int n)
{
    if (n >= BATCH_THREAD_THRESHOLD) {
        int nChunks = MAX_THREADS;
        if (n < nChunks * 1024) nChunks = 4;
        int chunkSize = n / nChunks;
        int total = n;
        const uint64_t *sa = a, *sb = b, *ss = scalar;
        uint64_t *sr = result;
        dispatch_apply(nChunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
            ^(size_t idx) {
                int start = (int)idx * chunkSize;
                int end = ((int)idx == nChunks - 1) ? total : start + chunkSize;
                uint64_t tmp[4];
                for (int i = start; i < end; i++) {
                    if (i + 2 < end) {
                        __builtin_prefetch(&sa[(i + 2) * 4], 0, 1);
                        __builtin_prefetch(&sb[(i + 2) * 4], 0, 1);
                    }
                    fr_mont_mul(ss, &sb[i * 4], tmp);
                    fr_add_branchless(&sa[i * 4], tmp, &sr[i * 4]);
                }
            });
        return;
    }
    uint64_t tmp[4];
    for (int i = 0; i < n; i++) {
        if (i + 2 < n) {
            __builtin_prefetch(&a[(i + 2) * 4], 0, 1);
            __builtin_prefetch(&b[(i + 2) * 4], 0, 1);
        }
        fr_mont_mul(scalar, &b[i * 4], tmp);
        fr_add_branchless(&a[i * 4], tmp, &result[i * 4]);
    }
}

/// Batch sub: result[i] = a[i] - b[i]
void bn254_fr_batch_sub(const uint64_t *a, const uint64_t *b,
                          uint64_t *result, int n)
{
    if (n >= BATCH_THREAD_THRESHOLD) {
        bn254_fr_batch_sub_parallel((uint64_t *)result, a, b, n);
        return;
    }
    for (int i = 0; i < n; i++) {
        if (i + 2 < n) {
            __builtin_prefetch(&a[(i + 2) * 4], 0, 1);
            __builtin_prefetch(&b[(i + 2) * 4], 0, 1);
        }
        fr_sub_branchless(&a[i * 4], &b[i * 4], &result[i * 4]);
    }
}

/// Batch add: result[i] = a[i] + b[i]
void bn254_fr_batch_add(const uint64_t *a, const uint64_t *b,
                          uint64_t *result, int n)
{
    if (n >= BATCH_THREAD_THRESHOLD) {
        bn254_fr_batch_add_parallel((uint64_t *)result, a, b, n);
        return;
    }
    for (int i = 0; i < n; i++) {
        if (i + 2 < n) {
            __builtin_prefetch(&a[(i + 2) * 4], 0, 1);
            __builtin_prefetch(&b[(i + 2) * 4], 0, 1);
        }
        fr_add_branchless(&a[i * 4], &b[i * 4], &result[i * 4]);
    }
}

/// Batch scalar multiply: result[i] = scalar * a[i]
void bn254_fr_batch_mul_scalar(const uint64_t *a, const uint64_t *scalar,
                                uint64_t *result, int n)
{
    if (n >= BATCH_THREAD_THRESHOLD) {
        bn254_fr_batch_mul_scalar_parallel((uint64_t *)result, a, scalar, n);
        return;
    }
    for (int i = 0; i < n; i++) {
        if (i + 2 < n) {
            __builtin_prefetch(&a[(i + 2) * 4], 0, 1);
        }
        fr_mont_mul(&a[i * 4], scalar, &result[i * 4]);
    }
}

/// Zero-safe batch inverse using Montgomery's trick.
/// out[i] = a[i]^{-1} if a[i] != 0, else 0.
void bn254_fr_batch_inverse_safe(const uint64_t *a, int n, uint64_t *out)
{
    if (n == 0) return;

    // Phase 1: prefix products, skipping zeros
    uint64_t running[4] = {0xac96341c4ffffffbULL, 0x36fc76959f60cd29ULL,
                           0x666ea36f7879462eULL, 0x0e0a77c19a07df2fULL};
    // Use out[] as scratch for prefix products
    for (int i = 0; i < n; i++) {
        int is_zero = (a[i*4] == 0 && a[i*4+1] == 0 && a[i*4+2] == 0 && a[i*4+3] == 0);
        if (!is_zero) {
            fr_mont_mul(running, &a[i*4], running);
        }
        memcpy(&out[i*4], running, 32);
    }

    // Phase 2: single inverse of total product
    uint64_t inv[4];
    bn254_fr_inverse(running, inv);

    // Phase 3: backward sweep
    for (int i = n - 1; i >= 0; i--) {
        int is_zero = (a[i*4] == 0 && a[i*4+1] == 0 && a[i*4+2] == 0 && a[i*4+3] == 0);
        if (is_zero) {
            memset(&out[i*4], 0, 32);
        } else {
            uint64_t tmp[4];
            if (i > 0) {
                fr_mont_mul(inv, &out[(i-1)*4], tmp);
            } else {
                memcpy(tmp, inv, 32);
            }
            // update inv = inv * a[i]
            uint64_t new_inv[4];
            fr_mont_mul(inv, &a[i*4], new_inv);
            memcpy(inv, new_inv, 32);
            memcpy(&out[i*4], tmp, 32);
        }
    }
}
