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

// ============================================================
// Multi-threaded batch operations for large arrays
// ============================================================

#define BATCH_THREAD_THRESHOLD 4096
#define MAX_THREADS 8

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
