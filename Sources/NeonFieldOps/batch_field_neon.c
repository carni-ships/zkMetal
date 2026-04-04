// Batch field operations for BN254 Fr using optimized C with branchless logic.
//
// The 256-bit carry chain limits NEON vectorization for individual additions,
// so the main wins come from:
// - Loop unrolling for ILP across independent elements
// - Branchless conditional subtract/add using masks
// - Prefetch hints for large batch sizes
// - __uint128_t for clean carry propagation

#include "NeonFieldOps.h"
#include <string.h>
#include <pthread.h>

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

// ============================================================
// Multi-threaded batch operations for large arrays
// ============================================================

#define BATCH_THREAD_THRESHOLD 4096
#define MAX_THREADS 8

typedef struct {
    uint64_t *result;
    const uint64_t *a;
    const uint64_t *b;
    const uint64_t *scalar;
    int start, end;
    int op; // 0=add, 1=sub, 2=neg, 3=mul_scalar
} BatchThreadArg;

static void *batch_thread_func(void *arg) {
    BatchThreadArg *t = (BatchThreadArg *)arg;
    int n = t->end - t->start;
    uint64_t *res = t->result + t->start * 4;
    const uint64_t *ap = t->a + t->start * 4;

    switch (t->op) {
    case 0: {
        const uint64_t *bp = t->b + t->start * 4;
        for (int i = 0; i < n; i++)
            fr_add_branchless(&ap[i * 4], &bp[i * 4], &res[i * 4]);
        break;
    }
    case 1: {
        const uint64_t *bp = t->b + t->start * 4;
        for (int i = 0; i < n; i++)
            fr_sub_branchless(&ap[i * 4], &bp[i * 4], &res[i * 4]);
        break;
    }
    case 2:
        for (int i = 0; i < n; i++)
            fr_neg_branchless(&ap[i * 4], &res[i * 4]);
        break;
    case 3:
        for (int i = 0; i < n; i++)
            fr_mont_mul(&ap[i * 4], t->scalar, &res[i * 4]);
        break;
    }
    return NULL;
}

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

    int nThreads = MAX_THREADS;
    if (n < nThreads * 1024) nThreads = 4;
    int chunkSize = n / nThreads;

    pthread_t threads[MAX_THREADS];
    BatchThreadArg args[MAX_THREADS];

    for (int t = 0; t < nThreads; t++) {
        args[t].result = result;
        args[t].a = a;
        args[t].b = b;
        args[t].scalar = scalar;
        args[t].start = t * chunkSize;
        args[t].end = (t == nThreads - 1) ? n : (t + 1) * chunkSize;
        args[t].op = op;
        pthread_create(&threads[t], NULL, batch_thread_func, &args[t]);
    }
    for (int t = 0; t < nThreads; t++)
        pthread_join(threads[t], NULL);
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
