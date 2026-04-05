// WHIR polynomial folding using CIOS Montgomery arithmetic.
// Folds n elements by reductionFactor: result[j] = sum_{k=0}^{r-1} beta^k * evals[j*r + k]
// Uses Horner's method for minimal multiplications.
//
// Multi-threaded for large fold sizes (n >= 4096).

#include "NeonFieldOps.h"
#include <string.h>
#include <pthread.h>

typedef unsigned __int128 uint128_t;

static const uint64_t WHIR_FR_P[4] = {
    0x43e1f593f0000001ULL, 0x2833e84879b97091ULL,
    0xb85045b68181585dULL, 0x30644e72e131a029ULL
};
static const uint64_t WHIR_FR_INV = 0xc2e1f593efffffffULL;

// Inline CIOS Montgomery multiply for BN254 Fr
static inline void whir_mont_mul(
    const uint64_t a[4], const uint64_t b[4], uint64_t result[4])
{
    uint64_t t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0;

    // Unrolled 4 iterations of CIOS
    #define WHIR_CIOS_ITER(I) do { \
        uint128_t w; uint64_t c; \
        w = (uint128_t)a[I] * b[0] + t0; \
        t0 = (uint64_t)w; c = (uint64_t)(w >> 64); \
        w = (uint128_t)a[I] * b[1] + t1 + c; \
        t1 = (uint64_t)w; c = (uint64_t)(w >> 64); \
        w = (uint128_t)a[I] * b[2] + t2 + c; \
        t2 = (uint64_t)w; c = (uint64_t)(w >> 64); \
        w = (uint128_t)a[I] * b[3] + t3 + c; \
        t3 = (uint64_t)w; c = (uint64_t)(w >> 64); \
        t4 += c; \
        uint64_t m = t0 * WHIR_FR_INV; \
        w = (uint128_t)m * WHIR_FR_P[0] + t0; \
        c = (uint64_t)(w >> 64); \
        w = (uint128_t)m * WHIR_FR_P[1] + t1 + c; \
        t0 = (uint64_t)w; c = (uint64_t)(w >> 64); \
        w = (uint128_t)m * WHIR_FR_P[2] + t2 + c; \
        t1 = (uint64_t)w; c = (uint64_t)(w >> 64); \
        w = (uint128_t)m * WHIR_FR_P[3] + t3 + c; \
        t2 = (uint64_t)w; c = (uint64_t)(w >> 64); \
        t3 = t4 + c; t4 = 0; \
    } while(0)

    WHIR_CIOS_ITER(0);
    WHIR_CIOS_ITER(1);
    WHIR_CIOS_ITER(2);
    WHIR_CIOS_ITER(3);

    #undef WHIR_CIOS_ITER

    // Conditional subtraction
    uint64_t borrow = 0;
    uint128_t d;
    uint64_t r0, r1, r2, r3;
    d = (uint128_t)t0 - WHIR_FR_P[0]; r0 = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)t1 - WHIR_FR_P[1] - borrow; r1 = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)t2 - WHIR_FR_P[2] - borrow; r2 = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)t3 - WHIR_FR_P[3] - borrow; r3 = (uint64_t)d; borrow = (d >> 127) & 1;

    if (!borrow) {
        result[0] = r0; result[1] = r1; result[2] = r2; result[3] = r3;
    } else {
        result[0] = t0; result[1] = t1; result[2] = t2; result[3] = t3;
    }
}

// Inline modular add
static inline void whir_mont_add(
    const uint64_t a[4], const uint64_t b[4], uint64_t result[4])
{
    uint128_t w;
    uint64_t c = 0;
    w = (uint128_t)a[0] + b[0]; result[0] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[1] + b[1] + c; result[1] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[2] + b[2] + c; result[2] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[3] + b[3] + c; result[3] = (uint64_t)w; c = (uint64_t)(w >> 64);

    uint64_t borrow = 0;
    uint64_t r0, r1, r2, r3;
    uint128_t d;
    d = (uint128_t)result[0] - WHIR_FR_P[0]; r0 = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)result[1] - WHIR_FR_P[1] - borrow; r1 = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)result[2] - WHIR_FR_P[2] - borrow; r2 = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)result[3] - WHIR_FR_P[3] - borrow; r3 = (uint64_t)d; borrow = (d >> 127) & 1;

    if (c || !borrow) {
        result[0] = r0; result[1] = r1; result[2] = r2; result[3] = r3;
    }
}

// ============================================================
// WHIR fold with reductionFactor = 4 (Horner's method)
// result[j] = evals[4j] + beta*(evals[4j+1] + beta*(evals[4j+2] + beta*evals[4j+3]))
// ============================================================

static void whir_fold_r4_range(
    const uint64_t *evals, const uint64_t beta[4],
    int start, int end, uint64_t *result)
{
    for (int j = start; j < end; j++) {
        const uint64_t *e0 = evals + (j * 4 + 0) * 4;
        const uint64_t *e1 = evals + (j * 4 + 1) * 4;
        const uint64_t *e2 = evals + (j * 4 + 2) * 4;
        const uint64_t *e3 = evals + (j * 4 + 3) * 4;

        // Horner: acc = e3
        // acc = e2 + beta * acc
        // acc = e1 + beta * acc
        // acc = e0 + beta * acc
        uint64_t tmp[4], acc[4];

        // acc = beta * e3 + e2
        whir_mont_mul(beta, e3, tmp);
        whir_mont_add(tmp, e2, acc);

        // acc = beta * acc + e1
        whir_mont_mul(beta, acc, tmp);
        whir_mont_add(tmp, e1, acc);

        // acc = beta * acc + e0
        whir_mont_mul(beta, acc, tmp);
        whir_mont_add(tmp, e0, acc);

        memcpy(result + j * 4, acc, 32);
    }
}

// Generic fold for arbitrary reductionFactor (power of 2)
static void whir_fold_generic_range(
    const uint64_t *evals, const uint64_t beta[4],
    int reductionFactor, int start, int end, uint64_t *result)
{
    for (int j = start; j < end; j++) {
        // Horner: start from the highest index and work down
        uint64_t acc[4];
        memcpy(acc, evals + (j * reductionFactor + reductionFactor - 1) * 4, 32);

        for (int k = reductionFactor - 2; k >= 0; k--) {
            uint64_t tmp[4];
            whir_mont_mul(beta, acc, tmp);
            whir_mont_add(tmp, evals + (j * reductionFactor + k) * 4, acc);
        }
        memcpy(result + j * 4, acc, 32);
    }
}

// ============================================================
// Thread pool for parallel fold
// ============================================================

typedef struct {
    const uint64_t *evals;
    const uint64_t *beta;
    int reductionFactor;
    int start;
    int end;
    uint64_t *result;
} whir_fold_task_t;

static void *whir_fold_thread_r4(void *arg) {
    whir_fold_task_t *t = (whir_fold_task_t *)arg;
    whir_fold_r4_range(t->evals, t->beta, t->start, t->end, t->result);
    return NULL;
}

static void *whir_fold_thread_generic(void *arg) {
    whir_fold_task_t *t = (whir_fold_task_t *)arg;
    whir_fold_generic_range(t->evals, t->beta, t->reductionFactor,
                            t->start, t->end, t->result);
    return NULL;
}

// ============================================================
// Public API
// ============================================================

void bn254_fr_whir_fold(const uint64_t *evals, int n,
                         const uint64_t beta[4],
                         int reductionFactor,
                         uint64_t *result)
{
    int newN = n / reductionFactor;
    if (newN <= 0) return;

    // For small sizes, single-threaded
    int numThreads = 1;
    if (newN >= 2048) {
        numThreads = 4;
    } else if (newN >= 512) {
        numThreads = 2;
    }

    if (numThreads == 1) {
        if (reductionFactor == 4) {
            whir_fold_r4_range(evals, beta, 0, newN, result);
        } else {
            whir_fold_generic_range(evals, beta, reductionFactor, 0, newN, result);
        }
        return;
    }

    // Multi-threaded
    pthread_t threads[8];
    whir_fold_task_t tasks[8];
    int chunk = newN / numThreads;

    for (int i = 0; i < numThreads; i++) {
        tasks[i].evals = evals;
        tasks[i].beta = beta;
        tasks[i].reductionFactor = reductionFactor;
        tasks[i].start = i * chunk;
        tasks[i].end = (i == numThreads - 1) ? newN : (i + 1) * chunk;
        tasks[i].result = result;

        if (reductionFactor == 4) {
            pthread_create(&threads[i], NULL, whir_fold_thread_r4, &tasks[i]);
        } else {
            pthread_create(&threads[i], NULL, whir_fold_thread_generic, &tasks[i]);
        }
    }
    for (int i = 0; i < numThreads; i++) {
        pthread_join(threads[i], NULL);
    }
}
