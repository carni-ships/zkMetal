// Spartan-specific C operations for sumcheck acceleration.
// Uses CIOS Montgomery multiplication for BN254 Fr field arithmetic.
// All arrays use 4 x uint64_t per Fr element (Montgomery form).

#include "NeonFieldOps.h"
#include <string.h>
#include <stdlib.h>
#include <dispatch/dispatch.h>

typedef unsigned __int128 uint128_t;

// BN254 Fr constants
static const uint64_t SP_FR_P[4] = {
    0x43e1f593f0000001ULL, 0x2833e84879b97091ULL,
    0xb85045b68181585dULL, 0x30644e72e131a029ULL
};
static const uint64_t SP_FR_INV = 0xc2e1f593efffffffULL;
static const uint64_t SP_FR_ONE[4] = {
    0xac96341c4ffffffbULL, 0x36fc76959f60cd29ULL,
    0x666ea36f7879462eULL, 0x0e0a77c19a07df2fULL
};

// CIOS Montgomery multiplication
static inline void sp_fr_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t t0=0,t1=0,t2=0,t3=0,t4=0;
    #define SP_ITER(I) { \
        uint128_t w; uint64_t c; \
        w=(uint128_t)a[I]*b[0]+t0; t0=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[1]+t1+c; t1=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[2]+t2+c; t2=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[3]+t3+c; t3=(uint64_t)w; c=(uint64_t)(w>>64); \
        t4+=c; \
        uint64_t m=t0*SP_FR_INV; \
        w=(uint128_t)m*SP_FR_P[0]+t0; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*SP_FR_P[1]+t1+c; t0=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*SP_FR_P[2]+t2+c; t1=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*SP_FR_P[3]+t3+c; t2=(uint64_t)w; c=(uint64_t)(w>>64); \
        t3=t4+c; t4=0; \
    }
    SP_ITER(0) SP_ITER(1) SP_ITER(2) SP_ITER(3)
    #undef SP_ITER
    uint64_t borrow=0; uint64_t r0,r1,r2,r3; uint128_t d;
    d=(uint128_t)t0-SP_FR_P[0]-borrow; r0=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t1-SP_FR_P[1]-borrow; r1=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t2-SP_FR_P[2]-borrow; r2=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t3-SP_FR_P[3]-borrow; r3=(uint64_t)d; borrow=(d>>127)&1;
    if(!borrow){r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;}
    else{r[0]=t0;r[1]=t1;r[2]=t2;r[3]=t3;}
}

// Modular add
static inline void sp_fr_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t w; uint64_t c=0;
    w=(uint128_t)a[0]+b[0]; r[0]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[1]+b[1]+c; r[1]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[2]+b[2]+c; r[2]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[3]+b[3]+c; r[3]=(uint64_t)w; c=(uint64_t)(w>>64);
    uint64_t borrow=0; uint64_t r0,r1,r2,r3; uint128_t d;
    d=(uint128_t)r[0]-SP_FR_P[0]; r0=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[1]-SP_FR_P[1]-borrow; r1=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[2]-SP_FR_P[2]-borrow; r2=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[3]-SP_FR_P[3]-borrow; r3=(uint64_t)d; borrow=(d>>127)&1;
    if(c||!borrow){r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;}
}

// Modular sub
static inline void sp_fr_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t d; uint64_t borrow=0;
    d=(uint128_t)a[0]-b[0]; r[0]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[1]-b[1]-borrow; r[1]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[2]-b[2]-borrow; r[2]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[3]-b[3]-borrow; r[3]=(uint64_t)d; borrow=(d>>127)&1;
    if(borrow){
        uint64_t c=0;
        d=(uint128_t)r[0]+SP_FR_P[0]; r[0]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[1]+SP_FR_P[1]+c; r[1]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[2]+SP_FR_P[2]+c; r[2]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[3]+SP_FR_P[3]+c; r[3]=(uint64_t)d;
    }
}

// ============================================================
// Sparse matrix-vector multiply
// ============================================================

void spartan_sparse_matvec(const uint64_t *entries, int numEntries,
                           const uint64_t *z, int zLen,
                           uint64_t *result, int numRows) {
    memset(result, 0, (size_t)numRows * 32);
    for (int i = 0; i < numEntries; i++) {
        const uint64_t *e = entries + i * 5;
        int row = (int)(e[0] & 0xFFFFFFFF);
        int col = (int)(e[0] >> 32);
        if (row >= numRows || col >= zLen) continue;
        const uint64_t *val = e + 1;
        const uint64_t *zj = z + col * 4;
        uint64_t prod[4];
        sp_fr_mul(val, zj, prod);
        uint64_t sum[4];
        sp_fr_add(result + row * 4, prod, sum);
        memcpy(result + row * 4, sum, 32);
    }
}

// ============================================================
// SC1: degree-3 sumcheck round
// ============================================================

void spartan_sc1_round(uint64_t *eqC, uint64_t *azC, uint64_t *bzC, uint64_t *czC,
                       int halfSize,
                       uint64_t s0[4], uint64_t s1[4], uint64_t s2[4], uint64_t s3[4]) {
    memset(s0, 0, 32);
    memset(s1, 0, 32);
    memset(s2, 0, 32);
    memset(s3, 0, 32);

    for (int j = 0; j < halfSize; j++) {
        const uint64_t *eL = eqC + j * 4;
        const uint64_t *eH = eqC + (j + halfSize) * 4;
        const uint64_t *aL = azC + j * 4;
        const uint64_t *aH = azC + (j + halfSize) * 4;
        const uint64_t *bL = bzC + j * 4;
        const uint64_t *bH = bzC + (j + halfSize) * 4;
        const uint64_t *cL = czC + j * 4;
        const uint64_t *cH = czC + (j + halfSize) * 4;

        uint64_t ab[4], abc[4], term[4], tmp[4];

        // t=0: eL*(aL*bL - cL)
        sp_fr_mul(aL, bL, ab);
        sp_fr_sub(ab, cL, abc);
        sp_fr_mul(eL, abc, term);
        sp_fr_add(s0, term, tmp); memcpy(s0, tmp, 32);

        // t=1: eH*(aH*bH - cH)
        sp_fr_mul(aH, bH, ab);
        sp_fr_sub(ab, cH, abc);
        sp_fr_mul(eH, abc, term);
        sp_fr_add(s1, term, tmp); memcpy(s1, tmp, 32);

        // t=2: v2 = 2*vH - vL
        uint64_t e2[4], a2[4], b2[4], c2[4];
        sp_fr_add(eH, eH, e2); sp_fr_sub(e2, eL, e2);
        sp_fr_add(aH, aH, a2); sp_fr_sub(a2, aL, a2);
        sp_fr_add(bH, bH, b2); sp_fr_sub(b2, bL, b2);
        sp_fr_add(cH, cH, c2); sp_fr_sub(c2, cL, c2);
        sp_fr_mul(a2, b2, ab);
        sp_fr_sub(ab, c2, abc);
        sp_fr_mul(e2, abc, term);
        sp_fr_add(s2, term, tmp); memcpy(s2, tmp, 32);

        // t=3: v3 = 3*vH - 2*vL
        uint64_t e3[4], a3[4], b3[4], c3[4], threeH[4], twoL[4];

        sp_fr_add(eH, eH, threeH); sp_fr_add(threeH, eH, threeH);
        sp_fr_add(eL, eL, twoL); sp_fr_sub(threeH, twoL, e3);

        sp_fr_add(aH, aH, threeH); sp_fr_add(threeH, aH, threeH);
        sp_fr_add(aL, aL, twoL); sp_fr_sub(threeH, twoL, a3);

        sp_fr_add(bH, bH, threeH); sp_fr_add(threeH, bH, threeH);
        sp_fr_add(bL, bL, twoL); sp_fr_sub(threeH, twoL, b3);

        sp_fr_add(cH, cH, threeH); sp_fr_add(threeH, cH, threeH);
        sp_fr_add(cL, cL, twoL); sp_fr_sub(threeH, twoL, c3);

        sp_fr_mul(a3, b3, ab);
        sp_fr_sub(ab, c3, abc);
        sp_fr_mul(e3, abc, term);
        sp_fr_add(s3, term, tmp); memcpy(s3, tmp, 32);
    }
}

// ============================================================
// Fold array in-place
// ============================================================

void spartan_fold_array(uint64_t *arr, int halfSize, const uint64_t ri[4]) {
    for (int j = 0; j < halfSize; j++) {
        uint64_t *lo = arr + j * 4;
        const uint64_t *hi = arr + (j + halfSize) * 4;
        uint64_t diff[4], prod[4], res[4];
        sp_fr_sub(hi, lo, diff);
        sp_fr_mul(ri, diff, prod);
        sp_fr_add(lo, prod, res);
        memcpy(lo, res, 32);
    }
}

// ============================================================
// SC2: degree-2 sumcheck round
// ============================================================

void spartan_sc2_round(uint64_t *wC, uint64_t *zC, int halfSize,
                       uint64_t s0[4], uint64_t s1[4], uint64_t s2[4]) {
    memset(s0, 0, 32);
    memset(s1, 0, 32);
    memset(s2, 0, 32);

    for (int j = 0; j < halfSize; j++) {
        const uint64_t *wL = wC + j * 4;
        const uint64_t *wH = wC + (j + halfSize) * 4;
        const uint64_t *zL = zC + j * 4;
        const uint64_t *zH = zC + (j + halfSize) * 4;

        uint64_t prod[4], tmp[4];
        sp_fr_mul(wL, zL, prod);
        sp_fr_add(s0, prod, tmp); memcpy(s0, tmp, 32);

        sp_fr_mul(wH, zH, prod);
        sp_fr_add(s1, prod, tmp); memcpy(s1, tmp, 32);

        uint64_t w2[4], z2[4];
        sp_fr_add(wH, wH, w2); sp_fr_sub(w2, wL, w2);
        sp_fr_add(zH, zH, z2); sp_fr_sub(z2, zL, z2);
        sp_fr_mul(w2, z2, prod);
        sp_fr_add(s2, prod, tmp); memcpy(s2, tmp, 32);
    }
}

// ============================================================
// Build weight vector for SC2
// ============================================================

void spartan_build_weight_vec(const uint64_t *entries, int numEntries,
                              const uint64_t *eqRx, int eqRxLen,
                              const uint64_t weight[4],
                              uint64_t *wVec, int paddedN) {
    for (int i = 0; i < numEntries; i++) {
        const uint64_t *e = entries + i * 5;
        int row = (int)(e[0] & 0xFFFFFFFF);
        int col = (int)(e[0] >> 32);
        if (row >= eqRxLen || col >= paddedN) continue;
        const uint64_t *val = e + 1;
        uint64_t wval[4], prod[4], sum[4];
        sp_fr_mul(weight, val, wval);
        sp_fr_mul(wval, eqRx + row * 4, prod);
        sp_fr_add(wVec + col * 4, prod, sum);
        memcpy(wVec + col * 4, sum, 32);
    }
}

// ============================================================
// Eq polynomial (matches Swift MultilinearPoly.eqPoly bit ordering)
// ============================================================

void spartan_eq_poly(const uint64_t *point, int n, uint64_t *eq) {
    int size = 1 << n;
    memset(eq, 0, (size_t)size * 32);
    memcpy(eq, SP_FR_ONE, 32);  // eq[0] = 1

    for (int i = 0; i < n; i++) {
        int half = 1 << i;
        const uint64_t *ri = point + i * 4;
        uint64_t oneMinusRi[4];
        sp_fr_sub(SP_FR_ONE, ri, oneMinusRi);
        for (int j = half - 1; j >= 0; j--) {
            uint64_t *src = eq + j * 4;
            uint64_t t0[4], t1[4];
            sp_fr_mul(src, oneMinusRi, t0);
            sp_fr_mul(src, ri, t1);
            memcpy(eq + (2 * j) * 4, t0, 32);
            memcpy(eq + (2 * j + 1) * 4, t1, 32);
        }
    }
}

// ============================================================
// MLE evaluation via successive halving
// ============================================================

void spartan_mle_eval(const uint64_t *evals, int numVars,
                      const uint64_t *point, uint64_t result[4]) {
    int n = 1 << numVars;
    uint64_t *buf = (uint64_t *)malloc((size_t)n * 32);
    memcpy(buf, evals, (size_t)n * 32);

    for (int k = 0; k < numVars; k++) {
        int half = n / 2;
        const uint64_t *pk = point + k * 4;
        for (int j = 0; j < half; j++) {
            uint64_t *lo = buf + j * 4;
            const uint64_t *hi = buf + (j + half) * 4;
            uint64_t diff[4], prod[4], res[4];
            sp_fr_sub(hi, lo, diff);
            sp_fr_mul(pk, diff, prod);
            sp_fr_add(lo, prod, res);
            memcpy(lo, res, 32);
        }
        n = half;
    }
    memcpy(result, buf, 32);
    free(buf);
}

// ============================================================
// Tensor Proof Compression: C-accelerated operations
// ============================================================

// Matrix-vector multiply: result[i] = sum_j M[i*cols+j] * vec[j].
// Multi-threaded for large matrices via GCD dispatch_apply.
void tensor_mat_vec_mul(const uint64_t *M, const uint64_t *vec,
                        int rows, int cols, uint64_t *result) {
    int totalOps = rows * cols;
    if (totalOps < 4096) {
        // Single-threaded for small sizes
        for (int i = 0; i < rows; i++) {
            uint64_t acc[4] = {0,0,0,0};
            const uint64_t *row = M + (size_t)i * cols * 4;
            for (int j = 0; j < cols; j++) {
                uint64_t prod[4];
                sp_fr_mul(row + j * 4, vec + j * 4, prod);
                sp_fr_add(acc, prod, acc);
            }
            memcpy(result + i * 4, acc, 32);
        }
        return;
    }
    int nChunks = 8;
    if (rows < nChunks) nChunks = rows;
    int perChunk = (rows + nChunks - 1) / nChunks;
    int totalRows = rows;

    dispatch_apply(nChunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
        ^(size_t idx) {
            int rowStart = (int)idx * perChunk;
            int rowEnd = rowStart + perChunk;
            if (rowEnd > totalRows) rowEnd = totalRows;
            for (int i = rowStart; i < rowEnd; i++) {
                uint64_t acc[4] = {0,0,0,0};
                const uint64_t *row = M + (size_t)i * cols * 4;
                for (int j = 0; j < cols; j++) {
                    uint64_t prod[4];
                    sp_fr_mul(row + j * 4, vec + j * 4, prod);
                    sp_fr_add(acc, prod, acc);
                }
                memcpy(result + i * 4, acc, 32);
            }
        });
}

// Full inner-product sumcheck: prove sum_i a[i]*b[i] = claimed.
// Pre-derived challenges[numVars] determine each round's folding.
// Output: rounds[numVars * 12] = 3 Fr per round (s0, s1, s2).
//         finalEval[4] = final evaluation a[0]*b[0] after all folds.
void tensor_inner_product_sumcheck(
    const uint64_t *evalsA, const uint64_t *evalsB,
    int numVars, const uint64_t *challenges,
    uint64_t *rounds, uint64_t *finalEval)
{
    int n = 1 << numVars;
    uint64_t *bufA = (uint64_t *)malloc((size_t)n * 32);
    uint64_t *bufB = (uint64_t *)malloc((size_t)n * 32);
    memcpy(bufA, evalsA, (size_t)n * 32);
    memcpy(bufB, evalsB, (size_t)n * 32);

    for (int round = 0; round < numVars; round++) {
        int halfN = n / 2;
        uint64_t *rout = rounds + round * 12;

        uint64_t s0[4] = {0,0,0,0};
        uint64_t s1[4] = {0,0,0,0};
        uint64_t s2[4] = {0,0,0,0};

        for (int i = 0; i < halfN; i++) {
            const uint64_t *aLo = bufA + i * 4;
            const uint64_t *aHi = bufA + (i + halfN) * 4;
            const uint64_t *bLo = bufB + i * 4;
            const uint64_t *bHi = bufB + (i + halfN) * 4;

            // s0 += aLo * bLo
            uint64_t prod[4];
            sp_fr_mul(aLo, bLo, prod);
            sp_fr_add(s0, prod, s0);

            // s1 += aHi * bHi
            sp_fr_mul(aHi, bHi, prod);
            sp_fr_add(s1, prod, s1);

            // s2 += (2*aHi - aLo) * (2*bHi - bLo)
            uint64_t a2[4], b2[4];
            sp_fr_add(aHi, aHi, a2);
            sp_fr_sub(a2, aLo, a2);
            sp_fr_add(bHi, bHi, b2);
            sp_fr_sub(b2, bLo, b2);
            sp_fr_mul(a2, b2, prod);
            sp_fr_add(s2, prod, s2);
        }

        memcpy(rout, s0, 32);
        memcpy(rout + 4, s1, 32);
        memcpy(rout + 8, s2, 32);

        // Fold: buf[i] = buf[i] + challenge * (buf[halfN+i] - buf[i])
        const uint64_t *ch = challenges + round * 4;
        for (int i = 0; i < halfN; i++) {
            uint64_t diffA[4], prodA[4], resA[4];
            sp_fr_sub(bufA + (i + halfN) * 4, bufA + i * 4, diffA);
            sp_fr_mul(ch, diffA, prodA);
            sp_fr_add(bufA + i * 4, prodA, resA);
            memcpy(bufA + i * 4, resA, 32);

            uint64_t diffB[4], prodB[4], resB[4];
            sp_fr_sub(bufB + (i + halfN) * 4, bufB + i * 4, diffB);
            sp_fr_mul(ch, diffB, prodB);
            sp_fr_add(bufB + i * 4, prodB, resB);
            memcpy(bufB + i * 4, resB, 32);
        }
        n = halfN;
    }

    sp_fr_mul(bufA, bufB, finalEval);
    free(bufA);
    free(bufB);
}

// Fused eq polynomial + weighted matrix row evaluation.
// Multi-threaded for large matrices via GCD dispatch_apply.
void tensor_eq_weighted_row(const uint64_t *M, const uint64_t *rowPoint,
                            int rows, int cols, uint64_t *result) {
    uint64_t *eq = (uint64_t *)malloc((size_t)rows * 32);
    int logRows = 0;
    { int tmp = rows; while (tmp > 1) { logRows++; tmp >>= 1; } }
    spartan_eq_poly(rowPoint, logRows, eq);

    int totalOps = rows * cols;
    if (totalOps < 4096) {
        // Single-threaded
        memset(result, 0, (size_t)cols * 32);
        for (int i = 0; i < rows; i++) {
            const uint64_t *weight = eq + i * 4;
            if (weight[0] == 0 && weight[1] == 0 && weight[2] == 0 && weight[3] == 0)
                continue;
            const uint64_t *row = M + (size_t)i * cols * 4;
            for (int j = 0; j < cols; j++) {
                uint64_t prod[4];
                sp_fr_mul(weight, row + j * 4, prod);
                sp_fr_add(result + j * 4, prod, result + j * 4);
            }
        }
        free(eq);
        return;
    }

    int nChunks = 8;
    if (rows < nChunks) nChunks = rows;
    int perChunk = (rows + nChunks - 1) / nChunks;
    uint64_t *partials = (uint64_t *)malloc((size_t)nChunks * cols * 32);
    int totalRows = rows;

    dispatch_apply(nChunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
        ^(size_t idx) {
            int rowStart = (int)idx * perChunk;
            int rowEnd = rowStart + perChunk;
            if (rowEnd > totalRows) rowEnd = totalRows;
            uint64_t *partial = partials + (size_t)idx * cols * 4;
            memset(partial, 0, (size_t)cols * 32);
            for (int i = rowStart; i < rowEnd; i++) {
                const uint64_t *weight = eq + i * 4;
                if (weight[0] == 0 && weight[1] == 0 && weight[2] == 0 && weight[3] == 0)
                    continue;
                const uint64_t *row = M + (size_t)i * cols * 4;
                for (int j = 0; j < cols; j++) {
                    uint64_t prod[4];
                    sp_fr_mul(weight, row + j * 4, prod);
                    sp_fr_add(partial + j * 4, prod, partial + j * 4);
                }
            }
        });

    // Reduce partial results
    memcpy(result, partials, (size_t)cols * 32);
    for (int t = 1; t < nChunks; t++) {
        const uint64_t *pr = partials + (size_t)t * cols * 4;
        for (int j = 0; j < cols; j++) {
            sp_fr_add(result + j * 4, pr + j * 4, result + j * 4);
        }
    }

    free(partials);
    free(eq);
}
