// CCS (Customizable Constraint System) C-accelerated operations.
// Uses CIOS Montgomery multiplication for BN254 Fr field arithmetic.
// All arrays use 4 x uint64_t per Fr element (Montgomery form).

#include "NeonFieldOps.h"
#include <string.h>
#include <stdlib.h>

typedef unsigned __int128 uint128_t;

// BN254 Fr constants
static const uint64_t CCS_FR_P[4] = {
    0x43e1f593f0000001ULL, 0x2833e84879b97091ULL,
    0xb85045b68181585dULL, 0x30644e72e131a029ULL
};
static const uint64_t CCS_FR_INV = 0xc2e1f593efffffffULL;
static const uint64_t CCS_FR_ONE[4] = {
    0xac96341c4ffffffbULL, 0x36fc76959f60cd29ULL,
    0x666ea36f7879462eULL, 0x0e0a77c19a07df2fULL
};

// CIOS Montgomery multiplication
static inline void ccs_fr_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t t0=0,t1=0,t2=0,t3=0,t4=0;
    #define CCS_ITER(I) { \
        uint128_t w; uint64_t c; \
        w=(uint128_t)a[I]*b[0]+t0; t0=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[1]+t1+c; t1=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[2]+t2+c; t2=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[3]+t3+c; t3=(uint64_t)w; c=(uint64_t)(w>>64); \
        t4+=c; \
        uint64_t m=t0*CCS_FR_INV; \
        w=(uint128_t)m*CCS_FR_P[0]+t0; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*CCS_FR_P[1]+t1+c; t0=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*CCS_FR_P[2]+t2+c; t1=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*CCS_FR_P[3]+t3+c; t2=(uint64_t)w; c=(uint64_t)(w>>64); \
        t3=t4+c; t4=0; \
    }
    CCS_ITER(0) CCS_ITER(1) CCS_ITER(2) CCS_ITER(3)
    #undef CCS_ITER
    uint64_t borrow=0; uint64_t r0,r1,r2,r3; uint128_t d;
    d=(uint128_t)t0-CCS_FR_P[0]-borrow; r0=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t1-CCS_FR_P[1]-borrow; r1=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t2-CCS_FR_P[2]-borrow; r2=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t3-CCS_FR_P[3]-borrow; r3=(uint64_t)d; borrow=(d>>127)&1;
    if(!borrow){r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;}
    else{r[0]=t0;r[1]=t1;r[2]=t2;r[3]=t3;}
}

// Modular add
static inline void ccs_fr_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t w; uint64_t c=0;
    w=(uint128_t)a[0]+b[0]; r[0]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[1]+b[1]+c; r[1]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[2]+b[2]+c; r[2]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[3]+b[3]+c; r[3]=(uint64_t)w; c=(uint64_t)(w>>64);
    uint64_t borrow=0; uint64_t r0,r1,r2,r3; uint128_t d;
    d=(uint128_t)r[0]-CCS_FR_P[0]; r0=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[1]-CCS_FR_P[1]-borrow; r1=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[2]-CCS_FR_P[2]-borrow; r2=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[3]-CCS_FR_P[3]-borrow; r3=(uint64_t)d; borrow=(d>>127)&1;
    if(c||!borrow){r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;}
}

// ============================================================
// CSR Sparse Matrix-Vector Multiply
// ============================================================

void ccs_sparse_matvec(uint64_t *result,
                       const int *rowPtr, const int *colIdx,
                       const uint64_t *values, const uint64_t *z,
                       int nRows) {
    memset(result, 0, (size_t)nRows * 32);
    for (int i = 0; i < nRows; i++) {
        uint64_t acc[4] = {0,0,0,0};
        int start = rowPtr[i];
        int end = rowPtr[i + 1];
        for (int k = start; k < end; k++) {
            int col = colIdx[k];
            uint64_t prod[4];
            ccs_fr_mul(values + k * 4, z + col * 4, prod);
            ccs_fr_add(acc, prod, acc);
        }
        memcpy(result + i * 4, acc, 32);
    }
}

// ============================================================
// Fused Hadamard Product + Coefficient-Weighted Accumulation
// ============================================================
//
// For each term j in 0..<nTerms:
//   hadamard[i] = product of matricesResults[j][k][i] for k in 0..<nMatricesPerTerm[j]
//   acc[i] += coefficients[j] * hadamard[i]
//
// matricesResults: flat array of pointers to pre-computed M*z vectors.
// matResultPtrs[j * maxDegree + k] points to m Fr elements.
// nMatricesPerTerm[j] = number of matrices in multiset j.

void ccs_hadamard_accumulate(uint64_t *acc,
                             const uint64_t * const *matResultPtrs,
                             const int *nMatricesPerTerm,
                             const uint64_t *coefficients,
                             int nTerms, int maxDegree, int m) {
    memset(acc, 0, (size_t)m * 32);
    for (int j = 0; j < nTerms; j++) {
        int d = nMatricesPerTerm[j];
        const uint64_t *coeff = coefficients + j * 4;
        for (int i = 0; i < m; i++) {
            // Compute hadamard product for row i across matrices in term j
            uint64_t h[4];
            memcpy(h, CCS_FR_ONE, 32);
            for (int k = 0; k < d; k++) {
                const uint64_t *mv = matResultPtrs[j * maxDegree + k];
                uint64_t tmp[4];
                ccs_fr_mul(h, mv + i * 4, tmp);
                memcpy(h, tmp, 32);
            }
            // acc[i] += coeff * h
            uint64_t scaled[4];
            ccs_fr_mul(coeff, h, scaled);
            uint64_t sum[4];
            ccs_fr_add(acc + i * 4, scaled, sum);
            memcpy(acc + i * 4, sum, 32);
        }
    }
}

// ============================================================
// Compute single CCS term: c_j * hadamard(M_{S_j} * z)
// ============================================================

void ccs_compute_term(uint64_t *result,
                      const uint64_t * const *matVecResults,
                      int nMatrices,
                      const uint64_t coeff[4],
                      int m) {
    for (int i = 0; i < m; i++) {
        uint64_t h[4];
        memcpy(h, CCS_FR_ONE, 32);
        for (int k = 0; k < nMatrices; k++) {
            uint64_t tmp[4];
            ccs_fr_mul(h, matVecResults[k] + i * 4, tmp);
            memcpy(h, tmp, 32);
        }
        ccs_fr_mul(coeff, h, result + i * 4);
    }
}
