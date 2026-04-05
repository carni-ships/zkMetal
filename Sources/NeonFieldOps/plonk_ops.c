// Plonk permutation Z accumulator computation in C with CIOS Montgomery arithmetic.
// Fuses the entire numerator/denominator loop + batch inverse + running product.

#include "NeonFieldOps.h"
#include <string.h>
#include <stdlib.h>

typedef unsigned __int128 uint128_t;

// BN254 Fr constants
static const uint64_t PK_FR_P[4] = {
    0x43e1f593f0000001ULL, 0x2833e84879b97091ULL,
    0xb85045b68181585dULL, 0x30644e72e131a029ULL
};
static const uint64_t PK_FR_INV = 0xc2e1f593efffffffULL;
static const uint64_t PK_FR_ONE[4] = {
    0xac96341c4ffffffbULL, 0x36fc76959f60cd29ULL,
    0x666ea36f7879462eULL, 0x0e0a77c19a07df2fULL
};

// CIOS Montgomery multiplication
static inline void pk_fr_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t t0=0,t1=0,t2=0,t3=0,t4=0;
    #define PK_ITER(I) { \
        uint128_t w; uint64_t c; \
        w=(uint128_t)a[I]*b[0]+t0; t0=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[1]+t1+c; t1=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[2]+t2+c; t2=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[3]+t3+c; t3=(uint64_t)w; c=(uint64_t)(w>>64); \
        t4+=c; \
        uint64_t m=t0*PK_FR_INV; \
        w=(uint128_t)m*PK_FR_P[0]+t0; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*PK_FR_P[1]+t1+c; t0=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*PK_FR_P[2]+t2+c; t1=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*PK_FR_P[3]+t3+c; t2=(uint64_t)w; c=(uint64_t)(w>>64); \
        t3=t4+c; t4=0; \
    }
    PK_ITER(0) PK_ITER(1) PK_ITER(2) PK_ITER(3)
    #undef PK_ITER
    uint64_t borrow=0; uint64_t r0,r1,r2,r3; uint128_t d;
    d=(uint128_t)t0-PK_FR_P[0]-borrow; r0=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t1-PK_FR_P[1]-borrow; r1=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t2-PK_FR_P[2]-borrow; r2=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t3-PK_FR_P[3]-borrow; r3=(uint64_t)d; borrow=(d>>127)&1;
    if(!borrow){r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;}
    else{r[0]=t0;r[1]=t1;r[2]=t2;r[3]=t3;}
}

// Modular add
static inline void pk_fr_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t w; uint64_t c=0;
    w=(uint128_t)a[0]+b[0]; r[0]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[1]+b[1]+c; r[1]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[2]+b[2]+c; r[2]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[3]+b[3]+c; r[3]=(uint64_t)w; c=(uint64_t)(w>>64);
    uint64_t borrow=0; uint64_t r0,r1,r2,r3; uint128_t d;
    d=(uint128_t)r[0]-PK_FR_P[0]; r0=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[1]-PK_FR_P[1]-borrow; r1=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[2]-PK_FR_P[2]-borrow; r2=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[3]-PK_FR_P[3]-borrow; r3=(uint64_t)d; borrow=(d>>127)&1;
    if(c||!borrow){r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;}
}

// Fermat inverse: a^(p-2) mod p
static void pk_fr_inv(const uint64_t a[4], uint64_t r[4]) {
    // p-2 for BN254 Fr
    static const uint64_t PM2[4] = {
        0x43e1f593effffffful, 0x2833e84879b97091ul,
        0xb85045b68181585dul, 0x30644e72e131a029ul
    };
    uint64_t base[4], acc[4];
    memcpy(base, a, 32);
    memcpy(acc, PK_FR_ONE, 32);
    for (int i = 0; i < 4; i++) {
        uint64_t limb = PM2[i];
        for (int j = 0; j < 64; j++) {
            if (limb & 1) pk_fr_mul(acc, base, acc);
            pk_fr_mul(base, base, base);
            limb >>= 1;
        }
    }
    memcpy(r, acc, 32);
}

// Batch inverse using Montgomery's trick: out[i] = a[i]^(-1)
static void pk_fr_batch_inverse(const uint64_t *a, int n, uint64_t *out) {
    if (n == 0) return;
    if (n == 1) { pk_fr_inv(a, out); return; }
    // Phase 1: prefix products
    memcpy(out, a, 32);
    for (int i = 1; i < n; i++) {
        pk_fr_mul(out + (i-1)*4, a + i*4, out + i*4);
    }
    // Phase 2: invert total product
    uint64_t inv[4];
    pk_fr_inv(out + (n-1)*4, inv);
    // Phase 3: back-propagate
    for (int i = n-1; i >= 1; i--) {
        uint64_t tmp[4];
        pk_fr_mul(inv, out + (i-1)*4, tmp);
        uint64_t new_inv[4];
        pk_fr_mul(inv, a + i*4, new_inv);
        memcpy(inv, new_inv, 32);
        memcpy(out + i*4, tmp, 32);
    }
    memcpy(out, inv, 32);
}

/// Fused Plonk permutation Z accumulator computation.
/// Computes zEvals[0..n-1] where:
///   zEvals[0] = 1
///   zEvals[i+1] = zEvals[i] * num[i] / den[i]
/// with:
///   num[i] = (a[i] + beta*domain[i] + gamma) * (b[i] + k1*beta*domain[i] + gamma) * (c[i] + k2*beta*domain[i] + gamma)
///   den[i] = (a[i] + beta*sigma1[i] + gamma) * (b[i] + beta*sigma2[i] + gamma) * (c[i] + beta*sigma3[i] + gamma)
///
/// All inputs/outputs are BN254 Fr in Montgomery form (4 x uint64_t per element).
void plonk_compute_z_accumulator(
    const uint64_t *aEvals,   // n Fr elements
    const uint64_t *bEvals,   // n Fr elements
    const uint64_t *cEvals,   // n Fr elements
    const uint64_t *sigma1,   // n Fr elements
    const uint64_t *sigma2,   // n Fr elements
    const uint64_t *sigma3,   // n Fr elements
    const uint64_t *domain,   // n Fr elements
    const uint64_t beta[4],
    const uint64_t gamma[4],
    const uint64_t k1[4],
    const uint64_t k2[4],
    int n,
    uint64_t *zEvals          // output: n Fr elements
) {
    int m = n - 1;
    if (m <= 0) {
        memcpy(zEvals, PK_FR_ONE, 32);
        return;
    }

    // Allocate workspace for numerators and denominators
    uint64_t *nums = (uint64_t *)malloc((size_t)m * 32);
    uint64_t *dens = (uint64_t *)malloc((size_t)m * 32);
    uint64_t *denInvs = (uint64_t *)malloc((size_t)m * 32);

    // Compute numerators and denominators
    for (int i = 0; i < m; i++) {
        const uint64_t *ai = aEvals + i*4;
        const uint64_t *bi = bEvals + i*4;
        const uint64_t *ci = cEvals + i*4;
        const uint64_t *di = domain + i*4;
        const uint64_t *s1i = sigma1 + i*4;
        const uint64_t *s2i = sigma2 + i*4;
        const uint64_t *s3i = sigma3 + i*4;

        // betaDomain = beta * domain[i]
        uint64_t betaDomain[4];
        pk_fr_mul(beta, di, betaDomain);

        // k1BetaDomain = k1 * betaDomain
        uint64_t k1bd[4];
        pk_fr_mul(k1, betaDomain, k1bd);

        // k2BetaDomain = k2 * betaDomain
        uint64_t k2bd[4];
        pk_fr_mul(k2, betaDomain, k2bd);

        // num1 = a[i] + betaDomain + gamma
        uint64_t tmp1[4], num1[4];
        pk_fr_add(ai, betaDomain, tmp1);
        pk_fr_add(tmp1, gamma, num1);

        // num2 = b[i] + k1*betaDomain + gamma
        uint64_t tmp2[4], num2[4];
        pk_fr_add(bi, k1bd, tmp2);
        pk_fr_add(tmp2, gamma, num2);

        // num3 = c[i] + k2*betaDomain + gamma
        uint64_t tmp3[4], num3[4];
        pk_fr_add(ci, k2bd, tmp3);
        pk_fr_add(tmp3, gamma, num3);

        // nums[i] = num1 * num2 * num3
        uint64_t nn12[4];
        pk_fr_mul(num1, num2, nn12);
        pk_fr_mul(nn12, num3, nums + i*4);

        // den1 = a[i] + beta*sigma1[i] + gamma
        uint64_t bs1[4], d1t[4], den1[4];
        pk_fr_mul(beta, s1i, bs1);
        pk_fr_add(ai, bs1, d1t);
        pk_fr_add(d1t, gamma, den1);

        // den2 = b[i] + beta*sigma2[i] + gamma
        uint64_t bs2[4], d2t[4], den2[4];
        pk_fr_mul(beta, s2i, bs2);
        pk_fr_add(bi, bs2, d2t);
        pk_fr_add(d2t, gamma, den2);

        // den3 = c[i] + beta*sigma3[i] + gamma
        uint64_t bs3[4], d3t[4], den3[4];
        pk_fr_mul(beta, s3i, bs3);
        pk_fr_add(ci, bs3, d3t);
        pk_fr_add(d3t, gamma, den3);

        // dens[i] = den1 * den2 * den3
        uint64_t dd12[4];
        pk_fr_mul(den1, den2, dd12);
        pk_fr_mul(dd12, den3, dens + i*4);
    }

    // Batch inverse of all denominators
    pk_fr_batch_inverse(dens, m, denInvs);

    // Running product: zEvals[0] = 1, zEvals[i+1] = zEvals[i] * nums[i] * denInvs[i]
    memcpy(zEvals, PK_FR_ONE, 32);
    for (int i = 0; i < m; i++) {
        uint64_t nd[4];
        pk_fr_mul(nums + i*4, denInvs + i*4, nd);
        pk_fr_mul(zEvals + i*4, nd, zEvals + (i+1)*4);
    }

    free(nums);
    free(dens);
    free(denInvs);
}
