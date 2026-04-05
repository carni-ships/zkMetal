// Stark252 field CIOS Montgomery arithmetic (CPU-side).
// p = 2^251 + 17 * 2^192 + 1
//   = 0x0800000000000011000000000000000000000000000000000000000000000001
// 4 x uint64_t limbs, little-endian Montgomery form.

#include "NeonFieldOps.h"
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

typedef unsigned __int128 uint128_t;

// ============================================================
// Stark252 Fp constants
// ============================================================

static const uint64_t SK_P[4] = {
    0x0000000000000001ULL, 0x0000000000000000ULL,
    0x0000000000000000ULL, 0x0800000000000011ULL
};

// -p^{-1} mod 2^64
// p mod 2^64 = 1, so p^{-1} mod 2^64 = 1, and -p^{-1} = 2^64 - 1
static const uint64_t SK_INV = 0xFFFFFFFFFFFFFFFFULL;

// R = 2^256 mod p (Montgomery form of 1)
static const uint64_t SK_ONE[4] = {
    0xFFFFFFFFFFFFFFE1ULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0x07FFFFFFFFFFFDF0ULL
};

// R^2 mod p
static const uint64_t SK_R2[4] = {
    0xFFFFFD737E000401ULL, 0x00000001330FFFFFULL,
    0xFFFFFFFFFF6F8000ULL, 0x07FFD4AB5E008810ULL
};

// ============================================================
// CIOS Montgomery multiplication: r = a * b * R^{-1} mod p
// ============================================================

void stark252_fp_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t t0=0,t1=0,t2=0,t3=0,t4=0;
    #define SK_ITER(I) { \
        uint128_t w; uint64_t c; \
        w=(uint128_t)a[I]*b[0]+t0; t0=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[1]+t1+c; t1=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[2]+t2+c; t2=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[3]+t3+c; t3=(uint64_t)w; c=(uint64_t)(w>>64); \
        t4+=c; \
        uint64_t m=t0*SK_INV; \
        w=(uint128_t)m*SK_P[0]+t0; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*SK_P[1]+t1+c; t0=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*SK_P[2]+t2+c; t1=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*SK_P[3]+t3+c; t2=(uint64_t)w; c=(uint64_t)(w>>64); \
        t3=t4+c; t4=0; \
    }
    SK_ITER(0) SK_ITER(1) SK_ITER(2) SK_ITER(3)
    #undef SK_ITER

    // Conditional subtraction
    uint64_t borrow=0; uint64_t r0,r1,r2,r3; uint128_t d;
    d=(uint128_t)t0-SK_P[0]-borrow; r0=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t1-SK_P[1]-borrow; r1=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t2-SK_P[2]-borrow; r2=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t3-SK_P[3]-borrow; r3=(uint64_t)d; borrow=(d>>127)&1;
    if(!borrow){r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;}
    else{r[0]=t0;r[1]=t1;r[2]=t2;r[3]=t3;}
}

// ============================================================
// CIOS Montgomery squaring (reuses mul for now)
// ============================================================

void stark252_fp_sqr(const uint64_t a[4], uint64_t r[4]) {
    stark252_fp_mul(a, a, r);
}

// ============================================================
// Modular addition: r = (a + b) mod p
// ============================================================

void stark252_fp_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t w; uint64_t c=0;
    w=(uint128_t)a[0]+b[0]; r[0]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[1]+b[1]+c; r[1]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[2]+b[2]+c; r[2]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[3]+b[3]+c; r[3]=(uint64_t)w; c=(uint64_t)(w>>64);
    // Conditional subtraction of p
    uint64_t borrow=0; uint64_t r0,r1,r2,r3; uint128_t d;
    d=(uint128_t)r[0]-SK_P[0]; r0=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[1]-SK_P[1]-borrow; r1=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[2]-SK_P[2]-borrow; r2=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[3]-SK_P[3]-borrow; r3=(uint64_t)d; borrow=(d>>127)&1;
    if(c||!borrow){r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;}
}

// ============================================================
// Modular subtraction: r = (a - b) mod p
// ============================================================

void stark252_fp_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t d; uint64_t borrow=0;
    d=(uint128_t)a[0]-b[0]; r[0]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[1]-b[1]-borrow; r[1]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[2]-b[2]-borrow; r[2]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[3]-b[3]-borrow; r[3]=(uint64_t)d; borrow=(d>>127)&1;
    if(borrow){
        uint64_t c=0;
        d=(uint128_t)r[0]+SK_P[0]; r[0]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[1]+SK_P[1]+c; r[1]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[2]+SK_P[2]+c; r[2]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[3]+SK_P[3]+c; r[3]=(uint64_t)d;
    }
}

// ============================================================
// Modular negation: r = (-a) mod p
// ============================================================

void stark252_fp_neg(const uint64_t a[4], uint64_t r[4]) {
    if((a[0] | a[1] | a[2] | a[3]) == 0) {
        r[0]=0; r[1]=0; r[2]=0; r[3]=0;
        return;
    }
    uint128_t d; uint64_t borrow=0;
    d=(uint128_t)SK_P[0]-a[0]; r[0]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)SK_P[1]-a[1]-borrow; r[1]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)SK_P[2]-a[2]-borrow; r[2]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)SK_P[3]-a[3]-borrow; r[3]=(uint64_t)d;
}

// ============================================================
// Stark252 NTT — Cooley-Tukey DIT forward, Gentleman-Sande DIF inverse
// ============================================================

// Primitive 2^192-th root of unity in Montgomery form
// = 3^((p-1)/2^192) * R mod p
static const uint64_t SK_ROOT_2_192[4] = {
    0x4106bccd64a2bdd8ULL, 0xaaada25731fe3be9ULL,
    0x0a35c5be60505574ULL, 0x07222e32c47afc26ULL
};

static const int SK_TWO_ADICITY = 192;

// ============================================================
// Helper: exponentiation and inversion
// ============================================================

static void sk_pow(const uint64_t base[4], uint64_t exp_val, uint64_t result[4]) {
    memcpy(result, SK_ONE, 32);
    uint64_t b[4];
    memcpy(b, base, 32);
    while (exp_val > 0) {
        if (exp_val & 1)
            stark252_fp_mul(result, b, result);
        stark252_fp_mul(b, b, b);
        exp_val >>= 1;
    }
}

static void sk_inv(const uint64_t a[4], uint64_t result[4]) {
    // a^(p-2) mod p via Fermat's little theorem
    uint64_t pm2[4] = {
        SK_P[0] - 2, SK_P[1], SK_P[2], SK_P[3]
    };

    memcpy(result, SK_ONE, 32);
    uint64_t b[4];
    memcpy(b, a, 32);
    for (int i = 0; i < 4; i++) {
        for (int bit = 0; bit < 64; bit++) {
            if ((pm2[i] >> bit) & 1)
                stark252_fp_mul(result, b, result);
            stark252_fp_mul(b, b, b);
        }
    }
}

// ============================================================
// Twiddle cache
// ============================================================

typedef struct {
    uint64_t *fwd;
    uint64_t *inv;
    int logN;
} SkCachedTwiddles;

#define SK_MAX_LOGN 31
static SkCachedTwiddles sk_cached[SK_MAX_LOGN + 1] = {{0}};
static pthread_mutex_t sk_cache_lock = PTHREAD_MUTEX_INITIALIZER;

static void sk_ensure_twiddles(int logN) {
    if (logN > SK_MAX_LOGN) return;
    if (sk_cached[logN].fwd) return;

    pthread_mutex_lock(&sk_cache_lock);
    if (sk_cached[logN].fwd) { pthread_mutex_unlock(&sk_cache_lock); return; }

    int n = 1 << logN;

    // omega = ROOT^(2^(TWO_ADICITY - logN))
    uint64_t omega[4];
    memcpy(omega, SK_ROOT_2_192, 32);
    for (int i = 0; i < SK_TWO_ADICITY - logN; i++)
        stark252_fp_mul(omega, omega, omega);

    uint64_t omega_inv[4];
    sk_inv(omega, omega_inv);

    uint64_t *fwd = (uint64_t *)malloc((size_t)(n - 1) * 32);
    uint64_t *inv = (uint64_t *)malloc((size_t)(n - 1) * 32);

    for (int s = 0; s < logN; s++) {
        int halfBlock = 1 << s;
        int offset = halfBlock - 1;

        uint64_t w_m[4], w_m_inv[4];
        sk_pow(omega, (uint64_t)(n >> (s + 1)), w_m);
        sk_pow(omega_inv, (uint64_t)(n >> (s + 1)), w_m_inv);

        uint64_t w[4], wi[4];
        memcpy(w, SK_ONE, 32);
        memcpy(wi, SK_ONE, 32);

        for (int j = 0; j < halfBlock; j++) {
            memcpy(&fwd[(offset + j) * 4], w, 32);
            memcpy(&inv[(offset + j) * 4], wi, 32);
            stark252_fp_mul(w, w_m, w);
            stark252_fp_mul(wi, w_m_inv, wi);
        }
    }

    sk_cached[logN].fwd = fwd;
    sk_cached[logN].inv = inv;
    sk_cached[logN].logN = logN;
    pthread_mutex_unlock(&sk_cache_lock);
}

static void sk_bit_reverse_permute(uint64_t *data, int logN) {
    int n = 1 << logN;
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) {
            uint64_t t0 = data[i*4+0]; data[i*4+0] = data[j*4+0]; data[j*4+0] = t0;
            uint64_t t1 = data[i*4+1]; data[i*4+1] = data[j*4+1]; data[j*4+1] = t1;
            uint64_t t2 = data[i*4+2]; data[i*4+2] = data[j*4+2]; data[j*4+2] = t2;
            uint64_t t3 = data[i*4+3]; data[i*4+3] = data[j*4+3]; data[j*4+3] = t3;
        }
    }
}

void stark252_ntt(uint64_t *data, int logN) {
    if (logN <= 0) return;
    int n = 1 << logN;

    sk_ensure_twiddles(logN);
    const uint64_t *tw = sk_cached[logN].fwd;

    sk_bit_reverse_permute(data, logN);

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
                stark252_fp_mul(twj, vp, v);

                uint64_t sum[4], diff[4];
                stark252_fp_add(u, v, sum);
                stark252_fp_sub(u, v, diff);

                memcpy(u, sum, 32);
                memcpy(vp, diff, 32);
            }
        }
    }
}

void stark252_intt(uint64_t *data, int logN) {
    if (logN <= 0) return;
    int n = 1 << logN;

    sk_ensure_twiddles(logN);
    const uint64_t *tw = sk_cached[logN].inv;

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
                stark252_fp_add(a, b, sum);
                stark252_fp_sub(a, b, diff);
                stark252_fp_mul(diff, twj, prod);

                memcpy(ap, sum, 32);
                memcpy(bp, prod, 32);
            }
        }
    }

    sk_bit_reverse_permute(data, logN);

    uint64_t n_plain[4] = {(uint64_t)n, 0, 0, 0};
    uint64_t n_mont[4];
    stark252_fp_mul(n_plain, SK_R2, n_mont);

    uint64_t n_inv[4];
    sk_inv(n_mont, n_inv);

    for (int i = 0; i < n; i++) {
        stark252_fp_mul(&data[i * 4], n_inv, &data[i * 4]);
    }
}
