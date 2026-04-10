// Pasta (Pallas + Vesta) NTT — Cooley-Tukey DIT / Gentleman-Sande DIF
// NTT operates on SCALAR fields:
//   "Pallas NTT" uses Pallas Fr = Vesta Fp arithmetic
//   "Vesta NTT"  uses Vesta Fr = Pallas Fp arithmetic
// Each element is 4 x uint64_t in Montgomery form.

#include "NeonFieldOps.h"
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <dispatch/dispatch.h>

typedef unsigned __int128 uint128_t;

// ============================================================
// Pallas Fr = Vesta Fp constants
// p = 0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001
// ============================================================

static const uint64_t PALLAS_FR_P[4] = {
    0x8c46eb2100000001ULL, 0x224698fc0994a8ddULL,
    0x0000000000000000ULL, 0x4000000000000000ULL
};
static const uint64_t PALLAS_FR_INV = 0x8c46eb20ffffffffULL; // -p^{-1} mod 2^64
static const uint64_t PALLAS_FR_ONE[4] = { // R mod p (Montgomery form of 1)
    0x5b2b3e9cfffffffdULL, 0x992c350be3420567ULL,
    0xffffffffffffffffULL, 0x3fffffffffffffffULL
};
static const uint64_t PALLAS_FR_R2[4] = { // R^2 mod p
    0xfc9678ff0000000fULL, 0x67bb433d891a16e3ULL,
    0x7fae231004ccf590ULL, 0x096d41af7ccfdaa9ULL
};

// Primitive 2^32-th root of unity for Pallas Fr (in STANDARD form, not Montgomery).
// = 5^((p-1)/2^32) mod p, where p is the Pallas scalar field (= Vesta base field).
// Computed: pow(5, (p-1)/2^32, p)
static const uint64_t PALLAS_FR_ROOT_2_32[4] = {
    0xa70e2c1102b6d05fULL, 0x9bb97ea3c106f049ULL,
    0x9e5c4dfd492ae26eULL, 0x2de6a9b8746d3f58ULL
};
static const int PALLAS_FR_TWO_ADICITY = 32;

// ============================================================
// Vesta Fr = Pallas Fp constants
// p = 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001
// ============================================================

static const uint64_t VESTA_FR_P[4] = {
    0x992d30ed00000001ULL, 0x224698fc094cf91bULL,
    0x0000000000000000ULL, 0x4000000000000000ULL
};
static const uint64_t VESTA_FR_INV = 0x992d30ecffffffffULL; // -p^{-1} mod 2^64
static const uint64_t VESTA_FR_ONE[4] = { // R mod p (Montgomery form of 1)
    0x34786d38fffffffdULL, 0x992c350be41914adULL,
    0xffffffffffffffffULL, 0x3fffffffffffffffULL
};
static const uint64_t VESTA_FR_R2[4] = { // R^2 mod p
    0x8c78ecb30000000fULL, 0xd7d30dbd8b0de0e7ULL,
    0x7797a99bc3c95d18ULL, 0x096d41af7b9cb714ULL
};

// Primitive 2^32-th root of unity for Vesta Fr (in STANDARD form, not Montgomery).
// = 5^((p-1)/2^32) mod p, where p is the Vesta scalar field (= Pallas base field).
// Computed: pow(5, (p-1)/2^32, p)
static const uint64_t VESTA_FR_ROOT_2_32[4] = {
    0xbdad6fabd87ea32fULL, 0xea322bf2b7bb7584ULL,
    0x362120830561f81aULL, 0x2bce74deac30ebdaULL
};
static const int VESTA_FR_TWO_ADICITY = 32;

// ============================================================
// CIOS Montgomery multiplication for Pallas Fr (= Vesta Fp)
// ============================================================

static inline void pfr_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t t0=0,t1=0,t2=0,t3=0,t4=0;
    #define PFR_ITER(I) { \
        uint128_t w; uint64_t c; \
        w=(uint128_t)a[I]*b[0]+t0; t0=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[1]+t1+c; t1=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[2]+t2+c; t2=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[3]+t3+c; t3=(uint64_t)w; c=(uint64_t)(w>>64); \
        t4+=c; \
        uint64_t m=t0*PALLAS_FR_INV; \
        w=(uint128_t)m*PALLAS_FR_P[0]+t0; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*PALLAS_FR_P[1]+t1+c; t0=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*PALLAS_FR_P[2]+t2+c; t1=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*PALLAS_FR_P[3]+t3+c; t2=(uint64_t)w; c=(uint64_t)(w>>64); \
        t3=t4+c; t4=0; \
    }
    PFR_ITER(0) PFR_ITER(1) PFR_ITER(2) PFR_ITER(3)
    #undef PFR_ITER
    uint64_t borrow=0; uint64_t r0,r1,r2,r3; uint128_t d;
    d=(uint128_t)t0-PALLAS_FR_P[0]-borrow; r0=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t1-PALLAS_FR_P[1]-borrow; r1=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t2-PALLAS_FR_P[2]-borrow; r2=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t3-PALLAS_FR_P[3]-borrow; r3=(uint64_t)d; borrow=(d>>127)&1;
    if(!borrow){r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;}
    else{r[0]=t0;r[1]=t1;r[2]=t2;r[3]=t3;}
}

static inline void pfr_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t w; uint64_t c=0;
    w=(uint128_t)a[0]+b[0]; r[0]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[1]+b[1]+c; r[1]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[2]+b[2]+c; r[2]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[3]+b[3]+c; r[3]=(uint64_t)w; c=(uint64_t)(w>>64);
    uint64_t borrow=0; uint64_t r0,r1,r2,r3; uint128_t d;
    d=(uint128_t)r[0]-PALLAS_FR_P[0]; r0=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[1]-PALLAS_FR_P[1]-borrow; r1=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[2]-PALLAS_FR_P[2]-borrow; r2=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[3]-PALLAS_FR_P[3]-borrow; r3=(uint64_t)d; borrow=(d>>127)&1;
    if(c||!borrow){r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;}
}

static inline void pfr_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t d; uint64_t borrow=0;
    d=(uint128_t)a[0]-b[0]; r[0]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[1]-b[1]-borrow; r[1]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[2]-b[2]-borrow; r[2]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[3]-b[3]-borrow; r[3]=(uint64_t)d; borrow=(d>>127)&1;
    if(borrow){
        uint64_t c=0;
        d=(uint128_t)r[0]+PALLAS_FR_P[0]; r[0]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[1]+PALLAS_FR_P[1]+c; r[1]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[2]+PALLAS_FR_P[2]+c; r[2]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[3]+PALLAS_FR_P[3]+c; r[3]=(uint64_t)d;
    }
}

// ============================================================
// CIOS Montgomery multiplication for Vesta Fr (= Pallas Fp)
// ============================================================

static inline void vfr_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t t0=0,t1=0,t2=0,t3=0,t4=0;
    #define VFR_ITER(I) { \
        uint128_t w; uint64_t c; \
        w=(uint128_t)a[I]*b[0]+t0; t0=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[1]+t1+c; t1=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[2]+t2+c; t2=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)a[I]*b[3]+t3+c; t3=(uint64_t)w; c=(uint64_t)(w>>64); \
        t4+=c; \
        uint64_t m=t0*VESTA_FR_INV; \
        w=(uint128_t)m*VESTA_FR_P[0]+t0; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*VESTA_FR_P[1]+t1+c; t0=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*VESTA_FR_P[2]+t2+c; t1=(uint64_t)w; c=(uint64_t)(w>>64); \
        w=(uint128_t)m*VESTA_FR_P[3]+t3+c; t2=(uint64_t)w; c=(uint64_t)(w>>64); \
        t3=t4+c; t4=0; \
    }
    VFR_ITER(0) VFR_ITER(1) VFR_ITER(2) VFR_ITER(3)
    #undef VFR_ITER
    uint64_t borrow=0; uint64_t r0,r1,r2,r3; uint128_t d;
    d=(uint128_t)t0-VESTA_FR_P[0]-borrow; r0=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t1-VESTA_FR_P[1]-borrow; r1=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t2-VESTA_FR_P[2]-borrow; r2=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)t3-VESTA_FR_P[3]-borrow; r3=(uint64_t)d; borrow=(d>>127)&1;
    if(!borrow){r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;}
    else{r[0]=t0;r[1]=t1;r[2]=t2;r[3]=t3;}
}

static inline void vfr_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t w; uint64_t c=0;
    w=(uint128_t)a[0]+b[0]; r[0]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[1]+b[1]+c; r[1]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[2]+b[2]+c; r[2]=(uint64_t)w; c=(uint64_t)(w>>64);
    w=(uint128_t)a[3]+b[3]+c; r[3]=(uint64_t)w; c=(uint64_t)(w>>64);
    uint64_t borrow=0; uint64_t r0,r1,r2,r3; uint128_t d;
    d=(uint128_t)r[0]-VESTA_FR_P[0]; r0=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[1]-VESTA_FR_P[1]-borrow; r1=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[2]-VESTA_FR_P[2]-borrow; r2=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[3]-VESTA_FR_P[3]-borrow; r3=(uint64_t)d; borrow=(d>>127)&1;
    if(c||!borrow){r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;}
}

static inline void vfr_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t d; uint64_t borrow=0;
    d=(uint128_t)a[0]-b[0]; r[0]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[1]-b[1]-borrow; r[1]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[2]-b[2]-borrow; r[2]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[3]-b[3]-borrow; r[3]=(uint64_t)d; borrow=(d>>127)&1;
    if(borrow){
        uint64_t c=0;
        d=(uint128_t)r[0]+VESTA_FR_P[0]; r[0]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[1]+VESTA_FR_P[1]+c; r[1]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[2]+VESTA_FR_P[2]+c; r[2]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[3]+VESTA_FR_P[3]+c; r[3]=(uint64_t)d;
    }
}

// ============================================================
// Helper: Fermat-based power and inverse
// ============================================================

static void pfr_pow(const uint64_t base[4], uint64_t exp_val, uint64_t result[4]) {
    memcpy(result, PALLAS_FR_ONE, 32);
    uint64_t b[4];
    memcpy(b, base, 32);
    while (exp_val > 0) {
        if (exp_val & 1) pfr_mul(result, b, result);
        pfr_mul(b, b, b);
        exp_val >>= 1;
    }
}

static void pfr_inv(const uint64_t a[4], uint64_t result[4]) {
    uint64_t pm2[4];
    pm2[0] = PALLAS_FR_P[0] - 2;
    pm2[1] = PALLAS_FR_P[1];
    pm2[2] = PALLAS_FR_P[2];
    pm2[3] = PALLAS_FR_P[3];

    memcpy(result, PALLAS_FR_ONE, 32);
    uint64_t b[4];
    memcpy(b, a, 32);
    for (int i = 0; i < 4; i++) {
        for (int bit = 0; bit < 64; bit++) {
            if ((pm2[i] >> bit) & 1)
                pfr_mul(result, b, result);
            pfr_mul(b, b, b);
        }
    }
}

static void vfr_pow(const uint64_t base[4], uint64_t exp_val, uint64_t result[4]) {
    memcpy(result, VESTA_FR_ONE, 32);
    uint64_t b[4];
    memcpy(b, base, 32);
    while (exp_val > 0) {
        if (exp_val & 1) vfr_mul(result, b, result);
        vfr_mul(b, b, b);
        exp_val >>= 1;
    }
}

static void vfr_inv(const uint64_t a[4], uint64_t result[4]) {
    uint64_t pm2[4];
    pm2[0] = VESTA_FR_P[0] - 2;
    pm2[1] = VESTA_FR_P[1];
    pm2[2] = VESTA_FR_P[2];
    pm2[3] = VESTA_FR_P[3];

    memcpy(result, VESTA_FR_ONE, 32);
    uint64_t b[4];
    memcpy(b, a, 32);
    for (int i = 0; i < 4; i++) {
        for (int bit = 0; bit < 64; bit++) {
            if ((pm2[i] >> bit) & 1)
                vfr_mul(result, b, result);
            vfr_mul(b, b, b);
        }
    }
}

// ============================================================
// Bit-reversal permutation (4-limb elements)
// ============================================================

static void pasta_bit_reverse_permute(uint64_t *data, int logN) {
    int n = 1 << logN;
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) {
            uint64_t t;
            t = data[i*4+0]; data[i*4+0] = data[j*4+0]; data[j*4+0] = t;
            t = data[i*4+1]; data[i*4+1] = data[j*4+1]; data[j*4+1] = t;
            t = data[i*4+2]; data[i*4+2] = data[j*4+2]; data[j*4+2] = t;
            t = data[i*4+3]; data[i*4+3] = data[j*4+3]; data[j*4+3] = t;
        }
    }
}

// ============================================================
// Pallas Fr twiddle cache
// ============================================================

typedef struct { uint64_t *fwd; uint64_t *inv; int logN; } pasta_tw_t;
static pasta_tw_t pfr_cached[33] = {{0}};
static pthread_mutex_t pfr_cache_lock = PTHREAD_MUTEX_INITIALIZER;

static void pfr_ensure_twiddles(int logN) {
    if (pfr_cached[logN].fwd) return;

    pthread_mutex_lock(&pfr_cache_lock);
    if (pfr_cached[logN].fwd) { pthread_mutex_unlock(&pfr_cache_lock); return; }

    int n = 1 << logN;

    // Convert root from standard to Montgomery form
    uint64_t root_mont[4];
    pfr_mul(PALLAS_FR_ROOT_2_32, PALLAS_FR_R2, root_mont);

    // omega = root^(2^(TWO_ADICITY - logN))
    uint64_t omega[4];
    memcpy(omega, root_mont, 32);
    for (int i = 0; i < PALLAS_FR_TWO_ADICITY - logN; i++)
        pfr_mul(omega, omega, omega);

    uint64_t omega_inv[4];
    pfr_inv(omega, omega_inv);

    uint64_t *fwd = (uint64_t *)malloc((size_t)(n - 1) * 32);
    uint64_t *inv = (uint64_t *)malloc((size_t)(n - 1) * 32);

    for (int s = 0; s < logN; s++) {
        int halfBlock = 1 << s;
        int offset = halfBlock - 1;

        uint64_t w_m[4], w_m_inv[4];
        pfr_pow(omega, (uint64_t)(n >> (s + 1)), w_m);
        pfr_pow(omega_inv, (uint64_t)(n >> (s + 1)), w_m_inv);

        uint64_t w[4], wi[4];
        memcpy(w, PALLAS_FR_ONE, 32);
        memcpy(wi, PALLAS_FR_ONE, 32);

        for (int j = 0; j < halfBlock; j++) {
            memcpy(&fwd[(offset + j) * 4], w, 32);
            memcpy(&inv[(offset + j) * 4], wi, 32);
            pfr_mul(w, w_m, w);
            pfr_mul(wi, w_m_inv, wi);
        }
    }

    pfr_cached[logN].fwd = fwd;
    pfr_cached[logN].inv = inv;
    pfr_cached[logN].logN = logN;
    pthread_mutex_unlock(&pfr_cache_lock);
}

// ============================================================
// Vesta Fr twiddle cache
// ============================================================

static pasta_tw_t vfr_cached[33] = {{0}};
static pthread_mutex_t vfr_cache_lock = PTHREAD_MUTEX_INITIALIZER;

static void vfr_ensure_twiddles(int logN) {
    if (vfr_cached[logN].fwd) return;

    pthread_mutex_lock(&vfr_cache_lock);
    if (vfr_cached[logN].fwd) { pthread_mutex_unlock(&vfr_cache_lock); return; }

    int n = 1 << logN;

    uint64_t root_mont[4];
    vfr_mul(VESTA_FR_ROOT_2_32, VESTA_FR_R2, root_mont);

    uint64_t omega[4];
    memcpy(omega, root_mont, 32);
    for (int i = 0; i < VESTA_FR_TWO_ADICITY - logN; i++)
        vfr_mul(omega, omega, omega);

    uint64_t omega_inv[4];
    vfr_inv(omega, omega_inv);

    uint64_t *fwd = (uint64_t *)malloc((size_t)(n - 1) * 32);
    uint64_t *inv = (uint64_t *)malloc((size_t)(n - 1) * 32);

    for (int s = 0; s < logN; s++) {
        int halfBlock = 1 << s;
        int offset = halfBlock - 1;

        uint64_t w_m[4], w_m_inv[4];
        vfr_pow(omega, (uint64_t)(n >> (s + 1)), w_m);
        vfr_pow(omega_inv, (uint64_t)(n >> (s + 1)), w_m_inv);

        uint64_t w[4], wi[4];
        memcpy(w, VESTA_FR_ONE, 32);
        memcpy(wi, VESTA_FR_ONE, 32);

        for (int j = 0; j < halfBlock; j++) {
            memcpy(&fwd[(offset + j) * 4], w, 32);
            memcpy(&inv[(offset + j) * 4], wi, 32);
            vfr_mul(w, w_m, w);
            vfr_mul(wi, w_m_inv, wi);
        }
    }

    vfr_cached[logN].fwd = fwd;
    vfr_cached[logN].inv = inv;
    vfr_cached[logN].logN = logN;
    pthread_mutex_unlock(&vfr_cache_lock);
}

// ============================================================
// Pallas Fr NTT: Forward (Cooley-Tukey DIT)
// ============================================================

void pallas_fr_ntt(uint64_t *data, int logN) {
    if (logN <= 0) return;
    int n = 1 << logN;

    pfr_ensure_twiddles(logN);
    const uint64_t *tw = pfr_cached[logN].fwd;

    pasta_bit_reverse_permute(data, logN);

    for (int s = 0; s < logN; s++) {
        int halfBlock = 1 << s;
        int blockSize = halfBlock << 1;
        int nBlocks = n / blockSize;
        int twOffset = halfBlock - 1;

        for (int bk = 0; bk < nBlocks; bk++) {
            int base = bk * blockSize;
            // j==0: twiddle==1, skip Montgomery mul
            {
                uint64_t *u = &data[base * 4];
                uint64_t *vp = &data[(base + halfBlock) * 4];
                uint64_t sum[4], diff[4];
                pfr_add(u, vp, sum);
                pfr_sub(u, vp, diff);
                memcpy(u, sum, 32);
                memcpy(vp, diff, 32);
            }
            for (int j = 1; j < halfBlock; j++) {
                uint64_t *u = &data[(base + j) * 4];
                uint64_t *vp = &data[(base + j + halfBlock) * 4];
                const uint64_t *twj = &tw[(twOffset + j) * 4];
                uint64_t v[4];
                pfr_mul(twj, vp, v);
                uint64_t sum[4], diff[4];
                pfr_add(u, v, sum);
                pfr_sub(u, v, diff);
                memcpy(u, sum, 32);
                memcpy(vp, diff, 32);
            }
        }
    }
}

// ============================================================
// Pallas Fr iNTT: Inverse (Gentleman-Sande DIF)
// ============================================================

void pallas_fr_intt(uint64_t *data, int logN) {
    if (logN <= 0) return;
    int n = 1 << logN;

    pfr_ensure_twiddles(logN);
    const uint64_t *tw = pfr_cached[logN].inv;

    for (int si = 0; si < logN; si++) {
        int s = logN - 1 - si;
        int halfBlock = 1 << s;
        int blockSize = halfBlock << 1;
        int nBlocks = n / blockSize;
        int twOffset = halfBlock - 1;

        for (int bk = 0; bk < nBlocks; bk++) {
            int base = bk * blockSize;
            {
                uint64_t *ap = &data[base * 4];
                uint64_t *bp = &data[(base + halfBlock) * 4];
                uint64_t a[4], b[4];
                memcpy(a, ap, 32);
                memcpy(b, bp, 32);
                uint64_t sum[4], diff[4];
                pfr_add(a, b, sum);
                pfr_sub(a, b, diff);
                memcpy(ap, sum, 32);
                memcpy(bp, diff, 32);
            }
            for (int j = 1; j < halfBlock; j++) {
                uint64_t *ap = &data[(base + j) * 4];
                uint64_t *bp = &data[(base + j + halfBlock) * 4];
                const uint64_t *twj = &tw[(twOffset + j) * 4];
                uint64_t a[4], b[4];
                memcpy(a, ap, 32);
                memcpy(b, bp, 32);
                uint64_t sum[4], diff[4], prod[4];
                pfr_add(a, b, sum);
                pfr_sub(a, b, diff);
                pfr_mul(diff, twj, prod);
                memcpy(ap, sum, 32);
                memcpy(bp, prod, 32);
            }
        }
    }

    pasta_bit_reverse_permute(data, logN);

    // Scale by 1/n
    uint64_t n_plain[4] = {(uint64_t)n, 0, 0, 0};
    uint64_t n_mont[4];
    pfr_mul(n_plain, PALLAS_FR_R2, n_mont);
    uint64_t n_inv[4];
    pfr_inv(n_mont, n_inv);
    for (int i = 0; i < n; i++) {
        pfr_mul(&data[i * 4], n_inv, &data[i * 4]);
    }
}

// ============================================================
// Vesta Fr NTT: Forward (Cooley-Tukey DIT)
// ============================================================

void vesta_fr_ntt(uint64_t *data, int logN) {
    if (logN <= 0) return;
    int n = 1 << logN;

    vfr_ensure_twiddles(logN);
    const uint64_t *tw = vfr_cached[logN].fwd;

    pasta_bit_reverse_permute(data, logN);

    for (int s = 0; s < logN; s++) {
        int halfBlock = 1 << s;
        int blockSize = halfBlock << 1;
        int nBlocks = n / blockSize;
        int twOffset = halfBlock - 1;

        for (int bk = 0; bk < nBlocks; bk++) {
            int base = bk * blockSize;
            {
                uint64_t *u = &data[base * 4];
                uint64_t *vp = &data[(base + halfBlock) * 4];
                uint64_t sum[4], diff[4];
                vfr_add(u, vp, sum);
                vfr_sub(u, vp, diff);
                memcpy(u, sum, 32);
                memcpy(vp, diff, 32);
            }
            for (int j = 1; j < halfBlock; j++) {
                uint64_t *u = &data[(base + j) * 4];
                uint64_t *vp = &data[(base + j + halfBlock) * 4];
                const uint64_t *twj = &tw[(twOffset + j) * 4];
                uint64_t v[4];
                vfr_mul(twj, vp, v);
                uint64_t sum[4], diff[4];
                vfr_add(u, v, sum);
                vfr_sub(u, v, diff);
                memcpy(u, sum, 32);
                memcpy(vp, diff, 32);
            }
        }
    }
}

// ============================================================
// Vesta Fr iNTT: Inverse (Gentleman-Sande DIF)
// ============================================================

void vesta_fr_intt(uint64_t *data, int logN) {
    if (logN <= 0) return;
    int n = 1 << logN;

    vfr_ensure_twiddles(logN);
    const uint64_t *tw = vfr_cached[logN].inv;

    for (int si = 0; si < logN; si++) {
        int s = logN - 1 - si;
        int halfBlock = 1 << s;
        int blockSize = halfBlock << 1;
        int nBlocks = n / blockSize;
        int twOffset = halfBlock - 1;

        for (int bk = 0; bk < nBlocks; bk++) {
            int base = bk * blockSize;
            {
                uint64_t *ap = &data[base * 4];
                uint64_t *bp = &data[(base + halfBlock) * 4];
                uint64_t a[4], b[4];
                memcpy(a, ap, 32);
                memcpy(b, bp, 32);
                uint64_t sum[4], diff[4];
                vfr_add(a, b, sum);
                vfr_sub(a, b, diff);
                memcpy(ap, sum, 32);
                memcpy(bp, diff, 32);
            }
            for (int j = 1; j < halfBlock; j++) {
                uint64_t *ap = &data[(base + j) * 4];
                uint64_t *bp = &data[(base + j + halfBlock) * 4];
                const uint64_t *twj = &tw[(twOffset + j) * 4];
                uint64_t a[4], b[4];
                memcpy(a, ap, 32);
                memcpy(b, bp, 32);
                uint64_t sum[4], diff[4], prod[4];
                vfr_add(a, b, sum);
                vfr_sub(a, b, diff);
                vfr_mul(diff, twj, prod);
                memcpy(ap, sum, 32);
                memcpy(bp, prod, 32);
            }
        }
    }

    pasta_bit_reverse_permute(data, logN);

    // Scale by 1/n
    uint64_t n_plain[4] = {(uint64_t)n, 0, 0, 0};
    uint64_t n_mont[4];
    vfr_mul(n_plain, VESTA_FR_R2, n_mont);
    uint64_t n_inv[4];
    vfr_inv(n_mont, n_inv);
    for (int i = 0; i < n; i++) {
        vfr_mul(&data[i * 4], n_inv, &data[i * 4]);
    }
}
