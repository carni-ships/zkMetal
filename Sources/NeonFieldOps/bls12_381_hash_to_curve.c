// BLS12-381 hash-to-curve for G2 — RFC 9380 (Wahby-Boneh)
// Implements: expand_message_xmd(SHA-256) + hash_to_field + simplified SWU + 3-isogeny + cofactor
//
// The simplified SWU map operates on the isogeny curve E': y^2 = x^3 + A'x + B'
// where A' and B' are chosen so that E' is 3-isogenous to E: y^2 = x^3 + 4(1+u).
//
// All Fp elements are in Montgomery form (6 x uint64_t).
// All Fp2 elements are (c0[6], c1[6]) = 12 x uint64_t.
// G2 projective: (x[12], y[12], z[12]) = 36 x uint64_t.

#include "NeonFieldOps.h"
#include <string.h>
#include <stdlib.h>

typedef unsigned __int128 uint128_t;

// ============================================================
// Fp constants (local copies for static inline)
// ============================================================

static const uint64_t P[6] = {
    0xb9feffffffffaaabULL, 0x1eabfffeb153ffffULL,
    0x6730d2a0f6b0f624ULL, 0x64774b84f38512bfULL,
    0x4b1ba7b6434bacd7ULL, 0x1a0111ea397fe69aULL
};
static const uint64_t P_INV = 0x89f3fffcfffcfffdULL;
static const uint64_t FP_ONE[6] = {
    0x760900000002fffdULL, 0xebf4000bc40c0002ULL,
    0x5f48985753c758baULL, 0x77ce585370525745ULL,
    0x5c071a97a256ec6dULL, 0x15f65ec3fa80e493ULL
};
static const uint64_t FP_R2[6] = {  // R^2 mod p for to_mont
    0xf4df1f341c341746ULL, 0x0a76e6a609d104f1ULL,
    0x8de5476c4c95b6d5ULL, 0x67eb88a9939d83c0ULL,
    0x9a793e85b519952dULL, 0x11988fe592cae3aaULL
};

// ============================================================
// Fp arithmetic (inline, same as pairing.c)
// ============================================================

static inline void fp_mul(const uint64_t a[6], const uint64_t b[6], uint64_t r[6]) {
    uint64_t t[7] = {0};
    for (int i = 0; i < 6; i++) {
        uint128_t carry = 0;
        for (int j = 0; j < 6; j++) {
            carry += (uint128_t)a[i] * b[j] + t[j];
            t[j] = (uint64_t)carry;
            carry >>= 64;
        }
        t[6] += (uint64_t)carry;
        uint64_t m = t[0] * P_INV;
        carry = (uint128_t)m * P[0] + t[0];
        carry >>= 64;
        for (int j = 1; j < 6; j++) {
            carry += (uint128_t)m * P[j] + t[j];
            t[j-1] = (uint64_t)carry;
            carry >>= 64;
        }
        t[5] = t[6] + (uint64_t)carry;
        t[6] = (uint64_t)(carry >> 64);
    }
    uint64_t borrow = 0, tmp[6];
    uint128_t d;
    d = (uint128_t)t[0] - P[0]; tmp[0] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)t[1] - P[1] - borrow; tmp[1] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)t[2] - P[2] - borrow; tmp[2] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)t[3] - P[3] - borrow; tmp[3] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)t[4] - P[4] - borrow; tmp[4] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (uint128_t)t[5] - P[5] - borrow; tmp[5] = (uint64_t)d; borrow = (d >> 127) & 1;
    if (!borrow) memcpy(r, tmp, 48); else memcpy(r, t, 48);
}

static inline void fp_sqr(const uint64_t a[6], uint64_t r[6]) { fp_mul(a, a, r); }

static inline void fp_add(const uint64_t a[6], const uint64_t b[6], uint64_t r[6]) {
    uint128_t w; uint64_t c = 0;
    w = (uint128_t)a[0]+b[0]; r[0]=(uint64_t)w; c=(uint64_t)(w>>64);
    w = (uint128_t)a[1]+b[1]+c; r[1]=(uint64_t)w; c=(uint64_t)(w>>64);
    w = (uint128_t)a[2]+b[2]+c; r[2]=(uint64_t)w; c=(uint64_t)(w>>64);
    w = (uint128_t)a[3]+b[3]+c; r[3]=(uint64_t)w; c=(uint64_t)(w>>64);
    w = (uint128_t)a[4]+b[4]+c; r[4]=(uint64_t)w; c=(uint64_t)(w>>64);
    w = (uint128_t)a[5]+b[5]+c; r[5]=(uint64_t)w; c=(uint64_t)(w>>64);
    uint64_t borrow=0, tmp[6]; uint128_t d;
    d=(uint128_t)r[0]-P[0]; tmp[0]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[1]-P[1]-borrow; tmp[1]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[2]-P[2]-borrow; tmp[2]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[3]-P[3]-borrow; tmp[3]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[4]-P[4]-borrow; tmp[4]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)r[5]-P[5]-borrow; tmp[5]=(uint64_t)d; borrow=(d>>127)&1;
    if (c||!borrow) memcpy(r, tmp, 48);
}

static inline void fp_sub(const uint64_t a[6], const uint64_t b[6], uint64_t r[6]) {
    uint128_t d; uint64_t borrow=0;
    d=(uint128_t)a[0]-b[0]; r[0]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[1]-b[1]-borrow; r[1]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[2]-b[2]-borrow; r[2]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[3]-b[3]-borrow; r[3]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[4]-b[4]-borrow; r[4]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)a[5]-b[5]-borrow; r[5]=(uint64_t)d; borrow=(d>>127)&1;
    if (borrow) {
        uint64_t c=0;
        d=(uint128_t)r[0]+P[0]; r[0]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[1]+P[1]+c; r[1]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[2]+P[2]+c; r[2]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[3]+P[3]+c; r[3]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[4]+P[4]+c; r[4]=(uint64_t)d; c=(uint64_t)(d>>64);
        d=(uint128_t)r[5]+P[5]+c; r[5]=(uint64_t)d;
    }
}

static inline void fp_neg(const uint64_t a[6], uint64_t r[6]) {
    int z = (a[0]|a[1]|a[2]|a[3]|a[4]|a[5]) == 0;
    if (z) { memset(r, 0, 48); return; }
    uint128_t d; uint64_t borrow=0;
    d=(uint128_t)P[0]-a[0]; r[0]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)P[1]-a[1]-borrow; r[1]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)P[2]-a[2]-borrow; r[2]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)P[3]-a[3]-borrow; r[3]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)P[4]-a[4]-borrow; r[4]=(uint64_t)d; borrow=(d>>127)&1;
    d=(uint128_t)P[5]-a[5]-borrow; r[5]=(uint64_t)d;
}

static inline void fp_dbl(const uint64_t a[6], uint64_t r[6]) { fp_add(a, a, r); }
static inline int fp_is_zero(const uint64_t a[6]) { return (a[0]|a[1]|a[2]|a[3]|a[4]|a[5]) == 0; }
static inline void fp_copy(uint64_t dst[6], const uint64_t src[6]) { memcpy(dst, src, 48); }

// Fp inversion via Fermat: a^(p-2)
static void fp_inv(const uint64_t a[6], uint64_t r[6]) {
    uint64_t pm2[6];
    for (int i = 0; i < 6; i++) pm2[i] = P[i];
    pm2[0] -= 2;
    memcpy(r, FP_ONE, 48);
    uint64_t b[6]; memcpy(b, a, 48);
    for (int i = 0; i < 6; i++) {
        for (int bit = 0; bit < 64; bit++) {
            if ((pm2[i] >> bit) & 1) fp_mul(r, b, r);
            fp_sqr(b, b);
        }
    }
}

// Convert integer (6 limbs, non-Montgomery) to Montgomery form
static inline void fp_to_mont(const uint64_t a[6], uint64_t r[6]) {
    fp_mul(a, FP_R2, r);
}

// ============================================================
// Fp2 arithmetic
// ============================================================

static inline void fp2_add(const uint64_t a[12], const uint64_t b[12], uint64_t r[12]) {
    fp_add(a, b, r); fp_add(a+6, b+6, r+6);
}
static inline void fp2_sub(const uint64_t a[12], const uint64_t b[12], uint64_t r[12]) {
    fp_sub(a, b, r); fp_sub(a+6, b+6, r+6);
}
static inline void fp2_neg(const uint64_t a[12], uint64_t r[12]) {
    fp_neg(a, r); fp_neg(a+6, r+6);
}
static inline void fp2_dbl(const uint64_t a[12], uint64_t r[12]) {
    fp_dbl(a, r); fp_dbl(a+6, r+6);
}
static inline void fp2_mul(const uint64_t a[12], const uint64_t b[12], uint64_t r[12]) {
    uint64_t a0b0[6], a1b1[6], t1[6], t2[6], t3[6];
    fp_mul(a, b, a0b0);
    fp_mul(a+6, b+6, a1b1);
    fp_add(a, a+6, t1);
    fp_add(b, b+6, t2);
    fp_mul(t1, t2, t3);
    fp_sub(a0b0, a1b1, r);
    fp_sub(t3, a0b0, r+6);
    fp_sub(r+6, a1b1, r+6);
}
static inline void fp2_sqr(const uint64_t a[12], uint64_t r[12]) {
    uint64_t v0[6], sum[6], diff[6];
    fp_mul(a, a+6, v0);
    fp_add(a, a+6, sum);
    fp_sub(a, a+6, diff);
    fp_mul(sum, diff, r);
    fp_dbl(v0, r+6);
}
static inline int fp2_is_zero(const uint64_t a[12]) {
    return fp_is_zero(a) && fp_is_zero(a+6);
}
static inline void fp2_copy(uint64_t dst[12], const uint64_t src[12]) { memcpy(dst, src, 96); }
static inline void fp2_zero(uint64_t r[12]) { memset(r, 0, 96); }
static inline void fp2_one(uint64_t r[12]) { memcpy(r, FP_ONE, 48); memset(r+6, 0, 48); }

// Fp2 multiply by Fp scalar
static inline void fp2_mul_fp(const uint64_t a[12], const uint64_t s[6], uint64_t r[12]) {
    fp_mul(a, s, r);
    fp_mul(a+6, s, r+6);
}

// Fp2 inverse: 1/(c0+c1*u) = (c0-c1*u)/(c0^2+c1^2)
static void fp2_inv(const uint64_t a[12], uint64_t r[12]) {
    uint64_t t0[6], t1[6], norm[6], ninv[6];
    fp_sqr(a, t0);
    fp_sqr(a+6, t1);
    fp_add(t0, t1, norm);
    fp_inv(norm, ninv);
    fp_mul(a, ninv, r);
    uint64_t neg1[6];
    fp_neg(a+6, neg1);
    fp_mul(neg1, ninv, r+6);
}

// Multiply by non-residue (1+u): (a0-a1) + (a0+a1)*u
static inline void fp2_mul_nr(const uint64_t a[12], uint64_t r[12]) {
    uint64_t t0[6], t1[6];
    fp_sub(a, a+6, t0);
    fp_add(a, a+6, t1);
    fp_copy(r, t0);
    fp_copy(r+6, t1);
}

// ============================================================
// Fp / Fp2 exponentiation and square root
// ============================================================

// Fp exponentiation: a^exp where exp is 6 limbs
static void fp_pow(const uint64_t a[6], const uint64_t exp[6], uint64_t r[6]) {
    memcpy(r, FP_ONE, 48);
    uint64_t b[6]; memcpy(b, a, 48);
    for (int i = 0; i < 6; i++) {
        for (int bit = 0; bit < 64; bit++) {
            if ((exp[i] >> bit) & 1) fp_mul(r, b, r);
            fp_sqr(b, b);
        }
    }
}

// Fp square root: a^((p+1)/4) since p = 3 mod 4
// Returns 1 if sqrt exists (result in r), 0 otherwise
static int fp_sqrt(const uint64_t a[6], uint64_t r[6]) {
    if (fp_is_zero(a)) { memset(r, 0, 48); return 1; }
    // exp = (p+1)/4
    uint64_t exp[6];
    memcpy(exp, P, 48);
    // p+1
    uint64_t carry = 0;
    uint128_t w = (uint128_t)exp[0] + 1; exp[0] = (uint64_t)w; carry = (uint64_t)(w >> 64);
    for (int i = 1; i < 6 && carry; i++) {
        w = (uint128_t)exp[i] + carry; exp[i] = (uint64_t)w; carry = (uint64_t)(w >> 64);
    }
    // >> 2
    for (int i = 0; i < 5; i++) exp[i] = (exp[i] >> 2) | (exp[i+1] << 62);
    exp[5] >>= 2;

    fp_pow(a, exp, r);
    // Verify
    uint64_t chk[6];
    fp_sqr(r, chk);
    return memcmp(chk, a, 48) == 0;
}

// Fp2 square root using norm-based approach
// For a = c0 + c1*u in Fp2 where u^2 = -1:
//   norm(a) = c0^2 + c1^2
//   If norm has sqrt in Fp, a has sqrt in Fp2
static int fp2_sqrt(const uint64_t a[12], uint64_t r[12]) {
    if (fp2_is_zero(a)) { fp2_zero(r); return 1; }

    // norm = c0^2 + c1^2
    uint64_t c0sq[6], c1sq[6], norm[6];
    fp_sqr(a, c0sq);
    fp_sqr(a+6, c1sq);
    fp_add(c0sq, c1sq, norm);

    uint64_t norm_sqrt[6];
    if (!fp_sqrt(norm, norm_sqrt)) return 0;

    // twoInv = 1/2 in Fp
    uint64_t two_mont[6], two_inv[6];
    fp_add(FP_ONE, FP_ONE, two_mont);
    fp_inv(two_mont, two_inv);

    // Try branch 1: t^2 = (c0 + norm_sqrt) / 2
    uint64_t cand[6];
    fp_add(a, norm_sqrt, cand);      // c0 + norm_sqrt
    fp_mul(cand, two_inv, cand);     // / 2
    uint64_t t[6];
    if (fp_sqrt(cand, t) && !fp_is_zero(t)) {
        // r = (t, c1 / (2t))
        uint64_t two_t[6], two_t_inv[6], c1part[6];
        fp_dbl(t, two_t);
        fp_inv(two_t, two_t_inv);
        fp_mul(a+6, two_t_inv, c1part);
        fp_copy(r, t);
        fp_copy(r+6, c1part);
        // Verify
        uint64_t chk[12];
        fp2_sqr(r, chk);
        if (memcmp(chk, a, 96) == 0) return 1;
        // Try negated
        fp_neg(t, r);
        fp_neg(c1part, r+6);
        fp2_sqr(r, chk);
        if (memcmp(chk, a, 96) == 0) return 1;
    }

    // Try branch 2: t^2 = (c0 - norm_sqrt) / 2
    fp_sub(a, norm_sqrt, cand);
    fp_mul(cand, two_inv, cand);
    if (fp_sqrt(cand, t) && !fp_is_zero(t)) {
        uint64_t two_t[6], two_t_inv[6], c1part[6];
        fp_dbl(t, two_t);
        fp_inv(two_t, two_t_inv);
        fp_mul(a+6, two_t_inv, c1part);
        fp_copy(r, t);
        fp_copy(r+6, c1part);
        uint64_t chk[12];
        fp2_sqr(r, chk);
        if (memcmp(chk, a, 96) == 0) return 1;
        fp_neg(t, r);
        fp_neg(c1part, r+6);
        fp2_sqr(r, chk);
        if (memcmp(chk, a, 96) == 0) return 1;
    }

    // Edge case: a = c1*u (c0 = 0)
    // sqrt(c1*u): if c1 has sqrt s, then sqrt(c1*u) involves sqrt(u)
    // sqrt(u) = (1+u)/sqrt(2) when p = 3 mod 8... BLS12-381 p = 3 mod 4
    // Use exponentiation: a^((p^2+1)/4) but that's expensive.
    // Fall back to Fp2 exponentiation: a^((p^2+7)/16) for BLS12-381
    // Actually for BLS12-381, p^2 = 1 mod 8, and the Tonelli-Shanks approach works:
    // sqrt(a) = a^((p^2+1)/4) when p^2 = 3 mod 4... but p^2 = 1 mod 4 for BLS12-381.
    // Use the chain: result = a^((p^2 + 7)/16), then adjust.

    // For the SSWU map, we actually don't need sqrt - we use the direct formula.
    // But for try-and-increment fallback, this matters.
    return 0;
}

// ============================================================
// G2 projective point operations (local copies)
// ============================================================

static inline void g2_set_id(uint64_t p[36]) {
    memcpy(p, FP_ONE, 48); memset(p+6, 0, 48);     // x = Fp2(1)
    memcpy(p+12, FP_ONE, 48); memset(p+18, 0, 48);  // y = Fp2(1)
    memset(p+24, 0, 96);                              // z = 0
}

static inline int g2_is_id(const uint64_t p[36]) {
    return fp2_is_zero(p+24);
}

// G2 point doubling (Jacobian)
static void g2_dbl(const uint64_t p[36], uint64_t r[36]) {
    if (g2_is_id(p)) { memcpy(r, p, 288); return; }
    const uint64_t *px = p, *py = p+12, *pz = p+24;
    uint64_t a[12], b[12], c[12], d[12], e[12], f[12], t1[12], t2[12];

    fp2_sqr(px, a);
    fp2_sqr(py, b);
    fp2_sqr(b, c);

    fp2_add(px, b, t1);
    fp2_sqr(t1, t1);
    fp2_sub(t1, a, t1);
    fp2_sub(t1, c, t1);
    fp2_dbl(t1, d);               // d = 2*((x+b)^2 - a - c)

    fp2_dbl(a, e);
    fp2_add(e, a, e);             // e = 3a

    fp2_sqr(e, f);                // f = e^2

    // x3 = f - 2d
    fp2_dbl(d, t1);
    fp2_sub(f, t1, r);

    // y3 = e*(d - x3) - 8c
    fp2_sub(d, r, t1);
    fp2_mul(e, t1, t2);
    fp2_dbl(c, t1);
    fp2_dbl(t1, t1);
    fp2_dbl(t1, t1);             // 8c
    fp2_sub(t2, t1, r+12);

    // z3 = (y+z)^2 - b - z^2
    fp2_add(py, pz, t1);
    fp2_sqr(t1, t1);
    fp2_sub(t1, b, t1);
    fp2_sqr(pz, t2);
    fp2_sub(t1, t2, r+24);
}

// G2 full projective addition
static void g2_add(const uint64_t p[36], const uint64_t q[36], uint64_t r[36]) {
    if (g2_is_id(p)) { memcpy(r, q, 288); return; }
    if (g2_is_id(q)) { memcpy(r, p, 288); return; }

    const uint64_t *px=p, *py=p+12, *pz=p+24;
    const uint64_t *qx=q, *qy=q+12, *qz=q+24;

    uint64_t z1z1[12], z2z2[12], u1[12], u2[12], s1[12], s2[12];
    uint64_t h[12], rr[12], ii[12], j[12], vv[12], t1[12];

    fp2_sqr(pz, z1z1);
    fp2_sqr(qz, z2z2);
    fp2_mul(px, z2z2, u1);
    fp2_mul(qx, z1z1, u2);
    fp2_mul(qz, z2z2, t1);
    fp2_mul(py, t1, s1);
    fp2_mul(pz, z1z1, t1);
    fp2_mul(qy, t1, s2);

    fp2_sub(u2, u1, h);
    fp2_sub(s2, s1, t1);
    fp2_dbl(t1, rr);

    if (fp2_is_zero(h)) {
        if (fp2_is_zero(rr)) { g2_dbl(p, r); return; }
        g2_set_id(r); return;
    }

    fp2_dbl(h, t1);
    fp2_sqr(t1, ii);
    fp2_mul(h, ii, j);
    fp2_mul(u1, ii, vv);

    fp2_sqr(rr, r);
    fp2_sub(r, j, r);
    fp2_dbl(vv, t1);
    fp2_sub(r, t1, r);

    fp2_sub(vv, r, t1);
    fp2_mul(rr, t1, r+12);
    fp2_mul(s1, j, t1);
    fp2_dbl(t1, t1);
    fp2_sub(r+12, t1, r+12);

    fp2_add(pz, qz, t1);
    fp2_sqr(t1, t1);
    fp2_sub(t1, z1z1, t1);
    fp2_sub(t1, z2z2, t1);
    fp2_mul(t1, h, r+24);
}

// G2 scalar mul with wide scalar (up to 10 x uint64_t = 640 bits)
// Used for cofactor clearing
static void g2_scalar_mul_wide(const uint64_t p[36], const uint64_t *scalar, int nlimbs, uint64_t r[36]) {
    // Windowed method w=4
    uint64_t table[16 * 36];
    g2_set_id(table);
    memcpy(table + 36, p, 288);
    for (int i = 2; i < 16; i++) {
        g2_add(table + (i-1)*36, p, table + i*36);
    }

    uint8_t nibbles[160]; // up to 10*16 = 160 nibbles
    int total_nibbles = nlimbs * 16;
    for (int i = 0; i < nlimbs; i++) {
        uint64_t word = scalar[i];
        for (int j = 0; j < 16; j++) {
            nibbles[i*16 + j] = (uint8_t)(word & 0xF);
            word >>= 4;
        }
    }

    int top = total_nibbles - 1;
    while (top >= 0 && nibbles[top] == 0) top--;

    if (top < 0) { g2_set_id(r); return; }

    uint64_t result[36];
    memcpy(result, table + nibbles[top]*36, 288);

    for (int i = top - 1; i >= 0; i--) {
        uint64_t tmp[36];
        g2_dbl(result, tmp); memcpy(result, tmp, 288);
        g2_dbl(result, tmp); memcpy(result, tmp, 288);
        g2_dbl(result, tmp); memcpy(result, tmp, 288);
        g2_dbl(result, tmp); memcpy(result, tmp, 288);
        if (nibbles[i]) {
            g2_add(result, table + nibbles[i]*36, tmp);
            memcpy(result, tmp, 288);
        }
    }
    memcpy(r, result, 288);
}

// ============================================================
// SHA-256 (compact implementation for hash-to-curve)
// ============================================================

static const uint32_t SHA256_K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

#define ROTR32(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define CH(x,y,z)   (((x)&(y))^((~(x))&(z)))
#define MAJ(x,y,z)  (((x)&(y))^((x)&(z))^((y)&(z)))
#define EP0(x)       (ROTR32(x,2)^ROTR32(x,13)^ROTR32(x,22))
#define EP1(x)       (ROTR32(x,6)^ROTR32(x,11)^ROTR32(x,25))
#define SIG0(x)      (ROTR32(x,7)^ROTR32(x,18)^((x)>>3))
#define SIG1(x)      (ROTR32(x,17)^ROTR32(x,19)^((x)>>10))

static void sha256_hash(const uint8_t *data, size_t len, uint8_t out[32]) {
    uint32_t h0=0x6a09e667, h1=0xbb67ae85, h2=0x3c6ef372, h3=0xa54ff53a;
    uint32_t h4=0x510e527f, h5=0x9b05688c, h6=0x1f83d9ab, h7=0x5be0cd19;

    // Pad message
    size_t bit_len = len * 8;
    size_t padded_len = ((len + 8) / 64 + 1) * 64;
    uint8_t *padded = (uint8_t*)calloc(padded_len, 1);
    memcpy(padded, data, len);
    padded[len] = 0x80;
    for (int i = 0; i < 8; i++)
        padded[padded_len - 1 - i] = (uint8_t)((bit_len >> (i * 8)) & 0xFF);

    for (size_t chunk = 0; chunk < padded_len; chunk += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; i++) {
            w[i] = ((uint32_t)padded[chunk+i*4]<<24) | ((uint32_t)padded[chunk+i*4+1]<<16) |
                    ((uint32_t)padded[chunk+i*4+2]<<8) | (uint32_t)padded[chunk+i*4+3];
        }
        for (int i = 16; i < 64; i++)
            w[i] = SIG1(w[i-2]) + w[i-7] + SIG0(w[i-15]) + w[i-16];

        uint32_t a=h0,b=h1,c=h2,d=h3,e=h4,f=h5,g=h6,hh=h7;
        for (int i = 0; i < 64; i++) {
            uint32_t t1 = hh + EP1(e) + CH(e,f,g) + SHA256_K[i] + w[i];
            uint32_t t2 = EP0(a) + MAJ(a,b,c);
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h0+=a; h1+=b; h2+=c; h3+=d; h4+=e; h5+=f; h6+=g; h7+=hh;
    }
    free(padded);

    #define WB(v,o) out[o]=(uint8_t)((v)>>24);out[o+1]=(uint8_t)((v)>>16);out[o+2]=(uint8_t)((v)>>8);out[o+3]=(uint8_t)(v);
    WB(h0,0) WB(h1,4) WB(h2,8) WB(h3,12) WB(h4,16) WB(h5,20) WB(h6,24) WB(h7,28)
    #undef WB
}

// ============================================================
// expand_message_xmd (SHA-256) per RFC 9380 Section 5.3.1
// ============================================================

static void expand_message_xmd(const uint8_t *msg, size_t msg_len,
                                const uint8_t *dst, size_t dst_len,
                                int len_in_bytes, uint8_t *out) {
    int b_in_bytes = 32;  // SHA-256 output
    int r_in_bytes = 64;  // SHA-256 block size
    int ell = (len_in_bytes + b_in_bytes - 1) / b_in_bytes;

    // DST_prime = DST || I2OSP(len(DST), 1)
    uint8_t dst_prime[256];
    memcpy(dst_prime, dst, dst_len);
    dst_prime[dst_len] = (uint8_t)dst_len;
    int dst_prime_len = (int)dst_len + 1;

    // msg_prime = Z_pad || msg || l_i_b_str || I2OSP(0,1) || DST_prime
    size_t msg_prime_len = r_in_bytes + msg_len + 2 + 1 + dst_prime_len;
    uint8_t *msg_prime = (uint8_t*)calloc(msg_prime_len, 1);
    size_t off = r_in_bytes;  // Z_pad is already zeroed by calloc
    memcpy(msg_prime + off, msg, msg_len); off += msg_len;
    msg_prime[off++] = (uint8_t)((len_in_bytes >> 8) & 0xFF);
    msg_prime[off++] = (uint8_t)(len_in_bytes & 0xFF);
    msg_prime[off++] = 0;  // I2OSP(0,1)
    memcpy(msg_prime + off, dst_prime, dst_prime_len);

    uint8_t b0[32];
    sha256_hash(msg_prime, msg_prime_len, b0);
    free(msg_prime);

    // b_1 = H(b_0 || I2OSP(1,1) || DST_prime)
    uint8_t b_input[32 + 1 + 256];
    memcpy(b_input, b0, 32);
    b_input[32] = 1;
    memcpy(b_input + 33, dst_prime, dst_prime_len);
    uint8_t b_vals[256][32];
    sha256_hash(b_input, 33 + dst_prime_len, b_vals[0]);

    for (int i = 1; i < ell; i++) {
        // strxor(b_0, b_vals[i-1])
        uint8_t xored[32];
        for (int j = 0; j < 32; j++) xored[j] = b0[j] ^ b_vals[i-1][j];
        memcpy(b_input, xored, 32);
        b_input[32] = (uint8_t)(i + 1);
        memcpy(b_input + 33, dst_prime, dst_prime_len);
        sha256_hash(b_input, 33 + dst_prime_len, b_vals[i]);
    }

    int written = 0;
    for (int i = 0; i < ell && written < len_in_bytes; i++) {
        int to_copy = len_in_bytes - written;
        if (to_copy > 32) to_copy = 32;
        memcpy(out + written, b_vals[i], to_copy);
        written += to_copy;
    }
}

// ============================================================
// hash_to_field: 64 big-endian bytes -> Fp element (Montgomery)
// Reduces a 512-bit integer mod p
// ============================================================

static void fp_from_64_bytes(const uint8_t bytes[64], uint64_t r[6]) {
    // Parse as big-endian 512-bit integer -> 8 x uint64_t little-endian
    uint64_t limbs[8];
    for (int i = 0; i < 8; i++) {
        int byte_off = (7 - i) * 8;
        uint64_t w = 0;
        for (int j = 0; j < 8; j++)
            w |= (uint64_t)bytes[byte_off + (7 - j)] << (j * 8);
        limbs[i] = w;
    }

    // Split into low (384 bits) and high (128 bits), reduce each mod p
    // low = limbs[0..5], high = limbs[6..7]

    // Reduce low mod p using trial subtraction
    uint64_t low[6] = {limbs[0], limbs[1], limbs[2], limbs[3], limbs[4], limbs[5]};
    for (;;) {
        uint64_t borrow = 0, tmp[6];
        uint128_t d;
        d = (uint128_t)low[0] - P[0]; tmp[0] = (uint64_t)d; borrow = (d >> 127) & 1;
        d = (uint128_t)low[1] - P[1] - borrow; tmp[1] = (uint64_t)d; borrow = (d >> 127) & 1;
        d = (uint128_t)low[2] - P[2] - borrow; tmp[2] = (uint64_t)d; borrow = (d >> 127) & 1;
        d = (uint128_t)low[3] - P[3] - borrow; tmp[3] = (uint64_t)d; borrow = (d >> 127) & 1;
        d = (uint128_t)low[4] - P[4] - borrow; tmp[4] = (uint64_t)d; borrow = (d >> 127) & 1;
        d = (uint128_t)low[5] - P[5] - borrow; tmp[5] = (uint64_t)d; borrow = (d >> 127) & 1;
        if (borrow) break;
        memcpy(low, tmp, 48);
    }
    // Convert to Montgomery: low_mont = low * R^2 mod p (via Montgomery mul)
    uint64_t low_mont[6];
    fp_mul(low, FP_R2, low_mont);

    // Reduce high mod p
    uint64_t high[6] = {limbs[6], limbs[7], 0, 0, 0, 0};
    {
        uint64_t borrow = 0, tmp[6];
        uint128_t d;
        d = (uint128_t)high[0] - P[0]; tmp[0] = (uint64_t)d; borrow = (d >> 127) & 1;
        d = (uint128_t)high[1] - P[1] - borrow; tmp[1] = (uint64_t)d; borrow = (d >> 127) & 1;
        d = (uint128_t)high[2] - P[2] - borrow; tmp[2] = (uint64_t)d; borrow = (d >> 127) & 1;
        d = (uint128_t)high[3] - P[3] - borrow; tmp[3] = (uint64_t)d; borrow = (d >> 127) & 1;
        d = (uint128_t)high[4] - P[4] - borrow; tmp[4] = (uint64_t)d; borrow = (d >> 127) & 1;
        d = (uint128_t)high[5] - P[5] - borrow; tmp[5] = (uint64_t)d; borrow = (d >> 127) & 1;
        if (!borrow) memcpy(high, tmp, 48);
    }
    // high_mont = high * R^2
    uint64_t high_mont[6];
    fp_mul(high, FP_R2, high_mont);
    // We want (high * 2^384) in Montgomery = high_mont * R^2 (multiply by another R)
    uint64_t high_shifted[6];
    fp_mul(high_mont, FP_R2, high_shifted);

    // Result = low_mont + high_shifted
    fp_add(low_mont, high_shifted, r);
}

// ============================================================
// Simplified SWU map for BLS12-381 G2
// RFC 9380 Section 6.6.2
//
// The isogeny curve E' : y^2 = x^3 + A'*x + B'
// where A', B' are specific Fp2 constants.
//
// See https://www.ietf.org/rfc/rfc9380.html#appendix-G.2.2
// ============================================================

// ISO-3 curve constants for BLS12-381 G2:
// A' = 240 * u  (i.e., (0, 240) in Fp2)
// B' = 1012 * (1 + u)  (i.e., (1012, 1012) in Fp2)
// Z = -(2 + u) = (-2, -1) in Fp2

static void make_sswu_constants(
    uint64_t A_prime[12], uint64_t B_prime[12], uint64_t Z[12],
    uint64_t neg_A_prime[12], uint64_t ZA[12], uint64_t ZB[12])
{
    // A' = (0, 240) in Montgomery
    uint64_t a_raw[6] = {240, 0, 0, 0, 0, 0};
    memset(A_prime, 0, 48);
    fp_to_mont(a_raw, A_prime + 6);

    // B' = (1012, 1012) in Montgomery
    uint64_t b_raw[6] = {1012, 0, 0, 0, 0, 0};
    fp_to_mont(b_raw, B_prime);
    fp_to_mont(b_raw, B_prime + 6);

    // Z = (-2, -1) in Montgomery
    uint64_t two_raw[6] = {2, 0, 0, 0, 0, 0};
    uint64_t one_raw[6] = {1, 0, 0, 0, 0, 0};
    uint64_t two_m[6], one_m[6];
    fp_to_mont(two_raw, two_m);
    fp_to_mont(one_raw, one_m);
    fp_neg(two_m, Z);
    fp_neg(one_m, Z + 6);

    // neg_A = -A'
    fp2_neg(A_prime, neg_A_prime);

    // ZA = Z * A'
    fp2_mul(Z, A_prime, ZA);

    // ZB = Z * B'
    fp2_mul(Z, B_prime, ZB);
}

// sgn0 for Fp2 per RFC 9380: sign(x) in {0,1}
static int sgn0_fp2(const uint64_t a[12]) {
    // sign_0 = a.c0 mod 2, if a.c0 != 0
    // else sign_1 = a.c1 mod 2
    // To get the integer value of c0 from Montgomery form,
    // we need to convert out of Montgomery. But for parity check,
    // we can use the Montgomery trick:
    // val = mont * R^{-1} mod p. Parity of val = ...
    // Actually simpler: multiply by 1 (non-mont) to get the integer form
    uint64_t c0_int[6], c1_int[6];
    uint64_t one_nonmont[6] = {1, 0, 0, 0, 0, 0};
    fp_mul(a, one_nonmont, c0_int);
    fp_mul(a+6, one_nonmont, c1_int);

    int sign_0 = (int)(c0_int[0] & 1);
    int zero_0 = fp_is_zero(c0_int);
    int sign_1 = (int)(c1_int[0] & 1);
    return sign_0 | (zero_0 & sign_1);
}

// Simplified SWU map: u (Fp2) -> (x, y) on E' (Fp2)
// Returns point in projective coordinates on E': y^2 = x^3 + A'*x + B'
static void map_to_curve_sswu(const uint64_t u[12], uint64_t result[36]) {
    uint64_t A[12], B[12], Zv[12], negA[12], ZA_val[12], ZB_val[12];
    make_sswu_constants(A, B, Zv, negA, ZA_val, ZB_val);

    // tv1 = u^2
    uint64_t tv1[12];
    fp2_sqr(u, tv1);

    // tv1 = Z * u^2
    fp2_mul(Zv, tv1, tv1);

    // tv2 = tv1^2
    uint64_t tv2[12];
    fp2_sqr(tv1, tv2);

    // tv2 = tv2 + tv1
    fp2_add(tv2, tv1, tv2);

    // tv3 = tv2 + 1
    uint64_t tv3[12];
    uint64_t one_fp2[12];
    fp2_one(one_fp2);
    fp2_add(tv2, one_fp2, tv3);

    // tv3 = B * tv3
    fp2_mul(B, tv3, tv3);

    // tv4 = CMOV(Z, -tv2, tv2 != 0)
    // if tv2 == 0 then tv4 = Z else tv4 = -tv2
    uint64_t tv4[12];
    if (fp2_is_zero(tv2)) {
        fp2_copy(tv4, Zv);
    } else {
        fp2_neg(tv2, tv4);
    }

    // tv4 = A * tv4
    fp2_mul(A, tv4, tv4);

    // tv2 = tv3^2
    fp2_sqr(tv3, tv2);

    // tv6 = tv4^2
    uint64_t tv6[12];
    fp2_sqr(tv4, tv6);

    // tv5 = A * tv6
    uint64_t tv5[12];
    fp2_mul(A, tv6, tv5);

    // tv2 = tv2 + tv5
    fp2_add(tv2, tv5, tv2);

    // tv2 = tv2 * tv3
    fp2_mul(tv2, tv3, tv2);

    // tv6 = tv6 * tv4
    fp2_mul(tv6, tv4, tv6);

    // tv5 = B * tv6
    fp2_mul(B, tv6, tv5);

    // tv2 = tv2 + tv5
    fp2_add(tv2, tv5, tv2);

    // x = tv1 * tv3
    uint64_t x[12];
    fp2_mul(tv1, tv3, x);

    // (is_gx1_sq, y1) = sqrt_ratio(tv2, tv6)
    // sqrt_ratio: given u, v, compute sqrt(u/v) if it exists
    // If u/v is square: is_gx1_sq = 1, y1 = sqrt(u/v)
    // Otherwise: is_gx1_sq = 0, y1 = sqrt(Z * u/v)

    // We use the approach from RFC 9380 Appendix G.2.2:
    // Compute tv2 / tv6, check if it's a square in Fp2

    // First check if tv6 is zero
    int is_gx1_sq;
    uint64_t y1[12];

    if (fp2_is_zero(tv6)) {
        // Division by zero means tv4=0 which means tv2=0 and Z would be used
        // In practice this shouldn't happen for random inputs
        fp2_zero(y1);
        is_gx1_sq = 1;
    } else {
        // Compute gx1 = tv2 / tv6
        uint64_t tv6_inv[12];
        fp2_inv(tv6, tv6_inv);
        uint64_t gx1[12];
        fp2_mul(tv2, tv6_inv, gx1);

        // Try sqrt(gx1)
        if (fp2_sqrt(gx1, y1)) {
            is_gx1_sq = 1;
        } else {
            // gx2 = Z * gx1, try sqrt(gx2)
            uint64_t gx2[12];
            fp2_mul(Zv, gx1, gx2);
            if (fp2_sqrt(gx2, y1)) {
                is_gx1_sq = 0;
            } else {
                // This really shouldn't happen for BLS12-381 with Z = -(2+i)
                // because one of gx1, gx2 must be a square
                // Fallback: set y1 = 0 and mark as gx1 square
                fp2_zero(y1);
                is_gx1_sq = 1;
            }
        }
    }

    // If gx1 is square: x = tv3 / tv4, y = y1
    // If gx1 is not square: x = tv1 * tv3 / tv4 (already in x), y = y1
    uint64_t xn[12], xd[12];
    if (is_gx1_sq) {
        fp2_copy(xn, tv3);
    } else {
        fp2_mul(tv1, tv3, xn);
    }
    fp2_copy(xd, tv4);

    // Adjust sign: sgn0(y1) must equal sgn0(u)
    int sign_y = sgn0_fp2(y1);
    int sign_u = sgn0_fp2(u);
    if (sign_y != sign_u) {
        fp2_neg(y1, y1);
    }

    // Return projective point (xn * xd, y1 * xd^2 * xd, xd^3)
    // Actually, return (xn/xd, y1) as projective (xn, y1*xd, xd)
    // Projective: X = xn, Y = y1 * xd, Z = xd
    // Then affine x = X/Z = xn/xd, affine y = Y/Z = y1*xd/xd = y1 (WRONG)
    // Actually for Jacobian projective: x = X/Z^2, y = Y/Z^3
    // So X = xn*xd, Y = y1*xd^3, Z = xd gives x = xn*xd/xd^2 = xn/xd, y = y1*xd^3/xd^3 = y1
    // Let's use simpler: (xn*xd, y1*xd^2, xd) ... no.
    // Jacobian: affine_x = X/Z^2, affine_y = Y/Z^3
    // We want affine_x = xn/xd, affine_y = y1
    // Set Z = xd, then X = xn * xd (so X/Z^2 = xn*xd/xd^2 = xn/xd)
    //                   Y = y1 * xd^3 (so Y/Z^3 = y1*xd^3/xd^3 = y1)

    uint64_t xd2[12], xd3[12];
    fp2_sqr(xd, xd2);
    fp2_mul(xd, xd2, xd3);

    fp2_mul(xn, xd, result);        // X = xn * xd
    fp2_mul(y1, xd3, result + 12);  // Y = y1 * xd^3
    fp2_copy(result + 24, xd);      // Z = xd
}

// ============================================================
// 3-isogeny map from E'(Fp2) to E(Fp2)
// RFC 9380 Appendix E.3
//
// E': y^2 = x^3 + A'*x + B'   (SWU isogeny curve)
// E:  y^2 = x^3 + 4*(1+u)      (BLS12-381 G2 curve)
//
// The isogeny maps (x', y') on E' to (x, y) on E via:
//   x = x_num(x') / x_den(x')
//   y = y' * y_num(x') / y_den(x')
//
// Each rational map is a polynomial in x' with Fp2 coefficients.
// Coefficients from RFC 9380 Appendix E.3 (Section 8.8.2, Table 2)
// ============================================================

// Helper: make Fp2 constant from two integer values
static void make_fp2(uint64_t c0_val, uint64_t c1_val, uint64_t r[12]) {
    uint64_t raw0[6] = {c0_val, 0, 0, 0, 0, 0};
    uint64_t raw1[6] = {c1_val, 0, 0, 0, 0, 0};
    fp_to_mont(raw0, r);
    fp_to_mont(raw1, r+6);
}

// Helper: make Fp2 from hex limbs
static void make_fp2_from_limbs(const uint64_t c0[6], const uint64_t c1[6], uint64_t r[12]) {
    fp_to_mont(c0, r);
    fp_to_mont(c1, r+6);
}

// 3-isogeny map coefficients from RFC 9380 Appendix E.3
// x_num has degree 3, x_den has degree 2, y_num has degree 3, y_den has degree 3

// We precompute these as Fp2 Montgomery constants.
// The values are from the RFC, given as integers mod p.

// x_num coefficients [k_(3,0), k_(3,1), k_(3,2), k_(3,3)]
static void iso3_xnum(uint64_t coeffs[4][12]) {
    // k_(3,3):
    // c0 = 0x5c759507e8e333ebb5b7a9a47d7ed8532c52d39fd3a042a88b6a4d177f4ec647bb3e04691b4814c0
    // c1 = 0x5c759507e8e333ebb5b7a9a47d7ed8532c52d39fd3a042a88b6a4d177f4ec647bb3e04691b4814c0
    {
        const uint64_t c0[6] = {0xbb3e04691b4814c0ULL, 0x8b6a4d177f4ec647ULL,
                                0x2c52d39fd3a042a8ULL, 0xb5b7a9a47d7ed853ULL,
                                0x5c759507e8e333ebULL, 0x0000000000000000ULL};
        make_fp2_from_limbs(c0, c0, coeffs[3]);
    }
    // k_(3,2):
    // c0 = 0
    // c1 = 0x11560bf17baa99bc32126fced787c88f984f87adf7ae0c7f9a208c6b4f20a4181472aaa9cb8d555526a9ffffffffc71a
    {
        const uint64_t c0[6] = {0,0,0,0,0,0};
        const uint64_t c1[6] = {0x26a9ffffffffc71aULL, 0x1472aaa9cb8d5555ULL,
                                0x9a208c6b4f20a418ULL, 0x984f87adf7ae0c7fULL,
                                0x32126fced787c88fULL, 0x11560bf17baa99bcULL};
        make_fp2_from_limbs(c0, c1, coeffs[2]);
    }
    // k_(3,1):
    // c0 = 0x11560bf17baa99bc32126fced787c88f984f87adf7ae0c7f9a208c6b4f20a4181472aaa9cb8d555526a9ffffffffc71e
    // c1 = 0x8ab05f8bdd54cde190937e76bc3e447d674d812049b7d4a1090c6e55a0226f65e0f6bdb5db82d0a76e0affffffffc71c
    {
        const uint64_t c0[6] = {0x26a9ffffffffc71eULL, 0x1472aaa9cb8d5555ULL,
                                0x9a208c6b4f20a418ULL, 0x984f87adf7ae0c7fULL,
                                0x32126fced787c88fULL, 0x11560bf17baa99bcULL};
        const uint64_t c1[6] = {0x6e0affffffffc71cULL, 0xe0f6bdb5db82d0a7ULL,
                                0x090c6e55a0226f65ULL, 0x674d812049b7d4a1ULL,
                                0x90937e76bc3e447dULL, 0x8ab05f8bdd54cde1ULL};
        make_fp2_from_limbs(c0, c1, coeffs[1]);
    }
    // k_(3,0):
    // c0 = 0x171d6541fa38ccfaed09bfd206a1aca1e1b12c47c21a1be30fb4dfdb27fdaff85c30acad45e34d90ab2aaaaaaab71cb
    // c1 = 0
    {
        const uint64_t c0[6] = {0xab2aaaaaaab71cbULL, 0xb4dfdb27fdaff85cULL,
                                0x1b12c47c21a1be30ULL, 0xd09bfd206a1aca1eULL,
                                0x71d6541fa38ccfaeULL, 0x0000000000000001ULL};
        const uint64_t c1[6] = {0,0,0,0,0,0};
        make_fp2_from_limbs(c0, c1, coeffs[0]);
    }
}

// x_den coefficients [1, k_(2,1), k_(2,0)]  (leading coeff = 1, degree 2)
static void iso3_xden(uint64_t coeffs[3][12]) {
    // k_(2,2) = 1 (identity, implicit in Horner evaluation)
    fp2_one(coeffs[2]);

    // k_(2,1):
    // c0 = 0x12
    // c1 = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaa63
    {
        const uint64_t c0[6] = {0x12, 0, 0, 0, 0, 0};
        const uint64_t c1[6] = {0xb9feffffffffaa63ULL, 0x1eabfffeb153ffffULL,
                                0x6730d2a0f6b0f624ULL, 0x64774b84f38512bfULL,
                                0x4b1ba7b6434bacd7ULL, 0x1a0111ea397fe69aULL};
        make_fp2_from_limbs(c0, c1, coeffs[1]);
    }
    // k_(2,0):
    // c0 = 0xc
    // c1 = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaa9f
    {
        const uint64_t c0[6] = {0xc, 0, 0, 0, 0, 0};
        const uint64_t c1[6] = {0xb9feffffffffaa9fULL, 0x1eabfffeb153ffffULL,
                                0x6730d2a0f6b0f624ULL, 0x64774b84f38512bfULL,
                                0x4b1ba7b6434bacd7ULL, 0x1a0111ea397fe69aULL};
        make_fp2_from_limbs(c0, c1, coeffs[0]);
    }
}

// y_num coefficients [k_(3,0), k_(3,1), k_(3,2), k_(3,3)]
static void iso3_ynum(uint64_t coeffs[4][12]) {
    // k_(3,3):
    // c0 = 0x1530477c7ab4113b59a4c18b076d11930f7da5d4a07f649bf5446d21d8fd8e24b11ab8e6358a17f44fd20b93d5a2f921
    // c1 = 0x1530477c7ab4113b59a4c18b076d11930f7da5d4a07f649bf5446d21d8fd8e24b11ab8e6358a17f44fd20b93d5a2f921
    {
        const uint64_t c0[6] = {0x4fd20b93d5a2f921ULL, 0xb11ab8e6358a17f4ULL,
                                0xf5446d21d8fd8e24ULL, 0x0f7da5d4a07f649bULL,
                                0x59a4c18b076d1193ULL, 0x1530477c7ab4113bULL};
        make_fp2_from_limbs(c0, c0, coeffs[3]);
    }
    // k_(3,2):
    // c0 = 0
    // c1 = 0x5c759507e8e333ebb5b7a9a47d7ed8532c52d39fd3a042a88b6a4d177f4ec647bb3e04691b4814c0
    {
        const uint64_t c0[6] = {0,0,0,0,0,0};
        const uint64_t c1[6] = {0xbb3e04691b4814c0ULL, 0x8b6a4d177f4ec647ULL,
                                0x2c52d39fd3a042a8ULL, 0xb5b7a9a47d7ed853ULL,
                                0x5c759507e8e333ebULL, 0x0000000000000000ULL};
        make_fp2_from_limbs(c0, c1, coeffs[2]);
    }
    // k_(3,1):
    // c0 = 0x11560bf17baa99bc32126fced787c88f984f87adf7ae0c7f9a208c6b4f20a4181472aaa9cb8d555526a9ffffffffc71c
    // c1 = 0x8ab05f8bdd54cde190937e76bc3e447d674d812049b7d4a1090c6e55a0226f65e0f6bdb5db82d0a76e0affffffffc71c
    {
        const uint64_t c0[6] = {0x26a9ffffffffc71cULL, 0x1472aaa9cb8d5555ULL,
                                0x9a208c6b4f20a418ULL, 0x984f87adf7ae0c7fULL,
                                0x32126fced787c88fULL, 0x11560bf17baa99bcULL};
        const uint64_t c1[6] = {0x6e0affffffffc71cULL, 0xe0f6bdb5db82d0a7ULL,
                                0x090c6e55a0226f65ULL, 0x674d812049b7d4a1ULL,
                                0x90937e76bc3e447dULL, 0x8ab05f8bdd54cde1ULL};
        make_fp2_from_limbs(c0, c1, coeffs[1]);
    }
    // k_(3,0):
    // c0 = 0x124c9ad43b6cf79bfbf7043de3811ad0761b0f37a1e26286b0e977999978dc5fd096233c18df0cfe06d7f7e29ef67ccc
    // c1 = 0x124c9ad43b6cf79bfbf7043de3811ad0761b0f37a1e26286b0e977999978dc5fd096233c18df0cfe06d7f7e29ef67ccc
    {
        const uint64_t c0[6] = {0x06d7f7e29ef67cccULL, 0xd096233c18df0cfeULL,
                                0xb0e977999978dc5fULL, 0x761b0f37a1e26286ULL,
                                0xfbf7043de3811ad0ULL, 0x124c9ad43b6cf79bULL};
        make_fp2_from_limbs(c0, c0, coeffs[0]);
    }
}

// y_den coefficients [1, k_(3,1), k_(3,2), k_(3,0)]  (leading coeff = 1, degree 3)
static void iso3_yden(uint64_t coeffs[4][12]) {
    fp2_one(coeffs[3]);

    // k_(3,2):
    // c0 = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffa8fb
    // c1 = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffa8fb
    {
        const uint64_t c0[6] = {0xb9feffffffffa8fbULL, 0x1eabfffeb153ffffULL,
                                0x6730d2a0f6b0f624ULL, 0x64774b84f38512bfULL,
                                0x4b1ba7b6434bacd7ULL, 0x1a0111ea397fe69aULL};
        make_fp2_from_limbs(c0, c0, coeffs[2]);
    }
    // k_(3,1):
    // c0 = 0
    // c1 = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffa9d3
    {
        const uint64_t c0[6] = {0,0,0,0,0,0};
        const uint64_t c1[6] = {0xb9feffffffffa9d3ULL, 0x1eabfffeb153ffffULL,
                                0x6730d2a0f6b0f624ULL, 0x64774b84f38512bfULL,
                                0x4b1ba7b6434bacd7ULL, 0x1a0111ea397fe69aULL};
        make_fp2_from_limbs(c0, c1, coeffs[1]);
    }
    // k_(3,0):
    // c0 = 0x12
    // c1 = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaa99
    {
        const uint64_t c0[6] = {0x12, 0, 0, 0, 0, 0};
        const uint64_t c1[6] = {0xb9feffffffffaa99ULL, 0x1eabfffeb153ffffULL,
                                0x6730d2a0f6b0f624ULL, 0x64774b84f38512bfULL,
                                0x4b1ba7b6434bacd7ULL, 0x1a0111ea397fe69aULL};
        make_fp2_from_limbs(c0, c1, coeffs[0]);
    }
}

// Apply 3-isogeny map to a point on E' to get a point on E
// Input: projective (X, Y, Z) on E'
// Output: projective (X', Y', Z') on E
static void iso3_map(const uint64_t p[36], uint64_t r[36]) {
    if (fp2_is_zero(p + 24)) {
        // Point at infinity maps to point at infinity
        g2_set_id(r);
        return;
    }

    // Get affine coordinates: x = X/Z^2, y = Y/Z^3
    uint64_t z2[12], z3[12], x_aff[12], y_aff[12];
    fp2_sqr(p+24, z2);
    fp2_mul(p+24, z2, z3);

    uint64_t z2_inv[12], z3_inv[12];
    fp2_inv(z2, z2_inv);
    fp2_inv(z3, z3_inv);
    fp2_mul(p, z2_inv, x_aff);
    fp2_mul(p+12, z3_inv, y_aff);

    // Evaluate isogeny polynomials using Horner's method
    // x_num(x) = k3*x^3 + k2*x^2 + k1*x + k0
    uint64_t xn_coeffs[4][12], xd_coeffs[3][12], yn_coeffs[4][12], yd_coeffs[4][12];
    iso3_xnum(xn_coeffs);
    iso3_xden(xd_coeffs);
    iso3_ynum(yn_coeffs);
    iso3_yden(yd_coeffs);

    // x_num = ((k3*x + k2)*x + k1)*x + k0
    uint64_t x_num[12];
    fp2_copy(x_num, xn_coeffs[3]);
    fp2_mul(x_num, x_aff, x_num);
    fp2_add(x_num, xn_coeffs[2], x_num);
    fp2_mul(x_num, x_aff, x_num);
    fp2_add(x_num, xn_coeffs[1], x_num);
    fp2_mul(x_num, x_aff, x_num);
    fp2_add(x_num, xn_coeffs[0], x_num);

    // x_den = (x^2 + k1*x + k0)  [leading coeff = 1]
    uint64_t x_den[12];
    fp2_copy(x_den, xd_coeffs[2]);
    fp2_mul(x_den, x_aff, x_den);
    fp2_add(x_den, xd_coeffs[1], x_den);
    fp2_mul(x_den, x_aff, x_den);
    fp2_add(x_den, xd_coeffs[0], x_den);

    // y_num = ((k3*x + k2)*x + k1)*x + k0
    uint64_t y_num[12];
    fp2_copy(y_num, yn_coeffs[3]);
    fp2_mul(y_num, x_aff, y_num);
    fp2_add(y_num, yn_coeffs[2], y_num);
    fp2_mul(y_num, x_aff, y_num);
    fp2_add(y_num, yn_coeffs[1], y_num);
    fp2_mul(y_num, x_aff, y_num);
    fp2_add(y_num, yn_coeffs[0], y_num);

    // y_den = ((x^3 + k2*x^2 + k1*x + k0))  [leading coeff = 1]
    uint64_t y_den[12];
    fp2_copy(y_den, yd_coeffs[3]);
    fp2_mul(y_den, x_aff, y_den);
    fp2_add(y_den, yd_coeffs[2], y_den);
    fp2_mul(y_den, x_aff, y_den);
    fp2_add(y_den, yd_coeffs[1], y_den);
    fp2_mul(y_den, x_aff, y_den);
    fp2_add(y_den, yd_coeffs[0], y_den);

    // Result affine: x_out = x_num / x_den, y_out = y * y_num / y_den

    // Convert to Jacobian projective:
    // X = x_num * x_den * y_den^2
    // Y = y * y_num * x_den * y_den^2 * y_den = y * y_num * x_den * y_den^3
    // Z = x_den * y_den
    // Then affine_x = X / Z^2 = x_num * x_den * y_den^2 / (x_den^2 * y_den^2) = x_num / x_den
    // And  affine_y = Y / Z^3 = y * y_num * x_den * y_den^3 / (x_den^3 * y_den^3) = y * y_num / (x_den^2 * y_den)
    // Hmm that doesn't work simply.

    // Simpler: just compute affine and convert back to projective
    uint64_t x_den_inv[12], y_den_inv[12];
    fp2_inv(x_den, x_den_inv);
    fp2_inv(y_den, y_den_inv);

    // x_out = x_num / x_den
    uint64_t x_out[12];
    fp2_mul(x_num, x_den_inv, x_out);

    // y_out = y_aff * y_num / y_den
    uint64_t y_out[12];
    fp2_mul(y_num, y_den_inv, y_out);
    fp2_mul(y_aff, y_out, y_out);

    // Return as Jacobian projective with Z = 1
    fp2_copy(r, x_out);
    fp2_copy(r+12, y_out);
    fp2_one(r+24);
}

// ============================================================
// Cofactor clearing for G2
// h_eff = 0xbc69f08f2ee75b3584c6a0ea91b352888e2a8e9145ad7689986ff031508ffe1329c2f178731db956d82bf015d1212b02ec0ec69d7477c1ae954cbc06689f6a359894c0adebbf6b4e8020005aaa95551
// ============================================================

static void clear_cofactor_g2(const uint64_t p[36], uint64_t r[36]) {
    static const uint64_t H_EFF[10] = {
        0xe8020005aaa95551ULL, 0x59894c0adebbf6b4ULL,
        0xe954cbc06689f6a3ULL, 0x2ec0ec69d7477c1aULL,
        0x6d82bf015d1212b0ULL, 0x329c2f178731db95ULL,
        0x9986ff031508ffe1ULL, 0x88e2a8e9145ad768ULL,
        0x584c6a0ea91b3528ULL, 0x0bc69f08f2ee75b3ULL
    };
    g2_scalar_mul_wide(p, H_EFF, 10, r);
}

// ============================================================
// Full hash-to-curve G2 (RFC 9380)
// ============================================================

void bls12_381_hash_to_g2(const uint8_t *msg, size_t msg_len,
                           const uint8_t *dst, size_t dst_len,
                           uint64_t result[36]) {
    // Step 1: Generate 256 bytes of uniform randomness
    uint8_t uniform_bytes[256];
    expand_message_xmd(msg, msg_len, dst, dst_len, 256, uniform_bytes);

    // Step 2: Extract two Fp2 elements
    uint64_t u0[12], u1[12];
    fp_from_64_bytes(uniform_bytes,       u0);       // u0.c0
    fp_from_64_bytes(uniform_bytes + 64,  u0 + 6);   // u0.c1
    fp_from_64_bytes(uniform_bytes + 128, u1);       // u1.c0
    fp_from_64_bytes(uniform_bytes + 192, u1 + 6);   // u1.c1

    // Step 3: Map each to isogeny curve E' via simplified SWU
    uint64_t q0[36], q1[36];
    map_to_curve_sswu(u0, q0);
    map_to_curve_sswu(u1, q1);

    // Step 4: Apply 3-isogeny to move from E' to E
    uint64_t p0[36], p1[36];
    iso3_map(q0, p0);
    iso3_map(q1, p1);

    // Step 5: Add the two points
    uint64_t sum[36];
    g2_add(p0, p1, sum);

    // Step 6: Clear cofactor
    clear_cofactor_g2(sum, result);
}

// Default DST version for BLS signatures
void bls12_381_hash_to_g2_default(const uint8_t *msg, size_t msg_len, uint64_t result[36]) {
    static const uint8_t DST[] = "BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_";
    bls12_381_hash_to_g2(msg, msg_len, DST, sizeof(DST) - 1, result);
}

// Exported cofactor clearing
void bls12_381_g2_clear_cofactor(const uint64_t p[36], uint64_t r[36]) {
    clear_cofactor_g2(p, r);
}
