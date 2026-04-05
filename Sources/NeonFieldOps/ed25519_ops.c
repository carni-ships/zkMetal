// Ed25519 Fp field arithmetic using Solinas special reduction
// p = 2^255 - 19
//
// Instead of Montgomery multiplication (CIOS), we use direct integer
// representation and Solinas reduction:
//   For a 510-bit product (h, l), since 2^255 ≡ 19 (mod p):
//   result = low_255_bits + 19 * high_255_bits
//   (equivalently: low_256_bits + 38 * high_256_bits, since 2^256 ≡ 38 mod p)
//
// This avoids Montgomery form entirely and is significantly faster for this
// specific prime.
//
// Limb layout: 4 x uint64_t, little-endian, direct integer (NOT Montgomery).
// Range: [0, p) after final reduction.

#include "NeonFieldOps.h"
#include <string.h>
#include <stdlib.h>
#include <dispatch/dispatch.h>

typedef unsigned __int128 uint128_t;

// p = 2^255 - 19
static const uint64_t ED_P[4] = {
    0xffffffffffffffedULL, 0xffffffffffffffffULL,
    0xffffffffffffffffULL, 0x7fffffffffffffffULL
};

// 2*p (for subtraction without branch)
static const uint64_t ED_2P[4] = {
    0xffffffffffffffdaULL, 0xffffffffffffffffULL,
    0xffffffffffffffffULL, 0xffffffffffffffffULL
};

// ============================================================
// Reduction helpers
// ============================================================

// Reduce a 5-limb value (t[0..3] + t4*2^256) mod p.
// Since 2^256 ≡ 38 (mod p), result = t[0..3] + 38*t4.
// t4 is small (at most ~38 or so), so 38*t4 fits in a uint64.
static inline void ed_reduce5(uint64_t t[4], uint64_t t4) {
    // Add 38 * t4 to t[0..3]
    uint128_t w = (uint128_t)38 * t4 + t[0];
    t[0] = (uint64_t)w;
    uint64_t c = (uint64_t)(w >> 64);
    w = (uint128_t)t[1] + c;
    t[1] = (uint64_t)w;
    c = (uint64_t)(w >> 64);
    w = (uint128_t)t[2] + c;
    t[2] = (uint64_t)w;
    c = (uint64_t)(w >> 64);
    w = (uint128_t)t[3] + c;
    t[3] = (uint64_t)w;
    c = (uint64_t)(w >> 64);
    // If there's still overflow, reduce again (c is 0 or 1)
    if (c) {
        w = (uint128_t)38 * c + t[0];
        t[0] = (uint64_t)w;
        c = (uint64_t)(w >> 64);
        t[1] += c; // no further overflow possible since we're adding at most 38
    }
}

// Final conditional subtraction: if t >= p, subtract p.
// Constant-time (branchless for side-channel resistance).
static inline void ed_final_reduce(uint64_t t[4]) {
    // Compute t - p, check if borrow
    uint128_t w = (uint128_t)t[0] + 19;  // t - p = t - (2^255-19) = t + 19 - 2^255
    uint64_t r0 = (uint64_t)w;
    uint64_t c = (uint64_t)(w >> 64);
    w = (uint128_t)t[1] + c;
    uint64_t r1 = (uint64_t)w;
    c = (uint64_t)(w >> 64);
    w = (uint128_t)t[2] + c;
    uint64_t r2 = (uint64_t)w;
    c = (uint64_t)(w >> 64);
    w = (uint128_t)t[3] + c;
    uint64_t r3 = (uint64_t)w;
    c = (uint64_t)(w >> 64);
    // After adding 19 and propagating carries, if bit 255 is set or there's
    // a carry out of limb 3, then t >= p.
    // If (r3 >> 63) or c is set, we had t + 19 >= 2^255, i.e. t >= p.
    // We want to return (r0,r1,r2,r3&0x7fff...) if t >= p, else (t0,t1,t2,t3).
    uint64_t top = (r3 >> 63) | c;
    r3 &= 0x7fffffffffffffffULL;
    // mask: all 1s if top != 0, all 0s otherwise
    uint64_t mask = (uint64_t)(-(int64_t)top);
    t[0] = (t[0] & ~mask) | (r0 & mask);
    t[1] = (t[1] & ~mask) | (r1 & mask);
    t[2] = (t[2] & ~mask) | (r2 & mask);
    t[3] = (t[3] & ~mask) | (r3 & mask);
}

// ============================================================
// Field arithmetic: add, sub, mul, sqr
// ============================================================

void ed25519_fp_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t w = (uint128_t)a[0] + b[0];
    r[0] = (uint64_t)w;
    uint64_t c = (uint64_t)(w >> 64);
    w = (uint128_t)a[1] + b[1] + c;
    r[1] = (uint64_t)w;
    c = (uint64_t)(w >> 64);
    w = (uint128_t)a[2] + b[2] + c;
    r[2] = (uint64_t)w;
    c = (uint64_t)(w >> 64);
    w = (uint128_t)a[3] + b[3] + c;
    r[3] = (uint64_t)w;
    c = (uint64_t)(w >> 64);
    // If carry out, reduce: add 38 * carry
    ed_reduce5(r, c);
    ed_final_reduce(r);
}

void ed25519_fp_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    // Compute a - b. If borrow, add p.
    uint128_t w = (uint128_t)a[0] - b[0];
    r[0] = (uint64_t)w;
    int64_t borrow = (int64_t)(w >> 64) & 1;  // 0 or 1
    // For subtraction, the top bit of the 128-bit result indicates borrow
    // Actually with unsigned: if a[i] < b[i] + borrow_in, we get borrow
    w = (uint128_t)a[1] - b[1] - borrow;
    r[1] = (uint64_t)w;
    borrow = (w >> 127) ? 1 : 0;
    w = (uint128_t)a[2] - b[2] - borrow;
    r[2] = (uint64_t)w;
    borrow = (w >> 127) ? 1 : 0;
    w = (uint128_t)a[3] - b[3] - borrow;
    r[3] = (uint64_t)w;
    borrow = (w >> 127) ? 1 : 0;
    // If borrow, add p: r += p
    if (borrow) {
        w = (uint128_t)r[0] + ED_P[0];
        r[0] = (uint64_t)w;
        uint64_t c2 = (uint64_t)(w >> 64);
        w = (uint128_t)r[1] + ED_P[1] + c2;
        r[1] = (uint64_t)w;
        c2 = (uint64_t)(w >> 64);
        w = (uint128_t)r[2] + ED_P[2] + c2;
        r[2] = (uint64_t)w;
        c2 = (uint64_t)(w >> 64);
        r[3] = r[3] + ED_P[3] + c2;
    }
}

void ed25519_fp_neg(const uint64_t a[4], uint64_t r[4]) {
    // Check if a is zero
    if ((a[0] | a[1] | a[2] | a[3]) == 0) {
        r[0] = r[1] = r[2] = r[3] = 0;
        return;
    }
    // r = p - a
    uint128_t w = (uint128_t)ED_P[0] - a[0];
    r[0] = (uint64_t)w;
    int64_t borrow = (w >> 127) ? 1 : 0;
    w = (uint128_t)ED_P[1] - a[1] - borrow;
    r[1] = (uint64_t)w;
    borrow = (w >> 127) ? 1 : 0;
    w = (uint128_t)ED_P[2] - a[2] - borrow;
    r[2] = (uint64_t)w;
    borrow = (w >> 127) ? 1 : 0;
    r[3] = ED_P[3] - a[3] - borrow;
}

// Schoolbook 4x4 multiply -> 8 limb product, then Solinas reduce.
// 2^256 ≡ 38 (mod p), so we multiply the top 4 limbs by 38 and add to bottom 4.
// Uses row-based accumulation to avoid carry overflow.
void ed25519_fp_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t t[9];
    uint128_t w;
    uint64_t c;

    memset(t, 0, sizeof(t));

    // Row 0: t += a[0] * b[j] for j=0..3
    w = (uint128_t)a[0] * b[0] + t[0];
    t[0] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[0] * b[1] + t[1] + c;
    t[1] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[0] * b[2] + t[2] + c;
    t[2] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[0] * b[3] + t[3] + c;
    t[3] = (uint64_t)w; t[4] = (uint64_t)(w >> 64);

    // Row 1: t += a[1] * b[j] shifted by 1 limb
    w = (uint128_t)a[1] * b[0] + t[1];
    t[1] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[1] * b[1] + t[2] + c;
    t[2] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[1] * b[2] + t[3] + c;
    t[3] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[1] * b[3] + t[4] + c;
    t[4] = (uint64_t)w; t[5] = (uint64_t)(w >> 64);

    // Row 2: t += a[2] * b[j] shifted by 2 limbs
    w = (uint128_t)a[2] * b[0] + t[2];
    t[2] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[2] * b[1] + t[3] + c;
    t[3] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[2] * b[2] + t[4] + c;
    t[4] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[2] * b[3] + t[5] + c;
    t[5] = (uint64_t)w; t[6] = (uint64_t)(w >> 64);

    // Row 3: t += a[3] * b[j] shifted by 3 limbs
    w = (uint128_t)a[3] * b[0] + t[3];
    t[3] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[3] * b[1] + t[4] + c;
    t[4] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[3] * b[2] + t[5] + c;
    t[5] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[3] * b[3] + t[6] + c;
    t[6] = (uint64_t)w; t[7] = (uint64_t)(w >> 64);

    // Solinas reduction: result = t[0..3] + 38 * t[4..7]
    // Since 2^256 ≡ 38 (mod p)
    // Compute 38 * t[4..7] and add to t[0..3]
    w = (uint128_t)38 * t[4] + t[0];
    r[0] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)38 * t[5] + t[1] + c;
    r[1] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)38 * t[6] + t[2] + c;
    r[2] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)38 * t[7] + t[3] + c;
    r[3] = (uint64_t)w; c = (uint64_t)(w >> 64);

    // c could be 0 or small; one more reduction
    ed_reduce5(r, c);
    ed_final_reduce(r);
}

// Dedicated squaring with upper-triangle optimization.
// Computes 2 * cross_terms + diagonal_terms, then Solinas reduce.
// Saves ~25% multiplies vs general mul.
void ed25519_fp_sqr(const uint64_t a[4], uint64_t r[4]) {
    uint64_t t[9];
    uint128_t w;
    uint64_t c;

    memset(t, 0, sizeof(t));

    // First compute all cross-products a[i]*a[j] for i<j into t,
    // shifted by (i+j) limbs. Then double, then add diagonal terms.

    // Cross-products (upper triangle only):
    // Row 0: a[0] * {a[1], a[2], a[3]} -> t[1..4]
    w = (uint128_t)a[0] * a[1] + t[1];
    t[1] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[0] * a[2] + t[2] + c;
    t[2] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[0] * a[3] + t[3] + c;
    t[3] = (uint64_t)w; t[4] = (uint64_t)(w >> 64);

    // Row 1: a[1] * {a[2], a[3]} -> t[3..5]
    w = (uint128_t)a[1] * a[2] + t[3];
    t[3] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)a[1] * a[3] + t[4] + c;
    t[4] = (uint64_t)w; t[5] = (uint64_t)(w >> 64);

    // Row 2: a[2] * a[3] -> t[5..6]
    w = (uint128_t)a[2] * a[3] + t[5];
    t[5] = (uint64_t)w; t[6] = (uint64_t)(w >> 64);

    // Double all cross terms (shift left by 1 bit)
    t[7] = t[6] >> 63;
    t[6] = (t[6] << 1) | (t[5] >> 63);
    t[5] = (t[5] << 1) | (t[4] >> 63);
    t[4] = (t[4] << 1) | (t[3] >> 63);
    t[3] = (t[3] << 1) | (t[2] >> 63);
    t[2] = (t[2] << 1) | (t[1] >> 63);
    t[1] = (t[1] << 1);

    // Add diagonal terms: a[i]^2 at position 2*i
    w = (uint128_t)a[0] * a[0] + t[0];
    t[0] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)t[1] + c;
    t[1] = (uint64_t)w; c = (uint64_t)(w >> 64);

    w = (uint128_t)a[1] * a[1] + t[2] + c;
    t[2] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)t[3] + c;
    t[3] = (uint64_t)w; c = (uint64_t)(w >> 64);

    w = (uint128_t)a[2] * a[2] + t[4] + c;
    t[4] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)t[5] + c;
    t[5] = (uint64_t)w; c = (uint64_t)(w >> 64);

    w = (uint128_t)a[3] * a[3] + t[6] + c;
    t[6] = (uint64_t)w; c = (uint64_t)(w >> 64);
    t[7] += c;

    // Solinas reduction: r = t[0..3] + 38 * t[4..7]
    w = (uint128_t)38 * t[4] + t[0];
    r[0] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)38 * t[5] + t[1] + c;
    r[1] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)38 * t[6] + t[2] + c;
    r[2] = (uint64_t)w; c = (uint64_t)(w >> 64);
    w = (uint128_t)38 * t[7] + t[3] + c;
    r[3] = (uint64_t)w; c = (uint64_t)(w >> 64);

    ed_reduce5(r, c);
    ed_final_reduce(r);
}

// ============================================================
// Field inverse via Fermat: a^(p-2) mod p
// ============================================================

void ed25519_fp_inverse(const uint64_t a[4], uint64_t r[4]) {
    // p - 2 = 2^255 - 21 = 0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffeb
    // Use addition chain optimized for this exponent
    // Simplified: just use square-and-multiply with ed25519_fp_sqr/mul
    uint64_t exp[4] = {
        0xffffffffffffffebULL, 0xffffffffffffffffULL,
        0xffffffffffffffffULL, 0x7fffffffffffffffULL
    };

    uint64_t result[4] = {1, 0, 0, 0};  // 1 in direct form
    uint64_t base[4];
    memcpy(base, a, 32);

    for (int i = 0; i < 4; i++) {
        uint64_t word = exp[i];
        for (int j = 0; j < 64; j++) {
            if (word & 1) {
                uint64_t tmp[4];
                ed25519_fp_mul(result, base, tmp);
                memcpy(result, tmp, 32);
            }
            uint64_t tmp2[4];
            ed25519_fp_sqr(base, tmp2);
            memcpy(base, tmp2, 32);
            word >>= 1;
        }
    }
    memcpy(r, result, 32);
}

// ============================================================
// Ed25519 curve point operations (extended coordinates)
// Twisted Edwards: -x^2 + y^2 = 1 + d*x^2*y^2 (a = -1)
//
// Extended coords: (X, Y, Z, T) where x = X/Z, y = Y/Z, T = X*Y/Z
// Point layout: 16 uint64_t = X[4], Y[4], Z[4], T[4]
// ============================================================

// d = -121665/121666 mod p (precomputed)
// This is a constant; we compute it once. For now, hardcode:
// d = 37095705934669439343138083508754565189542113879843219016388785533085940283555
static const uint64_t ED_D[4] = {
    0x75eb4dca135978a3ULL, 0x00700a4d4141d8abULL,
    0x8cc740797779e898ULL, 0x52036cee2b6ffe73ULL
};

// 2*d mod p
static const uint64_t ED_2D[4] = {
    0xebd69b9426b2f159ULL, 0x00e0149a8283b156ULL,
    0x198e80f2eef3d130ULL, 0x2406d9dc56dffce7ULL
};

// Identity point: (0, 1, 1, 0)
static void ed_identity(uint64_t p[16]) {
    memset(p, 0, 128);
    p[4] = 1;  // Y = 1
    p[8] = 1;  // Z = 1
}

// Point addition (unified formula for a=-1 twisted Edwards in extended coords)
// Cost: 8M + 1D (where D = mul by d constant)
static void ed_point_add(const uint64_t p[16], const uint64_t q[16], uint64_t out[16]) {
    const uint64_t *X1 = p, *Y1 = p+4, *Z1 = p+8, *T1 = p+12;
    const uint64_t *X2 = q, *Y2 = q+4, *Z2 = q+8, *T2 = q+12;
    uint64_t *X3 = out, *Y3 = out+4, *Z3 = out+8, *T3 = out+12;

    uint64_t A[4], B[4], C[4], D[4], E[4], F[4], G[4], H[4];
    uint64_t tmp1[4], tmp2[4];

    // A = X1 * X2
    ed25519_fp_mul(X1, X2, A);
    // B = Y1 * Y2
    ed25519_fp_mul(Y1, Y2, B);
    // C = d * T1 * T2
    ed25519_fp_mul(T1, T2, tmp1);
    ed25519_fp_mul(tmp1, ED_D, C);
    // D = Z1 * Z2
    ed25519_fp_mul(Z1, Z2, D);
    // E = (X1+Y1)*(X2+Y2) - A - B
    ed25519_fp_add(X1, Y1, tmp1);
    ed25519_fp_add(X2, Y2, tmp2);
    ed25519_fp_mul(tmp1, tmp2, E);
    ed25519_fp_sub(E, A, E);
    ed25519_fp_sub(E, B, E);
    // F = D - C
    ed25519_fp_sub(D, C, F);
    // G = D + C
    ed25519_fp_add(D, C, G);
    // H = B + A (since a = -1, H = B - a*A = B + A)
    ed25519_fp_add(B, A, H);

    // X3 = E * F
    ed25519_fp_mul(E, F, X3);
    // Y3 = G * H
    ed25519_fp_mul(G, H, Y3);
    // Z3 = F * G
    ed25519_fp_mul(F, G, Z3);
    // T3 = E * H
    ed25519_fp_mul(E, H, T3);
}

// Point doubling (more efficient, 4M + 4S)
static void ed_point_double(const uint64_t p[16], uint64_t out[16]) {
    const uint64_t *X1 = p, *Y1 = p+4, *Z1 = p+8;
    uint64_t *X3 = out, *Y3 = out+4, *Z3 = out+8, *T3 = out+12;

    uint64_t A[4], B[4], C[4], D[4], E[4], F_val[4], G[4], H[4];
    uint64_t tmp[4];

    // A = X1^2
    ed25519_fp_sqr(X1, A);
    // B = Y1^2
    ed25519_fp_sqr(Y1, B);
    // C = 2 * Z1^2
    ed25519_fp_sqr(Z1, tmp);
    ed25519_fp_add(tmp, tmp, C);
    // D = -A (since a = -1)
    ed25519_fp_neg(A, D);
    // E = (X1+Y1)^2 - A - B
    ed25519_fp_add(X1, Y1, tmp);
    ed25519_fp_sqr(tmp, E);
    ed25519_fp_sub(E, A, E);
    ed25519_fp_sub(E, B, E);
    // G = D + B
    ed25519_fp_add(D, B, G);
    // F = G - C
    ed25519_fp_sub(G, C, F_val);
    // H = D - B
    ed25519_fp_sub(D, B, H);

    // X3 = E * F
    ed25519_fp_mul(E, F_val, X3);
    // Y3 = G * H
    ed25519_fp_mul(G, H, Y3);
    // Z3 = F * G
    ed25519_fp_mul(F_val, G, Z3);
    // T3 = E * H
    ed25519_fp_mul(E, H, T3);
}

// Scalar multiplication: double-and-add (constant-time not needed for now)
void ed25519_scalar_mul(const uint64_t p[16], const uint64_t scalar[4], uint64_t r[16]) {
    uint64_t result[16], base[16];
    ed_identity(result);
    memcpy(base, p, 128);

    for (int i = 0; i < 4; i++) {
        uint64_t word = scalar[i];
        for (int j = 0; j < 64; j++) {
            if (word & 1) {
                uint64_t tmp[16];
                ed_point_add(result, base, tmp);
                memcpy(result, tmp, 128);
            }
            uint64_t tmp2[16];
            ed_point_double(base, tmp2);
            memcpy(base, tmp2, 128);
            word >>= 1;
        }
    }
    memcpy(r, result, 128);
}

// Point addition (public API)
void ed25519_point_add_c(const uint64_t p[16], const uint64_t q[16], uint64_t r[16]) {
    ed_point_add(p, q, r);
}

// Point doubling (public API)
void ed25519_point_double_c(const uint64_t p[16], uint64_t r[16]) {
    ed_point_double(p, r);
}

// Projective to affine: affine_x = X/Z, affine_y = Y/Z
// Output: 8 uint64_t (x[4], y[4])
void ed25519_point_to_affine(const uint64_t p[16], uint64_t aff[8]) {
    uint64_t zinv[4];
    ed25519_fp_inverse(p + 8, zinv);  // zinv = 1/Z
    ed25519_fp_mul(p, zinv, aff);      // x = X * zinv
    ed25519_fp_mul(p + 4, zinv, aff + 4);  // y = Y * zinv
}

// Convert Montgomery form (4x64 limbs) to Solinas direct integer form
// Montgomery: val * R mod p, where R = 2^256
// To convert out: multiply by R^{-1} = R^{p-2} mod p
// But actually: in Montgomery, to get the integer, we multiply by 1
// using Montgomery reduction. For 2^255-19, R = 38.
// MontToInt(m) = m * R^{-1} mod p.
// Since R = 38, R^{-1} = 38^{-1} mod p.
// Direct computation: 38^{-1} mod (2^255 - 19).
//
// Instead, we do: multiply m by R^{-1} mod p.
// R^{-1} mod p where R = 2^256 = 38 mod p.
// So R^{-1} = the modular inverse of 38 mod p.
//
// But it's easier to just do the Montgomery reduction trick:
// Multiply m by 1 in Montgomery = reduce.
// t = m * 1 = m (8 limbs with top 4 zero)
// Then CIOS reduction gives m * R^{-1} mod p.
//
// Actually, for simplicity, we can compute:
// int_value = mont_value * R^{-1} mod p
// We precompute R^{-1} = inverse(38) mod p
// Actually: R = 2^256 mod p = 38. R^{-1} = 38^{-1} mod p.
// 38 * R^{-1} ≡ 1 (mod p)
// Since p = 2^255 - 19, we have:
// 2 * 19 = 38, and 2^255 ≡ 19 (mod p), so 2^{-1} ≡ (p+1)/2 = 2^254 - 9 ... complicated.
// Let's just hardcode R_INV.
// R_INV = modpow(38, p-2, p)
//
// For a simpler approach for the conversion functions:
// We pass the Swift-side computed limbs directly since the Swift code
// can handle the conversion.

// Convert from Montgomery 4x64 to direct integer 4x64
// This performs Montgomery reduction: result = mont * R^{-1} mod p
// where R = 2^256, using the fact that for p = 2^255 - 19,
// we can use Solinas-style reduction.
//
// The Montgomery reduction of m: we want m * 2^{-256} mod p.
// Method: q = m * p_inv mod 2^256, then (m + q*p) / 2^256.
// But since we're replacing Montgomery entirely, conversion functions
// are only needed at the boundary.
void ed25519_mont_to_direct(const uint64_t mont[4], uint64_t direct[4]) {
    // Montgomery value = val * R mod p, where R = 2^256 mod p = 38
    // So val = mont * R^{-1} mod p
    // R^{-1} mod p: we need inverse of 38 mod p
    // Compute via: mont * (38^{-1}) mod p using our Solinas mul
    // 38^{-1} mod p (precomputed):
    // p = 2^255 - 19
    // 38 = 2 * 19, 2^255 = 19 + p, so 19 = 2^255 - p
    // 38 = 2^256 - 2p ≡ 2^256 mod p... wait that means 38 ≡ R ≡ 2^256 mod p
    // So R^{-1} is what we need.
    //
    // Actually, the simplest: do CIOS Montgomery reduction.
    // mont_reduce(m) = m * R^{-1} mod p using the standard Montgomery trick:
    // q = m[0] * (-p^{-1}) mod 2^64
    // then (m + q*p) >> 64, repeat 4 times
    //
    // For a 4-limb value (not an 8-limb product), we treat it as
    // m * 1 (where 1 has limbs [1,0,0,0]), and run Montgomery reduction.
    //
    // INV = -p^{-1} mod 2^64 = 0x86bca1af286bca1b

    static const uint64_t INV = 0x86bca1af286bca1bULL;
    uint64_t t[5];
    memcpy(t, mont, 32);
    t[4] = 0;

    for (int i = 0; i < 4; i++) {
        uint64_t m = t[0] * INV;
        uint128_t w = (uint128_t)m * ED_P[0] + t[0];
        uint64_t c = (uint64_t)(w >> 64);
        w = (uint128_t)m * ED_P[1] + t[1] + c;
        t[0] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * ED_P[2] + t[2] + c;
        t[1] = (uint64_t)w; c = (uint64_t)(w >> 64);
        w = (uint128_t)m * ED_P[3] + t[3] + c;
        t[2] = (uint64_t)w; c = (uint64_t)(w >> 64);
        t[3] = t[4] + c;
        t[4] = (t[3] < c) ? 1 : 0;
    }

    // Conditional subtract
    memcpy(direct, t, 32);
    if (t[4] || (t[3] > ED_P[3]) ||
        (t[3] == ED_P[3] && t[2] > ED_P[2]) ||
        (t[3] == ED_P[3] && t[2] == ED_P[2] && t[1] > ED_P[1]) ||
        (t[3] == ED_P[3] && t[2] == ED_P[2] && t[1] == ED_P[1] && t[0] >= ED_P[0])) {
        uint128_t w = (uint128_t)t[0] - ED_P[0];
        direct[0] = (uint64_t)w;
        int borrow = (w >> 127) ? 1 : 0;
        w = (uint128_t)t[1] - ED_P[1] - borrow;
        direct[1] = (uint64_t)w;
        borrow = (w >> 127) ? 1 : 0;
        w = (uint128_t)t[2] - ED_P[2] - borrow;
        direct[2] = (uint64_t)w;
        borrow = (w >> 127) ? 1 : 0;
        direct[3] = t[3] - ED_P[3] - borrow;
    }
}

// Convert from direct integer to Montgomery form
// result = val * R mod p = val * 38 mod p (since R = 2^256 ≡ 38 mod p)
void ed25519_direct_to_mont(const uint64_t direct[4], uint64_t mont[4]) {
    static const uint64_t R_VAL[4] = {38, 0, 0, 0};
    // Use Solinas mul: result = direct * 38 mod p
    ed25519_fp_mul(direct, R_VAL, mont);
}

// ============================================================
// Ed25519 Pippenger MSM (multi-scalar multiplication)
//
// Twisted Edwards Pippenger with:
//   - Extended coordinates (X, Y, Z, T)
//   - Solinas field ops (NOT Montgomery)
//   - Mixed addition (extended + affine, saves 1D = mul by d)
//   - Batch-to-affine via Montgomery's trick (1 inversion per window)
//   - Multi-threaded windows via pthreads
//   - Signed-digit scalar recoding
//   - Adaptive window sizing
// ============================================================

// Helper: check if extended point is identity
// Identity: (0, c, c, 0) — normalized identity is (0, 1, 1, 0)
// We check X == 0 (sufficient since T == 0 implies X == 0 or Y == 0,
// and for valid curve points X == 0 <=> point is identity).
static inline int ed_pt_is_id(const uint64_t p[16]) {
    return (p[0] | p[1] | p[2] | p[3]) == 0;
}

// Helper: set point to identity (0, 1, 1, 0)
static inline void ed_pt_set_id(uint64_t p[16]) {
    memset(p, 0, 128);
    p[4] = 1;  // Y = 1
    p[8] = 1;  // Z = 1
}

// Helper: check if affine point is identity (represented as x=0, y=0)
static inline int ed_aff_is_id(const uint64_t q[8]) {
    return (q[0] | q[1] | q[2] | q[3] | q[4] | q[5] | q[6] | q[7]) == 0;
}

// Mixed addition: extended P + affine Q (where Q has Z=1)
// Uses the fact that Z_Q = 1 to save one multiplication.
// For twisted Edwards a=-1:
//   A = X1 * X2
//   B = Y1 * Y2
//   C = d * T1 * T2  (T2 = X2*Y2 for affine point)
//   D = Z1  (since Z2 = 1)
//   E = (X1+Y1)*(X2+Y2) - A - B
//   F = D - C
//   G = D + C
//   H = B + A  (a = -1, so H = B - aA = B + A)
//   X3 = E*F, Y3 = G*H, Z3 = F*G, T3 = E*H
//
// Affine format: q_aff = [x[4], y[4]] (8 uint64_t)
static void ed_pt_add_mixed(const uint64_t p[16], const uint64_t q_aff[8], uint64_t out[16]) {
    if (ed_aff_is_id(q_aff)) {
        memcpy(out, p, 128);
        return;
    }
    if (ed_pt_is_id(p)) {
        // Convert affine to extended: X=x, Y=y, Z=1, T=x*y
        memcpy(out, q_aff, 64);         // X, Y
        out[8] = 1; out[9] = 0; out[10] = 0; out[11] = 0;  // Z = 1
        ed25519_fp_mul(q_aff, q_aff + 4, out + 12);         // T = x*y
        return;
    }

    const uint64_t *X1 = p, *Y1 = p+4, *Z1 = p+8, *T1 = p+12;
    const uint64_t *X2 = q_aff, *Y2 = q_aff + 4;
    uint64_t *X3 = out, *Y3 = out+4, *Z3 = out+8, *T3 = out+12;

    uint64_t A[4], B[4], C[4], E[4], F[4], G[4], H[4];
    uint64_t tmp1[4], tmp2[4], T2[4];

    // A = X1 * X2
    ed25519_fp_mul(X1, X2, A);
    // B = Y1 * Y2
    ed25519_fp_mul(Y1, Y2, B);
    // T2 = X2 * Y2 (affine T coordinate)
    ed25519_fp_mul(X2, Y2, T2);
    // C = d * T1 * T2
    ed25519_fp_mul(T1, T2, tmp1);
    ed25519_fp_mul(tmp1, ED_D, C);
    // D = Z1 (since Z2 = 1, D = Z1 * Z2 = Z1)
    // E = (X1+Y1)*(X2+Y2) - A - B
    ed25519_fp_add(X1, Y1, tmp1);
    ed25519_fp_add(X2, Y2, tmp2);
    ed25519_fp_mul(tmp1, tmp2, E);
    ed25519_fp_sub(E, A, E);
    ed25519_fp_sub(E, B, E);
    // F = D - C = Z1 - C
    ed25519_fp_sub(Z1, C, F);
    // G = D + C = Z1 + C
    ed25519_fp_add(Z1, C, G);
    // H = B + A (since a = -1)
    ed25519_fp_add(B, A, H);

    // X3 = E * F
    ed25519_fp_mul(E, F, X3);
    // Y3 = G * H
    ed25519_fp_mul(G, H, Y3);
    // Z3 = F * G
    ed25519_fp_mul(F, G, Z3);
    // T3 = E * H
    ed25519_fp_mul(E, H, T3);
}

// Negate an affine point: for twisted Edwards, neg(x,y) = (-x, y)
static inline void ed_aff_negate(const uint64_t q[8], uint64_t out[8]) {
    ed25519_fp_neg(q, out);          // out.x = -q.x
    memcpy(out + 4, q + 4, 32);     // out.y = q.y
}

// Batch projective-to-affine (Montgomery's trick) for extended coords.
// Converts n extended points to affine (x, y) = (X/Z, Y/Z).
// Output: n×8 uint64_t (x[4], y[4] per point).
// Identity points get (0,0) in affine output.
static void ed_batch_to_affine(const uint64_t *ext, uint64_t *aff, int n) {
    if (n == 0) return;

    // Compute cumulative products of Z values
    uint64_t *prods = (uint64_t *)malloc((size_t)n * 32);
    int first_valid = -1;

    for (int i = 0; i < n; i++) {
        const uint64_t *Z = ext + i * 16 + 8;
        int is_id = ed_pt_is_id(ext + i * 16);
        if (is_id) {
            if (i == 0) {
                prods[0] = 1; prods[1] = 0; prods[2] = 0; prods[3] = 0;
            } else {
                memcpy(prods + i * 4, prods + (i-1) * 4, 32);
            }
        } else {
            if (first_valid < 0) {
                first_valid = i;
                memcpy(prods + i * 4, Z, 32);
            } else {
                ed25519_fp_mul(prods + (i-1) * 4, Z, prods + i * 4);
            }
        }
    }

    if (first_valid < 0) {
        memset(aff, 0, (size_t)n * 64);
        free(prods);
        return;
    }

    // Invert the product
    uint64_t inv[4];
    ed25519_fp_inverse(prods + (n-1) * 4, inv);

    // Back-propagate
    for (int i = n - 1; i >= 0; i--) {
        if (ed_pt_is_id(ext + i * 16)) {
            memset(aff + i * 8, 0, 64);
            continue;
        }

        uint64_t zinv[4];
        if (i > first_valid) {
            ed25519_fp_mul(inv, prods + (i-1) * 4, zinv);
            ed25519_fp_mul(inv, ext + i * 16 + 8, inv);  // inv *= Z_i
        } else {
            memcpy(zinv, inv, 32);
        }

        // For twisted Edwards: x = X/Z, y = Y/Z (only need Z^{-1}, not Z^{-2}/Z^{-3})
        ed25519_fp_mul(ext + i * 16, zinv, aff + i * 8);       // x = X * Z^{-1}
        ed25519_fp_mul(ext + i * 16 + 4, zinv, aff + i * 8 + 4); // y = Y * Z^{-1}
    }

    free(prods);
}

// Scalar window extraction (from uint32_t limbs)
static inline uint32_t ed_extract_window(const uint32_t *scalar, int window_idx, int window_bits) {
    int bit_offset = window_idx * window_bits;
    int word_idx = bit_offset / 32;
    int bit_in_word = bit_offset % 32;

    uint64_t word = scalar[word_idx];
    if (word_idx + 1 < 8)
        word |= ((uint64_t)scalar[word_idx + 1]) << 32;

    return (uint32_t)((word >> bit_in_word) & ((1u << window_bits) - 1));
}

// Adaptive window sizing for Ed25519
static int ed_optimal_window_bits(int n) {
    if (n <= 4)     return 3;
    if (n <= 32)    return 5;
    if (n <= 256)   return 8;
    if (n <= 2048)  return 10;
    if (n <= 8192)  return 11;
    if (n <= 32768) return 13;
    if (n <= 131072) return 14;
    if (n <= 524288) return 15;
    return 16;
}

// Per-window worker with pre-extracted signed digits
typedef struct {
    const uint64_t *points;     // n affine points (8 uint64_t each)
    const uint32_t *digits;     // n signed digits for this window (high bit = negate)
    int n;
    int window_bits;
    int num_buckets;            // halfBuckets
    uint64_t result[16];        // output extended point
} EdSignedWindowTask;

static void *ed_signed_window_worker(void *arg) {
    EdSignedWindowTask *task = (EdSignedWindowTask *)arg;
    int nb = task->num_buckets;
    int nn = task->n;

    // Allocate extended projective buckets (identity-initialized)
    uint64_t *buckets = (uint64_t *)malloc((size_t)(nb + 1) * 128);
    for (int b = 0; b <= nb; b++)
        ed_pt_set_id(buckets + b * 16);

    // Phase 1: Bucket accumulation with signed digits
    for (int i = 0; i < nn; i++) {
        uint32_t raw = task->digits[i];
        uint32_t digit = raw & 0x7FFFFFFF;
        if (digit == 0) continue;

        if (raw & 0x80000000) {
            // Negate the affine point, then add
            uint64_t neg_pt[8];
            ed_aff_negate(task->points + i * 8, neg_pt);
            uint64_t tmp[16];
            ed_pt_add_mixed(buckets + digit * 16, neg_pt, tmp);
            memcpy(buckets + digit * 16, tmp, 128);
        } else {
            uint64_t tmp[16];
            ed_pt_add_mixed(buckets + digit * 16, task->points + i * 8, tmp);
            memcpy(buckets + digit * 16, tmp, 128);
        }
    }

    // Phase 2: Batch convert buckets to affine
    uint64_t *bucket_aff = (uint64_t *)malloc((size_t)nb * 64);
    ed_batch_to_affine(buckets + 16, bucket_aff, nb);

    // Phase 3: Running-sum reduction
    uint64_t running[16], window_sum[16];
    ed_pt_set_id(running);
    ed_pt_set_id(window_sum);

    for (int j = nb - 1; j >= 0; j--) {
        if (!ed_aff_is_id(bucket_aff + j * 8)) {
            uint64_t tmp[16];
            ed_pt_add_mixed(running, bucket_aff + j * 8, tmp);
            memcpy(running, tmp, 128);
        }
        uint64_t tmp[16];
        ed_point_add(window_sum, running, tmp);
        memcpy(window_sum, tmp, 128);
    }

    memcpy(task->result, window_sum, 128);
    free(buckets);
    free(bucket_aff);
    return NULL;
}

// ============================================================
// Main entry point: ed25519_pippenger_msm
//
// points:  n affine points as n×8 uint64_t (x[4], y[4] per point)
// scalars: n scalars as n×8 uint32_t (little-endian limbs, ~253-bit)
// n:       number of points
// result:  output extended point: 16 uint64_t (X[4], Y[4], Z[4], T[4])
// ============================================================

void ed25519_pippenger_msm(
    const uint64_t *points,
    const uint32_t *scalars,
    int n,
    uint64_t *result)
{
    if (n == 0) { ed_pt_set_id(result); return; }

    // For very small inputs, use direct scalar-mul accumulation
    if (n <= 2) {
        ed_pt_set_id(result);
        for (int i = 0; i < n; i++) {
            // Convert affine to extended
            uint64_t ext[16];
            memcpy(ext, points + i * 8, 32);       // X = x
            memcpy(ext + 4, points + i * 8 + 4, 32); // Y = y
            ext[8] = 1; ext[9] = 0; ext[10] = 0; ext[11] = 0; // Z = 1
            ed25519_fp_mul(points + i * 8, points + i * 8 + 4, ext + 12); // T = x*y

            // Convert scalar from 8×uint32 to 4×uint64
            uint64_t scalar64[4];
            for (int j = 0; j < 4; j++)
                scalar64[j] = (uint64_t)scalars[i * 8 + j * 2] |
                              ((uint64_t)scalars[i * 8 + j * 2 + 1] << 32);

            uint64_t sp[16];
            ed25519_scalar_mul(ext, scalar64, sp);

            uint64_t tmp[16];
            ed_point_add(result, sp, tmp);
            memcpy(result, tmp, 128);
        }
        return;
    }

    int wb = ed_optimal_window_bits(n);
    int scalar_bits = 253;  // Ed25519 scalar field is ~253 bits
    int num_windows = (scalar_bits + wb - 1) / wb;
    int full_buckets = 1 << wb;
    int half_buckets = full_buckets >> 1;

    // Use signed-digit recoding for better bucket distribution
    // Precompute all signed digits (requires sequential per-scalar processing)
    uint32_t *all_digits = (uint32_t *)malloc((size_t)num_windows * (size_t)n * sizeof(uint32_t));
    uint32_t mask = (uint32_t)((1u << wb) - 1);

    for (int i = 0; i < n; i++) {
        uint32_t carry = 0;
        const uint32_t *s = scalars + i * 8;
        for (int w = 0; w < num_windows; w++) {
            int bit_offset = w * wb;
            int word_idx = bit_offset / 32;
            int bit_in_word = bit_offset % 32;

            uint64_t word = s[word_idx];
            if (word_idx + 1 < 8)
                word |= ((uint64_t)s[word_idx + 1]) << 32;

            uint32_t digit = (uint32_t)((word >> bit_in_word) & mask) + carry;
            carry = 0;

            if ((int)digit > half_buckets) {
                digit = full_buckets - digit;
                carry = 1;
                all_digits[w * n + i] = digit | 0x80000000;  // high bit = negate
            } else {
                all_digits[w * n + i] = digit;
            }
        }
    }

    // Launch one block per window via GCD
    EdSignedWindowTask *tasks = (EdSignedWindowTask *)malloc(
        (size_t)num_windows * sizeof(EdSignedWindowTask));

    for (int w = 0; w < num_windows; w++) {
        tasks[w].points = points;
        tasks[w].digits = all_digits + w * n;
        tasks[w].n = n;
        tasks[w].window_bits = wb;
        tasks[w].num_buckets = half_buckets;
    }

    dispatch_apply(num_windows, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
        ^(size_t w) {
            ed_signed_window_worker(&tasks[w]);
        });

    // Horner combination: result = Sum windowResults[w] * 2^(w * wb)
    memcpy(result, tasks[num_windows - 1].result, 128);
    for (int w = num_windows - 2; w >= 0; w--) {
        uint64_t tmp[16];
        for (int s = 0; s < wb; s++) {
            ed_point_double(result, tmp);
            memcpy(result, tmp, 128);
        }
        ed_point_add(result, tasks[w].result, tmp);
        memcpy(result, tmp, 128);
    }

    free(all_digits);
    free(tasks);
}

// ============================================================
// Ed25519 Scalar Field Fq (order of the curve)
// q = 2^252 + 27742317777372353535851937790883648493
//   = 0x1000000000000000000000000000000014def9dea2f79cd65812631a5cf5d3ed
//
// Montgomery form with R = 2^256
// CIOS multiplication
// ============================================================

static const uint64_t ED_Q[4] = {
    0x5812631a5cf5d3edULL, 0x14def9dea2f79cd6ULL,
    0x0000000000000000ULL, 0x1000000000000000ULL
};

// -q^{-1} mod 2^64
static const uint64_t ED_Q_INV = 0xd2b51da312547e1bULL;

// R mod q = 2^256 mod q
static const uint64_t ED_R_MOD_Q[4] = {
    0xd6ec31748d98951dULL, 0xc6ef5bf4737dcf70ULL,
    0xfffffffffffffffeULL, 0x0fffffffffffffffULL
};

// R^2 mod q
static const uint64_t ED_R2_MOD_Q[4] = {
    0xa40611e3449c0f01ULL, 0xd00e1ba768859347ULL,
    0xceec73d217f5be65ULL, 0x0399411b7c309a3dULL
};

// Fq comparison: return 1 if a >= b
static inline int ed_fq_gte(const uint64_t a[4], const uint64_t b[4]) {
    for (int i = 3; i >= 0; i--) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return 0;
    }
    return 1;  // equal
}

// Fq conditional subtract: if a >= q, set a -= q
static inline void ed_fq_reduce(uint64_t a[4]) {
    if (!ed_fq_gte(a, ED_Q)) return;
    uint128_t w = (uint128_t)a[0] - ED_Q[0];
    a[0] = (uint64_t)w;
    int borrow = (w >> 127) ? 1 : 0;
    w = (uint128_t)a[1] - ED_Q[1] - borrow;
    a[1] = (uint64_t)w;
    borrow = (w >> 127) ? 1 : 0;
    w = (uint128_t)a[2] - ED_Q[2] - borrow;
    a[2] = (uint64_t)w;
    borrow = (w >> 127) ? 1 : 0;
    a[3] = a[3] - ED_Q[3] - borrow;
}

// Fq CIOS Montgomery multiplication
void ed25519_fq_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t t[6] = {0};

    for (int i = 0; i < 4; i++) {
        // Multiply: t += a[i] * b
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            uint128_t w = (uint128_t)a[i] * b[j] + t[j] + carry;
            t[j] = (uint64_t)w;
            carry = (uint64_t)(w >> 64);
        }
        uint128_t w = (uint128_t)t[4] + carry;
        t[4] = (uint64_t)w;
        t[5] = (uint64_t)(w >> 64);

        // Reduce: m = t[0] * (-q^{-1}) mod 2^64, then t += m * q, shift right
        uint64_t m = t[0] * ED_Q_INV;
        carry = 0;
        for (int j = 0; j < 4; j++) {
            uint128_t w2 = (uint128_t)m * ED_Q[j] + t[j] + carry;
            t[j] = (uint64_t)w2;
            carry = (uint64_t)(w2 >> 64);
        }
        w = (uint128_t)t[4] + carry;
        t[4] = (uint64_t)w;
        t[5] = t[5] + (uint64_t)(w >> 64);

        // Shift right by one limb
        t[0] = t[1]; t[1] = t[2]; t[2] = t[3]; t[3] = t[4]; t[4] = t[5]; t[5] = 0;
    }

    memcpy(r, t, 32);
    if (t[4] || ed_fq_gte(r, ED_Q)) {
        uint128_t w = (uint128_t)r[0] - ED_Q[0];
        r[0] = (uint64_t)w;
        int borrow = (w >> 127) ? 1 : 0;
        w = (uint128_t)r[1] - ED_Q[1] - borrow;
        r[1] = (uint64_t)w;
        borrow = (w >> 127) ? 1 : 0;
        w = (uint128_t)r[2] - ED_Q[2] - borrow;
        r[2] = (uint64_t)w;
        borrow = (w >> 127) ? 1 : 0;
        r[3] = r[3] - ED_Q[3] - borrow;
    }
}

void ed25519_fq_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        uint128_t w = (uint128_t)a[i] + b[i] + carry;
        r[i] = (uint64_t)w;
        carry = (uint64_t)(w >> 64);
    }
    if (carry || ed_fq_gte(r, ED_Q)) {
        uint128_t w = (uint128_t)r[0] - ED_Q[0];
        r[0] = (uint64_t)w;
        int borrow = (w >> 127) ? 1 : 0;
        w = (uint128_t)r[1] - ED_Q[1] - borrow;
        r[1] = (uint64_t)w;
        borrow = (w >> 127) ? 1 : 0;
        w = (uint128_t)r[2] - ED_Q[2] - borrow;
        r[2] = (uint64_t)w;
        borrow = (w >> 127) ? 1 : 0;
        r[3] = r[3] - ED_Q[3] - borrow;
    }
}

void ed25519_fq_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint128_t w = (uint128_t)a[0] - b[0];
    r[0] = (uint64_t)w;
    int borrow = (w >> 127) ? 1 : 0;
    w = (uint128_t)a[1] - b[1] - borrow;
    r[1] = (uint64_t)w;
    borrow = (w >> 127) ? 1 : 0;
    w = (uint128_t)a[2] - b[2] - borrow;
    r[2] = (uint64_t)w;
    borrow = (w >> 127) ? 1 : 0;
    w = (uint128_t)a[3] - b[3] - borrow;
    r[3] = (uint64_t)w;
    borrow = (w >> 127) ? 1 : 0;
    if (borrow) {
        w = (uint128_t)r[0] + ED_Q[0];
        r[0] = (uint64_t)w;
        uint64_t c = (uint64_t)(w >> 64);
        w = (uint128_t)r[1] + ED_Q[1] + c;
        r[1] = (uint64_t)w;
        c = (uint64_t)(w >> 64);
        w = (uint128_t)r[2] + ED_Q[2] + c;
        r[2] = (uint64_t)w;
        c = (uint64_t)(w >> 64);
        r[3] = r[3] + ED_Q[3] + c;
    }
}

// Convert raw integer to Montgomery: result = raw * R^2 mod q (via CIOS)
void ed25519_fq_from_raw(const uint64_t raw[4], uint64_t mont[4]) {
    ed25519_fq_mul(raw, ED_R2_MOD_Q, mont);
}

// Convert Montgomery to raw integer: result = mont * 1 (CIOS with b=1)
void ed25519_fq_to_raw(const uint64_t mont[4], uint64_t raw[4]) {
    uint64_t one[4] = {1, 0, 0, 0};
    ed25519_fq_mul(mont, one, raw);
}

// Reduce a 512-bit (64-byte) value mod q, result in Montgomery form.
// Used for hash-to-scalar in EdDSA.
// Method: split into lo[0..31] and hi[0..31], compute hi * 2^256 + lo mod q.
// 2^256 mod q = R_MOD_Q (constant).
void ed25519_fq_from_bytes64(const uint8_t bytes[64], uint64_t mont[4]) {
    uint64_t lo[4] = {0}, hi[4] = {0};
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            lo[i] |= (uint64_t)bytes[i * 8 + j] << (j * 8);
            hi[i] |= (uint64_t)bytes[32 + i * 8 + j] << (j * 8);
        }
    }
    // Convert to Montgomery: lo_mont = lo * R^2, hi_mont = hi * R^2
    uint64_t lo_mont[4], hi_mont[4];
    ed25519_fq_from_raw(lo, lo_mont);
    ed25519_fq_from_raw(hi, hi_mont);
    // R_MOD_Q in Montgomery = R_MOD_Q * R^2 / R = R_MOD_Q * R mod q
    // Actually we need R_MOD_Q in Montgomery form
    uint64_t r_mod_q_mont[4];
    ed25519_fq_from_raw(ED_R_MOD_Q, r_mod_q_mont);
    // hi_mont * r_mod_q_mont = hi * R_MOD_Q in Montgomery
    uint64_t hi_times_r[4];
    ed25519_fq_mul(hi_mont, r_mod_q_mont, hi_times_r);
    ed25519_fq_add(hi_times_r, lo_mont, mont);
}

// Convert Montgomery Fq to 32-byte little-endian output
void ed25519_fq_to_bytes(const uint64_t mont[4], uint8_t bytes[32]) {
    uint64_t raw[4];
    ed25519_fq_to_raw(mont, raw);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            bytes[i * 8 + j] = (uint8_t)(raw[i] >> (j * 8));
        }
    }
}

// ============================================================
// Shamir's Trick: simultaneous double-scalar multiplication
// Computes s*G + h*A using a joint scan (Straus/Shamir method).
//
// For Ed25519 verify: check s*G == R + h*A, i.e. s*G - h*A == R.
// We compute s*G + h*(-A) and compare with R.
//
// This uses a 2-bit window (one bit from each scalar) for ~25% savings.
// ============================================================

void ed25519_shamir_double_mul(
    const uint64_t G[16],     // base point (extended)
    const uint64_t s[4],      // scalar for G (raw integer, NOT Montgomery)
    const uint64_t A[16],     // second point (extended)
    const uint64_t h[4],      // scalar for A (raw integer, NOT Montgomery)
    uint64_t result[16])      // output (extended)
{
    // Precompute: table[0] = identity, table[1] = G, table[2] = A, table[3] = G+A
    uint64_t table[4][16];
    ed_identity(table[0]);
    memcpy(table[1], G, 128);
    memcpy(table[2], A, 128);
    ed_point_add(G, A, table[3]);

    ed_identity(result);

    // Scan from MSB to LSB
    for (int i = 252; i >= 0; i--) {
        // Double
        uint64_t tmp[16];
        ed_point_double(result, tmp);
        memcpy(result, tmp, 128);

        // Extract bits
        int word_s = i / 64, bit_s = i % 64;
        int word_h = i / 64, bit_h = i % 64;
        int bs = (s[word_s] >> bit_s) & 1;
        int bh = (h[word_h] >> bit_h) & 1;
        int idx = bs | (bh << 1);

        if (idx != 0) {
            ed_point_add(result, table[idx], tmp);
            memcpy(result, tmp, 128);
        }
    }
}

// ============================================================
// Ed25519 EdDSA Sign (C-accelerated)
//
// Given nonce scalar r (raw), compute R = r*G and return R as
// extended point. The caller handles hashing.
// ============================================================

void ed25519_eddsa_sign_compute_r(
    const uint64_t gen[16],     // generator (extended)
    const uint64_t r_scalar[4], // nonce scalar (raw integer)
    uint64_t r_point[16])       // output R (extended)
{
    ed25519_scalar_mul(gen, r_scalar, r_point);
}

// Compute S = (r + k * a) mod q, all in Montgomery form
void ed25519_eddsa_sign_compute_s(
    const uint64_t r_mont[4],   // nonce in Montgomery form
    const uint64_t k_mont[4],   // challenge hash in Montgomery form
    const uint64_t a_mont[4],   // secret scalar in Montgomery form
    uint64_t s_mont[4])         // output S in Montgomery form
{
    uint64_t ka[4];
    ed25519_fq_mul(k_mont, a_mont, ka);
    ed25519_fq_add(r_mont, ka, s_mont);
}

// ============================================================
// Ed25519 EdDSA Verify (C-accelerated)
//
// Check: s*G == R + h*A
// Equivalently: s*G + h*(-A) - R == identity
// Using Shamir's trick for s*G + h*(-A)
// ============================================================

int ed25519_eddsa_verify(
    const uint64_t gen[16],     // generator (extended)
    const uint64_t s_raw[4],    // S scalar (raw integer, NOT Montgomery)
    const uint64_t r_point[16], // R point (extended)
    const uint64_t h_raw[4],    // challenge hash (raw integer)
    const uint64_t pub_key[16]) // public key A (extended)
{
    // Negate A: for twisted Edwards, neg(X,Y,Z,T) = (-X,Y,Z,-T)
    uint64_t neg_a[16];
    ed25519_fp_neg(pub_key, neg_a);          // -X
    memcpy(neg_a + 4, pub_key + 4, 32);     // Y
    memcpy(neg_a + 8, pub_key + 8, 32);     // Z
    ed25519_fp_neg(pub_key + 12, neg_a + 12); // -T

    // Compute s*G + h*(-A) using Shamir's trick
    uint64_t lhs[16];
    ed25519_shamir_double_mul(gen, s_raw, neg_a, h_raw, lhs);

    // Compare lhs with R by converting both to affine
    uint64_t lhs_aff[8], r_aff[8];
    ed25519_point_to_affine(lhs, lhs_aff);
    ed25519_point_to_affine(r_point, r_aff);

    // Final reduce both for comparison
    ed_final_reduce(lhs_aff);
    ed_final_reduce(lhs_aff + 4);
    ed_final_reduce(r_aff);
    ed_final_reduce(r_aff + 4);

    return memcmp(lhs_aff, r_aff, 64) == 0;
}
