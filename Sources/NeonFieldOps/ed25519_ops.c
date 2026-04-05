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
#include <pthread.h>

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
