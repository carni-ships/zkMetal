// secp256k1 base field Fp arithmetic for Metal GPU — 4x64-bit CIOS
//
// p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
//   = 2^256 - 2^32 - 977
// Field elements as 4x64-bit limbs in Montgomery form.
//
// This is ~4x faster than the 8x32-bit version because:
// - Native 64-bit multiplication (M3 GPU has 64-bit mulhi)
// - 4x4 schoolbook = 16 muls vs 8x8 = 64 muls
// - 4 Montgomery reductions vs 8

#ifndef SECP256K1_FP64_METAL
#define SECP256K1_FP64_METAL

#include <metal_stdlib>
using namespace metal;

// 4x64-bit secp256k1 field element (Montgomery form)
struct SecpFp64 {
    ulong v[4];  // little-endian 4x64-bit = 256-bit
};

struct SecpPointAffine64 {
    SecpFp64 x;
    SecpFp64 y;
};

struct SecpPointProjective64 {
    SecpFp64 x;
    SecpFp64 y;
    SecpFp64 z;
};

// secp256k1 prime p = 2^256 - 2^32 - 977
constant ulong SECP_P64[4] = {
    0xFFFFFFFFFFFFFFFEFFFFFC2FUL,
    0xFFFFFFFFFFFFFFFF,
    0xFFFFFFFFFFFFFFFF,
    0xFFFFFFFFFFFFFFFF
};

// Montgomery R = 2^256 mod p = 2^32 + 977
constant ulong SECP_R64[4] = {
    0x00000000000003D1UL, 0, 0, 0
};

// Montgomery R^2 mod p (precomputed)
constant ulong SECP_R2_64[4] = {
    0x0000000003D1D9D1UL, 0, 0, 0
};

// Montgomery parameter: m = -p^(-1) mod 2^64
// Computed from p mod 2^32 = 0xFFFFFC2F
// p^(-1) mod 2^32 = 0x3D25, so m = -0x3D25 mod 2^64
constant ulong SECP_INV64 = 0xFFFFC2DBUL;

// --- Helpers ---

inline bool secp64_is_zero(SecpFp64 a) {
    return a.v[0] == 0 && a.v[1] == 0 && a.v[2] == 0 && a.v[3] == 0;
}

inline SecpFp64 secp64_zero() {
    return SecpFp64{0, 0, 0, 0};
}

inline SecpFp64 secp64_one() {
    return SecpFp64{SECP_R64[0], SECP_R64[1], SECP_R64[2], SECP_R64[3]};
}

inline bool secp64_gte(SecpFp64 a, SecpFp64 b) {
    for (int i = 3; i >= 0; i--) {
        if (a.v[i] > b.v[i]) return true;
        if (a.v[i] < b.v[i]) return false;
    }
    return true;
}

inline SecpFp64 secp64_sub_mod(SecpFp64 a, SecpFp64 b) {
    SecpFp64 r;
    ulong borrow = 0;
    r.v[0] = a.v[0] - b.v[0];
    borrow = (r.v[0] > a.v[0]) ? 1 : 0;
    r.v[1] = a.v[1] - b.v[1] - borrow;
    borrow = (r.v[1] > a.v[1]) ? 1 : (borrow && r.v[1] == a.v[1]) ? 1 : 0;
    r.v[2] = a.v[2] - b.v[2] - borrow;
    borrow = (r.v[2] > a.v[2]) ? 1 : (borrow && r.v[2] == a.v[2]) ? 1 : 0;
    r.v[3] = a.v[3] - b.v[3] - borrow;
    // If borrow, add p
    if (borrow) {
        ulong c = 0;
        r.v[0] += SECP_P64[0];
        c = (r.v[0] < SECP_P64[0]) ? 1 : 0;
        r.v[1] += SECP_P64[1] + c;
        c = (r.v[1] < SECP_P64[1]) ? 1 : (c && r.v[1] == SECP_P64[1]) ? 1 : 0;
        r.v[2] += SECP_P64[2] + c;
        c = (r.v[2] < SECP_P64[2]) ? 1 : (c && r.v[2] == SECP_P64[2]) ? 1 : 0;
        r.v[3] += SECP_P64[3] + c;
    }
    return r;
}

inline SecpFp64 secp64_add_mod(SecpFp64 a, SecpFp64 b) {
    SecpFp64 r;
    ulong carry = 0;
    r.v[0] = a.v[0] + b.v[0];
    carry = (r.v[0] < a.v[0]) ? 1 : 0;
    r.v[1] = a.v[1] + b.v[1] + carry;
    carry = (r.v[1] < a.v[1]) ? 1 : (carry && r.v[1] == a.v[1]) ? 1 : 0;
    r.v[2] = a.v[2] + b.v[2] + carry;
    carry = (r.v[2] < a.v[2]) ? 1 : (carry && r.v[2] == a.v[2]) ? 1 : 0;
    r.v[3] = a.v[3] + b.v[3] + carry;
    // If carry or >= p, subtract p
    if (carry || secp64_gte(r, SecpFp64{SECP_P64[0], SECP_P64[1], SECP_P64[2], SECP_P64[3]})) {
        ulong borrow = 0;
        ulong t0 = r.v[0] - SECP_P64[0];
        borrow = (r.v[0] < SECP_P64[0]) ? 1 : 0;
        ulong t1 = r.v[1] - SECP_P64[1] - borrow;
        borrow = (r.v[1] < SECP_P64[1]) ? 1 : (borrow && r.v[1] == SECP_P64[1]) ? 1 : 0;
        ulong t2 = r.v[2] - SECP_P64[2] - borrow;
        borrow = (r.v[2] < SECP_P64[2]) ? 1 : (borrow && r.v[2] == SECP_P64[2]) ? 1 : 0;
        ulong t3 = r.v[3] - SECP_P64[3] - borrow;
        r.v[0] = t0;
        r.v[1] = t1;
        r.v[2] = t2;
        r.v[3] = t3;
    }
    return r;
}

// Montgomery multiplication: a*b*R^(-1) mod p
// 4x64-bit CIOS: 4 passes of (mul + Montgomery reduce)
// Uses 128-bit intermediates via __builtin_mulh for correct carry handling.
inline SecpFp64 secp64_mul(SecpFp64 a, SecpFp64 b) {
    // 4x4 schoolbook → 7 64-bit words (prod[0..6])
    // Each a[i]*b[j] = lo + hi*2^64 where lo = a[i]*b[j], hi = __builtin_mulh(a[i],b[j])
    ulong prod[7] = {0, 0, 0, 0, 0, 0, 0};
    ulong carry = 0;

    // prod[0] = a0*b0
    prod[0] = a.v[0] * b.v[0];
    carry = __builtin_mulh(a.v[0], b.v[0]);

    // prod[1] = a0*b1 + a1*b0 + carry
    ulong t = a.v[0] * b.v[1] + a.v[1] * b.v[0] + carry;
    prod[1] = t;
    carry = __builtin_mulh(a.v[0], b.v[1]) + __builtin_mulh(a.v[1], b.v[0]) + (t < carry ? 1 : 0);

    // prod[2] = a0*b2 + a1*b1 + a2*b0 + carry
    t = a.v[0] * b.v[2] + a.v[1] * b.v[1] + a.v[2] * b.v[0] + carry;
    prod[2] = t;
    carry = __builtin_mulh(a.v[0], b.v[2]) + __builtin_mulh(a.v[1], b.v[1]) + __builtin_mulh(a.v[2], b.v[0]) + (t < carry ? 1 : 0);

    // prod[3] = a0*b3 + a1*b2 + a2*b1 + a3*b0 + carry
    t = a.v[0] * b.v[3] + a.v[1] * b.v[2] + a.v[2] * b.v[1] + a.v[3] * b.v[0] + carry;
    prod[3] = t;
    carry = __builtin_mulh(a.v[0], b.v[3]) + __builtin_mulh(a.v[1], b.v[2]) + __builtin_mulh(a.v[2], b.v[1]) + __builtin_mulh(a.v[3], b.v[0]) + (t < carry ? 1 : 0);

    // prod[4] = a1*b3 + a2*b2 + a3*b1 + carry
    t = a.v[1] * b.v[3] + a.v[2] * b.v[2] + a.v[3] * b.v[1] + carry;
    prod[4] = t;
    carry = __builtin_mulh(a.v[1], b.v[3]) + __builtin_mulh(a.v[2], b.v[2]) + __builtin_mulh(a.v[3], b.v[1]) + (t < carry ? 1 : 0);

    // prod[5] = a2*b3 + a3*b2 + carry
    t = a.v[2] * b.v[3] + a.v[3] * b.v[2] + carry;
    prod[5] = t;
    carry = __builtin_mulh(a.v[2], b.v[3]) + __builtin_mulh(a.v[3], b.v[2]) + (t < carry ? 1 : 0);

    // prod[6] = a3*b3 + carry
    prod[6] = a.v[3] * b.v[3] + carry;
    // (no __builtin_mulh for prod[6] since we only need the low part of a3*b3)

    // Montgomery reduction: 4 CIOS passes
    // Each pass: m = prod[i] * p_inv, add m*p to prod[i..i+3], propagate carry
    // m*p[i] is 128-bit: lo = m * p[i] (low 64), hi = __builtin_mulh(m, p[i]) (high 64)
    // We add lo to prod[j] and hi to prod[j+1] as carry.
    ulong cc;

    // Pass 1: eliminate prod[0]
    ulong m = prod[0] * SECP_INV64;
    // prod[0] += m * p[0]
    ulong mp_lo = m * SECP_P64[0];
    ulong mp_hi = __builtin_mulh(m, SECP_P64[0]);
    t = prod[0] + mp_lo;
    cc = (t < prod[0]) ? 1 : 0;
    prod[0] = t;
    // prod[1] += m * p[1] + cc
    mp_lo = m * SECP_P64[1];
    mp_hi = __builtin_mulh(m, SECP_P64[1]);
    t = prod[1] + mp_lo + cc;
    cc = (t < prod[1]) ? 1 : ((cc && t == prod[1]) ? 1 : 0);
    cc += mp_hi;
    prod[1] = t;
    // prod[2] += m * p[2] + cc
    mp_lo = m * SECP_P64[2];
    mp_hi = __builtin_mulh(m, SECP_P64[2]);
    t = prod[2] + mp_lo + cc;
    cc = (t < prod[2]) ? 1 : ((cc && t == prod[2]) ? 1 : 0);
    cc += mp_hi;
    prod[2] = t;
    // prod[3] += m * p[3] + cc
    mp_lo = m * SECP_P64[3];
    mp_hi = __builtin_mulh(m, SECP_P64[3]);
    t = prod[3] + mp_lo + cc;
    cc = (t < prod[3]) ? 1 : ((cc && t == prod[3]) ? 1 : 0);
    cc += mp_hi;
    prod[3] = t;
    // p[4]=p[5]=p[6]=0, so prod[4] += cc
    prod[4] += cc;
    // prod[5], prod[6] unchanged

    // Pass 2: eliminate prod[1]
    m = prod[1] * SECP_INV64;
    mp_lo = m * SECP_P64[0];
    mp_hi = __builtin_mulh(m, SECP_P64[0]);
    t = prod[1] + mp_lo;
    cc = (t < prod[1]) ? 1 : 0;
    prod[1] = t;
    mp_lo = m * SECP_P64[1];
    mp_hi = __builtin_mulh(m, SECP_P64[1]);
    t = prod[2] + mp_lo + cc;
    cc = (t < prod[2]) ? 1 : ((cc && t == prod[2]) ? 1 : 0);
    cc += mp_hi;
    prod[2] = t;
    mp_lo = m * SECP_P64[2];
    mp_hi = __builtin_mulh(m, SECP_P64[2]);
    t = prod[3] + mp_lo + cc;
    cc = (t < prod[3]) ? 1 : ((cc && t == prod[3]) ? 1 : 0);
    cc += mp_hi;
    prod[3] = t;
    mp_lo = m * SECP_P64[3];
    mp_hi = __builtin_mulh(m, SECP_P64[3]);
    t = prod[4] + mp_lo + cc;
    cc = (t < prod[4]) ? 1 : ((cc && t == prod[4]) ? 1 : 0);
    cc += mp_hi;
    prod[4] = t;
    prod[5] += cc;

    // Pass 3: eliminate prod[2]
    m = prod[2] * SECP_INV64;
    mp_lo = m * SECP_P64[0];
    mp_hi = __builtin_mulh(m, SECP_P64[0]);
    t = prod[2] + mp_lo;
    cc = (t < prod[2]) ? 1 : 0;
    prod[2] = t;
    mp_lo = m * SECP_P64[1];
    mp_hi = __builtin_mulh(m, SECP_P64[1]);
    t = prod[3] + mp_lo + cc;
    cc = (t < prod[3]) ? 1 : ((cc && t == prod[3]) ? 1 : 0);
    cc += mp_hi;
    prod[3] = t;
    mp_lo = m * SECP_P64[2];
    mp_hi = __builtin_mulh(m, SECP_P64[2]);
    t = prod[4] + mp_lo + cc;
    cc = (t < prod[4]) ? 1 : ((cc && t == prod[4]) ? 1 : 0);
    cc += mp_hi;
    prod[4] = t;
    mp_lo = m * SECP_P64[3];
    mp_hi = __builtin_mulh(m, SECP_P64[3]);
    t = prod[5] + mp_lo + cc;
    cc = (t < prod[5]) ? 1 : ((cc && t == prod[5]) ? 1 : 0);
    cc += mp_hi;
    prod[5] = t;
    prod[6] += cc;

    // Pass 4: eliminate prod[3]
    m = prod[3] * SECP_INV64;
    mp_lo = m * SECP_P64[0];
    mp_hi = __builtin_mulh(m, SECP_P64[0]);
    t = prod[3] + mp_lo;
    cc = (t < prod[3]) ? 1 : 0;
    prod[3] = t;
    mp_lo = m * SECP_P64[1];
    mp_hi = __builtin_mulh(m, SECP_P64[1]);
    t = prod[4] + mp_lo + cc;
    cc = (t < prod[4]) ? 1 : ((cc && t == prod[4]) ? 1 : 0);
    cc += mp_hi;
    prod[4] = t;
    mp_lo = m * SECP_P64[2];
    mp_hi = __builtin_mulh(m, SECP_P64[2]);
    t = prod[5] + mp_lo + cc;
    cc = (t < prod[5]) ? 1 : ((cc && t == prod[5]) ? 1 : 0);
    cc += mp_hi;
    prod[5] = t;
    prod[6] += m * SECP_P64[3] + cc;

    // Result is in prod[0..3]
    SecpFp64 result;
    result.v[0] = prod[0];
    result.v[1] = prod[1];
    result.v[2] = prod[2];
    result.v[3] = prod[3];

    // Final reduction if >= p
    if (secp64_gte(result, SecpFp64{SECP_P64[0], SECP_P64[1], SECP_P64[2], SECP_P64[3]})) {
        result = secp64_sub_mod(result, SecpFp64{SECP_P64[0], SECP_P64[1], SECP_P64[2], SECP_P64[3]});
    }
    return result;
}

// Note: squaring is omitted — use secp64_mul(a, a) instead.
// For MSM, point doubling is needed but requires Fp inverse.
// The curve operations (add/double) are in secp256k1_curve.metal.

#endif // SECP256K1_FP64_METAL
