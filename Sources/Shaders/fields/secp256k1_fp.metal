// secp256k1 base field Fp arithmetic for Metal GPU
//
// p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
//   = 2^256 - 2^32 - 977
// Field elements as 8x32-bit limbs in Montgomery form (little-endian).

#ifndef SECP256K1_FP_METAL
#define SECP256K1_FP_METAL

#include <metal_stdlib>
using namespace metal;

#define SECP_LIMBS 8

struct SecpFp {
    uint v[SECP_LIMBS]; // 256-bit value as 8x32-bit limbs (little-endian)
};

struct SecpPointAffine {
    SecpFp x;
    SecpFp y;
};

struct SecpPointProjective {
    SecpFp x;
    SecpFp y;
    SecpFp z;
};

// secp256k1 base field modulus p (little-endian 32-bit limbs)
constant uint SECP_P[8] = {
    0xfffffc2f, 0xfffffffe, 0xffffffff, 0xffffffff,
    0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff
};

// Montgomery parameter: R^2 mod p
constant uint SECP_R2[8] = {
    0x000e90a1, 0x000007a2, 0x00000001, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000
};

// Montgomery inverse: -p^(-1) mod 2^32
constant uint SECP_INV = 0xd2253531u;

// --- 256-bit Arithmetic ---

SecpFp secp_add_raw(SecpFp a, SecpFp b, thread uint &carry) {
    SecpFp r;
    ulong c = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        c += ulong(a.v[i]) + ulong(b.v[i]);
        r.v[i] = uint(c & 0xFFFFFFFF);
        c >>= 32;
    }
    carry = uint(c);
    return r;
}

SecpFp secp_sub_raw(SecpFp a, SecpFp b, thread uint &borrow) {
    SecpFp r;
    long c = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        c += long(a.v[i]) - long(b.v[i]);
        r.v[i] = uint(c & 0xFFFFFFFF);
        c >>= 32;
    }
    borrow = (c < 0) ? 1u : 0u;
    return r;
}

bool secp_gte(SecpFp a, SecpFp b) {
    for (int i = 7; i >= 0; i--) {
        if (a.v[i] > b.v[i]) return true;
        if (a.v[i] < b.v[i]) return false;
    }
    return true;
}

SecpFp secp_modulus() {
    SecpFp r;
    for (int i = 0; i < 8; i++) r.v[i] = SECP_P[i];
    return r;
}

SecpFp secp_add(SecpFp a, SecpFp b) {
    uint carry;
    SecpFp r = secp_add_raw(a, b, carry);
    SecpFp p = secp_modulus();
    if (carry || secp_gte(r, p)) {
        uint borrow;
        r = secp_sub_raw(r, p, borrow);
    }
    return r;
}

SecpFp secp_sub(SecpFp a, SecpFp b) {
    uint borrow;
    SecpFp r = secp_sub_raw(a, b, borrow);
    if (borrow) {
        uint carry;
        r = secp_add_raw(r, secp_modulus(), carry);
    }
    return r;
}

bool secp_is_zero(SecpFp a) {
    for (int i = 0; i < 8; i++) {
        if (a.v[i] != 0) return false;
    }
    return true;
}

SecpFp secp_zero() {
    SecpFp r;
    for (int i = 0; i < 8; i++) r.v[i] = 0;
    return r;
}

// R mod p in 32-bit limbs (Montgomery form of 1)
SecpFp secp_one() {
    SecpFp r;
    r.v[0] = 0x000003d1; r.v[1] = 0x00000001;
    r.v[2] = 0x00000000; r.v[3] = 0x00000000;
    r.v[4] = 0x00000000; r.v[5] = 0x00000000;
    r.v[6] = 0x00000000; r.v[7] = 0x00000000;
    return r;
}

SecpFp secp_double(SecpFp a) {
    SecpFp r;
    uint carry = 0;
    for (int i = 0; i < 8; i++) {
        uint doubled = (a.v[i] << 1) | carry;
        carry = a.v[i] >> 31;
        r.v[i] = doubled;
    }
    if (carry || secp_gte(r, secp_modulus())) {
        uint borrow;
        r = secp_sub_raw(r, secp_modulus(), borrow);
    }
    return r;
}

// Montgomery multiplication (CIOS with 32-bit limbs)
SecpFp secp_mul(SecpFp a, SecpFp b) {
    uint t[10];
    #pragma unroll
    for (int i = 0; i < 10; i++) t[i] = 0;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        ulong carry = 0;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            carry += ulong(t[j]) + ulong(a.v[i]) * ulong(b.v[j]);
            t[j] = uint(carry & 0xFFFFFFFF);
            carry >>= 32;
        }
        ulong ext = ulong(t[8]) + carry;
        t[8] = uint(ext & 0xFFFFFFFF);
        t[9] = uint(ext >> 32);

        uint m = t[0] * SECP_INV;
        carry = ulong(t[0]) + ulong(m) * ulong(SECP_P[0]);
        carry >>= 32;
        #pragma unroll
        for (int j = 1; j < 8; j++) {
            carry += ulong(t[j]) + ulong(m) * ulong(SECP_P[j]);
            t[j - 1] = uint(carry & 0xFFFFFFFF);
            carry >>= 32;
        }
        ext = ulong(t[8]) + carry;
        t[7] = uint(ext & 0xFFFFFFFF);
        t[8] = t[9] + uint(ext >> 32);
        t[9] = 0;
    }

    SecpFp r;
    for (int i = 0; i < 8; i++) r.v[i] = t[i];
    if (t[8] != 0 || secp_gte(r, secp_modulus())) {
        uint borrow;
        r = secp_sub_raw(r, secp_modulus(), borrow);
    }
    return r;
}

// Montgomery squaring (SOS: Separated Operand Scanning)
SecpFp secp_sqr(SecpFp a) {
    uint t[17];
    for (int i = 0; i < 17; i++) t[i] = 0;

    // Cross terms: 2 * sum(a[i]*a[j]) for i < j
    for (int i = 0; i < 7; i++) {
        ulong carry = 0;
        for (int j = i + 1; j < 8; j++) {
            carry += ulong(t[i + j]) + ulong(a.v[i]) * ulong(a.v[j]);
            t[i + j] = uint(carry & 0xFFFFFFFF);
            carry >>= 32;
        }
        t[i + 8] += uint(carry);
    }

    // Double the cross terms
    uint top_carry = 0;
    for (int i = 1; i < 16; i++) {
        ulong doubled = (ulong(t[i]) << 1) | ulong(top_carry);
        t[i] = uint(doubled & 0xFFFFFFFF);
        top_carry = uint(doubled >> 32);
    }

    // Add diagonal terms: a[i]*a[i]
    ulong carry = 0;
    for (int i = 0; i < 8; i++) {
        carry += ulong(t[2*i]) + ulong(a.v[i]) * ulong(a.v[i]);
        t[2*i] = uint(carry & 0xFFFFFFFF);
        carry >>= 32;
        carry += ulong(t[2*i + 1]);
        t[2*i + 1] = uint(carry & 0xFFFFFFFF);
        carry >>= 32;
    }

    // Montgomery reduction
    for (int i = 0; i < 8; i++) {
        uint m = t[i] * SECP_INV;
        ulong c = ulong(t[i]) + ulong(m) * ulong(SECP_P[0]);
        c >>= 32;
        for (int j = 1; j < 8; j++) {
            c += ulong(t[i + j]) + ulong(m) * ulong(SECP_P[j]);
            t[i + j] = uint(c & 0xFFFFFFFF);
            c >>= 32;
        }
        for (int j = i + 8; j < 16; j++) {
            c += ulong(t[j]);
            t[j] = uint(c & 0xFFFFFFFF);
            c >>= 32;
            if (c == 0) break;
        }
    }

    SecpFp r;
    for (int i = 0; i < 8; i++) r.v[i] = t[i + 8];
    if (secp_gte(r, secp_modulus())) {
        uint borrow;
        r = secp_sub_raw(r, secp_modulus(), borrow);
    }
    return r;
}

// Negation: -a mod p
SecpFp secp_neg(SecpFp a) {
    if (secp_is_zero(a)) return a;
    uint borrow;
    return secp_sub_raw(secp_modulus(), a, borrow);
}

// Convert to Montgomery form: a * R^2 * R^(-1) = a * R mod p
SecpFp secp_to_mont(SecpFp a) {
    SecpFp r2;
    for (int i = 0; i < 8; i++) r2.v[i] = SECP_R2[i];
    return secp_mul(a, r2);
}

// Convert from Montgomery form: a * 1 * R^(-1) = a * R^(-1) mod p
SecpFp secp_from_mont(SecpFp a) {
    SecpFp one;
    for (int i = 0; i < 8; i++) one.v[i] = 0;
    one.v[0] = 1;
    return secp_mul(a, one);
}

// Modular exponentiation
SecpFp secp_pow(SecpFp base, const thread uint exp[8]) {
    SecpFp result = secp_one();
    SecpFp b = base;
    for (int i = 0; i < 8; i++) {
        uint word = exp[i];
        for (int bit = 0; bit < 32; bit++) {
            if (word & 1u) {
                result = secp_mul(result, b);
            }
            b = secp_mul(b, b);
            word >>= 1;
        }
    }
    return result;
}

// Modular inverse via Fermat's little theorem: a^(p-2) mod p
SecpFp secp_inv(SecpFp a) {
    uint exp[8] = {
        0xfffffc2du, 0xfffffffeu, 0xffffffffu, 0xffffffffu,
        0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu
    };
    return secp_pow(a, exp);
}

#endif // SECP256K1_FP_METAL
