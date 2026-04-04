// Ed25519 base field Fp arithmetic for Metal GPU
//
// p = 2^255 - 19 = 0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffed
// Field elements as 8x32-bit limbs in Montgomery form (little-endian).

#ifndef ED25519_FP_METAL
#define ED25519_FP_METAL

#include <metal_stdlib>
using namespace metal;

#define ED_LIMBS 8

struct EdFp {
    uint v[ED_LIMBS]; // 256-bit value as 8x32-bit limbs (little-endian)
};

struct EdPointAffine {
    EdFp x;
    EdFp y;
};

struct EdPointExtended {
    EdFp x;
    EdFp y;
    EdFp z;
    EdFp t;
};

// Ed25519 base field modulus p (little-endian 32-bit limbs)
// p = 2^255 - 19
constant uint ED_P[8] = {
    0xffffffed, 0xffffffff, 0xffffffff, 0xffffffff,
    0xffffffff, 0xffffffff, 0xffffffff, 0x7fffffff
};

// Montgomery parameter: R^2 mod p = 1444 = 0x5A4
constant uint ED_R2[8] = {
    0x000005a4, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000
};

// R mod p = 38 = 0x26 (Montgomery form of 1)
constant uint ED_R[8] = {
    0x00000026, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000
};

// Montgomery inverse: -p^(-1) mod 2^32
// p[0] = 0xffffffed
// We need inv such that p[0] * inv ≡ -1 (mod 2^32)
// 0xffffffed * inv ≡ 0xffffffff (mod 2^32)
// inv = 0x286bca1b (low 32 bits of 64-bit INV 0x86bca1af286bca1b)
constant uint ED_INV = 0x286bca1bu;

// --- 256-bit Arithmetic ---

EdFp ed_add_raw(EdFp a, EdFp b, thread uint &carry) {
    EdFp r;
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

EdFp ed_sub_raw(EdFp a, EdFp b, thread uint &borrow) {
    EdFp r;
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

bool ed_gte(EdFp a, EdFp b) {
    for (int i = 7; i >= 0; i--) {
        if (a.v[i] > b.v[i]) return true;
        if (a.v[i] < b.v[i]) return false;
    }
    return true;
}

EdFp ed_modulus() {
    EdFp r;
    for (int i = 0; i < 8; i++) r.v[i] = ED_P[i];
    return r;
}

EdFp ed_add(EdFp a, EdFp b) {
    uint carry;
    EdFp r = ed_add_raw(a, b, carry);
    EdFp p = ed_modulus();
    if (carry || ed_gte(r, p)) {
        uint borrow;
        r = ed_sub_raw(r, p, borrow);
    }
    return r;
}

EdFp ed_sub(EdFp a, EdFp b) {
    uint borrow;
    EdFp r = ed_sub_raw(a, b, borrow);
    if (borrow) {
        uint carry;
        r = ed_add_raw(r, ed_modulus(), carry);
    }
    return r;
}

bool ed_is_zero(EdFp a) {
    for (int i = 0; i < 8; i++) {
        if (a.v[i] != 0) return false;
    }
    return true;
}

EdFp ed_zero() {
    EdFp r;
    for (int i = 0; i < 8; i++) r.v[i] = 0;
    return r;
}

EdFp ed_one() {
    EdFp r;
    r.v[0] = 0x00000026;
    for (int i = 1; i < 8; i++) r.v[i] = 0;
    return r;
}

EdFp ed_double(EdFp a) {
    EdFp r;
    uint carry = 0;
    for (int i = 0; i < 8; i++) {
        uint doubled = (a.v[i] << 1) | carry;
        carry = a.v[i] >> 31;
        r.v[i] = doubled;
    }
    if (carry || ed_gte(r, ed_modulus())) {
        uint borrow;
        r = ed_sub_raw(r, ed_modulus(), borrow);
    }
    return r;
}

// Montgomery multiplication (CIOS with 32-bit limbs)
EdFp ed_mul(EdFp a, EdFp b) {
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

        uint m = t[0] * ED_INV;
        carry = ulong(t[0]) + ulong(m) * ulong(ED_P[0]);
        carry >>= 32;
        #pragma unroll
        for (int j = 1; j < 8; j++) {
            carry += ulong(t[j]) + ulong(m) * ulong(ED_P[j]);
            t[j - 1] = uint(carry & 0xFFFFFFFF);
            carry >>= 32;
        }
        ext = ulong(t[8]) + carry;
        t[7] = uint(ext & 0xFFFFFFFF);
        t[8] = t[9] + uint(ext >> 32);
        t[9] = 0;
    }

    EdFp r;
    for (int i = 0; i < 8; i++) r.v[i] = t[i];
    if (t[8] != 0 || ed_gte(r, ed_modulus())) {
        uint borrow;
        r = ed_sub_raw(r, ed_modulus(), borrow);
    }
    return r;
}

// Montgomery squaring
EdFp ed_sqr(EdFp a) {
    return ed_mul(a, a);
}

// Negation: -a mod p
EdFp ed_neg(EdFp a) {
    if (ed_is_zero(a)) return a;
    uint borrow;
    return ed_sub_raw(ed_modulus(), a, borrow);
}

// Convert to Montgomery form
EdFp ed_to_mont(EdFp a) {
    EdFp r2;
    for (int i = 0; i < 8; i++) r2.v[i] = ED_R2[i];
    return ed_mul(a, r2);
}

// Convert from Montgomery form
EdFp ed_from_mont(EdFp a) {
    EdFp one;
    for (int i = 0; i < 8; i++) one.v[i] = 0;
    one.v[0] = 1;
    return ed_mul(a, one);
}

// Modular exponentiation
EdFp ed_pow(EdFp base, const thread uint exp[8]) {
    EdFp result = ed_one();
    EdFp b = base;
    for (int i = 0; i < 8; i++) {
        uint word = exp[i];
        for (int bit = 0; bit < 32; bit++) {
            if (word & 1u) {
                result = ed_mul(result, b);
            }
            b = ed_mul(b, b);
            word >>= 1;
        }
    }
    return result;
}

// Modular inverse via Fermat's little theorem: a^(p-2) mod p
EdFp ed_inv(EdFp a) {
    // p - 2 = 0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffeb
    uint exp[8] = {
        0xffffffebu, 0xffffffffu, 0xffffffffu, 0xffffffffu,
        0xffffffffu, 0xffffffffu, 0xffffffffu, 0x7fffffffu
    };
    return ed_pow(a, exp);
}

#endif // ED25519_FP_METAL
