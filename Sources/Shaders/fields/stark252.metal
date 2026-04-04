// StarkNet/Cairo native field (Stark252) arithmetic for Metal GPU
//
// p = 2^251 + 17 * 2^192 + 1
//   = 3618502788666131213697322783095070105623107215331596699973092056135872020481
// Field elements as 8x32-bit limbs in Montgomery form (little-endian).
// TWO_ADICITY = 192 (p-1 = 2^192 * t, t odd)

#ifndef STARK252_METAL
#define STARK252_METAL

#include <metal_stdlib>
using namespace metal;

struct Stark252 {
    uint v[8]; // 256-bit value as 8x32-bit limbs (little-endian)
};

// Stark252 field modulus p (little-endian 32-bit limbs)
constant uint STARK252_P[8] = {
    0x00000001, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000011, 0x08000000
};

// Montgomery parameter: R^2 mod p
constant uint STARK252_R2[8] = {
    0x7e000401, 0xfffffd73, 0x330fffff, 0x00000001,
    0xff6f8000, 0xffffffff, 0x5e008810, 0x07ffd4ab
};

// Montgomery inverse: -p^(-1) mod 2^32
constant uint STARK252_INV = 0xFFFFFFFFu;

// --- 256-bit Arithmetic ---

Stark252 stark252_add_raw(Stark252 a, Stark252 b, thread uint &carry) {
    Stark252 r;
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

Stark252 stark252_sub_raw(Stark252 a, Stark252 b, thread uint &borrow) {
    Stark252 r;
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

bool stark252_gte(Stark252 a, Stark252 b) {
    for (int i = 7; i >= 0; i--) {
        if (a.v[i] > b.v[i]) return true;
        if (a.v[i] < b.v[i]) return false;
    }
    return true;
}

Stark252 stark252_modulus() {
    Stark252 r;
    for (int i = 0; i < 8; i++) r.v[i] = STARK252_P[i];
    return r;
}

Stark252 stark252_add(Stark252 a, Stark252 b) {
    uint carry;
    Stark252 r = stark252_add_raw(a, b, carry);
    Stark252 p = stark252_modulus();
    if (carry || stark252_gte(r, p)) {
        uint borrow;
        r = stark252_sub_raw(r, p, borrow);
    }
    return r;
}

// Lazy addition: no modular reduction.
Stark252 stark252_add_lazy(Stark252 a, Stark252 b) {
    uint carry;
    return stark252_add_raw(a, b, carry);
}

// Reduce value to [0, p).
Stark252 stark252_reduce(Stark252 a) {
    if (stark252_gte(a, stark252_modulus())) {
        uint borrow;
        a = stark252_sub_raw(a, stark252_modulus(), borrow);
    }
    return a;
}

Stark252 stark252_sub(Stark252 a, Stark252 b) {
    uint borrow;
    Stark252 r = stark252_sub_raw(a, b, borrow);
    if (borrow) {
        uint carry;
        r = stark252_add_raw(r, stark252_modulus(), carry);
    }
    return r;
}

bool stark252_is_zero(Stark252 a) {
    for (int i = 0; i < 8; i++) {
        if (a.v[i] != 0) return false;
    }
    return true;
}

Stark252 stark252_zero() {
    Stark252 r;
    for (int i = 0; i < 8; i++) r.v[i] = 0;
    return r;
}

// R mod p in 32-bit limbs (Montgomery form of 1)
Stark252 stark252_one() {
    Stark252 r;
    r.v[0] = 0xffffffe1; r.v[1] = 0xffffffff;
    r.v[2] = 0xffffffff; r.v[3] = 0xffffffff;
    r.v[4] = 0xffffffff; r.v[5] = 0xffffffff;
    r.v[6] = 0xfffffdf0; r.v[7] = 0x07ffffff;
    return r;
}

// Specialized doubling
Stark252 stark252_double(Stark252 a) {
    Stark252 r;
    uint carry = 0;
    for (int i = 0; i < 8; i++) {
        uint doubled = (a.v[i] << 1) | carry;
        carry = a.v[i] >> 31;
        r.v[i] = doubled;
    }
    if (carry || stark252_gte(r, stark252_modulus())) {
        uint borrow;
        r = stark252_sub_raw(r, stark252_modulus(), borrow);
    }
    return r;
}

// Montgomery multiplication (CIOS with 32-bit limbs)
Stark252 stark252_mul(Stark252 a, Stark252 b) {
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

        uint m = t[0] * STARK252_INV;
        carry = ulong(t[0]) + ulong(m) * ulong(STARK252_P[0]);
        carry >>= 32;
        #pragma unroll
        for (int j = 1; j < 8; j++) {
            carry += ulong(t[j]) + ulong(m) * ulong(STARK252_P[j]);
            t[j - 1] = uint(carry & 0xFFFFFFFF);
            carry >>= 32;
        }
        ext = ulong(t[8]) + carry;
        t[7] = uint(ext & 0xFFFFFFFF);
        t[8] = t[9] + uint(ext >> 32);
        t[9] = 0;
    }

    Stark252 r;
    for (int i = 0; i < 8; i++) r.v[i] = t[i];
    if (t[8] != 0 || stark252_gte(r, stark252_modulus())) {
        uint borrow;
        r = stark252_sub_raw(r, stark252_modulus(), borrow);
    }
    return r;
}

// Montgomery squaring (SOS)
Stark252 stark252_sqr(Stark252 a) {
    uint t[17];
    for (int i = 0; i < 17; i++) t[i] = 0;

    for (int i = 0; i < 7; i++) {
        ulong carry = 0;
        for (int j = i + 1; j < 8; j++) {
            carry += ulong(t[i + j]) + ulong(a.v[i]) * ulong(a.v[j]);
            t[i + j] = uint(carry & 0xFFFFFFFF);
            carry >>= 32;
        }
        t[i + 8] += uint(carry);
    }

    uint top_carry = 0;
    for (int i = 1; i < 16; i++) {
        ulong doubled = (ulong(t[i]) << 1) | ulong(top_carry);
        t[i] = uint(doubled & 0xFFFFFFFF);
        top_carry = uint(doubled >> 32);
    }

    ulong carry = 0;
    for (int i = 0; i < 8; i++) {
        carry += ulong(t[2*i]) + ulong(a.v[i]) * ulong(a.v[i]);
        t[2*i] = uint(carry & 0xFFFFFFFF);
        carry >>= 32;
        carry += ulong(t[2*i + 1]);
        t[2*i + 1] = uint(carry & 0xFFFFFFFF);
        carry >>= 32;
    }

    for (int i = 0; i < 8; i++) {
        uint m = t[i] * STARK252_INV;
        ulong c = ulong(t[i]) + ulong(m) * ulong(STARK252_P[0]);
        c >>= 32;
        for (int j = 1; j < 8; j++) {
            c += ulong(t[i + j]) + ulong(m) * ulong(STARK252_P[j]);
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

    Stark252 r;
    for (int i = 0; i < 8; i++) r.v[i] = t[i + 8];
    if (stark252_gte(r, stark252_modulus())) {
        uint borrow;
        r = stark252_sub_raw(r, stark252_modulus(), borrow);
    }
    return r;
}

// Modular exponentiation by squaring
Stark252 stark252_pow(Stark252 base, const thread uint exp[8]) {
    Stark252 result = stark252_one();
    Stark252 b = base;
    for (int i = 0; i < 8; i++) {
        uint word = exp[i];
        for (int bit = 0; bit < 32; bit++) {
            if (word & 1u) {
                result = stark252_mul(result, b);
            }
            b = stark252_mul(b, b);
            word >>= 1;
        }
    }
    return result;
}

// Modular inverse via Fermat's little theorem: a^(p-2) mod p
Stark252 stark252_inv(Stark252 a) {
    // p - 2 in 32-bit limbs (little-endian)
    uint exp[8] = {
        0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu,
        0xffffffffu, 0xffffffffu, 0x00000010u, 0x08000000u
    };
    return stark252_pow(a, exp);
}

#endif // STARK252_METAL
