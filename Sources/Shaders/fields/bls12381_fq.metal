// BLS12-381 base field Fp arithmetic for Metal GPU
//
// p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
// 381-bit prime, field elements as 12x32-bit limbs in Montgomery form (little-endian).

#ifndef BLS12381_FQ_METAL
#define BLS12381_FQ_METAL

#include <metal_stdlib>
using namespace metal;

#define FP381_LIMBS 12

struct Fp381 {
    uint v[FP381_LIMBS]; // 384-bit value as 12x32-bit limbs (little-endian)
};

struct Point381Affine {
    Fp381 x;
    Fp381 y;
};

struct Point381Projective {
    Fp381 x;
    Fp381 y;
    Fp381 z;
};

// BLS12-381 base field modulus p (little-endian 32-bit limbs)
// p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
constant uint FP381_P[FP381_LIMBS] = {
    0xffffaaab, 0xb9feffff, 0xb153ffff, 0x1eabfffe,
    0xf6b0f624, 0x6730d2a0, 0xf38512bf, 0x64774b84,
    0x434bacd7, 0x4b1ba7b6, 0x397fe69a, 0x1a0111ea
};

// Montgomery parameter: R^2 mod p (2^768 mod p)
constant uint FP381_R2[FP381_LIMBS] = {
    0x1c341746, 0xf4df1f34, 0x09d104f1, 0x0a76e6a6,
    0x4c95b6d5, 0x8de5476c, 0x939d83c0, 0x67eb88a9,
    0xb519952d, 0x9a793e85, 0x92cae3aa, 0x11988fe5
};

// R mod p (Montgomery form of 1): 2^384 mod p
constant uint FP381_ONE[FP381_LIMBS] = {
    0x0002fffd, 0x76090000, 0xc40c0002, 0xebf4000b,
    0x53c758ba, 0x5f489857, 0x70525745, 0x77ce5853,
    0xa256ec6d, 0x5c071a97, 0xfa80e493, 0x15f65ec3
};

// Montgomery inverse: -p^(-1) mod 2^32
// p mod 2^32 = 0xffffaaab, inv = 0xfffcfffd (since p*inv = -1 mod 2^32)
constant uint FP381_INV = 0xfffcfffdu;

// --- 384-bit Arithmetic ---

Fp381 fp381_add_raw(Fp381 a, Fp381 b, thread uint &carry) {
    Fp381 r;
    ulong c = 0;
    #pragma unroll
    for (int i = 0; i < FP381_LIMBS; i++) {
        c += ulong(a.v[i]) + ulong(b.v[i]);
        r.v[i] = uint(c & 0xFFFFFFFF);
        c >>= 32;
    }
    carry = uint(c);
    return r;
}

Fp381 fp381_sub_raw(Fp381 a, Fp381 b, thread uint &borrow) {
    Fp381 r;
    long c = 0;
    #pragma unroll
    for (int i = 0; i < FP381_LIMBS; i++) {
        c += long(a.v[i]) - long(b.v[i]);
        r.v[i] = uint(c & 0xFFFFFFFF);
        c >>= 32;
    }
    borrow = (c < 0) ? 1u : 0u;
    return r;
}

bool fp381_gte(Fp381 a, Fp381 b) {
    for (int i = FP381_LIMBS - 1; i >= 0; i--) {
        if (a.v[i] > b.v[i]) return true;
        if (a.v[i] < b.v[i]) return false;
    }
    return true;
}

Fp381 fp381_modulus() {
    Fp381 r;
    for (int i = 0; i < FP381_LIMBS; i++) r.v[i] = FP381_P[i];
    return r;
}

Fp381 fp381_add(Fp381 a, Fp381 b) {
    uint carry;
    Fp381 r = fp381_add_raw(a, b, carry);
    if (carry || fp381_gte(r, fp381_modulus())) {
        uint borrow;
        r = fp381_sub_raw(r, fp381_modulus(), borrow);
    }
    return r;
}

Fp381 fp381_sub(Fp381 a, Fp381 b) {
    uint borrow;
    Fp381 r = fp381_sub_raw(a, b, borrow);
    if (borrow) {
        uint carry;
        r = fp381_add_raw(r, fp381_modulus(), carry);
    }
    return r;
}

bool fp381_is_zero(Fp381 a) {
    for (int i = 0; i < FP381_LIMBS; i++) {
        if (a.v[i] != 0) return false;
    }
    return true;
}

Fp381 fp381_neg(Fp381 a) {
    if (fp381_is_zero(a)) return a;
    return fp381_sub(fp381_modulus(), a);
}

Fp381 fp381_zero() {
    Fp381 r;
    for (int i = 0; i < FP381_LIMBS; i++) r.v[i] = 0;
    return r;
}

Fp381 fp381_one() {
    Fp381 r;
    for (int i = 0; i < FP381_LIMBS; i++) r.v[i] = FP381_ONE[i];
    return r;
}

// Specialized doubling: left-shift by 1
Fp381 fp381_double(Fp381 a) {
    Fp381 r;
    uint carry = 0;
    for (int i = 0; i < FP381_LIMBS; i++) {
        uint doubled = (a.v[i] << 1) | carry;
        carry = a.v[i] >> 31;
        r.v[i] = doubled;
    }
    if (carry || fp381_gte(r, fp381_modulus())) {
        uint borrow;
        r = fp381_sub_raw(r, fp381_modulus(), borrow);
    }
    return r;
}

// --- Montgomery Multiplication ---
// CIOS (Coarsely Integrated Operand Scanning) with 32-bit limbs for 12-limb field

Fp381 fp381_mul(Fp381 a, Fp381 b) {
    uint t[FP381_LIMBS + 2];
    #pragma unroll
    for (int i = 0; i < FP381_LIMBS + 2; i++) t[i] = 0;

    #pragma unroll
    for (int i = 0; i < FP381_LIMBS; i++) {
        ulong carry = 0;
        #pragma unroll
        for (int j = 0; j < FP381_LIMBS; j++) {
            carry += ulong(t[j]) + ulong(a.v[i]) * ulong(b.v[j]);
            t[j] = uint(carry & 0xFFFFFFFF);
            carry >>= 32;
        }
        ulong ext = ulong(t[FP381_LIMBS]) + carry;
        t[FP381_LIMBS] = uint(ext & 0xFFFFFFFF);
        t[FP381_LIMBS + 1] = uint(ext >> 32);

        uint m = t[0] * FP381_INV;
        carry = ulong(t[0]) + ulong(m) * ulong(FP381_P[0]);
        carry >>= 32;
        #pragma unroll
        for (int j = 1; j < FP381_LIMBS; j++) {
            carry += ulong(t[j]) + ulong(m) * ulong(FP381_P[j]);
            t[j - 1] = uint(carry & 0xFFFFFFFF);
            carry >>= 32;
        }
        ext = ulong(t[FP381_LIMBS]) + carry;
        t[FP381_LIMBS - 1] = uint(ext & 0xFFFFFFFF);
        t[FP381_LIMBS] = t[FP381_LIMBS + 1] + uint(ext >> 32);
        t[FP381_LIMBS + 1] = 0;
    }

    Fp381 r;
    for (int i = 0; i < FP381_LIMBS; i++) r.v[i] = t[i];
    if (t[FP381_LIMBS] != 0 || fp381_gte(r, fp381_modulus())) {
        uint borrow;
        r = fp381_sub_raw(r, fp381_modulus(), borrow);
    }
    return r;
}

// SOS squaring: exploits a^2 symmetry for fewer multiplications
Fp381 fp381_sqr(Fp381 a) {
    uint t[2 * FP381_LIMBS + 1];
    for (int i = 0; i < 2 * FP381_LIMBS + 1; i++) t[i] = 0;

    // Cross terms: a[i] * a[j] for i < j
    for (int i = 0; i < FP381_LIMBS - 1; i++) {
        ulong carry = 0;
        for (int j = i + 1; j < FP381_LIMBS; j++) {
            carry += ulong(t[i + j]) + ulong(a.v[i]) * ulong(a.v[j]);
            t[i + j] = uint(carry & 0xFFFFFFFF);
            carry >>= 32;
        }
        t[i + FP381_LIMBS] += uint(carry);
    }

    // Double the cross terms
    uint top_carry = 0;
    for (int i = 1; i < 2 * FP381_LIMBS; i++) {
        ulong doubled = (ulong(t[i]) << 1) | ulong(top_carry);
        t[i] = uint(doubled & 0xFFFFFFFF);
        top_carry = uint(doubled >> 32);
    }

    // Add diagonal terms: a[i]^2
    ulong carry = 0;
    for (int i = 0; i < FP381_LIMBS; i++) {
        carry += ulong(t[2*i]) + ulong(a.v[i]) * ulong(a.v[i]);
        t[2*i] = uint(carry & 0xFFFFFFFF);
        carry >>= 32;
        carry += ulong(t[2*i + 1]);
        t[2*i + 1] = uint(carry & 0xFFFFFFFF);
        carry >>= 32;
    }

    // Montgomery reduction
    for (int i = 0; i < FP381_LIMBS; i++) {
        uint m = t[i] * FP381_INV;
        ulong c = ulong(t[i]) + ulong(m) * ulong(FP381_P[0]);
        c >>= 32;
        for (int j = 1; j < FP381_LIMBS; j++) {
            c += ulong(t[i + j]) + ulong(m) * ulong(FP381_P[j]);
            t[i + j] = uint(c & 0xFFFFFFFF);
            c >>= 32;
        }
        for (int j = i + FP381_LIMBS; j < 2 * FP381_LIMBS; j++) {
            c += ulong(t[j]);
            t[j] = uint(c & 0xFFFFFFFF);
            c >>= 32;
            if (c == 0) break;
        }
    }

    Fp381 r;
    for (int i = 0; i < FP381_LIMBS; i++) r.v[i] = t[i + FP381_LIMBS];
    if (fp381_gte(r, fp381_modulus())) {
        uint borrow;
        r = fp381_sub_raw(r, fp381_modulus(), borrow);
    }
    return r;
}

#endif // BLS12381_FQ_METAL
