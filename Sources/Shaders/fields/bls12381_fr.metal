// BLS12-381 scalar field Fr arithmetic for Metal GPU
//
// r = 52435875175126190479447740508185965837690552500527637822603658699938581184513
// Field elements as 8x32-bit limbs in Montgomery form (little-endian).
// TWO_ADICITY = 32 (r-1 = 2^32 * t, t odd)

#ifndef BLS12381_FR_METAL
#define BLS12381_FR_METAL

#include <metal_stdlib>
using namespace metal;

struct Fr381 {
    uint v[8]; // 256-bit value as 8x32-bit limbs (little-endian)
};

// BLS12-381 scalar field modulus r (little-endian 32-bit limbs)
constant uint FR381_P[8] = {
    0x00000001, 0xffffffff, 0xfffe5bfe, 0x53bda402,
    0x09a1d805, 0x3339d808, 0x299d7d48, 0x73eda753
};

// Montgomery parameter: R^2 mod r
constant uint FR381_R2[8] = {
    0xf3f29c6d, 0xc999e990, 0x87925c23, 0x2b6cedcb,
    0x7254398f, 0x05d31496, 0x9f59ff11, 0x0748d9d9
};

// Montgomery inverse: -r^(-1) mod 2^32
constant uint FR381_INV = 0xFFFFFFFFu;

// --- 256-bit Arithmetic ---

Fr381 fr381_add_raw(Fr381 a, Fr381 b, thread uint &carry) {
    Fr381 r;
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

Fr381 fr381_sub_raw(Fr381 a, Fr381 b, thread uint &borrow) {
    Fr381 r;
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

bool fr381_gte(Fr381 a, Fr381 b) {
    for (int i = 7; i >= 0; i--) {
        if (a.v[i] > b.v[i]) return true;
        if (a.v[i] < b.v[i]) return false;
    }
    return true;
}

Fr381 fr381_modulus() {
    Fr381 r;
    for (int i = 0; i < 8; i++) r.v[i] = FR381_P[i];
    return r;
}

Fr381 fr381_add(Fr381 a, Fr381 b) {
    uint carry;
    Fr381 r = fr381_add_raw(a, b, carry);
    Fr381 p = fr381_modulus();
    if (carry || fr381_gte(r, p)) {
        uint borrow;
        r = fr381_sub_raw(r, p, borrow);
    }
    return r;
}

// Lazy addition: no modular reduction.
Fr381 fr381_add_lazy(Fr381 a, Fr381 b) {
    uint carry;
    return fr381_add_raw(a, b, carry);
}

// Reduce value to [0, p).
Fr381 fr381_reduce(Fr381 a) {
    if (fr381_gte(a, fr381_modulus())) {
        uint borrow;
        a = fr381_sub_raw(a, fr381_modulus(), borrow);
    }
    return a;
}

Fr381 fr381_sub(Fr381 a, Fr381 b) {
    uint borrow;
    Fr381 r = fr381_sub_raw(a, b, borrow);
    if (borrow) {
        uint carry;
        r = fr381_add_raw(r, fr381_modulus(), carry);
    }
    return r;
}

bool fr381_is_zero(Fr381 a) {
    for (int i = 0; i < 8; i++) {
        if (a.v[i] != 0) return false;
    }
    return true;
}

Fr381 fr381_zero() {
    Fr381 r;
    for (int i = 0; i < 8; i++) r.v[i] = 0;
    return r;
}

// R mod r in 32-bit limbs (Montgomery form of 1)
Fr381 fr381_one() {
    Fr381 r;
    r.v[0] = 0xfffffffe; r.v[1] = 0x00000001;
    r.v[2] = 0x00034802; r.v[3] = 0x5884b7fa;
    r.v[4] = 0xecbc4ff5; r.v[5] = 0x998c4fef;
    r.v[6] = 0xacc5056f; r.v[7] = 0x1824b159;
    return r;
}

// Specialized doubling
Fr381 fr381_double(Fr381 a) {
    Fr381 r;
    uint carry = 0;
    for (int i = 0; i < 8; i++) {
        uint doubled = (a.v[i] << 1) | carry;
        carry = a.v[i] >> 31;
        r.v[i] = doubled;
    }
    if (carry || fr381_gte(r, fr381_modulus())) {
        uint borrow;
        r = fr381_sub_raw(r, fr381_modulus(), borrow);
    }
    return r;
}

// Montgomery multiplication (CIOS with 32-bit limbs)
Fr381 fr381_mul(Fr381 a, Fr381 b) {
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

        uint m = t[0] * FR381_INV;
        carry = ulong(t[0]) + ulong(m) * ulong(FR381_P[0]);
        carry >>= 32;
        #pragma unroll
        for (int j = 1; j < 8; j++) {
            carry += ulong(t[j]) + ulong(m) * ulong(FR381_P[j]);
            t[j - 1] = uint(carry & 0xFFFFFFFF);
            carry >>= 32;
        }
        ext = ulong(t[8]) + carry;
        t[7] = uint(ext & 0xFFFFFFFF);
        t[8] = t[9] + uint(ext >> 32);
        t[9] = 0;
    }

    Fr381 r;
    for (int i = 0; i < 8; i++) r.v[i] = t[i];
    if (t[8] != 0 || fr381_gte(r, fr381_modulus())) {
        uint borrow;
        r = fr381_sub_raw(r, fr381_modulus(), borrow);
    }
    return r;
}

// Montgomery squaring (SOS)
Fr381 fr381_sqr(Fr381 a) {
    uint t[17];
    for (int i = 0; i < 17; i++) t[i] = 0;

    // Cross terms
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

    // Add diagonal terms
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
        uint m = t[i] * FR381_INV;
        ulong c = ulong(t[i]) + ulong(m) * ulong(FR381_P[0]);
        c >>= 32;
        for (int j = 1; j < 8; j++) {
            c += ulong(t[i + j]) + ulong(m) * ulong(FR381_P[j]);
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

    Fr381 r;
    for (int i = 0; i < 8; i++) r.v[i] = t[i + 8];
    if (fr381_gte(r, fr381_modulus())) {
        uint borrow;
        r = fr381_sub_raw(r, fr381_modulus(), borrow);
    }
    return r;
}

// Modular exponentiation
Fr381 fr381_pow(Fr381 base, const thread uint exp[8]) {
    Fr381 result = fr381_one();
    Fr381 b = base;
    for (int i = 0; i < 8; i++) {
        uint word = exp[i];
        for (int bit = 0; bit < 32; bit++) {
            if (word & 1u) {
                result = fr381_mul(result, b);
            }
            b = fr381_mul(b, b);
            word >>= 1;
        }
    }
    return result;
}

// Modular inverse via Fermat's little theorem: a^(r-2) mod r
Fr381 fr381_inv(Fr381 a) {
    // r - 2 in 32-bit limbs
    uint exp[8] = {
        0xFFFFFFFFu, 0xFFFFFFFEu, 0xFFFE5BFEu, 0x53BDA402u,
        0x09A1D805u, 0x3339D808u, 0x299D7D48u, 0x73EDA753u
    };
    return fr381_pow(a, exp);
}

#endif // BLS12381_FR_METAL
