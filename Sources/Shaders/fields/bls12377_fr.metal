// BLS12-377 scalar field Fr arithmetic for Metal GPU
//
// r = 8444461749428370424248824938781546531375899335154063827935233455917409239041
// Field elements as 8x32-bit limbs in Montgomery form (little-endian).
// TWO_ADICITY = 47 (r-1 = 2^47 * t, t odd)

#ifndef BLS12377_FR_METAL
#define BLS12377_FR_METAL

#include <metal_stdlib>
using namespace metal;

struct Fr377 {
    uint v[8]; // 256-bit value as 8x32-bit limbs (little-endian)
};

// BLS12-377 scalar field modulus r (little-endian 32-bit limbs)
constant uint FR377_P[8] = {
    0x00000001, 0x0a118000, 0xd0000001, 0x59aa76fe,
    0x5c37b001, 0x60b44d1e, 0x9a2ca556, 0x12ab655e
};

// Montgomery parameter: R^2 mod r
constant uint FR377_R2[8] = {
    0xb861857b, 0x25d577ba, 0x8860591f, 0xcc2c27b5,
    0xe5dc8593, 0xa7cc008f, 0xeff1c939, 0x011fdae7
};

// Montgomery inverse: -r^(-1) mod 2^32
constant uint FR377_INV = 0xFFFFFFFFu;

// --- 256-bit Arithmetic ---

Fr377 fr377_add_raw(Fr377 a, Fr377 b, thread uint &carry) {
    Fr377 r;
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

Fr377 fr377_sub_raw(Fr377 a, Fr377 b, thread uint &borrow) {
    Fr377 r;
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

bool fr377_gte(Fr377 a, Fr377 b) {
    for (int i = 7; i >= 0; i--) {
        if (a.v[i] > b.v[i]) return true;
        if (a.v[i] < b.v[i]) return false;
    }
    return true;
}

Fr377 fr377_modulus() {
    Fr377 r;
    for (int i = 0; i < 8; i++) r.v[i] = FR377_P[i];
    return r;
}

Fr377 fr377_add(Fr377 a, Fr377 b) {
    uint carry;
    Fr377 r = fr377_add_raw(a, b, carry);
    Fr377 p = fr377_modulus();
    if (carry || fr377_gte(r, p)) {
        uint borrow;
        r = fr377_sub_raw(r, p, borrow);
    }
    return r;
}

// Lazy addition: no modular reduction. Result may be in [0, 2^256).
// Safe as input to fr377_mul (CIOS handles inputs up to 2^256 - 1).
// Caller must ensure no overflow: a + b < 2^256.
Fr377 fr377_add_lazy(Fr377 a, Fr377 b) {
    uint carry;
    return fr377_add_raw(a, b, carry);
}

// Reduce value to [0, p). Use after lazy additions when reduction needed.
Fr377 fr377_reduce(Fr377 a) {
    if (fr377_gte(a, fr377_modulus())) {
        uint borrow;
        a = fr377_sub_raw(a, fr377_modulus(), borrow);
    }
    return a;
}

Fr377 fr377_sub(Fr377 a, Fr377 b) {
    uint borrow;
    Fr377 r = fr377_sub_raw(a, b, borrow);
    if (borrow) {
        uint carry;
        r = fr377_add_raw(r, fr377_modulus(), carry);
    }
    return r;
}

bool fr377_is_zero(Fr377 a) {
    for (int i = 0; i < 8; i++) {
        if (a.v[i] != 0) return false;
    }
    return true;
}

Fr377 fr377_zero() {
    Fr377 r;
    for (int i = 0; i < 8; i++) r.v[i] = 0;
    return r;
}

// R mod r in 32-bit limbs (Montgomery form of 1)
Fr377 fr377_one() {
    Fr377 r;
    r.v[0] = 0xfffffff3; r.v[1] = 0x7d1c7fff;
    r.v[2] = 0x6ffffff2; r.v[3] = 0x7257f50f;
    r.v[4] = 0x512c0fee; r.v[5] = 0x16d81575;
    r.v[6] = 0x2bbb9a9d; r.v[7] = 0x0d4bda32;
    return r;
}

// Specialized doubling: left-shift by 1 instead of full 256-bit addition
Fr377 fr377_double(Fr377 a) {
    Fr377 r;
    uint carry = 0;
    for (int i = 0; i < 8; i++) {
        uint doubled = (a.v[i] << 1) | carry;
        carry = a.v[i] >> 31;
        r.v[i] = doubled;
    }
    if (carry || fr377_gte(r, fr377_modulus())) {
        uint borrow;
        r = fr377_sub_raw(r, fr377_modulus(), borrow);
    }
    return r;
}

// Montgomery multiplication (CIOS with 32-bit limbs)
Fr377 fr377_mul(Fr377 a, Fr377 b) {
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

        uint m = t[0] * FR377_INV;
        carry = ulong(t[0]) + ulong(m) * ulong(FR377_P[0]);
        carry >>= 32;
        #pragma unroll
        for (int j = 1; j < 8; j++) {
            carry += ulong(t[j]) + ulong(m) * ulong(FR377_P[j]);
            t[j - 1] = uint(carry & 0xFFFFFFFF);
            carry >>= 32;
        }
        ext = ulong(t[8]) + carry;
        t[7] = uint(ext & 0xFFFFFFFF);
        t[8] = t[9] + uint(ext >> 32);
        t[9] = 0;
    }

    Fr377 r;
    for (int i = 0; i < 8; i++) r.v[i] = t[i];
    if (t[8] != 0 || fr377_gte(r, fr377_modulus())) {
        uint borrow;
        r = fr377_sub_raw(r, fr377_modulus(), borrow);
    }
    return r;
}

// Montgomery squaring (SOS: Separated Operand Scanning)
// Exploits a*a symmetry: 36 muls for product vs 64 for general mul.
// Then standard Montgomery reduction (64 muls). Total: 100 vs 128.
Fr377 fr377_sqr(Fr377 a) {
    // Step 1: Compute full 512-bit square product a*a into t[0..15]
    uint t[17]; // 16 limbs + carry
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

    // Step 2: Montgomery reduction (same as CIOS reduction phase)
    for (int i = 0; i < 8; i++) {
        uint m = t[i] * FR377_INV;
        ulong c = ulong(t[i]) + ulong(m) * ulong(FR377_P[0]);
        c >>= 32;
        for (int j = 1; j < 8; j++) {
            c += ulong(t[i + j]) + ulong(m) * ulong(FR377_P[j]);
            t[i + j] = uint(c & 0xFFFFFFFF);
            c >>= 32;
        }
        // Propagate carry through remaining limbs
        for (int j = i + 8; j < 16; j++) {
            c += ulong(t[j]);
            t[j] = uint(c & 0xFFFFFFFF);
            c >>= 32;
            if (c == 0) break;
        }
    }

    // Result is in t[8..15]
    Fr377 r;
    for (int i = 0; i < 8; i++) r.v[i] = t[i + 8];
    if (fr377_gte(r, fr377_modulus())) {
        uint borrow;
        r = fr377_sub_raw(r, fr377_modulus(), borrow);
    }
    return r;
}

// Modular exponentiation by squaring: a^exp mod r
// exp is given as 8x32-bit limbs (little-endian)
Fr377 fr377_pow(Fr377 base, const thread uint exp[8]) {
    Fr377 result = fr377_one();
    Fr377 b = base;
    for (int i = 0; i < 8; i++) {
        uint word = exp[i];
        for (int bit = 0; bit < 32; bit++) {
            if (word & 1u) {
                result = fr377_mul(result, b);
            }
            b = fr377_mul(b, b);
            word >>= 1;
        }
    }
    return result;
}

// Modular inverse via Fermat's little theorem: a^(r-2) mod r
Fr377 fr377_inv(Fr377 a) {
    uint exp[8] = {
        0xFFFFFFFFu, 0x0a117fffu, 0xd0000001u, 0x59aa76feu,
        0x5c37b001u, 0x60b44d1eu, 0x9a2ca556u, 0x12ab655eu
    };
    return fr377_pow(a, exp);
}

#endif // BLS12377_FR_METAL
