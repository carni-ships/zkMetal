// BN254 scalar field Fr arithmetic for Metal GPU
//
// r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
// Field elements as 8x32-bit limbs in Montgomery form (little-endian).
// TWO_ADICITY = 28 (r-1 = 2^28 * t, t odd)

#ifndef BN254_FR_METAL
#define BN254_FR_METAL

#include <metal_stdlib>
using namespace metal;

struct Fr {
    uint v[8]; // 256-bit value as 8x32-bit limbs (little-endian)
};

// BN254 scalar field modulus r (little-endian 32-bit limbs)
constant uint FR_P[8] = {
    0xf0000001, 0x43e1f593, 0x79b97091, 0x2833e848,
    0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72
};

// Montgomery parameter: R^2 mod r
constant uint FR_R2[8] = {
    0xae216da7, 0x1bb8e645, 0xe35c59e3, 0x53fe3ab1,
    0x53bb8085, 0x8c49833d, 0x7f4e44a5, 0x0216d0b1
};

// Montgomery inverse: -r^(-1) mod 2^32
constant uint FR_INV = 0xefffffffu;

// --- 256-bit Arithmetic ---

Fr fr_add_raw(Fr a, Fr b, thread uint &carry) {
    Fr r;
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

Fr fr_sub_raw(Fr a, Fr b, thread uint &borrow) {
    Fr r;
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

bool fr_gte(Fr a, Fr b) {
    for (int i = 7; i >= 0; i--) {
        if (a.v[i] > b.v[i]) return true;
        if (a.v[i] < b.v[i]) return false;
    }
    return true;
}

Fr fr_modulus() {
    Fr r;
    for (int i = 0; i < 8; i++) r.v[i] = FR_P[i];
    return r;
}

Fr fr_add(Fr a, Fr b) {
    uint carry;
    Fr r = fr_add_raw(a, b, carry);
    Fr p = fr_modulus();
    if (carry || fr_gte(r, p)) {
        uint borrow;
        r = fr_sub_raw(r, p, borrow);
    }
    return r;
}

// Lazy addition: no modular reduction. Result may be in [0, 2^256).
// Safe as input to fr_mul (CIOS handles inputs up to 2^256 - 1).
// Caller must ensure no overflow: a + b < 2^256.
Fr fr_add_lazy(Fr a, Fr b) {
    uint carry;
    return fr_add_raw(a, b, carry);
}

// Reduce value to [0, p). Use after lazy additions when reduction needed.
Fr fr_reduce(Fr a) {
    if (fr_gte(a, fr_modulus())) {
        uint borrow;
        a = fr_sub_raw(a, fr_modulus(), borrow);
    }
    return a;
}

Fr fr_sub(Fr a, Fr b) {
    uint borrow;
    Fr r = fr_sub_raw(a, b, borrow);
    if (borrow) {
        uint carry;
        r = fr_add_raw(r, fr_modulus(), carry);
    }
    return r;
}

bool fr_is_zero(Fr a) {
    for (int i = 0; i < 8; i++) {
        if (a.v[i] != 0) return false;
    }
    return true;
}

Fr fr_zero() {
    Fr r;
    for (int i = 0; i < 8; i++) r.v[i] = 0;
    return r;
}

// R mod r in 32-bit limbs (Montgomery form of 1)
Fr fr_one() {
    Fr r;
    r.v[0] = 0x4ffffffb; r.v[1] = 0xac96341c;
    r.v[2] = 0x9f60cd29; r.v[3] = 0x36fc7695;
    r.v[4] = 0x7879462e; r.v[5] = 0x666ea36f;
    r.v[6] = 0x9a07df2f; r.v[7] = 0x0e0a77c1;
    return r;
}

// Specialized doubling: left-shift by 1 instead of full 256-bit addition
Fr fr_double(Fr a) {
    Fr r;
    uint carry = 0;
    for (int i = 0; i < 8; i++) {
        uint doubled = (a.v[i] << 1) | carry;
        carry = a.v[i] >> 31;
        r.v[i] = doubled;
    }
    if (carry || fr_gte(r, fr_modulus())) {
        uint borrow;
        r = fr_sub_raw(r, fr_modulus(), borrow);
    }
    return r;
}

// Montgomery multiplication (CIOS with 32-bit limbs)
Fr fr_mul(Fr a, Fr b) {
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

        uint m = t[0] * FR_INV;
        carry = ulong(t[0]) + ulong(m) * ulong(FR_P[0]);
        carry >>= 32;
        #pragma unroll
        for (int j = 1; j < 8; j++) {
            carry += ulong(t[j]) + ulong(m) * ulong(FR_P[j]);
            t[j - 1] = uint(carry & 0xFFFFFFFF);
            carry >>= 32;
        }
        ext = ulong(t[8]) + carry;
        t[7] = uint(ext & 0xFFFFFFFF);
        t[8] = t[9] + uint(ext >> 32);
        t[9] = 0;
    }

    Fr r;
    for (int i = 0; i < 8; i++) r.v[i] = t[i];
    if (t[8] != 0 || fr_gte(r, fr_modulus())) {
        uint borrow;
        r = fr_sub_raw(r, fr_modulus(), borrow);
    }
    return r;
}

// Montgomery squaring (SOS: Separated Operand Scanning)
// Exploits a*a symmetry: 36 muls for product vs 64 for general mul.
// Then standard Montgomery reduction (64 muls). Total: 100 vs 128.
Fr fr_sqr(Fr a) {
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
        uint m = t[i] * FR_INV;
        ulong c = ulong(t[i]) + ulong(m) * ulong(FR_P[0]);
        c >>= 32;
        for (int j = 1; j < 8; j++) {
            c += ulong(t[i + j]) + ulong(m) * ulong(FR_P[j]);
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
    Fr r;
    for (int i = 0; i < 8; i++) r.v[i] = t[i + 8];
    if (fr_gte(r, fr_modulus())) {
        uint borrow;
        r = fr_sub_raw(r, fr_modulus(), borrow);
    }
    return r;
}

// Modular exponentiation by squaring: a^exp mod r
// exp is given as 8x32-bit limbs (little-endian)
Fr fr_pow(Fr base, const thread uint exp[8]) {
    Fr result = fr_one();
    Fr b = base;
    for (int i = 0; i < 8; i++) {
        uint word = exp[i];
        for (int bit = 0; bit < 32; bit++) {
            if (word & 1u) {
                result = fr_mul(result, b);
            }
            b = fr_mul(b, b);
            word >>= 1;
        }
    }
    return result;
}

// Modular inverse via Fermat's little theorem: a^(r-2) mod r
// r-2 = 0x30644e72 e131a029 b85045b6 8181585d 2833e848 79b97091 43e1f593 efffffffu
Fr fr_inv(Fr a) {
    uint exp[8] = {
        0xefffffffu, 0x43e1f593u, 0x79b97091u, 0x2833e848u,
        0x8181585du, 0xb85045b6u, 0xe131a029u, 0x30644e72u
    };
    return fr_pow(a, exp);
}

// --- Karatsuba Montgomery Multiplication ---
// See fp_mul_karatsuba for detailed explanation.
Fr fr_mul_karatsuba(Fr a, Fr b) {
    uint t[17];
    for (int i = 0; i < 17; i++) t[i] = 0;

    for (int i = 0; i < 4; i++) {
        ulong carry = 0;
        for (int j = 0; j < 4; j++) {
            carry += ulong(t[i + j]) + ulong(a.v[i]) * ulong(b.v[j]);
            t[i + j] = uint(carry & 0xFFFFFFFF);
            carry >>= 32;
        }
        t[i + 4] += uint(carry);
    }
    for (int i = 0; i < 4; i++) {
        ulong carry = 0;
        for (int j = 0; j < 4; j++) {
            carry += ulong(t[8 + i + j]) + ulong(a.v[4 + i]) * ulong(b.v[4 + j]);
            t[8 + i + j] = uint(carry & 0xFFFFFFFF);
            carry >>= 32;
        }
        t[12 + i] += uint(carry);
    }

    uint z0[8], z2[8];
    for (int i = 0; i < 8; i++) { z0[i] = t[i]; z2[i] = t[i + 8]; }

    uint sa[4], sb[4];
    uint ca = 0, cb = 0;
    { ulong c = 0; for (int i = 0; i < 4; i++) { c += ulong(a.v[i]) + ulong(a.v[4 + i]); sa[i] = uint(c & 0xFFFFFFFF); c >>= 32; } ca = uint(c); }
    { ulong c = 0; for (int i = 0; i < 4; i++) { c += ulong(b.v[i]) + ulong(b.v[4 + i]); sb[i] = uint(c & 0xFFFFFFFF); c >>= 32; } cb = uint(c); }

    uint z1[9];
    for (int i = 0; i < 9; i++) z1[i] = 0;
    for (int i = 0; i < 4; i++) {
        ulong carry = 0;
        for (int j = 0; j < 4; j++) {
            carry += ulong(z1[i + j]) + ulong(sa[i]) * ulong(sb[j]);
            z1[i + j] = uint(carry & 0xFFFFFFFF);
            carry >>= 32;
        }
        z1[i + 4] += uint(carry);
    }
    if (ca) { ulong carry = 0; for (int i = 0; i < 4; i++) { carry += ulong(z1[4 + i]) + ulong(sb[i]); z1[4 + i] = uint(carry & 0xFFFFFFFF); carry >>= 32; } z1[8] += uint(carry); }
    if (cb) { ulong carry = 0; for (int i = 0; i < 4; i++) { carry += ulong(z1[4 + i]) + ulong(sa[i]); z1[4 + i] = uint(carry & 0xFFFFFFFF); carry >>= 32; } z1[8] += uint(carry); }
    if (ca && cb) { z1[8] += 1; }

    { long borrow = 0; for (int i = 0; i < 8; i++) { borrow += long(z1[i]) - long(z0[i]); z1[i] = uint(borrow & 0xFFFFFFFF); borrow >>= 32; } z1[8] = uint(long(z1[8]) + borrow); }
    { long borrow = 0; for (int i = 0; i < 8; i++) { borrow += long(z1[i]) - long(z2[i]); z1[i] = uint(borrow & 0xFFFFFFFF); borrow >>= 32; } z1[8] = uint(long(z1[8]) + borrow); }

    { ulong carry = 0; for (int i = 0; i < 9; i++) { carry += ulong(t[4 + i]) + ulong(z1[i]); t[4 + i] = uint(carry & 0xFFFFFFFF); carry >>= 32; } for (int i = 13; i < 16 && carry; i++) { carry += ulong(t[i]); t[i] = uint(carry & 0xFFFFFFFF); carry >>= 32; } }

    for (int i = 0; i < 8; i++) {
        uint m = t[i] * FR_INV;
        ulong c = ulong(t[i]) + ulong(m) * ulong(FR_P[0]);
        c >>= 32;
        for (int j = 1; j < 8; j++) {
            c += ulong(t[i + j]) + ulong(m) * ulong(FR_P[j]);
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

    Fr r;
    for (int i = 0; i < 8; i++) r.v[i] = t[i + 8];
    if (fr_gte(r, fr_modulus())) {
        uint borrow;
        r = fr_sub_raw(r, fr_modulus(), borrow);
    }
    return r;
}

#endif // BN254_FR_METAL
