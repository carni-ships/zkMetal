// Vesta base field Fp arithmetic for Metal GPU
//
// p = 0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001
// (= Pallas scalar field Fq)
// 255-bit prime, field elements as 8x32-bit limbs in Montgomery form (little-endian).

#ifndef VESTA_FP_METAL
#define VESTA_FP_METAL

#include <metal_stdlib>
using namespace metal;

struct VestaFp {
    uint v[8]; // 256-bit value as 8x32-bit limbs (little-endian)
};

struct VestaPointAffine {
    VestaFp x;
    VestaFp y;
};

struct VestaPointProjective {
    VestaFp x;
    VestaFp y;
    VestaFp z;
};

// Vesta base field modulus p (little-endian 32-bit limbs)
constant uint VESTA_P[8] = {
    0x00000001, 0x8c46eb21, 0x0994a8dd, 0x224698fc,
    0x00000000, 0x00000000, 0x00000000, 0x40000000
};

// Montgomery parameter: R^2 mod p
constant uint VESTA_R2[8] = {
    0x0000000f, 0xfc9678ff, 0x891a16e3, 0x67bb433d,
    0x04ccf590, 0x7fae2310, 0x7ccfdaa9, 0x096d41af
};

// Montgomery inverse: -p^(-1) mod 2^32
constant uint VESTA_INV = 0xffffffffu;

// --- 256-bit Arithmetic ---

VestaFp vesta_add_raw(VestaFp a, VestaFp b, thread uint &carry) {
    VestaFp r;
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

VestaFp vesta_sub_raw(VestaFp a, VestaFp b, thread uint &borrow) {
    VestaFp r;
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

bool vesta_gte(VestaFp a, VestaFp b) {
    for (int i = 7; i >= 0; i--) {
        if (a.v[i] > b.v[i]) return true;
        if (a.v[i] < b.v[i]) return false;
    }
    return true;
}

VestaFp vesta_modulus() {
    VestaFp r;
    for (int i = 0; i < 8; i++) r.v[i] = VESTA_P[i];
    return r;
}

VestaFp vesta_add(VestaFp a, VestaFp b) {
    uint carry;
    VestaFp r = vesta_add_raw(a, b, carry);
    VestaFp p = vesta_modulus();
    if (carry || vesta_gte(r, p)) {
        uint borrow;
        r = vesta_sub_raw(r, p, borrow);
    }
    return r;
}

VestaFp vesta_add_lazy(VestaFp a, VestaFp b) {
    uint carry;
    return vesta_add_raw(a, b, carry);
}

VestaFp vesta_reduce(VestaFp a) {
    if (vesta_gte(a, vesta_modulus())) {
        uint borrow;
        a = vesta_sub_raw(a, vesta_modulus(), borrow);
    }
    return a;
}

VestaFp vesta_sub(VestaFp a, VestaFp b) {
    uint borrow;
    VestaFp r = vesta_sub_raw(a, b, borrow);
    if (borrow) {
        uint carry;
        r = vesta_add_raw(r, vesta_modulus(), carry);
    }
    return r;
}

bool vesta_is_zero(VestaFp a) {
    for (int i = 0; i < 8; i++) {
        if (a.v[i] != 0) return false;
    }
    return true;
}

VestaFp vesta_zero() {
    VestaFp r;
    for (int i = 0; i < 8; i++) r.v[i] = 0;
    return r;
}

// R mod p in 32-bit limbs (Montgomery form of 1)
VestaFp vesta_one() {
    VestaFp r;
    r.v[0] = 0xfffffffd; r.v[1] = 0x5b2b3e9c;
    r.v[2] = 0xe3420567; r.v[3] = 0x992c350b;
    r.v[4] = 0xffffffff; r.v[5] = 0xffffffff;
    r.v[6] = 0xffffffff; r.v[7] = 0x3fffffff;
    return r;
}

VestaFp vesta_double(VestaFp a) {
    VestaFp r;
    uint carry = 0;
    for (int i = 0; i < 8; i++) {
        uint doubled = (a.v[i] << 1) | carry;
        carry = a.v[i] >> 31;
        r.v[i] = doubled;
    }
    if (carry || vesta_gte(r, vesta_modulus())) {
        uint borrow;
        r = vesta_sub_raw(r, vesta_modulus(), borrow);
    }
    return r;
}

// Montgomery multiplication (CIOS with 32-bit limbs)
VestaFp vesta_mul(VestaFp a, VestaFp b) {
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

        uint m = t[0] * VESTA_INV;
        carry = ulong(t[0]) + ulong(m) * ulong(VESTA_P[0]);
        carry >>= 32;
        #pragma unroll
        for (int j = 1; j < 8; j++) {
            carry += ulong(t[j]) + ulong(m) * ulong(VESTA_P[j]);
            t[j - 1] = uint(carry & 0xFFFFFFFF);
            carry >>= 32;
        }
        ext = ulong(t[8]) + carry;
        t[7] = uint(ext & 0xFFFFFFFF);
        t[8] = t[9] + uint(ext >> 32);
        t[9] = 0;
    }

    VestaFp r;
    for (int i = 0; i < 8; i++) r.v[i] = t[i];
    if (t[8] != 0 || vesta_gte(r, vesta_modulus())) {
        uint borrow;
        r = vesta_sub_raw(r, vesta_modulus(), borrow);
    }
    return r;
}

// Montgomery squaring (SOS)
VestaFp vesta_sqr(VestaFp a) {
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
        uint m = t[i] * VESTA_INV;
        ulong c = ulong(t[i]) + ulong(m) * ulong(VESTA_P[0]);
        c >>= 32;
        for (int j = 1; j < 8; j++) {
            c += ulong(t[i + j]) + ulong(m) * ulong(VESTA_P[j]);
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

    VestaFp r;
    for (int i = 0; i < 8; i++) r.v[i] = t[i + 8];
    if (vesta_gte(r, vesta_modulus())) {
        uint borrow;
        r = vesta_sub_raw(r, vesta_modulus(), borrow);
    }
    return r;
}

// Negation: -a mod p
VestaFp vesta_neg(VestaFp a) {
    if (vesta_is_zero(a)) return a;
    uint borrow;
    return vesta_sub_raw(vesta_modulus(), a, borrow);
}

// Convert to Montgomery form
VestaFp vesta_to_mont(VestaFp a) {
    VestaFp r2;
    for (int i = 0; i < 8; i++) r2.v[i] = VESTA_R2[i];
    return vesta_mul(a, r2);
}

// Convert from Montgomery form
VestaFp vesta_from_mont(VestaFp a) {
    VestaFp one;
    for (int i = 0; i < 8; i++) one.v[i] = 0;
    one.v[0] = 1;
    return vesta_mul(a, one);
}

// Modular exponentiation
VestaFp vesta_pow(VestaFp base, const thread uint exp[8]) {
    VestaFp result = vesta_one();
    VestaFp b = base;
    for (int i = 0; i < 8; i++) {
        uint word = exp[i];
        for (int bit = 0; bit < 32; bit++) {
            if (word & 1u) {
                result = vesta_mul(result, b);
            }
            b = vesta_mul(b, b);
            word >>= 1;
        }
    }
    return result;
}

// Modular inverse via Fermat's little theorem: a^(p-2) mod p
VestaFp vesta_inv(VestaFp a) {
    uint exp[8] = {
        0xffffffffu, 0x8c46eb20u, 0x0994a8ddu, 0x224698fcu,
        0x00000000u, 0x00000000u, 0x00000000u, 0x40000000u
    };
    return vesta_pow(a, exp);
}

#endif // VESTA_FP_METAL
