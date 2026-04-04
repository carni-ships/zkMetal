// Pallas base field Fp arithmetic for Metal GPU
//
// p = 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001
// 255-bit prime, field elements as 8x32-bit limbs in Montgomery form (little-endian).

#ifndef PALLAS_FP_METAL
#define PALLAS_FP_METAL

#include <metal_stdlib>
using namespace metal;

struct PallasFp {
    uint v[8]; // 256-bit value as 8x32-bit limbs (little-endian)
};

struct PallasPointAffine {
    PallasFp x;
    PallasFp y;
};

struct PallasPointProjective {
    PallasFp x;
    PallasFp y;
    PallasFp z;
};

// Pallas base field modulus p (little-endian 32-bit limbs)
constant uint PALLAS_P[8] = {
    0x00000001, 0x992d30ed, 0x094cf91b, 0x224698fc,
    0x00000000, 0x00000000, 0x00000000, 0x40000000
};

// Montgomery parameter: R^2 mod p
constant uint PALLAS_R2[8] = {
    0x0000000f, 0x8c78ecb3, 0x8b0de0e7, 0xd7d30dbd,
    0xc3c95d18, 0x7797a99b, 0x7b9cb714, 0x096d41af
};

// Montgomery inverse: -p^(-1) mod 2^32
constant uint PALLAS_INV = 0xffffffffu;

// --- 256-bit Arithmetic ---

PallasFp pallas_add_raw(PallasFp a, PallasFp b, thread uint &carry) {
    PallasFp r;
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

PallasFp pallas_sub_raw(PallasFp a, PallasFp b, thread uint &borrow) {
    PallasFp r;
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

bool pallas_gte(PallasFp a, PallasFp b) {
    for (int i = 7; i >= 0; i--) {
        if (a.v[i] > b.v[i]) return true;
        if (a.v[i] < b.v[i]) return false;
    }
    return true;
}

PallasFp pallas_modulus() {
    PallasFp r;
    for (int i = 0; i < 8; i++) r.v[i] = PALLAS_P[i];
    return r;
}

PallasFp pallas_add(PallasFp a, PallasFp b) {
    uint carry;
    PallasFp r = pallas_add_raw(a, b, carry);
    PallasFp p = pallas_modulus();
    if (carry || pallas_gte(r, p)) {
        uint borrow;
        r = pallas_sub_raw(r, p, borrow);
    }
    return r;
}

PallasFp pallas_add_lazy(PallasFp a, PallasFp b) {
    uint carry;
    return pallas_add_raw(a, b, carry);
}

PallasFp pallas_reduce(PallasFp a) {
    if (pallas_gte(a, pallas_modulus())) {
        uint borrow;
        a = pallas_sub_raw(a, pallas_modulus(), borrow);
    }
    return a;
}

PallasFp pallas_sub(PallasFp a, PallasFp b) {
    uint borrow;
    PallasFp r = pallas_sub_raw(a, b, borrow);
    if (borrow) {
        uint carry;
        r = pallas_add_raw(r, pallas_modulus(), carry);
    }
    return r;
}

bool pallas_is_zero(PallasFp a) {
    for (int i = 0; i < 8; i++) {
        if (a.v[i] != 0) return false;
    }
    return true;
}

PallasFp pallas_zero() {
    PallasFp r;
    for (int i = 0; i < 8; i++) r.v[i] = 0;
    return r;
}

// R mod p in 32-bit limbs (Montgomery form of 1)
PallasFp pallas_one() {
    PallasFp r;
    r.v[0] = 0xfffffffd; r.v[1] = 0x34786d38;
    r.v[2] = 0xe41914ad; r.v[3] = 0x992c350b;
    r.v[4] = 0xffffffff; r.v[5] = 0xffffffff;
    r.v[6] = 0xffffffff; r.v[7] = 0x3fffffff;
    return r;
}

PallasFp pallas_double(PallasFp a) {
    PallasFp r;
    uint carry = 0;
    for (int i = 0; i < 8; i++) {
        uint doubled = (a.v[i] << 1) | carry;
        carry = a.v[i] >> 31;
        r.v[i] = doubled;
    }
    if (carry || pallas_gte(r, pallas_modulus())) {
        uint borrow;
        r = pallas_sub_raw(r, pallas_modulus(), borrow);
    }
    return r;
}

// Montgomery multiplication (CIOS with 32-bit limbs)
PallasFp pallas_mul(PallasFp a, PallasFp b) {
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

        uint m = t[0] * PALLAS_INV;
        carry = ulong(t[0]) + ulong(m) * ulong(PALLAS_P[0]);
        carry >>= 32;
        #pragma unroll
        for (int j = 1; j < 8; j++) {
            carry += ulong(t[j]) + ulong(m) * ulong(PALLAS_P[j]);
            t[j - 1] = uint(carry & 0xFFFFFFFF);
            carry >>= 32;
        }
        ext = ulong(t[8]) + carry;
        t[7] = uint(ext & 0xFFFFFFFF);
        t[8] = t[9] + uint(ext >> 32);
        t[9] = 0;
    }

    PallasFp r;
    for (int i = 0; i < 8; i++) r.v[i] = t[i];
    if (t[8] != 0 || pallas_gte(r, pallas_modulus())) {
        uint borrow;
        r = pallas_sub_raw(r, pallas_modulus(), borrow);
    }
    return r;
}

// Montgomery squaring (SOS)
PallasFp pallas_sqr(PallasFp a) {
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
        uint m = t[i] * PALLAS_INV;
        ulong c = ulong(t[i]) + ulong(m) * ulong(PALLAS_P[0]);
        c >>= 32;
        for (int j = 1; j < 8; j++) {
            c += ulong(t[i + j]) + ulong(m) * ulong(PALLAS_P[j]);
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

    PallasFp r;
    for (int i = 0; i < 8; i++) r.v[i] = t[i + 8];
    if (pallas_gte(r, pallas_modulus())) {
        uint borrow;
        r = pallas_sub_raw(r, pallas_modulus(), borrow);
    }
    return r;
}

// Negation: -a mod p
PallasFp pallas_neg(PallasFp a) {
    if (pallas_is_zero(a)) return a;
    uint borrow;
    return pallas_sub_raw(pallas_modulus(), a, borrow);
}

// Convert to Montgomery form
PallasFp pallas_to_mont(PallasFp a) {
    PallasFp r2;
    for (int i = 0; i < 8; i++) r2.v[i] = PALLAS_R2[i];
    return pallas_mul(a, r2);
}

// Convert from Montgomery form
PallasFp pallas_from_mont(PallasFp a) {
    PallasFp one;
    for (int i = 0; i < 8; i++) one.v[i] = 0;
    one.v[0] = 1;
    return pallas_mul(a, one);
}

// Modular exponentiation
PallasFp pallas_pow(PallasFp base, const thread uint exp[8]) {
    PallasFp result = pallas_one();
    PallasFp b = base;
    for (int i = 0; i < 8; i++) {
        uint word = exp[i];
        for (int bit = 0; bit < 32; bit++) {
            if (word & 1u) {
                result = pallas_mul(result, b);
            }
            b = pallas_mul(b, b);
            word >>= 1;
        }
    }
    return result;
}

// Modular inverse via Fermat's little theorem: a^(p-2) mod p
PallasFp pallas_inv(PallasFp a) {
    uint exp[8] = {
        0xffffffffu, 0x992d30ecu, 0x094cf91bu, 0x224698fcu,
        0x00000000u, 0x00000000u, 0x00000000u, 0x40000000u
    };
    return pallas_pow(a, exp);
}

#endif // PALLAS_FP_METAL
