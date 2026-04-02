// BN254 base field Fp arithmetic for Metal GPU
//
// p = 21888242871839275222246405745257275088696311157297823662689037894645226208583
// Field elements as 8x32-bit limbs in Montgomery form (little-endian).

#ifndef BN254_FP_METAL
#define BN254_FP_METAL

#include <metal_stdlib>
using namespace metal;

struct Fp {
    uint v[8]; // 256-bit value as 8x32-bit limbs (little-endian)
};

struct PointAffine {
    Fp x;
    Fp y;
};

struct PointProjective {
    Fp x;
    Fp y;
    Fp z;
};

// BN254 prime modulus p (little-endian 32-bit limbs)
constant uint P[8] = {
    0xd87cfd47, 0x3c208c16, 0x6871ca8d, 0x97816a91,
    0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72
};

// Montgomery parameter: R^2 mod p (for converting to Montgomery form)
constant uint R2[8] = {
    0x538afa89, 0xf32cfc5b, 0xd44501fb, 0xb5e71911,
    0x0a417ff6, 0x47ab1eff, 0xcab8351f, 0x06d89f71
};

// Montgomery inverse: -p^(-1) mod 2^32
constant uint INV = 0xe4866389u;

// --- 256-bit Arithmetic ---

Fp fp_add_raw(Fp a, Fp b, thread uint &carry) {
    Fp r;
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

Fp fp_sub_raw(Fp a, Fp b, thread uint &borrow) {
    Fp r;
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

bool fp_gte(Fp a, Fp b) {
    for (int i = 7; i >= 0; i--) {
        if (a.v[i] > b.v[i]) return true;
        if (a.v[i] < b.v[i]) return false;
    }
    return true;
}

Fp fp_modulus() {
    Fp r;
    for (int i = 0; i < 8; i++) r.v[i] = P[i];
    return r;
}

Fp fp_add(Fp a, Fp b) {
    uint carry;
    Fp r = fp_add_raw(a, b, carry);
    Fp p = fp_modulus();
    if (carry || fp_gte(r, p)) {
        uint borrow;
        r = fp_sub_raw(r, p, borrow);
    }
    return r;
}

Fp fp_sub(Fp a, Fp b) {
    uint borrow;
    Fp r = fp_sub_raw(a, b, borrow);
    if (borrow) {
        uint carry;
        r = fp_add_raw(r, fp_modulus(), carry);
    }
    return r;
}

bool fp_is_zero(Fp a) {
    for (int i = 0; i < 8; i++) {
        if (a.v[i] != 0) return false;
    }
    return true;
}

Fp fp_zero() {
    Fp r;
    for (int i = 0; i < 8; i++) r.v[i] = 0;
    return r;
}

Fp fp_one() {
    Fp r;
    r.v[0] = 0xc58f0d9d; r.v[1] = 0xd35d438d; r.v[2] = 0xf5c70b3d;
    r.v[3] = 0x0a78eb28; r.v[4] = 0x7879462c; r.v[5] = 0x666ea36f;
    r.v[6] = 0x9a07df2f; r.v[7] = 0x0e0a77c1;
    return r;
}

// --- Montgomery Multiplication ---
// CIOS (Coarsely Integrated Operand Scanning) with 32-bit limbs

Fp fp_mul(Fp a, Fp b) {
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

        uint m = t[0] * INV;
        carry = ulong(t[0]) + ulong(m) * ulong(P[0]);
        carry >>= 32;
        #pragma unroll
        for (int j = 1; j < 8; j++) {
            carry += ulong(t[j]) + ulong(m) * ulong(P[j]);
            t[j - 1] = uint(carry & 0xFFFFFFFF);
            carry >>= 32;
        }
        ext = ulong(t[8]) + carry;
        t[7] = uint(ext & 0xFFFFFFFF);
        t[8] = t[9] + uint(ext >> 32);
        t[9] = 0;
    }

    Fp r;
    for (int i = 0; i < 8; i++) r.v[i] = t[i];
    if (t[8] != 0 || fp_gte(r, fp_modulus())) {
        uint borrow;
        r = fp_sub_raw(r, fp_modulus(), borrow);
    }
    return r;
}

// SOS squaring: 36 cross-term muls + 8 diagonal vs 64 for general mul
Fp fp_sqr(Fp a) {
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
        uint m = t[i] * INV;
        ulong c = ulong(t[i]) + ulong(m) * ulong(P[0]);
        c >>= 32;
        for (int j = 1; j < 8; j++) {
            c += ulong(t[i + j]) + ulong(m) * ulong(P[j]);
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

    Fp r;
    for (int i = 0; i < 8; i++) r.v[i] = t[i + 8];
    if (fp_gte(r, fp_modulus())) {
        uint borrow;
        r = fp_sub_raw(r, fp_modulus(), borrow);
    }
    return r;
}

// Specialized doubling: left-shift by 1 instead of full 256-bit addition
Fp fp_double(Fp a) {
    Fp r;
    uint carry = 0;
    for (int i = 0; i < 8; i++) {
        uint doubled = (a.v[i] << 1) | carry;
        carry = a.v[i] >> 31;
        r.v[i] = doubled;
    }
    if (carry || fp_gte(r, fp_modulus())) {
        uint borrow;
        r = fp_sub_raw(r, fp_modulus(), borrow);
    }
    return r;
}

#endif // BN254_FP_METAL
