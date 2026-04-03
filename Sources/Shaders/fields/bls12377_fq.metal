// BLS12-377 base field Fq arithmetic for Metal GPU
//
// q = 258664426012969094010652733694893533536393512754914660539884262666720468348340822774968888139573360124440321458177
// 377-bit prime, field elements as 12x32-bit limbs in Montgomery form (little-endian).

#ifndef BLS12377_FQ_METAL
#define BLS12377_FQ_METAL

#include <metal_stdlib>
using namespace metal;

#define FQ_LIMBS 12

struct Fq377 {
    uint v[FQ_LIMBS]; // 384-bit value as 12x32-bit limbs (little-endian)
};

struct Point377Affine {
    Fq377 x;
    Fq377 y;
};

struct Point377Projective {
    Fq377 x;
    Fq377 y;
    Fq377 z;
};

// BLS12-377 base field modulus q (little-endian 32-bit limbs)
constant uint FQ377_P[FQ_LIMBS] = {
    0x00000001, 0x8508c000, 0x30000000, 0x170b5d44,
    0xba094800, 0x1ef3622f, 0x00f5138f, 0x1a22d9f3,
    0x6ca1493b, 0xc63b05c0, 0x17c510ea, 0x01ae3a46
};

// Montgomery parameter: R^2 mod q
constant uint FQ377_R2[FQ_LIMBS] = {
    0x9400cd22, 0xb786686c, 0xb00431b1, 0x0329fcaa,
    0x62d6b46d, 0x22a5f111, 0x827dc3ac, 0xbfdf7d03,
    0x41790bf9, 0x837e92f0, 0x1e914b88, 0x006dfccb
};

// Montgomery inverse: -q^(-1) mod 2^32
constant uint FQ377_INV = 0xFFFFFFFFu;

// --- 384-bit Arithmetic ---

Fq377 fq377_add_raw(Fq377 a, Fq377 b, thread uint &carry) {
    Fq377 r;
    ulong c = 0;
    #pragma unroll
    for (int i = 0; i < FQ_LIMBS; i++) {
        c += ulong(a.v[i]) + ulong(b.v[i]);
        r.v[i] = uint(c & 0xFFFFFFFF);
        c >>= 32;
    }
    carry = uint(c);
    return r;
}

Fq377 fq377_sub_raw(Fq377 a, Fq377 b, thread uint &borrow) {
    Fq377 r;
    long c = 0;
    #pragma unroll
    for (int i = 0; i < FQ_LIMBS; i++) {
        c += long(a.v[i]) - long(b.v[i]);
        r.v[i] = uint(c & 0xFFFFFFFF);
        c >>= 32;
    }
    borrow = (c < 0) ? 1u : 0u;
    return r;
}

bool fq377_gte(Fq377 a, Fq377 b) {
    for (int i = FQ_LIMBS - 1; i >= 0; i--) {
        if (a.v[i] > b.v[i]) return true;
        if (a.v[i] < b.v[i]) return false;
    }
    return true;
}

Fq377 fq377_modulus() {
    Fq377 r;
    for (int i = 0; i < FQ_LIMBS; i++) r.v[i] = FQ377_P[i];
    return r;
}

Fq377 fq377_add(Fq377 a, Fq377 b) {
    uint carry;
    Fq377 r = fq377_add_raw(a, b, carry);
    if (carry || fq377_gte(r, fq377_modulus())) {
        uint borrow;
        r = fq377_sub_raw(r, fq377_modulus(), borrow);
    }
    return r;
}

Fq377 fq377_sub(Fq377 a, Fq377 b) {
    uint borrow;
    Fq377 r = fq377_sub_raw(a, b, borrow);
    if (borrow) {
        uint carry;
        r = fq377_add_raw(r, fq377_modulus(), carry);
    }
    return r;
}

bool fq377_is_zero(Fq377 a) {
    for (int i = 0; i < FQ_LIMBS; i++) {
        if (a.v[i] != 0) return false;
    }
    return true;
}

Fq377 fq377_neg(Fq377 a) {
    if (fq377_is_zero(a)) return a;
    return fq377_sub(fq377_modulus(), a);
}

Fq377 fq377_zero() {
    Fq377 r;
    for (int i = 0; i < FQ_LIMBS; i++) r.v[i] = 0;
    return r;
}

Fq377 fq377_one() {
    Fq377 r;
    r.v[0]  = 0xffffff68; r.v[1]  = 0x02cdffff; r.v[2]  = 0x7fffffb1;
    r.v[3]  = 0x51409f83; r.v[4]  = 0x8a7d3ff2; r.v[5]  = 0x9f7db3a9;
    r.v[6]  = 0x6e7c6305; r.v[7]  = 0x7b4e97b7; r.v[8]  = 0x803c84e8;
    r.v[9]  = 0x4cf495bf; r.v[10] = 0xe2fdf49a; r.v[11] = 0x008d6661;
    return r;
}

// --- Montgomery Multiplication ---
// CIOS (Coarsely Integrated Operand Scanning) with 32-bit limbs for 12-limb field

Fq377 fq377_mul(Fq377 a, Fq377 b) {
    uint t[FQ_LIMBS + 2];
    #pragma unroll
    for (int i = 0; i < FQ_LIMBS + 2; i++) t[i] = 0;

    #pragma unroll
    for (int i = 0; i < FQ_LIMBS; i++) {
        ulong carry = 0;
        #pragma unroll
        for (int j = 0; j < FQ_LIMBS; j++) {
            carry += ulong(t[j]) + ulong(a.v[i]) * ulong(b.v[j]);
            t[j] = uint(carry & 0xFFFFFFFF);
            carry >>= 32;
        }
        ulong ext = ulong(t[FQ_LIMBS]) + carry;
        t[FQ_LIMBS] = uint(ext & 0xFFFFFFFF);
        t[FQ_LIMBS + 1] = uint(ext >> 32);

        uint m = t[0] * FQ377_INV;
        carry = ulong(t[0]) + ulong(m) * ulong(FQ377_P[0]);
        carry >>= 32;
        #pragma unroll
        for (int j = 1; j < FQ_LIMBS; j++) {
            carry += ulong(t[j]) + ulong(m) * ulong(FQ377_P[j]);
            t[j - 1] = uint(carry & 0xFFFFFFFF);
            carry >>= 32;
        }
        ext = ulong(t[FQ_LIMBS]) + carry;
        t[FQ_LIMBS - 1] = uint(ext & 0xFFFFFFFF);
        t[FQ_LIMBS] = t[FQ_LIMBS + 1] + uint(ext >> 32);
        t[FQ_LIMBS + 1] = 0;
    }

    Fq377 r;
    for (int i = 0; i < FQ_LIMBS; i++) r.v[i] = t[i];
    if (t[FQ_LIMBS] != 0 || fq377_gte(r, fq377_modulus())) {
        uint borrow;
        r = fq377_sub_raw(r, fq377_modulus(), borrow);
    }
    return r;
}

// SOS squaring: exploits a² symmetry for fewer multiplications
Fq377 fq377_sqr(Fq377 a) {
    uint t[2 * FQ_LIMBS + 1];
    for (int i = 0; i < 2 * FQ_LIMBS + 1; i++) t[i] = 0;

    // Cross terms: a[i] * a[j] for i < j
    for (int i = 0; i < FQ_LIMBS - 1; i++) {
        ulong carry = 0;
        for (int j = i + 1; j < FQ_LIMBS; j++) {
            carry += ulong(t[i + j]) + ulong(a.v[i]) * ulong(a.v[j]);
            t[i + j] = uint(carry & 0xFFFFFFFF);
            carry >>= 32;
        }
        t[i + FQ_LIMBS] += uint(carry);
    }

    // Double the cross terms
    uint top_carry = 0;
    for (int i = 1; i < 2 * FQ_LIMBS; i++) {
        ulong doubled = (ulong(t[i]) << 1) | ulong(top_carry);
        t[i] = uint(doubled & 0xFFFFFFFF);
        top_carry = uint(doubled >> 32);
    }

    // Add diagonal terms: a[i]^2
    ulong carry = 0;
    for (int i = 0; i < FQ_LIMBS; i++) {
        carry += ulong(t[2*i]) + ulong(a.v[i]) * ulong(a.v[i]);
        t[2*i] = uint(carry & 0xFFFFFFFF);
        carry >>= 32;
        carry += ulong(t[2*i + 1]);
        t[2*i + 1] = uint(carry & 0xFFFFFFFF);
        carry >>= 32;
    }

    // Montgomery reduction
    for (int i = 0; i < FQ_LIMBS; i++) {
        uint m = t[i] * FQ377_INV;
        ulong c = ulong(t[i]) + ulong(m) * ulong(FQ377_P[0]);
        c >>= 32;
        for (int j = 1; j < FQ_LIMBS; j++) {
            c += ulong(t[i + j]) + ulong(m) * ulong(FQ377_P[j]);
            t[i + j] = uint(c & 0xFFFFFFFF);
            c >>= 32;
        }
        for (int j = i + FQ_LIMBS; j < 2 * FQ_LIMBS; j++) {
            c += ulong(t[j]);
            t[j] = uint(c & 0xFFFFFFFF);
            c >>= 32;
            if (c == 0) break;
        }
    }

    Fq377 r;
    for (int i = 0; i < FQ_LIMBS; i++) r.v[i] = t[i + FQ_LIMBS];
    if (fq377_gte(r, fq377_modulus())) {
        uint borrow;
        r = fq377_sub_raw(r, fq377_modulus(), borrow);
    }
    return r;
}

// Specialized doubling: left-shift by 1
Fq377 fq377_double(Fq377 a) {
    Fq377 r;
    uint carry = 0;
    for (int i = 0; i < FQ_LIMBS; i++) {
        uint doubled = (a.v[i] << 1) | carry;
        carry = a.v[i] >> 31;
        r.v[i] = doubled;
    }
    if (carry || fq377_gte(r, fq377_modulus())) {
        uint borrow;
        r = fq377_sub_raw(r, fq377_modulus(), borrow);
    }
    return r;
}

// To Montgomery form: a_mont = a * R^2 * R^(-1) = a * R mod q
Fq377 fq377_to_mont(Fq377 a) {
    Fq377 r2;
    for (int i = 0; i < FQ_LIMBS; i++) r2.v[i] = FQ377_R2[i];
    return fq377_mul(a, r2);
}

// From Montgomery form: a = a_mont * 1 * R^(-1) = a_mont * R^(-1) mod q
Fq377 fq377_from_mont(Fq377 a) {
    Fq377 one_raw = fq377_zero();
    one_raw.v[0] = 1;
    return fq377_mul(a, one_raw);
}

// Field inverse via Fermat's little theorem: a^(-1) = a^(q-2) mod q
Fq377 fq377_inv(Fq377 a) {
    // q - 2 in binary: process with square-and-multiply
    // For efficiency, we store q-2 as limbs and iterate bits
    uint qm2[FQ_LIMBS];
    // q - 2 = q - 1 - 1 = ...FFFFFFFF (in the low limb, subtract 2 from q's low limb)
    for (int i = 0; i < FQ_LIMBS; i++) qm2[i] = FQ377_P[i];
    // Subtract 2 from q
    ulong borrow = 2;
    for (int i = 0; i < FQ_LIMBS; i++) {
        ulong tmp = ulong(qm2[i]) - borrow;
        qm2[i] = uint(tmp & 0xFFFFFFFF);
        borrow = (tmp >> 63) & 1;
    }

    Fq377 result = fq377_one();
    Fq377 base = a;

    for (int i = 0; i < FQ_LIMBS; i++) {
        for (int bit = 0; bit < 32; bit++) {
            if (qm2[i] & (1u << bit)) {
                result = fq377_mul(result, base);
            }
            base = fq377_sqr(base);
        }
    }
    return result;
}

#endif // BLS12377_FQ_METAL
