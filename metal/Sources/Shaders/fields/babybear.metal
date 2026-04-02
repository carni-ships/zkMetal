// BabyBear field arithmetic for Metal GPU
// p = 2^31 - 2^27 + 1 = 0x78000001 = 2013265921
// Used by SP1 (Succinct), RISC Zero, Plonky3
// Single 32-bit representation — fits natively in GPU registers.
// TWO_ADICITY = 27 (p - 1 = 2^27 * 15)

#ifndef BABYBEAR_METAL
#define BABYBEAR_METAL

#include <metal_stdlib>
using namespace metal;

constant uint BB_P = 0x78000001u;  // 2013265921

struct Bb {
    uint v;
};

Bb bb_zero() { return Bb{0}; }
Bb bb_one() { return Bb{1}; }
bool bb_is_zero(Bb a) { return a.v == 0; }

Bb bb_from_u32(uint v) {
    return Bb{v >= BB_P ? v - BB_P : v};
}

Bb bb_add(Bb a, Bb b) {
    uint sum = a.v + b.v;
    return Bb{sum >= BB_P ? sum - BB_P : sum};
}

Bb bb_sub(Bb a, Bb b) {
    if (a.v >= b.v) return Bb{a.v - b.v};
    return Bb{a.v + BB_P - b.v};
}

Bb bb_neg(Bb a) {
    if (a.v == 0) return a;
    return Bb{BB_P - a.v};
}

// Barrett reduction constant: MU = floor(2^62 / p) = 2290649224
constant uint BB_MU = 2290649223u;

// Multiply: a * b mod p using Barrett reduction
// a, b < p < 2^31, so a*b < 2^62, fits in ulong
// Barrett avoids the expensive 64-bit modulo operation.
Bb bb_mul(Bb a, Bb b) {
    ulong prod = ulong(a.v) * ulong(b.v);

    // Barrett: q ≈ (prod * MU) >> 62
    // Split prod into 32-bit halves for 32x32→64 multiplies
    uint prod_lo = uint(prod);
    uint prod_hi = uint(prod >> 32);

    ulong t1 = ulong(prod_lo) * ulong(BB_MU);
    ulong t2 = ulong(prod_hi) * ulong(BB_MU);

    // q = ((prod * MU) >> 32) >> 30, accurate to within 1
    uint q = uint((t2 + (t1 >> 32)) >> 30);

    // r = prod - q * p; since q ≤ q_exact, r ≥ 0 and r < 2p
    uint r = uint(prod - ulong(q) * ulong(BB_P));
    return Bb{r >= BB_P ? r - BB_P : r};
}

Bb bb_sqr(Bb a) { return bb_mul(a, a); }

// Power: a^n mod p (square-and-multiply)
Bb bb_pow(Bb base, uint n) {
    Bb result = bb_one();
    while (n > 0) {
        if (n & 1) result = bb_mul(result, base);
        base = bb_sqr(base);
        n >>= 1;
    }
    return result;
}

// Inverse via Fermat's little theorem: a^(p-2) mod p
Bb bb_inv(Bb a) {
    return bb_pow(a, BB_P - 2);
}

#endif // BABYBEAR_METAL
