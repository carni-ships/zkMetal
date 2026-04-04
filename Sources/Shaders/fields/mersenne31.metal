// Mersenne31 field arithmetic for Metal GPU
// p = 2^31 - 1 = 0x7FFFFFFF = 2147483647
// Used by Stwo (StarkWare), Circle STARKs
// Single 32-bit representation - extremely fast on GPU.

#ifndef MERSENNE31_METAL
#define MERSENNE31_METAL

#include <metal_stdlib>
using namespace metal;

constant uint M31_P = 0x7FFFFFFFu;  // 2147483647

struct M31 {
    uint v;
};

M31 m31_zero() { return M31{0}; }
M31 m31_one() { return M31{1}; }
bool m31_is_zero(M31 a) { return a.v == 0; }

M31 m31_from_u32(uint v) {
    // Reduce v mod p
    uint r = (v & M31_P) + (v >> 31);
    return M31{r == M31_P ? 0u : r};
}

M31 m31_add(M31 a, M31 b) {
    uint s = a.v + b.v;
    uint r = (s & M31_P) + (s >> 31);
    return M31{r == M31_P ? 0u : r};
}

M31 m31_sub(M31 a, M31 b) {
    if (a.v >= b.v) return M31{a.v - b.v};
    return M31{a.v + M31_P - b.v};
}

M31 m31_neg(M31 a) {
    if (a.v == 0) return a;
    return M31{M31_P - a.v};
}

M31 m31_mul(M31 a, M31 b) {
    ulong prod = ulong(a.v) * ulong(b.v);
    uint lo = uint(prod & ulong(M31_P));
    uint hi = uint(prod >> 31);
    uint s = lo + hi;
    uint r = (s & M31_P) + (s >> 31);
    return M31{r == M31_P ? 0u : r};
}

M31 m31_sqr(M31 a) { return m31_mul(a, a); }

// Power: a^n mod p (square-and-multiply)
M31 m31_pow(M31 base, uint n) {
    M31 result = m31_one();
    while (n > 0) {
        if (n & 1) result = m31_mul(result, base);
        base = m31_sqr(base);
        n >>= 1;
    }
    return result;
}

// Inverse via Fermat's little theorem: a^(p-2) mod p
M31 m31_inv(M31 a) {
    return m31_pow(a, M31_P - 2);
}

// CM31: Complex extension M31[i] / (i^2 + 1)
struct CM31 {
    M31 a;  // real
    M31 b;  // imaginary
};

CM31 cm31_zero() { return CM31{m31_zero(), m31_zero()}; }
CM31 cm31_one() { return CM31{m31_one(), m31_zero()}; }

CM31 cm31_add(CM31 x, CM31 y) {
    return CM31{m31_add(x.a, y.a), m31_add(x.b, y.b)};
}

CM31 cm31_sub(CM31 x, CM31 y) {
    return CM31{m31_sub(x.a, y.a), m31_sub(x.b, y.b)};
}

CM31 cm31_mul(CM31 x, CM31 y) {
    M31 real = m31_sub(m31_mul(x.a, y.a), m31_mul(x.b, y.b));
    M31 imag = m31_add(m31_mul(x.a, y.b), m31_mul(x.b, y.a));
    return CM31{real, imag};
}

// Circle point on x^2 + y^2 = 1
struct CirclePoint {
    M31 x;
    M31 y;
};

CirclePoint circle_identity() { return CirclePoint{m31_one(), m31_zero()}; }

CirclePoint circle_mul(CirclePoint a, CirclePoint b) {
    M31 x = m31_sub(m31_mul(a.x, b.x), m31_mul(a.y, b.y));
    M31 y = m31_add(m31_mul(a.x, b.y), m31_mul(a.y, b.x));
    return CirclePoint{x, y};
}

#endif // MERSENNE31_METAL
