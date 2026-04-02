// Goldilocks field arithmetic for Metal GPU
// p = 2^64 - 2^32 + 1 = 0xFFFFFFFF00000001
// epsilon = 2^32 - 1 (2^64 ≡ epsilon mod p)

#ifndef GOLDILOCKS_METAL
#define GOLDILOCKS_METAL

#include <metal_stdlib>
using namespace metal;

constant ulong GL_P = 0xFFFFFFFF00000001UL;
constant ulong GL_EPS = 0xFFFFFFFFUL;

struct Gl {
    ulong v;
};

Gl gl_zero() { return Gl{0}; }
Gl gl_one() { return Gl{1}; }
bool gl_is_zero(Gl a) { return a.v == 0; }

Gl gl_from_u64(ulong v) {
    return Gl{v >= GL_P ? v - GL_P : v};
}

Gl gl_add(Gl a, Gl b) {
    ulong sum = a.v + b.v;
    // If a+b overflows 64 bits, add epsilon (since 2^64 ≡ eps mod p)
    // Double overflow impossible: max sum after carry = 2^64 - 2^32 - 1 < 2^64
    sum += (sum < a.v) ? GL_EPS : 0UL;
    return Gl{sum >= GL_P ? sum - GL_P : sum};
}

Gl gl_sub(Gl a, Gl b) {
    if (a.v >= b.v) return Gl{a.v - b.v};
    return Gl{a.v + GL_P - b.v};
}

Gl gl_neg(Gl a) {
    if (a.v == 0) return a;
    return Gl{GL_P - a.v};
}

// Multiply using schoolbook 32x32→64 for 128-bit product, then reduce
// Reduction: val = hi*2^64 + lo ≡ lo + hi_lo*eps - hi_hi (mod p)
Gl gl_mul(Gl a, Gl b) {
    uint a0 = uint(a.v);
    uint a1 = uint(a.v >> 32);
    uint b0 = uint(b.v);
    uint b1 = uint(b.v >> 32);

    ulong t0 = ulong(a0) * ulong(b0);
    ulong t1 = ulong(a0) * ulong(b1);
    ulong t2 = ulong(a1) * ulong(b0);
    ulong t3 = ulong(a1) * ulong(b1);

    ulong mid = t1 + t2;
    bool mid_carry = mid < t1;

    ulong lo = t0 + (mid << 32);
    bool lo_carry = lo < t0;

    ulong hi = t3 + (mid >> 32) + (lo_carry ? 1UL : 0UL);
    if (mid_carry) hi += (1UL << 32);

    // Reduce: result ≡ lo + hi_lo * eps - hi_hi (mod p)
    // where eps = 2^32 - 1, so hi_lo * eps = (hi_lo << 32) - hi_lo
    uint hi_lo = uint(hi);
    uint hi_hi = uint(hi >> 32);

    ulong hi_lo_eps = (ulong(hi_lo) << 32) - ulong(hi_lo);
    ulong s = lo + hi_lo_eps;
    bool c1 = s < lo;

    ulong r = s - ulong(hi_hi);
    bool b2 = s < ulong(hi_hi);

    // Branchless corrections using select
    r += c1 ? GL_EPS : 0UL;
    r += b2 ? GL_P : 0UL;
    r -= (r >= GL_P) ? GL_P : 0UL;

    return Gl{r};
}

Gl gl_sqr(Gl a) { return gl_mul(a, a); }

#endif // GOLDILOCKS_METAL
