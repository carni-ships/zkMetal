// Binary tower field arithmetic for Metal GPU
// GF(2^8) → GF(2^16) → GF(2^32)
// Addition = XOR (1 cycle), Multiplication = carry-less multiply + Karatsuba tower
// Foundation for Binius-style binary STARKs on GPU
//
// Key advantage over M31/BabyBear: addition is literally XOR (no modular reduction),
// and the entire tower is built from 8-bit carry-less multiply as the base.

#ifndef BINARY_TOWER_METAL
#define BINARY_TOWER_METAL

#include <metal_stdlib>
using namespace metal;

// ========================================================================
// GF(2^8) — AES field, irreducible: x^8 + x^4 + x^3 + x + 1 (0x11B)
// ========================================================================

struct GF8 {
    uint8_t v;
};

GF8 gf8_zero() { return GF8{0}; }
GF8 gf8_one() { return GF8{1}; }
bool gf8_is_zero(GF8 a) { return a.v == 0; }

// Addition = XOR
GF8 gf8_add(GF8 a, GF8 b) { return GF8{uint8_t(a.v ^ b.v)}; }
GF8 gf8_sub(GF8 a, GF8 b) { return GF8{uint8_t(a.v ^ b.v)}; }  // same as add in char 2

// Carry-less multiply with reduction by x^8+x^4+x^3+x+1
// 8 iterations of shift-XOR — the critical inner operation
inline GF8 gf8_mul(GF8 a, GF8 b) {
    uint8_t result = 0;
    uint8_t aa = a.v;
    uint8_t bb = b.v;
    // Unrolled for GPU performance
    if (bb & 1) result ^= aa;
    uint8_t carry = aa >> 7; aa = (aa << 1) ^ (carry * 0x1Bu); bb >>= 1;
    if (bb & 1) result ^= aa;
    carry = aa >> 7; aa = (aa << 1) ^ (carry * 0x1Bu); bb >>= 1;
    if (bb & 1) result ^= aa;
    carry = aa >> 7; aa = (aa << 1) ^ (carry * 0x1Bu); bb >>= 1;
    if (bb & 1) result ^= aa;
    carry = aa >> 7; aa = (aa << 1) ^ (carry * 0x1Bu); bb >>= 1;
    if (bb & 1) result ^= aa;
    carry = aa >> 7; aa = (aa << 1) ^ (carry * 0x1Bu); bb >>= 1;
    if (bb & 1) result ^= aa;
    carry = aa >> 7; aa = (aa << 1) ^ (carry * 0x1Bu); bb >>= 1;
    if (bb & 1) result ^= aa;
    carry = aa >> 7; aa = (aa << 1) ^ (carry * 0x1Bu); bb >>= 1;
    if (bb & 1) result ^= aa;
    return GF8{result};
}

// Square (just mul by itself, but could be optimized with Frobenius)
inline GF8 gf8_sqr(GF8 a) { return gf8_mul(a, a); }

// Inverse via Fermat: a^254 = a^(-1) in GF(2^8) since |GF(2^8)*| = 255
inline GF8 gf8_inv(GF8 a) {
    // a^254 = ((a^2)^2 * a)^2)^2 * a)^2)^2 ...
    // Addition chain for 254 = 2*127 = 2*(128-1)
    // a^2, a^3=a^2*a, a^6=(a^3)^2, a^7=a^6*a, a^14=(a^7)^2,
    // a^15=a^14*a, a^30=(a^15)^2, a^31=a^30*a, a^62=(a^31)^2,
    // a^63=a^62*a, a^126=(a^63)^2, a^127=a^126*a, a^254=(a^127)^2
    GF8 a2 = gf8_sqr(a);
    GF8 a3 = gf8_mul(a2, a);
    GF8 a6 = gf8_sqr(a3);
    GF8 a7 = gf8_mul(a6, a);
    GF8 a14 = gf8_sqr(a7);
    GF8 a15 = gf8_mul(a14, a);
    GF8 a30 = gf8_sqr(a15);
    GF8 a31 = gf8_mul(a30, a);
    GF8 a62 = gf8_sqr(a31);
    GF8 a63 = gf8_mul(a62, a);
    GF8 a126 = gf8_sqr(a63);
    GF8 a127 = gf8_mul(a126, a);
    return gf8_sqr(a127);
}

// ========================================================================
// GF(2^16) = GF(2^8)[X] / (X^2 + X + α), α = 0x2B
// ========================================================================

struct GF16 {
    GF8 lo;  // coefficient of 1
    GF8 hi;  // coefficient of X
};

constant GF8 GF16_ALPHA = GF8{0x2Bu};  // tower extension parameter

GF16 gf16_zero() { return GF16{gf8_zero(), gf8_zero()}; }
GF16 gf16_one() { return GF16{gf8_one(), gf8_zero()}; }
bool gf16_is_zero(GF16 a) { return gf8_is_zero(a.lo) && gf8_is_zero(a.hi); }

GF16 gf16_add(GF16 a, GF16 b) {
    return GF16{gf8_add(a.lo, b.lo), gf8_add(a.hi, b.hi)};
}

GF16 gf16_sub(GF16 a, GF16 b) { return gf16_add(a, b); }

// Karatsuba multiplication:
// (a + bX)(c + dX) mod (X^2+X+α)
// = (ac + bd·α) + ((a+b)(c+d) + ac)·X
inline GF16 gf16_mul(GF16 a, GF16 b) {
    GF8 ac = gf8_mul(a.lo, b.lo);
    GF8 bd = gf8_mul(a.hi, b.hi);
    GF8 e = gf8_mul(gf8_add(a.lo, a.hi), gf8_add(b.lo, b.hi));
    GF8 bdAlpha = gf8_mul(bd, GF16_ALPHA);
    return GF16{gf8_add(ac, bdAlpha), gf8_add(e, ac)};
}

inline GF16 gf16_sqr(GF16 a) { return gf16_mul(a, a); }

// Inverse: use norm to GF(2^8)
// N = a^2 + ab + b^2·α, then (a+bX)^(-1) = N^(-1)·((a+b) + bX)
inline GF16 gf16_inv(GF16 a) {
    GF8 a2 = gf8_mul(a.lo, a.lo);
    GF8 ab = gf8_mul(a.lo, a.hi);
    GF8 b2a = gf8_mul(gf8_mul(a.hi, a.hi), GF16_ALPHA);
    GF8 norm = gf8_add(gf8_add(a2, ab), b2a);
    GF8 ni = gf8_inv(norm);
    return GF16{gf8_mul(gf8_add(a.lo, a.hi), ni), gf8_mul(a.hi, ni)};
}

// ========================================================================
// GF(2^32) = GF(2^16)[X] / (X^2 + X + β), β = (0x02, 0x01) in GF(2^16)
// ========================================================================

struct GF32 {
    GF16 lo;
    GF16 hi;
};

constant GF16 GF32_BETA = GF16{GF8{0x02u}, GF8{0x01u}};

GF32 gf32_zero() { return GF32{gf16_zero(), gf16_zero()}; }
GF32 gf32_one() { return GF32{gf16_one(), gf16_zero()}; }
bool gf32_is_zero(GF32 a) { return gf16_is_zero(a.lo) && gf16_is_zero(a.hi); }

GF32 gf32_add(GF32 a, GF32 b) {
    return GF32{gf16_add(a.lo, b.lo), gf16_add(a.hi, b.hi)};
}

GF32 gf32_sub(GF32 a, GF32 b) { return gf32_add(a, b); }

inline GF32 gf32_mul(GF32 a, GF32 b) {
    GF16 ac = gf16_mul(a.lo, b.lo);
    GF16 bd = gf16_mul(a.hi, b.hi);
    GF16 e = gf16_mul(gf16_add(a.lo, a.hi), gf16_add(b.lo, b.hi));
    GF16 bdBeta = gf16_mul(bd, GF32_BETA);
    return GF32{gf16_add(ac, bdBeta), gf16_add(e, ac)};
}

inline GF32 gf32_sqr(GF32 a) { return gf32_mul(a, a); }

inline GF32 gf32_inv(GF32 a) {
    GF16 a2 = gf16_mul(a.lo, a.lo);
    GF16 ab = gf16_mul(a.lo, a.hi);
    GF16 b2b = gf16_mul(gf16_mul(a.hi, a.hi), GF32_BETA);
    GF16 norm = gf16_add(gf16_add(a2, ab), b2b);
    GF16 ni = gf16_inv(norm);
    return GF32{gf16_mul(gf16_add(a.lo, a.hi), ni), gf16_mul(a.hi, ni)};
}

// ========================================================================
// GPU Compute Kernels — Batch operations over arrays
// ========================================================================

// Pack/unpack: GF(2^32) as uint for buffer transport
inline GF32 gf32_from_uint(uint v) {
    GF8 b0 = GF8{uint8_t(v & 0xFF)};
    GF8 b1 = GF8{uint8_t((v >> 8) & 0xFF)};
    GF8 b2 = GF8{uint8_t((v >> 16) & 0xFF)};
    GF8 b3 = GF8{uint8_t((v >> 24) & 0xFF)};
    return GF32{GF16{b0, b1}, GF16{b2, b3}};
}

inline uint gf32_to_uint(GF32 a) {
    return uint(a.lo.lo.v) | (uint(a.lo.hi.v) << 8) |
           (uint(a.hi.lo.v) << 16) | (uint(a.hi.hi.v) << 24);
}

// Batch add: out[i] = a[i] + b[i] (XOR — should be bandwidth-bound)
kernel void bt_batch_add(
    device const uint* a [[buffer(0)]],
    device const uint* b [[buffer(1)]],
    device uint* out     [[buffer(2)]],
    uint tid             [[thread_position_in_grid]]
) {
    out[tid] = a[tid] ^ b[tid];
}

// Batch multiply: out[i] = a[i] * b[i] in GF(2^32)
kernel void bt_batch_mul(
    device const uint* a [[buffer(0)]],
    device const uint* b [[buffer(1)]],
    device uint* out     [[buffer(2)]],
    uint tid             [[thread_position_in_grid]]
) {
    GF32 x = gf32_from_uint(a[tid]);
    GF32 y = gf32_from_uint(b[tid]);
    GF32 r = gf32_mul(x, y);
    out[tid] = gf32_to_uint(r);
}

// Batch multiply-accumulate: out[i] += a[i] * b[i] in GF(2^32)
kernel void bt_batch_mul_acc(
    device const uint* a [[buffer(0)]],
    device const uint* b [[buffer(1)]],
    device uint* out     [[buffer(2)]],
    uint tid             [[thread_position_in_grid]]
) {
    GF32 x = gf32_from_uint(a[tid]);
    GF32 y = gf32_from_uint(b[tid]);
    GF32 r = gf32_mul(x, y);
    out[tid] = out[tid] ^ gf32_to_uint(r);
}

// Batch inverse: out[i] = a[i]^(-1) in GF(2^32)
kernel void bt_batch_inv(
    device const uint* a [[buffer(0)]],
    device uint* out     [[buffer(1)]],
    uint tid             [[thread_position_in_grid]]
) {
    GF32 x = gf32_from_uint(a[tid]);
    GF32 r = gf32_inv(x);
    out[tid] = gf32_to_uint(r);
}

// Additive FFT butterfly: (a, b) -> (a + b, b * twiddle)
// For binary field additive FFT, the butterfly is cheaper than standard NTT
kernel void bt_additive_butterfly(
    device uint* data       [[buffer(0)]],
    device const uint* tw   [[buffer(1)]],  // twiddle factors
    constant uint& half_n   [[buffer(2)]],
    constant uint& stride   [[buffer(3)]],
    uint tid                [[thread_position_in_grid]]
) {
    uint block = tid / half_n;
    uint idx_in_block = tid % half_n;
    uint i = block * (half_n * 2) + idx_in_block;
    uint j = i + half_n;

    GF32 a = gf32_from_uint(data[i]);
    GF32 b = gf32_from_uint(data[j]);
    GF32 t = gf32_from_uint(tw[idx_in_block]);

    GF32 sum = gf32_add(a, b);
    GF32 bt = gf32_mul(b, t);

    data[i] = gf32_to_uint(sum);
    data[j] = gf32_to_uint(bt);
}

#endif // BINARY_TOWER_METAL
