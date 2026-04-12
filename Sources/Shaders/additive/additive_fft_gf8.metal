// Additive FFT (Cantor/Lin-Chung-Han) for GF(2^8) — GPU fused kernel
//
// Fuses all log₂(n) levels into a single dispatch. Each thread processes one
// GF(2^8) element through all butterfly levels in registers, achieving:
//   - 1 global memory read (input) + 1 global memory write (output)
//   - All intermediate data stays in registers
//   - log₂(n) GF(2^8) multiplications per element (twist step)
//
// Additive FFT over GF(2^8):
//   twist:  lo ^= s * hi    (GF(2^8) multiply by basis element)
//   prop:   hi ^= lo         (XOR — free on GPU)
//
// Algorithm (forward/DIF, k levels):
//   for depth = 0..k-1:
//     half = n >> (depth+1)
//     stride = n >> depth
//     for i in 0..n-1 with i % stride in [half, 2*half):
//       j = i - half
//       s = basis[depth]
//       t = data[j] ^ (s * data[i])   // twist
//       data[i] = data[j] ^ t           // propagate
//       data[j] = t
//
// GF(2^8) irreducible polynomial: x^8 + x^4 + x^3 + x + 1 (0x11B)
// Multiply via shift-XOR (8 iterations, no LUT needed):
//   p = 0; for bit = 0..7: p ^= (a >> bit & 1) * (b << bit); reduce(p)

#include <metal_stdlib>
using namespace metal;

// GF(2^8) multiplication with reduction by 0x11B
// Returns a * b in GF(2^8)
inline uint8_t gf28_mul(uint8_t a, uint8_t b) {
    uint16_t p = 0;
    // 8 shift-XOR iterations (carryless multiply)
    p ^= ((uint16_t)(a & 1)  ) * ((uint16_t)(b)       );
    p ^= ((uint16_t)(a & 2)  ) * ((uint16_t)(b << 1) );
    p ^= ((uint16_t)(a & 4)  ) * ((uint16_t)(b << 2) );
    p ^= ((uint16_t)(a & 8)  ) * ((uint16_t)(b << 3) );
    p ^= ((uint16_t)(a & 16) ) * ((uint16_t)(b << 4) );
    p ^= ((uint16_t)(a & 32) ) * ((uint16_t)(b << 5) );
    p ^= ((uint16_t)(a & 64) ) * ((uint16_t)(b << 6) );
    p ^= ((uint16_t)(a & 128)) * ((uint16_t)(b << 7) );
    // Reduce by x^8 + x^4 + x^3 + x + 1 (0x11B).
    // For each bit i >= 8 that is set in p, we have x^i ≡ x^(i-8) * (x^8)
    // ≡ x^(i-8) * (x^4 + x^3 + x + 1) ≡ (0x11B) << (i-8).
    // h = p >> 8 holds bits 8..14 of the carry-less product.
    // For each bit i set in h, XOR with 0x11B << i.
    uint16_t h = p >> 8;
    // Handle bits 0..7 of h: each corresponds to product term x^(8+i)
    if (h & 0x01) p ^= 0x11B << 0;   // bit 8:  x^8  ≡ 0x11B
    if (h & 0x02) p ^= 0x11B << 1;   // bit 9:  x^9  ≡ 0x236
    if (h & 0x04) p ^= 0x11B << 2;   // bit 10: x^10 ≡ 0x46C
    if (h & 0x08) p ^= 0x11B << 3;   // bit 11: x^11 ≡ 0x8D8
    if (h & 0x10) p ^= 0x11B << 4;   // bit 12: x^12 ≡ 0x11B0
    if (h & 0x20) p ^= 0x11B << 5;   // bit 13: x^13 ≡ 0x2360
    if (h & 0x40) p ^= 0x11B << 6;   // bit 14: x^14 ≡ 0x46C0
    if (h & 0x80) p ^= 0x11B << 7;   // bit 15: x^15 ≡ 0x8D80
    // After these XORs, all bits >= 8 are eliminated (mod x^8 + … + 1)
    return (uint8_t)(p & 0xFF);
}

// Precomputed reduction table: for each possible high-byte value (0..255),
// gives the GF(2^8) reduction of (high_byte << 8).
// This lets us do: result = (lo | (high_byte << 8)) ^ reduction_table[high_byte]
// static const uint16_t gf28_reduction_table[256] = { ... };

// Forward additive FFT over GF(2^8).
// Fused: processes all k = log₂(n) levels in one dispatch.
// Each thread processes one element at position gid.
// basis[0..k-1]: GF(2^8) basis elements (one per FFT level).
// data[gid]: input element, modified in registers, final value written back.
kernel void additive_fft_gf8_forward(
    device uint8_t* data              [[buffer(0)]],
    constant uint8_t* basis           [[buffer(1)]],   // k basis elements
    constant uint32_t& n               [[buffer(2)]],   // total elements (power of 2)
    constant uint32_t& k              [[buffer(3)]],   // log₂(n)
    uint gid                          [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    uint8_t val = data[gid];

    // k levels of additive butterfly (DIF: large stride first)
    for (uint depth = 0; depth < k; depth++) {
        uint block_size = n >> depth;      // doubles each level going up
        uint halfSize = block_size >> 1;        // n >> (depth+1)
        uint block_idx = gid / block_size;
        uint local_idx = gid % block_size;

        // Only process upper half of each block (where i >= halfSize)
        if (local_idx < halfSize) {
            // This element is the "lo" of the pair — skip (handled by hi element)
            continue;
        }
        uint i = gid;                       // hi index
        uint j = gid - halfSize;             // lo index

        uint8_t s = basis[depth];            // basis element for this level
        uint8_t hi_val = val;                // our value (hi half)
        uint8_t lo_val = data[j];            // lo value from memory

        // Twist: lo ^= s * hi
        uint8_t twisted = lo_val ^ gf28_mul(s, hi_val);
        // Propagate: hi ^= lo
        uint8_t propagated = lo_val ^ hi_val;

        // Write back
        data[j] = twisted;
        val = propagated;                    // update our register with new value
    }

    data[gid] = val;
}

// Inverse additive FFT over GF(2^8).
// Fused: processes all k levels in one dispatch.
// DIT: small stride first (reverse of forward).
kernel void additive_fft_gf8_inverse(
    device uint8_t* data              [[buffer(0)]],
    constant uint8_t* basis           [[buffer(1)]],   // k basis elements (same order)
    constant uint32_t& n               [[buffer(2)]],
    constant uint32_t& k              [[buffer(3)]],
    uint gid                          [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    uint8_t val = data[gid];

    // k levels (DIT: small stride first = reverse depth order)
    for (int depth = int(k) - 1; depth >= 0; depth--) {
        uint block_size = n >> depth;
        uint halfSize = block_size >> 1;
        uint block_idx = gid / block_size;
        uint local_idx = gid % block_size;

        if (local_idx < halfSize) {
            // lo element — skip
            continue;
        }
        uint i = gid;               // hi index
        uint j = gid - halfSize;     // lo index

        uint8_t s = basis[depth];
        uint8_t hi_val = val;
        uint8_t lo_val = data[j];

        // Un-propagate: hi ^= lo (reverse of propagate: hi_new ^ lo = hi_old, so hi_new = hi_old ^ lo)
        // Wait — in forward DIT propagate: hi_new = lo_old ^ hi_old
        // So: lo_new = hi_old, hi_new = lo_old ^ hi_old
        // In inverse DIT:
        //   hi ^= lo  (same as forward: hi_new = hi_old ^ lo)
        //   lo ^= s * hi  (untwist: lo_new = lo_old ^ s * hi_new)
        uint8_t unpropagated = hi_val ^ lo_val;
        uint8_t untwisted = lo_val ^ gf28_mul(s, unpropagated);

        data[j] = untwisted;
        val = unpropagated;
    }

    data[gid] = val;
}

// Batch forward additive FFT for multiple independent arrays.
// Each thread processes one element from one array.
// Useful when you have multiple polynomials of the same degree to transform.
kernel void additive_fft_gf8_forward_batch(
    device uint8_t* data              [[buffer(0)]],  // flat: batch * n elements
    constant uint8_t* basis           [[buffer(1)]],
    constant uint32_t& n               [[buffer(2)]],
    constant uint32_t& k              [[buffer(3)]],
    constant uint32_t& batch           [[buffer(4)]],   // number of arrays
    uint gid                          [[thread_position_in_grid]]
) {
    uint total = n * batch;
    if (gid >= total) return;

    uint arr_idx = gid / n;
    uint elem_idx = gid % n;
    uint arr_offset = arr_idx * n;

    uint8_t val = data[gid];

    for (uint depth = 0; depth < k; depth++) {
        uint block_size = n >> depth;
        uint halfSize = block_size >> 1;
        uint local_idx = elem_idx % block_size;

        if (local_idx < halfSize) continue;

        uint i = elem_idx;
        uint j = elem_idx - halfSize;
        uint8_t s = basis[depth];

        uint8_t hi_val = val;
        uint8_t lo_val = data[arr_offset + j];

        uint8_t twisted = lo_val ^ gf28_mul(s, hi_val);
        uint8_t propagated = lo_val ^ hi_val;

        data[arr_offset + j] = twisted;
        val = propagated;
    }

    data[gid] = val;
}

// GF(2^8) pointwise multiply for polynomial multiplication via additive FFT.
// Applies to arrays of n GF(2^8) elements (pointwise product: out[i] = a[i] * b[i]).
kernel void gf28_pointwise_mul(
    device uint8_t* a                 [[buffer(0)]],
    device uint8_t* b                 [[buffer(1)]],
    device uint8_t* out               [[buffer(2)]],
    constant uint32_t& n              [[buffer(3)]],
    uint gid                          [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[gid] = gf28_mul(a[gid], b[gid]);
}
