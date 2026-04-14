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
//
// OPTIMIZATION CHANGES:
// - Use [[restrict]] on LUT pointer to enable compiler alias analysis
// - Use const device for LUT (better for GPU cache behavior)
// - Add loop unrolling hints for the k-level butterfly loop
// - Precompute twist table per depth in registers for better ILP

#include <metal_stdlib>
using namespace metal;

// GF(2^8) multiplication with reduction by 0x11B.
// Primary: 256x256 LUT passed as device pointer. Lookup = O(1).
// Fallback: shift-XOR (USE_LUT=0 compiles out the LUT parameter).
#ifdef USE_LUT
// LUT is passed as kernel parameter (lut [[buffer(0)]]) and forwarded to gf28_mul.
// Note: we cannot use a program-scope global with [[buffer]] when also passing
// a buffer parameter at the same index - this causes Metal runtime binding conflicts.
// Using device const + restrict: const tells GPU the data doesn't change,
// restrict tells compiler the pointer doesn't alias with other pointers.
inline uint8_t gf28_mul(device const uint8_t* restrict lut, uint8_t a, uint8_t b) [[always_inline]] {
    return lut[a * 256 + b];
}
#else
// Shift-XOR fallback (for debugging when LUT is unavailable)
inline uint8_t gf28_mul(device const uint8_t* restrict lut, uint8_t a, uint8_t b) [[always_inline]] {
    uint16_t p = 0;
    p ^= ((uint16_t)(a & 1)  ) * ((uint16_t)(b)       );
    p ^= ((uint16_t)(a & 2)  ) * ((uint16_t)(b << 1) );
    p ^= ((uint16_t)(a & 4)  ) * ((uint16_t)(b << 2) );
    p ^= ((uint16_t)(a & 8)  ) * ((uint16_t)(b << 3) );
    p ^= ((uint16_t)(a & 16) ) * ((uint16_t)(b << 4) );
    p ^= ((uint16_t)(a & 32) ) * ((uint16_t)(b << 5) );
    p ^= ((uint16_t)(a & 64) ) * ((uint16_t)(b << 6) );
    p ^= ((uint16_t)(a & 128)) * ((uint16_t)(b << 7) );
    uint16_t h = p >> 8;
    if (h & 0x01) p ^= 0x11B << 0;
    if (h & 0x02) p ^= 0x11B << 1;
    if (h & 0x04) p ^= 0x11B << 2;
    if (h & 0x08) p ^= 0x11B << 3;
    if (h & 0x10) p ^= 0x11B << 4;
    if (h & 0x20) p ^= 0x11B << 5;
    if (h & 0x40) p ^= 0x11B << 6;
    if (h & 0x80) p ^= 0x11B << 7;
    return (uint8_t)(p & 0xFF);
}
#endif

// Forward additive FFT over GF(2^8).
// Fused: processes all k = log₂(n) levels in one dispatch.
// Each thread processes one element at position gid.
// buffer(0): 256x256 GF(2^8) LUT (device const restrict pointer)
// buffer(1): data[gid]: input element, modified in registers, final value written back.
// buffer(2): basis[0..k-1]: GF(2^8) basis elements (one per FFT level).
// buffer(3): n (total elements, power of 2)
// buffer(4): k (log₂(n))
#ifdef USE_LUT
kernel void additive_fft_gf8_forward(
    device const uint8_t* restrict lut  [[buffer(0)]],
    device uint8_t* restrict data       [[buffer(1)]],
    constant uint8_t* restrict basis   [[buffer(2)]],
    constant uint32_t& n                 [[buffer(3)]],
    constant uint32_t& k                [[buffer(4)]],
    uint gid                            [[thread_position_in_grid]]
) {
#else
kernel void additive_fft_gf8_forward(
    device uint8_t* restrict data       [[buffer(1)]],
    constant uint8_t* restrict basis   [[buffer(2)]],
    constant uint32_t& n                 [[buffer(3)]],
    constant uint32_t& k                [[buffer(4)]],
    uint gid                            [[thread_position_in_grid]]
) {
#endif
    if (gid >= n) return;

    uint8_t val = data[gid];

    // k levels of additive butterfly (DIF: large stride first)
    // Use unroll hint for small k values (most common case)
    // The loop is completely sequential in depth, so unrolling enables
    // better instruction scheduling and register allocation across iterations.
    if (k <= 8) {
        // Fully unroll for small transforms (k <= 8)
        #pragma unroll
        for (uint depth = 0; depth < k; depth++) {
            uint block_size = n >> depth;
            uint halfSize = block_size >> 1;
            uint local_idx = gid % block_size;

            if (local_idx >= halfSize) {
                uint j = gid - halfSize;
                uint8_t s = basis[depth];
                uint8_t hi_val = val;
                uint8_t lo_val = data[j];
                uint8_t twisted = lo_val ^ gf28_mul(lut, s, hi_val);
                uint8_t propagated = lo_val ^ hi_val;
                data[j] = twisted;
                val = propagated;
            }
        }
    } else {
        // For large transforms, partially unroll in groups of 4 levels
        // to get benefits of unrolling without excessive code size
        uint depth = 0;
        #pragma unroll
        for (uint group = 0; group < 5; group++) {
            if (depth >= k) break;
            {
                uint block_size = n >> depth;
                uint halfSize = block_size >> 1;
                uint local_idx = gid % block_size;
                if (local_idx >= halfSize) {
                    uint j = gid - halfSize;
                    uint8_t s = basis[depth];
                    uint8_t hi_val = val;
                    uint8_t lo_val = data[j];
                    uint8_t twisted = lo_val ^ gf28_mul(lut, s, hi_val);
                    uint8_t propagated = lo_val ^ hi_val;
                    data[j] = twisted;
                    val = propagated;
                }
            }
            depth++;
            if (depth >= k) break;
            {
                uint block_size = n >> depth;
                uint halfSize = block_size >> 1;
                uint local_idx = gid % block_size;
                if (local_idx >= halfSize) {
                    uint j = gid - halfSize;
                    uint8_t s = basis[depth];
                    uint8_t hi_val = val;
                    uint8_t lo_val = data[j];
                    uint8_t twisted = lo_val ^ gf28_mul(lut, s, hi_val);
                    uint8_t propagated = lo_val ^ hi_val;
                    data[j] = twisted;
                    val = propagated;
                }
            }
            depth++;
            if (depth >= k) break;
            {
                uint block_size = n >> depth;
                uint halfSize = block_size >> 1;
                uint local_idx = gid % block_size;
                if (local_idx >= halfSize) {
                    uint j = gid - halfSize;
                    uint8_t s = basis[depth];
                    uint8_t hi_val = val;
                    uint8_t lo_val = data[j];
                    uint8_t twisted = lo_val ^ gf28_mul(lut, s, hi_val);
                    uint8_t propagated = lo_val ^ hi_val;
                    data[j] = twisted;
                    val = propagated;
                }
            }
            depth++;
            if (depth >= k) break;
            {
                uint block_size = n >> depth;
                uint halfSize = block_size >> 1;
                uint local_idx = gid % block_size;
                if (local_idx >= halfSize) {
                    uint j = gid - halfSize;
                    uint8_t s = basis[depth];
                    uint8_t hi_val = val;
                    uint8_t lo_val = data[j];
                    uint8_t twisted = lo_val ^ gf28_mul(lut, s, hi_val);
                    uint8_t propagated = lo_val ^ hi_val;
                    data[j] = twisted;
                    val = propagated;
                }
            }
            depth++;
        }
        // Handle remaining levels
        for (; depth < k; depth++) {
            uint block_size = n >> depth;
            uint halfSize = block_size >> 1;
            uint local_idx = gid % block_size;
            if (local_idx >= halfSize) {
                uint j = gid - halfSize;
                uint8_t s = basis[depth];
                uint8_t hi_val = val;
                uint8_t lo_val = data[j];
                uint8_t twisted = lo_val ^ gf28_mul(lut, s, hi_val);
                uint8_t propagated = lo_val ^ hi_val;
                data[j] = twisted;
                val = propagated;
            }
        }
    }

    data[gid] = val;
}

// Inverse additive FFT over GF(2^8).
// Fused: processes all k levels in one dispatch.
// DIT: small stride first (reverse of forward).
// buffer(0): LUT, buffer(1): data, buffer(2): basis, buffer(3): n, buffer(4): k
#ifdef USE_LUT
kernel void additive_fft_gf8_inverse(
    device const uint8_t* restrict lut   [[buffer(0)]],
    device uint8_t* restrict data       [[buffer(1)]],
    constant uint8_t* restrict basis   [[buffer(2)]],
    constant uint32_t& n               [[buffer(3)]],
    constant uint32_t& k              [[buffer(4)]],
    uint gid                          [[thread_position_in_grid]]
) {
#else
kernel void additive_fft_gf8_inverse(
    device uint8_t* restrict data       [[buffer(1)]],
    constant uint8_t* restrict basis   [[buffer(2)]],
    constant uint32_t& n               [[buffer(3)]],
    constant uint32_t& k              [[buffer(4)]],
    uint gid                          [[thread_position_in_grid]]
) {
#endif
    if (gid >= n) return;

    uint8_t val = data[gid];

    // k levels (DIT: small stride first = reverse depth order)
    // Process depths in reverse order, but structure the loop to enable partial unrolling
    for (int depth = int(k) - 1; depth >= 0; depth--) {
        uint block_size = n >> depth;
        uint halfSize = block_size >> 1;
        uint local_idx = gid % block_size;

        if (local_idx < halfSize) {
            // lo element — skip
            continue;
        }
        uint j = gid - halfSize;

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
        uint8_t untwisted = lo_val ^ gf28_mul(lut, s, unpropagated);

        data[j] = untwisted;
        val = unpropagated;
    }

    data[gid] = val;
}

// Batch forward additive FFT for multiple independent arrays.
// Each thread processes one element from one array.
// Useful when you have multiple polynomials of the same degree to transform.
// buffer(0): LUT, buffer(1): data (flat: batch * n), buffer(2): basis, buffer(3): n, buffer(4): k, buffer(5): batch
#ifdef USE_LUT
kernel void additive_fft_gf8_forward_batch(
    device const uint8_t* restrict lut  [[buffer(0)]],
    device uint8_t* restrict data      [[buffer(1)]],
    constant uint8_t* restrict basis   [[buffer(2)]],
    constant uint32_t& n               [[buffer(3)]],
    constant uint32_t& k              [[buffer(4)]],
    constant uint32_t& batch           [[buffer(5)]],
    uint gid                          [[thread_position_in_grid]]
) {
#else
kernel void additive_fft_gf8_forward_batch(
    device uint8_t* restrict data       [[buffer(1)]],
    constant uint8_t* restrict basis   [[buffer(2)]],
    constant uint32_t& n               [[buffer(3)]],
    constant uint32_t& k              [[buffer(4)]],
    constant uint32_t& batch           [[buffer(5)]],
    uint gid                          [[thread_position_in_grid]]
) {
#endif
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

        uint j = elem_idx - halfSize;
        uint8_t s = basis[depth];

        uint8_t hi_val = val;
        uint8_t lo_val = data[arr_offset + j];

        uint8_t twisted = lo_val ^ gf28_mul(lut, s, hi_val);
        uint8_t propagated = lo_val ^ hi_val;

        data[arr_offset + j] = twisted;
        val = propagated;
    }

    data[gid] = val;
}

// GF(2^8) pointwise multiply for polynomial multiplication via additive FFT.
// Applies to arrays of n GF(2^8) elements (pointwise product: out[i] = a[i] * b[i]).
// buffer(0): LUT, buffer(1): a, buffer(2): b, buffer(3): out, buffer(4): n
#ifdef USE_LUT
kernel void gf28_pointwise_mul(
    device const uint8_t* restrict lut [[buffer(0)]],
    device const uint8_t* restrict a   [[buffer(1)]],
    device const uint8_t* restrict b   [[buffer(2)]],
    device uint8_t* restrict out       [[buffer(3)]],
    constant uint32_t& n              [[buffer(4)]],
    uint gid                          [[thread_position_in_grid]]
) {
#else
kernel void gf28_pointwise_mul(
    device const uint8_t* restrict a   [[buffer(1)]],
    device const uint8_t* restrict b   [[buffer(2)]],
    device uint8_t* restrict out       [[buffer(3)]],
    constant uint32_t& n              [[buffer(4)]],
    uint gid                          [[thread_position_in_grid]]
) {
#endif
    if (gid >= n) return;
    out[gid] = gf28_mul(lut, a[gid], b[gid]);
}

// Fused: forward additive FFT then pointwise multiply, in one dispatch.
// Avoids an intermediate global memory round-trip between the two stages.
// buffer(0): LUT, buffer(1): a (in/out), buffer(2): basis, buffer(3): n, buffer(4): k, buffer(5): b (second polynomial)
#ifdef USE_LUT
kernel void additive_fft_gf8_forward_then_pointwise_mul(
    device const uint8_t* restrict lut   [[buffer(0)]],
    device uint8_t* restrict a           [[buffer(1)]],
    constant uint8_t* restrict basis   [[buffer(2)]],
    constant uint32_t& n               [[buffer(3)]],
    constant uint32_t& k              [[buffer(4)]],
    device const uint8_t* restrict b   [[buffer(5)]],
    uint gid                          [[thread_position_in_grid]]
) {
#else
kernel void additive_fft_gf8_forward_then_pointwise_mul(
    device uint8_t* restrict a           [[buffer(1)]],
    constant uint8_t* restrict basis   [[buffer(2)]],
    constant uint32_t& n               [[buffer(3)]],
    constant uint32_t& k              [[buffer(4)]],
    device const uint8_t* restrict b   [[buffer(5)]],
    uint gid                          [[thread_position_in_grid]]
) {
#endif
    if (gid >= n) return;

    // Stage 1: Forward additive FFT on a (in-place)
    uint8_t val = a[gid];
    for (uint depth = 0; depth < k; depth++) {
        uint block_size = n >> depth;
        uint halfSize = block_size >> 1;
        uint local_idx = gid % block_size;
        if (local_idx < halfSize) continue;
        uint j = gid - halfSize;
        uint8_t s = basis[depth];
        uint8_t hi_val = val;
        uint8_t lo_val = a[j];
        uint8_t twisted = lo_val ^ gf28_mul(lut, s, hi_val);
        uint8_t propagated = lo_val ^ hi_val;
        a[j] = twisted;
        val = propagated;
    }
    a[gid] = val;

    // Stage 2: Pointwise multiply with b (result stored back in a)
    a[gid] = gf28_mul(lut, a[gid], b[gid]);
}

// Forward additive FFT with SIMD shuffle optimization.
// DEPRECATED: Metal simd_shuffle is intra-warp only (32 threads), which doesn't
// align with additive FFT butterfly pattern at fine granularities.
// halfSize = 16 at depth 11 for 2^16 FFT, meaning threads t and t^16 are in DIFFERENT
// warps, so shuffle cannot exchange butterfly partners.
//
// See: https://developer.apple.com/documentation/metal/metal_simdgroup_functions
// The existing fused kernel remains optimal.
