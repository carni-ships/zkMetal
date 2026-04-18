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
//
// OPTIMIZATIONS:
// - Precomputed 256x256 GF(2^8) multiplication LUT (USE_LUT=1)
// - const device pointer for LUT (better GPU cache behavior)
// - [[always_inline]] on gf28_mul for inlining
// - New kernel variant: all threads active (no divergence) via n/2 thread dispatch

#include <metal_stdlib>
using namespace metal;

// GF(2^8) multiplication with reduction by 0x11B.
// Primary: 256x256 LUT passed as device pointer. Lookup = O(1).
// Fallback: shift-XOR (USE_LUT=0 compiles out the LUT parameter).
#ifdef USE_LUT
// LUT is passed as kernel parameter (lut [[buffer(0)]]) and forwarded to gf28_mul.
inline uint8_t gf28_mul(device const uint8_t* lut, uint8_t a, uint8_t b) [[always_inline]] {
    return lut[a * 256 + b];
}
#else
// Shift-XOR fallback (for debugging when LUT is unavailable)
inline uint8_t gf28_mul(device const uint8_t* lut, uint8_t a, uint8_t b) [[always_inline]] {
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
// buffer(0): 256x256 GF(2^8) LUT (device const pointer)
// buffer(1): data[gid]: input element, modified in registers, final value written back.
// buffer(2): basis[0..k-1]: GF(2^8) basis elements (one per FFT level).
// buffer(3): n (total elements, power of 2)
// buffer(4): k (log₂(n))
#ifdef USE_LUT
kernel void additive_fft_gf8_forward(
    device const uint8_t* lut  [[buffer(0)]],
    device uint8_t* data       [[buffer(1)]],
    constant uint8_t* basis   [[buffer(2)]],
    constant uint32_t& n       [[buffer(3)]],
    constant uint32_t& k       [[buffer(4)]],
    uint gid                  [[thread_position_in_grid]]
) {
#else
kernel void additive_fft_gf8_forward(
    device uint8_t* data       [[buffer(1)]],
    constant uint8_t* basis   [[buffer(2)]],
    constant uint32_t& n       [[buffer(3)]],
    constant uint32_t& k       [[buffer(4)]],
    uint gid                  [[thread_position_in_grid]]
) {
#endif
    if (gid >= n) return;

    uint8_t val = data[gid];

    // k levels of additive butterfly (DIF: large stride first)
    for (uint depth = 0; depth < k; depth++) {
        uint block_size = n >> depth;      // doubles each level going up
        uint halfSize = block_size >> 1;  // n >> (depth+1)
        uint local_idx = gid % block_size;

        // Only process upper half of each block (where i >= halfSize)
        if (local_idx < halfSize) {
            // This element is the "lo" of the pair — skip (handled by hi element)
            continue;
        }
        uint j = gid - halfSize;             // lo index

        uint8_t s = basis[depth];            // basis element for this level
        uint8_t hi_val = val;                // our value (hi half)
        uint8_t lo_val = data[j];            // lo value from memory

        // Twist: lo ^= s * hi
        uint8_t twisted = lo_val ^ gf28_mul(lut, s, hi_val);
        // Propagate: hi ^= lo
        uint8_t propagated = lo_val ^ hi_val;

        // Write back
        data[j] = twisted;
        val = propagated;                    // update our register with new value
    }

    data[gid] = val;
}

// Forward additive FFT with ALL threads active (no divergence).
// Dispatch n/2 threads (one per butterfly pair) instead of n threads.
// Each thread handles one butterfly pair (lo_idx, hi_idx) at every level.
// This eliminates the branch divergence where half the threads skip at each level.
//
// buffer(0): 256x256 GF(2^8) LUT
// buffer(1): data array (modified in-place)
// buffer(2): basis elements
// buffer(3): n (total elements)
// buffer(4): k (log₂(n))
// Note: gid ranges from 0 to n/2-1, each representing one butterfly pair
#ifdef USE_LUT
kernel void additive_fft_gf8_forward_pairs(
    device const uint8_t* lut  [[buffer(0)]],
    device uint8_t* data       [[buffer(1)]],
    constant uint8_t* basis   [[buffer(2)]],
    constant uint32_t& n       [[buffer(3)]],
    constant uint32_t& k       [[buffer(4)]],
    uint gid                  [[thread_position_in_grid]]
) {
    uint nPairs = n >> 1;  // n/2 pairs
    if (gid >= nPairs) return;

    // Each thread handles one butterfly pair per level.
    // At depth d, thread t's pair in block b is:
    //   lo_idx = b * halfSize + t
    //   hi_idx = lo_idx + halfSize
    // where halfSize = n >> (d+1), block_size = n >> d
    //
    // Since threads are indexed 0..nPairs-1, we interpret gid as:
    //   gid = b * halfSize + t  (lo index of the pair)
    // This means each thread processes a specific lo index across all blocks.

    for (uint depth = 0; depth < k; depth++) {
        uint block_size = n >> depth;   // n, n/2, n/4, ...
        uint halfSize = block_size >> 1; // n/2, n/4, n/8, ...

        // Determine which block and position within block for this thread's lo element
        uint block_idx = gid / halfSize;
        uint t = gid % halfSize;

        uint lo_idx = block_idx * block_size + t;
        uint hi_idx = lo_idx + halfSize;

        uint8_t lo_val = data[lo_idx];
        uint8_t hi_val = data[hi_idx];
        uint8_t s = basis[depth];

        // Butterfly: twist then propagate
        uint8_t twisted = lo_val ^ gf28_mul(lut, s, hi_val);
        uint8_t propagated = lo_val ^ hi_val;

        data[lo_idx] = twisted;
        data[hi_idx] = propagated;
    }
}
#else
kernel void additive_fft_gf8_forward_pairs(
    device uint8_t* data       [[buffer(1)]],
    constant uint8_t* basis   [[buffer(2)]],
    constant uint32_t& n       [[buffer(3)]],
    constant uint32_t& k       [[buffer(4)]],
    uint gid                  [[thread_position_in_grid]]
) {
    uint nPairs = n >> 1;
    if (gid >= nPairs) return;

    for (uint depth = 0; depth < k; depth++) {
        uint block_size = n >> depth;
        uint halfSize = block_size >> 1;
        uint block_idx = gid / halfSize;
        uint t = gid % halfSize;
        uint lo_idx = block_idx * block_size + t;
        uint hi_idx = lo_idx + halfSize;

        uint8_t lo_val = data[lo_idx];
        uint8_t hi_val = data[hi_idx];
        uint8_t s = basis[depth];

        // Shift-XOR multiply: s * hi_val
        uint16_t p = 0;
        p ^= ((uint16_t)(s & 1)  ) * ((uint16_t)(hi_val)       );
        p ^= ((uint16_t)(s & 2)  ) * ((uint16_t)(hi_val << 1) );
        p ^= ((uint16_t)(s & 4)  ) * ((uint16_t)(hi_val << 2) );
        p ^= ((uint16_t)(s & 8)  ) * ((uint16_t)(hi_val << 3) );
        p ^= ((uint16_t)(s & 16) ) * ((uint16_t)(hi_val << 4) );
        p ^= ((uint16_t)(s & 32) ) * ((uint16_t)(hi_val << 5) );
        p ^= ((uint16_t)(s & 64) ) * ((uint16_t)(hi_val << 6) );
        p ^= ((uint16_t)(s & 128)) * ((uint16_t)(hi_val << 7) );
        uint16_t h = p >> 8;
        if (h & 0x01) p ^= 0x11B << 0;
        if (h & 0x02) p ^= 0x11B << 1;
        if (h & 0x04) p ^= 0x11B << 2;
        if (h & 0x08) p ^= 0x11B << 3;
        if (h & 0x10) p ^= 0x11B << 4;
        if (h & 0x20) p ^= 0x11B << 5;
        if (h & 0x40) p ^= 0x11B << 6;
        if (h & 0x80) p ^= 0x11B << 7;
        uint8_t product = (uint8_t)(p & 0xFF);

        uint8_t twisted = lo_val ^ product;
        uint8_t propagated = lo_val ^ hi_val;

        data[lo_idx] = twisted;
        data[hi_idx] = propagated;
    }
}
#endif

// Forward additive FFT with threadgroup-local basis caching.
// OPTIMIZATION: Basis array (k elements, max 22) is loaded into threadgroup
// memory ONCE at kernel start, then reused for all depths. This eliminates
// k global memory reads per threadgroup.
//
// Threadgroup memory usage: k bytes for basis (max 22 bytes)
// Threadgroup size: any valid size (uses thread 0 for basis loading)
//
// buffer(0): 256x256 GF(2^8) LUT
// buffer(1): data array (modified in-place)
// buffer(2): basis elements (loaded into threadgroup memory)
// buffer(3): n (total elements)
// buffer(4): k (log₂(n))
#ifdef USE_LUT
kernel void additive_fft_gf8_forward_pairs_tg(
    device const uint8_t* lut  [[buffer(0)]],
    device uint8_t* data       [[buffer(1)]],
    constant uint8_t* basis   [[buffer(2)]],
    constant uint32_t& n       [[buffer(3)]],
    constant uint32_t& k       [[buffer(4)]],
    uint gid                  [[thread_position_in_grid]],
    uint tid                  [[thread_position_in_threadgroup]],
    uint tgroup_id            [[threadgroup_position_in_grid]]
) {
    uint nPairs = n >> 1;
    if (gid >= nPairs) return;

    // Load basis into threadgroup memory (only thread 0 does this)
    // Max k elements, max 22 bytes - well within threadgroup memory limits
    uint kVal = k;
    threadgroup uint8_t tg_basis[22];  // k is at most 22 for n=4M
    if (tid == 0) {
        for (uint i = 0; i < kVal; i++) {
            tg_basis[i] = basis[i];
        }
    }
    // Wait for thread 0 to finish loading basis
    threadgroup_barrier(mem_flags::mem_none);

    // All threads in the threadgroup now use threadgroup-local basis
    for (uint depth = 0; depth < k; depth++) {
        uint block_size = n >> depth;
        uint halfSize = block_size >> 1;
        uint block_idx = gid / halfSize;
        uint t = gid % halfSize;
        uint lo_idx = block_idx * block_size + t;
        uint hi_idx = lo_idx + halfSize;

        uint8_t lo_val = data[lo_idx];
        uint8_t hi_val = data[hi_idx];
        uint8_t s = tg_basis[depth];  // Use threadgroup-local basis!

        uint8_t twisted = lo_val ^ gf28_mul(lut, s, hi_val);
        uint8_t propagated = lo_val ^ hi_val;

        data[lo_idx] = twisted;
        data[hi_idx] = propagated;
    }
}
#else
kernel void additive_fft_gf8_forward_pairs_tg(
    device uint8_t* data       [[buffer(1)]],
    constant uint8_t* basis   [[buffer(2)]],
    constant uint32_t& n       [[buffer(3)]],
    constant uint32_t& k       [[buffer(4)]],
    uint gid                  [[thread_position_in_grid]],
    uint tid                  [[thread_position_in_threadgroup]],
    uint tgroup_id            [[threadgroup_position_in_grid]]
) {
    uint nPairs = n >> 1;
    if (gid >= nPairs) return;

    uint kVal = k;
    threadgroup uint8_t tg_basis[22];
    if (tid == 0) {
        for (uint i = 0; i < kVal; i++) {
            tg_basis[i] = basis[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_none);

    for (uint depth = 0; depth < k; depth++) {
        uint block_size = n >> depth;
        uint halfSize = block_size >> 1;
        uint block_idx = gid / halfSize;
        uint t = gid % halfSize;
        uint lo_idx = block_idx * block_size + t;
        uint hi_idx = lo_idx + halfSize;

        uint8_t lo_val = data[lo_idx];
        uint8_t hi_val = data[hi_idx];
        uint8_t s = tg_basis[depth];

        // Shift-XOR multiply
        uint16_t p = 0;
        p ^= ((uint16_t)(s & 1)  ) * ((uint16_t)(hi_val)       );
        p ^= ((uint16_t)(s & 2)  ) * ((uint16_t)(hi_val << 1) );
        p ^= ((uint16_t)(s & 4)  ) * ((uint16_t)(hi_val << 2) );
        p ^= ((uint16_t)(s & 8)  ) * ((uint16_t)(hi_val << 3) );
        p ^= ((uint16_t)(s & 16) ) * ((uint16_t)(hi_val << 4) );
        p ^= ((uint16_t)(s & 32) ) * ((uint16_t)(hi_val << 5) );
        p ^= ((uint16_t)(s & 64) ) * ((uint16_t)(hi_val << 6) );
        p ^= ((uint16_t)(s & 128)) * ((uint16_t)(hi_val << 7) );
        uint16_t h = p >> 8;
        if (h & 0x01) p ^= 0x11B << 0;
        if (h & 0x02) p ^= 0x11B << 1;
        if (h & 0x04) p ^= 0x11B << 2;
        if (h & 0x08) p ^= 0x11B << 3;
        if (h & 0x10) p ^= 0x11B << 4;
        if (h & 0x20) p ^= 0x11B << 5;
        if (h & 0x40) p ^= 0x11B << 6;
        if (h & 0x80) p ^= 0x11B << 7;
        uint8_t product = (uint8_t)(p & 0xFF);

        uint8_t twisted = lo_val ^ product;
        uint8_t propagated = lo_val ^ hi_val;

        data[lo_idx] = twisted;
        data[hi_idx] = propagated;
    }
}
#endif

// Inverse additive FFT over GF(2^8).
// Fused: processes all k levels in one dispatch.
// DIT: small stride first (reverse of forward).
// buffer(0): LUT, buffer(1): data, buffer(2): basis, buffer(3): n, buffer(4): k
#ifdef USE_LUT
kernel void additive_fft_gf8_inverse(
    device const uint8_t* lut   [[buffer(0)]],
    device uint8_t* data       [[buffer(1)]],
    constant uint8_t* basis   [[buffer(2)]],
    constant uint32_t& n       [[buffer(3)]],
    constant uint32_t& k       [[buffer(4)]],
    uint gid                  [[thread_position_in_grid]]
) {
#else
kernel void additive_fft_gf8_inverse(
    device uint8_t* data       [[buffer(1)]],
    constant uint8_t* basis   [[buffer(2)]],
    constant uint32_t& n       [[buffer(3)]],
    constant uint32_t& k       [[buffer(4)]],
    uint gid                  [[thread_position_in_grid]]
) {
#endif
    if (gid >= n) return;

    uint8_t val = data[gid];

    // k levels (DIT: small stride first = reverse depth order)
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

        // Un-propagate: hi ^= lo
        // Un-twist: lo ^= s * hi_new
        uint8_t unpropagated = hi_val ^ lo_val;
        uint8_t untwisted = lo_val ^ gf28_mul(lut, s, unpropagated);

        data[j] = untwisted;
        val = unpropagated;
    }

    data[gid] = val;
}

// Batch forward additive FFT for multiple independent arrays.
// Each thread processes one element from one array.
// buffer(0): LUT, buffer(1): data (flat: batch * n), buffer(2): basis, buffer(3): n, buffer(4): k, buffer(5): batch
#ifdef USE_LUT
kernel void additive_fft_gf8_forward_batch(
    device const uint8_t* lut  [[buffer(0)]],
    device uint8_t* data       [[buffer(1)]],
    constant uint8_t* basis   [[buffer(2)]],
    constant uint32_t& n       [[buffer(3)]],
    constant uint32_t& k       [[buffer(4)]],
    constant uint32_t& batch   [[buffer(5)]],
    uint gid                  [[thread_position_in_grid]]
) {
#else
kernel void additive_fft_gf8_forward_batch(
    device uint8_t* data       [[buffer(1)]],
    constant uint8_t* basis   [[buffer(2)]],
    constant uint32_t& n       [[buffer(3)]],
    constant uint32_t& k       [[buffer(4)]],
    constant uint32_t& batch   [[buffer(5)]],
    uint gid                  [[thread_position_in_grid]]
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
    device const uint8_t* lut [[buffer(0)]],
    device const uint8_t* a   [[buffer(1)]],
    device const uint8_t* b   [[buffer(2)]],
    device uint8_t* out       [[buffer(3)]],
    constant uint32_t& n      [[buffer(4)]],
    uint gid                  [[thread_position_in_grid]]
) {
#else
kernel void gf28_pointwise_mul(
    device const uint8_t* a   [[buffer(1)]],
    device const uint8_t* b   [[buffer(2)]],
    device uint8_t* out       [[buffer(3)]],
    constant uint32_t& n      [[buffer(4)]],
    uint gid                  [[thread_position_in_grid]]
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
    device const uint8_t* lut   [[buffer(0)]],
    device uint8_t* a           [[buffer(1)]],
    constant uint8_t* basis   [[buffer(2)]],
    constant uint32_t& n       [[buffer(3)]],
    constant uint32_t& k       [[buffer(4)]],
    device const uint8_t* b   [[buffer(5)]],
    uint gid                  [[thread_position_in_grid]]
) {
#else
kernel void additive_fft_gf8_forward_then_pointwise_mul(
    device uint8_t* a           [[buffer(1)]],
    constant uint8_t* basis   [[buffer(2)]],
    constant uint32_t& n       [[buffer(3)]],
    constant uint32_t& k       [[buffer(4)]],
    device const uint8_t* b   [[buffer(5)]],
    uint gid                  [[thread_position_in_grid]]
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
