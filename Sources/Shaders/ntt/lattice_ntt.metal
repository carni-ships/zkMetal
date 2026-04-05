// Lattice NTT kernels for post-quantum cryptography
// Kyber (q=3329, 16-bit elements) and Dilithium (q=8380417, 23-bit modulus)
// Each polynomial is 256 elements -- fits entirely in threadgroup memory.
// One threadgroup per polynomial; batch thousands for GPU saturation.
//
// Kernel naming: lattice_ntt_kyber, lattice_intt_kyber, lattice_ntt_dilithium, etc.
// These are the "ntt/" directory variants with unified naming convention.

#include <metal_stdlib>
using namespace metal;

// ============================================================
// Kyber field arithmetic (q = 3329)
// ============================================================

constant ushort LNTT_KYBER_Q = 3329;

// Barrett reduction: floor(2^24 / 3329) = 5039
// For a < q^2 = 11082241 < 2^24, Barrett gives exact result.
constant uint LNTT_KYBER_BARRETT_M = 5039;
constant uint LNTT_KYBER_BARRETT_SHIFT = 24;

inline ushort lntt_kyber_reduce(uint a) {
    // Barrett reduction for small modulus
    uint t = (a * LNTT_KYBER_BARRETT_M) >> LNTT_KYBER_BARRETT_SHIFT;
    uint r = a - t * uint(LNTT_KYBER_Q);
    return ushort(r >= uint(LNTT_KYBER_Q) ? r - uint(LNTT_KYBER_Q) : r);
}

inline ushort lntt_kyber_add(ushort a, ushort b) {
    uint s = uint(a) + uint(b);
    return s >= uint(LNTT_KYBER_Q) ? ushort(s - uint(LNTT_KYBER_Q)) : ushort(s);
}

inline ushort lntt_kyber_sub(ushort a, ushort b) {
    return a >= b ? ushort(a - b) : ushort(uint(a) + uint(LNTT_KYBER_Q) - uint(b));
}

inline ushort lntt_kyber_mul(ushort a, ushort b) {
    return lntt_kyber_reduce(uint(a) * uint(b));
}

// ============================================================
// Dilithium field arithmetic (q = 8380417)
// Montgomery multiplication for efficiency on 23-bit modulus.
// R = 2^32, q' = q^{-1} mod R (Montgomery constant)
// ============================================================

constant uint LNTT_DIL_Q = 8380417;
// Montgomery: R = 2^32
// q_inv_neg = -q^{-1} mod 2^32 = 4236238847
// q * q_inv_neg mod 2^32 = 2^32 - 1 (i.e., -1 mod 2^32)
constant uint LNTT_DIL_QINV_NEG = 4236238847u;
// R^2 mod q = (2^32)^2 mod q = 2365951 (for converting to Montgomery form)
constant uint LNTT_DIL_R2 = 2365951u;
// R mod q = 4193792 (for converting single values)

inline uint lntt_dil_mont_reduce(ulong a) {
    // Montgomery reduction: given a < q*R, compute a*R^{-1} mod q
    uint lo = uint(a);                          // a mod R
    uint t = lo * LNTT_DIL_QINV_NEG;           // t = a * (-q^{-1}) mod R
    ulong u = a + ulong(t) * ulong(LNTT_DIL_Q); // u = a + t*q, divisible by R
    uint result = uint(u >> 32);                 // u / R
    return result >= LNTT_DIL_Q ? result - LNTT_DIL_Q : result;
}

inline uint lntt_dil_mont_mul(uint a, uint b) {
    // Both a,b in Montgomery form: a*R, b*R
    // Product: a*R * b*R = a*b*R^2
    // Montgomery reduce: a*b*R^2 * R^{-1} = a*b*R (still in Montgomery form)
    return lntt_dil_mont_reduce(ulong(a) * ulong(b));
}

inline uint lntt_dil_add(uint a, uint b) {
    uint s = a + b;
    return s >= LNTT_DIL_Q ? s - LNTT_DIL_Q : s;
}

inline uint lntt_dil_sub(uint a, uint b) {
    return a >= b ? (a - b) : (a + LNTT_DIL_Q - b);
}

// Standard (non-Montgomery) multiply for twiddle application
inline uint lntt_dil_mul(uint a, uint b) {
    return uint(ulong(a) * ulong(b) % ulong(LNTT_DIL_Q));
}

// ============================================================
// Kyber forward NTT (Cooley-Tukey, n=256)
// ============================================================
// Each threadgroup processes one polynomial.
// 32 threads per threadgroup, each handles 8 coefficients.
// 256 * 2 bytes = 512 bytes threadgroup memory.

kernel void lattice_ntt_kyber(
    device ushort* polys [[buffer(0)]],
    constant ushort* twiddles [[buffer(1)]],    // 128 precomputed twiddle factors (bit-reversed)
    constant uint& num_polys [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tgid >= num_polys) return;

    threadgroup ushort shared_poly[256];

    // Coalesced load into shared memory
    uint base = tgid * 256;
    for (uint i = lid; i < 256; i += tg_size) {
        shared_poly[i] = polys[base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // NTT: Cooley-Tukey butterfly, layers from len=128 down to len=2
    // Twiddle indexing: k starts at 1, increments by num_blocks per layer
    uint k = 1;
    for (uint len = 128; len >= 2; len >>= 1) {
        uint num_blocks = 256 / (2 * len);
        for (uint block = lid; block < num_blocks * len; block += tg_size) {
            uint block_idx = block / len;
            uint j = block % len;
            uint start = block_idx * 2 * len;
            ushort tw = twiddles[k + block_idx];
            uint i0 = start + j;
            uint i1 = i0 + len;
            ushort t = lntt_kyber_mul(tw, shared_poly[i1]);
            ushort u = shared_poly[i0];
            shared_poly[i0] = lntt_kyber_add(u, t);
            shared_poly[i1] = lntt_kyber_sub(u, t);
        }
        k += num_blocks;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Coalesced store
    for (uint i = lid; i < 256; i += tg_size) {
        polys[base + i] = shared_poly[i];
    }
}

// ============================================================
// Kyber inverse NTT (Gentleman-Sande, n=256)
// ============================================================

kernel void lattice_intt_kyber(
    device ushort* polys [[buffer(0)]],
    constant ushort* twiddles [[buffer(1)]],    // forward twiddles (INTT uses them in reverse order)
    constant uint& num_polys [[buffer(2)]],
    constant ushort& inv_n [[buffer(3)]],       // 128^{-1} mod q (scaling factor)
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tgid >= num_polys) return;

    threadgroup ushort shared_poly[256];

    uint base = tgid * 256;
    for (uint i = lid; i < 256; i += tg_size) {
        shared_poly[i] = polys[base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Inverse NTT: Gentleman-Sande, layers from len=2 up to len=128
    uint k = 127;
    for (uint len = 2; len <= 128; len <<= 1) {
        uint num_blocks = 256 / (2 * len);
        for (uint block = lid; block < num_blocks * len; block += tg_size) {
            uint block_idx = block / len;
            uint j = block % len;
            uint start = block_idx * 2 * len;
            ushort tw = twiddles[k - block_idx];
            uint i0 = start + j;
            uint i1 = i0 + len;
            ushort t = shared_poly[i0];
            shared_poly[i0] = lntt_kyber_add(t, shared_poly[i1]);
            shared_poly[i1] = lntt_kyber_mul(tw, lntt_kyber_sub(shared_poly[i1], t));
        }
        k -= num_blocks;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Scale by 1/128 and store
    for (uint i = lid; i < 256; i += tg_size) {
        polys[base + i] = lntt_kyber_mul(shared_poly[i], inv_n);
    }
}

// ============================================================
// Dilithium forward NTT (Cooley-Tukey, n=256)
// ============================================================
// 256 * 4 bytes = 1KB threadgroup memory per polynomial.

kernel void lattice_ntt_dilithium(
    device uint* polys [[buffer(0)]],
    constant uint* twiddles [[buffer(1)]],      // 128 precomputed twiddle factors
    constant uint& num_polys [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tgid >= num_polys) return;

    threadgroup uint shared_poly[256];

    uint base = tgid * 256;
    for (uint i = lid; i < 256; i += tg_size) {
        shared_poly[i] = polys[base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint k = 1;
    for (uint len = 128; len >= 2; len >>= 1) {
        uint num_blocks = 256 / (2 * len);
        for (uint block = lid; block < num_blocks * len; block += tg_size) {
            uint block_idx = block / len;
            uint j = block % len;
            uint start = block_idx * 2 * len;
            uint tw = twiddles[k + block_idx];
            uint i0 = start + j;
            uint i1 = i0 + len;
            uint t = lntt_dil_mul(tw, shared_poly[i1]);
            uint u = shared_poly[i0];
            shared_poly[i0] = lntt_dil_add(u, t);
            shared_poly[i1] = lntt_dil_sub(u, t);
        }
        k += num_blocks;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = lid; i < 256; i += tg_size) {
        polys[base + i] = shared_poly[i];
    }
}

// ============================================================
// Dilithium inverse NTT (Gentleman-Sande, n=256)
// ============================================================

kernel void lattice_intt_dilithium(
    device uint* polys [[buffer(0)]],
    constant uint* twiddles [[buffer(1)]],      // forward twiddles (used in reverse)
    constant uint& num_polys [[buffer(2)]],
    constant uint& inv_n [[buffer(3)]],         // 128^{-1} mod q
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tgid >= num_polys) return;

    threadgroup uint shared_poly[256];

    uint base = tgid * 256;
    for (uint i = lid; i < 256; i += tg_size) {
        shared_poly[i] = polys[base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint k = 127;
    for (uint len = 2; len <= 128; len <<= 1) {
        uint num_blocks = 256 / (2 * len);
        for (uint block = lid; block < num_blocks * len; block += tg_size) {
            uint block_idx = block / len;
            uint j = block % len;
            uint start = block_idx * 2 * len;
            uint tw = twiddles[k - block_idx];
            uint i0 = start + j;
            uint i1 = i0 + len;
            uint t = shared_poly[i0];
            shared_poly[i0] = lntt_dil_add(t, shared_poly[i1]);
            shared_poly[i1] = lntt_dil_mul(tw, lntt_dil_sub(shared_poly[i1], t));
        }
        k -= num_blocks;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = lid; i < 256; i += tg_size) {
        polys[base + i] = lntt_dil_mul(shared_poly[i], inv_n);
    }
}

// ============================================================
// Kyber pointwise multiplication in NTT domain
// ============================================================
// Element-wise: out[i] = a[i] * b[i] mod q
// Operates on flat arrays (num_polys * 256 elements).

kernel void lattice_pointwise_kyber(
    device const ushort* a [[buffer(0)]],
    device const ushort* b [[buffer(1)]],
    device ushort* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],         // total elements
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    out[gid] = lntt_kyber_mul(a[gid], b[gid]);
}

// ============================================================
// Dilithium pointwise multiplication in NTT domain
// ============================================================

kernel void lattice_pointwise_dilithium(
    device const uint* a [[buffer(0)]],
    device const uint* b [[buffer(1)]],
    device uint* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    out[gid] = lntt_dil_mul(a[gid], b[gid]);
}
