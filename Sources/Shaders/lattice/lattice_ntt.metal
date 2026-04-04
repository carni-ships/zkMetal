// Lattice NTT kernels for Kyber (q=3329, 16-bit) and Dilithium (q=8380417, 32-bit)
// Each polynomial is 256 elements — fits entirely in threadgroup memory.
// One threadgroup processes one polynomial; batch many polynomials for GPU saturation.
//
// Key advantage on Metal's 32-bit ALU:
// - Kyber: 16-bit multiply fits in a single 32-bit ALU op
// - Dilithium: 32-bit multiply → 64-bit result, Barrett reduce back to 32-bit
// No multi-limb arithmetic needed (unlike BN254's 8x32-bit limbs).

#include <metal_stdlib>
using namespace metal;

// ============================================================
// Kyber field arithmetic (q = 3329)
// ============================================================

constant ushort KYBER_Q = 3329;
// Barrett: floor(2^24 / 3329) = 5039
constant uint KYBER_BARRETT_M = 5039;
constant uint KYBER_BARRETT_SHIFT = 24;

inline ushort kyber_reduce(uint a) {
    // Direct modulo — safe for all inputs, Metal compiles this efficiently for constant divisor
    return ushort(a % uint(KYBER_Q));
}

inline ushort kyber_add(ushort a, ushort b) {
    uint s = uint(a) + uint(b);
    return s >= uint(KYBER_Q) ? ushort(s - uint(KYBER_Q)) : ushort(s);
}

inline ushort kyber_sub(ushort a, ushort b) {
    return a >= b ? (a - b) : (a + KYBER_Q - b);
}

inline ushort kyber_mul(ushort a, ushort b) {
    return kyber_reduce(uint(a) * uint(b));
}

// ============================================================
// Dilithium field arithmetic (q = 8380417)
// ============================================================

constant uint DIL_Q = 8380417;
// Barrett: floor(2^48 / q) = 33579385
constant ulong DIL_BARRETT_M = 33579385UL;
constant ulong DIL_BARRETT_SHIFT = 48;

inline uint dil_reduce(ulong a) {
    // Direct modulo — safe for all inputs
    return uint(a % ulong(DIL_Q));
}

inline uint dil_add(uint a, uint b) {
    ulong s = ulong(a) + ulong(b);
    return s >= ulong(DIL_Q) ? uint(s - ulong(DIL_Q)) : uint(s);
}

inline uint dil_sub(uint a, uint b) {
    return a >= b ? (a - b) : (a + DIL_Q - b);
}

inline uint dil_mul(uint a, uint b) {
    return dil_reduce(ulong(a) * ulong(b));
}

// ============================================================
// Kyber NTT batch kernel
// ============================================================
// Each threadgroup processes one polynomial (256 coefficients).
// With 32 threads per threadgroup, each thread handles 8 coefficients.
// 256 * 2 bytes = 512 bytes threadgroup memory (trivially fits in 32KB).

kernel void kyber_ntt_batch(
    device ushort* polys [[buffer(0)]],
    constant ushort* twiddles [[buffer(1)]],  // 128 precomputed twiddle factors
    constant uint& num_polys [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tgid >= num_polys) return;

    threadgroup ushort shared_poly[256];

    // Load polynomial into shared memory
    uint base = tgid * 256;
    for (uint i = lid; i < 256; i += tg_size) {
        shared_poly[i] = polys[base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // NTT: Cooley-Tukey butterfly, layers from len=128 down to len=2
    uint k = 1;
    for (uint len = 128; len >= 2; len >>= 1) {
        // Each butterfly pair: (start..start+len-1) and (start+len..start+2*len-1)
        uint num_blocks = 256 / (2 * len);
        for (uint block = lid; block < num_blocks * len; block += tg_size) {
            uint block_idx = block / len;
            uint j = block % len;
            uint start = block_idx * 2 * len;
            ushort tw = twiddles[k + block_idx];
            uint i0 = start + j;
            uint i1 = i0 + len;
            ushort t = kyber_mul(tw, shared_poly[i1]);
            ushort u = shared_poly[i0];
            shared_poly[i0] = kyber_add(u, t);
            shared_poly[i1] = kyber_sub(u, t);
        }
        k += num_blocks;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store result
    for (uint i = lid; i < 256; i += tg_size) {
        polys[base + i] = shared_poly[i];
    }
}

// Kyber inverse NTT batch kernel
kernel void kyber_intt_batch(
    device ushort* polys [[buffer(0)]],
    constant ushort* inv_twiddles [[buffer(1)]],  // 128 inverse twiddle factors
    constant uint& num_polys [[buffer(2)]],
    constant ushort& inv_n [[buffer(3)]],  // 128^{-1} mod q
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
    // CPU processes blocks 0,1,...,num_blocks-1 with twiddles[k], twiddles[k-1], ..., twiddles[k-num_blocks+1]
    uint k = 127;
    for (uint len = 2; len <= 128; len <<= 1) {
        uint num_blocks = 256 / (2 * len);
        for (uint block = lid; block < num_blocks * len; block += tg_size) {
            uint block_idx = block / len;
            uint j = block % len;
            uint start = block_idx * 2 * len;
            ushort tw = inv_twiddles[k - block_idx];
            uint i0 = start + j;
            uint i1 = i0 + len;
            ushort t = shared_poly[i0];
            shared_poly[i0] = kyber_add(t, shared_poly[i1]);
            shared_poly[i1] = kyber_mul(tw, kyber_sub(shared_poly[i1], t));
        }
        k -= num_blocks;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Scale by 1/128
    for (uint i = lid; i < 256; i += tg_size) {
        shared_poly[i] = kyber_mul(shared_poly[i], inv_n);
    }

    // Store result
    for (uint i = lid; i < 256; i += tg_size) {
        polys[base + i] = shared_poly[i];
    }
}

// ============================================================
// Dilithium NTT batch kernel
// ============================================================
// Each polynomial: 256 * 4 bytes = 1KB threadgroup memory.

kernel void dilithium_ntt_batch(
    device uint* polys [[buffer(0)]],
    constant uint* twiddles [[buffer(1)]],
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
            uint t = dil_mul(tw, shared_poly[i1]);
            uint u = shared_poly[i0];
            shared_poly[i0] = dil_add(u, t);
            shared_poly[i1] = dil_sub(u, t);
        }
        k += num_blocks;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = lid; i < 256; i += tg_size) {
        polys[base + i] = shared_poly[i];
    }
}

// Dilithium inverse NTT batch kernel
kernel void dilithium_intt_batch(
    device uint* polys [[buffer(0)]],
    constant uint* inv_twiddles [[buffer(1)]],
    constant uint& num_polys [[buffer(2)]],
    constant uint& inv_n [[buffer(3)]],  // 128^{-1} mod q
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
            uint tw = inv_twiddles[k - block_idx];
            uint i0 = start + j;
            uint i1 = i0 + len;
            uint t = shared_poly[i0];
            shared_poly[i0] = dil_add(t, shared_poly[i1]);
            shared_poly[i1] = dil_mul(tw, dil_sub(shared_poly[i1], t));
        }
        k -= num_blocks;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = lid; i < 256; i += tg_size) {
        shared_poly[i] = dil_mul(shared_poly[i], inv_n);
    }

    for (uint i = lid; i < 256; i += tg_size) {
        polys[base + i] = shared_poly[i];
    }
}
