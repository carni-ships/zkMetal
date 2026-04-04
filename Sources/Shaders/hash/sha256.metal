// SHA-256 GPU kernel — batch hashing on Metal
// Each thread computes one SHA-256 hash of a fixed-size input.
// SHA-256 is 32-bit ARX (add-rotate-xor) — native on Apple GPU's 32-bit ALU.

#include <metal_stdlib>
using namespace metal;

// SHA-256 initial hash values (first 32 bits of fractional parts of sqrt(2..19))
constant uint SHA256_H[8] = {
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
};

// SHA-256 round constants (first 32 bits of fractional parts of cube roots of first 64 primes)
constant uint SHA256_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

// --- SHA-256 helper functions ---

inline uint rotr(uint x, uint n) {
    return (x >> n) | (x << (32u - n));
}

inline uint Ch(uint x, uint y, uint z) {
    return (x & y) ^ (~x & z);
}

inline uint Maj(uint x, uint y, uint z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

inline uint Sigma0(uint x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

inline uint Sigma1(uint x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

inline uint sigma0(uint x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

inline uint sigma1(uint x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

// Load a big-endian uint32 from a byte pointer
inline uint load_be32(device const uchar* p) {
    return (uint(p[0]) << 24) | (uint(p[1]) << 16) | (uint(p[2]) << 8) | uint(p[3]);
}

// Store a big-endian uint32 to a byte pointer
inline void store_be32(device uchar* p, uint v) {
    p[0] = uchar(v >> 24);
    p[1] = uchar(v >> 16);
    p[2] = uchar(v >> 8);
    p[3] = uchar(v);
}

// Load big-endian uint32 from threadgroup memory
inline uint load_be32_tg(threadgroup const uchar* p) {
    return (uint(p[0]) << 24) | (uint(p[1]) << 16) | (uint(p[2]) << 8) | uint(p[3]);
}

// Store big-endian uint32 to threadgroup memory
inline void store_be32_tg(threadgroup uchar* p, uint v) {
    p[0] = uchar(v >> 24);
    p[1] = uchar(v >> 16);
    p[2] = uchar(v >> 8);
    p[3] = uchar(v);
}

// SHA-256 compression function: process one 64-byte block
// state: 8 uint32 working variables (modified in place)
// W: 16 uint32 message schedule words (big-endian from input)
void sha256_compress(thread uint state[8], thread uint W[16]) {
    uint a = state[0], b = state[1], c = state[2], d = state[3];
    uint e = state[4], f = state[5], g = state[6], h = state[7];

    // 64 rounds with on-the-fly message schedule expansion
    #pragma unroll
    for (uint i = 0; i < 64; i++) {
        uint wi;
        if (i < 16) {
            wi = W[i];
        } else {
            // Message schedule: W[i] = sigma1(W[i-2]) + W[i-7] + sigma0(W[i-15]) + W[i-16]
            wi = sigma1(W[(i - 2) & 15]) + W[(i - 7) & 15] + sigma0(W[(i - 15) & 15]) + W[(i - 16) & 15];
            W[i & 15] = wi;
        }

        uint T1 = h + Sigma1(e) + Ch(e, f, g) + SHA256_K[i] + wi;
        uint T2 = Sigma0(a) + Maj(a, b, c);
        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

// SHA-256 hash of a 64-byte input (single block with padding)
// Input: 64 bytes, Output: 32 bytes (big-endian digest)
// Padding: 64 bytes data | 0x80 | zeros | 64-bit big-endian length (512 bits = 0x200)
// This requires TWO compression rounds (block 0: data, block 1: padding).
kernel void sha256_hash_batch(
    device const uchar* input      [[buffer(0)]],
    device uchar* output           [[buffer(1)]],
    constant uint& count           [[buffer(2)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    device const uchar* in_ptr = input + gid * 64;

    // Initialize hash state
    uint state[8];
    for (uint i = 0; i < 8; i++) state[i] = SHA256_H[i];

    // Block 0: the 64 bytes of input data
    uint W[16];
    for (uint i = 0; i < 16; i++) {
        W[i] = load_be32(in_ptr + i * 4);
    }
    sha256_compress(state, W);

    // Block 1: padding block
    // byte 0 = 0x80, bytes 1..55 = 0, bytes 56..63 = big-endian length (512 bits)
    W[0] = 0x80000000u;
    for (uint i = 1; i < 14; i++) W[i] = 0;
    W[14] = 0;            // high 32 bits of bit length
    W[15] = 64 * 8;       // low 32 bits of bit length = 512
    sha256_compress(state, W);

    // Write output (big-endian)
    device uchar* out_ptr = output + gid * 32;
    for (uint i = 0; i < 8; i++) {
        store_be32(out_ptr + i * 4, state[i]);
    }
}

// SHA-256 hash of pairs of 32-byte values (for Merkle trees)
// Input: N pairs of 32 bytes (64 bytes each) -> N 32-byte digests
// Same as sha256_hash_batch but semantically for Merkle pair hashing
kernel void sha256_hash_pairs(
    device const uchar* input      [[buffer(0)]],
    device uchar* output           [[buffer(1)]],
    constant uint& count           [[buffer(2)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    device const uchar* in_ptr = input + gid * 64;

    uint state[8];
    for (uint i = 0; i < 8; i++) state[i] = SHA256_H[i];

    // Block 0: the 64 bytes of concatenated pair
    uint W[16];
    for (uint i = 0; i < 16; i++) {
        W[i] = load_be32(in_ptr + i * 4);
    }
    sha256_compress(state, W);

    // Block 1: padding (message is exactly 64 bytes = 512 bits)
    W[0] = 0x80000000u;
    for (uint i = 1; i < 14; i++) W[i] = 0;
    W[14] = 0;
    W[15] = 512;  // 64 * 8 bits
    sha256_compress(state, W);

    device uchar* out_ptr = output + gid * 32;
    for (uint i = 0; i < 8; i++) {
        store_be32(out_ptr + i * 4, state[i]);
    }
}

// SHA-256 hash pair in threadgroup memory for fused Merkle kernel
// left/right: 32 bytes each (as 8 uint32 big-endian)
// out: 8 uint32 big-endian result
void sha256_hash_pair_tg(threadgroup uint* left, threadgroup uint* right,
                          threadgroup uint* out) {
    uint state[8];
    for (uint i = 0; i < 8; i++) state[i] = SHA256_H[i];

    // Block 0: left[0..7] || right[0..7] = 16 words = 64 bytes
    uint W[16];
    for (uint i = 0; i < 8; i++) W[i] = left[i];
    for (uint i = 0; i < 8; i++) W[8 + i] = right[i];
    sha256_compress(state, W);

    // Block 1: padding
    W[0] = 0x80000000u;
    for (uint i = 1; i < 14; i++) W[i] = 0;
    W[14] = 0;
    W[15] = 512;
    sha256_compress(state, W);

    for (uint i = 0; i < 8; i++) out[i] = state[i];
}

// Fused multi-level SHA-256 Merkle tree: each threadgroup processes a subtree
// Shared memory: 1024 * 8 uint32 = 32KB (32 bytes per leaf as 8 big-endian uint32)
// subtree_size = 1 << num_levels (max 1024)
kernel void sha256_merkle_fused(
    device const uchar* leaves    [[buffer(0)]],
    device uchar* roots           [[buffer(1)]],
    constant uint& num_levels     [[buffer(2)]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tgid                     [[threadgroup_position_in_grid]],
    uint tg_size                  [[threads_per_threadgroup]]
) {
    threadgroup uint shared_data[1024 * 8];  // 32KB: 1024 leaves * 8 words each

    uint subtree_size = 1u << num_levels;
    uint leaf_base = tgid * subtree_size;

    // Load leaves: convert from byte stream (big-endian uint32 words)
    for (uint i = tid; i < subtree_size; i += tg_size) {
        device const uchar* leaf = leaves + (leaf_base + i) * 32;
        for (uint j = 0; j < 8; j++) {
            shared_data[i * 8 + j] = load_be32(leaf + j * 4);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint active = subtree_size;
    for (uint level = 0; level < num_levels; level++) {
        uint pairs = active >> 1;
        if (tid < pairs) {
            sha256_hash_pair_tg(&shared_data[tid * 2 * 8],
                                &shared_data[(tid * 2 + 1) * 8],
                                &shared_data[tid * 8]);
        }
        active = pairs;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write root (one per threadgroup)
    if (tid == 0) {
        device uchar* out_ptr = roots + tgid * 32;
        for (uint i = 0; i < 8; i++) {
            store_be32(out_ptr + i * 4, shared_data[i]);
        }
    }
}
