// Blake3 hash function for Metal GPU
// Batch hashing: one thread per input
// Supports 32-byte and 64-byte input modes

#include <metal_stdlib>
using namespace metal;

// Blake3 IV (same as Blake2s: first 8 primes' fractional parts as uint32)
constant uint BLAKE3_IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};

// Message word permutation per round
constant uchar BLAKE3_MSG_PERM[16] = {
    2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8
};

// Blake3 flags
constant uint BLAKE3_CHUNK_START = 1;
constant uint BLAKE3_CHUNK_END = 2;
constant uint BLAKE3_ROOT = 8;

// Quarter-round: G mixing function
inline void blake3_g(thread uint state[16], int a, int b, int c, int d, uint mx, uint my) {
    state[a] = state[a] + state[b] + mx;
    state[d] = (state[d] ^ state[a]);
    state[d] = (state[d] >> 16) | (state[d] << 16);  // rotr 16
    state[c] = state[c] + state[d];
    state[b] = (state[b] ^ state[c]);
    state[b] = (state[b] >> 12) | (state[b] << 20);  // rotr 12
    state[a] = state[a] + state[b] + my;
    state[d] = (state[d] ^ state[a]);
    state[d] = (state[d] >> 8) | (state[d] << 24);   // rotr 8
    state[c] = state[c] + state[d];
    state[b] = (state[b] ^ state[c]);
    state[b] = (state[b] >> 7) | (state[b] << 25);   // rotr 7
}

// Blake3 round function
inline void blake3_round(thread uint state[16], thread uint msg[16]) {
    // Column step
    blake3_g(state, 0, 4,  8, 12, msg[0],  msg[1]);
    blake3_g(state, 1, 5,  9, 13, msg[2],  msg[3]);
    blake3_g(state, 2, 6, 10, 14, msg[4],  msg[5]);
    blake3_g(state, 3, 7, 11, 15, msg[6],  msg[7]);
    // Diagonal step
    blake3_g(state, 0, 5, 10, 15, msg[8],  msg[9]);
    blake3_g(state, 1, 6, 11, 12, msg[10], msg[11]);
    blake3_g(state, 2, 7,  8, 13, msg[12], msg[13]);
    blake3_g(state, 3, 4,  9, 14, msg[14], msg[15]);
}

// Permute message words for next round
inline void blake3_permute(thread uint msg[16]) {
    uint tmp[16];
    for (int i = 0; i < 16; i++) {
        tmp[i] = msg[BLAKE3_MSG_PERM[i]];
    }
    for (int i = 0; i < 16; i++) {
        msg[i] = tmp[i];
    }
}

// Blake3 compression function
// Compresses one 64-byte block with given chaining value, counter, block_len, flags
inline void blake3_compress(
    thread const uint cv[8],
    thread uint msg[16],
    uint counter_lo, uint counter_hi,
    uint block_len, uint flags,
    thread uint out[16]
) {
    // Initialize state
    out[0] = cv[0]; out[1] = cv[1]; out[2] = cv[2]; out[3] = cv[3];
    out[4] = cv[4]; out[5] = cv[5]; out[6] = cv[6]; out[7] = cv[7];
    out[8]  = BLAKE3_IV[0]; out[9]  = BLAKE3_IV[1];
    out[10] = BLAKE3_IV[2]; out[11] = BLAKE3_IV[3];
    out[12] = counter_lo; out[13] = counter_hi;
    out[14] = block_len; out[15] = flags;

    // 7 rounds
    blake3_round(out, msg);
    blake3_permute(msg);
    blake3_round(out, msg);
    blake3_permute(msg);
    blake3_round(out, msg);
    blake3_permute(msg);
    blake3_round(out, msg);
    blake3_permute(msg);
    blake3_round(out, msg);
    blake3_permute(msg);
    blake3_round(out, msg);
    blake3_permute(msg);
    blake3_round(out, msg);

    // Finalize: XOR first 8 with last 8
    for (int i = 0; i < 8; i++) {
        out[i] ^= out[i + 8];
        out[i + 8] ^= cv[i];
    }
}

// Hash 64 bytes → 32 bytes (single chunk, single block)
// Input: n * 64 bytes, Output: n * 32 bytes
kernel void blake3_hash_64(
    device const uchar* input  [[buffer(0)]],
    device uchar* output       [[buffer(1)]],
    constant uint& count       [[buffer(2)]],
    uint gid                   [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    device const uint* in_words = (device const uint*)(input + gid * 64);

    uint msg[16];
    for (int i = 0; i < 16; i++) {
        msg[i] = in_words[i];
    }

    // Copy IV to thread storage (cannot pass constant to thread param)
    uint iv[8];
    for (int i = 0; i < 8; i++) iv[i] = BLAKE3_IV[i];

    // Single chunk, single block: flags = CHUNK_START | CHUNK_END | ROOT
    uint flags = BLAKE3_CHUNK_START | BLAKE3_CHUNK_END | BLAKE3_ROOT;
    uint state[16];
    blake3_compress(iv, msg, 0, 0, 64, flags, state);

    // Write first 8 words (32 bytes) as output
    device uint* out_words = (device uint*)(output + gid * 32);
    for (int i = 0; i < 8; i++) {
        out_words[i] = state[i];
    }
}

// Hash 32 bytes → 32 bytes (for Merkle tree internal nodes)
// Input: n * 32 bytes (pairs of child hashes), Output: n * 32 bytes
// This hashes pairs: hash(left || right) where left and right are each 32 bytes
// But the parent node in Blake3 uses PARENT flag, not chunk flags
kernel void blake3_hash_32(
    device const uchar* input  [[buffer(0)]],
    device uchar* output       [[buffer(1)]],
    constant uint& count       [[buffer(2)]],
    uint gid                   [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    // Each parent node hashes 64 bytes (left_child || right_child)
    device const uint* in_words = (device const uint*)(input + gid * 64);

    uint msg[16];
    for (int i = 0; i < 16; i++) {
        msg[i] = in_words[i];
    }

    // Copy IV to thread storage
    uint iv[8];
    for (int i = 0; i < 8; i++) iv[i] = BLAKE3_IV[i];

    // Parent node flag = 4
    uint flags = 4;  // BLAKE3_PARENT
    uint state[16];
    blake3_compress(iv, msg, 0, 0, 64, flags, state);

    device uint* out_words = (device uint*)(output + gid * 32);
    for (int i = 0; i < 8; i++) {
        out_words[i] = state[i];
    }
}

// Fused Merkle tree: one threadgroup per subtree of 1024 leaves
// Shared memory: 1024 nodes * 8 uints = 32KB
// 512 threads per threadgroup, each loads 2 leaves, then cooperative reduction
#define BLAKE3_SUBTREE_SIZE 1024

kernel void blake3_merkle_fused(
    device const uchar* leaves  [[buffer(0)]],
    device uchar* roots         [[buffer(1)]],
    constant uint& num_levels   [[buffer(2)]],
    uint tid                    [[thread_index_in_threadgroup]],
    uint tgid                   [[threadgroup_position_in_grid]],
    uint tg_size                [[threads_per_threadgroup]]
) {
    // Each node is 8 uints = 32 bytes
    threadgroup uint shared_data[BLAKE3_SUBTREE_SIZE * 8];

    // Load leaves: each thread loads 2 leaves (tid and tid + tg_size)
    uint leaf_base = tgid * BLAKE3_SUBTREE_SIZE;
    device const uint* leaves32 = (device const uint*)leaves;

    uint src_lo = (leaf_base + tid) * 8;
    uint src_hi = (leaf_base + tid + tg_size) * 8;
    uint dst_lo = tid * 8;
    uint dst_hi = (tid + tg_size) * 8;
    for (uint i = 0; i < 8; i++) shared_data[dst_lo + i] = leaves32[src_lo + i];
    for (uint i = 0; i < 8; i++) shared_data[dst_hi + i] = leaves32[src_hi + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Copy IV to thread registers
    uint iv[8];
    for (uint i = 0; i < 8; i++) iv[i] = BLAKE3_IV[i];

    uint active = BLAKE3_SUBTREE_SIZE;
    for (uint level = 0; level < num_levels; level++) {
        uint pairs = active >> 1;
        if (tid < pairs) {
            // Read left and right children into msg
            uint msg[16];
            uint left_base = tid * 2 * 8;
            uint right_base = (tid * 2 + 1) * 8;
            for (uint i = 0; i < 8; i++) msg[i] = shared_data[left_base + i];
            for (uint i = 0; i < 8; i++) msg[i + 8] = shared_data[right_base + i];

            uint state[16];
            blake3_compress(iv, msg, 0, 0, 64, 4, state); // PARENT flag

            // Write result to left child position (compact in-place)
            uint out_base = tid * 8;
            for (uint i = 0; i < 8; i++) shared_data[out_base + i] = state[i];
        }
        active = pairs;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 writes the subtree root
    if (tid == 0) {
        device uint* out32 = (device uint*)(roots + tgid * 32);
        for (uint i = 0; i < 8; i++) out32[i] = shared_data[i];
    }
}
