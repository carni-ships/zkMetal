// Poseidon2 GPU kernel for M31 (Mersenne31), t=16
// d=5 (x^5 S-box), rounds_f=14, rounds_p=21
// Each thread computes one independent Poseidon2 permutation.
// M31 elements are 4 bytes each — 8× more compact than BN254.

#include "../fields/mersenne31.metal"

// Width and round parameters
#define P2M31_T 16
#define P2M31_RATE 8
#define P2M31_RF_HALF 7
#define P2M31_RP 21
#define P2M31_TOTAL_ROUNDS 35

// Internal diagonal constants (from Plonky3/Stwo reference)
constant uint P2M31_INTERNAL_DIAG[16] = {
    1, 1, 2, 1, 8, 32, 2, 256, 4096, 8, 65536, 1024, 2, 16384, 512, 32768
};

// S-box: x -> x^5
M31 p2m31_sbox(M31 x) {
    M31 x2 = m31_sqr(x);
    M31 x4 = m31_sqr(x2);
    return m31_mul(x4, x);
}

// M4 circulant matrix: circ(2, 3, 1, 1) on 4 elements
// Efficient Feistel-like implementation
void p2m31_m4(thread M31 &s0, thread M31 &s1, thread M31 &s2, thread M31 &s3) {
    M31 t0 = m31_add(s0, s1);
    M31 t1 = m31_add(s2, s3);
    M31 t2 = m31_add(m31_add(s1, s1), t1);
    M31 t3 = m31_add(m31_add(s3, s3), t0);
    s0 = m31_add(t0, t3);
    s1 = m31_add(t1, t2);
    s2 = m31_add(t0, t2);
    s3 = m31_add(t1, t3);
}

// External linear layer for t=16: M4 on 4x4 blocks + cross-block mixing
void p2m31_external_layer(thread M31 *s) {
    // Apply M4 to each 4-element block
    p2m31_m4(s[0], s[1], s[2], s[3]);
    p2m31_m4(s[4], s[5], s[6], s[7]);
    p2m31_m4(s[8], s[9], s[10], s[11]);
    p2m31_m4(s[12], s[13], s[14], s[15]);

    // Cross-block mixing: each element += sum of corresponding positions
    for (uint i = 0; i < 4; i++) {
        M31 sum = m31_add(m31_add(s[i], s[i+4]), m31_add(s[i+8], s[i+12]));
        s[i]    = m31_add(s[i], sum);
        s[i+4]  = m31_add(s[i+4], sum);
        s[i+8]  = m31_add(s[i+8], sum);
        s[i+12] = m31_add(s[i+12], sum);
    }
}

// Internal linear layer: y_i = diag[i] * x_i + sum(x_j)
void p2m31_internal_layer(thread M31 *s) {
    M31 sum = m31_zero();
    for (uint i = 0; i < 16; i++) {
        sum = m31_add(sum, s[i]);
    }

    for (uint i = 0; i < 16; i++) {
        uint d = P2M31_INTERNAL_DIAG[i];
        M31 prod;
        if (d == 1) {
            prod = s[i];
        } else if (d == 2) {
            prod = m31_add(s[i], s[i]);
        } else {
            prod = m31_mul(s[i], M31{d % M31_P});
        }
        s[i] = m31_add(prod, sum);
    }
}

// Full Poseidon2 permutation on 16 M31 elements
void p2m31_permute(thread M31 *s, constant uint *rc) {
    // Initial external layer
    p2m31_external_layer(s);

    // First half of full rounds (0..6)
    for (uint r = 0; r < P2M31_RF_HALF; r++) {
        uint rc_base = r * P2M31_T;
        for (uint i = 0; i < P2M31_T; i++) s[i] = m31_add(s[i], M31{rc[rc_base + i]});
        for (uint i = 0; i < P2M31_T; i++) s[i] = p2m31_sbox(s[i]);
        p2m31_external_layer(s);
    }

    // Partial rounds (7..27)
    for (uint r = P2M31_RF_HALF; r < P2M31_RF_HALF + P2M31_RP; r++) {
        s[0] = m31_add(s[0], M31{rc[r * P2M31_T]});
        s[0] = p2m31_sbox(s[0]);
        p2m31_internal_layer(s);
    }

    // Second half of full rounds (28..34)
    for (uint r = P2M31_RF_HALF + P2M31_RP; r < P2M31_TOTAL_ROUNDS; r++) {
        uint rc_base = r * P2M31_T;
        for (uint i = 0; i < P2M31_T; i++) s[i] = m31_add(s[i], M31{rc[rc_base + i]});
        for (uint i = 0; i < P2M31_T; i++) s[i] = p2m31_sbox(s[i]);
        p2m31_external_layer(s);
    }
}

// Poseidon2 permutation kernel
// input: array of (16*n) M31 elements (n states of 16 elements each)
// rc: round constants, 35*16 = 560 uint values
// output: array of (16*n) M31 elements
kernel void poseidon2_m31_permute(
    device const uint* input        [[buffer(0)]],
    device uint* output             [[buffer(1)]],
    constant uint* rc               [[buffer(2)]],
    constant uint& count            [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    uint base = gid * P2M31_T;
    M31 s[P2M31_T];
    for (uint i = 0; i < P2M31_T; i++) {
        s[i] = M31{input[base + i]};
    }

    p2m31_permute(s, rc);

    for (uint i = 0; i < P2M31_T; i++) {
        output[base + i] = s[i].v;
    }
}

// 2-to-1 compression: hash pairs of 8-element M31 blocks
// input: array of 2*n * 8 = 16*n M31 elements
// output: array of n * 8 M31 elements
// Each thread hashes one pair: state[0..7] = left, state[8..15] = right (capacity=0)
kernel void poseidon2_m31_hash_pairs(
    device const uint* input        [[buffer(0)]],
    device uint* output             [[buffer(1)]],
    constant uint* rc               [[buffer(2)]],
    constant uint& count            [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    M31 s[P2M31_T];
    uint in_base = gid * P2M31_T;
    for (uint i = 0; i < P2M31_T; i++) {
        s[i] = M31{input[in_base + i]};
    }

    p2m31_permute(s, rc);

    // Output rate portion (first 8 elements)
    uint out_base = gid * P2M31_RATE;
    for (uint i = 0; i < P2M31_RATE; i++) {
        output[out_base + i] = s[i].v;
    }
}

// Inline hash pair for fused Merkle kernels
// left and right are 8-element M31 arrays, result written to out[0..7]
void p2m31_hash_pair(thread M31 *left, thread M31 *right, thread M31 *out, constant uint *rc) {
    M31 s[P2M31_T];
    for (uint i = 0; i < 8; i++) s[i] = left[i];
    for (uint i = 0; i < 8; i++) s[i + 8] = right[i];

    p2m31_permute(s, rc);

    for (uint i = 0; i < 8; i++) out[i] = s[i];
}

// Fused multi-level Merkle tree for M31 Poseidon2
// Each node is 8 M31 elements (32 bytes). With 32KB shared memory, we can fit 1024 nodes = 8192 M31 elements.
// But the working set at the leaf level is subtree_size * 8 M31 values.
// We use subtree_size up to 512 nodes (512 * 8 = 4096 M31 = 16KB in shared memory).
#define M31_MERKLE_SUBTREE_SIZE 512
#define M31_NODE_SIZE 8

kernel void poseidon2_m31_merkle_fused(
    device const uint* leaves       [[buffer(0)]],
    device uint* roots              [[buffer(1)]],
    constant uint* rc               [[buffer(2)]],
    constant uint& num_levels       [[buffer(3)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint tg_size                    [[threads_per_threadgroup]]
) {
    // Shared memory: up to 512 nodes * 8 M31 = 4096 uint values = 16KB
    threadgroup uint shared_data[M31_MERKLE_SUBTREE_SIZE * M31_NODE_SIZE];

    uint subtree_size = 1u << num_levels;
    uint leaf_base = tgid * subtree_size * M31_NODE_SIZE;
    uint total_vals = subtree_size * M31_NODE_SIZE;

    // Load all leaf node data into shared memory
    for (uint i = tid; i < total_vals; i += tg_size) {
        shared_data[i] = leaves[leaf_base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint active = subtree_size;
    for (uint level = 0; level < num_levels; level++) {
        uint pairs = active >> 1;
        if (tid < pairs) {
            // Read left and right child nodes
            M31 left[M31_NODE_SIZE], right[M31_NODE_SIZE], out[M31_NODE_SIZE];
            uint left_base = tid * 2 * M31_NODE_SIZE;
            uint right_base = left_base + M31_NODE_SIZE;
            for (uint i = 0; i < M31_NODE_SIZE; i++) {
                left[i] = M31{shared_data[left_base + i]};
                right[i] = M31{shared_data[right_base + i]};
            }

            p2m31_hash_pair(left, right, out, rc);

            // Write result back (compact to front)
            uint out_base = tid * M31_NODE_SIZE;
            for (uint i = 0; i < M31_NODE_SIZE; i++) {
                shared_data[out_base + i] = out[i].v;
            }
        }
        active = pairs;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write root (first node)
    if (tid < M31_NODE_SIZE) {
        roots[tgid * M31_NODE_SIZE + tid] = shared_data[tid];
    }
}

// Variable-size batch fused Merkle: each threadgroup processes an independent tree.
// tree_params[tgid*2] = leaf_offset (in M31 elements), tree_params[tgid*2+1] = num_levels
kernel void poseidon2_m31_merkle_fused_batch(
    device const uint* all_leaves   [[buffer(0)]],
    device uint* all_roots          [[buffer(1)]],
    constant uint* rc               [[buffer(2)]],
    device const uint* tree_params  [[buffer(3)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint tg_size                    [[threads_per_threadgroup]]
) {
    threadgroup uint shared_data[M31_MERKLE_SUBTREE_SIZE * M31_NODE_SIZE];

    uint leaf_offset = tree_params[tgid * 2];
    uint nlevels = tree_params[tgid * 2 + 1];
    uint subtree_size = 1u << nlevels;
    uint total_vals = subtree_size * M31_NODE_SIZE;

    for (uint i = tid; i < total_vals; i += tg_size) {
        shared_data[i] = all_leaves[leaf_offset + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint active = subtree_size;
    for (uint level = 0; level < nlevels; level++) {
        uint pairs = active >> 1;
        if (tid < pairs) {
            M31 left[M31_NODE_SIZE], right[M31_NODE_SIZE], out[M31_NODE_SIZE];
            uint left_base = tid * 2 * M31_NODE_SIZE;
            uint right_base = left_base + M31_NODE_SIZE;
            for (uint i = 0; i < M31_NODE_SIZE; i++) {
                left[i] = M31{shared_data[left_base + i]};
                right[i] = M31{shared_data[right_base + i]};
            }
            p2m31_hash_pair(left, right, out, rc);
            uint out_base = tid * M31_NODE_SIZE;
            for (uint i = 0; i < M31_NODE_SIZE; i++) {
                shared_data[out_base + i] = out[i].v;
            }
        }
        active = pairs;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < M31_NODE_SIZE) {
        all_roots[tgid * M31_NODE_SIZE + tid] = shared_data[tid];
    }
}
