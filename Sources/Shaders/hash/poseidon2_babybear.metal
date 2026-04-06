// Poseidon2 GPU kernel for BabyBear, t=16
// d=7 (x^7 S-box), rounds_f=8 (4+4), rounds_p=13
// Parameters match SP1/Plonky3 exactly for full compatibility.
// Each thread computes one independent Poseidon2 permutation.

#include "../fields/babybear.metal"

// Width and round parameters
#define P2BB_T 16
#define P2BB_RATE 8
#define P2BB_RF_HALF 4
#define P2BB_RP 13
#define P2BB_TOTAL_ROUNDS 21  // 8 + 13

// Internal diagonal constants (from Plonky3 BabyBear width-16)
// V = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/4, 1/8, 1/2^27, -1/2^8, -1/16, -1/2^27]
// Converted to canonical BabyBear field elements mod p=0x78000001
constant uint P2BB_INTERNAL_DIAG[16] = {
    0x77ffffffu, 0x00000001u, 0x00000002u, 0x3c000001u,
    0x00000003u, 0x00000004u, 0x3c000000u, 0x77fffffeu,
    0x77fffffdu, 0x77880001u, 0x5a000001u, 0x69000001u,
    0x77fffff2u, 0x00780000u, 0x07800000u, 0x0000000fu
};

// S-box: x -> x^7 (standard for BabyBear Poseidon2)
Bb p2bb_sbox(Bb x) {
    Bb x2 = bb_sqr(x);
    Bb x3 = bb_mul(x2, x);
    Bb x6 = bb_sqr(x3);
    return bb_mul(x6, x);
}

// M4 circulant matrix: circ(2, 3, 1, 1) on 4 elements
// Same structure as M31 Poseidon2 — standard Poseidon2 external layer building block
void p2bb_m4(thread Bb &s0, thread Bb &s1, thread Bb &s2, thread Bb &s3) {
    Bb t0 = bb_add(s0, s1);
    Bb t1 = bb_add(s2, s3);
    Bb t2 = bb_add(bb_add(s1, s1), t1);
    Bb t3 = bb_add(bb_add(s3, s3), t0);
    s0 = bb_add(t0, t3);
    s1 = bb_add(t1, t2);
    s2 = bb_add(t0, t2);
    s3 = bb_add(t1, t3);
}

// External linear layer for t=16: M4 on 4x4 blocks + cross-block mixing
void p2bb_external_layer(thread Bb *s) {
    // Apply M4 to each 4-element block
    p2bb_m4(s[0], s[1], s[2], s[3]);
    p2bb_m4(s[4], s[5], s[6], s[7]);
    p2bb_m4(s[8], s[9], s[10], s[11]);
    p2bb_m4(s[12], s[13], s[14], s[15]);

    // Cross-block mixing: each element += sum of corresponding positions
    for (uint i = 0; i < 4; i++) {
        Bb sum = bb_add(bb_add(s[i], s[i+4]), bb_add(s[i+8], s[i+12]));
        s[i]    = bb_add(s[i], sum);
        s[i+4]  = bb_add(s[i+4], sum);
        s[i+8]  = bb_add(s[i+8], sum);
        s[i+12] = bb_add(s[i+12], sum);
    }
}

// Internal linear layer: y_i = diag[i] * x_i + sum(x_j)
// Specializes small-integer diagonals to additions instead of multiplies.
// Diag = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/256, 1/4, 1/8, 1/2^27, -1/256, -1/16, -1/2^27]
// Indices 0,1,2,4,5,7,8 have small int diagonals → use adds/neg instead of bb_mul.
void p2bb_internal_layer(thread Bb *s) {
    // Tree-reduce sum for lower latency (4 levels vs 15 serial adds)
    Bb s03 = bb_add(bb_add(s[0], s[1]), bb_add(s[2], s[3]));
    Bb s47 = bb_add(bb_add(s[4], s[5]), bb_add(s[6], s[7]));
    Bb s8b = bb_add(bb_add(s[8], s[9]), bb_add(s[10], s[11]));
    Bb scf = bb_add(bb_add(s[12], s[13]), bb_add(s[14], s[15]));
    Bb sum = bb_add(bb_add(s03, s47), bb_add(s8b, scf));

    // [0] = -2: neg(2x) + sum
    Bb d0 = bb_neg(bb_add(s[0], s[0]));
    s[0] = bb_add(d0, sum);
    // [1] = 1: x + sum (identity, no multiply)
    s[1] = bb_add(s[1], sum);
    // [2] = 2: 2x + sum
    s[2] = bb_add(bb_add(s[2], s[2]), sum);
    // [3] = 1/2: needs multiply
    s[3] = bb_add(bb_mul(s[3], Bb{P2BB_INTERNAL_DIAG[3]}), sum);
    // [4] = 3: 3x + sum
    { Bb x2 = bb_add(s[4], s[4]); s[4] = bb_add(bb_add(x2, s[4]), sum); }
    // [5] = 4: 4x + sum
    { Bb x2 = bb_add(s[5], s[5]); s[5] = bb_add(bb_add(x2, x2), sum); }
    // [6] = -1/2: needs multiply
    s[6] = bb_add(bb_mul(s[6], Bb{P2BB_INTERNAL_DIAG[6]}), sum);
    // [7] = -3: neg(3x) + sum
    { Bb x2 = bb_add(s[7], s[7]); s[7] = bb_add(bb_neg(bb_add(x2, s[7])), sum); }
    // [8] = -4: neg(4x) + sum
    { Bb x2 = bb_add(s[8], s[8]); s[8] = bb_add(bb_neg(bb_add(x2, x2)), sum); }
    // [9..15]: non-trivial constants, use multiply
    s[9]  = bb_add(bb_mul(s[9],  Bb{P2BB_INTERNAL_DIAG[9]}),  sum);
    s[10] = bb_add(bb_mul(s[10], Bb{P2BB_INTERNAL_DIAG[10]}), sum);
    s[11] = bb_add(bb_mul(s[11], Bb{P2BB_INTERNAL_DIAG[11]}), sum);
    s[12] = bb_add(bb_mul(s[12], Bb{P2BB_INTERNAL_DIAG[12]}), sum);
    s[13] = bb_add(bb_mul(s[13], Bb{P2BB_INTERNAL_DIAG[13]}), sum);
    s[14] = bb_add(bb_mul(s[14], Bb{P2BB_INTERNAL_DIAG[14]}), sum);
    s[15] = bb_add(bb_mul(s[15], Bb{P2BB_INTERNAL_DIAG[15]}), sum);
}

// Full Poseidon2 permutation on 16 BabyBear elements
void p2bb_permute(thread Bb *s, constant uint *rc) {
    // Initial external layer
    p2bb_external_layer(s);

    // First half of full rounds (0..3)
    for (uint r = 0; r < P2BB_RF_HALF; r++) {
        uint rc_base = r * P2BB_T;
        for (uint i = 0; i < P2BB_T; i++) s[i] = bb_add(s[i], Bb{rc[rc_base + i]});
        for (uint i = 0; i < P2BB_T; i++) s[i] = p2bb_sbox(s[i]);
        p2bb_external_layer(s);
    }

    // Partial rounds (4..16)
    for (uint r = P2BB_RF_HALF; r < P2BB_RF_HALF + P2BB_RP; r++) {
        s[0] = bb_add(s[0], Bb{rc[r * P2BB_T]});
        s[0] = p2bb_sbox(s[0]);
        p2bb_internal_layer(s);
    }

    // Second half of full rounds (17..20)
    for (uint r = P2BB_RF_HALF + P2BB_RP; r < P2BB_TOTAL_ROUNDS; r++) {
        uint rc_base = r * P2BB_T;
        for (uint i = 0; i < P2BB_T; i++) s[i] = bb_add(s[i], Bb{rc[rc_base + i]});
        for (uint i = 0; i < P2BB_T; i++) s[i] = p2bb_sbox(s[i]);
        p2bb_external_layer(s);
    }
}

// Poseidon2 permutation kernel
kernel void poseidon2_bb_permute(
    device const uint* input        [[buffer(0)]],
    device uint* output             [[buffer(1)]],
    constant uint* rc               [[buffer(2)]],
    constant uint& count            [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    uint base = gid * P2BB_T;
    Bb s[P2BB_T];
    for (uint i = 0; i < P2BB_T; i++) {
        s[i] = Bb{input[base + i]};
    }

    p2bb_permute(s, rc);

    for (uint i = 0; i < P2BB_T; i++) {
        output[base + i] = s[i].v;
    }
}

// 2-to-1 compression: hash pairs of 8-element BabyBear blocks
// input: array of 2*n * 8 = 16*n BabyBear elements
// output: array of n * 8 BabyBear elements
kernel void poseidon2_bb_hash_pairs(
    device const uint* input        [[buffer(0)]],
    device uint* output             [[buffer(1)]],
    constant uint* rc               [[buffer(2)]],
    constant uint& count            [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    Bb s[P2BB_T];
    uint in_base = gid * P2BB_T;
    for (uint i = 0; i < P2BB_T; i++) {
        s[i] = Bb{input[in_base + i]};
    }

    p2bb_permute(s, rc);

    // Output rate portion (first 8 elements)
    uint out_base = gid * P2BB_RATE;
    for (uint i = 0; i < P2BB_RATE; i++) {
        output[out_base + i] = s[i].v;
    }
}

// Inline hash pair for fused Merkle kernels
void p2bb_hash_pair(thread Bb *left, thread Bb *right, thread Bb *out, constant uint *rc) {
    Bb s[P2BB_T];
    for (uint i = 0; i < 8; i++) s[i] = left[i];
    for (uint i = 0; i < 8; i++) s[i + 8] = right[i];

    p2bb_permute(s, rc);

    for (uint i = 0; i < 8; i++) out[i] = s[i];
}

// Fused multi-level Merkle tree for BabyBear Poseidon2
#define BB_MERKLE_SUBTREE_SIZE 512
#define BB_NODE_SIZE 8

kernel void poseidon2_bb_merkle_fused(
    device const uint* leaves       [[buffer(0)]],
    device uint* roots              [[buffer(1)]],
    constant uint* rc               [[buffer(2)]],
    constant uint& num_levels       [[buffer(3)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint tg_size                    [[threads_per_threadgroup]]
) {
    threadgroup uint shared_data[BB_MERKLE_SUBTREE_SIZE * BB_NODE_SIZE];

    uint subtree_size = 1u << num_levels;
    uint leaf_base = tgid * subtree_size * BB_NODE_SIZE;
    uint total_vals = subtree_size * BB_NODE_SIZE;

    for (uint i = tid; i < total_vals; i += tg_size) {
        shared_data[i] = leaves[leaf_base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint active = subtree_size;
    for (uint level = 0; level < num_levels; level++) {
        uint pairs = active >> 1;
        if (tid < pairs) {
            Bb left[BB_NODE_SIZE], right[BB_NODE_SIZE], out[BB_NODE_SIZE];
            uint left_base = tid * 2 * BB_NODE_SIZE;
            uint right_base = left_base + BB_NODE_SIZE;
            for (uint i = 0; i < BB_NODE_SIZE; i++) {
                left[i] = Bb{shared_data[left_base + i]};
                right[i] = Bb{shared_data[right_base + i]};
            }

            p2bb_hash_pair(left, right, out, rc);

            uint out_base = tid * BB_NODE_SIZE;
            for (uint i = 0; i < BB_NODE_SIZE; i++) {
                shared_data[out_base + i] = out[i].v;
            }
        }
        active = pairs;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < BB_NODE_SIZE) {
        roots[tgid * BB_NODE_SIZE + tid] = shared_data[tid];
    }
}

// Variable-size batch fused Merkle
kernel void poseidon2_bb_merkle_fused_batch(
    device const uint* all_leaves   [[buffer(0)]],
    device uint* all_roots          [[buffer(1)]],
    constant uint* rc               [[buffer(2)]],
    device const uint* tree_params  [[buffer(3)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint tg_size                    [[threads_per_threadgroup]]
) {
    threadgroup uint shared_data[BB_MERKLE_SUBTREE_SIZE * BB_NODE_SIZE];

    uint leaf_offset = tree_params[tgid * 2];
    uint nlevels = tree_params[tgid * 2 + 1];
    uint subtree_size = 1u << nlevels;
    uint total_vals = subtree_size * BB_NODE_SIZE;

    for (uint i = tid; i < total_vals; i += tg_size) {
        shared_data[i] = all_leaves[leaf_offset + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint active = subtree_size;
    for (uint level = 0; level < nlevels; level++) {
        uint pairs = active >> 1;
        if (tid < pairs) {
            Bb left[BB_NODE_SIZE], right[BB_NODE_SIZE], out[BB_NODE_SIZE];
            uint left_base = tid * 2 * BB_NODE_SIZE;
            uint right_base = left_base + BB_NODE_SIZE;
            for (uint i = 0; i < BB_NODE_SIZE; i++) {
                left[i] = Bb{shared_data[left_base + i]};
                right[i] = Bb{shared_data[right_base + i]};
            }
            p2bb_hash_pair(left, right, out, rc);
            uint out_base = tid * BB_NODE_SIZE;
            for (uint i = 0; i < BB_NODE_SIZE; i++) {
                shared_data[out_base + i] = out[i].v;
            }
        }
        active = pairs;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < BB_NODE_SIZE) {
        all_roots[tgid * BB_NODE_SIZE + tid] = shared_data[tid];
    }
}
