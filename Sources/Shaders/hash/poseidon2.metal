// Poseidon2 GPU kernel for BN254 Fr, t=3
// Each thread computes one independent Poseidon2 permutation.
// d=5 (x^5 S-box), rounds_f=8, rounds_p=56

#include "../fields/bn254_fr.metal"

// S-box: x -> x^5
// Input can be lazy (up to ~3p, fits in 256 bits).
// CIOS handles inputs < 2^256, output is fully reduced to [0, p).
Fr p2_sbox(Fr x) {
    Fr x2 = fr_mul(x, x);
    Fr x4 = fr_mul(x2, x2);
    return fr_mul(x4, x);
}

// External linear layer: circulant [2,1,1] for t=3
// M_E * [a,b,c] = [2a+b+c, a+2b+c, a+b+2c] = [a+(a+b+c), b+(a+b+c), c+(a+b+c)]
// Input: fully reduced [0, p). Output: lazy, up to 2p < 2^255.
void p2_external_layer(thread Fr &s0, thread Fr &s1, thread Fr &s2) {
    // Reduce sum to [0, p) so outputs stay bounded: si + sum <= 2p < 2^255
    Fr sum = fr_reduce(fr_add_lazy(fr_add_lazy(s0, s1), s2));
    s0 = fr_add_lazy(s0, sum);
    s1 = fr_add_lazy(s1, sum);
    s2 = fr_add_lazy(s2, sum);
}

// Internal linear layer: M_I = [[2,1,1],[1,2,1],[1,1,3]]
// In partial rounds, s0 comes from S-box (reduced [0,p)), s1/s2 carry from previous layer.
// To prevent value explosion across rounds, we reduce all inputs first.
void p2_internal_layer(thread Fr &s0, thread Fr &s1, thread Fr &s2) {
    Fr sum = fr_add(fr_add(s0, s1), s2);
    s0 = fr_add(s0, sum);
    s1 = fr_add(s1, sum);
    s2 = fr_add(fr_add(s2, sum), s2);  // s2 + sum + s2 = a+b+3c
}

// Poseidon2 permutation kernel
// input: array of (3*n) Fr elements (n states of 3 elements each)
// rc: round constants, 64*3 = 192 Fr elements
// output: array of (3*n) Fr elements
kernel void poseidon2_permute(
    device const Fr* input        [[buffer(0)]],
    device Fr* output             [[buffer(1)]],
    constant Fr* rc               [[buffer(2)]],    // constant address space for uniform broadcast
    constant uint& count          [[buffer(3)]],    // number of permutations
    uint gid                      [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    // Load state
    uint base = gid * 3;
    Fr s0 = input[base];
    Fr s1 = input[base + 1];
    Fr s2 = input[base + 2];

    // Initial external linear layer
    p2_external_layer(s0, s1, s2);
    // After external_layer: s0,s1,s2 in [0, 2p)

    // First half of full rounds (rounds 0..3)
    // Lazy RC add: si + rc <= 2p + p = 3p < 2^256, safe for S-box/CIOS.
    // S-box output: [0, p). External layer output: [0, 2p).
    #pragma unroll
    for (uint r = 0; r < 4; r++) {
        uint rc_base = r * 3;
        s0 = fr_add_lazy(s0, rc[rc_base]);
        s1 = fr_add_lazy(s1, rc[rc_base + 1]);
        s2 = fr_add_lazy(s2, rc[rc_base + 2]);
        s0 = p2_sbox(s0);
        s1 = p2_sbox(s1);
        s2 = p2_sbox(s2);
        p2_external_layer(s0, s1, s2);
    }

    // Reduce before partial rounds: external_layer outputs are [0, 2p),
    // but internal_layer's fr_add expects [0, p).
    s0 = fr_reduce(s0); s1 = fr_reduce(s1); s2 = fr_reduce(s2);

    // Partial rounds (rounds 4..59) — only s0 gets RC and S-box
    // After reduce: all in [0, p). RC lazy add: [0, 2p). S-box output: [0, p).
    // Internal layer (fr_add): expects [0, p), outputs [0, p).
    for (uint r = 4; r < 60; r++) {
        s0 = fr_add_lazy(s0, rc[r * 3]);
        s0 = p2_sbox(s0);
        p2_internal_layer(s0, s1, s2);
    }

    // Second half of full rounds (rounds 60..63)
    // After internal_layer: all reduced [0, p). External layer: [0, 2p). RC lazy: [0, 3p).
    #pragma unroll
    for (uint r = 60; r < 64; r++) {
        uint rc_base = r * 3;
        s0 = fr_add_lazy(s0, rc[rc_base]);
        s1 = fr_add_lazy(s1, rc[rc_base + 1]);
        s2 = fr_add_lazy(s2, rc[rc_base + 2]);
        s0 = p2_sbox(s0);
        s1 = p2_sbox(s1);
        s2 = p2_sbox(s2);
        p2_external_layer(s0, s1, s2);
    }

    // Reduce outputs: external_layer outputs are [0, 2p), need final reduction
    output[base] = fr_reduce(s0);
    output[base + 1] = fr_reduce(s1);
    output[base + 2] = fr_reduce(s2);
}

// 2-to-1 compression: hash pairs of field elements
// input: array of 2*n Fr elements (pairs)
// output: array of n Fr elements (hashes)
// Each thread hashes one pair: state = [a, b, 0], output = permute(state)[0]
kernel void poseidon2_hash_pairs(
    device const Fr* input        [[buffer(0)]],
    device Fr* output             [[buffer(1)]],
    constant Fr* rc               [[buffer(2)]],    // constant address space for uniform broadcast
    constant uint& count          [[buffer(3)]],
    uint gid                      [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    Fr s0 = input[gid * 2];
    Fr s1 = input[gid * 2 + 1];
    Fr s2 = fr_zero();

    p2_external_layer(s0, s1, s2);

    #pragma unroll
    for (uint r = 0; r < 4; r++) {
        uint rc_base = r * 3;
        s0 = fr_add_lazy(s0, rc[rc_base]);
        s1 = fr_add_lazy(s1, rc[rc_base + 1]);
        s2 = fr_add_lazy(s2, rc[rc_base + 2]);
        s0 = p2_sbox(s0);
        s1 = p2_sbox(s1);
        s2 = p2_sbox(s2);
        p2_external_layer(s0, s1, s2);
    }

    s0 = fr_reduce(s0); s1 = fr_reduce(s1); s2 = fr_reduce(s2);

    for (uint r = 4; r < 60; r++) {
        s0 = fr_add_lazy(s0, rc[r * 3]);
        s0 = p2_sbox(s0);
        p2_internal_layer(s0, s1, s2);
    }

    #pragma unroll
    for (uint r = 60; r < 64; r++) {
        uint rc_base = r * 3;
        s0 = fr_add_lazy(s0, rc[rc_base]);
        s1 = fr_add_lazy(s1, rc[rc_base + 1]);
        s2 = fr_add_lazy(s2, rc[rc_base + 2]);
        s0 = p2_sbox(s0);
        s1 = p2_sbox(s1);
        s2 = p2_sbox(s2);
        p2_external_layer(s0, s1, s2);
    }

    output[gid] = fr_reduce(s0);
}

// Inline Poseidon2 hash of a pair (a, b) → Fr result
// Used in fused Merkle tree kernel to avoid function call overhead
Fr p2_hash_pair(Fr a, Fr b, constant Fr* rc) {
    Fr s0 = a, s1 = b, s2 = fr_zero();

    p2_external_layer(s0, s1, s2);

    #pragma unroll
    for (uint r = 0; r < 4; r++) {
        uint rc_base = r * 3;
        s0 = fr_add_lazy(s0, rc[rc_base]);
        s1 = fr_add_lazy(s1, rc[rc_base + 1]);
        s2 = fr_add_lazy(s2, rc[rc_base + 2]);
        s0 = p2_sbox(s0); s1 = p2_sbox(s1); s2 = p2_sbox(s2);
        p2_external_layer(s0, s1, s2);
    }
    s0 = fr_reduce(s0); s1 = fr_reduce(s1); s2 = fr_reduce(s2);
    for (uint r = 4; r < 60; r++) {
        s0 = fr_add_lazy(s0, rc[r * 3]);
        s0 = p2_sbox(s0);
        p2_internal_layer(s0, s1, s2);
    }
    #pragma unroll
    for (uint r = 60; r < 64; r++) {
        uint rc_base = r * 3;
        s0 = fr_add_lazy(s0, rc[rc_base]);
        s1 = fr_add_lazy(s1, rc[rc_base + 1]);
        s2 = fr_add_lazy(s2, rc[rc_base + 2]);
        s0 = p2_sbox(s0); s1 = p2_sbox(s1); s2 = p2_sbox(s2);
        p2_external_layer(s0, s1, s2);
    }
    return fr_reduce(s0);
}

// Fused multi-level Merkle tree: each threadgroup processes a subtree
// of SUBTREE_SIZE leaves down to 1 hash, eliminating log2(SUBTREE_SIZE)-1 barriers.
// Variable-size fused Merkle: subtree_size = 1 << num_levels (max 1024).
// Shared memory: 1024 Fr max = 32KB.
#define MERKLE_SUBTREE_SIZE 1024

// Batch fused Merkle: each threadgroup processes a different-sized tree.
// tree_params[tgid*2] = leaf_offset (in Fr elements), tree_params[tgid*2+1] = num_levels
// Eliminates per-tree dispatch overhead by batching all independent trees into one dispatch.
kernel void poseidon2_merkle_fused_batch(
    device const Fr* all_leaves   [[buffer(0)]],
    device Fr* all_roots          [[buffer(1)]],
    constant Fr* rc               [[buffer(2)]],
    device const uint* tree_params [[buffer(3)]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tgid                     [[threadgroup_position_in_grid]],
    uint tg_size                  [[threads_per_threadgroup]]
) {
    threadgroup Fr shared_data[MERKLE_SUBTREE_SIZE];

    uint leaf_offset = tree_params[tgid * 2];
    uint num_levels = tree_params[tgid * 2 + 1];
    uint subtree_size = 1u << num_levels;

    for (uint i = tid; i < subtree_size; i += tg_size) {
        shared_data[i] = all_leaves[leaf_offset + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint active = subtree_size;
    for (uint level = 0; level < num_levels; level++) {
        uint pairs = active >> 1;
        if (tid < pairs) {
            Fr a = shared_data[tid * 2];
            Fr b = shared_data[tid * 2 + 1];
            shared_data[tid] = p2_hash_pair(a, b, rc);
        }
        active = pairs;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        all_roots[tgid] = shared_data[0];
    }
}

kernel void poseidon2_merkle_fused(
    device const Fr* leaves       [[buffer(0)]],
    device Fr* roots              [[buffer(1)]],
    constant Fr* rc               [[buffer(2)]],
    constant uint& num_levels     [[buffer(3)]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tgid                     [[threadgroup_position_in_grid]],
    uint tg_size                  [[threads_per_threadgroup]]
) {
    threadgroup Fr shared_data[MERKLE_SUBTREE_SIZE];

    uint subtree_size = 1u << num_levels;
    uint leaf_base = tgid * subtree_size;

    // Generic load: stride loop for variable subtree sizes
    for (uint i = tid; i < subtree_size; i += tg_size) {
        shared_data[i] = leaves[leaf_base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint active = subtree_size;
    for (uint level = 0; level < num_levels; level++) {
        uint pairs = active >> 1;
        if (tid < pairs) {
            Fr a = shared_data[tid * 2];
            Fr b = shared_data[tid * 2 + 1];
            shared_data[tid] = p2_hash_pair(a, b, rc);
        }
        active = pairs;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        roots[tgid] = shared_data[0];
    }
}

// Fused Merkle that writes ALL intermediate nodes to the tree buffer.
// Tree layout: [leaves (n), level1 (n/2), level2 (n/4), ..., root (1)]
// Each subtree writes its intermediate nodes at the correct global offsets.
// buffer(4) = offsets per level: [n, n + n/2, n + n/2 + n/4, ...] i.e. cumulative start of each level.
// buffer(5) = num_subtrees (for computing per-subtree stride at each level)
kernel void poseidon2_merkle_fused_full(
    device const Fr* leaves       [[buffer(0)]],    // n leaves
    device Fr* tree               [[buffer(1)]],    // full tree output (2n-1 nodes)
    constant Fr* rc               [[buffer(2)]],
    constant uint& num_levels     [[buffer(3)]],    // log2(SUBTREE_SIZE) = 10
    constant uint* level_offsets  [[buffer(4)]],    // start index of each internal level in tree
    uint tid                      [[thread_index_in_threadgroup]],
    uint tgid                     [[threadgroup_position_in_grid]],
    uint tg_size                  [[threads_per_threadgroup]]
) {
    threadgroup Fr shared_data[MERKLE_SUBTREE_SIZE];

    uint leaf_base = tgid * MERKLE_SUBTREE_SIZE;
    shared_data[tid] = leaves[leaf_base + tid];
    shared_data[tid + tg_size] = leaves[leaf_base + tid + tg_size];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint active = MERKLE_SUBTREE_SIZE;
    uint subtree_stride = MERKLE_SUBTREE_SIZE / 2;  // pairs at level 0 within each subtree

    for (uint level = 0; level < num_levels; level++) {
        uint pairs = active >> 1;
        if (tid < pairs) {
            Fr a = shared_data[tid * 2];
            Fr b = shared_data[tid * 2 + 1];
            Fr h = p2_hash_pair(a, b, rc);
            shared_data[tid] = h;
            // Write to global tree: level_offsets[level] + tgid * subtree_stride + tid
            tree[level_offsets[level] + tgid * subtree_stride + tid] = h;
        }
        active = pairs;
        subtree_stride >>= 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Scattered incremental Merkle update: each thread hashes one parent node in a 1-indexed heap.
// dirty_indices[gid] = parent node index in the heap (1-indexed).
// tree[i] is the node value. Children of tree[i] are tree[2*i] and tree[2*i+1].
// Each thread reads its two children, hashes them, and writes the parent in-place.
// This avoids CPU gather/scatter for scattered updates (e.g., random leaf modifications).
kernel void poseidon2_merkle_update_scattered(
    device Fr* tree                [[buffer(0)]],    // 1-indexed heap: tree[1]=root, tree[cap..2*cap-1]=leaves
    constant Fr* rc               [[buffer(1)]],
    device const uint* dirty_indices [[buffer(2)]],  // parent node indices to rehash
    constant uint& count          [[buffer(3)]],     // number of dirty nodes
    uint gid                      [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    uint parent = dirty_indices[gid];
    Fr left  = tree[parent * 2];
    Fr right = tree[parent * 2 + 1];
    tree[parent] = p2_hash_pair(left, right, rc);
}

