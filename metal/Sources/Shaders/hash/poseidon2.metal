// Poseidon2 GPU kernel for BN254 Fr, t=3
// Each thread computes one independent Poseidon2 permutation.
// d=5 (x^5 S-box), rounds_f=8, rounds_p=56

#include "../fields/bn254_fr.metal"

// S-box: x -> x^5
Fr p2_sbox(Fr x) {
    Fr x2 = fr_mul(x, x);
    Fr x4 = fr_mul(x2, x2);
    return fr_mul(x4, x);
}

// External linear layer: circulant [2,1,1] for t=3
// M_E * [a,b,c] = [2a+b+c, a+2b+c, a+b+2c] = [a+(a+b+c), b+(a+b+c), c+(a+b+c)]
void p2_external_layer(thread Fr &s0, thread Fr &s1, thread Fr &s2) {
    Fr sum = fr_add(fr_add(s0, s1), s2);
    s0 = fr_add(s0, sum);
    s1 = fr_add(s1, sum);
    s2 = fr_add(s2, sum);
}

// Internal linear layer: M_I = [[2,1,1],[1,2,1],[1,1,3]]
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

    // First half of full rounds (rounds 0..3)
    #pragma unroll
    for (uint r = 0; r < 4; r++) {
        uint rc_base = r * 3;
        s0 = fr_add(s0, rc[rc_base]);
        s1 = fr_add(s1, rc[rc_base + 1]);
        s2 = fr_add(s2, rc[rc_base + 2]);
        s0 = p2_sbox(s0);
        s1 = p2_sbox(s1);
        s2 = p2_sbox(s2);
        p2_external_layer(s0, s1, s2);
    }

    // Partial rounds (rounds 4..59) — only s0 gets RC and S-box
    for (uint r = 4; r < 60; r++) {
        s0 = fr_add(s0, rc[r * 3]);
        s0 = p2_sbox(s0);
        p2_internal_layer(s0, s1, s2);
    }

    // Second half of full rounds (rounds 60..63)
    #pragma unroll
    for (uint r = 60; r < 64; r++) {
        uint rc_base = r * 3;
        s0 = fr_add(s0, rc[rc_base]);
        s1 = fr_add(s1, rc[rc_base + 1]);
        s2 = fr_add(s2, rc[rc_base + 2]);
        s0 = p2_sbox(s0);
        s1 = p2_sbox(s1);
        s2 = p2_sbox(s2);
        p2_external_layer(s0, s1, s2);
    }

    // Write output
    output[base] = s0;
    output[base + 1] = s1;
    output[base + 2] = s2;
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
        s0 = fr_add(s0, rc[rc_base]);
        s1 = fr_add(s1, rc[rc_base + 1]);
        s2 = fr_add(s2, rc[rc_base + 2]);
        s0 = p2_sbox(s0);
        s1 = p2_sbox(s1);
        s2 = p2_sbox(s2);
        p2_external_layer(s0, s1, s2);
    }

    for (uint r = 4; r < 60; r++) {
        s0 = fr_add(s0, rc[r * 3]);
        s0 = p2_sbox(s0);
        p2_internal_layer(s0, s1, s2);
    }

    #pragma unroll
    for (uint r = 60; r < 64; r++) {
        uint rc_base = r * 3;
        s0 = fr_add(s0, rc[rc_base]);
        s1 = fr_add(s1, rc[rc_base + 1]);
        s2 = fr_add(s2, rc[rc_base + 2]);
        s0 = p2_sbox(s0);
        s1 = p2_sbox(s1);
        s2 = p2_sbox(s2);
        p2_external_layer(s0, s1, s2);
    }

    output[gid] = s0;
}

// Inline Poseidon2 hash of a pair (a, b) → Fr result
// Used in fused Merkle tree kernel to avoid function call overhead
Fr p2_hash_pair(Fr a, Fr b, constant Fr* rc) {
    Fr s0 = a, s1 = b, s2 = fr_zero();

    p2_external_layer(s0, s1, s2);

    #pragma unroll
    for (uint r = 0; r < 4; r++) {
        uint rc_base = r * 3;
        s0 = fr_add(s0, rc[rc_base]);
        s1 = fr_add(s1, rc[rc_base + 1]);
        s2 = fr_add(s2, rc[rc_base + 2]);
        s0 = p2_sbox(s0); s1 = p2_sbox(s1); s2 = p2_sbox(s2);
        p2_external_layer(s0, s1, s2);
    }
    for (uint r = 4; r < 60; r++) {
        s0 = fr_add(s0, rc[r * 3]);
        s0 = p2_sbox(s0);
        p2_internal_layer(s0, s1, s2);
    }
    #pragma unroll
    for (uint r = 60; r < 64; r++) {
        uint rc_base = r * 3;
        s0 = fr_add(s0, rc[rc_base]);
        s1 = fr_add(s1, rc[rc_base + 1]);
        s2 = fr_add(s2, rc[rc_base + 2]);
        s0 = p2_sbox(s0); s1 = p2_sbox(s1); s2 = p2_sbox(s2);
        p2_external_layer(s0, s1, s2);
    }
    return s0;
}

// Fused multi-level Merkle tree: each threadgroup processes a subtree
// of SUBTREE_SIZE leaves down to 1 hash, eliminating log2(SUBTREE_SIZE)-1 barriers.
// tg_size must be SUBTREE_SIZE/2 = 512. Shared memory: 1024 Fr = 32KB.
#define MERKLE_SUBTREE_SIZE 1024

kernel void poseidon2_merkle_fused(
    device const Fr* leaves       [[buffer(0)]],    // SUBTREE_SIZE * num_groups leaves
    device Fr* roots              [[buffer(1)]],    // one result per threadgroup
    constant Fr* rc               [[buffer(2)]],    // constant address space for uniform broadcast
    constant uint& num_levels     [[buffer(3)]],    // log2(SUBTREE_SIZE) = 10
    uint tid                      [[thread_index_in_threadgroup]],
    uint tgid                     [[threadgroup_position_in_grid]],
    uint tg_size                  [[threads_per_threadgroup]]
) {
    threadgroup Fr shared_data[MERKLE_SUBTREE_SIZE];

    // Load leaves into shared memory
    uint leaf_base = tgid * MERKLE_SUBTREE_SIZE;
    shared_data[tid] = leaves[leaf_base + tid];
    shared_data[tid + tg_size] = leaves[leaf_base + tid + tg_size];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint active = MERKLE_SUBTREE_SIZE;
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

