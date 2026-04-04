// Batch Merkle path verification on GPU
// Each thread verifies one Merkle authentication path using Poseidon2 2-to-1 hashing.
// Exploits Apple Silicon unified memory: CPU writes paths into shared MTLBuffer,
// GPU reads and verifies immediately without any data transfer.

#include "../fields/bn254_fr.metal"

// ---- Poseidon2 inline (2-to-1 hash for Merkle path verification) ----

// S-box: x -> x^5
Fr mv_p2_sbox(Fr x) {
    Fr x2 = fr_mul(x, x);
    Fr x4 = fr_mul(x2, x2);
    return fr_mul(x4, x);
}

// External linear layer: circulant [2,1,1] for t=3
void mv_p2_external_layer(thread Fr &s0, thread Fr &s1, thread Fr &s2) {
    Fr sum = fr_reduce(fr_add_lazy(fr_add_lazy(s0, s1), s2));
    s0 = fr_add_lazy(s0, sum);
    s1 = fr_add_lazy(s1, sum);
    s2 = fr_add_lazy(s2, sum);
}

// Internal linear layer: M_I = [[2,1,1],[1,2,1],[1,1,3]]
void mv_p2_internal_layer(thread Fr &s0, thread Fr &s1, thread Fr &s2) {
    Fr sum = fr_add(fr_add(s0, s1), s2);
    s0 = fr_add(s0, sum);
    s1 = fr_add(s1, sum);
    s2 = fr_add(fr_add(s2, sum), s2);
}

// Full Poseidon2 2-to-1 hash: hash(a, b) -> Fr
Fr mv_poseidon2_hash(Fr a, Fr b, constant Fr* rc) {
    Fr s0 = a;
    Fr s1 = b;
    Fr s2 = fr_zero();

    mv_p2_external_layer(s0, s1, s2);

    // First 4 full rounds
    for (uint r = 0; r < 4; r++) {
        uint rc_base = r * 3;
        s0 = fr_add_lazy(s0, rc[rc_base]);
        s1 = fr_add_lazy(s1, rc[rc_base + 1]);
        s2 = fr_add_lazy(s2, rc[rc_base + 2]);
        s0 = mv_p2_sbox(s0);
        s1 = mv_p2_sbox(s1);
        s2 = mv_p2_sbox(s2);
        mv_p2_external_layer(s0, s1, s2);
    }

    s0 = fr_reduce(s0); s1 = fr_reduce(s1); s2 = fr_reduce(s2);

    // 56 partial rounds
    for (uint r = 4; r < 60; r++) {
        s0 = fr_add_lazy(s0, rc[r * 3]);
        s0 = mv_p2_sbox(s0);
        mv_p2_internal_layer(s0, s1, s2);
    }

    // Last 4 full rounds
    for (uint r = 60; r < 64; r++) {
        uint rc_base = r * 3;
        s0 = fr_add_lazy(s0, rc[rc_base]);
        s1 = fr_add_lazy(s1, rc[rc_base + 1]);
        s2 = fr_add_lazy(s2, rc[rc_base + 2]);
        s0 = mv_p2_sbox(s0);
        s1 = mv_p2_sbox(s1);
        s2 = mv_p2_sbox(s2);
        mv_p2_external_layer(s0, s1, s2);
    }

    return fr_reduce(s0);
}

// Compare two Fr values for equality
bool fr_eq(Fr a, Fr b) {
    for (uint i = 0; i < 8; i++) {
        if (a.v[i] != b.v[i]) return false;
    }
    return true;
}

// ---- Batch Merkle Path Verification Kernel ----
// Each thread verifies one Merkle path by hashing up from the leaf to the root
// and comparing against the expected root.
//
// Buffer layout:
//   leaves:  count Fr elements, one leaf per path
//   paths:   count * max_depth Fr elements, packed sequentially
//            path[tid * max_depth + level] = sibling at that level
//   indices: count uint32 values, leaf index for each path
//   roots:   count Fr elements, expected Merkle root for each path
//   depths:  count uint32 values, actual depth of each path
//   results: count uint32 values, output: 1 = valid, 0 = invalid
//
kernel void batch_merkle_verify_poseidon2(
    device const Fr* leaves     [[buffer(0)]],
    device const Fr* paths      [[buffer(1)]],
    device const uint* indices  [[buffer(2)]],
    device const Fr* roots      [[buffer(3)]],
    device const uint* depths   [[buffer(4)]],
    device uint* results        [[buffer(5)]],
    constant Fr* rc             [[buffer(6)]],
    constant uint& max_depth    [[buffer(7)]],
    constant uint& count        [[buffer(8)]],
    uint tid                    [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    Fr current = leaves[tid];
    uint idx = indices[tid];
    uint depth = depths[tid];

    // Walk up the Merkle path, hashing at each level
    for (uint level = 0; level < depth; level++) {
        Fr sibling = paths[tid * max_depth + level];

        // If index is even, current is left child; otherwise right child
        Fr left, right;
        if (idx & 1) {
            left = sibling;
            right = current;
        } else {
            left = current;
            right = sibling;
        }

        current = mv_poseidon2_hash(left, right, rc);
        idx >>= 1;
    }

    // Compare computed root against expected root
    Fr expected = roots[tid];
    results[tid] = fr_eq(fr_reduce(current), fr_reduce(expected)) ? 1u : 0u;
}
