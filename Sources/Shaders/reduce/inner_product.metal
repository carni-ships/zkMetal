// GPU-accelerated inner product kernels for BN254 Fr field elements
//
// Kernels:
//   ip_field_mul_reduce   — fused multiply + tree reduction for single inner product
//   ip_field_partial_reduce — partial sums for multi-level reduction of products
//   ip_batch_field        — batch multiple inner products in one dispatch
//
// Architecture:
//   - Each thread computes a_i * b_i, then participates in threadgroup reduction.
//   - SIMD shuffle reduces within each 32-lane SIMD group (no barriers needed).
//   - Shared memory tree reduces across SIMD groups.
//   - Multi-pass: host dispatches pass 1 over products, pass 2 over partials.
//   - Batch kernel: each threadgroup handles one (a,b) pair, reducing independently.

#include "../fields/bn254_fr.metal"

// ============================================================================
// SIMD shuffle helper for Fr (8x uint32)
// ============================================================================

inline Fr ip_fr_simd_shuffle_down(Fr a, uint offset) {
    Fr r;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        r.v[k] = simd_shuffle_down(a.v[k], offset);
    }
    return r;
}

// ============================================================================
// SIMD-level sum reduction for Fr
// ============================================================================

inline Fr ip_fr_simd_reduce_sum(Fr val, uint lane) {
    #pragma unroll
    for (uint off = 16; off > 0; off >>= 1) {
        Fr other = ip_fr_simd_shuffle_down(val, off);
        if (lane < off) {
            val = fr_add(val, other);
        }
    }
    return val;
}

// ============================================================================
// ip_field_mul_reduce — single inner product: Sigma a_i * b_i
//
// Each thread loads one (a_i, b_i) pair, computes the product, then
// participates in threadgroup tree reduction. Output: one partial sum
// per threadgroup. Host runs a second pass to sum partials.
// ============================================================================

kernel void ip_field_mul_reduce(
    device const Fr* a              [[buffer(0)]],
    device const Fr* b              [[buffer(1)]],
    device Fr* output               [[buffer(2)]],
    constant uint& count            [[buffer(3)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint tg_size                    [[threads_per_threadgroup]],
    uint simd_lane                  [[thread_index_in_simdgroup]],
    uint simd_id                    [[simdgroup_index_in_threadgroup]]
) {
    threadgroup Fr simd_partials[32];

    // Fused multiply-accumulate: each thread handles one element
    Fr acc = fr_zero();
    uint global_id = tgid * tg_size + tid;
    if (global_id < count) {
        acc = fr_mul(a[global_id], b[global_id]);
    }

    // Phase 1: SIMD shuffle reduction within each 32-lane group
    acc = ip_fr_simd_reduce_sum(acc, simd_lane);

    // Lane 0 of each SIMD group writes to shared memory
    uint n_simd_groups = (tg_size + 31) / 32;
    if (simd_lane == 0) {
        simd_partials[simd_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: First SIMD group reduces the simd_partials
    if (simd_id == 0) {
        acc = (simd_lane < n_simd_groups) ? simd_partials[simd_lane] : fr_zero();
        acc = ip_fr_simd_reduce_sum(acc, simd_lane);
        if (simd_lane == 0) {
            output[tgid] = acc;
        }
    }
}

// ============================================================================
// ip_field_partial_reduce — sum reduction of pre-computed partial sums
//
// Used as the second pass: reduces the partial sums from ip_field_mul_reduce
// down to a single result. Identical to fr_reduce_sum but kept separate
// to avoid name collisions when both shaders are compiled together.
// ============================================================================

kernel void ip_field_partial_reduce(
    device const Fr* input          [[buffer(0)]],
    device Fr* output               [[buffer(1)]],
    constant uint& count            [[buffer(2)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint tg_size                    [[threads_per_threadgroup]],
    uint simd_lane                  [[thread_index_in_simdgroup]],
    uint simd_id                    [[simdgroup_index_in_threadgroup]]
) {
    threadgroup Fr simd_partials[32];

    Fr acc = fr_zero();
    uint global_id = tgid * tg_size + tid;
    if (global_id < count) {
        acc = input[global_id];
    }

    acc = ip_fr_simd_reduce_sum(acc, simd_lane);

    uint n_simd_groups = (tg_size + 31) / 32;
    if (simd_lane == 0) {
        simd_partials[simd_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        acc = (simd_lane < n_simd_groups) ? simd_partials[simd_lane] : fr_zero();
        acc = ip_fr_simd_reduce_sum(acc, simd_lane);
        if (simd_lane == 0) {
            output[tgid] = acc;
        }
    }
}

// ============================================================================
// ip_batch_field — batch multiple inner products in one dispatch
//
// Layout: a_data and b_data are concatenated vectors.
//   offsets[k] = start index of the k-th pair in a_data/b_data
//   lengths[k] = number of elements in the k-th pair
//   output[k]  = result of inner product k
//
// Each threadgroup handles one inner product. Threadgroup index maps to
// the batch index. Within the threadgroup, threads cooperatively compute
// the fused multiply-reduce.
// ============================================================================

kernel void ip_batch_field(
    device const Fr* a_data         [[buffer(0)]],
    device const Fr* b_data         [[buffer(1)]],
    device Fr* output               [[buffer(2)]],
    device const uint* offsets      [[buffer(3)]],
    device const uint* lengths      [[buffer(4)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint tg_size                    [[threads_per_threadgroup]],
    uint simd_lane                  [[thread_index_in_simdgroup]],
    uint simd_id                    [[simdgroup_index_in_threadgroup]]
) {
    threadgroup Fr simd_partials[32];

    uint offset = offsets[tgid];
    uint len = lengths[tgid];

    // Fused multiply-accumulate over elements assigned to this threadgroup
    Fr acc = fr_zero();
    for (uint i = tid; i < len; i += tg_size) {
        Fr prod = fr_mul(a_data[offset + i], b_data[offset + i]);
        acc = fr_add(acc, prod);
    }

    // SIMD shuffle reduction
    acc = ip_fr_simd_reduce_sum(acc, simd_lane);

    uint n_simd_groups = (tg_size + 31) / 32;
    if (simd_lane == 0) {
        simd_partials[simd_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        acc = (simd_lane < n_simd_groups) ? simd_partials[simd_lane] : fr_zero();
        acc = ip_fr_simd_reduce_sum(acc, simd_lane);
        if (simd_lane == 0) {
            output[tgid] = acc;
        }
    }
}
