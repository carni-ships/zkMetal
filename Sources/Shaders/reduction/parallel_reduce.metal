// Parallel reduction kernels for Metal GPU
// Supports sum, product over BN254 Fr / BabyBear, and min/max over uint32.
//
// Architecture:
//   - Each threadgroup reduces a tile of input to one partial result.
//   - SIMD shuffle reduces within each SIMD group (32 lanes) first,
//     then shared memory tree reduces across SIMD groups.
//   - Multi-pass: host dispatches pass 1 over input, pass 2 over partials, etc.
//   - Configurable threadgroup size (256 or 1024 via dispatch).

#include "../fields/bn254_fr.metal"
#include "../fields/babybear.metal"

// ============================================================================
// SIMD shuffle helpers
// ============================================================================

inline Fr simd_shuffle_down_fr(Fr a, uint offset) {
    Fr r;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        r.v[k] = simd_shuffle_down(a.v[k], offset);
    }
    return r;
}

inline uint simd_shuffle_down_u32(uint a, uint offset) {
    return simd_shuffle_down(a, offset);
}

// ============================================================================
// BN254 Fr — Sum reduction
// ============================================================================

// SIMD-level sum reduction (32 lanes -> lane 0)
inline Fr simd_reduce_sum_fr(Fr val, uint lane) {
    #pragma unroll
    for (uint off = 16; off > 0; off >>= 1) {
        Fr other = simd_shuffle_down_fr(val, off);
        if (lane < off) {
            val = fr_add(val, other);
        }
    }
    return val;
}

kernel void reduce_sum_bn254(
    device const Fr* input          [[buffer(0)]],
    device Fr* output               [[buffer(1)]],
    constant uint& count            [[buffer(2)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint tg_size                    [[threads_per_threadgroup]],
    uint simd_lane                  [[thread_index_in_simdgroup]],
    uint simd_id                    [[simdgroup_index_in_threadgroup]]
) {
    // Max 32 SIMD groups per threadgroup (1024 / 32)
    threadgroup Fr simd_partials[32];

    // Each thread accumulates strided elements
    Fr acc = fr_zero();
    uint base = tgid * tg_size;
    for (uint i = base + tid; i < count; i += tg_size) {
        acc = fr_add(acc, input[i]);
    }

    // Phase 1: SIMD shuffle reduction within each 32-lane group
    acc = simd_reduce_sum_fr(acc, simd_lane);

    // Lane 0 of each SIMD group writes to shared memory
    uint n_simd_groups = (tg_size + 31) / 32;
    if (simd_lane == 0) {
        simd_partials[simd_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: First SIMD group reduces the simd_partials
    if (simd_id == 0) {
        acc = (simd_lane < n_simd_groups) ? simd_partials[simd_lane] : fr_zero();
        acc = simd_reduce_sum_fr(acc, simd_lane);
        if (simd_lane == 0) {
            output[tgid] = acc;
        }
    }
}

// ============================================================================
// BabyBear — Sum reduction
// ============================================================================

inline Bb simd_reduce_sum_bb(Bb val, uint lane) {
    #pragma unroll
    for (uint off = 16; off > 0; off >>= 1) {
        uint other = simd_shuffle_down(val.v, off);
        if (lane < off) {
            val = bb_add(val, Bb{other});
        }
    }
    return val;
}

kernel void reduce_sum_babybear(
    device const Bb* input          [[buffer(0)]],
    device Bb* output               [[buffer(1)]],
    constant uint& count            [[buffer(2)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint tg_size                    [[threads_per_threadgroup]],
    uint simd_lane                  [[thread_index_in_simdgroup]],
    uint simd_id                    [[simdgroup_index_in_threadgroup]]
) {
    threadgroup Bb simd_partials[32];

    Bb acc = bb_zero();
    uint base = tgid * tg_size;
    for (uint i = base + tid; i < count; i += tg_size) {
        acc = bb_add(acc, input[i]);
    }

    acc = simd_reduce_sum_bb(acc, simd_lane);

    uint n_simd_groups = (tg_size + 31) / 32;
    if (simd_lane == 0) {
        simd_partials[simd_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        acc = (simd_lane < n_simd_groups) ? simd_partials[simd_lane] : bb_zero();
        acc = simd_reduce_sum_bb(acc, simd_lane);
        if (simd_lane == 0) {
            output[tgid] = acc;
        }
    }
}

// ============================================================================
// BN254 Fr — Product reduction
// ============================================================================

inline Fr simd_reduce_prod_fr(Fr val, uint lane) {
    #pragma unroll
    for (uint off = 16; off > 0; off >>= 1) {
        Fr other = simd_shuffle_down_fr(val, off);
        if (lane < off) {
            val = fr_mul(val, other);
        }
    }
    return val;
}

kernel void reduce_product_bn254(
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

    Fr acc = fr_one();
    uint base = tgid * tg_size;
    for (uint i = base + tid; i < count; i += tg_size) {
        acc = fr_mul(acc, input[i]);
    }

    acc = simd_reduce_prod_fr(acc, simd_lane);

    uint n_simd_groups = (tg_size + 31) / 32;
    if (simd_lane == 0) {
        simd_partials[simd_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        acc = (simd_lane < n_simd_groups) ? simd_partials[simd_lane] : fr_one();
        acc = simd_reduce_prod_fr(acc, simd_lane);
        if (simd_lane == 0) {
            output[tgid] = acc;
        }
    }
}

// ============================================================================
// uint32 — Min / Max reduction (fused: computes both in one pass)
// ============================================================================

struct MinMaxResult {
    uint min_val;
    uint max_val;
};

inline MinMaxResult simd_reduce_minmax(uint mn, uint mx, uint lane) {
    #pragma unroll
    for (uint off = 16; off > 0; off >>= 1) {
        uint other_mn = simd_shuffle_down_u32(mn, off);
        uint other_mx = simd_shuffle_down_u32(mx, off);
        if (lane < off) {
            mn = min(mn, other_mn);
            mx = max(mx, other_mx);
        }
    }
    return MinMaxResult{mn, mx};
}

kernel void reduce_min_max_u32(
    device const uint* input        [[buffer(0)]],
    device MinMaxResult* output     [[buffer(1)]],
    constant uint& count            [[buffer(2)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint tg_size                    [[threads_per_threadgroup]],
    uint simd_lane                  [[thread_index_in_simdgroup]],
    uint simd_id                    [[simdgroup_index_in_threadgroup]]
) {
    threadgroup uint simd_min[32];
    threadgroup uint simd_max[32];

    uint local_min = 0xFFFFFFFFu;
    uint local_max = 0u;
    uint base = tgid * tg_size;
    for (uint i = base + tid; i < count; i += tg_size) {
        uint val = input[i];
        local_min = min(local_min, val);
        local_max = max(local_max, val);
    }

    MinMaxResult r = simd_reduce_minmax(local_min, local_max, simd_lane);

    uint n_simd_groups = (tg_size + 31) / 32;
    if (simd_lane == 0) {
        simd_min[simd_id] = r.min_val;
        simd_max[simd_id] = r.max_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        uint mn = (simd_lane < n_simd_groups) ? simd_min[simd_lane] : 0xFFFFFFFFu;
        uint mx = (simd_lane < n_simd_groups) ? simd_max[simd_lane] : 0u;
        MinMaxResult final_r = simd_reduce_minmax(mn, mx, simd_lane);
        if (simd_lane == 0) {
            output[tgid] = final_r;
        }
    }
}
