// General-purpose parallel reduction kernels for BN254 Fr field elements
//
// Kernels:
//   fr_reduce_sum     — parallel sum reduction with shared memory
//   fr_reduce_product — parallel product reduction
//   generic_reduce    — configurable reduction op via function constants
//
// Architecture:
//   - Each threadgroup reduces a tile of input to one partial result.
//   - SIMD shuffle reduces within each SIMD group (32 lanes).
//   - Shared memory tree reduces across SIMD groups.
//   - Multi-pass: host dispatches pass 1 over input, pass 2 over partials, etc.
//   - Handles non-power-of-2 sizes: out-of-bounds threads use the identity element.

#include "../fields/bn254_fr.metal"

// ============================================================================
// SIMD shuffle helper for Fr (8x uint32)
// ============================================================================

inline Fr fr_simd_shuffle_down(Fr a, uint offset) {
    Fr r;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        r.v[k] = simd_shuffle_down(a.v[k], offset);
    }
    return r;
}

// ============================================================================
// fr_reduce_sum — parallel sum reduction
// ============================================================================

inline Fr fr_simd_reduce_sum(Fr val, uint lane) {
    #pragma unroll
    for (uint off = 16; off > 0; off >>= 1) {
        Fr other = fr_simd_shuffle_down(val, off);
        if (lane < off) {
            val = fr_add(val, other);
        }
    }
    return val;
}

kernel void fr_reduce_sum(
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

    // Each thread loads and accumulates strided elements for coalesced access
    Fr acc = fr_zero();
    uint global_id = tgid * tg_size + tid;
    uint grid_stride = tg_size * ((count + tg_size - 1) / tg_size);
    // Only stride within this threadgroup's tile for multi-pass correctness
    uint base = tgid * tg_size;
    for (uint i = base + tid; i < count; i += tg_size) {
        acc = fr_add(acc, input[i]);
    }

    // Phase 1: SIMD shuffle reduction within each 32-lane group
    acc = fr_simd_reduce_sum(acc, simd_lane);

    // Lane 0 of each SIMD group writes to shared memory
    uint n_simd_groups = (tg_size + 31) / 32;
    if (simd_lane == 0) {
        simd_partials[simd_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: First SIMD group reduces the simd_partials
    if (simd_id == 0) {
        acc = (simd_lane < n_simd_groups) ? simd_partials[simd_lane] : fr_zero();
        acc = fr_simd_reduce_sum(acc, simd_lane);
        if (simd_lane == 0) {
            output[tgid] = acc;
        }
    }
}

// ============================================================================
// fr_reduce_product — parallel product reduction
// ============================================================================

inline Fr fr_simd_reduce_prod(Fr val, uint lane) {
    #pragma unroll
    for (uint off = 16; off > 0; off >>= 1) {
        Fr other = fr_simd_shuffle_down(val, off);
        if (lane < off) {
            val = fr_mul(val, other);
        }
    }
    return val;
}

kernel void fr_reduce_product(
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

    acc = fr_simd_reduce_prod(acc, simd_lane);

    uint n_simd_groups = (tg_size + 31) / 32;
    if (simd_lane == 0) {
        simd_partials[simd_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        acc = (simd_lane < n_simd_groups) ? simd_partials[simd_lane] : fr_one();
        acc = fr_simd_reduce_prod(acc, simd_lane);
        if (simd_lane == 0) {
            output[tgid] = acc;
        }
    }
}

// ============================================================================
// generic_reduce — configurable reduction op via function constants
//
// Function constant REDUCE_OP selects the operation:
//   0 = sum (fr_add, identity = 0)
//   1 = product (fr_mul, identity = 1)
//   2 = min (lexicographic, identity = max)
//   3 = max (lexicographic, identity = 0)
// ============================================================================

constant int REDUCE_OP [[function_constant(0)]];

inline Fr fr_max_val() {
    // All-ones value (close to p, serves as a sentinel for min)
    Fr r;
    #pragma unroll
    for (int i = 0; i < 8; i++) r.v[i] = 0xFFFFFFFFu;
    return r;
}

// Lexicographic less-than (big-endian comparison of limbs)
inline bool fr_lt(Fr a, Fr b) {
    for (int i = 7; i >= 0; i--) {
        if (a.v[i] < b.v[i]) return true;
        if (a.v[i] > b.v[i]) return false;
    }
    return false; // equal
}

inline Fr generic_identity() {
    if (REDUCE_OP == 0) return fr_zero();
    if (REDUCE_OP == 1) return fr_one();
    if (REDUCE_OP == 2) return fr_max_val(); // min identity
    return fr_zero(); // max identity
}

inline Fr generic_combine(Fr a, Fr b) {
    if (REDUCE_OP == 0) return fr_add(a, b);
    if (REDUCE_OP == 1) return fr_mul(a, b);
    if (REDUCE_OP == 2) return fr_lt(a, b) ? a : b; // min
    return fr_lt(a, b) ? b : a; // max
}

inline Fr fr_simd_reduce_generic(Fr val, uint lane) {
    #pragma unroll
    for (uint off = 16; off > 0; off >>= 1) {
        Fr other = fr_simd_shuffle_down(val, off);
        if (lane < off) {
            val = generic_combine(val, other);
        }
    }
    return val;
}

kernel void generic_reduce(
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

    Fr acc = generic_identity();
    uint base = tgid * tg_size;
    for (uint i = base + tid; i < count; i += tg_size) {
        acc = generic_combine(acc, input[i]);
    }

    acc = fr_simd_reduce_generic(acc, simd_lane);

    uint n_simd_groups = (tg_size + 31) / 32;
    if (simd_lane == 0) {
        simd_partials[simd_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        acc = (simd_lane < n_simd_groups) ? simd_partials[simd_lane] : generic_identity();
        acc = fr_simd_reduce_generic(acc, simd_lane);
        if (simd_lane == 0) {
            output[tgid] = acc;
        }
    }
}
