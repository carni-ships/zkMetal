// Parallel prefix sum (scan) kernels for BN254 Fr field elements
//
// Algorithms:
//   - Blelloch work-efficient parallel scan: O(n) work, O(log n) depth
//   - Up-sweep (reduce) phase: build partial sums tree in threadgroup memory
//   - Down-sweep phase: propagate prefix sums back down the tree
//   - Multi-block: each threadgroup scans a tile, block sums are scanned
//     separately, then propagated back to produce the global scan.
//
// Kernels:
//   - inclusive_scan_fr: BN254 Fr field addition inclusive scan
//   - exclusive_scan_fr: BN254 Fr field addition exclusive scan
//   - segmented_scan_fr: BN254 Fr field addition with segment boundaries
//   - propagate_block_sum_fr: add Fr block offsets for multi-block scan
//   - propagate_block_sum_segmented_fr: conditional propagation respecting segments

#include "../fields/bn254_fr.metal"

// Block size for scan tiles — each threadgroup processes 2*FRSCAN_BLOCK threads
#define FRSCAN_BLOCK_SIZE 256  // threads per threadgroup; each handles 2 Fr elements

// ============================================================================
// BN254 Fr inclusive scan (Blelloch up-sweep + down-sweep)
// ============================================================================

kernel void inclusive_scan_fr(
    device const Fr* input         [[buffer(0)]],
    device Fr* output              [[buffer(1)]],
    device Fr* block_sums          [[buffer(2)]],  // one sum per threadgroup
    constant uint& count           [[buffer(3)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    // Each thread loads 2 elements
    uint n = tg_size * 2;  // elements per block
    uint block_offset = tgid * n;

    threadgroup Fr shared_mem[FRSCAN_BLOCK_SIZE * 2];

    // Load input into shared memory
    uint ai = tid;
    uint bi = tid + tg_size;
    uint global_ai = block_offset + ai;
    uint global_bi = block_offset + bi;

    shared_mem[ai] = (global_ai < count) ? input[global_ai] : fr_zero();
    shared_mem[bi] = (global_bi < count) ? input[global_bi] : fr_zero();

    // === Up-sweep (reduce) phase ===
    uint offset = 1;
    for (uint d = n >> 1; d > 0; d >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < d) {
            uint ai_idx = offset * (2 * tid + 1) - 1;
            uint bi_idx = offset * (2 * tid + 2) - 1;
            shared_mem[bi_idx] = fr_add(shared_mem[bi_idx], shared_mem[ai_idx]);
        }
        offset <<= 1;
    }

    // Save block total and clear last element for down-sweep
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        block_sums[tgid] = shared_mem[n - 1];
        shared_mem[n - 1] = fr_zero();
    }

    // === Down-sweep phase ===
    for (uint d = 1; d < n; d <<= 1) {
        offset >>= 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < d) {
            uint ai_idx = offset * (2 * tid + 1) - 1;
            uint bi_idx = offset * (2 * tid + 2) - 1;
            Fr temp = shared_mem[ai_idx];
            shared_mem[ai_idx] = shared_mem[bi_idx];
            shared_mem[bi_idx] = fr_add(shared_mem[bi_idx], temp);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Down-sweep produces exclusive scan; convert to inclusive by adding input
    // inclusive[i] = exclusive[i] + input[i]
    if (global_ai < count) {
        output[global_ai] = fr_add(shared_mem[ai], input[global_ai]);
    }
    if (global_bi < count) {
        output[global_bi] = fr_add(shared_mem[bi], input[global_bi]);
    }
}

// ============================================================================
// BN254 Fr exclusive scan (Blelloch)
// ============================================================================

kernel void exclusive_scan_fr(
    device const Fr* input         [[buffer(0)]],
    device Fr* output              [[buffer(1)]],
    device Fr* block_sums          [[buffer(2)]],
    constant uint& count           [[buffer(3)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint n = tg_size * 2;
    uint block_offset = tgid * n;

    threadgroup Fr shared_mem[FRSCAN_BLOCK_SIZE * 2];

    uint ai = tid;
    uint bi = tid + tg_size;
    uint global_ai = block_offset + ai;
    uint global_bi = block_offset + bi;

    shared_mem[ai] = (global_ai < count) ? input[global_ai] : fr_zero();
    shared_mem[bi] = (global_bi < count) ? input[global_bi] : fr_zero();

    // Up-sweep
    uint offset = 1;
    for (uint d = n >> 1; d > 0; d >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < d) {
            uint ai_idx = offset * (2 * tid + 1) - 1;
            uint bi_idx = offset * (2 * tid + 2) - 1;
            shared_mem[bi_idx] = fr_add(shared_mem[bi_idx], shared_mem[ai_idx]);
        }
        offset <<= 1;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        block_sums[tgid] = shared_mem[n - 1];
        shared_mem[n - 1] = fr_zero();
    }

    // Down-sweep
    for (uint d = 1; d < n; d <<= 1) {
        offset >>= 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < d) {
            uint ai_idx = offset * (2 * tid + 1) - 1;
            uint bi_idx = offset * (2 * tid + 2) - 1;
            Fr temp = shared_mem[ai_idx];
            shared_mem[ai_idx] = shared_mem[bi_idx];
            shared_mem[bi_idx] = fr_add(shared_mem[bi_idx], temp);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Exclusive scan: output directly from shared
    if (global_ai < count) { output[global_ai] = shared_mem[ai]; }
    if (global_bi < count) { output[global_bi] = shared_mem[bi]; }
}

// ============================================================================
// Propagate block sums for multi-block Fr scan
// ============================================================================

kernel void propagate_block_sum_fr(
    device Fr* data                [[buffer(0)]],
    device const Fr* block_sums    [[buffer(1)]],
    constant uint& count           [[buffer(2)]],
    uint gid                       [[thread_position_in_grid]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    // Block 0 needs no adjustment; blocks 1..N-1 add their prefix
    if (tgid == 0) return;

    uint block_n = tg_size * 2;  // elements per scan block
    uint idx0 = tgid * block_n + gid % tg_size;
    uint idx1 = idx0 + tg_size;

    Fr prefix = block_sums[tgid];
    if (idx0 < count) data[idx0] = fr_add(data[idx0], prefix);
    if (idx1 < count) data[idx1] = fr_add(data[idx1], prefix);
}

// ============================================================================
// BN254 Fr segmented inclusive scan
//
// Segments are defined by a flags buffer: flags[i] = 1 starts a new segment.
// The scan resets at each segment boundary.
// Uses Blelloch scan with segment flag propagation.
// ============================================================================

kernel void segmented_scan_fr(
    device const Fr* input         [[buffer(0)]],
    device Fr* output              [[buffer(1)]],
    device const uint* flags       [[buffer(2)]],  // 1 = segment start
    device Fr* block_sums          [[buffer(3)]],
    device uint* block_flags       [[buffer(4)]],  // 1 if block contains a segment start
    constant uint& count           [[buffer(5)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint n = tg_size * 2;
    uint block_offset = tgid * n;

    threadgroup Fr shared_val[FRSCAN_BLOCK_SIZE * 2];
    threadgroup uint shared_flag[FRSCAN_BLOCK_SIZE * 2];

    // Load input into shared memory
    uint ai = tid;
    uint bi = tid + tg_size;
    uint global_ai = block_offset + ai;
    uint global_bi = block_offset + bi;

    shared_val[ai] = (global_ai < count) ? input[global_ai] : fr_zero();
    shared_val[bi] = (global_bi < count) ? input[global_bi] : fr_zero();
    shared_flag[ai] = (global_ai < count) ? flags[global_ai] : 0;
    shared_flag[bi] = (global_bi < count) ? flags[global_bi] : 0;

    // === Up-sweep with segment flag propagation ===
    uint offset = 1;
    for (uint d = n >> 1; d > 0; d >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < d) {
            uint ai_idx = offset * (2 * tid + 1) - 1;
            uint bi_idx = offset * (2 * tid + 2) - 1;
            if (shared_flag[bi_idx]) {
                // Segment boundary in right (bi) range: bi already holds
                // the sum from that boundary to end. Keep val, propagate flag.
            } else {
                // No boundary in right range: add left (ai) contribution
                shared_val[bi_idx] = fr_add(shared_val[bi_idx], shared_val[ai_idx]);
                // Propagate flag from left range if present
                shared_flag[bi_idx] = shared_flag[ai_idx];
            }
        }
        offset <<= 1;
    }

    // Save block total and flag, clear last for down-sweep
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        block_sums[tgid] = shared_val[n - 1];
        block_flags[tgid] = shared_flag[n - 1];
        shared_val[n - 1] = fr_zero();
        shared_flag[n - 1] = 0;
    }

    // === Down-sweep with segment awareness ===
    for (uint d = 1; d < n; d <<= 1) {
        offset >>= 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < d) {
            uint ai_idx = offset * (2 * tid + 1) - 1;
            uint bi_idx = offset * (2 * tid + 2) - 1;

            Fr temp_val = shared_val[ai_idx];
            uint temp_flag = shared_flag[ai_idx];

            // Left child gets incoming prefix (stored in bi)
            shared_val[ai_idx] = shared_val[bi_idx];
            shared_flag[ai_idx] = shared_flag[bi_idx];

            // Right child gets: prefix ⊕ left_sum
            // (v_prefix, f_prefix) ⊕ (v_left, f_left)
            //   = (f_left ? v_left : v_prefix + v_left, f_prefix | f_left)
            if (temp_flag) {
                shared_val[bi_idx] = temp_val;
            } else {
                shared_val[bi_idx] = fr_add(shared_val[bi_idx], temp_val);
            }
            shared_flag[bi_idx] = shared_flag[bi_idx] | temp_flag;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Convert to inclusive: at segment boundaries, reset prefix to zero;
    // otherwise add the exclusive prefix to the input value.
    if (global_ai < count) {
        Fr prefix_ai = flags[global_ai] ? fr_zero() : shared_val[ai];
        output[global_ai] = fr_add(prefix_ai, input[global_ai]);
    }
    if (global_bi < count) {
        Fr prefix_bi = flags[global_bi] ? fr_zero() : shared_val[bi];
        output[global_bi] = fr_add(prefix_bi, input[global_bi]);
    }
}

// ============================================================================
// Propagate block sums for segmented scan (conditional on block flags)
// ============================================================================

kernel void propagate_block_sum_segmented_fr(
    device Fr* data                [[buffer(0)]],
    device const Fr* block_sums    [[buffer(1)]],
    device const uint* flags       [[buffer(2)]],  // original segment flags
    constant uint& count           [[buffer(3)]],
    uint gid                       [[thread_position_in_grid]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    if (tgid == 0) return;

    uint block_n = tg_size * 2;
    uint idx0 = tgid * block_n + gid % tg_size;
    uint idx1 = idx0 + tg_size;

    Fr prefix = block_sums[tgid];

    // Check if there's a segment start in this block before our position
    // If so, don't propagate (the segment resets the sum)
    // Simple approach: scan forward from block start to find first flag
    uint block_start = tgid * block_n;

    // For idx0: check if any flag between block_start and idx0 (inclusive)
    if (idx0 < count) {
        bool has_flag = false;
        for (uint i = block_start; i <= idx0 && i < count; i++) {
            if (flags[i]) { has_flag = true; break; }
        }
        if (!has_flag) {
            data[idx0] = fr_add(data[idx0], prefix);
        }
    }
    if (idx1 < count) {
        bool has_flag = false;
        for (uint i = block_start; i <= idx1 && i < count; i++) {
            if (flags[i]) { has_flag = true; break; }
        }
        if (!has_flag) {
            data[idx1] = fr_add(data[idx1], prefix);
        }
    }
}
