// Parallel prefix sum (scan) kernels for Metal GPU
//
// Algorithms:
//   - Blelloch work-efficient parallel scan: O(n) work, O(log n) depth
//   - Up-sweep (reduce) phase: build partial sums tree
//   - Down-sweep phase: propagate prefix sums
//   - Multi-block: each threadgroup scans a tile, block sums are scanned
//     separately, then propagated back to produce the global scan.
//
// Kernels:
//   - inclusive_scan_u32 / exclusive_scan_u32: uint32 addition
//   - inclusive_scan_u64 / exclusive_scan_u64: uint64 addition
//   - prefix_product_bn254: running product of BN254 Fr field elements
//   - prefix_product_babybear: running product of BabyBear field elements
//   - propagate_block_sum_u32 / propagate_block_sum_u64: add block offsets
//   - propagate_block_product_bn254 / propagate_block_product_babybear

#include "../fields/bn254_fr.metal"
#include "../fields/babybear.metal"

// ============================================================================
// Block size for scan tiles — each threadgroup processes 2*SCAN_BLOCK threads
// ============================================================================

#define SCAN_BLOCK_SIZE 512  // threads per threadgroup; each handles 2 elements

// ============================================================================
// uint32 inclusive scan (Blelloch)
// ============================================================================

kernel void inclusive_scan_u32(
    device const uint* input       [[buffer(0)]],
    device uint* output            [[buffer(1)]],
    device uint* block_sums        [[buffer(2)]],  // one sum per threadgroup
    constant uint& count           [[buffer(3)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    // Each thread loads 2 elements
    uint n = tg_size * 2;  // elements per block
    uint block_offset = tgid * n;

    threadgroup uint shared[SCAN_BLOCK_SIZE * 2];

    // Load input into shared memory
    uint ai = tid;
    uint bi = tid + tg_size;
    uint global_ai = block_offset + ai;
    uint global_bi = block_offset + bi;

    shared[ai] = (global_ai < count) ? input[global_ai] : 0;
    shared[bi] = (global_bi < count) ? input[global_bi] : 0;

    // === Up-sweep (reduce) phase ===
    uint offset = 1;
    for (uint d = n >> 1; d > 0; d >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < d) {
            uint ai_idx = offset * (2 * tid + 1) - 1;
            uint bi_idx = offset * (2 * tid + 2) - 1;
            shared[bi_idx] += shared[ai_idx];
        }
        offset <<= 1;
    }

    // Save block total and clear last element for down-sweep
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        block_sums[tgid] = shared[n - 1];
        shared[n - 1] = 0;
    }

    // === Down-sweep phase ===
    for (uint d = 1; d < n; d <<= 1) {
        offset >>= 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < d) {
            uint ai_idx = offset * (2 * tid + 1) - 1;
            uint bi_idx = offset * (2 * tid + 2) - 1;
            uint temp = shared[ai_idx];
            shared[ai_idx] = shared[bi_idx];
            shared[bi_idx] += temp;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Down-sweep produces exclusive scan; convert to inclusive by shifting
    // inclusive[i] = exclusive[i] + input[i]
    if (global_ai < count) {
        output[global_ai] = shared[ai] + input[global_ai];
    }
    if (global_bi < count) {
        output[global_bi] = shared[bi] + input[global_bi];
    }
}

// ============================================================================
// uint32 exclusive scan (Blelloch)
// ============================================================================

kernel void exclusive_scan_u32(
    device const uint* input       [[buffer(0)]],
    device uint* output            [[buffer(1)]],
    device uint* block_sums        [[buffer(2)]],
    constant uint& count           [[buffer(3)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint n = tg_size * 2;
    uint block_offset = tgid * n;

    threadgroup uint shared[SCAN_BLOCK_SIZE * 2];

    uint ai = tid;
    uint bi = tid + tg_size;
    uint global_ai = block_offset + ai;
    uint global_bi = block_offset + bi;

    shared[ai] = (global_ai < count) ? input[global_ai] : 0;
    shared[bi] = (global_bi < count) ? input[global_bi] : 0;

    // Up-sweep
    uint offset = 1;
    for (uint d = n >> 1; d > 0; d >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < d) {
            uint ai_idx = offset * (2 * tid + 1) - 1;
            uint bi_idx = offset * (2 * tid + 2) - 1;
            shared[bi_idx] += shared[ai_idx];
        }
        offset <<= 1;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        block_sums[tgid] = shared[n - 1];
        shared[n - 1] = 0;
    }

    // Down-sweep
    for (uint d = 1; d < n; d <<= 1) {
        offset >>= 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < d) {
            uint ai_idx = offset * (2 * tid + 1) - 1;
            uint bi_idx = offset * (2 * tid + 2) - 1;
            uint temp = shared[ai_idx];
            shared[ai_idx] = shared[bi_idx];
            shared[bi_idx] += temp;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Exclusive scan: output directly from shared
    if (global_ai < count) { output[global_ai] = shared[ai]; }
    if (global_bi < count) { output[global_bi] = shared[bi]; }
}

// ============================================================================
// Propagate block sums for multi-block scan (uint32)
// ============================================================================

kernel void propagate_block_sum_u32(
    device uint* data              [[buffer(0)]],
    device const uint* block_sums  [[buffer(1)]],
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

    uint prefix = block_sums[tgid];
    if (idx0 < count) data[idx0] += prefix;
    if (idx1 < count) data[idx1] += prefix;
}

// ============================================================================
// uint64 inclusive scan
// ============================================================================

kernel void inclusive_scan_u64(
    device const ulong* input      [[buffer(0)]],
    device ulong* output           [[buffer(1)]],
    device ulong* block_sums       [[buffer(2)]],
    constant uint& count           [[buffer(3)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint n = tg_size * 2;
    uint block_offset = tgid * n;

    threadgroup ulong shared[SCAN_BLOCK_SIZE * 2];

    uint ai = tid;
    uint bi = tid + tg_size;
    uint global_ai = block_offset + ai;
    uint global_bi = block_offset + bi;

    shared[ai] = (global_ai < count) ? input[global_ai] : 0;
    shared[bi] = (global_bi < count) ? input[global_bi] : 0;

    // Up-sweep
    uint offset = 1;
    for (uint d = n >> 1; d > 0; d >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < d) {
            uint ai_idx = offset * (2 * tid + 1) - 1;
            uint bi_idx = offset * (2 * tid + 2) - 1;
            shared[bi_idx] += shared[ai_idx];
        }
        offset <<= 1;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        block_sums[tgid] = shared[n - 1];
        shared[n - 1] = 0;
    }

    // Down-sweep
    for (uint d = 1; d < n; d <<= 1) {
        offset >>= 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < d) {
            uint ai_idx = offset * (2 * tid + 1) - 1;
            uint bi_idx = offset * (2 * tid + 2) - 1;
            ulong temp = shared[ai_idx];
            shared[ai_idx] = shared[bi_idx];
            shared[bi_idx] += temp;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Inclusive: exclusive + input
    if (global_ai < count) {
        output[global_ai] = shared[ai] + input[global_ai];
    }
    if (global_bi < count) {
        output[global_bi] = shared[bi] + input[global_bi];
    }
}

// ============================================================================
// uint64 exclusive scan
// ============================================================================

kernel void exclusive_scan_u64(
    device const uint64_t* input   [[buffer(0)]],
    device uint64_t* output        [[buffer(1)]],
    device uint64_t* block_sums    [[buffer(2)]],
    constant uint& count           [[buffer(3)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint n = tg_size * 2;
    uint block_offset = tgid * n;

    threadgroup ulong shared[SCAN_BLOCK_SIZE * 2];

    uint ai = tid;
    uint bi = tid + tg_size;
    uint global_ai = block_offset + ai;
    uint global_bi = block_offset + bi;

    shared[ai] = (global_ai < count) ? input[global_ai] : 0;
    shared[bi] = (global_bi < count) ? input[global_bi] : 0;

    uint offset = 1;
    for (uint d = n >> 1; d > 0; d >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < d) {
            uint ai_idx = offset * (2 * tid + 1) - 1;
            uint bi_idx = offset * (2 * tid + 2) - 1;
            shared[bi_idx] += shared[ai_idx];
        }
        offset <<= 1;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        block_sums[tgid] = shared[n - 1];
        shared[n - 1] = 0;
    }

    for (uint d = 1; d < n; d <<= 1) {
        offset >>= 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < d) {
            uint ai_idx = offset * (2 * tid + 1) - 1;
            uint bi_idx = offset * (2 * tid + 2) - 1;
            ulong temp = shared[ai_idx];
            shared[ai_idx] = shared[bi_idx];
            shared[bi_idx] += temp;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (global_ai < count) { output[global_ai] = shared[ai]; }
    if (global_bi < count) { output[global_bi] = shared[bi]; }
}

// ============================================================================
// Propagate block sums for uint64
// ============================================================================

kernel void propagate_block_sum_u64(
    device ulong* data             [[buffer(0)]],
    device const ulong* block_sums [[buffer(1)]],
    constant uint& count           [[buffer(2)]],
    uint gid                       [[thread_position_in_grid]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    if (tgid == 0) return;
    uint block_n = tg_size * 2;
    uint idx0 = tgid * block_n + gid % tg_size;
    uint idx1 = idx0 + tg_size;
    ulong prefix = block_sums[tgid];
    if (idx0 < count) data[idx0] += prefix;
    if (idx1 < count) data[idx1] += prefix;
}

// ============================================================================
// BN254 Fr prefix product (sequential within block — product is order-dependent)
//
// Unlike addition-based scan, multiplication prefix scan requires sequential
// ordering. We use a hybrid approach: each threadgroup serially scans its
// tile (which is small enough to be fast), and multi-block is handled by
// multiplying each block by the product of all preceding blocks.
// ============================================================================

kernel void prefix_product_bn254(
    device const Fr* input         [[buffer(0)]],
    device Fr* output              [[buffer(1)]],
    device Fr* block_products      [[buffer(2)]],
    constant uint& count           [[buffer(3)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    // Each threadgroup processes a contiguous tile of 'tg_size' elements.
    // Thread 0 does the sequential prefix product for the entire tile.
    uint tile_size = tg_size;
    uint base = tgid * tile_size;

    if (tid == 0) {
        Fr running = fr_one();
        uint end = min(base + tile_size, count);
        for (uint i = base; i < end; i++) {
            running = fr_mul(running, input[i]);
            output[i] = running;
        }
        // Store block total product
        block_products[tgid] = running;
    }
}

// Propagate block products: multiply each element by inclusive prefix product
// of all preceding blocks. block_prefix[i] = product of blocks 0..i-1 (exclusive).
kernel void propagate_block_product_bn254(
    device Fr* data                [[buffer(0)]],
    device const Fr* block_prefix  [[buffer(1)]],
    constant uint& count           [[buffer(2)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    if (tgid == 0) return;  // Block 0 needs no adjustment

    uint base = tgid * tg_size;
    uint idx = base + tid;
    if (idx < count) {
        Fr prefix = block_prefix[tgid];
        data[idx] = fr_mul(prefix, data[idx]);
    }
}

// ============================================================================
// BabyBear prefix product
// ============================================================================

kernel void prefix_product_babybear(
    device const Bb* input         [[buffer(0)]],
    device Bb* output              [[buffer(1)]],
    device Bb* block_products      [[buffer(2)]],
    constant uint& count           [[buffer(3)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint tile_size = tg_size;
    uint base = tgid * tile_size;

    if (tid == 0) {
        Bb running = bb_one();
        uint end = min(base + tile_size, count);
        for (uint i = base; i < end; i++) {
            running = bb_mul(running, input[i]);
            output[i] = running;
        }
        block_products[tgid] = running;
    }
}

kernel void propagate_block_product_babybear(
    device Bb* data                [[buffer(0)]],
    device const Bb* block_prefix  [[buffer(1)]],
    constant uint& count           [[buffer(2)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    if (tgid == 0) return;

    uint base = tgid * tg_size;
    uint idx = base + tid;
    if (idx < count) {
        Bb prefix = block_prefix[tgid];
        data[idx] = bb_mul(prefix, data[idx]);
    }
}
