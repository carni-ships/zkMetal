// GPU Radix Sort for Metal
// 4-pass radix sort on 32-bit keys with optional 32-bit values
// Radix = 256 (8-bit digits), 4 passes for full 32-bit sort
// Algorithm: per-block histogram → prefix sum → scatter

#include <metal_stdlib>
using namespace metal;

constant uint RADIX_BITS = 8;
constant uint RADIX = 256;      // 2^8

// Phase 1: Compute per-block histograms
// Each threadgroup processes a contiguous block of elements
// Output: histograms[block_idx * RADIX + digit] = count
kernel void radix_histogram(
    device const uint* keys        [[buffer(0)]],
    device uint* histograms        [[buffer(1)]],
    constant uint& count           [[buffer(2)]],
    constant uint& shift           [[buffer(3)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    threadgroup uint local_hist[256];

    // Clear local histogram
    if (tid < RADIX) local_hist[tid] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread processes multiple elements
    uint block_start = tgid * tg_size * 4;  // 4 elements per thread
    for (uint i = 0; i < 4; i++) {
        uint idx = block_start + tid + i * tg_size;
        if (idx < count) {
            uint digit = (keys[idx] >> shift) & 0xFF;
            // Atomic increment in threadgroup memory
            atomic_fetch_add_explicit(
                (threadgroup atomic_uint*)&local_hist[digit],
                1u, memory_order_relaxed);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write local histogram to global memory
    if (tid < RADIX) {
        uint num_blocks = (count + tg_size * 4 - 1) / (tg_size * 4);
        histograms[tid * num_blocks + tgid] = local_hist[tid];
    }
}

// Phase 2: Prefix sum (exclusive scan) on histograms
// Input layout: histograms[digit * num_blocks + block_idx]
// After scan: histograms[digit * num_blocks + block_idx] = global offset
// This is a Blelloch-style scan within each threadgroup, then cross-group fixup
kernel void radix_prefix_sum(
    device uint* histograms        [[buffer(0)]],
    device uint* block_sums        [[buffer(1)]],
    constant uint& total_entries   [[buffer(2)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    // Each threadgroup scans a contiguous chunk of the histogram array
    threadgroup uint shared[2048];  // Max 1024 threads * 2

    uint block_size = tg_size * 2;
    uint base = tgid * block_size;

    // Load into shared memory
    shared[tid] = (base + tid < total_entries) ? histograms[base + tid] : 0;
    shared[tid + tg_size] = (base + tid + tg_size < total_entries) ? histograms[base + tid + tg_size] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Up-sweep (reduce)
    uint offset = 1;
    for (uint d = block_size >> 1; d > 0; d >>= 1) {
        if (tid < d) {
            uint ai = offset * (2 * tid + 1) - 1;
            uint bi = offset * (2 * tid + 2) - 1;
            shared[bi] += shared[ai];
        }
        offset <<= 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Save block sum and clear last element
    if (tid == 0) {
        block_sums[tgid] = shared[block_size - 1];
        shared[block_size - 1] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Down-sweep
    for (uint d = 1; d < block_size; d <<= 1) {
        offset >>= 1;
        if (tid < d) {
            uint ai = offset * (2 * tid + 1) - 1;
            uint bi = offset * (2 * tid + 2) - 1;
            uint t = shared[ai];
            shared[ai] = shared[bi];
            shared[bi] += t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write back
    if (base + tid < total_entries) histograms[base + tid] = shared[tid];
    if (base + tid + tg_size < total_entries) histograms[base + tid + tg_size] = shared[tid + tg_size];
}

// Phase 2b: Add block sums back to scanned histograms
kernel void radix_add_block_sums(
    device uint* histograms        [[buffer(0)]],
    device const uint* block_sums  [[buffer(1)]],
    constant uint& total_entries   [[buffer(2)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    if (tgid == 0) return;  // First block needs no adjustment
    uint add_val = block_sums[tgid];
    uint block_size = tg_size * 2;
    uint base = tgid * block_size;

    uint idx0 = base + tid;
    uint idx1 = base + tid + tg_size;
    if (idx0 < total_entries) histograms[idx0] += add_val;
    if (idx1 < total_entries) histograms[idx1] += add_val;
}

// Phase 3: Parallel stable scatter using shared-memory ranking.
// Each thread owns one digit value (tid == digit since tg_size == 256 == RADIX).
// Thread scans all block elements in shared memory, assigns ranks for its digit,
// then all threads cooperatively write to global memory.
// Stability: elements are scanned in input order, so ranks preserve order.

// Shared memory layout per block (1024 elements max):
//   shared_keys[1024]  = 4KB   (loaded keys)
//   shared_vals[1024]  = 4KB   (loaded values)
//   shared_digits[1024] = 1KB  (extracted digits, uint8)
//   shared_pos[1024]   = 4KB   (computed output positions)
// Total KV: 13KB.  Keys-only: 9KB.

// Key-value scatter with segmented ranking (same optimization as keys-only)
kernel void radix_scatter(
    device const uint* keys_in     [[buffer(0)]],
    device const uint* vals_in     [[buffer(1)]],
    device uint* keys_out          [[buffer(2)]],
    device uint* vals_out          [[buffer(3)]],
    device uint* histograms        [[buffer(4)]],
    constant uint& count           [[buffer(5)]],
    constant uint& shift           [[buffer(6)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    const uint ELEMS_PER_THREAD = 4;
    const uint ELEMS_PER_GROUP = 1024;
    const uint SEG_BITS = 6;
    const uint NUM_SEGS = 16;

    threadgroup uint shared_keys[1024];
    threadgroup uint shared_vals[1024];
    threadgroup uint8_t shared_digits[1024];
    threadgroup uint seg_count[256 * 16];
    threadgroup uint shared_pos[1024];

    uint num_blocks = (count + ELEMS_PER_GROUP - 1) / ELEMS_PER_GROUP;
    uint block_start = tgid * ELEMS_PER_GROUP;
    uint block_count = min(ELEMS_PER_GROUP, count - block_start);

    // Phase 1: Cooperative load
    for (uint i = 0; i < ELEMS_PER_THREAD; i++) {
        uint pos = tid + i * tg_size;
        if (pos < block_count) {
            uint gidx = block_start + pos;
            uint key = keys_in[gidx];
            shared_keys[pos] = key;
            shared_vals[pos] = vals_in[gidx];
            shared_digits[pos] = uint8_t((key >> shift) & 0xFF);
        }
    }

    for (uint i = 0; i < NUM_SEGS; i++) {
        seg_count[tid * NUM_SEGS + i] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Build segment counts
    for (uint i = 0; i < ELEMS_PER_THREAD; i++) {
        uint pos = tid + i * tg_size;
        if (pos < block_count) {
            uint d = shared_digits[pos];
            uint seg = pos >> SEG_BITS;
            atomic_fetch_add_explicit(
                (threadgroup atomic_uint*)&seg_count[d * NUM_SEGS + seg],
                1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Prefix sum per digit
    {
        uint base = tid * NUM_SEGS;
        uint running = 0;
        for (uint s = 0; s < NUM_SEGS; s++) {
            uint c = seg_count[base + s];
            seg_count[base + s] = running;
            running += c;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: Rank elements
    uint my_base = tid * ELEMS_PER_THREAD;
    if (my_base < block_count) {
        uint my_elem_count = min(ELEMS_PER_THREAD, block_count - my_base);
        uint my_seg = my_base >> SEG_BITS;
        uint my_seg_end = (my_base + my_elem_count - 1) >> SEG_BITS;

        if (my_seg == my_seg_end) {
            uint seg_start = my_seg << SEG_BITS;
            uint local_rank[4] = {0, 0, 0, 0};
            uint8_t my_d[4];
            for (uint i = 0; i < my_elem_count; i++) {
                my_d[i] = shared_digits[my_base + i];
            }
            for (uint j = seg_start; j < my_base; j++) {
                uint8_t dj = shared_digits[j];
                for (uint i = 0; i < my_elem_count; i++) {
                    if (dj == my_d[i]) local_rank[i]++;
                }
            }
            for (uint i = 0; i < my_elem_count; i++) {
                uint pos = my_base + i;
                uint d = my_d[i];
                uint global_offset = histograms[d * num_blocks + tgid];
                shared_pos[pos] = global_offset + seg_count[d * NUM_SEGS + my_seg] + local_rank[i];
                for (uint k = i + 1; k < my_elem_count; k++) {
                    if (my_d[k] == d) local_rank[k]++;
                }
            }
        } else {
            for (uint i = 0; i < my_elem_count; i++) {
                uint pos = my_base + i;
                uint d = shared_digits[pos];
                uint seg = pos >> SEG_BITS;
                uint seg_start = seg << SEG_BITS;
                uint lr = 0;
                for (uint j = seg_start; j < pos; j++) {
                    if (shared_digits[j] == d) lr++;
                }
                uint global_offset = histograms[d * num_blocks + tgid];
                shared_pos[pos] = global_offset + seg_count[d * NUM_SEGS + seg] + lr;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 5: Cooperative scatter
    for (uint i = 0; i < ELEMS_PER_THREAD; i++) {
        uint pos = tid + i * tg_size;
        if (pos < block_count) {
            uint out_pos = shared_pos[pos];
            keys_out[out_pos] = shared_keys[pos];
            vals_out[out_pos] = shared_vals[pos];
        }
    }
}

// Parallel scatter keys only — segmented ranking approach.
// Instead of 256 threads each scanning all 1024 elements (262K ops),
// divide into 16 segments of 64 elements. Build per-digit per-segment counts,
// prefix sum, then each thread only scans its segment (~8K ops total).
kernel void radix_scatter_keys_only(
    device const uint* keys_in     [[buffer(0)]],
    device uint* keys_out          [[buffer(1)]],
    device uint* histograms        [[buffer(2)]],
    constant uint& count           [[buffer(3)]],
    constant uint& shift           [[buffer(4)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    const uint ELEMS_PER_THREAD = 4;
    const uint ELEMS_PER_GROUP = 1024;
    const uint SEG_BITS = 6;
    const uint SEG_SIZE = 64;
    const uint NUM_SEGS = 16;

    threadgroup uint shared_keys[1024];
    threadgroup uint8_t shared_digits[1024];
    // Per-digit per-segment counts, then prefix sums. Layout: [digit * NUM_SEGS + seg]
    threadgroup uint seg_count[256 * 16];  // 16KB
    threadgroup uint shared_pos[1024];

    uint num_blocks = (count + ELEMS_PER_GROUP - 1) / ELEMS_PER_GROUP;
    uint block_start = tgid * ELEMS_PER_GROUP;
    uint block_count = min(ELEMS_PER_GROUP, count - block_start);

    // Phase 1: Cooperative load (interleaved for coalesced global access)
    for (uint i = 0; i < ELEMS_PER_THREAD; i++) {
        uint pos = tid + i * tg_size;
        if (pos < block_count) {
            uint key = keys_in[block_start + pos];
            shared_keys[pos] = key;
            shared_digits[pos] = uint8_t((key >> shift) & 0xFF);
        }
    }

    // Zero seg_count: 4096 entries / 256 threads = 16 per thread
    for (uint i = 0; i < NUM_SEGS; i++) {
        seg_count[tid * NUM_SEGS + i] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Build per-segment per-digit counts
    for (uint i = 0; i < ELEMS_PER_THREAD; i++) {
        uint pos = tid + i * tg_size;
        if (pos < block_count) {
            uint d = shared_digits[pos];
            uint seg = pos >> SEG_BITS;
            atomic_fetch_add_explicit(
                (threadgroup atomic_uint*)&seg_count[d * NUM_SEGS + seg],
                1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Exclusive prefix sum per digit across segments
    // Each thread (tid == digit) processes its 16 segment counts
    {
        uint base = tid * NUM_SEGS;
        uint running = 0;
        for (uint s = 0; s < NUM_SEGS; s++) {
            uint c = seg_count[base + s];
            seg_count[base + s] = running;
            running += c;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: Compute output positions — each thread ranks its 4 contiguous elements
    uint my_base = tid * ELEMS_PER_THREAD;
    if (my_base < block_count) {
        uint my_elem_count = min(ELEMS_PER_THREAD, block_count - my_base);
        uint my_seg = my_base >> SEG_BITS;
        uint my_seg_end = (my_base + my_elem_count - 1) >> SEG_BITS;

        if (my_seg == my_seg_end) {
            // Fast path: all elements in same segment (common case: 240/256 threads)
            uint seg_start = my_seg << SEG_BITS;
            uint local_rank[4] = {0, 0, 0, 0};
            uint8_t my_d[4];
            for (uint i = 0; i < my_elem_count; i++) {
                my_d[i] = shared_digits[my_base + i];
            }

            // Scan from segment start to my_base, counting matches
            for (uint j = seg_start; j < my_base; j++) {
                uint8_t dj = shared_digits[j];
                for (uint i = 0; i < my_elem_count; i++) {
                    if (dj == my_d[i]) local_rank[i]++;
                }
            }

            // Assign positions
            for (uint i = 0; i < my_elem_count; i++) {
                uint pos = my_base + i;
                uint d = my_d[i];
                uint global_offset = histograms[d * num_blocks + tgid];
                shared_pos[pos] = global_offset + seg_count[d * NUM_SEGS + my_seg] + local_rank[i];
                // Update ranks for subsequent elements with same digit
                for (uint k = i + 1; k < my_elem_count; k++) {
                    if (my_d[k] == d) local_rank[k]++;
                }
            }
        } else {
            // Slow path: elements cross segment boundary (rare: ~16/256 threads)
            for (uint i = 0; i < my_elem_count; i++) {
                uint pos = my_base + i;
                uint d = shared_digits[pos];
                uint seg = pos >> SEG_BITS;
                uint seg_start = seg << SEG_BITS;
                uint lr = 0;
                for (uint j = seg_start; j < pos; j++) {
                    if (shared_digits[j] == d) lr++;
                }
                uint global_offset = histograms[d * num_blocks + tgid];
                shared_pos[pos] = global_offset + seg_count[d * NUM_SEGS + seg] + lr;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 5: Cooperative scatter (interleaved for coalesced writes)
    for (uint i = 0; i < ELEMS_PER_THREAD; i++) {
        uint pos = tid + i * tg_size;
        if (pos < block_count) {
            keys_out[shared_pos[pos]] = shared_keys[pos];
        }
    }
}
