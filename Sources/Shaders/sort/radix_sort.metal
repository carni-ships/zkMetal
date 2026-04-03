// GPU Radix Sort — 32-bit keys, 4 passes × 8-bit radix (256 buckets)
//
// Three-kernel pipeline per pass:
//   1. histogram: per-threadgroup local histograms of 8-bit digit
//   2. prefix_sum: exclusive prefix sum across all threadgroup histograms
//   3. scatter: SIMD-level parallel ranking + coalesced global scatter
//
// Scatter uses SIMD shuffle for intra-SIMD ranking, then cross-SIMD prefix
// sums to compute global destinations. All threads participate in scatter.

#include <metal_stdlib>
using namespace metal;

#define RADIX_BITS 8
#define RADIX_SIZE 256  // 2^RADIX_BITS
#define TILE_SIZE 2048  // keys per threadgroup

// --- Kernel 1: Per-threadgroup histogram ---
// Each threadgroup processes TILE_SIZE keys and writes a 256-bin histogram.
// Output layout: histograms[tgid * 256 + digit] = count of digit in this tile
kernel void radix_histogram(
    device const uint* keys       [[buffer(0)]],
    device uint* histograms       [[buffer(1)]],
    constant uint& n              [[buffer(2)]],
    constant uint& shift          [[buffer(3)]],  // bit shift for current digit (0, 8, 16, 24)
    uint tid                      [[thread_index_in_threadgroup]],
    uint tgid                     [[threadgroup_position_in_grid]],
    uint tg_size                  [[threads_per_threadgroup]]
) {
    threadgroup uint local_hist[RADIX_SIZE];

    // Clear local histogram
    for (uint i = tid; i < RADIX_SIZE; i += tg_size) {
        local_hist[i] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Count digits in this tile
    uint base = tgid * TILE_SIZE;
    for (uint i = tid; i < TILE_SIZE && (base + i) < n; i += tg_size) {
        uint key = keys[base + i];
        uint digit = (key >> shift) & 0xFF;
        atomic_fetch_add_explicit((threadgroup atomic_uint*)&local_hist[digit],
                                 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write local histogram to global memory
    for (uint i = tid; i < RADIX_SIZE; i += tg_size) {
        histograms[tgid * RADIX_SIZE + i] = local_hist[i];
    }
}

// --- Kernel 2: Global prefix sum over histograms ---
// Computes exclusive prefix sum across all threadgroup histograms for each digit.
// Input: histograms[tgid * 256 + digit]
// Output: offsets[tgid * 256 + digit] = global write position for this tile's first key with this digit
//
// This kernel processes one digit at a time across all threadgroups.
// Single threadgroup launch — process all digits sequentially (simple, sufficient for our sizes).
kernel void radix_prefix_sum(
    device const uint* histograms  [[buffer(0)]],
    device uint* offsets           [[buffer(1)]],
    constant uint& num_tiles       [[buffer(2)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    threadgroup uint digit_totals[RADIX_SIZE];
    threadgroup uint digit_prefix[RADIX_SIZE];

    // Pass 1: Compute per-tile prefix sums AND digit totals in a single pass
    for (uint digit = tid; digit < RADIX_SIZE; digit += tg_size) {
        uint running = 0;
        for (uint t = 0; t < num_tiles; t++) {
            uint count = histograms[t * RADIX_SIZE + digit];
            offsets[t * RADIX_SIZE + digit] = running;
            running += count;
        }
        digit_totals[digit] = running;  // total = final running sum
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 computes exclusive prefix sum of digit totals
    if (tid == 0) {
        uint sum = 0;
        for (uint d = 0; d < RADIX_SIZE; d++) {
            digit_prefix[d] = sum;
            sum += digit_totals[d];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: Add per-digit base offset to all tile offsets
    for (uint digit = tid; digit < RADIX_SIZE; digit += tg_size) {
        uint base = digit_prefix[digit];
        for (uint t = 0; t < num_tiles; t++) {
            offsets[t * RADIX_SIZE + digit] += base;
        }
    }
}

// --- Kernel 3: Stable scatter with parallel sub-tile ranking ---
// Process tile in sub-tiles of tg_size keys. Within each sub-tile:
//   1. SIMD shuffle ranks each key within its SIMD
//   2. Per-SIMD counts are prefix-summed across SIMDs
//   3. All threads scatter in parallel
kernel void radix_scatter(
    device const uint* keys_in    [[buffer(0)]],
    device uint* keys_out         [[buffer(1)]],
    device const uint* offsets    [[buffer(2)]],
    constant uint& n              [[buffer(3)]],
    constant uint& shift          [[buffer(4)]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tgid                     [[threadgroup_position_in_grid]],
    uint tg_size                  [[threads_per_threadgroup]],
    uint simd_lane                [[thread_index_in_simdgroup]],
    uint simd_id                  [[simdgroup_index_in_threadgroup]]
) {
    threadgroup uint running_count[RADIX_SIZE];
    threadgroup uint sub_count[8][RADIX_SIZE];  // per-SIMD digit counts (8 SIMDs max)

    uint base = tgid * TILE_SIZE;
    uint limit = min(uint(TILE_SIZE), n - base);
    uint num_simds = tg_size / 32;

    // Initialize running counts with global offsets
    for (uint i = tid; i < RADIX_SIZE; i += tg_size) {
        running_count[i] = offsets[tgid * RADIX_SIZE + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process sub-tiles of tg_size keys each
    for (uint sub = 0; sub < limit; sub += tg_size) {
        uint idx = sub + tid;
        uint key = 0;
        uint digit = 0xFFFF;  // sentinel for inactive
        bool active = idx < limit;

        if (active) {
            key = keys_in[base + idx];
            digit = (key >> shift) & 0xFF;
        }

        // Clear per-SIMD counts
        for (uint i = tid; i < num_simds * RADIX_SIZE; i += tg_size) {
            sub_count[i / RADIX_SIZE][i % RADIX_SIZE] = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // SIMD-level ranking
        uint rank_in_simd = 0;
        uint simd_total = 0;
        for (uint lane = 0; lane < 32; lane++) {
            uint other = simd_shuffle(digit, lane);
            if (active && other == digit) {
                simd_total++;
                if (lane < simd_lane) rank_in_simd++;
            }
        }
        if (active && rank_in_simd == 0) {
            sub_count[simd_id][digit] = simd_total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Cross-SIMD prefix sum per digit
        if (tid < RADIX_SIZE) {
            uint base_off = running_count[tid];
            uint sum = 0;
            for (uint s = 0; s < num_simds; s++) {
                uint c = sub_count[s][tid];
                sub_count[s][tid] = base_off + sum;
                sum += c;
            }
            running_count[tid] = base_off + sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Scatter
        if (active) {
            uint global_pos = sub_count[simd_id][digit] + rank_in_simd;
            keys_out[global_pos] = key;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// --- Key-value variant: stable scatter with parallel sub-tile ranking ---
kernel void radix_scatter_kv(
    device const uint* keys_in    [[buffer(0)]],
    device uint* keys_out         [[buffer(1)]],
    device const uint* vals_in    [[buffer(2)]],
    device uint* vals_out         [[buffer(3)]],
    device const uint* offsets    [[buffer(4)]],
    constant uint& n              [[buffer(5)]],
    constant uint& shift          [[buffer(6)]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tgid                     [[threadgroup_position_in_grid]],
    uint tg_size                  [[threads_per_threadgroup]],
    uint simd_lane                [[thread_index_in_simdgroup]],
    uint simd_id                  [[simdgroup_index_in_threadgroup]]
) {
    threadgroup uint running_count[RADIX_SIZE];
    threadgroup uint sub_count[8][RADIX_SIZE];

    uint base = tgid * TILE_SIZE;
    uint limit = min(uint(TILE_SIZE), n - base);
    uint num_simds = tg_size / 32;

    for (uint i = tid; i < RADIX_SIZE; i += tg_size) {
        running_count[i] = offsets[tgid * RADIX_SIZE + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint sub = 0; sub < limit; sub += tg_size) {
        uint idx = sub + tid;
        uint key = 0, val = 0;
        uint digit = 0xFFFF;
        bool active = idx < limit;

        if (active) {
            key = keys_in[base + idx];
            val = vals_in[base + idx];
            digit = (key >> shift) & 0xFF;
        }

        for (uint i = tid; i < num_simds * RADIX_SIZE; i += tg_size) {
            sub_count[i / RADIX_SIZE][i % RADIX_SIZE] = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint rank_in_simd = 0;
        uint simd_total = 0;
        for (uint lane = 0; lane < 32; lane++) {
            uint other = simd_shuffle(digit, lane);
            if (active && other == digit) {
                simd_total++;
                if (lane < simd_lane) rank_in_simd++;
            }
        }
        if (active && rank_in_simd == 0) {
            sub_count[simd_id][digit] = simd_total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < RADIX_SIZE) {
            uint base_off = running_count[tid];
            uint sum = 0;
            for (uint s = 0; s < num_simds; s++) {
                uint c = sub_count[s][tid];
                sub_count[s][tid] = base_off + sum;
                sum += c;
            }
            running_count[tid] = base_off + sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            uint global_pos = sub_count[simd_id][digit] + rank_in_simd;
            keys_out[global_pos] = key;
            vals_out[global_pos] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
