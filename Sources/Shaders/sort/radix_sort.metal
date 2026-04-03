// GPU Radix Sort — 32-bit keys, 4 passes × 8-bit radix (256 buckets)
//
// Three-kernel pipeline per pass:
//   1. histogram: per-threadgroup local histograms of 8-bit digit
//   2. prefix_sum: exclusive prefix sum across all threadgroup histograms
//   3. scatter: write keys to globally sorted positions
//
// Shared memory used for local histograms to avoid global atomic contention.

#include <metal_stdlib>
using namespace metal;

#define RADIX_BITS 8
#define RADIX_SIZE 256  // 2^RADIX_BITS
#define TILE_SIZE 1024  // keys per threadgroup

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
    // Each thread handles one or more digits
    for (uint digit = tid; digit < RADIX_SIZE; digit += tg_size) {
        uint running = 0;
        // For this digit, scan across all tiles
        for (uint t = 0; t < num_tiles; t++) {
            uint count = histograms[t * RADIX_SIZE + digit];
            offsets[t * RADIX_SIZE + digit] = running;
            running += count;
        }
    }

    // Now offsets[t * 256 + d] = sum of histograms[0..t-1][d] for digit d.
    // But we need the global offset: for digit d, base = sum of all counts for digits 0..d-1.
    // Do a second pass to add per-digit base offsets.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute per-digit totals and prefix sum
    threadgroup uint digit_totals[RADIX_SIZE];
    threadgroup uint digit_prefix[RADIX_SIZE];

    for (uint digit = tid; digit < RADIX_SIZE; digit += tg_size) {
        uint total = 0;
        for (uint t = 0; t < num_tiles; t++) {
            total += histograms[t * RADIX_SIZE + digit];
        }
        digit_totals[digit] = total;
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

    // Add per-digit base offset to all tile offsets
    for (uint digit = tid; digit < RADIX_SIZE; digit += tg_size) {
        uint base = digit_prefix[digit];
        for (uint t = 0; t < num_tiles; t++) {
            offsets[t * RADIX_SIZE + digit] += base;
        }
    }
}

// --- Kernel 3: Stable scatter keys to sorted positions ---
// Thread 0 processes keys sequentially within each tile to maintain input order (stability).
// For each key, it increments the per-digit counter to compute its rank,
// then writes to the global position = offsets[tile][digit] + rank.
kernel void radix_scatter(
    device const uint* keys_in    [[buffer(0)]],
    device uint* keys_out         [[buffer(1)]],
    device const uint* offsets    [[buffer(2)]],
    constant uint& n              [[buffer(3)]],
    constant uint& shift          [[buffer(4)]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tgid                     [[threadgroup_position_in_grid]],
    uint tg_size                  [[threads_per_threadgroup]]
) {
    // Only thread 0 — sequential for stability
    if (tid != 0) return;

    uint local_count[RADIX_SIZE];
    for (uint i = 0; i < RADIX_SIZE; i++) {
        local_count[i] = 0;
    }

    uint base = tgid * TILE_SIZE;
    uint limit = min(uint(TILE_SIZE), n - base);
    for (uint i = 0; i < limit; i++) {
        uint key = keys_in[base + i];
        uint digit = (key >> shift) & 0xFF;
        uint global_pos = offsets[tgid * RADIX_SIZE + digit] + local_count[digit];
        local_count[digit]++;
        keys_out[global_pos] = key;
    }
}

// --- Key-value variant: stable scatter with associated values ---
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
    uint tg_size                  [[threads_per_threadgroup]]
) {
    if (tid != 0) return;

    uint local_count[RADIX_SIZE];
    for (uint i = 0; i < RADIX_SIZE; i++) {
        local_count[i] = 0;
    }

    uint base = tgid * TILE_SIZE;
    uint limit = min(uint(TILE_SIZE), n - base);
    for (uint i = 0; i < limit; i++) {
        uint key = keys_in[base + i];
        uint digit = (key >> shift) & 0xFF;
        uint global_pos = offsets[tgid * RADIX_SIZE + digit] + local_count[digit];
        local_count[digit]++;
        keys_out[global_pos] = key;
        vals_out[global_pos] = vals_in[base + i];
    }
}
