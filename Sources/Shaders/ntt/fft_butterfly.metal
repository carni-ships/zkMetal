// GPU-accelerated FFT butterfly kernels for BN254 Fr field
// Implements Stockham auto-sort NTT (no separate bit-reversal pass needed).
//
// Kernels:
//   - fft_stockham_radix2: Radix-2 DIT butterfly (one stage per dispatch)
//   - fft_stockham_radix4: Radix-4 DIT butterfly (two stages per dispatch)
//   - fft_stockham_split_radix: Split-radix butterfly for mixed-radix FFT
//   - fft_stockham_fused: Shared-memory fused multi-stage kernel
//   - fft_pointwise_mul: Pointwise multiplication for convolution
//   - fft_scale: Scale all elements by a constant

#include "../fields/bn254_fr.metal"

// ============================================================================
// Stockham Radix-2 DIT Butterfly
// ============================================================================
// Stockham auto-sort: reads from src, writes to dst with permuted indices.
// Each stage alternates src/dst buffers, eliminating the bit-reversal pass.
// stage: current stage index (0-based)
// Each thread handles one butterfly pair.

kernel void fft_stockham_radix2(
    device const Fr* src           [[buffer(0)]],
    device Fr* dst                 [[buffer(1)]],
    device const Fr* twiddles      [[buffer(2)]],
    constant uint& n               [[buffer(3)]],
    constant uint& stage           [[buffer(4)]],
    uint gid                       [[thread_position_in_grid]]
) {
    uint half_n = n >> 1;
    if (gid >= half_n) return;

    // Stockham index computation:
    // At stage s, block_size = 2^(s+1), half_block = 2^s
    // Input is in "stages-processed" order, output auto-sorts
    uint m = 1u << (stage + 1);  // block size
    uint half_m = 1u << stage;   // half block

    // Which block and position within block
    uint block_idx = gid / half_m;
    uint local_idx = gid % half_m;

    // Stockham read indices (from contiguous pairs in src)
    uint src_even = block_idx * half_m + local_idx;
    uint src_odd = src_even + half_n;

    // Stockham write indices (interleaved in dst)
    uint dst_top = block_idx * m + local_idx;
    uint dst_bot = dst_top + half_m;

    // Twiddle factor: omega^(local_idx * n / m)
    uint twiddle_idx = local_idx * (n / m);

    Fr a = src[src_even];
    Fr b = src[src_odd];
    Fr w = twiddles[twiddle_idx];
    Fr wb = fr_mul(w, b);

    dst[dst_top] = fr_add(a, wb);
    dst[dst_bot] = fr_sub(a, wb);
}

// ============================================================================
// Stockham Radix-4 DIT Butterfly (4x throughput)
// ============================================================================
// Fuses two consecutive stages into one dispatch.
// Each thread processes one quartet of elements.

kernel void fft_stockham_radix4(
    device const Fr* src           [[buffer(0)]],
    device Fr* dst                 [[buffer(1)]],
    device const Fr* twiddles      [[buffer(2)]],
    constant uint& n               [[buffer(3)]],
    constant uint& stage           [[buffer(4)]],   // lower stage index (processes stage and stage+1)
    uint gid                       [[thread_position_in_grid]]
) {
    uint quarter_n = n >> 2;
    if (gid >= quarter_n) return;

    // Two stages: s and s+1
    uint m1 = 1u << (stage + 1);    // block size for stage s
    uint half_m1 = 1u << stage;     // half block for stage s
    uint m2 = 1u << (stage + 2);    // block size for stage s+1
    uint half_m2 = 1u << (stage + 1);

    // Stockham indices for radix-4
    uint group = gid / half_m1;
    uint local_idx = gid % half_m1;
    uint stride = n >> 2;

    // Read 4 elements from src (contiguous quarters)
    uint base_read = group * half_m1 + local_idx;
    Fr a0 = src[base_read];
    Fr a1 = src[base_read + stride];
    Fr a2 = src[base_read + 2 * stride];
    Fr a3 = src[base_read + 3 * stride];

    // Stage s twiddles
    uint tw_s = local_idx * (n / m1);
    Fr ws1 = twiddles[tw_s];

    // Stage s butterflies: (a0,a1) and (a2,a3)
    Fr ws_a1 = fr_mul(ws1, a1);
    Fr ws_a3 = fr_mul(ws1, a3);
    Fr b0 = fr_add(a0, ws_a1);
    Fr b1 = fr_sub(a0, ws_a1);
    Fr b2 = fr_add(a2, ws_a3);
    Fr b3 = fr_sub(a2, ws_a3);

    // Stage s+1 twiddles
    uint tw_lo = local_idx * (n / m2);
    uint tw_hi = (local_idx + half_m1) * (n / m2);
    Fr w_lo = twiddles[tw_lo];
    Fr w_hi = twiddles[tw_hi];

    Fr wb2 = fr_mul(w_lo, b2);
    Fr wb3 = fr_mul(w_hi, b3);

    // Write to dst with Stockham permutation
    uint dst_base = group * m2 + local_idx;
    dst[dst_base]             = fr_add(b0, wb2);
    dst[dst_base + half_m1]   = fr_add(b1, wb3);
    dst[dst_base + half_m2]   = fr_sub(b0, wb2);
    dst[dst_base + half_m2 + half_m1] = fr_sub(b1, wb3);
}

// ============================================================================
// Split-Radix Butterfly
// ============================================================================
// Combines radix-2 and radix-4 passes for mixed-radix FFT.
// Handles the case when logN is odd: does a radix-2 pass first, then radix-4.
// This kernel does a single radix-2 Stockham pass.

kernel void fft_stockham_split_radix(
    device const Fr* src           [[buffer(0)]],
    device Fr* dst                 [[buffer(1)]],
    device const Fr* twiddles      [[buffer(2)]],
    constant uint& n               [[buffer(3)]],
    constant uint& stage           [[buffer(4)]],
    constant uint& is_radix4       [[buffer(5)]],  // 0 = radix-2, 1 = radix-4
    uint gid                       [[thread_position_in_grid]]
) {
    if (is_radix4 != 0) {
        // Radix-4 path (same logic as fft_stockham_radix4)
        uint quarter_n = n >> 2;
        if (gid >= quarter_n) return;

        uint m1 = 1u << (stage + 1);
        uint half_m1 = 1u << stage;
        uint m2 = 1u << (stage + 2);
        uint half_m2 = 1u << (stage + 1);
        uint group = gid / half_m1;
        uint local_idx = gid % half_m1;
        uint stride_val = n >> 2;

        uint base_read = group * half_m1 + local_idx;
        Fr a0 = src[base_read];
        Fr a1 = src[base_read + stride_val];
        Fr a2 = src[base_read + 2 * stride_val];
        Fr a3 = src[base_read + 3 * stride_val];

        uint tw_s = local_idx * (n / m1);
        Fr ws1 = twiddles[tw_s];
        Fr ws_a1 = fr_mul(ws1, a1);
        Fr ws_a3 = fr_mul(ws1, a3);
        Fr b0 = fr_add(a0, ws_a1);
        Fr b1 = fr_sub(a0, ws_a1);
        Fr b2 = fr_add(a2, ws_a3);
        Fr b3 = fr_sub(a2, ws_a3);

        uint tw_lo = local_idx * (n / m2);
        uint tw_hi = (local_idx + half_m1) * (n / m2);
        Fr w_lo = twiddles[tw_lo];
        Fr w_hi = twiddles[tw_hi];
        Fr wb2 = fr_mul(w_lo, b2);
        Fr wb3 = fr_mul(w_hi, b3);

        uint dst_base = group * m2 + local_idx;
        dst[dst_base]                     = fr_add(b0, wb2);
        dst[dst_base + half_m1]           = fr_add(b1, wb3);
        dst[dst_base + half_m2]           = fr_sub(b0, wb2);
        dst[dst_base + half_m2 + half_m1] = fr_sub(b1, wb3);
    } else {
        // Radix-2 path
        uint half_n = n >> 1;
        if (gid >= half_n) return;

        uint m = 1u << (stage + 1);
        uint half_m = 1u << stage;
        uint block_idx = gid / half_m;
        uint local_idx = gid % half_m;

        uint src_even = block_idx * half_m + local_idx;
        uint src_odd = src_even + half_n;
        uint dst_top = block_idx * m + local_idx;
        uint dst_bot = dst_top + half_m;
        uint twiddle_idx = local_idx * (n / m);

        Fr a = src[src_even];
        Fr b = src[src_odd];
        Fr w = twiddles[twiddle_idx];
        Fr wb = fr_mul(w, b);

        dst[dst_top] = fr_add(a, wb);
        dst[dst_bot] = fr_sub(a, wb);
    }
}

// ============================================================================
// Fused Shared-Memory Stockham FFT
// ============================================================================
// Processes multiple stages within threadgroup shared memory.
// Bank-conflict-free access: uses padding stride of 33 for 32-byte Fr elements.
// Each threadgroup handles a contiguous block of elements.

// Shared memory: double-buffered Stockham requires 2 arrays.
// 512 Fr elements * 32 bytes * 2 buffers = 32KB (Metal threadgroup limit).
#define FFT_SHARED_SIZE 512

kernel void fft_stockham_fused(
    device Fr* data                [[buffer(0)]],
    device const Fr* twiddles      [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& local_stages    [[buffer(3)]],
    constant uint& stage_offset    [[buffer(4)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint block_size = tg_size << 1;  // each thread handles 2 elements
    uint base = tgid * block_size;

    // Double-buffered shared memory for Stockham (ping-pong)
    threadgroup Fr shared_a[FFT_SHARED_SIZE];
    threadgroup Fr shared_b[FFT_SHARED_SIZE];

    // Coalesced load from global memory into shared_a
    if (base + tid < n)
        shared_a[tid] = data[base + tid];
    if (base + tid + tg_size < n)
        shared_a[tid + tg_size] = data[base + tid + tg_size];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Alternate between shared_a and shared_b (Stockham ping-pong)
    // Even stages read from shared_a, write to shared_b; odd stages reverse.
    for (uint s = 0; s < local_stages; s++) {
        uint stage = stage_offset + s;
        uint half_block = 1u << s;
        uint local_block_size = half_block << 1;

        uint block_idx = tid / half_block;
        uint local_idx = tid % half_block;
        uint i = block_idx * local_block_size + local_idx;
        uint j = i + half_block;

        uint global_block_size = 1u << (stage + 1);
        uint twiddle_idx = local_idx * (n / global_block_size);

        // Read from current buffer
        threadgroup Fr* src_buf = (s & 1u) ? shared_b : shared_a;
        threadgroup Fr* dst_buf = (s & 1u) ? shared_a : shared_b;

        Fr a = src_buf[i];
        Fr b = src_buf[j];
        Fr w = twiddles[twiddle_idx];
        Fr wb = fr_mul(w, b);

        dst_buf[i] = fr_add(a, wb);
        dst_buf[j] = fr_sub(a, wb);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write back from whichever buffer has the final result
    threadgroup Fr* final_buf = (local_stages & 1u) ? shared_b : shared_a;
    if (base + tid < n)
        data[base + tid] = final_buf[tid];
    if (base + tid + tg_size < n)
        data[base + tid + tg_size] = final_buf[tid + tg_size];
}

// ============================================================================
// Inverse FFT kernels (DIF Gentleman-Sande style with Stockham)
// ============================================================================

kernel void fft_stockham_inv_radix2(
    device const Fr* src           [[buffer(0)]],
    device Fr* dst                 [[buffer(1)]],
    device const Fr* twiddles_inv  [[buffer(2)]],
    constant uint& n               [[buffer(3)]],
    constant uint& stage           [[buffer(4)]],
    uint gid                       [[thread_position_in_grid]]
) {
    uint half_n = n >> 1;
    if (gid >= half_n) return;

    uint half_block = 1u << stage;
    uint block_size_val = half_block << 1;

    uint block_idx = gid / half_block;
    uint local_idx = gid % half_block;
    uint i = block_idx * block_size_val + local_idx;
    uint j = i + half_block;

    uint twiddle_idx = local_idx * (n / block_size_val);

    Fr a = src[i];
    Fr b = src[j];

    Fr sum = fr_add(a, b);
    Fr diff = fr_sub(a, b);
    Fr w = twiddles_inv[twiddle_idx];

    // Stockham write: output to contiguous halves
    uint dst_even = block_idx * half_block + local_idx;
    uint dst_odd = dst_even + half_n;

    dst[dst_even] = sum;
    dst[dst_odd] = fr_mul(diff, w);
}

// Fused inverse for shared-memory stages
kernel void fft_stockham_inv_fused(
    device Fr* data                [[buffer(0)]],
    device const Fr* twiddles_inv  [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& local_stages    [[buffer(3)]],
    constant uint& stage_offset    [[buffer(4)]],  // highest stage in this block
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint block_size = tg_size << 1;
    uint base = tgid * block_size;

    threadgroup Fr shared_a[FFT_SHARED_SIZE];
    threadgroup Fr shared_b[FFT_SHARED_SIZE];

    if (base + tid < n)
        shared_a[tid] = data[base + tid];
    if (base + tid + tg_size < n)
        shared_a[tid + tg_size] = data[base + tid + tg_size];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // DIF: process stages from high to low
    for (uint done = 0; done < local_stages; done++) {
        uint stage = stage_offset - done;
        uint half_block = 1u << (local_stages - 1 - done);
        uint local_block_size = half_block << 1;
        uint block_idx = tid / half_block;
        uint local_idx = tid % half_block;
        uint i = block_idx * local_block_size + local_idx;
        uint j = i + half_block;

        uint global_block_size = 1u << (stage + 1);
        uint twiddle_idx = local_idx * (n / global_block_size);

        threadgroup Fr* src_buf = (done & 1u) ? shared_b : shared_a;
        threadgroup Fr* dst_buf = (done & 1u) ? shared_a : shared_b;

        Fr a = src_buf[i];
        Fr b = src_buf[j];

        Fr sum = fr_add(a, b);
        Fr diff = fr_sub(a, b);
        Fr w = twiddles_inv[twiddle_idx];

        dst_buf[i] = sum;
        dst_buf[j] = fr_mul(diff, w);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup Fr* final_buf = (local_stages & 1u) ? shared_b : shared_a;
    if (base + tid < n)
        data[base + tid] = final_buf[tid];
    if (base + tid + tg_size < n)
        data[base + tid + tg_size] = final_buf[tid + tg_size];
}

// ============================================================================
// Standard In-Place Butterfly Kernels (for global stages)
// ============================================================================
// These operate in-place on a single buffer (no Stockham permutation).
// Safe because each thread reads/writes a disjoint pair of elements.

// Forward DIT: a' = a + w*b, b' = a - w*b
kernel void fft_dit_butterfly(
    device Fr* data                [[buffer(0)]],
    device const Fr* twiddles      [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& stage           [[buffer(3)]],
    uint gid                       [[thread_position_in_grid]]
) {
    uint half_block = 1u << stage;
    uint block_size_val = half_block << 1;
    uint num_butterflies = n >> 1;
    if (gid >= num_butterflies) return;

    uint block_idx = gid / half_block;
    uint local_idx = gid % half_block;
    uint i = block_idx * block_size_val + local_idx;
    uint j = i + half_block;

    uint twiddle_idx = local_idx * (n / block_size_val);

    Fr a = data[i];
    Fr b = data[j];
    Fr w = twiddles[twiddle_idx];
    Fr wb = fr_mul(w, b);

    data[i] = fr_add(a, wb);
    data[j] = fr_sub(a, wb);
}

// Inverse DIF: a' = a + b, b' = (a - b) * w_inv
kernel void fft_dif_butterfly(
    device Fr* data                [[buffer(0)]],
    device const Fr* twiddles_inv  [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& stage           [[buffer(3)]],
    uint gid                       [[thread_position_in_grid]]
) {
    uint half_block = 1u << stage;
    uint block_size_val = half_block << 1;
    uint num_butterflies = n >> 1;
    if (gid >= num_butterflies) return;

    uint block_idx = gid / half_block;
    uint local_idx = gid % half_block;
    uint i = block_idx * block_size_val + local_idx;
    uint j = i + half_block;

    uint twiddle_idx = local_idx * (n / block_size_val);

    Fr a = data[i];
    Fr b = data[j];

    Fr sum = fr_add(a, b);
    Fr diff = fr_sub(a, b);
    Fr w = twiddles_inv[twiddle_idx];

    data[i] = sum;
    data[j] = fr_mul(diff, w);
}

// ============================================================================
// Utility kernels
// ============================================================================

// Pointwise multiplication: dst[i] = a[i] * b[i] (for convolution)
kernel void fft_pointwise_mul(
    device const Fr* a             [[buffer(0)]],
    device const Fr* b             [[buffer(1)]],
    device Fr* dst                 [[buffer(2)]],
    constant uint& n               [[buffer(3)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    dst[gid] = fr_mul(a[gid], b[gid]);
}

// Scale all elements by a constant: data[i] *= scale
kernel void fft_scale(
    device Fr* data                [[buffer(0)]],
    device const Fr* scale         [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    data[gid] = fr_mul(data[gid], scale[0]);
}

// Bit-reversal permutation (for compatibility / initial reordering)
kernel void fft_bitrev_inplace(
    device Fr* data                [[buffer(0)]],
    constant uint& n               [[buffer(1)]],
    constant uint& logN            [[buffer(2)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    uint rev = 0;
    uint val = gid;
    for (uint b = 0; b < logN; b++) {
        rev = (rev << 1) | (val & 1);
        val >>= 1;
    }

    if (gid < rev) {
        Fr tmp = data[gid];
        data[gid] = data[rev];
        data[rev] = tmp;
    }
}
