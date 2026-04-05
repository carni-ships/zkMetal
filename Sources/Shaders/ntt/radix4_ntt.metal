// Radix-4 NTT butterfly kernels for BabyBear and Goldilocks fields
// Each radix-4 butterfly fuses 2 radix-2 stages into one kernel dispatch,
// halving the number of GPU round-trips compared to pure radix-2.
//
// Provides both global-memory and threadgroup-shared-memory variants:
//   - Global: for large outer stages where data doesn't fit in threadgroup memory
//   - Fused: loads data into threadgroup shared memory, does multiple radix-4 passes

#include "../fields/babybear.metal"
#include "../fields/goldilocks.metal"

// ============================================================================
// BabyBear Radix-4 (32-bit field, 4 bytes/element)
// Shared memory: 8192 elements * 4 bytes = 32KB
// ============================================================================

// --- Forward NTT (DIT) ---

// Radix-4 DIT: fuses stages (stage) and (stage+1) into one dispatch.
// Each thread processes one quartet of elements.
// Twiddle buffer: omega^0, omega^1, ..., omega^(N/2-1)
kernel void radix4_butterfly_bb(
    device Bb* data                [[buffer(0)]],
    device const Bb* twiddles      [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& stage           [[buffer(3)]],    // lower stage index
    uint gid                       [[thread_position_in_grid]]
) {
    uint h = 1u << stage;          // half_block for stage s
    uint block4 = h << 2;          // block size spanning both stages
    uint num_quads = n >> 2;
    if (gid >= num_quads) return;

    uint block_idx = gid / h;
    uint local_idx = gid % h;
    uint base = block_idx * block4 + local_idx;

    // Load 4 elements
    Bb a0 = data[base];
    Bb a1 = data[base + h];
    Bb a2 = data[base + 2 * h];
    Bb a3 = data[base + 3 * h];

    // Stage s: twiddle for pairs (a0,a1) and (a2,a3)
    Bb ws = twiddles[local_idx * (n / (2 * h))];
    Bb ws_a1 = bb_mul(ws, a1);
    Bb ws_a3 = bb_mul(ws, a3);
    Bb b0 = bb_add(a0, ws_a1);
    Bb b1 = bb_sub(a0, ws_a1);
    Bb b2 = bb_add(a2, ws_a3);
    Bb b3 = bb_sub(a2, ws_a3);

    // Stage s+1: twiddle for pairs (b0,b2) and (b1,b3)
    Bb w_lo = twiddles[local_idx * (n / block4)];
    Bb w_hi = twiddles[(local_idx + h) * (n / block4)];
    Bb wb2 = bb_mul(w_lo, b2);
    Bb wb3 = bb_mul(w_hi, b3);

    data[base]         = bb_add(b0, wb2);
    data[base + 2 * h] = bb_sub(b0, wb2);
    data[base + h]     = bb_add(b1, wb3);
    data[base + 3 * h] = bb_sub(b1, wb3);
}

// --- Inverse NTT (DIF) ---

// Radix-4 DIF: fuses stages (stage) and (stage-1) into one dispatch.
// DIF processes from high stage down.
kernel void radix4_inv_butterfly_bb(
    device Bb* data                [[buffer(0)]],
    device const Bb* twiddles_inv  [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& stage           [[buffer(3)]],    // higher stage (s); also processes s-1
    uint gid                       [[thread_position_in_grid]]
) {
    uint h_hi = 1u << stage;       // half_block for stage s
    uint h_lo = h_hi >> 1;         // half_block for stage s-1
    uint block4 = h_hi << 1;       // block size for stage s
    uint num_quads = n >> 2;
    if (gid >= num_quads) return;

    uint block_idx = gid / h_lo;
    uint local_idx = gid % h_lo;
    uint base = block_idx * block4 + local_idx;

    Bb a0 = data[base];
    Bb a1 = data[base + h_lo];
    Bb a2 = data[base + h_hi];
    Bb a3 = data[base + h_hi + h_lo];

    // Stage s (DIF): pairs (a0,a2) and (a1,a3)
    Bb ws_lo = twiddles_inv[local_idx * (n / block4)];
    Bb ws_hi = twiddles_inv[(local_idx + h_lo) * (n / block4)];
    Bb b0 = bb_add(a0, a2);
    Bb b2 = bb_mul(bb_sub(a0, a2), ws_lo);
    Bb b1 = bb_add(a1, a3);
    Bb b3 = bb_mul(bb_sub(a1, a3), ws_hi);

    // Stage s-1 (DIF): pairs (b0,b1) and (b2,b3)
    Bb w_s1 = twiddles_inv[local_idx * (n / (2 * h_lo))];
    data[base]              = bb_add(b0, b1);
    data[base + h_lo]       = bb_mul(bb_sub(b0, b1), w_s1);
    data[base + h_hi]       = bb_add(b2, b3);
    data[base + h_hi + h_lo] = bb_mul(bb_sub(b2, b3), w_s1);
}

// --- Fused threadgroup radix-4 DIT for BabyBear ---
// Processes multiple radix-4 passes in shared memory before writing back.
// local_stages: total number of radix-2 stages to do in shared memory (must be even for pure radix-4)
kernel void radix4_butterfly_fused_bb(
    device Bb* data                [[buffer(0)]],
    device const Bb* twiddles      [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& local_stages    [[buffer(3)]],  // number of stages in shared memory
    constant uint& stage_offset    [[buffer(4)]],  // first global stage index
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    // Each threadgroup handles (tg_size * 2) elements
    uint block_size = tg_size << 1;
    uint base = tgid * block_size;

    threadgroup Bb shared[8192];  // 8192 * 4 = 32KB

    // Load into shared memory
    if (base + tid < n) shared[tid] = data[base + tid];
    if (base + tid + tg_size < n) shared[tid + tg_size] = data[base + tid + tg_size];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process stages in pairs (radix-4)
    uint s = 0;
    while (s + 1 < local_stages) {
        // Radix-4: fuse stages s and s+1
        uint h = 1u << s;
        uint block4 = h << 2;
        uint num_quads = block_size >> 2;

        if (tid < num_quads) {
            uint blk_idx = tid / h;
            uint loc_idx = tid % h;
            uint lbase = blk_idx * block4 + loc_idx;

            Bb a0 = shared[lbase];
            Bb a1 = shared[lbase + h];
            Bb a2 = shared[lbase + 2 * h];
            Bb a3 = shared[lbase + 3 * h];

            uint global_stage = stage_offset + s;
            uint global_half = 1u << global_stage;

            // Stage s twiddle
            Bb ws = twiddles[loc_idx * (n / (2 * global_half))];
            Bb ws_a1 = bb_mul(ws, a1);
            Bb ws_a3 = bb_mul(ws, a3);
            Bb b0 = bb_add(a0, ws_a1);
            Bb b1 = bb_sub(a0, ws_a1);
            Bb b2 = bb_add(a2, ws_a3);
            Bb b3 = bb_sub(a2, ws_a3);

            // Stage s+1 twiddles
            uint global_block4 = global_half << 2;
            Bb w_lo = twiddles[loc_idx * (n / global_block4)];
            Bb w_hi = twiddles[(loc_idx + h) * (n / global_block4)];
            Bb wb2 = bb_mul(w_lo, b2);
            Bb wb3 = bb_mul(w_hi, b3);

            shared[lbase]         = bb_add(b0, wb2);
            shared[lbase + 2 * h] = bb_sub(b0, wb2);
            shared[lbase + h]     = bb_add(b1, wb3);
            shared[lbase + 3 * h] = bb_sub(b1, wb3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s += 2;
    }

    // Handle odd remaining stage with radix-2
    if (s < local_stages) {
        uint half_block = 1u << s;
        uint local_block_size = half_block << 1;
        uint block_idx = tid / half_block;
        uint local_idx = tid % half_block;
        uint i = block_idx * local_block_size + local_idx;
        uint j = i + half_block;

        uint stage = stage_offset + s;
        uint global_block_size = 1u << (stage + 1);
        uint twiddle_idx = local_idx * (n / global_block_size);

        Bb a = shared[i];
        Bb b = shared[j];
        Bb w = twiddles[twiddle_idx];
        Bb wb = bb_mul(w, b);
        shared[i] = bb_add(a, wb);
        shared[j] = bb_sub(a, wb);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write back
    if (base + tid < n) data[base + tid] = shared[tid];
    if (base + tid + tg_size < n) data[base + tid + tg_size] = shared[tid + tg_size];
}

// --- Fused threadgroup radix-4 DIF for BabyBear iNTT ---
kernel void radix4_inv_butterfly_fused_bb(
    device Bb* data                [[buffer(0)]],
    device const Bb* twiddles_inv  [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& local_stages    [[buffer(3)]],
    constant uint& stage_offset    [[buffer(4)]],  // highest global stage in this block
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint block_size = tg_size << 1;
    uint base = tgid * block_size;

    threadgroup Bb shared[8192];

    if (base + tid < n) shared[tid] = data[base + tid];
    if (base + tid + tg_size < n) shared[tid + tg_size] = data[base + tid + tg_size];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // DIF: process stages from high to low, in pairs (radix-4)
    uint done = 0;
    while (done + 1 < local_stages) {
        // Current pair: stages (stage_offset - done) and (stage_offset - done - 1)
        uint stage_hi = stage_offset - done;
        uint h_hi = 1u << (local_stages - 1 - done);
        uint h_lo = h_hi >> 1;
        uint blk4 = h_hi << 1;
        uint num_quads = block_size >> 2;

        if (tid < num_quads) {
            uint blk_idx = tid / h_lo;
            uint loc_idx = tid % h_lo;
            uint lbase = blk_idx * blk4 + loc_idx;

            Bb a0 = shared[lbase];
            Bb a1 = shared[lbase + h_lo];
            Bb a2 = shared[lbase + h_hi];
            Bb a3 = shared[lbase + h_hi + h_lo];

            // Stage s (DIF high)
            uint global_block4 = 1u << (stage_hi + 1);
            Bb ws_lo = twiddles_inv[loc_idx * (n / global_block4)];
            Bb ws_hi = twiddles_inv[(loc_idx + h_lo) * (n / global_block4)];
            Bb b0 = bb_add(a0, a2);
            Bb b2 = bb_mul(bb_sub(a0, a2), ws_lo);
            Bb b1 = bb_add(a1, a3);
            Bb b3 = bb_mul(bb_sub(a1, a3), ws_hi);

            // Stage s-1 (DIF low)
            uint global_half_lo = 1u << (stage_hi - 1);
            Bb w_s1 = twiddles_inv[loc_idx * (n / (2 * global_half_lo))];
            shared[lbase]              = bb_add(b0, b1);
            shared[lbase + h_lo]       = bb_mul(bb_sub(b0, b1), w_s1);
            shared[lbase + h_hi]       = bb_add(b2, b3);
            shared[lbase + h_hi + h_lo] = bb_mul(bb_sub(b2, b3), w_s1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        done += 2;
    }

    // Handle odd remaining stage with radix-2 DIF
    if (done < local_stages) {
        uint stage = stage_offset - done;
        uint half_block = 1u << (local_stages - 1 - done);
        uint local_block_size = half_block << 1;
        uint block_idx = tid / half_block;
        uint local_idx = tid % half_block;
        uint i = block_idx * local_block_size + local_idx;
        uint j = i + half_block;

        uint global_block_size = 1u << (stage + 1);
        uint twiddle_idx = local_idx * (n / global_block_size);

        Bb a = shared[i];
        Bb b = shared[j];
        Bb sum = bb_add(a, b);
        Bb diff = bb_sub(a, b);
        Bb w = twiddles_inv[twiddle_idx];
        shared[i] = sum;
        shared[j] = bb_mul(diff, w);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (base + tid < n) data[base + tid] = shared[tid];
    if (base + tid + tg_size < n) data[base + tid + tg_size] = shared[tid + tg_size];
}


// ============================================================================
// Goldilocks Radix-4 (64-bit field, 8 bytes/element)
// Shared memory: 4096 elements * 8 bytes = 32KB
// ============================================================================

// --- Forward NTT (DIT) ---

kernel void radix4_butterfly_gl(
    device Gl* data                [[buffer(0)]],
    device const Gl* twiddles      [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& stage           [[buffer(3)]],
    uint gid                       [[thread_position_in_grid]]
) {
    uint h = 1u << stage;
    uint block4 = h << 2;
    uint num_quads = n >> 2;
    if (gid >= num_quads) return;

    uint block_idx = gid / h;
    uint local_idx = gid % h;
    uint base = block_idx * block4 + local_idx;

    Gl a0 = data[base];
    Gl a1 = data[base + h];
    Gl a2 = data[base + 2 * h];
    Gl a3 = data[base + 3 * h];

    // Stage s
    Gl ws = twiddles[local_idx * (n / (2 * h))];
    Gl ws_a1 = gl_mul(ws, a1);
    Gl ws_a3 = gl_mul(ws, a3);
    Gl b0 = gl_add(a0, ws_a1);
    Gl b1 = gl_sub(a0, ws_a1);
    Gl b2 = gl_add(a2, ws_a3);
    Gl b3 = gl_sub(a2, ws_a3);

    // Stage s+1
    Gl w_lo = twiddles[local_idx * (n / block4)];
    Gl w_hi = twiddles[(local_idx + h) * (n / block4)];
    Gl wb2 = gl_mul(w_lo, b2);
    Gl wb3 = gl_mul(w_hi, b3);

    data[base]         = gl_add(b0, wb2);
    data[base + 2 * h] = gl_sub(b0, wb2);
    data[base + h]     = gl_add(b1, wb3);
    data[base + 3 * h] = gl_sub(b1, wb3);
}

// --- Inverse NTT (DIF) ---

kernel void radix4_inv_butterfly_gl(
    device Gl* data                [[buffer(0)]],
    device const Gl* twiddles_inv  [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& stage           [[buffer(3)]],
    uint gid                       [[thread_position_in_grid]]
) {
    uint h_hi = 1u << stage;
    uint h_lo = h_hi >> 1;
    uint block4 = h_hi << 1;
    uint num_quads = n >> 2;
    if (gid >= num_quads) return;

    uint block_idx = gid / h_lo;
    uint local_idx = gid % h_lo;
    uint base = block_idx * block4 + local_idx;

    Gl a0 = data[base];
    Gl a1 = data[base + h_lo];
    Gl a2 = data[base + h_hi];
    Gl a3 = data[base + h_hi + h_lo];

    // Stage s (DIF)
    Gl ws_lo = twiddles_inv[local_idx * (n / block4)];
    Gl ws_hi = twiddles_inv[(local_idx + h_lo) * (n / block4)];
    Gl b0 = gl_add(a0, a2);
    Gl b2 = gl_mul(gl_sub(a0, a2), ws_lo);
    Gl b1 = gl_add(a1, a3);
    Gl b3 = gl_mul(gl_sub(a1, a3), ws_hi);

    // Stage s-1 (DIF)
    Gl w_s1 = twiddles_inv[local_idx * (n / (2 * h_lo))];
    data[base]              = gl_add(b0, b1);
    data[base + h_lo]       = gl_mul(gl_sub(b0, b1), w_s1);
    data[base + h_hi]       = gl_add(b2, b3);
    data[base + h_hi + h_lo] = gl_mul(gl_sub(b2, b3), w_s1);
}

// --- Fused threadgroup radix-4 DIT for Goldilocks ---
kernel void radix4_butterfly_fused_gl(
    device Gl* data                [[buffer(0)]],
    device const Gl* twiddles      [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& local_stages    [[buffer(3)]],
    constant uint& stage_offset    [[buffer(4)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint block_size = tg_size << 1;
    uint base = tgid * block_size;

    threadgroup Gl shared[4096];  // 4096 * 8 = 32KB

    if (base + tid < n) shared[tid] = data[base + tid];
    if (base + tid + tg_size < n) shared[tid + tg_size] = data[base + tid + tg_size];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint s = 0;
    while (s + 1 < local_stages) {
        uint h = 1u << s;
        uint block4 = h << 2;
        uint num_quads = block_size >> 2;

        if (tid < num_quads) {
            uint blk_idx = tid / h;
            uint loc_idx = tid % h;
            uint lbase = blk_idx * block4 + loc_idx;

            Gl a0 = shared[lbase];
            Gl a1 = shared[lbase + h];
            Gl a2 = shared[lbase + 2 * h];
            Gl a3 = shared[lbase + 3 * h];

            uint global_stage = stage_offset + s;
            uint global_half = 1u << global_stage;

            Gl ws = twiddles[loc_idx * (n / (2 * global_half))];
            Gl ws_a1 = gl_mul(ws, a1);
            Gl ws_a3 = gl_mul(ws, a3);
            Gl b0 = gl_add(a0, ws_a1);
            Gl b1 = gl_sub(a0, ws_a1);
            Gl b2 = gl_add(a2, ws_a3);
            Gl b3 = gl_sub(a2, ws_a3);

            uint global_block4 = global_half << 2;
            Gl w_lo = twiddles[loc_idx * (n / global_block4)];
            Gl w_hi = twiddles[(loc_idx + h) * (n / global_block4)];
            Gl wb2 = gl_mul(w_lo, b2);
            Gl wb3 = gl_mul(w_hi, b3);

            shared[lbase]         = gl_add(b0, wb2);
            shared[lbase + 2 * h] = gl_sub(b0, wb2);
            shared[lbase + h]     = gl_add(b1, wb3);
            shared[lbase + 3 * h] = gl_sub(b1, wb3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s += 2;
    }

    // Odd remaining stage: radix-2
    if (s < local_stages) {
        uint half_block = 1u << s;
        uint local_block_size = half_block << 1;
        uint block_idx = tid / half_block;
        uint local_idx = tid % half_block;
        uint i = block_idx * local_block_size + local_idx;
        uint j = i + half_block;

        uint stage = stage_offset + s;
        uint global_block_size = 1u << (stage + 1);
        uint twiddle_idx = local_idx * (n / global_block_size);

        Gl a = shared[i];
        Gl b = shared[j];
        Gl w = twiddles[twiddle_idx];
        Gl wb = gl_mul(w, b);
        shared[i] = gl_add(a, wb);
        shared[j] = gl_sub(a, wb);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (base + tid < n) data[base + tid] = shared[tid];
    if (base + tid + tg_size < n) data[base + tid + tg_size] = shared[tid + tg_size];
}

// --- Fused threadgroup radix-4 DIF for Goldilocks iNTT ---
kernel void radix4_inv_butterfly_fused_gl(
    device Gl* data                [[buffer(0)]],
    device const Gl* twiddles_inv  [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& local_stages    [[buffer(3)]],
    constant uint& stage_offset    [[buffer(4)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint block_size = tg_size << 1;
    uint base = tgid * block_size;

    threadgroup Gl shared[4096];

    if (base + tid < n) shared[tid] = data[base + tid];
    if (base + tid + tg_size < n) shared[tid + tg_size] = data[base + tid + tg_size];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint done = 0;
    while (done + 1 < local_stages) {
        uint stage_hi = stage_offset - done;
        uint h_hi = 1u << (local_stages - 1 - done);
        uint h_lo = h_hi >> 1;
        uint blk4 = h_hi << 1;
        uint num_quads = block_size >> 2;

        if (tid < num_quads) {
            uint blk_idx = tid / h_lo;
            uint loc_idx = tid % h_lo;
            uint lbase = blk_idx * blk4 + loc_idx;

            Gl a0 = shared[lbase];
            Gl a1 = shared[lbase + h_lo];
            Gl a2 = shared[lbase + h_hi];
            Gl a3 = shared[lbase + h_hi + h_lo];

            uint global_block4 = 1u << (stage_hi + 1);
            Gl ws_lo = twiddles_inv[loc_idx * (n / global_block4)];
            Gl ws_hi = twiddles_inv[(loc_idx + h_lo) * (n / global_block4)];
            Gl b0 = gl_add(a0, a2);
            Gl b2 = gl_mul(gl_sub(a0, a2), ws_lo);
            Gl b1 = gl_add(a1, a3);
            Gl b3 = gl_mul(gl_sub(a1, a3), ws_hi);

            uint global_half_lo = 1u << (stage_hi - 1);
            Gl w_s1 = twiddles_inv[loc_idx * (n / (2 * global_half_lo))];
            shared[lbase]              = gl_add(b0, b1);
            shared[lbase + h_lo]       = gl_mul(gl_sub(b0, b1), w_s1);
            shared[lbase + h_hi]       = gl_add(b2, b3);
            shared[lbase + h_hi + h_lo] = gl_mul(gl_sub(b2, b3), w_s1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        done += 2;
    }

    // Odd remaining stage: radix-2 DIF
    if (done < local_stages) {
        uint stage = stage_offset - done;
        uint half_block = 1u << (local_stages - 1 - done);
        uint local_block_size = half_block << 1;
        uint block_idx = tid / half_block;
        uint local_idx = tid % half_block;
        uint i = block_idx * local_block_size + local_idx;
        uint j = i + half_block;

        uint global_block_size = 1u << (stage + 1);
        uint twiddle_idx = local_idx * (n / global_block_size);

        Gl a = shared[i];
        Gl b = shared[j];
        Gl sum = gl_add(a, b);
        Gl diff = gl_sub(a, b);
        Gl w = twiddles_inv[twiddle_idx];
        shared[i] = sum;
        shared[j] = gl_mul(diff, w);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (base + tid < n) data[base + tid] = shared[tid];
    if (base + tid + tg_size < n) data[base + tid + tg_size] = shared[tid + tg_size];
}
