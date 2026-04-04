// Circle NTT kernels for Mersenne31 field (p = 2^31 - 1)
// Uses circle group twiddle factors instead of roots of unity.
// Layer 0 uses y-coordinate twiddles; layers 1+ use x-coordinate twiddles.

#include "../fields/mersenne31.metal"

// Forward Circle NTT butterfly (DIT): (a, b) -> (a + tw*b, a - tw*b)
// Used for layers 1..k-1 (x-twiddles) and layer 0 (y-twiddles)
kernel void circle_ntt_butterfly(
    device M31* data                [[buffer(0)]],
    device const M31* twiddles      [[buffer(1)]],
    constant uint& n                [[buffer(2)]],
    constant uint& stage            [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]]
) {
    uint half_block = 1u << stage;
    uint block_size = half_block << 1;
    uint num_butterflies = n >> 1;
    if (gid >= num_butterflies) return;

    uint block_idx = gid / half_block;
    uint local_idx = gid % half_block;
    uint i = block_idx * block_size + local_idx;
    uint j = i + half_block;
    uint twiddle_idx = local_idx * (n / block_size);

    M31 a = data[i];
    M31 b = data[j];
    M31 w = twiddles[twiddle_idx];
    M31 wb = m31_mul(w, b);

    data[i] = m31_add(a, wb);
    data[j] = m31_sub(a, wb);
}

// Inverse Circle NTT butterfly (DIF): (a, b) -> (a + b, (a - b) / tw)
// = (a + b, (a - b) * inv_tw)
kernel void circle_intt_butterfly(
    device M31* data                [[buffer(0)]],
    device const M31* inv_twiddles  [[buffer(1)]],
    constant uint& n                [[buffer(2)]],
    constant uint& stage            [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]]
) {
    uint half_block = 1u << stage;
    uint block_size = half_block << 1;
    uint num_butterflies = n >> 1;
    if (gid >= num_butterflies) return;

    uint block_idx = gid / half_block;
    uint local_idx = gid % half_block;
    uint i = block_idx * block_size + local_idx;
    uint j = i + half_block;
    uint twiddle_idx = local_idx * (n / block_size);

    M31 a = data[i];
    M31 b = data[j];
    M31 sum = m31_add(a, b);
    M31 diff = m31_sub(a, b);
    M31 w_inv = inv_twiddles[twiddle_idx];

    data[i] = sum;
    data[j] = m31_mul(diff, w_inv);
}

// Fused Circle NTT: process multiple stages in threadgroup memory
// Handles the initial small-block stages efficiently
kernel void circle_ntt_butterfly_fused(
    device M31* data                [[buffer(0)]],
    device const M31* twiddles      [[buffer(1)]],
    constant uint& n                [[buffer(2)]],
    constant uint& fused_stages     [[buffer(3)]],
    constant uint& stage_offset     [[buffer(4)]],
    uint gid                        [[thread_position_in_grid]],
    uint tid                        [[thread_position_in_threadgroup]],
    uint tg_id                      [[threadgroup_position_in_grid]],
    threadgroup M31* shared_data    [[threadgroup(0)]]
) {
    uint block_elems = 1u << fused_stages;
    uint base = tg_id * block_elems;
    uint half_n = block_elems >> 1;

    // Load into threadgroup memory
    shared_data[tid] = data[base + tid];
    shared_data[tid + half_n] = data[base + tid + half_n];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process fused stages (DIT: small blocks first)
    for (uint s = 0; s < fused_stages; s++) {
        uint h = 1u << s;
        uint bs = h << 1;
        uint block_idx = tid / h;
        uint local_idx = tid % h;
        uint i = block_idx * bs + local_idx;
        uint j = i + h;

        // Twiddle index in the full twiddle array
        uint global_stage = stage_offset + s;
        uint global_half_block = 1u << global_stage;
        uint global_block_size = global_half_block << 1;
        uint global_local = (base % global_block_size) + local_idx;
        uint tw_idx;
        if (global_stage < 31) {
            tw_idx = (global_local % global_half_block) * (n / global_block_size);
        } else {
            tw_idx = local_idx * (n / bs);
        }

        M31 a = shared_data[i];
        M31 b = shared_data[j];
        M31 w = twiddles[tw_idx];
        M31 wb = m31_mul(w, b);

        shared_data[i] = m31_add(a, wb);
        shared_data[j] = m31_sub(a, wb);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write back
    data[base + tid] = shared_data[tid];
    data[base + tid + half_n] = shared_data[tid + half_n];
}

// Fused inverse Circle NTT
kernel void circle_intt_butterfly_fused(
    device M31* data                [[buffer(0)]],
    device const M31* inv_twiddles  [[buffer(1)]],
    constant uint& n                [[buffer(2)]],
    constant uint& fused_stages     [[buffer(3)]],
    constant uint& stage_offset     [[buffer(4)]],
    uint gid                        [[thread_position_in_grid]],
    uint tid                        [[thread_position_in_threadgroup]],
    uint tg_id                      [[threadgroup_position_in_grid]],
    threadgroup M31* shared_data    [[threadgroup(0)]]
) {
    uint block_elems = 1u << fused_stages;
    uint base = tg_id * block_elems;
    uint half_n = block_elems >> 1;

    shared_data[tid] = data[base + tid];
    shared_data[tid + half_n] = data[base + tid + half_n];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process fused stages (DIF: large blocks first)
    for (uint si = 0; si < fused_stages; si++) {
        uint s = fused_stages - 1 - si;
        uint h = 1u << s;
        uint bs = h << 1;
        uint block_idx = tid / h;
        uint local_idx = tid % h;
        uint i = block_idx * bs + local_idx;
        uint j = i + h;

        uint global_stage = stage_offset - si;
        uint global_half_block = 1u << global_stage;
        uint global_block_size = global_half_block << 1;
        uint global_local = (base % global_block_size) + local_idx;
        uint tw_idx;
        if (global_stage < 31) {
            tw_idx = (global_local % global_half_block) * (n / global_block_size);
        } else {
            tw_idx = local_idx * (n / bs);
        }

        M31 a = shared_data[i];
        M31 b = shared_data[j];
        M31 sum = m31_add(a, b);
        M31 diff = m31_sub(a, b);
        M31 w_inv = inv_twiddles[tw_idx];

        shared_data[i] = sum;
        shared_data[j] = m31_mul(diff, w_inv);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    data[base + tid] = shared_data[tid];
    data[base + tid + half_n] = shared_data[tid + half_n];
}

// Scale kernel: multiply all elements by a scalar
kernel void circle_ntt_scale(
    device M31* data                [[buffer(0)]],
    device const M31* scalar        [[buffer(1)]],
    constant uint& n                [[buffer(2)]],
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    data[gid] = m31_mul(data[gid], scalar[0]);
}
