// Batch Circle NTT kernels for M31 (Mersenne31) field
// Single GPU dispatch processes multiple columns in parallel using grid Y dimension
// Each column is an independent Circle NTT transform

#include "../fields/mersenne31.metal"

// Forward Circle NTT butterfly (DIT) for batch processing
// Uses grid Y for column index, grid X for butterfly within column
// Circle NTT: layers k-1 down to 1 (x-twiddle), then layer 0 (y-twiddle)
kernel void batch_circle_ntt_butterfly_dit(
    device M31* data                [[buffer(0)]],
    device const M31* twiddles      [[buffer(1)]],   // layer-offset twiddles
    constant uint& n                [[buffer(2)]],    // size per column
    constant uint& stage            [[buffer(3)]],    // butterfly stage
    constant uint& num_cols         [[buffer(4)]],    // number of columns
    uint gid                        [[thread_position_in_grid]],
    uint gid_y                      [[thread_position_in_grid]]
) {
    uint col = gid_y;
    if (col >= num_cols) return;

    uint n2 = n >> 1;
    if (gid >= n2) return;

    uint half_block = 1u << stage;
    uint block_size = half_block << 1;
    uint num_butterflies = n >> 1;

    uint block_idx = gid / half_block;
    uint local_idx = gid % half_block;
    uint i = block_idx * block_size + local_idx;
    uint j = i + half_block;
    uint twiddle_idx = local_idx * (n / block_size);

    // Compute offset into column data
    uint col_offset = col * n;
    M31 a = data[col_offset + i];
    M31 b = data[col_offset + j];
    M31 w = twiddles[twiddle_idx];
    M31 wb = m31_mul(w, b);

    data[col_offset + i] = m31_add(a, wb);
    data[col_offset + j] = m31_sub(a, wb);
}

// Inverse Circle NTT butterfly (DIF) for batch processing
kernel void batch_circle_ntt_butterfly_dif(
    device M31* data                [[buffer(0)]],
    device const M31* inv_twiddles  [[buffer(1)]],   // layer-offset inv twiddles
    constant uint& n                [[buffer(2)]],
    constant uint& stage            [[buffer(3)]],
    constant uint& num_cols         [[buffer(4)]],
    uint gid                        [[thread_position_in_grid]],
    uint gid_y                      [[thread_position_in_grid]]
) {
    uint col = gid_y;
    if (col >= num_cols) return;

    uint n2 = n >> 1;
    if (gid >= n2) return;

    uint half_block = 1u << stage;
    uint block_size = half_block << 1;

    uint block_idx = gid / half_block;
    uint local_idx = gid % half_block;
    uint i = block_idx * block_size + local_idx;
    uint j = i + half_block;
    uint twiddle_idx = local_idx * (n / block_size);

    uint col_offset = col * n;
    M31 a = data[col_offset + i];
    M31 b = data[col_offset + j];
    M31 sum = m31_add(a, b);
    M31 diff = m31_sub(a, b);
    M31 w_inv = inv_twiddles[twiddle_idx];

    data[col_offset + i] = sum;
    data[col_offset + j] = m31_mul(diff, w_inv);
}

// Scale kernel for batch (multiply by 1/n for inverse)
kernel void batch_circle_ntt_scale(
    device M31* data                [[buffer(0)]],
    device const M31* scalar        [[buffer(1)]],
    constant uint& n                [[buffer(2)]],
    constant uint& num_cols         [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]],
    uint gid_y                      [[thread_position_in_grid]]
) {
    uint col = gid_y;
    if (col >= num_cols) return;
    if (gid >= n) return;

    uint col_offset = col * n;
    data[col_offset + gid] = m31_mul(data[col_offset + gid], scalar[0]);
}

// Bit-reversal permutation for batch
kernel void batch_circle_ntt_bitrev(
    device M31* data                [[buffer(0)]],
    constant uint& n                [[buffer(1)]],
    constant uint& log_n            [[buffer(2)]],
    constant uint& num_cols         [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]],
    uint gid_y                      [[thread_position_in_grid]]
) {
    uint col = gid_y;
    if (col >= num_cols) return;

    uint count = n >> 1;
    if (gid >= count) return;

    // Compute bit-reversed index
    uint src = gid;
    uint rev = 0;
    for (uint i = 0; i < log_n; i++) {
        rev = (rev << 1) | (src & 1);
        src >>= 1;
    }

    // Only swap if rev > gid (each pair processed once)
    if (rev > gid) {
        uint col_offset = col * n;
        uint idx_a = gid;
        uint idx_b = rev;
        M31 tmp = data[col_offset + idx_a];
        data[col_offset + idx_a] = data[col_offset + idx_b];
        data[col_offset + idx_b] = tmp;
    }
}

// Fused bit-reversal + early DIT butterfly stages for forward Circle NTT
// Processes local_stages DIT stages in threadgroup memory
// Each threadgroup handles one block (block_elems elements) within one column
kernel void batch_circle_ntt_fused_bitrev_dit(
    device M31* data                [[buffer(0)]],
    device const M31* twiddles      [[buffer(1)]],
    constant uint& n                [[buffer(2)]],
    constant uint& local_stages     [[buffer(3)]],
    constant uint& num_cols         [[buffer(4)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint tg_size                    [[threads_per_threadgroup]]
) {
    uint block_elems = 1u << local_stages;
    uint half_block = block_elems >> 1;
    uint num_blocks_per_col = n >> local_stages;
    uint total_blocks = num_blocks_per_col * num_cols;

    if (tgid >= total_blocks) return;

    uint block_idx = tgid % num_blocks_per_col;
    uint col = tgid / num_blocks_per_col;
    uint base_offset = col * n;
    uint base = block_idx * block_elems;

    // Load into threadgroup memory
    threadgroup M31 shared[1024];
    if (base + tid < n)
        shared[tid] = data[base_offset + base + tid];
    if (base + tid + half_block < n)
        shared[tid + half_block] = data[base_offset + base + tid + half_block];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Perform DIT butterfly stages
    for (uint s = 0; s < local_stages; s++) {
        uint stage = s;
        uint h = 1u << s;
        uint bs = h << 1;
        uint block_index = tid / h;
        uint local_index = tid % h;
        uint i = block_index * bs + local_index;
        uint j = i + h;

        uint global_block_size = 1u << (stage + 1);
        uint twiddle_idx = local_index * (n / global_block_size);

        M31 a = shared[i];
        M31 b = shared[j];
        M31 w = twiddles[twiddle_idx];
        M31 wb = m31_mul(w, b);
        shared[i] = m31_add(a, wb);
        shared[j] = m31_sub(a, wb);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write back
    if (base + tid < n)
        data[base_offset + base + tid] = shared[tid];
    if (base + tid + half_block < n)
        data[base_offset + base + tid + half_block] = shared[tid + half_block];
}

// Fused inverse butterfly + bit-reversal for inverse Circle NTT
kernel void batch_circle_ntt_fused_bitrev_dif(
    device M31* data                [[buffer(0)]],
    device const M31* inv_twiddles  [[buffer(1)]],
    constant uint& n                [[buffer(2)]],
    constant uint& local_stages     [[buffer(3)]],
    constant uint& stage_offset     [[buffer(4)]],  // highest stage in this fused block
    constant uint& num_cols         [[buffer(5)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint tg_size                    [[threads_per_threadgroup]]
) {
    uint block_elems = 1u << local_stages;
    uint half_block = block_elems >> 1;
    uint num_blocks_per_col = n >> local_stages;
    uint total_blocks = num_blocks_per_col * num_cols;

    if (tgid >= total_blocks) return;

    uint block_idx = tgid % num_blocks_per_col;
    uint col = tgid / num_blocks_per_col;
    uint base_offset = col * n;
    uint base = block_idx * block_elems;

    // Load into threadgroup memory
    threadgroup M31 shared[1024];
    if (base + tid < n)
        shared[tid] = data[base_offset + base + tid];
    if (base + tid + half_block < n)
        shared[tid + half_block] = data[base_offset + base + tid + half_block];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Perform DIF butterfly stages (from high to low)
    for (uint s = 0; s < local_stages; s++) {
        uint stage = stage_offset - s;
        uint h = 1u << s;
        uint bs = h << 1;
        uint block_index = tid / h;
        uint local_index = tid % h;
        uint i = block_index * bs + local_index;
        uint j = i + h;

        uint global_block_size = 1u << (stage + 1);
        uint twiddle_idx = block_index * (n / global_block_size);

        M31 a = shared[i];
        M31 b = shared[j];
        M31 sum = m31_add(a, b);
        M31 diff = m31_sub(a, b);
        M31 w_inv = inv_twiddles[twiddle_idx];
        shared[i] = sum;
        shared[j] = m31_mul(diff, w_inv);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write back
    if (base + tid < n)
        data[base_offset + base + tid] = shared[tid];
    if (base + tid + half_block < n)
        data[base_offset + base + tid + half_block] = shared[tid + half_block];
}

// Zero-pad kernel for LDE: copies coefficients to extended buffer with zero padding
// Assumes input buffer has N elements (coefficients after INTT), output has M elements (M >= N)
kernel void batch_circle_ntt_zero_pad(
    device const M31* input         [[buffer(0)]],
    device M31* output              [[buffer(1)]],
    constant uint& n_input          [[buffer(2)]],    // original size N
    constant uint& n_output         [[buffer(3)]],     // extended size M
    constant uint& num_cols         [[buffer(4)]],
    uint gid                        [[thread_position_in_grid]],
    uint gid_y                      [[thread_position_in_grid]]
) {
    uint col = gid_y;
    if (col >= num_cols) return;

    uint input_offset = col * n_input;
    uint output_offset = col * n_output;

    if (gid < n_input) {
        output[output_offset + gid] = input[input_offset + gid];
    } else if (gid < n_output) {
        output[output_offset + gid] = m31_zero();
    }
}

// Batch transpose kernel for four-step FFT
kernel void batch_circle_ntt_transpose(
    device const M31* input         [[buffer(0)]],
    device M31* output              [[buffer(1)]],
    constant uint& n_side           [[buffer(2)]],
    constant uint& num_cols         [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]],
    uint gid_y                      [[thread_position_in_grid]]
) {
    uint col = gid_y;
    if (col >= num_cols) return;

    uint total = n_side * n_side;
    if (gid >= total) return;

    uint row = gid / n_side;
    uint col_idx = gid % n_side;

    uint input_offset = col * total;
    uint output_offset = col * total;

    output[output_offset + col_idx * n_side + row] = input[input_offset + row * n_side + col_idx];
}

// Fused INTT + zero-pad + forward NTT for batch LDE
// This is the core kernel for EVMetal trace commitment
kernel void batch_circle_ntt_lde(
    device M31* data                [[buffer(0)]],
    device const M31* inv_twiddles  [[buffer(1)]],
    device const M31* fwd_twiddles  [[buffer(2)]],
    device const M31* inv_n         [[buffer(3)]],
    device const M31* coset_powers  [[buffer(4)]],   // shift^i for coset LDE
    constant uint& n_trace          [[buffer(5)]],    // trace size N
    constant uint& n_eval           [[buffer(6)]],    // evaluation size M
    constant uint& log_n_trace      [[buffer(7)]],
    constant uint& log_n_eval       [[buffer(8)]],
    constant uint& num_cols         [[buffer(9)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint tg_size                    [[threads_per_threadgroup]]
) {
    uint col = tgid;
    if (col >= num_cols) return;

    uint col_offset_trace = col * n_trace;
    uint col_offset_eval = col * n_eval;

    // This is a simplified version - full implementation would:
    // 1. Run INTT on trace domain (n_trace elements)
    // 2. Zero-pad to n_eval elements
    // 3. Apply coset shift (multiply by coset_powers[i])
    // 4. Run forward NTT on evaluation domain

    // For now, just do the forward NTT on the trace domain
    // The caller should handle INTT and padding separately
    threadgroup M31 shared[1024];

    uint rev_tid = bitrev(tid, log_n_trace);
    uint rev_hi = bitrev(tid + tg_size, log_n_trace);

    if (rev_tid < n_trace)
        shared[rev_tid] = data[col_offset_trace + tid];
    if (rev_hi < n_trace)
        shared[rev_hi] = data[col_offset_trace + tid + tg_size];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // DIT stages
    for (uint s = 0; s < log_n_trace; s++) {
        uint h = 1u << s;
        uint bs = h << 1;
        uint block_idx = tid / h;
        uint local_idx = tid % h;
        uint i = block_idx * bs + local_idx;
        uint j = i + h;

        uint twiddle_idx = local_idx * (n_trace / bs);
        uint global_block_size = 1u << (s + 1);

        M31 a = shared[i];
        M31 b = shared[j];
        if (twiddle_idx == 0) {
            shared[i] = m31_add(a, b);
            shared[j] = m31_sub(a, b);
        } else {
            M31 w = fwd_twiddles[twiddle_idx];
            shared[i] = m31_add(a, m31_mul(w, b));
            shared[j] = m31_sub(a, m31_mul(w, b));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < n_trace)
        data[col_offset_trace + tid] = shared[tid];
    if (tid + tg_size < n_trace)
        data[col_offset_trace + tid + tg_size] = shared[tid + tg_size];
}

// Bit reversal helper
inline uint bitrev(uint val, uint num_bits) {
    uint rev = 0;
    for (uint i = 0; i < num_bits; i++) {
        rev = (rev << 1) | (val & 1);
        val >>= 1;
    }
    return rev;
}