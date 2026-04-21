// P^1 Rational Function NTT kernels for Mersenne31 field (p = 2^31 - 1)
// Uses standard radix-2 butterflies on multiplicative coset domain.
//
// Unlike Circle NTT which uses y/x-coordinate twiddles, P^1 NTT uses:
// - Standard root-of-unity twiddles (when available)
// - Twiddle precomputation based on domain structure
//
// Forward butterfly (DIT): (a, b) -> (a + w*b, a - w*b)
// Inverse butterfly (DIF): (a, b) -> (a + b, (a - b) / w)

#include "../fields/mersenne31.metal"

// --- Bit-reversal helper for fused kernels ---
inline uint bitrev(uint val, uint num_bits) {
    uint rev = 0;
    for (uint i = 0; i < num_bits; i++) {
        rev = (rev << 1) | (val & 1);
        val >>= 1;
    }
    return rev;
}

// Forward P^1 NTT butterfly (DIT): (a, b) -> (a + tw*b, a - tw*b)
kernel void p1_ntt_butterfly(
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

// Inverse P^1 NTT butterfly (DIF): (a, b) -> (a + b, (a - b) / tw)
kernel void p1_intt_butterfly(
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

// Scale kernel: multiply all elements by a scalar (for 1/N)
kernel void p1_ntt_scale(
    device M31* data                [[buffer(0)]],
    device const M31* scalar        [[buffer(1)]],
    constant uint& n                [[buffer(2)]],
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    data[gid] = m31_mul(data[gid], scalar[0]);
}

// --- Fused P^1 NTT kernel: process multiple DIT stages in threadgroup memory ---
// Each threadgroup loads BLOCK_SIZE elements, performs local_stages butterfly stages
// in shared memory, then writes back.
// local_stages = log2(BLOCK_SIZE) = number of stages that fit in threadgroup.
// This replaces local_stages individual dispatches with a single dispatch.
kernel void p1_ntt_butterfly_fused(
    device M31* data                [[buffer(0)]],
    device const M31* twiddles      [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& local_stages     [[buffer(3)]],  // how many stages to fuse
    constant uint& stage_offset     [[buffer(4)]],  // first stage index
    uint tid                        [[thread_index_in_threadgroup]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint tg_size                    [[threads_per_threadgroup]]
) {
    // Block size = 2 * tg_size (each thread handles one butterfly = 2 elements)
    uint block_size = tg_size << 1;
    uint base = tgid * block_size;

    // Load into threadgroup memory (max 1024 M31 elements = 4KB for M31)
    threadgroup M31 shared[1024];
    if (base + tid < n)
        shared[tid] = data[base + tid];
    if (base + tid + tg_size < n)
        shared[tid + tg_size] = data[base + tid + tg_size];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Perform butterfly stages in shared memory
    for (uint s = 0; s < local_stages; s++) {
        uint stage = stage_offset + s;
        uint half_block = 1u << s;  // local half_block within the threadgroup
        uint local_block_size = half_block << 1;

        // Cumulative DIT permutation after stages 0..s-1 is: p ^ ((1<<s) - 1)
        // To find the original position before any permutation, reverse this:
        // original_tid = tid ^ ((1<<s) - 1)
        uint perm_mask = ((1u << s) - 1);
        uint original_tid = tid ^ perm_mask;

        uint block_idx = original_tid / half_block;
        uint local_idx = original_tid % half_block;
        uint i = block_idx * local_block_size + local_idx;
        uint j = i + half_block;

        // Global twiddle index: for DIT stage s, the stride in the full array
        uint global_block_size = 1u << (stage + 1);
        uint twiddle_idx = local_idx * (n / global_block_size);

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
        data[base + tid] = shared[tid];
    if (base + tid + tg_size < n)
        data[base + tid + tg_size] = shared[tid + tg_size];
}

// --- Fused inverse P^1 NTT kernel: process multiple DIF stages in threadgroup memory ---
kernel void p1_intt_butterfly_fused(
    device M31* data                [[buffer(0)]],
    device const M31* inv_twiddles  [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& local_stages     [[buffer(3)]],  // how many stages to fuse
    constant uint& stage_offset     [[buffer(4)]],  // first stage index (from high)
    uint tid                        [[thread_index_in_threadgroup]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint tg_size                    [[threads_per_threadgroup]]
) {
    // For DIF, we process from high stages to low (stage_offset is the highest stage in this group)
    uint block_size = tg_size << 1;
    uint base = tgid * block_size;

    // Load into threadgroup memory
    threadgroup M31 shared[1024];
    if (base + tid < n)
        shared[tid] = data[base + tid];
    if (base + tid + tg_size < n)
        shared[tid + tg_size] = data[base + tid + tg_size];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Perform butterfly stages in shared memory (DIF: high to low)
    for (uint s = 0; s < local_stages; s++) {
        // stage_offset - s because we're going from high to low
        uint stage = stage_offset - s;
        uint half_block = 1u << s;  // local half_block within the threadgroup
        uint local_block_size = half_block << 1;

        // Cumulative DIF permutation after s+1 stages: p ^ ((1<<(s+1)) - 1)
        // Reverse it to find original position before any permutation
        uint perm_mask = ((1u << (s + 1)) - 1);
        uint original_tid = tid ^ perm_mask;
        uint block_idx = original_tid / half_block;
        uint local_idx = original_tid % half_block;
        uint i = block_idx * local_block_size + local_idx;
        uint j = i + half_block;

        // Global twiddle index for DIF: use block_idx (butterfly index within block)
        uint global_block_size = 1u << (stage + 1);
        uint twiddle_idx = block_idx * (n / global_block_size);

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
        data[base + tid] = shared[tid];
    if (base + tid + tg_size < n)
        data[base + tid + tg_size] = shared[tid + tg_size];
}

// ===== Four-step FFT kernels =====
// For P^1 four-step FFT, data is conceptually N1×N2 (row×col) in standard layout.
// After transpose: N2×N1 (N2 rows of N1 contiguous elements each).
// "Column FFT" = FFT on each column of the N2×N1 matrix (coalesced via stride N2).
// "Row FFT" = FFT on each row of the N2×N1 matrix (contiguous within each row).

// --- Column DIT FFT: each threadgroup handles one column (size N1), N2 columns total ---
// Data layout: N2 rows × N1 cols (row-major). Column c at element r is data[r*N1 + c].
// This kernel reads with stride N2 (data[col + tid * n2]), computing N1-point DIT per column.
kernel void p1_ntt_column_fused(
    device M31* data                [[buffer(0)]],
    device const M31* twiddles      [[buffer(1)]],
    constant uint& n                [[buffer(2)]],    // total size N = N1 * N2
    constant uint& n1               [[buffer(3)]],    // column FFT size
    constant uint& n2               [[buffer(4)]],    // number of columns
    constant uint& local_stages     [[buffer(5)]],    // log2(sub-block size)
    uint tid                        [[thread_index_in_threadgroup]],
    uint tgid                       [[threadgroup_position_in_grid]],  // col index
    uint tg_size                    [[threads_per_threadgroup]]
) {
    uint col = tgid;
    threadgroup M31 shared[1024];

    uint idx_lo = tid;
    uint idx_hi = tid + tg_size;
    uint rev_lo = bitrev(idx_lo, local_stages);
    uint rev_hi = bitrev(idx_hi, local_stages);

    // Load: data[col + idx * n2] = element at row idx, column col
    if (idx_lo < n1)
        shared[rev_lo] = data[col + idx_lo * n2];
    if (idx_hi < n1)
        shared[rev_hi] = data[col + idx_hi * n2];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 0; s < local_stages; s++) {
        uint half_block = 1u << s;
        uint local_block_size = half_block << 1;

        // Cumulative DIT permutation after stages 0..s-1: p ^ ((1<<s) - 1)
        // Reverse it to find original position before any permutation
        uint perm_mask = ((1u << s) - 1);
        uint original_tid = tid ^ perm_mask;
        uint block_idx = original_tid / half_block;
        uint local_idx = original_tid % half_block;
        uint i = block_idx * local_block_size + local_idx;
        uint j = i + half_block;

        uint twiddle_idx = local_idx * (n1 / local_block_size) * n2;

        M31 a = shared[i];
        M31 b = shared[j];
        if (twiddle_idx == 0) {
            shared[i] = m31_add(a, b);
            shared[j] = m31_sub(a, b);
        } else {
            M31 w = twiddles[twiddle_idx];
            M31 wb = m31_mul(w, b);
            shared[i] = m31_add(a, wb);
            shared[j] = m31_sub(a, wb);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < n1) data[col + tid * n2] = shared[tid];
    if (tid + tg_size < n1) data[col + (tid + tg_size) * n2] = shared[tid + tg_size];
}

// --- Twiddle multiply for four-step (diagonal twiddles between column and row FFTs) ---
kernel void p1_ntt_twiddle_fourstep(
    device M31* data                [[buffer(0)]],
    device const M31* twiddles      [[buffer(1)]],
    constant uint& n                [[buffer(2)]],    // total size N
    constant uint& n1               [[buffer(3)]],    // first dimension
    constant uint& n2               [[buffer(4)]],    // second dimension
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    uint row = gid / n1;
    uint col = gid % n1;
    uint idx = (uint)((ulong(row) * ulong(col)) % ulong(n));
    M31 w = twiddles[idx];
    data[gid] = m31_mul(data[gid], w);
}

// --- Row DIT FFT with twiddle during load ---
kernel void p1_ntt_row_fused_twiddle(
    device M31* data                [[buffer(0)]],
    device const M31* twiddles      [[buffer(1)]],
    constant uint& n                [[buffer(2)]],    // total size N = N1 * N2
    constant uint& local_stages     [[buffer(3)]],    // log2(sub-block size) for N2
    uint tid                        [[thread_index_in_threadgroup]],
    uint tgid                       [[threadgroup_position_in_grid]],  // row index
    uint tg_size                    [[threads_per_threadgroup]]
) {
    uint block_size = tg_size << 1;
    uint base = tgid * block_size;
    threadgroup M31 shared[1024];

    uint idx_lo = tid;
    uint idx_hi = tid + tg_size;
    uint rev_lo = bitrev(idx_lo, local_stages);
    uint rev_hi = bitrev(idx_hi, local_stages);

    // Load with twiddle multiply: val *= omega_N^(row * col)
    if (base + idx_lo < n) {
        M31 val = data[base + idx_lo];
        uint tw_idx = (uint)((ulong(tgid) * ulong(idx_lo)) % ulong(n));
        if (tw_idx != 0) val = m31_mul(val, twiddles[tw_idx]);
        shared[rev_lo] = val;
    }
    if (base + idx_hi < n) {
        M31 val = data[base + idx_hi];
        uint tw_idx = (uint)((ulong(tgid) * ulong(idx_hi)) % ulong(n));
        if (tw_idx != 0) val = m31_mul(val, twiddles[tw_idx]);
        shared[rev_hi] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 0; s < local_stages; s++) {
        uint half_block = 1u << s;
        uint local_block_size = half_block << 1;

        // Cumulative DIT permutation after stages 0..s-1: p ^ ((1<<s) - 1)
        // Reverse it to find original position before any permutation
        uint perm_mask = ((1u << s) - 1);
        uint original_tid = tid ^ perm_mask;
        uint block_idx = original_tid / half_block;
        uint local_idx = original_tid % half_block;
        uint i = block_idx * local_block_size + local_idx;
        uint j = i + half_block;

        uint global_block_size = 1u << (s + 1);
        uint twiddle_idx = local_idx * (n / global_block_size);

        M31 a = shared[i];
        M31 b = shared[j];
        if (twiddle_idx == 0) {
            shared[i] = m31_add(a, b);
            shared[j] = m31_sub(a, b);
        } else {
            M31 w = twiddles[twiddle_idx];
            M31 wb = m31_mul(w, b);
            shared[i] = m31_add(a, wb);
            shared[j] = m31_sub(a, wb);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (base + tid < n) data[base + tid] = shared[tid];
    if (base + tid + tg_size < n) data[base + tid + tg_size] = shared[tid + tg_size];
}

// --- Out-of-place square transpose for M31 ---
kernel void p1_ntt_transpose_outofplace(
    device const M31* input         [[buffer(0)]],
    device M31* output              [[buffer(1)]],
    constant uint& n_side           [[buffer(2)]],
    uint gid                        [[thread_position_in_grid]]
) {
    uint total = n_side * n_side;
    if (gid >= total) return;

    uint row = gid / n_side;
    uint col = gid % n_side;
    output[col * n_side + row] = input[gid];
}

// --- Inverse column DIF FFT: each threadgroup handles one column (size N1), N2 columns total ---
kernel void p1_intt_column_fused(
    device M31* data                [[buffer(0)]],
    device const M31* inv_twiddles  [[buffer(1)]],
    constant uint& n                [[buffer(2)]],
    constant uint& n1               [[buffer(3)]],
    constant uint& n2               [[buffer(4)]],
    constant uint& local_stages     [[buffer(5)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint tg_size                    [[threads_per_threadgroup]]
) {
    uint col = tgid;
    threadgroup M31 shared[1024];

    uint idx_lo = tid;
    uint idx_hi = tid + tg_size;
    uint rev_lo = bitrev(idx_lo, local_stages);
    uint rev_hi = bitrev(idx_hi, local_stages);

    if (idx_lo < n1)
        shared[rev_lo] = data[col + idx_lo * n2];
    if (idx_hi < n1)
        shared[rev_hi] = data[col + idx_hi * n2];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 0; s < local_stages; s++) {
        uint stage = local_stages - 1 - s;
        uint half_block = 1u << s;
        uint local_block_size = half_block << 1;

        // Cumulative DIF permutation after s+1 stages: p ^ ((1<<(s+1)) - 1)
        // Reverse it to find original position before any permutation
        uint perm_mask = ((1u << (s + 1)) - 1);
        uint original_tid = tid ^ perm_mask;
        uint block_idx = original_tid / half_block;
        uint local_idx = original_tid % half_block;
        uint i = block_idx * local_block_size + local_idx;
        uint j = i + half_block;

        uint global_block_size = 1u << (stage + 1);
        uint twiddle_idx = local_idx * (n1 / global_block_size) * n2;

        M31 a = shared[i];
        M31 b = shared[j];
        M31 sum = m31_add(a, b);
        M31 diff = m31_sub(a, b);
        if (twiddle_idx == 0) {
            shared[i] = sum;
            shared[j] = diff;
        } else {
            M31 w_inv = inv_twiddles[twiddle_idx];
            shared[i] = sum;
            shared[j] = m31_mul(diff, w_inv);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < n1) data[col + tid * n2] = shared[tid];
    if (tid + tg_size < n1) data[col + (tid + tg_size) * n2] = shared[tid + tg_size];
}

// --- Inverse row DIF FFT with inverse twiddle during load ---
kernel void p1_intt_row_fused_twiddle(
    device M31* data                [[buffer(0)]],
    device const M31* inv_twiddles  [[buffer(1)]],
    constant uint& n                [[buffer(2)]],
    constant uint& local_stages     [[buffer(3)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint tg_size                    [[threads_per_threadgroup]]
) {
    uint block_size = tg_size << 1;
    uint base = tgid * block_size;
    threadgroup M31 shared[1024];

    uint idx_lo = tid;
    uint idx_hi = tid + tg_size;
    uint rev_lo = bitrev(idx_lo, local_stages);
    uint rev_hi = bitrev(idx_hi, local_stages);

    if (base + idx_lo < n) {
        M31 val = data[base + idx_lo];
        uint tw_idx = (uint)((ulong(tgid) * ulong(idx_lo)) % ulong(n));
        if (tw_idx != 0) val = m31_mul(val, inv_twiddles[tw_idx]);
        shared[rev_lo] = val;
    }
    if (base + idx_hi < n) {
        M31 val = data[base + idx_hi];
        uint tw_idx = (uint)((ulong(tgid) * ulong(idx_hi)) % ulong(n));
        if (tw_idx != 0) val = m31_mul(val, inv_twiddles[tw_idx]);
        shared[rev_hi] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 0; s < local_stages; s++) {
        uint stage = local_stages - 1 - s;
        uint half_block = 1u << s;
        uint local_block_size = half_block << 1;

        // Cumulative DIF permutation after s+1 stages: p ^ ((1<<(s+1)) - 1)
        // Reverse it to find original position before any permutation
        uint perm_mask = ((1u << (s + 1)) - 1);
        uint original_tid = tid ^ perm_mask;
        uint block_idx = original_tid / half_block;
        uint local_idx = original_tid % half_block;
        uint i = block_idx * local_block_size + local_idx;
        uint j = i + half_block;

        uint global_block_size = 1u << (stage + 1);
        uint twiddle_idx = local_idx * (n / global_block_size);

        M31 a = shared[i];
        M31 b = shared[j];
        M31 sum = m31_add(a, b);
        M31 diff = m31_sub(a, b);
        if (twiddle_idx == 0) {
            shared[i] = sum;
            shared[j] = diff;
        } else {
            M31 w_inv = inv_twiddles[twiddle_idx];
            shared[i] = sum;
            shared[j] = m31_mul(diff, w_inv);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (base + tid < n) data[base + tid] = shared[tid];
    if (base + tid + tg_size < n) data[base + tid + tg_size] = shared[tid + tg_size];
}
