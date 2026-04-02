// NTT kernels for BabyBear field (p = 0x78000001)
// Single u32 per element — 8x denser than BN254 Fr

#include "../fields/babybear.metal"

kernel void bb_ntt_butterfly(
    device Bb* data                [[buffer(0)]],
    device const Bb* twiddles      [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& stage           [[buffer(3)]],
    uint gid                       [[thread_position_in_grid]]
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

    Bb a = data[i];
    Bb b = data[j];
    Bb w = twiddles[twiddle_idx];
    Bb wb = bb_mul(w, b);

    data[i] = bb_add(a, wb);
    data[j] = bb_sub(a, wb);
}

kernel void bb_intt_butterfly(
    device Bb* data                [[buffer(0)]],
    device const Bb* twiddles_inv  [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& stage           [[buffer(3)]],
    uint gid                       [[thread_position_in_grid]]
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

    Bb a = data[i];
    Bb b = data[j];
    Bb sum = bb_add(a, b);
    Bb diff = bb_sub(a, b);
    Bb w = twiddles_inv[twiddle_idx];

    data[i] = sum;
    data[j] = bb_mul(diff, w);
}

// Radix-4 DIT butterfly: processes 2 stages at once, halving global memory passes.
kernel void bb_ntt_butterfly_radix4(
    device Bb* data                [[buffer(0)]],
    device const Bb* twiddles      [[buffer(1)]],
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

    Bb a0 = data[base];
    Bb a1 = data[base + h];
    Bb a2 = data[base + 2 * h];
    Bb a3 = data[base + 3 * h];

    Bb ws = twiddles[local_idx * (n / (2 * h))];
    Bb ws_a1 = bb_mul(ws, a1);
    Bb ws_a3 = bb_mul(ws, a3);
    Bb b0 = bb_add(a0, ws_a1);
    Bb b1 = bb_sub(a0, ws_a1);
    Bb b2 = bb_add(a2, ws_a3);
    Bb b3 = bb_sub(a2, ws_a3);

    Bb w_lo = twiddles[local_idx * (n / block4)];
    Bb w_hi = twiddles[(local_idx + h) * (n / block4)];
    Bb wb2 = bb_mul(w_lo, b2);
    Bb wb3 = bb_mul(w_hi, b3);

    data[base]         = bb_add(b0, wb2);
    data[base + 2 * h] = bb_sub(b0, wb2);
    data[base + h]     = bb_add(b1, wb3);
    data[base + 3 * h] = bb_sub(b1, wb3);
}

// Radix-4 DIF butterfly for iNTT: processes 2 stages at once.
kernel void bb_intt_butterfly_radix4(
    device Bb* data                [[buffer(0)]],
    device const Bb* twiddles_inv  [[buffer(1)]],
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

kernel void bb_ntt_scale(
    device Bb* data                [[buffer(0)]],
    device const Bb* scale_factor  [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    data[gid] = bb_mul(data[gid], scale_factor[0]);
}

kernel void bb_ntt_bitrev(
    device Bb* data                [[buffer(0)]],
    constant uint& n               [[buffer(1)]],
    constant uint& log_n           [[buffer(2)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    uint rev = 0;
    uint val = gid;
    for (uint i = 0; i < log_n; i++) {
        rev = (rev << 1) | (val & 1);
        val >>= 1;
    }
    if (gid < rev) {
        Bb tmp = data[gid];
        data[gid] = data[rev];
        data[rev] = tmp;
    }
}

// Fused butterfly kernel — process multiple stages in threadgroup memory
// Each BabyBear element is 4 bytes (vs 32 bytes for BN254 Fr),
// so we can fit 8x more elements: up to 8192 in 32KB shared memory
kernel void bb_ntt_butterfly_fused(
    device Bb* data                [[buffer(0)]],
    device const Bb* twiddles      [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& local_stages    [[buffer(3)]],
    constant uint& stage_offset    [[buffer(4)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint block_size = tg_size << 1;
    uint base = tgid * block_size;

    threadgroup Bb shared[8192];  // 8192 * 4 bytes = 32KB
    if (base + tid < n) shared[tid] = data[base + tid];
    if (base + tid + tg_size < n) shared[tid + tg_size] = data[base + tid + tg_size];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 0; s < local_stages; s++) {
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

    if (base + tid < n) data[base + tid] = shared[tid];
    if (base + tid + tg_size < n) data[base + tid + tg_size] = shared[tid + tg_size];
}

// Fused inverse butterfly kernel — DIF stages in threadgroup memory
kernel void bb_intt_butterfly_fused(
    device Bb* data                [[buffer(0)]],
    device const Bb* twiddles_inv  [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& local_stages    [[buffer(3)]],
    constant uint& stage_offset    [[buffer(4)]],
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

    // DIF: stages go from high to low
    for (uint s = 0; s < local_stages; s++) {
        uint stage = stage_offset - s;
        uint half_block = 1u << (local_stages - 1 - s);
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

// Fused bitrev + DIT butterfly: read from input with bit-reversed indexing,
// compute fused stages in shared memory, write to output buffer.
// Eliminates separate bitrev dispatch.
kernel void bb_ntt_butterfly_fused_bitrev(
    device const Bb* input         [[buffer(0)]],   // source (natural order)
    device Bb* output              [[buffer(1)]],   // destination
    device const Bb* twiddles      [[buffer(2)]],
    constant uint& n               [[buffer(3)]],
    constant uint& local_stages    [[buffer(4)]],
    constant uint& logN            [[buffer(5)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint block_size = tg_size << 1;
    uint base = tgid * block_size;

    threadgroup Bb shared[8192];

    // Load from input with bit-reversed global indices
    uint idx_lo = tid;
    uint idx_hi = tid + tg_size;
    uint global_lo = base + idx_lo;
    uint global_hi = base + idx_hi;

    // bit-reverse the full logN-bit global index, then load
    uint rev_lo = 0, rev_hi = 0;
    { uint v = global_lo; for (uint i = 0; i < logN; i++) { rev_lo = (rev_lo << 1) | (v & 1); v >>= 1; } }
    { uint v = global_hi; for (uint i = 0; i < logN; i++) { rev_hi = (rev_hi << 1) | (v & 1); v >>= 1; } }

    if (global_lo < n) shared[idx_lo] = input[rev_lo];
    if (global_hi < n) shared[idx_hi] = input[rev_hi];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // DIT butterfly stages (stage 0 through local_stages-1)
    for (uint s = 0; s < local_stages; s++) {
        uint half_block = 1u << s;
        uint local_block_size = half_block << 1;
        uint block_idx = tid / half_block;
        uint local_idx = tid % half_block;
        uint i = block_idx * local_block_size + local_idx;
        uint j = i + half_block;

        uint global_block_size = 1u << (s + 1);
        uint twiddle_idx = local_idx * (n / global_block_size);

        Bb a = shared[i];
        Bb b = shared[j];
        Bb w = twiddles[twiddle_idx];
        Bb wb = bb_mul(w, b);
        shared[i] = bb_add(a, wb);
        shared[j] = bb_sub(a, wb);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write to output (sequential positions)
    if (global_lo < n) output[global_lo] = shared[idx_lo];
    if (global_hi < n) output[global_hi] = shared[idx_hi];
}

// --- Four-step FFT kernels for BabyBear ---

inline uint bb_bitrev(uint val, uint num_bits) {
    uint rev = 0;
    for (uint i = 0; i < num_bits; i++) {
        rev = (rev << 1) | (val & 1);
        val >>= 1;
    }
    return rev;
}

// Column DIT FFT with bit-reversed loading
// Each threadgroup processes one column of the n1×n2 matrix
kernel void bb_ntt_column_fused(
    device Bb* data                [[buffer(0)]],
    device const Bb* twiddles      [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& n1              [[buffer(3)]],
    constant uint& n2              [[buffer(4)]],
    constant uint& local_stages    [[buffer(5)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint col = tgid;
    threadgroup Bb shared_data[8192];

    uint idx_lo = tid;
    uint idx_hi = tid + tg_size;
    uint rev_lo = bb_bitrev(idx_lo, local_stages);
    uint rev_hi = bb_bitrev(idx_hi, local_stages);

    if (idx_lo < n1)
        shared_data[rev_lo] = data[col + idx_lo * n2];
    if (idx_hi < n1)
        shared_data[rev_hi] = data[col + idx_hi * n2];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 0; s < local_stages; s++) {
        uint half_block = 1u << s;
        uint local_block_size = half_block << 1;
        uint block_idx = tid / half_block;
        uint local_idx = tid % half_block;
        uint i = block_idx * local_block_size + local_idx;
        uint j = i + half_block;

        uint twiddle_idx = local_idx * (n1 / local_block_size) * n2;

        Bb a = shared_data[i];
        Bb b = shared_data[j];
        Bb w = twiddles[twiddle_idx];
        Bb wb = bb_mul(w, b);
        shared_data[i] = bb_add(a, wb);
        shared_data[j] = bb_sub(a, wb);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < n1) data[col + tid * n2] = shared_data[tid];
    if (tid + tg_size < n1) data[col + (tid + tg_size) * n2] = shared_data[tid + tg_size];
}

// Row DIT FFT with bit-reversed loading
kernel void bb_ntt_row_fused(
    device Bb* data                [[buffer(0)]],
    device const Bb* twiddles      [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& local_stages    [[buffer(3)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint block_size = tg_size << 1;
    uint base = tgid * block_size;
    threadgroup Bb shared_data[8192];

    uint idx_lo = tid;
    uint idx_hi = tid + tg_size;
    uint rev_lo = bb_bitrev(idx_lo, local_stages);
    uint rev_hi = bb_bitrev(idx_hi, local_stages);

    if (base + idx_lo < n)
        shared_data[rev_lo] = data[base + idx_lo];
    if (base + idx_hi < n)
        shared_data[rev_hi] = data[base + idx_hi];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 0; s < local_stages; s++) {
        uint half_block = 1u << s;
        uint local_block_size = half_block << 1;
        uint block_idx = tid / half_block;
        uint local_idx = tid % half_block;
        uint i = block_idx * local_block_size + local_idx;
        uint j = i + half_block;

        uint global_block_size = 1u << (s + 1);
        uint twiddle_idx = local_idx * (n / global_block_size);

        Bb a = shared_data[i];
        Bb b = shared_data[j];
        Bb w = twiddles[twiddle_idx];
        Bb wb = bb_mul(w, b);
        shared_data[i] = bb_add(a, wb);
        shared_data[j] = bb_sub(a, wb);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (base + tid < n) data[base + tid] = shared_data[tid];
    if (base + tid + tg_size < n) data[base + tid + tg_size] = shared_data[tid + tg_size];
}

// Twiddle multiply: data[row*N2+col] *= twiddles[(row*col) % N]
kernel void bb_ntt_twiddle_multiply(
    device Bb* data                [[buffer(0)]],
    device const Bb* twiddles      [[buffer(1)]],
    constant uint& n2              [[buffer(2)]],
    constant uint& n               [[buffer(3)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    uint row = gid / n2;
    uint col = gid % n2;
    uint twiddle_idx = (uint)((ulong(row) * ulong(col)) % ulong(n));
    if (twiddle_idx == 0) return;
    data[gid] = bb_mul(data[gid], twiddles[twiddle_idx]);
}

// Fused row FFT + twiddle multiply: applies twiddle during load, eliminating separate twiddle pass.
// Saves one full memory pass over the data array.
kernel void bb_ntt_row_fused_twiddle(
    device Bb* data                [[buffer(0)]],
    device const Bb* twiddles      [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& local_stages    [[buffer(3)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint block_size = tg_size << 1;
    uint base = tgid * block_size;
    threadgroup Bb shared_data[8192];

    uint idx_lo = tid;
    uint idx_hi = tid + tg_size;
    uint rev_lo = bb_bitrev(idx_lo, local_stages);
    uint rev_hi = bb_bitrev(idx_hi, local_stages);

    // Load with twiddle multiply: val *= omega_N^(row * col)
    if (base + idx_lo < n) {
        Bb val = data[base + idx_lo];
        uint tw_idx = (uint)((ulong(tgid) * ulong(idx_lo)) % ulong(n));
        if (tw_idx != 0) val = bb_mul(val, twiddles[tw_idx]);
        shared_data[rev_lo] = val;
    }
    if (base + idx_hi < n) {
        Bb val = data[base + idx_hi];
        uint tw_idx = (uint)((ulong(tgid) * ulong(idx_hi)) % ulong(n));
        if (tw_idx != 0) val = bb_mul(val, twiddles[tw_idx]);
        shared_data[rev_hi] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 0; s < local_stages; s++) {
        uint half_block = 1u << s;
        uint local_block_size = half_block << 1;
        uint block_idx = tid / half_block;
        uint local_idx = tid % half_block;
        uint i = block_idx * local_block_size + local_idx;
        uint j = i + half_block;

        uint global_block_size = 1u << (s + 1);
        uint twiddle_idx = local_idx * (n / global_block_size);

        Bb a = shared_data[i];
        Bb b = shared_data[j];
        Bb w = twiddles[twiddle_idx];
        Bb wb = bb_mul(w, b);
        shared_data[i] = bb_add(a, wb);
        shared_data[j] = bb_sub(a, wb);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (base + tid < n) data[base + tid] = shared_data[tid];
    if (base + tid + tg_size < n) data[base + tid + tg_size] = shared_data[tid + tg_size];
}

// In-place square matrix transpose
kernel void bb_ntt_transpose(
    device Bb* data                [[buffer(0)]],
    constant uint& n_side          [[buffer(1)]],
    uint gid                       [[thread_position_in_grid]]
) {
    uint total = n_side * n_side;
    if (gid >= total) return;
    uint row = gid / n_side;
    uint col = gid % n_side;
    if (row >= col) return;
    uint i = row * n_side + col;
    uint j = col * n_side + row;
    Bb tmp = data[i];
    data[i] = data[j];
    data[j] = tmp;
}

// Fused column iFFT + inverse twiddle + scale: loads with inv twiddle multiply,
// does DIF stages, writes back with 1/N scale. Eliminates 2 memory passes.
kernel void bb_intt_column_fused_twiddle(
    device Bb* data                [[buffer(0)]],
    device const Bb* twiddles_inv  [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& n1              [[buffer(3)]],
    constant uint& n2              [[buffer(4)]],
    constant uint& local_stages    [[buffer(5)]],
    device const Bb* inv_n         [[buffer(6)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint col = tgid;
    threadgroup Bb shared_data[8192];
    Bb scale = inv_n[0];

    // Load with inverse twiddle: val *= omega_N^(-(row*col))
    if (tid < n1) {
        Bb val = data[col + tid * n2];
        uint tw_idx = (uint)((ulong(tid) * ulong(col)) % ulong(n));
        if (tw_idx != 0) {
            // inv_twiddle[k] = twiddles_inv[k] for the inverse twiddle
            val = bb_mul(val, twiddles_inv[tw_idx]);
        }
        shared_data[tid] = val;
    }
    if (tid + tg_size < n1) {
        Bb val = data[col + (tid + tg_size) * n2];
        uint tw_idx = (uint)((ulong(tid + tg_size) * ulong(col)) % ulong(n));
        if (tw_idx != 0) {
            val = bb_mul(val, twiddles_inv[tw_idx]);
        }
        shared_data[tid + tg_size] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // DIF butterfly stages
    for (uint s = 0; s < local_stages; s++) {
        uint half_block = 1u << (local_stages - 1 - s);
        uint local_block_size = half_block << 1;
        uint block_idx = tid / half_block;
        uint local_idx = tid % half_block;
        uint i = block_idx * local_block_size + local_idx;
        uint j = i + half_block;

        uint twiddle_idx = local_idx * (n1 / local_block_size) * n2;

        Bb a = shared_data[i];
        Bb b = shared_data[j];
        Bb sum = bb_add(a, b);
        Bb diff = bb_sub(a, b);
        Bb w = twiddles_inv[twiddle_idx];
        shared_data[i] = sum;
        shared_data[j] = bb_mul(diff, w);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write back with bit-reversal and 1/N scale
    uint rev_lo = bb_bitrev(tid, local_stages);
    uint rev_hi = bb_bitrev(tid + tg_size, local_stages);
    if (tid < n1) data[col + tid * n2] = bb_mul(shared_data[rev_lo], scale);
    if (tid + tg_size < n1) data[col + (tid + tg_size) * n2] = bb_mul(shared_data[rev_hi], scale);
}

// DIF column iFFT with fused 1/N scale: eliminates separate scale pass
kernel void bb_intt_column_fused_scale(
    device Bb* data                [[buffer(0)]],
    device const Bb* twiddles_inv  [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& n1              [[buffer(3)]],
    constant uint& n2              [[buffer(4)]],
    constant uint& local_stages    [[buffer(5)]],
    device const Bb* inv_n         [[buffer(6)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint col = tgid;
    threadgroup Bb shared_data[8192];
    Bb scale = inv_n[0];

    if (tid < n1) shared_data[tid] = data[col + tid * n2];
    if (tid + tg_size < n1) shared_data[tid + tg_size] = data[col + (tid + tg_size) * n2];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 0; s < local_stages; s++) {
        uint half_block = 1u << (local_stages - 1 - s);
        uint local_block_size = half_block << 1;
        uint block_idx = tid / half_block;
        uint local_idx = tid % half_block;
        uint i = block_idx * local_block_size + local_idx;
        uint j = i + half_block;

        uint twiddle_idx = local_idx * (n1 / local_block_size) * n2;

        Bb a = shared_data[i];
        Bb b = shared_data[j];
        Bb sum = bb_add(a, b);
        Bb diff = bb_sub(a, b);
        Bb w = twiddles_inv[twiddle_idx];
        shared_data[i] = sum;
        shared_data[j] = bb_mul(diff, w);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint rev_lo = bb_bitrev(tid, local_stages);
    uint rev_hi = bb_bitrev(tid + tg_size, local_stages);
    if (tid < n1) data[col + tid * n2] = bb_mul(shared_data[rev_lo], scale);
    if (tid + tg_size < n1) data[col + (tid + tg_size) * n2] = bb_mul(shared_data[rev_hi], scale);
}

// DIF column iFFT: natural load, DIF stages, bit-reversed gather write-back
kernel void bb_intt_column_fused(
    device Bb* data                [[buffer(0)]],
    device const Bb* twiddles_inv  [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& n1              [[buffer(3)]],
    constant uint& n2              [[buffer(4)]],
    constant uint& local_stages    [[buffer(5)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint col = tgid;
    threadgroup Bb shared_data[8192];

    if (tid < n1) shared_data[tid] = data[col + tid * n2];
    if (tid + tg_size < n1) shared_data[tid + tg_size] = data[col + (tid + tg_size) * n2];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 0; s < local_stages; s++) {
        uint half_block = 1u << (local_stages - 1 - s);
        uint local_block_size = half_block << 1;
        uint block_idx = tid / half_block;
        uint local_idx = tid % half_block;
        uint i = block_idx * local_block_size + local_idx;
        uint j = i + half_block;

        uint twiddle_idx = local_idx * (n1 / local_block_size) * n2;

        Bb a = shared_data[i];
        Bb b = shared_data[j];
        Bb sum = bb_add(a, b);
        Bb diff = bb_sub(a, b);
        Bb w = twiddles_inv[twiddle_idx];
        shared_data[i] = sum;
        shared_data[j] = bb_mul(diff, w);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint rev_lo = bb_bitrev(tid, local_stages);
    uint rev_hi = bb_bitrev(tid + tg_size, local_stages);
    if (tid < n1) data[col + tid * n2] = shared_data[rev_lo];
    if (tid + tg_size < n1) data[col + (tid + tg_size) * n2] = shared_data[rev_hi];
}

// DIF row iFFT: natural load, DIF stages, bit-reversed gather write-back
kernel void bb_intt_row_fused(
    device Bb* data                [[buffer(0)]],
    device const Bb* twiddles_inv  [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& local_stages    [[buffer(3)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint block_size = tg_size << 1;
    uint base = tgid * block_size;
    threadgroup Bb shared_data[8192];

    if (base + tid < n) shared_data[tid] = data[base + tid];
    if (base + tid + tg_size < n) shared_data[tid + tg_size] = data[base + tid + tg_size];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 0; s < local_stages; s++) {
        uint half_block = 1u << (local_stages - 1 - s);
        uint local_block_size = half_block << 1;
        uint block_idx = tid / half_block;
        uint local_idx = tid % half_block;
        uint i = block_idx * local_block_size + local_idx;
        uint j = i + half_block;

        uint global_block_size = 1u << (local_stages - s);
        uint twiddle_idx = local_idx * (n / global_block_size);

        Bb a = shared_data[i];
        Bb b = shared_data[j];
        Bb sum = bb_add(a, b);
        Bb diff = bb_sub(a, b);
        Bb w = twiddles_inv[twiddle_idx];
        shared_data[i] = sum;
        shared_data[j] = bb_mul(diff, w);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint rev_lo = bb_bitrev(tid, local_stages);
    uint rev_hi = bb_bitrev(tid + tg_size, local_stages);
    if (base + tid < n) data[base + tid] = shared_data[rev_lo];
    if (base + tid + tg_size < n) data[base + tid + tg_size] = shared_data[rev_hi];
}

// Fused DIF row iFFT with inverse twiddle multiply during writeback
kernel void bb_intt_row_fused_twiddle(
    device Bb* data                [[buffer(0)]],
    device const Bb* twiddles_inv  [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& local_stages    [[buffer(3)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint block_size = tg_size << 1;
    uint base = tgid * block_size;
    threadgroup Bb shared_data[8192];

    if (base + tid < n) shared_data[tid] = data[base + tid];
    if (base + tid + tg_size < n) shared_data[tid + tg_size] = data[base + tid + tg_size];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 0; s < local_stages; s++) {
        uint half_block = 1u << (local_stages - 1 - s);
        uint local_block_size = half_block << 1;
        uint block_idx = tid / half_block;
        uint local_idx = tid % half_block;
        uint i = block_idx * local_block_size + local_idx;
        uint j = i + half_block;

        uint global_block_size = 1u << (local_stages - s);
        uint twiddle_idx = local_idx * (n / global_block_size);

        Bb a = shared_data[i];
        Bb b = shared_data[j];
        Bb sum = bb_add(a, b);
        Bb diff = bb_sub(a, b);
        Bb w = twiddles_inv[twiddle_idx];
        shared_data[i] = sum;
        shared_data[j] = bb_mul(diff, w);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Writeback with fused inverse twiddle multiply
    uint rev_lo = bb_bitrev(tid, local_stages);
    uint rev_hi = bb_bitrev(tid + tg_size, local_stages);
    if (base + tid < n) {
        Bb val = shared_data[rev_lo];
        uint col = tid;
        uint tw_idx = (uint)((ulong(tgid) * ulong(col)) % ulong(n));
        if (tw_idx != 0) val = bb_mul(val, twiddles_inv[tw_idx]);
        data[base + tid] = val;
    }
    if (base + tid + tg_size < n) {
        Bb val = shared_data[rev_hi];
        uint col = tid + tg_size;
        uint tw_idx = (uint)((ulong(tgid) * ulong(col)) % ulong(n));
        if (tw_idx != 0) val = bb_mul(val, twiddles_inv[tw_idx]);
        data[base + tid + tg_size] = val;
    }
}

// In-place bit-reversal permutation
kernel void bb_ntt_bitrev_inplace(
    device Bb* data                [[buffer(0)]],
    constant uint& n               [[buffer(1)]],
    constant uint& log_n           [[buffer(2)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    uint rev = 0;
    uint val = gid;
    for (uint i = 0; i < log_n; i++) {
        rev = (rev << 1) | (val & 1);
        val >>= 1;
    }
    if (gid < rev) {
        Bb tmp = data[gid];
        data[gid] = data[rev];
        data[rev] = tmp;
    }
}
