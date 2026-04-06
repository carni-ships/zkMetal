// NTT kernels for Goldilocks field (p = 2^64 - 2^32 + 1)
// Single u64 per element — 4x denser than BN254 Fr

#include "../fields/goldilocks.metal"

kernel void gl_ntt_butterfly(
    device Gl* data                [[buffer(0)]],
    device const Gl* twiddles      [[buffer(1)]],
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

    Gl a = data[i];
    Gl b = data[j];
    if (twiddle_idx == 0) {
        data[i] = gl_add(a, b);
        data[j] = gl_sub(a, b);
    } else {
        Gl w = twiddles[twiddle_idx];
        Gl wb = gl_mul(w, b);
        data[i] = gl_add(a, wb);
        data[j] = gl_sub(a, wb);
    }
}

kernel void gl_intt_butterfly(
    device Gl* data                [[buffer(0)]],
    device const Gl* twiddles_inv  [[buffer(1)]],
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

    Gl a = data[i];
    Gl b = data[j];
    Gl sum = gl_add(a, b);
    Gl diff = gl_sub(a, b);
    data[i] = sum;
    if (twiddle_idx == 0) {
        data[j] = diff;
    } else {
        Gl w = twiddles_inv[twiddle_idx];
        data[j] = gl_mul(diff, w);
    }
}

// Radix-4 DIT butterfly: processes 2 stages at once, halving global memory passes.
// Each thread reads 4 elements, applies stages s and s+1, writes 4 elements.
kernel void gl_ntt_butterfly_radix4(
    device Gl* data                [[buffer(0)]],
    device const Gl* twiddles      [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& stage           [[buffer(3)]],    // lower stage (s); also processes s+1
    uint gid                       [[thread_position_in_grid]]
) {
    uint h = 1u << stage;          // half_block for stage s
    uint block4 = h << 2;          // 4h = block size for stage s+1
    uint num_quads = n >> 2;       // N/4 threads
    if (gid >= num_quads) return;

    uint block_idx = gid / h;
    uint local_idx = gid % h;
    uint base = block_idx * block4 + local_idx;

    uint i0 = base;
    uint i1 = base + h;
    uint i2 = base + 2 * h;
    uint i3 = base + 3 * h;

    Gl a0 = data[i0];
    Gl a1 = data[i1];
    Gl a2 = data[i2];
    Gl a3 = data[i3];

    // Stage s twiddle: same for (a0,a1) and (a2,a3)
    uint tw_s = local_idx * (n / (2 * h));

    // Stage s+1 twiddles
    uint tw_s1_lo = local_idx * (n / block4);
    uint tw_s1_hi = (local_idx + h) * (n / block4);

    // Stage s
    Gl b0, b1, b2, b3;
    if (tw_s == 0) {
        b0 = gl_add(a0, a1); b1 = gl_sub(a0, a1);
        b2 = gl_add(a2, a3); b3 = gl_sub(a2, a3);
    } else {
        Gl ws = twiddles[tw_s];
        Gl ws_a1 = gl_mul(ws, a1); Gl ws_a3 = gl_mul(ws, a3);
        b0 = gl_add(a0, ws_a1); b1 = gl_sub(a0, ws_a1);
        b2 = gl_add(a2, ws_a3); b3 = gl_sub(a2, ws_a3);
    }

    // Stage s+1
    Gl wb2 = (tw_s1_lo == 0) ? b2 : gl_mul(twiddles[tw_s1_lo], b2);
    Gl wb3 = (tw_s1_hi == 0) ? b3 : gl_mul(twiddles[tw_s1_hi], b3);

    data[i0] = gl_add(b0, wb2);
    data[i2] = gl_sub(b0, wb2);
    data[i1] = gl_add(b1, wb3);
    data[i3] = gl_sub(b1, wb3);
}

// Radix-4 DIF butterfly for iNTT: processes 2 stages at once.
kernel void gl_intt_butterfly_radix4(
    device Gl* data                [[buffer(0)]],
    device const Gl* twiddles_inv  [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& stage           [[buffer(3)]],    // higher stage (s); also processes s-1
    uint gid                       [[thread_position_in_grid]]
) {
    // DIF processes from high stage down. stage = s (higher), also does s-1.
    uint h_hi = 1u << stage;       // half_block for stage s
    uint h_lo = h_hi >> 1;         // half_block for stage s-1
    uint block4 = h_hi << 1;      // block size for stage s = 2*h_hi
    uint num_quads = n >> 2;
    if (gid >= num_quads) return;

    uint block_idx = gid / h_lo;
    uint local_idx = gid % h_lo;
    uint base = block_idx * block4 + local_idx;

    uint i0 = base;
    uint i1 = base + h_lo;
    uint i2 = base + h_hi;
    uint i3 = base + h_hi + h_lo;

    Gl a0 = data[i0];
    Gl a1 = data[i1];
    Gl a2 = data[i2];
    Gl a3 = data[i3];

    // Stage s (DIF): pairs (i0,i2) and (i1,i3)
    uint tw_s_lo = local_idx * (n / block4);
    uint tw_s_hi = (local_idx + h_lo) * (n / block4);
    Gl ws_lo = twiddles_inv[tw_s_lo];
    Gl ws_hi = twiddles_inv[tw_s_hi];

    Gl b0 = gl_add(a0, a2);
    Gl diff02 = gl_sub(a0, a2);
    Gl b2 = (tw_s_lo == 0) ? diff02 : gl_mul(diff02, ws_lo);
    Gl b1 = gl_add(a1, a3);
    Gl diff13 = gl_sub(a1, a3);
    Gl b3 = (tw_s_hi == 0) ? diff13 : gl_mul(diff13, ws_hi);

    // Stage s-1 (DIF): pairs (b0,b1) and (b2,b3)
    uint tw_s1 = local_idx * (n / (2 * h_lo));

    Gl diff01 = gl_sub(b0, b1);
    Gl diff23 = gl_sub(b2, b3);

    data[i0] = gl_add(b0, b1);
    data[i1] = (tw_s1 == 0) ? diff01 : gl_mul(diff01, twiddles_inv[tw_s1]);
    data[i2] = gl_add(b2, b3);
    data[i3] = (tw_s1 == 0) ? diff23 : gl_mul(diff23, twiddles_inv[tw_s1]);
}

kernel void gl_ntt_scale(
    device Gl* data                [[buffer(0)]],
    device const Gl* scale_factor  [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    data[gid] = gl_mul(data[gid], scale_factor[0]);
}

kernel void gl_ntt_bitrev_inplace(
    device Gl* data                [[buffer(0)]],
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
        Gl tmp = data[gid];
        data[gid] = data[rev];
        data[rev] = tmp;
    }
}

// Fused butterfly kernel — 4096 Gl elements * 8 bytes = 32KB shared memory
kernel void gl_ntt_butterfly_fused(
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

    threadgroup Gl shared[4096];  // 4096 * 8 bytes = 32KB
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

        Gl a = shared[i];
        Gl b = shared[j];
        if (twiddle_idx == 0) {
            shared[i] = gl_add(a, b);
            shared[j] = gl_sub(a, b);
        } else {
            Gl w = twiddles[twiddle_idx];
            Gl wb = gl_mul(w, b);
            shared[i] = gl_add(a, wb);
            shared[j] = gl_sub(a, wb);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (base + tid < n) data[base + tid] = shared[tid];
    if (base + tid + tg_size < n) data[base + tid + tg_size] = shared[tid + tg_size];
}

// Fused bitrev + DIT butterfly: read from input with bit-reversed indexing,
// compute fused stages in shared memory, write to output buffer.
kernel void gl_ntt_butterfly_fused_bitrev(
    device const Gl* input         [[buffer(0)]],
    device Gl* output              [[buffer(1)]],
    device const Gl* twiddles      [[buffer(2)]],
    constant uint& n               [[buffer(3)]],
    constant uint& local_stages    [[buffer(4)]],
    constant uint& logN            [[buffer(5)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint block_size = tg_size << 1;
    uint base = tgid * block_size;

    threadgroup Gl shared[4096];

    uint idx_lo = tid;
    uint idx_hi = tid + tg_size;
    uint global_lo = base + idx_lo;
    uint global_hi = base + idx_hi;

    uint rev_lo = 0, rev_hi = 0;
    { uint v = global_lo; for (uint i = 0; i < logN; i++) { rev_lo = (rev_lo << 1) | (v & 1); v >>= 1; } }
    { uint v = global_hi; for (uint i = 0; i < logN; i++) { rev_hi = (rev_hi << 1) | (v & 1); v >>= 1; } }

    if (global_lo < n) shared[idx_lo] = input[rev_lo];
    if (global_hi < n) shared[idx_hi] = input[rev_hi];
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

        Gl a = shared[i];
        Gl b = shared[j];
        if (twiddle_idx == 0) {
            shared[i] = gl_add(a, b);
            shared[j] = gl_sub(a, b);
        } else {
            Gl w = twiddles[twiddle_idx];
            Gl wb = gl_mul(w, b);
            shared[i] = gl_add(a, wb);
            shared[j] = gl_sub(a, wb);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (global_lo < n) output[global_lo] = shared[idx_lo];
    if (global_hi < n) output[global_hi] = shared[idx_hi];
}

// --- Four-step FFT kernels for Goldilocks ---

inline uint gl_bitrev(uint val, uint num_bits) {
    uint rev = 0;
    for (uint i = 0; i < num_bits; i++) {
        rev = (rev << 1) | (val & 1);
        val >>= 1;
    }
    return rev;
}

// Column DIT FFT with bit-reversed loading
kernel void gl_ntt_column_fused(
    device Gl* data                [[buffer(0)]],
    device const Gl* twiddles      [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& n1              [[buffer(3)]],
    constant uint& n2              [[buffer(4)]],
    constant uint& local_stages    [[buffer(5)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint col = tgid;
    threadgroup Gl shared_data[4096];

    uint idx_lo = tid;
    uint idx_hi = tid + tg_size;
    uint rev_lo = gl_bitrev(idx_lo, local_stages);
    uint rev_hi = gl_bitrev(idx_hi, local_stages);

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

        Gl a = shared_data[i];
        Gl b = shared_data[j];
        if (twiddle_idx == 0) {
            shared_data[i] = gl_add(a, b);
            shared_data[j] = gl_sub(a, b);
        } else {
            Gl w = twiddles[twiddle_idx];
            Gl wb = gl_mul(w, b);
            shared_data[i] = gl_add(a, wb);
            shared_data[j] = gl_sub(a, wb);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < n1) data[col + tid * n2] = shared_data[tid];
    if (tid + tg_size < n1) data[col + (tid + tg_size) * n2] = shared_data[tid + tg_size];
}

// Row DIT FFT with bit-reversed loading
kernel void gl_ntt_row_fused(
    device Gl* data                [[buffer(0)]],
    device const Gl* twiddles      [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& local_stages    [[buffer(3)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint block_size = tg_size << 1;
    uint base = tgid * block_size;
    threadgroup Gl shared_data[4096];

    uint idx_lo = tid;
    uint idx_hi = tid + tg_size;
    uint rev_lo = gl_bitrev(idx_lo, local_stages);
    uint rev_hi = gl_bitrev(idx_hi, local_stages);

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

        Gl a = shared_data[i];
        Gl b = shared_data[j];
        if (twiddle_idx == 0) {
            shared_data[i] = gl_add(a, b);
            shared_data[j] = gl_sub(a, b);
        } else {
            Gl w = twiddles[twiddle_idx];
            Gl wb = gl_mul(w, b);
            shared_data[i] = gl_add(a, wb);
            shared_data[j] = gl_sub(a, wb);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (base + tid < n) data[base + tid] = shared_data[tid];
    if (base + tid + tg_size < n) data[base + tid + tg_size] = shared_data[tid + tg_size];
}

// Fused row DIT FFT with twiddle multiply: eliminates separate twiddle pass
// Each thread applies twiddle[(row*col) % n] during load, then does DIT stages
kernel void gl_ntt_row_fused_twiddle(
    device Gl* data                [[buffer(0)]],
    device const Gl* twiddles      [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& local_stages    [[buffer(3)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint block_size = tg_size << 1;
    uint base = tgid * block_size;
    threadgroup Gl shared_data[4096];

    uint idx_lo = tid;
    uint idx_hi = tid + tg_size;
    uint rev_lo = gl_bitrev(idx_lo, local_stages);
    uint rev_hi = gl_bitrev(idx_hi, local_stages);

    // Load with fused twiddle multiply: tw_idx = (row * col) % n
    if (base + idx_lo < n) {
        Gl val = data[base + idx_lo];
        uint tw_idx = (uint)((ulong(tgid) * ulong(idx_lo)) % ulong(n));
        if (tw_idx != 0) val = gl_mul(val, twiddles[tw_idx]);
        shared_data[rev_lo] = val;
    }
    if (base + idx_hi < n) {
        Gl val = data[base + idx_hi];
        uint tw_idx = (uint)((ulong(tgid) * ulong(idx_hi)) % ulong(n));
        if (tw_idx != 0) val = gl_mul(val, twiddles[tw_idx]);
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

        Gl a = shared_data[i];
        Gl b = shared_data[j];
        if (twiddle_idx == 0) {
            shared_data[i] = gl_add(a, b);
            shared_data[j] = gl_sub(a, b);
        } else {
            Gl w = twiddles[twiddle_idx];
            Gl wb = gl_mul(w, b);
            shared_data[i] = gl_add(a, wb);
            shared_data[j] = gl_sub(a, wb);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (base + tid < n) data[base + tid] = shared_data[tid];
    if (base + tid + tg_size < n) data[base + tid + tg_size] = shared_data[tid + tg_size];
}

// Fused DIF row iFFT with inverse twiddle multiply during writeback
kernel void gl_intt_row_fused_twiddle(
    device Gl* data                [[buffer(0)]],
    device const Gl* twiddles_inv  [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& local_stages    [[buffer(3)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint block_size = tg_size << 1;
    uint base = tgid * block_size;
    threadgroup Gl shared_data[4096];

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

        Gl a = shared_data[i];
        Gl b = shared_data[j];
        Gl sum = gl_add(a, b);
        Gl diff = gl_sub(a, b);
        shared_data[i] = sum;
        if (twiddle_idx == 0) {
            shared_data[j] = diff;
        } else {
            Gl w = twiddles_inv[twiddle_idx];
            shared_data[j] = gl_mul(diff, w);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Writeback with fused inverse twiddle multiply
    uint rev_lo = gl_bitrev(tid, local_stages);
    uint rev_hi = gl_bitrev(tid + tg_size, local_stages);
    if (base + tid < n) {
        Gl val = shared_data[rev_lo];
        uint col = tid;
        uint tw_idx = (uint)((ulong(tgid) * ulong(col)) % ulong(n));
        if (tw_idx != 0) val = gl_mul(val, twiddles_inv[tw_idx]);
        data[base + tid] = val;
    }
    if (base + tid + tg_size < n) {
        Gl val = shared_data[rev_hi];
        uint col = tid + tg_size;
        uint tw_idx = (uint)((ulong(tgid) * ulong(col)) % ulong(n));
        if (tw_idx != 0) val = gl_mul(val, twiddles_inv[tw_idx]);
        data[base + tid + tg_size] = val;
    }
}

// Twiddle multiply: data[row*N2+col] *= twiddles[(row*col) % N]
kernel void gl_ntt_twiddle_multiply(
    device Gl* data                [[buffer(0)]],
    device const Gl* twiddles      [[buffer(1)]],
    constant uint& n2              [[buffer(2)]],
    constant uint& n               [[buffer(3)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    uint row = gid / n2;
    uint col = gid % n2;
    uint twiddle_idx = (uint)((ulong(row) * ulong(col)) % ulong(n));
    if (twiddle_idx == 0) return;
    data[gid] = gl_mul(data[gid], twiddles[twiddle_idx]);
}

// In-place square matrix transpose
kernel void gl_ntt_transpose(
    device Gl* data                [[buffer(0)]],
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
    Gl tmp = data[i];
    data[i] = data[j];
    data[j] = tmp;
}

// DIF column iFFT with fused 1/N scale: eliminates separate scale pass
kernel void gl_intt_column_fused_scale(
    device Gl* data                [[buffer(0)]],
    device const Gl* twiddles_inv  [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& n1              [[buffer(3)]],
    constant uint& n2              [[buffer(4)]],
    constant uint& local_stages    [[buffer(5)]],
    device const Gl* inv_n         [[buffer(6)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint col = tgid;
    threadgroup Gl shared_data[4096];
    Gl scale = inv_n[0];

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

        Gl a = shared_data[i];
        Gl b = shared_data[j];
        Gl sum = gl_add(a, b);
        Gl diff = gl_sub(a, b);
        shared_data[i] = sum;
        if (twiddle_idx == 0) {
            shared_data[j] = diff;
        } else {
            Gl w = twiddles_inv[twiddle_idx];
            shared_data[j] = gl_mul(diff, w);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint rev_lo = gl_bitrev(tid, local_stages);
    uint rev_hi = gl_bitrev(tid + tg_size, local_stages);
    if (tid < n1) data[col + tid * n2] = gl_mul(shared_data[rev_lo], scale);
    if (tid + tg_size < n1) data[col + (tid + tg_size) * n2] = gl_mul(shared_data[rev_hi], scale);
}

// DIF column iFFT: natural load, DIF stages, bit-reversed gather write-back
kernel void gl_intt_column_fused(
    device Gl* data                [[buffer(0)]],
    device const Gl* twiddles_inv  [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& n1              [[buffer(3)]],
    constant uint& n2              [[buffer(4)]],
    constant uint& local_stages    [[buffer(5)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint col = tgid;
    threadgroup Gl shared_data[4096];

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

        Gl a = shared_data[i];
        Gl b = shared_data[j];
        Gl sum = gl_add(a, b);
        Gl diff = gl_sub(a, b);
        shared_data[i] = sum;
        if (twiddle_idx == 0) {
            shared_data[j] = diff;
        } else {
            Gl w = twiddles_inv[twiddle_idx];
            shared_data[j] = gl_mul(diff, w);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint rev_lo = gl_bitrev(tid, local_stages);
    uint rev_hi = gl_bitrev(tid + tg_size, local_stages);
    if (tid < n1) data[col + tid * n2] = shared_data[rev_lo];
    if (tid + tg_size < n1) data[col + (tid + tg_size) * n2] = shared_data[rev_hi];
}

// DIF row iFFT: natural load, DIF stages, bit-reversed gather write-back
kernel void gl_intt_row_fused(
    device Gl* data                [[buffer(0)]],
    device const Gl* twiddles_inv  [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& local_stages    [[buffer(3)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint block_size = tg_size << 1;
    uint base = tgid * block_size;
    threadgroup Gl shared_data[4096];

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

        Gl a = shared_data[i];
        Gl b = shared_data[j];
        Gl sum = gl_add(a, b);
        Gl diff = gl_sub(a, b);
        shared_data[i] = sum;
        if (twiddle_idx == 0) {
            shared_data[j] = diff;
        } else {
            Gl w = twiddles_inv[twiddle_idx];
            shared_data[j] = gl_mul(diff, w);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint rev_lo = gl_bitrev(tid, local_stages);
    uint rev_hi = gl_bitrev(tid + tg_size, local_stages);
    if (base + tid < n) data[base + tid] = shared_data[rev_lo];
    if (base + tid + tg_size < n) data[base + tid + tg_size] = shared_data[rev_hi];
}

// Fused inverse butterfly kernel — DIF stages in threadgroup memory
kernel void gl_intt_butterfly_fused(
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
