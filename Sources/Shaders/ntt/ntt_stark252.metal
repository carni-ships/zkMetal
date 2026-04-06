// NTT/iNTT GPU kernels for Stark252 scalar field
// Cooley-Tukey radix-2 DIT forward transform
// Gentleman-Sande radix-2 DIF inverse transform
// Multi-pass: one kernel dispatch per butterfly stage
// Fused threadgroup-local kernel for small stages

#include "../fields/stark252.metal"

// --- NTT Butterfly Kernel (one stage per dispatch) ---
// Each thread processes one butterfly pair.
// stage: current stage index (0 = stride 1, 1 = stride 2, ...)
// For DIT (forward): butterfly at positions (j, j + half_block)
//   a' = a + w*b
//   b' = a - w*b

kernel void stark252_ntt_butterfly(
    device Stark252* data                [[buffer(0)]],
    device const Stark252* twiddles      [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& stage           [[buffer(3)]],
    uint gid                       [[thread_position_in_grid]]
) {
    uint half_block = 1u << stage;
    uint block_size = half_block << 1;
    uint num_butterflies = n >> 1;

    if (gid >= num_butterflies) return;

    // Determine which butterfly pair this thread handles
    uint block_idx = gid / half_block;
    uint local_idx = gid % half_block;
    uint i = block_idx * block_size + local_idx;
    uint j = i + half_block;

    // Twiddle factor index: for DIT stage s, twiddle[local_idx * (n / block_size)]
    uint twiddle_idx = local_idx * (n / block_size);

    Stark252 a = data[i];
    Stark252 b = data[j];
    if (twiddle_idx == 0) {
        data[i] = stark252_add(a, b);
        data[j] = stark252_sub(a, b);
    } else {
        Stark252 w = twiddles[twiddle_idx];
        Stark252 wb = stark252_mul(w, b);
        data[i] = stark252_add(a, wb);
        data[j] = stark252_sub(a, wb);
    }
}

// --- iNTT Butterfly Kernel (Gentleman-Sande DIF) ---
// For DIF (inverse): butterfly at positions (j, j + half_block)
//   a' = a + b
//   b' = (a - b) * w_inv

kernel void stark252_intt_butterfly(
    device Stark252* data                [[buffer(0)]],
    device const Stark252* twiddles_inv  [[buffer(1)]],
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

    Stark252 a = data[i];
    Stark252 b = data[j];

    Stark252 sum = stark252_add(a, b);
    Stark252 diff = stark252_sub(a, b);
    data[i] = sum;
    if (twiddle_idx == 0) {
        data[j] = diff;
    } else {
        Stark252 w = twiddles_inv[twiddle_idx];
        data[j] = stark252_mul(diff, w);
    }
}

// Radix-4 DIT butterfly: processes 2 stages at once for BN254.
kernel void stark252_ntt_butterfly_radix4(
    device Stark252* data                [[buffer(0)]],
    device const Stark252* twiddles      [[buffer(1)]],
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

    Stark252 a0 = data[base];
    Stark252 a1 = data[base + h];
    Stark252 a2 = data[base + 2 * h];
    Stark252 a3 = data[base + 3 * h];

    uint tw_s = local_idx * (n / (2 * h));
    Stark252 b0, b1, b2, b3;
    if (tw_s == 0) {
        b0 = stark252_add(a0, a1); b1 = stark252_sub(a0, a1);
        b2 = stark252_add(a2, a3); b3 = stark252_sub(a2, a3);
    } else {
        Stark252 ws = twiddles[tw_s];
        Stark252 ws_a1 = stark252_mul(ws, a1); Stark252 ws_a3 = stark252_mul(ws, a3);
        b0 = stark252_add(a0, ws_a1); b1 = stark252_sub(a0, ws_a1);
        b2 = stark252_add(a2, ws_a3); b3 = stark252_sub(a2, ws_a3);
    }

    uint tw_lo = local_idx * (n / block4);
    uint tw_hi = (local_idx + h) * (n / block4);
    Stark252 wb2 = (tw_lo == 0) ? b2 : stark252_mul(twiddles[tw_lo], b2);
    Stark252 wb3 = (tw_hi == 0) ? b3 : stark252_mul(twiddles[tw_hi], b3);

    data[base]         = stark252_add(b0, wb2);
    data[base + 2 * h] = stark252_sub(b0, wb2);
    data[base + h]     = stark252_add(b1, wb3);
    data[base + 3 * h] = stark252_sub(b1, wb3);
}

// Radix-4 DIF butterfly for BN254 iNTT: processes 2 stages at once.
kernel void stark252_intt_butterfly_radix4(
    device Stark252* data                [[buffer(0)]],
    device const Stark252* twiddles_inv  [[buffer(1)]],
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

    Stark252 a0 = data[base];
    Stark252 a1 = data[base + h_lo];
    Stark252 a2 = data[base + h_hi];
    Stark252 a3 = data[base + h_hi + h_lo];

    uint tw_s_lo = local_idx * (n / block4);
    uint tw_s_hi = (local_idx + h_lo) * (n / block4);
    Stark252 b0 = stark252_add(a0, a2);
    Stark252 diff02 = stark252_sub(a0, a2);
    Stark252 b2 = (tw_s_lo == 0) ? diff02 : stark252_mul(diff02, twiddles_inv[tw_s_lo]);
    Stark252 b1 = stark252_add(a1, a3);
    Stark252 diff13 = stark252_sub(a1, a3);
    Stark252 b3 = (tw_s_hi == 0) ? diff13 : stark252_mul(diff13, twiddles_inv[tw_s_hi]);

    uint tw_s1 = local_idx * (n / (2 * h_lo));
    Stark252 diff_b01 = stark252_sub(b0, b1);
    Stark252 diff_b23 = stark252_sub(b2, b3);
    data[base]              = stark252_add(b0, b1);
    data[base + h_lo]       = (tw_s1 == 0) ? diff_b01 : stark252_mul(diff_b01, twiddles_inv[tw_s1]);
    data[base + h_hi]       = stark252_add(b2, b3);
    data[base + h_hi + h_lo] = (tw_s1 == 0) ? diff_b23 : stark252_mul(diff_b23, twiddles_inv[tw_s1]);
}

// --- Fused NTT kernel: process multiple DIT stages in threadgroup memory ---
// Each threadgroup loads BLOCK_SIZE elements, performs local_stages butterfly stages
// in shared memory, then writes back.
// local_stages = log2(BLOCK_SIZE) = number of stages that fit in threadgroup.
// This replaces the first local_stages individual dispatches.

kernel void stark252_ntt_butterfly_fused(
    device Stark252* data                [[buffer(0)]],
    device const Stark252* twiddles      [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& local_stages    [[buffer(3)]],   // how many stages to fuse
    constant uint& stage_offset    [[buffer(4)]],   // first stage index
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    // Block size = 2 * tg_size (each thread handles one butterfly = 2 elements)
    uint block_size = tg_size << 1;
    uint base = tgid * block_size;

    // Load into threadgroup memory
    threadgroup Stark252 shared[1024]; // max 1024 Stark252 elements = 32KB
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

        uint block_idx = tid / half_block;
        uint local_idx = tid % half_block;
        uint i = block_idx * local_block_size + local_idx;
        uint j = i + half_block;

        // Global twiddle index: for DIT stage s, the stride in the full array
        // At stage (stage_offset + s), block_size_global = 2^(stage+1), n/block_size_global
        uint global_block_size = 1u << (stage + 1);
        uint twiddle_idx = local_idx * (n / global_block_size);

        Stark252 a = shared[i];
        Stark252 b = shared[j];
        if (twiddle_idx == 0) {
            shared[i] = stark252_add(a, b);
            shared[j] = stark252_sub(a, b);
        } else {
            Stark252 w = twiddles[twiddle_idx];
            Stark252 wb = stark252_mul(w, b);
            shared[i] = stark252_add(a, wb);
            shared[j] = stark252_sub(a, wb);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write back
    if (base + tid < n)
        data[base + tid] = shared[tid];
    if (base + tid + tg_size < n)
        data[base + tid + tg_size] = shared[tid + tg_size];
}

// Helper: compute bit-reversal of val with num_bits bits (forward declaration for fused-bitrev)
inline uint bitrev(uint val, uint num_bits);

// Fused DIT kernel with bit-reversed loading: eliminates the separate bitrev pass.
// Reads from input with bit-reversed indexing, writes to output sequentially.
// input and output MUST be different buffers (no aliasing) to avoid read-write races.
kernel void stark252_ntt_butterfly_fused_bitrev(
    device const Stark252* input         [[buffer(0)]],   // source data (natural order)
    device Stark252* output              [[buffer(1)]],   // destination (after bitrev + fused stages)
    device const Stark252* twiddles      [[buffer(2)]],
    constant uint& n               [[buffer(3)]],
    constant uint& local_stages    [[buffer(4)]],
    constant uint& logN            [[buffer(5)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint block_size = tg_size << 1;
    uint base = tgid * block_size;

    threadgroup Stark252 shared[1024];

    uint idx_lo = tid;
    uint idx_hi = tid + tg_size;
    uint global_lo = base + idx_lo;
    uint global_hi = base + idx_hi;

    // Load from input with bit-reversed indices
    if (global_lo < n)
        shared[idx_lo] = input[bitrev(global_lo, logN)];
    if (global_hi < n)
        shared[idx_hi] = input[bitrev(global_hi, logN)];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // DIT butterfly stages
    for (uint s = 0; s < local_stages; s++) {
        uint half_block = 1u << s;
        uint local_block_size = half_block << 1;

        uint block_idx = tid / half_block;
        uint local_idx = tid % half_block;
        uint i = block_idx * local_block_size + local_idx;
        uint j = i + half_block;

        uint global_block_size = 1u << (s + 1);
        uint twiddle_idx = local_idx * (n / global_block_size);

        Stark252 a = shared[i];
        Stark252 b = shared[j];
        if (twiddle_idx == 0) {
            shared[i] = stark252_add(a, b);
            shared[j] = stark252_sub(a, b);
        } else {
            Stark252 w = twiddles[twiddle_idx];
            Stark252 wb = stark252_mul(w, b);
            shared[i] = stark252_add(a, wb);
            shared[j] = stark252_sub(a, wb);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write to output buffer (sequential positions)
    if (global_lo < n)
        output[global_lo] = shared[idx_lo];
    if (global_hi < n)
        output[global_hi] = shared[idx_hi];
}

// --- Fused iNTT kernel: process multiple DIF stages in threadgroup memory ---
kernel void stark252_intt_butterfly_fused(
    device Stark252* data                [[buffer(0)]],
    device const Stark252* twiddles_inv  [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& local_stages    [[buffer(3)]],
    constant uint& stage_offset    [[buffer(4)]],   // first (highest) stage index for DIF
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint block_size = tg_size << 1;
    uint base = tgid * block_size;

    threadgroup Stark252 shared[1024];
    if (base + tid < n)
        shared[tid] = data[base + tid];
    if (base + tid + tg_size < n)
        shared[tid + tg_size] = data[base + tid + tg_size];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // DIF: stages go from high to low
    for (uint s = 0; s < local_stages; s++) {
        // stage_offset is the highest local stage, we go downward
        uint stage = stage_offset - s;
        uint half_block = 1u << (local_stages - 1 - s);  // local half_block
        uint local_block_size = half_block << 1;

        uint block_idx = tid / half_block;
        uint local_idx = tid % half_block;
        uint i = block_idx * local_block_size + local_idx;
        uint j = i + half_block;

        uint global_block_size = 1u << (stage + 1);
        uint twiddle_idx = local_idx * (n / global_block_size);

        Stark252 a = shared[i];
        Stark252 b = shared[j];
        Stark252 sum = stark252_add(a, b);
        Stark252 diff = stark252_sub(a, b);
        shared[i] = sum;
        if (twiddle_idx == 0) {
            shared[j] = diff;
        } else {
            Stark252 w = twiddles_inv[twiddle_idx];
            shared[j] = stark252_mul(diff, w);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (base + tid < n)
        data[base + tid] = shared[tid];
    if (base + tid + tg_size < n)
        data[base + tid + tg_size] = shared[tid + tg_size];
}

// --- Scale kernel: multiply each element by a scalar (for iNTT normalization) ---

kernel void stark252_ntt_scale(
    device Stark252* data                [[buffer(0)]],
    device const Stark252* scale_factor  [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    data[gid] = stark252_mul(data[gid], scale_factor[0]);
}

// --- Bit-reversal permutation kernel ---

kernel void stark252_ntt_bitrev(
    device const Stark252* input         [[buffer(0)]],
    device Stark252* output              [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& log_n           [[buffer(3)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    // Compute bit-reversal of gid with log_n bits
    uint rev = 0;
    uint val = gid;
    for (uint i = 0; i < log_n; i++) {
        rev = (rev << 1) | (val & 1);
        val >>= 1;
    }

    output[rev] = input[gid];
}

// --- In-place bit-reversal: swap pairs where gid < rev(gid) ---
kernel void stark252_ntt_bitrev_inplace(
    device Stark252* data                [[buffer(0)]],
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
        Stark252 tmp = data[gid];
        data[gid] = data[rev];
        data[rev] = tmp;
    }
}

// --- Four-step FFT kernels ---
// Computes N-point DFT as N1×N2 matrix operations:
//   1. N2 column DIT FFTs of size N1 (with bit-reversed loading)
//   2. Twiddle multiply: element[row,col] *= omega_N^(row*col)
//   3. N1 row DIT FFTs of size N2 (with bit-reversed loading)
//   4. Transpose: output[k1+k2*N1] is at position [k1,k2] in row-major

// Helper: compute bit-reversal of val with num_bits bits
inline uint bitrev(uint val, uint num_bits) {
    uint rev = 0;
    for (uint i = 0; i < num_bits; i++) {
        rev = (rev << 1) | (val & 1);
        val >>= 1;
    }
    return rev;
}

// Column FFT with bit-reversed loading: each threadgroup = one column of N1 elements
kernel void stark252_ntt_column_fused(
    device Stark252* data                [[buffer(0)]],
    device const Stark252* twiddles      [[buffer(1)]],
    constant uint& n               [[buffer(2)]],    // total size N = N1 * N2
    constant uint& n1              [[buffer(3)]],    // N1 (column size)
    constant uint& n2              [[buffer(4)]],    // N2 (number of columns)
    constant uint& local_stages    [[buffer(5)]],    // log2(N1)
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint col = tgid;
    threadgroup Stark252 shared[1024];

    // Load column into shared memory with bit-reversed positions (for DIT)
    uint idx_lo = tid;
    uint idx_hi = tid + tg_size;
    uint rev_lo = bitrev(idx_lo, local_stages);
    uint rev_hi = bitrev(idx_hi, local_stages);

    if (idx_lo < n1)
        shared[rev_lo] = data[col + idx_lo * n2];
    if (idx_hi < n1)
        shared[rev_hi] = data[col + idx_hi * n2];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // DIT butterfly stages
    for (uint s = 0; s < local_stages; s++) {
        uint half_block = 1u << s;
        uint local_block_size = half_block << 1;

        uint block_idx = tid / half_block;
        uint local_idx = tid % half_block;
        uint i = block_idx * local_block_size + local_idx;
        uint j = i + half_block;

        // Twiddle: omega_N1^(local_idx * N1/local_block_size) = omega_N^(local_idx * N2 * N1/local_block_size)
        uint twiddle_idx = local_idx * (n1 / local_block_size) * n2;

        Stark252 a = shared[i];
        Stark252 b = shared[j];
        if (twiddle_idx == 0) {
            shared[i] = stark252_add(a, b);
            shared[j] = stark252_sub(a, b);
        } else {
            Stark252 w = twiddles[twiddle_idx];
            Stark252 wb = stark252_mul(w, b);
            shared[i] = stark252_add(a, wb);
            shared[j] = stark252_sub(a, wb);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write back (strided)
    if (tid < n1)
        data[col + tid * n2] = shared[tid];
    if (tid + tg_size < n1)
        data[col + (tid + tg_size) * n2] = shared[tid + tg_size];
}

// Row FFT with bit-reversed loading: each threadgroup = one row of N2 contiguous elements
kernel void stark252_ntt_row_fused(
    device Stark252* data                [[buffer(0)]],
    device const Stark252* twiddles      [[buffer(1)]],
    constant uint& n               [[buffer(2)]],    // total size N
    constant uint& local_stages    [[buffer(3)]],    // log2(N2)
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint block_size = tg_size << 1;
    uint base = tgid * block_size;
    threadgroup Stark252 shared[1024];

    // Load with bit-reversed positions
    uint idx_lo = tid;
    uint idx_hi = tid + tg_size;
    uint rev_lo = bitrev(idx_lo, local_stages);
    uint rev_hi = bitrev(idx_hi, local_stages);

    if (base + idx_lo < n)
        shared[rev_lo] = data[base + idx_lo];
    if (base + idx_hi < n)
        shared[rev_hi] = data[base + idx_hi];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // DIT butterfly stages
    for (uint s = 0; s < local_stages; s++) {
        uint half_block = 1u << s;
        uint local_block_size = half_block << 1;

        uint block_idx = tid / half_block;
        uint local_idx = tid % half_block;
        uint i = block_idx * local_block_size + local_idx;
        uint j = i + half_block;

        // Twiddle: for N2-point NTT stage s, using N-point table
        uint global_block_size = 1u << (s + 1);
        uint twiddle_idx = local_idx * (n / global_block_size);

        Stark252 a = shared[i];
        Stark252 b = shared[j];
        if (twiddle_idx == 0) {
            shared[i] = stark252_add(a, b);
            shared[j] = stark252_sub(a, b);
        } else {
            Stark252 w = twiddles[twiddle_idx];
            Stark252 wb = stark252_mul(w, b);
            shared[i] = stark252_add(a, wb);
            shared[j] = stark252_sub(a, wb);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write back
    if (base + tid < n)
        data[base + tid] = shared[tid];
    if (base + tid + tg_size < n)
        data[base + tid + tg_size] = shared[tid + tg_size];
}

// Fused row FFT + twiddle multiply: applies omega_N^(row*col) during load,
// eliminating a separate twiddle multiply pass (saves one full memory round-trip).
// tgid = row index, tid = column index within the row.
kernel void stark252_ntt_row_fused_twiddle(
    device Stark252* data                [[buffer(0)]],
    device const Stark252* twiddles      [[buffer(1)]],
    constant uint& n               [[buffer(2)]],    // total size N
    constant uint& local_stages    [[buffer(3)]],    // log2(N2)
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint block_size = tg_size << 1;
    uint base = tgid * block_size;  // base = row * N2
    threadgroup Stark252 shared[1024];

    uint idx_lo = tid;
    uint idx_hi = tid + tg_size;
    uint rev_lo = bitrev(idx_lo, local_stages);
    uint rev_hi = bitrev(idx_hi, local_stages);

    // Load with twiddle multiply: val *= omega_N^(row * col)
    // row = tgid, col = idx
    if (base + idx_lo < n) {
        Stark252 val = data[base + idx_lo];
        uint tw_idx = (uint)((ulong(tgid) * ulong(idx_lo)) % ulong(n));
        if (tw_idx != 0) val = stark252_mul(val, twiddles[tw_idx]);
        shared[rev_lo] = val;
    }
    if (base + idx_hi < n) {
        Stark252 val = data[base + idx_hi];
        uint tw_idx = (uint)((ulong(tgid) * ulong(idx_hi)) % ulong(n));
        if (tw_idx != 0) val = stark252_mul(val, twiddles[tw_idx]);
        shared[rev_hi] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // DIT butterfly stages (identical to ntt_row_fused)
    for (uint s = 0; s < local_stages; s++) {
        uint half_block = 1u << s;
        uint local_block_size = half_block << 1;

        uint block_idx = tid / half_block;
        uint local_idx = tid % half_block;
        uint i = block_idx * local_block_size + local_idx;
        uint j = i + half_block;

        uint global_block_size = 1u << (s + 1);
        uint twiddle_idx = local_idx * (n / global_block_size);

        Stark252 a = shared[i];
        Stark252 b = shared[j];
        if (twiddle_idx == 0) {
            shared[i] = stark252_add(a, b);
            shared[j] = stark252_sub(a, b);
        } else {
            Stark252 w = twiddles[twiddle_idx];
            Stark252 wb = stark252_mul(w, b);
            shared[i] = stark252_add(a, wb);
            shared[j] = stark252_sub(a, wb);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write back
    if (base + tid < n)
        data[base + tid] = shared[tid];
    if (base + tid + tg_size < n)
        data[base + tid + tg_size] = shared[tid + tg_size];
}

// Fused row FFT + twiddle + transpose: combines steps 2+3+4 of four-step FFT.
// Reads from input buffer, writes to output buffer in transposed order.
// Uses separate buffers to avoid read/write conflicts between threadgroups.
kernel void stark252_ntt_row_fused_twiddle_transpose(
    device const Stark252* input         [[buffer(0)]],
    device Stark252* output              [[buffer(1)]],
    device const Stark252* twiddles      [[buffer(2)]],
    constant uint& n               [[buffer(3)]],    // total size N
    constant uint& local_stages    [[buffer(4)]],    // log2(N2)
    constant uint& n1              [[buffer(5)]],    // N1 (number of rows)
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint block_size = tg_size << 1;
    uint base = tgid * block_size;  // base = row * N2
    threadgroup Stark252 shared[1024];

    uint idx_lo = tid;
    uint idx_hi = tid + tg_size;
    uint rev_lo = bitrev(idx_lo, local_stages);
    uint rev_hi = bitrev(idx_hi, local_stages);

    // Load with twiddle multiply: val *= omega_N^(row * col)
    if (base + idx_lo < n) {
        Stark252 val = input[base + idx_lo];
        uint tw_idx = (uint)((ulong(tgid) * ulong(idx_lo)) % ulong(n));
        if (tw_idx != 0) val = stark252_mul(val, twiddles[tw_idx]);
        shared[rev_lo] = val;
    }
    if (base + idx_hi < n) {
        Stark252 val = input[base + idx_hi];
        uint tw_idx = (uint)((ulong(tgid) * ulong(idx_hi)) % ulong(n));
        if (tw_idx != 0) val = stark252_mul(val, twiddles[tw_idx]);
        shared[rev_hi] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // DIT butterfly stages
    for (uint s = 0; s < local_stages; s++) {
        uint half_block = 1u << s;
        uint local_block_size = half_block << 1;

        uint block_idx = tid / half_block;
        uint local_idx = tid % half_block;
        uint i = block_idx * local_block_size + local_idx;
        uint j = i + half_block;

        uint global_block_size = 1u << (s + 1);
        uint twiddle_idx = local_idx * (n / global_block_size);

        Stark252 a = shared[i];
        Stark252 b = shared[j];
        if (twiddle_idx == 0) {
            shared[i] = stark252_add(a, b);
            shared[j] = stark252_sub(a, b);
        } else {
            Stark252 w = twiddles[twiddle_idx];
            Stark252 wb = stark252_mul(w, b);
            shared[i] = stark252_add(a, wb);
            shared[j] = stark252_sub(a, wb);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write back transposed: output[col * N1 + row]
    uint row = tgid;
    uint col_lo = tid;
    uint col_hi = tid + tg_size;
    if (col_lo * n1 + row < n)
        output[col_lo * n1 + row] = shared[col_lo];
    if (col_hi * n1 + row < n)
        output[col_hi * n1 + row] = shared[col_hi];
}

// Twiddle multiply for four-step FFT: data[row * N2 + col] *= omega_N^(row * col)
kernel void stark252_ntt_twiddle_multiply(
    device Stark252* data                [[buffer(0)]],
    device const Stark252* twiddles      [[buffer(1)]],
    constant uint& n2              [[buffer(2)]],    // number of columns
    constant uint& n               [[buffer(3)]],    // total size N
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    uint row = gid / n2;
    uint col = gid % n2;

    uint twiddle_idx = (uint)((ulong(row) * ulong(col)) % ulong(n));
    if (twiddle_idx == 0) return;

    data[gid] = stark252_mul(data[gid], twiddles[twiddle_idx]);
}

// --- Four-step inverse FFT kernels ---
// DIF column iFFT: load column naturally, DIF stages, write back bit-reversed
kernel void stark252_intt_column_fused(
    device Stark252* data                [[buffer(0)]],
    device const Stark252* twiddles_inv  [[buffer(1)]],
    constant uint& n               [[buffer(2)]],    // total size N
    constant uint& n1              [[buffer(3)]],    // N1 (column size)
    constant uint& n2              [[buffer(4)]],    // N2 (number of columns)
    constant uint& local_stages    [[buffer(5)]],    // log2(N1)
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint col = tgid;
    threadgroup Stark252 shared[1024];

    // Load column naturally into shared memory
    if (tid < n1)
        shared[tid] = data[col + tid * n2];
    if (tid + tg_size < n1)
        shared[tid + tg_size] = data[col + (tid + tg_size) * n2];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // DIF stages (high to low)
    for (uint s = 0; s < local_stages; s++) {
        uint half_block = 1u << (local_stages - 1 - s);
        uint local_block_size = half_block << 1;

        uint block_idx = tid / half_block;
        uint local_idx = tid % half_block;
        uint i = block_idx * local_block_size + local_idx;
        uint j = i + half_block;

        // Twiddle: omega_N1_inv^(local_idx * N1/local_block_size) via N-point table
        uint twiddle_idx = local_idx * (n1 / local_block_size) * n2;

        Stark252 a = shared[i];
        Stark252 b = shared[j];
        Stark252 sum = stark252_add(a, b);
        Stark252 diff = stark252_sub(a, b);
        shared[i] = sum;
        if (twiddle_idx == 0) {
            shared[j] = diff;
        } else {
            Stark252 w = twiddles_inv[twiddle_idx];
            shared[j] = stark252_mul(diff, w);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write back: gather from bit-reversed positions in shared (fast), write sequentially to global
    // DIF output at shared[k] corresponds to natural index bitrev(k)
    uint rev_lo = bitrev(tid, local_stages);
    uint rev_hi = bitrev(tid + tg_size, local_stages);
    if (tid < n1)
        data[col + tid * n2] = shared[rev_lo];
    if (tid + tg_size < n1)
        data[col + (tid + tg_size) * n2] = shared[rev_hi];
}

// DIF column iFFT with inverse twiddle multiply fused into load phase
// and 1/N scale fused into writeback phase.
// Eliminates both the separate twiddle multiply pass and scale pass.
// Each threadgroup handles one column; during load, applies omega_inv^(row*col);
// during store, multiplies by inv_n.
kernel void stark252_intt_column_fused_twiddle(
    device Stark252* data                [[buffer(0)]],
    device const Stark252* twiddles_inv  [[buffer(1)]],
    constant uint& n               [[buffer(2)]],    // total size N
    constant uint& n1              [[buffer(3)]],    // N1 (column size)
    constant uint& n2              [[buffer(4)]],    // N2 (number of columns)
    constant uint& local_stages    [[buffer(5)]],    // log2(N1)
    device const Stark252* inv_n         [[buffer(6)]],    // 1/N in Montgomery form
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint col = tgid;
    Stark252 scale = inv_n[0];
    threadgroup Stark252 shared[1024];

    // Load column with inverse twiddle multiply: val *= omega_inv^(row * col)
    if (tid < n1) {
        Stark252 val = data[col + tid * n2];
        uint tw_idx = (uint)((ulong(tid) * ulong(col)) % ulong(n));
        if (tw_idx != 0) val = stark252_mul(val, twiddles_inv[tw_idx]);
        shared[tid] = val;
    }
    if (tid + tg_size < n1) {
        uint row = tid + tg_size;
        Stark252 val = data[col + row * n2];
        uint tw_idx = (uint)((ulong(row) * ulong(col)) % ulong(n));
        if (tw_idx != 0) val = stark252_mul(val, twiddles_inv[tw_idx]);
        shared[row] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // DIF stages (high to low) — identical to intt_column_fused
    for (uint s = 0; s < local_stages; s++) {
        uint half_block = 1u << (local_stages - 1 - s);
        uint local_block_size = half_block << 1;

        uint block_idx = tid / half_block;
        uint local_idx = tid % half_block;
        uint i = block_idx * local_block_size + local_idx;
        uint j = i + half_block;

        uint twiddle_idx = local_idx * (n1 / local_block_size) * n2;

        Stark252 a = shared[i];
        Stark252 b = shared[j];
        Stark252 sum = stark252_add(a, b);
        Stark252 diff = stark252_sub(a, b);
        shared[i] = sum;
        if (twiddle_idx == 0) {
            shared[j] = diff;
        } else {
            Stark252 w = twiddles_inv[twiddle_idx];
            shared[j] = stark252_mul(diff, w);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write back with bit-reversal and 1/N scale
    uint rev_lo = bitrev(tid, local_stages);
    uint rev_hi = bitrev(tid + tg_size, local_stages);
    if (tid < n1)
        data[col + tid * n2] = stark252_mul(shared[rev_lo], scale);
    if (tid + tg_size < n1)
        data[col + (tid + tg_size) * n2] = stark252_mul(shared[rev_hi], scale);
}

// DIF row iFFT: load row naturally, DIF stages, write back bit-reversed
kernel void stark252_intt_row_fused(
    device Stark252* data                [[buffer(0)]],
    device const Stark252* twiddles_inv  [[buffer(1)]],
    constant uint& n               [[buffer(2)]],    // total size N
    constant uint& local_stages    [[buffer(3)]],    // log2(N2)
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint block_size = tg_size << 1;
    uint base = tgid * block_size;
    threadgroup Stark252 shared[1024];

    // Load naturally
    if (base + tid < n)
        shared[tid] = data[base + tid];
    if (base + tid + tg_size < n)
        shared[tid + tg_size] = data[base + tid + tg_size];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // DIF stages (high to low)
    for (uint s = 0; s < local_stages; s++) {
        uint half_block = 1u << (local_stages - 1 - s);
        uint local_block_size = half_block << 1;

        uint block_idx = tid / half_block;
        uint local_idx = tid % half_block;
        uint i = block_idx * local_block_size + local_idx;
        uint j = i + half_block;

        // Twiddle: for N2-point DIF using N-point inverse twiddle table
        uint global_block_size = 1u << (local_stages - s);
        uint twiddle_idx = local_idx * (n / global_block_size);

        Stark252 a = shared[i];
        Stark252 b = shared[j];
        Stark252 sum = stark252_add(a, b);
        Stark252 diff = stark252_sub(a, b);
        shared[i] = sum;
        if (twiddle_idx == 0) {
            shared[j] = diff;
        } else {
            Stark252 w = twiddles_inv[twiddle_idx];
            shared[j] = stark252_mul(diff, w);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write back: gather from bit-reversed positions in shared, write contiguously to global
    uint rev_lo = bitrev(tid, local_stages);
    uint rev_hi = bitrev(tid + tg_size, local_stages);
    if (base + tid < n)
        data[base + tid] = shared[rev_lo];
    if (base + tid + tg_size < n)
        data[base + tid + tg_size] = shared[rev_hi];
}

// --- Extended four-step kernels for logN > 2*maxFusedLogN ---
// When N1 > 1024, column FFTs are decomposed into:
//   (a) Sub-block fused FFTs of 1024 elements each (first maxFusedLogN stages)
//   (b) Global butterfly stages within each column (remaining stages)

// DIT sub-block column FFT: processes a 1024-element sub-block within a larger column.
// Each threadgroup handles one (column, sub-block) pair.
// Dispatch: threadgroups = N2 * (N1/sub_size), threads_per_tg = sub_size/2
kernel void stark252_ntt_column_fused_subblock(
    device Stark252* data                [[buffer(0)]],
    device const Stark252* twiddles      [[buffer(1)]],
    constant uint& n               [[buffer(2)]],    // total size N
    constant uint& n1              [[buffer(3)]],    // N1 (full column size, e.g. 4096)
    constant uint& n2              [[buffer(4)]],    // N2 (number of columns / stride)
    constant uint& local_stages    [[buffer(5)]],    // log2(sub_size), e.g. 10
    constant uint& num_subblocks   [[buffer(6)]],    // N1 / sub_size
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint col = tgid / num_subblocks;
    uint sub_id = tgid % num_subblocks;
    uint sub_size = 1u << local_stages;  // 1024
    uint sub_base = sub_id * sub_size;   // row offset within column
    threadgroup Stark252 shared[1024];

    // Load from column with bit-reversed positions within sub-block
    uint idx_lo = tid;
    uint idx_hi = tid + tg_size;
    uint rev_lo = bitrev(idx_lo, local_stages);
    uint rev_hi = bitrev(idx_hi, local_stages);

    if (idx_lo < sub_size)
        shared[rev_lo] = data[col + (sub_base + idx_lo) * n2];
    if (idx_hi < sub_size)
        shared[rev_hi] = data[col + (sub_base + idx_hi) * n2];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // DIT butterfly stages — twiddles based on full N1
    for (uint s = 0; s < local_stages; s++) {
        uint half_block = 1u << s;
        uint local_block_size = half_block << 1;

        uint block_idx = tid / half_block;
        uint local_idx = tid % half_block;
        uint i = block_idx * local_block_size + local_idx;
        uint j = i + half_block;

        // Twiddle: omega_N1^(local_idx * N1/local_block_size) = omega_N^(local_idx * N1/local_block_size * N2)
        uint twiddle_idx = local_idx * (n1 / local_block_size) * n2;

        Stark252 a = shared[i];
        Stark252 b = shared[j];
        if (twiddle_idx == 0) {
            shared[i] = stark252_add(a, b);
            shared[j] = stark252_sub(a, b);
        } else {
            Stark252 w = twiddles[twiddle_idx];
            Stark252 wb = stark252_mul(w, b);
            shared[i] = stark252_add(a, wb);
            shared[j] = stark252_sub(a, wb);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write back to column (natural order within sub-block)
    if (tid < sub_size)
        data[col + (sub_base + tid) * n2] = shared[tid];
    if (tid + tg_size < sub_size)
        data[col + (sub_base + tid + tg_size) * n2] = shared[tid + tg_size];
}

// Global DIT butterfly within columns (strided access).
// For stages beyond the fused sub-block range.
// Each thread handles one butterfly pair across all columns.
kernel void stark252_ntt_column_butterfly(
    device Stark252* data                [[buffer(0)]],
    device const Stark252* twiddles      [[buffer(1)]],
    constant uint& n1              [[buffer(2)]],    // column size (N1)
    constant uint& n2              [[buffer(3)]],    // stride (N2)
    constant uint& stage           [[buffer(4)]],    // stage within column FFT (0-based)
    uint gid                       [[thread_position_in_grid]]
) {
    uint pairs_per_col = n1 >> 1;
    // Coalesced mapping: adjacent threads access adjacent columns
    uint col = gid % n2;
    uint pair_id = gid / n2;
    if (pair_id >= pairs_per_col) return;

    uint half_block = 1u << stage;
    uint block_size = half_block << 1;

    uint block_idx = pair_id / half_block;
    uint local_idx = pair_id % half_block;
    uint row_i = block_idx * block_size + local_idx;
    uint row_j = row_i + half_block;

    // Twiddle: omega_N1^(local_idx * N1/block_size) = omega_N^(local_idx * N1/block_size * N2)
    uint twiddle_idx = local_idx * (n1 / block_size) * n2;

    uint addr_i = col + row_i * n2;
    uint addr_j = col + row_j * n2;

    Stark252 a = data[addr_i];
    Stark252 b = data[addr_j];
    if (twiddle_idx == 0) {
        data[addr_i] = stark252_add(a, b);
        data[addr_j] = stark252_sub(a, b);
    } else {
        Stark252 w = twiddles[twiddle_idx];
        Stark252 wb = stark252_mul(w, b);
        data[addr_i] = stark252_add(a, wb);
        data[addr_j] = stark252_sub(a, wb);
    }
}

// Radix-4 DIT column butterfly: processes 2 stages at once (halves memory passes).
// For stages s and s+1 within the column FFT.
kernel void stark252_ntt_column_butterfly_radix4(
    device Stark252* data                [[buffer(0)]],
    device const Stark252* twiddles      [[buffer(1)]],
    constant uint& n1              [[buffer(2)]],    // column size (N1)
    constant uint& n2              [[buffer(3)]],    // stride (N2)
    constant uint& stage           [[buffer(4)]],    // lower stage index (processes stage and stage+1)
    uint gid                       [[thread_position_in_grid]]
) {
    uint quads_per_col = n1 >> 2;
    uint col = gid % n2;
    uint quad_id = gid / n2;
    if (quad_id >= quads_per_col) return;

    uint h = 1u << stage;
    uint block4 = h << 2;

    uint block_idx = quad_id / h;
    uint local_idx = quad_id % h;
    uint base_row = block_idx * block4 + local_idx;

    uint addr0 = col + base_row * n2;
    uint addr1 = col + (base_row + h) * n2;
    uint addr2 = col + (base_row + 2 * h) * n2;
    uint addr3 = col + (base_row + 3 * h) * n2;

    Stark252 a0 = data[addr0];
    Stark252 a1 = data[addr1];
    Stark252 a2 = data[addr2];
    Stark252 a3 = data[addr3];

    // Stage s twiddle: applied to pairs (a0,a1) and (a2,a3)
    uint tw_s = local_idx * (n1 / (2 * h)) * n2;
    Stark252 b0, b1, b2, b3;
    if (tw_s == 0) {
        b0 = stark252_add(a0, a1);
        b1 = stark252_sub(a0, a1);
        b2 = stark252_add(a2, a3);
        b3 = stark252_sub(a2, a3);
    } else {
        Stark252 ws = twiddles[tw_s];
        Stark252 ws_a1 = stark252_mul(ws, a1);
        Stark252 ws_a3 = stark252_mul(ws, a3);
        b0 = stark252_add(a0, ws_a1);
        b1 = stark252_sub(a0, ws_a1);
        b2 = stark252_add(a2, ws_a3);
        b3 = stark252_sub(a2, ws_a3);
    }

    // Stage s+1 twiddles: cross-combine
    uint tw_lo = local_idx * (n1 / block4) * n2;
    uint tw_hi = (local_idx + h) * (n1 / block4) * n2;
    Stark252 wb2 = (tw_lo == 0) ? b2 : stark252_mul(twiddles[tw_lo], b2);
    Stark252 wb3 = (tw_hi == 0) ? b3 : stark252_mul(twiddles[tw_hi], b3);

    data[addr0] = stark252_add(b0, wb2);
    data[addr2] = stark252_sub(b0, wb2);
    data[addr1] = stark252_add(b1, wb3);
    data[addr3] = stark252_sub(b1, wb3);
}

// Out-of-place rectangular transpose: input[r*cols+c] → output[c*rows+r]
kernel void stark252_ntt_transpose_rect(
    device const Stark252* input         [[buffer(0)]],
    device Stark252* output              [[buffer(1)]],
    constant uint& rows            [[buffer(2)]],    // N1
    constant uint& cols            [[buffer(3)]],    // N2
    uint gid                       [[thread_position_in_grid]]
) {
    uint total = rows * cols;
    if (gid >= total) return;
    uint r = gid / cols;
    uint c = gid % cols;
    output[c * rows + r] = input[gid];
}

// Global DIF butterfly within columns (for iNTT extended four-step).
// DIF processes stages from high to low; this handles the top stages.
kernel void stark252_intt_column_butterfly(
    device Stark252* data                [[buffer(0)]],
    device const Stark252* twiddles_inv  [[buffer(1)]],
    constant uint& n1              [[buffer(2)]],    // column size (N1)
    constant uint& n2              [[buffer(3)]],    // stride (N2)
    constant uint& stage           [[buffer(4)]],    // stage within column DIF (0-based from top)
    constant uint& log_n1          [[buffer(5)]],    // log2(N1)
    uint gid                       [[thread_position_in_grid]]
) {
    uint pairs_per_col = n1 >> 1;
    // Coalesced mapping: adjacent threads access adjacent columns
    uint col = gid % n2;
    uint pair_id = gid / n2;
    if (pair_id >= pairs_per_col) return;

    // DIF stage s (counting from top): half_block = 2^(logN1 - 1 - s)
    uint half_block = 1u << (log_n1 - 1 - stage);
    uint block_size = half_block << 1;

    uint block_idx = pair_id / half_block;
    uint local_idx = pair_id % half_block;
    uint row_i = block_idx * block_size + local_idx;
    uint row_j = row_i + half_block;

    // Twiddle: omega_N1_inv^(local_idx * N1/block_size) = omega_N_inv^(local_idx * N1/block_size * N2)
    uint twiddle_idx = local_idx * (n1 / block_size) * n2;

    uint addr_i = col + row_i * n2;
    uint addr_j = col + row_j * n2;

    Stark252 a = data[addr_i];
    Stark252 b = data[addr_j];
    Stark252 sum = stark252_add(a, b);
    Stark252 diff = stark252_sub(a, b);
    data[addr_i] = sum;
    if (twiddle_idx == 0) {
        data[addr_j] = diff;
    } else {
        Stark252 w = twiddles_inv[twiddle_idx];
        data[addr_j] = stark252_mul(diff, w);
    }
}

// Radix-4 DIF column butterfly for iNTT: processes 2 stages at once.
// stage is the higher (first) DIF stage index (counting from top).
kernel void stark252_intt_column_butterfly_radix4(
    device Stark252* data                [[buffer(0)]],
    device const Stark252* twiddles_inv  [[buffer(1)]],
    constant uint& n1              [[buffer(2)]],    // column size (N1)
    constant uint& n2              [[buffer(3)]],    // stride (N2)
    constant uint& stage           [[buffer(4)]],    // higher DIF stage index (from top)
    constant uint& log_n1          [[buffer(5)]],    // log2(N1)
    uint gid                       [[thread_position_in_grid]]
) {
    uint quads_per_col = n1 >> 2;
    uint col = gid % n2;
    uint quad_id = gid / n2;
    if (quad_id >= quads_per_col) return;

    // DIF stage s (high): half_block_hi = 2^(logN1 - 1 - stage)
    uint h_hi = 1u << (log_n1 - 1 - stage);
    // DIF stage s+1 (low): half_block_lo = h_hi >> 1
    uint h_lo = h_hi >> 1;
    uint block4 = h_hi << 1;  // = 4 * h_lo

    uint block_idx = quad_id / h_lo;
    uint local_idx = quad_id % h_lo;
    uint base_row = block_idx * block4 + local_idx;

    uint addr0 = col + base_row * n2;
    uint addr1 = col + (base_row + h_lo) * n2;
    uint addr2 = col + (base_row + 2 * h_lo) * n2;
    uint addr3 = col + (base_row + 3 * h_lo) * n2;

    Stark252 a0 = data[addr0];
    Stark252 a1 = data[addr1];
    Stark252 a2 = data[addr2];
    Stark252 a3 = data[addr3];

    // DIF high stage: butterflies (a0,a2) and (a1,a3) with stride h_hi
    Stark252 sum02 = stark252_add(a0, a2);
    Stark252 diff02 = stark252_sub(a0, a2);
    Stark252 sum13 = stark252_add(a1, a3);
    Stark252 diff13 = stark252_sub(a1, a3);

    // Twiddle for high stage
    uint tw_hi_idx = local_idx * (n1 / block4) * n2;
    uint tw_hi2_idx = (local_idx + h_lo) * (n1 / block4) * n2;
    if (tw_hi_idx != 0) {
        diff02 = stark252_mul(diff02, twiddles_inv[tw_hi_idx]);
    }
    if (tw_hi2_idx != 0) {
        diff13 = stark252_mul(diff13, twiddles_inv[tw_hi2_idx]);
    }

    // DIF low stage: butterflies (sum02,sum13) and (diff02,diff13)
    uint tw_lo_idx = local_idx * (n1 / (2 * h_lo)) * n2;

    Stark252 out0 = stark252_add(sum02, sum13);
    Stark252 d1 = stark252_sub(sum02, sum13);
    data[addr0] = out0;
    data[addr1] = (tw_lo_idx == 0) ? d1 : stark252_mul(d1, twiddles_inv[tw_lo_idx]);

    Stark252 out2 = stark252_add(diff02, diff13);
    Stark252 d3 = stark252_sub(diff02, diff13);
    data[addr2] = out2;
    data[addr3] = (tw_lo_idx == 0) ? d3 : stark252_mul(d3, twiddles_inv[tw_lo_idx]);
}

// DIF sub-block column iFFT with 1/N scale on store.
// Handles the bottom maxFusedLogN stages of a large column DIF.
// Twiddle multiply must be applied separately BEFORE this kernel.
// Each threadgroup = one (column, sub-block) pair.
kernel void stark252_intt_column_fused_subblock(
    device Stark252* data                [[buffer(0)]],
    device const Stark252* twiddles_inv  [[buffer(1)]],
    constant uint& n               [[buffer(2)]],    // total size N
    constant uint& n1              [[buffer(3)]],    // N1 (full column size)
    constant uint& n2              [[buffer(4)]],    // N2 (stride)
    constant uint& local_stages    [[buffer(5)]],    // log2(sub_size)
    constant uint& num_subblocks   [[buffer(6)]],    // N1 / sub_size
    device const Stark252* inv_n         [[buffer(7)]],    // 1/N in Montgomery form
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint col = tgid / num_subblocks;
    uint sub_id = tgid % num_subblocks;
    uint sub_size = 1u << local_stages;
    uint sub_base = sub_id * sub_size;
    Stark252 scale = inv_n[0];
    threadgroup Stark252 shared[1024];

    // Load sub-block naturally (twiddle already applied)
    uint idx_lo = tid;
    uint idx_hi = tid + tg_size;
    if (idx_lo < sub_size)
        shared[idx_lo] = data[col + (sub_base + idx_lo) * n2];
    if (idx_hi < sub_size)
        shared[idx_hi] = data[col + (sub_base + idx_hi) * n2];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // DIF stages (high to low within sub-block) — twiddles based on full N1
    for (uint s = 0; s < local_stages; s++) {
        uint half_block = 1u << (local_stages - 1 - s);
        uint local_block_size = half_block << 1;

        uint block_idx = tid / half_block;
        uint local_idx = tid % half_block;
        uint i = block_idx * local_block_size + local_idx;
        uint j = i + half_block;

        uint twiddle_idx = local_idx * (n1 / local_block_size) * n2;

        Stark252 a = shared[i];
        Stark252 b = shared[j];
        Stark252 sum = stark252_add(a, b);
        Stark252 diff = stark252_sub(a, b);
        shared[i] = sum;
        if (twiddle_idx == 0) {
            shared[j] = diff;
        } else {
            Stark252 w = twiddles_inv[twiddle_idx];
            shared[j] = stark252_mul(diff, w);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write back with bit-reversal and 1/N scale
    uint rev_lo = bitrev(tid, local_stages);
    uint rev_hi = bitrev(tid + tg_size, local_stages);
    if (tid < sub_size)
        data[col + (sub_base + tid) * n2] = stark252_mul(shared[rev_lo], scale);
    if (tid + tg_size < sub_size)
        data[col + (sub_base + tid + tg_size) * n2] = stark252_mul(shared[rev_hi], scale);
}

// DIF row iFFT with inverse twiddle fused into writeback:
// After DIF stages, each output element is multiplied by omega_inv^(row * col).
// Eliminates a separate twiddle multiply pass in the extended four-step iNTT.
kernel void stark252_intt_row_fused_twiddle(
    device Stark252* data                [[buffer(0)]],
    device const Stark252* twiddles_inv  [[buffer(1)]],
    constant uint& n               [[buffer(2)]],    // total size N
    constant uint& local_stages    [[buffer(3)]],    // log2(N2)
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint block_size = tg_size << 1;
    uint base = tgid * block_size;
    uint row = tgid;
    threadgroup Stark252 shared[1024];

    // Load naturally
    if (base + tid < n)
        shared[tid] = data[base + tid];
    if (base + tid + tg_size < n)
        shared[tid + tg_size] = data[base + tid + tg_size];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // DIF stages (high to low) — identical to intt_row_fused
    for (uint s = 0; s < local_stages; s++) {
        uint half_block = 1u << (local_stages - 1 - s);
        uint local_block_size = half_block << 1;

        uint block_idx = tid / half_block;
        uint local_idx = tid % half_block;
        uint i = block_idx * local_block_size + local_idx;
        uint j = i + half_block;

        uint global_block_size = 1u << (local_stages - s);
        uint twiddle_idx = local_idx * (n / global_block_size);

        Stark252 a = shared[i];
        Stark252 b = shared[j];
        Stark252 sum = stark252_add(a, b);
        Stark252 diff = stark252_sub(a, b);
        shared[i] = sum;
        if (twiddle_idx == 0) {
            shared[j] = diff;
        } else {
            Stark252 w = twiddles_inv[twiddle_idx];
            shared[j] = stark252_mul(diff, w);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write back with bit-reversal AND inverse twiddle: val *= omega_inv^(row * col)
    uint col_lo = tid;
    uint col_hi = tid + tg_size;
    uint rev_lo = bitrev(col_lo, local_stages);
    uint rev_hi = bitrev(col_hi, local_stages);

    if (base + col_lo < n) {
        Stark252 val = shared[rev_lo];
        uint tw_idx = (uint)((ulong(row) * ulong(col_lo)) % ulong(n));
        if (tw_idx != 0) val = stark252_mul(val, twiddles_inv[tw_idx]);
        data[base + col_lo] = val;
    }
    if (base + col_hi < n) {
        Stark252 val = shared[rev_hi];
        uint tw_idx = (uint)((ulong(row) * ulong(col_hi)) % ulong(n));
        if (tw_idx != 0) val = stark252_mul(val, twiddles_inv[tw_idx]);
        data[base + col_hi] = val;
    }
}

// In-place transpose of square N×N matrix: swap data[row*N+col] with data[col*N+row]
// Only processes pairs where row < col to avoid double-swapping.
kernel void stark252_ntt_transpose(
    device Stark252* data                [[buffer(0)]],
    constant uint& n_side          [[buffer(1)]],    // matrix side length (N1 = N2)
    uint gid                       [[thread_position_in_grid]]
) {
    uint total = n_side * n_side;
    if (gid >= total) return;

    uint row = gid / n_side;
    uint col = gid % n_side;
    if (row >= col) return;  // only upper triangle

    uint i = row * n_side + col;
    uint j = col * n_side + row;
    Stark252 tmp = data[i];
    data[i] = data[j];
    data[j] = tmp;
}
