// Vesta Fr NTT GPU kernels
// Vesta scalar field Fr = Pallas base field Fp (cycle property)
// Uses PallasFp arithmetic for all field operations.
// Cooley-Tukey radix-2 DIT forward, Gentleman-Sande radix-2 DIF inverse.

#include "../fields/pallas_fp.metal"

// Type alias for clarity: NTT operates on Vesta Fr elements = PallasFp
typedef PallasFp VestaFr;

inline VestaFr vfr_add(VestaFr a, VestaFr b) { return pallas_add(a, b); }
inline VestaFr vfr_sub(VestaFr a, VestaFr b) { return pallas_sub(a, b); }
inline VestaFr vfr_mul(VestaFr a, VestaFr b) { return pallas_mul(a, b); }

// Helper: compute bit-reversal of val with num_bits bits
inline uint vesta_bitrev(uint val, uint num_bits) {
    uint result = 0;
    for (uint i = 0; i < num_bits; i++) {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    return result;
}

// --- Forward NTT Butterfly (one stage per dispatch) ---
kernel void vesta_ntt_butterfly(
    device VestaFr* data           [[buffer(0)]],
    device const VestaFr* twiddles [[buffer(1)]],
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

    VestaFr a = data[i];
    VestaFr b = data[j];
    if (twiddle_idx == 0) {
        data[i] = vfr_add(a, b);
        data[j] = vfr_sub(a, b);
    } else {
        VestaFr w = twiddles[twiddle_idx];
        VestaFr wb = vfr_mul(w, b);
        data[i] = vfr_add(a, wb);
        data[j] = vfr_sub(a, wb);
    }
}

// --- Inverse NTT Butterfly (Gentleman-Sande DIF) ---
kernel void vesta_intt_butterfly(
    device VestaFr* data               [[buffer(0)]],
    device const VestaFr* twiddles_inv [[buffer(1)]],
    constant uint& n                   [[buffer(2)]],
    constant uint& stage               [[buffer(3)]],
    uint gid                           [[thread_position_in_grid]]
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

    VestaFr a = data[i];
    VestaFr b = data[j];

    VestaFr sum = vfr_add(a, b);
    VestaFr diff = vfr_sub(a, b);
    data[i] = sum;
    if (twiddle_idx == 0) {
        data[j] = diff;
    } else {
        data[j] = vfr_mul(diff, twiddles_inv[twiddle_idx]);
    }
}

// --- Fused NTT: multiple DIT stages in threadgroup memory ---
kernel void vesta_ntt_butterfly_fused(
    device VestaFr* data           [[buffer(0)]],
    device const VestaFr* twiddles [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& local_stages    [[buffer(3)]],
    constant uint& stage_offset    [[buffer(4)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint block_size = tg_size << 1;
    uint base = tgid * block_size;

    threadgroup VestaFr shared[1024];
    if (base + tid < n)
        shared[tid] = data[base + tid];
    if (base + tid + tg_size < n)
        shared[tid + tg_size] = data[base + tid + tg_size];
    threadgroup_barrier(mem_flags::mem_threadgroup);

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

        VestaFr a = shared[i];
        VestaFr b = shared[j];
        if (twiddle_idx == 0) {
            shared[i] = vfr_add(a, b);
            shared[j] = vfr_sub(a, b);
        } else {
            VestaFr w = twiddles[twiddle_idx];
            VestaFr wb = vfr_mul(w, b);
            shared[i] = vfr_add(a, wb);
            shared[j] = vfr_sub(a, wb);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (base + tid < n)
        data[base + tid] = shared[tid];
    if (base + tid + tg_size < n)
        data[base + tid + tg_size] = shared[tid + tg_size];
}

// --- Fused DIT with bit-reversed loading ---
kernel void vesta_ntt_butterfly_fused_bitrev(
    device const VestaFr* input    [[buffer(0)]],
    device VestaFr* output         [[buffer(1)]],
    device const VestaFr* twiddles [[buffer(2)]],
    constant uint& n               [[buffer(3)]],
    constant uint& local_stages    [[buffer(4)]],
    constant uint& logN            [[buffer(5)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint block_size = tg_size << 1;
    uint base = tgid * block_size;

    threadgroup VestaFr shared[1024];

    uint idx_lo = tid;
    uint idx_hi = tid + tg_size;
    uint global_lo = base + idx_lo;
    uint global_hi = base + idx_hi;

    if (global_lo < n)
        shared[idx_lo] = input[vesta_bitrev(global_lo, logN)];
    if (global_hi < n)
        shared[idx_hi] = input[vesta_bitrev(global_hi, logN)];
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

        VestaFr a = shared[i];
        VestaFr b = shared[j];
        if (twiddle_idx == 0) {
            shared[i] = vfr_add(a, b);
            shared[j] = vfr_sub(a, b);
        } else {
            VestaFr w = twiddles[twiddle_idx];
            VestaFr wb = vfr_mul(w, b);
            shared[i] = vfr_add(a, wb);
            shared[j] = vfr_sub(a, wb);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (global_lo < n)
        output[global_lo] = shared[idx_lo];
    if (global_hi < n)
        output[global_hi] = shared[idx_hi];
}

// --- Fused iNTT: multiple DIF stages in threadgroup memory ---
kernel void vesta_intt_butterfly_fused(
    device VestaFr* data                   [[buffer(0)]],
    device const VestaFr* twiddles_inv     [[buffer(1)]],
    constant uint& n                        [[buffer(2)]],
    constant uint& local_stages             [[buffer(3)]],
    constant uint& stage_offset             [[buffer(4)]],
    uint tid                                [[thread_index_in_threadgroup]],
    uint tgid                               [[threadgroup_position_in_grid]],
    uint tg_size                            [[threads_per_threadgroup]]
) {
    uint block_size = tg_size << 1;
    uint base = tgid * block_size;

    threadgroup VestaFr shared[1024];
    if (base + tid < n)
        shared[tid] = data[base + tid];
    if (base + tid + tg_size < n)
        shared[tid + tg_size] = data[base + tid + tg_size];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint si = 0; si < local_stages; si++) {
        uint s = local_stages - 1 - si;
        uint half_block = 1u << s;
        uint local_block_size = half_block << 1;

        uint block_idx = tid / half_block;
        uint local_idx = tid % half_block;
        uint i = block_idx * local_block_size + local_idx;
        uint j = i + half_block;

        uint stage = stage_offset + s;
        uint global_block_size = 1u << (stage + 1);
        uint twiddle_idx = local_idx * (n / global_block_size);

        VestaFr a = shared[i];
        VestaFr b = shared[j];

        VestaFr sum = vfr_add(a, b);
        VestaFr diff = vfr_sub(a, b);
        shared[i] = sum;
        if (twiddle_idx == 0) {
            shared[j] = diff;
        } else {
            shared[j] = vfr_mul(diff, twiddles_inv[twiddle_idx]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (base + tid < n)
        data[base + tid] = shared[tid];
    if (base + tid + tg_size < n)
        data[base + tid + tg_size] = shared[tid + tg_size];
}

// --- Bit-reversal in-place ---
kernel void vesta_ntt_bitrev_inplace(
    device VestaFr* data   [[buffer(0)]],
    constant uint& n       [[buffer(1)]],
    constant uint& logN    [[buffer(2)]],
    uint gid               [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    uint rev = vesta_bitrev(gid, logN);
    if (gid < rev) {
        VestaFr tmp = data[gid];
        data[gid] = data[rev];
        data[rev] = tmp;
    }
}

// --- Scale by 1/n ---
kernel void vesta_ntt_scale(
    device VestaFr* data         [[buffer(0)]],
    device const VestaFr* inv_n  [[buffer(1)]],
    constant uint& n             [[buffer(2)]],
    uint gid                     [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    data[gid] = vfr_mul(data[gid], inv_n[0]);
}

// --- Bit-reversal + scale (combined for iNTT final step) ---
kernel void vesta_ntt_bitrev_scale(
    device VestaFr* data         [[buffer(0)]],
    device const VestaFr* inv_n  [[buffer(1)]],
    constant uint& n             [[buffer(2)]],
    constant uint& logN          [[buffer(3)]],
    uint gid                     [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    uint rev = vesta_bitrev(gid, logN);
    if (gid <= rev) {
        VestaFr a = vfr_mul(data[gid], inv_n[0]);
        VestaFr b = vfr_mul(data[rev], inv_n[0]);
        data[gid] = b;
        data[rev] = a;
    }
}
