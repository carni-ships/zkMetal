// RNS NTT kernels for Homomorphic Encryption (CKKS/BFV)
// Each modulus qi is ~30 bits, fitting in a single uint32.
// Barrett reduction for multiplication: avoids expensive 64-bit modulo.
// Batch NTT: each threadgroup handles one (polynomial, modulus) pair.

#include <metal_stdlib>
using namespace metal;

// Barrett reduction: compute (a * b) mod q for 30-bit q
// a, b < q < 2^30, so a*b < 2^60, fits in ulong
// We precompute barrett_k = floor(2^62 / q) for each modulus.
inline uint rns_mul_mod(uint a, uint b, uint q, uint barrett_k) {
    ulong prod = ulong(a) * ulong(b);
    // Barrett: estimate quotient = (prod * barrett_k) >> 62
    uint prod_lo = uint(prod);
    uint prod_hi = uint(prod >> 32);
    ulong t1 = ulong(prod_lo) * ulong(barrett_k);
    ulong t2 = ulong(prod_hi) * ulong(barrett_k);
    uint est_q = uint((t2 + (t1 >> 32)) >> 30);
    uint r = uint(prod - ulong(est_q) * ulong(q));
    // At most one correction
    return r >= q ? r - q : r;
}

inline uint rns_add_mod(uint a, uint b, uint q) {
    uint s = a + b;
    return s >= q ? s - q : s;
}

inline uint rns_sub_mod(uint a, uint b, uint q) {
    return a >= b ? a - b : a + q - b;
}

// Batch NTT: one threadgroup per (polynomial, modulus) pair
// Uses threadgroup shared memory for the entire polynomial (N * 4 bytes)
// N must be <= 16384 (64KB shared memory limit / 4 bytes)
kernel void rns_ntt_batch(
    device uint* data               [[buffer(0)]],  // packed: poly_count * num_limbs * N
    constant uint* twiddles         [[buffer(1)]],  // num_limbs * N/2 twiddle factors
    constant uint* moduli           [[buffer(2)]],  // num_limbs moduli
    constant uint* barrett_ks       [[buffer(3)]],  // num_limbs Barrett constants
    constant uint& log_n            [[buffer(4)]],
    constant uint& num_limbs        [[buffer(5)]],
    constant uint& poly_count       [[buffer(6)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint tg_sz  [[threads_per_threadgroup]],
    threadgroup uint* shared        [[threadgroup(0)]]
) {
    uint limb_idx = tgid % num_limbs;
    uint poly_idx = tgid / num_limbs;
    if (poly_idx >= poly_count) return;

    uint n = 1u << log_n;
    uint half_n = n >> 1;
    uint q = moduli[limb_idx];
    uint bk = barrett_ks[limb_idx];

    // Base offset in data buffer
    uint base = (poly_idx * num_limbs + limb_idx) * n;

    // Twiddle base for this modulus
    uint tw_base = limb_idx * half_n;

    // Load into shared memory (multiple elements per thread if needed)
    for (uint i = lid; i < n; i += tg_sz) {
        shared[i] = data[base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cooley-Tukey DIT butterfly stages
    for (uint stage = 0; stage < log_n; stage++) {
        uint half_block = 1u << stage;
        uint block_size = half_block << 1;
        uint num_butterflies = half_n;

        for (uint gid = lid; gid < num_butterflies; gid += tg_sz) {
            uint block_idx = gid / half_block;
            uint local_idx = gid % half_block;
            uint i = block_idx * block_size + local_idx;
            uint j = i + half_block;
            uint tw_idx = local_idx * (n / block_size);

            uint a = shared[i];
            uint b = shared[j];
            uint w = twiddles[tw_base + tw_idx];
            uint wb = rns_mul_mod(w, b, q, bk);

            shared[i] = rns_add_mod(a, wb, q);
            shared[j] = rns_sub_mod(a, wb, q);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store back
    for (uint i = lid; i < n; i += tg_sz) {
        data[base + i] = shared[i];
    }
}

// Batch inverse NTT: Gentleman-Sande DIF
kernel void rns_intt_batch(
    device uint* data               [[buffer(0)]],
    constant uint* inv_twiddles     [[buffer(1)]],  // num_limbs * N/2 inverse twiddle factors
    constant uint* moduli           [[buffer(2)]],
    constant uint* barrett_ks       [[buffer(3)]],
    constant uint* inv_ns           [[buffer(4)]],   // num_limbs inverse-of-N values
    constant uint& log_n            [[buffer(5)]],
    constant uint& num_limbs        [[buffer(6)]],
    constant uint& poly_count       [[buffer(7)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint tg_sz  [[threads_per_threadgroup]],
    threadgroup uint* shared        [[threadgroup(0)]]
) {
    uint limb_idx = tgid % num_limbs;
    uint poly_idx = tgid / num_limbs;
    if (poly_idx >= poly_count) return;

    uint n = 1u << log_n;
    uint half_n = n >> 1;
    uint q = moduli[limb_idx];
    uint bk = barrett_ks[limb_idx];
    uint inv_n = inv_ns[limb_idx];

    uint base = (poly_idx * num_limbs + limb_idx) * n;
    uint tw_base = limb_idx * half_n;

    // Load
    for (uint i = lid; i < n; i += tg_sz) {
        shared[i] = data[base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // DIF butterfly stages (reverse order)
    for (uint si = 0; si < log_n; si++) {
        uint stage = log_n - 1 - si;
        uint half_block = 1u << stage;
        uint block_size = half_block << 1;
        uint num_butterflies = half_n;

        for (uint gid = lid; gid < num_butterflies; gid += tg_sz) {
            uint block_idx = gid / half_block;
            uint local_idx = gid % half_block;
            uint i = block_idx * block_size + local_idx;
            uint j = i + half_block;
            uint tw_idx = local_idx * (n / block_size);

            uint a = shared[i];
            uint b = shared[j];
            uint sum = rns_add_mod(a, b, q);
            uint diff = rns_sub_mod(a, b, q);
            uint w = inv_twiddles[tw_base + tw_idx];

            shared[i] = sum;
            shared[j] = rns_mul_mod(diff, w, q, bk);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Scale by 1/N and store
    for (uint i = lid; i < n; i += tg_sz) {
        data[base + i] = rns_mul_mod(shared[i], inv_n, q, bk);
    }
}

// Pointwise multiply two RNS polynomials in NTT domain
// Each thread handles one coefficient of one limb
kernel void rns_pointwise_mul(
    device const uint* a            [[buffer(0)]],
    device const uint* b            [[buffer(1)]],
    device uint* out                [[buffer(2)]],
    constant uint* moduli           [[buffer(3)]],
    constant uint* barrett_ks       [[buffer(4)]],
    constant uint& num_limbs        [[buffer(5)]],
    constant uint& degree           [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = num_limbs * degree;
    if (tid >= total) return;

    uint limb = tid / degree;
    uint q = moduli[limb];
    uint bk = barrett_ks[limb];

    out[tid] = rns_mul_mod(a[tid], b[tid], q, bk);
}

// Pointwise add two RNS polynomials
kernel void rns_pointwise_add(
    device const uint* a            [[buffer(0)]],
    device const uint* b            [[buffer(1)]],
    device uint* out                [[buffer(2)]],
    constant uint* moduli           [[buffer(3)]],
    constant uint& num_limbs        [[buffer(4)]],
    constant uint& degree           [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = num_limbs * degree;
    if (tid >= total) return;

    uint limb = tid / degree;
    uint q = moduli[limb];

    out[tid] = rns_add_mod(a[tid], b[tid], q);
}

// Global NTT butterfly for sizes > threadgroup memory (N > 16384)
// One butterfly per thread, no shared memory needed
kernel void rns_ntt_butterfly_global(
    device uint* data               [[buffer(0)]],
    constant uint* twiddles         [[buffer(1)]],
    constant uint& q                [[buffer(2)]],
    constant uint& barrett_k        [[buffer(3)]],
    constant uint& n                [[buffer(4)]],
    constant uint& stage            [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint half_block = 1u << stage;
    uint block_size = half_block << 1;
    uint num_butterflies = n >> 1;
    if (gid >= num_butterflies) return;

    uint block_idx = gid / half_block;
    uint local_idx = gid % half_block;
    uint i = block_idx * block_size + local_idx;
    uint j = i + half_block;
    uint tw_idx = local_idx * (n / block_size);

    uint a = data[i];
    uint b = data[j];
    uint w = twiddles[tw_idx];
    uint wb = rns_mul_mod(w, b, q, barrett_k);

    data[i] = rns_add_mod(a, wb, q);
    data[j] = rns_sub_mod(a, wb, q);
}

// Scale kernel for inverse NTT
kernel void rns_ntt_scale(
    device uint* data               [[buffer(0)]],
    constant uint& scalar           [[buffer(1)]],
    constant uint& q                [[buffer(2)]],
    constant uint& barrett_k        [[buffer(3)]],
    constant uint& n                [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    data[gid] = rns_mul_mod(data[gid], scalar, q, barrett_k);
}
