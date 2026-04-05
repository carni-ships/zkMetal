// GPU-accelerated sumcheck reduce and round polynomial kernels
// Multi-field: BN254 (Fr, 8x uint32 Montgomery), BabyBear (uint32), Goldilocks (uint64)
//
// Two kernel families per field:
// 1. sumcheck_reduce_*: fold multilinear table by challenge, halving size
//    out[i] = table[i]*(1-r) + table[i + half]*r = table[i] + r*(table[i+half] - table[i])
// 2. sumcheck_round_poly_*: compute degree-1 round univariate [sum_f0, sum_f1]
//    s0 = sum_{x} f(0,x),  s1 = sum_{x} f(1,x)

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// BN254 Fr field (8x uint32 Montgomery)
// ============================================================================

#include "../fields/bn254_fr.metal"

// SIMD-level Fr reduction using component-wise shuffle
inline Fr sc_simd_reduce_fr(Fr val) {
    for (uint offset = 16; offset > 0; offset >>= 1) {
        Fr other;
        for (int k = 0; k < 8; k++) {
            other.v[k] = simd_shuffle_down(val.v[k], offset);
        }
        val = fr_add(val, other);
    }
    return val;
}

// Fold multilinear evaluations: out[i] = evals[i] + r * (evals[i+half] - evals[i])
kernel void sumcheck_reduce_bn254(
    device const Fr* evals          [[buffer(0)]],
    device Fr* evals_out            [[buffer(1)]],
    constant Fr* challenge          [[buffer(2)]],
    constant uint& half_n           [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid >= half_n) return;

    Fr a = evals[gid];
    Fr b = evals[gid + half_n];
    Fr r = challenge[0];

    Fr diff = fr_sub(b, a);
    Fr r_diff = fr_mul(r, diff);
    evals_out[gid] = fr_add(a, r_diff);
}

// Compute round polynomial partial sums: s0 = sum(first_half), s1 = sum(second_half)
// Uses threadgroup reduction. Output: partial_sums[tgid*2] = s0, partial_sums[tgid*2+1] = s1
kernel void sumcheck_round_poly_bn254(
    device const Fr* evals          [[buffer(0)]],
    device Fr* partial_sums         [[buffer(1)]],
    constant uint& half_n           [[buffer(2)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint tg_size                    [[threads_per_threadgroup]],
    uint simd_lane                  [[thread_index_in_simdgroup]],
    uint simd_id                    [[simdgroup_index_in_threadgroup]]
) {
    // Inter-SIMD shared memory (max 8 SIMDs per threadgroup of 256)
    threadgroup Fr shared_s0[8];
    threadgroup Fr shared_s1[8];

    Fr local_s0 = fr_zero();
    Fr local_s1 = fr_zero();

    uint global_idx = tgid * tg_size + tid;

    if (global_idx < half_n) {
        local_s0 = evals[global_idx];
        local_s1 = evals[global_idx + half_n];
    }

    // SIMD-level reduction
    local_s0 = sc_simd_reduce_fr(local_s0);
    local_s1 = sc_simd_reduce_fr(local_s1);

    if (simd_lane == 0) {
        shared_s0[simd_id] = local_s0;
        shared_s1[simd_id] = local_s1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction by first SIMD group
    uint num_simds = (tg_size + 31) / 32;
    if (simd_id == 0 && simd_lane < num_simds) {
        local_s0 = shared_s0[simd_lane];
        local_s1 = shared_s1[simd_lane];
    } else if (simd_id == 0) {
        local_s0 = fr_zero();
        local_s1 = fr_zero();
    }

    if (simd_id == 0) {
        local_s0 = sc_simd_reduce_fr(local_s0);
        local_s1 = sc_simd_reduce_fr(local_s1);
        if (simd_lane == 0) {
            partial_sums[tgid * 2] = local_s0;
            partial_sums[tgid * 2 + 1] = local_s1;
        }
    }
}

// Fused: compute round poly AND reduce in one pass (reads evals once)
// Outputs: partial_sums[tgid*2], partial_sums[tgid*2+1] for round poly
//          evals_out[i] = folded value for next round
kernel void sumcheck_fused_round_reduce_bn254(
    device const Fr* evals          [[buffer(0)]],
    device Fr* evals_out            [[buffer(1)]],
    device Fr* partial_sums         [[buffer(2)]],
    constant Fr* challenge          [[buffer(3)]],
    constant uint& half_n           [[buffer(4)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint tg_size                    [[threads_per_threadgroup]],
    uint simd_lane                  [[thread_index_in_simdgroup]],
    uint simd_id                    [[simdgroup_index_in_threadgroup]]
) {
    threadgroup Fr shared_s0[8];
    threadgroup Fr shared_s1[8];

    Fr local_s0 = fr_zero();
    Fr local_s1 = fr_zero();
    Fr r = challenge[0];

    uint global_idx = tgid * tg_size + tid;

    if (global_idx < half_n) {
        Fr a = evals[global_idx];
        Fr b = evals[global_idx + half_n];
        local_s0 = a;
        local_s1 = b;

        // Write folded output
        Fr diff = fr_sub(b, a);
        Fr r_diff = fr_mul(r, diff);
        evals_out[global_idx] = fr_add(a, r_diff);
    }

    // Reduce for round poly
    local_s0 = sc_simd_reduce_fr(local_s0);
    local_s1 = sc_simd_reduce_fr(local_s1);

    if (simd_lane == 0) {
        shared_s0[simd_id] = local_s0;
        shared_s1[simd_id] = local_s1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simds = (tg_size + 31) / 32;
    if (simd_id == 0 && simd_lane < num_simds) {
        local_s0 = shared_s0[simd_lane];
        local_s1 = shared_s1[simd_lane];
    } else if (simd_id == 0) {
        local_s0 = fr_zero();
        local_s1 = fr_zero();
    }

    if (simd_id == 0) {
        local_s0 = sc_simd_reduce_fr(local_s0);
        local_s1 = sc_simd_reduce_fr(local_s1);
        if (simd_lane == 0) {
            partial_sums[tgid * 2] = local_s0;
            partial_sums[tgid * 2 + 1] = local_s1;
        }
    }
}

// ============================================================================
// BabyBear field (single uint32, p = 0x78000001)
// ============================================================================

constant uint BB_P_SC = 0x78000001u;

struct BbSc { uint v; };

BbSc bbsc_zero() { return BbSc{0}; }

BbSc bbsc_add(BbSc a, BbSc b) {
    uint sum = a.v + b.v;
    return BbSc{sum >= BB_P_SC ? sum - BB_P_SC : sum};
}

BbSc bbsc_sub(BbSc a, BbSc b) {
    if (a.v >= b.v) return BbSc{a.v - b.v};
    return BbSc{a.v + BB_P_SC - b.v};
}

constant uint BB_MU_SC = 2290649223u;

BbSc bbsc_mul(BbSc a, BbSc b) {
    ulong prod = ulong(a.v) * ulong(b.v);
    uint prod_lo = uint(prod);
    uint prod_hi = uint(prod >> 32);
    ulong t1 = ulong(prod_lo) * ulong(BB_MU_SC);
    ulong t2 = ulong(prod_hi) * ulong(BB_MU_SC);
    uint q = uint((t2 + (t1 >> 32)) >> 30);
    uint r = uint(prod - ulong(q) * ulong(BB_P_SC));
    return BbSc{r >= BB_P_SC ? r - BB_P_SC : r};
}

inline BbSc bbsc_simd_reduce(BbSc val) {
    for (uint offset = 16; offset > 0; offset >>= 1) {
        BbSc other;
        other.v = simd_shuffle_down(val.v, offset);
        val = bbsc_add(val, other);
    }
    return val;
}

kernel void sumcheck_reduce_babybear(
    device const uint* evals        [[buffer(0)]],
    device uint* evals_out          [[buffer(1)]],
    constant uint* challenge        [[buffer(2)]],
    constant uint& half_n           [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid >= half_n) return;

    BbSc a = BbSc{evals[gid]};
    BbSc b = BbSc{evals[gid + half_n]};
    BbSc r = BbSc{challenge[0]};

    BbSc diff = bbsc_sub(b, a);
    BbSc r_diff = bbsc_mul(r, diff);
    evals_out[gid] = bbsc_add(a, r_diff).v;
}

kernel void sumcheck_round_poly_babybear(
    device const uint* evals        [[buffer(0)]],
    device uint* partial_sums       [[buffer(1)]],
    constant uint& half_n           [[buffer(2)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint tg_size                    [[threads_per_threadgroup]],
    uint simd_lane                  [[thread_index_in_simdgroup]],
    uint simd_id                    [[simdgroup_index_in_threadgroup]]
) {
    threadgroup uint shared_s0[8];
    threadgroup uint shared_s1[8];

    BbSc local_s0 = bbsc_zero();
    BbSc local_s1 = bbsc_zero();

    uint global_idx = tgid * tg_size + tid;
    if (global_idx < half_n) {
        local_s0 = BbSc{evals[global_idx]};
        local_s1 = BbSc{evals[global_idx + half_n]};
    }

    local_s0 = bbsc_simd_reduce(local_s0);
    local_s1 = bbsc_simd_reduce(local_s1);

    if (simd_lane == 0) {
        shared_s0[simd_id] = local_s0.v;
        shared_s1[simd_id] = local_s1.v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simds = (tg_size + 31) / 32;
    if (simd_id == 0 && simd_lane < num_simds) {
        local_s0 = BbSc{shared_s0[simd_lane]};
        local_s1 = BbSc{shared_s1[simd_lane]};
    } else if (simd_id == 0) {
        local_s0 = bbsc_zero();
        local_s1 = bbsc_zero();
    }

    if (simd_id == 0) {
        local_s0 = bbsc_simd_reduce(local_s0);
        local_s1 = bbsc_simd_reduce(local_s1);
        if (simd_lane == 0) {
            partial_sums[tgid * 2] = local_s0.v;
            partial_sums[tgid * 2 + 1] = local_s1.v;
        }
    }
}

// ============================================================================
// Goldilocks field (uint64, p = 2^64 - 2^32 + 1)
// ============================================================================

constant ulong GL_P_SC = 0xFFFFFFFF00000001UL;
constant ulong GL_EPS_SC = 0xFFFFFFFFUL;

struct GlSc { ulong v; };

GlSc glsc_zero() { return GlSc{0}; }

GlSc glsc_add(GlSc a, GlSc b) {
    ulong sum = a.v + b.v;
    sum += (sum < a.v) ? GL_EPS_SC : 0UL;
    return GlSc{sum >= GL_P_SC ? sum - GL_P_SC : sum};
}

GlSc glsc_sub(GlSc a, GlSc b) {
    if (a.v >= b.v) return GlSc{a.v - b.v};
    return GlSc{a.v + GL_P_SC - b.v};
}

GlSc glsc_mul(GlSc a, GlSc b) {
    uint a0 = uint(a.v);
    uint a1 = uint(a.v >> 32);
    uint b0 = uint(b.v);
    uint b1 = uint(b.v >> 32);

    ulong t0 = ulong(a0) * ulong(b0);
    ulong t1 = ulong(a0) * ulong(b1);
    ulong t2 = ulong(a1) * ulong(b0);
    ulong t3 = ulong(a1) * ulong(b1);

    ulong mid = t1 + t2;
    bool mid_carry = mid < t1;

    ulong lo = t0 + (mid << 32);
    bool lo_carry = lo < t0;

    ulong hi = t3 + (mid >> 32) + (lo_carry ? 1UL : 0UL);
    if (mid_carry) hi += (1UL << 32);

    uint hi_lo = uint(hi);
    uint hi_hi = uint(hi >> 32);

    ulong hi_lo_eps = (ulong(hi_lo) << 32) - ulong(hi_lo);
    ulong s = lo + hi_lo_eps;
    bool c1 = s < lo;

    ulong r = s - ulong(hi_hi);
    bool b2 = s < ulong(hi_hi);

    r += c1 ? GL_EPS_SC : 0UL;
    r += b2 ? GL_P_SC : 0UL;

    return GlSc{r >= GL_P_SC ? r - GL_P_SC : r};
}

inline GlSc glsc_simd_reduce(GlSc val) {
    for (uint offset = 16; offset > 0; offset >>= 1) {
        GlSc other;
        other.v = simd_shuffle_down(val.v, offset);
        val = glsc_add(val, other);
    }
    return val;
}

kernel void sumcheck_reduce_goldilocks(
    device const ulong* evals       [[buffer(0)]],
    device ulong* evals_out         [[buffer(1)]],
    constant ulong* challenge       [[buffer(2)]],
    constant uint& half_n           [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid >= half_n) return;

    GlSc a = GlSc{evals[gid]};
    GlSc b = GlSc{evals[gid + half_n]};
    GlSc r = GlSc{challenge[0]};

    GlSc diff = glsc_sub(b, a);
    GlSc r_diff = glsc_mul(r, diff);
    evals_out[gid] = glsc_add(a, r_diff).v;
}

kernel void sumcheck_round_poly_goldilocks(
    device const ulong* evals       [[buffer(0)]],
    device ulong* partial_sums      [[buffer(1)]],
    constant uint& half_n           [[buffer(2)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint tg_size                    [[threads_per_threadgroup]],
    uint simd_lane                  [[thread_index_in_simdgroup]],
    uint simd_id                    [[simdgroup_index_in_threadgroup]]
) {
    threadgroup ulong shared_s0[8];
    threadgroup ulong shared_s1[8];

    GlSc local_s0 = glsc_zero();
    GlSc local_s1 = glsc_zero();

    uint global_idx = tgid * tg_size + tid;
    if (global_idx < half_n) {
        local_s0 = GlSc{evals[global_idx]};
        local_s1 = GlSc{evals[global_idx + half_n]};
    }

    local_s0 = glsc_simd_reduce(local_s0);
    local_s1 = glsc_simd_reduce(local_s1);

    if (simd_lane == 0) {
        shared_s0[simd_id] = local_s0.v;
        shared_s1[simd_id] = local_s1.v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simds = (tg_size + 31) / 32;
    if (simd_id == 0 && simd_lane < num_simds) {
        local_s0 = GlSc{shared_s0[simd_lane]};
        local_s1 = GlSc{shared_s1[simd_lane]};
    } else if (simd_id == 0) {
        local_s0 = glsc_zero();
        local_s1 = glsc_zero();
    }

    if (simd_id == 0) {
        local_s0 = glsc_simd_reduce(local_s0);
        local_s1 = glsc_simd_reduce(local_s1);
        if (simd_lane == 0) {
            partial_sums[tgid * 2] = local_s0.v;
            partial_sums[tgid * 2 + 1] = local_s1.v;
        }
    }
}
