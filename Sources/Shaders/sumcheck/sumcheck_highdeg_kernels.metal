// High-degree sumcheck kernels for Metal GPU
// Supports degree-d sumcheck where d = number of multilinear polynomials (up to 32).
//
// In standard sumcheck, we have k MLEs p_1,...,p_k and want to prove:
//   sum_{x in {0,1}^n} prod_{i=1}^k p_i(x) = claimed_sum
//
// Each round polynomial S_j(X) has degree k (one per MLE factor).
// We evaluate S_j at k+1 points: X = 0, 1, 2, ..., k.
//
// For each hypercube position i, each MLE p_j has values:
//   a_j = p_j[i]  (first half, x_current = 0)
//   b_j = p_j[i + halfN]  (second half, x_current = 1)
// Multilinear extension at point t: p_j(t) = a_j + t * (b_j - a_j)
// where t is in Montgomery form (precomputed on CPU and passed in eval_points buffer).

#include "../fields/bn254_fr.metal"

// SIMD-level Fr reduction using component-wise shuffle
inline Fr simd_reduce_fr(Fr val) {
    for (uint offset = 16; offset > 0; offset >>= 1) {
        Fr other;
        for (int k = 0; k < 8; k++) {
            other.v[k] = simd_shuffle_down(val.v[k], offset);
        }
        val = fr_add(val, other);
    }
    return val;
}

// Maximum supported degree (number of polynomials).
// Round polynomial has degree+1 evaluation points.
// MAX_DEGREE = 32 means up to 33 eval points.
#define HIGHDEG_MAX_POLYS 32
#define HIGHDEG_MAX_EVAL_POINTS 33  // MAX_POLYS + 1

// High-degree round polynomial + reduce kernel.
// Reads k polynomial evaluation tables (concatenated), computes partial sums
// for the round polynomial at degree+1 points, and reduces each poly by the challenge.
//
// Buffer layout:
//   polys[poly_idx * total_n + i] = evaluation of poly poly_idx at hypercube point i
//   The first half (i < halfN) is x_current=0, second half is x_current=1.
//
// Output:
//   polys_out[poly_idx * halfN + i] = reduced evaluation after fixing variable to challenge
//   partial_sums[tgid * num_eval_pts + t] = partial sum for eval point t
//
// Parameters:
//   half_n: n/2 where n is current domain size per polynomial
//   num_polys: k (number of multilinear polynomials, = degree of round poly)
//   num_eval_pts: k+1 (number of evaluation points)
//   eval_points: precomputed Montgomery-form values [0, 1, 2, ..., k] (num_eval_pts Fr elements)
//   challenge: the Fiat-Shamir challenge for reducing
kernel void sumcheck_highdeg_round_reduce(
    device const Fr* polys              [[buffer(0)]],    // k * total_n elements (total_n = 2*half_n)
    device Fr* polys_out                [[buffer(1)]],    // k * half_n elements (reduced)
    device Fr* partial_sums             [[buffer(2)]],    // num_groups * num_eval_pts entries
    constant Fr* eval_points            [[buffer(3)]],    // num_eval_pts Fr elements [0,1,...,degree]
    constant Fr* challenge              [[buffer(4)]],    // single Fr element
    constant uint& half_n               [[buffer(5)]],
    constant uint& num_polys            [[buffer(6)]],
    constant uint& num_eval_pts         [[buffer(7)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint tgid                           [[threadgroup_position_in_grid]],
    uint tg_size                        [[threads_per_threadgroup]],
    uint simd_lane                      [[thread_index_in_simdgroup]],
    uint simd_id                        [[simdgroup_index_in_threadgroup]]
) {
    uint total_n = half_n * 2;
    uint global_tid = tgid * tg_size + tid;

    // Local accumulators for each evaluation point (up to 33)
    Fr local_sums[HIGHDEG_MAX_EVAL_POINTS];
    for (uint t = 0; t < num_eval_pts; t++) {
        local_sums[t] = fr_zero();
    }

    Fr r = challenge[0];

    if (global_tid < half_n) {
        // Load a_j, b_j for each polynomial and compute diff
        Fr a_vals[HIGHDEG_MAX_POLYS];
        Fr diff_vals[HIGHDEG_MAX_POLYS];

        for (uint j = 0; j < num_polys; j++) {
            Fr a = polys[j * total_n + global_tid];
            Fr b = polys[j * total_n + global_tid + half_n];
            a_vals[j] = a;
            diff_vals[j] = fr_sub(b, a);

            // Reduce: poly_out[j][i] = a + r * (b - a)
            polys_out[j * half_n + global_tid] = fr_add(a, fr_mul(r, diff_vals[j]));
        }

        // For each evaluation point t, compute product of p_j(t) for all j
        for (uint t = 0; t < num_eval_pts; t++) {
            Fr eval_t = eval_points[t];
            Fr product = fr_one();
            for (uint j = 0; j < num_polys; j++) {
                // p_j(t) = a_j + t * diff_j
                Fr pj_t = fr_add(a_vals[j], fr_mul(eval_t, diff_vals[j]));
                product = fr_mul(product, pj_t);
            }
            local_sums[t] = product;
        }
    }

    // SIMD reduction for each eval point
    // We use shared memory with layout: shared_sums[eval_pt_idx][simd_group_idx]
    // Max 8 SIMD groups (256 threads / 32 per SIMD)
    threadgroup Fr shared_sums[HIGHDEG_MAX_EVAL_POINTS * 8];
    uint num_simds = tg_size / 32;

    for (uint t = 0; t < num_eval_pts; t++) {
        Fr val = local_sums[t];

        // SIMD-level reduction
        val = simd_reduce_fr(val);

        if (simd_lane == 0) {
            shared_sums[t * 8 + simd_id] = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Inter-SIMD reduction (first SIMD group only)
    for (uint t = 0; t < num_eval_pts; t++) {
        Fr val;
        if (tid < num_simds) {
            val = shared_sums[t * 8 + tid];
        } else {
            val = fr_zero();
        }
        if (simd_id == 0) {
            val = simd_reduce_fr(val);
        }
        if (tid == 0) {
            partial_sums[tgid * num_eval_pts + t] = val;
        }
    }
}

// Optimized high-degree kernel for small poly counts (2-8 polys).
// Unrolls the inner product loop for better register usage.
// Uses the same interface as sumcheck_highdeg_round_reduce.
kernel void sumcheck_highdeg_round_reduce_small(
    device const Fr* polys              [[buffer(0)]],
    device Fr* polys_out                [[buffer(1)]],
    device Fr* partial_sums             [[buffer(2)]],
    constant Fr* eval_points            [[buffer(3)]],
    constant Fr* challenge              [[buffer(4)]],
    constant uint& half_n               [[buffer(5)]],
    constant uint& num_polys            [[buffer(6)]],
    constant uint& num_eval_pts         [[buffer(7)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint tgid                           [[threadgroup_position_in_grid]],
    uint tg_size                        [[threads_per_threadgroup]],
    uint simd_lane                      [[thread_index_in_simdgroup]],
    uint simd_id                        [[simdgroup_index_in_threadgroup]]
) {
    uint total_n = half_n * 2;
    uint global_tid = tgid * tg_size + tid;

    // For small poly counts, use fixed-size arrays to help compiler
    Fr local_sums[9]; // max 8+1 = 9 eval points
    for (uint t = 0; t < num_eval_pts; t++) {
        local_sums[t] = fr_zero();
    }

    Fr r = challenge[0];

    if (global_tid < half_n) {
        Fr a_vals[8];
        Fr diff_vals[8];

        for (uint j = 0; j < num_polys; j++) {
            Fr a = polys[j * total_n + global_tid];
            Fr b = polys[j * total_n + global_tid + half_n];
            a_vals[j] = a;
            diff_vals[j] = fr_sub(b, a);
            polys_out[j * half_n + global_tid] = fr_add(a, fr_mul(r, diff_vals[j]));
        }

        for (uint t = 0; t < num_eval_pts; t++) {
            Fr eval_t = eval_points[t];
            Fr product = fr_one();
            for (uint j = 0; j < num_polys; j++) {
                Fr pj_t = fr_add(a_vals[j], fr_mul(eval_t, diff_vals[j]));
                product = fr_mul(product, pj_t);
            }
            local_sums[t] = product;
        }
    }

    threadgroup Fr shared_sums[9 * 8]; // 9 eval points * 8 SIMD groups
    uint num_simds = tg_size / 32;

    for (uint t = 0; t < num_eval_pts; t++) {
        Fr val = local_sums[t];
        val = simd_reduce_fr(val);
        if (simd_lane == 0) {
            shared_sums[t * 8 + simd_id] = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = 0; t < num_eval_pts; t++) {
        Fr val;
        if (tid < num_simds) {
            val = shared_sums[t * 8 + tid];
        } else {
            val = fr_zero();
        }
        if (simd_id == 0) {
            val = simd_reduce_fr(val);
        }
        if (tid == 0) {
            partial_sums[tgid * num_eval_pts + t] = val;
        }
    }
}
