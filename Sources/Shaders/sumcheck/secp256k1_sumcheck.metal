// GPU-accelerated Fused Sumcheck Round for secp256k1 Folding
//
// Folding schemes (Nova, HyperNova, Supernova) use sumcheck to prove:
//   sum_{x in {0,1}^s} eq(tau, x) * g(x) = claimed
//
// The fused kernel combines:
//   1. eq(tau, x) inner product computation
//   2. Weighting by fold challenges
//   3. Accumulation into fold accumulator
//   4. All in one GPU dispatch with shared memory
//
// This eliminates intermediate memory round-trips between eq-compute and fold.
//
// secp256k1 field: 8x32-bit Montgomery form (same layout as BN254 Fr)

#include "../fields/secp256k1_fp.metal"

// SIMD-level Fr reduction using component-wise shuffle
inline SecpFp sc_simd_reduce_secp(SecpFp val) {
    for (uint offset = 16; offset > 0; offset >>= 1) {
        SecpFp other;
        for (int k = 0; k < 8; k++) {
            other.v[k] = simd_shuffle_down(val.v[k], offset);
        }
        val = secp_add(val, other);
    }
    return val;
}

// ============================================================================
// Fused Sumcheck Round for secp256k1 Folding
// ============================================================================
//
// This kernel fuses the eq-computation with the fold operation.
//
// In folding, we need to compute:
//   accumulator = sum_i eq(tau, x_i) * g(x_i)
//
// The fold challenge r is applied as:
//   accumulator' = accumulator + r * cross_term
//
// The fused kernel reads each element once and:
//   1. Computes eq(tau, i) contribution
//   2. Multiplies by g(i)
//   3. Applies fold weight
//   4. Accumulates into fold result
//
// Buffer layout:
//   buffer(0): input evals (g(x) values) - size n
//   buffer(1): output accumulator - single element
//   buffer(2): fold challenge r - single Fr element
//   buffer(3): tau point (eq evaluation point) - s elements
//   buffer(4): n - domain size
//   buffer(5): s - number of variables (log2 n)
//
// Each thread processes one hypercube index, computes eq(tau, x) * g(x),
// and contributes to the shared accumulation.

// Fused eq-compute + fold accumulate kernel for secp256k1
// Reads evals once, computes eq(tau, x) * eval[x], accumulates with fold weight.
// Single GPU dispatch replaces separate eq-compute and fold steps.
kernel void secp_fused_sumcheck_fold(
    device const SecpFp* evals         [[buffer(0)]],    // g(x) evaluations - size n
    device SecpFp* accumulator         [[buffer(1)]],    // output: accumulated result
    constant SecpFp* challenge         [[buffer(2)]],    // fold challenge r
    constant SecpFp* tau                [[buffer(3)]],    // eq evaluation point - s elements
    constant uint& n                   [[buffer(4)]],    // domain size
    constant uint& s                   [[buffer(5)]],    // number of variables (log2 n)
    uint tid                           [[thread_index_in_threadgroup]],
    uint tgid                         [[threadgroup_position_in_grid]],
    uint tg_size                      [[threads_per_threadgroup]],
    uint simd_lane                    [[thread_index_in_simdgroup]],
    uint simd_id                      [[simdgroup_index_in_threadgroup]]
) {
    uint num_simds = tg_size / 32;
    uint global_tid = tgid * tg_size + tid;
    uint total_threads = tgid * tg_size + tg_size;

    // Threadgroup-level accumulation (max 8 SIMD groups for 256 threads)
    threadgroup SecpFp shared_acc[8];

    // Initialize accumulator
    SecpFp local_acc = secp_zero();
    SecpFp r = challenge[0];

    // Process this thread's share of the domain
    // Each thread processes multiple indices in strided fashion
    for (uint idx = global_tid; idx < n; idx += total_threads) {
        // Compute eq(tau, idx) for this hypercube position
        // eq(tau, x) = prod_i (tau_i * x_i + (1-tau_i) * (1-x_i))
        // where x_i is bit i of idx

        SecpFp eq_val = secp_one();
        uint temp = idx;
        for (uint i = 0; i < s; i++) {
            uint bit = temp & 1;
            SecpFp ti = tau[i];
            SecpFp one_minus_ti = secp_sub(secp_one(), ti);

            // eq_i = bit * ti + (1-bit) * (1-ti)
            // If bit=0: eq_i = 1 - ti
            // If bit=1: eq_i = ti
            SecpFp eq_i;
            if (bit == 0) {
                eq_i = one_minus_ti;
            } else {
                eq_i = ti;
            }
            eq_val = secp_mul(eq_val, eq_i);
            temp >>= 1;
        }

        // contribution = eq_val * eval[idx]
        SecpFp eval_i = evals[idx];
        SecpFp contribution = secp_mul(eq_val, eval_i);

        // Accumulate with fold weight: acc = acc + r * contribution
        // (or just acc += contribution for pure sumcheck)
        SecpFp weighted = secp_mul(r, contribution);
        local_acc = secp_add(local_acc, weighted);
    }

    // SIMD-level reduction
    local_acc = sc_simd_reduce_secp(local_acc);

    // Write to shared memory
    if (simd_lane == 0) {
        shared_acc[simd_id] = local_acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Inter-SIMD reduction (first SIMD group only)
    if (simd_id == 0 && tid < num_simds) {
        local_acc = shared_acc[tid];
    } else if (simd_id == 0) {
        local_acc = secp_zero();
    }

    if (simd_id == 0) {
        local_acc = sc_simd_reduce_secp(local_acc);
    }

    // Thread 0 writes final result
    if (tid == 0) {
        accumulator[0] = local_acc;
    }
}

// ============================================================================
// Fused Eq + Fold with Sequential Variables (for larger domains)
// ============================================================================
//
// For domains larger than fits in one threadgroup, we process variables
// sequentially. This kernel processes the first K variables in registers,
// then reduces over the remaining hypercube dimensions.
//
// Buffer layout:
//   buffer(0): input evals - size n
//   buffer(1): output accumulator - single element
//   buffer(2): fold challenge r
//   buffer(3): tau point
//   buffer(4): n (total domain size)
//   buffer(5): s (total variables)
//   buffer(6): k (variables to process in register)
//
// This kernel processes k variables per element in registers,
// reducing memory access pattern and improving cache utilization.

// Process variables in register, reduce remaining dimensions
// k variables are processed per thread, rest are reduced across threads
kernel void secp_fused_sumcheck_fold_reg(
    device const SecpFp* evals         [[buffer(0)]],
    device SecpFp* accumulator         [[buffer(1)]],
    constant SecpFp* challenge         [[buffer(2)]],
    constant SecpFp* tau               [[buffer(3)]],
    constant uint& n                   [[buffer(4)]],
    constant uint& s                   [[buffer(5)]],
    constant uint& k                   [[buffer(6)]],    // vars per thread
    uint tid                           [[thread_index_in_threadgroup]],
    uint tgid                         [[threadgroup_position_in_grid]],
    uint tg_size                      [[threads_per_threadgroup]],
    uint simd_lane                    [[thread_index_in_simdgroup]],
    uint simd_id                      [[simdgroup_index_in_threadgroup]]
) {
    uint num_simds = tg_size / 32;
    threadgroup SecpFp shared_acc[8];

    SecpFp local_acc = secp_zero();
    SecpFp r = challenge[0];

    // Each thread handles one base index
    uint base_idx = tgid * tg_size + tid;
    uint block_size = tg_size * (n / (tg_size * tg_size));

    if (base_idx >= n) {
        local_acc = secp_zero();
    } else {
        // Process k variables in register
        // idx = base + offset where offset ranges over remaining dimensions
        uint max_offset = 1 << (s - k);  // remaining dimensions
        uint base_partial = base_idx & ((1 << k) - 1);  // k LSB bits

        for (uint offset = 0; offset < max_offset; offset++) {
            uint idx = base_partial + (offset << k);
            if (idx >= n) continue;

            // Compute eq over k register variables
            SecpFp eq_val = secp_one();
            uint temp = idx;
            for (uint i = 0; i < k; i++) {
                uint bit = temp & 1;
                SecpFp ti = tau[i];
                SecpFp one_minus_ti = secp_sub(secp_one(), ti);
                SecpFp eq_i = (bit == 0) ? one_minus_ti : ti;
                eq_val = secp_mul(eq_val, eq_i);
                temp >>= 1;
            }

            // Multiply by eval
            SecpFp contribution = secp_mul(eq_val, evals[idx]);
            SecpFp weighted = secp_mul(r, contribution);
            local_acc = secp_add(local_acc, weighted);
        }
    }

    // SIMD reduction
    local_acc = sc_simd_reduce_secp(local_acc);

    if (simd_lane == 0) {
        shared_acc[simd_id] = local_acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0 && tid < num_simds) {
        local_acc = shared_acc[tid];
    } else if (simd_id == 0) {
        local_acc = secp_zero();
    }

    if (simd_id == 0) {
        local_acc = sc_simd_reduce_secp(local_acc);
    }

    if (tid == 0) {
        accumulator[0] = local_acc;
    }
}

// ============================================================================
// Fused Sumcheck Round for Multi-Table Folding (CCS-style)
// ============================================================================
//
// In CCS/HyperNova, we have multiple tables (polynomials) that are folded together.
// This kernel fuses the multi-table eq computation with the fold.
//
// Buffer layout:
//   buffer(0): concatenated table evaluations [t * n elements]
//   buffer(1): output accumulator
//   buffer(2): fold challenge r
//   buffer(3): tau point [s elements]
//   buffer(4): n (domain size per table)
//   buffer(5): s (variables)
//   buffer(6): t (number of tables)

// Fused multi-table sumcheck fold for CCS-style folding
// Each thread processes one hypercube index across all t tables
kernel void secp_fused_sumcheck_multitable_fold(
    device const SecpFp* tables        [[buffer(0)]],    // t * n elements
    device SecpFp* accumulator         [[buffer(1)]],
    constant SecpFp* challenge         [[buffer(2)]],
    constant SecpFp* tau               [[buffer(3)]],
    constant uint& n                   [[buffer(4)]],
    constant uint& s                   [[buffer(5)]],
    constant uint& t                   [[buffer(6)]],    // number of tables
    uint tid                           [[thread_index_in_threadgroup]],
    uint tgid                         [[threadgroup_position_in_grid]],
    uint tg_size                      [[threads_per_threadgroup]],
    uint simd_lane                    [[thread_index_in_simdgroup]],
    uint simd_id                      [[simdgroup_index_in_threadgroup]]
) {
    uint num_simds = tg_size / 32;
    threadgroup SecpFp shared_acc[8];

    SecpFp local_acc = secp_zero();
    SecpFp r = challenge[0];

    uint global_tid = tgid * tg_size + tid;
    uint total_threads = tgid * tg_size + tg_size;

    for (uint idx = global_tid; idx < n; idx += total_threads) {
        // Compute eq(tau, idx)
        SecpFp eq_val = secp_one();
        uint temp = idx;
        for (uint i = 0; i < s; i++) {
            uint bit = temp & 1;
            SecpFp ti = tau[i];
            SecpFp one_minus_ti = secp_sub(secp_one(), ti);
            SecpFp eq_i = (bit == 0) ? one_minus_ti : ti;
            eq_val = secp_mul(eq_val, eq_i);
            temp >>= 1;
        }

        // Accumulate contributions from all t tables
        // accumulator += r * sum_t(eq * table_t[idx])
        SecpFp table_sum = secp_zero();
        for (uint table_idx = 0; table_idx < t; table_idx++) {
            uint flat_idx = table_idx * n + idx;
            SecpFp contribution = secp_mul(eq_val, tables[flat_idx]);
            table_sum = secp_add(table_sum, contribution);
        }

        SecpFp weighted = secp_mul(r, table_sum);
        local_acc = secp_add(local_acc, weighted);
    }

    // SIMD reduction
    local_acc = sc_simd_reduce_secp(local_acc);

    if (simd_lane == 0) {
        shared_acc[simd_id] = local_acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0 && tid < num_simds) {
        local_acc = shared_acc[tid];
    } else if (simd_id == 0) {
        local_acc = secp_zero();
    }

    if (simd_id == 0) {
        local_acc = sc_simd_reduce_secp(local_acc);
    }

    if (tid == 0) {
        accumulator[0] = local_acc;
    }
}

// ============================================================================
// Fused Round Polynomial + Reduce (standard sumcheck pattern)
// ============================================================================
//
// This kernel follows the same pattern as sumcheck_fused_round_reduce_bn254
// but for secp256k1. It computes:
//   1. Round polynomial partial sums (s0, s1, s2)
//   2. Reduces the table by the challenge
// Both in one GPU dispatch.
//
// Buffer layout:
//   buffer(0): input evaluations - size 2*n
//   buffer(1): output reduced evaluations - size n
//   buffer(2): partial sums output [num_groups * 3]
//   buffer(3): fold challenge r
//   buffer(4): n (half the domain size)

// Fused round poly + reduce for secp256k1
// Computes partial sums AND writes folded output in one pass
kernel void secp_fused_round_reduce(
    device const SecpFp* evals          [[buffer(0)]],
    device SecpFp* evals_out           [[buffer(1)]],
    device SecpFp* partial_sums         [[buffer(2)]],
    constant SecpFp* challenge         [[buffer(3)]],
    constant uint& n                   [[buffer(4)]],
    uint tid                          [[thread_index_in_threadgroup]],
    uint tgid                         [[threadgroup_position_in_grid]],
    uint tg_size                      [[threads_per_threadgroup]],
    uint simd_lane                    [[thread_index_in_simdgroup]],
    uint simd_id                      [[simdgroup_index_in_threadgroup]]
) {
    threadgroup SecpFp shared_s0[8];
    threadgroup SecpFp shared_s1[8];
    threadgroup SecpFp shared_s2[8];

    SecpFp local_s0 = secp_zero();
    SecpFp local_s1 = secp_zero();
    SecpFp local_s2 = secp_zero();
    SecpFp r = challenge[0];

    uint global_idx = tgid * tg_size + tid;

    if (global_idx < n) {
        SecpFp a = evals[global_idx];
        SecpFp b = evals[global_idx + n];
        local_s0 = a;
        local_s1 = b;

        // f(2) = 2*f(1) - f(0) (linear extrapolation)
        SecpFp two_b = secp_double(b);
        local_s2 = secp_sub(two_b, a);

        // Reduce: write folded output
        SecpFp diff = secp_sub(b, a);
        SecpFp r_diff = secp_mul(r, diff);
        evals_out[global_idx] = secp_add(a, r_diff);
    }

    // SIMD reduction for all three sums
    local_s0 = sc_simd_reduce_secp(local_s0);
    local_s1 = sc_simd_reduce_secp(local_s1);
    local_s2 = sc_simd_reduce_secp(local_s2);

    if (simd_lane == 0) {
        shared_s0[simd_id] = local_s0;
        shared_s1[simd_id] = local_s1;
        shared_s2[simd_id] = local_s2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simds = tg_size / 32;
    if (simd_id == 0 && tid < num_simds) {
        local_s0 = shared_s0[tid];
        local_s1 = shared_s1[tid];
        local_s2 = shared_s2[tid];
    } else if (simd_id == 0) {
        local_s0 = secp_zero();
        local_s1 = secp_zero();
        local_s2 = secp_zero();
    }

    if (simd_id == 0) {
        local_s0 = sc_simd_reduce_secp(local_s0);
        local_s1 = sc_simd_reduce_secp(local_s1);
        local_s2 = sc_simd_reduce_secp(local_s2);
        if (tid == 0) {
            partial_sums[tgid * 3] = local_s0;
            partial_sums[tgid * 3 + 1] = local_s1;
            partial_sums[tgid * 3 + 2] = local_s2;
        }
    }
}

// ============================================================================
// Final Reduction for secp256k1 Sumcheck
// ============================================================================
//
// Reduces num_groups partial sums (each with 3 elements: s0, s1, s2)
// to final 3 values.

kernel void secp_sumcheck_final_reduce(
    device const SecpFp* partial_sums  [[buffer(0)]],    // 3 * num_groups entries
    device SecpFp* output              [[buffer(1)]],    // 3 entries
    constant uint& num_groups          [[buffer(2)]],
    uint tid                          [[thread_index_in_threadgroup]],
    uint tg_size                      [[threads_per_threadgroup]]
) {
    threadgroup SecpFp shared_s0[256];
    threadgroup SecpFp shared_s1[256];
    threadgroup SecpFp shared_s2[256];

    SecpFp local_s0 = secp_zero();
    SecpFp local_s1 = secp_zero();
    SecpFp local_s2 = secp_zero();

    // Accumulate over strided range
    for (uint g = tid; g < num_groups; g += tg_size) {
        local_s0 = secp_add(local_s0, partial_sums[g * 3]);
        local_s1 = secp_add(local_s1, partial_sums[g * 3 + 1]);
        local_s2 = secp_add(local_s2, partial_sums[g * 3 + 2]);
    }

    shared_s0[tid] = local_s0;
    shared_s1[tid] = local_s1;
    shared_s2[tid] = local_s2;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint s = tg_size >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            shared_s0[tid] = secp_add(shared_s0[tid], shared_s0[tid + s]);
            shared_s1[tid] = secp_add(shared_s1[tid], shared_s1[tid + s]);
            shared_s2[tid] = secp_add(shared_s2[tid], shared_s2[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[0] = shared_s0[0];
        output[1] = shared_s1[0];
        output[2] = shared_s2[0];
    }
}
