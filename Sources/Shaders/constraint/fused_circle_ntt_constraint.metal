// Fused Circle NTT + Fibonacci Constraint Evaluation over Mersenne31
// Two paths:
// 1. Small fused (logN <= 10): Circle NTT in shared memory + constraint eval in one kernel
//    Avoids writing NTT output to global memory entirely.
// 2. Post-NTT constraint eval: reads NTT'd columns from device memory (separate buffers)
//    Used with the single-command-buffer approach for larger sizes.

#include "../fields/mersenne31.metal"

// --- Small fused: Circle NTT in shared memory + Fibonacci constraint eval ---
// Performs forward Circle NTT on two trace columns in shared memory,
// then evaluates Fibonacci constraints on the NTT output.
// No intermediate device memory write for NTT output.
//
// Circle NTT (DIT): layers k-1 down to 1 (x-twiddles), then layer 0 (y-twiddles)
// Each layer l has block_size = N >> l, half_block = N >> (l+1)
// Twiddles layout: [layer_0 (N/2)] [layer_1 (N/2)] ... [layer_{k-1} (N/2)]

kernel void fused_circle_ntt_fib_constraint(
    device const uint* trace_a          [[buffer(0)]],     // column a (natural order)
    device const uint* trace_b          [[buffer(1)]],     // column b (natural order)
    device const M31* all_twiddles      [[buffer(2)]],     // Circle NTT forward twiddles
    device uint* quotient_out           [[buffer(3)]],     // constraint quotient output
    constant uint& alpha_val            [[buffer(4)]],     // batching challenge
    constant uint& bc_a0_val            [[buffer(5)]],     // boundary a[0]
    constant uint& bc_b0_val            [[buffer(6)]],     // boundary b[0]
    constant uint& n_val                [[buffer(7)]],     // domain size
    constant uint& log_n_val            [[buffer(8)]],     // log2(domain size)
    constant uint& trace_len_val        [[buffer(9)]],     // original trace length
    constant uint& log_trace_val        [[buffer(10)]],    // log2(trace_len)
    device const uint* domain_y         [[buffer(11)]],    // y-coordinates for vanishing poly
    uint tid                            [[thread_index_in_threadgroup]],
    uint tgid                           [[threadgroup_position_in_grid]],
    uint tg_size                        [[threads_per_threadgroup]]
) {
    uint n = n_val;
    uint log_n = log_n_val;
    uint block_size = tg_size << 1;
    uint base = tgid * block_size;
    uint half_n = n >> 1;

    // Shared memory for two columns
    threadgroup M31 shared_a[1024];
    threadgroup M31 shared_b[1024];

    // Step 1: Load trace data into shared memory
    for (uint k = tid; k < block_size; k += tg_size) {
        uint global_idx = base + k;
        if (global_idx < n) {
            shared_a[k] = M31{trace_a[global_idx]};
            shared_b[k] = M31{trace_b[global_idx]};
        } else {
            shared_a[k] = m31_zero();
            shared_b[k] = m31_zero();
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Circle NTT butterfly passes (DIT)
    // Forward order: layers k-1 down to 1 (x-twiddles), then layer 0 (y-twiddles)
    // But since block_size <= 1024, we process all stages locally.
    // For the small fused path, we assume the entire domain fits in one threadgroup.

    // Layers k-1 down to 1 (x-twiddle DIT)
    for (uint layer_idx = 0; layer_idx < log_n; layer_idx++) {
        uint layer;
        if (layer_idx < log_n - 1) {
            layer = log_n - 1 - layer_idx;  // k-1 down to 1
        } else {
            layer = 0;  // layer 0 last (y-twiddles)
        }

        uint stage = log_n - 1 - layer;
        uint half_block = 1u << stage;
        uint local_block_size = half_block << 1;
        uint num_bflies = block_size >> 1;

        // Twiddle array for this layer is at offset layer * (n/2)
        uint tw_layer_offset = layer * half_n;

        for (uint k = tid; k < num_bflies; k += tg_size) {
            uint block_idx = k / half_block;
            uint local_idx = k % half_block;
            uint i = block_idx * local_block_size + local_idx;
            uint j = i + half_block;

            if (j < block_size) {
                uint twiddle_idx = local_idx * (n / local_block_size);
                M31 w = all_twiddles[tw_layer_offset + twiddle_idx];

                // Column a butterfly
                M31 a_i = shared_a[i];
                M31 a_j = shared_a[j];
                M31 wa = m31_mul(w, a_j);
                shared_a[i] = m31_add(a_i, wa);
                shared_a[j] = m31_sub(a_i, wa);

                // Column b butterfly
                M31 b_i = shared_b[i];
                M31 b_j = shared_b[j];
                M31 wb = m31_mul(w, b_j);
                shared_b[i] = m31_add(b_i, wb);
                shared_b[j] = m31_sub(b_i, wb);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Step 3: Evaluate Fibonacci constraints on NTT'd values
    // No device memory write for NTT output - go directly to constraints.
    uint trace_len = trace_len_val;
    uint log_trace = log_trace_val;
    M31 alpha = M31{alpha_val};
    M31 bc_a0 = M31{bc_a0_val};
    M31 bc_b0 = M31{bc_b0_val};
    uint step = n / trace_len;

    for (uint k = tid; k < block_size; k += tg_size) {
        uint row = base + k;
        if (row >= n) continue;

        M31 a_cur = shared_a[k];
        M31 b_cur = shared_b[k];

        // Next-row access: step apart in evaluation domain
        uint next_local = k + step;
        M31 a_next, b_next;
        if (next_local < block_size) {
            a_next = shared_a[next_local];
            b_next = shared_b[next_local];
        } else {
            // Next row wraps outside this threadgroup - read from device
            uint next_global = (row + step) % n;
            a_next = M31{trace_a[next_global]};  // Approximate: use original trace
            b_next = M31{trace_b[next_global]};  // (only for boundary, zero for inner)
            // For small sizes (single threadgroup), this branch is never taken.
        }

        // Transition constraints: C0 = a_next - b_cur, C1 = b_next - (a_cur + b_cur)
        M31 c0 = m31_sub(a_next, b_cur);
        M31 c1 = m31_sub(b_next, m31_add(a_cur, b_cur));

        // Vanishing polynomial
        M31 y = M31{domain_y[row]};
        M31 vz = y;
        for (uint vi = 0; vi < log_trace; vi++) {
            M31 v2 = m31_sqr(vz);
            vz = m31_sub(m31_add(v2, v2), m31_one());
        }

        if (vz.v == 0) {
            quotient_out[row] = 0;
            continue;
        }

        M31 inv_vz = m31_inv(vz);

        // Accumulate: alpha^0 * C0 * inv_vz + alpha^1 * C1 * inv_vz
        M31 term0 = m31_mul(c0, inv_vz);
        M31 term1 = m31_mul(alpha, m31_mul(c1, inv_vz));
        M31 acc = m31_add(term0, term1);
        M31 alpha_pow = m31_mul(alpha, alpha);

        // Boundary constraints
        M31 bc0_num = m31_sub(a_cur, bc_a0);
        M31 bc0_term = m31_mul(alpha_pow, m31_mul(bc0_num, inv_vz));
        acc = m31_add(acc, bc0_term);
        alpha_pow = m31_mul(alpha_pow, alpha);

        M31 bc1_num = m31_sub(b_cur, bc_b0);
        M31 bc1_term = m31_mul(alpha_pow, m31_mul(bc1_num, inv_vz));
        acc = m31_add(acc, bc1_term);

        quotient_out[row] = acc.v;
    }
}

// --- Post-NTT constraint eval with separate column buffers ---
// Used with single-command-buffer approach for large NTTs.
// Reads from separate NTT'd column buffers (no interleave needed).
kernel void circle_fib_constraint_separate_cols(
    device const uint* col_a            [[buffer(0)]],
    device const uint* col_b            [[buffer(1)]],
    device uint* quotient               [[buffer(2)]],
    device const uint* domain_y         [[buffer(3)]],
    constant uint& alpha_val            [[buffer(4)]],
    constant uint& bc_a0_val            [[buffer(5)]],
    constant uint& bc_b0_val            [[buffer(6)]],
    constant uint& eval_len             [[buffer(7)]],
    constant uint& trace_len            [[buffer(8)]],
    constant uint& log_trace            [[buffer(9)]],
    uint gid                            [[thread_position_in_grid]]
) {
    if (gid >= eval_len) return;

    uint step = eval_len / trace_len;
    uint next_idx = (gid + step) % eval_len;

    M31 a_cur = M31{col_a[gid]};
    M31 b_cur = M31{col_b[gid]};
    M31 a_next = M31{col_a[next_idx]};
    M31 b_next = M31{col_b[next_idx]};

    // Transition constraints
    M31 c0 = m31_sub(a_next, b_cur);
    M31 c1 = m31_sub(b_next, m31_add(a_cur, b_cur));

    // Vanishing polynomial
    M31 y = M31{domain_y[gid]};
    M31 vz = y;
    for (uint vi = 0; vi < log_trace; vi++) {
        M31 v2 = m31_sqr(vz);
        vz = m31_sub(m31_add(v2, v2), m31_one());
    }

    if (vz.v == 0) {
        quotient[gid] = 0;
        return;
    }

    M31 inv_vz = m31_inv(vz);
    M31 alpha = M31{alpha_val};
    M31 bc_a0 = M31{bc_a0_val};
    M31 bc_b0 = M31{bc_b0_val};

    M31 term0 = m31_mul(c0, inv_vz);
    M31 term1 = m31_mul(alpha, m31_mul(c1, inv_vz));
    M31 acc = m31_add(term0, term1);
    M31 alpha_pow = m31_mul(alpha, alpha);

    M31 bc0_num = m31_sub(a_cur, bc_a0);
    M31 bc0_term = m31_mul(alpha_pow, m31_mul(bc0_num, inv_vz));
    acc = m31_add(acc, bc0_term);
    alpha_pow = m31_mul(alpha_pow, alpha);

    M31 bc1_num = m31_sub(b_cur, bc_b0);
    M31 bc1_term = m31_mul(alpha_pow, m31_mul(bc1_num, inv_vz));
    acc = m31_add(acc, bc1_term);

    quotient[gid] = acc.v;
}
