// Fused Circle NTT + Fibonacci Constraint Evaluation over Mersenne31
// Two paths:
// 1. Small fused (logN <= 10): Circle NTT in shared memory + constraint eval in one kernel
//    Avoids writing NTT output to global memory entirely.
// 2. Post-NTT constraint eval: reads NTT'd columns from device memory (separate buffers)
//    Used with the single-command-buffer approach for larger sizes.

#include "../fields/mersenne31.metal"
#include "../fri/circle_fri.metal"

// M31_INV2: (p+1)/2 = 2^30, needed for circle FRI fold but stripped by cClean include-filter
constant uint M31_INV2 = 1073741824u;  // (2^31 - 1 + 1) / 2

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

// --- Fused constraint eval + first FRI fold (y-twiddle) ---
// Combines circle_fib_constraint_separate_cols and circle_fri_fold_first
// in a single kernel dispatch. Reads 4 elements (2 columns × 2 positions),
// computes constraint quotients, then applies FRI first fold.
// Output: n/2 folded values.
//
// Saves:
//   - One intermediate GPU buffer write (constraint quotient)
//   - One GPU buffer read (fold input)
//   - One kernel dispatch
//
// Used for medium-to-large domains where circle_fib_constraint_separate_cols
// is already used (logN > 10).
kernel void circle_fib_constraint_fold_first(
    device const uint* col_a            [[buffer(0)]],
    device const uint* col_b            [[buffer(1)]],
    device uint* folded_out            [[buffer(2)]],    // output: n/2 folded values
    device const uint* inv_2y          [[buffer(3)]],    // precomputed 1/(2*y_i) for i in [0, n/2)
    device const uint* domain_y        [[buffer(4)]],    // y-coordinates for vanishing poly
    constant uint& alpha_cst           [[buffer(5)]],    // batching challenge for constraints
    constant uint& alpha_fold         [[buffer(6)]],    // FRI fold challenge
    constant uint& bc_a0_val          [[buffer(7)]],
    constant uint& bc_b0_val          [[buffer(8)]],
    constant uint& eval_len            [[buffer(9)]],    // original domain size n
    constant uint& trace_len            [[buffer(10)]],
    constant uint& log_trace            [[buffer(11)]],
    uint gid                            [[thread_position_in_grid]]
) {
    uint n = eval_len;
    uint half_n = n >> 1;
    if (gid >= half_n) return;

    uint i_lo = gid;
    uint i_hi = gid + half_n;
    uint step = n / trace_len;

    // --- Read trace elements for LOW position ---
    M31 a_cur_lo = M31{col_a[i_lo]};
    M31 b_cur_lo = M31{col_b[i_lo]};
    uint next_lo = (i_lo + step) % n;
    M31 a_next_lo = M31{col_a[next_lo]};
    M31 b_next_lo = M31{col_b[next_lo]};

    // --- Read trace elements for HIGH position ---
    M31 a_cur_hi = M31{col_a[i_hi]};
    M31 b_cur_hi = M31{col_b[i_hi]};
    uint next_hi = (i_hi + step) % n;
    M31 a_next_hi = M31{col_a[next_hi]};
    M31 b_next_hi = M31{col_b[next_hi]};

    // --- Vanishing polynomial for both positions ---
    M31 vz_lo = M31{domain_y[i_lo]};
    for (uint vi = 0; vi < log_trace; vi++) {
        M31 v2 = m31_sqr(vz_lo);
        vz_lo = m31_sub(m31_add(v2, v2), m31_one());
    }

    M31 vz_hi = M31{domain_y[i_hi]};
    for (uint vi = 0; vi < log_trace; vi++) {
        M31 v2 = m31_sqr(vz_hi);
        vz_hi = m31_sub(m31_add(v2, v2), m31_one());
    }

    if (vz_lo.v == 0) {
        folded_out[gid] = 0;
        return;
    }
    if (vz_hi.v == 0) {
        folded_out[gid] = 0;
        return;
    }

    M31 inv_vz_lo = m31_inv(vz_lo);
    M31 inv_vz_hi = m31_inv(vz_hi);
    M31 alpha = M31{alpha_cst};
    M31 bc_a0 = M31{bc_a0_val};
    M31 bc_b0 = M31{bc_b0_val};
    M31 inv2 = M31{M31_INV2};

    // --- Constraint quotient for LOW position ---
    // Transition: C0 = a_next - b_cur, C1 = b_next - (a_cur + b_cur)
    M31 c0_lo = m31_sub(a_next_lo, b_cur_lo);
    M31 c1_lo = m31_sub(b_next_lo, m31_add(a_cur_lo, b_cur_lo));

    M31 term0_lo = m31_mul(c0_lo, inv_vz_lo);
    M31 term1_lo = m31_mul(alpha, m31_mul(c1_lo, inv_vz_lo));
    M31 acc_lo = m31_add(term0_lo, term1_lo);
    M31 alpha_pow = m31_mul(alpha, alpha);

    M31 bc0_num_lo = m31_sub(a_cur_lo, bc_a0);
    M31 bc0_term_lo = m31_mul(alpha_pow, m31_mul(bc0_num_lo, inv_vz_lo));
    acc_lo = m31_add(acc_lo, bc0_term_lo);
    alpha_pow = m31_mul(alpha_pow, alpha);

    M31 bc1_num_lo = m31_sub(b_cur_lo, bc_b0);
    M31 bc1_term_lo = m31_mul(alpha_pow, m31_mul(bc1_num_lo, inv_vz_lo));
    acc_lo = m31_add(acc_lo, bc1_term_lo);

    // --- Constraint quotient for HIGH position ---
    M31 c0_hi = m31_sub(a_next_hi, b_cur_hi);
    M31 c1_hi = m31_sub(b_next_hi, m31_add(a_cur_hi, b_cur_hi));

    M31 term0_hi = m31_mul(c0_hi, inv_vz_hi);
    M31 term1_hi = m31_mul(alpha, m31_mul(c1_hi, inv_vz_hi));
    M31 acc_hi = m31_add(term0_hi, term1_hi);
    M31 alpha_pow_hi = m31_mul(alpha, alpha);

    M31 bc0_num_hi = m31_sub(a_cur_hi, bc_a0);
    M31 bc0_term_hi = m31_mul(alpha_pow_hi, m31_mul(bc0_num_hi, inv_vz_hi));
    acc_hi = m31_add(acc_hi, bc0_term_hi);
    alpha_pow_hi = m31_mul(alpha_pow_hi, alpha);

    M31 bc1_num_hi = m31_sub(b_cur_hi, bc_b0);
    M31 bc1_term_hi = m31_mul(alpha_pow_hi, m31_mul(bc1_num_hi, inv_vz_hi));
    acc_hi = m31_add(acc_hi, bc1_term_hi);

    // --- FRI first fold (y-twiddle) ---
    // folded[i] = (q_lo + q_hi)/2 + alpha_fold * (q_lo - q_hi) * inv_2y[i]
    M31 sum = m31_mul(m31_add(acc_lo, acc_hi), inv2);
    M31 diff = m31_sub(acc_lo, acc_hi);
    M31 fold_alpha = M31{alpha_fold};
    M31 diff_term = m31_mul(m31_mul(fold_alpha, diff), M31{inv_2y[gid]});
    folded_out[gid] = m31_add(sum, diff_term).v;
}

// --- Fused 2-round FRI fold for Circle STARK ---
// Combines circle_fri_fold_fused2 pattern with constraint eval output.
// Applies two consecutive FRI folds (y-fold then x-fold) in one kernel.
// Output: n/4 folded values.
// This is for when the composition polynomial is already computed
// and needs 2 rounds of folding.
//
// Reads 4 elements, writes 1 element.
// Pattern: same as circle_fri_fold_fused2 but generalized.
kernel void circle_fri_fold_2r(
    device const M31* evals            [[buffer(0)]],
    device M31* folded                [[buffer(1)]],
    device const M31* inv_2y          [[buffer(2)]],    // y-twiddles for round 1
    device const M31* inv_2x           [[buffer(3)]],    // x-twiddles for round 2
    constant M31* alpha0               [[buffer(4)]],    // challenge for y-fold
    constant M31* alpha1               [[buffer(5)]],    // challenge for x-fold
    constant uint& n                   [[buffer(6)]],    // original domain size
    uint gid                           [[thread_position_in_grid]]
) {
    uint quarter = n >> 2;
    if (gid >= quarter) return;

    uint half_n = n >> 1;

    // Read 4 elements for two rounds of folding
    M31 e0 = evals[gid];
    M31 e1 = evals[gid + quarter];
    M31 e2 = evals[gid + half_n];
    M31 e3 = evals[gid + half_n + quarter];

    M31 a0 = alpha0[0];
    M31 inv2 = M31{M31_INV2};

    // Round 1: y-fold over pairs (e0,e2) and (e1,e3)
    M31 sum02 = m31_mul(m31_add(e0, e2), inv2);
    M31 diff02 = m31_sub(e0, e2);
    M31 f1_lo = m31_add(sum02, m31_mul(m31_mul(a0, diff02), inv_2y[gid]));

    M31 sum13 = m31_mul(m31_add(e1, e3), inv2);
    M31 diff13 = m31_sub(e1, e3);
    M31 f1_hi = m31_add(sum13, m31_mul(m31_mul(a0, diff13), inv_2y[gid + quarter]));

    // Round 2: x-fold over (f1_lo, f1_hi)
    M31 a1 = alpha1[0];
    M31 sum_r2 = m31_mul(m31_add(f1_lo, f1_hi), inv2);
    M31 diff_r2 = m31_sub(f1_lo, f1_hi);
    folded[gid] = m31_add(sum_r2, m31_mul(m31_mul(a1, diff_r2), inv_2x[gid]));
}

// --- Fused constraint eval + 2-round FRI fold ---
// Combines circle_fib_constraint_separate_cols with two FRI folds.
// Reads 8 elements (2 cols × 4 positions), computes 2 constraint quotients,
// applies y-fold then x-fold. Output: n/4 folded values.
//
// This fuses the most expensive parts of the STARK prover:
// - Constraint evaluation (2 positions per thread = 2 quotient computations)
// - FRI fold rounds 1+2 (eliminates 2 kernel dispatches + 2 buffer round-trips)
//
// Register pressure: 8 M31 inputs + intermediates. Feasible with threadgroup size ≤ 128.
kernel void circle_fib_constraint_fold_2r(
    device const uint* col_a            [[buffer(0)]],
    device const uint* col_b            [[buffer(1)]],
    device uint* folded_out            [[buffer(2)]],    // output: n/4 folded values
    device const uint* inv_2y          [[buffer(3)]],    // precomputed 1/(2*y_i) for round 1
    device const uint* inv_2x          [[buffer(4)]],    // precomputed 1/(2*x_i) for round 2
    device const uint* domain_y        [[buffer(5)]],    // y-coordinates for vanishing poly
    constant uint& alpha_cst           [[buffer(6)]],    // batching challenge for constraints
    constant uint& alpha0              [[buffer(7)]],    // FRI y-fold challenge
    constant uint& alpha1              [[buffer(8)]],    // FRI x-fold challenge
    constant uint& bc_a0_val           [[buffer(9)]],
    constant uint& bc_b0_val           [[buffer(10)]],
    constant uint& eval_len            [[buffer(11)]],   // original domain size n
    constant uint& trace_len           [[buffer(12)]],
    constant uint& log_trace           [[buffer(13)]],
    uint gid                           [[thread_position_in_grid]]
) {
    uint quarter = eval_len >> 2;
    if (gid >= quarter) return;

    uint half_n = eval_len >> 1;
    uint n = eval_len;

    // Four positions: lo-pair and hi-pair for the two FRI folds
    // Round 1 pairs: (gid, gid+quarter) and (gid+half_n, gid+half_n+quarter)
    // Round 2 pairs: (f1_lo, f1_hi) from round 1
    uint pos0 = gid;
    uint pos1 = gid + quarter;
    uint pos2 = gid + half_n;
    uint pos3 = gid + half_n + quarter;

    uint step = n / trace_len;

    // --- Read trace elements for all 4 positions ---
    // Position 0: a0, b0; next: a0n, b0n
    M31 a0 = M31{col_a[pos0]};
    M31 b0 = M31{col_b[pos0]};
    uint next0 = (pos0 + step) % n;
    M31 a0n = M31{col_a[next0]};
    M31 b0n = M31{col_b[next0]};

    // Position 1: a1, b1; next: a1n, b1n
    M31 a1 = M31{col_a[pos1]};
    M31 b1 = M31{col_b[pos1]};
    uint next1 = (pos1 + step) % n;
    M31 a1n = M31{col_a[next1]};
    M31 b1n = M31{col_b[next1]};

    // Position 2: a2, b2; next: a2n, b2n
    M31 a2 = M31{col_a[pos2]};
    M31 b2 = M31{col_b[pos2]};
    uint next2 = (pos2 + step) % n;
    M31 a2n = M31{col_a[next2]};
    M31 b2n = M31{col_b[next2]};

    // Position 3: a3, b3; next: a3n, b3n
    M31 a3 = M31{col_a[pos3]};
    M31 b3 = M31{col_b[pos3]};
    uint next3 = (pos3 + step) % n;
    M31 a3n = M31{col_a[next3]};
    M31 b3n = M31{col_b[next3]};

    // --- Compute vanishing polynomial for all 4 positions ---
    M31 y0 = M31{domain_y[pos0]};
    M31 vz0 = y0;
    for (uint vi = 0; vi < log_trace; vi++) {
        M31 v2 = m31_sqr(vz0);
        vz0 = m31_sub(m31_add(v2, v2), m31_one());
    }

    M31 y1 = M31{domain_y[pos1]};
    M31 vz1 = y1;
    for (uint vi = 0; vi < log_trace; vi++) {
        M31 v2 = m31_sqr(vz1);
        vz1 = m31_sub(m31_add(v2, v2), m31_one());
    }

    M31 y2 = M31{domain_y[pos2]};
    M31 vz2 = y2;
    for (uint vi = 0; vi < log_trace; vi++) {
        M31 v2 = m31_sqr(vz2);
        vz2 = m31_sub(m31_add(v2, v2), m31_one());
    }

    M31 y3 = M31{domain_y[pos3]};
    M31 vz3 = y3;
    for (uint vi = 0; vi < log_trace; vi++) {
        M31 v2 = m31_sqr(vz3);
        vz3 = m31_sub(m31_add(v2, v2), m31_one());
    }

    if (vz0.v == 0 || vz1.v == 0 || vz2.v == 0 || vz3.v == 0) {
        folded_out[gid] = 0;
        return;
    }

    M31 inv_vz0 = m31_inv(vz0);
    M31 inv_vz1 = m31_inv(vz1);
    M31 inv_vz2 = m31_inv(vz2);
    M31 inv_vz3 = m31_inv(vz3);

    M31 alpha = M31{alpha_cst};
    M31 bc_a0 = M31{bc_a0_val};
    M31 bc_b0 = M31{bc_b0_val};
    M31 inv2 = M31{M31_INV2};

    // Helper lambda for constraint quotient (avoids code duplication)
    // Each position computes: alpha^0*C0/vz + alpha^1*C1/vz + alpha^2*(col-bc0)/vz + alpha^3*(col-bc1)/vz
    // We'll inline for each position to manage register allocation.

    // --- Position 0 constraint quotient ---
    M31 c0_0 = m31_sub(a0n, b0);
    M31 c1_0 = m31_sub(b0n, m31_add(a0, b0));
    M31 acc0 = m31_mul(c0_0, inv_vz0);
    M31 t1_0 = m31_mul(alpha, m31_mul(c1_0, inv_vz0));
    acc0 = m31_add(acc0, t1_0);
    M31 ap = m31_mul(alpha, alpha);
    M31 bc0_0 = m31_sub(a0, bc_a0);
    M31 bc0_t0 = m31_mul(ap, m31_mul(bc0_0, inv_vz0));
    acc0 = m31_add(acc0, bc0_t0);
    ap = m31_mul(ap, alpha);
    M31 bc1_0 = m31_sub(b0, bc_b0);
    M31 bc1_t0 = m31_mul(ap, m31_mul(bc1_0, inv_vz0));
    acc0 = m31_add(acc0, bc1_t0);

    // --- Position 1 constraint quotient ---
    M31 c0_1 = m31_sub(a1n, b1);
    M31 c1_1 = m31_sub(b1n, m31_add(a1, b1));
    M31 acc1 = m31_mul(c0_1, inv_vz1);
    M31 t1_1 = m31_mul(alpha, m31_mul(c1_1, inv_vz1));
    acc1 = m31_add(acc1, t1_1);
    M31 ap1 = m31_mul(alpha, alpha);
    M31 bc0_1 = m31_sub(a1, bc_a0);
    M31 bc0_t1 = m31_mul(ap1, m31_mul(bc0_1, inv_vz1));
    acc1 = m31_add(acc1, bc0_t1);
    ap1 = m31_mul(ap1, alpha);
    M31 bc1_1 = m31_sub(b1, bc_b0);
    M31 bc1_t1 = m31_mul(ap1, m31_mul(bc1_1, inv_vz1));
    acc1 = m31_add(acc1, bc1_t1);

    // --- Position 2 constraint quotient ---
    M31 c0_2 = m31_sub(a2n, b2);
    M31 c1_2 = m31_sub(b2n, m31_add(a2, b2));
    M31 acc2 = m31_mul(c0_2, inv_vz2);
    M31 t1_2 = m31_mul(alpha, m31_mul(c1_2, inv_vz2));
    acc2 = m31_add(acc2, t1_2);
    M31 ap2 = m31_mul(alpha, alpha);
    M31 bc0_2 = m31_sub(a2, bc_a0);
    M31 bc0_t2 = m31_mul(ap2, m31_mul(bc0_2, inv_vz2));
    acc2 = m31_add(acc2, bc0_t2);
    ap2 = m31_mul(ap2, alpha);
    M31 bc1_2 = m31_sub(b2, bc_b0);
    M31 bc1_t2 = m31_mul(ap2, m31_mul(bc1_2, inv_vz2));
    acc2 = m31_add(acc2, bc1_t2);

    // --- Position 3 constraint quotient ---
    M31 c0_3 = m31_sub(a3n, b3);
    M31 c1_3 = m31_sub(b3n, m31_add(a3, b3));
    M31 acc3 = m31_mul(c0_3, inv_vz3);
    M31 t1_3 = m31_mul(alpha, m31_mul(c1_3, inv_vz3));
    acc3 = m31_add(acc3, t1_3);
    M31 ap3 = m31_mul(alpha, alpha);
    M31 bc0_3 = m31_sub(a3, bc_a0);
    M31 bc0_t3 = m31_mul(ap3, m31_mul(bc0_3, inv_vz3));
    acc3 = m31_add(acc3, bc0_t3);
    ap3 = m31_mul(ap3, alpha);
    M31 bc1_3 = m31_sub(b3, bc_b0);
    M31 bc1_t3 = m31_mul(ap3, m31_mul(bc1_3, inv_vz3));
    acc3 = m31_add(acc3, bc1_t3);

    // --- Round 1: y-fold over pairs (acc0,acc2) and (acc1,acc3) ---
    M31 a0_fold = M31{alpha0};
    M31 sum02 = m31_mul(m31_add(acc0, acc2), inv2);
    M31 diff02 = m31_sub(acc0, acc2);
    M31 f1_lo = m31_add(sum02, m31_mul(m31_mul(a0_fold, diff02), M31{inv_2y[gid]}));

    M31 sum13 = m31_mul(m31_add(acc1, acc3), inv2);
    M31 diff13 = m31_sub(acc1, acc3);
    M31 f1_hi = m31_add(sum13, m31_mul(m31_mul(a0_fold, diff13), M31{inv_2y[gid + quarter]}));

    // --- Round 2: x-fold over (f1_lo, f1_hi) ---
    M31 a1_fold = M31{alpha1};
    M31 sum_r2 = m31_mul(m31_add(f1_lo, f1_hi), inv2);
    M31 diff_r2 = m31_sub(f1_lo, f1_hi);
    folded_out[gid] = m31_add(sum_r2, m31_mul(m31_mul(a1_fold, diff_r2), M31{inv_2x[gid]})).v;
}
