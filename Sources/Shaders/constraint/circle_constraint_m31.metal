// Circle STARK constraint evaluation over Mersenne31
// Evaluates Fibonacci AIR constraints on the evaluation domain:
//   C0: a_next - b_current = 0
//   C1: b_next - (a_current + b_current) = 0
// Then computes quotient: sum_i alpha^i * C_i * inv_vanishing[i]
// Plus boundary constraints.

#include "../fields/mersenne31.metal"

// Batch inversion on GPU using Montgomery's trick would be complex.
// Instead, each thread computes its own inverse via Fermat's little theorem.
// For M31, inv(a) = a^(p-2) requires ~30 multiplications — fast on GPU.

// Precompute vanishing polynomial: v_0 = y, v_{k+1} = 2*v_k^2 - 1
M31 circle_vanishing_gpu(M31 y, uint log_domain_size) {
    M31 v = y;
    for (uint i = 0; i < log_domain_size; i++) {
        // v = 2*v^2 - 1
        M31 v2 = m31_sqr(v);
        v = m31_sub(m31_add(v2, v2), m31_one());
    }
    return v;
}

// Fibonacci constraint evaluation kernel
// trace_a, trace_b: LDE evaluations of trace columns (evalLen elements each)
// domain_y: y-coordinates of evaluation domain points (evalLen elements)
// output: composition polynomial evaluations (evalLen elements)
// alpha: random challenge for constraint batching
// bc_a0, bc_b0: boundary constraint values (a[0]=a0, b[0]=b0)
// eval_len: total number of evaluation points
// trace_len: original trace length
// log_trace: log2(trace_len)
kernel void circle_fibonacci_constraint_eval(
    device const uint* trace_a       [[buffer(0)]],
    device const uint* trace_b       [[buffer(1)]],
    device const uint* domain_y      [[buffer(2)]],
    device uint* output              [[buffer(3)]],
    constant uint& alpha_val         [[buffer(4)]],
    constant uint& bc_a0_val         [[buffer(5)]],
    constant uint& bc_b0_val         [[buffer(6)]],
    constant uint& eval_len          [[buffer(7)]],
    constant uint& trace_len         [[buffer(8)]],
    constant uint& log_trace         [[buffer(9)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= eval_len) return;

    uint step = eval_len / trace_len;
    uint next_idx = (tid + step) % eval_len;

    M31 a_cur = M31{trace_a[tid]};
    M31 b_cur = M31{trace_b[tid]};
    M31 a_next = M31{trace_a[next_idx]};
    M31 b_next = M31{trace_b[next_idx]};

    // Transition constraints
    M31 c0 = m31_sub(a_next, b_cur);                           // a' - b
    M31 c1 = m31_sub(b_next, m31_add(a_cur, b_cur));           // b' - (a + b)

    // Vanishing polynomial
    M31 y = M31{domain_y[tid]};
    M31 vz = circle_vanishing_gpu(y, log_trace);

    if (vz.v == 0) {
        output[tid] = 0;
        return;
    }

    M31 inv_vz = m31_inv(vz);
    M31 alpha = M31{alpha_val};
    M31 bc_a0 = M31{bc_a0_val};
    M31 bc_b0 = M31{bc_b0_val};

    // acc = alpha^0 * C0 * inv_vz + alpha^1 * C1 * inv_vz
    M31 term0 = m31_mul(c0, inv_vz);
    M31 term1 = m31_mul(alpha, m31_mul(c1, inv_vz));
    M31 acc = m31_add(term0, term1);
    M31 alpha_pow = m31_mul(alpha, alpha);  // alpha^2

    // Boundary constraints: (trace[col][i] - value) * inv_vz
    M31 bc0_num = m31_sub(a_cur, bc_a0);
    M31 bc0_term = m31_mul(alpha_pow, m31_mul(bc0_num, inv_vz));
    acc = m31_add(acc, bc0_term);
    alpha_pow = m31_mul(alpha_pow, alpha);

    M31 bc1_num = m31_sub(b_cur, bc_b0);
    M31 bc1_term = m31_mul(alpha_pow, m31_mul(bc1_num, inv_vz));
    acc = m31_add(acc, bc1_term);

    output[tid] = acc.v;
}

// Generic 2-column constraint evaluation (same as Fibonacci but parameterized)
// Can be reused for any AIR with 2 columns and linear constraints
kernel void circle_generic_2col_constraint_eval(
    device const uint* trace_cols    [[buffer(0)]],   // interleaved: [a0,b0,a1,b1,...] or column-major
    device const uint* domain_y      [[buffer(1)]],
    device uint* output              [[buffer(2)]],
    device const uint* constraint_coeffs [[buffer(3)]],  // packed constraint coefficients
    constant uint& alpha_val         [[buffer(4)]],
    constant uint& eval_len          [[buffer(5)]],
    constant uint& trace_len         [[buffer(6)]],
    constant uint& log_trace         [[buffer(7)]],
    constant uint& num_constraints   [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= eval_len) return;

    M31 y = M31{domain_y[tid]};
    M31 vz = circle_vanishing_gpu(y, log_trace);

    if (vz.v == 0) {
        output[tid] = 0;
        return;
    }

    M31 inv_vz = m31_inv(vz);
    M31 alpha = M31{alpha_val};

    // Column-major layout: trace_cols[col * eval_len + tid]
    uint step = eval_len / trace_len;
    uint next_idx = (tid + step) % eval_len;

    M31 a_cur = M31{trace_cols[tid]};
    M31 b_cur = M31{trace_cols[eval_len + tid]};
    M31 a_next = M31{trace_cols[next_idx]};
    M31 b_next = M31{trace_cols[eval_len + next_idx]};

    // Fibonacci constraints hardcoded for now
    M31 c0 = m31_sub(a_next, b_cur);
    M31 c1 = m31_sub(b_next, m31_add(a_cur, b_cur));

    M31 acc = m31_mul(c0, inv_vz);
    M31 alpha_pow = alpha;
    acc = m31_add(acc, m31_mul(alpha_pow, m31_mul(c1, inv_vz)));

    output[tid] = acc.v;
}
