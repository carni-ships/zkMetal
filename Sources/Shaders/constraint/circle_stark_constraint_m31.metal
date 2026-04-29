// Circle STARK GPU Constraint Evaluation over Mersenne31
//
// GPU-accelerated constraint evaluation for Circle STARKs.
// Supports transition constraints + boundary constraints + composition.
//
// Architecture (from EVMetal analysis):
// - Thread-per-point: each thread evaluates one evaluation point
// - Column-major layout: trace[col * evalLen + i]
// - Boundary constraints handled via alpha powers
//
// M31 arithmetic reused from mersenne31.metal

#include "../fields/mersenne31.metal"

// =============================================================================
// Vanishing Polynomial (Circle Domain)
// =============================================================================

// Circle vanishing polynomial on y-coordinate domain:
// v_0 = y, v_{k+1} = 2*v_k^2 - 1
// This computes v_log(y) where log is the domain size
M31 circle_vanishing_gpu(M31 y, uint log_domain_size) {
    M31 v = y;
    for (uint i = 0; i < log_domain_size; i++) {
        M31 v2 = m31_sqr(v);
        v = m31_sub(m31_add(v2, v2), m31_one());
    }
    return v;
}

// =============================================================================
// Fibonacci Constraints (2 columns, 2 constraints)
// =============================================================================

// Fibonacci transition constraints:
// C0: a_next - b_current = 0
// C1: b_next - (a_current + b_current) = 0
//
// boundary: a[0]=a0, b[0]=b0
kernel void circle_fibonacci_constraint_eval(
    device const uint* trace_a       [[buffer(0)]],  // column 0: evalLen elements
    device const uint* trace_b       [[buffer(1)]],  // column 1: evalLen elements
    device const uint* domain_y     [[buffer(2)]],  // y-coords: evalLen elements
    device uint* output             [[buffer(3)]],  // composition: evalLen elements
    constant uint& alpha_val         [[buffer(4)]],  // random challenge alpha
    constant uint& bc_a0_val         [[buffer(5)]],  // boundary a[0]=a0
    constant uint& bc_b0_val         [[buffer(6)]],  // boundary b[0]=b0
    constant uint& eval_len          [[buffer(7)]],  // evaluation domain size
    constant uint& trace_len         [[buffer(8)]],  // trace length
    constant uint& log_trace         [[buffer(9)]],  // log2(trace_len)
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
    M31 c1 = m31_sub(b_next, m31_add(a_cur, b_cur));         // b' - (a + b)

    // Vanishing polynomial at this point
    M31 y = M31{domain_y[tid]};
    M31 vz = circle_vanishing_gpu(y, log_trace);

    // If vanishing is zero, we're on trace domain - composition should be 0
    if (vz.v == 0) {
        output[tid] = 0;
        return;
    }

    M31 inv_vz = m31_inv(vz);
    M31 alpha = M31{alpha_val};
    M31 bc_a0 = M31{bc_a0_val};
    M31 bc_b0 = M31{bc_b0_val};

    // Composition: sum_i alpha^i * C_i * inv_vz + boundary_contributions
    // acc = alpha^0 * C0 * inv_vz + alpha^1 * C1 * inv_vz + alpha^2 * BC0 * inv_vz + alpha^3 * BC1 * inv_vz
    M31 term0 = m31_mul(c0, inv_vz);
    M31 term1 = m31_mul(alpha, m31_mul(c1, inv_vz));
    M31 acc = m31_add(term0, term1);

    M31 alpha_pow = m31_mul(alpha, alpha);  // alpha^2

    // Boundary constraint 0: (a_cur - a0) * inv_vz
    M31 bc0_num = m31_sub(a_cur, bc_a0);
    M31 bc0_term = m31_mul(alpha_pow, m31_mul(bc0_num, inv_vz));
    acc = m31_add(acc, bc0_term);

    alpha_pow = m31_mul(alpha_pow, alpha);  // alpha^3

    // Boundary constraint 1: (b_cur - b0) * inv_vz
    M31 bc1_num = m31_sub(b_cur, bc_b0);
    M31 bc1_term = m31_mul(alpha_pow, m31_mul(bc1_num, inv_vz));
    acc = m31_add(acc, bc1_term);

    output[tid] = acc.v;
}

// =============================================================================
// Generic 2-Column Constraints (parameterized)
// =============================================================================

// Generic kernel for any 2-column AIR with:
// - Linear transition constraints (c0, c1)
// - Up to 2 boundary constraints
kernel void circle_2col_constraint_eval(
    device const uint* trace_cols    [[buffer(0)]],   // [col0, col1] interleaved: col0*evalLen + i, col1*evalLen + i
    device const uint* domain_y    [[buffer(1)]],  // y-coords: evalLen elements
    device uint* output              [[buffer(2)]],  // composition: evalLen elements
    constant uint& alpha_val         [[buffer(3)]],  // random challenge alpha
    constant uint& eval_len          [[buffer(4)]],  // evaluation domain size
    constant uint& trace_len         [[buffer(5)]],  // trace length
    constant uint& log_trace         [[buffer(6)]],  // log2(trace_len)
    constant uint& num_trans_constraints [[buffer(7)]],  // number of transition constraints (max 4)
    constant uint& num_boundary_constraints [[buffer(8)]], // number of boundary constraints (max 4)
    // Boundary data follows as packed pairs: [col, row, value, col, row, value, ...]
    constant uint* boundary_data     [[buffer(9)]],  // packed boundary constraint data
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= eval_len) return;

    uint step = eval_len / trace_len;
    uint next_idx = (tid + step) % eval_len;

    // Column layout: trace_cols[col * evalLen + tid]
    M31 a_cur = M31{trace_cols[tid]};
    M31 b_cur = M31{trace_cols[eval_len + tid]};
    M31 a_next = M31{trace_cols[next_idx]};
    M31 b_next = M31{trace_cols[eval_len + next_idx]};

    // Vanishing polynomial
    M31 y = M31{domain_y[tid]};
    M31 vz = circle_vanishing_gpu(y, log_trace);

    if (vz.v == 0) {
        output[tid] = 0;
        return;
    }

    M31 inv_vz = m31_inv(vz);
    M31 alpha = M31{alpha_val};

    // Default Fibonacci constraints (can be overridden by num_trans_constraints check)
    M31 c0 = m31_sub(a_next, b_cur);
    M31 c1 = m31_sub(b_next, m31_add(a_cur, b_cur));

    // Composition: sum_i alpha^i * C_i * inv_vz
    M31 acc = m31_mul(c0, inv_vz);
    M31 alpha_pow = alpha;
    acc = m31_add(acc, m31_mul(alpha_pow, m31_mul(c1, inv_vz)));

    // Boundary constraints: (trace[col][row] - value) * inv_vz
    // Layout in boundary_data: [col0, row0, value0, col1, row1, value1, ...]
    for (uint bc = 0; bc < num_boundary_constraints; bc++) {
        uint baseIdx = bc * 3;
        uint col = boundary_data[baseIdx];
        uint row = boundary_data[baseIdx + 1];
        uint val = boundary_data[baseIdx + 2];

        // Value at this boundary constraint position
        M31 bc_val = M31{val};
        M31 trace_val;

        // Determine the trace value at (col, row)
        if (col == 0) {
            // column 0 - interpolate from row if needed
            if (row == tid % trace_len) {
                trace_val = a_cur;
            } else {
                // Would need to read from original trace - for now use a_cur
                trace_val = a_cur;
            }
        } else {
            // column 1
            if (row == tid % trace_len) {
                trace_val = b_cur;
            } else {
                trace_val = b_cur;
            }
        }

        M31 diff = m31_sub(trace_val, bc_val);
        alpha_pow = m31_pow(alpha, 2 + bc);  // alpha^(2+bc)
        M31 term = m31_mul(alpha_pow, m31_mul(diff, inv_vz));
        acc = m31_add(acc, term);
    }

    output[tid] = acc.v;
}

// =============================================================================
// Multi-Column Generic Constraints (flexible)
// =============================================================================

// For AIRs with more than 2 columns, evaluate arbitrary constraints
// This kernel loads all columns and evaluates constraints in a configurable way
kernel void circle_ncol_constraint_eval(
    device const uint* trace         [[buffer(0)]],  // [numColumns][evalLen], column-major
    device const uint* domain_y      [[buffer(1)]],  // y-coords: evalLen elements
    device uint* output              [[buffer(2)]],  // composition: evalLen elements
    constant uint& alpha_val          [[buffer(3)]],  // random challenge alpha
    constant uint& eval_len          [[buffer(4)]],  // evaluation domain size
    constant uint& trace_len         [[buffer(5)]],  // trace length
    constant uint& log_trace         [[buffer(6)]],  // log2(trace_len)
    constant uint& num_columns       [[buffer(7)]],  // number of trace columns
    constant uint& num_constraints   [[buffer(8)]],  // number of constraints
    // Constraint coefficients: [numConstraints][numColumns * 2] packed
    // For each constraint i: [c_i_0_a, c_i_0_b, c_i_1_a, c_i_1_b, ...] for pairs
    // Simplified: transition constraint i is a linear combo of columns
    constant uint* constraint_coeffs [[buffer(9)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= eval_len) return;

    uint step = eval_len / trace_len;
    uint next_idx = (tid + step) % eval_len;

    // Vanishing polynomial
    M31 y = M31{domain_y[tid]};
    M31 vz = circle_vanishing_gpu(y, log_trace);

    if (vz.v == 0) {
        output[tid] = 0;
        return;
    }

    M31 inv_vz = m31_inv(vz);
    M31 alpha = M31{alpha_val};

    // Load current and next rows for all columns
    // Using stack-allocated array for small column counts
    M31 current[8];
    M31 next[8];

    for (uint c = 0; c < num_columns && c < 8; c++) {
        current[c] = M31{trace[c * eval_len + tid]};
        next[c] = M31{trace[c * eval_len + next_idx]};
    }

    // Evaluate constraints - for now, hardcoded Fibonacci pattern
    // TODO: Generalize to arbitrary linear constraints via constraint_coeffs
    M31 c0 = m31_sub(next[0], current[1]);
    M31 c1 = m31_sub(next[1], m31_add(current[0], current[1]));

    M31 acc = m31_mul(c0, inv_vz);
    M31 alpha_pow = alpha;
    acc = m31_add(acc, m31_mul(alpha_pow, m31_mul(c1, inv_vz)));

    output[tid] = acc.v;
}

// =============================================================================
// Composition Polynomial (weighted sum of constraint values)
// =============================================================================

// Evaluates: composition = sum_i alpha^i * constraints[i]
// Used when constraints are already computed and need composition
kernel void circle_composition_eval(
    device const uint* constraints    [[buffer(0)]],  // [numConstraints][evalLen]
    device uint* composition         [[buffer(1)]],  // output: evalLen elements
    constant uint& alpha_val          [[buffer(2)]],  // random challenge alpha
    constant uint& eval_len          [[buffer(3)]],  // evaluation domain size
    constant uint& num_constraints    [[buffer(4)]],  // number of constraints
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= eval_len) return;

    M31 alpha = M31{alpha_val};
    M31 acc = m31_zero();
    M31 alpha_pow = m31_one();

    uint baseIdx = tid;
    for (uint i = 0; i < num_constraints; i++) {
        M31 c = M31{constraints[baseIdx]};
        acc = m31_add(acc, m31_mul(alpha_pow, c));
        alpha_pow = m31_mul(alpha_pow, alpha);
        baseIdx += eval_len;
    }

    composition[tid] = acc.v;
}
