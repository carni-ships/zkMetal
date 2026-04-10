// BabyBear constraint evaluation + FRI fold Metal kernels
//
// Implements GPU constraint evaluation for BabyBear STARK with optional fusion
// with the FRI fold step. BabyBear field: p = 2^31 - 2^27 + 1.
//
// Three kernel variants:
//   1. babybear_constraint_eval: basic constraint eval (post-NTT/LDE), outputs n quotients
//   2. babybear_constraint_fold_first: constraint eval + 1st FRI fold, outputs n/2
//   3. babybear_constraint_fold_2r: constraint eval + 2 FRI folds, outputs n/4
//
// Constraint model: opcode-based, supports:
//   - Fibonacci-style AIR (a_next = f(a,b), b_next = g(a,b))
//   - General selector-weighted constraints
//
// For the prototype, we implement Fibonacci-specific constraints.
// General constraint support would use a similar opcode-based approach
// to the BN254 constraint_eval.metal kernel.

#include "../fields/babybear.metal"

// --- BabyBear Fibonacci constraint evaluation (post-NTT/LDE) ---
// Each thread evaluates Fibonacci constraints at one LDE point.
// Constraints: C0 = a_next - b, C1 = b_next - (a + b)
// Quotient = (alpha^0 * C0 + alpha^1 * C1) * vanishing_inv[i]
//
// Buffer layout:
//   buffer(0): trace columns (row-major: col_j at offset j * num_rows)
//   buffer(1): output quotient (num_rows elements)
//   buffer(2): vanishing polynomial inverses (1/Z_H)
//   buffer(3): alpha powers [alpha^0, alpha^1, alpha^2, alpha^3]
//   buffer(4): num_cols, num_rows as uints
kernel void babybear_constraint_eval(
    device const Bb* trace            [[buffer(0)]],     // [num_cols][num_rows] row-major
    device Bb* quotient              [[buffer(1)]],     // output: num_rows elements
    device const Bb* vanishing_inv [[buffer(2)]],     // 1/Z_H(x) at each point
    device const Bb* alpha_powers   [[buffer(3)]],    // [alpha^0, alpha^1, alpha^2, alpha^3]
    constant uint2& dims             [[buffer(4)]],     // (num_cols, num_rows)
    uint gid                         [[thread_position_in_grid]]
) {
    uint num_cols = dims[0];
    uint num_rows = dims[1];
    if (gid >= num_rows) return;

    uint step = 1;  // next-row step in evaluation domain
    uint next_i = (gid + step) % num_rows;

    // Load current and next-row values for each column
    Bb cur[8];   // max 8 columns
    Bb nxt[8];
    for (uint c = 0; c < num_cols; c++) {
        cur[c] = trace[c * num_rows + gid];
        nxt[c] = trace[c * num_rows + next_i];
    }

    // Fibonacci constraints (hardcoded for prototype):
    // Col 0: C0 = next[0] - cur[1]
    // Col 1: C1 = next[1] - (cur[0] + cur[1])
    // For general constraints, use the opcode-based evaluator instead.
    Bb c0 = bb_sub(nxt[0], cur[1]);
    Bb c1 = bb_sub(nxt[1], bb_add(cur[0], cur[1]));

    // Weighted sum: alpha^0 * C0 + alpha^1 * C1
    Bb acc = bb_add(bb_mul(alpha_powers[0], c0), bb_mul(alpha_powers[1], c1));

    // Divide by vanishing polynomial
    quotient[gid] = bb_mul(acc, vanishing_inv[gid]);
}

// --- BabyBear Fused constraint eval + first FRI fold ---
// Combines babybear_constraint_eval with FRI first fold (n -> n/2).
// Each thread processes positions i and i+half_n.
//
// FRI fold formula (BabyBear uses standard fold-by-2):
//   folded[i] = (q[i] + q[i+half]) + alpha * (q[i] - q[i+half]) * domain_inv[i]
kernel void babybear_constraint_fold_first(
    device const Bb* trace            [[buffer(0)]],     // [num_cols][num_rows]
    device Bb* folded_out            [[buffer(1)]],     // output: num_rows/2 elements
    device const Bb* vanishing_inv   [[buffer(2)]],    // 1/Z_H(x) (size num_rows)
    device const Bb* domain_inv      [[buffer(3)]],     // 1/domain[i] for fold (size num_rows/2)
    device const Bb* alpha_powers    [[buffer(4)]],    // constraint alpha powers
    constant Bb* alpha_fold         [[buffer(5)]],    // FRI fold challenge
    constant uint2& dims             [[buffer(6)]],    // (num_cols, num_rows)
    uint gid                         [[thread_position_in_grid]]
) {
    uint num_cols = dims[0];
    uint num_rows = dims[1];
    uint half_n = num_rows >> 1;
    if (gid >= half_n) return;

    uint i = gid;
    uint j = gid + half_n;

    uint next_i = (i + 1) % num_rows;
    uint next_j = (j + 1) % num_rows;

    // Load trace at position i
    Bb cur_i[8];
    Bb nxt_i[8];
    for (uint c = 0; c < num_cols; c++) {
        cur_i[c] = trace[c * num_rows + i];
        nxt_i[c] = trace[c * num_rows + next_i];
    }

    // Load trace at position j
    Bb cur_j[8];
    Bb nxt_j[8];
    for (uint c = 0; c < num_cols; c++) {
        cur_j[c] = trace[c * num_rows + j];
        nxt_j[c] = trace[c * num_rows + next_j];
    }

    // Fibonacci constraints at i
    Bb c0_i = bb_sub(nxt_i[0], cur_i[1]);
    Bb c1_i = bb_sub(nxt_i[1], bb_add(cur_i[0], cur_i[1]));
    Bb q_i = bb_mul(bb_add(bb_mul(alpha_powers[0], c0_i), bb_mul(alpha_powers[1], c1_i)), vanishing_inv[i]);

    // Fibonacci constraints at j
    Bb c0_j = bb_sub(nxt_j[0], cur_j[1]);
    Bb c1_j = bb_sub(nxt_j[1], bb_add(cur_j[0], cur_j[1]));
    Bb q_j = bb_mul(bb_add(bb_mul(alpha_powers[0], c0_j), bb_mul(alpha_powers[1], c1_j)), vanishing_inv[j]);

    // FRI fold: folded[i] = (q_i + q_j) + alpha * (q_i - q_j) * domain_inv[i]
    Bb sum = bb_add(q_i, q_j);
    Bb diff = bb_sub(q_i, q_j);
    Bb diff_term = bb_mul(bb_mul(alpha_fold[0], diff), domain_inv[gid]);
    folded_out[gid] = bb_add(sum, diff_term);
}

// --- BabyBear Fused constraint eval + 2-round FRI fold ---
// Combines constraint eval with two consecutive FRI fold rounds (n -> n/4).
// Each thread processes 4 positions, computes 2 fold rounds.
kernel void babybear_constraint_fold_2r(
    device const Bb* trace            [[buffer(0)]],     // [num_cols][num_rows]
    device Bb* folded_out            [[buffer(1)]],     // output: num_rows/4 elements
    device const Bb* vanishing_inv   [[buffer(2)]],    // 1/Z_H (size num_rows)
    device const Bb* domain_inv      [[buffer(3)]],    // round-1 domain inverses (size num_rows/2)
    device const Bb* domain_inv2     [[buffer(4)]],    // round-2 domain inverses (size num_rows/4)
    device const Bb* alpha_powers    [[buffer(5)]],    // constraint alpha
    constant Bb* alpha0             [[buffer(6)]],    // FRI challenge round 1
    constant Bb* alpha1            [[buffer(7)]],    // FRI challenge round 2
    constant uint2& dims             [[buffer(8)]],    // (num_cols, num_rows)
    uint gid                         [[thread_position_in_grid]]
) {
    uint num_cols = dims[0];
    uint num_rows = dims[1];
    uint quarter = num_rows >> 2;
    if (gid >= quarter) return;

    uint half_n = num_rows >> 1;
    uint i = gid;
    uint j = gid + quarter;
    uint k = gid + half_n;
    uint l = gid + half_n + quarter;

    // Helper to compute constraint quotient at position pos
    // Inlined for performance
    uint next_i = (i + 1) % num_rows;
    Bb c0_0 = bb_sub(trace[0 * num_rows + next_i], trace[1 * num_rows + i]);
    Bb c1_0 = bb_sub(trace[1 * num_rows + next_i], bb_add(trace[0 * num_rows + i], trace[1 * num_rows + i]));
    Bb q0 = bb_mul(bb_add(bb_mul(alpha_powers[0], c0_0), bb_mul(alpha_powers[1], c1_0)), vanishing_inv[i]);

    uint next_j = (j + 1) % num_rows;
    Bb c0_1 = bb_sub(trace[0 * num_rows + next_j], trace[1 * num_rows + j]);
    Bb c1_1 = bb_sub(trace[1 * num_rows + next_j], bb_add(trace[0 * num_rows + j], trace[1 * num_rows + j]));
    Bb q1 = bb_mul(bb_add(bb_mul(alpha_powers[0], c0_1), bb_mul(alpha_powers[1], c1_1)), vanishing_inv[j]);

    uint next_k = (k + 1) % num_rows;
    Bb c0_2 = bb_sub(trace[0 * num_rows + next_k], trace[1 * num_rows + k]);
    Bb c1_2 = bb_sub(trace[1 * num_rows + next_k], bb_add(trace[0 * num_rows + k], trace[1 * num_rows + k]));
    Bb q2 = bb_mul(bb_add(bb_mul(alpha_powers[0], c0_2), bb_mul(alpha_powers[1], c1_2)), vanishing_inv[k]);

    uint next_l = (l + 1) % num_rows;
    Bb c0_3 = bb_sub(trace[0 * num_rows + next_l], trace[1 * num_rows + l]);
    Bb c1_3 = bb_sub(trace[1 * num_rows + next_l], bb_add(trace[0 * num_rows + l], trace[1 * num_rows + l]));
    Bb q3 = bb_mul(bb_add(bb_mul(alpha_powers[0], c0_3), bb_mul(alpha_powers[1], c1_3)), vanishing_inv[l]);

    // Round 1: fold pairs (q0,q2) and (q1,q3) with alpha0
    Bb sum02 = bb_add(q0, q2);
    Bb diff02 = bb_sub(q0, q2);
    Bb f1_lo = bb_add(sum02, bb_mul(bb_mul(alpha0[0], diff02), domain_inv[gid]));

    Bb sum13 = bb_add(q1, q3);
    Bb diff13 = bb_sub(q1, q3);
    Bb f1_hi = bb_add(sum13, bb_mul(bb_mul(alpha0[0], diff13), domain_inv[gid + quarter]));

    // Round 2: fold (f1_lo, f1_hi) with alpha1 and domain_inv2
    Bb sum_f = bb_add(f1_lo, f1_hi);
    Bb diff_f = bb_sub(f1_lo, f1_hi);
    folded_out[gid] = bb_add(sum_f, bb_mul(bb_mul(alpha1[0], diff_f), domain_inv2[gid]));
}
