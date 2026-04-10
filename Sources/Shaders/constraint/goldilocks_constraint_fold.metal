// Goldilocks constraint evaluation + FRI fold Metal kernels
//
// Goldilocks field: p = 2^64 - 2^32 + 1
// Three kernel variants:
//   1. goldilocks_constraint_eval: basic constraint eval (post-NTT/LDE), outputs n quotients
//   2. goldilocks_constraint_fold_first: constraint eval + 1st FRI fold, outputs n/2
//   3. goldilocks_constraint_fold_2r: constraint eval + 2 FRI folds, outputs n/4
//
// Fibonacci-specific prototype. Vanishing inverses precomputed and passed as buffers.

#include "../fields/goldilocks.metal"

// --- Goldilocks Fibonacci constraint evaluation (post-NTT/LDE) ---
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
kernel void goldilocks_constraint_eval(
    device const Gl* trace            [[buffer(0)]],     // [num_cols][num_rows] row-major
    device Gl* quotient              [[buffer(1)]],     // output: num_rows elements
    device const Gl* vanishing_inv   [[buffer(2)]],    // 1/Z_H(x) at each point
    device const Gl* alpha_powers    [[buffer(3)]],    // [alpha^0, alpha^1, alpha^2, alpha^3]
    constant uint2& dims             [[buffer(4)]],     // (num_cols, num_rows)
    uint gid                         [[thread_position_in_grid]]
) {
    uint num_cols = dims[0];
    uint num_rows = dims[1];
    if (gid >= num_rows) return;

    uint next_i = (gid + 1) % num_rows;

    // Load current and next-row values
    Gl cur[8];
    Gl nxt[8];
    for (uint c = 0; c < num_cols; c++) {
        cur[c] = trace[c * num_rows + gid];
        nxt[c] = trace[c * num_rows + next_i];
    }

    // Fibonacci constraints: C0 = next[0] - cur[1], C1 = next[1] - (cur[0] + cur[1])
    Gl c0 = gl_sub(nxt[0], cur[1]);
    Gl c1 = gl_sub(nxt[1], gl_add(cur[0], cur[1]));

    // Weighted sum: alpha^0 * C0 + alpha^1 * C1
    Gl acc = gl_add(gl_mul(alpha_powers[0], c0), gl_mul(alpha_powers[1], c1));

    // Divide by vanishing polynomial
    quotient[gid] = gl_mul(acc, vanishing_inv[gid]);
}

// --- Goldilocks Fused constraint eval + first FRI fold ---
// Combines constraint eval with FRI first fold (n -> n/2).
// Each thread processes positions i and i+half_n.
//
// FRI fold formula:
//   folded[i] = (q[i] + q[i+half]) + alpha * (q[i] - q[i+half]) * domain_inv[i]
kernel void goldilocks_constraint_fold_first(
    device const Gl* trace            [[buffer(0)]],     // [num_cols][num_rows]
    device Gl* folded_out            [[buffer(1)]],     // output: num_rows/2 elements
    device const Gl* vanishing_inv  [[buffer(2)]],    // 1/Z_H(x) (size num_rows)
    device const Gl* domain_inv      [[buffer(3)]],    // 1/domain[i] for fold (size num_rows/2)
    device const Gl* alpha_powers   [[buffer(4)]],   // constraint alpha powers
    constant Gl* alpha_fold         [[buffer(5)]],   // FRI fold challenge
    constant uint2& dims            [[buffer(6)]],   // (num_cols, num_rows)
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
    Gl cur_i[8];
    Gl nxt_i[8];
    for (uint c = 0; c < num_cols; c++) {
        cur_i[c] = trace[c * num_rows + i];
        nxt_i[c] = trace[c * num_rows + next_i];
    }

    // Load trace at position j
    Gl cur_j[8];
    Gl nxt_j[8];
    for (uint c = 0; c < num_cols; c++) {
        cur_j[c] = trace[c * num_rows + j];
        nxt_j[c] = trace[c * num_rows + next_j];
    }

    // Fibonacci constraints at i
    Gl c0_i = gl_sub(nxt_i[0], cur_i[1]);
    Gl c1_i = gl_sub(nxt_i[1], gl_add(cur_i[0], cur_i[1]));
    Gl q_i = gl_mul(gl_add(gl_mul(alpha_powers[0], c0_i), gl_mul(alpha_powers[1], c1_i)), vanishing_inv[i]);

    // Fibonacci constraints at j
    Gl c0_j = gl_sub(nxt_j[0], cur_j[1]);
    Gl c1_j = gl_sub(nxt_j[1], gl_add(cur_j[0], cur_j[1]));
    Gl q_j = gl_mul(gl_add(gl_mul(alpha_powers[0], c0_j), gl_mul(alpha_powers[1], c1_j)), vanishing_inv[j]);

    // FRI fold: folded[i] = (q_i + q_j) + alpha * (q_i - q_j) * domain_inv[i]
    Gl sum = gl_add(q_i, q_j);
    Gl diff = gl_sub(q_i, q_j);
    Gl diff_term = gl_mul(gl_mul(alpha_fold[0], diff), domain_inv[gid]);
    folded_out[gid] = gl_add(sum, diff_term);
}

// --- Goldilocks Fused constraint eval + 2-round FRI fold ---
// Combines constraint eval with two consecutive FRI fold rounds (n -> n/4).
// Each thread processes 4 positions, computes 2 fold rounds.
kernel void goldilocks_constraint_fold_2r(
    device const Gl* trace            [[buffer(0)]],     // [num_cols][num_rows]
    device Gl* folded_out            [[buffer(1)]],     // output: num_rows/4 elements
    device const Gl* vanishing_inv  [[buffer(2)]],    // 1/Z_H (size num_rows)
    device const Gl* domain_inv      [[buffer(3)]],    // round-1 domain inverses (size num_rows/2)
    device const Gl* domain_inv2    [[buffer(4)]],    // round-2 domain inverses (size num_rows/4)
    device const Gl* alpha_powers   [[buffer(5)]],   // constraint alpha
    constant Gl* alpha0             [[buffer(6)]],   // FRI challenge round 1
    constant Gl* alpha1             [[buffer(7)]],   // FRI challenge round 2
    constant uint2& dims             [[buffer(8)]],   // (num_cols, num_rows)
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

    // Fibonacci constraints at 4 positions
    uint next_i = (i + 1) % num_rows;
    Gl c0_0 = gl_sub(trace[0 * num_rows + next_i], trace[1 * num_rows + i]);
    Gl c1_0 = gl_sub(trace[1 * num_rows + next_i], gl_add(trace[0 * num_rows + i], trace[1 * num_rows + i]));
    Gl q0 = gl_mul(gl_add(gl_mul(alpha_powers[0], c0_0), gl_mul(alpha_powers[1], c1_0)), vanishing_inv[i]);

    uint next_j = (j + 1) % num_rows;
    Gl c0_1 = gl_sub(trace[0 * num_rows + next_j], trace[1 * num_rows + j]);
    Gl c1_1 = gl_sub(trace[1 * num_rows + next_j], gl_add(trace[0 * num_rows + j], trace[1 * num_rows + j]));
    Gl q1 = gl_mul(gl_add(gl_mul(alpha_powers[0], c0_1), gl_mul(alpha_powers[1], c1_1)), vanishing_inv[j]);

    uint next_k = (k + 1) % num_rows;
    Gl c0_2 = gl_sub(trace[0 * num_rows + next_k], trace[1 * num_rows + k]);
    Gl c1_2 = gl_sub(trace[1 * num_rows + next_k], gl_add(trace[0 * num_rows + k], trace[1 * num_rows + k]));
    Gl q2 = gl_mul(gl_add(gl_mul(alpha_powers[0], c0_2), gl_mul(alpha_powers[1], c1_2)), vanishing_inv[k]);

    uint next_l = (l + 1) % num_rows;
    Gl c0_3 = gl_sub(trace[0 * num_rows + next_l], trace[1 * num_rows + l]);
    Gl c1_3 = gl_sub(trace[1 * num_rows + next_l], gl_add(trace[0 * num_rows + l], trace[1 * num_rows + l]));
    Gl q3 = gl_mul(gl_add(gl_mul(alpha_powers[0], c0_3), gl_mul(alpha_powers[1], c1_3)), vanishing_inv[l]);

    // Round 1: fold pairs (q0,q2) and (q1,q3) with alpha0
    Gl sum02 = gl_add(q0, q2);
    Gl diff02 = gl_sub(q0, q2);
    Gl f1_lo = gl_add(sum02, gl_mul(gl_mul(alpha0[0], diff02), domain_inv[gid]));

    Gl sum13 = gl_add(q1, q3);
    Gl diff13 = gl_sub(q1, q3);
    Gl f1_hi = gl_add(sum13, gl_mul(gl_mul(alpha0[0], diff13), domain_inv[gid + quarter]));

    // Round 2: fold (f1_lo, f1_hi) with alpha1 and domain_inv2
    Gl sum_f = gl_add(f1_lo, f1_hi);
    Gl diff_f = gl_sub(f1_lo, f1_hi);
    folded_out[gid] = gl_add(sum_f, gl_mul(gl_mul(alpha1[0], diff_f), domain_inv2[gid]));
}
