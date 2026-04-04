// M31 witness trace evaluation kernels for GPU witness generation
//
// Three kernels:
//   1. witness_m31_evaluate — generic instruction-stream trace evaluator (independent rows)
//   2. witness_m31_fibonacci — specialized Fibonacci trace: parallel prefix-doubling
//   3. witness_m31_fibonacci_batch — batch Fibonacci for large traces using scan
//
// For the Fibonacci AIR (a' = b, b' = a + b), rows are sequential so naive
// per-row parallelism fails. Instead we use the matrix-power doubling trick:
//   [a_{i+k}, b_{i+k}] = M^k * [a_i, b_i]  where M = [[0,1],[1,1]]
// We precompute M^{2^j} and use a parallel prefix scan to compute all rows.

#include <metal_stdlib>
using namespace metal;

// --- M31 field included at compile time (combined with mersenne31.metal) ---

// =========================================================================
// Kernel 1: Generic M31 instruction-stream trace evaluator
// Each thread evaluates one row independently.
// =========================================================================

// Opcode definitions
constant uint M31_OP_ADD    = 0;
constant uint M31_OP_MUL    = 1;
constant uint M31_OP_COPY   = 2;
constant uint M31_OP_SUB    = 3;
constant uint M31_OP_LOAD   = 4;
constant uint M31_OP_SQR    = 5;
constant uint M31_OP_DOUBLE = 6;
constant uint M31_OP_NEG    = 7;
constant uint M31_OP_SELECT = 8;
constant uint M31_FLAG_CONST = 0x80;

kernel void witness_m31_evaluate(
    device uint* trace              [[buffer(0)]],  // n x num_cols matrix of M31.v
    device const uint* inputs       [[buffer(1)]],  // n x input_width matrix
    constant uint* program          [[buffer(2)]],  // instruction stream
    device const uint* const_pool   [[buffer(3)]],  // constant pool (M31.v values)
    constant uint& num_cols         [[buffer(4)]],
    constant uint& num_instrs       [[buffer(5)]],
    constant uint& input_width      [[buffer(6)]],
    constant uint& num_rows         [[buffer(7)]],
    uint gid                        [[thread_position_in_grid]])
{
    uint row = gid;
    if (row >= num_rows) return;

    uint base = row * num_cols;

    for (uint i = 0; i < num_instrs; i++) {
        uint op_raw = program[i * 4];
        uint dst    = program[i * 4 + 1];
        uint src1   = program[i * 4 + 2];
        uint src2   = program[i * 4 + 3];

        uint op = op_raw & 0x7F;
        bool use_const = (op_raw & M31_FLAG_CONST) != 0;

        if (op == M31_OP_LOAD) {
            trace[base + dst] = inputs[row * input_width + src1];
            continue;
        }

        M31 a = M31{trace[base + src1]};
        M31 b;
        if (use_const) {
            b = M31{const_pool[src2]};
        } else {
            b = M31{trace[base + src2]};
        }

        M31 result;
        switch (op) {
            case M31_OP_ADD:    result = m31_add(a, b); break;
            case M31_OP_MUL:    result = m31_mul(a, b); break;
            case M31_OP_COPY:   result = a; break;
            case M31_OP_SUB:    result = m31_sub(a, b); break;
            case M31_OP_SQR:    result = m31_sqr(a); break;
            case M31_OP_DOUBLE: result = m31_add(a, a); break;
            case M31_OP_NEG:    result = m31_neg(a); break;
            case M31_OP_SELECT: {
                uint false_col = src2 & 0xFFFF;
                uint sel_col = (src2 >> 16) & 0xFFFF;
                uint sel = trace[base + sel_col];
                result = (sel == 0) ? M31{trace[base + false_col]} : a;
                break;
            }
            default: result = m31_zero(); break;
        }

        trace[base + dst] = result.v;
    }
}

// =========================================================================
// Kernel 2: Specialized Fibonacci trace via parallel prefix-doubling
//
// The Fibonacci recurrence [a',b'] = [b, a+b] has transfer matrix M=[[0,1],[1,1]].
// M^k applied to [a0,b0] gives [a_k, b_k]. M^k can be represented as a 2x2 matrix
// over M31. We use a parallel prefix scan on these matrices.
//
// Phase 1: Each thread computes M^(thread_id) via doubling, then applies to [a0,b0].
// This is O(log n) per thread but all threads run in parallel.
//
// For the Fibonacci matrix M = [[0,1],[1,1]]:
//   M^1  = [[0,1],[1,1]]
//   M^2  = [[1,1],[1,2]]
//   M^k  = [[F(k-1), F(k)], [F(k), F(k+1)]]
// where F(n) is the n-th Fibonacci number mod p.
//
// We use fast matrix exponentiation: compute M^k in O(log k) muls.
// =========================================================================

// 2x2 matrix over M31 (row-major: [[a,b],[c,d]])
struct Mat2x2_M31 {
    M31 a, b, c, d;
};

Mat2x2_M31 mat2_identity() {
    return Mat2x2_M31{m31_one(), m31_zero(), m31_zero(), m31_one()};
}

Mat2x2_M31 mat2_mul(Mat2x2_M31 x, Mat2x2_M31 y) {
    return Mat2x2_M31{
        m31_add(m31_mul(x.a, y.a), m31_mul(x.b, y.c)),
        m31_add(m31_mul(x.a, y.b), m31_mul(x.b, y.d)),
        m31_add(m31_mul(x.c, y.a), m31_mul(x.d, y.c)),
        m31_add(m31_mul(x.c, y.b), m31_mul(x.d, y.d))
    };
}

// Compute M^k where M = [[0,1],[1,1]] using repeated squaring
Mat2x2_M31 fib_matrix_pow(uint k) {
    Mat2x2_M31 result = mat2_identity();
    Mat2x2_M31 base = Mat2x2_M31{m31_zero(), m31_one(), m31_one(), m31_one()};
    while (k > 0) {
        if (k & 1) result = mat2_mul(result, base);
        base = mat2_mul(base, base);
        k >>= 1;
    }
    return result;
}

kernel void witness_m31_fibonacci(
    device uint* col_a        [[buffer(0)]],  // output column A (n values)
    device uint* col_b        [[buffer(1)]],  // output column B (n values)
    constant uint& a0_val     [[buffer(2)]],  // initial a0
    constant uint& b0_val     [[buffer(3)]],  // initial b0
    constant uint& num_rows   [[buffer(4)]],
    uint gid                  [[thread_position_in_grid]])
{
    uint row = gid;
    if (row >= num_rows) return;

    // Compute M^row * [a0, b0]
    Mat2x2_M31 mk = fib_matrix_pow(row);
    M31 a0 = M31{a0_val};
    M31 b0 = M31{b0_val};

    // [a_row, b_row] = M^row * [a0, b0]
    M31 a_row = m31_add(m31_mul(mk.a, a0), m31_mul(mk.b, b0));
    M31 b_row = m31_add(m31_mul(mk.c, a0), m31_mul(mk.d, b0));

    col_a[row] = a_row.v;
    col_b[row] = b_row.v;
}

// =========================================================================
// Kernel 3: Fibonacci with precomputed M^{2^j} powers (uploaded from CPU)
// CPU precomputes M^{2^j} for j=0..23 (only 24 matrices, trivial).
// Each GPU thread reads the precomputed powers from a constant buffer,
// decomposes its row index in binary, and multiplies the relevant powers.
// This reduces per-thread work from O(log^2 n) to O(log n) matrix lookups.
//
// Powers buffer layout: 24 * 4 uint values = [a,b,c,d] per matrix.
// =========================================================================

kernel void witness_m31_fibonacci_shared(
    device uint* col_a              [[buffer(0)]],
    device uint* col_b              [[buffer(1)]],
    constant uint& a0_val           [[buffer(2)]],
    constant uint& b0_val           [[buffer(3)]],
    constant uint& num_rows         [[buffer(4)]],
    constant uint& log_num_rows     [[buffer(5)]],
    constant uint* powers_buf       [[buffer(6)]],  // 24 * 4 uint values: precomputed M^{2^j}
    uint gid                        [[thread_position_in_grid]])
{
    uint row = gid;
    if (row >= num_rows) return;

    // Compose M^row from precomputed powers
    Mat2x2_M31 mk = mat2_identity();
    uint k = row;
    uint log_n = min(log_num_rows, 24u);
    for (uint j = 0; j < log_n; j++) {
        if (k & (1u << j)) {
            uint base = j * 4;
            Mat2x2_M31 pj = Mat2x2_M31{
                M31{powers_buf[base]},
                M31{powers_buf[base + 1]},
                M31{powers_buf[base + 2]},
                M31{powers_buf[base + 3]}
            };
            mk = mat2_mul(mk, pj);
        }
    }

    M31 a0 = M31{a0_val};
    M31 b0 = M31{b0_val};
    M31 a_row = m31_add(m31_mul(mk.a, a0), m31_mul(mk.b, b0));
    M31 b_row = m31_add(m31_mul(mk.c, a0), m31_mul(mk.d, b0));

    col_a[row] = a_row.v;
    col_b[row] = b_row.v;
}

// =========================================================================
// Kernel 4: Generic linear recurrence via matrix power
// For any AIR with state transition matrix T (constant coefficients),
// row[i] = T^i * row[0]. Each thread computes T^(gid) * initial_state.
//
// State vector has `state_width` elements. Transfer matrix is state_width x state_width.
// =========================================================================

kernel void witness_m31_linear_recurrence(
    device uint* trace              [[buffer(0)]],  // n x state_width (row-major)
    device const uint* transfer_mat [[buffer(1)]],  // state_width x state_width matrix
    device const uint* initial_state [[buffer(2)]],  // state_width initial values
    constant uint& state_width      [[buffer(3)]],
    constant uint& num_rows         [[buffer(4)]],
    uint gid                        [[thread_position_in_grid]])
{
    uint row = gid;
    if (row >= num_rows) return;
    if (state_width > 8) return;  // safety limit

    // Load transfer matrix
    M31 T[8][8];
    for (uint i = 0; i < state_width; i++)
        for (uint j = 0; j < state_width; j++)
            T[i][j] = M31{transfer_mat[i * state_width + j]};

    // Compute T^row via repeated squaring
    M31 result[8][8];  // result matrix = identity
    M31 base_m[8][8];  // base = T

    // Initialize result = identity, base = T
    for (uint i = 0; i < state_width; i++)
        for (uint j = 0; j < state_width; j++) {
            result[i][j] = (i == j) ? m31_one() : m31_zero();
            base_m[i][j] = T[i][j];
        }

    uint k = row;
    while (k > 0) {
        if (k & 1) {
            // result = result * base
            M31 tmp[8][8];
            for (uint i = 0; i < state_width; i++)
                for (uint j = 0; j < state_width; j++) {
                    M31 acc = m31_zero();
                    for (uint l = 0; l < state_width; l++)
                        acc = m31_add(acc, m31_mul(result[i][l], base_m[l][j]));
                    tmp[i][j] = acc;
                }
            for (uint i = 0; i < state_width; i++)
                for (uint j = 0; j < state_width; j++)
                    result[i][j] = tmp[i][j];
        }
        // base = base * base
        M31 tmp[8][8];
        for (uint i = 0; i < state_width; i++)
            for (uint j = 0; j < state_width; j++) {
                M31 acc = m31_zero();
                for (uint l = 0; l < state_width; l++)
                    acc = m31_add(acc, m31_mul(base_m[i][l], base_m[l][j]));
                tmp[i][j] = acc;
            }
        for (uint i = 0; i < state_width; i++)
            for (uint j = 0; j < state_width; j++)
                base_m[i][j] = tmp[i][j];
        k >>= 1;
    }

    // Apply result matrix to initial state: trace[row] = T^row * state0
    M31 state0[8];
    for (uint i = 0; i < state_width; i++)
        state0[i] = M31{initial_state[i]};

    uint base = row * state_width;
    for (uint i = 0; i < state_width; i++) {
        M31 acc = m31_zero();
        for (uint j = 0; j < state_width; j++)
            acc = m31_add(acc, m31_mul(result[i][j], state0[j]));
        trace[base + i] = acc.v;
    }
}
