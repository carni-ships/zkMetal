// Witness trace evaluation kernel for GPU witness generation
// Each thread evaluates one row of the execution trace by executing
// a shared instruction stream. All threads execute the same program
// (no warp divergence), making this a perfect GPU workload.

#include <metal_stdlib>
using namespace metal;

// Include bn254_fr inline (same pattern as Poseidon2Engine)
// This file is combined with bn254_fr.metal at compile time.

// Opcode definitions (low 7 bits)
constant uint OP_ADD       = 0;  // dst = src1 + src2
constant uint OP_MUL       = 1;  // dst = src1 * src2
constant uint OP_COPY      = 2;  // dst = src1 (src2 ignored)
constant uint OP_SUB       = 3;  // dst = src1 - src2
constant uint OP_LOAD      = 4;  // dst = inputs[row * inputWidth + src1] (src2 ignored)
constant uint OP_SQR       = 5;  // dst = src1^2 (src2 ignored)
constant uint OP_DOUBLE    = 6;  // dst = 2 * src1 (src2 ignored)
constant uint OP_NEG       = 7;  // dst = -src1 (src2 ignored)
constant uint OP_SELECT    = 8;  // dst = (selector_col != 0) ? src1 : src2
constant uint OP_ASSERT_EQ = 9;  // no-op on GPU (for future constraint checking)

// High bit: if set, src2 is an index into constant_pool instead of a column index
constant uint FLAG_CONST   = 0x80;

// Each instruction is 4 uint32 words: [opcode, dst_col, src1, src2]
// For LOAD: src1 = input column index, src2 = unused
// For SELECT: dst = dest, src1 = true_col, src2 = false_col, selector in program[i*4+3] >> 16

kernel void trace_evaluate(
    device Fr* trace              [[buffer(0)]],  // n x num_cols matrix (row-major)
    device const Fr* inputs       [[buffer(1)]],  // n x input_width matrix
    constant uint* program        [[buffer(2)]],  // instruction stream (num_instrs * 4 words)
    device const Fr* const_pool   [[buffer(3)]],  // constant values pool
    constant uint& num_cols       [[buffer(4)]],
    constant uint& num_instrs     [[buffer(5)]],
    constant uint& input_width    [[buffer(6)]],
    constant uint& num_rows       [[buffer(7)]],
    uint gid                      [[thread_position_in_grid]])
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
        bool use_const = (op_raw & FLAG_CONST) != 0;

        Fr a, b;

        // Load operands based on opcode
        if (op == OP_LOAD) {
            // src1 is the input column index
            trace[base + dst] = inputs[row * input_width + src1];
            continue;
        }

        a = trace[base + src1];
        if (use_const) {
            b = const_pool[src2];
        } else {
            b = trace[base + src2];
        }

        Fr result;
        switch (op) {
            case OP_ADD:
                result = fr_add(a, b);
                break;
            case OP_MUL:
                result = fr_mul(a, b);
                break;
            case OP_COPY:
                result = a;
                break;
            case OP_SUB:
                result = fr_sub(a, b);
                break;
            case OP_SQR:
                result = fr_sqr(a);
                break;
            case OP_DOUBLE:
                result = fr_double(a);
                break;
            case OP_NEG:
                result = fr_sub(fr_zero(), a);
                break;
            case OP_SELECT: {
                // src2 low 16 bits = false_col, high 16 bits = selector_col
                uint false_col = src2 & 0xFFFF;
                uint sel_col = (src2 >> 16) & 0xFFFF;
                Fr selector = trace[base + sel_col];
                result = fr_is_zero(selector) ? trace[base + false_col] : a;
                break;
            }
            default:
                result = fr_zero();
                break;
        }

        trace[base + dst] = result;
    }
}

// Batch range decomposition: decompose values into bit columns
// Each thread handles one row. Decomposes trace[row][src_col] into bits
// written to trace[row][dst_col_start .. dst_col_start + num_bits - 1].
// The value is read from Montgomery form and bits are extracted from the
// actual value (after Montgomery reduction).
kernel void trace_range_decompose(
    device Fr* trace              [[buffer(0)]],
    constant uint& num_cols       [[buffer(1)]],
    constant uint& src_col        [[buffer(2)]],
    constant uint& dst_col_start  [[buffer(3)]],
    constant uint& num_bits       [[buffer(4)]],
    constant uint& num_rows       [[buffer(5)]],
    uint gid                      [[thread_position_in_grid]])
{
    uint row = gid;
    if (row >= num_rows) return;

    uint base = row * num_cols;
    Fr val = trace[base + src_col];

    // Convert from Montgomery form: multiply by 1
    Fr one_raw;
    for (int i = 0; i < 8; i++) one_raw.v[i] = 0;
    one_raw.v[0] = 1;
    Fr actual = fr_mul(val, one_raw);

    // Extract bits from the 256-bit value
    Fr mont_one = fr_one();
    for (uint b = 0; b < num_bits && b < 256; b++) {
        uint limb_idx = b / 32;
        uint bit_idx = b % 32;
        uint bit = (actual.v[limb_idx] >> bit_idx) & 1;
        // Store bit as Fr element in Montgomery form (0 or 1)
        trace[base + dst_col_start + b] = bit ? mont_one : fr_zero();
    }
}
