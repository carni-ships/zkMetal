// GPU Constraint Evaluation Engine — Plonk/AIR gate-aware constraint evaluation
//
// Evaluates constraint polynomials over an evaluation domain in parallel.
// Each thread processes one domain point independently (embarrassingly parallel).
//
// Supports a generic constraint encoding via an opcode-based instruction format:
//   - Addition gates: qL*a + qR*b + qO*c + qM*a*b + qC = 0
//   - Multiplication gates: a * b - c = 0
//   - Poseidon2 full/partial round gates: S-box + MDS + round constant
//   - Custom gates via selector-weighted expressions
//
// Buffer layout:
//   buffer(0): trace columns (column-major: col_i at offset i * domain_size)
//   buffer(1): output constraint evaluations (domain_size * num_constraints)
//   buffer(2): gate descriptors (packed GateDescriptor structs)
//   buffer(3): selector polynomials (column-major: sel_i at offset i * domain_size)
//   buffer(4): constants pool (Fr values referenced by gate descriptors)
//   buffer(5): params (num_cols, domain_size, num_gates, num_selectors, num_constants)

#include "../fields/bn254_fr.metal"

// Gate type opcodes
constant uint GATE_ARITHMETIC = 0u;   // qL*a + qR*b + qO*c + qM*a*b + qC = 0
constant uint GATE_MUL        = 1u;   // a * b - c = 0
constant uint GATE_BOOL       = 2u;   // a * (1 - a) = 0
constant uint GATE_POSEIDON2_FULL  = 3u;  // Full Poseidon2 round (all S-boxes)
constant uint GATE_POSEIDON2_PARTIAL = 4u; // Partial Poseidon2 round (one S-box)
constant uint GATE_ADD        = 5u;   // a + b - c = 0
constant uint GATE_RANGE_DECOMP = 6u; // sum(bit_i * 2^i) - value = 0

// Packed gate descriptor (32 bytes each)
// For arithmetic gates:
//   type=0, col_a, col_b, col_c, qL_idx, qR_idx, qO_idx, qM_idx, qC_idx
//   (indices into constants pool)
// For mul gates:
//   type=1, col_a, col_b, col_c, unused...
// For bool gates:
//   type=2, col_a, unused...
// For Poseidon2 gates:
//   type=3/4, width (in aux0), row_offset_in (in aux1), const_base_idx (in aux2)
//   Columns: 0..width-1 = state_in, width..2*width-1 = state_out
//   Constants: const_base_idx..+width = round constants, +width..+width*width+width = MDS
struct GateDescriptor {
    uint type;
    uint col_a;      // first wire column index
    uint col_b;      // second wire column index
    uint col_c;      // third wire column index
    uint aux0;       // depends on gate type
    uint aux1;       // depends on gate type
    uint aux2;       // depends on gate type
    uint sel_idx;    // selector column index (0xFFFFFFFF = no selector, always active)
};

struct ConstraintParams {
    uint num_cols;
    uint domain_size;
    uint num_gates;
    uint num_selectors;
    uint num_constants;
};

// S-box for Poseidon2: x^5 = x * x^2 * x^2
inline Fr poseidon2_sbox(Fr x) {
    Fr x2 = fr_sqr(x);
    Fr x4 = fr_sqr(x2);
    return fr_mul(x, x4);
}

// Load a trace value at (col, row) with column-major layout
inline Fr load_trace(device const Fr* trace, uint col, uint row, uint domain_size) {
    return trace[col * domain_size + row];
}

// Load a trace value at (col, row + offset) with wrapping
inline Fr load_trace_offset(device const Fr* trace, uint col, uint row,
                            int offset, uint domain_size) {
    uint r = uint((int(row) + offset + int(domain_size)) % int(domain_size));
    return trace[col * domain_size + r];
}

// Evaluate a single arithmetic gate: qL*a + qR*b + qO*c + qM*a*b + qC
inline Fr eval_arithmetic_gate(Fr a, Fr b, Fr c,
                               Fr qL, Fr qR, Fr qO, Fr qM, Fr qC) {
    // qL*a
    Fr result = fr_mul(qL, a);
    // + qR*b
    result = fr_add(result, fr_mul(qR, b));
    // + qO*c
    result = fr_add(result, fr_mul(qO, c));
    // + qM*a*b
    result = fr_add(result, fr_mul(qM, fr_mul(a, b)));
    // + qC
    result = fr_add(result, qC);
    return result;
}

// Main constraint evaluation kernel:
// Each thread evaluates ALL gates at a single domain point.
kernel void constraint_eval_kernel(
    device const Fr* trace              [[buffer(0)]],
    device Fr* output                   [[buffer(1)]],
    device const GateDescriptor* gates  [[buffer(2)]],
    device const Fr* selectors          [[buffer(3)]],
    device const Fr* constants          [[buffer(4)]],
    constant ConstraintParams& params   [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint row = gid;
    if (row >= params.domain_size) return;

    uint n = params.domain_size;
    uint ng = params.num_gates;

    for (uint gi = 0; gi < ng; gi++) {
        GateDescriptor gate = gates[gi];

        // Check selector (if present)
        Fr sel_val = fr_one();
        if (gate.sel_idx != 0xFFFFFFFF) {
            sel_val = selectors[gate.sel_idx * n + row];
        }

        Fr eval;

        switch (gate.type) {
            case GATE_ARITHMETIC: {
                Fr a = load_trace(trace, gate.col_a, row, n);
                Fr b = load_trace(trace, gate.col_b, row, n);
                Fr c = load_trace(trace, gate.col_c, row, n);
                Fr qL = constants[gate.aux0];
                Fr qR = constants[gate.aux0 + 1u];
                Fr qO = constants[gate.aux0 + 2u];
                Fr qM = constants[gate.aux0 + 3u];
                Fr qC = constants[gate.aux0 + 4u];
                eval = eval_arithmetic_gate(a, b, c, qL, qR, qO, qM, qC);
                break;
            }
            case GATE_MUL: {
                Fr a = load_trace(trace, gate.col_a, row, n);
                Fr b = load_trace(trace, gate.col_b, row, n);
                Fr c = load_trace(trace, gate.col_c, row, n);
                eval = fr_sub(fr_mul(a, b), c);
                break;
            }
            case GATE_BOOL: {
                Fr a = load_trace(trace, gate.col_a, row, n);
                eval = fr_mul(a, fr_sub(fr_one(), a));
                break;
            }
            case GATE_ADD: {
                Fr a = load_trace(trace, gate.col_a, row, n);
                Fr b = load_trace(trace, gate.col_b, row, n);
                Fr c = load_trace(trace, gate.col_c, row, n);
                eval = fr_sub(fr_add(a, b), c);
                break;
            }
            case GATE_POSEIDON2_FULL: {
                // Full Poseidon2 round: all elements through S-box
                uint width = gate.aux0;
                int row_off = int(gate.aux1);  // offset to next-row state
                uint cb = gate.aux2;            // constants base index

                // Accumulate constraint error across all state elements
                eval = fr_zero();
                for (uint i = 0; i < width && i < 8; i++) {
                    // state_in[i] at current row, col i
                    Fr s_in = load_trace(trace, gate.col_a + i, row, n);
                    // state_out[i] at next row
                    Fr s_out = load_trace_offset(trace, gate.col_a + i, row, row_off, n);
                    // Round constant
                    Fr rc = constants[cb + i];

                    // Apply S-box to (s_in + rc)
                    Fr temp_i = poseidon2_sbox(fr_add(s_in, rc));

                    // MDS: expected_out[i] = sum_j mds[i*width+j] * temp[j]
                    // We accumulate contribution of this element to the overall error
                    // Actually need full MDS row — compute expected for element i
                    Fr expected = fr_zero();
                    for (uint j = 0; j < width && j < 8; j++) {
                        Fr s_in_j = load_trace(trace, gate.col_a + j, row, n);
                        Fr rc_j = constants[cb + j];
                        Fr temp_j = poseidon2_sbox(fr_add(s_in_j, rc_j));
                        Fr mds_ij = constants[cb + width + i * width + j];
                        expected = fr_add(expected, fr_mul(mds_ij, temp_j));
                    }

                    // Constraint: s_out - expected = 0
                    Fr diff = fr_sub(s_out, expected);
                    eval = fr_add(eval, fr_sqr(diff));
                }
                break;
            }
            case GATE_POSEIDON2_PARTIAL: {
                // Partial Poseidon2 round: only first element through S-box
                uint width = gate.aux0;
                int row_off = int(gate.aux1);
                uint cb = gate.aux2;

                eval = fr_zero();
                for (uint i = 0; i < width && i < 8; i++) {
                    Fr s_in_i = load_trace(trace, gate.col_a + i, row, n);
                    Fr rc_i = constants[cb + i];
                    Fr with_rc = fr_add(s_in_i, rc_i);

                    // MDS expected
                    Fr expected = fr_zero();
                    for (uint j = 0; j < width && j < 8; j++) {
                        Fr s_in_j = load_trace(trace, gate.col_a + j, row, n);
                        Fr rc_j = constants[cb + j];
                        Fr temp_j;
                        if (j == 0) {
                            temp_j = poseidon2_sbox(fr_add(s_in_j, rc_j));
                        } else {
                            temp_j = fr_add(s_in_j, rc_j);
                        }
                        Fr mds_ij = constants[cb + width + i * width + j];
                        expected = fr_add(expected, fr_mul(mds_ij, temp_j));
                    }

                    Fr s_out_i = load_trace_offset(trace, gate.col_a + i, row, row_off, n);
                    Fr diff = fr_sub(s_out_i, expected);
                    eval = fr_add(eval, fr_sqr(diff));
                }
                break;
            }
            case GATE_RANGE_DECOMP: {
                // sum(bit_i * 2^i) - value = 0
                // col_a = value column, col_b = first bit column, aux0 = num_bits
                uint num_bits = gate.aux0;
                Fr value = load_trace(trace, gate.col_a, row, n);
                Fr sum = fr_zero();
                Fr power_of_2 = fr_one();
                Fr two = fr_add(fr_one(), fr_one());
                for (uint i = 0; i < num_bits && i < 256; i++) {
                    Fr bit = load_trace(trace, gate.col_b + i, row, n);
                    sum = fr_add(sum, fr_mul(power_of_2, bit));
                    power_of_2 = fr_mul(power_of_2, two);
                }
                eval = fr_sub(sum, value);
                break;
            }
            default: {
                eval = fr_zero();
                break;
            }
        }

        // Multiply by selector
        if (gate.sel_idx != 0xFFFFFFFF) {
            eval = fr_mul(eval, sel_val);
        }

        output[row * ng + gi] = eval;
    }
}

// Quotient polynomial computation kernel:
// quotient[row] = sum_i (alpha^i * constraint_evals[row * num_gates + i]) * vanishing_inv[row]
kernel void compute_quotient_kernel(
    device const Fr* constraint_evals   [[buffer(0)]],
    device Fr* quotient                 [[buffer(1)]],
    device const Fr* alpha_powers       [[buffer(2)]],
    device const Fr* vanishing_inv      [[buffer(3)]],
    constant uint& domain_size          [[buffer(4)]],
    constant uint& num_gates            [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint row = gid;
    if (row >= domain_size) return;

    Fr acc = fr_zero();
    for (uint i = 0; i < num_gates; i++) {
        Fr c_eval = constraint_evals[row * num_gates + i];
        Fr alpha_i = alpha_powers[i];
        acc = fr_add(acc, fr_mul(alpha_i, c_eval));
    }

    // Divide by vanishing polynomial
    quotient[row] = fr_mul(acc, vanishing_inv[row]);
}

// Fused constraint eval + quotient in one pass (avoids intermediate buffer)
kernel void fused_constraint_quotient_kernel(
    device const Fr* trace              [[buffer(0)]],
    device Fr* quotient                 [[buffer(1)]],
    device const GateDescriptor* gates  [[buffer(2)]],
    device const Fr* selectors          [[buffer(3)]],
    device const Fr* constants          [[buffer(4)]],
    device const Fr* alpha_powers       [[buffer(5)]],
    device const Fr* vanishing_inv      [[buffer(6)]],
    constant ConstraintParams& params   [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    uint row = gid;
    if (row >= params.domain_size) return;

    uint n = params.domain_size;
    uint ng = params.num_gates;
    Fr acc = fr_zero();

    for (uint gi = 0; gi < ng; gi++) {
        GateDescriptor gate = gates[gi];

        // Check selector
        Fr sel_val = fr_one();
        if (gate.sel_idx != 0xFFFFFFFF) {
            sel_val = selectors[gate.sel_idx * n + row];
        }

        Fr eval;

        switch (gate.type) {
            case GATE_ARITHMETIC: {
                Fr a = load_trace(trace, gate.col_a, row, n);
                Fr b = load_trace(trace, gate.col_b, row, n);
                Fr c = load_trace(trace, gate.col_c, row, n);
                Fr qL = constants[gate.aux0];
                Fr qR = constants[gate.aux0 + 1u];
                Fr qO = constants[gate.aux0 + 2u];
                Fr qM = constants[gate.aux0 + 3u];
                Fr qC = constants[gate.aux0 + 4u];
                eval = eval_arithmetic_gate(a, b, c, qL, qR, qO, qM, qC);
                break;
            }
            case GATE_MUL: {
                Fr a = load_trace(trace, gate.col_a, row, n);
                Fr b = load_trace(trace, gate.col_b, row, n);
                Fr c = load_trace(trace, gate.col_c, row, n);
                eval = fr_sub(fr_mul(a, b), c);
                break;
            }
            case GATE_BOOL: {
                Fr a = load_trace(trace, gate.col_a, row, n);
                eval = fr_mul(a, fr_sub(fr_one(), a));
                break;
            }
            case GATE_ADD: {
                Fr a = load_trace(trace, gate.col_a, row, n);
                Fr b = load_trace(trace, gate.col_b, row, n);
                Fr c = load_trace(trace, gate.col_c, row, n);
                eval = fr_sub(fr_add(a, b), c);
                break;
            }
            default: {
                eval = fr_zero();
                break;
            }
        }

        if (gate.sel_idx != 0xFFFFFFFF) {
            eval = fr_mul(eval, sel_val);
        }

        acc = fr_add(acc, fr_mul(alpha_powers[gi], eval));
    }

    quotient[row] = fr_mul(acc, vanishing_inv[row]);
}
