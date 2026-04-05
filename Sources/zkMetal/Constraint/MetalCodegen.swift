// Metal Code Generator — compiles ConstraintIR into Metal shader source strings
// Uses runtime compilation (device.makeLibrary(source:options:)) like Poseidon2Engine.

import Foundation

/// Generates Metal shader source code from a constraint system.
public class MetalCodegen {

    public init() {}

    // MARK: - Constraint Evaluation Kernel

    /// Generate Metal source code for evaluating all constraints at each row.
    /// The kernel writes one Fr per constraint per row to the output buffer.
    /// output[row * numConstraints + constraintIdx] = constraint evaluation.
    /// A satisfied constraint evaluates to zero.
    public func generateConstraintEval(system: ConstraintSystem) -> String {
        let wires = collectWires(system: system)
        let (cseDecls, exprMap) = performCSE(system: system)
        var code = ""
        code += metalHeader()
        code += "\n"
        code += "kernel void eval_constraints(\n"
        code += "    device const Fr* trace [[buffer(0)]],\n"
        code += "    device Fr* output [[buffer(1)]],\n"
        code += "    constant uint& num_cols [[buffer(2)]],\n"
        code += "    constant uint& num_rows [[buffer(3)]],\n"
        code += "    uint gid [[thread_position_in_grid]]\n"
        code += ") {\n"
        code += "    uint row = gid;\n"
        code += "    if (row >= num_rows) return;\n"
        code += "\n"

        // Load wire values
        for w in wires.sorted(by: { ($0.row, $0.index) < ($1.row, $1.index) }) {
            let varName = wireVarName(w)
            if w.row == 0 {
                code += "    Fr \(varName) = trace[row * num_cols + \(w.index)];\n"
            } else if w.row > 0 {
                code += "    Fr \(varName) = ((row + \(w.row)) < num_rows) ? trace[(row + \(w.row)) * num_cols + \(w.index)] : fr_zero();\n"
            } else {
                let absRow = -w.row
                code += "    Fr \(varName) = (row >= \(absRow)) ? trace[(row - \(absRow)) * num_cols + \(w.index)] : fr_zero();\n"
            }
        }
        code += "\n"

        // CSE temporaries
        for decl in cseDecls {
            code += "    \(decl)\n"
        }
        if !cseDecls.isEmpty {
            code += "\n"
        }

        // Evaluate each constraint
        let numConstraints = system.constraints.count
        code += "    uint nc = \(numConstraints)u;\n"
        for (i, constraint) in system.constraints.enumerated() {
            let evalStr = exprToMetal(constraint.expr, exprMap: exprMap)
            if let label = constraint.label {
                code += "    // \(label)\n"
            }
            code += "    output[row * nc + \(i)u] = \(evalStr);\n"
        }

        code += "}\n"
        return code
    }

    // MARK: - Quotient Evaluation Kernel

    /// Generate Metal source for computing the quotient polynomial evaluation.
    /// quotient[row] = sum(alpha^i * C_i(row)) / Z_H(row)
    /// where Z_H is the vanishing polynomial for the trace domain.
    public func generateQuotientEval(system: ConstraintSystem) -> String {
        let wires = collectWires(system: system)
        let (cseDecls, exprMap) = performCSE(system: system)
        var code = ""
        code += metalHeader()
        code += "\n"
        code += "kernel void eval_quotient(\n"
        code += "    device const Fr* trace [[buffer(0)]],\n"
        code += "    device Fr* quotient [[buffer(1)]],\n"
        code += "    constant uint& num_cols [[buffer(2)]],\n"
        code += "    constant uint& num_rows [[buffer(3)]],\n"
        code += "    device const Fr* alpha_powers [[buffer(4)]],\n"
        code += "    device const Fr* vanishing_inv [[buffer(5)]],\n"
        code += "    uint gid [[thread_position_in_grid]]\n"
        code += ") {\n"
        code += "    uint row = gid;\n"
        code += "    if (row >= num_rows) return;\n"
        code += "\n"

        // Load wires
        for w in wires.sorted(by: { ($0.row, $0.index) < ($1.row, $1.index) }) {
            let varName = wireVarName(w)
            if w.row == 0 {
                code += "    Fr \(varName) = trace[row * num_cols + \(w.index)];\n"
            } else if w.row > 0 {
                code += "    Fr \(varName) = ((row + \(w.row)) < num_rows) ? trace[(row + \(w.row)) * num_cols + \(w.index)] : fr_zero();\n"
            } else {
                let absRow = -w.row
                code += "    Fr \(varName) = (row >= \(absRow)) ? trace[(row - \(absRow)) * num_cols + \(w.index)] : fr_zero();\n"
            }
        }
        code += "\n"

        // CSE temporaries
        for decl in cseDecls {
            code += "    \(decl)\n"
        }
        if !cseDecls.isEmpty {
            code += "\n"
        }

        // Accumulate alpha-weighted sum of constraint evaluations
        code += "    Fr acc = fr_zero();\n"
        for (i, constraint) in system.constraints.enumerated() {
            let evalStr = exprToMetal(constraint.expr, exprMap: exprMap)
            if let label = constraint.label {
                code += "    // \(label)\n"
            }
            code += "    acc = fr_add(acc, fr_mul(alpha_powers[\(i)], \(evalStr)));\n"
        }
        code += "\n"
        code += "    // Divide by vanishing polynomial\n"
        code += "    quotient[row] = fr_mul(acc, vanishing_inv[row]);\n"
        code += "}\n"
        return code
    }

    // MARK: - Fused NTT + Constraint Evaluation Kernel

    /// Generate Metal source for a fully fused NTT + constraint quotient kernel.
    /// The kernel performs NTT on all trace columns in threadgroup shared memory,
    /// then evaluates constraints directly on the NTT'd values — no global memory
    /// round-trip for NTT output. Each column is passed as a separate buffer.
    ///
    /// Threadgroup memory budget: numCols * blockSize * sizeof(Fr).
    /// For BN254 Fr (32 bytes) with 32KB threadgroup memory:
    ///   1 col: blockSize <= 1024 (logN <= 10)
    ///   2 cols: blockSize <= 512 (logN <= 9)
    ///   4 cols: blockSize <= 256 (logN <= 8)
    ///   8 cols: blockSize <= 128 (logN <= 7)
    ///
    /// For larger sizes, use the barrier-based approach (separate NTT + constraint dispatches
    /// in one command buffer).
    public func generateFusedNTTConstraintEval(system: ConstraintSystem, maxBlockSize: Int = 1024) -> String {
        let numCols = system.numWires
        let wires = collectWires(system: system)
        let (cseDecls, exprMap) = performCSE(system: system)

        // Determine max elements per column that fit in threadgroup memory
        // 32KB / (numCols * 32 bytes) = maxElements
        let frSize = 32 // sizeof(Fr) in bytes
        let threadgroupMemory = 32768 // 32KB
        let maxElements = min(maxBlockSize, threadgroupMemory / (numCols * frSize))

        var code = ""
        code += metalHeader()
        code += "\n\n"

        // Bit-reversal helper
        code += "inline uint bitrev_fused(uint val, uint num_bits) {\n"
        code += "    uint result = 0;\n"
        code += "    for (uint i = 0; i < num_bits; i++) {\n"
        code += "        result = (result << 1) | (val & 1);\n"
        code += "        val >>= 1;\n"
        code += "    }\n"
        code += "    return result;\n"
        code += "}\n\n"

        // Kernel signature: separate buffer per column + twiddles + output + params
        code += "kernel void fused_ntt_constraint_eval(\n"
        for i in 0..<numCols {
            code += "    device const Fr* col\(i) [[buffer(\(i))]],\n"
        }
        let twiddleBufIdx = numCols
        let quotientBufIdx = numCols + 1
        let alphaBufIdx = numCols + 2
        let vanishingBufIdx = numCols + 3
        let nBufIdx = numCols + 4
        let logNBufIdx = numCols + 5
        code += "    device const Fr* twiddles [[buffer(\(twiddleBufIdx))]],\n"
        code += "    device Fr* quotient_out [[buffer(\(quotientBufIdx))]],\n"
        code += "    device const Fr* alpha_powers [[buffer(\(alphaBufIdx))]],\n"
        code += "    device const Fr* vanishing_inv [[buffer(\(vanishingBufIdx))]],\n"
        code += "    constant uint& n [[buffer(\(nBufIdx))]],\n"
        code += "    constant uint& log_n [[buffer(\(logNBufIdx))]],\n"
        code += "    uint tid [[thread_index_in_threadgroup]],\n"
        code += "    uint tgid [[threadgroup_position_in_grid]],\n"
        code += "    uint tg_size [[threads_per_threadgroup]]\n"
        code += ") {\n"
        code += "    uint block_size = tg_size << 1;\n"
        code += "    uint base = tgid * block_size;\n\n"

        // Declare threadgroup shared memory arrays for each column
        code += "    // Shared memory for \(numCols) columns (max \(maxElements) elements each)\n"
        for i in 0..<numCols {
            code += "    threadgroup Fr shared_col\(i)[\(maxElements)];\n"
        }
        code += "\n"

        // Step 1: Load with bit-reversal
        code += "    // Step 1: Load all columns with bit-reversal into shared memory\n"
        code += "    uint local_logN = log_n;\n"
        code += "    for (uint k = tid; k < block_size; k += tg_size) {\n"
        code += "        uint global_idx = base + k;\n"
        code += "        if (global_idx < n) {\n"
        code += "            uint br = bitrev_fused(global_idx, local_logN);\n"
        for i in 0..<numCols {
            code += "            shared_col\(i)[k] = col\(i)[br];\n"
        }
        code += "        }\n"
        code += "    }\n"
        code += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n"

        // Step 2: NTT butterfly passes on all columns in shared memory
        code += "    // Step 2: NTT butterfly passes in shared memory (all \(numCols) columns)\n"
        code += "    for (uint s = 0; s < local_logN; s++) {\n"
        code += "        uint half_block = 1u << s;\n"
        code += "        uint local_block_size = half_block << 1;\n"
        code += "        for (uint k = tid; k < (block_size >> 1); k += tg_size) {\n"
        code += "            uint block_idx = k / half_block;\n"
        code += "            uint local_idx = k % half_block;\n"
        code += "            uint i = block_idx * local_block_size + local_idx;\n"
        code += "            uint j = i + half_block;\n"
        code += "            if (j < block_size) {\n"
        code += "                uint global_block_size = 1u << (s + 1);\n"
        code += "                uint twiddle_idx = local_idx * (n / global_block_size);\n"
        code += "                Fr w = twiddles[twiddle_idx];\n"
        for i in 0..<numCols {
            code += "                Fr c\(i)_i = shared_col\(i)[i];\n"
            code += "                Fr c\(i)_j = shared_col\(i)[j];\n"
            code += "                Fr wc\(i) = fr_mul(w, c\(i)_j);\n"
            code += "                shared_col\(i)[i] = fr_add(c\(i)_i, wc\(i));\n"
            code += "                shared_col\(i)[j] = fr_sub(c\(i)_i, wc\(i));\n"
        }
        code += "            }\n"
        code += "        }\n"
        code += "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        code += "    }\n\n"

        // Step 3: Evaluate constraints on NTT'd values from shared memory
        code += "    // Step 3: Evaluate constraints on NTT'd values (no global memory write of NTT output)\n"
        code += "    for (uint k = tid; k < block_size; k += tg_size) {\n"
        code += "        uint row = base + k;\n"
        code += "        if (row < n) {\n"

        // Load wire values from shared memory
        for w in wires.sorted(by: { ($0.row, $0.index) < ($1.row, $1.index) }) {
            let varName = wireVarName(w)
            if w.row == 0 {
                code += "            Fr \(varName) = shared_col\(w.index)[k];\n"
            } else if w.row > 0 {
                // next-row: circular wrap within threadgroup block
                code += "            uint idx_\(varName) = (k + \(w.row) < block_size) ? k + \(w.row) : k + \(w.row) - block_size;\n"
                code += "            Fr \(varName) = shared_col\(w.index)[idx_\(varName)];\n"
            } else {
                let absRow = -w.row
                code += "            uint idx_\(varName) = (k >= \(absRow)) ? k - \(absRow) : block_size - \(absRow) + k;\n"
                code += "            Fr \(varName) = shared_col\(w.index)[idx_\(varName)];\n"
            }
        }
        code += "\n"

        // CSE temporaries
        for decl in cseDecls {
            code += "            \(decl)\n"
        }
        if !cseDecls.isEmpty {
            code += "\n"
        }

        // Accumulate constraint quotient
        code += "            Fr acc = fr_zero();\n"
        for (i, constraint) in system.constraints.enumerated() {
            let evalStr = exprToMetal(constraint.expr, exprMap: exprMap)
            if let label = constraint.label {
                code += "            // \(label)\n"
            }
            code += "            acc = fr_add(acc, fr_mul(alpha_powers[\(i)], \(evalStr)));\n"
        }
        code += "\n"
        code += "            quotient_out[row] = fr_mul(acc, vanishing_inv[row]);\n"

        code += "        }\n"
        code += "    }\n"
        code += "}\n"

        return code
    }

    /// Maximum logN supported by the fused kernel for a given number of columns.
    /// Based on 32KB threadgroup memory and 32-byte Fr elements.
    public static func maxFusedLogN(numCols: Int) -> Int {
        let frSize = 32
        let threadgroupMemory = 32768
        let maxElements = threadgroupMemory / (numCols * frSize)
        var logN = 0
        while (1 << (logN + 1)) <= maxElements { logN += 1 }
        return logN
    }

    // MARK: - Separate-Column Quotient Evaluation Kernel

    /// Generate Metal source for constraint quotient evaluation that reads from
    /// separate column buffers (no interleaving required). Used as the constraint
    /// eval phase in the barrier-based approach for large NTT sizes.
    ///
    /// Buffer layout: col0 [[buffer(0)]], col1 [[buffer(1)]], ..., colN-1 [[buffer(N-1)]],
    ///   quotient [[buffer(N)]], alpha_powers [[buffer(N+1)]],
    ///   vanishing_inv [[buffer(N+2)]], num_rows [[buffer(N+3)]]
    public func generateSeparateColumnQuotientEval(system: ConstraintSystem) -> String {
        let numCols = system.numWires
        let wires = collectWires(system: system)
        let (cseDecls, exprMap) = performCSE(system: system)

        var code = ""
        code += metalHeader()
        code += "\n\n"

        // Kernel signature with separate buffer per column
        code += "kernel void eval_quotient_separate_cols(\n"
        for i in 0..<numCols {
            code += "    device const Fr* col\(i) [[buffer(\(i))]],\n"
        }
        code += "    device Fr* quotient [[buffer(\(numCols))]],\n"
        code += "    device const Fr* alpha_powers [[buffer(\(numCols + 1))]],\n"
        code += "    device const Fr* vanishing_inv [[buffer(\(numCols + 2))]],\n"
        code += "    constant uint& num_rows [[buffer(\(numCols + 3))]],\n"
        code += "    uint gid [[thread_position_in_grid]]\n"
        code += ") {\n"
        code += "    uint row = gid;\n"
        code += "    if (row >= num_rows) return;\n\n"

        // Load wire values from separate column buffers
        for w in wires.sorted(by: { ($0.row, $0.index) < ($1.row, $1.index) }) {
            let varName = wireVarName(w)
            if w.row == 0 {
                code += "    Fr \(varName) = col\(w.index)[row];\n"
            } else if w.row > 0 {
                code += "    Fr \(varName) = ((row + \(w.row)) < num_rows) ? col\(w.index)[row + \(w.row)] : fr_zero();\n"
            } else {
                let absRow = -w.row
                code += "    Fr \(varName) = (row >= \(absRow)) ? col\(w.index)[row - \(absRow)] : fr_zero();\n"
            }
        }
        code += "\n"

        // CSE temporaries
        for decl in cseDecls {
            code += "    \(decl)\n"
        }
        if !cseDecls.isEmpty {
            code += "\n"
        }

        // Accumulate constraint quotient
        code += "    Fr acc = fr_zero();\n"
        for (i, constraint) in system.constraints.enumerated() {
            let evalStr = exprToMetal(constraint.expr, exprMap: exprMap)
            if let label = constraint.label {
                code += "    // \(label)\n"
            }
            code += "    acc = fr_add(acc, fr_mul(alpha_powers[\(i)], \(evalStr)));\n"
        }
        code += "\n"
        code += "    quotient[row] = fr_mul(acc, vanishing_inv[row]);\n"
        code += "}\n"

        return code
    }

    // MARK: - Internal Helpers

    private func metalHeader() -> String {
        return """
        #include <metal_stdlib>
        using namespace metal;
        """
    }

    /// Collect all unique wire references from the system
    private func collectWires(system: ConstraintSystem) -> Set<Wire> {
        var wires = Set<Wire>()
        for c in system.constraints {
            wires.formUnion(c.expr.wires)
        }
        return wires
    }

    /// Variable name for a wire in generated Metal code
    private func wireVarName(_ w: Wire) -> String {
        if w.row == 0 {
            return "w\(w.index)"
        } else if w.row > 0 {
            return "w\(w.index)_next\(w.row)"
        } else {
            return "w\(w.index)_prev\(-w.row)"
        }
    }

    // MARK: - Common Subexpression Elimination

    /// Identify repeated subexpressions and assign them to temporaries.
    /// Returns (declarations, exprKey -> tmpVarName mapping).
    private func performCSE(system: ConstraintSystem) -> ([String], [String: String]) {
        // Count subexpression occurrences
        var counts: [String: Int] = [:]
        var exprs: [String: Expr] = [:]
        for c in system.constraints {
            countSubexprs(c.expr, counts: &counts, exprs: &exprs)
        }

        // Extract subexpressions used more than once (and non-trivial)
        var tmpMap: [String: String] = [:]
        var decls: [String] = []
        var tmpIdx = 0
        // Sort by key for determinism
        for key in counts.keys.sorted() where counts[key]! > 1 {
            let expr = exprs[key]!
            // Only CSE non-trivial expressions (not simple wires or constants)
            switch expr {
            case .wire, .constant: continue
            default: break
            }
            let name = "cse\(tmpIdx)"
            tmpMap[key] = name
            let metalExpr = exprToMetal(expr, exprMap: [:])  // no recursive CSE
            decls.append("Fr \(name) = \(metalExpr);")
            tmpIdx += 1
        }
        return (decls, tmpMap)
    }

    /// Canonical string key for an expression (for CSE dedup)
    private func exprKey(_ expr: Expr) -> String {
        switch expr {
        case .wire(let w):
            return "W(\(w.index),\(w.row))"
        case .constant(let fr):
            return "C(\(fr.v.0),\(fr.v.1),\(fr.v.2),\(fr.v.3),\(fr.v.4),\(fr.v.5),\(fr.v.6),\(fr.v.7))"
        case .add(let a, let b):
            return "ADD(\(exprKey(a)),\(exprKey(b)))"
        case .mul(let a, let b):
            return "MUL(\(exprKey(a)),\(exprKey(b)))"
        case .neg(let a):
            return "NEG(\(exprKey(a)))"
        case .pow(let a, let n):
            return "POW(\(exprKey(a)),\(n))"
        }
    }

    /// Count all subexpressions recursively
    private func countSubexprs(_ expr: Expr, counts: inout [String: Int], exprs: inout [String: Expr]) {
        let key = exprKey(expr)
        counts[key, default: 0] += 1
        exprs[key] = expr

        switch expr {
        case .wire, .constant: break
        case .add(let a, let b), .mul(let a, let b):
            countSubexprs(a, counts: &counts, exprs: &exprs)
            countSubexprs(b, counts: &counts, exprs: &exprs)
        case .neg(let a):
            countSubexprs(a, counts: &counts, exprs: &exprs)
        case .pow(let a, _):
            countSubexprs(a, counts: &counts, exprs: &exprs)
        }
    }

    /// Convert an Expr to Metal source code, using CSE map for shared subexpressions
    private func exprToMetal(_ expr: Expr, exprMap: [String: String]) -> String {
        let key = exprKey(expr)
        if let tmp = exprMap[key] { return tmp }

        switch expr {
        case .wire(let w):
            return wireVarName(w)

        case .constant(let fr):
            return frLiteral(fr)

        case .add(let a, let b):
            return "fr_add(\(exprToMetal(a, exprMap: exprMap)), \(exprToMetal(b, exprMap: exprMap)))"

        case .mul(let a, let b):
            return "fr_mul(\(exprToMetal(a, exprMap: exprMap)), \(exprToMetal(b, exprMap: exprMap)))"

        case .neg(let a):
            return "fr_sub(fr_zero(), \(exprToMetal(a, exprMap: exprMap)))"

        case .pow(let base, let n):
            precondition(n >= 2 && n <= 5, "pow exponent must be 2-5")
            // Expand small powers as repeated squaring
            let b = exprToMetal(base, exprMap: exprMap)
            switch n {
            case 2: return "fr_sqr(\(b))"
            case 3: return "fr_mul(fr_sqr(\(b)), \(b))"
            case 4: return "fr_sqr(fr_sqr(\(b)))"
            case 5: return "fr_mul(fr_sqr(fr_sqr(\(b))), \(b))"
            default: fatalError("unreachable")
            }
        }
    }

    /// Emit an Fr constant as a Metal struct literal
    private func frLiteral(_ fr: Fr) -> String {
        let v = fr.v
        return "Fr{{\(v.0)u, \(v.1)u, \(v.2)u, \(v.3)u, \(v.4)u, \(v.5)u, \(v.6)u, \(v.7)u}}"
    }
}
