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
