// Constraint IR — Intermediate representation for arithmetic constraint systems
// Compiles to Metal GPU kernels for parallel constraint evaluation.

import Foundation

/// A wire reference in the constraint system
public struct Wire: Hashable {
    public let index: Int
    public let row: Int  // 0 = current row, -1 = previous, +1 = next

    public init(index: Int, row: Int = 0) {
        self.index = index
        self.row = row
    }

    /// Shorthand for current-row wire
    public static func col(_ i: Int) -> Wire { Wire(index: i, row: 0) }
    /// Shorthand for next-row wire
    public static func next(_ i: Int) -> Wire { Wire(index: i, row: 1) }
    /// Shorthand for previous-row wire
    public static func prev(_ i: Int) -> Wire { Wire(index: i, row: -1) }
}

/// An arithmetic expression over wires
public indirect enum Expr: Hashable {
    case wire(Wire)
    case constant(Fr)
    case add(Expr, Expr)
    case mul(Expr, Expr)
    case neg(Expr)
    case pow(Expr, Int)  // small powers only (2-5)

    // Operator overloads for building expressions
    public static func + (lhs: Expr, rhs: Expr) -> Expr { .add(lhs, rhs) }
    public static func * (lhs: Expr, rhs: Expr) -> Expr { .mul(lhs, rhs) }
    public static func - (lhs: Expr, rhs: Expr) -> Expr { .add(lhs, .neg(rhs)) }
    public static prefix func - (expr: Expr) -> Expr { .neg(expr) }

    /// Count total nodes in this expression (for complexity estimates)
    public var nodeCount: Int {
        switch self {
        case .wire, .constant: return 1
        case .add(let a, let b), .mul(let a, let b): return 1 + a.nodeCount + b.nodeCount
        case .neg(let a): return 1 + a.nodeCount
        case .pow(let a, _): return 1 + a.nodeCount
        }
    }

    /// Constant-fold the expression: evaluate pure-constant subtrees at compile time.
    /// Reduces generated shader complexity by eliminating runtime arithmetic on constants.
    public func constantFolded() -> Expr {
        switch self {
        case .wire, .constant:
            return self

        case .add(let a, let b):
            let fa = a.constantFolded()
            let fb = b.constantFolded()
            // 0 + x = x, x + 0 = x
            if case .constant(let c) = fa, c.isZero { return fb }
            if case .constant(let c) = fb, c.isZero { return fa }
            // const + const = const
            if case .constant(let ca) = fa, case .constant(let cb) = fb {
                return .constant(frAdd(ca, cb))
            }
            return .add(fa, fb)

        case .mul(let a, let b):
            let fa = a.constantFolded()
            let fb = b.constantFolded()
            // 0 * x = 0, x * 0 = 0
            if case .constant(let c) = fa, c.isZero { return .constant(Fr.zero) }
            if case .constant(let c) = fb, c.isZero { return .constant(Fr.zero) }
            // 1 * x = x, x * 1 = x
            if case .constant(let c) = fa, c == Fr.one { return fb }
            if case .constant(let c) = fb, c == Fr.one { return fa }
            // const * const = const
            if case .constant(let ca) = fa, case .constant(let cb) = fb {
                return .constant(frMul(ca, cb))
            }
            return .mul(fa, fb)

        case .neg(let a):
            let fa = a.constantFolded()
            // neg(const) = const
            if case .constant(let c) = fa {
                return .constant(frSub(Fr.zero, c))
            }
            // neg(neg(x)) = x
            if case .neg(let inner) = fa {
                return inner
            }
            return .neg(fa)

        case .pow(let base, let n):
            let fb = base.constantFolded()
            // const^n = const
            if case .constant(let c) = fb {
                var result = c
                for _ in 1..<n {
                    result = frMul(result, c)
                }
                return .constant(result)
            }
            return .pow(fb, n)
        }
    }

    /// Collect all wire references used in this expression
    public var wires: Set<Wire> {
        switch self {
        case .wire(let w): return [w]
        case .constant: return []
        case .add(let a, let b), .mul(let a, let b): return a.wires.union(b.wires)
        case .neg(let a): return a.wires
        case .pow(let a, _): return a.wires
        }
    }
}

/// A constraint: expr == 0
public struct Constraint: Hashable {
    public let expr: Expr
    public let label: String?

    public init(expr: Expr, label: String? = nil) {
        self.expr = expr
        self.label = label
    }
}

/// A complete constraint system
public class ConstraintSystem {
    public let numWires: Int
    public private(set) var constraints: [Constraint] = []

    public init(numWires: Int) {
        precondition(numWires > 0, "Need at least one wire")
        self.numWires = numWires
    }

    /// Add a constraint: expr == 0
    public func addConstraint(_ expr: Expr, label: String? = nil) {
        constraints.append(Constraint(expr: expr, label: label))
    }

    /// Convenience: a == b  (i.e. a - b == 0)
    public func assertEqual(_ a: Expr, _ b: Expr, label: String? = nil) {
        addConstraint(a - b, label: label)
    }

    /// Convenience: w is boolean  (w * (1 - w) == 0)
    public func assertBool(_ w: Wire, label: String? = nil) {
        let e = Expr.wire(w)
        addConstraint(e * (.constant(Fr.one) - e), label: label ?? "bool(\(w.index))")
    }

    /// Convenience: a * b == c  (a*b - c == 0)
    public func assertMul(_ a: Wire, _ b: Wire, _ c: Wire, label: String? = nil) {
        addConstraint(.mul(.wire(a), .wire(b)) - .wire(c),
                       label: label ?? "mul(\(a.index)*\(b.index)=\(c.index))")
    }

    /// True if any constraint references non-zero row offsets
    public var hasCrossRowConstraints: Bool {
        for c in constraints {
            for w in c.expr.wires {
                if w.row != 0 { return true }
            }
        }
        return false
    }

    /// Total expression node count (complexity estimate)
    public var totalNodeCount: Int {
        constraints.reduce(0) { $0 + $1.expr.nodeCount }
    }

    /// Stable hash for caching compiled pipelines.
    /// Two systems with identical structure produce the same hash.
    public var stableHash: Int {
        var hasher = Hasher()
        hasher.combine(numWires)
        for c in constraints {
            hasher.combine(c)
        }
        return hasher.finalize()
    }
}

// MARK: - Common Constraint System Factories

extension ConstraintSystem {
    /// R1CS: for each gate i, wire[3*i] * wire[3*i+1] == wire[3*i+2]
    public static func r1cs(numGates: Int) -> ConstraintSystem {
        let cs = ConstraintSystem(numWires: numGates * 3)
        for i in 0..<numGates {
            cs.assertMul(Wire.col(3 * i), Wire.col(3 * i + 1), Wire.col(3 * i + 2),
                         label: "r1cs_gate_\(i)")
        }
        return cs
    }

    /// Fibonacci: w[0] + w[1] == w[2], w[1] + w[2] == w[3], ...
    /// Each row has `steps` wires, with constraints w[i] + w[i+1] == w[i+2]
    public static func fibonacci(steps: Int) -> ConstraintSystem {
        precondition(steps >= 3, "Need at least 3 wires for Fibonacci")
        let cs = ConstraintSystem(numWires: steps)
        for i in 0..<(steps - 2) {
            cs.assertEqual(
                .wire(Wire.col(i)) + .wire(Wire.col(i + 1)),
                .wire(Wire.col(i + 2)),
                label: "fib_\(i)")
        }
        return cs
    }

    /// Range check: wire `wire` decomposes into `bits` boolean wires starting at wire+1.
    /// Constraints: each bit is boolean, and sum(bit_i * 2^i) == wire.
    public static func rangeCheck(wire: Int, bits: Int) -> ConstraintSystem {
        let cs = ConstraintSystem(numWires: 1 + bits)
        // bit_i at wire index i+1 (shifting so wire 0 = value)
        // But we use `wire` as the value column
        // Actually: wire 0 = value, wires 1..bits = bit decomposition
        for i in 0..<bits {
            cs.assertBool(Wire.col(1 + i), label: "bit_\(i)")
        }
        // sum(bit_i * 2^i) == value
        var sum: Expr = .constant(Fr.zero)
        for i in 0..<bits {
            let coeff = frFromInt(UInt64(1) << UInt64(i))
            sum = sum + .mul(.constant(coeff), .wire(Wire.col(1 + i)))
        }
        cs.assertEqual(sum, .wire(Wire.col(0)), label: "range_decompose")
        return cs
    }
}
