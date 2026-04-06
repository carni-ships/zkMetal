// GPUPlonkCustomGateEngine — GPU-accelerated Plonk custom gate evaluation engine
//
// Evaluates custom gates beyond standard arithmetic (addition/multiplication) gates
// across the entire execution trace using Metal GPU or CPU fallback. Handles:
//
//   1. Defining custom gate types via a registry (range check, EC add, Poseidon round,
//      boolean, XOR, conditional select, binary decomposition, etc.)
//   2. GPU-parallel evaluation of custom gate constraints across all trace rows
//   3. Combining gate selector polynomials with gate constraints
//   4. Computing the quotient polynomial contribution from custom gates
//   5. Linearization of custom gate terms for the verifier
//
// Custom gate constraint model:
//   Each registered gate type has:
//     - A unique identifier and gate-specific selector polynomial
//     - A constraint function: f(wires, rotations, challenges) = 0
//     - A linearization scalar: computed from wire evaluations at zeta
//
//   The combined quotient contribution is:
//     t_custom(x) = sum_i alpha^{offset+i} * q_i(x) * gate_i(x)
//
//   where q_i(x) is the selector polynomial and gate_i(x) is the constraint.
//
//   Linearization replaces opened wire polynomials with their evaluations:
//     r_custom(X) = sum_i alpha^{offset+i} * scalar_i * q_i(X)
//
// GPU acceleration:
//   - Metal compute kernels for batched constraint evaluation over all rows
//   - Parallel selector * constraint multiplication
//   - CPU fallback for small domains where Metal dispatch overhead dominates

import Foundation
import Metal
import NeonFieldOps

// MARK: - Custom Gate Type Identifier

/// Unique identifier for a custom gate type in the registry.
public struct CustomGateTypeID: Hashable, Sendable {
    public let rawValue: Int
    public let name: String

    public init(rawValue: Int, name: String) {
        self.rawValue = rawValue
        self.name = name
    }
}

// MARK: - Well-known Gate Type IDs

extension CustomGateTypeID {
    public static let rangeCheck = CustomGateTypeID(rawValue: 0, name: "RangeCheck")
    public static let boolCheck = CustomGateTypeID(rawValue: 1, name: "BoolCheck")
    public static let ecAdd = CustomGateTypeID(rawValue: 2, name: "ECAdd")
    public static let ecDouble = CustomGateTypeID(rawValue: 3, name: "ECDouble")
    public static let poseidonRound = CustomGateTypeID(rawValue: 4, name: "PoseidonRound")
    public static let poseidonSbox = CustomGateTypeID(rawValue: 5, name: "PoseidonSbox")
    public static let xorGate = CustomGateTypeID(rawValue: 6, name: "XOR")
    public static let conditionalSelect = CustomGateTypeID(rawValue: 7, name: "ConditionalSelect")
    public static let binaryDecompose = CustomGateTypeID(rawValue: 8, name: "BinaryDecompose")
    public static let andGate = CustomGateTypeID(rawValue: 9, name: "AND")
    public static let notGate = CustomGateTypeID(rawValue: 10, name: "NOT")
    public static let isZeroGate = CustomGateTypeID(rawValue: 11, name: "IsZero")
}

// MARK: - Custom Gate Constraint Evaluation

/// Protocol for evaluating a custom gate's constraint at a single row.
/// Implementations provide the constraint polynomial and linearization scalar.
public protocol CustomGateConstraint {
    /// Gate type identifier
    var typeID: CustomGateTypeID { get }

    /// Number of wire columns this gate reads
    var wireCount: Int { get }

    /// Column references with rotations (for Halo2-style gates)
    var queriedCells: [ColumnRef] { get }

    /// Evaluate the constraint at a single row given wire values.
    /// Returns zero if the constraint is satisfied.
    ///
    /// - Parameters:
    ///   - wires: Wire values at the current row [a, b, c, ...]
    ///   - rotatedWires: Wire values at rotated positions, keyed by ColumnRef
    ///   - challenges: Protocol challenges (alpha, beta, gamma, etc.)
    /// - Returns: Constraint residual (zero for valid witness)
    func evaluateConstraint(
        wires: [Fr],
        rotatedWires: [ColumnRef: Fr],
        challenges: [Fr]
    ) -> Fr

    /// Compute the linearization scalar from wire evaluations at the challenge point zeta.
    /// This is the scalar multiplier for the selector polynomial in the linearization.
    ///
    /// - Parameters:
    ///   - wireEvals: Wire polynomial evaluations at zeta [a(zeta), b(zeta), c(zeta), ...]
    ///   - wireEvalsShifted: Wire polynomial evaluations at zeta*omega
    ///   - challenges: Protocol challenges
    /// - Returns: The linearization scalar for this gate type
    func linearizationScalar(
        wireEvals: [Fr],
        wireEvalsShifted: [Fr],
        challenges: [Fr]
    ) -> Fr

    /// Degree of the constraint polynomial (for quotient degree estimation)
    var constraintDegree: Int { get }
}

// MARK: - Built-in Gate Constraints

/// Boolean check: a * (1 - a) = 0
public struct BoolCheckConstraint: CustomGateConstraint {
    public let typeID = CustomGateTypeID.boolCheck
    public let wireCount = 1
    public let constraintDegree = 2

    public init() {}

    public var queriedCells: [ColumnRef] {
        [ColumnRef(column: 0, rotation: .cur)]
    }

    public func evaluateConstraint(wires: [Fr], rotatedWires: [ColumnRef: Fr], challenges: [Fr]) -> Fr {
        let a = wires.count > 0 ? wires[0] : Fr.zero
        return frMul(a, frSub(Fr.one, a))
    }

    public func linearizationScalar(wireEvals: [Fr], wireEvalsShifted: [Fr], challenges: [Fr]) -> Fr {
        let a = wireEvals.count > 0 ? wireEvals[0] : Fr.zero
        return frMul(a, frSub(Fr.one, a))
    }
}

/// Range check (single bit): a * (1 - a) = 0 with accumulator chain
public struct RangeCheckConstraint: CustomGateConstraint {
    public let typeID = CustomGateTypeID.rangeCheck
    public let wireCount = 2
    public let constraintDegree = 2
    public let bits: Int

    public init(bits: Int) {
        self.bits = bits
    }

    public var queriedCells: [ColumnRef] {
        [
            ColumnRef(column: 0, rotation: .cur),  // bit value
            ColumnRef(column: 1, rotation: .cur),  // accumulator current
            ColumnRef(column: 1, rotation: .prev), // accumulator previous
        ]
    }

    public func evaluateConstraint(wires: [Fr], rotatedWires: [ColumnRef: Fr], challenges: [Fr]) -> Fr {
        let bit = wires.count > 0 ? wires[0] : Fr.zero
        // Boolean constraint on the bit
        let boolCheck = frMul(bit, frSub(Fr.one, bit))

        // Accumulator chain: acc_cur = acc_prev + 2^i * bit
        let accCur = rotatedWires[ColumnRef(column: 1, rotation: .cur)] ?? Fr.zero
        let accPrev = rotatedWires[ColumnRef(column: 1, rotation: .prev)] ?? Fr.zero
        let expected = frAdd(accPrev, frMul(wires.count > 0 ? wires[0] : Fr.zero, Fr.one))
        let accCheck = frSub(accCur, expected)

        // Combine: boolCheck + alpha * accCheck
        if challenges.count > 0 {
            return frAdd(boolCheck, frMul(challenges[0], accCheck))
        }
        return frAdd(frMul(boolCheck, boolCheck), frMul(accCheck, accCheck))
    }

    public func linearizationScalar(wireEvals: [Fr], wireEvalsShifted: [Fr], challenges: [Fr]) -> Fr {
        let a = wireEvals.count > 0 ? wireEvals[0] : Fr.zero
        return frMul(a, frSub(Fr.one, a))
    }
}

/// EC point addition: P3 = P1 + P2 on short Weierstrass curve
/// Wire layout: [x1, y1, x2, y2, x3, y3, lambda]
public struct ECAddConstraint: CustomGateConstraint {
    public let typeID = CustomGateTypeID.ecAdd
    public let wireCount = 7
    public let constraintDegree = 3

    public init() {}

    public var queriedCells: [ColumnRef] {
        [
            ColumnRef(column: 0, rotation: .cur),  // x1
            ColumnRef(column: 1, rotation: .cur),  // y1
            ColumnRef(column: 2, rotation: .cur),  // x2
            ColumnRef(column: 0, rotation: .next), // y2
            ColumnRef(column: 1, rotation: .next), // x3
            ColumnRef(column: 2, rotation: .next), // y3
            ColumnRef(column: 3, rotation: .cur),  // lambda
        ]
    }

    public func evaluateConstraint(wires: [Fr], rotatedWires: [ColumnRef: Fr], challenges: [Fr]) -> Fr {
        let x1 = rotatedWires[ColumnRef(column: 0, rotation: .cur)] ?? (wires.count > 0 ? wires[0] : Fr.zero)
        let y1 = rotatedWires[ColumnRef(column: 1, rotation: .cur)] ?? (wires.count > 1 ? wires[1] : Fr.zero)
        let x2 = rotatedWires[ColumnRef(column: 2, rotation: .cur)] ?? (wires.count > 2 ? wires[2] : Fr.zero)
        let y2 = rotatedWires[ColumnRef(column: 0, rotation: .next)] ?? (wires.count > 3 ? wires[3] : Fr.zero)
        let x3 = rotatedWires[ColumnRef(column: 1, rotation: .next)] ?? (wires.count > 4 ? wires[4] : Fr.zero)
        let y3 = rotatedWires[ColumnRef(column: 2, rotation: .next)] ?? (wires.count > 5 ? wires[5] : Fr.zero)
        let lam = rotatedWires[ColumnRef(column: 3, rotation: .cur)] ?? (wires.count > 6 ? wires[6] : Fr.zero)

        // Constraint 1: lambda * (x2 - x1) - (y2 - y1) = 0
        let c1 = frSub(frMul(lam, frSub(x2, x1)), frSub(y2, y1))

        // Constraint 2: x3 + x1 + x2 - lambda^2 = 0
        let lamSq = frSqr(lam)
        let c2 = frSub(frAdd(x3, frAdd(x1, x2)), lamSq)

        // Constraint 3: y3 + y1 - lambda*(x1 - x3) = 0
        let c3 = frAdd(frSub(y3, frMul(lam, frSub(x1, x3))), y1)

        // Sum of squares (zero iff all zero)
        return frAdd(frAdd(frMul(c1, c1), frMul(c2, c2)), frMul(c3, c3))
    }

    public func linearizationScalar(wireEvals: [Fr], wireEvalsShifted: [Fr], challenges: [Fr]) -> Fr {
        // For EC add, the linearization scalar is the constraint evaluated at zeta
        return evaluateConstraint(wires: wireEvals, rotatedWires: [:], challenges: challenges)
    }
}

/// EC point doubling: P2 = 2*P1 on short Weierstrass curve
/// Wire layout: [x1, y1, lambda, x2, y2]
public struct ECDoubleConstraint: CustomGateConstraint {
    public let typeID = CustomGateTypeID.ecDouble
    public let wireCount = 5
    public let constraintDegree = 3

    public init() {}

    public var queriedCells: [ColumnRef] {
        [
            ColumnRef(column: 0, rotation: .cur),  // x1
            ColumnRef(column: 1, rotation: .cur),  // y1
            ColumnRef(column: 2, rotation: .cur),  // lambda
            ColumnRef(column: 0, rotation: .next), // x2
            ColumnRef(column: 1, rotation: .next), // y2
        ]
    }

    public func evaluateConstraint(wires: [Fr], rotatedWires: [ColumnRef: Fr], challenges: [Fr]) -> Fr {
        let x1 = rotatedWires[ColumnRef(column: 0, rotation: .cur)] ?? (wires.count > 0 ? wires[0] : Fr.zero)
        let y1 = rotatedWires[ColumnRef(column: 1, rotation: .cur)] ?? (wires.count > 1 ? wires[1] : Fr.zero)
        let lam = rotatedWires[ColumnRef(column: 2, rotation: .cur)] ?? (wires.count > 2 ? wires[2] : Fr.zero)
        let x2 = rotatedWires[ColumnRef(column: 0, rotation: .next)] ?? (wires.count > 3 ? wires[3] : Fr.zero)
        let y2 = rotatedWires[ColumnRef(column: 1, rotation: .next)] ?? (wires.count > 4 ? wires[4] : Fr.zero)

        let three = frFromInt(3)
        let two = frFromInt(2)

        // Constraint 1: lambda * 2*y1 - 3*x1^2 = 0
        let x1Sq = frSqr(x1)
        let c1 = frSub(frMul(lam, frMul(two, y1)), frMul(three, x1Sq))

        // Constraint 2: x2 + 2*x1 - lambda^2 = 0
        let lamSq = frSqr(lam)
        let c2 = frSub(frAdd(x2, frMul(two, x1)), lamSq)

        // Constraint 3: y2 + y1 - lambda*(x1 - x2) = 0
        let c3 = frAdd(frSub(y2, frMul(lam, frSub(x1, x2))), y1)

        return frAdd(frAdd(frMul(c1, c1), frMul(c2, c2)), frMul(c3, c3))
    }

    public func linearizationScalar(wireEvals: [Fr], wireEvalsShifted: [Fr], challenges: [Fr]) -> Fr {
        return evaluateConstraint(wires: wireEvals, rotatedWires: [:], challenges: challenges)
    }
}

/// Poseidon S-box constraint: c = a^5
/// Wire layout: [a, b(=a^2), c(=a^5)]
public struct PoseidonSboxConstraint: CustomGateConstraint {
    public let typeID = CustomGateTypeID.poseidonSbox
    public let wireCount = 3
    public let constraintDegree = 3

    public init() {}

    public var queriedCells: [ColumnRef] {
        [
            ColumnRef(column: 0, rotation: .cur), // a (input)
            ColumnRef(column: 1, rotation: .cur), // b (a^2)
            ColumnRef(column: 2, rotation: .cur), // c (a^5)
        ]
    }

    public func evaluateConstraint(wires: [Fr], rotatedWires: [ColumnRef: Fr], challenges: [Fr]) -> Fr {
        let a = wires.count > 0 ? wires[0] : Fr.zero
        let b = wires.count > 1 ? wires[1] : Fr.zero
        let c = wires.count > 2 ? wires[2] : Fr.zero

        // Two sub-constraints:
        //   1. b - a^2 = 0  (b is a^2)
        //   2. c - a*b^2 = 0  (c is a * (a^2)^2 = a^5)
        let c1 = frSub(b, frSqr(a))
        let bSq = frSqr(b)
        let c2 = frSub(c, frMul(a, bSq))

        // Combined constraint
        if challenges.count > 0 {
            return frAdd(c1, frMul(challenges[0], c2))
        }
        return frAdd(c1, c2)
    }

    public func linearizationScalar(wireEvals: [Fr], wireEvalsShifted: [Fr], challenges: [Fr]) -> Fr {
        let a = wireEvals.count > 0 ? wireEvals[0] : Fr.zero
        let c = wireEvals.count > 2 ? wireEvals[2] : Fr.zero
        // Simplified: c - a^5
        let a2 = frSqr(a)
        let a4 = frSqr(a2)
        let a5 = frMul(a, a4)
        return frSub(c, a5)
    }
}

/// XOR gate: c = a XOR b (for binary values 0 or 1)
/// Constraint: c = a + b - 2*a*b (valid for boolean a, b)
/// Wire layout: [a, b, c]
public struct XORConstraint: CustomGateConstraint {
    public let typeID = CustomGateTypeID.xorGate
    public let wireCount = 3
    public let constraintDegree = 2

    public init() {}

    public var queriedCells: [ColumnRef] {
        [
            ColumnRef(column: 0, rotation: .cur),
            ColumnRef(column: 1, rotation: .cur),
            ColumnRef(column: 2, rotation: .cur),
        ]
    }

    public func evaluateConstraint(wires: [Fr], rotatedWires: [ColumnRef: Fr], challenges: [Fr]) -> Fr {
        let a = wires.count > 0 ? wires[0] : Fr.zero
        let b = wires.count > 1 ? wires[1] : Fr.zero
        let c = wires.count > 2 ? wires[2] : Fr.zero

        let two = frFromInt(2)
        // XOR: c = a + b - 2*a*b
        // Constraint: c - a - b + 2*a*b = 0
        let ab = frMul(a, b)
        let expected = frSub(frAdd(a, b), frMul(two, ab))
        return frSub(c, expected)
    }

    public func linearizationScalar(wireEvals: [Fr], wireEvalsShifted: [Fr], challenges: [Fr]) -> Fr {
        return evaluateConstraint(wires: wireEvals, rotatedWires: [:], challenges: challenges)
    }
}

/// AND gate: c = a AND b (for binary values)
/// Constraint: c = a * b
/// Wire layout: [a, b, c]
public struct ANDConstraint: CustomGateConstraint {
    public let typeID = CustomGateTypeID.andGate
    public let wireCount = 3
    public let constraintDegree = 2

    public init() {}

    public var queriedCells: [ColumnRef] {
        [
            ColumnRef(column: 0, rotation: .cur),
            ColumnRef(column: 1, rotation: .cur),
            ColumnRef(column: 2, rotation: .cur),
        ]
    }

    public func evaluateConstraint(wires: [Fr], rotatedWires: [ColumnRef: Fr], challenges: [Fr]) -> Fr {
        let a = wires.count > 0 ? wires[0] : Fr.zero
        let b = wires.count > 1 ? wires[1] : Fr.zero
        let c = wires.count > 2 ? wires[2] : Fr.zero
        // c - a*b = 0
        return frSub(c, frMul(a, b))
    }

    public func linearizationScalar(wireEvals: [Fr], wireEvalsShifted: [Fr], challenges: [Fr]) -> Fr {
        return evaluateConstraint(wires: wireEvals, rotatedWires: [:], challenges: challenges)
    }
}

/// NOT gate: c = 1 - a (for binary values)
/// Constraint: c + a - 1 = 0
/// Wire layout: [a, _, c]
public struct NOTConstraint: CustomGateConstraint {
    public let typeID = CustomGateTypeID.notGate
    public let wireCount = 3
    public let constraintDegree = 1

    public init() {}

    public var queriedCells: [ColumnRef] {
        [
            ColumnRef(column: 0, rotation: .cur),
            ColumnRef(column: 2, rotation: .cur),
        ]
    }

    public func evaluateConstraint(wires: [Fr], rotatedWires: [ColumnRef: Fr], challenges: [Fr]) -> Fr {
        let a = wires.count > 0 ? wires[0] : Fr.zero
        let c = wires.count > 2 ? wires[2] : Fr.zero
        // c - (1 - a) = 0 => c + a - 1 = 0
        return frSub(frAdd(c, a), Fr.one)
    }

    public func linearizationScalar(wireEvals: [Fr], wireEvalsShifted: [Fr], challenges: [Fr]) -> Fr {
        return evaluateConstraint(wires: wireEvals, rotatedWires: [:], challenges: challenges)
    }
}

/// Conditional select: out = cond ? a : b
/// Constraint: out - cond*a - (1-cond)*b = 0, equivalently: out - b - cond*(a-b) = 0
/// Wire layout: [cond, a, b] at cur row, [out] at next row col 0
public struct ConditionalSelectConstraint: CustomGateConstraint {
    public let typeID = CustomGateTypeID.conditionalSelect
    public let wireCount = 4
    public let constraintDegree = 2

    public init() {}

    public var queriedCells: [ColumnRef] {
        [
            ColumnRef(column: 0, rotation: .cur),  // cond
            ColumnRef(column: 1, rotation: .cur),  // a
            ColumnRef(column: 2, rotation: .cur),  // b
            ColumnRef(column: 0, rotation: .next), // out
        ]
    }

    public func evaluateConstraint(wires: [Fr], rotatedWires: [ColumnRef: Fr], challenges: [Fr]) -> Fr {
        let cond = rotatedWires[ColumnRef(column: 0, rotation: .cur)] ?? (wires.count > 0 ? wires[0] : Fr.zero)
        let a = rotatedWires[ColumnRef(column: 1, rotation: .cur)] ?? (wires.count > 1 ? wires[1] : Fr.zero)
        let b = rotatedWires[ColumnRef(column: 2, rotation: .cur)] ?? (wires.count > 2 ? wires[2] : Fr.zero)
        let out = rotatedWires[ColumnRef(column: 0, rotation: .next)] ?? (wires.count > 3 ? wires[3] : Fr.zero)

        // out - b - cond*(a - b) = 0
        let diff = frSub(a, b)
        return frSub(frSub(out, b), frMul(cond, diff))
    }

    public func linearizationScalar(wireEvals: [Fr], wireEvalsShifted: [Fr], challenges: [Fr]) -> Fr {
        let cond = wireEvals.count > 0 ? wireEvals[0] : Fr.zero
        let a = wireEvals.count > 1 ? wireEvals[1] : Fr.zero
        let b = wireEvals.count > 2 ? wireEvals[2] : Fr.zero
        let out = wireEvalsShifted.count > 0 ? wireEvalsShifted[0] : Fr.zero
        let diff = frSub(a, b)
        return frSub(frSub(out, b), frMul(cond, diff))
    }
}

/// IsZero gate: out = (a == 0) ? 1 : 0
/// Uses auxiliary inverse witness: inv = a^{-1} if a != 0, else 0
/// Constraints:
///   1. a * out = 0  (if out=1 then a=0; if a!=0 then out=0)
///   2. a * inv + out - 1 = 0  (either a has inverse or out=1)
/// Wire layout: [a, inv(aux), out]
public struct IsZeroConstraint: CustomGateConstraint {
    public let typeID = CustomGateTypeID.isZeroGate
    public let wireCount = 3
    public let constraintDegree = 2

    public init() {}

    public var queriedCells: [ColumnRef] {
        [
            ColumnRef(column: 0, rotation: .cur),
            ColumnRef(column: 1, rotation: .cur),
            ColumnRef(column: 2, rotation: .cur),
        ]
    }

    public func evaluateConstraint(wires: [Fr], rotatedWires: [ColumnRef: Fr], challenges: [Fr]) -> Fr {
        let a = wires.count > 0 ? wires[0] : Fr.zero
        let inv = wires.count > 1 ? wires[1] : Fr.zero
        let out = wires.count > 2 ? wires[2] : Fr.zero

        // c1: a * out = 0
        let c1 = frMul(a, out)
        // c2: a * inv + out - 1 = 0
        let c2 = frSub(frAdd(frMul(a, inv), out), Fr.one)

        if challenges.count > 0 {
            return frAdd(c1, frMul(challenges[0], c2))
        }
        return frAdd(c1, c2)
    }

    public func linearizationScalar(wireEvals: [Fr], wireEvalsShifted: [Fr], challenges: [Fr]) -> Fr {
        return evaluateConstraint(wires: wireEvals, rotatedWires: [:], challenges: challenges)
    }
}

// MARK: - Registered Gate Entry

/// A gate type registered in the custom gate registry, with its selector polynomial.
public struct RegisteredGateEntry {
    /// The gate constraint implementation
    public let constraint: CustomGateConstraint
    /// Selector polynomial in evaluation form (one element per trace row)
    public let selectorEvals: [Fr]
    /// Selector polynomial in coefficient form (lazily computed)
    public var selectorCoeffs: [Fr]?
    /// Rows where this gate is active (selector != 0)
    public let activeRows: [Int]

    public init(constraint: CustomGateConstraint, selectorEvals: [Fr], selectorCoeffs: [Fr]? = nil) {
        self.constraint = constraint
        self.selectorEvals = selectorEvals
        self.selectorCoeffs = selectorCoeffs
        // Precompute active rows for sparse evaluation
        var rows = [Int]()
        for i in 0..<selectorEvals.count {
            if !selectorEvals[i].isZero {
                rows.append(i)
            }
        }
        self.activeRows = rows
    }
}

// MARK: - Custom Gate Registry

/// Registry of custom gate types and their selector polynomials.
/// The registry manages gate registration, deduplication, and efficient lookup.
public class CustomGateRegistry {
    /// All registered gate entries, ordered by registration time
    public private(set) var entries: [RegisteredGateEntry] = []

    /// Fast lookup by gate type ID
    private var indexByType: [CustomGateTypeID: Int] = [:]

    /// Domain size (number of trace rows)
    public let domainSize: Int

    public init(domainSize: Int) {
        self.domainSize = domainSize
    }

    /// Register a new custom gate type with its selector polynomial.
    /// The selector polynomial must have exactly domainSize evaluations.
    /// Returns the index of the registered entry.
    @discardableResult
    public func register(
        constraint: CustomGateConstraint,
        selectorEvals: [Fr]
    ) -> Int {
        precondition(selectorEvals.count == domainSize,
                     "Selector must have \(domainSize) evaluations, got \(selectorEvals.count)")
        let entry = RegisteredGateEntry(constraint: constraint, selectorEvals: selectorEvals)
        let idx = entries.count
        entries.append(entry)
        indexByType[constraint.typeID] = idx
        return idx
    }

    /// Register with coefficient-form selector.
    @discardableResult
    public func registerWithCoeffs(
        constraint: CustomGateConstraint,
        selectorCoeffs: [Fr],
        ntt: NTTEngine
    ) throws -> Int {
        var coeffsPadded = selectorCoeffs
        if coeffsPadded.count < domainSize {
            coeffsPadded += [Fr](repeating: Fr.zero, count: domainSize - coeffsPadded.count)
        }
        let evals = try ntt.ntt(Array(coeffsPadded.prefix(domainSize)))
        let entry = RegisteredGateEntry(
            constraint: constraint,
            selectorEvals: evals,
            selectorCoeffs: selectorCoeffs
        )
        let idx = entries.count
        entries.append(entry)
        indexByType[constraint.typeID] = idx
        return idx
    }

    /// Look up a registered gate by type ID.
    public func entry(for typeID: CustomGateTypeID) -> RegisteredGateEntry? {
        guard let idx = indexByType[typeID] else { return nil }
        return entries[idx]
    }

    /// Number of registered gate types.
    public var count: Int { entries.count }

    /// Compute total constraint degree across all registered gates.
    public var maxConstraintDegree: Int {
        entries.map { $0.constraint.constraintDegree }.max() ?? 0
    }

    /// Set of all column references across all registered gates (for wire polynomial setup).
    public var allQueriedCells: Set<ColumnRef> {
        var cells = Set<ColumnRef>()
        for entry in entries {
            for cell in entry.constraint.queriedCells {
                cells.insert(cell)
            }
        }
        return cells
    }

    /// Maximum rotation offset (absolute value) across all registered gates.
    public var maxRotation: Int {
        entries.flatMap { $0.constraint.queriedCells.map { abs($0.rotation.value) } }.max() ?? 0
    }

    /// Ensure selector polynomial coefficients are computed for all entries.
    public func ensureCoeffs(ntt: NTTEngine) throws {
        for i in 0..<entries.count {
            if entries[i].selectorCoeffs == nil {
                entries[i].selectorCoeffs = try ntt.intt(entries[i].selectorEvals)
            }
        }
    }
}

// MARK: - Batch Evaluation Result

/// Result of batch-evaluating all custom gates across the trace.
public struct CustomGateBatchResult {
    /// Per-row combined constraint residuals
    public let residuals: [Fr]
    /// Per-gate-type residual arrays (for debugging individual gate types)
    public let perGateResiduals: [[Fr]]
    /// Rows where at least one constraint is violated
    public let failingRows: [Int]
    /// Per-gate-type failing row indices
    public let perGateFailingRows: [[Int]]
    /// Whether all custom gate constraints are satisfied
    public var isSatisfied: Bool { failingRows.isEmpty }

    public init(residuals: [Fr], perGateResiduals: [[Fr]],
                failingRows: [Int], perGateFailingRows: [[Int]]) {
        self.residuals = residuals
        self.perGateResiduals = perGateResiduals
        self.failingRows = failingRows
        self.perGateFailingRows = perGateFailingRows
    }
}

// MARK: - Linearization Result

/// Result of computing custom gate linearization terms.
public struct CustomGateLinearizationResult {
    /// Linearization scalars for each registered gate type
    public let scalars: [Fr]
    /// Combined linearization evaluation: sum_i alpha^{offset+i} * scalar_i * selector_i(zeta)
    public let combinedEval: Fr
    /// Whether GPU was used for computation
    public let usedGPU: Bool

    public init(scalars: [Fr], combinedEval: Fr, usedGPU: Bool) {
        self.scalars = scalars
        self.combinedEval = combinedEval
        self.usedGPU = usedGPU
    }
}

// MARK: - GPUPlonkCustomGateEngine

/// GPU-accelerated engine for evaluating Plonk custom gate constraints.
///
/// The engine maintains a CustomGateRegistry and provides:
///   - Batch constraint evaluation across the full trace (GPU or CPU)
///   - Quotient polynomial contribution computation
///   - Linearization scalar computation for the verifier
///   - Selector isolation verification
public class GPUPlonkCustomGateEngine {
    public static let version = PrimitiveVersion(version: "1.0.0", updated: "2026-04-06")

    /// Minimum domain size to dispatch to GPU (below this, CPU is faster).
    private static let gpuThreshold = 1024

    private let device: MTLDevice?
    private let commandQueue: MTLCommandQueue?
    private let boolCheckPipeline: MTLComputePipelineState?
    private let xorPipeline: MTLComputePipelineState?
    private let sboxPipeline: MTLComputePipelineState?
    private let combinePipeline: MTLComputePipelineState?
    private let threadgroupSize: Int

    /// The gate registry (one per engine instance, or shared)
    public var registry: CustomGateRegistry?

    // MARK: - Initialization

    public init() {
        let dev = MTLCreateSystemDefaultDevice()
        self.device = dev
        self.commandQueue = dev?.makeCommandQueue()
        self.threadgroupSize = 256

        if let dev = dev {
            self.boolCheckPipeline = GPUPlonkCustomGateEngine.compileBoolCheckKernel(device: dev)
            self.xorPipeline = GPUPlonkCustomGateEngine.compileXORKernel(device: dev)
            self.sboxPipeline = GPUPlonkCustomGateEngine.compileSboxKernel(device: dev)
            self.combinePipeline = GPUPlonkCustomGateEngine.compileCombineKernel(device: dev)
        } else {
            self.boolCheckPipeline = nil
            self.xorPipeline = nil
            self.sboxPipeline = nil
            self.combinePipeline = nil
        }
    }

    // MARK: - Batch Constraint Evaluation

    /// Evaluate all registered custom gate constraints across the entire trace.
    ///
    /// For each row i and each gate type g with active selector at row i:
    ///   residual_g[i] = constraint_g(wires[i], rotatedWires[i])
    ///
    /// Combined residual per row: sum_g selector_g[i] * residual_g[i]
    ///
    /// - Parameters:
    ///   - registry: Gate registry with registered gate types and selectors
    ///   - wireColumns: Wire values organized by column: wireColumns[col][row]
    ///   - challenges: Protocol challenges available to gate constraints
    /// - Returns: CustomGateBatchResult with per-row and per-gate residuals
    public func evaluateAllConstraints(
        registry: CustomGateRegistry,
        wireColumns: [[Fr]],
        challenges: [Fr] = []
    ) -> CustomGateBatchResult {
        let n = registry.domainSize
        if n == 0 || registry.count == 0 {
            return CustomGateBatchResult(
                residuals: [], perGateResiduals: [], failingRows: [], perGateFailingRows: [])
        }

        var combinedResiduals = [Fr](repeating: Fr.zero, count: n)
        var perGateResiduals = [[Fr]]()
        var perGateFailingRows = [[Int]]()

        // Evaluate each gate type
        for entry in registry.entries {
            var gateResiduals = [Fr](repeating: Fr.zero, count: n)
            var gateFailing = [Int]()

            // Sparse evaluation: only process active rows
            for row in entry.activeRows {
                // Build wire values for this row
                var wires = [Fr]()
                for col in 0..<wireColumns.count {
                    wires.append(row < wireColumns[col].count ? wireColumns[col][row] : Fr.zero)
                }

                // Build rotated wire map
                var rotatedWires = [ColumnRef: Fr]()
                for ref in entry.constraint.queriedCells {
                    let rotatedRow = ((row + ref.rotation.value) % n + n) % n
                    if ref.column < wireColumns.count && rotatedRow < wireColumns[ref.column].count {
                        rotatedWires[ref] = wireColumns[ref.column][rotatedRow]
                    } else {
                        rotatedWires[ref] = Fr.zero
                    }
                }

                let residual = entry.constraint.evaluateConstraint(
                    wires: wires, rotatedWires: rotatedWires, challenges: challenges)

                // Weight by selector value
                let weighted = frMul(entry.selectorEvals[row], residual)
                gateResiduals[row] = weighted

                if !weighted.isZero {
                    gateFailing.append(row)
                }

                // Accumulate into combined
                combinedResiduals[row] = frAdd(combinedResiduals[row], weighted)
            }

            perGateResiduals.append(gateResiduals)
            perGateFailingRows.append(gateFailing)
        }

        // Compute combined failing rows
        var failingRows = [Int]()
        for i in 0..<n {
            if !combinedResiduals[i].isZero {
                failingRows.append(i)
            }
        }

        return CustomGateBatchResult(
            residuals: combinedResiduals,
            perGateResiduals: perGateResiduals,
            failingRows: failingRows,
            perGateFailingRows: perGateFailingRows
        )
    }

    // MARK: - Quotient Polynomial Contribution

    /// Compute the quotient polynomial contribution from all registered custom gates.
    ///
    /// For each gate type i with selector q_i(x) and constraint gate_i(x):
    ///   contribution_i(x) = q_i(x) * gate_i(x)
    ///
    /// Combined contribution with alpha separation:
    ///   t_custom(x) = sum_i alpha^{offset+i} * q_i(x) * gate_i(x)
    ///
    /// The computation proceeds:
    ///   1. Evaluate constraint polynomials on the domain (from wire polynomial evaluations)
    ///   2. Multiply pointwise by selector evaluations
    ///   3. Combine with powers of alpha
    ///   4. Convert to coefficient form via iNTT
    ///
    /// - Parameters:
    ///   - registry: Gate registry
    ///   - wireColumns: Wire column evaluations on the domain (wireColumns[col][row])
    ///   - alpha: Separation challenge
    ///   - alphaOffset: Starting power of alpha (after standard gate/perm constraints)
    ///   - ntt: NTT engine for polynomial transforms
    ///   - challenges: Additional protocol challenges
    /// - Returns: Quotient contribution polynomial in coefficient form
    public func computeQuotientContribution(
        registry: CustomGateRegistry,
        wireColumns: [[Fr]],
        alpha: Fr,
        alphaOffset: Int,
        ntt: NTTEngine,
        challenges: [Fr] = []
    ) throws -> [Fr] {
        let n = registry.domainSize
        guard n > 0 && (n & (n - 1)) == 0 && registry.count > 0 else {
            return [Fr](repeating: Fr.zero, count: max(n, 1))
        }

        // Compute alpha^offset
        var alphaPow = Fr.one
        for _ in 0..<alphaOffset {
            alphaPow = frMul(alphaPow, alpha)
        }

        // Accumulate contributions in evaluation form
        var combined = [Fr](repeating: Fr.zero, count: n)

        for entry in registry.entries {
            // Evaluate constraint at each active row
            var constraintEvals = [Fr](repeating: Fr.zero, count: n)

            for row in entry.activeRows {
                var wires = [Fr]()
                for col in 0..<wireColumns.count {
                    wires.append(row < wireColumns[col].count ? wireColumns[col][row] : Fr.zero)
                }

                var rotatedWires = [ColumnRef: Fr]()
                for ref in entry.constraint.queriedCells {
                    let rotatedRow = ((row + ref.rotation.value) % n + n) % n
                    if ref.column < wireColumns.count && rotatedRow < wireColumns[ref.column].count {
                        rotatedWires[ref] = wireColumns[ref.column][rotatedRow]
                    } else {
                        rotatedWires[ref] = Fr.zero
                    }
                }

                let val = entry.constraint.evaluateConstraint(
                    wires: wires, rotatedWires: rotatedWires, challenges: challenges)

                // selector * constraint
                constraintEvals[row] = frMul(entry.selectorEvals[row], val)
            }

            // Accumulate with alpha power: combined += alpha^k * constraintEvals
            for i in 0..<n {
                if !constraintEvals[i].isZero {
                    combined[i] = frAdd(combined[i], frMul(alphaPow, constraintEvals[i]))
                }
            }

            alphaPow = frMul(alphaPow, alpha)
        }

        // Convert to coefficient form
        return try ntt.intt(combined)
    }

    // MARK: - Quotient with Wire Polynomials (Coefficient Form)

    /// Compute quotient contribution using wire polynomials in coefficient form.
    /// This path uses NTT-based polynomial multiplication for higher accuracy.
    ///
    /// - Parameters:
    ///   - registry: Gate registry (must have selector coefficients precomputed)
    ///   - wirePolyCoeffs: Wire polynomials in coefficient form [col][coeff_idx]
    ///   - alpha: Separation challenge
    ///   - alphaOffset: Starting power of alpha
    ///   - ntt: NTT engine
    /// - Returns: Quotient contribution in coefficient form
    public func computeQuotientFromCoeffs(
        registry: CustomGateRegistry,
        wirePolyCoeffs: [[Fr]],
        alpha: Fr,
        alphaOffset: Int,
        ntt: NTTEngine
    ) throws -> [Fr] {
        let n = registry.domainSize
        guard n > 0 && registry.count > 0 else {
            return [Fr](repeating: Fr.zero, count: max(n, 1))
        }

        // Ensure all selector coefficients are computed
        try registry.ensureCoeffs(ntt: ntt)

        // Build CustomGateSet from registry entries for polynomial-level computation
        let gateSet = CustomGateSet()
        for entry in registry.entries {
            guard let coeffs = entry.selectorCoeffs else { continue }

            // Wrap constraint as a CustomGate for polynomial computation
            let wrapper = ConstraintToCustomGateAdapter(constraint: entry.constraint)
            gateSet.addGate(wrapper, selectorCoeffs: coeffs)
        }

        // Use existing efficient polynomial multiplication
        let domain = try buildDomain(n: n, ntt: ntt)
        let omega = computeOmega(n: n)

        return try gateSet.combinedQuotientContribution(
            wirePolys: wirePolyCoeffs,
            domain: domain,
            omega: omega,
            n: n,
            alpha: alpha,
            alphaOffset: alphaOffset,
            ntt: ntt
        )
    }

    // MARK: - Linearization

    /// Compute linearization scalars and combined evaluation for all custom gates.
    ///
    /// For the verifier, the custom gate linearization is:
    ///   r_custom(X) = sum_i alpha^{offset+i} * scalar_i * q_i(X)
    ///
    /// where scalar_i = gate_i(wireEvals, wireEvalsShifted)
    ///
    /// - Parameters:
    ///   - registry: Gate registry
    ///   - wireEvals: Wire polynomial evaluations at zeta
    ///   - wireEvalsShifted: Wire polynomial evaluations at zeta*omega
    ///   - selectorEvals: Selector polynomial evaluations at zeta (one per gate type)
    ///   - alpha: Separation challenge
    ///   - alphaOffset: Starting power of alpha
    ///   - challenges: Additional protocol challenges
    /// - Returns: Linearization result with scalars and combined evaluation
    public func computeLinearization(
        registry: CustomGateRegistry,
        wireEvals: [Fr],
        wireEvalsShifted: [Fr],
        selectorEvals: [Fr],
        alpha: Fr,
        alphaOffset: Int,
        challenges: [Fr] = []
    ) -> CustomGateLinearizationResult {
        guard registry.count > 0 else {
            return CustomGateLinearizationResult(scalars: [], combinedEval: Fr.zero, usedGPU: false)
        }

        var alphaPow = Fr.one
        for _ in 0..<alphaOffset {
            alphaPow = frMul(alphaPow, alpha)
        }

        var scalars = [Fr]()
        var combined = Fr.zero

        for (idx, entry) in registry.entries.enumerated() {
            let scalar = entry.constraint.linearizationScalar(
                wireEvals: wireEvals,
                wireEvalsShifted: wireEvalsShifted,
                challenges: challenges
            )
            scalars.append(scalar)

            // combined += alpha^k * selector_eval * scalar
            let selectorVal = idx < selectorEvals.count ? selectorEvals[idx] : Fr.zero
            let term = frMul(alphaPow, frMul(selectorVal, scalar))
            combined = frAdd(combined, term)

            alphaPow = frMul(alphaPow, alpha)
        }

        return CustomGateLinearizationResult(
            scalars: scalars, combinedEval: combined, usedGPU: false)
    }

    // MARK: - Linearization Polynomial Construction

    /// Construct the linearization polynomial for custom gates in coefficient form.
    ///
    /// r_custom(X) = sum_i alpha^{offset+i} * scalar_i * q_i(X)
    ///
    /// Each term is a scalar * polynomial multiplication (no NTT needed).
    ///
    /// - Parameters:
    ///   - registry: Gate registry (must have selector coefficients)
    ///   - wireEvals: Wire evaluations at zeta
    ///   - wireEvalsShifted: Wire evaluations at zeta*omega
    ///   - alpha: Separation challenge
    ///   - alphaOffset: Starting power of alpha
    ///   - ntt: NTT engine (for coefficient computation if needed)
    ///   - challenges: Protocol challenges
    /// - Returns: Linearization polynomial in coefficient form
    public func constructLinearizationPoly(
        registry: CustomGateRegistry,
        wireEvals: [Fr],
        wireEvalsShifted: [Fr],
        alpha: Fr,
        alphaOffset: Int,
        ntt: NTTEngine,
        challenges: [Fr] = []
    ) throws -> [Fr] {
        let n = registry.domainSize
        guard n > 0 && registry.count > 0 else {
            return [Fr](repeating: Fr.zero, count: max(n, 1))
        }

        try registry.ensureCoeffs(ntt: ntt)

        var alphaPow = Fr.one
        for _ in 0..<alphaOffset {
            alphaPow = frMul(alphaPow, alpha)
        }

        var combined = [Fr](repeating: Fr.zero, count: n)

        for entry in registry.entries {
            let scalar = entry.constraint.linearizationScalar(
                wireEvals: wireEvals,
                wireEvalsShifted: wireEvalsShifted,
                challenges: challenges
            )

            // combined += alpha^k * scalar * selectorCoeffs
            let multiplier = frMul(alphaPow, scalar)

            if let coeffs = entry.selectorCoeffs {
                let scaledLen = min(coeffs.count, n)
                for j in 0..<scaledLen {
                    let term = frMul(multiplier, coeffs[j])
                    combined[j] = frAdd(combined[j], term)
                }
            }

            alphaPow = frMul(alphaPow, alpha)
        }

        return combined
    }

    // MARK: - Selector Isolation Check

    /// Verify that at most one custom gate selector is active at each trace row.
    /// This ensures that the constraint separation via alpha is valid.
    ///
    /// Returns indices of rows where multiple custom gate selectors are simultaneously active.
    public func checkSelectorIsolation(registry: CustomGateRegistry) -> [Int] {
        let n = registry.domainSize
        var violations = [Int]()

        for row in 0..<n {
            var activeCount = 0
            for entry in registry.entries {
                if row < entry.selectorEvals.count && !entry.selectorEvals[row].isZero {
                    activeCount += 1
                }
            }
            if activeCount > 1 {
                violations.append(row)
            }
        }

        return violations
    }

    // MARK: - Selective Evaluation

    /// Evaluate a single gate type's constraints across the trace.
    /// Useful for debugging individual gate types.
    public func evaluateSingleGateType(
        typeID: CustomGateTypeID,
        registry: CustomGateRegistry,
        wireColumns: [[Fr]],
        challenges: [Fr] = []
    ) -> GateEvaluationResult {
        let n = registry.domainSize
        guard let entry = registry.entry(for: typeID), n > 0 else {
            return GateEvaluationResult(
                residuals: [Fr](repeating: Fr.zero, count: n),
                failingRows: []
            )
        }

        var residuals = [Fr](repeating: Fr.zero, count: n)
        var failing = [Int]()

        for row in entry.activeRows {
            var wires = [Fr]()
            for col in 0..<wireColumns.count {
                wires.append(row < wireColumns[col].count ? wireColumns[col][row] : Fr.zero)
            }

            var rotatedWires = [ColumnRef: Fr]()
            for ref in entry.constraint.queriedCells {
                let rotatedRow = ((row + ref.rotation.value) % n + n) % n
                if ref.column < wireColumns.count && rotatedRow < wireColumns[ref.column].count {
                    rotatedWires[ref] = wireColumns[ref.column][rotatedRow]
                } else {
                    rotatedWires[ref] = Fr.zero
                }
            }

            let val = entry.constraint.evaluateConstraint(
                wires: wires, rotatedWires: rotatedWires, challenges: challenges)
            residuals[row] = frMul(entry.selectorEvals[row], val)

            if !residuals[row].isZero {
                failing.append(row)
            }
        }

        return GateEvaluationResult(residuals: residuals, failingRows: failing)
    }

    // MARK: - GPU Batch Boolean Check (optimized path)

    /// GPU-accelerated batch boolean check for dense boolean constraint evaluation.
    /// Falls back to CPU for small domains.
    public func gpuBatchBoolCheck(
        values: [Fr],
        selectorEvals: [Fr]
    ) -> [Fr] {
        let n = values.count
        guard n >= GPUPlonkCustomGateEngine.gpuThreshold,
              let device = device,
              let commandQueue = commandQueue,
              let pipeline = boolCheckPipeline else {
            return cpuBatchBoolCheck(values: values, selectorEvals: selectorEvals)
        }

        let frSize = MemoryLayout<Fr>.stride
        guard let valuesBuf = device.makeBuffer(bytes: values, length: n * frSize, options: .storageModeShared),
              let selectorBuf = device.makeBuffer(bytes: selectorEvals, length: n * frSize, options: .storageModeShared),
              let resultBuf = device.makeBuffer(length: n * frSize, options: .storageModeShared),
              let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            return cpuBatchBoolCheck(values: values, selectorEvals: selectorEvals)
        }

        var count = UInt32(n)
        let countBuf = device.makeBuffer(bytes: &count, length: 4, options: .storageModeShared)!

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(valuesBuf, offset: 0, index: 0)
        encoder.setBuffer(selectorBuf, offset: 0, index: 1)
        encoder.setBuffer(resultBuf, offset: 0, index: 2)
        encoder.setBuffer(countBuf, offset: 0, index: 3)

        let gridSize = MTLSize(width: n, height: 1, depth: 1)
        let tgSize = MTLSize(width: min(threadgroupSize, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        var result = [Fr](repeating: Fr.zero, count: n)
        let ptr = resultBuf.contents().assumingMemoryBound(to: Fr.self)
        for i in 0..<n { result[i] = ptr[i] }
        return result
    }

    /// CPU fallback for batch boolean check
    private func cpuBatchBoolCheck(values: [Fr], selectorEvals: [Fr]) -> [Fr] {
        let n = values.count
        var result = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            if i < selectorEvals.count && !selectorEvals[i].isZero {
                let boolCheck = frMul(values[i], frSub(Fr.one, values[i]))
                result[i] = frMul(selectorEvals[i], boolCheck)
            }
        }
        return result
    }

    // MARK: - GPU Batch XOR Check

    /// GPU-accelerated batch XOR constraint evaluation.
    public func gpuBatchXORCheck(
        aValues: [Fr],
        bValues: [Fr],
        cValues: [Fr],
        selectorEvals: [Fr]
    ) -> [Fr] {
        let n = aValues.count
        guard n >= GPUPlonkCustomGateEngine.gpuThreshold,
              let device = device,
              let commandQueue = commandQueue,
              let pipeline = xorPipeline else {
            return cpuBatchXORCheck(aValues: aValues, bValues: bValues, cValues: cValues, selectorEvals: selectorEvals)
        }

        let frSize = MemoryLayout<Fr>.stride
        guard let aBuf = device.makeBuffer(bytes: aValues, length: n * frSize, options: .storageModeShared),
              let bBuf = device.makeBuffer(bytes: bValues, length: n * frSize, options: .storageModeShared),
              let cBuf = device.makeBuffer(bytes: cValues, length: n * frSize, options: .storageModeShared),
              let selBuf = device.makeBuffer(bytes: selectorEvals, length: n * frSize, options: .storageModeShared),
              let resBuf = device.makeBuffer(length: n * frSize, options: .storageModeShared),
              let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            return cpuBatchXORCheck(aValues: aValues, bValues: bValues, cValues: cValues, selectorEvals: selectorEvals)
        }

        var count = UInt32(n)
        let countBuf = device.makeBuffer(bytes: &count, length: 4, options: .storageModeShared)!

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(aBuf, offset: 0, index: 0)
        encoder.setBuffer(bBuf, offset: 0, index: 1)
        encoder.setBuffer(cBuf, offset: 0, index: 2)
        encoder.setBuffer(selBuf, offset: 0, index: 3)
        encoder.setBuffer(resBuf, offset: 0, index: 4)
        encoder.setBuffer(countBuf, offset: 0, index: 5)

        let gridSize = MTLSize(width: n, height: 1, depth: 1)
        let tgSize = MTLSize(width: min(threadgroupSize, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        var result = [Fr](repeating: Fr.zero, count: n)
        let ptr = resBuf.contents().assumingMemoryBound(to: Fr.self)
        for i in 0..<n { result[i] = ptr[i] }
        return result
    }

    private func cpuBatchXORCheck(aValues: [Fr], bValues: [Fr], cValues: [Fr], selectorEvals: [Fr]) -> [Fr] {
        let n = aValues.count
        let two = frFromInt(2)
        var result = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            if i < selectorEvals.count && !selectorEvals[i].isZero {
                let ab = frMul(aValues[i], bValues[i])
                let expected = frSub(frAdd(aValues[i], bValues[i]), frMul(two, ab))
                let residual = frSub(cValues[i], expected)
                result[i] = frMul(selectorEvals[i], residual)
            }
        }
        return result
    }

    // MARK: - GPU Batch Sbox Check

    /// GPU-accelerated batch Poseidon S-box check: c = a^5
    public func gpuBatchSboxCheck(
        aValues: [Fr],
        cValues: [Fr],
        selectorEvals: [Fr]
    ) -> [Fr] {
        let n = aValues.count
        guard n >= GPUPlonkCustomGateEngine.gpuThreshold,
              let device = device,
              let commandQueue = commandQueue,
              let pipeline = sboxPipeline else {
            return cpuBatchSboxCheck(aValues: aValues, cValues: cValues, selectorEvals: selectorEvals)
        }

        let frSize = MemoryLayout<Fr>.stride
        guard let aBuf = device.makeBuffer(bytes: aValues, length: n * frSize, options: .storageModeShared),
              let cBuf = device.makeBuffer(bytes: cValues, length: n * frSize, options: .storageModeShared),
              let selBuf = device.makeBuffer(bytes: selectorEvals, length: n * frSize, options: .storageModeShared),
              let resBuf = device.makeBuffer(length: n * frSize, options: .storageModeShared),
              let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            return cpuBatchSboxCheck(aValues: aValues, cValues: cValues, selectorEvals: selectorEvals)
        }

        var count = UInt32(n)
        let countBuf = device.makeBuffer(bytes: &count, length: 4, options: .storageModeShared)!

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(aBuf, offset: 0, index: 0)
        encoder.setBuffer(cBuf, offset: 0, index: 1)
        encoder.setBuffer(selBuf, offset: 0, index: 2)
        encoder.setBuffer(resBuf, offset: 0, index: 3)
        encoder.setBuffer(countBuf, offset: 0, index: 4)

        let gridSize = MTLSize(width: n, height: 1, depth: 1)
        let tgSize = MTLSize(width: min(threadgroupSize, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        var result = [Fr](repeating: Fr.zero, count: n)
        let ptr = resBuf.contents().assumingMemoryBound(to: Fr.self)
        for i in 0..<n { result[i] = ptr[i] }
        return result
    }

    private func cpuBatchSboxCheck(aValues: [Fr], cValues: [Fr], selectorEvals: [Fr]) -> [Fr] {
        let n = aValues.count
        var result = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            if i < selectorEvals.count && !selectorEvals[i].isZero {
                let a2 = frSqr(aValues[i])
                let a4 = frSqr(a2)
                let a5 = frMul(aValues[i], a4)
                let residual = frSub(cValues[i], a5)
                result[i] = frMul(selectorEvals[i], residual)
            }
        }
        return result
    }

    // MARK: - Combined Multi-Gate Quotient (GPU Optimized)

    /// Compute quotient contribution combining multiple gate types with GPU acceleration.
    /// Uses separate GPU kernels for each gate type that supports it, then combines.
    public func gpuComputeQuotientCombined(
        registry: CustomGateRegistry,
        wireColumns: [[Fr]],
        alpha: Fr,
        alphaOffset: Int,
        ntt: NTTEngine,
        challenges: [Fr] = []
    ) throws -> [Fr] {
        let n = registry.domainSize
        guard n > 0 && (n & (n - 1)) == 0 else {
            return [Fr](repeating: Fr.zero, count: max(n, 1))
        }

        // For small domains or no GPU, use CPU path
        if n < GPUPlonkCustomGateEngine.gpuThreshold || device == nil {
            return try computeQuotientContribution(
                registry: registry,
                wireColumns: wireColumns,
                alpha: alpha,
                alphaOffset: alphaOffset,
                ntt: ntt,
                challenges: challenges
            )
        }

        // GPU-accelerated path: dispatch optimized kernels per gate type
        var alphaPow = Fr.one
        for _ in 0..<alphaOffset {
            alphaPow = frMul(alphaPow, alpha)
        }

        var combined = [Fr](repeating: Fr.zero, count: n)

        for entry in registry.entries {
            var gateResiduals: [Fr]

            // Use specialized GPU kernel when available
            switch entry.constraint.typeID {
            case .boolCheck:
                let aCol = wireColumns.count > 0 ? wireColumns[0] : [Fr](repeating: Fr.zero, count: n)
                gateResiduals = gpuBatchBoolCheck(values: aCol, selectorEvals: entry.selectorEvals)

            case .xorGate:
                let aCol = wireColumns.count > 0 ? wireColumns[0] : [Fr](repeating: Fr.zero, count: n)
                let bCol = wireColumns.count > 1 ? wireColumns[1] : [Fr](repeating: Fr.zero, count: n)
                let cCol = wireColumns.count > 2 ? wireColumns[2] : [Fr](repeating: Fr.zero, count: n)
                gateResiduals = gpuBatchXORCheck(aValues: aCol, bValues: bCol, cValues: cCol, selectorEvals: entry.selectorEvals)

            case .poseidonSbox:
                let aCol = wireColumns.count > 0 ? wireColumns[0] : [Fr](repeating: Fr.zero, count: n)
                let cCol = wireColumns.count > 2 ? wireColumns[2] : [Fr](repeating: Fr.zero, count: n)
                gateResiduals = gpuBatchSboxCheck(aValues: aCol, cValues: cCol, selectorEvals: entry.selectorEvals)

            default:
                // Fallback to CPU for unsupported gate types
                gateResiduals = [Fr](repeating: Fr.zero, count: n)
                for row in entry.activeRows {
                    var wires = [Fr]()
                    for col in 0..<wireColumns.count {
                        wires.append(row < wireColumns[col].count ? wireColumns[col][row] : Fr.zero)
                    }
                    var rotatedWires = [ColumnRef: Fr]()
                    for ref in entry.constraint.queriedCells {
                        let rotatedRow = ((row + ref.rotation.value) % n + n) % n
                        if ref.column < wireColumns.count && rotatedRow < wireColumns[ref.column].count {
                            rotatedWires[ref] = wireColumns[ref.column][rotatedRow]
                        }
                    }
                    let val = entry.constraint.evaluateConstraint(
                        wires: wires, rotatedWires: rotatedWires, challenges: challenges)
                    gateResiduals[row] = frMul(entry.selectorEvals[row], val)
                }
            }

            // Combine with alpha power
            for i in 0..<n {
                if !gateResiduals[i].isZero {
                    combined[i] = frAdd(combined[i], frMul(alphaPow, gateResiduals[i]))
                }
            }
            alphaPow = frMul(alphaPow, alpha)
        }

        return try ntt.intt(combined)
    }

    // MARK: - Utility: Verify Witness Against All Custom Gates

    /// Convenience: register gates, evaluate constraints, and return pass/fail.
    public func verifyWitness(
        registry: CustomGateRegistry,
        wireColumns: [[Fr]],
        challenges: [Fr] = []
    ) -> Bool {
        let result = evaluateAllConstraints(
            registry: registry,
            wireColumns: wireColumns,
            challenges: challenges
        )
        return result.isSatisfied
    }

    // MARK: - Gate Statistics

    /// Compute statistics about the registered custom gates.
    public struct GateStatistics {
        public let totalGateTypes: Int
        public let totalActiveRows: Int
        public let maxConstraintDegree: Int
        public let gateTypeNames: [String]
        public let activeRowsPerGate: [Int]
        public let activationDensity: Double  // fraction of rows with any active gate

        public init(totalGateTypes: Int, totalActiveRows: Int, maxConstraintDegree: Int,
                    gateTypeNames: [String], activeRowsPerGate: [Int], activationDensity: Double) {
            self.totalGateTypes = totalGateTypes
            self.totalActiveRows = totalActiveRows
            self.maxConstraintDegree = maxConstraintDegree
            self.gateTypeNames = gateTypeNames
            self.activeRowsPerGate = activeRowsPerGate
            self.activationDensity = activationDensity
        }
    }

    public func computeStatistics(registry: CustomGateRegistry) -> GateStatistics {
        let n = registry.domainSize
        var totalActive = Set<Int>()
        var names = [String]()
        var perGate = [Int]()

        for entry in registry.entries {
            names.append(entry.constraint.typeID.name)
            perGate.append(entry.activeRows.count)
            for row in entry.activeRows {
                totalActive.insert(row)
            }
        }

        let density = n > 0 ? Double(totalActive.count) / Double(n) : 0.0

        return GateStatistics(
            totalGateTypes: registry.count,
            totalActiveRows: totalActive.count,
            maxConstraintDegree: registry.maxConstraintDegree,
            gateTypeNames: names,
            activeRowsPerGate: perGate,
            activationDensity: density
        )
    }

    // MARK: - Helper: Build Domain

    private func buildDomain(n: Int, ntt: NTTEngine) throws -> [Fr] {
        let omega = computeOmega(n: n)
        var domain = [Fr](repeating: Fr.zero, count: n)
        domain[0] = Fr.one
        for i in 1..<n {
            domain[i] = frMul(domain[i - 1], omega)
        }
        return domain
    }

    /// Compute omega (primitive n-th root of unity) for BN254 Fr.
    /// Uses the generator 5 raised to (p-1)/n.
    private func computeOmega(n: Int) -> Fr {
        // BN254 Fr: p - 1 = 2^28 * t, so max power-of-2 subgroup is 2^28
        // omega_2^28 = g^((p-1)/2^28) where g is a generator
        // For n = 2^k, omega_n = (omega_2^28)^(2^(28-k))
        var logN = 0
        var tmp = n
        while tmp > 1 { tmp >>= 1; logN += 1 }

        // Hardcoded omega for small powers of 2 (BN254 Fr)
        // omega_4 is a well-known constant
        let omega4 = Fr.from64([
            0x8b17ea266ef1c2ed, 0x3c28d666a5c2d854,
            0x9c1dc3c7eb6d9dca, 0x2e2419f9ec02ec39
        ])

        if logN == 2 { return omega4 }

        // For logN > 2, compute from primitive 2^28 root
        // Use repeated squaring from omega_2^28
        let omega28 = Fr.from64([
            0x2a3c09f0a58a7e85, 0x006a5bfc9bc17780,
            0x20567e17131db1c8, 0x2dc8f09be0352970
        ])

        var omega = omega28
        for _ in 0..<(28 - logN) {
            omega = frSqr(omega)
        }
        return omega
    }

    // MARK: - Metal Shader Compilation

    private static func compileBoolCheckKernel(device: MTLDevice) -> MTLComputePipelineState? {
        let shaderDir = findShaderDir()
        guard let frSource = try? String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8) else {
            return nil
        }
        let frClean = cleanFrSource(frSource)

        let kernel = """

        // Bool check: selector[i] * values[i] * (1 - values[i])
        kernel void custom_gate_bool_check(
            device const Fr *values [[buffer(0)]],
            device const Fr *selector [[buffer(1)]],
            device Fr *result [[buffer(2)]],
            constant uint &n [[buffer(3)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= n) return;
            Fr one = FR_ONE;
            Fr val = values[gid];
            Fr diff = fr_sub(one, val);
            Fr boolCheck = fr_mul(val, diff);
            result[gid] = fr_mul(selector[gid], boolCheck);
        }
        """
        return compileKernel(device: device, source: frClean + "\n" + kernel, name: "custom_gate_bool_check")
    }

    private static func compileXORKernel(device: MTLDevice) -> MTLComputePipelineState? {
        let shaderDir = findShaderDir()
        guard let frSource = try? String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8) else {
            return nil
        }
        let frClean = cleanFrSource(frSource)

        let kernel = """

        // XOR: selector * (c - (a + b - 2*a*b))
        kernel void custom_gate_xor(
            device const Fr *a [[buffer(0)]],
            device const Fr *b [[buffer(1)]],
            device const Fr *c [[buffer(2)]],
            device const Fr *selector [[buffer(3)]],
            device Fr *result [[buffer(4)]],
            constant uint &n [[buffer(5)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= n) return;
            Fr ab = fr_mul(a[gid], b[gid]);
            Fr twoAb = fr_add(ab, ab);
            Fr expected = fr_sub(fr_add(a[gid], b[gid]), twoAb);
            Fr residual = fr_sub(c[gid], expected);
            result[gid] = fr_mul(selector[gid], residual);
        }
        """
        return compileKernel(device: device, source: frClean + "\n" + kernel, name: "custom_gate_xor")
    }

    private static func compileSboxKernel(device: MTLDevice) -> MTLComputePipelineState? {
        let shaderDir = findShaderDir()
        guard let frSource = try? String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8) else {
            return nil
        }
        let frClean = cleanFrSource(frSource)

        let kernel = """

        // Sbox: selector * (c - a^5)
        kernel void custom_gate_sbox(
            device const Fr *a [[buffer(0)]],
            device const Fr *c [[buffer(1)]],
            device const Fr *selector [[buffer(2)]],
            device Fr *result [[buffer(3)]],
            constant uint &n [[buffer(4)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= n) return;
            Fr a2 = fr_mul(a[gid], a[gid]);
            Fr a4 = fr_mul(a2, a2);
            Fr a5 = fr_mul(a[gid], a4);
            Fr residual = fr_sub(c[gid], a5);
            result[gid] = fr_mul(selector[gid], residual);
        }
        """
        return compileKernel(device: device, source: frClean + "\n" + kernel, name: "custom_gate_sbox")
    }

    private static func compileCombineKernel(device: MTLDevice) -> MTLComputePipelineState? {
        let shaderDir = findShaderDir()
        guard let frSource = try? String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8) else {
            return nil
        }
        let frClean = cleanFrSource(frSource)

        let kernel = """

        // Combine: result[i] += alpha * contribution[i]
        kernel void custom_gate_combine(
            device Fr *result [[buffer(0)]],
            device const Fr *contribution [[buffer(1)]],
            constant Fr &alpha [[buffer(2)]],
            constant uint &n [[buffer(3)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= n) return;
            Fr scaled = fr_mul(alpha, contribution[gid]);
            result[gid] = fr_add(result[gid], scaled);
        }
        """
        return compileKernel(device: device, source: frClean + "\n" + kernel, name: "custom_gate_combine")
    }

    private static func cleanFrSource(_ source: String) -> String {
        source
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")
    }

    private static func compileKernel(device: MTLDevice, source: String, name: String) -> MTLComputePipelineState? {
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        guard let library = try? device.makeLibrary(source: source, options: options),
              let fn = library.makeFunction(name: name),
              let pipeline = try? device.makeComputePipelineState(function: fn) else {
            return nil
        }
        return pipeline
    }

    private static func findShaderDir() -> String {
        let execDir = (CommandLine.arguments[0] as NSString).deletingLastPathComponent
        for bundle in Bundle.allBundles {
            if let url = bundle.url(forResource: "Shaders", withExtension: nil) {
                if FileManager.default.fileExists(atPath: url.appendingPathComponent("fields/bn254_fr.metal").path) {
                    return url.path
                }
            }
        }
        let candidates = [
            execDir + "/Shaders",
            execDir + "/../share/zkMetal/Shaders",
            "Sources/zkMetal/Shaders",
            FileManager.default.currentDirectoryPath + "/Sources/zkMetal/Shaders",
        ]
        for c in candidates {
            if FileManager.default.fileExists(atPath: c + "/fields/bn254_fr.metal") {
                return c
            }
        }
        return "Sources/zkMetal/Shaders"
    }
}

// MARK: - Adapter: CustomGateConstraint -> CustomGate

/// Wraps a CustomGateConstraint into the existing CustomGate protocol for
/// interoperability with CustomGateSet's polynomial-level computation.
struct ConstraintToCustomGateAdapter: CustomGate {
    let constraint: CustomGateConstraint

    var name: String { constraint.typeID.name }

    var queriedCells: [ColumnRef] { constraint.queriedCells }

    func evaluate(rotations: [ColumnRef: Fr], challenges: [Fr]) -> Fr {
        // Extract wire values from rotations map in column order
        var wires = [Fr]()
        for col in 0..<constraint.wireCount {
            wires.append(rotations[ColumnRef(column: col, rotation: .cur)] ?? Fr.zero)
        }
        return constraint.evaluateConstraint(wires: wires, rotatedWires: rotations, challenges: challenges)
    }
}
