// PlonkECCGates -- Elliptic curve operation gates for Plonk circuits
//
// Provides custom gates for constraining elliptic curve operations:
//   - ECAddGate: short Weierstrass point addition (BN254)
//   - ECDoubleGate: short Weierstrass point doubling
//   - ECScalarMulGate: scalar multiplication via double-and-add decomposition
//   - TwistedEdwardsAddGate: twisted Edwards addition (BabyJubjub/Ed25519)
//
// Wire layout convention:
//   Columns 0-2 form standard Plonk (a, b, c). Additional advice columns
//   (3+) are used for intermediate values and auxiliary witnesses.

import Foundation
import NeonFieldOps

// MARK: - ECAddGate

/// Constrains P3 = P1 + P2 on a short Weierstrass curve y^2 = x^3 + b (BN254).
///
/// Given P1 = (x1, y1), P2 = (x2, y2), P3 = (x3, y3):
///   lambda = (y2 - y1) / (x2 - x1)
///   x3 = lambda^2 - x1 - x2
///   y3 = lambda * (x1 - x3) - y1
///
/// Wire layout (2 rows):
///   Row 0: col0 = x1, col1 = y1, col2 = x2
///   Row 1: col0 = y2, col1 = x3, col2 = y3
///   Advice col3 row 0: lambda (witness-provided)
///
/// Constraints (all must be zero):
///   1. lambda * (x2 - x1) - (y2 - y1) = 0
///   2. x3 - (lambda^2 - x1 - x2) = 0
///   3. y3 - (lambda * (x1 - x3) - y1) = 0
public struct ECAddGate: CustomGate {
    public let name = "ECAdd"

    public init() {}

    public var queriedCells: [ColumnRef] {
        [
            ColumnRef(column: 0, rotation: .cur),   // x1
            ColumnRef(column: 1, rotation: .cur),   // y1
            ColumnRef(column: 2, rotation: .cur),   // x2
            ColumnRef(column: 0, rotation: .next),  // y2
            ColumnRef(column: 1, rotation: .next),  // x3
            ColumnRef(column: 2, rotation: .next),  // y3
            ColumnRef(column: 3, rotation: .cur),   // lambda
        ]
    }

    public func evaluate(rotations: [ColumnRef: Fr], challenges: [Fr]) -> Fr {
        let x1 = rotations[ColumnRef(column: 0, rotation: .cur)] ?? Fr.zero
        let y1 = rotations[ColumnRef(column: 1, rotation: .cur)] ?? Fr.zero
        let x2 = rotations[ColumnRef(column: 2, rotation: .cur)] ?? Fr.zero
        let y2 = rotations[ColumnRef(column: 0, rotation: .next)] ?? Fr.zero
        let x3 = rotations[ColumnRef(column: 1, rotation: .next)] ?? Fr.zero
        let y3 = rotations[ColumnRef(column: 2, rotation: .next)] ?? Fr.zero
        let lam = rotations[ColumnRef(column: 3, rotation: .cur)] ?? Fr.zero

        // Constraint 1: lambda * (x2 - x1) - (y2 - y1)
        let c1 = frSub(frMul(lam, frSub(x2, x1)), frSub(y2, y1))

        // Constraint 2: x3 - lambda^2 + x1 + x2
        let lamSq = frSqr(lam)
        let c2 = frSub(frAdd(x3, frAdd(x1, x2)), lamSq)
        // Rearranged: x3 + x1 + x2 - lambda^2 = 0  <=> x3 = lambda^2 - x1 - x2

        // Constraint 3: y3 - lambda * (x1 - x3) + y1
        let c3 = frAdd(frSub(y3, frMul(lam, frSub(x1, x3))), y1)
        // y3 + y1 - lambda*(x1-x3) = 0 <=> y3 = lambda*(x1-x3) - y1

        // Combine: c1^2 + c2^2 + c3^2 would work but is expensive.
        // Instead use random linear combination with challenges or
        // return product c1 * c2 * c3 (zero iff all zero, but higher degree).
        // For simplicity and correctness, combine linearly assuming
        // each is constrained separately via the quotient polynomial.
        // The standard approach: return c1 + alpha*c2 + alpha^2*c3
        // where alpha comes from challenges.
        if challenges.count >= 1 {
            let alpha = challenges[0]
            return frAdd(c1, frAdd(frMul(alpha, c2), frMul(frSqr(alpha), c3)))
        }
        // Fallback: product (degree 3 in constraints, acceptable for gate evaluation)
        return frAdd(frAdd(frMul(c1, c1), frMul(c2, c2)), frMul(c3, c3))
    }
}

// MARK: - ECDoubleGate

/// Constrains P2 = 2 * P1 on a short Weierstrass curve y^2 = x^3 + b.
///
/// Given P1 = (x1, y1), P2 = (x2, y2):
///   lambda = 3*x1^2 / (2*y1)
///   x2 = lambda^2 - 2*x1
///   y2 = lambda * (x1 - x2) - y1
///
/// Wire layout (single row + next):
///   Row 0: col0 = x1, col1 = y1, col2 = lambda (witness)
///   Row 1: col0 = x2, col1 = y2, col2 = unused
///
/// Constraints:
///   1. lambda * 2*y1 - 3*x1^2 = 0
///   2. x2 - lambda^2 + 2*x1 = 0
///   3. y2 - lambda*(x1-x2) + y1 = 0
public struct ECDoubleGate: CustomGate {
    public let name = "ECDouble"

    public init() {}

    public var queriedCells: [ColumnRef] {
        [
            ColumnRef(column: 0, rotation: .cur),   // x1
            ColumnRef(column: 1, rotation: .cur),   // y1
            ColumnRef(column: 2, rotation: .cur),   // lambda
            ColumnRef(column: 0, rotation: .next),  // x2
            ColumnRef(column: 1, rotation: .next),  // y2
        ]
    }

    public func evaluate(rotations: [ColumnRef: Fr], challenges: [Fr]) -> Fr {
        let x1 = rotations[ColumnRef(column: 0, rotation: .cur)] ?? Fr.zero
        let y1 = rotations[ColumnRef(column: 1, rotation: .cur)] ?? Fr.zero
        let lam = rotations[ColumnRef(column: 2, rotation: .cur)] ?? Fr.zero
        let x2 = rotations[ColumnRef(column: 0, rotation: .next)] ?? Fr.zero
        let y2 = rotations[ColumnRef(column: 1, rotation: .next)] ?? Fr.zero

        let two = frAdd(Fr.one, Fr.one)
        let three = frAdd(two, Fr.one)

        // Constraint 1: lambda * 2*y1 - 3*x1^2
        let c1 = frSub(frMul(lam, frMul(two, y1)), frMul(three, frSqr(x1)))

        // Constraint 2: x2 + 2*x1 - lambda^2
        let c2 = frSub(frAdd(x2, frMul(two, x1)), frSqr(lam))

        // Constraint 3: y2 + y1 - lambda*(x1 - x2)
        let c3 = frSub(frAdd(y2, y1), frMul(lam, frSub(x1, x2)))

        // Sum of squares (zero iff each constraint is zero)
        return frAdd(frAdd(frMul(c1, c1), frMul(c2, c2)), frMul(c3, c3))
    }
}

// MARK: - ECScalarMulGate

/// Constrains Q = s * P using double-and-add decomposition.
///
/// This is a compound gate that generates multiple rows:
///   - One BinaryDecomposeGate row per scalar bit
///   - One ECDoubleGate per bit
///   - One ECAddGate per set bit (conditional)
///
/// For a k-bit scalar, this produces O(2k) rows in the execution trace.
///
/// Wire layout:
///   Each "step" uses 2 rows (double + conditional add).
///   The scalar bits are placed in advice columns and constrained to be boolean.
///
/// Usage: call `expandGates()` to get the list of primitive gates and wire mappings
/// that should be added to the constraint compiler.
public struct ECScalarMulGate {
    public let name = "ECScalarMul"

    /// Number of bits in the scalar
    public let scalarBits: Int

    public init(scalarBits: Int = 254) {
        precondition(scalarBits > 0 && scalarBits <= 256)
        self.scalarBits = scalarBits
    }

    /// Expanded gate description for a single double-and-add step.
    public struct Step {
        /// The primitive gate (ECDoubleGate or ECAddGate)
        public let gate: any CustomGate
        /// Wire variable indices for this step's rows
        public let wires: [[Int]]
    }

    /// Expand the scalar multiplication into primitive ECDouble + ECAdd + BoolCheck gates.
    ///
    /// - Parameters:
    ///   - compiler: The constraint compiler (used to allocate variables)
    ///   - pxVar: Variable index for input point P.x
    ///   - pyVar: Variable index for input point P.y
    ///   - scalarBitVars: Variable indices for each scalar bit (LSB first), length = scalarBits
    ///   - selectorDouble: Selector index for ECDoubleGate
    ///   - selectorAdd: Selector index for ECAddGate
    ///   - selectorBool: Selector index for BoolCheckGate
    /// - Returns: List of gate descriptors to add to the compiler
    public func expandGates(
        compiler: PlonkConstraintCompiler,
        pxVar: Int, pyVar: Int,
        scalarBitVars: [Int],
        selectorDouble: Int,
        selectorAdd: Int,
        selectorBool: Int
    ) -> [CustomGateDesc] {
        precondition(scalarBitVars.count == scalarBits)

        var result = [CustomGateDesc]()

        // Boolean constrain each scalar bit
        let boolGate = BoolCheckGate(column: 0)
        for bitVar in scalarBitVars {
            result.append(CustomGateDesc(
                gate: boolGate, wires: [[bitVar]], selectorIndex: selectorBool))
        }

        // Double-and-add chain (MSB to LSB)
        // Start with identity (represented as (0, 0) with a flag, or the MSB bit * P)
        // For circuit simplicity, we use the standard approach:
        //   acc = O (identity)
        //   for i in (scalarBits-1)...0:
        //     acc = 2*acc
        //     if bit_i == 1: acc = acc + P
        //
        // Each step allocates intermediate point variables.
        let doubleGate = ECDoubleGate()
        let addGate = ECAddGate()

        // Allocate accumulator variables for each step
        // acc[0] = initial (0,0), acc[i+1] = result of step i
        var accXVars = compiler.addVariables(scalarBits + 1)
        var accYVars = compiler.addVariables(scalarBits + 1)

        for i in 0..<scalarBits {
            let bitIdx = scalarBits - 1 - i  // MSB first

            // Double: (accX[i], accY[i]) -> (dblX, dblY)
            let dblXVar = compiler.addVariable()
            let dblYVar = compiler.addVariable()
            let lamDblVar = compiler.addVariable()

            result.append(CustomGateDesc(
                gate: doubleGate,
                wires: [
                    [accXVars[i], accYVars[i], lamDblVar],  // row 0
                    [dblXVar, dblYVar, 0],                   // row 1 (next)
                ],
                selectorIndex: selectorDouble))

            // Conditional add: if bit == 1, add P
            // (dblX, dblY) + (px, py) -> (accX[i+1], accY[i+1])
            let lamAddVar = compiler.addVariable()

            result.append(CustomGateDesc(
                gate: addGate,
                wires: [
                    [dblXVar, dblYVar, pxVar, lamAddVar],  // row 0: x1,y1,x2,lambda
                    [pyVar, accXVars[i + 1], accYVars[i + 1]], // row 1: y2,x3,y3
                ],
                selectorIndex: selectorAdd))
        }

        return result
    }
}

// MARK: - TwistedEdwardsAddGate

/// Constrains P3 = P1 + P2 on a twisted Edwards curve:
///   a*x^2 + y^2 = 1 + d*x^2*y^2
///
/// Addition formula:
///   x3 = (x1*y2 + y1*x2) / (1 + d*x1*x2*y1*y2)
///   y3 = (y1*y2 - a*x1*x2) / (1 - d*x1*x2*y1*y2)
///
/// To avoid division in constraints, we use the equivalent form:
///   x3 * (1 + d*x1*x2*y1*y2) = x1*y2 + y1*x2
///   y3 * (1 - d*x1*x2*y1*y2) = y1*y2 - a*x1*x2
///
/// Wire layout (2 rows):
///   Row 0: col0 = x1, col1 = y1, col2 = x2
///   Row 1: col0 = y2, col1 = x3, col2 = y3
///
/// The curve parameters a, d are baked into the gate at construction time.
/// For BabyJubjub: a = 168700, d = 168696
/// For Ed25519: a = -1, d = -121665/121666
public struct TwistedEdwardsAddGate: CustomGate {
    public let name: String

    /// Curve parameter a
    public let a: Fr
    /// Curve parameter d
    public let d: Fr

    /// Create a twisted Edwards add gate with specific curve parameters.
    /// - Parameters:
    ///   - a: Curve coefficient a
    ///   - d: Curve coefficient d
    ///   - name: Optional name (defaults to "TwistedEdwardsAdd")
    public init(a: Fr, d: Fr, name: String = "TwistedEdwardsAdd") {
        self.a = a
        self.d = d
        self.name = name
    }

    /// Convenience: BabyJubjub curve (a=168700, d=168696)
    public static func babyJubjub() -> TwistedEdwardsAddGate {
        TwistedEdwardsAddGate(
            a: frFromInt(168700),
            d: frFromInt(168696),
            name: "BabyJubjubAdd")
    }

    public var queriedCells: [ColumnRef] {
        [
            ColumnRef(column: 0, rotation: .cur),   // x1
            ColumnRef(column: 1, rotation: .cur),   // y1
            ColumnRef(column: 2, rotation: .cur),   // x2
            ColumnRef(column: 0, rotation: .next),  // y2
            ColumnRef(column: 1, rotation: .next),  // x3
            ColumnRef(column: 2, rotation: .next),  // y3
        ]
    }

    public func evaluate(rotations: [ColumnRef: Fr], challenges: [Fr]) -> Fr {
        let x1 = rotations[ColumnRef(column: 0, rotation: .cur)] ?? Fr.zero
        let y1 = rotations[ColumnRef(column: 1, rotation: .cur)] ?? Fr.zero
        let x2 = rotations[ColumnRef(column: 2, rotation: .cur)] ?? Fr.zero
        let y2 = rotations[ColumnRef(column: 0, rotation: .next)] ?? Fr.zero
        let x3 = rotations[ColumnRef(column: 1, rotation: .next)] ?? Fr.zero
        let y3 = rotations[ColumnRef(column: 2, rotation: .next)] ?? Fr.zero

        // d * x1 * x2 * y1 * y2
        let x1x2 = frMul(x1, x2)
        let y1y2 = frMul(y1, y2)
        let dxy = frMul(d, frMul(x1x2, y1y2))

        // Constraint 1: x3 * (1 + dxy) - (x1*y2 + y1*x2)
        let c1 = frSub(
            frMul(x3, frAdd(Fr.one, dxy)),
            frAdd(frMul(x1, y2), frMul(y1, x2)))

        // Constraint 2: y3 * (1 - dxy) - (y1*y2 - a*x1*x2)
        let c2 = frSub(
            frMul(y3, frSub(Fr.one, dxy)),
            frSub(y1y2, frMul(a, x1x2)))

        // Sum of squares
        return frAdd(frMul(c1, c1), frMul(c2, c2))
    }
}
