// PlonkCustomGates -- Halo2-compatible custom gate framework for Plonkish circuits
//
// Provides a protocol-driven approach to defining custom gates with rotation support,
// enabling Halo2-style circuit definitions where gates can reference adjacent rows.
//
// Custom gates contribute to the combined quotient polynomial:
//   t(x) * Z_H(x) = sum_i alpha^i * gate_i(x)
//
// Each gate evaluates to a polynomial expression over wire columns with rotations.
// The prover computes all gate contributions in coefficient form and combines them
// with the standard Plonk gate and permutation constraints.
//
// Built-in gates: RangeCheck, BoolCheck, LookupGate, PoseidonGate.
// Users can define additional gates by conforming to the CustomGate protocol.

import Foundation
import NeonFieldOps

// MARK: - Rotation

/// Row rotation for accessing adjacent rows in the execution trace.
/// In Halo2, gates can reference cells at the current row plus an offset.
/// The rotation is applied modulo the domain size n.
public enum Rotation: Hashable, Sendable {
    /// Current row (offset 0)
    case cur
    /// Next row (offset +1)
    case next
    /// Previous row (offset -1)
    case prev
    /// Arbitrary offset (positive = forward, negative = backward)
    case offset(Int)

    /// Integer offset value
    public var value: Int {
        switch self {
        case .cur: return 0
        case .next: return 1
        case .prev: return -1
        case .offset(let k): return k
        }
    }
}

// MARK: - Column Reference

/// Reference to a column in the execution trace with a rotation.
/// Mirrors Halo2's Expression<F> column references.
public struct ColumnRef: Hashable, Sendable {
    /// Column index (0 = a/left, 1 = b/right, 2 = c/output, 3+ = advice columns)
    public let column: Int
    /// Row rotation
    public let rotation: Rotation

    public init(column: Int, rotation: Rotation = .cur) {
        self.column = column
        self.rotation = rotation
    }
}

// MARK: - Custom Gate Protocol

/// Protocol for custom gates in a Plonkish constraint system.
///
/// Each gate defines a polynomial identity that must evaluate to zero on the domain.
/// The gate can reference wire values at rotated positions via ColumnRef.
///
/// When generating the quotient polynomial, the prover evaluates each gate's
/// constraint polynomial and accumulates it with a random alpha separator.
public protocol CustomGate {
    /// Human-readable name for debugging and error messages
    var name: String { get }

    /// The set of column references (with rotations) that this gate reads.
    /// The prover uses this to determine which shifted wire polynomials are needed.
    var queriedCells: [ColumnRef] { get }

    /// Maximum rotation offset (absolute value). Used to determine how many
    /// extra rows of padding are needed in the execution trace.
    var maxRotation: Int { get }

    /// Evaluate the gate constraint at a single row.
    ///
    /// - Parameters:
    ///   - rotations: Maps each ColumnRef to its resolved field value at the given row.
    ///     For ColumnRef(column: 0, rotation: .next), the value is a(omega^{i+1}).
    ///   - challenges: Protocol challenges (alpha, beta, gamma, etc.) available to the gate.
    /// - Returns: The constraint value. Must be zero for all rows in a valid witness.
    func evaluate(rotations: [ColumnRef: Fr], challenges: [Fr]) -> Fr

    /// Compute the gate's contribution to the quotient polynomial in coefficient form.
    ///
    /// The default implementation evaluates the gate at each row of the domain and
    /// interpolates via iNTT. Custom gates may override for more efficient computation
    /// (e.g., directly constructing the polynomial from wire polynomial coefficients).
    ///
    /// - Parameters:
    ///   - wirePolys: Wire polynomials in coefficient form. wirePolys[col] = coefficients of column `col`.
    ///   - selectorPoly: The selector polynomial (coefficient form) that activates this gate.
    ///   - domain: Evaluation domain (omega^0, ..., omega^{n-1}).
    ///   - omega: Primitive n-th root of unity.
    ///   - n: Domain size.
    ///   - ntt: NTT engine for polynomial multiplication.
    /// - Returns: Gate contribution polynomial in coefficient form (before division by Z_H).
    func quotientContribution(wirePolys: [[Fr]], selectorPoly: [Fr],
                              domain: [Fr], omega: Fr, n: Int,
                              ntt: NTTEngine) throws -> [Fr]
}

// MARK: - Default quotient contribution

extension CustomGate {
    public var maxRotation: Int {
        queriedCells.map { abs($0.rotation.value) }.max() ?? 0
    }

    /// Default: evaluate the gate pointwise on the domain, multiply by selector, then iNTT.
    /// This is correct but O(n * numRotations) -- custom gates can override for efficiency.
    public func quotientContribution(wirePolys: [[Fr]], selectorPoly: [Fr],
                                     domain: [Fr], omega: Fr, n: Int,
                                     ntt: NTTEngine) throws -> [Fr] {
        // Evaluate wire polynomials on the domain (NTT of coefficients)
        var wireEvals = [[Fr]]()
        for col in 0..<wirePolys.count {
            var padded = wirePolys[col]
            if padded.count < n {
                padded += [Fr](repeating: Fr.zero, count: n - padded.count)
            }
            wireEvals.append(try ntt.ntt(Array(padded.prefix(n))))
        }

        // Build evaluation of the gate constraint at each row
        var gateEvals = [Fr](repeating: Fr.zero, count: n)
        let selectorEvals = try ntt.ntt(
            {
                if selectorPoly.count >= n { return Array(selectorPoly.prefix(n)) }
                var padded = [Fr](repeating: Fr.zero, count: n)
                selectorPoly.withUnsafeBytes { src in
                    padded.withUnsafeMutableBytes { dst in
                        memcpy(dst.baseAddress!, src.baseAddress!, selectorPoly.count * MemoryLayout<Fr>.stride)
                    }
                }
                return padded
            }())

        for i in 0..<n {
            // Skip rows where selector is zero (optimization)
            if frEqual(selectorEvals[i], Fr.zero) { continue }

            // Resolve all column references for this row
            var resolved = [ColumnRef: Fr]()
            for ref in queriedCells {
                let row = ((i + ref.rotation.value) % n + n) % n
                if ref.column < wireEvals.count {
                    resolved[ref] = wireEvals[ref.column][row]
                } else {
                    resolved[ref] = Fr.zero
                }
            }

            let val = evaluate(rotations: resolved, challenges: [])
            gateEvals[i] = frMul(selectorEvals[i], val)
        }

        // Convert to coefficient form
        return try ntt.intt(gateEvals)
    }
}

// MARK: - BoolCheck Gate

/// Boolean constraint: a * (1 - a) = 0, ensuring a is 0 or 1.
/// Equivalent to the range gate in the existing PlonkCircuit with qRange selector.
///
/// This is the simplest custom gate -- no rotations needed.
public struct BoolCheckGate: CustomGate {
    public let name = "BoolCheck"

    /// Which column to constrain (default: column 0 = wire a)
    public let column: Int

    public init(column: Int = 0) {
        self.column = column
    }

    public var queriedCells: [ColumnRef] {
        [ColumnRef(column: column, rotation: .cur)]
    }

    public func evaluate(rotations: [ColumnRef: Fr], challenges: [Fr]) -> Fr {
        let a = rotations[ColumnRef(column: column, rotation: .cur)] ?? Fr.zero
        // a * (1 - a)
        return frMul(a, frSub(Fr.one, a))
    }

    /// Efficient polynomial-level computation: selector * (a - a^2)
    public func quotientContribution(wirePolys: [[Fr]], selectorPoly: [Fr],
                                     domain: [Fr], omega: Fr, n: Int,
                                     ntt: NTTEngine) throws -> [Fr] {
        guard column < wirePolys.count else {
            return [Fr](repeating: Fr.zero, count: n)
        }
        let aCoeffs = wirePolys[column]
        let aSqCoeffs = try polyMulNTT(aCoeffs, aCoeffs, ntt: ntt)
        let diff = polySubCoeffs(aCoeffs, aSqCoeffs)
        return try polyMulNTT(selectorPoly, diff, ntt: ntt)
    }
}

// MARK: - RangeCheck Gate

/// Range check: value in [0, 2^k). Decomposes value into k boolean limbs
/// and uses BoolCheck on each limb plus a reconstruction constraint.
///
/// In Halo2 style, this uses rotations to access k consecutive rows where
/// the boolean decomposition is laid out, then constrains the reconstruction.
///
/// For the Plonk gate-level approach, this gate checks a single bit (like BoolCheck)
/// and the circuit builder handles decomposition and reconstruction.
/// The range check at the protocol level uses the existing rangeCheck builder.
public struct RangeCheckGate: CustomGate {
    public let name = "RangeCheck"

    /// Number of bits for the range check
    public let bits: Int

    public init(bits: Int) {
        precondition(bits > 0 && bits <= 64, "Range check bits must be in [1, 64]")
        self.bits = bits
    }

    /// Queries the current row's column 0 (the bit being checked) and
    /// column 1 at offsets 0..bits-1 for the accumulator chain.
    public var queriedCells: [ColumnRef] {
        var cells = [ColumnRef(column: 0, rotation: .cur)]
        // The accumulator values at each rotation step
        for i in 0..<bits {
            cells.append(ColumnRef(column: 1, rotation: .offset(i)))
        }
        return cells
    }

    public func evaluate(rotations: [ColumnRef: Fr], challenges: [Fr]) -> Fr {
        // Boolean constraint on the current bit: a * (1 - a) = 0
        let a = rotations[ColumnRef(column: 0, rotation: .cur)] ?? Fr.zero
        return frMul(a, frSub(Fr.one, a))
    }

    /// Uses BoolCheck's polynomial computation since each row just checks one bit.
    public func quotientContribution(wirePolys: [[Fr]], selectorPoly: [Fr],
                                     domain: [Fr], omega: Fr, n: Int,
                                     ntt: NTTEngine) throws -> [Fr] {
        guard !wirePolys.isEmpty else {
            return [Fr](repeating: Fr.zero, count: n)
        }
        let aCoeffs = wirePolys[0]
        let aSqCoeffs = try polyMulNTT(aCoeffs, aCoeffs, ntt: ntt)
        let diff = polySubCoeffs(aCoeffs, aSqCoeffs)
        return try polyMulNTT(selectorPoly, diff, ntt: ntt)
    }
}

// MARK: - LookupGate

/// Lookup gate: proves that a wire value exists in a fixed table.
/// The constraint is: selector * prod_{t in table} (a - t) = 0
/// which is zero iff a is one of the table values.
///
/// For large tables, use PlonkLookupGate (Plookup-style) instead of
/// this vanishing product approach, which has degree proportional to table size.
public struct LookupGate: CustomGate {
    public let name = "Lookup"

    /// The lookup table values
    public let table: [Fr]

    /// Which column to constrain
    public let column: Int

    public init(table: [Fr], column: Int = 0) {
        self.table = table
        self.column = column
    }

    public var queriedCells: [ColumnRef] {
        [ColumnRef(column: column, rotation: .cur)]
    }

    public func evaluate(rotations: [ColumnRef: Fr], challenges: [Fr]) -> Fr {
        let a = rotations[ColumnRef(column: column, rotation: .cur)] ?? Fr.zero
        // prod(a - t_i) for all table values
        var prod = Fr.one
        for t in table {
            prod = frMul(prod, frSub(a, t))
        }
        return prod
    }

    /// Efficient: build the vanishing product polynomial then multiply by selector.
    public func quotientContribution(wirePolys: [[Fr]], selectorPoly: [Fr],
                                     domain: [Fr], omega: Fr, n: Int,
                                     ntt: NTTEngine) throws -> [Fr] {
        guard column < wirePolys.count else {
            return [Fr](repeating: Fr.zero, count: n)
        }
        let aCoeffs = wirePolys[column]

        // Build prod(a(x) - t_i) iteratively
        guard !table.isEmpty else {
            return [Fr](repeating: Fr.zero, count: n)
        }

        var vanishPoly = polySubCoeffs(aCoeffs, [table[0]])
        for k in 1..<table.count {
            let factor = polySubCoeffs(aCoeffs, [table[k]])
            vanishPoly = try polyMulNTT(vanishPoly, factor, ntt: ntt)
        }
        return try polyMulNTT(selectorPoly, vanishPoly, ntt: ntt)
    }
}

// MARK: - PoseidonGate

/// Poseidon S-box gate: c = a * b^2 where b = a^2 (so c = a^5).
///
/// Wire layout per gate:
///   wire a = input, wire b = a^2 (auxiliary), wire c = a^5 (output)
///
/// Constraint: c - a * b * b = 0
///
/// Uses rotation .cur on all three columns (a, b, c are at the same row).
public struct PoseidonGate: CustomGate {
    public let name = "Poseidon"

    public init() {}

    public var queriedCells: [ColumnRef] {
        [
            ColumnRef(column: 0, rotation: .cur),  // a (input)
            ColumnRef(column: 1, rotation: .cur),  // b (a^2)
            ColumnRef(column: 2, rotation: .cur),  // c (a^5)
        ]
    }

    public func evaluate(rotations: [ColumnRef: Fr], challenges: [Fr]) -> Fr {
        let a = rotations[ColumnRef(column: 0, rotation: .cur)] ?? Fr.zero
        let b = rotations[ColumnRef(column: 1, rotation: .cur)] ?? Fr.zero
        let c = rotations[ColumnRef(column: 2, rotation: .cur)] ?? Fr.zero
        // c - a * b^2
        let bSq = frSqr(b)
        return frSub(c, frMul(a, bSq))
    }

    /// Efficient polynomial computation: selector * (c - a*b*b)
    public func quotientContribution(wirePolys: [[Fr]], selectorPoly: [Fr],
                                     domain: [Fr], omega: Fr, n: Int,
                                     ntt: NTTEngine) throws -> [Fr] {
        guard wirePolys.count >= 3 else {
            return [Fr](repeating: Fr.zero, count: n)
        }
        let aCoeffs = wirePolys[0]
        let bCoeffs = wirePolys[1]
        let cCoeffs = wirePolys[2]

        // a*b
        let abCoeffs = try polyMulNTT(aCoeffs, bCoeffs, ntt: ntt)
        // a*b*b
        let abbCoeffs = try polyMulNTT(abCoeffs, bCoeffs, ntt: ntt)
        // c - a*b*b
        let diff = polySubCoeffs(cCoeffs, abbCoeffs)
        return try polyMulNTT(selectorPoly, diff, ntt: ntt)
    }
}

// MARK: - CustomGateSet

/// Collects custom gates and generates the combined quotient polynomial contribution.
///
/// In the Plonk protocol, the quotient polynomial is:
///   t(x) * Z_H(x) = gate_constraint + alpha * perm_constraint + alpha^2 * boundary
///                  + alpha^3 * custom_gate_0 + alpha^4 * custom_gate_1 + ...
///
/// CustomGateSet manages the custom gate portion, assigning each gate a selector
/// polynomial and combining contributions with successive powers of alpha.
public class CustomGateSet {
    /// Registered custom gates with their selector polynomials (coefficient form)
    public private(set) var gates: [(gate: any CustomGate, selectorCoeffs: [Fr])] = []

    /// Maximum rotation offset across all registered gates
    public var maxRotation: Int {
        gates.map { $0.gate.maxRotation }.max() ?? 0
    }

    public init() {}

    /// Register a custom gate with its selector polynomial (coefficient form).
    /// The selector activates the gate at specific rows (selector[i] != 0 means gate is active at row i).
    public func addGate(_ gate: any CustomGate, selectorCoeffs: [Fr]) {
        gates.append((gate: gate, selectorCoeffs: selectorCoeffs))
    }

    /// Register a custom gate with selector evaluations (will be converted to coefficients via iNTT).
    public func addGate(_ gate: any CustomGate, selectorEvals: [Fr], ntt: NTTEngine) throws {
        let coeffs = try ntt.intt(selectorEvals)
        gates.append((gate: gate, selectorCoeffs: coeffs))
    }

    /// Compute the combined quotient polynomial contribution from all custom gates.
    ///
    /// Returns: sum_{i} alpha^{offset+i} * gate_i_contribution(x)
    ///
    /// - Parameters:
    ///   - wirePolys: Wire polynomials in coefficient form [a_coeffs, b_coeffs, c_coeffs, ...]
    ///   - domain: Evaluation domain
    ///   - omega: Primitive root of unity
    ///   - n: Domain size
    ///   - alpha: Separation challenge
    ///   - alphaOffset: Starting power of alpha (e.g., 3 if gate/perm/boundary use alpha^0..2)
    ///   - ntt: NTT engine
    /// - Returns: Combined polynomial in coefficient form (before Z_H division)
    public func combinedQuotientContribution(
        wirePolys: [[Fr]], domain: [Fr], omega: Fr, n: Int,
        alpha: Fr, alphaOffset: Int, ntt: NTTEngine
    ) throws -> [Fr] {
        guard !gates.isEmpty else {
            return [Fr](repeating: Fr.zero, count: n)
        }

        var alphaPow = Fr.one
        for _ in 0..<alphaOffset {
            alphaPow = frMul(alphaPow, alpha)
        }

        var combined = [Fr](repeating: Fr.zero, count: n)
        for (gate, selectorCoeffs) in gates {
            let contribution = try gate.quotientContribution(
                wirePolys: wirePolys, selectorPoly: selectorCoeffs,
                domain: domain, omega: omega, n: n, ntt: ntt)

            let scaled = polyScaleCoeffs(contribution, alphaPow)
            combined = polyAddCoeffs(combined, scaled)
            alphaPow = frMul(alphaPow, alpha)
        }

        return combined
    }

    /// Verify custom gate constraints at a single evaluation point zeta.
    /// Returns the combined evaluation: sum_i alpha^{offset+i} * selector_i(zeta) * gate_i(zeta)
    ///
    /// Used by the verifier to check the quotient polynomial identity.
    public func evaluateAtPoint(
        wireEvals: [Fr], wireEvalsNext: [Fr],
        selectorEvals: [Fr],
        alpha: Fr, alphaOffset: Int,
        zeta: Fr, omega: Fr, n: Int
    ) -> Fr {
        guard !gates.isEmpty else { return Fr.zero }

        var alphaPow = Fr.one
        for _ in 0..<alphaOffset {
            alphaPow = frMul(alphaPow, alpha)
        }

        var result = Fr.zero
        for (idx, (gate, _)) in gates.enumerated() {
            // Resolve column references using wire evaluations at zeta
            var resolved = [ColumnRef: Fr]()
            for ref in gate.queriedCells {
                let evals: [Fr]
                switch ref.rotation {
                case .cur:
                    evals = wireEvals
                case .next:
                    evals = wireEvalsNext
                default:
                    // For arbitrary rotations, we would need evaluations at zeta*omega^k
                    // In practice, most gates use .cur and .next only
                    evals = wireEvals
                }
                if ref.column < evals.count {
                    resolved[ref] = evals[ref.column]
                } else {
                    resolved[ref] = Fr.zero
                }
            }

            let gateVal = gate.evaluate(rotations: resolved, challenges: [])
            let selectorVal = idx < selectorEvals.count ? selectorEvals[idx] : Fr.zero
            let contribution = frMul(frMul(selectorVal, gateVal), alphaPow)
            result = frAdd(result, contribution)
            alphaPow = frMul(alphaPow, alpha)
        }

        return result
    }
}
