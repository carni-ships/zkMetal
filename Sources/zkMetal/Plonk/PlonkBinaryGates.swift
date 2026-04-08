// PlonkBinaryGates -- Binary decomposition and comparison gates for Plonk circuits
//
// Provides custom gates for binary operations:
//   - BinaryDecomposeGate: constrain x = sum(2^i * b_i) with boolean b_i
//   - ComparisonGate: constrain a < b using binary decomposition of (b - a)
//   - ConditionalSelectGate: constrain out = cond ? a : b (multiplexer)
//
// These gates are fundamental building blocks for range proofs, comparison
// operations, and conditional logic in zero-knowledge circuits.

import Foundation
import NeonFieldOps

// MARK: - BinaryDecomposeGate

/// Constrains x = sum_{i=0}^{k-1} 2^i * b_i where each b_i is boolean.
///
/// This gate uses multiple rows with rotations to chain the decomposition.
/// The accumulator is checked against the original value at the final step.
///
/// Wire layout (k+1 rows):
///   Row 0: col0 = x (the value being decomposed), col1 = 0 (initial accumulator)
///   Row i (1..k): col0 = b_{i-1} (bit), col1 = acc_i (running sum)
///
/// Constraints per row i (1..k):
///   1. b_{i-1} * (1 - b_{i-1}) = 0   (boolean)
///   2. acc_i = acc_{i-1} + 2^{i-1} * b_{i-1}   (accumulation)
///
/// Final constraint:
///   acc_k = x  (reconstruction matches original value)
///
/// For efficiency, this gate checks one bit per row using cur/prev rotations.
public struct BinaryDecomposeGate: CustomGate {
    public let name = "BinaryDecompose"

    /// Number of bits in the decomposition
    public let bits: Int

    /// Power of 2 for the current bit position (baked in per-row gate instance)
    public let bitPower: Fr

    /// Bit position index (0-based)
    public let bitIndex: Int

    /// Whether this is the final row (adds reconstruction check)
    public let isFinal: Bool

    public init(bits: Int, bitIndex: Int, isFinal: Bool = false) {
        precondition(bits > 0 && bits <= 254, "Decomposition bits must be in [1, 254]")
        precondition(bitIndex >= 0 && bitIndex < bits)
        self.bits = bits
        self.bitIndex = bitIndex
        self.isFinal = isFinal

        // Compute 2^bitIndex in the field
        var pow = Fr.one
        let two = frAdd(Fr.one, Fr.one)
        for _ in 0..<bitIndex {
            pow = frMul(pow, two)
        }
        self.bitPower = pow
    }

    public var queriedCells: [ColumnRef] {
        var cells = [
            ColumnRef(column: 0, rotation: .cur),   // b_i (current bit)
            ColumnRef(column: 1, rotation: .cur),   // acc_i (current accumulator)
            ColumnRef(column: 1, rotation: .prev),  // acc_{i-1} (previous accumulator)
        ]
        if isFinal {
            // Need the original value x from the first row
            cells.append(ColumnRef(column: 0, rotation: .offset(-bitIndex - 1)))
        }
        return cells
    }

    public func evaluate(rotations: [ColumnRef: Fr], challenges: [Fr]) -> Fr {
        let bit = rotations[ColumnRef(column: 0, rotation: .cur)] ?? Fr.zero
        let accCur = rotations[ColumnRef(column: 1, rotation: .cur)] ?? Fr.zero
        let accPrev = rotations[ColumnRef(column: 1, rotation: .prev)] ?? Fr.zero

        // Constraint 1: bit * (1 - bit) = 0  (boolean check)
        let boolCheck = frMul(bit, frSub(Fr.one, bit))

        // Constraint 2: acc_cur - acc_prev - 2^bitIndex * bit = 0
        let expected = frAdd(accPrev, frMul(bitPower, bit))
        let accCheck = frSub(accCur, expected)

        var result = frAdd(frMul(boolCheck, boolCheck), frMul(accCheck, accCheck))

        // Final constraint: acc_k = x
        if isFinal {
            let x = rotations[ColumnRef(column: 0, rotation: .offset(-bitIndex - 1))] ?? Fr.zero
            let finalCheck = frSub(accCur, x)
            result = frAdd(result, frMul(finalCheck, finalCheck))
        }

        return result
    }
}

// MARK: - BinaryDecomposeHelper

/// Helper to expand a binary decomposition into individual per-bit gates
/// and allocate the necessary wire variables.
public struct BinaryDecomposeHelper {

    /// Expand a binary decomposition of `valueVar` into `bits` individual gate descriptors.
    ///
    /// - Parameters:
    ///   - compiler: Constraint compiler for variable allocation
    ///   - valueVar: Variable index of the value to decompose
    ///   - bits: Number of bits
    ///   - selectorIndex: Selector column for decomposition gates
    /// - Returns: (bitVars, gateDescs) where bitVars[i] is the variable for bit i (LSB=0)
    public static func expand(
        compiler: PlonkConstraintCompiler,
        valueVar: Int,
        bits: Int,
        selectorIndex: Int
    ) -> (bitVars: [Int], gateDescs: [CustomGateDesc]) {
        let bitVars = compiler.addVariables(bits)
        let accVars = compiler.addVariables(bits + 1)  // acc[0] = 0, acc[bits] = x

        var gateDescs = [CustomGateDesc]()

        // Initial row: x and acc=0
        // This is handled by wire assignments (accVars[0] constrained to 0 externally)

        for i in 0..<bits {
            let gate = BinaryDecomposeGate(bits: bits, bitIndex: i, isFinal: i == bits - 1)
            // Wire layout: col0 = bit, col1 = acc_cur, with prev row having acc_prev
            // The rotation-based access means we lay out rows sequentially
            var wires: [[Int]]
            if i == 0 {
                // First bit row also references the value row
                wires = [
                    [valueVar, accVars[0]],        // "prev" row (initial)
                    [bitVars[i], accVars[i + 1]],  // current row
                ]
            } else {
                wires = [
                    [bitVars[i - 1], accVars[i]],  // prev row
                    [bitVars[i], accVars[i + 1]],   // current row
                ]
            }

            gateDescs.append(CustomGateDesc(
                gate: gate, wires: wires, selectorIndex: selectorIndex))
        }

        return (bitVars: bitVars, gateDescs: gateDescs)
    }
}

// MARK: - ComparisonGate

/// Constrains a < b by proving that (b - a - 1) fits in k bits (is non-negative
/// and less than 2^k).
///
/// The comparison a < b over a field is equivalent to:
///   diff = b - a - 1
///   diff >= 0 and diff < 2^k  (i.e., diff has a k-bit binary decomposition)
///
/// This gate combines a subtraction constraint with a binary decomposition.
///
/// Wire layout (single row for the subtraction + decomposition rows):
///   Row 0: col0 = a, col1 = b, col2 = diff (= b - a - 1)
///
/// The binary decomposition of `diff` is handled by BinaryDecomposeGate instances
/// generated via `expandGates()`.
///
/// Note: This only works for values known to be less than 2^k (the comparison is
/// modular otherwise). For BN254, k should be at most 253.
public struct ComparisonGate: CustomGate {
    public let name = "Comparison"

    /// Number of bits for the range (both a and b must be < 2^bits)
    public let bits: Int

    public init(bits: Int) {
        precondition(bits > 0 && bits <= 253, "Comparison bits must be in [1, 253]")
        self.bits = bits
    }

    public var queriedCells: [ColumnRef] {
        [
            ColumnRef(column: 0, rotation: .cur),  // a
            ColumnRef(column: 1, rotation: .cur),  // b
            ColumnRef(column: 2, rotation: .cur),  // diff = b - a - 1
        ]
    }

    public func evaluate(rotations: [ColumnRef: Fr], challenges: [Fr]) -> Fr {
        let a = rotations[ColumnRef(column: 0, rotation: .cur)] ?? Fr.zero
        let b = rotations[ColumnRef(column: 1, rotation: .cur)] ?? Fr.zero
        let diff = rotations[ColumnRef(column: 2, rotation: .cur)] ?? Fr.zero

        // Constraint: diff = b - a - 1
        let expected = frSub(frSub(b, a), Fr.one)
        return frSub(diff, expected)
    }

    /// Expand the comparison into a subtraction gate + binary decomposition.
    ///
    /// - Parameters:
    ///   - compiler: Constraint compiler
    ///   - aVar: Variable index for a
    ///   - bVar: Variable index for b
    ///   - selectorComp: Selector for the comparison constraint
    ///   - selectorDecomp: Selector for binary decomposition
    /// - Returns: Gate descriptors for the comparison and decomposition
    public func expandGates(
        compiler: PlonkConstraintCompiler,
        aVar: Int, bVar: Int,
        selectorComp: Int,
        selectorDecomp: Int
    ) -> [CustomGateDesc] {
        let diffVar = compiler.addVariable()

        var result = [CustomGateDesc]()

        // Subtraction constraint: diff = b - a - 1
        result.append(CustomGateDesc(
            gate: self, wires: [[aVar, bVar, diffVar]], selectorIndex: selectorComp))

        // Binary decomposition of diff (proves diff >= 0 and diff < 2^bits)
        let (_, decompGates) = BinaryDecomposeHelper.expand(
            compiler: compiler, valueVar: diffVar, bits: bits,
            selectorIndex: selectorDecomp)
        result.append(contentsOf: decompGates)

        return result
    }
}

// MARK: - ConditionalSelectGate

/// Constrains out = cond ? a : b (2-to-1 multiplexer).
///
/// The algebraic constraint is:
///   out = cond * a + (1 - cond) * b
///   out = b + cond * (a - b)
///
/// Additionally, cond must be boolean: cond * (1 - cond) = 0
///
/// Wire layout (single row):
///   col0 = cond, col1 = a, col2 = b
///   Next row col0 = out
///
/// Constraints:
///   1. cond * (1 - cond) = 0   (boolean)
///   2. out - b - cond * (a - b) = 0   (selection)
public struct ConditionalSelectGate: CustomGate {
    public let name = "ConditionalSelect"

    public init() {}

    public var queriedCells: [ColumnRef] {
        [
            ColumnRef(column: 0, rotation: .cur),   // cond
            ColumnRef(column: 1, rotation: .cur),   // a
            ColumnRef(column: 2, rotation: .cur),   // b
            ColumnRef(column: 0, rotation: .next),  // out
        ]
    }

    public func evaluate(rotations: [ColumnRef: Fr], challenges: [Fr]) -> Fr {
        let cond = rotations[ColumnRef(column: 0, rotation: .cur)] ?? Fr.zero
        let a = rotations[ColumnRef(column: 1, rotation: .cur)] ?? Fr.zero
        let b = rotations[ColumnRef(column: 2, rotation: .cur)] ?? Fr.zero
        let out = rotations[ColumnRef(column: 0, rotation: .next)] ?? Fr.zero

        // Constraint 1: cond * (1 - cond) = 0
        let boolCheck = frMul(cond, frSub(Fr.one, cond))

        // Constraint 2: out - b - cond * (a - b) = 0
        let selectCheck = frSub(frSub(out, b), frMul(cond, frSub(a, b)))

        return frAdd(frMul(boolCheck, boolCheck), frMul(selectCheck, selectCheck))
    }

    /// Efficient polynomial-level computation.
    public func quotientContribution(wirePolys: [[Fr]], selectorPoly: [Fr],
                                     domain: [Fr], omega: Fr, n: Int,
                                     ntt: NTTEngine) throws -> [Fr] {
        // Use the default pointwise evaluation since we need rotation support
        // The default implementation in CustomGate handles this correctly.
        guard wirePolys.count >= 3 else {
            return [Fr](repeating: Fr.zero, count: n)
        }

        // Evaluate wire polynomials on the domain
        var wireEvals = [[Fr]]()
        for col in 0..<wirePolys.count {
            var padded = wirePolys[col]
            if padded.count < n {
                padded += [Fr](repeating: Fr.zero, count: n - padded.count)
            }
            wireEvals.append(try ntt.ntt(Array(padded.prefix(n))))
        }

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

        var gateEvals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            if frEqual(selectorEvals[i], Fr.zero) { continue }

            let cond = wireEvals[0][i]
            let a = wireEvals[1][i]
            let b = wireEvals[2][i]
            let out = wireEvals[0][(i + 1) % n]  // next row, col 0

            let boolCheck = frMul(cond, frSub(Fr.one, cond))
            let selectCheck = frSub(frSub(out, b), frMul(cond, frSub(a, b)))
            let val = frAdd(frMul(boolCheck, boolCheck), frMul(selectCheck, selectCheck))
            gateEvals[i] = frMul(selectorEvals[i], val)
        }

        return try ntt.intt(gateEvals)
    }
}
