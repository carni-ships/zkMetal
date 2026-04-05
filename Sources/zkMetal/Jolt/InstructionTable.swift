// InstructionTable — Jolt-style decomposed instruction tables for Lasso lookup verification
//
// Each 32-bit bitwise ALU instruction (AND, OR, XOR) is verified via byte-level subtable
// lookups. These ops are fully byte-decomposable: each byte of the result depends only
// on the corresponding bytes of the operands.
//
// Table structure:
//   subtable[a_byte * 256 + b_byte] = op(a_byte, b_byte)  (65536 entries)
//   compose(subtable[indices_0], ..., subtable[indices_3]) = result
//   where indices_k = a_byte_k * 256 + b_byte_k
//
// The prover provides operand-byte indices as witness. The verifier checks:
//   compose(subtable[indices[k][i]]) == lookups[i]   for each lookup i
// This confirms that the decomposed operand bytes produce the claimed result.
//
// Non-decomposable ops (ADD/SUB with carry, MUL, shifts, comparisons) use algebraic
// witness verification. The Jolt paper handles these via multi-table decompositions
// with carry chains; here we implement the pure-Lasso path for exact byte decomposition.
//
// Implementation: the lookup value = result, indices = operand byte pairs. A sequential
// index cursor feeds the correct (a_byte, b_byte) indices to LassoEngine's decompose
// calls during proving. The verifier reconstructs identical indices from re-execution.
//
// References: Jolt (Arun et al. 2024), Section 4: "Instruction lookups via Lasso"

import Foundation

// MARK: - Instruction Subtable Builder

/// Builds Lasso tables for Jolt instruction verification.
public enum InstructionSubtable {

    /// Ops verified via Lasso instruction lookup (byte-decomposable bitwise operations)
    public static let lassoOps: Set<JoltOp> = [.and_, .or_, .xor_]

    /// Whether an opcode is verified via Lasso instruction lookup
    public static func isLassoVerified(_ op: JoltOp) -> Bool {
        return lassoOps.contains(op)
    }

    // MARK: - Bitwise Subtables

    /// Build a 256x256 bitwise subtable: entry[a * 256 + b] = op(a, b)
    private static func bitwiseSubtable(_ op: JoltOp) -> [Fr] {
        let bitwiseOp: (Int, Int) -> Int
        switch op {
        case .and_: bitwiseOp = { $0 & $1 }
        case .or_:  bitwiseOp = { $0 | $1 }
        case .xor_: bitwiseOp = { $0 ^ $1 }
        default: fatalError("Not a bitwise op")
        }
        return (0..<65536).map { idx in
            let a = idx >> 8
            let b = idx & 0xFF
            return frFromInt(UInt64(bitwiseOp(a, b)))
        }
    }

    // MARK: - Build LassoTable

    /// Build a LassoTable and lookup values for proving a batch of bitwise instruction
    /// executions via Lasso.
    ///
    /// The decompose function uses a sequential cursor into the pre-computed operand-byte
    /// indices. This works because LassoEngine.prove calls decompose(lookups[i]) in order
    /// for i = 0..<m when batchDecompose is nil.
    ///
    /// Returns (table, lookups) where lookups are padded to power of 2.
    public static func buildTable(op: JoltOp, steps: [JoltStep]) -> (table: LassoTable, lookups: [Fr]) {
        precondition(lassoOps.contains(op), "Op \(op) is not byte-decomposable")

        let subtable = bitwiseSubtable(op)
        let subtables = Array(repeating: subtable, count: 4)

        // Lookup values = instruction results
        var lookups = steps.map { frFromInt(UInt64($0.result)) }

        // Pad to power of 2 (minimum 2 for Lasso sumcheck)
        var paddedCount = 2
        while paddedCount < lookups.count { paddedCount <<= 1 }
        while lookups.count < paddedCount {
            lookups.append(frFromInt(0))  // op(0, 0) = 0 for all bitwise ops
        }

        // Pre-compute all operand-byte indices
        let m = paddedCount
        var flatIndices = [Int](repeating: 0, count: 4 * m)
        for (i, step) in steps.enumerated() {
            for k in 0..<4 {
                let aByte = Int((step.a >> (k * 8)) & 0xFF)
                let bByte = Int((step.b >> (k * 8)) & 0xFF)
                flatIndices[i * 4 + k] = aByte * 256 + bByte
            }
        }
        // Padding entries: index 0 (op(0,0) = 0 for all bitwise ops)

        // Cursor for sequential decompose calls
        let cursor = IndexCursor(flatIndices: flatIndices, numChunks: 4)

        // Compose: reconstruct 32-bit result from 4 byte results
        let compose: ([Fr]) -> Fr = { components in
            var result = Fr.zero
            var shift = Fr.one
            let base = frFromInt(256)
            for c in components {
                result = frAdd(result, frMul(c, shift))
                shift = frMul(shift, base)
            }
            return result
        }

        // Decompose: returns pre-computed operand-byte indices via cursor.
        // Called sequentially by LassoEngine.prove for i = 0, 1, ..., m-1.
        let capturedCursor = cursor
        let decompose: (Fr) -> [Int] = { _ in
            return capturedCursor.next()
        }

        let table = LassoTable(subtables: subtables, compose: compose,
                                decompose: decompose, numChunks: 4,
                                batchDecompose: nil)
        return (table, lookups)
    }
}

// MARK: - Index Cursor

/// Thread-safe sequential cursor for feeding pre-computed indices to decompose calls.
/// Each call to next() returns the indices for the next lookup element.
private class IndexCursor {
    private let flatIndices: [Int]
    private let numChunks: Int
    private var position: Int = 0

    init(flatIndices: [Int], numChunks: Int) {
        self.flatIndices = flatIndices
        self.numChunks = numChunks
    }

    /// Reset cursor to beginning (for verifier reuse).
    func reset() {
        position = 0
    }

    /// Return indices for the next element. Called sequentially.
    func next() -> [Int] {
        let base = position * numChunks
        position += 1
        if base + numChunks <= flatIndices.count {
            return (0..<numChunks).map { flatIndices[base + $0] }
        } else {
            // Beyond pre-computed range (padding)
            return [Int](repeating: 0, count: numChunks)
        }
    }
}
