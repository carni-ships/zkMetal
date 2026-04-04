// Jolt Prover — Prove correct VM execution via Lasso structured lookups
//
// Proof strategy:
//   1. For each opcode group: algebraic witness (a, b, result) tuples
//      The verifier re-executes to check each tuple satisfies the instruction semantics.
//   2. Lasso range-check proof: all operand values (a, b, result) are in [0, 2^32).
//      Uses Lasso with 4 byte subtables (256 entries each), decomposing each 32-bit
//      value into 4 bytes. This is the core Lasso/Jolt insight: structured lookup
//      into tensor-decomposed tables dramatically reduces prover work.
//   3. Operand commitments: (a, b, result) encoded as Fr for each step,
//      binding the prover to a specific execution trace.

import Foundation

// MARK: - Proof Types

/// Proof for a group of instructions sharing the same opcode
public struct OpcodeProof {
    public let op: JoltOp
    /// Lasso proof (used for range-check; nil for algebraic-only proofs)
    public let lassoProof: LassoProof?
    /// Algebraic witness: (a, b, result) tuples as Fr (nil for Lasso-only proofs)
    public let algebraicWitness: [(Fr, Fr, Fr)]?
    /// Number of entries in this proof
    public let count: Int

    public init(op: JoltOp, lassoProof: LassoProof?,
                algebraicWitness: [(Fr, Fr, Fr)]?, count: Int) {
        self.op = op
        self.lassoProof = lassoProof
        self.algebraicWitness = algebraicWitness
        self.count = count
    }
}

/// Complete Jolt VM proof
public struct JoltProof {
    /// Total number of instructions proved
    public let numInstructions: Int
    /// Per-opcode algebraic proofs + one Lasso range-check proof
    public let opcodeProofs: [OpcodeProof]
    /// Operand commitments: (a, b, result) encoded as Fr for each step
    public let operandCommitments: [(Fr, Fr, Fr)]

    public init(numInstructions: Int, opcodeProofs: [OpcodeProof],
                operandCommitments: [(Fr, Fr, Fr)]) {
        self.numInstructions = numInstructions
        self.opcodeProofs = opcodeProofs
        self.operandCommitments = operandCommitments
    }
}

// MARK: - Jolt Engine

public class JoltEngine {
    public static let version = Versions.joltVM
    public let lassoEngine: LassoEngine

    public init() throws {
        self.lassoEngine = try LassoEngine()
    }

    // MARK: - Prove

    /// Prove correct execution of a Jolt trace.
    ///
    /// Produces:
    ///   - Per-opcode algebraic witnesses binding (a, b, result) tuples
    ///   - Lasso range-check proof ensuring all values are valid 32-bit
    ///   - Operand commitments for the full trace
    public func prove(trace: JoltTrace) throws -> JoltProof {
        let n = trace.steps.count
        precondition(n > 0, "Empty trace")

        // Build operand commitments
        var operandCommitments = [(Fr, Fr, Fr)]()
        operandCommitments.reserveCapacity(n)
        for step in trace.steps {
            operandCommitments.append((
                frFromInt(UInt64(step.a)),
                frFromInt(UInt64(step.b)),
                frFromInt(UInt64(step.result))
            ))
        }

        // Group steps by opcode
        var stepsByOp = [JoltOp: [JoltStep]]()
        for step in trace.steps {
            stepsByOp[step.op, default: []].append(step)
        }

        // Build per-opcode algebraic witness proofs
        var opcodeProofs = [OpcodeProof]()

        for op in JoltOp.allCases {
            guard let steps = stepsByOp[op], !steps.isEmpty else { continue }

            let witness = steps.map { step -> (Fr, Fr, Fr) in
                (frFromInt(UInt64(step.a)),
                 frFromInt(UInt64(step.b)),
                 frFromInt(UInt64(step.result)))
            }
            opcodeProofs.append(OpcodeProof(
                op: op, lassoProof: nil,
                algebraicWitness: witness, count: steps.count))
        }

        // Lasso range-check proof: all operand values are in [0, 2^32)
        // Decomposes each 32-bit value into 4 byte-level subtable lookups
        let rangeProof = try proveRangeCheck(trace: trace)
        opcodeProofs.append(rangeProof)

        return JoltProof(numInstructions: n, opcodeProofs: opcodeProofs,
                          operandCommitments: operandCommitments)
    }

    // MARK: - Range Check Proof

    /// Prove all operand values are in [0, 2^32) via Lasso structured lookup.
    /// Each 32-bit value is decomposed into 4 byte subtables (256 entries each).
    /// This is the canonical Lasso use case from the Jolt paper.
    private func proveRangeCheck(trace: JoltTrace) throws -> OpcodeProof {
        var values = [Fr]()
        values.reserveCapacity(trace.steps.count * 3)
        for step in trace.steps {
            values.append(frFromInt(UInt64(step.a)))
            values.append(frFromInt(UInt64(step.b)))
            values.append(frFromInt(UInt64(step.result)))
        }

        // Pad to power of 2 (Lasso requirement)
        var paddedCount = 1
        while paddedCount < values.count { paddedCount <<= 1 }
        while values.count < paddedCount {
            values.append(frFromInt(0))
        }

        let rangeTable = LassoTable.rangeCheck(bits: 32, chunks: 4)
        let lassoProof = try lassoEngine.prove(lookups: values, table: rangeTable)

        // Sentinel: op=.eq with Lasso proof and no algebraic witness identifies range check
        return OpcodeProof(op: .eq, lassoProof: lassoProof,
                           algebraicWitness: nil, count: values.count)
    }
}
