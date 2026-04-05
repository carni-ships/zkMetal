// Jolt Prover — Prove correct VM execution via Lasso structured lookups
//
// Jolt-style proof strategy (Arun et al. 2024):
//   1. For byte-decomposable ops (AND, OR, XOR): verify each instruction execution
//      as a Lasso lookup into decomposed 8-bit subtables. 4 subtables of 65536 entries
//      replace one table of 2^64 entries.
//   2. For non-decomposable ops (ADD, SUB, MUL, shifts, comparisons): algebraic
//      witness binding (a, b, result) tuples with semantic re-execution check.
//   3. Lasso range-check proof: all operand values (a, b, result) are in [0, 2^32).
//   4. Operand commitments: (a, b, result) encoded as Fr for each step.

import Foundation

// MARK: - Proof Types

/// Proof for a group of instructions sharing the same opcode.
/// Either verified via Lasso lookup (bitwise ops) or algebraic witness (others).
public struct OpcodeProof {
    public let op: JoltOp
    /// Lasso proof for instruction lookup verification (or range-check)
    public let lassoProof: LassoProof?
    /// Algebraic witness: (a, b, result) tuples as Fr (for non-Lasso-verified ops)
    public let algebraicWitness: [(Fr, Fr, Fr)]?
    /// Number of entries in this proof
    public let count: Int
    /// Whether this is a Lasso instruction lookup proof (vs range-check or algebraic)
    public let isInstructionLookup: Bool

    public init(op: JoltOp, lassoProof: LassoProof?,
                algebraicWitness: [(Fr, Fr, Fr)]?, count: Int,
                isInstructionLookup: Bool = false) {
        self.op = op
        self.lassoProof = lassoProof
        self.algebraicWitness = algebraicWitness
        self.count = count
        self.isInstructionLookup = isInstructionLookup
    }
}

/// Complete Jolt VM proof
public struct JoltProof {
    /// Total number of instructions proved
    public let numInstructions: Int
    /// Per-opcode proofs: Lasso instruction lookups + algebraic witnesses + range-check
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
    /// For each opcode group:
    ///   - Byte-decomposable ops (AND, OR, XOR): Lasso instruction lookup proof.
    ///     Each 32-bit operation decomposes into 4 byte-level subtable lookups.
    ///   - All other ops: algebraic witness (a, b, result) binding.
    /// Plus: Lasso range-check on all operand values.
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

        // Build per-opcode proofs
        var opcodeProofs = [OpcodeProof]()

        for op in JoltOp.allCases {
            guard let steps = stepsByOp[op], !steps.isEmpty else { continue }

            if InstructionSubtable.isLassoVerified(op) {
                // Jolt-style: prove via Lasso instruction lookup
                let proof = try proveInstructionLookup(op: op, steps: steps)
                opcodeProofs.append(proof)
            } else {
                // Algebraic witness for non-decomposable ops
                let witness = steps.map { step -> (Fr, Fr, Fr) in
                    (frFromInt(UInt64(step.a)),
                     frFromInt(UInt64(step.b)),
                     frFromInt(UInt64(step.result)))
                }
                opcodeProofs.append(OpcodeProof(
                    op: op, lassoProof: nil,
                    algebraicWitness: witness, count: steps.count,
                    isInstructionLookup: false))
            }
        }

        // Lasso range-check proof: all operand values are in [0, 2^32)
        let rangeProof = try proveRangeCheck(trace: trace)
        opcodeProofs.append(rangeProof)

        return JoltProof(numInstructions: n, opcodeProofs: opcodeProofs,
                          operandCommitments: operandCommitments)
    }

    // MARK: - Instruction Lookup Proof

    /// Prove a batch of bitwise instruction executions via Lasso lookup.
    /// Each (a, b) -> result is verified by decomposing into byte-level subtable lookups.
    private func proveInstructionLookup(
        op: JoltOp, steps: [JoltStep]
    ) throws -> OpcodeProof {
        let (table, lookups) = InstructionSubtable.buildTable(op: op, steps: steps)
        let lassoProof = try lassoEngine.prove(lookups: lookups, table: table)

        return OpcodeProof(
            op: op, lassoProof: lassoProof,
            algebraicWitness: nil, count: steps.count,
            isInstructionLookup: true)
    }

    // MARK: - Range Check Proof

    /// Prove all operand values are in [0, 2^32) via Lasso structured lookup.
    /// Each 32-bit value is decomposed into 4 byte subtables (256 entries each).
    private func proveRangeCheck(trace: JoltTrace) throws -> OpcodeProof {
        var values = [Fr]()
        values.reserveCapacity(trace.steps.count * 3)
        for step in trace.steps {
            values.append(frFromInt(UInt64(step.a)))
            values.append(frFromInt(UInt64(step.b)))
            values.append(frFromInt(UInt64(step.result)))
        }

        // Pad to power of 2
        var paddedCount = 1
        while paddedCount < values.count { paddedCount <<= 1 }
        while values.count < paddedCount {
            values.append(frFromInt(0))
        }

        let rangeTable = LassoTable.rangeCheck(bits: 32, chunks: 4)
        let lassoProof = try lassoEngine.prove(lookups: values, table: rangeTable)

        // Sentinel: isInstructionLookup=false, lassoProof!=nil, algebraicWitness==nil
        return OpcodeProof(op: .eq, lassoProof: lassoProof,
                           algebraicWitness: nil, count: values.count,
                           isInstructionLookup: false)
    }
}
