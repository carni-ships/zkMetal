// Jolt Prover — Prove correct VM execution via Lasso structured lookups
//
// Jolt-style proof strategy (Arun et al. 2024):
//   1. For each instruction step: verify instruction(a, b) = result via Lasso lookup
//      into a decomposed instruction table. Each 32-bit operation is decomposed into
//      4 byte-level subtable lookups, checked via sumcheck.
//   2. Lasso range-check proof: all operand values (a, b, result) are in [0, 2^32).
//   3. Operand commitments: (a, b, result) encoded as Fr for each step.
//
// The key Jolt insight: instead of arithmetic circuits per instruction, each instruction
// execution is a single lookup into a precomputed table. Lasso's tensor structure
// means only small (256 or 65536 entry) subtables are needed.

import Foundation

// MARK: - Proof Types

/// Proof for a group of instructions sharing the same opcode, verified via Lasso lookup.
public struct OpcodeProof {
    public let op: JoltOp
    /// Lasso proof for instruction lookup verification (or range-check)
    public let lassoProof: LassoProof?
    /// Algebraic witness: (a, b, result) tuples as Fr (fallback for ops without Lasso tables)
    public let algebraicWitness: [(Fr, Fr, Fr)]?
    /// Number of entries in this proof
    public let count: Int
    /// Whether this is an instruction lookup proof (vs range-check or algebraic fallback)
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
    /// Per-opcode proofs: instruction lookups + range-check
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
    public let instructionRegistry: InstructionTableRegistry

    public init() throws {
        self.lassoEngine = try LassoEngine()
        self.instructionRegistry = InstructionTableRegistry()
    }

    // MARK: - Prove

    /// Prove correct execution of a Jolt trace.
    ///
    /// For each opcode group:
    ///   - If the opcode has a Lasso instruction table: encode all (a, b) pairs as
    ///     lookup values, prove via Lasso that each lookup yields the claimed result.
    ///   - Otherwise: fall back to algebraic witness (a, b, result) tuples.
    /// Additionally, a Lasso range-check proof ensures all values are valid 32-bit.
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

            if let instrTable = instructionRegistry.table(for: op) {
                // Jolt-style: prove via Lasso instruction lookup
                let proof = try proveInstructionLookup(
                    op: op, steps: steps, table: instrTable)
                opcodeProofs.append(proof)
            } else {
                // Fallback: algebraic witness for unsupported ops (MUL, EQ)
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

    /// Prove a batch of instruction executions via Lasso lookup.
    /// Each (a, b, result) is encoded into a lookup value and proved against
    /// the instruction's decomposed subtable.
    private func proveInstructionLookup(
        op: JoltOp, steps: [JoltStep],
        table: any JoltInstructionTable
    ) throws -> OpcodeProof {
        let lassoTable = table.buildLassoTable()

        // Encode each step as a lookup value
        var lookups = [Fr]()
        lookups.reserveCapacity(steps.count)
        for step in steps {
            let encoded = table.encodeLookups(a: step.a, b: step.b, result: step.result)
            lookups.append(contentsOf: encoded)
        }

        // Pad to power of 2 (Lasso requirement)
        var paddedCount = 1
        while paddedCount < lookups.count { paddedCount <<= 1 }
        // Pad with valid lookups (zero op zero = zero for all supported ops)
        let zeroLookup = table.encodeLookups(a: 0, b: 0, result: 0)
        while lookups.count < paddedCount {
            lookups.append(contentsOf: zeroLookup)
        }

        let lassoProof = try lassoEngine.prove(lookups: lookups, table: lassoTable)

        return OpcodeProof(
            op: op, lassoProof: lassoProof,
            algebraicWitness: nil, count: steps.count,
            isInstructionLookup: true)
    }

    // MARK: - Range Check Proof

    /// Prove all operand values are in [0, 2^32) via Lasso structured lookup.
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
