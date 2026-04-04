// Jolt Verifier — Verify correct VM execution proof
//
// Verification steps:
//   1. Check operand commitments match claimed program execution
//   2. Verify algebraic witnesses for all instruction types
//   3. Verify Lasso range-check proof on all operand values (32-bit range)

import Foundation

extension JoltEngine {

    // MARK: - Verify

    /// Verify a Jolt execution proof against a program.
    /// Re-executes the program to get expected operand values, then checks:
    ///   - Operand commitments match execution
    ///   - Each opcode has a valid algebraic witness
    ///   - Lasso range-check covers all operand values
    public func verify(proof: JoltProof, program: [JoltInstruction],
                       numRegisters: Int = 32, initialRegs: [UInt32]? = nil) -> Bool {
        // Re-execute to get expected trace
        let expectedTrace = joltExecute(program: program, numRegisters: numRegisters,
                                         initialRegs: initialRegs)

        guard proof.numInstructions == program.count else { return false }
        guard proof.operandCommitments.count == program.count else { return false }

        // Check operand commitments match execution
        for (i, step) in expectedTrace.steps.enumerated() {
            let (aFr, bFr, rFr) = proof.operandCommitments[i]
            let expectedA = frFromInt(UInt64(step.a))
            let expectedB = frFromInt(UInt64(step.b))
            let expectedR = frFromInt(UInt64(step.result))

            if !frEqual(aFr, expectedA) || !frEqual(bFr, expectedB) || !frEqual(rFr, expectedR) {
                return false
            }
        }

        // Group expected steps by opcode
        var stepsByOp = [JoltOp: [JoltStep]]()
        for step in expectedTrace.steps {
            stepsByOp[step.op, default: []].append(step)
        }

        var hasRangeCheck = false

        // Verify each opcode proof
        for opcodeProof in proof.opcodeProofs {
            let op = opcodeProof.op

            // Range-check proof: identified by having a Lasso proof and no algebraic witness
            if opcodeProof.lassoProof != nil && opcodeProof.algebraicWitness == nil {
                if !verifyRangeCheck(proof: opcodeProof, trace: expectedTrace) {
                    return false
                }
                hasRangeCheck = true
                continue
            }

            // Algebraic witness verification
            guard let steps = stepsByOp[op], !steps.isEmpty else { return false }
            guard opcodeProof.count == steps.count else { return false }

            if !verifyAlgebraic(op: op, proof: opcodeProof, steps: steps) {
                return false
            }
        }

        // Must have a range-check proof
        if !hasRangeCheck { return false }

        // Check all opcodes with steps are covered
        for (op, steps) in stepsByOp where !steps.isEmpty {
            let hasProof = proof.opcodeProofs.contains { $0.op == op && $0.algebraicWitness != nil }
            if !hasProof { return false }
        }

        return true
    }

    // MARK: - Algebraic Verification

    /// Verify algebraic witness: each (a, b, result) must match expected execution.
    private func verifyAlgebraic(op: JoltOp, proof: OpcodeProof,
                                  steps: [JoltStep]) -> Bool {
        guard let witness = proof.algebraicWitness else { return false }
        guard witness.count == steps.count else { return false }

        for (i, (aFr, bFr, rFr)) in witness.enumerated() {
            let step = steps[i]
            let expectedA = frFromInt(UInt64(step.a))
            let expectedB = frFromInt(UInt64(step.b))
            let expectedR = frFromInt(UInt64(step.result))

            if !frEqual(aFr, expectedA) { return false }
            if !frEqual(bFr, expectedB) { return false }
            if !frEqual(rFr, expectedR) { return false }

            if step.result != executeOp(op, step.a, step.b) { return false }
        }
        return true
    }

    // MARK: - Range Check Verification

    /// Verify the range-check Lasso proof on all operand values.
    private func verifyRangeCheck(proof: OpcodeProof, trace: JoltTrace) -> Bool {
        guard let lassoProof = proof.lassoProof else { return false }

        var values = [Fr]()
        values.reserveCapacity(trace.steps.count * 3)
        for step in trace.steps {
            values.append(frFromInt(UInt64(step.a)))
            values.append(frFromInt(UInt64(step.b)))
            values.append(frFromInt(UInt64(step.result)))
        }

        var paddedCount = 1
        while paddedCount < values.count { paddedCount <<= 1 }
        while values.count < paddedCount {
            values.append(frFromInt(0))
        }

        guard proof.count == paddedCount else { return false }

        let rangeTable = LassoTable.rangeCheck(bits: 32, chunks: 4)
        do {
            return try lassoEngine.verify(proof: lassoProof, lookups: values, table: rangeTable)
        } catch {
            return false
        }
    }

    // MARK: - Helpers

    /// Execute a single operation on 32-bit values (for verification).
    private func executeOp(_ op: JoltOp, _ a: UInt32, _ b: UInt32) -> UInt32 {
        switch op {
        case .add:  return a &+ b
        case .sub:  return a &- b
        case .mul:  return a &* b
        case .and_: return a & b
        case .or_:  return a | b
        case .xor_: return a ^ b
        case .shl:  return a << (b & 31)
        case .shr:  return a >> (b & 31)
        case .lt:   return a < b ? 1 : 0
        case .eq:   return a == b ? 1 : 0
        }
    }
}
