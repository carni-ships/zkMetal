// Jolt Verifier — Verify correct VM execution proof via Lasso lookup verification
//
// Verification steps:
//   1. Re-execute program to get expected operand values
//   2. Check operand commitments match expected execution
//   3. For each opcode with a Lasso instruction lookup proof:
//      - Re-encode the expected (a, b, result) as lookup values
//      - Verify the Lasso proof against the instruction's decomposed table
//   4. For fallback opcodes: verify algebraic witnesses match expected values
//   5. Verify Lasso range-check proof on all operand values (32-bit range)

import Foundation

extension JoltEngine {

    // MARK: - Verify

    /// Verify a Jolt execution proof against a program.
    /// Checks Lasso instruction lookup proofs for supported ops,
    /// algebraic witnesses for unsupported ops, and range-check for all values.
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
        var verifiedOps = Set<JoltOp>()

        // Verify each opcode proof
        for opcodeProof in proof.opcodeProofs {
            let op = opcodeProof.op

            // Range-check proof: identified by having a Lasso proof, no algebraic witness,
            // and not being an instruction lookup
            if opcodeProof.lassoProof != nil && opcodeProof.algebraicWitness == nil
                && !opcodeProof.isInstructionLookup {
                if !verifyRangeCheck(proof: opcodeProof, trace: expectedTrace) {
                    return false
                }
                hasRangeCheck = true
                continue
            }

            // Instruction lookup proof (Jolt-style via Lasso)
            if opcodeProof.isInstructionLookup {
                guard let steps = stepsByOp[op], !steps.isEmpty else { return false }
                guard opcodeProof.count == steps.count else { return false }
                if !verifyInstructionLookup(op: op, proof: opcodeProof, steps: steps) {
                    return false
                }
                verifiedOps.insert(op)
                continue
            }

            // Algebraic witness verification (fallback)
            guard let steps = stepsByOp[op], !steps.isEmpty else { return false }
            guard opcodeProof.count == steps.count else { return false }

            if !verifyAlgebraic(op: op, proof: opcodeProof, steps: steps) {
                return false
            }
            verifiedOps.insert(op)
        }

        // Must have a range-check proof
        if !hasRangeCheck { return false }

        // Check all opcodes with steps are covered
        for (op, steps) in stepsByOp where !steps.isEmpty {
            if !verifiedOps.contains(op) { return false }
        }

        return true
    }

    // MARK: - Instruction Lookup Verification

    /// Verify a Lasso instruction lookup proof for a batch of instructions.
    /// Re-encodes the expected (a, b, result) tuples and verifies the Lasso proof.
    private func verifyInstructionLookup(op: JoltOp, proof: OpcodeProof,
                                          steps: [JoltStep]) -> Bool {
        guard let lassoProof = proof.lassoProof else { return false }
        guard let instrTable = instructionRegistry.table(for: op) else { return false }

        let lassoTable = instrTable.buildLassoTable()

        // Re-encode lookups from expected execution
        var lookups = [Fr]()
        lookups.reserveCapacity(steps.count)
        for step in steps {
            // First verify the instruction semantics
            if step.result != executeOp(op, step.a, step.b) { return false }
            let encoded = instrTable.encodeLookups(a: step.a, b: step.b, result: step.result)
            lookups.append(contentsOf: encoded)
        }

        // Pad to power of 2 (must match prover padding)
        var paddedCount = 1
        while paddedCount < lookups.count { paddedCount <<= 1 }
        let zeroLookup = instrTable.encodeLookups(a: 0, b: 0, result: 0)
        while lookups.count < paddedCount {
            lookups.append(contentsOf: zeroLookup)
        }

        do {
            return try lassoEngine.verify(proof: lassoProof, lookups: lookups, table: lassoTable)
        } catch {
            return false
        }
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
