// Jolt Engine — Prove/verify CPU instruction execution via Lasso structured lookups
// Each instruction is a lookup into a precomputed table, verified via sumcheck.
// Large 32-bit tables decompose into byte-level subtables via Lasso tensor structure.
//
// Architecture:
//   1. Execute program -> trace (operands + results)
//   2. Group instructions by opcode
//   3. For byte-decomposable ops (AND, OR, XOR, ADD, SUB): prove via Lasso over byte subtables
//   4. For non-decomposable ops (MUL, SHL, SHR, EQ, LT): prove via Lasso range check
//      + constraint that result = op(a, b)
//
// References: Jolt (Arun, Setty, Thaler 2024)

import Foundation

// MARK: - Proof Structure

/// Proof for a single opcode group's instruction lookups.
public struct OpcodeProof {
    public let op: JoltOp
    /// Number of instructions proved for this opcode
    public let count: Int
    /// Lasso proof for byte-level lookups (for decomposable ops) or range checks (direct ops)
    public let lassoProof: LassoProof?
    /// For non-decomposable ops: direct result commitments (a, b, result per instruction)
    public let directLookups: [Fr]?

    public init(op: JoltOp, count: Int, lassoProof: LassoProof?, directLookups: [Fr]?) {
        self.op = op
        self.count = count
        self.lassoProof = lassoProof
        self.directLookups = directLookups
    }
}

/// Complete Jolt proof for a program execution.
public struct JoltProof {
    /// Number of instructions in the trace
    public let numInstructions: Int
    /// Per-opcode proofs
    public let opcodeProofs: [OpcodeProof]
    /// Operand values as field elements: (a, b, result) per instruction
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

    public let lasso: LassoEngine
    public let sumcheck: SumcheckEngine

    public init() throws {
        self.lasso = try LassoEngine()
        self.sumcheck = try SumcheckEngine()
    }

    /// Prove correct execution of a trace.
    public func prove(trace: JoltTrace) throws -> JoltProof {
        let n = trace.instructions.count
        precondition(n > 0, "Empty trace")

        // Build operand commitments
        var operandCommitments = [(Fr, Fr, Fr)]()
        operandCommitments.reserveCapacity(n)
        for op in trace.operands {
            operandCommitments.append((
                frFromInt(UInt64(op.a)),
                frFromInt(UInt64(op.b)),
                frFromInt(UInt64(op.result))
            ))
        }

        // Group instructions by opcode
        var groups = [JoltOp: [(Int, UInt32, UInt32, UInt32)]]()
        for (i, instr) in trace.instructions.enumerated() {
            let op = trace.operands[i]
            groups[instr.op, default: []].append((i, op.a, op.b, op.result))
        }

        // Prove each opcode group
        var opcodeProofs = [OpcodeProof]()
        for op in JoltOp.allCases {
            guard let group = groups[op], !group.isEmpty else { continue }

            if isDecomposable(op) {
                let proof = try proveDecomposable(op: op, group: group)
                opcodeProofs.append(proof)
            } else {
                let proof = try proveDirect(op: op, group: group)
                opcodeProofs.append(proof)
            }
        }

        return JoltProof(numInstructions: n, opcodeProofs: opcodeProofs,
                         operandCommitments: operandCommitments)
    }

    /// Verify a Jolt proof against a program.
    public func verify(proof: JoltProof, program: [JoltInstruction]) -> Bool {
        let n = program.count
        guard proof.numInstructions == n else { return false }
        guard proof.operandCommitments.count == n else { return false }

        // Re-group instructions by opcode
        var groups = [JoltOp: [(Int, Fr, Fr, Fr)]]()
        for (i, instr) in program.enumerated() {
            let (a, b, r) = proof.operandCommitments[i]
            groups[instr.op, default: []].append((i, a, b, r))
        }

        // Verify each opcode proof
        for opProof in proof.opcodeProofs {
            guard let group = groups[opProof.op] else { return false }
            guard group.count == opProof.count else { return false }

            if isDecomposable(opProof.op) {
                if !verifyDecomposable(opProof: opProof, group: group) {
                    return false
                }
            } else {
                if !verifyDirect(opProof: opProof, group: group) {
                    return false
                }
            }
        }

        return true
    }

    // MARK: - Decomposable Ops (AND, OR, XOR, ADD, SUB)

    private func isDecomposable(_ op: JoltOp) -> Bool {
        switch op {
        case .and_, .or_, .xor_, .add, .sub: return true
        case .mul, .shl, .shr, .eq, .lt: return false
        }
    }

    /// Prove byte-decomposable instructions via Lasso.
    /// Each 32-bit operation decomposes into 4 byte-level lookups.
    /// The byte-level table: entries indexed by (a_byte * 256 + b_byte), value = op(a_byte, b_byte).
    private func proveDecomposable(op: JoltOp, group: [(Int, UInt32, UInt32, UInt32)]) throws -> OpcodeProof {
        let rawCount = group.count
        let paddedCount = nextPowerOf2(rawCount)

        // Build Lasso table for this op with 4 byte chunks
        let table = buildLassoTable(op: op)

        // Build lookup values: pack (a, b) per byte into subtable indices
        var lookups = [Fr]()
        lookups.reserveCapacity(paddedCount)

        for (_, a, b, _) in group {
            let packed = packForLookup(op: op, a: a, b: b)
            lookups.append(packed)
        }

        // Pad with valid lookups (repeat first entry)
        let padValue = lookups.isEmpty ? frFromInt(0) : lookups[0]
        while lookups.count < paddedCount {
            lookups.append(padValue)
        }

        let proof = try lasso.prove(lookups: lookups, table: table)
        return OpcodeProof(op: op, count: rawCount, lassoProof: proof, directLookups: nil)
    }

    /// Verify decomposable op proof.
    private func verifyDecomposable(opProof: OpcodeProof, group: [(Int, Fr, Fr, Fr)]) -> Bool {
        guard let lassoProof = opProof.lassoProof else { return false }

        let paddedCount = nextPowerOf2(opProof.count)
        let table = buildLassoTable(op: opProof.op)

        // Reconstruct lookup values from operand commitments
        var lookups = [Fr]()
        lookups.reserveCapacity(paddedCount)
        for (_, a, b, _) in group {
            let aVal = frToInt(a)[0]
            let bVal = frToInt(b)[0]
            let packed = packForLookup(op: opProof.op, a: UInt32(aVal), b: UInt32(bVal))
            lookups.append(packed)
        }
        let padValue = lookups.isEmpty ? frFromInt(0) : lookups[0]
        while lookups.count < paddedCount {
            lookups.append(padValue)
        }

        // Verify Lasso proof
        do {
            let valid = try lasso.verify(proof: lassoProof, lookups: lookups, table: table)
            if !valid { return false }
        } catch {
            return false
        }

        // Verify operand results match the operation
        for (_, a, b, r) in group {
            let aVal = UInt32(frToInt(a)[0])
            let bVal = UInt32(frToInt(b)[0])
            let expected = joltCompute(op: opProof.op, a: aVal, b: bVal)
            if !frEqual(r, frFromInt(UInt64(expected))) { return false }
        }

        return true
    }

    // MARK: - Non-Decomposable Ops (MUL, SHL, SHR, EQ, LT)

    /// For non-decomposable ops: range-check operands + results via Lasso,
    /// then the verifier re-computes to check correctness.
    private func proveDirect(op: JoltOp, group: [(Int, UInt32, UInt32, UInt32)]) throws -> OpcodeProof {
        let rawCount = group.count
        let paddedCount = nextPowerOf2(rawCount * 3)

        // Range check all operands and results via Lasso
        let rangeTable = LassoTable.rangeCheck(bits: 32, chunks: 4)
        var lookups = [Fr]()
        lookups.reserveCapacity(paddedCount)

        for (_, a, b, result) in group {
            lookups.append(frFromInt(UInt64(a)))
            lookups.append(frFromInt(UInt64(b)))
            lookups.append(frFromInt(UInt64(result)))
        }

        while lookups.count < paddedCount {
            lookups.append(frFromInt(0))
        }

        let rangeProof = try lasso.prove(lookups: lookups, table: rangeTable)

        var directLookups = [Fr]()
        for (_, a, b, result) in group {
            directLookups.append(frFromInt(UInt64(a)))
            directLookups.append(frFromInt(UInt64(b)))
            directLookups.append(frFromInt(UInt64(result)))
        }

        return OpcodeProof(op: op, count: rawCount, lassoProof: rangeProof, directLookups: directLookups)
    }

    /// Verify non-decomposable op proof.
    private func verifyDirect(opProof: OpcodeProof, group: [(Int, Fr, Fr, Fr)]) -> Bool {
        guard let lassoProof = opProof.lassoProof else { return false }
        guard let directLookups = opProof.directLookups else { return false }
        guard directLookups.count == opProof.count * 3 else { return false }

        let paddedCount = nextPowerOf2(opProof.count * 3)
        let rangeTable = LassoTable.rangeCheck(bits: 32, chunks: 4)

        var lookups = [Fr]()
        lookups.reserveCapacity(paddedCount)
        lookups.append(contentsOf: directLookups)
        while lookups.count < paddedCount {
            lookups.append(frFromInt(0))
        }

        do {
            let valid = try lasso.verify(proof: lassoProof, lookups: lookups, table: rangeTable)
            if !valid { return false }
        } catch {
            return false
        }

        for (idx, (_, a, b, r)) in group.enumerated() {
            let aVal = UInt32(frToInt(a)[0])
            let bVal = UInt32(frToInt(b)[0])
            let expected = joltCompute(op: opProof.op, a: aVal, b: bVal)
            if !frEqual(r, frFromInt(UInt64(expected))) { return false }

            let dA = directLookups[idx * 3]
            let dB = directLookups[idx * 3 + 1]
            let dR = directLookups[idx * 3 + 2]
            if !frEqual(a, dA) || !frEqual(b, dB) || !frEqual(r, dR) { return false }
        }

        return true
    }

    // MARK: - Lasso Table Construction

    /// Build a Lasso table for byte-decomposable instructions.
    /// Each subtable is 256*256 = 65536 entries: all (a_byte, b_byte) pairs.
    /// The full 32-bit table decomposes into 4 identical byte-level subtables.
    private func buildLassoTable(op: JoltOp) -> LassoTable {
        let chunkRange = 256
        let chunks = 4

        let tableSize = chunkRange * chunkRange
        let subtable: [Fr] = (0..<tableSize).map { idx in
            let a = UInt8(idx / chunkRange)
            let b = UInt8(idx % chunkRange)
            let r: UInt8
            switch op {
            case .and_: r = a & b
            case .or_:  r = a | b
            case .xor_: r = a ^ b
            case .add:  r = UInt8((UInt32(a) + UInt32(b)) & 0xFF)
            case .sub:
                let diff = Int32(a) - Int32(b)
                r = UInt8((diff < 0 ? diff + 256 : diff) & 0xFF)
            default: fatalError("Not byte-decomposable: \(op)")
            }
            return frFromInt(UInt64(r))
        }
        let subtables = Array(repeating: subtable, count: chunks)

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

        let decompose: (Fr) -> [Int] = { value in
            let limbs = frToInt(value)
            let v = limbs[0]
            var indices = [Int]()
            indices.reserveCapacity(chunks)
            for k in 0..<chunks {
                let shift = k * 16
                let idx = Int((v >> shift) & 0xFFFF)
                indices.append(idx)
            }
            return indices
        }

        return LassoTable(subtables: subtables, compose: compose,
                          decompose: decompose, numChunks: chunks)
    }

    /// Pack (a, b) into a single lookup index for a byte-decomposable op.
    /// For each byte k: index_k = a_byte_k * 256 + b_byte_k (16 bits).
    /// Packed = index_0 + index_1 * 2^16 + index_2 * 2^32 + index_3 * 2^48.
    private func packForLookup(op: JoltOp, a: UInt32, b: UInt32) -> Fr {
        var packed: UInt64 = 0
        for k in 0..<4 {
            let aByte = UInt64((a >> (k * 8)) & 0xFF)
            let bByte = UInt64((b >> (k * 8)) & 0xFF)
            let idx = aByte * 256 + bByte
            packed |= idx << (k * 16)
        }
        return frFromInt(packed)
    }

    // MARK: - Helpers

    private func nextPowerOf2(_ n: Int) -> Int {
        var p = 1
        while p < n { p *= 2 }
        return max(p, 2)
    }
}
