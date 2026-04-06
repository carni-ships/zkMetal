// Tests for GPUJoltSubtableEngine — GPU-accelerated Jolt subtable decomposition
//
// Validates subtable materialization, batch decomposition, MLE commitment,
// batch evaluation, lazy subtables, and instruction-specific pipelines.

import Foundation
import zkMetal

public func runGPUJoltSubtableTests() {
    suite("GPUJoltSubtable-Materialization")
    testMaterializeAllSubtables()
    testMaterializeInstructionSpecific()
    testMaterializeConsistencyWithCPU()

    suite("GPUJoltSubtable-Decomposition")
    testBatchDecomposeBasic()
    testBatchDecomposeReassemble()
    testBatchDecomposeEdgeCases()
    testBatchDecomposeLarge()

    suite("GPUJoltSubtable-Commitment")
    testCommitSubtable()
    testCommitAllSubtables()
    testMLEEvaluation()
    testMLEBooleanConsistency()

    suite("GPUJoltSubtable-BatchEval")
    testBatchEvaluateSingle()
    testBatchEvaluateMultiple()
    testBatchEvaluateConsistency()

    suite("GPUJoltSubtable-LazySubtable")
    testLazySubtableAND()
    testLazySubtableOR()
    testLazySubtableXOR()
    testLazySubtableIdentity()
    testLazySubtablePartialMaterialize()

    suite("GPUJoltSubtable-ExecuteVerify")
    testExecuteVerifyADD()
    testExecuteVerifySUB()
    testExecuteVerifyAND()
    testExecuteVerifyOR()
    testExecuteVerifyXOR()
    testExecuteVerifySLT()
    testExecuteVerifySLTU()
    testBatchExecuteVerify()

    suite("GPUJoltSubtable-LassoCompat")
    testBuildLassoComponents()
    testLassoDecomposeCompose()

    suite("GPUJoltSubtable-Stats")
    testStatsChunk6()
    testStatsChunk8()
}

// MARK: - Materialization Tests

func testMaterializeAllSubtables() {
    let engine = GPUJoltSubtableEngine(chunkBits: 6)
    let tables = engine.materializeAll()

    // Should have all 14 standard subtables
    expect(tables.count >= 14, "materializeAll produces >= 14 tables")
    expect(tables["identity"] != nil, "identity table present")
    expect(tables["and"] != nil, "and table present")
    expect(tables["or"] != nil, "or table present")
    expect(tables["xor"] != nil, "xor table present")
    expect(tables["eq"] != nil, "eq table present")
    expect(tables["ltu"] != nil, "ltu table present")
    expect(tables["lt_signed"] != nil, "lt_signed table present")
    expect(tables["sll"] != nil, "sll table present")
    expect(tables["srl"] != nil, "srl table present")
    expect(tables["sra"] != nil, "sra table present")
    expect(tables["truncate_8"] != nil, "truncate_8 table present")
    expect(tables["truncate_16"] != nil, "truncate_16 table present")
    expect(tables["sign_extend_8"] != nil, "sign_extend_8 table present")
    expect(tables["sign_extend_16"] != nil, "sign_extend_16 table present")

    // Check sizes
    expectEqual(tables["identity"]!.count, 64, "identity size at C=6")
    expectEqual(tables["and"]!.count, 4096, "and size at C=6 (binary)")
    expectEqual(tables["xor"]!.count, 4096, "xor size at C=6 (binary)")
    expectEqual(tables["truncate_8"]!.count, 64, "truncate_8 size at C=6 (unary)")
}

func testMaterializeInstructionSpecific() {
    let engine = GPUJoltSubtableEngine(chunkBits: 6)

    // AND subtable
    let andTable = engine.materializeForInstruction(.and)
    expectEqual(andTable.count, 4096, "AND subtable has 4096 entries at C=6")

    // Spot check: AND(5, 3) at index (5 << 6) | 3 = 323
    let expected53 = frFromInt(5 & 3)
    expect(frEqual(andTable[323], expected53), "AND(5,3) = 1")

    // OR subtable
    let orTable = engine.materializeForInstruction(.or)
    let expected53or = frFromInt(5 | 3)
    expect(frEqual(orTable[323], expected53or), "OR(5,3) = 7")

    // XOR subtable
    let xorTable = engine.materializeForInstruction(.xor)
    let expected53xor = frFromInt(5 ^ 3)
    expect(frEqual(xorTable[323], expected53xor), "XOR(5,3) = 6")

    // Identity (for ADD/SUB)
    let addTable = engine.materializeForInstruction(.add)
    expectEqual(addTable.count, 64, "ADD identity subtable has 64 entries at C=6")
    expect(frEqual(addTable[42], frFromInt(42)), "identity[42] = 42")
}

func testMaterializeConsistencyWithCPU() {
    let engine = GPUJoltSubtableEngine(chunkBits: 6)

    // Compare GPU engine materialization with SubtableMaterializer
    let gpuTables = engine.materializeAll()
    let cpuTables = SubtableMaterializer.materializeAllAsFr(chunkBits: 6)

    for (name, cpuTable) in cpuTables {
        guard let gpuTable = gpuTables[name] else {
            expect(false, "GPU missing table: \(name)")
            continue
        }
        expectEqual(gpuTable.count, cpuTable.count, "\(name) size matches")
        var allMatch = true
        for i in 0..<cpuTable.count {
            if !frEqual(gpuTable[i], cpuTable[i]) {
                allMatch = false
                break
            }
        }
        expect(allMatch, "\(name) values match CPU materializer")
    }
}

// MARK: - Decomposition Tests

func testBatchDecomposeBasic() {
    let engine = GPUJoltSubtableEngine(chunkBits: 6)

    let values: [UInt32] = [0, 1, 63, 64, 0xFFFFFFFF]
    let result = engine.batchDecompose(values: values)

    expectEqual(result.count, 5, "5 values decomposed")
    expectEqual(result.numChunks, 6, "6 chunks for 32-bit at C=6")

    // Value 0: all chunks should be 0
    let chunks0 = result.chunks(at: 0)
    for k in 0..<6 {
        expectEqual(chunks0[k], 0, "value 0, chunk \(k) = 0")
    }

    // Value 1: chunk 0 = 1, rest = 0
    let chunks1 = result.chunks(at: 1)
    expectEqual(chunks1[0], 1, "value 1, chunk 0 = 1")
    for k in 1..<6 {
        expectEqual(chunks1[k], 0, "value 1, chunk \(k) = 0")
    }

    // Value 63: chunk 0 = 63, rest = 0
    let chunks63 = result.chunks(at: 2)
    expectEqual(chunks63[0], 63, "value 63, chunk 0 = 63")

    // Value 64: chunk 0 = 0, chunk 1 = 1
    let chunks64 = result.chunks(at: 3)
    expectEqual(chunks64[0], 0, "value 64, chunk 0 = 0")
    expectEqual(chunks64[1], 1, "value 64, chunk 1 = 1")
}

func testBatchDecomposeReassemble() {
    let engine = GPUJoltSubtableEngine(chunkBits: 6)

    // Random values
    var rng: UInt64 = 0xDEADBEEF
    var values = [UInt32]()
    for _ in 0..<100 {
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        values.append(UInt32(truncatingIfNeeded: rng >> 16))
    }

    let decomp = engine.batchDecompose(values: values)

    for i in 0..<values.count {
        let chunks = decomp.chunks(at: i)
        let reassembled = engine.reassemble(chunks: chunks.map { UInt64($0) })
        expectEqual(reassembled, values[i], "reassemble(decompose(\(values[i]))) round-trip")
    }
}

func testBatchDecomposeEdgeCases() {
    let engine = GPUJoltSubtableEngine(chunkBits: 8)

    let values: [UInt32] = [0, 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF, 0x00FF00FF]
    let result = engine.batchDecompose(values: values, chunkBits: 8)

    expectEqual(result.numChunks, 4, "4 chunks for 32-bit at C=8")

    // 0xFF at chunk 0 = 0xFF
    let chunksFF = result.chunks(at: 1)
    for k in 0..<4 {
        expectEqual(chunksFF[k], 0xFF, "0xFFFFFFFF chunk \(k) = 0xFF")
    }

    // 0x00FF00FF: chunks alternate 0xFF and 0x00
    let chunksAlt = result.chunks(at: 4)
    expectEqual(chunksAlt[0], 0xFF, "0x00FF00FF chunk 0 = 0xFF")
    expectEqual(chunksAlt[1], 0x00, "0x00FF00FF chunk 1 = 0x00")
    expectEqual(chunksAlt[2], 0xFF, "0x00FF00FF chunk 2 = 0xFF")
    expectEqual(chunksAlt[3], 0x00, "0x00FF00FF chunk 3 = 0x00")
}

func testBatchDecomposeLarge() {
    let engine = GPUJoltSubtableEngine(chunkBits: 6)

    // Larger batch to potentially trigger GPU path
    let n = 8192
    var values = [UInt32](repeating: 0, count: n)
    for i in 0..<n {
        values[i] = UInt32(truncatingIfNeeded: i &* 0xDEADBEEF)
    }

    let result = engine.batchDecompose(values: values)
    expectEqual(result.count, n, "large batch count")

    // Verify a sample
    for i in stride(from: 0, to: n, by: 1000) {
        let chunks = result.chunks(at: i)
        let reassembled = engine.reassemble(chunks: chunks.map { UInt64($0) })
        expectEqual(reassembled, values[i], "large batch reassemble[\(i)]")
    }
}

// MARK: - Commitment Tests

func testCommitSubtable() {
    let engine = GPUJoltSubtableEngine(chunkBits: 6)

    let identity = IdentitySubtable(chunkBits: 6)
    let commitment = engine.commitSubtable(identity)

    expectEqual(commitment.name, "identity", "commitment name")
    expectEqual(commitment.numVariables, 6, "identity has 2^6=64 entries -> 6 vars")
    expectEqual(commitment.evaluations.count, 64, "64 evaluations")
    expect(!commitment.isBinary, "identity is unary")

    // Check evaluations match table values
    for i in 0..<64 {
        expect(frEqual(commitment.evaluations[i], frFromInt(UInt64(i))),
               "identity commitment eval[\(i)] = \(i)")
    }
}

func testCommitAllSubtables() {
    let engine = GPUJoltSubtableEngine(chunkBits: 6)
    let commitments = engine.commitAll()

    expect(commitments.count >= 14, "commitAll produces >= 14 commitments")
    expect(commitments["identity"] != nil, "identity commitment present")
    expect(commitments["and"] != nil, "and commitment present")

    // Identity: 6 variables (2^6 = 64 entries)
    expectEqual(commitments["identity"]!.numVariables, 6, "identity 6 vars")
    expect(!commitments["identity"]!.isBinary, "identity is unary")

    // AND: 12 variables (2^12 = 4096 entries)
    expectEqual(commitments["and"]!.numVariables, 12, "and 12 vars")
    expect(commitments["and"]!.isBinary, "and is binary")
}

func testMLEEvaluation() {
    let engine = GPUJoltSubtableEngine(chunkBits: 6)

    // Commit identity subtable with small chunk for manageable size
    let identity = IdentitySubtable(chunkBits: 3)
    let commitment = engine.commitSubtable(identity, chunkBits: 3)

    // MLE at all-zeros should equal T[0] = 0
    let zeroPoint = [Fr](repeating: Fr.zero, count: commitment.numVariables)
    let evalZero = engine.evaluateMLE(commitment, at: zeroPoint)
    expect(frEqual(evalZero, frFromInt(0)), "MLE at (0,0,0) = T[0] = 0")

    // MLE at (1,0,0) should equal T[1] = 1
    var point100 = [Fr](repeating: Fr.zero, count: commitment.numVariables)
    point100[0] = Fr.one
    let eval100 = engine.evaluateMLE(commitment, at: point100)
    expect(frEqual(eval100, frFromInt(1)), "MLE at (1,0,0) = T[1] = 1")

    // MLE at (0,1,0) should equal T[2] = 2
    var point010 = [Fr](repeating: Fr.zero, count: commitment.numVariables)
    point010[1] = Fr.one
    let eval010 = engine.evaluateMLE(commitment, at: point010)
    expect(frEqual(eval010, frFromInt(2)), "MLE at (0,1,0) = T[2] = 2")

    // MLE at (1,1,1) should equal T[7] = 7
    let onesPoint = [Fr](repeating: Fr.one, count: commitment.numVariables)
    let eval111 = engine.evaluateMLE(commitment, at: onesPoint)
    expect(frEqual(eval111, frFromInt(7)), "MLE at (1,1,1) = T[7] = 7")
}

func testMLEBooleanConsistency() {
    let engine = GPUJoltSubtableEngine(chunkBits: 6)

    // For a small subtable, verify MLE at every boolean point equals the table entry
    let andST = AndSubtable(chunkBits: 2)
    let commitment = engine.commitSubtable(andST, chunkBits: 2)

    // AND with 2-bit chunks: table size = 2^4 = 16, numVars = 4
    expectEqual(commitment.numVariables, 4, "AND(C=2) has 4 MLE variables")

    // Check all 16 boolean evaluation points
    for idx in 0..<16 {
        var point = [Fr](repeating: Fr.zero, count: 4)
        for bit in 0..<4 {
            if idx & (1 << bit) != 0 {
                point[bit] = Fr.one
            }
        }
        let mleVal = engine.evaluateMLE(commitment, at: point)
        let tableVal = commitment.evaluations[idx]
        expect(frEqual(mleVal, tableVal),
               "MLE boolean point \(idx): MLE = table[\(idx)]")
    }
}

// MARK: - Batch Evaluation Tests

func testBatchEvaluateSingle() {
    let engine = GPUJoltSubtableEngine(chunkBits: 6)

    let andST = AndSubtable(chunkBits: 6)
    let inputs: [UInt64] = [
        (5 << 6) | 3,   // AND(5,3) = 1
        (63 << 6) | 63, // AND(63,63) = 63
        (0 << 6) | 42,  // AND(0,42) = 0
        (42 << 6) | 0,  // AND(42,0) = 0
    ]

    let result = engine.batchEvaluate(subtable: andST, inputs: inputs)

    expectEqual(result.count, 4, "4 evaluations")
    expectEqual(result.subtableName, "and", "subtable name")
    expect(frEqual(result.outputs[0], frFromInt(1)), "AND(5,3) = 1")
    expect(frEqual(result.outputs[1], frFromInt(63)), "AND(63,63) = 63")
    expect(frEqual(result.outputs[2], frFromInt(0)), "AND(0,42) = 0")
    expect(frEqual(result.outputs[3], frFromInt(0)), "AND(42,0) = 0")
}

func testBatchEvaluateMultiple() {
    let engine = GPUJoltSubtableEngine(chunkBits: 6)

    let andST = AndSubtable(chunkBits: 6)
    let orST = OrSubtable(chunkBits: 6)
    let xorST = XorSubtable(chunkBits: 6)

    let input = (10 << 6) | 5  // x=10, y=5

    let results = engine.batchEvaluateMultiple(requests: [
        (subtable: andST, inputs: [UInt64(input)]),
        (subtable: orST, inputs: [UInt64(input)]),
        (subtable: xorST, inputs: [UInt64(input)]),
    ])

    expectEqual(results.count, 3, "3 subtable results")
    expect(frEqual(results[0].outputs[0], frFromInt(10 & 5)), "AND(10,5) = 0")
    expect(frEqual(results[1].outputs[0], frFromInt(10 | 5)), "OR(10,5) = 15")
    expect(frEqual(results[2].outputs[0], frFromInt(10 ^ 5)), "XOR(10,5) = 15")
}

func testBatchEvaluateConsistency() {
    let engine = GPUJoltSubtableEngine(chunkBits: 6)

    // Verify batch evaluation matches individual subtable evaluation
    let xorST = XorSubtable(chunkBits: 6)
    var inputs = [UInt64]()
    for x: UInt64 in 0..<64 {
        for y: UInt64 in stride(from: 0, to: 64, by: 8) {
            inputs.append((x << 6) | y)
        }
    }

    let batchResult = engine.batchEvaluate(subtable: xorST, inputs: inputs)

    for (i, input) in inputs.enumerated() {
        let expected = frFromInt(xorST.evaluate(input: input))
        expect(frEqual(batchResult.outputs[i], expected),
               "batch eval XOR consistent at \(i)")
    }
}

// MARK: - Lazy Subtable Tests

func testLazySubtableAND() {
    let engine = GPUJoltSubtableEngine(chunkBits: 6)
    let lazy = engine.lazySubtable(for: .and)

    expectEqual(lazy.name, "and", "lazy AND name")
    expect(lazy.isBinary, "AND is binary")
    expectEqual(lazy.tableSize, 4096, "AND table size at C=6")

    // Spot checks
    let packed = UInt64((10 << 6) | 5)
    expectEqual(lazy.evaluate(input: packed), 10 & 5, "lazy AND(10,5)")

    let fr_val = lazy.evaluateAsFr(input: packed)
    expect(frEqual(fr_val, frFromInt(10 & 5)), "lazy AND(10,5) as Fr")
}

func testLazySubtableOR() {
    let engine = GPUJoltSubtableEngine(chunkBits: 6)
    let lazy = engine.lazySubtable(for: .or)

    expectEqual(lazy.name, "or", "lazy OR name")
    let packed = UInt64((10 << 6) | 5)
    expectEqual(lazy.evaluate(input: packed), 10 | 5, "lazy OR(10,5)")
}

func testLazySubtableXOR() {
    let engine = GPUJoltSubtableEngine(chunkBits: 6)
    let lazy = engine.lazySubtable(for: .xor)

    expectEqual(lazy.name, "xor", "lazy XOR name")
    let packed = UInt64((10 << 6) | 5)
    expectEqual(lazy.evaluate(input: packed), 10 ^ 5, "lazy XOR(10,5)")
}

func testLazySubtableIdentity() {
    let engine = GPUJoltSubtableEngine(chunkBits: 6)

    // ADD and SUB use identity subtables
    let lazyAdd = engine.lazySubtable(for: .add)
    expect(!lazyAdd.isBinary, "ADD identity is unary")
    expectEqual(lazyAdd.tableSize, 64, "ADD identity size at C=6")
    expectEqual(lazyAdd.evaluate(input: 42), 42, "ADD identity(42) = 42")

    let lazySub = engine.lazySubtable(for: .sub)
    expectEqual(lazySub.evaluate(input: 17), 17, "SUB identity(17) = 17")
}

func testLazySubtablePartialMaterialize() {
    let engine = GPUJoltSubtableEngine(chunkBits: 6)
    let lazy = engine.lazySubtable(for: .and)

    // Materialize first 16 entries
    let partial = lazy.materializeRange(start: 0, count: 16)
    expectEqual(partial.count, 16, "partial materialization size")

    // Verify against full AND subtable
    let andST = AndSubtable(chunkBits: 6)
    for i in 0..<16 {
        let expected = frFromInt(andST.evaluate(input: UInt64(i)))
        expect(frEqual(partial[i], expected), "partial materialize[\(i)]")
    }
}

// MARK: - Execute and Verify Tests

func testExecuteVerifyADD() {
    let engine = GPUJoltSubtableEngine(chunkBits: 6)

    let (result, verified) = engine.executeAndVerify(instruction: .add, operandA: 100, operandB: 200)
    expectEqual(result, 300, "ADD 100+200=300")
    expect(verified, "ADD verified")

    let (r2, v2) = engine.executeAndVerify(instruction: .add, operandA: 0xFFFFFFFF, operandB: 1)
    expectEqual(r2, 0, "ADD overflow wraps to 0")
    expect(v2, "ADD overflow verified")

    let (r3, v3) = engine.executeAndVerify(instruction: .add, operandA: 0, operandB: 0)
    expectEqual(r3, 0, "ADD 0+0=0")
    expect(v3, "ADD zero verified")
}

func testExecuteVerifySUB() {
    let engine = GPUJoltSubtableEngine(chunkBits: 6)

    let (result, verified) = engine.executeAndVerify(instruction: .sub, operandA: 200, operandB: 100)
    expectEqual(result, 100, "SUB 200-100=100")
    expect(verified, "SUB verified")

    let (r2, v2) = engine.executeAndVerify(instruction: .sub, operandA: 0, operandB: 1)
    expectEqual(r2, 0xFFFFFFFF, "SUB underflow wraps")
    expect(v2, "SUB underflow verified")
}

func testExecuteVerifyAND() {
    let engine = GPUJoltSubtableEngine(chunkBits: 6)

    let (result, verified) = engine.executeAndVerify(instruction: .and, operandA: 0xFF00FF00, operandB: 0x0F0F0F0F)
    expectEqual(result, 0x0F000F00, "AND mask result")
    expect(verified, "AND verified")

    let (r2, v2) = engine.executeAndVerify(instruction: .and, operandA: 0xAAAAAAAA, operandB: 0x55555555)
    expectEqual(r2, 0, "AND complementary = 0")
    expect(v2, "AND complementary verified")
}

func testExecuteVerifyOR() {
    let engine = GPUJoltSubtableEngine(chunkBits: 6)

    let (result, verified) = engine.executeAndVerify(instruction: .or, operandA: 0xFF00, operandB: 0x00FF)
    expectEqual(result, 0xFFFF, "OR result")
    expect(verified, "OR verified")
}

func testExecuteVerifyXOR() {
    let engine = GPUJoltSubtableEngine(chunkBits: 6)

    let (result, verified) = engine.executeAndVerify(instruction: .xor, operandA: 0xFFFFFFFF, operandB: 0xFFFFFFFF)
    expectEqual(result, 0, "XOR self = 0")
    expect(verified, "XOR verified")

    let (r2, v2) = engine.executeAndVerify(instruction: .xor, operandA: 0xAAAAAAAA, operandB: 0x55555555)
    expectEqual(r2, 0xFFFFFFFF, "XOR complementary = all 1s")
    expect(v2, "XOR complementary verified")
}

func testExecuteVerifySLT() {
    let engine = GPUJoltSubtableEngine(chunkBits: 6)

    let (r1, v1) = engine.executeAndVerify(instruction: .slt, operandA: 5, operandB: 10)
    expectEqual(r1, 1, "SLT 5 < 10")
    expect(v1, "SLT verified")

    let (r2, v2) = engine.executeAndVerify(instruction: .slt, operandA: 10, operandB: 5)
    expectEqual(r2, 0, "SLT 10 >= 5")
    expect(v2, "SLT false verified")

    // Signed: -1 < 0
    let (r3, v3) = engine.executeAndVerify(instruction: .slt, operandA: 0xFFFFFFFF, operandB: 0)
    expectEqual(r3, 1, "SLT -1 < 0 (signed)")
    expect(v3, "SLT signed verified")
}

func testExecuteVerifySLTU() {
    let engine = GPUJoltSubtableEngine(chunkBits: 6)

    let (r1, v1) = engine.executeAndVerify(instruction: .sltu, operandA: 5, operandB: 10)
    expectEqual(r1, 1, "SLTU 5 < 10")
    expect(v1, "SLTU verified")

    // Unsigned: 0xFFFFFFFF > 0
    let (r2, v2) = engine.executeAndVerify(instruction: .sltu, operandA: 0xFFFFFFFF, operandB: 0)
    expectEqual(r2, 0, "SLTU max >= 0 (unsigned)")
    expect(v2, "SLTU unsigned verified")
}

func testBatchExecuteVerify() {
    let engine = GPUJoltSubtableEngine(chunkBits: 6)

    var operations = [(instruction: SubtableInstructionType, a: UInt32, b: UInt32)]()

    // Generate a mix of operations
    var rng: UInt64 = 0xCAFEBABE
    let instrs: [SubtableInstructionType] = [.add, .sub, .and, .or, .xor, .slt, .sltu]

    for _ in 0..<50 {
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        let a = UInt32(truncatingIfNeeded: rng >> 16)
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        let b = UInt32(truncatingIfNeeded: rng >> 16)
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        let instrIdx = Int(rng >> 32) % instrs.count
        operations.append((instruction: instrs[instrIdx], a: a, b: b))
    }

    let results = engine.batchExecuteAndVerify(operations: operations)

    expectEqual(results.count, 50, "50 batch results")
    var allVerified = true
    for (i, res) in results.enumerated() {
        if !res.verified {
            allVerified = false
            expect(false, "batch verify failed at \(i): \(operations[i].instruction)")
        }
    }
    if allVerified {
        expect(true, "all 50 batch operations verified")
    }
}

// MARK: - Lasso Compatibility Tests

func testBuildLassoComponents() {
    let engine = GPUJoltSubtableEngine(chunkBits: 6)

    let (subtables, compose, decompose) = engine.buildLassoComponents(for: .and)

    expectEqual(subtables.count, 6, "6 subtable copies for 32-bit/6-chunk")
    expectEqual(subtables[0].count, 4096, "AND subtable size at C=6")

    // Test decompose on a known value
    let val = frFromInt(0xAB)
    let indices = decompose(val)
    expectEqual(indices.count, 6, "6 chunk indices")

    // Chunk 0 of 0xAB = 0xAB & 0x3F = 0x2B = 43
    expectEqual(indices[0], 43, "chunk 0 of 0xAB")
    // Chunk 1 of 0xAB = (0xAB >> 6) & 0x3F = 2
    expectEqual(indices[1], 2, "chunk 1 of 0xAB")
}

func testLassoDecomposeCompose() {
    let engine = GPUJoltSubtableEngine(chunkBits: 8)

    // At C=8, decomposition is byte-level, 4 chunks
    let (subtables, compose, decompose) = engine.buildLassoComponents(for: .xor, chunkBits: 8)

    expectEqual(subtables.count, 4, "4 subtable copies for 32-bit/8-chunk")

    // Test round-trip: decompose, look up, compose
    let testValues: [UInt64] = [0, 1, 255, 256, 0xDEADBEEF, 0xFFFFFFFF]
    for v in testValues {
        let vFr = frFromInt(v & 0xFFFFFFFF)
        let indices = decompose(vFr)
        expectEqual(indices.count, 4, "4 indices at C=8")

        // Look up each chunk in subtable (identity for XOR subtable, but the
        // subtable contains XOR results; for identity of the value itself,
        // we just check the indices reconstruct the value)
        var chunks = [Fr]()
        for k in 0..<4 {
            let byte = Int((v >> (k * 8)) & 0xFF)
            expectEqual(indices[k], byte, "decompose byte \(k) of 0x\(String(v, radix: 16))")
            chunks.append(frFromInt(UInt64(byte)))
        }

        let composed = compose(chunks)
        expect(frEqual(composed, vFr), "compose(decompose(0x\(String(v, radix: 16)))) round-trip")
    }
}

// MARK: - Statistics Tests

func testStatsChunk6() {
    let engine = GPUJoltSubtableEngine(chunkBits: 6)
    let s = engine.stats()

    expectEqual(s.chunkBits, 6, "stats chunkBits=6")
    expectEqual(s.unaryTableSize, 64, "unary size 2^6")
    expectEqual(s.binaryTableSize, 4096, "binary size 2^12")
    expectEqual(s.totalSubtables, 14, "14 total subtables")
    expect(s.totalEntries > 0, "positive total entries")
}

func testStatsChunk8() {
    let engine = GPUJoltSubtableEngine(chunkBits: 8)
    let s = engine.stats(chunkBits: 8)

    expectEqual(s.chunkBits, 8, "stats chunkBits=8")
    expectEqual(s.unaryTableSize, 256, "unary size 2^8")
    expectEqual(s.binaryTableSize, 65536, "binary size 2^16")
}
