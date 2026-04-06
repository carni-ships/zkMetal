// Tests for GPUJoltInstructionEngine — GPU-accelerated Jolt instruction lookup engine
//
// Validates instruction classification, single/batch decomposition, lookup verification,
// fingerprinting, lookup argument construction, chunk extraction, trace processing,
// category filtering, subtable usage analysis, and cross-instruction consistency.

import Foundation
import zkMetal

public func runGPUJoltInstructionTests() {
    suite("GPUJoltInstruction-Classification")
    testClassifyALU()
    testClassifyMemory()
    testClassifyBranch()
    testClassifyJump()
    testClassifyUpperImm()
    testClassifyMExtension()
    testClassifySystem()
    testClassifyAllCategories()

    suite("GPUJoltInstruction-SingleDecompose")
    testDecomposeADD()
    testDecomposeSUB()
    testDecomposeAND()
    testDecomposeOR()
    testDecomposeXOR()
    testDecomposeSLL()
    testDecomposeSRL()
    testDecomposeSRA()
    testDecomposeSLT()
    testDecomposeSLTU()

    suite("GPUJoltInstruction-SingleVerify")
    testVerifyALUBasic()
    testVerifyALUOverflow()
    testVerifyALUEdgeCases()
    testVerifyMemory()
    testVerifyBranch()
    testVerifyJump()
    testVerifyUpperImmediate()
    testVerifyMExtension()
    testVerifySystem()

    suite("GPUJoltInstruction-BatchVerify")
    testBatchVerifySmall()
    testBatchVerifyMixed()
    testBatchVerifyRandom()
    testBatchVerifyLarge()
    testBatchVerifyAllVerified()

    suite("GPUJoltInstruction-BatchDecompose")
    testJoltInstrBatchDecomposeBasic()
    testBatchDecomposeRecords()
    testJoltInstrBatchDecomposeLarge()

    suite("GPUJoltInstruction-Fingerprint")
    testFingerprintBasic()
    testFingerprintDifferentChallenges()
    testFingerprintSubtableBreakdown()
    testFingerprintDeterministic()

    suite("GPUJoltInstruction-LookupArgument")
    testBuildLookupArgumentBasic()
    testBuildLookupArgumentMultipleSubtables()
    testBuildLookupArgumentConsistency()

    suite("GPUJoltInstruction-ChunkExtraction")
    testChunkPairsBasic()
    testChunkPairsEdgeCases()
    testChunkPairsReassemble()
    testChunkPairsLargeBatch()

    suite("GPUJoltInstruction-Consistency")
    testConsistencyAllPass()
    testConsistencyMixed()
    testConsistencyRandom()

    suite("GPUJoltInstruction-TraceProcessing")
    testProcessTraceBasic()
    testProcessTraceMixed()

    suite("GPUJoltInstruction-CategoryFilter")
    testVerifyByCategory()
    testVerifyByCategoryEmpty()

    suite("GPUJoltInstruction-SubtableUsage")
    testSubtableUsageALU()
    testSubtableUsageMixed()

    suite("GPUJoltInstruction-Stats")
    testEngineStats()
    testEngineStatsChunk8()

    suite("GPUJoltInstruction-ChunkSizes")
    testChunkSize6Verification()
    testChunkSize8Verification()
}

// MARK: - Classification Tests

private func testClassifyALU() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    let aluInstrs: [RV32IInstruction] = [
        .ADD, .SUB, .AND, .OR, .XOR, .SLT, .SLTU,
        .ADDI, .ANDI, .ORI, .XORI, .SLTI, .SLTIU,
        .SLL, .SRL, .SRA, .SLLI, .SRLI, .SRAI
    ]
    for instr in aluInstrs {
        let cat = engine.classify(.base(instr))
        expectEqual(cat, .alu, "\(instr) classified as ALU")
    }
}

private func testClassifyMemory() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    let memInstrs: [RV32IInstruction] = [.LB, .LH, .LW, .LBU, .LHU, .SB, .SH, .SW]
    for instr in memInstrs {
        let cat = engine.classify(.base(instr))
        expectEqual(cat, .memory, "\(instr) classified as Memory")
    }
}

private func testClassifyBranch() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    let branchInstrs: [RV32IInstruction] = [.BEQ, .BNE, .BLT, .BGE, .BLTU, .BGEU]
    for instr in branchInstrs {
        let cat = engine.classify(.base(instr))
        expectEqual(cat, .branch, "\(instr) classified as Branch")
    }
}

private func testClassifyJump() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    expectEqual(engine.classify(.base(.JAL)), .jump, "JAL classified as Jump")
    expectEqual(engine.classify(.base(.JALR)), .jump, "JALR classified as Jump")
}

private func testClassifyUpperImm() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    expectEqual(engine.classify(.base(.LUI)), .upperImm, "LUI classified as UpperImmediate")
    expectEqual(engine.classify(.base(.AUIPC)), .upperImm, "AUIPC classified as UpperImmediate")
}

private func testClassifyMExtension() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    let mInstrs: [RV32MInstruction] = [.MUL, .MULH, .MULHSU, .MULHU, .DIV, .DIVU, .REM, .REMU]
    for instr in mInstrs {
        let cat = engine.classify(.mul(instr))
        expectEqual(cat, .mExtension, "\(instr) classified as MExtension")
    }
}

private func testClassifySystem() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    expectEqual(engine.classify(.base(.ECALL)), .system, "ECALL classified as System")
    expectEqual(engine.classify(.base(.EBREAK)), .system, "EBREAK classified as System")
    expectEqual(engine.classify(.base(.FENCE)), .system, "FENCE classified as System")
}

private func testClassifyAllCategories() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    // Every category should be reachable
    var seenCategories = Set<InstructionCategory>()

    for instr in RV32IInstruction.allCases {
        seenCategories.insert(engine.classify(.base(instr)))
    }
    for instr in RV32MInstruction.allCases {
        seenCategories.insert(engine.classify(.mul(instr)))
    }

    for cat in InstructionCategory.allCases {
        expect(seenCategories.contains(cat), "category \(cat) is reachable")
    }
}

// MARK: - Single Decomposition Tests

private func testDecomposeADD() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    let record = engine.decomposeInstruction(.base(.ADD), operandA: 100, operandB: 200)
    expectEqual(record.result, 300, "ADD 100+200 = 300")
    expectEqual(record.subtableNames.count, 6, "ADD uses 6 chunks at C=6")

    // All subtables should be identity for ADD (result decomposition)
    for name in record.subtableNames {
        expectEqual(name, "identity", "ADD uses identity subtables")
    }
}

private func testDecomposeSUB() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    let record = engine.decomposeInstruction(.base(.SUB), operandA: 300, operandB: 100)
    expectEqual(record.result, 200, "SUB 300-100 = 200")

    let recordUnderflow = engine.decomposeInstruction(.base(.SUB), operandA: 0, operandB: 1)
    expectEqual(recordUnderflow.result, 0xFFFFFFFF, "SUB 0-1 wraps")
}

private func testDecomposeAND() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    let record = engine.decomposeInstruction(.base(.AND), operandA: 0xFF00FF00, operandB: 0x0F0F0F0F)
    expectEqual(record.result, 0x0F000F00, "AND mask result")

    // AND uses binary subtables
    for name in record.subtableNames {
        expectEqual(name, "and", "AND uses and subtables")
    }
}

private func testDecomposeOR() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    let record = engine.decomposeInstruction(.base(.OR), operandA: 0xFF00, operandB: 0x00FF)
    expectEqual(record.result, 0xFFFF, "OR result")

    for name in record.subtableNames {
        expectEqual(name, "or", "OR uses or subtables")
    }
}

private func testDecomposeXOR() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    let record = engine.decomposeInstruction(.base(.XOR), operandA: 0xAAAAAAAA, operandB: 0x55555555)
    expectEqual(record.result, 0xFFFFFFFF, "XOR complementary")

    for name in record.subtableNames {
        expectEqual(name, "xor", "XOR uses xor subtables")
    }
}

private func testDecomposeSLL() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    let record = engine.decomposeInstruction(.base(.SLL), operandA: 1, operandB: 4)
    expectEqual(record.result, 16, "SLL 1<<4 = 16")
    expectEqual(record.subtableNames.count, 6, "SLL uses 6 chunks")
}

private func testDecomposeSRL() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    let record = engine.decomposeInstruction(.base(.SRL), operandA: 0x80000000, operandB: 1)
    expectEqual(record.result, 0x40000000, "SRL logical right shift")
}

private func testDecomposeSRA() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    let record = engine.decomposeInstruction(.base(.SRA), operandA: 0x80000000, operandB: 1)
    expectEqual(record.result, 0xC0000000, "SRA arithmetic right shift preserves sign")
}

private func testDecomposeSLT() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    let record1 = engine.decomposeInstruction(.base(.SLT), operandA: 5, operandB: 10)
    expectEqual(record1.result, 1, "SLT 5 < 10")

    let record2 = engine.decomposeInstruction(.base(.SLT), operandA: 0xFFFFFFFF, operandB: 0)
    expectEqual(record2.result, 1, "SLT -1 < 0 (signed)")
}

private func testDecomposeSLTU() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    let record1 = engine.decomposeInstruction(.base(.SLTU), operandA: 5, operandB: 10)
    expectEqual(record1.result, 1, "SLTU 5 < 10")

    let record2 = engine.decomposeInstruction(.base(.SLTU), operandA: 0xFFFFFFFF, operandB: 0)
    expectEqual(record2.result, 0, "SLTU max >= 0 (unsigned)")
}

// MARK: - Single Verification Tests

private func testVerifyALUBasic() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    let entry = engine.verifyInstruction(.base(.ADD), operandA: 42, operandB: 58)
    expectEqual(entry.expectedResult, 100, "ADD expected 100")
    expectEqual(entry.decomposedResult, 100, "ADD decomposed 100")
    expect(entry.verified, "ADD verified")
    expectEqual(entry.category, .alu, "ADD category is ALU")
    expectEqual(entry.chunkIndices.count, 6, "6 chunk indices")
}

private func testVerifyALUOverflow() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    let entry = engine.verifyInstruction(.base(.ADD), operandA: 0xFFFFFFFF, operandB: 1)
    expectEqual(entry.expectedResult, 0, "ADD overflow wraps to 0")
    expect(entry.verified, "ADD overflow verified")

    let entry2 = engine.verifyInstruction(.base(.SUB), operandA: 0, operandB: 1)
    expectEqual(entry2.expectedResult, 0xFFFFFFFF, "SUB underflow wraps")
    expect(entry2.verified, "SUB underflow verified")

    let entry3 = engine.verifyInstruction(.base(.ADD), operandA: 0x80000000, operandB: 0x80000000)
    expectEqual(entry3.expectedResult, 0, "ADD large+large wraps")
    expect(entry3.verified, "ADD large overflow verified")
}

private func testVerifyALUEdgeCases() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    // Zero operands
    let e1 = engine.verifyInstruction(.base(.ADD), operandA: 0, operandB: 0)
    expectEqual(e1.expectedResult, 0, "ADD 0+0=0")
    expect(e1.verified, "ADD zeros verified")

    // Max operands
    let e2 = engine.verifyInstruction(.base(.AND), operandA: 0xFFFFFFFF, operandB: 0xFFFFFFFF)
    expectEqual(e2.expectedResult, 0xFFFFFFFF, "AND max&max=max")
    expect(e2.verified, "AND max verified")

    // XOR self
    let e3 = engine.verifyInstruction(.base(.XOR), operandA: 0xDEADBEEF, operandB: 0xDEADBEEF)
    expectEqual(e3.expectedResult, 0, "XOR self=0")
    expect(e3.verified, "XOR self verified")

    // OR with 0
    let e4 = engine.verifyInstruction(.base(.OR), operandA: 0xCAFEBABE, operandB: 0)
    expectEqual(e4.expectedResult, 0xCAFEBABE, "OR with 0 = identity")
    expect(e4.verified, "OR identity verified")
}

private func testVerifyMemory() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    // Memory ops compute effective address = a + b
    let loads: [RV32IInstruction] = [.LB, .LH, .LW, .LBU, .LHU]
    for instr in loads {
        let entry = engine.verifyInstruction(.base(instr), operandA: 0x1000, operandB: 0x100)
        expectEqual(entry.expectedResult, 0x1100, "\(instr) effective address")
        expect(entry.verified, "\(instr) verified")
        expectEqual(entry.category, .memory, "\(instr) category is Memory")
    }

    let stores: [RV32IInstruction] = [.SB, .SH, .SW]
    for instr in stores {
        let entry = engine.verifyInstruction(.base(instr), operandA: 0x2000, operandB: 0x50)
        expectEqual(entry.expectedResult, 0x2050, "\(instr) effective address")
        expect(entry.verified, "\(instr) verified")
    }
}

private func testVerifyBranch() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    // BEQ: equal -> 1, not equal -> 0
    let e1 = engine.verifyInstruction(.base(.BEQ), operandA: 42, operandB: 42)
    expectEqual(e1.expectedResult, 1, "BEQ equal")
    expect(e1.verified, "BEQ equal verified")

    let e2 = engine.verifyInstruction(.base(.BEQ), operandA: 42, operandB: 43)
    expectEqual(e2.expectedResult, 0, "BEQ not equal")
    expect(e2.verified, "BEQ not equal verified")

    // BNE
    let e3 = engine.verifyInstruction(.base(.BNE), operandA: 42, operandB: 43)
    expectEqual(e3.expectedResult, 1, "BNE not equal")
    expect(e3.verified, "BNE verified")

    // BLT signed: -1 < 0
    let e4 = engine.verifyInstruction(.base(.BLT), operandA: 0xFFFFFFFF, operandB: 0)
    expectEqual(e4.expectedResult, 1, "BLT -1 < 0")
    expect(e4.verified, "BLT signed verified")

    // BGE
    let e5 = engine.verifyInstruction(.base(.BGE), operandA: 0, operandB: 0xFFFFFFFF)
    expectEqual(e5.expectedResult, 1, "BGE 0 >= -1")
    expect(e5.verified, "BGE verified")

    // BLTU unsigned
    let e6 = engine.verifyInstruction(.base(.BLTU), operandA: 5, operandB: 10)
    expectEqual(e6.expectedResult, 1, "BLTU 5 < 10")
    expect(e6.verified, "BLTU verified")

    // BGEU unsigned
    let e7 = engine.verifyInstruction(.base(.BGEU), operandA: 10, operandB: 5)
    expectEqual(e7.expectedResult, 1, "BGEU 10 >= 5")
    expect(e7.verified, "BGEU verified")

    // Verify categories
    expectEqual(e1.category, .branch, "BEQ category is Branch")
    expectEqual(e4.category, .branch, "BLT category is Branch")
}

private func testVerifyJump() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    // JAL/JALR: return address = PC + 4
    let e1 = engine.verifyInstruction(.base(.JAL), operandA: 0x100, operandB: 0x200)
    expectEqual(e1.expectedResult, 0x104, "JAL returns PC+4")
    expect(e1.verified, "JAL verified")
    expectEqual(e1.category, .jump, "JAL category is Jump")

    let e2 = engine.verifyInstruction(.base(.JALR), operandA: 0x200, operandB: 0x50)
    expectEqual(e2.expectedResult, 0x204, "JALR returns PC+4")
    expect(e2.verified, "JALR verified")
}

private func testVerifyUpperImmediate() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    // LUI: result = immediate
    let e1 = engine.verifyInstruction(.base(.LUI), operandA: 0, operandB: 0x12345000)
    expectEqual(e1.expectedResult, 0x12345000, "LUI loads upper immediate")
    expect(e1.verified, "LUI verified")
    expectEqual(e1.category, .upperImm, "LUI category is UpperImmediate")

    // AUIPC: result = PC + immediate
    let e2 = engine.verifyInstruction(.base(.AUIPC), operandA: 0x1000, operandB: 0x12345000)
    expectEqual(e2.expectedResult, 0x12346000, "AUIPC adds to PC")
    expect(e2.verified, "AUIPC verified")
}

private func testVerifyMExtension() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    // MUL
    let e1 = engine.verifyInstruction(.mul(.MUL), operandA: 6, operandB: 7)
    expectEqual(e1.expectedResult, 42, "MUL 6*7=42")
    expect(e1.verified, "MUL verified")
    expectEqual(e1.category, .mExtension, "MUL category is MExtension")

    // MUL overflow
    let e2 = engine.verifyInstruction(.mul(.MUL), operandA: 0xFFFFFFFF, operandB: 2)
    expectEqual(e2.expectedResult, 0xFFFFFFFE, "MUL -1*2 (lower 32)")
    expect(e2.verified, "MUL overflow verified")

    // MULHU
    let e3 = engine.verifyInstruction(.mul(.MULHU), operandA: 0xFFFFFFFF, operandB: 0xFFFFFFFF)
    expectEqual(e3.expectedResult, 0xFFFFFFFE, "MULHU max*max upper")
    expect(e3.verified, "MULHU verified")

    // DIV
    let e4 = engine.verifyInstruction(.mul(.DIV), operandA: 42, operandB: 7)
    expectEqual(e4.expectedResult, 6, "DIV 42/7=6")
    expect(e4.verified, "DIV verified")

    // DIV by zero
    let e5 = engine.verifyInstruction(.mul(.DIV), operandA: 42, operandB: 0)
    expectEqual(e5.expectedResult, 0xFFFFFFFF, "DIV by zero = -1")
    expect(e5.verified, "DIV by zero verified")

    // REM
    let e6 = engine.verifyInstruction(.mul(.REM), operandA: 42, operandB: 5)
    expectEqual(e6.expectedResult, 2, "REM 42%5=2")
    expect(e6.verified, "REM verified")

    // REM by zero
    let e7 = engine.verifyInstruction(.mul(.REM), operandA: 42, operandB: 0)
    expectEqual(e7.expectedResult, 42, "REM by zero = dividend")
    expect(e7.verified, "REM by zero verified")
}

private func testVerifySystem() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    let e1 = engine.verifyInstruction(.base(.ECALL), operandA: 0, operandB: 0)
    expectEqual(e1.expectedResult, 0, "ECALL result 0")
    expect(e1.verified, "ECALL verified")
    expectEqual(e1.category, .system, "ECALL category is System")

    let e2 = engine.verifyInstruction(.base(.FENCE), operandA: 0, operandB: 0)
    expect(e2.verified, "FENCE verified")
}

// MARK: - Batch Verification Tests

private func testBatchVerifySmall() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    let ops: [(instruction: RV32IMInstruction, a: UInt32, b: UInt32)] = [
        (instruction: .base(.ADD), a: 10, b: 20),
        (instruction: .base(.SUB), a: 50, b: 30),
        (instruction: .base(.AND), a: 0xFF, b: 0x0F),
    ]

    let result = engine.batchVerify(operations: ops)
    expectEqual(result.count, 3, "3 operations")
    expect(result.allVerified, "all 3 verified")
    expectEqual(result.entries[0].expectedResult, 30, "ADD result")
    expectEqual(result.entries[1].expectedResult, 20, "SUB result")
    expectEqual(result.entries[2].expectedResult, 0x0F, "AND result")
}

private func testBatchVerifyMixed() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    let ops: [(instruction: RV32IMInstruction, a: UInt32, b: UInt32)] = [
        (instruction: .base(.ADD), a: 100, b: 200),
        (instruction: .base(.BEQ), a: 42, b: 42),
        (instruction: .base(.LW), a: 0x1000, b: 0x100),
        (instruction: .mul(.MUL), a: 6, b: 7),
        (instruction: .base(.JAL), a: 0x100, b: 0),
        (instruction: .base(.LUI), a: 0, b: 0x12345000),
        (instruction: .base(.ECALL), a: 0, b: 0),
    ]

    let result = engine.batchVerify(operations: ops)
    expectEqual(result.count, 7, "7 mixed operations")
    expect(result.allVerified, "all mixed operations verified")

    // Check category counts
    let counts = result.categoryCounts
    expectEqual(counts[.alu], 1, "1 ALU op")
    expectEqual(counts[.branch], 1, "1 Branch op")
    expectEqual(counts[.memory], 1, "1 Memory op")
    expectEqual(counts[.mExtension], 1, "1 MExtension op")
    expectEqual(counts[.jump], 1, "1 Jump op")
    expectEqual(counts[.upperImm], 1, "1 UpperImm op")
    expectEqual(counts[.system], 1, "1 System op")
}

private func testBatchVerifyRandom() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    let baseInstrs: [RV32IInstruction] = [
        .ADD, .SUB, .AND, .OR, .XOR, .SLT, .SLTU,
        .SLL, .SRL, .SRA, .BEQ, .BNE, .BLT, .BLTU
    ]

    var rng: UInt64 = 0xDEADBEEF12345678
    var ops = [(instruction: RV32IMInstruction, a: UInt32, b: UInt32)]()

    for _ in 0..<100 {
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        let a = UInt32(truncatingIfNeeded: rng >> 16)
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        let b = UInt32(truncatingIfNeeded: rng >> 16)
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        let idx = Int(rng >> 32) % baseInstrs.count
        // For shifts, mask b to 0-31
        var bVal = b
        if [.SLL, .SRL, .SRA].contains(baseInstrs[idx]) {
            bVal = b & 31
        }
        ops.append((instruction: .base(baseInstrs[idx]), a: a, b: bVal))
    }

    let result = engine.batchVerify(operations: ops)
    expectEqual(result.count, 100, "100 random operations")
    expect(result.allVerified, "all 100 random operations verified")
}

private func testBatchVerifyLarge() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    // Large batch to potentially trigger GPU path
    let n = 2048
    var ops = [(instruction: RV32IMInstruction, a: UInt32, b: UInt32)]()
    ops.reserveCapacity(n)

    var rng: UInt64 = 0xCAFEBABE
    for i in 0..<n {
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        let a = UInt32(truncatingIfNeeded: rng >> 16)
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        let b = UInt32(truncatingIfNeeded: rng >> 16)
        // Alternate instructions
        let instr: RV32IMInstruction
        switch i % 7 {
        case 0: instr = .base(.ADD)
        case 1: instr = .base(.SUB)
        case 2: instr = .base(.AND)
        case 3: instr = .base(.OR)
        case 4: instr = .base(.XOR)
        case 5: instr = .mul(.MUL)
        default: instr = .base(.SLT)
        }
        ops.append((instruction: instr, a: a, b: b))
    }

    let result = engine.batchVerify(operations: ops)
    expectEqual(result.count, n, "large batch count")
    expect(result.allVerified, "all large batch operations verified")
}

private func testBatchVerifyAllVerified() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    // Verify passedCount matches count when all pass
    let ops: [(instruction: RV32IMInstruction, a: UInt32, b: UInt32)] = [
        (instruction: .base(.ADD), a: 1, b: 2),
        (instruction: .base(.ADD), a: 3, b: 4),
        (instruction: .base(.ADD), a: 5, b: 6),
    ]

    let result = engine.batchVerify(operations: ops)
    expectEqual(result.passedCount, 3, "passedCount = 3")
    expectEqual(result.passedCount, result.count, "passedCount = count")
}

// MARK: - Batch Decomposition Tests

private func testJoltInstrBatchDecomposeBasic() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    let ops: [(instruction: RV32IMInstruction, a: UInt32, b: UInt32)] = [
        (instruction: .base(.ADD), a: 100, b: 200),
        (instruction: .base(.AND), a: 0xFF, b: 0x0F),
    ]

    let records = engine.batchDecompose(operations: ops)
    expectEqual(records.count, 2, "2 decomposition records")
    expectEqual(records[0].result, 300, "ADD result")
    expectEqual(records[1].result, 0x0F, "AND result")
}

private func testBatchDecomposeRecords() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    let record = engine.batchDecompose(operations: [
        (instruction: .base(.XOR), a: 0xAAAAAAAA, b: 0x55555555)
    ])

    expectEqual(record.count, 1, "1 record")
    let r = record[0]
    expectEqual(r.result, 0xFFFFFFFF, "XOR complementary")
    expectEqual(r.subtableNames.count, 6, "6 subtable entries")
    expectEqual(r.subtableInputs.count, 6, "6 inputs")
    expectEqual(r.subtableOutputs.count, 6, "6 outputs")

    // Verify each subtable output matches the chunk of the result
    for (idx, output) in r.subtableOutputs.enumerated() {
        let expectedChunk = UInt64((0xFFFFFFFF >> (idx * 6)) & 0x3F)
        expectEqual(output, expectedChunk, "XOR chunk \(idx) output")
    }
}

private func testJoltInstrBatchDecomposeLarge() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    var ops = [(instruction: RV32IMInstruction, a: UInt32, b: UInt32)]()
    for i: UInt32 in 0..<500 {
        ops.append((instruction: .base(.ADD), a: i, b: i &* 2))
    }

    let records = engine.batchDecompose(operations: ops)
    expectEqual(records.count, 500, "500 records")

    for i in 0..<500 {
        let expected = UInt32(i) &+ UInt32(i) &* 2
        expectEqual(records[i].result, expected, "ADD \(i)+\(i*2) record")
    }
}

// MARK: - Fingerprint Tests

private func testFingerprintBasic() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    let records = engine.batchDecompose(operations: [
        (instruction: .base(.ADD), a: 10, b: 20)
    ])

    let challenge = frFromInt(42)
    let fp = engine.computeFingerprint(records: records, challenge: challenge)

    expect(fp.count > 0, "fingerprint has lookups")
    expectEqual(fp.count, 6, "ADD produces 6 lookups")
    expect(!frEqual(fp.fingerprint, Fr.zero), "fingerprint is nonzero")
}

private func testFingerprintDifferentChallenges() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    let records = engine.batchDecompose(operations: [
        (instruction: .base(.ADD), a: 100, b: 200)
    ])

    let fp1 = engine.computeFingerprint(records: records, challenge: frFromInt(42))
    let fp2 = engine.computeFingerprint(records: records, challenge: frFromInt(99))

    // Different challenges should produce different fingerprints
    expect(!frEqual(fp1.fingerprint, fp2.fingerprint),
           "different challenges produce different fingerprints")
}

private func testFingerprintSubtableBreakdown() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    let records = engine.batchDecompose(operations: [
        (instruction: .base(.AND), a: 0xFF, b: 0x0F),
        (instruction: .base(.OR), a: 0xFF, b: 0x0F),
    ])

    let challenge = frFromInt(7)
    let fp = engine.computeFingerprint(records: records, challenge: challenge)

    // Should have per-subtable fingerprints for and and or
    expect(fp.subtableFingerprints["and"] != nil, "AND subtable fingerprint present")
    expect(fp.subtableFingerprints["or"] != nil, "OR subtable fingerprint present")
    expectEqual(fp.count, 12, "AND(6) + OR(6) = 12 lookups")
}

private func testFingerprintDeterministic() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    let records = engine.batchDecompose(operations: [
        (instruction: .base(.XOR), a: 0xDEAD, b: 0xBEEF)
    ])

    let challenge = frFromInt(123)
    let fp1 = engine.computeFingerprint(records: records, challenge: challenge)
    let fp2 = engine.computeFingerprint(records: records, challenge: challenge)

    expect(frEqual(fp1.fingerprint, fp2.fingerprint),
           "same inputs produce same fingerprint")
    expectEqual(fp1.count, fp2.count, "same lookup count")
}

// MARK: - Lookup Argument Tests

private func testBuildLookupArgumentBasic() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    let records = engine.batchDecompose(operations: [
        (instruction: .base(.ADD), a: 100, b: 200)
    ])

    let argument = engine.buildLookupArgument(records: records)

    // ADD uses identity subtables
    expect(argument["identity"] != nil, "identity subtable present")
    expectEqual(argument["identity"]!.inputs.count, 6, "6 identity lookups")
    expectEqual(argument["identity"]!.outputs.count, 6, "6 identity outputs")
}

private func testBuildLookupArgumentMultipleSubtables() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    let records = engine.batchDecompose(operations: [
        (instruction: .base(.AND), a: 0xFF, b: 0x0F),
        (instruction: .base(.ADD), a: 10, b: 20),
        (instruction: .base(.XOR), a: 0xAA, b: 0x55),
    ])

    let argument = engine.buildLookupArgument(records: records)

    // Should have entries for and, identity, xor subtables
    expect(argument["and"] != nil, "and subtable in argument")
    expect(argument["identity"] != nil, "identity subtable in argument")
    expect(argument["xor"] != nil, "xor subtable in argument")

    // AND should have 6 lookups (6 chunks)
    expectEqual(argument["and"]!.inputs.count, 6, "6 AND lookups")
    // identity from ADD = 6 lookups
    expectEqual(argument["identity"]!.inputs.count, 6, "6 identity lookups")
    // XOR should have 6 lookups
    expectEqual(argument["xor"]!.inputs.count, 6, "6 XOR lookups")
}

private func testBuildLookupArgumentConsistency() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    // Verify that lookup outputs match subtable evaluations
    let records = engine.batchDecompose(operations: [
        (instruction: .base(.OR), a: 0xF0F0, b: 0x0F0F)
    ])

    let argument = engine.buildLookupArgument(records: records)
    let orSubtable = OrSubtable(chunkBits: 6)

    if let orArg = argument["or"] {
        for (i, input) in orArg.inputs.enumerated() {
            let expected = frFromInt(orSubtable.evaluate(input: input))
            expect(frEqual(orArg.outputs[i], expected),
                   "lookup argument output[\(i)] matches subtable evaluation")
        }
    } else {
        expect(false, "or subtable should be present")
    }
}

// MARK: - Chunk Extraction Tests

private func testChunkPairsBasic() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    let operands: [(UInt32, UInt32)] = [(10, 5)]
    let packed = engine.batchExtractChunkPairs(operands: operands)

    expectEqual(packed.count, 6, "6 chunks for 32-bit at C=6")

    // Chunk 0: a_chunk=10, b_chunk=5 -> packed = (10 << 6) | 5 = 645
    expectEqual(packed[0], (10 << 6) | 5, "chunk 0 packed correctly")
}

private func testChunkPairsEdgeCases() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    // All zeros
    let packed0 = engine.batchExtractChunkPairs(operands: [(0, 0)])
    for k in 0..<6 {
        expectEqual(packed0[k], 0, "zero operands chunk \(k) = 0")
    }

    // Max values
    let packedMax = engine.batchExtractChunkPairs(operands: [(0xFFFFFFFF, 0xFFFFFFFF)])
    for k in 0..<6 {
        let aChunk = UInt64((0xFFFFFFFF >> (k * 6)) & 0x3F)
        let bChunk = UInt64((0xFFFFFFFF >> (k * 6)) & 0x3F)
        let expected = (aChunk << 6) | bChunk
        expectEqual(packedMax[k], expected, "max operands chunk \(k)")
    }
}

private func testChunkPairsReassemble() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    // Verify that extracting chunks and reassembling round-trips
    let testValues: [(UInt32, UInt32)] = [
        (0xDEADBEEF, 0xCAFEBABE),
        (0x12345678, 0x9ABCDEF0),
        (1, 0),
        (0, 1),
    ]

    for (a, b) in testValues {
        let packed = engine.batchExtractChunkPairs(operands: [(a, b)])

        // Extract a-chunks and b-chunks, reassemble
        var aChunks = [UInt64]()
        var bChunks = [UInt64]()
        for k in 0..<6 {
            aChunks.append(packed[k] >> 6)
            bChunks.append(packed[k] & 0x3F)
        }

        let reassembledA = engine.reassembleFromChunks(aChunks)
        let reassembledB = engine.reassembleFromChunks(bChunks)
        expectEqual(reassembledA, a, "reassemble A for (0x\(String(a, radix: 16)))")
        expectEqual(reassembledB, b, "reassemble B for (0x\(String(b, radix: 16)))")
    }
}

private func testChunkPairsLargeBatch() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    var operands = [(UInt32, UInt32)]()
    var rng: UInt64 = 0xABCD1234
    for _ in 0..<2000 {
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        let a = UInt32(truncatingIfNeeded: rng >> 16)
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        let b = UInt32(truncatingIfNeeded: rng >> 16)
        operands.append((a, b))
    }

    let packed = engine.batchExtractChunkPairs(operands: operands)
    expectEqual(packed.count, 2000 * 6, "2000 pairs * 6 chunks")

    // Spot-check reassembly
    for i in stride(from: 0, to: 2000, by: 200) {
        let base = i * 6
        var aChunks = [UInt64]()
        var bChunks = [UInt64]()
        for k in 0..<6 {
            aChunks.append(packed[base + k] >> 6)
            bChunks.append(packed[base + k] & 0x3F)
        }
        let ra = engine.reassembleFromChunks(aChunks)
        let rb = engine.reassembleFromChunks(bChunks)
        expectEqual(ra, operands[i].0, "large batch reassemble A[\(i)]")
        expectEqual(rb, operands[i].1, "large batch reassemble B[\(i)]")
    }
}

// MARK: - Consistency Tests

private func testConsistencyAllPass() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    let ops: [(instruction: RV32IMInstruction, a: UInt32, b: UInt32)] = [
        (instruction: .base(.ADD), a: 1, b: 2),
        (instruction: .base(.SUB), a: 5, b: 3),
        (instruction: .base(.AND), a: 0xFF, b: 0x0F),
        (instruction: .mul(.MUL), a: 6, b: 7),
    ]

    let (consistent, failed) = engine.checkConsistency(operations: ops)
    expect(consistent, "all operations consistent")
    expectEqual(failed.count, 0, "no failed indices")
}

private func testConsistencyMixed() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    // Test with a wide range of instructions
    var ops = [(instruction: RV32IMInstruction, a: UInt32, b: UInt32)]()

    // ALU
    ops.append((instruction: .base(.ADD), a: 0xFFFFFFFF, b: 1))
    ops.append((instruction: .base(.SUB), a: 0, b: 1))
    ops.append((instruction: .base(.XOR), a: 0xAAAAAAAA, b: 0x55555555))

    // Memory
    ops.append((instruction: .base(.LW), a: 0x1000, b: 0x100))

    // Branch
    ops.append((instruction: .base(.BEQ), a: 42, b: 42))
    ops.append((instruction: .base(.BLT), a: 0xFFFFFFFF, b: 0))

    // M-extension
    ops.append((instruction: .mul(.DIV), a: 42, b: 0))
    ops.append((instruction: .mul(.REM), a: 0x80000000, b: 0xFFFFFFFF))

    let (consistent, failed) = engine.checkConsistency(operations: ops)
    expect(consistent, "mixed operations consistent")
    expectEqual(failed.count, 0, "no failures in mixed consistency")
}

private func testConsistencyRandom() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    let instrs: [RV32IMInstruction] = [
        .base(.ADD), .base(.SUB), .base(.AND), .base(.OR), .base(.XOR),
        .base(.SLT), .base(.SLTU), .base(.BEQ), .base(.BNE),
        .mul(.MUL), .mul(.DIV), .mul(.REM)
    ]

    var rng: UInt64 = 0x1234ABCD5678EF01
    var ops = [(instruction: RV32IMInstruction, a: UInt32, b: UInt32)]()

    for _ in 0..<200 {
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        let a = UInt32(truncatingIfNeeded: rng >> 16)
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        var b = UInt32(truncatingIfNeeded: rng >> 16)
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        let idx = Int(rng >> 32) % instrs.count
        // Avoid div by zero for DIV/REM
        if case .mul(let m) = instrs[idx], (m == .DIV || m == .REM) {
            if b == 0 { b = 1 }
        }
        ops.append((instruction: instrs[idx], a: a, b: b))
    }

    let (consistent, _) = engine.checkConsistency(operations: ops)
    expect(consistent, "200 random operations all consistent")
}

// MARK: - Trace Processing Tests

private func testProcessTraceBasic() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    let steps = [
        JoltStep(op: .add, a: 10, b: 20, result: 30),
        JoltStep(op: .sub, a: 50, b: 30, result: 20),
        JoltStep(op: .and_, a: 0xFF, b: 0x0F, result: 0x0F),
    ]

    let result = engine.processTrace(steps: steps)
    expectEqual(result.count, 3, "3 trace steps processed")
    expect(result.allVerified, "all trace steps verified")
}

private func testProcessTraceMixed() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    let steps = [
        JoltStep(op: .add, a: 100, b: 200, result: 300),
        JoltStep(op: .mul, a: 6, b: 7, result: 42),
        JoltStep(op: .or_, a: 0xF0, b: 0x0F, result: 0xFF),
        JoltStep(op: .xor_, a: 0xFF, b: 0xFF, result: 0),
        JoltStep(op: .shl, a: 1, b: 4, result: 16),
        JoltStep(op: .shr, a: 16, b: 2, result: 4),
        JoltStep(op: .lt, a: 5, b: 10, result: 1),
    ]

    let result = engine.processTrace(steps: steps)
    expectEqual(result.count, 7, "7 trace steps (eq excluded)")
    expect(result.allVerified, "all mixed trace steps verified")
}

// MARK: - Category Filter Tests

private func testVerifyByCategory() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    let ops: [(instruction: RV32IMInstruction, a: UInt32, b: UInt32)] = [
        (instruction: .base(.ADD), a: 1, b: 2),
        (instruction: .base(.SUB), a: 5, b: 3),
        (instruction: .base(.BEQ), a: 42, b: 42),
        (instruction: .base(.LW), a: 0x1000, b: 0x100),
        (instruction: .mul(.MUL), a: 6, b: 7),
    ]

    let aluResult = engine.verifyByCategory(operations: ops, category: .alu)
    expectEqual(aluResult.count, 2, "2 ALU ops filtered")
    expect(aluResult.allVerified, "ALU ops verified")

    let branchResult = engine.verifyByCategory(operations: ops, category: .branch)
    expectEqual(branchResult.count, 1, "1 Branch op filtered")

    let memResult = engine.verifyByCategory(operations: ops, category: .memory)
    expectEqual(memResult.count, 1, "1 Memory op filtered")

    let mExtResult = engine.verifyByCategory(operations: ops, category: .mExtension)
    expectEqual(mExtResult.count, 1, "1 MExtension op filtered")
}

private func testVerifyByCategoryEmpty() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    let ops: [(instruction: RV32IMInstruction, a: UInt32, b: UInt32)] = [
        (instruction: .base(.ADD), a: 1, b: 2),
    ]

    let branchResult = engine.verifyByCategory(operations: ops, category: .branch)
    expectEqual(branchResult.count, 0, "no branch ops in ALU-only batch")
    expect(branchResult.allVerified, "empty batch trivially verified")
}

// MARK: - Subtable Usage Tests

private func testSubtableUsageALU() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    let ops: [(instruction: RV32IMInstruction, a: UInt32, b: UInt32)] = [
        (instruction: .base(.AND), a: 0xFF, b: 0x0F),
        (instruction: .base(.AND), a: 0xAA, b: 0x55),
    ]

    let usage = engine.analyzeSubtableUsage(operations: ops)
    // Each AND uses 6 and-subtable lookups
    expectEqual(usage["and"], 12, "12 AND subtable lookups (2 * 6 chunks)")
}

private func testSubtableUsageMixed() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    let ops: [(instruction: RV32IMInstruction, a: UInt32, b: UInt32)] = [
        (instruction: .base(.AND), a: 0xFF, b: 0x0F),
        (instruction: .base(.OR), a: 0xFF, b: 0x0F),
        (instruction: .base(.ADD), a: 10, b: 20),
        (instruction: .base(.XOR), a: 0xAA, b: 0x55),
    ]

    let usage = engine.analyzeSubtableUsage(operations: ops)

    expectEqual(usage["and"], 6, "6 AND lookups")
    expectEqual(usage["or"], 6, "6 OR lookups")
    expectEqual(usage["identity"], 6, "6 identity lookups (from ADD)")
    expectEqual(usage["xor"], 6, "6 XOR lookups")
}

// MARK: - Statistics Tests

private func testEngineStats() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)
    let s = engine.stats()

    expectEqual(s.chunkBits, 6, "stats chunkBits=6")
    expectEqual(s.numChunks, 6, "stats numChunks=6")
    expect(s.gpuAvailable, "GPU should be available on Apple Silicon")
    expectEqual(s.gpuThreshold, 1024, "gpuThreshold=1024")
    expectEqual(s.supportedCategories.count, InstructionCategory.allCases.count,
                "all categories supported")
}

private func testEngineStatsChunk8() {
    let engine = GPUJoltInstructionEngine(chunkBits: 8)
    let s = engine.stats()

    expectEqual(s.chunkBits, 8, "stats chunkBits=8")
    expectEqual(s.numChunks, 4, "stats numChunks=4 at C=8")
}

// MARK: - Chunk Size Verification Tests

private func testChunkSize6Verification() {
    let engine = GPUJoltInstructionEngine(chunkBits: 6)

    expectEqual(engine.numChunks, 6, "32/6 = 6 chunks (ceil)")
    expectEqual(Int(engine.chunkMask), 63, "6-bit mask = 63")

    // Verify across all instruction types
    let cases: [(RV32IMInstruction, UInt32, UInt32)] = [
        (.base(.ADD), 0xFFFFFFFF, 1),
        (.base(.SUB), 0x80000000, 1),
        (.base(.AND), 0xAAAAAAAA, 0x55555555),
        (.base(.OR), 0xFF00FF00, 0x00FF00FF),
        (.base(.XOR), 0xDEADBEEF, 0xCAFEBABE),
        (.base(.SLT), 0xFFFFFFFF, 0),
        (.base(.SLTU), 0, 0xFFFFFFFF),
        (.base(.SLL), 0x12345678, 15),
        (.base(.SRL), 0xDEADBEEF, 12),
        (.base(.SRA), 0x80000000, 4),
        (.base(.BEQ), 42, 42),
        (.base(.BLT), 0xFFFFFFFF, 0),
        (.base(.LW), 0x1000, 0x100),
        (.base(.LUI), 0, 0xABCD0000),
        (.base(.JAL), 0x100, 0),
        (.mul(.MUL), 0xDEAD, 0xBEEF),
        (.mul(.DIV), 1000, 7),
    ]

    for (instr, a, b) in cases {
        let entry = engine.verifyInstruction(instr, operandA: a, operandB: b)
        expect(entry.verified, "C=6 verify \(instr) (0x\(String(a, radix: 16)), 0x\(String(b, radix: 16)))")
    }
}

private func testChunkSize8Verification() {
    let engine = GPUJoltInstructionEngine(chunkBits: 8)

    expectEqual(engine.numChunks, 4, "32/8 = 4 chunks")
    expectEqual(Int(engine.chunkMask), 255, "8-bit mask = 255")

    let cases: [(RV32IMInstruction, UInt32, UInt32)] = [
        (.base(.ADD), 0xFFFFFFFF, 1),
        (.base(.SUB), 0, 1),
        (.base(.AND), 0xFF00FF00, 0x0F0F0F0F),
        (.base(.XOR), 0xAAAAAAAA, 0x55555555),
        (.base(.SLT), 0x80000000, 0),
        (.base(.SLL), 0x12345678, 8),
        (.base(.BEQ), 0xDEADBEEF, 0xDEADBEEF),
        (.mul(.MUL), 0xDEAD, 0xBEEF),
        (.mul(.DIV), 42, 7),
        (.mul(.REMU), 100, 7),
    ]

    for (instr, a, b) in cases {
        let entry = engine.verifyInstruction(instr, operandA: a, operandB: b)
        expect(entry.verified, "C=8 verify \(instr)")
    }
}
