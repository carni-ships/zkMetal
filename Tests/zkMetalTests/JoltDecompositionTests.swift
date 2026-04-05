// Tests for Jolt instruction decomposition tables
//
// Validates that every RV32IM instruction decomposes into subtable lookups
// and recombines to produce the correct result. This is the core correctness
// guarantee for Jolt's lookup-based instruction verification.

import Foundation
import zkMetal

public func runJoltDecompositionTests() {
    suite("JoltSubtables")
    testIdentitySubtable()
    testTruncateSubtable()
    testBitwiseSubtables()
    testComparisonSubtables()
    testShiftSubtables()
    testSignExtendSubtable()
    testEQSubtable()

    suite("JoltDecomposition-ALU")
    testADDDecomposition()
    testSUBDecomposition()
    testANDDecomposition()
    testORDecomposition()
    testXORDecomposition()

    suite("JoltDecomposition-Shifts")
    testSLLDecomposition()
    testSRLDecomposition()
    testSRADecomposition()

    suite("JoltDecomposition-Comparison")
    testSLTDecomposition()
    testSLTUDecomposition()

    suite("JoltDecomposition-Branches")
    testBranchDecomposition()

    suite("JoltDecomposition-MExtension")
    testMULDecomposition()

    suite("JoltDecomposition-FullCoverage")
    testFullInstructionCoverage()

    suite("JoltDecomposition-ChunkSizes")
    testChunkSize6()
    testChunkSize8()

    suite("SubtableMaterializer")
    testMaterialize()
    testMaterializeAll()
    testMaterializeStats()
}

// MARK: - Subtable Unit Tests

func testIdentitySubtable() {
    let st = IdentitySubtable(chunkBits: 6)
    expectEqual(Int(st.tableSize), 64, "identity table size = 2^6 = 64")
    expectEqual(st.evaluate(input: 0), 0, "identity(0) = 0")
    expectEqual(st.evaluate(input: 42), 42, "identity(42) = 42")
    expectEqual(st.evaluate(input: 63), 63, "identity(63) = 63")

    let st8 = IdentitySubtable(chunkBits: 8)
    expectEqual(Int(st8.tableSize), 256, "identity table size = 2^8 = 256")
    expectEqual(st8.evaluate(input: 255), 255, "identity(255) = 255")
}

func testTruncateSubtable() {
    let st = TruncateSubtable(chunkBits: 8, truncBits: 4)
    expectEqual(Int(st.tableSize), 256, "truncate table size = 256")
    expectEqual(st.evaluate(input: 0xFF), 0x0F, "truncate_4(0xFF) = 0x0F")
    expectEqual(st.evaluate(input: 0x37), 0x07, "truncate_4(0x37) = 0x07")
    expectEqual(st.evaluate(input: 0x10), 0x00, "truncate_4(0x10) = 0x00")
}

func testBitwiseSubtables() {
    let c = 6
    let andST = AndSubtable(chunkBits: c)
    let orST = OrSubtable(chunkBits: c)
    let xorST = XorSubtable(chunkBits: c)

    let binarySize = 1 << (2 * c)
    expectEqual(andST.tableSize, binarySize, "and table size = 2^12 = 4096")

    // Test AND: 42 & 51
    let x: UInt64 = 42
    let y: UInt64 = 51
    let packed = (x << c) | y
    expectEqual(andST.evaluate(input: packed), 42 & 51, "AND(42, 51)")
    expectEqual(orST.evaluate(input: packed), 42 | 51, "OR(42, 51)")
    expectEqual(xorST.evaluate(input: packed), 42 ^ 51, "XOR(42, 51)")

    // Edge cases
    let zero = UInt64(0)
    expectEqual(andST.evaluate(input: (zero << c) | zero), 0, "AND(0, 0) = 0")
    let mask = UInt64((1 << c) - 1)
    expectEqual(andST.evaluate(input: (mask << c) | mask), mask, "AND(max, max) = max")
    expectEqual(orST.evaluate(input: (zero << c) | mask), mask, "OR(0, max) = max")
    expectEqual(xorST.evaluate(input: (mask << c) | mask), 0, "XOR(max, max) = 0")
}

func testComparisonSubtables() {
    let c = 6
    let ltuST = LTUSubtable(chunkBits: c)
    let ltST = LTSubtable(chunkBits: c)

    expectEqual(ltuST.evaluate(input: (5 << c) | 10), 1, "LTU(5, 10) = 1")
    expectEqual(ltuST.evaluate(input: (10 << c) | 5), 0, "LTU(10, 5) = 0")
    expectEqual(ltuST.evaluate(input: (5 << c) | 5), 0, "LTU(5, 5) = 0")

    // LT signed: with 6-bit chunks, values 32-63 are negative (sign bit = bit 5)
    expectEqual(ltST.evaluate(input: (63 << c) | 0), 1, "LT(-1, 0) = 1 (signed)")
    expectEqual(ltST.evaluate(input: (0 << c) | 63), 0, "LT(0, -1) = 0 (signed)")
    expectEqual(ltST.evaluate(input: (1 << c) | 2), 1, "LT(1, 2) = 1")
}

func testShiftSubtables() {
    let c = 6
    let sllST = SllSubtable(chunkBits: c)
    let srlST = SrlSubtable(chunkBits: c)
    let sraST = SraSubtable(chunkBits: c)
    let mask = UInt64((1 << c) - 1)

    expectEqual(sllST.evaluate(input: (1 << c) | 3), (1 << 3) & mask, "SLL(1, 3)")
    expectEqual(sllST.evaluate(input: (5 << c) | 1), (5 << 1) & mask, "SLL(5, 1)")

    expectEqual(srlST.evaluate(input: (8 << c) | 2), 2, "SRL(8, 2) = 2")
    expectEqual(srlST.evaluate(input: (63 << c) | 3), (63 >> 3), "SRL(63, 3)")

    // SRA: 32 in 6 bits = -32 signed -> SRA(-32, 1) = -16 = 48 in 6 bits unsigned
    expectEqual(sraST.evaluate(input: (32 << c) | 1), UInt64(48) & mask, "SRA(-32, 1) = -16")
}

func testSignExtendSubtable() {
    let st8 = SignExtendSubtable(chunkBits: 8, fromBits: 8)
    expectEqual(st8.evaluate(input: 0x7F), 0x7F, "sign_extend_8(0x7F) = 0x7F")
    expectEqual(st8.evaluate(input: 0x80), 0xFFFFFF80, "sign_extend_8(0x80) = 0xFFFFFF80")
    expectEqual(st8.evaluate(input: 0xFF), 0xFFFFFFFF, "sign_extend_8(0xFF) = 0xFFFFFFFF")
    expectEqual(st8.evaluate(input: 0x00), 0x00, "sign_extend_8(0x00) = 0x00")

    let st16 = SignExtendSubtable(chunkBits: 16, fromBits: 16)
    expectEqual(st16.evaluate(input: 0x7FFF), 0x7FFF, "sign_extend_16(0x7FFF)")
    expectEqual(st16.evaluate(input: 0x8000), 0xFFFF8000, "sign_extend_16(0x8000)")
}

func testEQSubtable() {
    let c = 6
    let eqST = EQSubtable(chunkBits: c)
    expectEqual(eqST.evaluate(input: (5 << c) | 5), 1, "EQ(5, 5) = 1")
    expectEqual(eqST.evaluate(input: (5 << c) | 6), 0, "EQ(5, 6) = 0")
    expectEqual(eqST.evaluate(input: (0 << c) | 0), 1, "EQ(0, 0) = 1")
}

// MARK: - Decomposition Round-Trip Tests

/// Helper: verify decompose -> evaluate -> combine produces the expected result.
func verifyDecomposition(
    _ decomposer: JoltInstructionDecomposer,
    _ instr: RV32IMInstruction,
    _ a: UInt32,
    _ b: UInt32,
    _ label: String,
    file: String = #file, line: Int = #line
) {
    let expected = rv32imExecute(instr, a, b)
    let lookups = decomposer.decompose(instruction: instr, operands: (a, b))
    let results = lookups.map { $0.subtable.evaluate(input: $0.input) }
    let combined = decomposer.combine(results: results, instruction: instr)
    if combined != expected {
        let loc = URL(fileURLWithPath: file).lastPathComponent
        print("  [FAIL] \(loc):\(line) \(label): got 0x\(String(combined, radix: 16)), expected 0x\(String(expected, radix: 16))")
        _failed += 1
    } else {
        _passed += 1
    }
}

// Convenience wrapper: base instruction
func vd(_ d: JoltInstructionDecomposer, _ i: RV32IInstruction, _ a: UInt32, _ b: UInt32, _ l: String,
         file: String = #file, line: Int = #line) {
    verifyDecomposition(d, .base(i), a, b, l, file: file, line: line)
}

// Convenience wrapper: M-extension instruction
func vdm(_ d: JoltInstructionDecomposer, _ i: RV32MInstruction, _ a: UInt32, _ b: UInt32, _ l: String,
          file: String = #file, line: Int = #line) {
    verifyDecomposition(d, .mul(i), a, b, l, file: file, line: line)
}

func testADDDecomposition() {
    let d = JoltInstructionDecomposer(chunkBits: 6)

    vd(d, .ADD, 100, 200, "ADD 100+200")
    vd(d, .ADD, 0xFFFFFFFF, 1, "ADD overflow")
    vd(d, .ADD, 0, 0, "ADD 0+0")
    vd(d, .ADD, 0x80000000, 0x80000000, "ADD large+large")
    vd(d, .ADDI, 0xDEADBEEF, 42, "ADDI")

    // Random values
    var rng: UInt64 = 0x1234
    for _ in 0..<20 {
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        let a = UInt32(truncatingIfNeeded: rng >> 16)
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        let b = UInt32(truncatingIfNeeded: rng >> 16)
        vd(d, .ADD, a, b, "ADD random")
    }
}

func testSUBDecomposition() {
    let d = JoltInstructionDecomposer(chunkBits: 6)
    vd(d, .SUB, 200, 100, "SUB 200-100")
    vd(d, .SUB, 0, 1, "SUB underflow")
    vd(d, .SUB, 0x80000000, 1, "SUB from min signed")
    vd(d, .SUB, 0xFFFFFFFF, 0xFFFFFFFF, "SUB equal")
}

func testANDDecomposition() {
    let d = JoltInstructionDecomposer(chunkBits: 6)
    vd(d, .AND, 0xFF00FF00, 0x0F0F0F0F, "AND masks")
    vd(d, .AND, 0xFFFFFFFF, 0, "AND with 0")
    vd(d, .AND, 0xAAAAAAAA, 0x55555555, "AND complementary")
    vd(d, .ANDI, 0xDEADBEEF, 0xFF, "ANDI")
}

func testORDecomposition() {
    let d = JoltInstructionDecomposer(chunkBits: 6)
    vd(d, .OR, 0xFF00FF00, 0x00FF00FF, "OR complementary")
    vd(d, .OR, 0, 0, "OR zeros")
    vd(d, .OR, 0xAAAAAAAA, 0x55555555, "OR all bits")
    vd(d, .ORI, 0xDEAD0000, 0xBEEF, "ORI")
}

func testXORDecomposition() {
    let d = JoltInstructionDecomposer(chunkBits: 6)
    vd(d, .XOR, 0xFF00FF00, 0x0F0F0F0F, "XOR")
    vd(d, .XOR, 0xFFFFFFFF, 0xFFFFFFFF, "XOR self = 0")
    vd(d, .XOR, 0, 0, "XOR zeros")
    vd(d, .XORI, 0xDEADBEEF, 0xFF, "XORI")
}

func testSLLDecomposition() {
    let d = JoltInstructionDecomposer(chunkBits: 6)
    vd(d, .SLL, 1, 0, "SLL by 0")
    vd(d, .SLL, 1, 1, "SLL by 1")
    vd(d, .SLL, 1, 31, "SLL by 31")
    vd(d, .SLL, 0xDEADBEEF, 4, "SLL by 4")
    vd(d, .SLL, 0xFFFFFFFF, 16, "SLL by 16")
    vd(d, .SLLI, 0xAABBCCDD, 8, "SLLI by 8")

    for shamt: UInt32 in 0..<32 {
        vd(d, .SLL, 0x12345678, shamt, "SLL by \(shamt)")
    }
}

func testSRLDecomposition() {
    let d = JoltInstructionDecomposer(chunkBits: 6)
    vd(d, .SRL, 0x80000000, 0, "SRL by 0")
    vd(d, .SRL, 0x80000000, 1, "SRL by 1")
    vd(d, .SRL, 0x80000000, 31, "SRL by 31")
    vd(d, .SRL, 0xFFFFFFFF, 16, "SRL by 16")
    vd(d, .SRLI, 0xDEADBEEF, 4, "SRLI by 4")

    for shamt: UInt32 in 0..<32 {
        vd(d, .SRL, 0xDEADBEEF, shamt, "SRL by \(shamt)")
    }
}

func testSRADecomposition() {
    let d = JoltInstructionDecomposer(chunkBits: 6)
    vd(d, .SRA, 0x7FFFFFFF, 1, "SRA positive by 1")
    vd(d, .SRA, 0x80000000, 1, "SRA negative by 1")
    vd(d, .SRA, 0x80000000, 31, "SRA negative by 31")
    vd(d, .SRA, 0xFFFFFFFF, 4, "SRA -1 by 4")
    vd(d, .SRAI, 0xDEADBEEF, 8, "SRAI by 8")

    for shamt: UInt32 in 0..<32 {
        vd(d, .SRA, 0x80000000, shamt, "SRA by \(shamt)")
        vd(d, .SRA, 0x12345678, shamt, "SRA pos by \(shamt)")
    }
}

func testSLTDecomposition() {
    let d = JoltInstructionDecomposer(chunkBits: 6)
    vd(d, .SLT, 5, 10, "SLT 5 < 10")
    vd(d, .SLT, 10, 5, "SLT 10 >= 5")
    vd(d, .SLT, 5, 5, "SLT 5 == 5")
    vd(d, .SLT, 0xFFFFFFFF, 0, "SLT -1 < 0")
    vd(d, .SLT, 0, 0xFFFFFFFF, "SLT 0 >= -1")
    vd(d, .SLT, 0x80000000, 0x7FFFFFFF, "SLT min < max")
    vd(d, .SLT, 0x7FFFFFFF, 0x80000000, "SLT max >= min")
    vd(d, .SLTI, 0xDEADBEEF, 100, "SLTI")
}

func testSLTUDecomposition() {
    let d = JoltInstructionDecomposer(chunkBits: 6)
    vd(d, .SLTU, 5, 10, "SLTU 5 < 10")
    vd(d, .SLTU, 10, 5, "SLTU 10 >= 5")
    vd(d, .SLTU, 5, 5, "SLTU 5 == 5")
    vd(d, .SLTU, 0, 0xFFFFFFFF, "SLTU 0 < max")
    vd(d, .SLTU, 0xFFFFFFFF, 0, "SLTU max >= 0")
    vd(d, .SLTIU, 100, 200, "SLTIU")
}

func testBranchDecomposition() {
    let d = JoltInstructionDecomposer(chunkBits: 6)

    vd(d, .BEQ, 42, 42, "BEQ equal")
    vd(d, .BEQ, 42, 43, "BEQ not equal")
    vd(d, .BNE, 42, 43, "BNE not equal")
    vd(d, .BNE, 42, 42, "BNE equal")
    vd(d, .BLT, 0xFFFFFFFF, 0, "BLT -1 < 0")
    vd(d, .BLT, 0, 0xFFFFFFFF, "BLT 0 >= -1")
    vd(d, .BGE, 0, 0xFFFFFFFF, "BGE 0 >= -1")
    vd(d, .BGE, 0xFFFFFFFF, 0, "BGE -1 < 0")
    vd(d, .BLTU, 5, 10, "BLTU 5 < 10")
    vd(d, .BLTU, 10, 5, "BLTU 10 >= 5")
    vd(d, .BGEU, 10, 5, "BGEU 10 >= 5")
    vd(d, .BGEU, 5, 10, "BGEU 5 < 10")
}

func testMULDecomposition() {
    let d = JoltInstructionDecomposer(chunkBits: 6)

    vdm(d, .MUL, 6, 7, "MUL 6*7=42")
    vdm(d, .MUL, 0xFFFFFFFF, 2, "MUL -1*2")
    vdm(d, .MUL, 0x10000, 0x10000, "MUL overflow")
    vdm(d, .MUL, 0, 0xDEADBEEF, "MUL by 0")

    vdm(d, .MULH, 0x7FFFFFFF, 2, "MULH large")
    vdm(d, .MULH, 0xFFFFFFFF, 0xFFFFFFFF, "MULH -1*-1")

    vdm(d, .MULHU, 0xFFFFFFFF, 0xFFFFFFFF, "MULHU max*max")
    vdm(d, .MULHU, 100, 200, "MULHU small")

    vdm(d, .DIV, 42, 7, "DIV 42/7")
    vdm(d, .DIV, 42, 0, "DIV by zero")
    vdm(d, .DIVU, 42, 7, "DIVU 42/7")
    vdm(d, .REM, 42, 5, "REM 42%5")
    vdm(d, .REM, 42, 0, "REM by zero")
    vdm(d, .REMU, 42, 5, "REMU 42%5")

    // Signed division overflow: INT_MIN / -1
    vdm(d, .DIV, 0x80000000, 0xFFFFFFFF, "DIV overflow")
    vdm(d, .REM, 0x80000000, 0xFFFFFFFF, "REM overflow")
}

// MARK: - Full Coverage Tests

func testFullInstructionCoverage() {
    let d = JoltInstructionDecomposer(chunkBits: 6)

    // All RV32I base instructions
    let baseCases: [(RV32IInstruction, UInt32, UInt32)] = [
        (.ADD, 0xAABBCCDD, 0x11223344),
        (.SUB, 0xAABBCCDD, 0x11223344),
        (.AND, 0xFF00FF00, 0x0F0F0F0F),
        (.OR,  0xFF00FF00, 0x0F0F0F0F),
        (.XOR, 0xFF00FF00, 0x0F0F0F0F),
        (.SLT, 0x80000000, 0x7FFFFFFF),
        (.SLTU, 0x80000000, 0x7FFFFFFF),
        (.SLL, 0xDEADBEEF, 12),
        (.SRL, 0xDEADBEEF, 12),
        (.SRA, 0xDEADBEEF, 12),
        (.ADDI, 0xDEADBEEF, 42),
        (.ANDI, 0xDEADBEEF, 0xFF),
        (.ORI,  0xDEADBEEF, 0xF0),
        (.XORI, 0xDEADBEEF, 0x55),
        (.SLTI, 0xFFFFFFFF, 0),
        (.SLTIU, 0xFFFFFFFF, 0),
        (.SLLI, 0xAABBCCDD, 8),
        (.SRLI, 0xAABBCCDD, 8),
        (.SRAI, 0xAABBCCDD, 8),
        (.BEQ, 42, 42),
        (.BNE, 42, 43),
        (.BLT, 0xFFFFFFFF, 0),
        (.BGE, 0, 0xFFFFFFFF),
        (.BLTU, 5, 10),
        (.BGEU, 10, 5),
        (.LUI, 0, 0x12345000),
        (.AUIPC, 0x1000, 0x12345000),
        (.JAL, 0x100, 0x200),
        (.JALR, 0x100, 0x50),
        (.LW, 0x1000, 0x100),
        (.SW, 0x1000, 0x100),
        (.LB, 0x2000, 0x10),
        (.LH, 0x2000, 0x10),
        (.LBU, 0x2000, 0x10),
        (.LHU, 0x2000, 0x10),
        (.SB, 0x2000, 0x10),
        (.SH, 0x2000, 0x10),
        (.ECALL, 0, 0),
        (.EBREAK, 0, 0),
        (.FENCE, 0, 0),
    ]

    for (instr, a, b) in baseCases {
        vd(d, instr, a, b, "\(instr) full coverage")
    }

    // All M-extension instructions
    let mulCases: [(RV32MInstruction, UInt32, UInt32)] = [
        (.MUL, 0xDEAD, 0xBEEF),
        (.MULH, 0x7FFFFFFF, 0x7FFFFFFF),
        (.MULHSU, 0x80000000, 0xFFFFFFFF),
        (.MULHU, 0xFFFFFFFF, 0xFFFFFFFF),
        (.DIV, 1000, 7),
        (.DIVU, 1000, 7),
        (.REM, 1000, 7),
        (.REMU, 1000, 7),
    ]

    for (instr, a, b) in mulCases {
        vdm(d, instr, a, b, "\(instr) full coverage")
    }
}

// MARK: - Chunk Size Tests

func testChunkSize6() {
    let d = JoltInstructionDecomposer(chunkBits: 6)
    expectEqual(d.numChunks, 6, "32/6 = 6 chunks (ceil)")
    expectEqual(Int(d.chunkMask), 63, "6-bit mask = 63")

    vd(d, .ADD, 0xFFFFFFFF, 1, "C=6 ADD overflow")
    vd(d, .XOR, 0xAAAAAAAA, 0x55555555, "C=6 XOR")
    vd(d, .SLT, 0x80000000, 0, "C=6 SLT")
}

func testChunkSize8() {
    let d = JoltInstructionDecomposer(chunkBits: 8)
    expectEqual(d.numChunks, 4, "32/8 = 4 chunks")
    expectEqual(Int(d.chunkMask), 255, "8-bit mask = 255")

    vd(d, .ADD, 0xFFFFFFFF, 1, "C=8 ADD overflow")
    vd(d, .XOR, 0xAAAAAAAA, 0x55555555, "C=8 XOR")
    vd(d, .SLT, 0x80000000, 0, "C=8 SLT")
    vdm(d, .MUL, 0xDEAD, 0xBEEF, "C=8 MUL")
    vd(d, .SLL, 0x12345678, 15, "C=8 SLL")
}

// MARK: - Materializer Tests

func testMaterialize() {
    let identity = IdentitySubtable(chunkBits: 6)
    let table = SubtableMaterializer.materialize(subtable: identity, chunkBits: 6)
    expectEqual(table.count, 64, "identity table has 64 entries")
    for i in 0..<64 {
        expectEqual(table[i], UInt64(i), "identity[\(i)] = \(i)")
    }

    let andST = AndSubtable(chunkBits: 6)
    let andTable = SubtableMaterializer.materialize(subtable: andST, chunkBits: 6)
    expectEqual(andTable.count, 4096, "and table has 4096 entries")
    // Spot check: AND(5, 3) at index (5 << 6) | 3 = 323
    expectEqual(andTable[323], 5 & 3, "and[5,3] = 1")
}

func testMaterializeAll() {
    let tables6 = SubtableMaterializer.materializeAll(chunkBits: 6)
    expect(tables6.count >= 10, "materializeAll produces >= 10 tables")
    expect(tables6["identity"] != nil, "identity table present")
    expect(tables6["and"] != nil, "and table present")
    expect(tables6["or"] != nil, "or table present")
    expect(tables6["xor"] != nil, "xor table present")
    expect(tables6["eq"] != nil, "eq table present")
    expect(tables6["ltu"] != nil, "ltu table present")
    expect(tables6["lt_signed"] != nil, "lt_signed table present")

    expectEqual(tables6["identity"]!.count, 64, "identity size at C=6")
    expectEqual(tables6["and"]!.count, 4096, "and size at C=6")

    let tables8 = SubtableMaterializer.materializeAll(chunkBits: 8)
    expectEqual(tables8["identity"]!.count, 256, "identity size at C=8")
    expectEqual(tables8["and"]!.count, 65536, "and size at C=8")
}

func testMaterializeStats() {
    let stats6 = SubtableMaterializer.stats(chunkBits: 6)
    expectEqual(stats6.chunkBits, 6, "stats chunkBits")
    expectEqual(stats6.unaryTableSize, 64, "unary size at C=6")
    expectEqual(stats6.binaryTableSize, 4096, "binary size at C=6")
    expectEqual(stats6.totalSubtables, 14, "14 subtables total")
    expect(stats6.totalEntries > 0, "positive total entries")

    let stats8 = SubtableMaterializer.stats(chunkBits: 8)
    expectEqual(stats8.unaryTableSize, 256, "unary size at C=8")
    expectEqual(stats8.binaryTableSize, 65536, "binary size at C=8")
}
