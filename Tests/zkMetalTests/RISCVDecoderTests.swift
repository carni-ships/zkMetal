// Tests for RISCVDecoder and ExecutionTraceGenerator

import Foundation
import zkMetal

func runRISCVDecoderTests() {
    suite("RISCVDecoder")

    testDecodeRType()
    testDecodeIType()
    testDecodeSType()
    testDecodeBType()
    testDecodeUType()
    testDecodeJType()
    testDecodeSystem()
    testEncodeDecodeRoundtrip()
    testImmediateSignExtension()

    suite("ExecutionTraceGenerator")

    testFibonacci()
    testFactorial()
    testMemoryOperations()
    testBranching()
    testALUOperations()
    testTraceMetadata()
    testX0AlwaysZero()
}

// MARK: - Decoder Tests

func testDecodeRType() {
    // ADD x3, x1, x2
    let word = encodeADD(rd: 3, rs1: 1, rs2: 2)
    do {
        let d = try decodeInstruction(word: word)
        expect(d.instruction == .ADD, "ADD decoded")
        expect(d.format == .R, "ADD is R-type")
        expectEqual(d.rd, 3, "ADD rd=3")
        expectEqual(d.rs1, 1, "ADD rs1=1")
        expectEqual(d.rs2, 2, "ADD rs2=2")
        expectEqual(d.funct3, 0, "ADD funct3=0")
        expectEqual(d.funct7, 0, "ADD funct7=0")
        expectEqual(d.immediate, 0, "ADD imm=0")
    } catch {
        expect(false, "ADD decode threw: \(error)")
    }

    // SUB x5, x6, x7
    let subWord = encodeSUB(rd: 5, rs1: 6, rs2: 7)
    do {
        let d = try decodeInstruction(word: subWord)
        expect(d.instruction == .SUB, "SUB decoded")
        expectEqual(d.funct7, 0x20, "SUB funct7=0x20")
        expectEqual(d.rd, 5, "SUB rd=5")
    } catch {
        expect(false, "SUB decode threw: \(error)")
    }

    // SRA x8, x9, x10
    let sraWord = encodeSRA(rd: 8, rs1: 9, rs2: 10)
    do {
        let d = try decodeInstruction(word: sraWord)
        expect(d.instruction == .SRA, "SRA decoded")
        expectEqual(d.funct7, 0x20, "SRA funct7=0x20")
    } catch {
        expect(false, "SRA decode threw: \(error)")
    }

    // All 10 R-type instructions
    let rTypeTests: [(UInt32, RV32IInstruction)] = [
        (encodeADD(rd: 1, rs1: 2, rs2: 3), .ADD),
        (encodeSUB(rd: 1, rs1: 2, rs2: 3), .SUB),
        (encodeSLL(rd: 1, rs1: 2, rs2: 3), .SLL),
        (encodeSLT(rd: 1, rs1: 2, rs2: 3), .SLT),
        (encodeSLTU(rd: 1, rs1: 2, rs2: 3), .SLTU),
        (encodeXOR(rd: 1, rs1: 2, rs2: 3), .XOR),
        (encodeSRL(rd: 1, rs1: 2, rs2: 3), .SRL),
        (encodeSRA(rd: 1, rs1: 2, rs2: 3), .SRA),
        (encodeOR(rd: 1, rs1: 2, rs2: 3), .OR),
        (encodeAND(rd: 1, rs1: 2, rs2: 3), .AND),
    ]
    for (w, expected) in rTypeTests {
        do {
            let d = try decodeInstruction(word: w)
            expect(d.instruction == expected, "\(expected) R-type roundtrip")
        } catch {
            expect(false, "\(expected) R-type decode threw: \(error)")
        }
    }
}

func testDecodeIType() {
    // ADDI x5, x1, 42
    let word = encodeADDI(rd: 5, rs1: 1, imm: 42)
    do {
        let d = try decodeInstruction(word: word)
        expect(d.instruction == .ADDI, "ADDI decoded")
        expect(d.format == .I, "ADDI is I-type")
        expectEqual(d.rd, 5, "ADDI rd=5")
        expectEqual(d.rs1, 1, "ADDI rs1=1")
        expectEqual(d.immediate, 42, "ADDI imm=42")
    } catch {
        expect(false, "ADDI decode threw: \(error)")
    }

    // ADDI with negative immediate
    let negWord = encodeADDI(rd: 5, rs1: 1, imm: -1)
    do {
        let d = try decodeInstruction(word: negWord)
        expect(d.instruction == .ADDI, "ADDI neg decoded")
        expectEqual(d.immediate, -1, "ADDI imm=-1")
    } catch {
        expect(false, "ADDI neg decode threw: \(error)")
    }

    // SLLI x3, x1, 5
    let slliWord = encodeSLLI(rd: 3, rs1: 1, shamt: 5)
    do {
        let d = try decodeInstruction(word: slliWord)
        expect(d.instruction == .SLLI, "SLLI decoded")
        expectEqual(d.immediate, 5, "SLLI shamt=5")
    } catch {
        expect(false, "SLLI decode threw: \(error)")
    }

    // SRLI x3, x1, 7
    let srliWord = encodeSRLI(rd: 3, rs1: 1, shamt: 7)
    do {
        let d = try decodeInstruction(word: srliWord)
        expect(d.instruction == .SRLI, "SRLI decoded")
    } catch {
        expect(false, "SRLI decode threw: \(error)")
    }

    // SRAI x3, x1, 4
    let sraiWord = encodeSRAI(rd: 3, rs1: 1, shamt: 4)
    do {
        let d = try decodeInstruction(word: sraiWord)
        expect(d.instruction == .SRAI, "SRAI decoded")
    } catch {
        expect(false, "SRAI decode threw: \(error)")
    }
}

func testDecodeSType() {
    // SW x5, 8(x1)
    let word = encodeSW(rs2: 5, rs1: 1, offset: 8)
    do {
        let d = try decodeInstruction(word: word)
        expect(d.instruction == .SW, "SW decoded")
        expect(d.format == .S, "SW is S-type")
        expectEqual(d.rs1, 1, "SW rs1=1")
        expectEqual(d.rs2, 5, "SW rs2=5")
        expectEqual(d.immediate, 8, "SW imm=8")
        expectEqual(d.rd, 0, "SW rd=0 (stores have no rd)")
    } catch {
        expect(false, "SW decode threw: \(error)")
    }

    // SW with negative offset
    let negWord = encodeSW(rs2: 5, rs1: 1, offset: -4)
    do {
        let d = try decodeInstruction(word: negWord)
        expect(d.instruction == .SW, "SW neg decoded")
        expectEqual(d.immediate, -4, "SW imm=-4")
    } catch {
        expect(false, "SW neg decode threw: \(error)")
    }
}

func testDecodeBType() {
    // BEQ x1, x2, +8
    let word = encodeBEQ(rs1: 1, rs2: 2, offset: 8)
    do {
        let d = try decodeInstruction(word: word)
        expect(d.instruction == .BEQ, "BEQ decoded")
        expect(d.format == .B, "BEQ is B-type")
        expectEqual(d.rs1, 1, "BEQ rs1=1")
        expectEqual(d.rs2, 2, "BEQ rs2=2")
        expectEqual(d.immediate, 8, "BEQ offset=8")
    } catch {
        expect(false, "BEQ decode threw: \(error)")
    }

    // BNE with negative offset
    let bneWord = encodeBNE(rs1: 3, rs2: 4, offset: -16)
    do {
        let d = try decodeInstruction(word: bneWord)
        expect(d.instruction == .BNE, "BNE decoded")
        expectEqual(d.immediate, -16, "BNE offset=-16")
    } catch {
        expect(false, "BNE decode threw: \(error)")
    }

    // BLT x5, x6, +12
    let bltWord = encodeBLT(rs1: 5, rs2: 6, offset: 12)
    do {
        let d = try decodeInstruction(word: bltWord)
        expect(d.instruction == .BLT, "BLT decoded")
        expectEqual(d.immediate, 12, "BLT offset=12")
    } catch {
        expect(false, "BLT decode threw: \(error)")
    }
}

func testDecodeUType() {
    // LUI x5, 0x12345
    let luiWord = encodeLUI(rd: 5, imm: 0x12345)
    do {
        let d = try decodeInstruction(word: luiWord)
        expect(d.instruction == .LUI, "LUI decoded")
        expect(d.format == .U, "LUI is U-type")
        expectEqual(d.rd, 5, "LUI rd=5")
        // Upper 20 bits: 0x12345 << 12 = 0x12345000
        expectEqual(d.immediate, Int32(bitPattern: 0x12345000), "LUI imm=0x12345000")
    } catch {
        expect(false, "LUI decode threw: \(error)")
    }

    // AUIPC x3, 0x1000
    let auipcWord = encodeAUIPC(rd: 3, imm: 0x1000)
    do {
        let d = try decodeInstruction(word: auipcWord)
        expect(d.instruction == .AUIPC, "AUIPC decoded")
        expectEqual(d.rd, 3, "AUIPC rd=3")
    } catch {
        expect(false, "AUIPC decode threw: \(error)")
    }
}

func testDecodeJType() {
    // JAL x1, +100
    let jalWord = encodeJAL(rd: 1, offset: 100)
    do {
        let d = try decodeInstruction(word: jalWord)
        expect(d.instruction == .JAL, "JAL decoded")
        expect(d.format == .J, "JAL is J-type")
        expectEqual(d.rd, 1, "JAL rd=1")
        expectEqual(d.immediate, 100, "JAL offset=100")
    } catch {
        expect(false, "JAL decode threw: \(error)")
    }

    // JAL with negative offset
    let jalNeg = encodeJAL(rd: 1, offset: -200)
    do {
        let d = try decodeInstruction(word: jalNeg)
        expect(d.instruction == .JAL, "JAL neg decoded")
        expectEqual(d.immediate, -200, "JAL offset=-200")
    } catch {
        expect(false, "JAL neg decode threw: \(error)")
    }

    // JALR x1, x5, 12
    let jalrWord = encodeJALR(rd: 1, rs1: 5, offset: 12)
    do {
        let d = try decodeInstruction(word: jalrWord)
        expect(d.instruction == .JALR, "JALR decoded")
        expectEqual(d.rd, 1, "JALR rd=1")
        expectEqual(d.rs1, 5, "JALR rs1=5")
        expectEqual(d.immediate, 12, "JALR offset=12")
    } catch {
        expect(false, "JALR decode threw: \(error)")
    }
}

func testDecodeSystem() {
    // ECALL
    do {
        let d = try decodeInstruction(word: encodeECALL())
        expect(d.instruction == .ECALL, "ECALL decoded")
    } catch {
        expect(false, "ECALL decode threw: \(error)")
    }

    // EBREAK
    do {
        let d = try decodeInstruction(word: encodeEBREAK())
        expect(d.instruction == .EBREAK, "EBREAK decoded")
    } catch {
        expect(false, "EBREAK decode threw: \(error)")
    }
}

func testEncodeDecodeRoundtrip() {
    // Encode -> decode -> verify all fields survive the roundtrip
    let testCases: [(UInt32, RV32IInstruction, UInt8, UInt8, UInt8)] = [
        (encodeADD(rd: 15, rs1: 20, rs2: 25), .ADD, 15, 20, 25),
        (encodeSUB(rd: 1, rs1: 2, rs2: 3), .SUB, 1, 2, 3),
        (encodeAND(rd: 31, rs1: 0, rs2: 31), .AND, 31, 0, 31),
    ]
    for (word, expectedInstr, expectedRd, expectedRs1, expectedRs2) in testCases {
        do {
            let d = try decodeInstruction(word: word)
            expect(d.instruction == expectedInstr, "\(expectedInstr) roundtrip instruction")
            expectEqual(d.rd, expectedRd, "\(expectedInstr) roundtrip rd")
            expectEqual(d.rs1, expectedRs1, "\(expectedInstr) roundtrip rs1")
            expectEqual(d.rs2, expectedRs2, "\(expectedInstr) roundtrip rs2")
            expectEqual(d.word, word, "\(expectedInstr) roundtrip word preserved")
        } catch {
            expect(false, "\(expectedInstr) roundtrip threw: \(error)")
        }
    }
}

func testImmediateSignExtension() {
    // I-type: max positive 12-bit = 2047
    let pos = encodeADDI(rd: 1, rs1: 0, imm: 2047)
    do {
        let d = try decodeInstruction(word: pos)
        expectEqual(d.immediate, 2047, "I-type max positive imm")
    } catch {
        expect(false, "I-type positive threw: \(error)")
    }

    // I-type: min negative 12-bit = -2048
    let neg = encodeADDI(rd: 1, rs1: 0, imm: -2048)
    do {
        let d = try decodeInstruction(word: neg)
        expectEqual(d.immediate, -2048, "I-type min negative imm")
    } catch {
        expect(false, "I-type negative threw: \(error)")
    }

    // B-type: positive offset
    let bPos = encodeBEQ(rs1: 0, rs2: 0, offset: 4094)
    do {
        let d = try decodeInstruction(word: bPos)
        expectEqual(d.immediate, 4094, "B-type max positive offset")
    } catch {
        expect(false, "B-type positive threw: \(error)")
    }

    // J-type: large positive offset
    let jPos = encodeJAL(rd: 0, offset: 1048574)
    do {
        let d = try decodeInstruction(word: jPos)
        expectEqual(d.immediate, 1048574, "J-type large positive offset")
    } catch {
        expect(false, "J-type positive threw: \(error)")
    }
}

// MARK: - Execution Trace Tests

func testFibonacci() {
    // Compute fib(10) = 55 (0-indexed: fib(0)=0, fib(1)=1, ..., fib(10)=55)
    let program = rv32iFibonacciProgram()
    var initRegs = [UInt32](repeating: 0, count: 32)
    initRegs[10] = 10  // n = 10

    let trace = generateTrace(program: program, initialRegisters: initRegs)

    expect(trace.terminated, "Fibonacci terminated")
    expectEqual(trace.finalRegisters[10], 55, "fib(10) = 55")

    // fib(1) = 1
    initRegs[10] = 1
    let trace1 = generateTrace(program: program, initialRegisters: initRegs)
    expect(trace1.terminated, "fib(1) terminated")
    expectEqual(trace1.finalRegisters[10], 1, "fib(1) = 1")

    // fib(5) = 5 (0-indexed: fib(0)=0, fib(1)=1, ..., fib(5)=5)
    initRegs[10] = 5
    let trace5 = generateTrace(program: program, initialRegisters: initRegs)
    expect(trace5.terminated, "fib(5) terminated")
    expectEqual(trace5.finalRegisters[10], 5, "fib(5) = 5")
}

func testFactorial() {
    // Sum 1..10 = 55
    let program = rv32iFactorialProgram()
    var initRegs = [UInt32](repeating: 0, count: 32)
    initRegs[10] = 10

    let trace = generateTrace(program: program, initialRegisters: initRegs)

    expect(trace.terminated, "Sum program terminated")
    expectEqual(trace.finalRegisters[10], 55, "sum(1..10) = 55")

    // Sum 1..1 = 1
    initRegs[10] = 1
    let trace1 = generateTrace(program: program, initialRegisters: initRegs)
    expect(trace1.terminated, "sum(1) terminated")
    expectEqual(trace1.finalRegisters[10], 1, "sum(1..1) = 1")

    // Sum 1..0 = 0
    initRegs[10] = 0
    let trace0 = generateTrace(program: program, initialRegisters: initRegs)
    expect(trace0.terminated, "sum(0) terminated")
    expectEqual(trace0.finalRegisters[10], 0, "sum(1..0) = 0")
}

func testMemoryOperations() {
    let program = rv32iMemoryTestProgram()
    let trace = generateTrace(program: program)

    expect(trace.terminated, "Memory test terminated")
    expectEqual(trace.finalRegisters[10], 141, "42 + 99 = 141")
    expectEqual(trace.finalRegisters[12], 42, "loaded 42 from memory")
    expectEqual(trace.finalRegisters[13], 99, "loaded 99 from memory")

    // Verify memory access counts
    expect(trace.storeCount == 2, "2 stores")
    expect(trace.loadCount == 2, "2 loads")
    expect(trace.memoryAccessCount == 4, "4 total memory ops")
}

func testBranching() {
    // Simple program: if x1 == x2, x3 = 1, else x3 = 0
    let program: [UInt32] = [
        encodeADDI(rd: 1, rs1: 0, imm: 5),       // x1 = 5
        encodeADDI(rd: 2, rs1: 0, imm: 5),       // x2 = 5
        encodeBEQ(rs1: 1, rs2: 2, offset: 8),    // if x1==x2 skip next
        encodeADDI(rd: 3, rs1: 0, imm: 0),       // x3 = 0 (skipped)
        encodeADDI(rd: 3, rs1: 0, imm: 1),       // x3 = 1
        encodeECALL(),
    ]

    let trace = generateTrace(program: program)
    expect(trace.terminated, "Branch test terminated")
    expectEqual(trace.finalRegisters[3], 1, "Branch taken, x3=1")
    expectEqual(trace.branchesTaken, 1, "1 branch taken")

    // Not-taken case
    let program2: [UInt32] = [
        encodeADDI(rd: 1, rs1: 0, imm: 5),       // x1 = 5
        encodeADDI(rd: 2, rs1: 0, imm: 3),       // x2 = 3
        encodeBEQ(rs1: 1, rs2: 2, offset: 8),    // if x1==x2 skip (not taken)
        encodeADDI(rd: 3, rs1: 0, imm: 77),      // x3 = 77
        encodeECALL(),
    ]

    let trace2 = generateTrace(program: program2)
    expect(trace2.terminated, "Branch not-taken test terminated")
    expectEqual(trace2.finalRegisters[3], 77, "Branch not taken, x3=77")
    expectEqual(trace2.branchesNotTaken, 1, "1 branch not taken")
}

func testALUOperations() {
    // Test all R-type ALU ops
    let program: [UInt32] = [
        encodeADDI(rd: 1, rs1: 0, imm: 100),     // x1 = 100
        encodeADDI(rd: 2, rs1: 0, imm: 25),      // x2 = 25
        encodeADD(rd: 3, rs1: 1, rs2: 2),         // x3 = 125
        encodeSUB(rd: 4, rs1: 1, rs2: 2),         // x4 = 75
        encodeAND(rd: 5, rs1: 1, rs2: 2),         // x5 = 100 & 25 = 0
        encodeOR(rd: 6, rs1: 1, rs2: 2),          // x6 = 100 | 25 = 125
        encodeXOR(rd: 7, rs1: 1, rs2: 2),         // x7 = 100 ^ 25 = 125
        encodeSLT(rd: 8, rs1: 2, rs2: 1),         // x8 = (25 < 100) = 1
        encodeSLTU(rd: 9, rs1: 1, rs2: 2),        // x9 = (100 < 25) = 0
        encodeSLL(rd: 10, rs1: 1, rs2: 2),        // x10 = 100 << 25
        encodeSRL(rd: 11, rs1: 1, rs2: 2),        // x11 = 100 >> 25
        encodeECALL(),
    ]

    let trace = generateTrace(program: program)
    expect(trace.terminated, "ALU test terminated")
    expectEqual(trace.finalRegisters[3], 125, "ADD: 100+25=125")
    expectEqual(trace.finalRegisters[4], 75, "SUB: 100-25=75")
    expectEqual(trace.finalRegisters[5], 100 & 25, "AND: 100&25")
    expectEqual(trace.finalRegisters[6], 100 | 25, "OR: 100|25")
    expectEqual(trace.finalRegisters[7], 100 ^ 25, "XOR: 100^25")
    expectEqual(trace.finalRegisters[8], 1, "SLT: 25<100")
    expectEqual(trace.finalRegisters[9], 0, "SLTU: 100<25 = false")
    expectEqual(trace.finalRegisters[10], 100 << 25, "SLL: 100<<25")
    expectEqual(trace.finalRegisters[11], 100 >> 25, "SRL: 100>>25")
}

func testTraceMetadata() {
    let program = rv32iFibonacciProgram()
    var initRegs = [UInt32](repeating: 0, count: 32)
    initRegs[10] = 5  // fib(5)

    let trace = generateTrace(program: program, initialRegisters: initRegs)

    expect(trace.stepCount > 0, "Non-empty trace")
    expect(trace.rows.count == trace.stepCount, "Row count matches step count")
    expect(!trace.registersWritten.isEmpty, "Some registers written")
    expect(!trace.registersRead.isEmpty, "Some registers read")
    expect(trace.branchesTaken > 0, "Some branches taken (loop)")

    // Verify step indices are sequential
    for (i, row) in trace.rows.enumerated() {
        expectEqual(row.step, i, "Step index \(i) is sequential")
    }

    // Verify trace can be converted to RV32ISteps
    let steps = trace.toRV32ISteps()
    expectEqual(steps.count, trace.stepCount, "RV32IStep count matches")
}

func testX0AlwaysZero() {
    // Writing to x0 should be silently ignored
    let program: [UInt32] = [
        encodeADDI(rd: 0, rs1: 0, imm: 42),      // x0 = 42 (should be ignored)
        encodeADD(rd: 1, rs1: 0, rs2: 0),         // x1 = x0 + x0 = 0
        encodeECALL(),
    ]

    let trace = generateTrace(program: program)
    expect(trace.terminated, "x0 test terminated")
    expectEqual(trace.finalRegisters[0], 0, "x0 still 0 after write")
    expectEqual(trace.finalRegisters[1], 0, "x1 = x0 + x0 = 0")
}
