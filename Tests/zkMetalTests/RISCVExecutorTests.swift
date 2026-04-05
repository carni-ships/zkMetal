// Tests for RISCVExecutor (RV32IM execution engine)

import Foundation
import zkMetal

public func runRISCVExecutorTests() {
    suite("RISCVExecutor - Arithmetic")
    testExecutorALURType()
    testExecutorALUIType()
    testExecutorShifts()

    suite("RISCVExecutor - Memory")
    testExecutorLoadStore()
    testExecutorByteHalfOps()

    suite("RISCVExecutor - Branches")
    testExecutorBranches()
    testExecutorBranchAllTypes()

    suite("RISCVExecutor - Jumps")
    testExecutorJAL()
    testExecutorJALR()

    suite("RISCVExecutor - Upper Immediate")
    testExecutorLUI()
    testExecutorAUIPC()

    suite("RISCVExecutor - Programs")
    testExecutorFibonacci()
    testExecutorFactorial()
    testExecutorMemoryProgram()

    suite("RISCVExecutor - RV32M Multiply")
    testExecutorMUL()
    testExecutorMULH()
    testExecutorMULHSU()
    testExecutorMULHU()

    suite("RISCVExecutor - RV32M Divide")
    testExecutorDIV()
    testExecutorDIVU()
    testExecutorREM()
    testExecutorREMU()
    testExecutorDivByZero()
    testExecutorSignedOverflow()

    suite("RISCVExecutor - Edge Cases")
    testExecutorX0Hardwired()
    testExecutorStepLimit()
    testExecutorTraceCapture()

    suite("RISCVExecutor - RV32M Programs")
    testExecutorMulFactorial()
}

// MARK: - Helpers

private let exec = RISCVExecutor(stepLimit: 10_000)

private func run(_ program: [UInt32], regs: [UInt32] = []) -> ExecutorResult {
    exec.execute(program: program, initialRegisters: regs)
}

// MARK: - R-type ALU Tests

func testExecutorALURType() {
    let program: [UInt32] = [
        encodeADDI(rd: 1, rs1: 0, imm: 100),    // x1 = 100
        encodeADDI(rd: 2, rs1: 0, imm: 25),     // x2 = 25
        encodeADD(rd: 3, rs1: 1, rs2: 2),        // x3 = 125
        encodeSUB(rd: 4, rs1: 1, rs2: 2),        // x4 = 75
        encodeAND(rd: 5, rs1: 1, rs2: 2),        // x5 = 100 & 25
        encodeOR(rd: 6, rs1: 1, rs2: 2),         // x6 = 100 | 25
        encodeXOR(rd: 7, rs1: 1, rs2: 2),        // x7 = 100 ^ 25
        encodeSLT(rd: 8, rs1: 2, rs2: 1),        // x8 = (25 < 100) = 1
        encodeSLTU(rd: 9, rs1: 1, rs2: 2),       // x9 = (100 < 25) = 0
        encodeECALL(),
    ]
    let r = run(program)
    expect(r.halted, "ALU R-type halted")
    expectEqual(r.registers[3], 125, "ADD: 100+25=125")
    expectEqual(r.registers[4], 75, "SUB: 100-25=75")
    expectEqual(r.registers[5], 100 & 25, "AND: 100&25")
    expectEqual(r.registers[6], 100 | 25, "OR: 100|25")
    expectEqual(r.registers[7], 100 ^ 25, "XOR: 100^25")
    expectEqual(r.registers[8], 1, "SLT: 25<100 = 1")
    expectEqual(r.registers[9], 0, "SLTU: 100<25 = 0")
}

func testExecutorALUIType() {
    let program: [UInt32] = [
        encodeADDI(rd: 1, rs1: 0, imm: 50),     // x1 = 50
        encodeADDI(rd: 2, rs1: 1, imm: -10),    // x2 = 40
        encodeIType(imm: 0x0F, rs1: 1, funct3: 0x7, rd: 3, opcode: 0x13), // ANDI x3, x1, 0x0F
        encodeIType(imm: 0xFF, rs1: 1, funct3: 0x6, rd: 4, opcode: 0x13), // ORI x4, x1, 0xFF
        encodeIType(imm: 0xFF, rs1: 1, funct3: 0x4, rd: 5, opcode: 0x13), // XORI x5, x1, 0xFF
        encodeIType(imm: 100, rs1: 1, funct3: 0x2, rd: 6, opcode: 0x13),  // SLTI x6, x1, 100 -> 1
        encodeIType(imm: 10, rs1: 1, funct3: 0x3, rd: 7, opcode: 0x13),   // SLTIU x7, x1, 10 -> 0
        encodeECALL(),
    ]
    let r = run(program)
    expect(r.halted, "ALU I-type halted")
    expectEqual(r.registers[1], 50, "ADDI: x1=50")
    expectEqual(r.registers[2], 40, "ADDI: 50+(-10)=40")
    expectEqual(r.registers[3], 50 & 0x0F, "ANDI: 50&0xF")
    expectEqual(r.registers[4], 50 | 0xFF, "ORI: 50|0xFF")
    expectEqual(r.registers[5], 50 ^ 0xFF, "XORI: 50^0xFF")
    expectEqual(r.registers[6], 1, "SLTI: 50<100 = 1")
    expectEqual(r.registers[7], 0, "SLTIU: 50<10 = 0")
}

func testExecutorShifts() {
    let program: [UInt32] = [
        encodeADDI(rd: 1, rs1: 0, imm: 0x100),  // x1 = 256
        encodeADDI(rd: 2, rs1: 0, imm: 4),      // x2 = 4
        encodeSLL(rd: 3, rs1: 1, rs2: 2),        // x3 = 256 << 4 = 4096
        encodeSRL(rd: 4, rs1: 1, rs2: 2),        // x4 = 256 >> 4 = 16
        encodeSRA(rd: 5, rs1: 1, rs2: 2),        // x5 = 256 >>a 4 = 16
        encodeSLLI(rd: 6, rs1: 1, shamt: 2),     // x6 = 256 << 2 = 1024
        encodeSRLI(rd: 7, rs1: 1, shamt: 2),     // x7 = 256 >> 2 = 64
        // Test SRA with negative: load -16 into x8
        encodeADDI(rd: 8, rs1: 0, imm: -16),     // x8 = -16 (0xFFFFFFF0)
        encodeSRAI(rd: 9, rs1: 8, shamt: 2),     // x9 = -16 >>a 2 = -4
        encodeECALL(),
    ]
    let r = run(program)
    expect(r.halted, "Shift test halted")
    expectEqual(r.registers[3], 4096, "SLL: 256<<4=4096")
    expectEqual(r.registers[4], 16, "SRL: 256>>4=16")
    expectEqual(r.registers[5], 16, "SRA: 256>>a4=16")
    expectEqual(r.registers[6], 1024, "SLLI: 256<<2=1024")
    expectEqual(r.registers[7], 64, "SRLI: 256>>2=64")
    expectEqual(r.registers[9], UInt32(bitPattern: -4), "SRAI: -16>>a2=-4")
}

// MARK: - Memory Tests

func testExecutorLoadStore() {
    let program: [UInt32] = [
        encodeADDI(rd: 1, rs1: 0, imm: 200),    // x1 = 200 (base addr)
        encodeADDI(rd: 2, rs1: 0, imm: 42),     // x2 = 42
        encodeSW(rs2: 2, rs1: 1, offset: 0),    // mem[200] = 42
        encodeLW(rd: 3, rs1: 1, offset: 0),     // x3 = mem[200] = 42
        encodeECALL(),
    ]
    let r = run(program)
    expect(r.halted, "LoadStore halted")
    expectEqual(r.registers[3], 42, "LW after SW: 42")
}

func testExecutorByteHalfOps() {
    let program: [UInt32] = [
        encodeADDI(rd: 1, rs1: 0, imm: 300),    // base addr
        encodeADDI(rd: 2, rs1: 0, imm: -5),     // x2 = 0xFFFFFFFB

        // SB: store low byte (0xFB)
        encodeSType(imm: 0, rs2: 2, rs1: 1, funct3: 0x0), // SB x2, 0(x1)

        // LB: sign-extend byte (0xFB -> 0xFFFFFFFB = -5)
        encodeIType(imm: 0, rs1: 1, funct3: 0x0, rd: 3, opcode: 0x03), // LB x3, 0(x1)

        // LBU: zero-extend byte (0xFB -> 0x000000FB = 251)
        encodeIType(imm: 0, rs1: 1, funct3: 0x4, rd: 4, opcode: 0x03), // LBU x4, 0(x1)

        // SH: store low halfword (0xFFFB)
        encodeSType(imm: 4, rs2: 2, rs1: 1, funct3: 0x1), // SH x2, 4(x1)

        // LH: sign-extend half (0xFFFB -> 0xFFFFFFFB = -5)
        encodeIType(imm: 4, rs1: 1, funct3: 0x1, rd: 5, opcode: 0x03), // LH x5, 4(x1)

        // LHU: zero-extend half (0xFFFB -> 0x0000FFFB = 65531)
        encodeIType(imm: 4, rs1: 1, funct3: 0x5, rd: 6, opcode: 0x03), // LHU x6, 4(x1)

        encodeECALL(),
    ]
    let r = run(program)
    expect(r.halted, "ByteHalf halted")
    expectEqual(r.registers[3], UInt32(bitPattern: -5), "LB sign-extend: -5")
    expectEqual(r.registers[4], 251, "LBU zero-extend: 251")
    expectEqual(r.registers[5], UInt32(bitPattern: -5), "LH sign-extend: -5")
    expectEqual(r.registers[6], 65531, "LHU zero-extend: 65531")
}

// MARK: - Branch Tests

func testExecutorBranches() {
    // BEQ taken
    let program: [UInt32] = [
        encodeADDI(rd: 1, rs1: 0, imm: 5),      // x1 = 5
        encodeADDI(rd: 2, rs1: 0, imm: 5),      // x2 = 5
        encodeBEQ(rs1: 1, rs2: 2, offset: 8),   // BEQ x1,x2,+8 (skip next)
        encodeADDI(rd: 3, rs1: 0, imm: 99),     // x3 = 99 (skipped)
        encodeADDI(rd: 3, rs1: 0, imm: 1),      // x3 = 1 (target)
        encodeECALL(),
    ]
    let r = run(program)
    expect(r.halted, "BEQ taken halted")
    expectEqual(r.registers[3], 1, "BEQ taken: x3=1")

    // BEQ not taken
    let prog2: [UInt32] = [
        encodeADDI(rd: 1, rs1: 0, imm: 5),
        encodeADDI(rd: 2, rs1: 0, imm: 3),
        encodeBEQ(rs1: 1, rs2: 2, offset: 8),
        encodeADDI(rd: 3, rs1: 0, imm: 77),
        encodeECALL(),
    ]
    let r2 = run(prog2)
    expectEqual(r2.registers[3], 77, "BEQ not taken: x3=77")
}

func testExecutorBranchAllTypes() {
    // Test BNE, BLT, BGE, BLTU, BGEU
    // BNE taken
    let progBNE: [UInt32] = [
        encodeADDI(rd: 1, rs1: 0, imm: 3),
        encodeADDI(rd: 2, rs1: 0, imm: 5),
        encodeBNE(rs1: 1, rs2: 2, offset: 8),   // 3 != 5, taken
        encodeADDI(rd: 3, rs1: 0, imm: 0),      // skipped
        encodeADDI(rd: 3, rs1: 0, imm: 1),
        encodeECALL(),
    ]
    expectEqual(run(progBNE).registers[3], 1, "BNE taken")

    // BLT taken (signed: -1 < 1)
    let progBLT: [UInt32] = [
        encodeADDI(rd: 1, rs1: 0, imm: -1),
        encodeADDI(rd: 2, rs1: 0, imm: 1),
        encodeBLT(rs1: 1, rs2: 2, offset: 8),
        encodeADDI(rd: 3, rs1: 0, imm: 0),
        encodeADDI(rd: 3, rs1: 0, imm: 1),
        encodeECALL(),
    ]
    expectEqual(run(progBLT).registers[3], 1, "BLT taken: -1<1")

    // BGE taken (5 >= 5)
    let progBGE: [UInt32] = [
        encodeADDI(rd: 1, rs1: 0, imm: 5),
        encodeADDI(rd: 2, rs1: 0, imm: 5),
        encodeBGE(rs1: 1, rs2: 2, offset: 8),
        encodeADDI(rd: 3, rs1: 0, imm: 0),
        encodeADDI(rd: 3, rs1: 0, imm: 1),
        encodeECALL(),
    ]
    expectEqual(run(progBGE).registers[3], 1, "BGE taken: 5>=5")

    // BLTU not taken (0xFFFFFFFF > 1 unsigned)
    let progBLTU: [UInt32] = [
        encodeADDI(rd: 1, rs1: 0, imm: -1),     // x1 = 0xFFFFFFFF
        encodeADDI(rd: 2, rs1: 0, imm: 1),
        encodeBType(imm: 8, rs2: 2, rs1: 1, funct3: 0x6), // BLTU x1,x2,+8
        encodeADDI(rd: 3, rs1: 0, imm: 77),     // not skipped
        encodeECALL(),
    ]
    expectEqual(run(progBLTU).registers[3], 77, "BLTU not taken: 0xFFFFFFFF !< 1")

    // BGEU taken (0xFFFFFFFF >= 1 unsigned)
    let progBGEU: [UInt32] = [
        encodeADDI(rd: 1, rs1: 0, imm: -1),     // x1 = 0xFFFFFFFF
        encodeADDI(rd: 2, rs1: 0, imm: 1),
        encodeBType(imm: 8, rs2: 2, rs1: 1, funct3: 0x7), // BGEU x1,x2,+8
        encodeADDI(rd: 3, rs1: 0, imm: 0),      // skipped
        encodeADDI(rd: 3, rs1: 0, imm: 1),
        encodeECALL(),
    ]
    expectEqual(run(progBGEU).registers[3], 1, "BGEU taken: 0xFFFFFFFF >= 1")
}

// MARK: - Jump Tests

func testExecutorJAL() {
    let program: [UInt32] = [
        encodeADDI(rd: 1, rs1: 0, imm: 10),     // x1 = 10
        encodeJAL(rd: 5, offset: 8),             // x5 = PC+4, jump +8
        encodeADDI(rd: 1, rs1: 0, imm: 99),     // skipped
        encodeADDI(rd: 2, rs1: 0, imm: 20),     // x2 = 20 (target)
        encodeECALL(),
    ]
    let r = run(program)
    expect(r.halted, "JAL halted")
    expectEqual(r.registers[1], 10, "JAL: x1 unchanged")
    expectEqual(r.registers[2], 20, "JAL: jumped to target")
    expectEqual(r.registers[5], 8, "JAL: return addr = PC+4 = 8")
}

func testExecutorJALR() {
    let program: [UInt32] = [
        encodeADDI(rd: 1, rs1: 0, imm: 16),     // x1 = 16 (target addr)
        encodeJALR(rd: 5, rs1: 1, offset: 0),   // x5 = PC+4, jump to x1+0=16
        encodeADDI(rd: 2, rs1: 0, imm: 99),     // skipped
        encodeADDI(rd: 2, rs1: 0, imm: 99),     // skipped
        encodeADDI(rd: 2, rs1: 0, imm: 42),     // x2 = 42 (target at addr 16)
        encodeECALL(),
    ]
    let r = run(program)
    expect(r.halted, "JALR halted")
    expectEqual(r.registers[2], 42, "JALR: jumped to addr 16")
    expectEqual(r.registers[5], 8, "JALR: return addr = 8")
}

// MARK: - Upper Immediate Tests

func testExecutorLUI() {
    let program: [UInt32] = [
        encodeLUI(rd: 1, imm: 0x12345),         // x1 = 0x12345000
        encodeECALL(),
    ]
    let r = run(program)
    expectEqual(r.registers[1], 0x12345000, "LUI: 0x12345000")
}

func testExecutorAUIPC() {
    let program: [UInt32] = [
        encodeAUIPC(rd: 1, imm: 1),             // x1 = PC + 0x1000 = 0 + 0x1000
        encodeECALL(),
    ]
    let r = run(program)
    expectEqual(r.registers[1], 0x1000, "AUIPC: PC+0x1000")
}

// MARK: - Full Program Tests

func testExecutorFibonacci() {
    // fib(10) = 55 using the existing program builder
    let program = rv32iFibonacciProgram()
    var initRegs = [UInt32](repeating: 0, count: 32)
    initRegs[10] = 10

    let r = exec.execute(program: program, initialRegisters: initRegs)
    expect(r.halted, "Fibonacci halted")
    expectEqual(r.registers[10], 55, "fib(10)=55")

    // fib(1) = 1
    initRegs[10] = 1
    let r1 = exec.execute(program: program, initialRegisters: initRegs)
    expect(r1.halted, "fib(1) halted")
    expectEqual(r1.registers[10], 1, "fib(1)=1")

    // fib(5) = 5
    initRegs[10] = 5
    let r5 = exec.execute(program: program, initialRegisters: initRegs)
    expect(r5.halted, "fib(5) halted")
    expectEqual(r5.registers[10], 5, "fib(5)=5")
}

func testExecutorFactorial() {
    // sum(1..10) = 55
    let program = rv32iFactorialProgram()
    var initRegs = [UInt32](repeating: 0, count: 32)
    initRegs[10] = 10

    let r = exec.execute(program: program, initialRegisters: initRegs)
    expect(r.halted, "Sum halted")
    expectEqual(r.registers[10], 55, "sum(1..10)=55")

    // sum(1..0) = 0
    initRegs[10] = 0
    let r0 = exec.execute(program: program, initialRegisters: initRegs)
    expect(r0.halted, "sum(0) halted")
    expectEqual(r0.registers[10], 0, "sum(1..0)=0")
}

func testExecutorMemoryProgram() {
    let program = rv32iMemoryTestProgram()
    let r = exec.execute(program: program)
    expect(r.halted, "Memory program halted")
    expectEqual(r.registers[10], 141, "42+99=141")
    expectEqual(r.registers[12], 42, "loaded 42")
    expectEqual(r.registers[13], 99, "loaded 99")

    // Verify trace has memory operations
    var loads = 0, stores = 0
    for step in r.trace {
        if step.memoryAddr != 0 {
            // Check if it's a load or store by instruction
            switch step.instruction {
            case .base(let i):
                switch i {
                case .LB, .LH, .LW, .LBU, .LHU: loads += 1
                case .SB, .SH, .SW: stores += 1
                default: break
                }
            default: break
            }
        }
    }
    expectEqual(stores, 2, "2 stores in memory program")
    expectEqual(loads, 2, "2 loads in memory program")
}

// MARK: - RV32M Multiply Tests

func testExecutorMUL() {
    let program: [UInt32] = [
        encodeADDI(rd: 1, rs1: 0, imm: 7),
        encodeADDI(rd: 2, rs1: 0, imm: 6),
        encodeMUL(rd: 3, rs1: 1, rs2: 2),       // 7 * 6 = 42
        // Negative: -3 * 5 = -15
        encodeADDI(rd: 4, rs1: 0, imm: -3),
        encodeADDI(rd: 5, rs1: 0, imm: 5),
        encodeMUL(rd: 6, rs1: 4, rs2: 5),       // -3 * 5 = -15
        encodeECALL(),
    ]
    let r = run(program)
    expect(r.halted, "MUL halted")
    expectEqual(r.registers[3], 42, "MUL: 7*6=42")
    expectEqual(r.registers[6], UInt32(bitPattern: -15), "MUL: -3*5=-15")
}

func testExecutorMULH() {
    // MULH: upper 32 bits of signed*signed
    // 0x7FFFFFFF * 0x7FFFFFFF = 0x3FFFFFFF_00000001
    let program: [UInt32] = [
        encodeLUI(rd: 1, imm: 0x7FFFF),          // x1 = 0x7FFFF000
        encodeADDI(rd: 1, rs1: 1, imm: 0x7FF),   // x1 = 0x7FFFF7FF (close to max)
        encodeADDI(rd: 2, rs1: 0, imm: 2),
        encodeMULH(rd: 3, rs1: 1, rs2: 2),       // upper bits of x1*2
        encodeECALL(),
    ]
    let r = run(program)
    expect(r.halted, "MULH halted")
    // 0x7FFFF7FF * 2 = 0xFFFFEFFE -> upper 32 = 0
    expectEqual(r.registers[3], 0, "MULH: small product upper=0")

    // Large product: -1 * -1 = 1, upper bits = 0
    let prog2: [UInt32] = [
        encodeADDI(rd: 1, rs1: 0, imm: -1),
        encodeADDI(rd: 2, rs1: 0, imm: -1),
        encodeMULH(rd: 3, rs1: 1, rs2: 2),
        encodeECALL(),
    ]
    let r2 = run(prog2)
    expectEqual(r2.registers[3], 0, "MULH: (-1)*(-1) upper=0")
}

func testExecutorMULHSU() {
    // MULHSU: upper 32 bits of signed * unsigned
    // -1 (signed) * 1 (unsigned) = -1 as 64-bit = 0xFFFFFFFF_FFFFFFFF, upper = 0xFFFFFFFF
    let program: [UInt32] = [
        encodeADDI(rd: 1, rs1: 0, imm: -1),
        encodeADDI(rd: 2, rs1: 0, imm: 1),
        encodeMULHSU(rd: 3, rs1: 1, rs2: 2),
        encodeECALL(),
    ]
    let r = run(program)
    expect(r.halted, "MULHSU halted")
    expectEqual(r.registers[3], 0xFFFFFFFF, "MULHSU: -1*1 upper=0xFFFFFFFF")
}

func testExecutorMULHU() {
    // MULHU: upper 32 bits of unsigned * unsigned
    // 0xFFFFFFFF * 0xFFFFFFFF = 0xFFFFFFFE_00000001, upper = 0xFFFFFFFE
    let program: [UInt32] = [
        encodeADDI(rd: 1, rs1: 0, imm: -1),     // 0xFFFFFFFF
        encodeADDI(rd: 2, rs1: 0, imm: -1),     // 0xFFFFFFFF
        encodeMULHU(rd: 3, rs1: 1, rs2: 2),
        encodeECALL(),
    ]
    let r = run(program)
    expect(r.halted, "MULHU halted")
    expectEqual(r.registers[3], 0xFFFFFFFE, "MULHU: 0xFFFFFFFF*0xFFFFFFFF upper")
}

// MARK: - RV32M Division Tests

func testExecutorDIV() {
    let program: [UInt32] = [
        encodeADDI(rd: 1, rs1: 0, imm: 20),
        encodeADDI(rd: 2, rs1: 0, imm: 6),
        encodeDIV(rd: 3, rs1: 1, rs2: 2),       // 20 / 6 = 3
        // Negative: -20 / 6 = -3
        encodeADDI(rd: 4, rs1: 0, imm: -20),
        encodeDIV(rd: 5, rs1: 4, rs2: 2),       // -20 / 6 = -3
        encodeECALL(),
    ]
    let r = run(program)
    expect(r.halted, "DIV halted")
    expectEqual(r.registers[3], 3, "DIV: 20/6=3")
    expectEqual(r.registers[5], UInt32(bitPattern: -3), "DIV: -20/6=-3")
}

func testExecutorDIVU() {
    let program: [UInt32] = [
        encodeADDI(rd: 1, rs1: 0, imm: 100),
        encodeADDI(rd: 2, rs1: 0, imm: 7),
        encodeDIVU(rd: 3, rs1: 1, rs2: 2),      // 100 / 7 = 14
        encodeECALL(),
    ]
    let r = run(program)
    expectEqual(r.registers[3], 14, "DIVU: 100/7=14")
}

func testExecutorREM() {
    let program: [UInt32] = [
        encodeADDI(rd: 1, rs1: 0, imm: 20),
        encodeADDI(rd: 2, rs1: 0, imm: 6),
        encodeREM(rd: 3, rs1: 1, rs2: 2),       // 20 % 6 = 2
        // Negative: -20 % 6 = -2
        encodeADDI(rd: 4, rs1: 0, imm: -20),
        encodeREM(rd: 5, rs1: 4, rs2: 2),
        encodeECALL(),
    ]
    let r = run(program)
    expectEqual(r.registers[3], 2, "REM: 20%6=2")
    expectEqual(r.registers[5], UInt32(bitPattern: -2), "REM: -20%6=-2")
}

func testExecutorREMU() {
    let program: [UInt32] = [
        encodeADDI(rd: 1, rs1: 0, imm: 100),
        encodeADDI(rd: 2, rs1: 0, imm: 7),
        encodeREMU(rd: 3, rs1: 1, rs2: 2),      // 100 % 7 = 2
        encodeECALL(),
    ]
    let r = run(program)
    expectEqual(r.registers[3], 2, "REMU: 100%7=2")
}

func testExecutorDivByZero() {
    // RISC-V spec: DIV by zero -> all ones (-1), REM by zero -> dividend
    let program: [UInt32] = [
        encodeADDI(rd: 1, rs1: 0, imm: 42),
        encodeADDI(rd: 2, rs1: 0, imm: 0),      // divisor = 0
        encodeDIV(rd: 3, rs1: 1, rs2: 2),       // 42 / 0 = -1 (all ones)
        encodeDIVU(rd: 4, rs1: 1, rs2: 2),      // 42 / 0 = 0xFFFFFFFF
        encodeREM(rd: 5, rs1: 1, rs2: 2),       // 42 % 0 = 42 (dividend)
        encodeREMU(rd: 6, rs1: 1, rs2: 2),      // 42 % 0 = 42 (dividend)
        encodeECALL(),
    ]
    let r = run(program)
    expect(r.halted, "DivByZero halted")
    expectEqual(r.registers[3], UInt32(bitPattern: -1), "DIV by 0: -1")
    expectEqual(r.registers[4], UInt32.max, "DIVU by 0: 0xFFFFFFFF")
    expectEqual(r.registers[5], 42, "REM by 0: dividend")
    expectEqual(r.registers[6], 42, "REMU by 0: dividend")
}

func testExecutorSignedOverflow() {
    // RISC-V spec: INT32_MIN / -1 -> INT32_MIN, INT32_MIN % -1 -> 0
    // Build INT32_MIN = 0x80000000
    let program: [UInt32] = [
        encodeLUI(rd: 1, imm: 0x80000),         // x1 = 0x80000000 = INT32_MIN
        encodeADDI(rd: 2, rs1: 0, imm: -1),     // x2 = -1
        encodeDIV(rd: 3, rs1: 1, rs2: 2),       // INT32_MIN / -1 = INT32_MIN
        encodeREM(rd: 4, rs1: 1, rs2: 2),       // INT32_MIN % -1 = 0
        encodeECALL(),
    ]
    let r = run(program)
    expect(r.halted, "SignedOverflow halted")
    expectEqual(r.registers[3], UInt32(bitPattern: Int32.min), "DIV overflow: INT32_MIN")
    expectEqual(r.registers[4], 0, "REM overflow: 0")
}

// MARK: - Edge Case Tests

func testExecutorX0Hardwired() {
    // Writing to x0 must be silently ignored
    let program: [UInt32] = [
        encodeADDI(rd: 0, rs1: 0, imm: 42),     // x0 = 42 (ignored)
        encodeADD(rd: 1, rs1: 0, rs2: 0),        // x1 = x0 + x0 = 0
        // Also test M-extension writing to x0
        encodeADDI(rd: 2, rs1: 0, imm: 7),
        encodeADDI(rd: 3, rs1: 0, imm: 6),
        encodeMUL(rd: 0, rs1: 2, rs2: 3),        // x0 = 42 (ignored)
        encodeADD(rd: 4, rs1: 0, rs2: 0),        // x4 = x0 + x0 = 0
        encodeECALL(),
    ]
    let r = run(program)
    expect(r.halted, "x0 test halted")
    expectEqual(r.registers[0], 0, "x0 still 0 after ADDI")
    expectEqual(r.registers[1], 0, "x1 = x0+x0 = 0")
    expectEqual(r.registers[4], 0, "x4 = x0+x0 = 0 after MUL to x0")
}

func testExecutorStepLimit() {
    // Infinite loop: should hit step limit
    let smallExec = RISCVExecutor(stepLimit: 10)
    let program: [UInt32] = [
        encodeADDI(rd: 1, rs1: 1, imm: 1),      // x1++
        encodeJAL(rd: 0, offset: -4),            // jump back (infinite loop)
    ]
    let r = smallExec.execute(program: program)
    expect(!r.halted || r.haltReason == .stepLimit ||
           r.stepCount == 10, "Step limit enforced")
    expectEqual(r.trace.count, 10, "Trace has exactly 10 steps")
    // x1 should be 5 (5 increments interleaved with 5 jumps in 10 steps)
    expectEqual(r.registers[1], 5, "5 increments in 10 steps")
}

func testExecutorTraceCapture() {
    let program: [UInt32] = [
        encodeADDI(rd: 1, rs1: 0, imm: 10),     // x1 = 10
        encodeADDI(rd: 2, rs1: 0, imm: 20),     // x2 = 20
        encodeADD(rd: 3, rs1: 1, rs2: 2),        // x3 = 30
        encodeECALL(),
    ]
    let r = run(program)
    expect(r.halted, "Trace capture halted")
    expectEqual(r.trace.count, 4, "4 steps in trace (including ECALL)")

    // Verify step indices are sequential
    for (i, step) in r.trace.enumerated() {
        expectEqual(step.step, i, "Step \(i) sequential")
    }

    // Verify first step
    let s0 = r.trace[0]
    expectEqual(s0.pc, 0, "Step 0: PC=0")
    expectEqual(s0.rdVal, 10, "Step 0: rdVal=10")
    expectEqual(s0.rd, 1, "Step 0: rd=1")

    // Verify ADD step
    let s2 = r.trace[2]
    expectEqual(s2.rs1Val, 10, "ADD: rs1Val=10")
    expectEqual(s2.rs2Val, 20, "ADD: rs2Val=20")
    expectEqual(s2.rdVal, 30, "ADD: rdVal=30")

    // Verify PCs
    expectEqual(r.trace[0].pc, 0, "PC 0")
    expectEqual(r.trace[1].pc, 4, "PC 4")
    expectEqual(r.trace[2].pc, 8, "PC 8")
    expectEqual(r.trace[3].pc, 12, "PC 12 (ECALL)")
}

// MARK: - RV32M Program: Factorial with MUL

func testExecutorMulFactorial() {
    // Compute n! using MUL instruction
    // x10 = n (input), result in x10
    //   addi x11, x0, 1       # acc = 1
    //   beq  x10, x0, done    # if n==0 goto done
    // loop:
    //   mul  x11, x11, x10    # acc *= n
    //   addi x10, x10, -1     # n--
    //   bne  x10, x0, loop    # if n!=0 loop
    // done:
    //   add  x10, x11, x0     # x10 = result
    //   ecall
    let program: [UInt32] = [
        encodeADDI(rd: 11, rs1: 0, imm: 1),       // x11 = 1
        encodeBEQ(rs1: 10, rs2: 0, offset: 16),   // if n==0 goto done
        // loop (PC = 8):
        encodeMUL(rd: 11, rs1: 11, rs2: 10),      // x11 *= x10
        encodeADDI(rd: 10, rs1: 10, imm: -1),     // x10--
        encodeBNE(rs1: 10, rs2: 0, offset: -8),   // if x10!=0 goto loop
        // done (PC = 20):
        encodeADD(rd: 10, rs1: 11, rs2: 0),       // x10 = x11
        encodeECALL(),
    ]

    // 5! = 120
    var regs = [UInt32](repeating: 0, count: 32)
    regs[10] = 5
    let r = exec.execute(program: program, initialRegisters: regs)
    expect(r.halted, "Factorial halted")
    expectEqual(r.registers[10], 120, "5! = 120")

    // 10! = 3628800
    regs[10] = 10
    let r10 = exec.execute(program: program, initialRegisters: regs)
    expect(r10.halted, "10! halted")
    expectEqual(r10.registers[10], 3628800, "10! = 3628800")

    // 0! = 1
    regs[10] = 0
    let r0 = exec.execute(program: program, initialRegisters: regs)
    expect(r0.halted, "0! halted")
    expectEqual(r0.registers[10], 1, "0! = 1")

    // 1! = 1
    regs[10] = 1
    let r1 = exec.execute(program: program, initialRegisters: regs)
    expect(r1.halted, "1! halted")
    expectEqual(r1.registers[10], 1, "1! = 1")
}
