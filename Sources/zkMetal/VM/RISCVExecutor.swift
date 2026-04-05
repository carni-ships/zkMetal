// RISCVExecutor -- RV32IM execution engine with full trace capture
//
// Executes RISC-V RV32I base integer + RV32M multiply/divide extension instructions
// and records a per-step execution trace suitable for zkVM proof generation.
//
// Supports all 40 RV32I instructions from RISCVDecoder plus 8 RV32M instructions:
//   MUL, MULH, MULHSU, MULHU, DIV, DIVU, REM, REMU
//
// The execution trace captures register reads, ALU results, memory operations,
// and branch decisions at every step -- the exact witness a zkVM prover needs.
//
// Reference: RISC-V Unprivileged ISA Specification, Volume I, Version 20191213
//            Chapter 7: "M" Standard Extension for Integer Multiply and Divide

import Foundation

// MARK: - RV32M Instruction Enum

/// The 8 RV32M multiply/divide instructions (opcode 0x33, funct7 = 0x01).
public enum RV32MInstruction: UInt8, CaseIterable {
    case MUL    = 0   // funct3 = 0: lower 32 bits of rs1 * rs2
    case MULH   = 1   // funct3 = 1: upper 32 bits of signed * signed
    case MULHSU = 2   // funct3 = 2: upper 32 bits of signed * unsigned
    case MULHU  = 3   // funct3 = 3: upper 32 bits of unsigned * unsigned
    case DIV    = 4   // funct3 = 4: signed division
    case DIVU   = 5   // funct3 = 5: unsigned division
    case REM    = 6   // funct3 = 6: signed remainder
    case REMU   = 7   // funct3 = 7: unsigned remainder
}

// MARK: - Unified Instruction

/// A decoded instruction that can be either RV32I or RV32M.
public enum RV32IMInstruction {
    case base(RV32IInstruction)
    case mul(RV32MInstruction)
}

// MARK: - Executor Trace Step

/// A single step in the execution trace, capturing all state needed for zkVM proving.
public struct ExecutorTraceStep {
    /// Program counter at the start of this step
    public let pc: UInt32

    /// The instruction executed (base or M-extension)
    public let instruction: RV32IMInstruction

    /// Original 32-bit instruction word
    public let word: UInt32

    /// Value of rs1 register before execution
    public let rs1Val: UInt32

    /// Value of rs2 register before execution
    public let rs2Val: UInt32

    /// Value written to rd (or 0 if no write)
    public let rdVal: UInt32

    /// Memory address accessed (0 if no memory op)
    public let memoryAddr: UInt32

    /// Memory value read or written (0 if no memory op)
    public let memoryVal: UInt32

    /// Step index (0-based)
    public let step: Int

    /// Register indices
    public let rd: UInt8
    public let rs1: UInt8
    public let rs2: UInt8

    /// Immediate value
    public let immediate: Int32
}

// MARK: - Executor Result

/// Result of executing a program: the trace plus final machine state.
public struct ExecutorResult {
    /// Ordered trace steps, one per executed instruction
    public let trace: [ExecutorTraceStep]

    /// Final register file (32 entries, x0 == 0)
    public let registers: [UInt32]

    /// Final program counter
    public let finalPC: UInt32

    /// Whether execution terminated normally (ECALL/EBREAK) vs hitting step limit
    public let halted: Bool

    /// Halt reason
    public let haltReason: HaltReason

    /// Number of steps executed
    public var stepCount: Int { trace.count }

    public enum HaltReason {
        case ecall
        case ebreak
        case stepLimit
        case outOfBounds
        case decodeError
    }
}

// MARK: - RISCVExecutor

/// RV32IM executor with configurable step limit and full trace recording.
///
/// Usage:
///   let executor = RISCVExecutor(stepLimit: 100_000)
///   let result = executor.execute(program: myInstructions)
///   // result.trace has per-step witness data
///   // result.registers has final register state
public struct RISCVExecutor {
    /// Maximum number of instructions to execute before forced halt
    public let stepLimit: Int

    /// Starting PC (default 0)
    public let startPC: UInt32

    public init(stepLimit: Int = 1_000_000, startPC: UInt32 = 0) {
        self.stepLimit = stepLimit
        self.startPC = startPC
    }

    /// Execute a program given as an array of 32-bit instruction words.
    ///
    /// Instructions are loaded at address 0 (or startPC). Execution proceeds
    /// until ECALL, EBREAK, out-of-bounds PC, decode error, or step limit.
    ///
    /// - Parameters:
    ///   - program: Array of 32-bit instruction words
    ///   - initialRegisters: Optional initial register values (32 entries; x0 forced to 0)
    ///   - initialMemory: Optional pre-loaded memory contents
    /// - Returns: ExecutorResult with trace and final state
    public func execute(
        program: [UInt32],
        initialRegisters: [UInt32] = [],
        initialMemory: [UInt32: UInt8] = [:]
    ) -> ExecutorResult {
        // Load program into memory starting at startPC
        var mem = initialMemory
        for (i, word) in program.enumerated() {
            let addr = startPC &+ UInt32(i * 4)
            mem[addr]     = UInt8(word & 0xFF)
            mem[addr + 1] = UInt8((word >> 8) & 0xFF)
            mem[addr + 2] = UInt8((word >> 16) & 0xFF)
            mem[addr + 3] = UInt8((word >> 24) & 0xFF)
        }

        // Initialize registers
        var regs = [UInt32](repeating: 0, count: 32)
        for (i, v) in initialRegisters.prefix(32).enumerated() {
            regs[i] = v
        }
        regs[0] = 0  // x0 hardwired

        var pc = startPC
        var trace = [ExecutorTraceStep]()
        trace.reserveCapacity(min(stepLimit, program.count * 2))

        let programEnd = startPC &+ UInt32(program.count * 4)
        var haltReason = ExecutorResult.HaltReason.stepLimit

        for stepIdx in 0..<stepLimit {
            // Bounds check
            if pc < startPC || pc >= programEnd || (pc & 3) != 0 {
                haltReason = .outOfBounds
                break
            }

            // Fetch
            let word = readWord(mem, pc)

            // Decode: try RV32I first, then check for M-extension
            let opcode = UInt8(word & 0x7F)
            let funct3 = UInt8((word >> 12) & 0x7)
            let funct7 = UInt8((word >> 25) & 0x7F)
            let rdIdx  = UInt8((word >> 7) & 0x1F)
            let rs1Idx = UInt8((word >> 15) & 0x1F)
            let rs2Idx = UInt8((word >> 20) & 0x1F)

            let rs1Val = readReg(regs, rs1Idx)
            let rs2Val = readReg(regs, rs2Idx)

            // Check for RV32M (opcode 0x33, funct7 0x01)
            if opcode == 0x33 && funct7 == 0x01 {
                guard let mInstr = decodeMExtension(funct3: funct3) else {
                    haltReason = .decodeError
                    break
                }

                let rdVal = executeMInstruction(mInstr, rs1Val: rs1Val, rs2Val: rs2Val)

                let step = ExecutorTraceStep(
                    pc: pc, instruction: .mul(mInstr), word: word,
                    rs1Val: rs1Val, rs2Val: rs2Val, rdVal: rdVal,
                    memoryAddr: 0, memoryVal: 0, step: stepIdx,
                    rd: rdIdx, rs1: rs1Idx, rs2: rs2Idx, immediate: 0)
                trace.append(step)

                writeReg(&regs, rdIdx, rdVal)
                pc = pc &+ 4
                continue
            }

            // Decode RV32I
            let decoded: DecodedInstruction
            do {
                decoded = try decodeInstruction(word: word)
            } catch {
                haltReason = .decodeError
                break
            }

            var rdVal: UInt32 = 0
            var memAddr: UInt32 = 0
            var memVal: UInt32 = 0
            var nextPC = pc &+ 4

            switch decoded.instruction {
            // ---- R-type ALU ----
            case .ADD:  rdVal = rs1Val &+ rs2Val
            case .SUB:  rdVal = rs1Val &- rs2Val
            case .SLL:  rdVal = rs1Val << (rs2Val & 31)
            case .SLT:  rdVal = Int32(bitPattern: rs1Val) < Int32(bitPattern: rs2Val) ? 1 : 0
            case .SLTU: rdVal = rs1Val < rs2Val ? 1 : 0
            case .XOR:  rdVal = rs1Val ^ rs2Val
            case .SRL:  rdVal = rs1Val >> (rs2Val & 31)
            case .SRA:  rdVal = UInt32(bitPattern: Int32(bitPattern: rs1Val) >> Int32(rs2Val & 31))
            case .OR:   rdVal = rs1Val | rs2Val
            case .AND:  rdVal = rs1Val & rs2Val

            // ---- I-type ALU ----
            case .ADDI:  rdVal = rs1Val &+ UInt32(bitPattern: decoded.immediate)
            case .SLTI:  rdVal = Int32(bitPattern: rs1Val) < decoded.immediate ? 1 : 0
            case .SLTIU: rdVal = rs1Val < UInt32(bitPattern: decoded.immediate) ? 1 : 0
            case .XORI:  rdVal = rs1Val ^ UInt32(bitPattern: decoded.immediate)
            case .ORI:   rdVal = rs1Val | UInt32(bitPattern: decoded.immediate)
            case .ANDI:  rdVal = rs1Val & UInt32(bitPattern: decoded.immediate)
            case .SLLI:  rdVal = rs1Val << (UInt32(decoded.immediate) & 31)
            case .SRLI:  rdVal = rs1Val >> (UInt32(decoded.immediate) & 31)
            case .SRAI:  rdVal = UInt32(bitPattern: Int32(bitPattern: rs1Val) >> (Int32(decoded.immediate) & 31))

            // ---- Loads ----
            case .LB:
                let addr = rs1Val &+ UInt32(bitPattern: decoded.immediate)
                let b = readByte(mem, addr)
                rdVal = (b & 0x80) != 0 ? UInt32(b) | 0xFFFFFF00 : UInt32(b)
                memAddr = addr; memVal = UInt32(b)
            case .LH:
                let addr = rs1Val &+ UInt32(bitPattern: decoded.immediate)
                let h = readHalf(mem, addr)
                rdVal = (h & 0x8000) != 0 ? UInt32(h) | 0xFFFF0000 : UInt32(h)
                memAddr = addr; memVal = UInt32(h)
            case .LW:
                let addr = rs1Val &+ UInt32(bitPattern: decoded.immediate)
                rdVal = readWord(mem, addr)
                memAddr = addr; memVal = rdVal
            case .LBU:
                let addr = rs1Val &+ UInt32(bitPattern: decoded.immediate)
                rdVal = UInt32(readByte(mem, addr))
                memAddr = addr; memVal = rdVal
            case .LHU:
                let addr = rs1Val &+ UInt32(bitPattern: decoded.immediate)
                rdVal = UInt32(readHalf(mem, addr))
                memAddr = addr; memVal = rdVal

            // ---- Stores ----
            case .SB:
                let addr = rs1Val &+ UInt32(bitPattern: decoded.immediate)
                writeByte(&mem, addr, UInt8(rs2Val & 0xFF))
                memAddr = addr; memVal = rs2Val & 0xFF
            case .SH:
                let addr = rs1Val &+ UInt32(bitPattern: decoded.immediate)
                writeHalf(&mem, addr, UInt16(rs2Val & 0xFFFF))
                memAddr = addr; memVal = rs2Val & 0xFFFF
            case .SW:
                let addr = rs1Val &+ UInt32(bitPattern: decoded.immediate)
                writeWord(&mem, addr, rs2Val)
                memAddr = addr; memVal = rs2Val

            // ---- Branches ----
            case .BEQ:
                if rs1Val == rs2Val { nextPC = pc &+ UInt32(bitPattern: decoded.immediate) }
            case .BNE:
                if rs1Val != rs2Val { nextPC = pc &+ UInt32(bitPattern: decoded.immediate) }
            case .BLT:
                if Int32(bitPattern: rs1Val) < Int32(bitPattern: rs2Val) {
                    nextPC = pc &+ UInt32(bitPattern: decoded.immediate)
                }
            case .BGE:
                if Int32(bitPattern: rs1Val) >= Int32(bitPattern: rs2Val) {
                    nextPC = pc &+ UInt32(bitPattern: decoded.immediate)
                }
            case .BLTU:
                if rs1Val < rs2Val { nextPC = pc &+ UInt32(bitPattern: decoded.immediate) }
            case .BGEU:
                if rs1Val >= rs2Val { nextPC = pc &+ UInt32(bitPattern: decoded.immediate) }

            // ---- Jumps ----
            case .JAL:
                rdVal = pc &+ 4
                nextPC = pc &+ UInt32(bitPattern: decoded.immediate)
            case .JALR:
                rdVal = pc &+ 4
                nextPC = (rs1Val &+ UInt32(bitPattern: decoded.immediate)) & ~1

            // ---- Upper Immediate ----
            case .LUI:
                rdVal = UInt32(bitPattern: decoded.immediate)
            case .AUIPC:
                rdVal = pc &+ UInt32(bitPattern: decoded.immediate)

            // ---- System ----
            case .ECALL:
                let step = ExecutorTraceStep(
                    pc: pc, instruction: .base(.ECALL), word: word,
                    rs1Val: rs1Val, rs2Val: rs2Val, rdVal: 0,
                    memoryAddr: 0, memoryVal: 0, step: stepIdx,
                    rd: 0, rs1: rs1Idx, rs2: rs2Idx, immediate: 0)
                trace.append(step)
                haltReason = .ecall
                return ExecutorResult(trace: trace, registers: regs,
                                      finalPC: pc, halted: true, haltReason: haltReason)

            case .EBREAK:
                let step = ExecutorTraceStep(
                    pc: pc, instruction: .base(.EBREAK), word: word,
                    rs1Val: rs1Val, rs2Val: rs2Val, rdVal: 0,
                    memoryAddr: 0, memoryVal: 0, step: stepIdx,
                    rd: 0, rs1: rs1Idx, rs2: rs2Idx, immediate: 0)
                trace.append(step)
                haltReason = .ebreak
                return ExecutorResult(trace: trace, registers: regs,
                                      finalPC: pc, halted: true, haltReason: haltReason)

            case .FENCE:
                rdVal = 0  // No architectural effect
            }

            // Write rd for non-branch/store/system instructions
            switch decoded.instruction {
            case .BEQ, .BNE, .BLT, .BGE, .BLTU, .BGEU,
                 .SB, .SH, .SW, .ECALL, .EBREAK, .FENCE:
                break
            default:
                writeReg(&regs, decoded.rd, rdVal)
            }

            let step = ExecutorTraceStep(
                pc: pc, instruction: .base(decoded.instruction), word: word,
                rs1Val: rs1Val, rs2Val: rs2Val, rdVal: rdVal,
                memoryAddr: memAddr, memoryVal: memVal, step: stepIdx,
                rd: decoded.rd, rs1: decoded.rs1, rs2: decoded.rs2,
                immediate: decoded.immediate)
            trace.append(step)

            pc = nextPC
        }

        return ExecutorResult(trace: trace, registers: regs,
                              finalPC: pc, halted: haltReason != .stepLimit,
                              haltReason: haltReason)
    }

    // MARK: - RV32M Decode + Execute

    /// Decode an M-extension instruction from funct3 (funct7 already verified as 0x01).
    private func decodeMExtension(funct3: UInt8) -> RV32MInstruction? {
        return RV32MInstruction(rawValue: funct3)
    }

    /// Execute an RV32M instruction, returning the rd value.
    ///
    /// Division by zero behavior follows RISC-V spec:
    ///   - DIV:  dividend (all bits set = -1 for signed, max unsigned for DIVU)
    ///   - DIVU: 2^32 - 1 (all ones)
    ///   - REM:  dividend
    ///   - REMU: dividend
    ///
    /// Signed overflow (INT32_MIN / -1):
    ///   - DIV:  INT32_MIN
    ///   - REM:  0
    private func executeMInstruction(_ instr: RV32MInstruction,
                                     rs1Val: UInt32, rs2Val: UInt32) -> UInt32 {
        let s1 = Int32(bitPattern: rs1Val)
        let s2 = Int32(bitPattern: rs2Val)

        switch instr {
        case .MUL:
            // Lower 32 bits of product (same for signed/unsigned)
            return rs1Val &* rs2Val

        case .MULH:
            // Upper 32 bits of signed * signed
            let product = Int64(s1) * Int64(s2)
            return UInt32(bitPattern: Int32(truncatingIfNeeded: product >> 32))

        case .MULHSU:
            // Upper 32 bits of signed * unsigned
            let product = Int64(s1) * Int64(UInt64(rs2Val))
            return UInt32(bitPattern: Int32(truncatingIfNeeded: product >> 32))

        case .MULHU:
            // Upper 32 bits of unsigned * unsigned
            let product = UInt64(rs1Val) * UInt64(rs2Val)
            return UInt32(truncatingIfNeeded: product >> 32)

        case .DIV:
            // Signed division
            if rs2Val == 0 {
                return UInt32(bitPattern: -1)  // All ones
            }
            if s1 == Int32.min && s2 == -1 {
                return UInt32(bitPattern: Int32.min)  // Overflow
            }
            return UInt32(bitPattern: s1 / s2)

        case .DIVU:
            // Unsigned division
            if rs2Val == 0 {
                return UInt32.max
            }
            return rs1Val / rs2Val

        case .REM:
            // Signed remainder
            if rs2Val == 0 {
                return rs1Val  // Dividend
            }
            if s1 == Int32.min && s2 == -1 {
                return 0  // Overflow case
            }
            return UInt32(bitPattern: s1 % s2)

        case .REMU:
            // Unsigned remainder
            if rs2Val == 0 {
                return rs1Val  // Dividend
            }
            return rs1Val % rs2Val
        }
    }

    // MARK: - Memory Helpers (inline for performance)

    @inline(__always)
    private func readByte(_ mem: [UInt32: UInt8], _ addr: UInt32) -> UInt8 {
        return mem[addr] ?? 0
    }

    @inline(__always)
    private func readHalf(_ mem: [UInt32: UInt8], _ addr: UInt32) -> UInt16 {
        let lo = UInt16(mem[addr] ?? 0)
        let hi = UInt16(mem[addr &+ 1] ?? 0)
        return (hi << 8) | lo
    }

    @inline(__always)
    private func readWord(_ mem: [UInt32: UInt8], _ addr: UInt32) -> UInt32 {
        let b0 = UInt32(mem[addr] ?? 0)
        let b1 = UInt32(mem[addr &+ 1] ?? 0)
        let b2 = UInt32(mem[addr &+ 2] ?? 0)
        let b3 = UInt32(mem[addr &+ 3] ?? 0)
        return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
    }

    @inline(__always)
    private func writeByte(_ mem: inout [UInt32: UInt8], _ addr: UInt32, _ val: UInt8) {
        mem[addr] = val
    }

    @inline(__always)
    private func writeHalf(_ mem: inout [UInt32: UInt8], _ addr: UInt32, _ val: UInt16) {
        mem[addr] = UInt8(val & 0xFF)
        mem[addr &+ 1] = UInt8((val >> 8) & 0xFF)
    }

    @inline(__always)
    private func writeWord(_ mem: inout [UInt32: UInt8], _ addr: UInt32, _ val: UInt32) {
        mem[addr] = UInt8(val & 0xFF)
        mem[addr &+ 1] = UInt8((val >> 8) & 0xFF)
        mem[addr &+ 2] = UInt8((val >> 16) & 0xFF)
        mem[addr &+ 3] = UInt8((val >> 24) & 0xFF)
    }

    @inline(__always)
    private func readReg(_ regs: [UInt32], _ idx: UInt8) -> UInt32 {
        if idx == 0 { return 0 }
        return regs[Int(idx)]
    }

    @inline(__always)
    private func writeReg(_ regs: inout [UInt32], _ idx: UInt8, _ val: UInt32) {
        if idx == 0 { return }
        regs[Int(idx)] = val
    }
}

// MARK: - RV32M Encoding Helpers

/// Encode an RV32M R-type instruction: op rd, rs1, rs2 (opcode=0x33, funct7=0x01)
public func encodeMType(funct3: UInt8, rs2: UInt8, rs1: UInt8, rd: UInt8) -> UInt32 {
    return (UInt32(0x01) << 25) | (UInt32(rs2) << 20) | (UInt32(rs1) << 15) |
           (UInt32(funct3) << 12) | (UInt32(rd) << 7) | 0x33
}

/// Encode MUL rd, rs1, rs2
public func encodeMUL(rd: UInt8, rs1: UInt8, rs2: UInt8) -> UInt32 {
    encodeMType(funct3: 0, rs2: rs2, rs1: rs1, rd: rd)
}

/// Encode MULH rd, rs1, rs2
public func encodeMULH(rd: UInt8, rs1: UInt8, rs2: UInt8) -> UInt32 {
    encodeMType(funct3: 1, rs2: rs2, rs1: rs1, rd: rd)
}

/// Encode MULHSU rd, rs1, rs2
public func encodeMULHSU(rd: UInt8, rs1: UInt8, rs2: UInt8) -> UInt32 {
    encodeMType(funct3: 2, rs2: rs2, rs1: rs1, rd: rd)
}

/// Encode MULHU rd, rs1, rs2
public func encodeMULHU(rd: UInt8, rs1: UInt8, rs2: UInt8) -> UInt32 {
    encodeMType(funct3: 3, rs2: rs2, rs1: rs1, rd: rd)
}

/// Encode DIV rd, rs1, rs2
public func encodeDIV(rd: UInt8, rs1: UInt8, rs2: UInt8) -> UInt32 {
    encodeMType(funct3: 4, rs2: rs2, rs1: rs1, rd: rd)
}

/// Encode DIVU rd, rs1, rs2
public func encodeDIVU(rd: UInt8, rs1: UInt8, rs2: UInt8) -> UInt32 {
    encodeMType(funct3: 5, rs2: rs2, rs1: rs1, rd: rd)
}

/// Encode REM rd, rs1, rs2
public func encodeREM(rd: UInt8, rs1: UInt8, rs2: UInt8) -> UInt32 {
    encodeMType(funct3: 6, rs2: rs2, rs1: rs1, rd: rd)
}

/// Encode REMU rd, rs1, rs2
public func encodeREMU(rd: UInt8, rs1: UInt8, rs2: UInt8) -> UInt32 {
    encodeMType(funct3: 7, rs2: rs2, rs1: rs1, rd: rd)
}
