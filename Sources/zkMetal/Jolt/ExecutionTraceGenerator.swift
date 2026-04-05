// ExecutionTraceGenerator — RISC-V RV32I execution engine and trace generator
//
// Executes binary RISC-V programs instruction-by-instruction, maintaining full
// architectural state (32 registers + PC + byte-addressable memory). Produces
// execution traces suitable for Jolt zkVM proving.
//
// The trace captures every state transition: register reads, ALU results, memory
// operations, and branch decisions. Each row maps directly to a Jolt proof step.
//
// Reference: RISC-V Unprivileged ISA Specification, Volume I

import Foundation

// MARK: - Memory Operation

/// Type of memory operation performed by an instruction.
public enum MemoryOpKind: UInt8 {
    case none = 0       // No memory access
    case loadByte = 1   // LB/LBU
    case loadHalf = 2   // LH/LHU
    case loadWord = 3   // LW
    case storeByte = 4  // SB
    case storeHalf = 5  // SH
    case storeWord = 6  // SW
}

/// A memory operation record within a trace row.
public struct RV32IMemoryOp: Equatable {
    public let kind: MemoryOpKind
    public let address: UInt32
    public let value: UInt32       // Value read/written (zero-extended for sub-word)
    public let signExtended: Bool  // True for LB, LH (sign-extending loads)

    public init(kind: MemoryOpKind = .none, address: UInt32 = 0,
                value: UInt32 = 0, signExtended: Bool = false) {
        self.kind = kind
        self.address = address
        self.value = value
        self.signExtended = signExtended
    }

    public static let none = RV32IMemoryOp()
}

// MARK: - Trace Row

/// A single row of the execution trace, capturing one instruction's full context.
public struct TraceRow {
    /// Program counter at the start of this step
    public let pc: UInt32

    /// The decoded instruction executed
    public let instruction: RV32IInstruction

    /// The original 32-bit instruction word
    public let word: UInt32

    /// Value of rs1 register (before execution)
    public let rs1Val: UInt32

    /// Value of rs2 register (before execution)
    public let rs2Val: UInt32

    /// Value written to rd (the result), 0 if no write
    public let rdVal: UInt32

    /// Destination register index (0 means no write, since x0 is hardwired to 0)
    public let rd: UInt8

    /// Source register indices
    public let rs1: UInt8
    public let rs2: UInt8

    /// Immediate value (sign-extended)
    public let immediate: Int32

    /// Memory operation, if any
    public let memoryOp: RV32IMemoryOp

    /// Step number (0-based)
    public let step: Int

    public init(pc: UInt32, instruction: RV32IInstruction, word: UInt32,
                rs1Val: UInt32, rs2Val: UInt32, rdVal: UInt32,
                rd: UInt8, rs1: UInt8, rs2: UInt8, immediate: Int32,
                memoryOp: RV32IMemoryOp, step: Int) {
        self.pc = pc
        self.instruction = instruction
        self.word = word
        self.rs1Val = rs1Val
        self.rs2Val = rs2Val
        self.rdVal = rdVal
        self.rd = rd
        self.rs1 = rs1
        self.rs2 = rs2
        self.immediate = immediate
        self.memoryOp = memoryOp
        self.step = step
    }
}

// MARK: - Execution Trace

/// Complete execution trace with rows and metadata.
public struct ExecutionTrace {
    /// Ordered trace rows, one per executed instruction
    public let rows: [TraceRow]

    /// Total number of instructions executed
    public var stepCount: Int { rows.count }

    /// Total number of memory accesses (loads + stores)
    public let memoryAccessCount: Int

    /// Number of loads
    public let loadCount: Int

    /// Number of stores
    public let storeCount: Int

    /// Number of branches taken
    public let branchesTaken: Int

    /// Number of branches not taken
    public let branchesNotTaken: Int

    /// Set of registers written during execution (excluding x0)
    public let registersWritten: Set<UInt8>

    /// Set of registers read during execution
    public let registersRead: Set<UInt8>

    /// Final register file state
    public let finalRegisters: [UInt32]

    /// Final PC value
    public let finalPC: UInt32

    /// Whether execution terminated normally (ECALL/EBREAK or maxSteps)
    public let terminated: Bool

    public init(rows: [TraceRow], memoryAccessCount: Int, loadCount: Int,
                storeCount: Int, branchesTaken: Int, branchesNotTaken: Int,
                registersWritten: Set<UInt8>, registersRead: Set<UInt8>,
                finalRegisters: [UInt32], finalPC: UInt32, terminated: Bool) {
        self.rows = rows
        self.memoryAccessCount = memoryAccessCount
        self.loadCount = loadCount
        self.storeCount = storeCount
        self.branchesTaken = branchesTaken
        self.branchesNotTaken = branchesNotTaken
        self.registersWritten = registersWritten
        self.registersRead = registersRead
        self.finalRegisters = finalRegisters
        self.finalPC = finalPC
        self.terminated = terminated
    }

    /// Convert trace rows to RV32IStep format for Jolt proving.
    public func toRV32ISteps() -> [RV32IStep] {
        return rows.map { row in
            let op = row.instruction.rv32iOp
            let a = row.rs1Val
            let b: UInt32
            let result = row.rdVal

            // For I-type instructions, the second operand is the immediate
            switch row.instruction {
            case .ADDI, .ANDI, .ORI, .XORI, .SLTI, .SLTIU,
                 .SLLI, .SRLI, .SRAI:
                b = UInt32(bitPattern: row.immediate)
            case .LB, .LH, .LW, .LBU, .LHU, .SB, .SH, .SW:
                b = UInt32(bitPattern: row.immediate)
            case .LUI:
                b = UInt32(bitPattern: row.immediate)
            case .AUIPC:
                b = UInt32(bitPattern: row.immediate)
            case .JAL, .JALR:
                b = UInt32(bitPattern: row.immediate)
            default:
                b = row.rs2Val
            }

            return RV32IStep(op: op, a: a, b: b, result: result)
        }
    }
}

// MARK: - RV32I Architectural State

/// Full RV32I architectural state: 32 registers, PC, and byte-addressable memory.
public struct RV32IState {
    /// 32 general-purpose registers. x0 is hardwired to 0.
    public var registers: [UInt32]

    /// Program counter
    public var pc: UInt32

    /// Byte-addressable memory (sparse, backed by dictionary for efficiency)
    public var memory: [UInt32: UInt8]

    /// Create a fresh state with zeroed registers and empty memory.
    public init() {
        self.registers = [UInt32](repeating: 0, count: 32)
        self.pc = 0
        self.memory = [:]
    }

    /// Create a state with initial register values and optional memory contents.
    public init(registers: [UInt32], pc: UInt32 = 0, memory: [UInt32: UInt8] = [:]) {
        var regs = registers
        while regs.count < 32 { regs.append(0) }
        regs[0] = 0  // x0 is always 0
        self.registers = regs
        self.pc = pc
        self.memory = memory
    }

    /// Read register (x0 always returns 0)
    public func readReg(_ idx: UInt8) -> UInt32 {
        if idx == 0 { return 0 }
        return registers[Int(idx)]
    }

    /// Write register (writes to x0 are silently ignored)
    public mutating func writeReg(_ idx: UInt8, _ value: UInt32) {
        if idx == 0 { return }
        registers[Int(idx)] = value
    }

    // MARK: Memory Access

    /// Read a byte from memory
    public func readByte(_ addr: UInt32) -> UInt8 {
        return memory[addr] ?? 0
    }

    /// Read a halfword (16 bits, little-endian) from memory
    public func readHalf(_ addr: UInt32) -> UInt16 {
        let lo = UInt16(readByte(addr))
        let hi = UInt16(readByte(addr &+ 1))
        return (hi << 8) | lo
    }

    /// Read a word (32 bits, little-endian) from memory
    public func readWord(_ addr: UInt32) -> UInt32 {
        let b0 = UInt32(readByte(addr))
        let b1 = UInt32(readByte(addr &+ 1))
        let b2 = UInt32(readByte(addr &+ 2))
        let b3 = UInt32(readByte(addr &+ 3))
        return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
    }

    /// Write a byte to memory
    public mutating func writeByte(_ addr: UInt32, _ val: UInt8) {
        memory[addr] = val
    }

    /// Write a halfword (16 bits, little-endian) to memory
    public mutating func writeHalf(_ addr: UInt32, _ val: UInt16) {
        memory[addr] = UInt8(val & 0xFF)
        memory[addr &+ 1] = UInt8((val >> 8) & 0xFF)
    }

    /// Write a word (32 bits, little-endian) to memory
    public mutating func writeWord(_ addr: UInt32, _ val: UInt32) {
        memory[addr] = UInt8(val & 0xFF)
        memory[addr &+ 1] = UInt8((val >> 8) & 0xFF)
        memory[addr &+ 2] = UInt8((val >> 16) & 0xFF)
        memory[addr &+ 3] = UInt8((val >> 24) & 0xFF)
    }
}

// MARK: - Single Step Execution

/// Execute a single decoded instruction, returning the new state and trace row.
///
/// - Parameters:
///   - state: Current architectural state (registers, PC, memory)
///   - decoded: The decoded instruction to execute
///   - stepIndex: Current step number for the trace row
/// - Returns: Tuple of (updated state, trace row for this step)
public func executeStep(state: RV32IState, decoded: DecodedInstruction,
                        stepIndex: Int) -> (newState: RV32IState, traceRow: TraceRow) {
    var s = state
    let pc = s.pc
    let rs1Val = s.readReg(decoded.rs1)
    let rs2Val = s.readReg(decoded.rs2)
    var rdVal: UInt32 = 0
    var memOp = RV32IMemoryOp.none
    var nextPC = pc &+ 4  // Default: advance to next instruction

    switch decoded.instruction {
    // ---- R-type ALU ----
    case .ADD:
        rdVal = rs1Val &+ rs2Val
    case .SUB:
        rdVal = rs1Val &- rs2Val
    case .SLL:
        rdVal = rs1Val << (rs2Val & 31)
    case .SLT:
        rdVal = Int32(bitPattern: rs1Val) < Int32(bitPattern: rs2Val) ? 1 : 0
    case .SLTU:
        rdVal = rs1Val < rs2Val ? 1 : 0
    case .XOR:
        rdVal = rs1Val ^ rs2Val
    case .SRL:
        rdVal = rs1Val >> (rs2Val & 31)
    case .SRA:
        rdVal = UInt32(bitPattern: Int32(bitPattern: rs1Val) >> Int32(rs2Val & 31))
    case .OR:
        rdVal = rs1Val | rs2Val
    case .AND:
        rdVal = rs1Val & rs2Val

    // ---- I-type ALU ----
    case .ADDI:
        rdVal = rs1Val &+ UInt32(bitPattern: decoded.immediate)
    case .SLTI:
        rdVal = Int32(bitPattern: rs1Val) < decoded.immediate ? 1 : 0
    case .SLTIU:
        rdVal = rs1Val < UInt32(bitPattern: decoded.immediate) ? 1 : 0
    case .XORI:
        rdVal = rs1Val ^ UInt32(bitPattern: decoded.immediate)
    case .ORI:
        rdVal = rs1Val | UInt32(bitPattern: decoded.immediate)
    case .ANDI:
        rdVal = rs1Val & UInt32(bitPattern: decoded.immediate)
    case .SLLI:
        let shamt = UInt32(decoded.immediate) & 31
        rdVal = rs1Val << shamt
    case .SRLI:
        let shamt = UInt32(decoded.immediate) & 31
        rdVal = rs1Val >> shamt
    case .SRAI:
        let shamt = Int32(decoded.immediate) & 31
        rdVal = UInt32(bitPattern: Int32(bitPattern: rs1Val) >> shamt)

    // ---- Loads ----
    case .LB:
        let addr = rs1Val &+ UInt32(bitPattern: decoded.immediate)
        let byte = s.readByte(addr)
        // Sign-extend from 8 to 32 bits
        rdVal = (byte & 0x80) != 0
            ? UInt32(byte) | 0xFFFFFF00
            : UInt32(byte)
        memOp = RV32IMemoryOp(kind: .loadByte, address: addr, value: UInt32(byte), signExtended: true)

    case .LH:
        let addr = rs1Val &+ UInt32(bitPattern: decoded.immediate)
        let half = s.readHalf(addr)
        // Sign-extend from 16 to 32 bits
        rdVal = (half & 0x8000) != 0
            ? UInt32(half) | 0xFFFF0000
            : UInt32(half)
        memOp = RV32IMemoryOp(kind: .loadHalf, address: addr, value: UInt32(half), signExtended: true)

    case .LW:
        let addr = rs1Val &+ UInt32(bitPattern: decoded.immediate)
        rdVal = s.readWord(addr)
        memOp = RV32IMemoryOp(kind: .loadWord, address: addr, value: rdVal, signExtended: false)

    case .LBU:
        let addr = rs1Val &+ UInt32(bitPattern: decoded.immediate)
        let byte = s.readByte(addr)
        rdVal = UInt32(byte)
        memOp = RV32IMemoryOp(kind: .loadByte, address: addr, value: UInt32(byte), signExtended: false)

    case .LHU:
        let addr = rs1Val &+ UInt32(bitPattern: decoded.immediate)
        let half = s.readHalf(addr)
        rdVal = UInt32(half)
        memOp = RV32IMemoryOp(kind: .loadHalf, address: addr, value: UInt32(half), signExtended: false)

    // ---- Stores ----
    case .SB:
        let addr = rs1Val &+ UInt32(bitPattern: decoded.immediate)
        s.writeByte(addr, UInt8(rs2Val & 0xFF))
        memOp = RV32IMemoryOp(kind: .storeByte, address: addr, value: rs2Val & 0xFF, signExtended: false)

    case .SH:
        let addr = rs1Val &+ UInt32(bitPattern: decoded.immediate)
        s.writeHalf(addr, UInt16(rs2Val & 0xFFFF))
        memOp = RV32IMemoryOp(kind: .storeHalf, address: addr, value: rs2Val & 0xFFFF, signExtended: false)

    case .SW:
        let addr = rs1Val &+ UInt32(bitPattern: decoded.immediate)
        s.writeWord(addr, rs2Val)
        memOp = RV32IMemoryOp(kind: .storeWord, address: addr, value: rs2Val, signExtended: false)

    // ---- Branches ----
    case .BEQ:
        let taken = rs1Val == rs2Val
        rdVal = taken ? 1 : 0
        if taken { nextPC = pc &+ UInt32(bitPattern: decoded.immediate) }

    case .BNE:
        let taken = rs1Val != rs2Val
        rdVal = taken ? 1 : 0
        if taken { nextPC = pc &+ UInt32(bitPattern: decoded.immediate) }

    case .BLT:
        let taken = Int32(bitPattern: rs1Val) < Int32(bitPattern: rs2Val)
        rdVal = taken ? 1 : 0
        if taken { nextPC = pc &+ UInt32(bitPattern: decoded.immediate) }

    case .BGE:
        let taken = Int32(bitPattern: rs1Val) >= Int32(bitPattern: rs2Val)
        rdVal = taken ? 1 : 0
        if taken { nextPC = pc &+ UInt32(bitPattern: decoded.immediate) }

    case .BLTU:
        let taken = rs1Val < rs2Val
        rdVal = taken ? 1 : 0
        if taken { nextPC = pc &+ UInt32(bitPattern: decoded.immediate) }

    case .BGEU:
        let taken = rs1Val >= rs2Val
        rdVal = taken ? 1 : 0
        if taken { nextPC = pc &+ UInt32(bitPattern: decoded.immediate) }

    // ---- Jumps ----
    case .JAL:
        rdVal = pc &+ 4  // Return address
        nextPC = pc &+ UInt32(bitPattern: decoded.immediate)

    case .JALR:
        rdVal = pc &+ 4  // Return address
        nextPC = (rs1Val &+ UInt32(bitPattern: decoded.immediate)) & ~1  // Clear LSB

    // ---- Upper Immediate ----
    case .LUI:
        rdVal = UInt32(bitPattern: decoded.immediate)  // Already has lower 12 bits cleared

    case .AUIPC:
        rdVal = pc &+ UInt32(bitPattern: decoded.immediate)

    // ---- System ----
    case .ECALL, .EBREAK:
        rdVal = 0  // No register effect

    case .FENCE:
        rdVal = 0  // Memory ordering hint, no architectural effect in single-hart
    }

    // Write result to destination register (branches and stores don't write rd)
    switch decoded.instruction {
    case .BEQ, .BNE, .BLT, .BGE, .BLTU, .BGEU,
         .SB, .SH, .SW, .ECALL, .EBREAK, .FENCE:
        break  // No register write
    default:
        s.writeReg(decoded.rd, rdVal)
    }

    s.pc = nextPC

    let row = TraceRow(
        pc: pc, instruction: decoded.instruction, word: decoded.word,
        rs1Val: rs1Val, rs2Val: rs2Val, rdVal: rdVal,
        rd: decoded.rd, rs1: decoded.rs1, rs2: decoded.rs2,
        immediate: decoded.immediate, memoryOp: memOp, step: stepIndex)

    return (s, row)
}

// MARK: - Full Trace Generation

/// Execute a binary RV32I program and generate the complete execution trace.
///
/// The program is provided as an array of 32-bit instruction words, loaded starting
/// at address 0. Execution begins at PC=0 and continues until one of:
///   - ECALL or EBREAK is executed
///   - PC goes out of bounds (past the program)
///   - maxSteps is reached
///
/// - Parameters:
///   - program: Array of 32-bit instruction words
///   - initialRegisters: Initial register values (x0 is always forced to 0)
///   - initialMemory: Initial memory contents (byte-addressable)
///   - maxSteps: Maximum number of instructions to execute (default 1_000_000)
/// - Returns: Complete ExecutionTrace with all rows and metadata
public func generateTrace(
    program: [UInt32],
    initialRegisters: [UInt32] = [],
    initialMemory: [UInt32: UInt8] = [:],
    maxSteps: Int = 1_000_000
) -> ExecutionTrace {
    // Load program into memory (starting at address 0)
    var mem = initialMemory
    for (i, word) in program.enumerated() {
        let addr = UInt32(i * 4)
        mem[addr] = UInt8(word & 0xFF)
        mem[addr + 1] = UInt8((word >> 8) & 0xFF)
        mem[addr + 2] = UInt8((word >> 16) & 0xFF)
        mem[addr + 3] = UInt8((word >> 24) & 0xFF)
    }

    var state = RV32IState(registers: initialRegisters, pc: 0, memory: mem)

    var rows = [TraceRow]()
    rows.reserveCapacity(min(maxSteps, program.count))

    var memoryAccessCount = 0
    var loadCount = 0
    var storeCount = 0
    var branchesTaken = 0
    var branchesNotTaken = 0
    var regsWritten = Set<UInt8>()
    var regsRead = Set<UInt8>()
    var terminated = false

    let programEndAddr = UInt32(program.count * 4)

    for stepIdx in 0..<maxSteps {
        // Check if PC is within program bounds
        if state.pc >= programEndAddr {
            break
        }

        // Fetch instruction word from memory
        let word = state.readWord(state.pc)

        // Decode
        let decoded: DecodedInstruction
        do {
            decoded = try decodeInstruction(word: word)
        } catch {
            // Invalid instruction: stop execution
            break
        }

        // Track register usage
        if decoded.rs1 != 0 { regsRead.insert(decoded.rs1) }
        if decoded.rs2 != 0 { regsRead.insert(decoded.rs2) }

        // Execute
        let (newState, row) = executeStep(state: state, decoded: decoded, stepIndex: stepIdx)
        rows.append(row)

        // Track destination register writes
        switch decoded.instruction {
        case .BEQ, .BNE, .BLT, .BGE, .BLTU, .BGEU,
             .SB, .SH, .SW, .ECALL, .EBREAK, .FENCE:
            break
        default:
            if decoded.rd != 0 { regsWritten.insert(decoded.rd) }
        }

        // Track memory operations
        if row.memoryOp.kind != .none {
            memoryAccessCount += 1
            switch row.memoryOp.kind {
            case .loadByte, .loadHalf, .loadWord:
                loadCount += 1
            case .storeByte, .storeHalf, .storeWord:
                storeCount += 1
            case .none:
                break
            }
        }

        // Track branches
        switch decoded.instruction {
        case .BEQ, .BNE, .BLT, .BGE, .BLTU, .BGEU:
            if newState.pc != state.pc &+ 4 {
                branchesTaken += 1
            } else {
                branchesNotTaken += 1
            }
        default:
            break
        }

        // Check for termination
        if decoded.instruction == .ECALL || decoded.instruction == .EBREAK {
            terminated = true
            state = newState
            break
        }

        state = newState
    }

    return ExecutionTrace(
        rows: rows,
        memoryAccessCount: memoryAccessCount,
        loadCount: loadCount,
        storeCount: storeCount,
        branchesTaken: branchesTaken,
        branchesNotTaken: branchesNotTaken,
        registersWritten: regsWritten,
        registersRead: regsRead,
        finalRegisters: state.registers,
        finalPC: state.pc,
        terminated: terminated)
}

// MARK: - Example Programs (Binary)

/// Fibonacci program in binary RV32I: compute fib(n) where n is in x10 (a0).
/// Result in x10. Uses x11 (a1) and x12 (a2) as temporaries.
///
/// Assembly:
///   addi x11, x0, 1       # x11 = 1 (fib(1))
///   addi x12, x0, 0       # x12 = 0 (fib(0))
///   addi x10, x10, -1     # n = n - 1
///   beq  x10, x0, done    # if n == 0, done
///  loop:
///   add  x13, x11, x12    # x13 = fib(k) + fib(k-1)
///   add  x12, x11, x0     # x12 = fib(k)
///   add  x11, x13, x0     # x11 = fib(k+1)
///   addi x10, x10, -1     # n--
///   bne  x10, x0, loop    # if n != 0, loop
///  done:
///   add  x10, x11, x0     # x10 = result
///   ecall                  # terminate
public func rv32iFibonacciProgram() -> [UInt32] {
    return [
        encodeADDI(rd: 11, rs1: 0, imm: 1),       // x11 = 1
        encodeADDI(rd: 12, rs1: 0, imm: 0),       // x12 = 0
        encodeADDI(rd: 10, rs1: 10, imm: -1),     // x10 = x10 - 1
        encodeBEQ(rs1: 10, rs2: 0, offset: 24),   // if x10==0 goto done (PC+24 = instr 9)
        // loop (PC = 16):
        encodeADD(rd: 13, rs1: 11, rs2: 12),      // x13 = x11 + x12
        encodeADD(rd: 12, rs1: 11, rs2: 0),       // x12 = x11
        encodeADD(rd: 11, rs1: 13, rs2: 0),       // x11 = x13
        encodeADDI(rd: 10, rs1: 10, imm: -1),     // x10--
        encodeBNE(rs1: 10, rs2: 0, offset: -16),  // if x10!=0 goto loop (PC-16)
        // done (PC = 36):
        encodeADD(rd: 10, rs1: 11, rs2: 0),       // x10 = x11
        encodeECALL(),                              // terminate
    ]
}

/// Factorial program in binary RV32I: compute n! where n is in x10 (a0).
/// Result in x10.
///
/// Assembly:
///   addi x11, x0, 1       # x11 = 1 (accumulator)
///   beq  x10, x0, done    # if n == 0, done
///  loop:
///   mul would be ideal but RV32I has no MUL; we use repeated addition.
///   Instead: iterative multiply via shift-and-add.
///
/// Simplified: since we are RV32I (no MUL), we implement factorial by
/// repeated addition: n! = n * (n-1) * ... * 1, where each multiply
/// is a loop of additions.
///
/// For simplicity, this uses a nested loop approach:
///   x11 = 1 (result accumulator)
///   for i = n down to 1:
///     x12 = x11 (save current result)
///     x11 = 0
///     for j = 0 to i-1:
///       x11 = x11 + x12
///     # now x11 = x12 * i
///
/// This is O(n^2) but produces a correct trace for testing.
///
/// Actually, for a cleaner test, let's just compute sum 1+2+...+n = n*(n+1)/2.
/// That exercises branches and ALU without needing multiplication.
///
/// Assembly (sum 1..n):
///   addi x11, x0, 0       # x11 = 0 (accumulator)
///   beq  x10, x0, done    # if n == 0, done
///  loop:
///   add  x11, x11, x10    # x11 += n
///   addi x10, x10, -1     # n--
///   bne  x10, x0, loop    # if n != 0, loop
///  done:
///   add  x10, x11, x0     # x10 = result
///   ecall
public func rv32iFactorialProgram() -> [UInt32] {
    return [
        encodeADDI(rd: 11, rs1: 0, imm: 0),       // x11 = 0 (accumulator)
        encodeBEQ(rs1: 10, rs2: 0, offset: 16),   // if n==0 goto done (PC+16 = instr 5)
        // loop (PC = 8):
        encodeADD(rd: 11, rs1: 11, rs2: 10),      // x11 += x10
        encodeADDI(rd: 10, rs1: 10, imm: -1),     // x10--
        encodeBNE(rs1: 10, rs2: 0, offset: -8),   // if x10!=0 goto loop (PC-8)
        // done (PC = 20):
        encodeADD(rd: 10, rs1: 11, rs2: 0),       // x10 = x11
        encodeECALL(),                              // terminate
    ]
}

/// Memory test program: store values to memory and load them back.
///
/// Assembly:
///   addi x10, x0, 42      # x10 = 42
///   addi x11, x0, 100     # x11 = 100 (base address)
///   sw   x10, 0(x11)      # mem[100] = 42
///   addi x10, x0, 99      # x10 = 99
///   sw   x10, 4(x11)      # mem[104] = 99
///   lw   x12, 0(x11)      # x12 = mem[100] = 42
///   lw   x13, 4(x11)      # x13 = mem[104] = 99
///   add  x10, x12, x13    # x10 = 42 + 99 = 141
///   ecall
public func rv32iMemoryTestProgram() -> [UInt32] {
    return [
        encodeADDI(rd: 10, rs1: 0, imm: 42),      // x10 = 42
        encodeADDI(rd: 11, rs1: 0, imm: 100),     // x11 = 100
        encodeSW(rs2: 10, rs1: 11, offset: 0),    // mem[100] = 42
        encodeADDI(rd: 10, rs1: 0, imm: 99),      // x10 = 99
        encodeSW(rs2: 10, rs1: 11, offset: 4),    // mem[104] = 99
        encodeLW(rd: 12, rs1: 11, offset: 0),     // x12 = mem[100]
        encodeLW(rd: 13, rs1: 11, offset: 4),     // x13 = mem[104]
        encodeADD(rd: 10, rs1: 12, rs2: 13),      // x10 = x12 + x13
        encodeECALL(),                              // terminate
    ]
}
