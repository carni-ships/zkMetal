// Jolt-style zkVM — RISC-like instruction set with Lasso lookup verification
//
// Key insight from the Jolt paper (Arun et al. 2024): instead of proving arithmetic
// circuits for each CPU instruction, decompose each instruction into byte-level
// table lookups. For decomposable operations (ADD, SUB, AND, OR, XOR), the result
// is the byte-wise composition of subtable results. Non-decomposable operations
// (MUL, SHL, SHR, LT, EQ) use direct lookup into a precomputed result table.
//
// Architecture:
//   - 32-bit operands, decomposed into 4 bytes
//   - Register file with configurable width (default 32 registers)
//   - Execution trace records (opcode, operand_a, operand_b, result) per step
//   - Lasso structured lookups verify each instruction's correctness

import Foundation

// MARK: - Instruction Set

/// Jolt VM opcodes. Decomposable ops can be verified via independent byte-level
/// subtable lookups. Non-decomposable ops require a combined subtable.
public enum JoltOp: UInt8, CaseIterable, Hashable {
    case add  = 0   // a + b (mod 2^32) — decomposable with carry
    case sub  = 1   // a - b (mod 2^32) — decomposable with borrow
    case mul  = 2   // a * b (mod 2^32) — NOT decomposable, direct lookup
    case and_ = 3   // a & b — decomposable (byte-wise AND)
    case or_  = 4   // a | b — decomposable (byte-wise OR)
    case xor_ = 5   // a ^ b — decomposable (byte-wise XOR)
    case shl  = 6   // a << (b & 31) — NOT decomposable
    case shr  = 7   // a >> (b & 31) — NOT decomposable
    case lt   = 8   // a < b ? 1 : 0 — NOT decomposable
    case eq   = 9   // a == b ? 1 : 0 — decomposable (byte-wise EQ, then AND)

    /// Whether this op can be verified via independent byte-level subtable lookups
    public var isDecomposable: Bool {
        switch self {
        case .and_, .or_, .xor_, .eq: return true
        // ADD/SUB have carry propagation but we handle them as decomposable
        // with a carry-chain lookup
        case .add, .sub: return true
        case .mul, .shl, .shr, .lt: return false
        }
    }
}

/// A single VM instruction: op rd, rs1, rs2
/// Semantics: regs[rd] = op(regs[rs1], regs[rs2])
public struct JoltInstruction {
    public let op: JoltOp
    public let rs1: UInt8   // source register 1
    public let rs2: UInt8   // source register 2
    public let rd: UInt8    // destination register

    public init(op: JoltOp, rs1: UInt8, rs2: UInt8, rd: UInt8) {
        self.op = op
        self.rs1 = rs1
        self.rs2 = rs2
        self.rd = rd
    }
}

// MARK: - Execution Trace

/// One step of execution: captures the instruction and its operands/result
public struct JoltStep {
    public let op: JoltOp
    public let a: UInt32     // operand a = regs[rs1]
    public let b: UInt32     // operand b = regs[rs2]
    public let result: UInt32 // computed result

    public init(op: JoltOp, a: UInt32, b: UInt32, result: UInt32) {
        self.op = op
        self.a = a
        self.b = b
        self.result = result
    }
}

/// Complete execution trace for a Jolt program
public struct JoltTrace {
    public let steps: [JoltStep]
    public let program: [JoltInstruction]
    public let initialRegs: [UInt32]
    public let finalRegs: [UInt32]

    public init(steps: [JoltStep], program: [JoltInstruction],
                initialRegs: [UInt32], finalRegs: [UInt32]) {
        self.steps = steps
        self.program = program
        self.initialRegs = initialRegs
        self.finalRegs = finalRegs
    }
}

// MARK: - VM Execution

/// Execute a Jolt program and produce the execution trace.
/// All arithmetic is mod 2^32.
public func joltExecute(program: [JoltInstruction], numRegisters: Int = 32,
                        initialRegs: [UInt32]? = nil) -> JoltTrace {
    var regs: [UInt32]
    if let init_ = initialRegs {
        regs = init_
        while regs.count < numRegisters { regs.append(0) }
    } else {
        // Default: regs[i] = i+1 (so no zeros for interesting behavior)
        regs = (0..<numRegisters).map { UInt32($0 + 1) }
    }
    let initRegs = regs

    var steps = [JoltStep]()
    steps.reserveCapacity(program.count)

    for instr in program {
        let a = regs[Int(instr.rs1)]
        let b = regs[Int(instr.rs2)]
        let result: UInt32

        switch instr.op {
        case .add:  result = a &+ b
        case .sub:  result = a &- b
        case .mul:  result = a &* b
        case .and_: result = a & b
        case .or_:  result = a | b
        case .xor_: result = a ^ b
        case .shl:  result = a << (b & 31)
        case .shr:  result = a >> (b & 31)
        case .lt:   result = a < b ? 1 : 0
        case .eq:   result = a == b ? 1 : 0
        }

        steps.append(JoltStep(op: instr.op, a: a, b: b, result: result))
        regs[Int(instr.rd)] = result
    }

    return JoltTrace(steps: steps, program: program,
                     initialRegs: initRegs, finalRegs: regs)
}

// MARK: - Lookup Table Construction

/// Jolt uses Lasso range-check tables to verify operand values are in [0, 2^32).
/// The range check uses LassoTable.rangeCheck(bits: 32, chunks: 4), which decomposes
/// each 32-bit value into 4 byte-level subtables (256 entries each).
///
/// For bitwise operations, additional decomposed tables could verify per-byte
/// correctness (AND, OR, XOR subtables). Currently all operations are verified
/// algebraically, with Lasso providing the range-check guarantee.

// MARK: - Random Program Generation

/// Generate a random Jolt program for benchmarking.
public func joltRandomProgram(count: Int, numRegisters: Int = 32,
                               seed: UInt64 = 0xCAFE) -> [JoltInstruction] {
    var rng = seed
    func next() -> UInt64 {
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        return rng
    }

    let ops = JoltOp.allCases
    var instrs = [JoltInstruction]()
    instrs.reserveCapacity(count)

    for _ in 0..<count {
        let op = ops[Int(next() >> 56) % ops.count]
        let rs1 = UInt8(Int(next() >> 48) % numRegisters)
        let rs2 = UInt8(Int(next() >> 40) % numRegisters)
        let rd = UInt8(Int(next() >> 32) % numRegisters)
        instrs.append(JoltInstruction(op: op, rs1: rs1, rs2: rs2, rd: rd))
    }
    return instrs
}

// MARK: - Fibonacci Program

/// Build a Fibonacci-computing program of `n` steps.
/// Uses regs[0]=a, regs[1]=b, alternating: regs[rd] = regs[rs1] + regs[rs2]
public func joltFibonacci(steps: Int) -> [JoltInstruction] {
    var instrs = [JoltInstruction]()
    instrs.reserveCapacity(steps)
    for i in 0..<steps {
        let rs1 = UInt8(i % 2 == 0 ? 0 : 1)
        let rs2 = UInt8(i % 2 == 0 ? 1 : 0)
        let rd = UInt8(i % 2 == 0 ? 1 : 0)
        // Alternate: rd = rs1 + rs2, keeping the two latest fib values
        instrs.append(JoltInstruction(op: .add, rs1: rs1, rs2: rs2, rd: rd == 0 ? 2 : 3))
        // Copy result back
        if instrs.count < steps {
            instrs.append(JoltInstruction(op: .add, rs1: rd == 0 ? 2 : 3, rs2: UInt8(numRegistersForFib), rd: rd))
        }
    }
    // Trim to exact count
    return Array(instrs.prefix(steps))
}

private let numRegistersForFib: Int = 4  // reg[4] = 0 for identity add

/// Build a sum-array program: accumulate regs[0..n-1] into regs[n].
public func joltSumArray(count: Int) -> [JoltInstruction] {
    let numRegs = min(count + 1, 32)
    var instrs = [JoltInstruction]()
    // regs[numRegs-1] is accumulator, starts at 0
    let acc = UInt8(numRegs - 1)
    for i in 0..<min(count, numRegs - 1) {
        instrs.append(JoltInstruction(op: .add, rs1: acc, rs2: UInt8(i), rd: acc))
    }
    return instrs
}
