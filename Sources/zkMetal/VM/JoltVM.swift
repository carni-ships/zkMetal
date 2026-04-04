// Jolt-style zkVM — Instruction Set and Execution
// Implements a simple VM where each instruction is verified via Lasso structured lookups.
// Instead of encoding instructions as arithmetic circuits, each instruction maps to a
// table lookup: (opcode, operand1, operand2) -> result, decomposed into byte-level
// subtables via Lasso's tensor structure.
//
// References: Jolt (Arun, Setty, Thaler 2024)

import Foundation

// MARK: - Instruction Set

/// Simple instruction opcodes for the Jolt VM.
/// Each operates on 32-bit unsigned integers.
public enum JoltOp: UInt8, CaseIterable {
    case add = 0   // rd = rs1 + rs2 (mod 2^32)
    case sub = 1   // rd = rs1 - rs2 (mod 2^32)
    case mul = 2   // rd = rs1 * rs2 (low 32 bits)
    case and_ = 3  // rd = rs1 & rs2
    case or_ = 4   // rd = rs1 | rs2
    case xor_ = 5  // rd = rs1 ^ rs2
    case shl = 6   // rd = rs1 << (rs2 % 32)
    case shr = 7   // rd = rs1 >> (rs2 % 32)
    case eq = 8    // rd = (rs1 == rs2) ? 1 : 0
    case lt = 9    // rd = (rs1 < rs2) ? 1 : 0
}

/// A single VM instruction.
public struct JoltInstruction {
    public let op: JoltOp
    public let rs1: UInt32  // source operand 1 (register index or immediate)
    public let rs2: UInt32  // source operand 2
    public let rd: UInt32   // destination register

    public init(op: JoltOp, rs1: UInt32, rs2: UInt32, rd: UInt32) {
        self.op = op
        self.rs1 = rs1
        self.rs2 = rs2
        self.rd = rd
    }
}

/// Execution trace: instructions + register snapshots after each step.
public struct JoltTrace {
    /// The executed instructions
    public let instructions: [JoltInstruction]
    /// Register state after each instruction (registerFile[i] = state after instruction i)
    public let registerFile: [[UInt32]]
    /// Operand values: (a, b, result) for each instruction (resolved from registers)
    public let operands: [(a: UInt32, b: UInt32, result: UInt32)]

    public init(instructions: [JoltInstruction], registerFile: [[UInt32]],
                operands: [(a: UInt32, b: UInt32, result: UInt32)]) {
        self.instructions = instructions
        self.registerFile = registerFile
        self.operands = operands
    }
}

// MARK: - VM Execution

/// Execute a program and produce a trace.
/// The VM has `numRegisters` 32-bit registers, all initialized to 0.
/// rs1/rs2 are register indices; rd is the destination register index.
public func joltExecute(program: [JoltInstruction], numRegisters: Int = 32) -> JoltTrace {
    var regs = [UInt32](repeating: 0, count: numRegisters)
    var registerFile = [[UInt32]]()
    var operands = [(a: UInt32, b: UInt32, result: UInt32)]()
    registerFile.reserveCapacity(program.count)
    operands.reserveCapacity(program.count)

    for instr in program {
        let a = regs[Int(instr.rs1 % UInt32(numRegisters))]
        let b = regs[Int(instr.rs2 % UInt32(numRegisters))]
        let result = joltCompute(op: instr.op, a: a, b: b)
        regs[Int(instr.rd % UInt32(numRegisters))] = result
        registerFile.append(regs)
        operands.append((a: a, b: b, result: result))
    }

    return JoltTrace(instructions: program, registerFile: registerFile, operands: operands)
}

/// Compute instruction result (reference implementation).
public func joltCompute(op: JoltOp, a: UInt32, b: UInt32) -> UInt32 {
    switch op {
    case .add:  return a &+ b
    case .sub:  return a &- b
    case .mul:  return a &* b
    case .and_: return a & b
    case .or_:  return a | b
    case .xor_: return a ^ b
    case .shl:  return a << (b % 32)
    case .shr:  return a >> (b % 32)
    case .eq:   return a == b ? 1 : 0
    case .lt:   return a < b ? 1 : 0
    }
}

// MARK: - Byte-Level Decomposition for Lasso

/// Decompose a 32-bit instruction into 4 byte-level lookups.
/// For bitwise ops (AND, OR, XOR): byte k of result = op(byte k of a, byte k of b).
/// For arithmetic (ADD, SUB): byte-level with carry chain.
/// For MUL, EQ, LT, SHL, SHR: not byte-decomposable (handled via direct lookup).
///
/// Returns: array of 4 (a_byte, b_byte, result_byte) tuples, or nil for non-decomposable ops.
public func joltDecomposeBytes(op: JoltOp, a: UInt32, b: UInt32, result: UInt32)
    -> [(UInt8, UInt8, UInt8)]?
{
    switch op {
    case .and_, .or_, .xor_:
        var decomp = [(UInt8, UInt8, UInt8)]()
        for k in 0..<4 {
            let ab = UInt8((a >> (k * 8)) & 0xFF)
            let bb = UInt8((b >> (k * 8)) & 0xFF)
            let rb = UInt8((result >> (k * 8)) & 0xFF)
            decomp.append((ab, bb, rb))
        }
        return decomp

    case .add:
        var decomp = [(UInt8, UInt8, UInt8)]()
        var carry: UInt32 = 0
        for k in 0..<4 {
            let ab = UInt32((a >> (k * 8)) & 0xFF)
            let bb = UInt32((b >> (k * 8)) & 0xFF)
            let sum = ab + bb + carry
            let rb = UInt8(sum & 0xFF)
            carry = sum >> 8
            decomp.append((UInt8(ab), UInt8(bb), rb))
        }
        return decomp

    case .sub:
        var decomp = [(UInt8, UInt8, UInt8)]()
        var borrow: Int32 = 0
        for k in 0..<4 {
            let ab = Int32((a >> (k * 8)) & 0xFF)
            let bb = Int32((b >> (k * 8)) & 0xFF)
            var diff = ab - bb - borrow
            if diff < 0 {
                diff += 256
                borrow = 1
            } else {
                borrow = 0
            }
            decomp.append((UInt8(ab), UInt8(bb), UInt8(diff & 0xFF)))
        }
        return decomp

    case .mul, .eq, .lt, .shl, .shr:
        return nil
    }
}

// MARK: - Lookup Subtable Definitions

/// Byte-level subtable for a specific opcode.
public struct JoltSubtable {
    public let op: JoltOp
    public let entries: [Fr]
    public let size: Int

    /// Build subtable for a bitwise op (AND, OR, XOR) -- 256*256 = 65536 entries.
    public static func bitwise(_ op: JoltOp) -> JoltSubtable {
        precondition(op == .and_ || op == .or_ || op == .xor_)
        let size = 256 * 256
        var entries = [Fr]()
        entries.reserveCapacity(size)
        for a in 0..<256 {
            for b in 0..<256 {
                let r: UInt32
                switch op {
                case .and_: r = UInt32(a & b)
                case .or_:  r = UInt32(a | b)
                case .xor_: r = UInt32(a ^ b)
                default: fatalError()
                }
                entries.append(frFromInt(UInt64(r)))
            }
        }
        return JoltSubtable(op: op, entries: entries, size: size)
    }

    /// Build subtable for ADD with carry: 256*256*2 = 131072 entries.
    public static func addWithCarry() -> JoltSubtable {
        let size = 256 * 256 * 2
        var entries = [Fr]()
        entries.reserveCapacity(size)
        for carry in 0..<2 {
            for a in 0..<256 {
                for b in 0..<256 {
                    let sum = a + b + carry
                    entries.append(frFromInt(UInt64(sum & 0xFF)))
                }
            }
        }
        return JoltSubtable(op: .add, entries: entries, size: size)
    }

    /// Build subtable for SUB with borrow: 256*256*2 = 131072 entries.
    public static func subWithBorrow() -> JoltSubtable {
        let size = 256 * 256 * 2
        var entries = [Fr]()
        entries.reserveCapacity(size)
        for borrow in 0..<2 {
            for a in 0..<256 {
                for b in 0..<256 {
                    var diff = a - b - borrow
                    if diff < 0 { diff += 256 }
                    entries.append(frFromInt(UInt64(diff & 0xFF)))
                }
            }
        }
        return JoltSubtable(op: .sub, entries: entries, size: size)
    }
}

// MARK: - Program Generation

/// Generate a random program of `count` instructions using `numRegisters` registers.
public func joltRandomProgram(count: Int, numRegisters: Int = 32, seed: UInt64 = 0xDEAD_BEEF) -> [JoltInstruction] {
    var rng = seed
    func next() -> UInt64 {
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        return rng
    }

    let ops = JoltOp.allCases
    var program = [JoltInstruction]()
    program.reserveCapacity(count)

    for i in 0..<count {
        let op = ops[Int(next() >> 60) % ops.count]
        let rs1 = UInt32(Int(next() >> 48) % numRegisters)
        let rs2 = UInt32(Int(next() >> 48) % numRegisters)
        let rd = UInt32(i < numRegisters ? i % numRegisters : Int(next() >> 48) % numRegisters)
        program.append(JoltInstruction(op: op, rs1: rs1, rs2: rs2, rd: rd))
    }

    return program
}
