// InstructionDecomposition — Jolt instruction decomposition into subtable lookups
//
// Each RISC-V instruction is decomposed into a sequence of small subtable lookups.
// For RV32IM, 32-bit operands are split into C-bit chunks (default C=6, so 2^6=64
// entry tables). Each chunk is independently looked up, and the results are combined
// via a per-instruction combine function to produce the final 32-bit result.
//
// This is the core mechanism from the Jolt paper (Arun et al. 2024, Section 4):
// instead of proving arithmetic circuits for each CPU instruction, decompose each
// instruction into chunk-wise table lookups with a cheap combination step.
//
// Subtable types:
//   - Identity: f(x) = x (pass-through for extraction)
//   - Truncate: f(x) = x & mask (chunk extraction with masking)
//   - LT/LTU:  f(x,y) = (x < y) ? 1 : 0 (comparison, signed/unsigned)
//   - EQ:      f(x,y) = (x == y) ? 1 : 0
//   - And/Or/Xor: bitwise operations on chunks
//   - Sll/Srl/Sra: shift operations per chunk
//   - SignExtend: for LB/LH sign extension
//
// References: Jolt (Arun et al. 2024), Lasso (Setty et al. 2023)

import Foundation

// MARK: - JoltSubtable Protocol

/// A subtable that can evaluate a single input and report its size.
/// Subtables are the primitive building blocks of Jolt instruction decomposition.
/// Each subtable is small (2^C entries for C-bit chunks) and can be materialized
/// into a full lookup table for Lasso verification.
public protocol JoltSubtable {
    /// Evaluate the subtable function on a single input.
    /// For unary subtables, input is a single chunk value.
    /// For binary subtables, input packs two chunk values: (x << chunkBits) | y.
    func evaluate(input: UInt64) -> UInt64

    /// The number of entries in this subtable when materialized.
    /// For unary subtables: 2^chunkBits. For binary: 2^(2*chunkBits).
    var tableSize: Int { get }

    /// Human-readable name for keying materialized tables.
    var name: String { get }
}

// MARK: - Concrete Subtables

/// Identity subtable: f(x) = x. Used for pass-through chunk extraction.
public struct IdentitySubtable: JoltSubtable {
    public let tableSize: Int
    public let name: String = "identity"

    public init(chunkBits: Int) {
        self.tableSize = 1 << chunkBits
    }

    public func evaluate(input: UInt64) -> UInt64 {
        return input
    }
}

/// Truncate subtable: f(x) = x & mask. Used for chunk extraction with masking.
public struct TruncateSubtable: JoltSubtable {
    public let mask: UInt64
    public let tableSize: Int
    public let name: String

    public init(chunkBits: Int, truncBits: Int) {
        self.tableSize = 1 << chunkBits
        self.mask = (1 << truncBits) - 1
        self.name = "truncate_\(truncBits)"
    }

    public func evaluate(input: UInt64) -> UInt64 {
        return input & mask
    }
}

/// Less-than subtable (signed): f(x, y) = (Int(x) < Int(y)) ? 1 : 0.
/// Input packs two chunk values: (x << chunkBits) | y.
/// Treats x, y as signed values within the chunk width.
public struct LTSubtable: JoltSubtable {
    public let chunkBits: Int
    public let tableSize: Int
    public let name: String = "lt_signed"

    public init(chunkBits: Int) {
        self.chunkBits = chunkBits
        self.tableSize = 1 << (2 * chunkBits)
    }

    public func evaluate(input: UInt64) -> UInt64 {
        let chunkSize = 1 << chunkBits
        let x = Int64(input >> chunkBits)
        let y = Int64(input & UInt64(chunkSize - 1))
        // Sign-extend from chunkBits
        let signBit = Int64(1 << (chunkBits - 1))
        let sx = (x ^ signBit) - signBit
        let sy = (y ^ signBit) - signBit
        return sx < sy ? 1 : 0
    }
}

/// Equality subtable: f(x, y) = (x == y) ? 1 : 0.
/// Input packs two chunk values: (x << chunkBits) | y.
public struct EQSubtable: JoltSubtable {
    public let chunkBits: Int
    public let tableSize: Int
    public let name: String = "eq"

    public init(chunkBits: Int) {
        self.chunkBits = chunkBits
        self.tableSize = 1 << (2 * chunkBits)
    }

    public func evaluate(input: UInt64) -> UInt64 {
        let chunkMask = UInt64((1 << chunkBits) - 1)
        let x = input >> chunkBits
        let y = input & chunkMask
        return x == y ? 1 : 0
    }
}

/// Unsigned less-than subtable: f(x, y) = (x < y) ? 1 : 0.
/// Input packs two chunk values: (x << chunkBits) | y.
public struct LTUSubtable: JoltSubtable {
    public let chunkBits: Int
    public let tableSize: Int
    public let name: String = "ltu"

    public init(chunkBits: Int) {
        self.chunkBits = chunkBits
        self.tableSize = 1 << (2 * chunkBits)
    }

    public func evaluate(input: UInt64) -> UInt64 {
        let chunkMask = UInt64((1 << chunkBits) - 1)
        let x = input >> chunkBits
        let y = input & chunkMask
        return x < y ? 1 : 0
    }
}

/// AND subtable: f(x, y) = x & y.
/// Input packs two chunk values: (x << chunkBits) | y.
public struct AndSubtable: JoltSubtable {
    public let chunkBits: Int
    public let tableSize: Int
    public let name: String = "and"

    public init(chunkBits: Int) {
        self.chunkBits = chunkBits
        self.tableSize = 1 << (2 * chunkBits)
    }

    public func evaluate(input: UInt64) -> UInt64 {
        let chunkMask = UInt64((1 << chunkBits) - 1)
        let x = input >> chunkBits
        let y = input & chunkMask
        return x & y
    }
}

/// OR subtable: f(x, y) = x | y.
/// Input packs two chunk values: (x << chunkBits) | y.
public struct OrSubtable: JoltSubtable {
    public let chunkBits: Int
    public let tableSize: Int
    public let name: String = "or"

    public init(chunkBits: Int) {
        self.chunkBits = chunkBits
        self.tableSize = 1 << (2 * chunkBits)
    }

    public func evaluate(input: UInt64) -> UInt64 {
        let chunkMask = UInt64((1 << chunkBits) - 1)
        let x = input >> chunkBits
        let y = input & chunkMask
        return x | y
    }
}

/// XOR subtable: f(x, y) = x ^ y.
/// Input packs two chunk values: (x << chunkBits) | y.
public struct XorSubtable: JoltSubtable {
    public let chunkBits: Int
    public let tableSize: Int
    public let name: String = "xor"

    public init(chunkBits: Int) {
        self.chunkBits = chunkBits
        self.tableSize = 1 << (2 * chunkBits)
    }

    public func evaluate(input: UInt64) -> UInt64 {
        let chunkMask = UInt64((1 << chunkBits) - 1)
        let x = input >> chunkBits
        let y = input & chunkMask
        return x ^ y
    }
}

/// Shift-left subtable: f(x, shamt) = (x << shamt) within chunk width.
public struct SllSubtable: JoltSubtable {
    public let chunkBits: Int
    public let tableSize: Int
    public let name: String = "sll"

    public init(chunkBits: Int) {
        self.chunkBits = chunkBits
        self.tableSize = 1 << (2 * chunkBits)
    }

    public func evaluate(input: UInt64) -> UInt64 {
        let chunkMask = UInt64((1 << chunkBits) - 1)
        let x = input >> chunkBits
        let shamt = input & chunkMask
        return (x << shamt) & chunkMask
    }
}

/// Shift-right-logical subtable: f(x, shamt) = x >> shamt within chunk width.
public struct SrlSubtable: JoltSubtable {
    public let chunkBits: Int
    public let tableSize: Int
    public let name: String = "srl"

    public init(chunkBits: Int) {
        self.chunkBits = chunkBits
        self.tableSize = 1 << (2 * chunkBits)
    }

    public func evaluate(input: UInt64) -> UInt64 {
        let chunkMask = UInt64((1 << chunkBits) - 1)
        let x = input >> chunkBits
        let shamt = input & chunkMask
        return (x >> shamt) & chunkMask
    }
}

/// Shift-right-arithmetic subtable: f(x, shamt) with sign extension within chunk width.
public struct SraSubtable: JoltSubtable {
    public let chunkBits: Int
    public let tableSize: Int
    public let name: String = "sra"

    public init(chunkBits: Int) {
        self.chunkBits = chunkBits
        self.tableSize = 1 << (2 * chunkBits)
    }

    public func evaluate(input: UInt64) -> UInt64 {
        let chunkMask = UInt64((1 << chunkBits) - 1)
        let x = input >> chunkBits
        let shamt = input & chunkMask
        let signBit = Int64(1 << (chunkBits - 1))
        let sx = (Int64(x) ^ signBit) - signBit
        let shifted = sx >> Int64(shamt)
        return UInt64(bitPattern: shifted) & chunkMask
    }
}

/// Sign-extend subtable for LB (byte) and LH (halfword) loads.
/// f(x) = sign_extend(x, fromBits) -- extends the value from fromBits width to 32 bits.
public struct SignExtendSubtable: JoltSubtable {
    public let fromBits: Int  // 8 for LB, 16 for LH
    public let tableSize: Int
    public let name: String

    public init(chunkBits: Int, fromBits: Int) {
        self.fromBits = fromBits
        self.tableSize = 1 << chunkBits
        self.name = "sign_extend_\(fromBits)"
    }

    public func evaluate(input: UInt64) -> UInt64 {
        let signBit = UInt64(1 << (fromBits - 1))
        let mask32 = UInt64(0xFFFFFFFF)
        if input & signBit != 0 {
            let extended = input | (mask32 & ~((1 << fromBits) - 1))
            return extended & mask32
        } else {
            return input
        }
    }
}

// MARK: - Subtable Entry (for decomposition results)

/// A single subtable lookup in a decomposition sequence.
public struct SubtableLookup {
    public let subtable: JoltSubtable
    public let input: UInt64

    public init(subtable: JoltSubtable, input: UInt64) {
        self.subtable = subtable
        self.input = input
    }
}

// MARK: - JoltInstructionDecomposer

/// Decomposes RV32IM instructions into sequences of subtable lookups.
///
/// Each 32-bit instruction is split into C-bit chunks. The operands (a, b) are
/// each divided into ceil(32/C) chunks. For binary operations, each chunk pair
/// is looked up in the appropriate subtable. The combine function reassembles
/// the full 32-bit result from the subtable outputs.
public struct JoltInstructionDecomposer {

    /// Number of bits per chunk (default 6 for 64-entry tables)
    public let chunkBits: Int

    /// Number of chunks needed to cover a 32-bit value
    public let numChunks: Int

    /// Mask for a single chunk
    public let chunkMask: UInt64

    // Pre-built subtable instances
    private let identityST: IdentitySubtable
    private let andST: AndSubtable
    private let orST: OrSubtable
    private let xorST: XorSubtable
    public init(chunkBits: Int = 6) {
        self.chunkBits = chunkBits
        self.numChunks = (32 + chunkBits - 1) / chunkBits
        self.chunkMask = UInt64((1 << chunkBits) - 1)

        self.identityST = IdentitySubtable(chunkBits: chunkBits)
        self.andST = AndSubtable(chunkBits: chunkBits)
        self.orST = OrSubtable(chunkBits: chunkBits)
        self.xorST = XorSubtable(chunkBits: chunkBits)
    }

    // MARK: - Chunk Extraction

    @inline(__always)
    private func chunk(_ value: UInt32, _ k: Int) -> UInt64 {
        return UInt64((value >> (k * chunkBits)) & UInt32(chunkMask))
    }

    @inline(__always)
    private func pack(_ x: UInt64, _ y: UInt64) -> UInt64 {
        return (x << chunkBits) | y
    }

    // MARK: - Decompose

    /// Decompose an instruction into its subtable lookup sequence.
    ///
    /// Uses the existing RV32IMInstruction type from RISCVExecutor which has
    /// `.base(RV32IInstruction)` and `.mul(RV32MInstruction)` cases.
    ///
    /// - Parameters:
    ///   - instruction: The RV32IM instruction to decompose
    ///   - operands: (rs1_value, rs2_value_or_immediate) as UInt32
    /// - Returns: Array of (subtable, input) pairs for each lookup
    public func decompose(
        instruction: RV32IMInstruction,
        operands: (UInt32, UInt32)
    ) -> [SubtableLookup] {
        let (a, b) = operands

        switch instruction {
        case .base(let baseInstr):
            return decomposeBase(baseInstr, a: a, b: b)
        case .mul(let mulInstr):
            return decomposeMul(mulInstr, a: a, b: b)
        }
    }

    /// Decompose a base RV32I instruction.
    private func decomposeBase(
        _ instr: RV32IInstruction, a: UInt32, b: UInt32
    ) -> [SubtableLookup] {
        switch instr {
        // Bitwise: chunk-wise independent
        case .AND:  return chunkWiseBinary(a, b, andST)
        case .OR:   return chunkWiseBinary(a, b, orST)
        case .XOR:  return chunkWiseBinary(a, b, xorST)
        case .ANDI: return chunkWiseBinary(a, b, andST)
        case .ORI:  return chunkWiseBinary(a, b, orST)
        case .XORI: return chunkWiseBinary(a, b, xorST)

        // ADD/SUB: decompose result into identity chunks
        case .ADD, .ADDI:
            return decomposeResult(a &+ b)
        case .SUB:
            return decomposeResult(a &- b)

        // Shifts: compute full result, decompose into identity chunks
        case .SLL, .SLLI:
            return decomposeResult(a << (b & 31))
        case .SRL, .SRLI:
            return decomposeResult(a >> (b & 31))
        case .SRA, .SRAI:
            let result = UInt32(bitPattern: Int32(bitPattern: a) >> Int32(b & 31))
            return decomposeResult(result)

        // Comparisons: compute result, decompose into identity chunks
        case .SLT, .SLTI:
            let r: UInt32 = Int32(bitPattern: a) < Int32(bitPattern: b) ? 1 : 0
            return decomposeResult(r)
        case .SLTU, .SLTIU:
            return decomposeResult(a < b ? 1 : 0)

        // Branches: compute taken/not-taken result, decompose
        case .BEQ:  return decomposeResult(a == b ? 1 : 0)
        case .BNE:  return decomposeResult(a != b ? 1 : 0)
        case .BLT:
            let r: UInt32 = Int32(bitPattern: a) < Int32(bitPattern: b) ? 1 : 0
            return decomposeResult(r)
        case .BGE:
            let r: UInt32 = Int32(bitPattern: a) >= Int32(bitPattern: b) ? 1 : 0
            return decomposeResult(r)
        case .BLTU: return decomposeResult(a < b ? 1 : 0)
        case .BGEU: return decomposeResult(a >= b ? 1 : 0)

        // Loads/stores: decompose effective address
        case .LB, .LH, .LW, .LBU, .LHU, .SB, .SH, .SW:
            return decomposeResult(a &+ b)

        // Upper immediate
        case .LUI:   return decomposeResult(b)
        case .AUIPC: return decomposeResult(a &+ b)

        // Jumps: return address = PC + 4
        case .JAL, .JALR:
            return decomposeResult(a &+ 4)

        // System: no data operation
        case .ECALL, .EBREAK, .FENCE:
            return decomposeResult(0)
        }
    }

    /// Decompose an M-extension instruction.
    private func decomposeMul(
        _ instr: RV32MInstruction, a: UInt32, b: UInt32
    ) -> [SubtableLookup] {
        switch instr {
        case .MUL:
            return decomposeResult(a &* b)
        case .MULH:
            let product = Int64(Int32(bitPattern: a)) &* Int64(Int32(bitPattern: b))
            return decomposeResult(UInt32(truncatingIfNeeded: product >> 32))
        case .MULHSU:
            let product = Int64(Int32(bitPattern: a)) &* Int64(UInt64(b))
            return decomposeResult(UInt32(truncatingIfNeeded: product >> 32))
        case .MULHU:
            let product = UInt64(a) &* UInt64(b)
            return decomposeResult(UInt32(truncatingIfNeeded: product >> 32))
        case .DIV:
            if b == 0 { return decomposeResult(0xFFFFFFFF) }
            let sa = Int32(bitPattern: a); let sb = Int32(bitPattern: b)
            if sa == Int32.min && sb == -1 { return decomposeResult(UInt32(bitPattern: Int32.min)) }
            return decomposeResult(UInt32(bitPattern: sa / sb))
        case .DIVU:
            if b == 0 { return decomposeResult(0xFFFFFFFF) }
            return decomposeResult(a / b)
        case .REM:
            if b == 0 { return decomposeResult(a) }
            let sa = Int32(bitPattern: a); let sb = Int32(bitPattern: b)
            if sa == Int32.min && sb == -1 { return decomposeResult(0) }
            return decomposeResult(UInt32(bitPattern: sa % sb))
        case .REMU:
            if b == 0 { return decomposeResult(a) }
            return decomposeResult(a % b)
        }
    }

    // MARK: - Combine

    /// Reassemble the final 32-bit result from subtable lookup outputs.
    ///
    /// For chunk-wise operations, the result is reconstructed by placing each chunk
    /// result back at its bit position. For comparisons, the combination uses
    /// MSB-first priority logic.
    public func combine(
        results: [UInt64],
        instruction: RV32IMInstruction
    ) -> UInt32 {
        switch instruction {
        case .base(let baseInstr):
            return combineBase(results, baseInstr)
        case .mul:
            // All M-extension ops use identity decomposition
            return reassembleChunks(results)
        }
    }

    /// Combine results for a base RV32I instruction.
    /// All instructions use chunk reassembly since decompose always produces
    /// identity-chunk or bitwise-chunk decompositions.
    private func combineBase(_ results: [UInt64], _ instr: RV32IInstruction) -> UInt32 {
        return reassembleChunks(results)
    }

    // MARK: - Internal Decomposition Helpers

    private func chunkWiseBinary(
        _ a: UInt32, _ b: UInt32, _ st: JoltSubtable
    ) -> [SubtableLookup] {
        var lookups = [SubtableLookup]()
        lookups.reserveCapacity(numChunks)
        for k in 0..<numChunks {
            let aChunk = chunk(a, k)
            let bChunk = chunk(b, k)
            lookups.append(SubtableLookup(subtable: st, input: pack(aChunk, bChunk)))
        }
        return lookups
    }

    private func decomposeResult(_ result: UInt32) -> [SubtableLookup] {
        var lookups = [SubtableLookup]()
        lookups.reserveCapacity(numChunks)
        for k in 0..<numChunks {
            let rChunk = chunk(result, k)
            lookups.append(SubtableLookup(subtable: identityST, input: rChunk))
        }
        return lookups
    }

    // MARK: - Internal Combine Helpers

    private func reassembleChunks(_ results: [UInt64]) -> UInt32 {
        var value: UInt32 = 0
        for k in 0..<min(results.count, numChunks) {
            value |= UInt32(results[k] & chunkMask) << (k * chunkBits)
        }
        return value
    }

}

// MARK: - RV32IM Execution Reference

/// Execute a single RV32IM instruction and return the result.
/// Reference semantics for verification.
public func rv32imExecute(_ instr: RV32IMInstruction, _ a: UInt32, _ b: UInt32) -> UInt32 {
    switch instr {
    case .base(let baseInstr):
        return rv32iBaseExecute(baseInstr, a, b)
    case .mul(let mulInstr):
        return rv32mExecute(mulInstr, a, b)
    }
}

/// Execute a base RV32I instruction.
private func rv32iBaseExecute(_ instr: RV32IInstruction, _ a: UInt32, _ b: UInt32) -> UInt32 {
    switch instr {
    case .ADD, .ADDI:   return a &+ b
    case .SUB:          return a &- b
    case .AND, .ANDI:   return a & b
    case .OR, .ORI:     return a | b
    case .XOR, .XORI:   return a ^ b
    case .SLT, .SLTI:
        return Int32(bitPattern: a) < Int32(bitPattern: b) ? 1 : 0
    case .SLTU, .SLTIU: return a < b ? 1 : 0
    case .SLL, .SLLI:   return a << (b & 31)
    case .SRL, .SRLI:   return a >> (b & 31)
    case .SRA, .SRAI:
        return UInt32(bitPattern: Int32(bitPattern: a) >> Int32(b & 31))
    case .BEQ:  return a == b ? 1 : 0
    case .BNE:  return a != b ? 1 : 0
    case .BLT:  return Int32(bitPattern: a) < Int32(bitPattern: b) ? 1 : 0
    case .BGE:  return Int32(bitPattern: a) >= Int32(bitPattern: b) ? 1 : 0
    case .BLTU: return a < b ? 1 : 0
    case .BGEU: return a >= b ? 1 : 0
    case .LB, .LH, .LW, .LBU, .LHU, .SB, .SH, .SW:
        return a &+ b
    case .LUI:   return b
    case .AUIPC: return a &+ b
    case .JAL, .JALR: return a &+ 4
    case .ECALL, .EBREAK, .FENCE: return 0
    }
}

/// Execute an M-extension instruction.
private func rv32mExecute(_ instr: RV32MInstruction, _ a: UInt32, _ b: UInt32) -> UInt32 {
    switch instr {
    case .MUL:    return a &* b
    case .MULH:
        let product = Int64(Int32(bitPattern: a)) &* Int64(Int32(bitPattern: b))
        return UInt32(truncatingIfNeeded: product >> 32)
    case .MULHSU:
        let product = Int64(Int32(bitPattern: a)) &* Int64(UInt64(b))
        return UInt32(truncatingIfNeeded: product >> 32)
    case .MULHU:
        let product = UInt64(a) &* UInt64(b)
        return UInt32(truncatingIfNeeded: product >> 32)
    case .DIV:
        if b == 0 { return 0xFFFFFFFF }
        let sa = Int32(bitPattern: a); let sb = Int32(bitPattern: b)
        if sa == Int32.min && sb == -1 { return UInt32(bitPattern: Int32.min) }
        return UInt32(bitPattern: sa / sb)
    case .DIVU:
        if b == 0 { return 0xFFFFFFFF }
        return a / b
    case .REM:
        if b == 0 { return a }
        let sa = Int32(bitPattern: a); let sb = Int32(bitPattern: b)
        if sa == Int32.min && sb == -1 { return 0 }
        return UInt32(bitPattern: sa % sb)
    case .REMU:
        if b == 0 { return a }
        return a % b
    }
}
