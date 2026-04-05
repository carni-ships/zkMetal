// InstructionTable — Jolt-style decomposed instruction tables for Lasso lookup verification
//
// Each 32-bit ALU instruction is verified via byte-level subtable lookups:
//   - Bitwise ops (AND, OR, XOR): fully decomposable, 4 independent 8-bit subtables
//   - Arithmetic ops (ADD, SUB): decomposable with carry chain
//   - Shifts (SLL, SRL): lookup into 16-bit tables (byte, shift_amount) -> result_byte
//   - Comparison (SLT): decomposable with borrow chain
//
// Each subtable maps (a_byte, b_byte) -> result_byte (or carries additional state).
// The Lasso tensor structure means the prover only commits to small subtables (256 or 65536
// entries), not the full 2^32 * 2^32 instruction table.
//
// References: Jolt (Arun et al. 2024), Section 4: "Instruction lookups via Lasso"

import Foundation

// MARK: - Instruction Table Protocol

/// A Jolt instruction table: given (a, b) produces result, decomposed for Lasso.
/// Each table produces a LassoTable that can be fed to LassoEngine.prove().
public protocol JoltInstructionTable {
    /// The opcode this table handles
    var op: JoltOp { get }

    /// Number of byte chunks (typically 4 for 32-bit)
    var numChunks: Int { get }

    /// Build the LassoTable for this instruction.
    /// Lookup values encode (a_chunk, b_chunk) per byte position.
    func buildLassoTable() -> LassoTable

    /// Encode (a, b, result) into lookup values for Lasso.
    /// Returns one Fr per lookup (encoding the combined operand+result chunk indices).
    func encodeLookups(a: UInt32, b: UInt32, result: UInt32) -> [Fr]
}

// MARK: - Bitwise Tables (AND, OR, XOR) — Fully Decomposable

/// Bitwise instruction table: each byte of the result is computed independently.
/// Subtable[i * 256 + j] = op(i, j) for i, j in [0, 256).
/// 4 chunks, 65536 entries each, fully decomposable.
public struct BitwiseInstructionTable: JoltInstructionTable {
    public let op: JoltOp
    private let bitwiseOp: (UInt8, UInt8) -> UInt8

    public var numChunks: Int { 4 }

    public init(op: JoltOp) {
        self.op = op
        switch op {
        case .and_: self.bitwiseOp = { $0 & $1 }
        case .or_:  self.bitwiseOp = { $0 | $1 }
        case .xor_: self.bitwiseOp = { $0 ^ $1 }
        default: fatalError("BitwiseInstructionTable only supports AND, OR, XOR")
        }
    }

    public func buildLassoTable() -> LassoTable {
        let chunkRange = 256
        let tableSize = chunkRange * chunkRange  // 65536

        // Each subtable: entry[a * 256 + b] = op(a, b)
        let subtable: [Fr] = (0..<tableSize).map { idx in
            let a = UInt8(idx >> 8)
            let b = UInt8(idx & 0xFF)
            return frFromInt(UInt64(bitwiseOp(a, b)))
        }
        let subtables = Array(repeating: subtable, count: 4)

        // Compose: reconstruct 32-bit result from 4 byte results
        let compose: ([Fr]) -> Fr = { components in
            var result = Fr.zero
            var shift = Fr.one
            let base = frFromInt(256)
            for c in components {
                result = frAdd(result, frMul(c, shift))
                shift = frMul(shift, base)
            }
            return result
        }

        // Decompose: given a packed lookup value (a_chunk * 256 + b_chunk per chunk),
        // extract the 4 subtable indices
        let decompose: (Fr) -> [Int] = { value in
            let limbs = frToInt(value)
            let v = limbs[0]
            var indices = [Int]()
            indices.reserveCapacity(4)
            for k in 0..<4 {
                let shift = k * 16  // 16 bits per packed (a_byte, b_byte) pair
                let idx = Int((v >> shift) & 0xFFFF)
                indices.append(idx)
            }
            return indices
        }

        return LassoTable(subtables: subtables, compose: compose,
                          decompose: decompose, numChunks: 4,
                          batchDecompose: nil)
    }

    public func encodeLookups(a: UInt32, b: UInt32, result: UInt32) -> [Fr] {
        // Pack (a_byte, b_byte) into subtable index for each chunk
        // Then combine all 4 chunks into a single packed lookup value
        var packed: UInt64 = 0
        for k in 0..<4 {
            let aByte = UInt64((a >> (k * 8)) & 0xFF)
            let bByte = UInt64((b >> (k * 8)) & 0xFF)
            let idx = (aByte << 8) | bByte  // subtable index
            packed |= (idx << (k * 16))
        }
        return [frFromInt(packed)]
    }
}

// MARK: - Addition Table — Decomposable with Carry Chain

/// ADD instruction table: 32-bit addition decomposed into 4 byte additions with carry.
/// Each subtable entry: (a_byte, b_byte, carry_in) -> (result_byte, carry_out)
/// We use a 512-entry subtable (256 * 2 for carry_in = 0 or 1).
/// Subtable index = a_byte * 512 + b_byte * 2 + carry_in, value = result_byte.
/// Carry propagation is verified by chaining: carry_out[k] = carry_in[k+1].
public struct AddInstructionTable: JoltInstructionTable {
    public let op: JoltOp = .add
    public var numChunks: Int { 4 }

    public init() {}

    public func buildLassoTable() -> LassoTable {
        // Subtable: 256 * 256 * 2 = 131072 entries
        // Index = a * 512 + b * 2 + carry_in
        // Value = (a + b + carry_in) & 0xFF (result byte)
        let tableSize = 256 * 256 * 2
        let subtable: [Fr] = (0..<tableSize).map { idx in
            let a = idx >> 9          // bits [17:9]
            let b = (idx >> 1) & 0xFF // bits [8:1]
            let cin = idx & 1         // bit 0
            let sum = a + b + cin
            return frFromInt(UInt64(sum & 0xFF))
        }
        let subtables = Array(repeating: subtable, count: 4)

        let compose: ([Fr]) -> Fr = { components in
            var result = Fr.zero
            var shift = Fr.one
            let base = frFromInt(256)
            for c in components {
                result = frAdd(result, frMul(c, shift))
                shift = frMul(shift, base)
            }
            return result
        }

        let decompose: (Fr) -> [Int] = { value in
            let limbs = frToInt(value)
            let v = limbs[0]
            var indices = [Int]()
            indices.reserveCapacity(4)
            for k in 0..<4 {
                let shift = k * 18  // 18 bits per packed (a_byte, b_byte, carry_in)
                let idx = Int((v >> shift) & 0x3FFFF)
                indices.append(min(idx, tableSize - 1))
            }
            return indices
        }

        return LassoTable(subtables: subtables, compose: compose,
                          decompose: decompose, numChunks: 4,
                          batchDecompose: nil)
    }

    public func encodeLookups(a: UInt32, b: UInt32, result: UInt32) -> [Fr] {
        // Compute carry chain and pack indices
        var carry: UInt32 = 0
        var packed: UInt64 = 0
        for k in 0..<4 {
            let aByte = (a >> (k * 8)) & 0xFF
            let bByte = (b >> (k * 8)) & 0xFF
            let idx = UInt64(aByte) * 512 + UInt64(bByte) * 2 + UInt64(carry)
            packed |= (idx << (k * 18))
            let sum = aByte + bByte + carry
            carry = sum >> 8
        }
        return [frFromInt(packed)]
    }
}

// MARK: - Subtraction Table — Decomposable with Borrow Chain

/// SUB instruction table: 32-bit subtraction decomposed into 4 byte subtractions with borrow.
/// Subtable index = a_byte * 512 + b_byte * 2 + borrow_in
/// Value = (a_byte - b_byte - borrow_in) & 0xFF
public struct SubInstructionTable: JoltInstructionTable {
    public let op: JoltOp = .sub
    public var numChunks: Int { 4 }

    public init() {}

    public func buildLassoTable() -> LassoTable {
        let tableSize = 256 * 256 * 2
        let subtable: [Fr] = (0..<tableSize).map { idx in
            let a = idx >> 9
            let b = (idx >> 1) & 0xFF
            let bin = idx & 1
            let diff = Int(a) - Int(b) - Int(bin)
            return frFromInt(UInt64(diff & 0xFF))
        }
        let subtables = Array(repeating: subtable, count: 4)

        let compose: ([Fr]) -> Fr = { components in
            var result = Fr.zero
            var shift = Fr.one
            let base = frFromInt(256)
            for c in components {
                result = frAdd(result, frMul(c, shift))
                shift = frMul(shift, base)
            }
            return result
        }

        let decompose: (Fr) -> [Int] = { value in
            let limbs = frToInt(value)
            let v = limbs[0]
            var indices = [Int]()
            indices.reserveCapacity(4)
            for k in 0..<4 {
                let shift = k * 18
                let idx = Int((v >> shift) & 0x3FFFF)
                indices.append(min(idx, tableSize - 1))
            }
            return indices
        }

        return LassoTable(subtables: subtables, compose: compose,
                          decompose: decompose, numChunks: 4,
                          batchDecompose: nil)
    }

    public func encodeLookups(a: UInt32, b: UInt32, result: UInt32) -> [Fr] {
        var borrow: UInt32 = 0
        var packed: UInt64 = 0
        for k in 0..<4 {
            let aByte = (a >> (k * 8)) & 0xFF
            let bByte = (b >> (k * 8)) & 0xFF
            let idx = UInt64(aByte) * 512 + UInt64(bByte) * 2 + UInt64(borrow)
            packed |= (idx << (k * 18))
            let diff = Int(aByte) - Int(bByte) - Int(borrow)
            borrow = diff < 0 ? 1 : 0
        }
        return [frFromInt(packed)]
    }
}

// MARK: - Shift Tables (SLL, SRL)

/// Shift instruction table: shift amount is at most 31, so we use a direct
/// (byte_value, shift_amount) subtable per byte position.
/// For SLL byte k: subtable[(val << 8) | shamt] = (val << shamt >> (k*8)) & 0xFF
/// We use a simpler approach: one lookup per instruction with a 2^16 subtable
/// indexed by (byte_val, shift_amount_5bit).
public struct ShiftInstructionTable: JoltInstructionTable {
    public let op: JoltOp
    public var numChunks: Int { 4 }

    public init(op: JoltOp) {
        precondition(op == .shl || op == .shr, "ShiftInstructionTable only supports SHL, SHR")
        self.op = op
    }

    public func buildLassoTable() -> LassoTable {
        // For shifts, we use a per-byte subtable that takes (byte_value, shift_amount)
        // and returns the contribution of that byte to the output at that byte position.
        // This requires knowing the byte position, so each chunk gets its own subtable.
        //
        // Subtable for chunk k: indexed by (input_byte * 32 + shift_amount)
        // Value = byte k of (input_byte << (8*source_position) << shift_amount) or
        //         byte k of (input_byte << (8*source_position) >> shift_amount)
        //
        // Simplified: we use 256 * 32 = 8192 entries per chunk.
        // For each (byte_val, shamt), the subtable stores the result byte at this position.

        let tableSize = 256 * 32  // byte_val * 32 + shamt
        var subtables = [[Fr]]()
        subtables.reserveCapacity(4)

        for k in 0..<4 {
            let subtable: [Fr] = (0..<tableSize).map { idx in
                let byteVal = UInt32(idx >> 5)
                let shamt = UInt32(idx & 0x1F)
                // This byte's contribution to the full shift
                let fullVal = byteVal << (k * 8)
                let shifted: UInt32
                if self.op == .shl {
                    shifted = fullVal << shamt
                } else {
                    shifted = fullVal >> shamt
                }
                // Extract the byte at position k from the shifted result
                // But we need all 4 bytes of shifted, so we store byte k
                let resultByte = (shifted >> (k * 8)) & 0xFF
                return frFromInt(UInt64(resultByte))
            }
            subtables.append(subtable)
        }

        let compose: ([Fr]) -> Fr = { components in
            var result = Fr.zero
            var shift = Fr.one
            let base = frFromInt(256)
            for c in components {
                result = frAdd(result, frMul(c, shift))
                shift = frMul(shift, base)
            }
            return result
        }

        let decompose: (Fr) -> [Int] = { value in
            let limbs = frToInt(value)
            let v = limbs[0]
            var indices = [Int]()
            indices.reserveCapacity(4)
            for k in 0..<4 {
                let shift = k * 13  // 13 bits per (byte_val:8, shamt:5)
                let idx = Int((v >> shift) & 0x1FFF)
                indices.append(min(idx, tableSize - 1))
            }
            return indices
        }

        return LassoTable(subtables: subtables, compose: compose,
                          decompose: decompose, numChunks: 4,
                          batchDecompose: nil)
    }

    public func encodeLookups(a: UInt32, b: UInt32, result: UInt32) -> [Fr] {
        let shamt = b & 31
        var packed: UInt64 = 0
        for k in 0..<4 {
            let byteVal = UInt64((a >> (k * 8)) & 0xFF)
            let idx = byteVal * 32 + UInt64(shamt)
            packed |= (idx << (k * 13))
        }
        return [frFromInt(packed)]
    }
}

// MARK: - SLT (Set Less Than) Table

/// SLT instruction table: a < b ? 1 : 0
/// Decomposed with a comparison chain from MSB to LSB.
/// Each subtable: (a_byte, b_byte, prev_decided, prev_result) ->
///   result bit if decided at this byte, else propagate.
/// Simplified: 256 * 256 * 2 = 131072 entries, value encodes (lt, eq) pair.
public struct SltInstructionTable: JoltInstructionTable {
    public let op: JoltOp = .lt
    public var numChunks: Int { 4 }

    public init() {}

    public func buildLassoTable() -> LassoTable {
        // Comparison subtable: indexed by (a_byte * 512 + b_byte * 2 + undecided_flag)
        // Value encodes: if a_byte < b_byte -> 1, if a_byte > b_byte -> 0,
        //                if a_byte == b_byte -> propagate (undecided_flag of next chunk)
        // We encode: value = lt_flag (0 or 1)
        let tableSize = 256 * 256 * 2
        let subtable: [Fr] = (0..<tableSize).map { idx in
            let a = idx >> 9
            let b = (idx >> 1) & 0xFF
            let undecided = idx & 1
            if undecided == 0 {
                // Already decided by a more-significant byte
                return frFromInt(0)
            }
            // Undecided: this byte decides or propagates
            if a < b { return frFromInt(1) }
            if a > b { return frFromInt(0) }
            // a == b: still undecided, propagate
            return frFromInt(0)
        }
        let subtables = Array(repeating: subtable, count: 4)

        let compose: ([Fr]) -> Fr = { components in
            // For SLT, we just need the final result (0 or 1)
            // The MSB chunk's result is the answer
            return components.last ?? Fr.zero
        }

        let decompose: (Fr) -> [Int] = { value in
            let limbs = frToInt(value)
            let v = limbs[0]
            var indices = [Int]()
            indices.reserveCapacity(4)
            for k in 0..<4 {
                let shift = k * 18
                let idx = Int((v >> shift) & 0x3FFFF)
                indices.append(min(idx, tableSize - 1))
            }
            return indices
        }

        return LassoTable(subtables: subtables, compose: compose,
                          decompose: decompose, numChunks: 4,
                          batchDecompose: nil)
    }

    public func encodeLookups(a: UInt32, b: UInt32, result: UInt32) -> [Fr] {
        // Process from MSB (byte 3) to LSB (byte 0)
        // Track whether comparison is still undecided
        var undecided: UInt32 = 1
        var packed: UInt64 = 0
        for k in stride(from: 3, through: 0, by: -1) {
            let aByte = (a >> (k * 8)) & 0xFF
            let bByte = (b >> (k * 8)) & 0xFF
            let idx = UInt64(aByte) * 512 + UInt64(bByte) * 2 + UInt64(undecided)
            packed |= (idx << (k * 18))
            if undecided == 1 && aByte != bByte {
                undecided = 0  // Decided at this byte
            }
        }
        return [frFromInt(packed)]
    }
}

// MARK: - Instruction Table Registry

/// Registry of all instruction tables, one per supported opcode.
/// Provides the mapping from JoltOp -> JoltInstructionTable.
public struct InstructionTableRegistry {
    public let tables: [JoltOp: any JoltInstructionTable]

    /// Build registry with all supported ALU instruction tables.
    public init() {
        var t = [JoltOp: any JoltInstructionTable]()
        t[.and_] = BitwiseInstructionTable(op: .and_)
        t[.or_]  = BitwiseInstructionTable(op: .or_)
        t[.xor_] = BitwiseInstructionTable(op: .xor_)
        t[.add]  = AddInstructionTable()
        t[.sub]  = SubInstructionTable()
        t[.shl]  = ShiftInstructionTable(op: .shl)
        t[.shr]  = ShiftInstructionTable(op: .shr)
        t[.lt]   = SltInstructionTable()
        self.tables = t
    }

    /// Get the instruction table for an opcode, or nil if not supported via Lasso lookup.
    public func table(for op: JoltOp) -> (any JoltInstructionTable)? {
        return tables[op]
    }

    /// All opcodes that have Lasso lookup tables.
    public var supportedOps: [JoltOp] {
        return Array(tables.keys).sorted { $0.rawValue < $1.rawValue }
    }

    /// Build a LassoTable for a given opcode.
    public func lassoTable(for op: JoltOp) -> LassoTable? {
        return tables[op]?.buildLassoTable()
    }
}
