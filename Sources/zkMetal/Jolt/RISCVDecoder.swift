// RISCVDecoder — Binary RISC-V RV32I instruction decoder
//
// Parses 32-bit RISC-V instruction words into structured DecodedInstruction values.
// Supports all six RV32I instruction formats: R, I, S, B, U, J.
//
// The decoder extracts opcode, register fields (rd, rs1, rs2), function codes
// (funct3, funct7), and sign-extended immediates per the RISC-V specification.
//
// Reference: RISC-V Unprivileged ISA Specification, Volume I, Version 20191213

import Foundation

// MARK: - RV32I Instruction Enum

/// All 40 RV32I base integer instructions, identified by their unique
/// (opcode, funct3, funct7) combination in the binary encoding.
public enum RV32IInstruction: UInt8, CaseIterable {
    // R-type (opcode 0110011)
    case ADD = 0
    case SUB = 1
    case SLL = 2
    case SLT = 3
    case SLTU = 4
    case XOR = 5
    case SRL = 6
    case SRA = 7
    case OR = 8
    case AND = 9

    // I-type ALU (opcode 0010011)
    case ADDI = 10
    case SLTI = 11
    case SLTIU = 12
    case XORI = 13
    case ORI = 14
    case ANDI = 15
    case SLLI = 16
    case SRLI = 17
    case SRAI = 18

    // I-type Load (opcode 0000011)
    case LB = 19
    case LH = 20
    case LW = 21
    case LBU = 22
    case LHU = 23

    // S-type Store (opcode 0100011)
    case SB = 24
    case SH = 25
    case SW = 26

    // B-type Branch (opcode 1100011)
    case BEQ = 27
    case BNE = 28
    case BLT = 29
    case BGE = 30
    case BLTU = 31
    case BGEU = 32

    // J-type (opcode 1101111)
    case JAL = 33

    // I-type Jump (opcode 1100111)
    case JALR = 34

    // U-type (opcode 0110111)
    case LUI = 35

    // U-type (opcode 0010111)
    case AUIPC = 36

    // System (opcode 1110011)
    case ECALL = 37
    case EBREAK = 38

    // Fence (opcode 0001111)
    case FENCE = 39

    /// The corresponding RV32IOp for use with the Jolt instruction table
    public var rv32iOp: RV32IOp {
        switch self {
        case .ADD: return .ADD
        case .SUB: return .SUB
        case .SLL: return .SLL
        case .SLT: return .SLT
        case .SLTU: return .SLTU
        case .XOR: return .XOR
        case .SRL: return .SRL
        case .SRA: return .SRA
        case .OR: return .OR
        case .AND: return .AND
        case .ADDI: return .ADDI
        case .SLTI: return .SLTI
        case .SLTIU: return .SLTIU
        case .XORI: return .XORI
        case .ORI: return .ORI
        case .ANDI: return .ANDI
        case .SLLI: return .SLLI
        case .SRLI: return .SRLI
        case .SRAI: return .SRAI
        case .LB: return .LB
        case .LH: return .LH
        case .LW: return .LW
        case .LBU: return .LBU
        case .LHU: return .LHU
        case .SB: return .SB
        case .SH: return .SH
        case .SW: return .SW
        case .BEQ: return .BEQ
        case .BNE: return .BNE
        case .BLT: return .BLT
        case .BGE: return .BGE
        case .BLTU: return .BLTU
        case .BGEU: return .BGEU
        case .JAL: return .JAL
        case .JALR: return .JALR
        case .LUI: return .LUI
        case .AUIPC: return .AUIPC
        case .ECALL: return .ECALL
        case .EBREAK: return .EBREAK
        case .FENCE: return .FENCE
        }
    }
}

// MARK: - Instruction Format

/// The six RV32I instruction encoding formats.
public enum RV32IFormat {
    case R  // Register-register ALU
    case I  // Immediate ALU, loads, JALR
    case S  // Store
    case B  // Branch (conditional)
    case U  // Upper immediate (LUI, AUIPC)
    case J  // Jump (JAL)
}

// MARK: - Decoded Instruction

/// A fully decoded 32-bit RISC-V instruction with all extracted fields.
public struct DecodedInstruction {
    /// The identified instruction
    public let instruction: RV32IInstruction

    /// Instruction format
    public let format: RV32IFormat

    /// Original 32-bit instruction word
    public let word: UInt32

    /// Destination register (0-31), 0 for instructions with no destination
    public let rd: UInt8

    /// Source register 1 (0-31)
    public let rs1: UInt8

    /// Source register 2 (0-31), 0 for formats without rs2
    public let rs2: UInt8

    /// 3-bit function code (R, I, S, B formats)
    public let funct3: UInt8

    /// 7-bit function code (R-type only)
    public let funct7: UInt8

    /// Sign-extended immediate value. Encoding depends on format:
    ///   I-type: imm[11:0]
    ///   S-type: {imm[11:5], imm[4:0]}
    ///   B-type: {imm[12], imm[10:5], imm[4:1], 0} (halfword-aligned)
    ///   U-type: {imm[31:12], 0...0}
    ///   J-type: {imm[20], imm[10:1], imm[11], imm[19:12], 0} (halfword-aligned)
    ///   R-type: 0 (no immediate)
    public let immediate: Int32

    /// 7-bit opcode field from bits [6:0]
    public let opcode: UInt8

    public init(instruction: RV32IInstruction, format: RV32IFormat, word: UInt32,
                rd: UInt8, rs1: UInt8, rs2: UInt8,
                funct3: UInt8, funct7: UInt8, immediate: Int32, opcode: UInt8) {
        self.instruction = instruction
        self.format = format
        self.word = word
        self.rd = rd
        self.rs1 = rs1
        self.rs2 = rs2
        self.funct3 = funct3
        self.funct7 = funct7
        self.immediate = immediate
        self.opcode = opcode
    }
}

// MARK: - Decoder Error

public enum RISCVDecoderError: Error, CustomStringConvertible {
    case unknownOpcode(UInt8)
    case unknownInstruction(opcode: UInt8, funct3: UInt8, funct7: UInt8)

    public var description: String {
        switch self {
        case .unknownOpcode(let op):
            return "Unknown RISC-V opcode: 0x\(String(op, radix: 16))"
        case .unknownInstruction(let op, let f3, let f7):
            return "Unknown instruction: opcode=0x\(String(op, radix: 16)) funct3=\(f3) funct7=0x\(String(f7, radix: 16))"
        }
    }
}

// MARK: - Instruction Decoder

/// Decode a 32-bit RISC-V instruction word into a DecodedInstruction.
///
/// Extracts all fields according to the instruction format, sign-extends the
/// immediate, and identifies the specific RV32I instruction.
///
/// - Parameter word: The 32-bit instruction word (little-endian as stored in memory)
/// - Returns: A DecodedInstruction with all extracted fields
/// - Throws: RISCVDecoderError if the instruction is not a valid RV32I instruction
public func decodeInstruction(word: UInt32) throws -> DecodedInstruction {
    // Extract common fields
    let opcode = UInt8(word & 0x7F)
    let rd = UInt8((word >> 7) & 0x1F)
    let funct3 = UInt8((word >> 12) & 0x7)
    let rs1 = UInt8((word >> 15) & 0x1F)
    let rs2 = UInt8((word >> 20) & 0x1F)
    let funct7 = UInt8((word >> 25) & 0x7F)

    switch opcode {
    // R-type: opcode = 0110011 (0x33)
    case 0x33:
        let instr = try decodeRType(funct3: funct3, funct7: funct7)
        return DecodedInstruction(
            instruction: instr, format: .R, word: word,
            rd: rd, rs1: rs1, rs2: rs2,
            funct3: funct3, funct7: funct7, immediate: 0, opcode: opcode)

    // I-type ALU: opcode = 0010011 (0x13)
    case 0x13:
        let imm = signExtend12(UInt32((word >> 20) & 0xFFF))
        let instr = try decodeITypeALU(funct3: funct3, funct7: funct7, imm: imm)
        return DecodedInstruction(
            instruction: instr, format: .I, word: word,
            rd: rd, rs1: rs1, rs2: 0,
            funct3: funct3, funct7: funct7, immediate: imm, opcode: opcode)

    // I-type Load: opcode = 0000011 (0x03)
    case 0x03:
        let imm = signExtend12(UInt32((word >> 20) & 0xFFF))
        let instr = try decodeLoad(funct3: funct3)
        return DecodedInstruction(
            instruction: instr, format: .I, word: word,
            rd: rd, rs1: rs1, rs2: 0,
            funct3: funct3, funct7: 0, immediate: imm, opcode: opcode)

    // S-type Store: opcode = 0100011 (0x23)
    case 0x23:
        let imm = extractSImmediate(word)
        let instr = try decodeStore(funct3: funct3)
        return DecodedInstruction(
            instruction: instr, format: .S, word: word,
            rd: 0, rs1: rs1, rs2: rs2,
            funct3: funct3, funct7: 0, immediate: imm, opcode: opcode)

    // B-type Branch: opcode = 1100011 (0x63)
    case 0x63:
        let imm = extractBImmediate(word)
        let instr = try decodeBranch(funct3: funct3)
        return DecodedInstruction(
            instruction: instr, format: .B, word: word,
            rd: 0, rs1: rs1, rs2: rs2,
            funct3: funct3, funct7: 0, immediate: imm, opcode: opcode)

    // U-type LUI: opcode = 0110111 (0x37)
    case 0x37:
        let imm = Int32(bitPattern: word & 0xFFFFF000)
        return DecodedInstruction(
            instruction: .LUI, format: .U, word: word,
            rd: rd, rs1: 0, rs2: 0,
            funct3: 0, funct7: 0, immediate: imm, opcode: opcode)

    // U-type AUIPC: opcode = 0010111 (0x17)
    case 0x17:
        let imm = Int32(bitPattern: word & 0xFFFFF000)
        return DecodedInstruction(
            instruction: .AUIPC, format: .U, word: word,
            rd: rd, rs1: 0, rs2: 0,
            funct3: 0, funct7: 0, immediate: imm, opcode: opcode)

    // J-type JAL: opcode = 1101111 (0x6F)
    case 0x6F:
        let imm = extractJImmediate(word)
        return DecodedInstruction(
            instruction: .JAL, format: .J, word: word,
            rd: rd, rs1: 0, rs2: 0,
            funct3: 0, funct7: 0, immediate: imm, opcode: opcode)

    // I-type JALR: opcode = 1100111 (0x67)
    case 0x67:
        let imm = signExtend12(UInt32((word >> 20) & 0xFFF))
        return DecodedInstruction(
            instruction: .JALR, format: .I, word: word,
            rd: rd, rs1: rs1, rs2: 0,
            funct3: funct3, funct7: 0, immediate: imm, opcode: opcode)

    // System: opcode = 1110011 (0x73)
    case 0x73:
        let imm12 = (word >> 20) & 0xFFF
        let instr: RV32IInstruction = imm12 == 0 ? .ECALL : .EBREAK
        return DecodedInstruction(
            instruction: instr, format: .I, word: word,
            rd: 0, rs1: 0, rs2: 0,
            funct3: 0, funct7: 0, immediate: 0, opcode: opcode)

    // Fence: opcode = 0001111 (0x0F)
    case 0x0F:
        return DecodedInstruction(
            instruction: .FENCE, format: .I, word: word,
            rd: 0, rs1: 0, rs2: 0,
            funct3: funct3, funct7: 0, immediate: 0, opcode: opcode)

    default:
        throw RISCVDecoderError.unknownOpcode(opcode)
    }
}

// MARK: - Sub-decoders

/// Decode R-type instruction from funct3 and funct7.
private func decodeRType(funct3: UInt8, funct7: UInt8) throws -> RV32IInstruction {
    switch (funct3, funct7) {
    case (0x0, 0x00): return .ADD
    case (0x0, 0x20): return .SUB
    case (0x1, 0x00): return .SLL
    case (0x2, 0x00): return .SLT
    case (0x3, 0x00): return .SLTU
    case (0x4, 0x00): return .XOR
    case (0x5, 0x00): return .SRL
    case (0x5, 0x20): return .SRA
    case (0x6, 0x00): return .OR
    case (0x7, 0x00): return .AND
    default:
        throw RISCVDecoderError.unknownInstruction(opcode: 0x33, funct3: funct3, funct7: funct7)
    }
}

/// Decode I-type ALU instruction from funct3, with funct7 for shift variants.
private func decodeITypeALU(funct3: UInt8, funct7: UInt8, imm: Int32) throws -> RV32IInstruction {
    switch funct3 {
    case 0x0: return .ADDI
    case 0x2: return .SLTI
    case 0x3: return .SLTIU
    case 0x4: return .XORI
    case 0x6: return .ORI
    case 0x7: return .ANDI
    case 0x1: return .SLLI   // funct7 = 0x00
    case 0x5:
        // SRLI vs SRAI distinguished by bit 30 (funct7 bit 5)
        return funct7 == 0x20 ? .SRAI : .SRLI
    default:
        throw RISCVDecoderError.unknownInstruction(opcode: 0x13, funct3: funct3, funct7: funct7)
    }
}

/// Decode load instruction from funct3.
private func decodeLoad(funct3: UInt8) throws -> RV32IInstruction {
    switch funct3 {
    case 0x0: return .LB
    case 0x1: return .LH
    case 0x2: return .LW
    case 0x4: return .LBU
    case 0x5: return .LHU
    default:
        throw RISCVDecoderError.unknownInstruction(opcode: 0x03, funct3: funct3, funct7: 0)
    }
}

/// Decode store instruction from funct3.
private func decodeStore(funct3: UInt8) throws -> RV32IInstruction {
    switch funct3 {
    case 0x0: return .SB
    case 0x1: return .SH
    case 0x2: return .SW
    default:
        throw RISCVDecoderError.unknownInstruction(opcode: 0x23, funct3: funct3, funct7: 0)
    }
}

/// Decode branch instruction from funct3.
private func decodeBranch(funct3: UInt8) throws -> RV32IInstruction {
    switch funct3 {
    case 0x0: return .BEQ
    case 0x1: return .BNE
    case 0x4: return .BLT
    case 0x5: return .BGE
    case 0x6: return .BLTU
    case 0x7: return .BGEU
    default:
        throw RISCVDecoderError.unknownInstruction(opcode: 0x63, funct3: funct3, funct7: 0)
    }
}

// MARK: - Immediate Extraction

/// Sign-extend a 12-bit immediate to Int32.
private func signExtend12(_ val: UInt32) -> Int32 {
    if val & 0x800 != 0 {
        return Int32(bitPattern: val | 0xFFFFF000)
    }
    return Int32(val)
}

/// Extract S-type immediate: {inst[31:25], inst[11:7]} sign-extended to 32 bits.
private func extractSImmediate(_ word: UInt32) -> Int32 {
    let imm11_5 = (word >> 25) & 0x7F  // bits [31:25]
    let imm4_0 = (word >> 7) & 0x1F    // bits [11:7]
    let raw = (imm11_5 << 5) | imm4_0
    return signExtend12(raw)
}

/// Extract B-type immediate: {inst[31], inst[7], inst[30:25], inst[11:8], 0}
/// sign-extended to 32 bits (13-bit immediate, halfword-aligned).
private func extractBImmediate(_ word: UInt32) -> Int32 {
    let bit12  = (word >> 31) & 1       // inst[31]
    let bit11  = (word >> 7) & 1        // inst[7]
    let bit10_5 = (word >> 25) & 0x3F   // inst[30:25]
    let bit4_1  = (word >> 8) & 0xF     // inst[11:8]
    let raw = (bit12 << 12) | (bit11 << 11) | (bit10_5 << 5) | (bit4_1 << 1)
    // Sign-extend from bit 12
    if raw & 0x1000 != 0 {
        return Int32(bitPattern: raw | 0xFFFFE000)
    }
    return Int32(raw)
}

/// Extract J-type immediate: {inst[31], inst[19:12], inst[20], inst[30:21], 0}
/// sign-extended to 32 bits (21-bit immediate, halfword-aligned).
private func extractJImmediate(_ word: UInt32) -> Int32 {
    let bit20    = (word >> 31) & 1       // inst[31]
    let bit10_1  = (word >> 21) & 0x3FF   // inst[30:21]
    let bit11    = (word >> 20) & 1       // inst[20]
    let bit19_12 = (word >> 12) & 0xFF    // inst[19:12]
    let raw = (bit20 << 20) | (bit19_12 << 12) | (bit11 << 11) | (bit10_1 << 1)
    // Sign-extend from bit 20
    if raw & 0x100000 != 0 {
        return Int32(bitPattern: raw | 0xFFE00000)
    }
    return Int32(raw)
}

// MARK: - Instruction Encoding (for tests and program construction)

/// Encode an R-type instruction: op rd, rs1, rs2
public func encodeRType(funct7: UInt8, rs2: UInt8, rs1: UInt8, funct3: UInt8,
                         rd: UInt8, opcode: UInt8 = 0x33) -> UInt32 {
    return (UInt32(funct7) << 25) | (UInt32(rs2) << 20) | (UInt32(rs1) << 15) |
           (UInt32(funct3) << 12) | (UInt32(rd) << 7) | UInt32(opcode)
}

/// Encode an I-type instruction: op rd, rs1, imm
public func encodeIType(imm: Int32, rs1: UInt8, funct3: UInt8,
                         rd: UInt8, opcode: UInt8) -> UInt32 {
    let immBits = UInt32(bitPattern: imm) & 0xFFF
    return (immBits << 20) | (UInt32(rs1) << 15) |
           (UInt32(funct3) << 12) | (UInt32(rd) << 7) | UInt32(opcode)
}

/// Encode an S-type instruction: op rs2, imm(rs1)
public func encodeSType(imm: Int32, rs2: UInt8, rs1: UInt8,
                         funct3: UInt8, opcode: UInt8 = 0x23) -> UInt32 {
    let immU = UInt32(bitPattern: imm)
    let imm11_5 = (immU >> 5) & 0x7F
    let imm4_0 = immU & 0x1F
    return (imm11_5 << 25) | (UInt32(rs2) << 20) | (UInt32(rs1) << 15) |
           (UInt32(funct3) << 12) | (imm4_0 << 7) | UInt32(opcode)
}

/// Encode a B-type instruction: op rs1, rs2, offset
public func encodeBType(imm: Int32, rs2: UInt8, rs1: UInt8,
                         funct3: UInt8, opcode: UInt8 = 0x63) -> UInt32 {
    let immU = UInt32(bitPattern: imm)
    let bit12 = (immU >> 12) & 1
    let bit11 = (immU >> 11) & 1
    let bit10_5 = (immU >> 5) & 0x3F
    let bit4_1 = (immU >> 1) & 0xF
    return (bit12 << 31) | (bit10_5 << 25) | (UInt32(rs2) << 20) | (UInt32(rs1) << 15) |
           (UInt32(funct3) << 12) | (bit4_1 << 8) | (bit11 << 7) | UInt32(opcode)
}

/// Encode a U-type instruction: op rd, imm
public func encodeUType(imm: UInt32, rd: UInt8, opcode: UInt8) -> UInt32 {
    return (imm & 0xFFFFF000) | (UInt32(rd) << 7) | UInt32(opcode)
}

/// Encode a J-type instruction: jal rd, offset
public func encodeJType(imm: Int32, rd: UInt8, opcode: UInt8 = 0x6F) -> UInt32 {
    let immU = UInt32(bitPattern: imm)
    let bit20 = (immU >> 20) & 1
    let bit10_1 = (immU >> 1) & 0x3FF
    let bit11 = (immU >> 11) & 1
    let bit19_12 = (immU >> 12) & 0xFF
    return (bit20 << 31) | (bit10_1 << 21) | (bit11 << 20) | (bit19_12 << 12) |
           (UInt32(rd) << 7) | UInt32(opcode)
}

// MARK: - Convenience Encoders

/// Encode ADD rd, rs1, rs2
public func encodeADD(rd: UInt8, rs1: UInt8, rs2: UInt8) -> UInt32 {
    encodeRType(funct7: 0x00, rs2: rs2, rs1: rs1, funct3: 0x0, rd: rd)
}

/// Encode SUB rd, rs1, rs2
public func encodeSUB(rd: UInt8, rs1: UInt8, rs2: UInt8) -> UInt32 {
    encodeRType(funct7: 0x20, rs2: rs2, rs1: rs1, funct3: 0x0, rd: rd)
}

/// Encode ADDI rd, rs1, imm
public func encodeADDI(rd: UInt8, rs1: UInt8, imm: Int32) -> UInt32 {
    encodeIType(imm: imm, rs1: rs1, funct3: 0x0, rd: rd, opcode: 0x13)
}

/// Encode AND rd, rs1, rs2
public func encodeAND(rd: UInt8, rs1: UInt8, rs2: UInt8) -> UInt32 {
    encodeRType(funct7: 0x00, rs2: rs2, rs1: rs1, funct3: 0x7, rd: rd)
}

/// Encode OR rd, rs1, rs2
public func encodeOR(rd: UInt8, rs1: UInt8, rs2: UInt8) -> UInt32 {
    encodeRType(funct7: 0x00, rs2: rs2, rs1: rs1, funct3: 0x6, rd: rd)
}

/// Encode XOR rd, rs1, rs2
public func encodeXOR(rd: UInt8, rs1: UInt8, rs2: UInt8) -> UInt32 {
    encodeRType(funct7: 0x00, rs2: rs2, rs1: rs1, funct3: 0x4, rd: rd)
}

/// Encode SLT rd, rs1, rs2
public func encodeSLT(rd: UInt8, rs1: UInt8, rs2: UInt8) -> UInt32 {
    encodeRType(funct7: 0x00, rs2: rs2, rs1: rs1, funct3: 0x2, rd: rd)
}

/// Encode SLTU rd, rs1, rs2
public func encodeSLTU(rd: UInt8, rs1: UInt8, rs2: UInt8) -> UInt32 {
    encodeRType(funct7: 0x00, rs2: rs2, rs1: rs1, funct3: 0x3, rd: rd)
}

/// Encode SLL rd, rs1, rs2
public func encodeSLL(rd: UInt8, rs1: UInt8, rs2: UInt8) -> UInt32 {
    encodeRType(funct7: 0x00, rs2: rs2, rs1: rs1, funct3: 0x1, rd: rd)
}

/// Encode SRL rd, rs1, rs2
public func encodeSRL(rd: UInt8, rs1: UInt8, rs2: UInt8) -> UInt32 {
    encodeRType(funct7: 0x00, rs2: rs2, rs1: rs1, funct3: 0x5, rd: rd)
}

/// Encode SRA rd, rs1, rs2
public func encodeSRA(rd: UInt8, rs1: UInt8, rs2: UInt8) -> UInt32 {
    encodeRType(funct7: 0x20, rs2: rs2, rs1: rs1, funct3: 0x5, rd: rd)
}

/// Encode BEQ rs1, rs2, offset
public func encodeBEQ(rs1: UInt8, rs2: UInt8, offset: Int32) -> UInt32 {
    encodeBType(imm: offset, rs2: rs2, rs1: rs1, funct3: 0x0)
}

/// Encode BNE rs1, rs2, offset
public func encodeBNE(rs1: UInt8, rs2: UInt8, offset: Int32) -> UInt32 {
    encodeBType(imm: offset, rs2: rs2, rs1: rs1, funct3: 0x1)
}

/// Encode BLT rs1, rs2, offset
public func encodeBLT(rs1: UInt8, rs2: UInt8, offset: Int32) -> UInt32 {
    encodeBType(imm: offset, rs2: rs2, rs1: rs1, funct3: 0x4)
}

/// Encode BGE rs1, rs2, offset
public func encodeBGE(rs1: UInt8, rs2: UInt8, offset: Int32) -> UInt32 {
    encodeBType(imm: offset, rs2: rs2, rs1: rs1, funct3: 0x5)
}

/// Encode JAL rd, offset
public func encodeJAL(rd: UInt8, offset: Int32) -> UInt32 {
    encodeJType(imm: offset, rd: rd)
}

/// Encode JALR rd, rs1, offset
public func encodeJALR(rd: UInt8, rs1: UInt8, offset: Int32) -> UInt32 {
    encodeIType(imm: offset, rs1: rs1, funct3: 0x0, rd: rd, opcode: 0x67)
}

/// Encode LUI rd, imm (upper 20 bits)
public func encodeLUI(rd: UInt8, imm: UInt32) -> UInt32 {
    encodeUType(imm: imm << 12, rd: rd, opcode: 0x37)
}

/// Encode AUIPC rd, imm (upper 20 bits)
public func encodeAUIPC(rd: UInt8, imm: UInt32) -> UInt32 {
    encodeUType(imm: imm << 12, rd: rd, opcode: 0x17)
}

/// Encode LW rd, offset(rs1)
public func encodeLW(rd: UInt8, rs1: UInt8, offset: Int32) -> UInt32 {
    encodeIType(imm: offset, rs1: rs1, funct3: 0x2, rd: rd, opcode: 0x03)
}

/// Encode SW rs2, offset(rs1)
public func encodeSW(rs2: UInt8, rs1: UInt8, offset: Int32) -> UInt32 {
    encodeSType(imm: offset, rs2: rs2, rs1: rs1, funct3: 0x2)
}

/// Encode SLLI rd, rs1, shamt
public func encodeSLLI(rd: UInt8, rs1: UInt8, shamt: UInt8) -> UInt32 {
    encodeIType(imm: Int32(shamt & 0x1F), rs1: rs1, funct3: 0x1, rd: rd, opcode: 0x13)
}

/// Encode SRLI rd, rs1, shamt
public func encodeSRLI(rd: UInt8, rs1: UInt8, shamt: UInt8) -> UInt32 {
    encodeIType(imm: Int32(shamt & 0x1F), rs1: rs1, funct3: 0x5, rd: rd, opcode: 0x13)
}

/// Encode SRAI rd, rs1, shamt
public func encodeSRAI(rd: UInt8, rs1: UInt8, shamt: UInt8) -> UInt32 {
    let immBits = Int32(0x400) | Int32(shamt & 0x1F)  // funct7[5] = 1
    return encodeIType(imm: immBits, rs1: rs1, funct3: 0x5, rd: rd, opcode: 0x13)
}

/// Encode ECALL
public func encodeECALL() -> UInt32 {
    return 0x00000073
}

/// Encode EBREAK
public func encodeEBREAK() -> UInt32 {
    return 0x00100073
}
