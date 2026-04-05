// Witness (.wtns) binary format parser for Circom/snarkjs compatibility
//
// Spec: https://github.com/iden3/snarkjs/blob/master/src/wtns_utils.js
//
// File layout:
//   Magic: "wtns" (4 bytes)
//   Version: UInt32 LE
//   Number of sections: UInt32 LE
//   Section 1 (Header): field size, prime, nWitness
//   Section 2 (Witness data): nWitness field elements, each `fieldSize` bytes LE
//
// Field elements are little-endian byte arrays of `fieldSize` bytes.
// This parser targets BN254 (fieldSize = 32).

import Foundation

// MARK: - Witness File Structures

/// Parsed witness file header.
public struct WitnessHeader {
    public let fieldSize: UInt32    // bytes per field element (32 for BN254)
    public let prime: [UInt8]       // field prime as LE bytes
    public let nWitness: UInt32     // number of witness values (including wire 0 = "one")
}

/// Complete parsed witness file.
public struct WitnessFile {
    public let version: UInt32
    public let header: WitnessHeader
    public let values: [Fr]         // witness values in Montgomery form
}

// MARK: - Witness Parser Errors

public enum WitnessParserError: Error, CustomStringConvertible {
    case invalidMagic
    case unsupportedVersion(UInt32)
    case missingHeaderSection
    case missingWitnessSection
    case unsupportedFieldSize(UInt32)
    case primeMismatch
    case unexpectedEOF
    case witnessCountMismatch(expected: UInt32, got: Int)

    public var description: String {
        switch self {
        case .invalidMagic: return "WTNS: invalid magic bytes (expected 'wtns')"
        case .unsupportedVersion(let v): return "WTNS: unsupported version \(v)"
        case .missingHeaderSection: return "WTNS: missing header section (type 1)"
        case .missingWitnessSection: return "WTNS: missing witness section (type 2)"
        case .unsupportedFieldSize(let s): return "WTNS: unsupported field size \(s) (expected 32)"
        case .primeMismatch: return "WTNS: field prime does not match BN254 Fr"
        case .unexpectedEOF: return "WTNS: unexpected end of data"
        case .witnessCountMismatch(let e, let g): return "WTNS: expected \(e) witnesses, got \(g)"
        }
    }
}

// MARK: - Witness Parser

public struct WitnessParser {

    // BN254 scalar field prime r in LE bytes (same as R1CSParser)
    private static let bn254FrPrimeLE: [UInt8] = {
        var bytes = [UInt8](repeating: 0, count: 32)
        let limbs: [UInt64] = Fr.P
        for i in 0..<4 {
            let v = limbs[i]
            for j in 0..<8 {
                bytes[i * 8 + j] = UInt8((v >> (j * 8)) & 0xFF)
            }
        }
        return bytes
    }()

    /// Parse a .wtns binary file from raw bytes.
    public static func parse(_ data: Data) throws -> WitnessFile {
        var reader = BinaryReader(data)

        // Magic: "wtns"
        let magic = try reader.readBytes(4)
        guard magic == [0x77, 0x74, 0x6E, 0x73] else {
            throw WitnessParserError.invalidMagic
        }

        // Version
        let version = try reader.readUInt32()
        guard version == 2 else {
            throw WitnessParserError.unsupportedVersion(version)
        }

        // Number of sections
        let nSections = try reader.readUInt32()

        // Read all sections
        var sections: [UInt32: Data] = [:]
        for _ in 0..<nSections {
            let sectionType = try reader.readUInt32()
            let sectionSize = try reader.readUInt64()
            let sectionData = try reader.readBytes(Int(sectionSize))
            sections[sectionType] = Data(sectionData)
        }

        // Section 1: Header
        guard let headerData = sections[1] else {
            throw WitnessParserError.missingHeaderSection
        }
        let header = try parseHeader(headerData)

        // Validate field
        guard header.fieldSize == 32 else {
            throw WitnessParserError.unsupportedFieldSize(header.fieldSize)
        }
        guard header.prime == bn254FrPrimeLE else {
            throw WitnessParserError.primeMismatch
        }

        // Section 2: Witness values
        guard let witnessData = sections[2] else {
            throw WitnessParserError.missingWitnessSection
        }
        let values = try parseWitnessValues(witnessData, header: header)

        return WitnessFile(version: version, header: header, values: values)
    }

    /// Parse from a file path.
    public static func parse(contentsOf url: URL) throws -> WitnessFile {
        let data = try Data(contentsOf: url)
        return try parse(data)
    }

    // MARK: - Section Parsers

    private static func parseHeader(_ data: Data) throws -> WitnessHeader {
        var r = BinaryReader(data)
        let fieldSize = try r.readUInt32()
        let prime = try r.readBytes(Int(fieldSize))
        let nWitness = try r.readUInt32()
        return WitnessHeader(fieldSize: fieldSize, prime: prime, nWitness: nWitness)
    }

    private static func parseWitnessValues(_ data: Data, header: WitnessHeader) throws -> [Fr] {
        let fs = Int(header.fieldSize)
        let expectedSize = Int(header.nWitness) * fs
        guard data.count >= expectedSize else {
            throw WitnessParserError.witnessCountMismatch(
                expected: header.nWitness,
                got: data.count / fs
            )
        }

        var values = [Fr]()
        values.reserveCapacity(Int(header.nWitness))
        var r = BinaryReader(data)

        for _ in 0..<header.nWitness {
            let bytes = try r.readBytes(fs)
            values.append(R1CSParser.fieldElementFromLE(bytes))
        }
        return values
    }

    // MARK: - Extraction Helpers

    /// Extract the full z vector from a witness file: [one, outputs, pubInputs, privInputs, internal...]
    /// This is already the wire ordering Circom uses, and matches R1CSInstance's variable layout.
    public static func witnessVector(_ file: WitnessFile) -> [Fr] {
        return file.values
    }

    /// Extract public inputs (outputs + public inputs) from a witness file.
    /// These are wires 1..(nOutputs + nPubInputs) in Circom's ordering.
    public static func publicInputs(_ file: WitnessFile, r1cs: R1CSFile) -> [Fr] {
        let nPublic = Int(r1cs.header.nOutputs + r1cs.header.nPubInputs)
        guard file.values.count > nPublic else { return [] }
        return Array(file.values[1...(nPublic)])
    }

    /// Extract private witness (everything after public inputs) from a witness file.
    public static func privateWitness(_ file: WitnessFile, r1cs: R1CSFile) -> [Fr] {
        let nPublic = Int(r1cs.header.nOutputs + r1cs.header.nPubInputs)
        let start = 1 + nPublic
        guard file.values.count > start else { return [] }
        return Array(file.values[start...])
    }
}
