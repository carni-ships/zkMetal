// R1CS binary format parser for Circom/snarkjs compatibility
//
// Spec: https://github.com/iden3/r1csfile/blob/master/doc/r1cs_bin_format.md
//
// File layout:
//   Magic: "r1cs" (4 bytes)
//   Version: UInt32 LE
//   Number of sections: UInt32 LE
//   Sections (type: UInt32 LE, size: UInt64 LE, data: [UInt8])
//     Section 1 (Header): field size, prime, nWires, nOutputs, nPubInputs, nPrivInputs, nLabels, nConstraints
//     Section 2 (Constraints): each constraint = A, B, C sparse vectors
//     Section 3 (Wire-to-label mapping): [UInt64] per wire
//
// Field elements are little-endian byte arrays of `fieldSize` bytes.
// This parser targets BN254 (fieldSize = 32).

import Foundation

// MARK: - R1CS File Structures

/// Parsed R1CS file header.
public struct R1CSHeader {
    public let fieldSize: UInt32        // bytes per field element (32 for BN254)
    public let prime: [UInt8]           // field prime as LE bytes
    public let nWires: UInt32           // total number of wires (including wire 0 = "one")
    public let nOutputs: UInt32         // number of output signals
    public let nPubInputs: UInt32       // number of public input signals
    public let nPrivInputs: UInt32      // number of private input signals
    public let nLabels: UInt64          // number of labels (wire-to-label map entries)
    public let nConstraints: UInt32     // number of constraints
}

/// A sparse linear combination from the R1CS file: [(wireId, coefficient)].
public struct R1CSSparseVec {
    public let terms: [(wireId: UInt32, coeff: Fr)]
    public init(terms: [(wireId: UInt32, coeff: Fr)]) { self.terms = terms }
}

/// A single R1CS constraint: A * w . B * w = C * w.
public struct R1CSFileConstraint {
    public let a: R1CSSparseVec
    public let b: R1CSSparseVec
    public let c: R1CSSparseVec
    public init(a: R1CSSparseVec, b: R1CSSparseVec, c: R1CSSparseVec) {
        self.a = a; self.b = b; self.c = c
    }
}

/// Complete parsed R1CS file.
public struct R1CSFile {
    public let version: UInt32
    public let header: R1CSHeader
    public let constraints: [R1CSFileConstraint]
    public let wireToLabel: [UInt64]?   // optional section 3
}

// MARK: - R1CS Parser

public enum R1CSParserError: Error, CustomStringConvertible {
    case invalidMagic
    case unsupportedVersion(UInt32)
    case missingHeaderSection
    case missingConstraintSection
    case unsupportedFieldSize(UInt32)
    case primeMismatch
    case unexpectedEOF
    case invalidConstraintData

    public var description: String {
        switch self {
        case .invalidMagic: return "R1CS: invalid magic bytes (expected 'r1cs')"
        case .unsupportedVersion(let v): return "R1CS: unsupported version \(v)"
        case .missingHeaderSection: return "R1CS: missing header section (type 1)"
        case .missingConstraintSection: return "R1CS: missing constraint section (type 2)"
        case .unsupportedFieldSize(let s): return "R1CS: unsupported field size \(s) (expected 32)"
        case .primeMismatch: return "R1CS: field prime does not match BN254 Fr"
        case .unexpectedEOF: return "R1CS: unexpected end of data"
        case .invalidConstraintData: return "R1CS: malformed constraint data"
        }
    }
}

public struct R1CSParser {

    // BN254 scalar field prime r in LE bytes
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

    /// Parse an R1CS binary file from raw bytes.
    public static func parse(_ data: Data) throws -> R1CSFile {
        var reader = BinaryReader(data)

        // Magic: "r1cs"
        let magic = try reader.readBytes(4)
        guard magic == [0x72, 0x31, 0x63, 0x73] else {
            throw R1CSParserError.invalidMagic
        }

        // Version
        let version = try reader.readUInt32()
        guard version == 1 else {
            throw R1CSParserError.unsupportedVersion(version)
        }

        // Number of sections
        let nSections = try reader.readUInt32()

        // Read all sections into a dictionary keyed by section type
        var sections: [UInt32: Data] = [:]
        for _ in 0..<nSections {
            let sectionType = try reader.readUInt32()
            let sectionSize = try reader.readUInt64()
            let sectionData = try reader.readBytes(Int(sectionSize))
            sections[sectionType] = Data(sectionData)
        }

        // Section 1: Header
        guard let headerData = sections[1] else {
            throw R1CSParserError.missingHeaderSection
        }
        let header = try parseHeader(headerData)

        // Validate field
        guard header.fieldSize == 32 else {
            throw R1CSParserError.unsupportedFieldSize(header.fieldSize)
        }
        guard header.prime == bn254FrPrimeLE else {
            throw R1CSParserError.primeMismatch
        }

        // Section 2: Constraints
        guard let constraintData = sections[2] else {
            throw R1CSParserError.missingConstraintSection
        }
        let constraints = try parseConstraints(constraintData, header: header)

        // Section 3: Wire-to-label mapping (optional)
        var wireToLabel: [UInt64]? = nil
        if let labelData = sections[3] {
            wireToLabel = try parseLabelSection(labelData, nWires: header.nWires)
        }

        return R1CSFile(version: version, header: header,
                        constraints: constraints, wireToLabel: wireToLabel)
    }

    /// Parse from a file path.
    public static func parse(contentsOf url: URL) throws -> R1CSFile {
        let data = try Data(contentsOf: url)
        return try parse(data)
    }

    // MARK: - Section Parsers

    private static func parseHeader(_ data: Data) throws -> R1CSHeader {
        var r = BinaryReader(data)
        let fieldSize = try r.readUInt32()
        let prime = try r.readBytes(Int(fieldSize))
        let nWires = try r.readUInt32()
        let nOutputs = try r.readUInt32()
        let nPubInputs = try r.readUInt32()
        let nPrivInputs = try r.readUInt32()
        let nLabels = try r.readUInt64()
        let nConstraints = try r.readUInt32()
        return R1CSHeader(fieldSize: fieldSize, prime: prime,
                          nWires: nWires, nOutputs: nOutputs,
                          nPubInputs: nPubInputs, nPrivInputs: nPrivInputs,
                          nLabels: nLabels, nConstraints: nConstraints)
    }

    private static func parseConstraints(_ data: Data, header: R1CSHeader) throws -> [R1CSFileConstraint] {
        var r = BinaryReader(data)
        let fs = Int(header.fieldSize)
        var constraints = [R1CSFileConstraint]()
        constraints.reserveCapacity(Int(header.nConstraints))

        for _ in 0..<header.nConstraints {
            let a = try parseSparseVec(&r, fieldSize: fs)
            let b = try parseSparseVec(&r, fieldSize: fs)
            let c = try parseSparseVec(&r, fieldSize: fs)
            constraints.append(R1CSFileConstraint(a: a, b: b, c: c))
        }
        return constraints
    }

    private static func parseSparseVec(_ r: inout BinaryReader, fieldSize: Int) throws -> R1CSSparseVec {
        let nTerms = try r.readUInt32()
        var terms = [(wireId: UInt32, coeff: Fr)]()
        terms.reserveCapacity(Int(nTerms))

        for _ in 0..<nTerms {
            let wireId = try r.readUInt32()
            let coeffBytes = try r.readBytes(fieldSize)
            let coeff = fieldElementFromLE(coeffBytes)
            terms.append((wireId: wireId, coeff: coeff))
        }
        return R1CSSparseVec(terms: terms)
    }

    private static func parseLabelSection(_ data: Data, nWires: UInt32) throws -> [UInt64] {
        var r = BinaryReader(data)
        var labels = [UInt64]()
        labels.reserveCapacity(Int(nWires))
        for _ in 0..<nWires {
            labels.append(try r.readUInt64())
        }
        return labels
    }

    // MARK: - Field Element Conversion

    /// Convert a little-endian byte array to Fr (Montgomery form).
    /// Input: 32 bytes in LE representing a field element in standard form.
    /// Output: Fr in Montgomery representation.
    static func fieldElementFromLE(_ bytes: [UInt8]) -> Fr {
        precondition(bytes.count == 32)
        // Pack LE bytes into 4 UInt64 limbs (LE)
        var limbs = [UInt64](repeating: 0, count: 4)
        for i in 0..<4 {
            var v: UInt64 = 0
            for j in 0..<8 {
                v |= UInt64(bytes[i * 8 + j]) << (j * 8)
            }
            limbs[i] = v
        }
        // Convert standard form -> Montgomery form via multiplication by R^2
        let raw = Fr.from64(limbs)
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))
    }

    // MARK: - Conversion to R1CSInstance

    /// Convert a parsed R1CSFile to zkMetal's R1CSInstance (for Groth16 proving).
    ///
    /// Circom wire layout: [one, out_1..out_nO, pub_1..pub_nPub, priv_1..priv_nPriv, internal...]
    /// zkMetal R1CSInstance expects: [one, public_inputs..., witness...]
    /// where numPublic = nOutputs + nPubInputs (outputs are public in Circom).
    public static func toR1CSInstance(_ file: R1CSFile) -> R1CSInstance {
        let header = file.header
        let numPublic = Int(header.nOutputs + header.nPubInputs)
        let numVars = Int(header.nWires)

        var aEntries = [R1CSEntry]()
        var bEntries = [R1CSEntry]()
        var cEntries = [R1CSEntry]()

        for (row, constraint) in file.constraints.enumerated() {
            for term in constraint.a.terms {
                aEntries.append(R1CSEntry(row: row, col: Int(term.wireId), val: term.coeff))
            }
            for term in constraint.b.terms {
                bEntries.append(R1CSEntry(row: row, col: Int(term.wireId), val: term.coeff))
            }
            for term in constraint.c.terms {
                cEntries.append(R1CSEntry(row: row, col: Int(term.wireId), val: term.coeff))
            }
        }

        return R1CSInstance(
            numConstraints: Int(header.nConstraints),
            numVars: numVars,
            numPublic: numPublic,
            aEntries: aEntries,
            bEntries: bEntries,
            cEntries: cEntries
        )
    }
}

// MARK: - Binary Reader

struct BinaryReader {
    private let data: Data
    private var offset: Int = 0

    init(_ data: Data) {
        self.data = data
    }

    var remaining: Int { data.count - offset }

    mutating func readBytes(_ count: Int) throws -> [UInt8] {
        guard offset + count <= data.count else {
            throw R1CSParserError.unexpectedEOF
        }
        let bytes = [UInt8](data[offset..<(offset + count)])
        offset += count
        return bytes
    }

    mutating func readUInt32() throws -> UInt32 {
        let bytes = try readBytes(4)
        return UInt32(bytes[0]) | (UInt32(bytes[1]) << 8) |
               (UInt32(bytes[2]) << 16) | (UInt32(bytes[3]) << 24)
    }

    mutating func readUInt64() throws -> UInt64 {
        let bytes = try readBytes(8)
        var val: UInt64 = 0
        for i in 0..<8 { val |= UInt64(bytes[i]) << (i * 8) }
        return val
    }
}
