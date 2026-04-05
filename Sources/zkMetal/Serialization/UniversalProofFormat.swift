// UniversalProofFormat — Compact binary format for encoding/decoding any proof type.
//
// Header layout (14 bytes):
//   [0..3]   Magic bytes: "ZKPF" (0x5A 0x4B 0x50 0x46)
//   [4..5]   Format version (u16 LE)
//   [6]      ProofType (u8)
//   [7]      CurveId (u8)
//   [8]      FieldId (u8)
//   [9..12]  Data length (u32 LE) — total bytes after this header
//   [13]     Flags (u8) — reserved for future use
//
// After header: proofData, publicInputs, metadata (all length-prefixed).

import Foundation

// MARK: - Enumerations

/// Proof system type tag.
public enum ProofSystemType: UInt8, Codable, CaseIterable, CustomStringConvertible {
    case groth16  = 0
    case plonk    = 1
    case stark    = 2
    case fri      = 3
    case ipa      = 4
    case kzg      = 5
    case spartan  = 6
    case nova     = 7

    public var description: String {
        switch self {
        case .groth16:  return "groth16"
        case .plonk:    return "plonk"
        case .stark:    return "stark"
        case .fri:      return "fri"
        case .ipa:      return "ipa"
        case .kzg:      return "kzg"
        case .spartan:  return "spartan"
        case .nova:     return "nova"
        }
    }
}

/// Curve identifier for the proof's elliptic curve.
public enum CurveId: UInt8, Codable, CaseIterable, CustomStringConvertible {
    case none       = 0  // for field-only proofs (STARKs)
    case bn254      = 1
    case bls12_381  = 2
    case bls12_377  = 3
    case pallas     = 4
    case vesta      = 5
    case secp256k1  = 6
    case grumpkin   = 7
    case ed25519    = 8

    public var description: String {
        switch self {
        case .none:      return "none"
        case .bn254:     return "bn254"
        case .bls12_381: return "bls12-381"
        case .bls12_377: return "bls12-377"
        case .pallas:    return "pallas"
        case .vesta:     return "vesta"
        case .secp256k1: return "secp256k1"
        case .grumpkin:  return "grumpkin"
        case .ed25519:   return "ed25519"
        }
    }
}

/// Field identifier for the proof's base/scalar field.
public enum FieldId: UInt8, Codable, CaseIterable, CustomStringConvertible {
    case none       = 0
    case bn254Fr    = 1
    case bls12_381Fr = 2
    case bls12_377Fr = 3
    case babybear   = 4
    case goldilocks  = 5
    case mersenne31  = 6

    public var description: String {
        switch self {
        case .none:        return "none"
        case .bn254Fr:     return "bn254-fr"
        case .bls12_381Fr: return "bls12-381-fr"
        case .bls12_377Fr: return "bls12-377-fr"
        case .babybear:    return "babybear"
        case .goldilocks:  return "goldilocks"
        case .mersenne31:  return "mersenne31"
        }
    }
}

// MARK: - UniversalProof

/// A universal proof container that can hold any proof type in a compact binary format.
public struct UniversalProof: Codable, Equatable {
    /// The proof system type.
    public let type: ProofSystemType

    /// Format version for forward compatibility.
    public let version: UInt16

    /// Elliptic curve used (if applicable).
    public let curveId: CurveId

    /// Scalar/base field used.
    public let fieldId: FieldId

    /// Raw proof data bytes (scheme-specific encoding).
    public let proofData: [UInt8]

    /// Public inputs as raw bytes (each input is a fixed-size field element).
    public let publicInputs: [[UInt8]]

    /// Optional key-value metadata (prover version, timestamp, circuit hash, etc.).
    public let metadata: [String: String]

    public init(type: ProofSystemType, version: UInt16 = 1,
                curveId: CurveId = .none, fieldId: FieldId = .none,
                proofData: [UInt8], publicInputs: [[UInt8]] = [],
                metadata: [String: String] = [:]) {
        self.type = type
        self.version = version
        self.curveId = curveId
        self.fieldId = fieldId
        self.proofData = proofData
        self.publicInputs = publicInputs
        self.metadata = metadata
    }
}

// MARK: - Errors

public enum UniversalProofError: Error, CustomStringConvertible {
    case invalidMagic
    case unsupportedVersion(UInt16)
    case unknownProofType(UInt8)
    case unknownCurveId(UInt8)
    case unknownFieldId(UInt8)
    case truncatedHeader
    case truncatedData(expected: Int, available: Int)
    case invalidJSON(String)
    case codecNotRegistered(ProofSystemType)

    public var description: String {
        switch self {
        case .invalidMagic:
            return "Invalid magic bytes: expected ZKPF"
        case .unsupportedVersion(let v):
            return "Unsupported format version: \(v)"
        case .unknownProofType(let t):
            return "Unknown proof type: \(t)"
        case .unknownCurveId(let c):
            return "Unknown curve id: \(c)"
        case .unknownFieldId(let f):
            return "Unknown field id: \(f)"
        case .truncatedHeader:
            return "Data too short for header (need 14 bytes)"
        case .truncatedData(let expected, let available):
            return "Truncated data: need \(expected) bytes, have \(available)"
        case .invalidJSON(let msg):
            return "Invalid JSON: \(msg)"
        case .codecNotRegistered(let t):
            return "No codec registered for proof type: \(t)"
        }
    }
}

// MARK: - Binary Serialization

/// Magic bytes identifying a universal proof binary format.
private let universalProofMagic: [UInt8] = [0x5A, 0x4B, 0x50, 0x46] // "ZKPF"

/// Header size in bytes.
private let headerSize = 14

public extension UniversalProof {

    /// Serialize the proof to a compact binary format.
    func serialize() -> [UInt8] {
        var out = [UInt8]()
        // Reserve space; header + proof data + public inputs + metadata
        out.reserveCapacity(headerSize + proofData.count + publicInputs.count * 36 + 256)

        // Magic
        out.append(contentsOf: universalProofMagic)

        // Version (u16 LE)
        out.append(UInt8(version & 0xFF))
        out.append(UInt8(version >> 8))

        // ProofType, CurveId, FieldId
        out.append(type.rawValue)
        out.append(curveId.rawValue)
        out.append(fieldId.rawValue)

        // Data length placeholder (filled after body is written)
        let dataLenOffset = out.count
        out.append(contentsOf: [0, 0, 0, 0])

        // Flags (reserved)
        out.append(0)

        // --- Body ---
        let bodyStart = out.count

        // Proof data (length-prefixed)
        writeU32(&out, UInt32(proofData.count))
        out.append(contentsOf: proofData)

        // Public inputs (count, then each length-prefixed)
        writeU32(&out, UInt32(publicInputs.count))
        for input in publicInputs {
            writeU32(&out, UInt32(input.count))
            out.append(contentsOf: input)
        }

        // Metadata (count of key-value pairs, each string length-prefixed)
        writeU32(&out, UInt32(metadata.count))
        for (key, value) in metadata.sorted(by: { $0.key < $1.key }) {
            let keyBytes = Array(key.utf8)
            writeU32(&out, UInt32(keyBytes.count))
            out.append(contentsOf: keyBytes)
            let valBytes = Array(value.utf8)
            writeU32(&out, UInt32(valBytes.count))
            out.append(contentsOf: valBytes)
        }

        // Fill in data length
        let bodyLen = UInt32(out.count - bodyStart)
        out[dataLenOffset + 0] = UInt8(bodyLen & 0xFF)
        out[dataLenOffset + 1] = UInt8((bodyLen >> 8) & 0xFF)
        out[dataLenOffset + 2] = UInt8((bodyLen >> 16) & 0xFF)
        out[dataLenOffset + 3] = UInt8((bodyLen >> 24) & 0xFF)

        return out
    }

    /// Deserialize a proof from binary data.
    static func deserialize(_ data: [UInt8]) throws -> UniversalProof {
        guard data.count >= headerSize else {
            throw UniversalProofError.truncatedHeader
        }

        // Verify magic
        guard data[0] == universalProofMagic[0] &&
              data[1] == universalProofMagic[1] &&
              data[2] == universalProofMagic[2] &&
              data[3] == universalProofMagic[3] else {
            throw UniversalProofError.invalidMagic
        }

        // Version
        let version = UInt16(data[4]) | (UInt16(data[5]) << 8)
        guard version <= 1 else {
            throw UniversalProofError.unsupportedVersion(version)
        }

        // ProofType
        guard let proofType = ProofSystemType(rawValue: data[6]) else {
            throw UniversalProofError.unknownProofType(data[6])
        }

        // CurveId
        guard let curveId = CurveId(rawValue: data[7]) else {
            throw UniversalProofError.unknownCurveId(data[7])
        }

        // FieldId
        guard let fieldId = FieldId(rawValue: data[8]) else {
            throw UniversalProofError.unknownFieldId(data[8])
        }

        // Data length
        let dataLen = Int(readU32(data, offset: 9))
        // flags = data[13], reserved

        let bodyStart = headerSize
        guard data.count >= bodyStart + dataLen else {
            throw UniversalProofError.truncatedData(
                expected: bodyStart + dataLen, available: data.count)
        }

        var offset = bodyStart

        // Proof data
        let proofDataLen = Int(readU32(data, offset: offset)); offset += 4
        guard offset + proofDataLen <= data.count else {
            throw UniversalProofError.truncatedData(expected: proofDataLen, available: data.count - offset)
        }
        let proofData = Array(data[offset..<(offset + proofDataLen)]); offset += proofDataLen

        // Public inputs
        let numInputs = Int(readU32(data, offset: offset)); offset += 4
        var publicInputs = [[UInt8]]()
        publicInputs.reserveCapacity(numInputs)
        for _ in 0..<numInputs {
            let inputLen = Int(readU32(data, offset: offset)); offset += 4
            guard offset + inputLen <= data.count else {
                throw UniversalProofError.truncatedData(expected: inputLen, available: data.count - offset)
            }
            publicInputs.append(Array(data[offset..<(offset + inputLen)])); offset += inputLen
        }

        // Metadata
        let numMeta = Int(readU32(data, offset: offset)); offset += 4
        var metadata = [String: String]()
        for _ in 0..<numMeta {
            let keyLen = Int(readU32(data, offset: offset)); offset += 4
            guard offset + keyLen <= data.count else {
                throw UniversalProofError.truncatedData(expected: keyLen, available: data.count - offset)
            }
            let key = String(bytes: data[offset..<(offset + keyLen)], encoding: .utf8) ?? ""; offset += keyLen

            let valLen = Int(readU32(data, offset: offset)); offset += 4
            guard offset + valLen <= data.count else {
                throw UniversalProofError.truncatedData(expected: valLen, available: data.count - offset)
            }
            let value = String(bytes: data[offset..<(offset + valLen)], encoding: .utf8) ?? ""; offset += valLen
            metadata[key] = value
        }

        return UniversalProof(
            type: proofType, version: version,
            curveId: curveId, fieldId: fieldId,
            proofData: proofData, publicInputs: publicInputs,
            metadata: metadata
        )
    }
}

// MARK: - JSON Serialization

public extension UniversalProof {

    /// Encode the proof to a JSON string for human-readable interchange.
    func toJSON(prettyPrint: Bool = true) -> String {
        let encoder = JSONEncoder()
        if prettyPrint { encoder.outputFormatting = [.prettyPrinted, .sortedKeys] }
        guard let data = try? encoder.encode(self) else { return "{}" }
        return String(data: data, encoding: .utf8) ?? "{}"
    }

    /// Decode a proof from a JSON string.
    static func fromJSON(_ json: String) throws -> UniversalProof {
        guard let data = json.data(using: .utf8) else {
            throw UniversalProofError.invalidJSON("invalid UTF-8")
        }
        do {
            return try JSONDecoder().decode(UniversalProof.self, from: data)
        } catch {
            throw UniversalProofError.invalidJSON(error.localizedDescription)
        }
    }
}

// MARK: - Proof Hashing

public extension UniversalProof {

    /// Compute a SHA-256 fingerprint of the proof (type + curve + field + proofData + publicInputs).
    /// Useful for proof deduplication and content addressing.
    func proofHash() -> [UInt8] {
        var input = [UInt8]()
        input.append(type.rawValue)
        input.append(curveId.rawValue)
        input.append(fieldId.rawValue)
        input.append(contentsOf: proofData)
        for pi in publicInputs {
            input.append(contentsOf: pi)
        }
        return sha256(input)
    }
}

// MARK: - Convenience: Wrap existing proof types

public extension UniversalProof {

    /// Wrap a Groth16Proof into a UniversalProof.
    static func fromGroth16(_ proof: Groth16Proof, publicInputs: [Fr] = [],
                            metadata: [String: String] = [:]) -> UniversalProof {
        let w = ProofWriter()
        w.writeLabel("GROTH16-v1")
        w.writePointProjective(proof.a)
        // G2 point: write as 6 Fp values (x.c0, x.c1, y.c0, y.c1, z.c0, z.c1)
        w.writeFp(proof.b.x.c0); w.writeFp(proof.b.x.c1)
        w.writeFp(proof.b.y.c0); w.writeFp(proof.b.y.c1)
        w.writeFp(proof.b.z.c0); w.writeFp(proof.b.z.c1)
        w.writePointProjective(proof.c)
        let proofBytes = w.finalize()

        let inputs = publicInputs.map { fr -> [UInt8] in
            let iw = ProofWriter()
            iw.writeFr(fr)
            return iw.finalize()
        }

        return UniversalProof(
            type: .groth16, version: 1,
            curveId: .bn254, fieldId: .bn254Fr,
            proofData: proofBytes, publicInputs: inputs,
            metadata: metadata
        )
    }

    /// Extract a Groth16Proof from a universal proof container.
    func toGroth16() throws -> Groth16Proof {
        guard type == .groth16 else {
            throw UniversalProofError.codecNotRegistered(type)
        }
        let r = ProofReader(proofData)
        try r.expectLabel("GROTH16-v1")
        let a = try r.readPointProjective()
        let bxc0 = try r.readFp(); let bxc1 = try r.readFp()
        let byc0 = try r.readFp(); let byc1 = try r.readFp()
        let bzc0 = try r.readFp(); let bzc1 = try r.readFp()
        let b = G2ProjectivePoint(
            x: Fp2(c0: bxc0, c1: bxc1),
            y: Fp2(c0: byc0, c1: byc1),
            z: Fp2(c0: bzc0, c1: bzc1)
        )
        let c = try r.readPointProjective()
        return Groth16Proof(a: a, b: b, c: c)
    }

    /// Wrap a PlonkProof into a UniversalProof.
    static func fromPlonk(_ proof: PlonkProof,
                          metadata: [String: String] = [:]) -> UniversalProof {
        let w = ProofWriter()
        w.writeLabel("PLONK-v1")
        // Round 1 commitments
        w.writePointProjective(proof.aCommit)
        w.writePointProjective(proof.bCommit)
        w.writePointProjective(proof.cCommit)
        // Round 2
        w.writePointProjective(proof.zCommit)
        // Round 3
        w.writePointProjective(proof.tLoCommit)
        w.writePointProjective(proof.tMidCommit)
        w.writePointProjective(proof.tHiCommit)
        w.writePointProjectiveArray(proof.tExtraCommits)
        // Round 4 evaluations
        w.writeFr(proof.aEval)
        w.writeFr(proof.bEval)
        w.writeFr(proof.cEval)
        w.writeFr(proof.sigma1Eval)
        w.writeFr(proof.sigma2Eval)
        w.writeFr(proof.zOmegaEval)
        // Round 5 opening proofs
        w.writePointProjective(proof.openingProof)
        w.writePointProjective(proof.shiftedOpeningProof)
        let proofBytes = w.finalize()

        let inputs = proof.publicInputs.map { fr -> [UInt8] in
            let iw = ProofWriter()
            iw.writeFr(fr)
            return iw.finalize()
        }

        return UniversalProof(
            type: .plonk, version: 1,
            curveId: .bn254, fieldId: .bn254Fr,
            proofData: proofBytes, publicInputs: inputs,
            metadata: metadata
        )
    }

    /// Extract a PlonkProof from a universal proof container.
    func toPlonk() throws -> PlonkProof {
        guard type == .plonk else {
            throw UniversalProofError.codecNotRegistered(type)
        }
        let r = ProofReader(proofData)
        try r.expectLabel("PLONK-v1")
        let aCommit = try r.readPointProjective()
        let bCommit = try r.readPointProjective()
        let cCommit = try r.readPointProjective()
        let zCommit = try r.readPointProjective()
        let tLoCommit = try r.readPointProjective()
        let tMidCommit = try r.readPointProjective()
        let tHiCommit = try r.readPointProjective()
        let tExtraCommits = try r.readPointProjectiveArray()
        let aEval = try r.readFr()
        let bEval = try r.readFr()
        let cEval = try r.readFr()
        let sigma1Eval = try r.readFr()
        let sigma2Eval = try r.readFr()
        let zOmegaEval = try r.readFr()
        let openingProof = try r.readPointProjective()
        let shiftedOpeningProof = try r.readPointProjective()

        // Decode public inputs from the container
        var pubInputs = [Fr]()
        for inputBytes in publicInputs {
            let ir = ProofReader(inputBytes)
            pubInputs.append(try ir.readFr())
        }

        return PlonkProof(
            aCommit: aCommit, bCommit: bCommit, cCommit: cCommit,
            zCommit: zCommit,
            tLoCommit: tLoCommit, tMidCommit: tMidCommit, tHiCommit: tHiCommit,
            tExtraCommits: tExtraCommits,
            aEval: aEval, bEval: bEval, cEval: cEval,
            sigma1Eval: sigma1Eval, sigma2Eval: sigma2Eval,
            zOmegaEval: zOmegaEval,
            openingProof: openingProof,
            shiftedOpeningProof: shiftedOpeningProof,
            publicInputs: pubInputs
        )
    }
}

// MARK: - Internal helpers

private func writeU32(_ out: inout [UInt8], _ value: UInt32) {
    out.append(UInt8(value & 0xFF))
    out.append(UInt8((value >> 8) & 0xFF))
    out.append(UInt8((value >> 16) & 0xFF))
    out.append(UInt8((value >> 24) & 0xFF))
}

private func readU32(_ data: [UInt8], offset: Int) -> UInt32 {
    UInt32(data[offset]) |
    (UInt32(data[offset + 1]) << 8) |
    (UInt32(data[offset + 2]) << 16) |
    (UInt32(data[offset + 3]) << 24)
}
