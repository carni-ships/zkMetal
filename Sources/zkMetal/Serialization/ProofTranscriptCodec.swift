// ProofTranscriptCodec — Serialize/deserialize proof transcripts for interoperability
// with external verifiers (Ethereum Solidity, Gnark, SnarkJS, compact binary).
//
// Supported formats:
//   1. Ethereum ABI  — Solidity verifyProof()-compatible, big-endian 256-bit words
//   2. Gnark         — Big-endian, compressed points (SEC1 for BN254)
//   3. SnarkJS JSON  — Decimal string field elements, matches Circom/snarkjs schema
//   4. Binary Compact — Minimal size for on-chain submission (compressed points, varint lengths)
//
// Point compression uses SEC1 (BN254 G1: 33 bytes, G2: 65 bytes).
// All field elements are converted from Montgomery form before serialization.

import Foundation

// MARK: - Transcript Format Enum

/// Output format for proof transcript serialization.
public enum TranscriptFormat: String, CaseIterable {
    case ethereumABI   = "ethereum-abi"
    case gnark         = "gnark"
    case snarkjsJSON   = "snarkjs-json"
    case binaryCompact = "binary-compact"
}

// MARK: - Transcript Codec Errors

public enum TranscriptCodecError: Error, CustomStringConvertible {
    case unsupportedFormat(TranscriptFormat)
    case invalidData(String)
    case pointAtInfinity(String)
    case decompressFailed(String)
    case truncated(expected: Int, available: Int)
    case invalidMagic
    case jsonDecodingFailed(String)

    public var description: String {
        switch self {
        case .unsupportedFormat(let f):
            return "Unsupported transcript format: \(f.rawValue)"
        case .invalidData(let msg):
            return "Invalid transcript data: \(msg)"
        case .pointAtInfinity(let ctx):
            return "Point at infinity encountered: \(ctx)"
        case .decompressFailed(let ctx):
            return "Point decompression failed: \(ctx)"
        case .truncated(let expected, let available):
            return "Truncated data: need \(expected) bytes, have \(available)"
        case .invalidMagic:
            return "Invalid magic bytes for binary compact format"
        case .jsonDecodingFailed(let msg):
            return "JSON decoding failed: \(msg)"
        }
    }
}

// MARK: - Proof Transcript

/// A proof transcript holds the proof elements and public inputs in a format-agnostic way.
/// This is the intermediate representation used for cross-format conversion.
public struct ProofTranscript {
    /// The proof system type.
    public let system: ProofSystemType

    /// G1 points in the proof (affine coordinates).
    public let g1Points: [PointAffine]

    /// G2 points in the proof (affine coordinates).
    public let g2Points: [G2AffinePoint]

    /// Scalar field elements (evaluations, challenges).
    public let scalars: [Fr]

    /// Public inputs.
    public let publicInputs: [Fr]

    /// Optional metadata.
    public let metadata: [String: String]

    public init(system: ProofSystemType,
                g1Points: [PointAffine] = [],
                g2Points: [G2AffinePoint] = [],
                scalars: [Fr] = [],
                publicInputs: [Fr] = [],
                metadata: [String: String] = [:]) {
        self.system = system
        self.g1Points = g1Points
        self.g2Points = g2Points
        self.scalars = scalars
        self.publicInputs = publicInputs
        self.metadata = metadata
    }
}

// MARK: - Groth16 <-> ProofTranscript

public extension ProofTranscript {
    /// Create a transcript from a Groth16 proof.
    /// Layout: g1Points = [a, c], g2Points = [b], scalars = [], publicInputs as given.
    static func fromGroth16(_ proof: Groth16Proof, publicInputs: [Fr] = [],
                            metadata: [String: String] = [:]) -> ProofTranscript? {
        guard let aAff = pointToAffine(proof.a),
              let bAff = g2ToAffine(proof.b),
              let cAff = pointToAffine(proof.c) else { return nil }
        return ProofTranscript(
            system: .groth16,
            g1Points: [aAff, cAff],
            g2Points: [bAff],
            scalars: [],
            publicInputs: publicInputs,
            metadata: metadata
        )
    }

    /// Reconstruct a Groth16Proof from a transcript.
    func toGroth16() -> Groth16Proof? {
        guard system == .groth16, g1Points.count >= 2, g2Points.count >= 1 else { return nil }
        let a = PointProjective(x: g1Points[0].x, y: g1Points[0].y, z: .one)
        let bAff = g2Points[0]
        let b = G2ProjectivePoint(x: bAff.x, y: bAff.y, z: .one)
        let c = PointProjective(x: g1Points[1].x, y: g1Points[1].y, z: .one)
        return Groth16Proof(a: a, b: b, c: c)
    }
}

// MARK: - Plonk <-> ProofTranscript

public extension ProofTranscript {
    /// Create a transcript from a Plonk proof.
    /// Layout: g1Points = [a, b, c, z, tLo, tMid, tHi, ...extras, opening, shiftedOpening]
    ///         scalars = [aEval, bEval, cEval, sigma1Eval, sigma2Eval, zOmegaEval]
    static func fromPlonk(_ proof: PlonkProof,
                          metadata: [String: String] = [:]) -> ProofTranscript? {
        var g1s = [PointAffine]()
        let commits: [PointProjective] = [
            proof.aCommit, proof.bCommit, proof.cCommit,
            proof.zCommit,
            proof.tLoCommit, proof.tMidCommit, proof.tHiCommit
        ]
        for p in commits {
            guard let aff = pointToAffine(p) else { return nil }
            g1s.append(aff)
        }
        for p in proof.tExtraCommits {
            guard let aff = pointToAffine(p) else { return nil }
            g1s.append(aff)
        }
        guard let openAff = pointToAffine(proof.openingProof),
              let shiftAff = pointToAffine(proof.shiftedOpeningProof) else { return nil }
        g1s.append(openAff)
        g1s.append(shiftAff)

        let scalars = [proof.aEval, proof.bEval, proof.cEval,
                       proof.sigma1Eval, proof.sigma2Eval, proof.zOmegaEval]

        return ProofTranscript(
            system: .plonk,
            g1Points: g1s,
            g2Points: [],
            scalars: scalars,
            publicInputs: proof.publicInputs,
            metadata: metadata
        )
    }
}

// MARK: - ProofTranscriptCodec

/// Codec engine for serializing/deserializing proof transcripts across multiple formats.
public struct ProofTranscriptCodec {
    private init() {}

    // MARK: - Multi-format serialize

    /// Serialize a proof transcript to the specified format.
    public static func serialize(_ transcript: ProofTranscript,
                                 format: TranscriptFormat) throws -> [UInt8] {
        switch format {
        case .ethereumABI:
            return try serializeEthereumABI(transcript)
        case .gnark:
            return try serializeGnark(transcript)
        case .snarkjsJSON:
            return try serializeSnarkjsJSON(transcript)
        case .binaryCompact:
            return try serializeBinaryCompact(transcript)
        }
    }

    /// Deserialize a proof transcript from the specified format.
    public static func deserialize(_ data: [UInt8], format: TranscriptFormat,
                                   system: ProofSystemType) throws -> ProofTranscript {
        switch format {
        case .ethereumABI:
            return try deserializeEthereumABI(data, system: system)
        case .gnark:
            return try deserializeGnark(data, system: system)
        case .snarkjsJSON:
            return try deserializeSnarkjsJSON(data, system: system)
        case .binaryCompact:
            return try deserializeBinaryCompact(data)
        }
    }

    /// Convert a transcript between formats.
    public static func convert(_ data: [UInt8], from: TranscriptFormat, to: TranscriptFormat,
                               system: ProofSystemType) throws -> [UInt8] {
        let transcript = try deserialize(data, format: from, system: system)
        return try serialize(transcript, format: to)
    }

    // MARK: - Batch Encoding

    /// Encode multiple proof transcripts into a single binary blob.
    /// Format: [count: u32 LE] [len0: u32 LE] [data0] [len1: u32 LE] [data1] ...
    public static func batchEncode(_ transcripts: [ProofTranscript],
                                   format: TranscriptFormat) throws -> [UInt8] {
        var out = [UInt8]()
        out.reserveCapacity(transcripts.count * 512)
        appendU32LE(&out, UInt32(transcripts.count))
        for transcript in transcripts {
            let encoded = try serialize(transcript, format: format)
            appendU32LE(&out, UInt32(encoded.count))
            out.append(contentsOf: encoded)
        }
        return out
    }

    /// Decode multiple proof transcripts from a batch blob.
    public static func batchDecode(_ data: [UInt8], format: TranscriptFormat,
                                   system: ProofSystemType) throws -> [ProofTranscript] {
        guard data.count >= 4 else {
            throw TranscriptCodecError.truncated(expected: 4, available: data.count)
        }
        let count = Int(readU32LE(data, offset: 0))
        var offset = 4
        var results = [ProofTranscript]()
        results.reserveCapacity(count)
        for _ in 0..<count {
            guard offset + 4 <= data.count else {
                throw TranscriptCodecError.truncated(expected: offset + 4, available: data.count)
            }
            let len = Int(readU32LE(data, offset: offset)); offset += 4
            guard offset + len <= data.count else {
                throw TranscriptCodecError.truncated(expected: offset + len, available: data.count)
            }
            let chunk = Array(data[offset..<(offset + len)]); offset += len
            results.append(try deserialize(chunk, format: format, system: system))
        }
        return results
    }
}

// MARK: - BN254 G1 Point Compression (SEC1)

/// Compress a BN254 G1 affine point to 33 bytes (0x02/0x03 prefix + 32-byte BE x).
public func transcriptG1Compress(_ p: PointAffine) -> [UInt8] {
    return bn254G1CompressAffine(p)
}

/// Decompress a BN254 G1 point from 33 compressed bytes.
public func transcriptG1Decompress(_ data: [UInt8]) throws -> PointAffine {
    guard let proj = bn254G1Decompress(data) else {
        throw TranscriptCodecError.decompressFailed("BN254 G1")
    }
    guard let aff = pointToAffine(proj) else {
        throw TranscriptCodecError.pointAtInfinity("BN254 G1 decompressed to identity")
    }
    return aff
}

// MARK: - BN254 G2 Point Compression

/// Compress a BN254 G2 affine point to 65 bytes.
/// Layout: [prefix: 1 byte] [x.c1: 32 bytes BE] [x.c0: 32 bytes BE]
/// Prefix: 0x0a (y even parity) or 0x0b (y odd parity).
/// Parity is determined by the c1 component of y; if c1 == 0, use c0 parity.
public func transcriptG2Compress(_ p: G2AffinePoint) -> [UInt8] {
    let yc1Limbs = fpToInt(p.y.c1)
    let yc0Limbs = fpToInt(p.y.c0)
    let c1IsZero = yc1Limbs.allSatisfy { $0 == 0 }
    let yOdd: Bool
    if c1IsZero {
        yOdd = yc0Limbs[0] & 1 == 1
    } else {
        yOdd = yc1Limbs[0] & 1 == 1
    }

    var result = [UInt8](repeating: 0, count: 65)
    result[0] = yOdd ? 0x0b : 0x0a

    // x.c1 in big-endian (bytes 1..32)
    let xc1Limbs = fpToInt(p.x.c1)
    for i in 0..<4 {
        let limb = xc1Limbs[3 - i]
        let base = 1 + i * 8
        result[base + 0] = UInt8((limb >> 56) & 0xFF)
        result[base + 1] = UInt8((limb >> 48) & 0xFF)
        result[base + 2] = UInt8((limb >> 40) & 0xFF)
        result[base + 3] = UInt8((limb >> 32) & 0xFF)
        result[base + 4] = UInt8((limb >> 24) & 0xFF)
        result[base + 5] = UInt8((limb >> 16) & 0xFF)
        result[base + 6] = UInt8((limb >> 8) & 0xFF)
        result[base + 7] = UInt8(limb & 0xFF)
    }

    // x.c0 in big-endian (bytes 33..64)
    let xc0Limbs = fpToInt(p.x.c0)
    for i in 0..<4 {
        let limb = xc0Limbs[3 - i]
        let base = 33 + i * 8
        result[base + 0] = UInt8((limb >> 56) & 0xFF)
        result[base + 1] = UInt8((limb >> 48) & 0xFF)
        result[base + 2] = UInt8((limb >> 40) & 0xFF)
        result[base + 3] = UInt8((limb >> 32) & 0xFF)
        result[base + 4] = UInt8((limb >> 24) & 0xFF)
        result[base + 5] = UInt8((limb >> 16) & 0xFF)
        result[base + 6] = UInt8((limb >> 8) & 0xFF)
        result[base + 7] = UInt8(limb & 0xFF)
    }

    return result
}

/// Decompress a BN254 G2 point from 65 bytes.
public func transcriptG2Decompress(_ data: [UInt8]) throws -> G2AffinePoint {
    guard data.count == 65 else {
        throw TranscriptCodecError.invalidData("G2 compressed data must be 65 bytes, got \(data.count)")
    }
    let prefix = data[0]
    guard prefix == 0x0a || prefix == 0x0b else {
        throw TranscriptCodecError.invalidData("Invalid G2 prefix: 0x\(String(format: "%02x", prefix))")
    }
    let wantOdd = prefix == 0x0b

    // Parse x.c1 from bytes 1..32 (big-endian)
    let xc1Limbs = parseBE32Limbs(Array(data[1..<33]))
    // Parse x.c0 from bytes 33..64 (big-endian)
    let xc0Limbs = parseBE32Limbs(Array(data[33..<65]))

    // Convert to Montgomery form
    let xc1Raw = Fp.from64(xc1Limbs)
    let xc1 = fpMul(xc1Raw, Fp.from64(Fp.R2_MOD_P))
    let xc0Raw = Fp.from64(xc0Limbs)
    let xc0 = fpMul(xc0Raw, Fp.from64(Fp.R2_MOD_P))

    let x = Fp2(c0: xc0, c1: xc1)

    // y^2 = x^3 + B' where B' = 3/(9+u) for BN254 twist
    // BN254 twist: y^2 = x^3 + b/xi where xi = 9+u, b = 3
    // b' = 3 * inverse(9+u) in Fp2
    // Precomputed: b' = Fp2(c0: bTwistC0, c1: bTwistC1)
    let x2 = fp2Sqr(x)
    let x3 = fp2Mul(x2, x)
    let rhs = fp2Add(x3, bn254TwistB())

    guard let y = fp2Sqrt(rhs) else {
        throw TranscriptCodecError.decompressFailed("BN254 G2: no sqrt for y^2")
    }

    // Check parity
    let yc1Limbs = fpToInt(y.c1)
    let yc0Limbs = fpToInt(y.c0)
    let c1IsZero = yc1Limbs.allSatisfy { $0 == 0 }
    let isOdd: Bool
    if c1IsZero {
        isOdd = yc0Limbs[0] & 1 == 1
    } else {
        isOdd = yc1Limbs[0] & 1 == 1
    }

    let finalY = isOdd != wantOdd ? fp2Neg(y) : y
    return G2AffinePoint(x: x, y: finalY)
}

/// BN254 twist parameter b' = 3/(9+u) in Fp2.
private func bn254TwistB() -> Fp2 {
    // For BN254, the sextic twist has b' = b/xi where b=3, xi=9+u
    // We compute inverse(xi) * 3
    let xi = Fp2(c0: fpFromInt(9), c1: .one)
    let xiInv = fp2Inverse(xi)
    let three = Fp2(c0: fpFromInt(3), c1: .zero)
    return fp2Mul(three, xiInv)
}

/// Square root in Fp2 = Fp[u]/(u^2+1) for BN254 (p = 3 mod 4).
private func fp2Sqrt(_ a: Fp2) -> Fp2? {
    if fp2IsZero(a) { return .zero }

    // If c1 == 0, try sqrt(c0) or sqrt(-c0)*u
    if fpIsZero(a.c1) {
        if let s = fpSqrt(a.c0) {
            return Fp2(c0: s, c1: .zero)
        }
        if let s = fpSqrt(fpNeg(a.c0)) {
            return Fp2(c0: .zero, c1: s)
        }
        return nil
    }

    // General: norm = c0^2 + c1^2
    let norm = fpAdd(fpSqr(a.c0), fpSqr(a.c1))
    guard let sqrtNorm = fpSqrt(norm) else { return nil }

    let twoInv = fpInverse(fpFromInt(2))
    var alpha = fpMul(fpAdd(a.c0, sqrtNorm), twoInv)

    var sqrtAlpha = fpSqrt(alpha)
    if sqrtAlpha == nil {
        alpha = fpMul(fpSub(a.c0, sqrtNorm), twoInv)
        sqrtAlpha = fpSqrt(alpha)
    }
    guard let x0 = sqrtAlpha else { return nil }

    let x1 = fpMul(a.c1, fpInverse(fpMul(x0, fpFromInt(2))))
    let candidate = Fp2(c0: x0, c1: x1)

    // Verify
    let check = fp2Sqr(candidate)
    let diff = fp2Sub(check, a)
    guard fp2IsZero(diff) else { return nil }

    return candidate
}

/// Check if Fp2 element is zero.
private func fp2IsZero(_ a: Fp2) -> Bool {
    fpIsZero(a.c0) && fpIsZero(a.c1)
}

/// Check if Fp element is zero.
private func fpIsZero(_ a: Fp) -> Bool {
    a.v.0 == 0 && a.v.1 == 0 && a.v.2 == 0 && a.v.3 == 0 &&
    a.v.4 == 0 && a.v.5 == 0 && a.v.6 == 0 && a.v.7 == 0
}

// MARK: - Ethereum ABI Format

extension ProofTranscriptCodec {

    /// Serialize a Groth16 transcript to Ethereum ABI format.
    /// Layout: [a.x: 32] [a.y: 32] [b.x.c1: 32] [b.x.c0: 32] [b.y.c1: 32] [b.y.c0: 32]
    ///         [c.x: 32] [c.y: 32] [input0: 32] [input1: 32] ...
    static func serializeEthereumABI(_ t: ProofTranscript) throws -> [UInt8] {
        guard t.system == .groth16 else {
            throw TranscriptCodecError.unsupportedFormat(.ethereumABI)
        }
        guard t.g1Points.count >= 2, t.g2Points.count >= 1 else {
            throw TranscriptCodecError.invalidData("Groth16 requires 2 G1 + 1 G2 points")
        }

        let a = t.g1Points[0]
        let b = t.g2Points[0]
        let c = t.g1Points[1]

        var out = [UInt8]()
        let wordCount = 2 + 4 + 2 + t.publicInputs.count
        out.reserveCapacity(wordCount * 32)

        // a: [x, y]
        out.append(contentsOf: fpToBE32(fpToInt(a.x)))
        out.append(contentsOf: fpToBE32(fpToInt(a.y)))

        // b: [[x.c1, x.c0], [y.c1, y.c0]] (Ethereum convention: imaginary first)
        out.append(contentsOf: fpToBE32(fpToInt(b.x.c1)))
        out.append(contentsOf: fpToBE32(fpToInt(b.x.c0)))
        out.append(contentsOf: fpToBE32(fpToInt(b.y.c1)))
        out.append(contentsOf: fpToBE32(fpToInt(b.y.c0)))

        // c: [x, y]
        out.append(contentsOf: fpToBE32(fpToInt(c.x)))
        out.append(contentsOf: fpToBE32(fpToInt(c.y)))

        // public inputs
        for input in t.publicInputs {
            out.append(contentsOf: frToBE32(frToInt(input)))
        }

        return out
    }

    /// Deserialize a Groth16 transcript from Ethereum ABI format.
    static func deserializeEthereumABI(_ data: [UInt8],
                                       system: ProofSystemType) throws -> ProofTranscript {
        guard system == .groth16 else {
            throw TranscriptCodecError.unsupportedFormat(.ethereumABI)
        }
        // Minimum: 8 words (a + b + c) = 256 bytes
        guard data.count >= 256 else {
            throw TranscriptCodecError.truncated(expected: 256, available: data.count)
        }

        var off = 0
        func next32() -> [UInt8] {
            let w = Array(data[off..<(off + 32)]); off += 32; return w
        }

        let ax = be32ToFpMontgomery(next32())
        let ay = be32ToFpMontgomery(next32())
        let bxc1 = be32ToFpMontgomery(next32())
        let bxc0 = be32ToFpMontgomery(next32())
        let byc1 = be32ToFpMontgomery(next32())
        let byc0 = be32ToFpMontgomery(next32())
        let cx = be32ToFpMontgomery(next32())
        let cy = be32ToFpMontgomery(next32())

        let a = PointAffine(x: ax, y: ay)
        let b = G2AffinePoint(x: Fp2(c0: bxc0, c1: bxc1), y: Fp2(c0: byc0, c1: byc1))
        let c = PointAffine(x: cx, y: cy)

        // Remaining 32-byte words are public inputs
        let numInputs = (data.count - 256) / 32
        var inputs = [Fr]()
        inputs.reserveCapacity(numInputs)
        for _ in 0..<numInputs {
            inputs.append(be32ToFrMontgomery(next32()))
        }

        return ProofTranscript(system: .groth16, g1Points: [a, c],
                               g2Points: [b], publicInputs: inputs)
    }
}

// MARK: - Gnark Format (big-endian, compressed points)

extension ProofTranscriptCodec {

    /// Serialize to Gnark-compatible format.
    /// Gnark uses big-endian, compressed points for G1/G2, and big-endian scalars.
    /// Layout: [numG1: u32 BE] [g1_0: 33] ... [numG2: u32 BE] [g2_0: 65] ...
    ///         [numScalars: u32 BE] [s_0: 32] ... [numInputs: u32 BE] [i_0: 32] ...
    static func serializeGnark(_ t: ProofTranscript) throws -> [UInt8] {
        var out = [UInt8]()
        out.reserveCapacity(256)

        // G1 points (compressed)
        appendU32BE(&out, UInt32(t.g1Points.count))
        for p in t.g1Points {
            out.append(contentsOf: transcriptG1Compress(p))
        }

        // G2 points (compressed)
        appendU32BE(&out, UInt32(t.g2Points.count))
        for p in t.g2Points {
            out.append(contentsOf: transcriptG2Compress(p))
        }

        // Scalars (big-endian)
        appendU32BE(&out, UInt32(t.scalars.count))
        for s in t.scalars {
            out.append(contentsOf: frToBE32(frToInt(s)))
        }

        // Public inputs (big-endian)
        appendU32BE(&out, UInt32(t.publicInputs.count))
        for inp in t.publicInputs {
            out.append(contentsOf: frToBE32(frToInt(inp)))
        }

        return out
    }

    /// Deserialize from Gnark-compatible format.
    static func deserializeGnark(_ data: [UInt8],
                                 system: ProofSystemType) throws -> ProofTranscript {
        var off = 0

        func needBytes(_ n: Int) throws {
            guard off + n <= data.count else {
                throw TranscriptCodecError.truncated(expected: off + n, available: data.count)
            }
        }

        func readU32BE() throws -> UInt32 {
            try needBytes(4)
            let v = (UInt32(data[off]) << 24) | (UInt32(data[off+1]) << 16) |
                    (UInt32(data[off+2]) << 8) | UInt32(data[off+3])
            off += 4
            return v
        }

        // G1 points
        let numG1 = Int(try readU32BE())
        var g1Points = [PointAffine]()
        g1Points.reserveCapacity(numG1)
        for _ in 0..<numG1 {
            try needBytes(33)
            let chunk = Array(data[off..<(off + 33)]); off += 33
            g1Points.append(try transcriptG1Decompress(chunk))
        }

        // G2 points
        let numG2 = Int(try readU32BE())
        var g2Points = [G2AffinePoint]()
        g2Points.reserveCapacity(numG2)
        for _ in 0..<numG2 {
            try needBytes(65)
            let chunk = Array(data[off..<(off + 65)]); off += 65
            g2Points.append(try transcriptG2Decompress(chunk))
        }

        // Scalars
        let numScalars = Int(try readU32BE())
        var scalars = [Fr]()
        scalars.reserveCapacity(numScalars)
        for _ in 0..<numScalars {
            try needBytes(32)
            let w = Array(data[off..<(off + 32)]); off += 32
            scalars.append(be32ToFrMontgomery(w))
        }

        // Public inputs
        let numInputs = Int(try readU32BE())
        var inputs = [Fr]()
        inputs.reserveCapacity(numInputs)
        for _ in 0..<numInputs {
            try needBytes(32)
            let w = Array(data[off..<(off + 32)]); off += 32
            inputs.append(be32ToFrMontgomery(w))
        }

        return ProofTranscript(system: system, g1Points: g1Points, g2Points: g2Points,
                               scalars: scalars, publicInputs: inputs)
    }
}

// MARK: - SnarkJS JSON Format

/// Internal JSON model for transcript serialization.
private struct SnarkjsTranscriptJSON: Codable {
    var protocol_type: String
    var curve: String
    var pi_a: [String]?
    var pi_b: [[String]]?
    var pi_c: [String]?
    var commitments: [[String]]?
    var evaluations: [String]?
    var publicInputs: [String]

    enum CodingKeys: String, CodingKey {
        case protocol_type = "protocol"
        case curve
        case pi_a, pi_b, pi_c
        case commitments, evaluations
        case publicInputs = "public_inputs"
    }
}

extension ProofTranscriptCodec {

    /// Serialize to SnarkJS-compatible JSON format.
    static func serializeSnarkjsJSON(_ t: ProofTranscript) throws -> [UInt8] {
        var json = SnarkjsTranscriptJSON(
            protocol_type: t.system.description,
            curve: "bn128",
            publicInputs: t.publicInputs.map { frToDecimal($0) }
        )

        if t.system == .groth16 && t.g1Points.count >= 2 && t.g2Points.count >= 1 {
            let a = t.g1Points[0]
            let b = t.g2Points[0]
            let c = t.g1Points[1]
            json.pi_a = [fpToDecimal(a.x), fpToDecimal(a.y), "1"]
            json.pi_b = [
                [fpToDecimal(b.x.c0), fpToDecimal(b.x.c1)],
                [fpToDecimal(b.y.c0), fpToDecimal(b.y.c1)],
                ["1", "0"]
            ]
            json.pi_c = [fpToDecimal(c.x), fpToDecimal(c.y), "1"]
        } else {
            // Generic: encode all G1 points as commitments, scalars as evaluations
            json.commitments = t.g1Points.map { p in
                [fpToDecimal(p.x), fpToDecimal(p.y), "1"]
            }
            json.evaluations = t.scalars.map { frToDecimal($0) }
        }

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(json) else {
            throw TranscriptCodecError.invalidData("JSON encoding failed")
        }
        return Array(data)
    }

    /// Deserialize from SnarkJS-compatible JSON format.
    static func deserializeSnarkjsJSON(_ data: [UInt8],
                                       system: ProofSystemType) throws -> ProofTranscript {
        guard let jsonData = String(bytes: data, encoding: .utf8)?.data(using: .utf8) else {
            throw TranscriptCodecError.jsonDecodingFailed("invalid UTF-8")
        }

        let json: SnarkjsTranscriptJSON
        do {
            json = try JSONDecoder().decode(SnarkjsTranscriptJSON.self, from: jsonData)
        } catch {
            throw TranscriptCodecError.jsonDecodingFailed(error.localizedDescription)
        }

        var g1Points = [PointAffine]()
        var g2Points = [G2AffinePoint]()
        var scalars = [Fr]()

        if system == .groth16 {
            if let pia = json.pi_a, pia.count >= 2,
               let pib = json.pi_b, pib.count >= 2,
               let pic = json.pi_c, pic.count >= 2 {
                guard let ax = fpFromDecimal(pia[0]), let ay = fpFromDecimal(pia[1]) else {
                    throw TranscriptCodecError.invalidData("Cannot parse pi_a")
                }
                g1Points.append(PointAffine(x: ax, y: ay))

                guard pib[0].count >= 2, pib[1].count >= 2,
                      let bxc0 = fpFromDecimal(pib[0][0]), let bxc1 = fpFromDecimal(pib[0][1]),
                      let byc0 = fpFromDecimal(pib[1][0]), let byc1 = fpFromDecimal(pib[1][1]) else {
                    throw TranscriptCodecError.invalidData("Cannot parse pi_b")
                }
                g2Points.append(G2AffinePoint(x: Fp2(c0: bxc0, c1: bxc1),
                                              y: Fp2(c0: byc0, c1: byc1)))

                guard let cx = fpFromDecimal(pic[0]), let cy = fpFromDecimal(pic[1]) else {
                    throw TranscriptCodecError.invalidData("Cannot parse pi_c")
                }
                g1Points.append(PointAffine(x: cx, y: cy))
            }
        } else {
            // Generic: commitments as G1 points
            if let commits = json.commitments {
                for c in commits {
                    guard c.count >= 2, let x = fpFromDecimal(c[0]), let y = fpFromDecimal(c[1]) else {
                        throw TranscriptCodecError.invalidData("Cannot parse commitment")
                    }
                    g1Points.append(PointAffine(x: x, y: y))
                }
            }
            if let evals = json.evaluations {
                for e in evals {
                    guard let fr = frFromDecimal(e) else {
                        throw TranscriptCodecError.invalidData("Cannot parse evaluation")
                    }
                    scalars.append(fr)
                }
            }
        }

        var inputs = [Fr]()
        for s in json.publicInputs {
            guard let fr = frFromDecimal(s) else {
                throw TranscriptCodecError.invalidData("Cannot parse public input: \(s)")
            }
            inputs.append(fr)
        }

        return ProofTranscript(system: system, g1Points: g1Points, g2Points: g2Points,
                               scalars: scalars, publicInputs: inputs)
    }
}

// MARK: - Binary Compact Format

/// Binary compact format for minimal on-chain proof size.
/// Uses compressed points and varint-style length encoding.
/// Header: [magic: "ZKTX" 4 bytes] [version: u8] [system: u8] [flags: u8]
/// Body: [numG1: u16 LE] [g1_compressed...] [numG2: u16 LE] [g2_compressed...]
///       [numScalars: u16 LE] [scalars BE 32 each] [numInputs: u16 LE] [inputs BE 32 each]

private let compactMagic: [UInt8] = [0x5A, 0x4B, 0x54, 0x58] // "ZKTX"
private let compactVersion: UInt8 = 1

extension ProofTranscriptCodec {

    static func serializeBinaryCompact(_ t: ProofTranscript) throws -> [UInt8] {
        var out = [UInt8]()
        out.reserveCapacity(256)

        // Header
        out.append(contentsOf: compactMagic)
        out.append(compactVersion)
        out.append(t.system.rawValue)
        out.append(0) // flags reserved

        // G1 compressed
        appendU16LE(&out, UInt16(t.g1Points.count))
        for p in t.g1Points {
            out.append(contentsOf: transcriptG1Compress(p))
        }

        // G2 compressed
        appendU16LE(&out, UInt16(t.g2Points.count))
        for p in t.g2Points {
            out.append(contentsOf: transcriptG2Compress(p))
        }

        // Scalars
        appendU16LE(&out, UInt16(t.scalars.count))
        for s in t.scalars {
            out.append(contentsOf: frToBE32(frToInt(s)))
        }

        // Public inputs
        appendU16LE(&out, UInt16(t.publicInputs.count))
        for inp in t.publicInputs {
            out.append(contentsOf: frToBE32(frToInt(inp)))
        }

        return out
    }

    static func deserializeBinaryCompact(_ data: [UInt8]) throws -> ProofTranscript {
        guard data.count >= 7 else {
            throw TranscriptCodecError.truncated(expected: 7, available: data.count)
        }

        // Verify magic
        guard data[0] == compactMagic[0] && data[1] == compactMagic[1] &&
              data[2] == compactMagic[2] && data[3] == compactMagic[3] else {
            throw TranscriptCodecError.invalidMagic
        }

        // Version check
        guard data[4] <= compactVersion else {
            throw TranscriptCodecError.invalidData("Unsupported compact version: \(data[4])")
        }

        guard let system = ProofSystemType(rawValue: data[5]) else {
            throw TranscriptCodecError.invalidData("Unknown proof system type: \(data[5])")
        }

        var off = 7

        func needBytes(_ n: Int) throws {
            guard off + n <= data.count else {
                throw TranscriptCodecError.truncated(expected: off + n, available: data.count)
            }
        }

        func readU16() throws -> UInt16 {
            try needBytes(2)
            let v = UInt16(data[off]) | (UInt16(data[off + 1]) << 8)
            off += 2
            return v
        }

        // G1 points
        let numG1 = Int(try readU16())
        var g1Points = [PointAffine]()
        g1Points.reserveCapacity(numG1)
        for _ in 0..<numG1 {
            try needBytes(33)
            let chunk = Array(data[off..<(off + 33)]); off += 33
            g1Points.append(try transcriptG1Decompress(chunk))
        }

        // G2 points
        let numG2 = Int(try readU16())
        var g2Points = [G2AffinePoint]()
        g2Points.reserveCapacity(numG2)
        for _ in 0..<numG2 {
            try needBytes(65)
            let chunk = Array(data[off..<(off + 65)]); off += 65
            g2Points.append(try transcriptG2Decompress(chunk))
        }

        // Scalars
        let numScalars = Int(try readU16())
        var scalars = [Fr]()
        scalars.reserveCapacity(numScalars)
        for _ in 0..<numScalars {
            try needBytes(32)
            let w = Array(data[off..<(off + 32)]); off += 32
            scalars.append(be32ToFrMontgomery(w))
        }

        // Public inputs
        let numInputs = Int(try readU16())
        var inputs = [Fr]()
        inputs.reserveCapacity(numInputs)
        for _ in 0..<numInputs {
            try needBytes(32)
            let w = Array(data[off..<(off + 32)]); off += 32
            inputs.append(be32ToFrMontgomery(w))
        }

        return ProofTranscript(system: system, g1Points: g1Points, g2Points: g2Points,
                               scalars: scalars, publicInputs: inputs)
    }
}

// MARK: - Internal Helpers

/// Encode 4 UInt64 limbs (little-endian) as 32-byte big-endian.
private func fpToBE32(_ limbs: [UInt64]) -> [UInt8] {
    var bytes = [UInt8](repeating: 0, count: 32)
    for i in 0..<4 {
        let limb = limbs[3 - i]
        let base = i * 8
        bytes[base + 0] = UInt8((limb >> 56) & 0xFF)
        bytes[base + 1] = UInt8((limb >> 48) & 0xFF)
        bytes[base + 2] = UInt8((limb >> 40) & 0xFF)
        bytes[base + 3] = UInt8((limb >> 32) & 0xFF)
        bytes[base + 4] = UInt8((limb >> 24) & 0xFF)
        bytes[base + 5] = UInt8((limb >> 16) & 0xFF)
        bytes[base + 6] = UInt8((limb >> 8) & 0xFF)
        bytes[base + 7] = UInt8(limb & 0xFF)
    }
    return bytes
}

/// Alias: Fr limbs to BE 32.
private func frToBE32(_ limbs: [UInt64]) -> [UInt8] {
    fpToBE32(limbs)
}

/// Parse 32 big-endian bytes into 4 UInt64 limbs (little-endian).
private func parseBE32Limbs(_ bytes: [UInt8]) -> [UInt64] {
    var limbs = [UInt64](repeating: 0, count: 4)
    for i in 0..<4 {
        let base = (3 - i) * 8
        limbs[i] = (UInt64(bytes[base]) << 56) | (UInt64(bytes[base + 1]) << 48) |
                   (UInt64(bytes[base + 2]) << 40) | (UInt64(bytes[base + 3]) << 32) |
                   (UInt64(bytes[base + 4]) << 24) | (UInt64(bytes[base + 5]) << 16) |
                   (UInt64(bytes[base + 6]) << 8) | UInt64(bytes[base + 7])
    }
    return limbs
}

/// Decode 32-byte BE word to Fp in Montgomery form.
private func be32ToFpMontgomery(_ bytes: [UInt8]) -> Fp {
    let limbs = parseBE32Limbs(bytes)
    let raw = Fp.from64(limbs)
    return fpMul(raw, Fp.from64(Fp.R2_MOD_P))
}

/// Decode 32-byte BE word to Fr in Montgomery form.
private func be32ToFrMontgomery(_ bytes: [UInt8]) -> Fr {
    let limbs = parseBE32Limbs(bytes)
    let raw = Fr.from64(limbs)
    return frMul(raw, Fr.from64(Fr.R2_MOD_R))
}

/// Append a u32 in little-endian.
private func appendU32LE(_ out: inout [UInt8], _ v: UInt32) {
    out.append(UInt8(v & 0xFF))
    out.append(UInt8((v >> 8) & 0xFF))
    out.append(UInt8((v >> 16) & 0xFF))
    out.append(UInt8((v >> 24) & 0xFF))
}

/// Append a u32 in big-endian.
private func appendU32BE(_ out: inout [UInt8], _ v: UInt32) {
    out.append(UInt8((v >> 24) & 0xFF))
    out.append(UInt8((v >> 16) & 0xFF))
    out.append(UInt8((v >> 8) & 0xFF))
    out.append(UInt8(v & 0xFF))
}

/// Read a u32 in little-endian.
private func readU32LE(_ data: [UInt8], offset: Int) -> UInt32 {
    UInt32(data[offset]) | (UInt32(data[offset + 1]) << 8) |
    (UInt32(data[offset + 2]) << 16) | (UInt32(data[offset + 3]) << 24)
}

/// Append a u16 in little-endian.
private func appendU16LE(_ out: inout [UInt8], _ v: UInt16) {
    out.append(UInt8(v & 0xFF))
    out.append(UInt8((v >> 8) & 0xFF))
}
