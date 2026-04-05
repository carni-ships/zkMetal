// ProofEnvelope: Generic proof container for cross-system interoperability.
// Wraps any proof scheme (Groth16, Plonk, FRI/STARK) with metadata for
// routing, versioning, and type-safe deserialization.

import Foundation

// MARK: - Proof Scheme Identifier

/// Supported proof schemes.
public enum ProofScheme: String, Codable {
    case groth16
    case plonk
    case fri
    case kzg
    case ipa
    case stark
}

/// Supported curves.
public enum CurveIdentifier: String, Codable {
    case bn254
    case bn128   // alias for bn254 (snarkjs naming)
    case bls12_381 = "bls12-381"
    case bls12_377 = "bls12-377"
    case pallas
    case vesta
}

// MARK: - Proof Envelope

/// A generic proof envelope that wraps any proof with metadata.
/// The proof payload is stored as raw JSON (opaque to the envelope itself),
/// allowing scheme-specific deserialization by consumers.
public struct ProofEnvelope: Codable {
    /// Envelope format version for forward compatibility.
    public var version: Int

    /// The proof scheme used to generate this proof.
    public var scheme: ProofScheme

    /// The elliptic curve used.
    public var curve: CurveIdentifier

    /// The proof payload as a JSON object. Structure depends on `scheme`.
    /// For groth16: SnarkjsGroth16Proof format.
    /// For others: scheme-specific binary encoded as base64.
    public var proof: ProofPayload

    /// Public inputs as decimal string array.
    public var publicInputs: [String]

    /// Optional metadata (prover version, timestamp, etc.)
    public var metadata: [String: String]?

    public init(version: Int = 1, scheme: ProofScheme, curve: CurveIdentifier,
                proof: ProofPayload, publicInputs: [String],
                metadata: [String: String]? = nil) {
        self.version = version
        self.scheme = scheme
        self.curve = curve
        self.proof = proof
        self.publicInputs = publicInputs
        self.metadata = metadata
    }

    enum CodingKeys: String, CodingKey {
        case version, scheme, curve, proof
        case publicInputs = "public_inputs"
        case metadata
    }
}

// MARK: - Proof Payload

/// Proof payload: either structured JSON or base64-encoded binary.
public enum ProofPayload: Codable {
    /// Structured JSON proof (e.g., snarkjs Groth16 format).
    case json(SnarkjsGroth16Proof)

    /// Base64-encoded binary proof (for FRI, IPA, KZG, etc.).
    case binary(String)

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .json(let proof):
            try container.encode(proof)
        case .binary(let b64):
            try container.encode(["format": "binary", "data": b64])
        }
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        // Try structured Groth16 proof first
        if let proof = try? container.decode(SnarkjsGroth16Proof.self) {
            self = .json(proof)
            return
        }
        // Try binary format
        if let dict = try? container.decode([String: String].self),
           dict["format"] == "binary", let data = dict["data"] {
            self = .binary(data)
            return
        }
        throw DecodingError.dataCorruptedError(in: container,
            debugDescription: "Proof payload must be structured JSON or {\"format\":\"binary\",\"data\":\"...\"}")
    }
}

// MARK: - Envelope Builders

public extension ProofEnvelope {
    /// Create an envelope for a Groth16 proof with snarkjs-format payload.
    static func groth16(proof: Groth16Proof, publicInputs: [Fr],
                        curve: CurveIdentifier = .bn254,
                        metadata: [String: String]? = nil) -> ProofEnvelope {
        let sjProof = proof.toSnarkjs()
        let signals = publicInputs.map { frToDecimal($0) }
        return ProofEnvelope(
            scheme: .groth16, curve: curve,
            proof: .json(sjProof), publicInputs: signals,
            metadata: metadata
        )
    }

    /// Create an envelope wrapping a binary proof (FRI, IPA, KZG, etc.).
    static func binary(scheme: ProofScheme, proofBytes: [UInt8], publicInputs: [Fr],
                       curve: CurveIdentifier = .bn254,
                       metadata: [String: String]? = nil) -> ProofEnvelope {
        let b64 = Data(proofBytes).base64EncodedString()
        let signals = publicInputs.map { frToDecimal($0) }
        return ProofEnvelope(
            scheme: scheme, curve: curve,
            proof: .binary(b64), publicInputs: signals,
            metadata: metadata
        )
    }

    /// Encode the envelope to JSON data.
    func toJSON(prettyPrint: Bool = true) -> Data? {
        let encoder = JSONEncoder()
        if prettyPrint { encoder.outputFormatting = [.prettyPrinted, .sortedKeys] }
        return try? encoder.encode(self)
    }

    /// Decode an envelope from JSON data.
    static func fromJSON(_ data: Data) -> ProofEnvelope? {
        try? JSONDecoder().decode(ProofEnvelope.self, from: data)
    }

    /// Extract the Groth16 proof from a groth16-scheme envelope.
    func extractGroth16Proof() -> Groth16Proof? {
        guard scheme == .groth16, case .json(let sjProof) = proof else { return nil }
        return Groth16Proof.fromSnarkjs(sjProof)
    }

    /// Extract the binary proof bytes from a binary-payload envelope.
    func extractBinaryProof() -> [UInt8]? {
        guard case .binary(let b64) = proof,
              let data = Data(base64Encoded: b64) else { return nil }
        return Array(data)
    }

    /// Extract public inputs as Fr elements.
    func extractPublicInputs() -> [Fr]? {
        var result = [Fr]()
        result.reserveCapacity(publicInputs.count)
        for s in publicInputs {
            guard let fr = frFromDecimal(s) else { return nil }
            result.append(fr)
        }
        return result
    }
}
