// ProofRegistry — Extensible codec registry for universal proof serialization.
//
// Provides a protocol-based system for registering custom proof encoders/decoders,
// enabling new proof types to plug into the UniversalProof format without modifying
// the core serialization code.

import Foundation

// MARK: - Codec Protocol

/// Protocol for encoding/decoding a specific proof type to/from raw bytes.
/// Implement this for custom proof types and register with ProofRegistry.
public protocol ProofCodec {
    /// The proof system type this codec handles.
    var proofType: ProofSystemType { get }

    /// Encode a proof value into raw bytes for storage in UniversalProof.proofData.
    func encode(_ proof: Any) throws -> [UInt8]

    /// Decode raw bytes back into a proof value.
    func decode(_ data: [UInt8]) throws -> Any

    /// Default curve for this proof type (override if meaningful).
    var defaultCurveId: CurveId { get }

    /// Default field for this proof type (override if meaningful).
    var defaultFieldId: FieldId { get }
}

public extension ProofCodec {
    var defaultCurveId: CurveId { .none }
    var defaultFieldId: FieldId { .none }
}

// MARK: - Registry Errors

public enum ProofRegistryError: Error, CustomStringConvertible {
    case codecAlreadyRegistered(ProofSystemType)
    case noCodecRegistered(ProofSystemType)
    case typeMismatch(expected: String, got: String)

    public var description: String {
        switch self {
        case .codecAlreadyRegistered(let t):
            return "Codec already registered for \(t)"
        case .noCodecRegistered(let t):
            return "No codec registered for \(t)"
        case .typeMismatch(let expected, let got):
            return "Type mismatch: expected \(expected), got \(got)"
        }
    }
}

// MARK: - ProofRegistry

/// A registry of proof codecs keyed by ProofSystemType.
/// Thread-safe: uses a serial queue for mutation.
public final class ProofRegistry {

    /// Shared global registry with built-in codecs pre-registered.
    public static let shared: ProofRegistry = {
        let registry = ProofRegistry()
        // Register built-in codecs
        try? registry.register(Groth16Codec())
        try? registry.register(PlonkCodec())
        try? registry.register(KZGCodec())
        try? registry.register(IPACodec())
        return registry
    }()

    private var codecs: [ProofSystemType: ProofCodec] = [:]
    private let queue = DispatchQueue(label: "com.zkMetal.ProofRegistry")

    public init() {}

    /// Register a codec. Throws if one is already registered for that type.
    public func register(_ codec: ProofCodec) throws {
        try queue.sync {
            if codecs[codec.proofType] != nil {
                throw ProofRegistryError.codecAlreadyRegistered(codec.proofType)
            }
            codecs[codec.proofType] = codec
        }
    }

    /// Replace an existing codec (or register a new one).
    public func registerOrReplace(_ codec: ProofCodec) {
        queue.sync {
            codecs[codec.proofType] = codec
        }
    }

    /// Look up a codec by proof type.
    public func codec(for type: ProofSystemType) -> ProofCodec? {
        queue.sync { codecs[type] }
    }

    /// Encode any proof value using the registered codec.
    /// Returns a UniversalProof ready for serialization.
    public func encode(_ proof: Any, type: ProofSystemType,
                       publicInputs: [[UInt8]] = [],
                       metadata: [String: String] = [:]) throws -> UniversalProof {
        guard let codec = queue.sync(execute: { codecs[type] }) else {
            throw ProofRegistryError.noCodecRegistered(type)
        }
        let data = try codec.encode(proof)
        return UniversalProof(
            type: type, version: 1,
            curveId: codec.defaultCurveId,
            fieldId: codec.defaultFieldId,
            proofData: data, publicInputs: publicInputs,
            metadata: metadata
        )
    }

    /// Decode a UniversalProof back into a typed proof value.
    public func decode(_ universal: UniversalProof) throws -> Any {
        guard let codec = queue.sync(execute: { codecs[universal.type] }) else {
            throw ProofRegistryError.noCodecRegistered(universal.type)
        }
        return try codec.decode(universal.proofData)
    }

    /// All registered proof types.
    public var registeredTypes: [ProofSystemType] {
        queue.sync { Array(codecs.keys).sorted(by: { $0.rawValue < $1.rawValue }) }
    }
}

// MARK: - Built-in Codecs

/// Codec for Groth16Proof (BN254 G1/G2 points).
public struct Groth16Codec: ProofCodec {
    public let proofType: ProofSystemType = .groth16
    public let defaultCurveId: CurveId = .bn254
    public let defaultFieldId: FieldId = .bn254Fr

    public init() {}

    public func encode(_ proof: Any) throws -> [UInt8] {
        guard let g16 = proof as? Groth16Proof else {
            throw ProofRegistryError.typeMismatch(
                expected: "Groth16Proof", got: String(describing: Swift.type(of: proof)))
        }
        let w = ProofWriter()
        w.writeLabel("GROTH16-v1")
        w.writePointProjective(g16.a)
        w.writeFp(g16.b.x.c0); w.writeFp(g16.b.x.c1)
        w.writeFp(g16.b.y.c0); w.writeFp(g16.b.y.c1)
        w.writeFp(g16.b.z.c0); w.writeFp(g16.b.z.c1)
        w.writePointProjective(g16.c)
        return w.finalize()
    }

    public func decode(_ data: [UInt8]) throws -> Any {
        let r = ProofReader(data)
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
}

/// Codec for PlonkProof.
public struct PlonkCodec: ProofCodec {
    public let proofType: ProofSystemType = .plonk
    public let defaultCurveId: CurveId = .bn254
    public let defaultFieldId: FieldId = .bn254Fr

    public init() {}

    public func encode(_ proof: Any) throws -> [UInt8] {
        guard let plonk = proof as? PlonkProof else {
            throw ProofRegistryError.typeMismatch(
                expected: "PlonkProof", got: String(describing: Swift.type(of: proof)))
        }
        let w = ProofWriter()
        w.writeLabel("PLONK-v1")
        w.writePointProjective(plonk.aCommit)
        w.writePointProjective(plonk.bCommit)
        w.writePointProjective(plonk.cCommit)
        w.writePointProjective(plonk.zCommit)
        w.writePointProjective(plonk.tLoCommit)
        w.writePointProjective(plonk.tMidCommit)
        w.writePointProjective(plonk.tHiCommit)
        w.writePointProjectiveArray(plonk.tExtraCommits)
        w.writeFr(plonk.aEval)
        w.writeFr(plonk.bEval)
        w.writeFr(plonk.cEval)
        w.writeFr(plonk.sigma1Eval)
        w.writeFr(plonk.sigma2Eval)
        w.writeFr(plonk.zOmegaEval)
        w.writePointProjective(plonk.openingProof)
        w.writePointProjective(plonk.shiftedOpeningProof)
        w.writeFrArray(plonk.publicInputs)
        return w.finalize()
    }

    public func decode(_ data: [UInt8]) throws -> Any {
        let r = ProofReader(data)
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
        let publicInputs = try r.readFrArray()
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
            publicInputs: publicInputs
        )
    }
}

/// Codec for KZGProof.
public struct KZGCodec: ProofCodec {
    public let proofType: ProofSystemType = .kzg
    public let defaultCurveId: CurveId = .bn254
    public let defaultFieldId: FieldId = .bn254Fr

    public init() {}

    public func encode(_ proof: Any) throws -> [UInt8] {
        guard let kzg = proof as? KZGProof else {
            throw ProofRegistryError.typeMismatch(
                expected: "KZGProof", got: String(describing: Swift.type(of: proof)))
        }
        return kzg.serialize()
    }

    public func decode(_ data: [UInt8]) throws -> Any {
        return try KZGProof.deserialize(data)
    }
}

/// Codec for IPAProof.
public struct IPACodec: ProofCodec {
    public let proofType: ProofSystemType = .ipa
    public let defaultCurveId: CurveId = .bn254
    public let defaultFieldId: FieldId = .bn254Fr

    public init() {}

    public func encode(_ proof: Any) throws -> [UInt8] {
        guard let ipa = proof as? IPAProof else {
            throw ProofRegistryError.typeMismatch(
                expected: "IPAProof", got: String(describing: Swift.type(of: proof)))
        }
        return ipa.serialize()
    }

    public func decode(_ data: [UInt8]) throws -> Any {
        return try IPAProof.deserialize(data)
    }
}
