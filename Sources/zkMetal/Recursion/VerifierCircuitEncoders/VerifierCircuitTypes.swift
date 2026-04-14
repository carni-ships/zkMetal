// VerifierCircuitTypes — Shared types for recursive verifier circuit encoders
//
// Provides common types and utilities used by all VerifierCircuitProtocol implementations
// for recursive SNARK composition across different proof systems.

import Foundation
import NeonFieldOps

// MARK: - Inner Proof System Type

/// Enum identifying the underlying proof system for recursive verification.
/// Named differently from ProofSystemType to avoid conflicts with
/// the proof system type enum in UniversalProofFormat.
public enum InnerProofSystemType: String, CustomStringConvertible {
    case groth16 = "Groth16"
    case halo2 = "Halo2"
    case plonk = "Plonk"
    case plonky2 = "Plonky2"
    case ipa = "IPA"

    public var description: String { rawValue }
}

// MARK: - Encoder Configuration

/// Configuration for creating a recursive verifier encoder.
public enum RecursiveVerifierEncoder {
    /// Halo2 verifier encoder (compiles to Plonk)
    case halo2(numAdvice: Int, numFixed: Int)

    /// Plonk verifier encoder (BN254)
    case plonk(n: Int, numSelectors: Int)

    /// Plonky2 verifier encoder (Goldilocks)
    case plonky2(circuitRepr: Plonky2RecursiveCircuitRepr)

    /// IPA verifier encoder (Pasta cycle)
    case ipaPasta

    /// Create the appropriate encoder instance.
    public func createEncoder() -> any VerifierCircuitProtocol {
        switch self {
        case .halo2:
            return Halo2VerifierCircuitEncoder()
        case .plonk:
            return PlonkVerifierCircuitEncoder()
        case .plonky2(let repr):
            return Plonky2VerifierCircuitEncoder(circuitRepr: repr)
        case .ipaPasta:
            return IPAPastaVerifierCircuitEncoder()
        }
    }
}

// MARK: - In-Circuit Field Embedding

/// Helper for embedding foreign field elements into the circuit's field.
/// For Goldilocks (64-bit) into BN254 Fr (254-bit), we use direct limb embedding.
public struct FieldEmbedder {
    /// Embed a Goldilocks element into Fr by reinterpreting as limbs.
    /// Goldilocks fits in 64 bits, Fr has 256 bits, so we use 4 limbs.
    public static func embedGoldilocks(_ gl: Gl) -> Fr {
        // Goldilocks limb is a single UInt64, Fr uses 8 UInt32 limbs
        // Just reinterpret the Goldilocks representation directly
        return Fr(v: (
            UInt32(gl.v & 0xFFFFFFFF),
            UInt32((gl.v >> 32) & 0xFFFFFFFF),
            0, 0, 0, 0, 0, 0
        ))
    }

    /// Embed a BN254 Fr element (identity for BN254 circuits).
    public static func embedBN254(_ fr: Fr) -> Fr {
        return fr
    }

    /// Embed an Fp element into Fr by reinterpreting the limb representation.
    /// This is a raw bit reinterpretation, NOT a field homomorphism.
    /// Used for embedding point coordinates into R1CS witnesses.
    public static func embedFp(_ fp: Fp) -> Fr {
        Fr(v: fp.v)
    }
}

// MARK: - Poseidon2-based Transcript for In-Circuit Fiat-Shamir

/// In-circuit Fiat-Shamir transcript using Poseidon2.
/// This is used for deriving challenges inside the circuit rather than using
/// an external hash function like Blake3 (which is expensive in-circuit).
public struct InCircuitTranscript {
    public let builder: PlonkCircuitBuilder
    public var state: [Int] = []  // Poseidon state as circuit variables

    public init(builder: PlonkCircuitBuilder) {
        self.builder = builder
    }

    /// Absorb a field element into the transcript.
    public mutating func absorb(_ varIndex: Int) {
        state.append(varIndex)
    }

    /// Absorb a point (x, y) into the transcript.
    public mutating func absorbPoint(_ xVar: Int, _ yVar: Int) {
        state.append(xVar)
        state.append(yVar)
    }

    /// Squeeze to get a challenge variable.
    /// Uses Poseidon2 hash of the state.
    public mutating func squeeze() -> Int {
        // For now, use a placeholder - actual implementation would use Poseidon2 gates
        // This is a simplified version for the encoder structure
        let challenge = builder.addInput()
        state.append(challenge)
        return challenge
    }
}
