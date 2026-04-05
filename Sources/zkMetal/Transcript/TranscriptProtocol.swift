// TranscriptProtocol — Unified multi-hash transcript interface
//
// Provides:
//   - TranscriptProtocol: common interface for all transcript backends
//   - TranscriptHashType: enum for selecting hash backend
//   - TranscriptFactory: creates transcripts from hash type selection
//
// This unifies Keccak, Poseidon2, and Blake3 backends behind a single
// protocol, making it easy to swap hash functions without changing
// caller code.

import Foundation
import NeonFieldOps

// MARK: - Hash Type Enum

/// Selects the hash backend for transcript construction.
public enum TranscriptHashType {
    /// Keccak-256 sponge (rate=136 bytes). Ethereum-compatible.
    case keccak
    /// Poseidon2 algebraic sponge (t=3, rate=2). Fastest for field-native proofs.
    case poseidon2
    /// Blake3 hash chain. Fast for byte-oriented operations.
    case blake3
}

// MARK: - TranscriptProtocol

/// Unified protocol for all Fiat-Shamir transcript implementations.
///
/// This protocol abstracts over the hash backend, allowing proof systems
/// to be generic over the transcript type. All implementations provide:
///   - Domain-separated absorb/squeeze with string labels
///   - Deterministic challenge generation
///   - Fork support for sub-protocols
///   - Monotonic operation counter for replay protection
///
/// Conforming types: TranscriptWrapper (via TranscriptFactory),
/// DomainSeparatedTranscript.
public protocol TranscriptProtocol {
    /// Absorb a labeled byte message into the transcript.
    mutating func appendMessage(label: String, data: [UInt8])

    /// Absorb a labeled field element into the transcript.
    mutating func appendScalar(label: String, scalar: Fr)

    /// Absorb a labeled curve point into the transcript.
    mutating func appendPoint(label: String, point: PointProjective)

    /// Squeeze a single field element challenge.
    mutating func squeezeChallenge() -> Fr

    /// Squeeze multiple field element challenges.
    mutating func squeezeChallenges(count: Int) -> [Fr]

    /// Fork the transcript for a sub-protocol.
    func fork(label: String) -> any TranscriptProtocol

    /// Monotonic operation counter.
    var operationCount: UInt64 { get }

    /// The hash backend type used by this transcript.
    var hashType: TranscriptHashType { get }
}

// MARK: - TranscriptWrapper

/// Type-erased wrapper that bridges any TranscriptEngine to TranscriptProtocol.
///
/// This avoids conditional conformance issues with FiatShamirTranscript's
/// generic parameter while providing a uniform interface.
public struct TranscriptWrapper: TranscriptProtocol {
    private var keccak: KeccakTranscript?
    private var poseidon: Poseidon2Transcript?
    private var blake3: Blake3Transcript?
    public let hashType: TranscriptHashType

    public var operationCount: UInt64 {
        switch hashType {
        case .keccak:   return keccak!.operationCount
        case .poseidon2: return poseidon!.operationCount
        case .blake3:   return blake3!.operationCount
        }
    }

    /// Create a Keccak-backed wrapper.
    public init(keccak t: KeccakTranscript) {
        self.hashType = .keccak
        self.keccak = t
    }

    /// Create a Poseidon2-backed wrapper.
    public init(poseidon t: Poseidon2Transcript) {
        self.hashType = .poseidon2
        self.poseidon = t
    }

    /// Create a Blake3-backed wrapper.
    public init(blake3 t: Blake3Transcript) {
        self.hashType = .blake3
        self.blake3 = t
    }

    public mutating func appendMessage(label: String, data: [UInt8]) {
        switch hashType {
        case .keccak:   keccak!.appendMessage(label: label, data: data)
        case .poseidon2: poseidon!.appendMessage(label: label, data: data)
        case .blake3:   blake3!.appendMessage(label: label, data: data)
        }
    }

    public mutating func appendScalar(label: String, scalar: Fr) {
        switch hashType {
        case .keccak:   keccak!.appendScalar(label: label, scalar: scalar)
        case .poseidon2: poseidon!.appendScalar(label: label, scalar: scalar)
        case .blake3:   blake3!.appendScalar(label: label, scalar: scalar)
        }
    }

    public mutating func appendPoint(label: String, point: PointProjective) {
        switch hashType {
        case .keccak:   keccak!.appendPoint(label: label, point: point)
        case .poseidon2: poseidon!.appendPoint(label: label, point: point)
        case .blake3:   blake3!.appendPoint(label: label, point: point)
        }
    }

    public mutating func squeezeChallenge() -> Fr {
        switch hashType {
        case .keccak:   return keccak!.squeezeChallenge()
        case .poseidon2: return poseidon!.squeezeChallenge()
        case .blake3:   return blake3!.squeezeChallenge()
        }
    }

    public mutating func squeezeChallenges(count: Int) -> [Fr] {
        switch hashType {
        case .keccak:   return keccak!.squeezeChallenges(count: count)
        case .poseidon2: return poseidon!.squeezeChallenges(count: count)
        case .blake3:   return blake3!.squeezeChallenges(count: count)
        }
    }

    public func fork(label: String) -> any TranscriptProtocol {
        switch hashType {
        case .keccak:
            return TranscriptWrapper(keccak: keccak!.forkTyped(label: label))
        case .poseidon2:
            return TranscriptWrapper(poseidon: poseidon!.forkTyped(label: label))
        case .blake3:
            return TranscriptWrapper(blake3: blake3!.forkTyped(label: label))
        }
    }
}

// MARK: - TranscriptFactory

/// Factory for creating transcripts with a selected hash backend.
///
/// Usage:
///   ```
///   var t = TranscriptFactory.create(hash: .keccak, label: "MyProtocol")
///   t.appendMessage(label: "commitment", data: bytes)
///   let challenge = t.squeezeChallenge()
///   ```
public enum TranscriptFactory {
    /// Create a transcript with the specified hash backend and domain label.
    ///
    /// - Parameters:
    ///   - hash: Hash backend selection (.keccak, .poseidon2, or .blake3)
    ///   - label: Protocol-level domain separator
    /// - Returns: A transcript conforming to TranscriptProtocol
    public static func create(hash: TranscriptHashType, label: String = "default") -> TranscriptWrapper {
        switch hash {
        case .keccak:
            return TranscriptWrapper(keccak: KeccakTranscript(label: label))
        case .poseidon2:
            return TranscriptWrapper(poseidon: Poseidon2Transcript(label: label))
        case .blake3:
            return TranscriptWrapper(blake3: Blake3Transcript(label: label))
        }
    }
}
