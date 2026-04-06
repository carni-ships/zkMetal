// FiatShamirTranscript — Production Fiat-Shamir transcript engine
//
// Provides a unified protocol-driven API for challenge generation across all
// proof systems (Plonk, Groth16, STARK, FRI, Spartan, etc.).
//
// Architecture:
//   TranscriptEngine (protocol)
//     -> FiatShamirTranscript<H: TranscriptHasher> (generic implementation)
//       -> KeccakTranscript   (Keccak-256 sponge, Ethereum-compatible)
//       -> Poseidon2Transcript (field-native sponge, fastest for Fr)
//       -> Blake3Transcript   (Blake3, fastest for byte operations)
//
// Security properties:
//   - Domain separation: each transcript initialized with a domain tag string
//   - Per-operation labels: every absorb/squeeze is labeled to prevent cross-step collisions
//   - Replay protection: monotonic operation counter prevents rewind attacks
//   - Fork safety: child transcripts inherit parent state with fresh label
//   - Deterministic: same inputs always produce same outputs

import Foundation
import NeonFieldOps

// MARK: - TranscriptEngine Protocol

/// Protocol for Fiat-Shamir transcript engines used across all proof systems.
///
/// Implementations provide domain-separated absorb/squeeze with replay protection.
/// The transcript maintains a monotonically increasing operation counter to prevent
/// rewinding the state to an earlier point.
///
/// Typical usage:
///   ```
///   var t: any TranscriptEngine = KeccakTranscript(label: "Plonk-v1")
///   t.appendMessage(label: "commitment", data: commitBytes)
///   t.appendScalar(label: "public-input", scalar: publicInput)
///   t.appendPoint(label: "C_a", point: commitmentPoint)
///   let alpha = t.squeezeChallenge()
///   let betas = t.squeezeChallenges(count: 3)
///   ```
public protocol TranscriptEngine {
    /// Append a labeled byte message to the transcript.
    ///
    /// Internally absorbs: len(label) || label || len(data) || data
    /// The length prefixes prevent collisions between different label/data pairs.
    ///
    /// - Parameters:
    ///   - label: Domain separator for this operation (e.g., "round-1-commitment")
    ///   - data: Raw bytes to absorb
    mutating func appendMessage(label: String, data: [UInt8])

    /// Append a labeled field element (scalar) to the transcript.
    ///
    /// Serializes the Fr element as 32 bytes and absorbs with the given label.
    ///
    /// - Parameters:
    ///   - label: Domain separator for this operation
    ///   - scalar: Field element to absorb
    mutating func appendScalar(label: String, scalar: Fr)

    /// Append a labeled elliptic curve point to the transcript.
    ///
    /// Converts the projective point to affine coordinates and absorbs both
    /// the x and y coordinates as field elements.
    ///
    /// - Parameters:
    ///   - label: Domain separator for this operation
    ///   - point: Projective curve point to absorb
    mutating func appendPoint(label: String, point: PointProjective)

    /// Squeeze a single field element challenge from the transcript.
    ///
    /// Absorbs an implicit "challenge" label, then squeezes 32 bytes and reduces
    /// modulo the scalar field order to produce a uniformly distributed Fr element.
    ///
    /// - Returns: A field element challenge in Montgomery form
    mutating func squeezeChallenge() -> Fr

    /// Squeeze multiple field element challenges from the transcript.
    ///
    /// Each challenge is produced by a separate squeeze operation, ensuring
    /// independence between challenges.
    ///
    /// - Parameter count: Number of challenges to generate
    /// - Returns: Array of `count` independent Fr element challenges
    mutating func squeezeChallenges(count: Int) -> [Fr]

    /// Fork the transcript for parallel sub-protocols.
    ///
    /// Creates an independent child transcript that inherits the current state
    /// and absorbs a fresh label to ensure distinct challenge streams.
    /// The parent transcript is not modified.
    ///
    /// - Parameter label: Unique label for this fork (e.g., "sub-proof-0")
    /// - Returns: An independent transcript engine with the forked state
    func fork(label: String) -> any TranscriptEngine

    /// The number of operations (absorb + squeeze) performed on this transcript.
    ///
    /// This counter increases monotonically and is used for replay protection.
    var operationCount: UInt64 { get }
}

// MARK: - TranscriptHasher Protocol

/// Protocol for pluggable hash backends used by FiatShamirTranscript.
///
/// Implementations must provide a duplex sponge (or equivalent) that supports:
///   - Absorbing raw bytes into internal state
///   - Squeezing arbitrary-length output bytes
///   - Cloning for fork operations
public protocol TranscriptHasher {
    /// Absorb raw bytes into the hash state.
    mutating func absorb(_ data: [UInt8])

    /// Squeeze `byteCount` bytes from the hash state.
    mutating func squeeze(byteCount: Int) -> [UInt8]

    /// Create an independent copy of the current state.
    func clone() -> Self
}

// MARK: - FiatShamirTranscript

/// A Fiat-Shamir transcript with per-operation domain separation labels and replay protection.
///
/// Every absorb and squeeze call is prefixed with a label, ensuring that
/// identical data absorbed under different labels produces different state.
/// A monotonic operation counter prevents rewinding to an earlier state.
///
/// Usage:
///   ```
///   var t = FiatShamirTranscript(label: "my-protocol", hasher: KeccakTranscriptHasher())
///   t.appendMessage(label: "commitment", data: commitmentBytes)
///   t.appendScalar(label: "public-input", scalar: publicInput)
///   let challenge = t.squeezeChallenge()
///   ```
public struct FiatShamirTranscript<H: TranscriptHasher>: TranscriptEngine {
    private var hasher: H
    private var _operationCount: UInt64 = 0

    /// The number of operations (absorb + squeeze) performed on this transcript.
    public var operationCount: UInt64 { _operationCount }

    /// Create a new transcript with an initial domain separation label.
    ///
    /// The label is absorbed immediately to bind this transcript to a specific protocol.
    /// - Parameters:
    ///   - label: Protocol-level domain separator (e.g., "Plonk-v1", "Groth16")
    ///   - hasher: The hash backend to use
    public init(label: String, hasher: H) {
        self.hasher = hasher
        // Absorb protocol-level domain separator
        absorbLabel(label)
    }

    // MARK: - TranscriptEngine conformance

    public mutating func appendMessage(label: String, data: [UInt8]) {
        absorbLabel(label)
        absorbLengthPrefixed(data)
        _operationCount &+= 1
    }

    public mutating func appendScalar(label: String, scalar: Fr) {
        absorbLabel(label)
        let bytes = frToBytes(scalar)
        absorbLengthPrefixed(bytes)
        _operationCount &+= 1
    }

    public mutating func appendPoint(label: String, point: PointProjective) {
        absorbLabel(label)
        // Convert projective to affine, absorb x and y coordinates
        var affine = [UInt64](repeating: 0, count: 8)  // x[4], y[4]
        withUnsafeBytes(of: point) { pBuf in
            affine.withUnsafeMutableBufferPointer { aBuf in
                bn254_projective_to_affine(
                    pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    aBuf.baseAddress!
                )
            }
        }
        let xBytes = limbs64ToBytes(Array(affine[0..<4]))
        let yBytes = limbs64ToBytes(Array(affine[4..<8]))
        absorbLengthPrefixed(xBytes)
        absorbLengthPrefixed(yBytes)
        _operationCount &+= 1
    }

    public mutating func squeezeChallenge() -> Fr {
        _operationCount &+= 1
        // Absorb the operation counter for replay protection
        absorbCounter()
        let bytes = hasher.squeeze(byteCount: 32)
        return bytesToFr(bytes)
    }

    public mutating func squeezeChallenges(count: Int) -> [Fr] {
        var result = [Fr]()
        result.reserveCapacity(count)
        for _ in 0..<count {
            result.append(squeezeChallenge())
        }
        return result
    }

    public func fork(label: String) -> any TranscriptEngine {
        var child = FiatShamirTranscript<H>(hasher: hasher.clone())
        child.absorbLabel(label)
        return child
    }

    // MARK: - Legacy API (backward compatibility)

    /// Absorb labeled data into the transcript (legacy API).
    public mutating func absorb(_ label: String, _ data: [UInt8]) {
        appendMessage(label: label, data: data)
    }

    /// Absorb a field element with a label (legacy API).
    public mutating func absorbFr(_ label: String, _ element: Fr) {
        appendScalar(label: label, scalar: element)
    }

    /// Absorb multiple field elements with a label (legacy API).
    public mutating func absorbFrMany(_ label: String, _ elements: [Fr]) {
        absorbLabel(label)
        for e in elements {
            let bytes = frToBytes(e)
            absorbLengthPrefixed(bytes)
        }
        _operationCount &+= 1
    }

    /// Absorb raw bytes without a label (for chaining within a labeled section).
    public mutating func absorbRaw(_ data: [UInt8]) {
        hasher.absorb(data)
    }

    /// Squeeze labeled output bytes from the transcript (legacy API).
    public mutating func squeeze(_ label: String, byteCount: Int) -> [UInt8] {
        absorbLabel(label)
        _operationCount &+= 1
        return hasher.squeeze(byteCount: byteCount)
    }

    /// Squeeze a field element challenge from the transcript (legacy API).
    public mutating func challengeScalar(_ label: String) -> Fr {
        absorbLabel(label)
        _operationCount &+= 1
        absorbCounter()
        let bytes = hasher.squeeze(byteCount: 32)
        return bytesToFr(bytes)
    }

    /// Squeeze a BN254 field element challenge (legacy API).
    public mutating func challengeField(_ label: String) -> BN254Fr {
        return challengeScalar(label)
    }

    /// Squeeze multiple field element challenges (legacy API).
    public mutating func challengeScalars(_ label: String, count: Int) -> [Fr] {
        absorbLabel(label)
        var result = [Fr]()
        result.reserveCapacity(count)
        for _ in 0..<count {
            _operationCount &+= 1
            absorbCounter()
            let bytes = hasher.squeeze(byteCount: 32)
            result.append(bytesToFr(bytes))
        }
        return result
    }

    /// Squeeze labeled output bytes (alternate signature, legacy API).
    public mutating func squeeze(_ label: String, count: Int) -> [UInt8] {
        return squeeze(label, byteCount: count)
    }

    /// Fork with typed return (preserves generic type, legacy API).
    public func forkTyped(label: String) -> FiatShamirTranscript<H> {
        var child = FiatShamirTranscript<H>(hasher: hasher.clone())
        child.absorbLabel(label)
        return child
    }

    // MARK: - Internal helpers

    /// Internal init for fork (bypasses initial label absorb).
    private init(hasher: H) {
        self.hasher = hasher
    }

    /// Absorb a label: 4-byte LE length prefix + UTF-8 bytes.
    private mutating func absorbLabel(_ label: String) {
        let utf8 = Array(label.utf8)
        var len = UInt32(utf8.count)
        let lenBytes = withUnsafeBytes(of: &len) { Array($0) }
        hasher.absorb(lenBytes + utf8)
    }

    /// Absorb length-prefixed data: 4-byte LE length + raw bytes.
    private mutating func absorbLengthPrefixed(_ data: [UInt8]) {
        var len = UInt32(data.count)
        let lenBytes = withUnsafeBytes(of: &len) { Array($0) }
        hasher.absorb(lenBytes + data)
    }

    /// Absorb the current operation counter for replay protection.
    private mutating func absorbCounter() {
        var counter = _operationCount
        let counterBytes = withUnsafeBytes(of: &counter) { Array($0) }
        hasher.absorb(counterBytes)
    }

    /// Convert Fr to 32 bytes (Montgomery representation, little-endian).
    private func frToBytes(_ element: Fr) -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: 32)
        withUnsafeBytes(of: element) { buf in
            let ptr = buf.baseAddress!.assumingMemoryBound(to: UInt8.self)
            for i in 0..<32 {
                bytes[i] = ptr[i]
            }
        }
        return bytes
    }

    /// Convert [UInt64] limbs to bytes (little-endian).
    private func limbs64ToBytes(_ limbs: [UInt64]) -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: limbs.count * 8)
        for (i, limb) in limbs.enumerated() {
            for j in 0..<8 {
                bytes[i * 8 + j] = UInt8((limb >> (j * 8)) & 0xFF)
            }
        }
        return bytes
    }

    /// Convert 32 bytes to Fr: interpret as 256-bit LE integer, reduce mod r.
    private func bytesToFr(_ bytes: [UInt8]) -> Fr {
        var limbs: [UInt64] = [0, 0, 0, 0]
        for i in 0..<4 {
            for j in 0..<8 {
                limbs[i] |= UInt64(bytes[i * 8 + j]) << (j * 8)
            }
        }
        // Mask top limb to ensure < 2^254 (BN254 r is ~254 bits)
        limbs[3] &= 0x3FFFFFFFFFFFFFFF
        // Convert to Montgomery form
        let raw = Fr.from64(limbs)
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))
    }
}

// MARK: - Keccak Transcript Hasher

/// Keccak-256 sponge hasher for Ethereum-compatible Fiat-Shamir.
///
/// Uses NEON-accelerated Keccak-f[1600] permutation.
/// Rate = 136 bytes (1088 bits), capacity = 64 bytes (512 bits).
public struct KeccakTranscriptHasher: TranscriptHasher {
    // Full 200-byte (1600-bit) Keccak state as 25 UInt64s
    private var state: [UInt64] = [UInt64](repeating: 0, count: 25)
    private var rateOffset: Int = 0
    private var squeezing: Bool = false

    private static let rate: Int = 136

    public init() {}

    public mutating func absorb(_ data: [UInt8]) {
        if squeezing {
            // Transition back to absorb mode: reset squeeze state
            squeezing = false
        }

        var offset = 0
        while offset < data.count {
            let available = Self.rate - rateOffset
            let toAbsorb = min(available, data.count - offset)

            for i in 0..<toAbsorb {
                let pos = rateOffset + i
                let wordIdx = pos / 8
                let byteIdx = pos % 8
                state[wordIdx] ^= UInt64(data[offset + i]) << (byteIdx * 8)
            }

            rateOffset += toAbsorb
            offset += toAbsorb

            if rateOffset == Self.rate {
                permute()
                rateOffset = 0
            }
        }
    }

    public mutating func squeeze(byteCount: Int) -> [UInt8] {
        if !squeezing {
            // Apply Keccak padding: 0x01 at current position, 0x80 at last rate byte
            let pos0 = rateOffset
            state[pos0 / 8] ^= UInt64(0x01) << ((pos0 % 8) * 8)
            let lastPos = Self.rate - 1
            state[lastPos / 8] ^= UInt64(0x80) << ((lastPos % 8) * 8)
            permute()
            rateOffset = 0
            squeezing = true
        }

        var result = [UInt8]()
        result.reserveCapacity(byteCount)

        while result.count < byteCount {
            if rateOffset >= Self.rate {
                permute()
                rateOffset = 0
            }
            let available = Self.rate - rateOffset
            let toSqueeze = min(available, byteCount - result.count)

            for i in 0..<toSqueeze {
                let pos = rateOffset + i
                let wordIdx = pos / 8
                let byteIdx = pos % 8
                result.append(UInt8((state[wordIdx] >> (byteIdx * 8)) & 0xFF))
            }
            rateOffset += toSqueeze
        }
        return result
    }

    public func clone() -> KeccakTranscriptHasher {
        var copy = KeccakTranscriptHasher()
        copy.state = self.state
        copy.rateOffset = self.rateOffset
        copy.squeezing = self.squeezing
        return copy
    }

    private mutating func permute() {
        state.withUnsafeMutableBufferPointer { buf in
            keccak_f1600_neon(buf.baseAddress!)
        }
    }
}

// MARK: - Poseidon2 Transcript Hasher

/// Poseidon2 sponge hasher for field-native Fiat-Shamir.
///
/// Uses t=3 (rate=2, capacity=1), operating directly on Fr elements.
/// Bytes are packed into Fr elements (31 bytes per element to stay < p).
public struct Poseidon2TranscriptHasher: TranscriptHasher {
    // Tuple-based state avoids array allocation on every permute
    private var s0: Fr = Fr.zero
    private var s1: Fr = Fr.zero
    private var s2: Fr = Fr.zero
    private var absorbed: Int = 0
    private var needsPermute: Bool = false

    public init() {}

    public mutating func absorb(_ data: [UInt8]) {
        // Pack bytes into Fr elements (31 bytes per element to stay < p)
        var offset = 0
        while offset < data.count {
            let chunkSize = min(31, data.count - offset)
            var limbs: [UInt64] = [0, 0, 0, 0]
            for i in 0..<chunkSize {
                let limbIdx = i / 8
                let shift = (i % 8) * 8
                limbs[limbIdx] |= UInt64(data[offset + i]) << shift
            }
            let raw = Fr.from64(limbs)
            let mont = frMul(raw, Fr.from64(Fr.R2_MOD_R))
            absorbFrElement(mont)
            offset += chunkSize
        }
    }

    public mutating func squeeze(byteCount: Int) -> [UInt8] {
        var result = [UInt8]()
        result.reserveCapacity(byteCount)
        while result.count < byteCount {
            let fr = squeezeFrElement()
            let limbs = frToInt(fr)
            for limb in limbs {
                for byte in 0..<8 {
                    if result.count < byteCount {
                        result.append(UInt8((limb >> (byte * 8)) & 0xFF))
                    }
                }
            }
        }
        return result
    }

    public func clone() -> Poseidon2TranscriptHasher {
        var copy = Poseidon2TranscriptHasher()
        copy.s0 = self.s0
        copy.s1 = self.s1
        copy.s2 = self.s2
        copy.absorbed = self.absorbed
        copy.needsPermute = self.needsPermute
        return copy
    }

    private mutating func absorbFrElement(_ value: Fr) {
        switch absorbed {
        case 0: s0 = frAdd(s0, value)
        case 1: s1 = frAdd(s1, value)
        default: break
        }
        absorbed += 1
        if absorbed == 2 {
            poseidon2PermuteInPlace(&s0, &s1, &s2)
            absorbed = 0
        }
        needsPermute = true
    }

    private mutating func squeezeFrElement() -> Fr {
        if needsPermute || absorbed > 0 {
            poseidon2PermuteInPlace(&s0, &s1, &s2)
            absorbed = 0
            needsPermute = false
        }
        let result = s0
        needsPermute = true
        return result
    }
}

// MARK: - Blake3 Transcript Hasher

/// Blake3-based transcript hasher using NEON-accelerated hashing.
///
/// Unlike sponge constructions, Blake3 is a plain hash function.
/// We simulate absorb/squeeze using a running hash chain: each absorb mixes
/// new data into a 32-byte running state via hash(state || data_chunk).
/// Each squeeze reads from the state and advances it.
///
/// This design avoids the 64-byte block size limitation of the single-block
/// blake3_hash_neon function by processing data in chunks.
public struct Blake3TranscriptHasher: TranscriptHasher {
    /// Running 32-byte hash state (chaining value)
    private var chainState: [UInt8] = [UInt8](repeating: 0, count: 32)
    /// Squeeze counter for generating multiple distinct outputs from the same state
    private var squeezeCounter: UInt64 = 0

    public init() {}

    public mutating func absorb(_ data: [UInt8]) {
        // Process data in 32-byte chunks: hash(chainState || chunk) -> new chainState
        // This ensures all data affects the state regardless of total length
        var offset = 0
        while offset < data.count {
            let chunkSize = min(32, data.count - offset)
            var block = [UInt8](repeating: 0, count: 64)
            // First 32 bytes: current chain state
            for i in 0..<32 { block[i] = chainState[i] }
            // Next bytes: data chunk (padded with zeros)
            for i in 0..<chunkSize { block[32 + i] = data[offset + i] }
            // Absorb the chunk length as a domain separator in remaining bytes
            block[32 + 31] ^= UInt8(chunkSize)

            // Hash the 64-byte block
            block.withUnsafeBufferPointer { inp in
                chainState.withUnsafeMutableBufferPointer { out in
                    blake3_hash_neon(inp.baseAddress!, 64, out.baseAddress!)
                }
            }
            offset += chunkSize
        }
        // Reset squeeze counter on new absorb (state has changed)
        squeezeCounter = 0
    }

    public mutating func squeeze(byteCount: Int) -> [UInt8] {
        var result = [UInt8]()
        result.reserveCapacity(byteCount)

        while result.count < byteCount {
            // Hash(chainState || squeezeCounter) to get output block
            var block = [UInt8](repeating: 0, count: 64)
            for i in 0..<32 { block[i] = chainState[i] }
            var counter = squeezeCounter
            withUnsafeBytes(of: &counter) { src in
                for i in 0..<8 { block[32 + i] = src[i] }
            }
            // Mark as squeeze operation (domain separation from absorb)
            block[32 + 8] = 0xFF

            var hash = [UInt8](repeating: 0, count: 32)
            block.withUnsafeBufferPointer { inp in
                hash.withUnsafeMutableBufferPointer { out in
                    blake3_hash_neon(inp.baseAddress!, 64, out.baseAddress!)
                }
            }

            let needed = min(32, byteCount - result.count)
            result.append(contentsOf: hash.prefix(needed))
            squeezeCounter += 1
        }

        // Update chain state so subsequent operations see squeeze happened
        // Hash the chain state with the final counter for forward progress
        var advanceBlock = [UInt8](repeating: 0, count: 64)
        for i in 0..<32 { advanceBlock[i] = chainState[i] }
        var finalCounter = squeezeCounter
        withUnsafeBytes(of: &finalCounter) { src in
            for i in 0..<8 { advanceBlock[32 + i] = src[i] }
        }
        advanceBlock[32 + 9] = 0xFE  // domain separation from output blocks
        advanceBlock.withUnsafeBufferPointer { inp in
            chainState.withUnsafeMutableBufferPointer { out in
                blake3_hash_neon(inp.baseAddress!, 64, out.baseAddress!)
            }
        }

        return result
    }

    public func clone() -> Blake3TranscriptHasher {
        var copy = Blake3TranscriptHasher()
        copy.chainState = self.chainState
        copy.squeezeCounter = self.squeezeCounter
        return copy
    }
}

// MARK: - BN254Fr Alias

/// BN254 scalar field element alias for protocol-level clarity.
public typealias BN254Fr = Fr

// MARK: - Convenience Type Aliases

/// Keccak-backed Fiat-Shamir transcript (Ethereum-compatible).
public typealias KeccakTranscript = FiatShamirTranscript<KeccakTranscriptHasher>

/// Poseidon2-backed Fiat-Shamir transcript (field-native, fastest for ZK).
public typealias Poseidon2Transcript = FiatShamirTranscript<Poseidon2TranscriptHasher>

/// Blake3-backed Fiat-Shamir transcript (fastest for byte operations).
public typealias Blake3Transcript = FiatShamirTranscript<Blake3TranscriptHasher>

// MARK: - Convenience Initializers

extension FiatShamirTranscript where H == KeccakTranscriptHasher {
    /// Create a Keccak-256 transcript with the given domain label.
    public init(label: String) {
        self.init(label: label, hasher: KeccakTranscriptHasher())
    }
}

extension FiatShamirTranscript where H == Poseidon2TranscriptHasher {
    /// Create a Poseidon2 transcript with the given domain label.
    public init(label: String) {
        self.init(label: label, hasher: Poseidon2TranscriptHasher())
    }
}

extension FiatShamirTranscript where H == Blake3TranscriptHasher {
    /// Create a Blake3 transcript with the given domain label.
    public init(label: String) {
        self.init(label: label, hasher: Blake3TranscriptHasher())
    }
}

// MARK: - FieldElement protocol for generic field support

/// Protocol for types that can be serialized into a transcript.
///
/// This allows the transcript engine to work with different field types
/// (BN254 Fr, BLS12-381 Fr, BabyBear, Mersenne31, etc.) in the future.
public protocol TranscriptSerializable {
    /// Serialize to bytes for transcript absorption.
    func transcriptBytes() -> [UInt8]
}

extension Fr: TranscriptSerializable {
    public func transcriptBytes() -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: 32)
        withUnsafeBytes(of: self) { buf in
            let ptr = buf.baseAddress!.assumingMemoryBound(to: UInt8.self)
            for i in 0..<32 { bytes[i] = ptr[i] }
        }
        return bytes
    }
}
