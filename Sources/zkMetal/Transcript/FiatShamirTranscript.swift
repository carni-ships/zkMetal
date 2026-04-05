// FiatShamirTranscript — Reusable labeled absorb/squeeze with domain separation
//
// Higher-level wrapper providing per-operation labels for domain separation.
// Each absorb and squeeze call includes a label that is mixed into the state
// before the data, preventing cross-protocol and cross-round collisions.
//
// Supports pluggable hash backends via the TranscriptHasher protocol:
//   - KeccakTranscriptHasher:   Keccak-256 sponge (Ethereum-compatible)
//   - Poseidon2TranscriptHasher: Poseidon2 sponge (field-native, fastest for Fr)
//   - Blake3TranscriptHasher:   Blake3 hash (fastest for bytes, used in IPA/Lasso)

import Foundation
import NeonFieldOps

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

/// A Fiat-Shamir transcript with per-operation domain separation labels.
///
/// Every absorb and squeeze call is prefixed with a label, ensuring that
/// identical data absorbed under different labels produces different state.
///
/// Usage:
///   ```
///   var t = FiatShamirTranscript(label: "my-protocol", hasher: KeccakTranscriptHasher())
///   t.absorb("commitment", commitmentBytes)
///   t.absorb("public-input", publicInputBytes)
///   let challenge = t.squeeze("alpha", byteCount: 32)
///   let scalar = t.challengeScalar("beta")
///   ```
public struct FiatShamirTranscript<H: TranscriptHasher> {
    private var hasher: H

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

    // MARK: - Absorb

    /// Absorb labeled data into the transcript.
    ///
    /// Internally absorbs: len(label) || label || len(data) || data
    /// The length prefixes prevent collisions between different label/data pairs.
    ///
    /// - Parameters:
    ///   - label: Domain separator for this absorb operation (e.g., "round-1-commitment")
    ///   - data: Raw bytes to absorb
    public mutating func absorb(_ label: String, _ data: [UInt8]) {
        absorbLabel(label)
        absorbLengthPrefixed(data)
    }

    /// Absorb a field element with a label.
    ///
    /// Serializes the Fr element as 32 bytes (Montgomery representation) and absorbs.
    /// - Parameters:
    ///   - label: Domain separator for this absorb operation
    ///   - element: Field element to absorb
    public mutating func absorbFr(_ label: String, _ element: Fr) {
        absorbLabel(label)
        let bytes = frToBytes(element)
        absorbLengthPrefixed(bytes)
    }

    /// Absorb multiple field elements with a label.
    ///
    /// All elements share the same label; each is individually length-prefixed.
    /// - Parameters:
    ///   - label: Domain separator for this batch absorb
    ///   - elements: Field elements to absorb
    public mutating func absorbFrMany(_ label: String, _ elements: [Fr]) {
        absorbLabel(label)
        for e in elements {
            let bytes = frToBytes(e)
            absorbLengthPrefixed(bytes)
        }
    }

    /// Absorb raw bytes without a label (for chaining within a labeled section).
    public mutating func absorbRaw(_ data: [UInt8]) {
        hasher.absorb(data)
    }

    // MARK: - Squeeze

    /// Squeeze labeled output bytes from the transcript.
    ///
    /// Internally absorbs the label before squeezing, ensuring different labels
    /// produce different output even from the same prior state.
    ///
    /// - Parameters:
    ///   - label: Domain separator for this squeeze operation (e.g., "alpha-challenge")
    ///   - byteCount: Number of bytes to squeeze
    /// - Returns: `byteCount` pseudorandom bytes
    public mutating func squeeze(_ label: String, byteCount: Int) -> [UInt8] {
        absorbLabel(label)
        return hasher.squeeze(byteCount: byteCount)
    }

    /// Squeeze a field element challenge from the transcript.
    ///
    /// Squeezes 32 bytes, interprets as a 256-bit little-endian integer,
    /// and reduces modulo the BN254 scalar field order r.
    ///
    /// - Parameter label: Domain separator for this challenge
    /// - Returns: An Fr element in Montgomery form, uniformly distributed mod r
    public mutating func challengeScalar(_ label: String) -> Fr {
        absorbLabel(label)
        let bytes = hasher.squeeze(byteCount: 32)
        return bytesToFr(bytes)
    }

    /// Squeeze labeled output bytes from the transcript (alternate signature).
    ///
    /// Equivalent to `squeeze(_:byteCount:)` with a `count` parameter name.
    /// - Parameters:
    ///   - label: Domain separator for this squeeze operation
    ///   - count: Number of bytes to squeeze
    /// - Returns: `count` pseudorandom bytes
    public mutating func squeeze(_ label: String, count: Int) -> [UInt8] {
        return squeeze(label, byteCount: count)
    }

    /// Squeeze a BN254 field element challenge from the transcript.
    ///
    /// Squeezes 32 bytes, interprets as a 256-bit little-endian integer,
    /// and reduces modulo the BN254 scalar field order r.
    ///
    /// - Parameter label: Domain separator for this challenge
    /// - Returns: A BN254Fr element in Montgomery form, uniformly distributed mod r
    public mutating func challengeField(_ label: String) -> BN254Fr {
        return challengeScalar(label)
    }

    /// Squeeze multiple field element challenges.
    ///
    /// Each challenge gets an indexed sub-label for domain separation.
    /// - Parameters:
    ///   - label: Base domain separator
    ///   - count: Number of challenges to generate
    /// - Returns: Array of Fr elements
    public mutating func challengeScalars(_ label: String, count: Int) -> [Fr] {
        absorbLabel(label)
        return (0..<count).map { _ in
            let bytes = hasher.squeeze(byteCount: 32)
            return bytesToFr(bytes)
        }
    }

    // MARK: - Fork

    /// Fork the transcript for parallel sub-protocols.
    ///
    /// The child inherits the full parent state, then absorbs a fresh label
    /// to ensure its challenge stream diverges from the parent and any siblings.
    ///
    /// - Parameter label: Unique label for this fork (e.g., "sub-proof-0")
    /// - Returns: An independent transcript with the forked state
    public func fork(label: String) -> FiatShamirTranscript<H> {
        var child = FiatShamirTranscript<H>(hasher: hasher.clone())
        child.absorbLabel(label)
        return child
    }

    // MARK: - Internal

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
            // Transition back to absorb mode: reset
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
    private var state: [Fr] = [Fr.zero, Fr.zero, Fr.zero]
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
        copy.state = self.state
        copy.absorbed = self.absorbed
        copy.needsPermute = self.needsPermute
        return copy
    }

    private mutating func absorbFrElement(_ value: Fr) {
        state[absorbed] = frAdd(state[absorbed], value)
        absorbed += 1
        if absorbed == 2 {
            state = poseidon2Permutation(state)
            absorbed = 0
        }
        needsPermute = true
    }

    private mutating func squeezeFrElement() -> Fr {
        if needsPermute || absorbed > 0 {
            state = poseidon2Permutation(state)
            absorbed = 0
            needsPermute = false
        }
        let result = state[0]
        needsPermute = true
        return result
    }
}

// MARK: - Blake3 Transcript Hasher

/// Blake3-based transcript hasher using NEON-accelerated hashing.
///
/// Unlike sponge constructions, Blake3 is a plain hash function.
/// We simulate absorb/squeeze by accumulating absorbed data and hashing
/// on squeeze with a monotonic counter for multi-squeeze support.
///
/// This matches the ad-hoc pattern used in IPAEngine and LassoEngine.
public struct Blake3TranscriptHasher: TranscriptHasher {
    private var buffer: [UInt8] = []
    private var squeezeCounter: UInt64 = 0

    public init() {}

    public mutating func absorb(_ data: [UInt8]) {
        buffer.append(contentsOf: data)
        // Reset squeeze counter on new absorb (state has changed)
        squeezeCounter = 0
    }

    public mutating func squeeze(byteCount: Int) -> [UInt8] {
        // Append squeeze counter to make each squeeze unique
        var counter = squeezeCounter
        let counterBytes = withUnsafeBytes(of: &counter) { Array($0) }
        let input = buffer + counterBytes
        squeezeCounter += 1

        // Hash with NEON-accelerated Blake3
        var hash = [UInt8](repeating: 0, count: 32)
        input.withUnsafeBufferPointer { inp in
            hash.withUnsafeMutableBufferPointer { out in
                blake3_hash_neon(inp.baseAddress!, inp.count, out.baseAddress!)
            }
        }

        if byteCount <= 32 {
            return Array(hash.prefix(byteCount))
        }

        // For >32 bytes, chain hashes
        var result = hash
        while result.count < byteCount {
            var nextCounter = squeezeCounter
            let nextCounterBytes = withUnsafeBytes(of: &nextCounter) { Array($0) }
            let nextInput = buffer + nextCounterBytes
            squeezeCounter += 1

            var nextHash = [UInt8](repeating: 0, count: 32)
            nextInput.withUnsafeBufferPointer { inp in
                nextHash.withUnsafeMutableBufferPointer { out in
                    blake3_hash_neon(inp.baseAddress!, inp.count, out.baseAddress!)
                }
            }
            result.append(contentsOf: nextHash)
        }
        return Array(result.prefix(byteCount))
    }

    public func clone() -> Blake3TranscriptHasher {
        var copy = Blake3TranscriptHasher()
        copy.buffer = self.buffer
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
