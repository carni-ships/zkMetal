// Poseidon2 Sponge — duplex sponge construction for arbitrary-length hashing
// and Fiat-Shamir transcript generation.
//
// Supports two configurations:
//   - BN254 Fr: t=3, rate=2, capacity=1 (default for ZK proofs)
//   - BabyBear: t=16, rate=8, capacity=8 (for Plonky3/SP1 compatibility)
//
// Sponge construction:
//   Absorb: XOR (additive) data into rate portion, permute when rate is full
//   Squeeze: permute (if needed), read from rate portion
//   Duplex: interleave absorb/squeeze freely for transcript use
//
// Domain separation: initial state encodes domain tag in capacity element(s)
// to ensure distinct hash domains produce distinct outputs even with identical data.

import Foundation
import NeonFieldOps

// MARK: - Sponge Configuration

/// Configuration for Poseidon2 sponge instances.
public enum Poseidon2SpongeConfig {
    /// BN254 Fr: t=3, rate=2, capacity=1, x^5 S-box
    case bn254Fr
    /// BabyBear: t=16, rate=8, capacity=8, x^7 S-box
    case babyBear

    public var width: Int {
        switch self {
        case .bn254Fr: return 3
        case .babyBear: return 16
        }
    }

    public var rate: Int {
        switch self {
        case .bn254Fr: return 2
        case .babyBear: return 8
        }
    }

    public var capacity: Int {
        switch self {
        case .bn254Fr: return 1
        case .babyBear: return 8
        }
    }
}

// MARK: - Poseidon2 Sponge (BN254 Fr)

/// Duplex sponge construction over BN254 Fr using Poseidon2 permutation.
///
/// Supports arbitrary-length absorb and squeeze with domain separation.
/// The sponge maintains internal state of 3 Fr elements: [rate0, rate1, capacity].
///
/// Usage:
///   var sponge = Poseidon2Sponge(domainTag: 0)
///   sponge.absorb(elements: [a, b, c, d, e])
///   let hash = sponge.squeeze(count: 1)
///
/// Duplex mode (for transcripts):
///   var sponge = Poseidon2Sponge(domainTag: 1)
///   sponge.absorb(elements: [commitment])
///   let challenge1 = sponge.squeeze(count: 1)
///   sponge.absorb(elements: [response])
///   let challenge2 = sponge.squeeze(count: 1)
public struct Poseidon2Sponge {
    /// Sponge state: [rate0, rate1, capacity]
    private var state: (Fr, Fr, Fr)
    /// Number of rate cells filled since last permutation
    private var absorbed: Int = 0
    /// Whether state has been modified since last squeeze
    private var dirty: Bool = false

    /// Create a new sponge with domain separation tag.
    ///
    /// The domain tag is placed in the capacity element of the initial state,
    /// ensuring different domains produce different hash outputs even with
    /// identical input data.
    ///
    /// - Parameter domainTag: Integer domain separator (0 = generic hash,
    ///   1 = transcript, 2 = Merkle, etc.)
    public init(domainTag: UInt64 = 0) {
        let tag = frFromInt(domainTag)
        self.state = (Fr.zero, Fr.zero, tag)
    }

    /// Create a sponge with an Fr element as the domain tag.
    public init(domainTagFr: Fr) {
        self.state = (Fr.zero, Fr.zero, domainTagFr)
    }

    // MARK: - Absorb

    /// Absorb field elements into the sponge.
    ///
    /// Elements are added to the rate portion of the state. When the rate
    /// is full (2 elements absorbed), the permutation is applied automatically.
    /// Handles arbitrary-length input via multi-block absorption.
    ///
    /// - Parameter elements: Field elements to absorb
    public mutating func absorb(elements: [Fr]) {
        var i = 0
        while i < elements.count {
            // Add element to current rate position
            switch absorbed {
            case 0:
                state.0 = frAdd(state.0, elements[i])
            case 1:
                state.1 = frAdd(state.1, elements[i])
            default:
                fatalError("Internal error: absorbed count out of range")
            }
            absorbed += 1
            dirty = true

            if absorbed == 2 {
                // Rate is full, permute
                permute()
                absorbed = 0
            }
            i += 1
        }
    }

    /// Absorb a single field element (zero-allocation fast path).
    public mutating func absorb(element: Fr) {
        switch absorbed {
        case 0:
            state.0 = frAdd(state.0, element)
        case 1:
            state.1 = frAdd(state.1, element)
        default:
            fatalError("Internal error: absorbed count out of range")
        }
        absorbed += 1
        dirty = true

        if absorbed == 2 {
            permute()
            absorbed = 0
        }
    }

    /// Absorb raw bytes, packing into Fr elements (31 bytes per element).
    ///
    /// Bytes are packed in little-endian order, 31 bytes per Fr element
    /// to ensure the value stays below the field modulus.
    public mutating func absorbBytes(_ data: [UInt8]) {
        var offset = 0
        while offset < data.count {
            let chunkSize = min(31, data.count - offset)
            var limbs: [UInt64] = [0, 0, 0, 0]
            for j in 0..<chunkSize {
                let limbIdx = j / 8
                let shift = (j % 8) * 8
                limbs[limbIdx] |= UInt64(data[offset + j]) << shift
            }
            let raw = Fr.from64(limbs)
            let mont = frMul(raw, Fr.from64(Fr.R2_MOD_R))
            absorb(element: mont)
            offset += chunkSize
        }
    }

    // MARK: - Squeeze

    /// Squeeze field elements from the sponge.
    ///
    /// If any elements have been absorbed since the last squeeze (or if this
    /// is the first squeeze), the permutation is applied first. Each subsequent
    /// squeeze within the same squeeze phase reads the next rate element, then
    /// permutes again when the rate is exhausted.
    ///
    /// - Parameter count: Number of field elements to squeeze
    /// - Returns: Array of squeezed field elements
    public mutating func squeeze(count: Int) -> [Fr] {
        var result = [Fr]()
        result.reserveCapacity(count)

        // If we have absorbed data, finalize with a permutation
        if dirty || absorbed > 0 {
            permute()
            absorbed = 0
            dirty = false
        }

        var squeezePos = 0
        while result.count < count {
            if squeezePos >= 2 {
                // Exhausted rate, permute for more output
                permute()
                squeezePos = 0
            }
            switch squeezePos {
            case 0: result.append(state.0)
            case 1: result.append(state.1)
            default: break
            }
            squeezePos += 1
        }

        // Mark dirty so next absorb/squeeze knows state was read
        dirty = true
        return result
    }

    /// Squeeze a single field element (zero-allocation fast path).
    public mutating func squeezeOne() -> Fr {
        // If we have absorbed data, finalize with a permutation
        if dirty || absorbed > 0 {
            permute()
            absorbed = 0
            dirty = false
        }
        let result = state.0
        // Mark dirty so next absorb/squeeze knows state was read
        dirty = true
        return result
    }

    // MARK: - Convenience

    /// One-shot: absorb input, squeeze output.
    ///
    /// Equivalent to creating a fresh sponge, absorbing all input, and squeezing.
    /// - Parameters:
    ///   - input: Field elements to absorb
    ///   - outputCount: Number of field elements to squeeze
    ///   - domainTag: Domain separation tag (default 0)
    /// - Returns: Squeezed field elements
    public static func absorbAndSqueeze(
        input: [Fr],
        outputCount: Int,
        domainTag: UInt64 = 0
    ) -> [Fr] {
        var sponge = Poseidon2Sponge(domainTag: domainTag)
        sponge.absorb(elements: input)
        return sponge.squeeze(count: outputCount)
    }

    /// Hash arbitrary-length input to a single Fr element.
    public static func hash(_ input: [Fr], domainTag: UInt64 = 0) -> Fr {
        return absorbAndSqueeze(input: input, outputCount: 1, domainTag: domainTag)[0]
    }

    // MARK: - Internal

    /// Apply the Poseidon2 permutation to the internal state.
    /// Uses zero-allocation in-place permutation.
    private mutating func permute() {
        poseidon2PermuteInPlace(&state.0, &state.1, &state.2)
    }

    /// Clone the sponge state for fork operations.
    public func clone() -> Poseidon2Sponge {
        var copy = Poseidon2Sponge(domainTag: 0)
        copy.state = self.state
        copy.absorbed = self.absorbed
        copy.dirty = self.dirty
        return copy
    }
}

// MARK: - Poseidon2 BabyBear Sponge

/// Duplex sponge construction over BabyBear using Poseidon2 permutation.
///
/// Uses t=16 (rate=8, capacity=8), matching SP1/Plonky3 parameters.
/// Each hash output is 8 BabyBear elements.
///
/// Usage:
///   var sponge = Poseidon2BbSponge(domainTag: 0)
///   sponge.absorb(elements: inputBbElements)
///   let hash = sponge.squeeze(count: 8)
public struct Poseidon2BbSponge {
    /// Sponge state: 16 BabyBear elements [rate(8) | capacity(8)]
    private var state: [Bb]
    /// Number of rate cells filled since last permutation
    private var absorbed: Int = 0
    /// Whether state has been modified since last squeeze
    private var dirty: Bool = false

    /// Create a new BabyBear sponge with domain separation tag.
    ///
    /// The domain tag is placed in the first capacity element (index 8).
    public init(domainTag: UInt32 = 0) {
        self.state = [Bb](repeating: Bb.zero, count: 16)
        self.state[8] = Bb(v: domainTag)
    }

    // MARK: - Absorb

    /// Absorb BabyBear elements into the sponge.
    public mutating func absorb(elements: [Bb]) {
        var i = 0
        while i < elements.count {
            state[absorbed] = bbAdd(state[absorbed], elements[i])
            absorbed += 1
            dirty = true

            if absorbed == 8 {
                permute()
                absorbed = 0
            }
            i += 1
        }
    }

    /// Absorb a single BabyBear element.
    public mutating func absorb(element: Bb) {
        absorb(elements: [element])
    }

    // MARK: - Squeeze

    /// Squeeze BabyBear elements from the sponge.
    public mutating func squeeze(count: Int) -> [Bb] {
        var result = [Bb]()
        result.reserveCapacity(count)

        if dirty || absorbed > 0 {
            permute()
            absorbed = 0
            dirty = false
        }

        var squeezePos = 0
        while result.count < count {
            if squeezePos >= 8 {
                permute()
                squeezePos = 0
            }
            result.append(state[squeezePos])
            squeezePos += 1
        }

        dirty = true
        return result
    }

    /// Squeeze 8 BabyBear elements (one full rate block).
    public mutating func squeezeBlock() -> [Bb] {
        return squeeze(count: 8)
    }

    // MARK: - Convenience

    /// One-shot: absorb input, squeeze output.
    public static func absorbAndSqueeze(
        input: [Bb],
        outputCount: Int,
        domainTag: UInt32 = 0
    ) -> [Bb] {
        var sponge = Poseidon2BbSponge(domainTag: domainTag)
        sponge.absorb(elements: input)
        return sponge.squeeze(count: outputCount)
    }

    /// Hash arbitrary-length input to 8 BabyBear elements.
    public static func hash(_ input: [Bb], domainTag: UInt32 = 0) -> [Bb] {
        return absorbAndSqueeze(input: input, outputCount: 8, domainTag: domainTag)
    }

    // MARK: - Internal

    private mutating func permute() {
        poseidon2BbPermutation(state: &state)
    }

    /// Clone the sponge state.
    public func clone() -> Poseidon2BbSponge {
        var copy = Poseidon2BbSponge(domainTag: 0)
        copy.state = self.state
        copy.absorbed = self.absorbed
        copy.dirty = self.dirty
        return copy
    }
}

// MARK: - Poseidon2 Sponge Transcript

/// Fiat-Shamir transcript using Poseidon2 sponge in duplex mode.
///
/// Conforms to TranscriptHasher for use with FiatShamirTranscript.
/// Uses the Poseidon2Sponge internally with domain tag = 1 (transcript mode).
///
/// This provides a higher-level alternative to Poseidon2TranscriptHasher
/// with explicit sponge semantics and domain separation.
///
/// Usage:
///   var transcript = Poseidon2SpongeTranscript()
///   transcript.absorbFr(commitment)
///   let challenge = transcript.squeezeFr()
///   transcript.absorbFr(response)
///   let nextChallenge = transcript.squeezeFr()
public struct Poseidon2SpongeTranscript: TranscriptHasher {
    private var sponge: Poseidon2Sponge

    /// Create a transcript with the default domain tag (1 = transcript mode).
    public init() {
        self.sponge = Poseidon2Sponge(domainTag: 1)
    }

    /// Create a transcript with a custom domain tag.
    public init(domainTag: UInt64) {
        self.sponge = Poseidon2Sponge(domainTag: domainTag)
    }

    // MARK: - Field-native Operations

    /// Absorb a field element directly (no byte conversion overhead).
    public mutating func absorbFr(_ value: Fr) {
        sponge.absorb(element: value)
    }

    /// Absorb multiple field elements.
    public mutating func absorbFrMany(_ values: [Fr]) {
        sponge.absorb(elements: values)
    }

    /// Squeeze a field element challenge.
    public mutating func squeezeFr() -> Fr {
        return sponge.squeezeOne()
    }

    /// Squeeze multiple field element challenges.
    public mutating func squeezeFrMany(_ count: Int) -> [Fr] {
        return sponge.squeeze(count: count)
    }

    // MARK: - TranscriptHasher Conformance

    /// Absorb raw bytes (packed into Fr elements, 31 bytes each).
    public mutating func absorb(_ data: [UInt8]) {
        sponge.absorbBytes(data)
    }

    /// Squeeze bytes from the sponge.
    public mutating func squeeze(byteCount: Int) -> [UInt8] {
        var result = [UInt8]()
        result.reserveCapacity(byteCount)
        while result.count < byteCount {
            let fr = sponge.squeezeOne()
            let limbs = frToInt(fr)
            for limb in limbs {
                for b in 0..<8 {
                    if result.count < byteCount {
                        result.append(UInt8((limb >> (b * 8)) & 0xFF))
                    }
                }
            }
        }
        return result
    }

    /// Clone the transcript state for fork operations.
    public func clone() -> Poseidon2SpongeTranscript {
        var copy = Poseidon2SpongeTranscript()
        copy.sponge = self.sponge.clone()
        return copy
    }
}

/// Convenience type alias for Poseidon2 sponge-backed Fiat-Shamir transcript.
public typealias Poseidon2SpongeFiatShamir = FiatShamirTranscript<Poseidon2SpongeTranscript>
