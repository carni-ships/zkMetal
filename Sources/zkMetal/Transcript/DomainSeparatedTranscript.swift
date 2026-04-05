// DomainSeparatedTranscript — Hardened Fiat-Shamir with domain separation
//
// Wraps an existing TranscriptEngine with:
//   - Per-operation string labels hashed into state (prevents cross-protocol attacks)
//   - Length-prefixed encoding for all absorbed data (prevents length extension)
//   - Protocol identification via beginProtocol()
//   - Fork support for recursive composition
//
// Every absorb and squeeze call includes a label that becomes part of the hash
// input, so identical data absorbed under different labels produces different
// transcript states. Length prefixes on all variable-length data prevent an
// attacker from shifting bytes between adjacent absorb calls.

import Foundation
import NeonFieldOps

// MARK: - DomainSeparatedTranscript

/// A hardened Fiat-Shamir transcript that enforces domain separation on every operation.
///
/// Each `absorb*` and `squeeze*` call takes a string label that is hashed into the
/// transcript state before the payload, preventing cross-protocol and cross-step
/// confusion attacks. All variable-length data is length-prefixed to prevent
/// length-extension attacks where `absorb([1,2]) + absorb([3])` could collide
/// with `absorb([1]) + absorb([2,3])`.
///
/// Usage:
///   ```
///   var t = DomainSeparatedTranscript(hash: .keccak)
///   t.beginProtocol("UltraPlonk-v1")
///   t.absorbScalar("public_input_0", publicInput)
///   t.absorbPoint("commitment_a", commitA)
///   let alpha = t.squeezeChallenge("alpha")
///   ```
public struct DomainSeparatedTranscript {
    private var inner: FiatShamirTranscript<KeccakTranscriptHasher>?
    private var innerPoseidon: FiatShamirTranscript<Poseidon2TranscriptHasher>?
    private var innerBlake3: FiatShamirTranscript<Blake3TranscriptHasher>?
    private let hashType: TranscriptHashType

    /// The number of labeled operations performed on this transcript.
    public var operationCount: UInt64 {
        switch hashType {
        case .keccak:   return inner!.operationCount
        case .poseidon2: return innerPoseidon!.operationCount
        case .blake3:   return innerBlake3!.operationCount
        }
    }

    /// Create a domain-separated transcript with the specified hash backend.
    ///
    /// The transcript is initialized with a fixed domain tag identifying it as a
    /// domain-separated transcript, ensuring it cannot collide with raw transcripts.
    ///
    /// - Parameter hash: The hash backend to use (.keccak, .poseidon2, or .blake3)
    public init(hash: TranscriptHashType) {
        self.hashType = hash
        switch hash {
        case .keccak:
            self.inner = KeccakTranscript(label: "DomainSeparatedTranscript-v1")
        case .poseidon2:
            self.innerPoseidon = Poseidon2Transcript(label: "DomainSeparatedTranscript-v1")
        case .blake3:
            self.innerBlake3 = Blake3Transcript(label: "DomainSeparatedTranscript-v1")
        }
    }

    // MARK: - Protocol Identification

    /// Begin a named protocol, absorbing its identifier into the transcript state.
    ///
    /// This should be called once at the start of a proof to bind the transcript
    /// to a specific protocol, preventing challenges from one protocol being
    /// replayed in another.
    ///
    /// - Parameter name: Protocol identifier (e.g., "UltraPlonk-v1", "Groth16")
    public mutating func beginProtocol(_ name: String) {
        absorbTaggedData("begin_protocol", Array(name.utf8))
    }

    // MARK: - Absorb Operations

    /// Absorb a field element with a domain separation label.
    ///
    /// Internally: absorb(tag("scalar", label) || len(32) || Fr_bytes)
    ///
    /// - Parameters:
    ///   - label: Operation label (e.g., "public_input_0", "alpha")
    ///   - value: Field element to absorb
    public mutating func absorbScalar(_ label: String, _ value: Fr) {
        let tag = domainTag("scalar", label)
        var bytes = [UInt8](repeating: 0, count: 32)
        withUnsafeBytes(of: value) { buf in
            let ptr = buf.baseAddress!.assumingMemoryBound(to: UInt8.self)
            for i in 0..<32 { bytes[i] = ptr[i] }
        }
        absorbTaggedData(tag, bytes)
    }

    /// Absorb an elliptic curve point with a domain separation label.
    ///
    /// Converts the projective point to affine coordinates and absorbs both
    /// x and y as length-prefixed 32-byte field elements under the label.
    ///
    /// - Parameters:
    ///   - label: Operation label (e.g., "commitment_a", "W_z")
    ///   - point: Projective curve point to absorb
    public mutating func absorbPoint(_ label: String, _ point: PointProjective) {
        let tag = domainTag("point", label)
        var affine = [UInt64](repeating: 0, count: 8)
        withUnsafeBytes(of: point) { pBuf in
            affine.withUnsafeMutableBufferPointer { aBuf in
                bn254_projective_to_affine(
                    pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    aBuf.baseAddress!
                )
            }
        }
        // Serialize affine x, y as 64 bytes total
        var bytes = [UInt8](repeating: 0, count: 64)
        for i in 0..<8 {
            for j in 0..<8 {
                bytes[i * 8 + j] = UInt8((affine[i] >> (j * 8)) & 0xFF)
            }
        }
        absorbTaggedData(tag, bytes)
    }

    /// Absorb raw bytes with a domain separation label.
    ///
    /// The data is length-prefixed before absorption, so `absorbBytes("x", [1,2])`
    /// followed by `absorbBytes("y", [3])` can never collide with
    /// `absorbBytes("x", [1])` followed by `absorbBytes("y", [2,3])`.
    ///
    /// - Parameters:
    ///   - label: Operation label
    ///   - data: Raw bytes to absorb
    public mutating func absorbBytes(_ label: String, _ data: [UInt8]) {
        let tag = domainTag("bytes", label)
        absorbTaggedData(tag, data)
    }

    // MARK: - Squeeze Operations

    /// Squeeze a field element challenge with a domain separation label.
    ///
    /// The label is absorbed into the state before squeezing, ensuring that
    /// `squeezeChallenge("alpha")` and `squeezeChallenge("beta")` produce
    /// different values even from the same state.
    ///
    /// - Parameter label: Challenge label (e.g., "alpha", "beta", "gamma")
    /// - Returns: A uniformly distributed field element in Montgomery form
    public mutating func squeezeChallenge(_ label: String) -> Fr {
        let tag = domainTag("challenge", label)
        absorbDomainLabel(tag)
        switch hashType {
        case .keccak:   return inner!.squeezeChallenge()
        case .poseidon2: return innerPoseidon!.squeezeChallenge()
        case .blake3:   return innerBlake3!.squeezeChallenge()
        }
    }

    /// Squeeze raw bytes with a domain separation label.
    ///
    /// - Parameters:
    ///   - label: Operation label
    ///   - count: Number of bytes to squeeze
    /// - Returns: `count` pseudorandom bytes
    public mutating func squeezeBytes(_ label: String, count: Int) -> [UInt8] {
        let tag = domainTag("squeeze_bytes", label)
        switch hashType {
        case .keccak:   return inner!.squeeze(tag, byteCount: count)
        case .poseidon2: return innerPoseidon!.squeeze(tag, byteCount: count)
        case .blake3:   return innerBlake3!.squeeze(tag, byteCount: count)
        }
    }

    // MARK: - Fork

    /// Fork the transcript for parallel sub-protocols or recursive composition.
    ///
    /// The child transcript inherits the full parent state and absorbs a unique
    /// fork label, ensuring its challenge stream diverges from the parent and
    /// from other forks with different labels. The parent is not modified.
    ///
    /// - Parameter label: Unique label for this fork (e.g., "recursive-layer-0")
    /// - Returns: An independent `DomainSeparatedTranscript` with diverged state
    public func fork(_ label: String) -> DomainSeparatedTranscript {
        let forkTag = domainTag("fork", label)
        var child = DomainSeparatedTranscript(cloning: self)
        child.absorbDomainLabel(forkTag)
        return child
    }

    // MARK: - Internal Helpers

    /// Build a domain tag string: "type:label"
    private func domainTag(_ type: String, _ label: String) -> String {
        return "\(type):\(label)"
    }

    /// Absorb a domain label into the underlying transcript.
    /// Encodes as: len(tag_utf8) as 4-byte LE || tag_utf8 bytes
    private mutating func absorbDomainLabel(_ tag: String) {
        let tagBytes = Array(tag.utf8)
        var tagLen = UInt32(tagBytes.count)
        let tagLenBytes = withUnsafeBytes(of: &tagLen) { Array($0) }
        let payload = tagLenBytes + tagBytes
        switch hashType {
        case .keccak:   inner!.absorbRaw(payload)
        case .poseidon2: innerPoseidon!.absorbRaw(payload)
        case .blake3:   innerBlake3!.absorbRaw(payload)
        }
    }

    /// Absorb tagged, length-prefixed data into the underlying transcript.
    /// Format: domain_label || len(data) as 4-byte LE || data
    private mutating func absorbTaggedData(_ tag: String, _ data: [UInt8]) {
        let tagBytes = Array(tag.utf8)
        var tagLen = UInt32(tagBytes.count)
        let tagLenBytes = withUnsafeBytes(of: &tagLen) { Array($0) }
        var dataLen = UInt32(data.count)
        let dataLenBytes = withUnsafeBytes(of: &dataLen) { Array($0) }
        // Full payload: tagLen || tag || dataLen || data
        let payload = tagLenBytes + tagBytes + dataLenBytes + data
        switch hashType {
        case .keccak:
            inner!.appendMessage(label: tag, data: payload)
        case .poseidon2:
            innerPoseidon!.appendMessage(label: tag, data: payload)
        case .blake3:
            innerBlake3!.appendMessage(label: tag, data: payload)
        }
    }

    /// Clone constructor for fork.
    private init(cloning other: DomainSeparatedTranscript) {
        self.hashType = other.hashType
        switch other.hashType {
        case .keccak:
            self.inner = other.inner!.forkTyped(label: "__clone__")
            // Re-init: we want to clone state, not add fork label yet.
            // forkTyped adds "__clone__" label which is fine — fork() adds its own tag on top.
        case .poseidon2:
            self.innerPoseidon = other.innerPoseidon!.forkTyped(label: "__clone__")
        case .blake3:
            self.innerBlake3 = other.innerBlake3!.forkTyped(label: "__clone__")
        }
    }
}
