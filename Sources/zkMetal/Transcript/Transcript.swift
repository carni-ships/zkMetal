// Fiat-Shamir Transcript — Unified sponge-based challenge generation
//
// Provides domain-separated absorb/squeeze API with configurable hash backend:
//   - Poseidon2: field-native (t=3, rate=2), fastest for Fr elements
//   - Keccak-256: byte-native (rate=136), Ethereum-compatible
//
// Sponge construction:
//   Absorb: mix data into rate portion, permute when rate is full
//   Squeeze: permute (if needed), read from rate portion
//
// Security properties:
//   - Deterministic: same inputs always produce same outputs
//   - Domain separation: different labels produce different challenge streams
//   - Fork-safe: child transcripts inherit parent state with fresh label

import Foundation
import NeonFieldOps

// MARK: - Hash Backend Protocol

/// Internal protocol for sponge backends
protocol TranscriptBackend {
    /// Absorb a single Fr element
    mutating func absorbFr(_ value: Fr)

    /// Absorb raw bytes
    mutating func absorbBytes(_ data: [UInt8])

    /// Squeeze one Fr element
    mutating func squeezeFr() -> Fr

    /// Squeeze n raw bytes
    mutating func squeezeBytes(_ n: Int) -> [UInt8]

    /// Create a copy of current state
    func clone() -> any TranscriptBackend
}

// MARK: - Transcript

/// A Fiat-Shamir transcript using a sponge construction.
///
/// Usage:
///   let t = Transcript(label: "my-protocol", backend: .poseidon2)
///   t.absorb(commitmentHash)
///   t.absorbLabel("round-1")
///   let challenge = t.squeeze()
public class Transcript {
    public enum HashBackend {
        case poseidon2    // Field-native, fastest for Fr elements
        case keccak256    // Byte-native, standard for Ethereum compatibility
    }

    private var backend: any TranscriptBackend
    private let backendType: HashBackend

    /// Create a new transcript with domain separation label.
    /// The label is absorbed immediately to bind the transcript to a specific protocol.
    public init(label: String, backend: HashBackend = .poseidon2) {
        self.backendType = backend
        switch backend {
        case .poseidon2:
            self.backend = Poseidon2Backend()
        case .keccak256:
            self.backend = KeccakBackend()
        }
        // Domain separation: absorb label into fresh state
        absorbLabel(label)
    }

    /// Internal init for fork (copies existing backend state)
    private init(backend: any TranscriptBackend, backendType: HashBackend) {
        self.backend = backend
        self.backendType = backendType
    }

    // MARK: - Absorb

    /// Absorb a field element into the transcript.
    public func absorb(_ value: Fr) {
        backend.absorbFr(value)
    }

    /// Absorb multiple field elements into the transcript.
    public func absorbMany(_ values: [Fr]) {
        for v in values {
            backend.absorbFr(v)
        }
    }

    /// Absorb raw bytes into the transcript.
    public func absorbBytes(_ data: [UInt8]) {
        backend.absorbBytes(data)
    }

    /// Absorb a label for domain separation between protocol steps.
    /// Encodes as: length (4 bytes LE) || UTF-8 bytes
    public func absorbLabel(_ label: String) {
        let bytes = Array(label.utf8)
        // Length prefix prevents collisions between labels
        var len = UInt32(bytes.count)
        let lenBytes = withUnsafeBytes(of: &len) { Array($0) }
        backend.absorbBytes(lenBytes + bytes)
    }

    // MARK: - Squeeze

    /// Squeeze a field element challenge from the transcript.
    public func squeeze() -> Fr {
        return backend.squeezeFr()
    }

    /// Squeeze multiple field element challenges.
    public func squeezeN(_ n: Int) -> [Fr] {
        (0..<n).map { _ in squeeze() }
    }

    /// Squeeze raw bytes from the transcript.
    public func squeezeBytes(_ n: Int) -> [UInt8] {
        return backend.squeezeBytes(n)
    }

    // MARK: - Fork

    /// Fork the transcript for parallel sub-protocols.
    /// The child inherits the current state and absorbs the fork label
    /// to ensure distinct challenge streams.
    public func fork(label: String) -> Transcript {
        let child = Transcript(backend: backend.clone(), backendType: backendType)
        child.absorbLabel(label)
        return child
    }
}

// MARK: - Poseidon2 Backend

/// Poseidon2 sponge: t=3 (rate=2, capacity=1), operates on Fr elements.
/// Absorb: XOR (additive) into rate positions, permute when rate is full.
/// Squeeze: permute, output state[0].
private struct Poseidon2Backend: TranscriptBackend {
    // Sponge state: [rate0, rate1, capacity]
    var state: [Fr] = [Fr.zero, Fr.zero, Fr.zero]
    // How many rate cells have been filled since last permute
    var absorbed: Int = 0
    // Whether we need to permute before squeezing
    var needsPermute: Bool = false

    mutating func absorbFr(_ value: Fr) {
        state[absorbed] = frAdd(state[absorbed], value)
        absorbed += 1
        if absorbed == 2 {
            // Rate is full, permute
            state = poseidon2Permutation(state)
            absorbed = 0
        }
        needsPermute = true
    }

    mutating func absorbBytes(_ data: [UInt8]) {
        // Pack bytes into Fr elements (31 bytes per element to stay < p)
        var offset = 0
        while offset < data.count {
            let chunkSize = min(31, data.count - offset)
            var limbs: [UInt64] = [0, 0, 0, 0]
            for i in 0..<chunkSize {
                let byteIdx = i
                let limbIdx = byteIdx / 8
                let shift = (byteIdx % 8) * 8
                limbs[limbIdx] |= UInt64(data[offset + i]) << shift
            }
            // Convert to Montgomery form
            let raw = Fr.from64(limbs)
            let mont = frMul(raw, Fr.from64(Fr.R2_MOD_R))
            absorbFr(mont)
            offset += chunkSize
        }
    }

    mutating func squeezeFr() -> Fr {
        // If anything was absorbed since last squeeze, or buffer partially filled, permute
        if needsPermute || absorbed > 0 {
            // Pad: if absorbed < rate, we've already got zeros in remaining positions
            state = poseidon2Permutation(state)
            absorbed = 0
            needsPermute = false
        }
        // Output from rate portion
        let result = state[0]
        // Subsequent squeezes need fresh permutations
        needsPermute = true
        return result
    }

    mutating func squeezeBytes(_ n: Int) -> [UInt8] {
        var result = [UInt8]()
        result.reserveCapacity(n)
        while result.count < n {
            let fr = squeezeFr()
            // Extract 32 bytes from Fr (in integer form)
            let limbs = frToInt(fr)
            for limb in limbs {
                for byte in 0..<8 {
                    if result.count < n {
                        result.append(UInt8((limb >> (byte * 8)) & 0xFF))
                    }
                }
            }
        }
        return result
    }

    func clone() -> any TranscriptBackend {
        var copy = Poseidon2Backend()
        copy.state = self.state
        copy.absorbed = self.absorbed
        copy.needsPermute = self.needsPermute
        return copy
    }
}

// MARK: - Keccak-256 Backend

/// Keccak sponge: rate=136 bytes (1088 bits), capacity=64 bytes (512 bits).
/// Uses NEON-accelerated Keccak-f[1600] permutation.
private struct KeccakBackend: TranscriptBackend {
    // Full 200-byte (1600-bit) Keccak state as 25 UInt64s
    var state: [UInt64] = [UInt64](repeating: 0, count: 25)
    // Current position within the rate (0..135)
    var rateOffset: Int = 0
    // Track absorb/squeeze mode transitions
    var squeezing: Bool = false

    static let rate: Int = 136  // bytes (Keccak-256: r = 1088 bits)

    mutating func absorbFr(_ value: Fr) {
        // Convert Fr to integer form, then XOR 4 limbs directly into state
        // (avoids intermediate [UInt8] allocation)
        let limbs = frToInt(value)
        if squeezing { squeezing = false }

        // XOR 4 x UInt64 = 32 bytes directly into state
        if rateOffset % 8 == 0 && rateOffset + 32 <= KeccakBackend.rate {
            // Fast path: word-aligned, no need for byte decomposition
            let wIdx = rateOffset / 8
            state[wIdx] ^= limbs[0]
            state[wIdx + 1] ^= limbs[1]
            state[wIdx + 2] ^= limbs[2]
            state[wIdx + 3] ^= limbs[3]
            rateOffset += 32
            if rateOffset == KeccakBackend.rate {
                permute()
                rateOffset = 0
            }
        } else {
            // Slow path: byte-by-byte
            var bytes = [UInt8]()
            bytes.reserveCapacity(32)
            for limb in limbs {
                for b in 0..<8 {
                    bytes.append(UInt8((limb >> (b * 8)) & 0xFF))
                }
            }
            absorbBytes(bytes)
        }
    }

    mutating func absorbBytes(_ data: [UInt8]) {
        if squeezing {
            squeezing = false
        }

        var offset = 0
        while offset < data.count {
            let available = KeccakBackend.rate - rateOffset
            let toAbsorb = min(available, data.count - offset)

            // XOR bytes into state at current rate position
            for i in 0..<toAbsorb {
                let pos = rateOffset + i
                let wordIdx = pos / 8
                let byteIdx = pos % 8
                state[wordIdx] ^= UInt64(data[offset + i]) << (byteIdx * 8)
            }

            rateOffset += toAbsorb
            offset += toAbsorb

            if rateOffset == KeccakBackend.rate {
                permute()
                rateOffset = 0
            }
        }
    }

    mutating func squeezeFr() -> Fr {
        if !squeezing {
            let pos0 = rateOffset
            state[pos0 / 8] ^= UInt64(0x01) << ((pos0 % 8) * 8)
            let lastPos = KeccakBackend.rate - 1
            state[lastPos / 8] ^= UInt64(0x80) << ((lastPos % 8) * 8)
            permute()
            rateOffset = 0
            squeezing = true
        }

        // Fast path: read 4 limbs directly from state (no intermediate [UInt8])
        if rateOffset % 8 == 0 && rateOffset + 32 <= KeccakBackend.rate {
            let wIdx = rateOffset / 8
            var l3 = state[wIdx + 3]
            l3 &= 0x3FFFFFFFFFFFFFFF
            rateOffset += 32
            let raw = Fr.from64([state[wIdx], state[wIdx + 1], state[wIdx + 2], l3])
            return frMul(raw, Fr.from64(Fr.R2_MOD_R))
        }

        // Fallback
        let bytes = squeezeBytes(32)
        var limbs = [UInt64](repeating: 0, count: 4)
        for i in 0..<4 {
            for j in 0..<8 {
                limbs[i] |= UInt64(bytes[i * 8 + j]) << (j * 8)
            }
        }
        limbs[3] &= 0x3FFFFFFFFFFFFFFF
        let raw = Fr.from64(limbs)
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))
    }

    mutating func squeezeBytes(_ n: Int) -> [UInt8] {
        if !squeezing {
            // Transition from absorb to squeeze
            // Apply padding: 10*1 padding
            // XOR 0x01 at current position, XOR 0x80 at last rate byte
            let pos0 = rateOffset
            let wordIdx0 = pos0 / 8
            let byteIdx0 = pos0 % 8
            state[wordIdx0] ^= UInt64(0x01) << (byteIdx0 * 8)

            let lastPos = KeccakBackend.rate - 1
            let wordIdxLast = lastPos / 8
            let byteIdxLast = lastPos % 8
            state[wordIdxLast] ^= UInt64(0x80) << (byteIdxLast * 8)

            permute()
            rateOffset = 0
            squeezing = true
        }

        var result = [UInt8]()
        result.reserveCapacity(n)

        while result.count < n {
            if rateOffset >= KeccakBackend.rate {
                permute()
                rateOffset = 0
            }
            let available = KeccakBackend.rate - rateOffset
            let toSqueeze = min(available, n - result.count)

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

    private mutating func permute() {
        // Use NEON-accelerated Keccak-f[1600]
        state.withUnsafeMutableBufferPointer { buf in
            keccak_f1600_neon(buf.baseAddress!)
        }
    }

    func clone() -> any TranscriptBackend {
        var copy = KeccakBackend()
        copy.state = self.state
        copy.rateOffset = self.rateOffset
        copy.squeezing = self.squeezing
        return copy
    }
}
