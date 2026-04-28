// Poseidon2-based Fiat-Shamir transcript for Circle STARK
// Uses Poseidon2-M31 permutation (t=16) for field-native challenge derivation
//
// Advantages over Keccak-based transcript:
// - Field-native: squeeze() returns M31 directly (no uint32 conversion)
// - Potentially faster for ZK applications
// - No conversion overhead between field elements and bytes
//
// Security: Uses domain-separated labels and proper capacity initialization

import Foundation

public struct CircleSTARKPoseidon2Transcript {
    // Poseidon2 state: t=16 elements
    // Rate = 8 elements (indices 0-7), Capacity = 8 elements (indices 8-15)
    private var state: [M31]

    // Absorption buffer for M31 elements
    private var buffer: [M31]
    private let rate = 8

    public init() {
        self.state = [M31](repeating: M31.zero, count: 16)
        self.buffer = []
    }

    public mutating func absorbLabel(_ label: String) {
        let bytes = Array(label.utf8)
        var len = UInt32(bytes.count)
        let lenBytes = withUnsafeBytes(of: &len) { Array($0) }
        absorbBytes(lenBytes + bytes)
    }

    public mutating func absorbBytes(_ data: [UInt8]) {
        var elements: [M31] = []
        var i = 0
        while i < data.count {
            var val: UInt32 = 0
            var shift: UInt32 = 0
            while i < data.count && shift < 28 {
                val |= UInt32(data[i]) << shift
                shift += 8
                i += 1
            }
            let m31Val = val % M31.P
            elements.append(M31(v: m31Val == M31.P ? 0 : m31Val))
        }

        for elem in elements {
            absorb(elem)
        }
    }

    private mutating func absorb(_ elem: M31) {
        buffer.append(elem)
        if buffer.count == rate {
            for i in 0..<rate {
                state[i] = m31Add(state[i], buffer[i])
            }
            buffer.removeAll(keepingCapacity: true)
            poseidon2M31Permutation(state: &state)
        }
    }

    public mutating func absorbM31(_ v: M31) {
        buffer.append(v)
        if buffer.count == rate {
            for i in 0..<rate {
                state[i] = m31Add(state[i], buffer[i])
            }
            buffer.removeAll(keepingCapacity: true)
            poseidon2M31Permutation(state: &state)
        }
    }

    public mutating func absorbM31Many(_ values: [M31]) {
        for v in values {
            absorbM31(v)
        }
    }

    public mutating func squeezeM31() -> M31 {
        if !buffer.isEmpty {
            for i in 0..<rate {
                if i < buffer.count {
                    state[i] = m31Add(state[i], buffer[i])
                }
            }
            buffer.removeAll(keepingCapacity: true)
            poseidon2M31Permutation(state: &state)
        }

        let challenge = state[0]
        poseidon2M31Permutation(state: &state)
        return challenge
    }

    public mutating func squeezeM31Many(_ count: Int) -> [M31] {
        var results = [M31]()
        results.reserveCapacity(count)
        for _ in 0..<count {
            results.append(squeezeM31())
        }
        return results
    }

    public func stateHash() -> [M31] {
        return Array(state[0..<rate])
    }

    public mutating func forcePermutation() {
        if !buffer.isEmpty {
            for i in 0..<rate {
                if i < buffer.count {
                    state[i] = m31Add(state[i], buffer[i])
                }
            }
            buffer.removeAll(keepingCapacity: true)
        }
        poseidon2M31Permutation(state: &state)
    }
}

extension CircleSTARKPoseidon2Transcript {
    public mutating func absorbDigest(_ digest: M31Digest) {
        absorbM31Many(digest.values)
    }

    public mutating func absorbDigests(_ digests: [M31Digest]) {
        for digest in digests {
            absorbDigest(digest)
        }
    }
}
