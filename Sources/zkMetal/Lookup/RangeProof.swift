// Range Proof via LogUp Lookup
// Proves that all values in a vector are in the range [0, R) where R is a power of 2.
//
// Approach: construct a table T = [0, 1, ..., R-1] and use the LogUp lookup argument
// to prove that every value in the witness appears in T.
//
// For large ranges, uses decomposition: split each value into limbs of `limbBits` bits each,
// then prove each limb is in [0, 2^limbBits). This reduces the table size from R to 2^limbBits.
//
// Applications: confidential transactions, zk-rollup balance checks, zkVM memory range checks.

import Foundation

/// A proof that all values in a vector are in [0, R)
public struct RangeProofResult {
    /// The range upper bound (exclusive)
    public let range: UInt64
    /// Number of values proven
    public let count: Int
    /// Whether decomposition was used
    public let decomposed: Bool
    /// Number of limbs per value (1 if not decomposed)
    public let numLimbs: Int
    /// Bits per limb
    public let limbBits: Int
    /// The underlying lookup proof(s)
    public let lookupProofs: [LookupProof]
}

public class RangeProofEngine {
    public let lookupEngine: LookupEngine

    public init() throws {
        self.lookupEngine = try LookupEngine()
    }

    /// Prove that all values are in [0, range) where range must be a power of 2.
    /// For range ≤ 2^16, uses direct table lookup.
    /// For larger ranges, decomposes into limbs.
    public func prove(values: [UInt64], range: UInt64) throws -> RangeProofResult {
        precondition(range > 0 && (range & (range - 1)) == 0, "Range must be power of 2")
        let rangeBits = Int(log2(Double(range)))

        // Check all values are in range
        for v in values {
            precondition(v < range, "Value \(v) out of range [0, \(range))")
        }

        // Pad values to power of 2
        let m = nextPowerOf2(values.count)

        if rangeBits <= 16 {
            // Direct: table = [0, 1, ..., range-1]
            let N = Int(range)
            let table = buildRangeTable(size: N)
            var lookups = [Fr](repeating: Fr.zero, count: m)
            for i in 0..<values.count { lookups[i] = frFromInt(values[i]) }

            let beta = deriveBeta(values: values, range: range)
            let proof = try lookupEngine.prove(table: table, lookups: lookups, beta: beta)

            return RangeProofResult(
                range: range,
                count: values.count,
                decomposed: false,
                numLimbs: 1,
                limbBits: rangeBits,
                lookupProofs: [proof]
            )
        } else {
            // Decompose into limbs
            let limbBits = 8  // 8-bit limbs for reasonable table size (256 entries)
            let numLimbs = (rangeBits + limbBits - 1) / limbBits
            let limbRange = 1 << limbBits
            let table = buildRangeTable(size: limbRange)

            var proofs = [LookupProof]()
            let limbM = nextPowerOf2(values.count)

            for limb in 0..<numLimbs {
                let shift = limb * limbBits
                let mask = UInt64(limbRange - 1)

                var limbValues = [Fr](repeating: Fr.zero, count: limbM)
                for (idx, v) in values.enumerated() {
                    limbValues[idx] = frFromInt((v >> shift) & mask)
                }

                let beta = deriveBetaForLimb(values: values, range: range, limb: limb)
                let proof = try lookupEngine.prove(table: table, lookups: limbValues, beta: beta)
                proofs.append(proof)
            }

            return RangeProofResult(
                range: range,
                count: values.count,
                decomposed: true,
                numLimbs: numLimbs,
                limbBits: limbBits,
                lookupProofs: proofs
            )
        }
    }

    /// Verify a range proof.
    public func verify(proof: RangeProofResult, values: [UInt64]) throws -> Bool {
        let rangeBits = Int(log2(Double(proof.range)))
        let m = nextPowerOf2(values.count)

        if !proof.decomposed {
            guard proof.lookupProofs.count == 1 else { return false }
            let N = Int(proof.range)
            let table = buildRangeTable(size: N)
            var lookups = [Fr](repeating: Fr.zero, count: m)
            for i in 0..<values.count { lookups[i] = frFromInt(values[i]) }
            return try lookupEngine.verify(
                proof: proof.lookupProofs[0], table: table, lookups: lookups)
        } else {
            guard proof.lookupProofs.count == proof.numLimbs else { return false }
            let limbRange = 1 << proof.limbBits
            let table = buildRangeTable(size: limbRange)
            let limbM = nextPowerOf2(values.count)

            for limb in 0..<proof.numLimbs {
                let shift = limb * proof.limbBits
                let mask = UInt64(limbRange - 1)

                var limbValues = [Fr](repeating: Fr.zero, count: limbM)
                for (idx, v) in values.enumerated() {
                    limbValues[idx] = frFromInt((v >> shift) & mask)
                }

                let valid = try lookupEngine.verify(
                    proof: proof.lookupProofs[limb], table: table, lookups: limbValues)
                if !valid { return false }
            }
            return true
        }
    }

    // MARK: - Helpers

    private func buildRangeTable(size: Int) -> [Fr] {
        (0..<size).map { frFromInt(UInt64($0)) }
    }

    private func nextPowerOf2(_ n: Int) -> Int {
        var p = 1
        while p < n { p <<= 1 }
        return p
    }

    private func deriveBeta(values: [UInt64], range: UInt64) -> Fr {
        var transcript = [UInt8]()
        // Seed with range and first few values
        var r = range
        for _ in 0..<8 { transcript.append(UInt8(r & 0xFF)); r >>= 8 }
        for v in values.prefix(min(16, values.count)) {
            var vv = v
            for _ in 0..<8 { transcript.append(UInt8(vv & 0xFF)); vv >>= 8 }
        }
        let hash = blake3(transcript)
        var limbs = [UInt64](repeating: 0, count: 4)
        for i in 0..<4 {
            for j in 0..<8 { limbs[i] |= UInt64(hash[i * 8 + j]) << (j * 8) }
        }
        let raw = Fr.from64(limbs)
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))
    }

    private func deriveBetaForLimb(values: [UInt64], range: UInt64, limb: Int) -> Fr {
        var transcript = [UInt8]()
        var r = range
        for _ in 0..<8 { transcript.append(UInt8(r & 0xFF)); r >>= 8 }
        var l = UInt64(limb)
        for _ in 0..<8 { transcript.append(UInt8(l & 0xFF)); l >>= 8 }
        for v in values.prefix(min(16, values.count)) {
            var vv = v
            for _ in 0..<8 { transcript.append(UInt8(vv & 0xFF)); vv >>= 8 }
        }
        let hash = blake3(transcript)
        var limbs = [UInt64](repeating: 0, count: 4)
        for i in 0..<4 {
            for j in 0..<8 { limbs[i] |= UInt64(hash[i * 8 + j]) << (j * 8) }
        }
        let raw = Fr.from64(limbs)
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))
    }
}
