// GPUPlookupEngine — GPU-accelerated Plookup lookup argument
//
// Protocol (Gabizon-Williamson 2020, "plookup"):
//   Given table t[0..N-1] and witness f[0..n-1] where each f[i] in t:
//
//   1. Sort: merge f and t into sorted vector s of length n+N, such that s
//      is sorted w.r.t. the ordering in t (duplicates from f placed adjacent
//      to their matching table entry).
//
//   2. Grand products: with random challenges beta, gamma from the verifier,
//      prove the multiset equality via:
//
//        Z[0] = 1
//        Z[i+1] = Z[i] * (1 + beta)^{1} * (gamma + f[i]) * (gamma(1+beta) + t[i] + beta*t[i+1])
//                  / ( (gamma(1+beta) + s[2i] + beta*s[2i+1]) * (gamma(1+beta) + s[2i+1] + beta*s[2i+2]) )
//
//        where the last Z[n] must equal 1 (the accumulator "closes").
//
//   3. Polynomial commitments: commit to f, t, s, Z and prove evaluations
//      at a challenge point zeta via KZG or IPA.
//
// GPU acceleration targets:
//   - Sorted merge construction: parallel merge via key-value radix sort
//   - Grand product accumulator: GPU batch inverse + prefix product
//   - Polynomial commitments: GPU MSM (via KZG engine)
//
// References:
//   plookup: Gabizon-Williamson 2020 (eprint 2020/315)
//   Plonk+plookup: Gabizon-Williamson 2020

import Foundation
import Metal

// MARK: - Proof Structure

/// Proof produced by the Plookup protocol.
public struct PlookupProof {
    /// Sorted vector s (merged witness + table)
    public let sortedVector: [Fr]
    /// Grand product accumulator polynomial evaluations
    public let accumulatorZ: [Fr]
    /// Random challenges used
    public let beta: Fr
    public let gamma: Fr
    /// Claimed product at the end: Z[n] should be 1
    public let finalAccumulator: Fr
    /// KZG commitments (if available)
    public let fCommitment: PointProjective?
    public let tCommitment: PointProjective?
    public let sCommitment: PointProjective?
    public let zCommitment: PointProjective?
    /// Evaluation proof at challenge zeta
    public let zeta: Fr?
    public let fEvalAtZeta: Fr?
    public let tEvalAtZeta: Fr?
    public let sEvalAtZeta: Fr?
    public let zEvalAtZeta: Fr?

    public init(sortedVector: [Fr], accumulatorZ: [Fr],
                beta: Fr, gamma: Fr, finalAccumulator: Fr,
                fCommitment: PointProjective? = nil,
                tCommitment: PointProjective? = nil,
                sCommitment: PointProjective? = nil,
                zCommitment: PointProjective? = nil,
                zeta: Fr? = nil,
                fEvalAtZeta: Fr? = nil,
                tEvalAtZeta: Fr? = nil,
                sEvalAtZeta: Fr? = nil,
                zEvalAtZeta: Fr? = nil) {
        self.sortedVector = sortedVector
        self.accumulatorZ = accumulatorZ
        self.beta = beta
        self.gamma = gamma
        self.finalAccumulator = finalAccumulator
        self.fCommitment = fCommitment
        self.tCommitment = tCommitment
        self.sCommitment = sCommitment
        self.zCommitment = zCommitment
        self.zeta = zeta
        self.fEvalAtZeta = fEvalAtZeta
        self.tEvalAtZeta = tEvalAtZeta
        self.sEvalAtZeta = sEvalAtZeta
        self.zEvalAtZeta = zEvalAtZeta
    }
}

// MARK: - Engine

/// GPU-accelerated Plookup lookup argument engine.
///
/// Proves that every element in witness vector f is contained in table vector t,
/// using the sorted-merge + grand-product approach from the plookup paper.
///
/// GPU acceleration:
///   - Sorted merge via GPU radix sort (key-value)
///   - Grand product accumulator via GPUGrandProductEngine (batch inverse + prefix product)
///   - Optional KZG commitments via GPU MSM
public class GPUPlookupEngine {
    public static let version = Versions.plookup

    private let grandProductEngine: GPUGrandProductEngine
    private let polyEngine: PolyEngine
    private var kzgEngine: KZGEngine?

    /// Emit profiling info to stderr
    public var profile = false

    /// GPU threshold: arrays smaller than this use CPU path for sort/products
    public var gpuThreshold: Int = 4096

    public init(srs: [PointAffine]? = nil) throws {
        self.grandProductEngine = try GPUGrandProductEngine()
        self.polyEngine = try PolyEngine()
        if let srs = srs, !srs.isEmpty {
            self.kzgEngine = try KZGEngine(srs: srs)
        }
    }

    // MARK: - Prove

    /// Full Plookup proof: proves that every element in `witness` exists in `table`.
    ///
    /// - Parameters:
    ///   - witness: The lookup vector f. All elements must be in table.
    ///   - table: The table vector t. Must contain all values referenced by witness.
    ///   - beta: Random challenge (normally from Fiat-Shamir). Pass nil for auto-derivation.
    ///   - gamma: Random challenge (normally from Fiat-Shamir). Pass nil for auto-derivation.
    /// - Returns: A PlookupProof.
    public func prove(witness: [Fr], table: [Fr],
                      beta: Fr? = nil, gamma: Fr? = nil) throws -> PlookupProof {
        let n = witness.count
        let N = table.count
        precondition(n > 0, "Witness must be non-empty")
        precondition(N > 0, "Table must be non-empty")

        let _t0 = profile ? CFAbsoluteTimeGetCurrent() : 0
        var _tPhase = profile ? CFAbsoluteTimeGetCurrent() : 0

        // Step 1: Derive or use provided challenges
        let betaVal = beta ?? deriveBeta(witness: witness, table: table)
        let gammaVal = gamma ?? deriveGamma(witness: witness, table: table, beta: betaVal)

        if profile { let _t = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [plookup] challenge derivation: %.2f ms\n", (_t - _tPhase) * 1000), stderr); _tPhase = _t }

        // Step 2: Build sorted vector s by merging f into t
        let sorted = try buildSortedVector(witness: witness, table: table)

        if profile { let _t = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [plookup] sorted merge (n=%d, N=%d, |s|=%d): %.2f ms\n", n, N, sorted.count, (_t - _tPhase) * 1000), stderr); _tPhase = _t }

        // Step 3: Build grand product accumulator Z
        let accZ = try buildAccumulator(witness: witness, table: table,
                                         sorted: sorted, beta: betaVal, gamma: gammaVal)

        if profile { let _t = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [plookup] grand product accumulator (len=%d): %.2f ms\n", accZ.count, (_t - _tPhase) * 1000), stderr); _tPhase = _t }

        let finalAcc = accZ.last ?? Fr.one

        // Step 4: Optional KZG commitments
        var fCommit: PointProjective? = nil
        var tCommit: PointProjective? = nil
        var sCommit: PointProjective? = nil
        var zCommit: PointProjective? = nil

        if let kzg = kzgEngine {
            fCommit = try kzg.commit( witness)
            tCommit = try kzg.commit( table)
            sCommit = try kzg.commit( sorted)
            zCommit = try kzg.commit( accZ)

            if profile { let _t = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [plookup] KZG commitments: %.2f ms\n", (_t - _tPhase) * 1000), stderr); _tPhase = _t }
        }

        if profile {
            let total = (CFAbsoluteTimeGetCurrent() - _t0) * 1000
            fputs(String(format: "  [plookup] TOTAL prove: %.2f ms\n", total), stderr)
        }

        return PlookupProof(
            sortedVector: sorted,
            accumulatorZ: accZ,
            beta: betaVal,
            gamma: gammaVal,
            finalAccumulator: finalAcc,
            fCommitment: fCommit,
            tCommitment: tCommit,
            sCommitment: sCommit,
            zCommitment: zCommit
        )
    }

    // MARK: - Verify

    /// Verify a Plookup proof.
    ///
    /// Checks:
    ///   1. The sorted vector s is a valid merge of f and t.
    ///   2. The grand product accumulator Z closes (Z[n] == 1).
    ///   3. Each transition Z[i] -> Z[i+1] satisfies the Plookup relation.
    ///
    /// - Parameters:
    ///   - proof: The PlookupProof to verify.
    ///   - witness: The original witness vector f.
    ///   - table: The original table vector t.
    ///   - tableCommitment: Optional G1 commitment to the table polynomial (for KZG mode).
    /// - Returns: true if valid.
    public func verify(proof: PlookupProof, witness: [Fr], table: [Fr],
                       tableCommitment: PointProjective? = nil) -> Bool {
        let n = witness.count
        let N = table.count
        let s = proof.sortedVector
        let numSteps = n + N - 1

        // Check 1: sorted vector length = n + N
        guard s.count == n + N else { return false }

        // Check 2: s is a valid sorted merge (contains all of f and t as multisets)
        if !validateSortedMerge(witness: witness, table: table, sorted: s) {
            return false
        }

        // Check 3: accumulator closes (final value == 1)
        guard frEqual(proof.finalAccumulator, Fr.one) else { return false }

        // Check 4: verify each accumulator transition using GW20 h₁/h₂ split
        let accZ = proof.accumulatorZ
        guard accZ.count == numSteps + 1 else { return false }
        guard frEqual(accZ[0], Fr.one) else { return false }

        let beta = proof.beta
        let gamma = proof.gamma
        let onePlusBeta = frAdd(Fr.one, beta)
        let gammaTimesBetaPlusOne = frMul(gamma, onePlusBeta)

        for k in 0..<numSteps {
            var num = Fr.one
            var den = Fr.one

            // Witness term (k < n)
            if k < n {
                num = frMul(onePlusBeta, frAdd(gamma, witness[k]))
                // h₁ term: h₁[i] = s[i]
                den = frAdd(gammaTimesBetaPlusOne,
                            frAdd(s[k], frMul(beta, s[k + 1])))
            }

            // Table transition term (k < N-1)
            if k < N - 1 {
                let tTerm = frAdd(gammaTimesBetaPlusOne,
                                  frAdd(table[k], frMul(beta, table[k + 1])))
                num = frMul(num, tTerm)
                // h₂ term: h₂[j] = s[n+j]
                let h2Term = frAdd(gammaTimesBetaPlusOne,
                                   frAdd(s[n + k], frMul(beta, s[n + k + 1])))
                den = frMul(den, h2Term)
            }

            // Z[k+1] * den == Z[k] * num
            let lhs = frMul(accZ[k + 1], den)
            let rhs = frMul(accZ[k], num)
            if !frEqual(lhs, rhs) { return false }
        }

        return true
    }

    // MARK: - Sorted Merge Construction (GPU-accelerated)

    /// Build the sorted vector s by merging witness f into table t.
    ///
    /// The sorted vector preserves the table ordering: for each table element t[i],
    /// all witness elements equal to t[i] are placed immediately after t[i] in s.
    ///
    /// GPU path: assign sort keys (table position * (N+1) + sub-index), then
    /// radix sort the keys carrying the Fr values as payload.
    private func buildSortedVector(witness: [Fr], table: [Fr]) throws -> [Fr] {
        let n = witness.count
        let N = table.count
        let total = n + N

        // Build a lookup from table value -> table index using sorted binary search
        // (same approach as LookupEngine.computeMultiplicities)
        let tableEntries: [(limbs: [UInt64], idx: Int)] = (0..<N).map { j in
            (frToInt(table[j]), j)
        }
        var sortedEntries = tableEntries
        sortedEntries.sort { a, b in
            for k in stride(from: a.limbs.count - 1, through: 0, by: -1) {
                if a.limbs[k] != b.limbs[k] { return a.limbs[k] < b.limbs[k] }
            }
            return false
        }
        let sortedKeys = sortedEntries.map { $0.limbs }
        let sortedIndices = sortedEntries.map { $0.idx }

        // Assign sort keys: for table element at position j, key = j * (n+1)
        // For witness element matching table[j], key = j * (n+1) + (1 + subindex)
        // This ensures stable ordering: table entry first, then witness copies.

        // Track how many witness elements have been assigned to each table position
        var multCount = [Int](repeating: 0, count: N)

        // Build (sortKey, valueIndex, isTable) triples
        var sortKeys = [UInt64](repeating: 0, count: total)
        var values = [Fr](repeating: Fr.zero, count: total)

        // Table entries
        for j in 0..<N {
            let key = UInt64(j) * UInt64(n + 1)
            sortKeys[j] = key
            values[j] = table[j]
        }

        // Witness entries: binary search for table position
        for i in 0..<n {
            let wLimbs = frToInt(witness[i])
            var lo = 0, hi = N - 1
            var tableIdx = -1
            while lo <= hi {
                let mid = (lo + hi) >> 1
                let cmp = GPUPlookupEngine.compareLimbs(wLimbs, sortedKeys[mid])
                if cmp == 0 {
                    tableIdx = sortedIndices[mid]
                    break
                } else if cmp < 0 {
                    hi = mid - 1
                } else {
                    lo = mid + 1
                }
            }
            precondition(tableIdx >= 0, "Witness element at index \(i) not found in table")

            multCount[tableIdx] += 1
            let key = UInt64(tableIdx) * UInt64(n + 1) + UInt64(multCount[tableIdx])
            sortKeys[N + i] = key
            values[N + i] = witness[i]
        }

        // Sort by key: use GPU radix sort for large arrays, CPU for small
        if total >= gpuThreshold {
            return try gpuSortedMerge(keys: sortKeys, values: values)
        } else {
            return cpuSortedMerge(keys: sortKeys, values: values)
        }
    }

    /// GPU-accelerated sorted merge using RadixSortEngine (key-value sort).
    /// Since RadixSortEngine works on UInt32 keys, we do a two-pass sort
    /// (low 32 bits, then high 32 bits) or use CPU sort for keys > 32 bits.
    private func gpuSortedMerge(keys: [UInt64], values: [Fr]) throws -> [Fr] {
        let n = keys.count

        // Check if all keys fit in 32 bits (common case for tables < ~65K)
        let maxKey = keys.max() ?? 0
        if maxKey <= UInt64(UInt32.max) {
            let keys32 = keys.map { UInt32($0) }
            let indices32 = (0..<UInt32(n)).map { $0 }
            let sortEngine = try RadixSortEngine()
            let (_, sortedIndices) = try sortEngine.sortKV(keys: keys32, values: indices32)
            return sortedIndices.map { values[Int($0)] }
        }

        // Fallback: CPU sort for large key ranges
        return cpuSortedMerge(keys: keys, values: values)
    }

    /// CPU sorted merge: sort by key, return values in sorted order.
    private func cpuSortedMerge(keys: [UInt64], values: [Fr]) -> [Fr] {
        let indexed = keys.enumerated().sorted { $0.element < $1.element }
        return indexed.map { values[$0.offset] }
    }

    private static func compareLimbs(_ a: [UInt64], _ b: [UInt64]) -> Int {
        for k in stride(from: a.count - 1, through: 0, by: -1) {
            if a[k] < b[k] { return -1 }
            if a[k] > b[k] { return 1 }
        }
        return 0
    }

    // MARK: - Grand Product Accumulator (GPU-accelerated)

    /// Build the Plookup grand product accumulator using the GW20 identity.
    ///
    /// Sorted vector s (length n+N) is split into:
    ///   h₁ = s[0..n]   (n+1 elements)
    ///   h₂ = s[n..n+N-1] (N elements, overlapping with h₁ at s[n])
    ///
    /// GW20 identity (n witness terms + N-1 table transition terms = n + N - 1 total):
    ///   LHS: ∏_{i<n} (1+β)(γ+f[i])  ·  ∏_{j<N-1} (γ(1+β)+t[j]+β·t[j+1])
    ///   RHS: ∏_{i<n} (γ(1+β)+h₁[i]+β·h₁[i+1])  ·  ∏_{j<N-1} (γ(1+β)+h₂[j]+β·h₂[j+1])
    ///
    /// Accumulator: Z[0]=1, Z[k+1] = Z[k] · num[k] / den[k], Z[n+N-1] should equal 1.
    private func buildAccumulator(witness: [Fr], table: [Fr],
                                   sorted: [Fr], beta: Fr, gamma: Fr) throws -> [Fr] {
        let n = witness.count
        let N = table.count
        let numSteps = n + N - 1
        let onePlusBeta = frAdd(Fr.one, beta)
        let gammaTimesBetaPlusOne = frMul(gamma, onePlusBeta)

        var numerators = [Fr](repeating: Fr.one, count: numSteps)
        var denominators = [Fr](repeating: Fr.one, count: numSteps)

        for k in 0..<numSteps {
            var num = Fr.one
            var den = Fr.one

            // Witness term (k < n): (1+β)(γ+f[k])
            if k < n {
                num = frMul(onePlusBeta, frAdd(gamma, witness[k]))
                // h₁ term: (γ(1+β) + h₁[k] + β·h₁[k+1]) where h₁[i] = sorted[i]
                den = frAdd(gammaTimesBetaPlusOne,
                            frAdd(sorted[k], frMul(beta, sorted[k + 1])))
            }

            // Table transition term (k < N-1): (γ(1+β) + t[k] + β·t[k+1])
            if k < N - 1 {
                let tTerm = frAdd(gammaTimesBetaPlusOne,
                                  frAdd(table[k], frMul(beta, table[k + 1])))
                num = frMul(num, tTerm)
                // h₂ term: (γ(1+β) + h₂[k] + β·h₂[k+1]) where h₂[j] = sorted[n+j]
                let h2Term = frAdd(gammaTimesBetaPlusOne,
                                   frAdd(sorted[n + k], frMul(beta, sorted[n + k + 1])))
                den = frMul(den, h2Term)
            }

            numerators[k] = num
            denominators[k] = den
        }

        // Build accumulator: Z[0] = 1, Z[k+1] = Z[k] * num[k] / den[k]
        var accZ = [Fr](repeating: Fr.zero, count: numSteps + 1)
        accZ[0] = Fr.one
        for k in 0..<numSteps {
            let ratio = frMul(numerators[k], frInverse(denominators[k]))
            accZ[k + 1] = frMul(accZ[k], ratio)
        }

        return accZ
    }

    // MARK: - Sorted Merge Validation

    /// Validate that sorted vector s is a valid merge of witness f and table t.
    /// Checks multiset equality: the multiset of s equals the multiset union of f and t.
    private func validateSortedMerge(witness: [Fr], table: [Fr], sorted: [Fr]) -> Bool {
        let n = witness.count
        let N = table.count
        guard sorted.count == n + N else { return false }

        // Build frequency maps and compare
        var freqF = [FrKey: Int]()
        for v in witness {
            freqF[FrKey(v), default: 0] += 1
        }
        var freqT = [FrKey: Int]()
        for v in table {
            freqT[FrKey(v), default: 0] += 1
        }
        var freqS = [FrKey: Int]()
        for v in sorted {
            freqS[FrKey(v), default: 0] += 1
        }

        // Combined frequency of f + t should equal s
        var combined = freqT
        for (k, c) in freqF {
            combined[k, default: 0] += c
        }

        return combined == freqS
    }

    // MARK: - Fiat-Shamir Challenge Derivation

    private func deriveBeta(witness: [Fr], table: [Fr]) -> Fr {
        var transcript = [UInt8]()
        // Domain separator
        transcript.append(contentsOf: [0x50, 0x4C, 0x4B, 0x50]) // "PLKP"
        appendSizeAndSamples(&transcript, witness)
        appendSizeAndSamples(&transcript, table)
        return hashToFr(transcript)
    }

    private func deriveGamma(witness: [Fr], table: [Fr], beta: Fr) -> Fr {
        var transcript = [UInt8]()
        transcript.append(contentsOf: [0x50, 0x4C, 0x4B, 0x47]) // "PLKG"
        appendSizeAndSamples(&transcript, witness)
        appendSizeAndSamples(&transcript, table)
        appendFrBytes(&transcript, beta)
        return hashToFr(transcript)
    }

    private func appendSizeAndSamples(_ transcript: inout [UInt8], _ arr: [Fr]) {
        var size = UInt64(arr.count)
        for _ in 0..<8 {
            transcript.append(UInt8(size & 0xFF))
            size >>= 8
        }
        let sampleCount = min(8, arr.count)
        for i in 0..<sampleCount {
            appendFrBytes(&transcript, arr[i])
        }
    }

    private func appendFrBytes(_ transcript: inout [UInt8], _ v: Fr) {
        let limbs = frToInt(v)
        for limb in limbs {
            for byte in 0..<8 {
                transcript.append(UInt8((limb >> (byte * 8)) & 0xFF))
            }
        }
    }

    private func hashToFr(_ data: [UInt8]) -> Fr {
        let hash = blake3(data)
        var limbs = [UInt64](repeating: 0, count: 4)
        for i in 0..<4 {
            for j in 0..<8 {
                limbs[i] |= UInt64(hash[i * 8 + j]) << (j * 8)
            }
        }
        let raw = Fr.from64(limbs)
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))
    }
}
