// GPULogUpEngine — GPU-accelerated LogUp lookup argument
//
// Protocol (Haboeck 2022, "LogUp"):
//   Given table T[0..N-1] and witness f[0..m-1] where each f[i] in T:
//
//   1. Multiplicities: compute mult[j] = #{i : f[i] = T[j]}
//   2. Verifier sends random challenge beta
//   3. Fractional sumcheck: prove
//        sum_i 1/(beta + f[i]) = sum_j mult[j]/(beta + T[j])
//   4. Both sides are verified via a running sum (grand sum) accumulator
//      that must close to zero.
//
// Multi-table extension:
//   For k tables T_0..T_{k-1} and witness columns f_0..f_{k-1} with
//   table selectors sel[i] in {0..k-1}:
//     sum_i 1/(beta + f_{sel[i]}[i]) = sum over all tables of
//       sum_j mult_t[j]/(beta + T_t[j])
//
// GPU acceleration targets:
//   - Denominator computation: GPU batch add (beta + f[i]) for all i
//   - Batch inverse: GPU Montgomery batch inverse via GPUBatchInverseEngine
//   - Grand sum accumulator: prefix sum of inverses
//   - Optional KZG commitments via GPU MSM
//
// References:
//   LogUp: Haboeck 2022 (eprint 2022/1530)
//   LogUp-GKR: Papini-Haboeck-Starkware 2023

import Foundation
import Metal

// MARK: - Proof Structure

/// Proof produced by the LogUp protocol.
public struct LogUpProof {
    /// Multiplicities: mult[j] = number of times T[j] appears in witness
    public let multiplicities: [Fr]
    /// The random challenge beta
    public let beta: Fr
    /// Lookup side inverses: h_f[i] = 1/(beta + f[i])
    public let lookupInverses: [Fr]
    /// Table side terms: h_t[j] = mult[j]/(beta + T[j])
    public let tableTerms: [Fr]
    /// Grand sum accumulator (running sum from both sides, should close to zero)
    public let grandSumAccumulator: [Fr]
    /// Final grand sum value (should be zero for valid proof)
    public let finalGrandSum: Fr
    /// Lookup sum: S_f = sum_i 1/(beta + f[i])
    public let lookupSum: Fr
    /// Table sum: S_t = sum_j mult[j]/(beta + T[j])
    public let tableSum: Fr

    public init(multiplicities: [Fr], beta: Fr,
                lookupInverses: [Fr], tableTerms: [Fr],
                grandSumAccumulator: [Fr], finalGrandSum: Fr,
                lookupSum: Fr, tableSum: Fr) {
        self.multiplicities = multiplicities
        self.beta = beta
        self.lookupInverses = lookupInverses
        self.tableTerms = tableTerms
        self.grandSumAccumulator = grandSumAccumulator
        self.finalGrandSum = finalGrandSum
        self.lookupSum = lookupSum
        self.tableSum = tableSum
    }
}

/// Multi-table LogUp proof.
public struct MultiTableLogUpProof {
    /// Per-table multiplicities
    public let multiplicities: [[Fr]]
    /// The random challenge beta
    public let beta: Fr
    /// Lookup side inverses: h_f[i] = 1/(beta + f[i])
    public let lookupInverses: [Fr]
    /// Per-table terms: h_t[t][j] = mult_t[j]/(beta + T_t[j])
    public let tableTerms: [[Fr]]
    /// Final grand sum (should be zero)
    public let finalGrandSum: Fr
    /// Lookup sum
    public let lookupSum: Fr
    /// Table sum
    public let tableSum: Fr

    public init(multiplicities: [[Fr]], beta: Fr,
                lookupInverses: [Fr], tableTerms: [[Fr]],
                finalGrandSum: Fr, lookupSum: Fr, tableSum: Fr) {
        self.multiplicities = multiplicities
        self.beta = beta
        self.lookupInverses = lookupInverses
        self.tableTerms = tableTerms
        self.finalGrandSum = finalGrandSum
        self.lookupSum = lookupSum
        self.tableSum = tableSum
    }
}

// MARK: - Engine

/// GPU-accelerated LogUp lookup argument engine.
///
/// Proves that every element in witness vector f is contained in table vector T,
/// using the fractional sumcheck (logarithmic derivative) approach.
///
/// GPU acceleration:
///   - Batch inverse via GPUBatchInverseEngine (Montgomery's trick on GPU)
///   - Grand sum accumulator via prefix sum
///   - Multi-table support with table selectors
public class GPULogUpEngine {
    public static let version = Versions.gpuLogUp

    private let inverseEngine: GPUBatchInverseEngine

    /// Emit profiling info to stderr
    public var profile = false

    /// GPU threshold: arrays smaller than this use CPU path
    public var gpuThreshold: Int = 256

    public init() throws {
        self.inverseEngine = try GPUBatchInverseEngine()
    }

    // MARK: - Single Table Prove

    /// Full LogUp proof: proves that every element in `witness` exists in `table`.
    ///
    /// - Parameters:
    ///   - witness: The lookup vector f. All elements must be in table.
    ///   - table: The table vector T. Must contain all values referenced by witness.
    ///   - beta: Random challenge (normally from Fiat-Shamir). Pass nil for auto-derivation.
    /// - Returns: A LogUpProof.
    public func prove(witness: [Fr], table: [Fr],
                      beta: Fr? = nil) throws -> LogUpProof {
        let m = witness.count
        let N = table.count
        precondition(m > 0, "Witness must be non-empty")
        precondition(N > 0, "Table must be non-empty")

        let _t0 = profile ? CFAbsoluteTimeGetCurrent() : 0
        var _tPhase = profile ? CFAbsoluteTimeGetCurrent() : 0

        // Step 1: Derive or use provided challenge
        let betaVal = beta ?? deriveBeta(witness: witness, table: table)

        if profile { let _t = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [logup] challenge derivation: %.2f ms\n", (_t - _tPhase) * 1000), stderr); _tPhase = _t }

        // Step 2: Compute multiplicities
        let mult = computeMultiplicities(witness: witness, table: table)

        if profile { let _t = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [logup] multiplicities: %.2f ms\n", (_t - _tPhase) * 1000), stderr); _tPhase = _t }

        // Step 3: Compute denominators beta + f[i] and beta + T[j]
        var lookupDenoms = [Fr](repeating: Fr.zero, count: m)
        for i in 0..<m {
            lookupDenoms[i] = frAdd(betaVal, witness[i])
        }

        var tableDenoms = [Fr](repeating: Fr.zero, count: N)
        for j in 0..<N {
            tableDenoms[j] = frAdd(betaVal, table[j])
        }

        if profile { let _t = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [logup] denominators: %.2f ms\n", (_t - _tPhase) * 1000), stderr); _tPhase = _t }

        // Step 4: GPU batch inverse of all denominators
        let lookupInvs = try inverseEngine.batchInverseFr(lookupDenoms)
        let tableInvs = try inverseEngine.batchInverseFr(tableDenoms)

        if profile { let _t = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [logup] batch inverse (m=%d, N=%d): %.2f ms\n", m, N, (_t - _tPhase) * 1000), stderr); _tPhase = _t }

        // Step 5: Compute table terms: h_t[j] = mult[j] * tableInvs[j]
        var tableTerms = [Fr](repeating: Fr.zero, count: N)
        for j in 0..<N {
            tableTerms[j] = frMul(mult[j], tableInvs[j])
        }

        // Step 6: Compute sums
        var lookupSum = Fr.zero
        for i in 0..<m {
            lookupSum = frAdd(lookupSum, lookupInvs[i])
        }

        var tableSum = Fr.zero
        for j in 0..<N {
            tableSum = frAdd(tableSum, tableTerms[j])
        }

        if profile { let _t = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [logup] sums: %.2f ms\n", (_t - _tPhase) * 1000), stderr); _tPhase = _t }

        // Step 7: Build grand sum accumulator
        // Interleave lookup and table contributions: acc goes through
        // +1/(beta+f[i]) for lookup side, -mult[j]/(beta+T[j]) for table side.
        // If valid, the total should be zero.
        let totalLen = m + N
        var acc = [Fr](repeating: Fr.zero, count: totalLen + 1)
        acc[0] = Fr.zero
        // Add lookup contributions first
        for i in 0..<m {
            acc[i + 1] = frAdd(acc[i], lookupInvs[i])
        }
        // Subtract table contributions
        for j in 0..<N {
            acc[m + j + 1] = frSub(acc[m + j], tableTerms[j])
        }

        let finalSum = acc[totalLen]

        if profile {
            let total = (CFAbsoluteTimeGetCurrent() - _t0) * 1000
            fputs(String(format: "  [logup] TOTAL prove: %.2f ms\n", total), stderr)
        }

        return LogUpProof(
            multiplicities: mult,
            beta: betaVal,
            lookupInverses: lookupInvs,
            tableTerms: tableTerms,
            grandSumAccumulator: acc,
            finalGrandSum: finalSum,
            lookupSum: lookupSum,
            tableSum: tableSum
        )
    }

    // MARK: - Single Table Verify

    /// Verify a LogUp proof.
    ///
    /// Checks:
    ///   1. Multiplicities are consistent with witness and table.
    ///   2. Each lookup inverse h_f[i] = 1/(beta + f[i]).
    ///   3. Each table term h_t[j] = mult[j]/(beta + T[j]).
    ///   4. Lookup sum equals table sum (grand sum closes to zero).
    ///
    /// - Parameters:
    ///   - proof: The LogUpProof to verify.
    ///   - witness: The original witness vector f.
    ///   - table: The original table vector T.
    /// - Returns: true if valid.
    public func verify(proof: LogUpProof, witness: [Fr], table: [Fr]) -> Bool {
        let m = witness.count
        let N = table.count
        let beta = proof.beta

        // Check 1: final grand sum is zero
        guard frEqual(proof.finalGrandSum, Fr.zero) else { return false }

        // Check 2: lookup sum == table sum
        guard frEqual(proof.lookupSum, proof.tableSum) else { return false }

        // Check 3: verify each lookup inverse
        guard proof.lookupInverses.count == m else { return false }
        for i in 0..<m {
            let denom = frAdd(beta, witness[i])
            let product = frMul(proof.lookupInverses[i], denom)
            if !frEqual(product, Fr.one) { return false }
        }

        // Check 4: verify each table term
        guard proof.tableTerms.count == N else { return false }
        guard proof.multiplicities.count == N else { return false }
        for j in 0..<N {
            let denom = frAdd(beta, table[j])
            let expected = frMul(proof.multiplicities[j], proof.tableTerms[j])
            // h_t[j] * denom should equal mult[j]
            // equivalently: h_t[j] = mult[j] / denom
            // so h_t[j] * denom = mult[j]
            let product = frMul(proof.tableTerms[j], denom)
            if !frEqual(product, proof.multiplicities[j]) { return false }
        }

        // Check 5: multiplicities are consistent
        let computedMult = computeMultiplicities(witness: witness, table: table)
        for j in 0..<N {
            if !frEqual(computedMult[j], proof.multiplicities[j]) { return false }
        }

        return true
    }

    // MARK: - Multi-Table Prove

    /// Multi-table LogUp proof: proves lookups across multiple tables.
    ///
    /// - Parameters:
    ///   - witness: The lookup values f[0..m-1].
    ///   - tableSelectors: For each witness element, which table it looks up in (0-indexed).
    ///   - tables: Array of tables T_0..T_{k-1}.
    ///   - beta: Random challenge. Pass nil for auto-derivation.
    /// - Returns: A MultiTableLogUpProof.
    public func proveMultiTable(witness: [Fr], tableSelectors: [Int],
                                tables: [[Fr]],
                                beta: Fr? = nil) throws -> MultiTableLogUpProof {
        let m = witness.count
        let k = tables.count
        precondition(m > 0, "Witness must be non-empty")
        precondition(k > 0, "Must have at least one table")
        precondition(tableSelectors.count == m, "Selectors must match witness length")

        // Derive challenge
        let betaVal = beta ?? deriveMultiTableBeta(witness: witness, tables: tables)

        // Compute per-table multiplicities
        var perTableMult = [[Fr]]()
        for t in 0..<k {
            var mult = [Fr](repeating: Fr.zero, count: tables[t].count)
            perTableMult.append(mult)
        }
        // Build table lookup indices
        for i in 0..<m {
            let tIdx = tableSelectors[i]
            precondition(tIdx >= 0 && tIdx < k, "Invalid table selector \(tIdx)")
            let tab = tables[tIdx]
            // Find witness[i] in tables[tIdx]
            var found = false
            for j in 0..<tab.count {
                if frEqual(witness[i], tab[j]) {
                    perTableMult[tIdx][j] = frAdd(perTableMult[tIdx][j], Fr.one)
                    found = true
                    break
                }
            }
            precondition(found, "Witness element at index \(i) not found in table \(tIdx)")
        }

        // Compute lookup inverses: 1/(beta + f[i])
        var lookupDenoms = [Fr](repeating: Fr.zero, count: m)
        for i in 0..<m {
            lookupDenoms[i] = frAdd(betaVal, witness[i])
        }
        let lookupInvs = try inverseEngine.batchInverseFr(lookupDenoms)

        // Compute per-table terms
        var allTableTerms = [[Fr]]()
        var tableSum = Fr.zero
        for t in 0..<k {
            let tab = tables[t]
            let N = tab.count
            var denoms = [Fr](repeating: Fr.zero, count: N)
            for j in 0..<N {
                denoms[j] = frAdd(betaVal, tab[j])
            }
            let invs = try inverseEngine.batchInverseFr(denoms)
            var terms = [Fr](repeating: Fr.zero, count: N)
            for j in 0..<N {
                terms[j] = frMul(perTableMult[t][j], invs[j])
                tableSum = frAdd(tableSum, terms[j])
            }
            allTableTerms.append(terms)
        }

        // Compute lookup sum
        var lookupSum = Fr.zero
        for i in 0..<m {
            lookupSum = frAdd(lookupSum, lookupInvs[i])
        }

        let finalGrandSum = frSub(lookupSum, tableSum)

        return MultiTableLogUpProof(
            multiplicities: perTableMult,
            beta: betaVal,
            lookupInverses: lookupInvs,
            tableTerms: allTableTerms,
            finalGrandSum: finalGrandSum,
            lookupSum: lookupSum,
            tableSum: tableSum
        )
    }

    // MARK: - Multi-Table Verify

    /// Verify a multi-table LogUp proof.
    public func verifyMultiTable(proof: MultiTableLogUpProof, witness: [Fr],
                                 tableSelectors: [Int], tables: [[Fr]]) -> Bool {
        let m = witness.count
        let k = tables.count
        let beta = proof.beta

        // Check 1: grand sum is zero
        guard frEqual(proof.finalGrandSum, Fr.zero) else { return false }

        // Check 2: lookup sum == table sum
        guard frEqual(proof.lookupSum, proof.tableSum) else { return false }

        // Check 3: verify lookup inverses
        guard proof.lookupInverses.count == m else { return false }
        for i in 0..<m {
            let denom = frAdd(beta, witness[i])
            let product = frMul(proof.lookupInverses[i], denom)
            if !frEqual(product, Fr.one) { return false }
        }

        // Check 4: verify per-table terms and multiplicities
        guard proof.multiplicities.count == k else { return false }
        guard proof.tableTerms.count == k else { return false }
        for t in 0..<k {
            let tab = tables[t]
            let N = tab.count
            guard proof.multiplicities[t].count == N else { return false }
            guard proof.tableTerms[t].count == N else { return false }
            for j in 0..<N {
                let denom = frAdd(beta, tab[j])
                let product = frMul(proof.tableTerms[t][j], denom)
                if !frEqual(product, proof.multiplicities[t][j]) { return false }
            }
        }

        // Check 5: recompute sums
        var lookupSum = Fr.zero
        for i in 0..<m {
            lookupSum = frAdd(lookupSum, proof.lookupInverses[i])
        }
        guard frEqual(lookupSum, proof.lookupSum) else { return false }

        var tableSum = Fr.zero
        for t in 0..<k {
            for j in 0..<tables[t].count {
                tableSum = frAdd(tableSum, proof.tableTerms[t][j])
            }
        }
        guard frEqual(tableSum, proof.tableSum) else { return false }

        return true
    }

    // MARK: - Multiplicities

    /// Compute multiplicities: mult[j] = number of times table[j] appears in witness.
    func computeMultiplicities(witness: [Fr], table: [Fr]) -> [Fr] {
        let N = table.count
        var freq = [FrKey: Int]()
        for w in witness {
            freq[FrKey(w), default: 0] += 1
        }

        var mult = [Fr](repeating: Fr.zero, count: N)
        for j in 0..<N {
            let key = FrKey(table[j])
            if let count = freq[key] {
                mult[j] = frFromInt(UInt64(count))
            }
        }
        return mult
    }

    // MARK: - Fiat-Shamir Challenge Derivation

    private func deriveBeta(witness: [Fr], table: [Fr]) -> Fr {
        var transcript = [UInt8]()
        // Domain separator
        transcript.append(contentsOf: [0x4C, 0x47, 0x55, 0x50]) // "LGUP"
        appendSizeAndSamples(&transcript, witness)
        appendSizeAndSamples(&transcript, table)
        return hashToFr(transcript)
    }

    private func deriveMultiTableBeta(witness: [Fr], tables: [[Fr]]) -> Fr {
        var transcript = [UInt8]()
        transcript.append(contentsOf: [0x4C, 0x47, 0x4D, 0x54]) // "LGMT"
        appendSizeAndSamples(&transcript, witness)
        for t in tables {
            appendSizeAndSamples(&transcript, t)
        }
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
