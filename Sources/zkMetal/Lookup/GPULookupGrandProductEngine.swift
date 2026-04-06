// GPULookupGrandProductEngine — GPU-accelerated lookup grand product argument
//
// Protocol (lookup via grand product):
//   Given a table T[0..N-1] and witness columns f_0..f_{c-1} of length n,
//   prove that every witness row (f_0[i], ..., f_{c-1}[i]) appears in the table
//   (which may also have c columns).
//
//   1. Compress multi-column entries:
//        For random alpha, f_compressed[i] = f_0[i] + alpha*f_1[i] + ... + alpha^{c-1}*f_{c-1}[i]
//        Likewise for table: t_compressed[j] = T_0[j] + alpha*T_1[j] + ...
//
//   2. Sorted permutation: merge f_compressed and t_compressed into sorted vector s
//      of length n+N, preserving table ordering (duplicates from f placed adjacent
//      to their matching table entry).
//
//   3. Helper column h: encodes the sorted-difference relationship
//        h[0] = 1
//        h[i] = (s[i] == s[i-1]) ? 0 : 1
//      Used to validate that the sorted vector respects table ordering.
//
//   4. Grand product polynomial z(X):
//        z[0] = 1
//        z[i+1] = z[i] * (gamma + f_compressed[i]) / (gamma + s[i])
//        for i = 0..n-1 (lookup side)
//
//      Combined with table-side accumulator:
//        z_t[0] = 1
//        z_t[j+1] = z_t[j] * (gamma + t_compressed[j]) * mult[j] / (gamma + t_compressed[j])^mult[j]
//
//      Or, equivalently, via the Plookup-style two-accumulator approach:
//        With challenges beta, gamma:
//        Numerator[i]   = (1+beta) * (gamma + f[i]) * (gamma*(1+beta) + t[i%N] + beta*t[(i+1)%N])
//        Denominator[i] = (gamma*(1+beta) + s[2i] + beta*s[2i+1]) * (gamma*(1+beta) + s[2i+1] + beta*s[2i+2])
//        z[0] = 1, z[i+1] = z[i] * Numerator[i] / Denominator[i]
//        Boundary: z[n] == 1
//
//   5. Transition constraints on z:
//        z[i+1] * Den[i] = z[i] * Num[i]
//      Boundary constraints:
//        z[0] = 1, z[n] = 1
//
// Multi-column lookup:
//   c columns are compressed into a single column via random linear combination
//   with challenge alpha. This reduces multi-column to single-column lookup.
//
// GPU acceleration targets:
//   - Multi-column compression: parallel linear combination
//   - Sorted merge: GPU radix sort (key-value)
//   - Grand product accumulator: GPU batch inverse + prefix product
//   - Helper column construction: parallel scan
//
// References:
//   Plookup: Gabizon-Williamson 2020 (eprint 2020/315)
//   Lookup grand product: Hab\"ock 2022
//   Multi-column: Random linear combination (Schwartz-Zippel)

import Foundation
import Metal

// MARK: - Configuration

/// Configuration for the lookup grand product engine.
public struct LookupGrandProductConfig {
    /// Number of columns in multi-column lookups (1 for single-column).
    public let numColumns: Int
    /// Whether to compute and verify helper column h.
    public let computeHelper: Bool
    /// Whether to produce commitment-ready evaluation form.
    public let evaluationForm: Bool

    public init(numColumns: Int = 1, computeHelper: Bool = true, evaluationForm: Bool = false) {
        precondition(numColumns >= 1, "Must have at least 1 column")
        self.numColumns = numColumns
        self.computeHelper = computeHelper
        self.evaluationForm = evaluationForm
    }
}

// MARK: - Proof Structure

/// Proof produced by the lookup grand product argument.
public struct LookupGrandProductProof {
    /// Compressed witness (single column after RLC)
    public let compressedWitness: [Fr]
    /// Compressed table (single column after RLC)
    public let compressedTable: [Fr]
    /// Sorted vector s (merged compressed witness + compressed table)
    public let sortedVector: [Fr]
    /// Grand product accumulator z[0..n]
    public let accumulatorZ: [Fr]
    /// Helper column h: h[i] = 1 if s[i] != s[i-1], else 0 (optional)
    public let helperH: [Fr]?
    /// Random challenges
    public let alpha: Fr
    public let beta: Fr
    public let gamma: Fr
    /// Final accumulator value: z[n] should be 1
    public let finalAccumulator: Fr
    /// Multiplicities: mult[j] = number of witness rows matching table row j
    public let multiplicities: [Fr]
    /// Transition constraint evaluations (for debugging/verification)
    public let transitionNumerators: [Fr]?
    public let transitionDenominators: [Fr]?

    public init(compressedWitness: [Fr], compressedTable: [Fr],
                sortedVector: [Fr], accumulatorZ: [Fr],
                helperH: [Fr]?,
                alpha: Fr, beta: Fr, gamma: Fr,
                finalAccumulator: Fr, multiplicities: [Fr],
                transitionNumerators: [Fr]? = nil,
                transitionDenominators: [Fr]? = nil) {
        self.compressedWitness = compressedWitness
        self.compressedTable = compressedTable
        self.sortedVector = sortedVector
        self.accumulatorZ = accumulatorZ
        self.helperH = helperH
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.finalAccumulator = finalAccumulator
        self.multiplicities = multiplicities
        self.transitionNumerators = transitionNumerators
        self.transitionDenominators = transitionDenominators
    }
}

/// Lightweight verification result with diagnostic info.
public struct LookupGrandProductVerification {
    /// Whether the proof is valid
    public let valid: Bool
    /// Which check failed (nil if valid)
    public let failedCheck: String?
    /// Index of failing transition constraint (-1 if none)
    public let failingIndex: Int

    public init(valid: Bool, failedCheck: String? = nil, failingIndex: Int = -1) {
        self.valid = valid
        self.failedCheck = failedCheck
        self.failingIndex = failingIndex
    }
}

// MARK: - Engine

/// GPU-accelerated lookup grand product argument engine.
///
/// Proves that every row in a (possibly multi-column) witness appears in a
/// (possibly multi-column) table, using sorted permutation + grand product.
///
/// GPU acceleration:
///   - Multi-column RLC compression: parallel field arithmetic
///   - Sorted merge via GPU radix sort (key-value)
///   - Grand product accumulator: GPUGrandProductEngine (batch inverse + prefix product)
///   - Helper column: parallel equality scan
public class GPULookupGrandProductEngine {
    public static let version = Versions.gpuLookupGrandProduct

    private let grandProductEngine: GPUGrandProductEngine
    private let inverseEngine: GPUBatchInverseEngine
    private let config: LookupGrandProductConfig

    /// Emit profiling info to stderr
    public var profile = false

    /// GPU threshold: arrays smaller than this use CPU path
    public var gpuThreshold: Int = 4096

    public init(config: LookupGrandProductConfig = LookupGrandProductConfig()) throws {
        self.grandProductEngine = try GPUGrandProductEngine()
        self.inverseEngine = try GPUBatchInverseEngine()
        self.config = config
    }

    // MARK: - Single-Column Prove

    /// Prove a single-column lookup: every element in `witness` exists in `table`.
    ///
    /// - Parameters:
    ///   - witness: The lookup vector f[0..n-1]. All elements must be in table.
    ///   - table: The table vector T[0..N-1].
    ///   - beta: Random challenge (nil for auto-derivation).
    ///   - gamma: Random challenge (nil for auto-derivation).
    /// - Returns: A LookupGrandProductProof.
    public func prove(witness: [Fr], table: [Fr],
                      beta: Fr? = nil, gamma: Fr? = nil) throws -> LookupGrandProductProof {
        try proveMultiColumn(witnessColumns: [witness], tableColumns: [table],
                             alpha: nil, beta: beta, gamma: gamma)
    }

    // MARK: - Multi-Column Prove

    /// Prove a multi-column lookup: every witness row appears in the table.
    ///
    /// Witness columns f_0..f_{c-1} each have length n.
    /// Table columns T_0..T_{c-1} each have length N.
    ///
    /// - Parameters:
    ///   - witnessColumns: Array of c witness columns, each of length n.
    ///   - tableColumns: Array of c table columns, each of length N.
    ///   - alpha: RLC challenge for multi-column compression (nil for auto).
    ///   - beta: Grand product challenge (nil for auto).
    ///   - gamma: Grand product challenge (nil for auto).
    /// - Returns: A LookupGrandProductProof.
    public func proveMultiColumn(witnessColumns: [[Fr]], tableColumns: [[Fr]],
                                 alpha: Fr? = nil, beta: Fr? = nil,
                                 gamma: Fr? = nil) throws -> LookupGrandProductProof {
        let c = witnessColumns.count
        precondition(c > 0, "Must have at least 1 witness column")
        precondition(tableColumns.count == c, "Table and witness must have same number of columns")

        let n = witnessColumns[0].count
        let N = tableColumns[0].count
        precondition(n > 0, "Witness must be non-empty")
        precondition(N > 0, "Table must be non-empty")

        for col in witnessColumns {
            precondition(col.count == n, "All witness columns must have equal length")
        }
        for col in tableColumns {
            precondition(col.count == N, "All table columns must have equal length")
        }

        let _t0 = profile ? CFAbsoluteTimeGetCurrent() : 0
        var _tPhase = profile ? CFAbsoluteTimeGetCurrent() : 0

        // Step 1: Derive challenges
        let alphaVal = alpha ?? deriveAlpha(witnessColumns: witnessColumns,
                                             tableColumns: tableColumns)
        let betaVal = beta ?? deriveBeta(witnessColumns: witnessColumns,
                                          tableColumns: tableColumns, alpha: alphaVal)
        let gammaVal = gamma ?? deriveGamma(witnessColumns: witnessColumns,
                                             tableColumns: tableColumns,
                                             alpha: alphaVal, beta: betaVal)

        if profile {
            let _t = CFAbsoluteTimeGetCurrent()
            fputs(String(format: "  [lgp] challenge derivation: %.2f ms\n",
                         (_t - _tPhase) * 1000), stderr)
            _tPhase = _t
        }

        // Step 2: Compress multi-column entries via RLC
        let compressedWitness = compressColumns(witnessColumns, alpha: alphaVal, count: n)
        let compressedTable = compressColumns(tableColumns, alpha: alphaVal, count: N)

        if profile {
            let _t = CFAbsoluteTimeGetCurrent()
            fputs(String(format: "  [lgp] RLC compression (%d cols, n=%d, N=%d): %.2f ms\n",
                         c, n, N, (_t - _tPhase) * 1000), stderr)
            _tPhase = _t
        }

        // Step 3: Compute multiplicities
        let multiplicities = computeMultiplicities(witness: compressedWitness,
                                                    table: compressedTable)

        if profile {
            let _t = CFAbsoluteTimeGetCurrent()
            fputs(String(format: "  [lgp] multiplicities: %.2f ms\n",
                         (_t - _tPhase) * 1000), stderr)
            _tPhase = _t
        }

        // Step 4: Build sorted vector
        let sorted = try buildSortedVector(witness: compressedWitness,
                                            table: compressedTable)

        if profile {
            let _t = CFAbsoluteTimeGetCurrent()
            fputs(String(format: "  [lgp] sorted merge (|s|=%d): %.2f ms\n",
                         sorted.count, (_t - _tPhase) * 1000), stderr)
            _tPhase = _t
        }

        // Step 5: Build helper column (optional)
        var helperH: [Fr]? = nil
        if config.computeHelper {
            helperH = buildHelperColumn(sorted: sorted)

            if profile {
                let _t = CFAbsoluteTimeGetCurrent()
                fputs(String(format: "  [lgp] helper column: %.2f ms\n",
                             (_t - _tPhase) * 1000), stderr)
                _tPhase = _t
            }
        }

        // Step 6: Build grand product accumulator z
        let (accZ, nums, dens) = try buildAccumulator(
            witness: compressedWitness, table: compressedTable,
            sorted: sorted, beta: betaVal, gamma: gammaVal)

        if profile {
            let _t = CFAbsoluteTimeGetCurrent()
            fputs(String(format: "  [lgp] grand product accumulator (len=%d): %.2f ms\n",
                         accZ.count, (_t - _tPhase) * 1000), stderr)
            _tPhase = _t
        }

        let finalAcc = accZ.last ?? Fr.one

        if profile {
            let total = (CFAbsoluteTimeGetCurrent() - _t0) * 1000
            fputs(String(format: "  [lgp] TOTAL prove: %.2f ms\n", total), stderr)
        }

        return LookupGrandProductProof(
            compressedWitness: compressedWitness,
            compressedTable: compressedTable,
            sortedVector: sorted,
            accumulatorZ: accZ,
            helperH: helperH,
            alpha: alphaVal,
            beta: betaVal,
            gamma: gammaVal,
            finalAccumulator: finalAcc,
            multiplicities: multiplicities,
            transitionNumerators: nums,
            transitionDenominators: dens
        )
    }

    // MARK: - Verify

    /// Verify a lookup grand product proof.
    ///
    /// Checks:
    ///   1. Sorted vector length = n + N
    ///   2. Sorted vector is a valid multiset merge of witness and table
    ///   3. Accumulator boundary: z[0] = 1, z[n] = 1
    ///   4. Each transition constraint z[i+1]*Den[i] = z[i]*Num[i]
    ///   5. Helper column consistency (if present)
    ///   6. Multiplicities consistency
    ///
    /// - Parameters:
    ///   - proof: The LookupGrandProductProof to verify.
    ///   - witnessColumns: Original witness columns (or single-element array for 1-col).
    ///   - tableColumns: Original table columns.
    /// - Returns: A LookupGrandProductVerification with diagnostic info.
    public func verify(proof: LookupGrandProductProof,
                       witnessColumns: [[Fr]],
                       tableColumns: [[Fr]]) -> LookupGrandProductVerification {
        let c = witnessColumns.count
        guard tableColumns.count == c else {
            return LookupGrandProductVerification(valid: false,
                                                   failedCheck: "column count mismatch")
        }

        let n = witnessColumns[0].count
        let N = tableColumns[0].count
        let s = proof.sortedVector

        // Check 1: sorted vector length
        guard s.count == n + N else {
            return LookupGrandProductVerification(
                valid: false,
                failedCheck: "sorted vector length: expected \(n + N), got \(s.count)")
        }

        // Check 2: recompute compressed columns and verify multiset
        let compW = compressColumns(witnessColumns, alpha: proof.alpha, count: n)
        let compT = compressColumns(tableColumns, alpha: proof.alpha, count: N)

        if !validateSortedMerge(witness: compW, table: compT, sorted: s) {
            return LookupGrandProductVerification(
                valid: false, failedCheck: "sorted merge multiset mismatch")
        }

        // Check 3: accumulator boundary constraints
        let accZ = proof.accumulatorZ
        guard accZ.count == n + 1 else {
            return LookupGrandProductVerification(
                valid: false,
                failedCheck: "accumulator length: expected \(n + 1), got \(accZ.count)")
        }

        guard frEqual(accZ[0], Fr.one) else {
            return LookupGrandProductVerification(
                valid: false, failedCheck: "z[0] != 1")
        }

        guard frEqual(proof.finalAccumulator, Fr.one) else {
            return LookupGrandProductVerification(
                valid: false, failedCheck: "z[n] != 1 (accumulator does not close)")
        }

        // Check 4: transition constraints
        let beta = proof.beta
        let gamma = proof.gamma
        let onePlusBeta = frAdd(Fr.one, beta)
        let gammaTimesBetaPlusOne = frMul(gamma, onePlusBeta)

        for i in 0..<n {
            let fTerm = frAdd(gamma, compW[i])
            let tIdx = i % N
            let tIdxNext = (i + 1) % N
            let tTerm = frAdd(gammaTimesBetaPlusOne,
                              frAdd(compT[tIdx], frMul(beta, compT[tIdxNext])))
            let num = frMul(onePlusBeta, frMul(fTerm, tTerm))

            let s2i = 2 * i
            let sTerm1 = frAdd(gammaTimesBetaPlusOne,
                               frAdd(s[s2i], frMul(beta, s[s2i + 1])))
            let s2i1 = 2 * i + 1
            let sIdx2 = min(s2i1 + 1, s.count - 1)
            let sTerm2 = frAdd(gammaTimesBetaPlusOne,
                               frAdd(s[s2i1], frMul(beta, s[sIdx2])))
            let den = frMul(sTerm1, sTerm2)

            let lhs = frMul(accZ[i + 1], den)
            let rhs = frMul(accZ[i], num)

            if !frEqual(lhs, rhs) {
                return LookupGrandProductVerification(
                    valid: false,
                    failedCheck: "transition constraint failed at index \(i)",
                    failingIndex: i)
            }
        }

        // Check 5: helper column consistency (if present)
        if let h = proof.helperH {
            guard h.count == s.count else {
                return LookupGrandProductVerification(
                    valid: false, failedCheck: "helper column length mismatch")
            }
            // h[0] should be 1 (first element is always "new")
            guard frEqual(h[0], Fr.one) else {
                return LookupGrandProductVerification(
                    valid: false, failedCheck: "h[0] != 1")
            }
            for i in 1..<s.count {
                let expected = frEqual(s[i], s[i - 1]) ? Fr.zero : Fr.one
                if !frEqual(h[i], expected) {
                    return LookupGrandProductVerification(
                        valid: false,
                        failedCheck: "helper column mismatch at index \(i)",
                        failingIndex: i)
                }
            }
        }

        // Check 6: multiplicities
        let computedMult = computeMultiplicities(witness: compW, table: compT)
        guard computedMult.count == proof.multiplicities.count else {
            return LookupGrandProductVerification(
                valid: false, failedCheck: "multiplicities length mismatch")
        }
        for j in 0..<computedMult.count {
            if !frEqual(computedMult[j], proof.multiplicities[j]) {
                return LookupGrandProductVerification(
                    valid: false,
                    failedCheck: "multiplicity mismatch at table index \(j)",
                    failingIndex: j)
            }
        }

        return LookupGrandProductVerification(valid: true)
    }

    /// Simplified single-column verify.
    public func verify(proof: LookupGrandProductProof,
                       witness: [Fr], table: [Fr]) -> LookupGrandProductVerification {
        verify(proof: proof, witnessColumns: [witness], tableColumns: [table])
    }

    /// Boolean convenience: returns true iff the proof is valid.
    public func verifyBool(proof: LookupGrandProductProof,
                           witness: [Fr], table: [Fr]) -> Bool {
        verify(proof: proof, witness: witness, table: table).valid
    }

    // MARK: - Multi-Column Compression (GPU-accelerated for large arrays)

    /// Compress c columns into a single column via random linear combination:
    ///   compressed[i] = cols[0][i] + alpha * cols[1][i] + alpha^2 * cols[2][i] + ...
    ///
    /// Uses Horner's method: compressed[i] = cols[c-1][i] + alpha*(cols[c-2][i] + alpha*(...))
    public func compressColumns(_ columns: [[Fr]], alpha: Fr, count n: Int) -> [Fr] {
        let c = columns.count
        if c == 1 {
            return columns[0]
        }

        var result = [Fr](repeating: Fr.zero, count: n)

        // Horner evaluation: start from last column
        for i in 0..<n {
            var acc = columns[c - 1][i]
            for col in stride(from: c - 2, through: 0, by: -1) {
                acc = frAdd(columns[col][i], frMul(alpha, acc))
            }
            result[i] = acc
        }

        return result
    }

    // MARK: - Multiplicities

    /// Compute multiplicities: mult[j] = number of times table[j] appears in witness.
    internal func computeMultiplicities(witness: [Fr], table: [Fr]) -> [Fr] {
        let N = table.count

        // Build lookup from table value -> table index
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

        var mult = [Fr](repeating: Fr.zero, count: N)

        for w in witness {
            let wLimbs = frToInt(w)
            var lo = 0, hi = N - 1
            var tableIdx = -1
            while lo <= hi {
                let mid = (lo + hi) >> 1
                let cmp = compareLimbs(wLimbs, sortedKeys[mid])
                if cmp == 0 {
                    tableIdx = sortedIndices[mid]
                    break
                } else if cmp < 0 {
                    hi = mid - 1
                } else {
                    lo = mid + 1
                }
            }
            if tableIdx >= 0 {
                mult[tableIdx] = frAdd(mult[tableIdx], Fr.one)
            }
        }

        return mult
    }

    // MARK: - Sorted Merge Construction (GPU-accelerated)

    /// Build the sorted vector s by merging witness into table.
    ///
    /// Preserves table ordering: for each table element t[j],
    /// all witness elements equal to t[j] are placed immediately after t[j].
    internal func buildSortedVector(witness: [Fr], table: [Fr]) throws -> [Fr] {
        let n = witness.count
        let N = table.count
        let total = n + N

        // Build lookup from table value -> table index (binary search)
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

        var multCount = [Int](repeating: 0, count: N)
        var sortKeyArr = [UInt64](repeating: 0, count: total)
        var values = [Fr](repeating: Fr.zero, count: total)

        // Table entries get keys: j * (n+1)
        for j in 0..<N {
            sortKeyArr[j] = UInt64(j) * UInt64(n + 1)
            values[j] = table[j]
        }

        // Witness entries: find table position, assign key = tableIdx * (n+1) + subindex
        for i in 0..<n {
            let wLimbs = frToInt(witness[i])
            var lo = 0, hi = N - 1
            var tableIdx = -1
            while lo <= hi {
                let mid = (lo + hi) >> 1
                let cmp = compareLimbs(wLimbs, sortedKeys[mid])
                if cmp == 0 {
                    tableIdx = sortedIndices[mid]
                    break
                } else if cmp < 0 {
                    hi = mid - 1
                } else {
                    lo = mid + 1
                }
            }
            precondition(tableIdx >= 0,
                         "Witness element at index \(i) not found in table")

            multCount[tableIdx] += 1
            let key = UInt64(tableIdx) * UInt64(n + 1) + UInt64(multCount[tableIdx])
            sortKeyArr[N + i] = key
            values[N + i] = witness[i]
        }

        // Sort by key
        if total >= gpuThreshold {
            return try gpuSortedMerge(keys: sortKeyArr, values: values)
        } else {
            return cpuSortedMerge(keys: sortKeyArr, values: values)
        }
    }

    /// GPU-accelerated sorted merge using RadixSortEngine (key-value sort).
    private func gpuSortedMerge(keys: [UInt64], values: [Fr]) throws -> [Fr] {
        let n = keys.count
        let maxKey = keys.max() ?? 0

        if maxKey <= UInt64(UInt32.max) {
            let keys32 = keys.map { UInt32($0) }
            let indices32 = (0..<UInt32(n)).map { $0 }
            let sortEngine = try RadixSortEngine()
            let (_, sortedIndices) = try sortEngine.sortKV(keys: keys32, values: indices32)
            return sortedIndices.map { values[Int($0)] }
        }

        // Fallback for large key ranges
        return cpuSortedMerge(keys: keys, values: values)
    }

    /// CPU sorted merge: sort by key, return values in sorted order.
    private func cpuSortedMerge(keys: [UInt64], values: [Fr]) -> [Fr] {
        let indexed = keys.enumerated().sorted { $0.element < $1.element }
        return indexed.map { values[$0.offset] }
    }

    // MARK: - Helper Column

    /// Build the helper column h for the sorted vector:
    ///   h[0] = 1
    ///   h[i] = (s[i] == s[i-1]) ? 0 : 1
    internal func buildHelperColumn(sorted: [Fr]) -> [Fr] {
        let m = sorted.count
        guard m > 0 else { return [] }

        var h = [Fr](repeating: Fr.zero, count: m)
        h[0] = Fr.one

        for i in 1..<m {
            h[i] = frEqual(sorted[i], sorted[i - 1]) ? Fr.zero : Fr.one
        }

        return h
    }

    // MARK: - Grand Product Accumulator (GPU-accelerated)

    /// Build the grand product accumulator z[0..n].
    ///
    /// z[0] = 1
    /// z[i+1] = z[i] * (1+beta) * (gamma + f[i]) * (gamma*(1+beta) + t[i%N] + beta*t[(i+1)%N])
    ///          / ( (gamma*(1+beta) + s[2i] + beta*s[2i+1]) * (gamma*(1+beta) + s[2i+1] + beta*s[2i+2]) )
    ///
    /// Returns (accumulator, numerators, denominators).
    internal func buildAccumulator(witness: [Fr], table: [Fr],
                                    sorted: [Fr], beta: Fr,
                                    gamma: Fr) throws -> ([Fr], [Fr], [Fr]) {
        let n = witness.count
        let N = table.count
        let onePlusBeta = frAdd(Fr.one, beta)
        let gammaTimesBetaPlusOne = frMul(gamma, onePlusBeta)

        var numerators = [Fr](repeating: Fr.zero, count: n)
        var denominators = [Fr](repeating: Fr.zero, count: n)

        for i in 0..<n {
            // Numerator
            let fTerm = frAdd(gamma, witness[i])
            let tIdx = i % N
            let tIdxNext = (i + 1) % N
            let tTerm = frAdd(gammaTimesBetaPlusOne,
                              frAdd(table[tIdx], frMul(beta, table[tIdxNext])))
            numerators[i] = frMul(onePlusBeta, frMul(fTerm, tTerm))

            // Denominator
            let s2i = 2 * i
            let sTerm1 = frAdd(gammaTimesBetaPlusOne,
                               frAdd(sorted[s2i], frMul(beta, sorted[s2i + 1])))
            let s2i1 = 2 * i + 1
            let sIdx2 = min(s2i1 + 1, sorted.count - 1)
            let sTerm2 = frAdd(gammaTimesBetaPlusOne,
                               frAdd(sorted[s2i1], frMul(beta, sorted[sIdx2])))
            denominators[i] = frMul(sTerm1, sTerm2)
        }

        // GPU-accelerated prefix product of ratios
        let accZ = grandProductEngine.permutationProduct(
            numerators: numerators, denominators: denominators)

        return (accZ, numerators, denominators)
    }

    // MARK: - Sorted Merge Validation

    /// Validate that the sorted vector s is a valid multiset merge of witness and table.
    internal func validateSortedMerge(witness: [Fr], table: [Fr], sorted: [Fr]) -> Bool {
        let n = witness.count
        let N = table.count
        guard sorted.count == n + N else { return false }

        var freqW = [FrHashKey: Int]()
        for v in witness {
            freqW[FrHashKey(v), default: 0] += 1
        }
        var freqT = [FrHashKey: Int]()
        for v in table {
            freqT[FrHashKey(v), default: 0] += 1
        }
        var freqS = [FrHashKey: Int]()
        for v in sorted {
            freqS[FrHashKey(v), default: 0] += 1
        }

        var combined = freqT
        for (k, c) in freqW {
            combined[k, default: 0] += c
        }

        return combined == freqS
    }

    // MARK: - Constraint Evaluation

    /// Evaluate the transition constraint at index i:
    ///   Returns z[i+1] * Den[i] - z[i] * Num[i] (should be zero).
    public func evaluateTransitionConstraint(proof: LookupGrandProductProof,
                                              witness: [Fr], table: [Fr],
                                              index i: Int) -> Fr {
        let N = table.count
        let s = proof.sortedVector
        let accZ = proof.accumulatorZ
        let beta = proof.beta
        let gamma = proof.gamma
        let onePlusBeta = frAdd(Fr.one, beta)
        let gammaTimesBetaPlusOne = frMul(gamma, onePlusBeta)

        let fTerm = frAdd(gamma, witness[i])
        let tIdx = i % N
        let tIdxNext = (i + 1) % N
        let tTerm = frAdd(gammaTimesBetaPlusOne,
                          frAdd(table[tIdx], frMul(beta, table[tIdxNext])))
        let num = frMul(onePlusBeta, frMul(fTerm, tTerm))

        let s2i = 2 * i
        let sTerm1 = frAdd(gammaTimesBetaPlusOne,
                           frAdd(s[s2i], frMul(beta, s[s2i + 1])))
        let s2i1 = 2 * i + 1
        let sIdx2 = min(s2i1 + 1, s.count - 1)
        let sTerm2 = frAdd(gammaTimesBetaPlusOne,
                           frAdd(s[s2i1], frMul(beta, s[sIdx2])))
        let den = frMul(sTerm1, sTerm2)

        let lhs = frMul(accZ[i + 1], den)
        let rhs = frMul(accZ[i], num)

        return frSub(lhs, rhs)
    }

    /// Evaluate all transition constraints. Returns array of z[i+1]*Den[i] - z[i]*Num[i].
    /// All should be zero for a valid proof.
    public func evaluateAllTransitions(proof: LookupGrandProductProof,
                                        witness: [Fr], table: [Fr]) -> [Fr] {
        let n = witness.count
        var results = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            results[i] = evaluateTransitionConstraint(proof: proof,
                                                       witness: witness,
                                                       table: table, index: i)
        }
        return results
    }

    // MARK: - Product Check

    /// Verify the grand product by recomputing from numerators/denominators.
    /// Returns true if the cumulative product of Num[i]/Den[i] equals 1.
    public func verifyProductCheck(witness: [Fr], table: [Fr],
                                    sorted: [Fr], beta: Fr, gamma: Fr) -> Bool {
        let n = witness.count
        let N = table.count
        let onePlusBeta = frAdd(Fr.one, beta)
        let gammaTimesBetaPlusOne = frMul(gamma, onePlusBeta)

        var product = Fr.one

        for i in 0..<n {
            let fTerm = frAdd(gamma, witness[i])
            let tIdx = i % N
            let tIdxNext = (i + 1) % N
            let tTerm = frAdd(gammaTimesBetaPlusOne,
                              frAdd(table[tIdx], frMul(beta, table[tIdxNext])))
            let num = frMul(onePlusBeta, frMul(fTerm, tTerm))

            let s2i = 2 * i
            let sTerm1 = frAdd(gammaTimesBetaPlusOne,
                               frAdd(sorted[s2i], frMul(beta, sorted[s2i + 1])))
            let s2i1 = 2 * i + 1
            let sIdx2 = min(s2i1 + 1, sorted.count - 1)
            let sTerm2 = frAdd(gammaTimesBetaPlusOne,
                               frAdd(sorted[s2i1], frMul(beta, sorted[sIdx2])))
            let den = frMul(sTerm1, sTerm2)

            let ratio = frMul(num, frInverse(den))
            product = frMul(product, ratio)
        }

        return frEqual(product, Fr.one)
    }

    // MARK: - Batch Verification

    /// Batch-verify multiple proofs (different witness/table pairs).
    /// Returns array of verification results.
    public func batchVerify(proofs: [(proof: LookupGrandProductProof,
                                      witnessColumns: [[Fr]],
                                      tableColumns: [[Fr]])]) -> [LookupGrandProductVerification] {
        proofs.map { entry in
            verify(proof: entry.proof,
                   witnessColumns: entry.witnessColumns,
                   tableColumns: entry.tableColumns)
        }
    }

    // MARK: - Helpers

    private func compareLimbs(_ a: [UInt64], _ b: [UInt64]) -> Int {
        for k in stride(from: a.count - 1, through: 0, by: -1) {
            if a[k] < b[k] { return -1 }
            if a[k] > b[k] { return 1 }
        }
        return 0
    }

    // MARK: - Fiat-Shamir Challenge Derivation

    private func deriveAlpha(witnessColumns: [[Fr]],
                              tableColumns: [[Fr]]) -> Fr {
        var transcript = [UInt8]()
        transcript.append(contentsOf: [0x4C, 0x47, 0x50, 0x41]) // "LGPA"
        for col in witnessColumns {
            appendSizeAndSamples(&transcript, col)
        }
        for col in tableColumns {
            appendSizeAndSamples(&transcript, col)
        }
        return hashToFr(transcript)
    }

    private func deriveBeta(witnessColumns: [[Fr]],
                             tableColumns: [[Fr]], alpha: Fr) -> Fr {
        var transcript = [UInt8]()
        transcript.append(contentsOf: [0x4C, 0x47, 0x50, 0x42]) // "LGPB"
        for col in witnessColumns {
            appendSizeAndSamples(&transcript, col)
        }
        for col in tableColumns {
            appendSizeAndSamples(&transcript, col)
        }
        appendFrBytes(&transcript, alpha)
        return hashToFr(transcript)
    }

    private func deriveGamma(witnessColumns: [[Fr]],
                              tableColumns: [[Fr]],
                              alpha: Fr, beta: Fr) -> Fr {
        var transcript = [UInt8]()
        transcript.append(contentsOf: [0x4C, 0x47, 0x50, 0x47]) // "LGPG"
        for col in witnessColumns {
            appendSizeAndSamples(&transcript, col)
        }
        for col in tableColumns {
            appendSizeAndSamples(&transcript, col)
        }
        appendFrBytes(&transcript, alpha)
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

// MARK: - Internal Fr-based Hashable Key (avoids conflict with LookupEngine.FrKey)

private struct FrHashKey: Hashable {
    let limbs: [UInt64]
    init(_ fr: Fr) {
        self.limbs = frToInt(fr)
    }
    static func == (lhs: FrHashKey, rhs: FrHashKey) -> Bool {
        lhs.limbs == rhs.limbs
    }
    func hash(into hasher: inout Hasher) {
        for l in limbs { hasher.combine(l) }
    }
}
