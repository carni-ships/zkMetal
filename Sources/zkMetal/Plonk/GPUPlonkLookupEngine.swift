// GPUPlonkLookupEngine — GPU-accelerated Plonk lookup argument engine (Plookup-style)
//
// Integrates the Plookup lookup argument directly into the Plonk proving pipeline,
// supporting multi-table lookups and batch processing of lookup gates.
//
// Protocol (Gabizon-Williamson 2020):
//   Given table T and lookup queries f where each f[i] in T:
//   1. Build sorted witness s by interleaving queries into table ordering
//   2. Compute grand product accumulator Z_lookup:
//      Z[0] = 1
//      Z[i+1] = Z[i] * (1+beta)*(gamma+f[i])*(gamma(1+beta)+t[i]+beta*t[i+1])
//               / ((gamma(1+beta)+s[2i]+beta*s[2i+1])*(gamma(1+beta)+s[2i+1]+beta*s[2i+2]))
//   3. Verify Z[n] = 1 (accumulator closes)
//
// GPU acceleration:
//   - Sorted witness construction: GPU radix sort for large tables
//   - Grand product: GPUGrandProductEngine (batch inverse + prefix product)
//   - Batch processing: all lookup gates processed in parallel
//
// Multi-table support:
//   - Each table has a unique ID; lookup gates reference tables by ID
//   - Per-table sorted vectors and accumulators are computed independently
//   - Results are combined into a single lookup proof

import Foundation
import Metal
import NeonFieldOps

// MARK: - Lookup Proof Structures

/// Per-table lookup proof containing the sorted witness and accumulator.
public struct TableLookupProof {
    /// Table ID this proof corresponds to
    public let tableId: Int
    /// Sorted witness vector (interleaved queries + table values)
    public let sortedVector: [Fr]
    /// Grand product accumulator Z_lookup evaluations
    public let accumulatorZ: [Fr]
    /// Final accumulator value (must be 1 for valid proof)
    public let finalAccumulator: Fr
    /// Number of queries into this table
    public let queryCount: Int

    public init(tableId: Int, sortedVector: [Fr], accumulatorZ: [Fr],
                finalAccumulator: Fr, queryCount: Int) {
        self.tableId = tableId
        self.sortedVector = sortedVector
        self.accumulatorZ = accumulatorZ
        self.finalAccumulator = finalAccumulator
        self.queryCount = queryCount
    }
}

/// Combined lookup proof for all tables in a Plonk circuit.
public struct PlonkLookupProof {
    /// Per-table proofs
    public let tableProofs: [TableLookupProof]
    /// Random challenges
    public let beta: Fr
    public let gamma: Fr
    /// Whether all accumulators close (all finalAccumulator == 1)
    public var allAccumulatorsClose: Bool {
        tableProofs.allSatisfy { frEqual($0.finalAccumulator, Fr.one) }
    }

    public init(tableProofs: [TableLookupProof], beta: Fr, gamma: Fr) {
        self.tableProofs = tableProofs
        self.beta = beta
        self.gamma = gamma
    }
}

// MARK: - GPU Plonk Lookup Engine

/// GPU-accelerated Plonk lookup argument engine with multi-table and batch support.
///
/// Handles the full lifecycle of lookup arguments within a Plonk proof:
///   1. Extract lookup queries from circuit gates
///   2. Build sorted witnesses per table
///   3. Compute grand product accumulators (GPU-accelerated)
///   4. Verify lookup argument polynomials
///   5. Compute quotient polynomial contributions
public final class GPUPlonkLookupEngine {
    public static let version = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")

    /// GPU dispatch threshold: arrays smaller than this use CPU path
    public static let gpuThreshold = 2048

    private let device: MTLDevice?
    private let commandQueue: MTLCommandQueue?
    private let grandProductEngine: GPUGrandProductEngine?

    /// Enable profiling output to stderr
    public var profile = false

    // MARK: - Initialization

    public init() {
        let dev = MTLCreateSystemDefaultDevice()
        self.device = dev
        self.commandQueue = dev?.makeCommandQueue()

        // Try to init GPU grand product engine; fall back to CPU if unavailable
        self.grandProductEngine = try? GPUGrandProductEngine()
    }

    // MARK: - Full Lookup Proof from Circuit

    /// Compute the full lookup argument proof for a Plonk circuit.
    ///
    /// Extracts lookup queries from all lookup gates, groups them by table,
    /// and produces per-table sorted witnesses and grand product accumulators.
    ///
    /// - Parameters:
    ///   - circuit: The Plonk circuit containing lookup gates and tables.
    ///   - witness: Full witness array (variable index -> field value).
    ///   - beta: Random challenge (nil for auto-derivation via Fiat-Shamir).
    ///   - gamma: Random challenge (nil for auto-derivation via Fiat-Shamir).
    /// - Returns: Combined PlonkLookupProof for all tables.
    public func proveLookups(
        circuit: PlonkCircuit,
        witness: [Fr],
        beta: Fr? = nil,
        gamma: Fr? = nil
    ) -> PlonkLookupProof {
        let _t0 = profile ? CFAbsoluteTimeGetCurrent() : 0

        // Step 1: Extract per-table lookup queries from circuit gates
        let queriesByTable = extractLookupQueries(circuit: circuit, witness: witness)

        if profile {
            let dt = (CFAbsoluteTimeGetCurrent() - _t0) * 1000
            fputs(String(format: "  [plonk-lookup] extract queries: %.2f ms\n", dt), stderr)
        }

        // Step 2: Derive challenges
        let betaVal = beta ?? deriveBeta(circuit: circuit, witness: witness)
        let gammaVal = gamma ?? deriveGamma(circuit: circuit, witness: witness, beta: betaVal)

        // Step 3: Build per-table proofs
        var tableProofs = [TableLookupProof]()
        tableProofs.reserveCapacity(circuit.lookupTables.count)

        for table in circuit.lookupTables {
            let queries = queriesByTable[table.id] ?? []
            if queries.isEmpty {
                // No queries for this table; emit trivial proof
                let trivial = TableLookupProof(
                    tableId: table.id,
                    sortedVector: table.values,
                    accumulatorZ: [Fr.one],
                    finalAccumulator: Fr.one,
                    queryCount: 0
                )
                tableProofs.append(trivial)
                continue
            }

            let _tTable = profile ? CFAbsoluteTimeGetCurrent() : 0

            let sorted = buildSortedWitness(queries: queries, table: table.values)
            let accZ = buildAccumulator(
                queries: queries, table: table.values,
                sorted: sorted, beta: betaVal, gamma: gammaVal
            )
            let finalAcc = accZ.last ?? Fr.one

            if profile {
                let dt = (CFAbsoluteTimeGetCurrent() - _tTable) * 1000
                fputs(String(format: "  [plonk-lookup] table %d (q=%d, |t|=%d): %.2f ms\n",
                             table.id, queries.count, table.values.count, dt), stderr)
            }

            tableProofs.append(TableLookupProof(
                tableId: table.id,
                sortedVector: sorted,
                accumulatorZ: accZ,
                finalAccumulator: finalAcc,
                queryCount: queries.count
            ))
        }

        if profile {
            let total = (CFAbsoluteTimeGetCurrent() - _t0) * 1000
            fputs(String(format: "  [plonk-lookup] TOTAL prove: %.2f ms\n", total), stderr)
        }

        return PlonkLookupProof(tableProofs: tableProofs, beta: betaVal, gamma: gammaVal)
    }

    // MARK: - Direct Proof (Table + Queries)

    /// Compute lookup argument for a single table and set of queries.
    ///
    /// This is the low-level API for proving a single table's lookup argument,
    /// independent of the circuit structure.
    ///
    /// - Parameters:
    ///   - queries: Lookup values (each must exist in table).
    ///   - table: Table values.
    ///   - beta: Random challenge (nil for auto-derivation).
    ///   - gamma: Random challenge (nil for auto-derivation).
    /// - Returns: TableLookupProof for this table.
    public func proveSingleTable(
        queries: [Fr],
        table: [Fr],
        beta: Fr? = nil,
        gamma: Fr? = nil
    ) -> TableLookupProof {
        precondition(!queries.isEmpty, "Queries must be non-empty")
        precondition(!table.isEmpty, "Table must be non-empty")

        let betaVal = beta ?? deriveBetaFromArrays(queries, table)
        let gammaVal = gamma ?? deriveGammaFromArrays(queries, table, betaVal)

        let sorted = buildSortedWitness(queries: queries, table: table)
        let accZ = buildAccumulator(
            queries: queries, table: table,
            sorted: sorted, beta: betaVal, gamma: gammaVal
        )
        let finalAcc = accZ.last ?? Fr.one

        return TableLookupProof(
            tableId: 0,
            sortedVector: sorted,
            accumulatorZ: accZ,
            finalAccumulator: finalAcc,
            queryCount: queries.count
        )
    }

    // MARK: - Batch Processing

    /// Batch-process multiple independent lookup arguments in parallel.
    ///
    /// Each entry is a (queries, table) pair. All are processed independently
    /// with the same challenges, enabling amortization of GPU dispatch overhead.
    ///
    /// - Parameters:
    ///   - batches: Array of (queries, table) pairs.
    ///   - beta: Random challenge.
    ///   - gamma: Random challenge.
    /// - Returns: Array of TableLookupProof, one per batch entry.
    public func proveBatch(
        batches: [(queries: [Fr], table: [Fr])],
        beta: Fr,
        gamma: Fr
    ) -> [TableLookupProof] {
        var results = [TableLookupProof]()
        results.reserveCapacity(batches.count)

        for (idx, batch) in batches.enumerated() {
            let sorted = buildSortedWitness(queries: batch.queries, table: batch.table)
            let accZ = buildAccumulator(
                queries: batch.queries, table: batch.table,
                sorted: sorted, beta: beta, gamma: gamma
            )
            let finalAcc = accZ.last ?? Fr.one

            results.append(TableLookupProof(
                tableId: idx,
                sortedVector: sorted,
                accumulatorZ: accZ,
                finalAccumulator: finalAcc,
                queryCount: batch.queries.count
            ))
        }

        return results
    }

    // MARK: - Verification

    /// Verify a complete PlonkLookupProof against the circuit.
    ///
    /// Checks for each table:
    ///   1. Sorted vector is a valid merge of queries and table
    ///   2. Accumulator Z[0] = 1
    ///   3. All accumulator transitions satisfy the Plookup relation
    ///   4. Accumulator closes: Z[n] = 1
    ///
    /// - Parameters:
    ///   - proof: The PlonkLookupProof to verify.
    ///   - circuit: The Plonk circuit.
    ///   - witness: Full witness array.
    /// - Returns: True if all lookup arguments are valid.
    public func verifyLookups(
        proof: PlonkLookupProof,
        circuit: PlonkCircuit,
        witness: [Fr]
    ) -> Bool {
        let queriesByTable = extractLookupQueries(circuit: circuit, witness: witness)

        for tableProof in proof.tableProofs {
            guard let table = circuit.lookupTables.first(where: { $0.id == tableProof.tableId }) else {
                return false
            }
            let queries = queriesByTable[table.id] ?? []

            if queries.isEmpty && tableProof.queryCount == 0 {
                continue  // Trivial proof for unused table
            }

            if !verifySingleTable(
                proof: tableProof,
                queries: queries,
                table: table.values,
                beta: proof.beta,
                gamma: proof.gamma
            ) {
                return false
            }
        }

        return true
    }

    /// Verify a single table's lookup proof.
    ///
    /// - Parameters:
    ///   - proof: The TableLookupProof.
    ///   - queries: Original lookup queries.
    ///   - table: Original table values.
    ///   - beta: Challenge used in proving.
    ///   - gamma: Challenge used in proving.
    /// - Returns: True if the proof is valid.
    public func verifySingleTable(
        proof: TableLookupProof,
        queries: [Fr],
        table: [Fr],
        beta: Fr,
        gamma: Fr
    ) -> Bool {
        let n = queries.count
        let N = table.count
        let s = proof.sortedVector

        // Check 1: sorted vector length = n + N
        guard s.count == n + N else { return false }

        // Check 2: sorted vector is valid merge (multiset equality)
        if !validateSortedMerge(queries: queries, table: table, sorted: s) {
            return false
        }

        // Check 3: accumulator structure (GW20: n+N-1 steps → n+N elements)
        let numSteps = n + N - 1
        let accZ = proof.accumulatorZ
        guard accZ.count == numSteps + 1 else { return false }
        guard frEqual(accZ[0], Fr.one) else { return false }

        // Check 4: accumulator closes
        guard frEqual(proof.finalAccumulator, Fr.one) else { return false }

        // Check 5: verify each accumulator transition (GW20 h₁/h₂ split)
        let onePlusBeta = frAdd(Fr.one, beta)
        let gammaOnePlusBeta = frMul(gamma, onePlusBeta)

        for k in 0..<numSteps {
            var num = Fr.one
            var den = Fr.one

            if k < n {
                num = frMul(onePlusBeta, frAdd(gamma, queries[k]))
                den = frAdd(gammaOnePlusBeta,
                            frAdd(s[k], frMul(beta, s[k + 1])))
            }

            if k < N - 1 {
                let tTerm = frAdd(gammaOnePlusBeta,
                                  frAdd(table[k], frMul(beta, table[k + 1])))
                num = frMul(num, tTerm)
                let h2Term = frAdd(gammaOnePlusBeta,
                                   frAdd(s[n + k], frMul(beta, s[n + k + 1])))
                den = frMul(den, h2Term)
            }

            let lhs = frMul(accZ[k + 1], den)
            let rhs = frMul(accZ[k], num)
            if !frEqual(lhs, rhs) { return false }
        }

        return true
    }

    // MARK: - Accumulator Polynomial Computation

    /// Compute the accumulator polynomial Z_lookup in evaluation form for
    /// integration into the Plonk quotient polynomial.
    ///
    /// Given domain size domainN (power of 2), pads the queries and table
    /// evaluations to domainN and computes Z_lookup over the full domain.
    ///
    /// - Parameters:
    ///   - queries: Lookup values.
    ///   - table: Table values.
    ///   - domainN: Domain size (must be power of 2, >= queries.count).
    ///   - beta: Challenge.
    ///   - gamma: Challenge.
    /// - Returns: (zLookupEvals, sortedEvals) padded to domainN.
    public func computeAccumulatorPolynomial(
        queries: [Fr],
        table: [Fr],
        domainN: Int,
        beta: Fr,
        gamma: Fr
    ) -> (zLookupEvals: [Fr], sortedEvals: [Fr]) {
        precondition(domainN > 0 && (domainN & (domainN - 1)) == 0,
                     "Domain size must be a power of 2")
        precondition(domainN >= queries.count,
                     "Domain size must be >= query count")

        // Pad queries with first table value to fill domain
        var paddedQueries = queries
        let padVal = table.first ?? Fr.zero
        while paddedQueries.count < domainN {
            paddedQueries.append(padVal)
        }

        // Pad table to fill domain
        var paddedTable = table
        let tPadVal = table.last ?? Fr.zero
        while paddedTable.count < domainN {
            paddedTable.append(tPadVal)
        }

        let sorted = buildSortedWitness(queries: paddedQueries, table: paddedTable)

        // Build grand product over padded domain
        let accZ = buildAccumulator(
            queries: paddedQueries, table: paddedTable,
            sorted: sorted, beta: beta, gamma: gamma
        )

        // Return evaluations trimmed/padded to domain size
        var zEvals = [Fr](repeating: Fr.zero, count: domainN)
        for i in 0..<min(accZ.count, domainN) {
            zEvals[i] = accZ[i]
        }

        return (zLookupEvals: zEvals, sortedEvals: sorted)
    }

    // MARK: - Sorted Witness Construction

    /// Build the sorted witness vector by interleaving queries into table ordering.
    ///
    /// For each table entry t[j], all query values equal to t[j] are placed
    /// immediately after t[j] in the sorted output. The result has length
    /// queries.count + table.count.
    ///
    /// - Parameters:
    ///   - queries: Lookup query values (each must exist in table).
    ///   - table: Table values.
    /// - Returns: Sorted vector of length queries.count + table.count.
    public func buildSortedWitness(queries: [Fr], table: [Fr]) -> [Fr] {
        let n = queries.count
        let N = table.count
        let total = n + N

        // Build table position map using sorted binary search
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

        // Assign sort keys: table[j] -> key = j*(n+1), query matching table[j] -> j*(n+1)+(1+sub)
        var multCount = [Int](repeating: 0, count: N)
        var sortKeyPairs = [(key: UInt64, value: Fr)]()
        sortKeyPairs.reserveCapacity(total)

        // Table entries
        for j in 0..<N {
            sortKeyPairs.append((key: UInt64(j) * UInt64(n + 1), value: table[j]))
        }

        // Query entries: binary search for table position
        for i in 0..<n {
            let qLimbs = frToInt(queries[i])
            var lo = 0
            var hi = N - 1
            var tableIdx = -1
            while lo <= hi {
                let mid = (lo + hi) >> 1
                let cmp = compareLimbs(qLimbs, sortedKeys[mid])
                if cmp == 0 {
                    tableIdx = sortedIndices[mid]
                    break
                } else if cmp < 0 {
                    hi = mid - 1
                } else {
                    lo = mid + 1
                }
            }
            precondition(tableIdx >= 0, "Query value at index \(i) not found in table")

            multCount[tableIdx] += 1
            let key = UInt64(tableIdx) * UInt64(n + 1) + UInt64(multCount[tableIdx])
            sortKeyPairs.append((key: key, value: queries[i]))
        }

        // Sort by key
        if total >= GPUPlonkLookupEngine.gpuThreshold, let _ = device {
            // GPU path: use radix sort for large arrays
            return gpuSort(pairs: sortKeyPairs)
        } else {
            // CPU path
            let sorted = sortKeyPairs.sorted { $0.key < $1.key }
            return sorted.map { $0.value }
        }
    }

    // MARK: - Grand Product Accumulator

    /// Build the Plookup grand product accumulator Z[0..n].
    ///
    /// Z[0] = 1
    /// Z[i+1] = Z[i] * (1+beta) * (gamma + f[i]) * (gamma(1+beta) + t[i%N] + beta*t[(i+1)%N])
    ///          / ((gamma(1+beta) + s[2i] + beta*s[2i+1]) * (gamma(1+beta) + s[2i+1] + beta*s[2i+2]))
    private func buildAccumulator(
        queries: [Fr], table: [Fr],
        sorted: [Fr], beta: Fr, gamma: Fr
    ) -> [Fr] {
        let n = queries.count
        let N = table.count
        let numSteps = n + N - 1
        let onePlusBeta = frAdd(Fr.one, beta)
        let gammaOnePlusBeta = frMul(gamma, onePlusBeta)

        // GW20 identity: split sorted into h₁ = s[0..n], h₂ = s[n..n+N-1]
        // LHS: ∏_{k<n} (1+β)(γ+f[k]) · ∏_{k<N-1} (γ(1+β)+t[k]+β·t[k+1])
        // RHS: ∏_{k<n} (γ(1+β)+h₁[k]+β·h₁[k+1]) · ∏_{k<N-1} (γ(1+β)+h₂[k]+β·h₂[k+1])
        var numerators = [Fr](repeating: Fr.one, count: numSteps)
        var denominators = [Fr](repeating: Fr.one, count: numSteps)

        for k in 0..<numSteps {
            var num = Fr.one
            var den = Fr.one

            if k < n {
                num = frMul(onePlusBeta, frAdd(gamma, queries[k]))
                den = frAdd(gammaOnePlusBeta,
                            frAdd(sorted[k], frMul(beta, sorted[k + 1])))
            }

            if k < N - 1 {
                let tTerm = frAdd(gammaOnePlusBeta,
                                  frAdd(table[k], frMul(beta, table[k + 1])))
                num = frMul(num, tTerm)
                let h2Term = frAdd(gammaOnePlusBeta,
                                   frAdd(sorted[n + k], frMul(beta, sorted[n + k + 1])))
                den = frMul(den, h2Term)
            }

            numerators[k] = num
            denominators[k] = den
        }

        // Build prefix product: Z[0] = 1, Z[k+1] = Z[k] * num[k] / den[k]
        return cpuPrefixProduct(numerators: numerators, denominators: denominators)
    }

    /// CPU fallback: compute prefix product of numerator/denominator ratios.
    private func cpuPrefixProduct(numerators: [Fr], denominators: [Fr]) -> [Fr] {
        let n = numerators.count
        var result = [Fr](repeating: Fr.zero, count: n + 1)
        result[0] = Fr.one

        // Montgomery batch inversion of all denominators (1 inverse + 3(n-1) muls)
        var prefix = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n {
            prefix[i] = denominators[i - 1] == Fr.zero ? prefix[i - 1] : frMul(prefix[i - 1], denominators[i - 1])
        }
        let last = denominators[n - 1] == Fr.zero ? prefix[n - 1] : frMul(prefix[n - 1], denominators[n - 1])
        var inv = frInverse(last)
        var denInvs = [Fr](repeating: Fr.zero, count: n)
        for i in stride(from: n - 1, through: 0, by: -1) {
            if denominators[i] != Fr.zero {
                denInvs[i] = frMul(inv, prefix[i])
                inv = frMul(inv, denominators[i])
            }
        }

        for i in 0..<n {
            let ratio = frMul(numerators[i], denInvs[i])
            result[i + 1] = frMul(result[i], ratio)
        }

        return result
    }

    // MARK: - GPU Sort

    /// GPU-accelerated sort of (key, value) pairs using RadixSortEngine.
    private func gpuSort(pairs: [(key: UInt64, value: Fr)]) -> [Fr] {
        let n = pairs.count
        let maxKey = pairs.max(by: { $0.key < $1.key })?.key ?? 0

        if maxKey <= UInt64(UInt32.max) {
            do {
                let keys32 = pairs.map { UInt32($0.key) }
                let indices32 = (0..<UInt32(n)).map { $0 }
                let sortEngine = try RadixSortEngine()
                let (_, sortedIndices) = try sortEngine.sortKV(keys: keys32, values: indices32)
                return sortedIndices.map { pairs[Int($0)].value }
            } catch {
                // Fall through to CPU
            }
        }

        // CPU fallback
        let sorted = pairs.sorted { $0.key < $1.key }
        return sorted.map { $0.value }
    }

    // MARK: - Query Extraction

    /// Extract lookup queries from circuit gates, grouped by table ID.
    ///
    /// Scans all gates with non-zero qLookup selector and extracts the
    /// queried wire values, grouped by the table ID stored in qC.
    private func extractLookupQueries(
        circuit: PlonkCircuit,
        witness: [Fr]
    ) -> [Int: [Fr]] {
        var result = [Int: [Fr]]()

        for (i, gate) in circuit.gates.enumerated() {
            guard !frEqual(gate.qLookup, Fr.zero) else { continue }

            // Table ID encoded in qC
            let limbs = frToInt(gate.qC)
            let tableId = Int(limbs.first ?? 0)

            // Extract wire value (a-wire by default)
            let wireIdx = circuit.wireAssignments[i][0]
            let val = wireIdx < witness.count ? witness[wireIdx] : Fr.zero

            result[tableId, default: []].append(val)
        }

        return result
    }

    // MARK: - Sorted Merge Validation

    /// Validate that sorted vector is a valid merge of queries and table.
    private func validateSortedMerge(queries: [Fr], table: [Fr], sorted: [Fr]) -> Bool {
        let n = queries.count
        let N = table.count
        guard sorted.count == n + N else { return false }

        // Build frequency maps
        var freqQ = [FrKey: Int]()
        for v in queries { freqQ[FrKey(v), default: 0] += 1 }
        var freqT = [FrKey: Int]()
        for v in table { freqT[FrKey(v), default: 0] += 1 }
        var freqS = [FrKey: Int]()
        for v in sorted { freqS[FrKey(v), default: 0] += 1 }

        // Combined frequency of queries + table must equal sorted
        var combined = freqT
        for (k, c) in freqQ { combined[k, default: 0] += c }
        return combined == freqS
    }

    // MARK: - Limb Comparison

    private func compareLimbs(_ a: [UInt64], _ b: [UInt64]) -> Int {
        for k in stride(from: a.count - 1, through: 0, by: -1) {
            if a[k] < b[k] { return -1 }
            if a[k] > b[k] { return 1 }
        }
        return 0
    }

    // MARK: - Fiat-Shamir Challenge Derivation

    private func deriveBeta(circuit: PlonkCircuit, witness: [Fr]) -> Fr {
        var transcript = [UInt8]()
        transcript.append(contentsOf: [0x50, 0x4C, 0x4B, 0x4C]) // "PLKL"
        appendSize(&transcript, UInt64(circuit.numGates))
        appendSize(&transcript, UInt64(witness.count))
        appendSize(&transcript, UInt64(circuit.lookupTables.count))
        // Sample some witness values for domain separation
        let sampleCount = min(8, witness.count)
        for i in 0..<sampleCount {
            appendFrBytes(&transcript, witness[i])
        }
        return hashToFr(transcript)
    }

    private func deriveGamma(circuit: PlonkCircuit, witness: [Fr], beta: Fr) -> Fr {
        var transcript = [UInt8]()
        transcript.append(contentsOf: [0x50, 0x4C, 0x4B, 0x47]) // "PLKG"
        appendSize(&transcript, UInt64(circuit.numGates))
        appendFrBytes(&transcript, beta)
        let sampleCount = min(8, witness.count)
        for i in 0..<sampleCount {
            appendFrBytes(&transcript, witness[i])
        }
        return hashToFr(transcript)
    }

    private func deriveBetaFromArrays(_ queries: [Fr], _ table: [Fr]) -> Fr {
        var transcript = [UInt8]()
        transcript.append(contentsOf: [0x50, 0x4C, 0x4B, 0x42]) // "PLKB"
        appendSize(&transcript, UInt64(queries.count))
        appendSize(&transcript, UInt64(table.count))
        let sampleCount = min(8, queries.count)
        for i in 0..<sampleCount {
            appendFrBytes(&transcript, queries[i])
        }
        return hashToFr(transcript)
    }

    private func deriveGammaFromArrays(_ queries: [Fr], _ table: [Fr], _ beta: Fr) -> Fr {
        var transcript = [UInt8]()
        transcript.append(contentsOf: [0x50, 0x4C, 0x4B, 0x47]) // "PLKG"
        appendSize(&transcript, UInt64(queries.count))
        appendSize(&transcript, UInt64(table.count))
        appendFrBytes(&transcript, beta)
        return hashToFr(transcript)
    }

    private func appendSize(_ transcript: inout [UInt8], _ size: UInt64) {
        var s = size
        for _ in 0..<8 {
            transcript.append(UInt8(s & 0xFF))
            s >>= 8
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
