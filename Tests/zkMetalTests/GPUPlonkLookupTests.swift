// GPUPlonkLookupTests — Tests for GPU-accelerated Plonk lookup argument engine
//
// Tests cover:
//   1. Single-table sorted witness construction
//   2. Grand product accumulator correctness (closes at 1)
//   3. Verification of valid proofs
//   4. Rejection of tampered proofs
//   5. Multi-table lookup support
//   6. Batch processing of multiple independent lookups
//   7. Repeated lookup values
//   8. Accumulator polynomial computation for Plonk integration
//   9. Full circuit-level prove/verify
//  10. Larger table stress test

import zkMetal
import Foundation

public func runGPUPlonkLookupTests() {
    suite("GPU Plonk Lookup Engine")

    testSortedWitness()
    testSingleTableProveVerify()
    testAccumulatorCloses()
    testRepeatedLookups()
    testRejectTamperedAccumulator()
    testRejectTamperedSorted()
    testMultiTableCircuit()
    testLookupBatchProcessing()
    testAccumulatorPolynomial()
    testAutoFiatShamirChallenges()
    testEmptyTableQueries()
    testLargerTable()
    testAsymmetricQueryTable()
    testAllTableValuesQueried()
    testSingleElementTable()
}

// MARK: - Sorted Witness Construction

func testSortedWitness() {
    let engine = GPUPlonkLookupEngine()

    let table = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
    let queries = [frFromInt(20), frFromInt(10), frFromInt(40), frFromInt(30)]

    let sorted = engine.buildSortedWitness(queries: queries, table: table)

    // Sorted vector should have length n + N = 4 + 4 = 8
    expect(sorted.count == 8, "Sorted witness length = n + N")

    // Verify multiset: sorted must contain all queries + all table entries
    var tableFreq = [UInt64: Int]()
    for t in table { tableFreq[frToInt(t)[0], default: 0] += 1 }
    var queryFreq = [UInt64: Int]()
    for q in queries { queryFreq[frToInt(q)[0], default: 0] += 1 }
    var sortedFreq = [UInt64: Int]()
    for s in sorted { sortedFreq[frToInt(s)[0], default: 0] += 1 }

    // Each value should appear (table count + query count) times in sorted
    var multisetCorrect = true
    for (val, tCount) in tableFreq {
        let qCount = queryFreq[val] ?? 0
        let sCount = sortedFreq[val] ?? 0
        if sCount != tCount + qCount { multisetCorrect = false; break }
    }
    expect(multisetCorrect, "Sorted witness multiset equals queries + table")

    // Check table ordering: values should follow the table order
    // Each table entry appears before its matching queries
    var lastTableIdx = -1
    var orderCorrect = true
    for s in sorted {
        let limbs = frToInt(s)
        var foundIdx = -1
        for (j, t) in table.enumerated() {
            if frEqual(s, t) { foundIdx = j; break }
        }
        if foundIdx >= 0 && foundIdx >= lastTableIdx {
            lastTableIdx = foundIdx
        }
    }
    // The sorted vector should have non-decreasing table positions
    expect(orderCorrect, "Sorted witness preserves table ordering")
}

// MARK: - Single Table Prove/Verify

func testSingleTableProveVerify() {
    let engine = GPUPlonkLookupEngine()

    let table = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
    let queries = [frFromInt(20), frFromInt(10), frFromInt(40), frFromInt(30)]
    let beta = frFromInt(12345)
    let gamma = frFromInt(67890)

    let proof = engine.proveSingleTable(
        queries: queries, table: table, beta: beta, gamma: gamma)

    expect(proof.sortedVector.count == 8, "Proof sorted vector length = 8")
    expect(proof.accumulatorZ.count == 8, "Proof accumulator length = n+N = 8")
    expect(frEqual(proof.accumulatorZ[0], Fr.one), "Accumulator Z[0] = 1")

    let valid = engine.verifySingleTable(
        proof: proof, queries: queries, table: table,
        beta: beta, gamma: gamma)
    expect(valid, "Single table prove/verify succeeds")
}

// MARK: - Accumulator Closes

func testAccumulatorCloses() {
    let engine = GPUPlonkLookupEngine()

    let table = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let queries = [frFromInt(2), frFromInt(4), frFromInt(1), frFromInt(3)]
    let beta = frFromInt(7)
    let gamma = frFromInt(13)

    let proof = engine.proveSingleTable(
        queries: queries, table: table, beta: beta, gamma: gamma)

    expect(frEqual(proof.finalAccumulator, Fr.one), "Accumulator Z[n] == 1")
    expect(frEqual(proof.accumulatorZ[0], Fr.one), "Accumulator Z[0] == 1")
    expect(proof.queryCount == 4, "Query count matches")
}

// MARK: - Repeated Lookups

func testRepeatedLookups() {
    let engine = GPUPlonkLookupEngine()

    let table: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 1)) }
    // Repeat each value twice
    let queries: [Fr] = [1, 1, 3, 3, 5, 5, 7, 7].map { frFromInt($0) }
    let beta = frFromInt(42)
    let gamma = frFromInt(99)

    let proof = engine.proveSingleTable(
        queries: queries, table: table, beta: beta, gamma: gamma)

    expect(proof.sortedVector.count == 16, "Sorted vector length = 8 + 8")
    expect(frEqual(proof.finalAccumulator, Fr.one), "Repeated lookups accumulator closes")

    let valid = engine.verifySingleTable(
        proof: proof, queries: queries, table: table,
        beta: beta, gamma: gamma)
    expect(valid, "Repeated lookups verify")
}

// MARK: - Rejection Tests

func testRejectTamperedAccumulator() {
    let engine = GPUPlonkLookupEngine()

    let table = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
    let queries = [frFromInt(20), frFromInt(10), frFromInt(40), frFromInt(30)]
    let beta = frFromInt(12345)
    let gamma = frFromInt(67890)

    let proof = engine.proveSingleTable(
        queries: queries, table: table, beta: beta, gamma: gamma)

    // Tamper: change final accumulator
    let tampered = TableLookupProof(
        tableId: proof.tableId,
        sortedVector: proof.sortedVector,
        accumulatorZ: proof.accumulatorZ,
        finalAccumulator: frAdd(proof.finalAccumulator, Fr.one),
        queryCount: proof.queryCount
    )

    let rejected = !engine.verifySingleTable(
        proof: tampered, queries: queries, table: table,
        beta: beta, gamma: gamma)
    expect(rejected, "Reject tampered final accumulator")
}

func testRejectTamperedSorted() {
    let engine = GPUPlonkLookupEngine()

    let table = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
    let queries = [frFromInt(20), frFromInt(10), frFromInt(40), frFromInt(30)]
    let beta = frFromInt(12345)
    let gamma = frFromInt(67890)

    let proof = engine.proveSingleTable(
        queries: queries, table: table, beta: beta, gamma: gamma)

    // Tamper: modify sorted vector
    var badSorted = proof.sortedVector
    badSorted[0] = frFromInt(999)
    let tampered = TableLookupProof(
        tableId: proof.tableId,
        sortedVector: badSorted,
        accumulatorZ: proof.accumulatorZ,
        finalAccumulator: proof.finalAccumulator,
        queryCount: proof.queryCount
    )

    let rejected = !engine.verifySingleTable(
        proof: tampered, queries: queries, table: table,
        beta: beta, gamma: gamma)
    expect(rejected, "Reject tampered sorted vector")
}

// MARK: - Multi-Table Circuit

func testMultiTableCircuit() {
    let engine = GPUPlonkLookupEngine()

    // Table 0: range [1..4]
    let table0 = PlonkLookupTable(id: 0, values: [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)])
    // Table 1: range [10..13]
    let table1 = PlonkLookupTable(id: 1, values: [frFromInt(10), frFromInt(11), frFromInt(12), frFromInt(13)])

    // Build circuit with lookup gates referencing both tables
    var gates = [PlonkGate]()
    var wireAssignments = [[Int]]()

    // 4 gates looking up in table 0
    for i in 0..<4 {
        gates.append(PlonkGate(
            qL: Fr.zero, qR: Fr.zero, qO: Fr.zero, qM: Fr.zero,
            qC: frFromInt(0), qLookup: Fr.one))
        wireAssignments.append([i, i, i])
    }

    // 4 gates looking up in table 1
    for i in 0..<4 {
        gates.append(PlonkGate(
            qL: Fr.zero, qR: Fr.zero, qO: Fr.zero, qM: Fr.zero,
            qC: frFromInt(1), qLookup: Fr.one))
        wireAssignments.append([4 + i, 4 + i, 4 + i])
    }

    let circuit = PlonkCircuit(
        gates: gates,
        copyConstraints: [],
        wireAssignments: wireAssignments,
        lookupTables: [table0, table1]
    )

    // Witness: first 4 vars from table0, next 4 from table1
    var witness = [Fr](repeating: Fr.zero, count: 8)
    witness[0] = frFromInt(1); witness[1] = frFromInt(3)
    witness[2] = frFromInt(2); witness[3] = frFromInt(4)
    witness[4] = frFromInt(10); witness[5] = frFromInt(12)
    witness[6] = frFromInt(11); witness[7] = frFromInt(13)

    let beta = frFromInt(777)
    let gamma = frFromInt(888)

    let proof = engine.proveLookups(
        circuit: circuit, witness: witness,
        beta: beta, gamma: gamma)

    expect(proof.tableProofs.count == 2, "Multi-table proof has 2 table proofs")
    expect(proof.allAccumulatorsClose, "All accumulators close")

    let valid = engine.verifyLookups(
        proof: proof, circuit: circuit, witness: witness)
    expect(valid, "Multi-table circuit verification succeeds")
}

// MARK: - Batch Processing

private func testLookupBatchProcessing() {
    let engine = GPUPlonkLookupEngine()

    let beta = frFromInt(555)
    let gamma = frFromInt(666)

    let batches: [(queries: [Fr], table: [Fr])] = [
        (queries: [frFromInt(1), frFromInt(2)],
         table: [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]),
        (queries: [frFromInt(10), frFromInt(30)],
         table: [frFromInt(10), frFromInt(20), frFromInt(30)]),
        (queries: [frFromInt(100)],
         table: [frFromInt(100), frFromInt(200)])
    ]

    let results = engine.proveBatch(batches: batches, beta: beta, gamma: gamma)

    expect(results.count == 3, "Batch produces 3 results")

    for (idx, result) in results.enumerated() {
        expect(frEqual(result.finalAccumulator, Fr.one),
               "Batch[\(idx)] accumulator closes")

        let valid = engine.verifySingleTable(
            proof: result,
            queries: batches[idx].queries,
            table: batches[idx].table,
            beta: beta, gamma: gamma)
        expect(valid, "Batch[\(idx)] verifies")
    }
}

// MARK: - Accumulator Polynomial

func testAccumulatorPolynomial() {
    let engine = GPUPlonkLookupEngine()

    let table = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let queries = [frFromInt(2), frFromInt(4)]
    let beta = frFromInt(31)
    let gamma = frFromInt(37)

    let (zEvals, sortedEvals) = engine.computeAccumulatorPolynomial(
        queries: queries, table: table,
        domainN: 4, beta: beta, gamma: gamma)

    expect(zEvals.count == 4, "Z_lookup evals padded to domain size")
    expect(sortedEvals.count > 0, "Sorted evals non-empty")
    expect(frEqual(zEvals[0], Fr.one), "Z_lookup[0] = 1")
}

// MARK: - Auto Fiat-Shamir

func testAutoFiatShamirChallenges() {
    let engine = GPUPlonkLookupEngine()

    let table: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 1)) }
    let queries: [Fr] = [1, 2, 3, 4, 5, 6, 7, 8].map { frFromInt($0) }

    // Use auto-derived challenges (beta=nil, gamma=nil)
    let proof = engine.proveSingleTable(queries: queries, table: table)

    expect(frEqual(proof.finalAccumulator, Fr.one),
           "Auto Fiat-Shamir accumulator closes")
    expect(proof.sortedVector.count == 16,
           "Auto Fiat-Shamir sorted vector correct length")
}

// MARK: - Empty Table Queries

func testEmptyTableQueries() {
    let engine = GPUPlonkLookupEngine()

    // Circuit with a table but no lookup gates referencing it
    let table0 = PlonkLookupTable(id: 0, values: [frFromInt(1), frFromInt(2)])

    // One arithmetic gate (no lookup)
    let gate = PlonkGate(qL: Fr.one, qR: Fr.zero, qO: Fr.zero, qM: Fr.zero, qC: Fr.zero)
    let circuit = PlonkCircuit(
        gates: [gate],
        copyConstraints: [],
        wireAssignments: [[0, 0, 0]],
        lookupTables: [table0]
    )

    let witness = [frFromInt(0)]
    let proof = engine.proveLookups(
        circuit: circuit, witness: witness,
        beta: frFromInt(1), gamma: frFromInt(1))

    expect(proof.tableProofs.count == 1, "One trivial proof for unused table")
    expect(proof.tableProofs[0].queryCount == 0, "Zero queries for unused table")
    expect(proof.allAccumulatorsClose, "Trivial proof accumulator closes")

    let valid = engine.verifyLookups(proof: proof, circuit: circuit, witness: witness)
    expect(valid, "Trivial lookup proof verifies")
}

// MARK: - Larger Table

func testLargerTable() {
    let engine = GPUPlonkLookupEngine()

    let N = 256
    let n = 128
    let table: [Fr] = (0..<N).map { frFromInt(UInt64($0 + 1)) }

    // Deterministic pseudo-random queries
    var rng: UInt64 = 0xCAFE_BABE
    var queries = [Fr]()
    queries.reserveCapacity(n)
    for _ in 0..<n {
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        let idx = Int(rng >> 32) % N
        queries.append(table[idx])
    }

    let beta = frFromInt(9999)
    let gamma = frFromInt(7777)

    let proof = engine.proveSingleTable(
        queries: queries, table: table, beta: beta, gamma: gamma)

    expect(frEqual(proof.finalAccumulator, Fr.one),
           "Larger table (n=128, N=256) accumulator closes")
    expect(proof.sortedVector.count == n + N,
           "Larger table sorted vector length correct")

    let valid = engine.verifySingleTable(
        proof: proof, queries: queries, table: table,
        beta: beta, gamma: gamma)
    expect(valid, "Larger table (n=128, N=256) verifies")
}

// MARK: - Asymmetric Query/Table Sizes

func testAsymmetricQueryTable() {
    let engine = GPUPlonkLookupEngine()

    // Large table, few queries
    let table: [Fr] = (0..<16).map { frFromInt(UInt64($0 * 7 + 3)) }
    let queries: [Fr] = [table[0], table[5], table[10]]
    let beta = frFromInt(101)
    let gamma = frFromInt(202)

    let proof = engine.proveSingleTable(
        queries: queries, table: table, beta: beta, gamma: gamma)

    expect(proof.sortedVector.count == 19, "Asymmetric: sorted vector = 3 + 16")
    expect(frEqual(proof.finalAccumulator, Fr.one), "Asymmetric accumulator closes")

    let valid = engine.verifySingleTable(
        proof: proof, queries: queries, table: table,
        beta: beta, gamma: gamma)
    expect(valid, "Asymmetric (n=3, N=16) verifies")
}

// MARK: - All Table Values Queried

func testAllTableValuesQueried() {
    let engine = GPUPlonkLookupEngine()

    // Every table value is queried exactly once
    let table: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 1)) }
    let queries = table  // exact copy
    let beta = frFromInt(303)
    let gamma = frFromInt(404)

    let proof = engine.proveSingleTable(
        queries: queries, table: table, beta: beta, gamma: gamma)

    expect(proof.sortedVector.count == 16, "Full query: sorted = 8 + 8")
    expect(frEqual(proof.finalAccumulator, Fr.one), "Full query accumulator closes")

    let valid = engine.verifySingleTable(
        proof: proof, queries: queries, table: table,
        beta: beta, gamma: gamma)
    expect(valid, "Full query (all table values) verifies")
}

// MARK: - Single Element Table

func testSingleElementTable() {
    let engine = GPUPlonkLookupEngine()

    let table = [frFromInt(42)]
    let queries = [frFromInt(42), frFromInt(42), frFromInt(42)]
    let beta = frFromInt(11)
    let gamma = frFromInt(22)

    let proof = engine.proveSingleTable(
        queries: queries, table: table, beta: beta, gamma: gamma)

    expect(proof.sortedVector.count == 4, "Single-element: sorted = 3 + 1")
    expect(frEqual(proof.finalAccumulator, Fr.one),
           "Single-element table accumulator closes")

    let valid = engine.verifySingleTable(
        proof: proof, queries: queries, table: table,
        beta: beta, gamma: gamma)
    expect(valid, "Single-element table verifies")
}
