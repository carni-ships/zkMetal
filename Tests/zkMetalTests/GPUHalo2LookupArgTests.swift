// GPUHalo2LookupArgTests — Tests for GPU-accelerated Halo2 lookup argument engine
//
// Tests cover:
//   1. LookupTable construction and containment checks
//   2. Single-column permuted column generation
//   3. Product argument construction and verification
//   4. Multi-column lookup with RLC compression
//   5. Batch lookup across multiple tables
//   6. Edge cases: duplicates, full-table usage, single-row tables
//   7. GPU-accelerated sorting correctness
//   8. Statistics and diagnostics
//   9. Expression evaluation pipeline integration
//  10. Negative tests: invalid inputs, failing lookups

import zkMetal
import Foundation

public func runGPUHalo2LookupArgTests() {
    suite("GPU Halo2 Lookup Arg Engine")
    testLookupTableConstruction()
    testLookupTableContainsTuple()
    testSingleColumnContainment()
    testContainmentWithFailures()
    testPermutedColumnGeneration()
    testPermutedColumnsWithDuplicates()
    testProductArgumentBasic()
    testProductArgumentVerification()
    testProductVerificationRejectsInvalid()
    testSortFieldElements()
    testSortFieldElementsLarger()
    testExpressionEvaluation()
    testLookupExpressionEvaluationRLC()
    testSingleLookupEndToEnd()
    testSingleLookupRejectsInvalid()
    testBatchLookup()
    testBatchLookupPartialFailure()
    testBuildTableFromStore()
    testCompressRLC()
    testLookupStatistics()
    testSingleRowTable()
    testIdentityLookup()
    testAllSameInputs()
    testPermutedColumnConstraints()
    testHalo2LargerDomain()
}

// MARK: - Test: LookupTable construction

private func testLookupTableConstruction() {
    let values: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let table = LookupTable(name: "test", values: values)

    expectEqual(table.numRows, 4, "table has 4 rows")
    expectEqual(table.numColumns, 1, "single-column table")
    expectEqual(table.name, "test", "table name correct")

    expect(table.contains(column: 0, value: frFromInt(1)), "table contains 1")
    expect(table.contains(column: 0, value: frFromInt(4)), "table contains 4")
    expect(!table.contains(column: 0, value: frFromInt(5)), "table does not contain 5")
    expect(!table.contains(column: 0, value: frFromInt(0)), "table does not contain 0")
}

// MARK: - Test: Multi-column tuple containment

private func testLookupTableContainsTuple() {
    let col0: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3)]
    let col1: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30)]
    let table = LookupTable(name: "multi", tableValues: [col0, col1])

    expectEqual(table.numRows, 3, "multi-col table has 3 rows")
    expectEqual(table.numColumns, 2, "multi-col table has 2 columns")

    expect(table.containsTuple([frFromInt(1), frFromInt(10)]), "tuple (1,10) found")
    expect(table.containsTuple([frFromInt(2), frFromInt(20)]), "tuple (2,20) found")
    expect(table.containsTuple([frFromInt(3), frFromInt(30)]), "tuple (3,30) found")
    expect(!table.containsTuple([frFromInt(1), frFromInt(20)]), "tuple (1,20) not found")
    expect(!table.containsTuple([frFromInt(4), frFromInt(40)]), "tuple (4,40) not found")
    expect(!table.containsTuple([frFromInt(1)]), "wrong tuple width rejected")
}

// MARK: - Test: Single-column containment check

private func testSingleColumnContainment() {
    let engine = GPUHalo2LookupArgEngine()

    let table: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
    let inputs: [Fr] = [frFromInt(10), frFromInt(30), frFromInt(20), frFromInt(40)]

    let failing = engine.checkContainment(inputs: inputs, table: table)
    expect(failing.isEmpty, "all inputs found in table")
}

// MARK: - Test: Containment with failures

private func testContainmentWithFailures() {
    let engine = GPUHalo2LookupArgEngine()

    let table: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let inputs: [Fr] = [frFromInt(1), frFromInt(99), frFromInt(3), frFromInt(100)]

    let failing = engine.checkContainment(inputs: inputs, table: table)
    expectEqual(failing.count, 2, "two inputs not in table")
    expect(failing.contains(1), "row 1 fails (99 not in table)")
    expect(failing.contains(3), "row 3 fails (100 not in table)")
}

// MARK: - Test: Permuted column generation

private func testPermutedColumnGeneration() {
    let engine = GPUHalo2LookupArgEngine()

    // Table: [1, 2, 3, 4], Input: [3, 1, 2, 4] (a permutation)
    let table: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let inputs: [Fr] = [frFromInt(3), frFromInt(1), frFromInt(2), frFromInt(4)]

    do {
        let permuted = try engine.generatePermutedColumns(inputs: inputs, table: table)
        expectEqual(permuted.domainSize, 4, "permuted domain size correct")
        expectEqual(permuted.permutedInput.count, 4, "A' has 4 elements")
        expectEqual(permuted.permutedTable.count, 4, "S' has 4 elements")

        // A' should be sorted
        let aPrime = permuted.permutedInput
        for i in 1..<4 {
            let prev = aPrime[i - 1].to64()
            let cur = aPrime[i].to64()
            var prevLeq = false
            for j in stride(from: 3, through: 0, by: -1) {
                if prev[j] != cur[j] {
                    prevLeq = prev[j] < cur[j]
                    break
                }
                if j == 0 { prevLeq = true }  // equal
            }
            expect(prevLeq, "A' is sorted at index \(i)")
        }

        // Check permuted column constraint: A'[i] == S'[i] or A'[i] == A'[i-1]
        expect(frEqual(aPrime[0], permuted.permutedTable[0]),
               "A'[0] == S'[0]")
        for i in 1..<4 {
            let matchTable = frEqual(aPrime[i], permuted.permutedTable[i])
            let matchPrev = frEqual(aPrime[i], aPrime[i - 1])
            expect(matchTable || matchPrev,
                   "permuted constraint holds at row \(i)")
        }
    } catch {
        expect(false, "permuted column generation should not throw: \(error)")
    }
}

// MARK: - Test: Permuted columns with duplicate inputs

private func testPermutedColumnsWithDuplicates() {
    let engine = GPUHalo2LookupArgEngine()

    // Table: [1, 2, 3, 4], Input: [2, 2, 1, 1] (duplicates)
    let table: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let inputs: [Fr] = [frFromInt(2), frFromInt(2), frFromInt(1), frFromInt(1)]

    do {
        let permuted = try engine.generatePermutedColumns(inputs: inputs, table: table)
        expectEqual(permuted.domainSize, 4, "domain size with duplicates")

        // A' should be sorted: [1, 1, 2, 2]
        let aPrime = permuted.permutedInput
        expect(frEqual(aPrime[0], frFromInt(1)), "A'[0] = 1")
        expect(frEqual(aPrime[1], frFromInt(1)), "A'[1] = 1")
        expect(frEqual(aPrime[2], frFromInt(2)), "A'[2] = 2")
        expect(frEqual(aPrime[3], frFromInt(2)), "A'[3] = 2")

        // Check permuted constraints
        expect(frEqual(aPrime[0], permuted.permutedTable[0]), "A'[0] == S'[0]")
        for i in 1..<4 {
            let matchT = frEqual(aPrime[i], permuted.permutedTable[i])
            let matchP = frEqual(aPrime[i], aPrime[i - 1])
            expect(matchT || matchP, "permuted constraint with dups at row \(i)")
        }
    } catch {
        expect(false, "permuted column generation with dups should not throw: \(error)")
    }
}

// MARK: - Test: Product argument construction

private func testProductArgumentBasic() {
    let engine = GPUHalo2LookupArgEngine()

    let table: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let inputs: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]

    do {
        let permuted = try engine.generatePermutedColumns(inputs: inputs, table: table)
        let beta = frFromInt(7)
        let gamma = frFromInt(13)

        let product = engine.buildLookupProduct(
            inputs: inputs, table: table,
            permuted: permuted, beta: beta, gamma: gamma
        )

        expectEqual(product.zEvals.count, 4, "Z_L has n evaluations")
        expect(frEqual(product.zEvals[0], Fr.one), "Z_L(omega^0) = 1")
        expect(product.isValid, "product telescopes for identity permutation")
    } catch {
        expect(false, "product argument should not throw: \(error)")
    }
}

// MARK: - Test: Product argument verification

private func testProductArgumentVerification() {
    let engine = GPUHalo2LookupArgEngine()

    let table: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let inputs: [Fr] = [frFromInt(3), frFromInt(1), frFromInt(4), frFromInt(2)]
    let beta = frFromInt(11)
    let gamma = frFromInt(17)

    do {
        let permuted = try engine.generatePermutedColumns(inputs: inputs, table: table)
        let product = engine.buildLookupProduct(
            inputs: inputs, table: table,
            permuted: permuted, beta: beta, gamma: gamma
        )

        let valid = engine.verifyLookupProduct(product, inputs: inputs, table: table)
        expect(valid, "product verification passes for valid lookup")
    } catch {
        expect(false, "verification should not throw: \(error)")
    }
}

// MARK: - Test: Product verification rejects tampered product

private func testProductVerificationRejectsInvalid() {
    let engine = GPUHalo2LookupArgEngine()

    let table: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let inputs: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let beta = frFromInt(5)
    let gamma = frFromInt(9)

    do {
        let permuted = try engine.generatePermutedColumns(inputs: inputs, table: table)
        let product = engine.buildLookupProduct(
            inputs: inputs, table: table,
            permuted: permuted, beta: beta, gamma: gamma
        )

        // Tamper with Z_L evaluations
        var tamperedZ = product.zEvals
        tamperedZ[1] = frFromInt(999)
        let tampered = LookupProduct(
            zEvals: tamperedZ,
            permuted: permuted,
            beta: beta, gamma: gamma,
            isValid: product.isValid
        )

        let valid = engine.verifyLookupProduct(tampered, inputs: inputs, table: table)
        expect(!valid, "tampered product rejected by verifier")
    } catch {
        expect(false, "should not throw: \(error)")
    }
}

// MARK: - Test: Sort field elements

private func testSortFieldElements() {
    let engine = GPUHalo2LookupArgEngine()

    let values: [Fr] = [frFromInt(5), frFromInt(3), frFromInt(8), frFromInt(1)]
    let sorted = engine.sortFieldElements(values)

    expectEqual(sorted.count, 4, "sorted array has same count")

    // Verify all original values are present
    for v in values {
        expect(sorted.contains(where: { frEqual($0, v) }), "sorted contains \(v)")
    }

    // Verify to64() ordering is non-decreasing (Montgomery-form sort)
    for i in 1..<sorted.count {
        let prev = sorted[i - 1].to64()
        let cur = sorted[i].to64()
        var ok = true
        for j in stride(from: 3, through: 0, by: -1) {
            if prev[j] != cur[j] {
                ok = prev[j] <= cur[j]
                break
            }
        }
        expect(ok, "sorted order maintained at index \(i)")
    }
}

// MARK: - Test: Sort field elements (larger set)

private func testSortFieldElementsLarger() {
    let engine = GPUHalo2LookupArgEngine()

    // Create 16 random-ish elements
    var values = [Fr]()
    for i in stride(from: 32, through: 2, by: -2) {
        values.append(frFromInt(UInt64(i)))
    }

    let sorted = engine.sortFieldElements(values)
    expectEqual(sorted.count, values.count, "sorted count matches input")

    // Verify sorted order
    for i in 1..<sorted.count {
        let prev = sorted[i - 1].to64()
        let cur = sorted[i].to64()
        var ok = true
        for j in stride(from: 3, through: 0, by: -1) {
            if prev[j] != cur[j] {
                ok = prev[j] <= cur[j]
                break
            }
        }
        expect(ok, "sorted order maintained at index \(i)")
    }
}

// MARK: - Test: Expression evaluation

private func testExpressionEvaluation() {
    let engine = GPUHalo2LookupArgEngine()

    // Build a store with one advice column and one fixed column
    let advEvals: [[Fr]] = [[frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]]
    let fixEvals: [[Fr]] = [[frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]]
    let store = Halo2ColumnStore(
        adviceEvals: advEvals,
        fixedEvals: fixEvals,
        instanceEvals: [],
        selectorEvals: [],
        domainSize: 4
    )

    let advCol = Halo2Column(.advice, 0)
    let fixCol = Halo2Column(.fixed, 0)

    // Expression: advice[0] * fixed[0]
    let expr: Halo2Expression = .product(
        .advice(advCol, .cur),
        .fixed(fixCol, .cur)
    )

    let results = engine.evaluateExpressions([expr], store: store)
    expectEqual(results.count, 1, "one expression evaluated")
    expectEqual(results[0].count, 4, "4 rows evaluated")

    // Row 0: 10 * 1 = 10
    expect(frEqual(results[0][0], frFromInt(10)), "expr eval row 0: 10*1=10")
    // Row 1: 20 * 2 = 40
    expect(frEqual(results[0][1], frFromInt(40)), "expr eval row 1: 20*2=40")
    // Row 2: 30 * 3 = 90
    expect(frEqual(results[0][2], frFromInt(90)), "expr eval row 2: 30*3=90")
    // Row 3: 40 * 4 = 160
    expect(frEqual(results[0][3], frFromInt(160)), "expr eval row 3: 40*4=160")
}

// MARK: - Test: Lookup expression evaluation with RLC

private func testLookupExpressionEvaluationRLC() {
    let engine = GPUHalo2LookupArgEngine()

    // Two advice columns, two fixed columns (for 2-column lookup)
    let adv0: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let adv1: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
    let fix0: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let fix1: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]

    let store = Halo2ColumnStore(
        adviceEvals: [adv0, adv1],
        fixedEvals: [fix0, fix1],
        instanceEvals: [],
        selectorEvals: [],
        domainSize: 4
    )

    let advCol0 = Halo2Column(.advice, 0)
    let advCol1 = Halo2Column(.advice, 1)
    let fixCol0 = Halo2Column(.fixed, 0)
    let fixCol1 = Halo2Column(.fixed, 1)

    let lookup = Halo2Lookup(
        name: "two_col",
        inputExpressions: [
            .advice(advCol0, .cur),
            .advice(advCol1, .cur),
        ],
        tableExpressions: [
            .fixed(fixCol0, .cur),
            .fixed(fixCol1, .cur),
        ]
    )

    let theta = frFromInt(3)
    let (inputs, table) = engine.evaluateLookupExpressions(
        lookup: lookup, store: store, theta: theta
    )

    expectEqual(inputs.count, 4, "compressed inputs has 4 rows")
    expectEqual(table.count, 4, "compressed table has 4 rows")

    // Row 0: compressed = adv0[0] * theta^0 + adv1[0] * theta^1 = 1 + 10*3 = 31
    expect(frEqual(inputs[0], frFromInt(31)), "RLC input row 0: 1 + 10*3 = 31")
    // Row 0 table: 1 + 10*3 = 31
    expect(frEqual(table[0], frFromInt(31)), "RLC table row 0: 1 + 10*3 = 31")
    // inputs == table for this matching case
    for i in 0..<4 {
        expect(frEqual(inputs[i], table[i]), "RLC compressed match at row \(i)")
    }
}

// MARK: - Test: Single lookup end-to-end

private func testSingleLookupEndToEnd() {
    let engine = GPUHalo2LookupArgEngine()

    let table: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
    let inputs: [Fr] = [frFromInt(20), frFromInt(40), frFromInt(10), frFromInt(30)]
    let beta = frFromInt(7)
    let gamma = frFromInt(13)

    let valid = engine.evaluateSingleLookup(
        inputs: inputs, table: table, beta: beta, gamma: gamma
    )
    expect(valid, "single lookup end-to-end passes for valid inputs")
}

// MARK: - Test: Single lookup rejects invalid

private func testSingleLookupRejectsInvalid() {
    let engine = GPUHalo2LookupArgEngine()

    let table: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let inputs: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(99)]
    let beta = frFromInt(7)
    let gamma = frFromInt(13)

    let valid = engine.evaluateSingleLookup(
        inputs: inputs, table: table, beta: beta, gamma: gamma
    )
    expect(!valid, "single lookup rejects input not in table")
}

// MARK: - Test: Batch lookup across multiple tables

private func testBatchLookup() {
    let engine = GPUHalo2LookupArgEngine()

    // Set up CS with two lookups
    let cs = Halo2ConstraintSystem()
    let advA = cs.adviceColumn()
    let advB = cs.adviceColumn()
    let fixT1 = cs.fixedColumn()
    let fixT2 = cs.fixedColumn()

    // Lookup 1: advA in fixT1
    cs.lookup("range1") { vc in
        let input = vc.queryAdvice(advA, at: .cur)
        let table = vc.queryFixed(fixT1, at: .cur)
        return [(input, table)]
    }

    // Lookup 2: advB in fixT2
    cs.lookup("range2") { vc in
        let input = vc.queryAdvice(advB, at: .cur)
        let table = vc.queryFixed(fixT2, at: .cur)
        return [(input, table)]
    }

    let assignment = Halo2Assignment(
        numAdvice: 2, numFixed: 2, numInstance: 0,
        numSelectors: 0, numRows: 4
    )

    // Table 1: {1, 2, 3, 4}
    for i in 0..<4 {
        assignment.setFixed(column: 0, row: i, value: frFromInt(UInt64(i + 1)))
    }
    // Table 2: {10, 20, 30, 40}
    for i in 0..<4 {
        assignment.setFixed(column: 1, row: i, value: frFromInt(UInt64((i + 1) * 10)))
    }

    // Inputs: advA = [2, 3, 1, 4], advB = [30, 10, 40, 20]
    assignment.setAdvice(column: 0, row: 0, value: frFromInt(2))
    assignment.setAdvice(column: 0, row: 1, value: frFromInt(3))
    assignment.setAdvice(column: 0, row: 2, value: frFromInt(1))
    assignment.setAdvice(column: 0, row: 3, value: frFromInt(4))

    assignment.setAdvice(column: 1, row: 0, value: frFromInt(30))
    assignment.setAdvice(column: 1, row: 1, value: frFromInt(10))
    assignment.setAdvice(column: 1, row: 2, value: frFromInt(40))
    assignment.setAdvice(column: 1, row: 3, value: frFromInt(20))

    let store = Halo2ColumnStore(assignment: assignment, targetDomainSize: 4)

    let theta = frFromInt(3)
    let beta = frFromInt(7)
    let gamma = frFromInt(13)

    let result = engine.batchLookup(
        cs: cs, store: store, theta: theta, beta: beta, gamma: gamma
    )

    expectEqual(result.perLookupValid.count, 2, "batch: two lookups evaluated")
    expect(result.perLookupValid[0], "batch: lookup 1 passes")
    expect(result.perLookupValid[1], "batch: lookup 2 passes")
    expect(result.isValid, "batch: overall valid")
}

// MARK: - Test: Batch lookup with partial failure

private func testBatchLookupPartialFailure() {
    let engine = GPUHalo2LookupArgEngine()

    let cs = Halo2ConstraintSystem()
    let advA = cs.adviceColumn()
    let advB = cs.adviceColumn()
    let fixT1 = cs.fixedColumn()
    let fixT2 = cs.fixedColumn()

    cs.lookup("good") { vc in
        [(vc.queryAdvice(advA, at: .cur), vc.queryFixed(fixT1, at: .cur))]
    }
    cs.lookup("bad") { vc in
        [(vc.queryAdvice(advB, at: .cur), vc.queryFixed(fixT2, at: .cur))]
    }

    let assignment = Halo2Assignment(
        numAdvice: 2, numFixed: 2, numInstance: 0,
        numSelectors: 0, numRows: 4
    )

    // Table 1: {1, 2, 3, 4}
    for i in 0..<4 {
        assignment.setFixed(column: 0, row: i, value: frFromInt(UInt64(i + 1)))
    }
    // Table 2: {10, 20, 30, 40}
    for i in 0..<4 {
        assignment.setFixed(column: 1, row: i, value: frFromInt(UInt64((i + 1) * 10)))
    }

    // advA valid: [1, 2, 3, 4]
    for i in 0..<4 {
        assignment.setAdvice(column: 0, row: i, value: frFromInt(UInt64(i + 1)))
    }
    // advB invalid: [10, 20, 30, 999] -- 999 not in table 2
    assignment.setAdvice(column: 1, row: 0, value: frFromInt(10))
    assignment.setAdvice(column: 1, row: 1, value: frFromInt(20))
    assignment.setAdvice(column: 1, row: 2, value: frFromInt(30))
    assignment.setAdvice(column: 1, row: 3, value: frFromInt(999))

    let store = Halo2ColumnStore(assignment: assignment, targetDomainSize: 4)

    let result = engine.batchLookup(
        cs: cs, store: store,
        theta: frFromInt(3), beta: frFromInt(7), gamma: frFromInt(13)
    )

    expect(result.perLookupValid[0], "batch partial: good lookup passes")
    expect(!result.perLookupValid[1], "batch partial: bad lookup fails")
    expect(!result.isValid, "batch partial: overall invalid")
    expect(result.products[1] == nil, "batch partial: no product for failed lookup")
}

// MARK: - Test: Build table from store

private func testBuildTableFromStore() {
    let engine = GPUHalo2LookupArgEngine()

    let fix0: [Fr] = [frFromInt(100), frFromInt(200), frFromInt(300), frFromInt(400)]
    let fix1: [Fr] = [frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8)]
    let store = Halo2ColumnStore(
        adviceEvals: [],
        fixedEvals: [fix0, fix1],
        instanceEvals: [],
        selectorEvals: [],
        domainSize: 4
    )

    let table = engine.buildTable(
        name: "from_store",
        columnIndices: [0, 1],
        store: store,
        activeRows: 3
    )

    expectEqual(table.numRows, 3, "table from store has 3 active rows")
    expectEqual(table.numColumns, 2, "table from store has 2 columns")
    expect(frEqual(table.tableValues[0][0], frFromInt(100)), "table col0 row0 correct")
    expect(frEqual(table.tableValues[1][2], frFromInt(7)), "table col1 row2 correct")
}

// MARK: - Test: Compress RLC

private func testCompressRLC() {
    let engine = GPUHalo2LookupArgEngine()

    let values: [Fr] = [frFromInt(2), frFromInt(3), frFromInt(5)]
    let theta = frFromInt(10)

    // Expected: 2 * 10^0 + 3 * 10^1 + 5 * 10^2 = 2 + 30 + 500 = 532
    let compressed = engine.compressRLC(values: values, theta: theta)
    expect(frEqual(compressed, frFromInt(532)), "RLC compression: 2 + 3*10 + 5*100 = 532")
}

// MARK: - Test: Lookup statistics

private func testLookupStatistics() {
    let engine = GPUHalo2LookupArgEngine()

    let table: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let inputs: [Fr] = [frFromInt(1), frFromInt(1), frFromInt(2), frFromInt(2)]

    let stats = engine.lookupStatistics(inputs: inputs, table: table)

    expectEqual(stats["inputCount"], 4, "stats: 4 inputs")
    expectEqual(stats["tableCount"], 4, "stats: 4 table entries")
    expectEqual(stats["uniqueInputs"], 2, "stats: 2 unique inputs")
    expectEqual(stats["uniqueTableEntries"], 4, "stats: 4 unique table entries")
    expectEqual(stats["maxMultiplicity"], 2, "stats: max multiplicity is 2")
}

// MARK: - Test: Single row table

private func testSingleRowTable() {
    let engine = GPUHalo2LookupArgEngine()

    let table: [Fr] = [frFromInt(42)]
    let inputs: [Fr] = [frFromInt(42)]
    let beta = frFromInt(5)
    let gamma = frFromInt(11)

    let valid = engine.evaluateSingleLookup(
        inputs: inputs, table: table, beta: beta, gamma: gamma
    )
    expect(valid, "single row table: lookup passes")
}

// MARK: - Test: Identity lookup (inputs == table)

private func testIdentityLookup() {
    let engine = GPUHalo2LookupArgEngine()

    let values: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4),
                        frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8)]
    let beta = frFromInt(19)
    let gamma = frFromInt(23)

    let valid = engine.evaluateSingleLookup(
        inputs: values, table: values, beta: beta, gamma: gamma
    )
    expect(valid, "identity lookup: inputs == table passes")
}

// MARK: - Test: All same inputs

private func testAllSameInputs() {
    let engine = GPUHalo2LookupArgEngine()

    // All inputs are the same value, which appears in the table
    let table: [Fr] = [frFromInt(7), frFromInt(8), frFromInt(9), frFromInt(10)]
    let inputs: [Fr] = [frFromInt(7), frFromInt(7), frFromInt(7), frFromInt(7)]
    let beta = frFromInt(3)
    let gamma = frFromInt(5)

    let valid = engine.evaluateSingleLookup(
        inputs: inputs, table: table, beta: beta, gamma: gamma
    )
    expect(valid, "all same inputs: lookup passes when value is in table")
}

// MARK: - Test: Permuted column constraints explicitly

private func testPermutedColumnConstraints() {
    let engine = GPUHalo2LookupArgEngine()

    let table: [Fr] = [frFromInt(5), frFromInt(10), frFromInt(15), frFromInt(20)]
    let inputs: [Fr] = [frFromInt(10), frFromInt(5), frFromInt(20), frFromInt(15)]

    do {
        let permuted = try engine.generatePermutedColumns(inputs: inputs, table: table)
        let aPrime = permuted.permutedInput
        let sPrime = permuted.permutedTable

        // Constraint 1: A'[0] == S'[0]
        expect(frEqual(aPrime[0], sPrime[0]),
               "explicit permuted: A'[0] == S'[0]")

        // Constraint 2: for i > 0, A'[i] == S'[i] or A'[i] == A'[i-1]
        var allHold = true
        for i in 1..<4 {
            let c1 = frEqual(aPrime[i], sPrime[i])
            let c2 = frEqual(aPrime[i], aPrime[i - 1])
            if !c1 && !c2 {
                allHold = false
            }
        }
        expect(allHold, "explicit permuted: all row constraints hold")

        // Constraint 3: A' is sorted
        for i in 1..<4 {
            let prev = aPrime[i - 1].to64()
            let cur = aPrime[i].to64()
            var leq = true
            for j in stride(from: 3, through: 0, by: -1) {
                if prev[j] != cur[j] {
                    leq = prev[j] <= cur[j]
                    break
                }
            }
            expect(leq, "explicit permuted: A' sorted at \(i)")
        }
    } catch {
        expect(false, "permuted constraints should not throw: \(error)")
    }
}

// MARK: - Test: Larger domain (8 elements)

private func testHalo2LargerDomain() {
    let engine = GPUHalo2LookupArgEngine()

    // 8-element table and input
    var table = [Fr]()
    for i in 1...8 {
        table.append(frFromInt(UInt64(i * 100)))
    }

    // Input: reverse order
    var inputs = [Fr]()
    for i in stride(from: 8, through: 1, by: -1) {
        inputs.append(frFromInt(UInt64(i * 100)))
    }

    let beta = frFromInt(31)
    let gamma = frFromInt(37)

    // Containment check
    let failing = engine.checkContainment(inputs: inputs, table: table)
    expect(failing.isEmpty, "large domain: all inputs in table")

    // Full end-to-end
    let valid = engine.evaluateSingleLookup(
        inputs: inputs, table: table, beta: beta, gamma: gamma
    )
    expect(valid, "large domain: 8-element lookup passes")

    // Statistics
    let stats = engine.lookupStatistics(inputs: inputs, table: table)
    expectEqual(stats["uniqueInputs"], 8, "large domain: 8 unique inputs")
    expectEqual(stats["maxMultiplicity"], 1, "large domain: each input appears once")
}
