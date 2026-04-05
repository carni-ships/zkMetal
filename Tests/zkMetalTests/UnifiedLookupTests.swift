// Tests for UnifiedLookup engine and LookupTable

import Foundation
import zkMetal

func runUnifiedLookupTests() {
    suite("UnifiedLookup")

    // --- LookupTable tests ---

    testRangeTable()
    testXORTable()
    testANDTable()
    testByteDecompTable()
    testCrossProduct()
    testCompressedTable()
    testCustomTable()

    // --- UnifiedLookup engine tests ---

    testStrategySelection()
    testLogUpProveVerify()
    testLassoProveVerify()
    testBatchProve()
}

// MARK: - LookupTable Tests

func testRangeTable() {
    // Small range table
    let t8 = RangeTable(bits: 8)
    expect(t8.count == 256, "Range(8) has 256 entries")
    expect(t8.name == "Range(8)", "Range(8) name")
    expect(frEqual(t8.values[0], Fr.zero), "Range(8)[0] == 0")
    expect(frEqual(t8.values[255], frFromInt(255)), "Range(8)[255] == 255")

    // 16-bit range table
    let t16 = RangeTable(bits: 16)
    expect(t16.count == 65536, "Range(16) has 65536 entries")
    expect(t16.isStructured, "Range(16) is structured")
    expect(t16.lassoTable != nil, "Range(16) has Lasso table")

    // Small range table not structured
    let t4 = RangeTable(bits: 4)
    expect(t4.count == 16, "Range(4) has 16 entries")
    expect(!t4.isStructured, "Range(4) is not structured")
}

func testXORTable() {
    let t = XORTable(bits: 4)
    expect(t.count == 256, "XOR(4) has 256 entries (16*16)")
    expect(t.name == "XOR(4)", "XOR(4) name")
    // XOR(4)[3*16 + 5] = 3 XOR 5 = 6
    expect(frEqual(t.values[3 * 16 + 5], frFromInt(6)), "XOR(4): 3 XOR 5 = 6")
    // XOR(4)[7*16 + 7] = 7 XOR 7 = 0
    expect(frEqual(t.values[7 * 16 + 7], Fr.zero), "XOR(4): 7 XOR 7 = 0")

    let t8 = XORTable(bits: 8)
    expect(t8.isStructured, "XOR(8) is structured")
    expect(t8.lassoTable != nil, "XOR(8) has Lasso table")
}

func testANDTable() {
    let t = ANDTable(bits: 4)
    expect(t.count == 256, "AND(4) has 256 entries")
    // AND(4)[3*16 + 5] = 3 AND 5 = 1
    expect(frEqual(t.values[3 * 16 + 5], frFromInt(1)), "AND(4): 3 AND 5 = 1")
    // AND(4)[15*16 + 15] = 15 AND 15 = 15
    expect(frEqual(t.values[15 * 16 + 15], frFromInt(15)), "AND(4): 15 AND 15 = 15")
}

func testByteDecompTable() {
    let t = ByteDecompTable
    expect(t.count == 256, "ByteDecomp has 256 entries")
    expect(t.name == "ByteDecomp", "ByteDecomp name")
    expect(frEqual(t.values[0], Fr.zero), "ByteDecomp[0] == 0")
    expect(frEqual(t.values[128], frFromInt(128)), "ByteDecomp[128] == 128")
}

func testCrossProduct() {
    let tA = tableFromValues(name: "A", [0, 1, 2, 3])
    let tB = tableFromValues(name: "B", [10, 20, 30, 40])
    let cp = crossProductTable(tA, tB)
    expect(cp.count == 16, "Cross product has 16 entries")
    expect(cp.name == "A x B", "Cross product name")
    expect(cp.numColumns == 2, "Cross product has 2 columns")
    // Entry (0, 0): 0*4 + 10 = 10
    expect(frEqual(cp.values[0], frFromInt(10)), "CrossProduct[0,0] = 0*4 + 10")
    // Entry (1, 2): 1*4 + 30 = 34 (shift = sizeB = 4)
    expect(frEqual(cp.values[1 * 4 + 2], frFromInt(34)), "CrossProduct[1,2] = 1*4 + 30")
}

func testCompressedTable() {
    let entries: [(index: UInt64, value: Fr)] = [
        (0, frFromInt(42)),
        (100, frFromInt(99)),
        (1000, frFromInt(7)),
    ]
    let ct = CompressedLookupTable(name: "Sparse", entries: entries, logicalSize: 2048)
    expect(ct.density < 0.01, "Sparse table density < 1%")

    let materialized = ct.materialize()
    expect(materialized.count == 2048, "Materialized has 2048 entries")
    expect(frEqual(materialized[0], frFromInt(42)), "Materialized[0] == 42")
    expect(frEqual(materialized[100], frFromInt(99)), "Materialized[100] == 99")
    expect(frEqual(materialized[1000], frFromInt(7)), "Materialized[1000] == 7")
    expect(frEqual(materialized[500], Fr.zero), "Materialized[500] == 0 (default)")
}

func testCustomTable() {
    let t = customTable(name: "Squares", size: 16) { i in
        frFromInt(UInt64(i * i))
    }
    expect(t.count == 16, "Squares table has 16 entries")
    expect(frEqual(t.values[4], frFromInt(16)), "Squares[4] == 16")
    expect(frEqual(t.values[10], frFromInt(100)), "Squares[10] == 100")
}

// MARK: - Strategy Selection Tests

func testStrategySelection() {
    let engine = UnifiedLookupEngine()

    // Small unstructured table -> LogUp
    let small = tableFromValues(name: "Small", [0, 1, 2, 3, 4, 5, 6, 7])
    let s1 = engine.selectStrategy(table: small, numLookups: 4)
    expect(s1 == .logUp, "Small unstructured -> LogUp")

    // Structured table with Lasso -> Lasso
    let range16 = RangeTable(bits: 16)
    let s2 = engine.selectStrategy(table: range16, numLookups: 1024)
    expect(s2 == .lasso, "Range(16) structured -> Lasso")

    // With cq preprocessed -> cq
    let s3 = engine.selectStrategy(table: small, numLookups: 4,
                                    cqPreprocessed: nil)
    expect(s3 == .logUp, "No cq preprocessing -> not cq")
}

// MARK: - Prove/Verify Tests

func testLogUpProveVerify() {
    do {
        let engine = try UnifiedLookupEngine()
        let table = RangeTable(bits: 4)  // [0..15], not structured (bits < 8)

        // Lookup values: all in range [0, 15]
        let lookups: [Fr] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15].map { frFromInt($0) }

        let proof = try engine.prove(lookups: lookups, table: table, strategy: .logUp)
        expect(proof.strategy == .logUp, "LogUp strategy used")
        expect(proof.numLookups == 16, "16 lookups")
        expect(proof.logUpProof != nil, "LogUp proof present")

        let valid = try engine.verify(proof: proof, lookups: lookups, table: table)
        expect(valid, "LogUp proof verifies")
    } catch {
        expect(false, "LogUp prove/verify threw: \(error)")
    }
}

func testLassoProveVerify() {
    do {
        let engine = try UnifiedLookupEngine()
        let table = RangeTable(bits: 16)  // Structured, has Lasso decomposition

        // 256 random lookups in [0, 65535]
        var lookups = [Fr]()
        for i in 0..<256 {
            lookups.append(frFromInt(UInt64(i * 251 % 65536)))
        }

        let proof = try engine.prove(lookups: lookups, table: table, strategy: .lasso)
        expect(proof.strategy == .lasso, "Lasso strategy used")
        expect(proof.lassoProof != nil, "Lasso proof present")

        let valid = try engine.verify(proof: proof, lookups: lookups, table: table)
        expect(valid, "Lasso proof verifies")
    } catch {
        expect(false, "Lasso prove/verify threw: \(error)")
    }
}

func testBatchProve() {
    do {
        let engine = try UnifiedLookupEngine()
        let table = RangeTable(bits: 4)

        let batch1: [Fr] = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7].map { frFromInt($0) }
        let batch2: [Fr] = [8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15].map { frFromInt($0) }

        let proofs = try engine.proveBatch(batches: [batch1, batch2],
                                            table: table, strategy: .logUp)
        expect(proofs.count == 2, "Batch produced 2 proofs")

        for (i, proof) in proofs.enumerated() {
            let lookups = i == 0 ? batch1 : batch2
            let valid = try engine.verify(proof: proof, lookups: lookups, table: table)
            expect(valid, "Batch proof \(i) verifies")
        }
    } catch {
        expect(false, "Batch prove threw: \(error)")
    }
}
