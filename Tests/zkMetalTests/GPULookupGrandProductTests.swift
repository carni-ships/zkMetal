// Tests for GPULookupGrandProductEngine -- lookup grand product argument

import Foundation
import zkMetal

public func runGPULookupGrandProductTests() {
    suite("GPULookupGrandProduct")

    testSingleColumnSimple()
    testSingleColumnRepeated()
    testSingleColumnAsymmetric()
    testAccumulatorBoundary()
    testHelperColumn()
    testLGPRejectTamperedAccumulator()
    testLGPRejectTamperedSorted()
    testAutoChallenge()
    testProductCheckDirect()
    testTransitionConstraints()
    testMultiColumnTwoCol()
    testMultiColumnThreeCol()
    testMultiColumnSingleColFallback()
    testMultiplicitiesCorrectness()
    testSortedVectorOrdering()
    testLargerSingleColumn()
    testLargerMultiColumn()
    testVerificationDiagnostics()
    testBatchVerify()
    testCompressColumnsIdentity()
    testCompressColumnsLinear()
    testRejectBadMultisetMerge()
    testHelperColumnEdgeCases()
    testProofStructureFields()
    testConfigOptions()
}

// MARK: - Single-Column: Basic Prove/Verify

private func testSingleColumnSimple() {
    do {
        let engine = try GPULookupGrandProductEngine()
        let table = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
        let witness = [frFromInt(20), frFromInt(10), frFromInt(40), frFromInt(30)]
        let beta = frFromInt(12345)
        let gamma = frFromInt(67890)

        let proof = try engine.prove(witness: witness, table: table,
                                      beta: beta, gamma: gamma)
        let result = engine.verify(proof: proof, witness: witness, table: table)
        expect(result.valid, "Single-column simple (n=4, N=4)")
    } catch {
        expect(false, "Single-column simple threw: \(error)")
    }
}

private func testSingleColumnRepeated() {
    do {
        let engine = try GPULookupGrandProductEngine()
        let table: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 1)) }
        let witness: [Fr] = [1, 1, 3, 3, 5, 5, 7, 7].map { frFromInt($0) }

        let proof = try engine.prove(witness: witness, table: table)
        let result = engine.verify(proof: proof, witness: witness, table: table)
        expect(result.valid, "Repeated lookups (n=8, N=8)")
    } catch {
        expect(false, "Repeated lookups threw: \(error)")
    }
}

private func testSingleColumnAsymmetric() {
    do {
        let engine = try GPULookupGrandProductEngine()
        let table: [Fr] = (0..<16).map { frFromInt(UInt64($0 * 7 + 3)) }
        let witness: [Fr] = (0..<4).map { table[$0] }

        let proof = try engine.prove(witness: witness, table: table)
        let result = engine.verify(proof: proof, witness: witness, table: table)
        expect(result.valid, "Asymmetric (n=4, N=16)")
    } catch {
        expect(false, "Asymmetric threw: \(error)")
    }
}

// MARK: - Accumulator Boundary Constraints

private func testAccumulatorBoundary() {
    do {
        let engine = try GPULookupGrandProductEngine()
        let table = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let witness = [frFromInt(2), frFromInt(4), frFromInt(1), frFromInt(3)]

        let proof = try engine.prove(witness: witness, table: table)

        // z[0] = 1
        expect(frEqual(proof.accumulatorZ[0], Fr.one), "z[0] == 1")
        // z[n] = 1 (accumulator closes)
        expect(frEqual(proof.finalAccumulator, Fr.one), "z[n] == 1")
        // sorted vector length = n + N
        expectEqual(proof.sortedVector.count, 8, "Sorted vector length == n + N")
        // accumulator length = n + 1
        expectEqual(proof.accumulatorZ.count, 5, "Accumulator length == D + 1")
    } catch {
        expect(false, "AccumulatorBoundary threw: \(error)")
    }
}

// MARK: - Helper Column

private func testHelperColumn() {
    do {
        let config = LookupGrandProductConfig(numColumns: 1, computeHelper: true)
        let engine = try GPULookupGrandProductEngine(config: config)
        let table = [frFromInt(1), frFromInt(2), frFromInt(3)]
        // Witness with repeats: two 1s, two 2s, one 3
        let witness = [frFromInt(1), frFromInt(1), frFromInt(2), frFromInt(2), frFromInt(3)]

        let proof = try engine.prove(witness: witness, table: table)

        // Helper should exist
        expect(proof.helperH != nil, "Helper column computed")

        if let h = proof.helperH {
            // Length should match sorted vector
            expectEqual(h.count, proof.sortedVector.count, "Helper length == sorted length")

            // h[0] should always be 1
            expect(frEqual(h[0], Fr.one), "h[0] == 1")

            // Verify helper correctness against sorted vector
            let s = proof.sortedVector
            for i in 1..<s.count {
                let expected = frEqual(s[i], s[i - 1]) ? Fr.zero : Fr.one
                expect(frEqual(h[i], expected), "h[\(i)] correct")
            }
        }

        // Proof should still verify
        let result = engine.verify(proof: proof, witness: witness, table: table)
        expect(result.valid, "Proof with helper column verifies")
    } catch {
        expect(false, "HelperColumn threw: \(error)")
    }
}

// MARK: - Tamper Detection

private func testLGPRejectTamperedAccumulator() {
    do {
        let engine = try GPULookupGrandProductEngine()
        let table = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
        let witness = [frFromInt(20), frFromInt(10), frFromInt(40), frFromInt(30)]

        let proof = try engine.prove(witness: witness, table: table)

        // Tamper: change finalAccumulator
        let tampered = LookupGrandProductProof(
            compressedWitness: proof.compressedWitness,
            compressedTable: proof.compressedTable,
            sortedVector: proof.sortedVector,
            accumulatorZ: proof.accumulatorZ,
            helperH: proof.helperH,
            alpha: proof.alpha,
            beta: proof.beta,
            gamma: proof.gamma,
            finalAccumulator: frAdd(proof.finalAccumulator, Fr.one),
            multiplicities: proof.multiplicities
        )
        let result = engine.verify(proof: tampered, witness: witness, table: table)
        expect(!result.valid, "Reject tampered accumulator")
        expect(result.failedCheck != nil, "Has failure reason")
    } catch {
        expect(false, "RejectTamperedAccumulator threw: \(error)")
    }
}

private func testLGPRejectTamperedSorted() {
    do {
        let engine = try GPULookupGrandProductEngine()
        let table = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
        let witness = [frFromInt(20), frFromInt(10), frFromInt(40), frFromInt(30)]

        let proof = try engine.prove(witness: witness, table: table)

        // Tamper: change first sorted element
        var badSorted = proof.sortedVector
        badSorted[0] = frFromInt(999)
        let tampered = LookupGrandProductProof(
            compressedWitness: proof.compressedWitness,
            compressedTable: proof.compressedTable,
            sortedVector: badSorted,
            accumulatorZ: proof.accumulatorZ,
            helperH: proof.helperH,
            alpha: proof.alpha,
            beta: proof.beta,
            gamma: proof.gamma,
            finalAccumulator: proof.finalAccumulator,
            multiplicities: proof.multiplicities
        )
        let result = engine.verify(proof: tampered, witness: witness, table: table)
        expect(!result.valid, "Reject tampered sorted vector")
    } catch {
        expect(false, "RejectTamperedSorted threw: \(error)")
    }
}

// MARK: - Auto Fiat-Shamir Challenges

private func testAutoChallenge() {
    do {
        let engine = try GPULookupGrandProductEngine()
        let table: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 1)) }
        let witness: [Fr] = [1, 2, 3, 4, 5, 6, 7, 8].map { frFromInt($0) }

        let proof = try engine.prove(witness: witness, table: table)
        let result = engine.verify(proof: proof, witness: witness, table: table)
        expect(result.valid, "Auto Fiat-Shamir challenges verify")

        // Challenges should be non-zero
        expect(!proof.alpha.isZero, "Alpha is non-zero")
        expect(!proof.beta.isZero, "Beta is non-zero")
        expect(!proof.gamma.isZero, "Gamma is non-zero")

        // Challenges should be deterministic: same input -> same challenges
        let proof2 = try engine.prove(witness: witness, table: table)
        expect(frEqual(proof.alpha, proof2.alpha), "Alpha is deterministic")
        expect(frEqual(proof.beta, proof2.beta), "Beta is deterministic")
        expect(frEqual(proof.gamma, proof2.gamma), "Gamma is deterministic")
    } catch {
        expect(false, "AutoChallenge threw: \(error)")
    }
}

// MARK: - Product Check

private func testProductCheckDirect() {
    do {
        let engine = try GPULookupGrandProductEngine()
        let table = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let witness = [frFromInt(3), frFromInt(1), frFromInt(4), frFromInt(2)]

        let proof = try engine.prove(witness: witness, table: table)

        // Direct product check: cumulative product of Num/Den should be 1
        let productOk = engine.verifyProductCheck(
            witness: proof.compressedWitness,
            table: proof.compressedTable,
            sorted: proof.sortedVector,
            beta: proof.beta, gamma: proof.gamma)
        expect(productOk, "Direct product check passes")
    } catch {
        expect(false, "ProductCheckDirect threw: \(error)")
    }
}

// MARK: - Transition Constraints

private func testTransitionConstraints() {
    do {
        let engine = try GPULookupGrandProductEngine()
        let table = [frFromInt(5), frFromInt(10), frFromInt(15), frFromInt(20)]
        let witness = [frFromInt(10), frFromInt(5), frFromInt(20), frFromInt(15)]

        let proof = try engine.prove(witness: witness, table: table)

        // Every transition constraint should evaluate to zero
        let transitions = engine.evaluateAllTransitions(
            proof: proof,
            witness: proof.compressedWitness,
            table: proof.compressedTable)
        expectEqual(transitions.count, max(witness.count, table.count), "Transition count == D")

        for i in 0..<transitions.count {
            expect(frEqual(transitions[i], Fr.zero),
                   "Transition constraint \(i) == 0")
        }

        // Also test individual evaluation
        let t0 = engine.evaluateTransitionConstraint(
            proof: proof,
            witness: proof.compressedWitness,
            table: proof.compressedTable,
            index: 0)
        expect(frEqual(t0, Fr.zero), "Transition[0] individually == 0")
    } catch {
        expect(false, "TransitionConstraints threw: \(error)")
    }
}

// MARK: - Multi-Column Lookups

private func testMultiColumnTwoCol() {
    do {
        let engine = try GPULookupGrandProductEngine(
            config: LookupGrandProductConfig(numColumns: 2))

        // Table: (1,10), (2,20), (3,30), (4,40)
        let tableCol0: [Fr] = [1, 2, 3, 4].map { frFromInt($0) }
        let tableCol1: [Fr] = [10, 20, 30, 40].map { frFromInt($0) }

        // Witness: (2,20), (4,40), (1,10), (3,30) -- all in table
        let witCol0: [Fr] = [2, 4, 1, 3].map { frFromInt($0) }
        let witCol1: [Fr] = [20, 40, 10, 30].map { frFromInt($0) }

        let proof = try engine.proveMultiColumn(
            witnessColumns: [witCol0, witCol1],
            tableColumns: [tableCol0, tableCol1])

        let result = engine.verify(proof: proof,
                                    witnessColumns: [witCol0, witCol1],
                                    tableColumns: [tableCol0, tableCol1])
        expect(result.valid, "Two-column lookup verifies")
        expect(frEqual(proof.finalAccumulator, Fr.one), "Two-column accumulator closes")
    } catch {
        expect(false, "MultiColumnTwoCol threw: \(error)")
    }
}

private func testMultiColumnThreeCol() {
    do {
        let engine = try GPULookupGrandProductEngine(
            config: LookupGrandProductConfig(numColumns: 3))

        // Table: (a, b, c) = (1,2,3), (4,5,6), (7,8,9), (10,11,12)
        let tC0: [Fr] = [1, 4, 7, 10].map { frFromInt($0) }
        let tC1: [Fr] = [2, 5, 8, 11].map { frFromInt($0) }
        let tC2: [Fr] = [3, 6, 9, 12].map { frFromInt($0) }

        // Witness: rows (4,5,6), (1,2,3), (10,11,12), (7,8,9)
        let wC0: [Fr] = [4, 1, 10, 7].map { frFromInt($0) }
        let wC1: [Fr] = [5, 2, 11, 8].map { frFromInt($0) }
        let wC2: [Fr] = [6, 3, 12, 9].map { frFromInt($0) }

        let proof = try engine.proveMultiColumn(
            witnessColumns: [wC0, wC1, wC2],
            tableColumns: [tC0, tC1, tC2])

        let result = engine.verify(proof: proof,
                                    witnessColumns: [wC0, wC1, wC2],
                                    tableColumns: [tC0, tC1, tC2])
        expect(result.valid, "Three-column lookup verifies")
    } catch {
        expect(false, "MultiColumnThreeCol threw: \(error)")
    }
}

private func testMultiColumnSingleColFallback() {
    do {
        // Multi-column engine with numColumns=1 should behave same as single-column
        let engine = try GPULookupGrandProductEngine()
        let table: [Fr] = [1, 2, 3, 4].map { frFromInt($0) }
        let witness: [Fr] = [3, 1, 4, 2].map { frFromInt($0) }

        let beta = frFromInt(7777)
        let gamma = frFromInt(8888)

        let proof1 = try engine.prove(witness: witness, table: table,
                                       beta: beta, gamma: gamma)
        let proof2 = try engine.proveMultiColumn(
            witnessColumns: [witness], tableColumns: [table],
            alpha: nil, beta: beta, gamma: gamma)

        // Both should produce valid proofs
        let r1 = engine.verify(proof: proof1, witness: witness, table: table)
        let r2 = engine.verify(proof: proof2,
                                witnessColumns: [witness],
                                tableColumns: [table])
        expect(r1.valid, "Single-col via prove() verifies")
        expect(r2.valid, "Single-col via proveMultiColumn() verifies")

        // Compressed witness should equal raw witness (identity RLC for 1 col)
        for i in 0..<witness.count {
            expect(frEqual(proof1.compressedWitness[i], witness[i]),
                   "Compressed == raw for 1 col at \(i)")
        }
    } catch {
        expect(false, "MultiColumnSingleColFallback threw: \(error)")
    }
}

// MARK: - Multiplicities

private func testMultiplicitiesCorrectness() {
    do {
        let engine = try GPULookupGrandProductEngine()
        let table: [Fr] = [10, 20, 30, 40].map { frFromInt($0) }
        // Witness: 10 appears 3x, 20 appears 1x, 30 appears 2x, 40 appears 0x
        let witness: [Fr] = [10, 10, 10, 20, 30, 30].map { frFromInt($0) }

        let proof = try engine.prove(witness: witness, table: table)

        expectEqual(proof.multiplicities.count, 4, "Multiplicities length == N")
        expect(frEqual(proof.multiplicities[0], frFromInt(3)), "mult[0] == 3 (for 10)")
        expect(frEqual(proof.multiplicities[1], frFromInt(1)), "mult[1] == 1 (for 20)")
        expect(frEqual(proof.multiplicities[2], frFromInt(2)), "mult[2] == 2 (for 30)")
        expect(frEqual(proof.multiplicities[3], Fr.zero), "mult[3] == 0 (for 40)")

        let result = engine.verify(proof: proof, witness: witness, table: table)
        expect(result.valid, "Multiplicities proof verifies")
    } catch {
        expect(false, "MultiplicitiesCorrectness threw: \(error)")
    }
}

// MARK: - Sorted Vector Ordering

private func testSortedVectorOrdering() {
    do {
        let engine = try GPULookupGrandProductEngine()
        let table: [Fr] = [100, 200, 300].map { frFromInt($0) }
        let witness: [Fr] = [200, 100, 300, 200].map { frFromInt($0) }

        let proof = try engine.prove(witness: witness, table: table)
        let s = proof.sortedVector

        // Length should be 2*D = 2*max(4,3) = 8 (padded for Plookup)
        expectEqual(s.count, 8, "Sorted vector length == 8")

        // The sorted vector should group identical values together:
        // e.g., all 100s together, all 200s together, all 300s together
        var runs = [[Fr]]()
        var currentRun = [s[0]]
        for i in 1..<s.count {
            if frEqual(s[i], s[i - 1]) {
                currentRun.append(s[i])
            } else {
                runs.append(currentRun)
                currentRun = [s[i]]
            }
        }
        runs.append(currentRun)

        // Each run should contain table entry + witness copies
        // 100: 1 table + 1 witness = 2
        // 200: 1 table + 2 witness = 3
        // 300: 1 table + 1 witness = 2
        expectEqual(runs.count, 3, "3 distinct runs in sorted vector")
    } catch {
        expect(false, "SortedVectorOrdering threw: \(error)")
    }
}

// MARK: - Larger Tests

private func testLargerSingleColumn() {
    do {
        let engine = try GPULookupGrandProductEngine()
        let N = 256
        let n = 128
        let table: [Fr] = (0..<N).map { frFromInt(UInt64($0 + 1)) }

        var rng: UInt64 = 0xDEAD_BEEF
        var witness = [Fr]()
        witness.reserveCapacity(n)
        for _ in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let idx = Int(rng >> 32) % N
            witness.append(table[idx])
        }

        let proof = try engine.prove(witness: witness, table: table)
        let result = engine.verify(proof: proof, witness: witness, table: table)
        expect(result.valid, "Larger single-col (n=128, N=256)")
        expect(frEqual(proof.finalAccumulator, Fr.one), "Larger accumulator closes")
        expectEqual(proof.sortedVector.count, 2 * max(n, N), "Sorted length == 2*D")
    } catch {
        expect(false, "LargerSingleColumn threw: \(error)")
    }
}

private func testLargerMultiColumn() {
    do {
        let engine = try GPULookupGrandProductEngine(
            config: LookupGrandProductConfig(numColumns: 2))
        let N = 64
        let n = 32

        let tableCol0: [Fr] = (0..<N).map { frFromInt(UInt64($0 + 1)) }
        let tableCol1: [Fr] = (0..<N).map { frFromInt(UInt64($0 * 3 + 7)) }

        var rng: UInt64 = 0xCAFE_1234
        var witCol0 = [Fr]()
        var witCol1 = [Fr]()
        witCol0.reserveCapacity(n)
        witCol1.reserveCapacity(n)
        for _ in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let idx = Int(rng >> 32) % N
            witCol0.append(tableCol0[idx])
            witCol1.append(tableCol1[idx])
        }

        let proof = try engine.proveMultiColumn(
            witnessColumns: [witCol0, witCol1],
            tableColumns: [tableCol0, tableCol1])
        let result = engine.verify(proof: proof,
                                    witnessColumns: [witCol0, witCol1],
                                    tableColumns: [tableCol0, tableCol1])
        expect(result.valid, "Larger multi-col (n=32, N=64, c=2)")
    } catch {
        expect(false, "LargerMultiColumn threw: \(error)")
    }
}

// MARK: - Verification Diagnostics

private func testVerificationDiagnostics() {
    do {
        let engine = try GPULookupGrandProductEngine()
        let table = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let witness = [frFromInt(2), frFromInt(3)]

        let proof = try engine.prove(witness: witness, table: table)

        // Valid proof: no failure info
        let good = engine.verify(proof: proof, witness: witness, table: table)
        expect(good.valid, "Good proof is valid")
        expect(good.failedCheck == nil, "Good proof has no failure reason")
        expectEqual(good.failingIndex, -1, "Good proof has no failing index")

        // Wrong table: should fail
        let badTable = [frFromInt(99), frFromInt(100)]
        let bad = engine.verify(proof: proof, witness: witness, table: badTable)
        expect(!bad.valid, "Wrong table rejected")
        expect(bad.failedCheck != nil, "Has failure description")

        // Wrong column count
        let badCols = engine.verify(proof: proof,
                                     witnessColumns: [[frFromInt(1)], [frFromInt(2)]],
                                     tableColumns: [[frFromInt(1)]])
        expect(!badCols.valid, "Column count mismatch rejected")
    } catch {
        expect(false, "VerificationDiagnostics threw: \(error)")
    }
}

// MARK: - Batch Verify

private func testBatchVerify() {
    do {
        let engine = try GPULookupGrandProductEngine()

        let table1: [Fr] = [1, 2, 3, 4].map { frFromInt($0) }
        let witness1: [Fr] = [2, 3].map { frFromInt($0) }
        let proof1 = try engine.prove(witness: witness1, table: table1)

        let table2: [Fr] = [10, 20, 30].map { frFromInt($0) }
        let witness2: [Fr] = [10, 30, 20].map { frFromInt($0) }
        let proof2 = try engine.prove(witness: witness2, table: table2)

        let results = engine.batchVerify(proofs: [
            (proof: proof1, witnessColumns: [witness1], tableColumns: [table1]),
            (proof: proof2, witnessColumns: [witness2], tableColumns: [table2])
        ])

        expectEqual(results.count, 2, "Batch verify returns 2 results")
        expect(results[0].valid, "Batch proof 1 valid")
        expect(results[1].valid, "Batch proof 2 valid")
    } catch {
        expect(false, "BatchVerify threw: \(error)")
    }
}

// MARK: - Column Compression

private func testCompressColumnsIdentity() {
    do {
        let engine = try GPULookupGrandProductEngine()
        let col: [Fr] = [1, 2, 3, 4].map { frFromInt($0) }
        let alpha = frFromInt(999)

        // Single column: compression should be identity regardless of alpha
        let compressed = engine.compressColumns([col], alpha: alpha, count: 4)
        expectEqual(compressed.count, 4, "Compressed length == 4")
        for i in 0..<4 {
            expect(frEqual(compressed[i], col[i]),
                   "Single-col compression is identity at \(i)")
        }
    } catch {
        expect(false, "CompressColumnsIdentity threw: \(error)")
    }
}

private func testCompressColumnsLinear() {
    do {
        let engine = try GPULookupGrandProductEngine()
        let col0: [Fr] = [frFromInt(1), frFromInt(2)]
        let col1: [Fr] = [frFromInt(10), frFromInt(20)]
        let alpha = frFromInt(3)

        // compressed[i] = col0[i] + alpha * col1[i]
        let compressed = engine.compressColumns([col0, col1], alpha: alpha, count: 2)

        // compressed[0] = 1 + 3*10 = 31
        // compressed[1] = 2 + 3*20 = 62
        let expected0 = frAdd(frFromInt(1), frMul(frFromInt(3), frFromInt(10)))
        let expected1 = frAdd(frFromInt(2), frMul(frFromInt(3), frFromInt(20)))

        expect(frEqual(compressed[0], expected0), "RLC[0] = 1 + 3*10")
        expect(frEqual(compressed[1], expected1), "RLC[1] = 2 + 3*20")
    } catch {
        expect(false, "CompressColumnsLinear threw: \(error)")
    }
}

// MARK: - Reject Bad Multiset Merge

private func testRejectBadMultisetMerge() {
    do {
        let engine = try GPULookupGrandProductEngine()
        let table = [frFromInt(1), frFromInt(2), frFromInt(3)]
        let witness = [frFromInt(1), frFromInt(3)]

        let proof = try engine.prove(witness: witness, table: table)

        // Tamper: replace sorted vector with garbage
        var badSorted = [Fr](repeating: frFromInt(42), count: proof.sortedVector.count)
        badSorted[0] = frFromInt(1)
        let tampered = LookupGrandProductProof(
            compressedWitness: proof.compressedWitness,
            compressedTable: proof.compressedTable,
            sortedVector: badSorted,
            accumulatorZ: proof.accumulatorZ,
            helperH: proof.helperH,
            alpha: proof.alpha,
            beta: proof.beta,
            gamma: proof.gamma,
            finalAccumulator: proof.finalAccumulator,
            multiplicities: proof.multiplicities
        )
        let result = engine.verify(proof: tampered, witness: witness, table: table)
        expect(!result.valid, "Reject invalid multiset merge")
    } catch {
        expect(false, "RejectBadMultisetMerge threw: \(error)")
    }
}

// MARK: - Helper Column Edge Cases

private func testHelperColumnEdgeCases() {
    do {
        // All witness elements are the same (maximum repetition)
        let engine = try GPULookupGrandProductEngine(
            config: LookupGrandProductConfig(numColumns: 1, computeHelper: true))
        let table = [frFromInt(5), frFromInt(10), frFromInt(15)]
        let witness: [Fr] = [5, 5, 5, 5].map { frFromInt($0) }

        let proof = try engine.prove(witness: witness, table: table)
        let result = engine.verify(proof: proof, witness: witness, table: table)
        expect(result.valid, "All-same witness verifies")

        if let h = proof.helperH {
            // Count transitions: only changes at distinct value boundaries
            var transitions = 0
            for i in 0..<h.count {
                if frEqual(h[i], Fr.one) { transitions += 1 }
            }
            // Should have at least 1 transition (the first element)
            expect(transitions >= 1, "At least 1 transition in helper")
        }

        // Config with computeHelper=false should not include helper
        let engine2 = try GPULookupGrandProductEngine(
            config: LookupGrandProductConfig(numColumns: 1, computeHelper: false))
        let proof2 = try engine2.prove(witness: witness, table: table)
        expect(proof2.helperH == nil, "No helper when computeHelper=false")
        let result2 = engine2.verify(proof: proof2, witness: witness, table: table)
        expect(result2.valid, "Proof without helper verifies")
    } catch {
        expect(false, "HelperColumnEdgeCases threw: \(error)")
    }
}

// MARK: - Proof Structure Fields

private func testProofStructureFields() {
    do {
        let engine = try GPULookupGrandProductEngine()
        let table: [Fr] = [1, 2, 3].map { frFromInt($0) }
        let witness: [Fr] = [2, 1, 3].map { frFromInt($0) }
        let beta = frFromInt(111)
        let gamma = frFromInt(222)

        let proof = try engine.prove(witness: witness, table: table,
                                      beta: beta, gamma: gamma)

        // Check all expected fields are populated
        expectEqual(proof.compressedWitness.count, 3, "compressedWitness length")
        expectEqual(proof.compressedTable.count, 3, "compressedTable length")
        expectEqual(proof.sortedVector.count, 6, "sortedVector length")
        expectEqual(proof.accumulatorZ.count, 4, "accumulatorZ length (D+1)")
        expectEqual(proof.multiplicities.count, 3, "multiplicities length")

        // Challenges should match what was passed
        expect(frEqual(proof.beta, beta), "Beta matches input")
        expect(frEqual(proof.gamma, gamma), "Gamma matches input")

        // Transition arrays (if present) should have length D
        if let nums = proof.transitionNumerators {
            expectEqual(nums.count, 3, "transitionNumerators length")
        }
        if let dens = proof.transitionDenominators {
            expectEqual(dens.count, 3, "transitionDenominators length")
        }
    } catch {
        expect(false, "ProofStructureFields threw: \(error)")
    }
}

// MARK: - Config Options

private func testConfigOptions() {
    do {
        // Default config
        let c1 = LookupGrandProductConfig()
        expectEqual(c1.numColumns, 1, "Default numColumns == 1")
        expect(c1.computeHelper, "Default computeHelper == true")
        expect(!c1.evaluationForm, "Default evaluationForm == false")

        // Custom config
        let c2 = LookupGrandProductConfig(numColumns: 3,
                                            computeHelper: false,
                                            evaluationForm: true)
        expectEqual(c2.numColumns, 3, "Custom numColumns == 3")
        expect(!c2.computeHelper, "Custom computeHelper == false")
        expect(c2.evaluationForm, "Custom evaluationForm == true")

        // Engine with custom config should work
        let engine = try GPULookupGrandProductEngine(config: c2)
        let table: [Fr] = [1, 2, 3, 4].map { frFromInt($0) }
        let witness: [Fr] = [2, 4].map { frFromInt($0) }
        let proof = try engine.prove(witness: witness, table: table)
        let result = engine.verify(proof: proof, witness: witness, table: table)
        expect(result.valid, "Engine with custom config produces valid proof")
    } catch {
        expect(false, "ConfigOptions threw: \(error)")
    }
}
