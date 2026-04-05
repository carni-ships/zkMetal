import zkMetal
import Foundation

public func runLogUpGKRTests() {
    suite("LogUp-GKR Lookup Argument")

    // Helper: compare two Fr values
    func frEqual(_ a: Fr, _ b: Fr) -> Bool {
        return a.v.0 == b.v.0 && a.v.1 == b.v.1 && a.v.2 == b.v.2 && a.v.3 == b.v.3 &&
               a.v.4 == b.v.4 && a.v.5 == b.v.5 && a.v.6 == b.v.6 && a.v.7 == b.v.7
    }

    // Helper: deterministic pseudo-random values in range [0, bound)
    func pseudoRandomValues(_ n: Int, bound: Int, seed: UInt64 = 0xDEAD_BEEF_CAFE_1234) -> [UInt64] {
        var rng = seed
        return (0..<n).map { _ in
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            return (rng >> 32) % UInt64(bound)
        }
    }

    let prover = LogUpGKRProver()
    let verifier = LogUpGKRVerifier()

    // =========================================================================
    // Test 1: Simple lookup -- 8-element witness into 16-element table
    // =========================================================================
    do {
        let table = (0..<16).map { frFromInt(UInt64($0)) }
        let witness = [0, 3, 7, 1, 15, 2, 10, 5].map { frFromInt(UInt64($0)) }

        let proverTranscript = Transcript(label: "logup-gkr-test-1")
        let proof = prover.prove(witness: witness, table: table, transcript: proverTranscript)

        let verifierTranscript = Transcript(label: "logup-gkr-test-1")
        let valid = verifier.verify(witness: witness, table: table, proof: proof, transcript: verifierTranscript)
        expect(valid, "Simple lookup: 8 elements into 16-element table")
    }

    // =========================================================================
    // Test 2: Multi-column lookup (2 columns)
    // =========================================================================
    do {
        // Table: pairs (a, b) where a in 0..3, b in 0..3 => 16 rows
        var tableCol0 = [Fr]()
        var tableCol1 = [Fr]()
        for a in 0..<4 {
            for b in 0..<4 {
                tableCol0.append(frFromInt(UInt64(a)))
                tableCol1.append(frFromInt(UInt64(b)))
            }
        }

        // Witness: 8 rows, all valid pairs
        let witCol0 = [0, 1, 2, 3, 0, 1, 2, 3].map { frFromInt(UInt64($0)) }
        let witCol1 = [0, 1, 2, 3, 3, 2, 1, 0].map { frFromInt(UInt64($0)) }

        let proverT = Transcript(label: "logup-gkr-test-2")
        let proof = prover.proveMultiColumn(
            witness: [witCol0, witCol1],
            table: [tableCol0, tableCol1],
            transcript: proverT
        )

        let verifierT = Transcript(label: "logup-gkr-test-2")
        let valid = verifier.verifyMultiColumn(
            witness: [witCol0, witCol1],
            table: [tableCol0, tableCol1],
            proof: proof,
            transcript: verifierT
        )
        expect(valid, "Multi-column lookup (2 columns): 8 rows into 16-row table")
    }

    // =========================================================================
    // Test 3: Multi-column lookup (3 columns)
    // =========================================================================
    do {
        // Table: triples (a, b, c) for a,b,c in 0..1 => 8 rows
        var tCol0 = [Fr](), tCol1 = [Fr](), tCol2 = [Fr]()
        for a in 0..<2 {
            for b in 0..<2 {
                for c in 0..<2 {
                    tCol0.append(frFromInt(UInt64(a)))
                    tCol1.append(frFromInt(UInt64(b)))
                    tCol2.append(frFromInt(UInt64(c)))
                }
            }
        }

        // Witness: 4 rows
        let wCol0 = [0, 1, 0, 1].map { frFromInt(UInt64($0)) }
        let wCol1 = [0, 0, 1, 1].map { frFromInt(UInt64($0)) }
        let wCol2 = [0, 1, 1, 0].map { frFromInt(UInt64($0)) }

        let proverT = Transcript(label: "logup-gkr-test-3")
        let proof = prover.proveMultiColumn(
            witness: [wCol0, wCol1, wCol2],
            table: [tCol0, tCol1, tCol2],
            transcript: proverT
        )

        let verifierT = Transcript(label: "logup-gkr-test-3")
        let valid = verifier.verifyMultiColumn(
            witness: [wCol0, wCol1, wCol2],
            table: [tCol0, tCol1, tCol2],
            proof: proof,
            transcript: verifierT
        )
        expect(valid, "Multi-column lookup (3 columns): 4 rows into 8-row table")
    }

    // =========================================================================
    // Test 4: Multiplicity tracking (same element looked up multiple times)
    // =========================================================================
    do {
        let table = (0..<8).map { frFromInt(UInt64($0)) }
        // Witness has repeated elements: 0 appears 3 times, 5 appears 2 times, etc.
        let witness = [0, 0, 0, 5, 5, 3, 7, 1].map { frFromInt(UInt64($0)) }

        let proverT = Transcript(label: "logup-gkr-test-4")
        let proof = prover.prove(witness: witness, table: table, transcript: proverT)

        // Verify multiplicities are correct
        let m = proof.multiplicities
        expect(frEqual(m[0], frFromInt(3)), "Multiplicity of 0 should be 3")
        expect(frEqual(m[1], frFromInt(1)), "Multiplicity of 1 should be 1")
        expect(frEqual(m[3], frFromInt(1)), "Multiplicity of 3 should be 1")
        expect(frEqual(m[5], frFromInt(2)), "Multiplicity of 5 should be 2")
        expect(frEqual(m[7], frFromInt(1)), "Multiplicity of 7 should be 1")
        // Unused entries should have zero multiplicity
        expect(frEqual(m[2], frFromInt(0)), "Multiplicity of 2 should be 0")
        expect(frEqual(m[4], frFromInt(0)), "Multiplicity of 4 should be 0")
        expect(frEqual(m[6], frFromInt(0)), "Multiplicity of 6 should be 0")

        let verifierT = Transcript(label: "logup-gkr-test-4")
        let valid = verifier.verify(witness: witness, table: table, proof: proof, transcript: verifierT)
        expect(valid, "Multiplicity tracking: repeated lookups")
    }

    // =========================================================================
    // Test 5: Invalid lookup detection (witness element not in table)
    // =========================================================================
    do {
        let table = (0..<8).map { frFromInt(UInt64($0)) }
        // Witness has element 100, which is NOT in table
        let witness = [0, 1, 2, 100].map { frFromInt(UInt64($0)) }

        let proverT = Transcript(label: "logup-gkr-test-5")
        let proof = prover.prove(witness: witness, table: table, transcript: proverT)

        let verifierT = Transcript(label: "logup-gkr-test-5")
        let valid = verifier.verify(witness: witness, table: table, proof: proof, transcript: verifierT)
        expect(!valid, "Invalid lookup: element 100 not in table should fail verification")
    }

    // =========================================================================
    // Test 6: Batch lookup with 3 tables
    // =========================================================================
    do {
        let batchProver = LogUpBatchGKRProver()
        let batchVerifier = LogUpBatchGKRVerifier()

        // Instance 0: range check table [0..15], witness [0,1,2,3]
        let inst0 = LogUpBatchInstance(
            witness: [0, 1, 2, 3].map { frFromInt(UInt64($0)) },
            table: (0..<16).map { frFromInt(UInt64($0)) }
        )

        // Instance 1: small table [10, 20, 30, 40], witness [10, 30, 20, 40]
        let inst1 = LogUpBatchInstance(
            witness: [10, 30, 20, 40].map { frFromInt(UInt64($0)) },
            table: [10, 20, 30, 40].map { frFromInt(UInt64($0)) }
        )

        // Instance 2: powers of 2 table, witness uses some
        let inst2 = LogUpBatchInstance(
            witness: [1, 2, 4, 8, 16, 32, 64, 128].map { frFromInt(UInt64($0)) },
            table: (0..<8).map { frFromInt(UInt64(1 << $0)) }
        )

        let instances = [inst0, inst1, inst2]

        let proverT = Transcript(label: "logup-batch-test-6")
        let batchProof = batchProver.prove(instances: instances, transcript: proverT)

        expectEqual(batchProof.batchSize, 3, "Batch size should be 3")

        let verifierT = Transcript(label: "logup-batch-test-6")
        let valid = batchVerifier.verify(instances: instances, proof: batchProof, transcript: verifierT)
        expect(valid, "Batch lookup: 3 tables verified together")
    }

    // =========================================================================
    // Test 7: Larger test -- 1024-element witness into 256-element table
    // =========================================================================
    do {
        let tableSize = 256
        let witnessSize = 1024
        let table = (0..<tableSize).map { frFromInt(UInt64($0)) }

        // Generate random witness values all within [0, tableSize)
        let witnessRaw = pseudoRandomValues(witnessSize, bound: tableSize)
        let witness = witnessRaw.map { frFromInt($0) }

        let proverT = Transcript(label: "logup-gkr-test-7")
        let t0 = CFAbsoluteTimeGetCurrent()
        let proof = prover.prove(witness: witness, table: table, transcript: proverT)
        let proveTime = CFAbsoluteTimeGetCurrent() - t0

        let verifierT = Transcript(label: "logup-gkr-test-7")
        let t1 = CFAbsoluteTimeGetCurrent()
        let valid = verifier.verify(witness: witness, table: table, proof: proof, transcript: verifierT)
        let verifyTime = CFAbsoluteTimeGetCurrent() - t1

        expect(valid, "Large lookup: 1024 elements into 256-element table")
        print("    1024-into-256 prove: \(String(format: "%.2f", proveTime * 1000))ms, verify: \(String(format: "%.2f", verifyTime * 1000))ms")
    }

    // =========================================================================
    // Test 8: LogUp relation correctness -- witness sum equals table sum
    // =========================================================================
    do {
        let table = (0..<4).map { frFromInt(UInt64($0)) }
        let witness = [0, 1, 2, 3].map { frFromInt(UInt64($0)) }

        let proverT = Transcript(label: "logup-gkr-test-8")
        let proof = prover.prove(witness: witness, table: table, transcript: proverT)

        // The LogUp relation: witnessSum = tableSum
        expect(frEqual(proof.claimedWitnessSum, proof.claimedTableSum),
               "LogUp relation: witness sum equals table sum")
    }

    // =========================================================================
    // Test 9: Batch with invalid instance should fail
    // =========================================================================
    do {
        let batchProver = LogUpBatchGKRProver()
        let batchVerifier = LogUpBatchGKRVerifier()

        // Instance 0: valid
        let inst0 = LogUpBatchInstance(
            witness: [0, 1, 2, 3].map { frFromInt(UInt64($0)) },
            table: (0..<8).map { frFromInt(UInt64($0)) }
        )

        // Instance 1: invalid -- witness has 99 not in table
        let inst1 = LogUpBatchInstance(
            witness: [0, 1, 99, 3].map { frFromInt(UInt64($0)) },
            table: (0..<8).map { frFromInt(UInt64($0)) }
        )

        let instances = [inst0, inst1]

        let proverT = Transcript(label: "logup-batch-test-9")
        let batchProof = batchProver.prove(instances: instances, transcript: proverT)

        let verifierT = Transcript(label: "logup-batch-test-9")
        let valid = batchVerifier.verify(instances: instances, proof: batchProof, transcript: verifierT)
        expect(!valid, "Batch with invalid instance should fail")
    }

    // =========================================================================
    // Test 10: Sumcheck round count matches log2(size)
    // =========================================================================
    do {
        let table = (0..<16).map { frFromInt(UInt64($0)) }
        let witness = (0..<8).map { frFromInt(UInt64($0)) }

        let proverT = Transcript(label: "logup-gkr-test-10")
        let proof = prover.prove(witness: witness, table: table, transcript: proverT)

        // Witness has 8 elements -> log2(8) = 3 sumcheck rounds
        expectEqual(proof.witnessSumcheckMsgs.count, 3,
                    "Witness sumcheck should have log2(8) = 3 rounds")
        // Table has 16 elements -> log2(16) = 4 sumcheck rounds
        expectEqual(proof.tableSumcheckMsgs.count, 4,
                    "Table sumcheck should have log2(16) = 4 rounds")
    }
}
