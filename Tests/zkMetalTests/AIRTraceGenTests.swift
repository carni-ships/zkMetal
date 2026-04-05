// AIR Trace Generation Tests — validates AIRTraceGenerator for Plonky3-compatible traces
//
// Tests Fibonacci, range check, permutation, and hash chain trace generation,
// power-of-2 padding, and integration with Merkle commitment + challenge generation.

import zkMetal

public func runAIRTraceGenTests() {
    suite("AIR Trace Gen — Fibonacci")
    testFibonacciTraceTransitions()
    testFibonacciTraceInitialValues()
    testFibonacciTracePadding()

    suite("AIR Trace Gen — Range Check")
    testRangeCheckDecomposition()
    testRangeCheckBitsAreBoolean()
    testRangeCheckReconstruction()

    suite("AIR Trace Gen — Permutation")
    testPermutationGrandProduct()
    testPermutationIdentity()
    testPermutationNonPermutation()

    suite("AIR Trace Gen — Hash Chain")
    testHashChainTrace()
    testHashChainDeterminism()

    suite("AIR Trace Gen — Padding")
    testPaddingToPowerOfTwo()
    testAlreadyPowerOfTwo()

    suite("AIR Trace Gen — Integration")
    testTraceCommitAndChallenge()
}

// MARK: - Fibonacci Tests

func testFibonacciTraceTransitions() {
    let gen = AIRTraceGenerator()
    let trace = gen.generateFibonacciTrace(steps: 16, a0: Bb.one, a1: Bb.one)

    // Trace should have 2 columns
    expectEqual(trace.count, 2, "Fibonacci trace should have 2 columns")

    let colA = trace[0]
    let colB = trace[1]

    // Check transition constraints: a[i+1] = b[i], b[i+1] = a[i] + b[i]
    // Only check up to the original step count (before any padding replication)
    for i in 0..<15 {
        expectEqual(colA[i + 1], colB[i],
                    "row \(i): a[i+1] should equal b[i]")
        expectEqual(colB[i + 1], bbAdd(colA[i], colB[i]),
                    "row \(i): b[i+1] should equal a[i]+b[i]")
    }
    _passed += 1  // overall transition check passed
}

func testFibonacciTraceInitialValues() {
    let gen = AIRTraceGenerator()
    let a0 = Bb(v: 5)
    let a1 = Bb(v: 8)
    let trace = gen.generateFibonacciTrace(steps: 8, a0: a0, a1: a1)

    expectEqual(trace[0][0], a0, "a[0] should match a0")
    expectEqual(trace[1][0], a1, "b[0] should match a1")

    // Verify a few steps manually: 5, 8, 13, 21, 34, ...
    expectEqual(trace[0][1], Bb(v: 8), "a[1] = b[0] = 8")
    expectEqual(trace[1][1], Bb(v: 13), "b[1] = a[0]+b[0] = 13")
    expectEqual(trace[0][2], Bb(v: 13), "a[2] = b[1] = 13")
    expectEqual(trace[1][2], Bb(v: 21), "b[2] = a[1]+b[1] = 21")
}

func testFibonacciTracePadding() {
    let gen = AIRTraceGenerator()
    // 10 steps -> should be padded to 16 (next power of 2)
    let trace = gen.generateFibonacciTrace(steps: 10, a0: Bb.one, a1: Bb.one)
    expectEqual(trace[0].count, 16, "10 steps should pad to 16 rows")
    expectEqual(trace[1].count, 16, "both columns should have 16 rows")
}

// MARK: - Range Check Tests

func testRangeCheckDecomposition() {
    let gen = AIRTraceGenerator()
    let values: [Bb] = [Bb(v: 5), Bb(v: 13), Bb(v: 255), Bb(v: 0)]
    let trace = gen.generateRangeCheckTrace(values: values, bound: 256)

    // Should have 1 value column + 8 bit columns (ceil(log2(256)) = 8)
    expectEqual(trace.count, 9, "256-bound range check: 1 value + 8 bit columns")
}

func testRangeCheckBitsAreBoolean() {
    let gen = AIRTraceGenerator()
    let values: [Bb] = [Bb(v: 42), Bb(v: 100), Bb(v: 7), Bb(v: 63)]
    let trace = gen.generateRangeCheckTrace(values: values, bound: 128)

    // Bit columns (indices 1..) should contain only 0 or 1
    let originalRowCount = values.count
    for col in 1..<trace.count {
        for row in 0..<originalRowCount {
            let val = trace[col][row].v
            expect(val == 0 || val == 1,
                   "Bit column \(col-1) row \(row): value \(val) not boolean")
        }
    }
    _passed += 1
}

func testRangeCheckReconstruction() {
    let gen = AIRTraceGenerator()
    let values: [Bb] = [Bb(v: 42), Bb(v: 100), Bb(v: 7), Bb(v: 63)]
    let trace = gen.generateRangeCheckTrace(values: values, bound: 128)

    let numBits = trace.count - 1  // first column is the value
    let originalRowCount = values.count

    for row in 0..<originalRowCount {
        // Reconstruct value from bits
        var reconstructed: UInt32 = 0
        for bit in 0..<numBits {
            reconstructed += trace[bit + 1][row].v << bit
        }
        expectEqual(Bb(v: reconstructed), trace[0][row],
                    "Row \(row): reconstructed value should match original")
    }
}

// MARK: - Permutation Tests

func testPermutationGrandProduct() {
    let gen = AIRTraceGenerator()
    let original: [Bb] = [Bb(v: 1), Bb(v: 2), Bb(v: 3), Bb(v: 4)]
    let permuted: [Bb] = [Bb(v: 3), Bb(v: 1), Bb(v: 4), Bb(v: 2)]

    let trace = gen.generatePermutationTrace(original: original, permuted: permuted)

    // Should have 3 columns: original, permuted, accumulator
    expectEqual(trace.count, 3, "Permutation trace: 3 columns")

    // Grand product at final row of original data should be 1
    // (accumulator is at index 2, last original row is at index 3)
    let accum = trace[2]
    let lastOriginalRow = original.count - 1
    expectEqual(accum[lastOriginalRow], Bb.one,
                "Grand product should equal 1 for valid permutation")
}

func testPermutationIdentity() {
    let gen = AIRTraceGenerator()
    let values: [Bb] = [Bb(v: 10), Bb(v: 20), Bb(v: 30), Bb(v: 40)]

    // Identity permutation: same order
    let trace = gen.generatePermutationTrace(original: values, permuted: values)
    let lastOriginalRow = values.count - 1
    expectEqual(trace[2][lastOriginalRow], Bb.one,
                "Identity permutation grand product should be 1")
}

func testPermutationNonPermutation() {
    let gen = AIRTraceGenerator()
    let original: [Bb] = [Bb(v: 1), Bb(v: 2), Bb(v: 3), Bb(v: 4)]
    let notPermuted: [Bb] = [Bb(v: 1), Bb(v: 2), Bb(v: 3), Bb(v: 5)]  // 5 != 4

    let trace = gen.generatePermutationTrace(original: original, permuted: notPermuted)
    let lastOriginalRow = original.count - 1
    expect(trace[2][lastOriginalRow] != Bb.one,
           "Non-permutation grand product should NOT equal 1")
}

// MARK: - Hash Chain Tests

func testHashChainTrace() {
    let gen = AIRTraceGenerator()

    // Simple hash function for testing: sum all elements, return [sum, sum*2]
    let testHash: ([Bb]) -> [Bb] = { inputs in
        var sum = Bb.zero
        for v in inputs { sum = bbAdd(sum, v) }
        return [sum, bbMul(sum, Bb(v: 2))]
    }

    let inputs: [[Bb]] = [
        [Bb(v: 1), Bb(v: 2)],
        [Bb(v: 3), Bb(v: 4)],
        [Bb(v: 5), Bb(v: 6)],
        [Bb(v: 7), Bb(v: 8)]
    ]

    let trace = gen.generateHashChainTrace(inputs: inputs, hashFn: testHash)

    // Should have 2 columns (hash output width)
    expectEqual(trace.count, 2, "Hash chain trace should have 2 columns (output width)")

    // Row 0: hash([1,2]) = [3, 6]
    expectEqual(trace[0][0], Bb(v: 3), "Row 0 col 0: hash([1,2])[0] = 3")
    expectEqual(trace[1][0], Bb(v: 6), "Row 0 col 1: hash([1,2])[1] = 6")

    // Row 1: hash([3, 6, 3, 4]) = [16, 32]
    let expectedSum1 = bbAdd(bbAdd(Bb(v: 3), Bb(v: 6)), bbAdd(Bb(v: 3), Bb(v: 4)))
    expectEqual(trace[0][1], expectedSum1, "Row 1 col 0: chained hash")
}

func testHashChainDeterminism() {
    let gen = AIRTraceGenerator()
    let hash: ([Bb]) -> [Bb] = { inputs in
        var sum = Bb.zero
        for v in inputs { sum = bbAdd(sum, v) }
        return [sum]
    }
    let inputs: [[Bb]] = [[Bb(v: 10)], [Bb(v: 20)]]

    let trace1 = gen.generateHashChainTrace(inputs: inputs, hashFn: hash)
    let trace2 = gen.generateHashChainTrace(inputs: inputs, hashFn: hash)

    expectEqual(trace1[0][0], trace2[0][0], "Hash chain should be deterministic")
    expectEqual(trace1[0][1], trace2[0][1], "Hash chain should be deterministic (row 1)")
}

// MARK: - Padding Tests

func testPaddingToPowerOfTwo() {
    // 5 elements -> pad to 8
    let col1: [Bb] = [Bb(v: 1), Bb(v: 2), Bb(v: 3), Bb(v: 4), Bb(v: 5)]
    let col2: [Bb] = [Bb(v: 10), Bb(v: 20), Bb(v: 30), Bb(v: 40), Bb(v: 50)]
    let padded = AIRTraceGenerator.padToPowerOfTwo([col1, col2])

    expectEqual(padded[0].count, 8, "5 rows should pad to 8")
    expectEqual(padded[1].count, 8, "both columns should pad to 8")

    // Padding should replicate last value
    expectEqual(padded[0][5], Bb(v: 5), "Padded row 5 should replicate last value")
    expectEqual(padded[0][6], Bb(v: 5), "Padded row 6 should replicate last value")
    expectEqual(padded[0][7], Bb(v: 5), "Padded row 7 should replicate last value")
    expectEqual(padded[1][7], Bb(v: 50), "Column 2 padded row should replicate last value")
}

func testAlreadyPowerOfTwo() {
    let col: [Bb] = [Bb(v: 1), Bb(v: 2), Bb(v: 3), Bb(v: 4)]
    let padded = AIRTraceGenerator.padToPowerOfTwo([col])
    expectEqual(padded[0].count, 4, "Already power-of-2 should not change length")
}

// MARK: - Integration: Trace -> Commit -> Challenges

func testTraceCommitAndChallenge() {
    let gen = AIRTraceGenerator()

    // Generate a Fibonacci trace
    let trace = gen.generateFibonacciTrace(steps: 16, a0: Bb.one, a1: Bb.one)
    expectEqual(trace[0].count, 16, "Trace should have 16 rows")

    // Commit via Poseidon2 Merkle tree
    do {
        let commitment = try Plonky3TraceCommitment.commit(columns: trace)

        // Root should be 8 BabyBear elements
        expectEqual(commitment.root.count, 8, "Merkle root should be 8 Bb elements")
        expectEqual(commitment.numRows, 16, "Commitment numRows should match trace")
        expectEqual(commitment.numColumns, 2, "Commitment numColumns should match trace")

        // Root should not be all zeros (probabilistically)
        let allZero = commitment.root.allSatisfy { $0.v == 0 }
        expect(!allZero, "Merkle root should not be all zeros")

        // Generate challenges from the commitment using Plonky3 challenger
        let challenger = Plonky3Challenger()
        let challenges = PlonkyChallengeSet.generateFromChallenger(
            challenger,
            traceCommitment: commitment.root,
            logTraceLength: 4,  // log2(16)
            config: .fast
        )

        // Alpha and zeta should be valid BabyBear elements
        expect(challenges.alpha.v < Bb.P, "alpha should be valid BabyBear element")
        expect(challenges.zeta.v < Bb.P, "zeta should be valid BabyBear element")

        // FRI challenges: for logTraceLength=4, fold-by-4 -> ceil(4/2) = 2 rounds
        expectEqual(challenges.friChallenges.count, 2,
                    "FRI challenges count for log=4, fold=4")

        for (i, fc) in challenges.friChallenges.enumerated() {
            expect(fc.v < Bb.P, "FRI challenge \(i) should be valid BabyBear")
        }

        // Verify Merkle opening proof for a random row
        let row = 5
        let proof = commitment.openingProof(row: row)
        // Reconstruct the leaf hash
        var rowData = [Bb]()
        for col in 0..<trace.count {
            rowData.append(trace[col][row])
        }
        let leafHash = poseidon2BbHashMany(rowData)
        let verified = Plonky3TraceCommitment.verifyOpening(
            root: commitment.root,
            leaf: leafHash,
            row: row,
            path: proof
        )
        expect(verified, "Merkle opening proof should verify")

    } catch {
        expect(false, "Trace commitment failed: \(error)")
    }
}
