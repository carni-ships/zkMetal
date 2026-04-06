// GPU Circle STARK Prover Tests — comprehensive tests for GPU-accelerated Circle STARK engine
// Tests: M31 field ops, Poseidon2-M31 Merkle, circle domain, LDE, constraint eval,
// quotient splitting, FRI folding, prove/verify round-trip, soundness, proof structure

import zkMetal
import Foundation

public func runGPUCircleSTARKProverTests() {
    suite("GPU Circle STARK Prover -- Config")
    testGPUCircleSTARKConfig()

    suite("GPU Circle STARK Prover -- M31 Digest")
    testM31Digest()

    suite("GPU Circle STARK Prover -- Poseidon2-M31 Merkle")
    testPoseidon2M31MerkleTree()
    testPoseidon2M31MerkleProofVerify()
    testPoseidon2M31MerkleTampered()

    suite("GPU Circle STARK Prover -- Circle Domain")
    testCircleDomainCoset()
    testCircleVanishingPoly()

    suite("GPU Circle STARK Prover -- Quotient Splitting")
    testQuotientSplitIdentity()
    testQuotientSplitMultiple()

    suite("GPU Circle STARK Prover -- Fibonacci Prove")
    testFibonacciProve()
    testFibonacciProveVerifyRoundTrip()
    testFibonacciLargerTrace()

    suite("GPU Circle STARK Prover -- Range Check Prove")
    testRangeCheckProve()

    suite("GPU Circle STARK Prover -- Generic AIR")
    testGenericAIRProve()

    suite("GPU Circle STARK Prover -- Proof Structure")
    testProofStructure()
    testProofSizeEstimate()

    suite("GPU Circle STARK Prover -- Timing Data")
    testTimingData()

    suite("GPU Circle STARK Prover -- Soundness")
    gpuCsTestInvalidTraceRejected()
    gpuCsTestTamperedProofRejected()
    testWrongAlphaRejected()

    suite("GPU Circle STARK Prover -- Config Variants")
    testFastConfig()
    testHighSecurityConfig()

    suite("GPU Circle STARK Prover -- FRI Structure")
    testFRIRoundCount()
    testFRIQueryIndices()

    suite("GPU Circle STARK Prover -- Merkle Path Consistency")
    testTraceCommitmentConsistency()
    testCompositionCommitmentConsistency()

    suite("GPU Circle STARK Prover -- Determinism")
    testDeterministicProving()
}

// MARK: - Config Tests

private func testGPUCircleSTARKConfig() {
    let cfg = GPUCircleSTARKProverConfig.default
    expectEqual(cfg.logBlowup, 2, "Default logBlowup = 2")
    expectEqual(cfg.blowupFactor, 4, "Default blowup factor = 4")
    expectEqual(cfg.numQueries, 20, "Default numQueries = 20")
    expectEqual(cfg.extensionDegree, 4, "Default extension degree = 4")
    expect(cfg.usePoseidon2Merkle, "Default uses Poseidon2 Merkle")
    expectEqual(cfg.securityBits, 40, "Default security bits = 2 * 20 = 40")

    let fast = GPUCircleSTARKProverConfig.fast
    expectEqual(fast.logBlowup, 1, "Fast logBlowup = 1")
    expectEqual(fast.blowupFactor, 2, "Fast blowup = 2")
    expectEqual(fast.numQueries, 8, "Fast numQueries = 8")
    expectEqual(fast.securityBits, 8, "Fast security bits = 1 * 8 = 8")

    let hs = GPUCircleSTARKProverConfig.highSecurity
    expectEqual(hs.logBlowup, 4, "HighSec logBlowup = 4")
    expectEqual(hs.numQueries, 40, "HighSec numQueries = 40")
    expectEqual(hs.securityBits, 160, "HighSec security bits = 4 * 40 = 160")

    // Custom config
    let custom = GPUCircleSTARKProverConfig(
        logBlowup: 3, numQueries: 15, extensionDegree: 4,
        gpuConstraintThreshold: 64, gpuFRIFoldThreshold: 64,
        usePoseidon2Merkle: true, numQuotientSplits: 4
    )
    expectEqual(custom.blowupFactor, 8, "Custom blowup = 8")
    expectEqual(custom.securityBits, 45, "Custom security = 3 * 15 = 45")
    expectEqual(custom.numQuotientSplits, 4, "Custom quotient splits = 4")
}

// MARK: - M31 Digest Tests

private func testM31Digest() {
    let zero = M31Digest.zero
    expectEqual(zero.values.count, 8, "Zero digest has 8 elements")
    expect(!zero.isNonTrivial, "Zero digest is trivial")

    let vals = (0..<8).map { M31(v: UInt32($0 + 1)) }
    let d = M31Digest(values: vals)
    expectEqual(d.values.count, 8, "Digest has 8 elements")
    expect(d.isNonTrivial, "Non-zero digest is non-trivial")
    expectEqual(d.values[0].v, UInt32(1), "Digest value[0]")
    expectEqual(d.values[7].v, UInt32(8), "Digest value[7]")

    // Bytes conversion
    let bytes = d.bytes
    expectEqual(bytes.count, 32, "Digest bytes = 32")

    // Equality
    let d2 = M31Digest(values: vals)
    expect(d == d2, "Equal digests")
    let d3 = M31Digest(values: (0..<8).map { M31(v: UInt32($0 + 100)) })
    expect(d != d3, "Different digests not equal")
}

// MARK: - Poseidon2-M31 Merkle Tests

private func testPoseidon2M31MerkleTree() {
    let n = 8
    let values = (0..<n).map { M31(v: UInt32($0 + 1)) }
    let tree = buildPoseidon2M31MerkleTree(values, count: n)

    let treeSize = 2 * n - 1
    expectEqual(tree.count, treeSize, "Merkle tree size = 2n - 1 = 15")

    // Leaves should be non-trivial (hashed)
    for i in 0..<n {
        expect(tree[i].isNonTrivial, "Leaf \(i) is non-trivial")
    }

    // Internal nodes should be non-trivial
    for i in n..<treeSize {
        expect(tree[i].isNonTrivial, "Internal node \(i) is non-trivial")
    }

    // Root should be deterministic
    let root = poseidon2M31MerkleRoot(tree, n: n)
    let tree2 = buildPoseidon2M31MerkleTree(values, count: n)
    let root2 = poseidon2M31MerkleRoot(tree2, n: n)
    expect(root == root2, "Merkle root is deterministic")

    // Different values should give different root
    let values2 = (0..<n).map { M31(v: UInt32($0 + 100)) }
    let tree3 = buildPoseidon2M31MerkleTree(values2, count: n)
    let root3 = poseidon2M31MerkleRoot(tree3, n: n)
    expect(root != root3, "Different values give different root")
}

private func testPoseidon2M31MerkleProofVerify() {
    let n = 16
    let values = (0..<n).map { M31(v: UInt32($0 * 7 + 3)) }
    let tree = buildPoseidon2M31MerkleTree(values, count: n)
    let root = poseidon2M31MerkleRoot(tree, n: n)

    // Verify proof for every leaf
    for i in 0..<n {
        let path = poseidon2M31MerkleProof(tree, n: n, index: i)
        expectEqual(path.count, 4, "Merkle path depth = log2(16) = 4")

        // Reconstruct leaf digest
        let val = values[i]
        let leafInput = [val, M31(v: UInt32(i)), M31.zero, M31.zero,
                         M31.zero, M31.zero, M31.zero, M31.zero]
        let leafDigest = M31Digest(values: poseidon2M31HashSingle(leafInput))

        let valid = verifyPoseidon2M31MerkleProof(
            leafDigest: leafDigest, path: path,
            index: i, root: root
        )
        expect(valid, "Merkle proof valid for leaf \(i)")
    }
}

private func testPoseidon2M31MerkleTampered() {
    let n = 8
    let values = (0..<n).map { M31(v: UInt32($0 + 1)) }
    let tree = buildPoseidon2M31MerkleTree(values, count: n)
    let root = poseidon2M31MerkleRoot(tree, n: n)
    let path = poseidon2M31MerkleProof(tree, n: n, index: 3)

    // Tamper: use wrong value for leaf
    let wrongLeafInput = [M31(v: 999), M31(v: 3), M31.zero, M31.zero,
                          M31.zero, M31.zero, M31.zero, M31.zero]
    let wrongDigest = M31Digest(values: poseidon2M31HashSingle(wrongLeafInput))

    let invalid = verifyPoseidon2M31MerkleProof(
        leafDigest: wrongDigest, path: path,
        index: 3, root: root
    )
    expect(!invalid, "Tampered leaf rejected by Merkle proof")

    // Tamper: use wrong index
    let correctLeafInput = [values[3], M31(v: 3), M31.zero, M31.zero,
                            M31.zero, M31.zero, M31.zero, M31.zero]
    let correctDigest = M31Digest(values: poseidon2M31HashSingle(correctLeafInput))

    let wrongIdx = verifyPoseidon2M31MerkleProof(
        leafDigest: correctDigest, path: path,
        index: 5, root: root
    )
    expect(!wrongIdx, "Wrong index rejected by Merkle proof")
}

// MARK: - Circle Domain Tests

private func testCircleDomainCoset() {
    // Circle coset domain should produce points on the circle
    let logN = 4
    let domain = circleCosetDomain(logN: logN)
    expectEqual(domain.count, 16, "Domain size = 2^4 = 16")

    for (i, pt) in domain.enumerated() {
        expect(pt.isOnCircle, "Domain point \(i) is on circle")
    }

    // All points should be distinct
    var seen = Set<UInt64>()
    for pt in domain {
        let key = UInt64(pt.x.v) | (UInt64(pt.y.v) << 32)
        seen.insert(key)
    }
    expectEqual(seen.count, 16, "All 16 domain points are distinct")

    // Subgroup generator should have correct order
    let gen = circleSubgroupGenerator(logN: logN)
    expect(gen.isOnCircle, "Subgroup generator is on circle")
    let powered = circleGroupPow(gen, 1 << logN)
    expectEqual(powered.x.v, M31.one.v, "Generator^(2^logN) = identity.x")
    expectEqual(powered.y.v, M31.zero.v, "Generator^(2^logN) = identity.y")
}

private func testCircleVanishingPoly() {
    let logN = 3
    let domain = circleCosetDomain(logN: logN)

    // Vanishing polynomial evaluated at domain points
    // It vanishes on the trace domain but NOT on the full evaluation domain
    // since the coset is shifted
    var nonZeroCount = 0
    for pt in domain {
        let v = circleVanishing(point: pt, logDomainSize: logN)
        if v.v != 0 { nonZeroCount += 1 }
    }
    // Most evaluation domain points should give non-zero vanishing polynomial
    expect(nonZeroCount > 0, "Vanishing poly is non-zero at some coset points")
}

// MARK: - Quotient Splitting Tests

private func testQuotientSplitIdentity() {
    let logN = 4
    let n = 1 << logN
    let evals = (0..<n).map { M31(v: UInt32($0 + 1)) }

    // Split with 1 = identity
    let splits = circleQuotientSplit(evals: evals, logN: logN, numSplits: 1)
    expectEqual(splits.count, 1, "Single split = 1 piece")
    expectEqual(splits[0].count, n, "Single split size = n")
    for i in 0..<n {
        expectEqual(splits[0][i].v, evals[i].v, "Single split preserves values at \(i)")
    }
}

private func testQuotientSplitMultiple() {
    let logN = 4
    let n = 1 << logN
    let evals = (0..<n).map { M31(v: UInt32($0 + 1)) }

    // Split with 2
    let splits2 = circleQuotientSplit(evals: evals, logN: logN, numSplits: 2)
    expectEqual(splits2.count, 2, "Two splits")
    expectEqual(splits2[0].count, n / 2, "Split 0 size = n/2")
    expectEqual(splits2[1].count, n / 2, "Split 1 size = n/2")

    // Values should be interleaved: split_j[i] = evals[2*i + j]
    for i in 0..<(n / 2) {
        expectEqual(splits2[0][i].v, evals[2 * i].v, "Split 0 interleave at \(i)")
        expectEqual(splits2[1][i].v, evals[2 * i + 1].v, "Split 1 interleave at \(i)")
    }

    // Split with 4
    let splits4 = circleQuotientSplit(evals: evals, logN: logN, numSplits: 4)
    expectEqual(splits4.count, 4, "Four splits")
    for j in 0..<4 {
        expectEqual(splits4[j].count, n / 4, "Split \(j) size = n/4")
    }
}

// MARK: - Fibonacci Prove Tests

private func testFibonacciProve() {
    do {
        let air = FibonacciAIR(logTraceLength: 3)  // 8 rows
        let engine = GPUCircleSTARKProverEngine(config: .fast)
        let result = try engine.prove(air: air)

        expectEqual(result.traceLength, 8, "Trace length = 8")
        expectEqual(result.numColumns, 2, "Num columns = 2")
        expect(result.totalTimeSeconds > 0, "Total time recorded")

        let proof = result.proof
        expectEqual(proof.traceCommitments.count, 2, "2 trace commitments")
        expect(proof.compositionCommitment.isNonTrivial, "Composition commitment non-trivial")
        expect(proof.alpha.v != 0, "Alpha is non-zero")
        expectEqual(proof.traceLength, 8, "Proof trace length = 8")
        expectEqual(proof.numColumns, 2, "Proof num columns = 2")
    } catch {
        expect(false, "Fibonacci prove error: \(error)")
    }
}

private func testFibonacciProveVerifyRoundTrip() {
    do {
        let air = FibonacciAIR(logTraceLength: 3)
        let engine = GPUCircleSTARKProverEngine(config: .fast)
        let (result, verified) = try engine.proveAndVerify(air: air)

        expect(verified, "Fibonacci prove-verify round trip")
        expectEqual(result.traceLength, 8, "Result trace length")

        // Verify the proof independently
        let verified2 = engine.verify(air: air, proof: result.proof)
        expect(verified2, "Independent verify also passes")
    } catch {
        expect(false, "Fibonacci prove-verify error: \(error)")
    }
}

private func testFibonacciLargerTrace() {
    do {
        let air = FibonacciAIR(logTraceLength: 5)  // 32 rows
        let engine = GPUCircleSTARKProverEngine(config: .fast)
        let (result, verified) = try engine.proveAndVerify(air: air)

        expect(verified, "Larger Fibonacci (32 rows) prove-verify")
        expectEqual(result.traceLength, 32, "Trace length = 32")

        // Different initial values
        let air2 = FibonacciAIR(logTraceLength: 4, a0: M31(v: 5), b0: M31(v: 7))
        let (_, verified2) = try engine.proveAndVerify(air: air2)
        expect(verified2, "Fibonacci with custom initial values prove-verify")
    } catch {
        expect(false, "Larger Fibonacci prove error: \(error)")
    }
}

// MARK: - Range Check Prove Tests

private func testRangeCheckProve() {
    do {
        let values = (0..<8).map { M31(v: UInt32($0 * 100)) }
        let air = RangeCheckAIR(logTraceLength: 3, values: values, bound: 65536)
        let engine = GPUCircleSTARKProverEngine(config: .fast)
        let result = try engine.prove(air: air)

        expectEqual(result.traceLength, 8, "Range check trace length = 8")
        expectEqual(result.numColumns, 1, "Range check 1 column")
        expect(result.proof.traceCommitments.count == 1, "1 trace commitment for range check")
    } catch {
        expect(false, "Range check prove error: \(error)")
    }
}

// MARK: - Generic AIR Tests

private func testGenericAIRProve() {
    do {
        // Simple AIR: single column, value doubles each row
        let air = GenericAIR(
            numColumns: 1,
            logTraceLength: 3,
            constraintDegrees: [1],
            boundaryConstraints: [(0, 0, M31(v: 1))],
            generateTrace: {
                var col = [M31](repeating: M31.zero, count: 8)
                col[0] = M31.one
                for i in 1..<8 { col[i] = m31Add(col[i-1], col[i-1]) }
                return [col]
            },
            evaluateConstraints: { current, next in
                [m31Sub(next[0], m31Add(current[0], current[0]))]
            }
        )

        let engine = GPUCircleSTARKProverEngine(config: .fast)
        let result = try engine.prove(air: air)

        expectEqual(result.traceLength, 8, "Generic AIR trace length = 8")
        expectEqual(result.numColumns, 1, "Generic AIR 1 column")
        expect(result.proof.compositionCommitment.isNonTrivial, "Generic AIR composition non-trivial")
    } catch {
        expect(false, "Generic AIR prove error: \(error)")
    }
}

// MARK: - Proof Structure Tests

private func testProofStructure() {
    do {
        let air = FibonacciAIR(logTraceLength: 3)
        let engine = GPUCircleSTARKProverEngine(config: .fast)
        let result = try engine.prove(air: air)
        let proof = result.proof

        // Trace commitments
        expectEqual(proof.traceCommitments.count, 2, "2 trace commitments for Fibonacci")
        for (i, comm) in proof.traceCommitments.enumerated() {
            expect(comm.isNonTrivial, "Trace commitment \(i) non-trivial")
            expectEqual(comm.values.count, 8, "Trace commitment \(i) has 8 M31 elements")
        }

        // Different columns should have different commitments
        expect(proof.traceCommitments[0] != proof.traceCommitments[1],
               "Different columns have different commitments")

        // Composition commitment
        expect(proof.compositionCommitment.isNonTrivial, "Composition commitment non-trivial")

        // Quotient commitments
        expectEqual(proof.quotientCommitments.count, 2, "2 quotient splits (default)")
        for (i, qc) in proof.quotientCommitments.enumerated() {
            expect(qc.isNonTrivial, "Quotient commitment \(i) non-trivial")
        }

        // FRI proof
        expect(proof.friProof.rounds.count > 0, "FRI has at least 1 round")
        expectEqual(proof.friProof.queryIndices.count, 8, "8 queries (fast config)")

        // Query responses
        expectEqual(proof.queryResponses.count, 8, "8 query responses")
        for (i, qr) in proof.queryResponses.enumerated() {
            expectEqual(qr.traceValues.count, 2, "Query \(i) has 2 trace values")
            expectEqual(qr.tracePaths.count, 2, "Query \(i) has 2 trace paths")
            expect(qr.compositionPath.count > 0, "Query \(i) has composition path")
            expectEqual(qr.quotientSplitValues.count, 2, "Query \(i) has 2 quotient split values")
        }
    } catch {
        expect(false, "Proof structure test error: \(error)")
    }
}

private func testProofSizeEstimate() {
    do {
        let air = FibonacciAIR(logTraceLength: 3)
        let engine = GPUCircleSTARKProverEngine(config: .fast)
        let result = try engine.prove(air: air)
        let proof = result.proof

        let size = proof.estimatedSizeBytes
        expect(size > 0, "Proof size > 0")
        expect(size < 1024 * 1024, "Proof size < 1 MiB for small trace")

        let desc = proof.proofSizeDescription
        expect(!desc.isEmpty, "Proof size description not empty")
    } catch {
        expect(false, "Proof size test error: \(error)")
    }
}

// MARK: - Timing Data Tests

private func testTimingData() {
    do {
        let air = FibonacciAIR(logTraceLength: 3)
        let engine = GPUCircleSTARKProverEngine(config: .fast)
        let result = try engine.prove(air: air)

        expect(result.totalTimeSeconds > 0, "Total time > 0")
        expect(result.traceGenTimeSeconds >= 0, "Trace gen time >= 0")
        expect(result.ldeTimeSeconds >= 0, "LDE time >= 0")
        expect(result.commitTimeSeconds >= 0, "Commit time >= 0")
        expect(result.constraintTimeSeconds >= 0, "Constraint time >= 0")
        expect(result.friTimeSeconds >= 0, "FRI time >= 0")
        expect(result.queryTimeSeconds >= 0, "Query time >= 0")

        // Sum of phases should approximate total (allow for rounding)
        let phaseSum = result.traceGenTimeSeconds + result.ldeTimeSeconds +
                       result.commitTimeSeconds + result.constraintTimeSeconds +
                       result.friTimeSeconds + result.queryTimeSeconds
        let tolerance = 0.01  // 10ms tolerance
        expect(abs(phaseSum - result.totalTimeSeconds) < tolerance,
               "Phase sum approximates total time")
    } catch {
        expect(false, "Timing data test error: \(error)")
    }
}

// MARK: - Soundness Tests

private func gpuCsTestInvalidTraceRejected() {
    // Create a "bad" AIR where the trace does not satisfy constraints
    let badAIR = GenericAIR(
        numColumns: 2,
        logTraceLength: 3,
        constraintDegrees: [1, 1],
        boundaryConstraints: [(0, 0, M31(v: 1)), (1, 0, M31(v: 1))],
        generateTrace: {
            // Intentionally wrong: not a Fibonacci sequence
            let n = 8
            var colA = [M31](repeating: M31.zero, count: n)
            var colB = [M31](repeating: M31.zero, count: n)
            colA[0] = M31.one; colB[0] = M31.one
            for i in 1..<n {
                colA[i] = M31(v: UInt32(i * 3))  // wrong: not b[i-1]
                colB[i] = M31(v: UInt32(i * 5))  // wrong: not a[i-1] + b[i-1]
            }
            return [colA, colB]
        },
        evaluateConstraints: { current, next in
            // Fibonacci constraints
            let c0 = m31Sub(next[0], current[1])
            let c1 = m31Sub(next[1], m31Add(current[0], current[1]))
            return [c0, c1]
        }
    )

    // The trace itself should fail verification
    let trace = badAIR.generateTrace()
    let err = badAIR.verifyTrace(trace)
    expect(err != nil, "Bad trace detected by CPU verify")
}

private func gpuCsTestTamperedProofRejected() {
    do {
        let air = FibonacciAIR(logTraceLength: 3)
        let engine = GPUCircleSTARKProverEngine(config: .fast)
        let result = try engine.prove(air: air)

        // Tamper with trace commitment
        var tamperedCommitments = result.proof.traceCommitments
        let badDigest = M31Digest(values: (0..<8).map { M31(v: UInt32($0 * 999 + 1)) })
        tamperedCommitments[0] = badDigest

        let tamperedProof = GPUCircleSTARKProverProof(
            traceCommitments: tamperedCommitments,
            compositionCommitment: result.proof.compositionCommitment,
            quotientCommitments: result.proof.quotientCommitments,
            friProof: result.proof.friProof,
            queryResponses: result.proof.queryResponses,
            alpha: result.proof.alpha,
            traceLength: result.proof.traceLength,
            numColumns: result.proof.numColumns,
            logBlowup: result.proof.logBlowup
        )

        let valid = engine.verify(air: air, proof: tamperedProof)
        expect(!valid, "Tampered trace commitment rejected")
    } catch {
        expect(false, "Tampered proof test error: \(error)")
    }
}

private func testWrongAlphaRejected() {
    do {
        let air = FibonacciAIR(logTraceLength: 3)
        let engine = GPUCircleSTARKProverEngine(config: .fast)
        let result = try engine.prove(air: air)

        // Tamper with alpha
        let wrongAlpha = M31(v: result.proof.alpha.v ^ 0x12345678)
        let tamperedProof = GPUCircleSTARKProverProof(
            traceCommitments: result.proof.traceCommitments,
            compositionCommitment: result.proof.compositionCommitment,
            quotientCommitments: result.proof.quotientCommitments,
            friProof: result.proof.friProof,
            queryResponses: result.proof.queryResponses,
            alpha: wrongAlpha,
            traceLength: result.proof.traceLength,
            numColumns: result.proof.numColumns,
            logBlowup: result.proof.logBlowup
        )

        let valid = engine.verify(air: air, proof: tamperedProof)
        expect(!valid, "Wrong alpha rejected by verifier")
    } catch {
        expect(false, "Wrong alpha test error: \(error)")
    }
}

// MARK: - Config Variant Tests

private func testFastConfig() {
    do {
        let air = FibonacciAIR(logTraceLength: 3)
        let engine = GPUCircleSTARKProverEngine(config: .fast)
        let result = try engine.prove(air: air)

        expectEqual(result.proof.logBlowup, 1, "Fast config logBlowup = 1")
        expectEqual(result.proof.friProof.queryIndices.count, 8, "Fast config 8 queries")
    } catch {
        expect(false, "Fast config test error: \(error)")
    }
}

private func testHighSecurityConfig() {
    do {
        let air = FibonacciAIR(logTraceLength: 3)
        let engine = GPUCircleSTARKProverEngine(config: .highSecurity)
        let result = try engine.prove(air: air)

        expectEqual(result.proof.logBlowup, 4, "HighSec logBlowup = 4")
        expectEqual(result.proof.friProof.queryIndices.count, 40, "HighSec 40 queries")
        expect(result.proof.queryResponses.count <= 40, "HighSec at most 40 query responses")
    } catch {
        expect(false, "High security config test error: \(error)")
    }
}

// MARK: - FRI Structure Tests

private func testFRIRoundCount() {
    do {
        let air = FibonacciAIR(logTraceLength: 4)  // 16 rows
        let engine = GPUCircleSTARKProverEngine(config: .fast)
        let result = try engine.prove(air: air)

        let logEval = 4 + 1  // logTrace + logBlowup(fast=1)
        // FRI folds from logEval down to 2, so rounds = logEval - 2
        let expectedRounds = logEval - 2
        expectEqual(result.proof.friProof.rounds.count, expectedRounds,
                   "FRI rounds = logEval - 2 = \(expectedRounds)")

        // Final value should be a single M31 element
        let finalVal = result.proof.friProof.finalValue
        // Just check it exists (it is the constant term after folding)
        expect(true, "FRI final value: \(finalVal.v)")
    } catch {
        expect(false, "FRI round count test error: \(error)")
    }
}

private func testFRIQueryIndices() {
    do {
        let air = FibonacciAIR(logTraceLength: 3)
        let engine = GPUCircleSTARKProverEngine(config: .fast)
        let result = try engine.prove(air: air)

        let indices = result.proof.friProof.queryIndices
        expectEqual(indices.count, 8, "8 query indices for fast config")

        // All indices should be in valid range [0, evalLen/2)
        let logEval = 3 + 1  // fast config logBlowup = 1
        let halfEval = (1 << logEval) / 2
        for (i, idx) in indices.enumerated() {
            expect(idx >= 0 && idx < halfEval,
                   "Query index \(i) in range [0, \(halfEval))")
        }

        // Each FRI round should have query responses
        for (rIdx, round) in result.proof.friProof.rounds.enumerated() {
            expectEqual(round.queryResponses.count, 8,
                       "FRI round \(rIdx) has 8 query responses")
            expect(round.commitment.isNonTrivial,
                   "FRI round \(rIdx) commitment is non-trivial")
        }
    } catch {
        expect(false, "FRI query indices test error: \(error)")
    }
}

// MARK: - Merkle Path Consistency Tests

private func testTraceCommitmentConsistency() {
    do {
        let air = FibonacciAIR(logTraceLength: 3)
        let engine = GPUCircleSTARKProverEngine(config: .fast)
        let result = try engine.prove(air: air)

        // Each trace commitment should be unique per column
        let comms = result.proof.traceCommitments
        expect(comms[0] != comms[1], "Trace commitments differ between columns")

        // All trace commitments should be non-trivial
        for (i, c) in comms.enumerated() {
            expect(c.isNonTrivial, "Trace commitment \(i) is non-trivial")
        }
    } catch {
        expect(false, "Trace commitment consistency error: \(error)")
    }
}

private func testCompositionCommitmentConsistency() {
    do {
        let air = FibonacciAIR(logTraceLength: 3)
        let engine = GPUCircleSTARKProverEngine(config: .fast)
        let result = try engine.prove(air: air)

        // Composition commitment should differ from trace commitments
        let compComm = result.proof.compositionCommitment
        for (i, tc) in result.proof.traceCommitments.enumerated() {
            expect(compComm != tc,
                   "Composition commitment differs from trace commitment \(i)")
        }

        // Quotient commitments should each be non-trivial
        for (i, qc) in result.proof.quotientCommitments.enumerated() {
            expect(qc.isNonTrivial, "Quotient commitment \(i) non-trivial")
        }
    } catch {
        expect(false, "Composition commitment consistency error: \(error)")
    }
}

// MARK: - Determinism Tests

private func testDeterministicProving() {
    do {
        let air = FibonacciAIR(logTraceLength: 3)
        let engine = GPUCircleSTARKProverEngine(config: .fast)

        let result1 = try engine.prove(air: air)
        let result2 = try engine.prove(air: air)

        // Same AIR + same config should produce same proof
        expectEqual(result1.proof.alpha.v, result2.proof.alpha.v,
                   "Alpha is deterministic")
        for i in 0..<result1.proof.traceCommitments.count {
            expect(result1.proof.traceCommitments[i] == result2.proof.traceCommitments[i],
                   "Trace commitment \(i) is deterministic")
        }
        expect(result1.proof.compositionCommitment == result2.proof.compositionCommitment,
               "Composition commitment is deterministic")
        expectEqual(result1.proof.friProof.finalValue.v,
                   result2.proof.friProof.finalValue.v,
                   "FRI final value is deterministic")
        expectEqual(result1.proof.friProof.queryIndices,
                   result2.proof.friProof.queryIndices,
                   "FRI query indices are deterministic")
    } catch {
        expect(false, "Deterministic proving test error: \(error)")
    }
}
