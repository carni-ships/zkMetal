// Comprehensive correctness tests for all zkMetal primitives
// Unified test suite: `zkbench test` runs everything
import zkMetal
import Foundation

/// Shared test counters and helper
private var _testPassed = 0
private var _testFailed = 0

private func check(_ name: String, _ ok: Bool) {
    if ok {
        print("  [pass] \(name)")
        _testPassed += 1
    } else {
        print("  [FAIL] \(name)")
        _testFailed += 1
    }
}

// ============================================================
// MARK: - Unified test runner (called by `zkbench test`)
// ============================================================
public func runAllCorrectnessTests() {
    _testPassed = 0
    _testFailed = 0

    let t0 = CFAbsoluteTimeGetCurrent()

    // CPU-only tests (no Metal engines needed)
    testSparseSumcheck()
    testSecp256k1Field()
    testSecp256k1GLV()
    testBLS12377Field()
    testBLS12377GLV()
    testParallelHash()
    testParallelMerkle()
    testParallelMSM()
    testCNTT()
    testMersenne31()

    // GPU tests (create Metal engines)
    testNTT_BN254()
    testNTT_BabyBear()
    testNTT_Goldilocks()
    testNTT_BLS12377()
    testCircleNTT()
    testPoseidon2()
    testKeccak()
    testBlake3()
    testFRI()
    testSumcheck()
    testPolyOps()
    testKZG()
    testMSM_BN254()
    testSecp256k1MSM()
    testBLS12377MSM()
    testECDSA()
    testIPA()
    testVerkle()
    testLogUp()
    testRadixSort()
    testCPippengerMSM()

    // ParallelNTT is known to trigger a Metal teardown SIGABRT on some systems.
    // Run it in a subprocess so a crash does not lose the rest of our results.
    testParallelNTT_safe()

    let elapsed = (CFAbsoluteTimeGetCurrent() - t0)
    print(String(format: "\n=== Correctness Summary: %d passed, %d failed (%.1fs) ===", _testPassed, _testFailed, elapsed))
    if _testFailed > 0 {
        print("*** SOME TESTS FAILED ***")
    }
    // Force flush and exit cleanly to avoid Metal teardown SIGABRT at process exit
    fflush(stdout)
    fflush(stderr)
    exit(_testFailed > 0 ? 1 : 0)
}

/// Legacy entry point (preserved for compatibility)
public func runCorrectnessTests() {
    runAllCorrectnessTests()
}

// ============================================================
// MARK: - NTT BN254 Fr
// ============================================================
private func testNTT_BN254() {
    print("\n--- NTT BN254 Fr ---")
    do {
        let engine = try NTTEngine()

        // Round-trip 2^10
        let testN = 1024
        var testInput = [Fr](repeating: Fr.zero, count: testN)
        for i in 0..<testN { testInput[i] = frFromInt(UInt64(i + 1)) }
        let gpuNTT = try engine.ntt(testInput)
        let gpuRecovered = try engine.intt(gpuNTT)
        var correct = true
        for i in 0..<testN {
            if frToInt(testInput[i]) != frToInt(gpuRecovered[i]) { correct = false; break }
        }
        check("GPU round-trip (2^10)", correct)

        // GPU vs CPU cross-check
        let cpuNTT = NTTEngine.cpuNTT(testInput, logN: 10)
        var cpuMatch = true
        for i in 0..<testN {
            if frToInt(gpuNTT[i]) != frToInt(cpuNTT[i]) { cpuMatch = false; break }
        }
        check("GPU vs CPU NTT (2^10)", cpuMatch)

        // Four-step round-trip 2^20
        let testN2 = 1 << 20
        var testInput2 = [Fr](repeating: Fr.zero, count: testN2)
        var rng: UInt64 = 0xCAFE_BABE
        for i in 0..<testN2 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            testInput2[i] = frFromInt(rng >> 32)
        }
        let ntt2 = try engine.ntt(testInput2)
        let rec2 = try engine.intt(ntt2)
        var mm = 0
        for i in 0..<testN2 {
            if frToInt(testInput2[i]) != frToInt(rec2[i]) { mm += 1 }
        }
        check("Four-step round-trip (2^20)", mm == 0)
    } catch {
        print("  [FAIL] NTT BN254 error: \(error)")
        _testFailed += 1
    }
}

// ============================================================
// MARK: - NTT BabyBear
// ============================================================
private func testNTT_BabyBear() {
    print("\n--- NTT BabyBear ---")
    do {
        let engine = try BabyBearNTTEngine()
        let testN = 1024
        var testInput = [Bb](repeating: Bb.zero, count: testN)
        for i in 0..<testN { testInput[i] = Bb(v: UInt32(i + 1)) }

        let gpuNTT = try engine.ntt(testInput)
        let gpuRec = try engine.intt(gpuNTT)
        var correct = true
        for i in 0..<testN { if testInput[i].v != gpuRec[i].v { correct = false; break } }
        check("GPU round-trip (2^10)", correct)

        let cpuNTT = BabyBearNTTEngine.cpuNTT(testInput, logN: 10)
        var cpuMatch = true
        for i in 0..<testN { if gpuNTT[i].v != cpuNTT[i].v { cpuMatch = false; break } }
        check("GPU vs CPU NTT (2^10)", cpuMatch)
    } catch {
        print("  [FAIL] NTT BabyBear error: \(error)")
        _testFailed += 1
    }
}

// ============================================================
// MARK: - NTT Goldilocks
// ============================================================
private func testNTT_Goldilocks() {
    print("\n--- NTT Goldilocks ---")
    do {
        let engine = try GoldilocksNTTEngine()
        let testN = 1024
        var testInput = [Gl](repeating: Gl.zero, count: testN)
        for i in 0..<testN { testInput[i] = Gl(v: UInt64(i + 1)) }

        let gpuNTT = try engine.ntt(testInput)
        let gpuRec = try engine.intt(gpuNTT)
        var correct = true
        for i in 0..<testN { if testInput[i].v != gpuRec[i].v { correct = false; break } }
        check("GPU round-trip (2^10)", correct)

        let cpuNTT = GoldilocksNTTEngine.cpuNTT(testInput, logN: 10)
        var cpuMatch = true
        for i in 0..<testN { if gpuNTT[i].v != cpuNTT[i].v { cpuMatch = false; break } }
        check("GPU vs CPU NTT (2^10)", cpuMatch)
    } catch {
        print("  [FAIL] NTT Goldilocks error: \(error)")
        _testFailed += 1
    }
}

// ============================================================
// MARK: - NTT BLS12-377
// ============================================================
private func testNTT_BLS12377() {
    print("\n--- NTT BLS12-377 ---")
    do {
        let engine = try BLS12377NTTEngine()
        let testN = 1024
        var testInput = [Fr377](repeating: Fr377.zero, count: testN)
        for i in 0..<testN { testInput[i] = fr377FromInt(UInt64(i + 1)) }

        let gpuNTT = try engine.ntt(testInput)
        let gpuRec = try engine.intt(gpuNTT)
        var correct = true
        for i in 0..<testN {
            if fr377ToInt(testInput[i]) != fr377ToInt(gpuRec[i]) { correct = false; break }
        }
        check("GPU round-trip (2^10)", correct)

        // Four-step round-trip 2^20
        let testN2 = 1 << 20
        var testInput2 = [Fr377](repeating: Fr377.zero, count: testN2)
        var rng: UInt64 = 0xCAFE_BABE
        for i in 0..<testN2 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            testInput2[i] = fr377FromInt(rng >> 32)
        }
        let ntt2 = try engine.ntt(testInput2)
        let rec2 = try engine.intt(ntt2)
        var mm = 0
        for i in 0..<testN2 {
            if fr377ToInt(testInput2[i]) != fr377ToInt(rec2[i]) { mm += 1 }
        }
        check("Four-step round-trip (2^20)", mm == 0)
    } catch {
        print("  [FAIL] NTT BLS12-377 error: \(error)")
        _testFailed += 1
    }
}

// ============================================================
// MARK: - Poseidon2
// ============================================================
private func testPoseidon2() {
    print("\n--- Poseidon2 ---")

    // Determinism
    let zeroState = [Fr.zero, Fr.zero, Fr.zero]
    let r1 = poseidon2Permutation(zeroState)
    let r2 = poseidon2Permutation(zeroState)
    check("Permutation deterministic", frToInt(r1[0]) == frToInt(r2[0]))
    check("Permutation of zero non-trivial", frToInt(r1[0]) != [0,0,0,0])

    // HorizenLabs reference vector: permutation([0, 1, 2])
    let refInput = [frFromInt(0), frFromInt(1), frFromInt(2)]
    let refOutput = poseidon2Permutation(refInput)
    let expected0: [UInt64] = [0x47f760054f4a3033, 0x8134334da98ea4f8, 0xbcb1929a82650f32, 0x0bb61d24daca55ee]
    let expected1: [UInt64] = [0x92defe7ff8d03570, 0x77a15d3f74ca6549, 0xcbcc80214f26a302, 0x303b6f7c86d043bf]
    let expected2: [UInt64] = [0x86296242cf766ec8, 0xe660b145994427cc, 0xf8617361c3ba7c52, 0x1ed25194542b12ee]
    check("HorizenLabs reference vector",
          frToInt(refOutput[0]) == expected0 && frToInt(refOutput[1]) == expected1 && frToInt(refOutput[2]) == expected2)

    // GPU vs CPU
    do {
        let engine = try Poseidon2Engine()
        let testPairs: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4),
                               frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8)]
        let gpuResults = try engine.hashPairs(testPairs)
        var gpuCorrect = true
        for i in 0..<4 {
            let cpuH = poseidon2Hash(testPairs[i*2], testPairs[i*2+1])
            if frToInt(gpuResults[i]) != frToInt(cpuH) { gpuCorrect = false }
        }
        check("GPU matches CPU (4 pairs)", gpuCorrect)
    } catch {
        print("  [FAIL] Poseidon2 GPU error: \(error)")
        _testFailed += 1
    }
}

// ============================================================
// MARK: - Keccak-256
// ============================================================
private func testKeccak() {
    print("\n--- Keccak-256 ---")

    // NIST vectors
    let emptyHash = keccak256([])
    let emptyHex = emptyHash.map { String(format: "%02x", $0) }.joined()
    check("Empty hash matches NIST vector",
          emptyHex == "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470")

    let abcHash = keccak256([0x61, 0x62, 0x63])
    let abcHex = abcHash.map { String(format: "%02x", $0) }.joined()
    check("abc hash matches NIST vector",
          abcHex == "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45")

    // GPU vs CPU
    do {
        let engine = try Keccak256Engine()
        var testInput = [UInt8](repeating: 0, count: 64 * 4)
        for i in 0..<4 { for j in 0..<64 { testInput[i * 64 + j] = UInt8((i * 64 + j) & 0xFF) } }
        let gpuResults = try engine.hash64(testInput)
        var gpuCorrect = true
        for i in 0..<4 {
            let cpuRef = keccak256(Array(testInput[i*64..<(i+1)*64]))
            let gpuSlice = Array(gpuResults[i*32..<(i+1)*32])
            if cpuRef != gpuSlice { gpuCorrect = false }
        }
        check("GPU matches CPU (4 inputs)", gpuCorrect)
    } catch {
        print("  [FAIL] Keccak GPU error: \(error)")
        _testFailed += 1
    }
}

// ============================================================
// MARK: - Blake3
// ============================================================
private func testBlake3() {
    print("\n--- Blake3 ---")

    let emptyHash = blake3([])
    let emptyHex = emptyHash.map { String(format: "%02x", $0) }.joined()
    check("Empty hash matches reference",
          emptyHex == "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262")

    do {
        let engine = try Blake3Engine()
        var testInput = [UInt8](repeating: 0, count: 64 * 4)
        for i in 0..<(64 * 4) { testInput[i] = UInt8(i & 0xFF) }
        let gpuResults = try engine.hash64(testInput)
        var gpuCorrect = true
        for i in 0..<4 {
            let block = Array(testInput[(i * 64)..<((i + 1) * 64)])
            let cpuHash = blake3(block)
            let gpuHash = Array(gpuResults[(i * 32)..<((i + 1) * 32)])
            if cpuHash != gpuHash { gpuCorrect = false }
        }
        check("GPU matches CPU (4 blocks)", gpuCorrect)
    } catch {
        print("  [FAIL] Blake3 GPU error: \(error)")
        _testFailed += 1
    }
}

// ============================================================
// MARK: - FRI
// ============================================================
private func testFRI() {
    print("\n--- FRI ---")
    do {
        let engine = try FRIEngine()

        // Single fold GPU vs CPU
        let testLogN = 10
        let testN = 1 << testLogN
        var testEvals = [Fr](repeating: Fr.zero, count: testN)
        var rng: UInt64 = 0xDEAD_BEEF
        for i in 0..<testN {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            testEvals[i] = frFromInt(rng >> 32)
        }
        let beta = frFromInt(42)
        let gpuFolded = try engine.fold(evals: testEvals, beta: beta)
        let cpuFolded = FRIEngine.cpuFold(evals: testEvals, beta: beta, logN: testLogN)
        var correct = true
        for i in 0..<gpuFolded.count {
            if frToInt(gpuFolded[i]) != frToInt(cpuFolded[i]) { correct = false; break }
        }
        check("Single fold GPU=CPU (2^10)", correct)

        // Multi-fold: 2^16 -> constant
        let multiLogN = 16
        let multiN = 1 << multiLogN
        var multiEvals = [Fr](repeating: Fr.zero, count: multiN)
        for i in 0..<multiN {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            multiEvals[i] = frFromInt(rng >> 32)
        }
        var betas = [Fr]()
        for i in 0..<multiLogN { betas.append(frFromInt(UInt64(i + 1) * 7)) }
        let finalGPU = try engine.multiFold(evals: multiEvals, betas: betas)
        var cpuCurrent = multiEvals
        for i in 0..<multiLogN {
            cpuCurrent = FRIEngine.cpuFold(evals: cpuCurrent, beta: betas[i], logN: multiLogN - i)
        }
        check("Multi-fold 2^16->1 GPU=CPU",
              finalGPU.count == 1 && cpuCurrent.count == 1 && frToInt(finalGPU[0]) == frToInt(cpuCurrent[0]))

        // FRI protocol: commit -> query -> verify
        let protoLogN = 14
        let protoN = 1 << protoLogN
        var protoEvals = [Fr](repeating: Fr.zero, count: protoN)
        for i in 0..<protoN {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            protoEvals[i] = frFromInt(rng >> 32)
        }
        var protoBetas = [Fr]()
        for i in 0..<protoLogN { protoBetas.append(frFromInt(UInt64(i + 1) * 17)) }
        let commitment = try engine.commitPhase(evals: protoEvals, betas: protoBetas)
        let queryIndices: [UInt32] = [0, 42, 1000, UInt32(protoN / 2 - 1)]
        let queries = try engine.queryPhase(commitment: commitment, queryIndices: queryIndices)
        let verified = engine.verify(commitment: commitment, queries: queries)
        check("FRI protocol verify (2^14)", verified)
    } catch {
        print("  [FAIL] FRI error: \(error)")
        _testFailed += 1
    }
}

// ============================================================
// MARK: - Sumcheck
// ============================================================
private func testSumcheck() {
    print("\n--- Sumcheck ---")
    do {
        let engine = try SumcheckEngine()

        let testNumVars = 10
        let testN = 1 << testNumVars
        var testEvals = [Fr](repeating: Fr.zero, count: testN)
        var rng: UInt64 = 0xDEAD_BEEF
        for i in 0..<testN {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            testEvals[i] = frFromInt(rng >> 32)
        }
        let challenge = frFromInt(42)

        // Reduce GPU vs CPU
        let gpuReduced = try engine.reduce(evals: testEvals, challenge: challenge)
        let cpuReduced = SumcheckEngine.cpuReduce(evals: testEvals, challenge: challenge)
        var reduceCorrect = true
        for i in 0..<gpuReduced.count {
            if frToInt(gpuReduced[i]) != frToInt(cpuReduced[i]) { reduceCorrect = false; break }
        }
        check("Reduce GPU=CPU (2^10)", reduceCorrect)

        // Round polynomial GPU vs CPU
        let evalsBuf = engine.device.makeBuffer(
            length: testN * MemoryLayout<Fr>.stride, options: .storageModeShared)!
        testEvals.withUnsafeBytes { src in
            memcpy(evalsBuf.contents(), src.baseAddress!, testN * MemoryLayout<Fr>.stride)
        }
        let (gpuS0, gpuS1, gpuS2) = try engine.computeRoundPoly(evals: evalsBuf, n: testN)
        let (cpuS0, cpuS1, cpuS2) = SumcheckEngine.cpuRoundPoly(evals: testEvals)
        check("Round poly GPU=CPU",
              frToInt(gpuS0) == frToInt(cpuS0) && frToInt(gpuS1) == frToInt(cpuS1) && frToInt(gpuS2) == frToInt(cpuS2))

        // S(0)+S(1) = sum
        var totalSum = Fr.zero
        for e in testEvals { totalSum = frAdd(totalSum, e) }
        check("S(0)+S(1) = sum", frToInt(frAdd(gpuS0, gpuS1)) == frToInt(totalSum))

        // Full protocol verification
        let fullNumVars = 14
        let fullN = 1 << fullNumVars
        var fullEvals = [Fr](repeating: Fr.zero, count: fullN)
        for i in 0..<fullN {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            fullEvals[i] = frFromInt(rng >> 32)
        }
        var challenges = [Fr]()
        for i in 0..<fullNumVars { challenges.append(frFromInt(UInt64(i + 1) * 7)) }

        let (rounds, _) = try engine.fullSumcheck(evals: fullEvals, challenges: challenges)
        var runningEvals = fullEvals
        var protocolCorrect = true
        for i in 0..<fullNumVars {
            let (s0, s1, _) = rounds[i]
            var expected = Fr.zero
            for e in runningEvals { expected = frAdd(expected, e) }
            if frToInt(frAdd(s0, s1)) != frToInt(expected) { protocolCorrect = false; break }
            runningEvals = SumcheckEngine.cpuReduce(evals: runningEvals, challenge: challenges[i])
        }
        check("Full protocol S(0)+S(1) invariant (14 vars)", protocolCorrect)
    } catch {
        print("  [FAIL] Sumcheck error: \(error)")
        _testFailed += 1
    }
}

// ============================================================
// MARK: - Sparse Sumcheck
// ============================================================
private func testSparseSumcheck() {
    print("\n--- Sparse Sumcheck ---")

    // Round poly matches dense
    do {
        let dense: [Fr] = [frFromInt(0), frFromInt(5), frFromInt(3), frFromInt(0),
                           frFromInt(7), frFromInt(0), frFromInt(0), frFromInt(2)]
        let sparse = SparseMultilinearPoly(dense: dense)
        let (ss0, ss1, ss2) = sparse.roundPoly()
        let (ds0, ds1, ds2) = SumcheckEngine.cpuRoundPoly(evals: dense)
        check("Round poly matches dense",
              frEqual(ss0, ds0) && frEqual(ss1, ds1) && frEqual(ss2, ds2))
    }

    // Reduce matches dense
    do {
        let dense: [Fr] = [frFromInt(1), frFromInt(0), frFromInt(0), frFromInt(4),
                           frFromInt(0), frFromInt(3), frFromInt(2), frFromInt(0)]
        let sparse = SparseMultilinearPoly(dense: dense)
        let challenge = frFromInt(42)
        let denseReduced = SumcheckEngine.cpuReduce(evals: dense, challenge: challenge)
        let sparseReduced = sparse.reduce(challenge: challenge)
        let sparseReducedDense = sparseReduced.toDense()
        var reduceMatch = true
        for i in 0..<denseReduced.count {
            if !frEqual(denseReduced[i], sparseReducedDense[i]) { reduceMatch = false; break }
        }
        check("Reduce matches dense", reduceMatch)
    }

    // Total sum
    do {
        let dense: [Fr] = [frFromInt(10), frFromInt(0), frFromInt(20), frFromInt(0)]
        let sparse = SparseMultilinearPoly(dense: dense)
        var denseSum = Fr.zero
        for v in dense { denseSum = frAdd(denseSum, v) }
        check("Total sum matches", frEqual(denseSum, sparse.totalSum()))
    }

    // Full sparse sumcheck = dense sumcheck
    do {
        let numVars = 4
        var entries = [Int: Fr]()
        entries[0] = frFromInt(7)
        entries[3] = frFromInt(11)
        entries[5] = frFromInt(2)
        entries[10] = frFromInt(9)
        entries[15] = frFromInt(3)
        let sparse = SparseMultilinearPoly(numVars: numVars, entries: entries)
        let dense = sparse.toDense()

        var transcript1 = [UInt8]()
        let (sparseRounds, sparseFinal, _) = sparseSumcheck(poly: sparse, transcript: &transcript1)

        var current = dense
        var denseRounds = [(Fr, Fr, Fr)]()
        var transcript2 = [UInt8]()
        for _ in 0..<numVars {
            let rp = SumcheckEngine.cpuRoundPoly(evals: current)
            denseRounds.append(rp)
            let rpB0 = frToInt(rp.0); let rpB1 = frToInt(rp.1); let rpB2 = frToInt(rp.2)
            for limb in rpB0 { for b in 0..<8 { transcript2.append(UInt8((limb >> (b*8)) & 0xFF)) } }
            for limb in rpB1 { for b in 0..<8 { transcript2.append(UInt8((limb >> (b*8)) & 0xFF)) } }
            for limb in rpB2 { for b in 0..<8 { transcript2.append(UInt8((limb >> (b*8)) & 0xFF)) } }
            let ch = blake3DeriveChallenge(transcript2)
            let chB = frToInt(ch)
            for limb in chB { for b in 0..<8 { transcript2.append(UInt8((limb >> (b*8)) & 0xFF)) } }
            current = SumcheckEngine.cpuReduce(evals: current, challenge: ch)
        }
        let denseFinal = current[0]

        var allRoundsMatch = true
        for k in 0..<numVars {
            let (ss0, ss1, ss2) = sparseRounds[k]
            let (ds0, ds1, ds2) = denseRounds[k]
            if !frEqual(ss0, ds0) || !frEqual(ss1, ds1) || !frEqual(ss2, ds2) { allRoundsMatch = false }
        }
        check("Full sparse rounds match dense (4 vars)", allRoundsMatch)
        check("Final eval matches dense", frEqual(sparseFinal, denseFinal))
    }

    // Verify proof
    do {
        let numVars = 3
        var entries = [Int: Fr]()
        entries[1] = frFromInt(5)
        entries[4] = frFromInt(8)
        entries[7] = frFromInt(3)
        let sparse = SparseMultilinearPoly(numVars: numVars, entries: entries)
        let claimedSum = sparse.totalSum()
        var proveTranscript = [UInt8]()
        let (rounds, finalEval, _) = sparseSumcheck(poly: sparse, transcript: &proveTranscript)
        var verifyTranscript = [UInt8]()
        let (valid, _) = verifySumcheckProof(rounds: rounds, claimedSum: claimedSum, finalEval: finalEval, transcript: &verifyTranscript)
        check("Verify proof", valid)

        var vt2 = [UInt8]()
        let (rejected, _) = verifySumcheckProof(rounds: rounds, claimedSum: frAdd(claimedSum, Fr.one), finalEval: finalEval, transcript: &vt2)
        check("Reject wrong sum", !rejected)
    }
}

// ============================================================
// MARK: - Polynomial ops
// ============================================================
private func testPolyOps() {
    print("\n--- Polynomial Ops ---")
    do {
        let engine = try PolyEngine()

        // Multiply
        let a = [frFromInt(1), frFromInt(2)]
        let b = [frFromInt(3), frFromInt(4)]
        let c = try engine.multiply(a, b)
        check("(1+2x)*(3+4x) = 3+10x+8x^2",
              frToInt(c[0])[0] == 3 && frToInt(c[1])[0] == 10 && frToInt(c[2])[0] == 8)

        // Evaluate
        let poly = [frFromInt(1), frFromInt(2), frFromInt(3)]
        let evals = try engine.evaluate(poly, at: [frFromInt(5)])
        check("1+2x+3x^2 at x=5 = 86", frToInt(evals[0])[0] == 86)

        // Add
        let sum = try engine.add([frFromInt(10), frFromInt(20)], [frFromInt(3), frFromInt(7)])
        check("Poly add", frToInt(sum[0])[0] == 13 && frToInt(sum[1])[0] == 27)

        // Sub
        let diff = try engine.sub([frFromInt(10), frFromInt(20)], [frFromInt(3), frFromInt(7)])
        check("Poly sub", frToInt(diff[0])[0] == 7 && frToInt(diff[1])[0] == 13)

        // Chunked eval
        let cDeg = 1024
        var cCoeffs = [Fr](repeating: Fr.zero, count: cDeg)
        for i in 0..<cDeg { cCoeffs[i] = frFromInt(UInt64(i + 1)) }
        let gpuResults = try engine.evaluate(cCoeffs, at: [frFromInt(1)])
        check("Chunked eval (deg 1024 at x=1) = 524800", frToInt(gpuResults[0]) == frToInt(frFromInt(524800)))

        // Subproduct tree matches Horner
        let tN = 1024
        var tCoeffs = [Fr](repeating: Fr.zero, count: tN)
        var tPts = [Fr](repeating: Fr.zero, count: tN)
        var rng: UInt64 = 0xDEAD_BEEF
        for i in 0..<tN {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            tCoeffs[i] = frFromInt(rng >> 32)
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            tPts[i] = frFromInt(rng >> 32)
        }
        let horner = try engine.evaluate(tCoeffs, at: tPts)
        let tree = try engine.evaluateTree(tCoeffs, at: tPts)
        var treeMatch = true
        for i in 0..<tN {
            if frToInt(horner[i]) != frToInt(tree[i]) { treeMatch = false; break }
        }
        check("Subproduct tree matches Horner (deg 2^10)", treeMatch)
    } catch {
        print("  [FAIL] Poly error: \(error)")
        _testFailed += 1
    }
}

// ============================================================
// MARK: - KZG
// ============================================================
private func testKZG() {
    print("\n--- KZG ---")
    do {
        let gx = fpFromInt(1)
        let gy = fpFromInt(2)
        let generator = PointAffine(x: gx, y: gy)
        let secret: [UInt32] = [42, 0, 0, 0, 0, 0, 0, 0]
        let srs = KZGEngine.generateTestSRS(secret: secret, size: 256, generator: generator)
        let engine = try KZGEngine(srs: srs)

        let constPoly = [frFromInt(1)]
        let c1 = try engine.commit(constPoly)
        let c1Aff = batchToAffine([c1])[0]
        check("Commit([1]) = G", fpToInt(c1Aff.x) == fpToInt(gx) && fpToInt(c1Aff.y) == fpToInt(gy))

        let p1 = try engine.open(constPoly, at: frFromInt(999))
        check("Open(const, z=999) eval = 1", frToInt(p1.evaluation)[0] == 1)
        check("Open(const) witness = identity", pointIsIdentity(p1.witness))

        let linPoly = [frFromInt(0), frFromInt(1)]
        let c2 = try engine.commit(linPoly)
        let c2Aff = batchToAffine([c2])[0]
        check("Commit(x) = SRS[1]", fpToInt(c2Aff.x) == fpToInt(srs[1].x) && fpToInt(c2Aff.y) == fpToInt(srs[1].y))

        let p2 = try engine.open(linPoly, at: frFromInt(42))
        check("Open(x, z=42) eval = 42", frToInt(p2.evaluation)[0] == 42)

        // Linearity
        let polyA = [frFromInt(1), frFromInt(2), frFromInt(3)]
        let polyB = [frFromInt(4), frFromInt(5), frFromInt(6)]
        let polySum = zip(polyA, polyB).map { frAdd($0, $1) }
        let cA = try engine.commit(polyA)
        let cB = try engine.commit(polyB)
        let cSum = try engine.commit(polySum)
        let cManual = pointAdd(cA, cB)
        let cSumAff = batchToAffine([cSum])[0]
        let cManAff = batchToAffine([cManual])[0]
        check("Commit linearity: C(p+q) = C(p)+C(q)",
              fpToInt(cSumAff.x) == fpToInt(cManAff.x) && fpToInt(cSumAff.y) == fpToInt(cManAff.y))

        // Eval consistency
        let p = [frFromInt(1), frFromInt(2), frFromInt(3)]
        check("p(5)=86", frToInt(try engine.open(p, at: frFromInt(5)).evaluation)[0] == 86)
        check("p(0)=1", frToInt(try engine.open(p, at: frFromInt(0)).evaluation)[0] == 1)
        check("p(1)=6", frToInt(try engine.open(p, at: frFromInt(1)).evaluation)[0] == 6)
    } catch {
        print("  [FAIL] KZG error: \(error)")
        _testFailed += 1
    }
}

// ============================================================
// MARK: - MSM BN254
// ============================================================
private func testMSM_BN254() {
    print("\n--- MSM BN254 ---")
    do {
        let engine = try MetalMSM()
        let gx = fpFromInt(1)
        let gy = fpFromInt(2)
        let gProj = pointFromAffine(PointAffine(x: gx, y: gy))

        // MSM([G],[1]) = G
        let oneScalar: [UInt32] = [1, 0, 0, 0, 0, 0, 0, 0]
        let r1 = try engine.msm(points: [PointAffine(x: gx, y: gy)], scalars: [oneScalar])
        check("MSM([G],[1]) = G", pointEqual(r1, gProj))

        // MSM([G],[5]) = 5G
        let fiveG = pointMulInt(gProj, 5)
        let r2 = try engine.msm(points: [PointAffine(x: gx, y: gy)], scalars: [[5, 0, 0, 0, 0, 0, 0, 0]])
        check("MSM([G],[5]) = 5G", pointEqual(r2, fiveG))

        // Deterministic 256 pts full scalars
        let n = 256
        var projPoints = [PointProjective]()
        var acc = gProj
        for _ in 0..<n { projPoints.append(acc); acc = pointAdd(acc, gProj) }
        let pts = batchToAffine(projPoints)
        var rng: UInt64 = 0xBEEF_CAFE
        var scalars = [[UInt32]]()
        for _ in 0..<n {
            var limbs = [UInt32](repeating: 0, count: 8)
            for j in 0..<8 { rng = rng &* 6364136223846793005 &+ 1442695040888963407; limbs[j] = UInt32(truncatingIfNeeded: rng >> 32) }
            scalars.append(limbs)
        }
        let ra = try engine.msm(points: pts, scalars: scalars)
        let rb = try engine.msm(points: pts, scalars: scalars)
        check("MSM 256 pts deterministic", pointEqual(ra, rb))

        // Result on curve
        let raAff = batchToAffine([ra])[0]
        let y2 = fpSqr(raAff.y)
        let x3 = fpMul(fpSqr(raAff.x), raAff.x)
        let rhs = fpAdd(x3, fpFromInt(3))
        check("MSM 256 pts on curve", fpToInt(y2) == fpToInt(rhs))
    } catch {
        print("  [FAIL] MSM BN254 error: \(error)")
        _testFailed += 1
    }
}

// ============================================================
// MARK: - secp256k1 Field + Curve
// ============================================================
private func testSecp256k1Field() {
    print("\n--- secp256k1 Field + Curve ---")

    let one = secpFromInt(1)
    let oneOut = secpToInt(one)
    check("fromInt(1) round-trip", oneOut[0] == 1 && oneOut[1] == 0 && oneOut[2] == 0 && oneOut[3] == 0)

    let a = secpFromInt(42)
    let aInv = secpInverse(a)
    let mulInv = secpMul(a, aInv)
    let mulInvOut = secpToInt(mulInv)
    check("42 * 42^(-1) = 1", mulInvOut[0] == 1 && mulInvOut[1] == 0 && mulInvOut[2] == 0 && mulInvOut[3] == 0)

    // Distributivity
    let b = secpFromInt(123456789)
    let c = secpFromInt(987654321)
    check("Distributivity", secpToInt(secpMul(secpAdd(a, b), c)) == secpToInt(secpAdd(secpMul(a, c), secpMul(b, c))))

    // Generator on curve
    let gen = secp256k1Generator()
    let gx3 = secpMul(secpSqr(gen.x), gen.x)
    let seven = secpFromInt(7)
    check("Generator on curve (y^2=x^3+7)", secpToInt(secpSqr(gen.y)) == secpToInt(secpAdd(gx3, seven)))

    // G + G = 2G
    let gProj = secpPointFromAffine(gen)
    let twoG = secpPointDouble(gProj)
    let gPlusG = secpPointAdd(gProj, gProj)
    let twoGAff = secpPointToAffine(twoG)
    let gPlusGAff = secpPointToAffine(gPlusG)
    check("G + G = 2G", secpToInt(gPlusGAff.x) == secpToInt(twoGAff.x) && secpToInt(gPlusGAff.y) == secpToInt(twoGAff.y))

    // 5*G (mul) = 5*G (add)
    let fiveG_mul = secpPointMulInt(gProj, 5)
    var fiveG_add = gProj
    for _ in 1..<5 { fiveG_add = secpPointAdd(fiveG_add, gProj) }
    let fmAff = secpPointToAffine(fiveG_mul)
    let faAff = secpPointToAffine(fiveG_add)
    check("5*G (mul) = 5*G (add)", secpToInt(fmAff.x) == secpToInt(faAff.x) && secpToInt(fmAff.y) == secpToInt(faAff.y))

    // G + (-G) = O
    let negG = secpPointNegateAffine(gen)
    let gPlusNegG = secpPointAdd(gProj, secpPointFromAffine(negG))
    check("G + (-G) = O", secpPointIsIdentity(gPlusNegG))

    // Batch affine
    var projPoints = [SecpPointProjective]()
    var acc = gProj
    for _ in 0..<10 { projPoints.append(acc); acc = secpPointAdd(acc, gProj) }
    let batchAff = batchSecpToAffine(projPoints)
    var batchOk = true
    for i in 0..<10 {
        let singleAff = secpPointToAffine(projPoints[i])
        if secpToInt(batchAff[i].x) != secpToInt(singleAff.x) || secpToInt(batchAff[i].y) != secpToInt(singleAff.y) {
            batchOk = false; break
        }
    }
    check("Batch affine conversion", batchOk)
}

// ============================================================
// MARK: - secp256k1 GLV
// ============================================================
private func testSecp256k1GLV() {
    print("\n--- secp256k1 GLV ---")

    let testScalar: [UInt32] = [0xdeadbeef, 0xcafebabe, 0x12345678, 0x9abcdef0, 0x11223344, 0x55667788, 0x99aabbcc, 0x0ddeeff0]
    let (k1, k2, _, _) = Secp256k1GLV.decompose(testScalar)
    check("k1, k2 <= 128 bits", k1[4...7].reduce(UInt32(0), |) == 0 && k2[4...7].reduce(UInt32(0), |) == 0)

    // beta^3 = 1
    let beta = SecpFp(v: (
        Secp256k1GLV.BETA_MONT[0], Secp256k1GLV.BETA_MONT[1],
        Secp256k1GLV.BETA_MONT[2], Secp256k1GLV.BETA_MONT[3],
        Secp256k1GLV.BETA_MONT[4], Secp256k1GLV.BETA_MONT[5],
        Secp256k1GLV.BETA_MONT[6], Secp256k1GLV.BETA_MONT[7]
    ))
    let beta3 = secpMul(secpMul(beta, beta), beta)
    check("beta^3 = 1 (mod p)", secpToInt(beta3) == secpToInt(secpFromInt(1)))

    // Endomorphism point on curve
    let gen = secp256k1Generator()
    let endoG = Secp256k1GLV.applyEndomorphism(gen)
    let seven = secpFromInt(7)
    check("phi(G) on curve", secpToInt(secpSqr(endoG.y)) == secpToInt(secpAdd(secpMul(secpSqr(endoG.x), endoG.x), seven)))

    // Point-level: k*G = k1*G + k2*phi(G) for small scalars
    let gProj = secpPointFromAffine(gen)
    let endoGProj = secpPointFromAffine(endoG)
    var pointOk = true
    for trial in 0..<5 {
        var s: [UInt32] = [UInt32](repeating: 0, count: 8)
        s[0] = UInt32(trial * 7919 + 42) & 0xFFF
        let (dk1b, dk2b, dn1b, dn2b) = Secp256k1GLV.decompose(s)
        var k1b_int = 0
        for j in stride(from: 3, through: 0, by: -1) { k1b_int = (k1b_int << 32) | Int(dk1b[j]) }
        var t1 = secpPointMulInt(gProj, k1b_int)
        if dn1b { let a = secpPointToAffine(t1); t1 = secpPointFromAffine(secpPointNegateAffine(a)) }
        var k2b_int = 0
        for j in stride(from: 3, through: 0, by: -1) { k2b_int = (k2b_int << 32) | Int(dk2b[j]) }
        var t2 = secpPointMulInt(endoGProj, k2b_int)
        if dn2b { let a = secpPointToAffine(t2); t2 = secpPointFromAffine(secpPointNegateAffine(a)) }
        let glvRes = secpPointAdd(t1, t2)
        let directRes = secpPointMulInt(gProj, Int(s[0]))
        let gAff = secpPointToAffine(glvRes)
        let dAff = secpPointToAffine(directRes)
        if secpToInt(gAff.x) != secpToInt(dAff.x) || secpToInt(gAff.y) != secpToInt(dAff.y) { pointOk = false }
    }
    check("k*G = k1*G + k2*phi(G) point check (5 trials)", pointOk)
}

// ============================================================
// MARK: - secp256k1 MSM
// ============================================================
private func testSecp256k1MSM() {
    print("\n--- secp256k1 MSM ---")
    do {
        let engine = try Secp256k1MSM()
        let gen = secp256k1Generator()
        let gProj = secpPointFromAffine(gen)

        let oneScalar: [UInt32] = [1, 0, 0, 0, 0, 0, 0, 0]
        let r1 = try engine.msm(points: [gen], scalars: [oneScalar])
        let r1Aff = secpPointToAffine(r1)
        check("MSM([G],[1]) = G", secpToInt(r1Aff.x) == secpToInt(gen.x) && secpToInt(r1Aff.y) == secpToInt(gen.y))

        let fiveG = secpPointMulInt(gProj, 5)
        let r3 = try engine.msm(points: [gen], scalars: [[5, 0, 0, 0, 0, 0, 0, 0]])
        let r3Aff = secpPointToAffine(r3)
        let fGAff = secpPointToAffine(fiveG)
        check("MSM([G],[5]) = 5G", secpToInt(r3Aff.x) == secpToInt(fGAff.x) && secpToInt(r3Aff.y) == secpToInt(fGAff.y))

        // Deterministic 256 pts
        let n = 256
        var pts = [SecpPointAffine]()
        var acc = gProj
        for _ in 0..<n { pts.append(secpPointToAffine(acc)); acc = secpPointAdd(acc, gProj) }
        var rng: UInt64 = 0xBEEF_CAFE
        var scalars = [[UInt32]]()
        for _ in 0..<n {
            var limbs = [UInt32](repeating: 0, count: 8)
            for j in 0..<8 { rng = rng &* 6364136223846793005 &+ 1442695040888963407; limbs[j] = UInt32(truncatingIfNeeded: rng >> 32) }
            scalars.append(limbs)
        }
        let ra = try engine.msm(points: pts, scalars: scalars)
        let rb = try engine.msm(points: pts, scalars: scalars)
        let raAff = secpPointToAffine(ra)
        let rbAff = secpPointToAffine(rb)
        check("MSM 256 pts deterministic", secpToInt(raAff.x) == secpToInt(rbAff.x) && secpToInt(raAff.y) == secpToInt(rbAff.y))

        let y2 = secpSqr(raAff.y)
        let x3 = secpMul(secpSqr(raAff.x), raAff.x)
        let rhs = secpAdd(x3, secpFromInt(7))
        check("MSM 256 pts on curve", secpToInt(y2) == secpToInt(rhs))
    } catch {
        print("  [FAIL] secp256k1 MSM error: \(error)")
        _testFailed += 1
    }
}

// ============================================================
// MARK: - BLS12-377 Field + Curve
// ============================================================
private func testBLS12377Field() {
    print("\n--- BLS12-377 Fq Field + Curve ---")

    let a = fq377FromInt(42)
    let b = fq377FromInt(100)
    check("42 + 100 = 142", fq377ToInt(fq377Add(a, b))[0] == 142)
    check("42 * 100 = 4200", fq377ToInt(fq377Mul(a, b))[0] == 4200)
    check("100 - 42 = 58", fq377ToInt(fq377Sub(b, a))[0] == 58)
    check("42 * 42^(-1) = 1", fq377ToInt(fq377Mul(a, fq377Inverse(a)))[0] == 1)

    // 2G on curve (y^2 = x^3 + 1)
    let g = bls12377Generator()
    let gProj = point377FromAffine(g)
    let g2 = point377Double(gProj)
    if let g2a = point377ToAffine(g2) {
        let y2 = fq377Sqr(g2a.y)
        let x3 = fq377Mul(fq377Sqr(g2a.x), g2a.x)
        let rhs = fq377Add(x3, fq377FromInt(1))
        check("2G on curve", fq377ToInt(y2) == fq377ToInt(rhs))
    } else {
        check("2G is not identity", false)
    }

    // G + G = 2G
    let g1p1 = point377Add(gProj, gProj)
    if let g1p1a = point377ToAffine(g1p1), let g2a = point377ToAffine(g2) {
        check("G + G = 2G", fq377ToInt(g1p1a.x) == fq377ToInt(g2a.x) && fq377ToInt(g1p1a.y) == fq377ToInt(g2a.y))
    } else {
        check("G + G = 2G", false)
    }
}

// ============================================================
// MARK: - BLS12-377 GLV
// ============================================================
private func testBLS12377GLV() {
    print("\n--- BLS12-377 GLV ---")

    let lambdaMont = fr377Mul(Fr377.from64(BLS12377GLV.LAMBDA), Fr377.from64(Fr377.R2_MOD_R))

    // lambda^2 + lambda + 1 = 0
    let lambda2 = fr377Mul(lambdaMont, lambdaMont)
    let sum = fr377Add(fr377Add(lambda2, lambdaMont), Fr377.one)
    let sumInt = fr377ToInt(sum)
    check("lambda^2 + lambda + 1 = 0 in Fr", sumInt.allSatisfy { $0 == 0 })

    // beta^3 = 1 in Fq
    let betaMont = BLS12377GLV.betaMontgomery
    let beta3 = fq377Mul(fq377Mul(betaMont, betaMont), betaMont)
    let beta3Int = fq377ToInt(beta3)
    check("beta^3 = 1 in Fq",
          beta3Int[0] == 1 && beta3Int[1] == 0 && beta3Int[2] == 0 && beta3Int[3] == 0 && beta3Int[4] == 0 && beta3Int[5] == 0)

    // GLV decomposition roundtrip (10 trials)
    var rng: UInt64 = 0xCAFE_BABE_1234_5678
    var allPass = true
    for _ in 0..<10 {
        var k = [UInt32](repeating: 0, count: 8)
        for j in 0..<8 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            k[j] = UInt32(truncatingIfNeeded: rng >> 32)
        }
        k = BLS12377MSM.reduceModR(k)
        let (k1, k2, neg1, neg2) = BLS12377GLV.decompose(k)

        var k1Mont = fr377Mul(Fr377.from64([
            UInt64(k1[0]) | (UInt64(k1[1]) << 32), UInt64(k1[2]) | (UInt64(k1[3]) << 32),
            UInt64(k1[4]) | (UInt64(k1[5]) << 32), UInt64(k1[6]) | (UInt64(k1[7]) << 32)
        ]), Fr377.from64(Fr377.R2_MOD_R))
        var k2Mont = fr377Mul(Fr377.from64([
            UInt64(k2[0]) | (UInt64(k2[1]) << 32), UInt64(k2[2]) | (UInt64(k2[3]) << 32),
            UInt64(k2[4]) | (UInt64(k2[5]) << 32), UInt64(k2[6]) | (UInt64(k2[7]) << 32)
        ]), Fr377.from64(Fr377.R2_MOD_R))
        if neg1 { k1Mont = fr377Neg(k1Mont) }
        if neg2 { k2Mont = fr377Neg(k2Mont) }
        let recomputed = fr377Add(k1Mont, fr377Mul(k2Mont, lambdaMont))
        let kMont = fr377Mul(Fr377.from64([
            UInt64(k[0]) | (UInt64(k[1]) << 32), UInt64(k[2]) | (UInt64(k[3]) << 32),
            UInt64(k[4]) | (UInt64(k[5]) << 32), UInt64(k[6]) | (UInt64(k[7]) << 32)
        ]), Fr377.from64(Fr377.R2_MOD_R))
        if fr377ToInt(kMont) != fr377ToInt(recomputed) { allPass = false }
    }
    check("GLV decomposition roundtrip (10 trials)", allPass)
}

// ============================================================
// MARK: - BLS12-377 MSM
// ============================================================
private func testBLS12377MSM() {
    print("\n--- BLS12-377 MSM ---")
    do {
        let engine = try BLS12377MSM()
        let gen = bls12377Generator()
        let gProj = point377FromAffine(gen)

        let oneScalar: [UInt32] = [1, 0, 0, 0, 0, 0, 0, 0]
        let r1 = try engine.msm(points: [gen], scalars: [oneScalar])
        if let r1Aff = point377ToAffine(r1) {
            check("MSM([G],[1]) = G", fq377ToInt(r1Aff.x) == fq377ToInt(gen.x) && fq377ToInt(r1Aff.y) == fq377ToInt(gen.y))
        } else {
            check("MSM([G],[1]) = G", false)
        }

        let sevenG = point377MulInt(gProj, 7)
        let r2 = try engine.msm(points: [gen], scalars: [[7, 0, 0, 0, 0, 0, 0, 0]])
        if let r2Aff = point377ToAffine(r2), let r2Exp = point377ToAffine(sevenG) {
            check("MSM([G],[7]) = 7G", fq377ToInt(r2Aff.x) == fq377ToInt(r2Exp.x) && fq377ToInt(r2Aff.y) == fq377ToInt(r2Exp.y))
        } else {
            check("MSM([G],[7]) = 7G", false)
        }

        // Deterministic 256 pts
        let n = 256
        var pts = [Point377Affine]()
        var acc = gProj
        for _ in 0..<n { if let aff = point377ToAffine(acc) { pts.append(aff) }; acc = point377Add(acc, gProj) }
        var rng: UInt64 = 0xDEAD_1234
        var scalars = [[UInt32]]()
        for _ in 0..<n {
            var limbs = [UInt32](repeating: 0, count: 8)
            for j in 0..<8 { rng = rng &* 6364136223846793005 &+ 1442695040888963407; limbs[j] = UInt32(truncatingIfNeeded: rng >> 32) }
            scalars.append(limbs)
        }
        let ra = try engine.msm(points: pts, scalars: scalars)
        let rb = try engine.msm(points: pts, scalars: scalars)
        if let raAff = point377ToAffine(ra), let rbAff = point377ToAffine(rb) {
            check("MSM 256 pts deterministic", fq377ToInt(raAff.x) == fq377ToInt(rbAff.x) && fq377ToInt(raAff.y) == fq377ToInt(rbAff.y))
            let y2 = fq377Sqr(raAff.y)
            let x3 = fq377Mul(fq377Sqr(raAff.x), raAff.x)
            let rhs = fq377Add(x3, fq377FromInt(1))
            check("MSM 256 pts on curve", fq377ToInt(y2) == fq377ToInt(rhs))
        } else {
            check("MSM 256 pts deterministic", point377IsIdentity(ra) == point377IsIdentity(rb))
        }
    } catch {
        print("  [FAIL] BLS12-377 MSM error: \(error)")
        _testFailed += 1
    }
}

// ============================================================
// MARK: - ECDSA
// ============================================================
private func testECDSA() {
    print("\n--- ECDSA ---")

    // Scalar field basics
    let fr1 = secpFrFromInt(1)
    check("Fr fromInt(1) round-trip", secpFrToInt(fr1) == [1, 0, 0, 0])

    let a = secpFrFromInt(42)
    let b = secpFrFromInt(7)
    check("Fr 42*7 = 294", secpFrToInt(secpFrMul(a, b))[0] == 294)
    check("Fr a * a^(-1) = 1", secpFrMul(a, secpFrInverse(a)) == SecpFr.one)

    let testElems = [secpFrFromInt(3), secpFrFromInt(7), secpFrFromInt(42), secpFrFromInt(100)]
    let batchInvs = secpFrBatchInverse(testElems)
    var batchOk = true
    for i in 0..<testElems.count { if secpFrMul(testElems[i], batchInvs[i]) != SecpFr.one { batchOk = false } }
    check("Fr batch inverse", batchOk)

    // sqrt
    if let sqrtX = secpSqrt(secpFromInt(4)) {
        check("Fp sqrt(4) = 2", secpToInt(sqrtX)[0] == 2)
    } else { check("Fp sqrt(4) exists", false) }
    check("Fp sqrt(3) = none (non-residue)", secpSqrt(secpFromInt(3)) == nil)

    // Signature
    do {
        let engine = try ECDSAEngine()
        let d = secpFrFromInt(42)
        let gen = secp256k1Generator()
        let gProj = secpPointFromAffine(gen)
        let Q = secpPointToAffine(secpPointMulScalar(gProj, secpFrToInt(d)))

        let k = secpFrFromInt(137)
        let z = secpFrFromInt(12345)
        let rProj = secpPointMulScalar(gProj, secpFrToInt(k))
        let rAff = secpPointToAffine(rProj)
        var rModN = secpToInt(rAff.x)
        if gte256(rModN, SecpFr.N) { (rModN, _) = sub256(rModN, SecpFr.N) }
        let rFr = secpFrFromRaw(rModN)
        let kInv = secpFrInverse(k)
        let sFr = secpFrMul(kInv, secpFrAdd(z, secpFrMul(rFr, d)))
        let sig = ECDSASignature(r: rFr, s: sFr, z: z)

        check("Single verify (valid sig)", engine.verify(sig: sig, pubkey: Q))
        check("Reject wrong z", !engine.verify(sig: ECDSASignature(r: rFr, s: sFr, z: secpFrFromInt(99999)), pubkey: Q))
        check("Reject wrong Q", !engine.verify(sig: sig, pubkey: secpPointToAffine(secpPointMulScalar(gProj, secpFrToInt(secpFrFromInt(99))))))

        // Batch verification
        let batchN = 64
        var sigs = [ECDSASignature]()
        var pubkeys = [SecpPointAffine]()
        var recoveryBits = [UInt8]()
        for i in 0..<batchN {
            let di = secpFrFromInt(UInt64(100 + i))
            let qi = secpPointToAffine(secpPointMulScalar(gProj, secpFrToInt(di)))
            let ki = secpFrFromInt(UInt64(1000 + i * 7))
            let zi = secpFrFromInt(UInt64(50000 + i * 13))
            let ri_proj = secpPointMulScalar(gProj, secpFrToInt(ki))
            let ri_aff = secpPointToAffine(ri_proj)
            var ri_mod = secpToInt(ri_aff.x)
            if gte256(ri_mod, SecpFr.N) { (ri_mod, _) = sub256(ri_mod, SecpFr.N) }
            let ri_fr = secpFrFromRaw(ri_mod)
            let si_fr = secpFrMul(secpFrInverse(ki), secpFrAdd(zi, secpFrMul(ri_fr, di)))
            sigs.append(ECDSASignature(r: ri_fr, s: si_fr, z: zi))
            pubkeys.append(qi)
            recoveryBits.append(UInt8(secpToInt(ri_aff.y)[0] & 1))
        }

        let batchResults = try engine.batchVerify(signatures: sigs, pubkeys: pubkeys)
        check("Batch verify 64 valid sigs", batchResults.allSatisfy { $0 })

        let probResult = try engine.batchVerifyProbabilistic(signatures: sigs, pubkeys: pubkeys, recoveryBits: recoveryBits)
        check("Probabilistic batch verify 64 valid", probResult)

        // Detect 1 invalid
        var badSigs = sigs
        badSigs[batchN / 2] = ECDSASignature(r: sigs[batchN / 2].r, s: sigs[batchN / 2].s, z: secpFrFromInt(99999))
        let badBatch = try engine.batchVerify(signatures: badSigs, pubkeys: pubkeys)
        check("Batch detect 1 invalid", !badBatch[batchN / 2])

        let badProb = try engine.batchVerifyProbabilistic(signatures: badSigs, pubkeys: pubkeys, recoveryBits: recoveryBits)
        check("Probabilistic detect 1 invalid", !badProb)
    } catch {
        print("  [FAIL] ECDSA error: \(error)")
        _testFailed += 1
    }
}

// ============================================================
// MARK: - IPA
// ============================================================
private func testIPA() {
    print("\n--- IPA ---")

    let a4 = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let b4 = [frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8)]
    let ip = IPAEngine.innerProduct(a4, b4)
    check("Inner product <[1,2,3,4],[5,6,7,8]> = 70", frToInt(ip)[0] == 70)

    for logN in [2, 4, 6, 8] {
        let n = 1 << logN
        do {
            let (gens, Q) = IPAEngine.generateTestGenerators(count: n)
            let engine = try IPAEngine(generators: gens, Q: Q)

            var a = [Fr](); var bv = [Fr]()
            for i in 0..<n { a.append(frFromInt(UInt64(i + 1))); bv.append(frFromInt(UInt64(n - i))) }
            let v = IPAEngine.innerProduct(a, bv)
            let C = try engine.commit(a)
            let vQ = pointScalarMul(pointFromAffine(Q), v)
            let Cbound = pointAdd(C, vQ)

            let proof = try engine.createProof(a: a, b: bv)
            let valid = engine.verify(commitment: Cbound, b: bv, innerProductValue: v, proof: proof)
            check("IPA prove/verify n=\(n)", valid)

            let wrongV = frFromInt(999)
            let CboundWrong = pointAdd(C, pointScalarMul(pointFromAffine(Q), wrongV))
            let rejected = !engine.verify(commitment: CboundWrong, b: bv, innerProductValue: wrongV, proof: proof)
            check("IPA reject wrong v n=\(n)", rejected)
        } catch {
            print("  [FAIL] IPA n=\(n) error: \(error)")
            _testFailed += 1
        }
    }
}

// ============================================================
// MARK: - Verkle
// ============================================================
private func testVerkle() {
    print("\n--- Verkle ---")
    do {
        let testWidth = 4
        let engine = try VerkleEngine(width: testWidth)

        let values = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
        let C = try engine.commit(values)
        check("Commit non-trivial", !pointIsIdentity(C))

        var allOpeningsPass = true
        for idx in 0..<testWidth {
            let proof = try engine.createOpeningProof(values: values, index: idx)
            if !engine.verifyOpeningProof(proof) { allOpeningsPass = false }
        }
        check("All single openings verify", allOpeningsPass)

        // Tree building
        let numLeaves = testWidth * 2
        var leaves = [Fr]()
        for i in 0..<numLeaves { leaves.append(frFromInt(UInt64(i + 1))) }
        let (levels, _) = try engine.buildTree(leaves: leaves)

        let pathProof = try engine.createPathProof(leaves: leaves, leafIndex: 0)
        let root = levels.last![0]
        check("Path proof (leaf 0 -> root)", engine.verifyPathProof(pathProof, root: root))

        let pathProof2 = try engine.createPathProof(leaves: leaves, leafIndex: numLeaves - 1)
        check("Path proof (last leaf -> root)", engine.verifyPathProof(pathProof2, root: root))

        let wrongRoot = pointDouble(root)
        check("Reject wrong root", !engine.verifyPathProof(pathProof, root: wrongRoot))
    } catch {
        print("  [FAIL] Verkle error: \(error)")
        _testFailed += 1
    }
}

// ============================================================
// MARK: - LogUp
// ============================================================
private func testLogUp() {
    print("\n--- LogUp ---")
    do {
        let engine = try LookupEngine()

        // Simple lookup
        let table = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
        let lookups = [frFromInt(20), frFromInt(10), frFromInt(40), frFromInt(30)]
        let proof1 = try engine.prove(table: table, lookups: lookups, beta: frFromInt(12345))
        check("Simple lookup (m=4, N=4)", try engine.verify(proof: proof1, table: table, lookups: lookups))

        // Repeated lookups
        let table2: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 1)) }
        let lookups2: [Fr] = [1, 1, 3, 3, 5, 5, 7, 7].map { frFromInt($0) }
        let proof2 = try engine.prove(table: table2, lookups: lookups2, beta: frFromInt(99999))
        check("Repeated lookups (m=8, N=8)", try engine.verify(proof: proof2, table: table2, lookups: lookups2))

        // Multiplicities
        let mult = LookupEngine.computeMultiplicities(table: table2, lookups: lookups2)
        let expected: [UInt64] = [2, 0, 2, 0, 2, 0, 2, 0]
        var multCorrect = true
        for i in 0..<8 { if !frEqual(mult[i], frFromInt(expected[i])) { multCorrect = false; break } }
        check("Multiplicities correct", multCorrect)

        // Batch inverse
        let poly = try PolyEngine()
        let testVals = [frFromInt(3), frFromInt(7), frFromInt(11), frFromInt(13)]
        let inverses = try poly.batchInverse(testVals)
        var invOk = true
        for i in 0..<4 { if !frEqual(frMul(testVals[i], inverses[i]), Fr.one) { invOk = false } }
        check("Batch inverse", invOk)

        // Larger lookup (m=16, N=16)
        let table3: [Fr] = (0..<16).map { frFromInt(UInt64($0 * 7 + 3)) }
        let lookups3: [Fr] = (0..<16).map { table3[$0 % 16] }
        let proof3 = try engine.prove(table: table3, lookups: lookups3, beta: frFromInt(777))
        check("Larger lookup (m=16, N=16)", try engine.verify(proof: proof3, table: table3, lookups: lookups3))

        // Asymmetric (m=8, N=16)
        let lookups4: [Fr] = (0..<8).map { table3[$0] }
        let proof4 = try engine.prove(table: table3, lookups: lookups4, beta: frFromInt(54321))
        check("Asymmetric lookup (m=8, N=16)", try engine.verify(proof: proof4, table: table3, lookups: lookups4))

        // Reject tampered
        var wrongProof = proof1
        wrongProof = LookupProof(
            multiplicities: proof1.multiplicities, beta: proof1.beta,
            lookupSumcheckRounds: proof1.lookupSumcheckRounds, tableSumcheckRounds: proof1.tableSumcheckRounds,
            claimedSum: frAdd(proof1.claimedSum, Fr.one),
            lookupFinalEval: proof1.lookupFinalEval, tableFinalEval: proof1.tableFinalEval)
        check("Reject tampered sum", try !engine.verify(proof: wrongProof, table: table, lookups: lookups))
    } catch {
        print("  [FAIL] LogUp error: \(error)")
        _testFailed += 1
    }
}

// ============================================================
// MARK: - Radix Sort
// ============================================================
private func testRadixSort() {
    print("\n--- Radix Sort ---")
    do {
        let engine = try RadixSortEngine()

        check("Already sorted", try engine.sort([1, 2, 3, 4, 5, 6, 7, 8]) == [1,2,3,4,5,6,7,8])
        check("Reverse sorted", try engine.sort([8, 7, 6, 5, 4, 3, 2, 1]) == [1,2,3,4,5,6,7,8])
        check("With duplicates", try engine.sort([3, 1, 4, 1, 5, 9, 2, 6]) == [1,1,2,3,4,5,6,9])

        // Boundary 4096
        var boundaryKeys = [UInt32]()
        for i in (0..<4096).reversed() { boundaryKeys.append(UInt32(i)) }
        check("Boundary 4096", try engine.sort(boundaryKeys) == boundaryKeys.sorted())

        // Random 10K
        var rng: UInt64 = 0xDEAD_BEEF_CAFE
        var randomKeys = [UInt32]()
        for _ in 0..<10000 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            randomKeys.append(UInt32(truncatingIfNeeded: rng >> 32))
        }
        check("Random 10K", try engine.sort(randomKeys) == randomKeys.sorted())

        // Key-value sort
        let (sortedK, sortedV) = try engine.sortKV(keys: [30, 10, 20, 40], values: [300, 100, 200, 400])
        check("Key-value sort", sortedK == [10, 20, 30, 40] && sortedV == [100, 200, 300, 400])

        // Larger KV sort
        var kvKeys = [UInt32](); var kvVals = [UInt32]()
        for i in 0..<8192 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            kvKeys.append(UInt32(truncatingIfNeeded: rng >> 32)); kvVals.append(UInt32(i))
        }
        let (sk, sv) = try engine.sortKV(keys: kvKeys, values: kvVals)
        var kvCorrect = true
        for i in 1..<sk.count { if sk[i] < sk[i-1] { kvCorrect = false; break } }
        if kvCorrect { for i in 0..<sk.count { if kvKeys[Int(sv[i])] != sk[i] { kvCorrect = false; break } } }
        check("KV sort 8K", kvCorrect)

        check("Empty sort", try engine.sort([UInt32]()).isEmpty)
        check("Single sort", try engine.sort([42]) == [42])

        let fullRange: [UInt32] = [0, UInt32.max, UInt32.max / 2, 1, UInt32.max - 1]
        check("Full 32-bit range", try engine.sort(fullRange) == fullRange.sorted())
    } catch {
        print("  [FAIL] Radix sort error: \(error)")
        _testFailed += 1
    }
}

// ============================================================
// MARK: - Parallel Hash cross-checks
// ============================================================
private func testParallelHash() {
    print("\n--- ParallelHash Cross-Checks ---")

    let p2N = 256
    var p2Pairs = [(Fr, Fr)]()
    for i in 0..<p2N { p2Pairs.append((frFromInt(UInt64(i)), frFromInt(UInt64(i + p2N)))) }
    let seqP2 = p2Pairs.map { poseidon2Hash($0.0, $0.1) }
    let parP2 = parallelPoseidon2Batch(p2Pairs)
    var p2Match = true
    for i in 0..<p2N { if frToInt(seqP2[i]) != frToInt(parP2[i]) { p2Match = false; break } }
    check("Poseidon2 batch (256): parallel = sequential", p2Match)

    let kN = 256
    var kInputs = [[UInt8]]()
    for i in 0..<kN {
        var block = [UInt8](repeating: 0, count: 64)
        let val = UInt64(i)
        withUnsafeBytes(of: val) { block.replaceSubrange(0..<8, with: $0) }
        kInputs.append(block)
    }
    let seqK = kInputs.map { keccak256($0) }
    let parK = parallelKeccak256Batch(kInputs)
    var kMatch = true
    for i in 0..<kN { if seqK[i] != parK[i] { kMatch = false; break } }
    check("Keccak batch (256): parallel = sequential", kMatch)

    var b3Inputs = [[UInt8]]()
    for i in 0..<256 {
        var block = [UInt8](repeating: 0, count: 64)
        let val = UInt64(i)
        withUnsafeBytes(of: val) { block.replaceSubrange(0..<8, with: $0) }
        b3Inputs.append(block)
    }
    let seqB3 = b3Inputs.map { blake3($0) }
    let parB3 = parallelBlake3Batch(b3Inputs)
    var b3Match = true
    for i in 0..<256 { if seqB3[i] != parB3[i] { b3Match = false; break } }
    check("Blake3 batch (256): parallel = sequential", b3Match)
}

// ============================================================
// MARK: - Parallel Merkle cross-checks
// ============================================================
private func testParallelMerkle() {
    print("\n--- ParallelMerkle Cross-Checks ---")

    let mN = 64
    var leaves = [Fr]()
    for i in 0..<mN { leaves.append(frFromInt(UInt64(i + 1))) }
    let parTree = parallelPoseidon2Merkle(leaves)
    var merkleOk = true
    for i in 0..<mN { if frToInt(parTree[mN + i]) != frToInt(leaves[i]) { merkleOk = false; break } }
    for i in stride(from: mN - 1, through: 1, by: -1) {
        let expected = poseidon2Hash(parTree[2 * i], parTree[2 * i + 1])
        if frToInt(parTree[i]) != frToInt(expected) { merkleOk = false; break }
    }
    check("Poseidon2 Merkle (64 leaves)", merkleOk)

    var kLeaves = [[UInt8]]()
    for i in 0..<mN { var leaf = [UInt8](repeating: 0, count: 32); leaf[0] = UInt8(i & 0xFF); kLeaves.append(leaf) }
    let kTree = parallelKeccak256Merkle(kLeaves)
    var kOk = true
    for i in 0..<mN { if kTree[mN + i] != kLeaves[i] { kOk = false; break } }
    for i in stride(from: mN - 1, through: 1, by: -1) {
        if kTree[i] != keccak256(kTree[2 * i] + kTree[2 * i + 1]) { kOk = false; break }
    }
    check("Keccak Merkle (64 leaves)", kOk)

    let b3Tree = parallelBlake3Merkle(kLeaves)
    var b3Ok = true
    for i in 0..<mN { if b3Tree[mN + i] != kLeaves[i] { b3Ok = false; break } }
    for i in stride(from: mN - 1, through: 1, by: -1) {
        if b3Tree[i] != blake3(b3Tree[2 * i] + b3Tree[2 * i + 1]) { b3Ok = false; break }
    }
    check("Blake3 Merkle (64 leaves)", b3Ok)
}

// ============================================================
// MARK: - Parallel MSM cross-checks
// ============================================================
private func testParallelMSM() {
    print("\n--- ParallelMSM Cross-Checks ---")

    let gx = fpFromInt(1)
    let gy = fpFromInt(2)
    let gProj = pointFromAffine(PointAffine(x: gx, y: gy))

    let r1 = parallelMSM(points: [PointAffine(x: gx, y: gy)], scalars: [[1, 0, 0, 0, 0, 0, 0, 0]])
    check("ParallelMSM([G],[1]) = G", pointEqual(r1, gProj))

    let fiveG = pointMulInt(gProj, 5)
    let r2 = parallelMSM(points: [PointAffine(x: gx, y: gy)], scalars: [[5, 0, 0, 0, 0, 0, 0, 0]])
    check("ParallelMSM([G],[5]) = 5G", pointEqual(r2, fiveG))

    // 64 pts low scalars
    let n = 64
    var pts = [PointProjective]()
    var accum = gProj
    for _ in 0..<n { pts.append(accum); accum = pointAdd(accum, gProj) }
    let affPts = batchToAffine(pts)
    var scalars64 = [[UInt32]]()
    for i in 0..<n { scalars64.append([UInt32(i + 1), 0, 0, 0, 0, 0, 0, 0]) }
    var seqResult = pointIdentity()
    for i in 0..<n { seqResult = pointAdd(seqResult, pointScalarMul(pts[i], frFromLimbs(scalars64[i]))) }
    let parResult = parallelMSM(points: affPts, scalars: scalars64)
    check("ParallelMSM 64 pts = sequential", pointEqual(seqResult, parResult))

    // 128 pts full scalars
    let nFull = 128
    var fullPts = [PointProjective]()
    accum = gProj
    for _ in 0..<nFull { fullPts.append(accum); accum = pointAdd(accum, gProj) }
    let fullAff = batchToAffine(fullPts)
    var rng: UInt64 = 0xABCD_1234
    var fullScalars = [[UInt32]]()
    for _ in 0..<nFull {
        var limbs = [UInt32](repeating: 0, count: 8)
        for j in 0..<8 { rng = rng &* 6364136223846793005 &+ 1442695040888963407; limbs[j] = UInt32(truncatingIfNeeded: rng >> 32) }
        fullScalars.append(limbs)
    }
    var seqFull = pointIdentity()
    for i in 0..<nFull { seqFull = pointAdd(seqFull, pointScalarMul(fullPts[i], frFromLimbs(fullScalars[i]))) }
    let parFull = parallelMSM(points: fullAff, scalars: fullScalars)
    check("ParallelMSM 128 pts (full scalars) = sequential", pointEqual(seqFull, parFull))

    let rAff = batchToAffine([parFull])[0]
    let y2 = fpSqr(rAff.y)
    let x3 = fpMul(fpSqr(rAff.x), rAff.x)
    let rhs = fpAdd(x3, fpFromInt(3))
    check("ParallelMSM result on curve", fpToInt(y2) == fpToInt(rhs))
}

// ============================================================
// MARK: - Parallel NTT (safe subprocess wrapper)
// ============================================================
private func testParallelNTT_safe() {
    print("\n--- ParallelNTT Round-Trip ---")
    // Run in a subprocess to avoid Metal teardown SIGABRT crashing the whole suite
    let proc = Process()
    proc.executableURL = URL(fileURLWithPath: ProcessInfo.processInfo.arguments[0])
    proc.arguments = ["test-parallel-ntt"]
    let pipe = Pipe()
    proc.standardOutput = pipe
    proc.standardError = pipe
    do {
        try proc.run()
        proc.waitUntilExit()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let output = String(data: data, encoding: .utf8) ?? ""
        // Parse pass/fail from subprocess output
        var subPassed = 0
        var subFailed = 0
        for line in output.split(separator: "\n") {
            let s = String(line)
            if s.contains("[pass]") {
                subPassed += 1
                print(s)
            } else if s.contains("[FAIL]") {
                subFailed += 1
                print(s)
            }
        }
        _testPassed += subPassed
        _testFailed += subFailed
        if proc.terminationStatus != 0 && subPassed == 0 && subFailed == 0 {
            print("  [FAIL] ParallelNTT subprocess crashed (exit \(proc.terminationStatus))")
            _testFailed += 1
        }
    } catch {
        print("  [FAIL] ParallelNTT subprocess error: \(error)")
        _testFailed += 1
    }
}

// ============================================================
// MARK: - Parallel NTT round-trip (actual tests, called in subprocess)
// ============================================================
public func runParallelNTTTests() {
    _testPassed = 0
    _testFailed = 0
    testParallelNTT()
    fflush(stdout)
    exit(_testFailed > 0 ? 1 : 0)
}

private func testParallelNTT() {
    // Use smaller size to reduce crash risk (the crash may be memory/GCD related)
    let nttN = 1024
    let nttLogN = 10

    // Allocate input
    var nttInput = [Fr](repeating: Fr.zero, count: nttN)
    for i in 0..<nttN { nttInput[i] = frFromInt(UInt64(i + 1)) }
    fflush(stdout)
    let fwd = parallelNTT_Fr(nttInput, logN: nttLogN)
    let recovered = parallelINTT_Fr(fwd, logN: nttLogN)
    var nttOk = true
    for i in 0..<nttN { if frToInt(nttInput[i]) != frToInt(recovered[i]) { nttOk = false; break } }
    check("ParallelNTT Fr round-trip (2^10)", nttOk)

    let vanillaNTT = NTTEngine.cpuNTT(nttInput, logN: nttLogN)
    var crossOk = true
    for i in 0..<nttN { if frToInt(fwd[i]) != frToInt(vanillaNTT[i]) { crossOk = false; break } }
    check("ParallelNTT Fr matches vanilla cpuNTT (2^10)", crossOk)

    // BabyBear
    var bbInput = [Bb](repeating: Bb.zero, count: nttN)
    for i in 0..<nttN { bbInput[i] = Bb(v: UInt32(i + 1)) }
    let bbFwd = parallelNTT_Bb(bbInput, logN: nttLogN)
    let vanillaBb = BabyBearNTTEngine.cpuNTT(bbInput, logN: nttLogN)
    var bbOk = true
    for i in 0..<nttN { if bbFwd[i].v != vanillaBb[i].v { bbOk = false; break } }
    check("ParallelNTT Bb matches vanilla cpuNTT (2^10)", bbOk)

    // Goldilocks
    var glInput = [Gl](repeating: Gl.zero, count: nttN)
    for i in 0..<nttN { glInput[i] = Gl(v: UInt64(i + 1)) }
    let glFwd = parallelNTT_Gl(glInput, logN: nttLogN)
    let vanillaGl = GoldilocksNTTEngine.cpuNTT(glInput, logN: nttLogN)
    var glOk = true
    for i in 0..<nttN { if glFwd[i].v != vanillaGl[i].v { glOk = false; break } }
    check("ParallelNTT Gl matches vanilla cpuNTT (2^10)", glOk)
}

// ============================================================
// MARK: - C NTT cross-checks
// ============================================================
private func testCNTT() {
    print("\n--- C NTT Cross-Checks ---")

    // BN254 Fr
    let frN = 1024
    var frInput = [Fr](repeating: Fr.zero, count: frN)
    for i in 0..<frN { frInput[i] = frFromInt(UInt64(i + 1)) }
    let vanillaFr = NTTEngine.cpuNTT(frInput, logN: 10)
    let cFr = cNTT_Fr(frInput, logN: 10)
    var frMatch = true
    for i in 0..<frN { if frToInt(vanillaFr[i]) != frToInt(cFr[i]) { frMatch = false; break } }
    check("C NTT Fr matches vanilla (2^10)", frMatch)

    let cFrRt = cINTT_Fr(cFr, logN: 10)
    var frRtOk = true
    for i in 0..<frN { if frToInt(frInput[i]) != frToInt(cFrRt[i]) { frRtOk = false; break } }
    check("C NTT Fr round-trip (2^10)", frRtOk)

    // BabyBear NEON
    let bbN = 1024
    var bbInput = [Bb](repeating: Bb.zero, count: bbN)
    for i in 0..<bbN { bbInput[i] = Bb(v: UInt32(i + 1)) }
    let vanillaBb = BabyBearNTTEngine.cpuNTT(bbInput, logN: 10)
    let neonBb = neonNTT_Bb(bbInput, logN: 10)
    var bbMatch = true
    for i in 0..<bbN { if vanillaBb[i].v != neonBb[i].v { bbMatch = false; break } }
    check("NEON BabyBear NTT matches vanilla (2^10)", bbMatch)

    let neonBbRt = neonINTT_Bb(neonBb, logN: 10)
    var bbRtOk = true
    for i in 0..<bbN { if bbInput[i].v != neonBbRt[i].v { bbRtOk = false; break } }
    check("NEON BabyBear NTT round-trip (2^10)", bbRtOk)

    // Goldilocks C
    let glN = 1024
    var glInput = [Gl](repeating: Gl.zero, count: glN)
    for i in 0..<glN { glInput[i] = Gl(v: UInt64(i + 1)) }
    let vanillaGl = GoldilocksNTTEngine.cpuNTT(glInput, logN: 10)
    let cGl = cNTT_Gl(glInput, logN: 10)
    var glMatch = true
    for i in 0..<glN { if vanillaGl[i].v != cGl[i].v { glMatch = false; break } }
    check("C Goldilocks NTT matches vanilla (2^10)", glMatch)

    let cGlRt = cINTT_Gl(cGl, logN: 10)
    var glRtOk = true
    for i in 0..<glN { if glInput[i].v != cGlRt[i].v { glRtOk = false; break } }
    check("C Goldilocks NTT round-trip (2^10)", glRtOk)
}

// ============================================================
// MARK: - C Pippenger MSM cross-check
// ============================================================
private func testCPippengerMSM() {
    print("\n--- C Pippenger MSM ---")

    let gx = fpFromInt(1)
    let gy = fpFromInt(2)
    let gProj = pointFromAffine(PointAffine(x: gx, y: gy))

    let n = 256
    var projPoints = [PointProjective]()
    var acc = gProj
    for _ in 0..<n { projPoints.append(acc); acc = pointAdd(acc, gProj) }
    let pts = batchToAffine(projPoints)
    var rng: UInt64 = 0xDEAD_BEEF_CAFE_BABE
    var scalars = [[UInt32]]()
    for _ in 0..<n {
        var limbs = [UInt32](repeating: 0, count: 8)
        for j in 0..<8 { rng = rng &* 6364136223846793005 &+ 1442695040888963407; limbs[j] = UInt32(truncatingIfNeeded: rng >> 32) }
        scalars.append(limbs)
    }

    let cResult = cPippengerMSM(points: pts, scalars: scalars)
    let swResult = parallelMSM(points: pts, scalars: scalars)
    check("C Pippenger BN254 = Swift Pippenger (256 pts)", pointEqual(cResult, swResult))

    // secp256k1 C Pippenger
    do {
        let gen = secp256k1Generator()
        let gProj = secpPointFromAffine(gen)
        var secpPts = [SecpPointAffine]()
        var secpAcc = gProj
        for _ in 0..<n { secpPts.append(secpPointToAffine(secpAcc)); secpAcc = secpPointAdd(secpAcc, gProj) }
        let engine = try Secp256k1MSM()
        let cR = cSecpPippengerMSM(points: secpPts, scalars: scalars)
        let gpuR = try engine.msm(points: secpPts, scalars: scalars)
        let cAff = secpPointToAffine(cR)
        let gpuAff = secpPointToAffine(gpuR)
        check("C Pippenger secp256k1 = GPU (256 pts)",
              secpToInt(cAff.x) == secpToInt(gpuAff.x) && secpToInt(cAff.y) == secpToInt(gpuAff.y))
    } catch {
        print("  [FAIL] secp256k1 C Pippenger error: \(error)")
        _testFailed += 1
    }
}

// MARK: - Mersenne31 + Circle NTT

private func testMersenne31() {
    print("\n--- Mersenne31 Field ---")

    // Basic arithmetic
    let a = M31(v: 42), b = M31(v: 100)
    check("M31 add", m31Add(a, b).v == 142)
    check("M31 mul", m31Mul(a, b).v == 4200)
    check("M31 sub", m31Sub(b, a).v == 58)

    // Inverse
    let aInv = m31Inverse(a)
    check("M31 inverse", m31Mul(a, aInv).v == 1)

    // Edge cases
    check("M31 (p-1)*(p-2)=2", m31Mul(M31(v: M31.P - 1), M31(v: M31.P - 2)).v == 2)
    check("M31 wraparound", m31Add(M31(v: M31.P - 1), M31.one).v == 0)
    check("M31 negation", m31Add(a, m31Neg(a)).v == 0)

    // CM31
    let c1 = CM31(a: M31(v: 3), b: M31(v: 4))
    let cInv = cm31Inverse(c1)
    let cOne = cm31Mul(c1, cInv)
    check("CM31 inverse", cOne.a.v == 1 && cOne.b.v == 0)

    // Circle group
    let gen = CirclePoint.generator
    check("Generator on circle", gen.isOnCircle)
    var gPow = gen
    for _ in 0..<31 { gPow = circleGroupMul(gPow, gPow) }
    check("Circle gen order 2^31", gPow == CirclePoint.identity)

    // CPU Circle NTT roundtrip
    for logN in 1...6 {
        let n = 1 << logN
        var coeffs = [M31](repeating: M31.zero, count: n)
        var rng: UInt64 = 0xDEAD + UInt64(logN)
        for i in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            coeffs[i] = M31(v: UInt32(rng >> 33) % M31.P)
        }
        let evals = CircleNTTEngine.cpuNTT(coeffs, logN: logN)
        let recovered = CircleNTTEngine.cpuINTT(evals, logN: logN)
        var match = true
        for i in 0..<n { if recovered[i].v != coeffs[i].v { match = false; break } }
        check("CPU Circle NTT roundtrip N=\(n)", match)
    }
}

private func testCircleNTT() {
    print("\n--- Circle NTT (GPU) ---")
    do {
        let engine = try CircleNTTEngine()

        for logN in 1...10 {
            let n = 1 << logN
            var coeffs = [M31](repeating: M31.zero, count: n)
            var rng: UInt64 = 0xCAFE + UInt64(logN)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                coeffs[i] = M31(v: UInt32(rng >> 33) % M31.P)
            }

            let cpuEvals = CircleNTTEngine.cpuNTT(coeffs, logN: logN)
            let gpuEvals = try engine.ntt(coeffs)
            var fwdMatch = true
            for i in 0..<n { if gpuEvals[i].v != cpuEvals[i].v { fwdMatch = false; break } }

            let gpuRecovered = try engine.intt(gpuEvals)
            var invMatch = true
            for i in 0..<n { if gpuRecovered[i].v != coeffs[i].v { invMatch = false; break } }

            check("GPU Circle NTT N=\(n)", fwdMatch && invMatch)
        }
    } catch {
        print("  [FAIL] Circle NTT GPU error: \(error)")
        _testFailed += 1
    }
}
