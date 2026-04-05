import zkMetal

func runPolynomialTests() {
    suite("FRI")
    do {
        let engine = try FRIEngine()
        let n = 1024
        var evals = [Fr](repeating: Fr.zero, count: n)
        var rng: UInt64 = 0xDEAD_BEEF
        for i in 0..<n { rng = rng &* 6364136223846793005 &+ 1442695040888963407; evals[i] = frFromInt(rng >> 32) }
        let beta = frFromInt(42)
        let gpuF = try engine.fold(evals: evals, beta: beta)
        let cpuF = FRIEngine.cpuFold(evals: evals, beta: beta, logN: 10)
        var ok = true
        for i in 0..<gpuF.count { if frToInt(gpuF[i]) != frToInt(cpuF[i]) { ok = false; break } }
        expect(ok, "Single fold GPU=CPU 2^10")

        // Multi-fold
        let logN = 16; let mn = 1 << logN
        var me = [Fr](repeating: Fr.zero, count: mn)
        for i in 0..<mn { rng = rng &* 6364136223846793005 &+ 1442695040888963407; me[i] = frFromInt(rng >> 32) }
        var betas = [Fr]()
        for i in 0..<logN { betas.append(frFromInt(UInt64(i + 1) * 7)) }
        let finalGPU = try engine.multiFold(evals: me, betas: betas)
        var cpuCurrent = me
        for i in 0..<logN { cpuCurrent = FRIEngine.cpuFold(evals: cpuCurrent, beta: betas[i], logN: logN - i) }
        expect(finalGPU.count == 1 && cpuCurrent.count == 1 && frToInt(finalGPU[0]) == frToInt(cpuCurrent[0]),
               "Multi-fold 2^16→1")

        // FRI protocol
        let pN = 1 << 14
        var pe = [Fr](repeating: Fr.zero, count: pN)
        for i in 0..<pN { rng = rng &* 6364136223846793005 &+ 1442695040888963407; pe[i] = frFromInt(rng >> 32) }
        var pb = [Fr]()
        for i in 0..<14 { pb.append(frFromInt(UInt64(i + 1) * 17)) }
        let commitment = try engine.commitPhase(evals: pe, betas: pb)
        let queries = try engine.queryPhase(commitment: commitment, queryIndices: [0, 42, 1000])
        expect(engine.verify(commitment: commitment, queries: queries), "FRI protocol verify 2^14")
    } catch { expect(false, "FRI error: \(error)") }

    suite("Sumcheck")
    do {
        let engine = try SumcheckEngine()
        let logN = 14; let n = 1 << logN
        var evals = [Fr](repeating: Fr.zero, count: n)
        var rng: UInt64 = 0xABCD_1234
        var expectedSum = Fr.zero
        for i in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            evals[i] = frFromInt(rng >> 32)
            expectedSum = frAdd(expectedSum, evals[i])
        }
        var challenges = [Fr]()
        for i in 0..<logN { challenges.append(frFromInt(UInt64(i * 13 + 7))) }
        let (rounds, _) = try engine.fullSumcheck(evals: evals, challenges: challenges)
        expect(rounds.count == logN, "Round count")
        let s0 = rounds[0].0; let s1 = rounds[0].1
        expect(frToInt(frAdd(s0, s1)) == frToInt(expectedSum), "S(0)+S(1)=sum")
    } catch { expect(false, "Sumcheck error: \(error)") }

    suite("KZG")
    do {
        let g = PointAffine(x: fpFromInt(1), y: fpFromInt(2))
        let secret: [UInt32] = [42, 0, 0, 0, 0, 0, 0, 0]
        let srs = KZGEngine.generateTestSRS(secret: secret, size: 256, generator: g)
        let engine = try KZGEngine(srs: srs)

        let constP = [frFromInt(1)]
        let c1 = try engine.commit(constP)
        let c1A = batchToAffine([c1])[0]
        expect(fpToInt(c1A.x) == fpToInt(g.x) && fpToInt(c1A.y) == fpToInt(g.y), "Commit([1])=G")

        let p1 = try engine.open(constP, at: frFromInt(999))
        expect(frToInt(p1.evaluation)[0] == 1, "Open(const) eval=1")
        expect(pointIsIdentity(p1.witness), "Open(const) witness=identity")

        let p: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3)]
        expect(frToInt(try engine.open(p, at: frFromInt(5)).evaluation)[0] == 86, "p(5)=86")
        expect(frToInt(try engine.open(p, at: frFromInt(0)).evaluation)[0] == 1, "p(0)=1")
        expect(frToInt(try engine.open(p, at: frFromInt(1)).evaluation)[0] == 6, "p(1)=6")

        // Linearity
        let a: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3)]
        let b: [Fr] = [frFromInt(4), frFromInt(5), frFromInt(6)]
        let sum = zip(a, b).map { frAdd($0, $1) }
        let cA = try engine.commit(a); let cB = try engine.commit(b); let cS = try engine.commit(sum)
        let cM = pointAdd(cA, cB)
        let a1 = batchToAffine([cS])[0]; let a2 = batchToAffine([cM])[0]
        expect(fpToInt(a1.x) == fpToInt(a2.x) && fpToInt(a1.y) == fpToInt(a2.y), "Commit linearity")
    } catch { expect(false, "KZG error: \(error)") }

    suite("KZG Batch Open (same point)")
    do {
        let g = PointAffine(x: fpFromInt(1), y: fpFromInt(2))
        let secret: [UInt32] = [42, 0, 0, 0, 0, 0, 0, 0]
        let srs = KZGEngine.generateTestSRS(secret: secret, size: 256, generator: g)
        let engine = try KZGEngine(srs: srs)
        let sFr = frFromLimbs(secret)

        // Test 1: batch open 3 polynomials at the same point
        let p0: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3)]       // 1 + 2x + 3x^2
        let p1: [Fr] = [frFromInt(4), frFromInt(5)]                     // 4 + 5x
        let p2: [Fr] = [frFromInt(7), frFromInt(0), frFromInt(1)]       // 7 + x^2
        let point = frFromInt(5)
        let gamma = frFromInt(13)

        let batchProof = try engine.batchOpen(polynomials: [p0, p1, p2], point: point, gamma: gamma)

        // Verify evaluations match individual evaluations
        expect(frToInt(batchProof.evaluations[0])[0] == 86, "batch p0(5)=86")   // 1+10+75
        expect(frToInt(batchProof.evaluations[1])[0] == 29, "batch p1(5)=29")   // 4+25
        expect(frToInt(batchProof.evaluations[2])[0] == 32, "batch p2(5)=32")   // 7+25

        // Verify using srsSecret
        let bv = engine.batchVerify(
            commitments: batchProof.commitments, point: point,
            evaluations: batchProof.evaluations, proof: batchProof.proof,
            gamma: gamma, srsSecret: sFr)
        expect(bv, "batch verify same-point")

        // Verify by re-opening
        let rv = try engine.verifyBatchByReopen(
            polynomials: [p0, p1, p2], point: point,
            evaluations: batchProof.evaluations, proof: batchProof.proof, gamma: gamma)
        expect(rv, "batch verify-by-reopen same-point")

        // Test 2: single polynomial batch = same as non-batch
        let single = try engine.batchOpen(polynomials: [p0], point: point, gamma: gamma)
        let singleDirect = try engine.open(p0, at: point)
        expect(frToInt(single.evaluations[0]) == frToInt(singleDirect.evaluation), "single batch eval matches open")

        // Test 3: wrong gamma should fail verification (different proof)
        let wrongGamma = frFromInt(99)
        let wrongVerify = engine.batchVerify(
            commitments: batchProof.commitments, point: point,
            evaluations: batchProof.evaluations, proof: batchProof.proof,
            gamma: wrongGamma, srsSecret: sFr)
        expect(!wrongVerify, "wrong gamma fails verify")

    } catch { expect(false, "KZG batch same-point error: \(error)") }

    suite("KZG Batch Open (multi-point)")
    do {
        let g = PointAffine(x: fpFromInt(1), y: fpFromInt(2))
        let secret: [UInt32] = [42, 0, 0, 0, 0, 0, 0, 0]
        let srs = KZGEngine.generateTestSRS(secret: secret, size: 256, generator: g)
        let engine = try KZGEngine(srs: srs)

        // Open 3 polynomials at different points
        let p0: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3)]
        let p1: [Fr] = [frFromInt(4), frFromInt(5)]
        let p2: [Fr] = [frFromInt(7), frFromInt(0), frFromInt(1)]
        let points: [Fr] = [frFromInt(5), frFromInt(3), frFromInt(2)]
        let gamma = frFromInt(17)

        let mpProof = try engine.batchOpenMultiPoint(polynomials: [p0, p1, p2], points: points, gamma: gamma)

        // Verify evaluations: p0(5)=86, p1(3)=19, p2(2)=11
        expect(frToInt(mpProof.evaluations[0])[0] == 86, "multi-pt p0(5)=86")
        expect(frToInt(mpProof.evaluations[1])[0] == 19, "multi-pt p1(3)=19")
        expect(frToInt(mpProof.evaluations[2])[0] == 11, "multi-pt p2(2)=11")

        // Verify by re-opening
        let rv = try engine.verifyMultiPointByReopen(
            polynomials: [p0, p1, p2], points: points,
            evaluations: mpProof.evaluations, proof: mpProof.proof, gamma: gamma)
        expect(rv, "multi-point verify-by-reopen")

        // Test fused path matches non-fused
        let fusedProof = try engine.batchOpenMultiPointFused(polynomials: [p0, p1, p2], points: points, gamma: gamma)
        for i in 0..<3 {
            expect(frToInt(fusedProof.evaluations[i]) == frToInt(mpProof.evaluations[i]),
                   "fused eval[\(i)] matches")
        }
        let fusedAff = batchToAffine([fusedProof.proof])
        let mpAff = batchToAffine([mpProof.proof])
        expect(fpToInt(fusedAff[0].x) == fpToInt(mpAff[0].x) &&
               fpToInt(fusedAff[0].y) == fpToInt(mpAff[0].y),
               "fused proof point matches non-fused")

    } catch { expect(false, "KZG batch multi-point error: \(error)") }
}
