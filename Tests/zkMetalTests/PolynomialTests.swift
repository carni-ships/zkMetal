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

    suite("SubproductTree Multi-Point Evaluation")
    do {
        let engine = try PolyEngine()

        // Test 1: Small known polynomial p(x) = 1 + 2x + 3x^2 at specific points
        let p: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3)]
        let pts: [Fr] = [frFromInt(0), frFromInt(1), frFromInt(2), frFromInt(5)]
        let horner = try engine.evaluate(p, at: pts)
        // p(0)=1, p(1)=6, p(2)=17, p(5)=86
        expect(frToInt(horner[0])[0] == 1, "Horner p(0)=1")
        expect(frToInt(horner[1])[0] == 6, "Horner p(1)=6")
        expect(frToInt(horner[2])[0] == 17, "Horner p(2)=17")
        expect(frToInt(horner[3])[0] == 86, "Horner p(5)=86")

        // Test 2: Tree eval matches Horner for deg 2^10 (512 random points)
        let tN = 512
        var rng: UInt64 = 0xCAFE_BABE
        var tCoeffs = [Fr](repeating: Fr.zero, count: tN)
        var tPts = [Fr](repeating: Fr.zero, count: tN)
        for i in 0..<tN {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            tCoeffs[i] = frFromInt(rng >> 32)
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            tPts[i] = frFromInt(rng >> 32)
        }
        let tHorner = try engine.evaluate(tCoeffs, at: tPts)
        let tTree = try engine.evaluateTree(tCoeffs, at: tPts)
        var match = true
        for i in 0..<tN {
            if frToInt(tHorner[i]) != frToInt(tTree[i]) { match = false; break }
        }
        expect(match, "Tree eval matches Horner (deg 2^9, 512 pts)")

        // Test 3: Larger test — deg 2^12
        let bigN = 1 << 12
        var bigCoeffs = [Fr](repeating: Fr.zero, count: bigN)
        var bigPts = [Fr](repeating: Fr.zero, count: bigN)
        for i in 0..<bigN {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            bigCoeffs[i] = frFromInt(rng >> 32)
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            bigPts[i] = frFromInt(rng >> 32)
        }
        let bigHorner = try engine.evaluate(bigCoeffs, at: bigPts)
        let bigTree = try engine.evaluateTree(bigCoeffs, at: bigPts)
        var bigMatch = true
        for i in 0..<bigN {
            if frToInt(bigHorner[i]) != frToInt(bigTree[i]) { bigMatch = false; break }
        }
        expect(bigMatch, "Tree eval matches Horner (deg 2^12)")
    } catch { expect(false, "SubproductTree eval error: \(error)") }

    suite("SubproductTree Batch Interpolation")
    do {
        let engine = try PolyEngine()

        // Test 1: Interpolate through known points
        // p(x) = 1 + 2x + 3x^2 → p(0)=1, p(1)=6, p(2)=17
        let pts3: [Fr] = [frFromInt(0), frFromInt(1), frFromInt(2)]
        let vals3: [Fr] = [frFromInt(1), frFromInt(6), frFromInt(17)]
        let interp3 = try engine.interpolateTree(points: pts3, values: vals3)
        expect(interp3.count == 3, "Interp3 has 3 coefficients")
        expect(frToInt(interp3[0])[0] == 1, "Interp3 c0=1")
        expect(frToInt(interp3[1])[0] == 2, "Interp3 c1=2")
        expect(frToInt(interp3[2])[0] == 3, "Interp3 c2=3")

        // Test 2: Interpolate random polynomial, check round-trip
        // Generate random poly of degree n-1, evaluate at n points, interpolate back
        let n = 64
        var rng: UInt64 = 0xBEEF_CAFE
        var origCoeffs = [Fr](repeating: Fr.zero, count: n)
        var interpPts = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            origCoeffs[i] = frFromInt(rng >> 32)
            interpPts[i] = frFromInt(UInt64(i + 1))  // Use 1, 2, ..., n as points
        }
        let interpVals = try engine.evaluate(origCoeffs, at: interpPts)
        let recovered = try engine.interpolateTree(points: interpPts, values: interpVals)
        var roundTrip = true
        for i in 0..<n {
            if frToInt(recovered[i]) != frToInt(origCoeffs[i]) { roundTrip = false; break }
        }
        expect(roundTrip, "Interpolate round-trip (n=64)")

        // Test 3: Larger round-trip with random points (not just consecutive)
        let n2 = 128
        var coeffs2 = [Fr](repeating: Fr.zero, count: n2)
        var pts2 = [Fr](repeating: Fr.zero, count: n2)
        for i in 0..<n2 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            coeffs2[i] = frFromInt(rng >> 32)
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            pts2[i] = frFromInt(rng >> 32)
        }
        let vals2 = try engine.evaluate(coeffs2, at: pts2)
        let recovered2 = try engine.interpolateTree(points: pts2, values: vals2)
        var roundTrip2 = true
        for i in 0..<n2 {
            if frToInt(recovered2[i]) != frToInt(coeffs2[i]) { roundTrip2 = false; break }
        }
        expect(roundTrip2, "Interpolate round-trip (n=128, random pts)")

        // Test 4: Larger test that exercises GPU path (n > 256)
        let n3 = 512
        var coeffs3 = [Fr](repeating: Fr.zero, count: n3)
        var pts3b = [Fr](repeating: Fr.zero, count: n3)
        for i in 0..<n3 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            coeffs3[i] = frFromInt(rng >> 32)
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            pts3b[i] = frFromInt(rng >> 32)
        }
        let vals3b = try engine.evaluate(coeffs3, at: pts3b)
        let recovered3 = try engine.interpolateTree(points: pts3b, values: vals3b)
        var roundTrip3 = true
        for i in 0..<n3 {
            if frToInt(recovered3[i]) != frToInt(coeffs3[i]) { roundTrip3 = false; break }
        }
        expect(roundTrip3, "Interpolate round-trip (n=512, GPU path)")

        // Test 5: Single point interpolation (edge case)
        let p1 = try engine.interpolateTree(points: [frFromInt(7)], values: [frFromInt(42)])
        expect(p1.count == 1 && frToInt(p1[0])[0] == 42, "Single point interp")

        // Test 6: Two point interpolation
        // Line through (1, 3) and (2, 5) → p(x) = 1 + 2x
        let p2 = try engine.interpolateTree(points: [frFromInt(1), frFromInt(2)],
                                              values: [frFromInt(3), frFromInt(5)])
        expect(p2.count == 2, "Two point interp has 2 coeffs")
        expect(frToInt(p2[0])[0] == 1, "Two point c0=1")
        expect(frToInt(p2[1])[0] == 2, "Two point c1=2")

    } catch { expect(false, "SubproductTree interpolation error: \(error)") }
}
