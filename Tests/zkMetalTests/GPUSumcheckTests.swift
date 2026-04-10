import zkMetal
import Foundation

public func runGPUSumcheckTests() {
    suite("GPU Sumcheck Engine")

    // Helper: compare two Fr values
    func frEqual(_ a: Fr, _ b: Fr) -> Bool {
        return a.v.0 == b.v.0 && a.v.1 == b.v.1 && a.v.2 == b.v.2 && a.v.3 == b.v.3 &&
               a.v.4 == b.v.4 && a.v.5 == b.v.5 && a.v.6 == b.v.6 && a.v.7 == b.v.7
    }

    // Helper: deterministic pseudo-random Fr
    func pseudoRandomFr(seed: inout UInt64) -> Fr {
        seed = seed &* 6364136223846793005 &+ 1442695040888963407
        return frFromInt(seed >> 32)
    }

    // Helper: generate random evaluation table
    func randomEvals(_ logSize: Int, seed: UInt64 = 0xDEAD_BEEF_CAFE_1234) -> [Fr] {
        var rng = seed
        let n = 1 << logSize
        return (0..<n).map { _ in pseudoRandomFr(seed: &rng) }
    }

    // =========================================================================
    // SECTION 1: BN254 GPU vs CPU round polynomial
    // =========================================================================

    do {
        guard let engine = try? GPUSumcheckEngine() else {
            print("  [SKIP] No GPU available")
            return
        }

        // Test: GPU round poly matches CPU for small table (logSize=4, 16 elements)
        do {
            let logSize = 4
            let evals = randomEvals(logSize)
            let (cpuS0, cpuS1) = GPUSumcheckEngine.cpuRoundPoly(evals: evals)

            let stride = MemoryLayout<Fr>.stride
            guard let tableBuf = engine.device.makeBuffer(length: evals.count * stride,
                                                           options: .storageModeShared) else {
                expect(false, "Failed to create buffer")
                return
            }
            evals.withUnsafeBytes { src in
                memcpy(tableBuf.contents(), src.baseAddress!, evals.count * stride)
            }

            let (gpuS0, gpuS1) = try engine.computeRoundPolyBN254(table: tableBuf, logSize: logSize)
            expect(frEqual(cpuS0, gpuS0), "BN254 round poly s0 matches CPU (logSize=4)")
            expect(frEqual(cpuS1, gpuS1), "BN254 round poly s1 matches CPU (logSize=4)")
        }

        // Test: GPU round poly matches CPU for medium table (logSize=12, 4096 elements)
        do {
            let logSize = 12
            let evals = randomEvals(logSize)
            let (cpuS0, cpuS1) = GPUSumcheckEngine.cpuRoundPoly(evals: evals)

            let stride = MemoryLayout<Fr>.stride
            guard let tableBuf = engine.device.makeBuffer(length: evals.count * stride,
                                                           options: .storageModeShared) else {
                expect(false, "Failed to create buffer")
                return
            }
            evals.withUnsafeBytes { src in
                memcpy(tableBuf.contents(), src.baseAddress!, evals.count * stride)
            }

            let (gpuS0, gpuS1) = try engine.computeRoundPolyBN254(table: tableBuf, logSize: logSize)
            expect(frEqual(cpuS0, gpuS0), "BN254 round poly s0 matches CPU (logSize=12)")
            expect(frEqual(cpuS1, gpuS1), "BN254 round poly s1 matches CPU (logSize=12)")
        }

        // Test: GPU round poly matches CPU for large table (logSize=16, 65536 elements)
        do {
            let logSize = 16
            let evals = randomEvals(logSize)
            let (cpuS0, cpuS1) = GPUSumcheckEngine.cpuRoundPoly(evals: evals)

            let stride = MemoryLayout<Fr>.stride
            guard let tableBuf = engine.device.makeBuffer(length: evals.count * stride,
                                                           options: .storageModeShared) else {
                expect(false, "Failed to create buffer")
                return
            }
            evals.withUnsafeBytes { src in
                memcpy(tableBuf.contents(), src.baseAddress!, evals.count * stride)
            }

            let (gpuS0, gpuS1) = try engine.computeRoundPolyBN254(table: tableBuf, logSize: logSize)
            expect(frEqual(cpuS0, gpuS0), "BN254 round poly s0 matches CPU (logSize=16)")
            expect(frEqual(cpuS1, gpuS1), "BN254 round poly s1 matches CPU (logSize=16)")
        }

        // =========================================================================
        // SECTION 2: BN254 GPU vs CPU reduce (fold)
        // =========================================================================

        do {
            let logSize = 14
            let evals = randomEvals(logSize)
            var rng: UInt64 = 0xAAAA_BBBB_CCCC_DDDD
            let challenge = pseudoRandomFr(seed: &rng)

            let cpuResult = GPUSumcheckEngine.cpuReduce(evals: evals, challenge: challenge)

            let stride = MemoryLayout<Fr>.stride
            guard let tableBuf = engine.device.makeBuffer(length: evals.count * stride,
                                                           options: .storageModeShared) else {
                expect(false, "Failed to create buffer")
                return
            }
            evals.withUnsafeBytes { src in
                memcpy(tableBuf.contents(), src.baseAddress!, evals.count * stride)
            }

            let gpuOutBuf = try engine.reduceBN254Table(table: tableBuf, logSize: logSize, challenge: challenge)
            try engine.waitForPendingReduce()
            let halfN = evals.count / 2
            let gpuPtr = gpuOutBuf.contents().bindMemory(to: Fr.self, capacity: halfN)

            var allMatch = true
            for i in 0..<halfN {
                if !frEqual(cpuResult[i], gpuPtr[i]) {
                    allMatch = false
                    break
                }
            }
            expect(allMatch, "BN254 GPU reduce matches CPU (logSize=14)")
        }

        // =========================================================================
        // SECTION 3: Full sumcheck prove+verify (BN254)
        // =========================================================================

        // Full sumcheck protocol for logSize=10..14
        for logSize in [10, 12, 14] {
            let evals = randomEvals(logSize)
            let n = evals.count

            // Compute claimed sum = sum of all evaluations
            var claimedSum = Fr.zero
            for e in evals { claimedSum = frAdd(claimedSum, e) }

            // Prove
            let proverTranscript = Transcript(label: "gpu-sumcheck-test")
            let (rounds, challenges, finalEval) = try engine.fullSumcheckBN254(
                evals: evals, transcript: proverTranscript)

            expect(rounds.count == logSize, "Full sumcheck: \(logSize) rounds (logSize=\(logSize))")

            // Verify: s0 + s1 should equal the running claim at each round
            let verifierTranscript = Transcript(label: "gpu-sumcheck-test")
            var currentClaim = claimedSum
            var verifyOK = true

            for round in 0..<logSize {
                let (s0, s1) = rounds[round]
                let roundSum = frAdd(s0, s1)

                if !frEqual(roundSum, currentClaim) {
                    verifyOK = false
                    break
                }

                // Reconstruct challenge
                verifierTranscript.absorb(s0)
                verifierTranscript.absorb(s1)
                let vChallenge = verifierTranscript.squeeze()

                // Check challenge matches
                if !frEqual(vChallenge, challenges[round]) {
                    verifyOK = false
                    break
                }

                // Next claim: s(r) = (1-r)*s0 + r*s1 = s0 + r*(s1 - s0)
                let diff = frSub(s1, s0)
                let rDiff = frMul(vChallenge, diff)
                currentClaim = frAdd(s0, rDiff)
            }

            expect(verifyOK, "Full sumcheck verify: round sums match claims (logSize=\(logSize))")

            // Final check: the last claim should equal the final evaluation
            expect(frEqual(currentClaim, finalEval),
                   "Full sumcheck: final claim matches evaluation (logSize=\(logSize))")

            // Cross-check: evaluate MLE at the challenge point should equal finalEval
            let mle = MultilinearPoly(numVars: logSize, evals: evals)
            let mleEval = mle.evaluate(at: challenges)
            expect(frEqual(mleEval, finalEval),
                   "Full sumcheck: final eval = MLE(challenges) (logSize=\(logSize))")
        }

        // =========================================================================
        // SECTION 4: BabyBear field
        // =========================================================================

        do {
            let logSize = 12
            let bbP: UInt64 = 0x78000001
            var rng: UInt64 = 0x1234_5678_9ABC_DEF0
            let n = 1 << logSize
            var bbEvals = [UInt32](repeating: 0, count: n)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                bbEvals[i] = UInt32((rng >> 32) % bbP)
            }

            // CPU reference
            let halfN = n / 2
            var cpuS0: UInt64 = 0
            var cpuS1: UInt64 = 0
            for i in 0..<halfN {
                cpuS0 = (cpuS0 + UInt64(bbEvals[i])) % bbP
                cpuS1 = (cpuS1 + UInt64(bbEvals[i + halfN])) % bbP
            }

            let elemSize = MemoryLayout<UInt32>.stride
            guard let tableBuf = engine.device.makeBuffer(length: n * elemSize,
                                                           options: .storageModeShared) else {
                expect(false, "Failed to create BabyBear buffer")
                return
            }
            bbEvals.withUnsafeBytes { src in
                memcpy(tableBuf.contents(), src.baseAddress!, n * elemSize)
            }

            let (gpuS0, gpuS1) = try engine.computeRoundPolyBabyBear(table: tableBuf, logSize: logSize)
            expectEqual(gpuS0, UInt32(cpuS0), "BabyBear round poly s0 matches CPU")
            expectEqual(gpuS1, UInt32(cpuS1), "BabyBear round poly s1 matches CPU")
        }

        // =========================================================================
        // SECTION 5: Goldilocks field
        // =========================================================================

        do {
            let logSize = 12
            let glP: UInt64 = 0xFFFFFFFF00000001
            var rng: UInt64 = 0xFEDC_BA98_7654_3210
            let n = 1 << logSize
            var glEvals = [UInt64](repeating: 0, count: n)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                glEvals[i] = rng % glP
            }

            // CPU reference
            let halfN = n / 2
            var cpuS0: UInt64 = 0
            var cpuS1: UInt64 = 0
            for i in 0..<halfN {
                cpuS0 = glAddSimple(cpuS0, glEvals[i], glP)
                cpuS1 = glAddSimple(cpuS1, glEvals[i + halfN], glP)
            }

            let elemSize = MemoryLayout<UInt64>.stride
            guard let tableBuf = engine.device.makeBuffer(length: n * elemSize,
                                                           options: .storageModeShared) else {
                expect(false, "Failed to create Goldilocks buffer")
                return
            }
            glEvals.withUnsafeBytes { src in
                memcpy(tableBuf.contents(), src.baseAddress!, n * elemSize)
            }

            let (gpuS0, gpuS1) = try engine.computeRoundPolyGoldilocks(table: tableBuf, logSize: logSize)
            expectEqual(gpuS0, cpuS0, "Goldilocks round poly s0 matches CPU")
            expectEqual(gpuS1, cpuS1, "Goldilocks round poly s1 matches CPU")
        }

        // =========================================================================
        // SECTION 6: Performance comparison
        // =========================================================================

        do {
            for logSize in [16, 20] {
                let evals = randomEvals(logSize)
                let stride = MemoryLayout<Fr>.stride
                let n = evals.count

                guard let tableBuf = engine.device.makeBuffer(length: n * stride,
                                                               options: .storageModeShared) else {
                    continue
                }

                // GPU timing
                let gpuIters = logSize >= 20 ? 3 : 10
                var gpuTotal: Double = 0
                for _ in 0..<gpuIters {
                    evals.withUnsafeBytes { src in
                        memcpy(tableBuf.contents(), src.baseAddress!, n * stride)
                    }
                    let t0 = CFAbsoluteTimeGetCurrent()
                    let _ = try engine.computeRoundPolyBN254(table: tableBuf, logSize: logSize)
                    let t1 = CFAbsoluteTimeGetCurrent()
                    gpuTotal += t1 - t0
                }
                let gpuMs = (gpuTotal / Double(gpuIters)) * 1000.0

                // CPU timing
                let cpuIters = logSize >= 20 ? 1 : 5
                var cpuTotal: Double = 0
                for _ in 0..<cpuIters {
                    let t0 = CFAbsoluteTimeGetCurrent()
                    let _ = GPUSumcheckEngine.cpuRoundPoly(evals: evals)
                    let t1 = CFAbsoluteTimeGetCurrent()
                    cpuTotal += t1 - t0
                }
                let cpuMs = (cpuTotal / Double(cpuIters)) * 1000.0

                let speedup = cpuMs / max(gpuMs, 0.001)
                print(String(format: "  [PERF] logSize=%d: GPU=%.2fms CPU=%.2fms (%.1fx)",
                             logSize, gpuMs, cpuMs, speedup))
                expect(true, "Performance measured for logSize=\(logSize)")
            }
        }

    } catch {
        expect(false, "GPU Sumcheck threw: \(error)")
    }
}

// Goldilocks add helper for CPU reference in tests
private func glAddSimple(_ a: UInt64, _ b: UInt64, _ p: UInt64) -> UInt64 {
    let (sum, overflow) = a.addingReportingOverflow(b)
    if overflow {
        return sum &+ 0xFFFFFFFF
    }
    return sum >= p ? sum - p : sum
}
