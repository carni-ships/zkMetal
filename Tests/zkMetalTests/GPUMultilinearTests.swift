import zkMetal
import Foundation

public func runGPUMultilinearTests() {
    suite("GPU Multilinear Engine")

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

    // Helper: generate random point
    func randomPoint(_ n: Int, seed: UInt64 = 0xABCD_1234_5678_9ABC) -> [Fr] {
        var rng = seed
        return (0..<n).map { _ in pseudoRandomFr(seed: &rng) }
    }

    do {
        guard let engine = try? GPUMultilinearEngine() else {
            print("  [SKIP] No GPU available")
            return
        }

        // =========================================================================
        // SECTION 1: MLE Evaluation matches CPU reference
        // =========================================================================

        suite("GPU MLE Evaluate")

        // Test: small polynomial (CPU fallback path)
        do {
            let logSize = 3
            let evals = randomEvals(logSize)
            let point = randomPoint(logSize)

            let cpuResult = GPUMultilinearEngine.cpuEvaluate(evals: evals, point: point)
            let gpuResult = engine.evaluate(evals: evals, point: point)

            expect(frEqual(cpuResult, gpuResult),
                   "MLE evaluate matches CPU (logSize=3, CPU fallback)")
        }

        // Test: medium polynomial (GPU path, logSize=12)
        do {
            let logSize = 12
            let evals = randomEvals(logSize)
            let point = randomPoint(logSize)

            let cpuResult = GPUMultilinearEngine.cpuEvaluate(evals: evals, point: point)
            let gpuResult = engine.evaluate(evals: evals, point: point)

            expect(frEqual(cpuResult, gpuResult),
                   "MLE evaluate matches CPU (logSize=12, GPU)")
        }

        // Test: larger polynomial (GPU path, logSize=16)
        do {
            let logSize = 16
            let evals = randomEvals(logSize)
            let point = randomPoint(logSize)

            let cpuResult = GPUMultilinearEngine.cpuEvaluate(evals: evals, point: point)
            let gpuResult = engine.evaluate(evals: evals, point: point)

            expect(frEqual(cpuResult, gpuResult),
                   "MLE evaluate matches CPU (logSize=16, GPU)")
        }

        // Test: evaluate at boolean point recovers evaluation
        do {
            let evals: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
            // f(1, 0) = evals[2] = 30  (index 10 in binary, MSB convention)
            let result = engine.evaluate(evals: evals, point: [Fr.one, Fr.zero])
            expect(frEqual(result, frFromInt(30)),
                   "MLE evaluate at boolean point (1,0) = 30")
        }

        // =========================================================================
        // SECTION 2: Eq Polynomial
        // =========================================================================

        suite("GPU MLE Eq Polynomial")

        // Test: eq(r, r) sums to 1
        // sum_{x in {0,1}^n} eq(r, x) = 1 for any r
        do {
            let point = randomPoint(4, seed: 0x1111_2222_3333_4444)
            let eqArr = engine.eqPolyArray(point: point)
            var sum = Fr.zero
            for e in eqArr { sum = frAdd(sum, e) }
            expect(frEqual(sum, Fr.one),
                   "eq poly: sum over all x = 1")
        }

        // Test: eq at boolean point is indicator
        do {
            let point: [Fr] = [Fr.zero, Fr.one]
            let eq = engine.eqPolyArray(point: point)
            // eq(point, (0,0)) = eq([0,1], [0,0]) = (1-0)(1-0) * (1)(0) = 0
            expect(frEqual(eq[0], Fr.zero), "eq([0,1], (0,0)) = 0")
            // eq(point, (0,1)) = (1-0)(1-0) * (1)(1) = 1
            expect(frEqual(eq[1], Fr.one), "eq([0,1], (0,1)) = 1")
            // eq(point, (1,0)) = 0
            expect(frEqual(eq[2], Fr.zero), "eq([0,1], (1,0)) = 0")
            // eq(point, (1,1)) = 0
            expect(frEqual(eq[3], Fr.zero), "eq([0,1], (1,1)) = 0")
        }

        // Test: GPU eq matches CPU reference
        do {
            let point = randomPoint(10, seed: 0xAAAA_BBBB_CCCC_DDDD)
            let cpuEq = GPUMultilinearEngine.cpuEqPoly(point: point)
            let gpuEq = engine.eqPolyArray(point: point)

            var allMatch = true
            for i in 0..<cpuEq.count {
                if !frEqual(cpuEq[i], gpuEq[i]) {
                    allMatch = false
                    break
                }
            }
            expect(allMatch, "GPU eq poly matches CPU (logSize=10)")
        }

        // Test: inner product <eq(r, .), f> = f(r) (MLE evaluation via eq polynomial)
        do {
            let logSize = 8
            let evals = randomEvals(logSize)
            let point = randomPoint(logSize, seed: 0xFACE_FEED_DEAD_BEEF)

            let eq = engine.eqPolyArray(point: point)
            var innerProduct = Fr.zero
            for i in 0..<evals.count {
                innerProduct = frAdd(innerProduct, frMul(evals[i], eq[i]))
            }

            let directEval = engine.evaluate(evals: evals, point: point)
            expect(frEqual(innerProduct, directEval),
                   "<eq(r,.), f> = f(r) via inner product")
        }

        // =========================================================================
        // SECTION 3: Bind (Partial Evaluation) Consistency
        // =========================================================================

        suite("GPU MLE Bind")

        // Test: evaluate(bind(f, r0), [r1,...]) == evaluate(f, [r0, r1,...])
        do {
            let logSize = 12
            let evals = randomEvals(logSize)
            let point = randomPoint(logSize, seed: 0x1234_ABCD_5678_EF01)

            let r0 = point[0]
            let restPoint = Array(point[1...])

            // Path 1: bind first variable, then evaluate the rest
            let bound = engine.bindArray(evals: evals, logSize: logSize, value: r0)
            let evalBound = engine.evaluate(evals: bound, point: restPoint)

            // Path 2: evaluate all at once
            let evalFull = engine.evaluate(evals: evals, point: point)

            expect(frEqual(evalBound, evalFull),
                   "bind consistency: evaluate(bind(f,r0), rest) == evaluate(f, all)")
        }

        // Test: bind reduces size correctly
        do {
            let logSize = 10
            let evals = randomEvals(logSize)
            let bound = engine.bindArray(evals: evals, logSize: logSize, value: frFromInt(42))
            expectEqual(bound.count, 1 << (logSize - 1),
                        "bind halves the table size")
        }

        // Test: bind at 0 gives first half, bind at 1 gives second half
        do {
            let evals: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
            let logSize = 2

            let bound0 = engine.bindArray(evals: evals, logSize: logSize, value: Fr.zero)
            expect(frEqual(bound0[0], frFromInt(1)), "bind(0): first element")
            expect(frEqual(bound0[1], frFromInt(2)), "bind(0): second element")

            let bound1 = engine.bindArray(evals: evals, logSize: logSize, value: Fr.one)
            expect(frEqual(bound1[0], frFromInt(3)), "bind(1): first element")
            expect(frEqual(bound1[1], frFromInt(4)), "bind(1): second element")
        }

        // Test: multiple sequential binds
        do {
            let logSize = 8
            let evals = randomEvals(logSize)
            let point = randomPoint(logSize, seed: 0xDEAD_CAFE_BEEF_F00D)

            // Sequential bind all variables
            let stride = MemoryLayout<Fr>.stride
            guard let buf = engine.device.makeBuffer(length: evals.count * stride,
                                                      options: .storageModeShared) else {
                expect(false, "Failed to create buffer")
                return
            }
            evals.withUnsafeBytes { src in
                memcpy(buf.contents(), src.baseAddress!, evals.count * stride)
            }

            var currentBuf = buf
            for round in 0..<logSize {
                guard let nextBuf = engine.bind(evals: currentBuf, logSize: logSize - round,
                                                 value: point[round]) else {
                    expect(false, "bind failed at round \(round)")
                    return
                }
                currentBuf = nextBuf
            }
            let seqResult = currentBuf.contents().bindMemory(to: Fr.self, capacity: 1)[0]

            // Direct evaluate
            let directResult = engine.evaluate(evals: evals, point: point)

            expect(frEqual(seqResult, directResult),
                   "sequential bind matches direct evaluate (logSize=8)")
        }

        // =========================================================================
        // SECTION 4: Tensor Product
        // =========================================================================

        suite("GPU MLE Tensor Product")

        // Test: tensor product size
        do {
            let logA = 3
            let logB = 4
            let a = randomEvals(logA, seed: 0x1111)
            let b = randomEvals(logB, seed: 0x2222)
            let result = engine.tensorProductArray(a: a, logA: logA, b: b, logB: logB)
            expectEqual(result.count, (1 << logA) * (1 << logB),
                        "tensor product: size(a x b) = size(a) * size(b)")
        }

        // Test: tensor product entries are correct
        do {
            let a: [Fr] = [frFromInt(2), frFromInt(3)]
            let b: [Fr] = [frFromInt(5), frFromInt(7)]
            let result = engine.tensorProductArray(a: a, logA: 1, b: b, logB: 1)
            expect(frEqual(result[0], frFromInt(10)), "tensor: 2*5 = 10")
            expect(frEqual(result[1], frFromInt(14)), "tensor: 2*7 = 14")
            expect(frEqual(result[2], frFromInt(15)), "tensor: 3*5 = 15")
            expect(frEqual(result[3], frFromInt(21)), "tensor: 3*7 = 21")
        }

        // Test: tensor evaluation = product of evaluations
        do {
            let logA = 3
            let logB = 2
            let aEvals = randomEvals(logA, seed: 0x3333)
            let bEvals = randomEvals(logB, seed: 0x4444)
            let tensor = engine.tensorProductArray(a: aEvals, logA: logA, b: bEvals, logB: logB)

            let pointA = randomPoint(logA, seed: 0x5555)
            let pointB = randomPoint(logB, seed: 0x6666)
            let pointAB = pointA + pointB

            let tensorEval = engine.evaluate(evals: tensor, point: pointAB)
            let aEval = engine.evaluate(evals: aEvals, point: pointA)
            let bEval = engine.evaluate(evals: bEvals, point: pointB)
            let productEval = frMul(aEval, bEval)

            expect(frEqual(tensorEval, productEval),
                   "tensor eval = product of individual evals")
        }

        // =========================================================================
        // SECTION 5: Cross-validation with MultilinearPoly
        // =========================================================================

        suite("GPU MLE Cross-validation")

        // Test: GPU evaluate matches MultilinearPoly.evaluateC
        do {
            let logSize = 10
            let evals = randomEvals(logSize)
            let point = randomPoint(logSize, seed: 0x7777_8888_9999_AAAA)

            let mle = MultilinearPoly(numVars: logSize, evals: evals)
            let cpuResult = mle.evaluateC(at: point)
            let gpuResult = engine.evaluate(evals: evals, point: point)

            expect(frEqual(cpuResult, gpuResult),
                   "GPU evaluate matches MultilinearPoly.evaluateC (logSize=10)")
        }

        // Test: GPU eq matches MultilinearPoly.eqPolyC
        do {
            let point = randomPoint(8, seed: 0xBBBB_CCCC_DDDD_EEEE)
            let cpuEq = MultilinearPoly.eqPolyC(point: point)
            let gpuEq = engine.eqPolyArray(point: point)

            var allMatch = true
            for i in 0..<cpuEq.size {
                if !frEqual(cpuEq.evals[i], gpuEq[i]) {
                    allMatch = false
                    break
                }
            }
            expect(allMatch, "GPU eq matches MultilinearPoly.eqPolyC (logSize=8)")
        }

        // =========================================================================
        // SECTION 6: Performance
        // =========================================================================

        suite("GPU MLE Performance")

        do {
            let logSize = 20
            let evals = randomEvals(logSize)
            let point = randomPoint(logSize, seed: 0xBEEF_CAFE_DEAD_F00D)

            // Warmup
            let _ = engine.evaluate(evals: evals, point: point)

            // Timed runs
            let iters = 3
            var total: Double = 0
            for _ in 0..<iters {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = engine.evaluate(evals: evals, point: point)
                let t1 = CFAbsoluteTimeGetCurrent()
                total += t1 - t0
            }
            let avgMs = (total / Double(iters)) * 1000.0
            print(String(format: "  [PERF] MLE evaluate 2^20 (%d points): %.2fms",
                         logSize, avgMs))
            expect(true, "Performance measured for MLE evaluate 2^20")
        }

        // Eq polynomial performance
        do {
            let logSize = 16
            let point = randomPoint(logSize, seed: 0xEEEE_FFFF_0000_1111)

            // Warmup
            let _ = engine.eqPolyArray(point: point)

            let iters = 5
            var total: Double = 0
            for _ in 0..<iters {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = engine.eqPoly(point: point)
                let t1 = CFAbsoluteTimeGetCurrent()
                total += t1 - t0
            }
            let avgMs = (total / Double(iters)) * 1000.0
            print(String(format: "  [PERF] eq poly 2^%d: %.2fms", logSize, avgMs))
            expect(true, "Performance measured for eq poly 2^16")
        }

    } catch {
        expect(false, "GPU Multilinear Engine threw: \(error)")
    }
}
