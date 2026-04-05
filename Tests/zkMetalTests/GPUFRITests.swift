// GPU FRI query phase tests
// Validates fold layer, batch query, full fold, and cross-field correctness.
// Run: swift build && .build/debug/zkMetalTests

import zkMetal
import Foundation

// MARK: - Test helpers

private struct FRIRNG {
    var state: UInt64

    mutating func next32() -> UInt32 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return UInt32(state >> 33)
    }

    mutating func nextFr() -> Fr {
        let raw = Fr(v: (next32() & 0x0FFFFFFF, next32(), next32(), next32(),
                         next32(), next32(), next32(), next32() & 0x0FFFFFFF))
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))
    }

    mutating func nextBb() -> UInt32 {
        return next32() % Bb.P
    }

    mutating func nextM31() -> UInt32 {
        return next32() % M31.P
    }
}

/// CPU reference fold for BN254 Fr.
private func cpuFoldBn254(evals: [Fr], alpha: Fr, logN: Int) -> [Fr] {
    let n = evals.count
    let half = n / 2
    let invTwiddles = precomputeInverseTwiddles(logN: logN)
    var folded = [Fr](repeating: Fr.zero, count: half)
    for i in 0..<half {
        let a = evals[i]
        let b = evals[i + half]
        let sum = frAdd(a, b)
        let diff = frSub(a, b)
        let term = frMul(frMul(alpha, invTwiddles[i]), diff)
        folded[i] = frAdd(sum, term)
    }
    return folded
}

/// CPU reference fold for BabyBear.
private func cpuFoldBb(evals: [UInt32], alpha: UInt32, logN: Int) -> [UInt32] {
    let n = evals.count
    let half = n / 2
    let omega = bbRootOfUnity(logN: logN)
    let omegaInv = bbInverse(omega)
    var w = Bb.one
    var folded = [UInt32](repeating: 0, count: half)
    for i in 0..<half {
        let a = Bb(v: evals[i])
        let b = Bb(v: evals[i + half])
        let sum = bbAdd(a, b)
        let diff = bbSub(a, b)
        let term = bbMul(bbMul(Bb(v: alpha), w), diff)
        folded[i] = bbAdd(sum, term).v
        w = bbMul(w, omegaInv)
    }
    return folded
}

/// CPU reference fold for M31.
private func cpuFoldM31(evals: [UInt32], alpha: UInt32, logN: Int) -> [UInt32] {
    let n = evals.count
    let half = n / 2
    let exp = (M31.P - 1) / UInt32(n)
    let omega = m31Pow(M31(v: 3), exp)
    let omegaInv = m31Inverse(omega)
    var w = M31.one
    var folded = [UInt32](repeating: 0, count: half)
    for i in 0..<half {
        let a = M31(v: evals[i])
        let b = M31(v: evals[i + half])
        let sum = m31Add(a, b)
        let diff = m31Sub(a, b)
        let term = m31Mul(m31Mul(M31(v: alpha), w), diff)
        folded[i] = m31Add(sum, term).v
        w = m31Mul(w, omegaInv)
    }
    return folded
}

// MARK: - Individual tests

private func testSingleFoldBn254() {
    suite("GPU FRI: Single fold layer BN254")
    do {
        let engine = try GPUFRIEngine()
        var rng = FRIRNG(state: 42)
        let logN = 10
        let n = 1 << logN

        // Generate random evaluations
        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = rng.nextFr() }
        let alpha = rng.nextFr()

        // GPU fold
        let evalsBuf = engine.device.makeBuffer(
            bytes: evals, length: n * MemoryLayout<Fr>.stride,
            options: .storageModeShared)!
        let gpuResult = try engine.foldLayer(evals: evalsBuf, logSize: logN,
                                              alpha: alpha, field: .bn254)

        // CPU reference
        let cpuResult = cpuFoldBn254(evals: evals, alpha: alpha, logN: logN)

        // Compare
        let half = n / 2
        let ptr = gpuResult.contents().bindMemory(to: Fr.self, capacity: half)
        var allMatch = true
        for i in 0..<half {
            if ptr[i] != cpuResult[i] {
                allMatch = false
                break
            }
        }
        expect(allMatch, "BN254 fold layer matches CPU reference (n=\(n))")
        print("  BN254 fold layer: \(allMatch ? "PASS" : "FAIL") (n=\(n))")
    } catch {
        print("  [FAIL] BN254 fold: \(error)")
        expect(false, "BN254 fold threw: \(error)")
    }
}

private func testSingleFoldBabyBear() {
    suite("GPU FRI: Single fold layer BabyBear")
    do {
        let engine = try GPUFRIEngine()
        var rng = FRIRNG(state: 99)
        let logN = 10
        let n = 1 << logN

        var evals = [UInt32](repeating: 0, count: n)
        for i in 0..<n { evals[i] = rng.nextBb() }
        let alpha = rng.nextBb()

        let evalsBuf = engine.device.makeBuffer(
            bytes: evals, length: n * 4, options: .storageModeShared)!
        let alphaFr = Fr(v: (alpha, 0, 0, 0, 0, 0, 0, 0))
        let gpuResult = try engine.foldLayer(evals: evalsBuf, logSize: logN,
                                              alpha: alphaFr, field: .babybear)

        let cpuResult = cpuFoldBb(evals: evals, alpha: alpha, logN: logN)

        let half = n / 2
        let ptr = gpuResult.contents().bindMemory(to: UInt32.self, capacity: half)
        var allMatch = true
        for i in 0..<half {
            if ptr[i] != cpuResult[i] {
                allMatch = false
                break
            }
        }
        expect(allMatch, "BabyBear fold layer matches CPU reference (n=\(n))")
        print("  BabyBear fold layer: \(allMatch ? "PASS" : "FAIL") (n=\(n))")
    } catch {
        print("  [FAIL] BabyBear fold: \(error)")
        expect(false, "BabyBear fold threw: \(error)")
    }
}

private func testMultiLayerFoldBn254() {
    suite("GPU FRI: Multi-layer fold BN254")
    do {
        let engine = try GPUFRIEngine()
        var rng = FRIRNG(state: 77)
        let logN = 12
        let n = 1 << logN
        let numLayers = 3

        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = rng.nextFr() }

        var challenges = [Fr]()
        for _ in 0..<numLayers { challenges.append(rng.nextFr()) }

        // GPU: fold through all layers using fullFold
        let evalsBuf = engine.device.makeBuffer(
            bytes: evals, length: n * MemoryLayout<Fr>.stride,
            options: .storageModeShared)!
        let gpuResult = try engine.fullFold(evals: evalsBuf, logSize: logN,
                                             challenges: challenges, field: .bn254)

        // CPU: fold sequentially
        var cpuEvals = evals
        var cpuLogN = logN
        for c in challenges {
            cpuEvals = cpuFoldBn254(evals: cpuEvals, alpha: c, logN: cpuLogN)
            cpuLogN -= 1
        }

        let finalSize = 1 << (logN - numLayers)
        let ptr = gpuResult.contents().bindMemory(to: Fr.self, capacity: finalSize)
        var allMatch = true
        for i in 0..<finalSize {
            if ptr[i] != cpuEvals[i] {
                allMatch = false
                break
            }
        }
        expect(allMatch, "Multi-layer fold matches CPU (logN=\(logN), layers=\(numLayers))")
        print("  Multi-layer fold: \(allMatch ? "PASS" : "FAIL") (logN=\(logN), layers=\(numLayers))")
    } catch {
        print("  [FAIL] Multi-layer fold: \(error)")
        expect(false, "Multi-layer fold threw: \(error)")
    }
}

private func testBatchQueryBn254() {
    suite("GPU FRI: Batch query BN254")
    do {
        let engine = try GPUFRIEngine()
        var rng = FRIRNG(state: 55)
        let n = 1024

        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = rng.nextFr() }

        let evalsBuf = engine.device.makeBuffer(
            bytes: evals, length: n * MemoryLayout<Fr>.stride,
            options: .storageModeShared)!

        // Query at specific indices
        let indices = [0, 1, 42, 511, 512, 1023]
        let results = try engine.batchQuery(evals: evalsBuf, indices: indices, field: .bn254)

        var allMatch = true
        for (qi, idx) in indices.enumerated() {
            if results[qi] != evals[idx] {
                allMatch = false
                break
            }
        }
        expect(allMatch, "Batch query returns correct evaluations")
        print("  Batch query: \(allMatch ? "PASS" : "FAIL") (\(indices.count) queries)")
    } catch {
        print("  [FAIL] Batch query: \(error)")
        expect(false, "Batch query threw: \(error)")
    }
}

private func testBatchQueryBabyBear() {
    suite("GPU FRI: Batch query BabyBear")
    do {
        let engine = try GPUFRIEngine()
        var rng = FRIRNG(state: 66)
        let n = 1024

        var evals = [UInt32](repeating: 0, count: n)
        for i in 0..<n { evals[i] = rng.nextBb() }

        let evalsBuf = engine.device.makeBuffer(
            bytes: evals, length: n * 4, options: .storageModeShared)!

        let indices = [0, 7, 100, 500, 1023]
        let results = try engine.batchQuery(evals: evalsBuf, indices: indices, field: .babybear)

        var allMatch = true
        for (qi, idx) in indices.enumerated() {
            if results[qi].v.0 != evals[idx] {
                allMatch = false
                break
            }
        }
        expect(allMatch, "BabyBear batch query correct")
        print("  BabyBear batch query: \(allMatch ? "PASS" : "FAIL")")
    } catch {
        print("  [FAIL] BabyBear batch query: \(error)")
        expect(false, "BabyBear batch query threw: \(error)")
    }
}

private func testFullFoldToConstantBn254() {
    suite("GPU FRI: Full fold to constant BN254")
    do {
        let engine = try GPUFRIEngine()
        var rng = FRIRNG(state: 123)
        let logN = 10
        let n = 1 << logN

        // Generate random evaluations
        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = rng.nextFr() }

        // Generate logN challenges to fold down to a single element
        var challenges = [Fr]()
        for _ in 0..<logN { challenges.append(rng.nextFr()) }

        let evalsBuf = engine.device.makeBuffer(
            bytes: evals, length: n * MemoryLayout<Fr>.stride,
            options: .storageModeShared)!
        let gpuResult = try engine.fullFold(evals: evalsBuf, logSize: logN,
                                             challenges: challenges, field: .bn254)

        // CPU reference: fold all the way down
        var cpuEvals = evals
        var cpuLogN = logN
        for c in challenges {
            cpuEvals = cpuFoldBn254(evals: cpuEvals, alpha: c, logN: cpuLogN)
            cpuLogN -= 1
        }

        expect(cpuEvals.count == 1, "CPU fold produces 1 element")

        let ptr = gpuResult.contents().bindMemory(to: Fr.self, capacity: 1)
        let gpuVal = ptr[0]
        let match = gpuVal == cpuEvals[0]
        expect(match, "Full fold to constant matches CPU")
        print("  Full fold to constant: \(match ? "PASS" : "FAIL") (logN=\(logN))")
    } catch {
        print("  [FAIL] Full fold to constant: \(error)")
        expect(false, "Full fold to constant threw: \(error)")
    }
}

private func testCrossFieldBabyBear() {
    suite("GPU FRI: Cross-field BabyBear fold")
    do {
        let engine = try GPUFRIEngine()
        var rng = FRIRNG(state: 200)
        let logN = 10
        let n = 1 << logN

        var evals = [UInt32](repeating: 0, count: n)
        for i in 0..<n { evals[i] = rng.nextBb() }
        let alpha = rng.nextBb()

        // GPU fold
        let evalsBuf = engine.device.makeBuffer(
            bytes: evals, length: n * 4, options: .storageModeShared)!
        let alphaFr = Fr(v: (alpha, 0, 0, 0, 0, 0, 0, 0))
        let gpuResult = try engine.foldLayer(evals: evalsBuf, logSize: logN,
                                              alpha: alphaFr, field: .babybear)

        // Multi-layer fold: 3 layers
        let logN2 = logN
        var challenges = [Fr]()
        for _ in 0..<3 {
            let c = rng.nextBb()
            challenges.append(Fr(v: (c, 0, 0, 0, 0, 0, 0, 0)))
        }

        let evalsBuf2 = engine.device.makeBuffer(
            bytes: evals, length: n * 4, options: .storageModeShared)!
        let fullResult = try engine.fullFold(evals: evalsBuf2, logSize: logN2,
                                              challenges: challenges, field: .babybear)

        // CPU reference for multi-layer
        var cpuEvals = evals
        var cpuLog = logN2
        for c in challenges {
            cpuEvals = cpuFoldBb(evals: cpuEvals, alpha: c.v.0, logN: cpuLog)
            cpuLog -= 1
        }

        let finalSize = 1 << (logN2 - 3)
        let ptr = fullResult.contents().bindMemory(to: UInt32.self, capacity: finalSize)
        var allMatch = true
        for i in 0..<finalSize {
            if ptr[i] != cpuEvals[i] {
                allMatch = false
                break
            }
        }
        expect(allMatch, "BabyBear multi-layer fold matches CPU")
        print("  BabyBear cross-field: \(allMatch ? "PASS" : "FAIL")")
    } catch {
        print("  [FAIL] BabyBear cross-field: \(error)")
        expect(false, "BabyBear cross-field threw: \(error)")
    }
}

private func testCrossFieldM31() {
    suite("GPU FRI: Cross-field M31 fold")
    do {
        let engine = try GPUFRIEngine()
        var rng = FRIRNG(state: 300)
        // M31 has two-adicity of 1 for multiplicative group (p-1 = 2 * odd).
        // But we can still do FRI fold for small power-of-2 sizes where
        // (p-1) is divisible by n. Actually p-1 = 2147483646 = 2 * 3 * ...
        // For test, use a size where n | (p-1).
        // p-1 = 2 * 3 * 7 * 11 * 31 * 151 * 331
        // So n=2 is fine (logN=1). For larger: n must divide p-1.
        // 2^10 = 1024. p-1 / 1024 = 2097151.999... not integer.
        // Actually p-1 = 2147483646. 2147483646 / 1024 = 2097151.998... no.
        // So we cannot use power-of-2 domains > 2 for standard multiplicative M31 FRI.
        // Use logN=1 for correctness test.
        let logN = 1
        let n = 1 << logN

        var evals = [UInt32](repeating: 0, count: n)
        for i in 0..<n { evals[i] = rng.nextM31() }
        let alpha = rng.nextM31()

        let evalsBuf = engine.device.makeBuffer(
            bytes: evals, length: n * 4, options: .storageModeShared)!
        let alphaFr = Fr(v: (alpha, 0, 0, 0, 0, 0, 0, 0))

        // This will use CPU fallback (n=2 < 512)
        let gpuResult = try engine.foldLayer(evals: evalsBuf, logSize: logN,
                                              alpha: alphaFr, field: .m31)

        let cpuResult = cpuFoldM31(evals: evals, alpha: alpha, logN: logN)
        let ptr = gpuResult.contents().bindMemory(to: UInt32.self, capacity: 1)
        let match = ptr[0] == cpuResult[0]
        expect(match, "M31 fold matches CPU (n=\(n))")
        print("  M31 cross-field: \(match ? "PASS" : "FAIL") (n=\(n))")
    } catch {
        print("  [FAIL] M31 cross-field: \(error)")
        expect(false, "M31 cross-field threw: \(error)")
    }
}

private func testPerformanceFold() {
    suite("GPU FRI: Performance fold 2^20 x 10 layers")
    do {
        let engine = try GPUFRIEngine()
        var rng = FRIRNG(state: 500)
        let logN = 20
        let n = 1 << logN
        let numLayers = 10

        // Generate random evaluations
        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = rng.nextFr() }

        var challenges = [Fr]()
        for _ in 0..<numLayers { challenges.append(rng.nextFr()) }

        let evalsBuf = engine.device.makeBuffer(
            bytes: evals, length: n * MemoryLayout<Fr>.stride,
            options: .storageModeShared)!

        // Warmup
        _ = try engine.fullFold(evals: evalsBuf, logSize: logN,
                                 challenges: challenges, field: .bn254)

        // Re-upload (fullFold may have consumed the buffer state)
        let evalsBuf2 = engine.device.makeBuffer(
            bytes: evals, length: n * MemoryLayout<Fr>.stride,
            options: .storageModeShared)!

        let start = DispatchTime.now()
        let result = try engine.fullFold(evals: evalsBuf2, logSize: logN,
                                          challenges: challenges, field: .bn254)
        let end = DispatchTime.now()
        let ms = Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000.0

        let finalSize = 1 << (logN - numLayers)
        let ptr = result.contents().bindMemory(to: Fr.self, capacity: finalSize)
        let nonZero = !ptr[0].isZero
        expect(nonZero, "Performance fold produces non-zero result")
        print("  Performance: 2^20 x 10 layers in \(String(format: "%.2f", ms)) ms")
        print("  Final size: \(finalSize) elements")
    } catch {
        print("  [FAIL] Performance fold: \(error)")
        expect(false, "Performance fold threw: \(error)")
    }
}

private func testCPUFallback() {
    suite("GPU FRI: CPU fallback for small layers")
    do {
        let engine = try GPUFRIEngine()
        var rng = FRIRNG(state: 444)
        let logN = 8  // 256 elements — below threshold of 512
        let n = 1 << logN

        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = rng.nextFr() }
        let alpha = rng.nextFr()

        let evalsBuf = engine.device.makeBuffer(
            bytes: evals, length: n * MemoryLayout<Fr>.stride,
            options: .storageModeShared)!

        // Should use CPU fallback
        let result = try engine.foldLayer(evals: evalsBuf, logSize: logN,
                                           alpha: alpha, field: .bn254)

        let cpuResult = cpuFoldBn254(evals: evals, alpha: alpha, logN: logN)
        let half = n / 2
        let ptr = result.contents().bindMemory(to: Fr.self, capacity: half)
        var allMatch = true
        for i in 0..<half {
            if ptr[i] != cpuResult[i] {
                allMatch = false
                break
            }
        }
        expect(allMatch, "CPU fallback matches reference (n=\(n))")
        print("  CPU fallback: \(allMatch ? "PASS" : "FAIL") (n=\(n))")
    } catch {
        print("  [FAIL] CPU fallback: \(error)")
        expect(false, "CPU fallback threw: \(error)")
    }
}

// MARK: - Public entry point

public func runGPUFRITests() {
    testSingleFoldBn254()
    testSingleFoldBabyBear()
    testMultiLayerFoldBn254()
    testBatchQueryBn254()
    testBatchQueryBabyBear()
    testFullFoldToConstantBn254()
    testCrossFieldBabyBear()
    testCrossFieldM31()
    testPerformanceFold()
    testCPUFallback()
}
