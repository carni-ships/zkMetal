import zkMetal
import Foundation

// MARK: - Test helpers

private struct QuotientTestRNG {
    var state: UInt64

    mutating func next32() -> UInt32 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return UInt32(state >> 33)
    }

    mutating func nextBb() -> UInt32 {
        return next32() % Bb.P
    }

    mutating func nextFr() -> Fr {
        let raw = Fr(v: (next32() & 0x0FFFFFFF, next32(), next32(), next32(),
                         next32(), next32(), next32(), next32() & 0x0FFFFFFF))
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))
    }
}

private func frToWords(_ f: Fr) -> [UInt32] {
    [f.v.0, f.v.1, f.v.2, f.v.3, f.v.4, f.v.5, f.v.6, f.v.7]
}

private func wordsToFr(_ w: [UInt32], offset: Int = 0) -> Fr {
    Fr(v: (w[offset], w[offset+1], w[offset+2], w[offset+3],
           w[offset+4], w[offset+5], w[offset+6], w[offset+7]))
}

// MARK: - Public test entry point

public func runGPUQuotientEngineTests() {
    testFusedQuotientBn254()
    testFusedQuotientBabyBear()
    testPrecomputedQuotientBn254()
    testPrecomputedQuotientBabyBear()
    testVanishingInverseCaching()
    testSplitQuotientBn254()
    testSplitQuotientBabyBear()
    testFusedVsPrecomputedConsistency()
    testGPUvsCP_QuotientBn254()
    testLargeQuotientPerformance()
}

// MARK: - Fused quotient tests

/// Construct constraint evals = t(x) * Z_H(x) over a coset, verify quotient = t(x)
private func testFusedQuotientBn254() {
    suite("GPUQuotientEngine BN254 fused")
    do {
        let engine = try GPUQuotientEngine()

        let logDomain = 4  // 16 points
        let logTraceLen = 2  // trace length = 4
        let domainSize = 1 << logDomain

        // Coset generator
        let g = frFromInt(7)
        let cosetGen = frToWords(g)
        let omega = frRootOfUnity(logN: logDomain)

        // Build coset points and constraint evals
        var evalData = [UInt32]()
        evalData.reserveCapacity(domainSize * 8)
        var expectedQ = [Fr]()
        expectedQ.reserveCapacity(domainSize)

        var cosetPt = g
        for _ in 0..<domainSize {
            // t(x) = x + 2 (simple test polynomial)
            let tVal = frAdd(cosetPt, frFromInt(2))

            // Z_H(x) = x^4 - 1
            var zh = cosetPt
            for _ in 0..<logTraceLen { zh = frSqr(zh) }
            zh = frSub(zh, Fr.one)

            // constraint eval = t(x) * Z_H(x)
            let eval = frMul(tVal, zh)
            evalData.append(contentsOf: frToWords(eval))
            expectedQ.append(tVal)

            cosetPt = frMul(cosetPt, omega)
        }

        let cosetBuf = try engine.getOrBuildCosetPoints(logDomain: logDomain,
                                                         cosetGen: cosetGen, field: .bn254)
        let evalsBuf = engine.device.makeBuffer(bytes: evalData,
                                                 length: evalData.count * 4,
                                                 options: .storageModeShared)!

        let result = try engine.computeQuotientFused(constraintEvals: evalsBuf,
                                                      cosetPoints: cosetBuf,
                                                      domainSize: domainSize,
                                                      logTraceLen: logTraceLen,
                                                      field: .bn254)

        let outPtr = result.contents().bindMemory(to: UInt32.self, capacity: domainSize * 8)
        var allMatch = true
        for i in 0..<domainSize {
            let got = wordsToFr(Array(UnsafeBufferPointer(start: outPtr + i * 8, count: 8)))
            if got != expectedQ[i] {
                allMatch = false
                break
            }
        }
        expect(allMatch, "fused quotient recovers t(x) for all domain points")

    } catch {
        expect(false, "GPUQuotientEngine BN254 fused failed: \(error)")
    }
}

private func testFusedQuotientBabyBear() {
    suite("GPUQuotientEngine BabyBear fused")
    do {
        let engine = try GPUQuotientEngine()

        let logDomain = 4
        let logTraceLen = 2
        let domainSize = 1 << logDomain

        let g = Bb(v: 7)
        let cosetGen: [UInt32] = [g.v]
        let omega = bbRootOfUnity(logN: logDomain)

        var evalData = [UInt32]()
        evalData.reserveCapacity(domainSize)
        var expectedQ = [UInt32]()
        expectedQ.reserveCapacity(domainSize)

        var cosetPt = g
        for _ in 0..<domainSize {
            let tVal = bbAdd(cosetPt, Bb(v: 3))
            var zh = cosetPt
            for _ in 0..<logTraceLen { zh = bbSqr(zh) }
            zh = bbSub(zh, Bb.one)
            evalData.append(bbMul(tVal, zh).v)
            expectedQ.append(tVal.v)
            cosetPt = bbMul(cosetPt, omega)
        }

        let cosetBuf = try engine.getOrBuildCosetPoints(logDomain: logDomain,
                                                         cosetGen: cosetGen, field: .babybear)
        let evalsBuf = engine.device.makeBuffer(bytes: evalData,
                                                 length: evalData.count * 4,
                                                 options: .storageModeShared)!

        let result = try engine.computeQuotientFused(constraintEvals: evalsBuf,
                                                      cosetPoints: cosetBuf,
                                                      domainSize: domainSize,
                                                      logTraceLen: logTraceLen,
                                                      field: .babybear)

        let outPtr = result.contents().bindMemory(to: UInt32.self, capacity: domainSize)
        var allMatch = true
        for i in 0..<domainSize {
            if outPtr[i] != expectedQ[i] {
                allMatch = false
                break
            }
        }
        expect(allMatch, "BabyBear fused quotient recovers t(x)")

    } catch {
        expect(false, "GPUQuotientEngine BabyBear fused failed: \(error)")
    }
}

// MARK: - Precomputed quotient tests

private func testPrecomputedQuotientBn254() {
    suite("GPUQuotientEngine BN254 precomputed")
    do {
        let engine = try GPUQuotientEngine()

        let logDomain = 4
        let logTraceLen = 2
        let domainSize = 1 << logDomain

        let g = frFromInt(5)
        let cosetGen = frToWords(g)
        let omega = frRootOfUnity(logN: logDomain)

        // Precompute vanishing inverses
        let zhInvBuf = try engine.precomputeVanishingInverses(logDomain: logDomain,
                                                               logTraceLen: logTraceLen,
                                                               cosetGen: cosetGen,
                                                               field: .bn254)

        // Build evals = t(x) * Z_H(x) where t(x) = 3x + 1
        var evalData = [UInt32]()
        evalData.reserveCapacity(domainSize * 8)
        var expectedQ = [Fr]()
        expectedQ.reserveCapacity(domainSize)

        var cosetPt = g
        for _ in 0..<domainSize {
            let tVal = frAdd(frMul(frFromInt(3), cosetPt), frFromInt(1))
            var zh = cosetPt
            for _ in 0..<logTraceLen { zh = frSqr(zh) }
            zh = frSub(zh, Fr.one)
            evalData.append(contentsOf: frToWords(frMul(tVal, zh)))
            expectedQ.append(tVal)
            cosetPt = frMul(cosetPt, omega)
        }

        let evalsBuf = engine.device.makeBuffer(bytes: evalData,
                                                 length: evalData.count * 4,
                                                 options: .storageModeShared)!

        let result = try engine.computeQuotient(constraintEvals: evalsBuf,
                                                 vanishingInverses: zhInvBuf,
                                                 domainSize: domainSize,
                                                 field: .bn254)

        let outPtr = result.contents().bindMemory(to: UInt32.self, capacity: domainSize * 8)
        var allMatch = true
        for i in 0..<domainSize {
            let got = wordsToFr(Array(UnsafeBufferPointer(start: outPtr + i * 8, count: 8)))
            if got != expectedQ[i] {
                allMatch = false
                break
            }
        }
        expect(allMatch, "precomputed quotient recovers t(x)")

    } catch {
        expect(false, "GPUQuotientEngine BN254 precomputed failed: \(error)")
    }
}

private func testPrecomputedQuotientBabyBear() {
    suite("GPUQuotientEngine BabyBear precomputed")
    do {
        let engine = try GPUQuotientEngine()

        let logDomain = 4
        let logTraceLen = 2
        let domainSize = 1 << logDomain

        let g = Bb(v: 11)
        let cosetGen: [UInt32] = [g.v]
        let omega = bbRootOfUnity(logN: logDomain)

        let zhInvBuf = try engine.precomputeVanishingInverses(logDomain: logDomain,
                                                               logTraceLen: logTraceLen,
                                                               cosetGen: cosetGen,
                                                               field: .babybear)

        var evalData = [UInt32]()
        var expectedQ = [UInt32]()

        var cosetPt = g
        for _ in 0..<domainSize {
            let tVal = bbAdd(bbMul(Bb(v: 5), cosetPt), Bb(v: 2))
            var zh = cosetPt
            for _ in 0..<logTraceLen { zh = bbSqr(zh) }
            zh = bbSub(zh, Bb.one)
            evalData.append(bbMul(tVal, zh).v)
            expectedQ.append(tVal.v)
            cosetPt = bbMul(cosetPt, omega)
        }

        let evalsBuf = engine.device.makeBuffer(bytes: evalData,
                                                 length: evalData.count * 4,
                                                 options: .storageModeShared)!

        let result = try engine.computeQuotient(constraintEvals: evalsBuf,
                                                 vanishingInverses: zhInvBuf,
                                                 domainSize: domainSize,
                                                 field: .babybear)

        let outPtr = result.contents().bindMemory(to: UInt32.self, capacity: domainSize)
        var allMatch = true
        for i in 0..<domainSize {
            if outPtr[i] != expectedQ[i] { allMatch = false; break }
        }
        expect(allMatch, "BabyBear precomputed quotient correct")

    } catch {
        expect(false, "GPUQuotientEngine BabyBear precomputed failed: \(error)")
    }
}

// MARK: - Caching test

private func testVanishingInverseCaching() {
    suite("GPUQuotientEngine vanishing inverse caching")
    do {
        let engine = try GPUQuotientEngine()

        let cosetGen = frToWords(frFromInt(7))

        let buf1 = try engine.precomputeVanishingInverses(logDomain: 4, logTraceLen: 2,
                                                           cosetGen: cosetGen, field: .bn254)
        let buf2 = try engine.precomputeVanishingInverses(logDomain: 4, logTraceLen: 2,
                                                           cosetGen: cosetGen, field: .bn254)

        // Same pointer = same cached buffer
        expect(buf1.contents() == buf2.contents(), "cached vanishing inverses return same buffer")

        // After clearing, should get a new buffer
        engine.clearCaches()
        let buf3 = try engine.precomputeVanishingInverses(logDomain: 4, logTraceLen: 2,
                                                           cosetGen: cosetGen, field: .bn254)

        // Content should still be correct (compare first element)
        let ptr1 = buf1.contents().bindMemory(to: UInt32.self, capacity: 8)
        let ptr3 = buf3.contents().bindMemory(to: UInt32.self, capacity: 8)
        let v1 = wordsToFr(Array(UnsafeBufferPointer(start: ptr1, count: 8)))
        let v3 = wordsToFr(Array(UnsafeBufferPointer(start: ptr3, count: 8)))
        expectEqual(v1, v3, "cleared cache recomputes same values")

    } catch {
        expect(false, "GPUQuotientEngine caching test failed: \(error)")
    }
}

// MARK: - Split quotient tests

private func testSplitQuotientBn254() {
    suite("GPUQuotientEngine BN254 split quotient")
    do {
        let engine = try GPUQuotientEngine()

        // Create a polynomial with 8 coefficients, split into 2 chunks
        // coeffs = [c0, c1, c2, c3, c4, c5, c6, c7]
        // chunk_0 = [c0, c2, c4, c6] (even indices)
        // chunk_1 = [c1, c3, c5, c7] (odd indices)
        var rng = QuotientTestRNG(state: 0xABCDEF01)
        var coeffs = [UInt32]()
        var frCoeffs = [Fr]()
        for _ in 0..<8 {
            let f = rng.nextFr()
            frCoeffs.append(f)
            coeffs.append(contentsOf: frToWords(f))
        }

        let chunks = engine.splitQuotient(quotientCoeffs: coeffs, numChunks: 2, field: .bn254)

        expectEqual(chunks.count, 2, "2 chunks produced")
        expectEqual(chunks[0].count, 4 * 8, "chunk 0 has 4 elements * 8 words")
        expectEqual(chunks[1].count, 4 * 8, "chunk 1 has 4 elements * 8 words")

        // Verify chunk_0 = [c0, c2, c4, c6]
        for (j, idx) in [0, 2, 4, 6].enumerated() {
            let got = wordsToFr(chunks[0], offset: j * 8)
            expectEqual(got, frCoeffs[idx], "chunk_0[\(j)] = coeff[\(idx)]")
        }

        // Verify chunk_1 = [c1, c3, c5, c7]
        for (j, idx) in [1, 3, 5, 7].enumerated() {
            let got = wordsToFr(chunks[1], offset: j * 8)
            expectEqual(got, frCoeffs[idx], "chunk_1[\(j)] = coeff[\(idx)]")
        }

    } catch {
        expect(false, "GPUQuotientEngine BN254 split failed: \(error)")
    }
}

private func testSplitQuotientBabyBear() {
    suite("GPUQuotientEngine BabyBear split quotient")
    do {
        let engine = try GPUQuotientEngine()

        // 12 coefficients, split into 3 chunks of 4
        let coeffs: [UInt32] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]

        let chunks = engine.splitQuotient(quotientCoeffs: coeffs, numChunks: 3, field: .babybear)

        expectEqual(chunks.count, 3, "3 chunks")
        // chunk_0 = [c0, c3, c6, c9] = [10, 40, 70, 100]
        expectEqual(chunks[0], [UInt32(10), 40, 70, 100], "chunk_0 correct")
        // chunk_1 = [c1, c4, c7, c10] = [20, 50, 80, 110]
        expectEqual(chunks[1], [UInt32(20), 50, 80, 110], "chunk_1 correct")
        // chunk_2 = [c2, c5, c8, c11] = [30, 60, 90, 120]
        expectEqual(chunks[2], [UInt32(30), 60, 90, 120], "chunk_2 correct")

    } catch {
        expect(false, "GPUQuotientEngine BabyBear split failed: \(error)")
    }
}

// MARK: - Consistency: fused vs precomputed

private func testFusedVsPrecomputedConsistency() {
    suite("GPUQuotientEngine fused vs precomputed consistency")
    do {
        let engine = try GPUQuotientEngine()

        let logDomain = 4
        let logTraceLen = 2
        let domainSize = 1 << logDomain

        let g = frFromInt(13)
        let cosetGen = frToWords(g)
        let omega = frRootOfUnity(logN: logDomain)

        // Build random-ish constraint evals (not necessarily divisible by Z_H)
        var rng = QuotientTestRNG(state: 0x98765432)
        var evalData = [UInt32]()
        evalData.reserveCapacity(domainSize * 8)
        for _ in 0..<domainSize {
            evalData.append(contentsOf: frToWords(rng.nextFr()))
        }

        let evalsBuf1 = engine.device.makeBuffer(bytes: evalData, length: evalData.count * 4,
                                                   options: .storageModeShared)!
        let evalsBuf2 = engine.device.makeBuffer(bytes: evalData, length: evalData.count * 4,
                                                   options: .storageModeShared)!

        // Fused path
        let cosetBuf = try engine.getOrBuildCosetPoints(logDomain: logDomain,
                                                         cosetGen: cosetGen, field: .bn254)
        let fusedResult = try engine.computeQuotientFused(constraintEvals: evalsBuf1,
                                                           cosetPoints: cosetBuf,
                                                           domainSize: domainSize,
                                                           logTraceLen: logTraceLen,
                                                           field: .bn254)

        // Precomputed path
        let zhInvBuf = try engine.precomputeVanishingInverses(logDomain: logDomain,
                                                               logTraceLen: logTraceLen,
                                                               cosetGen: cosetGen,
                                                               field: .bn254)
        let precompResult = try engine.computeQuotient(constraintEvals: evalsBuf2,
                                                        vanishingInverses: zhInvBuf,
                                                        domainSize: domainSize,
                                                        field: .bn254)

        // Compare results
        let fPtr = fusedResult.contents().bindMemory(to: UInt32.self, capacity: domainSize * 8)
        let pPtr = precompResult.contents().bindMemory(to: UInt32.self, capacity: domainSize * 8)

        var allMatch = true
        for i in 0..<domainSize {
            let fVal = wordsToFr(Array(UnsafeBufferPointer(start: fPtr + i * 8, count: 8)))
            let pVal = wordsToFr(Array(UnsafeBufferPointer(start: pPtr + i * 8, count: 8)))
            if fVal != pVal {
                allMatch = false
                break
            }
        }
        expect(allMatch, "fused and precomputed paths produce identical results")

    } catch {
        expect(false, "GPUQuotientEngine consistency test failed: \(error)")
    }
}

// MARK: - GPU vs CPU correctness

private func testGPUvsCP_QuotientBn254() {
    suite("GPUQuotientEngine GPU vs CPU BN254")
    do {
        let engine = try GPUQuotientEngine()

        // Use a domain large enough to hit GPU path (>= 512)
        let logDomain = 10  // 1024
        let logTraceLen = 4  // trace = 16
        let domainSize = 1 << logDomain

        let g = frFromInt(7)
        let cosetGen = frToWords(g)
        let omega = frRootOfUnity(logN: logDomain)

        // Build evals = t(x) * Z_H(x) where t(x) = x^2 + 1
        var evalData = [UInt32]()
        evalData.reserveCapacity(domainSize * 8)
        var expectedQ = [Fr]()
        expectedQ.reserveCapacity(domainSize)

        var cosetPt = g
        for _ in 0..<domainSize {
            let tVal = frAdd(frSqr(cosetPt), frFromInt(1))
            var zh = cosetPt
            for _ in 0..<logTraceLen { zh = frSqr(zh) }
            zh = frSub(zh, Fr.one)
            evalData.append(contentsOf: frToWords(frMul(tVal, zh)))
            expectedQ.append(tVal)
            cosetPt = frMul(cosetPt, omega)
        }

        let cosetBuf = try engine.getOrBuildCosetPoints(logDomain: logDomain,
                                                         cosetGen: cosetGen, field: .bn254)
        let evalsBuf = engine.device.makeBuffer(bytes: evalData,
                                                 length: evalData.count * 4,
                                                 options: .storageModeShared)!

        let result = try engine.computeQuotientFused(constraintEvals: evalsBuf,
                                                      cosetPoints: cosetBuf,
                                                      domainSize: domainSize,
                                                      logTraceLen: logTraceLen,
                                                      field: .bn254)

        // Spot-check first, middle, last
        let outPtr = result.contents().bindMemory(to: UInt32.self, capacity: domainSize * 8)
        let indices = [0, domainSize / 2, domainSize - 1]
        var allOk = true
        for idx in indices {
            let got = wordsToFr(Array(UnsafeBufferPointer(start: outPtr + idx * 8, count: 8)))
            if got != expectedQ[idx] {
                allOk = false
                break
            }
        }
        expect(allOk, "GPU quotient matches CPU reference at spot-check indices (N=1024)")

    } catch {
        expect(false, "GPUQuotientEngine GPU vs CPU failed: \(error)")
    }
}

// MARK: - Performance

private func testLargeQuotientPerformance() {
    suite("GPUQuotientEngine performance (2^14)")
    do {
        let engine = try GPUQuotientEngine()

        let logDomain = 14  // 16384
        let logTraceLen = 10  // trace = 1024
        let domainSize = 1 << logDomain

        let g = frFromInt(7)
        let cosetGen = frToWords(g)

        // Random evals
        var rng = QuotientTestRNG(state: 0x11223344)
        var evalData = [UInt32]()
        evalData.reserveCapacity(domainSize * 8)
        for _ in 0..<domainSize {
            evalData.append(contentsOf: frToWords(rng.nextFr()))
        }

        let cosetBuf = try engine.getOrBuildCosetPoints(logDomain: logDomain,
                                                         cosetGen: cosetGen, field: .bn254)
        let evalsBuf = engine.device.makeBuffer(bytes: evalData,
                                                 length: evalData.count * 4,
                                                 options: .storageModeShared)!

        // Warmup
        let _ = try engine.computeQuotientFused(constraintEvals: evalsBuf,
                                                 cosetPoints: cosetBuf,
                                                 domainSize: domainSize,
                                                 logTraceLen: logTraceLen,
                                                 field: .bn254)

        // Timed fused pass
        let t0 = CFAbsoluteTimeGetCurrent()
        let _ = try engine.computeQuotientFused(constraintEvals: evalsBuf,
                                                 cosetPoints: cosetBuf,
                                                 domainSize: domainSize,
                                                 logTraceLen: logTraceLen,
                                                 field: .bn254)
        let fusedTime = CFAbsoluteTimeGetCurrent() - t0

        // Timed precomputed pass
        let zhInvBuf = try engine.precomputeVanishingInverses(logDomain: logDomain,
                                                               logTraceLen: logTraceLen,
                                                               cosetGen: cosetGen,
                                                               field: .bn254)

        // Re-create evalsBuf for the precomputed pass
        let evalsBuf2 = engine.device.makeBuffer(bytes: evalData,
                                                  length: evalData.count * 4,
                                                  options: .storageModeShared)!

        let t1 = CFAbsoluteTimeGetCurrent()
        let _ = try engine.computeQuotient(constraintEvals: evalsBuf2,
                                            vanishingInverses: zhInvBuf,
                                            domainSize: domainSize,
                                            field: .bn254)
        let precompTime = CFAbsoluteTimeGetCurrent() - t1

        print(String(format: "  Fused quotient (2^14 BN254): %.2fms", fusedTime * 1000))
        print(String(format: "  Precomputed quotient (2^14 BN254): %.2fms", precompTime * 1000))

        expect(true, "performance test completed")

    } catch {
        expect(false, "GPUQuotientEngine performance test failed: \(error)")
    }
}
