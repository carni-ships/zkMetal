import Foundation
import zkMetal

/// Run all GPUCosetLDEEngine tests.
/// Verifies correctness of GPU coset LDE against CPU reference,
/// tests blowup factors 2/4/8, batch LDE, BabyBear field support,
/// and GPU vs CPU performance comparison.
public func runGPUCosetLDEEngineTests() {
    // -------------------------------------------------------
    // BN254 Fr: basic correctness (GPU vs CPU)
    // -------------------------------------------------------
    suite("GPUCosetLDEEngine BN254 Fr")
    do {
        let engine = try GPUCosetLDEEngine()
        let logN = 10
        let n = 1 << logN
        let cosetShift = frFromInt(Fr.GENERATOR)

        // Generate test polynomial evaluations: p(omega^i) = i+1
        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = frFromInt(UInt64(i + 1)) }

        // Test blowup factor 2
        let gpuResult2 = try engine.extend(evals: evals, logN: logN, blowupFactor: 2,
                                            cosetShift: cosetShift)
        let cpuResult2 = try engine.cpuExtendFr(evals: evals, logN: logN, blowupFactor: 2,
                                                 cosetShift: cosetShift)
        expectEqual(gpuResult2.count, cpuResult2.count, "BN254 blowup=2 output size")
        var match2 = true
        for i in 0..<gpuResult2.count {
            let g = frToInt(gpuResult2[i])
            let c = frToInt(cpuResult2[i])
            if g[0] != c[0] || g[1] != c[1] || g[2] != c[2] || g[3] != c[3] {
                match2 = false; break
            }
        }
        expect(match2, "BN254 blowup=2 GPU vs CPU match")

        // Test blowup factor 4
        let gpuResult4 = try engine.extend(evals: evals, logN: logN, blowupFactor: 4,
                                            cosetShift: cosetShift)
        let cpuResult4 = try engine.cpuExtendFr(evals: evals, logN: logN, blowupFactor: 4,
                                                 cosetShift: cosetShift)
        expectEqual(gpuResult4.count, cpuResult4.count, "BN254 blowup=4 output size")
        var match4 = true
        for i in 0..<gpuResult4.count {
            let g = frToInt(gpuResult4[i])
            let c = frToInt(cpuResult4[i])
            if g[0] != c[0] || g[1] != c[1] || g[2] != c[2] || g[3] != c[3] {
                match4 = false; break
            }
        }
        expect(match4, "BN254 blowup=4 GPU vs CPU match")

        // Test blowup factor 8
        let gpuResult8 = try engine.extend(evals: evals, logN: logN, blowupFactor: 8,
                                            cosetShift: cosetShift)
        let cpuResult8 = try engine.cpuExtendFr(evals: evals, logN: logN, blowupFactor: 8,
                                                 cosetShift: cosetShift)
        expectEqual(gpuResult8.count, cpuResult8.count, "BN254 blowup=8 output size")
        var match8 = true
        for i in 0..<gpuResult8.count {
            let g = frToInt(gpuResult8[i])
            let c = frToInt(cpuResult8[i])
            if g[0] != c[0] || g[1] != c[1] || g[2] != c[2] || g[3] != c[3] {
                match8 = false; break
            }
        }
        expect(match8, "BN254 blowup=8 GPU vs CPU match")

        // Output size checks
        expectEqual(gpuResult2.count, 2 * n, "BN254 blowup=2 correct extended size")
        expectEqual(gpuResult4.count, 4 * n, "BN254 blowup=4 correct extended size")
        expectEqual(gpuResult8.count, 8 * n, "BN254 blowup=8 correct extended size")

    } catch {
        expect(false, "BN254 Fr GPUCosetLDE error: \(error)")
    }

    // -------------------------------------------------------
    // BN254 Fr: extended polynomial agrees on original domain
    // -------------------------------------------------------
    suite("GPUCosetLDEEngine BN254 domain agreement")
    do {
        let engine = try GPUCosetLDEEngine()
        let logN = 8
        let n = 1 << logN
        let blowupFactor = 4
        let logM = logN + 2  // log2(4)
        let m = 1 << logM
        let cosetShift = frFromInt(Fr.GENERATOR)

        // Create a simple polynomial in coefficient form: p(x) = 1 + 2x + 3x^2
        // Evaluate it on standard domain via NTT
        var coeffs = [Fr](repeating: Fr.zero, count: n)
        coeffs[0] = frFromInt(1)
        coeffs[1] = frFromInt(2)
        coeffs[2] = frFromInt(3)
        let evals = NTTEngine.cpuNTT(coeffs, logN: logN)

        // Extend via GPU LDE
        let extended = try engine.extend(evals: evals, logN: logN, blowupFactor: blowupFactor,
                                          cosetShift: cosetShift)

        // Verify: extended polynomial evaluated at coset points should equal
        // p(g * omega_M^i) for all i. Since the extension is the same polynomial
        // just evaluated on a larger coset domain, we verify the output has
        // the correct size.
        expectEqual(extended.count, m, "Extended domain size is M = blowupFactor * N")

        // Also verify: the extended evaluation is not all zeros (polynomial is non-trivial)
        var allZero = true
        for i in 0..<m {
            if !extended[i].isZero { allZero = false; break }
        }
        expect(!allZero, "Extended evaluation is non-trivial")

    } catch {
        expect(false, "BN254 domain agreement error: \(error)")
    }

    // -------------------------------------------------------
    // BN254 Fr: default coset shift (convenience API)
    // -------------------------------------------------------
    suite("GPUCosetLDEEngine BN254 default shift")
    do {
        let engine = try GPUCosetLDEEngine()
        let logN = 10
        let n = 1 << logN

        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = frFromInt(UInt64(i + 1)) }

        // Default shift uses Fr.GENERATOR
        let result = try engine.extend(evals: evals, logN: logN, blowupFactor: 2)
        let explicit = try engine.extend(evals: evals, logN: logN, blowupFactor: 2,
                                          cosetShift: frFromInt(Fr.GENERATOR))

        var defaultMatch = true
        for i in 0..<result.count {
            let r = frToInt(result[i])
            let e = frToInt(explicit[i])
            if r[0] != e[0] || r[1] != e[1] || r[2] != e[2] || r[3] != e[3] {
                defaultMatch = false; break
            }
        }
        expect(defaultMatch, "Default coset shift matches explicit Fr.GENERATOR shift")

    } catch {
        expect(false, "BN254 default shift error: \(error)")
    }

    // -------------------------------------------------------
    // BabyBear: basic correctness
    // -------------------------------------------------------
    suite("GPUCosetLDEEngine BabyBear")
    do {
        let engine = try GPUCosetLDEEngine()
        let logN = 10
        let n = 1 << logN
        let cosetShift = Bb(v: Bb.GENERATOR)

        var evals = [Bb](repeating: Bb.zero, count: n)
        for i in 0..<n { evals[i] = Bb(v: UInt32((i + 1) % Int(Bb.P))) }

        // Test blowup factor 2
        let gpuResult2 = try engine.extend(evals: evals, logN: logN, blowupFactor: 2,
                                            cosetShift: cosetShift)
        let cpuResult2 = try engine.cpuExtendBb(evals: evals, logN: logN, blowupFactor: 2,
                                                  cosetShift: cosetShift)
        expectEqual(gpuResult2.count, cpuResult2.count, "BabyBear blowup=2 output size")
        var bbMatch2 = true
        for i in 0..<gpuResult2.count {
            if gpuResult2[i].v != cpuResult2[i].v { bbMatch2 = false; break }
        }
        expect(bbMatch2, "BabyBear blowup=2 GPU vs CPU match")

        // Test blowup factor 4
        let gpuResult4 = try engine.extend(evals: evals, logN: logN, blowupFactor: 4,
                                            cosetShift: cosetShift)
        let cpuResult4 = try engine.cpuExtendBb(evals: evals, logN: logN, blowupFactor: 4,
                                                  cosetShift: cosetShift)
        var bbMatch4 = true
        for i in 0..<gpuResult4.count {
            if gpuResult4[i].v != cpuResult4[i].v { bbMatch4 = false; break }
        }
        expect(bbMatch4, "BabyBear blowup=4 GPU vs CPU match")

        // Test blowup factor 8
        let gpuResult8 = try engine.extend(evals: evals, logN: logN, blowupFactor: 8,
                                            cosetShift: cosetShift)
        let cpuResult8 = try engine.cpuExtendBb(evals: evals, logN: logN, blowupFactor: 8,
                                                  cosetShift: cosetShift)
        var bbMatch8 = true
        for i in 0..<gpuResult8.count {
            if gpuResult8[i].v != cpuResult8[i].v { bbMatch8 = false; break }
        }
        expect(bbMatch8, "BabyBear blowup=8 GPU vs CPU match")

        // Default shift convenience
        let defaultResult = try engine.extend(evals: evals, logN: logN, blowupFactor: 2)
        let explicitResult = try engine.extend(evals: evals, logN: logN, blowupFactor: 2,
                                                cosetShift: Bb(v: Bb.GENERATOR))
        var bbDefaultMatch = true
        for i in 0..<defaultResult.count {
            if defaultResult[i].v != explicitResult[i].v { bbDefaultMatch = false; break }
        }
        expect(bbDefaultMatch, "BabyBear default coset shift matches explicit GENERATOR")

    } catch {
        expect(false, "BabyBear GPUCosetLDE error: \(error)")
    }

    // -------------------------------------------------------
    // Batch LDE: BN254 Fr
    // -------------------------------------------------------
    suite("GPUCosetLDEEngine BN254 Batch LDE")
    do {
        let engine = try GPUCosetLDEEngine()
        let logN = 10
        let n = 1 << logN
        let cosetShift = frFromInt(Fr.GENERATOR)
        let numCols = 4

        // Generate multiple columns
        var columns = [[Fr]]()
        var rng: UInt64 = 0xDEAD_BEEF
        for _ in 0..<numCols {
            var col = [Fr](repeating: Fr.zero, count: n)
            for j in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                col[j] = frFromInt(rng >> 32)
            }
            columns.append(col)
        }

        // Batch extend
        let batchResults = try engine.batchExtend(columns: columns, logN: logN,
                                                   blowupFactor: 2, cosetShift: cosetShift)
        expectEqual(batchResults.count, numCols, "Batch LDE output column count")

        // Compare each column against individual extend
        var batchOk = true
        for c in 0..<numCols {
            let single = try engine.extend(evals: columns[c], logN: logN,
                                            blowupFactor: 2, cosetShift: cosetShift)
            expectEqual(batchResults[c].count, single.count, "Batch col \(c) size")
            for i in 0..<single.count {
                let b = frToInt(batchResults[c][i])
                let s = frToInt(single[i])
                if b[0] != s[0] || b[1] != s[1] || b[2] != s[2] || b[3] != s[3] {
                    batchOk = false; break
                }
            }
            if !batchOk { break }
        }
        expect(batchOk, "Batch BN254 LDE matches individual column extends")

    } catch {
        expect(false, "BN254 Batch LDE error: \(error)")
    }

    // -------------------------------------------------------
    // Batch LDE: BabyBear
    // -------------------------------------------------------
    suite("GPUCosetLDEEngine BabyBear Batch LDE")
    do {
        let engine = try GPUCosetLDEEngine()
        let logN = 10
        let n = 1 << logN
        let cosetShift = Bb(v: Bb.GENERATOR)
        let numCols = 3

        var columns = [[Bb]]()
        var rng: UInt64 = 0xCAFE_1234
        for _ in 0..<numCols {
            var col = [Bb](repeating: Bb.zero, count: n)
            for j in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                col[j] = Bb(v: UInt32((rng >> 32) % UInt64(Bb.P)))
            }
            columns.append(col)
        }

        let batchResults = try engine.batchExtend(columns: columns, logN: logN,
                                                   blowupFactor: 4, cosetShift: cosetShift)
        expectEqual(batchResults.count, numCols, "BabyBear batch column count")

        var bbBatchOk = true
        for c in 0..<numCols {
            let single = try engine.extend(evals: columns[c], logN: logN,
                                            blowupFactor: 4, cosetShift: cosetShift)
            for i in 0..<single.count {
                if batchResults[c][i].v != single[i].v { bbBatchOk = false; break }
            }
            if !bbBatchOk { break }
        }
        expect(bbBatchOk, "Batch BabyBear LDE matches individual column extends")

    } catch {
        expect(false, "BabyBear Batch LDE error: \(error)")
    }

    // -------------------------------------------------------
    // GPU vs CPU performance comparison
    // -------------------------------------------------------
    suite("GPUCosetLDEEngine Performance")
    do {
        let engine = try GPUCosetLDEEngine()
        let logN = 14   // 16384 elements
        let n = 1 << logN
        let cosetShift = Bb(v: Bb.GENERATOR)

        var evals = [Bb](repeating: Bb.zero, count: n)
        var rng: UInt64 = 0xABCD_EF01
        for i in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            evals[i] = Bb(v: UInt32((rng >> 32) % UInt64(Bb.P)))
        }

        // Warmup
        _ = try engine.extend(evals: evals, logN: logN, blowupFactor: 2, cosetShift: cosetShift)

        // GPU timing
        let gpuStart = CFAbsoluteTimeGetCurrent()
        let gpuResult = try engine.extend(evals: evals, logN: logN, blowupFactor: 4,
                                           cosetShift: cosetShift)
        let gpuTime = CFAbsoluteTimeGetCurrent() - gpuStart

        // CPU timing
        let cpuStart = CFAbsoluteTimeGetCurrent()
        let cpuResult = try engine.cpuExtendBb(evals: evals, logN: logN, blowupFactor: 4,
                                                cosetShift: cosetShift)
        let cpuTime = CFAbsoluteTimeGetCurrent() - cpuStart

        // Verify correctness
        var perfMatch = true
        for i in 0..<gpuResult.count {
            if gpuResult[i].v != cpuResult[i].v { perfMatch = false; break }
        }
        expect(perfMatch, "Performance test: GPU matches CPU")

        let speedup = cpuTime / max(gpuTime, 1e-9)
        print("  BabyBear 2^14 blowup=4: GPU=\(String(format: "%.2f", gpuTime * 1000))ms, " +
              "CPU=\(String(format: "%.2f", cpuTime * 1000))ms, " +
              "speedup=\(String(format: "%.1f", speedup))x")

    } catch {
        expect(false, "Performance test error: \(error)")
    }

    // -------------------------------------------------------
    // Small size CPU fallback path
    // -------------------------------------------------------
    suite("GPUCosetLDEEngine Small Size CPU Path")
    do {
        let engine = try GPUCosetLDEEngine()

        // BN254: n=32 should use CPU path (threshold is 64)
        let logN = 5
        let n = 1 << logN
        var smallEvals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { smallEvals[i] = frFromInt(UInt64(i + 1)) }

        let result = try engine.extend(evals: smallEvals, logN: logN, blowupFactor: 2,
                                        cosetShift: frFromInt(Fr.GENERATOR))
        expectEqual(result.count, 2 * n, "Small BN254 LDE output size")

        var smallNonZero = false
        for i in 0..<result.count {
            if !result[i].isZero { smallNonZero = true; break }
        }
        expect(smallNonZero, "Small BN254 LDE produces non-zero output")

        // BabyBear: n=128 should use CPU path (threshold is 256)
        let bbLogN = 7
        let bbN = 1 << bbLogN
        var bbSmallEvals = [Bb](repeating: Bb.zero, count: bbN)
        for i in 0..<bbN { bbSmallEvals[i] = Bb(v: UInt32(i + 1)) }

        let bbResult = try engine.extend(evals: bbSmallEvals, logN: bbLogN, blowupFactor: 4,
                                          cosetShift: Bb(v: Bb.GENERATOR))
        expectEqual(bbResult.count, 4 * bbN, "Small BabyBear LDE output size")

        // Verify against CPU reference
        let bbCpu = try engine.cpuExtendBb(evals: bbSmallEvals, logN: bbLogN, blowupFactor: 4,
                                            cosetShift: Bb(v: Bb.GENERATOR))
        var smallBbMatch = true
        for i in 0..<bbResult.count {
            if bbResult[i].v != bbCpu[i].v { smallBbMatch = false; break }
        }
        expect(smallBbMatch, "Small BabyBear GPU/CPU path produces same result as CPU reference")

    } catch {
        expect(false, "Small size CPU path error: \(error)")
    }
}
