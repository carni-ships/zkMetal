import Foundation
import zkMetal

/// Run all GPUCosetNTTEngine tests.
/// Verifies GPU coset NTT against manual (shift + regular NTT),
/// round-trip cosetNTT->cosetINTT, coset LDE correctness, and performance.
public func runGPUCosetNTTEngineTests() {
    // -------------------------------------------------------
    // BN254 Fr: cosetNTT matches manual shift + NTT
    // -------------------------------------------------------
    suite("GPUCosetNTTEngine BN254 cosetNTT vs manual shift+NTT")
    do {
        let engine = try GPUCosetNTTEngine()
        let nttEngine = try NTTEngine()
        let logN = 10
        let n = 1 << logN
        let shift = frFromInt(Fr.GENERATOR)

        // Random coefficients
        var coeffs = [Fr](repeating: Fr.zero, count: n)
        var rng: UInt64 = 0xDEAD_BEEF
        for i in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            coeffs[i] = frFromInt(rng >> 32)
        }

        // GPU coset NTT
        let gpuResult = try engine.cosetNTT(coeffs: coeffs, shift: shift)

        // Manual: shift then NTT
        var shifted = [Fr](repeating: Fr.zero, count: n)
        var sPow = Fr.one
        for i in 0..<n {
            shifted[i] = frMul(coeffs[i], sPow)
            sPow = frMul(sPow, shift)
        }
        let manualResult = try nttEngine.ntt(shifted)

        expectEqual(gpuResult.count, manualResult.count, "BN254 cosetNTT output size")
        var match = true
        for i in 0..<gpuResult.count {
            let g = frToInt(gpuResult[i])
            let m = frToInt(manualResult[i])
            if g[0] != m[0] || g[1] != m[1] || g[2] != m[2] || g[3] != m[3] {
                match = false; break
            }
        }
        expect(match, "BN254 GPU cosetNTT matches manual shift+NTT")

    } catch {
        expect(false, "BN254 cosetNTT vs manual error: \(error)")
    }

    // -------------------------------------------------------
    // BN254 Fr: cosetNTT pointwise correctness
    // -------------------------------------------------------
    suite("GPUCosetNTTEngine BN254 cosetNTT pointwise")
    do {
        let engine = try GPUCosetNTTEngine()
        let logN = 8
        let n = 1 << logN
        let shift = frFromInt(Fr.GENERATOR)
        let omega = frRootOfUnity(logN: logN)

        // Create polynomial: p(x) = 1 + 2x + 3x^2
        var coeffs = [Fr](repeating: Fr.zero, count: n)
        coeffs[0] = frFromInt(1)
        coeffs[1] = frFromInt(2)
        coeffs[2] = frFromInt(3)

        let cosetEvals = try engine.cosetNTT(coeffs: coeffs, shift: shift)

        // Verify at a few coset points: p(shift * omega^i)
        var ok = true
        for i in stride(from: 0, to: min(n, 32), by: 5) {
            var oi = Fr.one
            for _ in 0..<i { oi = frMul(oi, omega) }
            let point = frMul(shift, oi)

            // Horner: p(point) = 1 + 2*point + 3*point^2
            let p2 = frMul(frFromInt(3), frMul(point, point))
            let p1 = frMul(frFromInt(2), point)
            let expected = frAdd(frAdd(frFromInt(1), p1), p2)

            if frToInt(cosetEvals[i]) != frToInt(expected) { ok = false; break }
        }
        expect(ok, "BN254 cosetNTT matches pointwise evaluation")

    } catch {
        expect(false, "BN254 cosetNTT pointwise error: \(error)")
    }

    // -------------------------------------------------------
    // BN254 Fr: cosetNTT -> cosetINTT round-trip
    // -------------------------------------------------------
    suite("GPUCosetNTTEngine BN254 round-trip")
    do {
        let engine = try GPUCosetNTTEngine()
        let logN = 10
        let n = 1 << logN
        let shift = frFromInt(Fr.GENERATOR)

        var coeffs = [Fr](repeating: Fr.zero, count: n)
        var rng: UInt64 = 0xBEEF_C0DE
        for i in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            coeffs[i] = frFromInt(rng >> 32)
        }

        let cosetEvals = try engine.cosetNTT(coeffs: coeffs, shift: shift)
        let recovered = try engine.cosetINTT(evals: cosetEvals, shift: shift)

        var ok = true
        for i in 0..<n {
            if frToInt(coeffs[i]) != frToInt(recovered[i]) { ok = false; break }
        }
        expect(ok, "BN254 cosetNTT->cosetINTT round-trip 2^10")

    } catch {
        expect(false, "BN254 round-trip error: \(error)")
    }

    // -------------------------------------------------------
    // BN254 Fr: cosetINTT matches manual (INTT + unshift)
    // -------------------------------------------------------
    suite("GPUCosetNTTEngine BN254 cosetINTT vs manual")
    do {
        let engine = try GPUCosetNTTEngine()
        let nttEngine = try NTTEngine()
        let logN = 10
        let n = 1 << logN
        let shift = frFromInt(Fr.GENERATOR)

        // Generate random coset evaluations
        var evals = [Fr](repeating: Fr.zero, count: n)
        var rng: UInt64 = 0xCAFE_BABE
        for i in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            evals[i] = frFromInt(rng >> 32)
        }

        // GPU coset INTT
        let gpuResult = try engine.cosetINTT(evals: evals, shift: shift)

        // Manual: INTT then unshift
        let inttResult = try nttEngine.intt(evals)
        let shiftInv = frInverse(shift)
        var manual = [Fr](repeating: Fr.zero, count: n)
        var sPow = Fr.one
        for i in 0..<n {
            manual[i] = frMul(inttResult[i], sPow)
            sPow = frMul(sPow, shiftInv)
        }

        var match = true
        for i in 0..<n {
            let g = frToInt(gpuResult[i])
            let m = frToInt(manual[i])
            if g[0] != m[0] || g[1] != m[1] || g[2] != m[2] || g[3] != m[3] {
                match = false; break
            }
        }
        expect(match, "BN254 GPU cosetINTT matches manual INTT+unshift")

    } catch {
        expect(false, "BN254 cosetINTT vs manual error: \(error)")
    }

    // -------------------------------------------------------
    // BabyBear: cosetNTT matches manual shift + NTT
    // -------------------------------------------------------
    suite("GPUCosetNTTEngine BabyBear cosetNTT vs manual shift+NTT")
    do {
        let engine = try GPUCosetNTTEngine()
        let bbEngine = try BabyBearNTTEngine()
        let logN = 12
        let n = 1 << logN
        let shift = Bb(v: Bb.GENERATOR)

        var coeffs = [Bb](repeating: Bb.zero, count: n)
        var rng: UInt64 = 0xFEED_FACE
        for i in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            coeffs[i] = Bb(v: UInt32((rng >> 33) % UInt64(Bb.P)))
        }

        let gpuResult = try engine.cosetNTT(coeffs: coeffs, shift: shift)

        // Manual
        var shifted = [Bb](repeating: Bb.zero, count: n)
        var sPow = Bb.one
        for i in 0..<n {
            shifted[i] = bbMul(coeffs[i], sPow)
            sPow = bbMul(sPow, shift)
        }
        let manualResult = try bbEngine.ntt(shifted)

        var match = true
        for i in 0..<gpuResult.count {
            if gpuResult[i].v != manualResult[i].v { match = false; break }
        }
        expect(match, "BabyBear GPU cosetNTT matches manual shift+NTT")

    } catch {
        expect(false, "BabyBear cosetNTT vs manual error: \(error)")
    }

    // -------------------------------------------------------
    // BabyBear: cosetNTT -> cosetINTT round-trip
    // -------------------------------------------------------
    suite("GPUCosetNTTEngine BabyBear round-trip")
    do {
        let engine = try GPUCosetNTTEngine()
        let logN = 12
        let n = 1 << logN
        let shift = Bb(v: Bb.GENERATOR)

        var coeffs = [Bb](repeating: Bb.zero, count: n)
        var rng: UInt64 = 0xC0FFEE42
        for i in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            coeffs[i] = Bb(v: UInt32((rng >> 33) % UInt64(Bb.P)))
        }

        let cosetEvals = try engine.cosetNTT(coeffs: coeffs, shift: shift)
        let recovered = try engine.cosetINTT(evals: cosetEvals, shift: shift)

        var ok = true
        for i in 0..<n {
            if coeffs[i].v != recovered[i].v { ok = false; break }
        }
        expect(ok, "BabyBear cosetNTT->cosetINTT round-trip 2^12")

    } catch {
        expect(false, "BabyBear round-trip error: \(error)")
    }

    // -------------------------------------------------------
    // BN254 Fr: Coset LDE correctness
    // -------------------------------------------------------
    suite("GPUCosetNTTEngine BN254 cosetLDE")
    do {
        let engine = try GPUCosetNTTEngine()
        let logN = 10
        let n = 1 << logN
        let shift = frFromInt(Fr.GENERATOR)

        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = frFromInt(UInt64(i + 1)) }

        // GPU LDE blowup=2
        let gpuResult = try engine.cosetLDE(evals: evals, blowupFactor: 2, shift: shift)
        let cpuResult = engine.cpuCosetLDEFr(evals: evals, blowupFactor: 2, shift: shift)

        expectEqual(gpuResult.count, 2 * n, "BN254 LDE blowup=2 output size")
        expectEqual(gpuResult.count, cpuResult.count, "BN254 LDE GPU vs CPU size")

        var match = true
        for i in 0..<gpuResult.count {
            let g = frToInt(gpuResult[i])
            let c = frToInt(cpuResult[i])
            if g[0] != c[0] || g[1] != c[1] || g[2] != c[2] || g[3] != c[3] {
                match = false; break
            }
        }
        expect(match, "BN254 cosetLDE blowup=2 GPU vs CPU match")

        // Blowup=4
        let gpuResult4 = try engine.cosetLDE(evals: evals, blowupFactor: 4, shift: shift)
        let cpuResult4 = engine.cpuCosetLDEFr(evals: evals, blowupFactor: 4, shift: shift)
        var match4 = true
        for i in 0..<gpuResult4.count {
            let g = frToInt(gpuResult4[i])
            let c = frToInt(cpuResult4[i])
            if g[0] != c[0] || g[1] != c[1] || g[2] != c[2] || g[3] != c[3] {
                match4 = false; break
            }
        }
        expect(match4, "BN254 cosetLDE blowup=4 GPU vs CPU match")

    } catch {
        expect(false, "BN254 cosetLDE error: \(error)")
    }

    // -------------------------------------------------------
    // BabyBear: Coset LDE correctness
    // -------------------------------------------------------
    suite("GPUCosetNTTEngine BabyBear cosetLDE")
    do {
        let engine = try GPUCosetNTTEngine()
        let logN = 10
        let n = 1 << logN
        let shift = Bb(v: Bb.GENERATOR)

        var evals = [Bb](repeating: Bb.zero, count: n)
        for i in 0..<n { evals[i] = Bb(v: UInt32((i + 1) % Int(Bb.P))) }

        let gpuResult = try engine.cosetLDE(evals: evals, blowupFactor: 2, shift: shift)
        let cpuResult = engine.cpuCosetLDEBb(evals: evals, blowupFactor: 2, shift: shift)

        expectEqual(gpuResult.count, 2 * n, "BabyBear LDE blowup=2 output size")
        var match = true
        for i in 0..<gpuResult.count {
            if gpuResult[i].v != cpuResult[i].v { match = false; break }
        }
        expect(match, "BabyBear cosetLDE blowup=2 GPU vs CPU match")

        // Default shift convenience
        let defaultResult = try engine.cosetLDE(evals: evals, blowupFactor: 2)
        let explicitResult = try engine.cosetLDE(evals: evals, blowupFactor: 2,
                                                  shift: Bb(v: Bb.GENERATOR))
        var defaultMatch = true
        for i in 0..<defaultResult.count {
            if defaultResult[i].v != explicitResult[i].v { defaultMatch = false; break }
        }
        expect(defaultMatch, "BabyBear default shift matches explicit GENERATOR")

    } catch {
        expect(false, "BabyBear cosetLDE error: \(error)")
    }

    // -------------------------------------------------------
    // Custom shift values (non-generator)
    // -------------------------------------------------------
    suite("GPUCosetNTTEngine custom shift values")
    do {
        let engine = try GPUCosetNTTEngine()
        let logN = 8
        let n = 1 << logN

        // Use shift = 7 (arbitrary non-generator)
        let shift7 = frFromInt(7)

        var coeffs = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { coeffs[i] = frFromInt(UInt64(i + 1)) }

        // Round-trip with custom shift
        let evals = try engine.cosetNTT(coeffs: coeffs, shift: shift7)
        let recovered = try engine.cosetINTT(evals: evals, shift: shift7)

        var ok = true
        for i in 0..<n {
            if frToInt(coeffs[i]) != frToInt(recovered[i]) { ok = false; break }
        }
        expect(ok, "BN254 round-trip with custom shift=7")

        // Same for BabyBear
        let bbShift = Bb(v: 11)  // arbitrary
        var bbCoeffs = [Bb](repeating: Bb.zero, count: n)
        for i in 0..<n { bbCoeffs[i] = Bb(v: UInt32((i + 1) % Int(Bb.P))) }

        let bbEvals = try engine.cosetNTT(coeffs: bbCoeffs, shift: bbShift)
        let bbRecovered = try engine.cosetINTT(evals: bbEvals, shift: bbShift)

        var bbOk = true
        for i in 0..<n {
            if bbCoeffs[i].v != bbRecovered[i].v { bbOk = false; break }
        }
        expect(bbOk, "BabyBear round-trip with custom shift=11")

    } catch {
        expect(false, "Custom shift error: \(error)")
    }

    // -------------------------------------------------------
    // CPU fallback path (small inputs)
    // -------------------------------------------------------
    suite("GPUCosetNTTEngine CPU fallback")
    do {
        let engine = try GPUCosetNTTEngine()

        // BN254: n=16 (below GPU threshold of 64)
        let logN = 4
        let n = 1 << logN
        let shift = frFromInt(Fr.GENERATOR)

        var coeffs = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { coeffs[i] = frFromInt(UInt64(i + 1)) }

        let evals = try engine.cosetNTT(coeffs: coeffs, shift: shift)
        let recovered = try engine.cosetINTT(evals: evals, shift: shift)

        var ok = true
        for i in 0..<n {
            if frToInt(coeffs[i]) != frToInt(recovered[i]) { ok = false; break }
        }
        expect(ok, "BN254 CPU fallback round-trip 2^4")

        // BabyBear: n=64 (below GPU threshold of 256)
        let bbLogN = 6
        let bbN = 1 << bbLogN
        let bbShift = Bb(v: Bb.GENERATOR)

        var bbCoeffs = [Bb](repeating: Bb.zero, count: bbN)
        for i in 0..<bbN { bbCoeffs[i] = Bb(v: UInt32(i + 1)) }

        let bbEvals = try engine.cosetNTT(coeffs: bbCoeffs, shift: bbShift)
        let bbRecovered = try engine.cosetINTT(evals: bbEvals, shift: bbShift)

        var bbOk = true
        for i in 0..<bbN {
            if bbCoeffs[i].v != bbRecovered[i].v { bbOk = false; break }
        }
        expect(bbOk, "BabyBear CPU fallback round-trip 2^6")

    } catch {
        expect(false, "CPU fallback error: \(error)")
    }

    // -------------------------------------------------------
    // Performance: GPU cosetNTT vs manual shift+NTT
    // -------------------------------------------------------
    suite("GPUCosetNTTEngine Performance")
    do {
        let engine = try GPUCosetNTTEngine()
        let logN = 14   // 16384 elements
        let n = 1 << logN
        let shift = Bb(v: Bb.GENERATOR)

        var coeffs = [Bb](repeating: Bb.zero, count: n)
        var rng: UInt64 = 0xABCD_EF01
        for i in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            coeffs[i] = Bb(v: UInt32((rng >> 32) % UInt64(Bb.P)))
        }

        // Warmup
        _ = try engine.cosetNTT(coeffs: coeffs, shift: shift)

        // GPU cosetNTT timing
        let gpuStart = CFAbsoluteTimeGetCurrent()
        let gpuResult = try engine.cosetNTT(coeffs: coeffs, shift: shift)
        let gpuTime = CFAbsoluteTimeGetCurrent() - gpuStart

        // CPU timing
        let cpuStart = CFAbsoluteTimeGetCurrent()
        let cpuResult = engine.cpuCosetNTTBb(coeffs: coeffs, shift: shift)
        let cpuTime = CFAbsoluteTimeGetCurrent() - cpuStart

        // Verify
        var perfMatch = true
        for i in 0..<gpuResult.count {
            if gpuResult[i].v != cpuResult[i].v { perfMatch = false; break }
        }
        expect(perfMatch, "Performance test: GPU matches CPU")

        let speedup = cpuTime / max(gpuTime, 1e-9)
        print("  BabyBear 2^14 cosetNTT: GPU=\(String(format: "%.2f", gpuTime * 1000))ms, " +
              "CPU=\(String(format: "%.2f", cpuTime * 1000))ms, " +
              "speedup=\(String(format: "%.1f", speedup))x")

    } catch {
        expect(false, "Performance test error: \(error)")
    }
}
