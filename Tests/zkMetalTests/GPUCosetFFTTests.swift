import Foundation
import zkMetal

private func frEqual(_ a: Fr, _ b: Fr) -> Bool {
    return frToInt(a) == frToInt(b)
}

/// Run all GPUCosetFFTEngine tests.
public func runGPUCosetFFTTests() {
    // -------------------------------------------------------
    // Coset FFT round-trip: cosetFFT -> cosetIFFT recovers original
    // -------------------------------------------------------
    suite("GPUCosetFFTEngine round-trip")
    do {
        let engine = try GPUCosetFFTEngine()
        let logN = 10
        let n = 1 << logN
        let shift = frFromInt(Fr.GENERATOR)

        var coeffs = [Fr](repeating: Fr.zero, count: n)
        var rng: UInt64 = 0xDEAD_BEEF
        for i in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            coeffs[i] = frFromInt(rng >> 32)
        }

        let cosetEvals = try engine.cosetFFT(coeffs: coeffs, shift: shift)
        let recovered = try engine.cosetIFFT(evals: cosetEvals, shift: shift)

        expectEqual(recovered.count, n, "round-trip output size")
        var ok = true
        for i in 0..<n {
            if !frEqual(coeffs[i], recovered[i]) { ok = false; break }
        }
        expect(ok, "cosetFFT->cosetIFFT round-trip 2^10")

    } catch {
        expect(false, "round-trip error: \(error)")
    }

    // -------------------------------------------------------
    // Coset FFT vs standard FFT: results should differ
    // -------------------------------------------------------
    suite("GPUCosetFFTEngine coset vs standard FFT difference")
    do {
        let engine = try GPUCosetFFTEngine()
        let fftEngine = try GPUFFTEngine()
        let logN = 8
        let n = 1 << logN
        let shift = frFromInt(Fr.GENERATOR)

        // Non-trivial polynomial
        var coeffs = [Fr](repeating: Fr.zero, count: n)
        coeffs[0] = frFromInt(1)
        coeffs[1] = frFromInt(2)
        coeffs[2] = frFromInt(3)

        let cosetResult = try engine.cosetFFT(coeffs: coeffs, shift: shift)
        let standardResult = try fftEngine.fft(data: coeffs, logN: logN, inverse: false)

        // They should differ (coset shifts the evaluation domain)
        var allSame = true
        for i in 0..<n {
            if !frEqual(cosetResult[i], standardResult[i]) { allSame = false; break }
        }
        expect(!allSame, "coset FFT differs from standard FFT")

    } catch {
        expect(false, "coset vs standard error: \(error)")
    }

    // -------------------------------------------------------
    // Blowup factor 2 LDE
    // -------------------------------------------------------
    suite("GPUCosetFFTEngine LDE blowup=2")
    do {
        let engine = try GPUCosetFFTEngine()
        let logN = 8
        let n = 1 << logN
        let shift = frFromInt(Fr.GENERATOR)

        // Polynomial coefficients
        var coeffs = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { coeffs[i] = frFromInt(UInt64(i + 1)) }

        let gpuResult = try engine.cosetLDE(coeffs: coeffs, blowupFactor: 2, shift: shift)
        let cpuResult = engine.cpuCosetLDE(coeffs: coeffs, blowupFactor: 2, shift: shift)

        expectEqual(gpuResult.count, 2 * n, "LDE blowup=2 output size")
        expectEqual(gpuResult.count, cpuResult.count, "LDE GPU vs CPU size")

        var match = true
        for i in 0..<gpuResult.count {
            if !frEqual(gpuResult[i], cpuResult[i]) { match = false; break }
        }
        expect(match, "LDE blowup=2 GPU vs CPU match")

    } catch {
        expect(false, "LDE blowup=2 error: \(error)")
    }

    // -------------------------------------------------------
    // Blowup factor 4 LDE
    // -------------------------------------------------------
    suite("GPUCosetFFTEngine LDE blowup=4")
    do {
        let engine = try GPUCosetFFTEngine()
        let logN = 8
        let n = 1 << logN
        let shift = frFromInt(Fr.GENERATOR)

        var coeffs = [Fr](repeating: Fr.zero, count: n)
        var rng: UInt64 = 0xCAFE_BABE
        for i in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            coeffs[i] = frFromInt(rng >> 32)
        }

        let gpuResult = try engine.cosetLDE(coeffs: coeffs, blowupFactor: 4, shift: shift)
        let cpuResult = engine.cpuCosetLDE(coeffs: coeffs, blowupFactor: 4, shift: shift)

        expectEqual(gpuResult.count, 4 * n, "LDE blowup=4 output size")

        var match = true
        for i in 0..<gpuResult.count {
            if !frEqual(gpuResult[i], cpuResult[i]) { match = false; break }
        }
        expect(match, "LDE blowup=4 GPU vs CPU match")

    } catch {
        expect(false, "LDE blowup=4 error: \(error)")
    }

    // -------------------------------------------------------
    // Batch coset FFT consistency
    // -------------------------------------------------------
    suite("GPUCosetFFTEngine batch coset FFT")
    do {
        let engine = try GPUCosetFFTEngine()
        let logN = 8
        let n = 1 << logN
        let shift = frFromInt(Fr.GENERATOR)
        let numCols = 4

        // Generate columns
        var columns = [[Fr]]()
        var rng: UInt64 = 0xBEEF_C0DE
        for _ in 0..<numCols {
            var col = [Fr](repeating: Fr.zero, count: n)
            for j in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                col[j] = frFromInt(rng >> 32)
            }
            columns.append(col)
        }

        // Batch coset FFT
        let batchResults = try engine.batchCosetFFT(columns: columns, shift: shift)
        expectEqual(batchResults.count, numCols, "batch output column count")

        // Compare with individual coset FFTs
        var allMatch = true
        for c in 0..<numCols {
            let individual = try engine.cosetFFT(coeffs: columns[c], shift: shift)
            expectEqual(batchResults[c].count, individual.count, "batch col \(c) size")
            for i in 0..<individual.count {
                if !frEqual(batchResults[c][i], individual[i]) { allMatch = false; break }
            }
        }
        expect(allMatch, "batch coset FFT matches individual coset FFTs")

    } catch {
        expect(false, "batch coset FFT error: \(error)")
    }

    // -------------------------------------------------------
    // Known polynomial evaluation on coset
    // -------------------------------------------------------
    suite("GPUCosetFFTEngine known polynomial evaluation")
    do {
        let engine = try GPUCosetFFTEngine()
        let logN = 8
        let n = 1 << logN
        let shift = frFromInt(Fr.GENERATOR)
        let omega = frRootOfUnity(logN: logN)

        // p(x) = 1 + 2x + 3x^2
        var coeffs = [Fr](repeating: Fr.zero, count: n)
        coeffs[0] = frFromInt(1)
        coeffs[1] = frFromInt(2)
        coeffs[2] = frFromInt(3)

        let cosetEvals = try engine.cosetFFT(coeffs: coeffs, shift: shift)

        // Verify at several coset points: p(shift * omega^i)
        var ok = true
        for i in stride(from: 0, to: min(n, 32), by: 7) {
            var oi = Fr.one
            for _ in 0..<i { oi = frMul(oi, omega) }
            let point = frMul(shift, oi)

            // Horner: p(point) = 1 + 2*point + 3*point^2
            let p2 = frMul(frFromInt(3), frMul(point, point))
            let p1 = frMul(frFromInt(2), point)
            let expected = frAdd(frAdd(frFromInt(1), p1), p2)

            if !frEqual(cosetEvals[i], expected) { ok = false; break }
        }
        expect(ok, "known polynomial p(x)=1+2x+3x^2 evaluates correctly on coset")

    } catch {
        expect(false, "known polynomial error: \(error)")
    }

    // -------------------------------------------------------
    // CPU fallback path (small inputs)
    // -------------------------------------------------------
    suite("GPUCosetFFTEngine CPU fallback")
    do {
        let engine = try GPUCosetFFTEngine()
        let logN = 4
        let n = 1 << logN
        let shift = frFromInt(Fr.GENERATOR)

        var coeffs = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { coeffs[i] = frFromInt(UInt64(i + 1)) }

        let evals = try engine.cosetFFT(coeffs: coeffs, shift: shift)
        let recovered = try engine.cosetIFFT(evals: evals, shift: shift)

        var ok = true
        for i in 0..<n {
            if !frEqual(coeffs[i], recovered[i]) { ok = false; break }
        }
        expect(ok, "CPU fallback round-trip 2^4")

    } catch {
        expect(false, "CPU fallback error: \(error)")
    }

    // -------------------------------------------------------
    // Custom shift value round-trip
    // -------------------------------------------------------
    suite("GPUCosetFFTEngine custom shift")
    do {
        let engine = try GPUCosetFFTEngine()
        let logN = 8
        let n = 1 << logN
        let shift7 = frFromInt(7)

        var coeffs = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { coeffs[i] = frFromInt(UInt64(i + 1)) }

        let evals = try engine.cosetFFT(coeffs: coeffs, shift: shift7)
        let recovered = try engine.cosetIFFT(evals: evals, shift: shift7)

        var ok = true
        for i in 0..<n {
            if !frEqual(coeffs[i], recovered[i]) { ok = false; break }
        }
        expect(ok, "round-trip with custom shift=7")

    } catch {
        expect(false, "custom shift error: \(error)")
    }
}
