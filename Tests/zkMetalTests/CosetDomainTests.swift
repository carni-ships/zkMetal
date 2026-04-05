import zkMetal
import Foundation

public func runCosetDomainTests() {
    suite("CosetDomain: Shift then Unshift recovers original (BN254)")
    do {
        let engine = try CosetDomainEngine()
        let logN = 10
        let n = 1 << logN
        let g = frFromInt(Fr.GENERATOR)

        // Create random-ish input
        var input = [Fr](repeating: Fr.zero, count: n)
        var rng: UInt64 = 0xDEAD_BEEF
        for i in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            input[i] = frFromInt(rng >> 32)
        }

        let shifted = try engine.cosetShift(evals: input, logSize: logN, generator: g)
        let recovered = try engine.cosetUnshift(evals: shifted, logSize: logN, generator: g)

        var ok = true
        for i in 0..<n {
            if frToInt(input[i]) != frToInt(recovered[i]) { ok = false; break }
        }
        expect(ok, "BN254 shift->unshift round-trip 2^10")

        // Verify shifted != original (non-trivial operation)
        var allSame = true
        for i in 1..<n {
            if frToInt(shifted[i]) != frToInt(input[i]) { allSame = false; break }
        }
        expect(!allSame, "BN254 shift actually modifies data")

    } catch { expect(false, "BN254 shift/unshift error: \(error)") }

    suite("CosetDomain: Shift then Unshift recovers original (BabyBear)")
    do {
        let engine = try CosetDomainEngine()
        let logN = 12
        let n = 1 << logN
        let g = Bb(v: Bb.GENERATOR)

        var input = [Bb](repeating: Bb.zero, count: n)
        var rng: UInt64 = 0xCAFE_1234
        for i in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            input[i] = Bb(v: UInt32((rng >> 33) % UInt64(Bb.P)))
        }

        let shifted = try engine.cosetShift(evals: input, logSize: logN, generator: g)
        let recovered = try engine.cosetUnshift(evals: shifted, logSize: logN, generator: g)

        var ok = true
        for i in 0..<n {
            if input[i].v != recovered[i].v { ok = false; break }
        }
        expect(ok, "BabyBear shift->unshift round-trip 2^12")
    } catch { expect(false, "BabyBear shift/unshift error: \(error)") }

    suite("CosetDomain: Vanishing polynomial (BN254)")
    do {
        let engine = try CosetDomainEngine()
        let logN = 8
        let n = 1 << logN

        // Z_H(omega^i) = (omega^i)^n - 1 = (omega^n)^i - 1 = 1^i - 1 = 0
        let omega = frRootOfUnity(logN: logN)
        var roots = [Fr](repeating: Fr.zero, count: n)
        var omegaPow = Fr.one
        for i in 0..<n {
            roots[i] = omegaPow
            omegaPow = frMul(omegaPow, omega)
        }

        let zhAtRoots = try engine.evaluateVanishing(points: roots, domainSize: n)
        var allZero = true
        for i in 0..<n {
            if !zhAtRoots[i].isZero { allZero = false; break }
        }
        expect(allZero, "Z_H(omega^i) = 0 for all roots of unity")

        // Z_H(g * omega^i) != 0 for coset points (g = multiplicative generator)
        let g = frFromInt(Fr.GENERATOR)
        var cosetPoints = [Fr](repeating: Fr.zero, count: n)
        omegaPow = Fr.one
        for i in 0..<n {
            cosetPoints[i] = frMul(g, omegaPow)
            omegaPow = frMul(omegaPow, omega)
        }

        let zhAtCoset = try engine.evaluateVanishing(points: cosetPoints, domainSize: n)
        var anyZero = false
        for i in 0..<n {
            if zhAtCoset[i].isZero { anyZero = true; break }
        }
        expect(!anyZero, "Z_H(g*omega^i) != 0 for all coset points")

        // All coset vanishing values should be equal (g^n - 1)
        var gn = g
        for _ in 0..<logN {
            gn = frSqr(gn)
        }
        let expected = frSub(gn, Fr.one)
        var allEqual = true
        for i in 0..<n {
            if frToInt(zhAtCoset[i]) != frToInt(expected) { allEqual = false; break }
        }
        expect(allEqual, "Z_H constant on coset = g^n - 1")

    } catch { expect(false, "BN254 vanishing error: \(error)") }

    suite("CosetDomain: Vanishing polynomial (BabyBear)")
    do {
        let engine = try CosetDomainEngine()
        let logN = 8
        let n = 1 << logN

        let omega = bbRootOfUnity(logN: logN)
        var roots = [Bb](repeating: Bb.zero, count: n)
        var omegaPow = Bb.one
        for i in 0..<n {
            roots[i] = omegaPow
            omegaPow = bbMul(omegaPow, omega)
        }

        let zhAtRoots = try engine.evaluateVanishing(points: roots, domainSize: n)
        var allZero = true
        for i in 0..<n {
            if !zhAtRoots[i].isZero { allZero = false; break }
        }
        expect(allZero, "BabyBear Z_H(omega^i) = 0 for all roots")

        let g = Bb(v: Bb.GENERATOR)
        var cosetPoints = [Bb](repeating: Bb.zero, count: n)
        omegaPow = Bb.one
        for i in 0..<n {
            cosetPoints[i] = bbMul(g, omegaPow)
            omegaPow = bbMul(omegaPow, omega)
        }

        let zhAtCoset = try engine.evaluateVanishing(points: cosetPoints, domainSize: n)
        var anyZero = false
        for i in 0..<n {
            if zhAtCoset[i].isZero { anyZero = true; break }
        }
        expect(!anyZero, "BabyBear Z_H(g*omega^i) != 0 on coset")

    } catch { expect(false, "BabyBear vanishing error: \(error)") }

    suite("CosetDomain: Divide by vanishing (BN254)")
    do {
        let engine = try CosetDomainEngine()
        let logN = 10
        let n = 1 << logN
        let g = frFromInt(Fr.GENERATOR)

        // Create random evals
        var evals = [Fr](repeating: Fr.zero, count: n)
        var rng: UInt64 = 0xABCD_5678
        for i in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            evals[i] = frFromInt(rng >> 32)
        }

        let divided = try engine.divideByVanishing(evals: evals, logSize: logN, cosetGen: g)

        // Manual check: divided[i] should equal evals[i] * (g^n - 1)^(-1)
        var gn = g
        for _ in 0..<logN {
            gn = frSqr(gn)
        }
        let zh = frSub(gn, Fr.one)
        let zhInv = frInverse(zh)

        var ok = true
        // Check a sample of points
        for i in stride(from: 0, to: n, by: 17) {
            let expected = frMul(evals[i], zhInv)
            if frToInt(divided[i]) != frToInt(expected) { ok = false; break }
        }
        expect(ok, "BN254 divide by vanishing matches manual")

    } catch { expect(false, "BN254 div vanishing error: \(error)") }

    suite("CosetDomain: Divide by vanishing (BabyBear)")
    do {
        let engine = try CosetDomainEngine()
        let logN = 10
        let n = 1 << logN
        let g = Bb(v: Bb.GENERATOR)

        var evals = [Bb](repeating: Bb.zero, count: n)
        var rng: UInt64 = 0x1234_ABCD
        for i in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            evals[i] = Bb(v: UInt32((rng >> 33) % UInt64(Bb.P)))
        }

        let divided = try engine.divideByVanishing(evals: evals, logSize: logN, cosetGen: g)

        // Manual: zh = g^n - 1
        var gn = g
        for _ in 0..<logN {
            gn = bbSqr(gn)
        }
        let zh = bbSub(gn, Bb.one)
        let zhInv = bbInverse(zh)

        var ok = true
        for i in stride(from: 0, to: n, by: 17) {
            let expected = bbMul(evals[i], zhInv)
            if divided[i].v != expected.v { ok = false; break }
        }
        expect(ok, "BabyBear divide by vanishing matches manual")

    } catch { expect(false, "BabyBear div vanishing error: \(error)") }

    suite("CosetDomain: CosetNTT (BN254)")
    do {
        let engine = try CosetDomainEngine()
        let logN = 10
        let n = 1 << logN
        let g = frFromInt(Fr.GENERATOR)
        let omega = frRootOfUnity(logN: logN)

        // Create polynomial coefficients
        var coeffs = [Fr](repeating: Fr.zero, count: n)
        var rng: UInt64 = 0xFEED_FACE
        for i in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            coeffs[i] = frFromInt(rng >> 32)
        }

        // CosetNTT should give evaluations on coset {g * omega^i}
        let cosetEvals = try engine.cosetNTT(coeffs: coeffs, logSize: logN, cosetGen: g)

        // Verify: manually evaluate polynomial at a few coset points
        // p(x) = sum_{j=0}^{n-1} coeffs[j] * x^j
        // Evaluate at x = g * omega^i for a few values of i
        var omegaPow = Fr.one
        var ok = true
        for i in stride(from: 0, to: min(n, 64), by: 7) {
            // Compute g * omega^i
            var oi = Fr.one
            for _ in 0..<i { oi = frMul(oi, omega) }
            let point = frMul(g, oi)

            // Evaluate polynomial at point using Horner's method
            var val = Fr.zero
            for j in stride(from: n - 1, through: 0, by: -1) {
                val = frMul(val, point)
                val = frAdd(val, coeffs[j])
            }

            if frToInt(cosetEvals[i]) != frToInt(val) { ok = false; break }
        }
        expect(ok, "BN254 cosetNTT matches pointwise evaluation")

    } catch { expect(false, "BN254 cosetNTT error: \(error)") }

    suite("CosetDomain: CosetNTT round-trip (BN254)")
    do {
        let engine = try CosetDomainEngine()
        let logN = 10
        let n = 1 << logN
        let g = frFromInt(Fr.GENERATOR)

        var coeffs = [Fr](repeating: Fr.zero, count: n)
        var rng: UInt64 = 0xBEEF_C0DE
        for i in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            coeffs[i] = frFromInt(rng >> 32)
        }

        let cosetEvals = try engine.cosetNTT(coeffs: coeffs, logSize: logN, cosetGen: g)
        let recovered = try engine.cosetINTT(evals: cosetEvals, logSize: logN, cosetGen: g)

        var ok = true
        for i in 0..<n {
            if frToInt(coeffs[i]) != frToInt(recovered[i]) { ok = false; break }
        }
        expect(ok, "BN254 cosetNTT->cosetINTT round-trip 2^10")

    } catch { expect(false, "BN254 cosetNTT round-trip error: \(error)") }

    suite("CosetDomain: CosetNTT round-trip (BabyBear)")
    do {
        let engine = try CosetDomainEngine()
        let logN = 12
        let n = 1 << logN
        let g = Bb(v: Bb.GENERATOR)

        var coeffs = [Bb](repeating: Bb.zero, count: n)
        var rng: UInt64 = 0xC0FFEE42
        for i in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            coeffs[i] = Bb(v: UInt32((rng >> 33) % UInt64(Bb.P)))
        }

        let cosetEvals = try engine.cosetNTT(coeffs: coeffs, logSize: logN, cosetGen: g)
        let recovered = try engine.cosetINTT(evals: cosetEvals, logSize: logN, cosetGen: g)

        var ok = true
        for i in 0..<n {
            if coeffs[i].v != recovered[i].v { ok = false; break }
        }
        expect(ok, "BabyBear cosetNTT->cosetINTT round-trip 2^12")

    } catch { expect(false, "BabyBear cosetNTT round-trip error: \(error)") }

    suite("CosetDomain: CPU fallback (small inputs)")
    do {
        let engine = try CosetDomainEngine()
        // BN254 small (below threshold of 64)
        let logN = 4
        let n = 1 << logN
        let g = frFromInt(Fr.GENERATOR)

        var input = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { input[i] = frFromInt(UInt64(i + 1)) }

        let shifted = try engine.cosetShift(evals: input, logSize: logN, generator: g)
        let recovered = try engine.cosetUnshift(evals: shifted, logSize: logN, generator: g)

        var ok = true
        for i in 0..<n {
            if frToInt(input[i]) != frToInt(recovered[i]) { ok = false; break }
        }
        expect(ok, "BN254 CPU fallback shift/unshift 2^4")

        // BabyBear small (below threshold of 256)
        let bbLogN = 6
        let bbN = 1 << bbLogN
        let bbG = Bb(v: Bb.GENERATOR)
        var bbInput = [Bb](repeating: Bb.zero, count: bbN)
        for i in 0..<bbN { bbInput[i] = Bb(v: UInt32(i + 1)) }

        let bbShifted = try engine.cosetShift(evals: bbInput, logSize: bbLogN, generator: bbG)
        let bbRecovered = try engine.cosetUnshift(evals: bbShifted, logSize: bbLogN, generator: bbG)

        var bbOk = true
        for i in 0..<bbN {
            if bbInput[i].v != bbRecovered[i].v { bbOk = false; break }
        }
        expect(bbOk, "BabyBear CPU fallback shift/unshift 2^6")

    } catch { expect(false, "CPU fallback error: \(error)") }

    suite("CosetDomain: Performance 2^18")
    do {
        let engine = try CosetDomainEngine()
        let logN = 18
        let n = 1 << logN
        let g = frFromInt(Fr.GENERATOR)

        var input = [Fr](repeating: Fr.zero, count: n)
        var rng: UInt64 = 0x12345678
        for i in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            input[i] = frFromInt(rng >> 32)
        }

        // Warmup
        _ = try engine.cosetShift(evals: input, logSize: logN, generator: g)

        let iters = 3
        let startShift = DispatchTime.now()
        for _ in 0..<iters {
            _ = try engine.cosetShift(evals: input, logSize: logN, generator: g)
        }
        let endShift = DispatchTime.now()
        let shiftMs = Double(endShift.uptimeNanoseconds - startShift.uptimeNanoseconds) / Double(iters) / 1_000_000

        let startDiv = DispatchTime.now()
        for _ in 0..<iters {
            _ = try engine.divideByVanishing(evals: input, logSize: logN, cosetGen: g)
        }
        let endDiv = DispatchTime.now()
        let divMs = Double(endDiv.uptimeNanoseconds - startDiv.uptimeNanoseconds) / Double(iters) / 1_000_000

        print("  BN254 2^18: cosetShift \(String(format: "%.2f", shiftMs))ms, divByVanishing \(String(format: "%.2f", divMs))ms")
        expect(true, "Performance benchmark completed")

        // BabyBear performance
        let bbG = Bb(v: Bb.GENERATOR)
        var bbInput = [Bb](repeating: Bb.zero, count: n)
        rng = 0xABCD_EF01
        for i in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            bbInput[i] = Bb(v: UInt32((rng >> 33) % UInt64(Bb.P)))
        }

        _ = try engine.cosetShift(evals: bbInput, logSize: logN, generator: bbG)

        let startBb = DispatchTime.now()
        for _ in 0..<iters {
            _ = try engine.cosetShift(evals: bbInput, logSize: logN, generator: bbG)
        }
        let endBb = DispatchTime.now()
        let bbMs = Double(endBb.uptimeNanoseconds - startBb.uptimeNanoseconds) / Double(iters) / 1_000_000
        print("  BabyBear 2^18: cosetShift \(String(format: "%.2f", bbMs))ms")

    } catch { expect(false, "Performance test error: \(error)") }
}
