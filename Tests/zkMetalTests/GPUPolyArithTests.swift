import zkMetal
import Foundation

// MARK: - Test helpers

private struct ArithTestRNG {
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
}

private func frEq(_ a: Fr, _ b: Fr) -> Bool {
    a.v.0 == b.v.0 && a.v.1 == b.v.1 && a.v.2 == b.v.2 && a.v.3 == b.v.3 &&
    a.v.4 == b.v.4 && a.v.5 == b.v.5 && a.v.6 == b.v.6 && a.v.7 == b.v.7
}

// MARK: - Public entry point

public func runGPUPolyArithTests() {
    testAddBasic()
    testSubBasic()
    testScaleBasic()
    testMulBasic()
    testMulLinearTimesLinear()
    testEvaluateBasic()
    testEvaluateHornerCorrectness()
    testBufferAPIRoundTrip()
    testLargeAddSubConsistency()
    testMulVsSchoolbook()
}

// MARK: - Add tests

private func testAddBasic() {
    suite("GPUPolyArith add basic")
    do {
        let engine = try GPUPolyArithEngine()

        // (1 + 2x) + (3 + 4x) = (4 + 6x)
        let a = [frFromInt(1), frFromInt(2)]
        let b = [frFromInt(3), frFromInt(4)]
        let c = try engine.add(a, b)

        expect(c.count == 2, "result length")
        expect(frEq(c[0], frFromInt(4)), "c[0] = 4")
        expect(frEq(c[1], frFromInt(6)), "c[1] = 6")
    } catch {
        expect(false, "add error: \(error)")
    }
}

// MARK: - Sub tests

private func testSubBasic() {
    suite("GPUPolyArith sub basic")
    do {
        let engine = try GPUPolyArithEngine()

        let a = [frFromInt(10), frFromInt(20)]
        let b = [frFromInt(3), frFromInt(5)]
        let c = try engine.sub(a, b)

        expect(c.count == 2, "result length")
        expect(frEq(c[0], frFromInt(7)), "c[0] = 7")
        expect(frEq(c[1], frFromInt(15)), "c[1] = 15")
    } catch {
        expect(false, "sub error: \(error)")
    }
}

// MARK: - Scale tests

private func testScaleBasic() {
    suite("GPUPolyArith scale basic")
    do {
        let engine = try GPUPolyArithEngine()

        // (1 + 2x + 3x^2) * 5 = (5 + 10x + 15x^2)
        let a = [frFromInt(1), frFromInt(2), frFromInt(3)]
        let s = frFromInt(5)
        let c = try engine.scale(a, s)

        expect(c.count == 3, "result length")
        expect(frEq(c[0], frFromInt(5)), "c[0] = 5")
        expect(frEq(c[1], frFromInt(10)), "c[1] = 10")
        expect(frEq(c[2], frFromInt(15)), "c[2] = 15")
    } catch {
        expect(false, "scale error: \(error)")
    }
}

// MARK: - Mul tests

private func testMulBasic() {
    suite("GPUPolyArith mul basic")
    do {
        let engine = try GPUPolyArithEngine()

        // (1 + 2x) * (3 + 4x) = 3 + 10x + 8x^2
        let a = [frFromInt(1), frFromInt(2)]
        let b = [frFromInt(3), frFromInt(4)]
        let c = try engine.mul(a, b)

        expect(c.count == 3, "result length = 3")
        expect(frEq(c[0], frFromInt(3)), "c[0] = 3")
        expect(frEq(c[1], frFromInt(10)), "c[1] = 10")
        expect(frEq(c[2], frFromInt(8)), "c[2] = 8")
    } catch {
        expect(false, "mul error: \(error)")
    }
}

private func testMulLinearTimesLinear() {
    suite("GPUPolyArith mul (x+1)*(x-1)")
    do {
        let engine = try GPUPolyArithEngine()

        // (1 + x) * (-1 + x) = -1 + 0x + x^2
        let a = [frFromInt(1), Fr.one]
        let minusOne = frSub(Fr.zero, Fr.one)
        let b = [minusOne, Fr.one]
        let c = try engine.mul(a, b)

        expect(c.count == 3, "result length = 3")
        expect(frEq(c[0], minusOne), "c[0] = -1")
        expect(frEq(c[1], Fr.zero), "c[1] = 0")
        expect(frEq(c[2], Fr.one), "c[2] = 1")
    } catch {
        expect(false, "mul error: \(error)")
    }
}

// MARK: - Evaluate tests

private func testEvaluateBasic() {
    suite("GPUPolyArith evaluate basic")
    do {
        let engine = try GPUPolyArithEngine()

        // p(x) = 1 + 2x + 3x^2, p(5) = 1 + 10 + 75 = 86
        let coeffs = [frFromInt(1), frFromInt(2), frFromInt(3)]
        let result = try engine.evaluate(coeffs, at: frFromInt(5))

        expect(frEq(result, frFromInt(86)), "p(5) = 86")
    } catch {
        expect(false, "evaluate error: \(error)")
    }
}

private func testEvaluateHornerCorrectness() {
    suite("GPUPolyArith evaluate Horner random")
    do {
        let engine = try GPUPolyArithEngine()
        var rng = ArithTestRNG(state: 0xCAFE_BABE)

        // Generate a random polynomial of degree 127
        let n = 128
        var coeffs = [Fr]()
        for _ in 0..<n { coeffs.append(rng.nextFr()) }
        let point = rng.nextFr()

        // CPU reference Horner
        var expected = coeffs[n - 1]
        for i in stride(from: n - 2, through: 0, by: -1) {
            expected = frAdd(frMul(expected, point), coeffs[i])
        }

        let result = try engine.evaluate(coeffs, at: point)
        expect(frEq(result, expected), "GPU Horner matches CPU Horner (n=128)")
    } catch {
        expect(false, "evaluate error: \(error)")
    }
}

// MARK: - Buffer API roundtrip

private func testBufferAPIRoundTrip() {
    suite("GPUPolyArith buffer API roundtrip")
    do {
        let engine = try GPUPolyArithEngine()

        let data = [frFromInt(42), frFromInt(99), frFromInt(7)]
        let buf = engine.createBuffer(data)
        let back = engine.readBuffer(buf, count: 3)

        expect(frEq(back[0], data[0]) && frEq(back[1], data[1]) && frEq(back[2], data[2]),
               "roundtrip preserves data")

        // Test buffer-level add
        let a = engine.createBuffer([frFromInt(1), frFromInt(2)])
        let b = engine.createBuffer([frFromInt(3), frFromInt(4)])
        let c = try engine.add(a: a, b: b, n: 2)
        let result = engine.readBuffer(c, count: 2)
        expect(frEq(result[0], frFromInt(4)), "buffer add c[0] = 4")
        expect(frEq(result[1], frFromInt(6)), "buffer add c[1] = 6")
    } catch {
        expect(false, "buffer API error: \(error)")
    }
}

// MARK: - Large add/sub consistency: a + b - b == a

private func testLargeAddSubConsistency() {
    suite("GPUPolyArith add/sub consistency (n=4096)")
    do {
        let engine = try GPUPolyArithEngine()
        // Force GPU path
        engine.cpuThresholdEW = 0

        var rng = ArithTestRNG(state: 0xDEAD_BEEF)
        let n = 4096

        var a = [Fr](); var b = [Fr]()
        for _ in 0..<n { a.append(rng.nextFr()); b.append(rng.nextFr()) }

        let sum = try engine.add(a, b)
        let roundtrip = try engine.sub(sum, b)

        var ok = true
        for i in 0..<n {
            if !frEq(roundtrip[i], a[i]) { ok = false; break }
        }
        expect(ok, "a + b - b == a for all 4096 elements")
    } catch {
        expect(false, "large add/sub error: \(error)")
    }
}

// MARK: - Mul vs schoolbook reference

private func testMulVsSchoolbook() {
    suite("GPUPolyArith mul vs schoolbook (deg 63)")
    do {
        let engine = try GPUPolyArithEngine()
        // Force GPU path
        engine.cpuThresholdMul = 0

        var rng = ArithTestRNG(state: 0x1234_5678)
        let na = 32, nb = 32

        var a = [Fr](); var b = [Fr]()
        for _ in 0..<na { a.append(rng.nextFr()) }
        for _ in 0..<nb { b.append(rng.nextFr()) }

        // GPU multiplication
        let gpuResult = try engine.mul(a, b)

        // CPU schoolbook reference
        let resultLen = na + nb - 1
        var cpuResult = [Fr](repeating: Fr.zero, count: resultLen)
        for i in 0..<na {
            for j in 0..<nb {
                cpuResult[i + j] = frAdd(cpuResult[i + j], frMul(a[i], b[j]))
            }
        }

        expect(gpuResult.count == resultLen, "result length = \(resultLen)")
        var ok = true
        for i in 0..<resultLen {
            if !frEq(gpuResult[i], cpuResult[i]) { ok = false; break }
        }
        expect(ok, "GPU NTT mul matches schoolbook for all \(resultLen) coefficients")
    } catch {
        expect(false, "mul vs schoolbook error: \(error)")
    }
}
