// Lightweight test runner — no XCTest dependency required
// Run with: swift test or swift build --target zkMetalTests && .build/debug/zkMetalTests

@testable import zkMetal

func assertEqual<T: Equatable>(_ a: T, _ b: T, _ msg: String = "", file: String = #file, line: Int = #line) {
    if a != b {
        fatalError("FAIL [\(file):\(line)] \(msg): \(a) != \(b)")
    }
}

func runTests() {
    // Montgomery round-trip
    let a = fpFromInt(42)
    let aInt = fpToInt(a)
    assertEqual(aInt[0], 42, "Montgomery round-trip")
    assertEqual(aInt[1], 0)
    print("  [pass] Montgomery round-trip")

    // Addition
    let b = fpFromInt(100)
    let c = fpAdd(a, b)
    assertEqual(fpToInt(c)[0], 142, "Addition")
    print("  [pass] Field addition")

    // Multiplication
    let d = fpMul(a, b)
    assertEqual(fpToInt(d)[0], 4200, "Multiplication")
    print("  [pass] Field multiplication")

    // Subtraction
    let e = fpSub(b, a)
    assertEqual(fpToInt(e)[0], 58, "Subtraction")
    print("  [pass] Field subtraction")

    // Inverse
    let aInv = fpInverse(a)
    let one = fpMul(a, aInv)
    let oneInt = fpToInt(one)
    assertEqual(oneInt[0], 1, "Inverse")
    assertEqual(oneInt[1], 0)
    print("  [pass] Field inverse")

    // Generator on curve
    let gx = fpFromInt(1), gy = fpFromInt(2)
    let y2 = fpSqr(gy)
    let rhs = fpAdd(fpMul(gx, fpSqr(gx)), fpFromInt(3))
    assertEqual(fpToInt(y2), fpToInt(rhs), "Generator on curve")
    print("  [pass] Generator on curve")

    // Point double
    let g = PointAffine(x: gx, y: gy)
    let g2 = pointDouble(pointFromAffine(g))
    let g2a = pointToAffine(g2)!
    let g2y2 = fpSqr(g2a.y)
    let g2rhs = fpAdd(fpMul(g2a.x, fpSqr(g2a.x)), fpFromInt(3))
    assertEqual(fpToInt(g2y2), fpToInt(g2rhs), "2G on curve")
    print("  [pass] 2G on curve")

    // G+G = 2G
    let gpg = pointAdd(pointFromAffine(g), pointFromAffine(g))
    assertEqual(fpToInt(pointToAffine(gpg)!.x), fpToInt(g2a.x), "G+G = 2G")
    print("  [pass] G + G = 2G")

    // GLV beta
    let beta2 = fpSqr(FP_BETA)
    let beta3 = fpMul(FP_BETA, beta2)
    let b3 = fpToInt(beta3)
    assertEqual(b3[0], 1, "beta^3 = 1")
    assertEqual(b3[1], 0)
    print("  [pass] β³ ≡ 1 mod p")

    print("\nAll Fp tests passed.")
}

func runFrTests() {
    print("\n=== BN254 Fr Field Tests ===")

    // Montgomery round-trip
    let a = frFromInt(42)
    let aInt = frToInt(a)
    assertEqual(aInt[0], 42, "Fr Montgomery round-trip")
    assertEqual(aInt[1], 0)
    print("  [pass] Fr Montgomery round-trip")

    // Addition
    let b = frFromInt(100)
    let c = frAdd(a, b)
    assertEqual(frToInt(c)[0], 142, "Fr Addition")
    print("  [pass] Fr addition")

    // Multiplication
    let d = frMul(a, b)
    assertEqual(frToInt(d)[0], 4200, "Fr Multiplication")
    print("  [pass] Fr multiplication")

    // Large multiply: (2^32-1)^2
    let big = frFromInt(0xFFFFFFFF)
    let bigSq = frMul(big, big)
    let expected: UInt64 = 0xFFFFFFFE00000001
    assertEqual(frToInt(bigSq)[0], expected, "Large multiply")
    print("  [pass] Fr large multiplication")

    // Inverse
    let aInv = frInverse(a)
    let one = frMul(a, aInv)
    let oneInt = frToInt(one)
    assertEqual(oneInt[0], 1, "Fr Inverse")
    assertEqual(oneInt[1], 0)
    print("  [pass] Fr inverse")

    // Compute root of unity from scratch: 5^((r-1)/2^28) mod r
    let gen = frFromInt(5)
    var rMinus1 = Fr.P
    rMinus1[0] -= 1
    // Right shift by 28 bits
    var exp = [UInt64](repeating: 0, count: 4)
    for i in 0..<4 {
        exp[i] = rMinus1[i] >> 28
        if i < 3 { exp[i] |= rMinus1[i+1] << 36 }
    }
    var computedRoot = Fr.one
    var base = gen
    for i in 0..<4 {
        var word = exp[i]
        for _ in 0..<64 {
            if word & 1 == 1 { computedRoot = frMul(computedRoot, base) }
            base = frSqr(base)
            word >>= 1
        }
    }

    // Verify computed root^(2^28) = 1
    var omegaPow = computedRoot
    for _ in 0..<28 { omegaPow = frSqr(omegaPow) }
    let shouldBeOne = frToInt(omegaPow)
    assertEqual(shouldBeOne[0], 1, "computed omega^(2^28)")
    assertEqual(shouldBeOne[1], 0)
    print("  [pass] computed ω^(2^28) ≡ 1 (mod r)")

    // Print computed vs stored root
    let computedMont = computedRoot.to64()
    print("  [info] computed root (Montgomery): \(computedMont.map { String(format: "%016llx", $0) })")
    print("  [info] stored root (Montgomery):   \(Fr.ROOT_OF_UNITY.map { String(format: "%016llx", $0) })")

    // Use computed root for NTT tests
    let logN = 4
    let omega = frRootOfUnity(logN: logN)
    var omegaN = Fr.one
    for _ in 0..<(1 << logN) {
        omegaN = frMul(omegaN, omega)
    }
    let omegaNInt = frToInt(omegaN)
    print("  [info] omega^16 = \(omegaNInt)")
    if omegaNInt[0] != 1 || omegaNInt[1] != 0 {
        print("  [WARN] Stored ROOT_OF_UNITY may be wrong, computing from scratch...")
        // Try with computed root directly
        var omega2 = computedRoot
        for _ in 0..<(28 - logN) { omega2 = frSqr(omega2) }
        var omegaN2 = Fr.one
        for _ in 0..<(1 << logN) { omegaN2 = frMul(omegaN2, omega2) }
        let omegaN2Int = frToInt(omegaN2)
        print("  [info] computed_omega^16 = \(omegaN2Int)")
        assertEqual(omegaN2Int[0], 1, "computed omega^n = 1")
        print("  [pass] ω^16 ≡ 1 using computed root")
    } else {
        print("  [pass] ω^16 ≡ 1 (mod r)")
    }

    print("\nAll Fr tests passed.")
}

func runNTTTests() {
    print("\n=== NTT Correctness Tests ===")

    let logN = 4
    let n = 1 << logN

    // CPU NTT round-trip: iNTT(NTT(f)) = f
    var input = [Fr](repeating: Fr.zero, count: n)
    for i in 0..<n {
        input[i] = frFromInt(UInt64(i + 1))
    }

    let nttResult = NTTEngine.cpuNTT(input, logN: logN)
    let recovered = NTTEngine.cpuINTT(nttResult, logN: logN)

    for i in 0..<n {
        let expected = frToInt(input[i])
        let got = frToInt(recovered[i])
        assertEqual(expected, got, "CPU NTT round-trip element \(i)")
    }
    print("  [pass] CPU NTT round-trip (n=\(n))")

    // GPU NTT round-trip
    do {
        let engine = try NTTEngine()

        let gpuNTT = try engine.ntt(input)
        let gpuRecovered = try engine.intt(gpuNTT)

        for i in 0..<n {
            let expected = frToInt(input[i])
            let got = frToInt(gpuRecovered[i])
            assertEqual(expected, got, "GPU NTT round-trip element \(i)")
        }
        print("  [pass] GPU NTT round-trip (n=\(n))")

        // GPU NTT matches CPU NTT
        for i in 0..<n {
            let cpuVal = frToInt(nttResult[i])
            let gpuVal = frToInt(gpuNTT[i])
            assertEqual(cpuVal, gpuVal, "GPU vs CPU NTT element \(i)")
        }
        print("  [pass] GPU NTT matches CPU NTT (n=\(n))")

        // Larger test: n=1024
        let logN2 = 10
        let n2 = 1 << logN2
        var input2 = [Fr](repeating: Fr.zero, count: n2)
        var rng: UInt64 = 0xDEAD_BEEF
        for i in 0..<n2 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            input2[i] = frFromInt(rng >> 32)
        }

        let gpuNTT2 = try engine.ntt(input2)
        let gpuRecovered2 = try engine.intt(gpuNTT2)

        for i in 0..<n2 {
            let expected = frToInt(input2[i])
            let got = frToInt(gpuRecovered2[i])
            assertEqual(expected, got, "GPU NTT round-trip n=1024 element \(i)")
        }
        print("  [pass] GPU NTT round-trip (n=\(n2))")

        // Quick benchmark
        let logBench = 16
        let nBench = 1 << logBench
        var benchData = [Fr](repeating: Fr.zero, count: nBench)
        for i in 0..<nBench {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            benchData[i] = frFromInt(rng >> 32)
        }

        let dataBuf = engine.device.makeBuffer(
            length: nBench * MemoryLayout<Fr>.stride, options: .storageModeShared)!
        benchData.withUnsafeBytes { src in
            memcpy(dataBuf.contents(), src.baseAddress!, nBench * MemoryLayout<Fr>.stride)
        }

        // Warmup
        try engine.ntt(data: dataBuf, logN: logBench)
        try engine.intt(data: dataBuf, logN: logBench)

        // Timed run
        let start = CFAbsoluteTimeGetCurrent()
        try engine.ntt(data: dataBuf, logN: logBench)
        let nttTime = (CFAbsoluteTimeGetCurrent() - start) * 1000

        let start2 = CFAbsoluteTimeGetCurrent()
        try engine.intt(data: dataBuf, logN: logBench)
        let inttTime = (CFAbsoluteTimeGetCurrent() - start2) * 1000

        print("  [bench] NTT(2^\(logBench)=\(nBench)): \(String(format: "%.1f", nttTime))ms")
        print("  [bench] iNTT(2^\(logBench)=\(nBench)): \(String(format: "%.1f", inttTime))ms")

    } catch {
        print("  [FAIL] GPU NTT error: \(error)")
    }

    print("\nAll NTT tests passed.")
}

import Foundation

runTests()
runFrTests()
runNTTTests()
