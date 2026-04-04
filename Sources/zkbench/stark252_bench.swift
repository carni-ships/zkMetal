// Stark252 Field + NTT Benchmark
import zkMetal
import Foundation

public func runStark252Bench() {
    print("\n=== Stark252 (StarkNet) Field + NTT Benchmark ===")
    print("Stark252 stride: \(MemoryLayout<Stark252>.stride) bytes")

    // --- Field correctness tests ---
    print("\n--- Stark252 Field Correctness ---")

    let a = stark252FromInt(42)
    let b = stark252FromInt(100)

    // add
    let c = stark252Add(a, b)
    let cInt = stark252ToInt(c)
    print("  42 + 100 = \(cInt[0]) \(cInt[0] == 142 ? "[pass]" : "[FAIL]")")

    // mul
    let d = stark252Mul(a, b)
    let dInt = stark252ToInt(d)
    print("  42 * 100 = \(dInt[0]) \(dInt[0] == 4200 ? "[pass]" : "[FAIL]")")

    // sub
    let e = stark252Sub(b, a)
    let eInt = stark252ToInt(e)
    print("  100 - 42 = \(eInt[0]) \(eInt[0] == 58 ? "[pass]" : "[FAIL]")")

    // inv
    let ainv = stark252Inverse(a)
    let check = stark252Mul(a, ainv)
    let checkInt = stark252ToInt(check)
    print("  42 * 42^(-1) = \(checkInt[0]) \(checkInt[0] == 1 && checkInt[1] == 0 && checkInt[2] == 0 && checkInt[3] == 0 ? "[pass]" : "[FAIL]")")

    // neg
    let neg_a = stark252Neg(a)
    let sum = stark252Add(a, neg_a)
    print("  a + (-a) = 0: \(sum.isZero ? "[pass]" : "[FAIL]")")

    // commutativity: a*b = b*a
    let ab = stark252Mul(a, b)
    let ba = stark252Mul(b, a)
    let comm = stark252ToInt(ab) == stark252ToInt(ba)
    print("  a*b = b*a: \(comm ? "[pass]" : "[FAIL]")")

    // distributivity: (a+b)*c = a*c + b*c
    let cc = stark252FromInt(7)
    let lhs = stark252Mul(stark252Add(a, b), cc)
    let rhs = stark252Add(stark252Mul(a, cc), stark252Mul(b, cc))
    let dist = stark252ToInt(lhs) == stark252ToInt(rhs)
    print("  (a+b)*c = a*c + b*c: \(dist ? "[pass]" : "[FAIL]")")

    // Montgomery round-trip
    let val: UInt64 = 123456789
    let mont = stark252FromInt(val)
    let back = stark252ToInt(mont)
    print("  Montgomery round-trip \(val): \(back[0] == val ? "[pass]" : "[FAIL]")")

    // pow
    let base = stark252FromInt(3)
    let p10 = stark252Pow(base, 10)
    let p10Int = stark252ToInt(p10)
    print("  3^10 = \(p10Int[0]) \(p10Int[0] == 59049 ? "[pass]" : "[FAIL]")")

    // root of unity check
    let omega16 = stark252RootOfUnity(logN: 16)
    let omega16_pow = stark252Pow(omega16, UInt64(1 << 16))
    let omega16Int = stark252ToInt(omega16_pow)
    print("  omega_16^(2^16) = 1: \(omega16Int[0] == 1 && omega16Int[1] == 0 && omega16Int[2] == 0 && omega16Int[3] == 0 ? "[pass]" : "[FAIL]")")

    // --- Field performance ---
    print("\n--- Stark252 Field Performance ---")
    let iters = 100_000
    var acc = stark252FromInt(1)
    let mul_operand = stark252FromInt(7)

    let t0 = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iters {
        acc = stark252Mul(acc, mul_operand)
    }
    let mulTime = (CFAbsoluteTimeGetCurrent() - t0) * 1e9 / Double(iters)
    print(String(format: "  Mul: %.1f ns/op", mulTime))

    acc = stark252FromInt(1)
    let t1 = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iters {
        acc = stark252Add(acc, mul_operand)
    }
    let addTime = (CFAbsoluteTimeGetCurrent() - t1) * 1e9 / Double(iters)
    print(String(format: "  Add: %.1f ns/op", addTime))

    let t2 = CFAbsoluteTimeGetCurrent()
    for _ in 0..<100 {
        let _ = stark252Inverse(mul_operand)
    }
    let invTime = (CFAbsoluteTimeGetCurrent() - t2) * 1e6 / 100.0
    print(String(format: "  Inv: %.1f us/op", invTime))

    // --- NTT ---
    do {
        let engine = try Stark252NTTEngine()

        // Correctness: round-trip at 2^10
        let testN = 1024
        var testInput = [Stark252](repeating: Stark252.zero, count: testN)
        for i in 0..<testN {
            testInput[i] = stark252FromInt(UInt64(i + 1))
        }
        let gpuNTT = try engine.ntt(testInput)
        let gpuRecovered = try engine.intt(gpuNTT)
        var correct = true
        for i in 0..<testN {
            let expected = stark252ToInt(testInput[i])
            let got = stark252ToInt(gpuRecovered[i])
            if expected != got {
                print("  MISMATCH at \(i): expected \(expected), got \(got)")
                correct = false
                break
            }
        }
        print("\n  NTT Round-trip (2^10): \(correct ? "PASS" : "FAIL")")

        // Round-trip at 2^16
        let testN2 = 1 << 16
        var testInput2 = [Stark252](repeating: Stark252.zero, count: testN2)
        var rng: UInt64 = 0xCAFE_BABE
        for i in 0..<testN2 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            testInput2[i] = stark252FromInt(rng >> 32)
        }
        let gpuNTT2 = try engine.ntt(testInput2)
        let gpuRecovered2 = try engine.intt(gpuNTT2)
        var correct2 = true
        var mismatches2 = 0
        for i in 0..<testN2 {
            let expected = stark252ToInt(testInput2[i])
            let got = stark252ToInt(gpuRecovered2[i])
            if expected != got {
                if mismatches2 < 3 {
                    print("  RT MISMATCH at \(i): expected \(expected[0]), got \(got[0])")
                }
                correct2 = false
                mismatches2 += 1
            }
        }
        print("  NTT Round-trip (2^16): \(correct2 ? "PASS" : "FAIL") (\(mismatches2) mismatches)")

        // Round-trip at 2^20 (four-step path)
        let testN3 = 1 << 20
        var testInput3 = [Stark252](repeating: Stark252.zero, count: testN3)
        rng = 0xFACE_CAFE
        for i in 0..<testN3 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            testInput3[i] = stark252FromInt(rng >> 32)
        }
        let gpuNTT3 = try engine.ntt(testInput3)
        let gpuRecovered3 = try engine.intt(gpuNTT3)
        var correct3 = true
        var mismatches3 = 0
        for i in 0..<testN3 {
            let expected = stark252ToInt(testInput3[i])
            let got = stark252ToInt(gpuRecovered3[i])
            if expected != got {
                if mismatches3 < 3 {
                    print("  RT MISMATCH at \(i): expected \(expected[0]), got \(got[0])")
                }
                correct3 = false
                mismatches3 += 1
            }
        }
        print("  NTT Round-trip (2^20, four-step): \(correct3 ? "PASS" : "FAIL") (\(mismatches3) mismatches)")

        // Performance benchmark
        print("\n--- Stark252 NTT Performance ---")
        let sizes = [10, 12, 14, 16, 18, 20, 22]

        for logN in sizes {
            let n = 1 << logN
            var data = [Stark252](repeating: Stark252.zero, count: n)
            rng = 0xDEAD_BEEF
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                data[i] = stark252FromInt(rng >> 32)
            }

            let dataBuf = engine.device.makeBuffer(
                length: n * MemoryLayout<Stark252>.stride, options: .storageModeShared)!
            data.withUnsafeBytes { src in
                memcpy(dataBuf.contents(), src.baseAddress!, n * MemoryLayout<Stark252>.stride)
            }

            // Warmup
            for _ in 0..<3 {
                try engine.ntt(data: dataBuf, logN: logN)
                try engine.intt(data: dataBuf, logN: logN)
            }

            // Reload fresh data
            data.withUnsafeBytes { src in
                memcpy(dataBuf.contents(), src.baseAddress!, n * MemoryLayout<Stark252>.stride)
            }

            var nttTimes = [Double]()
            for _ in 0..<10 {
                data.withUnsafeBytes { src in
                    memcpy(dataBuf.contents(), src.baseAddress!, n * MemoryLayout<Stark252>.stride)
                }
                let t0 = CFAbsoluteTimeGetCurrent()
                try engine.ntt(data: dataBuf, logN: logN)
                let t1 = CFAbsoluteTimeGetCurrent()
                nttTimes.append((t1 - t0) * 1000)
            }

            nttTimes.sort()
            let nttMedian = nttTimes[5]
            let elemPerSec = Double(n) / (nttMedian / 1000)
            print(String(format: "  2^%-2d = %7d | GPU: %7.2fms | %.1fM elem/s",
                        logN, n, nttMedian, elemPerSec / 1e6))
        }

    } catch {
        print("  [FAIL] Stark252 NTT error: \(error)")
    }
}
