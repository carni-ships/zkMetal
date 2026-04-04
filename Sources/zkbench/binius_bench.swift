// Binius benchmark: binary tower field arithmetic + additive FFT
// Validates that XOR-based binary field arithmetic on Metal GPU
// gives a massive advantage over M31/BabyBear for addition-heavy workloads.

import Foundation
import zkMetal

public func runBiniusBench() {
    print("=== Binius Binary Tower Benchmark ===")

    // ---- Correctness Tests ----
    print("\n--- GF(2^8) Correctness ---")

    let a8 = BinaryField8(value: 0x53)
    let b8 = BinaryField8(value: 0xCA)

    // Addition = XOR
    let sum8 = a8 + b8
    if sum8.value == (0x53 ^ 0xCA) {
        print("  [pass] GF(2^8) add: 0x53 + 0xCA = 0x\(String(sum8.value, radix: 16))")
    } else {
        print("  [FAIL] GF(2^8) add: expected 0x\(String(0x53 ^ 0xCA, radix: 16)), got 0x\(String(sum8.value, radix: 16))")
        return
    }

    // Multiplication
    let prod8 = a8 * b8
    print("  [info] GF(2^8) mul: 0x53 * 0xCA = 0x\(String(prod8.value, radix: 16))")

    // Verify: a * 1 = a
    let one8 = BinaryField8.one
    let aMulOne = a8 * one8
    if aMulOne == a8 {
        print("  [pass] GF(2^8) mul identity: a * 1 = a")
    } else {
        print("  [FAIL] GF(2^8) mul identity"); return
    }

    // Verify: a * 0 = 0
    let aMulZero = a8 * BinaryField8.zero
    if aMulZero.isZero {
        print("  [pass] GF(2^8) mul zero: a * 0 = 0")
    } else {
        print("  [FAIL] GF(2^8) mul zero"); return
    }

    // Inverse
    let a8Inv = a8.inverse()
    let a8Check = a8 * a8Inv
    if a8Check == one8 {
        print("  [pass] GF(2^8) inverse: a * a^(-1) = 1")
    } else {
        print("  [FAIL] GF(2^8) inverse: got \(a8Check.value)"); return
    }

    // Verify all nonzero elements have inverses
    var invOK = true
    for i in 1..<256 {
        let x = BinaryField8(value: UInt8(i))
        let xi = x.inverse()
        if (x * xi) != one8 { invOK = false; break }
    }
    if invOK { print("  [pass] GF(2^8) all 255 inverses verified") }
    else { print("  [FAIL] GF(2^8) inverse check"); return }

    // ---- GF(2^16) ----
    print("\n--- GF(2^16) Correctness ---")

    let a16 = BinaryField16(value: 0xABCD)
    let b16 = BinaryField16(value: 0x1234)
    let sum16 = a16 + b16
    if sum16.toUInt16 == (0xABCD ^ 0x1234) {
        print("  [pass] GF(2^16) add: XOR correct")
    } else {
        print("  [FAIL] GF(2^16) add"); return
    }

    let prod16 = a16 * b16
    print("  [info] GF(2^16) mul: 0xABCD * 0x1234 = 0x\(String(prod16.toUInt16, radix: 16))")

    // Identity
    if (a16 * BinaryField16.one) == a16 {
        print("  [pass] GF(2^16) mul identity")
    } else {
        print("  [FAIL] GF(2^16) mul identity"); return
    }

    // Inverse
    let a16Inv = a16.inverse()
    if (a16 * a16Inv) == BinaryField16.one {
        print("  [pass] GF(2^16) inverse: a * a^(-1) = 1")
    } else {
        print("  [FAIL] GF(2^16) inverse"); return
    }

    // ---- GF(2^32) ----
    print("\n--- GF(2^32) Correctness ---")

    let a32 = BinaryField32(value: 0xDEADBEEF)
    let b32 = BinaryField32(value: 0xCAFEBABE)
    let sum32 = a32 + b32
    if sum32.toUInt32 == (0xDEADBEEF ^ 0xCAFEBABE) {
        print("  [pass] GF(2^32) add: XOR correct")
    } else {
        print("  [FAIL] GF(2^32) add"); return
    }

    if (a32 * BinaryField32.one) == a32 {
        print("  [pass] GF(2^32) mul identity")
    } else {
        print("  [FAIL] GF(2^32) mul identity"); return
    }

    let a32Inv = a32.inverse()
    if (a32 * a32Inv) == BinaryField32.one {
        print("  [pass] GF(2^32) inverse: a * a^(-1) = 1")
    } else {
        print("  [FAIL] GF(2^32) inverse"); return
    }

    // Commutativity
    if (a32 * b32) == (b32 * a32) {
        print("  [pass] GF(2^32) commutativity")
    } else {
        print("  [FAIL] GF(2^32) commutativity"); return
    }

    // Associativity
    let c32 = BinaryField32(value: 0x12345678)
    if ((a32 * b32) * c32) == (a32 * (b32 * c32)) {
        print("  [pass] GF(2^32) associativity")
    } else {
        print("  [FAIL] GF(2^32) associativity"); return
    }

    // Distributivity
    if (a32 * (b32 + c32)) == ((a32 * b32) + (a32 * c32)) {
        print("  [pass] GF(2^32) distributivity")
    } else {
        print("  [FAIL] GF(2^32) distributivity"); return
    }

    // ---- GF(2^64), GF(2^128) ----
    print("\n--- GF(2^64) / GF(2^128) Correctness ---")

    let a64val: UInt64 = 0xDEADBEEFCAFEBABE
    let b64val: UInt64 = 0x1234567890ABCDEF
    let a64 = BinaryField64(value: a64val)
    let b64 = BinaryField64(value: b64val)
    if (a64 + b64).toUInt64 == (a64val ^ b64val) {
        print("  [pass] GF(2^64) add: XOR correct")
    } else {
        print("  [FAIL] GF(2^64) add"); return
    }
    if (a64 * BinaryField64.one) == a64 {
        print("  [pass] GF(2^64) mul identity")
    } else {
        print("  [FAIL] GF(2^64) mul identity"); return
    }
    let a64Inv = a64.inverse()
    if (a64 * a64Inv) == BinaryField64.one {
        print("  [pass] GF(2^64) inverse")
    } else {
        print("  [FAIL] GF(2^64) inverse"); return
    }

    let a128 = BinaryField128(lo: a64, hi: b64)
    if (a128 * BinaryField128.one) == a128 {
        print("  [pass] GF(2^128) mul identity")
    } else {
        print("  [FAIL] GF(2^128) mul identity"); return
    }
    let a128Inv = a128.inverse()
    if (a128 * a128Inv) == BinaryField128.one {
        print("  [pass] GF(2^128) inverse")
    } else {
        print("  [FAIL] GF(2^128) inverse"); return
    }

    // ---- Additive FFT roundtrip ----
    print("\n--- Additive FFT (Binary) ---")

    for logN in [4, 8, 10, 12] {
        if BinaryFFT.verifyRoundtrip(logN: logN) {
            print("  [pass] FFT roundtrip 2^\(logN) = \(1 << logN)")
        } else {
            print("  [FAIL] FFT roundtrip 2^\(logN)"); return
        }
    }

    // ---- CPU Throughput Benchmarks ----
    print("\n--- CPU Field Throughput ---")

    let iters = 1_000_000

    // GF(2^8) multiply throughput
    do {
        var x = BinaryField8(value: 0x53)
        let y = BinaryField8(value: 0xCA)
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iters {
            x = x * y
        }
        let dt = CFAbsoluteTimeGetCurrent() - t0
        _ = x  // prevent optimization
        let mops = Double(iters) / dt / 1e6
        print(String(format: "  GF(2^8)   mul: %.1f Mop/s (%.1f ns/op)", mops, dt / Double(iters) * 1e9))
    }

    // GF(2^16) multiply throughput
    do {
        var x = BinaryField16(value: 0xABCD)
        let y = BinaryField16(value: 0x1234)
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iters {
            x = x * y
        }
        let dt = CFAbsoluteTimeGetCurrent() - t0
        _ = x
        let mops = Double(iters) / dt / 1e6
        print(String(format: "  GF(2^16)  mul: %.1f Mop/s (%.1f ns/op)", mops, dt / Double(iters) * 1e9))
    }

    // GF(2^32) multiply throughput
    do {
        var x = BinaryField32(value: 0xDEADBEEF)
        let y = BinaryField32(value: 0xCAFEBABE)
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iters {
            x = x * y
        }
        let dt = CFAbsoluteTimeGetCurrent() - t0
        _ = x
        let mops = Double(iters) / dt / 1e6
        print(String(format: "  GF(2^32)  mul: %.1f Mop/s (%.1f ns/op)", mops, dt / Double(iters) * 1e9))
    }

    // GF(2^64) multiply throughput
    do {
        var x = BinaryField64(value: 0xDEADBEEFCAFEBABE)
        let y = BinaryField64(value: 0x1234567890ABCDEF)
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iters {
            x = x * y
        }
        let dt = CFAbsoluteTimeGetCurrent() - t0
        _ = x
        let mops = Double(iters) / dt / 1e6
        print(String(format: "  GF(2^64)  mul: %.1f Mop/s (%.1f ns/op)", mops, dt / Double(iters) * 1e9))
    }

    // GF(2^128) multiply throughput
    do {
        var x = BinaryField128(lo: BinaryField64(value: 0xDEADBEEFCAFEBABE),
                                hi: BinaryField64(value: 0x1234567890ABCDEF))
        let y = BinaryField128(lo: BinaryField64(value: 0xFEDCBA9876543210),
                                hi: BinaryField64(value: 0x0123456789ABCDEF))
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iters {
            x = x * y
        }
        let dt = CFAbsoluteTimeGetCurrent() - t0
        _ = x
        let mops = Double(iters) / dt / 1e6
        print(String(format: "  GF(2^128) mul: %.1f Mop/s (%.1f ns/op)", mops, dt / Double(iters) * 1e9))
    }

    // M31 multiply for comparison
    do {
        var x = M31(v: 42)
        let y = M31(v: 100)
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iters {
            x = m31Mul(x, y)
        }
        let dt = CFAbsoluteTimeGetCurrent() - t0
        _ = x
        let mops = Double(iters) / dt / 1e6
        print(String(format: "  M31       mul: %.1f Mop/s (%.1f ns/op) [comparison]", mops, dt / Double(iters) * 1e9))
    }

    // BabyBear multiply for comparison
    do {
        var x = Bb(v: 42)
        let y = Bb(v: 100)
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iters {
            x = bbMul(x, y)
        }
        let dt = CFAbsoluteTimeGetCurrent() - t0
        _ = x
        let mops = Double(iters) / dt / 1e6
        print(String(format: "  BabyBear  mul: %.1f Mop/s (%.1f ns/op) [comparison]", mops, dt / Double(iters) * 1e9))
    }

    // Addition comparison: binary XOR vs M31 vs BabyBear
    print("\n--- CPU Addition Throughput (XOR vs modular) ---")
    let addIters = 10_000_000

    do {
        var x = BinaryField32(value: 0xDEADBEEF)
        let y = BinaryField32(value: 0xCAFEBABE)
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<addIters {
            x = x + y
        }
        let dt = CFAbsoluteTimeGetCurrent() - t0
        _ = x
        let mops = Double(addIters) / dt / 1e6
        print(String(format: "  GF(2^32) add (XOR):  %.0f Mop/s (%.2f ns/op)", mops, dt / Double(addIters) * 1e9))
    }

    do {
        var x = M31(v: 42)
        let y = M31(v: 100)
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<addIters {
            x = m31Add(x, y)
        }
        let dt = CFAbsoluteTimeGetCurrent() - t0
        _ = x
        let mops = Double(addIters) / dt / 1e6
        print(String(format: "  M31 add (mod):       %.0f Mop/s (%.2f ns/op)", mops, dt / Double(addIters) * 1e9))
    }

    do {
        var x = Bb(v: 42)
        let y = Bb(v: 100)
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<addIters {
            x = bbAdd(x, y)
        }
        let dt = CFAbsoluteTimeGetCurrent() - t0
        _ = x
        let mops = Double(addIters) / dt / 1e6
        print(String(format: "  BabyBear add (mod):  %.0f Mop/s (%.2f ns/op)", mops, dt / Double(addIters) * 1e9))
    }

    // ---- CPU Additive FFT Benchmark ----
    print("\n--- CPU Additive FFT (Binary) ---")

    for logN in [10, 12, 14, 16] {
        let n = 1 << logN
        var rng: UInt64 = 0xDEADBEEF
        var data = [BinaryField32]()
        data.reserveCapacity(n)
        for _ in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1
            data.append(BinaryField32(value: UInt32(truncatingIfNeeded: rng)))
        }

        // Warmup
        var warm = data
        BinaryFFT.forward(data: &warm, logN: logN)

        let runs = logN >= 16 ? 3 : 5
        var times = [Double]()
        for _ in 0..<runs {
            var d = data
            let t0 = CFAbsoluteTimeGetCurrent()
            BinaryFFT.forward(data: &d, logN: logN)
            let dt = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            times.append(dt)
        }
        times.sort()
        let median = times[runs / 2]
        print(String(format: "  Binary FFT 2^%-2d = %7d: %.2f ms", logN, n, median))
    }

    // ---- GPU Benchmark ----
    print("\n--- GPU Batch Operations (GF(2^32)) ---")

    do {
        let engine = try BinaryFFTEngine()
        print("  GPU: \(engine.device.name)")

        for logN in [14, 16, 18, 20] {
            let n = 1 << logN
            var rng: UInt64 = 0xCAFEBABE
            var aArr = [UInt32]()
            var bArr = [UInt32]()
            aArr.reserveCapacity(n)
            bArr.reserveCapacity(n)
            for _ in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1
                aArr.append(UInt32(truncatingIfNeeded: rng))
                rng = rng &* 6364136223846793005 &+ 1
                bArr.append(UInt32(truncatingIfNeeded: rng))
            }

            // Warmup
            let _ = try engine.batchAdd(a: aArr, b: bArr)

            // Batch add (XOR)
            var addTimes = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.batchAdd(a: aArr, b: bArr)
                let dt = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                addTimes.append(dt)
            }
            addTimes.sort()

            // Batch mul
            let _ = try engine.batchMul(a: aArr, b: bArr)
            var mulTimes = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.batchMul(a: aArr, b: bArr)
                let dt = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                mulTimes.append(dt)
            }
            mulTimes.sort()

            let addMedian = addTimes[2]
            let mulMedian = mulTimes[2]
            let addGops = Double(n) / addMedian / 1e6
            let mulGops = Double(n) / mulMedian / 1e6
            print(String(format: "  2^%-2d = %7d: add %.2f ms (%.1f Gop/s) | mul %.2f ms (%.1f Gop/s)",
                        logN, n, addMedian, addGops, mulMedian, mulGops))
        }

        // GPU batch correctness check
        print("\n  GPU correctness:")
        let testN = 1024
        var testA = [UInt32]()
        var testB = [UInt32]()
        var rng2: UInt64 = 0x1234
        for _ in 0..<testN {
            rng2 = rng2 &* 6364136223846793005 &+ 1
            testA.append(UInt32(truncatingIfNeeded: rng2))
            rng2 = rng2 &* 6364136223846793005 &+ 1
            testB.append(UInt32(truncatingIfNeeded: rng2))
        }

        // Verify GPU add = XOR
        let gpuAdd = try engine.batchAdd(a: testA, b: testB)
        var addOK = true
        for i in 0..<testN {
            if gpuAdd[i] != (testA[i] ^ testB[i]) { addOK = false; break }
        }
        print("    [" + (addOK ? "pass" : "FAIL") + "] GPU batch_add matches XOR")

        // Verify GPU mul matches CPU
        let gpuMul = try engine.batchMul(a: testA, b: testB)
        var mulOK = true
        for i in 0..<testN {
            let cpuA = BinaryField32(value: testA[i])
            let cpuB = BinaryField32(value: testB[i])
            let cpuResult = (cpuA * cpuB).toUInt32
            if gpuMul[i] != cpuResult { mulOK = false; break }
        }
        print("    [" + (mulOK ? "pass" : "FAIL") + "] GPU batch_mul matches CPU")

    } catch {
        print("  GPU error: \(error)")
    }
}
