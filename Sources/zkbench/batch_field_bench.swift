// Batch field operations benchmark for BN254 Fr
import zkMetal
import Foundation
import NeonFieldOps

// Helper: compare two Fr by limbs
private func frEqual(_ a: Fr, _ b: Fr) -> Bool {
    let al = a.to64(), bl = b.to64()
    return al[0] == bl[0] && al[1] == bl[1] && al[2] == bl[2] && al[3] == bl[3]
}

// Helper: call batch_add on flat arrays, returning result array
private func callBatchAdd(_ a: [UInt64], _ b: [UInt64], n: Int) -> [UInt64] {
    var result = [UInt64](repeating: 0, count: n * 4)
    a.withUnsafeBufferPointer { aPtr in
        b.withUnsafeBufferPointer { bPtr in
            result.withUnsafeMutableBufferPointer { rPtr in
                bn254_fr_batch_add_neon(rPtr.baseAddress!, aPtr.baseAddress!, bPtr.baseAddress!, Int32(n))
            }
        }
    }
    return result
}

private func callBatchSub(_ a: [UInt64], _ b: [UInt64], n: Int) -> [UInt64] {
    var result = [UInt64](repeating: 0, count: n * 4)
    a.withUnsafeBufferPointer { aPtr in
        b.withUnsafeBufferPointer { bPtr in
            result.withUnsafeMutableBufferPointer { rPtr in
                bn254_fr_batch_sub_neon(rPtr.baseAddress!, aPtr.baseAddress!, bPtr.baseAddress!, Int32(n))
            }
        }
    }
    return result
}

private func callBatchNeg(_ a: [UInt64], n: Int) -> [UInt64] {
    var result = [UInt64](repeating: 0, count: n * 4)
    a.withUnsafeBufferPointer { aPtr in
        result.withUnsafeMutableBufferPointer { rPtr in
            bn254_fr_batch_neg_neon(rPtr.baseAddress!, aPtr.baseAddress!, Int32(n))
        }
    }
    return result
}

private func callBatchMulScalar(_ a: [UInt64], _ scalar: [UInt64], n: Int) -> [UInt64] {
    var result = [UInt64](repeating: 0, count: n * 4)
    a.withUnsafeBufferPointer { aPtr in
        scalar.withUnsafeBufferPointer { sPtr in
            result.withUnsafeMutableBufferPointer { rPtr in
                bn254_fr_batch_mul_scalar_neon(rPtr.baseAddress!, aPtr.baseAddress!, sPtr.baseAddress!, Int32(n))
            }
        }
    }
    return result
}

public func runBatchFieldBench() {
    print("=== Batch Field Benchmark (BN254 Fr) ===")

    // --- Correctness tests ---
    print("\n--- Correctness ---")

    let a1 = frFromInt(42)
    let b1 = frFromInt(100)
    let a2 = frFromInt(999)
    let b2 = frFromInt(7)

    // Test add
    let swiftAdd1 = frAdd(a1, b1)
    let swiftAdd2 = frAdd(a2, b2)

    let addResult = callBatchAdd(a1.to64() + a2.to64(), b1.to64() + b2.to64(), n: 2)
    let cAdd1 = Fr.from64(Array(addResult[0..<4]))
    let cAdd2 = Fr.from64(Array(addResult[4..<8]))
    if frEqual(cAdd1, swiftAdd1) && frEqual(cAdd2, swiftAdd2) {
        print("  [pass] batch_add: 42+100=142, 999+7=1006")
    } else {
        print("  [FAIL] batch_add: got \(frToInt(cAdd1)[0]), \(frToInt(cAdd2)[0])")
        return
    }

    // Test sub
    let swiftSub1 = frSub(b1, a1)
    let swiftSub2 = frSub(a2, b2)

    let subResult = callBatchSub(b1.to64() + a2.to64(), a1.to64() + b2.to64(), n: 2)
    let cSub1 = Fr.from64(Array(subResult[0..<4]))
    let cSub2 = Fr.from64(Array(subResult[4..<8]))
    if frEqual(cSub1, swiftSub1) && frEqual(cSub2, swiftSub2) {
        print("  [pass] batch_sub: 100-42=58, 999-7=992")
    } else {
        print("  [FAIL] batch_sub: got \(frToInt(cSub1)[0]), \(frToInt(cSub2)[0])")
        return
    }

    // Test neg
    let negResult = callBatchNeg(a1.to64() + a2.to64(), n: 2)
    let neg1 = Fr.from64(Array(negResult[0..<4]))
    let neg2 = Fr.from64(Array(negResult[4..<8]))
    let check1 = frAdd(a1, neg1)
    let check2 = frAdd(a2, neg2)
    if frEqual(check1, Fr.zero) && frEqual(check2, Fr.zero) {
        print("  [pass] batch_neg: a + (-a) = 0")
    } else {
        print("  [FAIL] batch_neg: a + (-a) != 0")
        return
    }

    // Test mul_scalar
    let scalar = frFromInt(5)
    let swiftMul1 = frMul(a1, scalar)
    let swiftMul2 = frMul(a2, scalar)

    let mulResult = callBatchMulScalar(a1.to64() + a2.to64(), scalar.to64(), n: 2)
    let cMul1 = Fr.from64(Array(mulResult[0..<4]))
    let cMul2 = Fr.from64(Array(mulResult[4..<8]))
    if frEqual(cMul1, swiftMul1) && frEqual(cMul2, swiftMul2) {
        print("  [pass] batch_mul_scalar: 42*5=210, 999*5=4995")
    } else {
        print("  [FAIL] batch_mul_scalar: got \(frToInt(cMul1)[0]), \(frToInt(cMul2)[0])")
        return
    }

    // Edge case: large values near p
    let bigVal = frSub(Fr.zero, frFromInt(1)) // p - 1
    let bigArr = bigVal.to64() + bigVal.to64()
    let bigResult = callBatchAdd(bigArr, bigArr, n: 2)
    let bigAdd = Fr.from64(Array(bigResult[0..<4]))
    let expected = frAdd(bigVal, bigVal)
    if frEqual(bigAdd, expected) {
        print("  [pass] batch_add edge: (p-1) + (p-1) = p-2")
    } else {
        print("  [FAIL] batch_add edge: mismatch")
        return
    }

    // --- Performance benchmarks ---
    print("\n--- Performance ---")

    let sizes = [1024, 10_000, 100_000, 1_000_000]

    for n in sizes {
        var rng: UInt64 = 0xDEAD_BEEF_CAFE_BABE &+ UInt64(n)
        var aData = [UInt64](repeating: 0, count: n * 4)
        var bData = [UInt64](repeating: 0, count: n * 4)
        for i in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let val = frFromInt(rng >> 32)
            let limbs = val.to64()
            for j in 0..<4 { aData[i * 4 + j] = limbs[j] }

            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let val2 = frFromInt(rng >> 32)
            let limbs2 = val2.to64()
            for j in 0..<4 { bData[i * 4 + j] = limbs2[j] }
        }

        var resultData = [UInt64](repeating: 0, count: n * 4)
        let scArr = scalar.to64()

        let nk = n >= 1_000_000 ? "\(n / 1_000_000)M" : (n >= 1000 ? "\(n / 1000)K" : "\(n)")

        // Warmup
        _ = callBatchAdd(aData, bData, n: n)

        // Batch add
        var addTimes = [Double]()
        for _ in 0..<5 {
            let t = CFAbsoluteTimeGetCurrent()
            aData.withUnsafeBufferPointer { aP in
                bData.withUnsafeBufferPointer { bP in
                    resultData.withUnsafeMutableBufferPointer { rP in
                        bn254_fr_batch_add_neon(rP.baseAddress!, aP.baseAddress!, bP.baseAddress!, Int32(n))
                    }
                }
            }
            addTimes.append((CFAbsoluteTimeGetCurrent() - t) * 1e6)
        }
        addTimes.sort()
        let addUs = addTimes[2]
        let addNsOp = addUs * 1000.0 / Double(n)

        // Batch sub
        var subTimes = [Double]()
        for _ in 0..<5 {
            let t = CFAbsoluteTimeGetCurrent()
            aData.withUnsafeBufferPointer { aP in
                bData.withUnsafeBufferPointer { bP in
                    resultData.withUnsafeMutableBufferPointer { rP in
                        bn254_fr_batch_sub_neon(rP.baseAddress!, aP.baseAddress!, bP.baseAddress!, Int32(n))
                    }
                }
            }
            subTimes.append((CFAbsoluteTimeGetCurrent() - t) * 1e6)
        }
        subTimes.sort()
        let subUs = subTimes[2]
        let subNsOp = subUs * 1000.0 / Double(n)

        // Batch neg
        var negTimes = [Double]()
        for _ in 0..<5 {
            let t = CFAbsoluteTimeGetCurrent()
            aData.withUnsafeBufferPointer { aP in
                resultData.withUnsafeMutableBufferPointer { rP in
                    bn254_fr_batch_neg_neon(rP.baseAddress!, aP.baseAddress!, Int32(n))
                }
            }
            negTimes.append((CFAbsoluteTimeGetCurrent() - t) * 1e6)
        }
        negTimes.sort()
        let negUs = negTimes[2]
        let negNsOp = negUs * 1000.0 / Double(n)

        // Batch mul_scalar
        var mulTimes = [Double]()
        for _ in 0..<5 {
            let t = CFAbsoluteTimeGetCurrent()
            aData.withUnsafeBufferPointer { aP in
                scArr.withUnsafeBufferPointer { sP in
                    resultData.withUnsafeMutableBufferPointer { rP in
                        bn254_fr_batch_mul_scalar_neon(rP.baseAddress!, aP.baseAddress!, sP.baseAddress!, Int32(n))
                    }
                }
            }
            mulTimes.append((CFAbsoluteTimeGetCurrent() - t) * 1e6)
        }
        mulTimes.sort()
        let mulUs = mulTimes[2]
        let mulNsOp = mulUs * 1000.0 / Double(n)

        // Swift baseline (add only, skip for 1M)
        var swiftAddUs = 0.0
        if n <= 100_000 {
            var frA = [Fr](); frA.reserveCapacity(n)
            var frB = [Fr](); frB.reserveCapacity(n)
            for i in 0..<n {
                frA.append(Fr.from64(Array(aData[i*4..<i*4+4])))
                frB.append(Fr.from64(Array(bData[i*4..<i*4+4])))
            }
            for i in 0..<min(100, n) { _ = frAdd(frA[i], frB[i]) }
            let swT = CFAbsoluteTimeGetCurrent()
            for i in 0..<n { _ = frAdd(frA[i], frB[i]) }
            swiftAddUs = (CFAbsoluteTimeGetCurrent() - swT) * 1e6
        }

        // Parallel add
        var parAddUs = 0.0
        if n >= 4096 {
            // warmup
            aData.withUnsafeBufferPointer { aP in
                bData.withUnsafeBufferPointer { bP in
                    resultData.withUnsafeMutableBufferPointer { rP in
                        bn254_fr_batch_add_parallel(rP.baseAddress!, aP.baseAddress!, bP.baseAddress!, Int32(n))
                    }
                }
            }
            var parTimes = [Double]()
            for _ in 0..<5 {
                let t = CFAbsoluteTimeGetCurrent()
                aData.withUnsafeBufferPointer { aP in
                    bData.withUnsafeBufferPointer { bP in
                        resultData.withUnsafeMutableBufferPointer { rP in
                            bn254_fr_batch_add_parallel(rP.baseAddress!, aP.baseAddress!, bP.baseAddress!, Int32(n))
                        }
                    }
                }
                parTimes.append((CFAbsoluteTimeGetCurrent() - t) * 1e6)
            }
            parTimes.sort()
            parAddUs = parTimes[2]
        }

        print(String(format: "  n=%@: add %7.0f us (%4.1f ns/op) | sub %7.0f us (%4.1f ns/op) | neg %7.0f us (%4.1f ns/op) | mul %7.0f us (%4.1f ns/op)",
                      nk, addUs, addNsOp, subUs, subNsOp, negUs, negNsOp, mulUs, mulNsOp))

        if swiftAddUs > 0 {
            let speedup = swiftAddUs / addUs
            print(String(format: "         Swift add: %.0f us (%.1f ns/op) -> C batch %.1fx faster",
                          swiftAddUs, swiftAddUs * 1000.0 / Double(n), speedup))
        }
        if parAddUs > 0 {
            let parSpeedup = addUs / parAddUs
            print(String(format: "         Parallel add: %.0f us (%.1f ns/op) -> %.1fx vs single-thread",
                          parAddUs, parAddUs * 1000.0 / Double(n), parSpeedup))
        }
    }

    print("\nBatch field benchmark complete.")
}
