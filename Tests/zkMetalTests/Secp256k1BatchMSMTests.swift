// secp256k1 Batch MSM Correctness and Benchmark Tests
// Tests the GPU batchMSM and batchNAFMSM methods against sequential CPU MSMs

import Foundation
import zkMetal

// Helper: generate random scalar
func secpNextScalar(_ rng: inout UInt64) -> [UInt32] {
    var limbs = [UInt32](repeating: 0, count: 8)
    for j in 0..<8 {
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        limbs[j] = UInt32(truncatingIfNeeded: rng >> 32)
    }
    return limbs
}

// Helper: check if two projective points are equal using secpToInt comparison
func secpPointEqual(_ a: SecpPointProjective, _ b: SecpPointProjective) -> Bool {
    if secpPointIsIdentity(a) && secpPointIsIdentity(b) { return true }
    if secpPointIsIdentity(a) || secpPointIsIdentity(b) { return false }
    // Compare in affine form using secpToInt
    let aAff = secpPointToAffine(a)
    let bAff = secpPointToAffine(b)
    return secpToInt(aAff.x) == secpToInt(bAff.x) && secpToInt(aAff.y) == secpToInt(bAff.y)
}

// Helper: check if point is on curve
func secpPointOnCurve(_ p: SecpPointProjective) -> Bool {
    if secpPointIsIdentity(p) { return true }
    let aff = secpPointToAffine(p)
    let y2 = secpSqr(aff.y)
    let x3 = secpMul(secpSqr(aff.x), aff.x)
    let rhs = secpAdd(x3, secpFromInt(7))
    return secpToInt(y2) == secpToInt(rhs)
}

func runSecp256k1BatchMSMTests() {
    // secp256k1 generator point
    let gAff = secp256k1Generator()
    let gProj = secpPointFromAffine(gAff)

    // ============================================================================
    // Correctness Tests: Batch MSM vs Sequential Individual MSMs
    // ============================================================================

    suite("Secp256k1 Batch MSM Correctness")

    // Test small batch: M=16, B=4
    do {
        let M = 16, B = 4, totalPoints = M * B
        var rng: UInt64 = 0xDEAD_BEEF_CAFE_1234

        // Generate points: sequential multiples of G
        var allPoints = [SecpPointAffine]()
        var acc = gProj
        for _ in 0..<totalPoints {
            allPoints.append(secpPointToAffine(acc))
            acc = secpPointAdd(acc, gProj)
        }

        // Generate random scalars for each MSM
        var allScalars = [[UInt32]]()
        for _ in 0..<totalPoints {
            allScalars.append(secpNextScalar(&rng))
        }

        // Run batch MSM
        let engine = try Secp256k1MSM()
        let batchResults = try engine.batchMSM(allPoints: allPoints, allScalars: allScalars, M: M, B: B)

        // Run individual MSMs sequentially and compare
        expect(batchResults.count == B, "batchMSM returns B results")

        for b in 0..<B {
            let startIdx = b * M
            let endIdx = startIdx + M
            let mScalars = Array(allScalars[startIdx..<endIdx])
            let mPts = Array(allPoints[startIdx..<endIdx])
            let expected = cSecpPippengerMSM(points: mPts, scalars: mScalars)

            let equal = secpPointEqual(batchResults[b], expected)
            expect(equal, "batchMSM[\(b)] M=\(M) B=\(B) matches sequential MSM")
        }
    } catch {
        expect(false, "Batch MSM M=16 B=4 error: \(error)")
    }

    // Test medium batch: M=32, B=8
    do {
        let M = 32, B = 8, totalPoints = M * B
        var rng: UInt64 = 0x1234_5678_9ABC_DEF0

        var allPoints = [SecpPointAffine]()
        var acc = gProj
        for _ in 0..<totalPoints {
            allPoints.append(secpPointToAffine(acc))
            acc = secpPointAdd(acc, gProj)
        }

        var allScalars = [[UInt32]]()
        for _ in 0..<totalPoints {
            allScalars.append(secpNextScalar(&rng))
        }

        let engine = try Secp256k1MSM()
        let batchResults = try engine.batchMSM(allPoints: allPoints, allScalars: allScalars, M: M, B: B)

        expect(batchResults.count == B, "batchMSM returns B results")

        for b in 0..<B {
            let startIdx = b * M
            let endIdx = startIdx + M
            let mScalars = Array(allScalars[startIdx..<endIdx])
            let mPts = Array(allPoints[startIdx..<endIdx])
            let expected = cSecpPippengerMSM(points: mPts, scalars: mScalars)

            let equal = secpPointEqual(batchResults[b], expected)
            expect(equal, "batchMSM[\(b)] M=\(M) B=\(B) matches sequential MSM")
        }
    } catch {
        expect(false, "Batch MSM M=32 B=8 error: \(error)")
    }

    // Test max batch: M=64, B=16 (kernel limit)
    do {
        let M = 64, B = 16, totalPoints = M * B
        var rng: UInt64 = 0xFEC8_9A7B_6543_2100

        var allPoints = [SecpPointAffine]()
        var acc = gProj
        for _ in 0..<totalPoints {
            allPoints.append(secpPointToAffine(acc))
            acc = secpPointAdd(acc, gProj)
        }

        var allScalars = [[UInt32]]()
        for _ in 0..<totalPoints {
            allScalars.append(secpNextScalar(&rng))
        }

        let engine = try Secp256k1MSM()
        let batchResults = try engine.batchMSM(allPoints: allPoints, allScalars: allScalars, M: M, B: B)

        expect(batchResults.count == B, "batchMSM returns B results")

        for b in 0..<B {
            let startIdx = b * M
            let endIdx = startIdx + M
            let mScalars = Array(allScalars[startIdx..<endIdx])
            let mPts = Array(allPoints[startIdx..<endIdx])
            let expected = cSecpPippengerMSM(points: mPts, scalars: mScalars)

            let equal = secpPointEqual(batchResults[b], expected)
            expect(equal, "batchMSM[\(b)] M=\(M) B=\(B) matches sequential MSM")
        }
    } catch {
        expect(false, "Batch MSM M=64 B=16 error: \(error)")
    }

    // ============================================================================
    // NAF Batch MSM Correctness Tests
    // ============================================================================

    suite("Secp256k1 Batch NAF MSM Correctness")

    // Test NAF batch: M=32, B=8
    do {
        let M = 32, B = 8, totalPoints = M * B
        var rng: UInt64 = 0xABCD_EF01_2345_6789

        var allPoints = [SecpPointAffine]()
        var acc = gProj
        for _ in 0..<totalPoints {
            allPoints.append(secpPointToAffine(acc))
            acc = secpPointAdd(acc, gProj)
        }

        var allScalars = [[UInt32]]()
        for _ in 0..<totalPoints {
            allScalars.append(secpNextScalar(&rng))
        }

        let engine = try Secp256k1MSM()
        let batchResults = try engine.batchNAFMSM(allPoints: allPoints, allScalars: allScalars, M: M, B: B)

        expect(batchResults.count == B, "batchNAFMSM returns B results")

        for b in 0..<B {
            let startIdx = b * M
            let endIdx = startIdx + M
            let mScalars = Array(allScalars[startIdx..<endIdx])
            let mPts = Array(allPoints[startIdx..<endIdx])
            let expected = cSecpPippengerMSM(points: mPts, scalars: mScalars)

            let equal = secpPointEqual(batchResults[b], expected)
            expect(equal, "batchNAFMSM[\(b)] M=\(M) B=\(B) matches sequential MSM")
        }
    } catch {
        expect(false, "Batch NAF MSM M=32 B=8 error: \(error)")
    }

    // ============================================================================
    // Benchmark Tests: Measure throughput for various M, B configurations
    // ============================================================================

    suite("Secp256k1 Batch MSM Benchmarks")

    // Benchmark: various M values at fixed totalPoints=1024 (smaller for faster testing)
    let totalPoints = 1024

    do {
        var rng: UInt64 = 0x1111_2222_3333_4444

        var allPoints = [SecpPointAffine]()
        var acc = gProj
        for _ in 0..<totalPoints {
            allPoints.append(secpPointToAffine(acc))
            acc = secpPointAdd(acc, gProj)
        }

        var allScalars = [[UInt32]]()
        for _ in 0..<totalPoints {
            allScalars.append(secpNextScalar(&rng))
        }

        let engine = try Secp256k1MSM()

        // Test M=16, B=64
        let t0 = CFAbsoluteTimeGetCurrent()
        let results16 = try engine.batchMSM(allPoints: allPoints, allScalars: allScalars, M: 16, B: 64)
        let t16 = CFAbsoluteTimeGetCurrent() - t0
        print(String(format: "  M=16 B=64: %.2f ms (%d results)", t16 * 1000, results16.count))
        expect(results16.count == 64, "M=16 B=64 returns 64 results")

        // Test M=32, B=32
        let t1 = CFAbsoluteTimeGetCurrent()
        let results32 = try engine.batchMSM(allPoints: allPoints, allScalars: allScalars, M: 32, B: 32)
        let t32 = CFAbsoluteTimeGetCurrent() - t1
        print(String(format: "  M=32 B=32: %.2f ms (%d results)", t32 * 1000, results32.count))
        expect(results32.count == 32, "M=32 B=32 returns 32 results")

        // Test M=64, B=16
        let t2 = CFAbsoluteTimeGetCurrent()
        let results64 = try engine.batchMSM(allPoints: allPoints, allScalars: allScalars, M: 64, B: 16)
        let t64 = CFAbsoluteTimeGetCurrent() - t2
        print(String(format: "  M=64 B=16: %.2f ms (%d results)", t64 * 1000, results64.count))
        expect(results64.count == 16, "M=64 B=16 returns 16 results")
    } catch {
        expect(false, "Batch MSM benchmark error: \(error)")
    }

    // ============================================================================
    // Large Batch Test: 64 x 64 = 4096 points (kernel limit test)
    // ============================================================================

    suite("Secp256k1 Batch MSM Large Scale")

    do {
        let M = 64
        let B = 64  // 64 * 64 = 4096 points total
        let totalPoints = M * B
        var rng: UInt64 = 0xAAAA_BBBB_CCCC_DDDD

        var allPoints = [SecpPointAffine]()
        allPoints.reserveCapacity(totalPoints)
        var acc = gProj
        for _ in 0..<totalPoints {
            allPoints.append(secpPointToAffine(acc))
            acc = secpPointAdd(acc, gProj)
        }

        var allScalars = [[UInt32]]()
        allScalars.reserveCapacity(totalPoints)
        for _ in 0..<totalPoints {
            allScalars.append(secpNextScalar(&rng))
        }

        let engine = try Secp256k1MSM()

        let t0 = CFAbsoluteTimeGetCurrent()
        let results = try engine.batchMSM(allPoints: allPoints, allScalars: allScalars, M: M, B: B)
        let elapsed = CFAbsoluteTimeGetCurrent() - t0

        print(String(format: "  M=%d B=%d (%d total points): %.2f ms", M, B, totalPoints, elapsed * 1000))
        expect(results.count == B, "Large batch returns B results")

        // Verify first few results
        for b in 0..<min(4, B) {
            let startIdx = b * M
            let endIdx = startIdx + M
            let mScalars = Array(allScalars[startIdx..<endIdx])
            let mPts = Array(allPoints[startIdx..<endIdx])
            let expected = cSecpPippengerMSM(points: mPts, scalars: mScalars)
            let equal = secpPointEqual(results[b], expected)
            expect(equal, "Large batch[\(b)] correct")
        }
    } catch {
        expect(false, "Large batch MSM error: \(error)")
    }

    // ============================================================================
    // On-Curve Verification
    // ============================================================================

    suite("Secp256k1 Batch MSM On-Curve")

    do {
        let M = 32, B = 8, totalPoints = M * B
        var rng: UInt64 = 0x5555_6666_7777_8888

        var allPoints = [SecpPointAffine]()
        var acc = gProj
        for _ in 0..<totalPoints {
            allPoints.append(secpPointToAffine(acc))
            acc = secpPointAdd(acc, gProj)
        }

        var allScalars = [[UInt32]]()
        for _ in 0..<totalPoints {
            allScalars.append(secpNextScalar(&rng))
        }

        let engine = try Secp256k1MSM()
        let results = try engine.batchMSM(allPoints: allPoints, allScalars: allScalars, M: M, B: B)

        // Check all results are on curve
        for (i, r) in results.enumerated() {
            let onCurve = secpPointOnCurve(r)
            expect(onCurve, "Result[\(i)] on curve")
        }
    } catch {
        expect(false, "On-curve verification error: \(error)")
    }

    // ============================================================================
    // NAF vs Signed-Digit Comparison (both should produce same results)
    // ============================================================================

    suite("Secp256k1 Batch NAF vs Signed-Digit")

    do {
        let M = 32, B = 8, totalPoints = M * B
        var rng: UInt64 = 0x9999_AAAA_BBBB_CCCC

        var allPoints = [SecpPointAffine]()
        var acc = gProj
        for _ in 0..<totalPoints {
            allPoints.append(secpPointToAffine(acc))
            acc = secpPointAdd(acc, gProj)
        }

        var allScalars = [[UInt32]]()
        for _ in 0..<totalPoints {
            allScalars.append(secpNextScalar(&rng))
        }

        let engine = try Secp256k1MSM()

        let sdResults = try engine.batchMSM(allPoints: allPoints, allScalars: allScalars, M: M, B: B, wb: 7)
        let nafResults = try engine.batchNAFMSM(allPoints: allPoints, allScalars: allScalars, M: M, B: B)

        expect(sdResults.count == B, "Signed-digit returns B results")
        expect(nafResults.count == B, "NAF returns B results")

        for b in 0..<B {
            let equal = secpPointEqual(sdResults[b], nafResults[b])
            expect(equal, "Signed-digit and NAF result[\(b)] match")
        }
    } catch {
        expect(false, "NAF vs Signed-Digit comparison error: \(error)")
    }
}