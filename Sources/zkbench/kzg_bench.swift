// KZG Commitment Benchmark and correctness test
import zkMetal
import Foundation

public func runKZGBench() {
    print("=== KZG Commitment Benchmark (BN254 G1) ===")

    do {
        // Use BN254 generator point
        let gx = fpFromInt(1)
        let gy = fpFromInt(2)
        let generator = PointAffine(x: gx, y: gy)

        // Generate test SRS with known secret (toy setup, NOT secure)
        let secret: [UInt32] = [42, 0, 0, 0, 0, 0, 0, 0]
        let srsSize = 1024

        let t0 = CFAbsoluteTimeGetCurrent()
        let srs = KZGEngine.generateTestSRS(secret: secret, size: srsSize, generator: generator)
        let srsTime = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        print(String(format: "  SRS generation (%d points): %.0f ms", srsSize, srsTime))

        let engine = try KZGEngine(srs: srs)

        // Correctness: commit to polynomial p(x) = 1 + 2x + 3x^2
        let poly = [frFromInt(1), frFromInt(2), frFromInt(3)]
        let commitment = try engine.commit(poly)
        print("  Commitment computed: \(pointIsIdentity(commitment) ? "identity (BAD)" : "non-trivial")")

        // Open at z=5: p(5) = 1 + 10 + 75 = 86
        let z = frFromInt(5)
        let proof = try engine.open(poly, at: z)
        let eval = frToInt(proof.evaluation)[0]
        if eval == 86 {
            print("  [pass] Open: p(5) = 86")
        } else {
            print("  [FAIL] Open: p(5) = \(eval), expected 86")
        }
        print("  Witness computed: \(pointIsIdentity(proof.witness) ? "identity (BAD)" : "non-trivial")")

        // Verify commitment is deterministic
        let c2 = try engine.commit(poly)
        let c1Affine = batchToAffine([commitment])
        let c2Affine = batchToAffine([c2])
        if fpToInt(c1Affine[0].x) == fpToInt(c2Affine[0].x) && fpToInt(c1Affine[0].y) == fpToInt(c2Affine[0].y) {
            print("  [pass] Commitment is deterministic")
        } else {
            print("  [FAIL] Commitment is not deterministic")
        }

        // Benchmark: commit at various polynomial degrees
        print("\n--- KZG Commit Benchmark ---")
        let srsProj = srs.map { pointFromAffine($0) }
        for logN in [8, 10] {
            let n = 1 << logN
            guard n <= srsSize else { continue }
            var coeffs = [Fr](repeating: Fr.zero, count: n)
            var rng: UInt64 = 0xDEAD_BEEF
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                coeffs[i] = frFromInt(rng >> 32)
            }

            // CPU baseline: sequential scalar multiplication (double-and-add)
            let cpuT0 = CFAbsoluteTimeGetCurrent()
            var cpuResult = pointIdentity()
            for i in 0..<n {
                let scalar = frToLimbs(coeffs[i])
                var acc = pointIdentity()
                var base = srsProj[i]
                for limb in scalar {
                    var word = limb
                    for _ in 0..<32 {
                        if word & 1 == 1 { acc = pointAdd(acc, base) }
                        base = pointDouble(base)
                        word >>= 1
                    }
                }
                cpuResult = pointAdd(cpuResult, acc)
            }
            let cpuTime = (CFAbsoluteTimeGetCurrent() - cpuT0) * 1000

            // GPU (warmup + timed)
            let _ = try engine.commit(coeffs)
            var times = [Double]()
            for _ in 0..<5 {
                let t = CFAbsoluteTimeGetCurrent()
                let _ = try engine.commit(coeffs)
                times.append((CFAbsoluteTimeGetCurrent() - t) * 1000)
            }
            times.sort()
            let gpuTime = times[2]
            let speedup = cpuTime / gpuTime
            print(String(format: "  Commit deg 2^%-2d | Vanilla CPU: %8.1fms | GPU: %6.1fms | GPU vs Vanilla: **%.0fx**",
                        logN, cpuTime, gpuTime, speedup))
        }

        // Benchmark: open at various polynomial degrees
        print("\n--- KZG Open Benchmark ---")
        for logN in [8, 10] {
            let n = 1 << logN
            guard n <= srsSize else { continue }
            var coeffs = [Fr](repeating: Fr.zero, count: n)
            var rng: UInt64 = 0xCAFE_BABE
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                coeffs[i] = frFromInt(rng >> 32)
            }
            let challenge = frFromInt(12345)

            // CPU baseline: eval + synthetic division + sequential MSM
            let cpuT0 = CFAbsoluteTimeGetCurrent()
            // Evaluate p(z) via Horner's
            var pz = Fr.zero
            for i in stride(from: n - 1, through: 0, by: -1) {
                pz = frAdd(coeffs[i], frMul(challenge, pz))
            }
            // Synthetic division for quotient
            var quotient = [Fr](repeating: Fr.zero, count: n - 1)
            quotient[n - 2] = coeffs[n - 1]
            for i in stride(from: n - 3, through: 0, by: -1) {
                quotient[i] = frAdd(coeffs[i + 1], frMul(challenge, quotient[i + 1]))
            }
            // Sequential MSM for witness
            var witness = pointIdentity()
            for i in 0..<(n - 1) {
                let scalar = frToLimbs(quotient[i])
                var acc = pointIdentity()
                var base = srsProj[i]
                for limb in scalar {
                    var word = limb
                    for _ in 0..<32 {
                        if word & 1 == 1 { acc = pointAdd(acc, base) }
                        base = pointDouble(base)
                        word >>= 1
                    }
                }
                witness = pointAdd(witness, acc)
            }
            let cpuTime = (CFAbsoluteTimeGetCurrent() - cpuT0) * 1000

            // GPU (warmup + timed)
            let _ = try engine.open(coeffs, at: challenge)
            var times = [Double]()
            for _ in 0..<5 {
                let t = CFAbsoluteTimeGetCurrent()
                let _ = try engine.open(coeffs, at: challenge)
                times.append((CFAbsoluteTimeGetCurrent() - t) * 1000)
            }
            times.sort()
            let gpuTime = times[2]
            let speedup = cpuTime / gpuTime
            print(String(format: "  Open deg 2^%-2d   | Vanilla CPU: %8.1fms | GPU: %6.1fms | GPU vs Vanilla: **%.0fx**",
                        logN, cpuTime, gpuTime, speedup))
        }

    } catch {
        print("  [FAIL] KZG error: \(error)")
    }

    print("\nKZG benchmark complete.")
}
