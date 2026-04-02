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
        for logN in [8, 10, 12] {
            let n = 1 << logN
            guard n <= srsSize else { continue }
            var coeffs = [Fr](repeating: Fr.zero, count: n)
            var rng: UInt64 = 0xDEAD_BEEF
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                coeffs[i] = frFromInt(rng >> 32)
            }

            // Warmup
            let _ = try engine.commit(coeffs)

            var times = [Double]()
            for _ in 0..<5 {
                let t = CFAbsoluteTimeGetCurrent()
                let _ = try engine.commit(coeffs)
                times.append((CFAbsoluteTimeGetCurrent() - t) * 1000)
            }
            times.sort()
            print(String(format: "  Commit deg 2^%-2d: %7.2f ms", logN, times[2]))
        }

        // Benchmark: open at various polynomial degrees
        print("\n--- KZG Open Benchmark ---")
        for logN in [8, 10, 12] {
            let n = 1 << logN
            guard n <= srsSize else { continue }
            var coeffs = [Fr](repeating: Fr.zero, count: n)
            var rng: UInt64 = 0xCAFE_BABE
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                coeffs[i] = frFromInt(rng >> 32)
            }
            let challenge = frFromInt(12345)

            // Warmup
            let _ = try engine.open(coeffs, at: challenge)

            var times = [Double]()
            for _ in 0..<5 {
                let t = CFAbsoluteTimeGetCurrent()
                let _ = try engine.open(coeffs, at: challenge)
                times.append((CFAbsoluteTimeGetCurrent() - t) * 1000)
            }
            times.sort()
            print(String(format: "  Open deg 2^%-2d:   %7.2f ms", logN, times[2]))
        }

    } catch {
        print("  [FAIL] KZG error: \(error)")
    }

    print("\nKZG benchmark complete.")
}
