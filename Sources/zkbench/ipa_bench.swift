// IPA (Inner Product Argument) Benchmark & Correctness Tests
import zkMetal
import Foundation

public func runIPABench() {
    fputs("\n=== IPA (Inner Product Argument) ===\n", stderr)

    // --- Basic tests ---
    fputs("\n--- Basic Tests ---\n", stderr)

    let a4 = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let b4 = [frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8)]
    let ip = IPAEngine.innerProduct(a4, b4)
    let ipVal = frToInt(ip)
    let ipOk = ipVal[0] == 70 && ipVal[1] == 0 && ipVal[2] == 0 && ipVal[3] == 0
    fputs("  Inner product <[1,2,3,4],[5,6,7,8]> = 70: \(ipOk ? "PASS" : "FAIL")\n", stderr)

    let gx = fpFromInt(1)
    let gy = fpFromInt(2)
    let gProj = pointFromAffine(PointAffine(x: gx, y: gy))
    let g2 = pointDouble(gProj)
    let eqOk = pointEqual(g2, pointAdd(gProj, gProj))
    fputs("  pointEqual(2G, G+G): \(eqOk ? "PASS" : "FAIL")\n", stderr)

    let three = frFromInt(3)
    let threeG = pointScalarMul(gProj, three)
    let smulOk = pointEqual(threeG, pointAdd(gProj, g2))
    fputs("  3*G == G + 2G: \(smulOk ? "PASS" : "FAIL")\n", stderr)

    // --- IPA proof tests at various sizes ---
    for logN in [2, 4, 6, 8] {
        let n = 1 << logN
        fputs("\n--- IPA n=\(n) ---\n", stderr)
        do {
            let (gens, Q) = IPAEngine.generateTestGenerators(count: n)
            let engine = try IPAEngine(generators: gens, Q: Q)

            var a = [Fr]()
            var b = [Fr]()
            for i in 0..<n {
                a.append(frFromInt(UInt64(i + 1)))
                b.append(frFromInt(UInt64(n - i)))
            }

            let v = IPAEngine.innerProduct(a, b)
            let C = try engine.commit(a)
            let vQ = cPointScalarMul(pointFromAffine(Q), v)
            let Cbound = pointAdd(C, vQ)

            let t0 = CFAbsoluteTimeGetCurrent()
            let proof = try engine.createProof(a: a, b: b)
            let proveTime = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            fputs("  Prove: \(String(format: "%.1f", proveTime)) ms (\(proof.L.count) rounds)\n", stderr)

            let t1 = CFAbsoluteTimeGetCurrent()
            let valid = engine.verify(commitment: Cbound, b: b, innerProductValue: v, proof: proof)
            let verifyTime = (CFAbsoluteTimeGetCurrent() - t1) * 1000
            fputs("  Verify: \(String(format: "%.1f", verifyTime)) ms — \(valid ? "PASS" : "FAIL")\n", stderr)

            // Wrong value test
            let wrongV = frFromInt(999)
            let CboundWrong = pointAdd(C, cPointScalarMul(pointFromAffine(Q), wrongV))
            let rejected = !engine.verify(commitment: CboundWrong, b: b, innerProductValue: wrongV, proof: proof)
            fputs("  Reject wrong v: \(rejected ? "PASS" : "FAIL")\n", stderr)

        } catch {
            fputs("  ERROR: \(error)\n", stderr)
        }
    }

    fputs("\n  IPA basic tests: \(ipOk && eqOk && smulOk ? "ALL PASS" : "SOME FAILED")\n", stderr)
}
