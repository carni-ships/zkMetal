// Accumulation Scheme Benchmark & Correctness Tests
//
// Tests Halo-style IPA accumulation over Pallas curve:
//   - Single IPA proof/verify correctness
//   - Accumulation + decide vs direct verification
//   - Fold multiple accumulators
//   - Recursive composition via iterated hashing
//   - Performance: accumulate N vs verify N individually

import zkMetal
import Foundation

public func runAccumulationBench() {
    fputs("\n=== IPA Accumulation Scheme (Pallas) ===\n", stderr)

    // --- Pallas curve sanity checks ---
    fputs("\n--- Pallas Curve Sanity ---\n", stderr)

    let g = pallasGenerator()
    let gProj = pallasPointFromAffine(g)
    let g2 = pallasPointDouble(gProj)
    let g2add = pallasPointAdd(gProj, gProj)
    let dblOk = pallasPointEqual(g2, g2add)
    fputs("  2G == G+G: \(dblOk ? "PASS" : "FAIL")\n", stderr)

    let three = vestaFromInt(3)
    let threeG = pallasPointScalarMul(gProj, three)
    let g3add = pallasPointAdd(gProj, pallasPointAdd(gProj, gProj))
    let smulOk = pallasPointEqual(threeG, g3add)
    fputs("  3*G == G+G+G (scalar mul): \(smulOk ? "PASS" : "FAIL")\n", stderr)

    // Identity checks
    let id = pallasPointIdentity()
    let idAdd = pallasPointAdd(gProj, id)
    let idOk = pallasPointEqual(idAdd, gProj)
    fputs("  G + O == G: \(idOk ? "PASS" : "FAIL")\n", stderr)

    // --- IPA Proof on Pallas ---
    var allPass = dblOk && smulOk && idOk

    for logN in [2, 4] {
        let n = 1 << logN
        fputs("\n--- Pallas IPA n=\(n) ---\n", stderr)

        let (gens, Q) = PallasAccumulationEngine.generateTestGenerators(count: n)
        let engine = PallasAccumulationEngine(generators: gens, Q: Q)

        // Create test vectors
        var a = [VestaFp]()
        var b = [VestaFp]()
        for i in 0..<n {
            a.append(vestaFromInt(UInt64(i + 1)))
            b.append(vestaFromInt(UInt64(n - i)))
        }

        // Inner product test
        let v = PallasAccumulationEngine.innerProduct(a, b)
        let vInt = vestaToInt(v)
        fputs("  Inner product <a,b> = [\(vInt[0]), \(vInt[1]), ...]\n", stderr)

        // Commit
        let C = engine.commit(a)
        let qProj = pallasPointFromAffine(Q)
        let vQ = pallasPointScalarMul(qProj, v)
        let Cbound = pallasPointAdd(C, vQ)
        fputs("  Commitment computed\n", stderr)

        // Prove
        let t0 = CFAbsoluteTimeGetCurrent()
        let proof = engine.createProof(a: a, b: b)
        let proveTime = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        fputs("  Prove: \(String(format: "%.1f", proveTime)) ms (\(proof.L.count) rounds)\n", stderr)

        // Verify
        let t1 = CFAbsoluteTimeGetCurrent()
        let valid = engine.verify(commitment: Cbound, b: b, innerProductValue: v, proof: proof)
        let verifyTime = (CFAbsoluteTimeGetCurrent() - t1) * 1000
        fputs("  Verify: \(String(format: "%.1f", verifyTime)) ms — \(valid ? "PASS" : "FAIL")\n", stderr)
        allPass = allPass && valid

        // Reject wrong value
        let wrongV = vestaFromInt(999)
        let CboundWrong = pallasPointAdd(C, pallasPointScalarMul(qProj, wrongV))
        let rejected = !engine.verify(commitment: CboundWrong, b: b, innerProductValue: wrongV, proof: proof)
        fputs("  Reject wrong v: \(rejected ? "PASS" : "FAIL")\n", stderr)
        allPass = allPass && rejected

        // --- Accumulation test ---
        fputs("  --- Accumulation ---\n", stderr)

        let t2 = CFAbsoluteTimeGetCurrent()
        let acc = engine.accumulate(proof: proof, commitment: Cbound, b: b, innerProductValue: v)
        let accumTime = (CFAbsoluteTimeGetCurrent() - t2) * 1000
        fputs("  Accumulate: \(String(format: "%.3f", accumTime)) ms\n", stderr)

        let t3 = CFAbsoluteTimeGetCurrent()
        let decideOk = engine.decide(acc, proofA: proof.a)
        let decideTime = (CFAbsoluteTimeGetCurrent() - t3) * 1000
        fputs("  Decide: \(String(format: "%.1f", decideTime)) ms — \(decideOk ? "PASS" : "FAIL")\n", stderr)
        allPass = allPass && decideOk
    }

    // --- Fold multiple accumulators ---
    fputs("\n--- Fold Multiple Accumulators ---\n", stderr)
    do {
        let n = 4
        let (gens, Q) = PallasAccumulationEngine.generateTestGenerators(count: n)
        let engine = PallasAccumulationEngine(generators: gens, Q: Q)

        let numProofs = 5
        var accumulators = [IPAAccumulator]()
        var proofAValues = [VestaFp]()

        for p in 0..<numProofs {
            var a = [VestaFp]()
            var b = [VestaFp]()
            for i in 0..<n {
                a.append(vestaFromInt(UInt64(p * n + i + 1)))
                b.append(vestaFromInt(UInt64(i + 1)))
            }
            let proof = engine.createProof(a: a, b: b)
            let v = PallasAccumulationEngine.innerProduct(a, b)
            let C = engine.commit(a)
            let qProj = pallasPointFromAffine(Q)
            let Cbound = pallasPointAdd(C, pallasPointScalarMul(qProj, v))

            // Verify individually
            let valid = engine.verify(commitment: Cbound, b: b, innerProductValue: v, proof: proof)
            if !valid {
                fputs("  Proof \(p) verification FAILED\n", stderr)
                allPass = false
            }

            let acc = engine.accumulate(proof: proof, commitment: Cbound, b: b, innerProductValue: v)
            accumulators.append(acc)
            proofAValues.append(proof.a)
        }

        // Fold all
        let t4 = CFAbsoluteTimeGetCurrent()
        let folded = engine.foldMany(accumulators)
        let foldTime = (CFAbsoluteTimeGetCurrent() - t4) * 1000
        fputs("  Fold \(numProofs) accumulators: \(String(format: "%.3f", foldTime)) ms\n", stderr)

        // Individual decides (for comparison)
        var individualTotal = 0.0
        var individualAllPass = true
        for i in 0..<numProofs {
            let t = CFAbsoluteTimeGetCurrent()
            let ok = engine.decide(accumulators[i], proofA: proofAValues[i])
            individualTotal += CFAbsoluteTimeGetCurrent() - t
            if !ok { individualAllPass = false }
        }
        fputs("  Individual decide \(numProofs)x: \(String(format: "%.1f", individualTotal * 1000)) ms — \(individualAllPass ? "ALL PASS" : "SOME FAILED")\n", stderr)
        allPass = allPass && individualAllPass
    }

    // --- Recursive Iterated Hash ---
    fputs("\n--- Recursive Iterated Hash ---\n", stderr)
    do {
        let numSteps = 3
        let t5 = CFAbsoluteTimeGetCurrent()
        let (finalState, hashValid) = IteratedHashDemo.run(steps: numSteps, generatorCount: 4)
        let recursiveTime = (CFAbsoluteTimeGetCurrent() - t5) * 1000
        let fsInt = vestaToInt(finalState)
        fputs("  \(numSteps) steps: \(String(format: "%.1f", recursiveTime)) ms\n", stderr)
        fputs("  Final state: [\(fsInt[0]), ...]\n", stderr)
        fputs("  Decider: \(hashValid ? "PASS" : "FAIL")\n", stderr)
        allPass = allPass && hashValid
    }

    // --- Performance comparison ---
    fputs("\n--- Performance Comparison ---\n", stderr)
    do {
        let n = 4
        let (gens, Q) = PallasAccumulationEngine.generateTestGenerators(count: n)
        let engine = PallasAccumulationEngine(generators: gens, Q: Q)

        let counts = [5, 10, 20]
        for numProofs in counts {
            // Generate proofs
            var proofs = [(PallasIPAProof, PallasPointProjective, [VestaFp], VestaFp)]()
            for p in 0..<numProofs {
                var a = [VestaFp]()
                var b = [VestaFp]()
                for i in 0..<n {
                    a.append(vestaFromInt(UInt64(p * n + i + 1)))
                    b.append(vestaFromInt(UInt64(i + 1)))
                }
                let proof = engine.createProof(a: a, b: b)
                let v = PallasAccumulationEngine.innerProduct(a, b)
                let C = engine.commit(a)
                let qProj = pallasPointFromAffine(Q)
                let Cbound = pallasPointAdd(C, pallasPointScalarMul(qProj, v))
                proofs.append((proof, Cbound, b, v))
            }

            // Method A: Verify all individually
            let tA0 = CFAbsoluteTimeGetCurrent()
            for (proof, Cbound, b, v) in proofs {
                let _ = engine.verify(commitment: Cbound, b: b, innerProductValue: v, proof: proof)
            }
            let verifyAllTime = (CFAbsoluteTimeGetCurrent() - tA0) * 1000

            // Method B: Accumulate all + one decide
            let tB0 = CFAbsoluteTimeGetCurrent()
            var accs = [IPAAccumulator]()
            for (proof, Cbound, b, v) in proofs {
                accs.append(engine.accumulate(proof: proof, commitment: Cbound, b: b, innerProductValue: v))
            }
            let accumAllTime = (CFAbsoluteTimeGetCurrent() - tB0) * 1000

            let tB1 = CFAbsoluteTimeGetCurrent()
            let folded = engine.foldMany(accs)
            let foldTime = (CFAbsoluteTimeGetCurrent() - tB1) * 1000

            let tB2 = CFAbsoluteTimeGetCurrent()
            let _ = engine.decide(folded, proofA: proofs[0].0.a)
            let decideTime = (CFAbsoluteTimeGetCurrent() - tB2) * 1000

            let totalAccumTime = accumAllTime + foldTime + decideTime
            let speedup = verifyAllTime / totalAccumTime

            fputs("  N=\(numProofs): verify-all \(String(format: "%.1f", verifyAllTime))ms vs " +
                  "accumulate+fold+decide \(String(format: "%.1f", totalAccumTime))ms " +
                  "(accum \(String(format: "%.1f", accumAllTime))ms + fold \(String(format: "%.2f", foldTime))ms + " +
                  "decide \(String(format: "%.1f", decideTime))ms) " +
                  "speedup: \(String(format: "%.2f", speedup))x\n", stderr)
        }
    }

    fputs("\n  Accumulation tests: \(allPass ? "ALL PASS" : "SOME FAILED")\n", stderr)
}
