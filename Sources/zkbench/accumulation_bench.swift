// Accumulation Scheme Benchmark and Correctness Tests
//
// Tests Halo-style IPA accumulation over Pallas curve:
//   - Single IPA proof/verify correctness
//   - Accumulation + decide vs direct verification
//   - Batch decide multiple accumulators
//   - Recursive composition via iterated hashing
//   - Performance: accumulate N + batch decide vs verify N individually

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

        let (gens, qPoint) = PallasAccumulationEngine.generateTestGenerators(count: n)
        let engine = PallasAccumulationEngine(generators: gens, Q: qPoint)

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
        let commitment = engine.commit(a)
        let qProj = pallasPointFromAffine(qPoint)
        let vQ = pallasPointScalarMul(qProj, v)
        let cBound = pallasPointAdd(commitment, vQ)
        fputs("  Commitment computed\n", stderr)

        // Prove
        let t0 = CFAbsoluteTimeGetCurrent()
        let proof = engine.createProof(a: a, b: b)
        let proveTime = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        fputs("  Prove: \(String(format: "%.1f", proveTime)) ms (\(proof.L.count) rounds)\n", stderr)

        // Verify
        let t1 = CFAbsoluteTimeGetCurrent()
        let valid = engine.verify(commitment: cBound, b: b, innerProductValue: v, proof: proof)
        let verifyTime = (CFAbsoluteTimeGetCurrent() - t1) * 1000
        fputs("  Verify: \(String(format: "%.1f", verifyTime)) ms -- \(valid ? "PASS" : "FAIL")\n", stderr)
        allPass = allPass && valid

        // Reject wrong value
        let wrongV = vestaFromInt(999)
        let cBoundWrong = pallasPointAdd(commitment, pallasPointScalarMul(qProj, wrongV))
        let rejected = !engine.verify(commitment: cBoundWrong, b: b, innerProductValue: wrongV, proof: proof)
        fputs("  Reject wrong v: \(rejected ? "PASS" : "FAIL")\n", stderr)
        allPass = allPass && rejected

        // --- Accumulation test ---
        fputs("  --- Accumulation ---\n", stderr)

        let t2 = CFAbsoluteTimeGetCurrent()
        let acc = engine.accumulate(proof: proof, commitment: cBound, b: b, innerProductValue: v)
        let accumTime = (CFAbsoluteTimeGetCurrent() - t2) * 1000
        fputs("  Accumulate: \(String(format: "%.3f", accumTime)) ms\n", stderr)

        let t3 = CFAbsoluteTimeGetCurrent()
        let decideOk = engine.decide(acc)
        let decideTime = (CFAbsoluteTimeGetCurrent() - t3) * 1000
        fputs("  Decide: \(String(format: "%.1f", decideTime)) ms -- \(decideOk ? "PASS" : "FAIL")\n", stderr)
        allPass = allPass && decideOk
    }

    // --- Batch decide multiple accumulators ---
    fputs("\n--- Batch Decide Multiple Accumulators ---\n", stderr)
    do {
        let n = 4
        let (gens, qPoint) = PallasAccumulationEngine.generateTestGenerators(count: n)
        let engine = PallasAccumulationEngine(generators: gens, Q: qPoint)

        let numProofs = 5
        var accumulators = [IPAAccumulator]()

        for p in 0..<numProofs {
            var a = [VestaFp]()
            var b = [VestaFp]()
            for i in 0..<n {
                a.append(vestaFromInt(UInt64(p * n + i + 1)))
                b.append(vestaFromInt(UInt64(i + 1)))
            }
            let proof = engine.createProof(a: a, b: b)
            let v = PallasAccumulationEngine.innerProduct(a, b)
            let c = engine.commit(a)
            let qProj = pallasPointFromAffine(qPoint)
            let cBound = pallasPointAdd(c, pallasPointScalarMul(qProj, v))

            // Verify individually first to confirm proofs are valid
            let valid = engine.verify(commitment: cBound, b: b, innerProductValue: v, proof: proof)
            if !valid {
                fputs("  Proof \(p) verification FAILED\n", stderr)
                allPass = false
            }

            let acc = engine.accumulate(proof: proof, commitment: cBound, b: b, innerProductValue: v)
            accumulators.append(acc)
        }

        // Individual decides
        var individualTotal = 0.0
        var individualAllPass = true
        for i in 0..<numProofs {
            let t = CFAbsoluteTimeGetCurrent()
            let ok = engine.decide(accumulators[i])
            individualTotal += CFAbsoluteTimeGetCurrent() - t
            if !ok { individualAllPass = false }
        }
        fputs("  Individual decide \(numProofs)x: \(String(format: "%.1f", individualTotal * 1000)) ms -- \(individualAllPass ? "ALL PASS" : "SOME FAILED")\n", stderr)
        allPass = allPass && individualAllPass

        // Batch decide
        let t4 = CFAbsoluteTimeGetCurrent()
        let batchOk = engine.batchDecide(accumulators)
        let batchTime = (CFAbsoluteTimeGetCurrent() - t4) * 1000
        let speedup = (individualTotal * 1000) / batchTime
        fputs("  Batch decide \(numProofs)x: \(String(format: "%.1f", batchTime)) ms -- \(batchOk ? "PASS" : "FAIL") (\(String(format: "%.2f", speedup))x vs individual)\n", stderr)
        allPass = allPass && batchOk
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
        fputs("  Batch decider: \(hashValid ? "PASS" : "FAIL")\n", stderr)
        allPass = allPass && hashValid
    }

    // --- Performance comparison ---
    fputs("\n--- Performance Comparison ---\n", stderr)
    do {
        let n = 4
        let (gens, qPoint) = PallasAccumulationEngine.generateTestGenerators(count: n)
        let engine = PallasAccumulationEngine(generators: gens, Q: qPoint)

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
                let c = engine.commit(a)
                let qProj = pallasPointFromAffine(qPoint)
                let cBound = pallasPointAdd(c, pallasPointScalarMul(qProj, v))
                proofs.append((proof, cBound, b, v))
            }

            // Method A: Verify all individually
            let tA0 = CFAbsoluteTimeGetCurrent()
            for (proof, cBound, b, v) in proofs {
                let _ = engine.verify(commitment: cBound, b: b, innerProductValue: v, proof: proof)
            }
            let verifyAllTime = (CFAbsoluteTimeGetCurrent() - tA0) * 1000

            // Method B: Accumulate all + batch decide
            let tB0 = CFAbsoluteTimeGetCurrent()
            var accs = [IPAAccumulator]()
            for (proof, cBound, b, v) in proofs {
                accs.append(engine.accumulate(proof: proof, commitment: cBound, b: b, innerProductValue: v))
            }
            let accumAllTime = (CFAbsoluteTimeGetCurrent() - tB0) * 1000

            let tB1 = CFAbsoluteTimeGetCurrent()
            let _ = engine.batchDecide(accs)
            let batchDecideTime = (CFAbsoluteTimeGetCurrent() - tB1) * 1000

            let totalAccumTime = accumAllTime + batchDecideTime
            let speedup = verifyAllTime / totalAccumTime

            fputs("  N=\(numProofs): verify-all \(String(format: "%.1f", verifyAllTime))ms vs " +
                  "accumulate+batchDecide \(String(format: "%.1f", totalAccumTime))ms " +
                  "(accum \(String(format: "%.1f", accumAllTime))ms + " +
                  "batchDecide \(String(format: "%.1f", batchDecideTime))ms) " +
                  "speedup: \(String(format: "%.2f", speedup))x\n", stderr)
        }
    }

    // --- IPAClaim + AccumulationVerifier + AccumulationDecider ---
    fputs("\n--- IPAClaim / Verifier / Decider ---\n", stderr)
    do {
        let n = 4
        let (gens, qPoint) = PallasAccumulationEngine.generateTestGenerators(count: n)
        let engine = PallasAccumulationEngine(generators: gens, Q: qPoint)

        // Create two claims
        var claims = [PallasIPAClaim]()
        var accumulators = [IPAAccumulator]()
        for p in 0..<3 {
            var a = [VestaFp]()
            var b = [VestaFp]()
            for i in 0..<n {
                a.append(vestaFromInt(UInt64(p * n + i + 1)))
                b.append(vestaFromInt(UInt64(i + 1)))
            }
            let proof = engine.createProof(a: a, b: b)
            let claim = engine.extractClaim(witness: a, evaluationVector: b, proof: proof)
            claims.append(claim)
            accumulators.append(engine.accumulateClaim(claim))
        }

        // Test IPAClaim -> accumulate -> decide
        let claimDecideOk = engine.decide(accumulators[0])
        fputs("  IPAClaim -> accumulate -> decide: \(claimDecideOk ? "PASS" : "FAIL")\n", stderr)
        allPass = allPass && claimDecideOk

        // Test fold two accumulators + verify the fold step
        let (folded, foldProof) = engine.foldAccumulators(accumulators[0], accumulators[1])
        let foldVerifyOk = AccumulationVerifier.verifyStep(
            accPrev: accumulators[0],
            accNew: accumulators[1],
            accOut: folded,
            proof: foldProof
        )
        fputs("  Fold + verify step: \(foldVerifyOk ? "PASS" : "FAIL")\n", stderr)
        allPass = allPass && foldVerifyOk

        // Test chain verification: fold 3 accumulators sequentially
        var chain = [(newAcc: IPAAccumulator, proof: AccumulationProof, resultAcc: IPAAccumulator)]()
        var running = accumulators[0]
        for i in 1..<accumulators.count {
            let (result, proof) = engine.foldAccumulators(running, accumulators[i])
            chain.append((newAcc: accumulators[i], proof: proof, resultAcc: result))
            running = result
        }
        let chainOk = AccumulationVerifier.verifyChain(initialAcc: accumulators[0], steps: chain)
        fputs("  Chain verify (3 folds): \(chainOk ? "PASS" : "FAIL")\n", stderr)
        allPass = allPass && chainOk

        // Test AccumulationDecider.batchDecide
        let batchDecideOk = AccumulationDecider.batchDecide(accumulators, engine: engine)
        fputs("  AccumulationDecider.batchDecide: \(batchDecideOk ? "PASS" : "FAIL")\n", stderr)
        allPass = allPass && batchDecideOk

        // Test AccumulationDecider.accumulateAndDecide
        let (_, fullPipelineOk) = AccumulationDecider.accumulateAndDecide(claims: claims, engine: engine)
        fputs("  Full pipeline (accumulateAndDecide): \(fullPipelineOk ? "PASS" : "FAIL")\n", stderr)
        allPass = allPass && fullPipelineOk
    }

    fputs("\n  Accumulation tests: \(allPass ? "ALL PASS" : "SOME FAILED")\n", stderr)
}
