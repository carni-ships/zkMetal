// Folding Benchmark — HyperNova folding scheme performance
// Benchmarks: Fibonacci step folding at various step counts

import Foundation
import zkMetal

public func runFoldingBench() {
    fputs("\n--- HyperNova Folding Benchmark ---\n", stderr)
    fputs("Version: \(HyperNovaEngine.version.description)\n", stderr)

    // Verify correctness first
    fputs("\nCorrectness check...\n", stderr)
    let ccs = buildFibonacciCCS()

    // Test CCS satisfaction
    let a0 = frFromInt(1)
    let b0 = frFromInt(1)
    let (pub, wit) = fibonacciWitness(a: a0, b: b0)
    let z = [Fr.one] + pub + wit
    let sat = ccs.isSatisfied(z: z)
    fputs("  CCS satisfied: \(sat)\n", stderr)
    if !sat {
        fputs("  ERROR: CCS not satisfied for basic Fibonacci step!\n", stderr)
        return
    }

    // Test initialize + decide
    let engine = HyperNovaEngine(ccs: ccs)
    let lcccs0 = engine.initialize(witness: wit, publicInput: pub)
    let decided = engine.decide(lcccs: lcccs0, witness: wit)
    fputs("  Initial decide: \(decided)\n", stderr)

    // Test single fold
    let a1 = b0  // = 1
    let b1 = frAdd(a0, b0)  // = 2
    let (pub1, wit1) = fibonacciWitness(a: a1, b: b1)
    let newC = engine.pp.commit(witness: wit1)
    let (cAx1, cAy1) = engine.commitmentToAffineFr(newC)
    let newCCCS = CCCS(commitment: newC, publicInput: pub1, affineX: cAx1, affineY: cAy1)
    let (folded1, foldedWit1, proof1) = engine.fold(
        running: lcccs0, runningWitness: wit,
        new: newCCCS, newWitness: wit1)

    let verifiedFold = engine.verifyFold(running: lcccs0, new: newCCCS,
                                          folded: folded1, proof: proof1)
    fputs("  Fold verified: \(verifiedFold)\n", stderr)

    let decided1 = engine.decide(lcccs: folded1, witness: foldedWit1)
    fputs("  Folded decide: \(decided1)\n", stderr)

    // Benchmark various step counts
    let stepCounts = CommandLine.arguments.contains("--quick")
        ? [10, 100]
        : [10, 100, 1000]

    fputs("\n  Steps |  Total ms | Per-fold us | Commit ms |   Fold ms |  Init ms\n", stderr)
    fputs("  ------|-----------|-------------|-----------|-----------|--------\n", stderr)

    for steps in stepCounts {
        // Warmup
        let _ = foldFibonacci(steps: steps, a0: frFromInt(1), b0: frFromInt(1))

        // Timed run (3 iterations, take median)
        var times = [FibFoldTimings]()
        let iters = steps >= 1000 ? 1 : 3
        for _ in 0..<iters {
            let (_, _, _, timings) = foldFibonacci(steps: steps, a0: frFromInt(1), b0: frFromInt(1))
            times.append(timings)
        }

        // Sort by total time, take median
        times.sort { $0.totalTime < $1.totalTime }
        let t = times[times.count / 2]

        let totalMs = t.totalTime * 1000
        let perFoldUs = t.perFoldTime * 1_000_000
        let commitMs = t.commitTime * 1000
        let foldMs = t.foldTime * 1000
        let initMs = t.initTime * 1000

        fputs(String(format: "  %5d | %9.1f | %11.1f | %9.1f | %9.1f | %7.1f\n",
                     steps, totalMs, perFoldUs, commitMs, foldMs, initMs), stderr)
    }

    // Microbenchmark: measure individual components of a single fold
    fputs("\n  Component breakdown (single fold, 1000 iterations):\n", stderr)
    do {
        let ccsB = buildFibonacciCCS()
        let engineB = HyperNovaEngine(ccs: ccsB)
        let (pubB, witB) = fibonacciWitness(a: frFromInt(1), b: frFromInt(1))
        let lcccsB = engineB.initialize(witness: witB, publicInput: pubB)
        let a1B = frFromInt(1)
        let b1B = frFromInt(2)
        let (pub1B, wit1B) = fibonacciWitness(a: a1B, b: b1B)
        let newCB = engineB.pp.commit(witness: wit1B)
        let (cAxB, cAyB) = engineB.commitmentToAffineFr(newCB)
        let newCCCSB = CCCS(commitment: newCB, publicInput: pub1B, affineX: cAxB, affineY: cAyB)

        let iters = 1000
        // Measure fold
        let tFold = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iters {
            let _ = engineB.fold(running: lcccsB, runningWitness: witB,
                                  new: newCCCSB, newWitness: wit1B)
        }
        let foldUs = (CFAbsoluteTimeGetCurrent() - tFold) / Double(iters) * 1_000_000
        fputs(String(format: "    fold total:     %7.1f us\n", foldUs), stderr)

        // Measure commit alone
        let tCommit = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iters {
            let _ = engineB.pp.commit(witness: wit1B)
        }
        let commitUs = (CFAbsoluteTimeGetCurrent() - tCommit) / Double(iters) * 1_000_000
        fputs(String(format: "    commit:         %7.1f us\n", commitUs), stderr)

        // Measure cPointScalarMul alone
        let tScalar = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iters {
            let _ = cPointScalarMul(newCB, frFromInt(42))
        }
        let scalarUs = (CFAbsoluteTimeGetCurrent() - tScalar) / Double(iters) * 1_000_000
        fputs(String(format: "    cPointScalarMul:%7.1f us\n", scalarUs), stderr)

        // Measure transcript overhead (Poseidon2 default)
        let tTrans = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iters {
            let tr = Transcript(label: "hypernova-fold")
            tr.absorb(Fr.one)
            tr.absorb(Fr.one)
            tr.absorb(Fr.one)
            tr.absorb(Fr.one)
            tr.absorb(Fr.one)
            let _ = tr.squeeze()
        }
        let transUs = (CFAbsoluteTimeGetCurrent() - tTrans) / Double(iters) * 1_000_000
        fputs(String(format: "    transcript P2:  %7.1f us\n", transUs), stderr)

        // Measure Keccak transcript
        let tTransK = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iters {
            let tr = Transcript(label: "hypernova-fold", backend: .keccak256)
            tr.absorb(Fr.one)
            tr.absorb(Fr.one)
            tr.absorb(Fr.one)
            tr.absorb(Fr.one)
            tr.absorb(Fr.one)
            let _ = tr.squeeze()
        }
        let transKUs = (CFAbsoluteTimeGetCurrent() - tTransK) / Double(iters) * 1_000_000
        fputs(String(format: "    transcript K256:%7.1f us\n", transKUs), stderr)
    }

    // Print comparison: N folds vs N individual proofs (estimated)
    fputs("\n  Folding advantage: fold N steps into 1 instance.\n", stderr)
    fputs("  Final instance size is O(1) regardless of N.\n", stderr)
    fputs("  A single SNARK proof on the final instance proves all N steps.\n", stderr)

    // --- HyperNovaProver / HyperNovaVerifier API ---
    fputs("\n--- HyperNova Prover/Verifier API ---\n", stderr)

    // Test Prover + Verifier with CommittedCCSInstance
    let prover = HyperNovaProver(ccs: ccs)
    let verifier = HyperNovaVerifier(engine: prover.engine)

    // Initialize
    let (running0, runWit0) = prover.initialize(witness: wit, publicInput: pub)
    let decided0 = prover.decide(instance: running0, witness: runWit0)
    fputs("  Prover init decide: \(decided0)\n", stderr)

    // 2-fold via Prover API
    let newInst1 = prover.commitWitness(wit1, publicInput: pub1)
    let (folded2, foldedWit2, proof2) = prover.fold(
        running: running0, runningWitness: runWit0,
        new: newInst1, newWitness: wit1)
    let vOk2 = verifier.verifyFold(running: running0, new: newInst1,
                                    folded: folded2, proof: proof2)
    fputs("  2-fold verify: \(vOk2)\n", stderr)
    let decided2 = prover.decide(instance: folded2, witness: foldedWit2)
    fputs("  2-fold decide: \(decided2)\n", stderr)

    // Multi-fold: fold 4 instances at once
    fputs("\n  Multi-fold (4 instances):\n", stderr)
    let a2 = frFromInt(2), b2 = frFromInt(3)
    let a3 = frFromInt(3), b3 = frFromInt(5)
    let (pub2, wit2) = fibonacciWitness(a: a2, b: b2)
    let (pub3, wit3) = fibonacciWitness(a: a3, b: b3)
    let inst2 = prover.commitWitness(wit2, publicInput: pub2)
    let inst3 = prover.commitWitness(wit3, publicInput: pub3)

    let (foldedMulti, foldedMultiWit, multiProof) = prover.multiFold(
        instances: [folded2, inst2, newInst1, inst3],
        witnesses: [foldedWit2, wit2, wit1, wit3])

    let vOkMulti = verifier.verifyMultiFold(
        instances: [folded2, inst2, newInst1, inst3],
        folded: foldedMulti, proof: multiProof)
    fputs("    multi-fold verify: \(vOkMulti)\n", stderr)
    let decidedMulti = prover.decide(instance: foldedMulti, witness: foldedMultiWit)
    fputs("    multi-fold decide: \(decidedMulti)\n", stderr)

    // IVC chain benchmark
    fputs("\n  IVC chain (10 Fibonacci steps):\n", stderr)
    var ivcSteps = [(publicInput: [Fr], witness: [Fr])]()
    var ivcA = frFromInt(1), ivcB = frFromInt(1)
    for _ in 0..<10 {
        let (p, w) = fibonacciWitness(a: ivcA, b: ivcB)
        ivcSteps.append((publicInput: p, witness: w))
        let nextA = ivcB
        ivcB = frAdd(ivcA, ivcB)
        ivcA = nextA
    }
    let tIVC = CFAbsoluteTimeGetCurrent()
    let (ivcResult, ivcWit) = prover.ivcChain(steps: ivcSteps)
    let ivcMs = (CFAbsoluteTimeGetCurrent() - tIVC) * 1000
    let ivcDecided = prover.decide(instance: ivcResult, witness: ivcWit)
    fputs(String(format: "    IVC chain: %.1f ms, decide: %@\n", ivcMs, ivcDecided ? "true" : "false"), stderr)

    // Multi-fold timing
    fputs("\n  Multi-fold timing (N=2,4,8):\n", stderr)
    for batchN in [2, 4, 8] {
        // Build N instances: first is running, rest are fresh
        let proverN = HyperNovaProver(ccs: ccs)
        let (runN, runWitN) = proverN.initialize(witness: wit, publicInput: pub)
        var instArr = [runN]
        var witArr = [runWitN]
        var fA = frFromInt(1), fB = frFromInt(2)
        for _ in 1..<batchN {
            let (p, w) = fibonacciWitness(a: fA, b: fB)
            instArr.append(proverN.commitWitness(w, publicInput: p))
            witArr.append(w)
            let nA = fB; fB = frAdd(fA, fB); fA = nA
        }
        // Warmup
        let _ = proverN.multiFold(instances: instArr, witnesses: witArr)
        // Timed
        let tMF = CFAbsoluteTimeGetCurrent()
        let iters = 100
        for _ in 0..<iters {
            let _ = proverN.multiFold(instances: instArr, witnesses: witArr)
        }
        let mfUs = (CFAbsoluteTimeGetCurrent() - tMF) / Double(iters) * 1_000_000
        fputs(String(format: "    N=%d: %.1f us/fold\n", batchN, mfUs), stderr)
    }
}
