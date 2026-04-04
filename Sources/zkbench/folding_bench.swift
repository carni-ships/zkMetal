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
    let newCCCS = CCCS(commitment: newC, publicInput: pub1)
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
        let newCCCSB = CCCS(commitment: newCB, publicInput: pub1B)

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
}
