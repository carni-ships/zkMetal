// Fibonacci CCS — Example R1CS/CCS for iterated Fibonacci computation
//
// One Fibonacci step: given (a, b), produce (b, a+b).
// R1CS encoding:
//   Variables: z = [1, a, b, a_next, b_next]
//   Constraints:
//     1) a_next = b              -> a_next - b = 0
//     2) b_next = a + b          -> b_next - a - b = 0
//
// As R1CS (A*z . B*z = C*z):
//   We encode as:  a_next * 1 = b * 1   and   b_next * 1 = (a + b) * 1
//   Constraint 1: A1*z = a_next, B1*z = 1, C1*z = b
//   Constraint 2: A2*z = b_next, B2*z = 1, C2*z = a + b

import Foundation

// MARK: - Fibonacci CCS Construction

/// Build a CCS instance for a single Fibonacci step.
///
/// Variables: z = [1, a, b, a_next, b_next]
///   index:        0  1  2    3       4
///
/// Constraints (2 rows):
///   Row 0: a_next = b         (a_next - b = 0)
///   Row 1: b_next = a + b     (b_next - a - b = 0)
///
/// R1CS formulation (A*z . B*z = C*z):
///   A = [[0, 0, 0, 1, 0],    B = [[1, 0, 0, 0, 0],    C = [[0, 0, 1, 0, 0],
///        [0, 0, 0, 0, 1]]         [1, 0, 0, 0, 0]]         [0, 1, 1, 0, 0]]
public func buildFibonacciCCS() -> CCSInstance {
    let m = 2  // 2 constraints
    let n = 5  // [1, a, b, a_next, b_next]

    // Matrix A
    var aBuilder = SparseMatrixBuilder(rows: m, cols: n)
    aBuilder.set(row: 0, col: 3, value: Fr.one)  // a_next
    aBuilder.set(row: 1, col: 4, value: Fr.one)  // b_next
    let matA = aBuilder.build()

    // Matrix B (identity-like: just 1 in first column)
    var bBuilder = SparseMatrixBuilder(rows: m, cols: n)
    bBuilder.set(row: 0, col: 0, value: Fr.one)  // 1
    bBuilder.set(row: 1, col: 0, value: Fr.one)  // 1
    let matB = bBuilder.build()

    // Matrix C
    var cBuilder = SparseMatrixBuilder(rows: m, cols: n)
    cBuilder.set(row: 0, col: 2, value: Fr.one)  // b
    cBuilder.set(row: 1, col: 1, value: Fr.one)  // a
    cBuilder.set(row: 1, col: 2, value: Fr.one)  // + b
    let matC = cBuilder.build()

    // Convert to CCS: c1*(A*z . B*z) + c2*(C*z) = 0  where c1=1, c2=-1
    return CCSInstance.fromR1CS(A: matA, B: matB, C: matC, numPublicInputs: 2)
}

/// Generate witness for one Fibonacci step: (a, b) -> (b, a+b)
/// Returns (publicInput=[a, b], witness=[a_next, b_next])
public func fibonacciWitness(a: Fr, b: Fr) -> (publicInput: [Fr], witness: [Fr]) {
    let aNext = b
    let bNext = frAdd(a, b)
    return (publicInput: [a, b], witness: [aNext, bNext])
}

/// Compute N Fibonacci steps starting from (a0, b0) using HyperNova folding.
///
/// Returns the final LCCCS (accumulated proof of all N steps), the final witness,
/// and the sequence of (a, b) values.
public func foldFibonacci(steps: Int, a0: Fr, b0: Fr)
    -> (finalLCCCS: LCCCS, finalWitness: [Fr], values: [(Fr, Fr)], timings: FibFoldTimings)
{
    let ccs = buildFibonacciCCS()
    let engine = HyperNovaEngine(ccs: ccs)

    var timings = FibFoldTimings()

    // Step 0: Initialize with first Fibonacci step
    var a = a0
    var b = b0
    let (pub0, wit0) = fibonacciWitness(a: a, b: b)
    var values = [(a, b)]

    let t0 = CFAbsoluteTimeGetCurrent()
    var runningLCCCS = engine.initialize(witness: wit0, publicInput: pub0)
    var runningWitness = wit0
    timings.initTime = CFAbsoluteTimeGetCurrent() - t0

    // Advance Fibonacci state
    let aNext = b
    b = frAdd(a, b)
    a = aNext
    values.append((a, b))

    // Steps 1..N-1: Fold each new step
    for step in 1..<steps {
        let (pub, wit) = fibonacciWitness(a: a, b: b)

        // Commit new instance + pre-compute affine for transcript
        let tCommit = CFAbsoluteTimeGetCurrent()
        let newCommitment = engine.pp.commit(witness: wit)
        let (cAx, cAy) = engine.commitmentToAffineFr(newCommitment)
        timings.commitTime += CFAbsoluteTimeGetCurrent() - tCommit

        let newCCCS = CCCS(commitment: newCommitment, publicInput: pub,
                           affineX: cAx, affineY: cAy)

        // Fold
        let tFold = CFAbsoluteTimeGetCurrent()
        let (folded, foldedWit, _proof) = engine.fold(
            running: runningLCCCS, runningWitness: runningWitness,
            new: newCCCS, newWitness: wit)
        timings.foldTime += CFAbsoluteTimeGetCurrent() - tFold

        runningLCCCS = folded
        runningWitness = foldedWit
        timings.foldCount += 1

        // Advance Fibonacci state
        let an = b
        b = frAdd(a, b)
        a = an
        values.append((a, b))
    }

    timings.totalTime = timings.initTime + timings.commitTime + timings.foldTime

    return (runningLCCCS, runningWitness, values, timings)
}

/// Timing breakdown for Fibonacci folding.
public struct FibFoldTimings {
    public var initTime: Double = 0
    public var commitTime: Double = 0
    public var foldTime: Double = 0
    public var totalTime: Double = 0
    public var foldCount: Int = 0

    public var perFoldTime: Double {
        foldCount > 0 ? foldTime / Double(foldCount) : 0
    }
}

// MARK: - Verification

/// Verify the Fibonacci folding result.
/// Checks that the accumulated LCCCS is valid (the "decider").
public func verifyFibonacci(engine: HyperNovaEngine, lcccs: LCCCS, witness: [Fr]) -> Bool {
    return engine.decide(lcccs: lcccs, witness: witness)
}
