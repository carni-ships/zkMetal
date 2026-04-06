// GPUHyperPlonkIOPEngine tests
//
// Verifies:
//   - Eq polynomial computation and properties
//   - MLE evaluation via folding
//   - Constraint evaluation over hypercube
//   - Zero-check protocol (prove + verify)
//   - Permutation check via grand product
//   - Lookup argument via logarithmic derivative (LogUp)
//   - Sumcheck round polynomial interpolation
//   - Full HyperPlonk proof generation
//   - Edge cases: single variable, identity permutation, all-zero constraint
//   - GPU/CPU path consistency

import Foundation
import zkMetal

public func runGPUHyperPlonkIOPTests() {
    suite("GPUHyperPlonkIOPEngine")

    let engine = GPUHyperPlonkIOPEngine()

    // ================================================================
    // Helpers
    // ================================================================

    func pseudoRandomFr(seed: inout UInt64) -> Fr {
        seed = seed &* 6364136223846793005 &+ 1442695040888963407
        return frFromInt(seed >> 32)
    }

    func randomEvals(_ logSize: Int, seed: UInt64 = 0xDEAD_BEEF_CAFE_1234) -> [Fr] {
        var rng = seed
        let n = 1 << logSize
        return (0..<n).map { _ in pseudoRandomFr(seed: &rng) }
    }

    func computeSum(_ evals: [Fr]) -> Fr {
        var s = Fr.zero
        for e in evals { s = frAdd(s, e) }
        return s
    }

    // ================================================================
    // SECTION 1: Eq Polynomial
    // ================================================================

    suite("HyperPlonk — Eq Polynomial")

    // Test: eq polynomial sums to 1
    do {
        for numVars in 1...5 {
            let point = (0..<numVars).map { i -> Fr in frFromInt(UInt64(i + 3)) }
            let eqEvals = engine.computeEqPoly(point: point)

            expectEqual(eqEvals.count, 1 << numVars,
                        "eq poly has 2^\(numVars) evaluations")

            // sum of eq(r, x) over all x in {0,1}^n should be 1
            let sum = computeSum(eqEvals)
            expect(frEqual(sum, Fr.one),
                   "eq(\(numVars) vars) sums to 1 over hypercube")
        }
    }

    // Test: eq polynomial at specific points
    do {
        // For 2 variables, r = (r0, r1):
        // eq(r, (0,0)) = (1-r0)(1-r1)
        // eq(r, (0,1)) = (1-r0)*r1
        // eq(r, (1,0)) = r0*(1-r1)
        // eq(r, (1,1)) = r0*r1
        let r0 = frFromInt(3)
        let r1 = frFromInt(7)
        let eqEvals = engine.computeEqPoly(point: [r0, r1])

        let oneMinusR0 = frSub(Fr.one, r0)
        let oneMinusR1 = frSub(Fr.one, r1)

        let expected00 = frMul(oneMinusR0, oneMinusR1)
        let expected01 = frMul(oneMinusR0, r1)
        let expected10 = frMul(r0, oneMinusR1)
        let expected11 = frMul(r0, r1)

        expect(frEqual(eqEvals[0], expected00), "eq(r, (0,0)) = (1-r0)(1-r1)")
        expect(frEqual(eqEvals[1], expected01), "eq(r, (0,1)) = (1-r0)*r1")
        expect(frEqual(eqEvals[2], expected10), "eq(r, (1,0)) = r0*(1-r1)")
        expect(frEqual(eqEvals[3], expected11), "eq(r, (1,1)) = r0*r1")
    }

    // Test: evaluateEq matches computeEqPoly
    do {
        let r = [frFromInt(5), frFromInt(11), frFromInt(17)]
        let eqEvals = engine.computeEqPoly(point: r)

        for idx in 0..<8 {
            // Convert index to binary point
            let x: [Fr] = (0..<3).map { bit in
                (idx >> (2 - bit)) & 1 == 1 ? Fr.one : Fr.zero
            }
            let pointEval = engine.evaluateEq(r: r, x: x)
            expect(frEqual(eqEvals[idx], pointEval),
                   "evaluateEq matches computeEqPoly at index \(idx)")
        }
    }

    // ================================================================
    // SECTION 2: MLE Evaluation
    // ================================================================

    suite("HyperPlonk — MLE Evaluation")

    // Test: constant polynomial
    do {
        let val = frFromInt(42)
        let evals = [val, val, val, val]
        let config = HyperPlonkConfig(numVars: 2)

        let point = [frFromInt(3), frFromInt(7)]
        let result = engine.evaluateMLE(evals: evals, point: point, config: config)
        expect(frEqual(result, val), "Constant MLE evaluates to constant at any point")
    }

    // Test: single variable
    do {
        // f(x) = 3 + 4x => f(0) = 3, f(1) = 7
        let evals = [frFromInt(3), frFromInt(7)]
        let config = HyperPlonkConfig(numVars: 1)

        // f(0) = 3
        let at0 = engine.evaluateMLE(evals: evals, point: [Fr.zero], config: config)
        expect(frEqual(at0, frFromInt(3)), "f(0) = 3 for single-var MLE")

        // f(1) = 7
        let at1 = engine.evaluateMLE(evals: evals, point: [Fr.one], config: config)
        expect(frEqual(at1, frFromInt(7)), "f(1) = 7 for single-var MLE")

        // f(2) = 3 + 4*2 = 11
        let at2 = engine.evaluateMLE(evals: evals, point: [frFromInt(2)], config: config)
        let expected = frAdd(frFromInt(3), frMul(frFromInt(4), frFromInt(2)))
        expect(frEqual(at2, expected), "f(2) = 11 for single-var MLE")
    }

    // Test: two variables, known polynomial
    do {
        // f(x0, x1) evaluations on {0,1}^2:
        // f(0,0) = 1, f(0,1) = 2, f(1,0) = 3, f(1,1) = 4
        // Note: index = x0 * 2 + x1, but MLE stores MSB first:
        // index 0 = (0,0), index 1 = (0,1), index 2 = (1,0), index 3 = (1,1)
        // Wait — the MLE uses MSB = variable 0, so index i's bits are (x0, x1).
        // index 0 = 00 = (x0=0, x1=0), index 1 = 01 = (x0=0, x1=1),
        // index 2 = 10 = (x0=1, x1=0), index 3 = 11 = (x0=1, x1=1)
        let evals: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let config = HyperPlonkConfig(numVars: 2)

        // Evaluate at (0,0) => should be f(0,0) = 1
        let at00 = engine.evaluateMLE(evals: evals, point: [Fr.zero, Fr.zero], config: config)
        expect(frEqual(at00, frFromInt(1)), "MLE at (0,0) = 1")

        // Evaluate at (1,1) => should be f(1,1) = 4
        let at11 = engine.evaluateMLE(evals: evals, point: [Fr.one, Fr.one], config: config)
        expect(frEqual(at11, frFromInt(4)), "MLE at (1,1) = 4")

        // Evaluate at (0,1) => should be f(0,1) = 2
        let at01 = engine.evaluateMLE(evals: evals, point: [Fr.zero, Fr.one], config: config)
        expect(frEqual(at01, frFromInt(2)), "MLE at (0,1) = 2")
    }

    // Test: MLE evaluation matches standalone function
    do {
        let numVars = 4
        let evals = randomEvals(numVars)
        let config = HyperPlonkConfig(numVars: numVars)
        var rng: UInt64 = 0xABCD_1234
        let point = (0..<numVars).map { _ in pseudoRandomFr(seed: &rng) }

        let engineResult = engine.evaluateMLE(evals: evals, point: point, config: config)
        let standaloneResult = evaluateMLEAtPoint(evals: evals, point: point)
        expect(frEqual(engineResult, standaloneResult),
               "Engine MLE matches standalone evaluateMLEAtPoint")
    }

    // Test: sumOverHypercube
    do {
        let evals: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let sum = sumOverHypercube(evals: evals)
        expect(frEqual(sum, frFromInt(10)), "sumOverHypercube([1,2,3,4]) = 10")
    }

    // ================================================================
    // SECTION 3: Constraint Evaluation
    // ================================================================

    suite("HyperPlonk — Constraint Evaluation")

    // Test: arithmetic constraint satisfaction
    do {
        let numVars = 3
        let n = 1 << numVars

        // w0 * w1 = w2, all with selector = 1
        var a = [Fr](repeating: Fr.zero, count: n)
        var b = [Fr](repeating: Fr.zero, count: n)
        var c = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            a[i] = frFromInt(UInt64(i + 1))
            b[i] = frFromInt(UInt64(i + 2))
            c[i] = frMul(a[i], b[i])
        }

        let witness = HyperPlonkWitness(columns: [a, b, c], numVars: numVars)
        let selector = [Fr](repeating: Fr.one, count: n)
        let config = HyperPlonkConfig(numVars: numVars, numWitnessCols: 3, numSelectorCols: 1)
        let constraint = GPUHyperPlonkIOPEngine.arithmeticConstraint()

        let (satisfied, failIdx) = engine.checkConstraintSatisfaction(
            constraint: constraint, witness: witness,
            selectors: [selector], config: config)

        expect(satisfied, "Arithmetic constraint satisfied when w0*w1=w2")
        expectEqual(failIdx, -1, "No failing index for satisfied constraint")
    }

    // Test: arithmetic constraint violation
    do {
        let numVars = 2
        let n = 1 << numVars

        var a = [Fr](repeating: Fr.one, count: n)
        var b = [Fr](repeating: Fr.one, count: n)
        var c = [Fr](repeating: Fr.one, count: n)
        // Make c[2] wrong: a[2]*b[2] = 1 but c[2] = 5
        c[2] = frFromInt(5)

        let witness = HyperPlonkWitness(columns: [a, b, c], numVars: numVars)
        let selector = [Fr](repeating: Fr.one, count: n)
        let config = HyperPlonkConfig(numVars: numVars, numWitnessCols: 3)
        let constraint = GPUHyperPlonkIOPEngine.arithmeticConstraint()

        let (satisfied, failIdx) = engine.checkConstraintSatisfaction(
            constraint: constraint, witness: witness,
            selectors: [selector], config: config)

        expect(!satisfied, "Arithmetic constraint fails when w0*w1 != w2")
        expectEqual(failIdx, 2, "Failing at index 2")
    }

    // Test: selector gating — zero selector turns off constraint
    do {
        let numVars = 2
        let n = 1 << numVars

        var a = [Fr](repeating: frFromInt(2), count: n)
        var b = [Fr](repeating: frFromInt(3), count: n)
        // c is wrong everywhere: 2*3 = 6 but c = 0
        var c = [Fr](repeating: Fr.zero, count: n)
        // But selector is all zeros, so constraint is vacuously satisfied
        let selector = [Fr](repeating: Fr.zero, count: n)

        let witness = HyperPlonkWitness(columns: [a, b, c], numVars: numVars)
        let config = HyperPlonkConfig(numVars: numVars, numWitnessCols: 3)
        let constraint = GPUHyperPlonkIOPEngine.arithmeticConstraint()

        let (satisfied, _) = engine.checkConstraintSatisfaction(
            constraint: constraint, witness: witness,
            selectors: [selector], config: config)

        expect(satisfied, "Zero selector disables constraint")
    }

    // Test: boolean constraint
    do {
        let numVars = 3
        let n = 1 << numVars

        // Valid boolean witness: all 0 or 1
        let values: [Fr] = [Fr.zero, Fr.one, Fr.one, Fr.zero,
                            Fr.one, Fr.zero, Fr.zero, Fr.one]
        let witness = HyperPlonkWitness(columns: [values], numVars: numVars)
        let config = HyperPlonkConfig(numVars: numVars, numWitnessCols: 1, numSelectorCols: 0)
        let constraint = GPUHyperPlonkIOPEngine.booleanConstraint()

        let (satisfied, _) = engine.checkConstraintSatisfaction(
            constraint: constraint, witness: witness,
            selectors: [], config: config)

        expect(satisfied, "Boolean constraint satisfied for {0,1} values")
    }

    // Test: boolean constraint violation
    do {
        let numVars = 2
        let n = 1 << numVars

        // w[1] = 2, which is not boolean: 2*(1-2) = -2 != 0
        let values: [Fr] = [Fr.zero, frFromInt(2), Fr.one, Fr.zero]
        let witness = HyperPlonkWitness(columns: [values], numVars: numVars)
        let config = HyperPlonkConfig(numVars: numVars, numWitnessCols: 1, numSelectorCols: 0)
        let constraint = GPUHyperPlonkIOPEngine.booleanConstraint()

        let (satisfied, failIdx) = engine.checkConstraintSatisfaction(
            constraint: constraint, witness: witness,
            selectors: [], config: config)

        expect(!satisfied, "Boolean constraint fails for value 2")
        expectEqual(failIdx, 1, "Fails at index 1 where value is 2")
    }

    // Test: addition constraint
    do {
        let numVars = 2
        let n = 1 << numVars

        var a = [Fr](repeating: Fr.zero, count: n)
        var b = [Fr](repeating: Fr.zero, count: n)
        var c = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            a[i] = frFromInt(UInt64(i + 1))
            b[i] = frFromInt(UInt64(i + 5))
            c[i] = frAdd(a[i], b[i])
        }

        let witness = HyperPlonkWitness(columns: [a, b, c], numVars: numVars)
        let selector = [Fr](repeating: Fr.one, count: n)
        let config = HyperPlonkConfig(numVars: numVars, numWitnessCols: 3)
        let constraint = GPUHyperPlonkIOPEngine.additionConstraint()

        let (satisfied, _) = engine.checkConstraintSatisfaction(
            constraint: constraint, witness: witness,
            selectors: [selector], config: config)

        expect(satisfied, "Addition constraint satisfied when w0+w1=w2")
    }

    // Test: linear combination constraint
    do {
        let numVars = 2
        let n = 1 << numVars
        let c0 = frFromInt(3)
        let c1 = frFromInt(5)

        var a = [Fr](repeating: Fr.zero, count: n)
        var b = [Fr](repeating: Fr.zero, count: n)
        var c = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            a[i] = frFromInt(UInt64(i + 1))
            b[i] = frFromInt(UInt64(i + 2))
            c[i] = frAdd(frMul(c0, a[i]), frMul(c1, b[i]))
        }

        let witness = HyperPlonkWitness(columns: [a, b, c], numVars: numVars)
        let selector = [Fr](repeating: Fr.one, count: n)
        let config = HyperPlonkConfig(numVars: numVars, numWitnessCols: 3)
        let constraint = GPUHyperPlonkIOPEngine.linearCombinationConstraint(c0: c0, c1: c1)

        let (satisfied, _) = engine.checkConstraintSatisfaction(
            constraint: constraint, witness: witness,
            selectors: [selector], config: config)

        expect(satisfied, "LinCom constraint: 3*w0 + 5*w1 = w2")
    }

    // ================================================================
    // SECTION 4: Zero-Check Protocol
    // ================================================================

    suite("HyperPlonk — Zero-Check Protocol")

    // Test: zero-check on satisfied arithmetic constraint
    do {
        let numVars = 3
        let n = 1 << numVars

        var a = [Fr](repeating: Fr.zero, count: n)
        var b = [Fr](repeating: Fr.zero, count: n)
        var c = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            a[i] = frFromInt(UInt64(i + 1))
            b[i] = frFromInt(UInt64(i + 2))
            c[i] = frMul(a[i], b[i])
        }

        let witness = HyperPlonkWitness(columns: [a, b, c], numVars: numVars)
        let selector = [Fr](repeating: Fr.one, count: n)
        let config = HyperPlonkConfig(numVars: numVars, numWitnessCols: 3, maxConstraintDegree: 3)
        let constraint = GPUHyperPlonkIOPEngine.arithmeticConstraint()

        // Evaluate constraint over hypercube — should be all zeros
        let constraintEvals = engine.evaluateConstraintOverHypercube(
            constraint: constraint, witness: witness,
            selectors: [selector], config: config)

        expect(allZeroOnHypercube(evals: constraintEvals),
               "Constraint evals are all zero for satisfied constraint")

        // Run zero-check
        let transcript = Transcript(label: "test-zc")
        let proof = engine.zeroCheck(
            constraintEvals: constraintEvals,
            numVars: numVars,
            constraintDegree: 3,
            transcript: transcript,
            config: config)

        expectEqual(proof.numVars, numVars, "Zero-check proof has correct numVars")
        expectEqual(proof.roundPolys.count, numVars, "Zero-check has numVars round polys")
        expectEqual(proof.challenges.count, numVars, "Zero-check has numVars challenges")

        // Verify the proof
        let verifyTranscript = Transcript(label: "test-zc")
        let verified = engine.verifyZeroCheck(proof: proof, transcript: verifyTranscript)
        expect(verified, "Zero-check proof verifies for satisfied constraint")
    }

    // Test: zero-check round polynomial consistency
    do {
        let numVars = 2
        let n = 1 << numVars

        // All-zero constraint evals (trivially satisfied)
        let constraintEvals = [Fr](repeating: Fr.zero, count: n)
        let config = HyperPlonkConfig(numVars: numVars, maxConstraintDegree: 2)
        let transcript = Transcript(label: "test-zc-trivial")

        let proof = engine.zeroCheck(
            constraintEvals: constraintEvals,
            numVars: numVars,
            constraintDegree: 2,
            transcript: transcript,
            config: config)

        // For all-zero constraint, h(x) = eq(r,x) * 0 = 0, so sum = 0
        // All round polys should have p(0) + p(1) = 0
        for (i, poly) in proof.roundPolys.enumerated() {
            let sum = frAdd(poly.atZero, poly.atOne)
            expect(frEqual(sum, Fr.zero),
                   "Round \(i): p(0)+p(1) = 0 for all-zero constraint")
        }
    }

    // Test: zero-check with various sizes
    do {
        for numVars in 1...5 {
            let n = 1 << numVars
            // Satisfied boolean constraint: all zeros
            let values = [Fr](repeating: Fr.zero, count: n)
            let witness = HyperPlonkWitness(columns: [values], numVars: numVars)
            let config = HyperPlonkConfig(numVars: numVars, numWitnessCols: 1, numSelectorCols: 0,
                                          maxConstraintDegree: 2)
            let constraint = GPUHyperPlonkIOPEngine.booleanConstraint()

            let constraintEvals = engine.evaluateConstraintOverHypercube(
                constraint: constraint, witness: witness, selectors: [], config: config)

            let transcript = Transcript(label: "test-zc-size-\(numVars)")
            let proof = engine.zeroCheck(
                constraintEvals: constraintEvals, numVars: numVars,
                constraintDegree: 2, transcript: transcript, config: config)

            let verifyTranscript = Transcript(label: "test-zc-size-\(numVars)")
            let ok = engine.verifyZeroCheck(proof: proof, transcript: verifyTranscript)
            expect(ok, "Zero-check verifies for \(numVars) vars boolean constraint")
        }
    }

    // ================================================================
    // SECTION 5: Round Polynomial Interpolation
    // ================================================================

    suite("HyperPlonk — Round Polynomial")

    // Test: degree-1 interpolation
    do {
        let p = HyperPlonkRoundPoly(evals: [frFromInt(5), frFromInt(13)])
        expect(frEqual(p.atZero, frFromInt(5)), "Degree-1: p(0) = 5")
        expect(frEqual(p.atOne, frFromInt(13)), "Degree-1: p(1) = 13")
        expectEqual(p.degree, 1, "Degree-1: degree = 1")

        // p(2) = 5 + 2*(13-5) = 21
        let at2 = p.evaluate(at: frFromInt(2))
        expect(frEqual(at2, frFromInt(21)), "Degree-1: p(2) = 21")

        // p(3) = 5 + 3*8 = 29
        let at3 = p.evaluate(at: frFromInt(3))
        expect(frEqual(at3, frFromInt(29)), "Degree-1: p(3) = 29")
    }

    // Test: degree-2 interpolation
    do {
        // p(x) = 1 + 3x + 2x^2 => p(0)=1, p(1)=6, p(2)=15
        let p = HyperPlonkRoundPoly(evals: [frFromInt(1), frFromInt(6), frFromInt(15)])
        expectEqual(p.degree, 2, "Degree-2: degree = 2")

        // p(3) = 1 + 9 + 18 = 28
        let at3 = p.evaluate(at: frFromInt(3))
        expect(frEqual(at3, frFromInt(28)), "Degree-2: p(3) = 28")

        // p(0) and p(1) match
        expect(frEqual(p.atZero, frFromInt(1)), "Degree-2: p(0) = 1")
        expect(frEqual(p.atOne, frFromInt(6)), "Degree-2: p(1) = 6")
    }

    // Test: degree-3 interpolation (via general Lagrange)
    do {
        // p(x) = x^3: p(0)=0, p(1)=1, p(2)=8, p(3)=27
        let p = HyperPlonkRoundPoly(evals: [Fr.zero, Fr.one, frFromInt(8), frFromInt(27)])
        expectEqual(p.degree, 3, "Degree-3: degree = 3")

        // p(4) = 64
        let at4 = p.evaluate(at: frFromInt(4))
        expect(frEqual(at4, frFromInt(64)), "Degree-3: p(4) = 64")

        // p(5) = 125
        let at5 = p.evaluate(at: frFromInt(5))
        expect(frEqual(at5, frFromInt(125)), "Degree-3: p(5) = 125")
    }

    // Test: equality
    do {
        let p1 = HyperPlonkRoundPoly(evals: [frFromInt(3), frFromInt(7)])
        let p2 = HyperPlonkRoundPoly(evals: [frFromInt(3), frFromInt(7)])
        let p3 = HyperPlonkRoundPoly(evals: [frFromInt(3), frFromInt(8)])
        expect(p1 == p2, "Equal round polys compare equal")
        expect(!(p1 == p3), "Different round polys compare not equal")
    }

    // ================================================================
    // SECTION 6: Permutation Check
    // ================================================================

    suite("HyperPlonk — Permutation Check")

    // Test: identity permutation (always valid)
    do {
        let numVars = 3
        let n = 1 << numVars

        var rng: UInt64 = 0x1234_5678
        let col0 = (0..<n).map { _ in pseudoRandomFr(seed: &rng) }
        let col1 = (0..<n).map { _ in pseudoRandomFr(seed: &rng) }

        let witness = HyperPlonkWitness(columns: [col0, col1], numVars: numVars)
        let perm = HyperPlonkPermutation.identity(numCols: 2, size: n)
        let config = HyperPlonkConfig(numVars: numVars, numWitnessCols: 2)
        let transcript = Transcript(label: "test-perm-id")

        let proof = engine.permutationCheck(
            witness: witness, permutation: perm,
            transcript: transcript, config: config)

        expect(proof.isValid, "Identity permutation is always valid")
        expect(frEqual(proof.finalProduct, Fr.one), "Grand product = 1 for identity perm")
    }

    // Test: simple swap permutation (2 columns, swap positions)
    do {
        let numVars = 2
        let n = 1 << numVars

        // Two columns with same values but permuted
        // col0 = [a, b, c, d], col1 = [a, b, c, d]
        // Permutation: col0[0] <-> col1[0], col0[1] <-> col1[1], etc.
        // For this to be valid, col0[i] must equal col1[i] for all i
        let values: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
        let witness = HyperPlonkWitness(columns: [values, values], numVars: numVars)

        // sigma: col0[i] -> col1[i], col1[i] -> col0[i]
        var sigma0 = [Int](repeating: 0, count: n)
        var sigma1 = [Int](repeating: 0, count: n)
        for i in 0..<n {
            sigma0[i] = 1 * n + i  // col0[i] maps to col1[i]
            sigma1[i] = 0 * n + i  // col1[i] maps to col0[i]
        }
        let perm = HyperPlonkPermutation(sigma: [sigma0, sigma1], numCols: 2, size: n)
        let config = HyperPlonkConfig(numVars: numVars, numWitnessCols: 2)
        let transcript = Transcript(label: "test-perm-swap")

        let proof = engine.permutationCheck(
            witness: witness, permutation: perm,
            transcript: transcript, config: config)

        expect(proof.isValid, "Swap permutation valid when columns have matching values")
    }

    // Test: permutation with larger hypercube
    do {
        let numVars = 4
        let n = 1 << numVars

        // Single column, identity permutation
        var rng: UInt64 = 0xCAFE_BABE
        let col0 = (0..<n).map { _ in pseudoRandomFr(seed: &rng) }

        let witness = HyperPlonkWitness(columns: [col0], numVars: numVars)
        let perm = HyperPlonkPermutation.identity(numCols: 1, size: n)
        let config = HyperPlonkConfig(numVars: numVars, numWitnessCols: 1)
        let transcript = Transcript(label: "test-perm-large")

        let proof = engine.permutationCheck(
            witness: witness, permutation: perm,
            transcript: transcript, config: config)

        expect(proof.isValid, "Identity perm valid for 4-var hypercube")
        expectEqual(proof.grandProductEvals.count, n, "Grand product has n evaluations")
        expect(frEqual(proof.grandProductEvals[0], Fr.one), "Grand product starts at 1")
    }

    // ================================================================
    // SECTION 7: Lookup Argument (LogUp)
    // ================================================================

    suite("HyperPlonk — LogUp Lookup")

    // Test: basic lookup — all queries in table
    do {
        let table = HyperPlonkLookupTable(entries: [
            frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)
        ])
        let queries: [Fr] = [frFromInt(10), frFromInt(30), frFromInt(20), frFromInt(40),
                             frFromInt(10), frFromInt(10)]
        let config = HyperPlonkConfig(numVars: 3, enableLookups: true)
        let transcript = Transcript(label: "test-logup-basic")

        let proof = engine.lookupCheck(
            queries: queries, table: table,
            transcript: transcript, config: config)

        expect(proof.isValid, "LogUp valid when all queries are in table")
        expect(frEqual(proof.logDerivativeSum, Fr.zero),
               "Log-derivative sum is zero for valid lookup")
    }

    // Test: lookup multiplicities
    do {
        let table = HyperPlonkLookupTable(entries: [frFromInt(1), frFromInt(2), frFromInt(3)])
        // Query: 1 appears 3 times, 2 appears 1 time, 3 appears 2 times
        let queries: [Fr] = [frFromInt(1), frFromInt(1), frFromInt(1),
                             frFromInt(2), frFromInt(3), frFromInt(3)]
        let config = HyperPlonkConfig(numVars: 3, enableLookups: true)
        let transcript = Transcript(label: "test-logup-mult")

        let proof = engine.lookupCheck(
            queries: queries, table: table,
            transcript: transcript, config: config)

        // Check multiplicities
        expect(frEqual(proof.multiplicities[0], frFromInt(3)), "m[0] = 3 (value 1 queried 3x)")
        expect(frEqual(proof.multiplicities[1], frFromInt(1)), "m[1] = 1 (value 2 queried 1x)")
        expect(frEqual(proof.multiplicities[2], frFromInt(2)), "m[2] = 2 (value 3 queried 2x)")
        expect(proof.isValid, "LogUp valid with correct multiplicities")
    }

    // Test: lookup with single entry
    do {
        let table = HyperPlonkLookupTable(entries: [frFromInt(42)])
        let queries: [Fr] = [frFromInt(42), frFromInt(42), frFromInt(42), frFromInt(42)]
        let config = HyperPlonkConfig(numVars: 2, enableLookups: true)
        let transcript = Transcript(label: "test-logup-single")

        let proof = engine.lookupCheck(
            queries: queries, table: table,
            transcript: transcript, config: config)

        expect(proof.isValid, "LogUp valid for single-entry table")
        expect(frEqual(proof.multiplicities[0], frFromInt(4)), "Multiplicity = 4")
    }

    // Test: lookup with table entry queried zero times
    do {
        let table = HyperPlonkLookupTable(entries: [frFromInt(1), frFromInt(2), frFromInt(3)])
        // Only query value 2
        let queries: [Fr] = [frFromInt(2), frFromInt(2)]
        let config = HyperPlonkConfig(numVars: 1, enableLookups: true)
        let transcript = Transcript(label: "test-logup-zero-mult")

        let proof = engine.lookupCheck(
            queries: queries, table: table,
            transcript: transcript, config: config)

        expect(proof.isValid, "LogUp valid when some table entries unused")
        expect(frEqual(proof.multiplicities[0], Fr.zero), "m[0] = 0 (value 1 not queried)")
        expect(frEqual(proof.multiplicities[1], frFromInt(2)), "m[1] = 2 (value 2 queried 2x)")
        expect(frEqual(proof.multiplicities[2], Fr.zero), "m[2] = 0 (value 3 not queried)")
    }

    // ================================================================
    // SECTION 8: Witness Generation
    // ================================================================

    suite("HyperPlonk — Witness Generation")

    // Test: arithmetic witness generation
    do {
        let numVars = 3
        let n = 1 << numVars

        let a: [Fr] = (0..<n).map { frFromInt(UInt64($0 + 1)) }
        let b: [Fr] = (0..<n).map { frFromInt(UInt64($0 + 10)) }

        let witness = engine.generateArithmeticWitness(a: a, b: b, numVars: numVars)

        expectEqual(witness.numCols, 3, "Arithmetic witness has 3 columns")
        expectEqual(witness.size, n, "Witness size = 2^numVars")

        // Verify w0 * w1 = w2 at each point
        for i in 0..<n {
            let prod = frMul(witness.columns[0][i], witness.columns[1][i])
            expect(frEqual(prod, witness.columns[2][i]),
                   "Arithmetic witness: w0[\(i)] * w1[\(i)] = w2[\(i)]")
        }
    }

    // Test: boolean witness generation
    do {
        let numVars = 2
        let values: [Fr] = [Fr.zero, Fr.one, Fr.one, Fr.zero]
        let witness = engine.generateBooleanWitness(values: values, numVars: numVars)

        expectEqual(witness.numCols, 1, "Boolean witness has 1 column")
        expectEqual(witness.size, 4, "Boolean witness size = 4")

        // Verify boolean constraint
        for i in 0..<4 {
            let val = witness.columns[0][i]
            let check = frMul(val, frSub(Fr.one, val))
            expect(frEqual(check, Fr.zero), "Boolean witness[\(i)] is 0 or 1")
        }
    }

    // ================================================================
    // SECTION 9: Full Proof Generation
    // ================================================================

    suite("HyperPlonk — Full Proof")

    // Test: full proof with arithmetic constraint
    do {
        let numVars = 3
        let n = 1 << numVars

        let a: [Fr] = (0..<n).map { frFromInt(UInt64($0 + 1)) }
        let b: [Fr] = (0..<n).map { frFromInt(UInt64($0 + 2)) }
        let witness = engine.generateArithmeticWitness(a: a, b: b, numVars: numVars)

        let selector = [Fr](repeating: Fr.one, count: n)
        let constraint = GPUHyperPlonkIOPEngine.arithmeticConstraint()
        let config = HyperPlonkConfig(numVars: numVars, numWitnessCols: 3,
                                      numSelectorCols: 1, maxConstraintDegree: 3)
        let transcript = Transcript(label: "test-full-arith")

        let proof = engine.prove(
            witness: witness, selectors: [selector],
            constraints: [constraint],
            permutation: nil, lookupQueries: nil, lookupTable: nil,
            transcript: transcript, config: config)

        expectEqual(proof.zeroCheckProof.numVars, numVars, "Proof has correct numVars")
        expectEqual(proof.witnessEvals.count, 3, "3 witness evaluations")
        expectEqual(proof.selectorEvals.count, 1, "1 selector evaluation")
        expect(proof.permutationProof == nil, "No permutation proof")
        expect(proof.lookupProof == nil, "No lookup proof")
    }

    // Test: full proof with permutation
    do {
        let numVars = 3
        let n = 1 << numVars

        let a: [Fr] = (0..<n).map { frFromInt(UInt64($0 + 1)) }
        let b: [Fr] = (0..<n).map { frFromInt(UInt64($0 + 2)) }
        let witness = engine.generateArithmeticWitness(a: a, b: b, numVars: numVars)

        let selector = [Fr](repeating: Fr.one, count: n)
        let constraint = GPUHyperPlonkIOPEngine.arithmeticConstraint()
        let perm = HyperPlonkPermutation.identity(numCols: 3, size: n)
        let config = HyperPlonkConfig(numVars: numVars, numWitnessCols: 3,
                                      numSelectorCols: 1, maxConstraintDegree: 3)
        let transcript = Transcript(label: "test-full-perm")

        let proof = engine.prove(
            witness: witness, selectors: [selector],
            constraints: [constraint],
            permutation: perm, lookupQueries: nil, lookupTable: nil,
            transcript: transcript, config: config)

        expect(proof.permutationProof != nil, "Has permutation proof")
        expect(proof.permutationProof?.isValid == true, "Permutation proof is valid")
    }

    // Test: full proof with lookup
    do {
        let numVars = 3
        let n = 1 << numVars

        // Simple circuit: w0 is looked up in a range table [0..15]
        let a: [Fr] = (0..<n).map { frFromInt(UInt64($0)) }
        let b: [Fr] = (0..<n).map { _ in frFromInt(UInt64(1)) }
        let witness = engine.generateArithmeticWitness(a: a, b: b, numVars: numVars)

        let selector = [Fr](repeating: Fr.one, count: n)
        let constraint = GPUHyperPlonkIOPEngine.arithmeticConstraint()
        let table = HyperPlonkLookupTable(entries: (0..<16).map { frFromInt(UInt64($0)) })
        let config = HyperPlonkConfig(numVars: numVars, numWitnessCols: 3,
                                      numSelectorCols: 1, maxConstraintDegree: 3,
                                      enableLookups: true)
        let transcript = Transcript(label: "test-full-lookup")

        let proof = engine.prove(
            witness: witness, selectors: [selector],
            constraints: [constraint],
            permutation: nil, lookupQueries: a, lookupTable: table,
            transcript: transcript, config: config)

        expect(proof.lookupProof != nil, "Has lookup proof")
        expect(proof.lookupProof?.isValid == true, "Lookup proof is valid")
    }

    // Test: full proof with multiple constraints
    do {
        let numVars = 2
        let n = 1 << numVars

        // Boolean values that also satisfy addition
        // w0 = [0, 1, 0, 1], w1 = [0, 0, 1, 0], w2 = w0 + w1 = [0, 1, 1, 1]
        let w0: [Fr] = [Fr.zero, Fr.one, Fr.zero, Fr.one]
        let w1: [Fr] = [Fr.zero, Fr.zero, Fr.one, Fr.zero]
        var w2 = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { w2[i] = frAdd(w0[i], w1[i]) }

        let witness = HyperPlonkWitness(columns: [w0, w1, w2], numVars: numVars)
        let addSelector = [Fr](repeating: Fr.one, count: n)

        let boolConstraint = GPUHyperPlonkIOPEngine.booleanConstraint()
        let addConstraint = GPUHyperPlonkIOPEngine.additionConstraint()

        let config = HyperPlonkConfig(numVars: numVars, numWitnessCols: 3,
                                      numSelectorCols: 1, maxConstraintDegree: 2)
        let transcript = Transcript(label: "test-full-multi")

        let proof = engine.prove(
            witness: witness, selectors: [addSelector],
            constraints: [boolConstraint, addConstraint],
            permutation: nil, lookupQueries: nil, lookupTable: nil,
            transcript: transcript, config: config)

        expectEqual(proof.zeroCheckProof.roundPolys.count, numVars,
                    "Multi-constraint proof has correct number of rounds")
    }

    // ================================================================
    // SECTION 10: Eq Tensor / Batch MLE / Helper Functions
    // ================================================================

    suite("HyperPlonk — MLE Helpers")

    // Test: computeEqTensor matches engine.computeEqPoly
    do {
        let point = [frFromInt(3), frFromInt(7), frFromInt(11)]
        let engineEq = engine.computeEqPoly(point: point)
        let helperEq = computeEqTensor(point: point)

        expectEqual(engineEq.count, helperEq.count, "Same size eq tensors")
        for i in 0..<engineEq.count {
            expect(frEqual(engineEq[i], helperEq[i]),
                   "computeEqTensor matches engine at index \(i)")
        }
    }

    // Test: batchEvaluateMLE
    do {
        let numVars = 3
        let n = 1 << numVars
        var rng: UInt64 = 0xBEEF_DEAD
        let evals1 = (0..<n).map { _ in pseudoRandomFr(seed: &rng) }
        let evals2 = (0..<n).map { _ in pseudoRandomFr(seed: &rng) }
        let evals3 = (0..<n).map { _ in pseudoRandomFr(seed: &rng) }
        let point = (0..<numVars).map { _ in pseudoRandomFr(seed: &rng) }

        let batchResult = batchEvaluateMLE(evalsArray: [evals1, evals2, evals3], point: point)

        expectEqual(batchResult.count, 3, "Batch evaluates 3 MLEs")

        let individual1 = evaluateMLEAtPoint(evals: evals1, point: point)
        let individual2 = evaluateMLEAtPoint(evals: evals2, point: point)
        let individual3 = evaluateMLEAtPoint(evals: evals3, point: point)

        expect(frEqual(batchResult[0], individual1), "Batch[0] matches individual eval")
        expect(frEqual(batchResult[1], individual2), "Batch[1] matches individual eval")
        expect(frEqual(batchResult[2], individual3), "Batch[2] matches individual eval")
    }

    // Test: allZeroOnHypercube
    do {
        let zeros = [Fr](repeating: Fr.zero, count: 8)
        expect(allZeroOnHypercube(evals: zeros), "All-zero evals detected")

        var nonzero = zeros
        nonzero[3] = Fr.one
        expect(!allZeroOnHypercube(evals: nonzero), "Non-zero eval detected")
    }

    // Test: computeMLE is identity (MLE evals ARE the MLE)
    do {
        let evals: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let mle = computeMLE(evals: evals, numVars: 2)
        expectEqual(mle.count, 4, "computeMLE returns same size")
        for i in 0..<4 {
            expect(frEqual(mle[i], evals[i]), "computeMLE is identity at index \(i)")
        }
    }

    // ================================================================
    // SECTION 11: Edge Cases
    // ================================================================

    suite("HyperPlonk — Edge Cases")

    // Test: single variable (numVars = 1)
    do {
        let numVars = 1
        let n = 2

        let a: [Fr] = [frFromInt(3), frFromInt(5)]
        let b: [Fr] = [frFromInt(7), frFromInt(11)]
        let witness = engine.generateArithmeticWitness(a: a, b: b, numVars: numVars)

        let selector = [Fr](repeating: Fr.one, count: n)
        let constraint = GPUHyperPlonkIOPEngine.arithmeticConstraint()
        let config = HyperPlonkConfig(numVars: numVars, numWitnessCols: 3,
                                      maxConstraintDegree: 3)

        let constraintEvals = engine.evaluateConstraintOverHypercube(
            constraint: constraint, witness: witness,
            selectors: [selector], config: config)

        expect(allZeroOnHypercube(evals: constraintEvals),
               "Single-var arithmetic constraint satisfied")

        let transcript = Transcript(label: "test-edge-1var")
        let proof = engine.zeroCheck(
            constraintEvals: constraintEvals, numVars: numVars,
            constraintDegree: 3, transcript: transcript, config: config)

        expectEqual(proof.roundPolys.count, 1, "Single-var zero-check has 1 round")
    }

    // Test: large hypercube (numVars = 8)
    do {
        let numVars = 8
        let n = 1 << numVars

        // Use a simple constraint: w0 * (1 - w0) = 0 with all-zero witness
        let values = [Fr](repeating: Fr.zero, count: n)
        let witness = HyperPlonkWitness(columns: [values], numVars: numVars)
        let config = HyperPlonkConfig(numVars: numVars, numWitnessCols: 1, numSelectorCols: 0,
                                      maxConstraintDegree: 2)
        let constraint = GPUHyperPlonkIOPEngine.booleanConstraint()

        let constraintEvals = engine.evaluateConstraintOverHypercube(
            constraint: constraint, witness: witness, selectors: [], config: config)

        expect(allZeroOnHypercube(evals: constraintEvals),
               "8-var boolean constraint satisfied for all-zero witness")

        let transcript = Transcript(label: "test-edge-8var")
        let proof = engine.zeroCheck(
            constraintEvals: constraintEvals, numVars: numVars,
            constraintDegree: 2, transcript: transcript, config: config)

        expectEqual(proof.roundPolys.count, numVars, "8-var zero-check has 8 rounds")
    }

    // Test: witness with all same values
    do {
        let numVars = 3
        let n = 1 << numVars
        let val = frFromInt(7)

        let a = [Fr](repeating: val, count: n)
        let b = [Fr](repeating: val, count: n)
        let c = [Fr](repeating: frMul(val, val), count: n)

        let witness = HyperPlonkWitness(columns: [a, b, c], numVars: numVars)
        let selector = [Fr](repeating: Fr.one, count: n)
        let constraint = GPUHyperPlonkIOPEngine.arithmeticConstraint()
        let config = HyperPlonkConfig(numVars: numVars, numWitnessCols: 3)

        let (satisfied, _) = engine.checkConstraintSatisfaction(
            constraint: constraint, witness: witness,
            selectors: [selector], config: config)

        expect(satisfied, "Constant witness satisfies arithmetic constraint")
    }

    // Test: HyperPlonkConfig properties
    do {
        let config = HyperPlonkConfig(numVars: 5)
        expectEqual(config.hypercubeSize, 32, "hypercubeSize = 2^5 = 32")
        expectEqual(config.numVars, 5, "numVars = 5")
        expectEqual(config.numWitnessCols, 3, "default numWitnessCols = 3")
        expectEqual(config.maxConstraintDegree, 2, "default maxConstraintDegree = 2")
        expectEqual(config.gpuThreshold, 1024, "default gpuThreshold = 1024")
        expect(!config.enableLookups, "lookups disabled by default")
    }

    // Test: HyperPlonkWitness properties
    do {
        let numVars = 2
        let n = 1 << numVars
        let col0 = [Fr](repeating: Fr.one, count: n)
        let col1 = [Fr](repeating: Fr.zero, count: n)
        let witness = HyperPlonkWitness(columns: [col0, col1], numVars: numVars)

        expectEqual(witness.numCols, 2, "Witness has 2 columns")
        expectEqual(witness.size, 4, "Witness size = 4")
        expectEqual(witness.numVars, 2, "Witness numVars = 2")
    }

    // Test: identity permutation construction
    do {
        let perm = HyperPlonkPermutation.identity(numCols: 3, size: 8)
        expectEqual(perm.numCols, 3, "Identity perm has 3 cols")
        expectEqual(perm.size, 8, "Identity perm has size 8")

        // Verify: sigma[col][i] = col * size + i
        for col in 0..<3 {
            for i in 0..<8 {
                expectEqual(perm.sigma[col][i], col * 8 + i,
                            "Identity perm: sigma[\(col)][\(i)] = \(col*8+i)")
            }
        }
    }

    // ================================================================
    // SECTION 12: Sumcheck consistency checks
    // ================================================================

    suite("HyperPlonk — Sumcheck Consistency")

    // Test: zero-check sumcheck rounds satisfy p(0) + p(1) = claim
    do {
        let numVars = 4
        let n = 1 << numVars

        let a: [Fr] = (0..<n).map { frFromInt(UInt64($0 + 1)) }
        let b: [Fr] = (0..<n).map { frFromInt(UInt64($0 + 2)) }
        let witness = engine.generateArithmeticWitness(a: a, b: b, numVars: numVars)

        let selector = [Fr](repeating: Fr.one, count: n)
        let config = HyperPlonkConfig(numVars: numVars, numWitnessCols: 3,
                                      maxConstraintDegree: 3)
        let constraint = GPUHyperPlonkIOPEngine.arithmeticConstraint()

        let constraintEvals = engine.evaluateConstraintOverHypercube(
            constraint: constraint, witness: witness,
            selectors: [selector], config: config)

        let transcript = Transcript(label: "test-sc-consistency")
        let proof = engine.zeroCheck(
            constraintEvals: constraintEvals, numVars: numVars,
            constraintDegree: 3, transcript: transcript, config: config)

        // First round: p_0(0) + p_0(1) = 0 (the total claim for zero-check)
        let firstSum = frAdd(proof.roundPolys[0].atZero, proof.roundPolys[0].atOne)
        expect(frEqual(firstSum, Fr.zero),
               "First round: p_0(0) + p_0(1) = 0 (zero-check claim)")

        // Subsequent rounds: p_i(0) + p_i(1) = p_{i-1}(r_{i-1})
        for i in 1..<numVars {
            let prevEval = proof.roundPolys[i - 1].evaluate(at: proof.challenges[i - 1])
            let roundSum = frAdd(proof.roundPolys[i].atZero, proof.roundPolys[i].atOne)
            expect(frEqual(roundSum, prevEval),
                   "Round \(i): p_i(0)+p_i(1) = p_{i-1}(r_{i-1})")
        }
    }

    // Test: zero-check verify/reject with tampered proof
    do {
        let numVars = 3
        let n = 1 << numVars

        // Create a valid proof first
        let constraintEvals = [Fr](repeating: Fr.zero, count: n)
        let config = HyperPlonkConfig(numVars: numVars, maxConstraintDegree: 2)
        let transcript = Transcript(label: "test-tamper")
        let proof = engine.zeroCheck(
            constraintEvals: constraintEvals, numVars: numVars,
            constraintDegree: 2, transcript: transcript, config: config)

        // Valid proof should verify
        let vt1 = Transcript(label: "test-tamper")
        expect(engine.verifyZeroCheck(proof: proof, transcript: vt1),
               "Valid proof verifies")

        // Tamper with a round poly
        if !proof.roundPolys.isEmpty {
            var tamperedRoundPolys = proof.roundPolys
            let badEvals = [frFromInt(999), frFromInt(888)]
            tamperedRoundPolys[0] = HyperPlonkRoundPoly(evals: badEvals)

            let tamperedProof = ZeroCheckProof(
                roundPolys: tamperedRoundPolys,
                challenges: proof.challenges,
                finalEval: proof.finalEval,
                randomPoint: proof.randomPoint,
                numVars: proof.numVars)

            let vt2 = Transcript(label: "test-tamper")
            let tamperedResult = engine.verifyZeroCheck(proof: tamperedProof, transcript: vt2)
            expect(!tamperedResult, "Tampered proof does not verify")
        }
    }

    // ================================================================
    // SECTION 13: GPU availability check
    // ================================================================

    suite("HyperPlonk — GPU Status")

    do {
        // Just check that gpuAvailable returns a bool without crashing
        let _ = engine.gpuAvailable
        expect(true, "gpuAvailable check does not crash")
    }

    // Test: version
    do {
        let version = GPUHyperPlonkIOPEngine.version
        expect(!version.version.isEmpty, "Engine has a version string")
        expect(!version.updated.isEmpty, "Engine has an update date")
    }

    // Test: profile flag can be toggled
    do {
        let e = GPUHyperPlonkIOPEngine()
        expect(!e.profile, "Profile is off by default")
        e.profile = true
        expect(e.profile, "Profile can be enabled")
    }

    // ================================================================
    // SECTION 14: Stress test — larger hypercube
    // ================================================================

    suite("HyperPlonk — Stress")

    // Test: 10-variable hypercube full proof (1024 points)
    do {
        let numVars = 10
        let n = 1 << numVars

        var rng: UInt64 = 0xFEED_FACE
        let a = (0..<n).map { _ in pseudoRandomFr(seed: &rng) }
        let b = (0..<n).map { _ in pseudoRandomFr(seed: &rng) }
        let witness = engine.generateArithmeticWitness(a: a, b: b, numVars: numVars)

        let selector = [Fr](repeating: Fr.one, count: n)
        let constraint = GPUHyperPlonkIOPEngine.arithmeticConstraint()
        let config = HyperPlonkConfig(numVars: numVars, numWitnessCols: 3,
                                      maxConstraintDegree: 3)

        let constraintEvals = engine.evaluateConstraintOverHypercube(
            constraint: constraint, witness: witness,
            selectors: [selector], config: config)

        expect(allZeroOnHypercube(evals: constraintEvals),
               "10-var arithmetic constraint satisfied")

        let transcript = Transcript(label: "test-stress-10var")
        let proof = engine.zeroCheck(
            constraintEvals: constraintEvals, numVars: numVars,
            constraintDegree: 3, transcript: transcript, config: config)

        expectEqual(proof.roundPolys.count, numVars, "10-var proof has 10 rounds")

        let verifyTranscript = Transcript(label: "test-stress-10var")
        let verified = engine.verifyZeroCheck(proof: proof, transcript: verifyTranscript)
        expect(verified, "10-var zero-check proof verifies")
    }
}
