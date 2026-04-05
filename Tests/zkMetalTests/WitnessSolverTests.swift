// Witness Solver tests: R1CS witness computation from partial inputs
import zkMetal
import Foundation

public func runWitnessSolverTests() {
    suite("Witness Solver")

    // --- Test 1: Simple multiply: a * b = c ---
    do {
        let one = Fr.one
        var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

        // Constraint: w1 * w2 = w3
        aE.append(R1CSEntry(row: 0, col: 1, val: one))
        bE.append(R1CSEntry(row: 0, col: 2, val: one))
        cE.append(R1CSEntry(row: 0, col: 3, val: one))

        let r1cs = R1CSInstance(numConstraints: 1, numVars: 4, numPublic: 2,
                                aEntries: aE, bEntries: bE, cEntries: cE)

        let a = frFromInt(3)
        let b = frFromInt(7)
        let solver = WitnessSolver()
        let result = solver.solve(r1cs: r1cs, publicInputs: [a, b])

        expect(result.isFullySolved, "Simple multiply: all vars solved")
        let expected_c = frMul(a, b)  // 21
        expect(frEq(result.witness[3], expected_c), "Simple multiply: c = a*b = 21")
        expect(r1cs.isSatisfied(z: result.witness), "Simple multiply: R1CS satisfied")
    }

    // --- Test 2: Chain: a*b=c, c*d=e ---
    do {
        let one = Fr.one
        var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

        // Vars: [1, a, b, d, c, e]  (a,b,d public => numPublic=3, c,e witness)
        // Constraint 0: a * b = c
        aE.append(R1CSEntry(row: 0, col: 1, val: one))
        bE.append(R1CSEntry(row: 0, col: 2, val: one))
        cE.append(R1CSEntry(row: 0, col: 4, val: one))

        // Constraint 1: c * d = e
        aE.append(R1CSEntry(row: 1, col: 4, val: one))
        bE.append(R1CSEntry(row: 1, col: 3, val: one))
        cE.append(R1CSEntry(row: 1, col: 5, val: one))

        let r1cs = R1CSInstance(numConstraints: 2, numVars: 6, numPublic: 3,
                                aEntries: aE, bEntries: bE, cEntries: cE)

        let a = frFromInt(2)
        let b = frFromInt(5)
        let d = frFromInt(4)
        let solver = WitnessSolver()
        let result = solver.solve(r1cs: r1cs, publicInputs: [a, b, d])

        expect(result.isFullySolved, "Chain: all vars solved")
        let expected_c = frMul(a, b)  // 10
        let expected_e = frMul(expected_c, d)  // 40
        expect(frEq(result.witness[4], expected_c), "Chain: c = a*b = 10")
        expect(frEq(result.witness[5], expected_e), "Chain: e = c*d = 40")
        expect(r1cs.isSatisfied(z: result.witness), "Chain: R1CS satisfied")
        expect(result.iterations <= 2, "Chain: solved in <= 2 iterations (got \(result.iterations))")
    }

    // --- Test 3: Addition circuit: a + b = c via R1CS ---
    // R1CS encodes addition as: (a + b) * 1 = c
    do {
        let one = Fr.one
        var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

        // Vars: [1, a, b, c]  (a,b public => numPublic=2)
        // Constraint: (a + b) * 1 = c
        aE.append(R1CSEntry(row: 0, col: 1, val: one))
        aE.append(R1CSEntry(row: 0, col: 2, val: one))
        bE.append(R1CSEntry(row: 0, col: 0, val: one))  // B = 1 (constant)
        cE.append(R1CSEntry(row: 0, col: 3, val: one))

        let r1cs = R1CSInstance(numConstraints: 1, numVars: 4, numPublic: 2,
                                aEntries: aE, bEntries: bE, cEntries: cE)

        let a = frFromInt(13)
        let b = frFromInt(29)
        let solver = WitnessSolver()
        let result = solver.solve(r1cs: r1cs, publicInputs: [a, b])

        expect(result.isFullySolved, "Addition: all vars solved")
        let expected_c = frAdd(a, b)  // 42
        expect(frEq(result.witness[3], expected_c), "Addition: c = a + b = 42")
        expect(r1cs.isSatisfied(z: result.witness), "Addition: R1CS satisfied")
    }

    // --- Test 4: Circom-style circuit: Poseidon-like witness computation ---
    // Simulates a multi-round permutation circuit with intermediate variables
    // R1CS: round_i = (round_{i-1} + constant_i) * round_{i-1} for each round
    // This tests forward propagation through many dependent constraints
    do {
        let numRounds = 8
        // Vars: [1, input, round_0, round_1, ..., round_{n-1}, output]
        // input is public, all rounds + output are witness
        let numVars = 2 + numRounds + 1  // one, input, rounds..., output
        var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

        let roundConstants: [Fr] = (0..<numRounds).map { frFromInt(UInt64($0 + 3)) }

        for i in 0..<numRounds {
            let prevVar = (i == 0) ? 1 : (2 + i - 1)  // input or previous round
            let currVar = 2 + i  // current round output

            // (prev + constant) * prev = curr
            // A = prev + constant (via wire 0 for constant)
            aE.append(R1CSEntry(row: i, col: prevVar, val: Fr.one))
            aE.append(R1CSEntry(row: i, col: 0, val: roundConstants[i]))
            // B = prev
            bE.append(R1CSEntry(row: i, col: prevVar, val: Fr.one))
            // C = curr
            cE.append(R1CSEntry(row: i, col: currVar, val: Fr.one))
        }

        // Final constraint: output = last_round (output * 1 = last_round)
        let lastRound = 2 + numRounds - 1
        let outputVar = 2 + numRounds
        aE.append(R1CSEntry(row: numRounds, col: outputVar, val: Fr.one))
        bE.append(R1CSEntry(row: numRounds, col: 0, val: Fr.one))
        cE.append(R1CSEntry(row: numRounds, col: lastRound, val: Fr.one))

        let r1cs = R1CSInstance(numConstraints: numRounds + 1, numVars: numVars,
                                numPublic: 1, aEntries: aE, bEntries: bE, cEntries: cE)

        let input = frFromInt(5)
        let solver = WitnessSolver()
        let result = solver.solve(r1cs: r1cs, publicInputs: [input])

        expect(result.isFullySolved, "Poseidon-like: all vars solved")

        // Verify by computing expected values manually
        var prev = input
        for i in 0..<numRounds {
            let sum = frAdd(prev, roundConstants[i])
            prev = frMul(sum, prev)
        }
        expect(frEq(result.witness[outputVar], prev),
               "Poseidon-like: output matches manual computation")
        expect(r1cs.isSatisfied(z: result.witness), "Poseidon-like: R1CS satisfied")
    }

    // --- Test 5: Unsolvable detection (missing input) ---
    do {
        let one = Fr.one
        var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

        // Vars: [1, a, b, c] but we only provide a, not b
        // Constraint: a * b = c  -- 2 unknowns, unsolvable without b
        aE.append(R1CSEntry(row: 0, col: 1, val: one))
        bE.append(R1CSEntry(row: 0, col: 2, val: one))
        cE.append(R1CSEntry(row: 0, col: 3, val: one))

        let r1cs = R1CSInstance(numConstraints: 1, numVars: 4, numPublic: 1,
                                aEntries: aE, bEntries: bE, cEntries: cE)

        let solver = WitnessSolver()
        let result = solver.solve(r1cs: r1cs, publicInputs: [frFromInt(5)])

        expect(!result.isFullySolved, "Unsolvable: not fully solved")
        expect(result.unsolvedIndices.contains(2), "Unsolvable: b (idx 2) unsolved")
        expect(result.unsolvedIndices.contains(3), "Unsolvable: c (idx 3) unsolved")
    }

    // --- Test 6: Large circuit: 1000 constraints with timing ---
    do {
        let n = 1000
        // Build a chain: x_i * x_i = x_{i+1} for i in 0..<n
        // Vars: [1, x_0, x_1, ..., x_n]  (x_0 is public input)
        let numVars = 2 + n
        var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

        for i in 0..<n {
            let prevVar = 1 + i
            let nextVar = 2 + i
            aE.append(R1CSEntry(row: i, col: prevVar, val: Fr.one))
            bE.append(R1CSEntry(row: i, col: prevVar, val: Fr.one))
            cE.append(R1CSEntry(row: i, col: nextVar, val: Fr.one))
        }

        let r1cs = R1CSInstance(numConstraints: n, numVars: numVars, numPublic: 1,
                                aEntries: aE, bEntries: bE, cEntries: cE)

        let x0 = frFromInt(2)
        let solver = WitnessSolver()

        let t0 = CFAbsoluteTimeGetCurrent()
        let result = solver.solve(r1cs: r1cs, publicInputs: [x0])
        let elapsed = CFAbsoluteTimeGetCurrent() - t0

        expect(result.isFullySolved, "Large circuit (n=\(n)): all vars solved")
        expect(r1cs.isSatisfied(z: result.witness), "Large circuit (n=\(n)): R1CS satisfied")
        print("  Large circuit (\(n) constraints): \(String(format: "%.2fms", elapsed * 1000)), \(result.iterations) iterations")

        // Verify first few values: 2, 4, 16, 256, ...
        var expected = x0
        for i in 0..<min(5, n) {
            expect(frEq(result.witness[1 + i], expected),
                   "Large circuit: x_\(i) correct")
            expected = frMul(expected, expected)
        }
    }

    // --- Test 7: WitnessGraph dependency analysis ---
    do {
        let one = Fr.one
        var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

        // Two independent constraints followed by one dependent:
        // Constraint 0: a * b = c
        aE.append(R1CSEntry(row: 0, col: 1, val: one))
        bE.append(R1CSEntry(row: 0, col: 2, val: one))
        cE.append(R1CSEntry(row: 0, col: 5, val: one))

        // Constraint 1: d * e = f  (independent of constraint 0)
        aE.append(R1CSEntry(row: 1, col: 3, val: one))
        bE.append(R1CSEntry(row: 1, col: 4, val: one))
        cE.append(R1CSEntry(row: 1, col: 6, val: one))

        // Constraint 2: c * f = g  (depends on both 0 and 1)
        aE.append(R1CSEntry(row: 2, col: 5, val: one))
        bE.append(R1CSEntry(row: 2, col: 6, val: one))
        cE.append(R1CSEntry(row: 2, col: 7, val: one))

        let r1cs = R1CSInstance(numConstraints: 3, numVars: 8, numPublic: 4,
                                aEntries: aE, bEntries: bE, cEntries: cE)
        let cs = R1CSConstraintSet(from: r1cs)
        let graph = WitnessGraph(constraintSet: cs)

        // Known: wire 0 (one) + wires 1,2,3,4 (public)
        let known: Set<Int> = [0, 1, 2, 3, 4]

        // Parallel layers: constraints 0 and 1 are independent (layer 0),
        // constraint 2 depends on both (layer 1)
        let layers = graph.parallelLayers(knownVariables: known)
        expect(layers.count == 2, "Graph layers: 2 layers")
        if layers.count >= 2 {
            expect(layers[0].count == 2, "Graph layers: layer 0 has 2 independent constraints")
            expect(layers[1].count == 1, "Graph layers: layer 1 has 1 dependent constraint")
            expect(layers[1].contains(2), "Graph layers: constraint 2 in layer 1")
        }

        // No circular dependencies
        let cycles = graph.detectCircularDependencies(knownVariables: known)
        expect(cycles.isEmpty, "Graph: no circular dependencies")

        // Stats
        let stats = graph.solvabilityStats(knownVariables: known)
        expect(stats.solvable == 3, "Graph stats: all 3 constraints solvable")
        expect(stats.unsolvable == 0, "Graph stats: 0 unsolvable")
    }

    // --- Test 8: Circular dependency detection ---
    do {
        let one = Fr.one
        var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

        // Vars: [1, a, b] where a and b are both unknown
        // Constraint 0: a * 1 = b  (need a to find b)
        aE.append(R1CSEntry(row: 0, col: 1, val: one))
        bE.append(R1CSEntry(row: 0, col: 0, val: one))
        cE.append(R1CSEntry(row: 0, col: 2, val: one))

        // Constraint 1: b * 1 = a  (need b to find a)
        aE.append(R1CSEntry(row: 1, col: 2, val: one))
        bE.append(R1CSEntry(row: 1, col: 0, val: one))
        cE.append(R1CSEntry(row: 1, col: 1, val: one))

        let r1cs = R1CSInstance(numConstraints: 2, numVars: 3, numPublic: 0,
                                aEntries: aE, bEntries: bE, cEntries: cE)
        let cs = R1CSConstraintSet(from: r1cs)
        let graph = WitnessGraph(constraintSet: cs)

        // Known: only wire 0
        let known: Set<Int> = [0]
        let cycles = graph.detectCircularDependencies(knownVariables: known)
        // Both constraints form a cycle since neither can be solved without the other
        // (a depends on b depends on a)
        // Note: constraint 0 has a as the sole unknown and can be solved if constraint 1
        // is solved first, but constraint 1 similarly depends on constraint 0.
        // Actually: constraint 0 has A=[a], B=[1], C=[b] => 2 unknowns (a and b)
        // constraint 1 has A=[b], B=[1], C=[a] => 2 unknowns
        // So neither is solvable alone.
        expect(!cycles.isEmpty, "Circular deps: detected unsolvable constraints")

        let stats = graph.solvabilityStats(knownVariables: known)
        expect(stats.unsolvable > 0, "Circular deps: unsolvable count > 0")
    }

    // --- Test 9: R1CSConstraintSet from R1CSFileConstraint ---
    do {
        // Build constraints matching Test 1 using Circom parser types
        let fc = R1CSFileConstraint(
            a: R1CSSparseVec(terms: [(wireId: 1, coeff: Fr.one)]),
            b: R1CSSparseVec(terms: [(wireId: 2, coeff: Fr.one)]),
            c: R1CSSparseVec(terms: [(wireId: 3, coeff: Fr.one)])
        )
        let cs = R1CSConstraintSet(from: [fc], numVars: 4, numPublic: 2)
        expect(cs.constraints.count == 1, "FileConstraint: 1 constraint")
        expect(cs.numVars == 4, "FileConstraint: 4 vars")

        let solver = WitnessSolver()
        let result = solver.solve(constraintSet: cs,
                                  knownValues: [0: Fr.one, 1: frFromInt(6), 2: frFromInt(7)])
        expect(result.isFullySolved, "FileConstraint: solved")
        expect(frEq(result.witness[3], frFromInt(42)), "FileConstraint: 6*7=42")
    }

    // --- Test 10: Solve with private hints ---
    do {
        let one = Fr.one
        var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

        // a * b = c, but b is private and not derivable
        // Vars: [1, a, b, c], a public, b private hint, c witness
        aE.append(R1CSEntry(row: 0, col: 1, val: one))
        bE.append(R1CSEntry(row: 0, col: 2, val: one))
        cE.append(R1CSEntry(row: 0, col: 3, val: one))

        let r1cs = R1CSInstance(numConstraints: 1, numVars: 4, numPublic: 1,
                                aEntries: aE, bEntries: bE, cEntries: cE)

        let solver = WitnessSolver()
        let result = solver.solve(r1cs: r1cs,
                                  publicInputs: [frFromInt(5)],
                                  privateHints: [2: frFromInt(11)])

        expect(result.isFullySolved, "Private hints: solved with hint")
        expect(frEq(result.witness[3], frFromInt(55)), "Private hints: 5*11=55")
        expect(r1cs.isSatisfied(z: result.witness), "Private hints: R1CS satisfied")
    }
}
