// GPUHalo2BackendTests — Tests for GPU-accelerated Halo2 backend engine
//
// Tests cover:
//   1. Column assignment: advice, fixed, instance column extraction
//   2. Custom gate evaluation: arithmetic constraints via Halo2Expression
//   3. Lookup integration: table containment check
//   4. Permutation check: copy constraint satisfaction via grand product
//   5. Vanishing argument: quotient construction from combined constraints

import zkMetal
import Foundation

public func runGPUHalo2BackendTests() {
    suite("GPU Halo2 Backend Engine")

    // ========== Test 1: Column assignment ==========
    do {
        let engine = GPUHalo2BackendEngine()

        // Create a constraint system with advice, fixed, instance columns
        let cs = Halo2ConstraintSystem()
        let advA = cs.adviceColumn()
        let advB = cs.adviceColumn()
        let fixC = cs.fixedColumn()
        let inst = cs.instanceColumn()
        let _ = (advA, advB, fixC, inst)

        // Create assignment with 4 rows
        let assignment = Halo2Assignment(
            numAdvice: 2, numFixed: 1, numInstance: 1,
            numSelectors: 0, numRows: 4
        )

        // Fill in values
        let v1 = frFromInt(10)
        let v2 = frFromInt(20)
        let v3 = frFromInt(30)
        let v4 = frFromInt(40)

        assignment.setAdvice(column: 0, row: 0, value: v1)
        assignment.setAdvice(column: 0, row: 1, value: v2)
        assignment.setAdvice(column: 1, row: 0, value: v3)
        assignment.setAdvice(column: 1, row: 1, value: v4)
        assignment.setFixed(column: 0, row: 0, value: Fr.one)
        assignment.setFixed(column: 0, row: 1, value: Fr.one)

        let store = engine.extractColumns(cs: cs, assignment: assignment)

        expect(store.domainSize == 4, "column store: domain padded to power of 2")
        expect(store.adviceEvals.count == 2, "column store: 2 advice columns")
        expect(store.fixedEvals.count == 1, "column store: 1 fixed column")
        expect(store.instanceEvals.count == 1, "column store: 1 instance column")
        expect(frEqual(store.adviceEvals[0][0], v1), "column store: advice[0][0] correct")
        expect(frEqual(store.adviceEvals[0][1], v2), "column store: advice[0][1] correct")
        expect(frEqual(store.adviceEvals[1][0], v3), "column store: advice[1][0] correct")
        expect(frEqual(store.fixedEvals[0][0], Fr.one), "column store: fixed[0][0] correct")
        // Padding rows should be zero
        expect(frEqual(store.adviceEvals[0][2], Fr.zero), "column store: padding row is zero")
        expect(frEqual(store.adviceEvals[0][3], Fr.zero), "column store: padding row 3 is zero")
    }

    // ========== Test 2: Custom gate evaluation ==========
    do {
        let engine = GPUHalo2BackendEngine()

        // Build a simple multiplication gate: s * (a * b - c) = 0
        let cs = Halo2ConstraintSystem()
        let advA = cs.adviceColumn()
        let advB = cs.adviceColumn()
        let advC = cs.adviceColumn()
        let sel = cs.selector()

        cs.createGate("mul") { vc in
            let s = vc.querySelector(sel)
            let a = vc.queryAdvice(advA, at: .cur)
            let b = vc.queryAdvice(advB, at: .cur)
            let c = vc.queryAdvice(advC, at: .cur)
            return [s * (a * b - c)]
        }

        // Assignment: row 0: a=3, b=5, c=15 (valid: 3*5=15)
        //             row 1: a=2, b=7, c=14 (valid: 2*7=14)
        let assignment = Halo2Assignment(
            numAdvice: 3, numFixed: 0, numInstance: 0,
            numSelectors: 1, numRows: 4
        )

        assignment.setAdvice(column: 0, row: 0, value: frFromInt(3))
        assignment.setAdvice(column: 1, row: 0, value: frFromInt(5))
        assignment.setAdvice(column: 2, row: 0, value: frFromInt(15))
        assignment.setSelector(index: 0, row: 0, value: Fr.one)

        assignment.setAdvice(column: 0, row: 1, value: frFromInt(2))
        assignment.setAdvice(column: 1, row: 1, value: frFromInt(7))
        assignment.setAdvice(column: 2, row: 1, value: frFromInt(14))
        assignment.setSelector(index: 0, row: 1, value: Fr.one)

        let store = engine.extractColumns(cs: cs, assignment: assignment)
        let result = engine.evaluateGates(cs: cs, store: store)

        expect(result.isSatisfied, "gate eval: valid multiplication gate satisfied")
        expect(result.failingRows.isEmpty, "gate eval: no failing rows for valid witness")

        // Now test with invalid witness: c = 16 instead of 15
        let badAssignment = Halo2Assignment(
            numAdvice: 3, numFixed: 0, numInstance: 0,
            numSelectors: 1, numRows: 4
        )
        badAssignment.setAdvice(column: 0, row: 0, value: frFromInt(3))
        badAssignment.setAdvice(column: 1, row: 0, value: frFromInt(5))
        badAssignment.setAdvice(column: 2, row: 0, value: frFromInt(16))  // wrong!
        badAssignment.setSelector(index: 0, row: 0, value: Fr.one)

        let badStore = engine.extractColumns(cs: cs, assignment: badAssignment)
        let badResult = engine.evaluateGates(cs: cs, store: badStore)

        expect(!badResult.isSatisfied, "gate eval: invalid witness detected")
        expect(badResult.failingRows.contains(0), "gate eval: row 0 fails for bad witness")
    }

    // ========== Test 3: Lookup integration ==========
    do {
        let engine = GPUHalo2BackendEngine()

        let cs = Halo2ConstraintSystem()
        let advA = cs.adviceColumn()
        let tableCol = cs.fixedColumn()

        // Register a lookup: advice values must appear in fixed table
        cs.lookup("range_check") { vc in
            let input = vc.queryAdvice(advA, at: .cur)
            let table = vc.queryFixed(tableCol, at: .cur)
            return [(input, table)]
        }

        // Table contains: {1, 2, 3, 4}
        // Witness uses: {1, 3, 0, 0} -- 0 is skipped (treated as inactive)
        let assignment = Halo2Assignment(
            numAdvice: 1, numFixed: 1, numInstance: 0,
            numSelectors: 0, numRows: 4
        )

        assignment.setAdvice(column: 0, row: 0, value: frFromInt(1))
        assignment.setAdvice(column: 0, row: 1, value: frFromInt(3))

        assignment.setFixed(column: 0, row: 0, value: frFromInt(1))
        assignment.setFixed(column: 0, row: 1, value: frFromInt(2))
        assignment.setFixed(column: 0, row: 2, value: frFromInt(3))
        assignment.setFixed(column: 0, row: 3, value: frFromInt(4))

        let store = engine.extractColumns(cs: cs, assignment: assignment)
        let lookupResults = engine.checkLookups(cs: cs, store: store)

        expect(lookupResults.count == 1, "lookup: one lookup argument")
        expect(lookupResults[0] == true, "lookup: valid inputs found in table")

        // Now test with a value NOT in the table
        let badAssignment = Halo2Assignment(
            numAdvice: 1, numFixed: 1, numInstance: 0,
            numSelectors: 0, numRows: 4
        )
        badAssignment.setAdvice(column: 0, row: 0, value: frFromInt(99)) // not in table!
        badAssignment.setFixed(column: 0, row: 0, value: frFromInt(1))
        badAssignment.setFixed(column: 0, row: 1, value: frFromInt(2))
        badAssignment.setFixed(column: 0, row: 2, value: frFromInt(3))
        badAssignment.setFixed(column: 0, row: 3, value: frFromInt(4))

        let badStore = engine.extractColumns(cs: cs, assignment: badAssignment)
        let badLookup = engine.checkLookups(cs: cs, store: badStore)
        expect(badLookup[0] == false, "lookup: invalid input detected")
    }

    // ========== Test 4: Permutation check ==========
    do {
        let engine = GPUHalo2BackendEngine()

        let cs = Halo2ConstraintSystem()
        let advA = cs.adviceColumn()
        let advB = cs.adviceColumn()
        cs.enableEquality(advA)
        cs.enableEquality(advB)

        // Assignment: a[0] = 42, b[1] = 42 with copy constraint a[0] == b[1]
        let assignment = Halo2Assignment(
            numAdvice: 2, numFixed: 0, numInstance: 0,
            numSelectors: 0, numRows: 4
        )

        let sharedVal = frFromInt(42)
        assignment.setAdvice(column: 0, row: 0, value: sharedVal)
        assignment.setAdvice(column: 0, row: 1, value: frFromInt(10))
        assignment.setAdvice(column: 1, row: 0, value: frFromInt(20))
        assignment.setAdvice(column: 1, row: 1, value: sharedVal)
        // Fill remaining rows to avoid zeros everywhere
        assignment.setAdvice(column: 0, row: 2, value: frFromInt(100))
        assignment.setAdvice(column: 0, row: 3, value: frFromInt(200))
        assignment.setAdvice(column: 1, row: 2, value: frFromInt(300))
        assignment.setAdvice(column: 1, row: 3, value: frFromInt(400))

        // Add copy constraint: advA[0] == advB[1]
        assignment.addCopyConstraint(
            lhsColumn: advA, lhsRow: 0,
            rhsColumn: advB, rhsRow: 1
        )

        let store = engine.extractColumns(cs: cs, assignment: assignment)
        let beta = frFromInt(7)
        let gamma = frFromInt(13)

        let (zPoly, sigma, domain) = engine.computePermutation(
            cs: cs, assignment: assignment, store: store,
            beta: beta, gamma: gamma
        )

        expect(zPoly.count == store.domainSize, "permutation: Z poly has correct length")
        expect(frEqual(zPoly[0], Fr.one), "permutation: Z[0] = 1")
        expect(sigma.count == 2, "permutation: 2 sigma polynomials")

        let permValid = engine.verifyPermutation(
            cs: cs, store: store, zPoly: zPoly, sigma: sigma,
            domain: domain, beta: beta, gamma: gamma
        )
        expect(permValid, "permutation: valid copy constraint verified")
    }

    // ========== Test 5: Vanishing argument ==========
    do {
        let engine = GPUHalo2BackendEngine()

        // Build a simple circuit: s * (a + b - c) = 0
        let cs = Halo2ConstraintSystem()
        let advA = cs.adviceColumn()
        let advB = cs.adviceColumn()
        let advC = cs.adviceColumn()
        let sel = cs.selector()

        cs.createGate("add") { vc in
            let s = vc.querySelector(sel)
            let a = vc.queryAdvice(advA, at: .cur)
            let b = vc.queryAdvice(advB, at: .cur)
            let c = vc.queryAdvice(advC, at: .cur)
            return [s * (a + b - c)]
        }

        // Valid assignment: 3 + 5 = 8
        let assignment = Halo2Assignment(
            numAdvice: 3, numFixed: 0, numInstance: 0,
            numSelectors: 1, numRows: 4
        )
        assignment.setAdvice(column: 0, row: 0, value: frFromInt(3))
        assignment.setAdvice(column: 1, row: 0, value: frFromInt(5))
        assignment.setAdvice(column: 2, row: 0, value: frFromInt(8))
        assignment.setSelector(index: 0, row: 0, value: Fr.one)

        let store = engine.extractColumns(cs: cs, assignment: assignment)

        let alpha = frFromInt(11)
        let beta = frFromInt(7)
        let gamma = frFromInt(13)

        let vanishing = engine.constructVanishingArgument(
            cs: cs, store: store,
            zPoly: nil, sigma: [], domain: [],
            alpha: alpha, beta: beta, gamma: gamma
        )

        expect(vanishing.numGateConstraints == 1, "vanishing: 1 gate constraint")
        expect(vanishing.quotientCoeffs.count == store.domainSize,
               "vanishing: quotient has correct length")

        // For a valid witness, constraint evals should be zero on active rows
        // (row 0 has selector=1 and 3+5-8=0, so constraint is satisfied)
        expect(frEqual(vanishing.constraintEvals[0], Fr.zero),
               "vanishing: constraint eval is zero at row 0 (valid witness)")
    }

    // ========== Test 6: Full prove pipeline ==========
    do {
        let engine = GPUHalo2BackendEngine()

        let cs = Halo2ConstraintSystem()
        let advA = cs.adviceColumn()
        let advB = cs.adviceColumn()
        let advC = cs.adviceColumn()
        let sel = cs.selector()

        cs.createGate("mul") { vc in
            let s = vc.querySelector(sel)
            let a = vc.queryAdvice(advA, at: .cur)
            let b = vc.queryAdvice(advB, at: .cur)
            let c = vc.queryAdvice(advC, at: .cur)
            return [s * (a * b - c)]
        }

        let assignment = Halo2Assignment(
            numAdvice: 3, numFixed: 0, numInstance: 0,
            numSelectors: 1, numRows: 4
        )
        // 3 * 7 = 21
        assignment.setAdvice(column: 0, row: 0, value: frFromInt(3))
        assignment.setAdvice(column: 1, row: 0, value: frFromInt(7))
        assignment.setAdvice(column: 2, row: 0, value: frFromInt(21))
        assignment.setSelector(index: 0, row: 0, value: Fr.one)

        let alpha = frFromInt(17)
        let beta = frFromInt(7)
        let gamma = frFromInt(13)

        let result = engine.prove(
            cs: cs, assignment: assignment,
            challenges: (alpha: alpha, beta: beta, gamma: gamma)
        )

        expect(result.gatesSatisfied, "full prove: gates satisfied")
        expect(result.permutationValid, "full prove: permutation valid (trivial)")
        expect(result.lookupResults.isEmpty, "full prove: no lookups")
        expect(result.vanishing.numGateConstraints == 1, "full prove: 1 gate constraint")
    }

    // ========== Test 7: Expression evaluation with rotations ==========
    do {
        let engine = GPUHalo2BackendEngine()

        // Build gate using next-row rotation: s * (a_cur + a_next - b) = 0
        let cs = Halo2ConstraintSystem()
        let advA = cs.adviceColumn()
        let advB = cs.adviceColumn()
        let sel = cs.selector()

        cs.createGate("accumulate") { vc in
            let s = vc.querySelector(sel)
            let aCur = vc.queryAdvice(advA, at: .cur)
            let aNext = vc.queryAdvice(advA, at: .next)
            let b = vc.queryAdvice(advB, at: .cur)
            return [s * (aCur + aNext - b)]
        }

        // a = [3, 5, ?, ?], b = [8, ?, ?, ?]
        // Row 0: a[0] + a[1] - b[0] = 3 + 5 - 8 = 0 (valid)
        let assignment = Halo2Assignment(
            numAdvice: 2, numFixed: 0, numInstance: 0,
            numSelectors: 1, numRows: 4
        )
        assignment.setAdvice(column: 0, row: 0, value: frFromInt(3))
        assignment.setAdvice(column: 0, row: 1, value: frFromInt(5))
        assignment.setAdvice(column: 1, row: 0, value: frFromInt(8))
        assignment.setSelector(index: 0, row: 0, value: Fr.one)

        let store = engine.extractColumns(cs: cs, assignment: assignment)
        let result = engine.evaluateGates(cs: cs, store: store)

        expect(result.isSatisfied, "rotation: gate with next-row rotation satisfied")
    }
}
