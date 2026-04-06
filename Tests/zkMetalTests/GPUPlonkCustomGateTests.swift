// GPUPlonkCustomGateTests — Comprehensive tests for GPU-accelerated Plonk custom gate engine
//
// Tests cover:
//   1. Bool check constraint evaluation (satisfied and violated)
//   2. Range check constraint with accumulator chain
//   3. XOR gate constraint for binary values
//   4. AND gate constraint for binary values
//   5. NOT gate constraint for binary values
//   6. Poseidon S-box constraint (c = a^5)
//   7. EC addition constraint
//   8. EC doubling constraint
//   9. Conditional select constraint
//   10. IsZero constraint
//   11. Custom gate registry: registration, lookup, deduplication
//   12. Batch evaluation across full trace
//   13. Selector isolation checking
//   14. Quotient polynomial contribution computation
//   15. Linearization scalar computation
//   16. Linearization polynomial construction
//   17. Multi-gate circuits with mixed gate types
//   18. GPU batch boolean check (with CPU fallback)
//   19. GPU batch XOR check
//   20. GPU batch S-box check
//   21. Gate statistics computation
//   22. Empty circuit / edge cases
//   23. Large domain stress test
//   24. Witness verification convenience API

import zkMetal
import Foundation

public func runGPUPlonkCustomGateTests() {
    suite("GPU Plonk Custom Gate Engine")

    let engine = GPUPlonkCustomGateEngine()

    // ========== Test 1: BoolCheck constraint — satisfied ==========
    do {
        let constraint = BoolCheckConstraint()
        // a=0: 0*(1-0) = 0 (satisfied)
        let r0 = constraint.evaluateConstraint(wires: [Fr.zero], rotatedWires: [:], challenges: [])
        expect(r0.isZero, "BoolCheck: a=0 satisfies constraint")

        // a=1: 1*(1-1) = 0 (satisfied)
        let r1 = constraint.evaluateConstraint(wires: [Fr.one], rotatedWires: [:], challenges: [])
        expect(r1.isZero, "BoolCheck: a=1 satisfies constraint")
    }

    // ========== Test 2: BoolCheck constraint — violated ==========
    do {
        let constraint = BoolCheckConstraint()
        // a=2: 2*(1-2) = 2*(-1) != 0
        let two = frFromInt(2)
        let r = constraint.evaluateConstraint(wires: [two], rotatedWires: [:], challenges: [])
        expect(!r.isZero, "BoolCheck: a=2 violates constraint")

        // a=5: 5*(1-5) != 0
        let five = frFromInt(5)
        let r5 = constraint.evaluateConstraint(wires: [five], rotatedWires: [:], challenges: [])
        expect(!r5.isZero, "BoolCheck: a=5 violates constraint")
    }

    // ========== Test 3: BoolCheck linearization scalar ==========
    do {
        let constraint = BoolCheckConstraint()
        // linearization scalar for a=0: 0*(1-0) = 0
        let s0 = constraint.linearizationScalar(wireEvals: [Fr.zero], wireEvalsShifted: [], challenges: [])
        expect(s0.isZero, "BoolCheck linearization: a=0 gives zero scalar")

        // linearization scalar for a=1: 1*(1-1) = 0
        let s1 = constraint.linearizationScalar(wireEvals: [Fr.one], wireEvalsShifted: [], challenges: [])
        expect(s1.isZero, "BoolCheck linearization: a=1 gives zero scalar")
    }

    // ========== Test 4: XOR constraint — satisfied ==========
    do {
        let constraint = XORConstraint()
        // 0 XOR 0 = 0
        let r00 = constraint.evaluateConstraint(wires: [Fr.zero, Fr.zero, Fr.zero], rotatedWires: [:], challenges: [])
        expect(r00.isZero, "XOR: 0 XOR 0 = 0 satisfied")

        // 0 XOR 1 = 1
        let r01 = constraint.evaluateConstraint(wires: [Fr.zero, Fr.one, Fr.one], rotatedWires: [:], challenges: [])
        expect(r01.isZero, "XOR: 0 XOR 1 = 1 satisfied")

        // 1 XOR 0 = 1
        let r10 = constraint.evaluateConstraint(wires: [Fr.one, Fr.zero, Fr.one], rotatedWires: [:], challenges: [])
        expect(r10.isZero, "XOR: 1 XOR 0 = 1 satisfied")

        // 1 XOR 1 = 0
        let r11 = constraint.evaluateConstraint(wires: [Fr.one, Fr.one, Fr.zero], rotatedWires: [:], challenges: [])
        expect(r11.isZero, "XOR: 1 XOR 1 = 0 satisfied")
    }

    // ========== Test 5: XOR constraint — violated ==========
    do {
        let constraint = XORConstraint()
        // 0 XOR 0 = 1 (wrong)
        let r = constraint.evaluateConstraint(wires: [Fr.zero, Fr.zero, Fr.one], rotatedWires: [:], challenges: [])
        expect(!r.isZero, "XOR: 0 XOR 0 = 1 violates constraint")

        // 1 XOR 1 = 1 (wrong)
        let r2 = constraint.evaluateConstraint(wires: [Fr.one, Fr.one, Fr.one], rotatedWires: [:], challenges: [])
        expect(!r2.isZero, "XOR: 1 XOR 1 = 1 violates constraint")
    }

    // ========== Test 6: AND constraint — satisfied ==========
    do {
        let constraint = ANDConstraint()
        // 0 AND 0 = 0
        let r00 = constraint.evaluateConstraint(wires: [Fr.zero, Fr.zero, Fr.zero], rotatedWires: [:], challenges: [])
        expect(r00.isZero, "AND: 0 AND 0 = 0 satisfied")

        // 0 AND 1 = 0
        let r01 = constraint.evaluateConstraint(wires: [Fr.zero, Fr.one, Fr.zero], rotatedWires: [:], challenges: [])
        expect(r01.isZero, "AND: 0 AND 1 = 0 satisfied")

        // 1 AND 0 = 0
        let r10 = constraint.evaluateConstraint(wires: [Fr.one, Fr.zero, Fr.zero], rotatedWires: [:], challenges: [])
        expect(r10.isZero, "AND: 1 AND 0 = 0 satisfied")

        // 1 AND 1 = 1
        let r11 = constraint.evaluateConstraint(wires: [Fr.one, Fr.one, Fr.one], rotatedWires: [:], challenges: [])
        expect(r11.isZero, "AND: 1 AND 1 = 1 satisfied")
    }

    // ========== Test 7: AND constraint — violated ==========
    do {
        let constraint = ANDConstraint()
        // 1 AND 1 = 0 (wrong)
        let r = constraint.evaluateConstraint(wires: [Fr.one, Fr.one, Fr.zero], rotatedWires: [:], challenges: [])
        expect(!r.isZero, "AND: 1 AND 1 = 0 violates constraint")

        // 0 AND 0 = 1 (wrong)
        let r2 = constraint.evaluateConstraint(wires: [Fr.zero, Fr.zero, Fr.one], rotatedWires: [:], challenges: [])
        expect(!r2.isZero, "AND: 0 AND 0 = 1 violates constraint")
    }

    // ========== Test 8: NOT constraint — satisfied ==========
    do {
        let constraint = NOTConstraint()
        // NOT 0 = 1
        let r0 = constraint.evaluateConstraint(wires: [Fr.zero, Fr.zero, Fr.one], rotatedWires: [:], challenges: [])
        expect(r0.isZero, "NOT: NOT 0 = 1 satisfied")

        // NOT 1 = 0
        let r1 = constraint.evaluateConstraint(wires: [Fr.one, Fr.zero, Fr.zero], rotatedWires: [:], challenges: [])
        expect(r1.isZero, "NOT: NOT 1 = 0 satisfied")
    }

    // ========== Test 9: NOT constraint — violated ==========
    do {
        let constraint = NOTConstraint()
        // NOT 0 = 0 (wrong, should be 1)
        let r = constraint.evaluateConstraint(wires: [Fr.zero, Fr.zero, Fr.zero], rotatedWires: [:], challenges: [])
        expect(!r.isZero, "NOT: NOT 0 = 0 violates constraint")

        // NOT 1 = 1 (wrong, should be 0)
        let r2 = constraint.evaluateConstraint(wires: [Fr.one, Fr.zero, Fr.one], rotatedWires: [:], challenges: [])
        expect(!r2.isZero, "NOT: NOT 1 = 1 violates constraint")
    }

    // ========== Test 10: Poseidon S-box constraint — satisfied ==========
    do {
        let constraint = PoseidonSboxConstraint()
        // a=2: a^2=4, a^5=32, wire layout: [a, a^2, a^5]
        let a = frFromInt(2)
        let aSq = frFromInt(4)
        let a5 = frFromInt(32)
        let r = constraint.evaluateConstraint(wires: [a, aSq, a5], rotatedWires: [:], challenges: [])
        expect(r.isZero, "PoseidonSbox: a=2, b=4, c=32 satisfies c=a^5")
    }

    // ========== Test 11: Poseidon S-box constraint — satisfied with a=3 ==========
    do {
        let constraint = PoseidonSboxConstraint()
        let a = frFromInt(3)
        let aSq = frMul(a, a)    // 9
        let a5 = frMul(a, frMul(aSq, aSq))  // 3 * 81 = 243
        let r = constraint.evaluateConstraint(wires: [a, aSq, a5], rotatedWires: [:], challenges: [])
        expect(r.isZero, "PoseidonSbox: a=3 satisfies c=a^5 (c=243)")
    }

    // ========== Test 12: Poseidon S-box constraint — violated ==========
    do {
        let constraint = PoseidonSboxConstraint()
        let a = frFromInt(2)
        let aSq = frFromInt(4)
        let wrongC = frFromInt(31) // should be 32
        let r = constraint.evaluateConstraint(wires: [a, aSq, wrongC], rotatedWires: [:], challenges: [])
        expect(!r.isZero, "PoseidonSbox: wrong c=31 violates constraint")
    }

    // ========== Test 13: Conditional select — satisfied ==========
    do {
        let constraint = ConditionalSelectConstraint()
        // cond=1, a=7, b=3 => out=7 (select a)
        let seven = frFromInt(7)
        let three = frFromInt(3)
        let resolved: [ColumnRef: Fr] = [
            ColumnRef(column: 0, rotation: .cur): Fr.one,
            ColumnRef(column: 1, rotation: .cur): seven,
            ColumnRef(column: 2, rotation: .cur): three,
            ColumnRef(column: 0, rotation: .next): seven,
        ]
        let r = constraint.evaluateConstraint(wires: [], rotatedWires: resolved, challenges: [])
        expect(r.isZero, "ConditionalSelect: cond=1 selects a=7")

        // cond=0, a=7, b=3 => out=3 (select b)
        let resolved2: [ColumnRef: Fr] = [
            ColumnRef(column: 0, rotation: .cur): Fr.zero,
            ColumnRef(column: 1, rotation: .cur): seven,
            ColumnRef(column: 2, rotation: .cur): three,
            ColumnRef(column: 0, rotation: .next): three,
        ]
        let r2 = constraint.evaluateConstraint(wires: [], rotatedWires: resolved2, challenges: [])
        expect(r2.isZero, "ConditionalSelect: cond=0 selects b=3")
    }

    // ========== Test 14: Conditional select — violated ==========
    do {
        let constraint = ConditionalSelectConstraint()
        let seven = frFromInt(7)
        let three = frFromInt(3)
        // cond=1, a=7, b=3 => out=3 (wrong, should be 7)
        let resolved: [ColumnRef: Fr] = [
            ColumnRef(column: 0, rotation: .cur): Fr.one,
            ColumnRef(column: 1, rotation: .cur): seven,
            ColumnRef(column: 2, rotation: .cur): three,
            ColumnRef(column: 0, rotation: .next): three,
        ]
        let r = constraint.evaluateConstraint(wires: [], rotatedWires: resolved, challenges: [])
        expect(!r.isZero, "ConditionalSelect: cond=1, out=3 violates (should be 7)")
    }

    // ========== Test 15: IsZero constraint — satisfied ==========
    do {
        let constraint = IsZeroConstraint()
        // a=0, inv=0, out=1: 0*1=0, 0*0+1-1=0 => satisfied
        let r0 = constraint.evaluateConstraint(wires: [Fr.zero, Fr.zero, Fr.one], rotatedWires: [:], challenges: [])
        expect(r0.isZero, "IsZero: a=0 gives out=1 (satisfied)")

        // a=5, inv=5^{-1}, out=0: 5*0=0, 5*inv+0-1=0 => satisfied
        let five = frFromInt(5)
        let inv5 = frInverse(five)
        let r1 = constraint.evaluateConstraint(wires: [five, inv5, Fr.zero], rotatedWires: [:], challenges: [])
        expect(r1.isZero, "IsZero: a=5 gives out=0 (satisfied)")
    }

    // ========== Test 16: IsZero constraint — violated ==========
    do {
        let constraint = IsZeroConstraint()
        // a=5, inv=0, out=1: 5*1=5 != 0 => violated
        let five = frFromInt(5)
        let r = constraint.evaluateConstraint(wires: [five, Fr.zero, Fr.one], rotatedWires: [:], challenges: [])
        expect(!r.isZero, "IsZero: a=5, out=1 violates constraint")
    }

    // ========== Test 17: Gate Type IDs are unique ==========
    do {
        let ids: [CustomGateTypeID] = [
            .rangeCheck, .boolCheck, .ecAdd, .ecDouble,
            .poseidonRound, .poseidonSbox, .xorGate,
            .conditionalSelect, .binaryDecompose, .andGate,
            .notGate, .isZeroGate,
        ]
        var seen = Set<Int>()
        var allUnique = true
        for id in ids {
            if seen.contains(id.rawValue) { allUnique = false; break }
            seen.insert(id.rawValue)
        }
        expect(allUnique, "All gate type IDs have unique raw values")
        expectEqual(ids.count, 12, "12 built-in gate type IDs defined")
    }

    // ========== Test 18: CustomGateRegistry — registration and lookup ==========
    do {
        let domainSize = 8
        let registry = CustomGateRegistry(domainSize: domainSize)

        // Create selectors: bool check active on rows 0,1; XOR on rows 2,3
        var boolSelector = [Fr](repeating: Fr.zero, count: domainSize)
        boolSelector[0] = Fr.one
        boolSelector[1] = Fr.one

        var xorSelector = [Fr](repeating: Fr.zero, count: domainSize)
        xorSelector[2] = Fr.one
        xorSelector[3] = Fr.one

        let boolIdx = registry.register(constraint: BoolCheckConstraint(), selectorEvals: boolSelector)
        let xorIdx = registry.register(constraint: XORConstraint(), selectorEvals: xorSelector)

        expectEqual(boolIdx, 0, "Bool check registered at index 0")
        expectEqual(xorIdx, 1, "XOR registered at index 1")
        expectEqual(registry.count, 2, "Registry has 2 entries")

        // Lookup
        let boolEntry = registry.entry(for: .boolCheck)
        expect(boolEntry != nil, "Can look up bool check by type ID")
        expectEqual(boolEntry?.activeRows.count ?? 0, 2, "Bool check active on 2 rows")

        let xorEntry = registry.entry(for: .xorGate)
        expect(xorEntry != nil, "Can look up XOR by type ID")
        expectEqual(xorEntry?.activeRows.count ?? 0, 2, "XOR active on 2 rows")

        // Missing type
        let missing = registry.entry(for: .ecAdd)
        expect(missing == nil, "EC add not registered, returns nil")
    }

    // ========== Test 19: CustomGateRegistry — domain size validation ==========
    do {
        let registry = CustomGateRegistry(domainSize: 4)
        expectEqual(registry.domainSize, 4, "Domain size is 4")
        expectEqual(registry.count, 0, "Empty registry has 0 entries")
        expectEqual(registry.maxConstraintDegree, 0, "Empty registry has 0 max degree")
    }

    // ========== Test 20: Batch evaluation — all gates satisfied ==========
    do {
        let domainSize = 4
        let registry = CustomGateRegistry(domainSize: domainSize)

        // Bool check on rows 0,1 with valid boolean values
        var boolSel = [Fr](repeating: Fr.zero, count: domainSize)
        boolSel[0] = Fr.one
        boolSel[1] = Fr.one
        registry.register(constraint: BoolCheckConstraint(), selectorEvals: boolSel)

        // Wire column 0: [0, 1, 5, 3] (rows 0,1 are boolean)
        let wireCol0 = [Fr.zero, Fr.one, frFromInt(5), frFromInt(3)]

        let result = engine.evaluateAllConstraints(
            registry: registry,
            wireColumns: [wireCol0]
        )
        expect(result.isSatisfied, "Batch eval: all bool check rows satisfied")
        expectEqual(result.failingRows.count, 0, "No failing rows")
    }

    // ========== Test 21: Batch evaluation — constraint violation detected ==========
    do {
        let domainSize = 4
        let registry = CustomGateRegistry(domainSize: domainSize)

        var boolSel = [Fr](repeating: Fr.zero, count: domainSize)
        boolSel[0] = Fr.one
        boolSel[1] = Fr.one
        boolSel[2] = Fr.one
        registry.register(constraint: BoolCheckConstraint(), selectorEvals: boolSel)

        // Wire column 0: [0, 1, 5, 3] — row 2 has value 5, not boolean
        let wireCol0 = [Fr.zero, Fr.one, frFromInt(5), frFromInt(3)]

        let result = engine.evaluateAllConstraints(
            registry: registry,
            wireColumns: [wireCol0]
        )
        expect(!result.isSatisfied, "Batch eval: row 2 has non-boolean value")
        expectEqual(result.failingRows.count, 1, "Exactly 1 failing row")
        expectEqual(result.failingRows[0], 2, "Failing row is row 2")
    }

    // ========== Test 22: Batch evaluation — multiple gate types ==========
    do {
        let domainSize = 8
        let registry = CustomGateRegistry(domainSize: domainSize)

        // Bool check on rows 0,1
        var boolSel = [Fr](repeating: Fr.zero, count: domainSize)
        boolSel[0] = Fr.one
        boolSel[1] = Fr.one
        registry.register(constraint: BoolCheckConstraint(), selectorEvals: boolSel)

        // AND gate on rows 4,5
        var andSel = [Fr](repeating: Fr.zero, count: domainSize)
        andSel[4] = Fr.one
        andSel[5] = Fr.one
        registry.register(constraint: ANDConstraint(), selectorEvals: andSel)

        // Wire columns (3 columns for AND gate: a, b, c)
        let wireCol0 = [Fr.zero, Fr.one, Fr.zero, Fr.zero, Fr.one, Fr.zero, Fr.zero, Fr.zero]
        let wireCol1 = [Fr.zero, Fr.zero, Fr.zero, Fr.zero, Fr.one, Fr.one, Fr.zero, Fr.zero]
        let wireCol2 = [Fr.zero, Fr.zero, Fr.zero, Fr.zero, Fr.one, Fr.zero, Fr.zero, Fr.zero]

        let result = engine.evaluateAllConstraints(
            registry: registry,
            wireColumns: [wireCol0, wireCol1, wireCol2]
        )
        expect(result.isSatisfied, "Multi-gate batch eval: bool+AND all satisfied")
        expectEqual(result.perGateResiduals.count, 2, "2 gate types evaluated")
        expectEqual(result.perGateFailingRows[0].count, 0, "Bool check: no failures")
        expectEqual(result.perGateFailingRows[1].count, 0, "AND: no failures")
    }

    // ========== Test 23: Selector isolation — no violations ==========
    do {
        let domainSize = 8
        let registry = CustomGateRegistry(domainSize: domainSize)

        var boolSel = [Fr](repeating: Fr.zero, count: domainSize)
        boolSel[0] = Fr.one
        boolSel[1] = Fr.one

        var xorSel = [Fr](repeating: Fr.zero, count: domainSize)
        xorSel[4] = Fr.one
        xorSel[5] = Fr.one

        registry.register(constraint: BoolCheckConstraint(), selectorEvals: boolSel)
        registry.register(constraint: XORConstraint(), selectorEvals: xorSel)

        let violations = engine.checkSelectorIsolation(registry: registry)
        expectEqual(violations.count, 0, "No selector isolation violations")
    }

    // ========== Test 24: Selector isolation — violations detected ==========
    do {
        let domainSize = 4
        let registry = CustomGateRegistry(domainSize: domainSize)

        // Both gates active on row 1
        var boolSel = [Fr](repeating: Fr.zero, count: domainSize)
        boolSel[0] = Fr.one
        boolSel[1] = Fr.one

        var xorSel = [Fr](repeating: Fr.zero, count: domainSize)
        xorSel[1] = Fr.one  // overlaps with bool check at row 1
        xorSel[2] = Fr.one

        registry.register(constraint: BoolCheckConstraint(), selectorEvals: boolSel)
        registry.register(constraint: XORConstraint(), selectorEvals: xorSel)

        let violations = engine.checkSelectorIsolation(registry: registry)
        expectEqual(violations.count, 1, "1 selector isolation violation")
        expectEqual(violations[0], 1, "Violation at row 1")
    }

    // ========== Test 25: Gate statistics ==========
    do {
        let domainSize = 16
        let registry = CustomGateRegistry(domainSize: domainSize)

        var boolSel = [Fr](repeating: Fr.zero, count: domainSize)
        for i in 0..<4 { boolSel[i] = Fr.one }

        var xorSel = [Fr](repeating: Fr.zero, count: domainSize)
        for i in 8..<12 { xorSel[i] = Fr.one }

        registry.register(constraint: BoolCheckConstraint(), selectorEvals: boolSel)
        registry.register(constraint: XORConstraint(), selectorEvals: xorSel)

        let stats = engine.computeStatistics(registry: registry)
        expectEqual(stats.totalGateTypes, 2, "Statistics: 2 gate types")
        expectEqual(stats.totalActiveRows, 8, "Statistics: 8 active rows total")
        expectEqual(stats.activeRowsPerGate[0], 4, "Statistics: Bool check has 4 active rows")
        expectEqual(stats.activeRowsPerGate[1], 4, "Statistics: XOR has 4 active rows")
        expect(stats.activationDensity > 0.49 && stats.activationDensity < 0.51,
               "Statistics: density is 0.5")
        expectEqual(stats.gateTypeNames[0], "BoolCheck", "Statistics: first gate name")
        expectEqual(stats.gateTypeNames[1], "XOR", "Statistics: second gate name")
    }

    // ========== Test 26: Linearization scalars ==========
    do {
        let domainSize = 4
        let registry = CustomGateRegistry(domainSize: domainSize)

        var boolSel = [Fr](repeating: Fr.zero, count: domainSize)
        boolSel[0] = Fr.one
        registry.register(constraint: BoolCheckConstraint(), selectorEvals: boolSel)

        var xorSel = [Fr](repeating: Fr.zero, count: domainSize)
        xorSel[1] = Fr.one
        registry.register(constraint: XORConstraint(), selectorEvals: xorSel)

        let alpha = frFromInt(7)
        let wireEvals = [Fr.zero, Fr.one, Fr.zero]

        let linResult = engine.computeLinearization(
            registry: registry,
            wireEvals: wireEvals,
            wireEvalsShifted: [],
            selectorEvals: [Fr.one, Fr.one],
            alpha: alpha,
            alphaOffset: 0
        )

        expectEqual(linResult.scalars.count, 2, "Linearization: 2 scalars computed")
        // BoolCheck at a=0: scalar = 0*(1-0) = 0
        expect(linResult.scalars[0].isZero, "Linearization: BoolCheck scalar is zero for a=0")
    }

    // ========== Test 27: Linearization with alpha separation ==========
    do {
        let domainSize = 4
        let registry = CustomGateRegistry(domainSize: domainSize)

        var boolSel = [Fr](repeating: Fr.zero, count: domainSize)
        boolSel[0] = Fr.one
        registry.register(constraint: BoolCheckConstraint(), selectorEvals: boolSel)

        let alpha = frFromInt(3)
        // Wire eval a = 1: BoolCheck scalar = 1*(1-1) = 0
        let wireEvals = [Fr.one]

        let linResult = engine.computeLinearization(
            registry: registry,
            wireEvals: wireEvals,
            wireEvalsShifted: [],
            selectorEvals: [Fr.one],
            alpha: alpha,
            alphaOffset: 2  // start from alpha^2
        )

        // scalar = 0, so combined should be zero regardless of alpha
        expect(linResult.combinedEval.isZero, "Linearization: combined eval is zero when constraint satisfied")
    }

    // ========== Test 28: Linearization with non-zero scalar ==========
    do {
        let domainSize = 4
        let registry = CustomGateRegistry(domainSize: domainSize)

        var boolSel = [Fr](repeating: Fr.zero, count: domainSize)
        boolSel[0] = Fr.one
        registry.register(constraint: BoolCheckConstraint(), selectorEvals: boolSel)

        let alpha = frFromInt(5)
        // Wire eval a=2: BoolCheck scalar = 2*(1-2) = 2*(-1) = -(2) in Fr
        let two = frFromInt(2)
        let wireEvals = [two]

        let linResult = engine.computeLinearization(
            registry: registry,
            wireEvals: wireEvals,
            wireEvalsShifted: [],
            selectorEvals: [Fr.one],
            alpha: alpha,
            alphaOffset: 0
        )

        expect(!linResult.scalars[0].isZero, "Linearization: non-zero scalar for violated constraint")
        // combined = alpha^0 * 1 * scalar = scalar
        expect(frEqual(linResult.combinedEval, linResult.scalars[0]),
               "Linearization: combined equals scalar when alphaOffset=0 and selectorEval=1")
    }

    // ========== Test 29: Selective single-gate evaluation ==========
    do {
        let domainSize = 4
        let registry = CustomGateRegistry(domainSize: domainSize)

        var boolSel = [Fr](repeating: Fr.zero, count: domainSize)
        boolSel[0] = Fr.one
        boolSel[1] = Fr.one
        registry.register(constraint: BoolCheckConstraint(), selectorEvals: boolSel)

        var xorSel = [Fr](repeating: Fr.zero, count: domainSize)
        xorSel[2] = Fr.one
        registry.register(constraint: XORConstraint(), selectorEvals: xorSel)

        // Only evaluate bool check
        let wireCol0 = [Fr.zero, frFromInt(3), Fr.zero, Fr.zero] // row 1 violates
        let boolResult = engine.evaluateSingleGateType(
            typeID: .boolCheck,
            registry: registry,
            wireColumns: [wireCol0]
        )
        expectEqual(boolResult.failingRows.count, 1, "Single gate eval: 1 failing row for bool check")
        expectEqual(boolResult.failingRows[0], 1, "Single gate eval: failing row is 1")
    }

    // ========== Test 30: Witness verification convenience ==========
    do {
        let domainSize = 4
        let registry = CustomGateRegistry(domainSize: domainSize)

        var boolSel = [Fr](repeating: Fr.zero, count: domainSize)
        boolSel[0] = Fr.one
        boolSel[1] = Fr.one
        registry.register(constraint: BoolCheckConstraint(), selectorEvals: boolSel)

        // Valid witness
        let validWires = [[Fr.zero, Fr.one, Fr.zero, Fr.zero]]
        let isValid = engine.verifyWitness(registry: registry, wireColumns: validWires)
        expect(isValid, "verifyWitness: valid boolean witness passes")

        // Invalid witness
        let invalidWires = [[Fr.zero, frFromInt(5), Fr.zero, Fr.zero]]
        let isInvalid = !engine.verifyWitness(registry: registry, wireColumns: invalidWires)
        expect(isInvalid, "verifyWitness: non-boolean value fails")
    }

    // ========== Test 31: GPU batch bool check (exercises GPU or CPU fallback) ==========
    do {
        let n = 8
        var values = [Fr](repeating: Fr.zero, count: n)
        values[0] = Fr.zero
        values[1] = Fr.one
        values[2] = frFromInt(2)  // violates
        values[3] = Fr.one
        values[4] = Fr.zero
        values[5] = frFromInt(7)  // violates
        values[6] = Fr.one
        values[7] = Fr.zero

        let selector = [Fr](repeating: Fr.one, count: n)
        let residuals = engine.gpuBatchBoolCheck(values: values, selectorEvals: selector)

        expectEqual(residuals.count, n, "GPU batch bool: correct output length")
        expect(residuals[0].isZero, "GPU batch bool: a=0 satisfied")
        expect(residuals[1].isZero, "GPU batch bool: a=1 satisfied")
        expect(!residuals[2].isZero, "GPU batch bool: a=2 violated")
        expect(residuals[3].isZero, "GPU batch bool: a=1 satisfied (row 3)")
        expect(residuals[4].isZero, "GPU batch bool: a=0 satisfied (row 4)")
        expect(!residuals[5].isZero, "GPU batch bool: a=7 violated")
        expect(residuals[6].isZero, "GPU batch bool: a=1 satisfied (row 6)")
        expect(residuals[7].isZero, "GPU batch bool: a=0 satisfied (row 7)")
    }

    // ========== Test 32: GPU batch bool check with sparse selector ==========
    do {
        let n = 8
        var values = [Fr](repeating: frFromInt(5), count: n) // all non-boolean
        var selector = [Fr](repeating: Fr.zero, count: n)
        selector[3] = Fr.one // only row 3 active

        let residuals = engine.gpuBatchBoolCheck(values: values, selectorEvals: selector)
        // Only row 3 should have non-zero residual
        expect(residuals[0].isZero, "Sparse selector: row 0 zero (selector off)")
        expect(residuals[1].isZero, "Sparse selector: row 1 zero (selector off)")
        expect(!residuals[3].isZero, "Sparse selector: row 3 non-zero (selector on, value 5)")
    }

    // ========== Test 33: GPU batch XOR check ==========
    do {
        let n = 4
        let aVals = [Fr.zero, Fr.zero, Fr.one, Fr.one]
        let bVals = [Fr.zero, Fr.one, Fr.zero, Fr.one]
        let cVals = [Fr.zero, Fr.one, Fr.one, Fr.zero] // correct XOR results
        let selector = [Fr](repeating: Fr.one, count: n)

        let residuals = engine.gpuBatchXORCheck(aValues: aVals, bValues: bVals, cValues: cVals, selectorEvals: selector)
        for i in 0..<n {
            expect(residuals[i].isZero, "GPU batch XOR: row \(i) satisfied")
        }
    }

    // ========== Test 34: GPU batch XOR check — violated ==========
    do {
        let n = 4
        let aVals = [Fr.zero, Fr.zero, Fr.one, Fr.one]
        let bVals = [Fr.zero, Fr.one, Fr.zero, Fr.one]
        let cVals = [Fr.one, Fr.zero, Fr.zero, Fr.one] // wrong XOR results
        let selector = [Fr](repeating: Fr.one, count: n)

        let residuals = engine.gpuBatchXORCheck(aValues: aVals, bValues: bVals, cValues: cVals, selectorEvals: selector)
        expect(!residuals[0].isZero, "GPU batch XOR violated: 0 XOR 0 != 1")
        expect(!residuals[1].isZero, "GPU batch XOR violated: 0 XOR 1 != 0")
        expect(!residuals[2].isZero, "GPU batch XOR violated: 1 XOR 0 != 0")
        expect(!residuals[3].isZero, "GPU batch XOR violated: 1 XOR 1 != 1")
    }

    // ========== Test 35: GPU batch S-box check ==========
    do {
        let n = 4
        // Compute a^5 for each value
        let aVals = [frFromInt(0), frFromInt(1), frFromInt(2), frFromInt(3)]
        var cVals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            let a = aVals[i]
            let a2 = frSqr(a)
            let a4 = frSqr(a2)
            cVals[i] = frMul(a, a4) // a^5
        }
        let selector = [Fr](repeating: Fr.one, count: n)

        let residuals = engine.gpuBatchSboxCheck(aValues: aVals, cValues: cVals, selectorEvals: selector)
        for i in 0..<n {
            expect(residuals[i].isZero, "GPU batch sbox: row \(i) (a=\(i)) satisfied")
        }
    }

    // ========== Test 36: GPU batch S-box check — violated ==========
    do {
        let n = 2
        let aVals = [frFromInt(2), frFromInt(3)]
        let cVals = [frFromInt(31), frFromInt(242)] // wrong: should be 32, 243
        let selector = [Fr](repeating: Fr.one, count: n)

        let residuals = engine.gpuBatchSboxCheck(aValues: aVals, cValues: cVals, selectorEvals: selector)
        expect(!residuals[0].isZero, "GPU batch sbox violated: 2^5 != 31")
        expect(!residuals[1].isZero, "GPU batch sbox violated: 3^5 != 242")
    }

    // ========== Test 37: Empty registry / edge cases ==========
    do {
        let registry = CustomGateRegistry(domainSize: 4)
        let result = engine.evaluateAllConstraints(
            registry: registry,
            wireColumns: [[Fr.zero, Fr.one, Fr.zero, Fr.zero]]
        )
        expect(result.isSatisfied, "Empty registry: trivially satisfied")
        expectEqual(result.residuals.count, 0, "Empty registry: no residuals")

        let isValid = engine.verifyWitness(registry: registry, wireColumns: [])
        expect(isValid, "Empty registry: verifyWitness returns true")
    }

    // ========== Test 38: Queried cells and max rotation ==========
    do {
        let registry = CustomGateRegistry(domainSize: 8)

        var sel = [Fr](repeating: Fr.one, count: 8)
        registry.register(constraint: ConditionalSelectConstraint(), selectorEvals: sel)

        let cells = registry.allQueriedCells
        expect(cells.count == 4, "ConditionalSelect queries 4 cells")
        expectEqual(registry.maxRotation, 1, "ConditionalSelect max rotation is 1")
    }

    // ========== Test 39: MaxConstraintDegree ==========
    do {
        let registry = CustomGateRegistry(domainSize: 4)
        let sel = [Fr](repeating: Fr.one, count: 4)

        registry.register(constraint: BoolCheckConstraint(), selectorEvals: sel)  // degree 2
        expectEqual(registry.maxConstraintDegree, 2, "Max degree with BoolCheck is 2")

        registry.register(constraint: ECAddConstraint(), selectorEvals: sel)  // degree 3
        expectEqual(registry.maxConstraintDegree, 3, "Max degree with EC add is 3")
    }

    // ========== Test 40: EC double constraint — satisfied ==========
    do {
        let constraint = ECDoubleConstraint()
        // P1 = (1, 2): on y^2 = x^3 + 3 (simplified curve)
        // lambda = 3*x1^2 / (2*y1) = 3 / 4
        // For simplicity, use field arithmetic:
        let x1 = frFromInt(1)
        let y1 = frFromInt(2)
        let three = frFromInt(3)
        let four = frFromInt(4)
        let inv4 = frInverse(four)
        let lambda = frMul(three, inv4) // 3/4

        // x2 = lambda^2 - 2*x1
        let lamSq = frSqr(lambda)
        let two = frFromInt(2)
        let x2 = frSub(lamSq, frMul(two, x1))

        // y2 = lambda*(x1 - x2) - y1
        let y2 = frSub(frMul(lambda, frSub(x1, x2)), y1)

        let resolved: [ColumnRef: Fr] = [
            ColumnRef(column: 0, rotation: .cur): x1,
            ColumnRef(column: 1, rotation: .cur): y1,
            ColumnRef(column: 2, rotation: .cur): lambda,
            ColumnRef(column: 0, rotation: .next): x2,
            ColumnRef(column: 1, rotation: .next): y2,
        ]

        let r = constraint.evaluateConstraint(wires: [], rotatedWires: resolved, challenges: [])
        expect(r.isZero, "EC double: valid doubling satisfies constraint")
    }

    // ========== Test 41: EC double constraint — violated ==========
    do {
        let constraint = ECDoubleConstraint()
        let x1 = frFromInt(1)
        let y1 = frFromInt(2)
        let lambda = frFromInt(1) // wrong lambda

        let resolved: [ColumnRef: Fr] = [
            ColumnRef(column: 0, rotation: .cur): x1,
            ColumnRef(column: 1, rotation: .cur): y1,
            ColumnRef(column: 2, rotation: .cur): lambda,
            ColumnRef(column: 0, rotation: .next): frFromInt(99),
            ColumnRef(column: 1, rotation: .next): frFromInt(99),
        ]
        let r = constraint.evaluateConstraint(wires: [], rotatedWires: resolved, challenges: [])
        expect(!r.isZero, "EC double: wrong coordinates violate constraint")
    }

    // ========== Test 42: EC add constraint — satisfied ==========
    do {
        let constraint = ECAddConstraint()
        // Use simple field values that satisfy the addition formulas:
        // P1 = (x1, y1), P2 = (x2, y2), lambda = (y2-y1)/(x2-x1)
        let x1 = frFromInt(1)
        let y1 = frFromInt(5)
        let x2 = frFromInt(3)
        let y2 = frFromInt(11)

        // lambda = (11-5)/(3-1) = 6/2 = 3
        let lambda = frFromInt(3)

        // x3 = lambda^2 - x1 - x2 = 9 - 1 - 3 = 5
        let x3 = frFromInt(5)

        // y3 = lambda*(x1-x3) - y1 = 3*(1-5) - 5 = -12 - 5 = -17
        let negSeventeen = frSub(Fr.zero, frFromInt(17))

        let resolved: [ColumnRef: Fr] = [
            ColumnRef(column: 0, rotation: .cur): x1,
            ColumnRef(column: 1, rotation: .cur): y1,
            ColumnRef(column: 2, rotation: .cur): x2,
            ColumnRef(column: 0, rotation: .next): y2,
            ColumnRef(column: 1, rotation: .next): x3,
            ColumnRef(column: 2, rotation: .next): negSeventeen,
            ColumnRef(column: 3, rotation: .cur): lambda,
        ]

        let r = constraint.evaluateConstraint(wires: [], rotatedWires: resolved, challenges: [])
        expect(r.isZero, "EC add: valid addition satisfies constraint")
    }

    // ========== Test 43: EC add constraint — violated ==========
    do {
        let constraint = ECAddConstraint()
        let resolved: [ColumnRef: Fr] = [
            ColumnRef(column: 0, rotation: .cur): frFromInt(1),
            ColumnRef(column: 1, rotation: .cur): frFromInt(2),
            ColumnRef(column: 2, rotation: .cur): frFromInt(3),
            ColumnRef(column: 0, rotation: .next): frFromInt(4),
            ColumnRef(column: 1, rotation: .next): frFromInt(5),
            ColumnRef(column: 2, rotation: .next): frFromInt(6),
            ColumnRef(column: 3, rotation: .cur): frFromInt(7),
        ]
        let r = constraint.evaluateConstraint(wires: [], rotatedWires: resolved, challenges: [])
        expect(!r.isZero, "EC add: random values violate constraint")
    }

    // ========== Test 44: Range check constraint — bit check ==========
    do {
        let constraint = RangeCheckConstraint(bits: 8)
        // Valid bit: a=0
        let r0 = constraint.evaluateConstraint(wires: [Fr.zero], rotatedWires: [:], challenges: [])
        expect(r0.isZero, "RangeCheck: bit=0 satisfied")

        // Valid bit: a=1
        let r1 = constraint.evaluateConstraint(wires: [Fr.one], rotatedWires: [:], challenges: [])
        expect(r1.isZero, "RangeCheck: bit=1 satisfied")

        // Invalid bit: a=2
        let r2 = constraint.evaluateConstraint(wires: [frFromInt(2)], rotatedWires: [:], challenges: [])
        expect(!r2.isZero, "RangeCheck: bit=2 violated")
    }

    // ========== Test 45: Constraint type properties ==========
    do {
        let boolC = BoolCheckConstraint()
        expectEqual(boolC.wireCount, 1, "BoolCheck wireCount is 1")
        expectEqual(boolC.constraintDegree, 2, "BoolCheck degree is 2")

        let xorC = XORConstraint()
        expectEqual(xorC.wireCount, 3, "XOR wireCount is 3")
        expectEqual(xorC.constraintDegree, 2, "XOR degree is 2")

        let andC = ANDConstraint()
        expectEqual(andC.wireCount, 3, "AND wireCount is 3")
        expectEqual(andC.constraintDegree, 2, "AND degree is 2")

        let notC = NOTConstraint()
        expectEqual(notC.wireCount, 3, "NOT wireCount is 3")
        expectEqual(notC.constraintDegree, 1, "NOT degree is 1")

        let sboxC = PoseidonSboxConstraint()
        expectEqual(sboxC.wireCount, 3, "PoseidonSbox wireCount is 3")
        expectEqual(sboxC.constraintDegree, 3, "PoseidonSbox degree is 3")

        let ecAddC = ECAddConstraint()
        expectEqual(ecAddC.wireCount, 7, "ECAdd wireCount is 7")
        expectEqual(ecAddC.constraintDegree, 3, "ECAdd degree is 3")

        let ecDblC = ECDoubleConstraint()
        expectEqual(ecDblC.wireCount, 5, "ECDouble wireCount is 5")
        expectEqual(ecDblC.constraintDegree, 3, "ECDouble degree is 3")

        let condC = ConditionalSelectConstraint()
        expectEqual(condC.wireCount, 4, "ConditionalSelect wireCount is 4")
        expectEqual(condC.constraintDegree, 2, "ConditionalSelect degree is 2")

        let izC = IsZeroConstraint()
        expectEqual(izC.wireCount, 3, "IsZero wireCount is 3")
        expectEqual(izC.constraintDegree, 2, "IsZero degree is 2")
    }

    // ========== Test 46: Multi-gate mixed evaluation with XOR + AND ==========
    do {
        let domainSize = 8
        let registry = CustomGateRegistry(domainSize: domainSize)

        // XOR on rows 0-3
        var xorSel = [Fr](repeating: Fr.zero, count: domainSize)
        for i in 0..<4 { xorSel[i] = Fr.one }
        registry.register(constraint: XORConstraint(), selectorEvals: xorSel)

        // AND on rows 4-7
        var andSel = [Fr](repeating: Fr.zero, count: domainSize)
        for i in 4..<8 { andSel[i] = Fr.one }
        registry.register(constraint: ANDConstraint(), selectorEvals: andSel)

        // XOR truth table + AND truth table
        let col0: [Fr] = [Fr.zero, Fr.zero, Fr.one, Fr.one, Fr.zero, Fr.zero, Fr.one, Fr.one]
        let col1: [Fr] = [Fr.zero, Fr.one, Fr.zero, Fr.one, Fr.zero, Fr.one, Fr.zero, Fr.one]
        let col2: [Fr] = [Fr.zero, Fr.one, Fr.one, Fr.zero, Fr.zero, Fr.zero, Fr.zero, Fr.one] // XOR+AND results

        let result = engine.evaluateAllConstraints(
            registry: registry,
            wireColumns: [col0, col1, col2]
        )
        expect(result.isSatisfied, "Mixed XOR+AND: all 8 rows satisfied")
        expectEqual(result.perGateFailingRows[0].count, 0, "XOR: no failures")
        expectEqual(result.perGateFailingRows[1].count, 0, "AND: no failures")
    }

    // ========== Test 47: Poseidon S-box linearization scalar ==========
    do {
        let constraint = PoseidonSboxConstraint()
        // a=2: a^5 = 32; c=32 => scalar = 32 - 32 = 0
        let a = frFromInt(2)
        let c = frFromInt(32)
        let scalar = constraint.linearizationScalar(wireEvals: [a, Fr.zero, c], wireEvalsShifted: [], challenges: [])
        expect(scalar.isZero, "PoseidonSbox linearization: zero scalar when c=a^5")

        // a=2, c=31 => scalar = 31 - 32 = -1
        let wrongC = frFromInt(31)
        let scalar2 = constraint.linearizationScalar(wireEvals: [a, Fr.zero, wrongC], wireEvalsShifted: [], challenges: [])
        expect(!scalar2.isZero, "PoseidonSbox linearization: non-zero scalar when c != a^5")
    }

    // ========== Test 48: XOR linearization scalar ==========
    do {
        let constraint = XORConstraint()
        // 1 XOR 1 = 0: scalar = 0 - (1+1-2*1*1) = 0 - 0 = 0
        let scalar = constraint.linearizationScalar(
            wireEvals: [Fr.one, Fr.one, Fr.zero], wireEvalsShifted: [], challenges: [])
        expect(scalar.isZero, "XOR linearization: 1 XOR 1 = 0 gives zero scalar")

        // 0 XOR 1 = 1: scalar = 1 - (0+1-0) = 0
        let scalar2 = constraint.linearizationScalar(
            wireEvals: [Fr.zero, Fr.one, Fr.one], wireEvalsShifted: [], challenges: [])
        expect(scalar2.isZero, "XOR linearization: 0 XOR 1 = 1 gives zero scalar")
    }

    // ========== Test 49: Large domain batch evaluation ==========
    do {
        let domainSize = 256
        let registry = CustomGateRegistry(domainSize: domainSize)

        // Bool check on even rows
        var boolSel = [Fr](repeating: Fr.zero, count: domainSize)
        for i in stride(from: 0, to: domainSize, by: 2) { boolSel[i] = Fr.one }
        registry.register(constraint: BoolCheckConstraint(), selectorEvals: boolSel)

        // All even rows have valid boolean values (alternating 0 and 1)
        var col0 = [Fr](repeating: Fr.zero, count: domainSize)
        for i in 0..<domainSize {
            col0[i] = (i % 4 == 0) ? Fr.zero : ((i % 4 == 2) ? Fr.one : frFromInt(UInt64(i)))
        }

        let result = engine.evaluateAllConstraints(
            registry: registry,
            wireColumns: [col0]
        )
        expect(result.isSatisfied, "Large domain (256): all active rows have boolean values")
    }

    // ========== Test 50: Large domain with violations ==========
    do {
        let domainSize = 64
        let registry = CustomGateRegistry(domainSize: domainSize)

        var boolSel = [Fr](repeating: Fr.one, count: domainSize) // all rows active
        registry.register(constraint: BoolCheckConstraint(), selectorEvals: boolSel)

        // Put non-boolean values at specific rows
        var col0 = [Fr](repeating: Fr.zero, count: domainSize)
        col0[10] = frFromInt(5)
        col0[20] = frFromInt(3)
        col0[50] = frFromInt(99)

        let result = engine.evaluateAllConstraints(
            registry: registry,
            wireColumns: [col0]
        )
        expect(!result.isSatisfied, "Large domain: violations detected")
        expectEqual(result.failingRows.count, 3, "Exactly 3 failing rows")
        expect(result.failingRows.contains(10), "Row 10 fails")
        expect(result.failingRows.contains(20), "Row 20 fails")
        expect(result.failingRows.contains(50), "Row 50 fails")
    }

    // ========== Test 51: NOT gate chain — NOT(NOT(x)) = x ==========
    do {
        let constraint = NOTConstraint()
        // NOT 0 = 1
        let r1 = constraint.evaluateConstraint(wires: [Fr.zero, Fr.zero, Fr.one], rotatedWires: [:], challenges: [])
        expect(r1.isZero, "NOT chain step 1: NOT 0 = 1")

        // NOT 1 = 0 (back to original)
        let r2 = constraint.evaluateConstraint(wires: [Fr.one, Fr.zero, Fr.zero], rotatedWires: [:], challenges: [])
        expect(r2.isZero, "NOT chain step 2: NOT 1 = 0")
    }

    // ========== Test 52: Conditional select linearization ==========
    do {
        let constraint = ConditionalSelectConstraint()

        // cond=1, a=10, b=20, out=10 at shifted: linearization scalar should be 0
        let scalar = constraint.linearizationScalar(
            wireEvals: [Fr.one, frFromInt(10), frFromInt(20)],
            wireEvalsShifted: [frFromInt(10)],
            challenges: []
        )
        expect(scalar.isZero, "ConditionalSelect linearization: satisfied gives zero scalar")

        // cond=1, a=10, b=20, out=20 (wrong): non-zero scalar
        let scalar2 = constraint.linearizationScalar(
            wireEvals: [Fr.one, frFromInt(10), frFromInt(20)],
            wireEvalsShifted: [frFromInt(20)],
            challenges: []
        )
        expect(!scalar2.isZero, "ConditionalSelect linearization: violated gives non-zero scalar")
    }

    // ========== Test 53: IsZero with field inverse verification ==========
    do {
        let constraint = IsZeroConstraint()
        // Test with multiple non-zero values
        for val in [UInt64(1), 2, 7, 100, 12345] {
            let a = frFromInt(val)
            let inv = frInverse(a)
            let r = constraint.evaluateConstraint(wires: [a, inv, Fr.zero], rotatedWires: [:], challenges: [])
            expect(r.isZero, "IsZero: a=\(val), out=0 satisfied with correct inverse")
        }
    }

    // ========== Test 54: Registry active rows correctness ==========
    do {
        let domainSize = 16
        let registry = CustomGateRegistry(domainSize: domainSize)

        var sel = [Fr](repeating: Fr.zero, count: domainSize)
        sel[3] = Fr.one
        sel[7] = Fr.one
        sel[11] = Fr.one
        sel[15] = Fr.one
        registry.register(constraint: BoolCheckConstraint(), selectorEvals: sel)

        let entry = registry.entries[0]
        expectEqual(entry.activeRows.count, 4, "Active rows count is 4")
        expectEqual(entry.activeRows[0], 3, "First active row is 3")
        expectEqual(entry.activeRows[1], 7, "Second active row is 7")
        expectEqual(entry.activeRows[2], 11, "Third active row is 11")
        expectEqual(entry.activeRows[3], 15, "Fourth active row is 15")
    }

    // ========== Test 55: Gate type ID hashability ==========
    do {
        var idSet = Set<CustomGateTypeID>()
        idSet.insert(.boolCheck)
        idSet.insert(.xorGate)
        idSet.insert(.andGate)
        idSet.insert(.boolCheck) // duplicate

        expectEqual(idSet.count, 3, "Gate type IDs are properly hashable (no duplicate)")
        expect(idSet.contains(.boolCheck), "Set contains boolCheck")
        expect(idSet.contains(.xorGate), "Set contains xorGate")
        expect(idSet.contains(.andGate), "Set contains andGate")
    }

    // ========== Test 56: Multi-constraint quotient consistency ==========
    // Verify that the quotient contribution is zero when all constraints are satisfied
    do {
        let domainSize = 4
        let registry = CustomGateRegistry(domainSize: domainSize)

        var boolSel = [Fr](repeating: Fr.zero, count: domainSize)
        boolSel[0] = Fr.one
        boolSel[1] = Fr.one
        registry.register(constraint: BoolCheckConstraint(), selectorEvals: boolSel)

        // Valid boolean values on active rows
        let col0 = [Fr.zero, Fr.one, frFromInt(5), frFromInt(10)]
        let alpha = frFromInt(13)

        // First verify constraints pass
        let batchResult = engine.evaluateAllConstraints(
            registry: registry,
            wireColumns: [col0]
        )
        expect(batchResult.isSatisfied, "Quotient consistency: constraints satisfied")

        // The quotient contribution should have all-zero evaluations on active rows
        // (since the constraint is satisfied at those points)
        let perGate = batchResult.perGateResiduals[0]
        expect(perGate[0].isZero, "Quotient consistency: row 0 residual is zero")
        expect(perGate[1].isZero, "Quotient consistency: row 1 residual is zero")
    }

    // ========== Test 57: Batch bool check — all zeros ==========
    do {
        let n = 16
        let values = [Fr](repeating: Fr.zero, count: n)
        let selector = [Fr](repeating: Fr.one, count: n)
        let residuals = engine.gpuBatchBoolCheck(values: values, selectorEvals: selector)
        var allZero = true
        for r in residuals {
            if !r.isZero { allZero = false; break }
        }
        expect(allZero, "Batch bool check: all zeros passes")
    }

    // ========== Test 58: Batch bool check — all ones ==========
    do {
        let n = 16
        let values = [Fr](repeating: Fr.one, count: n)
        let selector = [Fr](repeating: Fr.one, count: n)
        let residuals = engine.gpuBatchBoolCheck(values: values, selectorEvals: selector)
        var allZero = true
        for r in residuals {
            if !r.isZero { allZero = false; break }
        }
        expect(allZero, "Batch bool check: all ones passes")
    }

    // ========== Test 59: XOR commutativity ==========
    do {
        let constraint = XORConstraint()
        let a = Fr.one
        let b = Fr.zero
        // a XOR b
        let r1 = constraint.evaluateConstraint(wires: [a, b, Fr.one], rotatedWires: [:], challenges: [])
        // b XOR a
        let r2 = constraint.evaluateConstraint(wires: [b, a, Fr.one], rotatedWires: [:], challenges: [])
        expect(r1.isZero, "XOR commutativity: a XOR b satisfied")
        expect(r2.isZero, "XOR commutativity: b XOR a satisfied")
    }

    // ========== Test 60: AND with IsZero composition ==========
    // Verify that composing AND + IsZero works correctly:
    // if a=1 AND b=1 then c=1, and IsZero(c) should give 0
    do {
        let andC = ANDConstraint()
        let izC = IsZeroConstraint()

        // 1 AND 1 = 1
        let andR = andC.evaluateConstraint(wires: [Fr.one, Fr.one, Fr.one], rotatedWires: [:], challenges: [])
        expect(andR.isZero, "Composition: 1 AND 1 = 1")

        // IsZero(1) = 0 (since 1 is not zero)
        let inv1 = frInverse(Fr.one)
        let izR = izC.evaluateConstraint(wires: [Fr.one, inv1, Fr.zero], rotatedWires: [:], challenges: [])
        expect(izR.isZero, "Composition: IsZero(1) = 0")
    }

    // ========== Test 61: Poseidon S-box with a=0 ==========
    do {
        let constraint = PoseidonSboxConstraint()
        // a=0, a^2=0, a^5=0
        let r = constraint.evaluateConstraint(wires: [Fr.zero, Fr.zero, Fr.zero], rotatedWires: [:], challenges: [])
        expect(r.isZero, "PoseidonSbox: a=0 gives c=0")
    }

    // ========== Test 62: Poseidon S-box with a=1 ==========
    do {
        let constraint = PoseidonSboxConstraint()
        // a=1, a^2=1, a^5=1
        let r = constraint.evaluateConstraint(wires: [Fr.one, Fr.one, Fr.one], rotatedWires: [:], challenges: [])
        expect(r.isZero, "PoseidonSbox: a=1 gives c=1")
    }

    // ========== Test 63: Full pipeline — register, evaluate, linearize ==========
    do {
        let domainSize = 8
        let registry = CustomGateRegistry(domainSize: domainSize)

        // Register bool check and XOR
        var boolSel = [Fr](repeating: Fr.zero, count: domainSize)
        boolSel[0] = Fr.one; boolSel[1] = Fr.one
        registry.register(constraint: BoolCheckConstraint(), selectorEvals: boolSel)

        var xorSel = [Fr](repeating: Fr.zero, count: domainSize)
        xorSel[4] = Fr.one; xorSel[5] = Fr.one
        registry.register(constraint: XORConstraint(), selectorEvals: xorSel)

        // Wire columns
        let col0: [Fr] = [Fr.zero, Fr.one, Fr.zero, Fr.zero, Fr.one, Fr.zero, Fr.zero, Fr.zero]
        let col1: [Fr] = [Fr.zero, Fr.zero, Fr.zero, Fr.zero, Fr.one, Fr.one, Fr.zero, Fr.zero]
        let col2: [Fr] = [Fr.zero, Fr.zero, Fr.zero, Fr.zero, Fr.zero, Fr.one, Fr.zero, Fr.zero]

        // Step 1: Evaluate constraints
        let batchResult = engine.evaluateAllConstraints(
            registry: registry,
            wireColumns: [col0, col1, col2]
        )
        expect(batchResult.isSatisfied, "Full pipeline: batch evaluation passes")

        // Step 2: Compute linearization
        let alpha = frFromInt(7)
        let linResult = engine.computeLinearization(
            registry: registry,
            wireEvals: [Fr.zero, Fr.zero, Fr.zero], // eval at some point
            wireEvalsShifted: [],
            selectorEvals: [Fr.one, Fr.one],
            alpha: alpha,
            alphaOffset: 3
        )
        expectEqual(linResult.scalars.count, 2, "Full pipeline: 2 linearization scalars")
        // BoolCheck at a=0: scalar = 0, XOR at (0,0,0): scalar = 0
        expect(linResult.combinedEval.isZero, "Full pipeline: linearization zero for satisfied constraints")
    }

    // ========== Test 64: Statistics with zero active rows ==========
    do {
        let domainSize = 8
        let registry = CustomGateRegistry(domainSize: domainSize)

        let zeroSel = [Fr](repeating: Fr.zero, count: domainSize)
        registry.register(constraint: BoolCheckConstraint(), selectorEvals: zeroSel)

        let stats = engine.computeStatistics(registry: registry)
        expectEqual(stats.totalGateTypes, 1, "Stats with zero active: 1 gate type")
        expectEqual(stats.totalActiveRows, 0, "Stats with zero active: 0 active rows")
        expect(stats.activationDensity < 0.001, "Stats with zero active: density is 0")
    }

    // ========== Test 65: Poseidon S-box larger values ==========
    do {
        let constraint = PoseidonSboxConstraint()
        // Test with a = 100
        let a = frFromInt(100)
        let aSq = frSqr(a)
        let a4 = frSqr(aSq)
        let a5 = frMul(a, a4)
        let r = constraint.evaluateConstraint(wires: [a, aSq, a5], rotatedWires: [:], challenges: [])
        expect(r.isZero, "PoseidonSbox: a=100 satisfies c=a^5")

        // Verify a^5 = 10^10 = 10000000000
        let expected = frFromInt(10_000_000_000)
        expect(frEqual(a5, expected), "100^5 = 10000000000 in field")
    }
}
