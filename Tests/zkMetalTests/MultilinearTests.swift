import zkMetal

func runMultilinearTests() {
    suite("MultilinearPoly basics")

    // Test 1: Create from evaluation table
    do {
        let evals: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let p = MultilinearPoly.extend(fromEvals: evals)
        expectEqual(p.numVars, 2, "extend numVars")
        expectEqual(p.size, 4, "extend size")
    }

    // Test 2: Evaluate at boolean point recovers evaluation table
    do {
        let evals: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
        let p = MultilinearPoly(numVars: 2, evals: evals)

        // f(0,0) = evals[0] = 10
        let v00 = p.evaluate(at: [Fr.zero, Fr.zero])
        expect(frToInt(v00) == frToInt(frFromInt(10)), "f(0,0) = 10")

        // f(0,1) = evals[1] = 20
        let v01 = p.evaluate(at: [Fr.zero, Fr.one])
        expect(frToInt(v01) == frToInt(frFromInt(20)), "f(0,1) = 20")

        // f(1,0) = evals[2] = 30
        let v10 = p.evaluate(at: [Fr.one, Fr.zero])
        expect(frToInt(v10) == frToInt(frFromInt(30)), "f(1,0) = 30")

        // f(1,1) = evals[3] = 40
        let v11 = p.evaluate(at: [Fr.one, Fr.one])
        expect(frToInt(v11) == frToInt(frFromInt(40)), "f(1,1) = 40")
    }

    // Test 3: C-accelerated evaluate matches Swift evaluate
    do {
        let evals: [Fr] = [frFromInt(5), frFromInt(15), frFromInt(25), frFromInt(35),
                           frFromInt(45), frFromInt(55), frFromInt(65), frFromInt(75)]
        let p = MultilinearPoly(numVars: 3, evals: evals)
        let point: [Fr] = [frFromInt(3), frFromInt(7), frFromInt(11)]

        let swiftResult = p.evaluate(at: point)
        let cResult = p.evaluateC(at: point)
        expect(frToInt(swiftResult) == frToInt(cResult), "evaluateC matches Swift evaluate")
    }

    // Test 4: Fix variable 0 (MSB)
    do {
        let evals: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let p = MultilinearPoly(numVars: 2, evals: evals)

        // Fix x0 = 0 => get [f(0,0), f(0,1)] = [1, 2]
        let p0 = p.fixVariable(Fr.zero)
        expect(frToInt(p0.evals[0]) == frToInt(frFromInt(1)), "fixVar(0): eval[0]")
        expect(frToInt(p0.evals[1]) == frToInt(frFromInt(2)), "fixVar(0): eval[1]")

        // Fix x0 = 1 => get [f(1,0), f(1,1)] = [3, 4]
        let p1 = p.fixVariable(Fr.one)
        expect(frToInt(p1.evals[0]) == frToInt(frFromInt(3)), "fixVar(1): eval[0]")
        expect(frToInt(p1.evals[1]) == frToInt(frFromInt(4)), "fixVar(1): eval[1]")
    }

    // Test 5: Fix variable at arbitrary index
    do {
        // 3-variable polynomial f(x0, x1, x2) with evals indexed as 000, 001, 010, 011, 100, 101, 110, 111
        let evals: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4),
                           frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8)]
        let p = MultilinearPoly(numVars: 3, evals: evals)

        // Fix x1 = 0 (variable at index 1): result should be f(x0, 0, x2)
        // For x0=0, x2=0: f(0,0,0) = 1
        // For x0=0, x2=1: f(0,0,1) = 2
        // For x0=1, x2=0: f(1,0,0) = 5
        // For x0=1, x2=1: f(1,0,1) = 6
        let fixedAt1 = p.fixVariable(index: 1, value: Fr.zero)
        expect(fixedAt1.numVars == 2, "fixVariable(index:) numVars")
        expect(frToInt(fixedAt1.evals[0]) == frToInt(frFromInt(1)), "fix x1=0: (0,0)")
        expect(frToInt(fixedAt1.evals[1]) == frToInt(frFromInt(2)), "fix x1=0: (0,1)")
        expect(frToInt(fixedAt1.evals[2]) == frToInt(frFromInt(5)), "fix x1=0: (1,0)")
        expect(frToInt(fixedAt1.evals[3]) == frToInt(frFromInt(6)), "fix x1=0: (1,1)")

        // Fix x2 = 1 (variable at index 2): result should be f(x0, x1, 1)
        // For x0=0, x1=0: f(0,0,1) = 2
        // For x0=0, x1=1: f(0,1,1) = 4
        // For x0=1, x1=0: f(1,0,1) = 6
        // For x0=1, x1=1: f(1,1,1) = 8
        let fixedAt2 = p.fixVariable(index: 2, value: Fr.one)
        expect(fixedAt2.numVars == 2, "fixVariable(index:2) numVars")
        expect(frToInt(fixedAt2.evals[0]) == frToInt(frFromInt(2)), "fix x2=1: (0,0)")
        expect(frToInt(fixedAt2.evals[1]) == frToInt(frFromInt(4)), "fix x2=1: (0,1)")
        expect(frToInt(fixedAt2.evals[2]) == frToInt(frFromInt(6)), "fix x2=1: (1,0)")
        expect(frToInt(fixedAt2.evals[3]) == frToInt(frFromInt(8)), "fix x2=1: (1,1)")
    }

    // Test 6: fixVariable(index:) is consistent with evaluate
    do {
        let evals: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4),
                           frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8)]
        let p = MultilinearPoly(numVars: 3, evals: evals)
        let r = frFromInt(42)

        // Fix variable 1 at r, then evaluate remaining at (3, 7) should equal
        // evaluating original at (3, 42, 7)
        let fixed = p.fixVariable(index: 1, value: r)
        let v1 = fixed.evaluate(at: [frFromInt(3), frFromInt(7)])
        let v2 = p.evaluate(at: [frFromInt(3), r, frFromInt(7)])
        expect(frToInt(v1) == frToInt(v2), "fixVariable(index:) consistent with evaluate")
    }

    suite("MultilinearPoly arithmetic")

    // Test 7: Add
    do {
        let a = MultilinearPoly(numVars: 2, evals: [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)])
        let b = MultilinearPoly(numVars: 2, evals: [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)])
        let c = MultilinearPoly.add(a, b)
        expect(frToInt(c.evals[0]) == frToInt(frFromInt(11)), "add: eval[0]")
        expect(frToInt(c.evals[3]) == frToInt(frFromInt(44)), "add: eval[3]")
    }

    // Test 8: Sub
    do {
        let a = MultilinearPoly(numVars: 1, evals: [frFromInt(100), frFromInt(200)])
        let b = MultilinearPoly(numVars: 1, evals: [frFromInt(30), frFromInt(50)])
        let c = MultilinearPoly.sub(a, b)
        expect(frToInt(c.evals[0]) == frToInt(frFromInt(70)), "sub: eval[0]")
        expect(frToInt(c.evals[1]) == frToInt(frFromInt(150)), "sub: eval[1]")
    }

    // Test 9: Mul (Hadamard)
    do {
        let a = MultilinearPoly(numVars: 2, evals: [frFromInt(2), frFromInt(3), frFromInt(5), frFromInt(7)])
        let b = MultilinearPoly(numVars: 2, evals: [frFromInt(11), frFromInt(13), frFromInt(17), frFromInt(19)])
        let c = MultilinearPoly.mul(a, b)
        expect(frToInt(c.evals[0]) == frToInt(frFromInt(22)), "mul: eval[0]")
        expect(frToInt(c.evals[1]) == frToInt(frFromInt(39)), "mul: eval[1]")
        expect(frToInt(c.evals[2]) == frToInt(frFromInt(85)), "mul: eval[2]")
        expect(frToInt(c.evals[3]) == frToInt(frFromInt(133)), "mul: eval[3]")
    }

    // Test 10: Scale
    do {
        let a = MultilinearPoly(numVars: 2, evals: [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)])
        let c = MultilinearPoly.scale(a, by: frFromInt(10))
        expect(frToInt(c.evals[0]) == frToInt(frFromInt(10)), "scale: eval[0]")
        expect(frToInt(c.evals[3]) == frToInt(frFromInt(40)), "scale: eval[3]")
    }

    suite("MultilinearPoly eq polynomial")

    // Test 11: eq poly at boolean points
    do {
        // eq(r, x) at x = r should give product of r_i^2 + (1-r_i)^2
        // But at boolean points: eq(r, b) = product_i (r_i * b_i + (1-r_i)(1-b_i))
        // eq(0, 0) = 1, eq(1, 1) = 1, eq(0, 1) = 0, eq(1, 0) = 0
        let eq = MultilinearPoly.eqPoly(point: [Fr.zero, Fr.one])
        // Only eq(0,1) = evals[1] should be 1, rest 0
        expect(frToInt(eq[0]) == frToInt(Fr.zero), "eq([0,1]): (0,0)")
        expect(frToInt(eq[1]) == frToInt(Fr.one), "eq([0,1]): (0,1)")
        expect(frToInt(eq[2]) == frToInt(Fr.zero), "eq([0,1]): (1,0)")
        expect(frToInt(eq[3]) == frToInt(Fr.zero), "eq([0,1]): (1,1)")
    }

    // Test 12: C-accelerated eq poly matches Swift
    do {
        let point = [frFromInt(3), frFromInt(7), frFromInt(11)]
        let swiftEq = MultilinearPoly.eqPoly(point: point)
        let cEq = MultilinearPoly.eqPolyC(point: point)
        var ok = true
        for i in 0..<8 {
            if frToInt(swiftEq[i]) != frToInt(cEq.evals[i]) { ok = false; break }
        }
        expect(ok, "eqPolyC matches Swift eqPoly")
    }

    suite("MultilinearPoly tensor product")

    // Test 13: Tensor of two 1-var polys gives 2-var poly
    do {
        let a = MultilinearPoly(numVars: 1, evals: [frFromInt(2), frFromInt(3)])
        let b = MultilinearPoly(numVars: 1, evals: [frFromInt(5), frFromInt(7)])
        let t = MultilinearPoly.tensor(a, b)
        expectEqual(t.numVars, 2, "tensor numVars")
        // result[i*2+j] = a[i] * b[j]
        expect(frToInt(t.evals[0]) == frToInt(frFromInt(10)), "tensor: 2*5")
        expect(frToInt(t.evals[1]) == frToInt(frFromInt(14)), "tensor: 2*7")
        expect(frToInt(t.evals[2]) == frToInt(frFromInt(15)), "tensor: 3*5")
        expect(frToInt(t.evals[3]) == frToInt(frFromInt(21)), "tensor: 3*7")
    }

    // Test 14: Tensor evaluation = product of evaluations
    do {
        let a = MultilinearPoly(numVars: 2, evals: [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)])
        let b = MultilinearPoly(numVars: 1, evals: [frFromInt(5), frFromInt(7)])
        let t = MultilinearPoly.tensor(a, b)
        let ra = [frFromInt(3), frFromInt(11)]
        let rb = [frFromInt(17)]
        let point = ra + rb
        let tVal = t.evaluate(at: point)
        let aVal = a.evaluate(at: ra)
        let bVal = b.evaluate(at: rb)
        expect(frToInt(tVal) == frToInt(frMul(aVal, bVal)), "tensor eval = product of evals")
    }

    suite("MultilinearPoly batch evaluate")

    // Test 15: Batch evaluate matches individual evaluations
    do {
        let p1 = MultilinearPoly(numVars: 2, evals: [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)])
        let p2 = MultilinearPoly(numVars: 2, evals: [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)])
        let p3 = MultilinearPoly(numVars: 2, evals: [frFromInt(5), frFromInt(15), frFromInt(25), frFromInt(35)])
        let point = [frFromInt(7), frFromInt(13)]

        let batchResults = MultilinearPoly.batchEvaluate(polys: [p1, p2, p3], point: point)
        let v1 = p1.evaluateC(at: point)
        let v2 = p2.evaluateC(at: point)
        let v3 = p3.evaluateC(at: point)

        expect(frToInt(batchResults[0]) == frToInt(v1), "batch eval[0]")
        expect(frToInt(batchResults[1]) == frToInt(v2), "batch eval[1]")
        expect(frToInt(batchResults[2]) == frToInt(v3), "batch eval[2]")
    }

    suite("MultilinearPoly linear combination")

    // Test 16: Linear combination
    do {
        let p1 = MultilinearPoly(numVars: 1, evals: [frFromInt(1), frFromInt(2)])
        let p2 = MultilinearPoly(numVars: 1, evals: [frFromInt(3), frFromInt(4)])
        let lc = MultilinearPoly.linearCombination(coeffs: [frFromInt(10), frFromInt(100)], polys: [p1, p2])
        // 10*[1,2] + 100*[3,4] = [310, 420]
        expect(frToInt(lc.evals[0]) == frToInt(frFromInt(310)), "linComb: eval[0]")
        expect(frToInt(lc.evals[1]) == frToInt(frFromInt(420)), "linComb: eval[1]")
    }

    suite("MultilinearPoly randomize")

    // Test 17: Random polynomial has correct dimensions
    do {
        let r = MultilinearPoly.randomize(numVars: 4)
        expectEqual(r.numVars, 4, "random numVars")
        expectEqual(r.size, 16, "random size")
        // Check it's not all zeros (astronomically unlikely)
        expect(!r.isZeroPoly, "random poly is not zero")
    }

    suite("MultilinearPoly sumcheck round sums")

    // Test 18: s(0) + s(1) = total sum
    do {
        let evals: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let p = MultilinearPoly(numVars: 2, evals: evals)
        let (s0, s1) = p.sumcheckRoundSums()
        let totalSum = frAdd(frAdd(frFromInt(1), frFromInt(2)), frAdd(frFromInt(3), frFromInt(4)))
        expect(frToInt(frAdd(s0, s1)) == frToInt(totalSum), "s(0)+s(1) = total sum")
    }

    suite("MultilinearPoly partial evaluate")

    // Test 19: Partial evaluate multiple variables
    do {
        let evals: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4),
                           frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8)]
        let p = MultilinearPoly(numVars: 3, evals: evals)
        let point = [frFromInt(3), frFromInt(7), frFromInt(11)]

        // Partial evaluate first 2 variables should give 1-var poly
        let partial = p.partialEvaluate(values: [frFromInt(3), frFromInt(7)])
        let fromPartial = partial.evaluate(at: [frFromInt(11)])
        let full = p.evaluate(at: point)
        expect(frToInt(fromPartial) == frToInt(full), "partial evaluate consistent with full")
    }

    suite("MultilinearPoly sparse conversion")

    // Test 20: Convert sparse -> dense and evaluate
    do {
        let entries = [
            SparseEntry(idx: 0, val: frFromInt(10)),
            SparseEntry(idx: 3, val: frFromInt(20))
        ]
        let sparse = SparseMultilinearPoly(numVars: 2, sortedEntries: entries)
        let dense = MultilinearPoly.fromSparse(sparse)
        expect(frToInt(dense.evals[0]) == frToInt(frFromInt(10)), "sparse->dense: idx 0")
        expect(dense.evals[1].isZero, "sparse->dense: idx 1 zero")
        expect(dense.evals[2].isZero, "sparse->dense: idx 2 zero")
        expect(frToInt(dense.evals[3]) == frToInt(frFromInt(20)), "sparse->dense: idx 3")
    }
}
