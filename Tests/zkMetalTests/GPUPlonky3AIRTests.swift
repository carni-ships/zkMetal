// GPU Plonky3 AIR Engine Tests — validates multi-matrix AIR, extension field arithmetic,
// LogUp interactions, cross-table lookups, constraint evaluation, quotient polynomial,
// preprocessing, and the full proving pipeline.

import Foundation
import zkMetal

public func runGPUPlonky3AIRTests() {
    suite("GPUPlonky3AIR - Extension Field Basics")
    testBbExt4Zero()
    testBbExt4One()
    testBbExt4Lift()
    testBbExt4Equality()

    suite("GPUPlonky3AIR - Extension Field Arithmetic")
    testBbExt4Add()
    testBbExt4Sub()
    testBbExt4Neg()
    testBbExt4Scale()
    testBbExt4MulByOne()
    testBbExt4MulCommutativity()
    testBbExt4MulAssociativity()
    testBbExt4Distributivity()
    testBbExt4Sqr()

    suite("GPUPlonky3AIR - Extension Field Inverse")
    testBbExt4InvBase()
    testBbExt4InvExtension()

    suite("GPUPlonky3AIR - Extension Field Pow + UInt128")
    testBbExt4PowSmall()
    testUInt128BitOps()

    suite("GPUPlonky3AIR - Fibonacci Multi-Matrix AIR")
    testFibAIRDimensions()
    testFibAIRTraceGen()
    testFibAIRConstraintsSatisfied()
    testFibAIRConstraintsFailOnBadTrace()
    testFibAIRCustomInitialValues()

    suite("GPUPlonky3AIR - Range Check AIR")
    testRangeCheckDimensions()
    testRangeCheckPreprocessed()
    testRangeCheckConstraintsSatisfied()
    testRangeCheckConstraintsFail()

    suite("GPUPlonky3AIR - Arithmetic AIR")
    testArithAIRDimensions()
    testArithAIRPreprocessedTrace()
    testArithAIRMainTrace()
    testArithAIRConstraintsSatisfied()

    suite("GPUPlonky3AIR - Lookup AIR")
    testLookupAIRDimensions()
    testLookupAIRInteractions()
    testLookupAIRMainTrace()
    testLookupAIRConstraints()

    suite("GPUPlonky3AIR - LogUp Trace Generator")
    testLogUpTraceColumns()
    testLogUpTraceNonZero()
    testLogUpSingleBus()

    suite("GPUPlonky3AIR - Interaction Bus")
    testInteractionBusSendReceiveCount()

    suite("GPUPlonky3AIR - Engine Construction")
    testEngineCreation()
    testEngineBlowupFactor()
    testEngineDegreeBound()

    suite("GPUPlonky3AIR - Quotient Polynomial")
    testQuotientFibonacci()
    testQuotientZeroOnValidTrace()
    testQuotientNonZeroOnBadTrace()
    testQuotientChunking()
    testQuotientChunkingSingle()

    suite("GPUPlonky3AIR - Trace Verification")
    testVerifyValidFibTrace()
    testVerifyInvalidFibTrace()
    testVerifyColumnCountMismatch()
    testVerifyRowCountMismatch()

    suite("GPUPlonky3AIR - Batch and Point Eval")
    testBatchEvalFibonacci()
    testPointEvalFibonacci()

    suite("GPUPlonky3AIR - Full Proving Pipeline")
    testProveFibonacci()
    testProveRangeCheck()
    testProveArithmetic()
    testProveLookup()

    suite("GPUPlonky3AIR - Cross-Table Verifier")
    testCrossTableVerifierCreation()

    suite("GPUPlonky3AIR - Edge Cases")
    testMinimalTrace()
    testLargeTrace()
    testExtFieldChain()
}

// MARK: - Extension Field Basics

private func testBbExt4Zero() {
    let z = BbExt4.zero
    expect(z.isZero, "zero element is zero")
    expect(z.c.0 == Bb.zero, "zero c0")
    expect(z.c.1 == Bb.zero, "zero c1")
    expect(z.c.2 == Bb.zero, "zero c2")
    expect(z.c.3 == Bb.zero, "zero c3")
}

private func testBbExt4One() {
    let o = BbExt4.one
    expect(!o.isZero, "one is not zero")
    expect(o.c.0 == Bb.one, "one c0 is 1")
    expect(o.c.1 == Bb.zero, "one c1 is 0")
    expect(o.c.2 == Bb.zero, "one c2 is 0")
    expect(o.c.3 == Bb.zero, "one c3 is 0")
}

private func testBbExt4Lift() {
    let val = Bb(v: 42)
    let ext = BbExt4(base: val)
    expect(ext.c.0 == val, "lifted c0 matches")
    expect(ext.c.1 == Bb.zero, "lifted c1 is zero")
    expect(!ext.isZero, "lifted nonzero is not zero")

    let zeroLift = BbExt4(base: Bb.zero)
    expect(zeroLift.isZero, "lifted zero is zero")
}

private func testBbExt4Equality() {
    let a = BbExt4(c: (Bb(v: 1), Bb(v: 2), Bb(v: 3), Bb(v: 4)))
    let b = BbExt4(c: (Bb(v: 1), Bb(v: 2), Bb(v: 3), Bb(v: 4)))
    let c = BbExt4(c: (Bb(v: 1), Bb(v: 2), Bb(v: 3), Bb(v: 5)))
    expect(a == b, "same elements are equal")
    expect(a != c, "different elements are not equal")
}

// MARK: - Extension Field Arithmetic

private func testBbExt4Add() {
    let a = BbExt4(c: (Bb(v: 10), Bb(v: 20), Bb(v: 30), Bb(v: 40)))
    let b = BbExt4(c: (Bb(v: 1), Bb(v: 2), Bb(v: 3), Bb(v: 4)))
    let c = bbExt4Add(a, b)
    expectEqual(c.c.0.v, UInt32(11), "add c0")
    expectEqual(c.c.1.v, UInt32(22), "add c1")
    expectEqual(c.c.2.v, UInt32(33), "add c2")
    expectEqual(c.c.3.v, UInt32(44), "add c3")

    // a + 0 = a
    let d = bbExt4Add(a, BbExt4.zero)
    expect(d == a, "a + 0 = a")
}

private func testBbExt4Sub() {
    let a = BbExt4(c: (Bb(v: 10), Bb(v: 20), Bb(v: 30), Bb(v: 40)))
    let b = BbExt4(c: (Bb(v: 1), Bb(v: 2), Bb(v: 3), Bb(v: 4)))
    let c = bbExt4Sub(a, b)
    expectEqual(c.c.0.v, UInt32(9), "sub c0")
    expectEqual(c.c.1.v, UInt32(18), "sub c1")
    expectEqual(c.c.2.v, UInt32(27), "sub c2")
    expectEqual(c.c.3.v, UInt32(36), "sub c3")

    // a - a = 0
    let z = bbExt4Sub(a, a)
    expect(z.isZero, "a - a = 0")
}

private func testBbExt4Neg() {
    let a = BbExt4(c: (Bb(v: 5), Bb(v: 10), Bb(v: 15), Bb(v: 20)))
    let neg = bbExt4Neg(a)
    let sum = bbExt4Add(a, neg)
    expect(sum.isZero, "a + (-a) = 0")

    // neg(0) = 0
    let negZero = bbExt4Neg(BbExt4.zero)
    expect(negZero.isZero, "neg(0) = 0")
}

private func testBbExt4Scale() {
    let a = BbExt4(c: (Bb(v: 2), Bb(v: 3), Bb(v: 4), Bb(v: 5)))
    let s = Bb(v: 10)
    let scaled = bbExt4Scale(a, s)
    expectEqual(scaled.c.0.v, UInt32(20), "scale c0")
    expectEqual(scaled.c.1.v, UInt32(30), "scale c1")
    expectEqual(scaled.c.2.v, UInt32(40), "scale c2")
    expectEqual(scaled.c.3.v, UInt32(50), "scale c3")
}

private func testBbExt4MulByOne() {
    let a = BbExt4(c: (Bb(v: 7), Bb(v: 13), Bb(v: 19), Bb(v: 23)))
    let product = bbExt4Mul(a, BbExt4.one)
    expect(product == a, "a * 1 = a")

    let product2 = bbExt4Mul(BbExt4.one, a)
    expect(product2 == a, "1 * a = a")
}

private func testBbExt4MulCommutativity() {
    let a = BbExt4(c: (Bb(v: 3), Bb(v: 7), Bb(v: 11), Bb(v: 13)))
    let b = BbExt4(c: (Bb(v: 5), Bb(v: 9), Bb(v: 17), Bb(v: 19)))
    let ab = bbExt4Mul(a, b)
    let ba = bbExt4Mul(b, a)
    expect(ab == ba, "a * b = b * a")
}

private func testBbExt4MulAssociativity() {
    let a = BbExt4(c: (Bb(v: 2), Bb(v: 3), Bb(v: 0), Bb(v: 0)))
    let b = BbExt4(c: (Bb(v: 5), Bb(v: 0), Bb(v: 7), Bb(v: 0)))
    let c = BbExt4(c: (Bb(v: 11), Bb(v: 0), Bb(v: 0), Bb(v: 13)))
    let ab_c = bbExt4Mul(bbExt4Mul(a, b), c)
    let a_bc = bbExt4Mul(a, bbExt4Mul(b, c))
    expect(ab_c == a_bc, "(a*b)*c = a*(b*c)")
}

private func testBbExt4Distributivity() {
    let a = BbExt4(c: (Bb(v: 3), Bb(v: 5), Bb(v: 7), Bb(v: 11)))
    let b = BbExt4(c: (Bb(v: 2), Bb(v: 4), Bb(v: 6), Bb(v: 8)))
    let c = BbExt4(c: (Bb(v: 13), Bb(v: 17), Bb(v: 19), Bb(v: 23)))
    let lhs = bbExt4Mul(a, bbExt4Add(b, c))
    let rhs = bbExt4Add(bbExt4Mul(a, b), bbExt4Mul(a, c))
    expect(lhs == rhs, "a*(b+c) = a*b + a*c")
}

private func testBbExt4Sqr() {
    let a = BbExt4(c: (Bb(v: 3), Bb(v: 7), Bb(v: 11), Bb(v: 13)))
    let sq = bbExt4Sqr(a)
    let mulSelf = bbExt4Mul(a, a)
    expect(sq == mulSelf, "sqr(a) = a * a")
}

// MARK: - Extension Field Inverse

private func testBbExt4InvBase() {
    // Inverse of a base field lift: (5, 0, 0, 0)^{-1} should have c0 = 5^{-1}, rest 0
    let a = BbExt4(base: Bb(v: 5))
    let inv = bbExt4Inv(a)
    let product = bbExt4Mul(a, inv)
    expect(product == BbExt4.one, "a * a^{-1} = 1 for base element")
}

private func testBbExt4InvExtension() {
    let a = BbExt4(c: (Bb(v: 3), Bb(v: 7), Bb(v: 11), Bb(v: 13)))
    let inv = bbExt4Inv(a)
    let product = bbExt4Mul(a, inv)
    expect(product == BbExt4.one, "a * a^{-1} = 1 for extension element")
}

// MARK: - Extension Field Pow + UInt128

private func testBbExt4PowSmall() {
    let a = BbExt4(base: Bb(v: 3))
    let a3 = bbExt4Pow(a, 3)
    let manual = bbExt4Mul(bbExt4Mul(a, a), a)
    expect(a3 == manual, "a^3 = a*a*a")

    let a4 = bbExt4Pow(a, 4)
    let manual4 = bbExt4Mul(a3, a)
    expect(a4 == manual4, "a^4 = a^3 * a")
}

private func testUInt128BitOps() {
    let a = UInt128(lo: 0b1010, hi: 0)
    expect(a.bit(0) == false, "bit 0 is 0")
    expect(a.bit(1) == true, "bit 1 is 1")
    expect(a.bit(2) == false, "bit 2 is 0")
    expect(a.bit(3) == true, "bit 3 is 1")
    expectEqual(a.bitWidth, 3, "highest bit is 3")

    let b = UInt128(lo: 0, hi: 1)
    expectEqual(b.bitWidth, 64, "bit 64 set")
    expect(b.bit(64) == true, "bit 64 is 1")
}

// MARK: - Fibonacci Multi-Matrix AIR

private func testFibAIRDimensions() {
    let air = Plonky3FibonacciMultiAIR(logTraceLength: 4)
    expectEqual(air.preprocessedWidth, 0, "no preprocessed cols")
    expectEqual(air.mainWidth, 2, "2 main cols")
    expectEqual(air.permutationWidth, 0, "no perm cols")
    expectEqual(air.totalWidth, 2, "total width 2")
    expectEqual(air.logTraceLength, 4, "logN = 4")
    expectEqual(air.traceLength, 16, "N = 16")
    expectEqual(air.maxConstraintDegree, 1, "degree 1")
    expectEqual(air.numTransitionConstraints, 2, "2 constraints")
    expectEqual(air.interactionBuses.count, 0, "no interactions")
}

private func testFibAIRTraceGen() {
    let air = Plonky3FibonacciMultiAIR(logTraceLength: 3)
    let trace = air.generateMainTrace()
    expectEqual(trace.count, 2, "2 columns")
    expectEqual(trace[0].count, 8, "8 rows")

    // Verify Fibonacci sequence: 1,1,2,3,5,8,13,21
    expectEqual(trace[0][0].v, UInt32(1), "a[0] = 1")
    expectEqual(trace[1][0].v, UInt32(1), "b[0] = 1")
    expectEqual(trace[0][1].v, UInt32(1), "a[1] = b[0] = 1")
    expectEqual(trace[1][1].v, UInt32(2), "b[1] = a[0]+b[0] = 2")
    expectEqual(trace[0][2].v, UInt32(2), "a[2] = b[1] = 2")
    expectEqual(trace[1][2].v, UInt32(3), "b[2] = a[1]+b[1] = 3")
    expectEqual(trace[0][3].v, UInt32(3), "a[3] = 3")
    expectEqual(trace[1][3].v, UInt32(5), "b[3] = 5")

    expect(air.generatePreprocessedTrace() == nil, "no preprocessed trace")
}

private func testFibAIRConstraintsSatisfied() {
    let air = Plonky3FibonacciMultiAIR(logTraceLength: 4)
    let trace = air.generateMainTrace()
    let n = air.traceLength

    for row in 0..<(n - 1) {
        let evals = air.evaluateTransitionConstraints(
            preprocessed: nil,
            main: (current: trace.map { $0[row] }, next: trace.map { $0[row + 1] }),
            permutation: nil,
            challenges: []
        )
        for (ci, ev) in evals.enumerated() {
            expect(ev.isZero, "fib constraint \(ci) row \(row) is zero")
        }
    }
}

private func testFibAIRConstraintsFailOnBadTrace() {
    let air = Plonky3FibonacciMultiAIR(logTraceLength: 3)
    var trace = air.generateMainTrace()
    // Corrupt one value
    trace[1][3] = Bb(v: 999)

    let evals = air.evaluateTransitionConstraints(
        preprocessed: nil,
        main: (current: trace.map { $0[2] }, next: trace.map { $0[3] }),
        permutation: nil,
        challenges: []
    )
    let anyFailed = evals.contains { !$0.isZero }
    expect(anyFailed, "corrupted trace fails constraints")
}

private func testFibAIRCustomInitialValues() {
    let air = Plonky3FibonacciMultiAIR(logTraceLength: 3, a0: Bb(v: 2), b0: Bb(v: 3))
    let trace = air.generateMainTrace()
    expectEqual(trace[0][0].v, UInt32(2), "custom a0 = 2")
    expectEqual(trace[1][0].v, UInt32(3), "custom b0 = 3")
    expectEqual(trace[0][1].v, UInt32(3), "a[1] = b[0] = 3")
    expectEqual(trace[1][1].v, UInt32(5), "b[1] = 2+3 = 5")
}

// MARK: - Range Check AIR

private func testRangeCheckDimensions() {
    let air = Plonky3RangeCheckAIR(logTraceLength: 4)
    expectEqual(air.preprocessedWidth, 1, "1 preprocessed col")
    expectEqual(air.mainWidth, 1, "1 main col")
    expectEqual(air.permutationWidth, 0, "no perm cols")
    expectEqual(air.totalWidth, 2, "total 2")
    expectEqual(air.maxConstraintDegree, 2, "degree 2")
}

private func testRangeCheckPreprocessed() {
    let air = Plonky3RangeCheckAIR(logTraceLength: 3, activeRows: 5)
    let pp = air.generatePreprocessedTrace()
    expect(pp != nil, "has preprocessed")
    let selector = pp![0]
    expectEqual(selector.count, 8, "8 rows")
    // First 5 rows are active
    for i in 0..<5 {
        expectEqual(selector[i].v, UInt32(1), "row \(i) active")
    }
    // Rest are inactive
    for i in 5..<8 {
        expectEqual(selector[i].v, UInt32(0), "row \(i) inactive")
    }
}

private func testRangeCheckConstraintsSatisfied() {
    let air = Plonky3RangeCheckAIR(logTraceLength: 3)
    let pp = air.generatePreprocessedTrace()!
    let main = air.generateMainTrace()
    let n = air.traceLength

    for row in 0..<(n - 1) {
        let evals = air.evaluateTransitionConstraints(
            preprocessed: (current: pp.map { $0[row] }, next: pp.map { $0[row + 1] }),
            main: (current: main.map { $0[row] }, next: main.map { $0[row + 1] }),
            permutation: nil,
            challenges: []
        )
        for ev in evals {
            expect(ev.isZero, "range check constraint row \(row) satisfied")
        }
    }
}

private func testRangeCheckConstraintsFail() {
    let air = Plonky3RangeCheckAIR(logTraceLength: 3)
    let pp = air.generatePreprocessedTrace()!
    var main = air.generateMainTrace()
    // Corrupt: make value jump by 2 instead of 1
    main[0][2] = Bb(v: 5) // should be 2

    let evals = air.evaluateTransitionConstraints(
        preprocessed: (current: pp.map { $0[1] }, next: pp.map { $0[2] }),
        main: (current: main.map { $0[1] }, next: main.map { $0[2] }),
        permutation: nil,
        challenges: []
    )
    let anyFailed = evals.contains { !$0.isZero }
    expect(anyFailed, "corrupted range check fails")
}

// MARK: - Arithmetic AIR

private func testArithAIRDimensions() {
    let air = Plonky3ArithmeticAIR(logTraceLength: 3)
    expectEqual(air.preprocessedWidth, 2, "2 preprocessed")
    expectEqual(air.mainWidth, 4, "4 main")
    expectEqual(air.permutationWidth, 0, "0 perm")
    expectEqual(air.totalWidth, 6, "6 total")
    expectEqual(air.maxConstraintDegree, 3, "degree 3")
    expectEqual(air.numTransitionConstraints, 2, "2 constraints")
}

private func testArithAIRPreprocessedTrace() {
    let gates: [(isAdd: Bool, a: UInt32, b: UInt32)] = [
        (true, 1, 2), (false, 3, 4), (true, 5, 6), (false, 7, 8)
    ]
    let air = Plonky3ArithmeticAIR(logTraceLength: 2, gates: gates)
    let pp = air.generatePreprocessedTrace()!
    expectEqual(pp.count, 2, "2 selector columns")
    // Row 0: add
    expectEqual(pp[0][0].v, UInt32(1), "add_sel[0] = 1")
    expectEqual(pp[1][0].v, UInt32(0), "mul_sel[0] = 0")
    // Row 1: mul
    expectEqual(pp[0][1].v, UInt32(0), "add_sel[1] = 0")
    expectEqual(pp[1][1].v, UInt32(1), "mul_sel[1] = 1")
}

private func testArithAIRMainTrace() {
    let gates: [(isAdd: Bool, a: UInt32, b: UInt32)] = [
        (true, 3, 5), (false, 4, 6), (true, 10, 20), (false, 7, 8)
    ]
    let air = Plonky3ArithmeticAIR(logTraceLength: 2, gates: gates)
    let main = air.generateMainTrace()
    expectEqual(main.count, 4, "4 columns")
    expectEqual(main[0][0].v, UInt32(3), "a[0] = 3")
    expectEqual(main[1][0].v, UInt32(5), "b[0] = 5")
    expectEqual(main[2][0].v, UInt32(8), "c[0] = 3+5 = 8")
    expectEqual(main[3][0].v, UInt32(15), "d[0] = 3*5 = 15")
}

private func testArithAIRConstraintsSatisfied() {
    let gates: [(isAdd: Bool, a: UInt32, b: UInt32)] = [
        (true, 3, 5), (false, 4, 6), (true, 10, 20), (false, 7, 8)
    ]
    let air = Plonky3ArithmeticAIR(logTraceLength: 2, gates: gates)
    let pp = air.generatePreprocessedTrace()!
    let main = air.generateMainTrace()
    let n = air.traceLength

    for row in 0..<(n - 1) {
        let evals = air.evaluateTransitionConstraints(
            preprocessed: (current: pp.map { $0[row] }, next: pp.map { $0[row + 1] }),
            main: (current: main.map { $0[row] }, next: main.map { $0[row + 1] }),
            permutation: nil,
            challenges: []
        )
        for (ci, ev) in evals.enumerated() {
            expect(ev.isZero, "arith constraint \(ci) row \(row) satisfied")
        }
    }
}

// MARK: - Lookup AIR

private func testLookupAIRDimensions() {
    let air = Plonky3LookupAIR(logTraceLength: 3)
    expectEqual(air.preprocessedWidth, 0, "no preprocessed")
    expectEqual(air.mainWidth, 2, "2 main cols")
    expectEqual(air.permutationWidth, 2, "2 perm cols")
    expectEqual(air.totalWidth, 4, "4 total")
}

private func testLookupAIRInteractions() {
    let air = Plonky3LookupAIR(logTraceLength: 3)
    expectEqual(air.interactionBuses.count, 1, "1 bus")
    let bus = air.interactionBuses[0]
    expectEqual(bus.busIndex, 0, "bus index 0")
    expectEqual(bus.interactions.count, 1, "1 interaction")
    expect(bus.interactions[0].isSend, "interaction is send")
    expectEqual(bus.sendCount, 1, "1 send")
    expectEqual(bus.receiveCount, 0, "0 receive")
}

private func testLookupAIRMainTrace() {
    let air = Plonky3LookupAIR(logTraceLength: 3)
    let trace = air.generateMainTrace()
    expectEqual(trace.count, 2, "2 columns")
    expectEqual(trace[0].count, 8, "8 rows")
    // Values: 0,1,2,...,7
    for i in 0..<8 {
        expectEqual(trace[0][i].v, UInt32(i), "value[\(i)] = \(i)")
        expectEqual(trace[1][i].v, UInt32(1), "mult[\(i)] = 1")
    }
}

private func testLookupAIRConstraints() {
    let air = Plonky3LookupAIR(logTraceLength: 3)
    let trace = air.generateMainTrace()

    let evals = air.evaluateTransitionConstraints(
        preprocessed: nil,
        main: (current: trace.map { $0[0] }, next: trace.map { $0[1] }),
        permutation: nil,
        challenges: []
    )
    for ev in evals {
        expect(ev.isZero, "lookup constraint satisfied")
    }
}

// MARK: - LogUp Trace Generator

private func testLogUpTraceColumns() {
    let interaction = Plonky3Interaction(
        airIndex: 0, valueColumns: [0], numeratorColumn: 1, isSend: true)
    let bus = Plonky3InteractionBus(busIndex: 0, interactions: [interaction])
    let gen = Plonky3LogUpTraceGenerator(buses: [bus])

    let mainTrace: [[Bb]] = [
        [Bb(v: 1), Bb(v: 2), Bb(v: 3), Bb(v: 4)],  // values
        [Bb.one, Bb.one, Bb.one, Bb.one]              // multiplicities
    ]

    let alpha = BbExt4(base: Bb(v: 7))
    let beta = BbExt4(base: Bb(v: 13))
    let perm = gen.generatePermutationTrace(mainTrace: mainTrace, alpha: alpha, beta: beta)

    // 1 interaction * 2 columns = 2 perm columns
    expectEqual(perm.count, 2, "2 permutation columns")
    expectEqual(perm[0].count, 4, "4 rows")
    expectEqual(perm[1].count, 4, "4 rows invDenom")
}

private func testLogUpTraceNonZero() {
    let interaction = Plonky3Interaction(
        airIndex: 0, valueColumns: [0], numeratorColumn: 1, isSend: true)
    let bus = Plonky3InteractionBus(busIndex: 0, interactions: [interaction])
    let gen = Plonky3LogUpTraceGenerator(buses: [bus])

    let mainTrace: [[Bb]] = [
        [Bb(v: 10), Bb(v: 20)],
        [Bb.one, Bb.one]
    ]

    let alpha = BbExt4.one
    let beta = BbExt4(base: Bb(v: 5))
    let perm = gen.generatePermutationTrace(mainTrace: mainTrace, alpha: alpha, beta: beta)

    // Accumulators should not be zero (running sum of 1/(beta + v))
    expect(!perm[0][0].isZero, "accumulator[0] is nonzero")
    expect(!perm[0][1].isZero, "accumulator[1] is nonzero")
}

private func testLogUpSingleBus() {
    // With matching send and receive, the bus should close to zero
    let sendInteraction = Plonky3Interaction(
        airIndex: 0, valueColumns: [0], numeratorColumn: -1, isSend: true)
    let recvInteraction = Plonky3Interaction(
        airIndex: 0, valueColumns: [0], numeratorColumn: -1, isSend: false)
    let bus = Plonky3InteractionBus(
        busIndex: 0, interactions: [sendInteraction, recvInteraction])
    let gen = Plonky3LogUpTraceGenerator(buses: [bus])

    // Same values for send and receive => should balance
    let mainTrace: [[Bb]] = [
        [Bb(v: 1), Bb(v: 2), Bb(v: 3), Bb(v: 4)]
    ]

    let alpha = BbExt4.one
    let beta = BbExt4(base: Bb(v: 100))
    let perm = gen.generatePermutationTrace(mainTrace: mainTrace, alpha: alpha, beta: beta)

    // 2 interactions * 2 cols = 4 perm columns
    expectEqual(perm.count, 4, "4 permutation columns for send+receive")

    let verified = gen.verifyLogUpClosure(permColumns: perm, traceLength: 4)
    expect(verified, "send+receive with same values closes to zero")
}

// MARK: - Interaction Bus

private func testInteractionBusSendReceiveCount() {
    var bus = Plonky3InteractionBus(busIndex: 0)
    bus.addInteraction(Plonky3Interaction(airIndex: 0, valueColumns: [0], numeratorColumn: 1, isSend: true))
    bus.addInteraction(Plonky3Interaction(airIndex: 1, valueColumns: [0], numeratorColumn: 1, isSend: false))
    bus.addInteraction(Plonky3Interaction(airIndex: 2, valueColumns: [0], numeratorColumn: 1, isSend: true))
    expectEqual(bus.sendCount, 2, "2 sends")
    expectEqual(bus.receiveCount, 1, "1 receive")
}

// MARK: - Engine Construction

private func testEngineCreation() {
    let air = Plonky3FibonacciMultiAIR(logTraceLength: 4)
    let engine = GPUPlonky3AIREngine(air: air, logBlowup: 2, numQueries: 50, grindingBits: 8)
    expectEqual(engine.logBlowup, 2, "logBlowup 2")
    expectEqual(engine.numQueries, 50, "50 queries")
    expectEqual(engine.grindingBits, 8, "8 grinding bits")
}

private func testEngineBlowupFactor() {
    let air = Plonky3FibonacciMultiAIR(logTraceLength: 4)
    let e1 = GPUPlonky3AIREngine(air: air, logBlowup: 1)
    expectEqual(e1.blowupFactor, 2, "blowup 2x")
    let e2 = GPUPlonky3AIREngine(air: air, logBlowup: 3)
    expectEqual(e2.blowupFactor, 8, "blowup 8x")
    expectEqual(e2.evaluationDomainSize, 16 * 8, "eval domain 128")
}

private func testEngineDegreeBound() {
    let air = Plonky3FibonacciMultiAIR(logTraceLength: 4)
    let engine = GPUPlonky3AIREngine(air: air)
    expectEqual(engine.compositionDegreeBound, 16, "degree 1 * 16 rows")
    expectEqual(engine.numQuotientChunks, 1, "1 chunk for degree-1")

    let airArith = Plonky3ArithmeticAIR(logTraceLength: 3)
    let engineArith = GPUPlonky3AIREngine(air: airArith)
    expectEqual(engineArith.compositionDegreeBound, 24, "degree 3 * 8 rows")
    expectEqual(engineArith.numQuotientChunks, 3, "3 chunks for degree-3")
}

// MARK: - Quotient Polynomial

private func testQuotientFibonacci() {
    let air = Plonky3FibonacciMultiAIR(logTraceLength: 3)
    let engine = GPUPlonky3AIREngine(air: air)
    let trace = air.generateMainTrace()
    let alpha = BbExt4(base: Bb(v: 7))
    let quotient = engine.evaluateQuotient(
        preprocessed: nil, main: trace, permutation: nil,
        challenges: [], alpha: alpha)
    expectEqual(quotient.count, 8, "8 quotient evaluations")
}

private func testQuotientZeroOnValidTrace() {
    let air = Plonky3FibonacciMultiAIR(logTraceLength: 3)
    let engine = GPUPlonky3AIREngine(air: air)
    let trace = air.generateMainTrace()
    let alpha = BbExt4.one
    let quotient = engine.evaluateQuotient(
        preprocessed: nil, main: trace, permutation: nil,
        challenges: [], alpha: alpha)

    // On valid trace, all quotient evaluations should be zero (no constraint violations)
    for i in 0..<(quotient.count - 1) {
        expect(quotient[i].isZero, "quotient[\(i)] is zero on valid trace")
    }
    // Last row has no transition constraint
    expect(quotient[quotient.count - 1].isZero, "last row quotient is zero")
}

private func testQuotientNonZeroOnBadTrace() {
    let air = Plonky3FibonacciMultiAIR(logTraceLength: 3)
    let engine = GPUPlonky3AIREngine(air: air)
    var trace = air.generateMainTrace()
    trace[1][2] = Bb(v: 999)  // corrupt
    let alpha = BbExt4.one
    let quotient = engine.evaluateQuotient(
        preprocessed: nil, main: trace, permutation: nil,
        challenges: [], alpha: alpha)

    let anyNonZero = quotient.contains { !$0.isZero }
    expect(anyNonZero, "corrupted trace has nonzero quotient")
}

private func testQuotientChunking() {
    let air = Plonky3ArithmeticAIR(logTraceLength: 3)
    let engine = GPUPlonky3AIREngine(air: air)
    let quotient = [BbExt4](repeating: BbExt4(base: Bb(v: 1)), count: 12)
    let chunks = engine.chunkQuotient(quotient: quotient, numChunks: 3)
    expectEqual(chunks.count, 3, "3 chunks")
    expectEqual(chunks[0].count, 4, "chunk 0 has 4 elements")
    expectEqual(chunks[1].count, 4, "chunk 1 has 4 elements")
    expectEqual(chunks[2].count, 4, "chunk 2 has 4 elements")
}

private func testQuotientChunkingSingle() {
    let air = Plonky3FibonacciMultiAIR(logTraceLength: 3)
    let engine = GPUPlonky3AIREngine(air: air)
    let quotient = [BbExt4](repeating: BbExt4.one, count: 8)
    let chunks = engine.chunkQuotient(quotient: quotient, numChunks: 1)
    expectEqual(chunks.count, 1, "1 chunk")
    expectEqual(chunks[0].count, 8, "all elements in one chunk")
}

// MARK: - Trace Verification

private func testVerifyValidFibTrace() {
    let air = Plonky3FibonacciMultiAIR(logTraceLength: 4)
    let engine = GPUPlonky3AIREngine(air: air)
    let trace = air.generateMainTrace()
    let result = engine.verifyTrace(
        preprocessed: nil, main: trace, permutation: nil, challenges: [])
    expect(result == nil, "valid trace passes verification")
}

private func testVerifyInvalidFibTrace() {
    let air = Plonky3FibonacciMultiAIR(logTraceLength: 4)
    let engine = GPUPlonky3AIREngine(air: air)
    var trace = air.generateMainTrace()
    trace[0][5] = Bb(v: 12345)
    let result = engine.verifyTrace(
        preprocessed: nil, main: trace, permutation: nil, challenges: [])
    expect(result != nil, "invalid trace fails verification")
}

private func testVerifyColumnCountMismatch() {
    let air = Plonky3FibonacciMultiAIR(logTraceLength: 3)
    let engine = GPUPlonky3AIREngine(air: air)
    // Only provide 1 column when 2 are expected
    let badTrace: [[Bb]] = [[Bb](repeating: Bb.one, count: 8)]
    let result = engine.verifyTrace(
        preprocessed: nil, main: badTrace, permutation: nil, challenges: [])
    expect(result != nil, "wrong column count detected")
    expect(result!.contains("Expected 2"), "error mentions expected count")
}

private func testVerifyRowCountMismatch() {
    let air = Plonky3FibonacciMultiAIR(logTraceLength: 3)
    let engine = GPUPlonky3AIREngine(air: air)
    // Provide wrong row count
    let badTrace: [[Bb]] = [
        [Bb](repeating: Bb.one, count: 4),
        [Bb](repeating: Bb.one, count: 4)
    ]
    let result = engine.verifyTrace(
        preprocessed: nil, main: badTrace, permutation: nil, challenges: [])
    expect(result != nil, "wrong row count detected")
}

// MARK: - Batch Constraint Eval

private func testBatchEvalFibonacci() {
    let air = Plonky3FibonacciMultiAIR(logTraceLength: 3)
    let engine = GPUPlonky3AIREngine(air: air)
    let trace = air.generateMainTrace()
    let allEvals = engine.batchEvaluateConstraints(
        preprocessed: nil, main: trace, permutation: nil, challenges: [])

    expectEqual(allEvals.count, 8, "8 rows of evaluations")
    // Transition rows (0..6) should be zero on valid trace
    for row in 0..<7 {
        for ev in allEvals[row] {
            expect(ev.isZero, "batch eval row \(row) is zero")
        }
    }
    // Last row has no evals
    expectEqual(allEvals[7].count, 0, "last row has no evaluations")
}

private func testPointEvalFibonacci() {
    let air = Plonky3FibonacciMultiAIR(logTraceLength: 3)
    let engine = GPUPlonky3AIREngine(air: air)
    let trace = air.generateMainTrace()
    let alpha = BbExt4(base: Bb(v: 3))

    // Evaluate at row 0 (valid transition: should be zero)
    let eval = engine.evaluateAtPoint(
        preprocessedAtPoint: nil,
        mainAtPoint: (current: trace.map { $0[0] }, next: trace.map { $0[1] }),
        permutationAtPoint: nil,
        challenges: [],
        alpha: alpha)
    expect(eval.isZero, "point eval at valid row is zero")

    // Evaluate at a bad point
    let evalBad = engine.evaluateAtPoint(
        preprocessedAtPoint: nil,
        mainAtPoint: (current: [Bb(v: 1), Bb(v: 1)], next: [Bb(v: 99), Bb(v: 99)]),
        permutationAtPoint: nil,
        challenges: [],
        alpha: alpha)
    expect(!evalBad.isZero, "point eval at bad row is nonzero")
}

// MARK: - Full Proving Pipeline

private func testProveFibonacci() {
    let air = Plonky3FibonacciMultiAIR(logTraceLength: 4)
    let engine = GPUPlonky3AIREngine(air: air, logBlowup: 1, numQueries: 10, grindingBits: 0)
    let result = engine.prove()

    expectEqual(result.traceRows, 16, "16 trace rows")
    expectEqual(result.traceCols, 2, "2 trace cols")
    expectEqual(result.numConstraints, 2, "2 constraints")
    expect(result.preprocessedCommitment == nil, "no preprocessed commitment")
    expectEqual(result.mainCommitment.count, 8, "8-element Merkle root")
    expectEqual(result.quotientEvals.count, 16, "16 quotient evals")
    expect(result.constraintEvalTimeSeconds >= 0, "timing is non-negative")
    expect(result.traceGenTimeSeconds >= 0, "trace gen timing non-negative")
}

private func testProveRangeCheck() {
    let air = Plonky3RangeCheckAIR(logTraceLength: 3)
    let engine = GPUPlonky3AIREngine(air: air)
    let result = engine.prove()

    expectEqual(result.traceRows, 8, "8 rows")
    expect(result.preprocessedCommitment != nil, "has preprocessed commitment")
    expectEqual(result.preprocessedCommitment!.count, 8, "8-element root")
}

private func testProveArithmetic() {
    let air = Plonky3ArithmeticAIR(logTraceLength: 3)
    let engine = GPUPlonky3AIREngine(air: air)
    let result = engine.prove()

    expectEqual(result.traceRows, 8, "8 rows")
    expectEqual(result.traceCols, 6, "6 total cols")
    expectEqual(result.numConstraints, 2, "2 constraints")
    expect(result.preprocessedCommitment != nil, "has preprocessed commitment")
}

private func testProveLookup() {
    let air = Plonky3LookupAIR(logTraceLength: 3)
    let engine = GPUPlonky3AIREngine(air: air)
    let result = engine.prove()

    expectEqual(result.traceRows, 8, "8 rows")
    expectEqual(result.traceCols, 4, "4 total cols (2 main + 2 perm)")
}

// MARK: - Cross-Table Verifier

private func testCrossTableVerifierCreation() {
    let air1 = Plonky3FibonacciMultiAIR(logTraceLength: 3)
    let air2 = Plonky3RangeCheckAIR(logTraceLength: 3)
    let e1 = GPUPlonky3AIREngine(air: air1)
    let e2 = GPUPlonky3AIREngine(air: air2)
    let verifier = Plonky3CrossTableVerifier(engines: [e1, e2])
    expectEqual(verifier.engines.count, 2, "2 engines")
}

// MARK: - Edge Cases

private func testMinimalTrace() {
    // Minimum trace: logN = 2 (4 rows)
    let air = Plonky3FibonacciMultiAIR(logTraceLength: 2)
    let engine = GPUPlonky3AIREngine(air: air)
    let trace = air.generateMainTrace()
    let result = engine.verifyTrace(
        preprocessed: nil, main: trace, permutation: nil, challenges: [])
    expect(result == nil, "minimal trace verifies")
    expectEqual(air.traceLength, 4, "4 rows")
}

private func testLargeTrace() {
    // Larger trace: logN = 8 (256 rows)
    let air = Plonky3FibonacciMultiAIR(logTraceLength: 8)
    let engine = GPUPlonky3AIREngine(air: air)
    let trace = air.generateMainTrace()
    let result = engine.verifyTrace(
        preprocessed: nil, main: trace, permutation: nil, challenges: [])
    expect(result == nil, "large trace verifies")
    expectEqual(air.traceLength, 256, "256 rows")
}

private func testExtFieldChain() {
    // Chain of operations: ((a + b) * c - d)^2
    let a = BbExt4(c: (Bb(v: 3), Bb(v: 5), Bb(v: 7), Bb(v: 11)))
    let b = BbExt4(c: (Bb(v: 13), Bb(v: 17), Bb(v: 19), Bb(v: 23)))
    let c = BbExt4(c: (Bb(v: 2), Bb(v: 0), Bb(v: 0), Bb(v: 0)))
    let d = BbExt4(c: (Bb(v: 1), Bb(v: 0), Bb(v: 0), Bb(v: 0)))

    let sum = bbExt4Add(a, b)
    let prod = bbExt4Mul(sum, c)
    let diff = bbExt4Sub(prod, d)
    let sq = bbExt4Sqr(diff)

    // Verify: sq = diff * diff
    let manual = bbExt4Mul(diff, diff)
    expect(sq == manual, "chain: sqr matches mul self")

    // Verify: result is not zero (extremely unlikely with random-ish inputs)
    expect(!sq.isZero, "chain result is nonzero")
}
