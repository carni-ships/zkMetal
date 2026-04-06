// GPUTraceGenerator Tests — validates GPU-accelerated STARK trace generation
//
// Tests trace generation, LDE extension, commitment, and common AIR patterns
// (Fibonacci, hash chain, state machine).

import zkMetal

public func runGPUTraceGeneratorTests() {
    suite("GPUTraceGenerator — Fibonacci Trace")
    testFibonacciTraceGeneration()
    testFibonacciTraceValues()
    testGPUFibonacciTracePadding()

    suite("GPUTraceGenerator — State Machine Trace")
    testStateMachineTrace()
    testStateMachineMultiColumn()

    suite("GPUTraceGenerator — Trace Extension (LDE)")
    testTraceExtensionSize()
    testTraceExtensionDeterminism()
    testTraceExtensionBlowup4()

    suite("GPUTraceGenerator — BabyBear Traces")
    testBbFibonacciTrace()
    testBbTraceExtension()

    suite("GPUTraceGenerator — TraceMatrix")
    testTraceMatrixSubscript()
    testTraceMatrixDimensions()

    suite("GPUTraceGenerator — AIR Patterns")
    testAIRPatternsFibonacci()
    testAIRPatternsStateMachine()
}

// MARK: - Fibonacci Trace Tests

func testFibonacciTraceGeneration() {
    do {
        let gen = try GPUTraceGenerator()
        let trace = gen.generateTrace(
            initialState: [Fr.one, Fr.one],
            transitionFn: AIRPatterns.fibonacci(),
            numSteps: 16
        )
        expect(trace.numColumns == 2, "Fibonacci trace should have 2 columns")
        expect(trace.numRows == 16, "16 steps -> 16 rows (already power of 2)")
    } catch {
        expect(false, "GPUTraceGenerator init failed: \(error)")
    }
}

func testFibonacciTraceValues() {
    do {
        let gen = try GPUTraceGenerator()
        let trace = gen.generateTrace(
            initialState: [Fr.one, Fr.one],
            transitionFn: AIRPatterns.fibonacci(),
            numSteps: 8
        )

        // Verify Fibonacci transition: row i+1 = (col1[i], col0[i] + col1[i])
        for row in 0..<7 {
            let a = trace[row, 0]
            let b = trace[row, 1]
            let nextA = trace[row + 1, 0]
            let nextB = trace[row + 1, 1]

            // next_a should equal b
            expect(frEqual(nextA, b),
                   "Row \(row+1) col0 should equal row \(row) col1")

            // next_b should equal a + b
            let expected = frAdd(a, b)
            expect(frEqual(nextB, expected),
                   "Row \(row+1) col1 should equal row \(row) col0 + col1")
        }
    } catch {
        expect(false, "Fibonacci trace failed: \(error)")
    }
}

func testGPUFibonacciTracePadding() {
    do {
        let gen = try GPUTraceGenerator()
        // 10 steps -> should pad to 16
        let trace = gen.generateTrace(
            initialState: [Fr.one, Fr.one],
            transitionFn: AIRPatterns.fibonacci(),
            numSteps: 10
        )
        expect(trace.numRows == 16, "10 steps should pad to 16 rows, got \(trace.numRows)")

        // Padding rows should replicate last valid state
        let lastValid0 = trace[9, 0]
        let lastValid1 = trace[9, 1]
        for row in 10..<16 {
            expect(frEqual(trace[row, 0], lastValid0),
                   "Padding row \(row) col0 should match last valid row")
            expect(frEqual(trace[row, 1], lastValid1),
                   "Padding row \(row) col1 should match last valid row")
        }
    } catch {
        expect(false, "Padding test failed: \(error)")
    }
}

// MARK: - State Machine Tests

func testStateMachineTrace() {
    do {
        let gen = try GPUTraceGenerator()
        // Simple counter: 1 column, next = current + 1
        let one = Fr.one
        let counter = AIRPatterns.stateMachine(numRegisters: 1) { state in
            [frAdd(state[0], one)]
        }

        let trace = gen.generateTrace(
            initialState: [Fr.zero],
            transitionFn: counter,
            numSteps: 8
        )

        expect(trace.numColumns == 1, "Counter has 1 column")
        expect(trace.numRows == 8, "8 steps")

        // Verify counting: row i should have value i
        for i in 0..<8 {
            let expected = frFromInt(UInt64(i))
            expect(frEqual(trace[i, 0], expected),
                   "Row \(i) should equal \(i)")
        }
    } catch {
        expect(false, "State machine test failed: \(error)")
    }
}

func testStateMachineMultiColumn() {
    do {
        let gen = try GPUTraceGenerator()
        // 3-column state: (x, y, z) -> (y, z, x+y+z)
        let transition = AIRPatterns.stateMachine(numRegisters: 3) { s in
            [s[1], s[2], frAdd(frAdd(s[0], s[1]), s[2])]
        }

        let one = Fr.one
        let two = frAdd(one, one)
        let three = frAdd(two, one)
        let trace = gen.generateTrace(
            initialState: [one, two, three],
            transitionFn: transition,
            numSteps: 4
        )

        expect(trace.numColumns == 3, "3 columns")
        expect(trace.numRows == 4, "4 rows")

        // Row 0: (1, 2, 3)
        expect(frEqual(trace[0, 0], one), "Row 0 col 0")
        expect(frEqual(trace[0, 1], two), "Row 0 col 1")
        expect(frEqual(trace[0, 2], three), "Row 0 col 2")

        // Row 1: (2, 3, 6)
        expect(frEqual(trace[1, 0], two), "Row 1 col 0")
        expect(frEqual(trace[1, 1], three), "Row 1 col 1")
        let six = frFromInt(6)
        expect(frEqual(trace[1, 2], six), "Row 1 col 2")
    } catch {
        expect(false, "Multi-column state machine failed: \(error)")
    }
}

// MARK: - Trace Extension Tests

func testTraceExtensionSize() {
    do {
        let gen = try GPUTraceGenerator()
        let trace = gen.generateTrace(
            initialState: [Fr.one, Fr.one],
            transitionFn: AIRPatterns.fibonacci(),
            numSteps: 256
        )

        let extended = try gen.extendTrace(trace: trace, blowupFactor: 2)
        expect(extended.numColumns == 2, "Extension preserves column count")
        expect(extended.numRows == 512, "2x blowup: 256 -> 512 rows")
    } catch {
        expect(false, "Trace extension size test failed: \(error)")
    }
}

func testTraceExtensionDeterminism() {
    do {
        let gen = try GPUTraceGenerator()
        let trace = gen.generateTrace(
            initialState: [Fr.one, Fr.one],
            transitionFn: AIRPatterns.fibonacci(),
            numSteps: 128
        )

        let ext1 = try gen.extendTrace(trace: trace, blowupFactor: 2)
        let ext2 = try gen.extendTrace(trace: trace, blowupFactor: 2)

        // Same input -> same output
        var match = true
        for col in 0..<ext1.numColumns {
            for row in 0..<ext1.numRows {
                if !frEqual(ext1[row, col], ext2[row, col]) {
                    match = false
                    break
                }
            }
        }
        expect(match, "Trace extension should be deterministic")
    } catch {
        expect(false, "Determinism test failed: \(error)")
    }
}

func testTraceExtensionBlowup4() {
    do {
        let gen = try GPUTraceGenerator()
        let trace = gen.generateTrace(
            initialState: [Fr.one, Fr.one],
            transitionFn: AIRPatterns.fibonacci(),
            numSteps: 128
        )

        let extended = try gen.extendTrace(trace: trace, blowupFactor: 4)
        expect(extended.numRows == 512, "4x blowup: 128 -> 512 rows")

        // Extended trace should have non-zero values (not all zeros)
        var nonZeroCount = 0
        for row in 0..<min(64, extended.numRows) {
            if !frEqual(extended[row, 0], Fr.zero) {
                nonZeroCount += 1
            }
        }
        expect(nonZeroCount > 0, "Extended trace should contain non-zero values")
    } catch {
        expect(false, "Blowup 4 test failed: \(error)")
    }
}

// MARK: - BabyBear Trace Tests

func testBbFibonacciTrace() {
    do {
        let gen = try GPUTraceGenerator()
        let trace = gen.generateTraceBb(
            initialState: [Bb.one, Bb.one],
            transitionFn: AIRPatterns.fibonacciBb(),
            numSteps: 16
        )

        expect(trace.numColumns == 2, "Bb Fibonacci has 2 columns")
        expect(trace.numRows == 16, "16 rows")

        // Verify first few Fibonacci values
        // F(0)=1, F(1)=1, F(2)=2, F(3)=3, F(4)=5
        let expected: [UInt32] = [1, 1, 2, 3, 5, 8, 13, 21]
        for i in 0..<8 {
            // Column 0 holds "a" of the pair; for Fibonacci a[i] = F(i)
            // At row i: col0 = F(i), col1 = F(i+1)
            let colB = trace[i, 1]
            if i < 7 {
                expect(trace[i + 1, 0].v == colB.v,
                       "Bb Fib: next a should equal current b at row \(i)")
            }
        }
        // Check initial values
        expect(trace[0, 0].v == 1, "Bb Fib row 0 col 0 = 1")
        expect(trace[0, 1].v == 1, "Bb Fib row 0 col 1 = 1")
        expect(trace[1, 0].v == 1, "Bb Fib row 1 col 0 = 1")
        expect(trace[1, 1].v == 2, "Bb Fib row 1 col 1 = 2")
    } catch {
        expect(false, "Bb Fibonacci test failed: \(error)")
    }
}

func testBbTraceExtension() {
    do {
        let gen = try GPUTraceGenerator()
        let trace = gen.generateTraceBb(
            initialState: [Bb.one, Bb.one],
            transitionFn: AIRPatterns.fibonacciBb(),
            numSteps: 256
        )

        let extended = try gen.extendTraceBb(trace: trace, blowupFactor: 2)
        expect(extended.numColumns == 2, "Extension preserves columns")
        expect(extended.numRows == 512, "2x blowup: 256 -> 512")
    } catch {
        expect(false, "Bb trace extension failed: \(error)")
    }
}

// MARK: - TraceMatrix Tests

func testTraceMatrixSubscript() {
    let col0: [Fr] = [Fr.zero, Fr.one, frFromInt(2), frFromInt(3)]
    let col1: [Fr] = [frFromInt(10), frFromInt(11), frFromInt(12), frFromInt(13)]
    let matrix = TraceMatrix(columns: [col0, col1])

    expect(frEqual(matrix[0, 0], Fr.zero), "matrix[0,0] = 0")
    expect(frEqual(matrix[0, 1], frFromInt(10)), "matrix[0,1] = 10")
    expect(frEqual(matrix[2, 0], frFromInt(2)), "matrix[2,0] = 2")
    expect(frEqual(matrix[3, 1], frFromInt(13)), "matrix[3,1] = 13")
}

func testTraceMatrixDimensions() {
    let cols: [[Fr]] = [
        [Fr.zero, Fr.one, frFromInt(2), frFromInt(3)],
        [Fr.zero, Fr.one, frFromInt(2), frFromInt(3)],
        [Fr.zero, Fr.one, frFromInt(2), frFromInt(3)],
    ]
    let matrix = TraceMatrix(columns: cols)
    expect(matrix.numColumns == 3, "3 columns")
    expect(matrix.numRows == 4, "4 rows")
}

// MARK: - AIR Patterns Tests

func testAIRPatternsFibonacci() {
    let fib = AIRPatterns.fibonacci()
    expect(fib.numColumns == 2, "Fibonacci has 2 columns")

    let state0 = [Fr.one, Fr.one]
    let state1 = fib.apply(state0)
    // (1, 1) -> (1, 2)
    expect(frEqual(state1[0], Fr.one), "Fib(1,1) -> (1, _)")
    expect(frEqual(state1[1], frFromInt(2)), "Fib(1,1) -> (_, 2)")

    let state2 = fib.apply(state1)
    // (1, 2) -> (2, 3)
    expect(frEqual(state2[0], frFromInt(2)), "Fib(1,2) -> (2, _)")
    expect(frEqual(state2[1], frFromInt(3)), "Fib(1,2) -> (_, 3)")
}

func testAIRPatternsStateMachine() {
    let one = Fr.one
    let sm = AIRPatterns.stateMachine(numRegisters: 2) { state in
        // (a, b) -> (b, a * b)
        [state[1], frMul(state[0], state[1])]
    }

    expect(sm.numColumns == 2, "State machine has 2 columns")

    let s0 = [frFromInt(2), frFromInt(3)]
    let s1 = sm.apply(s0)
    // (2, 3) -> (3, 6)
    expect(frEqual(s1[0], frFromInt(3)), "SM: (2,3) -> (3, _)")
    expect(frEqual(s1[1], frFromInt(6)), "SM: (2,3) -> (_, 6)")
}
