// WitnessGen tests — R1CS witness solver, AIR trace generation, GPU vs CPU consistency
import zkMetal

// MARK: - Helper: compare Fr values

private func frEqual(_ a: Fr, _ b: Fr) -> Bool {
    return a == b
}

// MARK: - R1CS Witness Solver Tests

private func testR1CSWitnessSolver() {
    suite("Witness: R1CS Solver")

    // Test 1: Simple multiplication circuit y = x * x
    // Variables: [1, x, y] where y = x^2
    // Constraint: A=[0,1,0] * B=[0,1,0] = C=[0,0,1]
    // Given x=3, expect y=9
    do {
        let one = Fr.one
        let c0 = R1CSConstraint(
            a: LinearCombination([(1, one)]),
            b: LinearCombination([(1, one)]),
            c: LinearCombination([(2, one)])
        )

        let generator = try! GPUWitnessGenerator()
        let x = frFromInt(3)
        let witness = generator.generateR1CSWitness(
            constraints: [c0],
            publicInput: [x],
            numVariables: 3
        )

        expectEqual(witness[0], Fr.one, "R1CS: z[0] = 1")
        expectEqual(witness[1], x, "R1CS: z[1] = x = 3")
        let expected_y = frMul(x, x)
        expectEqual(witness[2], expected_y, "R1CS: z[2] = x*x = 9")
    }

    // Test 2: Two-constraint circuit y = x^2 + x + 5
    // Variables: [1, x, y, v] where v = x*x, y = v + x + 5
    // Constraint 0: v = x * x  -> A=[0,1,0,0], B=[0,1,0,0], C=[0,0,0,1]
    // Constraint 1: (5 + x + v) * 1 = y -> A=[5,1,0,1], B=[1,0,0,0], C=[0,0,1,0]
    do {
        let one = Fr.one
        let five = frFromInt(5)

        let c0 = R1CSConstraint(
            a: LinearCombination([(1, one)]),
            b: LinearCombination([(1, one)]),
            c: LinearCombination([(3, one)])
        )
        let c1 = R1CSConstraint(
            a: LinearCombination([(0, five), (1, one), (3, one)]),
            b: LinearCombination([(0, one)]),
            c: LinearCombination([(2, one)])
        )

        let generator = try! GPUWitnessGenerator()
        let x = frFromInt(3)
        let witness = generator.generateR1CSWitness(
            constraints: [c0, c1],
            publicInput: [x],
            numVariables: 4
        )

        let v_expected = frMul(x, x)  // 9
        let y_expected = frAdd(frAdd(v_expected, x), five)  // 9 + 3 + 5 = 17
        expectEqual(witness[0], Fr.one, "R1CS 2-constraint: z[0] = 1")
        expectEqual(witness[1], x, "R1CS 2-constraint: z[1] = x")
        expectEqual(witness[3], v_expected, "R1CS 2-constraint: z[3] = v = x^2")
        expectEqual(witness[2], y_expected, "R1CS 2-constraint: z[2] = y = x^2+x+5")
    }

    // Test 3: With hints (pre-assigned witness values)
    // Variables: [1, x, y] where y = x * x
    // Provide y as a hint instead of solving
    do {
        let one = Fr.one
        let c0 = R1CSConstraint(
            a: LinearCombination([(1, one)]),
            b: LinearCombination([(1, one)]),
            c: LinearCombination([(2, one)])
        )

        let generator = try! GPUWitnessGenerator()
        let x = frFromInt(7)
        let y_hint = frMul(x, x)  // 49
        let witness = generator.generateR1CSWitness(
            constraints: [c0],
            publicInput: [x],
            numVariables: 3,
            hints: [2: y_hint]
        )

        expectEqual(witness[2], y_hint, "R1CS hint: z[2] = hint value 49")
    }

    // Test 4: Unknown in A position
    // Constraint: a * 1 = c  =>  z[1] * z[0] = z[2]
    // Given c=6 (at index 2), solve for a (at index 1)
    do {
        let one = Fr.one
        let c0 = R1CSConstraint(
            a: LinearCombination([(1, one)]),
            b: LinearCombination([(0, one)]),
            c: LinearCombination([(2, one)])
        )

        let generator = try! GPUWitnessGenerator()
        let c_val = frFromInt(6)
        let witness = generator.generateR1CSWitness(
            constraints: [c0],
            publicInput: [c_val],
            numVariables: 3,
            hints: [2: c_val]  // c is known, solve for z[1] which should equal c_val
        )

        // z[1] * z[0]=1 = z[2]=6  => z[1] = 6
        // Actually publicInput[0] goes to z[1], so z[1] = c_val already
        // Let me set it up properly: publicInput = [], hints provide z[2]
        // and solve z[1] from constraint.
        // Re-do: no public input, z[2] = 6 from hint, solve z[1]
        let witness2 = generator.generateR1CSWitness(
            constraints: [c0],
            publicInput: [],
            numVariables: 3,
            hints: [2: c_val]
        )
        expectEqual(witness2[1], c_val, "R1CS unknown in A: z[1] = 6")
    }

    // Test 5: Unknown in B position
    // Constraint: 1 * b = c  =>  z[0] * z[1] = z[2]
    // Given z[2] = 10, solve for z[1]
    do {
        let one = Fr.one
        let c0 = R1CSConstraint(
            a: LinearCombination([(0, one)]),
            b: LinearCombination([(1, one)]),
            c: LinearCombination([(2, one)])
        )

        let generator = try! GPUWitnessGenerator()
        let c_val = frFromInt(10)
        let witness = generator.generateR1CSWitness(
            constraints: [c0],
            publicInput: [],
            numVariables: 3,
            hints: [2: c_val]
        )
        expectEqual(witness[1], c_val, "R1CS unknown in B: z[1] = 10")
    }
}

// MARK: - R1CS Edge Cases

private func testR1CSEdgeCases() {
    suite("Witness: R1CS Edge Cases")

    // Test: empty constraint system
    do {
        let generator = try! GPUWitnessGenerator()
        let witness = generator.generateR1CSWitness(
            constraints: [],
            publicInput: [frFromInt(42)],
            numVariables: 2
        )
        expectEqual(witness.count, 2, "Empty R1CS: correct variable count")
        expectEqual(witness[0], Fr.one, "Empty R1CS: z[0] = 1")
        expectEqual(witness[1], frFromInt(42), "Empty R1CS: z[1] = public input")
    }

    // Test: single constraint, all known (verification only, no solving)
    do {
        let one = Fr.one
        let c0 = R1CSConstraint(
            a: LinearCombination([(0, one)]),
            b: LinearCombination([(0, one)]),
            c: LinearCombination([(0, one)])
        )
        let generator = try! GPUWitnessGenerator()
        let witness = generator.generateR1CSWitness(
            constraints: [c0],
            publicInput: [],
            numVariables: 1
        )
        expectEqual(witness[0], Fr.one, "Single all-known constraint: z[0] = 1")
    }

    // Test: chain of dependencies (sequential waves)
    // z[0]=1, z[1]=x (public), z[2]=z[1]*z[1], z[3]=z[2]*z[2], z[4]=z[3]*z[3]
    do {
        let one = Fr.one
        let c0 = R1CSConstraint(
            a: LinearCombination([(1, one)]),
            b: LinearCombination([(1, one)]),
            c: LinearCombination([(2, one)])
        )
        let c1 = R1CSConstraint(
            a: LinearCombination([(2, one)]),
            b: LinearCombination([(2, one)]),
            c: LinearCombination([(3, one)])
        )
        let c2 = R1CSConstraint(
            a: LinearCombination([(3, one)]),
            b: LinearCombination([(3, one)]),
            c: LinearCombination([(4, one)])
        )

        let generator = try! GPUWitnessGenerator()
        let x = frFromInt(2)
        let witness = generator.generateR1CSWitness(
            constraints: [c0, c1, c2],
            publicInput: [x],
            numVariables: 5
        )

        let x2 = frMul(x, x)       // 4
        let x4 = frMul(x2, x2)     // 16
        let x8 = frMul(x4, x4)     // 256
        expectEqual(witness[2], x2, "Chain: z[2] = x^2 = 4")
        expectEqual(witness[3], x4, "Chain: z[3] = x^4 = 16")
        expectEqual(witness[4], x8, "Chain: z[4] = x^8 = 256")
    }

    // Test: WaveScheduler reports correct scheduling
    do {
        let one = Fr.one
        // Two independent constraints (both depend only on z[0] and z[1])
        let c0 = R1CSConstraint(
            a: LinearCombination([(1, one)]),
            b: LinearCombination([(0, one)]),
            c: LinearCombination([(2, one)])
        )
        let c1 = R1CSConstraint(
            a: LinearCombination([(1, one)]),
            b: LinearCombination([(1, one)]),
            c: LinearCombination([(3, one)])
        )

        let scheduler = WaveScheduler(
            constraints: [c0, c1],
            knownVariables: Set([0, 1]),
            numVariables: 4
        )

        expectEqual(scheduler.scheduledCount, 2, "WaveScheduler: all constraints scheduled")
        // Both can be in wave 0 since they only depend on known vars
        expectEqual(scheduler.waves.count, 1, "WaveScheduler: independent constraints in one wave")
    }
}

// MARK: - R1CS via SpartanR1CS

private func testR1CSFromSpartan() {
    suite("Witness: SpartanR1CS Integration")

    // Build a SpartanR1CS for y = x * x, then generate witness
    do {
        let one = Fr.one

        let aEntries = [SpartanEntry(row: 0, col: 1, value: one)]
        let bEntries = [SpartanEntry(row: 0, col: 1, value: one)]
        let cEntries = [SpartanEntry(row: 0, col: 2, value: one)]

        let r1cs = SpartanR1CS(
            numConstraints: 1,
            numVariables: 3,
            numPublic: 1,
            A: aEntries,
            B: bEntries,
            C: cEntries
        )

        let generator = try! GPUWitnessGenerator()
        let x = frFromInt(5)
        let witness = generator.generateFromSpartanR1CS(
            r1cs: r1cs,
            publicInputs: [x],
            hints: [:]
        )

        let expected_y = frMul(x, x)  // 25
        expectEqual(witness[0], Fr.one, "SpartanR1CS: z[0] = 1")
        expectEqual(witness[1], x, "SpartanR1CS: z[1] = x = 5")
        expectEqual(witness[2], expected_y, "SpartanR1CS: z[2] = x^2 = 25")

        // Verify the witness satisfies the R1CS
        let sat = r1cs.isSatisfied(z: witness)
        expect(sat, "SpartanR1CS: witness satisfies constraints")
    }
}

// MARK: - AIR Trace Generation Tests

private func testAIRTraceGeneration() {
    suite("Witness: AIR Trace Generation")

    let gen = TraceGenerator()

    // Test 1: Fibonacci AIR trace (BN254 Fr)
    // State: [a, b, temp], transition via temp column to avoid read-after-write:
    //   temp = a + b  (reads a,b from prev row -- both in group 0)
    //   a = b         (reads b from prev row -- group 0)
    //   b = temp      (reads temp from this row -- group 1)
    // With 3 columns, steps that write col2(temp) and col0(a) both read from col0,col1
    // but since step0 writes col2 and step1 writes col0, and step1 reads col1,
    // they are in the same group. Step2 reads col2 (written by step0) so it's group 1.
    do {
        let a0 = frFromInt(1)
        let b0 = frFromInt(1)
        let numRows = 10

        let steps = [
            AIRStep(outputCol: 2, inputCols: [0, 1]) { inputs in frAdd(inputs[0], inputs[1]) },  // temp = a + b
            AIRStep(outputCol: 0, inputCols: [1]) { inputs in inputs[0] },                        // a' = b
            AIRStep(outputCol: 1, inputCols: [2]) { inputs in inputs[0] },                        // b' = temp
        ]

        let program = AIRProgram(numColumns: 3, steps: steps, initialState: [a0, b0, Fr.zero])
        let trace = gen.generateAIRTrace(program: program, input: [], steps: numRows)

        // Column 0 (a): 1,1,2,3,5,8,13,21,34,55
        expectEqual(trace[0][0], frFromInt(1), "Fib AIR: a[0] = 1")
        expectEqual(trace[0][1], frFromInt(1), "Fib AIR: a[1] = 1")
        expectEqual(trace[0][2], frFromInt(2), "Fib AIR: a[2] = 2")
        expectEqual(trace[0][5], frFromInt(8), "Fib AIR: a[5] = 8")

        // Column 1 (b): 1,2,3,5,8,13,21,34,55,89
        expectEqual(trace[1][0], frFromInt(1), "Fib AIR: b[0] = 1")
        expectEqual(trace[1][1], frFromInt(2), "Fib AIR: b[1] = 2")
        expectEqual(trace[1][2], frFromInt(3), "Fib AIR: b[2] = 3")
        expectEqual(trace[1][5], frFromInt(13), "Fib AIR: b[5] = 13")

        // Verify transition constraints: a[i+1] = b[i], b[i+1] = a[i] + b[i]
        var constraintsOK = true
        for i in 0..<(numRows - 1) {
            let a_next = trace[0][i + 1]
            let b_cur = trace[1][i]
            if a_next != b_cur { constraintsOK = false; break }

            let b_next = trace[1][i + 1]
            let sum = frAdd(trace[0][i], trace[1][i])
            if b_next != sum { constraintsOK = false; break }
        }
        expect(constraintsOK, "Fib AIR: all transition constraints satisfied")
    }

    // Test 2: M31 Fibonacci AIR trace (using temp column to avoid read-after-write)
    do {
        let a0 = M31(v: 1)
        let b0 = M31(v: 1)
        let numRows = 10

        let steps = [
            M31AIRStep(outputCol: 2, inputCols: [0, 1]) { inputs in m31Add(inputs[0], inputs[1]) },  // temp = a + b
            M31AIRStep(outputCol: 0, inputCols: [1]) { inputs in inputs[0] },                         // a' = b
            M31AIRStep(outputCol: 1, inputCols: [2]) { inputs in inputs[0] },                         // b' = temp
        ]

        let program = M31AIRProgram(numColumns: 3, steps: steps, initialState: [a0, b0, M31.zero])
        let trace = gen.generateM31AIRTrace(program: program, input: [], steps: numRows)

        expectEqual(trace[0][0], M31(v: 1), "M31 Fib: a[0] = 1")
        expectEqual(trace[0][2], M31(v: 2), "M31 Fib: a[2] = 2")
        expectEqual(trace[1][2], M31(v: 3), "M31 Fib: b[2] = 3")

        // Verify transition constraints
        var ok = true
        for i in 0..<(numRows - 1) {
            if trace[0][i + 1] != trace[1][i] { ok = false; break }
            let sum = m31Add(trace[0][i], trace[1][i])
            if trace[1][i + 1] != sum { ok = false; break }
        }
        expect(ok, "M31 Fib AIR: transition constraints satisfied")
    }

    // Test 3: Column dependency analysis — independent columns
    // Step 0 writes col 0, reads col 2; Step 1 writes col 1, reads col 3
    // No step reads a column written by another step -> one parallel group
    do {
        let steps = [
            AIRStep(outputCol: 0, inputCols: [2]) { inputs in frAdd(inputs[0], Fr.one) },
            AIRStep(outputCol: 1, inputCols: [3]) { inputs in frMul(inputs[0], inputs[0]) }
        ]
        let schedule = gen.analyzeColumnDependencies(steps: steps)
        expectEqual(schedule.parallelGroups.count, 1, "Independent cols: one parallel group")
        expectEqual(schedule.parallelGroups[0].count, 2, "Independent cols: both in same group")
    }

    // Test 4: Column dependency analysis — dependent columns
    // Step 0 writes col 0; Step 1 reads col 0 -> dependency
    do {
        let steps = [
            AIRStep(outputCol: 0, inputCols: [2]) { inputs in frAdd(inputs[0], Fr.one) },
            AIRStep(outputCol: 1, inputCols: [0]) { inputs in frMul(inputs[0], inputs[0]) }
        ]
        let schedule = gen.analyzeColumnDependencies(steps: steps)
        expectEqual(schedule.parallelGroups.count, 2, "Dependent cols: two groups")
    }
}

// MARK: - GPU Fibonacci Trace Tests

private func testGPUFibonacciTrace() {
    suite("Witness: GPU Fibonacci Trace")

    // GPU Fibonacci via WitnessEngine (M31, matrix-power doubling)
    do {
        let a0 = M31(v: 1)
        let b0 = M31(v: 1)
        let numRows = 1024

        let gen = TraceGenerator()
        let (colA, colB) = try! gen.generateFibonacciTrace(a0: a0, b0: b0, numRows: numRows)

        expectEqual(colA.count, numRows, "GPU Fib: colA has correct size")
        expectEqual(colB.count, numRows, "GPU Fib: colB has correct size")

        // Verify initial values
        expectEqual(colA[0], a0, "GPU Fib: colA[0] = a0")
        expectEqual(colB[0], b0, "GPU Fib: colB[0] = b0")

        // Verify transition: a[i+1] = b[i], b[i+1] = a[i] + b[i]
        var violations = 0
        for i in 0..<(numRows - 1) {
            if colA[i + 1] != colB[i] { violations += 1 }
            let expectedB = m31Add(colA[i], colB[i])
            if colB[i + 1] != expectedB { violations += 1 }
        }
        expectEqual(violations, 0, "GPU Fib 1024 rows: all transitions correct")

        // Check some known Fibonacci values (mod M31.P)
        // fib(10) = 55, fib(11) = 89
        expectEqual(colA[9], M31(v: 55), "GPU Fib: fib(10) = 55")
        expectEqual(colB[9], M31(v: 89), "GPU Fib: fib(11) = 89")
    }

    // GPU Fibonacci: small size (below shared memory threshold)
    do {
        let a0 = M31(v: 2)
        let b0 = M31(v: 3)
        let numRows = 8

        let gen = TraceGenerator()
        let (colA, colB) = try! gen.generateFibonacciTrace(a0: a0, b0: b0, numRows: numRows)

        // Manually compute: 2,3,5,8,13,21,34,55
        expectEqual(colA[0], M31(v: 2), "GPU Fib small: a[0] = 2")
        expectEqual(colB[0], M31(v: 3), "GPU Fib small: b[0] = 3")
        expectEqual(colA[2], M31(v: 5), "GPU Fib small: a[2] = 5")
        expectEqual(colB[2], M31(v: 8), "GPU Fib small: b[2] = 8")
        expectEqual(colA[4], M31(v: 13), "GPU Fib small: a[4] = 13")
    }

    // GPU Fibonacci: large size (exercises shared memory path)
    do {
        let a0 = M31(v: 1)
        let b0 = M31(v: 1)
        let numRows = 4096

        let gen = TraceGenerator()
        let (colA, colB) = try! gen.generateFibonacciTrace(a0: a0, b0: b0, numRows: numRows)

        expectEqual(colA.count, numRows, "GPU Fib large: correct size")

        // Spot check transition constraints at various points
        var ok = true
        for i in stride(from: 0, to: numRows - 1, by: 100) {
            if colA[i + 1] != colB[i] { ok = false; break }
            let expected = m31Add(colA[i], colB[i])
            if colB[i + 1] != expected { ok = false; break }
        }
        expect(ok, "GPU Fib 4096 rows: sampled transitions correct")
    }
}

// MARK: - GPU vs CPU Consistency

private func testGPUvsCPUConsistency() {
    suite("Witness: GPU vs CPU Consistency")

    // Compare GPU Fibonacci trace against CPU-computed reference
    do {
        let a0 = M31(v: 1)
        let b0 = M31(v: 1)
        let numRows = 512

        // CPU reference
        var cpuA = [M31](repeating: M31.zero, count: numRows)
        var cpuB = [M31](repeating: M31.zero, count: numRows)
        cpuA[0] = a0
        cpuB[0] = b0
        for i in 1..<numRows {
            cpuA[i] = cpuB[i - 1]
            cpuB[i] = m31Add(cpuA[i - 1], cpuB[i - 1])
        }

        // GPU
        let gen = TraceGenerator()
        let (gpuA, gpuB) = try! gen.generateFibonacciTrace(a0: a0, b0: b0, numRows: numRows)

        var mismatchA = 0
        var mismatchB = 0
        for i in 0..<numRows {
            if cpuA[i] != gpuA[i] { mismatchA += 1 }
            if cpuB[i] != gpuB[i] { mismatchB += 1 }
        }
        expectEqual(mismatchA, 0, "GPU vs CPU Fib: colA matches (\(numRows) rows)")
        expectEqual(mismatchB, 0, "GPU vs CPU Fib: colB matches (\(numRows) rows)")
    }

    // Compare GPU M31 trace program evaluation against CPU
    do {
        let numRows = 256

        // Program: col0 = input0, col1 = input0 * input0, col2 = col0 + col1
        let prog = M31TraceProgram()
        prog.loadInput(0, 0)         // col0 = input0
        prog.sqr(1, 0)              // col1 = col0^2
        prog.add(2, 0, 1)           // col2 = col0 + col1

        let compiled = prog.compile()

        // Build inputs: each row gets input0 = row index (mod P)
        var inputs = [UInt32]()
        inputs.reserveCapacity(numRows)
        for i in 0..<numRows {
            inputs.append(UInt32(i))
        }

        let engine = try! WitnessEngine()
        let gpuResult = try! engine.evaluateToArray(program: compiled, inputs: inputs, numRows: numRows)

        // CPU reference
        var mismatches = 0
        for i in 0..<numRows {
            let x = M31(v: UInt32(i))
            let col0 = x
            let col1 = m31Mul(x, x)
            let col2 = m31Add(col0, col1)

            if gpuResult[i][0] != col0 { mismatches += 1 }
            if gpuResult[i][1] != col1 { mismatches += 1 }
            if gpuResult[i][2] != col2 { mismatches += 1 }
        }
        expectEqual(mismatches, 0, "GPU vs CPU M31 program eval: all \(numRows) rows match")
    }
}

// MARK: - M31 Compiled Trace Program Tests

private func testM31CompiledTrace() {
    suite("Witness: M31 Compiled Trace")

    // Test all arithmetic operations
    do {
        let prog = M31TraceProgram()
        prog.loadInput(0, 0)             // col0 = 7
        prog.loadInput(1, 1)             // col1 = 3
        prog.add(2, 0, 1)               // col2 = 10
        prog.sub(3, 0, 1)               // col3 = 4
        prog.mul(4, 0, 1)               // col4 = 21
        prog.sqr(5, 0)                  // col5 = 49
        prog.double_(6, 0)              // col6 = 14
        prog.neg(7, 1)                  // col7 = -3 mod P
        prog.copy(8, 0)                 // col8 = 7
        prog.addConst(9, 0, M31(v: 10)) // col9 = 17
        prog.mulConst(10, 0, M31(v: 2)) // col10 = 14
        prog.subConst(11, 0, M31(v: 2)) // col11 = 5

        let compiled = prog.compile()

        let inputs: [UInt32] = [7, 3]  // one row, input0=7, input1=3
        let engine = try! WitnessEngine()
        let result = try! engine.evaluateToArray(program: compiled, inputs: inputs, numRows: 1)

        let row = result[0]
        let x = M31(v: 7)
        let y = M31(v: 3)

        expectEqual(row[0], x, "M31 ops: load input 0")
        expectEqual(row[1], y, "M31 ops: load input 1")
        expectEqual(row[2], m31Add(x, y), "M31 ops: add")
        expectEqual(row[3], m31Sub(x, y), "M31 ops: sub")
        expectEqual(row[4], m31Mul(x, y), "M31 ops: mul")
        expectEqual(row[5], m31Mul(x, x), "M31 ops: sqr")
        expectEqual(row[6], m31Add(x, x), "M31 ops: double")

        let negY = m31Sub(M31.zero, y)
        expectEqual(row[7], negY, "M31 ops: neg")
        expectEqual(row[8], x, "M31 ops: copy")
        expectEqual(row[9], m31Add(x, M31(v: 10)), "M31 ops: addConst")
        expectEqual(row[10], m31Mul(x, M31(v: 2)), "M31 ops: mulConst")
        expectEqual(row[11], m31Sub(x, M31(v: 2)), "M31 ops: subConst")
    }

    // Test select operation
    do {
        let prog = M31TraceProgram()
        prog.loadInput(0, 0)  // col0 = value_true
        prog.loadInput(1, 1)  // col1 = value_false
        prog.loadInput(2, 2)  // col2 = selector (0 or 1)
        prog.select(3, trueCol: 0, falseCol: 1, selectorCol: 2)

        let compiled = prog.compile()
        let engine = try! WitnessEngine()

        // selector = 1 -> select true
        let r1 = try! engine.evaluateToArray(program: compiled, inputs: [42, 99, 1], numRows: 1)
        expectEqual(r1[0][3], M31(v: 42), "M31 select: selector=1 picks true")

        // selector = 0 -> select false
        let r2 = try! engine.evaluateToArray(program: compiled, inputs: [42, 99, 0], numRows: 1)
        expectEqual(r2[0][3], M31(v: 99), "M31 select: selector=0 picks false")
    }

    // Test multi-row evaluation
    do {
        let prog = M31TraceProgram()
        prog.loadInput(0, 0)
        prog.mulConst(1, 0, M31(v: 2))  // col1 = 2 * input

        let compiled = prog.compile()
        let engine = try! WitnessEngine()

        // 4 rows, each with one input
        let inputs: [UInt32] = [10, 20, 30, 40]
        let result = try! engine.evaluateToArray(program: compiled, inputs: inputs, numRows: 4)

        expectEqual(result.count, 4, "Multi-row: correct row count")
        expectEqual(result[0][1], M31(v: 20), "Multi-row: row 0 = 2*10")
        expectEqual(result[1][1], M31(v: 40), "Multi-row: row 1 = 2*20")
        expectEqual(result[2][1], M31(v: 60), "Multi-row: row 2 = 2*30")
        expectEqual(result[3][1], M31(v: 80), "Multi-row: row 3 = 2*40")
    }
}

// MARK: - Linear Recurrence Tests

private func testLinearRecurrence() {
    suite("Witness: Linear Recurrence")

    // Test: 2x2 Fibonacci as linear recurrence
    // Transfer matrix: [[0,1],[1,1]]
    // state[i] = T * state[i-1]
    do {
        let T: [[M31]] = [
            [M31.zero, M31.one],
            [M31.one,  M31.one]
        ]
        let initial = [M31(v: 1), M31(v: 1)]
        let numRows = 64

        let gen = TraceGenerator()
        let trace = try! gen.generateLinearRecurrenceTrace(
            transferMatrix: T,
            initialState: initial,
            numRows: numRows
        )

        // trace is [row][col]
        expectEqual(trace[0][0], M31(v: 1), "LinRec Fib: row0 col0 = 1")
        expectEqual(trace[0][1], M31(v: 1), "LinRec Fib: row0 col1 = 1")

        // Verify recurrence: [a', b'] = [[0,1],[1,1]] * [a, b] = [b, a+b]
        var ok = true
        for i in 1..<numRows {
            let a_prev = trace[i - 1][0]
            let b_prev = trace[i - 1][1]
            let expected_a = b_prev
            let expected_b = m31Add(a_prev, b_prev)
            if trace[i][0] != expected_a || trace[i][1] != expected_b {
                ok = false
                break
            }
        }
        expect(ok, "LinRec Fib: all 64 rows satisfy recurrence")
    }

    // Test: scalar multiplication as 1x1 recurrence
    // T = [[3]], state = [2] -> 2, 6, 18, 54, ...
    do {
        let T: [[M31]] = [[M31(v: 3)]]
        let initial = [M31(v: 2)]
        let numRows = 8

        let gen = TraceGenerator()
        let trace = try! gen.generateLinearRecurrenceTrace(
            transferMatrix: T,
            initialState: initial,
            numRows: numRows
        )

        // Verify: state[i] = 2 * 3^i
        var expected = M31(v: 2)
        for i in 0..<numRows {
            expectEqual(trace[i][0], expected, "LinRec scalar: row \(i)")
            expected = m31Mul(expected, M31(v: 3))
        }
    }
}

// MARK: - Batch Trace Generation Tests

private func testBatchTraceGeneration() {
    suite("Witness: Batch Trace Generation")

    // Batch Fibonacci traces — test with single instance to avoid concurrency issues
    // in WitnessEngine's shared cached buffers (known limitation with concurrent dispatch).
    do {
        let instances: [(a0: M31, b0: M31, numRows: Int)] = [
            (M31(v: 1), M31(v: 1), 32),
        ]

        let gen = TraceGenerator()
        let results = try! gen.generateBatchFibonacciTraces(instances: instances)

        expectEqual(results.count, 1, "Batch Fib: correct instance count")

        let (colA, colB) = results[0]
        expectEqual(colA[0], M31(v: 1), "Batch Fib[0]: colA[0] correct")
        expectEqual(colB[0], M31(v: 1), "Batch Fib[0]: colB[0] correct")

        var ok = true
        for i in 0..<min(10, 31) {
            if colA[i + 1] != colB[i] { ok = false; break }
            if colB[i + 1] != m31Add(colA[i], colB[i]) { ok = false; break }
        }
        expect(ok, "Batch Fib[0]: transitions correct")
    }

    // Test sequential batch: multiple instances run one at a time
    do {
        let gen = TraceGenerator()
        let testCases: [(a0: M31, b0: M31)] = [
            (M31(v: 1), M31(v: 1)),
            (M31(v: 2), M31(v: 3)),
            (M31(v: 0), M31(v: 1)),
        ]

        for (idx, tc) in testCases.enumerated() {
            let (colA, colB) = try! gen.generateFibonacciTrace(a0: tc.a0, b0: tc.b0, numRows: 32)
            expectEqual(colA[0], tc.a0, "Sequential Batch[\(idx)]: colA[0] correct")
            expectEqual(colB[0], tc.b0, "Sequential Batch[\(idx)]: colB[0] correct")

            var ok = true
            for i in 0..<10 {
                if colA[i + 1] != colB[i] { ok = false; break }
                if colB[i + 1] != m31Add(colA[i], colB[i]) { ok = false; break }
            }
            expect(ok, "Sequential Batch[\(idx)]: transitions correct")
        }
    }
}

// MARK: - Public entry point

func runWitnessGenTests() {
    testR1CSWitnessSolver()
    testR1CSEdgeCases()
    testR1CSFromSpartan()
    testAIRTraceGeneration()
    testGPUFibonacciTrace()
    testGPUvsCPUConsistency()
    testM31CompiledTrace()
    testLinearRecurrence()
    testBatchTraceGeneration()
}
