// GPU Nova Decider Circuit Engine Tests — circuit synthesis, witness generation,
// satisfaction, proving, verification, SuperNova, diagnostics
import zkMetal

// MARK: - Test Helpers

/// Build a squaring circuit: w * w = y
/// Variables: z = [1, x, y, w] where x is public input, y is public output, w = x
/// Constraint: w * w = y => A=[0,0,0,1], B=[0,0,0,1], C=[0,0,1,0]
/// Public: x, y (numPublic=2), Witness: w (1 element)
private func makeDCSquaringR1CS() -> NovaR1CSShape {
    let m = 1, n = 4, numPublic = 2
    var aBuilder = SparseMatrixBuilder(rows: m, cols: n)
    aBuilder.set(row: 0, col: 3, value: Fr.one)

    var bBuilder = SparseMatrixBuilder(rows: m, cols: n)
    bBuilder.set(row: 0, col: 3, value: Fr.one)

    var cBuilder = SparseMatrixBuilder(rows: m, cols: n)
    cBuilder.set(row: 0, col: 2, value: Fr.one)

    return NovaR1CSShape(numConstraints: m, numVariables: n, numPublicInputs: numPublic,
                         A: aBuilder.build(), B: bBuilder.build(), C: cBuilder.build())
}

/// Create a valid squaring pair: x^2 = y
private func makeDCSquaringPair(_ val: UInt64) -> (NovaR1CSInput, NovaR1CSWitness) {
    let x = frFromInt(val)
    let y = frMul(x, x)
    return (NovaR1CSInput(x: [x, y]), NovaR1CSWitness(W: [x]))
}

/// Build a multiply circuit: a * b = c
/// Variables: z = [1, c, a, b] (numPublic=1, witness: a, b)
private func makeDCMultiplyR1CS() -> NovaR1CSShape {
    let m = 1, n = 4, numPublic = 1
    var aBuilder = SparseMatrixBuilder(rows: m, cols: n)
    aBuilder.set(row: 0, col: 2, value: Fr.one)

    var bBuilder = SparseMatrixBuilder(rows: m, cols: n)
    bBuilder.set(row: 0, col: 3, value: Fr.one)

    var cBuilder = SparseMatrixBuilder(rows: m, cols: n)
    cBuilder.set(row: 0, col: 1, value: Fr.one)

    return NovaR1CSShape(numConstraints: m, numVariables: n, numPublicInputs: numPublic,
                         A: aBuilder.build(), B: bBuilder.build(), C: cBuilder.build())
}

private func makeDCMultiplyPair(_ a: UInt64, _ b: UInt64) -> (NovaR1CSInput, NovaR1CSWitness) {
    let fa = frFromInt(a)
    let fb = frFromInt(b)
    let fc = frMul(fa, fb)
    return (NovaR1CSInput(x: [fc]), NovaR1CSWitness(W: [fa, fb]))
}

/// Build a 2-constraint circuit for testing larger circuits:
///   a * b = c   (row 0)
///   c * 1 = c   (row 1, identity)
/// Variables: z = [1, c, a, b] (numPublic=1, witness: a, b)
private func makeDCTwoConstraintR1CS() -> NovaR1CSShape {
    let m = 2, n = 4, numPublic = 1
    var aBuilder = SparseMatrixBuilder(rows: m, cols: n)
    aBuilder.set(row: 0, col: 2, value: Fr.one) // a
    aBuilder.set(row: 1, col: 1, value: Fr.one) // c

    var bBuilder = SparseMatrixBuilder(rows: m, cols: n)
    bBuilder.set(row: 0, col: 3, value: Fr.one) // b
    bBuilder.set(row: 1, col: 0, value: Fr.one) // 1

    var cBuilder = SparseMatrixBuilder(rows: m, cols: n)
    cBuilder.set(row: 0, col: 1, value: Fr.one) // c
    cBuilder.set(row: 1, col: 1, value: Fr.one) // c

    return NovaR1CSShape(numConstraints: m, numVariables: n, numPublicInputs: numPublic,
                         A: aBuilder.build(), B: bBuilder.build(), C: cBuilder.build())
}

private func makeDCTwoConstraintPair(_ a: UInt64, _ b: UInt64) -> (NovaR1CSInput, NovaR1CSWitness) {
    let fa = frFromInt(a)
    let fb = frFromInt(b)
    let fc = frMul(fa, fb)
    return (NovaR1CSInput(x: [fc]), NovaR1CSWitness(W: [fa, fb]))
}

/// Build a 3-constraint circuit with mixed arithmetic:
///   a * b = c     (row 0)
///   c + a = d     (row 1, encoded: (c+a)*1 = d)
///   d * 1 = d     (row 2, identity)
/// Variables: z = [1, d, a, b, c] (numPublic=1, witness: a, b, c)
private func makeDCThreeConstraintR1CS() -> NovaR1CSShape {
    let m = 3, n = 5, numPublic = 1
    var aBuilder = SparseMatrixBuilder(rows: m, cols: n)
    aBuilder.set(row: 0, col: 2, value: Fr.one) // a
    aBuilder.set(row: 1, col: 4, value: Fr.one) // c
    aBuilder.set(row: 1, col: 2, value: Fr.one) // + a
    aBuilder.set(row: 2, col: 1, value: Fr.one) // d

    var bBuilder = SparseMatrixBuilder(rows: m, cols: n)
    bBuilder.set(row: 0, col: 3, value: Fr.one) // b
    bBuilder.set(row: 1, col: 0, value: Fr.one) // 1
    bBuilder.set(row: 2, col: 0, value: Fr.one) // 1

    var cBuilder = SparseMatrixBuilder(rows: m, cols: n)
    cBuilder.set(row: 0, col: 4, value: Fr.one) // c
    cBuilder.set(row: 1, col: 1, value: Fr.one) // d
    cBuilder.set(row: 2, col: 1, value: Fr.one) // d

    return NovaR1CSShape(numConstraints: m, numVariables: n, numPublicInputs: numPublic,
                         A: aBuilder.build(), B: bBuilder.build(), C: cBuilder.build())
}

private func makeDCThreeConstraintPair(_ a: UInt64, _ b: UInt64) -> (NovaR1CSInput, NovaR1CSWitness) {
    let fa = frFromInt(a)
    let fb = frFromInt(b)
    let fc = frMul(fa, fb)
    let fd = frAdd(fc, fa)
    return (NovaR1CSInput(x: [fd]), NovaR1CSWitness(W: [fa, fb, fc]))
}

// MARK: - Tests

public func runGPUNovaDeciderCircuitTests() {
    suite("GPUNovaDeciderCircuit")

    // =========================================================================
    // Test 1: DeciderCircuitConfig defaults
    // =========================================================================
    do {
        let config = DeciderCircuitConfig()
        expect(config.maxConstraints == 4096, "Default maxConstraints should be 4096")
        expect(config.maxVariables == 8192, "Default maxVariables should be 8192")
        expect(config.includeNIFSCheck == true, "Default includeNIFSCheck should be true")
        expect(config.includeCommitmentCheck == true, "Default includeCommitmentCheck")
        expect(config.useGPU == true, "Default useGPU should be true")
        expect(config.gpuThreshold == 512, "Default gpuThreshold should be 512")
        expect(config.scalarBits == 254, "Default scalarBits should be 254")
        expect(config.isSuperNova == false, "Default isSuperNova should be false")
        expect(config.numCircuitTypes == 1, "Default numCircuitTypes should be 1")
    }

    // =========================================================================
    // Test 2: DeciderCircuitConfig custom values
    // =========================================================================
    do {
        let config = DeciderCircuitConfig(
            maxConstraints: 1024,
            maxVariables: 2048,
            includeNIFSCheck: false,
            includeCommitmentCheck: false,
            useGPU: false,
            gpuThreshold: 256,
            scalarBits: 128,
            isSuperNova: true,
            numCircuitTypes: 3)
        expect(config.maxConstraints == 1024, "Custom maxConstraints")
        expect(config.useGPU == false, "Custom useGPU")
        expect(config.isSuperNova == true, "Custom isSuperNova")
        expect(config.numCircuitTypes == 3, "Custom numCircuitTypes")
    }

    // =========================================================================
    // Test 3: Wire reference basics
    // =========================================================================
    do {
        let wire = DeciderCircuitWire(index: 5, label: "test")
        expect(wire.index == 5, "Wire index should be 5")
        expect(wire.label == "test", "Wire label should be 'test'")

        let oneWire = DeciderCircuitWire.one
        expect(oneWire.index == 0, "One wire index should be 0")
    }

    // =========================================================================
    // Test 4: Constraint set variable allocation
    // =========================================================================
    do {
        var cs = DeciderCircuitConstraintSet(numPublicInputs: 2)
        // Initial state: z = [1, pub0, pub1] => nextFreeVar = 3
        expect(cs.numVariables == 3, "Initial numVariables = 1 + 2 pub")
        expect(cs.numPublicInputs == 2, "numPublicInputs should be 2")

        let w0 = cs.allocWitness(label: "w0")
        expect(w0.index == 3, "First witness at index 3")
        expect(cs.numVariables == 4, "After 1 alloc, numVariables = 4")

        let block = cs.allocWitnessBlock(count: 3, prefix: "blk")
        expect(block.count == 3, "Block should have 3 wires")
        expect(block[0].index == 4, "Block[0] at index 4")
        expect(block[1].index == 5, "Block[1] at index 5")
        expect(block[2].index == 6, "Block[2] at index 6")
        expect(cs.numVariables == 7, "After block alloc, numVariables = 7")
    }

    // =========================================================================
    // Test 5: Constraint set — add and build shape
    // =========================================================================
    do {
        var cs = DeciderCircuitConstraintSet(numPublicInputs: 1)
        let a = cs.allocWitness(label: "a")
        let b = cs.allocWitness(label: "b")
        let c = cs.allocWitness(label: "c")

        // a * b = c
        cs.addMulConstraint(a, b, c)
        expect(cs.constraints.count == 1, "Should have 1 constraint")

        let shape = cs.buildShape()
        expect(shape.numConstraints == 1, "Shape should have 1 constraint")
        expect(shape.numVariables == 5, "Shape: 1 + 1pub + 3wit = 5 vars")
        expect(shape.numPublicInputs == 1, "Shape: 1 public input")

        // Verify satisfaction: a=3, b=5, c=15, pub=42
        let pub = frFromInt(42)
        let inst = NovaR1CSInput(x: [pub])
        let wit = NovaR1CSWitness(W: [frFromInt(3), frFromInt(5), frFromInt(15)])
        let sat = shape.satisfies(instance: inst, witness: wit)
        expect(sat, "Shape should be satisfied with a=3, b=5, c=15")
    }

    // =========================================================================
    // Test 6: Constraint set — addition constraint
    // =========================================================================
    do {
        var cs = DeciderCircuitConstraintSet(numPublicInputs: 0)
        let a = cs.allocWitness(label: "a")
        let b = cs.allocWitness(label: "b")
        let c = cs.allocWitness(label: "c")

        // a + b = c
        cs.addAddConstraint(a, b, c)
        let shape = cs.buildShape()

        // z = [1, a, b, c] with a=7, b=11, c=18
        let inst = NovaR1CSInput(x: [])
        let wit = NovaR1CSWitness(W: [frFromInt(7), frFromInt(11), frFromInt(18)])
        expect(shape.satisfies(instance: inst, witness: wit),
               "Addition constraint: 7 + 11 = 18")

        // Should fail with wrong c
        let witBad = NovaR1CSWitness(W: [frFromInt(7), frFromInt(11), frFromInt(19)])
        expect(!shape.satisfies(instance: inst, witness: witBad),
               "Addition constraint should fail: 7 + 11 != 19")
    }

    // =========================================================================
    // Test 7: Constraint set — linear constraint
    // =========================================================================
    do {
        var cs = DeciderCircuitConstraintSet(numPublicInputs: 0)
        let a = cs.allocWitness(label: "a")
        let b = cs.allocWitness(label: "b")

        // 3 * a = b
        cs.addLinearConstraint(coeff: frFromInt(3), a, b)
        let shape = cs.buildShape()

        let inst = NovaR1CSInput(x: [])
        let wit = NovaR1CSWitness(W: [frFromInt(5), frFromInt(15)])
        expect(shape.satisfies(instance: inst, witness: wit),
               "Linear constraint: 3 * 5 = 15")
    }

    // =========================================================================
    // Test 8: Constraint set — FMA constraint
    // =========================================================================
    do {
        var cs = DeciderCircuitConstraintSet(numPublicInputs: 0)
        let a = cs.allocWitness(label: "a")
        let b = cs.allocWitness(label: "b")
        let c = cs.allocWitness(label: "c")
        let d = cs.allocWitness(label: "d")

        // a * b + c = d
        cs.addFMAConstraint(a, b, c, d)
        let shape = cs.buildShape()

        // 3 * 5 + 7 = 22
        let inst = NovaR1CSInput(x: [])
        let wit = NovaR1CSWitness(W: [frFromInt(3), frFromInt(5), frFromInt(7), frFromInt(22)])
        expect(shape.satisfies(instance: inst, witness: wit),
               "FMA constraint: 3*5+7 = 22")
    }

    // =========================================================================
    // Test 9: Engine initialization with default config
    // =========================================================================
    do {
        let shape = makeDCSquaringR1CS()
        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape)
        expect(engine.innerShape.numConstraints == 1, "Inner shape has 1 constraint")
        expect(engine.innerShape.numPublicInputs == 2, "Inner shape has 2 public inputs")
    }

    // =========================================================================
    // Test 10: Engine initialization with CPU-only config
    // =========================================================================
    do {
        let shape = makeDCSquaringR1CS()
        let config = DeciderCircuitConfig(useGPU: false)
        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape, config: config)
        expect(!engine.gpuAvailable, "CPU-only engine should not have GPU")
    }

    // =========================================================================
    // Test 11: Circuit synthesis — shape dimensions
    // =========================================================================
    do {
        let shape = makeDCSquaringR1CS()
        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape)
        let (circuitShape, layout) = engine.synthesizeCircuit()

        // The decider circuit should have more constraints than the inner circuit
        expect(circuitShape.numConstraints > shape.numConstraints,
               "Decider circuit should have more constraints than inner")

        // Public inputs: [stateHash, u, x[0], x[1]] = 4
        expect(circuitShape.numPublicInputs == 4,
               "Decider circuit public inputs: stateHash + u + 2 inner x = 4")

        expect(layout.totalWitnessVars > 0, "Layout should have witness variables")
        expect(layout.errorWires.count == 1, "Error wires count = inner constraints = 1")
        expect(layout.azWires.count == 1, "Az wires count = inner constraints = 1")
        expect(layout.innerWitnessWires.count == 1, "Inner witness wires = 1")
    }

    // =========================================================================
    // Test 12: Circuit synthesis — multiply circuit shape
    // =========================================================================
    do {
        let shape = makeDCMultiplyR1CS()
        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape)
        let (circuitShape, layout) = engine.synthesizeCircuit()

        // Public inputs: [stateHash, u, x[0]] = 3
        expect(circuitShape.numPublicInputs == 3,
               "Multiply decider circuit: 3 public inputs")

        expect(layout.innerWitnessWires.count == 2, "Multiply inner witness: a, b => 2 wires")
        expect(layout.publicInputWires.count == 1, "Multiply inner public: c => 1 wire")
    }

    // =========================================================================
    // Test 13: Circuit synthesis — two-constraint circuit shape
    // =========================================================================
    do {
        let shape = makeDCTwoConstraintR1CS()
        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape)
        let (circuitShape, layout) = engine.synthesizeCircuit()

        expect(layout.errorWires.count == 2, "Two-constraint inner => 2 error wires")
        expect(layout.azWires.count == 2, "Two-constraint inner => 2 Az wires")
        expect(layout.abProductWires.count == 2, "Two-constraint inner => 2 AB product wires")

        expect(circuitShape.numConstraints > 2,
               "Decider circuit constraints > inner constraints")
    }

    // =========================================================================
    // Test 14: Witness generation — base case (u=1, E=0)
    // =========================================================================
    do {
        let shape = makeDCSquaringR1CS()
        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape)

        let (inst, wit) = makeDCSquaringPair(3)
        let pp = PedersenParams.generate(size: max(shape.numWitness, shape.numConstraints))
        let (relaxedInst, relaxedWit) = shape.relax(instance: inst, witness: wit, pp: pp)

        let (pubInput, witVec) = engine.generateWitness(
            instance: relaxedInst, witness: relaxedWit)

        // Public input should have stateHash, u, x[0], x[1]
        expect(pubInput.count == 4, "Public input length = 4")
        expect(frEq(pubInput[1], Fr.one), "u should be 1 for base case")
        expect(frEq(pubInput[2], frFromInt(3)), "x[0] should be 3")
        expect(frEq(pubInput[3], frMul(frFromInt(3), frFromInt(3))), "x[1] should be 9")

        expect(witVec.count > 0, "Witness vector should be non-empty")
    }

    // =========================================================================
    // Test 15: Circuit satisfaction — base case squaring
    // =========================================================================
    do {
        let shape = makeDCSquaringR1CS()
        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape)

        let (inst, wit) = makeDCSquaringPair(5)
        let pp = PedersenParams.generate(size: max(shape.numWitness, shape.numConstraints))
        let (relaxedInst, relaxedWit) = shape.relax(instance: inst, witness: wit, pp: pp)

        let sat = engine.checkCircuitSatisfaction(
            instance: relaxedInst, witness: relaxedWit)
        expect(sat, "Decider circuit should be satisfiable for base case squaring (5^2=25)")
    }

    // =========================================================================
    // Test 16: Circuit satisfaction — base case multiply
    // =========================================================================
    do {
        let shape = makeDCMultiplyR1CS()
        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape)

        let (inst, wit) = makeDCMultiplyPair(3, 7)
        let pp = PedersenParams.generate(size: max(shape.numWitness, shape.numConstraints))
        let (relaxedInst, relaxedWit) = shape.relax(instance: inst, witness: wit, pp: pp)

        let sat = engine.checkCircuitSatisfaction(
            instance: relaxedInst, witness: relaxedWit)
        expect(sat, "Decider circuit should be satisfiable for multiply (3*7=21)")
    }

    // =========================================================================
    // Test 17: Circuit satisfaction — two-constraint circuit
    // =========================================================================
    do {
        let shape = makeDCTwoConstraintR1CS()
        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape)

        let (inst, wit) = makeDCTwoConstraintPair(3, 7)
        let pp = PedersenParams.generate(size: max(shape.numWitness, shape.numConstraints))
        let (relaxedInst, relaxedWit) = shape.relax(instance: inst, witness: wit, pp: pp)

        let sat = engine.checkCircuitSatisfaction(
            instance: relaxedInst, witness: relaxedWit)
        expect(sat, "Decider circuit should be satisfiable for two-constraint (3*7=21, 21*1=21)")
    }

    // =========================================================================
    // Test 18: Circuit satisfaction — three-constraint circuit
    // =========================================================================
    do {
        let shape = makeDCThreeConstraintR1CS()
        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape)

        let (inst, wit) = makeDCThreeConstraintPair(4, 5)
        let pp = PedersenParams.generate(size: max(shape.numWitness, shape.numConstraints))
        let (relaxedInst, relaxedWit) = shape.relax(instance: inst, witness: wit, pp: pp)

        let sat = engine.checkCircuitSatisfaction(
            instance: relaxedInst, witness: relaxedWit)
        expect(sat, "Three-constraint circuit: 4*5=20, 20+4=24, 24*1=24")
    }

    // =========================================================================
    // Test 19: Circuit satisfaction — folded instance (u != 1, E != 0)
    // =========================================================================
    do {
        let shape = makeDCSquaringR1CS()
        let foldEngine = GPUNovaFoldEngine(shape: shape)

        let (inst1, wit1) = makeDCSquaringPair(3)
        foldEngine.initialize(instance: inst1, witness: wit1)

        let (inst2, wit2) = makeDCSquaringPair(5)
        let _ = foldEngine.foldStep(newInstance: inst2, newWitness: wit2)

        let accInst = foldEngine.runningInstance!
        let accWit = foldEngine.runningWitness!

        // Verify the inner R1CS is satisfied first
        let innerOk = shape.satisfiesRelaxed(instance: accInst, witness: accWit)
        expect(innerOk, "Inner relaxed R1CS should be satisfied after folding")

        // Now check the decider circuit
        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape)
        let sat = engine.checkCircuitSatisfaction(instance: accInst, witness: accWit)
        expect(sat, "Decider circuit should be satisfiable for folded instance")
    }

    // =========================================================================
    // Test 20: Circuit satisfaction — multi-step folded instance
    // =========================================================================
    do {
        let shape = makeDCSquaringR1CS()
        let foldEngine = GPUNovaFoldEngine(shape: shape)

        var steps = [(instance: NovaR1CSInput, witness: NovaR1CSWitness)]()
        for i: UInt64 in 2...5 {
            steps.append(makeDCSquaringPair(i))
        }
        let (finalInst, finalWit) = foldEngine.ivcChain(steps: steps)

        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape)
        let sat = engine.checkCircuitSatisfaction(instance: finalInst, witness: finalWit)
        expect(sat, "Decider circuit should be satisfiable after 4-step fold")
    }

    // =========================================================================
    // Test 21: Diagnostic — base case passes
    // =========================================================================
    do {
        let shape = makeDCSquaringR1CS()
        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape)

        let (inst, wit) = makeDCSquaringPair(7)
        let pp = PedersenParams.generate(size: max(shape.numWitness, shape.numConstraints))
        let (relaxedInst, relaxedWit) = shape.relax(instance: inst, witness: wit, pp: pp)

        let (satisfied, failing, total) = engine.diagnoseCircuitSatisfaction(
            instance: relaxedInst, witness: relaxedWit)
        expect(satisfied, "Diagnostic should report satisfied")
        expect(failing.isEmpty, "No failing constraints for valid instance")
        expect(total > 0, "Total constraints should be positive")
    }

    // =========================================================================
    // Test 22: Circuit stats — squaring circuit
    // =========================================================================
    do {
        let shape = makeDCSquaringR1CS()
        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape)
        let stats = engine.circuitStats()

        expect(stats.constraints > 0, "Should have constraints")
        expect(stats.variables > 0, "Should have variables")
        expect(stats.publicInputs == 4, "Public inputs: stateHash + u + 2 inner x")
        expect(stats.witnessVars > 0, "Should have witness variables")
        expect(stats.variables == 1 + stats.publicInputs + stats.witnessVars,
               "variables = 1 + public + witness")
    }

    // =========================================================================
    // Test 23: Circuit stats — multiply circuit
    // =========================================================================
    do {
        let shape = makeDCMultiplyR1CS()
        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape)
        let stats = engine.circuitStats()

        expect(stats.publicInputs == 3, "Multiply: stateHash + u + 1 inner x = 3")
    }

    // =========================================================================
    // Test 24: State hash computation determinism
    // =========================================================================
    do {
        let shape = makeDCSquaringR1CS()
        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape)

        let (inst, wit) = makeDCSquaringPair(3)
        let pp = PedersenParams.generate(size: max(shape.numWitness, shape.numConstraints))
        let (relaxedInst, _) = shape.relax(instance: inst, witness: wit, pp: pp)

        let h1 = engine.computeStateHash(instance: relaxedInst)
        let h2 = engine.computeStateHash(instance: relaxedInst)
        expect(frEq(h1, h2), "State hash should be deterministic")
        expect(!h1.isZero, "State hash should be non-zero")
    }

    // =========================================================================
    // Test 25: Circuit hash computation
    // =========================================================================
    do {
        let shape = makeDCSquaringR1CS()
        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape)

        let (inst, wit) = makeDCSquaringPair(3)
        let pp = PedersenParams.generate(size: max(shape.numWitness, shape.numConstraints))
        let (relaxedInst, _) = shape.relax(instance: inst, witness: wit, pp: pp)

        let h1 = engine.computeCircuitHash(instance: relaxedInst, stepCount: 1)
        let h2 = engine.computeCircuitHash(instance: relaxedInst, stepCount: 2)
        expect(!frEq(h1, h2), "Circuit hash should differ for different step counts")
    }

    // =========================================================================
    // Test 26: GPU inner product helper
    // =========================================================================
    do {
        let shape = makeDCSquaringR1CS()
        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape)

        let a = [frFromInt(2), frFromInt(3), frFromInt(5)]
        let b = [frFromInt(7), frFromInt(11), frFromInt(13)]
        // Expected: 2*7 + 3*11 + 5*13 = 14 + 33 + 65 = 112
        let result = engine.gpuFieldInnerProduct(a, b)
        expect(frEq(result, frFromInt(112)), "Inner product: 2*7+3*11+5*13 = 112")
    }

    // =========================================================================
    // Test 27: GPU inner product — empty vectors
    // =========================================================================
    do {
        let shape = makeDCSquaringR1CS()
        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape)

        let result = engine.gpuFieldInnerProduct([], [])
        expect(result.isZero, "Inner product of empty vectors = 0")
    }

    // =========================================================================
    // Test 28: GPU inner product — single element
    // =========================================================================
    do {
        let shape = makeDCSquaringR1CS()
        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape)

        let result = engine.gpuFieldInnerProduct([frFromInt(6)], [frFromInt(7)])
        expect(frEq(result, frFromInt(42)), "Inner product of [6] . [7] = 42")
    }

    // =========================================================================
    // Test 29: Prove — base case squaring
    // =========================================================================
    do {
        let shape = makeDCSquaringR1CS()
        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape)

        let (inst, wit) = makeDCSquaringPair(3)
        let pp = PedersenParams.generate(size: max(shape.numWitness, shape.numConstraints))
        let (relaxedInst, relaxedWit) = shape.relax(instance: inst, witness: wit, pp: pp)

        let proof = engine.prove(instance: relaxedInst, witness: relaxedWit, stepCount: 1)
        expect(proof != nil, "Should produce a proof for valid base case")

        if let p = proof {
            expect(p.sumcheckRounds.count > 0, "Proof should have sumcheck rounds")
            expect(p.stepCount == 1, "Step count should be 1")
            expect(frEq(p.accumulatorU, Fr.one), "Base case u = 1")
            expect(p.accumulatorX.count == 2, "Squaring circuit has 2 public inputs")
            expect(!p.isSuperNova, "Should not be SuperNova")
            expect(p.circuitTypeIndex == 0, "Circuit type index should be 0")
        }
    }

    // =========================================================================
    // Test 30: Prove — base case multiply
    // =========================================================================
    do {
        let shape = makeDCMultiplyR1CS()
        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape)

        let (inst, wit) = makeDCMultiplyPair(7, 11)
        let pp = PedersenParams.generate(size: max(shape.numWitness, shape.numConstraints))
        let (relaxedInst, relaxedWit) = shape.relax(instance: inst, witness: wit, pp: pp)

        let proof = engine.prove(instance: relaxedInst, witness: relaxedWit, stepCount: 1)
        expect(proof != nil, "Should produce a proof for multiply (7*11=77)")

        if let p = proof {
            expect(p.accumulatorX.count == 1, "Multiply has 1 public input")
            expect(frEq(p.accumulatorX[0], frFromInt(77)), "Public input should be 77")
        }
    }

    // =========================================================================
    // Test 31: Prove — folded instance
    // =========================================================================
    do {
        let shape = makeDCSquaringR1CS()
        let foldEngine = GPUNovaFoldEngine(shape: shape)

        let (inst1, wit1) = makeDCSquaringPair(3)
        foldEngine.initialize(instance: inst1, witness: wit1)

        let (inst2, wit2) = makeDCSquaringPair(5)
        let _ = foldEngine.foldStep(newInstance: inst2, newWitness: wit2)

        let accInst = foldEngine.runningInstance!
        let accWit = foldEngine.runningWitness!

        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape)
        let proof = engine.prove(instance: accInst, witness: accWit, stepCount: 2)
        expect(proof != nil, "Should produce a proof for folded instance")

        if let p = proof {
            expect(p.stepCount == 2, "Step count should be 2")
            expect(!frEq(p.accumulatorU, Fr.one), "Folded u != 1")
        }
    }

    // =========================================================================
    // Test 32: Prove — multi-step folded instance
    // =========================================================================
    do {
        let shape = makeDCMultiplyR1CS()
        let foldEngine = GPUNovaFoldEngine(shape: shape)

        var steps = [(instance: NovaR1CSInput, witness: NovaR1CSWitness)]()
        steps.append(makeDCMultiplyPair(2, 3))
        steps.append(makeDCMultiplyPair(5, 7))
        steps.append(makeDCMultiplyPair(11, 13))
        let (finalInst, finalWit) = foldEngine.ivcChain(steps: steps)

        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape)
        let proof = engine.prove(instance: finalInst, witness: finalWit, stepCount: 3)
        expect(proof != nil, "Should produce a proof for 3-step folded multiply circuit")
    }

    // =========================================================================
    // Test 33: Prove — two-constraint circuit
    // =========================================================================
    do {
        let shape = makeDCTwoConstraintR1CS()
        let foldEngine = GPUNovaFoldEngine(shape: shape)

        var steps = [(instance: NovaR1CSInput, witness: NovaR1CSWitness)]()
        steps.append(makeDCTwoConstraintPair(3, 7))
        steps.append(makeDCTwoConstraintPair(5, 11))
        let (finalInst, finalWit) = foldEngine.ivcChain(steps: steps)

        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape)
        let proof = engine.prove(instance: finalInst, witness: finalWit, stepCount: 2)
        expect(proof != nil, "Should produce a proof for two-constraint circuit")
    }

    // =========================================================================
    // Test 34: Prove — three-constraint circuit
    // =========================================================================
    do {
        let shape = makeDCThreeConstraintR1CS()
        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape)

        let (inst, wit) = makeDCThreeConstraintPair(4, 5)
        let pp = PedersenParams.generate(size: max(shape.numWitness, shape.numConstraints))
        let (relaxedInst, relaxedWit) = shape.relax(instance: inst, witness: wit, pp: pp)

        let proof = engine.prove(instance: relaxedInst, witness: relaxedWit, stepCount: 1)
        expect(proof != nil, "Should produce a proof for three-constraint circuit")
    }

    // =========================================================================
    // Test 35: Circuit hash matches verifier expectation
    // =========================================================================
    do {
        let shape = makeDCSquaringR1CS()
        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape)

        let (inst, wit) = makeDCSquaringPair(3)
        let pp = PedersenParams.generate(size: max(shape.numWitness, shape.numConstraints))
        let (relaxedInst, relaxedWit) = shape.relax(instance: inst, witness: wit, pp: pp)

        let proof = engine.prove(instance: relaxedInst, witness: relaxedWit, stepCount: 5)
        if let p = proof {
            let verifier = DeciderCircuitVerifier()
            let expectedHash = verifier.computeExpectedCircuitHash(
                u: p.accumulatorU, x: p.accumulatorX, stepCount: 5)
            expect(frEq(p.circuitHash, expectedHash),
                   "Proof circuit hash should match verifier expectation")
        }
    }

    // =========================================================================
    // Test 36: State hash matches between engine and verifier
    // =========================================================================
    do {
        let shape = makeDCSquaringR1CS()
        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape)

        let (inst, wit) = makeDCSquaringPair(7)
        let pp = PedersenParams.generate(size: max(shape.numWitness, shape.numConstraints))
        let (relaxedInst, _) = shape.relax(instance: inst, witness: wit, pp: pp)

        let engineHash = engine.computeStateHash(instance: relaxedInst)
        let verifier = DeciderCircuitVerifier()
        let verifierHash = verifier.computeExpectedStateHash(
            u: relaxedInst.u, x: relaxedInst.x)
        expect(frEq(engineHash, verifierHash),
               "State hash from engine and verifier should match")
    }

    // =========================================================================
    // Test 37: Synthesized circuit is cached
    // =========================================================================
    do {
        let shape = makeDCSquaringR1CS()
        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape)

        let (shape1, layout1) = engine.synthesizeCircuit()
        let (shape2, layout2) = engine.synthesizeCircuit()

        // Same object references (cached)
        expect(shape1.numConstraints == shape2.numConstraints,
               "Cached shape should have same constraint count")
        expect(layout1.totalWitnessVars == layout2.totalWitnessVars,
               "Cached layout should have same witness count")
    }

    // =========================================================================
    // Test 38: SuperNova — two circuit types
    // =========================================================================
    do {
        let shape1 = makeDCSquaringR1CS()
        let shape2 = makeDCMultiplyR1CS()

        // Build accumulator for circuit 1
        let fold1 = GPUNovaFoldEngine(shape: shape1)
        let steps1 = [makeDCSquaringPair(3), makeDCSquaringPair(5)]
        let (inst1, wit1) = fold1.ivcChain(steps: steps1)

        // Build accumulator for circuit 2
        let fold2 = GPUNovaFoldEngine(shape: shape2)
        let steps2 = [makeDCMultiplyPair(2, 7), makeDCMultiplyPair(3, 11)]
        let (inst2, wit2) = fold2.ivcChain(steps: steps2)

        let superAccum = SuperNovaAccumulator(
            instances: [inst1, inst2],
            witnesses: [wit1, wit2],
            shapes: [shape1, shape2],
            circuitSchedule: [0, 0, 1, 1],
            stepCount: 4)

        // Use the main engine's SuperNova prove
        let engine = GPUNovaDeciderCircuitEngine(
            innerShape: shape1,
            config: DeciderCircuitConfig(isSuperNova: true, numCircuitTypes: 2))

        let proofs = engine.proveSuperNova(accumulator: superAccum)
        expect(proofs != nil, "SuperNova should produce proofs")

        if let ps = proofs {
            expect(ps.count == 2, "Should have 2 per-circuit proofs")
            expect(ps[0].isSuperNova, "Proof 0 should be SuperNova")
            expect(ps[1].isSuperNova, "Proof 1 should be SuperNova")
            expect(ps[0].circuitTypeIndex == 0, "Proof 0 circuit type = 0")
            expect(ps[1].circuitTypeIndex == 1, "Proof 1 circuit type = 1")
        }
    }

    // =========================================================================
    // Test 39: SuperNovaDeciderCircuitEngine — convenience wrapper
    // =========================================================================
    do {
        let shape1 = makeDCSquaringR1CS()
        let shape2 = makeDCMultiplyR1CS()

        let fold1 = GPUNovaFoldEngine(shape: shape1)
        let (inst1, wit1) = fold1.ivcChain(steps: [
            makeDCSquaringPair(2), makeDCSquaringPair(4)])

        let fold2 = GPUNovaFoldEngine(shape: shape2)
        let (inst2, wit2) = fold2.ivcChain(steps: [
            makeDCMultiplyPair(3, 5), makeDCMultiplyPair(7, 9)])

        let superAccum = SuperNovaAccumulator(
            instances: [inst1, inst2],
            witnesses: [wit1, wit2],
            shapes: [shape1, shape2],
            circuitSchedule: [0, 0, 1, 1],
            stepCount: 4)

        let superEngine = SuperNovaDeciderCircuitEngine(shapes: [shape1, shape2])
        let proofs = superEngine.prove(accumulator: superAccum)
        expect(proofs != nil, "SuperNovaDeciderCircuitEngine should produce proofs")
    }

    // =========================================================================
    // Test 40: SuperNova — checkAllCircuitsSatisfied
    // =========================================================================
    do {
        let shape1 = makeDCSquaringR1CS()
        let shape2 = makeDCMultiplyR1CS()

        let fold1 = GPUNovaFoldEngine(shape: shape1)
        let (inst1, wit1) = fold1.ivcChain(steps: [
            makeDCSquaringPair(3), makeDCSquaringPair(7)])

        let fold2 = GPUNovaFoldEngine(shape: shape2)
        let (inst2, wit2) = fold2.ivcChain(steps: [
            makeDCMultiplyPair(5, 11), makeDCMultiplyPair(2, 13)])

        let superAccum = SuperNovaAccumulator(
            instances: [inst1, inst2],
            witnesses: [wit1, wit2],
            shapes: [shape1, shape2],
            circuitSchedule: [0, 0, 1, 1],
            stepCount: 4)

        let superEngine = SuperNovaDeciderCircuitEngine(shapes: [shape1, shape2])
        let results = superEngine.checkAllCircuitsSatisfied(accumulator: superAccum)
        expect(results.count == 2, "Should have 2 results")
        expect(results[0], "Circuit 0 should be satisfied")
        expect(results[1], "Circuit 1 should be satisfied")
    }

    // =========================================================================
    // Test 41: SuperNova — allCircuitStats
    // =========================================================================
    do {
        let shape1 = makeDCSquaringR1CS()
        let shape2 = makeDCMultiplyR1CS()

        let superEngine = SuperNovaDeciderCircuitEngine(shapes: [shape1, shape2])
        let stats = superEngine.allCircuitStats()

        expect(stats.count == 2, "Should have 2 stat entries")
        expect(stats[0].constraints > 0, "Circuit 0 should have constraints")
        expect(stats[1].constraints > 0, "Circuit 1 should have constraints")

        // Different inner shapes lead to different decider circuit sizes
        expect(stats[0].publicInputs == 4, "Squaring: stateHash + u + 2x = 4 public")
        expect(stats[1].publicInputs == 3, "Multiply: stateHash + u + 1x = 3 public")
    }

    // =========================================================================
    // Test 42: Witness generation — values are consistent
    // =========================================================================
    do {
        let shape = makeDCSquaringR1CS()
        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape)

        let (inst, wit) = makeDCSquaringPair(4)
        let pp = PedersenParams.generate(size: max(shape.numWitness, shape.numConstraints))
        let (relaxedInst, relaxedWit) = shape.relax(instance: inst, witness: wit, pp: pp)

        let (pubInput, _) = engine.generateWitness(
            instance: relaxedInst, witness: relaxedWit)

        // Verify state hash in public input matches engine computation
        let expectedHash = engine.computeStateHash(instance: relaxedInst)
        expect(frEq(pubInput[0], expectedHash),
               "Public input[0] should be the state hash")
    }

    // =========================================================================
    // Test 43: Prove returns nil for inconsistent witness (wrong W)
    // =========================================================================
    do {
        let shape = makeDCSquaringR1CS()
        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape)

        let (inst, _) = makeDCSquaringPair(3)
        let pp = PedersenParams.generate(size: max(shape.numWitness, shape.numConstraints))
        // Make a bad witness: W = [7] instead of [3], so w*w = 49 != 9
        let badWit = NovaR1CSWitness(W: [frFromInt(7)])
        let (relaxedInst, relaxedWit) = shape.relax(instance: inst, witness: badWit, pp: pp)

        // The inner R1CS won't be satisfied, so the decider circuit won't be either
        let proof = engine.prove(instance: relaxedInst, witness: relaxedWit, stepCount: 1)
        expect(proof == nil, "Should not produce a proof for invalid witness")
    }

    // =========================================================================
    // Test 44: Diagnostic — detects failing constraints
    // =========================================================================
    do {
        let shape = makeDCSquaringR1CS()
        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape)

        let (inst, _) = makeDCSquaringPair(3)
        let pp = PedersenParams.generate(size: max(shape.numWitness, shape.numConstraints))
        let badWit = NovaR1CSWitness(W: [frFromInt(7)])
        let (relaxedInst, relaxedWit) = shape.relax(instance: inst, witness: badWit, pp: pp)

        let (satisfied, failing, total) = engine.diagnoseCircuitSatisfaction(
            instance: relaxedInst, witness: relaxedWit)
        expect(!satisfied, "Diagnostic should report not satisfied")
        expect(!failing.isEmpty, "Should have failing constraints")
        expect(total > 0, "Total constraints should be positive")
    }

    // =========================================================================
    // Test 45: Synthesize twice yields same constraint count
    // =========================================================================
    do {
        let shape = makeDCMultiplyR1CS()
        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape)

        let (shape1, _) = engine.synthesizeCircuit()
        let (shape2, _) = engine.synthesizeCircuit()

        expect(shape1.numConstraints == shape2.numConstraints,
               "Synthesize should be idempotent")
        expect(shape1.numVariables == shape2.numVariables,
               "Variable count should be stable")
    }

    // =========================================================================
    // Test 46: DeciderCircuitProof fields
    // =========================================================================
    do {
        let dummyShape = makeDCSquaringR1CS()
        let proof = DeciderCircuitProof(
            circuitShape: dummyShape,
            sumcheckRounds: [(Fr.one, Fr.zero, Fr.zero)],
            matVecEvals: (az: frFromInt(2), bz: frFromInt(3), cz: frFromInt(6)),
            commitW: pointIdentity(),
            accumulatorU: Fr.one,
            accumulatorX: [frFromInt(5)],
            accumulatorCommitW: pointIdentity(),
            accumulatorCommitE: pointIdentity(),
            circuitHash: frFromInt(42),
            stepCount: 7,
            isSuperNova: true,
            circuitTypeIndex: 3)

        expect(proof.sumcheckRounds.count == 1, "1 sumcheck round")
        expect(frEq(proof.matVecEvals.az, frFromInt(2)), "az eval")
        expect(frEq(proof.matVecEvals.bz, frFromInt(3)), "bz eval")
        expect(frEq(proof.matVecEvals.cz, frFromInt(6)), "cz eval")
        expect(frEq(proof.accumulatorU, Fr.one), "u = 1")
        expect(proof.accumulatorX.count == 1, "1 public input")
        expect(proof.stepCount == 7, "Step count = 7")
        expect(proof.isSuperNova, "Is SuperNova")
        expect(proof.circuitTypeIndex == 3, "Circuit type index = 3")
    }

    // =========================================================================
    // Test 47: DeciderCircuitWitnessLayout fields
    // =========================================================================
    do {
        let shape = makeDCSquaringR1CS()
        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape)
        let (_, layout) = engine.synthesizeCircuit()

        expect(layout.errorWires.count == shape.numConstraints,
               "Error wires count = inner constraints")
        expect(layout.azWires.count == shape.numConstraints,
               "Az wires count = inner constraints")
        expect(layout.bzWires.count == shape.numConstraints,
               "Bz wires count = inner constraints")
        expect(layout.czWires.count == shape.numConstraints,
               "Cz wires count = inner constraints")
        expect(layout.abProductWires.count == shape.numConstraints,
               "AB product wires count = inner constraints")
        expect(layout.uCzWires.count == shape.numConstraints,
               "uCz wires count = inner constraints")
        expect(layout.innerWitnessWires.count == shape.numWitness,
               "Inner witness wires = inner numWitness")
        expect(layout.publicInputWires.count == shape.numPublicInputs,
               "Public input wires = inner numPublicInputs")
    }

    // =========================================================================
    // Test 48: Engine with pre-generated Pedersen params
    // =========================================================================
    do {
        let shape = makeDCSquaringR1CS()
        let pp = PedersenParams.generate(size: 128)
        let ppE = PedersenParams.generate(size: 64)
        let engine = GPUNovaDeciderCircuitEngine(
            innerShape: shape, pp: pp, ppE: ppE)

        let (inst, wit) = makeDCSquaringPair(3)
        let ppRelax = PedersenParams.generate(size: max(shape.numWitness, shape.numConstraints))
        let (relaxedInst, relaxedWit) = shape.relax(instance: inst, witness: wit, pp: ppRelax)

        let sat = engine.checkCircuitSatisfaction(
            instance: relaxedInst, witness: relaxedWit)
        expect(sat, "Should work with pre-generated Pedersen params")
    }

    // =========================================================================
    // Test 49: Verifier construction
    // =========================================================================
    do {
        let verifier = DeciderCircuitVerifier()
        // Just check it can be constructed
        let hash = verifier.computeExpectedStateHash(u: Fr.one, x: [frFromInt(5)])
        expect(!hash.isZero, "Verifier state hash should be non-zero")
    }

    // =========================================================================
    // Test 50: Verifier — circuit hash recomputation
    // =========================================================================
    do {
        let verifier = DeciderCircuitVerifier()
        let h1 = verifier.computeExpectedCircuitHash(
            u: Fr.one, x: [frFromInt(5)], stepCount: 1)
        let h2 = verifier.computeExpectedCircuitHash(
            u: Fr.one, x: [frFromInt(5)], stepCount: 1)
        expect(frEq(h1, h2), "Circuit hash should be deterministic")

        let h3 = verifier.computeExpectedCircuitHash(
            u: Fr.one, x: [frFromInt(5)], stepCount: 2)
        expect(!frEq(h1, h3), "Different step count -> different hash")
    }

    // =========================================================================
    // Test 51: Multiple different values produce valid circuits
    // =========================================================================
    do {
        let shape = makeDCSquaringR1CS()
        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape)

        for val: UInt64 in [1, 2, 10, 100, 1000] {
            let (inst, wit) = makeDCSquaringPair(val)
            let pp = PedersenParams.generate(
                size: max(shape.numWitness, shape.numConstraints))
            let (ri, rw) = shape.relax(instance: inst, witness: wit, pp: pp)
            let sat = engine.checkCircuitSatisfaction(instance: ri, witness: rw)
            expect(sat, "Decider circuit should be satisfiable for val=\(val)")
        }
    }

    // =========================================================================
    // Test 52: Multiple different multiply pairs produce valid circuits
    // =========================================================================
    do {
        let shape = makeDCMultiplyR1CS()
        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape)

        let pairs: [(UInt64, UInt64)] = [(1, 1), (2, 3), (7, 11), (100, 200)]
        for (a, b) in pairs {
            let (inst, wit) = makeDCMultiplyPair(a, b)
            let pp = PedersenParams.generate(
                size: max(shape.numWitness, shape.numConstraints))
            let (ri, rw) = shape.relax(instance: inst, witness: wit, pp: pp)
            let sat = engine.checkCircuitSatisfaction(instance: ri, witness: rw)
            expect(sat, "Decider circuit satisfiable for \(a)*\(b)")
        }
    }

    // =========================================================================
    // Test 53: Folded multiply circuit satisfaction
    // =========================================================================
    do {
        let shape = makeDCMultiplyR1CS()
        let foldEngine = GPUNovaFoldEngine(shape: shape)

        var steps = [(instance: NovaR1CSInput, witness: NovaR1CSWitness)]()
        steps.append(makeDCMultiplyPair(3, 5))
        steps.append(makeDCMultiplyPair(7, 11))
        let (finalInst, finalWit) = foldEngine.ivcChain(steps: steps)

        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape)
        let sat = engine.checkCircuitSatisfaction(instance: finalInst, witness: finalWit)
        expect(sat, "Folded multiply circuit should satisfy decider circuit")
    }

    // =========================================================================
    // Test 54: Constraint set — multiple constraints build correctly
    // =========================================================================
    do {
        var cs = DeciderCircuitConstraintSet(numPublicInputs: 1)
        let a = cs.allocWitness(label: "a")
        let b = cs.allocWitness(label: "b")
        let c = cs.allocWitness(label: "c")
        let d = cs.allocWitness(label: "d")

        // a * b = c
        cs.addMulConstraint(a, b, c)
        // c * 1 = d  (via linear with coeff=1)
        cs.addLinearConstraint(coeff: Fr.one, c, d)

        expect(cs.constraints.count == 2, "Should have 2 constraints")

        let shape = cs.buildShape()
        expect(shape.numConstraints == 2, "Built shape has 2 constraints")

        // z = [1, pub, a, b, c, d] with a=3, b=5, c=15, d=15
        let inst = NovaR1CSInput(x: [frFromInt(99)])
        let wit = NovaR1CSWitness(W: [frFromInt(3), frFromInt(5), frFromInt(15), frFromInt(15)])
        expect(shape.satisfies(instance: inst, witness: wit),
               "Multi-constraint shape: 3*5=15, 15=15")
    }

    // =========================================================================
    // Test 55: gpuMatVec matches direct multiplication
    // =========================================================================
    do {
        let shape = makeDCSquaringR1CS()
        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape)

        // z = [1, 3, 9, 3] for squaring 3
        let z: [Fr] = [Fr.one, frFromInt(3), frFromInt(9), frFromInt(3)]

        let gpuAz = engine.gpuMatVec(shape.A, z)
        let directAz = shape.A.mulVec(z)

        expect(gpuAz.count == directAz.count, "gpuMatVec length matches")
        for i in 0..<gpuAz.count {
            expect(frEq(gpuAz[i], directAz[i]), "gpuMatVec[\(i)] matches direct")
        }
    }

    // =========================================================================
    // Test 56: Engine with custom Pedersen params and ppE
    // =========================================================================
    do {
        let shape = makeDCSquaringR1CS()
        let pp = PedersenParams.generate(size: 256)
        let ppE = PedersenParams.generate(size: 32)
        let config = DeciderCircuitConfig(useGPU: false)
        let engine = GPUNovaDeciderCircuitEngine(
            innerShape: shape, pp: pp, ppE: ppE, config: config)

        expect(!engine.gpuAvailable, "Should not have GPU with useGPU=false")

        let (inst, wit) = makeDCSquaringPair(5)
        let ppRelax = PedersenParams.generate(size: max(shape.numWitness, shape.numConstraints))
        let (ri, rw) = shape.relax(instance: inst, witness: wit, pp: ppRelax)
        let sat = engine.checkCircuitSatisfaction(instance: ri, witness: rw)
        expect(sat, "Should work with custom Pedersen params")
    }

    // =========================================================================
    // Test 57: Decider circuit constraint count scaling
    // =========================================================================
    do {
        let shape1 = makeDCSquaringR1CS()  // 1 constraint
        let shape2 = makeDCTwoConstraintR1CS()  // 2 constraints
        let shape3 = makeDCThreeConstraintR1CS()  // 3 constraints

        let engine1 = GPUNovaDeciderCircuitEngine(innerShape: shape1)
        let engine2 = GPUNovaDeciderCircuitEngine(innerShape: shape2)
        let engine3 = GPUNovaDeciderCircuitEngine(innerShape: shape3)

        let stats1 = engine1.circuitStats()
        let stats2 = engine2.circuitStats()
        let stats3 = engine3.circuitStats()

        // More inner constraints should yield more decider constraints
        expect(stats2.constraints > stats1.constraints,
               "2-constraint inner should yield more decider constraints than 1")
        expect(stats3.constraints > stats2.constraints,
               "3-constraint inner should yield more decider constraints than 2")
    }

    // =========================================================================
    // Test 58: SuperNova — verifier with valid proofs
    // =========================================================================
    do {
        let shape1 = makeDCSquaringR1CS()
        let shape2 = makeDCMultiplyR1CS()

        let fold1 = GPUNovaFoldEngine(shape: shape1)
        let (inst1, wit1) = fold1.ivcChain(steps: [
            makeDCSquaringPair(2), makeDCSquaringPair(3)])

        let fold2 = GPUNovaFoldEngine(shape: shape2)
        let (inst2, wit2) = fold2.ivcChain(steps: [
            makeDCMultiplyPair(5, 7), makeDCMultiplyPair(2, 3)])

        let superAccum = SuperNovaAccumulator(
            instances: [inst1, inst2],
            witnesses: [wit1, wit2],
            shapes: [shape1, shape2],
            circuitSchedule: [0, 0, 1, 1],
            stepCount: 4)

        let superEngine = SuperNovaDeciderCircuitEngine(shapes: [shape1, shape2])
        let satisfaction = superEngine.checkAllCircuitsSatisfied(accumulator: superAccum)
        expect(satisfaction.allSatisfy({ $0 }),
               "All circuits should be satisfied in SuperNova accumulator")
    }

    // =========================================================================
    // Test 59: DeciderCircuitConstraint fields
    // =========================================================================
    do {
        let constraint = DeciderCircuitConstraint(
            a: [(0, Fr.one), (1, frFromInt(2))],
            b: [(2, frFromInt(3))],
            c: [(3, frFromInt(4))])

        expect(constraint.a.count == 2, "a has 2 terms")
        expect(constraint.b.count == 1, "b has 1 term")
        expect(constraint.c.count == 1, "c has 1 term")
        expect(frEq(constraint.a[1].1, frFromInt(2)), "a[1] coeff = 2")
    }

    // =========================================================================
    // Test 60: Witness layout consistency with circuit shape
    // =========================================================================
    do {
        let shape = makeDCSquaringR1CS()
        let engine = GPUNovaDeciderCircuitEngine(innerShape: shape)
        let (circuitShape, layout) = engine.synthesizeCircuit()

        // Total variables = 1 (constant) + public inputs + witness vars
        let expectedTotal = 1 + circuitShape.numPublicInputs + layout.totalWitnessVars
        expect(circuitShape.numVariables == expectedTotal,
               "Circuit numVariables = 1 + pub + witness")

        // The number of witness vars should equal numVariables - 1 - numPublicInputs
        expect(layout.totalWitnessVars == circuitShape.numWitness,
               "Layout witness count = shape numWitness")
    }
}
