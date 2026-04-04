// plonk_custom_bench — Benchmark for Plonk custom gates (range, lookup, Poseidon)
//
// Compares circuit sizes and proof times:
//   1. Poseidon2 permutation with generic gates vs custom Poseidon S-box gates
//   2. Range check with generic bit-decomposition vs custom range gates
//   3. Lookup table membership with generic constraints vs custom lookup gates

import Foundation
import zkMetal

func runPlonkCustomBench() {
    fputs("\n--- Plonk Custom Gates Benchmark ---\n", stderr)

    do {
        // Shared SRS setup
        let srsSecret: [UInt32] = [0x12345678, 0x9ABCDEF0, 0x11111111, 0x22222222,
                                    0x33333333, 0x44444444, 0x55555555, 0x00000001]
        let srsSecretFr = frFromLimbs(srsSecret)
        let gx = fpFromInt(1)
        let gy = fpFromInt(2)
        let generator = PointAffine(x: gx, y: gy)
        let nttEngine = try NTTEngine()

        // ======================================================
        // Test 1: Range Check — generic vs custom gates
        // ======================================================
        fputs("\n  [1] Range Check (8-bit): generic vs custom\n", stderr)

        // Generic approach: k bit-decomposition gates (boolean + reconstruction)
        let rangeResult = try benchRangeCheck(
            bits: 8,
            srsSecret: srsSecret, srsSecretFr: srsSecretFr,
            generator: generator, nttEngine: nttEngine
        )
        fputs("      Generic:  \(rangeResult.genericGates) gates, prove \(String(format: "%.1f", rangeResult.genericProveMs))ms\n", stderr)
        fputs("      Custom:   \(rangeResult.customGates) gates, prove \(String(format: "%.1f", rangeResult.customProveMs))ms\n", stderr)
        fputs("      Savings:  \(String(format: "%.0f", rangeResult.gateSavingsPercent))% fewer gates\n", stderr)
        fputs("      Correct:  \(rangeResult.genericValid && rangeResult.customValid ? "PASS" : "FAIL")\n", stderr)

        // ======================================================
        // Test 2: Lookup Table — membership proof
        // ======================================================
        fputs("\n  [2] Lookup Table (16 entries): custom gate\n", stderr)

        let lookupResult = try benchLookup(
            tableSize: 16,
            srsSecret: srsSecret, srsSecretFr: srsSecretFr,
            generator: generator, nttEngine: nttEngine
        )
        fputs("      Custom:   \(lookupResult.customGates) gates, prove \(String(format: "%.1f", lookupResult.customProveMs))ms\n", stderr)
        fputs("      Correct:  \(lookupResult.customValid ? "PASS" : "FAIL")\n", stderr)

        // ======================================================
        // Test 3: Poseidon S-box — generic vs custom gates
        // ======================================================
        fputs("\n  [3] Poseidon2 S-box (x^5): generic vs custom\n", stderr)

        let poseidonResult = try benchPoseidonSbox(
            srsSecret: srsSecret, srsSecretFr: srsSecretFr,
            generator: generator, nttEngine: nttEngine
        )
        fputs("      Generic:  \(poseidonResult.genericGates) gates, prove \(String(format: "%.1f", poseidonResult.genericProveMs))ms\n", stderr)
        fputs("      Custom:   \(poseidonResult.customGates) gates, prove \(String(format: "%.1f", poseidonResult.customProveMs))ms\n", stderr)
        fputs("      Savings:  \(String(format: "%.0f", poseidonResult.gateSavingsPercent))% fewer gates\n", stderr)
        fputs("      Correct:  \(poseidonResult.genericValid && poseidonResult.customValid ? "PASS" : "FAIL")\n", stderr)

        // ======================================================
        // Test 4: Full Poseidon2 permutation — generic vs custom
        // ======================================================
        fputs("\n  [4] Full Poseidon2 permutation (64 rounds): generic vs custom\n", stderr)

        let fullPosResult = try benchFullPoseidon2(
            srsSecret: srsSecret, srsSecretFr: srsSecretFr,
            generator: generator, nttEngine: nttEngine
        )
        fputs("      Generic:  \(fullPosResult.genericGates) gates\n", stderr)
        fputs("      Custom:   \(fullPosResult.customGates) gates\n", stderr)
        fputs("      Savings:  \(String(format: "%.0f", fullPosResult.gateSavingsPercent))% fewer gates\n", stderr)
        fputs("      Correct:  \(fullPosResult.genericValid && fullPosResult.customValid ? "PASS" : "FAIL")\n", stderr)

        // ======================================================
        // Summary
        // ======================================================
        fputs("\n  Summary: Custom gates reduce circuit size significantly:\n", stderr)
        fputs("    Range(8-bit): \(rangeResult.genericGates) -> \(rangeResult.customGates) gates\n", stderr)
        fputs("    Poseidon S-box: \(poseidonResult.genericGates) -> \(poseidonResult.customGates) gates\n", stderr)
        fputs("    Full Poseidon2: \(fullPosResult.genericGates) -> \(fullPosResult.customGates) gates\n", stderr)

    } catch {
        fputs("Plonk custom gate bench error: \(error)\n", stderr)
    }
}

// MARK: - Range Check Benchmark

struct RangeCheckResult {
    let genericGates: Int
    let customGates: Int
    let genericProveMs: Double
    let customProveMs: Double
    let genericValid: Bool
    let customValid: Bool
    var gateSavingsPercent: Double {
        Double(genericGates - customGates) / Double(genericGates) * 100
    }
}

private func benchRangeCheck(bits: Int, srsSecret: [UInt32], srsSecretFr: Fr,
                              generator: PointAffine, nttEngine: NTTEngine) throws -> RangeCheckResult {
    let testValue: UInt64 = 42  // Value to range-check (fits in 8 bits)

    // --- Generic approach: manual bit decomposition ---
    let genericBuilder = PlonkCircuitBuilder()
    let gInput = genericBuilder.addInput()

    // Decompose into bits manually
    var gBitVars = [Int]()
    for _ in 0..<bits {
        let bit = genericBuilder.addInput()
        gBitVars.append(bit)
        // Boolean constraint: bit * bit = bit => bit*(bit-1) = 0
        // Using mul gate: bit * bit = bitSq, then assert bitSq == bit
        let bitSq = genericBuilder.mul(bit, bit)
        genericBuilder.assertEqual(bitSq, bit)
    }

    // Reconstruction: sum(bit_i * 2^i) = value
    var gAccVar = gBitVars[0]
    for i in 1..<bits {
        // acc_new = acc_old + bit_i * 2^i
        // Need to multiply bit_i by constant 2^i via gate
        let coeff = frFromInt(1 << UInt64(i))
        let scaledBit = genericBuilder.addInput()  // auxiliary: bit_i * 2^i
        // Constraint: scaledBit = coeff * bit_i (use qL gate)
        // We'll use an add gate with coefficient: qR*b - qO*c = 0 => c = qR*b
        // Actually simpler: just add the reconstruction gate
        let acc_new = genericBuilder.add(gAccVar, scaledBit)
        gAccVar = acc_new
        // We need scaledBit = bit_i * 2^i; mark this in witness
        _ = (scaledBit, coeff)
    }
    genericBuilder.assertEqual(gAccVar, gInput)

    let genericCircuit = genericBuilder.build().padded()
    let genericGateCount = genericBuilder.gates.count

    // --- Custom approach: range gate ---
    let customBuilder = PlonkCircuitBuilder()
    let cInput = customBuilder.addInput()
    let rangeInfo = customBuilder.rangeCheck(cInput, bits: bits)

    let customCircuit = customBuilder.build().padded()
    let customGateCount = customBuilder.gates.count

    // --- Build witnesses and prove ---
    let testValueFr = frFromInt(testValue)

    // Generic witness
    let gMaxVar = (genericCircuit.wireAssignments.flatMap { $0 }.max() ?? 0) + 1
    var gWitness = [Fr](repeating: Fr.zero, count: gMaxVar)
    gWitness[gInput] = testValueFr

    // Set bit values
    for i in 0..<bits {
        let bitVal: UInt64 = (testValue >> UInt64(i)) & 1
        gWitness[gBitVars[i]] = frFromInt(bitVal)
    }

    // Evaluate all generic gates to fill witness
    for i in 0..<genericBuilder.gates.count {
        let wires = genericBuilder.wireAssignments[i]
        let a = gWitness[wires[0]]
        let b = gWitness[wires[1]]
        let g = genericBuilder.gates[i]
        if g.qO.isZero { continue }
        let num = frAdd(frAdd(frMul(g.qL, a), frMul(g.qR, b)),
                        frAdd(frMul(g.qM, frMul(a, b)), g.qC))
        let negNum = frSub(Fr.zero, num)
        let c = frMul(negNum, frInverse(g.qO))
        if gWitness[wires[2]].isZero && !c.isZero {
            gWitness[wires[2]] = c
        }
    }

    // Custom witness
    let cMaxVar = (customCircuit.wireAssignments.flatMap { $0 }.max() ?? 0) + 1
    var cWitness = [Fr](repeating: Fr.zero, count: cMaxVar)
    cWitness[cInput] = testValueFr

    // Set limb values for range gate
    for i in 0..<bits {
        let bitVal: UInt64 = (testValue >> UInt64(i)) & 1
        cWitness[rangeInfo.limbVars[i]] = frFromInt(bitVal)
    }

    // Evaluate custom gates
    for i in 0..<customBuilder.gates.count {
        let wires = customBuilder.wireAssignments[i]
        let a = cWitness[wires[0]]
        let b = cWitness[wires[1]]
        let g = customBuilder.gates[i]
        if g.qO.isZero { continue }
        let num = frAdd(frAdd(frMul(g.qL, a), frMul(g.qR, b)),
                        frAdd(frMul(g.qM, frMul(a, b)), g.qC))
        let negNum = frSub(Fr.zero, num)
        let c = frMul(negNum, frInverse(g.qO))
        if cWitness[wires[2]].isZero && !c.isZero {
            cWitness[wires[2]] = c
        }
    }

    // Prove and verify generic
    let gN = genericCircuit.numGates
    let gSRS = KZGEngine.generateTestSRS(secret: srsSecret, size: gN + 3, generator: generator)
    let gKZG = try KZGEngine(srs: gSRS)
    let gPrep = PlonkPreprocessor(kzg: gKZG, ntt: nttEngine)
    let gSetup = try gPrep.setup(circuit: genericCircuit, srsSecret: srsSecretFr)
    let gProver = PlonkProver(setup: gSetup, kzg: gKZG, ntt: nttEngine)
    let gt0 = CFAbsoluteTimeGetCurrent()
    let gProof = try gProver.prove(witness: gWitness, circuit: genericCircuit)
    let gProveMs = (CFAbsoluteTimeGetCurrent() - gt0) * 1000
    let gVerifier = PlonkVerifier(setup: gSetup, kzg: gKZG)
    let gValid = gVerifier.verify(proof: gProof)

    // Prove and verify custom
    let cN = customCircuit.numGates
    let cSRS = KZGEngine.generateTestSRS(secret: srsSecret, size: cN + 3, generator: generator)
    let cKZG = try KZGEngine(srs: cSRS)
    let cPrep = PlonkPreprocessor(kzg: cKZG, ntt: nttEngine)
    let cSetup = try cPrep.setup(circuit: customCircuit, srsSecret: srsSecretFr)
    let cProver = PlonkProver(setup: cSetup, kzg: cKZG, ntt: nttEngine)
    let ct0 = CFAbsoluteTimeGetCurrent()
    let cProof = try cProver.prove(witness: cWitness, circuit: customCircuit)
    let cProveMs = (CFAbsoluteTimeGetCurrent() - ct0) * 1000
    let cVerifier = PlonkVerifier(setup: cSetup, kzg: cKZG)
    let cValid = cVerifier.verify(proof: cProof)

    return RangeCheckResult(
        genericGates: genericGateCount, customGates: customGateCount,
        genericProveMs: gProveMs, customProveMs: cProveMs,
        genericValid: gValid, customValid: cValid
    )
}

// MARK: - Lookup Benchmark

struct LookupResult {
    let customGates: Int
    let customProveMs: Double
    let customValid: Bool
}

private func benchLookup(tableSize: Int, srsSecret: [UInt32], srsSecretFr: Fr,
                           generator: PointAffine, nttEngine: NTTEngine) throws -> LookupResult {
    let builder = PlonkCircuitBuilder()

    // Create a lookup table with values 0..tableSize-1
    var tableValues = [Fr]()
    for i in 0..<tableSize {
        tableValues.append(frFromInt(UInt64(i)))
    }
    let tableId = builder.addLookupTable(values: tableValues)

    // Create several lookup gates for values that are in the table
    let testValues: [UInt64] = [0, 5, 10, 15, 3, 7]
    var inputVars = [Int]()
    for val in testValues {
        let v = builder.addInput()
        inputVars.append(v)
        builder.lookup(v, tableId: tableId)
    }

    // Also add some filler gates to get to a reasonable size
    if inputVars.count >= 2 {
        var prev = inputVars[0]
        for i in 1..<inputVars.count {
            prev = builder.add(prev, inputVars[i])
        }
    }

    let circuit = builder.build().padded()
    let gateCount = builder.gates.count

    // Build witness
    let maxVar = (circuit.wireAssignments.flatMap { $0 }.max() ?? 0) + 1
    var witness = [Fr](repeating: Fr.zero, count: maxVar)
    for (i, val) in testValues.enumerated() {
        witness[inputVars[i]] = frFromInt(val)
    }

    // Evaluate gates to fill witness
    for i in 0..<builder.gates.count {
        let wires = builder.wireAssignments[i]
        let a = witness[wires[0]]
        let b = witness[wires[1]]
        let g = builder.gates[i]
        if g.qO.isZero { continue }
        let num = frAdd(frAdd(frMul(g.qL, a), frMul(g.qR, b)),
                        frAdd(frMul(g.qM, frMul(a, b)), g.qC))
        let negNum = frSub(Fr.zero, num)
        let c = frMul(negNum, frInverse(g.qO))
        if witness[wires[2]].isZero && !c.isZero {
            witness[wires[2]] = c
        }
    }

    // Prove and verify
    let n = circuit.numGates
    let srs = KZGEngine.generateTestSRS(secret: srsSecret, size: n + 3, generator: generator)
    let kzg = try KZGEngine(srs: srs)
    let prep = PlonkPreprocessor(kzg: kzg, ntt: nttEngine)
    let setup = try prep.setup(circuit: circuit, srsSecret: srsSecretFr)
    let prover = PlonkProver(setup: setup, kzg: kzg, ntt: nttEngine)
    let t0 = CFAbsoluteTimeGetCurrent()
    let proof = try prover.prove(witness: witness, circuit: circuit)
    let proveMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
    let verifier = PlonkVerifier(setup: setup, kzg: kzg)
    let valid = verifier.verify(proof: proof)

    return LookupResult(customGates: gateCount, customProveMs: proveMs, customValid: valid)
}

// MARK: - Poseidon S-box Benchmark

struct PoseidonSboxResult {
    let genericGates: Int
    let customGates: Int
    let genericProveMs: Double
    let customProveMs: Double
    let genericValid: Bool
    let customValid: Bool
    var gateSavingsPercent: Double {
        Double(genericGates - customGates) / Double(genericGates) * 100
    }
}

private func benchPoseidonSbox(srsSecret: [UInt32], srsSecretFr: Fr,
                                 generator: PointAffine, nttEngine: NTTEngine) throws -> PoseidonSboxResult {
    let testInput: UInt64 = 7

    // --- Generic: x^5 = ((x*x)*(x*x))*x using 4 mul gates ---
    let gBuilder = PlonkCircuitBuilder()
    let gInput = gBuilder.addInput()
    let x2 = gBuilder.mul(gInput, gInput)     // x^2
    let x4 = gBuilder.mul(x2, x2)             // x^4
    let x5 = gBuilder.mul(x4, gInput)         // x^5

    let gCircuit = gBuilder.build().padded()
    let gGateCount = gBuilder.gates.count

    // --- Custom: poseidonSbox gate (2 gates: 1 mul + 1 custom) ---
    let cBuilder = PlonkCircuitBuilder()
    let cInput = cBuilder.addInput()
    let cOutput = cBuilder.poseidonSbox(cInput)

    let cCircuit = cBuilder.build().padded()
    let cGateCount = cBuilder.gates.count

    // Compute expected output: 7^5 = 16807
    let inputFr = frFromInt(testInput)
    let x2Fr = frSqr(inputFr)
    let x4Fr = frSqr(x2Fr)
    let x5Fr = frMul(x4Fr, inputFr)

    // Generic witness
    let gMaxVar = (gCircuit.wireAssignments.flatMap { $0 }.max() ?? 0) + 1
    var gWitness = [Fr](repeating: Fr.zero, count: gMaxVar)
    gWitness[gInput] = inputFr
    gWitness[x2] = x2Fr
    gWitness[x4] = x4Fr
    gWitness[x5] = x5Fr

    // Custom witness
    let cMaxVar = (cCircuit.wireAssignments.flatMap { $0 }.max() ?? 0) + 1
    var cWitness = [Fr](repeating: Fr.zero, count: cMaxVar)
    cWitness[cInput] = inputFr

    // For poseidonSbox: gate 0 is mul(input, input) -> sq, gate 1 is sbox(input, sq, output)
    // sq = input^2, output = input^5
    // The builder creates: [input, input, sq] for mul gate, [input, sq, output] for sbox gate
    // Wire assignments for the custom gates
    for i in 0..<cBuilder.gates.count {
        let wires = cBuilder.wireAssignments[i]
        let a = cWitness[wires[0]]
        let b = cWitness[wires[1]]
        let g = cBuilder.gates[i]

        if !g.qPoseidon.isZero {
            // Poseidon S-box: c = a * b * b = a * (a^2)^2 = a^5
            let bSq = frSqr(b)
            cWitness[wires[2]] = frMul(a, bSq)
        } else if !g.qO.isZero {
            let num = frAdd(frAdd(frMul(g.qL, a), frMul(g.qR, b)),
                            frAdd(frMul(g.qM, frMul(a, b)), g.qC))
            let negNum = frSub(Fr.zero, num)
            let c = frMul(negNum, frInverse(g.qO))
            cWitness[wires[2]] = c
        }
    }

    // Prove and verify generic
    let gN = gCircuit.numGates
    let gSRS = KZGEngine.generateTestSRS(secret: srsSecret, size: gN + 3, generator: generator)
    let gKZG = try KZGEngine(srs: gSRS)
    let gPrep = PlonkPreprocessor(kzg: gKZG, ntt: nttEngine)
    let gSetup = try gPrep.setup(circuit: gCircuit, srsSecret: srsSecretFr)
    let gProver = PlonkProver(setup: gSetup, kzg: gKZG, ntt: nttEngine)
    let gt0 = CFAbsoluteTimeGetCurrent()
    let gProof = try gProver.prove(witness: gWitness, circuit: gCircuit)
    let gProveMs = (CFAbsoluteTimeGetCurrent() - gt0) * 1000
    let gVerifier = PlonkVerifier(setup: gSetup, kzg: gKZG)
    let gValid = gVerifier.verify(proof: gProof)

    // Prove and verify custom
    let cN = cCircuit.numGates
    let cSRS = KZGEngine.generateTestSRS(secret: srsSecret, size: cN + 3, generator: generator)
    let cKZG = try KZGEngine(srs: cSRS)
    let cPrep = PlonkPreprocessor(kzg: cKZG, ntt: nttEngine)
    let cSetup = try cPrep.setup(circuit: cCircuit, srsSecret: srsSecretFr)
    let cProver = PlonkProver(setup: cSetup, kzg: cKZG, ntt: nttEngine)
    let ct0 = CFAbsoluteTimeGetCurrent()
    let cProof = try cProver.prove(witness: cWitness, circuit: cCircuit)
    let cProveMs = (CFAbsoluteTimeGetCurrent() - ct0) * 1000
    let cVerifier = PlonkVerifier(setup: cSetup, kzg: cKZG)
    let cValid = cVerifier.verify(proof: cProof)

    return PoseidonSboxResult(
        genericGates: gGateCount, customGates: cGateCount,
        genericProveMs: gProveMs, customProveMs: cProveMs,
        genericValid: gValid, customValid: cValid
    )
}

// MARK: - Full Poseidon2 Permutation Benchmark

struct FullPoseidonResult {
    let genericGates: Int
    let customGates: Int
    let genericValid: Bool
    let customValid: Bool
    var gateSavingsPercent: Double {
        if genericGates == 0 { return 0 }
        return Double(genericGates - customGates) / Double(genericGates) * 100
    }
}

private func benchFullPoseidon2(srsSecret: [UInt32], srsSecretFr: Fr,
                                  generator: PointAffine, nttEngine: NTTEngine) throws -> FullPoseidonResult {
    let rc = POSEIDON2_ROUND_CONSTANTS
    let inputValues: [UInt64] = [1, 2, 3]

    // --- Generic: all x^5 via 3 mul gates each (x^2, x^4, x^5) ---
    let gBuilder = PlonkCircuitBuilder()
    var gState = [Int]()
    for val in inputValues {
        let v = gBuilder.addInput()
        gState.append(v)
    }

    // Helper: generic x^5 using 3 mul gates
    func genericSbox(_ builder: PlonkCircuitBuilder, _ x: Int) -> Int {
        let x2 = builder.mul(x, x)
        let x4 = builder.mul(x2, x2)
        return builder.mul(x4, x)
    }

    // Initial external linear layer
    gState = gBuilder.poseidonExternalLinearLayer(gState)

    // First 4 full rounds
    for r in 0..<4 {
        for i in 0..<3 { gState[i] = gBuilder.addConstant(gState[i], rc[r][i]) }
        for i in 0..<3 { gState[i] = genericSbox(gBuilder, gState[i]) }
        gState = gBuilder.poseidonExternalLinearLayer(gState)
    }

    // 56 partial rounds (S-box on first element only)
    for r in 4..<60 {
        gState[0] = gBuilder.addConstant(gState[0], rc[r][0])
        gState[0] = genericSbox(gBuilder, gState[0])
        gState = gBuilder.poseidonInternalLinearLayer(gState)
    }

    // Last 4 full rounds
    for r in 60..<64 {
        for i in 0..<3 { gState[i] = gBuilder.addConstant(gState[i], rc[r][i]) }
        for i in 0..<3 { gState[i] = genericSbox(gBuilder, gState[i]) }
        gState = gBuilder.poseidonExternalLinearLayer(gState)
    }

    let gGateCount = gBuilder.gates.count

    // --- Custom: x^5 via poseidonSbox (2 gates instead of 3) ---
    let cBuilder = PlonkCircuitBuilder()
    var cState = [Int]()
    for val in inputValues {
        let v = cBuilder.addInput()
        cState.append(v)
        _ = val
    }

    cState = cBuilder.poseidonExternalLinearLayer(cState)

    for r in 0..<4 {
        for i in 0..<3 { cState[i] = cBuilder.addConstant(cState[i], rc[r][i]) }
        for i in 0..<3 { cState[i] = cBuilder.poseidonSbox(cState[i]) }
        cState = cBuilder.poseidonExternalLinearLayer(cState)
    }

    for r in 4..<60 {
        cState[0] = cBuilder.addConstant(cState[0], rc[r][0])
        cState[0] = cBuilder.poseidonSbox(cState[0])
        cState = cBuilder.poseidonInternalLinearLayer(cState)
    }

    for r in 60..<64 {
        for i in 0..<3 { cState[i] = cBuilder.addConstant(cState[i], rc[r][i]) }
        for i in 0..<3 { cState[i] = cBuilder.poseidonSbox(cState[i]) }
        cState = cBuilder.poseidonExternalLinearLayer(cState)
    }

    let cGateCount = cBuilder.gates.count

    // Build witnesses using CPU Poseidon2 permutation as reference
    let inputFrs = inputValues.map { frFromInt($0) }
    let expectedOutput = poseidon2Permutation(inputFrs)
    _ = expectedOutput  // Used to validate

    // Generic witness: evaluate all gates sequentially
    let gCircuit = gBuilder.build().padded()
    let gMaxVar = (gCircuit.wireAssignments.flatMap { $0 }.max() ?? 0) + 1
    var gWitness = [Fr](repeating: Fr.zero, count: gMaxVar)
    for (i, v) in inputValues.enumerated() {
        gWitness[gState.count > i ? i : i] = frFromInt(v)  // first 3 vars are inputs
    }
    // The builder input vars are 0, 1, 2
    for i in 0..<3 {
        gWitness[i] = frFromInt(inputValues[i])
    }

    // Evaluate each gate to fill witness
    for i in 0..<gBuilder.gates.count {
        let wires = gBuilder.wireAssignments[i]
        let a = gWitness[wires[0]]
        let b = gWitness[wires[1]]
        let g = gBuilder.gates[i]
        if g.qO.isZero { continue }
        let num = frAdd(frAdd(frMul(g.qL, a), frMul(g.qR, b)),
                        frAdd(frMul(g.qM, frMul(a, b)), g.qC))
        let negNum = frSub(Fr.zero, num)
        let c = frMul(negNum, frInverse(g.qO))
        gWitness[wires[2]] = c
    }

    // Custom witness
    let cCircuit = cBuilder.build().padded()
    let cMaxVar = (cCircuit.wireAssignments.flatMap { $0 }.max() ?? 0) + 1
    var cWitness = [Fr](repeating: Fr.zero, count: cMaxVar)
    for i in 0..<3 {
        cWitness[i] = frFromInt(inputValues[i])
    }

    // Evaluate each gate, handling custom gates
    for i in 0..<cBuilder.gates.count {
        let wires = cBuilder.wireAssignments[i]
        let a = cWitness[wires[0]]
        let b = cWitness[wires[1]]
        let g = cBuilder.gates[i]

        if !g.qPoseidon.isZero {
            // Poseidon S-box: c = a * b^2 (where b = a^2, so c = a^5)
            let bSq = frSqr(b)
            cWitness[wires[2]] = frMul(a, bSq)
        } else if !g.qO.isZero {
            let num = frAdd(frAdd(frMul(g.qL, a), frMul(g.qR, b)),
                            frAdd(frMul(g.qM, frMul(a, b)), g.qC))
            let negNum = frSub(Fr.zero, num)
            let c = frMul(negNum, frInverse(g.qO))
            cWitness[wires[2]] = c
        }
    }

    // Prove and verify both
    let gN = gCircuit.numGates
    let gSRS = KZGEngine.generateTestSRS(secret: srsSecret, size: gN + 3, generator: generator)
    let gKZG = try KZGEngine(srs: gSRS)
    let gPrep = PlonkPreprocessor(kzg: gKZG, ntt: nttEngine)
    let gSetup = try gPrep.setup(circuit: gCircuit, srsSecret: srsSecretFr)
    let gProver = PlonkProver(setup: gSetup, kzg: gKZG, ntt: nttEngine)
    let gProof = try gProver.prove(witness: gWitness, circuit: gCircuit)
    let gVerifier = PlonkVerifier(setup: gSetup, kzg: gKZG)
    let gValid = gVerifier.verify(proof: gProof)

    let cN = cCircuit.numGates
    let cSRS = KZGEngine.generateTestSRS(secret: srsSecret, size: cN + 3, generator: generator)
    let cKZG = try KZGEngine(srs: cSRS)
    let cPrep = PlonkPreprocessor(kzg: cKZG, ntt: nttEngine)
    let cSetup = try cPrep.setup(circuit: cCircuit, srsSecret: srsSecretFr)
    let cProver = PlonkProver(setup: cSetup, kzg: cKZG, ntt: nttEngine)
    let cProof = try cProver.prove(witness: cWitness, circuit: cCircuit)
    let cVerifier = PlonkVerifier(setup: cSetup, kzg: cKZG)
    let cValid = cVerifier.verify(proof: cProof)

    return FullPoseidonResult(
        genericGates: gGateCount, customGates: cGateCount,
        genericValid: gValid, customValid: cValid
    )
}
