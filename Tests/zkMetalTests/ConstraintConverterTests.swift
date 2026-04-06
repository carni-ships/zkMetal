// ConstraintConverterTests — Tests for R1CS <-> Plonk constraint system conversion
//
// Verifies round-trip correctness, simple/linear/mixed circuits, and satisfaction checks.

import Foundation
import zkMetal

public func runConstraintConverterTests() {
    suite("ConstraintConverter")

    testSimpleMultiply()
    testAdditionOnly()
    testQuadraticCircuit()
    testMixedCircuit()
    testRoundTrip()
    testPlonkToR1CSSatisfaction()
    testR1CSToPlonkSatisfaction()
    testConversionStats()
    testConstantConstraints()
    testMultiTermComplex()
}

// MARK: - Simple multiply: x * y = z

private func testSimpleMultiply() {
    // Build R1CS: x * y = z
    let b = SpartanR1CSBuilder()
    let zOut = b.addPublicInput()  // variable 1
    let x = b.addWitness()         // variable 2
    let y = b.addWitness()         // variable 3
    b.mulGate(a: x, b: y, out: zOut)
    let r1cs = b.build()

    // Witness: x=3, y=7, z=21
    let xVal = frFromInt(3)
    let yVal = frFromInt(7)
    let zVal = frMul(xVal, yVal) // 21
    let z = SpartanR1CS.buildZ(publicInputs: [zVal], witness: [xVal, yVal])

    // Verify R1CS is satisfied
    expect(r1cs.isSatisfied(z: z), "R1CS should be satisfied before conversion")

    // Convert to Plonk
    let (circuit, witness, stats) = R1CSToPlonkConverter.convert(r1cs: r1cs, z: z)

    expect(circuit.numGates > 0, "Should produce at least one Plonk gate")
    expect(stats.r1csConstraints == 1, "Should have 1 R1CS constraint")

    // Verify Plonk satisfaction
    let sat = R1CSToPlonkConverter.verifySatisfaction(circuit: circuit, witness: witness)
    expect(sat, "Converted Plonk circuit should be satisfied")

    print("  [OK] simple multiply x*y=z")
}

// MARK: - Addition-only circuit

private func testAdditionOnly() {
    // Build R1CS: (x + y) * 1 = z  (addition expressed as R1CS)
    let b = SpartanR1CSBuilder()
    let zOut = b.addPublicInput()
    let x = b.addWitness()
    let y = b.addWitness()
    b.addGate(a: x, b: y, out: zOut) // (x+y)*1 = z

    let r1cs = b.build()

    let xVal = frFromInt(5)
    let yVal = frFromInt(8)
    let zVal = frAdd(xVal, yVal) // 13
    let z = SpartanR1CS.buildZ(publicInputs: [zVal], witness: [xVal, yVal])

    expect(r1cs.isSatisfied(z: z), "Addition R1CS should be satisfied")

    let (circuit, witness, stats) = R1CSToPlonkConverter.convert(r1cs: r1cs, z: z)

    // Linear constraint should be efficient (no extra multiplication gates)
    expect(stats.linearConstraints == 1, "Should detect 1 linear constraint")
    expect(stats.quadraticConstraints == 0, "No quadratic constraints expected")

    let sat = R1CSToPlonkConverter.verifySatisfaction(circuit: circuit, witness: witness)
    expect(sat, "Converted addition Plonk circuit should be satisfied")

    print("  [OK] addition-only circuit stays efficient")
}

// MARK: - Quadratic circuit: x^2 + x + 5 = y

private func testQuadraticCircuit() {
    let (r1cs, gen) = SpartanR1CSBuilder.exampleQuadratic()
    let xVal = frFromInt(3)
    let (publicInputs, wit) = gen(xVal)
    let z = SpartanR1CS.buildZ(publicInputs: publicInputs, witness: wit)

    expect(r1cs.isSatisfied(z: z), "Quadratic R1CS should be satisfied")

    let (circuit, witness, stats) = R1CSToPlonkConverter.convert(r1cs: r1cs, z: z)

    expect(circuit.numGates > 0, "Should produce Plonk gates")
    expect(stats.r1csConstraints == 3, "Quadratic example has 3 constraints")

    let sat = R1CSToPlonkConverter.verifySatisfaction(circuit: circuit, witness: witness)
    expect(sat, "Converted quadratic Plonk circuit should be satisfied")

    print("  [OK] quadratic circuit x^2+x+5=y")
}

// MARK: - Mixed circuit: 10 constraints with varying complexity

private func testMixedCircuit() {
    let b = SpartanR1CSBuilder()
    let out = b.addPublicInput()
    var vars = [Int]()

    // Create 5 witness variables
    for _ in 0..<5 {
        vars.append(b.addWitness())
    }

    // Constraint 1-3: multiply gates (quadratic)
    let v5 = b.addWitness(); vars.append(v5)
    b.mulGate(a: vars[0], b: vars[1], out: v5)

    let v6 = b.addWitness(); vars.append(v6)
    b.mulGate(a: vars[2], b: vars[3], out: v6)

    let v7 = b.addWitness(); vars.append(v7)
    b.mulGate(a: v5, b: v6, out: v7)

    // Constraint 4-6: add gates (linear)
    let v8 = b.addWitness(); vars.append(v8)
    b.addGate(a: vars[0], b: vars[1], out: v8)

    let v9 = b.addWitness(); vars.append(v9)
    b.addGate(a: v8, b: vars[4], out: v9)

    let v10 = b.addWitness(); vars.append(v10)
    b.addGate(a: v9, b: v7, out: v10)

    // Constraint 7: add constant
    let v11 = b.addWitness(); vars.append(v11)
    b.addConstGate(a: v10, constant: frFromInt(42), out: v11)

    // Constraint 8-9: more multiplies
    let v12 = b.addWitness(); vars.append(v12)
    b.mulGate(a: v11, b: vars[4], out: v12)

    let v13 = b.addWitness(); vars.append(v13)
    b.mulGate(a: v12, b: v12, out: v13)

    // Constraint 10: final output
    b.addConstraint(a: [(v13, Fr.one)], b: [(0, Fr.one)], c: [(out, Fr.one)])

    let r1cs = b.build()
    expect(r1cs.numConstraints == 10, "Should have 10 constraints")

    // Generate witness values
    let vals: [Fr] = (0..<5).map { frFromInt(UInt64($0 + 2)) } // 2,3,4,5,6
    var allVals = [Fr](repeating: Fr.zero, count: r1cs.numVariables)
    allVals[0] = Fr.one
    for i in 0..<5 { allVals[vars[i]] = vals[i] }

    // Compute intermediate values
    allVals[v5] = frMul(vals[0], vals[1])       // 2*3 = 6
    allVals[v6] = frMul(vals[2], vals[3])       // 4*5 = 20
    allVals[v7] = frMul(allVals[v5], allVals[v6]) // 6*20 = 120
    allVals[v8] = frAdd(vals[0], vals[1])       // 2+3 = 5
    allVals[v9] = frAdd(allVals[v8], vals[4])   // 5+6 = 11
    allVals[v10] = frAdd(allVals[v9], allVals[v7]) // 11+120 = 131
    allVals[v11] = frAdd(allVals[v10], frFromInt(42)) // 131+42 = 173
    allVals[v12] = frMul(allVals[v11], vals[4]) // 173*6 = 1038
    allVals[v13] = frMul(allVals[v12], allVals[v12]) // 1038^2 = 1077444
    allVals[out] = allVals[v13]

    expect(r1cs.isSatisfied(z: allVals), "Mixed R1CS should be satisfied")

    let (circuit, witness, stats) = R1CSToPlonkConverter.convert(r1cs: r1cs, z: allVals)

    expect(stats.r1csConstraints == 10, "Should report 10 R1CS constraints")
    expect(stats.plonkGates > 0, "Should produce Plonk gates")

    let sat = R1CSToPlonkConverter.verifySatisfaction(circuit: circuit, witness: witness)
    expect(sat, "Converted mixed Plonk circuit should be satisfied")

    print("  [OK] mixed circuit (10 constraints, ratio: \(String(format: "%.1f", stats.conversionRatio))x)")
}

// MARK: - Round-trip: R1CS -> Plonk -> R1CS

private func testRoundTrip() {
    // Build a circuit: x * y = z, then convert R1CS -> Plonk -> R1CS
    let b = SpartanR1CSBuilder()
    let zOut = b.addPublicInput()
    let x = b.addWitness()
    let y = b.addWitness()
    b.mulGate(a: x, b: y, out: zOut)
    let r1cs = b.build()

    let xVal = frFromInt(4)
    let yVal = frFromInt(9)
    let zVal = frMul(xVal, yVal) // 36
    let z = SpartanR1CS.buildZ(publicInputs: [zVal], witness: [xVal, yVal])

    expect(r1cs.isSatisfied(z: z), "Original R1CS should be satisfied")

    // R1CS -> Plonk
    let (plonkCircuit, plonkWitness, _) = R1CSToPlonkConverter.convert(r1cs: r1cs, z: z)
    let plonkSat = R1CSToPlonkConverter.verifySatisfaction(circuit: plonkCircuit, witness: plonkWitness)
    expect(plonkSat, "Intermediate Plonk should be satisfied")

    // Plonk -> R1CS
    let (r1cs2, z2, _) = PlonkToR1CSConverter.convert(
        circuit: plonkCircuit, witness: plonkWitness, numPublic: 1
    )

    let r1csSat = r1cs2.isSatisfied(z: z2)
    expect(r1csSat, "Round-trip R1CS should be satisfied")

    print("  [OK] round-trip R1CS -> Plonk -> R1CS preserves satisfaction")
}

// MARK: - Plonk to R1CS satisfaction check

private func testPlonkToR1CSSatisfaction() {
    // Build Plonk circuit directly: a * b = c, then c + d = e
    let builder = PlonkCircuitBuilder()
    let a = builder.addInput()
    let b = builder.addInput()
    let c = builder.mul(a, b)
    let d = builder.addInput()
    let e = builder.add(c, d)

    let circuit = builder.build()

    // Witness: a=3, b=5, c=15, d=7, e=22
    let aVal = frFromInt(3)
    let bVal = frFromInt(5)
    let cVal = frMul(aVal, bVal)
    let dVal = frFromInt(7)
    let eVal = frAdd(cVal, dVal)

    var witness = [Int: Fr]()
    witness[a] = aVal
    witness[b] = bVal
    witness[c] = cVal
    witness[d] = dVal
    witness[e] = eVal

    // Convert to R1CS
    let (r1cs, z, stats) = PlonkToR1CSConverter.convert(
        circuit: circuit, witness: witness
    )

    expect(stats.plonkGates == 2, "Should have 2 Plonk gates")
    expect(stats.r1csConstraints > 0, "Should produce R1CS constraints")
    expect(r1cs.isSatisfied(z: z), "Converted R1CS should be satisfied")

    print("  [OK] Plonk -> R1CS produces satisfiable constraint system")
}

// MARK: - R1CS to Plonk satisfaction check

private func testR1CSToPlonkSatisfaction() {
    // Use the synthetic R1CS builder for a chain of squares
    let (r1cs, publicInputs, witnessVals) = SpartanR1CSBuilder.syntheticR1CS(numConstraints: 5)
    let z = SpartanR1CS.buildZ(publicInputs: publicInputs, witness: witnessVals)

    expect(r1cs.isSatisfied(z: z), "Synthetic R1CS should be satisfied")

    let (circuit, witness, stats) = R1CSToPlonkConverter.convert(r1cs: r1cs, z: z)

    expect(stats.r1csConstraints == 6, "Synthetic(5) has 6 constraints (5 mul + 1 output)")

    let sat = R1CSToPlonkConverter.verifySatisfaction(circuit: circuit, witness: witness)
    expect(sat, "Converted synthetic Plonk should be satisfied")

    print("  [OK] R1CS to Plonk satisfaction verified for chain-of-squares")
}

// MARK: - Conversion statistics

private func testConversionStats() {
    // Build a known circuit and check stats
    let (r1cs, gen) = SpartanR1CSBuilder.exampleQuadratic()
    let xVal = frFromInt(7)
    let (pub, wit) = gen(xVal)
    let z = SpartanR1CS.buildZ(publicInputs: pub, witness: wit)

    let (_, _, stats) = R1CSToPlonkConverter.convert(r1cs: r1cs, z: z)

    expect(stats.r1csConstraints == 3, "exampleQuadratic has 3 constraints")
    expect(stats.plonkGates >= 3, "Should produce at least 3 Plonk gates")
    expect(stats.conversionRatio >= 1.0, "Conversion ratio should be >= 1")
    expect(stats.linearConstraints + stats.quadraticConstraints + stats.complexConstraints == 3,
           "Constraint type counts should sum to total")

    print("  [OK] conversion stats: \(stats.r1csConstraints) R1CS -> \(stats.plonkGates) Plonk (ratio \(String(format: "%.1f", stats.conversionRatio))x)")
}

// MARK: - Constant constraints

private func testConstantConstraints() {
    // Build R1CS with a constant constraint: 1 * x = 5
    // i.e., A=[var0:1], B=[x:1], C=[var0:5]
    let b = SpartanR1CSBuilder()
    let x = b.addWitness()
    b.addConstraint(
        a: [(0, Fr.one)],
        b: [(x, Fr.one)],
        c: [(0, frFromInt(5))]
    )
    let r1cs = b.build()

    let z = SpartanR1CS.buildZ(publicInputs: [], witness: [frFromInt(5)])
    expect(r1cs.isSatisfied(z: z), "Constant R1CS should be satisfied")

    let (circuit, witness, _) = R1CSToPlonkConverter.convert(r1cs: r1cs, z: z)
    let sat = R1CSToPlonkConverter.verifySatisfaction(circuit: circuit, witness: witness)
    expect(sat, "Converted constant constraint Plonk should be satisfied")

    print("  [OK] constant constraint 1*x=5")
}

// MARK: - Multi-term complex constraint

private func testMultiTermComplex() {
    // Build R1CS: (a + b) * (c + d) = e
    // Both A and B have multiple terms
    let b = SpartanR1CSBuilder()
    let e = b.addPublicInput()
    let a = b.addWitness()
    let bVar = b.addWitness()
    let c = b.addWitness()
    let d = b.addWitness()
    b.addConstraint(
        a: [(a, Fr.one), (bVar, Fr.one)],
        b: [(c, Fr.one), (d, Fr.one)],
        c: [(e, Fr.one)]
    )
    let r1cs = b.build()

    // a=2, b=3, c=4, d=5 => (2+3)*(4+5) = 45
    let aVal = frFromInt(2)
    let bVal = frFromInt(3)
    let cVal = frFromInt(4)
    let dVal = frFromInt(5)
    let eVal = frMul(frAdd(aVal, bVal), frAdd(cVal, dVal)) // 5*9 = 45

    let z = SpartanR1CS.buildZ(publicInputs: [eVal], witness: [aVal, bVal, cVal, dVal])
    expect(r1cs.isSatisfied(z: z), "Complex R1CS should be satisfied")

    let (circuit, witness, stats) = R1CSToPlonkConverter.convert(r1cs: r1cs, z: z)

    expect(stats.complexConstraints == 1, "Should detect 1 complex constraint")

    let sat = R1CSToPlonkConverter.verifySatisfaction(circuit: circuit, witness: witness)
    expect(sat, "Converted complex constraint Plonk should be satisfied")

    print("  [OK] multi-term complex (a+b)*(c+d)=e")
}
