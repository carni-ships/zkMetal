// GPUPlonky2Engine Tests — GPU-accelerated Plonky2-style proving engine
//
// Tests:
//   - Arithmetic gate evaluation
//   - Wire routing and copy constraints
//   - FRI commitment generation
//   - Proof generation (x^2 = y end-to-end)
//   - Recursive circuit representation

import Foundation
import zkMetal

public func runGPUPlonky2EngineTests() {
    suite("GPUPlonky2Engine — Arithmetic gate evaluation")
    testPlonky2ArithmeticGate()

    suite("GPUPlonky2Engine — Wire routing and copy constraints")
    testPlonky2WireRouting()

    suite("GPUPlonky2Engine — FRI commitment in proof")
    testPlonky2FRICommitment()

    suite("GPUPlonky2Engine — Proof generation (x^2 = y)")
    testPlonky2ProofGeneration()

    suite("GPUPlonky2Engine — Recursive circuit representation")
    testPlonky2RecursiveCircuit()
}

// MARK: - Arithmetic Gate

func testPlonky2ArithmeticGate() {
    // Build a circuit: c0 * a * b + c1 * c = d
    // With c0=1, c1=1: a*b + c = d
    // a=3, b=5, c=7 => d = 3*5 + 7 = 22
    let circuit = Plonky2ProvingCircuit(numWires: 4, numRoutedWires: 4, degreeBits: 2)
    circuit.addArithmeticGate(a: 0, b: 1, c: 2, d: 3, c0: Gl.one, c1: Gl.one)
    circuit.padToFull()

    let witness = Plonky2WitnessBuilder(circuit: circuit)
    witness.setWire(row: 0, col: 0, value: Gl(v: 3))   // a
    witness.setWire(row: 0, col: 1, value: Gl(v: 5))   // b
    witness.setWire(row: 0, col: 2, value: Gl(v: 7))   // c
    witness.setWire(row: 0, col: 3, value: Gl(v: 22))  // d = 3*5 + 7

    let residuals = Plonky2GateEvaluator.evaluate(circuit: circuit, trace: witness.trace)
    expect(residuals[0] == Gl.zero, "Arithmetic gate satisfied: 3*5 + 7 = 22")

    // Test with wrong output
    let witness2 = Plonky2WitnessBuilder(circuit: circuit)
    witness2.setWire(row: 0, col: 0, value: Gl(v: 3))
    witness2.setWire(row: 0, col: 1, value: Gl(v: 5))
    witness2.setWire(row: 0, col: 2, value: Gl(v: 7))
    witness2.setWire(row: 0, col: 3, value: Gl(v: 99))  // wrong

    let residuals2 = Plonky2GateEvaluator.evaluate(circuit: circuit, trace: witness2.trace)
    expect(residuals2[0] != Gl.zero, "Arithmetic gate fails with wrong output")

    // Test with custom coefficients: c0=2, c1=3 => 2*a*b + 3*c = d
    // a=4, b=6, c=2 => d = 2*24 + 3*2 = 54
    let circuit2 = Plonky2ProvingCircuit(numWires: 4, numRoutedWires: 4, degreeBits: 2)
    circuit2.addArithmeticGate(a: 0, b: 1, c: 2, d: 3, c0: Gl(v: 2), c1: Gl(v: 3))
    circuit2.padToFull()

    let w3 = Plonky2WitnessBuilder(circuit: circuit2)
    w3.setWire(row: 0, col: 0, value: Gl(v: 4))
    w3.setWire(row: 0, col: 1, value: Gl(v: 6))
    w3.setWire(row: 0, col: 2, value: Gl(v: 2))
    w3.setWire(row: 0, col: 3, value: Gl(v: 54))

    let r3 = Plonky2GateEvaluator.evaluate(circuit: circuit2, trace: w3.trace)
    expect(r3[0] == Gl.zero, "Arithmetic gate with custom coefficients: 2*4*6 + 3*2 = 54")
}

// MARK: - Wire Routing

func testPlonky2WireRouting() {
    // Build circuit with copy constraints:
    // Row 0: PI (x)
    // Row 1: PI (y)
    // Row 2: arithmetic gate: x * x = y (wires: a=x, b=x, c=0, d=y)
    // Copy constraints: (0,0)->(2,0), (0,0)->(2,1), (1,0)->(2,3)
    let circuit = Plonky2ProvingCircuit(numWires: 4, numRoutedWires: 4, degreeBits: 2)
    let pi0 = circuit.addPublicInput()  // x at row 0
    let pi1 = circuit.addPublicInput()  // y at row 1
    let arithRow = circuit.addArithmeticGate(a: 0, b: 1, c: 2, d: 3)
    circuit.padToFull()

    // Wire x to arithmetic inputs a and b
    circuit.addCopyConstraint(srcRow: pi0.row, srcCol: pi0.col,
                              dstRow: arithRow, dstCol: 0)
    circuit.addCopyConstraint(srcRow: pi0.row, srcCol: pi0.col,
                              dstRow: arithRow, dstCol: 1)
    // Wire y to arithmetic output d
    circuit.addCopyConstraint(srcRow: pi1.row, srcCol: pi1.col,
                              dstRow: arithRow, dstCol: 3)

    expect(circuit.copyConstraints.count == 3, "3 copy constraints added")
    expect(circuit.numPublicInputs == 2, "2 public inputs")

    // Fill witness: x=7, y=49
    let witness = Plonky2WitnessBuilder(circuit: circuit)
    witness.setPublicInputs([Gl(v: 7), Gl(v: 49)])
    witness.propagateCopyConstraints()

    // Check wire propagation
    expect(witness.trace[arithRow][0] == Gl(v: 7), "Copy constraint propagated x to a")
    expect(witness.trace[arithRow][1] == Gl(v: 7), "Copy constraint propagated x to b")
    expect(witness.trace[arithRow][3] == Gl(v: 49), "Copy constraint propagated y to d")

    // Evaluate gate constraint: 1*7*7 + 1*0 = 49
    let residuals = Plonky2GateEvaluator.evaluate(circuit: circuit, trace: witness.trace)
    expect(residuals[2] == Gl.zero, "Arithmetic gate satisfied after wire routing")
}

// MARK: - FRI Commitment

func testPlonky2FRICommitment() {
    // Build a minimal circuit, generate a proof, check FRI roots are present
    let circuit = Plonky2ProvingCircuit(numWires: 4, numRoutedWires: 2, degreeBits: 2)
    circuit.addPublicInput()
    circuit.addPublicInput()
    circuit.addArithmeticGate(a: 0, b: 1, c: 2, d: 3)
    circuit.padToFull()

    circuit.addCopyConstraint(srcRow: 0, srcCol: 0, dstRow: 2, dstCol: 0)
    circuit.addCopyConstraint(srcRow: 0, srcCol: 0, dstRow: 2, dstCol: 1)
    circuit.addCopyConstraint(srcRow: 1, srcCol: 0, dstRow: 2, dstCol: 3)

    let witness = Plonky2WitnessBuilder(circuit: circuit)
    witness.setPublicInputs([Gl(v: 5), Gl(v: 25)])
    witness.propagateCopyConstraints()

    let engine = GPUPlonky2Engine(friConfig: .standard)
    let proof = engine.prove(circuit: circuit, witness: witness)

    // FRI should produce commit roots (at least 1 fold round for degreeBits=2+rateBits=1 => 3 down to 2)
    expect(proof.friCommitRoots.count >= 1, "FRI produced at least 1 commit root")
    // Each FRI root is a 4-element Poseidon digest
    for root in proof.friCommitRoots {
        expect(root.count == 4, "FRI root is 4-element Poseidon digest")
    }
    // Final poly should be small
    expect(proof.friFinalPoly.count <= 4, "FRI final poly is small (<= 4 elements)")
}

// MARK: - Proof Generation

func testPlonky2ProofGeneration() {
    // End-to-end: prove x^2 = y for x=6, y=36
    let circuit = Plonky2ProvingCircuit(numWires: 4, numRoutedWires: 2, degreeBits: 2)
    let pi0 = circuit.addPublicInput()
    let pi1 = circuit.addPublicInput()
    let arithRow = circuit.addArithmeticGate(a: 0, b: 1, c: 2, d: 3)
    circuit.padToFull()

    circuit.addCopyConstraint(srcRow: pi0.row, srcCol: pi0.col,
                              dstRow: arithRow, dstCol: 0)
    circuit.addCopyConstraint(srcRow: pi0.row, srcCol: pi0.col,
                              dstRow: arithRow, dstCol: 1)
    circuit.addCopyConstraint(srcRow: pi1.row, srcCol: pi1.col,
                              dstRow: arithRow, dstCol: 3)

    let witness = Plonky2WitnessBuilder(circuit: circuit)
    witness.setPublicInputs([Gl(v: 6), Gl(v: 36)])
    witness.propagateCopyConstraints()

    let engine = GPUPlonky2Engine()
    let proof = engine.prove(circuit: circuit, witness: witness)

    // Public inputs present
    expect(proof.publicInputs.count == 2, "Proof has 2 public inputs")
    expect(proof.publicInputs[0] == Gl(v: 6), "PI[0] = 6")
    expect(proof.publicInputs[1] == Gl(v: 36), "PI[1] = 36")

    // Wire commitments: one per wire column
    expect(proof.wireCommitments.count == 4, "4 wire commitments")
    for wc in proof.wireCommitments {
        expect(wc.count == 4, "Wire commitment is 4-element digest")
    }

    // Permutation commitment present
    expect(proof.permutationCommitment.count == 4, "Permutation commitment present")

    // Quotient commitments present
    expect(proof.quotientCommitments.count >= 1, "Quotient commitment present")

    // Circuit digest present
    expect(proof.circuitDigest.count == 4, "Circuit digest present")

    // Openings at zeta present
    expect(proof.openingsAtZeta.count > 0, "Openings at zeta present")
    expect(proof.openingsAtZetaNext.count > 0, "Openings at zeta*omega present")

    // Test serialization round-trip
    let bytes = proof.serialize()
    expect(bytes.count > 0, "Serialized proof is non-empty")
    let restored = Plonky2EngineProof.deserialize(bytes)
    expect(restored != nil, "Proof deserialized successfully")
    if let r = restored {
        expect(r.publicInputs.count == proof.publicInputs.count, "PI count preserved")
        expect(r.publicInputs[0] == proof.publicInputs[0], "PI[0] preserved")
        expect(r.publicInputs[1] == proof.publicInputs[1], "PI[1] preserved")
        expect(r.wireCommitments.count == proof.wireCommitments.count, "Wire commits preserved")
        expect(r.friCommitRoots.count == proof.friCommitRoots.count, "FRI roots preserved")
        expect(r.circuitDigest == proof.circuitDigest, "Circuit digest preserved")
    }
}

// MARK: - Recursive Circuit

func testPlonky2RecursiveCircuit() {
    // Build a circuit and its recursive representation
    let circuit = Plonky2ProvingCircuit(numWires: 4, numRoutedWires: 2, degreeBits: 2)
    circuit.addPublicInput()
    circuit.addPublicInput()
    circuit.addArithmeticGate(a: 0, b: 1, c: 2, d: 3)
    circuit.padToFull()

    let repr = Plonky2RecursiveCircuitRepr(circuit: circuit)
    expect(repr.numPublicInputs == 2, "Recursive repr has 2 PIs")
    expect(repr.degreeBits == 2, "Recursive repr degreeBits = 2")
    expect(repr.numWires == 4, "Recursive repr numWires = 4")
    expect(repr.digest.count == 4, "Recursive repr digest is 4 elements")
    expect(repr.gateTypes.count == 4, "Recursive repr has 4 gate types (padded)")

    // Check that repr matches a proof from this circuit
    let witness = Plonky2WitnessBuilder(circuit: circuit)
    witness.setPublicInputs([Gl(v: 3), Gl(v: 9)])
    witness.propagateCopyConstraints()
    circuit.addCopyConstraint(srcRow: 0, srcCol: 0, dstRow: 2, dstCol: 0)
    circuit.addCopyConstraint(srcRow: 0, srcCol: 0, dstRow: 2, dstCol: 1)
    circuit.addCopyConstraint(srcRow: 1, srcCol: 0, dstRow: 2, dstCol: 3)

    let engine = GPUPlonky2Engine()
    let proof = engine.prove(circuit: circuit, witness: witness)
    expect(repr.matchesProof(proof), "Recursive repr matches proof circuit digest")

    // Check determinism: same circuit produces same digest
    let repr2 = Plonky2RecursiveCircuitRepr(circuit: circuit)
    expect(repr2.digest == repr.digest, "Circuit digest is deterministic")
}
