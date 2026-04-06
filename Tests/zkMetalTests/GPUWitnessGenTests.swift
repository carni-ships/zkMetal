// GPUWitnessGenTests — Tests for GPU-accelerated arithmetic circuit witness generation
//
// Tests topological sort, parallel evaluation, public input injection,
// constraint validation, and diamond dependency patterns.

import Foundation
import zkMetal

// MARK: - Test: Simple Addition Circuit

private func testSimpleAddition() {
    suite("GPUWitnessGen: Simple Addition")

    // Circuit: wire2 = wire0 + wire1
    // wire0 = 3, wire1 = 5, expected wire2 = 8
    let three = frFromInt(3)
    let five = frFromInt(5)

    let gates = [
        ArithGate(type: .publicInput, output: 0, inputIndex: 0),
        ArithGate(type: .publicInput, output: 1, inputIndex: 1),
        ArithGate(type: .add, output: 2, left: 0, right: 1),
    ]

    do {
        let engine = try GPUWitnessGenEngine(cpuOnly: true)
        let result = try engine.generateWitness(gates: gates, numWires: 3,
                                                publicInputs: [three, five])
        let expected = frAdd(three, five)
        expect(frEqual(result.wires[0], three), "wire0 = 3")
        expect(frEqual(result.wires[1], five), "wire1 = 5")
        expect(frEqual(result.wires[2], expected), "wire2 = 3 + 5 = 8")
        expect(result.numLayers >= 1, "at least 1 layer")
    } catch {
        expect(false, "Simple addition threw: \(error)")
    }
}

// MARK: - Test: Multiplication Chain

private func testMultiplicationChain() {
    suite("GPUWitnessGen: Multiplication Chain")

    // Circuit: wire0=const(2), wire1=const(3),
    //          wire2 = wire0 * wire1 = 6
    //          wire3 = wire2 * wire0 = 12
    //          wire4 = wire3 * wire1 = 36
    let two = frFromInt(2)
    let three = frFromInt(3)

    let gates = [
        ArithGate(type: .constant, output: 0, constantValue: two),
        ArithGate(type: .constant, output: 1, constantValue: three),
        ArithGate(type: .mul, output: 2, left: 0, right: 1),
        ArithGate(type: .mul, output: 3, left: 2, right: 0),
        ArithGate(type: .mul, output: 4, left: 3, right: 1),
    ]

    do {
        let engine = try GPUWitnessGenEngine(cpuOnly: true)
        let result = try engine.generateWitness(gates: gates, numWires: 5)

        let six = frMul(two, three)
        let twelve = frMul(six, two)
        let thirtySix = frMul(twelve, three)

        expect(frEqual(result.wires[0], two), "wire0 = 2")
        expect(frEqual(result.wires[1], three), "wire1 = 3")
        expect(frEqual(result.wires[2], six), "wire2 = 2*3 = 6")
        expect(frEqual(result.wires[3], twelve), "wire3 = 6*2 = 12")
        expect(frEqual(result.wires[4], thirtySix), "wire4 = 12*3 = 36")

        // Chain has 3 sequential dependencies, so at least 3 layers
        expect(result.numLayers >= 3, "mul chain needs >= 3 layers")
    } catch {
        expect(false, "Multiplication chain threw: \(error)")
    }
}

// MARK: - Test: Public Input Injection

private func testPublicInputInjection() {
    suite("GPUWitnessGen: Public Input Injection")

    // Circuit: 3 public inputs, then add them pairwise
    // wire0=pub[0], wire1=pub[1], wire2=pub[2]
    // wire3 = wire0 + wire1
    // wire4 = wire1 + wire2
    // wire5 = wire3 * wire4
    let a = frFromInt(7)
    let b = frFromInt(11)
    let c = frFromInt(13)

    let gates = [
        ArithGate(type: .publicInput, output: 0, inputIndex: 0),
        ArithGate(type: .publicInput, output: 1, inputIndex: 1),
        ArithGate(type: .publicInput, output: 2, inputIndex: 2),
        ArithGate(type: .add, output: 3, left: 0, right: 1),
        ArithGate(type: .add, output: 4, left: 1, right: 2),
        ArithGate(type: .mul, output: 5, left: 3, right: 4),
    ]

    do {
        let engine = try GPUWitnessGenEngine(cpuOnly: true)
        let result = try engine.generateWitness(gates: gates, numWires: 6,
                                                publicInputs: [a, b, c])

        expect(frEqual(result.wires[0], a), "wire0 = pub[0] = 7")
        expect(frEqual(result.wires[1], b), "wire1 = pub[1] = 11")
        expect(frEqual(result.wires[2], c), "wire2 = pub[2] = 13")

        let ab = frAdd(a, b)         // 18
        let bc = frAdd(b, c)         // 24
        let product = frMul(ab, bc)  // 432

        expect(frEqual(result.wires[3], ab), "wire3 = 7+11 = 18")
        expect(frEqual(result.wires[4], bc), "wire4 = 11+13 = 24")
        expect(frEqual(result.wires[5], product), "wire5 = 18*24 = 432")
    } catch {
        expect(false, "Public input injection threw: \(error)")
    }
}

// MARK: - Test: Constraint Validation Pass/Fail

private func testConstraintValidation() {
    suite("GPUWitnessGen: Constraint Validation")

    let two = frFromInt(2)
    let three = frFromInt(3)
    let six = frMul(two, three)
    let five = frFromInt(5)

    let wires = [two, three, six, five]

    // Constraint: wire0 * wire1 == wire2 (2 * 3 = 6) -- should pass
    let goodConstraints = [
        ArithConstraint(type: .mul, leftWire: 0, rightWire: 1, outputWire: 2)
    ]

    // Constraint: wire0 * wire1 == wire3 (2 * 3 != 5) -- should fail
    let badConstraints = [
        ArithConstraint(type: .mul, leftWire: 0, rightWire: 1, outputWire: 3)
    ]

    // Add constraint: wire0 + wire1 == wire3 (2 + 3 = 5) -- should pass
    let addConstraint = [
        ArithConstraint(type: .add, leftWire: 0, rightWire: 1, outputWire: 3)
    ]

    do {
        let engine = try GPUWitnessGenEngine(cpuOnly: true)

        let passResult = engine.validateWitness(wires: wires, constraints: goodConstraints)
        expect(passResult, "mul constraint 2*3=6 should pass")

        let failResult = engine.validateWitness(wires: wires, constraints: badConstraints)
        expect(!failResult, "mul constraint 2*3!=5 should fail")

        let addResult = engine.validateWitness(wires: wires, constraints: addConstraint)
        expect(addResult, "add constraint 2+3=5 should pass")
    } catch {
        expect(false, "Constraint validation threw: \(error)")
    }
}

// MARK: - Test: Diamond Dependency

private func testDiamondDependency() {
    suite("GPUWitnessGen: Diamond Dependency")

    // Diamond DAG:
    //        wire0 (const 5)
    //       /            \
    //   wire1=wire0+wire0  wire2=wire0*wire0
    //       \            /
    //     wire3 = wire1 + wire2
    //
    // wire0 = 5
    // wire1 = 5 + 5 = 10
    // wire2 = 5 * 5 = 25
    // wire3 = 10 + 25 = 35

    let five = frFromInt(5)

    let gates = [
        ArithGate(type: .constant, output: 0, constantValue: five),
        ArithGate(type: .add, output: 1, left: 0, right: 0),
        ArithGate(type: .mul, output: 2, left: 0, right: 0),
        ArithGate(type: .add, output: 3, left: 1, right: 2),
    ]

    do {
        let engine = try GPUWitnessGenEngine(cpuOnly: true)
        let result = try engine.generateWitness(gates: gates, numWires: 4)

        let ten = frAdd(five, five)
        let twentyFive = frMul(five, five)
        let thirtyFive = frAdd(ten, twentyFive)

        expect(frEqual(result.wires[0], five), "wire0 = 5")
        expect(frEqual(result.wires[1], ten), "wire1 = 5+5 = 10")
        expect(frEqual(result.wires[2], twentyFive), "wire2 = 5*5 = 25")
        expect(frEqual(result.wires[3], thirtyFive), "wire3 = 10+25 = 35")

        // Layer 0: const(wire0)
        // Layer 1: add(wire1), mul(wire2)  -- independent, same layer
        // Layer 2: add(wire3)
        expect(result.numLayers == 3, "diamond has 3 layers (got \(result.numLayers))")

        // Validate constraints
        let constraints = [
            ArithConstraint(type: .add, leftWire: 0, rightWire: 0, outputWire: 1),
            ArithConstraint(type: .mul, leftWire: 0, rightWire: 0, outputWire: 2),
            ArithConstraint(type: .add, leftWire: 1, rightWire: 2, outputWire: 3),
        ]
        let valid = engine.validateWitness(wires: result.wires, constraints: constraints)
        expect(valid, "diamond witness satisfies all constraints")
    } catch {
        expect(false, "Diamond dependency threw: \(error)")
    }
}

// MARK: - Public Entry Point

public func runGPUWitnessGenTests() {
    testSimpleAddition()
    testMultiplicationChain()
    testPublicInputInjection()
    testConstraintValidation()
    testDiamondDependency()
}
