// GPUPlonkCopyConstraintTests — Tests for GPU-accelerated copy constraint engine
//
// Tests cover:
//   1. Wire assignment extraction from circuit
//   2. Cycle detection with union-find
//   3. Sigma polynomial computation (identity permutation)
//   4. Sigma polynomial computation (with copy constraints)
//   5. Copy constraint satisfaction verification (pass case)
//   6. Copy constraint satisfaction verification (fail case)
//   7. Multi-table merging
//   8. Sigma consistency with witness
//   9. Sigma from circuit integration
//  10. Larger domain stress test
//  11. Transitive copy constraints
//  12. GPU sigma computation path

import zkMetal
import Foundation

public func runGPUPlonkCopyConstraintTests() {
    suite("GPU Plonk Copy Constraint Engine")

    testWireAssignmentExtraction()
    testCycleDetectionSimple()
    testCycleDetectionTransitive()
    testSigmaIdentityPermutation()
    testSigmaWithCopyConstraints()
    testVerifyCopyConstraintsPass()
    testVerifyCopyConstraintsFail()
    testMultiTableMerge()
    testSigmaConsistencyPass()
    testSigmaConsistencyFail()
    testSigmaFromCircuit()
    testLargerDomain()
    testEmptyConstraints()
    testSingleCycleMultiplePositions()
    testGPUSigmaComputation()
}

// MARK: - Wire Assignment Extraction

func testWireAssignmentExtraction() {
    let engine = GPUPlonkCopyConstraintEngine()

    // Simple circuit: 2 gates, 3 wires, 4 variables
    // Gate 0: a=var0, b=var1, c=var2
    // Gate 1: a=var2, b=var3, c=var0
    let gates = [
        PlonkGate(qL: Fr.one, qR: Fr.one, qO: frNeg(Fr.one), qM: Fr.zero, qC: Fr.zero),
        PlonkGate(qL: Fr.one, qR: Fr.one, qO: frNeg(Fr.one), qM: Fr.zero, qC: Fr.zero),
    ]
    let wireAssignments = [[0, 1, 2], [2, 3, 0]]
    let circuit = PlonkCircuit(gates: gates, copyConstraints: [], wireAssignments: wireAssignments)

    let witness = [frFromInt(5), frFromInt(7), frFromInt(12), frFromInt(3)]
    let wireValues = engine.extractWireValues(circuit: circuit, witness: witness)

    expect(wireValues.count == 3, "3 wire columns extracted")
    expect(wireValues[0].count == 2, "2 rows per wire")

    // Wire 0 (a): row0=var0=5, row1=var2=12
    expect(frEqual(wireValues[0][0], frFromInt(5)), "wire a row 0 = 5")
    expect(frEqual(wireValues[0][1], frFromInt(12)), "wire a row 1 = 12")

    // Wire 1 (b): row0=var1=7, row1=var3=3
    expect(frEqual(wireValues[1][0], frFromInt(7)), "wire b row 0 = 7")
    expect(frEqual(wireValues[1][1], frFromInt(3)), "wire b row 1 = 3")

    // Wire 2 (c): row0=var2=12, row1=var0=5
    expect(frEqual(wireValues[2][0], frFromInt(12)), "wire c row 0 = 12")
    expect(frEqual(wireValues[2][1], frFromInt(5)), "wire c row 1 = 5")
}

// MARK: - Cycle Detection

func testCycleDetectionSimple() {
    let engine = GPUPlonkCopyConstraintEngine()

    // Two constraints forming separate pairs:
    // (wire0, row0) <-> (wire1, row1)
    // (wire0, row2) <-> (wire2, row3)
    let constraints = [
        PlonkCopyConstraint(srcWire: 0, srcRow: 0, dstWire: 1, dstRow: 1),
        PlonkCopyConstraint(srcWire: 0, srcRow: 2, dstWire: 2, dstRow: 3),
    ]

    let cycles = engine.detectCycles(constraints: constraints, numWires: 3, domainSize: 4)

    expect(cycles.count == 2, "Two separate cycles detected")

    // Each cycle has 2 positions
    let sizes = cycles.map { $0.count }.sorted()
    expect(sizes == [2, 2], "Both cycles have 2 positions")
}

func testCycleDetectionTransitive() {
    let engine = GPUPlonkCopyConstraintEngine()

    // Three constraints forming a chain: A <-> B, B <-> C => A,B,C in one cycle
    // (wire0, row0) <-> (wire1, row0)
    // (wire1, row0) <-> (wire2, row0)
    let constraints = [
        PlonkCopyConstraint(srcWire: 0, srcRow: 0, dstWire: 1, dstRow: 0),
        PlonkCopyConstraint(srcWire: 1, srcRow: 0, dstWire: 2, dstRow: 0),
    ]

    let cycles = engine.detectCycles(constraints: constraints, numWires: 3, domainSize: 4)

    expect(cycles.count == 1, "Transitive constraints merge into one cycle")
    expect(cycles[0].count == 3, "Transitive cycle has 3 positions")
}

// MARK: - Sigma Polynomial Computation

func testSigmaIdentityPermutation() {
    let engine = GPUPlonkCopyConstraintEngine()

    let numWires = 3
    let domainSize = 8
    let logN = 3
    let omega = computeNthRootOfUnity(logN: logN)

    // No constraints => identity permutation
    let result = engine.computeSigmaPolynomials(
        constraints: [],
        numWires: numWires,
        domainSize: domainSize
    )

    expect(result.sigmas.count == numWires, "3 sigma columns")
    expect(result.sigmas[0].count == domainSize, "sigma column length = domain size")
    expect(result.cycles.isEmpty, "No cycles for identity permutation")

    // Verify identity: sigma[0][i] = omega^i, sigma[1][i] = 2*omega^i, sigma[2][i] = 3*omega^i
    var domainVal = Fr.one
    for i in 0..<domainSize {
        let expected0 = domainVal
        let expected1 = frMul(frFromInt(2), domainVal)
        let expected2 = frMul(frFromInt(3), domainVal)

        expect(frEqual(result.sigmas[0][i], expected0), "sigma[0][\(i)] = omega^\(i)")
        expect(frEqual(result.sigmas[1][i], expected1), "sigma[1][\(i)] = 2*omega^\(i)")
        expect(frEqual(result.sigmas[2][i], expected2), "sigma[2][\(i)] = 3*omega^\(i)")

        domainVal = frMul(domainVal, omega)
    }
}

func testSigmaWithCopyConstraints() {
    let engine = GPUPlonkCopyConstraintEngine()

    let numWires = 3
    let domainSize = 8
    let logN = 3
    let omega = computeNthRootOfUnity(logN: logN)

    var domain = [Fr](repeating: Fr.zero, count: domainSize)
    domain[0] = Fr.one
    for i in 1..<domainSize { domain[i] = frMul(domain[i - 1], omega) }

    // Copy constraint: (wire0, row0) <-> (wire1, row2)
    // This creates a 2-cycle that swaps sigma values at those positions.
    let constraints = [
        PlonkCopyConstraint(srcWire: 0, srcRow: 0, dstWire: 1, dstRow: 2),
    ]

    let result = engine.computeSigmaPolynomials(
        constraints: constraints,
        numWires: numWires,
        domainSize: domainSize
    )

    expect(result.cycles.count == 1, "One cycle from one constraint")

    // After the swap:
    //   sigma[0][0] should point to (wire1, row2): 2 * omega^2
    //   sigma[1][2] should point to (wire0, row0): 1 * omega^0 = 1
    let k1 = frFromInt(2)
    let expectedSigma0_0 = frMul(k1, domain[2])   // 2 * omega^2
    let expectedSigma1_2 = domain[0]               // omega^0 = 1

    expect(frEqual(result.sigmas[0][0], expectedSigma0_0), "sigma[0][0] points to (wire1, row2)")
    expect(frEqual(result.sigmas[1][2], expectedSigma1_2), "sigma[1][2] points to (wire0, row0)")

    // Other positions should remain identity
    let expectedSigma0_1 = domain[1]  // omega^1
    expect(frEqual(result.sigmas[0][1], expectedSigma0_1), "sigma[0][1] unchanged (identity)")
}

// MARK: - Copy Constraint Verification

func testVerifyCopyConstraintsPass() {
    let engine = GPUPlonkCopyConstraintEngine()

    let numWires = 3
    let n = 4

    // Wire values where constraints are satisfied
    var wireValues = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: n), count: numWires)
    wireValues[0][0] = frFromInt(42)
    wireValues[0][1] = frFromInt(7)
    wireValues[0][2] = frFromInt(100)
    wireValues[0][3] = frFromInt(99)
    wireValues[1][0] = frFromInt(42)  // same as wire0, row0
    wireValues[1][1] = frFromInt(100) // same as wire0, row2
    wireValues[1][2] = frFromInt(50)
    wireValues[1][3] = frFromInt(60)
    wireValues[2][0] = frFromInt(10)
    wireValues[2][1] = frFromInt(20)
    wireValues[2][2] = frFromInt(30)
    wireValues[2][3] = frFromInt(40)

    let constraints = [
        PlonkCopyConstraint(srcWire: 0, srcRow: 0, dstWire: 1, dstRow: 0),  // both 42
        PlonkCopyConstraint(srcWire: 0, srcRow: 2, dstWire: 1, dstRow: 1),  // both 100
    ]

    let result = engine.verifyCopyConstraints(constraints: constraints, wireValues: wireValues)

    expect(result.satisfied, "Copy constraints should be satisfied")
    expect(result.failingIndices.isEmpty, "No failing constraints")
}

func testVerifyCopyConstraintsFail() {
    let engine = GPUPlonkCopyConstraintEngine()

    let numWires = 3
    let n = 4

    var wireValues = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: n), count: numWires)
    wireValues[0][0] = frFromInt(42)
    wireValues[0][1] = frFromInt(7)
    wireValues[1][0] = frFromInt(99)  // NOT equal to wire0, row0

    let constraints = [
        PlonkCopyConstraint(srcWire: 0, srcRow: 0, dstWire: 1, dstRow: 0),  // 42 != 99
    ]

    let result = engine.verifyCopyConstraints(constraints: constraints, wireValues: wireValues)

    expect(!result.satisfied, "Copy constraints should fail")
    expect(result.failingIndices.count == 1, "One failing constraint")
    expect(result.failingIndices[0] == 0, "Constraint 0 fails")

    // Check failing values
    expect(frEqual(result.failingValues[0].0, frFromInt(42)), "Failing src value = 42")
    expect(frEqual(result.failingValues[0].1, frFromInt(99)), "Failing dst value = 99")
}

// MARK: - Multi-Table Merge

func testMultiTableMerge() {
    let engine = GPUPlonkCopyConstraintEngine()

    let numWires = 3
    let domainSize = 8

    // Table A: wire0,row0 <-> wire1,row1
    let tableA = CopyConstraintTable(id: "A", constraints: [
        PlonkCopyConstraint(srcWire: 0, srcRow: 0, dstWire: 1, dstRow: 1),
    ])

    // Table B: wire1,row1 <-> wire2,row2 (transitively links with table A)
    let tableB = CopyConstraintTable(id: "B", constraints: [
        PlonkCopyConstraint(srcWire: 1, srcRow: 1, dstWire: 2, dstRow: 2),
    ])

    let merged = engine.mergeTables(
        tables: [tableA, tableB],
        numWires: numWires,
        domainSize: domainSize
    )

    // Both tables' constraints should be merged transitively
    expect(merged.cycles.count == 1, "Transitive merge creates one cycle")
    expect(merged.cycles[0].count == 3, "Merged cycle has 3 positions")
}

// MARK: - Sigma Consistency

func testSigmaConsistencyPass() {
    let engine = GPUPlonkCopyConstraintEngine()

    let numWires = 3
    let domainSize = 8

    // Constraint: wire0,row0 <-> wire1,row1
    let constraints = [
        PlonkCopyConstraint(srcWire: 0, srcRow: 0, dstWire: 1, dstRow: 1),
    ]

    let sigmaResult = engine.computeSigmaPolynomials(
        constraints: constraints,
        numWires: numWires,
        domainSize: domainSize
    )

    // Witness where constrained positions have equal values
    var wireValues = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: domainSize), count: numWires)
    for j in 0..<numWires {
        for i in 0..<domainSize {
            wireValues[j][i] = frFromInt(UInt64(j * domainSize + i + 1))
        }
    }
    // Make constrained positions equal
    wireValues[0][0] = frFromInt(42)
    wireValues[1][1] = frFromInt(42)

    let consistent = engine.verifySigmaConsistency(
        sigmaResult: sigmaResult,
        wireValues: wireValues
    )
    expect(consistent, "Sigma consistency passes when constrained values are equal")
}

func testSigmaConsistencyFail() {
    let engine = GPUPlonkCopyConstraintEngine()

    let numWires = 3
    let domainSize = 8

    let constraints = [
        PlonkCopyConstraint(srcWire: 0, srcRow: 0, dstWire: 1, dstRow: 1),
    ]

    let sigmaResult = engine.computeSigmaPolynomials(
        constraints: constraints,
        numWires: numWires,
        domainSize: domainSize
    )

    // Witness where constrained positions have DIFFERENT values
    var wireValues = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: domainSize), count: numWires)
    wireValues[0][0] = frFromInt(42)
    wireValues[1][1] = frFromInt(99)

    let consistent = engine.verifySigmaConsistency(
        sigmaResult: sigmaResult,
        wireValues: wireValues
    )
    expect(!consistent, "Sigma consistency fails when constrained values differ")
}

// MARK: - Sigma from Circuit

func testSigmaFromCircuit() {
    let engine = GPUPlonkCopyConstraintEngine()

    // Circuit with 4 gates, copy constraint between position (gate0, wire_c) and (gate1, wire_a)
    // In PlonkCircuit encoding: flat index = gateIndex * 3 + wireType
    // (gate0, c=2) -> flat 0*3+2=2, (gate1, a=0) -> flat 1*3+0=3
    let gates = [
        PlonkGate(qL: Fr.one, qR: Fr.one, qO: frNeg(Fr.one), qM: Fr.zero, qC: Fr.zero),
        PlonkGate(qL: Fr.one, qR: Fr.zero, qO: frNeg(Fr.one), qM: Fr.zero, qC: Fr.zero),
        PlonkGate(qL: Fr.zero, qR: Fr.zero, qO: Fr.zero, qM: Fr.zero, qC: Fr.zero),
        PlonkGate(qL: Fr.zero, qR: Fr.zero, qO: Fr.zero, qM: Fr.zero, qC: Fr.zero),
    ]
    let wireAssignments = [
        [0, 1, 2],
        [2, 3, 4],
        [0, 0, 0],
        [0, 0, 0],
    ]
    let copyConstraints: [(Int, Int)] = [(2, 3)]  // flat encoding
    let circuit = PlonkCircuit(
        gates: gates,
        copyConstraints: copyConstraints,
        wireAssignments: wireAssignments
    )

    let sigmaResult = engine.sigmaFromCircuit(circuit: circuit, numWires: 3)

    expect(sigmaResult.numWires == 3, "3 wires from circuit")
    expect(sigmaResult.domainSize == 4, "Domain padded to 4 (next pow2 of 4)")
    expect(sigmaResult.cycles.count >= 1, "At least one cycle from copy constraint")
}

// MARK: - Larger Domain

func testLargerDomain() {
    let engine = GPUPlonkCopyConstraintEngine()

    let numWires = 3
    let domainSize = 64  // 2^6

    // Create a chain of constraints across all rows of wire 0
    var constraints = [PlonkCopyConstraint]()
    for i in 0..<(domainSize - 1) {
        constraints.append(PlonkCopyConstraint(srcWire: 0, srcRow: i, dstWire: 0, dstRow: i + 1))
    }

    let result = engine.computeSigmaPolynomials(
        constraints: constraints,
        numWires: numWires,
        domainSize: domainSize
    )

    // All wire0 rows should be in one cycle
    expect(result.cycles.count == 1, "One large cycle")
    expect(result.cycles[0].count == domainSize, "Cycle spans all rows of wire 0")

    // Verify sigma is NOT identity for wire 0
    let logN = 6
    let omega = computeNthRootOfUnity(logN: logN)
    var domain = [Fr](repeating: Fr.zero, count: domainSize)
    domain[0] = Fr.one
    for i in 1..<domainSize { domain[i] = frMul(domain[i - 1], omega) }

    var nonIdentityCount = 0
    for i in 0..<domainSize {
        if !frEqual(result.sigmas[0][i], domain[i]) {
            nonIdentityCount += 1
        }
    }
    expect(nonIdentityCount > 0, "Sigma differs from identity for constrained wire")

    // Wire 1 and 2 should remain identity
    let k1 = frFromInt(2)
    let k2 = frFromInt(3)
    var wire1Identity = true
    var wire2Identity = true
    for i in 0..<domainSize {
        if !frEqual(result.sigmas[1][i], frMul(k1, domain[i])) { wire1Identity = false }
        if !frEqual(result.sigmas[2][i], frMul(k2, domain[i])) { wire2Identity = false }
    }
    expect(wire1Identity, "Wire 1 sigma is identity (unconstrained)")
    expect(wire2Identity, "Wire 2 sigma is identity (unconstrained)")

    // Verify with a consistent witness (all wire0 values equal)
    var wireValues = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: domainSize), count: numWires)
    let sharedVal = frFromInt(77)
    for i in 0..<domainSize {
        wireValues[0][i] = sharedVal
        wireValues[1][i] = frFromInt(UInt64(100 + i))
        wireValues[2][i] = frFromInt(UInt64(200 + i))
    }

    let consistent = engine.verifySigmaConsistency(
        sigmaResult: result,
        wireValues: wireValues
    )
    expect(consistent, "Large cycle is consistent when all values equal")
}

// MARK: - Edge Cases

func testEmptyConstraints() {
    let engine = GPUPlonkCopyConstraintEngine()

    let result = engine.computeSigmaPolynomials(
        constraints: [],
        numWires: 3,
        domainSize: 8
    )
    expect(result.cycles.isEmpty, "No cycles for empty constraints")
    expect(result.sigmas.count == 3, "Still produces 3 sigma columns")

    let checkResult = engine.verifyCopyConstraints(constraints: [], wireValues: [[Fr.one]])
    expect(checkResult.satisfied, "Empty constraints always satisfied")
}

func testSingleCycleMultiplePositions() {
    let engine = GPUPlonkCopyConstraintEngine()

    // Create a 4-position cycle: w0r0 <-> w1r0 <-> w2r0 <-> w0r1
    let constraints = [
        PlonkCopyConstraint(srcWire: 0, srcRow: 0, dstWire: 1, dstRow: 0),
        PlonkCopyConstraint(srcWire: 1, srcRow: 0, dstWire: 2, dstRow: 0),
        PlonkCopyConstraint(srcWire: 2, srcRow: 0, dstWire: 0, dstRow: 1),
    ]

    let cycles = engine.detectCycles(constraints: constraints, numWires: 3, domainSize: 4)
    expect(cycles.count == 1, "One cycle from chain of 4 positions")
    expect(cycles[0].count == 4, "Cycle has 4 positions")

    // Verify sigma with consistent witness
    let sigmaResult = engine.computeSigmaPolynomials(
        constraints: constraints,
        numWires: 3,
        domainSize: 4
    )

    var wireValues = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: 4), count: 3)
    let val = frFromInt(55)
    wireValues[0][0] = val
    wireValues[1][0] = val
    wireValues[2][0] = val
    wireValues[0][1] = val

    let consistent = engine.verifySigmaConsistency(
        sigmaResult: sigmaResult,
        wireValues: wireValues
    )
    expect(consistent, "4-position cycle consistent with equal values")
}

// MARK: - GPU Sigma Path

func testGPUSigmaComputation() {
    let engine = GPUPlonkCopyConstraintEngine()

    let numWires = 3
    let domainSize = 8

    let constraints = [
        PlonkCopyConstraint(srcWire: 0, srcRow: 0, dstWire: 1, dstRow: 1),
        PlonkCopyConstraint(srcWire: 0, srcRow: 3, dstWire: 2, dstRow: 5),
    ]

    // Compute via both paths
    let cpuResult = engine.computeSigmaPolynomials(
        constraints: constraints,
        numWires: numWires,
        domainSize: domainSize
    )

    let gpuResult = engine.computeSigmaPolynomialsGPU(
        constraints: constraints,
        numWires: numWires,
        domainSize: domainSize
    )

    // Both should produce the same sigma values
    expect(gpuResult.sigmas.count == cpuResult.sigmas.count, "Same number of sigma columns")

    var allMatch = true
    for j in 0..<numWires {
        for i in 0..<domainSize {
            if !frEqual(gpuResult.sigmas[j][i], cpuResult.sigmas[j][i]) {
                allMatch = false
                break
            }
        }
        if !allMatch { break }
    }
    expect(allMatch, "GPU sigma matches CPU sigma")

    // Same cycle count
    expect(gpuResult.cycles.count == cpuResult.cycles.count, "Same number of cycles")
}
