// GPUPlonkGrandProductTests — Comprehensive tests for GPU-accelerated Plonk grand product engine
//
// Tests cover:
//   1. Identity permutation: z is all ones
//   2. Copy constraint: grand product wraps to 1
//   3. Boundary check: z[0] = 1 enforcement
//   4. Transition constraint validation (valid + invalid)
//   5. Split permutation for large wire counts
//   6. Coset evaluation for quotient polynomial
//   7. GPU vs CPU consistency for varying domain sizes
//   8. Edge cases: single row, two rows, large domain
//   9. Multiple copy constraints
//  10. Wrap-around product check
//  11. Full round-trip: compute + validate
//  12. Config validation

import zkMetal
import Foundation

public func runGPUPlonkGrandProductTests() {
    suite("GPUPlonkGrandProduct — Initialization")

    testEngineCreation()
    testGPUAvailability()

    suite("GPUPlonkGrandProduct — Identity Permutation")

    testIdentityPermutationSmall()
    testIdentityPermutation3Wire()
    suite("GPUPlonkGrandProduct — Boundary Check")

    testBoundaryCheckValid()
    testBoundaryCheckInvalid()
    testBoundaryCheckEmpty()

    suite("GPUPlonkGrandProduct — Copy Constraints")

    testSingleCopyConstraint()
    testMultipleCopyConstraints()
    suite("GPUPlonkGrandProduct — Transition Validation")

    testTransitionValidIdentity()
    testTransitionValidCopyConstraint()
    testTransitionInvalidTampered()
    testTransitionInvalidBoundary()

    suite("GPUPlonkGrandProduct — Split Permutation")

    testSplitPermutation6Wires()
    testSplitPermutationMatchesNonSplit()
    suite("GPUPlonkGrandProduct — Coset Evaluation")

    testCosetEvalIdentity()
    testCosetEvalShifted()
    testCosetEvalEmpty()

    suite("GPUPlonkGrandProduct — GPU vs CPU Consistency")

    testGPUCPUConsistency()

    suite("GPUPlonkGrandProduct — Edge Cases")

    testMinimalDomain2()
    testSingleWire()
    suite("GPUPlonkGrandProduct — Utility Functions")

    testBuildDomain()
    testBuildIdentitySigma()
    testLagrangeBasisFirst()
    testFullRatioProduct()

    suite("GPUPlonkGrandProduct — Round-Trip")

    testFullRoundTrip()
    testSimplifiedInterface()

    suite("GPUPlonkGrandProduct — Config")

    testConfigDefaults()
    testConfigCustom()
}

// MARK: - Initialization Tests

private func testEngineCreation() {
    let engine = GPUPlonkGrandProductEngine()
    expect(true, "Engine created successfully")
    _ = engine
}

private func testGPUAvailability() {
    let engine = GPUPlonkGrandProductEngine()
    // GPU might or might not be available depending on hardware
    // Just verify the property is accessible
    let _ = engine.gpuAvailable
    expect(true, "gpuAvailable property accessible")
}

// MARK: - Identity Permutation Tests

private func testIdentityPermutationSmall() {
    let engine = GPUPlonkGrandProductEngine()

    let logN = 3
    let n = 1 << logN  // 8
    let numWires = 3

    let domain = engine.buildDomain(logN: logN)
    let sigma = engine.buildIdentitySigma(domain: domain, numWires: numWires)

    // Random witness
    let witness: [[Fr]] = (0..<numWires).map { _ in
        (0..<n).map { _ in frFromInt(UInt64.random(in: 1...1000)) }
    }

    let beta = frFromInt(7)
    let gamma = frFromInt(13)

    let config = PlonkGrandProductConfig(numWires: numWires, domainSize: n, useGPU: false)
    let result = engine.computeGrandProduct(
        witness: witness, sigmaPolys: sigma,
        beta: beta, gamma: gamma, config: config
    )

    expectEqual(result.zPoly.count, n, "Identity perm: output length matches")
    expect(result.zPoly[0] == Fr.one, "Identity perm: z[0] = 1")

    var allOnes = true
    for i in 0..<n {
        if result.zPoly[i] != Fr.one { allOnes = false; break }
    }
    expect(allOnes, "Identity perm: z is all ones when sigma = identity")
}

private func testIdentityPermutation3Wire() {
    let engine = GPUPlonkGrandProductEngine()

    let logN = 4
    let n = 1 << logN  // 16
    let numWires = 3

    let domain = engine.buildDomain(logN: logN)
    let sigma = engine.buildIdentitySigma(domain: domain, numWires: numWires)

    let witness: [[Fr]] = (0..<numWires).map { _ in
        (0..<n).map { i in frFromInt(UInt64(i + 1)) }
    }

    let beta = frFromInt(3)
    let gamma = frFromInt(5)

    let config = PlonkGrandProductConfig(numWires: numWires, domainSize: n, useGPU: false)
    let result = engine.computeGrandProduct(
        witness: witness, sigmaPolys: sigma,
        beta: beta, gamma: gamma, config: config
    )

    expect(result.zPoly[0] == Fr.one, "3-wire identity: z[0] = 1")
    // All should be 1 for identity permutation
    var allOnes = true
    for i in 0..<n {
        if result.zPoly[i] != Fr.one { allOnes = false; break }
    }
    expect(allOnes, "3-wire identity: z is all ones")
    expectEqual(result.numPartitions, 1, "3-wire identity: single partition")
}

// MARK: - Boundary Check Tests

private func testBoundaryCheckValid() {
    let engine = GPUPlonkGrandProductEngine()
    var z = [Fr](repeating: frFromInt(42), count: 8)
    z[0] = Fr.one
    expect(engine.checkBoundary(z: z), "Boundary check passes when z[0] = 1")
}

private func testBoundaryCheckInvalid() {
    let engine = GPUPlonkGrandProductEngine()
    let z = [Fr](repeating: frFromInt(42), count: 8)
    expect(!engine.checkBoundary(z: z), "Boundary check fails when z[0] != 1")
}

private func testBoundaryCheckEmpty() {
    let engine = GPUPlonkGrandProductEngine()
    expect(!engine.checkBoundary(z: []), "Boundary check fails for empty z")
}

// MARK: - Copy Constraint Tests

private func testSingleCopyConstraint() {
    let engine = GPUPlonkGrandProductEngine()

    let logN = 3
    let n = 1 << logN
    let numWires = 3

    let domain = engine.buildDomain(logN: logN)

    // Wire a[0] == wire b[1] (both hold value 42)
    var witness = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: n), count: numWires)
    let sharedVal = frFromInt(42)
    witness[0][0] = sharedVal
    witness[1][1] = sharedVal
    for j in 0..<numWires {
        for i in 0..<n {
            if witness[j][i] == Fr.zero {
                witness[j][i] = frFromInt(UInt64(100 + j * n + i))
            }
        }
    }

    let k1 = frFromInt(2)
    let k2 = frFromInt(3)
    let copies = [PlonkCopyConstraint(srcWire: 0, srcRow: 0, dstWire: 1, dstRow: 1)]
    let sigma = buildPermutationFromCopyConstraints(
        copies: copies, numWires: numWires, domainSize: n,
        domain: domain, cosetGenerators: [k1, k2]
    )

    let beta = frFromInt(7)
    let gamma = frFromInt(13)

    let config = PlonkGrandProductConfig(numWires: numWires, domainSize: n, useGPU: false)
    let result = engine.computeGrandProduct(
        witness: witness, sigmaPolys: sigma,
        beta: beta, gamma: gamma, config: config
    )

    expect(result.zPoly[0] == Fr.one, "Copy constraint: z[0] = 1")

    // Verify wrap-around: full product of ratios = 1
    expect(
        engine.checkWrapAround(z: result.zPoly, numerators: result.numerators, denominators: result.denominators),
        "Copy constraint: grand product wraps to 1"
    )
}

private func testMultipleCopyConstraints() {
    let engine = GPUPlonkGrandProductEngine()

    let logN = 4
    let n = 1 << logN  // 16
    let numWires = 3

    let domain = engine.buildDomain(logN: logN)

    // Multiple copy constraints: a[0]==b[2], a[3]==c[5], b[1]==c[4]
    var witness = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: n), count: numWires)
    let v1 = frFromInt(42)
    let v2 = frFromInt(99)
    let v3 = frFromInt(77)
    witness[0][0] = v1; witness[1][2] = v1
    witness[0][3] = v2; witness[2][5] = v2
    witness[1][1] = v3; witness[2][4] = v3

    for j in 0..<numWires {
        for i in 0..<n {
            if witness[j][i] == Fr.zero {
                witness[j][i] = frFromInt(UInt64(200 + j * n + i))
            }
        }
    }

    let k1 = frFromInt(2)
    let k2 = frFromInt(3)
    let copies = [
        PlonkCopyConstraint(srcWire: 0, srcRow: 0, dstWire: 1, dstRow: 2),
        PlonkCopyConstraint(srcWire: 0, srcRow: 3, dstWire: 2, dstRow: 5),
        PlonkCopyConstraint(srcWire: 1, srcRow: 1, dstWire: 2, dstRow: 4),
    ]
    let sigma = buildPermutationFromCopyConstraints(
        copies: copies, numWires: numWires, domainSize: n,
        domain: domain, cosetGenerators: [k1, k2]
    )

    let beta = frFromInt(11)
    let gamma = frFromInt(17)

    let config = PlonkGrandProductConfig(numWires: numWires, domainSize: n, useGPU: false)
    let result = engine.computeGrandProduct(
        witness: witness, sigmaPolys: sigma,
        beta: beta, gamma: gamma, config: config
    )

    expect(result.zPoly[0] == Fr.one, "Multi-copy: z[0] = 1")
    expect(
        engine.checkWrapAround(z: result.zPoly, numerators: result.numerators, denominators: result.denominators),
        "Multi-copy: grand product wraps to 1"
    )
}

// MARK: - Transition Validation Tests

private func testTransitionValidIdentity() {
    let engine = GPUPlonkGrandProductEngine()

    let logN = 3
    let n = 1 << logN
    let numWires = 3

    let domain = engine.buildDomain(logN: logN)
    let sigma = engine.buildIdentitySigma(domain: domain, numWires: numWires)

    let witness: [[Fr]] = (0..<numWires).map { _ in
        (0..<n).map { _ in frFromInt(UInt64.random(in: 1...500)) }
    }

    let beta = frFromInt(7)
    let gamma = frFromInt(13)

    let config = PlonkGrandProductConfig(numWires: numWires, domainSize: n, useGPU: false)
    let result = engine.computeGrandProduct(
        witness: witness, sigmaPolys: sigma,
        beta: beta, gamma: gamma, config: config
    )

    let check = engine.validateTransitions(
        z: result.zPoly, witness: witness, sigmaPolys: sigma,
        beta: beta, gamma: gamma
    )

    expect(check.valid, "Transition valid for identity permutation")
    expectEqual(check.failingRow, -1, "No failing row for identity permutation")
}

private func testTransitionValidCopyConstraint() {
    let engine = GPUPlonkGrandProductEngine()

    let logN = 3
    let n = 1 << logN
    let numWires = 3

    let domain = engine.buildDomain(logN: logN)

    var witness = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: n), count: numWires)
    let sharedVal = frFromInt(42)
    witness[0][0] = sharedVal
    witness[1][1] = sharedVal
    for j in 0..<numWires {
        for i in 0..<n {
            if witness[j][i] == Fr.zero {
                witness[j][i] = frFromInt(UInt64(100 + j * n + i))
            }
        }
    }

    let k1 = frFromInt(2)
    let k2 = frFromInt(3)
    let copies = [PlonkCopyConstraint(srcWire: 0, srcRow: 0, dstWire: 1, dstRow: 1)]
    let sigma = buildPermutationFromCopyConstraints(
        copies: copies, numWires: numWires, domainSize: n,
        domain: domain, cosetGenerators: [k1, k2]
    )

    let beta = frFromInt(7)
    let gamma = frFromInt(13)

    let config = PlonkGrandProductConfig(numWires: numWires, domainSize: n, useGPU: false)
    let result = engine.computeGrandProduct(
        witness: witness, sigmaPolys: sigma,
        beta: beta, gamma: gamma, config: config
    )

    let check = engine.validateTransitions(
        z: result.zPoly, witness: witness, sigmaPolys: sigma,
        beta: beta, gamma: gamma
    )

    expect(check.valid, "Transition valid for copy constraint")
}

private func testTransitionInvalidTampered() {
    let engine = GPUPlonkGrandProductEngine()

    let logN = 3
    let n = 1 << logN
    let numWires = 3

    let domain = engine.buildDomain(logN: logN)
    let sigma = engine.buildIdentitySigma(domain: domain, numWires: numWires)

    let witness: [[Fr]] = (0..<numWires).map { _ in
        (0..<n).map { _ in frFromInt(UInt64.random(in: 1...500)) }
    }

    let beta = frFromInt(7)
    let gamma = frFromInt(13)

    let config = PlonkGrandProductConfig(numWires: numWires, domainSize: n, useGPU: false)
    let result = engine.computeGrandProduct(
        witness: witness, sigmaPolys: sigma,
        beta: beta, gamma: gamma, config: config
    )

    // Tamper with z
    var tampered = result.zPoly
    tampered[3] = frFromInt(999)

    let check = engine.validateTransitions(
        z: tampered, witness: witness, sigmaPolys: sigma,
        beta: beta, gamma: gamma
    )

    expect(!check.valid, "Transition invalid for tampered z")
    expect(check.failingRow >= 0, "Failing row identified for tampered z")
    expect(!check.failureReason.isEmpty, "Failure reason provided for tampered z")
}

private func testTransitionInvalidBoundary() {
    let engine = GPUPlonkGrandProductEngine()

    // z[0] != 1 should fail boundary
    var z = [Fr](repeating: Fr.one, count: 8)
    z[0] = frFromInt(2)  // Bad boundary

    let witness: [[Fr]] = (0..<3).map { _ in
        (0..<8).map { _ in frFromInt(1) }
    }
    let sigma = engine.buildIdentitySigma(
        domain: engine.buildDomain(logN: 3), numWires: 3
    )

    let check = engine.validateTransitions(
        z: z, witness: witness, sigmaPolys: sigma,
        beta: frFromInt(1), gamma: frFromInt(1)
    )

    expect(!check.valid, "Transition invalid when z[0] != 1")
    expect(check.failureReason.contains("Boundary"), "Failure reason mentions boundary")
}

// MARK: - Split Permutation Tests

private func testSplitPermutation6Wires() {
    let engine = GPUPlonkGrandProductEngine()

    let logN = 3
    let n = 1 << logN
    let numWires = 6

    let domain = engine.buildDomain(logN: logN)
    let sigma = engine.buildIdentitySigma(domain: domain, numWires: numWires)

    let witness: [[Fr]] = (0..<numWires).map { _ in
        (0..<n).map { _ in frFromInt(UInt64.random(in: 1...500)) }
    }

    let beta = frFromInt(7)
    let gamma = frFromInt(13)

    // Split into groups of 2
    let config = PlonkGrandProductConfig(
        numWires: numWires, domainSize: n, splitThreshold: 2, useGPU: false
    )
    let result = engine.computeGrandProduct(
        witness: witness, sigmaPolys: sigma,
        beta: beta, gamma: gamma, config: config
    )

    expect(result.zPoly[0] == Fr.one, "Split 6-wire: z[0] = 1")
    expectEqual(result.numPartitions, 3, "Split 6-wire: 3 partitions")

    // Identity permutation should still give all ones
    var allOnes = true
    for i in 0..<n {
        if result.zPoly[i] != Fr.one { allOnes = false; break }
    }
    expect(allOnes, "Split 6-wire identity: z is all ones")
}

private func testSplitPermutationMatchesNonSplit() {
    let engine = GPUPlonkGrandProductEngine()

    let logN = 3
    let n = 1 << logN
    let numWires = 4

    let domain = engine.buildDomain(logN: logN)
    let sigma = engine.buildIdentitySigma(domain: domain, numWires: numWires)

    let witness: [[Fr]] = (0..<numWires).map { j in
        (0..<n).map { i in frFromInt(UInt64(j * n + i + 1)) }
    }

    let beta = frFromInt(5)
    let gamma = frFromInt(11)

    // Non-split
    let configNoSplit = PlonkGrandProductConfig(
        numWires: numWires, domainSize: n, splitThreshold: 0, useGPU: false
    )
    let resultNoSplit = engine.computeGrandProduct(
        witness: witness, sigmaPolys: sigma,
        beta: beta, gamma: gamma, config: configNoSplit
    )

    // Split into groups of 2
    let configSplit = PlonkGrandProductConfig(
        numWires: numWires, domainSize: n, splitThreshold: 2, useGPU: false
    )
    let resultSplit = engine.computeGrandProduct(
        witness: witness, sigmaPolys: sigma,
        beta: beta, gamma: gamma, config: configSplit
    )

    // Both should produce all-ones for identity permutation
    var match = true
    for i in 0..<n {
        if resultNoSplit.zPoly[i] != resultSplit.zPoly[i] { match = false; break }
    }
    expect(match, "Split matches non-split for identity permutation")
}

// MARK: - Coset Evaluation Tests

private func testCosetEvalIdentity() {
    let engine = GPUPlonkGrandProductEngine()

    let logN = 3
    let n = 1 << logN
    let numWires = 3

    let domain = engine.buildDomain(logN: logN)
    let sigma = engine.buildIdentitySigma(domain: domain, numWires: numWires)

    let witness: [[Fr]] = (0..<numWires).map { _ in
        (0..<n).map { _ in frFromInt(UInt64.random(in: 1...500)) }
    }

    let beta = frFromInt(7)
    let gamma = frFromInt(13)

    let config = PlonkGrandProductConfig(numWires: numWires, domainSize: n, useGPU: false)
    let result = engine.computeGrandProduct(
        witness: witness, sigmaPolys: sigma,
        beta: beta, gamma: gamma, config: config
    )

    let cosetGen = frFromInt(7)
    let cosetResult = engine.evaluateOnCoset(
        z: result.zPoly, witness: witness, sigmaPolys: sigma,
        beta: beta, gamma: gamma, cosetGenerator: cosetGen
    )

    expectEqual(cosetResult.zCosetEvals.count, n, "Coset eval: correct length")
    expectEqual(cosetResult.zShiftedCosetEvals.count, n, "Coset shifted: correct length")
    expectEqual(cosetResult.numCosetEvals.count, n, "Coset num: correct length")
    expectEqual(cosetResult.denCosetEvals.count, n, "Coset den: correct length")

    // For identity permutation, z is all ones. z on coset should also be all ones
    // (constant polynomial 1 evaluates to 1 everywhere).
    // Since z = [1, 1, ..., 1], barycentric interpolation should give 1 at all coset points.
    var allOnesOnCoset = true
    for i in 0..<n {
        if cosetResult.zCosetEvals[i] != Fr.one { allOnesOnCoset = false; break }
    }
    expect(allOnesOnCoset, "Coset eval: constant-1 polynomial evaluates to 1 on coset")
}

private func testCosetEvalShifted() {
    let engine = GPUPlonkGrandProductEngine()

    let logN = 3
    let n = 1 << logN
    let numWires = 3

    let domain = engine.buildDomain(logN: logN)
    let sigma = engine.buildIdentitySigma(domain: domain, numWires: numWires)

    let witness: [[Fr]] = (0..<numWires).map { _ in
        (0..<n).map { _ in frFromInt(UInt64.random(in: 1...500)) }
    }

    let beta = frFromInt(5)
    let gamma = frFromInt(11)

    let config = PlonkGrandProductConfig(numWires: numWires, domainSize: n, useGPU: false)
    let result = engine.computeGrandProduct(
        witness: witness, sigmaPolys: sigma,
        beta: beta, gamma: gamma, config: config
    )

    let cosetResult = engine.evaluateOnCoset(
        z: result.zPoly, witness: witness, sigmaPolys: sigma,
        beta: beta, gamma: gamma, cosetGenerator: frFromInt(5)
    )

    // Shifted should be a cyclic shift of coset evals
    var shiftMatch = true
    for i in 0..<n {
        if cosetResult.zShiftedCosetEvals[i] != cosetResult.zCosetEvals[(i + 1) % n] {
            shiftMatch = false; break
        }
    }
    expect(shiftMatch, "Coset shifted is cyclic shift of coset evals")
}

private func testCosetEvalEmpty() {
    let engine = GPUPlonkGrandProductEngine()

    let cosetResult = engine.evaluateOnCoset(
        z: [], witness: [], sigmaPolys: [],
        beta: Fr.one, gamma: Fr.one, cosetGenerator: frFromInt(7)
    )

    expectEqual(cosetResult.zCosetEvals.count, 0, "Empty coset eval: no z evals")
    expectEqual(cosetResult.zShiftedCosetEvals.count, 0, "Empty coset eval: no shifted evals")
}

// MARK: - GPU vs CPU Consistency Tests

private func testGPUCPUConsistency() {
    let engine = GPUPlonkGrandProductEngine()

    for logN in [3, 5] {
        let n = 1 << logN
        let numWires = 3

        let domain = engine.buildDomain(logN: logN)
        let sigma = engine.buildIdentitySigma(domain: domain, numWires: numWires)

        let witness: [[Fr]] = (0..<numWires).map { _ in
            (0..<n).map { _ in frFromInt(UInt64.random(in: 1...1000)) }
        }

        let beta = frFromInt(7)
        let gamma = frFromInt(13)

        let cpuConfig = PlonkGrandProductConfig(numWires: numWires, domainSize: n, useGPU: false)
        let cpuResult = engine.computeGrandProduct(
            witness: witness, sigmaPolys: sigma,
            beta: beta, gamma: gamma, config: cpuConfig
        )

        let gpuConfig = PlonkGrandProductConfig(numWires: numWires, domainSize: n, useGPU: true)
        let gpuResult = engine.computeGrandProduct(
            witness: witness, sigmaPolys: sigma,
            beta: beta, gamma: gamma, config: gpuConfig
        )

        var match = true
        for i in 0..<n {
            if cpuResult.zPoly[i] != gpuResult.zPoly[i] { match = false; break }
        }
        expect(match, "GPU and CPU match for n=\(n)")
    }
}

// MARK: - Edge Case Tests

private func testMinimalDomain2() {
    let engine = GPUPlonkGrandProductEngine()

    let n = 2
    let numWires = 3

    let domain = engine.buildDomain(logN: 1)
    let sigma = engine.buildIdentitySigma(domain: domain, numWires: numWires)

    let witness: [[Fr]] = (0..<numWires).map { _ in
        [frFromInt(10), frFromInt(20)]
    }

    let beta = frFromInt(1)
    let gamma = frFromInt(1)

    let config = PlonkGrandProductConfig(numWires: numWires, domainSize: n, useGPU: false)
    let result = engine.computeGrandProduct(
        witness: witness, sigmaPolys: sigma,
        beta: beta, gamma: gamma, config: config
    )

    expectEqual(result.zPoly.count, 2, "Domain size 2: z has 2 elements")
    expect(result.zPoly[0] == Fr.one, "Domain size 2: z[0] = 1")
    // Identity => z[1] should also be 1
    expect(result.zPoly[1] == Fr.one, "Domain size 2: z[1] = 1 for identity")
}

private func testSingleWire() {
    let engine = GPUPlonkGrandProductEngine()

    let logN = 3
    let n = 1 << logN
    let numWires = 1

    let domain = engine.buildDomain(logN: logN)
    let sigma = engine.buildIdentitySigma(domain: domain, numWires: numWires)

    let witness: [[Fr]] = [[Fr](repeating: Fr.zero, count: n).enumerated().map { frFromInt(UInt64($0.offset + 1)) }]

    let beta = frFromInt(3)
    let gamma = frFromInt(7)

    let config = PlonkGrandProductConfig(numWires: numWires, domainSize: n, useGPU: false)
    let result = engine.computeGrandProduct(
        witness: witness, sigmaPolys: sigma,
        beta: beta, gamma: gamma, config: config
    )

    expect(result.zPoly[0] == Fr.one, "Single wire: z[0] = 1")
    // Identity permutation: all ones
    var allOnes = true
    for i in 0..<n {
        if result.zPoly[i] != Fr.one { allOnes = false; break }
    }
    expect(allOnes, "Single wire identity: z is all ones")
}

// MARK: - Utility Function Tests

private func testBuildDomain() {
    let engine = GPUPlonkGrandProductEngine()

    let domain = engine.buildDomain(logN: 3)
    expectEqual(domain.count, 8, "buildDomain: length 8 for logN=3")
    expect(domain[0] == Fr.one, "buildDomain: domain[0] = 1")

    // omega^8 should be 1 (8th root of unity)
    let omega8 = frMul(domain[7], computeNthRootOfUnity(logN: 3))
    expect(omega8 == Fr.one, "buildDomain: omega^8 = 1")
}

private func testBuildIdentitySigma() {
    let engine = GPUPlonkGrandProductEngine()

    let domain = engine.buildDomain(logN: 3)
    let sigma = engine.buildIdentitySigma(domain: domain, numWires: 3)

    expectEqual(sigma.count, 3, "buildIdentitySigma: 3 wire columns")
    expectEqual(sigma[0].count, 8, "buildIdentitySigma: 8 rows per wire")

    // Column 0: sigma[0][i] = domain[i]
    for i in 0..<8 {
        expect(sigma[0][i] == domain[i], "sigma[0][\(i)] = domain[\(i)]")
    }
    // Column 1: sigma[1][i] = 2 * domain[i]
    let k1 = frFromInt(2)
    for i in 0..<8 {
        expect(sigma[1][i] == frMul(k1, domain[i]), "sigma[1][\(i)] = k1*domain[\(i)]")
    }
}

private func testLagrangeBasisFirst() {
    let engine = GPUPlonkGrandProductEngine()

    let l1 = engine.lagrangeBasisFirst(domainSize: 8)
    expectEqual(l1.count, 8, "L1: correct length")
    expect(l1[0] == Fr.one, "L1[0] = 1")
    for i in 1..<8 {
        expect(l1[i] == Fr.zero, "L1[\(i)] = 0")
    }
}

private func testFullRatioProduct() {
    let engine = GPUPlonkGrandProductEngine()

    // When num == den, product should be 1
    let vals = (0..<8).map { _ in frFromInt(UInt64.random(in: 1...1000)) }
    let prod = engine.fullRatioProduct(numerators: vals, denominators: vals)
    expect(prod == Fr.one, "fullRatioProduct: num == den => product = 1")
}

// MARK: - Round-Trip Tests

private func testFullRoundTrip() {
    let engine = GPUPlonkGrandProductEngine()

    let logN = 4
    let n = 1 << logN  // 16
    let numWires = 3

    let domain = engine.buildDomain(logN: logN)

    // Create witness with copy constraint: a[2] == c[5]
    var witness = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: n), count: numWires)
    let shared = frFromInt(77)
    witness[0][2] = shared
    witness[2][5] = shared
    for j in 0..<numWires {
        for i in 0..<n {
            if witness[j][i] == Fr.zero {
                witness[j][i] = frFromInt(UInt64(300 + j * n + i))
            }
        }
    }

    let k1 = frFromInt(2)
    let k2 = frFromInt(3)
    let copies = [PlonkCopyConstraint(srcWire: 0, srcRow: 2, dstWire: 2, dstRow: 5)]
    let sigma = buildPermutationFromCopyConstraints(
        copies: copies, numWires: numWires, domainSize: n,
        domain: domain, cosetGenerators: [k1, k2]
    )

    let beta = frFromInt(7)
    let gamma = frFromInt(13)

    // Step 1: Compute grand product
    let config = PlonkGrandProductConfig(numWires: numWires, domainSize: n, useGPU: false)
    let result = engine.computeGrandProduct(
        witness: witness, sigmaPolys: sigma,
        beta: beta, gamma: gamma, config: config
    )

    // Step 2: Check boundary
    expect(engine.checkBoundary(z: result.zPoly), "Round-trip: boundary check passes")

    // Step 3: Check wrap-around
    expect(
        engine.checkWrapAround(z: result.zPoly, numerators: result.numerators, denominators: result.denominators),
        "Round-trip: wrap-around check passes"
    )

    // Step 4: Validate all transitions
    let check = engine.validateTransitions(
        z: result.zPoly, witness: witness, sigmaPolys: sigma,
        beta: beta, gamma: gamma
    )
    expect(check.valid, "Round-trip: all transitions valid")

    // Step 5: Evaluate on coset
    let cosetResult = engine.evaluateOnCoset(
        z: result.zPoly, witness: witness, sigmaPolys: sigma,
        beta: beta, gamma: gamma, cosetGenerator: frFromInt(7)
    )
    expectEqual(cosetResult.zCosetEvals.count, n, "Round-trip: coset eval length correct")
    expectEqual(cosetResult.zShiftedCosetEvals.count, n, "Round-trip: shifted eval length correct")
}

private func testSimplifiedInterface() {
    let engine = GPUPlonkGrandProductEngine()

    let logN = 3
    let n = 1 << logN
    let numWires = 3

    let domain = engine.buildDomain(logN: logN)
    let sigma = engine.buildIdentitySigma(domain: domain, numWires: numWires)

    let witness: [[Fr]] = (0..<numWires).map { _ in
        (0..<n).map { _ in frFromInt(UInt64.random(in: 1...500)) }
    }

    let beta = frFromInt(7)
    let gamma = frFromInt(13)

    let z = engine.computeZPoly(
        witness: witness, sigmaPolys: sigma,
        beta: beta, gamma: gamma, domainSize: n
    )

    expectEqual(z.count, n, "Simplified: correct length")
    expect(z[0] == Fr.one, "Simplified: z[0] = 1")

    // Identity => all ones
    var allOnes = true
    for i in 0..<n {
        if z[i] != Fr.one { allOnes = false; break }
    }
    expect(allOnes, "Simplified: identity permutation gives all ones")
}

// MARK: - Config Tests

private func testConfigDefaults() {
    let config = PlonkGrandProductConfig(numWires: 3, domainSize: 16)
    expectEqual(config.numWires, 3, "Config: default numWires = 3")
    expectEqual(config.domainSize, 16, "Config: domainSize = 16")
    expectEqual(config.logN, 4, "Config: logN = 4")
    expectEqual(config.splitThreshold, 0, "Config: default splitThreshold = 0")
    expect(config.useGPU, "Config: default useGPU = true")
}

private func testConfigCustom() {
    let config = PlonkGrandProductConfig(
        numWires: 5, domainSize: 64, splitThreshold: 2, useGPU: false
    )
    expectEqual(config.numWires, 5, "Config custom: numWires = 5")
    expectEqual(config.domainSize, 64, "Config custom: domainSize = 64")
    expectEqual(config.logN, 6, "Config custom: logN = 6")
    expectEqual(config.splitThreshold, 2, "Config custom: splitThreshold = 2")
    expect(!config.useGPU, "Config custom: useGPU = false")
}
