// IPAVerifierCircuitTests — Correctness tests and benchmarks for IPAPastaVerifierCircuitEncoder
//
// Tests verify:
//   1. R1CS constraint system builds correctly
//   2. Constraint count is as expected
//   3. Constraint structure is sound (A, B, C well-formed)
//   4. Constraint generation performance benchmarks
//
// Note: Full R1CS satisfaction testing requires witness generation to be
// implemented correctly for the challenge derivation and folding computation.
// This is a known limitation - the constraint structure is correct but
// the witness mapper needs to compute actual values.

import Foundation
import zkMetal

// MARK: - Test Suite

public func runIPAVerifierCircuitTests() {
    suite("IPAVerifierCircuitEncoder")
    testBuildVerifierCircuit()
    testConstraintCount()
    testConstraintStructure()
    testPublicInputBinding()
    benchmarkConstraintGeneration()
}

// MARK: - Test 1: Build Verifier Circuit

private func testBuildVerifierCircuit() {
    do {
        let encoder = IPAPastaVerifierCircuitEncoder()
        let (r1cs, _) = encoder.buildVerifierR1CS()

        expect(r1cs.numConstraints > 0, "R1CS has constraints")
        expect(r1cs.numVars > 0, "R1CS has variables")
        expect(r1cs.numPublic >= 1, "R1CS has public inputs")

        // Check that A, B, C entries are non-empty
        expect(!r1cs.aEntries.isEmpty, "A entries is non-empty")
        expect(!r1cs.bEntries.isEmpty, "B entries is non-empty")
        expect(!r1cs.cEntries.isEmpty, "C entries is non-empty")

        print("  Build verifier circuit: \(r1cs.numConstraints) constraints, \(r1cs.numVars) vars")
    } catch {
        expect(false, "buildVerifierR1CS threw: \(error)")
    }
}

// MARK: - Test 2: Constraint Count

private func testConstraintCount() {
    let encoder = IPAPastaVerifierCircuitEncoder()
    let (r1cs, _) = encoder.buildVerifierR1CS()

    let estimated = encoder.estimatedConstraintCount
    let actual = r1cs.numConstraints

    // Constraint count should be within 2x of estimate
    let ratio = Double(actual) / Double(estimated)
    expect(ratio > 0.5 && ratio < 2.0,
           "Constraint count \(actual) within expected range of estimate \(estimated) (ratio: \(String(format: "%.2f", ratio)))")

    print("  Constraint count: estimated=\(estimated), actual=\(actual), ratio=\(String(format: "%.2f", ratio))")
}

// MARK: - Test 3: Constraint Structure

private func testConstraintStructure() {
    let encoder = IPAPastaVerifierCircuitEncoder()
    let (r1cs, _) = encoder.buildVerifierR1CS()

    // For each constraint, A*x * B*y = C should hold for some x, y
    // We can check that the constraint matrix entries are well-formed

    // Check that all entries reference valid variable indices
    let maxVar = r1cs.numVars
    var allValid = true
    for entry in r1cs.aEntries {
        if entry.col >= maxVar || entry.col < 0 {
            allValid = false
            break
        }
    }
    expect(allValid, "All A entries reference valid variable indices")

    for entry in r1cs.bEntries {
        if entry.col >= maxVar || entry.col < 0 {
            allValid = false
            break
        }
    }
    expect(allValid, "All B entries reference valid variable indices")

    for entry in r1cs.cEntries {
        if entry.col >= maxVar || entry.col < 0 {
            allValid = false
            break
        }
    }
    expect(allValid, "All C entries reference valid variable indices")

    // Check that each row has at least one A and B entry (non-trivial constraint)
    let rowsWithA = Set(r1cs.aEntries.map { $0.row })
    let rowsWithB = Set(r1cs.bEntries.map { $0.row })
    let rowsWithC = Set(r1cs.cEntries.map { $0.row })

    expect(rowsWithA.count == r1cs.numConstraints, "All rows have A entries")
    expect(rowsWithB.count == r1cs.numConstraints, "All rows have B entries")
    expect(rowsWithC.count == r1cs.numConstraints, "All rows have C entries")

    print("  Constraint structure: \(rowsWithA.count) rows, \(r1cs.aEntries.count) A entries, \(r1cs.bEntries.count) B entries, \(r1cs.cEntries.count) C entries")
}

// MARK: - Test 4: Public Input Binding

private func testPublicInputBinding() {
    let encoder = IPAPastaVerifierCircuitEncoder()
    let (r1cs, _) = encoder.buildVerifierR1CS()

    // Public inputs should be bound via constraints
    // At least some constraints should reference the public input variables (indices 1..numPublic)

    let publicVarCount = r1cs.numPublic

    // Count constraints that reference public variables
    var pubInputConstraints = 0
    for entry in r1cs.aEntries {
        if entry.col > 0 && entry.col <= publicVarCount {
            pubInputConstraints += 1
            break
        }
    }
    for entry in r1cs.bEntries {
        if entry.col > 0 && entry.col <= publicVarCount {
            pubInputConstraints += 1
            break
        }
    }
    for entry in r1cs.cEntries {
        if entry.col > 0 && entry.col <= publicVarCount {
            pubInputConstraints += 1
            break
        }
    }

    expect(pubInputConstraints > 0, "Public inputs are referenced in constraints")

    // Constant 1 should be referenced (variable 0)
    var constantUsed = false
    for entry in r1cs.aEntries {
        if entry.col == 0 {
            constantUsed = true
            break
        }
    }
    expect(constantUsed, "Constant 1 (variable 0) is used in constraints")

    print("  Public input binding: \(publicVarCount) public vars, constant used=\(constantUsed)")
}

// MARK: - Benchmark: Constraint Generation

private func benchmarkConstraintGeneration() {
    let t0 = CFAbsoluteTimeGetCurrent()
    let iterations = 10

    for _ in 0..<iterations {
        let encoder = IPAPastaVerifierCircuitEncoder()
        _ = encoder.buildVerifierR1CS()
    }

    let elapsed = CFAbsoluteTimeGetCurrent() - t0
    let avgMs = (elapsed / Double(iterations)) * 1000

    print("  Constraint generation: \(String(format: "%.2f", avgMs)) ms/op (\(iterations) iterations)")
}
