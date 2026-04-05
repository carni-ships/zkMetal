// Halo2PermutationTests — Tests for the Halo2-style generalized permutation argument
//
// Tests cover:
//   1. Simple 2-cell equality constraint
//   2. 3-cell cycle (a = b = c)
//   3. Multi-column permutation (wire between column 0 and column 2)
//   4. Grand product Z starts at 1 and ends at 1 for valid permutation
//   5. Invalid permutation (mismatched values) -> Z doesn't close
//   6. Identity permutation produces all-ones Z
//   7. Verifier accepts valid permutation
//   8. Verifier rejects invalid permutation
//   9. Point-evaluation verification

import zkMetal
import Foundation

public func runHalo2PermutationTests() {
    suite("Halo2 Permutation Argument")

    let logN = 3
    let n = 1 << logN  // 8

    // Build evaluation domain
    let omega = computeNthRootOfUnity(logN: logN)
    var domain = [Fr](repeating: Fr.zero, count: n)
    domain[0] = Fr.one
    for i in 1..<n {
        domain[i] = frMul(domain[i - 1], omega)
    }

    // ========== Test 1: Simple 2-cell equality ==========
    do {
        var assembly = Halo2PermutationAssembly(numColumns: 3, domainSize: n)

        // Constrain cell (col=0, row=0) == cell (col=1, row=2)
        assembly.addEquality(a: (col: 0, row: 0), b: (col: 1, row: 2))

        let sigma = assembly.buildSigmaPolynomials(domain: domain)

        expect(sigma.count == 3, "2-cell: 3 sigma polynomials")

        // sigma[0][0] should point to (col=1, row=2) = cosetMult[1] * omega^2
        let expectedTarget = frMul(frFromInt(2), domain[2])
        expect(sigma[0][0] == expectedTarget, "2-cell: sigma[0][0] -> (1,2)")

        // sigma[1][2] should point back to (col=0, row=0) = cosetMult[0] * omega^0
        let expectedBack = frMul(frFromInt(1), domain[0])
        expect(sigma[1][2] == expectedBack, "2-cell: sigma[1][2] -> (0,0)")

        // All other positions should remain identity
        let id1_0 = frMul(frFromInt(2), domain[0])  // col=1, row=0 identity
        expect(sigma[1][0] == id1_0, "2-cell: sigma[1][0] unchanged (identity)")
    }

    // ========== Test 2: 3-cell cycle (a = b = c) ==========
    do {
        var assembly = Halo2PermutationAssembly(numColumns: 3, domainSize: n)

        // Chain: (0,0) == (1,1) and (1,1) == (2,3)
        // This creates a 3-element cycle: (0,0) -> (1,1) -> (2,3) -> (0,0)
        assembly.addEquality(a: (col: 0, row: 0), b: (col: 1, row: 1))
        assembly.addEquality(a: (col: 1, row: 1), b: (col: 2, row: 3))

        expect(assembly.equalityCount == 2, "3-cell: 2 equality constraints")

        let sigma = assembly.buildSigmaPolynomials(domain: domain)

        // The three positions should form a cycle.
        // The exact cycle order depends on union-find internal structure,
        // but we can verify the cycle property: following sigma three times
        // from any starting position returns to start.

        // Collect the three coset elements for the cycle members
        let pos00 = frMul(frFromInt(1), domain[0])  // col=0, row=0
        let pos11 = frMul(frFromInt(2), domain[1])  // col=1, row=1
        let pos23 = frMul(frFromInt(3), domain[3])  // col=2, row=3

        let cycleElements = Set([pos00, pos11, pos23])

        // sigma[0][0], sigma[1][1], sigma[2][3] should all be in the cycle set
        expect(cycleElements.contains(sigma[0][0]), "3-cell: sigma[0][0] in cycle")
        expect(cycleElements.contains(sigma[1][1]), "3-cell: sigma[1][1] in cycle")
        expect(cycleElements.contains(sigma[2][3]), "3-cell: sigma[2][3] in cycle")

        // Each mapped-to element should be different (it's a 3-cycle, not self-loops)
        let mappedSet = Set([sigma[0][0], sigma[1][1], sigma[2][3]])
        expect(mappedSet.count == 3, "3-cell: all three sigma targets are distinct")
    }

    // ========== Test 3: Multi-column permutation (wire between col 0 and col 2) ==========
    do {
        var assembly = Halo2PermutationAssembly(numColumns: 4, domainSize: n)

        // Wire col 0, row 1 to col 2, row 5
        assembly.addEquality(a: (col: 0, row: 1), b: (col: 2, row: 5))

        let sigma = assembly.buildSigmaPolynomials(domain: domain)

        expect(sigma.count == 4, "Multi-col: 4 sigma polynomials")

        // sigma[0][1] should point to col=2, row=5
        let target = frMul(frFromInt(3), domain[5])  // cosetMult[2] = 3
        expect(sigma[0][1] == target, "Multi-col: sigma[0][1] -> (2,5)")

        // sigma[2][5] should point back to col=0, row=1
        let back = frMul(frFromInt(1), domain[1])  // cosetMult[0] = 1
        expect(sigma[2][5] == back, "Multi-col: sigma[2][5] -> (0,1)")

        // Column 3 should be completely identity
        var col3identity = true
        for i in 0..<n {
            let expected = frMul(frFromInt(4), domain[i])  // cosetMult[3] = 4
            if sigma[3][i] != expected { col3identity = false; break }
        }
        expect(col3identity, "Multi-col: column 3 fully identity")
    }

    // ========== Test 4: Grand product Z starts at 1 and ends at 1 ==========
    do {
        var assembly = Halo2PermutationAssembly(numColumns: 3, domainSize: n)

        // Create a copy constraint: (0,0) == (1,2), both hold value 42
        assembly.addEquality(a: (col: 0, row: 0), b: (col: 1, row: 2))

        let sigma = assembly.buildSigmaPolynomials(domain: domain)

        // Build witness: constrained cells hold the same value
        var witness = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: n), count: 3)
        let sharedVal = frFromInt(42)
        witness[0][0] = sharedVal
        witness[1][2] = sharedVal

        // Fill remaining cells with distinct values
        for j in 0..<3 {
            for i in 0..<n {
                if witness[j][i] == Fr.zero {
                    witness[j][i] = frFromInt(UInt64(100 + j * n + i))
                }
            }
        }

        let beta = frFromInt(7)
        let gamma = frFromInt(13)

        let prover = Halo2PermutationProver(assembly: assembly)
        let zEvals = prover.computeGrandProduct(
            witness: witness, sigma: sigma,
            beta: beta, gamma: gamma, domain: domain
        )

        // Z[0] = 1
        expect(zEvals[0] == Fr.one, "Valid perm: Z[0] = 1")

        // Closing value should be 1
        let closingVal = prover.grandProductClosingValue(
            zEvals: zEvals, witness: witness, sigma: sigma,
            beta: beta, gamma: gamma, domain: domain
        )
        expect(closingVal == Fr.one, "Valid perm: Z closes at 1")
    }

    // ========== Test 5: Invalid permutation (mismatched values) -> Z doesn't close ==========
    do {
        var assembly = Halo2PermutationAssembly(numColumns: 3, domainSize: n)

        // Constrain (0,0) == (1,2), but assign DIFFERENT values
        assembly.addEquality(a: (col: 0, row: 0), b: (col: 1, row: 2))

        let sigma = assembly.buildSigmaPolynomials(domain: domain)

        var witness = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: n), count: 3)
        witness[0][0] = frFromInt(42)   // cell (0,0) = 42
        witness[1][2] = frFromInt(99)   // cell (1,2) = 99  -- MISMATCH!

        // Fill remaining with distinct values
        for j in 0..<3 {
            for i in 0..<n {
                if witness[j][i] == Fr.zero {
                    witness[j][i] = frFromInt(UInt64(200 + j * n + i))
                }
            }
        }

        let beta = frFromInt(7)
        let gamma = frFromInt(13)

        let prover = Halo2PermutationProver(assembly: assembly)
        let zEvals = prover.computeGrandProduct(
            witness: witness, sigma: sigma,
            beta: beta, gamma: gamma, domain: domain
        )

        // Z[0] still starts at 1
        expect(zEvals[0] == Fr.one, "Invalid perm: Z[0] = 1 (always)")

        // But closing value should NOT be 1
        let closingVal = prover.grandProductClosingValue(
            zEvals: zEvals, witness: witness, sigma: sigma,
            beta: beta, gamma: gamma, domain: domain
        )
        expect(closingVal != Fr.one, "Invalid perm: Z does NOT close at 1")
    }

    // ========== Test 6: Identity permutation -> Z is all ones ==========
    do {
        let assembly = Halo2PermutationAssembly(numColumns: 3, domainSize: n)
        // No equalities added -> identity permutation

        let sigma = assembly.buildSigmaPolynomials(domain: domain)

        let witness: [[Fr]] = (0..<3).map { _ in
            (0..<n).map { _ in frFromInt(UInt64.random(in: 1...1000)) }
        }

        let beta = frFromInt(11)
        let gamma = frFromInt(17)

        let prover = Halo2PermutationProver(assembly: assembly)
        let zEvals = prover.computeGrandProduct(
            witness: witness, sigma: sigma,
            beta: beta, gamma: gamma, domain: domain
        )

        var allOnes = true
        for i in 0..<n {
            if zEvals[i] != Fr.one { allOnes = false; break }
        }
        expect(allOnes, "Identity perm: Z is all ones")
    }

    // ========== Test 7: Verifier accepts valid permutation ==========
    do {
        var assembly = Halo2PermutationAssembly(numColumns: 3, domainSize: n)
        assembly.addEquality(a: (col: 0, row: 1), b: (col: 2, row: 4))

        let sigma = assembly.buildSigmaPolynomials(domain: domain)

        var witness = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: n), count: 3)
        let sharedVal = frFromInt(77)
        witness[0][1] = sharedVal
        witness[2][4] = sharedVal
        for j in 0..<3 {
            for i in 0..<n {
                if witness[j][i] == Fr.zero {
                    witness[j][i] = frFromInt(UInt64(300 + j * n + i))
                }
            }
        }

        let beta = frFromInt(19)
        let gamma = frFromInt(23)

        let prover = Halo2PermutationProver(assembly: assembly)
        let zEvals = prover.computeGrandProduct(
            witness: witness, sigma: sigma,
            beta: beta, gamma: gamma, domain: domain
        )

        let verifier = Halo2PermutationVerifier(assembly: assembly)
        let valid = verifier.verify(
            zEvals: zEvals, witness: witness, sigma: sigma,
            beta: beta, gamma: gamma, domain: domain
        )
        expect(valid, "Verifier accepts valid permutation")
    }

    // ========== Test 8: Verifier rejects invalid permutation ==========
    do {
        var assembly = Halo2PermutationAssembly(numColumns: 3, domainSize: n)
        assembly.addEquality(a: (col: 0, row: 1), b: (col: 2, row: 4))

        let sigma = assembly.buildSigmaPolynomials(domain: domain)

        var witness = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: n), count: 3)
        witness[0][1] = frFromInt(77)   // cell (0,1) = 77
        witness[2][4] = frFromInt(88)   // cell (2,4) = 88  -- MISMATCH
        for j in 0..<3 {
            for i in 0..<n {
                if witness[j][i] == Fr.zero {
                    witness[j][i] = frFromInt(UInt64(400 + j * n + i))
                }
            }
        }

        let beta = frFromInt(19)
        let gamma = frFromInt(23)

        let prover = Halo2PermutationProver(assembly: assembly)
        let zEvals = prover.computeGrandProduct(
            witness: witness, sigma: sigma,
            beta: beta, gamma: gamma, domain: domain
        )

        let verifier = Halo2PermutationVerifier(assembly: assembly)
        let invalid = verifier.verify(
            zEvals: zEvals, witness: witness, sigma: sigma,
            beta: beta, gamma: gamma, domain: domain
        )
        expect(!invalid, "Verifier rejects invalid permutation (mismatched values)")
    }

    // ========== Test 9: Point-evaluation verification ==========
    do {
        let assembly = Halo2PermutationAssembly(numColumns: 3, domainSize: n)
        let sigma = assembly.buildSigmaPolynomials(domain: domain)

        let witness: [[Fr]] = (0..<3).map { j in
            (0..<n).map { i in frFromInt(UInt64(10 + j * n + i)) }
        }

        let beta = frFromInt(11)
        let gamma = frFromInt(17)

        let prover = Halo2PermutationProver(assembly: assembly)
        let zEvals = prover.computeGrandProduct(
            witness: witness, sigma: sigma,
            beta: beta, gamma: gamma, domain: domain
        )

        let verifier = Halo2PermutationVerifier(assembly: assembly)

        // Verify at omega^0 where L_1 = 1 and Z should be 1
        let valid = verifier.verifyAtPoint(
            zEval: zEvals[0],
            zOmegaEval: zEvals[1],
            witnessEvals: [witness[0][0], witness[1][0], witness[2][0]],
            sigmaEvals: [sigma[0][0], sigma[1][0], sigma[2][0]],
            beta: beta,
            gamma: gamma,
            zeta: domain[0],
            l1Zeta: Fr.one
        )
        expect(valid, "Point verification passes at omega^0 (identity perm)")

        // Wrong Z value should fail
        let badZ = verifier.verifyAtPoint(
            zEval: frFromInt(999),
            zOmegaEval: zEvals[1],
            witnessEvals: [witness[0][0], witness[1][0], witness[2][0]],
            sigmaEvals: [sigma[0][0], sigma[1][0], sigma[2][0]],
            beta: beta,
            gamma: gamma,
            zeta: domain[0],
            l1Zeta: Fr.one
        )
        expect(!badZ, "Point verification fails with wrong Z value")
    }

    // ========== Test 10: Column types and Cell struct ==========
    do {
        let advCol = Column.advice(0)
        let fixCol = Column.fixed(1)
        let instCol = Column.instance(2)

        expect(advCol.type == .advice, "Column.advice has advice type")
        expect(fixCol.type == .fixed, "Column.fixed has fixed type")
        expect(instCol.type == .instance, "Column.instance has instance type")
        expect(advCol.index == 0, "Column.advice(0) has index 0")

        let cell = Cell(column: advCol, row: 5)
        expect(cell.column == advCol, "Cell column matches")
        expect(cell.row == 5, "Cell row matches")

        let cell2 = Cell(col: 2, row: 3)
        expect(cell2.column.type == .advice, "Cell(col:row:) defaults to advice")
        expect(cell2.column.index == 2, "Cell(col:row:) index correct")

        // PermutationColumns
        var permCols = PermutationColumns()
        let idx0 = permCols.add(advCol)
        let idx1 = permCols.add(fixCol)
        let idx0dup = permCols.add(advCol)  // duplicate, should return same index

        expect(idx0 == 0, "First column gets index 0")
        expect(idx1 == 1, "Second column gets index 1")
        expect(idx0dup == 0, "Duplicate add returns existing index")
        expect(permCols.count == 2, "Two unique columns")

        let flatIdx = permCols.flatIndex(cell: cell, domainSize: n)
        expect(flatIdx == 0 * n + 5, "Flat index for advice[0], row 5")
    }
}
