// PlonkPermutationTests — Tests for the standalone permutation argument
//
// Tests cover:
//   1. Grand product computation (identity permutation -> all ones)
//   2. Grand product with copy constraints (Z[n-1] product should be 1)
//   3. CPU grand product engine (prefix product correctness)
//   4. GPU grand product engine (matches CPU for small/medium sizes)
//   5. Build permutation from copy constraints
//   6. Build permutation from variable-level assignments
//   7. Permutation verification (standalone check)
//   8. Multi-wire permutation (4 wires)

import zkMetal
import Foundation

func runPlonkPermutationTests() {
    suite("Plonk Permutation Argument")

    let logN = 3
    let n = 1 << logN  // 8

    // Build evaluation domain
    let omega = computeNthRootOfUnity(logN: logN)
    var domain = [Fr](repeating: Fr.zero, count: n)
    domain[0] = Fr.one
    for i in 1..<n {
        domain[i] = frMul(domain[i - 1], omega)
    }

    let k1 = frFromInt(2)
    let k2 = frFromInt(3)

    // ========== Test 1: Identity permutation -> Z = all ones ==========
    do {
        let permArg = PermutationArgument(numWires: 3, cosetGenerators: [k1, k2])

        // Identity permutation: sigma[j][i] = kj * omega^i
        var sigma = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: n), count: 3)
        for i in 0..<n {
            sigma[0][i] = domain[i]
            sigma[1][i] = frMul(k1, domain[i])
            sigma[2][i] = frMul(k2, domain[i])
        }

        // Random witness
        let witness: [[Fr]] = (0..<3).map { _ in
            (0..<n).map { _ in frFromInt(UInt64.random(in: 1...1000)) }
        }

        let beta = frFromInt(7)
        let gamma = frFromInt(13)

        let Z = permArg.computeGrandProduct(
            witness: witness, sigma: sigma,
            beta: beta, gamma: gamma, domain: domain
        )

        expect(Z[0] == Fr.one, "Z[0] = 1 for identity permutation")

        // With identity permutation, all ratios should be 1, so Z should be all ones
        var allOnes = true
        for i in 0..<n {
            if Z[i] != Fr.one {
                allOnes = false
                break
            }
        }
        expect(allOnes, "Identity permutation: Z is all ones")
    }

    // ========== Test 2: Copy constraint -> Z[n] product = 1 ==========
    do {
        let permArg = PermutationArgument(numWires: 3, cosetGenerators: [k1, k2])

        // Circuit: wire a[0] == wire b[1] (both hold value 42)
        var witness = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: n), count: 3)
        let sharedVal = frFromInt(42)
        witness[0][0] = sharedVal  // a[0] = 42
        witness[1][1] = sharedVal  // b[1] = 42
        // Fill rest with distinct values
        for j in 0..<3 {
            for i in 0..<n {
                if witness[j][i] == Fr.zero {
                    witness[j][i] = frFromInt(UInt64(100 + j * n + i))
                }
            }
        }

        // Build sigma with the copy constraint: position (0,0) <-> position (1,1)
        let copies = [PlonkCopyConstraint(srcWire: 0, srcRow: 0, dstWire: 1, dstRow: 1)]
        let sigma = buildPermutationFromCopyConstraints(
            copies: copies, numWires: 3, domainSize: n,
            domain: domain, cosetGenerators: [k1, k2]
        )

        let beta = frFromInt(7)
        let gamma = frFromInt(13)

        let Z = permArg.computeGrandProduct(
            witness: witness, sigma: sigma,
            beta: beta, gamma: gamma, domain: domain
        )

        expect(Z[0] == Fr.one, "Z[0] = 1 with copy constraint")

        // The last ratio times Z[n-1] should give 1 (grand product property)
        // Compute the final ratio at position n-1
        var finalNum = Fr.one
        var finalDen = Fr.one
        for j in 0..<3 {
            let kj: Fr = j == 0 ? Fr.one : (j == 1 ? k1 : k2)
            let idVal = frMul(kj, domain[n - 1])
            finalNum = frMul(finalNum, frAdd(frAdd(witness[j][n - 1], frMul(beta, idVal)), gamma))
            finalDen = frMul(finalDen, frAdd(frAdd(witness[j][n - 1], frMul(beta, sigma[j][n - 1])), gamma))
        }
        let finalZ = frMul(Z[n - 1], frMul(finalNum, frInverse(finalDen)))
        expect(finalZ == Fr.one, "Grand product Z[n] = 1 (copy constraint satisfied)")
    }

    // ========== Test 3: CPU grand product engine ==========
    do {
        let values: [Fr] = [frFromInt(2), frFromInt(3), frFromInt(5), frFromInt(7)]
        let result = PlonkGrandProductEngine.cpuGrandProduct(values: values)

        expect(result[0] == Fr.one, "CPU prefix product[0] = 1")
        expect(result[1] == frFromInt(2), "CPU prefix product[1] = 2")
        expect(result[2] == frFromInt(6), "CPU prefix product[2] = 6")
        expect(result[3] == frFromInt(30), "CPU prefix product[3] = 30")
    }

    // ========== Test 4: GPU grand product engine matches CPU ==========
    do {
        let engine = PlonkGrandProductEngine()

        // Small test
        let values: [Fr] = (0..<16).map { frFromInt(UInt64($0 + 1)) }
        let cpuResult = PlonkGrandProductEngine.cpuGrandProduct(values: values)
        let gpuResult = engine.gpuGrandProduct(values: values)

        var match = true
        for i in 0..<values.count {
            if cpuResult[i] != gpuResult[i] {
                match = false
                break
            }
        }
        expect(match, "GPU grand product matches CPU for small arrays")

        // Full product check
        let fullProd = PlonkGrandProductEngine.fullProduct(values)
        // 16! = 20922789888000
        let expected = frFromInt(20922789888000)
        expect(fullProd == expected, "Full product of 1..16 = 16!")
    }

    // ========== Test 5: buildPermutationFromCopyConstraints ==========
    do {
        // No copy constraints -> identity permutation
        let sigma = buildPermutationFromCopyConstraints(
            copies: [], numWires: 3, domainSize: n,
            domain: domain, cosetGenerators: [k1, k2]
        )

        expect(sigma.count == 3, "3 sigma polynomials")
        expect(sigma[0].count == n, "sigma1 has n elements")

        // Check identity: sigma[0][i] = omega^i, sigma[1][i] = k1*omega^i
        var identityCorrect = true
        for i in 0..<n {
            if sigma[0][i] != domain[i] { identityCorrect = false; break }
            if sigma[1][i] != frMul(k1, domain[i]) { identityCorrect = false; break }
            if sigma[2][i] != frMul(k2, domain[i]) { identityCorrect = false; break }
        }
        expect(identityCorrect, "No copies -> identity permutation")
    }

    // ========== Test 6: buildPermutationFromVariables ==========
    do {
        // Wire assignments: variable 0 appears at (row=0, col=0) and (row=1, col=1)
        // This creates an implicit copy constraint.
        let wireAssignments: [[Int]] = [
            [0, 1, 2],   // gate 0: a=var0, b=var1, c=var2
            [3, 0, 4],   // gate 1: a=var3, b=var0, c=var4 (var0 shared with gate 0 col a)
            [5, 6, 7],   // gate 2
            [8, 9, 10],  // gate 3
            [11, 12, 13], // gate 4
            [14, 15, 16], // gate 5
            [17, 18, 19], // gate 6
            [20, 21, 22], // gate 7
        ]

        let sigma = buildPermutationFromVariables(
            wireAssignments: wireAssignments,
            numWires: 3, domainSize: n,
            domain: domain, cosetGenerators: [k1, k2]
        )

        // sigma[0][0] should map to position (col=1, row=1) = k1 * omega^1
        let expected01 = frMul(k1, domain[1])
        expect(sigma[0][0] == expected01, "Variable sharing: sigma[0][0] -> (1,1)")

        // sigma[1][1] should map back to position (col=0, row=0) = omega^0 = 1
        expect(sigma[1][1] == domain[0], "Variable sharing: sigma[1][1] -> (0,0)")
    }

    // ========== Test 7: Permutation verification ==========
    do {
        let permArg = PermutationArgument(numWires: 3, cosetGenerators: [k1, k2])

        // Identity permutation (trivial case)
        var sigma = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: n), count: 3)
        for i in 0..<n {
            sigma[0][i] = domain[i]
            sigma[1][i] = frMul(k1, domain[i])
            sigma[2][i] = frMul(k2, domain[i])
        }

        let witness: [[Fr]] = (0..<3).map { j in
            (0..<n).map { i in frFromInt(UInt64(10 + j * n + i)) }
        }

        let beta = frFromInt(11)
        let gamma = frFromInt(17)

        let Z = permArg.computeGrandProduct(
            witness: witness, sigma: sigma,
            beta: beta, gamma: gamma, domain: domain
        )

        // For identity permutation, Z should be all 1s,
        // so verification at any point should pass
        // Evaluate at zeta = omega^0 (the first domain point)
        // L_1(omega^0) = 1, Z(omega^0) = 1
        let valid = permArg.verifyPermutation(
            zEval: Z[0],
            zOmegaEval: Z[1],
            witnessEvals: [witness[0][0], witness[1][0], witness[2][0]],
            sigmaEvals: [sigma[0][0], sigma[1][0], sigma[2][0]],
            beta: beta,
            gamma: gamma,
            zeta: domain[0],
            l1Zeta: Fr.one  // L_1(omega^0) = 1
        )
        expect(valid, "Identity permutation verification passes at omega^0")

        // Verification should fail with wrong Z value
        let invalid = permArg.verifyPermutation(
            zEval: frFromInt(999),  // wrong
            zOmegaEval: Z[1],
            witnessEvals: [witness[0][0], witness[1][0], witness[2][0]],
            sigmaEvals: [sigma[0][0], sigma[1][0], sigma[2][0]],
            beta: beta,
            gamma: gamma,
            zeta: domain[0],
            l1Zeta: Fr.one
        )
        expect(!invalid, "Wrong Z value: verification fails")
    }

    // ========== Test 8: Multi-wire permutation (4 wires) ==========
    do {
        let k3 = frFromInt(4)
        let permArg = PermutationArgument(numWires: 4, cosetGenerators: [k1, k2, k3])

        // Identity permutation for 4 wires
        var sigma = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: n), count: 4)
        for i in 0..<n {
            sigma[0][i] = domain[i]
            sigma[1][i] = frMul(k1, domain[i])
            sigma[2][i] = frMul(k2, domain[i])
            sigma[3][i] = frMul(k3, domain[i])
        }

        let witness: [[Fr]] = (0..<4).map { j in
            (0..<n).map { i in frFromInt(UInt64(5 + j * n + i)) }
        }

        let beta = frFromInt(19)
        let gamma = frFromInt(23)

        let Z = permArg.computeGrandProduct(
            witness: witness, sigma: sigma,
            beta: beta, gamma: gamma, domain: domain
        )

        // Identity permutation should give all ones
        var allOnes = true
        for i in 0..<n {
            if Z[i] != Fr.one { allOnes = false; break }
        }
        expect(allOnes, "4-wire identity permutation: Z is all ones")
    }

    // ========== Test 9: Quotient contribution evaluates to zero on domain ==========
    do {
        let permArg = PermutationArgument(numWires: 3, cosetGenerators: [k1, k2])

        // Identity permutation
        var sigma = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: n), count: 3)
        for i in 0..<n {
            sigma[0][i] = domain[i]
            sigma[1][i] = frMul(k1, domain[i])
            sigma[2][i] = frMul(k2, domain[i])
        }

        let witness: [[Fr]] = (0..<3).map { j in
            (0..<n).map { i in frFromInt(UInt64(10 + j * n + i)) }
        }

        let beta = frFromInt(7)
        let gamma = frFromInt(13)
        let alpha = frFromInt(29)

        let Z = permArg.computeGrandProduct(
            witness: witness, sigma: sigma,
            beta: beta, gamma: gamma, domain: domain
        )

        let quotientContrib = permArg.evaluatePermutationQuotient(
            Z: Z, witness: witness, sigma: sigma,
            beta: beta, gamma: gamma, alpha: alpha,
            domain: domain
        )

        // On the evaluation domain, the permutation constraint should be satisfied,
        // so the quotient contribution should be zero everywhere
        var allZero = true
        for i in 0..<n {
            if !quotientContrib[i].isZero {
                allZero = false
                break
            }
        }
        expect(allZero, "Permutation quotient is zero on domain (identity perm)")
    }
}
