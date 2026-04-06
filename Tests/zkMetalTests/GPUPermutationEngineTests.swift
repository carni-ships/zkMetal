// GPUPermutationEngineTests — Tests for GPU-accelerated permutation argument engine
//
// Tests cover:
//   1. Identity permutation: Z is all ones
//   2. Copy constraint: grand product wraps to 1
//   3. GPU matches CPU (PermutationArgument) for random witness
//   4. Verification passes for valid Z, fails for tampered Z
//   5. Multi-wire (4 wires) GPU permutation
//   6. Larger domain (2^10) GPU correctness

import zkMetal
import Foundation

public func runGPUPermutationEngineTests() {
    suite("GPU Permutation Engine")

    // ========== Test 1: Identity permutation -> Z is all ones ==========
    do {
        let engine = try! GPUPermutationEngine()

        let logN = 3
        let n = 1 << logN  // 8
        let numWires = 3

        // Identity permutation: sigma[j][i] = cosetMul(j) * omega^i
        let omega = computeNthRootOfUnity(logN: logN)
        var domain = [Fr](repeating: Fr.zero, count: n)
        domain[0] = Fr.one
        for i in 1..<n { domain[i] = frMul(domain[i - 1], omega) }

        let k1 = frFromInt(2)
        let k2 = frFromInt(3)

        var sigma = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: n), count: numWires)
        for i in 0..<n {
            sigma[0][i] = domain[i]
            sigma[1][i] = frMul(k1, domain[i])
            sigma[2][i] = frMul(k2, domain[i])
        }

        // Random witness
        let witness: [[Fr]] = (0..<numWires).map { _ in
            (0..<n).map { _ in frFromInt(UInt64.random(in: 1...1000)) }
        }

        let beta = frFromInt(7)
        let gamma = frFromInt(13)

        let Z = engine.computePermutationPoly(
            witness: witness, sigmaPolys: sigma,
            beta: beta, gamma: gamma
        )

        expect(Z.count == n, "GPU perm: output length matches")
        expect(Z[0] == Fr.one, "GPU perm: Z[0] = 1 for identity permutation")

        var allOnes = true
        for i in 0..<n {
            if Z[i] != Fr.one { allOnes = false; break }
        }
        expect(allOnes, "GPU perm: identity permutation -> Z is all ones")
    }

    // ========== Test 2: Copy constraint -> grand product wraps to 1 ==========
    do {
        let engine = try! GPUPermutationEngine()

        let logN = 3
        let n = 1 << logN
        let numWires = 3

        let omega = computeNthRootOfUnity(logN: logN)
        var domain = [Fr](repeating: Fr.zero, count: n)
        domain[0] = Fr.one
        for i in 1..<n { domain[i] = frMul(domain[i - 1], omega) }

        let k1 = frFromInt(2)
        let k2 = frFromInt(3)

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

        let copies = [PlonkCopyConstraint(srcWire: 0, srcRow: 0, dstWire: 1, dstRow: 1)]
        let sigma = buildPermutationFromCopyConstraints(
            copies: copies, numWires: numWires, domainSize: n,
            domain: domain, cosetGenerators: [k1, k2]
        )

        let beta = frFromInt(7)
        let gamma = frFromInt(13)

        let Z = engine.computePermutationPoly(
            witness: witness, sigmaPolys: sigma,
            beta: beta, gamma: gamma
        )

        expect(Z[0] == Fr.one, "GPU perm: Z[0] = 1 with copy constraint")

        // Verify the grand product wraps: Z[n-1] * ratio[n-1] = 1
        var finalNum = Fr.one
        var finalDen = Fr.one
        for j in 0..<numWires {
            let kj: Fr = j == 0 ? Fr.one : (j == 1 ? k1 : k2)
            let idVal = frMul(kj, domain[n - 1])
            finalNum = frMul(finalNum, frAdd(frAdd(witness[j][n - 1], frMul(beta, idVal)), gamma))
            finalDen = frMul(finalDen, frAdd(frAdd(witness[j][n - 1], frMul(beta, sigma[j][n - 1])), gamma))
        }
        let finalZ = frMul(Z[n - 1], frMul(finalNum, frInverse(finalDen)))
        expect(finalZ == Fr.one, "GPU perm: grand product wraps to 1 (copy constraint satisfied)")
    }

    // ========== Test 3: GPU matches CPU PermutationArgument ==========
    do {
        let engine = try! GPUPermutationEngine()

        let logN = 4
        let n = 1 << logN  // 16
        let numWires = 3

        let omega = computeNthRootOfUnity(logN: logN)
        var domain = [Fr](repeating: Fr.zero, count: n)
        domain[0] = Fr.one
        for i in 1..<n { domain[i] = frMul(domain[i - 1], omega) }

        let k1 = frFromInt(2)
        let k2 = frFromInt(3)

        let witness: [[Fr]] = (0..<numWires).map { _ in
            (0..<n).map { _ in frFromInt(UInt64.random(in: 1...10000)) }
        }

        // Create a few copy constraints
        var w = witness
        let shared1 = frFromInt(777)
        w[0][2] = shared1
        w[1][5] = shared1
        let shared2 = frFromInt(888)
        w[0][7] = shared2
        w[2][3] = shared2

        let copies = [
            PlonkCopyConstraint(srcWire: 0, srcRow: 2, dstWire: 1, dstRow: 5),
            PlonkCopyConstraint(srcWire: 0, srcRow: 7, dstWire: 2, dstRow: 3),
        ]
        let sigma = buildPermutationFromCopyConstraints(
            copies: copies, numWires: numWires, domainSize: n,
            domain: domain, cosetGenerators: [k1, k2]
        )

        let beta = frFromInt(31)
        let gamma = frFromInt(41)

        // CPU reference
        let permArg = PermutationArgument(numWires: numWires, cosetGenerators: [k1, k2])
        let zCPU = permArg.computeGrandProduct(
            witness: w, sigma: sigma,
            beta: beta, gamma: gamma, domain: domain
        )

        // GPU
        let zGPU = engine.computePermutationPoly(
            witness: w, sigmaPolys: sigma,
            beta: beta, gamma: gamma
        )

        var match = true
        for i in 0..<n {
            if zCPU[i] != zGPU[i] {
                match = false
                break
            }
        }
        expect(match, "GPU perm: matches CPU PermutationArgument result")
    }

    // ========== Test 4: Verification passes/fails ==========
    do {
        let engine = try! GPUPermutationEngine()

        let logN = 3
        let n = 1 << logN
        let numWires = 3

        let omega = computeNthRootOfUnity(logN: logN)
        var domain = [Fr](repeating: Fr.zero, count: n)
        domain[0] = Fr.one
        for i in 1..<n { domain[i] = frMul(domain[i - 1], omega) }

        let k1 = frFromInt(2)
        let k2 = frFromInt(3)

        // Identity permutation
        var sigma = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: n), count: numWires)
        for i in 0..<n {
            sigma[0][i] = domain[i]
            sigma[1][i] = frMul(k1, domain[i])
            sigma[2][i] = frMul(k2, domain[i])
        }

        let witness: [[Fr]] = (0..<numWires).map { j in
            (0..<n).map { i in frFromInt(UInt64(10 + j * n + i)) }
        }

        let beta = frFromInt(11)
        let gamma = frFromInt(17)

        let Z = engine.computePermutationPoly(
            witness: witness, sigmaPolys: sigma,
            beta: beta, gamma: gamma
        )

        // Valid Z should verify
        let valid = engine.verifyPermutation(
            z: Z, witness: witness, sigmaPolys: sigma,
            beta: beta, gamma: gamma
        )
        expect(valid, "GPU perm: verification passes for valid Z")

        // Tampered Z should fail
        var badZ = Z
        badZ[3] = frFromInt(999)
        let invalid = engine.verifyPermutation(
            z: badZ, witness: witness, sigmaPolys: sigma,
            beta: beta, gamma: gamma
        )
        expect(!invalid, "GPU perm: verification fails for tampered Z")

        // Wrong Z[0] should fail
        var badStart = Z
        badStart[0] = frFromInt(2)
        let invalidStart = engine.verifyPermutation(
            z: badStart, witness: witness, sigmaPolys: sigma,
            beta: beta, gamma: gamma
        )
        expect(!invalidStart, "GPU perm: verification fails for Z[0] != 1")
    }

    // ========== Test 5: Multi-wire (4 wires) ==========
    do {
        let engine = try! GPUPermutationEngine()

        let logN = 3
        let n = 1 << logN
        let numWires = 4

        let omega = computeNthRootOfUnity(logN: logN)
        var domain = [Fr](repeating: Fr.zero, count: n)
        domain[0] = Fr.one
        for i in 1..<n { domain[i] = frMul(domain[i - 1], omega) }

        // Identity permutation for 4 wires with cosetMuls = [1, 2, 3, 4]
        var sigma = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: n), count: numWires)
        for j in 0..<numWires {
            let km = j == 0 ? Fr.one : frFromInt(UInt64(j + 1))
            for i in 0..<n {
                sigma[j][i] = frMul(km, domain[i])
            }
        }

        let witness: [[Fr]] = (0..<numWires).map { j in
            (0..<n).map { i in frFromInt(UInt64(5 + j * n + i)) }
        }

        let beta = frFromInt(19)
        let gamma = frFromInt(23)

        let Z = engine.computePermutationPoly(
            witness: witness, sigmaPolys: sigma,
            beta: beta, gamma: gamma
        )

        var allOnes = true
        for i in 0..<n {
            if Z[i] != Fr.one { allOnes = false; break }
        }
        expect(allOnes, "GPU perm: 4-wire identity permutation -> Z is all ones")

        let valid = engine.verifyPermutation(
            z: Z, witness: witness, sigmaPolys: sigma,
            beta: beta, gamma: gamma
        )
        expect(valid, "GPU perm: 4-wire identity verification passes")
    }

    // ========== Test 6: Larger domain (2^10) ==========
    do {
        let engine = try! GPUPermutationEngine()

        let logN = 10
        let n = 1 << logN  // 1024
        let numWires = 3

        let omega = computeNthRootOfUnity(logN: logN)
        var domain = [Fr](repeating: Fr.zero, count: n)
        domain[0] = Fr.one
        for i in 1..<n { domain[i] = frMul(domain[i - 1], omega) }

        let k1 = frFromInt(2)
        let k2 = frFromInt(3)

        // Create several copy constraints
        var witness: [[Fr]] = (0..<numWires).map { _ in
            (0..<n).map { _ in frFromInt(UInt64.random(in: 1...100000)) }
        }

        // Shared values at several positions
        for idx in stride(from: 0, to: 100, by: 2) {
            let val = frFromInt(UInt64(50000 + idx))
            witness[0][idx] = val
            witness[1][idx + 1] = val
        }

        var copies = [PlonkCopyConstraint]()
        for idx in stride(from: 0, to: 100, by: 2) {
            copies.append(PlonkCopyConstraint(srcWire: 0, srcRow: idx, dstWire: 1, dstRow: idx + 1))
        }

        let sigma = buildPermutationFromCopyConstraints(
            copies: copies, numWires: numWires, domainSize: n,
            domain: domain, cosetGenerators: [k1, k2]
        )

        let beta = frFromInt(37)
        let gamma = frFromInt(53)

        let Z = engine.computePermutationPoly(
            witness: witness, sigmaPolys: sigma,
            beta: beta, gamma: gamma
        )

        expect(Z[0] == Fr.one, "GPU perm: 2^10 domain Z[0] = 1")

        // Verify the full grand product wraps
        var finalNum = Fr.one
        var finalDen = Fr.one
        for j in 0..<numWires {
            let kj: Fr = j == 0 ? Fr.one : (j == 1 ? k1 : k2)
            let idVal = frMul(kj, domain[n - 1])
            finalNum = frMul(finalNum, frAdd(frAdd(witness[j][n - 1], frMul(beta, idVal)), gamma))
            finalDen = frMul(finalDen, frAdd(frAdd(witness[j][n - 1], frMul(beta, sigma[j][n - 1])), gamma))
        }
        let finalZ = frMul(Z[n - 1], frMul(finalNum, frInverse(finalDen)))
        expect(finalZ == Fr.one, "GPU perm: 2^10 domain grand product wraps to 1")

        let valid = engine.verifyPermutation(
            z: Z, witness: witness, sigmaPolys: sigma,
            beta: beta, gamma: gamma
        )
        expect(valid, "GPU perm: 2^10 domain verification passes")
    }
}
