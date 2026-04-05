// Plonky2RecursiveTests — Tests for Plonky2 recursive verification circuit
//
// Tests cover:
//   1. GoldilocksExtField arithmetic (add, mul, inv)
//   2. GoldilocksPoseidon hash correctness
//   3. Plonky2VerifierCircuit construction (circuit builds without error)
//   4. ProofComposition pipeline (composable proof interface)
//   5. Plonky2ToGroth16Bridge setup

import zkMetal
import Foundation

func runPlonky2RecursiveTests() {
    suite("Goldilocks Extension Field")

    // Test 1: Extension field zero and one
    do {
        let zero = GoldilocksExtField.zero
        let one = GoldilocksExtField.one
        expect(zero.isZero, "zero should be zero")
        expect(!one.isZero, "one should not be zero")
        expectEqual(one.c0, Gl.one, "one.c0 should be Gl.one")
        expectEqual(one.c1, Gl.zero, "one.c1 should be Gl.zero")
    }

    // Test 2: Extension field addition
    do {
        let a = GoldilocksExtField(c0: Gl(v: 5), c1: Gl(v: 3))
        let b = GoldilocksExtField(c0: Gl(v: 7), c1: Gl(v: 11))
        let c = glExtAdd(a, b)
        expectEqual(c.c0.v, 12, "ext add c0")
        expectEqual(c.c1.v, 14, "ext add c1")
    }

    // Test 3: Extension field subtraction
    do {
        let a = GoldilocksExtField(c0: Gl(v: 10), c1: Gl(v: 20))
        let b = GoldilocksExtField(c0: Gl(v: 3), c1: Gl(v: 5))
        let c = glExtSub(a, b)
        expectEqual(c.c0.v, 7, "ext sub c0")
        expectEqual(c.c1.v, 15, "ext sub c1")
    }

    // Test 4: Extension field multiplication identity
    do {
        let a = GoldilocksExtField(c0: Gl(v: 123456789), c1: Gl(v: 987654321))
        let one = GoldilocksExtField.one
        let aMulOne = glExtMul(a, one)
        expectEqual(aMulOne.c0, a.c0, "a * 1 = a (c0)")
        expectEqual(aMulOne.c1, a.c1, "a * 1 = a (c1)")
    }

    // Test 5: Extension field multiplication: (a+bW)*(a-bW) = a^2 - 7*b^2
    do {
        let a = Gl(v: 100)
        let b = Gl(v: 10)
        let x = GoldilocksExtField(c0: a, c1: b)
        let xConj = GoldilocksExtField(c0: a, c1: glNeg(b))
        let prod = glExtMul(x, xConj)
        // c1 should be 0 (norm is in base field)
        expectEqual(prod.c1.v, 0, "conjugate product c1 should be 0")
        // c0 should be a^2 - 7*b^2 = 10000 - 700 = 9300
        let expected = glSub(glSqr(a), glMul(Gl(v: 7), glSqr(b)))
        expectEqual(prod.c0.v, expected.v, "conjugate product c0 = a^2 - 7*b^2")
    }

    // Test 6: Extension field inverse
    do {
        let a = GoldilocksExtField(c0: Gl(v: 42), c1: Gl(v: 17))
        let aInv = glExtInverse(a)
        let prod = glExtMul(a, aInv)
        expectEqual(prod.c0.v, 1, "a * a^-1 = 1 (c0)")
        expectEqual(prod.c1.v, 0, "a * a^-1 = 1 (c1)")
    }

    // Test 7: Extension field squaring consistency
    do {
        let a = GoldilocksExtField(c0: Gl(v: 999), c1: Gl(v: 777))
        let sqr = glExtSqr(a)
        let mulSelf = glExtMul(a, a)
        expectEqual(sqr.c0.v, mulSelf.c0.v, "sqr(a) = a*a (c0)")
        expectEqual(sqr.c1.v, mulSelf.c1.v, "sqr(a) = a*a (c1)")
    }

    suite("Goldilocks Poseidon Hash")

    // Test 8: Poseidon permutation determinism
    do {
        let input = (0..<GoldilocksPoseidon.stateWidth).map { Gl(v: UInt64($0 + 1)) }
        let out1 = GoldilocksPoseidon.permutation(input)
        let out2 = GoldilocksPoseidon.permutation(input)
        for i in 0..<GoldilocksPoseidon.stateWidth {
            expectEqual(out1[i].v, out2[i].v, "poseidon deterministic at \(i)")
        }
    }

    // Test 9: Poseidon permutation non-trivial
    do {
        let input = [Gl](repeating: Gl.zero, count: GoldilocksPoseidon.stateWidth)
        let output = GoldilocksPoseidon.permutation(input)
        // Output should not be all zeros
        let allZero = output.allSatisfy { $0.v == 0 }
        expect(!allZero, "poseidon of zeros should not be all zeros")
    }

    // Test 10: Poseidon compression (2-to-1)
    do {
        let left = [Gl(v: 1), Gl(v: 2), Gl(v: 3), Gl(v: 4)]
        let right = [Gl(v: 5), Gl(v: 6), Gl(v: 7), Gl(v: 8)]
        let hash = GoldilocksPoseidon.compress(left, right)
        expectEqual(hash.count, GoldilocksPoseidon.capacity, "compress output size")
        // Compression should be different from left or right
        expect(hash[0].v != left[0].v, "compress should mix inputs")
    }

    // Test 11: Poseidon hashMany
    do {
        let inputs = (0..<20).map { Gl(v: UInt64($0 * 7 + 1)) }
        let hash = GoldilocksPoseidon.hashMany(inputs)
        expectEqual(hash.count, GoldilocksPoseidon.capacity, "hashMany output size")
        expect(!hash.allSatisfy { $0.v == 0 }, "hashMany should produce non-zero output")
    }

    suite("Plonky2 Verifier Circuit Construction")

    // Test 12: Circuit builder constructs without crash
    do {
        let builder = PlonkCircuitBuilder()
        let verifier = Plonky2VerifierCircuit(builder: builder)

        // Allocate some variables
        let a = verifier.allocateGl()
        let b = verifier.allocateGl()
        let _ = verifier.glAddCircuit(a, b)

        let extA = verifier.allocateExt()
        let extB = verifier.allocateExt()
        let _ = verifier.extMulCircuit(extA, extB)

        let circuit = builder.build()
        expect(circuit.numGates > 0, "verifier circuit should have gates")
        print("  Plonky2 verifier circuit: \(circuit.numGates) gates for basic ops")
    }

    // Test 13: FRI fold step builds
    do {
        let builder = PlonkCircuitBuilder()
        let verifier = Plonky2VerifierCircuit(builder: builder)

        let fAtX = verifier.allocateExt()
        let fAtNegX = verifier.allocateExt()
        let beta = verifier.allocateExt()
        let xVal = verifier.allocateExt()
        let result = verifier.allocateExt()

        verifier.friFoldStepCircuit(
            fAtX: fAtX, fAtNegX: fAtNegX,
            beta: beta, xVal: xVal, result: result
        )

        let circuit = builder.build()
        expect(circuit.numGates > 0, "FRI fold step should produce gates")
        print("  FRI fold step: \(circuit.numGates) gates")
    }

    // Test 14: Poseidon in-circuit permutation builds
    do {
        let builder = PlonkCircuitBuilder()
        let verifier = Plonky2VerifierCircuit(builder: builder)

        let input = (0..<GoldilocksPoseidon.stateWidth).map { _ in verifier.allocateGl() }
        let _ = verifier.poseidonPermutationCircuit(input)

        let circuit = builder.build()
        expect(circuit.numGates > 100, "in-circuit poseidon should produce many gates")
        print("  In-circuit Poseidon: \(circuit.numGates) gates")
    }

    suite("Plonky2 Proof Composition")

    // Test 15: Full FRI verification circuit builds
    do {
        let vk = Plonky2VerificationKey(
            numWires: 80,
            numRoutedWires: 80,
            degreeBits: 4,
            friRateBits: 3,
            numFRIQueries: 2,
            numConstants: 4,
            numPublicInputs: 2,
            circuitDigest: [Gl(v: 123), Gl(v: 456), Gl(v: 789), Gl(v: 101112)]
        )

        let bridge = Plonky2ToGroth16Bridge(vk: vk)
        let (circuit, r1cs) = bridge.setup()

        expect(circuit.numGates > 0, "bridge circuit should have gates")
        expect(r1cs.numConstraints > 0, "bridge R1CS should have constraints")
        print("  Plonky2->Groth16 bridge: \(circuit.numGates) gates, \(r1cs.numConstraints) R1CS constraints")
    }

    // Test 16: Composable proof interface works
    do {
        let vk = Plonky2VerificationKey(
            numWires: 80, numRoutedWires: 80,
            degreeBits: 3, friRateBits: 2,
            numFRIQueries: 1, numConstants: 4,
            numPublicInputs: 1,
            circuitDigest: [Gl(v: 1), Gl(v: 2), Gl(v: 3), Gl(v: 4)]
        )

        let bridge = Plonky2ToGroth16Bridge(vk: vk)
        let dummyProof = bridge.wrapProof(proof: makeDummyPlonky2ProofForTest(vk: vk))
        expect(dummyProof.circuit.numGates > 0, "composed circuit should have gates")
        print("  Composed proof circuit: \(dummyProof.circuit.numGates) gates")
    }

    // Test 17: Estimated gas cost
    do {
        let vk = Plonky2VerificationKey(
            numWires: 80, numRoutedWires: 80,
            degreeBits: 10, friRateBits: 3,
            numFRIQueries: 28, numConstants: 4,
            numPublicInputs: 4,
            circuitDigest: [Gl(v: 1), Gl(v: 2), Gl(v: 3), Gl(v: 4)]
        )
        let bridge = Plonky2ToGroth16Bridge(vk: vk)
        expectEqual(bridge.estimatedEVMGasCost, 200_000, "EVM gas cost")
        expect(bridge.estimatedConstraints > 0, "estimated constraints > 0")
        print("  Estimated constraints for 28-query FRI: \(bridge.estimatedConstraints)")
    }
}

// Helper: create a dummy Plonky2 proof for testing
private func makeDummyPlonky2ProofForTest(vk: Plonky2VerificationKey) -> Plonky2Proof {
    let cap = GoldilocksPoseidon.capacity
    let numRounds = vk.degreeBits
    let merkleDepth = vk.degreeBits + vk.friRateBits

    let dummyRoot = [Gl](repeating: .zero, count: cap)
    let dummyMerklePath = Plonky2MerklePath(
        siblings: [[Gl]](repeating: [Gl](repeating: .zero, count: cap), count: merkleDepth),
        index: 0
    )
    let dummyRound = Plonky2FRIQueryRound(cosetEvals: [.zero, .zero], merklePath: dummyMerklePath)
    let friProof = Plonky2FRIProof(
        initialTreeRoot: dummyRoot,
        commitRoots: [[Gl]](repeating: dummyRoot, count: numRounds),
        queryRoundData: [[Plonky2FRIQueryRound]](
            repeating: [Plonky2FRIQueryRound](repeating: dummyRound, count: numRounds),
            count: vk.numFRIQueries
        ),
        finalPoly: [.zero, .zero],
        powNonce: 0
    )
    let openings = Plonky2Openings(atZeta: [.zero], atZetaNext: [.zero])
    return Plonky2Proof(
        publicInputs: [Gl](repeating: .zero, count: vk.numPublicInputs),
        wires: [[Gl](repeating: .zero, count: cap)],
        plonkZsPartialProducts: [[Gl](repeating: .zero, count: cap)],
        quotientPolys: [[Gl](repeating: .zero, count: cap)],
        openingProof: friProof,
        openings: openings
    )
}
