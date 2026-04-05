// Groth16 Recursive Verifier and Proof Aggregation Tests
import zkMetal
import Foundation

// MARK: - Test Helpers

private var passed = 0
private var failed = 0

private func check(_ condition: Bool, _ msg: String) {
    if condition {
        passed += 1
        fputs("  PASS: \(msg)\n", stderr)
    } else {
        failed += 1
        fputs("  FAIL: \(msg)\n", stderr)
    }
}

/// Generate a Groth16 proof for the example circuit (x^3 + x + 5 = y) with a given x value.
private func generateExampleProof(x: UInt64, pk: Groth16ProvingKey, r1cs: R1CSInstance) -> (Groth16Proof, [Fr]) {
    let (pubInputs, witness) = computeExampleWitness(x: x)
    let prover = try! Groth16Prover()
    let proof = try! prover.prove(pk: pk, r1cs: r1cs, publicInputs: pubInputs, witness: witness)
    return (proof, pubInputs)
}

// MARK: - Tests

public func runGroth16RecursiveTests() {
    passed = 0
    failed = 0
    fputs("[Groth16 Recursive Tests]\n", stderr)

    // Setup: build example circuit and generate keys
    let r1cs = buildExampleCircuit()
    let setup = Groth16Setup()
    let (pk, vk) = setup.setup(r1cs: r1cs)
    let verifier = Groth16Verifier()

    // --- Verifier Circuit Tests ---

    testVerifierCircuitBuild(vk: vk)
    testVerifierCircuitWitness(vk: vk)
    testVerifierCircuitSatisfaction(vk: vk)

    // --- Batch Verification Tests ---

    testBatchVerify2Proofs(pk: pk, vk: vk, r1cs: r1cs, verifier: verifier)
    testBatchVerify4Proofs(pk: pk, vk: vk, r1cs: r1cs, verifier: verifier)
    testInvalidProofInBatch(pk: pk, vk: vk, r1cs: r1cs, verifier: verifier)
    testSingleProofBatchVerify(pk: pk, vk: vk, r1cs: r1cs, verifier: verifier)

    fputs("[Groth16 Recursive] \(passed) passed, \(failed) failed\n", stderr)
}

// MARK: - Verifier Circuit Tests

/// Test: build verifier circuit for a VK with 2 public inputs.
private func testVerifierCircuitBuild(vk: Groth16VerificationKey) {
    let circuit = Groth16VerifierCircuit(innerPublicInputCount: 2)
    let (r1cs, _) = circuit.buildCircuit(vkSize: 3)

    // Should have constraints for:
    // 2 products (x and y each) = 4 mul constraints
    // 2 sum constraints (sum_x, sum_y)
    // 2 equality constraints (sum_x == accum_x, sum_y == accum_y)
    // Total = 8
    check(r1cs.numConstraints == 8, "verifier circuit has 8 constraints for 2 public inputs")
    check(r1cs.numPublic == 4, "verifier circuit has 4 public vars (2 scalars + 2 accum coords)")
    check(r1cs.numVars > 4, "verifier circuit has witness variables")
}

/// Test: witness generation for the MSM accumulation part.
private func testVerifierCircuitWitness(vk: Groth16VerificationKey) {
    let circuit = Groth16VerifierCircuit(innerPublicInputCount: 2)
    let (_, witnessMapper) = circuit.buildCircuit(vkSize: 3)

    // Use dummy proof (not needed for MSM check, only for witness mapper signature)
    let dummyProof = Groth16Proof(
        a: pointIdentity(), b: g2Identity(), c: pointIdentity()
    )

    let pubInputs: [Fr] = [frFromInt(3), frFromInt(35)]  // x=3, y=35
    let z = witnessMapper(dummyProof, vk, pubInputs)

    // z[0] should be 1
    check(frEq(z[0], .one), "witness z[0] = 1")
    // z[1], z[2] should be the public inputs
    check(frEq(z[1], frFromInt(3)), "witness z[1] = public input 0")
    check(frEq(z[2], frFromInt(35)), "witness z[2] = public input 1")
    // z[3], z[4] should be the accumulation point coordinates (non-zero)
    check(!z[3].isZero || !z[4].isZero, "witness has non-zero accumulation point")
}

/// Test: the generated witness satisfies the verifier circuit R1CS.
private func testVerifierCircuitSatisfaction(vk: Groth16VerificationKey) {
    let circuit = Groth16VerifierCircuit(innerPublicInputCount: 2)
    let (r1cs, witnessMapper) = circuit.buildCircuit(vkSize: 3)

    let dummyProof = Groth16Proof(
        a: pointIdentity(), b: g2Identity(), c: pointIdentity()
    )

    let pubInputs: [Fr] = [frFromInt(3), frFromInt(35)]
    let z = witnessMapper(dummyProof, vk, pubInputs)

    let satisfied = r1cs.isSatisfied(z: z)
    check(satisfied, "verifier circuit R1CS is satisfied with valid witness")

    // Test with different inputs
    let pubInputs2: [Fr] = [frFromInt(7), frFromInt(355)]  // 7^3 + 7 + 5 = 355
    let z2 = witnessMapper(dummyProof, vk, pubInputs2)
    check(r1cs.isSatisfied(z: z2), "verifier circuit satisfied with second set of inputs")

    // Test with zero inputs
    let pubInputs3: [Fr] = [.zero, frFromInt(5)]  // 0^3 + 0 + 5 = 5
    let z3 = witnessMapper(dummyProof, vk, pubInputs3)
    check(r1cs.isSatisfied(z: z3), "verifier circuit satisfied with zero input")
}

// MARK: - Batch Verification Tests

/// Test: batch verify 2 proofs.
private func testBatchVerify2Proofs(pk: Groth16ProvingKey, vk: Groth16VerificationKey,
                                     r1cs: R1CSInstance, verifier: Groth16Verifier) {
    let (proof1, pub1) = generateExampleProof(x: 3, pk: pk, r1cs: r1cs)
    let (proof2, pub2) = generateExampleProof(x: 5, pk: pk, r1cs: r1cs)

    // Sanity: individual proofs verify
    check(verifier.verify(proof: proof1, vk: vk, publicInputs: pub1), "proof1 individually valid")
    check(verifier.verify(proof: proof2, vk: vk, publicInputs: pub2), "proof2 individually valid")

    let aggregator = RecursiveAggregator()
    let bundle = aggregator.aggregate(
        proofs: [(proof1, pub1), (proof2, pub2)],
        vk: vk
    )

    check(bundle.proofs.count == 2, "aggregation bundle contains 2 proofs")
    check(bundle.challenges.count == 2, "aggregation bundle has 2 challenges")

    let valid = aggregator.verifyAggregated(bundle: bundle, vk: vk)
    check(valid, "aggregated 2-proof verification passes")
}

/// Test: batch verify 4 proofs.
private func testBatchVerify4Proofs(pk: Groth16ProvingKey, vk: Groth16VerificationKey,
                                     r1cs: R1CSInstance, verifier: Groth16Verifier) {
    var proofsAndInputs = [(Groth16Proof, [Fr])]()
    for x: UInt64 in [2, 3, 7, 10] {
        let (proof, pub) = generateExampleProof(x: x, pk: pk, r1cs: r1cs)
        proofsAndInputs.append((proof, pub))
    }

    let aggregator = RecursiveAggregator()
    let bundle = aggregator.aggregate(
        proofs: proofsAndInputs.map { (proof: $0.0, publicInputs: $0.1) },
        vk: vk
    )

    check(bundle.proofs.count == 4, "aggregation bundle contains 4 proofs")

    let valid = aggregator.verifyAggregated(bundle: bundle, vk: vk)
    check(valid, "aggregated 4-proof verification passes")
}

/// Test: invalid proof in batch is detected.
private func testInvalidProofInBatch(pk: Groth16ProvingKey, vk: Groth16VerificationKey,
                                      r1cs: R1CSInstance, verifier: Groth16Verifier) {
    let (proof1, pub1) = generateExampleProof(x: 3, pk: pk, r1cs: r1cs)
    let (proof2, _) = generateExampleProof(x: 5, pk: pk, r1cs: r1cs)

    // Create wrong public inputs for proof2
    let wrongPub2: [Fr] = [frFromInt(5), frFromInt(999)]  // wrong y value

    // Sanity: proof2 with wrong inputs fails individually
    check(!verifier.verify(proof: proof2, vk: vk, publicInputs: wrongPub2),
          "proof2 with wrong inputs fails individually")

    let aggregator = RecursiveAggregator()

    // Batch with one invalid should fail (with overwhelming probability)
    let bundle = aggregator.aggregate(
        proofs: [(proof1, pub1), (proof2, wrongPub2)],
        vk: vk
    )

    let valid = aggregator.verifyAggregated(bundle: bundle, vk: vk)
    check(!valid, "batch with invalid proof is rejected")
}

/// Test: single proof through batch pipeline matches standard verify.
private func testSingleProofBatchVerify(pk: Groth16ProvingKey, vk: Groth16VerificationKey,
                                         r1cs: R1CSInstance, verifier: Groth16Verifier) {
    let (proof, pub) = generateExampleProof(x: 42, pk: pk, r1cs: r1cs)

    // Standard verification
    let stdValid = verifier.verify(proof: proof, vk: vk, publicInputs: pub)
    check(stdValid, "standard verification of proof passes")

    // Batch pipeline with single proof
    let batchValid = groth16VerifySingle(proof: proof, vk: vk, publicInputs: pub)
    check(batchValid, "single-proof batch verification passes")

    // Both should agree
    check(stdValid == batchValid, "batch verification matches standard verification")
}
