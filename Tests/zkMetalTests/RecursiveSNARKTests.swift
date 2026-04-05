// RecursiveSNARK Tests — recursive proof composition engine
import zkMetal
import Foundation

// MARK: - Test Runner

public func runRecursiveSNARKTests() {
    _recursivePassed = 0
    _recursiveFailed = 0
    fputs("\n--- Recursive SNARK Composition ---\n", stderr)

    // Build inner circuit and keys once (shared across tests)
    let innerR1CS = buildExampleCircuit()  // x^3 + x + 5 = y
    let setup = Groth16Setup()
    let (innerPK, innerVK) = setup.setup(r1cs: innerR1CS)

    testVerifierCircuitEncoding(innerVK: innerVK)
    testVerifierCircuitWitnessGeneration(innerVK: innerVK)
    testVerifierCircuitSatisfaction(innerVK: innerVK)
    testSingleLevelRecursion(innerPK: innerPK, innerVK: innerVK, innerR1CS: innerR1CS)
    testPublicInputPropagation(innerPK: innerPK, innerVK: innerVK, innerR1CS: innerR1CS)
    testRecursiveProofSize(innerPK: innerPK, innerVK: innerVK, innerR1CS: innerR1CS)
    testSoundnessInvalidInnerProof(innerPK: innerPK, innerVK: innerVK, innerR1CS: innerR1CS)

    fputs("[Recursive SNARK] \(_recursivePassed) passed, \(_recursiveFailed) failed\n", stderr)
}

// MARK: - Test Helpers

private var _recursivePassed = 0
private var _recursiveFailed = 0

private func check(_ condition: Bool, _ msg: String) {
    if condition {
        _recursivePassed += 1
        fputs("  PASS: \(msg)\n", stderr)
    } else {
        _recursiveFailed += 1
        fputs("  FAIL: \(msg)\n", stderr)
    }
}

/// Generate a valid inner Groth16 proof for x^3 + x + 5 = y.
private func makeInnerProof(x: UInt64, pk: Groth16ProvingKey,
                             r1cs: R1CSInstance) -> (Groth16Proof, [Fr]) {
    let (pubInputs, witness) = computeExampleWitness(x: x)
    let prover = try! Groth16Prover()
    let proof = try! prover.prove(pk: pk, r1cs: r1cs, publicInputs: pubInputs, witness: witness)
    return (proof, pubInputs)
}

// MARK: - Tests

/// Test 1: Encode Groth16 verifier as R1CS circuit constraints.
private func testVerifierCircuitEncoding(innerVK: Groth16VerificationKey) {
    let encoder = Groth16VerifierCircuitEncoder(innerPublicInputCount: 2)
    let (r1cs, _) = encoder.buildVerifierR1CS()

    // For 2 public inputs:
    // 2*2 product constraints (x and y for each input) = 4
    // 2 sum constraints (sum_x, sum_y) = 2
    // 2 equality constraints (sum_x == accum_x, sum_y == accum_y) = 2
    // Total = 8
    check(r1cs.numConstraints == 8,
          "verifier circuit has 8 constraints for 2 public inputs (got \(r1cs.numConstraints))")
    check(r1cs.numPublic == 4,
          "verifier circuit has 4 public vars: 2 scalars + 2 accum coords (got \(r1cs.numPublic))")
    check(r1cs.numVars > r1cs.numPublic + 1,
          "verifier circuit has witness variables beyond public inputs")

    // Verify estimated count matches actual
    check(encoder.estimatedConstraintCount == r1cs.numConstraints,
          "estimated constraint count matches actual")
}

/// Test 2: Witness generation produces valid assignments.
private func testVerifierCircuitWitnessGeneration(innerVK: Groth16VerificationKey) {
    let encoder = Groth16VerifierCircuitEncoder(innerPublicInputCount: 2)
    let (r1cs, witnessMapper) = encoder.buildVerifierR1CS()

    let dummyProof = Groth16Proof(a: pointIdentity(), b: g2Identity(), c: pointIdentity())
    let pubInputs: [Fr] = [frFromInt(3), frFromInt(35)]  // x=3, y=x^3+x+5=35
    let z = witnessMapper(dummyProof, innerVK, pubInputs)

    // Check z[0] = 1 (constant)
    check(frEq(z[0], .one), "witness z[0] = 1 (constant)")

    // Check public inputs are at expected positions
    check(frEq(z[1], frFromInt(3)), "witness z[1] = first public input (x=3)")
    check(frEq(z[2], frFromInt(35)), "witness z[2] = second public input (y=35)")

    // Check accumulation point coordinates are non-trivial
    check(!z[3].isZero || !z[4].isZero, "accumulation point is non-zero")

    // Check total size matches R1CS numVars
    check(z.count == r1cs.numVars, "witness vector size matches R1CS numVars")
}

/// Test 3: Generated witness satisfies the verifier circuit R1CS.
private func testVerifierCircuitSatisfaction(innerVK: Groth16VerificationKey) {
    let encoder = Groth16VerifierCircuitEncoder(innerPublicInputCount: 2)
    let (r1cs, witnessMapper) = encoder.buildVerifierR1CS()
    let dummyProof = Groth16Proof(a: pointIdentity(), b: g2Identity(), c: pointIdentity())

    // Test with x=3, y=35
    let z1 = witnessMapper(dummyProof, innerVK, [frFromInt(3), frFromInt(35)])
    check(r1cs.isSatisfied(z: z1), "verifier circuit R1CS satisfied for (x=3, y=35)")

    // Test with x=7, y=355 (7^3 + 7 + 5 = 355)
    let z2 = witnessMapper(dummyProof, innerVK, [frFromInt(7), frFromInt(355)])
    check(r1cs.isSatisfied(z: z2), "verifier circuit R1CS satisfied for (x=7, y=355)")

    // Test with x=0, y=5 (0^3 + 0 + 5 = 5)
    let z3 = witnessMapper(dummyProof, innerVK, [.zero, frFromInt(5)])
    check(r1cs.isSatisfied(z: z3), "verifier circuit R1CS satisfied for (x=0, y=5)")
}

/// Test 4: Single-level recursion: prove(verify(inner_proof)).
private func testSingleLevelRecursion(innerPK: Groth16ProvingKey,
                                       innerVK: Groth16VerificationKey,
                                       innerR1CS: R1CSInstance) {
    // Generate a valid inner proof for x=3
    let (innerProof, innerPubInputs) = makeInnerProof(x: 3, pk: innerPK, r1cs: innerR1CS)

    // Sanity: inner proof verifies natively
    let verifier = Groth16Verifier()
    let innerValid = verifier.verify(proof: innerProof, vk: innerVK, publicInputs: innerPubInputs)
    check(innerValid, "inner proof verifies natively before recursion")

    // Produce recursive proof
    let recursiveProver = RecursiveSNARKProver()
    do {
        let recursiveProof = try recursiveProver.prove(
            innerProof: innerProof,
            innerVK: innerVK,
            innerPublicInputs: innerPubInputs,
            depth: 1
        )

        check(recursiveProof.depth == 1, "recursive proof has depth 1")

        // Verify the recursive proof
        guard let outerVK = recursiveProver.outerVerificationKey else {
            check(false, "outer VK should be available after proving")
            return
        }

        let recursiveVerifier = RecursiveSNARKVerifier()
        let valid = recursiveVerifier.verify(
            recursiveProof: recursiveProof,
            outerVK: outerVK
        )
        check(valid, "single-level recursive proof verifies")

        // Also check outer-only verification
        let outerOnlyValid = recursiveVerifier.verifyOuterOnly(
            recursiveProof: recursiveProof,
            outerVK: outerVK
        )
        check(outerOnlyValid, "outer-only verification passes")

    } catch {
        check(false, "recursive proving threw: \(error)")
    }
}

/// Test 5: Public input propagation through recursion.
private func testPublicInputPropagation(innerPK: Groth16ProvingKey,
                                         innerVK: Groth16VerificationKey,
                                         innerR1CS: R1CSInstance) {
    // Test with multiple different x values to verify public inputs flow correctly
    for x: UInt64 in [2, 5, 10] {
        let (innerProof, innerPubInputs) = makeInnerProof(x: x, pk: innerPK, r1cs: innerR1CS)

        let recursiveProver = RecursiveSNARKProver()
        do {
            let recursiveProof = try recursiveProver.prove(
                innerProof: innerProof,
                innerVK: innerVK,
                innerPublicInputs: innerPubInputs
            )

            // Verify propagated public inputs match the originals
            check(recursiveProof.propagatedPublicInputs.count == innerPubInputs.count,
                  "propagated input count matches for x=\(x)")

            var allMatch = true
            for i in 0..<innerPubInputs.count {
                if !frEq(recursiveProof.propagatedPublicInputs[i], innerPubInputs[i]) {
                    allMatch = false
                    break
                }
            }
            check(allMatch, "propagated public inputs match inner inputs for x=\(x)")

            // Verify the recursive proof
            guard let outerVK = recursiveProver.outerVerificationKey else {
                check(false, "outer VK not available for x=\(x)")
                continue
            }

            let recursiveVerifier = RecursiveSNARKVerifier()
            let valid = recursiveVerifier.verify(
                recursiveProof: recursiveProof,
                outerVK: outerVK
            )
            check(valid, "recursive proof with propagated inputs verifies for x=\(x)")
        } catch {
            check(false, "recursive proving threw for x=\(x): \(error)")
        }
    }
}

/// Test 6: Recursive proof size measurement.
private func testRecursiveProofSize(innerPK: Groth16ProvingKey,
                                     innerVK: Groth16VerificationKey,
                                     innerR1CS: R1CSInstance) {
    let (innerProof, innerPubInputs) = makeInnerProof(x: 3, pk: innerPK, r1cs: innerR1CS)

    let recursiveProver = RecursiveSNARKProver()
    do {
        let recursiveProof = try recursiveProver.prove(
            innerProof: innerProof,
            innerVK: innerVK,
            innerPublicInputs: innerPubInputs
        )

        let size = recursiveProof.sizeBytes
        // Expected: 2 * 256 (two Groth16 proofs) + 2 * 32 (two Fr public inputs) = 576 bytes
        let expectedSize = 2 * 256 + innerPubInputs.count * 32
        check(size == expectedSize,
              "recursive proof size is \(size) bytes (expected \(expectedSize))")
        check(size < 1024, "recursive proof is compact (< 1 KB)")

        fputs("  INFO: recursive proof size = \(size) bytes, depth = \(recursiveProof.depth)\n", stderr)
    } catch {
        check(false, "recursive proving threw for size test: \(error)")
    }
}

/// Test 7: Soundness -- invalid inner proof causes outer verification to fail.
private func testSoundnessInvalidInnerProof(innerPK: Groth16ProvingKey,
                                              innerVK: Groth16VerificationKey,
                                              innerR1CS: R1CSInstance) {
    // Generate a valid proof, then corrupt it
    let (validProof, validPubInputs) = makeInnerProof(x: 3, pk: innerPK, r1cs: innerR1CS)

    // Test A: Wrong public inputs (valid proof, incorrect inputs)
    let wrongPubInputs: [Fr] = [frFromInt(3), frFromInt(999)]  // wrong y

    let verifier = Groth16Verifier()
    check(!verifier.verify(proof: validProof, vk: innerVK, publicInputs: wrongPubInputs),
          "inner proof with wrong public inputs fails native verification")

    // The recursive prover should detect the invalid inner proof and produce
    // an invalid outer proof (or the verification should fail)
    let recursiveProver = RecursiveSNARKProver()
    do {
        let recursiveProof = try recursiveProver.prove(
            innerProof: validProof,
            innerVK: innerVK,
            innerPublicInputs: wrongPubInputs,
            depth: 1
        )

        // Even if proving "succeeds" (with dummy proof for invalid inner),
        // verification must fail
        if let outerVK = recursiveProver.outerVerificationKey {
            let recursiveVerifier = RecursiveSNARKVerifier()
            let valid = recursiveVerifier.verify(
                recursiveProof: recursiveProof,
                outerVK: outerVK
            )
            check(!valid, "recursive proof with invalid inner proof is rejected")
        } else {
            // Outer VK not cached means the prover detected invalidity early
            // and returned a dummy proof -- also acceptable for soundness
            check(true, "recursive prover detected invalid inner proof (no outer VK)")
        }
    } catch {
        // Throwing is also a valid soundness response
        check(true, "recursive prover threw on invalid inner proof: \(error)")
    }

    // Test B: Corrupted proof element (tamper with proof.a)
    let corruptedA = pointDouble(validProof.a)  // modify A point
    let corruptedProof = Groth16Proof(a: corruptedA, b: validProof.b, c: validProof.c)

    check(!verifier.verify(proof: corruptedProof, vk: innerVK, publicInputs: validPubInputs),
          "corrupted inner proof fails native verification")

    do {
        let recursiveProver2 = RecursiveSNARKProver()
        let recursiveProof2 = try recursiveProver2.prove(
            innerProof: corruptedProof,
            innerVK: innerVK,
            innerPublicInputs: validPubInputs,
            depth: 1
        )

        if let outerVK2 = recursiveProver2.outerVerificationKey {
            let recursiveVerifier2 = RecursiveSNARKVerifier()
            let valid2 = recursiveVerifier2.verify(
                recursiveProof: recursiveProof2,
                outerVK: outerVK2
            )
            check(!valid2, "recursive proof with corrupted inner proof.a is rejected")
        } else {
            check(true, "recursive prover detected corrupted proof (no outer VK)")
        }
    } catch {
        check(true, "recursive prover threw on corrupted proof: \(error)")
    }
}
