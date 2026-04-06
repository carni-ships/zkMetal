// GPUGroth16ProverEngine tests: GPU-accelerated Groth16 proof generation
import zkMetal

public func runGPUGroth16ProverTests() {
    suite("GPU Groth16 Prover Engine")

    // --- Test 1: Simple circuit proof generation ---
    do {
        let r1cs = buildExampleCircuit()  // x^3 + x + 5 = y
        let (pubInputs, witness) = computeExampleWitness(x: 3)

        let setup = Groth16Setup()
        let (pk, vk) = setup.setup(r1cs: r1cs)
        let engine = try GPUGroth16ProverEngine()
        let proof = try engine.generateProof(pk: pk, r1cs: r1cs,
                                              publicInputs: pubInputs, witness: witness)

        let verifier = Groth16Verifier()
        let valid = verifier.verify(proof: proof, vk: vk, publicInputs: pubInputs)
        expect(valid, "GPU engine proof verifies (x^3+x+5, x=3)")
    } catch {
        expect(false, "GPU engine simple circuit error: \(error)")
    }

    // --- Test 2: Proof elements are non-zero ---
    do {
        let r1cs = buildExampleCircuit()
        let (pubInputs, witness) = computeExampleWitness(x: 5)

        let setup = Groth16Setup()
        let (pk, _) = setup.setup(r1cs: r1cs)
        let engine = try GPUGroth16ProverEngine()
        let proof = try engine.generateProof(pk: pk, r1cs: r1cs,
                                              publicInputs: pubInputs, witness: witness)

        expect(!pointIsIdentity(proof.a), "GPU engine piA non-identity")
        expect(!pointIsIdentity(proof.c), "GPU engine piC non-identity")
    } catch {
        expect(false, "GPU engine non-zero check error: \(error)")
    }

    // --- Test 3: Random blinding produces different proofs ---
    do {
        let r1cs = buildExampleCircuit()
        let (pubInputs, witness) = computeExampleWitness(x: 7)

        let setup = Groth16Setup()
        let (pk, vk) = setup.setup(r1cs: r1cs)
        let engine = try GPUGroth16ProverEngine()
        let proof1 = try engine.generateProof(pk: pk, r1cs: r1cs,
                                               publicInputs: pubInputs, witness: witness)
        let proof2 = try engine.generateProof(pk: pk, r1cs: r1cs,
                                               publicInputs: pubInputs, witness: witness)

        // Both should verify
        let verifier = Groth16Verifier()
        expect(verifier.verify(proof: proof1, vk: vk, publicInputs: pubInputs),
               "GPU engine proof1 verifies")
        expect(verifier.verify(proof: proof2, vk: vk, publicInputs: pubInputs),
               "GPU engine proof2 verifies")

        // Proofs should differ due to random blinding (r, s)
        // Compare A.x coordinate as a proxy for inequality
        // If blinding differs, proof.a should differ
        let differ = !pointEqual(proof1.a, proof2.a)
        expect(differ, "GPU engine random blinding produces distinct proofs")
    } catch {
        expect(false, "GPU engine blinding error: \(error)")
    }

    // --- Test 4: Proof structure has all required fields ---
    do {
        let r1cs = buildExampleCircuit()
        let (pubInputs, witness) = computeExampleWitness(x: 2)

        let setup = Groth16Setup()
        let (pk, _) = setup.setup(r1cs: r1cs)
        let engine = try GPUGroth16ProverEngine()
        let proof = try engine.generateProof(pk: pk, r1cs: r1cs,
                                              publicInputs: pubInputs, witness: witness)

        // Proof should have A (G1), B (G2), C (G1)
        // Verify they are valid curve points by checking affine conversion succeeds
        let aOk = pointToAffine(proof.a) != nil
        let bOk = g2ToAffine(proof.b) != nil
        let cOk = pointToAffine(proof.c) != nil
        expect(aOk, "GPU engine proof.a is valid G1 point")
        expect(bOk, "GPU engine proof.b is valid G2 point")
        expect(cOk, "GPU engine proof.c is valid G1 point")
    } catch {
        expect(false, "GPU engine structure error: \(error)")
    }

    // --- Test 5: Determinism check (same blinding -> same proof via Groth16Prover) ---
    // Note: GPUGroth16ProverEngine uses random blinding each time, so we test that
    // two engines with the same setup both produce valid (but different) proofs.
    do {
        let (r1cs, pubInputs, witness) = buildBenchCircuit(numConstraints: 8)

        let setup = Groth16Setup()
        let (pk, vk) = setup.setup(r1cs: r1cs)

        let engine1 = try GPUGroth16ProverEngine()
        let engine2 = try GPUGroth16ProverEngine()

        let proof1 = try engine1.generateProof(pk: pk, r1cs: r1cs,
                                                publicInputs: pubInputs, witness: witness)
        let proof2 = try engine2.generateProof(pk: pk, r1cs: r1cs,
                                                publicInputs: pubInputs, witness: witness)

        let verifier = Groth16Verifier()
        expect(verifier.verify(proof: proof1, vk: vk, publicInputs: pubInputs),
               "GPU engine determinism: proof1 valid (n=8)")
        expect(verifier.verify(proof: proof2, vk: vk, publicInputs: pubInputs),
               "GPU engine determinism: proof2 valid (n=8)")
    } catch {
        expect(false, "GPU engine determinism error: \(error)")
    }

    // --- Test 6: Version is set ---
    do {
        let v = GPUGroth16ProverEngine.version
        expect(!v.version.isEmpty, "GPU Groth16 Prover Engine version is set")
    }
}
