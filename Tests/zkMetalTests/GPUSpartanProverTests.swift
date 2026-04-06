// GPUSpartanProverTests — Tests for GPU-accelerated Spartan prover engine
import zkMetal
import Foundation

public func runGPUSpartanProverTests() {
    suite("GPUSpartanProver")

    // --- Helper: create IPA-backed GPU Spartan prover for a given paddedN ---
    func makeIPAProver(paddedN: Int) throws -> GPUSpartanProverEngine<IPAPCSAdapter> {
        let g = bn254G1Generator()
        var generators = [PointAffine]()
        generators.reserveCapacity(paddedN)
        for i in 0..<paddedN {
            let scalar = frFromInt(UInt64(i + 1) * 13 + 19)
            let pt = cPointScalarMul(pointFromAffine(g), scalar)
            if let aff = pointToAffine(pt) { generators.append(aff) }
        }
        let qScalar = frFromInt(UInt64(paddedN + 1) * 13 + 19)
        guard let qAff = pointToAffine(cPointScalarMul(pointFromAffine(g), qScalar)) else {
            throw NSError(domain: "GPUSpartanProverTests", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Q gen failed"])
        }
        let ipaEngine = try IPAEngine(generators: generators, Q: qAff)
        let adapter = IPAPCSAdapter(engine: ipaEngine)
        return GPUSpartanProverEngine(pcs: adapter)
    }

    // --- Test 1: Simple circuit proof (x * x = y) ---
    do {
        let b = SpartanR1CSBuilder()
        let y = b.addPublicInput()
        let x = b.addWitness()
        let v = b.addWitness()

        b.mulGate(a: x, b: x, out: v)
        b.addConstraint(a: [(v, Fr.one)], b: [(0, Fr.one)], c: [(y, Fr.one)])

        let instance = b.build()

        let xVal = frFromInt(3)
        let vVal = frMul(xVal, xVal)
        let yVal = vVal
        let publicInputs: [Fr] = [yVal]
        let witness: [Fr] = [xVal, vVal]

        let prover = try makeIPAProver(paddedN: instance.paddedN)
        let proof = try prover.prove(instance: instance, publicInputs: publicInputs, witness: witness)
        let valid = prover.verify(instance: instance, publicInputs: publicInputs, proof: proof)
        expect(valid, "GPU Spartan: simple circuit x*x=y prove+verify (x=3)")

        // Verify proof structure
        expect(proof.sc1Rounds.count == instance.logM,
               "GPU Spartan: SC1 rounds = logM = \(instance.logM)")
        expect(proof.sc2Rounds.count == instance.logN,
               "GPU Spartan: SC2 rounds = logN = \(instance.logN)")
    } catch {
        expect(false, "GPU Spartan simple circuit error: \(error)")
    }

    // --- Test 2: Sumcheck integration (quadratic circuit x^2+x+5=y) ---
    do {
        let (instance, gen) = SpartanR1CSBuilder.exampleQuadratic()
        let xVal = frFromInt(4)
        let (publicInputs, witness) = gen(xVal)

        let z = SpartanR1CS.buildZ(publicInputs: publicInputs, witness: witness)
        expect(instance.isSatisfied(z: z), "GPU Spartan: quadratic R1CS satisfied for x=4")

        let prover = try makeIPAProver(paddedN: instance.paddedN)
        let proof = try prover.prove(instance: instance, publicInputs: publicInputs, witness: witness)
        let valid = prover.verify(instance: instance, publicInputs: publicInputs, proof: proof)
        expect(valid, "GPU Spartan: quadratic circuit prove+verify (x=4, y=25)")

        // Memory-checking digest should be nonzero
        let digestLimbs = frToInt(proof.memoryCheckDigest)
        let digestNonZero = digestLimbs[0] != 0 || digestLimbs[1] != 0 ||
                            digestLimbs[2] != 0 || digestLimbs[3] != 0
        expect(digestNonZero, "GPU Spartan: memory check digest is nonzero")
    } catch {
        expect(false, "GPU Spartan sumcheck integration error: \(error)")
    }

    // --- Test 3: Witness commitment (multiply circuit, IPA commitment) ---
    do {
        let (instance, gen) = SpartanR1CSBuilder.buildMultiplyCircuit()
        let xVal = frFromInt(5)
        let yVal = frFromInt(7)
        let (publicInputs, witness) = gen(xVal, yVal)

        let prover = try makeIPAProver(paddedN: instance.paddedN)
        let proof = try prover.prove(instance: instance, publicInputs: publicInputs, witness: witness)

        // Valid proof should pass
        let valid = prover.verify(instance: instance, publicInputs: publicInputs, proof: proof)
        expect(valid, "GPU Spartan: witness commitment via IPA prove+verify (5*7=35)")

        // Prove with different witness to ensure commitment changes
        let (pub2, wit2) = gen(frFromInt(2), frFromInt(3))
        let proof2 = try prover.prove(instance: instance, publicInputs: pub2, witness: wit2)
        let valid2 = prover.verify(instance: instance, publicInputs: pub2, proof: proof2)
        expect(valid2, "GPU Spartan: second witness commitment prove+verify (2*3=6)")

        // Commitments should differ for different witnesses
        let tag1 = proof.witnessCommitment.transcriptTag
        let tag2 = proof2.witnessCommitment.transcriptTag
        expect(!spartanFrEqual(tag1, tag2),
               "GPU Spartan: different witnesses produce different commitments")
    } catch {
        expect(false, "GPU Spartan witness commitment error: \(error)")
    }

    // --- Test 4: Verification pass (medium circuit, 16 constraints) ---
    do {
        let (instance, publicInputs, witness) = SpartanR1CSBuilder.syntheticR1CS(numConstraints: 16)
        let z = SpartanR1CS.buildZ(publicInputs: publicInputs, witness: witness)
        expect(instance.isSatisfied(z: z), "GPU Spartan: synthetic R1CS (16) satisfied")

        let prover = try makeIPAProver(paddedN: instance.paddedN)
        let proof = try prover.prove(instance: instance, publicInputs: publicInputs, witness: witness)
        let valid = prover.verify(instance: instance, publicInputs: publicInputs, proof: proof)
        expect(valid, "GPU Spartan: synthetic 16-constraint prove+verify")
    } catch {
        expect(false, "GPU Spartan 16-constraint error: \(error)")
    }

    // --- Test 5: Wrong witness rejection ---
    do {
        let (instance, gen) = SpartanR1CSBuilder.exampleQuadratic()
        let xVal = frFromInt(5)
        let (publicInputs, witness) = gen(xVal)

        let prover = try makeIPAProver(paddedN: instance.paddedN)
        let proof = try prover.prove(instance: instance, publicInputs: publicInputs, witness: witness)

        // Valid proof should pass
        let valid = prover.verify(instance: instance, publicInputs: publicInputs, proof: proof)
        expect(valid, "GPU Spartan: valid proof accepted before rejection test")

        // Wrong public input should be rejected
        let wrongPub: [Fr] = [frFromInt(999)]
        let rejected1 = prover.verify(instance: instance, publicInputs: wrongPub, proof: proof)
        expect(!rejected1, "GPU Spartan: rejects proof with wrong public input")

        // Tampered azRx should be rejected
        let forgedProof1 = GPUSpartanProverProof<IPAPCSAdapter>(
            witnessCommitment: proof.witnessCommitment,
            sc1Rounds: proof.sc1Rounds,
            azRx: frAdd(proof.azRx, Fr.one),  // tampered
            bzRx: proof.bzRx,
            czRx: proof.czRx,
            sc2Rounds: proof.sc2Rounds,
            zEval: proof.zEval,
            openingProof: proof.openingProof,
            memoryCheckDigest: proof.memoryCheckDigest)
        let rejected2 = prover.verify(instance: instance, publicInputs: publicInputs, proof: forgedProof1)
        expect(!rejected2, "GPU Spartan: rejects forged azRx claim")

        // Tampered zEval should be rejected
        let forgedProof2 = GPUSpartanProverProof<IPAPCSAdapter>(
            witnessCommitment: proof.witnessCommitment,
            sc1Rounds: proof.sc1Rounds,
            azRx: proof.azRx,
            bzRx: proof.bzRx,
            czRx: proof.czRx,
            sc2Rounds: proof.sc2Rounds,
            zEval: frAdd(proof.zEval, Fr.one),  // tampered
            openingProof: proof.openingProof,
            memoryCheckDigest: proof.memoryCheckDigest)
        let rejected3 = prover.verify(instance: instance, publicInputs: publicInputs, proof: forgedProof2)
        expect(!rejected3, "GPU Spartan: rejects forged zEval claim")

        // Tampered SC1 round should be rejected
        var badRounds = proof.sc1Rounds
        if !badRounds.isEmpty {
            let (r0, r1, r2, r3) = badRounds[0]
            badRounds[0] = (frAdd(r0, Fr.one), r1, r2, r3)
        }
        let forgedProof3 = GPUSpartanProverProof<IPAPCSAdapter>(
            witnessCommitment: proof.witnessCommitment,
            sc1Rounds: badRounds,
            azRx: proof.azRx,
            bzRx: proof.bzRx,
            czRx: proof.czRx,
            sc2Rounds: proof.sc2Rounds,
            zEval: proof.zEval,
            openingProof: proof.openingProof,
            memoryCheckDigest: proof.memoryCheckDigest)
        let rejected4 = prover.verify(instance: instance, publicInputs: publicInputs, proof: forgedProof3)
        expect(!rejected4, "GPU Spartan: rejects forged SC1 round poly")
    } catch {
        expect(false, "GPU Spartan wrong witness rejection error: \(error)")
    }

    // --- Test 6: Repeated prove calls (tests buffer caching) ---
    do {
        let (instance, gen) = SpartanR1CSBuilder.exampleQuadratic()
        let prover = try makeIPAProver(paddedN: instance.paddedN)

        for x in [2, 7, 13] as [UInt64] {
            let xVal = frFromInt(x)
            let (pub, wit) = gen(xVal)
            let proof = try prover.prove(instance: instance, publicInputs: pub, witness: wit)
            let valid = prover.verify(instance: instance, publicInputs: pub, proof: proof)
            expect(valid, "GPU Spartan: repeated prove x=\(x)")
        }
    } catch {
        expect(false, "GPU Spartan repeated prove error: \(error)")
    }

    // --- Test 7: GPU availability check ---
    do {
        let (instance, _) = SpartanR1CSBuilder.buildMultiplyCircuit()
        let prover = try makeIPAProver(paddedN: instance.paddedN)
        // On macOS with Metal, hasGPU should be true
        // On CI without GPU, hasGPU may be false but engine still works via CPU fallback
        expect(true, "GPU Spartan: engine initialized successfully (hasGPU=\(prover.hasGPU))")
    } catch {
        expect(false, "GPU Spartan init error: \(error)")
    }

    // --- Test 8: Version check ---
    do {
        let version = Versions.gpuSpartanProver
        expect(version.version == "1.0.0", "GPU Spartan: version is 1.0.0")
        expect(version.updated == "2026-04-05", "GPU Spartan: updated date correct")
    }
}
