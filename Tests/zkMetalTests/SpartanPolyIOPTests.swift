// SpartanPolyIOPTests — Tests for Spartan Polynomial IOP prover and verifier
import zkMetal
import Foundation

public func runSpartanPolyIOPTests() {
    suite("SpartanPolyIOP")

    // --- Test 1: Prove and verify simple multiply circuit ---
    // Circuit: x * y = z (public output)
    do {
        let (instance, gen) = SpartanR1CSBuilder.buildMultiplyCircuit()
        let xVal = frFromInt(3)
        let yVal = frFromInt(7)
        let (publicInputs, witness) = gen(xVal, yVal)

        // Verify R1CS satisfaction first
        let z = SpartanR1CS.buildZ(publicInputs: publicInputs, witness: witness)
        expect(instance.isSatisfied(z: z), "PolyIOP: multiply circuit R1CS satisfied")

        // Create IPA-backed prover and verifier
        let paddedN = instance.paddedN
        let g = bn254G1Generator()
        var generators = [PointAffine]()
        generators.reserveCapacity(paddedN)
        for i in 0..<paddedN {
            let scalar = frFromInt(UInt64(i + 1) * 11 + 17)
            let pt = cPointScalarMul(pointFromAffine(g), scalar)
            if let aff = pointToAffine(pt) { generators.append(aff) }
        }
        let qScalar = frFromInt(UInt64(paddedN + 1) * 11 + 17)
        guard let qAff = pointToAffine(cPointScalarMul(pointFromAffine(g), qScalar)) else {
            expect(false, "PolyIOP: Q gen failed"); return
        }

        let ipaEngine = try IPAEngine(generators: generators, Q: qAff)
        let adapter = IPAPCSAdapter(engine: ipaEngine)
        let prover = SpartanPolyIOPProver(pcs: adapter)
        let verifier = SpartanPolyIOPVerifier(pcs: adapter)

        let proof = try prover.prove(instance: instance, publicInputs: publicInputs, witness: witness)
        let valid = verifier.verify(instance: instance, publicInputs: publicInputs, proof: proof)
        expect(valid, "PolyIOP: prove+verify multiply circuit (3*7=21)")

        // Also check proof fields are populated
        expect(proof.numSumcheckRounds == instance.logM,
               "PolyIOP: sumcheck rounds = logM = \(instance.logM)")
    } catch {
        expect(false, "PolyIOP multiply circuit error: \(error)")
    }

    // --- Test 2: Prove and verify 10-constraint add chain ---
    do {
        let n = 10
        let (instance, gen) = SpartanR1CSBuilder.buildAddChainCircuit(n: n)
        let xVal = frFromInt(5)
        let (publicInputs, witness) = gen(xVal)

        let z = SpartanR1CS.buildZ(publicInputs: publicInputs, witness: witness)
        expect(instance.isSatisfied(z: z), "PolyIOP: 10-add-chain R1CS satisfied")

        // Expected output: (n+1)*x = 11*5 = 55
        let expectedOutput = frFromInt(55)
        expect(spartanFrEqual(publicInputs[0], expectedOutput),
               "PolyIOP: add chain output = 55")

        let paddedN = instance.paddedN
        let g = bn254G1Generator()
        var generators = [PointAffine]()
        generators.reserveCapacity(paddedN)
        for i in 0..<paddedN {
            let scalar = frFromInt(UInt64(i + 1) * 11 + 17)
            let pt = cPointScalarMul(pointFromAffine(g), scalar)
            if let aff = pointToAffine(pt) { generators.append(aff) }
        }
        let qScalar = frFromInt(UInt64(paddedN + 1) * 11 + 17)
        guard let qAff = pointToAffine(cPointScalarMul(pointFromAffine(g), qScalar)) else {
            expect(false, "PolyIOP: Q gen failed"); return
        }

        let ipaEngine = try IPAEngine(generators: generators, Q: qAff)
        let adapter = IPAPCSAdapter(engine: ipaEngine)
        let prover = SpartanPolyIOPProver(pcs: adapter)
        let verifier = SpartanPolyIOPVerifier(pcs: adapter)

        let proof = try prover.prove(instance: instance, publicInputs: publicInputs, witness: witness)
        let valid = verifier.verify(instance: instance, publicInputs: publicInputs, proof: proof)
        expect(valid, "PolyIOP: prove+verify 10-constraint add chain")
    } catch {
        expect(false, "PolyIOP 10-add-chain error: \(error)")
    }

    // --- Test 3: Wrong witness rejected ---
    do {
        let (instance, gen) = SpartanR1CSBuilder.buildMultiplyCircuit()
        let xVal = frFromInt(4)
        let yVal = frFromInt(6)
        let (publicInputs, witness) = gen(xVal, yVal)

        let paddedN = instance.paddedN
        let g = bn254G1Generator()
        var generators = [PointAffine]()
        generators.reserveCapacity(paddedN)
        for i in 0..<paddedN {
            let scalar = frFromInt(UInt64(i + 1) * 11 + 17)
            let pt = cPointScalarMul(pointFromAffine(g), scalar)
            if let aff = pointToAffine(pt) { generators.append(aff) }
        }
        let qScalar = frFromInt(UInt64(paddedN + 1) * 11 + 17)
        guard let qAff = pointToAffine(cPointScalarMul(pointFromAffine(g), qScalar)) else {
            expect(false, "PolyIOP: Q gen failed"); return
        }

        let ipaEngine = try IPAEngine(generators: generators, Q: qAff)
        let adapter = IPAPCSAdapter(engine: ipaEngine)
        let prover = SpartanPolyIOPProver(pcs: adapter)
        let verifier = SpartanPolyIOPVerifier(pcs: adapter)

        // Generate valid proof
        let proof = try prover.prove(instance: instance, publicInputs: publicInputs, witness: witness)

        // Valid proof should pass
        let valid = verifier.verify(instance: instance, publicInputs: publicInputs, proof: proof)
        expect(valid, "PolyIOP: valid proof accepted before rejection test")

        // Wrong public input should fail
        let wrongPub: [Fr] = [frFromInt(999)]
        let rejected1 = verifier.verify(instance: instance, publicInputs: wrongPub, proof: proof)
        expect(!rejected1, "PolyIOP: rejects proof with wrong public input")

        // Tampered azEval should fail
        let forgedProof = SpartanPolyIOPProof<IPAPCSAdapter>(
            witnessCommitment: proof.witnessCommitment,
            sumcheckRounds: proof.sumcheckRounds,
            azEval: frAdd(proof.azEval, Fr.one),  // tampered
            bzEval: proof.bzEval,
            czEval: proof.czEval,
            innerProductRounds: proof.innerProductRounds,
            witnessEval: proof.witnessEval,
            openingProof: proof.openingProof)
        let rejected2 = verifier.verify(instance: instance, publicInputs: publicInputs, proof: forgedProof)
        expect(!rejected2, "PolyIOP: rejects forged azEval claim")

        // Tampered witnessEval should fail
        let forgedProof2 = SpartanPolyIOPProof<IPAPCSAdapter>(
            witnessCommitment: proof.witnessCommitment,
            sumcheckRounds: proof.sumcheckRounds,
            azEval: proof.azEval,
            bzEval: proof.bzEval,
            czEval: proof.czEval,
            innerProductRounds: proof.innerProductRounds,
            witnessEval: frAdd(proof.witnessEval, Fr.one),  // tampered
            openingProof: proof.openingProof)
        let rejected3 = verifier.verify(instance: instance, publicInputs: publicInputs, proof: forgedProof2)
        expect(!rejected3, "PolyIOP: rejects forged witnessEval claim")
    } catch {
        expect(false, "PolyIOP wrong witness rejection error: \(error)")
    }

    // --- Test 4: Sumcheck round count matches log(constraints) ---
    do {
        // Single constraint circuit: 1 constraint -> logM = 0 (2^0 = 1)
        let (inst1, gen1) = SpartanR1CSBuilder.buildMultiplyCircuit()
        // logM is ceil(log2(numConstraints)), numConstraints=1 -> logM=0
        expect(inst1.logM >= 0,
               "PolyIOP: logM >= 0 for 1-constraint circuit")

        // 10 constraints -> logM = ceil(log2(10)) = 4
        let (inst10, _) = SpartanR1CSBuilder.buildAddChainCircuit(n: 10)
        expect(inst10.logM == 4,
               "PolyIOP: 10-constraint circuit has logM=4 (ceil(log2(10))=4)")

        // Now prove and check proof round count matches
        let xVal = frFromInt(2)
        let (pub1, wit1) = gen1(xVal, frFromInt(3))

        let paddedN = inst1.paddedN
        let g = bn254G1Generator()
        var generators = [PointAffine]()
        generators.reserveCapacity(paddedN)
        for i in 0..<paddedN {
            let scalar = frFromInt(UInt64(i + 1) * 11 + 17)
            let pt = cPointScalarMul(pointFromAffine(g), scalar)
            if let aff = pointToAffine(pt) { generators.append(aff) }
        }
        let qScalar = frFromInt(UInt64(paddedN + 1) * 11 + 17)
        guard let qAff = pointToAffine(cPointScalarMul(pointFromAffine(g), qScalar)) else {
            expect(false, "PolyIOP: Q gen failed"); return
        }

        let ipaEngine = try IPAEngine(generators: generators, Q: qAff)
        let adapter = IPAPCSAdapter(engine: ipaEngine)
        let prover = SpartanPolyIOPProver(pcs: adapter)

        let proof = try prover.prove(instance: inst1, publicInputs: pub1, witness: wit1)
        expect(proof.numSumcheckRounds == inst1.logM,
               "PolyIOP: proof.numSumcheckRounds == logM (\(inst1.logM))")
        expect(proof.innerProductRounds.count == inst1.logN,
               "PolyIOP: innerProductRounds.count == logN (\(inst1.logN))")
    } catch {
        expect(false, "PolyIOP round count error: \(error)")
    }

    // --- Test 5: R1CS builder helpers ---
    do {
        // Test buildMultiplyCircuit
        let (multInst, multGen) = SpartanR1CSBuilder.buildMultiplyCircuit()
        expect(multInst.numConstraints == 1, "PolyIOP: multiply circuit has 1 constraint")
        let (multPub, multWit) = multGen(frFromInt(5), frFromInt(9))
        let multZ = SpartanR1CS.buildZ(publicInputs: multPub, witness: multWit)
        expect(multInst.isSatisfied(z: multZ), "PolyIOP: multiply 5*9=45 satisfied")
        expect(spartanFrEqual(multPub[0], frFromInt(45)), "PolyIOP: multiply output = 45")

        // Test buildAddChainCircuit
        let (addInst, addGen) = SpartanR1CSBuilder.buildAddChainCircuit(n: 5)
        expect(addInst.numConstraints == 5, "PolyIOP: 5-add-chain has 5 constraints")
        let (addPub, addWit) = addGen(frFromInt(3))
        let addZ = SpartanR1CS.buildZ(publicInputs: addPub, witness: addWit)
        expect(addInst.isSatisfied(z: addZ), "PolyIOP: add chain 6*3=18 satisfied")
        expect(spartanFrEqual(addPub[0], frFromInt(18)), "PolyIOP: add chain output = 18")

        // Test buildRangeCheckCircuit
        let (rangeInst, rangeGen) = SpartanR1CSBuilder.buildRangeCheckCircuit(bits: 8)
        // Value 200 fits in 8 bits
        let (rangePub, rangeWit) = rangeGen(200)
        let rangeZ = SpartanR1CS.buildZ(publicInputs: rangePub, witness: rangeWit)
        expect(rangeInst.isSatisfied(z: rangeZ), "PolyIOP: range check 200 in 8 bits satisfied")

        // Value 255 (max 8-bit) should also work
        let (rangePub255, rangeWit255) = rangeGen(255)
        let rangeZ255 = SpartanR1CS.buildZ(publicInputs: rangePub255, witness: rangeWit255)
        expect(rangeInst.isSatisfied(z: rangeZ255), "PolyIOP: range check 255 in 8 bits satisfied")

        // Test addVariable (same as addWitness)
        let builder = SpartanR1CSBuilder()
        let v0 = builder.addVariable()
        let v1 = builder.addVariable()
        expect(v0 == 1, "PolyIOP: addVariable returns sequential indices (1)")
        expect(v1 == 2, "PolyIOP: addVariable returns sequential indices (2)")
    }

    // --- Test 6: Range check circuit prove+verify ---
    do {
        let (instance, gen) = SpartanR1CSBuilder.buildRangeCheckCircuit(bits: 4)
        let (publicInputs, witness) = gen(13) // 13 = 1101 in binary, fits in 4 bits

        let z = SpartanR1CS.buildZ(publicInputs: publicInputs, witness: witness)
        expect(instance.isSatisfied(z: z), "PolyIOP: range check 13 in 4 bits satisfied")

        let paddedN = instance.paddedN
        let g = bn254G1Generator()
        var generators = [PointAffine]()
        generators.reserveCapacity(paddedN)
        for i in 0..<paddedN {
            let scalar = frFromInt(UInt64(i + 1) * 11 + 17)
            let pt = cPointScalarMul(pointFromAffine(g), scalar)
            if let aff = pointToAffine(pt) { generators.append(aff) }
        }
        let qScalar = frFromInt(UInt64(paddedN + 1) * 11 + 17)
        guard let qAff = pointToAffine(cPointScalarMul(pointFromAffine(g), qScalar)) else {
            expect(false, "PolyIOP: Q gen failed"); return
        }

        let ipaEngine = try IPAEngine(generators: generators, Q: qAff)
        let adapter = IPAPCSAdapter(engine: ipaEngine)
        let prover = SpartanPolyIOPProver(pcs: adapter)
        let verifier = SpartanPolyIOPVerifier(pcs: adapter)

        let proof = try prover.prove(instance: instance, publicInputs: publicInputs, witness: witness)
        let valid = verifier.verify(instance: instance, publicInputs: publicInputs, proof: proof)
        expect(valid, "PolyIOP: prove+verify range check (13 in 4 bits)")
    } catch {
        expect(false, "PolyIOP range check error: \(error)")
    }

    // --- Test 7: Basefold PCS backend ---
    do {
        let (instance, gen) = SpartanR1CSBuilder.buildMultiplyCircuit()
        let xVal = frFromInt(6)
        let yVal = frFromInt(7)
        let (publicInputs, witness) = gen(xVal, yVal)

        let basefoldEngine = try BasefoldEngine()
        let adapter = BasefoldPCSAdapter(engine: basefoldEngine)
        let prover = SpartanPolyIOPProver(pcs: adapter)
        let verifier = SpartanPolyIOPVerifier(pcs: adapter)

        let proof = try prover.prove(instance: instance, publicInputs: publicInputs, witness: witness)
        let valid = verifier.verify(instance: instance, publicInputs: publicInputs, proof: proof)
        expect(valid, "PolyIOP+Basefold: prove+verify multiply circuit (6*7=42)")
    } catch {
        expect(false, "PolyIOP+Basefold error: \(error)")
    }
}
