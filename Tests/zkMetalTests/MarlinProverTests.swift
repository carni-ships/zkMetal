// MarlinProver end-to-end tests: index -> prove -> verify round-trip
import zkMetal

public func runMarlinProverTests() {
    suite("Marlin Prover")

    // Shared SRS setup
    let gen = bn254G1Generator()
    let secret: [UInt32] = [0xCAFE, 0xBEEF, 0xDEAD, 0x1234, 0x5678, 0x9ABC, 0xDEF0, 0x0001]
    let srsSecret = frFromLimbs(secret)
    let srs = KZGEngine.generateTestSRS(secret: secret, size: 512, generator: gen)

    // --- Test 1: Simple multiply circuit ---
    // R1CS: a * b = c (one constraint, one public output)
    // Variables: [1, c, a, b] where numPublic=1 (c is public)
    // Constraint: A=[0,0,1,0] B=[0,0,0,1] C=[0,1,0,0]
    do {
        let one = Fr.one
        var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

        // Constraint 0: a * b = c
        aE.append(R1CSEntry(row: 0, col: 2, val: one))
        bE.append(R1CSEntry(row: 0, col: 3, val: one))
        cE.append(R1CSEntry(row: 0, col: 1, val: one))

        let r1cs = R1CSInstance(numConstraints: 1, numVars: 4, numPublic: 1,
                                aEntries: aE, bEntries: bE, cEntries: cE)

        // Witness: a=3, b=5, c=15
        let aVal = frFromInt(3)
        let bVal = frFromInt(5)
        let cVal = frMul(aVal, bVal) // 15

        let publicInputs: [Fr] = [cVal]
        let witness: [Fr] = [aVal, bVal]

        // Verify R1CS is satisfied
        var z = [Fr](repeating: .zero, count: 4)
        z[0] = .one; z[1] = cVal; z[2] = aVal; z[3] = bVal
        expect(r1cs.isSatisfied(z: z), "Multiply R1CS satisfied")

        // Index + Prove + Verify via MarlinProver
        let kzg = try KZGEngine(srs: srs)
        let ntt = try NTTEngine()
        let prover = MarlinProver(kzg: kzg, ntt: ntt)
        let verifier = MarlinVerifier(kzg: kzg)

        let (pk, vk) = try prover.index(r1cs: r1cs, srsSecret: srsSecret)
        expect(pk.indexPolynomials.count == 12, "12 index polynomials (4 per matrix)")
        expect(pk.indexCommitments.count == 12, "12 index commitments")

        let proof = try prover.prove(r1cs: r1cs, publicInputs: publicInputs,
                                      witness: witness, pk: pk)

        // Verify the proof
        let valid = verifier.verify(vk: vk, publicInput: publicInputs, proof: proof)
        expect(valid, "Multiply circuit proof verifies")

        // Diagnostic check
        let diag = verifier.verifyDiag(vk: vk, publicInput: publicInputs, proof: proof)
        expect(diag == "PASS", "Multiply circuit diagnostic: \(diag)")

    } catch {
        expect(false, "Multiply circuit test threw: \(error)")
    }

    // --- Test 2: 10-constraint circuit ---
    // Each constraint i: x_i * x_i = y_i (square check)
    // Variables: [1, y_0, ..., y_9, x_0, ..., x_9] (numVars = 21, numPublic = 10)
    do {
        let one = Fr.one
        let numCons = 10
        var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

        for i in 0..<numCons {
            // Constraint i: x_i * x_i = y_i
            // x_i is at index 1 + numCons + i, y_i is at index 1 + i
            aE.append(R1CSEntry(row: i, col: 1 + numCons + i, val: one))
            bE.append(R1CSEntry(row: i, col: 1 + numCons + i, val: one))
            cE.append(R1CSEntry(row: i, col: 1 + i, val: one))
        }

        let r1cs = R1CSInstance(numConstraints: numCons, numVars: 1 + 2 * numCons,
                                numPublic: numCons,
                                aEntries: aE, bEntries: bE, cEntries: cE)

        // Witness: x_i = i+2, y_i = (i+2)^2
        var publicInputs = [Fr]()
        var witness = [Fr]()
        for i in 0..<numCons {
            let x = frFromInt(UInt64(i + 2))
            let y = frMul(x, x)
            publicInputs.append(y)
            witness.append(x)
        }

        // Verify satisfaction
        var z = [Fr](repeating: .zero, count: 1 + 2 * numCons)
        z[0] = .one
        for i in 0..<numCons { z[1 + i] = publicInputs[i] }
        for i in 0..<numCons { z[1 + numCons + i] = witness[i] }
        expect(r1cs.isSatisfied(z: z), "10-constraint R1CS satisfied")

        let kzg = try KZGEngine(srs: srs)
        let ntt = try NTTEngine()
        let prover = MarlinProver(kzg: kzg, ntt: ntt)
        let verifier = MarlinVerifier(kzg: kzg)

        let (pk, vk) = try prover.index(r1cs: r1cs, srsSecret: srsSecret)
        let proof = try prover.prove(r1cs: r1cs, publicInputs: publicInputs,
                                      witness: witness, pk: pk)

        let valid = verifier.verify(vk: vk, publicInput: publicInputs, proof: proof)
        expect(valid, "10-constraint circuit proof verifies")

        let diag = verifier.verifyDiag(vk: vk, publicInput: publicInputs, proof: proof)
        expect(diag == "PASS", "10-constraint diagnostic: \(diag)")

    } catch {
        expect(false, "10-constraint test threw: \(error)")
    }

    // --- Test 3: Wrong witness rejected ---
    // Same multiply circuit but with incorrect witness
    do {
        let one = Fr.one
        var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

        aE.append(R1CSEntry(row: 0, col: 2, val: one))
        bE.append(R1CSEntry(row: 0, col: 3, val: one))
        cE.append(R1CSEntry(row: 0, col: 1, val: one))

        let r1cs = R1CSInstance(numConstraints: 1, numVars: 4, numPublic: 1,
                                aEntries: aE, bEntries: bE, cEntries: cE)

        // Claim c=15 but provide wrong witness a=3, b=4 (3*4=12, not 15)
        let cVal = frFromInt(15)
        let aVal = frFromInt(3)
        let bVal = frFromInt(4) // Wrong! 3*4=12, not 15

        let publicInputs: [Fr] = [cVal]
        let witness: [Fr] = [aVal, bVal]

        // Confirm R1CS is NOT satisfied
        var z = [Fr](repeating: .zero, count: 4)
        z[0] = .one; z[1] = cVal; z[2] = aVal; z[3] = bVal
        expect(!r1cs.isSatisfied(z: z), "Wrong witness does not satisfy R1CS")

        let kzg = try KZGEngine(srs: srs)
        let ntt = try NTTEngine()
        let prover = MarlinProver(kzg: kzg, ntt: ntt)
        let verifier = MarlinVerifier(kzg: kzg)

        let (pk, vk) = try prover.index(r1cs: r1cs, srsSecret: srsSecret)
        let proof = try prover.prove(r1cs: r1cs, publicInputs: publicInputs,
                                      witness: witness, pk: pk)

        // This proof should NOT verify (bad witness => bad quotient polynomial)
        let valid = verifier.verify(vk: vk, publicInput: publicInputs, proof: proof)
        expect(!valid, "Wrong witness proof is rejected")

    } catch {
        // Throwing during prove with bad witness is also acceptable rejection
        expect(true, "Wrong witness rejected via exception")
    }

    // --- Test 4: Index reuse - prove multiple instances with same VK ---
    // Same circuit structure, different witness values
    do {
        let one = Fr.one
        var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

        // Circuit: a * b = c
        aE.append(R1CSEntry(row: 0, col: 2, val: one))
        bE.append(R1CSEntry(row: 0, col: 3, val: one))
        cE.append(R1CSEntry(row: 0, col: 1, val: one))

        let r1cs = R1CSInstance(numConstraints: 1, numVars: 4, numPublic: 1,
                                aEntries: aE, bEntries: bE, cEntries: cE)

        let kzg = try KZGEngine(srs: srs)
        let ntt = try NTTEngine()
        let prover = MarlinProver(kzg: kzg, ntt: ntt)
        let verifier = MarlinVerifier(kzg: kzg)

        // Index once
        let (pk, vk) = try prover.index(r1cs: r1cs, srsSecret: srsSecret)

        // Instance 1: 3 * 7 = 21
        let a1 = frFromInt(3), b1 = frFromInt(7)
        let c1 = frMul(a1, b1)
        let proof1 = try prover.prove(r1cs: r1cs, publicInputs: [c1],
                                       witness: [a1, b1], pk: pk)
        let valid1 = verifier.verify(vk: vk, publicInput: [c1], proof: proof1)
        expect(valid1, "Index reuse: instance 1 (3*7=21) verifies")

        // Instance 2: 5 * 11 = 55
        let a2 = frFromInt(5), b2 = frFromInt(11)
        let c2 = frMul(a2, b2)
        let proof2 = try prover.prove(r1cs: r1cs, publicInputs: [c2],
                                       witness: [a2, b2], pk: pk)
        let valid2 = verifier.verify(vk: vk, publicInput: [c2], proof: proof2)
        expect(valid2, "Index reuse: instance 2 (5*11=55) verifies")

        // Instance 3: 100 * 200 = 20000
        let a3 = frFromInt(100), b3 = frFromInt(200)
        let c3 = frMul(a3, b3)
        let proof3 = try prover.prove(r1cs: r1cs, publicInputs: [c3],
                                       witness: [a3, b3], pk: pk)
        let valid3 = verifier.verify(vk: vk, publicInput: [c3], proof: proof3)
        expect(valid3, "Index reuse: instance 3 (100*200=20000) verifies")

        // Cross-check: proof1 should NOT verify against instance 2's public input
        let crossValid = verifier.verify(vk: vk, publicInput: [c2], proof: proof1)
        expect(!crossValid, "Index reuse: cross-instance proof rejected")

    } catch {
        expect(false, "Index reuse test threw: \(error)")
    }
}
