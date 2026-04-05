// Groth16 Prover end-to-end tests: setup -> prove -> verify round-trip
import zkMetal

func runGroth16ProverTests() {
    suite("Groth16 Prover")

    // --- Pre-test: verify bilinearity of Swift pairing ---
    do {
        let g1 = pointFromAffine(bn254G1Generator())
        let g2 = g2FromAffine(bn254G2Generator())
        let k = frFromInt(7)
        let kG1 = pointScalarMul(g1, k)
        let kG2 = g2ScalarMul(g2, frToInt(k))

        guard let kG1a = pointToAffine(kG1), let g1a = pointToAffine(g1),
              let kG2a = g2ToAffine(kG2), let g2a = g2ToAffine(g2) else {
            expect(false, "Pairing pre-test: affine conversion failed"); return
        }
        // e(kG1, G2) * e(-G1, kG2) = 1
        let check = bn254PairingCheck([(kG1a, g2a), (pointNegateAffine(g1a), kG2a)])
        expect(check, "Pairing bilinearity via Swift pairing")

        // e(2G1, G2) * e(-G1, G2)^2 = 1
        let twoG1 = pointDouble(g1)
        guard let twoG1a = pointToAffine(twoG1) else { expect(false, "2G1 affine"); return }
        let g1neg = pointNegateAffine(g1a)
        let g1bilin = bn254PairingCheck([(twoG1a, g2a), (g1neg, g2a), (g1neg, g2a)])
        expect(g1bilin, "G1 bilinearity e(2G1,G2)*e(-G1,G2)^2=1")

        // --- C pairing bilinearity: e(kG1, G2) * e(-G1, kG2) = 1 ---
        let cCheck = cBN254PairingCheck([(kG1a, g2a), (pointNegateAffine(g1a), kG2a)])
        expect(cCheck, "C pairing bilinearity e(kG1,G2)*e(-G1,kG2)=1")

        // C pairing: e(2G1, G2) * e(-G1, G2)^2 = 1
        let cG1bilin = cBN254PairingCheck([(twoG1a, g2a), (g1neg, g2a), (g1neg, g2a)])
        expect(cG1bilin, "C pairing bilinearity e(2G1,G2)*e(-G1,G2)^2=1")
    }

    // --- Test 1: Classic circuit y = x^2 + x + 5 ---
    // Variables: [1, x, y, v] where v = x*x
    // numPublic = 2 (x and y are public)
    // Constraints:
    //   (0) v = x * x         -> A=[0,1,0,0], B=[0,1,0,0], C=[0,0,0,1]
    //   (1) y = v + x + 5     -> A=[5,1,0,1], B=[1,0,0,0], C=[0,0,1,0]
    do {
        let one = Fr.one
        var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

        // Constraint 0: v = x * x
        aE.append(R1CSEntry(row: 0, col: 1, val: one))
        bE.append(R1CSEntry(row: 0, col: 1, val: one))
        cE.append(R1CSEntry(row: 0, col: 3, val: one))

        // Constraint 1: (5 + x + v) * 1 = y
        aE.append(R1CSEntry(row: 1, col: 0, val: frFromInt(5)))
        aE.append(R1CSEntry(row: 1, col: 1, val: one))
        aE.append(R1CSEntry(row: 1, col: 3, val: one))
        bE.append(R1CSEntry(row: 1, col: 0, val: one))
        cE.append(R1CSEntry(row: 1, col: 2, val: one))

        let r1cs = R1CSInstance(numConstraints: 2, numVars: 4, numPublic: 2,
                                aEntries: aE, bEntries: bE, cEntries: cE)

        // Witness for x=3: v=9, y=3^2+3+5=17
        let x: UInt64 = 3
        let xFr = frFromInt(x)
        let v = frMul(xFr, xFr)
        let y = frAdd(frAdd(v, xFr), frFromInt(5))
        let publicInputs: [Fr] = [xFr, y]
        let witness: [Fr] = [v]

        // Verify R1CS is satisfied
        var z = [Fr](repeating: .zero, count: 4)
        z[0] = .one; z[1] = xFr; z[2] = y; z[3] = v
        expect(r1cs.isSatisfied(z: z), "R1CS satisfied for x=3")

        // Setup
        let setup = Groth16Setup()
        let (pk, vk) = setup.setup(r1cs: r1cs)
        expect(vk.ic.count == 3, "VK IC has numPublic+1 entries")

        // Prove
        let prover = try Groth16Prover()
        let proof = try prover.prove(pk: pk, r1cs: r1cs, publicInputs: publicInputs, witness: witness)

        // Proof elements should be non-identity
        expect(!pointIsIdentity(proof.a), "piA non-identity")
        expect(!pointIsIdentity(proof.c), "piC non-identity")

        // Verify
        let verifier = Groth16Verifier()
        let valid = verifier.verify(proof: proof, vk: vk, publicInputs: publicInputs)
        expect(valid, "Groth16 verify x^2+x+5 (x=3, y=17)")

        // Wrong witness should fail verification
        let wrongY = frFromInt(18)
        let wrongValid = verifier.verify(proof: proof, vk: vk, publicInputs: [xFr, wrongY])
        expect(!wrongValid, "Groth16 reject wrong public input")

        // Different input, same circuit
        let x2: UInt64 = 7
        let x2Fr = frFromInt(x2)
        let v2 = frMul(x2Fr, x2Fr)
        let y2 = frAdd(frAdd(v2, x2Fr), frFromInt(5))
        let pub2: [Fr] = [x2Fr, y2]
        let wit2: [Fr] = [v2]
        let proof2 = try prover.prove(pk: pk, r1cs: r1cs, publicInputs: pub2, witness: wit2)
        let valid2 = verifier.verify(proof: proof2, vk: vk, publicInputs: pub2)
        expect(valid2, "Groth16 verify x^2+x+5 (x=7, y=61)")

    } catch {
        expect(false, "Groth16 x^2+x+5 error: \(error)")
    }

    // --- Test 2: Existing x^3+x+5 example circuit ---
    do {
        let r1cs = buildExampleCircuit()
        let (pubInputs, witness) = computeExampleWitness(x: 3)

        let setup = Groth16Setup()
        let (pk, vk) = setup.setup(r1cs: r1cs)
        let prover = try Groth16Prover()
        let proof = try prover.prove(pk: pk, r1cs: r1cs, publicInputs: pubInputs, witness: witness)

        let verifier = Groth16Verifier()
        expect(verifier.verify(proof: proof, vk: vk, publicInputs: pubInputs),
               "Groth16 verify x^3+x+5 (x=3, y=35)")

        // Wrong witness
        let wrongPub: [Fr] = [frFromInt(3), frFromInt(999)]
        expect(!verifier.verify(proof: proof, vk: vk, publicInputs: wrongPub),
               "Groth16 reject wrong y for x^3+x+5")

    } catch {
        expect(false, "Groth16 x^3+x+5 error: \(error)")
    }

    // --- Test 3: Bench circuit with more constraints ---
    do {
        let (r1cs, pubInputs, witness) = buildBenchCircuit(numConstraints: 8)

        let setup = Groth16Setup()
        let (pk, vk) = setup.setup(r1cs: r1cs)
        let prover = try Groth16Prover()
        let proof = try prover.prove(pk: pk, r1cs: r1cs, publicInputs: pubInputs, witness: witness)

        let verifier = Groth16Verifier()
        expect(verifier.verify(proof: proof, vk: vk, publicInputs: pubInputs),
               "Groth16 verify bench circuit (8 constraints)")

    } catch {
        expect(false, "Groth16 bench circuit error: \(error)")
    }

    // --- Test 4: GPU witness generation ---
    do {
        let (r1cs, pubInputs, expectedWitness) = buildBenchCircuit(numConstraints: 8)
        let prover = try Groth16Prover()

        // Generate witness from just public inputs
        let genZ = prover.generateWitness(r1cs: r1cs, publicInputs: pubInputs)
        expect(r1cs.isSatisfied(z: genZ), "GPU witness gen R1CS satisfied (n=8)")

        // Check the generated witness matches expected
        let nP = r1cs.numPublic
        var match = true
        for i in 0..<expectedWitness.count {
            if !frEq(genZ[1 + nP + i], expectedWitness[i]) { match = false; break }
        }
        expect(match, "GPU witness gen matches expected witness (n=8)")

    } catch {
        expect(false, "GPU witness gen error: \(error)")
    }

    // --- Test 5: Witness generation for larger circuits ---
    for testN in [16, 64, 256] {
        do {
            let (r1cs, pubInputs, expectedWitness) = buildBenchCircuit(numConstraints: testN)
            let prover = try Groth16Prover()
            let genZ = prover.generateWitness(r1cs: r1cs, publicInputs: pubInputs)
            let satisfied = r1cs.isSatisfied(z: genZ)
            expect(satisfied, "Witness gen R1CS satisfied (n=\(testN))")

            // Verify witness values match
            let nP = r1cs.numPublic
            var match = true
            for i in 0..<expectedWitness.count {
                if !frEq(genZ[1 + nP + i], expectedWitness[i]) { match = false; break }
            }
            expect(match, "Witness gen values match (n=\(testN))")
        } catch {
            expect(false, "Witness gen n=\(testN) error: \(error)")
        }
    }

    // --- Test 6: proveWithWitnessGen round-trip ---
    do {
        let (r1cs, pubInputs, _) = buildBenchCircuit(numConstraints: 8)
        let setup = Groth16Setup()
        let (pk, vk) = setup.setup(r1cs: r1cs)
        let prover = try Groth16Prover()
        let proof = try prover.proveWithWitnessGen(pk: pk, r1cs: r1cs, publicInputs: pubInputs)

        // Proof elements should be non-identity
        expect(!pointIsIdentity(proof.a), "proveWithWitnessGen piA non-identity")
        expect(!pointIsIdentity(proof.c), "proveWithWitnessGen piC non-identity")

        // Verify (note: may fail due to pre-existing C pairing issues)
        let verifier = Groth16Verifier()
        let valid = verifier.verify(proof: proof, vk: vk, publicInputs: pubInputs)
        expect(valid, "proveWithWitnessGen verify (n=8)")
    } catch {
        expect(false, "proveWithWitnessGen error: \(error)")
    }
}
