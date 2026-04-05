// Spartan proving system tests: R1CS, sumcheck, prove/verify round-trips
import zkMetal
import Foundation

func runSpartanTests() {
    suite("Spartan")

    // --- Test 1: Small R1CS (x * x = y, 2 constraints) prove + verify ---
    // Circuit: constraint 0: x * x = v  (multiplication gate)
    //          constraint 1: v * 1 = y  (copy gate, expressed as multiplication)
    // Public input: y.  Witness: x, v.
    do {
        let b = SpartanR1CSBuilder()
        let y = b.addPublicInput()   // var 1
        let x = b.addWitness()       // var 2
        let v = b.addWitness()       // var 3

        // Constraint 0: x * x = v
        b.mulGate(a: x, b: x, out: v)
        // Constraint 1: v * 1 = y
        b.addConstraint(a: [(v, Fr.one)], b: [(0, Fr.one)], c: [(y, Fr.one)])

        let instance = b.build()

        // Witness for x = 3: v = 9, y = 9
        let xVal = frFromInt(3)
        let vVal = frMul(xVal, xVal)         // 9
        let yVal = vVal
        let publicInputs: [Fr] = [yVal]
        let witness: [Fr] = [xVal, vVal]

        // Verify R1CS satisfaction
        let z = SpartanR1CS.buildZ(publicInputs: publicInputs, witness: witness)
        expect(instance.isSatisfied(z: z), "Small R1CS (x*x=y) satisfied")

        // Prove and verify
        let engine = try SpartanEngine()
        let proof = try engine.prove(instance: instance, publicInputs: publicInputs, witness: witness)
        let valid = engine.verify(instance: instance, publicInputs: publicInputs, proof: proof)
        expect(valid, "Spartan prove+verify small R1CS (x*x=y, x=3)")
    } catch {
        expect(false, "Spartan small R1CS error: \(error)")
    }

    // --- Test 2: Quadratic example circuit (x^2 + x + 5 = y, 3 constraints) ---
    do {
        let (instance, gen) = SpartanR1CSBuilder.exampleQuadratic()
        let xVal = frFromInt(4)
        let (publicInputs, witness) = gen(xVal)
        // x=4: x^2=16, x^2+x=20, x^2+x+5=25
        let z = SpartanR1CS.buildZ(publicInputs: publicInputs, witness: witness)
        expect(instance.isSatisfied(z: z), "Quadratic R1CS (x^2+x+5=y) satisfied for x=4")

        let engine = try SpartanEngine()
        let proof = try engine.prove(instance: instance, publicInputs: publicInputs, witness: witness)
        let valid = engine.verify(instance: instance, publicInputs: publicInputs, proof: proof)
        expect(valid, "Spartan prove+verify quadratic (x=4, y=25)")
    } catch {
        expect(false, "Spartan quadratic error: \(error)")
    }

    // --- Test 3: Medium circuit (16 multiply constraints) ---
    do {
        let (instance, publicInputs, witness) = SpartanR1CSBuilder.syntheticR1CS(numConstraints: 16)
        let z = SpartanR1CS.buildZ(publicInputs: publicInputs, witness: witness)
        expect(instance.isSatisfied(z: z), "Synthetic R1CS (16 constraints) satisfied")

        let engine = try SpartanEngine()
        let proof = try engine.prove(instance: instance, publicInputs: publicInputs, witness: witness)
        let valid = engine.verify(instance: instance, publicInputs: publicInputs, proof: proof)
        expect(valid, "Spartan prove+verify synthetic (16 constraints)")
    } catch {
        expect(false, "Spartan 16-constraint error: \(error)")
    }

    // --- Test 4: Invalid witness rejection ---
    // Build a valid circuit then tamper with the proof to check verifier rejects.
    do {
        let (instance, gen) = SpartanR1CSBuilder.exampleQuadratic()
        let xVal = frFromInt(5)
        let (publicInputs, witness) = gen(xVal)

        let engine = try SpartanEngine()
        let proof = try engine.prove(instance: instance, publicInputs: publicInputs, witness: witness)

        // Verify correct proof passes
        let valid = engine.verify(instance: instance, publicInputs: publicInputs, proof: proof)
        expect(valid, "Spartan valid proof accepted (x=5)")

        // Tamper: use wrong public input
        let wrongPublic: [Fr] = [frFromInt(999)]
        let rejected = engine.verify(instance: instance, publicInputs: wrongPublic, proof: proof)
        expect(!rejected, "Spartan rejects proof with wrong public input")

        // Tamper: modify azRx claim in proof (create a forged proof)
        let forgedProof = SpartanProof(
            witnessCommitment: proof.witnessCommitment,
            sc1Rounds: proof.sc1Rounds,
            azRx: frAdd(proof.azRx, Fr.one),  // tamper
            bzRx: proof.bzRx,
            czRx: proof.czRx,
            sc2Rounds: proof.sc2Rounds,
            zEval: proof.zEval,
            openingProof: proof.openingProof
        )
        let rejected2 = engine.verify(instance: instance, publicInputs: publicInputs, proof: forgedProof)
        expect(!rejected2, "Spartan rejects forged azRx claim")
    } catch {
        expect(false, "Spartan invalid witness error: \(error)")
    }

    // --- Test 5: Sumcheck helper functions independently ---
    do {
        // Test eq polynomial: eq([0,0], [0,0]) = 1, eq([1,0], [1,0]) = 1
        let eq00 = spartanEvalEq([Fr.zero, Fr.zero], [Fr.zero, Fr.zero])
        expect(spartanFrEqual(eq00, Fr.one), "eq([0,0],[0,0]) = 1")

        let eq11 = spartanEvalEq([Fr.one, Fr.one], [Fr.one, Fr.one])
        expect(spartanFrEqual(eq11, Fr.one), "eq([1,1],[1,1]) = 1")

        let eq01 = spartanEvalEq([Fr.zero, Fr.one], [Fr.one, Fr.zero])
        expect(spartanFrEqual(eq01, Fr.zero), "eq([0,1],[1,0]) = 0")

        // Test MLE evaluation: f(x1,x2) with evals [1,2,3,4] on {0,1}^2
        // f(0,0)=1, f(0,1)=2, f(1,0)=3, f(1,1)=4
        // f(0,0) should equal 1 via MLE
        let evals: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let mle00 = spartanEvalML(evals: evals, pt: [Fr.zero, Fr.zero])
        expect(spartanFrEqual(mle00, frFromInt(1)), "MLE eval at (0,0) = 1")

        let mle11 = spartanEvalML(evals: evals, pt: [Fr.one, Fr.one])
        expect(spartanFrEqual(mle11, frFromInt(4)), "MLE eval at (1,1) = 4")

        let mle10 = spartanEvalML(evals: evals, pt: [Fr.one, Fr.zero])
        expect(spartanFrEqual(mle10, frFromInt(3)), "MLE eval at (1,0) = 3")

        // Test quadratic interpolation: s(0)=2, s(1)=5, s(2)=10 => s(t) = t^2 + 2t + 2
        // s(3) should be 9 + 6 + 2 = 17
        let s0q = frFromInt(2), s1q = frFromInt(5), s2q = frFromInt(10)
        let interp3 = spartanInterpQuadratic(s0: s0q, s1: s1q, s2: s2q, t: frFromInt(3))
        expect(spartanFrEqual(interp3, frFromInt(17)), "Quadratic interp s(3) = 17")

        // Test cubic interpolation: s(0)=1, s(1)=2, s(2)=9, s(3)=28 => s(t) = t^3 - t + 2..?
        // Actually just check round-trip: interp at 0,1,2,3 should return the original values
        let c0 = frFromInt(1), c1 = frFromInt(8), c2 = frFromInt(27), c3 = frFromInt(64)
        let ic0 = spartanInterpCubic(s0: c0, s1: c1, s2: c2, s3: c3, t: Fr.zero)
        let ic1 = spartanInterpCubic(s0: c0, s1: c1, s2: c2, s3: c3, t: Fr.one)
        expect(spartanFrEqual(ic0, c0), "Cubic interp round-trip at t=0")
        expect(spartanFrEqual(ic1, c1), "Cubic interp round-trip at t=1")
    }

    // --- Test 6: R1CS builder gates ---
    do {
        // Test addGate: a + b = out
        let b = SpartanR1CSBuilder()
        let pub = b.addPublicInput()  // var 1
        let a = b.addWitness()        // var 2
        let bv = b.addWitness()       // var 3
        let sum = b.addWitness()      // var 4
        let prod = b.addWitness()     // var 5

        b.addGate(a: a, b: bv, out: sum)     // a + b = sum
        b.mulGate(a: a, b: bv, out: prod)    // a * b = prod
        b.addConstraint(a: [(prod, Fr.one)], b: [(0, Fr.one)], c: [(pub, Fr.one)]) // prod = pub

        let instance = b.build()
        expect(instance.numConstraints == 3, "Builder: 3 constraints")
        expect(instance.numVariables == 6, "Builder: 6 variables (1 + 1pub + 4wit)")

        // Witness: a=3, b=4 => sum=7, prod=12
        let aVal = frFromInt(3)
        let bVal = frFromInt(4)
        let sumVal = frAdd(aVal, bVal)
        let prodVal = frMul(aVal, bVal)
        let pubVal = prodVal

        let z = SpartanR1CS.buildZ(publicInputs: [pubVal], witness: [aVal, bVal, sumVal, prodVal])
        expect(instance.isSatisfied(z: z), "Builder gates R1CS satisfied")

        // Wrong witness: tamper sum
        let zBad = SpartanR1CS.buildZ(publicInputs: [pubVal],
                                       witness: [aVal, bVal, frFromInt(99), prodVal])
        expect(!instance.isSatisfied(z: zBad), "Builder gates R1CS rejects wrong sum")
    }

    // --- Test 7: Spartan with IPA commitment (generic engine) ---
    do {
        // Build a small circuit
        let (instance, gen) = SpartanR1CSBuilder.exampleQuadratic()
        let xVal = frFromInt(3)
        let (publicInputs, witness) = gen(xVal)

        // We need IPA generators of size >= paddedN
        let paddedN = instance.paddedN
        // Generate random generators for IPA
        var generators = [PointAffine]()
        generators.reserveCapacity(paddedN)
        let g = bn254G1Generator()
        for i in 0..<paddedN {
            let scalar = frFromInt(UInt64(i + 1) * 7 + 13)
            let pt = cPointScalarMul(pointFromAffine(g), scalar)
            if let aff = pointToAffine(pt) {
                generators.append(aff)
            }
        }
        // Q generator for blinding
        let qScalar = frFromInt(UInt64(paddedN + 1) * 7 + 13)
        let qPt = cPointScalarMul(pointFromAffine(g), qScalar)
        guard let qAff = pointToAffine(qPt) else {
            expect(false, "IPA Q generator affine conversion failed"); return
        }

        let ipaEngine = try IPAEngine(generators: generators, Q: qAff)
        let adapter = IPAPCSAdapter(engine: ipaEngine)
        let spartanIPA = SpartanGenericEngine(pcs: adapter)

        let proof = try spartanIPA.prove(instance: instance, publicInputs: publicInputs, witness: witness)
        let valid = spartanIPA.verify(instance: instance, publicInputs: publicInputs, proof: proof)
        expect(valid, "Spartan+IPA prove+verify quadratic (x=3)")

        // Wrong public input should be rejected
        let wrongPub: [Fr] = [frFromInt(42)]
        let rejected = spartanIPA.verify(instance: instance, publicInputs: wrongPub, proof: proof)
        expect(!rejected, "Spartan+IPA rejects wrong public input")
    } catch {
        expect(false, "Spartan+IPA error: \(error)")
    }

    // --- Test 8: Repeated prove calls (tests buffer caching) ---
    do {
        let (instance, gen) = SpartanR1CSBuilder.exampleQuadratic()
        let engine = try SpartanEngine()

        for x in [2, 5, 10] as [UInt64] {
            let xVal = frFromInt(x)
            let (pub, wit) = gen(xVal)
            let proof = try engine.prove(instance: instance, publicInputs: pub, witness: wit)
            let valid = engine.verify(instance: instance, publicInputs: pub, proof: proof)
            expect(valid, "Spartan repeated prove x=\(x)")
        }
    } catch {
        expect(false, "Spartan repeated prove error: \(error)")
    }

    // --- Test 9: Performance test (2^10 = 1024 constraints) ---
    do {
        let n = 1024
        let (instance, publicInputs, witness) = SpartanR1CSBuilder.syntheticR1CS(numConstraints: n)
        let z = SpartanR1CS.buildZ(publicInputs: publicInputs, witness: witness)
        expect(instance.isSatisfied(z: z), "Synthetic R1CS (1024 constraints) satisfied")

        let engine = try SpartanEngine()

        // Warm-up
        let _ = try engine.prove(instance: instance, publicInputs: publicInputs, witness: witness)

        // Timed run
        let t0 = CFAbsoluteTimeGetCurrent()
        let proof = try engine.prove(instance: instance, publicInputs: publicInputs, witness: witness)
        let proveTime = CFAbsoluteTimeGetCurrent() - t0

        let t1 = CFAbsoluteTimeGetCurrent()
        let valid = engine.verify(instance: instance, publicInputs: publicInputs, proof: proof)
        let verifyTime = CFAbsoluteTimeGetCurrent() - t1

        expect(valid, "Spartan 2^10 constraints: prove+verify correct")
        print(String(format: "  Spartan 2^10: prove %.1fms, verify %.1fms",
                     proveTime * 1000, verifyTime * 1000))
    } catch {
        expect(false, "Spartan 2^10 performance error: \(error)")
    }
}
