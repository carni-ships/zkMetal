// PlonkTests — End-to-end tests for the Plonk preprocessed prover
//
// Tests cover:
//   1. Simple addition circuit: a + b = c
//   2. Multiplication circuit: a * b = c
//   3. Public input circuit
//   4. Invalid witness rejected
//   5. Round-trip: prove then verify
//   6. Generic addGate API

import zkMetal
import Foundation

func runPlonkTests() {
    suite("Plonk Prover/Verifier")

    // Shared test SRS setup
    let gen = bn254G1Generator()
    let secret: [UInt32] = [0xABCD, 0x1234, 0x5678, 0x9ABC, 0xDEF0, 0x1111, 0x2222, 0x0001]
    let srsSecret = frFromLimbs(secret)
    let srs = KZGEngine.generateTestSRS(secret: secret, size: 64, generator: gen)

    do {
        let kzg = try KZGEngine(srs: srs)
        let ntt = try NTTEngine()
        let preprocessor = PlonkPreprocessor(kzg: kzg, ntt: ntt)

        // ========== Test 1: Addition circuit a + b = c ==========
        do {
            let builder = PlonkCircuitBuilder()
            let a = builder.addInput()     // var 0
            let b = builder.addInput()     // var 1
            let c = builder.add(a, b)      // var 2: a + b = c

            let circuit = builder.build().padded()

            let setup = try preprocessor.setup(circuit: circuit, srsSecret: srsSecret)
            let prover = PlonkProver(setup: setup, kzg: kzg, ntt: ntt)
            let verifier = PlonkVerifier(setup: setup, kzg: kzg)

            // Witness: a=3, b=5, c=8
            let numVars = circuit.wireAssignments.flatMap { $0 }.max()! + 1
            var witness = [Fr](repeating: Fr.zero, count: numVars)
            witness[a] = frFromInt(3)
            witness[b] = frFromInt(5)
            witness[c] = frFromInt(8)

            let proof = try prover.prove(witness: witness, circuit: circuit)
            let valid = verifier.verify(proof: proof)
            expect(valid, "Addition circuit: prove then verify")
        } catch {
            expect(false, "Addition circuit error: \(error)")
        }

        // ========== Test 2: Multiplication circuit a * b = c ==========
        do {
            let builder = PlonkCircuitBuilder()
            let a = builder.addInput()     // var 0
            let b = builder.addInput()     // var 1
            let c = builder.mul(a, b)      // var 2: a * b = c

            let circuit = builder.build().padded()

            let setup = try preprocessor.setup(circuit: circuit, srsSecret: srsSecret)
            let prover = PlonkProver(setup: setup, kzg: kzg, ntt: ntt)
            let verifier = PlonkVerifier(setup: setup, kzg: kzg)

            // Witness: a=7, b=6, c=42
            let numVars = circuit.wireAssignments.flatMap { $0 }.max()! + 1
            var witness = [Fr](repeating: Fr.zero, count: numVars)
            witness[a] = frFromInt(7)
            witness[b] = frFromInt(6)
            witness[c] = frFromInt(42)

            let proof = try prover.prove(witness: witness, circuit: circuit)
            let valid = verifier.verify(proof: proof)
            expect(valid, "Multiplication circuit: prove then verify")
        } catch {
            expect(false, "Multiplication circuit error: \(error)")
        }

        // ========== Test 3: Public input circuit ==========
        do {
            let builder = PlonkCircuitBuilder()
            let a = builder.addInput()     // var 0 - public
            let b = builder.addInput()     // var 1
            let c = builder.add(a, b)      // var 2: a + b = c

            builder.addPublicInput(wireIndex: a)

            let circuit = builder.build().padded()

            let setup = try preprocessor.setup(circuit: circuit, srsSecret: srsSecret)
            let prover = PlonkProver(setup: setup, kzg: kzg, ntt: ntt)
            let verifier = PlonkVerifier(setup: setup, kzg: kzg)

            // Witness: a=10 (public), b=20, c=30
            let numVars = circuit.wireAssignments.flatMap { $0 }.max()! + 1
            var witness = [Fr](repeating: Fr.zero, count: numVars)
            witness[a] = frFromInt(10)
            witness[b] = frFromInt(20)
            witness[c] = frFromInt(30)

            let proof = try prover.prove(witness: witness, circuit: circuit)

            // Verify proof has public inputs
            expect(proof.publicInputs.count == 1, "Public input count")
            let pubVal = frToInt(proof.publicInputs[0])
            expect(pubVal[0] == 10, "Public input value")

            let valid = verifier.verify(proof: proof)
            expect(valid, "Public input circuit: prove then verify")
        } catch {
            expect(false, "Public input circuit error: \(error)")
        }

        // ========== Test 4: Invalid witness rejected ==========
        do {
            let builder = PlonkCircuitBuilder()
            let a = builder.addInput()     // var 0
            let b = builder.addInput()     // var 1
            let c = builder.add(a, b)      // var 2: a + b = c

            let circuit = builder.build().padded()

            let setup = try preprocessor.setup(circuit: circuit, srsSecret: srsSecret)
            let prover = PlonkProver(setup: setup, kzg: kzg, ntt: ntt)
            let verifier = PlonkVerifier(setup: setup, kzg: kzg)

            // Bad witness: a=3, b=5, c=9 (should be 8)
            let numVars = circuit.wireAssignments.flatMap { $0 }.max()! + 1
            var witness = [Fr](repeating: Fr.zero, count: numVars)
            witness[a] = frFromInt(3)
            witness[b] = frFromInt(5)
            witness[c] = frFromInt(9)  // WRONG!

            let proof = try prover.prove(witness: witness, circuit: circuit)
            let valid = verifier.verify(proof: proof)
            expect(!valid, "Invalid witness rejected")
        } catch {
            // An error during proving with bad witness is also acceptable
            expect(true, "Invalid witness rejected (threw)")
        }

        // ========== Test 5: Larger circuit round-trip ==========
        do {
            let builder = PlonkCircuitBuilder()
            let x = builder.addInput()       // var 0
            let y = builder.addInput()       // var 1
            let sum = builder.add(x, y)      // var 2: x + y
            let prod = builder.mul(x, y)     // var 3: x * y
            let result = builder.add(sum, prod) // var 4: (x+y) + (x*y)

            let circuit = builder.build().padded()

            let setup = try preprocessor.setup(circuit: circuit, srsSecret: srsSecret)
            let prover = PlonkProver(setup: setup, kzg: kzg, ntt: ntt)
            let verifier = PlonkVerifier(setup: setup, kzg: kzg)

            // Witness: x=4, y=3, sum=7, prod=12, result=19
            let numVars = circuit.wireAssignments.flatMap { $0 }.max()! + 1
            var witness = [Fr](repeating: Fr.zero, count: numVars)
            witness[x] = frFromInt(4)
            witness[y] = frFromInt(3)
            witness[sum] = frFromInt(7)
            witness[prod] = frFromInt(12)
            witness[result] = frFromInt(19)

            let proof = try prover.prove(witness: witness, circuit: circuit)
            let valid = verifier.verify(proof: proof)
            expect(valid, "Larger circuit round-trip: prove then verify")
        } catch {
            expect(false, "Larger circuit error: \(error)")
        }

        // ========== Test 6: Generic addGate API ==========
        do {
            let builder = PlonkCircuitBuilder()
            let a = builder.addInput()     // var 0
            let b = builder.addInput()     // var 1
            let c = builder.addInput()     // var 2 (output)

            // Add custom gate: 2*a + 3*b - c = 0  =>  c = 2a + 3b
            builder.addGate(
                qL: frFromInt(2), qR: frFromInt(3),
                qO: frSub(Fr.zero, Fr.one),
                qM: Fr.zero, qC: Fr.zero,
                a: a, b: b, c: c
            )

            let circuit = builder.build().padded()

            let setup = try preprocessor.setup(circuit: circuit, srsSecret: srsSecret)
            let prover = PlonkProver(setup: setup, kzg: kzg, ntt: ntt)
            let verifier = PlonkVerifier(setup: setup, kzg: kzg)

            // Witness: a=5, b=7, c=2*5+3*7=31
            let numVars = circuit.wireAssignments.flatMap { $0 }.max()! + 1
            var witness = [Fr](repeating: Fr.zero, count: numVars)
            witness[a] = frFromInt(5)
            witness[b] = frFromInt(7)
            witness[c] = frFromInt(31)

            let proof = try prover.prove(witness: witness, circuit: circuit)
            let valid = verifier.verify(proof: proof)
            expect(valid, "Generic addGate API: prove then verify")
        } catch {
            expect(false, "Generic addGate error: \(error)")
        }

        // ========== Test 7: Constant gate ==========
        do {
            let builder = PlonkCircuitBuilder()
            let a = builder.addInput()     // var 0
            let five = builder.constant(frFromInt(5))  // var 1 (output of constant gate)
            let c = builder.add(a, five)   // var 3: a + 5

            let circuit = builder.build().padded()

            let setup = try preprocessor.setup(circuit: circuit, srsSecret: srsSecret)
            let prover = PlonkProver(setup: setup, kzg: kzg, ntt: ntt)
            let verifier = PlonkVerifier(setup: setup, kzg: kzg)

            // Witness: a=10, five=5, c=15
            let numVars = circuit.wireAssignments.flatMap { $0 }.max()! + 1
            var witness = [Fr](repeating: Fr.zero, count: numVars)
            witness[a] = frFromInt(10)
            witness[five] = frFromInt(5)
            // The constant gate also has a dummy variable (var 2)
            witness[c] = frFromInt(15)

            let proof = try prover.prove(witness: witness, circuit: circuit)
            let valid = verifier.verify(proof: proof)
            expect(valid, "Constant gate: prove then verify")
        } catch {
            expect(false, "Constant gate error: \(error)")
        }

        // ========== Test 8: Verification Key extraction ==========
        do {
            let builder = PlonkCircuitBuilder()
            let a = builder.addInput()
            let b = builder.addInput()
            _ = builder.add(a, b)

            let circuit = builder.build().padded()
            let setup = try preprocessor.setup(circuit: circuit, srsSecret: srsSecret)

            let vk = PlonkVerificationKey(from: setup)
            expect(vk.n == setup.n, "VK domain size matches setup")
            expect(vk.selectorCommitments.count == setup.selectorCommitments.count,
                   "VK selector commitments count")
            expect(vk.permutationCommitments.count == setup.permutationCommitments.count,
                   "VK permutation commitments count")
        } catch {
            expect(false, "Verification key error: \(error)")
        }

        // ========== Test 9: Full pipeline — x^2 + x + 5 = 35, x=5 ==========
        // Exercises: circuit construction -> constraint eval -> KZG commit -> prove -> verify
        do {
            let builder = PlonkCircuitBuilder()

            // Allocate input variable x
            let x = builder.addInput()         // var 0

            // Gate 1: x_sq = x * x
            let x_sq = builder.mul(x, x)       // var 1: x^2

            // Gate 2: sum1 = x_sq + x
            let sum1 = builder.add(x_sq, x)    // var 2: x^2 + x

            // Gate 3: five = constant(5)
            let five = builder.constant(frFromInt(5))  // var 3 (output), var 4 (dummy)

            // Gate 4: result = sum1 + five  =>  x^2 + x + 5
            let result = builder.add(sum1, five) // var 5: x^2 + x + 5

            // Gate 5: thirty_five = constant(35)
            let thirty_five = builder.constant(frFromInt(35)) // var 6 (output), var 7 (dummy)

            // Copy constraint: result == thirty_five
            builder.assertEqual(result, thirty_five)

            // Mark x as public input
            builder.addPublicInput(wireIndex: x)

            let circuit = builder.build().padded()

            let setup = try preprocessor.setup(circuit: circuit, srsSecret: srsSecret)
            let prover = PlonkProver(setup: setup, kzg: kzg, ntt: ntt)
            let verifier = PlonkVerifier(setup: setup, kzg: kzg)

            // Witness: x=5, x^2=25, x^2+x=30, 5=5, x^2+x+5=35, 35=35
            let numVars = circuit.wireAssignments.flatMap { $0 }.max()! + 1
            var witness = [Fr](repeating: Fr.zero, count: numVars)
            witness[x] = frFromInt(5)
            witness[x_sq] = frFromInt(25)
            witness[sum1] = frFromInt(30)
            witness[five] = frFromInt(5)
            witness[result] = frFromInt(35)
            witness[thirty_five] = frFromInt(35)

            let proof = try prover.prove(witness: witness, circuit: circuit)

            // Verify public input is x=5
            expect(proof.publicInputs.count == 1, "Full pipeline: public input count")
            let pubX = frToInt(proof.publicInputs[0])
            expect(pubX[0] == 5, "Full pipeline: public input x=5")

            let valid = verifier.verify(proof: proof)
            expect(valid, "Full pipeline x^2+x+5=35: prove then verify")
        } catch {
            expect(false, "Full pipeline error: \(error)")
        }

        // ========== Test 10: Tampered proof rejected ==========
        // Generate a valid proof, then tamper with a commitment to ensure rejection
        do {
            let builder = PlonkCircuitBuilder()
            let a = builder.addInput()
            let b = builder.addInput()
            let c = builder.mul(a, b)

            let circuit = builder.build().padded()

            let setup = try preprocessor.setup(circuit: circuit, srsSecret: srsSecret)
            let prover = PlonkProver(setup: setup, kzg: kzg, ntt: ntt)
            let verifier = PlonkVerifier(setup: setup, kzg: kzg)

            // Valid witness: a=4, b=9, c=36
            let numVars = circuit.wireAssignments.flatMap { $0 }.max()! + 1
            var witness = [Fr](repeating: Fr.zero, count: numVars)
            witness[a] = frFromInt(4)
            witness[b] = frFromInt(9)
            witness[c] = frFromInt(36)

            let proof = try prover.prove(witness: witness, circuit: circuit)

            // Sanity: original proof verifies
            let valid = verifier.verify(proof: proof)
            expect(valid, "Tampered proof baseline: valid proof passes")

            // Tamper: replace aCommit with a different point (scalar mul of generator)
            let tamperedACommit = cPointScalarMul(proof.bCommit, frFromInt(42))
            let tamperedProof = PlonkProof(
                aCommit: tamperedACommit,
                bCommit: proof.bCommit,
                cCommit: proof.cCommit,
                zCommit: proof.zCommit,
                tLoCommit: proof.tLoCommit,
                tMidCommit: proof.tMidCommit,
                tHiCommit: proof.tHiCommit,
                tExtraCommits: proof.tExtraCommits,
                aEval: proof.aEval,
                bEval: proof.bEval,
                cEval: proof.cEval,
                sigma1Eval: proof.sigma1Eval,
                sigma2Eval: proof.sigma2Eval,
                zOmegaEval: proof.zOmegaEval,
                openingProof: proof.openingProof,
                shiftedOpeningProof: proof.shiftedOpeningProof,
                publicInputs: proof.publicInputs
            )
            let tamperedValid = verifier.verify(proof: tamperedProof)
            expect(!tamperedValid, "Tampered proof (modified aCommit) rejected")

            // Tamper: modify wire evaluation aEval
            let tamperedProof2 = PlonkProof(
                aCommit: proof.aCommit,
                bCommit: proof.bCommit,
                cCommit: proof.cCommit,
                zCommit: proof.zCommit,
                tLoCommit: proof.tLoCommit,
                tMidCommit: proof.tMidCommit,
                tHiCommit: proof.tHiCommit,
                tExtraCommits: proof.tExtraCommits,
                aEval: frAdd(proof.aEval, Fr.one),  // tamper: a(zeta) += 1
                bEval: proof.bEval,
                cEval: proof.cEval,
                sigma1Eval: proof.sigma1Eval,
                sigma2Eval: proof.sigma2Eval,
                zOmegaEval: proof.zOmegaEval,
                openingProof: proof.openingProof,
                shiftedOpeningProof: proof.shiftedOpeningProof,
                publicInputs: proof.publicInputs
            )
            let tamperedValid2 = verifier.verify(proof: tamperedProof2)
            expect(!tamperedValid2, "Tampered proof (modified aEval) rejected")
        } catch {
            expect(false, "Tampered proof error: \(error)")
        }

        // ========== Test 11: Constraint Compiler — addition circuit ==========
        do {
            let compiler = PlonkConstraintCompiler()
            let a = compiler.addVariable()
            let b = compiler.addVariable()
            let c = compiler.addVariable()

            // a + b - c = 0
            compiler.addArithmeticGate(
                qL: Fr.one, qR: Fr.one,
                qO: frSub(Fr.zero, Fr.one),
                qM: Fr.zero, qC: Fr.zero,
                wires: [a, b, c])

            let data = compiler.compile()
            expect(data.domainSize >= 1, "Compiler: domain size >= 1")
            expect(data.domainSize & (data.domainSize - 1) == 0, "Compiler: domain is power of 2")
            expect(data.circuit.gates.count == data.domainSize, "Compiler: gate count matches domain")
            expect(data.circuit.wireAssignments.count == data.domainSize, "Compiler: wire assignments match domain")

            // Full prove-verify round trip through compiler
            let setup = try preprocessor.setup(circuit: data.circuit, srsSecret: srsSecret)
            let prover = PlonkProver(setup: setup, kzg: kzg, ntt: ntt)
            let verifier = PlonkVerifier(setup: setup, kzg: kzg)

            var witness = [Fr](repeating: Fr.zero, count: data.numVariables)
            witness[a] = frFromInt(7)
            witness[b] = frFromInt(11)
            witness[c] = frFromInt(18)

            let proof = try prover.prove(witness: witness, circuit: data.circuit)
            let valid = verifier.verify(proof: proof)
            expect(valid, "Constraint Compiler addition: prove then verify")
        } catch {
            expect(false, "Constraint Compiler addition error: \(error)")
        }

        // ========== Test 12: Constraint Compiler — multiplication circuit ==========
        do {
            let compiler = PlonkConstraintCompiler()
            let a = compiler.addVariable()
            let b = compiler.addVariable()
            let c = compiler.addVariable()

            // a * b - c = 0
            compiler.addArithmeticGate(
                qL: Fr.zero, qR: Fr.zero,
                qO: frSub(Fr.zero, Fr.one),
                qM: Fr.one, qC: Fr.zero,
                wires: [a, b, c])

            let data = compiler.compile()
            let setup = try preprocessor.setup(circuit: data.circuit, srsSecret: srsSecret)
            let prover = PlonkProver(setup: setup, kzg: kzg, ntt: ntt)
            let verifier = PlonkVerifier(setup: setup, kzg: kzg)

            var witness = [Fr](repeating: Fr.zero, count: data.numVariables)
            witness[a] = frFromInt(6)
            witness[b] = frFromInt(9)
            witness[c] = frFromInt(54)

            let proof = try prover.prove(witness: witness, circuit: data.circuit)
            let valid = verifier.verify(proof: proof)
            expect(valid, "Constraint Compiler multiplication: prove then verify")
        } catch {
            expect(false, "Constraint Compiler multiplication error: \(error)")
        }

        // ========== Test 13: Constraint Compiler — copy constraints ==========
        do {
            let compiler = PlonkConstraintCompiler()
            let x = compiler.addVariable()
            let y = compiler.addVariable()
            let sum = compiler.addVariable()
            let prod = compiler.addVariable()
            let result = compiler.addVariable()

            // Gate 1: x + y = sum
            compiler.addArithmeticGate(
                qL: Fr.one, qR: Fr.one,
                qO: frSub(Fr.zero, Fr.one),
                qM: Fr.zero, qC: Fr.zero,
                wires: [x, y, sum])

            // Gate 2: x * y = prod
            compiler.addArithmeticGate(
                qL: Fr.zero, qR: Fr.zero,
                qO: frSub(Fr.zero, Fr.one),
                qM: Fr.one, qC: Fr.zero,
                wires: [x, y, prod])

            // Gate 3: sum + prod = result
            compiler.addArithmeticGate(
                qL: Fr.one, qR: Fr.one,
                qO: frSub(Fr.zero, Fr.one),
                qM: Fr.zero, qC: Fr.zero,
                wires: [sum, prod, result])

            let data = compiler.compile()
            let setup = try preprocessor.setup(circuit: data.circuit, srsSecret: srsSecret)
            let prover = PlonkProver(setup: setup, kzg: kzg, ntt: ntt)
            let verifier = PlonkVerifier(setup: setup, kzg: kzg)

            // x=4, y=3 => sum=7, prod=12, result=19
            var witness = [Fr](repeating: Fr.zero, count: data.numVariables)
            witness[x] = frFromInt(4)
            witness[y] = frFromInt(3)
            witness[sum] = frFromInt(7)
            witness[prod] = frFromInt(12)
            witness[result] = frFromInt(19)

            let proof = try prover.prove(witness: witness, circuit: data.circuit)
            let valid = verifier.verify(proof: proof)
            expect(valid, "Constraint Compiler copy constraints: prove then verify")
        } catch {
            expect(false, "Constraint Compiler copy constraints error: \(error)")
        }

        // ========== Test 14: Constraint Compiler — compileAndPreprocess ==========
        do {
            let compiler = PlonkConstraintCompiler()
            let a = compiler.addVariable()
            let b = compiler.addVariable()
            let c = compiler.addVariable()

            compiler.addArithmeticGate(
                qL: Fr.one, qR: Fr.one,
                qO: frSub(Fr.zero, Fr.one),
                qM: Fr.zero, qC: Fr.zero,
                wires: [a, b, c])

            let (setup, data) = try compiler.compileAndPreprocess(
                kzg: kzg, ntt: ntt, srsSecret: srsSecret)

            let prover = PlonkProver(setup: setup, kzg: kzg, ntt: ntt)
            let verifier = PlonkVerifier(setup: setup, kzg: kzg)

            var witness = [Fr](repeating: Fr.zero, count: data.numVariables)
            witness[a] = frFromInt(100)
            witness[b] = frFromInt(200)
            witness[c] = frFromInt(300)

            let proof = try prover.prove(witness: witness, circuit: data.circuit)
            let valid = verifier.verify(proof: proof)
            expect(valid, "compileAndPreprocess: prove then verify")

            // Verify VK extraction works
            let vk = VerifierKeyBuilder.build(from: setup)
            expect(vk.n == data.domainSize, "compileAndPreprocess: VK domain size")
        } catch {
            expect(false, "compileAndPreprocess error: \(error)")
        }

        // ========== Test 15: Constraint Compiler — bad witness rejected ==========
        do {
            let compiler = PlonkConstraintCompiler()
            let a = compiler.addVariable()
            let b = compiler.addVariable()
            let c = compiler.addVariable()

            compiler.addArithmeticGate(
                qL: Fr.one, qR: Fr.one,
                qO: frSub(Fr.zero, Fr.one),
                qM: Fr.zero, qC: Fr.zero,
                wires: [a, b, c])

            let data = compiler.compile()
            let setup = try preprocessor.setup(circuit: data.circuit, srsSecret: srsSecret)
            let prover = PlonkProver(setup: setup, kzg: kzg, ntt: ntt)
            let verifier = PlonkVerifier(setup: setup, kzg: kzg)

            // Bad witness: 7 + 11 != 99
            var witness = [Fr](repeating: Fr.zero, count: data.numVariables)
            witness[a] = frFromInt(7)
            witness[b] = frFromInt(11)
            witness[c] = frFromInt(99)

            let proof = try prover.prove(witness: witness, circuit: data.circuit)
            let valid = verifier.verify(proof: proof)
            expect(!valid, "Constraint Compiler bad witness rejected")
        } catch {
            expect(true, "Constraint Compiler bad witness rejected (threw)")
        }

        // ========== Test 16: Constraint Compiler — lookup gate ==========
        do {
            let compiler = PlonkConstraintCompiler()
            let tableValues: [Fr] = [frFromInt(0), frFromInt(1), frFromInt(2), frFromInt(3)]
            let tableId = compiler.addLookupTable(values: tableValues)

            let v = compiler.addVariable()
            let dummyB = compiler.addVariable()
            let dummyC = compiler.addVariable()

            compiler.addGate(.lookup(LookupGateDesc(
                inputWire: v, tableId: tableId, auxWires: [dummyB, dummyC])))

            let data = compiler.compile()
            expect(data.lookupTables.count == 1, "Compiler lookup: table registered")
            expect(data.circuit.lookupTables.count == 1, "Compiler lookup: circuit has table")

            // Verify the lookup gate was encoded with qLookup=1
            let g0 = data.circuit.gates[0]
            expect(!frEqual(g0.qLookup, Fr.zero), "Compiler lookup: qLookup is set")
        } catch {
            expect(false, "Constraint Compiler lookup error: \(error)")
        }

        // ========== Test 17: Constraint Compiler — public inputs ==========
        do {
            let compiler = PlonkConstraintCompiler()
            let a = compiler.addVariable()
            let b = compiler.addVariable()
            let c = compiler.addVariable()

            compiler.addPublicInput(a)

            compiler.addArithmeticGate(
                qL: Fr.one, qR: Fr.one,
                qO: frSub(Fr.zero, Fr.one),
                qM: Fr.zero, qC: Fr.zero,
                wires: [a, b, c])

            let data = compiler.compile()
            expect(data.circuit.publicInputIndices.count == 1, "Compiler public input: count")
            expect(data.circuit.publicInputIndices[0] == a, "Compiler public input: index")

            let setup = try preprocessor.setup(circuit: data.circuit, srsSecret: srsSecret)
            let prover = PlonkProver(setup: setup, kzg: kzg, ntt: ntt)

            var witness = [Fr](repeating: Fr.zero, count: data.numVariables)
            witness[a] = frFromInt(42)
            witness[b] = frFromInt(8)
            witness[c] = frFromInt(50)

            let proof = try prover.prove(witness: witness, circuit: data.circuit)
            expect(proof.publicInputs.count == 1, "Compiler public input: proof has PI")
            let pubVal = frToInt(proof.publicInputs[0])
            expect(pubVal[0] == 42, "Compiler public input: value = 42")
        } catch {
            expect(false, "Constraint Compiler public input error: \(error)")
        }

        // ========== Test 18: Constraint Compiler — explicit domain size ==========
        do {
            let compiler = PlonkConstraintCompiler()
            let a = compiler.addVariable()
            let b = compiler.addVariable()
            let c = compiler.addVariable()

            compiler.addArithmeticGate(
                qL: Fr.one, qR: Fr.one,
                qO: frSub(Fr.zero, Fr.one),
                qM: Fr.zero, qC: Fr.zero,
                wires: [a, b, c])

            let data = compiler.compile(domainSize: 16)
            expect(data.domainSize == 16, "Compiler explicit domain: size = 16")
            expect(data.circuit.gates.count == 16, "Compiler explicit domain: gate count = 16")
        } catch {
            expect(false, "Constraint Compiler explicit domain error: \(error)")
        }

        // ========== Test 19: SigmaPermutationBuilder ==========
        do {
            // Two gates sharing variable 'x' across different wire positions
            // Gate 0: a=x, b=y, c=z
            // Gate 1: a=z, b=x, c=w
            // x appears at (col=0, row=0) and (col=1, row=1) -> cycle
            // z appears at (col=2, row=0) and (col=0, row=1) -> cycle
            let wireAssignments = [[0, 1, 2], [2, 0, 3]]
            let n = 4
            let logN = 2
            let omega = computeNthRootOfUnity(logN: logN)
            var domain = [Fr](repeating: Fr.zero, count: n)
            domain[0] = Fr.one
            for i in 1..<n { domain[i] = frMul(domain[i-1], omega) }

            let k1 = frFromInt(2)
            let k2 = frFromInt(3)

            let sigmas = SigmaPermutationBuilder.buildSigmaEvals(
                wireAssignments: wireAssignments,
                copyConstraints: [],
                n: n, domain: domain, k1: k1, k2: k2)

            expect(sigmas.count == 3, "SigmaBuilder: 3 sigma columns")
            expect(sigmas[0].count == n, "SigmaBuilder: sigma1 length = n")

            // Variable 0 (x): positions (0,0) and (1,1)
            // sigma[0][0] should point to (1,1) = k1 * omega^1
            let expected_s0_0 = frMul(k1, domain[1])
            expect(frEqual(sigmas[0][0], expected_s0_0), "SigmaBuilder: x cycle (0,0)->(1,1)")

            // sigma[1][1] should point back to (0,0) = 1 * omega^0
            let expected_s1_1 = domain[0]
            expect(frEqual(sigmas[1][1], expected_s1_1), "SigmaBuilder: x cycle (1,1)->(0,0)")

            // Variable 2 (z): positions (2,0) and (0,1)
            // sigma[2][0] should point to (0,1) = 1 * omega^1
            let expected_s2_0 = domain[1]
            expect(frEqual(sigmas[2][0], expected_s2_0), "SigmaBuilder: z cycle (2,0)->(0,1)")

            // sigma[0][1] should point back to (2,0) = k2 * omega^0
            let expected_s0_1 = frMul(k2, domain[0])
            expect(frEqual(sigmas[0][1], expected_s0_1), "SigmaBuilder: z cycle (0,1)->(2,0)")
        }

        // ========== Test 20: Constraint Compiler — custom BoolCheck gate ==========
        do {
            let compiler = PlonkConstraintCompiler()
            let bit = compiler.addVariable()
            let dummyB = compiler.addVariable()
            let dummyC = compiler.addVariable()

            let selectorIdx = compiler.allocateCustomSelector()
            let boolGate = BoolCheckGate(column: 0)
            compiler.addGate(.custom(CustomGateDesc(
                gate: boolGate,
                wires: [[bit, dummyB, dummyC]],
                selectorIndex: selectorIdx)))

            let data = compiler.compile()

            // The gate should have qRange set (BoolCheckGate maps to range selector)
            let g0 = data.circuit.gates[0]
            expect(!frEqual(g0.qRange, Fr.zero), "Compiler BoolCheck: qRange is set")
            expect(data.maxRotation == 0, "Compiler BoolCheck: no rotation")
        } catch {
            expect(false, "Constraint Compiler BoolCheck error: \(error)")
        }

        // ========== Original Test 11: Wrong witness — gate constraint violation ==========
        // Mul gate a*b=c with inconsistent witness: a=4, b=9, c=99 (should be 36)
        // This violates the gate constraint directly (not just copy constraints).
        do {
            let builder = PlonkCircuitBuilder()
            let a = builder.addInput()
            let b = builder.addInput()
            let c = builder.mul(a, b)

            let circuit = builder.build().padded()

            let setup = try preprocessor.setup(circuit: circuit, srsSecret: srsSecret)
            let prover = PlonkProver(setup: setup, kzg: kzg, ntt: ntt)
            let verifier = PlonkVerifier(setup: setup, kzg: kzg)

            // Bad witness: a=4, b=9, c=99 (should be 36)
            let numVars = circuit.wireAssignments.flatMap { $0 }.max()! + 1
            var witness = [Fr](repeating: Fr.zero, count: numVars)
            witness[a] = frFromInt(4)
            witness[b] = frFromInt(9)
            witness[c] = frFromInt(99)

            let proof = try prover.prove(witness: witness, circuit: circuit)
            let valid = verifier.verify(proof: proof)
            expect(!valid, "Wrong witness (gate violation a*b != c) rejected")
        } catch {
            // Throwing during prove with bad witness is also acceptable
            expect(true, "Wrong witness gate violation rejected (threw)")
        }

    } catch {
        expect(false, "Plonk setup failed: \(error)")
    }
}
