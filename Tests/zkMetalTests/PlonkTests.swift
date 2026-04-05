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

    } catch {
        expect(false, "Plonk setup failed: \(error)")
    }
}
