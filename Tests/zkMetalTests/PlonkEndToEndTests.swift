// PlonkEndToEndTests -- Full pipeline integration tests
//
// Exercises the complete Plonk stack from circuit definition through
// proof generation to verification, using the constraint compiler,
// permutation argument, custom gates, and lookup gates.
//
// Tests:
//   1. 3-gate addition circuit via compiler -> compile -> prove -> verify
//   2. Multiplication circuit with copy constraints -> prove -> verify
//   3. Circuit with BoolCheck custom gate -> prove -> verify
//   4. Circuit with range check lookup -> prove -> verify
//   5. Larger circuit (32+ gates) -> prove -> verify -> print timing
//   6. Wrong witness produces verification failure
//   7. Tampered proof produces verification failure

import zkMetal
import Foundation

func runPlonkEndToEndTests() {
    suite("Plonk End-to-End Pipeline")

    // Shared test SRS setup -- large enough for 64-gate circuits (domain size up to 128)
    let gen = bn254G1Generator()
    let secret: [UInt32] = [0xABCD, 0x1234, 0x5678, 0x9ABC, 0xDEF0, 0x1111, 0x2222, 0x0001]
    let srsSecret = frFromLimbs(secret)
    let srs = KZGEngine.generateTestSRS(secret: secret, size: 256, generator: gen)

    do {
        let kzg = try KZGEngine(srs: srs)
        let ntt = try NTTEngine()
        let preprocessor = PlonkPreprocessor(kzg: kzg, ntt: ntt)

        // ========== Test 1: 3-gate addition circuit via compiler ==========
        do {
            let compiler = PlonkConstraintCompiler()
            let v0 = compiler.addVariable()  // a
            let v1 = compiler.addVariable()  // b
            let v2 = compiler.addVariable()  // c = a + b
            let v3 = compiler.addVariable()  // d
            let v4 = compiler.addVariable()  // e = c + d

            // Gate 0: a + b - c = 0
            compiler.addArithmeticGate(
                qL: Fr.one, qR: Fr.one, qO: frSub(Fr.zero, Fr.one),
                qM: Fr.zero, qC: Fr.zero, wires: [v0, v1, v2])

            // Gate 1: c + d - e = 0
            compiler.addArithmeticGate(
                qL: Fr.one, qR: Fr.one, qO: frSub(Fr.zero, Fr.one),
                qM: Fr.zero, qC: Fr.zero, wires: [v2, v3, v4])

            // Gate 2: e + 0 - e = 0 (identity, just to have 3 gates)
            let vDummy = compiler.addVariable()
            compiler.addArithmeticGate(
                qL: Fr.one, qR: Fr.zero, qO: frSub(Fr.zero, Fr.one),
                qM: Fr.zero, qC: Fr.zero, wires: [v4, vDummy, v4])

            let data = compiler.compile()
            let circuit = data.circuit
            let setup = try preprocessor.setup(circuit: circuit, srsSecret: srsSecret)
            let prover = PlonkProver(setup: setup, kzg: kzg, ntt: ntt)
            let verifier = PlonkVerifier(setup: setup, kzg: kzg)

            // Witness: a=3, b=5, c=8, d=2, e=10
            var witness = [Fr](repeating: Fr.zero, count: data.numVariables)
            witness[v0] = frFromInt(3)
            witness[v1] = frFromInt(5)
            witness[v2] = frFromInt(8)
            witness[v3] = frFromInt(2)
            witness[v4] = frFromInt(10)

            let proof = try prover.prove(witness: witness, circuit: circuit)
            let valid = verifier.verify(proof: proof)
            expect(valid, "E2E Test 1: 3-gate addition circuit via compiler")
        } catch {
            expect(false, "E2E Test 1 error: \(error)")
        }

        // ========== Test 2: Multiplication circuit with copy constraints ==========
        do {
            let compiler = PlonkConstraintCompiler()
            let x = compiler.addVariable()   // input x
            let y = compiler.addVariable()   // input y
            let xy = compiler.addVariable()  // x * y
            let z = compiler.addVariable()   // input z
            let r = compiler.addVariable()   // xy + z = result

            // Gate 0: x * y = xy  (qM=1, qO=-1)
            compiler.addArithmeticGate(
                qL: Fr.zero, qR: Fr.zero, qO: frSub(Fr.zero, Fr.one),
                qM: Fr.one, qC: Fr.zero, wires: [x, y, xy])

            // Gate 1: xy + z = r  (qL=1, qR=1, qO=-1)
            compiler.addArithmeticGate(
                qL: Fr.one, qR: Fr.one, qO: frSub(Fr.zero, Fr.one),
                qM: Fr.zero, qC: Fr.zero, wires: [xy, z, r])

            // Copy constraint: xy in gate 0 output == xy in gate 1 input
            // The compiler handles this via shared variable index xy

            let data = compiler.compile()
            let circuit = data.circuit
            let setup = try preprocessor.setup(circuit: circuit, srsSecret: srsSecret)
            let prover = PlonkProver(setup: setup, kzg: kzg, ntt: ntt)
            let verifier = PlonkVerifier(setup: setup, kzg: kzg)

            // Witness: x=7, y=6, xy=42, z=8, r=50
            var witness = [Fr](repeating: Fr.zero, count: data.numVariables)
            witness[x] = frFromInt(7)
            witness[y] = frFromInt(6)
            witness[xy] = frFromInt(42)
            witness[z] = frFromInt(8)
            witness[r] = frFromInt(50)

            let proof = try prover.prove(witness: witness, circuit: circuit)
            let valid = verifier.verify(proof: proof)
            expect(valid, "E2E Test 2: Multiplication with copy constraints")
        } catch {
            expect(false, "E2E Test 2 error: \(error)")
        }

        // ========== Test 3: Circuit with BoolCheck custom gate ==========
        do {
            // Use the PlonkCircuitBuilder which has rangeCheck support
            // BoolCheck is a 1-bit range check
            let builder = PlonkCircuitBuilder()
            let a = builder.addInput()  // boolean variable
            let b = builder.addInput()  // another boolean
            let c = builder.add(a, b)   // sum

            // Add range (bool) gates for a and b
            let dummyA = builder.nextVariable; builder.nextVariable += 1
            builder.gates.append(PlonkGate(
                qL: Fr.zero, qR: Fr.zero, qO: Fr.zero, qM: Fr.zero, qC: Fr.zero,
                qRange: Fr.one))
            builder.wireAssignments.append([a, dummyA, dummyA])

            let dummyB = builder.nextVariable; builder.nextVariable += 1
            builder.gates.append(PlonkGate(
                qL: Fr.zero, qR: Fr.zero, qO: Fr.zero, qM: Fr.zero, qC: Fr.zero,
                qRange: Fr.one))
            builder.wireAssignments.append([b, dummyB, dummyB])

            let circuit = builder.build().padded()
            let setup = try preprocessor.setup(circuit: circuit, srsSecret: srsSecret)
            let prover = PlonkProver(setup: setup, kzg: kzg, ntt: ntt)
            let verifier = PlonkVerifier(setup: setup, kzg: kzg)

            // Witness: a=1, b=0, c=1
            let numVars = circuit.wireAssignments.flatMap { $0 }.max()! + 1
            var witness = [Fr](repeating: Fr.zero, count: numVars)
            witness[a] = Fr.one
            witness[b] = Fr.zero
            witness[c] = Fr.one

            let proof = try prover.prove(witness: witness, circuit: circuit)
            let valid = verifier.verify(proof: proof)
            expect(valid, "E2E Test 3: BoolCheck custom gate (a=1, b=0)")
        } catch {
            expect(false, "E2E Test 3 error: \(error)")
        }

        // ========== Test 4: Circuit with range check lookup ==========
        do {
            let builder = PlonkCircuitBuilder()
            let val = builder.addInput()  // value to look up

            // Create a small lookup table: {0, 1, 2, 3, 4, 5, 6, 7}
            let tableValues = (0..<8).map { frFromInt(UInt64($0)) }
            let tableId = builder.addLookupTable(values: tableValues)

            // Lookup gate: prove val is in the table
            builder.lookup(val, tableId: tableId)

            let circuit = builder.build().padded()
            let setup = try preprocessor.setup(circuit: circuit, srsSecret: srsSecret)
            let prover = PlonkProver(setup: setup, kzg: kzg, ntt: ntt)
            let verifier = PlonkVerifier(setup: setup, kzg: kzg)

            // Witness: val=5 (which is in the table)
            let numVars = circuit.wireAssignments.flatMap { $0 }.max()! + 1
            var witness = [Fr](repeating: Fr.zero, count: numVars)
            witness[val] = frFromInt(5)

            let proof = try prover.prove(witness: witness, circuit: circuit)
            let valid = verifier.verify(proof: proof)
            expect(valid, "E2E Test 4: Lookup gate (val=5 in range [0,8))")
        } catch {
            expect(false, "E2E Test 4 error: \(error)")
        }

        // ========== Test 5: Larger circuit (32+ gates) with timing ==========
        do {
            let t0 = CFAbsoluteTimeGetCurrent()

            let builder = PlonkCircuitBuilder()

            // Build a chain: x_0 = input, x_{i+1} = x_i + x_i (doubling chain)
            let x0 = builder.addInput()
            var prev = x0
            for _ in 0..<35 {
                prev = builder.add(prev, prev)
            }

            let circuit = builder.build().padded()

            let tSetup0 = CFAbsoluteTimeGetCurrent()
            let largeSrs = KZGEngine.generateTestSRS(secret: secret, size: circuit.numGates + 1, generator: gen)
            let largeKzg = try KZGEngine(srs: largeSrs)
            let setup = try preprocessor.setup(circuit: circuit, srsSecret: srsSecret)
            let tSetup1 = CFAbsoluteTimeGetCurrent()

            let prover = PlonkProver(setup: setup, kzg: largeKzg, ntt: ntt)
            let verifier = PlonkVerifier(setup: setup, kzg: largeKzg)

            // Witness: x0 = 1, each step doubles
            let numVars = circuit.wireAssignments.flatMap { $0 }.max()! + 1
            var witness = [Fr](repeating: Fr.zero, count: numVars)
            witness[x0] = Fr.one

            // Compute doubling chain witness values
            var curVal = Fr.one
            var varIdx = x0
            for i in 0..<35 {
                let newVal = frAdd(curVal, curVal)
                // The add gate at position i+1 (after the initial input usage)
                // has wires [prev, prev, output]
                let gateWires = circuit.wireAssignments[i]
                // Wire a and b point to prev, wire c points to output
                witness[gateWires[2]] = newVal
                // Also ensure the input wires have the right values
                witness[gateWires[0]] = curVal
                witness[gateWires[1]] = curVal
                curVal = newVal
            }

            let tProve0 = CFAbsoluteTimeGetCurrent()
            let proof = try prover.prove(witness: witness, circuit: circuit)
            let tProve1 = CFAbsoluteTimeGetCurrent()

            let tVerify0 = CFAbsoluteTimeGetCurrent()
            let valid = verifier.verify(proof: proof)
            let tVerify1 = CFAbsoluteTimeGetCurrent()

            expect(valid, "E2E Test 5: 35-gate doubling chain")

            let tTotal = CFAbsoluteTimeGetCurrent() - t0
            print(String(format: "    35-gate circuit: setup=%.1fms prove=%.1fms verify=%.1fms total=%.1fms (domain=%d)",
                         (tSetup1 - tSetup0) * 1000,
                         (tProve1 - tProve0) * 1000,
                         (tVerify1 - tVerify0) * 1000,
                         tTotal * 1000,
                         circuit.numGates))
        } catch {
            expect(false, "E2E Test 5 error: \(error)")
        }

        // ========== Test 6: Wrong witness produces verification failure ==========
        do {
            let builder = PlonkCircuitBuilder()
            let a = builder.addInput()
            let b = builder.addInput()
            let c = builder.add(a, b)

            let circuit = builder.build().padded()
            let setup = try preprocessor.setup(circuit: circuit, srsSecret: srsSecret)
            let prover = PlonkProver(setup: setup, kzg: kzg, ntt: ntt)
            let verifier = PlonkVerifier(setup: setup, kzg: kzg)

            // Wrong witness: a=3, b=5, c=9 (should be 8)
            let numVars = circuit.wireAssignments.flatMap { $0 }.max()! + 1
            var witness = [Fr](repeating: Fr.zero, count: numVars)
            witness[a] = frFromInt(3)
            witness[b] = frFromInt(5)
            witness[c] = frFromInt(9)  // WRONG: should be 8

            let proof = try prover.prove(witness: witness, circuit: circuit)
            let valid = verifier.verify(proof: proof)
            expect(!valid, "E2E Test 6: Wrong witness rejected by verifier")
        } catch {
            // If proving itself throws on bad witness, that's also acceptable
            expect(true, "E2E Test 6: Wrong witness caused proving error (acceptable)")
        }

        // ========== Test 7: Tampered proof produces verification failure ==========
        do {
            let builder = PlonkCircuitBuilder()
            let a = builder.addInput()
            let b = builder.addInput()
            let c = builder.add(a, b)

            let circuit = builder.build().padded()
            let setup = try preprocessor.setup(circuit: circuit, srsSecret: srsSecret)
            let prover = PlonkProver(setup: setup, kzg: kzg, ntt: ntt)
            let verifier = PlonkVerifier(setup: setup, kzg: kzg)

            // Correct witness
            let numVars = circuit.wireAssignments.flatMap { $0 }.max()! + 1
            var witness = [Fr](repeating: Fr.zero, count: numVars)
            witness[a] = frFromInt(3)
            witness[b] = frFromInt(5)
            witness[c] = frFromInt(8)

            let proof = try prover.prove(witness: witness, circuit: circuit)

            // Verify the correct proof first
            let validOriginal = verifier.verify(proof: proof)
            expect(validOriginal, "E2E Test 7: Original proof is valid")

            // Tamper: change aEval to a different value
            let tamperedProof = PlonkProof(
                aCommit: proof.aCommit, bCommit: proof.bCommit, cCommit: proof.cCommit,
                zCommit: proof.zCommit,
                tLoCommit: proof.tLoCommit, tMidCommit: proof.tMidCommit, tHiCommit: proof.tHiCommit,
                tExtraCommits: proof.tExtraCommits,
                aEval: frFromInt(999),  // TAMPERED
                bEval: proof.bEval, cEval: proof.cEval,
                sigma1Eval: proof.sigma1Eval, sigma2Eval: proof.sigma2Eval,
                zOmegaEval: proof.zOmegaEval,
                openingProof: proof.openingProof,
                shiftedOpeningProof: proof.shiftedOpeningProof,
                publicInputs: proof.publicInputs)

            let validTampered = verifier.verify(proof: tamperedProof)
            expect(!validTampered, "E2E Test 7: Tampered proof (aEval) rejected")
        } catch {
            expect(false, "E2E Test 7 error: \(error)")
        }

    } catch {
        expect(false, "E2E setup error: \(error)")
    }
}
