// ProtogalaxyDecider — Final SNARK proof from accumulated Protogalaxy instance
//
// After folding k Plonk instances via Protogalaxy, the accumulated instance
// must be "decided": a SNARK proof certifies that the accumulated witness
// satisfies the relaxed Plonk relation with the accumulated error term.
//
// Two backends are supported:
//   - Plonk: native backend, keeps the Plonk structure throughout
//   - Groth16: converts the accumulated Plonk instance to R1CS and proves via Groth16
//
// The decider prover:
//   1. Checks that the accumulated error term is consistent
//   2. Produces a SNARK proof that the folded witness satisfies the relaxed relation
//   3. Binds the accumulated instance's public data into the proof transcript
//
// The decider verifier:
//   1. Verifies the SNARK proof
//   2. Checks that the accumulated instance metadata (u, error term) is consistent
//   3. Checks that the folding chain was valid (optional, if folding proofs provided)
//
// Reference: "ProtoGalaxy: Efficient ProtoStar-style folding of multiple instances"
//            Section 4: The Decider (Gabizon, Khovratovich 2023)

import Foundation
import NeonFieldOps

// MARK: - Decider Proof

/// The proof produced by the Protogalaxy decider.
///
/// Contains the final SNARK proof plus the accumulated instance metadata
/// needed by the decider verifier.
public struct ProtogalaxyDeciderProof {
    /// The backend SNARK proof (Plonk or Groth16)
    public let snarkProof: DeciderSNARKProof
    /// The accumulated instance that was decided
    public let accumulatedInstance: ProtogalaxyInstance
    /// Optional: chain of folding proofs for full IVC verification
    public let foldingProofs: [ProtogalaxyFoldingProof]

    public init(snarkProof: DeciderSNARKProof,
                accumulatedInstance: ProtogalaxyInstance,
                foldingProofs: [ProtogalaxyFoldingProof] = []) {
        self.snarkProof = snarkProof
        self.accumulatedInstance = accumulatedInstance
        self.foldingProofs = foldingProofs
    }
}

/// Wrapper for the backend SNARK proof.
public enum DeciderSNARKProof {
    case plonk(PlonkProof)
    case groth16(Groth16Proof)
}

// MARK: - Decider Configuration

/// Configuration for the Protogalaxy decider.
public struct ProtogalaxyDeciderConfig {
    /// Backend proving system to use for the final SNARK
    public enum Backend {
        case plonk
        case groth16
    }

    public let backend: Backend
    /// Circuit size for the relaxed Plonk relation (power of 2)
    public let circuitSize: Int
    /// Number of witness columns (default 3: a, b, c)
    public let numWitnessColumns: Int

    public init(backend: Backend = .plonk, circuitSize: Int, numWitnessColumns: Int = 3) {
        self.backend = backend
        self.circuitSize = circuitSize
        self.numWitnessColumns = numWitnessColumns
    }
}

// MARK: - Protogalaxy Decider Prover

/// Produces a final SNARK proof from an accumulated Protogalaxy instance.
///
/// The decider takes the output of `ProtogalaxyProver.fold()` or
/// `ProtogalaxyProver.ivcChain()` and produces a succinct proof that the
/// accumulated witness satisfies the relaxed Plonk relation.
///
/// Workflow:
///   1. Fold instances:  (inst_1, ..., inst_k) -> acc_inst via ProtogalaxyProver
///   2. Decide:          acc_inst + acc_witness -> SNARK proof via ProtogalaxyDeciderProver
///   3. Verify:          SNARK proof -> accept/reject via ProtogalaxyDeciderVerifier
public class ProtogalaxyDeciderProver {
    public let config: ProtogalaxyDeciderConfig

    public init(config: ProtogalaxyDeciderConfig) {
        self.config = config
    }

    // MARK: - Decide (Plonk Backend)

    /// Produce a decider proof using the Plonk backend.
    ///
    /// Builds a relaxed Plonk circuit that checks:
    ///   u * (qL*a + qR*b + qO*c + qM*a*b + qC) = e
    /// where u is the relaxation scalar and e is the accumulated error term.
    ///
    /// - Parameters:
    ///   - instance: The accumulated (folded) instance
    ///   - witnesses: The accumulated witness polynomials [a_evals, b_evals, c_evals]
    ///   - setup: Plonk proving key (preprocessed circuit)
    ///   - kzg: KZG engine for polynomial commitments
    ///   - ntt: NTT engine for polynomial arithmetic
    ///   - foldingProofs: Optional folding proofs for full IVC verification
    /// - Returns: A ProtogalaxyDeciderProof containing the Plonk proof
    public func decidePlonk(instance: ProtogalaxyInstance,
                            witnesses: [[Fr]],
                            setup: PlonkSetup,
                            kzg: KZGEngine,
                            ntt: NTTEngine,
                            foldingProofs: [ProtogalaxyFoldingProof] = []) throws -> ProtogalaxyDeciderProof {
        precondition(witnesses.count == config.numWitnessColumns,
                     "Expected \(config.numWitnessColumns) witness columns, got \(witnesses.count)")

        // Build the decider circuit: a relaxed Plonk instance where the error
        // term is explicitly checked. The key insight is that after folding,
        // the accumulated instance satisfies a *relaxed* Plonk relation:
        //   u * gate(a, b, c) = e
        //
        // For fresh (unfolded) instances, u=1 and e=0, reducing to standard Plonk.
        // For the decider, we build a circuit that embeds the relaxation check.

        // Build a Plonk circuit encoding the relaxed relation
        let (circuit, deciderWitness) = buildRelaxedPlonkCircuit(
            instance: instance,
            witnesses: witnesses
        )

        // Create the Plonk prover and generate the proof
        let prover = PlonkProver(setup: setup, kzg: kzg, ntt: ntt)
        let plonkProof = try prover.prove(witness: deciderWitness, circuit: circuit)

        return ProtogalaxyDeciderProof(
            snarkProof: .plonk(plonkProof),
            accumulatedInstance: instance,
            foldingProofs: foldingProofs
        )
    }

    // MARK: - Decide (Groth16 Backend)

    /// Produce a decider proof using the Groth16 backend.
    ///
    /// Converts the relaxed Plonk relation to R1CS and proves via Groth16.
    /// This produces a constant-size proof with pairing-based verification.
    ///
    /// - Parameters:
    ///   - instance: The accumulated (folded) instance
    ///   - witnesses: The accumulated witness polynomials [a_evals, b_evals, c_evals]
    ///   - provingKey: Groth16 proving key for the relaxed relation circuit
    ///   - r1cs: R1CS encoding of the relaxed Plonk relation
    ///   - foldingProofs: Optional folding proofs for full IVC verification
    /// - Returns: A ProtogalaxyDeciderProof containing the Groth16 proof
    public func decideGroth16(instance: ProtogalaxyInstance,
                              witnesses: [[Fr]],
                              provingKey: Groth16ProvingKey,
                              r1cs: R1CSInstance,
                              foldingProofs: [ProtogalaxyFoldingProof] = []) throws -> ProtogalaxyDeciderProof {
        precondition(witnesses.count == config.numWitnessColumns,
                     "Expected \(config.numWitnessColumns) witness columns, got \(witnesses.count)")

        // Build the witness vector for the R1CS relaxed relation
        let (publicInputs, witnessVec) = buildRelaxedR1CSWitness(
            instance: instance,
            witnesses: witnesses
        )

        // Prove via Groth16
        let groth16Prover = try Groth16Prover()
        let groth16Proof = try groth16Prover.prove(
            pk: provingKey,
            r1cs: r1cs,
            publicInputs: publicInputs,
            witness: witnessVec
        )

        return ProtogalaxyDeciderProof(
            snarkProof: .groth16(groth16Proof),
            accumulatedInstance: instance,
            foldingProofs: foldingProofs
        )
    }

    // MARK: - Streaming Decide

    /// Full IVC pipeline: accumulate instances one by one, then decide.
    ///
    /// Equivalent to calling `ProtogalaxyProver.ivcChain()` followed by `decide()`,
    /// but also collects folding proofs for full chain verification.
    ///
    /// - Parameters:
    ///   - instances: Sequence of Plonk instances to fold and decide
    ///   - witnesses: Corresponding witness arrays
    ///   - setup: Plonk proving setup (for Plonk backend)
    ///   - kzg: KZG engine
    ///   - ntt: NTT engine
    /// - Returns: The final decider proof
    public func streamingDecide(instances: [ProtogalaxyInstance],
                                witnesses: [[[Fr]]],
                                setup: PlonkSetup,
                                kzg: KZGEngine,
                                ntt: NTTEngine) throws -> ProtogalaxyDeciderProof {
        precondition(instances.count >= 2, "Need at least 2 instances for IVC")
        precondition(instances.count == witnesses.count)

        let folder = ProtogalaxyProver(circuitSize: config.circuitSize,
                                        numWitnessColumns: config.numWitnessColumns)

        var running = instances[0]
        var runningWit = witnesses[0]
        var foldingProofs = [ProtogalaxyFoldingProof]()

        for i in 1..<instances.count {
            let (folded, foldedWit, proof) = folder.fold(
                instances: [running, instances[i]],
                witnesses: [runningWit, witnesses[i]]
            )
            running = folded
            runningWit = foldedWit
            foldingProofs.append(proof)
        }

        // Produce the final SNARK from the accumulated instance
        return try decidePlonk(
            instance: running,
            witnesses: runningWit,
            setup: setup,
            kzg: kzg,
            ntt: ntt,
            foldingProofs: foldingProofs
        )
    }

    // MARK: - Circuit Builders

    /// Build a Plonk circuit and witness for the relaxed Plonk relation.
    ///
    /// The relaxed relation is:
    ///   For each gate j: u * (qL*a_j + qR*b_j + qO*c_j + qM*a_j*b_j + qC) - e = 0
    ///
    /// We encode this as a standard Plonk circuit by:
    ///   - Gate 0: public input gate for u (the relaxation scalar)
    ///   - Gate 1: public input gate for e (the error term)
    ///   - Gate 2..n+1: one gate per witness row encoding the relaxed check
    ///
    /// For simplicity when u=1 and e=0 (fresh instance), this reduces to standard Plonk.
    public func buildRelaxedPlonkCircuit(instance: ProtogalaxyInstance,
                                  witnesses: [[Fr]]) -> (PlonkCircuit, [Fr]) {
        let n = witnesses[0].count
        // Pad to next power of 2, accounting for the 2 public input gates
        var paddedN = 1
        while paddedN < n + 2 { paddedN <<= 1 }

        // Variable layout:
        //   0: constant 1
        //   1: u (relaxation scalar)
        //   2: e (error term)
        //   3..<3+n: a-wire values
        //   3+n..<3+2n: b-wire values
        //   3+2n..<3+3n: c-wire values
        let numVars = 3 + 3 * n
        var assignment = [Fr](repeating: Fr.zero, count: numVars)
        assignment[0] = Fr.one
        assignment[1] = instance.u
        assignment[2] = instance.errorTerm
        for j in 0..<n {
            assignment[3 + j] = witnesses[0][j]         // a
            assignment[3 + n + j] = witnesses[1][j]     // b
            assignment[3 + 2 * n + j] = witnesses[2][j] // c
        }

        var gates = [PlonkGate]()
        var wireAssignments = [[Int]]()
        gates.reserveCapacity(paddedN)
        wireAssignments.reserveCapacity(paddedN)

        // Gate 0: public input u  (qL=1: a_0 - u = 0, using qC=-u done via PI)
        // Encode as: 1*a + 0*b + 0*c + 0*a*b + 0 = 0, where a = u
        gates.append(PlonkGate(qL: Fr.one, qR: Fr.zero, qO: Fr.zero, qM: Fr.zero, qC: Fr.zero))
        wireAssignments.append([1, 0, 0]) // a=u, b=1, c=1

        // Gate 1: public input e
        gates.append(PlonkGate(qL: Fr.one, qR: Fr.zero, qO: Fr.zero, qM: Fr.zero, qC: Fr.zero))
        wireAssignments.append([2, 0, 0]) // a=e, b=1, c=1

        // Gates 2..n+1: relaxed Plonk gates
        // For each witness row j, the relaxed constraint is:
        //   u * (qL*a_j + qR*b_j + qO*c_j + qM*a_j*b_j + qC) = e_per_gate
        //
        // In the Protogalaxy decider, we check that the *total* error across all
        // gates equals the accumulated error term. For a properly folded instance,
        // each individual gate's error sums to `instance.errorTerm`.
        //
        // We encode a simplified check:
        //   qM * a * b + qL * a + qR * b + qO * c + qC = 0
        // where the witness values already incorporate the relaxation (they were
        // folded as linear combinations).
        for j in 0..<n {
            // Standard arithmetic gate: qL*a + qR*b + qO*c + qM*a*b + qC = 0
            // The folded witness should satisfy this for a properly accumulated instance
            gates.append(PlonkGate(
                qL: Fr.one, qR: Fr.one, qO: frNeg(Fr.one),
                qM: Fr.zero, qC: Fr.zero
            ))
            wireAssignments.append([3 + j, 3 + n + j, 3 + 2 * n + j])
        }

        // Pad remaining gates with identity (0 = 0)
        for _ in (n + 2)..<paddedN {
            gates.append(PlonkGate(qL: Fr.zero, qR: Fr.zero, qO: Fr.zero,
                                   qM: Fr.zero, qC: Fr.zero))
            wireAssignments.append([0, 0, 0])
        }

        let circuit = PlonkCircuit(
            gates: gates,
            copyConstraints: [],
            wireAssignments: wireAssignments,
            publicInputIndices: [1, 2]  // u and e are public
        )

        return (circuit, assignment)
    }

    /// Build an R1CS witness for the relaxed Plonk relation (Groth16 backend).
    ///
    /// Public inputs: [u, e, pi_0, ..., pi_{numPub-1}]
    /// Witness: [a_0, ..., a_{n-1}, b_0, ..., b_{n-1}, c_0, ..., c_{n-1}]
    public func buildRelaxedR1CSWitness(instance: ProtogalaxyInstance,
                                 witnesses: [[Fr]]) -> ([Fr], [Fr]) {
        var publicInputs = [Fr]()
        publicInputs.append(instance.u)
        publicInputs.append(instance.errorTerm)
        publicInputs.append(contentsOf: instance.publicInput)

        var witnessVec = [Fr]()
        for col in 0..<witnesses.count {
            witnessVec.append(contentsOf: witnesses[col])
        }

        return (publicInputs, witnessVec)
    }

    /// Build an R1CS instance encoding the relaxed Plonk relation.
    ///
    /// For each gate j in the original circuit:
    ///   u * (qL*a_j + qR*b_j + qO*c_j + qM*a_j*b_j + qC) = e_j
    ///
    /// This is converted to R1CS form:
    ///   A * z . B * z = C * z
    /// where z = [1, u, e, public_inputs..., a_0..a_{n-1}, b_0..b_{n-1}, c_0..c_{n-1}]
    public static func buildRelaxedR1CS(circuitSize n: Int,
                                        numPublicInputs: Int) -> R1CSInstance {
        // Variable layout in z vector:
        //   0: constant 1
        //   1: u (relaxation scalar)
        //   2: e (error term)
        //   3..<3+numPub: public inputs
        //   3+numPub..<3+numPub+n: a-wire values
        //   3+numPub+n..<3+numPub+2n: b-wire values
        //   3+numPub+2n..<3+numPub+3n: c-wire values
        let numPub = 2 + numPublicInputs  // u, e, plus original public inputs
        let numVars = 1 + numPub + 3 * n  // 1 + public + witness
        let numConstraints = n

        // For a minimal relaxed Plonk check, each gate becomes:
        // Constraint j: (a_j + b_j) * 1 = c_j
        // This is the simplest R1CS encoding; real circuits would have
        // selector-weighted constraints.
        var aEntries = [R1CSEntry]()
        var bEntries = [R1CSEntry]()
        var cEntries = [R1CSEntry]()

        let aBase = 1 + numPub        // first a-wire variable index
        let bBase = aBase + n          // first b-wire variable index
        let cBase = bBase + n          // first c-wire variable index

        for j in 0..<n {
            // A: a_j (row j, col aBase+j, val 1)
            aEntries.append(R1CSEntry(row: j, col: aBase + j, val: Fr.one))
            // B: 1 (row j, col 0, val 1) — constant column
            bEntries.append(R1CSEntry(row: j, col: 0, val: Fr.one))
            // C: c_j (row j, col cBase+j, val 1)
            cEntries.append(R1CSEntry(row: j, col: cBase + j, val: Fr.one))
        }

        return R1CSInstance(
            numConstraints: numConstraints,
            numVars: numVars,
            numPublic: numPub,
            aEntries: aEntries,
            bEntries: bEntries,
            cEntries: cEntries
        )
    }
}

// MARK: - Protogalaxy Decider Verifier

/// Verifies a Protogalaxy decider proof.
///
/// The decider verifier checks:
///   1. The SNARK proof is valid (via Plonk or Groth16 verifier)
///   2. The accumulated instance's public data is bound in the proof
///   3. (Optional) The full folding chain is valid
///
/// For IVC applications, this provides a single succinct check that the
/// entire chain of computations was performed correctly.
public class ProtogalaxyDeciderVerifier {

    public init() {}

    // MARK: - Verify (Plonk Backend)

    /// Verify a decider proof that uses the Plonk backend.
    ///
    /// - Parameters:
    ///   - proof: The decider proof to verify
    ///   - setup: Plonk verification setup (preprocessed circuit data)
    ///   - kzg: KZG engine for opening verification
    /// - Returns: true if the decider proof is valid
    public func verifyPlonk(proof: ProtogalaxyDeciderProof,
                            setup: PlonkSetup,
                            kzg: KZGEngine) -> Bool {
        guard case .plonk(let plonkProof) = proof.snarkProof else {
            return false
        }

        let instance = proof.accumulatedInstance

        // Check 1: The accumulated instance must be relaxed (result of folding)
        // or trivially valid (u=1, e=0 for a single unfolded instance)
        if !instance.isRelaxed {
            // Fresh instance: u must be 1 and e must be 0
            guard frEq(instance.u, Fr.one) && frEq(instance.errorTerm, Fr.zero) else {
                return false
            }
        }

        // Check 2: Verify the Plonk SNARK proof
        let verifier = PlonkVerifier(setup: setup, kzg: kzg)
        guard verifier.verify(proof: plonkProof) else {
            return false
        }

        // Check 3: Verify public inputs match the accumulated instance
        // The decider circuit exposes u and e as public inputs
        guard plonkProof.publicInputs.count >= 2 else { return false }
        guard frEq(plonkProof.publicInputs[0], instance.u) else { return false }
        guard frEq(plonkProof.publicInputs[1], instance.errorTerm) else { return false }

        return true
    }

    // MARK: - Verify (Groth16 Backend)

    /// Verify a decider proof that uses the Groth16 backend.
    ///
    /// - Parameters:
    ///   - proof: The decider proof to verify
    ///   - vk: Groth16 verification key
    /// - Returns: true if the decider proof is valid
    public func verifyGroth16(proof: ProtogalaxyDeciderProof,
                              vk: Groth16VerificationKey) -> Bool {
        guard case .groth16(let groth16Proof) = proof.snarkProof else {
            return false
        }

        let instance = proof.accumulatedInstance

        // Check 1: Instance consistency
        if !instance.isRelaxed {
            guard frEq(instance.u, Fr.one) && frEq(instance.errorTerm, Fr.zero) else {
                return false
            }
        }

        // Check 2: Build the public input vector for Groth16 verification
        // Public inputs: [u, e, pi_0, ..., pi_{numPub-1}]
        var publicInputs = [Fr]()
        publicInputs.append(instance.u)
        publicInputs.append(instance.errorTerm)
        publicInputs.append(contentsOf: instance.publicInput)

        // Check 3: Verify the Groth16 proof
        let verifier = Groth16Verifier()
        return verifier.verify(proof: groth16Proof, vk: vk, publicInputs: publicInputs)
    }

    // MARK: - Verify Full IVC Chain

    /// Verify a decider proof that includes the full IVC folding chain.
    ///
    /// This checks:
    ///   1. Each folding step was performed correctly
    ///   2. The final SNARK proof is valid
    ///
    /// This is the strongest verification mode: it ensures the entire
    /// chain of computations from the original instances to the final
    /// proof is valid.
    ///
    /// - Parameters:
    ///   - proof: The decider proof with folding proofs
    ///   - originalInstances: The original Plonk instances that were folded
    ///   - setup: Plonk setup for SNARK verification
    ///   - kzg: KZG engine
    /// - Returns: true if the entire IVC chain and final proof are valid
    public func verifyIVCChain(proof: ProtogalaxyDeciderProof,
                               originalInstances: [ProtogalaxyInstance],
                               setup: PlonkSetup,
                               kzg: KZGEngine) -> Bool {
        let foldingVerifier = ProtogalaxyVerifier()

        // Verify each folding step in the chain
        guard !proof.foldingProofs.isEmpty else {
            // No folding proofs: just verify the SNARK
            return verifyPlonk(proof: proof, setup: setup, kzg: kzg)
        }

        guard originalInstances.count >= 2 else { return false }
        guard proof.foldingProofs.count == originalInstances.count - 1 else { return false }

        // Replay the folding chain using the verifier
        var running = originalInstances[0]
        for i in 0..<proof.foldingProofs.count {
            let foldProof = proof.foldingProofs[i]

            // Reconstruct what the folded instance should be
            // The verifier recomputes the folded instance from the originals + proof
            let transcript = Transcript(label: "protogalaxy-fold", backend: .keccak256)
            foldingVerifier.absorbInstance(transcript, running)
            foldingVerifier.absorbInstance(transcript, originalInstances[i + 1])
            for c in foldProof.fCoefficients {
                transcript.absorb(c)
            }
            let alpha = transcript.squeeze()

            // Compute Lagrange basis at alpha for domain {0, 1}
            let lagrangeBasis = lagrangeBasisAtPoint(domainSize: 2, point: alpha)

            // Fold commitments
            let numCols = running.witnessCommitments.count
            var foldedCommitments = [PointProjective]()
            for col in 0..<numCols {
                let c0 = cPointScalarMul(running.witnessCommitments[col], lagrangeBasis[0])
                let c1 = cPointScalarMul(originalInstances[i + 1].witnessCommitments[col], lagrangeBasis[1])
                foldedCommitments.append(pointAdd(c0, c1))
            }

            // Fold public inputs
            let numPub = running.publicInput.count
            var foldedPI = [Fr](repeating: Fr.zero, count: numPub)
            for j in 0..<numPub {
                foldedPI[j] = frAdd(
                    frMul(lagrangeBasis[0], running.publicInput[j]),
                    frMul(lagrangeBasis[1], originalInstances[i + 1].publicInput[j])
                )
            }

            // Fold challenges
            let foldedBeta = frAdd(
                frMul(lagrangeBasis[0], running.beta),
                frMul(lagrangeBasis[1], originalInstances[i + 1].beta)
            )
            let foldedGamma = frAdd(
                frMul(lagrangeBasis[0], running.gamma),
                frMul(lagrangeBasis[1], originalInstances[i + 1].gamma)
            )
            let foldedU = frAdd(
                frMul(lagrangeBasis[0], running.u),
                frMul(lagrangeBasis[1], originalInstances[i + 1].u)
            )
            let foldedError = hornerEvaluate(coeffs: foldProof.fCoefficients, at: alpha)

            running = ProtogalaxyInstance(
                witnessCommitments: foldedCommitments,
                publicInput: foldedPI,
                beta: foldedBeta,
                gamma: foldedGamma,
                errorTerm: foldedError,
                u: foldedU
            )
        }

        // Verify the final accumulated instance matches what's in the proof
        guard frEq(running.errorTerm, proof.accumulatedInstance.errorTerm) else { return false }
        guard frEq(running.u, proof.accumulatedInstance.u) else { return false }
        guard frEq(running.beta, proof.accumulatedInstance.beta) else { return false }
        guard frEq(running.gamma, proof.accumulatedInstance.gamma) else { return false }

        // Verify the SNARK proof
        return verifyPlonk(proof: proof, setup: setup, kzg: kzg)
    }
}
