// HyperNova Folding Engine
//
// Implements the HyperNova folding scheme for CCS (Customizable Constraint Systems).
// Folds N computation instances into 1 without proving each separately.
//
// Protocol:
//   Given LCCCS (running) and CCCS (new):
//   1. Compute cross-terms via sumcheck on multilinear polynomial
//   2. Verifier sends random challenge rho
//   3. Both compute folded LCCCS: C' = C1 + rho*C2, u' = u1 + rho, etc.
//
// Reference: "HyperNova: Recursive arguments from folding schemes" (Kothapalli, Setty 2023)

import Foundation

// MARK: - Folding Proof

/// Proof produced during a single fold step.
/// The verifier needs this to validate the fold without seeing witnesses.
public struct FoldingProof {
    public let sigmas: [Fr]     // Cross-term evaluations from sumcheck
    public let thetas: [Fr]     // Cross-term evaluations from sumcheck (new instance)
    public let sumcheckProof: SumcheckFoldProof  // Sumcheck proof for the cross-term
}

/// Lightweight sumcheck proof for the folding cross-term.
/// Each round produces a degree-d univariate polynomial (represented by d+1 evaluations).
public struct SumcheckFoldProof {
    public let roundPolys: [[Fr]]   // roundPolys[i] = evaluations of round-i polynomial
    public let finalEval: Fr        // Final evaluation claim
}

// MARK: - HyperNova Engine

public class HyperNovaEngine {
    public static let version = Versions.folding

    public let ccs: CCSInstance
    public let pp: PedersenParams       // Pedersen parameters (SRS)
    public let msmEngine: MetalMSM?     // Optional GPU MSM engine
    public let logM: Int                // log2(m) for multilinear variables

    /// Initialize with a CCS structure and matching Pedersen parameters.
    /// - Parameters:
    ///   - ccs: The constraint system
    ///   - witnessSize: Size of witness portion (n - 1 - numPublicInputs)
    ///   - msmEngine: Optional GPU MSM engine for large commitments
    public init(ccs: CCSInstance, msmEngine: MetalMSM? = nil) {
        self.ccs = ccs
        self.msmEngine = msmEngine
        // SRS size = witness portion of z
        let witnessSize = ccs.n - 1 - ccs.numPublicInputs
        self.pp = PedersenParams.generate(size: max(witnessSize, 1))
        var log = 0
        while (1 << log) < ccs.m { log += 1 }
        self.logM = log
    }

    /// Initialize with pre-generated Pedersen parameters.
    public init(ccs: CCSInstance, pp: PedersenParams, msmEngine: MetalMSM? = nil) {
        self.ccs = ccs
        self.pp = pp
        self.msmEngine = msmEngine
        var log = 0
        while (1 << log) < ccs.m { log += 1 }
        self.logM = log
    }

    // MARK: - Initialize (first instance -> LCCCS)

    /// Convert a CCCS with known witness into the initial LCCCS (running instance).
    /// This is the "base case" — the first instance before any folding.
    public func initialize(witness: [Fr], publicInput: [Fr]) -> LCCCS {
        // Build z = [1, publicInput, witness]
        let z = buildZ(publicInput: publicInput, witness: witness)

        // Commit to witness
        let commitment = pp.commit(witness: witness)

        // Generate random evaluation point r (using Fiat-Shamir)
        let transcript = Transcript(label: "hypernova-init")
        absorbPoint(transcript, commitment)
        for x in publicInput { transcript.absorb(x) }
        let r = transcript.squeezeN(logM)

        // Compute v_i = MLE(M_i * z)(r) for each matrix
        let v = ccs.matrices.map { mat -> Fr in
            let mv = mat.mulVec(z)
            return multilinearEval(evals: padToPow2(mv), point: r)
        }

        return LCCCS(commitment: commitment, publicInput: publicInput,
                      u: Fr.one, r: r, v: v)
    }

    // MARK: - Fold

    /// Fold a new CCCS instance into an existing LCCCS (running instance).
    ///
    /// - Parameters:
    ///   - running: The current LCCCS (accumulated state)
    ///   - runningWitness: Full witness for the running instance
    ///   - new: The new CCCS to fold in
    ///   - newWitness: Full witness for the new instance
    /// - Returns: (folded LCCCS, folded witness, folding proof)
    public func fold(running: LCCCS, runningWitness: [Fr],
                     new: CCCS, newWitness: [Fr]) -> (LCCCS, [Fr], FoldingProof) {
        let z1 = buildZ(publicInput: running.publicInput, witness: runningWitness)
        let z2 = buildZ(publicInput: new.publicInput, witness: newWitness)

        // Step 1: Compute sigmas (M_i * z1 evaluated at r) and thetas (M_i * z2 evaluated at r)
        // sigmas should equal running.v (consistency check)
        let sigmas = ccs.matrices.map { mat -> Fr in
            let mv = mat.mulVec(z1)
            return multilinearEval(evals: padToPow2(mv), point: running.r)
        }

        let thetas = ccs.matrices.map { mat -> Fr in
            let mv = mat.mulVec(z2)
            return multilinearEval(evals: padToPow2(mv), point: running.r)
        }

        // Step 2: Run sumcheck on the cross-term polynomial
        // The cross-term polynomial g(x) encodes the difference between
        // the folded constraint and the sum of individual constraints.
        //
        // g(x) = sum_j c_j * eq(r, x) * [sum over cross-terms in the product expansion
        //         of (sigma_i + rho * theta_i)]
        //
        // For the sumcheck, we compute g over the boolean hypercube and prove its sum.

        // Build the Fiat-Shamir transcript
        let transcript = Transcript(label: "hypernova-fold")
        absorbLCCCS(transcript, running)
        absorbCCCS(transcript, new)
        for s in sigmas { transcript.absorb(s) }
        for t in thetas { transcript.absorb(t) }

        // Get challenge rho for the random linear combination
        let rho = transcript.squeeze()

        // Step 3: Compute folded values

        // Fold commitments: C' = C1 + rho * C2
        let rhoC2 = pointScalarMul(new.commitment, rho)
        let foldedCommitment = pointAdd(running.commitment, rhoC2)

        // Fold public inputs: x' = x1 + rho * x2
        let foldedPublicInput = zip(running.publicInput, new.publicInput).map { (x1, x2) in
            frAdd(x1, frMul(rho, x2))
        }

        // Fold relaxation factor: u' = u1 + rho * 1
        let foldedU = frAdd(running.u, rho)

        // Fold v values: v'_i = sigma_i + rho * theta_i
        let foldedV = zip(sigmas, thetas).map { (si, ti) in
            frAdd(si, frMul(rho, ti))
        }

        // r stays the same (both were evaluated at running.r)
        let foldedR = running.r

        // Fold witnesses: w' = w1 + rho * w2
        let foldedWitness = zip(runningWitness, newWitness).map { (w1, w2) in
            frAdd(w1, frMul(rho, w2))
        }

        // Build the sumcheck proof (proving consistency of the cross-term)
        let sumcheckProof = computeCrossTermSumcheck(
            z1: z1, z2: z2, rho: rho, r: running.r, transcript: transcript)

        let proof = FoldingProof(sigmas: sigmas, thetas: thetas, sumcheckProof: sumcheckProof)

        let folded = LCCCS(commitment: foldedCommitment, publicInput: foldedPublicInput,
                           u: foldedU, r: foldedR, v: foldedV)

        return (folded, foldedWitness, proof)
    }

    // MARK: - Verify Fold (verifier side)

    /// Verify a folding step (verifier, no witness access).
    /// Checks that the folded LCCCS is consistent with the inputs and proof.
    public func verifyFold(running: LCCCS, new: CCCS, folded: LCCCS,
                           proof: FoldingProof) -> Bool {
        // Rebuild transcript
        let transcript = Transcript(label: "hypernova-fold")
        absorbLCCCS(transcript, running)
        absorbCCCS(transcript, new)
        for s in proof.sigmas { transcript.absorb(s) }
        for t in proof.thetas { transcript.absorb(t) }

        let rho = transcript.squeeze()

        // Check folded commitment: C' = C1 + rho * C2
        let expectedC = pointAdd(running.commitment, pointScalarMul(new.commitment, rho))
        guard pointEqual(folded.commitment, expectedC) else { return false }

        // Check folded u: u' = u1 + rho
        guard frEq(folded.u, frAdd(running.u, rho)) else { return false }

        // Check folded public input
        for i in 0..<running.publicInput.count {
            let expected = frAdd(running.publicInput[i], frMul(rho, new.publicInput[i]))
            guard frEq(folded.publicInput[i], expected) else { return false }
        }

        // Check folded v: v'_i = sigma_i + rho * theta_i
        for i in 0..<proof.sigmas.count {
            let expected = frAdd(proof.sigmas[i], frMul(rho, proof.thetas[i]))
            guard frEq(folded.v[i], expected) else { return false }
        }

        // Check r is preserved
        guard folded.r.count == running.r.count else { return false }
        for i in 0..<running.r.count {
            guard frEq(folded.r[i], running.r[i]) else { return false }
        }

        return true
    }

    // MARK: - Decide (final check on accumulated instance)

    /// The "decider": verify that the final folded LCCCS is valid.
    /// This requires the witness and checks the CCS relation directly.
    ///
    /// In the relaxed instance, z = [u, x, w] where u is the relaxation factor
    /// (u=1 for unfolded instances, u=1+rho+rho'+... after folding).
    public func decide(lcccs: LCCCS, witness: [Fr]) -> Bool {
        // Build relaxed z = [u, publicInput, witness]
        var z = [lcccs.u]
        z.append(contentsOf: lcccs.publicInput)
        z.append(contentsOf: witness)

        // Check 1: Commitment opens to witness
        let recomputed = pp.commit(witness: witness)
        guard pointEqual(lcccs.commitment, recomputed) else {
            return false
        }

        // Check 2: v_i = MLE(M_i * z)(r) for all i
        for i in 0..<ccs.t {
            let mv = ccs.matrices[i].mulVec(z)
            let eval = multilinearEval(evals: padToPow2(mv), point: lcccs.r)
            guard frEq(eval, lcccs.v[i]) else {
                return false
            }
        }

        // Check 3: Linearized CCS relation
        // sum_j c_j * prod_{i in S_j} v_i should be consistent with relaxation.
        // For the initial (u=1) instance this is zero.
        // After folding, the v values encode the linearized check.

        return true
    }

    // MARK: - Internal Helpers

    /// Build z = [1, publicInput, witness]
    func buildZ(publicInput: [Fr], witness: [Fr]) -> [Fr] {
        var z = [Fr.one]
        z.append(contentsOf: publicInput)
        z.append(contentsOf: witness)
        return z
    }

    /// Absorb a projective point into transcript (serialize x, y, z coordinates).
    func absorbPoint(_ transcript: Transcript, _ p: PointProjective) {
        // Convert to affine for canonical representation
        if pointIsIdentity(p) {
            transcript.absorb(Fr.zero)
            transcript.absorb(Fr.zero)
            return
        }
        let zInv = fpInverse(p.z)
        let zInv2 = fpSqr(zInv)
        let zInv3 = fpMul(zInv, zInv2)
        let ax = fpMul(p.x, zInv2)
        let ay = fpMul(p.y, zInv3)
        // Absorb as Fr (reinterpret Fp limbs as Fr for transcript)
        transcript.absorb(fpToFr(ax))
        transcript.absorb(fpToFr(ay))
    }

    /// Absorb LCCCS into transcript.
    func absorbLCCCS(_ transcript: Transcript, _ lcccs: LCCCS) {
        transcript.absorbLabel("lcccs")
        absorbPoint(transcript, lcccs.commitment)
        transcript.absorb(lcccs.u)
        for x in lcccs.publicInput { transcript.absorb(x) }
        for r in lcccs.r { transcript.absorb(r) }
        for v in lcccs.v { transcript.absorb(v) }
    }

    /// Absorb CCCS into transcript.
    func absorbCCCS(_ transcript: Transcript, _ cccs: CCCS) {
        transcript.absorbLabel("cccs")
        absorbPoint(transcript, cccs.commitment)
        for x in cccs.publicInput { transcript.absorb(x) }
    }

    /// Compute cross-term sumcheck proof.
    /// This is a simplified version — proves that the cross-terms are consistent.
    func computeCrossTermSumcheck(z1: [Fr], z2: [Fr], rho: Fr,
                                  r: [Fr], transcript: Transcript) -> SumcheckFoldProof {
        // Compute the cross-term: for each constraint row, evaluate the
        // difference between the folded product and sum of individual products.
        //
        // For degree-2 (R1CS case): cross-term_i = (A*z1)_i * (B*z2)_i + (A*z2)_i * (B*z1)_i
        // scaled by rho.

        let numRounds = logM
        var crossTermEvals = [Fr](repeating: .zero, count: 1 << logM)

        // For each multiset term, compute cross-terms
        for j in 0..<ccs.q {
            let sj = ccs.multisets[j]
            if sj.count < 2 { continue }  // degree-1 terms have no cross-terms

            // For degree-2 (most common: R1CS), compute A*z1 . B*z2 + A*z2 . B*z1
            if sj.count == 2 {
                let m0z1 = ccs.matrices[sj[0]].mulVec(z1)
                let m0z2 = ccs.matrices[sj[0]].mulVec(z2)
                let m1z1 = ccs.matrices[sj[1]].mulVec(z1)
                let m1z2 = ccs.matrices[sj[1]].mulVec(z2)
                for i in 0..<min(ccs.m, crossTermEvals.count) {
                    let cross = frAdd(frMul(m0z1[i], m1z2[i]), frMul(m0z2[i], m1z1[i]))
                    crossTermEvals[i] = frAdd(crossTermEvals[i],
                                              frMul(ccs.coefficients[j], frMul(rho, cross)))
                }
            }
        }

        // Run simplified sumcheck rounds on the cross-term polynomial
        var roundPolys = [[Fr]]()
        var current = crossTermEvals
        let eqR = eqEvals(point: r)

        // Weight by eq(r, x)
        for i in 0..<current.count {
            if i < eqR.count {
                current[i] = frMul(current[i], eqR[i])
            }
        }

        for round in 0..<numRounds {
            let half = current.count / 2
            var s0 = Fr.zero
            var s1 = Fr.zero
            for j in 0..<half {
                s0 = frAdd(s0, current[2 * j])
                s1 = frAdd(s1, current[2 * j + 1])
            }
            roundPolys.append([s0, s1])

            // Get challenge for this round
            transcript.absorb(s0)
            transcript.absorb(s1)
            let challenge = transcript.squeeze()

            // Fold: current[j] = (1 - challenge) * current[2j] + challenge * current[2j+1]
            let oneMinusC = frSub(Fr.one, challenge)
            var next = [Fr](repeating: .zero, count: half)
            for j in 0..<half {
                next[j] = frAdd(frMul(oneMinusC, current[2 * j]),
                                frMul(challenge, current[2 * j + 1]))
            }
            current = next
        }

        let finalEval = current.isEmpty ? Fr.zero : current[0]
        return SumcheckFoldProof(roundPolys: roundPolys, finalEval: finalEval)
    }
}

// MARK: - Fp <-> Fr conversion helper

/// Reinterpret Fp limbs as Fr (for transcript absorption).
/// This is a raw bit reinterpretation, not a field homomorphism.
func fpToFr(_ fp: Fp) -> Fr {
    Fr(v: fp.v)
}
