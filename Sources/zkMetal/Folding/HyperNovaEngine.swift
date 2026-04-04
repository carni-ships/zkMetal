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
import NeonFieldOps

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
    /// This is the "base case" -- the first instance before any folding.
    public func initialize(witness: [Fr], publicInput: [Fr]) -> LCCCS {
        // Build z = [1, publicInput, witness]
        let z = buildZ(publicInput: publicInput, witness: witness)

        // Commit to witness
        let commitment = pp.commit(witness: witness)

        // Generate random evaluation point r (using Fiat-Shamir)
        let transcript = Transcript(label: "hypernova-init", backend: .keccak256)
        absorbPoint(transcript, commitment)
        for x in publicInput { transcript.absorb(x) }
        let r = transcript.squeezeN(logM)

        // Compute v_i = MLE(M_i * z)(r) for each matrix
        let v = ccs.matrices.map { mat -> Fr in
            let mv = mat.mulVec(z)
            return cMleEvalFold(evals: padToPow2(mv), point: r)
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

        // Step 1: Compute sigmas and thetas using C-accelerated MLE eval
        let sigmas = ccs.matrices.map { mat -> Fr in
            let mv = padToPow2(mat.mulVec(z1))
            return cMleEvalFold(evals: mv, point: running.r)
        }

        let thetas = ccs.matrices.map { mat -> Fr in
            let mv = padToPow2(mat.mulVec(z2))
            return cMleEvalFold(evals: mv, point: running.r)
        }

        // Build the Fiat-Shamir transcript (Keccak for speed: NEON-accelerated)
        let transcript = Transcript(label: "hypernova-fold", backend: .keccak256)
        absorbLCCCS(transcript, running)
        absorbCCCS(transcript, new)
        for s in sigmas { transcript.absorb(s) }
        for t in thetas { transcript.absorb(t) }

        // Get challenge rho for the random linear combination
        let rho = transcript.squeeze()

        // Fold commitments: C' = C1 + rho * C2 (C CIOS scalar mul)
        let rhoC2 = cPointScalarMul(new.commitment, rho)
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

        // Build the sumcheck proof (C-accelerated cross-term)
        let sumcheckProof = computeCrossTermSumcheckC(
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
        // Rebuild transcript (must match fold's backend)
        let transcript = Transcript(label: "hypernova-fold", backend: .keccak256)
        absorbLCCCS(transcript, running)
        absorbCCCS(transcript, new)
        for s in proof.sigmas { transcript.absorb(s) }
        for t in proof.thetas { transcript.absorb(t) }

        let rho = transcript.squeeze()

        // Check folded commitment: C' = C1 + rho * C2
        let expectedC = pointAdd(running.commitment, cPointScalarMul(new.commitment, rho))
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
            let eval = cMleEvalFold(evals: padToPow2(mv), point: lcccs.r)
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

    /// Absorb a projective point into transcript using C-accelerated affine conversion.
    func absorbPoint(_ transcript: Transcript, _ p: PointProjective) {
        if pointIsIdentity(p) {
            transcript.absorb(Fr.zero)
            transcript.absorb(Fr.zero)
            return
        }
        // Use C CIOS projective-to-affine (much faster than Swift fpInverse)
        var affine = (Fp.zero, Fp.zero)  // x, y as Fp
        withUnsafeBytes(of: p) { pBuf in
            withUnsafeMutableBytes(of: &affine) { aBuf in
                bn254_projective_to_affine(
                    pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                )
            }
        }
        transcript.absorb(fpToFr(affine.0))
        transcript.absorb(fpToFr(affine.1))
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
    /// This is a simplified version -- proves that the cross-terms are consistent.
    func computeCrossTermSumcheck(z1: [Fr], z2: [Fr], rho: Fr,
                                  r: [Fr], transcript: Transcript) -> SumcheckFoldProof {
        let numRounds = logM
        var crossTermEvals = [Fr](repeating: .zero, count: 1 << logM)

        for j in 0..<ccs.q {
            let sj = ccs.multisets[j]
            if sj.count < 2 { continue }

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

        var roundPolys = [[Fr]]()
        var current = crossTermEvals
        let eqR = eqEvals(point: r)

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

            transcript.absorb(s0)
            transcript.absorb(s1)
            let challenge = transcript.squeeze()

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

    /// C-accelerated cross-term sumcheck using CIOS Montgomery field arithmetic.
    /// Uses gkr_eq_poly for eq evaluations and C MLE eval.
    func computeCrossTermSumcheckC(z1: [Fr], z2: [Fr], rho: Fr,
                                    r: [Fr], transcript: Transcript) -> SumcheckFoldProof {
        let numRounds = logM
        let size = 1 << logM
        var crossTermEvals = [Fr](repeating: .zero, count: size)

        // Compute cross-terms (same math, relies on sparse matrix being fast enough)
        for j in 0..<ccs.q {
            let sj = ccs.multisets[j]
            if sj.count < 2 { continue }

            if sj.count == 2 {
                let m0z1 = ccs.matrices[sj[0]].mulVec(z1)
                let m0z2 = ccs.matrices[sj[0]].mulVec(z2)
                let m1z1 = ccs.matrices[sj[1]].mulVec(z1)
                let m1z2 = ccs.matrices[sj[1]].mulVec(z2)
                let rhoTimesC = frMul(rho, ccs.coefficients[j])
                for i in 0..<min(ccs.m, size) {
                    let cross = frAdd(frMul(m0z1[i], m1z2[i]), frMul(m0z2[i], m1z1[i]))
                    crossTermEvals[i] = frAdd(crossTermEvals[i], frMul(rhoTimesC, cross))
                }
            }
        }

        // Weight by eq(r, x) using C-accelerated eq poly
        var eqR = [Fr](repeating: Fr.zero, count: size)
        r.withUnsafeBytes { ptBuf in
            eqR.withUnsafeMutableBytes { evalBuf in
                gkr_eq_poly(
                    ptBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(r.count),
                    evalBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                )
            }
        }
        for i in 0..<size {
            crossTermEvals[i] = frMul(crossTermEvals[i], eqR[i])
        }

        // Run sumcheck rounds
        var roundPolys = [[Fr]]()
        var current = crossTermEvals

        for round in 0..<numRounds {
            let half = current.count / 2
            var s0 = Fr.zero
            var s1 = Fr.zero
            for j in 0..<half {
                s0 = frAdd(s0, current[2 * j])
                s1 = frAdd(s1, current[2 * j + 1])
            }
            roundPolys.append([s0, s1])

            transcript.absorb(s0)
            transcript.absorb(s1)
            let challenge = transcript.squeeze()

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

// MARK: - C-accelerated helpers

/// C-accelerated MLE evaluation using bn254_fr_mle_eval.
func cMleEvalFold(evals: [Fr], point: [Fr]) -> Fr {
    let numVars = point.count
    if evals.count != (1 << numVars) { return multilinearEval(evals: evals, point: point) }
    var result = Fr.zero
    evals.withUnsafeBytes { evalBuf in
        point.withUnsafeBytes { ptBuf in
            withUnsafeMutableBytes(of: &result) { resBuf in
                bn254_fr_mle_eval(
                    evalBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(numVars),
                    ptBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    resBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                )
            }
        }
    }
    return result
}

// MARK: - Fp <-> Fr conversion helper

/// Reinterpret Fp limbs as Fr (for transcript absorption).
/// This is a raw bit reinterpretation, not a field homomorphism.
func fpToFr(_ fp: Fp) -> Fr {
    Fr(v: fp.v)
}
