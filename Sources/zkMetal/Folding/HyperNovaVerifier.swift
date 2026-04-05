// HyperNova Verifier — Lightweight fold verification
//
// The verifier checks that a folding step was performed correctly WITHOUT
// seeing witnesses. This is the key efficiency win: the verifier only does
// O(t) field ops + 1 MSM (commitment homomorphism), no sumcheck execution.
//
// For N-instance multi-fold, the verifier cost is O(N * t) field ops + N MSMs.
//
// Reference: "HyperNova: Recursive arguments from folding schemes" (Kothapalli, Setty 2023)

import Foundation
import NeonFieldOps

// MARK: - HyperNova Verifier

public class HyperNovaVerifier {
    public let ccs: CCSInstance
    public let logM: Int

    /// Initialize verifier with CCS structure (no SRS needed).
    public init(ccs: CCSInstance) {
        self.ccs = ccs
        var log = 0
        while (1 << log) < ccs.m { log += 1 }
        self.logM = log
    }

    /// Initialize verifier from an engine (shares CCS reference).
    public init(engine: HyperNovaEngine) {
        self.ccs = engine.ccs
        self.logM = engine.logM
    }

    // MARK: - Verify 2-Instance Fold

    /// Verify a 2-instance fold (LCCCS + CCCS -> LCCCS).
    ///
    /// Checks:
    ///   1. C' = C_running + rho * C_new  (commitment homomorphism)
    ///   2. u' = u_running + rho
    ///   3. x'_i = x_running_i + rho * x_new_i  (public input linearity)
    ///   4. v'_j = sigma_j + rho * theta_j  (MLE evaluation consistency)
    ///   5. r' = r  (evaluation point preserved)
    ///
    /// The verifier reconstructs rho from the transcript (Fiat-Shamir).
    public func verifyFold(running: CommittedCCSInstance, new: CommittedCCSInstance,
                           folded: CommittedCCSInstance, proof: FoldingProof) -> Bool {
        precondition(running.isRelaxed, "Running instance must be relaxed")

        // Rebuild transcript (must match prover's)
        let transcript = Transcript(label: "hypernova-fold", backend: .keccak256)
        absorbLCCCS(transcript, running)
        absorbCCCS(transcript, new)
        for s in proof.sigmas { transcript.absorb(s) }
        for th in proof.thetas { transcript.absorb(th) }

        let rho = transcript.squeeze()

        // Check 1: Commitment homomorphism
        let expectedC = pointAdd(running.commitment, cPointScalarMul(new.commitment, rho))
        guard pointEqual(folded.commitment, expectedC) else { return false }

        // Check 2: Relaxation scalar
        guard frEq(folded.u, frAdd(running.u, rho)) else { return false }

        // Check 3: Public input linearity
        for i in 0..<running.publicInput.count {
            let expected = frAdd(running.publicInput[i], frMul(rho, new.publicInput[i]))
            guard frEq(folded.publicInput[i], expected) else { return false }
        }

        // Check 4: MLE evaluation consistency
        for i in 0..<proof.sigmas.count {
            let expected = frAdd(proof.sigmas[i], frMul(rho, proof.thetas[i]))
            guard frEq(folded.v[i], expected) else { return false }
        }

        // Check 5: Evaluation point preserved
        guard folded.r.count == running.r.count else { return false }
        for i in 0..<running.r.count {
            guard frEq(folded.r[i], running.r[i]) else { return false }
        }

        return true
    }

    // MARK: - Verify Multi-Instance Fold

    /// Verify a multi-instance fold (N instances -> 1).
    ///
    /// The verifier:
    ///   1. Reconstructs challenges rho_1, ..., rho_{N-1} from transcript
    ///   2. Checks C' = C_0 + sum_i rho_i * C_i
    ///   3. Checks u' = u_0 + sum_i rho_i
    ///   4. Checks x'_k = x_0_k + sum_i rho_i * x_i_k
    ///   5. Checks v'_j = sigma_0_j + sum_i rho_i * sigma_i_j
    ///   6. Checks r is preserved
    ///
    /// Cost: O(N * t) field multiplications + N point scalar multiplications
    public func verifyMultiFold(instances: [CommittedCCSInstance],
                                folded: CommittedCCSInstance,
                                proof: MultiFoldProof) -> Bool {
        let n = proof.instanceCount
        precondition(instances.count == n, "Instance count mismatch")
        precondition(instances[0].isRelaxed, "First instance must be relaxed")

        let t = ccs.t

        // Rebuild transcript
        let transcript = Transcript(label: "hypernova-multifold", backend: .keccak256)
        absorbLCCCS(transcript, instances[0])
        for i in 1..<n {
            absorbCCCS(transcript, instances[i])
        }

        // Absorb all evaluations (sigmas for running, thetas for new)
        // The prover absorbs allSigmas[i][j] for all i, j
        let allEvals: [[Fr]]
        if proof.sigmas.count == 1 && proof.thetas.count == n - 1 {
            allEvals = proof.sigmas + proof.thetas
        } else {
            return false
        }
        for evals in allEvals {
            for v in evals {
                transcript.absorb(v)
            }
        }

        // Squeeze N-1 challenges
        let rhos = transcript.squeezeN(n - 1)

        let running = instances[0]

        // Check 1: Commitment homomorphism
        var expectedC = running.commitment
        for i in 1..<n {
            expectedC = pointAdd(expectedC, cPointScalarMul(instances[i].commitment, rhos[i - 1]))
        }
        guard pointEqual(folded.commitment, expectedC) else { return false }

        // Check 2: Relaxation scalar
        var expectedU = running.u
        for i in 0..<(n - 1) {
            expectedU = frAdd(expectedU, rhos[i])
        }
        guard frEq(folded.u, expectedU) else { return false }

        // Check 3: Public input linearity
        let numPub = running.publicInput.count
        for k in 0..<numPub {
            var expected = running.publicInput[k]
            for i in 1..<n {
                expected = frAdd(expected, frMul(rhos[i - 1], instances[i].publicInput[k]))
            }
            guard frEq(folded.publicInput[k], expected) else { return false }
        }

        // Check 4: MLE evaluation consistency
        // v'_j = allEvals[0][j] + sum_i rho_i * allEvals[i][j]
        for j in 0..<t {
            var expected = allEvals[0][j]
            for i in 1..<n {
                expected = frAdd(expected, frMul(rhos[i - 1], allEvals[i][j]))
            }
            guard frEq(folded.v[j], expected) else { return false }
        }

        // Check 5: Evaluation point preserved
        guard folded.r.count == running.r.count else { return false }
        for i in 0..<running.r.count {
            guard frEq(folded.r[i], running.r[i]) else { return false }
        }

        return true
    }

    // MARK: - Transcript Helpers

    /// Absorb a relaxed instance (LCCCS) into the transcript.
    func absorbLCCCS(_ transcript: Transcript, _ instance: CommittedCCSInstance) {
        transcript.absorbLabel("lcccs")
        if let ax = instance.cachedAffineX, let ay = instance.cachedAffineY {
            transcript.absorb(ax)
            transcript.absorb(ay)
        } else {
            absorbPoint(transcript, instance.commitment)
        }
        transcript.absorb(instance.u)
        for x in instance.publicInput { transcript.absorb(x) }
        for r in instance.r { transcript.absorb(r) }
        for v in instance.v { transcript.absorb(v) }
    }

    /// Absorb a fresh instance (CCCS) into the transcript.
    func absorbCCCS(_ transcript: Transcript, _ instance: CommittedCCSInstance) {
        transcript.absorbLabel("cccs")
        if let ax = instance.cachedAffineX, let ay = instance.cachedAffineY {
            transcript.absorb(ax)
            transcript.absorb(ay)
        } else {
            absorbPoint(transcript, instance.commitment)
        }
        for x in instance.publicInput { transcript.absorb(x) }
    }

    /// Absorb a projective point using C-accelerated affine conversion.
    func absorbPoint(_ transcript: Transcript, _ p: PointProjective) {
        if pointIsIdentity(p) {
            transcript.absorb(Fr.zero)
            transcript.absorb(Fr.zero)
            return
        }
        var affine = (Fp.zero, Fp.zero)
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
}
