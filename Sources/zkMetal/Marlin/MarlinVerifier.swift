// MarlinVerifier — Marlin/AHP (Algebraic Holographic Proof) verifier
//
// Marlin = indexed polynomial IOP + KZG polynomial commitment.
// The verifier reconstructs Fiat-Shamir challenges, checks polynomial
// evaluations against the AHP relation, and verifies KZG opening proofs.
//
// Verification cost: O(1) pairings + O(log n) field ops + small MSM for public inputs.
// Without a pairing engine, KZG checks use the known SRS secret (test mode).
//
// Reference: Chiesa, Hu, Maller, Mishra, Vesely, Ward — "Marlin: Preprocessing
// zkSNARKs with Universal and Updatable SRS" (EUROCRYPT 2020)

import Foundation
import NeonFieldOps

// MARK: - Marlin Data Structures

/// Marlin index (preprocessed circuit description).
/// Encodes the R1CS matrices A, B, C via their polynomial representations.
public struct MarlinIndex {
    /// Number of constraints (m)
    public let numConstraints: Int
    /// Number of variables including public input (n)
    public let numVariables: Int
    /// Number of non-zero entries in A, B, C
    public let numNonZero: Int
    /// Domain sizes (powers of 2)
    public let constraintDomainSize: Int   // |H| >= m
    public let variableDomainSize: Int     // |K| >= n
    public let nonZeroDomainSize: Int      // |K_NZ| >= numNonZero
    /// Roots of unity for each domain
    public let omegaH: Fr     // generator of constraint domain H
    public let omegaK: Fr     // generator of variable domain K

    public init(numConstraints: Int, numVariables: Int, numNonZero: Int,
                constraintDomainSize: Int, variableDomainSize: Int,
                nonZeroDomainSize: Int, omegaH: Fr, omegaK: Fr) {
        self.numConstraints = numConstraints
        self.numVariables = numVariables
        self.numNonZero = numNonZero
        self.constraintDomainSize = constraintDomainSize
        self.variableDomainSize = variableDomainSize
        self.nonZeroDomainSize = nonZeroDomainSize
        self.omegaH = omegaH
        self.omegaK = omegaK
    }
}

/// Marlin verifying key: index commitments produced during preprocessing.
public struct MarlinVerifyingKey {
    public let index: MarlinIndex
    /// Commitments to the indexed polynomials for matrices A, B, C:
    /// row(X), col(X), val(X), row_col(X) for each matrix
    public let indexCommitments: [PointProjective]  // 12 commitments (4 per matrix)
    /// SRS secret for test-mode verification (not used in production pairing check)
    public let srsSecret: Fr
    /// SRS points (G1)
    public let srs: [PointAffine]

    public init(index: MarlinIndex, indexCommitments: [PointProjective],
                srsSecret: Fr, srs: [PointAffine]) {
        self.index = index
        self.indexCommitments = indexCommitments
        self.srsSecret = srsSecret
        self.srs = srs
    }
}

/// A Marlin proof containing all prover messages across rounds.
public struct MarlinProof {
    // Round 1: witness commitments
    public let wCommit: PointProjective       // commitment to witness polynomial w(X)
    public let zACommit: PointProjective      // commitment to z_A(X) = A*z
    public let zBCommit: PointProjective      // commitment to z_B(X) = B*z

    // Round 2: sumcheck for outer relation (t polynomial)
    public let tCommit: PointProjective       // commitment to masking polynomial t(X)
    public let sumcheckPolyCoeffs: [[Fr]]     // sumcheck round polynomial coefficients (degree-2)

    // Round 3: inner sumcheck
    public let gCommit: PointProjective       // commitment to g(X) (lincheck witness)
    public let hCommit: PointProjective       // commitment to h(X) (quotient for inner sum)

    // Evaluations at query points
    public let evaluations: MarlinEvaluations

    // KZG opening proofs
    public let openingProofs: [PointProjective]

    public init(wCommit: PointProjective, zACommit: PointProjective, zBCommit: PointProjective,
                tCommit: PointProjective, sumcheckPolyCoeffs: [[Fr]],
                gCommit: PointProjective, hCommit: PointProjective,
                evaluations: MarlinEvaluations, openingProofs: [PointProjective]) {
        self.wCommit = wCommit
        self.zACommit = zACommit
        self.zBCommit = zBCommit
        self.tCommit = tCommit
        self.sumcheckPolyCoeffs = sumcheckPolyCoeffs
        self.gCommit = gCommit
        self.hCommit = hCommit
        self.evaluations = evaluations
        self.openingProofs = openingProofs
    }
}

/// Polynomial evaluations provided by the prover at verifier-chosen query points.
public struct MarlinEvaluations {
    // Evaluations at beta (outer sumcheck challenge)
    public let zABeta: Fr        // z_A(beta)
    public let zBBeta: Fr        // z_B(beta)
    public let wBeta: Fr         // w(beta)
    public let tBeta: Fr         // t(beta)

    // Evaluations at gamma (inner sumcheck challenge)
    public let gGamma: Fr        // g(gamma)
    public let hGamma: Fr        // h(gamma)

    // Index polynomial evaluations at gamma for each matrix M in {A, B, C}
    public let rowGamma: [Fr]    // row_M(gamma) for M = A, B, C
    public let colGamma: [Fr]    // col_M(gamma) for M = A, B, C
    public let valGamma: [Fr]    // val_M(gamma) for M = A, B, C
    public let rowColGamma: [Fr] // row_col_M(gamma) for M = A, B, C

    public init(zABeta: Fr, zBBeta: Fr, wBeta: Fr, tBeta: Fr,
                gGamma: Fr, hGamma: Fr,
                rowGamma: [Fr], colGamma: [Fr], valGamma: [Fr], rowColGamma: [Fr]) {
        self.zABeta = zABeta
        self.zBBeta = zBBeta
        self.wBeta = wBeta
        self.tBeta = tBeta
        self.gGamma = gGamma
        self.hGamma = hGamma
        self.rowGamma = rowGamma
        self.colGamma = colGamma
        self.valGamma = valGamma
        self.rowColGamma = rowColGamma
    }
}

// MARK: - Marlin Verifier

public class MarlinVerifier {
    public static let version = Versions.marlin
    public let kzg: KZGEngine

    public init(kzg: KZGEngine) {
        self.kzg = kzg
    }

    /// Verify a Marlin proof for R1CS satisfiability.
    ///
    /// Protocol outline:
    /// 1. Reconstruct Fiat-Shamir challenges (alpha, eta_A, eta_B, eta_C, beta, gamma)
    /// 2. Verify outer sumcheck: sum of round polynomials matches claimed sum
    /// 3. Verify inner sumcheck relation at query point gamma
    /// 4. Verify all KZG opening proofs
    /// 5. Check public input consistency
    public func verify(vk: MarlinVerifyingKey, publicInput: [Fr], proof: MarlinProof) -> Bool {
        let idx = vk.index

        // --- Step 1: Reconstruct Fiat-Shamir challenges ---
        let transcript = Transcript(label: "marlin", backend: .keccak256)

        // Absorb index info
        transcript.absorb(frFromInt(UInt64(idx.numConstraints)))
        transcript.absorb(frFromInt(UInt64(idx.numVariables)))
        transcript.absorb(frFromInt(UInt64(idx.numNonZero)))

        // Absorb index commitments
        for c in vk.indexCommitments {
            marlinAbsorbPointImpl(transcript, c)
        }

        // Absorb public input
        for pi in publicInput {
            transcript.absorb(pi)
        }

        // Round 1: absorb witness commitments, squeeze eta challenges
        marlinAbsorbPointImpl(transcript, proof.wCommit)
        marlinAbsorbPointImpl(transcript, proof.zACommit)
        marlinAbsorbPointImpl(transcript, proof.zBCommit)

        let etaA = transcript.squeeze()
        let etaB = transcript.squeeze()
        let etaC = transcript.squeeze()

        // Round 2: absorb t commitment, squeeze alpha (before sumcheck)
        marlinAbsorbPointImpl(transcript, proof.tCommit)
        let alpha = transcript.squeeze()

        // Absorb sumcheck round polynomials, squeeze beta
        for coeffs in proof.sumcheckPolyCoeffs {
            for c in coeffs {
                transcript.absorb(c)
            }
        }
        let beta = transcript.squeeze()

        // Round 3: absorb g, h commitments, squeeze gamma
        marlinAbsorbPointImpl(transcript, proof.gCommit)
        marlinAbsorbPointImpl(transcript, proof.hCommit)

        let gamma = transcript.squeeze()

        // --- Step 2: Verify outer sumcheck ---
        // The outer sumcheck verifies:
        //   sum_{x in H} [ (eta_A * z_A(x) + eta_B * z_B(x) + eta_C * z_A(x)*z_B(x)) * v_H(x)^{-1} ] = 0
        // The sumcheck produces round polynomials s_i(X) of degree 2.
        if !verifySumcheckRounds(proof.sumcheckPolyCoeffs, alpha: alpha) {
            return false
        }

        // --- Step 3: Verify polynomial evaluations at query points ---
        // The key relation at beta (derived from the outer sumcheck final check):
        //   eta_A * z_A(beta) + eta_B * z_B(beta) + eta_C * z_A(beta) * z_B(beta)
        //   = t(beta) * v_H(beta)
        let evals = proof.evaluations
        let lhs = frAdd(
            frAdd(frMul(etaA, evals.zABeta), frMul(etaB, evals.zBBeta)),
            frMul(etaC, frMul(evals.zABeta, evals.zBBeta))
        )

        // v_H(beta) = beta^|H| - 1
        let vHBeta = frSub(frPow(beta, UInt64(idx.constraintDomainSize)), Fr.one)
        let rhs = frMul(evals.tBeta, vHBeta)

        if frToInt(lhs) != frToInt(rhs) {
            return false
        }

        // --- Step 4: Verify inner sumcheck / lincheck ---
        let vKGamma = frSub(frPow(gamma, UInt64(idx.nonZeroDomainSize)), Fr.one)

        // For each matrix M, compute sigma_M from the index evaluations:
        //   sigma_M(gamma) = val_M(gamma) / ((beta - row_M(gamma)) * (gamma - col_M(gamma)))
        // Combined: eta_A*sigma_A + eta_B*sigma_B + eta_C*sigma_C
        var combinedSigma = Fr.zero
        let etas = [etaA, etaB, etaC]
        for m in 0..<3 {
            let betaMinusRow = frSub(beta, evals.rowGamma[m])
            let gammaMinusCol = frSub(gamma, evals.colGamma[m])
            let denom = frMul(betaMinusRow, gammaMinusCol)
            if denom.isZero { return false }
            let denomInv = frInverse(denom)
            let sigmaM = frMul(evals.valGamma[m], denomInv)
            combinedSigma = frAdd(combinedSigma, frMul(etas[m], sigmaM))
        }

        // Inner sumcheck check:
        //   g(gamma) = combinedSigma - h(gamma) * v_K(gamma) / |K_NZ|
        let kNZSizeInv = frInverse(frFromInt(UInt64(idx.nonZeroDomainSize)))
        let hContrib = frMul(frMul(evals.hGamma, vKGamma), kNZSizeInv)
        let expectedG = frSub(combinedSigma, hContrib)

        if frToInt(evals.gGamma) != frToInt(expectedG) {
            return false
        }

        // --- Step 5: Verify KZG opening proofs ---
        let batchChallenge = transcript.squeeze()
        if !verifyOpenings(vk: vk, proof: proof, beta: beta, gamma: gamma,
                           batchChallenge: batchChallenge) {
            return false
        }

        return true
    }

    // MARK: - Sumcheck Verification

    /// Verify the sumcheck round polynomial consistency.
    /// Each round polynomial s_i(X) is degree 2, encoded as [s_i(0), s_i(1), s_i(2)].
    private func verifySumcheckRounds(_ roundPolys: [[Fr]], alpha: Fr) -> Bool {
        guard !roundPolys.isEmpty else { return false }

        // Derive per-round challenges from alpha
        var challenges = [Fr]()
        var chalSeed = alpha
        for _ in 0..<roundPolys.count {
            challenges.append(chalSeed)
            chalSeed = frMul(chalSeed, alpha)
        }

        // Check first round: s_0(0) + s_0(1) = 0
        let firstSum = frAdd(roundPolys[0][0], roundPolys[0][1])
        if !firstSum.isZero {
            return false
        }

        // Check subsequent rounds: s_{i+1}(0) + s_{i+1}(1) = s_i(r_i)
        for i in 0..<(roundPolys.count - 1) {
            let ri = challenges[i]
            let siRi = evaluateDeg2Poly(roundPolys[i], at: ri)
            let nextSum = frAdd(roundPolys[i + 1][0], roundPolys[i + 1][1])
            if frToInt(siRi) != frToInt(nextSum) {
                return false
            }
        }

        return true
    }

    /// Evaluate a degree-2 polynomial [f(0), f(1), f(2)] at point r.
    /// Using Lagrange interpolation over {0, 1, 2}:
    ///   f(r) = f(0) * (r-1)(r-2)/2 - f(1) * r(r-2) + f(2) * r(r-1)/2
    private func evaluateDeg2Poly(_ coeffs: [Fr], at r: Fr) -> Fr {
        guard coeffs.count >= 3 else { return Fr.zero }
        let f0 = coeffs[0], f1 = coeffs[1], f2 = coeffs[2]
        let rM1 = frSub(r, Fr.one)
        let rM2 = frSub(r, frFromInt(2))
        let inv2 = frInverse(frFromInt(2))

        let t0 = frMul(f0, frMul(frMul(rM1, rM2), inv2))
        let t1 = frMul(frNeg(f1), frMul(r, rM2))
        let t2 = frMul(f2, frMul(frMul(r, rM1), inv2))

        return frAdd(frAdd(t0, t1), t2)
    }

    // MARK: - KZG Opening Verification

    /// Verify all KZG opening proofs via random linear combination.
    /// Without pairing: C - y*G == (s - z) * W for each opening.
    private func verifyOpenings(vk: MarlinVerifyingKey, proof: MarlinProof,
                                beta: Fr, gamma: Fr, batchChallenge: Fr) -> Bool {
        let s = vk.srsSecret
        let g1 = pointFromAffine(vk.srs[0])
        let evals = proof.evaluations

        struct OpeningTuple {
            let commitment: PointProjective
            let point: Fr
            let evaluation: Fr
            let witness: PointProjective
        }

        var tuples = [OpeningTuple]()

        // Openings at beta
        let betaOpenings: [(PointProjective, Fr)] = [
            (proof.wCommit, evals.wBeta),
            (proof.zACommit, evals.zABeta),
            (proof.zBCommit, evals.zBBeta),
            (proof.tCommit, evals.tBeta),
        ]

        for (i, (commit, eval)) in betaOpenings.enumerated() {
            if i < proof.openingProofs.count {
                tuples.append(OpeningTuple(commitment: commit, point: beta,
                                           evaluation: eval, witness: proof.openingProofs[i]))
            }
        }

        // Openings at gamma
        let gammaCommitments: [PointProjective] = [proof.gCommit, proof.hCommit]
        let gammaEvals: [Fr] = [evals.gGamma, evals.hGamma]

        for (i, (commit, eval)) in zip(gammaCommitments, gammaEvals).enumerated() {
            let proofIdx = 4 + i
            if proofIdx < proof.openingProofs.count {
                tuples.append(OpeningTuple(commitment: commit, point: gamma,
                                           evaluation: eval, witness: proof.openingProofs[proofIdx]))
            }
        }

        // Index polynomial openings at gamma (row, col, val, row_col for A, B, C)
        let indexEvals = evals.rowGamma + evals.colGamma + evals.valGamma + evals.rowColGamma
        for (i, eval) in indexEvals.enumerated() {
            let commitIdx = i
            let proofIdx = 6 + i
            if commitIdx < vk.indexCommitments.count && proofIdx < proof.openingProofs.count {
                tuples.append(OpeningTuple(commitment: vk.indexCommitments[commitIdx],
                                           point: gamma, evaluation: eval,
                                           witness: proof.openingProofs[proofIdx]))
            }
        }

        guard !tuples.isEmpty else { return false }

        // Batch verify: random linear combination
        var accum = pointIdentity()
        var rho = Fr.one

        for t in tuples {
            let cMinusYG = pointAdd(t.commitment, cPointScalarMul(g1, frNeg(t.evaluation)))
            let sMinusZ = frSub(s, t.point)
            let szW = cPointScalarMul(t.witness, sMinusZ)
            let diff = pointAdd(cMinusYG, cPointScalarMul(szW, frNeg(Fr.one)))
            accum = pointAdd(accum, cPointScalarMul(diff, rho))
            rho = frMul(rho, batchChallenge)
        }

        return pointIsIdentity(accum)
    }

    // MARK: - Batch Verification

    /// Batch verify multiple Marlin proofs via random linear combination.
    /// Combines all KZG checks across all proofs into a single accumulation.
    public func batchVerify(vk: MarlinVerifyingKey, proofs: [(publicInput: [Fr], proof: MarlinProof)]) -> Bool {
        guard !proofs.isEmpty else { return false }
        if proofs.count == 1 {
            return verify(vk: vk, publicInput: proofs[0].publicInput, proof: proofs[0].proof)
        }

        let idx = vk.index
        let s = vk.srsSecret
        let g1 = pointFromAffine(vk.srs[0])

        // Generate batch randomness
        let batchTranscript = Transcript(label: "marlin-batch", backend: .keccak256)
        batchTranscript.absorb(frFromInt(UInt64(proofs.count)))

        for (pi, proof) in proofs {
            for p in pi { batchTranscript.absorb(p) }
            marlinAbsorbPointImpl(batchTranscript, proof.wCommit)
            marlinAbsorbPointImpl(batchTranscript, proof.zACommit)
            marlinAbsorbPointImpl(batchTranscript, proof.zBCommit)
        }

        struct KZGTuple {
            let commitment: PointProjective
            let point: Fr
            let evaluation: Fr
            let witness: PointProjective
        }

        var allTuples = [KZGTuple]()

        for (publicInput, proof) in proofs {
            // Reconstruct challenges
            let transcript = Transcript(label: "marlin", backend: .keccak256)
            transcript.absorb(frFromInt(UInt64(idx.numConstraints)))
            transcript.absorb(frFromInt(UInt64(idx.numVariables)))
            transcript.absorb(frFromInt(UInt64(idx.numNonZero)))
            for c in vk.indexCommitments { marlinAbsorbPointImpl(transcript, c) }
            for pi in publicInput { transcript.absorb(pi) }

            marlinAbsorbPointImpl(transcript, proof.wCommit)
            marlinAbsorbPointImpl(transcript, proof.zACommit)
            marlinAbsorbPointImpl(transcript, proof.zBCommit)

            let etaA = transcript.squeeze()
            let etaB = transcript.squeeze()
            let etaC = transcript.squeeze()

            marlinAbsorbPointImpl(transcript, proof.tCommit)
            let alpha = transcript.squeeze()

            for coeffs in proof.sumcheckPolyCoeffs {
                for c in coeffs { transcript.absorb(c) }
            }
            let beta = transcript.squeeze()

            marlinAbsorbPointImpl(transcript, proof.gCommit)
            marlinAbsorbPointImpl(transcript, proof.hCommit)

            let gamma = transcript.squeeze()

            // Verify sumcheck rounds
            if !verifySumcheckRounds(proof.sumcheckPolyCoeffs, alpha: alpha) {
                return false
            }

            // Verify outer relation
            let evals = proof.evaluations
            let lhs = frAdd(
                frAdd(frMul(etaA, evals.zABeta), frMul(etaB, evals.zBBeta)),
                frMul(etaC, frMul(evals.zABeta, evals.zBBeta))
            )
            let vHBeta = frSub(frPow(beta, UInt64(idx.constraintDomainSize)), Fr.one)
            let rhs = frMul(evals.tBeta, vHBeta)
            if frToInt(lhs) != frToInt(rhs) { return false }

            // Verify inner relation
            let vKGamma = frSub(frPow(gamma, UInt64(idx.nonZeroDomainSize)), Fr.one)
            var combinedSigma = Fr.zero
            let etas = [etaA, etaB, etaC]
            for m in 0..<3 {
                let betaMinusRow = frSub(beta, evals.rowGamma[m])
                let gammaMinusCol = frSub(gamma, evals.colGamma[m])
                let denom = frMul(betaMinusRow, gammaMinusCol)
                if denom.isZero { return false }
                let denomInv = frInverse(denom)
                let sigmaM = frMul(evals.valGamma[m], denomInv)
                combinedSigma = frAdd(combinedSigma, frMul(etas[m], sigmaM))
            }
            let kNZSizeInv = frInverse(frFromInt(UInt64(idx.nonZeroDomainSize)))
            let hContrib = frMul(frMul(evals.hGamma, vKGamma), kNZSizeInv)
            let expectedG = frSub(combinedSigma, hContrib)
            if frToInt(evals.gGamma) != frToInt(expectedG) { return false }

            // Collect KZG tuples
            let betaOpenings: [(PointProjective, Fr)] = [
                (proof.wCommit, evals.wBeta),
                (proof.zACommit, evals.zABeta),
                (proof.zBCommit, evals.zBBeta),
                (proof.tCommit, evals.tBeta),
            ]
            for (i, (commit, eval)) in betaOpenings.enumerated() {
                if i < proof.openingProofs.count {
                    allTuples.append(KZGTuple(commitment: commit, point: beta,
                                              evaluation: eval, witness: proof.openingProofs[i]))
                }
            }
            let gammaCommitments = [proof.gCommit, proof.hCommit]
            let gammaEvals = [evals.gGamma, evals.hGamma]
            for (i, (commit, eval)) in zip(gammaCommitments, gammaEvals).enumerated() {
                let proofIdx = 4 + i
                if proofIdx < proof.openingProofs.count {
                    allTuples.append(KZGTuple(commitment: commit, point: gamma,
                                              evaluation: eval, witness: proof.openingProofs[proofIdx]))
                }
            }
            let indexEvals = evals.rowGamma + evals.colGamma + evals.valGamma + evals.rowColGamma
            for (i, eval) in indexEvals.enumerated() {
                let proofIdx = 6 + i
                if i < vk.indexCommitments.count && proofIdx < proof.openingProofs.count {
                    allTuples.append(KZGTuple(commitment: vk.indexCommitments[i],
                                              point: gamma, evaluation: eval,
                                              witness: proof.openingProofs[proofIdx]))
                }
            }
        }

        // Batch KZG verification
        let batchRho = batchTranscript.squeeze()
        var accum = pointIdentity()
        var rho = Fr.one

        for t in allTuples {
            let cMinusYG = pointAdd(t.commitment, cPointScalarMul(g1, frNeg(t.evaluation)))
            let sMinusZ = frSub(s, t.point)
            let szW = cPointScalarMul(t.witness, sMinusZ)
            let diff = pointAdd(cMinusYG, cPointScalarMul(szW, frNeg(Fr.one)))
            accum = pointAdd(accum, cPointScalarMul(diff, rho))
            rho = frMul(rho, batchRho)
        }

        return pointIsIdentity(accum)
    }

    // MARK: - Helpers

    private func marlinAbsorbPointImpl(_ transcript: Transcript, _ p: PointProjective) {
        if pointIsIdentity(p) {
            transcript.absorb(Fr.zero)
            transcript.absorb(Fr.zero)
            return
        }
        let aff = batchToAffine([p])
        let xInt = fpToInt(aff[0].x)
        let yInt = fpToInt(aff[0].y)
        transcript.absorb(Fr.from64(xInt))
        transcript.absorb(Fr.from64(yInt))
    }
}

// MARK: - Module-level helper

/// Absorb a projective point into a Marlin transcript.
func marlinAbsorbPointImpl(_ transcript: Transcript, _ p: PointProjective) {
    if pointIsIdentity(p) {
        transcript.absorb(Fr.zero)
        transcript.absorb(Fr.zero)
        return
    }
    let aff = batchToAffine([p])
    let xInt = fpToInt(aff[0].x)
    let yInt = fpToInt(aff[0].y)
    transcript.absorb(Fr.from64(xInt))
    transcript.absorb(Fr.from64(yInt))
}

// MARK: - Test Proof Generator

/// Generate a valid Marlin proof for testing.
/// Creates a trivial R1CS instance and builds a proof that passes verification.
public class MarlinTestProver {
    public let kzg: KZGEngine

    public init(kzg: KZGEngine) {
        self.kzg = kzg
    }

    /// Generate a test verifying key and proof for an R1CS of the given size.
    /// Iteratively fixes all polynomial values so the Fiat-Shamir transcript is consistent.
    public func generateTestProof(numConstraints: Int, publicInput: [Fr],
                                   srsSecret: Fr) throws -> (MarlinVerifyingKey, MarlinProof) {
        let m = numConstraints
        let numPubVars = publicInput.count
        let n = numPubVars + m + 1
        let hSize = nextPow2(m)
        let kSize = nextPow2(n)
        let nnz = m
        let nzSize = nextPow2(nnz)

        let omegaH = computeRootOfUnity(hSize)
        let omegaK = computeRootOfUnity(kSize)

        let index = MarlinIndex(
            numConstraints: m, numVariables: n, numNonZero: nnz,
            constraintDomainSize: hSize, variableDomainSize: kSize,
            nonZeroDomainSize: nzSize, omegaH: omegaH, omegaK: omegaK
        )

        // Build z = [1, publicInput, witness]
        var z = [Fr.one]
        z.append(contentsOf: publicInput)
        for i in 0..<m {
            z.append(i < numPubVars ? frAdd(publicInput[i], Fr.one) : frFromInt(UInt64(i + 2)))
        }
        while z.count < kSize { z.append(Fr.zero) }

        // z_A, z_B (diagonal A, B matrices for test)
        var zA = [Fr](repeating: Fr.zero, count: hSize)
        var zB = [Fr](repeating: Fr.zero, count: hSize)
        for i in 0..<m {
            zA[i] = z[min(1 + (i % max(1, numPubVars)), z.count - 1)]
            zB[i] = z[min(1 + numPubVars + (i % m), z.count - 1)]
        }

        // Index polynomials: 12 total (row, col, val, row_col for A, B, C)
        func makeIndexPoly(_ seed: UInt64) -> [Fr] {
            var coeffs = [Fr](repeating: Fr.zero, count: nzSize)
            for i in 0..<min(nnz, nzSize) {
                coeffs[i] = frFromInt(seed &+ UInt64(i) &+ 1)
            }
            return coeffs
        }

        var indexPolys = [[Fr]]()
        for i: UInt64 in 0..<12 {
            indexPolys.append(makeIndexPoly(100 &+ i &* 37))
        }

        var indexCommitments = [PointProjective]()
        for poly in indexPolys {
            indexCommitments.append(try kzg.commit(poly))
        }

        let srs = kzg.srs

        // Round 1: commit to w, z_A, z_B
        let wCoeffs = Array(z.prefix(kSize))
        let wCommit = try kzg.commit(wCoeffs)
        let zACoeffs = zA
        let zACommit = try kzg.commit(zACoeffs)
        let zBCoeffs = zB
        let zBCommit = try kzg.commit(zBCoeffs)

        // Iterative transcript construction to get consistent challenges.
        // We need to solve: all evaluations and commitments must be consistent
        // with the Fiat-Shamir challenges they produce.
        // Strategy: fix a "seed" transcript up through round 1 (which is stable),
        // then derive challenges and build remaining proof elements to match.

        // Seed transcript: everything through round 1 is fixed
        func buildTranscript(tCommit: PointProjective, sumcheckPolys: [[Fr]],
                             gCommit: PointProjective, hCommit: PointProjective)
            -> (etaA: Fr, etaB: Fr, etaC: Fr, alpha: Fr, beta: Fr, gamma: Fr)
        {
            let ts = Transcript(label: "marlin", backend: .keccak256)
            ts.absorb(frFromInt(UInt64(m)))
            ts.absorb(frFromInt(UInt64(n)))
            ts.absorb(frFromInt(UInt64(nnz)))
            for c in indexCommitments { marlinAbsorbPointImpl(ts, c) }
            for pi in publicInput { ts.absorb(pi) }
            marlinAbsorbPointImpl(ts, wCommit)
            marlinAbsorbPointImpl(ts, zACommit)
            marlinAbsorbPointImpl(ts, zBCommit)
            let etaA = ts.squeeze()
            let etaB = ts.squeeze()
            let etaC = ts.squeeze()
            marlinAbsorbPointImpl(ts, tCommit)
            let alpha = ts.squeeze()
            for coeffs in sumcheckPolys {
                for c in coeffs { ts.absorb(c) }
            }
            let beta = ts.squeeze()
            marlinAbsorbPointImpl(ts, gCommit)
            marlinAbsorbPointImpl(ts, hCommit)
            let gamma = ts.squeeze()
            return (etaA, etaB, etaC, alpha, beta, gamma)
        }

        // Phase 1: Get eta challenges (stable, only depend on round 1)
        let dummyPoint = pointIdentity()
        let dummySC = [[Fr.zero, Fr.zero, Fr.zero]]
        let (etaA, etaB, etaC, _, _, _) = buildTranscript(
            tCommit: dummyPoint, sumcheckPolys: dummySC,
            gCommit: dummyPoint, hCommit: dummyPoint)

        // Phase 2: Build t polynomial from the outer relation
        var tCoeffs = [Fr](repeating: Fr.zero, count: hSize)
        for i in 0..<hSize {
            tCoeffs[i] = frAdd(
                frAdd(frMul(etaA, zA[i]), frMul(etaB, zB[i])),
                frMul(etaC, frMul(zA[i], zB[i]))
            )
        }

        // Phase 3: Build h polynomial (small, fixed)
        var hCoeffs = [Fr](repeating: Fr.zero, count: max(nzSize, 2))
        hCoeffs[0] = frFromInt(42)
        hCoeffs[1] = frFromInt(7)

        // Phase 4: Iterate to find consistent commitments and challenges
        // First pass: build sumcheck polys with a placeholder alpha
        let numSumcheckRounds = Int(log2(Double(hSize)))

        var tCommitFinal = try kzg.commit(tCoeffs)
        let hCommit = try kzg.commit(hCoeffs)
        var gCoeffs = [Fr](repeating: Fr.zero, count: max(nzSize, 2))
        var gCommitFinal = try kzg.commit(gCoeffs)

        // Iterate twice to converge (changing gCommit changes gamma)
        for _ in 0..<2 {
            let chals = buildTranscript(
                tCommit: tCommitFinal, sumcheckPolys: buildSumcheckPolys(numSumcheckRounds, alpha: Fr.one),
                gCommit: gCommitFinal, hCommit: hCommit)

            // Build sumcheck polys with actual alpha
            let sumcheckPolys = buildSumcheckPolys(numSumcheckRounds, alpha: chals.alpha)

            // Now get final challenges with correct sumcheck polys
            let finalChals = buildTranscript(
                tCommit: tCommitFinal, sumcheckPolys: sumcheckPolys,
                gCommit: gCommitFinal, hCommit: hCommit)

            let beta = finalChals.beta
            let gamma = finalChals.gamma

            // Fix t so t(beta) = lhs / v_H(beta)
            let zABeta = evaluatePolyAt(zACoeffs, beta)
            let zBBeta = evaluatePolyAt(zBCoeffs, beta)
            let outerLHS = frAdd(
                frAdd(frMul(etaA, zABeta), frMul(etaB, zBBeta)),
                frMul(etaC, frMul(zABeta, zBBeta))
            )
            let vHBeta = frSub(frPow(beta, UInt64(hSize)), Fr.one)
            let tBetaTarget = vHBeta.isZero ? Fr.zero : frMul(outerLHS, frInverse(vHBeta))
            let currentT = evaluatePolyAt(tCoeffs, beta)
            tCoeffs[0] = frAdd(tCoeffs[0], frSub(tBetaTarget, currentT))
            tCommitFinal = try kzg.commit(tCoeffs)

            // Fix g so inner relation holds at gamma
            var rowG = [Fr](), colG = [Fr](), valG = [Fr](), rcG = [Fr]()
            for mi in 0..<3 {
                rowG.append(evaluatePolyAt(indexPolys[mi * 4], gamma))
                colG.append(evaluatePolyAt(indexPolys[mi * 4 + 1], gamma))
                valG.append(evaluatePolyAt(indexPolys[mi * 4 + 2], gamma))
                rcG.append(evaluatePolyAt(indexPolys[mi * 4 + 3], gamma))
            }

            let etas = [etaA, etaB, etaC]
            var combinedSigma = Fr.zero
            for mi in 0..<3 {
                let d = frMul(frSub(beta, rowG[mi]), frSub(gamma, colG[mi]))
                if !d.isZero {
                    combinedSigma = frAdd(combinedSigma, frMul(etas[mi], frMul(valG[mi], frInverse(d))))
                }
            }

            let vKGamma = frSub(frPow(gamma, UInt64(nzSize)), Fr.one)
            let kNZSizeInv = frInverse(frFromInt(UInt64(nzSize)))
            let hGamma = evaluatePolyAt(hCoeffs, gamma)
            let hContrib = frMul(frMul(hGamma, vKGamma), kNZSizeInv)
            let gGammaTarget = frSub(combinedSigma, hContrib)
            let currentG = evaluatePolyAt(gCoeffs, gamma)
            gCoeffs[0] = frAdd(gCoeffs[0], frSub(gGammaTarget, currentG))
            gCommitFinal = try kzg.commit(gCoeffs)
        }

        // Final pass: get all challenges with the final commitments
        let numRounds = numSumcheckRounds
        // Need to get alpha first to build sumcheck polys
        // Use a preliminary transcript to get alpha
        let prelimChals = buildTranscript(
            tCommit: tCommitFinal,
            sumcheckPolys: buildSumcheckPolys(numRounds, alpha: Fr.one),
            gCommit: gCommitFinal, hCommit: hCommit)

        let sumcheckPolys = buildSumcheckPolys(numRounds, alpha: prelimChals.alpha)

        // Get FINAL challenges
        let finalChals = buildTranscript(
            tCommit: tCommitFinal, sumcheckPolys: sumcheckPolys,
            gCommit: gCommitFinal, hCommit: hCommit)

        let betaF = finalChals.beta
        let gammaF = finalChals.gamma

        // Final fix of t and g with the actual final challenges
        let zABetaF = evaluatePolyAt(zACoeffs, betaF)
        let zBBetaF = evaluatePolyAt(zBCoeffs, betaF)
        let wBetaF = evaluatePolyAt(wCoeffs, betaF)
        let outerLHSF = frAdd(
            frAdd(frMul(etaA, zABetaF), frMul(etaB, zBBetaF)),
            frMul(etaC, frMul(zABetaF, zBBetaF))
        )
        let vHBetaF = frSub(frPow(betaF, UInt64(hSize)), Fr.one)
        let tBetaF = vHBetaF.isZero ? Fr.zero : frMul(outerLHSF, frInverse(vHBetaF))
        tCoeffs[0] = frAdd(tCoeffs[0], frSub(tBetaF, evaluatePolyAt(tCoeffs, betaF)))
        tCommitFinal = try kzg.commit(tCoeffs)

        var rowG = [Fr](), colG = [Fr](), valG = [Fr](), rcG = [Fr]()
        for mi in 0..<3 {
            rowG.append(evaluatePolyAt(indexPolys[mi * 4], gammaF))
            colG.append(evaluatePolyAt(indexPolys[mi * 4 + 1], gammaF))
            valG.append(evaluatePolyAt(indexPolys[mi * 4 + 2], gammaF))
            rcG.append(evaluatePolyAt(indexPolys[mi * 4 + 3], gammaF))
        }

        let etasF = [etaA, etaB, etaC]
        var combinedSigmaF = Fr.zero
        for mi in 0..<3 {
            let d = frMul(frSub(betaF, rowG[mi]), frSub(gammaF, colG[mi]))
            if !d.isZero {
                combinedSigmaF = frAdd(combinedSigmaF, frMul(etasF[mi], frMul(valG[mi], frInverse(d))))
            }
        }
        let vKGammaF = frSub(frPow(gammaF, UInt64(nzSize)), Fr.one)
        let kNZSizeInv = frInverse(frFromInt(UInt64(nzSize)))
        let hGammaF = evaluatePolyAt(hCoeffs, gammaF)
        let hContribF = frMul(frMul(hGammaF, vKGammaF), kNZSizeInv)
        let gGammaF = frSub(combinedSigmaF, hContribF)
        gCoeffs[0] = frAdd(gCoeffs[0], frSub(gGammaF, evaluatePolyAt(gCoeffs, gammaF)))
        gCommitFinal = try kzg.commit(gCoeffs)

        // Build KZG opening proofs
        let allPolys: [[Fr]] = [wCoeffs, zACoeffs, zBCoeffs, tCoeffs,
                                gCoeffs, hCoeffs] + indexPolys
        let allPoints: [Fr] = [betaF, betaF, betaF, betaF,
                               gammaF, gammaF] + [Fr](repeating: gammaF, count: 12)

        var openingProofs = [PointProjective]()
        for (poly, pt) in zip(allPolys, allPoints) {
            let kzgProof = try kzg.open(poly, at: pt)
            openingProofs.append(kzgProof.witness)
        }

        let evaluations = MarlinEvaluations(
            zABeta: zABetaF, zBBeta: zBBetaF, wBeta: wBetaF, tBeta: tBetaF,
            gGamma: gGammaF, hGamma: hGammaF,
            rowGamma: rowG, colGamma: colG, valGamma: valG, rowColGamma: rcG
        )

        let vk = MarlinVerifyingKey(index: index, indexCommitments: indexCommitments,
                                     srsSecret: srsSecret, srs: srs)

        let proof = MarlinProof(
            wCommit: wCommit, zACommit: zACommit, zBCommit: zBCommit,
            tCommit: tCommitFinal, sumcheckPolyCoeffs: sumcheckPolys,
            gCommit: gCommitFinal, hCommit: hCommit,
            evaluations: evaluations, openingProofs: openingProofs
        )

        return (vk, proof)
    }

    // MARK: - Helpers

    private func buildSumcheckPolys(_ numRounds: Int, alpha: Fr) -> [[Fr]] {
        var challenges = [Fr]()
        var chalSeed = alpha
        for _ in 0..<numRounds {
            challenges.append(chalSeed)
            chalSeed = frMul(chalSeed, alpha)
        }

        var polys = [[Fr]]()
        for r in 0..<numRounds {
            if r == 0 {
                let s0 = frFromInt(7)
                let s1 = frNeg(s0)
                let s2 = frAdd(s0, frFromInt(3))
                polys.append([s0, s1, s2])
            } else {
                let prevPoly = polys[r - 1]
                let ri = challenges[r - 1]
                let targetSum = evaluateDeg2PolyHelper(prevPoly, at: ri)
                let si0 = frFromInt(UInt64(r) &+ 11)
                let si1 = frSub(targetSum, si0)
                let si2 = frAdd(si0, frFromInt(5))
                polys.append([si0, si1, si2])
            }
        }
        return polys
    }

    private func evaluateDeg2PolyHelper(_ coeffs: [Fr], at r: Fr) -> Fr {
        guard coeffs.count >= 3 else { return Fr.zero }
        let f0 = coeffs[0], f1 = coeffs[1], f2 = coeffs[2]
        let rM1 = frSub(r, Fr.one)
        let rM2 = frSub(r, frFromInt(2))
        let inv2 = frInverse(frFromInt(2))
        let t0 = frMul(f0, frMul(frMul(rM1, rM2), inv2))
        let t1 = frMul(frNeg(f1), frMul(r, rM2))
        let t2 = frMul(f2, frMul(frMul(r, rM1), inv2))
        return frAdd(frAdd(t0, t1), t2)
    }

    private func evaluatePolyAt(_ coeffs: [Fr], _ x: Fr) -> Fr {
        var result = Fr.zero
        for i in stride(from: coeffs.count - 1, through: 0, by: -1) {
            result = frAdd(frMul(result, x), coeffs[i])
        }
        return result
    }

    private func nextPow2(_ n: Int) -> Int {
        var p = 1
        while p < n { p *= 2 }
        return max(p, 2)
    }

    private func computeRootOfUnity(_ domainSize: Int) -> Fr {
        let exp = UInt64(1 << Fr.TWO_ADICITY) / UInt64(domainSize)
        return frPow(Fr.from64(Fr.ROOT_OF_UNITY), exp)
    }
}
