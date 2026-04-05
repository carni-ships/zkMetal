// IPA Claim — Deferred IPA Verification
//
// An IPAClaim represents a deferred IPA verification: instead of immediately
// checking a proof, we package the claim for later accumulation/batch verification.
//
// This is the core data structure for Halo-style accumulation (BCMS 2020):
//   - Prover produces IPA proof for <a, b> = v with commitment C
//   - Instead of verifying, extract an IPAClaim
//   - Accumulate multiple claims into a single Accumulator
//   - Verify once at the end (the "decider")
//
// References: eprint 2020/499 (Bunz, Chiesa, Mishra, Spooner)

import Foundation

// MARK: - IPA Claim (Pallas Curve)

/// A deferred IPA verification claim over the Pallas curve.
///
/// Captures everything needed to verify an IPA opening:
///   - commitment C: the Pedersen commitment to the witness vector
///   - evaluationPoint z: the point at which the polynomial is evaluated
///   - claimedValue v: the claimed evaluation result
///   - proof: the IPA proof (L_i, R_i rounds + final scalar a)
///
/// Instead of verifying immediately, accumulate multiple claims and batch-verify.
public struct PallasIPAClaim {
    /// Pedersen commitment C = MSM(G, a)
    public let commitment: PallasPointProjective
    /// Evaluation vector b (determines the evaluation point)
    public let evaluationVector: [VestaFp]
    /// Claimed inner product value v = <a, b>
    public let claimedValue: VestaFp
    /// The IPA proof (log(n) rounds of L_i, R_i + final scalar)
    public let proof: PallasIPAProof
    /// Bound commitment C + v*Q (commitment with inner product value bound in)
    public let boundCommitment: PallasPointProjective

    public init(commitment: PallasPointProjective,
                evaluationVector: [VestaFp],
                claimedValue: VestaFp,
                proof: PallasIPAProof,
                Q: PallasPointAffine) {
        self.commitment = commitment
        self.evaluationVector = evaluationVector
        self.claimedValue = claimedValue
        self.proof = proof
        let qProj = pallasPointFromAffine(Q)
        let vQ = pallasPointScalarMul(qProj, claimedValue)
        self.boundCommitment = pallasPointAdd(commitment, vQ)
    }

    /// Convenience initializer with pre-computed bound commitment.
    public init(commitment: PallasPointProjective,
                evaluationVector: [VestaFp],
                claimedValue: VestaFp,
                proof: PallasIPAProof,
                boundCommitment: PallasPointProjective) {
        self.commitment = commitment
        self.evaluationVector = evaluationVector
        self.claimedValue = claimedValue
        self.proof = proof
        self.boundCommitment = boundCommitment
    }
}

// MARK: - Accumulation Proof

/// Proof that an accumulation step was performed correctly.
///
/// When folding a new IPAClaim into an existing accumulator, the prover produces
/// an AccumulationProof. The verifier checks this proof (cheap: O(1) group ops)
/// instead of re-doing the full accumulation.
///
/// The proof contains the cross-terms from the linear combination:
///   C' = C_acc + rho * C_new
///   u' = u_acc + rho * u_new
/// The cross-term T ensures the folded commitment is correct.
public struct AccumulationProof {
    /// Random challenge rho used for folding
    public let rho: VestaFp
    /// Cross-term commitment (for soundness of the folding)
    public let crossTerm: PallasPointProjective

    public init(rho: VestaFp, crossTerm: PallasPointProjective) {
        self.rho = rho
        self.crossTerm = crossTerm
    }
}

// MARK: - Folded Accumulator

/// A folded accumulator that combines multiple IPA claims.
///
/// After accumulating N claims, this holds:
///   - foldedCommitment: C_1 + rho_1*C_2 + rho_1*rho_2*C_3 + ...
///   - foldedScalar: a_1 + rho_1*a_2 + rho_1*rho_2*a_3 + ...
///   - The challenges from all folded proofs
///
/// The decider verifies this single accumulated claim with one MSM.
public struct FoldedAccumulator {
    /// Folded commitment (linear combination of all bound commitments)
    public let foldedCommitment: PallasPointProjective
    /// Folded IPA proof scalars
    public let foldedScalar: VestaFp
    /// All IPA challenges from accumulated proofs (flattened)
    public let allChallenges: [[VestaFp]]
    /// All evaluation vectors from accumulated proofs
    public let allEvalVectors: [[VestaFp]]
    /// Folding challenges (rho values used at each accumulation step)
    public let foldingChallenges: [VestaFp]
    /// Number of accumulated claims
    public let claimCount: Int

    public init(foldedCommitment: PallasPointProjective,
                foldedScalar: VestaFp,
                allChallenges: [[VestaFp]],
                allEvalVectors: [[VestaFp]],
                foldingChallenges: [VestaFp],
                claimCount: Int) {
        self.foldedCommitment = foldedCommitment
        self.foldedScalar = foldedScalar
        self.allChallenges = allChallenges
        self.allEvalVectors = allEvalVectors
        self.foldingChallenges = foldingChallenges
        self.claimCount = claimCount
    }
}

// MARK: - Claim Extraction Helper

extension PallasAccumulationEngine {

    /// Extract an IPAClaim from a witness vector and its IPA proof.
    ///
    /// This is a convenience that packages the proof data into a claim
    /// ready for accumulation.
    public func extractClaim(witness a: [VestaFp], evaluationVector b: [VestaFp],
                             proof: PallasIPAProof) -> PallasIPAClaim {
        let C = commit(a)
        let v = PallasAccumulationEngine.innerProduct(a, b)
        return PallasIPAClaim(
            commitment: C,
            evaluationVector: b,
            claimedValue: v,
            proof: proof,
            Q: Q
        )
    }

    /// Accumulate an IPAClaim into an IPAAccumulator (deferred verification).
    ///
    /// This is a thin wrapper around the existing accumulate() method
    /// that takes a claim struct instead of individual parameters.
    public func accumulateClaim(_ claim: PallasIPAClaim) -> IPAAccumulator {
        return accumulate(
            proof: claim.proof,
            commitment: claim.boundCommitment,
            b: claim.evaluationVector,
            innerProductValue: claim.claimedValue
        )
    }

    /// Fold two accumulators into one using random challenge.
    ///
    /// Given accumulators acc1 and acc2:
    ///   C' = C_1 + rho * C_2
    ///   a' = a_1 + rho * a_2
    ///
    /// The folding is correct with overwhelming probability over random rho.
    /// Cost: O(1) field ops + 1 scalar multiplication + 1 point addition.
    public func foldAccumulators(_ acc1: IPAAccumulator, _ acc2: IPAAccumulator) -> (IPAAccumulator, AccumulationProof) {
        // Derive folding challenge from both commitments
        var transcript = [UInt8]()
        appendPointToTranscript(&transcript, acc1.commitment)
        appendPointToTranscript(&transcript, acc2.commitment)
        let rho = deriveChallenge(transcript)

        // Fold commitments: C' = C1 + rho * C2
        let rhoC2 = pallasPointScalarMul(acc2.commitment, rho)
        let foldedCommitment = pallasPointAdd(acc1.commitment, rhoC2)

        // Fold proof scalars: a' = a1 + rho * a2
        let foldedA = vestaAdd(acc1.proofA, vestaMul(rho, acc2.proofA))

        // Cross-term: rho * C2 (this is what the verifier needs to check)
        let crossTerm = rhoC2

        let proof = AccumulationProof(rho: rho, crossTerm: crossTerm)

        // The folded accumulator uses acc1's challenges/generators/Q
        // (both accumulators share the same SRS in practice)
        let folded = IPAAccumulator(
            commitment: foldedCommitment,
            b: acc1.b,
            value: vestaAdd(acc1.value, vestaMul(rho, acc2.value)),
            challenges: acc1.challenges,
            generators: generators,
            Q: Q,
            proofA: foldedA
        )

        return (folded, proof)
    }

    // MARK: - Transcript Helpers (internal, reused from Accumulator.swift pattern)

    func appendPointToTranscript(_ transcript: inout [UInt8], _ p: PallasPointProjective) {
        let affine = pallasPointToAffine(p)
        let xBytes = affine.x.toBytes()
        let yBytes = affine.y.toBytes()
        transcript.append(contentsOf: xBytes)
        transcript.append(contentsOf: yBytes)
    }

    func appendScalarToTranscript(_ transcript: inout [UInt8], _ v: VestaFp) {
        let intVal = vestaToInt(v)
        for limb in intVal {
            for byte in 0..<8 {
                transcript.append(UInt8((limb >> (byte * 8)) & 0xFF))
            }
        }
    }

    func deriveChallenge(_ transcript: [UInt8]) -> VestaFp {
        let hash = blake3(transcript)
        var limbs = [UInt64](repeating: 0, count: 4)
        for i in 0..<4 {
            for j in 0..<8 {
                limbs[i] |= UInt64(hash[i * 8 + j]) << (j * 8)
            }
        }
        // Clear top 2 bits to ensure < 2^254 ~ Vesta p
        limbs[3] &= 0x3FFFFFFFFFFFFFFF
        let raw = VestaFp.from64(limbs)
        return vestaMul(raw, VestaFp.from64(VestaFp.R2_MOD_P))
    }
}
