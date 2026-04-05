// Groth16 Proof Aggregation Engine — recursive composition support
// Provides the aggregation primitives needed for recursive Groth16 verification:
//   1. Random-linear-combination batch verification (multi-pairing)
//   2. Proof bundling for recursive verifier circuits
//
// Complements existing SnarkPack aggregator (Aggregation/Groth16ProofAggregator.swift)
// and batch verifier (Aggregation/Groth16BatchVerifier.swift) with a simpler
// aggregation API designed for use with the Groth16VerifierCircuit.
//
// For BN254 curve.

import Foundation
import NeonFieldOps

// MARK: - Recursive Aggregation Bundle

/// A bundle of Groth16 proofs prepared for recursive verification.
/// Contains pre-computed aggregated values needed by the verifier circuit.
public struct RecursiveAggregationBundle {
    /// Random challenges used for aggregation.
    public let challenges: [Fr]
    /// Individual proof components.
    public let proofs: [Groth16Proof]
    /// Public inputs for each individual proof.
    public let allPublicInputs: [[Fr]]
    /// Pre-computed aggregated vk_accum: sum(r_i * vk_accum_i) in G1.
    public let aggregatedVKAccum: PointProjective
    /// Pre-computed aggregated C: sum(r_i * C_i) in G1.
    public let aggregatedC: PointProjective
    /// Sum of challenges: sum(r_i).
    public let challengeSum: Fr

    public init(challenges: [Fr], proofs: [Groth16Proof], allPublicInputs: [[Fr]],
                aggregatedVKAccum: PointProjective, aggregatedC: PointProjective,
                challengeSum: Fr) {
        self.challenges = challenges
        self.proofs = proofs
        self.allPublicInputs = allPublicInputs
        self.aggregatedVKAccum = aggregatedVKAccum
        self.aggregatedC = aggregatedC
        self.challengeSum = challengeSum
    }
}

// MARK: - Recursive Aggregator

/// Aggregator for recursive Groth16 proof composition.
///
/// Given N proofs, creates a bundle with pre-computed aggregated values.
/// The bundle can then be fed into a recursive verifier circuit, or verified
/// directly via a multi-pairing check.
///
/// Aggregation equation:
///   prod_i e(r_i*A_i, B_i) = e(alpha, beta)^(sum r_i) * e(aggVKAccum, gamma) * e(aggC, delta)
///
/// Rearranged as product=1:
///   prod_i e(r_i*A_i, B_i) * e(-sum_r*alpha, beta) * e(-aggVKAccum, gamma) * e(-aggC, delta) = 1
public class RecursiveAggregator {
    public init() {}

    /// Aggregate N Groth16 proofs into a recursive aggregation bundle.
    public func aggregate(
        proofs: [(proof: Groth16Proof, publicInputs: [Fr])],
        vk: Groth16VerificationKey
    ) -> RecursiveAggregationBundle {
        let n = proofs.count
        precondition(n > 0, "Must aggregate at least one proof")

        // Generate random challenges
        var challenges = [Fr]()
        challenges.reserveCapacity(n)
        for _ in 0..<n {
            challenges.append(groth16RandomFr())
        }

        var aggVKAccum = pointIdentity()
        var aggC = pointIdentity()
        var challengeSum = Fr.zero

        for i in 0..<n {
            let r = challenges[i]
            let pub = proofs[i].publicInputs
            let proof = proofs[i].proof

            // Compute vk_accum_i
            var vkAccum = vk.ic[0]
            for j in 0..<pub.count {
                if !pub[j].isZero {
                    vkAccum = pointAdd(vkAccum, pointScalarMul(vk.ic[j + 1], pub[j]))
                }
            }

            aggVKAccum = pointAdd(aggVKAccum, pointScalarMul(vkAccum, r))
            aggC = pointAdd(aggC, pointScalarMul(proof.c, r))
            challengeSum = frAdd(challengeSum, r)
        }

        return RecursiveAggregationBundle(
            challenges: challenges,
            proofs: proofs.map { $0.proof },
            allPublicInputs: proofs.map { $0.publicInputs },
            aggregatedVKAccum: aggVKAccum,
            aggregatedC: aggC,
            challengeSum: challengeSum
        )
    }

    /// Verify a recursive aggregation bundle using a multi-pairing check.
    public func verifyAggregated(
        bundle: RecursiveAggregationBundle,
        vk: Groth16VerificationKey
    ) -> Bool {
        let n = bundle.proofs.count
        precondition(bundle.challenges.count == n)

        var pairs = [(PointAffine, G2AffinePoint)]()
        pairs.reserveCapacity(n + 3)

        // e(r_i * A_i, B_i) for each proof
        for i in 0..<n {
            let rA = pointScalarMul(bundle.proofs[i].a, bundle.challenges[i])
            guard let rA_aff = pointToAffine(rA) else { return false }
            guard let B_aff = g2ToAffine(bundle.proofs[i].b) else { return false }
            pairs.append((rA_aff, B_aff))
        }

        // e(-sum_r * alpha, beta)
        let negSumRAlpha = pointNeg(pointScalarMul(vk.alpha_g1, bundle.challengeSum))
        guard let negSumRAlpha_aff = pointToAffine(negSumRAlpha) else { return false }
        guard let beta_aff = g2ToAffine(vk.beta_g2) else { return false }
        pairs.append((negSumRAlpha_aff, beta_aff))

        // e(-aggVKAccum, gamma)
        let negAggVKAccum = pointNeg(bundle.aggregatedVKAccum)
        guard let negAggVKAccum_aff = pointToAffine(negAggVKAccum) else { return false }
        guard let gamma_aff = g2ToAffine(vk.gamma_g2) else { return false }
        pairs.append((negAggVKAccum_aff, gamma_aff))

        // e(-aggC, delta)
        let negAggC = pointNeg(bundle.aggregatedC)
        guard let negAggC_aff = pointToAffine(negAggC) else { return false }
        guard let delta_aff = g2ToAffine(vk.delta_g2) else { return false }
        pairs.append((negAggC_aff, delta_aff))

        return cBN254PairingCheck(pairs)
    }

    /// Convenience: batch verify N proofs in one call.
    public func batchVerify(
        proofs: [(proof: Groth16Proof, publicInputs: [Fr])],
        vk: Groth16VerificationKey
    ) -> Bool {
        let bundle = aggregate(proofs: proofs, vk: vk)
        return verifyAggregated(bundle: bundle, vk: vk)
    }
}

// MARK: - Convenience

/// Verify a single Groth16 proof using the recursive aggregator pipeline.
public func groth16VerifySingle(proof: Groth16Proof, vk: Groth16VerificationKey, publicInputs: [Fr]) -> Bool {
    let aggregator = RecursiveAggregator()
    return aggregator.batchVerify(proofs: [(proof, publicInputs)], vk: vk)
}
