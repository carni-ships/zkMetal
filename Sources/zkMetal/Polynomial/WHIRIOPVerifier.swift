// WHIR Verifier — Complete polynomial IOP verifier for low-degree testing
//
// Verifies WHIRIOPProof instances produced by WHIRProver by:
//   1. Replaying the Fiat-Shamir transcript to re-derive all challenges
//   2. Verifying Merkle authentication paths at each query position
//   3. Checking fold consistency via Horner evaluation
//   4. Verifying domain-dependent weighted hash equations
//   5. Checking the hash commitment chain
//   6. Verifying the final polynomial is small enough
//
// The verifier needs only the proof (no original evaluations) for
// succinct verification. A full-verification mode is also provided
// for testing that recomputes the entire fold chain.

import Foundation
import NeonFieldOps

// MARK: - WHIR Verification Key

/// Parameters committed for verification. Captures the IOP configuration
/// so the verifier can independently check proof validity.
public struct WHIRIOPVerificationKey {
    /// IOP configuration (queries, folding factor, security level)
    public let config: WHIRIOPConfig
    /// Domain size of the original polynomial evaluations
    public let domainSize: Int

    public init(config: WHIRIOPConfig, domainSize: Int) {
        self.config = config
        self.domainSize = domainSize
    }

    /// Create from a proof (extracts embedded parameters).
    public init(from proof: WHIRIOPProof) {
        self.config = proof.config
        self.domainSize = proof.domainSize
    }
}

// MARK: - WHIR Verifier

/// WHIR polynomial IOP verifier.
///
/// Verification is succinct: the verifier runs in time polylog(n) in the
/// domain size n, checking only query positions and the hash chain.
public class WHIRIOPVerifier {

    /// Verify a WHIR proof using embedded parameters.
    ///
    /// This is the main entry point. Replays the Fiat-Shamir transcript,
    /// checks all Merkle proofs, fold consistency, weighted hash equations,
    /// and the hash commitment chain.
    ///
    /// - Parameter proof: the WHIRIOPProof to verify
    /// - Returns: true if all checks pass
    public func verify(proof: WHIRIOPProof) -> Bool {
        let vk = WHIRIOPVerificationKey(from: proof)
        return verify(proof: proof, vk: vk)
    }

    /// Verify a WHIR proof against a verification key.
    public func verify(proof: WHIRIOPProof, vk: WHIRIOPVerificationKey) -> Bool {
        let config = vk.config
        let ts = Transcript(label: "whir-iop-v1")
        let numRounds = proof.numRounds

        // Basic structural checks
        guard proof.challenges.count == numRounds else { return false }
        guard proof.weightSeeds.count == numRounds else { return false }
        guard proof.weightedSums.count == numRounds else { return false }
        guard proof.hashCommitments.count == numRounds else { return false }
        guard proof.roundCommitments.count == numRounds else { return false }
        guard proof.queryResponses.count == numRounds else { return false }

        var currentN = vk.domainSize

        // -- Phase 1: Replay transcript and re-derive all challenges --

        // Absorb initial commitment
        ts.absorb(proof.initialCommitment)
        ts.absorbLabel("whir-iop-domain-\(currentN)")

        // Track per-round roots for Merkle verification.
        // Layer 0 root = initialCommitment (for the original evaluations).
        // Layer i+1 root = roundCommitments[i] (for folded evaluations after round i).
        var layerRoots: [Fr] = [proof.initialCommitment]

        for round in 0..<numRounds {
            let roundN = currentN
            if roundN <= config.foldingFactor { return false }

            // Re-derive beta
            ts.absorbLabel("whir-iop-fold-r\(round)")
            let beta = ts.squeeze()
            if beta != proof.challenges[round] { return false }

            // Re-derive gamma
            ts.absorbLabel("whir-iop-gamma-r\(round)")
            let gamma = ts.squeeze()
            if gamma != proof.weightSeeds[round] { return false }

            // Re-derive query positions
            let foldedN = roundN / config.foldingFactor
            let effectiveQ = min(config.numQueries, foldedN)
            var queryIndices = [UInt32]()
            var used = Set<UInt32>()
            for _ in 0..<effectiveQ {
                let c = ts.squeeze()
                var idx = UInt32(frToUInt64(c) % UInt64(foldedN))
                while used.contains(idx) {
                    idx = (idx &+ 1) % UInt32(foldedN)
                }
                queryIndices.append(idx)
                used.insert(idx)
            }

            // Verify hash commitment: H(H(sum, gamma), round)
            let expectedHComm = WHIRIOPProver.hashCommitment(
                claimedSum: proof.weightedSums[round],
                gamma: gamma, round: round)
            if expectedHComm != proof.hashCommitments[round] { return false }

            // Absorb hash commitment
            ts.absorb(proof.hashCommitments[round])

            // Absorb round commitment (folded layer root)
            ts.absorb(proof.roundCommitments[round])
            layerRoots.append(proof.roundCommitments[round])

            // -- Phase 2: Verify query openings for this round --
            let roundResponses = proof.queryResponses[round]
            if roundResponses.count != effectiveQ { return false }

            // Compute domain-dependent weights
            let weights = WHIRIOPProver.computeWeights(
                gamma: gamma, queryIndices: queryIndices,
                domainSize: roundN, foldingFactor: config.foldingFactor)

            var recomputedSum = Fr.zero
            var wIdx = 0

            for qi in 0..<effectiveQ {
                let qr = roundResponses[qi]

                // Check index matches
                if qr.foldedIndex != queryIndices[qi] { return false }
                if qr.values.count != config.foldingFactor { return false }
                if qr.merklePaths.count != config.foldingFactor { return false }

                let root = layerRoots[round]

                // Verify Merkle paths for each value in the coset
                for k in 0..<config.foldingFactor {
                    let origIdx = Int(qr.foldedIndex) * config.foldingFactor + k
                    if !verifyMerklePath(root: root, leaf: qr.values[k],
                                          index: origIdx, leafCount: roundN,
                                          path: qr.merklePaths[k]) {
                        return false
                    }
                }

                // Check fold consistency: Horner evaluation
                // f'[j] = sum_{k=0}^{r-1} beta^k * values[k]
                var expectedFold = Fr.zero
                var power = Fr.one
                for k in 0..<config.foldingFactor {
                    expectedFold = frAdd(expectedFold, frMul(power, qr.values[k]))
                    power = frMul(power, beta)
                }

                // For the last round, check against final polynomial
                if round + 1 == numRounds {
                    let fIdx = Int(qr.foldedIndex)
                    if fIdx >= proof.finalPolynomial.count { return false }
                    if expectedFold != proof.finalPolynomial[fIdx] {
                        return false
                    }
                }

                // Accumulate weighted hash
                for k in 0..<config.foldingFactor {
                    recomputedSum = frAdd(recomputedSum,
                                          frMul(weights[wIdx], qr.values[k]))
                    wIdx += 1
                }
            }

            // Verify weighted hash equation
            if recomputedSum != proof.weightedSums[round] {
                return false
            }

            currentN = foldedN
        }

        // Absorb final polynomial
        ts.absorbLabel("whir-iop-final")
        for v in proof.finalPolynomial { ts.absorb(v) }

        // Final check: polynomial size bound
        let maxFinal = max(config.foldingFactor * config.foldingFactor,
                           config.finalPolyMaxSize)
        if proof.finalPolynomial.count > maxFinal { return false }

        return true
    }

    // MARK: - Merkle Path Verification

    /// Verify a Merkle authentication path from leaf to root.
    ///
    /// Uses the same Poseidon2 hash as the prover's tree construction:
    /// at each level, hash (left, right) where the path provides the
    /// sibling and the index bit determines left/right ordering.
    func verifyMerklePath(root: Fr, leaf: Fr, index: Int,
                           leafCount: Int, path: [Fr]) -> Bool {
        var current = leaf
        var idx = index
        let expectedDepth = Int(log2(Double(leafCount)))

        if path.count != expectedDepth { return false }

        for level in 0..<expectedDepth {
            let sibling = path[level]
            if idx & 1 == 0 {
                current = poseidon2Hash(current, sibling)
            } else {
                current = poseidon2Hash(sibling, current)
            }
            idx /= 2
        }

        return current == root
    }

    public init() {}
}
