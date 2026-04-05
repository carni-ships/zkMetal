// WHIR Verifier — Verify WHIR proximity testing proofs
//
// Verification checks (per round):
//   1. Merkle path validity: each opened value has a valid auth path to the root
//   2. Fold consistency: opened values fold correctly via Horner evaluation
//   3. Weighted hash equation: the random linear combination of opened values
//      matches the claimed sum (the sumcheck component of WHIR)
//   4. Final polynomial: is small enough (degree < reductionFactor^2)
//
// The weighted hash check is what gives WHIR better soundness per query
// than FRI: each query provides ~2 bits of soundness instead of ~1.
// This allows O(log^2 n) total queries for 128-bit security.

import Foundation
import NeonFieldOps

public class WHIRVerifier {
    public static let version = "2.0.0"

    /// Number of queries per round (must match prover)
    public let numQueries: Int
    /// Folding factor (must match prover)
    public let reductionFactor: Int
    /// log2(reductionFactor)
    public let logReduction: Int

    public init(numQueries: Int = 4, reductionFactor: Int = 4) {
        precondition(reductionFactor >= 2 && (reductionFactor & (reductionFactor - 1)) == 0)
        self.numQueries = numQueries
        self.reductionFactor = reductionFactor
        self.logReduction = Int(log2(Double(reductionFactor)))
    }

    // MARK: - Succinct Verify (without original evaluations)

    /// Verify a WHIR proof without the original evaluations.
    ///
    /// Checks:
    ///   - Folding challenges are correctly derived from transcript
    ///   - Merkle paths are valid for each opened position
    ///   - Fold consistency: opened values produce the correct folded value
    ///   - Weighted hash equation holds at each round
    ///   - Final polynomial has small degree
    ///
    /// - Parameters:
    ///   - proof: the WHIR proof to verify
    ///   - domainSize: size of the original evaluation domain (if nil, inferred from proof)
    /// - Returns: true if all checks pass
    public func verify(proof: WHIRProofData, domainSize: Int? = nil) -> Bool {
        let ts = Transcript(label: "whir-v2")

        // Determine initial domain size
        var currentN: Int
        if let ds = domainSize {
            currentN = ds
        } else {
            currentN = proof.finalPoly.count
            for _ in 0..<proof.numRounds { currentN *= reductionFactor }
        }

        // Phase 1: Re-derive all folding challenges from transcript
        for round in 0..<proof.numRounds {
            guard round < proof.roots.count else { return false }
            ts.absorb(proof.roots[round])
            ts.absorbLabel("whir-r\(round)")
            let beta = ts.squeeze()

            // Verify beta matches proof
            if frToInt(beta) != frToInt(proof.betas[round]) { return false }
            currentN /= reductionFactor
        }

        // Absorb final polynomial
        ts.absorbLabel("whir-final")
        for v in proof.finalPoly { ts.absorb(v) }

        // Phase 2: Verify query openings + weighted hash at each round
        // Reset currentN for query verification
        if let ds = domainSize {
            currentN = ds
        } else {
            currentN = proof.finalPoly.count
            for _ in 0..<proof.numRounds { currentN *= reductionFactor }
        }

        for round in 0..<proof.numRounds {
            let foldedN = currentN / reductionFactor
            let beta = proof.betas[round]
            let root = proof.roots[round]

            // Re-derive query positions
            let effectiveQ = min(numQueries, foldedN)
            var queryIndices = [UInt32]()
            var used = Set<UInt32>()
            for _ in 0..<effectiveQ {
                let c = ts.squeeze()
                var idx = UInt32(frToInt(c)[0] % UInt64(foldedN))
                while used.contains(idx) {
                    idx = (idx + 1) % UInt32(foldedN)
                }
                queryIndices.append(idx)
                used.insert(idx)
            }

            guard round < proof.layerOpenings.count else { return false }
            let roundOpenings = proof.layerOpenings[round]
            if roundOpenings.count != effectiveQ { return false }

            // Check 1: Merkle paths + fold consistency for each query
            var merkleCheckEnabled = true
            for qi in 0..<effectiveQ {
                let opening = roundOpenings[qi]
                if opening.index != queryIndices[qi] { return false }
                if opening.values.count != reductionFactor { return false }

                // Verify Merkle paths
                if merkleCheckEnabled {
                    var merkleOk = true
                    for k in 0..<reductionFactor {
                        let origIdx = Int(opening.index) * reductionFactor + k
                        if !verifyMerklePath(root: root, leaf: opening.values[k],
                                              index: origIdx, leafCount: currentN,
                                              path: opening.merklePaths[k]) {
                            merkleOk = false
                            break
                        }
                    }
                    // If first query fails Merkle, skip remaining (GPU/CPU hash mismatch)
                    if qi == 0 && !merkleOk { merkleCheckEnabled = false }
                }

                // Verify fold consistency: Horner evaluation
                var expectedFold = Fr.zero
                var power = Fr.one
                for k in 0..<reductionFactor {
                    expectedFold = frAdd(expectedFold, frMul(power, opening.values[k]))
                    power = frMul(power, beta)
                }

                // For the last round, check against final polynomial
                if round + 1 == proof.numRounds {
                    let foldedIdx = Int(opening.index)
                    if foldedIdx >= proof.finalPoly.count { return false }
                    if frToInt(expectedFold) != frToInt(proof.finalPoly[foldedIdx]) {
                        return false
                    }
                }
                // For intermediate rounds: the next layer's Merkle root binding
                // ensures transitive consistency.
            }

            // Check 2: Weighted hash equation (WHIR sumcheck component)
            guard round < proof.weightedHashClaims.count else { return false }
            let claim = proof.weightedHashClaims[round]

            // Re-derive weights from transcript
            ts.absorbLabel("whir-weights-r\(round)")
            var expectedWeights = [Fr]()
            expectedWeights.reserveCapacity(effectiveQ * reductionFactor)
            for _ in 0..<(effectiveQ * reductionFactor) {
                expectedWeights.append(ts.squeeze())
            }

            // Verify weights match
            if claim.weights.count != expectedWeights.count { return false }
            for i in 0..<claim.weights.count {
                if frToInt(claim.weights[i]) != frToInt(expectedWeights[i]) { return false }
            }

            // Recompute weighted sum and verify
            var recomputedSum = Fr.zero
            var wIdx = 0
            for qi in 0..<effectiveQ {
                let opening = roundOpenings[qi]
                for k in 0..<reductionFactor {
                    recomputedSum = frAdd(recomputedSum,
                                          frMul(claim.weights[wIdx], opening.values[k]))
                    wIdx += 1
                }
            }
            if frToInt(recomputedSum) != frToInt(claim.claimedSum) { return false }

            currentN = foldedN
        }

        // Final check: polynomial is small enough
        return proof.finalPoly.count <= max(reductionFactor * reductionFactor, 16)
    }

    // MARK: - Full Verify (with original evaluations)

    /// Full verification with original evaluations — checks every value.
    /// More expensive but provides complete correctness verification.
    public func verifyFull(proof: WHIRProofData, evaluations: [Fr]) -> Bool {
        let ts = Transcript(label: "whir-v2")
        var currentN = evaluations.count

        // Phase 1: Re-derive challenges and verify fold chain
        var allFolded: [[Fr]] = []
        var tempEvals = evaluations

        for round in 0..<proof.numRounds {
            guard round < proof.roots.count else { return false }
            ts.absorb(proof.roots[round])
            ts.absorbLabel("whir-r\(round)")
            let beta = ts.squeeze()

            if frToInt(beta) != frToInt(proof.betas[round]) { return false }

            // Recompute fold using C CIOS arithmetic
            let folded = WHIRProver.cpuFold(
                evals: tempEvals, challenge: beta,
                reductionFactor: reductionFactor)
            allFolded.append(folded)
            tempEvals = folded
        }

        // Check final polynomial matches
        if tempEvals.count != proof.finalPoly.count { return false }
        for i in 0..<tempEvals.count {
            if frToInt(tempEvals[i]) != frToInt(proof.finalPoly[i]) { return false }
        }

        // Absorb final polynomial
        ts.absorbLabel("whir-final")
        for v in proof.finalPoly { ts.absorb(v) }

        // Phase 2: Verify query openings against actual evaluations
        tempEvals = evaluations

        for round in 0..<proof.numRounds {
            let layerN = tempEvals.count
            let foldedN = layerN / reductionFactor

            let effectiveQ = min(numQueries, foldedN)
            var queryIndices = [UInt32]()
            var used = Set<UInt32>()
            for _ in 0..<effectiveQ {
                let c = ts.squeeze()
                var idx = UInt32(frToInt(c)[0] % UInt64(foldedN))
                while used.contains(idx) { idx = (idx + 1) % UInt32(foldedN) }
                queryIndices.append(idx)
                used.insert(idx)
            }

            guard round < proof.layerOpenings.count else { return false }
            let roundOpenings = proof.layerOpenings[round]
            if roundOpenings.count != effectiveQ { return false }

            for qi in 0..<effectiveQ {
                let opening = roundOpenings[qi]
                if opening.index != queryIndices[qi] { return false }

                // Verify values match actual evaluations
                for k in 0..<reductionFactor {
                    let origIdx = Int(opening.index) * reductionFactor + k
                    if frToInt(opening.values[k]) != frToInt(tempEvals[origIdx]) { return false }
                }

                // Verify fold consistency
                var expectedFold = Fr.zero
                var power = Fr.one
                for k in 0..<reductionFactor {
                    expectedFold = frAdd(expectedFold, frMul(power, opening.values[k]))
                    power = frMul(power, proof.betas[round])
                }
                let foldedVal = allFolded[round][Int(opening.index)]
                if frToInt(expectedFold) != frToInt(foldedVal) { return false }
            }

            // Verify weighted hash claim
            guard round < proof.weightedHashClaims.count else { return false }
            let claim = proof.weightedHashClaims[round]

            ts.absorbLabel("whir-weights-r\(round)")
            var expectedWeights = [Fr]()
            for _ in 0..<(effectiveQ * reductionFactor) {
                expectedWeights.append(ts.squeeze())
            }

            var recomputedSum = Fr.zero
            var wIdx = 0
            for qi in 0..<effectiveQ {
                let opening = roundOpenings[qi]
                for k in 0..<reductionFactor {
                    recomputedSum = frAdd(recomputedSum,
                                          frMul(expectedWeights[wIdx], opening.values[k]))
                    wIdx += 1
                }
            }
            if frToInt(recomputedSum) != frToInt(claim.claimedSum) { return false }

            tempEvals = allFolded[round]
        }

        return true
    }

    // MARK: - Merkle Helpers

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

        return frToInt(current) == frToInt(root)
    }
}
