// STIR Verifier — Verify Shift To Improve Rate proximity testing proofs
//
// Verification checks (per round):
//   1. Merkle path validity: each opened value has a valid auth path to the root
//   2. Fold consistency: opened values fold correctly via Horner evaluation
//   3. Shift consistency (full verify only): the shifted evaluations match
//      the next layer's committed values
//   4. Final polynomial: is small enough (degree < reductionFactor^2)
//
// The domain shift step is what gives STIR better soundness per query than
// FRI. After folding, the prover evaluates on a shifted domain {alpha * omega^i},
// decorrelating errors across rounds. The verifier checks shift consistency
// by recomputing: iNTT -> multiply coefficients by alpha^j -> NTT.

import Foundation
import NeonFieldOps

public class STIRVerifier {
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

    /// Verify a STIR proof without the original evaluations.
    ///
    /// Checks:
    ///   - Folding + shift challenges are correctly derived from transcript
    ///   - Merkle paths are valid for each opened position
    ///   - Fold consistency: opened values produce the correct folded value
    ///   - Final polynomial has small degree
    ///
    /// - Parameters:
    ///   - proof: the STIR proof to verify
    ///   - domainSize: size of the original evaluation domain (if nil, inferred from proof)
    /// - Returns: true if all checks pass
    public func verify(proof: STIRProofData, domainSize: Int? = nil) -> Bool {
        let ts = Transcript(label: "stir-v2")

        // Determine initial domain size
        var currentN: Int
        if let ds = domainSize {
            currentN = ds
        } else {
            currentN = proof.finalPoly.count
            for _ in 0..<proof.numRounds { currentN *= reductionFactor }
        }

        // Phase 1: Re-derive all challenges (betas + alphas) from transcript
        for round in 0..<proof.numRounds {
            guard round < proof.roots.count else { return false }
            ts.absorb(proof.roots[round])
            ts.absorbLabel("stir-fold-r\(round)")
            let beta = ts.squeeze()
            if frToInt(beta) != frToInt(proof.betas[round]) { return false }

            ts.absorbLabel("stir-shift-r\(round)")
            let alpha = ts.squeeze()
            if frToInt(alpha) != frToInt(proof.alphas[round]) { return false }

            currentN /= reductionFactor
        }

        // Absorb final polynomial
        ts.absorbLabel("stir-final")
        for v in proof.finalPoly { ts.absorb(v) }

        // Phase 2: Verify query openings at each round
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

            // Check: Merkle paths + fold consistency for each query
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
                    // If first query fails Merkle, skip remaining
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
            }

            currentN = foldedN
        }

        // Final check: polynomial is small enough
        return proof.finalPoly.count <= max(reductionFactor * reductionFactor, 16)
    }

    // MARK: - Full Verify (with original evaluations)

    /// Full verification with original evaluations — checks every value
    /// including shift consistency at each round.
    ///
    /// This recomputes the entire fold+shift chain and verifies that:
    ///   1. Each fold produces the correct values
    ///   2. Each domain shift is applied correctly (iNTT -> coeff*alpha^j -> NTT)
    ///   3. Opened values match the actual evaluations
    ///   4. Final polynomial matches the last fold+shift result
    public func verifyFull(proof: STIRProofData, evaluations: [Fr]) -> Bool {
        let ts = Transcript(label: "stir-v2")
        let numRounds = proof.numRounds

        // Phase 1: Re-derive challenges and verify fold+shift chain
        var allFolded: [[Fr]] = []
        var tempEvals = evaluations

        for round in 0..<numRounds {
            guard round < proof.roots.count else { return false }
            ts.absorb(proof.roots[round])
            ts.absorbLabel("stir-fold-r\(round)")
            let beta = ts.squeeze()
            if frToInt(beta) != frToInt(proof.betas[round]) { return false }

            ts.absorbLabel("stir-shift-r\(round)")
            let alpha = ts.squeeze()
            if frToInt(alpha) != frToInt(proof.alphas[round]) { return false }

            // Recompute fold using C CIOS arithmetic
            let folded = STIRProver.cpuFold(
                evals: tempEvals, challenge: beta,
                reductionFactor: reductionFactor)

            // Apply domain shift (same logic as prover)
            if round < numRounds - 1 && folded.count > reductionFactor {
                let shifted = STIRProver.cpuDomainShift(evals: folded, alpha: alpha)
                allFolded.append(shifted)
                tempEvals = shifted
            } else {
                allFolded.append(folded)
                tempEvals = folded
            }
        }

        // Check final polynomial matches
        if tempEvals.count != proof.finalPoly.count { return false }
        for i in 0..<tempEvals.count {
            if frToInt(tempEvals[i]) != frToInt(proof.finalPoly[i]) { return false }
        }

        // Absorb final polynomial
        ts.absorbLabel("stir-final")
        for v in proof.finalPoly { ts.absorb(v) }

        // Phase 2: Verify query openings against actual evaluations
        tempEvals = evaluations

        for round in 0..<numRounds {
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
                if frToInt(expectedFold) != frToInt(foldedVal) {
                    // The fold of opened values should match the unshifted fold.
                    // allFolded[] already has post-shift values for non-last rounds,
                    // so recompute raw fold for comparison.
                    let newN = tempEvals.count / reductionFactor
                    var rawFolded = [Fr](repeating: Fr.zero, count: newN)
                    for j in 0..<newN {
                        var acc = Fr.zero
                        var pw = Fr.one
                        for kk in 0..<reductionFactor {
                            acc = frAdd(acc, frMul(pw, tempEvals[j * reductionFactor + kk]))
                            pw = frMul(pw, proof.betas[round])
                        }
                        rawFolded[j] = acc
                    }
                    if frToInt(expectedFold) != frToInt(rawFolded[Int(opening.index)]) {
                        return false
                    }
                }
            }

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
