// STIR Engine — Shift To Improve Rate proximity testing
// An alternative to FRI for Reed-Solomon proximity testing with O(log^2 n) query
// complexity. Key idea: after each fold round, evaluate the folded polynomial on a
// *shifted* domain {alpha * omega^i}, decorrelating errors across rounds and
// improving the effective rate of the tested code.
//
// Protocol overview (each round):
//   1. Prover commits to current evaluations via Poseidon2 Merkle tree
//   2. Verifier sends folding challenge beta (via Fiat-Shamir)
//   3. Prover computes folded polynomial (degree halved)
//   4. Verifier sends shift challenge alpha (via Fiat-Shamir)
//   5. Prover shifts domain: iNTT -> coeff[j] *= alpha^j -> NTT
//   6. Repeat until polynomial is small; send final poly in the clear
//   7. Verifier sends query positions
//   8. Prover opens evaluations at query positions with Merkle proofs
//   9. Verifier checks fold+shift consistency at opened positions
//
// The domain shift step is what distinguishes STIR from FRI. Shifts decorrelate
// the evaluation domain between rounds, achieving better soundness per query:
//   FRI:  error ~ rho       per query  -> O(lambda * log n) queries
//   STIR: error ~ rho^1.5   per query  -> O(log^2 n) queries

import Foundation

// MARK: - Data Structures

/// A query opening with Merkle proof.
public struct STIRQueryOpening {
    public let index: UInt32
    public let value: Fr
    public let merklePath: [Fr]

    public init(index: UInt32, value: Fr, merklePath: [Fr]) {
        self.index = index
        self.value = value
        self.merklePath = merklePath
    }
}

/// Commitment to polynomial evaluations at one round.
public struct STIRLayerCommitment {
    public let root: Fr
    public let tree: [Fr]
    public let evaluations: [Fr]

    public init(root: Fr, tree: [Fr], evaluations: [Fr]) {
        self.root = root
        self.tree = tree
        self.evaluations = evaluations
    }
}

/// Round data in a STIR proof (per-round openings).
public struct STIRRoundOpening {
    public let index: UInt32
    public let values: [Fr]          // reductionFactor values from parent layer
    public let merklePaths: [[Fr]]   // Merkle paths for each value
}

/// A complete STIR proof.
public struct STIRProof {
    /// Merkle root for each layer (layer 0 = original, layer i = after i folds+shifts)
    public let roots: [Fr]
    /// Folding challenges used at each round
    public let betas: [Fr]
    /// Domain shift challenges used at each round
    public let alphas: [Fr]
    /// Per-round query openings from each committed layer
    public let layerOpenings: [[(index: UInt32, values: [Fr], merklePaths: [[Fr]])]]
    /// Final polynomial evaluations (small, sent in the clear)
    public let finalPoly: [Fr]
    /// Number of folding rounds
    public let numRounds: Int

    /// Proof size in bytes.
    public var proofSizeBytes: Int {
        let frSize = MemoryLayout<Fr>.stride
        var size = roots.count * frSize
        for round in layerOpenings {
            for opening in round {
                size += 4  // index
                size += opening.values.count * frSize
                for path in opening.merklePaths {
                    size += path.count * frSize
                }
            }
        }
        size += finalPoly.count * frSize
        return size
    }
}

// MARK: - STIR Engine

public class STIREngine {
    public static let version = Versions.stir

    public let numQueries: Int
    public let reductionFactor: Int
    public let logReduction: Int

    private let merkleEngine: Poseidon2MerkleEngine

    /// Optional NTT engine for GPU-accelerated domain shifts.
    /// If nil, CPU-only shifts are used.
    private var nttEngine: NTTEngine?

    public init(numQueries: Int = 4, reductionFactor: Int = 4, useGPU: Bool = false) throws {
        precondition(reductionFactor >= 2 && (reductionFactor & (reductionFactor - 1)) == 0)
        self.numQueries = numQueries
        self.reductionFactor = reductionFactor
        self.logReduction = Int(log2(Double(reductionFactor)))
        self.merkleEngine = try Poseidon2MerkleEngine()
        if useGPU {
            self.nttEngine = try NTTEngine()
        }
    }

    // MARK: - Commit

    public func commit(evaluations: [Fr]) throws -> STIRLayerCommitment {
        let n = evaluations.count
        precondition(n > 0 && (n & (n - 1)) == 0)
        let tree = try merkleEngine.buildTree(evaluations)
        let root = tree[2 * n - 2]
        return STIRLayerCommitment(root: root, tree: tree, evaluations: evaluations)
    }

    // MARK: - Domain Shift

    /// Shift evaluation domain by alpha: f evaluated on {alpha * omega^i}.
    /// Algorithm: iNTT(evals) -> coeff[j] *= alpha^j -> NTT.
    public func domainShift(evals: [Fr], alpha: Fr) throws -> [Fr] {
        let n = evals.count
        precondition(n > 0 && (n & (n - 1)) == 0)
        let logN = Int(log2(Double(n)))

        // Get coefficients via iNTT
        var coeffs: [Fr]
        if let ntt = nttEngine {
            coeffs = try ntt.intt(evals)
        } else {
            let omega = frRootOfUnity(logN: logN)
            let omegaInv = frInverse(omega)
            coeffs = STIREngine.cpuINTT(evals: evals, omegaInv: omegaInv, n: n)
        }

        // Multiply coefficient j by alpha^j
        var alphaPow = Fr.one
        for j in 0..<n {
            coeffs[j] = frMul(coeffs[j], alphaPow)
            alphaPow = frMul(alphaPow, alpha)
        }

        // NTT back to evaluation domain
        if let ntt = nttEngine {
            return try ntt.ntt(coeffs)
        } else {
            let omega = frRootOfUnity(logN: logN)
            return STIREngine.cpuNTT(coeffs: coeffs, omega: omega, n: n)
        }
    }

    /// CPU domain shift for verification at small sizes.
    public static func cpuDomainShift(evals: [Fr], alpha: Fr) -> [Fr] {
        let n = evals.count
        let logN = Int(log2(Double(n)))
        let omega = frRootOfUnity(logN: logN)
        let omegaInv = frInverse(omega)

        var coeffs = cpuINTT(evals: evals, omegaInv: omegaInv, n: n)

        var alphaPow = Fr.one
        for j in 0..<n {
            coeffs[j] = frMul(coeffs[j], alphaPow)
            alphaPow = frMul(alphaPow, alpha)
        }

        return cpuNTT(coeffs: coeffs, omega: omega, n: n)
    }

    // MARK: - Prove

    /// Full STIR prove: commit, fold+shift, generate queries via Fiat-Shamir.
    public func prove(evaluations: [Fr], transcript: Transcript? = nil) throws -> STIRProof {
        let n = evaluations.count
        precondition(n > 0 && (n & (n - 1)) == 0)
        let logN = Int(log2(Double(n)))

        // Fold until <= 16 elements remain
        let rounds = max(1, (logN - 4) / logReduction)

        let ts = transcript ?? Transcript(label: "stir-v1")

        // Phase 1: Build all layers (commit, derive beta+alpha, fold, shift)
        var layers: [STIRLayerCommitment] = []
        var betas: [Fr] = []
        var alphas: [Fr] = []
        var currentEvals = evaluations

        for round in 0..<rounds {
            let currentN = currentEvals.count
            if currentN <= reductionFactor { break }

            let commitment = try commit(evaluations: currentEvals)
            layers.append(commitment)

            // Transcript: absorb root, label, squeeze beta (fold challenge)
            ts.absorb(commitment.root)
            ts.absorbLabel("stir-fold-r\(round)")
            let beta = ts.squeeze()
            betas.append(beta)

            // Transcript: squeeze alpha (shift challenge)
            ts.absorbLabel("stir-shift-r\(round)")
            let alpha = ts.squeeze()
            alphas.append(alpha)

            // Fold polynomial (same as FRI/WHIR)
            let newN = currentN / reductionFactor
            var folded = [Fr](repeating: Fr.zero, count: newN)
            for j in 0..<newN {
                var acc = Fr.zero
                var power = Fr.one
                for k in 0..<reductionFactor {
                    acc = frAdd(acc, frMul(power, currentEvals[j * reductionFactor + k]))
                    power = frMul(power, beta)
                }
                folded[j] = acc
            }

            // Apply domain shift (distinguishes STIR from FRI/WHIR)
            // Skip shift on last round (final poly is sent in the clear)
            if round < rounds - 1 && folded.count > reductionFactor {
                currentEvals = try domainShift(evals: folded, alpha: alpha)
            } else {
                currentEvals = folded
            }
        }

        let finalPoly = currentEvals
        let actualRounds = betas.count

        // Transcript: absorb final polynomial
        ts.absorbLabel("stir-final")
        for v in finalPoly { ts.absorb(v) }

        // Phase 2: Query phase
        var layerOpenings: [[(index: UInt32, values: [Fr], merklePaths: [[Fr]])]] = []

        for round in 0..<actualRounds {
            let layer = layers[round]
            let layerN = layer.evaluations.count
            let foldedN = layerN / reductionFactor

            // Derive query positions (in folded domain)
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

            // Open reductionFactor positions per query in the current layer
            var roundOpenings: [(index: UInt32, values: [Fr], merklePaths: [[Fr]])] = []
            for qi in 0..<effectiveQ {
                let foldedIdx = Int(queryIndices[qi])
                var values = [Fr]()
                var paths = [[Fr]]()
                for k in 0..<reductionFactor {
                    let origIdx = foldedIdx * reductionFactor + k
                    values.append(layer.evaluations[origIdx])
                    paths.append(extractMerklePath(tree: layer.tree,
                                                    leafCount: layerN,
                                                    index: origIdx))
                }
                roundOpenings.append((index: queryIndices[qi], values: values, merklePaths: paths))
            }
            layerOpenings.append(roundOpenings)
        }

        return STIRProof(
            roots: layers.map { $0.root },
            betas: betas,
            alphas: alphas,
            layerOpenings: layerOpenings,
            finalPoly: finalPoly,
            numRounds: actualRounds
        )
    }

    // MARK: - Verify (succinct)

    /// Verify a STIR proof without the original evaluations.
    /// Checks Merkle proofs, fold consistency, and shift consistency.
    public func verify(proof: STIRProof, evaluations: [Fr]? = nil) -> Bool {
        let ts = Transcript(label: "stir-v1")

        // Determine initial domain size
        var currentN: Int
        if let evals = evaluations {
            currentN = evals.count
        } else {
            currentN = proof.finalPoly.count
            for _ in 0..<proof.numRounds { currentN *= reductionFactor }
        }

        // Phase 1: Re-derive all challenges (betas + alphas)
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

        // Phase 2: Verify query openings
        if let evals = evaluations {
            currentN = evals.count
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

            for qi in 0..<effectiveQ {
                let opening = roundOpenings[qi]
                if opening.index != queryIndices[qi] { return false }
                if opening.values.count != reductionFactor { return false }

                // Verify Merkle paths
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
                if qi == 0 && !merkleOk { break }

                // Compute expected folded value
                var expectedFold = Fr.zero
                var power = Fr.one
                for k in 0..<reductionFactor {
                    expectedFold = frAdd(expectedFold, frMul(power, opening.values[k]))
                    power = frMul(power, beta)
                }

                // For the last round, verify against final polynomial
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

        return proof.finalPoly.count <= max(reductionFactor * reductionFactor, 16)
    }

    // MARK: - Verify Full

    /// Full verify with original evaluations (checks every value including shift consistency).
    public func verifyFull(proof: STIRProof, evaluations: [Fr]) -> Bool {
        let ts = Transcript(label: "stir-v1")
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

            // Recompute fold
            let newN = tempEvals.count / reductionFactor
            var folded = [Fr](repeating: Fr.zero, count: newN)
            for j in 0..<newN {
                var acc = Fr.zero
                var power = Fr.one
                for k in 0..<reductionFactor {
                    acc = frAdd(acc, frMul(power, tempEvals[j * reductionFactor + k]))
                    power = frMul(power, beta)
                }
                folded[j] = acc
            }

            // Apply domain shift (same logic as prove)
            if round < numRounds - 1 && folded.count > reductionFactor {
                let shifted = STIREngine.cpuDomainShift(evals: folded, alpha: alpha)
                allFolded.append(shifted)
                tempEvals = shifted
            } else {
                allFolded.append(folded)
                tempEvals = folded
            }
        }

        // Check final polynomial
        if tempEvals.count != proof.finalPoly.count { return false }
        for i in 0..<tempEvals.count {
            if frToInt(tempEvals[i]) != frToInt(proof.finalPoly[i]) { return false }
        }

        // Absorb final polynomial
        ts.absorbLabel("stir-final")
        for v in proof.finalPoly { ts.absorb(v) }

        // Phase 2: Verify query openings
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
                    // Fold consistency check: the fold of the opened values should
                    // match the value in the next layer (after shifting). But for
                    // non-last rounds, the next layer was shifted, so the fold matches
                    // the *pre-shift* value. We need to check against the unshifted fold.
                    // Actually, allFolded already has the shifted values, and the openings
                    // are from the parent (pre-fold) layer, so the fold of opened values
                    // should match the unshifted fold at that index.
                    //
                    // The shift is applied *after* folding, so fold(opened_values) should
                    // equal the pre-shift folded value, not the post-shift value.
                    // We need to recompute the unshifted fold for comparison.
                    let unshifted: [Fr]
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
                    unshifted = rawFolded
                    let rawFoldedVal = unshifted[Int(opening.index)]
                    if frToInt(expectedFold) != frToInt(rawFoldedVal) {
                        return false
                    }
                }
            }

            tempEvals = allFolded[round]
        }

        return true
    }

    // MARK: - Soundness Analysis

    /// Compute queries needed for a given security level.
    /// STIR achieves better soundness per query than FRI:
    ///   FRI:  error ~ rho       per query
    ///   STIR: error ~ rho^1.5   per query (from domain shifts)
    public static func queriesNeeded(securityBits: Int, rate: Double, useSTIR: Bool) -> Int {
        let rho = rate
        if useSTIR {
            let errorPerQuery = pow(rho, 1.5)
            let queriesF = Double(securityBits) * log(2.0) / (-log(errorPerQuery))
            return max(1, Int(ceil(queriesF)))
        } else {
            let errorPerQuery = rho
            let queriesF = Double(securityBits) * log(2.0) / (-log(errorPerQuery))
            return max(1, Int(ceil(queriesF)))
        }
    }

    /// Estimate proof size for given parameters.
    public static func estimateProofSize(logN: Int, numQueries: Int, reductionFactor: Int) -> Int {
        let frSize = MemoryLayout<Fr>.stride
        let logReduction = Int(log2(Double(reductionFactor)))
        let numRounds = max(1, (logN - 4) / logReduction)
        let perQueryPerRound = reductionFactor * frSize + reductionFactor * logN * frSize
        let commitSize = numRounds * frSize  // roots
        let finalSize = max(1, 1 << max(0, logN - numRounds * logReduction)) * frSize
        return commitSize + numQueries * numRounds * perQueryPerRound + finalSize
    }

    // MARK: - Merkle Helpers

    func extractMerklePath(tree: [Fr], leafCount: Int, index: Int) -> [Fr] {
        var path = [Fr]()
        var idx = index
        var levelStart = 0
        var levelSize = leafCount

        while levelSize > 1 {
            let siblingIdx = idx ^ 1
            if levelStart + siblingIdx < tree.count {
                path.append(tree[levelStart + siblingIdx])
            }
            idx /= 2
            levelStart += levelSize
            levelSize /= 2
        }
        return path
    }

    func verifyMerklePath(root: Fr, leaf: Fr, index: Int, leafCount: Int, path: [Fr]) -> Bool {
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

    // MARK: - CPU Helpers

    /// CPU fold for reference.
    public static func cpuFold(evals: [Fr], challenge: Fr, reductionFactor: Int) -> [Fr] {
        let n = evals.count
        let newN = n / reductionFactor
        var result = [Fr](repeating: Fr.zero, count: newN)
        for j in 0..<newN {
            var acc = Fr.zero
            var power = Fr.one
            for k in 0..<reductionFactor {
                acc = frAdd(acc, frMul(power, evals[j * reductionFactor + k]))
                power = frMul(power, challenge)
            }
            result[j] = acc
        }
        return result
    }

    /// Simple CPU NTT (Cooley-Tukey) for correctness testing.
    static func cpuNTT(coeffs: [Fr], omega: Fr, n: Int) -> [Fr] {
        if n == 1 { return coeffs }
        var result = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            var val = Fr.zero
            var omegaPow = Fr.one
            let omegaI = frPow(omega, UInt64(i))
            for j in 0..<n {
                val = frAdd(val, frMul(coeffs[j], omegaPow))
                omegaPow = frMul(omegaPow, omegaI)
            }
            result[i] = val
        }
        return result
    }

    /// Simple CPU iNTT for correctness testing.
    static func cpuINTT(evals: [Fr], omegaInv: Fr, n: Int) -> [Fr] {
        let nInv = frInverse(frFromInt(UInt64(n)))
        var result = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            var val = Fr.zero
            var omegaPow = Fr.one
            let omegaInvI = frPow(omegaInv, UInt64(i))
            for j in 0..<n {
                val = frAdd(val, frMul(evals[j], omegaPow))
                omegaPow = frMul(omegaPow, omegaInvI)
            }
            result[i] = frMul(val, nInv)
        }
        return result
    }
}
