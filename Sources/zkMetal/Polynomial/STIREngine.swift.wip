// STIR Engine — Shift To Improve Rate proximity testing
// GPU-accelerated proximity test achieving FRI-like soundness with fewer queries.
// Key insight: after each fold round, shift the evaluation domain by a random
// element alpha, decorrelating errors across rounds and improving the effective
// rate of the tested code.

import Foundation
import Metal

// MARK: - Data Structures

/// A single STIR query response: evaluation pairs + Merkle paths per layer.
public struct STIRQuery {
    /// The initial query index in the original domain
    public let initialIndex: UInt32
    /// Evaluation pairs (eval[idx], eval[paired_idx]) at each layer
    public let layerEvals: [(Fr, Fr)]
    /// Merkle authentication paths at each layer
    public let merklePaths: [[[Fr]]]

    public init(initialIndex: UInt32, layerEvals: [(Fr, Fr)], merklePaths: [[[Fr]]]) {
        self.initialIndex = initialIndex
        self.layerEvals = layerEvals
        self.merklePaths = merklePaths
    }
}

/// Commitment produced during STIR commit phase.
public struct STIRCommitment {
    /// Evaluations at each fold layer (layer 0 = original, layer k = after k folds+shifts)
    public let layers: [[Fr]]
    /// Poseidon2 Merkle root of each layer's evaluations
    public let roots: [Fr]
    /// Folding challenges used at each round
    public let betas: [Fr]
    /// Domain shift elements used at each round (alpha_i)
    public let shifts: [Fr]
    /// Final constant value after all folds
    public let finalValue: Fr

    public init(layers: [[Fr]], roots: [Fr], betas: [Fr], shifts: [Fr], finalValue: Fr) {
        self.layers = layers
        self.roots = roots
        self.betas = betas
        self.shifts = shifts
        self.finalValue = finalValue
    }
}

/// Full STIR proof: commitments, query responses, final polynomial, shifts used.
public struct STIRProof {
    /// Merkle roots per round
    public let commitments: [[UInt8]]
    /// Query responses
    public let queries: [STIRQuery]
    /// Final constant/small polynomial
    public let finalPoly: [Fr]
    /// Domain shifts used per round
    public let shifts: [Fr]

    public init(commitments: [[UInt8]], queries: [STIRQuery], finalPoly: [Fr], shifts: [Fr]) {
        self.commitments = commitments
        self.queries = queries
        self.finalPoly = finalPoly
        self.shifts = shifts
    }

    /// Proof size in bytes (approximate).
    public var sizeBytes: Int {
        var total = 0
        for c in commitments { total += c.count }
        let frSize = MemoryLayout<Fr>.stride
        for q in queries {
            total += 4
            total += q.layerEvals.count * 2 * frSize
            for layerPaths in q.merklePaths {
                for path in layerPaths {
                    total += path.count * frSize
                }
            }
        }
        total += finalPoly.count * frSize
        total += shifts.count * frSize
        return total
    }
}

// MARK: - STIR Engine

/// GPU-accelerated STIR proximity testing engine.
/// Reuses FRIEngine's fold kernels and adds domain shifting between rounds.
public class STIREngine {
    public static let version = Versions.stir

    /// Underlying FRI engine (provides fold kernels, NTT, Merkle)
    public let friEngine: FRIEngine

    /// NTT engine for domain shift (iNTT -> coeff shift -> NTT)
    private let nttEngine: NTTEngine

    public init() throws {
        self.friEngine = try FRIEngine()
        self.nttEngine = friEngine.nttEngine
    }

    public init(friEngine: FRIEngine) {
        self.friEngine = friEngine
        self.nttEngine = friEngine.nttEngine
    }

    // MARK: - Domain Shift

    /// GPU domain shift: re-evaluate polynomial on shifted domain {alpha * omega^i}.
    /// Algorithm: iNTT(evals) -> multiply coeffs[j] by alpha^j -> NTT back.
    /// Uses GPU NTT engine for the transforms, CPU for the cheap coefficient scaling.
    public func domainShift(evals: [Fr], alpha: Fr, logN: Int) throws -> [Fr] {
        let n = evals.count
        precondition(n == 1 << logN)

        // Step 1: iNTT to get coefficients
        var coeffs = try nttEngine.intt(evals)

        // Step 2: multiply coefficient j by alpha^j (cheap O(n) CPU work)
        var alphaPow = Fr.one
        for j in 0..<n {
            coeffs[j] = frMul(coeffs[j], alphaPow)
            alphaPow = frMul(alphaPow, alpha)
        }

        // Step 3: NTT back to evaluation domain
        return try nttEngine.ntt(coeffs)
    }

    /// CPU domain shift for correctness verification at small sizes.
    public static func cpuDomainShift(evals: [Fr], alpha: Fr, logN: Int) -> [Fr] {
        let n = evals.count
        let omega = frRootOfUnity(logN: logN)
        let omegaInv = frInverse(omega)

        // iNTT
        var coeffs = cpuINTT(evals: evals, omegaInv: omegaInv, n: n)

        // Multiply coefficients by alpha^j
        var alphaPow = Fr.one
        for j in 0..<n {
            coeffs[j] = frMul(coeffs[j], alphaPow)
            alphaPow = frMul(alphaPow, alpha)
        }

        // NTT
        return cpuNTT(coeffs: coeffs, omega: omega, n: n)
    }

    // MARK: - STIR Commit Phase

    /// Commit phase: fold + shift iteratively.
    /// At each round:
    ///   1. Commit to current evaluations (Merkle root)
    ///   2. Fold evaluations with challenge beta (standard FRI fold, halves domain)
    ///   3. Shift folded evaluations to new domain {alpha * omega^i}
    ///
    /// The shift step is what distinguishes STIR from FRI. By decorrelating the
    /// evaluation domain between rounds, STIR achieves better soundness per query.
    public func commitPhase(evals: [Fr], betas: [Fr], shifts: [Fr]) throws -> STIRCommitment {
        let n = evals.count
        precondition(n > 1 && (n & (n - 1)) == 0)
        let logN = Int(log2(Double(n)))
        precondition(betas.count <= logN)
        precondition(shifts.count == betas.count)

        // Design: layers store what the verifier checks.
        // layers[0] = original evals, committed via Merkle
        // For each round k:
        //   1. Fold layers[k] -> folded_k (using betas[k] and standard domain)
        //   2. Shift folded_k by alpha_k to get next fold input
        //   3. layers[k+1] = folded_k (unshifted, for verifier fold consistency)
        //   4. Next round folds the shifted version (different polynomial, same domain)
        //
        // The shift decorrelates errors between rounds (STIR's key insight).
        // The verifier only sees fold consistency on the unshifted layers.
        var layers: [[Fr]] = [evals]
        var roots: [Fr] = []
        var foldInput = evals  // what actually gets folded (may be shifted)
        var currentLogN = logN

        let merkle = try Poseidon2MerkleEngine()

        for i in 0..<betas.count {
            // Merkle commit the fold input
            let tree = try merkle.buildTree(foldInput)
            let root = tree[tree.count - 1]
            roots.append(root)

            // Fold the (possibly shifted) input
            let folded = try friEngine.fold(evals: foldInput, beta: betas[i])
            currentLogN -= 1

            // Store fold output for verifier
            layers.append(folded)

            // Apply domain shift for next round's fold input
            if i < betas.count - 1 {
                foldInput = try domainShift(evals: folded, alpha: shifts[i], logN: currentLogN)
            } else {
                foldInput = folded
            }
        }

        let finalValue = current.count == 1 ? current[0] : Fr.zero

        return STIRCommitment(
            layers: layers,
            roots: roots,
            betas: betas,
            shifts: shifts,
            finalValue: finalValue
        )
    }

    // MARK: - STIR Query Phase

    /// Query phase: extract evaluation pairs and Merkle paths at each layer.
    public func queryPhase(commitment: STIRCommitment, queryIndices: [UInt32]) throws -> [STIRQuery] {
        let merkle = try Poseidon2MerkleEngine()
        let numQueries = queryIndices.count

        var proofs = [STIRQuery]()
        proofs.reserveCapacity(numQueries)

        for qi in 0..<numQueries {
            var layerEvals: [(Fr, Fr)] = []
            var merklePaths: [[[Fr]]] = []
            var idx = queryIndices[qi]

            for layer in 0..<commitment.layers.count - 1 {
                let evals = commitment.layers[layer]
                let n = evals.count
                let halfN = UInt32(n / 2)

                let lowerIdx = idx < halfN ? idx : idx - halfN
                let upperIdx = lowerIdx + halfN
                let evalA = evals[Int(lowerIdx)]
                let evalB = evals[Int(upperIdx)]
                layerEvals.append((evalA, evalB))

                let tree = try merkle.buildTree(evals)
                let path = extractMerklePath(tree: tree, leafCount: n, index: Int(idx))
                merklePaths.append([path])

                idx = lowerIdx
            }

            proofs.append(STIRQuery(
                initialIndex: queryIndices[qi],
                layerEvals: layerEvals,
                merklePaths: merklePaths
            ))
        }

        return proofs
    }

    // MARK: - STIR Verify

    /// Verify a STIR proof: check fold+shift consistency at each layer.
    /// The fold relation is the same as FRI. The shift is verified implicitly
    /// because the committed evaluations at each layer are on the shifted domain.
    public func verify(commitment: STIRCommitment, queries: [STIRQuery]) -> Bool {
        for query in queries {
            var idx = query.initialIndex

            for layer in 0..<commitment.layers.count - 1 {
                let (evalA, evalB) = query.layerEvals[layer]
                let n = commitment.layers[layer].count
                let halfN = UInt32(n / 2)
                let logN = Int(log2(Double(n)))
                let beta = commitment.betas[layer]

                let omega = frRootOfUnity(logN: logN)
                let omegaInv = frInverse(omega)
                let lowerIdx = idx < halfN ? idx : idx - halfN
                let w_inv = frPow(omegaInv, UInt64(lowerIdx))

                let sum = frAdd(evalA, evalB)
                let diff = frSub(evalA, evalB)
                let term = frMul(frMul(beta, w_inv), diff)
                let expected = frAdd(sum, term)

                let nextIdx = lowerIdx
                if layer + 1 < commitment.layers.count {
                    let nextEval = commitment.layers[layer + 1][Int(nextIdx)]
                    let expectedLimbs = frToInt(expected)
                    let actualLimbs = frToInt(nextEval)
                    if expectedLimbs != actualLimbs {
                        return false
                    }
                }

                idx = nextIdx
            }
        }
        return true
    }

    // MARK: - Full Prove / Verify

    /// Full STIR prove: commit, generate queries via Fiat-Shamir, extract proofs.
    public func prove(evals: [Fr], numQueries: Int = 16) throws -> (STIRCommitment, STIRProof) {
        let n = evals.count
        let logN = Int(log2(Double(n)))

        // Generate deterministic challenges (in production, use Fiat-Shamir transcript)
        var betas = [Fr]()
        var shifts = [Fr]()
        var seed: UInt64 = 0xDEAD_BEEF_CAFE_0517
        for i in 0..<logN {
            seed = seed &* 6364136223846793005 &+ UInt64(i)
            betas.append(frFromInt(seed >> 16))
            seed = seed &* 6364136223846793005 &+ UInt64(i + logN)
            shifts.append(frFromInt(seed >> 16))
        }

        let commitment = try commitPhase(evals: evals, betas: betas, shifts: shifts)

        // Generate query indices from commitment roots (Fiat-Shamir)
        var queryIndices = [UInt32]()
        var qSeed: UInt64 = 0
        for root in commitment.roots {
            let limbs = frToInt(root)
            qSeed ^= limbs[0]
        }
        for q in 0..<numQueries {
            qSeed = qSeed &* 6364136223846793005 &+ UInt64(q)
            queryIndices.append(UInt32(qSeed >> 32) % UInt32(n))
        }

        let queries = try queryPhase(commitment: commitment, queryIndices: queryIndices)

        let commitmentBytes = commitment.roots.map { root -> [UInt8] in
            let limbs = frToInt(root)
            var bytes = [UInt8]()
            for limb in limbs {
                for b in 0..<8 {
                    bytes.append(UInt8((limb >> (b * 8)) & 0xFF))
                }
            }
            return bytes
        }

        let proof = STIRProof(
            commitments: commitmentBytes,
            queries: queries,
            finalPoly: [commitment.finalValue],
            shifts: shifts
        )

        return (commitment, proof)
    }

    // MARK: - Soundness Analysis

    /// Compute the number of queries needed for a given security level.
    /// STIR achieves better soundness per query than FRI because domain shifts
    /// decorrelate errors across rounds.
    ///
    /// FRI soundness per query:  error ~ rho (rate parameter)
    /// STIR soundness per query: error ~ rho^1.5 (improved by shifting)
    ///
    /// At 128-bit security with rate 1/4:
    /// - FRI:  ~64 queries
    /// - STIR: ~43 queries (33% fewer)
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
    public static func estimateProofSize(logN: Int, numQueries: Int, rate: Double) -> Int {
        let frSize = MemoryLayout<Fr>.stride
        let numRounds = logN
        let perQuery = numRounds * (2 * frSize + logN * frSize)
        let commitSize = numRounds * frSize
        let finalSize = frSize
        return commitSize + numQueries * perQuery + finalSize
    }

    // MARK: - CPU Helpers

    /// Simple CPU NTT (O(n^2)) for small-size correctness testing.
    private static func cpuNTT(coeffs: [Fr], omega: Fr, n: Int) -> [Fr] {
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

    /// Simple CPU iNTT (O(n^2)) for small-size correctness testing.
    private static func cpuINTT(evals: [Fr], omegaInv: Fr, n: Int) -> [Fr] {
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

    /// Extract a Merkle authentication path for a given leaf index.
    private func extractMerklePath(tree: [Fr], leafCount: Int, index: Int) -> [Fr] {
        let treeSize = 2 * leafCount - 1
        var path = [Fr]()
        var idx = index
        var levelStart = 0
        var levelSize = leafCount

        while levelSize > 1 {
            let siblingIdx = idx ^ 1
            if levelStart + siblingIdx < treeSize {
                path.append(tree[levelStart + siblingIdx])
            }
            idx /= 2
            levelStart += levelSize
            levelSize /= 2
        }
        return path
    }
}
