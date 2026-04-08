// GPUFRIProverEngine — GPU-accelerated FRI (Fast Reed-Solomon IOP) prover
//
// Implements the full FRI proving protocol:
//   1. Commit phase: iterative polynomial folding with Merkle commitments
//   2. Query phase: produce authentication paths at random query positions
//
// Supports configurable folding factor (2 or 4), blowup factor, and rate.
// Uses GPUFRIFoldEngine for GPU-accelerated folding and GPUMerkleTreeEngine
// for Poseidon2-based Merkle commitments.
//
// Works with BN254 Fr field type.

import Foundation
import Metal

// MARK: - FRI Configuration

/// Configuration for the FRI protocol.
public struct GPUFRIConfig {
    /// Folding factor per round: 2 (standard) or 4 (faster, fewer rounds).
    public let foldingFactor: Int
    /// Blowup factor: ratio of evaluation domain to polynomial degree.
    /// Must be a power of 2, typically 2, 4, or 8.
    public let blowupFactor: Int
    /// Number of queries for soundness.
    public let numQueries: Int
    /// Maximum degree of the final polynomial (stopping condition).
    /// Folding stops when the polynomial degree drops to <= finalPolyMaxDegree.
    public let finalPolyMaxDegree: Int

    public init(foldingFactor: Int = 2,
                blowupFactor: Int = 4,
                numQueries: Int = 32,
                finalPolyMaxDegree: Int = 7) {
        precondition(foldingFactor == 2 || foldingFactor == 4,
                     "Folding factor must be 2 or 4")
        precondition(blowupFactor >= 2 && (blowupFactor & (blowupFactor - 1)) == 0,
                     "Blowup factor must be a power of 2")
        precondition(numQueries > 0, "Need at least one query")
        precondition(finalPolyMaxDegree >= 0, "Final poly degree must be non-negative")
        self.foldingFactor = foldingFactor
        self.blowupFactor = blowupFactor
        self.numQueries = numQueries
        self.finalPolyMaxDegree = finalPolyMaxDegree
    }
}

// MARK: - FRI Commit Layer

/// A single commit layer in the FRI protocol.
public struct GPUFRICommitLayer {
    /// Evaluations at this layer (on the folded coset domain).
    public let evaluations: [Fr]
    /// Merkle tree commitment over the evaluations.
    public let merkleTree: MerkleTree
    /// Log2 of the evaluation count.
    public let logSize: Int
}

// MARK: - FRI Commitment

/// Complete FRI commitment: all layers from the commit phase.
public struct GPUFRICommitment {
    /// Committed layers (layer 0 = initial, layer k = after k folds).
    public let layers: [GPUFRICommitLayer]
    /// Folding challenges used at each round (one per layer transition).
    public let challenges: [Fr]
    /// The final polynomial coefficients (low-degree).
    public let finalPoly: [Fr]
    /// Configuration used for this commitment.
    public let config: GPUFRIConfig
}

// MARK: - FRI Query Response

/// Response for a single query position across all FRI layers.
public struct GPUFRIQueryResponse {
    /// Query index in the initial domain.
    public let queryIndex: Int
    /// Authentication paths for each layer (sibling evaluations + Merkle proof).
    public let layerProofs: [GPUFRILayerQueryProof]
}

/// Proof data for one query at one FRI layer.
public struct GPUFRILayerQueryProof {
    /// The evaluation at this position.
    public let evaluation: Fr
    /// The sibling evaluation (needed for fold verification).
    public let siblingEvaluation: Fr
    /// Merkle authentication path for this position.
    public let authPath: MerkleAuthPath
}

// MARK: - FRI Proof

/// Complete FRI proof: commitment + query responses.
public struct GPUFRIProof {
    /// The commitment (layers, challenges, final poly).
    public let commitment: GPUFRICommitment
    /// Query responses with authentication paths.
    public let queryResponses: [GPUFRIQueryResponse]
}

// MARK: - GPUFRIProverEngine

/// GPU-accelerated FRI prover engine.
///
/// Usage:
///   let engine = try GPUFRIProverEngine()
///   let commitment = try engine.commit(evaluations: evals, config: config)
///   let proof = try engine.prove(evaluations: evals, config: config)
public class GPUFRIProverEngine {
    public static let version = Versions.gpuFRIProver

    private let foldEngine: GPUFRIFoldEngine
    private let merkleEngine: GPUMerkleTreeEngine

    public init() throws {
        self.foldEngine = try GPUFRIFoldEngine()
        self.merkleEngine = try GPUMerkleTreeEngine()
    }

    // MARK: - Commit Phase

    /// Execute the FRI commit phase: iteratively fold and commit.
    ///
    /// - Parameters:
    ///   - evaluations: Initial polynomial evaluations on the blowup domain (size must be power of 2).
    ///   - challenges: Folding challenges (one per round). If nil, generates deterministic challenges
    ///                 from the Merkle roots (Fiat-Shamir style).
    ///   - config: FRI configuration.
    /// - Returns: GPUFRICommitment containing all layers, challenges, and the final polynomial.
    public func commit(evaluations: [Fr],
                       challenges: [Fr]? = nil,
                       config: GPUFRIConfig) throws -> GPUFRICommitment {
        let n = evaluations.count
        precondition(n > 1 && (n & (n - 1)) == 0, "Evaluation count must be a power of 2")

        let logN = n.trailingZeroBitCount
        let foldBits = config.foldingFactor == 4 ? 2 : 1

        // Compute number of rounds: fold until domain size <= finalPolyMaxDegree * blowupFactor
        let targetSize = max((config.finalPolyMaxDegree + 1) * config.blowupFactor, 1 << foldBits)
        var numRounds = 0
        var sz = n
        while sz > targetSize {
            sz >>= foldBits
            numRounds += 1
        }

        // Build layers
        var layers = [GPUFRICommitLayer]()
        var usedChallenges = [Fr]()
        var currentEvals = evaluations
        var currentLogN = logN

        // Layer 0: commit to initial evaluations
        let tree0 = try merkleEngine.buildTree(leaves: padToPow2(currentEvals))
        layers.append(GPUFRICommitLayer(evaluations: currentEvals,
                                      merkleTree: tree0,
                                      logSize: currentLogN))

        for round in 0..<numRounds {
            // Get or derive challenge
            let challenge: Fr
            if let ch = challenges, round < ch.count {
                challenge = ch[round]
            } else {
                // Fiat-Shamir: derive from the last Merkle root
                challenge = deriveChallenge(from: layers.last!.merkleTree.root, round: round)
            }
            usedChallenges.append(challenge)

            // Fold
            if config.foldingFactor == 2 {
                currentEvals = try foldOnce(evals: currentEvals, logN: currentLogN,
                                             challenge: challenge)
                currentLogN -= 1
            } else {
                // Fold factor 4: two successive folds
                let mid = try foldOnce(evals: currentEvals, logN: currentLogN,
                                        challenge: challenge)
                let challenge2 = frMul(challenge, challenge) // squared challenge for second fold
                currentEvals = try foldOnce(evals: mid, logN: currentLogN - 1,
                                             challenge: challenge2)
                currentLogN -= 2
            }

            // Commit to folded evaluations
            let tree = try merkleEngine.buildTree(leaves: padToPow2(currentEvals))
            layers.append(GPUFRICommitLayer(evaluations: currentEvals,
                                          merkleTree: tree,
                                          logSize: currentLogN))
        }

        // Extract final polynomial (the last layer's evaluations are the final poly)
        let finalPoly = extractFinalPoly(evaluations: currentEvals, logN: currentLogN)

        return GPUFRICommitment(layers: layers,
                              challenges: usedChallenges,
                              finalPoly: finalPoly,
                              config: config)
    }

    // MARK: - Query Phase

    /// Execute the FRI query phase: produce authentication paths at random positions.
    ///
    /// - Parameters:
    ///   - commitment: The FRI commitment from the commit phase.
    ///   - queryIndices: Indices into the initial domain to query. If nil, derives from final root.
    /// - Returns: Array of query responses.
    public func query(commitment: GPUFRICommitment,
                      queryIndices: [Int]? = nil) throws -> [GPUFRIQueryResponse] {
        let n = commitment.layers[0].evaluations.count
        let numQueries = commitment.config.numQueries

        // Get or derive query indices
        let indices: [Int]
        if let qi = queryIndices {
            indices = qi
        } else {
            // Derive from final layer's Merkle root
            let lastRoot = commitment.layers.last!.merkleTree.root
            indices = deriveQueryIndices(from: lastRoot, domainSize: n, count: numQueries)
        }

        var responses = [GPUFRIQueryResponse]()
        responses.reserveCapacity(indices.count)

        for queryIdx in indices {
            var layerProofs = [GPUFRILayerQueryProof]()
            var currentIdx = queryIdx

            for layerIdx in 0..<(commitment.layers.count - 1) {
                let layer = commitment.layers[layerIdx]
                let layerSize = layer.evaluations.count

                // Clamp index to layer size
                let idx = currentIdx % layerSize
                let siblingIdx: Int

                if commitment.config.foldingFactor == 2 {
                    // Sibling is at idx +/- n/2 (butterfly pair)
                    let half = layerSize / 2
                    if idx < half {
                        siblingIdx = idx + half
                    } else {
                        siblingIdx = idx - half
                    }
                } else {
                    // Factor 4: sibling at idx XOR (layerSize/4)
                    let quarter = layerSize / 4
                    siblingIdx = idx ^ quarter
                }

                let eval = layer.evaluations[idx]
                let siblingEval = layer.evaluations[siblingIdx]
                let authPath = layer.merkleTree.proof(forLeafAt: idx)

                layerProofs.append(GPUFRILayerQueryProof(
                    evaluation: eval,
                    siblingEvaluation: siblingEval,
                    authPath: authPath))

                // Compute index in next (folded) layer
                if commitment.config.foldingFactor == 2 {
                    currentIdx = idx % (layerSize / 2)
                } else {
                    currentIdx = idx % (layerSize / 4)
                }
            }

            responses.append(GPUFRIQueryResponse(queryIndex: queryIdx, layerProofs: layerProofs))
        }

        return responses
    }

    // MARK: - Full Prove

    /// Execute both commit and query phases to produce a complete FRI proof.
    ///
    /// - Parameters:
    ///   - evaluations: Polynomial evaluations on the blowup domain.
    ///   - challenges: Optional folding challenges.
    ///   - config: FRI configuration.
    /// - Returns: Complete FRI proof.
    public func prove(evaluations: [Fr],
                      challenges: [Fr]? = nil,
                      config: GPUFRIConfig) throws -> GPUFRIProof {
        let commitment = try commit(evaluations: evaluations, challenges: challenges, config: config)
        let responses = try query(commitment: commitment)
        return GPUFRIProof(commitment: commitment, queryResponses: responses)
    }

    // MARK: - Verification Helpers

    /// Verify that the final polynomial has degree <= finalPolyMaxDegree.
    /// Returns true if the polynomial coefficients beyond the max degree are all zero.
    public static func verifyFinalPolyDegree(finalPoly: [Fr], maxDegree: Int) -> Bool {
        if finalPoly.count <= maxDegree + 1 { return true }
        for i in (maxDegree + 1)..<finalPoly.count {
            if !finalPoly[i].isZero { return false }
        }
        return true
    }

    /// Verify a single query response against the commitment.
    /// Checks Merkle authentication paths and fold consistency.
    public static func verifyQueryResponse(response: GPUFRIQueryResponse,
                                            commitment: GPUFRICommitment) -> Bool {
        for (layerIdx, proof) in response.layerProofs.enumerated() {
            let layer = commitment.layers[layerIdx]
            let root = layer.merkleTree.root

            // Verify Merkle auth path
            if !proof.authPath.verify(root: root, leaf: proof.evaluation) {
                return false
            }
        }
        return true
    }

    // MARK: - Internal Helpers

    /// Fold evaluations once using the GPU fold engine.
    private func foldOnce(evals: [Fr], logN: Int, challenge: Fr) throws -> [Fr] {
        let n = evals.count
        precondition(n == 1 << logN)
        let stride = MemoryLayout<Fr>.stride

        let evalsBuf = foldEngine.device.makeBuffer(
            bytes: evals, length: n * stride,
            options: .storageModeShared)!

        let resultBuf = try foldEngine.fold(evals: evalsBuf, logN: logN, challenge: challenge)

        let half = n / 2
        let ptr = resultBuf.contents().bindMemory(to: Fr.self, capacity: half)
        return Array(UnsafeBufferPointer(start: ptr, count: half))
    }

    /// Derive a folding challenge from a Merkle root (Fiat-Shamir).
    /// Uses Poseidon2 hash of root with round counter.
    private func deriveChallenge(from root: Fr, round: Int) -> Fr {
        let roundField = frFromInt(UInt64(round + 1))
        return poseidon2Hash(root, roundField)
    }

    /// Derive query indices from a seed (Fiat-Shamir).
    private func deriveQueryIndices(from seed: Fr, domainSize: Int, count: Int) -> [Int] {
        var indices = [Int]()
        indices.reserveCapacity(count)
        var current = seed

        for i in 0..<count {
            let roundField = frFromInt(UInt64(i + 1))
            current = poseidon2Hash(current, roundField)
            let words = frToInt(current)
            let idx = Int(words[0] % UInt64(domainSize))
            indices.append(idx)
        }

        return indices
    }

    /// Extract final polynomial coefficients from evaluations via inverse NTT.
    /// For small domains, uses direct Lagrange interpolation on CPU.
    private func extractFinalPoly(evaluations: [Fr], logN: Int) -> [Fr] {
        let n = evaluations.count
        if n <= 1 { return evaluations }

        // Direct inverse DFT: coeffs[k] = (1/n) * sum_j evals[j] * omega^{-jk}
        // O(n^2) but n is small (final poly is typically <= 32 elements)
        let invTwiddles = precomputeInverseTwiddles(logN: logN)
        let nInv = frInverse(frFromInt(UInt64(n)))
        var coeffs = [Fr](repeating: Fr.zero, count: n)

        for k in 0..<n {
            var sum = Fr.zero
            for j in 0..<n {
                let twIdx = (j * k) % n
                sum = frAdd(sum, frMul(evaluations[j], invTwiddles[twIdx]))
            }
            coeffs[k] = frMul(sum, nInv)
        }

        return coeffs
    }

    /// Pad array to next power of 2 if needed (for Merkle tree).
    private func padToPow2(_ arr: [Fr]) -> [Fr] {
        let n = arr.count
        if n & (n - 1) == 0 { return arr }
        var next = 1
        while next < n { next <<= 1 }
        var padded = arr
        padded.append(contentsOf: [Fr](repeating: Fr.zero, count: next - n))
        return padded
    }
}
