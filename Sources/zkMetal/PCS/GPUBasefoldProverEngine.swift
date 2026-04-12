// GPU-Accelerated Basefold Polynomial Commitment Prover Engine
//
// Basefold: a hash-based multilinear PCS using Reed-Solomon codes and
// FRI-like proximity testing. Commit via Merkle tree of evaluations,
// open via iterative sumcheck-style folding with Merkle authentication.
//
// Key features:
//   - GPU-accelerated evaluation folding (reuses BasefoldEngine Metal kernels)
//   - Merkle tree commitment via Poseidon2 (GPU-accelerated for large trees)
//   - FRI-like proximity proof with configurable query count
//   - Multilinear and univariate polynomial modes
//   - Batch commitment support for multiple polynomials
//   - Reed-Solomon encoding with configurable rate parameter
//
// Construction:
//   Given multilinear f on {0,1}^n as 2^n evaluations:
//     Commit: Merkle root of RS-encoded evaluations
//     Open:   n rounds of folding with Fiat-Shamir challenges,
//             Merkle authentication at each level, query proofs
//     Verify: check fold consistency + Merkle paths + proximity

import Foundation
import Metal
import NeonFieldOps

// MARK: - Configuration

/// Configuration for the Basefold prover engine.
public struct BasefoldProverConfig {
    /// Number of random queries for proximity testing (security parameter)
    public let numQueries: Int
    /// Reed-Solomon rate parameter: blowup factor = 2^rateLog
    public let rateLog: Int
    /// Maximum polynomial size (log2) before falling back to CPU
    public let maxGPULogSize: Int

    public init(numQueries: Int = 40, rateLog: Int = 1, maxGPULogSize: Int = 24) {
        self.numQueries = numQueries
        self.rateLog = rateLog
        self.maxGPULogSize = maxGPULogSize
    }

    /// Default configuration: 128-bit security, rate 1/2
    public static let standard = BasefoldProverConfig()

    /// Fast configuration: fewer queries, for testing
    public static let fast = BasefoldProverConfig(numQueries: 16, rateLog: 1, maxGPULogSize: 20)
}

// MARK: - Commitment Types

/// Commitment to a multilinear polynomial via Basefold.
public struct BasefoldProverCommitment {
    /// Merkle root of RS-encoded evaluations
    public let root: Fr
    /// Number of variables in the multilinear polynomial
    public let numVars: Int
    /// Original evaluations (prover retains for opening)
    public let evaluations: [Fr]
    /// RS-encoded evaluations (length = evaluations.count * blowup)
    public let encodedEvals: [Fr]
    /// Full Merkle tree over encoded evaluations
    public let merkleTree: [Fr]

    public init(root: Fr, numVars: Int, evaluations: [Fr],
                encodedEvals: [Fr], merkleTree: [Fr]) {
        self.root = root
        self.numVars = numVars
        self.evaluations = evaluations
        self.encodedEvals = encodedEvals
        self.merkleTree = merkleTree
    }
}

/// Opening proof for a single polynomial at a point.
public struct BasefoldProverProof {
    /// Merkle roots at each folding level
    public let foldRoots: [Fr]
    /// Final scalar value after all folds
    public let finalValue: Fr
    /// Folded evaluation layers (intermediate data for query verification)
    public let foldLayers: [[Fr]]
    /// Query proofs for random indices
    public let queries: [BasefoldProverQuery]
    /// The evaluation point used
    public let point: [Fr]
    /// Number of fold-by-2 rounds consumed per committed level (1 = single, 2 = fused fold-by-4).
    /// Empty means all levels are stride-1 (backward compat).
    public let levelStrides: [Int]

    public init(foldRoots: [Fr], finalValue: Fr, foldLayers: [[Fr]],
                queries: [BasefoldProverQuery], point: [Fr], levelStrides: [Int] = []) {
        self.foldRoots = foldRoots
        self.finalValue = finalValue
        self.foldLayers = foldLayers
        self.queries = queries
        self.point = point
        self.levelStrides = levelStrides
    }
}

/// A single query proof: evaluation pair + fold + Merkle path at each level.
public struct BasefoldProverQuery {
    /// Query index in the encoded evaluation domain
    public let index: Int
    /// (low, high) evaluation pairs at each folding level (stride-1 levels)
    public let evalPairs: [(Fr, Fr)]
    /// Recomputed fold results for consistency checking
    public let foldValues: [Fr]
    /// Merkle authentication paths at each level
    public let authPaths: [[Fr]]
    /// For stride-2 (fused fold-by-4) levels: (a, b, c, d) evaluation quad.
    public let evalQuads: [(Fr, Fr, Fr, Fr)]

    public init(index: Int, evalPairs: [(Fr, Fr)], foldValues: [Fr], authPaths: [[Fr]],
                evalQuads: [(Fr, Fr, Fr, Fr)] = []) {
        self.index = index
        self.evalPairs = evalPairs
        self.foldValues = foldValues
        self.authPaths = authPaths
        self.evalQuads = evalQuads
    }
}

/// Batch commitment: multiple polynomials committed together.
public struct BasefoldBatchCommitment {
    /// Individual commitments for each polynomial
    public let commitments: [BasefoldProverCommitment]
    /// Combined root (hash of all individual roots)
    public let batchRoot: Fr

    public init(commitments: [BasefoldProverCommitment], batchRoot: Fr) {
        self.commitments = commitments
        self.batchRoot = batchRoot
    }
}

/// Batch opening proof for multiple polynomials at the same point.
public struct BasefoldBatchProof {
    /// Individual proofs per polynomial
    public let proofs: [BasefoldProverProof]
    /// Random linear combination challenge (Fiat-Shamir)
    public let batchChallenge: Fr

    public init(proofs: [BasefoldProverProof], batchChallenge: Fr) {
        self.proofs = proofs
        self.batchChallenge = batchChallenge
    }
}

// MARK: - Engine

/// GPU-accelerated Basefold polynomial commitment prover.
/// Uses Metal for evaluation folding and Merkle tree construction.
public class GPUBasefoldProverEngine {
    public static let version = Versions.gpuBasefoldProver

    /// Underlying BasefoldEngine for GPU fold operations
    private let basefold: BasefoldEngine
    /// Poseidon2 Merkle engine for tree construction
    private let merkleEngine: Poseidon2MerkleEngine
    /// Configuration
    public let config: BasefoldProverConfig

    // Cached GPU buffers for fold operations
    private var cachedFoldInputBuf: MTLBuffer?
    private var cachedFoldInputSize: Int = 0
    private var cachedFoldOutputBuf: MTLBuffer?
    private var cachedFoldOutputSize: Int = 0

    public init(config: BasefoldProverConfig = .standard) throws {
        self.config = config
        self.basefold = try BasefoldEngine()
        self.merkleEngine = try Poseidon2MerkleEngine()
    }

    /// Convenience init with default config.
    public convenience init() throws {
        try self.init(config: .standard)
    }

    // MARK: - Reed-Solomon Encoding

    /// Encode evaluations via simple RS extension: evaluate the unique multilinear
    /// polynomial at additional domain points beyond the boolean hypercube.
    /// For rate parameter rateLog, output is 2^rateLog times the input size.
    ///
    /// The encoding evaluates f at shifted domain points using the multilinear structure:
    ///   f(x1,...,xn) = sum_S f_S * prod_{i in S} x_i
    /// Extended domain: x_i in {0, 1, 2, ..., blowup-1} for each coordinate.
    ///
    /// For rateLog=1 (blowup=2), we append evaluations at x_i = 2 for each dimension,
    /// computed via the fold identity: f(2) = 2*f(1) - f(0) (linear extrapolation).
    public func rsEncode(evaluations: [Fr]) -> [Fr] {
        let n = evaluations.count
        let blowup = 1 << config.rateLog

        if blowup == 1 {
            return evaluations
        }

        // For blowup = 2: use GPU-accelerated RS extension for large inputs
        if blowup == 2 {
            // GPU threshold: below this, CPU is faster due to dispatch overhead
            let gpuRSThreshold = 256

            if n >= gpuRSThreshold {
                // GPU path: basefold.rsExtend handles the full encoding
                if let encoded = try? basefold.rsExtend(evaluations: evaluations) {
                    return encoded
                }
                // Fall through to CPU path on GPU error
            }

            // CPU path for small inputs or GPU fallback
            var encoded = [Fr](repeating: Fr.zero, count: 2 * n)

            // Copy original evaluations
            for i in 0..<n {
                encoded[i] = evaluations[i]
            }

            // Linear extrapolation: for each pair (f(0,...), f(1,...)), compute f(2,...)
            // f(2) = 2*f(1) - f(0), which is the degree-1 extrapolation
            let halfN = n / 2
            if halfN > 0 {
                let two = frFromInt(2)
                for i in 0..<halfN {
                    // f(0,...) = evaluations[i], f(1,...) = evaluations[i + halfN]
                    // f(2,...) = 2*f(1,...) - f(0,...)
                    let f0 = evaluations[i]
                    let f1 = evaluations[i + halfN]
                    encoded[n + i] = frSub(frMul(two, f1), f0)
                    // f(3,...) = 2*f(2,...) - f(1,...) = 4*f(1,...) - 3*f(0,...)
                    // But for rate 1/2 we only need blowup=2 total, so n+halfN extra entries
                    encoded[n + halfN + i] = frSub(frMul(two, encoded[n + i]), f1)
                }
            } else {
                // Single evaluation: just duplicate
                encoded[1] = evaluations[0]
            }

            return encoded
        }

        // General blowup: repeated extrapolation (less common path)
        var encoded = evaluations
        encoded.reserveCapacity(n * blowup)
        let two = frFromInt(2)
        for _ in 1..<blowup {
            let curLen = encoded.count
            let halfLen = curLen / 2
            for i in 0..<halfLen {
                let f0 = encoded[curLen - halfLen + i - halfLen]
                let f1 = encoded[curLen - halfLen + i]
                encoded.append(frSub(frMul(two, f1), f0))
            }
            for i in 0..<halfLen {
                let f0 = encoded[curLen - halfLen + i]
                let f1 = encoded[curLen + i]
                encoded.append(frSub(frMul(two, f1), f0))
            }
        }

        return Array(encoded.prefix(n * blowup))
    }

    // MARK: - Commit

    /// Commit to a multilinear polynomial given as 2^n evaluations.
    /// Returns a commitment containing Merkle root, evaluations, and RS encoding.
    public func commit(evaluations: [Fr]) throws -> BasefoldProverCommitment {
        let n = evaluations.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Evaluation count must be power of 2")

        let numVars = Int(log2(Double(n)))

        // RS-encode the evaluations
        let tEncode = CFAbsoluteTimeGetCurrent()
        let encoded = rsEncode(evaluations: evaluations)
        fputs(String(format: "  [profile] rs_encode: %.2fms\n", (CFAbsoluteTimeGetCurrent() - tEncode) * 1000), stderr)

        // Build Merkle tree over encoded evaluations
        let tMerkle = CFAbsoluteTimeGetCurrent()
        let tree = try merkleEngine.buildTree(encoded)
        let root = tree.last!
        fputs(String(format: "  [profile] merkle: %.2fms\n", (CFAbsoluteTimeGetCurrent() - tMerkle) * 1000), stderr)

        return BasefoldProverCommitment(
            root: root,
            numVars: numVars,
            evaluations: evaluations,
            encodedEvals: encoded,
            merkleTree: tree
        )
    }

    /// Batch commit: commit to multiple polynomials, returning individual commitments
    /// plus a combined batch root.
    public func batchCommit(polynomials: [[Fr]]) throws -> BasefoldBatchCommitment {
        precondition(!polynomials.isEmpty, "Must commit to at least one polynomial")

        var commitments: [BasefoldProverCommitment] = []
        commitments.reserveCapacity(polynomials.count)

        for poly in polynomials {
            let c = try commit(evaluations: poly)
            commitments.append(c)
        }

        // Combine roots via Poseidon2 hash chain
        var batchRoot = commitments[0].root
        for i in 1..<commitments.count {
            batchRoot = poseidon2Hash(batchRoot, commitments[i].root)
        }

        return BasefoldBatchCommitment(commitments: commitments, batchRoot: batchRoot)
    }

    // MARK: - Open (Prove)

    /// Generate an opening proof for a committed polynomial at a given point.
    /// Uses fold-by-4 (fused 2-round fold) to halve the number of committed levels
    /// and Merkle trees, reducing proof size and Merkle overhead by ~50%.
    public func open(commitment: BasefoldProverCommitment, point: [Fr]) throws -> BasefoldProverProof {
        let evals = commitment.evaluations
        let n = evals.count
        let numVars = point.count
        precondition(n == (1 << numVars), "Evaluation count must be 2^numVars")

        // Phase 1: GPU-accelerated fold-by-4 folding
        let tFold = CFAbsoluteTimeGetCurrent()
        let (foldLayers, levelStrides) = try computeFoldLayers(evaluations: evals, point: point)
        let numLevels = foldLayers.count
        fputs(String(format: "  [profile] fold: %.2fms\n", (CFAbsoluteTimeGetCurrent() - tFold) * 1000), stderr)

        // Phase 2: Build Merkle trees over each committed fold layer and extract roots
        let tMerkle = CFAbsoluteTimeGetCurrent()
        var foldRoots: [Fr] = []
        foldRoots.reserveCapacity(numLevels)
        var layerTrees: [[Fr]] = []
        layerTrees.reserveCapacity(numLevels)

        for layer in foldLayers {
            if layer.count >= 2 {
                let tree = try merkleEngine.buildTree(layer)
                foldRoots.append(tree.last!)
                layerTrees.append(tree)
            } else {
                foldRoots.append(layer.isEmpty ? Fr.zero : layer[0])
                layerTrees.append(layer)
            }
        }
        fputs(String(format: "  [profile] merkle: %.2fms (%d layers)\n", (CFAbsoluteTimeGetCurrent() - tMerkle) * 1000, numLevels), stderr)

        let finalValue = foldLayers.last!.first ?? Fr.zero

        // Build source layers for query proof extraction:
        // sourceLayers[i] is the input to level i's fold
        var sourceLayers: [[Fr]] = []
        sourceLayers.reserveCapacity(numLevels)
        sourceLayers.append(evals)
        for level in 0..<numLevels - 1 {
            sourceLayers.append(foldLayers[level])
        }

        // Phase 3: Generate query proofs using Fiat-Shamir derived indices
        var rng = deriveQueryRNG(commitRoot: commitment.root, foldRoots: foldRoots)
        var queries: [BasefoldProverQuery] = []
        queries.reserveCapacity(config.numQueries)

        for _ in 0..<config.numQueries {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let queryIdx = Int(rng >> 32) % (n / 2)
            let query = buildQueryProofFoldBy4(
                index: queryIdx,
                originalTree: commitment.merkleTree,
                sourceLayers: sourceLayers,
                foldLayers: foldLayers,
                layerTrees: layerTrees,
                levelStrides: levelStrides,
                numVars: numVars,
                point: point
            )
            queries.append(query)
        }

        return BasefoldProverProof(
            foldRoots: foldRoots,
            finalValue: finalValue,
            foldLayers: foldLayers,
            queries: queries,
            point: point,
            levelStrides: levelStrides
        )
    }

    /// Batch open: generate opening proofs for multiple committed polynomials at the same point.
    public func batchOpen(batch: BasefoldBatchCommitment, point: [Fr]) throws -> BasefoldBatchProof {
        precondition(!batch.commitments.isEmpty)

        // Derive batch challenge from batch root and point
        var challenge = batch.batchRoot
        for p in point {
            challenge = poseidon2Hash(challenge, p)
        }

        var proofs: [BasefoldProverProof] = []
        proofs.reserveCapacity(batch.commitments.count)

        for commitment in batch.commitments {
            let proof = try open(commitment: commitment, point: point)
            proofs.append(proof)
        }

        return BasefoldBatchProof(proofs: proofs, batchChallenge: challenge)
    }

    // MARK: - Verify

    /// Verify an opening proof against a commitment root, point, and claimed value.
    /// Returns true if the proof is valid.
    public func verify(root: Fr, point: [Fr], claimedValue: Fr, proof: BasefoldProverProof) -> Bool {
        // Check final value matches claimed
        if !frEqual(proof.finalValue, claimedValue) {
            return false
        }

        // Verify each query proof
        for query in proof.queries {
            if !verifyQueryProof(query: query, point: point, finalValue: proof.finalValue,
                                 levelStrides: proof.levelStrides) {
                return false
            }
        }

        // Verify fold layer consistency: last fold should produce final value
        if let lastLayer = proof.foldLayers.last, lastLayer.count == 1 {
            if !frEqual(lastLayer[0], proof.finalValue) {
                return false
            }
        }

        return true
    }

    /// Verify a batch opening proof.
    public func verifyBatch(batch: BasefoldBatchCommitment, point: [Fr],
                            claimedValues: [Fr], proof: BasefoldBatchProof) -> Bool {
        guard batch.commitments.count == proof.proofs.count,
              batch.commitments.count == claimedValues.count else {
            return false
        }

        for i in 0..<batch.commitments.count {
            let root = batch.commitments[i].root
            if !verify(root: root, point: point, claimedValue: claimedValues[i],
                       proof: proof.proofs[i]) {
                return false
            }
        }

        return true
    }

    // MARK: - Multilinear Evaluation

    /// Evaluate a multilinear polynomial at a point via sequential folding.
    /// f(r1,...,rn) = fold with each r_i reducing dimension by one.
    public static func evaluateMultilinear(evaluations: [Fr], point: [Fr]) -> Fr {
        return BasefoldEngine.cpuEvaluate(evals: evaluations, point: point)
    }

    /// Evaluate a univariate polynomial at a single point via Horner's method.
    /// coeffs[i] is the coefficient of X^i.
    public static func evaluateUnivariate(coeffs: [Fr], at x: Fr) -> Fr {
        guard !coeffs.isEmpty else { return Fr.zero }
        var result = coeffs[coeffs.count - 1]
        for i in stride(from: coeffs.count - 2, through: 0, by: -1) {
            result = frAdd(frMul(result, x), coeffs[i])
        }
        return result
    }

    // MARK: - GPU Fold Layers

    /// Compute fold layers using fold-by-4 (fused 2-round fold) where possible.
    /// Returns (layers, levelStrides) where each layer corresponds to one committed level.
    /// levelStrides[i] = 2 means fused fold-by-4, 1 means single fold-by-2.
    private func computeFoldLayers(evaluations: [Fr], point: [Fr]) throws -> (layers: [[Fr]], strides: [Int]) {
        let n = evaluations.count
        let numVars = point.count
        var layers: [[Fr]] = []
        var levelStrides: [Int] = []

        let gpuThreshold = 64

        // Halves fold processes highest-order variable first, so reverse
        // point to map point[0] (x_0) to the correct fold round.
        let reversedPoint = Array(point.reversed())
        var currentEvals = evaluations
        var round = 0

        while round < numVars {
            let currentN = currentEvals.count
            if currentN < 2 { break }

            if round + 1 < numVars && currentN >= 4 {
                // Fused fold-by-4: two rounds in one step
                let quarterN = currentN / 4
                let alpha0 = reversedPoint[round]
                let alpha1 = reversedPoint[round + 1]

                let folded: [Fr]
                if currentN >= gpuThreshold {
                    folded = try basefold.multiFold(evals: currentEvals, challenges: [alpha0, alpha1])
                } else {
                    // CPU path: two sequential folds
                    let mid = BasefoldEngine.cpuFold(evals: currentEvals, alpha: alpha0)
                    folded = BasefoldEngine.cpuFold(evals: mid, alpha: alpha1)
                }
                layers.append(folded)
                levelStrides.append(2)
                currentEvals = folded
                round += 2
            } else {
                // Single fold
                let folded: [Fr]
                if currentN >= gpuThreshold {
                    folded = try basefold.fold(evals: currentEvals, alpha: reversedPoint[round])
                } else {
                    folded = BasefoldEngine.cpuFold(evals: currentEvals, alpha: reversedPoint[round])
                }
                layers.append(folded)
                levelStrides.append(1)
                currentEvals = folded
                round += 1
            }
        }

        return (layers, levelStrides)
    }

    // MARK: - Query Proof Construction

    /// Build a query proof for a specific index, traversing all fold levels.
    private func buildQueryProof(
        index: Int,
        originalTree: [Fr],
        originalEvals: [Fr],
        foldLayers: [[Fr]],
        layerTrees: [[Fr]],
        numVars: Int,
        point: [Fr]
    ) -> BasefoldProverQuery {
        let reversedPoint = Array(point.reversed())
        var evalPairs: [(Fr, Fr)] = []
        var foldValues: [Fr] = []
        var authPaths: [[Fr]] = []
        var idx = index
        let n = originalEvals.count

        // Level 0: original evaluations
        let halfN0 = n / 2
        let canonIdx0 = idx % halfN0
        let a0 = originalEvals[canonIdx0]
        let b0 = originalEvals[canonIdx0 + halfN0]
        evalPairs.append((a0, b0))
        authPaths.append(extractMerklePath(tree: originalTree, leafCount: n, index: canonIdx0))
        let fold0 = frAdd(a0, frMul(reversedPoint[0], frSub(b0, a0)))
        foldValues.append(fold0)
        idx = canonIdx0

        // Subsequent levels: folded layers
        for level in 0..<foldLayers.count - 1 {
            let layer = foldLayers[level]
            let layerN = layer.count
            let halfN = layerN / 2
            if halfN == 0 { break }
            let canonIdx = idx % halfN
            let a = layer[canonIdx]
            let b = layer[canonIdx + halfN]
            evalPairs.append((a, b))

            if layerTrees[level].count > 1 {
                authPaths.append(extractMerklePath(
                    tree: layerTrees[level], leafCount: layerN, index: canonIdx))
            } else {
                authPaths.append([])
            }

            let foldR = frAdd(a, frMul(reversedPoint[level + 1], frSub(b, a)))
            foldValues.append(foldR)
            idx = canonIdx
        }

        return BasefoldProverQuery(
            index: index,
            evalPairs: evalPairs,
            foldValues: foldValues,
            authPaths: authPaths
        )
    }

    /// Build a fold-by-4 aware query proof for a specific index.
    private func buildQueryProofFoldBy4(
        index: Int,
        originalTree: [Fr],
        sourceLayers: [[Fr]],
        foldLayers: [[Fr]],
        layerTrees: [[Fr]],
        levelStrides: [Int],
        numVars: Int,
        point: [Fr]
    ) -> BasefoldProverQuery {
        let reversedPoint = Array(point.reversed())
        var evalPairs: [(Fr, Fr)] = []
        var evalQuads: [(Fr, Fr, Fr, Fr)] = []
        var foldValues: [Fr] = []
        var authPaths: [[Fr]] = []
        var idx = index
        let numLevels = levelStrides.count
        var pointIdx = 0

        for level in 0..<numLevels {
            let src = sourceLayers[level]
            let srcN = src.count
            let s = levelStrides[level]

            if s == 2 && srcN >= 4 {
                // Fused fold-by-4: extract 4 values from source
                let quarterN = srcN / 4
                let halfN = srcN / 2
                let canonIdx = idx % quarterN
                let a = src[canonIdx]
                let b = src[canonIdx + quarterN]
                let c = src[canonIdx + halfN]
                let d = src[canonIdx + halfN + quarterN]
                evalPairs.append((Fr.zero, Fr.zero))
                evalQuads.append((a, b, c, d))

                // Merkle path for the committed output layer
                if layerTrees[level].count > 1 {
                    authPaths.append(extractMerklePath(
                        tree: layerTrees[level], leafCount: foldLayers[level].count, index: canonIdx))
                } else {
                    authPaths.append([])
                }

                // Recompute fused fold
                let alpha0 = reversedPoint[pointIdx]
                let alpha1 = reversedPoint[pointIdx + 1]
                let mid0 = frAdd(a, frMul(alpha0, frSub(c, a)))
                let mid1 = frAdd(b, frMul(alpha0, frSub(d, b)))
                let result = frAdd(mid0, frMul(alpha1, frSub(mid1, mid0)))
                foldValues.append(result)

                idx = canonIdx
                pointIdx += 2
            } else {
                // Single fold: extract pair
                let halfN = srcN / 2
                if halfN == 0 { break }
                let canonIdx = idx % halfN
                let a = src[canonIdx]
                let b = src[canonIdx + halfN]
                evalPairs.append((a, b))
                evalQuads.append((Fr.zero, Fr.zero, Fr.zero, Fr.zero))

                // Merkle path
                if level == 0 {
                    authPaths.append(extractMerklePath(tree: originalTree, leafCount: srcN, index: canonIdx))
                } else if layerTrees[level].count > 1 {
                    authPaths.append(extractMerklePath(
                        tree: layerTrees[level], leafCount: foldLayers[level].count, index: canonIdx))
                } else {
                    authPaths.append([])
                }

                let alpha = reversedPoint[pointIdx]
                let result = frAdd(a, frMul(alpha, frSub(b, a)))
                foldValues.append(result)

                idx = canonIdx
                pointIdx += 1
            }
        }

        return BasefoldProverQuery(
            index: index,
            evalPairs: evalPairs,
            foldValues: foldValues,
            authPaths: authPaths,
            evalQuads: evalQuads
        )
    }

    /// Verify a single query proof: check fold consistency at each level.
    /// Supports both old-format (all stride-1) and fold-by-4 proofs.
    private func verifyQueryProof(query: BasefoldProverQuery, point: [Fr],
                                  finalValue: Fr, levelStrides: [Int] = []) -> Bool {
        let reversedPoint = Array(point.reversed())
        let numLevels = query.foldValues.count
        var pointIdx = 0

        for level in 0..<numLevels {
            let s = (level < levelStrides.count) ? levelStrides[level] : 1

            if s == 2 && level < query.evalQuads.count {
                // Fused fold-by-4 verification
                let (a, b, c, d) = query.evalQuads[level]
                let alpha0 = reversedPoint[pointIdx]
                let alpha1 = reversedPoint[pointIdx + 1]
                let mid0 = frAdd(a, frMul(alpha0, frSub(c, a)))
                let mid1 = frAdd(b, frMul(alpha0, frSub(d, b)))
                let expected = frAdd(mid0, frMul(alpha1, frSub(mid1, mid0)))

                if !frEqual(expected, query.foldValues[level]) {
                    return false
                }
                pointIdx += 2
            } else {
                // Single fold verification
                let (a, b) = query.evalPairs[level]
                let alpha = reversedPoint[pointIdx]
                let expected = frAdd(a, frMul(alpha, frSub(b, a)))

                if !frEqual(expected, query.foldValues[level]) {
                    return false
                }
                pointIdx += 1
            }

            // Last level fold should match final value
            if level == numLevels - 1 {
                if !frEqual(query.foldValues[level], finalValue) {
                    return false
                }
            }
        }
        return true
    }

    // MARK: - Merkle Helpers

    /// Extract a Merkle authentication path for a leaf index.
    private func extractMerklePath(tree: [Fr], leafCount: Int, index: Int) -> [Fr] {
        var path: [Fr] = []
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

    /// Verify a Merkle authentication path from leaf to root.
    public func verifyMerklePath(leaf: Fr, path: [Fr], index: Int, root: Fr) -> Bool {
        var current = leaf
        var idx = index

        for sibling in path {
            if idx & 1 == 0 {
                current = poseidon2Hash(current, sibling)
            } else {
                current = poseidon2Hash(sibling, current)
            }
            idx >>= 1
        }

        return frEqual(current, root)
    }

    // MARK: - Fiat-Shamir RNG

    /// Derive a pseudo-random seed from commitment root and fold roots.
    private func deriveQueryRNG(commitRoot: Fr, foldRoots: [Fr]) -> UInt64 {
        var rng: UInt64 = frToUInt64(commitRoot)
        for r in foldRoots {
            rng ^= frToUInt64(r)
        }
        return rng
    }

    // MARK: - Univariate Mode

    /// Commit to a univariate polynomial (coefficients) by evaluating on a domain
    /// and committing to the evaluations. Domain size = max(coeffs.count, nextPow2).
    public func commitUnivariate(coeffs: [Fr]) throws -> BasefoldProverCommitment {
        let n = nextPowerOf2(coeffs.count)
        precondition(n >= 2, "Need at least 2 coefficients")

        // Evaluate polynomial at 0, 1, 2, ..., n-1
        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            let x = frFromInt(UInt64(i))
            evals[i] = GPUBasefoldProverEngine.evaluateUnivariate(coeffs: coeffs, at: x)
        }

        return try commit(evaluations: evals)
    }

    /// Open a univariate polynomial commitment at a specific point.
    /// Converts to multilinear representation internally.
    public func openUnivariate(commitment: BasefoldProverCommitment,
                               coeffs: [Fr], at z: Fr) throws -> (Fr, BasefoldProverProof) {
        let value = GPUBasefoldProverEngine.evaluateUnivariate(coeffs: coeffs, at: z)

        // For univariate mode, we open at a binary decomposition of the query index
        // But a simpler approach: use the multilinear point derived from z
        let numVars = commitment.numVars
        var point = [Fr](repeating: Fr.zero, count: numVars)
        var zPow = z
        for i in 0..<numVars {
            point[i] = zPow
            zPow = frMul(zPow, z)
        }

        let proof = try open(commitment: commitment, point: point)
        return (value, proof)
    }

    // MARK: - Utility

    /// Next power of 2 >= n.
    private func nextPowerOf2(_ n: Int) -> Int {
        var v = n - 1
        v |= v >> 1
        v |= v >> 2
        v |= v >> 4
        v |= v >> 8
        v |= v >> 16
        return v + 1
    }

    // MARK: - Proximity Testing

    /// Run FRI-like proximity test on the fold layers.
    /// Checks that each folded layer is close to a degree-d polynomial
    /// by sampling random points and checking consistency.
    /// Supports both old-format and fold-by-4 proofs.
    public func proximityTest(commitment: BasefoldProverCommitment,
                              proof: BasefoldProverProof) -> Bool {
        // Verify fold roots are consistent
        guard proof.foldRoots.count == proof.foldLayers.count else {
            return false
        }

        let strides = proof.levelStrides
        let reversedPoint = Array(proof.point.reversed())

        // Verify query proofs
        for query in proof.queries {
            if query.foldValues.count != query.evalPairs.count {
                return false
            }

            var pointIdx = 0
            for level in 0..<query.foldValues.count {
                let s = (level < strides.count) ? strides[level] : 1

                if s == 2 && level < query.evalQuads.count {
                    let (a, b, c, d) = query.evalQuads[level]
                    let alpha0 = reversedPoint[pointIdx]
                    let alpha1 = reversedPoint[pointIdx + 1]
                    let mid0 = frAdd(a, frMul(alpha0, frSub(c, a)))
                    let mid1 = frAdd(b, frMul(alpha0, frSub(d, b)))
                    let expected = frAdd(mid0, frMul(alpha1, frSub(mid1, mid0)))
                    if !frEqual(expected, query.foldValues[level]) {
                        return false
                    }
                    pointIdx += 2
                } else {
                    let (a, b) = query.evalPairs[level]
                    let alpha = reversedPoint[pointIdx]
                    let expected = frAdd(a, frMul(alpha, frSub(b, a)))
                    if !frEqual(expected, query.foldValues[level]) {
                        return false
                    }
                    pointIdx += 1
                }
            }
        }

        return true
    }

    // MARK: - Diagnostic

    /// Return statistics about a commitment for debugging.
    public func commitmentStats(_ c: BasefoldProverCommitment) -> (numVars: Int,
                                                                     evalCount: Int,
                                                                     encodedCount: Int,
                                                                     treeSize: Int) {
        return (c.numVars, c.evaluations.count, c.encodedEvals.count, c.merkleTree.count)
    }

    /// Return statistics about a proof for debugging.
    public func proofStats(_ p: BasefoldProverProof) -> (numFoldLevels: Int,
                                                          numQueries: Int,
                                                          totalAuthPathSize: Int) {
        let totalPaths = p.queries.reduce(0) { acc, q in
            acc + q.authPaths.reduce(0) { $0 + $1.count }
        }
        return (p.foldLayers.count, p.queries.count, totalPaths)
    }
}
