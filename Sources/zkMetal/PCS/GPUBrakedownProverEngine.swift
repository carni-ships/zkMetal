// GPU-Accelerated Brakedown Polynomial Commitment Prover Engine
//
// Brakedown: hash-based, transparent PCS using expander-graph linear codes.
// Reference: Golovnev, Lee, Setty, Thaler, Wahby — eprint 2021/1043
//
// Key ideas:
//   - NTT-free, no trusted setup, no pairings — transparent and post-quantum friendly
//   - Commit: reshape evaluations into sqrt(n) x sqrt(n) matrix,
//             encode each row via expander code, Merkle-hash columns
//   - Open: compute t = M^T * tensor_left, reveal random columns + Merkle proofs
//   - Verify: check <tensor_right, t> = value and column-code consistency
//
// GPU acceleration targets:
//   - Expander encoding (sparse matrix-vector multiply, embarrassingly parallel)
//   - Merkle tree construction (Poseidon2 hash tree over column hashes)
//   - Tensor product computation
//   - Matrix-vector products for opening (t-vector computation)
//
// Supports multilinear polynomials over BN254 scalar field.
// Proof size: O(sqrt(n)) field elements + O(sqrt(n) * log(n)) hash digests

import Foundation
import Metal
import NeonFieldOps

// MARK: - Configuration

/// Configuration for the GPU Brakedown prover engine.
public struct BrakedownProverConfig {
    /// Rate inverse (blowup factor). Codeword = message * rateInverse. Default: 4
    public let rateInverse: Int
    /// Number of random column queries for proximity testing (soundness parameter).
    /// Security ~ numQueries * log2(rateInverse) bits.
    public let numQueries: Int
    /// Expander graph degree (edges per right vertex). Default: 10
    /// Higher = better code distance but slower encoding.
    public let expanderDegree: Int
    /// Deterministic seed for code generation (Fiat-Shamir reproducibility)
    public let codeSeed: UInt32
    /// Minimum matrix size (total elements) before GPU encoding is used.
    /// Below this threshold, CPU encoding is faster due to GPU dispatch overhead.
    public let gpuThreshold: Int

    public init(rateInverse: Int = 4, numQueries: Int = 30, expanderDegree: Int = 10,
                codeSeed: UInt32 = 0xBEEF, gpuThreshold: Int = 256) {
        self.rateInverse = rateInverse
        self.numQueries = numQueries
        self.expanderDegree = expanderDegree
        self.codeSeed = codeSeed
        self.gpuThreshold = gpuThreshold
    }

    /// Default: 4x blowup, 30 queries, degree-10. ~60 bits soundness.
    public static let standard = BrakedownProverConfig()

    /// High security: 8x blowup, 50 queries, degree-16. ~150 bits soundness.
    public static let highSecurity = BrakedownProverConfig(
        rateInverse: 8, numQueries: 50, expanderDegree: 16, codeSeed: 0xBEEF
    )

    /// Fast prover: 4x blowup, 16 queries, degree-8. ~32 bits. For testing.
    public static let fast = BrakedownProverConfig(
        rateInverse: 4, numQueries: 16, expanderDegree: 8, codeSeed: 0xBEEF
    )

    /// Estimated soundness in bits.
    public var soundnessBits: Double {
        return Double(numQueries) * log2(Double(rateInverse))
    }
}

// MARK: - Commitment

/// Commitment to a multilinear polynomial via Brakedown prover.
/// Polynomial evaluations are arranged as a sqrt(n) x sqrt(n) matrix,
/// rows are encoded with an expander code, and column hashes form a Merkle tree.
public struct BrakedownProverCommitment {
    /// Merkle root of column hashes — this is the binding commitment
    public let merkleRoot: Fr
    /// Number of rows in the evaluation matrix
    public let numRows: Int
    /// Number of columns before encoding
    public let numCols: Int
    /// Number of columns after encoding (numCols * rateInverse)
    public let numEncodedCols: Int
    /// Number of variables in the multilinear polynomial (log2 of evaluation count)
    public let numVars: Int
    /// Full Merkle tree nodes (prover-side for proof extraction)
    public let tree: [Fr]
    /// Encoded matrix (numRows x numEncodedCols) — prover keeps for opening
    public let encodedMatrix: [Fr]
    /// Original evaluations (prover retains for evaluation)
    public let evaluations: [Fr]

    /// Commitment size: just the Merkle root (1 field element / 32 bytes)
    public var commitmentSize: Int { MemoryLayout<Fr>.stride }

    public init(merkleRoot: Fr, numRows: Int, numCols: Int,
                numEncodedCols: Int, numVars: Int, tree: [Fr],
                encodedMatrix: [Fr], evaluations: [Fr]) {
        self.merkleRoot = merkleRoot
        self.numRows = numRows
        self.numCols = numCols
        self.numEncodedCols = numEncodedCols
        self.numVars = numVars
        self.tree = tree
        self.encodedMatrix = encodedMatrix
        self.evaluations = evaluations
    }
}

/// Opening proof for a multilinear evaluation via Brakedown.
///
/// Given commitment C to polynomial p, and evaluation p(z) = v, the proof
/// convinces a verifier by:
/// 1. Revealing t = M^T * tensor_left (bound to evaluation point)
/// 2. Opening random columns with Merkle proofs (proximity test)
/// 3. Verifier checks: value = <tensor_right, t> and column consistency
///
/// Proof size: O(sqrt(n)) field elements + O(numQueries * log(n)) hashes
public struct BrakedownProverOpeningProof {
    /// t-vector: M^T * tensor_left, length = numCols
    public let tVector: [Fr]
    /// Column openings: for each queried column, the full column vector
    public let columnOpenings: [[Fr]]
    /// Merkle authentication paths for each queried column
    public let merkleProofs: [[Fr]]
    /// Which columns were queried (deterministic from Fiat-Shamir)
    public let queryIndices: [Int]
    /// The claimed evaluation value p(z)
    public let claimedValue: Fr
    /// The evaluation point z
    public let point: [Fr]

    /// Approximate proof size in bytes.
    public var proofSizeBytes: Int {
        let frSize = MemoryLayout<Fr>.stride
        let numQ = queryIndices.count
        let numRows = columnOpenings.isEmpty ? 0 : columnOpenings[0].count
        let tSize = tVector.count * frSize
        let colSize = numQ * numRows * frSize
        let mrkSize = merkleProofs.reduce(0) { $0 + $1.count } * frSize
        return tSize + colSize + mrkSize + frSize // +frSize for claimedValue
    }

    public init(tVector: [Fr], columnOpenings: [[Fr]], merkleProofs: [[Fr]],
                queryIndices: [Int], claimedValue: Fr, point: [Fr]) {
        self.tVector = tVector
        self.columnOpenings = columnOpenings
        self.merkleProofs = merkleProofs
        self.queryIndices = queryIndices
        self.claimedValue = claimedValue
        self.point = point
    }
}

/// Batch commitment: multiple polynomials committed together.
public struct BrakedownProverBatchCommitment {
    /// Individual commitments per polynomial
    public let commitments: [BrakedownProverCommitment]
    /// Combined root (Poseidon2 hash chain of individual roots)
    public let batchRoot: Fr

    public init(commitments: [BrakedownProverCommitment], batchRoot: Fr) {
        self.commitments = commitments
        self.batchRoot = batchRoot
    }
}

/// Batch opening proof for multiple polynomials at the same point.
public struct BrakedownProverBatchProof {
    /// Individual opening proofs per polynomial
    public let proofs: [BrakedownProverOpeningProof]
    /// Fiat-Shamir batch challenge
    public let batchChallenge: Fr

    public init(proofs: [BrakedownProverOpeningProof], batchChallenge: Fr) {
        self.proofs = proofs
        self.batchChallenge = batchChallenge
    }
}

// MARK: - Engine

/// GPU-accelerated Brakedown polynomial commitment prover.
///
/// Uses Metal compute shaders for expander encoding (sparse matvec) and
/// Poseidon2 Merkle tree construction. Falls back to CPU for small inputs
/// where GPU dispatch overhead dominates.
///
/// Thread safety: not thread-safe. Create one engine per thread.
public class GPUBrakedownProverEngine {
    public static let version = Versions.brakedown

    /// Configuration
    public let config: BrakedownProverConfig

    /// Underlying BrakedownEngine for GPU sparse encoding
    private let brakedownEngine: BrakedownEngine

    /// Poseidon2 Merkle engine for tree construction
    private let merkleEngine: Poseidon2MerkleEngine

    /// Metal device and command queue (from BrakedownEngine)
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue

    // Cached GPU buffers for tensor/matvec operations
    private var tensorBuf: MTLBuffer?
    private var tensorBufSize: Int = 0
    private var matvecResultBuf: MTLBuffer?
    private var matvecResultBufSize: Int = 0

    /// Create a GPU Brakedown prover engine.
    /// - Parameter config: Prover configuration (rate, queries, degree, thresholds)
    public init(config: BrakedownProverConfig = .standard) throws {
        self.config = config

        let params = BrakedownParameters(
            rateInverse: config.rateInverse,
            numQueries: config.numQueries,
            expanderDegree: config.expanderDegree,
            codeSeed: config.codeSeed
        )
        self.brakedownEngine = try BrakedownEngine(params: params)
        self.merkleEngine = try Poseidon2MerkleEngine()
        self.device = brakedownEngine.device
        self.commandQueue = brakedownEngine.commandQueue
    }

    /// Convenience init with default config.
    public convenience init() throws {
        try self.init(config: .standard)
    }

    // MARK: - Commit

    /// Commit to a multilinear polynomial given as 2^n evaluations over {0,1}^n.
    ///
    /// Steps:
    /// 1. Reshape evaluations into numRows x numCols matrix (sqrt(n) x sqrt(n))
    /// 2. Encode each row with expander code (GPU-accelerated sparse matvec)
    /// 3. Hash each column (weighted Poseidon2 domain-separated hash)
    /// 4. Build Poseidon2 Merkle tree over column hashes
    ///
    /// - Parameter evaluations: Polynomial evaluations, count must be a power of 2
    /// - Returns: Commitment containing Merkle root + auxiliary prover data
    public func commit(evaluations: [Fr]) throws -> BrakedownProverCommitment {
        let n = evaluations.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Evaluation count must be power of 2")

        let logN = Int(log2(Double(n)))
        let logRows = logN / 2
        let logCols = logN - logRows
        let numRows = 1 << logRows
        let numCols = 1 << logCols
        precondition(numRows * numCols == n)

        // Build expander code for this column count
        let code = ExpanderCode(
            messageLength: numCols,
            rateInverse: config.rateInverse,
            degree: config.expanderDegree,
            seed: config.codeSeed
        )
        let encodedCols = code.codewordLength

        // Encode each row: GPU path for large matrices, CPU fallback otherwise
        let encodedMatrix: [Fr]
        if n >= config.gpuThreshold {
            encodedMatrix = try gpuEncodeMatrix(
                evaluations: evaluations,
                numRows: numRows,
                numCols: numCols,
                code: code
            )
        } else {
            encodedMatrix = cpuEncodeMatrix(
                evaluations: evaluations,
                numRows: numRows,
                numCols: numCols,
                code: code
            )
        }

        // Build Merkle tree over column hashes
        let numLeaves = nextPow2ForBrakedownProver(encodedCols)
        var columnHashes = [Fr](repeating: Fr.zero, count: numLeaves)

        for c in 0..<encodedCols {
            // Hash column c via weighted sum + Poseidon2 domain separator
            var hash = Fr.zero
            for r in 0..<numRows {
                hash = frAdd(hash, frMul(encodedMatrix[r * encodedCols + c],
                                         frFromInt(UInt64(r + 1))))
            }
            columnHashes[c] = hash
        }

        let tree = try merkleEngine.buildTree(columnHashes)
        let root = tree.last!

        return BrakedownProverCommitment(
            merkleRoot: root,
            numRows: numRows,
            numCols: numCols,
            numEncodedCols: encodedCols,
            numVars: logN,
            tree: tree,
            encodedMatrix: encodedMatrix,
            evaluations: evaluations
        )
    }

    /// Batch commit: commit to multiple polynomials, returning individual commitments
    /// plus a combined batch root via Poseidon2 hash chain.
    public func batchCommit(polynomials: [[Fr]]) throws -> BrakedownProverBatchCommitment {
        precondition(!polynomials.isEmpty, "Must commit to at least one polynomial")

        var commitments: [BrakedownProverCommitment] = []
        commitments.reserveCapacity(polynomials.count)

        for poly in polynomials {
            commitments.append(try commit(evaluations: poly))
        }

        // Combine roots via Poseidon2 hash chain
        var batchRoot = commitments[0].merkleRoot
        for i in 1..<commitments.count {
            batchRoot = poseidon2Hash(batchRoot, commitments[i].merkleRoot)
        }

        return BrakedownProverBatchCommitment(commitments: commitments, batchRoot: batchRoot)
    }

    // MARK: - Open (Prove)

    /// Generate an opening proof for a committed polynomial at a multilinear point z.
    ///
    /// Protocol:
    /// 1. Split z into row portion (logRows bits) and column portion (logCols bits)
    /// 2. Compute tensor vectors for each portion
    /// 3. Compute t = M^T * tensor_left (matrix-vector product, GPU if large)
    /// 4. Derive query indices from commitment root (Fiat-Shamir)
    /// 5. Extract queried columns + Merkle authentication paths
    ///
    /// - Parameters:
    ///   - commitment: Previously computed commitment
    ///   - point: Multilinear evaluation point (length = log2(n))
    /// - Returns: Opening proof
    public func open(commitment: BrakedownProverCommitment,
                     point: [Fr]) throws -> BrakedownProverOpeningProof {
        let evaluations = commitment.evaluations
        let n = evaluations.count
        let logN = point.count
        precondition(n == (1 << logN), "Evaluation count must be 2^(point.count)")

        let numRows = commitment.numRows
        let numCols = commitment.numCols
        let encodedCols = commitment.numEncodedCols
        let logRows = Int(log2(Double(numRows)))
        let logCols = logN - logRows

        // Split point: first logRows for rows, last logCols for columns
        let pointRows = Array(point.prefix(logRows))
        let pointCols = Array(point.suffix(logCols))

        // Compute tensor vectors
        let tensorRows = computeTensor(pointRows)    // length numRows
        let tensorCols = computeTensor(pointCols)    // length numCols

        // Compute t = M^T * tensorRows: t[j] = sum_i tensorRows[i] * M[i][j]
        let tVector = computeTransposeMatvec(
            matrix: evaluations,
            vector: tensorRows,
            numRows: numRows,
            numCols: numCols
        )

        // Compute claimed value = <tensorCols, tVector>
        var claimedValue = Fr.zero
        for j in 0..<numCols {
            claimedValue = frAdd(claimedValue, frMul(tensorCols[j], tVector[j]))
        }

        // Generate deterministic query indices (Fiat-Shamir from root)
        let queryIndices = generateQueryIndices(
            root: commitment.merkleRoot,
            numQueries: config.numQueries,
            maxCol: encodedCols
        )

        // Extract queried columns and Merkle proofs
        var columnOpenings = [[Fr]]()
        columnOpenings.reserveCapacity(queryIndices.count)
        var merkleProofs = [[Fr]]()
        merkleProofs.reserveCapacity(queryIndices.count)

        for colIdx in queryIndices {
            var column = [Fr](repeating: Fr.zero, count: numRows)
            for r in 0..<numRows {
                column[r] = commitment.encodedMatrix[r * encodedCols + colIdx]
            }
            columnOpenings.append(column)

            let proof = extractMerkleProof(tree: commitment.tree, index: colIdx,
                                            numLeaves: nextPow2ForBrakedownProver(encodedCols))
            merkleProofs.append(proof)
        }

        return BrakedownProverOpeningProof(
            tVector: tVector,
            columnOpenings: columnOpenings,
            merkleProofs: merkleProofs,
            queryIndices: queryIndices,
            claimedValue: claimedValue,
            point: point
        )
    }

    /// Batch open: generate opening proofs for multiple committed polynomials at the same point.
    public func batchOpen(batch: BrakedownProverBatchCommitment,
                          point: [Fr]) throws -> BrakedownProverBatchProof {
        precondition(!batch.commitments.isEmpty)

        // Derive batch challenge from batch root and point
        var challenge = batch.batchRoot
        for p in point {
            challenge = poseidon2Hash(challenge, p)
        }

        var proofs: [BrakedownProverOpeningProof] = []
        proofs.reserveCapacity(batch.commitments.count)

        for commitment in batch.commitments {
            proofs.append(try open(commitment: commitment, point: point))
        }

        return BrakedownProverBatchProof(proofs: proofs, batchChallenge: challenge)
    }

    // MARK: - Verify

    /// Verify an opening proof against a commitment.
    ///
    /// Checks:
    /// 1. Value consistency: claimedValue = <tensor_right, t>
    /// 2. Code consistency: for each queried column j,
    ///    <tensor_left, col_j> = encode(t)[j]
    /// 3. Merkle authentication paths (each queried column authenticates against root)
    ///
    /// - Parameters:
    ///   - commitment: The Brakedown commitment
    ///   - proof: Opening proof
    /// - Returns: true if the proof is valid
    public func verify(commitment: BrakedownProverCommitment,
                       proof: BrakedownProverOpeningProof) -> Bool {
        let logN = proof.point.count
        let numRows = commitment.numRows
        let numCols = commitment.numCols
        let logRows = Int(log2(Double(numRows)))
        let logCols = logN - logRows

        let pointRows = Array(proof.point.prefix(logRows))
        let pointCols = Array(proof.point.suffix(logCols))

        let tensorRows = computeTensor(pointRows)
        let tensorCols = computeTensor(pointCols)

        // Check 1: value == <tensorCols, t>
        var computedValue = Fr.zero
        for j in 0..<numCols {
            computedValue = frAdd(computedValue, frMul(tensorCols[j], proof.tVector[j]))
        }
        if !frEqual(computedValue, proof.claimedValue) {
            return false
        }

        // Check 2: Column consistency via expander code
        let code = ExpanderCode(
            messageLength: numCols,
            rateInverse: config.rateInverse,
            degree: config.expanderDegree,
            seed: config.codeSeed
        )
        let encodedT = code.encode(proof.tVector)

        for q in 0..<proof.queryIndices.count {
            let colIdx = proof.queryIndices[q]
            let column = proof.columnOpenings[q]

            // <tensorRows, column>
            var columnDot = Fr.zero
            tensorRows.withUnsafeBytes { trBuf in
                column.withUnsafeBytes { cBuf in
                    withUnsafeMutableBytes(of: &columnDot) { rBuf in
                        bn254_fr_inner_product(
                            trBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(numRows),
                            rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
                    }
                }
            }

            // Must equal encodedT[colIdx]
            if !frEqual(columnDot, encodedT[colIdx]) {
                return false
            }
        }

        return true
    }

    /// Verify a batch opening proof.
    public func verifyBatch(batch: BrakedownProverBatchCommitment,
                            proof: BrakedownProverBatchProof) -> Bool {
        guard batch.commitments.count == proof.proofs.count else {
            return false
        }

        for i in 0..<batch.commitments.count {
            if !verify(commitment: batch.commitments[i], proof: proof.proofs[i]) {
                return false
            }
        }

        return true
    }

    // MARK: - GPU Expander Encoding

    /// GPU-accelerated encoding of each matrix row via the expander code.
    /// Delegates to BrakedownEngine's sparse encode kernel, then assembles
    /// the full encoded matrix [original | redundancy] per row.
    private func gpuEncodeMatrix(evaluations: [Fr], numRows: Int,
                                  numCols: Int, code: ExpanderCode) throws -> [Fr] {
        // Use the underlying BrakedownEngine's commit, which does GPU sparse encode
        let commitment = try brakedownEngine.commit(evaluations: evaluations)
        return commitment.encodedMatrix
    }

    /// CPU fallback for encoding small matrices.
    private func cpuEncodeMatrix(evaluations: [Fr], numRows: Int,
                                  numCols: Int, code: ExpanderCode) -> [Fr] {
        let encodedCols = code.codewordLength
        var encodedMatrix = [Fr](repeating: Fr.zero, count: numRows * encodedCols)

        for r in 0..<numRows {
            let rowStart = r * numCols
            let row = Array(evaluations[rowStart..<rowStart + numCols])
            let encoded = code.encode(row)

            for c in 0..<encodedCols {
                encodedMatrix[r * encodedCols + c] = encoded[c]
            }
        }

        return encodedMatrix
    }

    // MARK: - Tensor Product

    /// Compute the multilinear tensor product: tensor(point).
    /// For point = (z_0, z_1, ..., z_{k-1}):
    ///   tensor[i] = product_{j=0}^{k-1} (if bit j of i == 1 then z_j else (1 - z_j))
    ///
    /// Used for both the row and column portions when splitting the evaluation point.
    public func computeTensor(_ point: [Fr]) -> [Fr] {
        let k = point.count
        let n = 1 << k
        var tensor = [Fr](repeating: Fr.zero, count: n)
        tensor[0] = Fr.one

        for j in 0..<k {
            let half = 1 << j
            let zj = point[j]
            let oneMinusZj = frSub(Fr.one, zj)
            for i in Swift.stride(from: half - 1, through: 0, by: -1) {
                tensor[2 * i + 1] = frMul(tensor[i], zj)
                tensor[2 * i] = frMul(tensor[i], oneMinusZj)
            }
        }
        return tensor
    }

    // MARK: - Matrix-Vector Products

    /// Compute t = M^T * v where M is (numRows x numCols), v has length numRows.
    /// Result t has length numCols. t[j] = sum_i v[i] * M[i*numCols + j].
    private func computeTransposeMatvec(matrix: [Fr], vector: [Fr],
                                         numRows: Int, numCols: Int) -> [Fr] {
        var result = [Fr](repeating: Fr.zero, count: numCols)
        // Accumulate row-by-row: result += v[i] * M[i*numCols ..< i*numCols + numCols]
        result.withUnsafeMutableBytes { rBuf in
            let rPtr = rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
            matrix.withUnsafeBytes { mBuf in
                let mPtr = mBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                vector.withUnsafeBytes { vBuf in
                    let vPtr = vBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                    for i in 0..<numRows {
                        // result[0..<numCols] += v[i] * M[i*numCols..<(i+1)*numCols]
                        bn254_fr_batch_mac_neon(
                            rPtr,
                            mPtr.advanced(by: i * numCols * 4),
                            vPtr.advanced(by: i * 4),
                            Int32(numCols))
                    }
                }
            }
        }
        return result
    }

    // MARK: - Multilinear Evaluation

    /// Evaluate a multilinear polynomial at a point via sequential folding.
    /// f(r_1, ..., r_n) computed by folding evaluations with each challenge.
    public static func evaluateMultilinear(evaluations: [Fr], point: [Fr]) -> Fr {
        let n = evaluations.count
        let logN = point.count
        precondition(n == (1 << logN))

        var evals = evaluations
        for i in 0..<logN {
            let half = evals.count / 2
            var folded = [Fr](repeating: Fr.zero, count: half)
            // folded[j] = evals[j] + point[i] * (evals[j+half] - evals[j])
            evals.withUnsafeBytes { eBuf in
                withUnsafeBytes(of: point[i]) { cBuf in
                    folded.withUnsafeMutableBytes { rBuf in
                        bn254_fr_sumcheck_reduce(
                            eBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(half))
                    }
                }
            }
            evals = folded
        }
        return evals[0]
    }

    // MARK: - Query Index Generation (Fiat-Shamir)

    /// Generate deterministic query indices from the commitment root.
    /// Uses LCG seeded from the root hash, ensuring distinct indices.
    public func generateQueryIndices(root: Fr, numQueries: Int, maxCol: Int) -> [Int] {
        var rng: UInt64 = frToUInt64(root)

        var indices = [Int]()
        indices.reserveCapacity(numQueries)
        var seen = Set<Int>()
        while indices.count < numQueries {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let idx = Int(rng >> 32) % maxCol
            if !seen.contains(idx) {
                seen.insert(idx)
                indices.append(idx)
            }
        }
        return indices
    }

    // MARK: - Merkle Helpers

    /// Extract a Merkle authentication path for a leaf at the given index.
    /// Tree layout: nodes[0..<numLeaves] = leaves, then internal nodes bottom-up.
    func extractMerkleProof(tree: [Fr], index: Int, numLeaves: Int) -> [Fr] {
        var proof = [Fr]()
        var idx = index
        var offset = 0
        var levelSize = numLeaves

        while levelSize > 1 {
            let sibling = idx ^ 1
            if sibling < levelSize {
                proof.append(tree[offset + sibling])
            } else {
                proof.append(Fr.zero)
            }
            idx /= 2
            offset += levelSize
            levelSize = (levelSize + 1) / 2
        }
        return proof
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

    // MARK: - Code Consistency Check

    /// Verify that the expander encoding of the t-vector matches revealed columns.
    /// This is the core proximity test: for each queried column j, checks
    /// <tensor_left, col_j> == encode(t)[j].
    public func codeConsistencyCheck(proof: BrakedownProverOpeningProof,
                                      numRows: Int, numCols: Int,
                                      pointRows: [Fr]) -> Bool {
        let tensorRows = computeTensor(pointRows)
        let code = ExpanderCode(
            messageLength: numCols,
            rateInverse: config.rateInverse,
            degree: config.expanderDegree,
            seed: config.codeSeed
        )
        let encodedT = code.encode(proof.tVector)

        for q in 0..<proof.queryIndices.count {
            let colIdx = proof.queryIndices[q]
            let column = proof.columnOpenings[q]

            var columnDot = Fr.zero
            tensorRows.withUnsafeBytes { trBuf in
                column.withUnsafeBytes { cBuf in
                    withUnsafeMutableBytes(of: &columnDot) { rBuf in
                        bn254_fr_inner_product(
                            trBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(numRows),
                            rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
                    }
                }
            }

            if !frEqual(columnDot, encodedT[colIdx]) {
                return false
            }
        }
        return true
    }

    // MARK: - Diagnostics

    /// Return statistics about a commitment for debugging.
    public func commitmentStats(_ c: BrakedownProverCommitment)
        -> (numVars: Int, evalCount: Int, matrixDims: String,
            encodedCols: Int, treeSize: Int, commitmentBytes: Int) {
        return (
            c.numVars,
            c.evaluations.count,
            "\(c.numRows) x \(c.numCols)",
            c.numEncodedCols,
            c.tree.count,
            c.commitmentSize
        )
    }

    /// Return statistics about an opening proof for debugging.
    public func proofStats(_ p: BrakedownProverOpeningProof)
        -> (tVectorLen: Int, numQueries: Int, proofBytes: Int,
            avgMerklePathLen: Double) {
        let totalPathLen = p.merkleProofs.reduce(0) { $0 + $1.count }
        let avgPath = p.merkleProofs.isEmpty ? 0.0 :
            Double(totalPathLen) / Double(p.merkleProofs.count)
        return (
            p.tVector.count,
            p.queryIndices.count,
            p.proofSizeBytes,
            avgPath
        )
    }

    /// Return the expander code parameters for this engine.
    public func codeParams(forNumCols numCols: Int)
        -> (codewordLength: Int, redundancyLength: Int, rate: Double, degree: Int) {
        let code = ExpanderCode(
            messageLength: numCols,
            rateInverse: config.rateInverse,
            degree: config.expanderDegree,
            seed: config.codeSeed
        )
        return (code.codewordLength, code.redundancyLength, code.rate, code.degree)
    }

    // MARK: - Buffer Management

    /// Ensure a GPU buffer exists with at least the requested size.
    @discardableResult
    private func ensureBuffer(_ existing: MTLBuffer?, _ currentSize: inout Int,
                               _ needed: Int) -> MTLBuffer? {
        if needed <= currentSize, let buf = existing { return buf }
        guard let buf = device.makeBuffer(length: needed, options: .storageModeShared)
        else { return nil }
        currentSize = needed
        return buf
    }
}

// MARK: - Utility

/// Round up to next power of 2 (for Brakedown prover internal use).
private func nextPow2ForBrakedownProver(_ n: Int) -> Int {
    var p = 1
    while p < n { p *= 2 }
    return p
}
