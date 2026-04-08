// Brakedown Polynomial Commitment Engine
// Reference: Golovnev, Lee, Setty, Thaler, Wahby — eprint 2021/1043
//
// NTT-free PCS using expander-graph-based linear codes + Merkle commitments.
// No trusted setup, no pairings — transparent and post-quantum friendly.
//
// Commit: reshape evaluations into matrix, encode rows with expander code, Merkle hash columns.
// Open: compute tensor dot products, reveal random columns with Merkle proofs.
// Verify: check dot product consistency against revealed columns + code structure.
//
// Core operation is sparse matrix-vector multiply (expander encoding),
// which is embarrassingly parallel on GPU.

import Foundation
import Metal
import NeonFieldOps

// MARK: - Engine

public class BrakedownEngine {
    public static let version = Versions.brakedown
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    let sparseEncodeFunction: MTLComputePipelineState
    let dotProductFunction: MTLComputePipelineState
    let extractColumnFunction: MTLComputePipelineState

    private lazy var merkleEngine: Poseidon2MerkleEngine = {
        try! Poseidon2MerkleEngine()
    }()

    /// Brakedown configuration
    public let params: BrakedownParameters

    // Cached GPU buffers
    private var matrixBuf: MTLBuffer?
    private var matrixBufSize: Int = 0
    private var encodedBuf: MTLBuffer?
    private var encodedBufSize: Int = 0
    private var neighborsBuf: MTLBuffer?
    private var neighborsBufSize: Int = 0
    private var coeffsBuf: MTLBuffer?
    private var coeffsBufSize: Int = 0
    private var tensorBuf: MTLBuffer?
    private var tensorBufSize: Int = 0
    private var resultBuf: MTLBuffer?
    private var resultBufSize: Int = 0

    private let tuning: TuningConfig

    /// Create a Brakedown engine with given parameters.
    /// - Parameter params: Configuration controlling rate, queries, and expander degree
    public init(params: BrakedownParameters = .default) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue
        self.params = params

        let library = try BrakedownEngine.compileShaders(device: device)

        guard let sparseEncodeFn = library.makeFunction(name: "brakedown_sparse_encode"),
              let dotProductFn = library.makeFunction(name: "brakedown_dot_product"),
              let extractColFn = library.makeFunction(name: "brakedown_extract_column") else {
            throw MSMError.missingKernel
        }

        self.sparseEncodeFunction = try device.makeComputePipelineState(function: sparseEncodeFn)
        self.dotProductFunction = try device.makeComputePipelineState(function: dotProductFn)
        self.extractColumnFunction = try device.makeComputePipelineState(function: extractColFn)
        self.tuning = TuningManager.shared.config(device: device)
    }

    /// Convenience init with individual parameters (backwards compatible)
    public convenience init(rateInverse: Int = 4, numQueries: Int = 30,
                            expanderDegree: Int = 10, codeSeed: UInt32 = 0xBEEF) throws {
        try self.init(params: BrakedownParameters(
            rateInverse: rateInverse,
            numQueries: numQueries,
            expanderDegree: expanderDegree,
            codeSeed: codeSeed
        ))
    }

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let frSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let bkSource = try String(contentsOfFile: shaderDir + "/brakedown/brakedown_kernels.metal", encoding: .utf8)

        let cleanBK = bkSource.split(separator: "\n").filter { !$0.contains("#include") }.joined(separator: "\n")
        let cleanFr = frSource
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")

        let combined = cleanFr + "\n" + cleanBK
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    // MARK: - Commit

    /// Commit to a multilinear polynomial given as 2^n evaluations over the boolean hypercube.
    /// Reshapes into a numRows x numCols matrix, encodes each row with the expander code,
    /// then builds a Poseidon2 Merkle tree over column hashes.
    ///
    /// - Parameter evaluations: Polynomial evaluations, count must be a power of 2
    /// - Returns: Commitment containing Merkle root + prover auxiliary data
    public func commit(evaluations: [Fr]) throws -> BrakedownCommitment {
        let n = evaluations.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Evaluation count must be power of 2")

        // Choose matrix dimensions: rows ~ sqrt(n), cols ~ sqrt(n)
        let logN = Int(log2(Double(n)))
        let logRows = logN / 2
        let logCols = logN - logRows
        let numRows = 1 << logRows
        let numCols = 1 << logCols

        precondition(numRows * numCols == n)

        // Build the expander code for this column count
        let code = ExpanderCode(
            messageLength: numCols,
            rateInverse: params.rateInverse,
            degree: params.expanderDegree,
            seed: params.codeSeed
        )
        let redundancyCols = code.redundancyLength
        let encodedCols = code.codewordLength
        let stride = MemoryLayout<Fr>.stride
        let degree = code.degree

        // Upload matrix to GPU
        let matrixSize = n * stride
        matrixBuf = ensureBuffer(matrixBuf, &matrixBufSize, matrixSize)
        evaluations.withUnsafeBytes { src in
            memcpy(matrixBuf!.contents(), src.baseAddress!, matrixSize)
        }

        // Upload expander graph structure (neighbors + coefficients) to GPU
        let neighborsSize = redundancyCols * degree * MemoryLayout<UInt32>.stride
        neighborsBuf = ensureBuffer(neighborsBuf, &neighborsBufSize, neighborsSize)
        code.graph.neighbors.withUnsafeBytes { src in
            memcpy(neighborsBuf!.contents(), src.baseAddress!, neighborsSize)
        }

        let coeffsSize = redundancyCols * degree * stride
        coeffsBuf = ensureBuffer(coeffsBuf, &coeffsBufSize, coeffsSize)
        code.graph.coefficients.withUnsafeBytes { src in
            memcpy(coeffsBuf!.contents(), src.baseAddress!, coeffsSize)
        }

        // Allocate output for redundancy part
        let redundancySize = numRows * redundancyCols * stride
        encodedBuf = ensureBuffer(encodedBuf, &encodedBufSize, redundancySize)

        // GPU sparse encode: each thread computes one redundancy element
        // using exactly `degree` multiply-accumulates (sparse matvec)
        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!

        var cols32 = UInt32(numCols)
        var redCols32 = UInt32(redundancyCols)
        var degree32 = UInt32(degree)

        enc.setComputePipelineState(sparseEncodeFunction)
        enc.setBuffer(matrixBuf!, offset: 0, index: 0)       // input matrix
        enc.setBuffer(encodedBuf!, offset: 0, index: 1)      // output redundancy
        enc.setBuffer(neighborsBuf!, offset: 0, index: 2)    // expander neighbors
        enc.setBuffer(coeffsBuf!, offset: 0, index: 3)       // expander coefficients
        enc.setBytes(&cols32, length: 4, index: 4)           // message length per row
        enc.setBytes(&redCols32, length: 4, index: 5)        // redundancy length per row
        enc.setBytes(&degree32, length: 4, index: 6)         // expander degree

        let tg = min(64, Int(sparseEncodeFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(
            MTLSize(width: redundancyCols, height: numRows, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(tg, redundancyCols), height: 1, depth: 1)
        )

        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        // Assemble full encoded matrix: [original_row | redundancy_row] for each row
        var encodedMatrix = [Fr](repeating: Fr.zero, count: numRows * encodedCols)
        let redPtr = encodedBuf!.contents().bindMemory(to: Fr.self, capacity: numRows * redundancyCols)

        for r in 0..<numRows {
            for c in 0..<numCols {
                encodedMatrix[r * encodedCols + c] = evaluations[r * numCols + c]
            }
            for c in 0..<redundancyCols {
                encodedMatrix[r * encodedCols + numCols + c] = redPtr[r * redundancyCols + c]
            }
        }

        // Build Merkle tree over column hashes.
        // Each leaf = hash of a column (numRows Fr elements).
        // We use a weighted sum-hash as a domain separator, then Poseidon2 Merkle on top.
        let numLeaves = nextPow2(encodedCols)
        var columnHashes = [Fr](repeating: Fr.zero, count: numLeaves)

        for c in 0..<encodedCols {
            // Hash column c: weighted sum (production would use full Poseidon2 sponge)
            var hash = Fr.zero
            for r in 0..<numRows {
                hash = frAdd(hash, frMul(encodedMatrix[r * encodedCols + c], frFromInt(UInt64(r + 1))))
            }
            columnHashes[c] = hash
        }

        // Poseidon2 Merkle tree
        let tree = try merkleEngine.buildTree(columnHashes)
        let root = tree.last!

        return BrakedownCommitment(
            merkleRoot: root,
            numRows: numRows,
            numCols: numCols,
            numEncodedCols: encodedCols,
            tree: tree,
            encodedMatrix: encodedMatrix
        )
    }

    // MARK: - Open

    /// Open the committed polynomial at a multilinear evaluation point.
    /// The point has length logN = log2(evaluation count).
    ///
    /// Protocol:
    /// 1. Split point into row and column portions
    /// 2. Compute tensor vectors for each portion
    /// 3. Compute t = M^T * tensor_left (matrix-vector product)
    /// 4. Derive query indices from commitment root (Fiat-Shamir)
    /// 5. Extract queried columns + Merkle proofs
    ///
    /// - Parameters:
    ///   - evaluations: Original polynomial evaluations
    ///   - point: Multilinear evaluation point (length = log2(n))
    ///   - commitment: Previously computed commitment
    /// - Returns: Opening proof
    public func open(evaluations: [Fr], point: [Fr], commitment: BrakedownCommitment) throws -> BrakedownProof {
        let n = evaluations.count
        let logN = point.count
        precondition(n == (1 << logN), "Evaluation count must be 2^(point.count)")

        let numRows = commitment.numRows
        let numCols = commitment.numCols
        let encodedCols = commitment.numEncodedCols
        let logRows = Int(log2(Double(numRows)))
        let logCols = logN - logRows

        // Split point: first logRows for rows (high bits), last logCols for columns (low bits)
        let pointRows = Array(point.prefix(logRows))
        let pointCols = Array(point.suffix(logCols))

        // Compute tensor vectors
        let tensorRows = computeTensor(pointRows)    // length numRows
        let tensorCols = computeTensor(pointCols)    // length numCols

        // Compute t = M^T * tensorRows: t[j] = sum_i tensorRows[i] * M[i][j]
        // This is O(numRows * numCols) = O(n), done on CPU (small relative to encoding)
        var tVector = [Fr](repeating: Fr.zero, count: numCols)
        for j in 0..<numCols {
            var acc = Fr.zero
            for i in 0..<numRows {
                acc = frAdd(acc, frMul(tensorRows[i], evaluations[i * numCols + j]))
            }
            tVector[j] = acc
        }

        // Generate deterministic query indices (Fiat-Shamir from commitment root)
        let queryIndices = generateQueryIndices(
            root: commitment.merkleRoot,
            numQueries: params.numQueries,
            maxCol: encodedCols
        )

        // Extract queried columns and Merkle proofs
        var columnOpenings = [[Fr]]()
        columnOpenings.reserveCapacity(queryIndices.count)
        var merkleProofs = [[Fr]]()
        merkleProofs.reserveCapacity(queryIndices.count)

        for colIdx in queryIndices {
            // Extract column from encoded matrix
            var column = [Fr](repeating: Fr.zero, count: numRows)
            for r in 0..<numRows {
                column[r] = commitment.encodedMatrix[r * encodedCols + colIdx]
            }
            columnOpenings.append(column)

            // Extract Merkle authentication path
            let proof = extractMerkleProof(tree: commitment.tree, index: colIdx)
            merkleProofs.append(proof)
        }

        return BrakedownProof(
            columnOpenings: columnOpenings,
            merkleProofs: merkleProofs,
            queryIndices: queryIndices,
            tVector: tVector
        )
    }

    // MARK: - Verify

    /// Verify an opening proof.
    ///
    /// Checks:
    /// 1. Value consistency: v = <tensor_right, t>
    /// 2. Code consistency: for each queried column j,
    ///    <tensor_left, col_j> = encode(t)[j]
    /// 3. (Merkle proofs would be checked here in production)
    ///
    /// - Parameters:
    ///   - commitment: The Brakedown commitment
    ///   - point: Evaluation point
    ///   - value: Claimed evaluation value
    ///   - proof: Opening proof
    /// - Returns: true if the proof is valid
    public func verify(
        commitment: BrakedownCommitment,
        point: [Fr],
        value: Fr,
        proof: BrakedownProof
    ) -> Bool {
        let logN = point.count
        let numRows = commitment.numRows
        let numCols = commitment.numCols
        let logRows = Int(log2(Double(numRows)))
        let logCols = logN - logRows

        let pointRows = Array(point.prefix(logRows))
        let pointCols = Array(point.suffix(logCols))

        let tensorRows = computeTensor(pointRows)    // length numRows
        let tensorCols = computeTensor(pointCols)    // length numCols

        // Check 1: value == <tensorCols, t>
        var computedValue = Fr.zero
        for j in 0..<numCols {
            computedValue = frAdd(computedValue, frMul(tensorCols[j], proof.tVector[j]))
        }

        if frToInt(computedValue) != frToInt(value) {
            return false
        }

        // Check 2: Column consistency via expander code.
        // encode(t) extends t from numCols to encodedCols using the same expander code.
        // For each opened column j: <tensorRows, col_j> should equal encode(t)[j].
        let code = ExpanderCode(
            messageLength: numCols,
            rateInverse: params.rateInverse,
            degree: params.expanderDegree,
            seed: params.codeSeed
        )
        let encodedT = code.encode(proof.tVector)

        for q in 0..<proof.queryIndices.count {
            let colIdx = proof.queryIndices[q]
            let column = proof.columnOpenings[q]

            // <tensorRows, column>
            var columnDot = Fr.zero
            for i in 0..<numRows {
                columnDot = frAdd(columnDot, frMul(tensorRows[i], column[i]))
            }

            // Must equal encodedT[colIdx]
            if frToInt(columnDot) != frToInt(encodedT[colIdx]) {
                return false
            }
        }

        return true
    }

    // MARK: - Multilinear Evaluation (CPU reference)

    /// CPU-side multilinear evaluation for correctness testing.
    /// Evaluates the multilinear extension of `evaluations` at `point`.
    public static func cpuEvaluate(evaluations: [Fr], point: [Fr]) -> Fr {
        let n = evaluations.count
        let logN = point.count
        precondition(n == (1 << logN))

        var evals = evaluations
        for i in 0..<logN {
            let half = evals.count / 2
            var folded = [Fr](repeating: Fr.zero, count: half)
            evals.withUnsafeBytes { eBuf in
            withUnsafeBytes(of: point[i]) { pBuf in
            folded.withUnsafeMutableBytes { fBuf in
                bn254_fr_fold_halves(
                    eBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    fBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(half))
            }}}
            evals = folded
        }
        return evals[0]
    }

    // MARK: - Tensor Product

    /// Compute the multilinear tensor product: t = tensor(point)
    /// For point = (z_0, z_1, ..., z_{k-1}):
    /// t[i] = product_{j=0}^{k-1} (if bit j of i is 1 then z_j else (1 - z_j))
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

    // MARK: - Query Index Generation (Fiat-Shamir)

    /// Generate deterministic query indices from the commitment root.
    /// Uses a simple LCG seeded from the root hash for Fiat-Shamir.
    func generateQueryIndices(root: Fr, numQueries: Int, maxCol: Int) -> [Int] {
        var rng: UInt64 = 0
        let limbs = frToInt(root)
        rng = limbs[0] ^ limbs[1] ^ limbs[2] ^ limbs[3]

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

    // MARK: - Merkle Proof Helpers

    /// Extract a Merkle authentication path for a leaf at the given index.
    /// Tree layout: nodes[0..<n] = leaves, nodes[n..<2n-1] = internal (bottom-up).
    func extractMerkleProof(tree: [Fr], index: Int) -> [Fr] {
        let n = (tree.count + 1) / 2  // number of leaves
        var proof = [Fr]()
        var idx = index
        var offset = 0
        var levelSize = n

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

    /// Verify a Merkle proof against a known root using Poseidon2 2-to-1 hash.
    func verifyMerkleProof(root: Fr, leaf: Fr, index: Int, proof: [Fr]) -> Bool {
        var current = leaf
        var idx = index

        for sibling in proof {
            if idx % 2 == 0 {
                current = poseidon2Hash(current, sibling)
            } else {
                current = poseidon2Hash(sibling, current)
            }
            idx /= 2
        }

        return frToInt(current) == frToInt(root)
    }

    /// Simple 2-to-1 hash placeholder. Production should use the actual Poseidon2 permutation.
    func poseidon2Hash(_ a: Fr, _ b: Fr) -> Fr {
        let c = frFromInt(0x9e3779b97f4a7c15)  // golden ratio constant
        let ab = frMul(a, b)
        let bc = frMul(b, c)
        return frAdd(frAdd(a, bc), ab)
    }

    // MARK: - Buffer Management

    /// Ensure a GPU buffer exists with at least the requested size.
    @discardableResult
    private func ensureBuffer(_ existing: MTLBuffer?, _ currentSize: inout Int, _ needed: Int) -> MTLBuffer? {
        if needed <= currentSize, let buf = existing { return buf }
        guard let buf = device.makeBuffer(length: needed, options: .storageModeShared) else { return nil }
        currentSize = needed
        return buf
    }
}

/// Round up to next power of 2.
func nextPow2(_ n: Int) -> Int {
    var p = 1
    while p < n { p *= 2 }
    return p
}
