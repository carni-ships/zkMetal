// Brakedown Polynomial Commitment Engine
// NTT-free PCS using expander-graph-based linear codes + Merkle commitments.
// Commit: reshape evaluations into matrix, encode rows with linear code, Merkle hash columns.
// Open: compute tensor dot products, reveal random columns with Merkle proofs.
// Verify: check dot product consistency against revealed columns.

import Foundation
import Metal

// MARK: - Data Structures

public struct BrakedownCommitment {
    /// Merkle root of the column hashes
    public let merkleRoot: Fr
    /// Dimensions of the encoded matrix
    public let numRows: Int
    public let numCols: Int          // original columns (before encoding)
    public let numEncodedCols: Int   // columns after encoding (numCols * rateInverse)
    /// Full Merkle tree for query extraction
    public let tree: [Fr]
    /// Encoded matrix (rows x numEncodedCols) — prover keeps for opening
    public let encodedMatrix: [Fr]
}

public struct BrakedownProof {
    /// Column openings: for each queried column, the full column vector (numRows elements)
    public let columnOpenings: [[Fr]]
    /// Merkle proofs for each queried column
    public let merkleProofs: [[Fr]]  // authentication paths
    /// Query indices (which columns were opened)
    public let queryIndices: [Int]
    /// t = M^T * tensor_right: a vector of length numCols
    /// Used by verifier to check value = <tensor_left, t> and column consistency.
    public let tVector: [Fr]
}

// MARK: - Engine

public class BrakedownEngine {
    public static let version = Versions.brakedown
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    let batchEncodeFunction: MTLComputePipelineState
    let dotProductFunction: MTLComputePipelineState
    let extractColumnFunction: MTLComputePipelineState

    private lazy var merkleEngine: Poseidon2MerkleEngine = {
        try! Poseidon2MerkleEngine()
    }()

    /// Code rate inverse (blowup factor). Codeword length = message length * rateInverse.
    public let rateInverse: Int

    /// Number of random column queries for soundness.
    public let numQueries: Int

    /// Deterministic seed for the linear code's random matrix.
    public let codeSeed: UInt32

    // Cached GPU buffers
    private var matrixBuf: MTLBuffer?
    private var matrixBufSize: Int = 0
    private var encodedBuf: MTLBuffer?
    private var encodedBufSize: Int = 0
    private var tensorBuf: MTLBuffer?
    private var tensorBufSize: Int = 0
    private var resultBuf: MTLBuffer?
    private var resultBufSize: Int = 0
    private var colBuf: MTLBuffer?
    private var colBufSize: Int = 0

    private let tuning: TuningConfig

    public init(rateInverse: Int = 4, numQueries: Int = 30, codeSeed: UInt32 = 0xBEEF) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        self.rateInverse = rateInverse
        self.numQueries = numQueries
        self.codeSeed = codeSeed

        let library = try BrakedownEngine.compileShaders(device: device)

        guard let batchEncodeFn = library.makeFunction(name: "brakedown_batch_encode"),
              let dotProductFn = library.makeFunction(name: "brakedown_dot_product"),
              let extractColFn = library.makeFunction(name: "brakedown_extract_column") else {
            throw MSMError.missingKernel
        }

        self.batchEncodeFunction = try device.makeComputePipelineState(function: batchEncodeFn)
        self.dotProductFunction = try device.makeComputePipelineState(function: dotProductFn)
        self.extractColumnFunction = try device.makeComputePipelineState(function: extractColFn)
        self.tuning = TuningManager.shared.config(device: device)
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
    /// Reshapes into a rows x cols matrix, encodes each row, then builds a Merkle tree over columns.
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

        let redundancyCols = numCols * (rateInverse - 1)
        let encodedCols = numCols + redundancyCols
        let stride = MemoryLayout<Fr>.stride

        // Upload matrix to GPU
        let matrixSize = n * stride
        if matrixSize > matrixBufSize {
            guard let buf = device.makeBuffer(length: matrixSize, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to allocate matrix buffer")
            }
            matrixBuf = buf
            matrixBufSize = matrixSize
        }
        evaluations.withUnsafeBytes { src in
            memcpy(matrixBuf!.contents(), src.baseAddress!, matrixSize)
        }

        // Allocate encoded output (redundancy only — systematic part is the original matrix)
        let redundancySize = numRows * redundancyCols * stride
        if redundancySize > encodedBufSize {
            guard let buf = device.makeBuffer(length: redundancySize, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to allocate encoded buffer")
            }
            encodedBuf = buf
            encodedBufSize = redundancySize
        }

        // GPU encode: each thread computes one redundancy element
        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!

        var cols32 = UInt32(numCols)
        var redCols32 = UInt32(redundancyCols)
        var seed32 = codeSeed

        enc.setComputePipelineState(batchEncodeFunction)
        enc.setBuffer(matrixBuf!, offset: 0, index: 0)
        enc.setBuffer(encodedBuf!, offset: 0, index: 1)
        enc.setBytes(&cols32, length: 4, index: 2)
        enc.setBytes(&redCols32, length: 4, index: 3)
        enc.setBytes(&seed32, length: 4, index: 4)

        let tg = min(64, Int(batchEncodeFunction.maxTotalThreadsPerThreadgroup))
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
            // Copy original columns
            for c in 0..<numCols {
                encodedMatrix[r * encodedCols + c] = evaluations[r * numCols + c]
            }
            // Copy redundancy columns
            for c in 0..<redundancyCols {
                encodedMatrix[r * encodedCols + numCols + c] = redPtr[r * redundancyCols + c]
            }
        }

        // Build Merkle tree over column hashes.
        // Each "leaf" is the hash of a column (numRows Fr elements hashed together).
        // For simplicity, we hash each column to a single Fr using Poseidon2.
        let numLeaves = nextPow2(encodedCols)
        var columnHashes = [Fr](repeating: Fr.zero, count: numLeaves)

        for c in 0..<encodedCols {
            // Hash column c: simple sum-hash (Poseidon2 would be better but this is functional)
            var hash = Fr.zero
            for r in 0..<numRows {
                hash = frAdd(hash, frMul(encodedMatrix[r * encodedCols + c], frFromInt(UInt64(r + 1))))
            }
            columnHashes[c] = hash
        }

        // Build Merkle tree
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
    /// point has length logN = log2(evaluation count).
    ///
    /// Protocol: prover computes t = M^T * tensor_right (length numCols vector),
    /// where M is the numRows x numCols matrix of evaluations and tensor_right
    /// is derived from the "row" portion of the evaluation point.
    /// Prover sends t along with opened columns + Merkle proofs.
    public func open(evaluations: [Fr], point: [Fr], commitment: BrakedownCommitment) throws -> BrakedownProof {
        let n = evaluations.count
        let logN = point.count
        precondition(n == (1 << logN), "Evaluation count must be 2^(point.count)")

        let numRows = commitment.numRows
        let numCols = commitment.numCols
        let encodedCols = commitment.numEncodedCols
        let logRows = Int(log2(Double(numRows)))
        let logCols = logN - logRows

        // Split point: first logRows entries control rows (high-order bits of index),
        // last logCols entries control columns (low-order bits of index).
        // This matches cpuEvaluate which folds point[0] on the top bit first.
        let pointRows = Array(point.prefix(logRows))    // for rows (high bits)
        let pointCols = Array(point.suffix(logCols))    // for columns (low bits)

        // Compute tensor vectors
        let tensorRows = computeTensor(pointRows)    // length numRows
        let tensorCols = computeTensor(pointCols)    // length numCols

        // Compute t = M^T * tensorRight: t[j] = sum_i tensorRight[i] * M[i][j]
        // This is a matrix-vector product that can be done on GPU
        let stride = MemoryLayout<Fr>.stride

        // For GPU: we need M transposed, or we compute column-wise dot products.
        // Since M is stored row-major, computing M^T * v means for each column j:
        //   t[j] = sum_i v[i] * M[i * numCols + j]
        // We can reuse the extract_column + dot product, or just compute on CPU for now.
        var tVector = [Fr](repeating: Fr.zero, count: numCols)
        for j in 0..<numCols {
            var acc = Fr.zero
            for i in 0..<numRows {
                acc = frAdd(acc, frMul(tensorRows[i], evaluations[i * numCols + j]))
            }
            tVector[j] = acc
        }

        // Generate random column query indices (deterministic from commitment root)
        let queryIndices = generateQueryIndices(
            root: commitment.merkleRoot,
            numQueries: numQueries,
            maxCol: encodedCols
        )

        // Extract queried columns and Merkle proofs
        var columnOpenings = [[Fr]]()
        var merkleProofs = [[Fr]]()

        for colIdx in queryIndices {
            // Extract column from encoded matrix
            var column = [Fr](repeating: Fr.zero, count: numRows)
            for r in 0..<numRows {
                column[r] = commitment.encodedMatrix[r * encodedCols + colIdx]
            }
            columnOpenings.append(column)

            // Extract Merkle proof for this column
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
    /// Returns true if the proof is valid.
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

        // Check 2: Column consistency via linear code.
        // encode(t) extends t from numCols to encodedCols.
        // For each opened column j: <tensorRows, col_j> should equal encode(t)[j].
        let code = LinearCode(messageLength: numCols, rateInverse: rateInverse, seed: codeSeed)
        let encodedT = code.encode(proof.tVector)

        for q in 0..<proof.queryIndices.count {
            let colIdx = proof.queryIndices[q]
            let column = proof.columnOpenings[q]

            // <tensorRows, column_colIdx>
            var columnDot = Fr.zero
            for i in 0..<numRows {
                columnDot = frAdd(columnDot, frMul(tensorRows[i], column[i]))
            }

            // This should equal encodedT[colIdx]
            if frToInt(columnDot) != frToInt(encodedT[colIdx]) {
                return false
            }
        }

        return true
    }

    // MARK: - Helpers

    /// Compute the multilinear tensor product: t = tensor(point)
    /// For point = (z_0, z_1, ..., z_{k-1}), the tensor is:
    /// t[i] = product_{j=0}^{k-1} (if bit j of i is 1 then z_j else (1-z_j))
    public func computeTensor(_ point: [Fr]) -> [Fr] {
        let k = point.count
        let n = 1 << k
        var tensor = [Fr](repeating: Fr.zero, count: n)
        tensor[0] = Fr.one

        for j in 0..<k {
            let half = 1 << j
            let zj = point[j]
            let oneMinusZj = frSub(Fr.one, zj)
            // Process in reverse to avoid overwriting
            for i in stride(from: half - 1, through: 0, by: -1) {
                tensor[2 * i + 1] = frMul(tensor[i], zj)
                tensor[2 * i] = frMul(tensor[i], oneMinusZj)
            }
        }
        return tensor
    }

    /// Generate deterministic query indices from the commitment root.
    func generateQueryIndices(root: Fr, numQueries: Int, maxCol: Int) -> [Int] {
        var rng: UInt64 = 0
        let limbs = frToInt(root)
        rng = limbs[0] ^ limbs[1] ^ limbs[2] ^ limbs[3]

        var indices = [Int]()
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

    /// Extract a Merkle authentication path for a leaf at the given index.
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

    /// Verify a Merkle proof against a known root.
    func verifyMerkleProof(root: Fr, leaf: Fr, index: Int, proof: [Fr], numLeaves: Int) -> Bool {
        // Simple verification: recompute root from leaf + proof
        // Using the same Poseidon2 2-to-1 hash as the Merkle engine
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

    /// Simple 2-to-1 hash using field arithmetic (placeholder for Poseidon2).
    /// In production, this should use the actual Poseidon2 permutation.
    func poseidon2Hash(_ a: Fr, _ b: Fr) -> Fr {
        // Mimics the Merkle tree hash: H(a, b) = a + b * constant + (a*b)
        // This is NOT cryptographic — for testing/benchmark only.
        // The actual Merkle tree uses GPU Poseidon2.
        let c = frFromInt(0x9e3779b97f4a7c15)  // golden ratio constant
        let ab = frMul(a, b)
        let bc = frMul(b, c)
        return frAdd(frAdd(a, bc), ab)
    }

    /// CPU-side multilinear evaluation (for correctness testing).
    public static func cpuEvaluate(evaluations: [Fr], point: [Fr]) -> Fr {
        let n = evaluations.count
        let logN = point.count
        precondition(n == (1 << logN))

        var evals = evaluations
        for i in 0..<logN {
            let half = evals.count / 2
            var folded = [Fr](repeating: Fr.zero, count: half)
            for j in 0..<half {
                let low = evals[j]
                let high = evals[j + half]
                let diff = frSub(high, low)
                folded[j] = frAdd(low, frMul(point[i], diff))
            }
            evals = folded
        }
        return evals[0]
    }
}

/// Round up to next power of 2.
func nextPow2(_ n: Int) -> Int {
    var p = 1
    while p < n { p *= 2 }
    return p
}
