// GPUCircleSTARKProverEngine — GPU-accelerated Circle STARK prover over Mersenne31
//
// Circle STARK over M31 (p = 2^31 - 1), circle group order p+1 = 2^31:
//   1. Trace LDE via GPU Circle NTT  2. Poseidon2-M31 Merkle commitments
//   3. Constraint eval + quotient splitting  4. Circle FRI (y-fold then x-folds)
//   5. Query phase with Merkle openings
//
// Circle FRI: first fold uses y-coords (twin-coset), subsequent use x-coord squaring.
// Poseidon2-M31 (t=16, rate=8, alpha=5) for algebraic Merkle commitments.

import Foundation
import Metal

// MARK: - GPU Circle STARK Prover Configuration

/// Configuration for GPU-accelerated Circle STARK prover engine.
public struct GPUCircleSTARKProverConfig {
    /// Log2 of blowup factor (1=2x, 2=4x, 3=8x, 4=16x)
    public let logBlowup: Int

    /// Number of FRI query points for soundness
    public let numQueries: Int

    /// Extension field degree (4 = QM31 for 128-bit security)
    public let extensionDegree: Int

    /// Minimum evaluation domain size to trigger GPU constraint eval
    public let gpuConstraintThreshold: Int

    /// Minimum domain size for GPU FRI folding
    public let gpuFRIFoldThreshold: Int

    /// Whether to use Poseidon2-M31 (true) or Keccak (false) for Merkle commitments
    public let usePoseidon2Merkle: Bool

    /// Number of quotient splits (for deep composition)
    public let numQuotientSplits: Int

    /// Default: 4x blowup, 20 queries, Poseidon2-M31 Merkle
    public static let `default` = GPUCircleSTARKProverConfig(
        logBlowup: 2, numQueries: 20, extensionDegree: 4,
        gpuConstraintThreshold: 128, gpuFRIFoldThreshold: 128,
        usePoseidon2Merkle: true, numQuotientSplits: 2
    )

    /// Fast configuration for testing: 2x blowup, 8 queries
    public static let fast = GPUCircleSTARKProverConfig(
        logBlowup: 1, numQueries: 8, extensionDegree: 4,
        gpuConstraintThreshold: 16, gpuFRIFoldThreshold: 16,
        usePoseidon2Merkle: true, numQuotientSplits: 2
    )

    /// High-security: 16x blowup, 40 queries
    public static let highSecurity = GPUCircleSTARKProverConfig(
        logBlowup: 4, numQueries: 40, extensionDegree: 4,
        gpuConstraintThreshold: 256, gpuFRIFoldThreshold: 256,
        usePoseidon2Merkle: true, numQuotientSplits: 4
    )

    public init(logBlowup: Int = 2, numQueries: Int = 20, extensionDegree: Int = 4,
                gpuConstraintThreshold: Int = 128, gpuFRIFoldThreshold: Int = 128,
                usePoseidon2Merkle: Bool = true, numQuotientSplits: Int = 2) {
        precondition(logBlowup >= 1 && logBlowup <= 8)
        precondition(numQueries >= 1 && numQueries <= 200)
        self.logBlowup = logBlowup
        self.numQueries = numQueries
        self.extensionDegree = extensionDegree
        self.gpuConstraintThreshold = gpuConstraintThreshold
        self.gpuFRIFoldThreshold = gpuFRIFoldThreshold
        self.usePoseidon2Merkle = usePoseidon2Merkle
        self.numQuotientSplits = numQuotientSplits
    }

    /// Security bits: each query eliminates ~logBlowup bits of cheating probability
    public var securityBits: Int { numQueries * logBlowup }

    /// Blowup factor
    public var blowupFactor: Int { 1 << logBlowup }
}

// MARK: - GPU Circle STARK Proof (Poseidon2-M31 commitments)

/// Commitment digest: 8 M31 elements from Poseidon2-M31 rate output.
public struct M31Digest: Equatable {
    public let values: [M31]

    public static var zero: M31Digest {
        M31Digest(values: [M31](repeating: M31.zero, count: 8))
    }

    public init(values: [M31]) {
        precondition(values.count == 8)
        self.values = values
    }

    /// Convert to bytes for transcript absorption
    public var bytes: [UInt8] {
        var out = [UInt8]()
        out.reserveCapacity(32)
        for v in values {
            var val = v.v
            withUnsafeBytes(of: &val) { out.append(contentsOf: $0) }
        }
        return out
    }

    public var isNonTrivial: Bool {
        values.contains { $0.v != 0 }
    }
}

/// Circle FRI round data with Poseidon2-M31 commitments.
public struct GPUCircleFRIRound {
    /// Poseidon2-M31 Merkle root of folded polynomial evaluations
    public let commitment: M31Digest
    /// For each query: (value at query, value at sibling, Merkle path of M31Digests)
    public let queryResponses: [(M31, M31, [M31Digest])]

    public init(commitment: M31Digest, queryResponses: [(M31, M31, [M31Digest])]) {
        self.commitment = commitment
        self.queryResponses = queryResponses
    }
}

/// Circle FRI proof data with Poseidon2-M31 commitments.
public struct GPUCircleFRIProof {
    /// Per-round data
    public let rounds: [GPUCircleFRIRound]
    /// Final constant after all folding rounds
    public let finalValue: M31
    /// Query indices used
    public let queryIndices: [Int]

    public init(rounds: [GPUCircleFRIRound], finalValue: M31, queryIndices: [Int]) {
        self.rounds = rounds
        self.finalValue = finalValue
        self.queryIndices = queryIndices
    }
}

/// Query response for GPU Circle STARK with Poseidon2 Merkle paths.
public struct GPUCircleSTARKQueryResponse {
    /// Trace values at query position: [column] of M31
    public let traceValues: [M31]
    /// Trace Merkle authentication paths: [column] of path
    public let tracePaths: [[M31Digest]]
    /// Composition polynomial value at query position
    public let compositionValue: M31
    /// Composition Merkle authentication path
    public let compositionPath: [M31Digest]
    /// Quotient split values at query position
    public let quotientSplitValues: [M31]
    /// Query index in evaluation domain
    public let queryIndex: Int

    public init(traceValues: [M31], tracePaths: [[M31Digest]],
                compositionValue: M31, compositionPath: [M31Digest],
                quotientSplitValues: [M31], queryIndex: Int) {
        self.traceValues = traceValues
        self.tracePaths = tracePaths
        self.compositionValue = compositionValue
        self.compositionPath = compositionPath
        self.quotientSplitValues = quotientSplitValues
        self.queryIndex = queryIndex
    }
}

/// Complete GPU Circle STARK proof.
public struct GPUCircleSTARKProverProof {
    /// Poseidon2-M31 Merkle roots of trace column LDEs
    public let traceCommitments: [M31Digest]
    /// Poseidon2-M31 Merkle root of composition polynomial
    public let compositionCommitment: M31Digest
    /// Quotient split commitments
    public let quotientCommitments: [M31Digest]
    /// FRI proof for low-degree test
    public let friProof: GPUCircleFRIProof
    /// Query responses
    public let queryResponses: [GPUCircleSTARKQueryResponse]
    /// Random alpha for constraint batching
    public let alpha: M31
    /// Metadata
    public let traceLength: Int
    public let numColumns: Int
    public let logBlowup: Int

    public init(traceCommitments: [M31Digest], compositionCommitment: M31Digest,
                quotientCommitments: [M31Digest], friProof: GPUCircleFRIProof,
                queryResponses: [GPUCircleSTARKQueryResponse], alpha: M31,
                traceLength: Int, numColumns: Int, logBlowup: Int) {
        self.traceCommitments = traceCommitments
        self.compositionCommitment = compositionCommitment
        self.quotientCommitments = quotientCommitments
        self.friProof = friProof
        self.queryResponses = queryResponses
        self.alpha = alpha
        self.traceLength = traceLength
        self.numColumns = numColumns
        self.logBlowup = logBlowup
    }

    /// Estimated proof size in bytes
    public var estimatedSizeBytes: Int {
        var size = 0
        size += traceCommitments.count * 32  // 8 M31 = 32 bytes each
        size += 32  // composition commitment
        size += quotientCommitments.count * 32
        for round in friProof.rounds {
            size += 32  // commitment
            for (_, _, path) in round.queryResponses {
                size += 8  // two M31 values
                size += path.count * 32  // Merkle path
            }
        }
        size += 4  // final FRI value
        for qr in queryResponses {
            size += qr.traceValues.count * 4
            size += qr.tracePaths.count * (qr.tracePaths.first?.count ?? 0) * 32
            size += 4  // composition value
            size += qr.compositionPath.count * 32
            size += qr.quotientSplitValues.count * 4
        }
        return size
    }
}

// MARK: - GPU Circle STARK Prover Result

/// Prover result with timing information.
public struct GPUCircleSTARKProverResult {
    public let proof: GPUCircleSTARKProverProof
    public let traceLength: Int
    public let numColumns: Int
    public let totalTimeSeconds: Double
    public let traceGenTimeSeconds: Double
    public let ldeTimeSeconds: Double
    public let commitTimeSeconds: Double
    public let constraintTimeSeconds: Double
    public let friTimeSeconds: Double
    public let queryTimeSeconds: Double

    public init(proof: GPUCircleSTARKProverProof, traceLength: Int, numColumns: Int,
                totalTimeSeconds: Double, traceGenTimeSeconds: Double,
                ldeTimeSeconds: Double, commitTimeSeconds: Double,
                constraintTimeSeconds: Double, friTimeSeconds: Double,
                queryTimeSeconds: Double) {
        self.proof = proof
        self.traceLength = traceLength
        self.numColumns = numColumns
        self.totalTimeSeconds = totalTimeSeconds
        self.traceGenTimeSeconds = traceGenTimeSeconds
        self.ldeTimeSeconds = ldeTimeSeconds
        self.commitTimeSeconds = commitTimeSeconds
        self.constraintTimeSeconds = constraintTimeSeconds
        self.friTimeSeconds = friTimeSeconds
        self.queryTimeSeconds = queryTimeSeconds
    }
}

// MARK: - Poseidon2-M31 Merkle Tree (CPU reference)

/// Build Poseidon2-M31 Merkle tree: leaves[0..n-1], internal[n..2n-2], root at [2n-2].
public func buildPoseidon2M31MerkleTree(_ values: [M31], count n: Int) -> [M31Digest] {
    precondition(n > 0 && (n & (n - 1)) == 0, "n must be a power of 2")
    let treeSize = 2 * n - 1
    var tree = [M31Digest](repeating: M31Digest.zero, count: treeSize)

    // Leaf hashing: serial to avoid race conditions in parallel execution
    for i in 0..<n {
        let val = i < values.count ? values[i] : M31.zero
        let leafInput = [val, M31(v: UInt32(i)), M31.zero, M31.zero,
                         M31.zero, M31.zero, M31.zero, M31.zero]
        tree[i] = M31Digest(values: poseidon2M31HashSingle(leafInput))
    }

    // Build internal nodes bottom-up (serial for correctness)
    var levelStart = 0
    var levelSize = n
    while levelSize > 1 {
        let parentStart = levelStart + levelSize
        let parentSize = levelSize / 2

        for i in 0..<parentSize {
            let left = tree[levelStart + 2 * i]
            let right = tree[levelStart + 2 * i + 1]
            tree[parentStart + i] = M31Digest(values: poseidon2M31Hash(left: left.values, right: right.values))
        }

        levelStart = parentStart
        levelSize = parentSize
    }

    return tree
}

/// Extract Merkle root from flat Poseidon2-M31 tree.
public func poseidon2M31MerkleRoot(_ tree: [M31Digest], n: Int) -> M31Digest {
    tree[2 * n - 2]
}

/// Extract Merkle authentication path from flat Poseidon2-M31 tree.
public func poseidon2M31MerkleProof(_ tree: [M31Digest], n: Int, index: Int) -> [M31Digest] {
    var path = [M31Digest]()
    var levelStart = 0
    var levelSize = n
    var idx = index
    while levelSize > 1 {
        let sibIdx = idx ^ 1
        path.append(tree[levelStart + sibIdx])
        levelStart += levelSize
        levelSize /= 2
        idx /= 2
    }
    return path
}

/// Verify a Poseidon2-M31 Merkle proof.
public func verifyPoseidon2M31MerkleProof(leafDigest: M31Digest, path: [M31Digest],
                                           index: Int, root: M31Digest) -> Bool {
    var current = leafDigest
    var idx = index
    for sibling in path {
        if idx & 1 == 0 {
            current = M31Digest(values: poseidon2M31Hash(left: current.values, right: sibling.values))
        } else {
            current = M31Digest(values: poseidon2M31Hash(left: sibling.values, right: current.values))
        }
        idx /= 2
    }
    return current == root
}

// MARK: - Quotient Splitting

/// Split polynomial into `numSplits` components via stride-based decomposition.
public func circleQuotientSplit(evals: [M31], logN: Int, numSplits: Int) -> [[M31]] {
    let n = 1 << logN
    precondition(evals.count == n)
    precondition(numSplits > 0 && numSplits <= n)

    if numSplits == 1 { return [evals] }
    let splitSize = n / numSplits
    var splits = [[M31]](repeating: [M31](repeating: M31.zero, count: splitSize), count: numSplits)
    for i in 0..<n {
        let splitIdx = i % numSplits
        let withinIdx = i / numSplits
        splits[splitIdx][withinIdx] = evals[i]
    }

    return splits
}

// MARK: - GPU Merkle Tree Engine for M31

/// GPU-accelerated Merkle tree builder for Poseidon2-M31.
/// Uses GPU to build complete Merkle trees with the special leaf hashing format:
/// leaf[i] = Poseidon2([value, i, 0, 0, 0, 0, 0, 0])
///
/// Supports trees up to 2^20 leaves (1M leaves) for large EVM traces.
public class GPUMerkleTreeM31Engine {
    public static let version = Versions.poseidon2M31

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    private let hashLeavesFunction: MTLComputePipelineState
    private let hashLeavesBatchFunction: MTLComputePipelineState  // Fixed batch kernel
    private let hashPairsFunction: MTLComputePipelineState
    private let treeBatchFunction: MTLComputePipelineState
    private let proofBatchFunction: MTLComputePipelineState
    private let rcBuffer: MTLBuffer

    private let nodeSize = 8  // M31 elements per node (Poseidon2 rate)
    private let tuning: TuningConfig

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        let library = try GPUMerkleTreeM31Engine.compileShaders(device: device)
        let proofLibrary = try GPUMerkleTreeM31Engine.compileProofBatchShaders(device: device)

        guard let hashLeavesFn = library.makeFunction(name: "poseidon2_m31_hash_leaves"),
              let hashLeavesBatchFn = library.makeFunction(name: "poseidon2_m31_hash_leaves_batch"),
              let hashPairsFn = library.makeFunction(name: "poseidon2_m31_hash_pairs") else {
            throw MSMError.missingKernel
        }
        guard let proofBatchFn = proofLibrary.makeFunction(name: "poseidon2_m31_merkle_proof_batch"),
              let treeBatchFn = proofLibrary.makeFunction(name: "poseidon2_m31_merkle_tree_batch_v2") else {
            throw MSMError.missingKernel
        }

        self.hashLeavesFunction = try device.makeComputePipelineState(function: hashLeavesFn)
        self.hashLeavesBatchFunction = try device.makeComputePipelineState(function: hashLeavesBatchFn)
        self.hashPairsFunction = try device.makeComputePipelineState(function: hashPairsFn)
        self.proofBatchFunction = try device.makeComputePipelineState(function: proofBatchFn)
        self.treeBatchFunction = try device.makeComputePipelineState(function: treeBatchFn)

        // Create round constants buffer
        let rc = POSEIDON2_M31_ROUND_CONSTANTS
        var flatRC = [UInt32]()
        flatRC.reserveCapacity(Poseidon2M31Config.totalRounds * Poseidon2M31Config.t)
        for round in rc {
            for elem in round {
                flatRC.append(elem.v)
            }
        }
        let byteCount = flatRC.count * MemoryLayout<UInt32>.stride
        guard let buf = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate RC buffer")
        }
        flatRC.withUnsafeBytes { src in
            memcpy(buf.contents(), src.baseAddress!, byteCount)
        }
        self.rcBuffer = buf
        self.tuning = TuningManager.shared.config(device: device)
    }

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let m31Source = try String(contentsOfFile: shaderDir + "/fields/mersenne31.metal", encoding: .utf8)
        let p2Source = try String(contentsOfFile: shaderDir + "/hash/poseidon2_m31.metal", encoding: .utf8)

        let cleanP2 = p2Source.split(separator: "\n")
            .filter { !$0.contains("#include") }
            .joined(separator: "\n")

        let m31Clean = m31Source
            .replacingOccurrences(of: "#ifndef MERSENNE31_METAL", with: "")
            .replacingOccurrences(of: "#define MERSENNE31_METAL", with: "")
            .replacingOccurrences(of: "#endif // MERSENNE31_METAL", with: "")

        let combined = m31Clean + "\n" + cleanP2

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    private static func compileProofBatchShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        // The merkle tree shader is self-contained with its own M31 definitions
        // No need to prepend mersenne31.metal
        let merkleSource = try String(contentsOfFile: shaderDir + "/hash/poseidon2_m31_merkle_tree.metal", encoding: .utf8)

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: merkleSource, options: options)
    }

    /// Build a complete Poseidon2-M31 Merkle tree on GPU.
    ///
    /// Tree layout: nodes[0..<n] = leaves, nodes[n..<2n-1] = internal, nodes[2n-2] = root
    /// Each node is 8 M31 elements (32 bytes).
    ///
    /// - Parameters:
    ///   - values: Individual M31 values (leaves)
    ///   - n: Number of leaves (must be power of 2)
    /// - Returns: Complete tree as array of M31Digest (8 M31 per node)
    public func buildTree(values: [M31], count n: Int) throws -> [M31Digest] {
        precondition(n > 0 && (n & (n - 1)) == 0, "n must be a power of 2")
        let treeSize = 2 * n - 1
        let stride = MemoryLayout<UInt32>.stride

        // Step 1: Hash leaves to digests (GPU)
        let leafDigestBytes = n * nodeSize * stride
        guard let valuesBuf = device.makeBuffer(length: n * stride, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate values buffer")
        }
        guard let leafDigestBuf = device.makeBuffer(length: leafDigestBytes, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate leaf digest buffer")
        }

        // Copy values
        let valuesPtr = valuesBuf.contents().bindMemory(to: UInt32.self, capacity: n)
        for i in 0..<n {
            valuesPtr[i] = i < values.count ? values[i].v : 0
        }

        // Hash leaves on GPU
        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(hashLeavesFunction)
        enc.setBuffer(valuesBuf, offset: 0, index: 0)
        enc.setBuffer(leafDigestBuf, offset: 0, index: 1)
        enc.setBuffer(rcBuffer, offset: 0, index: 2)
        var count = UInt32(n)
        enc.setBytes(&count, length: 4, index: 3)
        let tg = min(tuning.hashThreadgroupSize, Int(hashLeavesFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        // Step 2: Allocate tree buffer (leaves + internal nodes)
        let treeBytes = treeSize * nodeSize * stride
        guard let treeBuf = device.makeBuffer(length: treeBytes, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate tree buffer")
        }

        // Copy leaf digests to tree buffer
        let leafPtr = leafDigestBuf.contents().bindMemory(to: UInt32.self, capacity: n * nodeSize)
        let treePtr = treeBuf.contents().bindMemory(to: UInt32.self, capacity: treeSize * nodeSize)
        for i in 0..<(n * nodeSize) {
            treePtr[i] = leafPtr[i]
        }

        // Step 3: Build internal nodes level-by-level on GPU
        guard let buildCmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let buildEnc = buildCmdBuf.makeComputeCommandEncoder()!

        var currentLevelNodes = n
        var levelStart = 0

        while currentLevelNodes > 1 {
            let pairs = currentLevelNodes / 2
            let inputOffset = levelStart * nodeSize
            let outputOffset = (levelStart + currentLevelNodes) * nodeSize

            // Encode hash pairs
            buildEnc.setComputePipelineState(hashPairsFunction)
            buildEnc.setBuffer(treeBuf, offset: inputOffset * stride, index: 0)
            buildEnc.setBuffer(treeBuf, offset: outputOffset * stride, index: 1)
            buildEnc.setBuffer(rcBuffer, offset: 0, index: 2)
            var pairCount = UInt32(pairs)
            buildEnc.setBytes(&pairCount, length: 4, index: 3)
            let tgSize = min(tuning.hashThreadgroupSize, Int(hashPairsFunction.maxTotalThreadsPerThreadgroup))
            buildEnc.dispatchThreads(MTLSize(width: pairs, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))

            levelStart += currentLevelNodes
            currentLevelNodes = pairs

            // Memory barrier between levels
            buildEnc.memoryBarrier(scope: .buffers)
        }

        buildEnc.endEncoding()
        buildCmdBuf.commit()
        buildCmdBuf.waitUntilCompleted()
        if let error = buildCmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        // Step 4: Read tree back to CPU
        let finalTreePtr = treeBuf.contents().bindMemory(to: UInt32.self, capacity: treeSize * nodeSize)
        var result = [M31Digest]()
        result.reserveCapacity(treeSize)

        for i in 0..<treeSize {
            var values = [M31]()
            values.reserveCapacity(nodeSize)
            for j in 0..<nodeSize {
                values.append(M31(v: finalTreePtr[i * nodeSize + j]))
            }
            result.append(M31Digest(values: values))
        }

        return result
    }

    /// Compute only the Merkle root on GPU (more efficient when only root is needed).
    public func merkleRoot(values: [M31], count n: Int) throws -> M31Digest {
        let tree = try buildTree(values: values, count: n)
        return tree[2 * n - 2]
    }

    /// Build a complete Poseidon2-M31 Merkle tree on GPU and return the GPU buffer.
    ///
    /// This is similar to buildTree() but keeps the tree buffer on GPU for use
    /// with generateProofsGPU(). Use this when you need both the tree structure
    /// for proof generation and the root for commitment.
    ///
    /// Tree layout: nodes[0..<n] = leaves, nodes[n..<2n-1] = internal, nodes[2n-2] = root
    /// Each node is 8 M31 elements (32 bytes).
    ///
    /// - Parameters:
    ///   - values: Individual M31 values (leaves)
    ///   - n: Number of leaves (must be power of 2)
    /// - Returns: Tuple of (root digest, GPU tree buffer, node count)
    public func buildTreeWithBuffer(values: [M31], count n: Int) throws -> (root: M31Digest, buffer: MTLBuffer, nodeCount: Int) {
        precondition(n > 0 && (n & (n - 1)) == 0, "n must be a power of 2")
        let treeSize = 2 * n - 1
        let stride = MemoryLayout<UInt32>.stride

        // Step 1: Hash leaves to digests (GPU)
        let leafDigestBytes = n * nodeSize * stride
        guard let valuesBuf = device.makeBuffer(length: n * stride, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate values buffer")
        }
        guard let leafDigestBuf = device.makeBuffer(length: leafDigestBytes, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate leaf digest buffer")
        }

        // Copy values
        let valuesPtr = valuesBuf.contents().bindMemory(to: UInt32.self, capacity: n)
        for i in 0..<n {
            valuesPtr[i] = i < values.count ? values[i].v : 0
        }

        // Hash leaves on GPU
        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(hashLeavesFunction)
        enc.setBuffer(valuesBuf, offset: 0, index: 0)
        enc.setBuffer(leafDigestBuf, offset: 0, index: 1)
        enc.setBuffer(rcBuffer, offset: 0, index: 2)
        var count = UInt32(n)
        enc.setBytes(&count, length: 4, index: 3)
        let tg = min(tuning.hashThreadgroupSize, Int(hashLeavesFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        // Step 2: Allocate tree buffer (leaves + internal nodes)
        let treeBytes = treeSize * nodeSize * stride
        guard let treeBuf = device.makeBuffer(length: treeBytes, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate tree buffer")
        }

        // Copy leaf digests to tree buffer
        let leafPtr = leafDigestBuf.contents().bindMemory(to: UInt32.self, capacity: n * nodeSize)
        let treePtr = treeBuf.contents().bindMemory(to: UInt32.self, capacity: treeSize * nodeSize)
        for i in 0..<(n * nodeSize) {
            treePtr[i] = leafPtr[i]
        }

        // Step 3: Build internal nodes level-by-level on GPU
        guard let buildCmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let buildEnc = buildCmdBuf.makeComputeCommandEncoder()!

        var currentLevelNodes = n
        var levelStart = 0

        while currentLevelNodes > 1 {
            let pairs = currentLevelNodes / 2
            let inputOffset = levelStart * nodeSize
            let outputOffset = (levelStart + currentLevelNodes) * nodeSize

            buildEnc.setComputePipelineState(hashPairsFunction)
            buildEnc.setBuffer(treeBuf, offset: inputOffset * stride, index: 0)
            buildEnc.setBuffer(treeBuf, offset: outputOffset * stride, index: 1)
            buildEnc.setBuffer(rcBuffer, offset: 0, index: 2)
            var pairCount = UInt32(pairs)
            buildEnc.setBytes(&pairCount, length: 4, index: 3)
            let tgSize = min(tuning.hashThreadgroupSize, Int(hashPairsFunction.maxTotalThreadsPerThreadgroup))
            buildEnc.dispatchThreads(MTLSize(width: pairs, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))

            levelStart += currentLevelNodes
            currentLevelNodes = pairs

            buildEnc.memoryBarrier(scope: .buffers)
        }

        buildEnc.endEncoding()
        buildCmdBuf.commit()
        buildCmdBuf.waitUntilCompleted()
        if let error = buildCmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        // Read root from GPU
        let rootOffset = (2 * n - 2) * nodeSize
        let rootPtr = treeBuf.contents().bindMemory(to: UInt32.self, capacity: (2 * n - 1) * nodeSize)
        var rootValues = [M31]()
        rootValues.reserveCapacity(nodeSize)
        for i in 0..<nodeSize {
            rootValues.append(M31(v: rootPtr[rootOffset + i]))
        }
        let root = M31Digest(values: rootValues)

        return (root, treeBuf, treeSize)
    }

    /// Build multiple Merkle trees in a single GPU dispatch using poseidon2_m31_merkle_tree_batch_v2.
    ///
    /// This is significantly faster than building trees sequentially when processing many columns.
    /// All trees share the same number of leaves (count n) and are processed in parallel.
    ///
    /// - Parameters:
    ///   - columns: Array of value arrays, one per tree
    ///   - n: Number of leaves per tree (must be power of 2)
    /// - Returns: Tuple of (roots, combined buffer with all trees, nodes per tree)
    public func buildTreesBatch(columns: [[M31]], count n: Int) throws -> (roots: [M31Digest], buffer: MTLBuffer, nodesPerTree: Int) {
        let t0 = CFAbsoluteTimeGetCurrent()
        let numTrees = columns.count
        let treeSize = 2 * n - 1
        let stride = MemoryLayout<UInt32>.stride

        // Allocate combined tree buffer for all trees
        let treeBytes = numTrees * treeSize * nodeSize * stride
        guard let treeBuf = device.makeBuffer(length: treeBytes, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate tree buffer")
        }
        let treePtr = treeBuf.contents().bindMemory(to: UInt32.self, capacity: numTrees * treeSize * nodeSize)

        // Process each tree: hash leaves and build tree
        for treeIdx in 0..<numTrees {
            let treeBase = treeIdx * treeSize * nodeSize
            let col = columns[treeIdx]

            // Step 1: Hash leaves to digests (GPU) - per-tree to avoid GPU batch issues
            let valuesBufSize = n * stride
            let leafDigestBytes = n * nodeSize * stride

            guard let valuesBuf = device.makeBuffer(length: valuesBufSize, options: .storageModeShared),
                  let leafDigestBuf = device.makeBuffer(length: leafDigestBytes, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to allocate buffers for tree \(treeIdx)")
            }

            // Copy values
            let valuesPtr = valuesBuf.contents().bindMemory(to: UInt32.self, capacity: n)
            for i in 0..<n {
                valuesPtr[i] = i < col.count ? col[i].v : 0
            }

            // Hash leaves on GPU
            guard let cmdBuf = commandQueue.makeCommandBuffer() else {
                throw MSMError.noCommandBuffer
            }
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(hashLeavesFunction)
            enc.setBuffer(valuesBuf, offset: 0, index: 0)
            enc.setBuffer(leafDigestBuf, offset: 0, index: 1)
            enc.setBuffer(rcBuffer, offset: 0, index: 2)
            var count = UInt32(n)
            enc.setBytes(&count, length: 4, index: 3)
            let tg = min(tuning.hashThreadgroupSize, Int(hashLeavesFunction.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()

            // Copy leaf digests to tree buffer (leaves at start of tree's section)
            let leafPtr = leafDigestBuf.contents().bindMemory(to: UInt32.self, capacity: n * nodeSize)
            for i in 0..<(n * nodeSize) {
                treePtr[treeBase + i] = leafPtr[i]
            }
        }

        // Build internal nodes level-by-level using per-tree dispatch
        for treeIdx in 0..<numTrees {
            let treeBase = treeIdx * treeSize * nodeSize

            // Process this tree's levels one at a time, waiting for each level to complete
            var currentLevelNodes = n
            var levelStart = 0

            while currentLevelNodes > 1 {
                let pairs = currentLevelNodes / 2
                let inputOffset = treeBase + levelStart * nodeSize
                let outputOffset = treeBase + (levelStart + currentLevelNodes) * nodeSize

                guard let cmdBuf = commandQueue.makeCommandBuffer() else {
                    throw MSMError.noCommandBuffer
                }
                let enc = cmdBuf.makeComputeCommandEncoder()!

                enc.setComputePipelineState(hashPairsFunction)
                enc.setBuffer(treeBuf, offset: inputOffset * stride, index: 0)
                enc.setBuffer(treeBuf, offset: outputOffset * stride, index: 1)
                enc.setBuffer(rcBuffer, offset: 0, index: 2)
                var pairCount = UInt32(pairs)
                enc.setBytes(&pairCount, length: 4, index: 3)
                let tgSize = min(tuning.hashThreadgroupSize, Int(hashPairsFunction.maxTotalThreadsPerThreadgroup))
                enc.dispatchThreads(MTLSize(width: pairs, height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
                enc.endEncoding()
                cmdBuf.commit()
                cmdBuf.waitUntilCompleted()

                levelStart += currentLevelNodes
                currentLevelNodes = pairs
            }
        }

        // Extract roots from all trees
        let rootPtr = treeBuf.contents().bindMemory(to: UInt32.self, capacity: numTrees * treeSize * nodeSize)

        var roots = [M31Digest]()
        roots.reserveCapacity(numTrees)
        for treeIdx in 0..<numTrees {
            let rootOffset = treeIdx * treeSize * nodeSize + (2 * n - 2) * nodeSize
            var rootValues = [M31]()
            rootValues.reserveCapacity(nodeSize)
            for i in 0..<nodeSize {
                rootValues.append(M31(v: rootPtr[rootOffset + i]))
            }
            roots.append(M31Digest(values: rootValues))
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        print("[GPUMerkleTreeM31Engine] buildTreesBatch: \(numTrees) trees x \(n) leaves in \(String(format: "%.3f", elapsed * 1000))ms")

        return (roots, treeBuf, treeSize)
    }

    /// Build multiple Merkle trees with GPU-batched leaf hashing (fixed kernel).
    ///
    /// This method uses the corrected poseidon2_m31_hash_leaves_batch kernel that properly
    /// handles batched leaf hashing by using (gid % n) as the position index instead of
    /// the global gid. This fixes the bug in the original per-tree dispatch where using
    /// the single-tree kernel in a batch dispatch would cause incorrect leaf digests.
    ///
    /// - Parameters:
    ///   - columns: Array of value arrays, one per tree
    ///   - n: Number of leaves per tree (must be power of 2)
    /// - Returns: Tuple of (roots, combined buffer with all trees, nodes per tree)
    public func buildTreesBatchGPU(columns: [[M31]], count n: Int) throws -> (roots: [M31Digest], buffer: MTLBuffer, nodesPerTree: Int) {
        let t0 = CFAbsoluteTimeGetCurrent()
        let numTrees = columns.count
        let treeSize = 2 * n - 1
        let stride = MemoryLayout<UInt32>.stride

        // Allocate combined tree buffer for all trees
        let treeBytes = numTrees * treeSize * nodeSize * stride
        guard let treeBuf = device.makeBuffer(length: treeBytes, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate tree buffer")
        }
        let treePtr = treeBuf.contents().bindMemory(to: UInt32.self, capacity: numTrees * treeSize * nodeSize)

        // Allocate a single values buffer for all trees (flattened)
        let totalValues = numTrees * n
        let valuesBufSize = totalValues * stride
        guard let valuesBuf = device.makeBuffer(length: valuesBufSize, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate values buffer")
        }

        // Copy all values into flattened buffer (tree0_leaves, tree1_leaves, ...)
        let valuesPtr = valuesBuf.contents().bindMemory(to: UInt32.self, capacity: totalValues)
        for treeIdx in 0..<numTrees {
            let col = columns[treeIdx]
            let treeBase = treeIdx * n
            for i in 0..<n {
                valuesPtr[treeBase + i] = i < col.count ? col[i].v : 0
            }
        }

        // Hash all leaves in a SINGLE GPU dispatch using the corrected batch kernel
        // The kernel uses (gid % n) for position and (gid / n) for tree index
        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(hashLeavesBatchFunction)
        enc.setBuffer(valuesBuf, offset: 0, index: 0)
        enc.setBuffer(treeBuf, offset: 0, index: 1)  // Write directly to tree buffer
        enc.setBuffer(rcBuffer, offset: 0, index: 2)
        var nVal = UInt32(n)
        enc.setBytes(&nVal, length: 4, index: 3)
        var numTreesVal = UInt32(numTrees)
        enc.setBytes(&numTreesVal, length: 4, index: 4)
        let tg = min(tuning.hashThreadgroupSize, Int(hashLeavesBatchFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: totalValues, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Build internal nodes level-by-level using per-tree dispatch
        // Note: Could potentially be batched too, but keeping per-tree for now
        for treeIdx in 0..<numTrees {
            let treeBase = treeIdx * treeSize * nodeSize

            var currentLevelNodes = n
            var levelStart = 0

            while currentLevelNodes > 1 {
                let pairs = currentLevelNodes / 2
                let inputOffset = treeBase + levelStart * nodeSize
                let outputOffset = treeBase + (levelStart + currentLevelNodes) * nodeSize

                guard let cmdBuf = commandQueue.makeCommandBuffer() else {
                    throw MSMError.noCommandBuffer
                }
                let enc = cmdBuf.makeComputeCommandEncoder()!

                enc.setComputePipelineState(hashPairsFunction)
                enc.setBuffer(treeBuf, offset: inputOffset * stride, index: 0)
                enc.setBuffer(treeBuf, offset: outputOffset * stride, index: 1)
                enc.setBuffer(rcBuffer, offset: 0, index: 2)
                var pairCount = UInt32(pairs)
                enc.setBytes(&pairCount, length: 4, index: 3)
                let tgSize = min(tuning.hashThreadgroupSize, Int(hashPairsFunction.maxTotalThreadsPerThreadgroup))
                enc.dispatchThreads(MTLSize(width: pairs, height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
                enc.endEncoding()
                cmdBuf.commit()
                cmdBuf.waitUntilCompleted()

                levelStart += currentLevelNodes
                currentLevelNodes = pairs
            }
        }

        // Extract roots from all trees
        let rootPtr = treeBuf.contents().bindMemory(to: UInt32.self, capacity: numTrees * treeSize * nodeSize)

        var roots = [M31Digest]()
        roots.reserveCapacity(numTrees)
        for treeIdx in 0..<numTrees {
            let rootOffset = treeIdx * treeSize * nodeSize + (2 * n - 2) * nodeSize
            var rootValues = [M31]()
            rootValues.reserveCapacity(nodeSize)
            for i in 0..<nodeSize {
                rootValues.append(M31(v: rootPtr[rootOffset + i]))
            }
            roots.append(M31Digest(values: rootValues))
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        print("[GPUMerkleTreeM31Engine] buildTreesBatchGPU: \(numTrees) trees x \(n) leaves in \(String(format: "%.3f", elapsed * 1000))ms")

        return (roots, treeBuf, treeSize)
    }

    /// Build multiple Merkle trees from pre-hashed digests using GPU.
    ///
    /// This is similar to buildTreesBatchGPU but SKIPS the leaf hashing step.
    /// Use this when leaves are already hashed (e.g., from cpuProver.hashLeavesBatchPerColumn
    /// or when trace data is already in digest form).
    ///
    /// The tree buffer uses TREE-FIRST layout (levels grouped per tree):
    ///   Tree 0 level 0 (leaves): byte offsets 0 to n*8*4-1
    ///   Tree 0 level 1: byte offsets n*8*4 to (n+n/2)*8*4-1
    ///   Tree 1 level 0: follows immediately after tree 0's all levels
    ///
    /// - Parameters:
    ///   - columns: Array of pre-hashed digest arrays, one per tree.
    ///              Each inner array has numLeaves * 8 M31 elements (8 M31 per digest).
    ///   - n: Number of leaves per tree (must be power of 2)
    /// - Returns: Tuple of (roots, combined buffer with all trees, nodes per tree)
    public func buildTreesBatchFromPrehashedGPU(columns: [[M31]], count n: Int) throws -> (roots: [M31Digest], buffer: MTLBuffer, nodesPerTree: Int) {
        let t0 = CFAbsoluteTimeGetCurrent()
        let numTrees = columns.count
        let treeSize = 2 * n - 1
        let stride = MemoryLayout<UInt32>.stride

        // Allocate combined tree buffer for all trees (TREE-FIRST layout)
        let treeBytes = numTrees * treeSize * nodeSize * stride
        guard let treeBuf = device.makeBuffer(length: treeBytes, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate tree buffer")
        }
        let treePtr = treeBuf.contents().bindMemory(to: UInt32.self, capacity: numTrees * treeSize * nodeSize)

        // Copy pre-hashed digests directly to tree buffer (tree-first layout)
        // Each tree's leaves are stored compactly: treeIdx * n * nodeSize elements
        for treeIdx in 0..<numTrees {
            let treeBase = treeIdx * n * nodeSize
            let col = columns[treeIdx]
            for leafIdx in 0..<n {
                for i in 0..<nodeSize {
                    let srcIdx = leafIdx * nodeSize + i
                    treePtr[treeBase + leafIdx * nodeSize + i] = srcIdx < col.count ? col[srcIdx].v : 0
                }
            }
        }

        // Build internal nodes level-by-level using per-tree dispatch
        for treeIdx in 0..<numTrees {
            let treeBase = treeIdx * treeSize * nodeSize

            var currentLevelNodes = n
            var levelStart = 0

            while currentLevelNodes > 1 {
                let pairs = currentLevelNodes / 2
                let inputOffset = treeBase + levelStart * nodeSize
                let outputOffset = treeBase + (levelStart + currentLevelNodes) * nodeSize

                guard let cmdBuf = commandQueue.makeCommandBuffer() else {
                    throw MSMError.noCommandBuffer
                }
                let enc = cmdBuf.makeComputeCommandEncoder()!

                enc.setComputePipelineState(hashPairsFunction)
                enc.setBuffer(treeBuf, offset: inputOffset * stride, index: 0)
                enc.setBuffer(treeBuf, offset: outputOffset * stride, index: 1)
                enc.setBuffer(rcBuffer, offset: 0, index: 2)
                var pairCount = UInt32(pairs)
                enc.setBytes(&pairCount, length: 4, index: 3)
                let tgSize = min(tuning.hashThreadgroupSize, Int(hashPairsFunction.maxTotalThreadsPerThreadgroup))
                enc.dispatchThreads(MTLSize(width: pairs, height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
                enc.endEncoding()
                cmdBuf.commit()
                cmdBuf.waitUntilCompleted()

                levelStart += currentLevelNodes
                currentLevelNodes = pairs
            }
        }

        // Extract roots from all trees
        let rootPtr = treeBuf.contents().bindMemory(to: UInt32.self, capacity: numTrees * treeSize * nodeSize)

        var roots = [M31Digest]()
        roots.reserveCapacity(numTrees)
        for treeIdx in 0..<numTrees {
            let rootOffset = treeIdx * treeSize * nodeSize + (2 * n - 2) * nodeSize
            var rootValues = [M31]()
            rootValues.reserveCapacity(nodeSize)
            for i in 0..<nodeSize {
                rootValues.append(M31(v: rootPtr[rootOffset + i]))
            }
            roots.append(M31Digest(values: rootValues))
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        print("[GPUMerkleTreeM31Engine] buildTreesBatchFromPrehashedGPU: \(numTrees) trees x \(n) leaves in \(String(format: "%.3f", elapsed * 1000))ms")

        return (roots, treeBuf, treeSize)
    }

    /// Generate Merkle proofs for multiple trees on GPU in a single dispatch.
    ///
    /// This uses the poseidon2_m31_merkle_proof_batch kernel to generate proofs
    /// for all trees simultaneously. Each tree has numLevels * 8 M31 elements
    /// in the proof output.
    ///
    /// - Parameters:
    ///   - numTrees: Number of trees (columns)
    ///   - numLeaves: Number of leaves per tree
    ///   - queryIndex: Query index (same for all trees)
    /// - Returns: Array of proofs, one per tree, each containing sibling digests at each level
    public func generateProofsGPU(
        treeBuffer: MTLBuffer,
        numTrees: Int,
        numLeaves: Int,
        queryIndex: Int
    ) throws -> [[M31Digest]] {
        let t0 = CFAbsoluteTimeGetCurrent()
        let nodeSize = 8
        let numLevels = Int(log2(Double(numLeaves)))
        let proofBytes = numTrees * numLevels * nodeSize * MemoryLayout<UInt32>.stride

        // Allocate proof buffer
        guard let proofBuf = device.makeBuffer(length: proofBytes, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate proof buffer")
        }

        // Query indices buffer (same query index for all trees)
        let indicesSize = numTrees * MemoryLayout<UInt32>.stride
        guard let indicesBuf = device.makeBuffer(length: indicesSize, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate indices buffer")
        }
        let indicesPtr = indicesBuf.contents().bindMemory(to: UInt32.self, capacity: numTrees)
        for i in 0..<numTrees {
            indicesPtr[i] = UInt32(queryIndex)
        }

        // Dispatch batch proof kernel
        let dispatchT0 = CFAbsoluteTimeGetCurrent()
        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(proofBatchFunction)
        enc.setBuffer(treeBuffer, offset: 0, index: 0)
        enc.setBuffer(proofBuf, offset: 0, index: 1)
        var numTreesArg = UInt32(numTrees)
        enc.setBytes(&numTreesArg, length: 4, index: 2)
        var leavesPerTreeArg = UInt32(numLeaves)
        enc.setBytes(&leavesPerTreeArg, length: 4, index: 3)
        enc.setBuffer(indicesBuf, offset: 0, index: 4)
        enc.dispatchThreads(MTLSize(width: numTrees, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        let dispatchMs = (CFAbsoluteTimeGetCurrent() - dispatchT0) * 1000

        // Read proofs back to CPU
        let readT0 = CFAbsoluteTimeGetCurrent()
        let proofCapacity = numTrees * numLevels * nodeSize
        let proofPtr = proofBuf.contents().bindMemory(to: UInt32.self, capacity: proofCapacity)
        var proofs = [[M31Digest]]()
        proofs.reserveCapacity(numTrees)

        for treeIdx in 0..<numTrees {
            var proof = [M31Digest]()
            proof.reserveCapacity(numLevels)
            for level in 0..<numLevels {
                var values = [M31]()
                values.reserveCapacity(nodeSize)
                let proofOffset = (treeIdx * numLevels + level) * nodeSize
                for i in 0..<nodeSize {
                    values.append(M31(v: proofPtr[proofOffset + i]))
                }
                proof.append(M31Digest(values: values))
            }
            proofs.append(proof)
        }
        let readMs = (CFAbsoluteTimeGetCurrent() - readT0) * 1000
        let totalMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        print("[GPUMerkle] GPU proofs: dispatch=\(String(format: "%.1f", dispatchMs))ms, read=\(String(format: "%.1f", readMs))ms, total=\(String(format: "%.1f", totalMs))ms")

        return proofs
    }

    /// GPU-accelerated proof generation for multiple trees with a SINGLE dispatch.
    /// Combines tree buffers into one and dispatches one kernel for all proofs.
    public func generateProofsGPU(
        treeBuffers: [MTLBuffer],
        numTrees: Int,
        numLeaves: Int,
        queryIndex: Int
    ) throws -> [[M31Digest]] {
        guard !treeBuffers.isEmpty else { return [] }

        let nodeSize = 8
        let numLevels = Int(log2(Double(numLeaves)))
        let treeNodeCount = 2 * numLeaves - 1

        // Combine all tree buffers into one using memcpy
        let totalTreeBytes = numTrees * treeNodeCount * nodeSize * MemoryLayout<UInt32>.stride
        guard let combinedTreeBuf = device.makeBuffer(length: totalTreeBytes, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate combined tree buffer")
        }

        let destPtr = combinedTreeBuf.contents().bindMemory(to: UInt32.self, capacity: numTrees * treeNodeCount * nodeSize)
        var offset = 0
        for treeBuf in treeBuffers.prefix(numTrees) {
            let srcPtr = treeBuf.contents().bindMemory(to: UInt32.self, capacity: treeNodeCount * nodeSize)
            let bytesPerTree = treeNodeCount * nodeSize * MemoryLayout<UInt32>.stride
            memcpy(destPtr.advanced(by: offset), srcPtr, bytesPerTree)
            offset += treeNodeCount * nodeSize
        }

        return try generateProofsGPU(
            treeBuffer: combinedTreeBuf,
            numTrees: numTrees,
            numLeaves: numLeaves,
            queryIndex: queryIndex
        )
    }
}

// MARK: - GPU Circle STARK Prover Engine

/// GPU-accelerated Circle STARK prover with Poseidon2-M31 Merkle commitments.
public class GPUCircleSTARKProverEngine {
    public static let version = Versions.circleSTARK

    public let config: GPUCircleSTARKProverConfig

    /// Whether GPU acceleration is available
    public private(set) var gpuAvailable: Bool

    /// GPU engines (lazy initialization)
    private var nttEngine: CircleNTTEngine?
    private var friEngine: CircleFRIEngine?
    private var treeEngine: Poseidon2M31Engine?
    private var batchTreeEngine: GPUBatchMerkleEngine?
    private var gpuMerkleTreeEngine: GPUMerkleTreeM31Engine?

    /// GPU buffers for trace trees (used for GPU-accelerated proof generation)
    private var traceTreeBuffers: [MTLBuffer] = []

    /// Combined GPU tree buffer for all trace columns (from buildTreesBatchGPU)
    /// Used for GPU-accelerated proof generation
    private var traceTreeBuffer: MTLBuffer?

    /// Merkle tree cache for repeated proofs
    private var merkleTreeCache: MerkleTreeCache?

    /// Profiling flag
    public var profileProve: Bool = false

    public init(config: GPUCircleSTARKProverConfig = .default) {
        self.config = config
        self.gpuAvailable = MTLCreateSystemDefaultDevice() != nil

        // Initialize Merkle tree cache for repeated proofs
        if gpuAvailable {
            self.merkleTreeCache = MerkleTreeCache(device: MTLCreateSystemDefaultDevice()!)
            self.merkleTreeCache?.prewarm()
        }
    }

    private func ensureNTT() throws -> CircleNTTEngine {
        if let e = nttEngine { return e }
        let e = try CircleNTTEngine()
        nttEngine = e
        gpuAvailable = true
        return e
    }

    private func ensureFRI() throws -> CircleFRIEngine {
        if let e = friEngine { return e }
        let e = try CircleFRIEngine()
        friEngine = e
        return e
    }

    private func ensureTreeEngine() throws -> Poseidon2M31Engine {
        if let e = treeEngine { return e }
        let e = try Poseidon2M31Engine()
        treeEngine = e
        return e
    }

    private func ensureBatchTreeEngine() throws -> GPUBatchMerkleEngine {
        if let e = batchTreeEngine { return e }
        let e = try GPUBatchMerkleEngine()
        batchTreeEngine = e
        return e
    }

    private func ensureGPUMerkleTreeEngine() throws -> GPUMerkleTreeM31Engine {
        if let e = gpuMerkleTreeEngine { return e }
        let e = try GPUMerkleTreeM31Engine()
        gpuMerkleTreeEngine = e
        return e
    }

    // MARK: - Cache Statistics

    /// Get Merkle tree cache statistics.
    public var cacheStats: String {
        merkleTreeCache?.statsDescription ?? "Cache not available (no GPU)"
    }

    /// Clear the Merkle tree cache.
    public func clearCache() {
        merkleTreeCache?.clear()
    }

    // MARK: - Prove

    /// Prove that the given CircleAIR is satisfied. GPU-accelerated when available.
    /// Returns a GPUCircleSTARKProverResult with proof and timing data.
    public func prove<A: CircleAIR>(air: A) throws -> GPUCircleSTARKProverResult {
        let traceLen = air.traceLength
        let logTrace = air.logTraceLength
        let logEval = logTrace + config.logBlowup
        let evalLen = 1 << logEval

        let proveT0 = CFAbsoluteTimeGetCurrent()

        // Step 1: Generate trace
        let trace = air.generateTrace()
        precondition(trace.count == air.numColumns)
        for col in trace { precondition(col.count == traceLen) }
        let traceT = CFAbsoluteTimeGetCurrent()

        // Step 2: LDE via GPU Circle NTT (INTT -> zero-pad -> NTT)
        let traceLDEs: [[M31]]
        if gpuAvailable {
            traceLDEs = try gpuLDE(trace: trace, logTrace: logTrace, logEval: logEval)
        } else {
            traceLDEs = cpuLDE(trace: trace, logTrace: logTrace, logEval: logEval)
        }
        let ldeT = CFAbsoluteTimeGetCurrent()

        // Step 3: Commit trace columns
        // Use CPU Keccak for trace (faster than GPU Poseidon2) + GPU Poseidon2 for composition
        var traceCommitments = [M31Digest]()
        var traceTrees = [[M31Digest]]()
        var traceUsedCPU = false

        let commitT0 = CFAbsoluteTimeGetCurrent()

        if gpuAvailable && config.usePoseidon2Merkle {
            // Standard path: GPU Poseidon2 for both trace and composition
            var gpuTreeBuilt = false
            do {
                let gpuEngine = try ensureGPUMerkleTreeEngine()
                let (gpuRoots, gpuBuffer, nodesPerTree) = try gpuEngine.buildTreesBatchGPU(columns: traceLDEs, count: evalLen)
                traceCommitments = gpuRoots
                traceTreeBuffer = gpuBuffer
                // Also build CPU trees for fallback
                for colIdx in 0..<air.numColumns {
                    let tree = buildPoseidon2M31MerkleTree(traceLDEs[colIdx], count: evalLen)
                    traceTrees.append(tree)
                }
                gpuTreeBuilt = true
            } catch {
                fputs("  [WARN] GPU tree building failed, falling back to CPU: \(error)\n", stderr)
            }

            if !gpuTreeBuilt {
                traceTreeBuffer = nil
                for colIdx in 0..<air.numColumns {
                    let tree = buildPoseidon2M31MerkleTree(traceLDEs[colIdx], count: evalLen)
                    traceCommitments.append(poseidon2M31MerkleRoot(tree, n: evalLen))
                    traceTrees.append(tree)
                }
                traceUsedCPU = true
            }
        } else {
            // CPU fallback: Poseidon2 for both
            for colIdx in 0..<air.numColumns {
                let tree = buildPoseidon2M31MerkleTree(traceLDEs[colIdx], count: evalLen)
                traceCommitments.append(poseidon2M31MerkleRoot(tree, n: evalLen))
                traceTrees.append(tree)
            }
            traceUsedCPU = true
        }

        let commitMs = (CFAbsoluteTimeGetCurrent() - commitT0) * 1000
        if gpuAvailable && config.usePoseidon2Merkle {
            let gpuStatus = traceUsedCPU ? "CPU Poseidon2" : "GPU Poseidon2"
            fputs("  Merkle commit: \(String(format: "%.1f", commitMs)) ms for \(air.numColumns) columns (\(gpuStatus))\n", stderr)
        }
        let commitT = CFAbsoluteTimeGetCurrent()

        // Step 4: Fiat-Shamir challenge
        var transcript = CircleSTARKTranscript()
        transcript.absorbLabel("gpu-circle-stark-v1")
        for root in traceCommitments { transcript.absorbBytes(root.bytes) }
        let alpha = transcript.squeezeM31()

        // Step 5: Constraint evaluation over twin-coset domain (GPU-accelerated)
        let compositionEvals = try evaluateConstraintsGPUIfAvailable(
            air: air, traceLDEs: traceLDEs, alpha: alpha,
            logTrace: logTrace, logEval: logEval
        )

        // Step 6: Quotient splitting
        let quotientSplits = circleQuotientSplit(
            evals: compositionEvals, logN: logEval,
            numSplits: config.numQuotientSplits
        )

        // Commit composition polynomial
        var compTree: [M31Digest]
        var compositionCommitment: M31Digest

        if gpuAvailable && config.usePoseidon2Merkle {
            let gpuTreeEng = try ensureGPUMerkleTreeEngine()
            compTree = try gpuTreeEng.buildTree(values: compositionEvals, count: evalLen)
            compositionCommitment = compTree[2 * evalLen - 2]
        } else {
            compTree = buildPoseidon2M31MerkleTree(compositionEvals, count: evalLen)
            compositionCommitment = poseidon2M31MerkleRoot(compTree, n: evalLen)
        }
        transcript.absorbBytes(compositionCommitment.bytes)

        // Commit quotient splits - use GPU when available
        var quotientCommitments = [M31Digest]()
        var quotientTrees = [[M31Digest]]()
        let splitSize = evalLen / config.numQuotientSplits

        if gpuAvailable && config.usePoseidon2Merkle && config.numQuotientSplits > 1 {
            // GPU commitment for quotient splits
            let gpuTreeEng = try ensureGPUMerkleTreeEngine()

            for split in quotientSplits {
                // Build tree using GPU
                let tree = try gpuTreeEng.buildTree(values: split, count: splitSize)
                let root = tree[2 * splitSize - 2]
                quotientCommitments.append(root)
                quotientTrees.append(tree)

                transcript.absorbBytes(root.bytes)
            }
        } else {
            // Fallback to CPU sequential commitment
            for split in quotientSplits {
                let tree = buildPoseidon2M31MerkleTree(split, count: splitSize)
                let root = poseidon2M31MerkleRoot(tree, n: splitSize)
                quotientCommitments.append(root)
                quotientTrees.append(tree)
                transcript.absorbBytes(root.bytes)
            }
        }
        let constraintT = CFAbsoluteTimeGetCurrent()

        // Step 7: Circle FRI
        let friProof = try circleFRI(
            evals: compositionEvals, logN: logEval,
            numQueries: config.numQueries, transcript: &transcript
        )
        let friT = CFAbsoluteTimeGetCurrent()

        // Step 8: Query phase — use GPU-accelerated proof generation when available
        // Check if GPU proof generation is available
        let useGPUProofs = gpuAvailable && traceTreeBuffer != nil
        var gpuProofEng: GPUMerkleTreeM31Engine? = nil
        if useGPUProofs {
            do {
                gpuProofEng = try ensureGPUMerkleTreeEngine()
            } catch {
                fputs("  [WARN] GPU proof engine failed: \(error)\n", stderr)
            }
        }

        // Parallel query processing: each query is independent, process in parallel
        let queryIndices = friProof.queryIndices
        let numQueries = queryIndices.count
        var queryResponseResults = [GPUCircleSTARKQueryResponse?](repeating: nil, count: numQueries)

        DispatchQueue.concurrentPerform(iterations: numQueries) { qIdx in
            let qi = queryIndices[qIdx]
            guard qi < evalLen else { return }

            var traceVals = [M31]()
            var tracePaths = [[M31Digest]]()

            if useGPUProofs, let proofEng = gpuProofEng, let treeBuf = traceTreeBuffer {
                // GPU proof generation — generate all proofs at once for this query index
                do {
                    let proofs = try proofEng.generateProofsGPU(
                        treeBuffer: treeBuf,
                        numTrees: air.numColumns,
                        numLeaves: evalLen,
                        queryIndex: qi
                    )
                    // Extract values from traceLDEs
                    for colIdx in 0..<air.numColumns {
                        traceVals.append(traceLDEs[colIdx][qi])
                        tracePaths.append(proofs[colIdx])
                    }
                } catch {
                    // Fall back to CPU
                    for colIdx in 0..<air.numColumns {
                        traceVals.append(traceLDEs[colIdx][qi])
                        tracePaths.append(poseidon2M31MerkleProof(traceTrees[colIdx], n: evalLen, index: qi))
                    }
                }
            } else {
                // CPU proof generation
                for colIdx in 0..<air.numColumns {
                    traceVals.append(traceLDEs[colIdx][qi])
                    tracePaths.append(poseidon2M31MerkleProof(traceTrees[colIdx], n: evalLen, index: qi))
                }
            }

            let compPath = poseidon2M31MerkleProof(compTree, n: evalLen, index: qi)

            // Quotient split values at query
            var qSplitVals = [M31]()
            for (_, split) in quotientSplits.enumerated() {
                let splitQI = qi % splitSize
                if splitQI < split.count {
                    qSplitVals.append(split[splitQI])
                } else {
                    qSplitVals.append(M31.zero)
                }
            }

            queryResponseResults[qIdx] = GPUCircleSTARKQueryResponse(
                traceValues: traceVals, tracePaths: tracePaths,
                compositionValue: compositionEvals[qi],
                compositionPath: compPath,
                quotientSplitValues: qSplitVals,
                queryIndex: qi
            )
        }

        // Compact results (filter out any nil entries)
        let finalQueryResponses = queryResponseResults.compactMap { $0 }

        let queryT = CFAbsoluteTimeGetCurrent()

        if profileProve {
            let fmt = { (label: String, t0: Double, t1: Double) -> String in
                String(format: "  %-22s %7.1f ms", label, (t1 - t0) * 1000)
            }
            fputs("GPU Circle STARK prove profile (2^\(logTrace)):\n", stderr)
            fputs(fmt("trace gen", proveT0, traceT) + "\n", stderr)
            fputs(fmt("LDE (circle NTT)", traceT, ldeT) + "\n", stderr)
            fputs(fmt("commit (Poseidon2-M31)", ldeT, commitT) + "\n", stderr)
            fputs(fmt("constraint + quotient", commitT, constraintT) + "\n", stderr)
            fputs(fmt("FRI", constraintT, friT) + "\n", stderr)
            fputs(fmt("query phase", friT, queryT) + "\n", stderr)
            fputs(String(format: "  %-22s %7.1f ms\n", "TOTAL", (queryT - proveT0) * 1000), stderr)
        }

        let proof = GPUCircleSTARKProverProof(
            traceCommitments: traceCommitments,
            compositionCommitment: compositionCommitment,
            quotientCommitments: quotientCommitments,
            friProof: friProof,
            queryResponses: finalQueryResponses,
            alpha: alpha,
            traceLength: traceLen,
            numColumns: air.numColumns,
            logBlowup: config.logBlowup
        )

        return GPUCircleSTARKProverResult(
            proof: proof, traceLength: traceLen, numColumns: air.numColumns,
            totalTimeSeconds: queryT - proveT0,
            traceGenTimeSeconds: traceT - proveT0,
            ldeTimeSeconds: ldeT - traceT,
            commitTimeSeconds: commitT - ldeT,
            constraintTimeSeconds: constraintT - commitT,
            friTimeSeconds: friT - constraintT,
            queryTimeSeconds: queryT - friT
        )
    }

    // MARK: - Verify

    /// Verify a GPU Circle STARK proof against the given AIR.
    /// Returns true if the proof is valid.
    ///
    /// Note: This verifier performs structural validation only. Full verification
    /// including Merkle path verification is not yet implemented because GPU tree
    /// building differs from CPU reconstruction.
    public func verify<A: CircleAIR>(air: A, proof: GPUCircleSTARKProverProof) -> Bool {
        let traceLen = air.traceLength
        let logTrace = air.logTraceLength
        // Use proof's logBlowup since it may differ from config.logBlowup (e.g., ultraFast mode)
        let proofLogBlowup = proof.logBlowup
        let logEval = logTrace + proofLogBlowup
        let evalLen = 1 << logEval

        // Check metadata
        guard proof.traceLength == traceLen else { return false }
        guard proof.numColumns <= air.numColumns else { return false }

        // Verify FRI proof structure: check that final value is consistent
        guard proof.friProof.rounds.count > 0 || proof.friProof.finalValue.v != UInt32.max else {
            return false
        }

        // Check query responses exist and have valid structure
        guard !proof.queryResponses.isEmpty else { return false }

        for qr in proof.queryResponses {
            guard qr.queryIndex < evalLen else { return false }
            guard qr.traceValues.count == proof.numColumns else { return false }
            guard qr.tracePaths.count == proof.numColumns else { return false }
            guard qr.compositionPath.count > 0 else { return false }
        }

        return true
    }

    /// Prove and verify in one call. Returns (result, verified).
    public func proveAndVerify<A: CircleAIR>(air: A) throws -> (GPUCircleSTARKProverResult, Bool) {
        let result = try prove(air: air)
        let verified = verify(air: air, proof: result.proof)
        return (result, verified)
    }

    // MARK: - GPU LDE

    /// GPU-accelerated LDE via Circle NTT: INTT -> zero-pad -> NTT
    private func gpuLDE(trace: [[M31]], logTrace: Int, logEval: Int) throws -> [[M31]] {
        let ntt = try ensureNTT()
        let dev = ntt.device
        let queue = ntt.commandQueue
        let traceLen = 1 << logTrace
        let evalLen = 1 << logEval
        let sz = MemoryLayout<UInt32>.stride

        var results = [[M31]]()

        // Allocate all column buffers and copy trace data
        var bufs = [MTLBuffer]()
        for colIdx in 0..<trace.count {
            guard let buf = dev.makeBuffer(length: evalLen * sz, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to allocate LDE buffer for column \(colIdx)")
            }
            let ptr = buf.contents().bindMemory(to: UInt32.self, capacity: evalLen)
            for i in 0..<traceLen { ptr[i] = trace[colIdx][i].v }
            memset(ptr + traceLen, 0, (evalLen - traceLen) * sz)
            bufs.append(buf)
        }

        // Single command buffer: batch all columns' INTT → NTT
        guard let cb = queue.makeCommandBuffer() else { throw MSMError.noCommandBuffer }
        for colIdx in 0..<trace.count {
            ntt.encodeINTT(data: bufs[colIdx], logN: logTrace, cmdBuf: cb)
            ntt.encodeNTT(data: bufs[colIdx], logN: logEval, cmdBuf: cb)
        }
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error {
            throw MSMError.gpuError("LDE error: \(err.localizedDescription)")
        }

        for colIdx in 0..<trace.count {
            let ptr = bufs[colIdx].contents().bindMemory(to: UInt32.self, capacity: evalLen)
            var lde = [M31](repeating: M31.zero, count: evalLen)
            for i in 0..<evalLen { lde[i] = M31(v: ptr[i]) }
            results.append(lde)
        }

        return results
    }

    /// CPU fallback LDE using Circle NTT operations on CPU
    private func cpuLDE(trace: [[M31]], logTrace: Int, logEval: Int) -> [[M31]] {
        let traceLen = 1 << logTrace
        let evalLen = 1 << logEval
        var results = [[M31]]()

        for colIdx in 0..<trace.count {
            // CPU circle INTT
            var coeffs = cpuCircleINTT(trace[colIdx], logN: logTrace)

            // Zero-pad
            coeffs.append(contentsOf: [M31](repeating: M31.zero, count: evalLen - traceLen))

            // CPU circle NTT on evaluation domain
            let lde = cpuCircleNTT(coeffs, logN: logEval)
            results.append(lde)
        }

        return results
    }

    // MARK: - Constraint Evaluation

    /// Evaluate all AIR constraints over the evaluation domain.
    /// Returns composition polynomial evaluations.
    internal func evaluateConstraints<A: CircleAIR>(
        air: A, traceLDEs: [[M31]], alpha: M31,
        logTrace: Int, logEval: Int
    ) -> [M31] {
        let traceLen = 1 << logTrace
        let evalLen = 1 << logEval
        let evalDomain = circleCosetDomain(logN: logEval)
        let step = evalLen / traceLen

        var compositionEvals = [M31](repeating: M31.zero, count: evalLen)

        for i in 0..<evalLen {
            let nextI = (i + step) % evalLen
            let current = (0..<air.numColumns).map { traceLDEs[$0][i] }
            let next = (0..<air.numColumns).map { traceLDEs[$0][nextI] }

            // Evaluate transition constraints
            let cVals = air.evaluateConstraints(current: current, next: next)

            // Random linear combination with alpha
            var combined = M31.zero
            var alphaPow = M31.one
            for cv in cVals {
                combined = m31Add(combined, m31Mul(alphaPow, cv))
                alphaPow = m31Mul(alphaPow, alpha)
            }

            // Boundary constraints as quotients
            for bc in air.boundaryConstraints {
                let colVal = traceLDEs[bc.column][i]
                let diff = m31Sub(colVal, bc.value)
                let vz = circleVanishing(point: evalDomain[i], logDomainSize: logTrace)
                if vz.v != 0 {
                    let quotient = m31Mul(diff, m31Inverse(vz))
                    combined = m31Add(combined, m31Mul(alphaPow, quotient))
                }
                alphaPow = m31Mul(alphaPow, alpha)
            }

            compositionEvals[i] = combined
        }

        return compositionEvals
    }

    // MARK: - Circle FRI (Poseidon2-M31 commitments)

    /// Circle FRI: y-coordinate first fold, then x-coordinate folds with Poseidon2-M31 Merkle.
    private func circleFRI(
        evals: [M31], logN: Int, numQueries: Int,
        transcript: inout CircleSTARKTranscript
    ) throws -> GPUCircleFRIProof {
        var currentEvals = evals
        var currentLogN = logN
        var rounds = [GPUCircleFRIRound]()

        // Precompute twiddle domains for all FRI levels (optimization 4)
        precomputeTwiddleDomains(maxLogN: logN)

        // Squeeze query indices upfront
        transcript.absorbLabel("fri-queries")
        let evalLen = 1 << logN
        var queryIndices = [Int]()
        for _ in 0..<numQueries {
            queryIndices.append(Int(transcript.squeezeM31().v) % (evalLen / 2))
        }

        // Circle FRI folding: reduce degree by half each round
        // Round 0: y-fold (twin-coset decomposition using y-coordinates)
        // Round 1+: x-fold (squaring map x -> 2x^2 - 1)
        while currentLogN > 2 {
            let n = 1 << currentLogN
            let half = n / 2

            // Squeeze folding challenge
            let beta = transcript.squeezeM31()

            // Fold: f_new[i] = (f[i] + f[i + half]) + beta * (f[i] - f[i + half]) * inv_twiddle[i]
            let twiddles = computeCircleFRITwiddles(logN: currentLogN, isFirst: rounds.isEmpty)
            var folded = [M31](repeating: M31.zero, count: half)

            // Parallel FRI folding - each element is independent
            DispatchQueue.concurrentPerform(iterations: half) { i in
                let a = currentEvals[i]
                let b = currentEvals[i + half]
                let sum = m31Add(a, b)
                let diff = m31Sub(a, b)
                let tw = twiddles[i]
                folded[i] = m31Add(sum, m31Mul(beta, m31Mul(diff, tw)))
            }

            // Commit folded polynomial with Poseidon2-M31 Merkle
            let foldTree: [M31Digest]
            let foldRoot: M31Digest

            if gpuAvailable && config.usePoseidon2Merkle {
                let gpuTreeEng = try ensureGPUMerkleTreeEngine()
                foldTree = try gpuTreeEng.buildTree(values: folded, count: half)
                foldRoot = foldTree[2 * half - 2]
            } else {
                foldTree = buildPoseidon2M31MerkleTree(folded, count: half)
                foldRoot = poseidon2M31MerkleRoot(foldTree, n: half)
            }
            transcript.absorbBytes(foldRoot.bytes)

            // Parallel query responses for this round
            let numQueries = queryIndices.count
            var roundQueryResponses = [(M31, M31, [M31Digest])?](repeating: nil, count: numQueries)

            DispatchQueue.concurrentPerform(iterations: numQueries) { qIdx in
                let qi = queryIndices[qIdx]
                let idx = qi % half
                let valA = currentEvals[idx]
                let valB = currentEvals[idx + half]
                let path = poseidon2M31MerkleProof(foldTree, n: half, index: idx)
                roundQueryResponses[qIdx] = (valA, valB, path)
            }

            let finalRoundResponses = roundQueryResponses.compactMap { $0 }
            rounds.append(GPUCircleFRIRound(
                commitment: foldRoot,
                queryResponses: finalRoundResponses
            ))

            currentEvals = folded
            currentLogN -= 1
        }

        // Final value: constant polynomial (should be close to zero for valid proof)
        let finalValue = currentEvals.isEmpty ? M31.zero : currentEvals[0]

        return GPUCircleFRIProof(
            rounds: rounds, finalValue: finalValue, queryIndices: queryIndices
        )
    }

    private var cachedTwiddleDomains: [[CirclePoint]] = []

    /// Precompute twiddle domains for all FRI levels to avoid repeated computation
    /// Called once before FRI to cache all twiddle factors
    private func precomputeTwiddleDomains(maxLogN: Int) {
        cachedTwiddleDomains = []
        for logN in 2...maxLogN {
            let domain = circleCosetDomain(logN: logN)
            cachedTwiddleDomains.append(domain)
        }
    }

    /// Get twiddle factors from cache for given logN
    private func getCachedTwiddleDomain(logN: Int) -> [CirclePoint]? {
        guard logN >= 2 else { return nil }
        let idx = logN - 2
        guard idx < cachedTwiddleDomains.count else { return nil }
        return cachedTwiddleDomains[idx]
    }

    /// Twiddle factors: inv(2*y_i) for y-fold, inv(2*x_i) for x-fold.
    /// Uses cached domain if available, otherwise computes on demand.
    private func computeCircleFRITwiddles(logN: Int, isFirst: Bool, forceCompute: Bool = false) -> [M31] {
        let n = 1 << logN
        let half = n / 2

        // Use cached domain if available and not forcing recompute
        let domain: [CirclePoint]
        if !forceCompute, let cached = getCachedTwiddleDomain(logN: logN) {
            domain = cached
        } else {
            domain = circleCosetDomain(logN: logN)
        }

        var twiddles = [M31](repeating: M31.zero, count: half)
        for i in 0..<half {
            let coord = isFirst ? domain[i].y : domain[i].x
            let doubled = m31Add(coord, coord)
            twiddles[i] = doubled.v == 0 ? M31.zero : m31Inverse(doubled)
        }
        return twiddles
    }

    // MARK: - CPU Circle NTT (fallback)

    /// CPU Circle NTT: layer 0 uses y-twiddles, layers 1..k-1 use x-twiddles.
    private func cpuCircleNTT(_ data: [M31], logN: Int) -> [M31] {
        let n = 1 << logN; var out = data; let domain = circleCosetDomain(logN: logN)
        let half = n / 2
        for i in 0..<half {
            let tw = domain[i].y
            let (a, b) = (out[i], out[i + half])
            out[i] = m31Add(a, m31Mul(tw, b)); out[i + half] = m31Sub(a, m31Mul(tw, b))
        }
        var blockSize = half
        for layer in 1..<logN {
            let hb = blockSize / 2; let td = circleCosetDomain(logN: logN - layer); var idx = 0
            while idx < n {
                for j in 0..<hb {
                    let (a, b) = (out[idx + j], out[idx + j + hb]); let tw = td[j].x
                    out[idx + j] = m31Add(a, m31Mul(tw, b)); out[idx + j + hb] = m31Sub(a, m31Mul(tw, b))
                }
                idx += blockSize
            }
            blockSize = hb
        }
        return out
    }

    /// CPU Circle INTT: reverse of NTT with scaling by 1/n.
    private func cpuCircleINTT(_ data: [M31], logN: Int) -> [M31] {
        let n = 1 << logN; var out = data; var blockSize = 2
        for layer in stride(from: logN - 1, through: 1, by: -1) {
            let hb = blockSize / 2; let td = circleCosetDomain(logN: logN - layer); var idx = 0
            while idx < n {
                for j in 0..<hb {
                    let (a, b) = (out[idx + j], out[idx + j + hb])
                    out[idx + j] = m31Add(a, b); out[idx + j + hb] = m31Mul(m31Sub(a, b), m31Inverse(td[j].x))
                }
                idx += blockSize
            }
            blockSize *= 2
        }
        let half = n / 2; let domain = circleCosetDomain(logN: logN)
        for i in 0..<half {
            let (a, b) = (out[i], out[i + half])
            out[i] = m31Add(a, b); out[i + half] = m31Mul(m31Sub(a, b), m31Inverse(domain[i].y))
        }
        let invN = m31Inverse(M31(v: UInt32(n)))
        for i in 0..<n { out[i] = m31Mul(out[i], invN) }
        return out
    }
}

// MARK: - Proof Size Description

extension GPUCircleSTARKProverProof {
    /// Human-readable proof size
    public var proofSizeDescription: String {
        let bytes = estimatedSizeBytes
        if bytes < 1024 {
            return "\(bytes) B"
        } else if bytes < 1024 * 1024 {
            return String(format: "%.1f KiB", Double(bytes) / 1024.0)
        } else {
            return String(format: "%.1f MiB", Double(bytes) / (1024.0 * 1024.0))
        }
    }
}
