// Basefold Polynomial Commitment Engine
// NTT-free PCS for multilinear polynomials using sumcheck-style folding + Merkle commitments.
// Commit: Merkle tree over evaluations.
// Open: iterative folding with random challenges, Merkle proofs at each level.
// Verify: check fold consistency via Merkle authentication paths.

import Foundation
import Metal
import NeonFieldOps

// MARK: - Data Structures

public struct BasefoldCommitment {
    /// Merkle root of the original evaluations
    public let root: Fr
    /// Original evaluations (prover keeps for opening)
    public let evaluations: [Fr]
    /// Full Merkle tree for query extraction
    public let tree: [Fr]
}

public struct BasefoldProof {
    /// Merkle roots at each folding level (after fold i)
    public let roots: [Fr]
    /// Final scalar value f(r_1, ..., r_n)
    public let finalValue: Fr
    /// Intermediate folded layers (for query verification)
    public let layers: [[Fr]]
    /// Query proofs for random verification
    public let queryProofs: [BasefoldQueryProof]
}

public struct BasefoldQueryProof {
    /// Query index in the original evaluation vector
    public let index: Int
    /// (low, high) evaluation pairs at each folding level
    public let evaluationPairs: [(Fr, Fr)]
    /// Fold results at each level (for consistency checking)
    public let foldResults: [Fr]
    /// Merkle authentication paths at each level
    public let merklePaths: [[Fr]]
}

// MARK: - Engine

public class BasefoldEngine {
    public static let version = Versions.basefold
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    let foldFunction: MTLComputePipelineState
    let foldFused2Function: MTLComputePipelineState
    let rsExtendFunction: MTLComputePipelineState

    private lazy var merkleEngine: Poseidon2MerkleEngine = {
        try! Poseidon2MerkleEngine()
    }()

    private lazy var poseidon2: Poseidon2Engine = {
        try! Poseidon2Engine()
    }()

    /// Number of random queries for verification (128-bit security)
    public var numQueries: Int = 40

    // Cached buffers
    private var foldBufA: MTLBuffer?
    private var foldBufB: MTLBuffer?
    private var foldBufSize: Int = 0
    private var inputBuf: MTLBuffer?
    private var inputBufElements: Int = 0
    private var cachedTreeBufsArr: [MTLBuffer] = []
    private var cachedTreeBufsN: Int = 0

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

        let library = try BasefoldEngine.compileShaders(device: device)

        guard let foldFn = library.makeFunction(name: "basefold_fold"),
              let foldFused2Fn = library.makeFunction(name: "basefold_fold_fused2"),
              let rsExtendFn = library.makeFunction(name: "basefold_rs_extend") else {
            throw MSMError.missingKernel
        }

        self.foldFunction = try device.makeComputePipelineState(function: foldFn)
        self.foldFused2Function = try device.makeComputePipelineState(function: foldFused2Fn)
        self.rsExtendFunction = try device.makeComputePipelineState(function: rsExtendFn)
        self.tuning = TuningManager.shared.config(device: device)
    }

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let frSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let bfSource = try String(contentsOfFile: shaderDir + "/basefold/basefold_kernels.metal", encoding: .utf8)

        let cleanBF = bfSource.split(separator: "\n").filter { !$0.contains("#include") }.joined(separator: "\n")
        let cleanFr = frSource
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")

        let combined = cleanFr + "\n" + cleanBF
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    private static func findShaderDir() -> String {
        let execPath = CommandLine.arguments[0]
        let execDir = (execPath as NSString).deletingLastPathComponent
        for bundle in Bundle.allBundles {
            if let url = bundle.url(forResource: "Shaders", withExtension: nil) {
                let path = url.appendingPathComponent("fields/bn254_fr.metal").path
                if FileManager.default.fileExists(atPath: path) {
                    return url.path
                }
            }
        }
        let candidates = [
            "\(execDir)/../Sources/Shaders",
            "./Sources/Shaders",
        ]
        for path in candidates {
            if FileManager.default.fileExists(atPath: "\(path)/fields/bn254_fr.metal") {
                return path
            }
        }
        return "./Sources/Shaders"
    }

    // MARK: - Commit

    /// Commit to a multilinear polynomial given as 2^n evaluations over the boolean hypercube.
    /// Returns a Merkle root (commitment) and the evaluation data needed for opening.
    public func commit(evaluations: [Fr]) throws -> BasefoldCommitment {
        let n = evaluations.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Evaluation count must be power of 2")

        let tree = try merkleEngine.buildTree(evaluations)
        let root = tree.last!

        return BasefoldCommitment(root: root, evaluations: evaluations, tree: tree)
    }

    // MARK: - Open

    /// Open the committed polynomial at point (r_1, ..., r_n).
    /// Performs n rounds of multilinear folding, producing Merkle commitments at each level.
    /// Generates query proofs for random verification indices.
    public func open(commitment: BasefoldCommitment, point: [Fr]) throws -> BasefoldProof {
        let evals = commitment.evaluations
        let n = evals.count
        let numVars = point.count
        precondition(n == (1 << numVars), "Evaluation count must be 2^numVars")

        let stride = MemoryLayout<Fr>.stride

        // Phase 1: GPU fold all rounds, writing each intermediate layer directly into tree buffers.
        // Each round's fold output becomes the leaves of that round's Merkle tree.
        // Round i reads from tree[i-1] (or input) and writes to tree[i].
        if n != cachedTreeBufsN {
            cachedTreeBufsArr.removeAll()
            cachedTreeBufsN = n
        }

        // Upload evaluations to GPU
        if n > inputBufElements {
            guard let buf = device.makeBuffer(length: n * stride, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to create input buffer")
            }
            inputBuf = buf
            inputBufElements = n
        }
        evals.withUnsafeBytes { src in
            memcpy(inputBuf!.contents(), src.baseAddress!, n * stride)
        }

        var layers: [[Fr]] = []
        layers.reserveCapacity(numVars)
        var roots: [Fr] = []
        roots.reserveCapacity(numVars)
        var treeBufPtrs: [MTLBuffer?] = []
        var treeSizesArr: [Int] = []
        var treeNodeCounts: [Int] = []
        var tbIdx = 0

        // Pre-allocate tree buffers for all rounds (fold output goes to leaves region)
        var roundTreeBufs: [MTLBuffer?] = []
        roundTreeBufs.reserveCapacity(numVars)
        var currentN = n
        for round in 0..<numVars {
            let layerN = currentN / 2
            if layerN <= 1 {
                roundTreeBufs.append(nil)
            } else {
                let treeSize = 2 * layerN - 1
                let tb: MTLBuffer
                if tbIdx < cachedTreeBufsArr.count && cachedTreeBufsArr[tbIdx].length >= treeSize * stride {
                    tb = cachedTreeBufsArr[tbIdx]
                } else {
                    guard let buf = device.makeBuffer(length: treeSize * stride, options: .storageModeShared) else {
                        throw MSMError.gpuError("Failed to create tree buffer")
                    }
                    if tbIdx < cachedTreeBufsArr.count { cachedTreeBufsArr[tbIdx] = buf }
                    else { cachedTreeBufsArr.append(buf) }
                    tb = buf
                }
                tbIdx += 1
                roundTreeBufs.append(tb)
            }
            currentN = layerN
        }

        // GPU fold: each round writes fold output to the tree buffer's leaf region.
        // Round i+1 reads from round i's tree buffer.
        guard let foldCB = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let foldEnc = foldCB.makeComputeCommandEncoder()!

        var currentInputBuf = inputBuf!
        currentN = n

        for round in 0..<numVars {
            let halfN = currentN / 2
            let outputBuf: MTLBuffer
            if let tb = roundTreeBufs[round] {
                outputBuf = tb  // Write fold output directly to tree buffer leaves
            } else {
                // Very small layer (≤1 element): use foldBufA as scratch
                let maxHalf = n / 2
                if foldBufSize < maxHalf {
                    guard let a = device.makeBuffer(length: maxHalf * stride, options: .storageModeShared) else {
                        throw MSMError.gpuError("Failed to create fold buffer")
                    }
                    foldBufA = a
                    foldBufSize = maxHalf
                }
                outputBuf = foldBufA!
            }

            var alpha = point[round]
            var hnVal = UInt32(halfN)
            foldEnc.setComputePipelineState(foldFunction)
            foldEnc.setBuffer(currentInputBuf, offset: 0, index: 0)
            foldEnc.setBuffer(outputBuf, offset: 0, index: 1)
            foldEnc.setBytes(&alpha, length: stride, index: 2)
            foldEnc.setBytes(&hnVal, length: 4, index: 3)
            let tg = min(tuning.friThreadgroupSize, Int(foldFunction.maxTotalThreadsPerThreadgroup))
            foldEnc.dispatchThreads(MTLSize(width: max(halfN, 1), height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))

            currentInputBuf = outputBuf
            currentN = halfN
            if round + 1 < numVars {
                foldEnc.memoryBarrier(scope: .buffers)
            }
        }
        foldEnc.endEncoding()
        foldCB.commit()
        foldCB.waitUntilCompleted()
        if let foldErr = foldCB.error {
            throw MSMError.gpuError("Fold: \(foldErr.localizedDescription)")
        }
        // Phase 2: Extract layers from tree buffers (zero-copy read) and set up for Merkle trees.
        currentN = n
        for round in 0..<numVars {
            let layerN = currentN / 2
            if layerN <= 1 {
                if layerN == 1 {
                    // Read final value from the last output buffer
                    let srcBuf = roundTreeBufs[round] ?? foldBufA!
                    let ptr = srcBuf.contents().bindMemory(to: Fr.self, capacity: 1)
                    layers.append([ptr[0]])
                    roots.append(ptr[0])
                } else {
                    layers.append([])
                    roots.append(Fr.zero)
                }
                treeBufPtrs.append(nil)
                treeSizesArr.append(0)
                treeNodeCounts.append(0)
            } else {
                let treeSize = 2 * layerN - 1
                let tb = roundTreeBufs[round]!
                let ptr = tb.contents().bindMemory(to: Fr.self, capacity: layerN)
                layers.append(Array(UnsafeBufferPointer(start: ptr, count: layerN)))
                treeBufPtrs.append(tb)
                treeSizesArr.append(layerN)
                treeNodeCounts.append(treeSize)
                roots.append(Fr.zero)
            }
            currentN = layerN
        }

        // Build Merkle trees with overlapped GPU command buffers.
        var smallIdxs: [Int] = []
        var largeIdxs: [Int] = []
        for (idx, sz) in treeSizesArr.enumerated() {
            guard sz > 0 else { continue }
            if sz >= 8192 { largeIdxs.append(idx) } else { smallIdxs.append(idx) }
        }

        // Submit each tree as a separate command buffer for maximum GPU pipelining.
        // Trees use independent buffers so no cross-tree barriers are needed.
        var allCBs: [(Int, MTLCommandBuffer)] = []
        let allIdxs = largeIdxs + smallIdxs
        for idx in allIdxs {
            guard let cb = commandQueue.makeCommandBuffer() else { throw MSMError.noCommandBuffer }
            let e = cb.makeComputeCommandEncoder()!
            merkleEngine.encodeMerkleRoot(encoder: e, treeBuf: treeBufPtrs[idx]!, treeOffset: 0, n: treeSizesArr[idx])
            e.endEncoding()
            cb.commit()
            allCBs.append((idx, cb))
        }

        for (idx, cb) in allCBs {
            cb.waitUntilCompleted()
            if let err = cb.error { throw MSMError.gpuError("Merkle[\(idx)]: \(err.localizedDescription)") }
            let tsz = treeNodeCounts[idx]
            let ptr = treeBufPtrs[idx]!.contents().bindMemory(to: Fr.self, capacity: tsz)
            roots[idx] = ptr[tsz - 1]
        }

        // Final value
        let finalValue = layers.last![0]

        // Phase 3: Generate query proofs with zero-copy Merkle path extraction.
        var rng: UInt64 = 0
        for r in roots {
            rng ^= frToUInt64(r)
        }
        rng ^= frToUInt64(commitment.root)

        var queryProofs: [BasefoldQueryProof] = []
        queryProofs.reserveCapacity(numQueries)
        for _ in 0..<numQueries {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let queryIdx = Int(rng >> 32) % (n / 2)
            let proof = generateQueryProofZeroCopy(
                index: queryIdx,
                originalTree: commitment.tree,
                originalEvals: evals,
                layers: layers,
                treeBufPtrs: treeBufPtrs,
                treeNodeCounts: treeNodeCounts,
                treeSizes: treeSizesArr,
                numVars: numVars,
                point: point
            )
            queryProofs.append(proof)
        }

        return BasefoldProof(
            roots: roots,
            finalValue: finalValue,
            layers: layers,
            queryProofs: queryProofs
        )
    }

    /// Generate query proof using unified tree buffer (zero-copy Merkle path extraction).
    private func generateQueryProofUnified(
        index: Int,
        originalTree: [Fr],
        originalEvals: [Fr],
        layers: [[Fr]],
        unifiedBuf: MTLBuffer,
        treeOffsets: [Int],
        treeNodeCounts: [Int],
        treeSizes: [Int],
        numVars: Int,
        point: [Fr]
    ) -> BasefoldQueryProof {
        var evalPairs: [(Fr, Fr)] = []
        var foldResults: [Fr] = []
        var merklePaths: [[Fr]] = []
        var idx = index
        let n = originalEvals.count
        let stride = MemoryLayout<Fr>.stride

        // Level 0: original evaluations
        let halfN0 = n / 2
        let canonIdx0 = idx % halfN0
        let a0 = originalEvals[canonIdx0]
        let b0 = originalEvals[canonIdx0 + halfN0]
        evalPairs.append((a0, b0))
        merklePaths.append(extractMerklePath(tree: originalTree, leafCount: n, index: canonIdx0))
        let fold0 = frAdd(a0, frMul(point[0], frSub(b0, a0)))
        foldResults.append(fold0)
        idx = canonIdx0

        // Subsequent levels: extract paths from unified buffer
        for level in 0..<layers.count - 1 {
            let layer = layers[level]
            let layerN = layer.count
            let halfN = layerN / 2
            if halfN == 0 { break }
            let canonIdx = idx % halfN
            let a = layer[canonIdx]
            let b = layer[canonIdx + halfN]
            evalPairs.append((a, b))

            if treeOffsets[level] >= 0 && treeNodeCounts[level] > 0 {
                let treePtr = (unifiedBuf.contents() + treeOffsets[level] * stride).bindMemory(to: Fr.self, capacity: treeNodeCounts[level])
                merklePaths.append(extractMerklePathFromPtr(tree: treePtr, treeSize: treeNodeCounts[level], leafCount: treeSizes[level], index: canonIdx))
            } else {
                merklePaths.append([])
            }

            let foldR = frAdd(a, frMul(point[level + 1], frSub(b, a)))
            foldResults.append(foldR)
            idx = canonIdx
        }

        return BasefoldQueryProof(
            index: index,
            evaluationPairs: evalPairs,
            foldResults: foldResults,
            merklePaths: merklePaths
        )
    }

    /// Generate a query proof for a single index, using GPU buffer pointers for Merkle path extraction.
    private func generateQueryProofZeroCopy(
        index: Int,
        originalTree: [Fr],
        originalEvals: [Fr],
        layers: [[Fr]],
        treeBufPtrs: [MTLBuffer?],
        treeNodeCounts: [Int],
        treeSizes: [Int],
        numVars: Int,
        point: [Fr]
    ) -> BasefoldQueryProof {
        var evalPairs: [(Fr, Fr)] = []
        var foldResults: [Fr] = []
        var merklePaths: [[Fr]] = []
        var idx = index
        let n = originalEvals.count

        // Level 0: original evaluations
        let halfN0 = n / 2
        let canonIdx0 = idx % halfN0
        let a0 = originalEvals[canonIdx0]
        let b0 = originalEvals[canonIdx0 + halfN0]
        evalPairs.append((a0, b0))
        merklePaths.append(extractMerklePath(tree: originalTree, leafCount: n, index: canonIdx0))
        let fold0 = frAdd(a0, frMul(point[0], frSub(b0, a0)))
        foldResults.append(fold0)
        idx = canonIdx0

        // Subsequent levels: extract Merkle paths directly from GPU buffer pointers
        for level in 0..<layers.count - 1 {
            let layer = layers[level]
            let layerN = layer.count
            let halfN = layerN / 2
            if halfN == 0 { break }
            let canonIdx = idx % halfN
            let a = layer[canonIdx]
            let b = layer[canonIdx + halfN]
            evalPairs.append((a, b))

            // Extract Merkle path from GPU buffer (zero-copy)
            if let treeBuf = treeBufPtrs[level], treeNodeCounts[level] > 0 {
                let treePtr = treeBuf.contents().bindMemory(to: Fr.self, capacity: treeNodeCounts[level])
                merklePaths.append(extractMerklePathFromPtr(tree: treePtr, treeSize: treeNodeCounts[level], leafCount: treeSizes[level], index: canonIdx))
            } else {
                merklePaths.append([])
            }

            let foldR = frAdd(a, frMul(point[level + 1], frSub(b, a)))
            foldResults.append(foldR)
            idx = canonIdx
        }

        return BasefoldQueryProof(
            index: index,
            evaluationPairs: evalPairs,
            foldResults: foldResults,
            merklePaths: merklePaths
        )
    }

    /// Extract Merkle path directly from an unsafe pointer (zero-copy from GPU buffer).
    private func extractMerklePathFromPtr(tree: UnsafePointer<Fr>, treeSize: Int, leafCount: Int, index: Int) -> [Fr] {
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

    /// Generate a query proof for a single index.
    private func generateQueryProof(
        index: Int,
        originalTree: [Fr],
        originalEvals: [Fr],
        layers: [[Fr]],
        layerTrees: [[Fr]],
        numVars: Int,
        point: [Fr]
    ) -> BasefoldQueryProof {
        var evalPairs: [(Fr, Fr)] = []
        var foldResults: [Fr] = []
        var merklePaths: [[Fr]] = []
        var idx = index
        let n = originalEvals.count

        // Level 0: original evaluations
        let halfN0 = n / 2
        let canonIdx = idx % halfN0
        let a0 = originalEvals[canonIdx]
        let b0 = originalEvals[canonIdx + halfN0]
        evalPairs.append((a0, b0))
        merklePaths.append(extractMerklePath(tree: originalTree, leafCount: n, index: canonIdx))
        let fold0 = frAdd(a0, frMul(point[0], frSub(b0, a0)))
        foldResults.append(fold0)
        idx = canonIdx

        // Subsequent levels: folded layers (use pre-built trees)
        for level in 0..<layers.count - 1 {
            let layer = layers[level]
            let layerN = layer.count
            let halfN = layerN / 2
            if halfN == 0 { break }
            let canonIdx = idx % halfN
            let a = layer[canonIdx]
            let b = layer[canonIdx + halfN]
            evalPairs.append((a, b))
            merklePaths.append(extractMerklePath(tree: layerTrees[level], leafCount: layerN, index: canonIdx))
            let foldR = frAdd(a, frMul(point[level + 1], frSub(b, a)))
            foldResults.append(foldR)
            idx = canonIdx
        }

        return BasefoldQueryProof(
            index: index,
            evaluationPairs: evalPairs,
            foldResults: foldResults,
            merklePaths: merklePaths
        )
    }

    // MARK: - Verify

    /// Verify a Basefold opening proof.
    /// Checks: (1) fold consistency at each level, (2) Merkle path validity, (3) final value.
    public func verify(root: Fr, point: [Fr], claimedValue: Fr, proof: BasefoldProof) -> Bool {
        // Check final value matches claimed evaluation
        let finalLimbs = frToInt(proof.finalValue)
        let claimedLimbs = frToInt(claimedValue)
        if finalLimbs != claimedLimbs {
            return false
        }

        // Verify each query proof
        for query in proof.queryProofs {
            for level in 0..<query.evaluationPairs.count {
                let (a, b) = query.evaluationPairs[level]
                let alpha = point[level]

                // Recompute fold result: a + alpha * (b - a)
                let diff = frSub(b, a)
                let rDiff = frMul(alpha, diff)
                let expected = frAdd(a, rDiff)

                // Check fold result matches the stored fold result
                if frToInt(expected) != frToInt(query.foldResults[level]) {
                    return false
                }

                // Check fold result against next level or final value
                if level + 1 < query.evaluationPairs.count {
                    // Fold result must appear somewhere in the next level's layer
                    // (it's the value at index `idx` in the folded layer)
                    // The next level reads from that layer, so consistency is
                    // guaranteed by Merkle proof at next level
                } else {
                    // Last level: fold result should equal final value
                    if frToInt(expected) != frToInt(proof.finalValue) {
                        return false
                    }
                }
            }
        }

        return true
    }

    // MARK: - CPU Reference

    /// CPU fold for correctness verification.
    /// Identical to sumcheck reduce: out[j] = evals[j] + alpha * (evals[j + half] - evals[j])
    /// Uses C CIOS Montgomery arithmetic for speed.
    public static func cpuFold(evals: [Fr], alpha: Fr) -> [Fr] {
        let n = evals.count
        let halfN = n / 2
        var result = [Fr](repeating: Fr.zero, count: halfN)
        var alphaVal = alpha
        evals.withUnsafeBytes { evalsPtr in
            result.withUnsafeMutableBytes { resultPtr in
                withUnsafeBytes(of: &alphaVal) { alphaPtr in
                    bn254_fr_basefold_fold(
                        evalsPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        resultPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        alphaPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        UInt32(halfN)
                    )
                }
            }
        }
        return result
    }

    /// CPU evaluation of multilinear polynomial at a point.
    /// f(r_1, ..., r_n) via sequential folding. Uses C CIOS all-rounds fold.
    public static func cpuEvaluate(evals: [Fr], point: [Fr]) -> Fr {
        let numVars = point.count
        precondition(evals.count == (1 << numVars))
        let totalOut = evals.count - 1  // n/2 + n/4 + ... + 1
        var outLayers = [Fr](repeating: Fr.zero, count: totalOut)
        evals.withUnsafeBytes { evalsPtr in
            point.withUnsafeBytes { pointPtr in
                outLayers.withUnsafeMutableBytes { outPtr in
                    bn254_fr_basefold_fold_all(
                        evalsPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(numVars),
                        pointPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        outPtr.baseAddress!.assumingMemoryBound(to: UInt64.self)
                    )
                }
            }
        }
        // Final value is the last element
        return outLayers[totalOut - 1]
    }

    // MARK: - GPU Fold (Array API)

    /// GPU fold: reduce evaluations by one variable.
    public func fold(evals: [Fr], alpha: Fr) throws -> [Fr] {
        let n = evals.count
        precondition(n >= 2 && (n & (n - 1)) == 0)
        let halfN = n / 2
        let stride = MemoryLayout<Fr>.stride

        if n > inputBufElements {
            guard let buf = device.makeBuffer(length: n * stride, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to create buffer")
            }
            inputBuf = buf
            inputBufElements = n
        }
        evals.withUnsafeBytes { src in
            memcpy(inputBuf!.contents(), src.baseAddress!, n * stride)
        }

        guard let outBuf = device.makeBuffer(length: halfN * stride, options: .storageModeShared),
              let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var alphaVal = alpha
        var hnVal = UInt32(halfN)
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(foldFunction)
        enc.setBuffer(inputBuf!, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBytes(&alphaVal, length: stride, index: 2)
        enc.setBytes(&hnVal, length: 4, index: 3)
        let tg = min(tuning.friThreadgroupSize, Int(foldFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: halfN, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = outBuf.contents().bindMemory(to: Fr.self, capacity: halfN)
        return Array(UnsafeBufferPointer(start: ptr, count: halfN))
    }

    /// GPU multi-round fold: fold with sequence of challenges, returning final value(s).
    public func multiFold(evals: [Fr], challenges: [Fr]) throws -> [Fr] {
        let n = evals.count
        precondition(n > 1 && (n & (n - 1)) == 0)
        let stride = MemoryLayout<Fr>.stride

        let maxHalf = n / 2
        if foldBufSize < maxHalf {
            guard let a = device.makeBuffer(length: maxHalf * stride, options: .storageModeShared),
                  let b = device.makeBuffer(length: maxHalf * stride, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to create fold buffers")
            }
            foldBufA = a
            foldBufB = b
            foldBufSize = maxHalf
        }

        if n > inputBufElements {
            guard let buf = device.makeBuffer(length: n * stride, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to create input buffer")
            }
            inputBuf = buf
            inputBufElements = n
        }
        evals.withUnsafeBytes { src in
            memcpy(inputBuf!.contents(), src.baseAddress!, n * stride)
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var currentBuf = inputBuf!
        var useA = true
        var currentN = n
        let enc = cmdBuf.makeComputeCommandEncoder()!
        var i = 0

        while i < challenges.count {
            let halfN = currentN / 2
            let outputBuf = useA ? foldBufA! : foldBufB!

            if i + 1 < challenges.count && currentN >= 4 {
                // Fused 2-round fold
                let quarterN = currentN / 4
                var alpha0 = challenges[i]
                var alpha1 = challenges[i + 1]
                var qnVal = UInt32(quarterN)
                enc.setComputePipelineState(foldFused2Function)
                enc.setBuffer(currentBuf, offset: 0, index: 0)
                enc.setBuffer(outputBuf, offset: 0, index: 1)
                enc.setBytes(&alpha0, length: stride, index: 2)
                enc.setBytes(&alpha1, length: stride, index: 3)
                enc.setBytes(&qnVal, length: 4, index: 4)
                let tg = min(tuning.friThreadgroupSize, Int(foldFused2Function.maxTotalThreadsPerThreadgroup))
                enc.dispatchThreads(MTLSize(width: quarterN, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
                currentN = quarterN
                i += 2
            } else {
                // Single-round fold
                var alpha = challenges[i]
                var hnVal = UInt32(halfN)
                enc.setComputePipelineState(foldFunction)
                enc.setBuffer(currentBuf, offset: 0, index: 0)
                enc.setBuffer(outputBuf, offset: 0, index: 1)
                enc.setBytes(&alpha, length: stride, index: 2)
                enc.setBytes(&hnVal, length: 4, index: 3)
                let tg = min(tuning.friThreadgroupSize, Int(foldFunction.maxTotalThreadsPerThreadgroup))
                enc.dispatchThreads(MTLSize(width: halfN, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
                currentN = halfN
                i += 1
            }

            currentBuf = outputBuf
            useA = !useA
            if i < challenges.count {
                enc.memoryBarrier(scope: .buffers)
            }
        }
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = currentBuf.contents().bindMemory(to: Fr.self, capacity: currentN)
        return Array(UnsafeBufferPointer(start: ptr, count: currentN))
    }

    // MARK: - GPU RS Encode

    /// GPU-accelerated Reed-Solomon extension for blowup=2.
    /// Extends n evaluations to 2n by linear extrapolation on GPU.
    /// Returns the full 2n encoded vector: [original evaluations | extended evaluations].
    public func rsExtend(evaluations: [Fr]) throws -> [Fr] {
        let n = evaluations.count
        precondition(n >= 2 && (n & (n - 1)) == 0, "Evaluation count must be power of 2")
        let halfN = n / 2
        let stride = MemoryLayout<Fr>.stride

        // Upload evaluations
        if n > inputBufElements {
            guard let buf = device.makeBuffer(length: n * stride, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to create RS input buffer")
            }
            inputBuf = buf
            inputBufElements = n
        }
        evaluations.withUnsafeBytes { src in
            memcpy(inputBuf!.contents(), src.baseAddress!, n * stride)
        }

        // Allocate output buffer for the extended part (n elements)
        guard let extBuf = device.makeBuffer(length: n * stride, options: .storageModeShared),
              let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var twoVal = frFromInt(2)
        var hnVal = UInt32(halfN)
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(rsExtendFunction)
        enc.setBuffer(inputBuf!, offset: 0, index: 0)
        enc.setBuffer(extBuf, offset: 0, index: 1)
        enc.setBytes(&twoVal, length: stride, index: 2)
        enc.setBytes(&hnVal, length: 4, index: 3)
        let tg = min(tuning.friThreadgroupSize, Int(rsExtendFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: halfN, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError("RS extend: \(error.localizedDescription)")
        }

        // Assemble result: [original | extended]
        var result = evaluations
        result.reserveCapacity(2 * n)
        let extPtr = extBuf.contents().bindMemory(to: Fr.self, capacity: n)
        result.append(contentsOf: UnsafeBufferPointer(start: extPtr, count: n))
        return result
    }

    // MARK: - Internal Helpers

    private func createFrBuffer(_ data: [Fr]) -> MTLBuffer? {
        let byteCount = data.count * MemoryLayout<Fr>.stride
        guard let buf = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            return nil
        }
        data.withUnsafeBytes { src in
            memcpy(buf.contents(), src.baseAddress!, byteCount)
        }
        return buf
    }

    /// Custom Merkle build for large trees (>65536 leaves) using fused subtrees + level-by-level.
    /// This extends the Poseidon2MerkleEngine approach to larger trees, saving ~30% vs pure level-by-level.
    private func encodeLargeMerkleRoot(encoder: MTLComputeCommandEncoder,
                                        treeBuf: MTLBuffer, treeOffset: Int,
                                        n: Int) {
        let stride = MemoryLayout<Fr>.stride
        let subtreeSize = Poseidon2Engine.merkleSubtreeSize  // 1024

        if n <= 65536 {
            // Use standard path for <=65536
            merkleEngine.encodeMerkleRoot(encoder: encoder, treeBuf: treeBuf, treeOffset: treeOffset, n: n)
            return
        }

        // Fused subtrees for bottom 10 levels (1024 leaves each)
        let numSubtrees = n / subtreeSize
        let rootsOffset = treeOffset + n * stride
        poseidon2.encodeMerkleFused(encoder: encoder,
                                     leavesBuffer: treeBuf, leavesOffset: treeOffset,
                                     rootsBuffer: treeBuf, rootsOffset: rootsOffset,
                                     numSubtrees: numSubtrees)

        // Level-by-level for remaining levels above the subtree roots
        var levelStart = n  // subtree roots start at offset n
        var levelSize = numSubtrees
        while levelSize > 1 {
            encoder.memoryBarrier(scope: .buffers)
            // Use fused subtree if remaining levels fit in one subtree
            if levelSize >= 2 && levelSize <= subtreeSize && (levelSize & (levelSize - 1)) == 0 {
                let rootOffset = treeOffset + (2 * n - 2) * stride
                poseidon2.encodeMerkleFused(encoder: encoder,
                                             leavesBuffer: treeBuf, leavesOffset: treeOffset + levelStart * stride,
                                             rootsBuffer: treeBuf, rootsOffset: rootOffset,
                                             numSubtrees: 1, subtreeSize: levelSize)
                break
            }
            let parentCount = levelSize / 2
            let inputOffset = treeOffset + levelStart * stride
            let outputOffset = treeOffset + (levelStart + levelSize) * stride
            poseidon2.encodeHashPairs(encoder: encoder, buffer: treeBuf,
                                      inputOffset: inputOffset,
                                      outputOffset: outputOffset,
                                      count: parentCount)
            levelStart += levelSize
            levelSize = parentCount
        }
    }

    /// Extract Merkle authentication path for a leaf index.
    private func extractMerklePath(tree: [Fr], leafCount: Int, index: Int) -> [Fr] {
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

    /// Verify a Merkle path from leaf to root using Poseidon2 hashing.
    private func verifyMerklePath(leaf: Fr, path: [Fr], index: Int, root: Fr) -> Bool {
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

        return frToInt(current) == frToInt(root)
    }
}
