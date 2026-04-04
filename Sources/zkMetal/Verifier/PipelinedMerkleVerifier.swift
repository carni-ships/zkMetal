// Pipelined Merkle Path Verification — GPU batch verification via unified memory
// Exploits Apple Silicon shared memory: CPU fills buffers, GPU reads directly.
// No explicit data transfers needed.

import Foundation
import Metal

public class PipelinedMerkleVerifier {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    let verifyFunction: MTLComputePipelineState
    let rcBuffer: MTLBuffer  // Poseidon2 round constants

    // Cached buffers for batch verification
    private var cachedLeavesBuf: MTLBuffer?
    private var cachedPathsBuf: MTLBuffer?
    private var cachedIndicesBuf: MTLBuffer?
    private var cachedRootsBuf: MTLBuffer?
    private var cachedDepthsBuf: MTLBuffer?
    private var cachedResultsBuf: MTLBuffer?
    private var cachedCount: Int = 0
    private var cachedMaxDepth: Int = 0

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

        let library = try PipelinedMerkleVerifier.compileShaders(device: device)

        guard let verifyFn = library.makeFunction(name: "batch_merkle_verify_poseidon2") else {
            throw MSMError.missingKernel
        }
        self.verifyFunction = try device.makeComputePipelineState(function: verifyFn)

        // Load Poseidon2 round constants (same as Poseidon2Engine)
        self.rcBuffer = try PipelinedMerkleVerifier.loadRoundConstants(device: device)
        self.tuning = TuningManager.shared.config(device: device)
    }

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let frSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let verifySource = try String(contentsOfFile: shaderDir + "/verify/merkle_verify.metal", encoding: .utf8)

        let cleanVerify = verifySource.split(separator: "\n")
            .filter { !$0.contains("#include") }
            .joined(separator: "\n")
        let cleanFr = frSource
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")

        let combined = cleanFr + "\n" + cleanVerify
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    private static func loadRoundConstants(device: MTLDevice) throws -> MTLBuffer {
        // Poseidon2 BN254 round constants: 64 rounds * 3 = 192 Fr elements
        // Same constants as Poseidon2Engine
        let rc = POSEIDON2_ROUND_CONSTANTS
        var flatRC = [Fr]()
        flatRC.reserveCapacity(192)
        for round in rc {
            for elem in round {
                flatRC.append(elem)
            }
        }
        let byteCount = flatRC.count * MemoryLayout<Fr>.stride
        guard let buf = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate RC buffer")
        }
        flatRC.withUnsafeBytes { src in
            memcpy(buf.contents(), src.baseAddress!, byteCount)
        }
        return buf
    }

    /// Ensure cached buffers can hold `count` paths of `maxDepth`.
    private func ensureBuffers(count: Int, maxDepth: Int) throws {
        if count <= cachedCount && maxDepth <= cachedMaxDepth { return }

        let frStride = MemoryLayout<Fr>.stride
        let actualCount = max(count, cachedCount)
        let actualDepth = max(maxDepth, cachedMaxDepth)

        cachedLeavesBuf = device.makeBuffer(length: actualCount * frStride, options: .storageModeShared)
        cachedPathsBuf = device.makeBuffer(length: actualCount * actualDepth * frStride, options: .storageModeShared)
        cachedIndicesBuf = device.makeBuffer(length: actualCount * 4, options: .storageModeShared)
        cachedRootsBuf = device.makeBuffer(length: actualCount * frStride, options: .storageModeShared)
        cachedDepthsBuf = device.makeBuffer(length: actualCount * 4, options: .storageModeShared)
        cachedResultsBuf = device.makeBuffer(length: actualCount * 4, options: .storageModeShared)

        guard cachedLeavesBuf != nil && cachedPathsBuf != nil && cachedIndicesBuf != nil &&
              cachedRootsBuf != nil && cachedDepthsBuf != nil && cachedResultsBuf != nil else {
            throw MSMError.gpuError("Failed to allocate Merkle verification buffers")
        }

        cachedCount = actualCount
        cachedMaxDepth = actualDepth
    }

    /// Batch-verify multiple Merkle paths on GPU.
    /// Each entry: (leaf, index, path siblings, expected root).
    /// Returns array of bools indicating which paths verified.
    ///
    /// Zero-copy on unified memory: CPU writes leaf/path/root data into shared MTLBuffers,
    /// GPU reads the same physical memory to hash and verify.
    public func batchVerify(
        leaves: [Fr],
        indices: [UInt32],
        paths: [[Fr]],       // paths[i] = siblings for path i
        roots: [Fr],
        maxDepth: Int
    ) throws -> [Bool] {
        let count = leaves.count
        precondition(count == indices.count && count == paths.count && count == roots.count)
        if count == 0 { return [] }

        try ensureBuffers(count: count, maxDepth: maxDepth)

        let frStride = MemoryLayout<Fr>.stride
        let leavesBuf = cachedLeavesBuf!
        let pathsBuf = cachedPathsBuf!
        let indicesBuf = cachedIndicesBuf!
        let rootsBuf = cachedRootsBuf!
        let depthsBuf = cachedDepthsBuf!
        let resultsBuf = cachedResultsBuf!

        // Fill leaves (zero-copy: CPU writes, GPU reads from same physical memory)
        leaves.withUnsafeBytes { src in
            memcpy(leavesBuf.contents(), src.baseAddress!, count * frStride)
        }

        // Fill paths: packed as [path0_level0, path0_level1, ..., pad, path1_level0, ...]
        let pathsPtr = pathsBuf.contents().bindMemory(to: Fr.self, capacity: count * maxDepth)
        for i in 0..<count {
            let path = paths[i]
            for j in 0..<path.count {
                pathsPtr[i * maxDepth + j] = path[j]
            }
            // Remaining slots left as-is (depth controls traversal)
        }

        // Fill indices
        indices.withUnsafeBytes { src in
            memcpy(indicesBuf.contents(), src.baseAddress!, count * 4)
        }

        // Fill roots
        roots.withUnsafeBytes { src in
            memcpy(rootsBuf.contents(), src.baseAddress!, count * frStride)
        }

        // Fill depths
        var depths = [UInt32]()
        depths.reserveCapacity(count)
        for path in paths {
            depths.append(UInt32(path.count))
        }
        depths.withUnsafeBytes { src in
            memcpy(depthsBuf.contents(), src.baseAddress!, count * 4)
        }

        // Dispatch GPU verification
        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(verifyFunction)
        enc.setBuffer(leavesBuf, offset: 0, index: 0)
        enc.setBuffer(pathsBuf, offset: 0, index: 1)
        enc.setBuffer(indicesBuf, offset: 0, index: 2)
        enc.setBuffer(rootsBuf, offset: 0, index: 3)
        enc.setBuffer(depthsBuf, offset: 0, index: 4)
        enc.setBuffer(resultsBuf, offset: 0, index: 5)
        enc.setBuffer(rcBuffer, offset: 0, index: 6)
        var maxDepthVal = UInt32(maxDepth)
        enc.setBytes(&maxDepthVal, length: 4, index: 7)
        var countVal = UInt32(count)
        enc.setBytes(&countVal, length: 4, index: 8)
        let tg = min(tuning.hashThreadgroupSize, Int(verifyFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        // Read results (zero-copy: GPU wrote, CPU reads same physical memory)
        let resPtr = resultsBuf.contents().bindMemory(to: UInt32.self, capacity: count)
        var results = [Bool]()
        results.reserveCapacity(count)
        for i in 0..<count {
            results.append(resPtr[i] != 0)
        }
        return results
    }

    /// Convenience: verify a single Merkle path on GPU.
    public func verifySingle(leaf: Fr, index: UInt32, path: [Fr], root: Fr) throws -> Bool {
        let results = try batchVerify(
            leaves: [leaf], indices: [index], paths: [path],
            roots: [root], maxDepth: path.count)
        return results[0]
    }

    // MARK: - Double-Buffered Pipeline Interface

    /// Double-buffered batch verification for streaming use.
    /// Returns (leavesBuf, pathsBuf, indicesBuf, rootsBuf, depthsBuf, resultsBuf)
    /// for direct CPU filling. The caller fills one set while GPU processes the other.
    public func makeDoubleBuffers(maxCount: Int, maxDepth: Int) throws -> (
        bufA: MerkleVerifyBufferSet, bufB: MerkleVerifyBufferSet
    ) {
        let a = try MerkleVerifyBufferSet(device: device, maxCount: maxCount, maxDepth: maxDepth)
        let b = try MerkleVerifyBufferSet(device: device, maxCount: maxCount, maxDepth: maxDepth)
        return (a, b)
    }

    /// Encode batch verification into an existing command buffer (for pipelining).
    /// Returns the command buffer for caller to track completion.
    public func encodeBatchVerify(
        buffers: MerkleVerifyBufferSet, count: Int, maxDepth: Int
    ) throws -> MTLCommandBuffer {
        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(verifyFunction)
        enc.setBuffer(buffers.leavesBuf, offset: 0, index: 0)
        enc.setBuffer(buffers.pathsBuf, offset: 0, index: 1)
        enc.setBuffer(buffers.indicesBuf, offset: 0, index: 2)
        enc.setBuffer(buffers.rootsBuf, offset: 0, index: 3)
        enc.setBuffer(buffers.depthsBuf, offset: 0, index: 4)
        enc.setBuffer(buffers.resultsBuf, offset: 0, index: 5)
        enc.setBuffer(rcBuffer, offset: 0, index: 6)
        var maxDepthVal = UInt32(maxDepth)
        enc.setBytes(&maxDepthVal, length: 4, index: 7)
        var countVal = UInt32(count)
        enc.setBytes(&countVal, length: 4, index: 8)
        let tg = min(tuning.hashThreadgroupSize, Int(verifyFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        return cmdBuf
    }
}

// MARK: - Buffer Set for Double-Buffering

/// A set of GPU buffers for one side of the double-buffer pipeline.
/// On unified memory, CPU writes directly into these; GPU reads the same physical memory.
public class MerkleVerifyBufferSet {
    public let leavesBuf: MTLBuffer
    public let pathsBuf: MTLBuffer
    public let indicesBuf: MTLBuffer
    public let rootsBuf: MTLBuffer
    public let depthsBuf: MTLBuffer
    public let resultsBuf: MTLBuffer
    public let maxCount: Int
    public let maxDepth: Int

    public init(device: MTLDevice, maxCount: Int, maxDepth: Int) throws {
        let frStride = MemoryLayout<Fr>.stride
        guard let leaves = device.makeBuffer(length: maxCount * frStride, options: .storageModeShared),
              let paths = device.makeBuffer(length: maxCount * maxDepth * frStride, options: .storageModeShared),
              let indices = device.makeBuffer(length: maxCount * 4, options: .storageModeShared),
              let roots = device.makeBuffer(length: maxCount * frStride, options: .storageModeShared),
              let depths = device.makeBuffer(length: maxCount * 4, options: .storageModeShared),
              let results = device.makeBuffer(length: maxCount * 4, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate MerkleVerifyBufferSet")
        }
        self.leavesBuf = leaves
        self.pathsBuf = paths
        self.indicesBuf = indices
        self.rootsBuf = roots
        self.depthsBuf = depths
        self.resultsBuf = results
        self.maxCount = maxCount
        self.maxDepth = maxDepth
    }

    /// Fill one path entry. Zero-copy on unified memory.
    public func fillEntry(index: Int, leaf: Fr, leafIndex: UInt32, path: [Fr], root: Fr) {
        let frStride = MemoryLayout<Fr>.stride

        let leavesPtr = leavesBuf.contents().bindMemory(to: Fr.self, capacity: maxCount)
        leavesPtr[index] = leaf

        let pathsPtr = pathsBuf.contents().bindMemory(to: Fr.self, capacity: maxCount * maxDepth)
        for j in 0..<path.count {
            pathsPtr[index * maxDepth + j] = path[j]
        }

        let indicesPtr = indicesBuf.contents().bindMemory(to: UInt32.self, capacity: maxCount)
        indicesPtr[index] = leafIndex

        let rootsPtr = rootsBuf.contents().bindMemory(to: Fr.self, capacity: maxCount)
        rootsPtr[index] = root

        let depthsPtr = depthsBuf.contents().bindMemory(to: UInt32.self, capacity: maxCount)
        depthsPtr[index] = UInt32(path.count)
    }

    /// Read verification result for entry i.
    public func result(at index: Int) -> Bool {
        let ptr = resultsBuf.contents().bindMemory(to: UInt32.self, capacity: maxCount)
        return ptr[index] != 0
    }
}
