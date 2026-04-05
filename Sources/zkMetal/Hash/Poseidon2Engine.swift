// Poseidon2 GPU Engine — batch Poseidon2 hashing on Metal
import Foundation
import Metal

public class Poseidon2Engine {
    public static let version = Versions.poseidon2
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    let permuteFunction: MTLComputePipelineState
    let hashPairsFunction: MTLComputePipelineState
    let merkleFusedFunction: MTLComputePipelineState
    let merkleFusedFullFunction: MTLComputePipelineState
    let merkleFusedBatchFunction: MTLComputePipelineState
    let merkleUpdateScatteredFunction: MTLComputePipelineState
    public let rcBuffer: MTLBuffer  // round constants in Montgomery form

    // Cached buffers for hashPairs to avoid per-call allocation
    private var cachedInputBuf: MTLBuffer?
    private var cachedOutputBuf: MTLBuffer?
    private var cachedBufPairs: Int = 0
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

        let library = try Poseidon2Engine.compileShaders(device: device)

        guard let permuteFn = library.makeFunction(name: "poseidon2_permute"),
              let hashPairsFn = library.makeFunction(name: "poseidon2_hash_pairs"),
              let merkleFusedFn = library.makeFunction(name: "poseidon2_merkle_fused"),
              let merkleFusedFullFn = library.makeFunction(name: "poseidon2_merkle_fused_full"),
              let merkleFusedBatchFn = library.makeFunction(name: "poseidon2_merkle_fused_batch"),
              let merkleUpdateScatteredFn = library.makeFunction(name: "poseidon2_merkle_update_scattered") else {
            throw MSMError.missingKernel
        }

        self.permuteFunction = try device.makeComputePipelineState(function: permuteFn)
        self.hashPairsFunction = try device.makeComputePipelineState(function: hashPairsFn)
        self.merkleFusedFunction = try device.makeComputePipelineState(function: merkleFusedFn)
        self.merkleFusedFullFunction = try device.makeComputePipelineState(function: merkleFusedFullFn)
        self.merkleFusedBatchFunction = try device.makeComputePipelineState(function: merkleFusedBatchFn)
        self.merkleUpdateScatteredFunction = try device.makeComputePipelineState(function: merkleUpdateScatteredFn)

        // Create round constants buffer (64 rounds * 3 elements = 192 Fr values)
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
        self.rcBuffer = buf
        self.tuning = TuningManager.shared.config(device: device)
    }

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let frSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let p2Source = try String(contentsOfFile: shaderDir + "/hash/poseidon2.metal", encoding: .utf8)

        let cleanP2 = p2Source.split(separator: "\n")
            .filter { !$0.contains("#include") }
            .joined(separator: "\n")

        let frClean = frSource
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")

        let combined = frClean + "\n" + cleanP2

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    // Uses global findShaderDir() from zkMetal.swift

    /// Batch hash pairs of field elements on GPU.
    /// Input: array of 2n Fr elements (pairs [a0,b0, a1,b1, ...])
    /// Output: array of n Fr elements (hashes)
    public func hashPairs(_ input: [Fr]) throws -> [Fr] {
        precondition(input.count % 2 == 0, "Input must have even number of elements")
        let n = input.count / 2
        let stride = MemoryLayout<Fr>.stride

        // Reuse cached buffers when possible
        if n > cachedBufPairs {
            guard let inBuf = device.makeBuffer(length: input.count * stride, options: .storageModeShared),
                  let outBuf = device.makeBuffer(length: n * stride, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to allocate buffers")
            }
            cachedInputBuf = inBuf
            cachedOutputBuf = outBuf
            cachedBufPairs = n
        }

        let inputBuf = cachedInputBuf!
        let outputBuf = cachedOutputBuf!
        let inputBytes = input.count * stride
        input.withUnsafeBytes { src in
            memcpy(inputBuf.contents(), src.baseAddress!, inputBytes)
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(hashPairsFunction)
        enc.setBuffer(inputBuf, offset: 0, index: 0)
        enc.setBuffer(outputBuf, offset: 0, index: 1)
        enc.setBuffer(rcBuffer, offset: 0, index: 2)
        var count = UInt32(n)
        enc.setBytes(&count, length: 4, index: 3)
        let tg = min(tuning.hashThreadgroupSize, Int(hashPairsFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = outputBuf.contents().bindMemory(to: Fr.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    /// Batch hash pairs on pre-allocated GPU buffers (zero-copy).
    public func hashPairs(input: MTLBuffer, output: MTLBuffer, count: Int) throws {
        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(hashPairsFunction)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(output, offset: 0, index: 1)
        enc.setBuffer(rcBuffer, offset: 0, index: 2)
        var n = UInt32(count)
        enc.setBytes(&n, length: 4, index: 3)
        let tg = min(tuning.hashThreadgroupSize, Int(hashPairsFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    /// Encode hash pairs dispatch into an existing compute encoder (for batched Merkle).
    /// Input buffer at inputOffset contains 2*count Fr elements; output at outputOffset receives count Fr elements.
    public func encodeHashPairs(encoder: MTLComputeCommandEncoder,
                                 buffer: MTLBuffer, inputOffset: Int,
                                 outputOffset: Int, count: Int) {
        encoder.setComputePipelineState(hashPairsFunction)
        encoder.setBuffer(buffer, offset: inputOffset, index: 0)
        encoder.setBuffer(buffer, offset: outputOffset, index: 1)
        encoder.setBuffer(rcBuffer, offset: 0, index: 2)
        var n = UInt32(count)
        encoder.setBytes(&n, length: 4, index: 3)
        let tg = min(tuning.hashThreadgroupSize, Int(hashPairsFunction.maxTotalThreadsPerThreadgroup))
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
    }

    /// Encode fused Merkle subtree: variable-size subtrees in shared memory.
    /// subtreeSize must be power of 2, max 1024. Default 1024.
    public func encodeMerkleFused(encoder: MTLComputeCommandEncoder,
                                   leavesBuffer: MTLBuffer, leavesOffset: Int,
                                   rootsBuffer: MTLBuffer, rootsOffset: Int,
                                   numSubtrees: Int, subtreeSize: Int = 1024) {
        encoder.setComputePipelineState(merkleFusedFunction)
        encoder.setBuffer(leavesBuffer, offset: leavesOffset, index: 0)
        encoder.setBuffer(rootsBuffer, offset: rootsOffset, index: 1)
        encoder.setBuffer(rcBuffer, offset: 0, index: 2)
        var numLevels = UInt32(subtreeSize.trailingZeroBitCount)
        encoder.setBytes(&numLevels, length: 4, index: 3)
        let tgSize = min(subtreeSize / 2, 512)
        encoder.dispatchThreadgroups(MTLSize(width: numSubtrees, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: max(tgSize, 1), height: 1, depth: 1))
    }

    /// Encode fused Merkle (full tree): writes all intermediate nodes at correct tree offsets.
    /// levelOffsets buffer contains the start index in the tree for each of the 10 internal levels.
    public func encodeMerkleFusedFull(encoder: MTLComputeCommandEncoder,
                                       leavesBuffer: MTLBuffer, leavesOffset: Int,
                                       treeBuffer: MTLBuffer, treeOffset: Int,
                                       levelOffsetsBuffer: MTLBuffer,
                                       numSubtrees: Int) {
        encoder.setComputePipelineState(merkleFusedFullFunction)
        encoder.setBuffer(leavesBuffer, offset: leavesOffset, index: 0)
        encoder.setBuffer(treeBuffer, offset: treeOffset, index: 1)
        encoder.setBuffer(rcBuffer, offset: 0, index: 2)
        var numLevels: UInt32 = 10  // log2(1024) = 10
        encoder.setBytes(&numLevels, length: 4, index: 3)
        encoder.setBuffer(levelOffsetsBuffer, offset: 0, index: 4)
        encoder.dispatchThreadgroups(MTLSize(width: numSubtrees, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
    }

    /// Encode batch fused Merkle: each threadgroup processes an independent tree of variable size.
    /// treeParams buffer: [leaf_offset_0, num_levels_0, leaf_offset_1, num_levels_1, ...]
    /// All leaves in leavesBuffer, results in rootsBuffer[tgid].
    public func encodeMerkleFusedBatch(encoder: MTLComputeCommandEncoder,
                                        leavesBuffer: MTLBuffer,
                                        rootsBuffer: MTLBuffer, rootsOffset: Int,
                                        rcBuffer: MTLBuffer,
                                        treeParamsBuffer: MTLBuffer,
                                        numTrees: Int) {
        encoder.setComputePipelineState(merkleFusedBatchFunction)
        encoder.setBuffer(leavesBuffer, offset: 0, index: 0)
        encoder.setBuffer(rootsBuffer, offset: rootsOffset, index: 1)
        encoder.setBuffer(self.rcBuffer, offset: 0, index: 2)
        encoder.setBuffer(treeParamsBuffer, offset: 0, index: 3)
        // Max threadgroup size is 512 (half of 1024 max subtree)
        let tg = min(512, Int(merkleFusedBatchFunction.maxTotalThreadsPerThreadgroup))
        encoder.dispatchThreadgroups(MTLSize(width: numTrees, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
    }

    /// Encode scattered incremental Merkle update into an existing compute encoder.
    /// For each dirty parent index in dirtyIndicesBuf, reads children from the 1-indexed heap
    /// in treeBuf, hashes them with Poseidon2, and writes the result back to the parent slot.
    /// This eliminates CPU gather/scatter overhead for scattered updates.
    public func encodeScatteredUpdate(encoder: MTLComputeCommandEncoder,
                                      treeBuf: MTLBuffer,
                                      dirtyIndicesBuf: MTLBuffer,
                                      count: Int) {
        encoder.setComputePipelineState(merkleUpdateScatteredFunction)
        encoder.setBuffer(treeBuf, offset: 0, index: 0)
        encoder.setBuffer(rcBuffer, offset: 0, index: 1)
        encoder.setBuffer(dirtyIndicesBuf, offset: 0, index: 2)
        var n = UInt32(count)
        encoder.setBytes(&n, length: 4, index: 3)
        let tg = min(tuning.hashThreadgroupSize, Int(merkleUpdateScatteredFunction.maxTotalThreadsPerThreadgroup))
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
    }

    /// Subtree size for fused Merkle kernel
    public static let merkleSubtreeSize = 1024

    /// Batch permute: apply Poseidon2 permutation to N independent states on GPU.
    /// Input: array of N*3 Fr elements (N states of 3 elements each).
    /// Output: array of N*3 Fr elements (permuted states).
    public func batchPermute(_ states: [Fr]) throws -> [Fr] {
        precondition(states.count % 3 == 0, "States must have a multiple of 3 elements")
        let n = states.count / 3
        let stride = MemoryLayout<Fr>.stride
        let bytes = states.count * stride

        guard let inBuf = device.makeBuffer(length: bytes, options: .storageModeShared),
              let outBuf = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate batch permute buffers")
        }

        states.withUnsafeBytes { src in
            memcpy(inBuf.contents(), src.baseAddress!, bytes)
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(permuteFunction)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBuffer(rcBuffer, offset: 0, index: 2)
        var count = UInt32(n)
        enc.setBytes(&count, length: 4, index: 3)
        let tg = min(tuning.hashThreadgroupSize, Int(permuteFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = outBuf.contents().bindMemory(to: Fr.self, capacity: states.count)
        return Array(UnsafeBufferPointer(start: ptr, count: states.count))
    }
}
