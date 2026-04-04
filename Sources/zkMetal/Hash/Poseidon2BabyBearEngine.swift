// Poseidon2 BabyBear GPU Engine — batch Poseidon2 hashing over BabyBear on Metal
// t=16, rate=8, capacity=8, x^7 S-box, 8 full + 13 partial rounds
// Parameters match SP1/Plonky3 exactly.
// Each hash node is 8 BabyBear elements (32 bytes).
import Foundation
import Metal

public class Poseidon2BabyBearEngine {
    public static let version = Versions.poseidon2BabyBear
    public static let nodeSize = 8  // BabyBear elements per Merkle node
    public static let merkleSubtreeSize = 512  // max nodes per fused subtree

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    let permuteFunction: MTLComputePipelineState
    let hashPairsFunction: MTLComputePipelineState
    let merkleFusedFunction: MTLComputePipelineState
    let merkleFusedBatchFunction: MTLComputePipelineState
    let rcBuffer: MTLBuffer

    // Cached buffers (grow-only)
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

        let library = try Poseidon2BabyBearEngine.compileShaders(device: device)

        guard let permuteFn = library.makeFunction(name: "poseidon2_bb_permute"),
              let hashPairsFn = library.makeFunction(name: "poseidon2_bb_hash_pairs"),
              let merkleFusedFn = library.makeFunction(name: "poseidon2_bb_merkle_fused"),
              let merkleFusedBatchFn = library.makeFunction(name: "poseidon2_bb_merkle_fused_batch") else {
            throw MSMError.missingKernel
        }

        self.permuteFunction = try device.makeComputePipelineState(function: permuteFn)
        self.hashPairsFunction = try device.makeComputePipelineState(function: hashPairsFn)
        self.merkleFusedFunction = try device.makeComputePipelineState(function: merkleFusedFn)
        self.merkleFusedBatchFunction = try device.makeComputePipelineState(function: merkleFusedBatchFn)

        // Create round constants buffer: 21 rounds * 16 elements = 336 UInt32 values
        let rc = POSEIDON2_BB_ROUND_CONSTANTS
        var flatRC = [UInt32]()
        flatRC.reserveCapacity(Poseidon2BabyBearConfig.totalRounds * Poseidon2BabyBearConfig.t)
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
        let bbSource = try String(contentsOfFile: shaderDir + "/fields/babybear.metal", encoding: .utf8)
        let p2Source = try String(contentsOfFile: shaderDir + "/hash/poseidon2_babybear.metal", encoding: .utf8)

        let cleanP2 = p2Source.split(separator: "\n")
            .filter { !$0.contains("#include") }
            .joined(separator: "\n")

        let bbClean = bbSource
            .replacingOccurrences(of: "#ifndef BABYBEAR_METAL", with: "")
            .replacingOccurrences(of: "#define BABYBEAR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BABYBEAR_METAL", with: "")

        let combined = bbClean + "\n" + cleanP2

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    /// Batch hash pairs of 8-element BabyBear nodes on GPU.
    public func hashPairs(_ input: [Bb]) throws -> [Bb] {
        let nodeSize = Poseidon2BabyBearEngine.nodeSize
        precondition(input.count % (2 * nodeSize) == 0, "Input must have pairs of 8-element nodes")
        let n = input.count / (2 * nodeSize)
        let stride = MemoryLayout<UInt32>.stride

        if n > cachedBufPairs {
            guard let inBuf = device.makeBuffer(length: input.count * stride, options: .storageModeShared),
                  let outBuf = device.makeBuffer(length: n * nodeSize * stride, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to allocate buffers")
            }
            cachedInputBuf = inBuf
            cachedOutputBuf = outBuf
            cachedBufPairs = n
        }

        let inputBuf = cachedInputBuf!
        let outputBuf = cachedOutputBuf!
        let ptr = inputBuf.contents().bindMemory(to: UInt32.self, capacity: input.count)
        for i in 0..<input.count {
            ptr[i] = input[i].v
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

        let outPtr = outputBuf.contents().bindMemory(to: UInt32.self, capacity: n * nodeSize)
        var result = [Bb](repeating: Bb.zero, count: n * nodeSize)
        for i in 0..<(n * nodeSize) {
            result[i] = Bb(v: outPtr[i])
        }
        return result
    }

    /// Merkle tree commitment: hash leaves down to a single root.
    /// Each leaf is 8 BabyBear elements. Number of leaves must be a power of 2.
    /// Returns 8 BabyBear elements (the root hash).
    public func merkleCommit(leaves: [Bb]) throws -> [Bb] {
        let nodeSize = Poseidon2BabyBearEngine.nodeSize
        let subtreeMax = Poseidon2BabyBearEngine.merkleSubtreeSize
        precondition(leaves.count % nodeSize == 0, "Leaves must be multiple of 8 BabyBear elements")
        let numLeaves = leaves.count / nodeSize
        precondition(numLeaves > 0 && (numLeaves & (numLeaves - 1)) == 0, "Number of leaves must be power of 2")

        if numLeaves == 1 {
            return Array(leaves[0..<nodeSize])
        }

        let stride = MemoryLayout<UInt32>.stride

        // Upload leaves
        let leafBytes = leaves.count * stride
        guard let leafBuf = device.makeBuffer(length: leafBytes, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate leaf buffer")
        }
        let leafPtr = leafBuf.contents().bindMemory(to: UInt32.self, capacity: leaves.count)
        for i in 0..<leaves.count { leafPtr[i] = leaves[i].v }

        // Small tree: use fused kernel directly
        if numLeaves <= subtreeMax {
            let rootBytes = nodeSize * stride
            guard let rootBuf = device.makeBuffer(length: rootBytes, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to allocate root buffer")
            }

            guard let cmdBuf = commandQueue.makeCommandBuffer() else {
                throw MSMError.noCommandBuffer
            }

            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(merkleFusedFunction)
            enc.setBuffer(leafBuf, offset: 0, index: 0)
            enc.setBuffer(rootBuf, offset: 0, index: 1)
            enc.setBuffer(rcBuffer, offset: 0, index: 2)
            var numLevels = UInt32(numLeaves.trailingZeroBitCount)
            enc.setBytes(&numLevels, length: 4, index: 3)
            let tgSize = min(numLeaves / 2, 256)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: max(tgSize, 1), height: 1, depth: 1))
            enc.endEncoding()

            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            if let error = cmdBuf.error {
                throw MSMError.gpuError(error.localizedDescription)
            }

            let outPtr = rootBuf.contents().bindMemory(to: UInt32.self, capacity: nodeSize)
            var result = [Bb](repeating: Bb.zero, count: nodeSize)
            for i in 0..<nodeSize { result[i] = Bb(v: outPtr[i]) }
            return result
        }

        // Large tree: fused subtrees + iterative hash-pairs
        let numSubtrees = numLeaves / subtreeMax
        let subtreeLogN = UInt32(subtreeMax.trailingZeroBitCount)

        let rootsBytes = numSubtrees * nodeSize * stride
        guard let rootsBuf = device.makeBuffer(length: rootsBytes, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate roots buffer")
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        // Phase 1: Fused subtrees
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(merkleFusedFunction)
        enc.setBuffer(leafBuf, offset: 0, index: 0)
        enc.setBuffer(rootsBuf, offset: 0, index: 1)
        enc.setBuffer(rcBuffer, offset: 0, index: 2)
        var numLevels = subtreeLogN
        enc.setBytes(&numLevels, length: 4, index: 3)
        let tgSize = min(subtreeMax / 2, 256)
        enc.dispatchThreadgroups(MTLSize(width: numSubtrees, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        // Phase 2: Iteratively hash remaining levels
        var currentNodes = numSubtrees
        var currentBuf = rootsBuf

        while currentNodes > 1 {
            let pairs = currentNodes / 2
            let outBytes = pairs * nodeSize * stride
            guard let outBuf = device.makeBuffer(length: outBytes, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to allocate level buffer")
            }

            guard let cb = commandQueue.makeCommandBuffer() else {
                throw MSMError.noCommandBuffer
            }

            let e = cb.makeComputeCommandEncoder()!
            e.setComputePipelineState(hashPairsFunction)
            e.setBuffer(currentBuf, offset: 0, index: 0)
            e.setBuffer(outBuf, offset: 0, index: 1)
            e.setBuffer(rcBuffer, offset: 0, index: 2)
            var pairCount = UInt32(pairs)
            e.setBytes(&pairCount, length: 4, index: 3)
            let tg = min(tuning.hashThreadgroupSize, Int(hashPairsFunction.maxTotalThreadsPerThreadgroup))
            e.dispatchThreads(MTLSize(width: pairs, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            e.endEncoding()

            cb.commit()
            cb.waitUntilCompleted()
            if let error = cb.error {
                throw MSMError.gpuError(error.localizedDescription)
            }

            currentBuf = outBuf
            currentNodes = pairs
        }

        let outPtr = currentBuf.contents().bindMemory(to: UInt32.self, capacity: nodeSize)
        var result = [Bb](repeating: Bb.zero, count: nodeSize)
        for i in 0..<nodeSize { result[i] = Bb(v: outPtr[i]) }
        return result
    }

    /// Encode fused Merkle subtree dispatch into an existing encoder.
    public func encodeMerkleFused(encoder: MTLComputeCommandEncoder,
                                   leavesBuffer: MTLBuffer, leavesOffset: Int,
                                   rootsBuffer: MTLBuffer, rootsOffset: Int,
                                   numSubtrees: Int, subtreeSize: Int = 512) {
        encoder.setComputePipelineState(merkleFusedFunction)
        encoder.setBuffer(leavesBuffer, offset: leavesOffset, index: 0)
        encoder.setBuffer(rootsBuffer, offset: rootsOffset, index: 1)
        encoder.setBuffer(rcBuffer, offset: 0, index: 2)
        var numLevels = UInt32(subtreeSize.trailingZeroBitCount)
        encoder.setBytes(&numLevels, length: 4, index: 3)
        let tgSize = min(subtreeSize / 2, 256)
        encoder.dispatchThreadgroups(MTLSize(width: numSubtrees, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: max(tgSize, 1), height: 1, depth: 1))
    }

    /// Encode hash pairs dispatch into an existing encoder.
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
}
