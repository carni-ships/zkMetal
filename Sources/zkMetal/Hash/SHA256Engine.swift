// SHA-256 GPU Engine — batch SHA-256 hashing on Metal
import Foundation
import Metal

public class SHA256Engine {
    public static let version = Versions.sha256
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    let hashBatchFunction: MTLComputePipelineState   // hash 64-byte inputs
    let hashPairsFunction: MTLComputePipelineState   // hash pairs of 32-byte values
    let merkleFusedFunction: MTLComputePipelineState // fused 1024-leaf subtree

    // Cached buffers (grow-only pattern)
    private var cachedH64InputBuf: MTLBuffer?
    private var cachedH64OutputBuf: MTLBuffer?
    private var cachedH64Count: Int = 0
    private let tuning: TuningConfig

    public static let merkleSubtreeSize = 1024

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        let library = try SHA256Engine.compileShaders(device: device)
        guard let hashBatchFn = library.makeFunction(name: "sha256_hash_batch"),
              let hashPairsFn = library.makeFunction(name: "sha256_hash_pairs"),
              let merkleFusedFn = library.makeFunction(name: "sha256_merkle_fused") else {
            throw MSMError.missingKernel
        }
        self.hashBatchFunction = try device.makeComputePipelineState(function: hashBatchFn)
        self.hashPairsFunction = try device.makeComputePipelineState(function: hashPairsFn)
        self.merkleFusedFunction = try device.makeComputePipelineState(function: merkleFusedFn)
        self.tuning = TuningManager.shared.config(device: device)
    }

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let source = try String(contentsOfFile: shaderDir + "/hash/sha256.metal", encoding: .utf8)
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: source, options: options)
    }

    /// Batch SHA-256 hash of 64-byte inputs.
    /// Input: n * 64 bytes, Output: n * 32 bytes
    public func hashBatch(_ input: [UInt8], messageSize: Int = 64) throws -> [UInt8] {
        precondition(messageSize == 64, "SHA256Engine currently supports 64-byte inputs only")
        precondition(input.count % 64 == 0)
        let n = input.count / 64

        // Reuse cached buffers when possible (grow-only)
        if n > cachedH64Count {
            guard let inBuf = device.makeBuffer(length: n * 64, options: .storageModeShared),
                  let outBuf = device.makeBuffer(length: n * 32, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to allocate buffers")
            }
            cachedH64InputBuf = inBuf
            cachedH64OutputBuf = outBuf
            cachedH64Count = n
        }

        let inputBuf = cachedH64InputBuf!
        let outputBuf = cachedH64OutputBuf!
        input.withUnsafeBytes { src in
            memcpy(inputBuf.contents(), src.baseAddress!, input.count)
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(hashBatchFunction)
        enc.setBuffer(inputBuf, offset: 0, index: 0)
        enc.setBuffer(outputBuf, offset: 0, index: 1)
        var countVal = UInt32(n)
        enc.setBytes(&countVal, length: 4, index: 2)
        let tg = min(tuning.hashThreadgroupSize, Int(hashBatchFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = outputBuf.contents().bindMemory(to: UInt8.self, capacity: n * 32)
        return Array(UnsafeBufferPointer(start: ptr, count: n * 32))
    }

    /// Hash pairs of 32-byte values (for Merkle trees).
    /// Input: n * 64 bytes (n pairs of 32-byte left || right), Output: n * 32 bytes
    public func hashPairs(_ input: [UInt8]) throws -> [UInt8] {
        precondition(input.count % 64 == 0)
        let n = input.count / 64

        if n > cachedH64Count {
            guard let inBuf = device.makeBuffer(length: n * 64, options: .storageModeShared),
                  let outBuf = device.makeBuffer(length: n * 32, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to allocate buffers")
            }
            cachedH64InputBuf = inBuf
            cachedH64OutputBuf = outBuf
            cachedH64Count = n
        }

        let inputBuf = cachedH64InputBuf!
        let outputBuf = cachedH64OutputBuf!
        input.withUnsafeBytes { src in
            memcpy(inputBuf.contents(), src.baseAddress!, input.count)
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(hashPairsFunction)
        enc.setBuffer(inputBuf, offset: 0, index: 0)
        enc.setBuffer(outputBuf, offset: 0, index: 1)
        var countVal = UInt32(n)
        enc.setBytes(&countVal, length: 4, index: 2)
        let tg = min(tuning.hashThreadgroupSize, Int(hashPairsFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = outputBuf.contents().bindMemory(to: UInt8.self, capacity: n * 32)
        return Array(UnsafeBufferPointer(start: ptr, count: n * 32))
    }

    /// Batch hash on GPU buffers (zero-copy).
    public func hashBatch(input: MTLBuffer, output: MTLBuffer, count: Int) throws {
        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(hashBatchFunction)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(output, offset: 0, index: 1)
        var n = UInt32(count)
        enc.setBytes(&n, length: 4, index: 2)
        let tg = min(tuning.hashThreadgroupSize, Int(hashBatchFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    /// Encode fused Merkle subtree dispatch.
    /// Reads 32-byte leaves from leavesBuffer, writes one 32-byte root per subtree to rootsBuffer.
    /// subtreeSize must be a power of 2, max 1024. Default 1024.
    public func encodeMerkleFused(encoder: MTLComputeCommandEncoder,
                                   leavesBuffer: MTLBuffer, leavesOffset: Int,
                                   rootsBuffer: MTLBuffer, rootsOffset: Int,
                                   numSubtrees: Int, subtreeSize: Int = 1024) {
        encoder.setComputePipelineState(merkleFusedFunction)
        encoder.setBuffer(leavesBuffer, offset: leavesOffset, index: 0)
        encoder.setBuffer(rootsBuffer, offset: rootsOffset, index: 1)
        var numLevels = UInt32(subtreeSize.trailingZeroBitCount)
        encoder.setBytes(&numLevels, length: 4, index: 2)
        let tgSize = min(subtreeSize / 2, 512)
        encoder.dispatchThreadgroups(MTLSize(width: numSubtrees, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
    }

    /// Encode hash-pairs into an existing compute encoder (for Merkle tree construction).
    /// Hashes `count` pairs from buffer at inputOffset, writes to outputOffset.
    public func encodeHashPairs(encoder: MTLComputeCommandEncoder, buffer: MTLBuffer,
                                inputOffset: Int, outputOffset: Int, count: Int) {
        encoder.setComputePipelineState(hashPairsFunction)
        encoder.setBuffer(buffer, offset: inputOffset, index: 0)
        encoder.setBuffer(buffer, offset: outputOffset, index: 1)
        var n = UInt32(count)
        encoder.setBytes(&n, length: 4, index: 2)
        let tg = min(tuning.hashThreadgroupSize, Int(hashPairsFunction.maxTotalThreadsPerThreadgroup))
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
    }
}

// MARK: - SHA-256 Merkle Tree

public class SHA256MerkleEngine {
    public static let version = Versions.sha256Merkle
    public let engine: SHA256Engine
    private var cachedTreeBuf: MTLBuffer?
    private var cachedTreeBufNodes: Int = 0

    public init() throws {
        self.engine = try SHA256Engine()
    }

    /// Build a Merkle tree from 32-byte leaf hashes using SHA-256.
    /// Returns flat buffer: treeSize * 32 bytes. Node i is at bytes [i*32..<(i+1)*32].
    /// Layout: nodes[0..<n] = leaves, nodes[n..<2n-1] = internal, node[2n-2] = root.
    public func buildTree(_ leaves: [[UInt8]]) throws -> [UInt8] {
        let n = leaves.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Leaf count must be power of 2")
        precondition(leaves.allSatisfy { $0.count == 32 }, "Leaves must be 32 bytes each")

        let treeSize = 2 * n - 1

        if treeSize > cachedTreeBufNodes {
            guard let buf = engine.device.makeBuffer(length: treeSize * 32, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to allocate SHA-256 Merkle buffer")
            }
            cachedTreeBuf = buf
            cachedTreeBufNodes = treeSize
        }
        let treeBuf = cachedTreeBuf!

        // Copy leaves to GPU buffer
        let ptr = treeBuf.contents().assumingMemoryBound(to: UInt8.self)
        for i in 0..<n {
            leaves[i].withUnsafeBytes { src in
                memcpy(ptr + i * 32, src.baseAddress!, 32)
            }
        }

        guard let cmdBuf = engine.commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!

        // Level-by-level construction (preserves ALL internal nodes for proof extraction).
        // Note: fused subtrees only output roots, not intermediate nodes, so they
        // cannot be used here. Fused subtrees ARE used in encodeMerkleRoot.
        var levelStart = 0
        var levelSize = n

        while levelSize > 1 {
            let parentCount = levelSize / 2
            let inputOffset = levelStart * 32
            let outputOffset = (levelStart + levelSize) * 32

            engine.encodeHashPairs(encoder: enc, buffer: treeBuf,
                                   inputOffset: inputOffset,
                                   outputOffset: outputOffset,
                                   count: parentCount)

            levelStart += levelSize
            levelSize = parentCount

            if levelSize > 1 {
                enc.memoryBarrier(scope: .buffers)
            }
        }
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let readPtr = treeBuf.contents().assumingMemoryBound(to: UInt8.self)
        return Array(UnsafeBufferPointer(start: readPtr, count: treeSize * 32))
    }

    /// Access a specific node from a flat tree buffer returned by buildTree.
    public static func node(_ tree: [UInt8], at index: Int) -> [UInt8] {
        let start = index * 32
        return Array(tree[start..<start + 32])
    }

    /// Get the Merkle root from 32-byte leaf hashes.
    public func merkleRoot(_ leaves: [[UInt8]]) throws -> [UInt8] {
        let tree = try buildTree(leaves)
        let treeSize = tree.count / 32
        return SHA256MerkleEngine.node(tree, at: treeSize - 1)
    }
}

// MARK: - CPU SHA-256 Reference

/// CPU SHA-256 reference implementation for testing.
public func sha256(_ input: [UInt8]) -> [UInt8] {
    let K: [UInt32] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
        0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    ]

    func rotr(_ x: UInt32, _ n: Int) -> UInt32 {
        return (x >> n) | (x << (32 - n))
    }

    // Pre-processing: padding
    var msg = input
    let originalLen = input.count
    msg.append(0x80)
    while msg.count % 64 != 56 {
        msg.append(0)
    }
    // Append length in bits as 64-bit big-endian
    let bitLen = UInt64(originalLen) * 8
    for i in stride(from: 56, through: 0, by: -8) {
        msg.append(UInt8((bitLen >> i) & 0xFF))
    }

    // Initialize hash values
    var h0: UInt32 = 0x6a09e667
    var h1: UInt32 = 0xbb67ae85
    var h2: UInt32 = 0x3c6ef372
    var h3: UInt32 = 0xa54ff53a
    var h4: UInt32 = 0x510e527f
    var h5: UInt32 = 0x9b05688c
    var h6: UInt32 = 0x1f83d9ab
    var h7: UInt32 = 0x5be0cd19

    // Process each 64-byte block
    var offset = 0
    while offset < msg.count {
        var W = [UInt32](repeating: 0, count: 64)
        // Load 16 words (big-endian)
        for i in 0..<16 {
            let j = offset + i * 4
            W[i] = (UInt32(msg[j]) << 24) | (UInt32(msg[j+1]) << 16) |
                   (UInt32(msg[j+2]) << 8) | UInt32(msg[j+3])
        }
        // Extend to 64 words
        for i in 16..<64 {
            let s0 = rotr(W[i-15], 7) ^ rotr(W[i-15], 18) ^ (W[i-15] >> 3)
            let s1 = rotr(W[i-2], 17) ^ rotr(W[i-2], 19) ^ (W[i-2] >> 10)
            W[i] = W[i-16] &+ s0 &+ W[i-7] &+ s1
        }

        var a = h0, b = h1, c = h2, d = h3
        var e = h4, f = h5, g = h6, h = h7

        for i in 0..<64 {
            let S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
            let ch = (e & f) ^ (~e & g)
            let temp1 = h &+ S1 &+ ch &+ K[i] &+ W[i]
            let S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
            let maj = (a & b) ^ (a & c) ^ (b & c)
            let temp2 = S0 &+ maj

            h = g; g = f; f = e; e = d &+ temp1
            d = c; c = b; b = a; a = temp1 &+ temp2
        }

        h0 = h0 &+ a; h1 = h1 &+ b; h2 = h2 &+ c; h3 = h3 &+ d
        h4 = h4 &+ e; h5 = h5 &+ f; h6 = h6 &+ g; h7 = h7 &+ h

        offset += 64
    }

    // Output big-endian
    var result = [UInt8](repeating: 0, count: 32)
    let hvals: [UInt32] = [h0, h1, h2, h3, h4, h5, h6, h7]
    for i in 0..<8 {
        result[i*4]   = UInt8((hvals[i] >> 24) & 0xFF)
        result[i*4+1] = UInt8((hvals[i] >> 16) & 0xFF)
        result[i*4+2] = UInt8((hvals[i] >> 8) & 0xFF)
        result[i*4+3] = UInt8(hvals[i] & 0xFF)
    }
    return result
}
