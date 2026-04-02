// Keccak-256 GPU Engine — batch Keccak-256 hashing on Metal
import Foundation
import Metal

public class Keccak256Engine {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    let hash64Function: MTLComputePipelineState   // hash 64-byte inputs
    let hash32Function: MTLComputePipelineState   // hash 32-byte inputs
    let merkleFusedFunction: MTLComputePipelineState  // fused 1024-leaf subtree

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

        let library = try Keccak256Engine.compileShaders(device: device)
        guard let hash64Fn = library.makeFunction(name: "keccak256_hash_64"),
              let hash32Fn = library.makeFunction(name: "keccak256_hash_32"),
              let merkleFusedFn = library.makeFunction(name: "keccak256_merkle_fused") else {
            throw MSMError.missingKernel
        }
        self.hash64Function = try device.makeComputePipelineState(function: hash64Fn)
        self.hash32Function = try device.makeComputePipelineState(function: hash32Fn)
        self.merkleFusedFunction = try device.makeComputePipelineState(function: merkleFusedFn)
    }

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let source = try String(contentsOfFile: shaderDir + "/hash/keccak256.metal", encoding: .utf8)
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: source, options: options)
    }

    private static func findShaderDir() -> String {
        let execPath = CommandLine.arguments[0]
        let execDir = (execPath as NSString).deletingLastPathComponent
        for bundle in Bundle.allBundles {
            if let url = bundle.url(forResource: "Shaders", withExtension: nil) {
                let path = url.appendingPathComponent("hash/keccak256.metal").path
                if FileManager.default.fileExists(atPath: path) { return url.path }
            }
        }
        let candidates = [
            "\(execDir)/../Sources/Shaders",
            "./metal/Sources/Shaders",
            "./Sources/Shaders",
        ]
        for path in candidates {
            if FileManager.default.fileExists(atPath: "\(path)/hash/keccak256.metal") { return path }
        }
        return "./Sources/Shaders"
    }

    /// Batch Keccak-256 hash of 64-byte inputs (e.g., Merkle node hashing: two 32-byte children).
    /// Input: n * 64 bytes, Output: n * 32 bytes
    public func hash64(_ input: [UInt8]) throws -> [UInt8] {
        precondition(input.count % 64 == 0)
        let n = input.count / 64

        guard let inputBuf = device.makeBuffer(bytes: input, length: input.count, options: .storageModeShared),
              let outputBuf = device.makeBuffer(length: n * 32, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate buffers")
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(hash64Function)
        enc.setBuffer(inputBuf, offset: 0, index: 0)
        enc.setBuffer(outputBuf, offset: 0, index: 1)
        var count = UInt32(n)
        enc.setBytes(&count, length: 4, index: 2)
        let tg = min(256, Int(hash64Function.maxTotalThreadsPerThreadgroup))
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

    /// Batch Keccak-256 on GPU buffers (zero-copy).
    public func hash64(input: MTLBuffer, output: MTLBuffer, count: Int) throws {
        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(hash64Function)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(output, offset: 0, index: 1)
        var n = UInt32(count)
        enc.setBytes(&n, length: 4, index: 2)
        let tg = min(256, Int(hash64Function.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    /// Encode fused Merkle subtree dispatch (1024 leaves per threadgroup).
    /// Reads 32-byte leaves from leavesBuffer, writes one 32-byte root per subtree to rootsBuffer.
    public func encodeMerkleFused(encoder: MTLComputeCommandEncoder,
                                   leavesBuffer: MTLBuffer, leavesOffset: Int,
                                   rootsBuffer: MTLBuffer, rootsOffset: Int,
                                   numSubtrees: Int) {
        encoder.setComputePipelineState(merkleFusedFunction)
        encoder.setBuffer(leavesBuffer, offset: leavesOffset, index: 0)
        encoder.setBuffer(rootsBuffer, offset: rootsOffset, index: 1)
        var numLevels = UInt32(10)  // log2(1024) = 10
        encoder.setBytes(&numLevels, length: 4, index: 2)
        // 512 threads per threadgroup (handles 1024 leaves: each thread loads 2)
        let tgSize = 512
        encoder.dispatchThreadgroups(MTLSize(width: numSubtrees, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
    }

    /// Encode hash64 dispatch into an existing compute encoder (for batched Merkle).
    /// Each node: 64 bytes input (two 32-byte children) → 32 bytes output.
    /// Buffer layout: input at inputOffset (count*64 bytes), output at outputOffset (count*32 bytes).
    public func encodeHash64(encoder: MTLComputeCommandEncoder,
                              buffer: MTLBuffer, inputOffset: Int,
                              outputOffset: Int, count: Int) {
        encoder.setComputePipelineState(hash64Function)
        encoder.setBuffer(buffer, offset: inputOffset, index: 0)
        encoder.setBuffer(buffer, offset: outputOffset, index: 1)
        var n = UInt32(count)
        encoder.setBytes(&n, length: 4, index: 2)
        let tg = min(256, Int(hash64Function.maxTotalThreadsPerThreadgroup))
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
    }
}

// MARK: - CPU Keccak-256 Reference

/// CPU Keccak-256 reference implementation for testing.
public func keccak256(_ input: [UInt8]) -> [UInt8] {
    var state = [UInt64](repeating: 0, count: 25)
    let rate = 136 // bytes (1088 bits for Keccak-256)

    // Absorb
    var offset = 0
    while offset < input.count {
        let blockSize = min(rate, input.count - offset)
        for i in 0..<(blockSize / 8) {
            var word: UInt64 = 0
            for b in 0..<8 {
                let idx = offset + i * 8 + b
                if idx < input.count {
                    word |= UInt64(input[idx]) << (b * 8)
                }
            }
            state[i] ^= word
        }
        // Handle partial last word
        let fullWords = blockSize / 8
        let remaining = blockSize % 8
        if remaining > 0 {
            var word: UInt64 = 0
            for b in 0..<remaining {
                word |= UInt64(input[offset + fullWords * 8 + b]) << (b * 8)
            }
            state[fullWords] ^= word
        }

        if blockSize == rate {
            keccakF1600(&state)
            offset += rate
        } else {
            offset += blockSize
        }
    }

    // Padding
    let padOffset = input.count % rate
    let padLane = padOffset / 8
    let padByte = padOffset % 8
    state[padLane] ^= UInt64(0x01) << (padByte * 8)
    state[(rate - 1) / 8] ^= 0x8000000000000000

    keccakF1600(&state)

    // Squeeze 32 bytes
    var output = [UInt8](repeating: 0, count: 32)
    for i in 0..<4 {
        for b in 0..<8 {
            output[i * 8 + b] = UInt8((state[i] >> (b * 8)) & 0xFF)
        }
    }
    return output
}

private let KECCAK_RC: [UInt64] = [
    0x0000000000000001, 0x0000000000008082,
    0x800000000000808a, 0x8000000080008000,
    0x000000000000808b, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009,
    0x000000000000008a, 0x0000000000000088,
    0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b,
    0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080,
    0x000000000000800a, 0x800000008000000a,
    0x8000000080008081, 0x8000000000008080,
    0x0000000080000001, 0x8000000080008008,
]

private let KECCAK_ROT: [Int] = [
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
]

private let KECCAK_PI: [Int] = [
     0, 10, 20,  5, 15,
    16,  1, 11, 21,  6,
     7, 17,  2, 12, 22,
    23,  8, 18,  3, 13,
    14, 24,  9, 19,  4,
]

private func keccakF1600(_ state: inout [UInt64]) {
    for round in 0..<24 {
        // Theta
        var C = [UInt64](repeating: 0, count: 5)
        for x in 0..<5 {
            C[x] = state[x] ^ state[x+5] ^ state[x+10] ^ state[x+15] ^ state[x+20]
        }
        var D = [UInt64](repeating: 0, count: 5)
        for x in 0..<5 {
            D[x] = C[(x+4) % 5] ^ ((C[(x+1) % 5] << 1) | (C[(x+1) % 5] >> 63))
        }
        for i in 0..<25 { state[i] ^= D[i % 5] }

        // Rho + Pi
        var tmp = [UInt64](repeating: 0, count: 25)
        for i in 0..<25 {
            let r = KECCAK_ROT[i]
            tmp[KECCAK_PI[i]] = r == 0 ? state[i] : (state[i] << r) | (state[i] >> (64 - r))
        }

        // Chi
        for y in 0..<5 {
            for x in 0..<5 {
                state[y*5 + x] = tmp[y*5 + x] ^ (~tmp[y*5 + (x+1)%5] & tmp[y*5 + (x+2)%5])
            }
        }

        // Iota
        state[0] ^= KECCAK_RC[round]
    }
}
