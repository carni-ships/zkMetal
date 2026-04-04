// Blake3 GPU Engine — batch Blake3 hashing on Metal
import Foundation
import Metal

public class Blake3Engine {
    public static let version = Versions.blake3
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    let hash64Function: MTLComputePipelineState   // hash 64-byte inputs
    let hash32Function: MTLComputePipelineState   // hash 32-byte parent nodes

    // Cached buffers for hash64 array API
    private var cachedH64InputBuf: MTLBuffer?
    private var cachedH64OutputBuf: MTLBuffer?
    private var cachedH64Count: Int = 0
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

        let library = try Blake3Engine.compileShaders(device: device)
        guard let hash64Fn = library.makeFunction(name: "blake3_hash_64"),
              let hash32Fn = library.makeFunction(name: "blake3_hash_32") else {
            throw MSMError.missingKernel
        }
        self.hash64Function = try device.makeComputePipelineState(function: hash64Fn)
        self.hash32Function = try device.makeComputePipelineState(function: hash32Fn)
        self.tuning = TuningManager.shared.config(device: device)
    }

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let source = try String(contentsOfFile: shaderDir + "/hash/blake3.metal", encoding: .utf8)
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: source, options: options)
    }

    private static func findShaderDir() -> String {
        let execPath = CommandLine.arguments[0]
        let execDir = (execPath as NSString).deletingLastPathComponent
        for bundle in Bundle.allBundles {
            if let url = bundle.url(forResource: "Shaders", withExtension: nil) {
                let path = url.appendingPathComponent("hash/blake3.metal").path
                if FileManager.default.fileExists(atPath: path) { return url.path }
            }
        }
        let candidates = [
            "\(execDir)/../Sources/Shaders",
            "./Sources/Shaders",
        ]
        for path in candidates {
            if FileManager.default.fileExists(atPath: "\(path)/hash/blake3.metal") { return path }
        }
        return "./Sources/Shaders"
    }

    /// Batch Blake3 hash of 64-byte inputs.
    /// Input: n * 64 bytes, Output: n * 32 bytes
    public func hash64(_ input: [UInt8]) throws -> [UInt8] {
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
        enc.setComputePipelineState(hash64Function)
        enc.setBuffer(inputBuf, offset: 0, index: 0)
        enc.setBuffer(outputBuf, offset: 0, index: 1)
        var count = UInt32(n)
        enc.setBytes(&count, length: 4, index: 2)
        let tg = min(tuning.hashThreadgroupSize, Int(hash64Function.maxTotalThreadsPerThreadgroup))
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

    /// Batch Blake3 on GPU buffers (zero-copy).
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
        let tg = min(tuning.hashThreadgroupSize, Int(hash64Function.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    /// Encode hash-pairs into an existing compute encoder (for Merkle tree construction).
    /// Hashes `count` pairs from buffer at inputOffset, writes to outputOffset.
    public func encodeHashPairs(encoder: MTLComputeCommandEncoder, buffer: MTLBuffer,
                                inputOffset: Int, outputOffset: Int, count: Int) {
        encoder.setComputePipelineState(hash32Function)
        encoder.setBuffer(buffer, offset: inputOffset, index: 0)
        encoder.setBuffer(buffer, offset: outputOffset, index: 1)
        var n = UInt32(count)
        encoder.setBytes(&n, length: 4, index: 2)
        let tg = min(tuning.hashThreadgroupSize, Int(hash32Function.maxTotalThreadsPerThreadgroup))
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
    }

    /// Hash parent nodes for Merkle tree: input is pairs of 32-byte child hashes.
    /// Input: n * 64 bytes (n pairs), Output: n * 32 bytes
    public func hashParents(_ input: [UInt8]) throws -> [UInt8] {
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
        enc.setComputePipelineState(hash32Function)
        enc.setBuffer(inputBuf, offset: 0, index: 0)
        enc.setBuffer(outputBuf, offset: 0, index: 1)
        var count = UInt32(n)
        enc.setBytes(&count, length: 4, index: 2)
        let tg = min(tuning.hashThreadgroupSize, Int(hash32Function.maxTotalThreadsPerThreadgroup))
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
}

// MARK: - CPU Blake3 reference implementation

/// CPU Blake3 hash of arbitrary-length input (single-threaded reference)
public func blake3(_ input: [UInt8]) -> [UInt8] {
    let iv: [UInt32] = [
        0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
        0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
    ]

    let msgPerm: [Int] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]

    func g(_ state: inout [UInt32], _ a: Int, _ b: Int, _ c: Int, _ d: Int, _ mx: UInt32, _ my: UInt32) {
        state[a] = state[a] &+ state[b] &+ mx
        state[d] = (state[d] ^ state[a]).rotateRight(by: 16)
        state[c] = state[c] &+ state[d]
        state[b] = (state[b] ^ state[c]).rotateRight(by: 12)
        state[a] = state[a] &+ state[b] &+ my
        state[d] = (state[d] ^ state[a]).rotateRight(by: 8)
        state[c] = state[c] &+ state[d]
        state[b] = (state[b] ^ state[c]).rotateRight(by: 7)
    }

    func round(_ state: inout [UInt32], _ msg: [UInt32]) {
        g(&state, 0, 4,  8, 12, msg[0],  msg[1])
        g(&state, 1, 5,  9, 13, msg[2],  msg[3])
        g(&state, 2, 6, 10, 14, msg[4],  msg[5])
        g(&state, 3, 7, 11, 15, msg[6],  msg[7])
        g(&state, 0, 5, 10, 15, msg[8],  msg[9])
        g(&state, 1, 6, 11, 12, msg[10], msg[11])
        g(&state, 2, 7,  8, 13, msg[12], msg[13])
        g(&state, 3, 4,  9, 14, msg[14], msg[15])
    }

    func permute(_ msg: [UInt32]) -> [UInt32] {
        var out = [UInt32](repeating: 0, count: 16)
        for i in 0..<16 { out[i] = msg[msgPerm[i]] }
        return out
    }

    func compress(_ cv: [UInt32], _ block: [UInt8], _ counterLo: UInt32, _ counterHi: UInt32,
                  _ blockLen: UInt32, _ flags: UInt32) -> [UInt32] {
        var msg = [UInt32](repeating: 0, count: 16)
        for i in 0..<16 {
            let offset = i * 4
            if offset + 3 < block.count {
                msg[i] = UInt32(block[offset]) |
                         (UInt32(block[offset + 1]) << 8) |
                         (UInt32(block[offset + 2]) << 16) |
                         (UInt32(block[offset + 3]) << 24)
            } else {
                var val: UInt32 = 0
                for j in 0..<4 {
                    if offset + j < block.count {
                        val |= UInt32(block[offset + j]) << (j * 8)
                    }
                }
                msg[i] = val
            }
        }

        var state: [UInt32] = [
            cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7],
            iv[0], iv[1], iv[2], iv[3],
            counterLo, counterHi, blockLen, flags
        ]

        var m = msg
        for r in 0..<7 {
            round(&state, m)
            if r < 6 { m = permute(m) }
        }

        for i in 0..<8 {
            state[i] ^= state[i + 8]
            state[i + 8] ^= cv[i]
        }
        return state
    }

    // Pad input to 64 bytes
    var padded = input
    let blockLen = UInt32(min(input.count, 64))
    while padded.count < 64 { padded.append(0) }

    // Single chunk, single block
    let flags: UInt32 = 1 | 2 | 8  // CHUNK_START | CHUNK_END | ROOT
    let result = compress(iv, padded, 0, 0, blockLen, flags)

    var output = [UInt8](repeating: 0, count: 32)
    for i in 0..<8 {
        output[i * 4] = UInt8(result[i] & 0xFF)
        output[i * 4 + 1] = UInt8((result[i] >> 8) & 0xFF)
        output[i * 4 + 2] = UInt8((result[i] >> 16) & 0xFF)
        output[i * 4 + 3] = UInt8((result[i] >> 24) & 0xFF)
    }
    return output
}

/// CPU Blake3 parent compression (for Merkle tree nodes)
/// Input: 64 bytes (left || right child hashes), Output: 32-byte parent hash
public func blake3Parent(_ input: [UInt8]) -> [UInt8] {
    // Use blake3 compress with PARENT flag (4)
    let iv: [UInt32] = [
        0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
        0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
    ]

    let msgPerm: [Int] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]

    func g(_ state: inout [UInt32], _ a: Int, _ b: Int, _ c: Int, _ d: Int, _ mx: UInt32, _ my: UInt32) {
        state[a] = state[a] &+ state[b] &+ mx
        state[d] = (state[d] ^ state[a]).rotateRight(by: 16)
        state[c] = state[c] &+ state[d]
        state[b] = (state[b] ^ state[c]).rotateRight(by: 12)
        state[a] = state[a] &+ state[b] &+ my
        state[d] = (state[d] ^ state[a]).rotateRight(by: 8)
        state[c] = state[c] &+ state[d]
        state[b] = (state[b] ^ state[c]).rotateRight(by: 7)
    }

    func round(_ state: inout [UInt32], _ msg: [UInt32]) {
        g(&state, 0, 4,  8, 12, msg[0],  msg[1])
        g(&state, 1, 5,  9, 13, msg[2],  msg[3])
        g(&state, 2, 6, 10, 14, msg[4],  msg[5])
        g(&state, 3, 7, 11, 15, msg[6],  msg[7])
        g(&state, 0, 5, 10, 15, msg[8],  msg[9])
        g(&state, 1, 6, 11, 12, msg[10], msg[11])
        g(&state, 2, 7,  8, 13, msg[12], msg[13])
        g(&state, 3, 4,  9, 14, msg[14], msg[15])
    }

    func permute(_ msg: [UInt32]) -> [UInt32] {
        var out = [UInt32](repeating: 0, count: 16)
        for i in 0..<16 { out[i] = msg[msgPerm[i]] }
        return out
    }

    var msg = [UInt32](repeating: 0, count: 16)
    for i in 0..<16 {
        let offset = i * 4
        msg[i] = UInt32(input[offset]) |
                 (UInt32(input[offset + 1]) << 8) |
                 (UInt32(input[offset + 2]) << 16) |
                 (UInt32(input[offset + 3]) << 24)
    }

    var state: [UInt32] = [
        iv[0], iv[1], iv[2], iv[3], iv[4], iv[5], iv[6], iv[7],
        iv[0], iv[1], iv[2], iv[3],
        0, 0, 64, 4  // counter=0, blockLen=64, flags=PARENT
    ]

    var m = msg
    for r in 0..<7 {
        round(&state, m)
        if r < 6 { m = permute(m) }
    }

    for i in 0..<8 {
        state[i] ^= state[i + 8]
    }

    var output = [UInt8](repeating: 0, count: 32)
    for i in 0..<8 {
        output[i * 4] = UInt8(state[i] & 0xFF)
        output[i * 4 + 1] = UInt8((state[i] >> 8) & 0xFF)
        output[i * 4 + 2] = UInt8((state[i] >> 16) & 0xFF)
        output[i * 4 + 3] = UInt8((state[i] >> 24) & 0xFF)
    }
    return output
}

private extension UInt32 {
    func rotateRight(by n: Int) -> UInt32 {
        return (self >> n) | (self << (32 - n))
    }
}
