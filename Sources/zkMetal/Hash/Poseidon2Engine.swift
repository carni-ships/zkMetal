// Poseidon2 GPU Engine — batch Poseidon2 hashing on Metal
import Foundation
import Metal

public class Poseidon2Engine {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    let permuteFunction: MTLComputePipelineState
    let hashPairsFunction: MTLComputePipelineState
    let merkleFusedFunction: MTLComputePipelineState
    let rcBuffer: MTLBuffer  // round constants in Montgomery form

    // Cached buffers for hashPairs to avoid per-call allocation
    private var cachedInputBuf: MTLBuffer?
    private var cachedOutputBuf: MTLBuffer?
    private var cachedBufPairs: Int = 0

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
              let merkleFusedFn = library.makeFunction(name: "poseidon2_merkle_fused") else {
            throw MSMError.missingKernel
        }

        self.permuteFunction = try device.makeComputePipelineState(function: permuteFn)
        self.hashPairsFunction = try device.makeComputePipelineState(function: hashPairsFn)
        self.merkleFusedFunction = try device.makeComputePipelineState(function: merkleFusedFn)

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

    private static func findShaderDir() -> String {
        let execPath = CommandLine.arguments[0]
        let execDir = (execPath as NSString).deletingLastPathComponent

        for bundle in Bundle.allBundles {
            if let url = bundle.url(forResource: "Shaders", withExtension: nil) {
                let frPath = url.appendingPathComponent("fields/bn254_fr.metal").path
                if FileManager.default.fileExists(atPath: frPath) {
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
        let tg = min(256, Int(hashPairsFunction.maxTotalThreadsPerThreadgroup))
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
        let tg = min(256, Int(hashPairsFunction.maxTotalThreadsPerThreadgroup))
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
        let tg = min(256, Int(hashPairsFunction.maxTotalThreadsPerThreadgroup))
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
    }

    /// Encode fused Merkle subtree: processes 1024-leaf subtrees in shared memory,
    /// producing one root per threadgroup. Eliminates 9 memory barriers per subtree.
    public func encodeMerkleFused(encoder: MTLComputeCommandEncoder,
                                   leavesBuffer: MTLBuffer, leavesOffset: Int,
                                   rootsBuffer: MTLBuffer, rootsOffset: Int,
                                   numSubtrees: Int) {
        encoder.setComputePipelineState(merkleFusedFunction)
        encoder.setBuffer(leavesBuffer, offset: leavesOffset, index: 0)
        encoder.setBuffer(rootsBuffer, offset: rootsOffset, index: 1)
        encoder.setBuffer(rcBuffer, offset: 0, index: 2)
        var numLevels: UInt32 = 10  // log2(1024) = 10
        encoder.setBytes(&numLevels, length: 4, index: 3)
        // Each threadgroup = 512 threads, processes 1024 leaves
        encoder.dispatchThreadgroups(MTLSize(width: numSubtrees, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
    }

    /// Subtree size for fused Merkle kernel
    public static let merkleSubtreeSize = 1024
}
