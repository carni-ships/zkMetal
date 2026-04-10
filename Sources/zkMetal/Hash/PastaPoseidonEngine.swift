// Pasta Poseidon GPU Engine — batch Poseidon hashing on Metal for Pallas/Vesta
// Mina Kimchi variant: 55 full rounds, x^7 S-box, full MDS, width=3
import Foundation
import Metal

public class PastaPoseidonEngine {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    private let tuning: TuningConfig

    // Pallas kernels
    let pallasPermuteFunction: MTLComputePipelineState
    let pallasHashPairsFunction: MTLComputePipelineState
    let pallasSpongeFunction: MTLComputePipelineState

    // Vesta kernels
    let vestaPermuteFunction: MTLComputePipelineState
    let vestaHashPairsFunction: MTLComputePipelineState
    let vestaSpongeFunction: MTLComputePipelineState

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        let library = try PastaPoseidonEngine.compileShaders(device: device)

        guard let pallasPerm = library.makeFunction(name: "pallas_poseidon_permute_kernel"),
              let pallasHP = library.makeFunction(name: "pallas_poseidon_hash_pairs"),
              let pallasSp = library.makeFunction(name: "pallas_poseidon_sponge"),
              let vestaPerm = library.makeFunction(name: "vesta_poseidon_permute_kernel"),
              let vestaHP = library.makeFunction(name: "vesta_poseidon_hash_pairs"),
              let vestaSp = library.makeFunction(name: "vesta_poseidon_sponge") else {
            throw MSMError.missingKernel
        }

        self.pallasPermuteFunction = try device.makeComputePipelineState(function: pallasPerm)
        self.pallasHashPairsFunction = try device.makeComputePipelineState(function: pallasHP)
        self.pallasSpongeFunction = try device.makeComputePipelineState(function: pallasSp)
        self.vestaPermuteFunction = try device.makeComputePipelineState(function: vestaPerm)
        self.vestaHashPairsFunction = try device.makeComputePipelineState(function: vestaHP)
        self.vestaSpongeFunction = try device.makeComputePipelineState(function: vestaSp)

        self.tuning = TuningManager.shared.config(device: device)
    }

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let pallasFpSrc = try String(contentsOfFile: shaderDir + "/fields/pallas_fp.metal", encoding: .utf8)
        let vestaFpSrc = try String(contentsOfFile: shaderDir + "/fields/vesta_fp.metal", encoding: .utf8)
        let poseidonSrc = try String(contentsOfFile: shaderDir + "/hash/pasta_poseidon.metal", encoding: .utf8)

        // Strip include guards and #include directives
        func stripIncludes(_ src: String) -> String {
            src.split(separator: "\n")
                .filter { !$0.contains("#include") && !$0.contains("#ifndef") &&
                          !$0.contains("#define") && !$0.contains("#endif") }
                .joined(separator: "\n")
        }

        let combined = stripIncludes(pallasFpSrc) + "\n" +
                        stripIncludes(vestaFpSrc) + "\n" +
                        stripIncludes(poseidonSrc)

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    // MARK: - Pallas GPU operations

    /// Batch hash pairs of Pallas field elements on GPU.
    /// Input: array of 2n PallasFp elements (pairs [a0,b0, a1,b1, ...])
    /// Output: array of n PallasFp elements (hashes)
    public func pallasHashPairs(_ input: [PallasFp]) throws -> [PallasFp] {
        precondition(input.count % 2 == 0, "Input must have even number of elements")
        let n = input.count / 2
        let stride = MemoryLayout<PallasFp>.stride

        guard let inBuf = device.makeBuffer(length: input.count * stride, options: .storageModeShared),
              let outBuf = device.makeBuffer(length: n * stride, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate buffers")
        }

        input.withUnsafeBytes { src in
            memcpy(inBuf.contents(), src.baseAddress!, input.count * stride)
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pallasHashPairsFunction)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        var count = UInt32(n)
        enc.setBytes(&count, length: 4, index: 2)
        let tg = min(tuning.hashThreadgroupSize, Int(pallasHashPairsFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = outBuf.contents().bindMemory(to: PallasFp.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    /// Batch Pallas Poseidon permutation on GPU.
    /// Input: N*3 PallasFp elements. Output: N*3 PallasFp elements.
    public func pallasBatchPermute(_ states: [PallasFp]) throws -> [PallasFp] {
        precondition(states.count % 3 == 0)
        let n = states.count / 3
        let stride = MemoryLayout<PallasFp>.stride
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
        enc.setComputePipelineState(pallasPermuteFunction)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        var count = UInt32(n)
        enc.setBytes(&count, length: 4, index: 2)
        let tg = min(tuning.hashThreadgroupSize, Int(pallasPermuteFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = outBuf.contents().bindMemory(to: PallasFp.self, capacity: states.count)
        return Array(UnsafeBufferPointer(start: ptr, count: states.count))
    }

    // MARK: - Vesta GPU operations

    /// Batch hash pairs of Vesta field elements on GPU.
    public func vestaHashPairs(_ input: [VestaFp]) throws -> [VestaFp] {
        precondition(input.count % 2 == 0, "Input must have even number of elements")
        let n = input.count / 2
        let stride = MemoryLayout<VestaFp>.stride

        guard let inBuf = device.makeBuffer(length: input.count * stride, options: .storageModeShared),
              let outBuf = device.makeBuffer(length: n * stride, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate buffers")
        }

        input.withUnsafeBytes { src in
            memcpy(inBuf.contents(), src.baseAddress!, input.count * stride)
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(vestaHashPairsFunction)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        var count = UInt32(n)
        enc.setBytes(&count, length: 4, index: 2)
        let tg = min(tuning.hashThreadgroupSize, Int(vestaHashPairsFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = outBuf.contents().bindMemory(to: VestaFp.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    /// Batch Vesta Poseidon permutation on GPU.
    public func vestaBatchPermute(_ states: [VestaFp]) throws -> [VestaFp] {
        precondition(states.count % 3 == 0)
        let n = states.count / 3
        let stride = MemoryLayout<VestaFp>.stride
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
        enc.setComputePipelineState(vestaPermuteFunction)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        var count = UInt32(n)
        enc.setBytes(&count, length: 4, index: 2)
        let tg = min(tuning.hashThreadgroupSize, Int(vestaPermuteFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = outBuf.contents().bindMemory(to: VestaFp.self, capacity: states.count)
        return Array(UnsafeBufferPointer(start: ptr, count: states.count))
    }
}
