// GPUPoseidon2Permutation — Metal GPU engine for Poseidon2 permutation
// Supports width-3 (t=3) and width-4 (t=4) over BN254 Fr.
//
// Width-3: standard ZK hash (same parameters as Poseidon2Engine)
// Width-4: STARK trace hashing with circ(5,7,1,3) external matrix
//
// CPU fallback for < 64 permutations to avoid GPU dispatch overhead.

import Foundation
import Metal

public class GPUPoseidon2PermutationEngine {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // Width-3 kernels
    let permuteW3Function: MTLComputePipelineState
    let compressW3Function: MTLComputePipelineState

    // Width-4 kernel
    let permuteW4Function: MTLComputePipelineState

    // Round constants buffers
    public let rcBufferW3: MTLBuffer   // 64 rounds * 3 = 192 Fr (Montgomery)
    public let rcBufferW4: MTLBuffer   // 64 rounds * 4 = 256 Fr (Montgomery)
    public let diagBufferW4: MTLBuffer // 4 Fr (internal diagonal constants)

    private let tuning: TuningConfig

    /// Minimum batch size before GPU dispatch (below this, use CPU fallback)
    public static let gpuThreshold = 64

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        let library = try GPUPoseidon2PermutationEngine.compileShaders(device: device)

        guard let permuteW3Fn = library.makeFunction(name: "poseidon2_permutation_bn254"),
              let compressW3Fn = library.makeFunction(name: "poseidon2_compress_bn254"),
              let permuteW4Fn = library.makeFunction(name: "poseidon2_permutation_bn254_width4") else {
            throw MSMError.missingKernel
        }

        self.permuteW3Function = try device.makeComputePipelineState(function: permuteW3Fn)
        self.compressW3Function = try device.makeComputePipelineState(function: compressW3Fn)
        self.permuteW4Function = try device.makeComputePipelineState(function: permuteW4Fn)

        // Width-3 round constants (reuse existing POSEIDON2_ROUND_CONSTANTS)
        let rc3 = POSEIDON2_ROUND_CONSTANTS
        var flatRC3 = [Fr]()
        flatRC3.reserveCapacity(192)
        for round in rc3 {
            for elem in round {
                flatRC3.append(elem)
            }
        }
        let byteCount3 = flatRC3.count * MemoryLayout<Fr>.stride
        guard let buf3 = device.makeBuffer(length: byteCount3, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate W3 RC buffer")
        }
        flatRC3.withUnsafeBytes { src in
            memcpy(buf3.contents(), src.baseAddress!, byteCount3)
        }
        self.rcBufferW3 = buf3

        // Width-4 round constants (derived deterministically)
        let (rc4Flat, diag4) = GPUPoseidon2PermutationEngine.generateWidth4Constants()
        let byteCount4 = rc4Flat.count * MemoryLayout<Fr>.stride
        guard let buf4 = device.makeBuffer(length: byteCount4, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate W4 RC buffer")
        }
        rc4Flat.withUnsafeBytes { src in
            memcpy(buf4.contents(), src.baseAddress!, byteCount4)
        }
        self.rcBufferW4 = buf4

        let diagBytes = diag4.count * MemoryLayout<Fr>.stride
        guard let diagBuf = device.makeBuffer(length: diagBytes, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate W4 diagonal buffer")
        }
        diag4.withUnsafeBytes { src in
            memcpy(diagBuf.contents(), src.baseAddress!, diagBytes)
        }
        self.diagBufferW4 = diagBuf

        self.tuning = TuningManager.shared.config(device: device)
    }

    // MARK: - Shader Compilation

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let frSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let p2Source = try String(contentsOfFile: shaderDir + "/hash/poseidon2_permutation.metal", encoding: .utf8)

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

    // MARK: - Width-4 Round Constants Generation

    /// Generate width-4 round constants deterministically using Keccak-based PRNG.
    /// Domain separation: "Poseidon2_BN254_t4_v1"
    /// 64 rounds * 4 elements = 256 Fr values (partial rounds: only element 0 is nonzero)
    /// Also returns 4 internal diagonal constants.
    static func generateWidth4Constants() -> ([Fr], [Fr]) {
        // Seed: Keccak256("Poseidon2_BN254_t4_v1")
        var seed = keccak256(Array("Poseidon2_BN254_t4_v1".utf8))

        // Generate field elements by hashing seed repeatedly
        func nextFr() -> Fr {
            seed = keccak256(seed)
            // Take 32 bytes as a 256-bit number, reduce mod r
            var limbs = [UInt64](repeating: 0, count: 4)
            seed.withUnsafeBytes { ptr in
                for i in 0..<4 {
                    limbs[i] = ptr.load(fromByteOffset: i * 8, as: UInt64.self)
                }
            }
            // Reduce mod r by masking top bit and using Montgomery conversion
            // Since seed is random 256-bit, we need proper reduction
            limbs[3] &= 0x3FFFFFFFFFFFFFFF  // Ensure < 2^254 (well within field)
            let raw = Fr.from64(limbs)
            return frMul(raw, Fr.from64(Fr.R2_MOD_R))  // to Montgomery form
        }

        var flatRC = [Fr]()
        flatRC.reserveCapacity(256)

        // Full rounds (first 4): all 4 elements per round
        for _ in 0..<4 {
            for _ in 0..<4 {
                flatRC.append(nextFr())
            }
        }

        // Partial rounds (56): only element 0, rest zero
        for _ in 4..<60 {
            flatRC.append(nextFr())
            flatRC.append(Fr.zero)
            flatRC.append(Fr.zero)
            flatRC.append(Fr.zero)
        }

        // Full rounds (last 4): all 4 elements per round
        for _ in 60..<64 {
            for _ in 0..<4 {
                flatRC.append(nextFr())
            }
        }

        // Internal diagonal constants: [5, 7, 1, 3] in Montgomery form
        let diag: [Fr] = [
            frFromInt(5),
            frFromInt(7),
            frFromInt(1),
            frFromInt(3),
        ]

        return (flatRC, diag)
    }

    // MARK: - Width-3 CPU Fallback

    /// CPU Poseidon2 permutation (width-3) for small batches
    private func cpuPermuteW3(_ states: [[Fr]]) -> [[Fr]] {
        return states.map { poseidon2Permutation($0) }
    }

    /// CPU Poseidon2 compression for small batches
    private func cpuCompress(_ pairs: [(Fr, Fr)]) -> [Fr] {
        return pairs.map { poseidon2Hash($0.0, $0.1) }
    }

    // MARK: - Width-4 CPU Fallback

    /// CPU Poseidon2 permutation (width-4) using the same round constants
    public func cpuPermuteW4(_ state: [Fr]) -> [Fr] {
        precondition(state.count == 4)

        // Read round constants and diagonal from GPU buffers
        let rcPtr = rcBufferW4.contents().bindMemory(to: Fr.self, capacity: 256)
        let rc = UnsafeBufferPointer(start: rcPtr, count: 256)
        let diagPtr = diagBufferW4.contents().bindMemory(to: Fr.self, capacity: 4)
        let diag = UnsafeBufferPointer(start: diagPtr, count: 4)

        var s = state

        // External layer: circ(5,7,1,3)
        func externalLayer(_ s: inout [Fr]) {
            let a = s[0], b = s[1], c = s[2], d = s[3]
            let sum = frAdd(frAdd(a, b), frAdd(c, d))
            // out[i] = sum + 4*s[i] + 6*s[(i+1)%4] + 2*s[(i+3)%4]
            let a2 = frAdd(a, a); let a4 = frAdd(a2, a2); let a6 = frAdd(a4, a2)
            let b2 = frAdd(b, b); let b4 = frAdd(b2, b2); let b6 = frAdd(b4, b2)
            let c2 = frAdd(c, c); let c4 = frAdd(c2, c2); let c6 = frAdd(c4, c2)
            let d2 = frAdd(d, d); let d4 = frAdd(d2, d2); let d6 = frAdd(d4, d2)
            s[0] = frAdd(frAdd(sum, a4), frAdd(b6, d2))
            s[1] = frAdd(frAdd(sum, b4), frAdd(c6, a2))
            s[2] = frAdd(frAdd(sum, c4), frAdd(d6, b2))
            s[3] = frAdd(frAdd(sum, d4), frAdd(a6, c2))
        }

        // Internal layer: y_i = diag[i] * x_i + sum
        func internalLayer(_ s: inout [Fr]) {
            let sum = frAdd(frAdd(s[0], s[1]), frAdd(s[2], s[3]))
            for i in 0..<4 {
                s[i] = frAdd(frMul(s[i], diag[i]), sum)
            }
        }

        // S-box
        func sbox(_ x: Fr) -> Fr {
            let x2 = frMul(x, x)
            let x4 = frMul(x2, x2)
            return frMul(x4, x)
        }

        // Initial external layer
        externalLayer(&s)

        // First half of full rounds (0..3)
        for r in 0..<4 {
            let rcBase = r * 4
            for i in 0..<4 { s[i] = frAdd(s[i], rc[rcBase + i]) }
            for i in 0..<4 { s[i] = sbox(s[i]) }
            externalLayer(&s)
        }

        // Partial rounds (4..59)
        for r in 4..<60 {
            s[0] = frAdd(s[0], rc[r * 4])
            s[0] = sbox(s[0])
            internalLayer(&s)
        }

        // Second half of full rounds (60..63)
        for r in 60..<64 {
            let rcBase = r * 4
            for i in 0..<4 { s[i] = frAdd(s[i], rc[rcBase + i]) }
            for i in 0..<4 { s[i] = sbox(s[i]) }
            externalLayer(&s)
        }

        return s
    }

    // MARK: - Public API: Batch Permutation

    /// Batch permutation of many independent width-3 states on GPU.
    /// Each state is [Fr, Fr, Fr]. Falls back to CPU for small batches.
    public func permute(states: [[Fr]]) throws -> [[Fr]] {
        precondition(states.allSatisfy { $0.count == 3 }, "All states must have width 3")
        let n = states.count

        if n < GPUPoseidon2PermutationEngine.gpuThreshold {
            return cpuPermuteW3(states)
        }

        let flat = states.flatMap { $0 }
        let result = try permuteFlat(flat: flat, count: n, width: 3)

        // Unflatten
        var out = [[Fr]]()
        out.reserveCapacity(n)
        for i in 0..<n {
            out.append(Array(result[i*3..<(i+1)*3]))
        }
        return out
    }

    /// Batch permutation of many independent width-4 states on GPU.
    /// Each state is [Fr, Fr, Fr, Fr]. Falls back to CPU for small batches.
    public func permuteWidth4(states: [[Fr]]) throws -> [[Fr]] {
        precondition(states.allSatisfy { $0.count == 4 }, "All states must have width 4")
        let n = states.count

        if n < GPUPoseidon2PermutationEngine.gpuThreshold {
            return states.map { cpuPermuteW4($0) }
        }

        let flat = states.flatMap { $0 }
        let result = try permuteFlat(flat: flat, count: n, width: 4)

        var out = [[Fr]]()
        out.reserveCapacity(n)
        for i in 0..<n {
            out.append(Array(result[i*4..<(i+1)*4]))
        }
        return out
    }

    // MARK: - Public API: Batch Compression

    /// Batch two-to-one compression: for each (a, b), compute Poseidon2([a, b, 0])[0].
    /// Falls back to CPU for small batches.
    public func compress(pairs: [(Fr, Fr)]) throws -> [Fr] {
        let n = pairs.count

        if n < GPUPoseidon2PermutationEngine.gpuThreshold {
            return cpuCompress(pairs)
        }

        // Flatten pairs to [a0, b0, a1, b1, ...]
        var flat = [Fr]()
        flat.reserveCapacity(n * 2)
        for (a, b) in pairs {
            flat.append(a)
            flat.append(b)
        }

        let stride = MemoryLayout<Fr>.stride
        let inBytes = flat.count * stride
        let outBytes = n * stride

        guard let inBuf = device.makeBuffer(length: inBytes, options: .storageModeShared),
              let outBuf = device.makeBuffer(length: outBytes, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate compress buffers")
        }

        flat.withUnsafeBytes { src in
            memcpy(inBuf.contents(), src.baseAddress!, inBytes)
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(compressW3Function)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBuffer(rcBufferW3, offset: 0, index: 2)
        var count = UInt32(n)
        enc.setBytes(&count, length: 4, index: 3)
        let tg = min(tuning.hashThreadgroupSize, Int(compressW3Function.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = outBuf.contents().bindMemory(to: Fr.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    // MARK: - Public API: Zero-copy MTLBuffer API

    /// Permute states directly on GPU buffers.
    /// buf contains count * width Fr elements. Returns new buffer with results.
    public func permuteBuffer(buf: MTLBuffer, count: Int, width: Int) throws -> MTLBuffer {
        precondition(width == 3 || width == 4, "Width must be 3 or 4")
        let totalElements = count * width
        let bytes = totalElements * MemoryLayout<Fr>.stride

        guard let outBuf = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate output buffer")
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!

        if width == 3 {
            enc.setComputePipelineState(permuteW3Function)
            enc.setBuffer(buf, offset: 0, index: 0)
            enc.setBuffer(outBuf, offset: 0, index: 1)
            enc.setBuffer(rcBufferW3, offset: 0, index: 2)
            var n = UInt32(count)
            enc.setBytes(&n, length: 4, index: 3)
            let tg = min(tuning.hashThreadgroupSize, Int(permuteW3Function.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        } else {
            enc.setComputePipelineState(permuteW4Function)
            enc.setBuffer(buf, offset: 0, index: 0)
            enc.setBuffer(outBuf, offset: 0, index: 1)
            enc.setBuffer(rcBufferW4, offset: 0, index: 2)
            var n = UInt32(count)
            enc.setBytes(&n, length: 4, index: 3)
            enc.setBuffer(diagBufferW4, offset: 0, index: 4)
            let tg = min(tuning.hashThreadgroupSize, Int(permuteW4Function.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        }

        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        return outBuf
    }

    // MARK: - Internal Helpers

    /// Flat permutation: dispatch GPU kernel for width-3 or width-4
    private func permuteFlat(flat: [Fr], count: Int, width: Int) throws -> [Fr] {
        let stride = MemoryLayout<Fr>.stride
        let bytes = flat.count * stride

        guard let inBuf = device.makeBuffer(length: bytes, options: .storageModeShared),
              let outBuf = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate permute buffers")
        }

        flat.withUnsafeBytes { src in
            memcpy(inBuf.contents(), src.baseAddress!, bytes)
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!

        if width == 3 {
            enc.setComputePipelineState(permuteW3Function)
            enc.setBuffer(inBuf, offset: 0, index: 0)
            enc.setBuffer(outBuf, offset: 0, index: 1)
            enc.setBuffer(rcBufferW3, offset: 0, index: 2)
            var n = UInt32(count)
            enc.setBytes(&n, length: 4, index: 3)
            let tg = min(tuning.hashThreadgroupSize, Int(permuteW3Function.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        } else {
            enc.setComputePipelineState(permuteW4Function)
            enc.setBuffer(inBuf, offset: 0, index: 0)
            enc.setBuffer(outBuf, offset: 0, index: 1)
            enc.setBuffer(rcBufferW4, offset: 0, index: 2)
            var n = UInt32(count)
            enc.setBytes(&n, length: 4, index: 3)
            enc.setBuffer(diagBufferW4, offset: 0, index: 4)
            let tg = min(tuning.hashThreadgroupSize, Int(permuteW4Function.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        }

        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let totalElements = flat.count
        let ptr = outBuf.contents().bindMemory(to: Fr.self, capacity: totalElements)
        return Array(UnsafeBufferPointer(start: ptr, count: totalElements))
    }
}
