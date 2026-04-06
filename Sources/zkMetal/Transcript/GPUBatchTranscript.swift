// GPUBatchTranscript — GPU-accelerated Fiat-Shamir transcript for batch proof generation
//
// When generating multiple proofs simultaneously, the Poseidon2 sponge permutations
// across all independent transcripts can be batched on the GPU. Each Metal thread
// processes one transcript's sponge, achieving parallelism across N proofs.
//
// Architecture:
//   - TranscriptState: captures the full sponge state (s0, s1, s2, absorbed count)
//   - GPUBatchTranscript: dispatches Metal kernels for batch absorb/squeeze
//   - CPU fallback for small batches (< 64 transcripts)
//
// Usage:
//   let engine = try GPUBatchTranscript()
//   let values: [[Fr]] = ... // N messages, one per proof
//   let challenges = try engine.batchAbsorb(values: values)
//   // Or multi-step:
//   var states = engine.initStates(count: N, domainTag: 42)
//   states = try engine.absorbUniform(states: states, messages: values)
//   let (newStates, squeezed) = try engine.batchSqueeze(states: states, count: 3)

import Foundation
import Metal
import NeonFieldOps

// MARK: - TranscriptState

/// Captures the full Poseidon2 sponge state for one independent transcript.
///
/// This allows saving/restoring transcript state across batch GPU operations.
/// The state consists of 3 Fr elements (rate0, rate1, capacity) plus the
/// absorbed count tracking how many rate cells are filled since last permutation.
public struct TranscriptState {
    /// Sponge state element 0 (rate position 0)
    public var s0: Fr
    /// Sponge state element 1 (rate position 1)
    public var s1: Fr
    /// Sponge state element 2 (capacity)
    public var s2: Fr
    /// Number of rate cells filled since last permutation (0 or 1)
    public var absorbed: UInt32

    /// Create a fresh transcript state with a domain tag in the capacity element.
    public init(domainTag: UInt64 = 0) {
        self.s0 = Fr.zero
        self.s1 = Fr.zero
        self.s2 = frFromInt(domainTag)
        self.absorbed = 0
    }

    /// Create a state from explicit values.
    public init(s0: Fr, s1: Fr, s2: Fr, absorbed: UInt32) {
        self.s0 = s0
        self.s1 = s1
        self.s2 = s2
        self.absorbed = absorbed
    }
}

// MARK: - GPUBatchTranscript

/// GPU-accelerated batch Fiat-Shamir transcript engine.
///
/// Processes N independent Poseidon2 sponge transcripts in parallel on Metal GPU.
/// Each thread handles one transcript's absorb/squeeze operations.
///
/// Falls back to CPU (GCD parallel) for small batches (< 64) to avoid GPU dispatch overhead.
public class GPUBatchTranscript {
    public static let version = Versions.gpuBatchTranscript

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    private let absorbSqueezeKernel: MTLComputePipelineState
    private let absorbVarlenKernel: MTLComputePipelineState
    private let squeezeKernel: MTLComputePipelineState
    public let rcBuffer: MTLBuffer

    private let tuning: TuningConfig

    /// Minimum batch size for GPU dispatch. Below this, CPU fallback is used.
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

        let library = try GPUBatchTranscript.compileShaders(device: device)

        guard let absorbSqueezeFn = library.makeFunction(name: "poseidon2_batch_absorb_squeeze"),
              let absorbVarlenFn = library.makeFunction(name: "poseidon2_batch_absorb_varlen"),
              let squeezeFn = library.makeFunction(name: "poseidon2_batch_squeeze") else {
            throw MSMError.missingKernel
        }

        self.absorbSqueezeKernel = try device.makeComputePipelineState(function: absorbSqueezeFn)
        self.absorbVarlenKernel = try device.makeComputePipelineState(function: absorbVarlenFn)
        self.squeezeKernel = try device.makeComputePipelineState(function: squeezeFn)

        // Round constants buffer (64 rounds * 3 = 192 Fr, Montgomery form)
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
            throw MSMError.gpuError("Failed to allocate batch transcript RC buffer")
        }
        flatRC.withUnsafeBytes { src in
            memcpy(buf.contents(), src.baseAddress!, byteCount)
        }
        self.rcBuffer = buf

        self.tuning = TuningManager.shared.config(device: device)
    }

    // MARK: - Shader Compilation

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let frSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let spongeSource = try String(contentsOfFile: shaderDir + "/hash/poseidon2_batch_sponge.metal", encoding: .utf8)

        let cleanSponge = spongeSource.split(separator: "\n")
            .filter { !$0.contains("#include") }
            .joined(separator: "\n")

        let frClean = frSource
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")

        let combined = frClean + "\n" + cleanSponge

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    // MARK: - State Initialization

    /// Create N fresh transcript states with a given domain tag.
    ///
    /// - Parameters:
    ///   - count: Number of independent transcripts
    ///   - domainTag: Domain separation tag (placed in capacity element)
    /// - Returns: Array of N initialized TranscriptState values
    public func initStates(count: Int, domainTag: UInt64 = 0) -> [TranscriptState] {
        return [TranscriptState](repeating: TranscriptState(domainTag: domainTag), count: count)
    }

    // MARK: - High-Level API

    /// Absorb values for N independent proofs in parallel, squeeze one challenge each.
    ///
    /// This is the primary batch API: given N messages (one per proof), absorb each
    /// into a fresh Poseidon2 sponge transcript and squeeze one Fr challenge.
    ///
    /// - Parameters:
    ///   - values: N messages, each an array of Fr elements. All must have the same length.
    ///   - domainTag: Domain separation tag (default 0)
    /// - Returns: N Fr challenges, one per transcript
    public func batchAbsorb(values: [[Fr]], domainTag: UInt64 = 0) throws -> [Fr] {
        let n = values.count
        guard n > 0 else { return [] }

        // Validate uniform length
        let msgLen = values[0].count
        precondition(values.allSatisfy { $0.count == msgLen },
                     "batchAbsorb requires uniform-length messages; use absorbVarlen for variable lengths")

        let states = initStates(count: n, domainTag: domainTag)

        if n < GPUBatchTranscript.gpuThreshold {
            return cpuAbsorbSqueeze(states: states, messages: values, squeezeCount: 1)
                .squeezed.map { $0[0] }
        }

        let result = try gpuAbsorbSqueeze(states: states, messages: values,
                                           msgLen: msgLen, squeezeCount: 1)
        return result.squeezed.map { $0[0] }
    }

    /// Squeeze from N transcript states in parallel.
    ///
    /// - Parameters:
    ///   - states: N TranscriptState values
    ///   - count: Number of Fr elements to squeeze per transcript
    /// - Returns: Tuple of (updated states, squeezed values as N arrays of `count` Fr elements)
    public func batchSqueeze(states: [TranscriptState], count: Int) throws -> (states: [TranscriptState], squeezed: [[Fr]]) {
        let n = states.count
        guard n > 0 else { return ([], []) }

        if n < GPUBatchTranscript.gpuThreshold {
            return cpuSqueeze(states: states, squeezeCount: count)
        }

        return try gpuSqueeze(states: states, squeezeCount: count)
    }

    /// Absorb uniform-length messages into existing states (no squeeze).
    ///
    /// - Parameters:
    ///   - states: N TranscriptState values
    ///   - messages: N messages, each an array of Fr elements (must be uniform length)
    /// - Returns: Updated TranscriptState values
    public func absorbUniform(states: [TranscriptState], messages: [[Fr]]) throws -> [TranscriptState] {
        let n = states.count
        guard n > 0 else { return [] }
        precondition(messages.count == n, "Message count must match state count")

        let msgLen = messages[0].count
        precondition(messages.allSatisfy { $0.count == msgLen },
                     "All messages must have same length for absorbUniform")

        if n < GPUBatchTranscript.gpuThreshold {
            return cpuAbsorbSqueeze(states: states, messages: messages, squeezeCount: 0).states
        }

        return try gpuAbsorbSqueeze(states: states, messages: messages,
                                     msgLen: msgLen, squeezeCount: 0).states
    }

    /// Absorb variable-length messages into existing states.
    ///
    /// - Parameters:
    ///   - states: N TranscriptState values
    ///   - messages: N messages of potentially different lengths
    /// - Returns: Updated TranscriptState values
    public func absorbVarlen(states: [TranscriptState], messages: [[Fr]]) throws -> [TranscriptState] {
        let n = states.count
        guard n > 0 else { return [] }
        precondition(messages.count == n, "Message count must match state count")

        if n < GPUBatchTranscript.gpuThreshold {
            return cpuAbsorbSqueeze(states: states, messages: messages, squeezeCount: 0).states
        }

        return try gpuAbsorbVarlen(states: states, messages: messages)
    }

    // MARK: - GPU Implementation

    private func gpuAbsorbSqueeze(
        states: [TranscriptState],
        messages: [[Fr]],
        msgLen: Int,
        squeezeCount: Int
    ) throws -> (states: [TranscriptState], squeezed: [[Fr]]) {
        let n = states.count
        let frStride = MemoryLayout<Fr>.stride

        // Flatten states to [s0_0, s1_0, s2_0, s0_1, s1_1, s2_1, ...]
        var flatStates = [Fr]()
        flatStates.reserveCapacity(n * 3)
        var absorbedCounts = [UInt32]()
        absorbedCounts.reserveCapacity(n)
        for s in states {
            flatStates.append(s.s0)
            flatStates.append(s.s1)
            flatStates.append(s.s2)
            absorbedCounts.append(s.absorbed)
        }

        // Flatten messages
        let flatMessages = messages.flatMap { $0 }

        // Allocate GPU buffers
        let stateBytes = n * 3 * frStride
        let msgBytes = max(1, flatMessages.count * frStride)
        let squeezeBytes = max(1, n * squeezeCount * frStride)
        let countBytes = n * MemoryLayout<UInt32>.stride

        guard let statesInBuf = device.makeBuffer(length: stateBytes, options: .storageModeShared),
              let msgBuf = device.makeBuffer(length: msgBytes, options: .storageModeShared),
              let statesOutBuf = device.makeBuffer(length: stateBytes, options: .storageModeShared),
              let squeezeBuf = device.makeBuffer(length: squeezeBytes, options: .storageModeShared),
              let absorbedInBuf = device.makeBuffer(length: countBytes, options: .storageModeShared),
              let absorbedOutBuf = device.makeBuffer(length: countBytes, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate batch transcript buffers")
        }

        flatStates.withUnsafeBytes { src in memcpy(statesInBuf.contents(), src.baseAddress!, stateBytes) }
        if !flatMessages.isEmpty {
            flatMessages.withUnsafeBytes { src in memcpy(msgBuf.contents(), src.baseAddress!, flatMessages.count * frStride) }
        }
        absorbedCounts.withUnsafeBytes { src in memcpy(absorbedInBuf.contents(), src.baseAddress!, countBytes) }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(absorbSqueezeKernel)
        enc.setBuffer(statesInBuf, offset: 0, index: 0)
        enc.setBuffer(msgBuf, offset: 0, index: 1)
        enc.setBuffer(statesOutBuf, offset: 0, index: 2)
        enc.setBuffer(squeezeBuf, offset: 0, index: 3)
        enc.setBuffer(rcBuffer, offset: 0, index: 4)
        var nCount = UInt32(n)
        enc.setBytes(&nCount, length: 4, index: 5)
        var mLen = UInt32(msgLen)
        enc.setBytes(&mLen, length: 4, index: 6)
        var sqCount = UInt32(squeezeCount)
        enc.setBytes(&sqCount, length: 4, index: 7)
        enc.setBuffer(absorbedInBuf, offset: 0, index: 8)
        enc.setBuffer(absorbedOutBuf, offset: 0, index: 9)

        let tg = min(tuning.hashThreadgroupSize, Int(absorbSqueezeKernel.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        // Read results
        let outStatesPtr = statesOutBuf.contents().bindMemory(to: Fr.self, capacity: n * 3)
        let outAbsorbedPtr = absorbedOutBuf.contents().bindMemory(to: UInt32.self, capacity: n)

        var newStates = [TranscriptState]()
        newStates.reserveCapacity(n)
        for i in 0..<n {
            newStates.append(TranscriptState(
                s0: outStatesPtr[i * 3],
                s1: outStatesPtr[i * 3 + 1],
                s2: outStatesPtr[i * 3 + 2],
                absorbed: outAbsorbedPtr[i]
            ))
        }

        var squeezedResults = [[Fr]]()
        if squeezeCount > 0 {
            let sqPtr = squeezeBuf.contents().bindMemory(to: Fr.self, capacity: n * squeezeCount)
            squeezedResults.reserveCapacity(n)
            for i in 0..<n {
                squeezedResults.append(Array(UnsafeBufferPointer(start: sqPtr + i * squeezeCount, count: squeezeCount)))
            }
        }

        return (newStates, squeezedResults)
    }

    private func gpuAbsorbVarlen(states: [TranscriptState], messages: [[Fr]]) throws -> [TranscriptState] {
        let n = states.count
        let frStride = MemoryLayout<Fr>.stride

        // Build flat messages and offsets
        var flatMessages = [Fr]()
        var offsets = [UInt32]()
        offsets.reserveCapacity(n + 1)
        var offset: UInt32 = 0
        for msg in messages {
            offsets.append(offset)
            flatMessages.append(contentsOf: msg)
            offset += UInt32(msg.count)
        }
        offsets.append(offset)

        // Flatten states
        var flatStates = [Fr]()
        flatStates.reserveCapacity(n * 3)
        var absorbedCounts = [UInt32]()
        absorbedCounts.reserveCapacity(n)
        for s in states {
            flatStates.append(s.s0)
            flatStates.append(s.s1)
            flatStates.append(s.s2)
            absorbedCounts.append(s.absorbed)
        }

        let stateBytes = n * 3 * frStride
        let msgBytes = max(1, flatMessages.count * frStride)
        let offsetBytes = offsets.count * MemoryLayout<UInt32>.stride
        let countBytes = n * MemoryLayout<UInt32>.stride

        guard let statesInBuf = device.makeBuffer(length: stateBytes, options: .storageModeShared),
              let msgBuf = device.makeBuffer(length: msgBytes, options: .storageModeShared),
              let statesOutBuf = device.makeBuffer(length: stateBytes, options: .storageModeShared),
              let offsetsBuf = device.makeBuffer(length: offsetBytes, options: .storageModeShared),
              let absorbedInBuf = device.makeBuffer(length: countBytes, options: .storageModeShared),
              let absorbedOutBuf = device.makeBuffer(length: countBytes, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate varlen transcript buffers")
        }

        flatStates.withUnsafeBytes { src in memcpy(statesInBuf.contents(), src.baseAddress!, stateBytes) }
        if !flatMessages.isEmpty {
            flatMessages.withUnsafeBytes { src in memcpy(msgBuf.contents(), src.baseAddress!, flatMessages.count * frStride) }
        }
        offsets.withUnsafeBytes { src in memcpy(offsetsBuf.contents(), src.baseAddress!, offsetBytes) }
        absorbedCounts.withUnsafeBytes { src in memcpy(absorbedInBuf.contents(), src.baseAddress!, countBytes) }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(absorbVarlenKernel)
        enc.setBuffer(statesInBuf, offset: 0, index: 0)
        enc.setBuffer(msgBuf, offset: 0, index: 1)
        enc.setBuffer(statesOutBuf, offset: 0, index: 2)
        enc.setBuffer(rcBuffer, offset: 0, index: 3)
        var nCount = UInt32(n)
        enc.setBytes(&nCount, length: 4, index: 4)
        enc.setBuffer(offsetsBuf, offset: 0, index: 5)
        enc.setBuffer(absorbedInBuf, offset: 0, index: 6)
        enc.setBuffer(absorbedOutBuf, offset: 0, index: 7)

        let tg = min(tuning.hashThreadgroupSize, Int(absorbVarlenKernel.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let outStatesPtr = statesOutBuf.contents().bindMemory(to: Fr.self, capacity: n * 3)
        let outAbsorbedPtr = absorbedOutBuf.contents().bindMemory(to: UInt32.self, capacity: n)

        var newStates = [TranscriptState]()
        newStates.reserveCapacity(n)
        for i in 0..<n {
            newStates.append(TranscriptState(
                s0: outStatesPtr[i * 3],
                s1: outStatesPtr[i * 3 + 1],
                s2: outStatesPtr[i * 3 + 2],
                absorbed: outAbsorbedPtr[i]
            ))
        }
        return newStates
    }

    private func gpuSqueeze(states: [TranscriptState], squeezeCount: Int) throws -> (states: [TranscriptState], squeezed: [[Fr]]) {
        let n = states.count
        let frStride = MemoryLayout<Fr>.stride

        var flatStates = [Fr]()
        flatStates.reserveCapacity(n * 3)
        var absorbedCounts = [UInt32]()
        absorbedCounts.reserveCapacity(n)
        for s in states {
            flatStates.append(s.s0)
            flatStates.append(s.s1)
            flatStates.append(s.s2)
            absorbedCounts.append(s.absorbed)
        }

        let stateBytes = n * 3 * frStride
        let squeezeBytes = n * squeezeCount * frStride
        let countBytes = n * MemoryLayout<UInt32>.stride

        guard let statesInBuf = device.makeBuffer(length: stateBytes, options: .storageModeShared),
              let statesOutBuf = device.makeBuffer(length: stateBytes, options: .storageModeShared),
              let squeezeBuf = device.makeBuffer(length: squeezeBytes, options: .storageModeShared),
              let absorbedInBuf = device.makeBuffer(length: countBytes, options: .storageModeShared),
              let absorbedOutBuf = device.makeBuffer(length: countBytes, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate squeeze buffers")
        }

        flatStates.withUnsafeBytes { src in memcpy(statesInBuf.contents(), src.baseAddress!, stateBytes) }
        absorbedCounts.withUnsafeBytes { src in memcpy(absorbedInBuf.contents(), src.baseAddress!, countBytes) }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(squeezeKernel)
        enc.setBuffer(statesInBuf, offset: 0, index: 0)
        enc.setBuffer(statesOutBuf, offset: 0, index: 1)
        enc.setBuffer(squeezeBuf, offset: 0, index: 2)
        enc.setBuffer(rcBuffer, offset: 0, index: 3)
        var nCount = UInt32(n)
        enc.setBytes(&nCount, length: 4, index: 4)
        var sqCount = UInt32(squeezeCount)
        enc.setBytes(&sqCount, length: 4, index: 5)
        enc.setBuffer(absorbedInBuf, offset: 0, index: 6)
        enc.setBuffer(absorbedOutBuf, offset: 0, index: 7)

        let tg = min(tuning.hashThreadgroupSize, Int(squeezeKernel.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let outStatesPtr = statesOutBuf.contents().bindMemory(to: Fr.self, capacity: n * 3)
        let outAbsorbedPtr = absorbedOutBuf.contents().bindMemory(to: UInt32.self, capacity: n)
        let sqPtr = squeezeBuf.contents().bindMemory(to: Fr.self, capacity: n * squeezeCount)

        var newStates = [TranscriptState]()
        newStates.reserveCapacity(n)
        var squeezedResults = [[Fr]]()
        squeezedResults.reserveCapacity(n)

        for i in 0..<n {
            newStates.append(TranscriptState(
                s0: outStatesPtr[i * 3],
                s1: outStatesPtr[i * 3 + 1],
                s2: outStatesPtr[i * 3 + 2],
                absorbed: outAbsorbedPtr[i]
            ))
            squeezedResults.append(Array(UnsafeBufferPointer(start: sqPtr + i * squeezeCount, count: squeezeCount)))
        }

        return (newStates, squeezedResults)
    }

    // MARK: - CPU Fallback

    /// CPU absorb+squeeze for small batches. Uses the same Poseidon2Sponge as the
    /// non-batch transcript, ensuring identical results.
    private func cpuAbsorbSqueeze(
        states: [TranscriptState],
        messages: [[Fr]],
        squeezeCount: Int
    ) -> (states: [TranscriptState], squeezed: [[Fr]]) {
        let n = states.count
        var newStates = [TranscriptState](repeating: TranscriptState(), count: n)
        var squeezed = [[Fr]](repeating: [], count: n)

        for i in 0..<n {
            var s0 = states[i].s0
            var s1 = states[i].s1
            var s2 = states[i].s2
            var absorbed = Int(states[i].absorbed)

            // Absorb
            for elem in messages[i] {
                if absorbed == 0 {
                    s0 = frAdd(s0, elem)
                } else {
                    s1 = frAdd(s1, elem)
                }
                absorbed += 1
                if absorbed == 2 {
                    poseidon2PermuteInPlace(&s0, &s1, &s2)
                    absorbed = 0
                }
            }

            // Squeeze
            if squeezeCount > 0 {
                if absorbed > 0 || !messages[i].isEmpty {
                    poseidon2PermuteInPlace(&s0, &s1, &s2)
                    absorbed = 0
                }

                var results = [Fr]()
                results.reserveCapacity(squeezeCount)
                var squeezePos = 0
                for _ in 0..<squeezeCount {
                    if squeezePos >= 2 {
                        poseidon2PermuteInPlace(&s0, &s1, &s2)
                        squeezePos = 0
                    }
                    if squeezePos == 0 {
                        results.append(s0)
                    } else {
                        results.append(s1)
                    }
                    squeezePos += 1
                }
                squeezed[i] = results
            }

            newStates[i] = TranscriptState(s0: s0, s1: s1, s2: s2, absorbed: UInt32(absorbed))
        }

        return (newStates, squeezed)
    }

    /// CPU squeeze-only for small batches.
    private func cpuSqueeze(states: [TranscriptState], squeezeCount: Int) -> (states: [TranscriptState], squeezed: [[Fr]]) {
        let n = states.count
        var newStates = [TranscriptState](repeating: TranscriptState(), count: n)
        var squeezed = [[Fr]](repeating: [], count: n)

        for i in 0..<n {
            var s0 = states[i].s0
            var s1 = states[i].s1
            var s2 = states[i].s2
            let absorbed = Int(states[i].absorbed)

            if absorbed > 0 {
                poseidon2PermuteInPlace(&s0, &s1, &s2)
            }

            var results = [Fr]()
            results.reserveCapacity(squeezeCount)
            var squeezePos = 0
            for _ in 0..<squeezeCount {
                if squeezePos >= 2 {
                    poseidon2PermuteInPlace(&s0, &s1, &s2)
                    squeezePos = 0
                }
                if squeezePos == 0 {
                    results.append(s0)
                } else {
                    results.append(s1)
                }
                squeezePos += 1
            }
            squeezed[i] = results

            newStates[i] = TranscriptState(s0: s0, s1: s1, s2: s2, absorbed: 0)
        }

        return (newStates, squeezed)
    }
}
