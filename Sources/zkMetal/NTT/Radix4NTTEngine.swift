// Radix-4 NTT Engine — unified wrapper for BabyBear and Goldilocks radix-4 kernels
// Dispatches radix-4 butterflies (2 stages per kernel), falling back to radix-2
// for the final stage when log2(N) is odd. Halves GPU round-trips vs pure radix-2.

import Foundation
import Metal

/// Unified radix-4 NTT engine supporting BabyBear (32-bit) and Goldilocks (64-bit).
public class Radix4NTTEngine {
    public static let version = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")

    public enum Field {
        case babyBear   // 32-bit, p = 0x78000001
        case goldilocks // 64-bit, p = 2^64 - 2^32 + 1
    }

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let field: Field

    // Radix-4 global kernels
    private let r4ButterflyFn: MTLComputePipelineState
    private let r4InvButterflyFn: MTLComputePipelineState

    // Radix-4 fused threadgroup kernels
    private let r4FusedFn: MTLComputePipelineState
    private let r4InvFusedFn: MTLComputePipelineState

    // Radix-2 fallback (from field-specific shader files)
    private let r2ButterflyFn: MTLComputePipelineState
    private let r2InvButterflyFn: MTLComputePipelineState

    // Utility kernels
    private let scaleFn: MTLComputePipelineState
    private let bitrevFn: MTLComputePipelineState

    // Caches
    private var twiddleCache: [Int: MTLBuffer] = [:]
    private var invTwiddleCache: [Int: MTLBuffer] = [:]
    private var invNCache: [Int: MTLBuffer] = [:]
    private var cachedDataBuf: MTLBuffer?
    private var cachedDataBufElements: Int = 0

    private let tuning: TuningConfig

    // BabyBear: 8192 * 4B = 32KB, Goldilocks: 4096 * 8B = 32KB
    public var maxFusedElements: Int { field == .babyBear ? 8192 : 4096 }
    public var maxFusedLogN: Int { field == .babyBear ? 13 : 12 }
    private var elementSize: Int { field == .babyBear ? 4 : 8 }

    public init(field: Field) throws {
        self.field = field

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue
        self.tuning = TuningManager.shared.config(device: device)

        // Compile the radix-4 shader (includes both field headers)
        let radix4Lib = try Radix4NTTEngine.compileRadix4Shaders(device: device)

        // Compile the field-specific shader for radix-2 fallback + utility kernels
        let fieldLib = try Radix4NTTEngine.compileFieldShaders(device: device, field: field)

        switch field {
        case .babyBear:
            guard let r4bf = radix4Lib.makeFunction(name: "radix4_butterfly_bb"),
                  let r4ibf = radix4Lib.makeFunction(name: "radix4_inv_butterfly_bb"),
                  let r4ff = radix4Lib.makeFunction(name: "radix4_butterfly_fused_bb"),
                  let r4iff = radix4Lib.makeFunction(name: "radix4_inv_butterfly_fused_bb"),
                  let r2bf = fieldLib.makeFunction(name: "bb_ntt_butterfly"),
                  let r2ibf = fieldLib.makeFunction(name: "bb_intt_butterfly"),
                  let sfn = fieldLib.makeFunction(name: "bb_ntt_scale"),
                  let brfn = fieldLib.makeFunction(name: "bb_ntt_bitrev_inplace") else {
                throw MSMError.missingKernel
            }
            self.r4ButterflyFn = try device.makeComputePipelineState(function: r4bf)
            self.r4InvButterflyFn = try device.makeComputePipelineState(function: r4ibf)
            self.r4FusedFn = try device.makeComputePipelineState(function: r4ff)
            self.r4InvFusedFn = try device.makeComputePipelineState(function: r4iff)
            self.r2ButterflyFn = try device.makeComputePipelineState(function: r2bf)
            self.r2InvButterflyFn = try device.makeComputePipelineState(function: r2ibf)
            self.scaleFn = try device.makeComputePipelineState(function: sfn)
            self.bitrevFn = try device.makeComputePipelineState(function: brfn)

        case .goldilocks:
            guard let r4bf = radix4Lib.makeFunction(name: "radix4_butterfly_gl"),
                  let r4ibf = radix4Lib.makeFunction(name: "radix4_inv_butterfly_gl"),
                  let r4ff = radix4Lib.makeFunction(name: "radix4_butterfly_fused_gl"),
                  let r4iff = radix4Lib.makeFunction(name: "radix4_inv_butterfly_fused_gl"),
                  let r2bf = fieldLib.makeFunction(name: "gl_ntt_butterfly"),
                  let r2ibf = fieldLib.makeFunction(name: "gl_intt_butterfly"),
                  let sfn = fieldLib.makeFunction(name: "gl_ntt_scale"),
                  let brfn = fieldLib.makeFunction(name: "gl_ntt_bitrev_inplace") else {
                throw MSMError.missingKernel
            }
            self.r4ButterflyFn = try device.makeComputePipelineState(function: r4bf)
            self.r4InvButterflyFn = try device.makeComputePipelineState(function: r4ibf)
            self.r4FusedFn = try device.makeComputePipelineState(function: r4ff)
            self.r4InvFusedFn = try device.makeComputePipelineState(function: r4iff)
            self.r2ButterflyFn = try device.makeComputePipelineState(function: r2bf)
            self.r2InvButterflyFn = try device.makeComputePipelineState(function: r2ibf)
            self.scaleFn = try device.makeComputePipelineState(function: sfn)
            self.bitrevFn = try device.makeComputePipelineState(function: brfn)
        }
    }

    // MARK: - Shader compilation

    private static func compileRadix4Shaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let bbField = try String(contentsOfFile: shaderDir + "/fields/babybear.metal", encoding: .utf8)
        let glField = try String(contentsOfFile: shaderDir + "/fields/goldilocks.metal", encoding: .utf8)
        let r4Source = try String(contentsOfFile: shaderDir + "/ntt/radix4_ntt.metal", encoding: .utf8)

        let cleanR4 = r4Source.split(separator: "\n").filter { !$0.contains("#include") }.joined(separator: "\n")
        let cleanBB = bbField
            .replacingOccurrences(of: "#ifndef BABYBEAR_METAL", with: "")
            .replacingOccurrences(of: "#define BABYBEAR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BABYBEAR_METAL", with: "")
        let cleanGL = glField
            .replacingOccurrences(of: "#ifndef GOLDILOCKS_METAL", with: "")
            .replacingOccurrences(of: "#define GOLDILOCKS_METAL", with: "")
            .replacingOccurrences(of: "#endif // GOLDILOCKS_METAL", with: "")

        let combined = "#include <metal_stdlib>\nusing namespace metal;\n" + cleanBB + "\n" + cleanGL + "\n" + cleanR4
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    private static func compileFieldShaders(device: MTLDevice, field: Field) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let (fieldFile, nttFile, guardPre, guardDef, guardEnd) = {
            switch field {
            case .babyBear:
                return ("fields/babybear.metal", "ntt/ntt_babybear.metal",
                        "#ifndef BABYBEAR_METAL", "#define BABYBEAR_METAL", "#endif // BABYBEAR_METAL")
            case .goldilocks:
                return ("fields/goldilocks.metal", "ntt/ntt_goldilocks.metal",
                        "#ifndef GOLDILOCKS_METAL", "#define GOLDILOCKS_METAL", "#endif // GOLDILOCKS_METAL")
            }
        }()

        let fieldSrc = try String(contentsOfFile: shaderDir + "/" + fieldFile, encoding: .utf8)
        let nttSrc = try String(contentsOfFile: shaderDir + "/" + nttFile, encoding: .utf8)

        let cleanNTT = nttSrc.split(separator: "\n").filter { !$0.contains("#include") }.joined(separator: "\n")
        let cleanField = fieldSrc
            .replacingOccurrences(of: guardPre, with: "")
            .replacingOccurrences(of: guardDef, with: "")
            .replacingOccurrences(of: guardEnd, with: "")

        let combined = cleanField + "\n" + cleanNTT
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    private static func findShaderDir() -> String {
        let execPath = CommandLine.arguments[0]
        let execDir = (execPath as NSString).deletingLastPathComponent
        for bundle in Bundle.allBundles {
            if let url = bundle.url(forResource: "Shaders", withExtension: nil) {
                let path = url.appendingPathComponent("fields/babybear.metal").path
                if FileManager.default.fileExists(atPath: path) {
                    return url.path
                }
            }
        }
        let candidates = [
            "\(execDir)/../Sources/Shaders",
            "./Sources/Shaders",
        ]
        for path in candidates {
            if FileManager.default.fileExists(atPath: "\(path)/fields/babybear.metal") {
                return path
            }
        }
        return "./Sources/Shaders"
    }

    // MARK: - Twiddle factor caching

    private func getTwiddles(logN: Int) -> MTLBuffer {
        if let cached = twiddleCache[logN] { return cached }
        let buf: MTLBuffer
        switch field {
        case .babyBear:
            let twiddles = bbPrecomputeTwiddles(logN: logN)
            buf = createBuffer(twiddles)!
        case .goldilocks:
            let twiddles = glPrecomputeTwiddles(logN: logN)
            buf = createBuffer(twiddles)!
        }
        twiddleCache[logN] = buf
        return buf
    }

    private func getInvTwiddles(logN: Int) -> MTLBuffer {
        if let cached = invTwiddleCache[logN] { return cached }
        let buf: MTLBuffer
        switch field {
        case .babyBear:
            let twiddles = bbPrecomputeInverseTwiddles(logN: logN)
            buf = createBuffer(twiddles)!
        case .goldilocks:
            let twiddles = glPrecomputeInverseTwiddles(logN: logN)
            buf = createBuffer(twiddles)!
        }
        invTwiddleCache[logN] = buf
        return buf
    }

    private func getInvN(logN: Int) -> MTLBuffer {
        if let cached = invNCache[logN] { return cached }
        let buf: MTLBuffer
        switch field {
        case .babyBear:
            let invN = bbInverse(Bb(v: UInt32(1 << logN)))
            buf = createBuffer([invN])!
        case .goldilocks:
            let invN = glInverse(Gl(v: UInt64(1 << logN)))
            buf = createBuffer([invN])!
        }
        invNCache[logN] = buf
        return buf
    }

    private func createBuffer<T>(_ data: [T]) -> MTLBuffer? {
        let byteCount = data.count * MemoryLayout<T>.stride
        guard let buf = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            return nil
        }
        data.withUnsafeBytes { src in
            memcpy(buf.contents(), src.baseAddress!, byteCount)
        }
        return buf
    }

    // MARK: - Forward NTT (radix-4)

    /// Forward NTT using radix-4 butterflies. Falls back to radix-2 for final stage if logN is odd.
    public func ntt(data: MTLBuffer, logN: Int) throws {
        let n = UInt32(1 << logN)
        let nInt = Int(n)
        let twiddles = getTwiddles(logN: logN)

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var nVal = n
        var logNVal = UInt32(logN)

        let enc = cmdBuf.makeComputeCommandEncoder()!

        // Step 1: Bit-reversal permutation
        enc.setComputePipelineState(bitrevFn)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBytes(&nVal, length: 4, index: 1)
        enc.setBytes(&logNVal, length: 4, index: 2)
        let tgBR = min(tuning.nttThreadgroupSize, Int(bitrevFn.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: nInt, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tgBR, height: 1, depth: 1))

        // Step 2: Fused stages in threadgroup shared memory (if applicable)
        let fusedStages = min(logN, maxFusedLogN)
        if fusedStages >= 2 {
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(r4FusedFn)
            enc.setBuffer(data, offset: 0, index: 0)
            enc.setBuffer(twiddles, offset: 0, index: 1)
            enc.setBytes(&nVal, length: 4, index: 2)
            var fusedVal = UInt32(fusedStages)
            enc.setBytes(&fusedVal, length: 4, index: 3)
            var stageOffset = UInt32(0)
            enc.setBytes(&stageOffset, length: 4, index: 4)
            let tgThreads = (1 << fusedStages) / 2
            let numGroups = nInt / (1 << fusedStages)
            enc.dispatchThreadgroups(MTLSize(width: max(numGroups, 1), height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: tgThreads, height: 1, depth: 1))
        }

        // Step 3: Remaining global stages using radix-4, with radix-2 fallback
        var stage = UInt32(fusedStages >= 2 ? fusedStages : 0)

        // If we skipped fused stages (logN < 2), do all stages as global
        if fusedStages < 2 {
            stage = 0
        }

        while stage + 1 < UInt32(logN) {
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(r4ButterflyFn)
            enc.setBuffer(data, offset: 0, index: 0)
            enc.setBuffer(twiddles, offset: 0, index: 1)
            enc.setBytes(&nVal, length: 4, index: 2)
            var stageVal = stage
            enc.setBytes(&stageVal, length: 4, index: 3)
            let numQuads = nInt / 4
            let tg4 = min(tuning.nttThreadgroupSize, Int(r4ButterflyFn.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreads(MTLSize(width: numQuads, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tg4, height: 1, depth: 1))
            stage += 2
        }

        // Final odd stage: radix-2 fallback
        if stage < UInt32(logN) {
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(r2ButterflyFn)
            enc.setBuffer(data, offset: 0, index: 0)
            enc.setBuffer(twiddles, offset: 0, index: 1)
            enc.setBytes(&nVal, length: 4, index: 2)
            var stageVal = stage
            enc.setBytes(&stageVal, length: 4, index: 3)
            let numButterflies = nInt / 2
            let tg2 = min(tuning.nttThreadgroupSize, Int(r2ButterflyFn.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreads(MTLSize(width: numButterflies, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tg2, height: 1, depth: 1))
        }

        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    // MARK: - Inverse NTT (radix-4)

    /// Inverse NTT using radix-4 DIF butterflies. Falls back to radix-2 for odd stage.
    public func intt(data: MTLBuffer, logN: Int) throws {
        let n = UInt32(1 << logN)
        let nInt = Int(n)
        let invTwiddles = getInvTwiddles(logN: logN)
        let invN = getInvN(logN: logN)

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var nVal = n
        let fusedStages = min(logN, maxFusedLogN)

        let enc = cmdBuf.makeComputeCommandEncoder()!

        // Step 1: Global DIF stages (from highest down to fused boundary)
        let globalEnd = fusedStages >= 2 ? UInt32(fusedStages) : 0
        let numGlobalStages = UInt32(logN) - globalEnd
        if numGlobalStages > 0 {
            var s: UInt32 = 0

            // Radix-4 DIF pairs
            while s + 1 < numGlobalStages {
                if s > 0 { enc.memoryBarrier(scope: .buffers) }
                let stage = UInt32(logN) - 1 - s
                enc.setComputePipelineState(r4InvButterflyFn)
                enc.setBuffer(data, offset: 0, index: 0)
                enc.setBuffer(invTwiddles, offset: 0, index: 1)
                enc.setBytes(&nVal, length: 4, index: 2)
                var stageVal = stage
                enc.setBytes(&stageVal, length: 4, index: 3)
                let numQuads = nInt / 4
                let tg4 = min(tuning.nttThreadgroupSize, Int(r4InvButterflyFn.maxTotalThreadsPerThreadgroup))
                enc.dispatchThreads(MTLSize(width: numQuads, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: tg4, height: 1, depth: 1))
                s += 2
            }

            // Odd remaining global stage
            if s < numGlobalStages {
                if s > 0 { enc.memoryBarrier(scope: .buffers) }
                let stage = UInt32(logN) - 1 - s
                enc.setComputePipelineState(r2InvButterflyFn)
                enc.setBuffer(data, offset: 0, index: 0)
                enc.setBuffer(invTwiddles, offset: 0, index: 1)
                enc.setBytes(&nVal, length: 4, index: 2)
                var stageVal = stage
                enc.setBytes(&stageVal, length: 4, index: 3)
                let numButterflies = nInt / 2
                let tg2 = min(tuning.nttThreadgroupSize, Int(r2InvButterflyFn.maxTotalThreadsPerThreadgroup))
                enc.dispatchThreads(MTLSize(width: numButterflies, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: tg2, height: 1, depth: 1))
            }
        }

        // Step 2: Fused DIF stages in shared memory
        if fusedStages >= 2 {
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(r4InvFusedFn)
            enc.setBuffer(data, offset: 0, index: 0)
            enc.setBuffer(invTwiddles, offset: 0, index: 1)
            enc.setBytes(&nVal, length: 4, index: 2)
            var fusedVal = UInt32(fusedStages)
            enc.setBytes(&fusedVal, length: 4, index: 3)
            var stageOffset = UInt32(fusedStages - 1)
            enc.setBytes(&stageOffset, length: 4, index: 4)
            let tgThreads = (1 << fusedStages) / 2
            let numGroups = nInt / (1 << fusedStages)
            enc.dispatchThreadgroups(MTLSize(width: max(numGroups, 1), height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: tgThreads, height: 1, depth: 1))
        }

        // Step 3: Bit-reversal
        enc.memoryBarrier(scope: .buffers)
        var logNVal = UInt32(logN)
        enc.setComputePipelineState(bitrevFn)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBytes(&nVal, length: 4, index: 1)
        enc.setBytes(&logNVal, length: 4, index: 2)
        let tgBR = min(tuning.nttThreadgroupSize, Int(bitrevFn.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: nInt, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tgBR, height: 1, depth: 1))

        // Step 4: Scale by 1/N
        enc.memoryBarrier(scope: .buffers)
        enc.setComputePipelineState(scaleFn)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(invN, offset: 0, index: 1)
        enc.setBytes(&nVal, length: 4, index: 2)
        let tgScale = min(tuning.nttThreadgroupSize, Int(scaleFn.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: nInt, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tgScale, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    // MARK: - High-level array API

    public func nttBabyBear(_ input: [Bb]) throws -> [Bb] {
        precondition(field == .babyBear, "Field mismatch: engine is not BabyBear")
        let n = input.count
        precondition(n > 0 && (n & (n - 1)) == 0, "NTT size must be power of 2")
        let logN = Int(log2(Double(n)))
        let buf = getOrCreateDataBuffer(elementCount: n)
        input.withUnsafeBytes { src in
            memcpy(buf.contents(), src.baseAddress!, n * MemoryLayout<Bb>.stride)
        }
        try ntt(data: buf, logN: logN)
        let ptr = buf.contents().bindMemory(to: Bb.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    public func inttBabyBear(_ input: [Bb]) throws -> [Bb] {
        precondition(field == .babyBear, "Field mismatch: engine is not BabyBear")
        let n = input.count
        precondition(n > 0 && (n & (n - 1)) == 0, "NTT size must be power of 2")
        let logN = Int(log2(Double(n)))
        let buf = getOrCreateDataBuffer(elementCount: n)
        input.withUnsafeBytes { src in
            memcpy(buf.contents(), src.baseAddress!, n * MemoryLayout<Bb>.stride)
        }
        try intt(data: buf, logN: logN)
        let ptr = buf.contents().bindMemory(to: Bb.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    public func nttGoldilocks(_ input: [Gl]) throws -> [Gl] {
        precondition(field == .goldilocks, "Field mismatch: engine is not Goldilocks")
        let n = input.count
        precondition(n > 0 && (n & (n - 1)) == 0, "NTT size must be power of 2")
        let logN = Int(log2(Double(n)))
        let buf = getOrCreateDataBuffer(elementCount: n)
        input.withUnsafeBytes { src in
            memcpy(buf.contents(), src.baseAddress!, n * MemoryLayout<Gl>.stride)
        }
        try ntt(data: buf, logN: logN)
        let ptr = buf.contents().bindMemory(to: Gl.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    public func inttGoldilocks(_ input: [Gl]) throws -> [Gl] {
        precondition(field == .goldilocks, "Field mismatch: engine is not Goldilocks")
        let n = input.count
        precondition(n > 0 && (n & (n - 1)) == 0, "NTT size must be power of 2")
        let logN = Int(log2(Double(n)))
        let buf = getOrCreateDataBuffer(elementCount: n)
        input.withUnsafeBytes { src in
            memcpy(buf.contents(), src.baseAddress!, n * MemoryLayout<Gl>.stride)
        }
        try intt(data: buf, logN: logN)
        let ptr = buf.contents().bindMemory(to: Gl.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    private func getOrCreateDataBuffer(elementCount: Int) -> MTLBuffer {
        let needed = elementCount * elementSize
        if elementCount <= cachedDataBufElements, let buf = cachedDataBuf { return buf }
        let buf = device.makeBuffer(length: needed, options: .storageModeShared)!
        cachedDataBuf = buf
        cachedDataBufElements = elementCount
        return buf
    }

    // MARK: - Benchmark: Radix-2 vs Radix-4

    /// Benchmark comparing radix-2 (existing engine) vs radix-4 (this engine) NTT performance.
    /// Returns (radix2_ms, radix4_ms) averaged over `iterations` runs.
    public static func benchmarkRadix2vsRadix4(size logN: Int, field: Field = .babyBear, iterations: Int = 10) throws -> (radix2: Double, radix4: Double) {
        let n = 1 << logN

        // Prepare radix-4 engine
        let r4Engine = try Radix4NTTEngine(field: field)

        // Prepare data buffer
        let elemSize = field == .babyBear ? MemoryLayout<Bb>.stride : MemoryLayout<Gl>.stride
        guard let dataBuf = r4Engine.device.makeBuffer(length: n * elemSize, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to create benchmark buffer")
        }

        // Fill with test data
        let ptr = dataBuf.contents().assumingMemoryBound(to: UInt8.self)
        for i in 0..<(n * elemSize) {
            ptr[i] = UInt8(i & 0xFF)
        }

        // Benchmark radix-2 (via existing field-specific engine)
        var r2Total: Double = 0
        switch field {
        case .babyBear:
            let r2Engine = try BabyBearNTTEngine()
            for _ in 0..<iterations {
                // Reset data
                for i in 0..<(n * elemSize) { ptr[i] = UInt8(i & 0xFF) }
                let start = CFAbsoluteTimeGetCurrent()
                try r2Engine.ntt(data: dataBuf, logN: logN)
                r2Total += (CFAbsoluteTimeGetCurrent() - start) * 1000.0
            }
        case .goldilocks:
            let r2Engine = try GoldilocksNTTEngine()
            for _ in 0..<iterations {
                for i in 0..<(n * elemSize) { ptr[i] = UInt8(i & 0xFF) }
                let start = CFAbsoluteTimeGetCurrent()
                try r2Engine.ntt(data: dataBuf, logN: logN)
                r2Total += (CFAbsoluteTimeGetCurrent() - start) * 1000.0
            }
        }

        // Benchmark radix-4
        var r4Total: Double = 0
        for _ in 0..<iterations {
            for i in 0..<(n * elemSize) { ptr[i] = UInt8(i & 0xFF) }
            let start = CFAbsoluteTimeGetCurrent()
            try r4Engine.ntt(data: dataBuf, logN: logN)
            r4Total += (CFAbsoluteTimeGetCurrent() - start) * 1000.0
        }

        let r2Avg = r2Total / Double(iterations)
        let r4Avg = r4Total / Double(iterations)

        print("Radix-4 NTT Benchmark (\(field == .babyBear ? "BabyBear" : "Goldilocks"), N=2^\(logN)=\(n))")
        print("  Radix-2: \(String(format: "%.3f", r2Avg)) ms")
        print("  Radix-4: \(String(format: "%.3f", r4Avg)) ms")
        let speedup = r2Avg / r4Avg
        print("  Speedup: \(String(format: "%.2f", speedup))x")

        return (radix2: r2Avg, radix4: r4Avg)
    }
}
