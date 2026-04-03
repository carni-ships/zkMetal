// Metal MSM Engine — Pippenger's bucket method with GPU acceleration
// Uses GLV endomorphism, count-sorted bucket reduce, and pipelined sort/reduce.

import Foundation
import Metal

public enum MSMError: Error {
    case noGPU
    case noCommandQueue
    case noCommandBuffer
    case missingKernel
    case invalidInput
    case gpuError(String)
}

public class MetalMSM {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let reduceSortedFunction: MTLComputePipelineState
    public let reduceCooperativeFunction: MTLComputePipelineState
    public let bucketSumDirectFunction: MTLComputePipelineState
    public let combineSegmentsFunction: MTLComputePipelineState
    public let hornerCombineFunction: MTLComputePipelineState
    public let endomorphismFunction: MTLComputePipelineState
    public let glvDecomposeFunction: MTLComputePipelineState
    public let signedDigitFunction: MTLComputePipelineState

    // Pre-allocated buffers (lazily sized, reused across calls)
    private var maxAllocatedPoints = 0
    private var maxAllocatedBuckets = 0
    private var pointsBuffer: MTLBuffer?
    private var sortedIndicesBuffer: MTLBuffer?
    private var allOffsetsBuffer: MTLBuffer?
    private var allCountsBuffer: MTLBuffer?
    private var bucketsBuffer: MTLBuffer?
    private var segmentResultsBuffer: MTLBuffer?
    private var windowResultsBuffer: MTLBuffer?
    private var finalResultBuffer: MTLBuffer?
    private var countSortedMapBuffer: MTLBuffer?
    private var cpuCountsPtr: UnsafeMutablePointer<Int>?
    private var cpuPositionsPtr: UnsafeMutablePointer<Int>?
    private var cpuScratchCapacity = 0
    private var signedDigitPtr: UnsafeMutablePointer<UInt32>?
    private var signedDigitCapacity = 0
    private var signedDigitBuffer: MTLBuffer?
    // Cached GLV buffers (reused across MSM calls)
    private var glvScalarInBufCached: MTLBuffer?
    private var glvK1MetalBufCached: MTLBuffer?
    private var glvNeg1BufCached: MTLBuffer?
    private var glvNeg2BufCached: MTLBuffer?
    private var glvCachedN: Int = 0
    public var windowBitsOverride: UInt32?
    private let tuning: TuningConfig

    public static let cacheDir = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent(".zkmsm").appendingPathComponent("cache")

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        let library: MTLLibrary
        let cacheFile = MetalMSM.cacheDir.appendingPathComponent("bn254.metallib")

        let requiredKernels = ["msm_reduce_sorted_buckets", "msm_bucket_sum_direct", "msm_combine_segments", "glv_endomorphism", "glv_decompose", "signed_digit_extract"]
        if FileManager.default.fileExists(atPath: cacheFile.path) {
            do {
                let cached = try device.makeLibrary(URL: cacheFile)
                if requiredKernels.allSatisfy({ cached.makeFunction(name: $0) != nil }) {
                    library = cached
                } else {
                    library = try MetalMSM.compileAndCache(device: device, cacheFile: cacheFile)
                }
            } catch {
                library = try MetalMSM.compileAndCache(device: device, cacheFile: cacheFile)
            }
        } else {
            library = try MetalMSM.compileAndCache(device: device, cacheFile: cacheFile)
        }

        guard let reduceSortedFn = library.makeFunction(name: "msm_reduce_sorted_buckets"),
              let reduceCoopFn = library.makeFunction(name: "msm_reduce_cooperative"),
              let sumDirectFn = library.makeFunction(name: "msm_bucket_sum_direct"),
              let combineFn = library.makeFunction(name: "msm_combine_segments"),
              let hornerFn = library.makeFunction(name: "msm_horner_combine"),
              let endoFn = library.makeFunction(name: "glv_endomorphism"),
              let glvDecomposeFn = library.makeFunction(name: "glv_decompose"),
              let signedDigitFn = library.makeFunction(name: "signed_digit_extract") else {
            throw MSMError.missingKernel
        }

        self.reduceSortedFunction = try device.makeComputePipelineState(function: reduceSortedFn)
        self.reduceCooperativeFunction = try device.makeComputePipelineState(function: reduceCoopFn)
        self.bucketSumDirectFunction = try device.makeComputePipelineState(function: sumDirectFn)
        self.combineSegmentsFunction = try device.makeComputePipelineState(function: combineFn)
        self.hornerCombineFunction = try device.makeComputePipelineState(function: hornerFn)
        self.endomorphismFunction = try device.makeComputePipelineState(function: endoFn)
        self.glvDecomposeFunction = try device.makeComputePipelineState(function: glvDecomposeFn)
        self.signedDigitFunction = try device.makeComputePipelineState(function: signedDigitFn)
        self.tuning = TuningManager.shared.config(device: device)

    }

    /// Compile shader from source and cache the library for next time.
    private static func compileAndCache(device: MTLDevice, cacheFile: URL) throws -> MTLLibrary {
        let shaderDir = findShaderDir()

        // Load and concatenate shader sources in dependency order
        let fpSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fp.metal", encoding: .utf8)
        let curveSource = try String(contentsOfFile: shaderDir + "/geometry/bn254_curve.metal", encoding: .utf8)
        let glvSource = try String(contentsOfFile: shaderDir + "/msm/glv_kernels.metal", encoding: .utf8)
        let msmSource = try String(contentsOfFile: shaderDir + "/msm/msm_kernels.metal", encoding: .utf8)

        func stripIncludes(_ s: String) -> String {
            s.split(separator: "\n").filter { !$0.contains("#include") }.joined(separator: "\n")
        }
        func stripGuards(_ s: String) -> String {
            s.replacingOccurrences(of: "#ifndef BN254_FP_METAL", with: "")
             .replacingOccurrences(of: "#define BN254_FP_METAL", with: "")
             .replacingOccurrences(of: "#endif // BN254_FP_METAL", with: "")
             .replacingOccurrences(of: "#ifndef BN254_CURVE_METAL", with: "")
             .replacingOccurrences(of: "#define BN254_CURVE_METAL", with: "")
             .replacingOccurrences(of: "#endif // BN254_CURVE_METAL", with: "")
        }

        let combined = stripGuards(fpSource) + "\n" +
                        stripGuards(stripIncludes(curveSource)) + "\n" +
                        stripIncludes(glvSource) + "\n" +
                        stripIncludes(msmSource)

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        let library = try device.makeLibrary(source: combined, options: options)

        try? FileManager.default.createDirectory(
            at: MetalMSM.cacheDir, withIntermediateDirectories: true)

        if #available(macOS 11.0, *) {
            let archiveDesc = MTLBinaryArchiveDescriptor()
            if let archive = try? device.makeBinaryArchive(descriptor: archiveDesc) {
                for name in ["msm_reduce_sorted_buckets", "msm_bucket_sum_direct"] {
                    let desc = MTLComputePipelineDescriptor()
                    desc.computeFunction = library.makeFunction(name: name)
                    try? archive.addComputePipelineFunctions(descriptor: desc)
                }
                try? archive.serialize(to: cacheFile)
            }
        }

        return library
    }

    private static func findShaderDir() -> String {
        let execPath = CommandLine.arguments[0]
        let execDir = (execPath as NSString).deletingLastPathComponent
        for bundle in Bundle.allBundles {
            if let url = bundle.url(forResource: "Shaders", withExtension: nil) {
                let path = url.appendingPathComponent("fields/bn254_fp.metal").path
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
            if FileManager.default.fileExists(atPath: "\(path)/fields/bn254_fp.metal") {
                return path
            }
        }
        return "./Sources/Shaders"
    }

    // BN254 scalar field order r as 8x32-bit limbs (little-endian)
    private static let R_LIMBS: [UInt32] = [
        0xf0000001, 0x43e1f593, 0x79b97091, 0x2833e848,
        0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72
    ]

    /// Reduce a 256-bit scalar mod r (BN254 scalar field order)
    static func reduceModR(_ scalar: [UInt32]) -> [UInt32] {
        var current = scalar
        while true {
            var gte = true
            for i in stride(from: 7, through: 0, by: -1) {
                if current[i] > R_LIMBS[i] { break }
                if current[i] < R_LIMBS[i] { gte = false; break }
            }
            if !gte { return current }
            var result = [UInt32](repeating: 0, count: 8)
            var borrow: Int64 = 0
            for i in 0..<8 {
                borrow += Int64(current[i]) - Int64(R_LIMBS[i])
                result[i] = UInt32(truncatingIfNeeded: borrow & 0xFFFFFFFF)
                borrow >>= 32
            }
            current = result
        }
    }

    @inline(__always)
    private func extractBucketIndex(_ scalarPtr: UnsafePointer<UInt32>, windowBits: UInt32, windowIndex: Int) -> Int {
        let bitOffset = windowIndex * Int(windowBits)
        let limbIdx = bitOffset / 32
        let bitPos = bitOffset % 32
        guard limbIdx < 8 else { return 0 }
        var idx = Int(scalarPtr[limbIdx] >> bitPos)
        if bitPos + Int(windowBits) > 32 && limbIdx + 1 < 8 {
            idx |= Int(scalarPtr[limbIdx + 1]) << (32 - bitPos)
        }
        idx &= (1 << windowBits) - 1
        return idx
    }

    private func extractBucketIndex(_ scalar: [UInt32], windowBits: UInt32, windowIndex: Int) -> Int {
        let bitOffset = windowIndex * Int(windowBits)
        let limbIdx = bitOffset / 32
        let bitPos = bitOffset % 32
        guard limbIdx < 8 else { return 0 }
        var idx = Int(scalar[limbIdx] >> bitPos)
        if bitPos + Int(windowBits) > 32 && limbIdx + 1 < 8 {
            idx |= Int(scalar[limbIdx + 1]) << (32 - bitPos)
        }
        idx &= (1 << windowBits) - 1
        return idx
    }

    private var maxAllocatedWindows = 0
    private var maxAllocatedSegments = 0

    private func ensureBuffers(n: Int, nBuckets: Int, nSegments: Int, nWindows: Int) {
        let needRealloc = n > maxAllocatedPoints || nBuckets > maxAllocatedBuckets || nWindows > maxAllocatedWindows || nSegments > maxAllocatedSegments
        if needRealloc {
            let np = max(n, maxAllocatedPoints)
            let nb = max(nBuckets, maxAllocatedBuckets)
            let nw = max(nWindows, maxAllocatedWindows)
            let ns = nSegments
            pointsBuffer = device.makeBuffer(
                length: MemoryLayout<PointAffine>.stride * np, options: .storageModeShared)
            sortedIndicesBuffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride * np * nw, options: .storageModeShared)
            allOffsetsBuffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride * nb * nw, options: .storageModeShared)
            allCountsBuffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride * nb * nw, options: .storageModeShared)
            bucketsBuffer = device.makeBuffer(
                length: MemoryLayout<PointProjective>.stride * nb * nw, options: .storageModeShared)
            segmentResultsBuffer = device.makeBuffer(
                length: MemoryLayout<PointProjective>.stride * ns * nw, options: .storageModeShared)
            windowResultsBuffer = device.makeBuffer(
                length: MemoryLayout<PointProjective>.stride * nw, options: .storageModeShared)
            finalResultBuffer = device.makeBuffer(
                length: MemoryLayout<PointProjective>.stride, options: .storageModeShared)
            countSortedMapBuffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride * nb * nw, options: .storageModeShared)
            signedDigitBuffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride * np * nw, options: .storageModeShared)
            maxAllocatedPoints = np
            maxAllocatedBuckets = nb
            maxAllocatedWindows = nw
            maxAllocatedSegments = nSegments
            let scratchSize = nw * nb
            if scratchSize > cpuScratchCapacity {
                cpuCountsPtr?.deallocate()
                cpuPositionsPtr?.deallocate()
                cpuCountsPtr = .allocate(capacity: scratchSize)
                cpuPositionsPtr = .allocate(capacity: scratchSize)
                cpuScratchCapacity = scratchSize
            }
        }
    }

    deinit {
        cpuCountsPtr?.deallocate()
        cpuPositionsPtr?.deallocate()
    }

    public var useGLV = true

    public func msm(points: [PointAffine], scalars: [[UInt32]]) throws -> PointProjective {
        let n = points.count
        guard n == scalars.count, n > 0 else {
            throw MSMError.invalidInput
        }

        var msmPoints = points
        // Reduce scalars mod r to prevent signed-digit carry overflow
        var msmScalars = scalars.map { Self.reduceModR($0) }
        var scalarBits = 256

        var flatScalarBuf: UnsafeMutablePointer<UInt32>? = nil
        var scalarOutMetalBuf: MTLBuffer? = nil

        var neg1Buf: MTLBuffer? = nil
        var neg2Buf: MTLBuffer? = nil
        var glvN: Int = 0

        // GLV: allocate buffers and copy scalars (CPU work only, no GPU wait)
        var glvScalarInBuf: MTLBuffer? = nil
        var glvK1MetalBuf: MTLBuffer? = nil
        var glvK2Offset: Int = 0

        if useGLV && n >= 256 {
            let scalarByteCount = n * 8 * MemoryLayout<UInt32>.stride
            // Reuse cached GLV buffers when possible
            if n > glvCachedN {
                guard let sib = device.makeBuffer(length: scalarByteCount, options: .storageModeShared),
                      let k1b = device.makeBuffer(length: 2 * scalarByteCount, options: .storageModeShared),
                      let n1b = device.makeBuffer(length: n, options: .storageModeShared),
                      let n2b = device.makeBuffer(length: n, options: .storageModeShared) else {
                    throw MSMError.gpuError("Failed to allocate GLV buffers")
                }
                glvScalarInBufCached = sib
                glvK1MetalBufCached = k1b
                glvNeg1BufCached = n1b
                glvNeg2BufCached = n2b
                glvCachedN = n
            }
            let scalarInBuf = glvScalarInBufCached!
            let k1MetalBuf = glvK1MetalBufCached!

            let scalarDst = scalarInBuf.contents().assumingMemoryBound(to: UInt8.self)
            for i in 0..<n {
                scalars[i].withUnsafeBufferPointer { sp in
                    memcpy(scalarDst + i * 32, sp.baseAddress!, 32)
                }
            }

            glvScalarInBuf = scalarInBuf
            glvK1MetalBuf = k1MetalBuf
            glvK2Offset = scalarByteCount
            scalarOutMetalBuf = k1MetalBuf
            flatScalarBuf = k1MetalBuf.contents().bindMemory(to: UInt32.self, capacity: 2 * n * 8)
            neg1Buf = glvNeg1BufCached
            neg2Buf = glvNeg2BufCached

            glvN = n
            scalarBits = 128
        }

        let effectiveN = glvN > 0 ? 2 * glvN : msmPoints.count

        var windowBits: UInt32
        if effectiveN <= 256 {
            windowBits = 8
        } else if effectiveN <= 4096 {
            windowBits = 10
        } else if effectiveN <= 32768 {
            windowBits = 12
        } else {
            windowBits = UInt32(tuning.msmWindowBitsLarge)
        }
        if let wbOverride = windowBitsOverride {
            windowBits = wbOverride
        }
        let nWindows = (scalarBits + Int(windowBits) - 1) / Int(windowBits)
        let fullBuckets = 1 << Int(windowBits)
        let halfBuckets = fullBuckets >> 1
        let nBuckets = halfBuckets + 1  // signed-digit: bucket indices in [0, halfBuckets]
        let nSegments = min(256, max(1, nBuckets / 2))

        ensureBuffers(n: effectiveN, nBuckets: nBuckets, nSegments: nSegments, nWindows: nWindows)
        guard let pointsBuffer = pointsBuffer,
              let sortedIndicesBuffer = sortedIndicesBuffer,
              let allOffsetsBuffer = allOffsetsBuffer,
              let allCountsBuffer = allCountsBuffer,
              let bucketsBuffer = bucketsBuffer,
              let segmentResultsBuffer = segmentResultsBuffer,
              let windowResultsBuffer = windowResultsBuffer,
              let _ = finalResultBuffer,
              let countSortedMapBuffer = countSortedMapBuffer else {
            throw MSMError.gpuError("Failed to allocate Metal buffers")
        }


        let ts = CFAbsoluteTimeGetCurrent()

        let gpuPtsPtr = pointsBuffer.contents().bindMemory(to: PointAffine.self, capacity: effectiveN)
        var endoCmdBuf: MTLCommandBuffer? = nil
        if glvN > 0 {
            // Copy points to GPU buffer before dispatching (shared mode = CPU writes visible to GPU)
            msmPoints.withUnsafeBufferPointer { src in
                gpuPtsPtr.update(from: src.baseAddress!, count: glvN)
            }
            guard let cmdBuf = commandQueue.makeCommandBuffer() else {
                throw MSMError.gpuError("Failed to create preprocessing command buffer")
            }

            let enc = cmdBuf.makeComputeCommandEncoder()!

            // Step 1: GLV decompose (scalars → k1/k2 + neg flags)
            enc.setComputePipelineState(glvDecomposeFunction)
            enc.setBuffer(glvScalarInBuf, offset: 0, index: 0)
            enc.setBuffer(glvK1MetalBuf, offset: 0, index: 1)
            enc.setBuffer(glvK1MetalBuf, offset: glvK2Offset, index: 2)
            enc.setBuffer(neg1Buf, offset: 0, index: 3)
            enc.setBuffer(neg2Buf, offset: 0, index: 4)
            var nVal0 = UInt32(glvN)
            enc.setBytes(&nVal0, length: 4, index: 5)
            let tg0 = min(glvDecomposeFunction.maxTotalThreadsPerThreadgroup, tuning.msmThreadgroupSize)
            enc.dispatchThreads(MTLSize(width: glvN, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: tg0, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)

            // Step 2: Endomorphism (apply neg flags, compute beta*x for second half)
            enc.setComputePipelineState(endomorphismFunction)
            enc.setBuffer(pointsBuffer, offset: 0, index: 0)
            enc.setBuffer(neg1Buf, offset: 0, index: 1)
            enc.setBuffer(neg2Buf, offset: 0, index: 2)
            var nVal1 = UInt32(glvN)
            enc.setBytes(&nVal1, length: 4, index: 3)
            let tg1 = min(endomorphismFunction.maxTotalThreadsPerThreadgroup, tuning.msmThreadgroupSize)
            enc.dispatchThreads(MTLSize(width: glvN, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: tg1, height: 1, depth: 1))

            // Step 3: GPU signed-digit extraction (reads decomposed scalars)
            if let sdBuf = signedDigitBuffer {
                enc.memoryBarrier(scope: .buffers)
                enc.setComputePipelineState(signedDigitFunction)
                enc.setBuffer(scalarOutMetalBuf, offset: 0, index: 0)
                enc.setBuffer(sdBuf, offset: 0, index: 1)
                var enVal = UInt32(effectiveN)
                enc.setBytes(&enVal, length: 4, index: 2)
                var wbVal = windowBits
                enc.setBytes(&wbVal, length: 4, index: 3)
                var nwVal = UInt32(nWindows)
                enc.setBytes(&nwVal, length: 4, index: 4)
                let tg2 = min(signedDigitFunction.maxTotalThreadsPerThreadgroup, tuning.msmThreadgroupSize)
                enc.dispatchThreads(MTLSize(width: effectiveN, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: tg2, height: 1, depth: 1))
            }
            enc.endEncoding()

            cmdBuf.commit()
            endoCmdBuf = cmdBuf
        } else {
            msmPoints.withUnsafeBufferPointer { src in
                gpuPtsPtr.update(from: src.baseAddress!, count: effectiveN)
            }
        }

        let allOffsets = allOffsetsBuffer.contents().bindMemory(to: UInt32.self, capacity: nBuckets * nWindows)
        let allCounts = allCountsBuffer.contents().bindMemory(to: UInt32.self, capacity: nBuckets * nWindows)
        let sortedIdxPtr = sortedIndicesBuffer.contents().bindMemory(to: UInt32.self, capacity: effectiveN * nWindows)
        let countSortedMap = countSortedMapBuffer.contents().bindMemory(to: UInt32.self, capacity: nBuckets * nWindows)

        var params = MsmParams(
            n_points: UInt32(effectiveN),
            window_bits: windowBits,
            n_buckets: UInt32(nBuckets)
        )
        let totalSegments = nSegments * nWindows
        var nSegs = UInt32(nSegments)

        // Capture flat pointers for thread-safe concurrent access (no Swift Array CoW races)
        let countsBase = cpuCountsPtr!
        let positionsBase = cpuPositionsPtr!

        // Phase 0: Signed-digit extraction.
        // If GPU path available (GLV mode with Metal buffer), GPU already computed them.
        // Otherwise, fall back to CPU extraction.
        let signedDigitBuf: UnsafeMutablePointer<UInt32>
        let useGpuSignedDigits = (glvN > 0 && signedDigitBuffer != nil)
        if useGpuSignedDigits {
            // GPU signed_digit_extract was chained into the endo command buffer.
            // Just point to the shared Metal buffer output.
            signedDigitBuf = signedDigitBuffer!.contents().bindMemory(to: UInt32.self, capacity: effectiveN * nWindows)
        } else {
            // CPU fallback for non-GLV or when Metal buffer unavailable
            let sdNeeded = effectiveN * nWindows
            if sdNeeded > signedDigitCapacity {
                signedDigitPtr?.deallocate()
                signedDigitPtr = .allocate(capacity: sdNeeded)
                signedDigitCapacity = sdNeeded
            }
            signedDigitBuf = signedDigitPtr!
            let halfBk = UInt32(halfBuckets)
            let fullBk = UInt32(fullBuckets)
            let chunkSize = 4096
            let nChunks = (effectiveN + chunkSize - 1) / chunkSize
            let wbLocal = windowBits
            let nWLocal = nWindows
            let eN = effectiveN
            let mask = UInt32((1 << windowBits) - 1)
            DispatchQueue.concurrentPerform(iterations: nChunks) { chunk in
                let start = chunk * chunkSize
                let end = min(start + chunkSize, eN)
                for i in start..<end {
                    var carry: UInt32 = 0
                    if let buf = flatScalarBuf {
                        let sp = buf + (i * 8)
                        if wbLocal == 16 {
                            let s0 = sp[0]; let s1 = sp[1]; let s2 = sp[2]; let s3 = sp[3]
                            var d: UInt32
                            d = (s0 & 0xFFFF) &+ carry; carry = 0
                            if d > halfBk { d = fullBk &- d; carry = 1; signedDigitBuf[i] = d | 0x80000000 } else { signedDigitBuf[i] = d }
                            d = (s0 >> 16) &+ carry; carry = 0
                            if d > halfBk { d = fullBk &- d; carry = 1; signedDigitBuf[eN + i] = d | 0x80000000 } else { signedDigitBuf[eN + i] = d }
                            d = (s1 & 0xFFFF) &+ carry; carry = 0
                            if d > halfBk { d = fullBk &- d; carry = 1; signedDigitBuf[2*eN + i] = d | 0x80000000 } else { signedDigitBuf[2*eN + i] = d }
                            d = (s1 >> 16) &+ carry; carry = 0
                            if d > halfBk { d = fullBk &- d; carry = 1; signedDigitBuf[3*eN + i] = d | 0x80000000 } else { signedDigitBuf[3*eN + i] = d }
                            d = (s2 & 0xFFFF) &+ carry; carry = 0
                            if d > halfBk { d = fullBk &- d; carry = 1; signedDigitBuf[4*eN + i] = d | 0x80000000 } else { signedDigitBuf[4*eN + i] = d }
                            d = (s2 >> 16) &+ carry; carry = 0
                            if d > halfBk { d = fullBk &- d; carry = 1; signedDigitBuf[5*eN + i] = d | 0x80000000 } else { signedDigitBuf[5*eN + i] = d }
                            d = (s3 & 0xFFFF) &+ carry; carry = 0
                            if d > halfBk { d = fullBk &- d; carry = 1; signedDigitBuf[6*eN + i] = d | 0x80000000 } else { signedDigitBuf[6*eN + i] = d }
                            d = (s3 >> 16) &+ carry; carry = 0
                            if d > halfBk { d = fullBk &- d; carry = 1; signedDigitBuf[7*eN + i] = d | 0x80000000 } else { signedDigitBuf[7*eN + i] = d }
                        } else {
                            for w in 0..<nWLocal {
                                let bitOff = w * Int(wbLocal)
                                let limbIdx = bitOff / 32
                                let bitPos = bitOff % 32
                                var idx: UInt32 = 0
                                if limbIdx < 8 {
                                    idx = sp[limbIdx] >> bitPos
                                    if bitPos + Int(wbLocal) > 32 && limbIdx + 1 < 8 {
                                        idx |= sp[limbIdx + 1] << (32 - bitPos)
                                    }
                                    idx &= mask
                                }
                                var digit = idx &+ carry
                                carry = 0
                                if digit > halfBk {
                                    digit = fullBk &- digit
                                    carry = 1
                                    signedDigitBuf[w * eN + i] = digit | 0x80000000
                                } else {
                                    signedDigitBuf[w * eN + i] = digit
                                }
                            }
                        }
                    } else {
                        for w in 0..<nWLocal {
                            var digit = UInt32(self.extractBucketIndex(msmScalars[i], windowBits: wbLocal, windowIndex: w)) &+ carry
                            carry = 0
                            if digit > halfBk {
                                digit = fullBk &- digit
                                carry = 1
                                signedDigitBuf[w * eN + i] = digit | 0x80000000
                            } else {
                                signedDigitBuf[w * eN + i] = digit
                            }
                        }
                    }
                }
            }
        }

        func sortWindows(_ windowRange: Range<Int>) {
            DispatchQueue.concurrentPerform(iterations: windowRange.count) { i in
                let w = windowRange.lowerBound + i
                let wOff = w * nBuckets
                let idxBase = w * effectiveN
                let counts = countsBase + w * nBuckets
                let positions = positionsBase + w * nBuckets
                let sdBuf = signedDigitBuf + w * effectiveN

                // Count buckets using pre-computed signed digits
                for i in 0..<nBuckets { counts[i] = 0 }
                for i in 0..<effectiveN {
                    counts[Int(sdBuf[i] & 0x7FFFFFFF)] += 1
                }

                // Prefix sum
                var runningOffset = 0
                for i in 0..<nBuckets {
                    allOffsets[wOff + i] = UInt32(runningOffset)
                    allCounts[wOff + i] = UInt32(counts[i])
                    positions[i] = runningOffset
                    runningOffset += counts[i]
                }

                // Scatter into sorted array, encoding sign bit in upper bit of index
                for i in 0..<effectiveN {
                    let raw = sdBuf[i]
                    let digit = Int(raw & 0x7FFFFFFF)
                    if digit == 0 { continue }
                    var idx = UInt32(i)
                    if (raw & 0x80000000) != 0 { idx |= 0x80000000 }
                    sortedIdxPtr[idxBase + positions[digit]] = idx
                    positions[digit] += 1
                }

                // Build count-sorted map (buckets ordered by descending count for SIMD coherence)
                var maxCount: Int = 0
                for i in 0..<nBuckets {
                    let c = Int(allCounts[wOff + i])
                    if c > maxCount { maxCount = c }
                }
                for i in 0...maxCount { counts[i] = 0 }
                for i in 0..<nBuckets {
                    counts[Int(allCounts[wOff + i])] += 1
                }
                var running = 0
                for c in stride(from: maxCount, through: 0, by: -1) {
                    positions[c] = running
                    running += counts[c]
                }
                for i in 0..<nBuckets {
                    let c = Int(allCounts[wOff + i])
                    let dest = positions[c]
                    positions[c] = dest + 1
                    // Pack: upper 16 bits = window, lower 16 bits = bucket index
                    countSortedMap[wOff + dest] = UInt32(w << 16) | UInt32(i)
                }
            }
        }

        func dispatchReduce(cb: MTLCommandBuffer, windowStart: Int, windowCount: Int) {
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(reduceSortedFunction)
            enc.setBuffer(pointsBuffer, offset: 0, index: 0)
            enc.setBuffer(bucketsBuffer, offset: 0, index: 1)
            enc.setBuffer(allOffsetsBuffer, offset: 0, index: 2)
            enc.setBuffer(allCountsBuffer, offset: 0, index: 3)
            enc.setBytes(&params, length: MemoryLayout<MsmParams>.stride, index: 4)
            var nw = UInt32(windowCount)
            enc.setBytes(&nw, length: MemoryLayout<UInt32>.stride, index: 5)
            enc.setBuffer(sortedIndicesBuffer, offset: 0, index: 6)
            enc.setBuffer(countSortedMapBuffer, offset: windowStart * nBuckets * MemoryLayout<UInt32>.stride, index: 7)
            let numBucketsTotal = windowCount * nBuckets
            let tg = min(tuning.msmThreadgroupSize, Int(reduceSortedFunction.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreads(
                MTLSize(width: numBucketsTotal, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            enc.endEncoding()
        }

        let tPrecompDone = CFAbsoluteTimeGetCurrent()

        // Wait for endo + GPU signed-digit extraction before sort reads the data
        let tBeforeEndoWait = CFAbsoluteTimeGetCurrent()
        endoCmdBuf?.waitUntilCompleted()
        let tAfterEndoWait = CFAbsoluteTimeGetCurrent()

        // Sort ALL windows upfront
        sortWindows(0..<nWindows)

        let tSortDone = CFAbsoluteTimeGetCurrent()

        // Single command buffer: reduce + bucket_sum + combine
        guard let cb = commandQueue.makeCommandBuffer() else { throw MSMError.noCommandBuffer }
        dispatchReduce(cb: cb, windowStart: 0, windowCount: nWindows)

        do {
            var nWinsBatch = UInt32(nWindows)
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(bucketSumDirectFunction)
            enc.setBuffer(bucketsBuffer, offset: 0, index: 0)
            enc.setBuffer(segmentResultsBuffer, offset: 0, index: 1)
            enc.setBytes(&params, length: MemoryLayout<MsmParams>.stride, index: 2)
            enc.setBytes(&nSegs, length: MemoryLayout<UInt32>.stride, index: 3)
            enc.setBytes(&nWinsBatch, length: MemoryLayout<UInt32>.stride, index: 4)
            enc.dispatchThreads(
                MTLSize(width: totalSegments, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(tuning.msmThreadgroupSize, totalSegments), height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)

            enc.setComputePipelineState(combineSegmentsFunction)
            enc.setBuffer(segmentResultsBuffer, offset: 0, index: 0)
            enc.setBuffer(windowResultsBuffer, offset: 0, index: 1)
            enc.setBytes(&nSegs, length: MemoryLayout<UInt32>.stride, index: 2)
            enc.dispatchThreads(
                MTLSize(width: nWindows, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(tuning.msmThreadgroupSize, nWindows), height: 1, depth: 1))
            enc.endEncoding()
        }
        cb.commit()
        cb.waitUntilCompleted()
        let gpuDone = CFAbsoluteTimeGetCurrent()

        if let error = cb.error { throw MSMError.gpuError(error.localizedDescription) }

        let winResultsPtr = windowResultsBuffer.contents().bindMemory(to: PointProjective.self, capacity: nWindows)
        var windowResults = [PointProjective](repeating: pointIdentity(), count: nWindows)
        for w in 0..<nWindows {
            windowResults[w] = winResultsPtr[w]
        }


        let t2 = CFAbsoluteTimeGetCurrent()
        var result = windowResults.last!
        for w in stride(from: nWindows - 2, through: 0, by: -1) {
            for _ in 0..<windowBits {
                result = pointDouble(result)
            }
            result = pointAdd(result, windowResults[w])
        }
        let hornerTime = CFAbsoluteTimeGetCurrent() - t2

        let precompTime = tPrecompDone - ts
        let endoWaitTime = tAfterEndoWait - tBeforeEndoWait
        let actualSortTime = tSortDone - tAfterEndoWait
        let gpuTotalWait = gpuDone - tSortDone
        fputs("  prep: \(String(format: "%.1f", precompTime * 1000))ms, " +
              "endo: \(String(format: "%.1f", endoWaitTime * 1000))ms, " +
              "sort: \(String(format: "%.1f", actualSortTime * 1000))ms, " +
              "gpu: \(String(format: "%.1f", gpuTotalWait * 1000))ms, " +
              "horner: \(String(format: "%.1f", hornerTime * 1000))ms\n", stderr)

        if scalarOutMetalBuf == nil { flatScalarBuf?.deallocate() }
        _ = scalarOutMetalBuf
        return result
    }

}

