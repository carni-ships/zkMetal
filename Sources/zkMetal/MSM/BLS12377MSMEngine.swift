// BLS12-377 Metal MSM Engine — Pippenger's bucket method with GPU acceleration
// Supports GLV endomorphism for ~2× speedup (128-bit scalar decomposition).

import Foundation
import Metal

public class BLS12377MSM {
    public static let version = Versions.msmBLS12377
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    private let reduceSortedFunction: MTLComputePipelineState
    private let reduceCooperativeFunction: MTLComputePipelineState
    private let bucketSumDirectFunction: MTLComputePipelineState
    private let combineSegmentsFunction: MTLComputePipelineState
    private let hornerCombineFunction: MTLComputePipelineState
    private let signedDigitFunction: MTLComputePipelineState
    private let gpuSortHistogramFunction: MTLComputePipelineState
    private let gpuSortScatterFunction: MTLComputePipelineState
    private let gpuBuildCsmFunction: MTLComputePipelineState
    private let glvDecomposeFunction: MTLComputePipelineState
    private let glvEndomorphismFunction: MTLComputePipelineState
    private let glvCopyAndEndoFunction: MTLComputePipelineState?
    private let glvSignedDigitFunction: MTLComputePipelineState

    // Pre-allocated buffers
    private var maxAllocatedPoints = 0
    private var maxAllocatedBuckets = 0
    private var maxAllocatedWindows = 0
    private var maxAllocatedSegments = 0
    private var pointsBuffer: MTLBuffer?
    private var sortedIndicesBuffer: MTLBuffer?
    private var allOffsetsBuffer: MTLBuffer?
    private var allCountsBuffer: MTLBuffer?
    private var bucketsBuffer: MTLBuffer?
    private var segmentResultsBuffer: MTLBuffer?
    private var windowResultsBuffer: MTLBuffer?
    private var finalResultBuffer: MTLBuffer?
    private var countSortedMapBuffer: MTLBuffer?
    private var signedDigitBuffer: MTLBuffer?
    private var gpuSortPositionsBuffer: MTLBuffer?
    private var cpuCountsPtr: UnsafeMutablePointer<Int>?
    private var cpuPositionsPtr: UnsafeMutablePointer<Int>?
    private var cpuScratchCapacity = 0
    private var signedDigitPtr: UnsafeMutablePointer<UInt16>?
    private var signedDigitCapacity = 0
    // GLV buffers
    private var glvScalarInBufCached: MTLBuffer?
    private var glvK1MetalBufCached: MTLBuffer?
    private var glvNeg1BufCached: MTLBuffer?
    private var glvNeg2BufCached: MTLBuffer?
    private var glvEndoFlagBuffer: MTLBuffer?  // flags for k1 (0) vs k2 (1) path
    private var glvCachedN: Int = 0
    public var windowBitsOverride: UInt32?
    // GLV endomorphism: halves scalar width (253→128 bits) at cost of 2× points
    // For BLS12-377's 12-limb (384-bit) Fq, point additions are ~2× costlier than BN254,
    // so doubling points outweighs halving windows. Disabled by default for GPU path.
    // CPU Pippenger still uses GLV (in the C implementation).
    public var useGLV = false
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
        let cacheFile = BLS12377MSM.cacheDir.appendingPathComponent("bls12377_msm.metallib")

        let requiredKernels = [
            "msm377_reduce_sorted_buckets", "msm377_bucket_sum_direct",
            "msm377_combine_segments", "msm377_signed_digit_extract",
              "msm377_signed_digit_extract_glv"
        ]
        if FileManager.default.fileExists(atPath: cacheFile.path) {
            do {
                let cached = try device.makeLibrary(URL: cacheFile)
                if requiredKernels.allSatisfy({ cached.makeFunction(name: $0) != nil }) {
                    library = cached
                } else {
                    library = try BLS12377MSM.compileAndCache(device: device, cacheFile: cacheFile)
                }
            } catch {
                library = try BLS12377MSM.compileAndCache(device: device, cacheFile: cacheFile)
            }
        } else {
            library = try BLS12377MSM.compileAndCache(device: device, cacheFile: cacheFile)
        }

        guard let reduceSortedFn = library.makeFunction(name: "msm377_reduce_sorted_buckets"),
              let reduceCoopFn = library.makeFunction(name: "msm377_reduce_cooperative"),
              let sumDirectFn = library.makeFunction(name: "msm377_bucket_sum_direct"),
              let combineFn = library.makeFunction(name: "msm377_combine_segments"),
              let hornerFn = library.makeFunction(name: "msm377_horner_combine"),
              let signedDigitFn = library.makeFunction(name: "msm377_signed_digit_extract"),
              let gpuSortHistFn = library.makeFunction(name: "msm377_sort_histogram"),
              let gpuSortScatFn = library.makeFunction(name: "msm377_sort_scatter"),
              let gpuBuildCsmFn = library.makeFunction(name: "msm377_build_csm"),
              let glvDecomposeFn = library.makeFunction(name: "glv377_decompose"),
              let glvEndoFn = library.makeFunction(name: "glv377_endomorphism"),
              let glvCopyEndoFn = library.makeFunction(name: "glv377_copy_and_endo") else {
            throw MSMError.missingKernel
        }

        self.reduceSortedFunction = try device.makeComputePipelineState(function: reduceSortedFn)
        self.reduceCooperativeFunction = try device.makeComputePipelineState(function: reduceCoopFn)
        self.bucketSumDirectFunction = try device.makeComputePipelineState(function: sumDirectFn)
        self.combineSegmentsFunction = try device.makeComputePipelineState(function: combineFn)
        self.hornerCombineFunction = try device.makeComputePipelineState(function: hornerFn)
        self.signedDigitFunction = try device.makeComputePipelineState(function: signedDigitFn)
        self.gpuSortHistogramFunction = try device.makeComputePipelineState(function: gpuSortHistFn)
        self.gpuSortScatterFunction = try device.makeComputePipelineState(function: gpuSortScatFn)
        self.gpuBuildCsmFunction = try device.makeComputePipelineState(function: gpuBuildCsmFn)
        self.glvDecomposeFunction = try device.makeComputePipelineState(function: glvDecomposeFn)
        self.glvEndomorphismFunction = try device.makeComputePipelineState(function: glvEndoFn)
        self.glvCopyAndEndoFunction = try? device.makeComputePipelineState(function: glvCopyEndoFn)
        self.glvSignedDigitFunction = try device.makeComputePipelineState(function: library.makeFunction(name: "msm377_signed_digit_extract_glv")!)
        self.tuning = TuningManager.shared.config(device: device)
    }

    private static func compileAndCache(device: MTLDevice, cacheFile: URL) throws -> MTLLibrary {
        let shaderDir = findShaderDir()

        let fqSource = try String(contentsOfFile: shaderDir + "/fields/bls12377_fq.metal", encoding: .utf8)
        let curveSource = try String(contentsOfFile: shaderDir + "/geometry/bls12377_curve.metal", encoding: .utf8)
        let glvSource = try String(contentsOfFile: shaderDir + "/msm/bls12377_glv_kernels.metal", encoding: .utf8)
        let msmSource = try String(contentsOfFile: shaderDir + "/msm/bls12377_msm_kernels.metal", encoding: .utf8)

        func stripIncludes(_ s: String) -> String {
            s.split(separator: "\n").filter { !$0.contains("#include") }.joined(separator: "\n")
        }
        func stripGuards(_ s: String) -> String {
            s.replacingOccurrences(of: "#ifndef BLS12377_FQ_METAL", with: "")
             .replacingOccurrences(of: "#define BLS12377_FQ_METAL", with: "")
             .replacingOccurrences(of: "#endif // BLS12377_FQ_METAL", with: "")
             .replacingOccurrences(of: "#ifndef BLS12377_CURVE_METAL", with: "")
             .replacingOccurrences(of: "#define BLS12377_CURVE_METAL", with: "")
             .replacingOccurrences(of: "#endif // BLS12377_CURVE_METAL", with: "")
        }

        let combined = stripGuards(fqSource) + "\n" +
                        stripGuards(stripIncludes(curveSource)) + "\n" +
                        stripIncludes(glvSource) + "\n" +
                        stripIncludes(msmSource)

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        let library = try device.makeLibrary(source: combined, options: options)

        try? FileManager.default.createDirectory(
            at: BLS12377MSM.cacheDir, withIntermediateDirectories: true)

        if #available(macOS 11.0, *) {
            let archiveDesc = MTLBinaryArchiveDescriptor()
            if let archive = try? device.makeBinaryArchive(descriptor: archiveDesc) {
                for name in ["msm377_reduce_sorted_buckets", "msm377_bucket_sum_direct"] {
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
                let path = url.appendingPathComponent("fields/bls12377_fq.metal").path
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
            if FileManager.default.fileExists(atPath: "\(path)/fields/bls12377_fq.metal") {
                return path
            }
        }
        return "./Sources/Shaders"
    }

    // Fr377 scalar field order r as 8x32-bit limbs (little-endian)
    private static let R_LIMBS: [UInt32] = [
        0x00000001, 0x0a118000, 0xd0000001, 0x59aa76fe,
        0x5c37b001, 0x60b44d1e, 0x9a2ca556, 0x12ab655e
    ]

    public static func reduceModR(_ scalar: [UInt32]) -> [UInt32] {
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

    private func ensureBuffers(n: Int, nBuckets: Int, nSegments: Int, nWindows: Int) {
        let needRealloc = n > maxAllocatedPoints || nBuckets > maxAllocatedBuckets ||
                          nWindows > maxAllocatedWindows || nSegments > maxAllocatedSegments
        if needRealloc {
            let np = max(n, maxAllocatedPoints)
            let nb = max(nBuckets, maxAllocatedBuckets)
            let nw = max(nWindows, maxAllocatedWindows)
            let ns = nSegments
            pointsBuffer = device.makeBuffer(
                length: MemoryLayout<Point377Affine>.stride * np, options: .storageModeShared)
            sortedIndicesBuffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride * np * nw, options: .storageModeShared)
            allOffsetsBuffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride * nb * nw, options: .storageModeShared)
            allCountsBuffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride * nb * nw, options: .storageModeShared)
            bucketsBuffer = device.makeBuffer(
                length: MemoryLayout<Point377Projective>.stride * nb * nw, options: .storageModeShared)
            segmentResultsBuffer = device.makeBuffer(
                length: MemoryLayout<Point377Projective>.stride * ns * nw, options: .storageModeShared)
            windowResultsBuffer = device.makeBuffer(
                length: MemoryLayout<Point377Projective>.stride * nw, options: .storageModeShared)
            finalResultBuffer = device.makeBuffer(
                length: MemoryLayout<Point377Projective>.stride, options: .storageModeShared)
            countSortedMapBuffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride * nb * nw, options: .storageModeShared)
            // GLV uses 2x point count for k1+k2 digits, stored as UInt16 (15-bit digits fit in 16 bits)
            let sdSize = np * 2 * nw
            signedDigitBuffer = device.makeBuffer(
                length: 2 * sdSize, options: .storageModeShared)  // 2 bytes per digit (UInt16)
            maxAllocatedPoints = np
            maxAllocatedBuckets = nb
            maxAllocatedWindows = nw
            maxAllocatedSegments = ns
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

    public func msm(points: [Point377Affine], scalars: [[UInt32]]) throws -> Point377Projective {
        let n = points.count
        guard n == scalars.count, n > 0 else {
            throw MSMError.invalidInput
        }

        // For small n, CPU Pippenger is faster than GPU (avoids command buffer overhead)
        if n <= 2048 {
            let msmScalars = scalars.map { Self.reduceModR($0) }
            return bls12377CpuMSM(points: points, scalars: msmScalars)
        }

        let msmScalars: [[UInt32]]
        if n >= 4096 {
            var par = [[UInt32]](repeating: [], count: n)
            DispatchQueue.concurrentPerform(iterations: n) { i in
                par[i] = Self.reduceModR(scalars[i])
            }
            msmScalars = par
        } else {
            msmScalars = scalars.map { Self.reduceModR($0) }
        }
        var scalarBits = 253
        var glvN = 0

        // GLV setup
        var flatScalarBuf: UnsafeMutablePointer<UInt32>? = nil
        var scalarOutMetalBuf: MTLBuffer? = nil
        var neg1Buf: MTLBuffer? = nil
        var neg2Buf: MTLBuffer? = nil
        var glvScalarInBuf: MTLBuffer? = nil
        var glvK1MetalBuf: MTLBuffer? = nil
        var glvK2Offset: Int = 0

        if useGLV && n >= 256 {
            let scalarByteCount = n * 8 * MemoryLayout<UInt32>.stride
            if n > glvCachedN {
                guard let sib = device.makeBuffer(length: scalarByteCount, options: .storageModeShared),
                      let k1b = device.makeBuffer(length: 2 * scalarByteCount, options: .storageModeShared),
                      let n1b = device.makeBuffer(length: n, options: .storageModeShared),
                      let n2b = device.makeBuffer(length: n, options: .storageModeShared),
                      let endoBuf = device.makeBuffer(length: 2 * n, options: .storageModeShared) else {
                    throw MSMError.gpuError("Failed to allocate GLV buffers")
                }
                glvScalarInBufCached = sib
                glvK1MetalBufCached = k1b
                glvNeg1BufCached = n1b
                glvNeg2BufCached = n2b
                glvEndoFlagBuffer = endoBuf
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

        // For on-the-fly endomorphism: keep effectiveN = n (not 2*n)
        // This halves memory usage for points buffer
        let effectiveN = n

        // Window sizing tuned for 12-limb Fq377.
        // All window sizes perform similarly on M3 Pro for large n.
        var windowBits: UInt32
        if glvN > 0 {
            // GLV path: 128-bit scalars, 2× points. Fewer windows needed.
            if effectiveN <= 512 {
                windowBits = 8
            } else if effectiveN <= 8192 {
                windowBits = 11
            } else if effectiveN <= 65536 {
                windowBits = 15
            } else {
                windowBits = 15
            }
        } else {
            // Non-GLV path: 253-bit scalars.
            if effectiveN <= 256 {
                windowBits = 8
            } else if effectiveN <= 2048 {
                windowBits = 10
            } else if effectiveN <= 32768 {
                windowBits = 11
            } else {
                windowBits = 15
            }
        }
        if let wbOverride = windowBitsOverride {
            windowBits = wbOverride
        }
        let nWindows = (scalarBits + Int(windowBits) - 1) / Int(windowBits)
        let fullBuckets = 1 << Int(windowBits)
        let halfBuckets = fullBuckets >> 1
        let nBuckets = halfBuckets + 1
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

        let gpuPtsPtr = pointsBuffer.contents().bindMemory(to: Point377Affine.self, capacity: effectiveN)
        var endoCmdBuf: MTLCommandBuffer? = nil

        if glvN > 0 {
            // Copy points to GPU buffer (no precomputed endomorphism - using on-the-fly)
            points.withUnsafeBufferPointer { src in
                gpuPtsPtr.update(from: src.baseAddress!, count: glvN)
            }
            guard let cmdBuf = commandQueue.makeCommandBuffer() else {
                throw MSMError.gpuError("Failed to create preprocessing command buffer")
            }

            let enc = cmdBuf.makeComputeCommandEncoder()!

            // Step 1: GLV decompose (k1, k2, neg flags)
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

            // Step 2: GLV-aware GPU signed-digit extraction (combines k1 and k2 digits)
            // Digits for k1 stored at [0, n), k2 at [n, 2n)
            // On-the-fly endomorphism will use idx >= n to trigger β·x computation
            if let sdBuf = signedDigitBuffer, let endoBuf = glvEndoFlagBuffer {
                enc.setComputePipelineState(glvSignedDigitFunction)
                enc.setBuffer(glvK1MetalBuf, offset: 0, index: 0)            // k1 scalars
                enc.setBuffer(glvK1MetalBuf, offset: glvK2Offset, index: 1)   // k2 scalars
                enc.setBuffer(sdBuf, offset: 0, index: 2)                      // combined digits
                enc.setBuffer(endoBuf, offset: 0, index: 3)                     // endo flags
                var nVal2 = UInt32(glvN)
                enc.setBytes(&nVal2, length: 4, index: 4)
                var wbVal2 = windowBits
                enc.setBytes(&wbVal2, length: 4, index: 5)
                var nwVal2 = UInt32(nWindows)
                enc.setBytes(&nwVal2, length: 4, index: 6)
                let tg2 = min(glvSignedDigitFunction.maxTotalThreadsPerThreadgroup, tuning.msmThreadgroupSize)
                enc.dispatchThreads(MTLSize(width: glvN, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: tg2, height: 1, depth: 1))
            }

            enc.endEncoding()
            cmdBuf.commit()
            endoCmdBuf = cmdBuf
        } else {
            points.withUnsafeBufferPointer { src in
                gpuPtsPtr.update(from: src.baseAddress!, count: effectiveN)
            }
        }

        let allOffsets = allOffsetsBuffer.contents().bindMemory(to: UInt32.self, capacity: nBuckets * nWindows)
        let allCounts = allCountsBuffer.contents().bindMemory(to: UInt32.self, capacity: nBuckets * nWindows)
        let sortedIdxPtr = sortedIndicesBuffer.contents().bindMemory(to: UInt32.self, capacity: effectiveN * nWindows)
        let countSortedMap = countSortedMapBuffer.contents().bindMemory(to: UInt32.self, capacity: nBuckets * nWindows)

        var params = Msm377Params(
            n_points: UInt32(effectiveN),
            window_bits: windowBits,
            n_buckets: UInt32(nBuckets)
        )
        var nSegs = UInt32(nSegments)

        let countsBase = cpuCountsPtr!
        let positionsBase = cpuPositionsPtr!

        // Signed-digit extraction
        let useGpuSignedDigits = (glvN > 0 && signedDigitBuffer != nil)
        let signedDigitBuf: UnsafeMutablePointer<UInt16>

        if useGpuSignedDigits {
            signedDigitBuf = signedDigitBuffer!.contents().bindMemory(to: UInt16.self, capacity: effectiveN * 2 * nWindows)
        } else {
            let sdNeeded = effectiveN * nWindows
            if sdNeeded > signedDigitCapacity {
                signedDigitPtr?.deallocate()
                signedDigitPtr = .allocate(capacity: sdNeeded)
                signedDigitCapacity = sdNeeded
            }
            signedDigitBuf = signedDigitPtr!
            let halfBk = UInt32(halfBuckets)
            let fullBk = UInt32(fullBuckets)
            let mask = UInt32((1 << windowBits) - 1)
            let wbLocal = windowBits
            let nWLocal = nWindows
            let eN = effectiveN

            let chunkSize = 4096
            let nChunks = (effectiveN + chunkSize - 1) / chunkSize
            DispatchQueue.concurrentPerform(iterations: nChunks) { chunk in
                let start = chunk * chunkSize
                let end = min(start + chunkSize, eN)
                for i in start..<end {
                    var carry: UInt32 = 0
                    msmScalars[i].withUnsafeBufferPointer { sp in
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
                                // UInt16: low 15 bits for digit, bit 15 for neg flag (0x8000)
                                signedDigitBuf[w * eN + i] = UInt16(digit) | 0x8000
                            } else {
                                signedDigitBuf[w * eN + i] = UInt16(digit)
                            }
                        }
                    }
                }
            }
        }

        // Wait for endo + GPU signed-digit extraction
        endoCmdBuf?.waitUntilCompleted()

        // Count-sort per window
        // For GLV: process both k1 [0, n) and k2 [n, 2n) digits
        let glvTotalPoints = (glvN > 0) ? (2 * glvN) : effectiveN
        DispatchQueue.concurrentPerform(iterations: nWindows) { w in
            let wOff = w * nBuckets
            let idxBase = w * glvTotalPoints
            let counts = countsBase + w * nBuckets
            let positions = positionsBase + w * nBuckets
            let sdBuf = signedDigitBuf + w * glvTotalPoints

            for i in 0..<nBuckets { counts[i] = 0 }
            for i in 0..<glvTotalPoints {
                counts[Int(sdBuf[i] & 0x7FFF)] += 1
            }

            var runningOffset = 0
            for i in 0..<nBuckets {
                allOffsets[wOff + i] = UInt32(runningOffset)
                allCounts[wOff + i] = UInt32(counts[i])
                positions[i] = runningOffset
                runningOffset += counts[i]
            }

            for i in 0..<glvTotalPoints {
                let raw = sdBuf[i]
                let digit = Int(raw & 0x7FFF)
                if digit == 0 { continue }
                var idx = UInt32(i)
                if (raw & 0x8000) != 0 { idx |= 0x80000000 }
                sortedIdxPtr[idxBase + positions[digit]] = idx
                positions[digit] += 1
            }

            // Build count-sorted map
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
                countSortedMap[wOff + dest] = UInt32(w << 16) | UInt32(i)
            }
        }

        let tSortDone = CFAbsoluteTimeGetCurrent()

        // Single command buffer: reduce + bucket_sum + combine
        guard let cb = commandQueue.makeCommandBuffer() else { throw MSMError.noCommandBuffer }

        // Phase 1: Reduce sorted buckets
        do {
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(reduceSortedFunction)
            enc.setBuffer(pointsBuffer, offset: 0, index: 0)
            enc.setBuffer(bucketsBuffer, offset: 0, index: 1)
            enc.setBuffer(allOffsetsBuffer, offset: 0, index: 2)
            enc.setBuffer(allCountsBuffer, offset: 0, index: 3)
            enc.setBytes(&params, length: MemoryLayout<Msm377Params>.stride, index: 4)
            var nw = UInt32(nWindows)
            enc.setBytes(&nw, length: MemoryLayout<UInt32>.stride, index: 5)
            enc.setBuffer(sortedIndicesBuffer, offset: 0, index: 6)
            enc.setBuffer(countSortedMapBuffer, offset: 0, index: 7)
            let numBucketsTotal = nWindows * nBuckets
            let tg = min(tuning.msmThreadgroupSize, Int(reduceSortedFunction.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreads(
                MTLSize(width: numBucketsTotal, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            enc.endEncoding()
        }

        // Phase 2: Bucket sum + combine
        do {
            var nWinsBatch = UInt32(nWindows)
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(bucketSumDirectFunction)
            enc.setBuffer(bucketsBuffer, offset: 0, index: 0)
            enc.setBuffer(segmentResultsBuffer, offset: 0, index: 1)
            enc.setBytes(&params, length: MemoryLayout<Msm377Params>.stride, index: 2)
            enc.setBytes(&nSegs, length: MemoryLayout<UInt32>.stride, index: 3)
            enc.setBytes(&nWinsBatch, length: MemoryLayout<UInt32>.stride, index: 4)
            let totalSegments = nSegments * nWindows
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

        let winResultsPtr = windowResultsBuffer.contents().bindMemory(to: Point377Projective.self, capacity: nWindows)
        var windowResults = [Point377Projective](repeating: point377Identity(), count: nWindows)
        for w in 0..<nWindows {
            windowResults[w] = winResultsPtr[w]
        }

        // Horner's method on CPU
        var result = windowResults.last!
        for w in stride(from: nWindows - 2, through: 0, by: -1) {
            for _ in 0..<windowBits {
                result = point377Double(result)
            }
            result = point377Add(result, windowResults[w])
        }
        let totalTime = CFAbsoluteTimeGetCurrent() - ts

        fputs("  sort: \(String(format: "%.1f", (tSortDone - ts) * 1000))ms, " +
              "gpu: \(String(format: "%.1f", (gpuDone - tSortDone) * 1000))ms, " +
              "total: \(String(format: "%.1f", totalTime * 1000))ms\n", stderr)

        return result
    }
}

// Msm377Params must match Metal struct layout
public struct Msm377Params {
    public var n_points: UInt32
    public var window_bits: UInt32
    public var n_buckets: UInt32

    public init(n_points: UInt32, window_bits: UInt32, n_buckets: UInt32) {
        self.n_points = n_points
        self.window_bits = window_bits
        self.n_buckets = n_buckets
    }
}
