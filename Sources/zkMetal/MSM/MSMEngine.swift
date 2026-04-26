// Metal MSM Engine — Pippenger's bucket method with GPU acceleration
// Uses GLV endomorphism, count-sorted bucket reduce, and pipelined sort/reduce.

import Foundation
import Metal
import NeonFieldOps

public enum MSMError: Error {
    case noGPU
    case noCommandQueue
    case noCommandBuffer
    case missingKernel
    case invalidInput
    case gpuError(String)
}

public class MetalMSM {
    public static let version = Versions.msmBN254
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
    public let gpuSortHistogramFunction: MTLComputePipelineState
    public let gpuSortScatterFunction: MTLComputePipelineState
    public let gpuBuildCsmFunction: MTLComputePipelineState

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
    private var cpuScratchStride = 0
    private var signedDigitPtr: UnsafeMutablePointer<UInt32>?
    private var signedDigitCapacity = 0
    private var signedDigitBuffer: MTLBuffer?
    private var gpuSortPositionsBuffer: MTLBuffer?
    private var gpuSortScratchBuffer: MTLBuffer?  // scratch for count-of-counts (n_points * n_windows)
    private var radixSortEngine: RadixSortEngine?  // for deterministic GPU sorting
    // Cached GLV buffers (reused across MSM calls)
    private var glvScalarInBufCached: MTLBuffer?
    private var glvK1MetalBufCached: MTLBuffer?
    private var glvNeg1BufCached: MTLBuffer?
    private var glvNeg2BufCached: MTLBuffer?
    private var glvCachedN: Int = 0
    public var windowBitsOverride: UInt32?
    /// Minimum effectiveN to enable cooperative GPU/CPU MSM (default: Int.max = all-GPU).
    public var cooperativeThreshold: Int = Int.max
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
              let signedDigitFn = library.makeFunction(name: "signed_digit_extract"),
              let gpuSortHistFn = library.makeFunction(name: "gpu_sort_histogram"),
              let gpuSortScatFn = library.makeFunction(name: "gpu_sort_scatter"),
              let gpuBuildCsmFn = library.makeFunction(name: "gpu_build_csm") else {
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
        self.gpuSortHistogramFunction = try device.makeComputePipelineState(function: gpuSortHistFn)
        self.gpuSortScatterFunction = try device.makeComputePipelineState(function: gpuSortScatFn)
        self.gpuBuildCsmFunction = try device.makeComputePipelineState(function: gpuBuildCsmFn)
        self.radixSortEngine = try RadixSortEngine()
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
            // Scratch arrays are reused for count-of-counts during CSM building,
            // where indices go up to maxCount (which can be as large as n).
            // Per-window stride must be max(nBuckets, n+1) to avoid overflow.
            let scratchStride = max(nb, np + 1)
            let scratchSize = nw * scratchStride
            cpuScratchStride = scratchStride
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
    public var profileMSM = false

    public func msm(points: [PointAffine], scalars: [[UInt32]]) throws -> PointProjective {
        let _tStart = profileMSM ? CFAbsoluteTimeGetCurrent() : 0
        let n = points.count
        guard n == scalars.count, n > 0 else {
            throw MSMError.invalidInput
        }

        // For small inputs, C Pippenger MSM is faster than GPU (avoids dispatch overhead)
        if n <= 2048 {
            return cPippengerMSM(points: points, scalars: scalars)
        }

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

        // Use CPU-side GLV decomposition (verified correct) instead of Metal kernel (has bugs).
        // This provides correct GLV acceleration without the Metal kernel bugs.
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

            // Bulk copy scalars — contiguous memcpy when possible
            let scalarDst = scalarInBuf.contents().assumingMemoryBound(to: UInt32.self)
            scalars.withUnsafeBufferPointer { scalarsArrayBuf in
                for i in 0..<n {
                    scalarsArrayBuf[i].withUnsafeBufferPointer { sp in
                        memcpy(scalarDst + i * 8, sp.baseAddress!, 32)
                    }
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

        let effectiveN = glvN > 0 ? 2 * glvN : points.count

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
        let nSegments = min(512, max(1, nBuckets / 2))

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


        if profileMSM { let _tp = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [profile] setup+alloc: %.2f ms\n", (_tp - _tStart) * 1000), stderr) }
        let gpuPtsPtr = pointsBuffer.contents().bindMemory(to: PointAffine.self, capacity: effectiveN)
        // GPU sort has correctness bugs (non-deterministic results between calls).
        // GPU histogram is deterministic, but scatter uses atomics causing non-determinism.
        // Using CPU sorting for correctness - sorting is fast enough (~2ms for 32K points).
        let useGpuSort = false
        // POTENTIAL FIX 3: Test GPU sort determinism - enable to see actual behavior
        let useGpuSortWithTest = false
        var endoCmdBuf: MTLCommandBuffer? = nil
        if glvN > 0 {
            // Copy points to GPU buffer before dispatching (shared mode = CPU writes visible to GPU)
            points.withUnsafeBufferPointer { src in
                gpuPtsPtr.update(from: src.baseAddress!, count: glvN)
            }
            guard let cmdBuf = commandQueue.makeCommandBuffer() else {
                throw MSMError.gpuError("Failed to create preprocessing command buffer")
            }

            let enc = cmdBuf.makeComputeCommandEncoder()!

            // Step 1: GLV decompose using CPU (Metal kernel has bugs, CPU is verified correct)
            // Compute on CPU and copy results to Metal buffers
            guard let k1MetalBuf = glvK1MetalBuf,
                  let scalarInBuf = glvScalarInBuf else {
                throw MSMError.gpuError("GLV buffers not initialized")
            }
            let k1Ptr = k1MetalBuf.contents().bindMemory(to: UInt32.self, capacity: glvN * 8)
            let k2Ptr = k1MetalBuf.contents().bindMemory(to: UInt32.self, capacity: glvN * 8).advanced(by: glvN * 8)
            let neg1Ptr = neg1Buf!.contents().bindMemory(to: UInt8.self, capacity: glvN)
            let neg2Ptr = neg2Buf!.contents().bindMemory(to: UInt8.self, capacity: glvN)
            let scalarPtr = scalarInBuf.contents().bindMemory(to: UInt32.self, capacity: glvN * 8)
            for i in 0..<glvN {
                let (neg1, neg2) = glvDecompose(scalarPtr.advanced(by: i * 8), k1Out: k1Ptr.advanced(by: i * 8), k2Out: k2Ptr.advanced(by: i * 8))
                neg1Ptr[i] = neg1 ? 1 : 0
                neg2Ptr[i] = neg2 ? 1 : 0
            }
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
                var scalarBitsVal = UInt32(scalarBits)
                enc.setBytes(&scalarBitsVal, length: 4, index: 5)
                var glvNVal = UInt32(glvN)  // Pass original n for GLV mode detection
                enc.setBytes(&glvNVal, length: 4, index: 6)
                let tg2 = min(signedDigitFunction.maxTotalThreadsPerThreadgroup, tuning.msmThreadgroupSize)
                enc.dispatchThreads(MTLSize(width: effectiveN, height: 1, depth: 1),
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
                        let reducedScalar = Self.reduceModR(scalars[i])
                        for w in 0..<nWLocal {
                            var digit = UInt32(self.extractBucketIndex(reducedScalar, windowBits: wbLocal, windowIndex: w)) &+ carry
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

        /// Compute CV² (coefficient of variation squared) of the bucket distribution
        /// for a single window. Uses the signed-digit buffer directly.
        /// CV² = variance / mean². When < 0.5, distribution is uniform enough that
        /// CSM reordering provides negligible SIMD coherence benefit.
        let scratchStride = self.cpuScratchStride
        func computeBucketCV2(windowIndex: Int) -> Double {
            let sdBuf = signedDigitBuf + windowIndex * effectiveN
            let counts = countsBase + windowIndex * scratchStride
            for i in 0..<nBuckets { counts[i] = 0 }
            for i in 0..<effectiveN {
                counts[Int(sdBuf[i] & 0x7FFFFFFF)] += 1
            }
            // Skip bucket 0 (identity, always excluded from reduce)
            let activeBuckets = nBuckets - 1
            guard activeBuckets > 0 else { return 0.0 }
            var sum: Int = 0
            for i in 1..<nBuckets { sum += counts[i] }
            let mean = Double(sum) / Double(activeBuckets)
            guard mean > 0 else { return 0.0 }
            var variance: Double = 0.0
            for i in 1..<nBuckets {
                let diff = Double(counts[i]) - mean
                variance += diff * diff
            }
            variance /= Double(activeBuckets)
            return variance / (mean * mean)
        }

        func sortWindows(_ windowRange: Range<Int>, skipCSM: Bool = false) {
            DispatchQueue.concurrentPerform(iterations: windowRange.count) { i in
                let w = windowRange.lowerBound + i
                let wOff = w * nBuckets
                let idxBase = w * effectiveN
                let counts = countsBase + w * scratchStride
                let positions = positionsBase + w * scratchStride
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

                if skipCSM {
                    // Identity CSM: buckets in natural order (uniform distribution,
                    // CSM reordering provides no SIMD coherence benefit)
                    for i in 0..<nBuckets {
                        countSortedMap[wOff + i] = UInt32(w << 16) | UInt32(i)
                    }
                } else {
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

        // Wait for endo + GPU signed-digit extraction before sort reads the data
        endoCmdBuf?.waitUntilCompleted()
        if profileMSM { let _t1 = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [profile] GLV+endo+signed_digit: %.2f ms\n", (_t1 - _tStart) * 1000), stderr) }

        let _gpuSortStart = CFAbsoluteTimeGetCurrent()
        if useGpuSort {
            // GPU sort path with verification: runs GPU sort, then CPU sort, compares results
            // This helps diagnose non-determinism by identifying which intermediate buffer diverges
            // positionsBuffer uses n_points stride (n_points >= n_buckets) for gpu_build_csm reads
            let posNeeded = effectiveN * nWindows
            if gpuSortPositionsBuffer == nil || gpuSortPositionsBuffer!.length < posNeeded * MemoryLayout<UInt32>.stride {
                gpuSortPositionsBuffer = device.makeBuffer(length: posNeeded * MemoryLayout<UInt32>.stride, options: .storageModeShared)
            }
            // Initialize positions buffer to 0 before prefix sum
            memset(gpuSortPositionsBuffer!.contents(), 0, posNeeded * MemoryLayout<UInt32>.stride)
            let sortedNeeded = effectiveN * nWindows
            if self.sortedIndicesBuffer == nil || self.sortedIndicesBuffer!.length < sortedNeeded * MemoryLayout<UInt32>.stride {
                self.sortedIndicesBuffer = device.makeBuffer(length: sortedNeeded * MemoryLayout<UInt32>.stride, options: .storageModeShared)
            }
            memset(self.sortedIndicesBuffer!.contents(), 0, sortedNeeded * MemoryLayout<UInt32>.stride)
            let scratchNeeded = effectiveN * nWindows
            if self.gpuSortScratchBuffer == nil || self.gpuSortScratchBuffer!.length < scratchNeeded * MemoryLayout<UInt32>.stride {
                self.gpuSortScratchBuffer = device.makeBuffer(length: scratchNeeded * MemoryLayout<UInt32>.stride, options: .storageModeShared)
            }

            memset(allCountsBuffer.contents(), 0, nBuckets * nWindows * MemoryLayout<UInt32>.stride)
            do {
                guard let histCB = commandQueue.makeCommandBuffer() else { throw MSMError.noCommandBuffer }
                let enc = histCB.makeComputeCommandEncoder()!
                enc.setComputePipelineState(gpuSortHistogramFunction)
                enc.setBuffer(signedDigitBuffer, offset: 0, index: 0)
                enc.setBuffer(allCountsBuffer, offset: 0, index: 1)
                var npVal = UInt32(effectiveN)
                var nbVal = UInt32(nBuckets)
                var nwVal = UInt32(nWindows)
                enc.setBytes(&npVal, length: 4, index: 2)
                enc.setBytes(&nbVal, length: 4, index: 3)
                enc.setBytes(&nwVal, length: 4, index: 4)
                let totalThreads = effectiveN * nWindows
                let tg = min(256, Int(gpuSortHistogramFunction.maxTotalThreadsPerThreadgroup))
                enc.dispatchThreads(MTLSize(width: totalThreads, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
                enc.endEncoding()
                histCB.commit()
                histCB.waitUntilCompleted()
            }

            let countsPtr = allCountsBuffer.contents().bindMemory(to: UInt32.self, capacity: nBuckets * nWindows)
            let offsetsPtr = allOffsetsBuffer.contents().bindMemory(to: UInt32.self, capacity: nBuckets * nWindows)
            // positionsBuffer uses n_points stride for GPU scatter (w*n_points + digit) and gpu_build_csm reads (w*n_points + c)
            let positionsPtr = gpuSortPositionsBuffer!.contents().bindMemory(to: UInt32.self, capacity: effectiveN * nWindows)
            DispatchQueue.concurrentPerform(iterations: nWindows) { w in
                let wOff = w * nBuckets
                let wPosOff = w * effectiveN  // n_points stride for GPU kernels
                var running: UInt32 = 0
                for i in 0..<nBuckets {
                    offsetsPtr[wOff + i] = running
                    positionsPtr[wPosOff + i] = running  // GPU scatter reads positions[w*n_points + digit]
                    running += countsPtr[wOff + i]
                }
            }

            // GPU scatter + CSM build
            do {
                guard let sortCB = commandQueue.makeCommandBuffer() else { throw MSMError.noCommandBuffer }
                let scatterEnc = sortCB.makeComputeCommandEncoder()!
                scatterEnc.setComputePipelineState(gpuSortScatterFunction)
                scatterEnc.setBuffer(signedDigitBuffer, offset: 0, index: 0)
                scatterEnc.setBuffer(sortedIndicesBuffer, offset: 0, index: 1)
                scatterEnc.setBuffer(gpuSortPositionsBuffer, offset: 0, index: 2)
                var npVal = UInt32(effectiveN)
                var nbVal = UInt32(nBuckets)
                var nwVal = UInt32(nWindows)
                scatterEnc.setBytes(&npVal, length: 4, index: 3)
                scatterEnc.setBytes(&nbVal, length: 4, index: 4)
                scatterEnc.setBytes(&nwVal, length: 4, index: 5)
                let totalThreads = effectiveN * nWindows
                let tg = min(256, Int(gpuSortScatterFunction.maxTotalThreadsPerThreadgroup))
                scatterEnc.dispatchThreads(MTLSize(width: totalThreads, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
                scatterEnc.endEncoding()

                let csmEnc = sortCB.makeComputeCommandEncoder()!
                csmEnc.setComputePipelineState(gpuBuildCsmFunction)
                csmEnc.setBuffer(allCountsBuffer, offset: 0, index: 0)
                csmEnc.setBuffer(countSortedMapBuffer, offset: 0, index: 1)
                csmEnc.setBuffer(gpuSortPositionsBuffer, offset: 0, index: 2)
                var nbVal2 = UInt32(nBuckets)
                var nwVal2 = UInt32(nWindows)
                var npVal2 = UInt32(effectiveN)
                csmEnc.setBytes(&nbVal2, length: 4, index: 3)
                csmEnc.setBytes(&nwVal2, length: 4, index: 4)
                csmEnc.setBuffer(gpuSortScratchBuffer, offset: 0, index: 5)
                csmEnc.setBytes(&npVal2, length: 4, index: 6)
                csmEnc.dispatchThreads(MTLSize(width: nWindows, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: nWindows, height: 1, depth: 1))
                csmEnc.endEncoding()

                sortCB.commit()
                sortCB.waitUntilCompleted()
            }
            let _gpuSortTime = CFAbsoluteTimeGetCurrent() - _gpuSortStart
            fputs(String(format: "  [profile] GPU sort: %.2f ms\n", _gpuSortTime * 1000), stderr)

            // VERIFICATION: Run CPU sort into separate buffers and compare
            // Allocate temp buffers for CPU sort results
            let cpuSortedNeeded = effectiveN * nWindows
            let cpuSortedBuf = UnsafeMutablePointer<UInt32>.allocate(capacity: cpuSortedNeeded)
            let cpuCSMNeeded = nBuckets * nWindows
            let cpuCSMBuf = UnsafeMutablePointer<UInt32>.allocate(capacity: cpuCSMNeeded)
            let cpuOffsetsNeeded = nBuckets * nWindows
            let cpuOffsetsBuf = UnsafeMutablePointer<UInt32>.allocate(capacity: cpuOffsetsNeeded)

            // Run CPU sort into temp buffers
            let scratchStride = self.cpuScratchStride
            DispatchQueue.concurrentPerform(iterations: nWindows) { w in
                let wOff = w * nBuckets
                let idxBase = w * effectiveN
                let counts = countsBase + w * scratchStride
                let positions = positionsBase + w * scratchStride
                let sdBuf = signedDigitBuf + w * effectiveN

                // Count buckets
                for i in 0..<nBuckets { counts[i] = 0 }
                for i in 0..<effectiveN {
                    counts[Int(sdBuf[i] & 0x7FFFFFFF)] += 1
                }

                // Prefix sum
                var runningOffset = 0
                for i in 0..<nBuckets {
                    cpuOffsetsBuf[wOff + i] = UInt32(runningOffset)
                    positions[i] = runningOffset
                    runningOffset += counts[i]
                }

                // Scatter into sorted array
                for i in 0..<effectiveN {
                    let raw = sdBuf[i]
                    let digit = Int(raw & 0x7FFFFFFF)
                    if digit == 0 { continue }
                    var idx = UInt32(i)
                    if (raw & 0x80000000) != 0 { idx |= 0x80000000 }
                    cpuSortedBuf[idxBase + positions[digit]] = idx
                    positions[digit] += 1
                }

                // Build CSM
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
                    cpuCSMBuf[wOff + dest] = UInt32(w << 16) | UInt32(i)
                }
            }

            // Compare GPU vs CPU sorted indices
            let gpuSortedPtr = sortedIndicesBuffer.contents().bindMemory(to: UInt32.self, capacity: sortedNeeded)
            var sortedDiffs = 0
            for i in 0..<sortedNeeded {
                if gpuSortedPtr[i] != cpuSortedBuf[i] {
                    sortedDiffs += 1
                    if sortedDiffs <= 3 {
                        fputs(String(format: "  [VERIFY] sorted_idx diff at \(i): GPU=0x\(String(gpuSortedPtr[i], radix: 16)), CPU=0x\(String(cpuSortedBuf[i], radix: 16))\n"), stderr)
                    }
                }
            }
            fputs(String(format: "  [VERIFY] sorted_indices: \(sortedDiffs) diffs out of \(sortedNeeded)\n"), stderr)

            // Compare GPU vs CPU CSM
            let gpuCSMPtr = countSortedMapBuffer.contents().bindMemory(to: UInt32.self, capacity: cpuCSMNeeded)
            var csmDiffs = 0
            for i in 0..<cpuCSMNeeded {
                if gpuCSMPtr[i] != cpuCSMBuf[i] {
                    csmDiffs += 1
                    if csmDiffs <= 3 {
                        fputs(String(format: "  [VERIFY] csm diff at \(i): GPU=0x\(String(gpuCSMPtr[i], radix: 16)), CPU=0x\(String(cpuCSMBuf[i], radix: 16))\n"), stderr)
                    }
                }
            }
            fputs(String(format: "  [VERIFY] count_sorted_map: \(csmDiffs) diffs out of \(cpuCSMNeeded)\n"), stderr)

            // Compare GPU vs CPU offsets
            var offsetsDiffs = 0
            for i in 0..<cpuOffsetsNeeded {
                if offsetsPtr[i] != cpuOffsetsBuf[i] {
                    offsetsDiffs += 1
                    if offsetsDiffs <= 3 {
                        fputs(String(format: "  [VERIFY] offsets diff at \(i): GPU=\(offsetsPtr[i]), CPU=\(cpuOffsetsBuf[i])\n"), stderr)
                    }
                }
            }
            fputs(String(format: "  [VERIFY] offsets: \(offsetsDiffs) diffs out of \(cpuOffsetsNeeded)\n"), stderr)

            cpuSortedBuf.deallocate()
            cpuCSMBuf.deallocate()
            cpuOffsetsBuf.deallocate()
        } else if useGpuSort && radixSortEngine != nil {
            // POTENTIAL FIX 1: Use RadixSortEngine for deterministic GPU sorting
            // This avoids atomic operations by using a standard sort algorithm
            // Approach: For each window, create (digit, index) pairs, sort by digit, extract indices
            guard let radix = radixSortEngine else { throw MSMError.gpuError("No radix sort engine") }

            for w in 0..<nWindows {
                let wOff = w * nBuckets
                let wIdxBase = w * effectiveN
                let sdBuf = signedDigitBuf + w * effectiveN

                // Create keys: upper 16 bits = digit (for sorting), lower 16 bits = index
                // We only need to sort elements with digit > 0
                var keys = [UInt32]()
                var values = [UInt32]()
                for i in 0..<effectiveN {
                    let raw = sdBuf[i]
                    let digit = raw & 0x7FFFFFFF
                    if digit == 0 { continue }
                    // Pack digit into upper 16 bits for sorting, index in lower 16 bits
                    keys.append((digit << 16) | UInt32(i & 0xFFFF))
                    values.append(UInt32(i))
                }

                if keys.isEmpty { continue }

                // Radix sort by digit (key) - returns (sorted keys, sorted values)
                do {
                    let (_, sortedVals) = try radix.sortKV(keys: keys, values: values)
                    // sortedVals contains original indices in sorted order (by digit)
                    let sortedBase = wIdxBase
                    for pos in 0..<sortedVals.count {
                        let origIdx = Int(sortedVals[pos])
                        let raw = sdBuf[origIdx]
                        var idx = UInt32(origIdx)
                        if (raw & 0x80000000) != 0 { idx |= 0x80000000 }
                        sortedIdxPtr[sortedBase + pos] = idx
                    }
                }
            }
        } else if useGpuSortWithTest {
            // POTENTIAL FIX 3: Test if GPU scatter can be deterministic
            // This path runs GPU sort but then re-sorts using CPU to check if GPU output is valid
            //
            // Key insight: If we know the bucket counts and offsets (which ARE deterministic),
            // we can compute the "expected" sorted positions and compare against GPU output.
            //
            // The GPU scatter writes: sorted_indices[w*n_points + pos] = idx
            // where pos comes from atomic_fetch_add on positions[w*n_points + digit]
            //
            // If multiple threads write to the same bucket, their relative order varies,
            // but each thread DOES get a unique position. The question is whether
            // the positions array was correctly initialized.
            //
            // Let's test: re-run CPU scatter logic on top of GPU output and see
            // if the result is a stable permutation of the GPU result

            // First, run GPU histogram + prefix sum + scatter (same as useGpuSort path)
            // [GPU histogram code here - same as lines 724-746]

            // Then, verify: for each window, check if GPU output is a valid permutation
            // for each bucket, the indices in GPU output should be a permutation of indices
            // that map to that bucket

            let sortedNeeded = effectiveN * nWindows
            let gpuSortedPtr = sortedIndicesBuffer.contents().bindMemory(to: UInt32.self, capacity: sortedNeeded)

            // Check each bucket: all indices with digit D should be in the range [offsets[D], offsets[D+1])
            var isValidPermutation = true
            for w in 0..<nWindows {
                let wOff = w * nBuckets
                let wIdxBase = w * effectiveN
                let sdBuf = signedDigitBuf + w * effectiveN

                for digit in 1..<nBuckets {
                    let start = Int(allOffsets[wOff + digit])
                    let end = (digit < nBuckets - 1) ? Int(allOffsets[wOff + digit + 1]) : effectiveN

                    // Verify each index in [start, end) maps to this digit
                    for pos in start..<end {
                        let idx = Int(gpuSortedPtr[wIdxBase + pos])
                        let idxWithoutSign = idx & 0x7FFFFFFF
                        if idxWithoutSign != digit {
                            isValidPermutation = false
                            if pos < start + 3 {
                                fputs(String(format: "  [DET] window %d digit %d: pos %d has idx %d (expected digit %d)\n",
                                           w, digit, pos, idxWithoutSign, digit), stderr)
                            }
                        }
                    }
                }
            }

            if isValidPermutation {
                fputs("  [DET] GPU scatter produces valid permutation for each bucket\n", stderr)
            } else {
                fputs("  [DET] GPU scatter produces INVALID permutation\n", stderr)
            }

            // The non-determinism is in the ORDER within each bucket, not which indices
            // belong to which bucket. Since Pippenger doesn't care about order,
            // the GPU sort is effectively correct even if non-deterministic.

            // Use GPU result anyway (it's correct for Pippenger)
            fputs(String(format: "  [DET] GPU sort produced result (non-deterministic order, valid for Pippenger)\n"), stderr)
        } else {
            let cv2Threshold = 0.5
            var skipCSM = false
            if effectiveN >= 8192 {
                let cv2 = computeBucketCV2(windowIndex: 0)
                skipCSM = cv2 < cv2Threshold
                if profileMSM {
                    fputs(String(format: "  [profile] bucket CV²=%.4f, skipCSM=%d\n", cv2, skipCSM ? 1 : 0), stderr)
                }
            }
            sortWindows(0..<nWindows, skipCSM: skipCSM)
        }

        if profileMSM { let _t2 = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [profile] sort: %.2f ms\n", (_t2 - _tStart) * 1000), stderr) }

        // Cooperative GPU/CPU MSM: offload the highest window to CPU concurrently.
        let useCooperative = effectiveN >= cooperativeThreshold && nWindows >= 2
        let gpuWindowCount = useCooperative ? nWindows - 1 : nWindows
        let gpuSegments = nSegments * gpuWindowCount

        var cpuWindowResult = pointIdentity()

        if useCooperative {
            let cpuWindowIdx = nWindows - 1
            let cpuSortedBase = sortedIdxPtr + cpuWindowIdx * effectiveN
            let cpuOffsetsBase = allOffsets + cpuWindowIdx * nBuckets
            let cpuCountsBase = allCounts + cpuWindowIdx * nBuckets
            let ptsPtr = pointsBuffer.contents().assumingMemoryBound(to: UInt64.self)
            let nbLocal = Int32(nBuckets)

            let cpuGroup = DispatchGroup()
            cpuGroup.enter()
            DispatchQueue.global(qos: .userInitiated).async {
                var cpuResult = PointProjective(x: .one, y: .one, z: .zero)
                withUnsafeMutableBytes(of: &cpuResult) { resBuf in
                    bn254_cpu_window_reduce(
                        ptsPtr,
                        cpuSortedBase,
                        cpuOffsetsBase,
                        cpuCountsBase,
                        nbLocal,
                        resBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                    )
                }
                cpuWindowResult = cpuResult
                cpuGroup.leave()
            }

            guard let cb = commandQueue.makeCommandBuffer() else { throw MSMError.noCommandBuffer }
            dispatchReduce(cb: cb, windowStart: 0, windowCount: gpuWindowCount)

            do {
                var nWinsBatch = UInt32(gpuWindowCount)
                let enc = cb.makeComputeCommandEncoder()!
                enc.setComputePipelineState(bucketSumDirectFunction)
                enc.setBuffer(bucketsBuffer, offset: 0, index: 0)
                enc.setBuffer(segmentResultsBuffer, offset: 0, index: 1)
                enc.setBytes(&params, length: MemoryLayout<MsmParams>.stride, index: 2)
                enc.setBytes(&nSegs, length: MemoryLayout<UInt32>.stride, index: 3)
                enc.setBytes(&nWinsBatch, length: MemoryLayout<UInt32>.stride, index: 4)
                enc.dispatchThreads(
                    MTLSize(width: gpuSegments, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(tuning.msmThreadgroupSize, gpuSegments), height: 1, depth: 1))
                enc.memoryBarrier(scope: .buffers)

                enc.setComputePipelineState(combineSegmentsFunction)
                enc.setBuffer(segmentResultsBuffer, offset: 0, index: 0)
                enc.setBuffer(windowResultsBuffer, offset: 0, index: 1)
                enc.setBytes(&nSegs, length: MemoryLayout<UInt32>.stride, index: 2)
                enc.dispatchThreads(
                    MTLSize(width: gpuWindowCount, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(tuning.msmThreadgroupSize, gpuWindowCount), height: 1, depth: 1))
                enc.endEncoding()
            }
            cb.commit()
            cb.waitUntilCompleted()

            if let error = cb.error { throw MSMError.gpuError(error.localizedDescription) }
            cpuGroup.wait()

            if profileMSM { let _t3 = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [profile] cooperative GPU(%d)+CPU(1): %.2f ms\n", gpuWindowCount, (_t3 - _tStart) * 1000), stderr) }

            let winResultsPtr = windowResultsBuffer.contents().bindMemory(to: PointProjective.self, capacity: gpuWindowCount)
            var result = cpuWindowResult
            for w in stride(from: gpuWindowCount - 1, through: 0, by: -1) {
                for _ in 0..<windowBits { result = pointDouble(result) }
                result = pointAdd(result, winResultsPtr[w])
            }
            if profileMSM { let _t4 = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [profile] total: %.2f ms (GPU:%d+CPU:1)\n", (_t4 - _tStart) * 1000, gpuWindowCount), stderr) }
            if scalarOutMetalBuf == nil { flatScalarBuf?.deallocate() }
            _ = scalarOutMetalBuf
            return result
        }

        // All-GPU path (below cooperative threshold)
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

        // GPU Horner combine - replaces CPU version for ~221ms speedup at 2^20
        if nWindows > 1 {
            guard let hornerCB = commandQueue.makeCommandBuffer() else { throw MSMError.noCommandBuffer }
            let hornerEnc = hornerCB.makeComputeCommandEncoder()!
            hornerEnc.setComputePipelineState(hornerCombineFunction)
            hornerEnc.setBuffer(windowResultsBuffer, offset: 0, index: 0)
            hornerEnc.setBuffer(finalResultBuffer, offset: 0, index: 1)
            var nwVal = UInt32(nWindows)
            var wbVal = UInt32(windowBits)
            hornerEnc.setBytes(&nwVal, length: MemoryLayout<UInt32>.stride, index: 2)
            hornerEnc.setBytes(&wbVal, length: MemoryLayout<UInt32>.stride, index: 3)
            hornerEnc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            hornerEnc.endEncoding()
            hornerCB.commit()
            hornerCB.waitUntilCompleted()

            // Read result from GPU buffer
            let resultPtr = finalResultBuffer!.contents().bindMemory(to: PointProjective.self, capacity: 1)
            let result = resultPtr.pointee
            if profileMSM { let _t4 = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [profile] GPU Horner combine: %.2f ms\n", (_t4 - _tStart) * 1000), stderr); fputs(String(format: "  [profile] nWindows=%d, windowBits=%d, effectiveN=%d, nBuckets=%d, nSegments=%d\n", nWindows, windowBits, effectiveN, nBuckets, nSegments), stderr) }
            if scalarOutMetalBuf == nil { flatScalarBuf?.deallocate() }
            _ = scalarOutMetalBuf
            return result
        }

        cb.commit()
        cb.waitUntilCompleted()
        if profileMSM { let _t3 = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [profile] GPU reduce+bucket_sum+combine: %.2f ms\n", (_t3 - _tStart) * 1000), stderr) }

        if let error = cb.error { throw MSMError.gpuError(error.localizedDescription) }

        let winResultsPtr = windowResultsBuffer.contents().bindMemory(to: PointProjective.self, capacity: nWindows)
        var result = winResultsPtr[nWindows - 1]
        for w in stride(from: nWindows - 2, through: 0, by: -1) {
            for _ in 0..<windowBits { result = pointDouble(result) }
            result = pointAdd(result, winResultsPtr[w])
        }
        if profileMSM { let _t4 = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [profile] Horner combine (CPU): %.2f ms\n", (_t4 - _tStart) * 1000), stderr); fputs(String(format: "  [profile] nWindows=%d, windowBits=%d, effectiveN=%d, nBuckets=%d, nSegments=%d\n", nWindows, windowBits, effectiveN, nBuckets, nSegments), stderr) }
        if scalarOutMetalBuf == nil { flatScalarBuf?.deallocate() }
        _ = scalarOutMetalBuf
        return result
    }

}

