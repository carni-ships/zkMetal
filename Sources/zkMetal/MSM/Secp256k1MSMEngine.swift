// secp256k1 Metal MSM Engine — Pippenger's bucket method with GPU acceleration
// GLV endomorphism: k = k1 + k2·λ, φ(P) = (β·x, y)

import Foundation
import Metal
import NeonFieldOps

public class Secp256k1MSM {
    public static let version = Versions.msmSecp256k1
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    private let reduceSortedFunction: MTLComputePipelineState
    private let reduceWarpPerBucketFunction: MTLComputePipelineState
    private let reduceCooperativeFunction: MTLComputePipelineState
    private let reduceSharedMemFunction: MTLComputePipelineState
    private let bucketSumDirectFunction: MTLComputePipelineState
    private let combineSegmentsFunction: MTLComputePipelineState
    private let hornerCombineFunction: MTLComputePipelineState
    private let signedDigitFunction: MTLComputePipelineState
    private let gpuSortHistogramFunction: MTLComputePipelineState
    private let gpuSortScatterFunction: MTLComputePipelineState
    private let gpuBuildCsmFunction: MTLComputePipelineState
    private let glvDecomposeFunction: MTLComputePipelineState
    private let glvEndomorphismFunction: MTLComputePipelineState
    private let batchMSMBatchFunction: MTLComputePipelineState
    private let batchNAFBatchFunction: MTLComputePipelineState

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
    private var cpuCountsPtr: UnsafeMutablePointer<Int>?
    private var cpuPositionsPtr: UnsafeMutablePointer<Int>?
    private var cpuScratchCapacity = 0
    private var signedDigitPtr: UnsafeMutablePointer<UInt32>?
    private var signedDigitCapacity = 0
    private var scalarsGPUBuffer: MTLBuffer?
    // GLV cached buffers
    private var glvScalarInBufCached: MTLBuffer?
    private var glvK1MetalBufCached: MTLBuffer?
    private var glvNeg1BufCached: MTLBuffer?
    private var glvNeg2BufCached: MTLBuffer?
    private var glvCachedN: Int = 0
    // Precomputed GLV pairs: stores (P, beta*P) pairs precomputed during SRS loading
    // Eliminating the GPU endomorphism kernel (~50ms) with ~5ms CPU precomputation
    private var precomputedGLVPairsBuffer: MTLBuffer?
    private var precomputedGLVPairsCount: Int = 0
    private var precomputedGLVPairsInitialized: Bool = false
    public var windowBitsOverride: UInt32?
    public var useGLV = false  // GLV regresses on M3 GPU: 2x points costs more than halved scalars
    public var useGPUSort = false  // Use GPU sorting kernels instead of CPU (experimental)
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
        let cacheFile = Secp256k1MSM.cacheDir.appendingPathComponent("secp256k1_msm.metallib")

        let requiredKernels = [
            "secp_msm_reduce_sorted_buckets", "secp_msm_reduce_warp_per_bucket",
            "secp_msm_reduce_shared_mem", "secp_msm_bucket_sum_direct",
            "secp_msm_combine_segments", "secp_msm_signed_digit_extract",
            "secp_glv_decompose", "secp_glv_endomorphism",
            "secp_msm_batch_small", "secp_msm_batch_small_naf"
        ]
        if FileManager.default.fileExists(atPath: cacheFile.path) {
            do {
                let cached = try device.makeLibrary(URL: cacheFile)
                if requiredKernels.allSatisfy({ cached.makeFunction(name: $0) != nil }) {
                    library = cached
                } else {
                    library = try Secp256k1MSM.compileAndCache(device: device, cacheFile: cacheFile)
                }
            } catch {
                library = try Secp256k1MSM.compileAndCache(device: device, cacheFile: cacheFile)
            }
        } else {
            library = try Secp256k1MSM.compileAndCache(device: device, cacheFile: cacheFile)
        }

        guard let reduceSortedFn = library.makeFunction(name: "secp_msm_reduce_sorted_buckets"),
              let reduceWarpFn = library.makeFunction(name: "secp_msm_reduce_warp_per_bucket"),
              let reduceCoopFn = library.makeFunction(name: "secp_msm_reduce_cooperative"),
              let reduceSharedMemFn = library.makeFunction(name: "secp_msm_reduce_shared_mem"),
              let sumDirectFn = library.makeFunction(name: "secp_msm_bucket_sum_direct"),
              let combineFn = library.makeFunction(name: "secp_msm_combine_segments"),
              let hornerFn = library.makeFunction(name: "secp_msm_horner_combine"),
              let signedDigitFn = library.makeFunction(name: "secp_msm_signed_digit_extract"),
              let gpuSortHistFn = library.makeFunction(name: "secp_msm_sort_histogram"),
              let gpuSortScatFn = library.makeFunction(name: "secp_msm_sort_scatter"),
              let gpuBuildCsmFn = library.makeFunction(name: "secp_msm_build_csm"),
              let glvDecomposeFn = library.makeFunction(name: "secp_glv_decompose"),
              let glvEndoFn = library.makeFunction(name: "secp_glv_endomorphism"),
              let batchMSMFn = library.makeFunction(name: "secp_msm_batch_small"),
              let batchNAFFn = library.makeFunction(name: "secp_msm_batch_small_naf") else {
            throw MSMError.missingKernel
        }

        self.reduceSortedFunction = try device.makeComputePipelineState(function: reduceSortedFn)
        self.reduceWarpPerBucketFunction = try device.makeComputePipelineState(function: reduceWarpFn)
        self.reduceCooperativeFunction = try device.makeComputePipelineState(function: reduceCoopFn)
        self.reduceSharedMemFunction = try device.makeComputePipelineState(function: reduceSharedMemFn)
        self.bucketSumDirectFunction = try device.makeComputePipelineState(function: sumDirectFn)
        self.combineSegmentsFunction = try device.makeComputePipelineState(function: combineFn)
        self.hornerCombineFunction = try device.makeComputePipelineState(function: hornerFn)
        self.signedDigitFunction = try device.makeComputePipelineState(function: signedDigitFn)
        self.gpuSortHistogramFunction = try device.makeComputePipelineState(function: gpuSortHistFn)
        self.gpuSortScatterFunction = try device.makeComputePipelineState(function: gpuSortScatFn)
        self.gpuBuildCsmFunction = try device.makeComputePipelineState(function: gpuBuildCsmFn)
        self.glvDecomposeFunction = try device.makeComputePipelineState(function: glvDecomposeFn)
        self.glvEndomorphismFunction = try device.makeComputePipelineState(function: glvEndoFn)
        self.batchMSMBatchFunction = try device.makeComputePipelineState(function: batchMSMFn)
        self.batchNAFBatchFunction = try device.makeComputePipelineState(function: batchNAFFn)
        self.tuning = TuningManager.shared.config(device: device)
    }

    private static func compileAndCache(device: MTLDevice, cacheFile: URL) throws -> MTLLibrary {
        let shaderDir = findShaderDir()

        let fpSource = try String(contentsOfFile: shaderDir + "/fields/secp256k1_fp.metal", encoding: .utf8)
        let curveSource = try String(contentsOfFile: shaderDir + "/geometry/secp256k1_curve.metal", encoding: .utf8)
        let glvSource = try String(contentsOfFile: shaderDir + "/msm/secp256k1_glv_kernels.metal", encoding: .utf8)
        let msmSource = try String(contentsOfFile: shaderDir + "/msm/secp256k1_msm_kernels.metal", encoding: .utf8)

        func stripIncludes(_ s: String) -> String {
            s.split(separator: "\n").filter { !$0.contains("#include") }.joined(separator: "\n")
        }
        func stripGuards(_ s: String) -> String {
            s.replacingOccurrences(of: "#ifndef SECP256K1_FP_METAL", with: "")
             .replacingOccurrences(of: "#define SECP256K1_FP_METAL", with: "")
             .replacingOccurrences(of: "#endif // SECP256K1_FP_METAL", with: "")
             .replacingOccurrences(of: "#ifndef SECP256K1_CURVE_METAL", with: "")
             .replacingOccurrences(of: "#define SECP256K1_CURVE_METAL", with: "")
             .replacingOccurrences(of: "#endif // SECP256K1_CURVE_METAL", with: "")
        }

        let combined = stripGuards(fpSource) + "\n" +
                        stripGuards(stripIncludes(curveSource)) + "\n" +
                        stripIncludes(glvSource) + "\n" +
                        stripIncludes(msmSource)

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        let library = try device.makeLibrary(source: combined, options: options)

        try? FileManager.default.createDirectory(
            at: Secp256k1MSM.cacheDir, withIntermediateDirectories: true)

        if #available(macOS 11.0, *) {
            let archiveDesc = MTLBinaryArchiveDescriptor()
            if let archive = try? device.makeBinaryArchive(descriptor: archiveDesc) {
                for name in ["secp_msm_reduce_sorted_buckets", "secp_msm_bucket_sum_direct"] {
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
                let path = url.appendingPathComponent("fields/secp256k1_fp.metal").path
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
            if FileManager.default.fileExists(atPath: "\(path)/fields/secp256k1_fp.metal") {
                return path
            }
        }
        return "./Sources/Shaders"
    }

    // secp256k1 curve order n as 8x32-bit limbs (little-endian)
    private static let N_LIMBS: [UInt32] = [
        0xd0364141, 0xbfd25e8c, 0xaf48a03b, 0xbaaedce6,
        0xfffffffe, 0xffffffff, 0xffffffff, 0xffffffff
    ]

    // n/2 (half order) for scalar centering
    private static let HALF_N: [UInt32] = {
        var r = [UInt32](repeating: 0, count: 8)
        var carry: UInt32 = 0
        for i in stride(from: 7, through: 0, by: -1) {
            let v = UInt32(truncatingIfNeeded: (UInt64(N_LIMBS[i]) + UInt64(carry)) >> 1)
            carry = N_LIMBS[i] & 1
            r[i] = v | (i < 7 ? 0 : 0)
        }
        // Simpler: shift right by 1
        for i in 0..<8 {
            r[i] = N_LIMBS[i] >> 1
            if i < 7 { r[i] |= (N_LIMBS[i+1] & 1) << 31 }
        }
        return r
    }()

    public static func reduceModN(_ scalar: [UInt32]) -> [UInt32] {
        var current = scalar
        while true {
            if !gte(current, N_LIMBS) { return current }
            var result = [UInt32](repeating: 0, count: 8)
            var borrow: Int64 = 0
            for i in 0..<8 {
                borrow += Int64(current[i]) - Int64(N_LIMBS[i])
                result[i] = UInt32(truncatingIfNeeded: borrow & 0xFFFFFFFF)
                borrow >>= 32
            }
            current = result
        }
    }

    private static func gte(_ a: [UInt32], _ b: [UInt32]) -> Bool {
        for i in stride(from: 7, through: 0, by: -1) {
            if a[i] > b[i] { return true }
            if a[i] < b[i] { return false }
        }
        return true
    }

    // n - scalar (assumes scalar < n)
    private static func subN(_ scalar: [UInt32]) -> [UInt32] {
        var result = [UInt32](repeating: 0, count: 8)
        var borrow: Int64 = 0
        for i in 0..<8 {
            borrow += Int64(N_LIMBS[i]) - Int64(scalar[i])
            result[i] = UInt32(truncatingIfNeeded: borrow & 0xFFFFFFFF)
            borrow >>= 32
        }
        return result
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
                length: MemoryLayout<SecpPointAffine>.stride * np, options: .storageModeShared)
            sortedIndicesBuffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride * np * nw, options: .storageModeShared)
            allOffsetsBuffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride * nb * nw, options: .storageModeShared)
            allCountsBuffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride * nb * nw, options: .storageModeShared)
            bucketsBuffer = device.makeBuffer(
                length: MemoryLayout<SecpPointProjective>.stride * nb * nw, options: .storageModeShared)
            segmentResultsBuffer = device.makeBuffer(
                length: MemoryLayout<SecpPointProjective>.stride * ns * nw, options: .storageModeShared)
            windowResultsBuffer = device.makeBuffer(
                length: MemoryLayout<SecpPointProjective>.stride * nw, options: .storageModeShared)
            finalResultBuffer = device.makeBuffer(
                length: MemoryLayout<SecpPointProjective>.stride, options: .storageModeShared)
            countSortedMapBuffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride * nb * nw, options: .storageModeShared)
            signedDigitBuffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride * np * nw, options: .storageModeShared)
            // GPU scalars buffer: n * 8 uint32s for GPU signed-digit extract kernel
            scalarsGPUBuffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride * np * 8, options: .storageModeShared)
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

    // MARK: - GLV Precomputation

    /// Precompute GLV endomorphism pairs (P, beta*P) for a set of generator points.
    /// Call this during SRS loading to eliminate the GPU endomorphism kernel at prove time.
    /// The endomorphism beta*P is computed on CPU (~10x faster than GPU for secp_fp mul).
    ///
    /// - Parameter points: The G1 generator points (e.g., SRS powers) to precompute endomorphism for
    /// - Precondition: points.count must be > 0 and <= 2^20
    public func precomputeGLVPairs(for points: [SecpPointAffine]) {
        let n = points.count
        guard n > 0, n <= 1 << 20 else { return }

        // Allocate buffer for 2n points (pairs)
        let pairCount = 2 * n
        if precomputedGLVPairsBuffer == nil || precomputedGLVPairsCount < pairCount {
            precomputedGLVPairsBuffer = device.makeBuffer(
                length: MemoryLayout<SecpPointAffine>.stride * pairCount,
                options: .storageModeShared)
            precomputedGLVPairsCount = pairCount
        }
        guard let buf = precomputedGLVPairsBuffer else { return }

        let pairsPtr = buf.contents().bindMemory(to: SecpPointAffine.self, capacity: pairCount)

        // Compute (P, beta*P) pairs in parallel on CPU
        // beta is a constant, so this is embarrassingly parallel
        DispatchQueue.concurrentPerform(iterations: n) { i in
            let p = points[i]
            let betaP = Secp256k1GLV.applyEndomorphism(p)
            pairsPtr[i] = p           // Original point at [0, n)
            pairsPtr[n + i] = betaP   // Endomorphed point at [n, 2n)
        }

        precomputedGLVPairsInitialized = true
    }

    /// Check if GLV pairs are precomputed for the given point count.
    public func hasPrecomputedGLVPairs(count: Int) -> Bool {
        return precomputedGLVPairsInitialized && precomputedGLVPairsCount >= 2 * count
    }

    public func msm(points: [SecpPointAffine], scalars: [[UInt32]], useCPUGLV: Bool = false) throws -> SecpPointProjective {
        let n = points.count
        guard n == scalars.count, n > 0 else {
            throw MSMError.invalidInput
        }

        // For small inputs, C Pippenger MSM is faster than GPU
        if n <= 1024 {
            return cSecpPippengerMSM(points: points, scalars: scalars)
        }

        let msmScalars: [[UInt32]]
        if n >= 4096 {
            var par = [[UInt32]](repeating: [], count: n)
            DispatchQueue.concurrentPerform(iterations: n) { i in
                par[i] = Self.reduceModN(scalars[i])
            }
            msmScalars = par
        } else {
            msmScalars = scalars.map { Self.reduceModN($0) }
        }
        var scalarBits = 256

        var glvN: Int = 0
        var neg1Buf: MTLBuffer? = nil
        var neg2Buf: MTLBuffer? = nil
        var glvScalars: [[UInt32]]? = nil
        // Centered points for non-GLV (scalar > n/2 → negate point, use n-scalar)
        var centeredPoints: [SecpPointAffine]? = nil
        var centeredScalars: [[UInt32]]? = nil

        // Center non-GLV scalars to prevent signed-digit carry overflow
        // secp256k1 n ≈ 2^256, so uncented scalars can have top byte 0xFF
        // causing carry to overflow past the last window
        if !(useGLV || useCPUGLV) && n >= 256 {
            var cPts = points
            var cScls = msmScalars
            for i in 0..<n {
                if Self.gte(cScls[i], Self.HALF_N) {
                    // scalar > n/2: use (n - scalar) and negate point
                    cScls[i] = Self.subN(cScls[i])
                    cPts[i] = secpPointNegateAffine(cPts[i])
                }
            }
            centeredPoints = cPts
            centeredScalars = cScls
        }

        if (useGLV || useCPUGLV) && n >= 256 {
            // CPU-side GLV decomposition (verified correct, 3.5ms)
            // k1/k2 written to GPU buffer — MSM kernels read from it unchanged
            var k1s = [[UInt32]]()
            var k2s = [[UInt32]]()
            var neg1s = [UInt8](repeating: 0, count: n)
            var neg2s = [UInt8](repeating: 0, count: n)
            k1s.reserveCapacity(n)
            k2s.reserveCapacity(n)
            for i in 0..<n {
                let (k1, k2, n1, n2) = Secp256k1GLV.decompose(scalars[i])
                k1s.append(k1)
                k2s.append(k2)
                neg1s[i] = n1 ? 1 : 0
                neg2s[i] = n2 ? 1 : 0
            }
            glvScalars = k1s + k2s  // 2*n scalars: first n are k1, next n are k2

            // Allocate neg flag buffers
            if n > glvCachedN {
                glvNeg1BufCached = device.makeBuffer(length: n, options: .storageModeShared)
                glvNeg2BufCached = device.makeBuffer(length: n, options: .storageModeShared)
                glvCachedN = n
            }
            neg1Buf = glvNeg1BufCached!
            neg2Buf = glvNeg2BufCached!

            // Copy neg flags to GPU
            memcpy(neg1Buf!.contents(), neg1s, n)
            memcpy(neg2Buf!.contents(), neg2s, n)

            // CPU GLV: decompose on CPU (3.5ms) and write k1/k2 to GPU buffer
            // This avoids the GPU secp_glv_decompose kernel (12ms) — MSM kernels read
            // k1/k2 from this buffer just as they would from the GPU-kernel-produced buffer.
            let scalarByteCount = n * 8 * MemoryLayout<UInt32>.stride  // n scalars × 8 uint32s × 4 bytes
            let neededSize = 2 * scalarByteCount
            if n > glvCachedN {
                // Always reallocate to ensure sufficient size
                glvK1MetalBufCached = device.makeBuffer(length: neededSize, options: .storageModeShared)
                glvCachedN = n
            }
            let k1MetalBuf = glvK1MetalBufCached!
            // glvScalars = [k1_0..k1_{n-1}, k2_0..k2_{n-1}], each scalar is 8 uint32s
            glvScalars!.withUnsafeBufferPointer { scalarsArrayBuf in
                let flat = k1MetalBuf.contents().bindMemory(to: UInt32.self, capacity: 2 * n * 8)
                // Copy k1 scalars (first n scalars)
                for i in 0..<n {
                    scalarsArrayBuf[i].withUnsafeBufferPointer { sp in
                        memcpy(flat + i * 8, sp.baseAddress!, 32)
                    }
                }
                // Copy k2 scalars (next n scalars)
                for i in 0..<n {
                    scalarsArrayBuf[n + i].withUnsafeBufferPointer { sp in
                        memcpy(flat + (n + i) * 8, sp.baseAddress!, 32)
                    }
                }
            }

            glvN = n
            scalarBits = 129  // k1/k2 ≈ 128 bits, +1 for signed-digit carry
        }

        let effectiveN = glvN > 0 ? 2 * glvN : n

        var windowBits: UInt32
        if effectiveN <= 256 {
            windowBits = 8
        } else if effectiveN <= 4096 {
            windowBits = 10
        } else if effectiveN <= 65536 {
            windowBits = 13  // secp256k1: wb=13 (4097 bkts) avoids M3 GPU pathology at 2049/16K bkts
        } else {
            windowBits = UInt32(tuning.msmWindowBitsLarge)  // wb=16 for large N
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

        let gpuPtsPtr = pointsBuffer.contents().bindMemory(to: SecpPointAffine.self, capacity: effectiveN)

        if glvN > 0 {
            // Check if precomputed GLV pairs are available (from SRS precomputation)
            if hasPrecomputedGLVPairs(count: glvN) {
                // Use precomputed pairs: (P, beta*P) precomputed during SRS loading
                // neg1 and neg2 flags still need to be applied per-point
                // neg1: negate y of original point, neg2: negate y of endomorphed point
                let neg1Ptr = neg1Buf!.contents().bindMemory(to: UInt8.self, capacity: glvN)
                let neg2Ptr = neg2Buf!.contents().bindMemory(to: UInt8.self, capacity: glvN)
                let prePairsPtr = precomputedGLVPairsBuffer!.contents().bindMemory(to: SecpPointAffine.self, capacity: 2 * glvN)

                // Apply neg1 to original points and copy both original and precomputed endomorphed
                DispatchQueue.concurrentPerform(iterations: glvN) { i in
                    let orig = points[i]
                    let betaP = prePairsPtr[glvN + i]  // Precomputed beta*P

                    if neg1Ptr[i] != 0 {
                        // Apply neg1: negate y of original point
                        gpuPtsPtr[i] = SecpPointAffine(x: orig.x, y: secpNeg(orig.y))
                    } else {
                        gpuPtsPtr[i] = orig
                    }

                    // Apply neg2 to precomputed beta*P if needed
                    if neg2Ptr[i] != 0 {
                        gpuPtsPtr[glvN + i] = SecpPointAffine(x: betaP.x, y: secpNeg(betaP.y))
                    } else {
                        gpuPtsPtr[glvN + i] = betaP
                    }
                }
            } else {
                // No precomputed pairs: use GPU endomorphism kernel (original path)
                // Copy original points (endomorphism will create second half)
                points.withUnsafeBufferPointer { src in
                    gpuPtsPtr.update(from: src.baseAddress!, count: glvN)
                }
                // GPU endomorphism: apply neg flags + create (β·x, y) points
                guard let cmdBuf = commandQueue.makeCommandBuffer() else {
                    throw MSMError.gpuError("Failed to create endomorphism command buffer")
                }
                let enc = cmdBuf.makeComputeCommandEncoder()!
                enc.setComputePipelineState(glvEndomorphismFunction)
                enc.setBuffer(pointsBuffer, offset: 0, index: 0)
                enc.setBuffer(neg1Buf, offset: 0, index: 1)
                enc.setBuffer(neg2Buf, offset: 0, index: 2)
                var nVal = UInt32(glvN)
                enc.setBytes(&nVal, length: 4, index: 3)
                let tg = min(glvEndomorphismFunction.maxTotalThreadsPerThreadgroup, tuning.msmThreadgroupSize)
                enc.dispatchThreads(MTLSize(width: glvN, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
                enc.endEncoding()
                cmdBuf.commit()
                cmdBuf.waitUntilCompleted()
                if let error = cmdBuf.error {
                    throw MSMError.gpuError("Endomorphism error: \(error.localizedDescription)")
                }
            }
        } else {
            let ptsToUse = centeredPoints ?? points
            ptsToUse.withUnsafeBufferPointer { src in
                gpuPtsPtr.update(from: src.baseAddress!, count: effectiveN)
            }
        }

        let allOffsets = allOffsetsBuffer.contents().bindMemory(to: UInt32.self, capacity: nBuckets * nWindows)
        let allCounts = allCountsBuffer.contents().bindMemory(to: UInt32.self, capacity: nBuckets * nWindows)
        let sortedIdxPtr = sortedIndicesBuffer.contents().bindMemory(to: UInt32.self, capacity: effectiveN * nWindows)
        let countSortedMap = countSortedMapBuffer.contents().bindMemory(to: UInt32.self, capacity: nBuckets * nWindows)

        var params = SecpMsmParamsSwift(
            n_points: UInt32(effectiveN),
            window_bits: windowBits,
            n_buckets: UInt32(nBuckets)
        )
        var nSegs = UInt32(nSegments)

        let countsBase = cpuCountsPtr!
        let positionsBase = cpuPositionsPtr!

        // activeScalars is used by both GPU and CPU paths
        let activeScalars = glvScalars ?? centeredScalars ?? msmScalars

        // GPU sort path: use GPU kernels for extraction and counting sort
        if useGPUSort && effectiveN >= 4096 && nWindows >= 2 {
            // scalarsGPUBuffer and signedDigitBuffer are class properties (optional)
            // The rest are shadowed local variables from guard let (non-optional)
            let scalarsGPUBuf = scalarsGPUBuffer!
            let digitsBuf = signedDigitBuffer!
            // countsBuf, sortedIdxBuf, csmBuf are already non-optional (shadowed)
            let countsBuf = allCountsBuffer
            let sortedIdxBuf = sortedIndicesBuffer
            let csmBuf = countSortedMapBuffer

            // Copy scalars to GPU buffer
            let scalarByteCount = effectiveN * 8 * MemoryLayout<UInt32>.stride
            _ = scalarByteCount  // suppress unused warning
            activeScalars.withUnsafeBufferPointer { scalarsArrayBuf in
                let gpuScalarsPtr = scalarsGPUBuf.contents().bindMemory(to: UInt32.self, capacity: effectiveN * 8)
                for i in 0..<effectiveN {
                    scalarsArrayBuf[i].withUnsafeBufferPointer { sp in
                        memcpy(gpuScalarsPtr + i * 8, sp.baseAddress!, 32)
                    }
                }
            }

            guard let cb = commandQueue.makeCommandBuffer() else {
                throw MSMError.noCommandBuffer
            }

            // Phase 1: GPU signed-digit extraction
            do {
                let enc = cb.makeComputeCommandEncoder()!
                enc.setComputePipelineState(signedDigitFunction)
                enc.setBuffer(scalarsGPUBuf, offset: 0, index: 0)
                enc.setBuffer(digitsBuf, offset: 0, index: 1)
                var nPts = UInt32(effectiveN)
                var wb = windowBits
                var nWin = UInt32(nWindows)
                enc.setBytes(&nPts, length: 4, index: 2)
                enc.setBytes(&wb, length: 4, index: 3)
                enc.setBytes(&nWin, length: 4, index: 4)
                enc.dispatchThreads(
                    MTLSize(width: effectiveN, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(256, effectiveN), height: 1, depth: 1))
                enc.endEncoding()
            }

            // Phase 2: GPU histogram (zero counts first via CPU, then GPU adds to them)
            let countsPtr = countsBuf.contents().bindMemory(to: UInt32.self, capacity: nBuckets * nWindows)
            for i in 0..<nBuckets * nWindows { countsPtr[i] = 0 }

            do {
                let enc = cb.makeComputeCommandEncoder()!
                enc.setComputePipelineState(gpuSortHistogramFunction)
                enc.setBuffer(digitsBuf, offset: 0, index: 0)
                enc.setBuffer(countsBuf, offset: 0, index: 1)
                var nPts = UInt32(effectiveN)
                var nBkts = UInt32(nBuckets)
                var nWin = UInt32(nWindows)
                enc.setBytes(&nPts, length: 4, index: 2)
                enc.setBytes(&nBkts, length: 4, index: 3)
                enc.setBytes(&nWin, length: 4, index: 4)
                enc.dispatchThreads(
                    MTLSize(width: effectiveN * nWindows, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(256, effectiveN * nWindows), height: 1, depth: 1))
                enc.endEncoding()
            }

            // Phase 3: GPU scatter
            do {
                let enc = cb.makeComputeCommandEncoder()!
                enc.setComputePipelineState(gpuSortScatterFunction)
                enc.setBuffer(digitsBuf, offset: 0, index: 0)
                enc.setBuffer(sortedIdxBuf, offset: 0, index: 1)
                enc.setBuffer(countsBuf, offset: 0, index: 2)
                var nPts = UInt32(effectiveN)
                var nBkts = UInt32(nBuckets)
                var nWin = UInt32(nWindows)
                enc.setBytes(&nPts, length: 4, index: 3)
                enc.setBytes(&nBkts, length: 4, index: 4)
                enc.setBytes(&nWin, length: 4, index: 5)
                enc.dispatchThreads(
                    MTLSize(width: effectiveN * nWindows, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(256, effectiveN * nWindows), height: 1, depth: 1))
                enc.endEncoding()
            }

            // Phase 4: GPU build count-sorted map
            do {
                let enc = cb.makeComputeCommandEncoder()!
                enc.setComputePipelineState(gpuBuildCsmFunction)
                enc.setBuffer(countsBuf, offset: 0, index: 0)
                enc.setBuffer(csmBuf, offset: 0, index: 1)
                enc.setBuffer(countsBuf, offset: 0, index: 2)  // reuse counts as offsets output
                var nBkts = UInt32(nBuckets)
                var nWin = UInt32(nWindows)
                enc.setBytes(&nBkts, length: 4, index: 3)
                enc.setBytes(&nWin, length: 4, index: 4)
                enc.dispatchThreads(
                    MTLSize(width: nWindows, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(256, nWindows), height: 1, depth: 1))
                enc.endEncoding()
            }

            cb.commit()
            cb.waitUntilCompleted()

            // Skip CPU extraction and counting sort - go directly to GPU reduction
            guard let cb2 = commandQueue.makeCommandBuffer() else { throw MSMError.noCommandBuffer }

            // Phase 1: Reduce sorted buckets
            do {
                let enc = cb2.makeComputeCommandEncoder()!
                let numBucketsTotal = nWindows * nBuckets
                if nBuckets <= 1024 {
                    enc.setComputePipelineState(reduceWarpPerBucketFunction)
                    enc.setBuffer(pointsBuffer, offset: 0, index: 0)
                    enc.setBuffer(bucketsBuffer, offset: 0, index: 1)
                    enc.setBuffer(allOffsetsBuffer, offset: 0, index: 2)
                    enc.setBuffer(allCountsBuffer, offset: 0, index: 3)
                    enc.setBytes(&params, length: MemoryLayout<SecpMsmParamsSwift>.stride, index: 4)
                    var nw = UInt32(nWindows)
                    enc.setBytes(&nw, length: MemoryLayout<UInt32>.stride, index: 5)
                    enc.setBuffer(sortedIndicesBuffer, offset: 0, index: 6)
                    enc.setBuffer(countSortedMapBuffer, offset: 0, index: 7)
                    enc.dispatchThreadgroups(
                        MTLSize(width: numBucketsTotal, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                } else {
                    enc.setComputePipelineState(reduceSharedMemFunction)
                    enc.setBuffer(pointsBuffer, offset: 0, index: 0)
                    enc.setBuffer(bucketsBuffer, offset: 0, index: 1)
                    enc.setBuffer(allOffsetsBuffer, offset: 0, index: 2)
                    enc.setBuffer(allCountsBuffer, offset: 0, index: 3)
                    enc.setBytes(&params, length: MemoryLayout<SecpMsmParamsSwift>.stride, index: 4)
                    var nw = UInt32(nWindows)
                    enc.setBytes(&nw, length: MemoryLayout<UInt32>.stride, index: 5)
                    enc.setBuffer(sortedIndicesBuffer, offset: 0, index: 6)
                    enc.setBuffer(countSortedMapBuffer, offset: 0, index: 7)
                    enc.dispatchThreadgroups(
                        MTLSize(width: numBucketsTotal, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                }
                enc.endEncoding()
            }

            // Phase 2: Bucket sum + combine
            do {
                var nWinsBatch = UInt32(nWindows)
                let enc = cb2.makeComputeCommandEncoder()!
                enc.setComputePipelineState(bucketSumDirectFunction)
                enc.setBuffer(bucketsBuffer, offset: 0, index: 0)
                enc.setBuffer(segmentResultsBuffer, offset: 0, index: 1)
                enc.setBytes(&params, length: MemoryLayout<SecpMsmParamsSwift>.stride, index: 2)
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
            cb2.commit()
            cb2.waitUntilCompleted()

            if let error = cb2.error { throw MSMError.gpuError(error.localizedDescription) }

            let winResultsPtr = windowResultsBuffer.contents().bindMemory(to: SecpPointProjective.self, capacity: nWindows)
            var windowResults = [SecpPointProjective](repeating: secpPointIdentity(), count: nWindows)
            for w in 0..<nWindows {
                windowResults[w] = winResultsPtr[w]
            }

            var result = windowResults.last!
            for w in stride(from: nWindows - 2, through: 0, by: -1) {
                for _ in 0..<windowBits {
                    result = secpPointDouble(result)
                }
                result = secpPointAdd(result, windowResults[w])
            }
            return result
        }

        // CPU signed-digit extraction (works for both GLV and non-GLV)
        // Note: activeScalars is declared above for GPU sort compatibility
        do {
            let sdNeeded = effectiveN * nWindows
            if sdNeeded > signedDigitCapacity {
                signedDigitPtr?.deallocate()
                signedDigitPtr = .allocate(capacity: sdNeeded)
                signedDigitCapacity = sdNeeded
            }
            let signedDigitBuf = signedDigitPtr!
            let halfBk = UInt32(halfBuckets)
            let fullBk = UInt32(fullBuckets)
            let mask = UInt32((1 << windowBits) - 1)
            let wbLocal = windowBits
            let nWLocal = nWindows
            let eN = effectiveN

            // Flatten scalars into contiguous buffer for better cache/pointer performance
            let flatBuf = UnsafeMutablePointer<UInt32>.allocate(capacity: eN * 8)
            activeScalars.withUnsafeBufferPointer { scalarsArrayBuf in
                for i in 0..<eN {
                    scalarsArrayBuf[i].withUnsafeBufferPointer { sp in
                        memcpy(flatBuf + i * 8, sp.baseAddress!, 32)
                    }
                }
            }

            let chunkSize = 4096
            let nChunks = (effectiveN + chunkSize - 1) / chunkSize
            DispatchQueue.concurrentPerform(iterations: nChunks) { chunk in
                let start = chunk * chunkSize
                let end = min(start + chunkSize, eN)
                for i in start..<end {
                    var carry: UInt32 = 0
                    let sp = flatBuf + (i * 8)
                    if wbLocal == 16 {
                        // Unrolled wb=16 path: each window is a 16-bit half-limb
                        let s0 = sp[0]; let s1 = sp[1]; let s2 = sp[2]; let s3 = sp[3]
                        let s4 = sp[4]; let s5 = sp[5]; let s6 = sp[6]; let s7 = sp[7]
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
                        d = (s4 & 0xFFFF) &+ carry; carry = 0
                        if d > halfBk { d = fullBk &- d; carry = 1; signedDigitBuf[8*eN + i] = d | 0x80000000 } else { signedDigitBuf[8*eN + i] = d }
                        d = (s4 >> 16) &+ carry; carry = 0
                        if d > halfBk { d = fullBk &- d; carry = 1; signedDigitBuf[9*eN + i] = d | 0x80000000 } else { signedDigitBuf[9*eN + i] = d }
                        d = (s5 & 0xFFFF) &+ carry; carry = 0
                        if d > halfBk { d = fullBk &- d; carry = 1; signedDigitBuf[10*eN + i] = d | 0x80000000 } else { signedDigitBuf[10*eN + i] = d }
                        d = (s5 >> 16) &+ carry; carry = 0
                        if d > halfBk { d = fullBk &- d; carry = 1; signedDigitBuf[11*eN + i] = d | 0x80000000 } else { signedDigitBuf[11*eN + i] = d }
                        d = (s6 & 0xFFFF) &+ carry; carry = 0
                        if d > halfBk { d = fullBk &- d; carry = 1; signedDigitBuf[12*eN + i] = d | 0x80000000 } else { signedDigitBuf[12*eN + i] = d }
                        d = (s6 >> 16) &+ carry; carry = 0
                        if d > halfBk { d = fullBk &- d; carry = 1; signedDigitBuf[13*eN + i] = d | 0x80000000 } else { signedDigitBuf[13*eN + i] = d }
                        d = (s7 & 0xFFFF) &+ carry; carry = 0
                        if d > halfBk { d = fullBk &- d; carry = 1; signedDigitBuf[14*eN + i] = d | 0x80000000 } else { signedDigitBuf[14*eN + i] = d }
                        d = (s7 >> 16) &+ carry; carry = 0
                        if d > halfBk { d = fullBk &- d; carry = 1; signedDigitBuf[15*eN + i] = d | 0x80000000 } else { signedDigitBuf[15*eN + i] = d }
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
                }
            }
            flatBuf.deallocate()
        }

        // Compute CV2 (coefficient of variation squared) of bucket distribution
        // for a single window. When < 0.5, distribution is uniform enough that
        // CSM reordering provides negligible SIMD coherence benefit.
        let signedDigitBufFinal = signedDigitPtr!
        let scratchStride = nBuckets
        func computeBucketCV2(windowIndex: Int) -> Double {
            let sdBuf = signedDigitBufFinal + windowIndex * effectiveN
            let counts = countsBase + windowIndex * scratchStride
            for i in 0..<nBuckets { counts[i] = 0 }
            for i in 0..<effectiveN {
                counts[Int(sdBuf[i] & 0x7FFFFFFF)] += 1
            }
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

        // Adaptive bucket sort: compute CV2 to decide whether CSM reordering is worth it
        var skipCSM = false
        if effectiveN >= 8192 {
            let cv2 = computeBucketCV2(windowIndex: 0)
            skipCSM = cv2 < 0.5
        }

        // Count-sort per window with adaptive CSM
        DispatchQueue.concurrentPerform(iterations: nWindows) { w in
            let wOff = w * nBuckets
            let idxBase = w * effectiveN
            let counts = countsBase + w * scratchStride
            let positions = positionsBase + w * scratchStride
            let sdBuf = signedDigitBufFinal + w * effectiveN

            // Count buckets
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

            // Scatter into sorted array
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
                // Identity CSM: buckets in natural order (uniform distribution)
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
                    countSortedMap[wOff + dest] = UInt32(w << 16) | UInt32(i)
                }
            }
        }

        // Single command buffer: reduce + bucket_sum + combine
        guard let cb = commandQueue.makeCommandBuffer() else { throw MSMError.noCommandBuffer }

        // Phase 1: Reduce sorted buckets
        // Use warp-per-bucket model when n_buckets <= 1024: each warp (32 threads) processes
        // one bucket cooperatively, dramatically reducing threadgroup scheduling overhead
        // vs the thread-per-bucket sorted-buckets kernel.
        do {
            let enc = cb.makeComputeCommandEncoder()!
            let numBucketsTotal = nWindows * nBuckets
            if nBuckets <= 1024 {
                // Warp-per-bucket: 32 threads per bucket, tree-reduce via shuffle
                enc.setComputePipelineState(reduceWarpPerBucketFunction)
                enc.setBuffer(pointsBuffer, offset: 0, index: 0)
                enc.setBuffer(bucketsBuffer, offset: 0, index: 1)
                enc.setBuffer(allOffsetsBuffer, offset: 0, index: 2)
                enc.setBuffer(allCountsBuffer, offset: 0, index: 3)
                enc.setBytes(&params, length: MemoryLayout<SecpMsmParamsSwift>.stride, index: 4)
                var nw = UInt32(nWindows)
                enc.setBytes(&nw, length: MemoryLayout<UInt32>.stride, index: 5)
                enc.setBuffer(sortedIndicesBuffer, offset: 0, index: 6)
                enc.setBuffer(countSortedMapBuffer, offset: 0, index: 7)
                // Each threadgroup = 1 warp = 32 threads, one per bucket
                enc.dispatchThreadgroups(
                    MTLSize(width: numBucketsTotal, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            } else {
                // Shared memory reduction: 256 threads per bucket for large buckets
                // Tree reduction in shared memory: 256->128->64->32->16->8->4->2->1
                enc.setComputePipelineState(reduceSharedMemFunction)
                enc.setBuffer(pointsBuffer, offset: 0, index: 0)
                enc.setBuffer(bucketsBuffer, offset: 0, index: 1)
                enc.setBuffer(allOffsetsBuffer, offset: 0, index: 2)
                enc.setBuffer(allCountsBuffer, offset: 0, index: 3)
                enc.setBytes(&params, length: MemoryLayout<SecpMsmParamsSwift>.stride, index: 4)
                var nw = UInt32(nWindows)
                enc.setBytes(&nw, length: MemoryLayout<UInt32>.stride, index: 5)
                enc.setBuffer(sortedIndicesBuffer, offset: 0, index: 6)
                enc.setBuffer(countSortedMapBuffer, offset: 0, index: 7)
                // Each threadgroup = 256 threads, one per bucket
                enc.dispatchThreadgroups(
                    MTLSize(width: numBucketsTotal, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
            enc.endEncoding()
        }

        // Phase 2: Bucket sum + combine
        do {
            var nWinsBatch = UInt32(nWindows)
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(bucketSumDirectFunction)
            enc.setBuffer(bucketsBuffer, offset: 0, index: 0)
            enc.setBuffer(segmentResultsBuffer, offset: 0, index: 1)
            enc.setBytes(&params, length: MemoryLayout<SecpMsmParamsSwift>.stride, index: 2)
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

        if let error = cb.error { throw MSMError.gpuError(error.localizedDescription) }

        let winResultsPtr = windowResultsBuffer.contents().bindMemory(to: SecpPointProjective.self, capacity: nWindows)
        var windowResults = [SecpPointProjective](repeating: secpPointIdentity(), count: nWindows)
        for w in 0..<nWindows {
            windowResults[w] = winResultsPtr[w]
        }

        // Horner's method on CPU
        var result = windowResults.last!
        for w in stride(from: nWindows - 2, through: 0, by: -1) {
            for _ in 0..<windowBits {
                result = secpPointDouble(result)
            }
            result = secpPointAdd(result, windowResults[w])
        }
        return result
    }

    // MARK: - Batch MSM (multiple small MSMs in parallel on GPU)

    /// Batch MSM: compute B small MSMs in parallel on GPU.
    /// Each MSM has M points. Total points = B * M.
    /// This is efficient for small M (≤64) where shared memory fits in 32KB.
    ///
    /// - Parameters:
    ///   - allPoints: Flat array of B×M affine points
    ///   - allScalars: Flat array of B×M scalars, each scalar is 8 UInt32 limbs
    ///   - M: Points per MSM (≤64)
    ///   - B: Number of parallel MSMs
    ///   - wb: Window bits (≤7, typically 4-7)
    /// - Returns: Array of B projective points (one per MSM result)
    public func batchMSM(allPoints: [SecpPointAffine], allScalars: [[UInt32]], M: Int, B: Int, wb: Int = 7) throws -> [SecpPointProjective] {
        let totalPoints = M * B
        guard totalPoints > 0, totalPoints == allPoints.count else {
            throw MSMError.invalidInput
        }
        guard M <= 64, wb <= 7 else {
            throw MSMError.invalidInput
        }

        // Allocate buffers
        let pointsSize = MemoryLayout<SecpPointAffine>.stride * totalPoints
        let scalarsSize = MemoryLayout<UInt32>.stride * totalPoints * 8
        let resultsSize = MemoryLayout<SecpPointProjective>.stride * B

        guard let pointsBuf = device.makeBuffer(length: pointsSize, options: .storageModeShared),
              let scalarsBuf = device.makeBuffer(length: scalarsSize, options: .storageModeShared),
              let resultsBuf = device.makeBuffer(length: resultsSize, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate Metal buffers")
        }

        // Copy points to GPU
        allPoints.withUnsafeBufferPointer { src in
            memcpy(pointsBuf.contents(), src.baseAddress!, pointsSize)
        }

        // Copy scalars to GPU (flattened: B×M×8 UInt32)
        scalarsBuf.contents().bindMemory(to: UInt32.self, capacity: totalPoints * 8)
        for i in 0..<totalPoints {
            allScalars[i].withUnsafeBufferPointer { src in
                memcpy(scalarsBuf.contents().advanced(by: i * 8 * MemoryLayout<UInt32>.stride), src.baseAddress!, 8 * MemoryLayout<UInt32>.stride)
            }
        }

        // Dispatch kernel
        guard let cb = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(batchMSMBatchFunction)
        enc.setBuffer(pointsBuf, offset: 0, index: 0)
        enc.setBuffer(scalarsBuf, offset: 0, index: 1)
        enc.setBuffer(resultsBuf, offset: 0, index: 2)

        var mVal = UInt32(M)
        var bVal = UInt32(B)
        var wbVal = UInt32(wb)
        enc.setBytes(&mVal, length: MemoryLayout<UInt32>.stride, index: 3)
        enc.setBytes(&bVal, length: MemoryLayout<UInt32>.stride, index: 4)
        enc.setBytes(&wbVal, length: MemoryLayout<UInt32>.stride, index: 5)

        // One threadgroup per MSM (B threadgroups)
        enc.dispatchThreadgroups(MTLSize(width: B, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        if let error = cb.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        // Copy results back
        let resultsPtr = resultsBuf.contents().bindMemory(to: SecpPointProjective.self, capacity: B)
        var results = [SecpPointProjective]()
        results.reserveCapacity(B)
        for i in 0..<B {
            results.append(resultsPtr[i])
        }
        return results
    }

    // MARK: - NAF Batch MSM

    /// NAF Batch MSM: compute B small MSMs using NAF representation.
    ///
    /// NAF (Non-Adjacent Form) produces digits -1, 0, +1 with at most n/3 non-zero
    /// digits (vs n/2 for binary). This results in ~33% fewer point additions.
    ///
    /// NAF digits are precomputed on CPU since NAF extraction is sequential.
    ///
    /// - Parameters:
    ///   - allPoints: Flat array of B×M affine points
    ///   - allScalars: Flat array of B×M scalars, each scalar is 8 UInt32 limbs
    ///   - M: Points per MSM (≤64)
    ///   - B: Number of parallel MSMs
    /// - Returns: Array of B projective points (one per MSM result)
    public func batchNAFMSM(allPoints: [SecpPointAffine], allScalars: [[UInt32]], M: Int, B: Int) throws -> [SecpPointProjective] {
        let totalPoints = M * B
        guard totalPoints > 0, totalPoints == allPoints.count else {
            throw MSMError.invalidInput
        }
        guard M <= 64 else {
            throw MSMError.invalidInput
        }

        // Allocate buffers
        let pointsSize = MemoryLayout<SecpPointAffine>.stride * totalPoints
        let nafDigitsSize = totalPoints * 256  // 256 NAF digits per scalar
        let resultsSize = MemoryLayout<SecpPointProjective>.stride * B

        guard let pointsBuf = device.makeBuffer(length: pointsSize, options: .storageModeShared),
              let nafDigitsBuf = device.makeBuffer(length: nafDigitsSize, options: .storageModeShared),
              let resultsBuf = device.makeBuffer(length: resultsSize, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate Metal buffers")
        }

        // Copy points to GPU
        allPoints.withUnsafeBufferPointer { src in
            memcpy(pointsBuf.contents(), src.baseAddress!, pointsSize)
        }

        // Precompute NAF digits on CPU (parallel)
        let nafDigitsPtr = nafDigitsBuf.contents().bindMemory(to: UInt8.self, capacity: nafDigitsSize)

        // Process scalars in parallel
        DispatchQueue.concurrentPerform(iterations: totalPoints) { idx in
            let scalar = Secp256k1MSM.reduceModN(allScalars[idx])
            let naf = Self.nafDecompose(scalar)
            let baseOffset = idx * 256

            // Store NAF digits (0 = zero, 1 = +1, 2 = -1)
            for bit in 0..<256 {
                let digit: UInt8
                if bit < naf.count {
                    switch naf[bit] {
                    case 1: digit = 1   // +1
                    case -1: digit = 2  // -1
                    default: digit = 0  // 0
                    }
                } else {
                    digit = 0  // No more digits = 0
                }
                nafDigitsPtr[baseOffset + bit] = digit
            }
        }

        // Dispatch NAF kernel
        guard let cb = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(batchNAFBatchFunction)
        enc.setBuffer(pointsBuf, offset: 0, index: 0)
        enc.setBuffer(nafDigitsBuf, offset: 0, index: 1)
        enc.setBuffer(resultsBuf, offset: 0, index: 2)

        var mVal = UInt32(M)
        var bVal = UInt32(B)
        enc.setBytes(&mVal, length: MemoryLayout<UInt32>.stride, index: 3)
        enc.setBytes(&bVal, length: MemoryLayout<UInt32>.stride, index: 4)

        // One threadgroup per MSM (B threadgroups)
        enc.dispatchThreadgroups(MTLSize(width: B, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        if let error = cb.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        // Copy results back
        let resultsPtr = resultsBuf.contents().bindMemory(to: SecpPointProjective.self, capacity: B)
        var results = [SecpPointProjective]()
        results.reserveCapacity(B)
        for i in 0..<B {
            results.append(resultsPtr[i])
        }
        return results
    }

    /// NAF decomposition for a scalar (8 limbs, little-endian).
    /// Returns NAF digits (0, +1, -1) from LSB to MSB.
    private static func nafDecompose(_ scalar: [UInt32]) -> [Int8] {
        // Convert to big integer representation for NAF algorithm
        var s = scalar
        var result = [Int8]()
        result.reserveCapacity(256)

        // NAF algorithm: process until scalar is zero
        while !Self.isZero(s) {
            let isOdd = (s[0] & 1) != 0
            var k: Int8 = 0

            if isOdd {
                // k = 2 - (s mod 4)
                // s mod 4 = s[0] & 3
                let sMod4 = Int(s[0] & 3)
                if sMod4 == 1 {
                    k = 1   // +1
                } else {
                    k = -1  // -1 (sMod4 == 3)
                }
                // s = s - k
                s = Self.subScalar(s, k)
            }
            // else k = 0

            result.append(k)

            // s = s / 2 (shift right by 1)
            s = Self.shiftRight1(s)
        }

        return result
    }

    /// Subtract k (where k is -1 or +1) from scalar.
    /// s = s - k = s + (-k) for k=1, or s + (2^32-1) + 1 for k=-1
    private static func subScalar(_ s: [UInt32], _ k: Int8) -> [UInt32] {
        var result = [UInt32](repeating: 0, count: 8)
        var borrow: Int64 = 0

        for i in 0..<8 {
            if k == 1 {
                borrow += Int64(s[i]) - 1
            } else {
                // k == -1: add UInt32.max (which is -1 in two's complement)
                borrow += Int64(s[i]) + Int64(UInt32.max)
            }
            result[i] = UInt32(truncatingIfNeeded: borrow & 0xFFFFFFFF)
            borrow >>= 32
        }
        return result
    }

    /// Shift scalar right by 1 bit (s = s / 2)
    private static func shiftRight1(_ s: [UInt32]) -> [UInt32] {
        var result = [UInt32](repeating: 0, count: 8)
        var carry: UInt32 = 0

        for i in stride(from: 7, through: 0, by: -1) {
            result[i] = (s[i] >> 1) | carry
            carry = (s[i] & 1) << 31
        }
        return result
    }

    /// Check if scalar is zero
    private static func isZero(_ s: [UInt32]) -> Bool {
        for i in 0..<8 {
            if s[i] != 0 { return false }
        }
        return true
    }

    /// NAF MSM reference implementation for correctness verification.
    /// Computes MSM by explicit NAF decomposition (CPU, not GPU).
    public static func nafMSM(points: [SecpPointAffine], scalars: [[UInt32]]) -> SecpPointProjective {
        let n = points.count
        guard n == scalars.count, n > 0 else {
            return secpPointIdentity()
        }

        var result = secpPointIdentity()

        for i in 0..<n {
            let naf = nafDecompose(reduceModN(scalars[i]))
            var point = secpPointFromAffine(points[i])

            // Process NAF digits from MSB to LSB (right to left in the array)
            // NAF representation: result = sum(d_i * 2^i * P_i) where d_i in {-1, 0, +1}
            for j in stride(from: naf.count - 1, through: 0, by: -1) {
                result = secpPointDouble(result)
                let digit = naf[j]
                if digit == 1 {
                    result = secpPointAdd(result, point)
                } else if digit == -1 {
                    let negPoint = SecpPointAffine(x: point.x, y: secpNeg(point.y))
                    result = secpPointAdd(result, secpPointFromAffine(negPoint))
                }
                // digit == 0: no addition
            }
        }

        return result
    }
}

// SecpMsmParamsSwift must match Metal SecpMsmParams struct layout
public struct SecpMsmParamsSwift {
    public var n_points: UInt32
    public var window_bits: UInt32
    public var n_buckets: UInt32

    public init(n_points: UInt32, window_bits: UInt32, n_buckets: UInt32) {
        self.n_points = n_points
        self.window_bits = window_bits
        self.n_buckets = n_buckets
    }
}
