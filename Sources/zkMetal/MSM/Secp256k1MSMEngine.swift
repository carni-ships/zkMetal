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
    // GLV cached buffers
    private var glvScalarInBufCached: MTLBuffer?
    private var glvK1MetalBufCached: MTLBuffer?
    private var glvNeg1BufCached: MTLBuffer?
    private var glvNeg2BufCached: MTLBuffer?
    private var glvCachedN: Int = 0
    public var windowBitsOverride: UInt32?
    public var useGLV = false  // GLV regresses on M3 GPU: 2x points costs more than halved scalars
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
            "secp_msm_reduce_sorted_buckets", "secp_msm_bucket_sum_direct",
            "secp_msm_combine_segments", "secp_msm_signed_digit_extract",
            "secp_glv_decompose", "secp_glv_endomorphism"
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
              let reduceCoopFn = library.makeFunction(name: "secp_msm_reduce_cooperative"),
              let sumDirectFn = library.makeFunction(name: "secp_msm_bucket_sum_direct"),
              let combineFn = library.makeFunction(name: "secp_msm_combine_segments"),
              let hornerFn = library.makeFunction(name: "secp_msm_horner_combine"),
              let signedDigitFn = library.makeFunction(name: "secp_msm_signed_digit_extract"),
              let gpuSortHistFn = library.makeFunction(name: "secp_msm_sort_histogram"),
              let gpuSortScatFn = library.makeFunction(name: "secp_msm_sort_scatter"),
              let gpuBuildCsmFn = library.makeFunction(name: "secp_msm_build_csm"),
              let glvDecomposeFn = library.makeFunction(name: "secp_glv_decompose"),
              let glvEndoFn = library.makeFunction(name: "secp_glv_endomorphism") else {
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

    public func msm(points: [SecpPointAffine], scalars: [[UInt32]]) throws -> SecpPointProjective {
        let n = points.count
        guard n == scalars.count, n > 0 else {
            throw MSMError.invalidInput
        }

        // For small inputs, C Pippenger MSM is faster than GPU
        if n <= 1024 {
            return cSecpPippengerMSM(points: points, scalars: scalars)
        }

        let msmScalars = scalars.map { Self.reduceModN($0) }
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
        if !(useGLV && n >= 256) {
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

        if useGLV && n >= 256 {
            // CPU-side GLV decomposition (verified correct)
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

        // CPU signed-digit extraction (works for both GLV and non-GLV)
        let activeScalars = glvScalars ?? centeredScalars ?? msmScalars
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

            let chunkSize = 4096
            let nChunks = (effectiveN + chunkSize - 1) / chunkSize
            DispatchQueue.concurrentPerform(iterations: nChunks) { chunk in
                let start = chunk * chunkSize
                let end = min(start + chunkSize, eN)
                for i in start..<end {
                    var carry: UInt32 = 0
                    activeScalars[i].withUnsafeBufferPointer { sp in
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
        }

        // Count-sort per window
        let signedDigitBufFinal = signedDigitPtr!
        DispatchQueue.concurrentPerform(iterations: nWindows) { w in
            let wOff = w * nBuckets
            let idxBase = w * effectiveN
            let counts = countsBase + w * nBuckets
            let positions = positionsBase + w * nBuckets
            let sdBuf = signedDigitBufFinal + w * effectiveN

            for i in 0..<nBuckets { counts[i] = 0 }
            for i in 0..<effectiveN {
                counts[Int(sdBuf[i] & 0x7FFFFFFF)] += 1
            }

            var runningOffset = 0
            for i in 0..<nBuckets {
                allOffsets[wOff + i] = UInt32(runningOffset)
                allCounts[wOff + i] = UInt32(counts[i])
                positions[i] = runningOffset
                runningOffset += counts[i]
            }

            for i in 0..<effectiveN {
                let raw = sdBuf[i]
                let digit = Int(raw & 0x7FFFFFFF)
                if digit == 0 { continue }
                var idx = UInt32(i)
                if (raw & 0x80000000) != 0 { idx |= 0x80000000 }
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
            enc.setBytes(&params, length: MemoryLayout<SecpMsmParamsSwift>.stride, index: 4)
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
