// BN254 G2 Metal MSM Engine — Pippenger's bucket method with GPU acceleration
// G2 points have Fp2 coordinates (4x wider than G1), scalars are BN254 Fr.
// Used for Dory opening proofs and Groth16 batch verification.

import Foundation
import Metal

/// GPU point types matching the Metal shader layout.
/// G2PointAffineGPU: 4 Fp elements = 128 bytes
/// G2PointProjectiveGPU: 6 Fp elements = 192 bytes
public struct G2PointAffineGPU {
    public var x_c0: Fp
    public var x_c1: Fp
    public var y_c0: Fp
    public var y_c1: Fp

    public init(from p: G2AffinePoint) {
        self.x_c0 = p.x.c0
        self.x_c1 = p.x.c1
        self.y_c0 = p.y.c0
        self.y_c1 = p.y.c1
    }

    public func toG2Affine() -> G2AffinePoint {
        G2AffinePoint(x: Fp2(c0: x_c0, c1: x_c1), y: Fp2(c0: y_c0, c1: y_c1))
    }
}

public struct G2PointProjectiveGPU {
    public var x_c0: Fp
    public var x_c1: Fp
    public var y_c0: Fp
    public var y_c1: Fp
    public var z_c0: Fp
    public var z_c1: Fp

    public func toG2Projective() -> G2ProjectivePoint {
        G2ProjectivePoint(
            x: Fp2(c0: x_c0, c1: x_c1),
            y: Fp2(c0: y_c0, c1: y_c1),
            z: Fp2(c0: z_c0, c1: z_c1))
    }
}

struct G2MsmParams {
    var n_points: UInt32
    var window_bits: UInt32
    var n_buckets: UInt32
}

public class BN254G2MSM {
    public static let version = Versions.msmBN254G2
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    private let reduceSortedFunction: MTLComputePipelineState
    private let bucketSumDirectFunction: MTLComputePipelineState
    private let combineSegmentsFunction: MTLComputePipelineState
    private let hornerCombineFunction: MTLComputePipelineState
    private let signedDigitFunction: MTLComputePipelineState
    private let gpuSortHistogramFunction: MTLComputePipelineState
    private let gpuSortScatterFunction: MTLComputePipelineState
    private let gpuBuildCsmFunction: MTLComputePipelineState

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
    private var cpuScratchStride = 0
    private var signedDigitPtr: UnsafeMutablePointer<UInt32>?
    private var signedDigitCapacity = 0
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
        let cacheFile = BN254G2MSM.cacheDir.appendingPathComponent("bn254_g2_msm.metallib")

        let requiredKernels = [
            "g2_msm_reduce_sorted_buckets", "g2_msm_bucket_sum_direct",
            "g2_msm_combine_segments", "g2_signed_digit_extract"
        ]
        if FileManager.default.fileExists(atPath: cacheFile.path) {
            do {
                let cached = try device.makeLibrary(URL: cacheFile)
                if requiredKernels.allSatisfy({ cached.makeFunction(name: $0) != nil }) {
                    library = cached
                } else {
                    library = try BN254G2MSM.compileAndCache(device: device, cacheFile: cacheFile)
                }
            } catch {
                library = try BN254G2MSM.compileAndCache(device: device, cacheFile: cacheFile)
            }
        } else {
            library = try BN254G2MSM.compileAndCache(device: device, cacheFile: cacheFile)
        }

        guard let reduceSortedFn = library.makeFunction(name: "g2_msm_reduce_sorted_buckets"),
              let sumDirectFn = library.makeFunction(name: "g2_msm_bucket_sum_direct"),
              let combineFn = library.makeFunction(name: "g2_msm_combine_segments"),
              let hornerFn = library.makeFunction(name: "g2_msm_horner_combine"),
              let signedDigitFn = library.makeFunction(name: "g2_signed_digit_extract"),
              let gpuSortHistFn = library.makeFunction(name: "g2_sort_histogram"),
              let gpuSortScatFn = library.makeFunction(name: "g2_sort_scatter"),
              let gpuBuildCsmFn = library.makeFunction(name: "g2_build_csm") else {
            throw MSMError.missingKernel
        }

        self.reduceSortedFunction = try device.makeComputePipelineState(function: reduceSortedFn)
        self.bucketSumDirectFunction = try device.makeComputePipelineState(function: sumDirectFn)
        self.combineSegmentsFunction = try device.makeComputePipelineState(function: combineFn)
        self.hornerCombineFunction = try device.makeComputePipelineState(function: hornerFn)
        self.signedDigitFunction = try device.makeComputePipelineState(function: signedDigitFn)
        self.gpuSortHistogramFunction = try device.makeComputePipelineState(function: gpuSortHistFn)
        self.gpuSortScatterFunction = try device.makeComputePipelineState(function: gpuSortScatFn)
        self.gpuBuildCsmFunction = try device.makeComputePipelineState(function: gpuBuildCsmFn)
        self.tuning = TuningManager.shared.config(device: device)
    }

    private static func compileAndCache(device: MTLDevice, cacheFile: URL) throws -> MTLLibrary {
        let shaderDir = findShaderDir()

        let fpSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fp.metal", encoding: .utf8)
        let msmSource = try String(contentsOfFile: shaderDir + "/msm/bn254_g2_msm_kernels.metal", encoding: .utf8)

        func stripIncludes(_ s: String) -> String {
            s.split(separator: "\n").filter { !$0.contains("#include") }.joined(separator: "\n")
        }
        func stripGuards(_ s: String) -> String {
            s.replacingOccurrences(of: "#ifndef BN254_FP_METAL", with: "")
             .replacingOccurrences(of: "#define BN254_FP_METAL", with: "")
             .replacingOccurrences(of: "#endif // BN254_FP_METAL", with: "")
        }

        let combined = stripGuards(fpSource) + "\n" + stripGuards(stripIncludes(msmSource))

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        let library = try device.makeLibrary(source: combined, options: options)

        try? FileManager.default.createDirectory(
            at: BN254G2MSM.cacheDir, withIntermediateDirectories: true)

        if #available(macOS 11.0, *) {
            let archiveDesc = MTLBinaryArchiveDescriptor()
            if let archive = try? device.makeBinaryArchive(descriptor: archiveDesc) {
                for name in ["g2_msm_reduce_sorted_buckets", "g2_msm_bucket_sum_direct"] {
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
                length: MemoryLayout<G2PointAffineGPU>.stride * np, options: .storageModeShared)
            sortedIndicesBuffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride * np * nw, options: .storageModeShared)
            allOffsetsBuffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride * nb * nw, options: .storageModeShared)
            allCountsBuffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride * nb * nw, options: .storageModeShared)
            bucketsBuffer = device.makeBuffer(
                length: MemoryLayout<G2PointProjectiveGPU>.stride * nb * nw, options: .storageModeShared)
            segmentResultsBuffer = device.makeBuffer(
                length: MemoryLayout<G2PointProjectiveGPU>.stride * ns * nw, options: .storageModeShared)
            windowResultsBuffer = device.makeBuffer(
                length: MemoryLayout<G2PointProjectiveGPU>.stride * nw, options: .storageModeShared)
            finalResultBuffer = device.makeBuffer(
                length: MemoryLayout<G2PointProjectiveGPU>.stride, options: .storageModeShared)
            countSortedMapBuffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride * nb * nw, options: .storageModeShared)
            signedDigitBuffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride * np * nw, options: .storageModeShared)
            maxAllocatedPoints = np
            maxAllocatedBuckets = nb
            maxAllocatedWindows = nw
            maxAllocatedSegments = ns
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

    /// Perform G2 MSM: result = sum(scalars[i] * points[i])
    /// Points are G2 affine (Fp2 coordinates), scalars are BN254 Fr as 8x UInt32 limbs.
    public func msm(points: [G2AffinePoint], scalars: [[UInt32]]) throws -> G2ProjectivePoint {
        let n = points.count
        guard n == scalars.count, n > 0 else {
            throw MSMError.invalidInput
        }

        // For very small inputs, use CPU
        if n <= 64 {
            return cpuMSM(points: points, scalars: scalars)
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
        let scalarBits = 256
        let effectiveN = n

        // Window bits: smaller than G1 because each bucket is 4x larger
        var windowBits: UInt32
        if effectiveN <= 256 {
            windowBits = 8
        } else if effectiveN <= 4096 {
            windowBits = 10
        } else if effectiveN <= 16384 {
            windowBits = 11
        } else {
            windowBits = 13
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

        // Copy G2 points to GPU buffer
        let gpuPtsPtr = pointsBuffer.contents().bindMemory(to: G2PointAffineGPU.self, capacity: effectiveN)
        for i in 0..<n {
            gpuPtsPtr[i] = G2PointAffineGPU(from: points[i])
        }

        let allOffsets = allOffsetsBuffer.contents().bindMemory(to: UInt32.self, capacity: nBuckets * nWindows)
        let allCounts = allCountsBuffer.contents().bindMemory(to: UInt32.self, capacity: nBuckets * nWindows)
        let sortedIdxPtr = sortedIndicesBuffer.contents().bindMemory(to: UInt32.self, capacity: effectiveN * nWindows)
        let countSortedMap = countSortedMapBuffer.contents().bindMemory(to: UInt32.self, capacity: nBuckets * nWindows)

        var params = G2MsmParams(
            n_points: UInt32(effectiveN),
            window_bits: windowBits,
            n_buckets: UInt32(nBuckets)
        )
        var nSegs = UInt32(nSegments)

        let countsBase = cpuCountsPtr!
        let positionsBase = cpuPositionsPtr!

        // Signed-digit extraction (CPU)
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
                            signedDigitBuf[w * eN + i] = digit | 0x80000000
                        } else {
                            signedDigitBuf[w * eN + i] = digit
                        }
                    }
                }
            }
        }

        // Count-sort per window (CPU — avoids GPU sync overhead on M-series)
        let scratchStride = self.cpuScratchStride
        DispatchQueue.concurrentPerform(iterations: nWindows) { w in
            let wOff = w * nBuckets
            let idxBase = w * effectiveN
            let counts = countsBase + w * scratchStride
            let positions = positionsBase + w * scratchStride
            let sdBuf = signedDigitBuf + w * effectiveN

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
            enc.setBytes(&params, length: MemoryLayout<G2MsmParams>.stride, index: 4)
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
            enc.setBytes(&params, length: MemoryLayout<G2MsmParams>.stride, index: 2)
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

        let winResultsPtr = windowResultsBuffer.contents().bindMemory(to: G2PointProjectiveGPU.self, capacity: nWindows)

        // Horner's method on CPU to combine window results
        var result = winResultsPtr[nWindows - 1].toG2Projective()
        for w in stride(from: nWindows - 2, through: 0, by: -1) {
            for _ in 0..<windowBits {
                result = g2Double(result)
            }
            let wr = winResultsPtr[w].toG2Projective()
            if !g2IsIdentity(wr) {
                result = g2IsIdentity(result) ? wr : g2Add(result, wr)
            }
        }

        return result
    }

    /// CPU fallback MSM for small inputs (Straus/Shamir's trick)
    private func cpuMSM(points: [G2AffinePoint], scalars: [[UInt32]]) -> G2ProjectivePoint {
        let n = points.count
        if n == 0 { return g2Identity() }

        // Convert UInt32 scalars to UInt64 for g2ScalarMul
        var result = g2Identity()
        for i in 0..<n {
            let sc = scalars[i]
            let reduced = Self.reduceModR(sc)
            var u64 = [UInt64](repeating: 0, count: 4)
            for j in 0..<4 {
                u64[j] = UInt64(reduced[j * 2]) | (UInt64(reduced[j * 2 + 1]) << 32)
            }
            let p = g2ScalarMul(g2FromAffine(points[i]), u64)
            result = g2IsIdentity(result) ? p : g2Add(result, p)
        }
        return result
    }
}
