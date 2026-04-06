// BLS12-381 Metal MSM Engine -- Pippenger's bucket method with GPU acceleration
// Uses BLS12-381 G1 curve (y^2 = x^3 + 4) over 381-bit Fp.
// No GLV endomorphism -- full 255-bit scalar windows.

import Foundation
import Metal

/// GPU-accelerated multi-scalar multiplication on BLS12-381 G1.
/// Dispatches Pippenger bucket reduce, bucket sum, and Horner combine on Metal.
public class BLS12381MSM {
    public static let version = Versions.msmBLS12381
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
        let cacheFile = BLS12381MSM.cacheDir.appendingPathComponent("bls12381_msm.metallib")

        let requiredKernels = [
            "msm381_reduce_sorted_buckets", "msm381_bucket_sum_direct",
            "msm381_combine_segments", "msm381_signed_digit_extract"
        ]
        if FileManager.default.fileExists(atPath: cacheFile.path) {
            do {
                let cached = try device.makeLibrary(URL: cacheFile)
                if requiredKernels.allSatisfy({ cached.makeFunction(name: $0) != nil }) {
                    library = cached
                } else {
                    library = try BLS12381MSM.compileAndCache(device: device, cacheFile: cacheFile)
                }
            } catch {
                library = try BLS12381MSM.compileAndCache(device: device, cacheFile: cacheFile)
            }
        } else {
            library = try BLS12381MSM.compileAndCache(device: device, cacheFile: cacheFile)
        }

        guard let reduceSortedFn = library.makeFunction(name: "msm381_reduce_sorted_buckets"),
              let reduceCoopFn = library.makeFunction(name: "msm381_reduce_cooperative"),
              let sumDirectFn = library.makeFunction(name: "msm381_bucket_sum_direct"),
              let combineFn = library.makeFunction(name: "msm381_combine_segments"),
              let hornerFn = library.makeFunction(name: "msm381_horner_combine"),
              let signedDigitFn = library.makeFunction(name: "msm381_signed_digit_extract"),
              let gpuSortHistFn = library.makeFunction(name: "msm381_sort_histogram"),
              let gpuSortScatFn = library.makeFunction(name: "msm381_sort_scatter"),
              let gpuBuildCsmFn = library.makeFunction(name: "msm381_build_csm") else {
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
        self.tuning = TuningManager.shared.config(device: device)
    }

    private static func compileAndCache(device: MTLDevice, cacheFile: URL) throws -> MTLLibrary {
        let shaderDir = findShaderDir()

        let fqSource = try String(contentsOfFile: shaderDir + "/fields/bls12381_fq.metal", encoding: .utf8)
        let curveSource = try String(contentsOfFile: shaderDir + "/geometry/bls12381_curve.metal", encoding: .utf8)
        let msmSource = try String(contentsOfFile: shaderDir + "/msm/bls12381_msm_kernels.metal", encoding: .utf8)

        func stripIncludes(_ s: String) -> String {
            s.split(separator: "\n").filter { !$0.contains("#include") }.joined(separator: "\n")
        }
        func stripGuards(_ s: String) -> String {
            s.replacingOccurrences(of: "#ifndef BLS12381_FQ_METAL", with: "")
             .replacingOccurrences(of: "#define BLS12381_FQ_METAL", with: "")
             .replacingOccurrences(of: "#endif // BLS12381_FQ_METAL", with: "")
             .replacingOccurrences(of: "#ifndef BLS12381_CURVE_METAL", with: "")
             .replacingOccurrences(of: "#define BLS12381_CURVE_METAL", with: "")
             .replacingOccurrences(of: "#endif // BLS12381_CURVE_METAL", with: "")
        }

        let combined = stripGuards(fqSource) + "\n" +
                        stripGuards(stripIncludes(curveSource)) + "\n" +
                        stripIncludes(msmSource)

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        let library = try device.makeLibrary(source: combined, options: options)

        try? FileManager.default.createDirectory(
            at: BLS12381MSM.cacheDir, withIntermediateDirectories: true)

        if #available(macOS 11.0, *) {
            let archiveDesc = MTLBinaryArchiveDescriptor()
            if let archive = try? device.makeBinaryArchive(descriptor: archiveDesc) {
                for name in ["msm381_reduce_sorted_buckets", "msm381_bucket_sum_direct"] {
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
                let path = url.appendingPathComponent("fields/bls12381_fq.metal").path
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
            if FileManager.default.fileExists(atPath: "\(path)/fields/bls12381_fq.metal") {
                return path
            }
        }
        return "./Sources/Shaders"
    }

    // Fr381 scalar field order r as 8x32-bit limbs (little-endian)
    // r = 52435875175126190479447740508185965837690552500527637822603658699938581184513
    private static let R_LIMBS: [UInt32] = [
        0x00000001, 0xffffffff, 0xfffe5bfe, 0x53bda402,
        0x09a1d805, 0x3339d808, 0x299d7d48, 0x73eda753
    ]

    /// Reduce a 256-bit scalar mod r (Fr381 scalar field order).
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
                length: MemoryLayout<G1Affine381>.stride * np, options: .storageModeShared)
            sortedIndicesBuffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride * np * nw, options: .storageModeShared)
            allOffsetsBuffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride * nb * nw, options: .storageModeShared)
            allCountsBuffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride * nb * nw, options: .storageModeShared)
            bucketsBuffer = device.makeBuffer(
                length: MemoryLayout<G1Projective381>.stride * nb * nw, options: .storageModeShared)
            segmentResultsBuffer = device.makeBuffer(
                length: MemoryLayout<G1Projective381>.stride * ns * nw, options: .storageModeShared)
            windowResultsBuffer = device.makeBuffer(
                length: MemoryLayout<G1Projective381>.stride * nw, options: .storageModeShared)
            finalResultBuffer = device.makeBuffer(
                length: MemoryLayout<G1Projective381>.stride, options: .storageModeShared)
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
        signedDigitPtr?.deallocate()
    }

    /// GPU-accelerated MSM on BLS12-381 G1.
    /// points: affine G1 points, scalars: 8x32-bit limbs per scalar (standard form, NOT Montgomery).
    /// Returns the MSM result in projective coordinates.
    public func msm(points: [G1Affine381], scalars: [[UInt32]]) throws -> G1Projective381 {
        let n = points.count
        guard n == scalars.count, n > 0 else {
            throw MSMError.invalidInput
        }

        // For small n, CPU Pippenger is faster
        if n <= 2048 {
            let msmScalars = scalars.map { Self.reduceModR($0) }
            return cpuMSM381(points: points, scalars: msmScalars)
        }

        let msmScalars = scalars.map { Self.reduceModR($0) }
        let scalarBits = 255

        // Window sizing tuned for 12-limb Fp381 (same register pressure as BLS12-377).
        // Avoid w=14 which triggers M3 Pro GPU pathology.
        var windowBits: UInt32
        if n <= 256 {
            windowBits = 8
        } else if n <= 4096 {
            windowBits = 10
        } else if n <= 16384 {
            windowBits = 11
        } else if n <= 65536 {
            windowBits = 13
        } else {
            windowBits = 15
        }
        if let wbOverride = windowBitsOverride {
            windowBits = wbOverride
        }
        let nWindows = (scalarBits + Int(windowBits) - 1) / Int(windowBits)
        let fullBuckets = 1 << Int(windowBits)
        let halfBuckets = fullBuckets >> 1
        let nBuckets = halfBuckets + 1
        let nSegments = min(256, max(1, nBuckets / 2))

        ensureBuffers(n: n, nBuckets: nBuckets, nSegments: nSegments, nWindows: nWindows)
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

        // Copy points to GPU buffer
        let gpuPtsPtr = pointsBuffer.contents().bindMemory(to: G1Affine381.self, capacity: n)
        points.withUnsafeBufferPointer { src in
            gpuPtsPtr.update(from: src.baseAddress!, count: n)
        }

        let allOffsets = allOffsetsBuffer.contents().bindMemory(to: UInt32.self, capacity: nBuckets * nWindows)
        let allCounts = allCountsBuffer.contents().bindMemory(to: UInt32.self, capacity: nBuckets * nWindows)
        let sortedIdxPtr = sortedIndicesBuffer.contents().bindMemory(to: UInt32.self, capacity: n * nWindows)
        let countSortedMap = countSortedMapBuffer.contents().bindMemory(to: UInt32.self, capacity: nBuckets * nWindows)

        // CPU signed-digit extraction
        let sdNeeded = n * nWindows
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
        let eN = n

        let chunkSize = 4096
        let nChunks = (n + chunkSize - 1) / chunkSize
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

        // Count-sort per window
        let countsBase = cpuCountsPtr!
        let positionsBase = cpuPositionsPtr!
        DispatchQueue.concurrentPerform(iterations: nWindows) { w in
            let wOff = w * nBuckets
            let idxBase = w * n
            let counts = countsBase + w * nBuckets
            let positions = positionsBase + w * nBuckets
            let sdBuf = signedDigitBuf + w * n

            for i in 0..<nBuckets { counts[i] = 0 }
            for i in 0..<n {
                counts[Int(sdBuf[i] & 0x7FFFFFFF)] += 1
            }

            var runningOffset = 0
            for i in 0..<nBuckets {
                allOffsets[wOff + i] = UInt32(runningOffset)
                allCounts[wOff + i] = UInt32(counts[i])
                positions[i] = runningOffset
                runningOffset += counts[i]
            }

            for i in 0..<n {
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

        // GPU dispatch: reduce + bucket_sum + combine
        var params = Msm381Params(
            n_points: UInt32(n),
            window_bits: windowBits,
            n_buckets: UInt32(nBuckets)
        )
        var nSegs = UInt32(nSegments)

        guard let cb = commandQueue.makeCommandBuffer() else { throw MSMError.noCommandBuffer }

        // Phase 1: Reduce sorted buckets
        do {
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(reduceSortedFunction)
            enc.setBuffer(pointsBuffer, offset: 0, index: 0)
            enc.setBuffer(bucketsBuffer, offset: 0, index: 1)
            enc.setBuffer(allOffsetsBuffer, offset: 0, index: 2)
            enc.setBuffer(allCountsBuffer, offset: 0, index: 3)
            enc.setBytes(&params, length: MemoryLayout<Msm381Params>.stride, index: 4)
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
            enc.setBytes(&params, length: MemoryLayout<Msm381Params>.stride, index: 2)
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

        let winResultsPtr = windowResultsBuffer.contents().bindMemory(to: G1Projective381.self, capacity: nWindows)
        var windowResults = [G1Projective381](repeating: g1_381Identity(), count: nWindows)
        for w in 0..<nWindows {
            windowResults[w] = winResultsPtr[w]
        }

        // Horner's method on CPU
        var result = windowResults.last!
        for w in stride(from: nWindows - 2, through: 0, by: -1) {
            for _ in 0..<windowBits {
                result = g1_381Double(result)
            }
            result = g1_381Add(result, windowResults[w])
        }

        return result
    }

    /// Convenience MSM accepting Fr381 scalars (Montgomery form).
    /// Converts from Montgomery to standard form internally.
    public func msmFr(points: [G1Affine381], scalars: [Fr381]) throws -> G1Projective381 {
        let scalarArrays = scalars.map { s -> [UInt32] in
            let std = fr381ToInt(s)
            return [
                UInt32(std[0] & 0xFFFFFFFF), UInt32(std[0] >> 32),
                UInt32(std[1] & 0xFFFFFFFF), UInt32(std[1] >> 32),
                UInt32(std[2] & 0xFFFFFFFF), UInt32(std[2] >> 32),
                UInt32(std[3] & 0xFFFFFFFF), UInt32(std[3] >> 32),
            ]
        }
        return try msm(points: points, scalars: scalarArrays)
    }

    /// CPU fallback MSM using naive double-and-add (for small inputs).
    private func cpuMSM381(points: [G1Affine381], scalars: [[UInt32]]) -> G1Projective381 {
        // Use the existing CPU Pippenger
        var flatScalars = [UInt32]()
        flatScalars.reserveCapacity(points.count * 8)
        for s in scalars {
            flatScalars.append(contentsOf: s)
        }
        return g1_381PippengerMSMFlat(points: points, flatScalars: flatScalars)
    }
}

/// Msm381Params must match Metal struct layout
public struct Msm381Params {
    public var n_points: UInt32
    public var window_bits: UInt32
    public var n_buckets: UInt32

    public init(n_points: UInt32, window_bits: UInt32, n_buckets: UInt32) {
        self.n_points = n_points
        self.window_bits = window_bits
        self.n_buckets = n_buckets
    }
}
