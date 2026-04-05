// Ed25519 Metal MSM Engine -- Pippenger's bucket method with GPU acceleration
// Adapted for twisted Edwards curve with extended coordinates.

import Foundation
import Metal

public class Ed25519MSM {
    public static let version = Versions.msmEd25519
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    private let reduceSortedFunction: MTLComputePipelineState
    private let bucketSumDirectFunction: MTLComputePipelineState
    private let combineSegmentsFunction: MTLComputePipelineState
    private let signedDigitFunction: MTLComputePipelineState

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
    private var countSortedMapBuffer: MTLBuffer?
    private var dBuffer: MTLBuffer?
    private var cpuCountsPtr: UnsafeMutablePointer<Int>?
    private var cpuPositionsPtr: UnsafeMutablePointer<Int>?
    private var cpuScratchCapacity = 0
    private var signedDigitPtr: UnsafeMutablePointer<UInt32>?
    private var signedDigitCapacity = 0

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
        let cacheFile = Ed25519MSM.cacheDir.appendingPathComponent("ed25519_msm.metallib")

        let requiredKernels = [
            "ed_msm_reduce_sorted_buckets", "ed_msm_bucket_sum_direct",
            "ed_msm_combine_segments", "ed_msm_signed_digit_extract"
        ]
        if FileManager.default.fileExists(atPath: cacheFile.path) {
            do {
                let cached = try device.makeLibrary(URL: cacheFile)
                if requiredKernels.allSatisfy({ cached.makeFunction(name: $0) != nil }) {
                    library = cached
                } else {
                    library = try Ed25519MSM.compileAndCache(device: device, cacheFile: cacheFile)
                }
            } catch {
                library = try Ed25519MSM.compileAndCache(device: device, cacheFile: cacheFile)
            }
        } else {
            library = try Ed25519MSM.compileAndCache(device: device, cacheFile: cacheFile)
        }

        guard let reduceSortedFn = library.makeFunction(name: "ed_msm_reduce_sorted_buckets"),
              let sumDirectFn = library.makeFunction(name: "ed_msm_bucket_sum_direct"),
              let combineFn = library.makeFunction(name: "ed_msm_combine_segments"),
              let signedDigitFn = library.makeFunction(name: "ed_msm_signed_digit_extract") else {
            throw MSMError.missingKernel
        }

        self.reduceSortedFunction = try device.makeComputePipelineState(function: reduceSortedFn)
        self.bucketSumDirectFunction = try device.makeComputePipelineState(function: sumDirectFn)
        self.combineSegmentsFunction = try device.makeComputePipelineState(function: combineFn)
        self.signedDigitFunction = try device.makeComputePipelineState(function: signedDigitFn)

        // Pre-compute d in Montgomery form and store as GPU buffer
        // CPU now uses direct integer form; GPU needs Montgomery form
        let dConst = ed25519D()
        let dMont = dConst.toMontgomery()
        let dLimbs = dMont.to32()
        self.dBuffer = device.makeBuffer(bytes: dLimbs, length: 32, options: .storageModeShared)
    }

    private static func compileAndCache(device: MTLDevice, cacheFile: URL) throws -> MTLLibrary {
        let shaderDir = findShaderDir()

        let fpSource = try String(contentsOfFile: shaderDir + "/fields/ed25519_fp.metal", encoding: .utf8)
        let curveSource = try String(contentsOfFile: shaderDir + "/geometry/ed25519_curve.metal", encoding: .utf8)
        let msmSource = try String(contentsOfFile: shaderDir + "/msm/ed25519_msm_kernels.metal", encoding: .utf8)

        func stripIncludes(_ s: String) -> String {
            s.split(separator: "\n").filter { !$0.contains("#include") }.joined(separator: "\n")
        }
        func stripGuards(_ s: String) -> String {
            s.replacingOccurrences(of: "#ifndef ED25519_FP_METAL", with: "")
             .replacingOccurrences(of: "#define ED25519_FP_METAL", with: "")
             .replacingOccurrences(of: "#endif // ED25519_FP_METAL", with: "")
             .replacingOccurrences(of: "#ifndef ED25519_CURVE_METAL", with: "")
             .replacingOccurrences(of: "#define ED25519_CURVE_METAL", with: "")
             .replacingOccurrences(of: "#endif // ED25519_CURVE_METAL", with: "")
        }

        let combined = stripGuards(fpSource) + "\n" +
                        stripGuards(stripIncludes(curveSource)) + "\n" +
                        stripIncludes(msmSource)

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        let library = try device.makeLibrary(source: combined, options: options)

        try? FileManager.default.createDirectory(
            at: Ed25519MSM.cacheDir, withIntermediateDirectories: true)

        return library
    }

    // Ed25519 scalar field order q as 8x32-bit limbs (little-endian)
    private static let Q_LIMBS: [UInt32] = [
        0x5cf5d3ed, 0x5812631a, 0xa2f79cd6, 0x14def9de,
        0x00000000, 0x00000000, 0x00000000, 0x10000000
    ]

    private static let HALF_Q: [UInt32] = {
        var r = [UInt32](repeating: 0, count: 8)
        for i in 0..<8 {
            r[i] = Q_LIMBS[i] >> 1
            if i < 7 { r[i] |= (Q_LIMBS[i+1] & 1) << 31 }
        }
        return r
    }()

    public static func reduceModQ(_ scalar: [UInt32]) -> [UInt32] {
        var current = scalar
        while true {
            if !gte(current, Q_LIMBS) { return current }
            var result = [UInt32](repeating: 0, count: 8)
            var borrow: Int64 = 0
            for i in 0..<8 {
                borrow += Int64(current[i]) - Int64(Q_LIMBS[i])
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

    private static func subQ(_ scalar: [UInt32]) -> [UInt32] {
        var result = [UInt32](repeating: 0, count: 8)
        var borrow: Int64 = 0
        for i in 0..<8 {
            borrow += Int64(Q_LIMBS[i]) - Int64(scalar[i])
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
                length: MemoryLayout<Ed25519PointAffine>.stride * np, options: .storageModeShared)
            sortedIndicesBuffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride * np * nw, options: .storageModeShared)
            allOffsetsBuffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride * nb * nw, options: .storageModeShared)
            allCountsBuffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride * nb * nw, options: .storageModeShared)
            bucketsBuffer = device.makeBuffer(
                length: MemoryLayout<Ed25519PointExtended>.stride * nb * nw, options: .storageModeShared)
            segmentResultsBuffer = device.makeBuffer(
                length: MemoryLayout<Ed25519PointExtended>.stride * ns * nw, options: .storageModeShared)
            windowResultsBuffer = device.makeBuffer(
                length: MemoryLayout<Ed25519PointExtended>.stride * nw, options: .storageModeShared)
            countSortedMapBuffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride * nb * nw, options: .storageModeShared)
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

    /// CPU-only Pippenger for small inputs
    private func cpuPippenger(points: [Ed25519PointAffine], scalars: [[UInt32]]) -> Ed25519PointExtended {
        let n = points.count
        if n == 0 { return ed25519PointIdentity() }

        var result = ed25519PointIdentity()
        for i in 0..<n {
            var limbs: [UInt64] = [0, 0, 0, 0]
            for j in 0..<4 {
                limbs[j] = UInt64(scalars[i][j * 2]) | (UInt64(scalars[i][j * 2 + 1]) << 32)
            }
            let p = ed25519PointFromAffine(points[i])
            let sp = ed25519PointMulScalar(p, limbs)
            result = ed25519PointAdd(result, sp)
        }
        return result
    }

    public func msm(points: [Ed25519PointAffine], scalars: [[UInt32]]) throws -> Ed25519PointExtended {
        let n = points.count
        guard n == scalars.count, n > 0 else {
            throw MSMError.invalidInput
        }

        // For small inputs, CPU is faster
        if n <= 512 {
            return cpuPippenger(points: points, scalars: scalars)
        }

        let msmScalars = scalars.map { Self.reduceModQ($0) }
        let scalarBits = 253  // q is ~252 bits, +1 for signed-digit carry

        // Center scalars: if scalar > q/2, negate point and use q-scalar
        var centeredPoints = points
        var centeredScalars = msmScalars
        for i in 0..<n {
            if Self.gte(centeredScalars[i], Self.HALF_Q) {
                centeredScalars[i] = Self.subQ(centeredScalars[i])
                centeredPoints[i] = ed25519PointNegAffine(centeredPoints[i])
            }
        }

        var windowBits: UInt32
        if n <= 256 { windowBits = 8 }
        else if n <= 4096 { windowBits = 10 }
        else if n <= 65536 { windowBits = 13 }
        else { windowBits = 16 }

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
              let countSortedMapBuffer = countSortedMapBuffer,
              let dBuffer = dBuffer else {
            throw MSMError.gpuError("Failed to allocate Metal buffers")
        }

        // Convert points from direct integer form to Montgomery form for GPU
        let gpuPtsPtr = pointsBuffer.contents().bindMemory(to: Ed25519PointAffine.self, capacity: n)
        var montPoints = centeredPoints.map { pt in
            Ed25519PointAffine(x: pt.x.toMontgomery(), y: pt.y.toMontgomery())
        }
        montPoints.withUnsafeBufferPointer { src in
            gpuPtsPtr.update(from: src.baseAddress!, count: n)
        }

        let allOffsets = allOffsetsBuffer.contents().bindMemory(to: UInt32.self, capacity: nBuckets * nWindows)
        let allCounts = allCountsBuffer.contents().bindMemory(to: UInt32.self, capacity: nBuckets * nWindows)
        let sortedIdxPtr = sortedIndicesBuffer.contents().bindMemory(to: UInt32.self, capacity: n * nWindows)
        let countSortedMap = countSortedMapBuffer.contents().bindMemory(to: UInt32.self, capacity: nBuckets * nWindows)

        var params = EdMsmParamsSwift(n_points: UInt32(n), window_bits: windowBits, n_buckets: UInt32(nBuckets))
        var nSegs = UInt32(nSegments)

        let countsBase = cpuCountsPtr!
        let positionsBase = cpuPositionsPtr!

        // CPU signed-digit extraction
        do {
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

            let chunkSize = 4096
            let nChunks = (n + chunkSize - 1) / chunkSize
            DispatchQueue.concurrentPerform(iterations: nChunks) { chunk in
                let start = chunk * chunkSize
                let end = min(start + chunkSize, n)
                for i in start..<end {
                    var carry: UInt32 = 0
                    centeredScalars[i].withUnsafeBufferPointer { sp in
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
                                signedDigitBuf[w * n + i] = digit | 0x80000000
                            } else {
                                signedDigitBuf[w * n + i] = digit
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
            let idxBase = w * n
            let counts = countsBase + w * nBuckets
            let positions = positionsBase + w * nBuckets
            let sdBuf = signedDigitBufFinal + w * n

            for i in 0..<nBuckets { counts[i] = 0 }
            for i in 0..<n { counts[Int(sdBuf[i] & 0x7FFFFFFF)] += 1 }

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
            for i in 0..<nBuckets { counts[Int(allCounts[wOff + i])] += 1 }
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

        // GPU: reduce + bucket_sum + combine
        guard let cb = commandQueue.makeCommandBuffer() else { throw MSMError.noCommandBuffer }

        // Phase 1: Reduce sorted buckets
        do {
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(reduceSortedFunction)
            enc.setBuffer(pointsBuffer, offset: 0, index: 0)
            enc.setBuffer(bucketsBuffer, offset: 0, index: 1)
            enc.setBuffer(allOffsetsBuffer, offset: 0, index: 2)
            enc.setBuffer(allCountsBuffer, offset: 0, index: 3)
            enc.setBytes(&params, length: MemoryLayout<EdMsmParamsSwift>.stride, index: 4)
            var nw = UInt32(nWindows)
            enc.setBytes(&nw, length: 4, index: 5)
            enc.setBuffer(sortedIndicesBuffer, offset: 0, index: 6)
            enc.setBuffer(countSortedMapBuffer, offset: 0, index: 7)
            enc.setBuffer(dBuffer, offset: 0, index: 8)
            let numBucketsTotal = nWindows * nBuckets
            let tg = min(256, Int(reduceSortedFunction.maxTotalThreadsPerThreadgroup))
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
            enc.setBytes(&params, length: MemoryLayout<EdMsmParamsSwift>.stride, index: 2)
            enc.setBytes(&nSegs, length: 4, index: 3)
            enc.setBytes(&nWinsBatch, length: 4, index: 4)
            enc.setBuffer(dBuffer, offset: 0, index: 5)
            let totalSegments = nSegments * nWindows
            enc.dispatchThreads(
                MTLSize(width: totalSegments, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(256, totalSegments), height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)

            enc.setComputePipelineState(combineSegmentsFunction)
            enc.setBuffer(segmentResultsBuffer, offset: 0, index: 0)
            enc.setBuffer(windowResultsBuffer, offset: 0, index: 1)
            enc.setBytes(&nSegs, length: 4, index: 2)
            enc.setBuffer(dBuffer, offset: 0, index: 3)
            enc.dispatchThreads(
                MTLSize(width: nWindows, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(256, nWindows), height: 1, depth: 1))
            enc.endEncoding()
        }
        cb.commit()
        cb.waitUntilCompleted()

        if let error = cb.error { throw MSMError.gpuError(error.localizedDescription) }

        // Read GPU results (Montgomery form) and convert to direct integer form for CPU
        let winResultsPtr = windowResultsBuffer.contents().bindMemory(to: Ed25519PointExtended.self, capacity: nWindows)
        var windowResults = [Ed25519PointExtended](repeating: ed25519PointIdentity(), count: nWindows)
        for w in 0..<nWindows {
            let gpt = winResultsPtr[w]
            windowResults[w] = Ed25519PointExtended(
                x: Ed25519Fp.fromMontgomery(gpt.x),
                y: Ed25519Fp.fromMontgomery(gpt.y),
                z: Ed25519Fp.fromMontgomery(gpt.z),
                t: Ed25519Fp.fromMontgomery(gpt.t)
            )
        }

        // Horner's method on CPU
        var result = windowResults.last!
        for w in stride(from: nWindows - 2, through: 0, by: -1) {
            for _ in 0..<windowBits {
                result = ed25519PointDouble(result)
            }
            result = ed25519PointAdd(result, windowResults[w])
        }
        return result
    }
}

// Must match Metal EdMsmParams struct layout
public struct EdMsmParamsSwift {
    public var n_points: UInt32
    public var window_bits: UInt32
    public var n_buckets: UInt32

    public init(n_points: UInt32, window_bits: UInt32, n_buckets: UInt32) {
        self.n_points = n_points
        self.window_bits = window_bits
        self.n_buckets = n_buckets
    }
}
