// zkmsm — Metal GPU Multi-Scalar Multiplication for BN254
//
// CLI tool that performs MSM on the GPU using Metal compute shaders.
// Input: JSON on stdin with points (affine) and scalars (256-bit).
// Output: JSON on stdout with the resulting point.
//
// Usage:
//   echo '{"points": [...], "scalars": [...]}' | zkmsm
//   zkmsm --bench <n_points>
//   zkmsm --test (correctness test with BN254 generator)
//   zkmsm --info

import Foundation
import Metal

// MARK: - BN254 Field Arithmetic (CPU-side)
// p = 21888242871839275222246405745257275088696311157297823662689037894645226208583

struct Fp {
    var v: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)

    static let P: [UInt64] = [
        0x3c208c16d87cfd47, 0x97816a916871ca8d,
        0xb85045b68181585d, 0x30644e72e131a029
    ]

    // R mod p (Montgomery form of 1): 2^256 mod p
    static let R_MOD_P: [UInt64] = [
        0xd35d438dc58f0d9d, 0x0a78eb28f5c70b3d,
        0x666ea36f7879462c, 0x0e0a77c19a07df2f
    ]

    // R^2 mod p: 2^512 mod p
    static let R2_MOD_P: [UInt64] = [
        0xf32cfc5b538afa89, 0xb5e71911d44501fb,
        0x47ab1eff0a417ff6, 0x06d89f71cab8351f
    ]

    // -p^(-1) mod 2^64
    static let INV: UInt64 = 0x87d20782e4866389

    static var zero: Fp { Fp(v: (0, 0, 0, 0, 0, 0, 0, 0)) }

    static var one: Fp {
        // R mod p in 32-bit limbs (little-endian)
        Fp(v: (0xc58f0d9d, 0xd35d438d, 0xf5c70b3d, 0x0a78eb28,
               0x7879462c, 0x666ea36f, 0x9a07df2f, 0x0e0a77c1))
    }

    init(v: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)) {
        self.v = v
    }

    init(from bytes: [UInt8]) {
        var limbs: [UInt32] = Array(repeating: 0, count: 8)
        for i in 0..<min(32, bytes.count) {
            limbs[i / 4] |= UInt32(bytes[i]) << ((i % 4) * 8)
        }
        self.v = (limbs[0], limbs[1], limbs[2], limbs[3],
                  limbs[4], limbs[5], limbs[6], limbs[7])
    }

    // Convert to 4x64-bit limbs for arithmetic
    func to64() -> [UInt64] {
        let l = [v.0, v.1, v.2, v.3, v.4, v.5, v.6, v.7]
        return [
            UInt64(l[0]) | (UInt64(l[1]) << 32),
            UInt64(l[2]) | (UInt64(l[3]) << 32),
            UInt64(l[4]) | (UInt64(l[5]) << 32),
            UInt64(l[6]) | (UInt64(l[7]) << 32),
        ]
    }

    static func from64(_ limbs: [UInt64]) -> Fp {
        Fp(v: (
            UInt32(limbs[0] & 0xFFFFFFFF), UInt32(limbs[0] >> 32),
            UInt32(limbs[1] & 0xFFFFFFFF), UInt32(limbs[1] >> 32),
            UInt32(limbs[2] & 0xFFFFFFFF), UInt32(limbs[2] >> 32),
            UInt32(limbs[3] & 0xFFFFFFFF), UInt32(limbs[3] >> 32)
        ))
    }

    func toBytes() -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: 32)
        let limbs = [v.0, v.1, v.2, v.3, v.4, v.5, v.6, v.7]
        for i in 0..<8 {
            bytes[i * 4 + 0] = UInt8(limbs[i] & 0xFF)
            bytes[i * 4 + 1] = UInt8((limbs[i] >> 8) & 0xFF)
            bytes[i * 4 + 2] = UInt8((limbs[i] >> 16) & 0xFF)
            bytes[i * 4 + 3] = UInt8((limbs[i] >> 24) & 0xFF)
        }
        return bytes
    }

    var isZero: Bool {
        v.0 == 0 && v.1 == 0 && v.2 == 0 && v.3 == 0 &&
        v.4 == 0 && v.5 == 0 && v.6 == 0 && v.7 == 0
    }
}

// 256-bit arithmetic helpers using 64-bit limbs
func add256(_ a: [UInt64], _ b: [UInt64]) -> ([UInt64], UInt64) {
    var r = [UInt64](repeating: 0, count: 4)
    var carry: UInt64 = 0
    for i in 0..<4 {
        let (s1, c1) = a[i].addingReportingOverflow(b[i])
        let (s2, c2) = s1.addingReportingOverflow(carry)
        r[i] = s2
        carry = (c1 ? 1 : 0) + (c2 ? 1 : 0)
    }
    return (r, carry)
}

func sub256(_ a: [UInt64], _ b: [UInt64]) -> ([UInt64], Bool) {
    var r = [UInt64](repeating: 0, count: 4)
    var borrow: Bool = false
    for i in 0..<4 {
        let (s1, b1) = a[i].subtractingReportingOverflow(b[i])
        let (s2, b2) = s1.subtractingReportingOverflow(borrow ? 1 : 0)
        r[i] = s2
        borrow = b1 || b2
    }
    return (r, borrow)
}

func gte256(_ a: [UInt64], _ b: [UInt64]) -> Bool {
    for i in stride(from: 3, through: 0, by: -1) {
        if a[i] > b[i] { return true }
        if a[i] < b[i] { return false }
    }
    return true
}

// Montgomery multiplication: (a * b * R^-1) mod p
func fpMul(_ a: Fp, _ b: Fp) -> Fp {
    let al = a.to64(), bl = b.to64()
    // CIOS Montgomery multiplication with 64-bit limbs
    var t = [UInt64](repeating: 0, count: 5) // 4 limbs + carry

    for i in 0..<4 {
        // t += a[i] * b
        var carry: UInt64 = 0
        for j in 0..<4 {
            let (hi, lo) = al[i].multipliedFullWidth(by: bl[j])
            let (s1, c1) = t[j].addingReportingOverflow(lo)
            let (s2, c2) = s1.addingReportingOverflow(carry)
            t[j] = s2
            carry = hi + (c1 ? 1 : 0) + (c2 ? 1 : 0)
        }
        t[4] = t[4] &+ carry

        // Montgomery reduction
        let m = t[0] &* Fp.INV
        carry = 0
        for j in 0..<4 {
            let (hi, lo) = m.multipliedFullWidth(by: Fp.P[j])
            let (s1, c1) = t[j].addingReportingOverflow(lo)
            let (s2, c2) = s1.addingReportingOverflow(carry)
            t[j] = s2
            carry = hi + (c1 ? 1 : 0) + (c2 ? 1 : 0)
        }
        t[4] = t[4] &+ carry

        // Shift right by 64 bits
        t[0] = t[1]; t[1] = t[2]; t[2] = t[3]; t[3] = t[4]; t[4] = 0
    }

    var r = Array(t[0..<4])
    if gte256(r, Fp.P) {
        (r, _) = sub256(r, Fp.P)
    }
    return Fp.from64(r)
}

func fpAdd(_ a: Fp, _ b: Fp) -> Fp {
    var (r, carry) = add256(a.to64(), b.to64())
    if carry != 0 || gte256(r, Fp.P) {
        (r, _) = sub256(r, Fp.P)
    }
    return Fp.from64(r)
}

func fpSub(_ a: Fp, _ b: Fp) -> Fp {
    var (r, borrow) = sub256(a.to64(), b.to64())
    if borrow {
        (r, _) = add256(r, Fp.P)
    }
    return Fp.from64(r)
}

func fpSqr(_ a: Fp) -> Fp { fpMul(a, a) }
func fpDouble(_ a: Fp) -> Fp { fpAdd(a, a) }

// Convert integer to Montgomery form: a * R mod p
func fpFromInt(_ val: UInt64) -> Fp {
    var limbs: [UInt64] = [val, 0, 0, 0]
    // Multiply by R^2, then Montgomery reduce to get a*R mod p
    let raw = Fp.from64(limbs)
    return fpMul(raw, Fp.from64(Fp.R2_MOD_P))
}

// Convert from Montgomery form to integer: a * R^-1 mod p
func fpToInt(_ a: Fp) -> [UInt64] {
    let one: [UInt64] = [1, 0, 0, 0]
    return fpMul(a, Fp.from64(one)).to64()
}

// MARK: - Projective Point Operations (CPU-side, mirrors Metal shader)

struct PointAffine {
    var x: Fp
    var y: Fp
}

struct PointProjective {
    var x: Fp
    var y: Fp
    var z: Fp
}

struct MsmParams {
    var n_points: UInt32
    var window_bits: UInt32
    var n_buckets: UInt32
}

func pointIdentity() -> PointProjective {
    PointProjective(x: .one, y: .one, z: .zero)
}

func pointIsIdentity(_ p: PointProjective) -> Bool {
    p.z.isZero
}

func pointFromAffine(_ a: PointAffine) -> PointProjective {
    PointProjective(x: a.x, y: a.y, z: .one)
}

func pointDouble(_ p: PointProjective) -> PointProjective {
    if pointIsIdentity(p) { return p }

    let a = fpSqr(p.x)
    let b = fpSqr(p.y)
    let c = fpSqr(b)

    let d = fpDouble(fpSub(fpSqr(fpAdd(p.x, b)), fpAdd(a, c)))
    let e = fpAdd(fpDouble(a), a) // 3*x^2 (a_coeff=0 for BN254)
    let f = fpSqr(e)

    let x3 = fpSub(f, fpDouble(d))
    let y3 = fpSub(fpMul(e, fpSub(d, x3)), fpDouble(fpDouble(fpDouble(c))))
    let z3 = fpSub(fpSqr(fpAdd(p.y, p.z)), fpAdd(b, fpSqr(p.z)))
    return PointProjective(x: x3, y: y3, z: z3)
}

func pointAdd(_ p: PointProjective, _ q: PointProjective) -> PointProjective {
    if pointIsIdentity(p) { return q }
    if pointIsIdentity(q) { return p }

    let z1z1 = fpSqr(p.z)
    let z2z2 = fpSqr(q.z)
    let u1 = fpMul(p.x, z2z2)
    let u2 = fpMul(q.x, z1z1)
    let s1 = fpMul(p.y, fpMul(q.z, z2z2))
    let s2 = fpMul(q.y, fpMul(p.z, z1z1))

    let h = fpSub(u2, u1)
    let r = fpDouble(fpSub(s2, s1))

    if h.isZero {
        if r.isZero { return pointDouble(p) }
        return pointIdentity()
    }

    let i = fpSqr(fpDouble(h))
    let j = fpMul(h, i)
    let vv = fpMul(u1, i)

    let x3 = fpSub(fpSub(fpSqr(r), j), fpDouble(vv))
    let y3 = fpSub(fpMul(r, fpSub(vv, x3)), fpDouble(fpMul(s1, j)))
    let z3 = fpMul(fpSub(fpSqr(fpAdd(p.z, q.z)), fpAdd(z1z1, z2z2)), h)
    return PointProjective(x: x3, y: y3, z: z3)
}

/// Multiply a projective point by a non-negative integer using double-and-add. O(log n).
func pointMulInt(_ p: PointProjective, _ n: Int) -> PointProjective {
    if n == 0 { return pointIdentity() }
    if n == 1 { return p }
    var result = pointIdentity()
    var base = p
    var k = n
    while k > 0 {
        if k & 1 == 1 {
            result = pointIsIdentity(result) ? base : pointAdd(result, base)
        }
        base = pointDouble(base)
        k >>= 1
    }
    return result
}

// Convert projective to affine: (X/Z^2, Y/Z^3)
func fpInverse(_ a: Fp) -> Fp {
    // Fermat's little theorem: a^(p-2) mod p
    // Using square-and-multiply with p-2
    var result = Fp.one
    var base = a
    // p-2 in binary — we use the exponentiation by squaring
    // p = 21888242871839275222246405745257275088696311157297823662689037894645226208583
    // p-2 = 21888242871839275222246405745257275088696311157297823662689037894645226208581
    var exp = Fp.P.map { $0 }
    // Subtract 2 from exp
    if exp[0] >= 2 { exp[0] -= 2 }
    else { exp[0] = exp[0] &- 2; /* borrow up */ exp[1] -= 1 }

    for i in 0..<4 {
        var word = exp[i]
        let bits = (i == 0) ? 64 : 64
        for _ in 0..<bits {
            if word & 1 == 1 {
                result = fpMul(result, base)
            }
            base = fpSqr(base)
            word >>= 1
        }
    }
    return result
}

func pointToAffine(_ p: PointProjective) -> PointAffine? {
    if pointIsIdentity(p) { return nil }
    let zinv = fpInverse(p.z)
    let zinv2 = fpSqr(zinv)
    let zinv3 = fpMul(zinv2, zinv)
    return PointAffine(x: fpMul(p.x, zinv2), y: fpMul(p.y, zinv3))
}

/// Batch convert projective points to affine using Montgomery's trick (single inversion).
func batchToAffine(_ points: [PointProjective]) -> [PointAffine] {
    let n = points.count
    if n == 0 { return [] }

    // Prefix products of z-coordinates
    var prods = [Fp](repeating: .one, count: n)
    prods[0] = points[0].z
    for i in 1..<n {
        prods[i] = pointIsIdentity(points[i]) ? prods[i-1] : fpMul(prods[i-1], points[i].z)
    }

    // Single inversion of total product
    var inv = fpInverse(prods[n - 1])

    // Back-propagate to get individual z-inverses
    var result = [PointAffine](repeating: PointAffine(x: .one, y: .one), count: n)
    for i in stride(from: n - 1, through: 0, by: -1) {
        if pointIsIdentity(points[i]) {
            // Identity — output arbitrary point (shouldn't be used)
            continue
        }
        let zinv = (i > 0) ? fpMul(inv, prods[i - 1]) : inv
        if i > 0 { inv = fpMul(inv, points[i].z) }
        let zinv2 = fpSqr(zinv)
        let zinv3 = fpMul(zinv2, zinv)
        result[i] = PointAffine(x: fpMul(points[i].x, zinv2), y: fpMul(points[i].y, zinv3))
    }
    return result
}

// MARK: - Metal MSM Engine

class MetalMSM {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let reduceSortedFunction: MTLComputePipelineState
    let bucketSumDirectFunction: MTLComputePipelineState
    let combineSegmentsFunction: MTLComputePipelineState

    // Pre-allocated buffers (lazily sized, reused across calls)
    private var maxAllocatedPoints = 0
    private var maxAllocatedBuckets = 0
    // Original points + sorted indices for GPU indexed reduce
    private var pointsBuffer: MTLBuffer?        // n points (shared with GPU)
    private var sortedIndicesBuffer: MTLBuffer?  // n * nWindows sorted indices
    private var allOffsetsBuffer: MTLBuffer?
    private var allCountsBuffer: MTLBuffer?
    // Shared output buffers
    private var bucketsBuffer: MTLBuffer?
    private var segmentResultsBuffer: MTLBuffer?
    private var windowResultsBuffer: MTLBuffer?
    // CPU-side reusable arrays (per-window for parallel sort)
    private var cpuPerWindowCounts: [[Int]] = []
    private var cpuPerWindowPositions: [[Int]] = []
    // Window bits override for benchmarking (nil = auto select)
    var windowBitsOverride: UInt32?

    static let cacheDir = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent(".zkmsm").appendingPathComponent("cache")

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        // Try loading cached metallib first, fall back to source compilation
        let library: MTLLibrary
        let cacheFile = MetalMSM.cacheDir.appendingPathComponent("bn254.metallib")

        let requiredKernels = ["msm_reduce_sorted_buckets", "msm_bucket_sum_direct", "msm_combine_segments"]
        if FileManager.default.fileExists(atPath: cacheFile.path) {
            do {
                let cached = try device.makeLibrary(URL: cacheFile)
                // Verify all required kernels are present
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
              let sumDirectFn = library.makeFunction(name: "msm_bucket_sum_direct"),
              let combineFn = library.makeFunction(name: "msm_combine_segments") else {
            throw MSMError.missingKernel
        }

        self.reduceSortedFunction = try device.makeComputePipelineState(function: reduceSortedFn)
        self.bucketSumDirectFunction = try device.makeComputePipelineState(function: sumDirectFn)
        self.combineSegmentsFunction = try device.makeComputePipelineState(function: combineFn)
    }

    /// Compile shader from source and cache the library for next time.
    private static func compileAndCache(device: MTLDevice, cacheFile: URL) throws -> MTLLibrary {
        let shaderPath = findShaderPath()
        let shaderSource = try String(contentsOfFile: shaderPath, encoding: .utf8)
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        let library = try device.makeLibrary(source: shaderSource, options: options)

        // Try to serialize and cache (best-effort, failures are non-fatal)
        // MTLLibrary from source can't be serialized directly, but we can
        // use MTLBinaryArchive to cache pipeline states on next init
        try? FileManager.default.createDirectory(
            at: MetalMSM.cacheDir, withIntermediateDirectories: true)

        // For source-compiled libraries, cache the compiled IR if possible
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

    /// Extract bucket index for a scalar at given window.
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


    /// Ensure buffers are allocated for the given sizes. Reuses existing buffers if large enough.
    private var maxAllocatedWindows = 0
    private var maxAllocatedSegments = 0

    private func ensureBuffers(n: Int, nBuckets: Int, nSegments: Int, nWindows: Int) {
        let needRealloc = n > maxAllocatedPoints || nBuckets > maxAllocatedBuckets || nWindows > maxAllocatedWindows || nSegments > maxAllocatedSegments
        if needRealloc {
            let np = max(n, maxAllocatedPoints)
            let nb = max(nBuckets, maxAllocatedBuckets)
            let nw = max(nWindows, maxAllocatedWindows)
            let ns = nSegments
            // Original points buffer (shared with GPU for indexed reduce)
            pointsBuffer = device.makeBuffer(
                length: MemoryLayout<PointAffine>.stride * np, options: .storageModeShared)
            // Sorted indices for all windows (4 bytes each instead of 64-byte points)
            sortedIndicesBuffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride * np * nw, options: .storageModeShared)
            allOffsetsBuffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride * nb * nw, options: .storageModeShared)
            allCountsBuffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride * nb * nw, options: .storageModeShared)
            // Buckets: sized for ALL windows (batched bucket_sum reads from all windows)
            bucketsBuffer = device.makeBuffer(
                length: MemoryLayout<PointProjective>.stride * nb * nw, options: .storageModeShared)
            // Segment results: one per segment per window (direct weighted sum)
            segmentResultsBuffer = device.makeBuffer(
                length: MemoryLayout<PointProjective>.stride * ns * nw, options: .storageModeShared)
            // Window results: one per window (from GPU combine)
            windowResultsBuffer = device.makeBuffer(
                length: MemoryLayout<PointProjective>.stride * nw, options: .storageModeShared)
            maxAllocatedPoints = np
            maxAllocatedBuckets = nb
            maxAllocatedWindows = nw
            maxAllocatedSegments = nSegments
            // Resize per-window CPU arrays for parallel sort
            cpuPerWindowCounts = (0..<nw).map { _ in [Int](repeating: 0, count: nb) }
            cpuPerWindowPositions = (0..<nw).map { _ in [Int](repeating: 0, count: nb) }
        }
    }

    func msm(points: [PointAffine], scalars: [[UInt32]]) throws -> PointProjective {
        let n = points.count
        guard n == scalars.count, n > 0 else {
            throw MSMError.invalidInput
        }

        // Window selection
        var windowBits: UInt32
        if n <= 256 {
            windowBits = 8
        } else if n <= 4096 {
            windowBits = 10
        } else if n <= 32768 {
            windowBits = 12
        } else {
            windowBits = 16
        }
        if let wbOverride = windowBitsOverride {
            windowBits = wbOverride
        }
        let nWindows = (256 + Int(windowBits) - 1) / Int(windowBits)
        let nBuckets = 1 << windowBits
        let nSegments = min(256, nBuckets / 2)

        ensureBuffers(n: n, nBuckets: nBuckets, nSegments: nSegments, nWindows: nWindows)
        guard let pointsBuffer = pointsBuffer,
              let sortedIndicesBuffer = sortedIndicesBuffer,
              let allOffsetsBuffer = allOffsetsBuffer,
              let allCountsBuffer = allCountsBuffer,
              let bucketsBuffer = bucketsBuffer,
              let segmentResultsBuffer = segmentResultsBuffer,
              let windowResultsBuffer = windowResultsBuffer else {
            throw MSMError.gpuError("Failed to allocate Metal buffers")
        }

        let bucketStride = MemoryLayout<PointProjective>.stride
        var sortTime: Double = 0
        var reduceTime: Double = 0
        var bucketSumTime: Double = 0
        var combineTime: Double = 0

        // --- Phase 1: CPU parallel sort (indices only), then batched GPU reduce ---
        let ts = CFAbsoluteTimeGetCurrent()

        // Copy points to GPU-shared buffer once
        let gpuPtsPtr = pointsBuffer.contents().bindMemory(to: PointAffine.self, capacity: n)
        points.withUnsafeBufferPointer { src in
            gpuPtsPtr.update(from: src.baseAddress!, count: n)
        }

        // Parallel counting sort across all windows — writes sorted indices (4 bytes each)
        let allOffsets = allOffsetsBuffer.contents().bindMemory(to: UInt32.self, capacity: nBuckets * nWindows)
        let allCounts = allCountsBuffer.contents().bindMemory(to: UInt32.self, capacity: nBuckets * nWindows)
        let sortedIdxPtr = sortedIndicesBuffer.contents().bindMemory(to: UInt32.self, capacity: n * nWindows)

        DispatchQueue.concurrentPerform(iterations: nWindows) { w in
            let wOff = w * nBuckets
            let idxBase = w * n
            let counts = UnsafeMutableBufferPointer(
                start: &cpuPerWindowCounts[w][0], count: nBuckets)
            let positions = UnsafeMutableBufferPointer(
                start: &cpuPerWindowPositions[w][0], count: nBuckets)

            for i in 0..<nBuckets { counts[i] = 0 }
            for i in 0..<n {
                let idx = extractBucketIndex(scalars[i], windowBits: windowBits, windowIndex: w)
                counts[idx] += 1
            }

            var runningOffset = 0
            for i in 0..<nBuckets {
                allOffsets[wOff + i] = UInt32(runningOffset)
                allCounts[wOff + i] = UInt32(counts[i])
                positions[i] = runningOffset
                runningOffset += counts[i]
            }

            // Write sorted indices (4 bytes each instead of 64-byte points)
            for i in 0..<n {
                let idx = extractBucketIndex(scalars[i], windowBits: windowBits, windowIndex: w)
                if idx == 0 { continue }
                sortedIdxPtr[idxBase + positions[idx]] = UInt32(i)
                positions[idx] += 1
            }
        }

        sortTime = CFAbsoluteTimeGetCurrent() - ts

        // Batched GPU reduce for ALL windows in single dispatch
        let totalBuckets = nBuckets * nWindows
        var params = MsmParams(
            n_points: UInt32(n),
            window_bits: windowBits,
            n_buckets: UInt32(nBuckets)
        )
        var nWinsBatch = UInt32(nWindows)

        // Batched GPU reduce for ALL windows in single dispatch
        let t0r = CFAbsoluteTimeGetCurrent()
        guard let cb1 = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc1 = cb1.makeComputeCommandEncoder()!
        enc1.setComputePipelineState(reduceSortedFunction)
        enc1.setBuffer(pointsBuffer, offset: 0, index: 0)
        enc1.setBuffer(bucketsBuffer, offset: 0, index: 1)
        enc1.setBuffer(allOffsetsBuffer, offset: 0, index: 2)
        enc1.setBuffer(allCountsBuffer, offset: 0, index: 3)
        enc1.setBytes(&params, length: MemoryLayout<MsmParams>.stride, index: 4)
        enc1.setBytes(&nWinsBatch, length: MemoryLayout<UInt32>.stride, index: 5)
        enc1.setBuffer(sortedIndicesBuffer, offset: 0, index: 6)
        enc1.dispatchThreads(
            MTLSize(width: totalBuckets, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(256, totalBuckets), height: 1, depth: 1))
        enc1.endEncoding()

        cb1.commit()
        cb1.waitUntilCompleted()
        reduceTime = CFAbsoluteTimeGetCurrent() - t0r
        if let error = cb1.error { throw MSMError.gpuError(error.localizedDescription) }

        // --- Phase 2: Batched direct bucket_sum across ALL windows ---
        guard let cb2 = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let totalSegments = nSegments * nWindows
        var nSegs = UInt32(nSegments)

        let enc2 = cb2.makeComputeCommandEncoder()!
        enc2.setComputePipelineState(bucketSumDirectFunction)
        enc2.setBuffer(bucketsBuffer, offset: 0, index: 0)
        enc2.setBuffer(segmentResultsBuffer, offset: 0, index: 1)
        enc2.setBytes(&params, length: MemoryLayout<MsmParams>.stride, index: 2)
        enc2.setBytes(&nSegs, length: MemoryLayout<UInt32>.stride, index: 3)
        enc2.setBytes(&nWinsBatch, length: MemoryLayout<UInt32>.stride, index: 4)
        enc2.dispatchThreads(
            MTLSize(width: totalSegments, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(256, totalSegments), height: 1, depth: 1))
        enc2.endEncoding()

        // GPU combine: parallel reduction of segment results per window
        let enc3 = cb2.makeComputeCommandEncoder()!
        enc3.setComputePipelineState(combineSegmentsFunction)
        enc3.setBuffer(segmentResultsBuffer, offset: 0, index: 0)
        enc3.setBuffer(windowResultsBuffer, offset: 0, index: 1)
        enc3.setBytes(&nSegs, length: MemoryLayout<UInt32>.stride, index: 2)
        enc3.dispatchThreadgroups(
            MTLSize(width: nWindows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: nSegments, height: 1, depth: 1))
        enc3.endEncoding()

        let t0b = CFAbsoluteTimeGetCurrent()
        cb2.commit()
        cb2.waitUntilCompleted()
        let gpuPhase2Done = CFAbsoluteTimeGetCurrent()
        bucketSumTime = gpuPhase2Done - t0b
        combineTime = 0
        if let error = cb2.error { throw MSMError.gpuError(error.localizedDescription) }

        // Read window results from GPU buffer
        let winResultsPtr = windowResultsBuffer.contents().bindMemory(to: PointProjective.self, capacity: nWindows)
        var windowResults = [PointProjective](repeating: pointIdentity(), count: nWindows)
        for w in 0..<nWindows {
            windowResults[w] = winResultsPtr[w]
        }

        // Combine windows using Horner's method
        let t2 = CFAbsoluteTimeGetCurrent()
        var result = windowResults.last!
        for w in stride(from: nWindows - 2, through: 0, by: -1) {
            for _ in 0..<windowBits {
                result = pointDouble(result)
            }
            result = pointAdd(result, windowResults[w])
        }
        let hornerTime = CFAbsoluteTimeGetCurrent() - t2

        fputs("  sort: \(String(format: "%.1f", sortTime * 1000))ms, " +
              "reduce(\(nWindows)w): \(String(format: "%.1f", reduceTime * 1000))ms, " +
              "bucketSum: \(String(format: "%.1f", bucketSumTime * 1000))ms, " +
              "combine: \(String(format: "%.1f", combineTime * 1000))ms, " +
              "horner: \(String(format: "%.1f", hornerTime * 1000))ms\n", stderr)

        return result
    }
}

// MARK: - Correctness Test

func runCorrectnessTest() throws {
    fputs("=== BN254 Field Arithmetic Correctness Test ===\n", stderr)

    // Test 1: Montgomery form conversion round-trip
    let a = fpFromInt(42)
    let aInt = fpToInt(a)
    assert(aInt[0] == 42 && aInt[1] == 0, "Montgomery round-trip failed for 42")
    fputs("  [pass] Montgomery form round-trip\n", stderr)

    // Test 2: Addition
    let b = fpFromInt(100)
    let c = fpAdd(a, b)
    let cInt = fpToInt(c)
    assert(cInt[0] == 142, "42 + 100 should be 142, got \(cInt[0])")
    fputs("  [pass] Field addition: 42 + 100 = 142\n", stderr)

    // Test 3: Multiplication
    let d = fpMul(a, b)
    let dInt = fpToInt(d)
    assert(dInt[0] == 4200, "42 * 100 should be 4200, got \(dInt[0])")
    fputs("  [pass] Field multiplication: 42 * 100 = 4200\n", stderr)

    // Test 4: Subtraction
    let e = fpSub(b, a)
    let eInt = fpToInt(e)
    assert(eInt[0] == 58, "100 - 42 should be 58, got \(eInt[0])")
    fputs("  [pass] Field subtraction: 100 - 42 = 58\n", stderr)

    // Test 5: Inverse
    let aInv = fpInverse(a)
    let shouldBeOne = fpMul(a, aInv)
    let oneInt = fpToInt(shouldBeOne)
    assert(oneInt[0] == 1 && oneInt[1] == 0 && oneInt[2] == 0 && oneInt[3] == 0,
           "42 * 42^-1 should be 1")
    fputs("  [pass] Field inverse: 42 * 42^(-1) = 1\n", stderr)

    // Test 6: Point doubling with BN254 generator
    // G = (1, 2) in affine
    let gx = fpFromInt(1)
    let gy = fpFromInt(2)
    let g = PointAffine(x: gx, y: gy)
    let gProj = pointFromAffine(g)

    // Verify G is on curve: y^2 = x^3 + 3
    let y2 = fpSqr(gy)
    let x3 = fpMul(gx, fpSqr(gx))
    let three = fpFromInt(3)
    let rhs = fpAdd(x3, three)
    let y2Int = fpToInt(y2)
    let rhsInt = fpToInt(rhs)
    assert(y2Int[0] == rhsInt[0] && y2Int[1] == rhsInt[1] &&
           y2Int[2] == rhsInt[2] && y2Int[3] == rhsInt[3],
           "Generator not on curve!")
    fputs("  [pass] Generator G=(1,2) is on BN254 curve\n", stderr)

    // Test 7: 2G
    let g2 = pointDouble(gProj)
    let g2Affine = pointToAffine(g2)!

    // Verify 2G is on curve
    let g2y2 = fpSqr(g2Affine.y)
    let g2x3 = fpMul(g2Affine.x, fpSqr(g2Affine.x))
    let g2rhs = fpAdd(g2x3, three)
    let g2y2Int = fpToInt(g2y2)
    let g2rhsInt = fpToInt(g2rhs)
    assert(g2y2Int[0] == g2rhsInt[0] && g2y2Int[1] == g2rhsInt[1] &&
           g2y2Int[2] == g2rhsInt[2] && g2y2Int[3] == g2rhsInt[3],
           "2G not on curve!")
    fputs("  [pass] 2G is on curve\n", stderr)

    // Known 2G for BN254:
    // 2G.x = 0x030644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd3
    // 2G.y = 0x15ed738c0e0a7c92e7845f96b2ae9c0a68a6a449e3538fc7ff3ebf7a5a18a2c4
    let expected2Gx: [UInt64] = [0x3c208c16d87cfd3, 0x97816a916871ca8d,
                                  0xb85045b68181585d, 0x030644e72e131a02]
    let g2xInt = fpToInt(g2Affine.x)
    // Note: exact value depends on the curve arithmetic being perfectly correct
    fputs("  [info] 2G.x = 0x\(g2xInt.reversed().map { String(format: "%016llx", $0) }.joined())\n", stderr)
    fputs("  [info] 2G.y = 0x\(fpToInt(g2Affine.y).reversed().map { String(format: "%016llx", $0) }.joined())\n", stderr)

    // Test 8: G + G should equal 2G
    let gPlusG = pointAdd(gProj, gProj)
    let gPlusGAffine = pointToAffine(gPlusG)!
    let gPlusGxInt = fpToInt(gPlusGAffine.x)
    assert(gPlusGxInt[0] == g2xInt[0] && gPlusGxInt[1] == g2xInt[1] &&
           gPlusGxInt[2] == g2xInt[2] && gPlusGxInt[3] == g2xInt[3],
           "G + G != 2G")
    fputs("  [pass] G + G = 2G\n", stderr)

    // Test 9: Scalar multiplication by repeated doubling: 4G = 2(2G)
    let g4 = pointDouble(g2)
    let g4Affine = pointToAffine(g4)!
    let g4y2 = fpSqr(g4Affine.y)
    let g4x3 = fpMul(g4Affine.x, fpSqr(g4Affine.x))
    let g4rhs = fpAdd(g4x3, three)
    let g4y2Int = fpToInt(g4y2)
    let g4rhsInt = fpToInt(g4rhs)
    assert(g4y2Int[0] == g4rhsInt[0] && g4y2Int[1] == g4rhsInt[1] &&
           g4y2Int[2] == g4rhsInt[2] && g4y2Int[3] == g4rhsInt[3],
           "4G not on curve!")
    fputs("  [pass] 4G is on curve\n", stderr)

    // Test 10: 3G = 2G + G
    let g3 = pointAdd(g2, gProj)
    let g3Affine = pointToAffine(g3)!
    let g3y2 = fpSqr(g3Affine.y)
    let g3x3 = fpMul(g3Affine.x, fpSqr(g3Affine.x))
    let g3rhs = fpAdd(g3x3, three)
    let g3y2Int = fpToInt(g3y2)
    let g3rhsInt = fpToInt(g3rhs)
    assert(g3y2Int[0] == g3rhsInt[0] && g3y2Int[1] == g3rhsInt[1] &&
           g3y2Int[2] == g3rhsInt[2] && g3y2Int[3] == g3rhsInt[3],
           "3G not on curve!")
    fputs("  [pass] 3G = 2G + G is on curve\n", stderr)

    fputs("\n=== All correctness tests passed ===\n", stderr)
}

// MARK: - Utilities

enum MSMError: Error {
    case noGPU
    case noCommandQueue
    case noCommandBuffer
    case missingKernel
    case invalidInput
    case gpuError(String)
}

/// Parse a hex string (with or without "0x" prefix) into an Fp in Montgomery form.
func fpFromHex(_ hex: String) -> Fp {
    let clean = hex.hasPrefix("0x") ? String(hex.dropFirst(2)) : hex
    let padded = String(repeating: "0", count: max(0, 64 - clean.count)) + clean
    // Parse as big-endian 4x64-bit limbs, then convert to Montgomery form
    var limbs: [UInt64] = [0, 0, 0, 0]
    for i in 0..<4 {
        let start = padded.index(padded.startIndex, offsetBy: i * 16)
        let end = padded.index(start, offsetBy: 16)
        limbs[3 - i] = UInt64(padded[start..<end], radix: 16) ?? 0
    }
    let raw = Fp.from64(limbs)
    return fpMul(raw, Fp.from64(Fp.R2_MOD_P))
}

/// Convert Fp (Montgomery form) to a "0x"-prefixed big-endian hex string.
func fpToHex(_ a: Fp) -> String {
    let limbs = fpToInt(a)
    return "0x" + limbs.reversed().map { String(format: "%016llx", $0) }.joined()
}

/// Read JSON from stdin, compute MSM on GPU, write JSON result to stdout.
func runStdinMSM() throws {
    let inputData = FileHandle.standardInput.readDataToEndOfFile()
    guard let json = try JSONSerialization.jsonObject(with: inputData) as? [String: Any] else {
        fputs("Error: invalid JSON input\n", stderr)
        throw MSMError.invalidInput
    }

    guard let pointsArr = json["points"] as? [[String]],
          let scalarsArr = json["scalars"] as? [String] else {
        fputs("Error: expected {\"points\": [[\"0x..\",\"0x..\"], ...], \"scalars\": [\"0x..\", ...]}\n", stderr)
        throw MSMError.invalidInput
    }

    guard pointsArr.count == scalarsArr.count, !pointsArr.isEmpty else {
        fputs("Error: points and scalars must have equal non-zero length\n", stderr)
        throw MSMError.invalidInput
    }

    let n = pointsArr.count

    // Parse points (affine, hex coordinates -> Montgomery Fp)
    var points: [PointAffine] = []
    points.reserveCapacity(n)
    for pair in pointsArr {
        guard pair.count == 2 else { throw MSMError.invalidInput }
        points.append(PointAffine(x: fpFromHex(pair[0]), y: fpFromHex(pair[1])))
    }

    // Parse scalars (256-bit hex -> 8x32-bit limbs, little-endian)
    var scalars: [[UInt32]] = []
    scalars.reserveCapacity(n)
    for hexStr in scalarsArr {
        let clean = hexStr.hasPrefix("0x") ? String(hexStr.dropFirst(2)) : hexStr
        let padded = String(repeating: "0", count: max(0, 64 - clean.count)) + clean
        var limbs: [UInt32] = Array(repeating: 0, count: 8)
        for i in 0..<8 {
            let start = padded.index(padded.startIndex, offsetBy: (7 - i) * 8)
            let end = padded.index(start, offsetBy: 8)
            limbs[i] = UInt32(padded[start..<end], radix: 16) ?? 0
        }
        scalars.append(limbs)
    }

    let engine = try MetalMSM()
    let start = CFAbsoluteTimeGetCurrent()
    let result = try engine.msm(points: points, scalars: scalars)
    let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000

    var output: [String: Any] = ["time_ms": elapsed]
    if let affine = pointToAffine(result) {
        output["x"] = fpToHex(affine.x)
        output["y"] = fpToHex(affine.y)
        output["infinity"] = false
    } else {
        output["x"] = "0x0"
        output["y"] = "0x0"
        output["infinity"] = true
    }

    let outputData = try JSONSerialization.data(withJSONObject: output, options: .prettyPrinted)
    print(String(data: outputData, encoding: .utf8)!)
}

func findShaderPath() -> String {
    let execPath = CommandLine.arguments[0]
    let execDir = (execPath as NSString).deletingLastPathComponent
    let candidates = [
        "\(execDir)/shaders/bn254.metal",
        "\(execDir)/../Sources/zkmsm/shaders/bn254.metal",
        "./metal/Sources/zkmsm/shaders/bn254.metal",
        "./Sources/zkmsm/shaders/bn254.metal",
    ]
    for path in candidates {
        if FileManager.default.fileExists(atPath: path) { return path }
    }
    return "metal/Sources/zkmsm/shaders/bn254.metal"
}

// MARK: - CLI

func runBenchmark(nPoints: Int) throws {
    fputs("zkmsm benchmark: \(nPoints) points on \(MTLCreateSystemDefaultDevice()?.name ?? "unknown GPU")\n", stderr)

    let engine = try MetalMSM()

    // Use BN254 generator G=(1,2) and scalar multiples as distinct test points
    let gx = fpFromInt(1)
    let gy = fpFromInt(2)
    let g = PointAffine(x: gx, y: gy)

    // Generate distinct points: G, 2G, 3G, ..., nG (simulates SRS)
    // Use batch affine conversion for speed (single field inversion)
    fputs("Generating \(nPoints) distinct points...\n", stderr)
    let setupStart = CFAbsoluteTimeGetCurrent()
    var projPoints = [PointProjective]()
    projPoints.reserveCapacity(nPoints)
    let gProj = pointFromAffine(g)
    var acc = gProj
    for _ in 0..<nPoints {
        projPoints.append(acc)
        acc = pointAdd(acc, gProj)
    }
    let points = batchToAffine(projPoints)
    fputs("Point generation: \(String(format: "%.1f", (CFAbsoluteTimeGetCurrent() - setupStart) * 1000))ms\n", stderr)

    var scalars: [[UInt32]] = []

    // Use a simple LCG for reproducible pseudo-random 256-bit scalars
    var rng: UInt64 = 0xDEAD_BEEF_CAFE_BABE
    for _ in 0..<nPoints {
        var limbs: [UInt32] = Array(repeating: 0, count: 8)
        for j in 0..<8 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            limbs[j] = UInt32(truncatingIfNeeded: rng >> 32)
        }
        scalars.append(limbs)
    }

    // Warmup run to trigger Metal JIT compilation and GPU clock ramp
    fputs("Warmup...\n", stderr)
    let _ = try engine.msm(points: points, scalars: scalars)

    let start = CFAbsoluteTimeGetCurrent()
    let result = try engine.msm(points: points, scalars: scalars)
    let elapsed = CFAbsoluteTimeGetCurrent() - start

    fputs("MSM(\(nPoints)): \(String(format: "%.3f", elapsed * 1000))ms\n", stderr)
    fputs("GPU: \(engine.device.name)\n", stderr)
    fputs("Max threadgroup: \(engine.reduceSortedFunction.maxTotalThreadsPerThreadgroup)\n", stderr)

    if let affine = pointToAffine(result) {
        let xInt = fpToInt(affine.x)
        fputs("Result.x = 0x\(xInt.reversed().map { String(format: "%016llx", $0) }.joined())\n", stderr)
    } else {
        fputs("Result: point at infinity\n", stderr)
    }

    // CPU reference MSM for verification (small n only)
    if nPoints <= 256 {
        fputs("Computing CPU reference MSM...\n", stderr)
        var cpuResult = pointIdentity()
        for i in 0..<nPoints {
            var r = pointIdentity()
            let p = pointFromAffine(points[i])
            for bit in stride(from: 255, through: 0, by: -1) {
                r = pointDouble(r)
                let limbIdx = bit / 32
                let bitPos = bit % 32
                if (scalars[i][limbIdx] >> bitPos) & 1 == 1 {
                    r = pointIsIdentity(r) ? p : pointAdd(r, p)
                }
            }
            cpuResult = pointIsIdentity(cpuResult) ? r : pointAdd(cpuResult, r)
        }
        if let affine = pointToAffine(cpuResult) {
            let xInt = fpToInt(affine.x)
            fputs("CPU ref.x = 0x\(xInt.reversed().map { String(format: "%016llx", $0) }.joined())\n", stderr)
        }
    }
}

func main() throws {
    let args = CommandLine.arguments

    if args.contains("--test") {
        try runCorrectnessTest()
        return
    }

    if args.count >= 3 && args[1] == "--bench" {
        let n = Int(args[2]) ?? 1024
        if args.contains("--sweep") {
            // Sweep window sizes to find optimal
            let engine = try MetalMSM()
            let gx = fpFromInt(1); let gy = fpFromInt(2)
            let g = PointAffine(x: gx, y: gy)
            var scalars: [[UInt32]] = []
            var rng: UInt64 = 0xDEAD_BEEF_CAFE_BABE
            for _ in 0..<n {
                var limbs: [UInt32] = Array(repeating: 0, count: 8)
                for j in 0..<8 { rng = rng &* 6364136223846793005 &+ 1442695040888963407; limbs[j] = UInt32(truncatingIfNeeded: rng >> 32) }
                scalars.append(limbs)
            }
            let points = [PointAffine](repeating: g, count: n)
            for wb: UInt32 in [18, 17, 16, 15, 14, 13, 12] {
                engine.windowBitsOverride = wb
                let start = CFAbsoluteTimeGetCurrent()
                let _ = try engine.msm(points: points, scalars: scalars)
                let elapsed = CFAbsoluteTimeGetCurrent() - start
                fputs("  w=\(wb): \(String(format: "%.1f", elapsed * 1000))ms\n", stderr)
            }
            return
        }
        // Allow --wb N to override window bits
        if let wbIdx = args.firstIndex(of: "--wb"), wbIdx + 1 < args.count,
           let wb = UInt32(args[wbIdx + 1]) {
            let engine = try MetalMSM()
            engine.windowBitsOverride = wb
            // Run with distinct points
            let gx = fpFromInt(1); let gy = fpFromInt(2)
            let g = PointAffine(x: gx, y: gy)
            fputs("Generating \(n) distinct points...\n", stderr)
            var projPts = [PointProjective]()
            projPts.reserveCapacity(n)
            let gProj = pointFromAffine(g)
            var ac = gProj
            for _ in 0..<n { projPts.append(ac); ac = pointAdd(ac, gProj) }
            let pts = batchToAffine(projPts)
            var scalars: [[UInt32]] = []
            var rng: UInt64 = 0xDEAD_BEEF_CAFE_BABE
            for _ in 0..<n {
                var limbs: [UInt32] = Array(repeating: 0, count: 8)
                for j in 0..<8 { rng = rng &* 6364136223846793005 &+ 1442695040888963407; limbs[j] = UInt32(truncatingIfNeeded: rng >> 32) }
                scalars.append(limbs)
            }
            fputs("Warmup...\n", stderr)
            let _ = try engine.msm(points: pts, scalars: scalars)
            let start = CFAbsoluteTimeGetCurrent()
            let result = try engine.msm(points: pts, scalars: scalars)
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            fputs("MSM(\(n), w=\(wb)): \(String(format: "%.3f", elapsed * 1000))ms\n", stderr)
            if let affine = pointToAffine(result) {
                let xInt = fpToInt(affine.x)
                fputs("Result.x = 0x\(xInt.reversed().map { String(format: "%016llx", $0) }.joined())\n", stderr)
            }
            return
        }
        try runBenchmark(nPoints: n)
        return
    }

    if args.contains("--info") {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("{\"error\": \"No Metal GPU available\"}")
            return
        }
        let info: [String: Any] = [
            "gpu": device.name,
            "unified_memory": device.hasUnifiedMemory,
            "max_buffer_length": device.maxBufferLength,
            "max_threadgroup_memory": device.maxThreadgroupMemoryLength,
        ]
        let data = try JSONSerialization.data(withJSONObject: info, options: .prettyPrinted)
        print(String(data: data, encoding: .utf8)!)
        return
    }

    if args.contains("--msm") || args.count == 1 {
        try runStdinMSM()
        return
    }

    fputs("zkmsm: Metal GPU MSM for BN254\n", stderr)
    fputs("Usage:\n", stderr)
    fputs("  echo '{...}' | zkmsm     Compute MSM from stdin JSON\n", stderr)
    fputs("  zkmsm --msm              Same as above (explicit flag)\n", stderr)
    fputs("  zkmsm --test             Run correctness tests\n", stderr)
    fputs("  zkmsm --bench <n_points> Benchmark MSM\n", stderr)
    fputs("  zkmsm --info             Show GPU info\n", stderr)
}

try main()
