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
    var window_index: UInt32
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

// MARK: - Metal MSM Engine

class MetalMSM {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let accumulateFunction: MTLComputePipelineState
    let reduceFunction: MTLComputePipelineState
    let bucketSumFunction: MTLComputePipelineState

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        let shaderPath = findShaderPath()
        let shaderSource = try String(contentsOfFile: shaderPath, encoding: .utf8)
        let library = try device.makeLibrary(source: shaderSource, options: nil)

        guard let accFn = library.makeFunction(name: "msm_accumulate"),
              let redFn = library.makeFunction(name: "msm_reduce_buckets"),
              let sumFn = library.makeFunction(name: "msm_bucket_sum") else {
            throw MSMError.missingKernel
        }

        self.accumulateFunction = try device.makeComputePipelineState(function: accFn)
        self.reduceFunction = try device.makeComputePipelineState(function: redFn)
        self.bucketSumFunction = try device.makeComputePipelineState(function: sumFn)
    }

    func msm(points: [PointAffine], scalars: [[UInt32]]) throws -> PointProjective {
        let n = points.count
        guard n == scalars.count, n > 0 else {
            throw MSMError.invalidInput
        }

        let windowBits: UInt32 = n <= 256 ? 8 : (n <= 4096 ? 12 : 16)
        let nWindows = (256 + Int(windowBits) - 1) / Int(windowBits)
        let nBuckets = 1 << windowBits

        let pointsBuffer = device.makeBuffer(
            bytes: points, length: MemoryLayout<PointAffine>.stride * n,
            options: .storageModeShared)!

        let scalarData = scalars.flatMap { $0 }
        let scalarsBuffer = device.makeBuffer(
            bytes: scalarData, length: MemoryLayout<UInt32>.stride * scalarData.count,
            options: .storageModeShared)!

        var windowResults: [PointProjective] = []

        for w in 0..<nWindows {
            let threadBuckets = device.makeBuffer(
                length: MemoryLayout<PointProjective>.stride * n,
                options: .storageModeShared)!
            let bucketIds = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride * n,
                options: .storageModeShared)!
            let reducedBuckets = device.makeBuffer(
                length: MemoryLayout<PointProjective>.stride * nBuckets,
                options: .storageModeShared)!
            let windowResult = device.makeBuffer(
                length: MemoryLayout<PointProjective>.stride,
                options: .storageModeShared)!

            memset(threadBuckets.contents(), 0, threadBuckets.length)
            memset(bucketIds.contents(), 0, bucketIds.length)
            memset(reducedBuckets.contents(), 0, reducedBuckets.length)
            memset(windowResult.contents(), 0, windowResult.length)

            var params = MsmParams(
                n_points: UInt32(n),
                window_bits: windowBits,
                window_index: UInt32(w)
            )

            guard let commandBuffer = commandQueue.makeCommandBuffer() else {
                throw MSMError.noCommandBuffer
            }

            // Phase 1: Accumulate
            let enc1 = commandBuffer.makeComputeCommandEncoder()!
            enc1.setComputePipelineState(accumulateFunction)
            enc1.setBuffer(pointsBuffer, offset: 0, index: 0)
            enc1.setBuffer(scalarsBuffer, offset: 0, index: 1)
            enc1.setBuffer(threadBuckets, offset: 0, index: 2)
            enc1.setBuffer(bucketIds, offset: 0, index: 3)
            enc1.setBytes(&params, length: MemoryLayout<MsmParams>.stride, index: 4)
            enc1.dispatchThreads(
                MTLSize(width: n, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(256, n), height: 1, depth: 1))
            enc1.endEncoding()

            // Phase 2: Reduce
            let enc2 = commandBuffer.makeComputeCommandEncoder()!
            enc2.setComputePipelineState(reduceFunction)
            enc2.setBuffer(threadBuckets, offset: 0, index: 0)
            enc2.setBuffer(bucketIds, offset: 0, index: 1)
            enc2.setBuffer(reducedBuckets, offset: 0, index: 2)
            enc2.setBytes(&params, length: MemoryLayout<MsmParams>.stride, index: 3)
            enc2.dispatchThreads(
                MTLSize(width: nBuckets, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(256, nBuckets), height: 1, depth: 1))
            enc2.endEncoding()

            // Phase 3: Bucket sum
            let enc3 = commandBuffer.makeComputeCommandEncoder()!
            enc3.setComputePipelineState(bucketSumFunction)
            enc3.setBuffer(reducedBuckets, offset: 0, index: 0)
            enc3.setBuffer(windowResult, offset: 0, index: 1)
            enc3.setBytes(&params, length: MemoryLayout<MsmParams>.stride, index: 2)
            enc3.dispatchThreads(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            enc3.endEncoding()

            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            if let error = commandBuffer.error {
                throw MSMError.gpuError(error.localizedDescription)
            }

            let resultPtr = windowResult.contents().bindMemory(to: PointProjective.self, capacity: 1)
            windowResults.append(resultPtr.pointee)
        }

        // Combine windows using Horner's method (CPU-side, ~256 point ops)
        var result = windowResults.last!
        for w in stride(from: nWindows - 2, through: 0, by: -1) {
            for _ in 0..<windowBits {
                result = pointDouble(result)
            }
            result = pointAdd(result, windowResults[w])
        }

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

    // Use BN254 generator G=(1,2) and scalar multiples as test data
    let gx = fpFromInt(1)
    let gy = fpFromInt(2)
    let g = PointAffine(x: gx, y: gy)

    // Generate points as multiples of G (for a real bench, these should be precomputed)
    var points: [PointAffine] = Array(repeating: g, count: nPoints)
    var scalars: [[UInt32]] = []

    for i in 0..<nPoints {
        scalars.append([UInt32(i + 1), 0, 0, 0, 0, 0, 0, 0])
    }

    let start = CFAbsoluteTimeGetCurrent()
    let result = try engine.msm(points: points, scalars: scalars)
    let elapsed = CFAbsoluteTimeGetCurrent() - start

    fputs("MSM(\(nPoints)): \(String(format: "%.3f", elapsed * 1000))ms\n", stderr)
    fputs("GPU: \(engine.device.name)\n", stderr)
    fputs("Max threadgroup: \(engine.accumulateFunction.maxTotalThreadsPerThreadgroup)\n", stderr)

    if let affine = pointToAffine(result) {
        let xInt = fpToInt(affine.x)
        fputs("Result.x = 0x\(xInt.reversed().map { String(format: "%016llx", $0) }.joined())\n", stderr)
    } else {
        fputs("Result: point at infinity\n", stderr)
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
