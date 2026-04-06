// Packed NEON small-field arithmetic tests
// Tests packed BabyBear and M31 add/sub/mul, butterfly ops, and NTT round-trip

import Foundation
import NeonFieldOps

public func runPackedNEONTests() {
    suite("Packed NEON -- BabyBear NTT Round-trip")
    testPackedBBNTTSize4()
    testPackedBBNTTSize8()
    testPackedBBNTTSize256()
    testPackedBBNTTEdgeCases()

    suite("Packed NEON -- BabyBear NTT Convolution")
    testPackedBBNTTConvolution()

    suite("Packed NEON -- BabyBear NTT vs Existing")
    testPackedBBNTTMatchesExisting()

    suite("Packed NEON -- M31 Batch Ops")
    testPackedM31BatchAdd()
    testPackedM31BatchSub()
    testPackedM31BatchMul()
    testPackedM31EdgeCases()

    suite("Packed NEON -- M31 Butterfly")
    testPackedM31ButterflyDIT()
    testPackedM31ButterflyDIF()
}

// MARK: - Constants

private let BB_P: UInt32 = 2013265921
private let M31_P: UInt32 = 0x7FFFFFFF

// Scalar reference
private func bbMulRef(_ a: UInt32, _ b: UInt32) -> UInt32 {
    UInt32(UInt64(a) * UInt64(b) % UInt64(BB_P))
}
private func bbAddRef(_ a: UInt32, _ b: UInt32) -> UInt32 {
    let s = UInt64(a) + UInt64(b)
    return s >= UInt64(BB_P) ? UInt32(s - UInt64(BB_P)) : UInt32(s)
}
private func m31AddRef(_ a: UInt32, _ b: UInt32) -> UInt32 {
    let s = UInt64(a) + UInt64(b)
    var r = UInt32(s >> 31) + UInt32(s & UInt64(M31_P))
    if r >= M31_P { r -= M31_P }
    return r
}
private func m31SubRef(_ a: UInt32, _ b: UInt32) -> UInt32 {
    a >= b ? a - b : a + M31_P - b
}
private func m31MulRef(_ a: UInt32, _ b: UInt32) -> UInt32 {
    let prod = UInt64(a) * UInt64(b)
    let lo = UInt32(prod & UInt64(M31_P))
    let hi = UInt32(prod >> 31)
    var r = lo + hi
    r = (r >> 31) + (r & M31_P)
    if r >= M31_P { r -= M31_P }
    return r
}

// MARK: - BabyBear NTT Tests

func testPackedBBNTTSize4() {
    var data: [UInt32] = [10, 20, 30, 40]
    let orig = data
    data.withUnsafeMutableBufferPointer { ptr in
        packed_bb_ntt(ptr.baseAddress!, 2)
        packed_bb_intt(ptr.baseAddress!, 2)
    }
    for i in 0..<4 {
        expectEqual(data[i], orig[i], "NTT4 round-trip[\(i)]")
    }
}

func testPackedBBNTTSize8() {
    var data: [UInt32] = [1, 2, 3, 4, 5, 6, 7, 8]
    let orig = data
    data.withUnsafeMutableBufferPointer { ptr in
        packed_bb_ntt(ptr.baseAddress!, 3)
        packed_bb_intt(ptr.baseAddress!, 3)
    }
    for i in 0..<8 {
        expectEqual(data[i], orig[i], "NTT8 round-trip[\(i)]")
    }
}

func testPackedBBNTTSize256() {
    let logN: Int32 = 8
    let n = 1 << logN
    var data = (0..<n).map { UInt32(($0 * 17 + 3) % Int(BB_P)) }
    let orig = data
    data.withUnsafeMutableBufferPointer { ptr in
        packed_bb_ntt(ptr.baseAddress!, logN)
        packed_bb_intt(ptr.baseAddress!, logN)
    }
    var allMatch = true
    for i in 0..<n {
        if data[i] != orig[i] { allMatch = false; break }
    }
    expect(allMatch, "NTT256 round-trip")
}

func testPackedBBNTTEdgeCases() {
    // All zeros
    var zeros = [UInt32](repeating: 0, count: 8)
    zeros.withUnsafeMutableBufferPointer { ptr in
        packed_bb_ntt(ptr.baseAddress!, 3)
        packed_bb_intt(ptr.baseAddress!, 3)
    }
    expect(zeros.allSatisfy { $0 == 0 }, "NTT zeros round-trip")

    // Values near p
    var near: [UInt32] = (1...8).map { BB_P - $0 }
    let nearOrig = near
    near.withUnsafeMutableBufferPointer { ptr in
        packed_bb_ntt(ptr.baseAddress!, 3)
        packed_bb_intt(ptr.baseAddress!, 3)
    }
    expect(near == nearOrig, "NTT near-p round-trip")

    // Size 2
    var two: [UInt32] = [100, 200]
    let twoOrig = two
    two.withUnsafeMutableBufferPointer { ptr in
        packed_bb_ntt(ptr.baseAddress!, 1)
        packed_bb_intt(ptr.baseAddress!, 1)
    }
    expect(two == twoOrig, "NTT2 round-trip")
}

func testPackedBBNTTConvolution() {
    let logN: Int32 = 4
    let n = 1 << logN

    // f(x) = 1 + 2x + 3x^2, g(x) = 4 + 5x
    var f = [UInt32](repeating: 0, count: n)
    f[0] = 1; f[1] = 2; f[2] = 3
    var g = [UInt32](repeating: 0, count: n)
    g[0] = 4; g[1] = 5

    f.withUnsafeMutableBufferPointer { ptr in packed_bb_ntt(ptr.baseAddress!, logN) }
    g.withUnsafeMutableBufferPointer { ptr in packed_bb_ntt(ptr.baseAddress!, logN) }

    var h = (0..<n).map { bbMulRef(f[$0], g[$0]) }
    h.withUnsafeMutableBufferPointer { ptr in packed_bb_intt(ptr.baseAddress!, logN) }

    // f*g = 4 + 13x + 22x^2 + 15x^3
    expectEqual(h[0], 4, "conv[0]=4")
    expectEqual(h[1], 13, "conv[1]=13")
    expectEqual(h[2], 22, "conv[2]=22")
    expectEqual(h[3], 15, "conv[3]=15")
    expect((4..<n).allSatisfy { h[$0] == 0 }, "conv tail zeros")
}

func testPackedBBNTTMatchesExisting() {
    let logN: Int32 = 8
    let n = 1 << logN
    var data1 = (0..<n).map { UInt32(($0 * 13 + 5) % Int(BB_P)) }
    var data2 = data1

    data1.withUnsafeMutableBufferPointer { ptr in babybear_ntt_neon(ptr.baseAddress!, logN) }
    data2.withUnsafeMutableBufferPointer { ptr in packed_bb_ntt(ptr.baseAddress!, logN) }

    expect(data1 == data2, "packed_bb_ntt matches babybear_ntt_neon (n=256)")
}

// MARK: - M31 Tests

func testPackedM31BatchAdd() {
    var a: [UInt32] = [100, M31_P - 1, 0, M31_P / 2]
    var b: [UInt32] = [200, 1, 0, M31_P / 2]
    var result = [UInt32](repeating: 0, count: 4)
    a.withUnsafeBufferPointer { ap in
        b.withUnsafeBufferPointer { bp in
            result.withUnsafeMutableBufferPointer { rp in
                m31_batch_add_neon(ap.baseAddress!, bp.baseAddress!, rp.baseAddress!, 4)
            }
        }
    }
    for i in 0..<4 {
        expectEqual(result[i], m31AddRef(a[i], b[i]), "M31 add[\(i)]")
    }
}

func testPackedM31BatchSub() {
    var a: [UInt32] = [100, 0, M31_P - 1, 500]
    var b: [UInt32] = [50, 1, M31_P - 2, 1000]
    var result = [UInt32](repeating: 0, count: 4)
    a.withUnsafeBufferPointer { ap in
        b.withUnsafeBufferPointer { bp in
            result.withUnsafeMutableBufferPointer { rp in
                m31_batch_sub_neon(ap.baseAddress!, bp.baseAddress!, rp.baseAddress!, 4)
            }
        }
    }
    for i in 0..<4 {
        expectEqual(result[i], m31SubRef(a[i], b[i]), "M31 sub[\(i)]")
    }
}

func testPackedM31BatchMul() {
    var a: [UInt32] = [100, M31_P - 1, 12345, 0]
    var b: [UInt32] = [200, 2, 67890, 999]
    var result = [UInt32](repeating: 0, count: 4)
    a.withUnsafeBufferPointer { ap in
        b.withUnsafeBufferPointer { bp in
            result.withUnsafeMutableBufferPointer { rp in
                m31_batch_mul_neon(ap.baseAddress!, bp.baseAddress!, rp.baseAddress!, 4)
            }
        }
    }
    for i in 0..<4 {
        expectEqual(result[i], m31MulRef(a[i], b[i]), "M31 mul[\(i)]")
    }
}

func testPackedM31EdgeCases() {
    var a: [UInt32] = [0, 0, 0, 0]
    var b: [UInt32] = [1, M31_P - 1, 12345, M31_P / 2]
    var result = [UInt32](repeating: 0, count: 4)
    a.withUnsafeBufferPointer { ap in
        b.withUnsafeBufferPointer { bp in
            result.withUnsafeMutableBufferPointer { rp in
                m31_batch_mul_neon(ap.baseAddress!, bp.baseAddress!, rp.baseAddress!, 4)
            }
        }
    }
    expect(result.allSatisfy { $0 == 0 }, "M31 zero*x = 0")

    // 1 * x = x
    a = [1, 1, 1, 1]
    a.withUnsafeBufferPointer { ap in
        b.withUnsafeBufferPointer { bp in
            result.withUnsafeMutableBufferPointer { rp in
                m31_batch_mul_neon(ap.baseAddress!, bp.baseAddress!, rp.baseAddress!, 4)
            }
        }
    }
    for i in 0..<4 {
        expectEqual(result[i], b[i], "M31 1*x = x[\(i)]")
    }

    // a + (p-a) = 0
    a = [1, 100, M31_P / 2, M31_P - 1]
    var neg = a.map { M31_P - $0 }
    a.withUnsafeBufferPointer { ap in
        neg.withUnsafeBufferPointer { np in
            result.withUnsafeMutableBufferPointer { rp in
                m31_batch_add_neon(ap.baseAddress!, np.baseAddress!, rp.baseAddress!, 4)
            }
        }
    }
    expect(result.allSatisfy { $0 == 0 }, "M31 a + (-a) = 0")
}

func testPackedM31ButterflyDIT() {
    var data: [UInt32] = [10, 20, 30, 40, 50, 60, 70, 80]
    let tw: [UInt32] = [1, 1, 1, 1]
    data.withUnsafeMutableBufferPointer { dp in
        tw.withUnsafeBufferPointer { tp in
            packed_m31_butterfly_dit(dp.baseAddress!, 4, 1, tp.baseAddress!)
        }
    }
    // DIT w=1: (u,v) -> (u + 1*v, u - 1*v)
    expectEqual(data[0], m31AddRef(10, 50), "M31 DIT[0]")
    expectEqual(data[1], m31AddRef(20, 60), "M31 DIT[1]")
    expectEqual(data[2], m31AddRef(30, 70), "M31 DIT[2]")
    expectEqual(data[3], m31AddRef(40, 80), "M31 DIT[3]")
    expectEqual(data[4], m31SubRef(10, 50), "M31 DIT[4]")
    expectEqual(data[5], m31SubRef(20, 60), "M31 DIT[5]")
    expectEqual(data[6], m31SubRef(30, 70), "M31 DIT[6]")
    expectEqual(data[7], m31SubRef(40, 80), "M31 DIT[7]")
}

func testPackedM31ButterflyDIF() {
    var data: [UInt32] = [10, 20, 30, 40, 50, 60, 70, 80]
    let tw: [UInt32] = [1, 1, 1, 1]
    data.withUnsafeMutableBufferPointer { dp in
        tw.withUnsafeBufferPointer { tp in
            packed_m31_butterfly_dif(dp.baseAddress!, 4, 1, tp.baseAddress!)
        }
    }
    // DIF w=1: (a,b) -> (a+b, (a-b)*1)
    expectEqual(data[0], m31AddRef(10, 50), "M31 DIF sum[0]")
    expectEqual(data[1], m31AddRef(20, 60), "M31 DIF sum[1]")
    expectEqual(data[2], m31AddRef(30, 70), "M31 DIF sum[2]")
    expectEqual(data[3], m31AddRef(40, 80), "M31 DIF sum[3]")
    expectEqual(data[4], m31SubRef(10, 50), "M31 DIF diff[0]")
    expectEqual(data[5], m31SubRef(20, 60), "M31 DIF diff[1]")
    expectEqual(data[6], m31SubRef(30, 70), "M31 DIF diff[2]")
    expectEqual(data[7], m31SubRef(40, 80), "M31 DIF diff[3]")
}
