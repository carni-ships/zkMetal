import zkMetal
import Foundation

public func runBinaryTowerTests() {
    // Initialize GF(2^8) lookup tables (required before any NEON binary tower ops)
    BinaryTowerNeon.initialize()

    suite("BinaryTower8 — GF(2^8)")
    testBT8BasicAxioms()
    testBT8AESSBoxVectors()
    testBT8Exhaustive()

    suite("BinaryTower16 — GF(2^16)")
    testBT16FieldAxioms()

    suite("BinaryTower32 — GF(2^32)")
    testBT32FieldAxioms()

    suite("BinaryTower64 — GF(2^64)")
    testBT64FieldAxioms()

    suite("BinaryTower128 — GF(2^128)")
    testBT128FieldAxioms()

    suite("BinaryTower Cross-Level Embedding")
    testCrossLevelEmbedding()

    suite("BinaryTower Square-and-Multiply")
    testSquareAndMultiply()

    suite("BinaryTower128 Performance (1M muls)")
    benchmarkBT128Mul()
}

// MARK: - GF(2^8) Tests

private func testBT8BasicAxioms() {
    let a = BinaryTower8(value: 0x53)
    let b = BinaryTower8(value: 0xCA)
    let c = BinaryTower8(value: 0x3F)

    // Commutativity of addition
    expectEqual(a + b, b + a, "BT8 add commutative")

    // Commutativity of multiplication
    expectEqual(a * b, b * a, "BT8 mul commutative")

    // Associativity of addition
    expectEqual((a + b) + c, a + (b + c), "BT8 add associative")

    // Associativity of multiplication
    expectEqual((a * b) * c, a * (b * c), "BT8 mul associative")

    // Distributivity
    expectEqual(a * (b + c), (a * b) + (a * c), "BT8 distributive")

    // Identity elements
    expectEqual(a + BinaryTower8.zero, a, "BT8 add identity")
    expectEqual(a * BinaryTower8.one, a, "BT8 mul identity")

    // Self-inverse in addition (char 2)
    expectEqual(a + a, BinaryTower8.zero, "BT8 add self-inverse")

    // Multiplicative inverse
    let aInv = a.inverse()
    expectEqual(a * aInv, BinaryTower8.one, "BT8 mul inverse")

    let bInv = b.inverse()
    expectEqual(b * bInv, BinaryTower8.one, "BT8 mul inverse b")
}

/// AES S-box known test vectors for GF(2^8) inverse.
/// In AES, the S-box is computed as affine(inverse(byte)) where inverse is in GF(2^8)/0x11B.
/// We test the raw GF(2^8) inverse against known values.
private func testBT8AESSBoxVectors() {
    // Known GF(2^8) inverses under polynomial x^8+x^4+x^3+x+1
    // inv(1) = 1, inv(2) = 0x8D, inv(3) = 0xF6, inv(0x53) = 0xCA
    let inv1 = BinaryTower8(value: 1).inverse()
    expectEqual(inv1.value, 1, "BT8 inv(1) = 1")

    let inv2 = BinaryTower8(value: 2).inverse()
    expectEqual(inv2.value, 0x8D, "BT8 inv(2) = 0x8D")

    let inv3 = BinaryTower8(value: 3).inverse()
    expectEqual(inv3.value, 0xF6, "BT8 inv(3) = 0xF6")

    // Verify round-trip: inv(inv(x)) = x
    let inv_inv2 = inv2.inverse()
    expectEqual(inv_inv2.value, 2, "BT8 inv(inv(2)) = 2")

    // inv(0x53) via Fermat: 0x53^254
    let val = BinaryTower8(value: 0x53)
    let valInv = val.inverse()
    expectEqual((val * valInv).value, 1, "BT8 0x53 * inv(0x53) = 1")
}

/// Exhaustive test: verify a * inv(a) = 1 for all nonzero elements of GF(2^8)
private func testBT8Exhaustive() {
    var allPass = true
    for i in 1..<256 {
        let a = BinaryTower8(value: UInt8(i))
        let aInv = a.inverse()
        if (a * aInv) != BinaryTower8.one {
            allPass = false
            break
        }
    }
    expect(allPass, "BT8 exhaustive inverse check (255 elements)")
}

// MARK: - GF(2^16) Tests

private func testBT16FieldAxioms() {
    let a = BinaryTower16(value: 0xABCD)
    let b = BinaryTower16(value: 0x1234)
    let c = BinaryTower16(value: 0x5678)

    // Commutativity
    expectEqual(a + b, b + a, "BT16 add commutative")
    expectEqual(a * b, b * a, "BT16 mul commutative")

    // Associativity
    expectEqual((a + b) + c, a + (b + c), "BT16 add associative")
    expectEqual((a * b) * c, a * (b * c), "BT16 mul associative")

    // Distributivity
    expectEqual(a * (b + c), (a * b) + (a * c), "BT16 distributive")

    // Identity
    expectEqual(a + BinaryTower16.zero, a, "BT16 add identity")
    expectEqual(a * BinaryTower16.one, a, "BT16 mul identity")

    // Char 2
    expectEqual(a + a, BinaryTower16.zero, "BT16 add self-inverse")

    // Inverse
    let aInv = a.inverse()
    expectEqual(a * aInv, BinaryTower16.one, "BT16 mul inverse a")

    let bInv = b.inverse()
    expectEqual(b * bInv, BinaryTower16.one, "BT16 mul inverse b")

    // Several random-ish values
    for val: UInt16 in [0x0001, 0x0002, 0x00FF, 0xFF00, 0xFFFF, 0x8000, 0x0100] {
        let x = BinaryTower16(value: val)
        let xInv = x.inverse()
        expectEqual(x * xInv, BinaryTower16.one, "BT16 inv round-trip 0x\(String(val, radix: 16))")
    }
}

// MARK: - GF(2^32) Tests

private func testBT32FieldAxioms() {
    let a = BinaryTower32(value: 0xDEADBEEF)
    let b = BinaryTower32(value: 0x12345678)
    let c = BinaryTower32(value: 0xCAFEBABE)

    // Commutativity
    expectEqual(a + b, b + a, "BT32 add commutative")
    expectEqual(a * b, b * a, "BT32 mul commutative")

    // Associativity
    expectEqual((a + b) + c, a + (b + c), "BT32 add associative")
    expectEqual((a * b) * c, a * (b * c), "BT32 mul associative")

    // Distributivity
    expectEqual(a * (b + c), (a * b) + (a * c), "BT32 distributive")

    // Identity
    expectEqual(a * BinaryTower32.one, a, "BT32 mul identity")

    // Char 2
    expectEqual(a + a, BinaryTower32.zero, "BT32 self-inverse")

    // Inverse
    let aInv = a.inverse()
    expectEqual(a * aInv, BinaryTower32.one, "BT32 mul inverse a")

    let bInv = b.inverse()
    expectEqual(b * bInv, BinaryTower32.one, "BT32 mul inverse b")
}

// MARK: - GF(2^64) Tests

private func testBT64FieldAxioms() {
    let a = BinaryTower64(value: 0xDEADBEEFCAFEBABE)
    let b = BinaryTower64(value: 0x123456789ABCDEF0)
    let c = BinaryTower64(value: 0xFEDCBA9876543210)

    // Commutativity
    expectEqual(a + b, b + a, "BT64 add commutative")
    expectEqual(a * b, b * a, "BT64 mul commutative")

    // Associativity
    expectEqual((a + b) + c, a + (b + c), "BT64 add associative")
    expectEqual((a * b) * c, a * (b * c), "BT64 mul associative")

    // Distributivity
    expectEqual(a * (b + c), (a * b) + (a * c), "BT64 distributive")

    // Identity
    expectEqual(a * BinaryTower64.one, a, "BT64 mul identity")
    expectEqual(a + BinaryTower64.zero, a, "BT64 add identity")

    // Char 2
    expectEqual(a + a, BinaryTower64.zero, "BT64 self-inverse")

    // Inverse
    let aInv = a.inverse()
    expectEqual(a * aInv, BinaryTower64.one, "BT64 mul inverse a")

    let bInv = b.inverse()
    expectEqual(b * bInv, BinaryTower64.one, "BT64 mul inverse b")
}

// MARK: - GF(2^128) Tests

private func testBT128FieldAxioms() {
    let a = BinaryTower128(lo: 0xDEADBEEFCAFEBABE, hi: 0x1234567890ABCDEF)
    let b = BinaryTower128(lo: 0x123456789ABCDEF0, hi: 0xFEDCBA9876543210)
    let c = BinaryTower128(lo: 0xAAAABBBBCCCCDDDD, hi: 0x1111222233334444)

    // Commutativity
    expectEqual(a + b, b + a, "BT128 add commutative")
    expectEqual(a * b, b * a, "BT128 mul commutative")

    // Associativity
    expectEqual((a + b) + c, a + (b + c), "BT128 add associative")
    expectEqual((a * b) * c, a * (b * c), "BT128 mul associative")

    // Distributivity
    expectEqual(a * (b + c), (a * b) + (a * c), "BT128 distributive")

    // Identity
    expectEqual(a * BinaryTower128.one, a, "BT128 mul identity")
    expectEqual(a + BinaryTower128.zero, a, "BT128 add identity")

    // Char 2
    expectEqual(a + a, BinaryTower128.zero, "BT128 self-inverse")

    // Inverse
    let aInv = a.inverse()
    expectEqual(a * aInv, BinaryTower128.one, "BT128 mul inverse a")

    let bInv = b.inverse()
    expectEqual(b * bInv, BinaryTower128.one, "BT128 mul inverse b")

    // Edge case: inverse of 1
    let oneInv = BinaryTower128.one.inverse()
    expectEqual(oneInv, BinaryTower128.one, "BT128 inv(1) = 1")
}

// MARK: - Cross-Level Embedding Tests

/// Test that the tower-form types (BinaryField8/16/32/64/128 from BinaryTower.swift)
/// correctly embed GF(2^8) into higher levels via the Karatsuba tower construction.
/// Also test that BinaryTower types (flat polynomial) are self-consistent.
private func testCrossLevelEmbedding() {
    // Test tower-form embedding: BinaryField8 -> BinaryField16 -> ... -> BinaryField128
    // These use X^2+X+alpha tower extensions, so GF(2^8) embeds canonically.
    for aVal: UInt8 in [0x01, 0x02, 0x03, 0x53, 0xCA, 0xFF] {
        for bVal: UInt8 in [0x01, 0x02, 0x03, 0x53, 0xCA, 0xFF] {
            let gf8Result = BinaryField8(value: aVal) * BinaryField8(value: bVal)

            // Tower GF(2^16): embed GF(2^8) as (lo=val, hi=0)
            let a16 = BinaryField16(lo: BinaryField8(value: aVal), hi: .zero)
            let b16 = BinaryField16(lo: BinaryField8(value: bVal), hi: .zero)
            let r16 = a16 * b16
            expectEqual(r16.lo.value, gf8Result.value,
                        "Tower GF8->GF16 embed \(aVal)*\(bVal)")
            expect(r16.hi.isZero, "Tower GF8 in GF16 stays in subfield")

            // Tower GF(2^32): embed GF(2^16) which embeds GF(2^8)
            let a32 = BinaryField32(lo: a16, hi: .zero)
            let b32 = BinaryField32(lo: b16, hi: .zero)
            let r32 = a32 * b32
            expectEqual(r32.lo.lo.value, gf8Result.value,
                        "Tower GF8->GF32 embed \(aVal)*\(bVal)")
            expect(r32.hi.isZero, "Tower GF8 in GF32 stays in subfield")
        }
    }

    // Test flat BinaryTower types: verify 1 is identity, 0 is absorbing
    let one128 = BinaryTower128.one
    let zero128 = BinaryTower128.zero
    let x128 = BinaryTower128(lo: 0xDEADBEEFCAFEBABE, hi: 0x1234567890ABCDEF)
    expectEqual(x128 * one128, x128, "BT128 mul by 1 = identity")
    expectEqual(x128 * zero128, zero128, "BT128 mul by 0 = 0")
    expectEqual(x128 + zero128, x128, "BT128 add 0 = identity")

    let one64 = BinaryTower64.one
    let x64 = BinaryTower64(value: 0xCAFEBABE12345678)
    expectEqual(x64 * one64, x64, "BT64 mul by 1 = identity")

    // Verify flat GF(2^8) self-consistency (this always works since it's the same field)
    for aVal: UInt8 in [0x01, 0x53, 0xFF] {
        let bt8 = BinaryTower8(value: aVal)
        let bf8 = BinaryField8(value: aVal)
        let bt8_sq = bt8.squared()
        let bf8_sq = bf8.squared()
        expectEqual(bt8_sq.value, bf8_sq.value, "BT8 vs BF8 square consistency \(aVal)")
    }
}

// MARK: - Square-and-Multiply Tests

private func testSquareAndMultiply() {
    // GF(2^8): a^255 = 1 for all nonzero a (Fermat)
    let a8 = BinaryTower8(value: 0x53)
    expectEqual(a8.pow(255), BinaryTower8.one, "BT8 a^255 = 1 (Fermat)")
    expectEqual(a8.pow(0), BinaryTower8.one, "BT8 a^0 = 1")
    expectEqual(a8.pow(1), a8, "BT8 a^1 = a")
    expectEqual(a8.pow(2), a8.squared(), "BT8 a^2 = a.squared()")

    // GF(2^16): a^(2^16-1) = 1
    let a16 = BinaryTower16(value: 0xABCD)
    expectEqual(a16.pow(65535), BinaryTower16.one, "BT16 a^(2^16-1) = 1")

    // GF(2^32): verify pow consistency
    let a32 = BinaryTower32(value: 0xDEADBEEF)
    let a32_sq = a32.squared()
    let a32_pow2 = a32.pow(2)
    expectEqual(a32_sq, a32_pow2, "BT32 squared() == pow(2)")
    let a32_cube = a32 * a32 * a32
    expectEqual(a32.pow(3), a32_cube, "BT32 pow(3) == a*a*a")

    // GF(2^64): verify pow consistency
    let a64 = BinaryTower64(value: 0xCAFEBABEDEADBEEF)
    expectEqual(a64.pow(2), a64.squared(), "BT64 pow(2) == squared()")
    let a64_4 = a64.squared().squared()
    expectEqual(a64.pow(4), a64_4, "BT64 pow(4)")

    // GF(2^128): verify pow consistency
    let a128 = BinaryTower128(lo: 0xDEADBEEF, hi: 0xCAFEBABE)
    expectEqual(a128.pow(2), a128.squared(), "BT128 pow(2) == squared()")
    expectEqual(a128.pow(1), a128, "BT128 pow(1) == a")
    expectEqual(a128.pow(0), BinaryTower128.one, "BT128 pow(0) == 1")
}

// MARK: - Performance Benchmark

private func benchmarkBT128Mul() {
    let count = 1_000_000
    var a = BinaryTower128(lo: 0xDEADBEEFCAFEBABE, hi: 0x1234567890ABCDEF)
    let b = BinaryTower128(lo: 0x123456789ABCDEF0, hi: 0xFEDCBA9876543210)

    let start = CFAbsoluteTimeGetCurrent()
    for _ in 0..<count {
        a = a * b
    }
    let elapsed = CFAbsoluteTimeGetCurrent() - start
    let nsPerMul = elapsed * 1e9 / Double(count)

    // Prevent dead-code elimination
    expect(!a.isZero || a.isZero, "BT128 benchmark anti-DCE")
    print("  BT128 1M muls: \(String(format: "%.1f", elapsed * 1000))ms (\(String(format: "%.1f", nsPerMul))ns/mul)")
}
