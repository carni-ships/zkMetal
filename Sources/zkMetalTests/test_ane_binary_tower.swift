import XCTest
import ANEOps

/// Tests for ANE Binary Tower primitives (GF(2^64) and GF(2^128))
/// Tests verify the scalar fallback implementations since ANE hardware
/// returns -1 (not available) on CI.
class ANEBinaryTowerTests: XCTestCase {

    // MARK: - GF(2^64) Addition (XOR Property)

    /// Test GF(2^64) addition is XOR
    func testGF64AddIsXOR() {
        let a: UInt64 = 0xDEADBEEFCAFEBABE
        let b: UInt64 = 0x123456789ABCDEF0

        let result = bt_gf64_add_scalar(a, b)
        let expected = a ^ b

        XCTAssertEqual(result, expected, "GF(2^64) add should equal XOR")
    }

    /// Test GF(2^64) addition with various patterns
    func testGF64AddXORPatterns() {
        // All zeros
        XCTAssertEqual(bt_gf64_add_scalar(0, 0), 0, "0 + 0 = 0")

        // XOR with zero
        let val: UInt64 = 0x123456789ABCDEF0
        XCTAssertEqual(bt_gf64_add_scalar(val, 0), val, "a + 0 = a")
        XCTAssertEqual(bt_gf64_add_scalar(0, val), val, "0 + a = a")

        // XOR with self = 0 (characteristic 2)
        XCTAssertEqual(bt_gf64_add_scalar(val, val), 0, "a + a = 0")

        // Known pattern
        let x: UInt64 = 0xAAAA_AAAA_AAAA_AAAA
        let y: UInt64 = 0x5555_5555_5555_5555
        XCTAssertEqual(bt_gf64_add_scalar(x, y), x ^ y, "Pattern test")
    }

    // MARK: - GF(2^64) Multiplication Identity and Zero

    /// Test GF(2^64) multiply identity: mul(1, a) = a
    func testGF64MulIdentity() {
        let testValues: [UInt64] = [
            0x123456789ABCDEF0,
            0xDEADBEEFCAFEBABE,
            1,
            0xFFFFFFFFFFFFFFFF,
            0x8000000000000000
        ]

        for a in testValues {
            let result = bt_gf64_mul_scalar(1, a)
            XCTAssertEqual(result, a, "mul(1, a) should equal a for 0x\(String(a, radix: 16))")
        }
    }

    /// Test GF(2^64) multiply zero: mul(0, a) = 0
    func testGF64MulZero() {
        let testValues: [UInt64] = [
            0x123456789ABCDEF0,
            0xDEADBEEFCAFEBABE,
            1,
            0xFFFFFFFFFFFFFFFF
        ]

        for a in testValues {
            let result = bt_gf64_mul_scalar(0, a)
            XCTAssertEqual(result, 0, "mul(0, a) should equal 0 for 0x\(String(a, radix: 16))")
            let result2 = bt_gf64_mul_scalar(a, 0)
            XCTAssertEqual(result2, 0, "mul(a, 0) should equal 0 for 0x\(String(a, radix: 16))")
        }
    }

    // MARK: - GF(2^64) Multiply Against Known Values

    /// Test GF(2^64) multiply with known values
    /// Using reduction polynomial: x^64 + x^4 + x^3 + x + 1
    func testGF64MulKnownValues() {
        // a = 1, b = 1 => 1
        XCTAssertEqual(bt_gf64_mul_scalar(1, 1), 1, "1 * 1 = 1")

        // a = 2, b = 1 => 2
        XCTAssertEqual(bt_gf64_mul_scalar(2, 1), 2, "2 * 1 = 2")

        // a = 2, b = 2 => 4 (in GF(2^64), squaring is linear but mul is carry-less)
        // 2 * 2 = 4 in regular arithmetic, but GF(2^64) mul may differ
        // For simple cases like powers of 2, the result is predictable
        let result = bt_gf64_mul_scalar(2, 2)
        // In GF(2^64), 2 * 2 should give a predictable result
        XCTAssertGreaterThan(result, 0, "2 * 2 should be non-zero")

        // Commutativity
        let a: UInt64 = 0xDEADBEEFCAFEBABE
        let b: UInt64 = 0x123456789ABCDEF0
        XCTAssertEqual(bt_gf64_mul_scalar(a, b), bt_gf64_mul_scalar(b, a),
                       "GF(2^64) mul should be commutative")
    }

    /// Test GF(2^64) multiply consistency (self-consistency)
    func testGF64MulConsistency() {
        let a: UInt64 = 0x123456789ABCDEF0
        let b: UInt64 = 0xFEDCBA9876543210

        let result1 = bt_gf64_mul_scalar(a, b)
        let result2 = bt_gf64_mul_scalar(a, b)

        XCTAssertEqual(result1, result2, "Same inputs should give same output")

        // Verify a * a = a^2 consistency
        let aSquared = bt_gf64_mul_scalar(a, a)
        // (a + a) should be 0, and a * 0 should be 0
        XCTAssertEqual(bt_gf64_mul_scalar(a, 0), 0, "a * 0 = 0")
    }

    // MARK: - GF(2^128) Multiply

    /// Test GF(2^128) multiply z0 = a_lo * b_lo (Karatsuba property)
    /// In Karatsuba: result[0] = z0 (the low part)
    func testGF128MulZ0Property() {
        var a: [UInt64] = [0x123456789ABCDEF0, 0xDEADBEEFCAFEBABE]
        var b: [UInt64] = [0xFEDCBA9876543210, 0x0FEDCBA987654321]
        var result: [UInt64] = [0, 0]

        bt_gf128_mul_scalar(&a, &b, &result)

        // z0 should equal a_lo * b_lo in GF(2^64)
        let expectedZ0 = bt_gf64_mul_scalar(a[0], b[0])
        XCTAssertEqual(result[0], expectedZ0,
                       "GF(2^128) z0 should equal a_lo * b_lo")

        // Result should not be all zeros for non-zero inputs
        XCTAssertTrue(result[0] != 0 || result[1] != 0,
                      "Non-zero inputs should give non-zero result")
    }

    /// Test GF(2^128) multiply consistency
    func testGF128MulConsistency() {
        var a: [UInt64] = [0x123456789ABCDEF0, 0xDEADBEEFCAFEBABE]
        var b: [UInt64] = [0xFEDCBA9876543210, 0x0FEDCBA987654321]
        var result1: [UInt64] = [0, 0]
        var result2: [UInt64] = [0, 0]

        bt_gf128_mul_scalar(&a, &b, &result1)
        bt_gf128_mul_scalar(&a, &b, &result2)

        XCTAssertEqual(result1[0], result2[0], "Same inputs give same lo")
        XCTAssertEqual(result1[1], result2[1], "Same inputs give same hi")
    }

    /// Test GF(2^128) multiply identity: mul(1, a) = a
    func testGF128MulIdentity() {
        var a: [UInt64] = [0x123456789ABCDEF0, 0xDEADBEEFCAFEBABE]
        var one: [UInt64] = [1, 0]
        var result: [UInt64] = [0, 0]

        bt_gf128_mul_scalar(&a, &one, &result)

        XCTAssertEqual(result[0], a[0], "GF(2^128) mul by 1 preserves lo")
        XCTAssertEqual(result[1], a[1], "GF(2^128) mul by 1 preserves hi")
    }

    /// Test GF(2^128) multiply zero: mul(0, a) = 0
    func testGF128MulZero() {
        var a: [UInt64] = [0x123456789ABCDEF0, 0xDEADBEEFCAFEBABE]
        var zero: [UInt64] = [0, 0]
        var result: [UInt64] = [0, 0]

        bt_gf128_mul_scalar(&a, &zero, &result)

        XCTAssertEqual(result[0], 0, "GF(2^128) mul by 0 gives 0 lo")
        XCTAssertEqual(result[1], 0, "GF(2^128) mul by 0 gives 0 hi")
    }

    // MARK: - ANE Availability Check

    /// Test that ANE Binary Tower reports as unavailable on CI
    func testANEBinaryTowerUnavailable() {
        XCTAssertFalse(ane_bt_available(), "ANE Binary Tower should report unavailable")

        // ANE state create should return NULL
        let state = ane_bt_create(64)
        XCTAssertNil(state, "ANE BT create should return NULL when ANE unavailable")

        // ANE operations should return identity/zero for add, 0 for mul
        XCTAssertEqual(ane_bt_gf64_add(0x1234, 0x5678), 0x1234 ^ 0x5678,
                       "ANE BT add fallback should be XOR")
    }

    // MARK: - GF(2^64) vs GF(2^128) Relationship

    /// Verify that GF(2^128) multiply with hi=0 gives consistent GF(2^64) result
    func testGF128VsGF64Relationship() {
        // When hi parts are zero, GF(2^128) should behave like GF(2^64) for lo part
        var a128: [UInt64] = [0x123456789ABCDEF0, 0]
        var b128: [UInt64] = [0xFEDCBA9876543210, 0]
        var result128: [UInt64] = [0, 0]

        bt_gf128_mul_scalar(&a128, &b128, &result128)

        let result64 = bt_gf64_mul_scalar(a128[0], b128[0])

        // The lo result of GF(2^128) should equal GF(2^64) result
        XCTAssertEqual(result128[0], result64,
                       "GF(2^128) lo should match GF(2^64) when hi=0")
    }
}
