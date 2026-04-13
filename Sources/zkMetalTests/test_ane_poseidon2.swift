import XCTest
import ANEOps

/// Tests for ANE Poseidon2 S-box primitives (BabyBear x^7 and M31 x^5)
/// Tests verify the scalar fallback implementations match expected behavior.
class ANEPoseidon2Tests: XCTestCase {

    // MARK: - BabyBear Field Arithmetic Reference

    /// BabyBear field: p = 2^31 - 2^27 + 1 = 0x78000001 = 2013265921
    private let bbP: UInt32 = 2013265921
    private let bbPInv: UInt32 = 2281701377
    private let bbR2: UInt32 = 1172168163

    /// Montgomery reduce 64-bit to BabyBear field element
    private func bbMontyReduce64(_ x: UInt64) -> UInt32 {
        let lo = UInt32(x & 0xFFFFFFFF)
        let q = lo &* bbPInv
        let t = Int64(x) - Int64(q) &* Int64(bbP)
        let r = Int32(t >> 32)
        return r < 0 ? UInt32(r + Int32(bbP)) : UInt32(r)
    }

    /// Convert to Montgomery form
    private func bbToMonty(_ a: UInt32) -> UInt32 {
        bbMontyReduce64(UInt64(a) &* UInt64(bbR2))
    }

    /// BabyBear multiplication
    private func bbMul(_ a: UInt32, _ b: UInt32) -> UInt32 {
        bbMontyReduce64(UInt64(a) &* UInt64(b))
    }

    /// BabyBear addition
    private func bbAdd(_ a: UInt32, _ b: UInt32) -> UInt32 {
        let s = UInt64(a) + UInt64(b)
        return s >= bbP ? UInt32(s - bbP) : UInt32(s)
    }

    /// BabyBear x^7 S-box: x^7 = x * x^2 * x^4
    private func bbX7Scalar(_ x: UInt32) -> UInt32 {
        let x2 = bbMul(x, x)
        let x4 = bbMul(x2, x2)
        return bbMul(x4, x)
    }

    // MARK: - M31 Field Arithmetic Reference

    /// M31 field: p = 2^31 - 1 = 0x7FFFFFFF
    private let m31P: UInt32 = 0x7FFFFFFF

    /// M31 reduce
    private func m31Reduce(_ x: UInt32) -> UInt32 {
        let r = (x >> 31) + (x & m31P)
        return r >= m31P ? r - m31P : r
    }

    /// M31 addition
    private func m31Add(_ a: UInt32, _ b: UInt32) -> UInt32 {
        m31Reduce(a &+ b)
    }

    /// M31 multiplication
    private func m31Mul(_ a: UInt32, _ b: UInt32) -> UInt32 {
        let prod = UInt64(a) &* UInt64(b)
        let lo = UInt32(prod & UInt64(m31P))
        let hi = UInt32(prod >> 31)
        return m31Reduce(lo &+ hi)
    }

    /// M31 x^5 S-box: x^5 = x * x^4
    private func m31X5Scalar(_ x: UInt32) -> UInt32 {
        let x2 = m31Mul(x, x)
        let x4 = m31Mul(x2, x2)
        return m31Mul(x4, x)
    }

    // MARK: - BabyBear x^7 S-box Tests

    /// Test bb_poseidon2_sbox_ane vs scalar x^7 computation
    func testBabyBearSBoxVsScalar() {
        let testValues: [UInt32] = [0, 1, 2, 3, 100, 1000, 0x78000000]

        for xRaw in testValues {
            // Convert to Montgomery form for S-box
            let x = bbToMonty(xRaw)

            // Compute expected using scalar reference
            let expected = bbX7Scalar(x)

            // Test via ANE function (which currently calls scalar fallback)
            var state: [UInt32] = [x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            bb_poseidon2_sbox_ane(&state)

            XCTAssertEqual(state[0], expected,
                           "BabyBear S-box should give x^7 for input 0x\(String(x, radix: 16))")
        }
    }

    /// Test bb_poseidon2_sbox_ane on full state array (16 elements)
    func testBabyBearSBoxFullState() {
        // Initialize state with known values in Montgomery form
        var state: [UInt32] = [
            bbToMonty(1),
            bbToMonty(2),
            bbToMonty(3),
            bbToMonty(4),
            bbToMonty(5),
            bbToMonty(6),
            bbToMonty(7),
            bbToMonty(8),
            bbToMonty(9),
            bbToMonty(10),
            bbToMonty(11),
            bbToMonty(12),
            bbToMonty(13),
            bbToMonty(14),
            bbToMonty(15),
            bbToMonty(16)
        ]

        let expected = state.map { bbX7Scalar($0) }

        bb_poseidon2_sbox_ane(&state)

        for i in 0..<16 {
            XCTAssertEqual(state[i], expected[i],
                           "BabyBear S-box element \(i) should be x^7")
        }
    }

    /// Test BabyBear S-box identity property: sbox(1) = 1
    func testBabyBearSBoxIdentity() {
        var state: [UInt32] = Array(repeating: 0, count: 16)
        state[0] = bbToMonty(1)

        bb_poseidon2_sbox_ane(&state)

        // 1^7 = 1 in any field
        XCTAssertEqual(state[0], bbToMonty(1), "S-box(1) should equal 1")
    }

    /// Test BabyBear S-box zero property: sbox(0) = 0
    func testBabyBearSBoxZero() {
        var state: [UInt32] = Array(repeating: 0, count: 16)
        state[0] = 0 // Already in Montgomery form for 0

        bb_poseidon2_sbox_ane(&state)

        XCTAssertEqual(state[0], 0, "S-box(0) should equal 0")
    }

    /// Test BabyBear S-box computes x^7 correctly via verification
    func testBabyBearX7Verification() {
        // x = 2 in Montgomery form
        let x = bbToMonty(2)

        // Manually compute x^7
        let x2 = bbMul(x, x)           // x^2
        let x4 = bbMul(x2, x2)          // x^4
        let x7 = bbMul(x4, x)           // x^7 = x^4 * x

        // Verify via S-box
        var state: [UInt32] = Array(repeating: 0, count: 16)
        state[0] = x
        bb_poseidon2_sbox_ane(&state)

        XCTAssertEqual(state[0], x7, "S-box should compute x^7 = x^4 * x")

        // Also verify x^7 = x * x^2 * x^4
        let x3 = bbMul(x2, x)           // x^3
        let x7_v2 = bbMul(x3, x4)        // x^3 * x^4 = x^7

        XCTAssertEqual(state[0], x7_v2, "S-box should equal x * x^2 * x^4")
    }

    // MARK: - M31 x^5 S-box Tests

    /// Test m31_poseidon2_sbox_ane vs scalar x^5 computation
    func testM31SBoxVsScalar() {
        // M31 values should be in range [0, p)
        let testValues: [UInt32] = [0, 1, 2, 3, 100, 1000, 0x3FFFFFFF]

        for x in testValues {
            // Compute expected using scalar reference
            let expected = m31X5Scalar(x)

            // Test via ANE function (which currently calls scalar fallback)
            var state: [UInt32] = [x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            m31_poseidon2_sbox_ane(&state)

            XCTAssertEqual(state[0], expected,
                           "M31 S-box should give x^5 for input 0x\(String(x, radix: 16))")
        }
    }

    /// Test M31 S-box on full state array
    func testM31SBoxFullState() {
        var state: [UInt32] = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
        ]

        let expected = state.map { m31X5Scalar($0) }

        m31_poseidon2_sbox_ane(&state)

        for i in 0..<16 {
            XCTAssertEqual(state[i], expected[i],
                           "M31 S-box element \(i) should be x^5")
        }
    }

    /// Test M31 S-box identity property: sbox(1) = 1
    func testM31SBoxIdentity() {
        var state: [UInt32] = Array(repeating: 0, count: 16)
        state[0] = 1

        m31_poseidon2_sbox_ane(&state)

        // 1^5 = 1 in any field
        XCTAssertEqual(state[0], 1, "M31 S-box(1) should equal 1")
    }

    /// Test M31 S-box zero property: sbox(0) = 0
    func testM31SBoxZero() {
        var state: [UInt32] = Array(repeating: 0, count: 16)
        state[0] = 0

        m31_poseidon2_sbox_ane(&state)

        XCTAssertEqual(state[0], 0, "M31 S-box(0) should equal 0")
    }

    /// Test M31 S-box computes x^5 correctly
    func testM31X5Verification() {
        let x: UInt32 = 2

        // Manually compute x^5 = x * x^4
        let x2 = m31Mul(x, x)           // x^2
        let x4 = m31Mul(x2, x2)         // x^4
        let x5 = m31Mul(x4, x)          // x^5 = x^4 * x

        // Verify via S-box
        var state: [UInt32] = Array(repeating: 0, count: 16)
        state[0] = x
        m31_poseidon2_sbox_ane(&state)

        XCTAssertEqual(state[0], x5, "M31 S-box should compute x^5")
    }

    // MARK: - Batch S-box Tests

    /// Test BabyBear batch S-box
    func testBabyBearBatchSBox() {
        let input: [UInt32] = [
            bbToMonty(1), bbToMonty(2), bbToMonty(3), bbToMonty(4),
            bbToMonty(5), bbToMonty(6), bbToMonty(7), bbToMonty(8),
            bbToMonty(9), bbToMonty(10), bbToMonty(11), bbToMonty(12),
            bbToMonty(13), bbToMonty(14), bbToMonty(15), bbToMonty(16)
        ]

        var output: [UInt32] = Array(repeating: 0, count: 16)
        var stateCopy = input

        // Test single first, then batch
        bb_poseidon2_sbox_ane(&stateCopy)

        bb_poseidon2_sbox_batch_ane(input, 1, &output)

        for i in 0..<16 {
            XCTAssertEqual(output[i], stateCopy[i],
                           "Batch S-box[\(i)] should equal single S-box")
        }
    }

    /// Test M31 batch S-box
    func testM31BatchSBox() {
        let input: [UInt32] = Array(1...16)

        var output: [UInt32] = Array(repeating: 0, count: 16)
        var stateCopy = input

        m31_poseidon2_sbox_ane(&stateCopy)
        m31_poseidon2_sbox_batch_ane(input, 1, &output)

        for i in 0..<16 {
            XCTAssertEqual(output[i], stateCopy[i],
                           "M31 Batch S-box[\(i)] should equal single S-box")
        }
    }

    // MARK: - Stress Tests

    /// Test BabyBear S-box with random-ish values
    func testBabyBearSBoxStress() {
        let testValues: [UInt32] = [
            0x100000, 0x200000, 0x400000, 0x78000000,
            0x123456, 0xABCDEF, 0xFFFFFF, 0x7777777
        ]

        for xRaw in testValues {
            let x = bbToMonty(xRaw % bbP) // Ensure valid range

            var state: [UInt32] = Array(repeating: 0, count: 16)
            state[0] = x

            let expected = bbX7Scalar(x)

            bb_poseidon2_sbox_ane(&state)

            XCTAssertEqual(state[0], expected,
                           "BabyBear S-box stress test for 0x\(String(x, radix: 16))")
        }
    }
}
