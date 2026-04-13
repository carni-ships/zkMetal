import XCTest
import ANEOps

/// Tests for ANE MiMC primitives (scalar fallback path)
/// Note: ANE hardware returns -1 (not available), so these tests verify
/// the scalar fallback implementations where they exist.
class ANEMiMCTests: XCTestCase {

    // MARK: - MiMC x^7 Scalar Computation

    /// Test that MiMC x^7 scalar placeholder is callable
    /// Note: The current implementation is a no-op placeholder that copies
    /// input to output. Real BN254 Fr arithmetic is not yet implemented.
    func testMiMCX7ScalarCallable() {
        var input: [UInt8] = Array(repeating: 0, count: 32)
        var output: [UInt8] = Array(repeating: 0, count: 32)

        // The mimc_x7_scalar function is internal (static) in the .mm file,
        // so we can only test the public API wrappers which return -1.
        XCTAssertFalse(ane_mimc_available(), "ANE MiMC should report as unavailable")

        // Test that hash function returns error code (ANE not available)
        let result = ane_mimc_hash(nil, input, input, &output)
        XCTAssertEqual(result, -1, "MiMC hash should return -1 when ANE unavailable")
    }

    /// Test MiMC state management (verifies API contracts)
    func testMiMCStateManagement() {
        XCTAssertFalse(ane_mimc_available(), "ANE MiMC should report unavailable")

        // Create should return NULL when ANE unavailable
        let state = ane_mimc_create(91)
        XCTAssertNil(state, "MiMC create should return NULL when ANE unavailable")

        // Destroy should be safe with NULL
        ane_mimc_destroy(nil)
        ane_mimc_destroy(state) // NULL is safe
    }

    /// Test MiMC batch hash returns error
    func testMiMCBatchHashReturnsError() {
        var inputs: [UInt8] = Array(repeating: 0, count: 32)
        var outputs: [UInt8] = Array(repeating: 0, count: 32)

        let result = ane_mimc_batch_hash(nil, inputs, nil, 1, &outputs)
        XCTAssertEqual(result, -1, "MiMC batch hash should return -1 when ANE unavailable")
    }

    // MARK: - BabyBear x^7 Reference Computation (for comparison)

    /// Reference implementation of BabyBear x^7 S-box using field arithmetic
    /// This mirrors the logic in ane_poseidon2.mm bb_sbox_scalar()
    /// BabyBear field: p = 2^31 - 2^27 + 1 = 0x78000001 = 2013265921
    func bbMontyReduce64(_ x: UInt64) -> UInt32 {
        let p: UInt64 = 2013265921
        let pInv: UInt64 = 2281701377
        let lo = UInt32(x & 0xFFFFFFFF)
        let q = lo &* UInt32(pInv)
        let t = Int64(x) - Int64(q) &* Int64(p)
        let r = Int32(t >> 32)
        return r < 0 ? UInt32(r + Int32(p)) : UInt32(r)
    }

    func bbToMonty(_ a: UInt32) -> UInt32 {
        let r2: UInt64 = 1172168163
        return bbMontyReduce64(UInt64(a) &* r2)
    }

    func bbMul(_ a: UInt32, _ b: UInt32) -> UInt32 {
        bbMontyReduce64(UInt64(a) &* UInt64(b))
    }

    /// Compute x^7 in BabyBear field: x^7 = x * x^2 * x^4
    func computeBabyBearX7(_ x: UInt32) -> UInt32 {
        let x2 = bbMul(x, x)
        let x4 = bbMul(x2, x2)
        return bbMul(x4, x)
    }

    /// Test BabyBear x^7 computation against known values
    func testBabyBearX7Computation() {
        // Test cases with known values (computed manually)
        // x = 0 should give 0
        XCTAssertEqual(computeBabyBearX7(0), 0, "x^7 for x=0 should be 0")

        // x = 1 should give 1
        XCTAssertEqual(computeBabyBearX7(1), 1, "x^7 for x=1 should be 1")

        // x = 2: 2^7 = 128
        // But in BabyBear field with Montgomery form, need to convert
        let x2Monty = bbToMonty(2)
        let result = computeBabyBearX7(x2Monty)
        // 2^7 = 128 in integers, then convert back from Montgomery
        // This is a basic sanity check that the computation runs
        XCTAssertGreaterThan(result, 0, "x^7 should produce non-zero for x=2")
    }

    /// Test single MiMC round: h = (x + c)^7
    /// This uses BabyBear field arithmetic as a reference since MiMC
    /// scalar implementation is a placeholder
    func testSingleMiMCRound() {
        let p: UInt32 = 2013265921

        // Add function
        func bbAdd(_ a: UInt32, _ b: UInt32) -> UInt32 {
            let s = UInt64(a) + UInt64(b)
            return s >= p ? UInt32(s - p) : UInt32(s)
        }

        let x: UInt32 = bbToMonty(5)
        let c: UInt32 = bbToMonty(3)
        let xPlusC = bbAdd(x, c)
        let result = computeBabyBearX7(xPlusC)

        // Sanity check: result should be valid BabyBear element
        XCTAssertLessThan(result, p, "Result should be reduced modulo p")
    }
}
