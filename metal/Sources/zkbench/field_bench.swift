// Multi-field benchmark: Goldilocks and BabyBear
import zkMetal
import Foundation

public func runFieldBench() {
    print("=== Multi-Field Benchmark ===")

    // Goldilocks tests
    print("\n--- Goldilocks (p = 2^64 - 2^32 + 1) ---")

    // Basic arithmetic
    let ga = Gl(v: 42), gb = Gl(v: 100)
    let gsum = glAdd(ga, gb)
    if gsum.v == 142 { print("  [pass] GL add: 42 + 100 = 142") }
    else { print("  [FAIL] GL add: \(gsum.v)"); return }

    let gprod = glMul(ga, gb)
    if gprod.v == 4200 { print("  [pass] GL mul: 42 * 100 = 4200") }
    else { print("  [FAIL] GL mul: \(gprod.v)"); return }

    let gdiff = glSub(gb, ga)
    if gdiff.v == 58 { print("  [pass] GL sub: 100 - 58 = 58") }
    else { print("  [FAIL] GL sub: \(gdiff.v)") }

    // Inverse
    let gaInv = glInverse(ga)
    let gaTimesInv = glMul(ga, gaInv)
    if gaTimesInv.v == 1 { print("  [pass] GL inverse: 42 * 42^(-1) = 1") }
    else { print("  [FAIL] GL inverse: \(gaTimesInv.v)"); return }

    // Large multiply: near-modulus values
    let gBig1 = Gl(v: Gl.P - 1)
    let gBig2 = Gl(v: Gl.P - 2)
    let gBigProd = glMul(gBig1, gBig2)
    // (p-1)(p-2) = p^2 - 3p + 2 ≡ 2 mod p
    if gBigProd.v == 2 { print("  [pass] GL large mul: (p-1)*(p-2) = 2") }
    else { print("  [FAIL] GL large mul: \(gBigProd.v)"); return }

    // Root of unity
    let gRoot = Gl(v: Gl.ROOT_OF_UNITY)
    var gPow = gRoot
    for _ in 0..<31 { gPow = glSqr(gPow) }
    // gRoot^(2^31) should not be 1 (since TWO_ADICITY=32)
    if gPow.v != 1 {
        gPow = glSqr(gPow)
        if gPow.v == 1 { print("  [pass] GL root of unity: ω^(2^32) = 1") }
        else { print("  [FAIL] GL root: ω^(2^32) = \(gPow.v)"); return }
    } else {
        print("  [WARN] GL root: ω^(2^31) = 1 (order too small)")
    }

    // CPU benchmark
    var warmup = Gl.one
    for _ in 0..<10000 { warmup = glMul(warmup, ga) }
    let glIters = 1000000
    let glStart = CFAbsoluteTimeGetCurrent()
    var acc = Gl.one
    for _ in 0..<glIters { acc = glMul(acc, ga) }
    let glElapsed = (CFAbsoluteTimeGetCurrent() - glStart) * 1e9 / Double(glIters)
    print(String(format: "  GL mul: %.1f ns/op (%.0f M ops/s)", glElapsed, 1e3 / glElapsed))

    // BabyBear tests
    print("\n--- BabyBear (p = 2^31 - 2^27 + 1) ---")

    let ba = Bb(v: 42), bb = Bb(v: 100)
    let bsum = bbAdd(ba, bb)
    if bsum.v == 142 { print("  [pass] BB add: 42 + 100 = 142") }
    else { print("  [FAIL] BB add: \(bsum.v)"); return }

    let bprod = bbMul(ba, bb)
    if bprod.v == 4200 { print("  [pass] BB mul: 42 * 100 = 4200") }
    else { print("  [FAIL] BB mul: \(bprod.v)"); return }

    let baInv = bbInverse(ba)
    let baTimesInv = bbMul(ba, baInv)
    if baTimesInv.v == 1 { print("  [pass] BB inverse: 42 * 42^(-1) = 1") }
    else { print("  [FAIL] BB inverse: \(baTimesInv.v)"); return }

    // Large multiply
    let bBig1 = Bb(v: Bb.P - 1)
    let bBig2 = Bb(v: Bb.P - 2)
    let bBigProd = bbMul(bBig1, bBig2)
    if bBigProd.v == 2 { print("  [pass] BB large mul: (p-1)*(p-2) = 2") }
    else { print("  [FAIL] BB large mul: \(bBigProd.v)"); return }

    // Root of unity: ω^(2^27) should be 1
    let bRoot = bbRootOfUnity(logN: 27)
    var bPow = bRoot
    for _ in 0..<27 { bPow = bbSqr(bPow) }
    if bPow.v == 1 { print("  [pass] BB root of unity: ω^(2^27) = 1") }
    else { print("  [FAIL] BB root: ω^(2^27) = \(bPow.v)"); return }

    // Primitive check: ω^(2^26) ≠ 1
    var bHalf = bRoot
    for _ in 0..<26 { bHalf = bbSqr(bHalf) }
    if bHalf.v != 1 { print("  [pass] BB root is primitive: ω^(2^26) ≠ 1") }
    else { print("  [FAIL] BB root is not primitive") }

    // CPU benchmark
    var bw = Bb.one
    for _ in 0..<10000 { bw = bbMul(bw, ba) }
    let bbIters = 1000000
    let bbStart = CFAbsoluteTimeGetCurrent()
    var bacc = Bb.one
    for _ in 0..<bbIters { bacc = bbMul(bacc, ba) }
    let bbElapsed = (CFAbsoluteTimeGetCurrent() - bbStart) * 1e9 / Double(bbIters)
    print(String(format: "  BB mul: %.1f ns/op (%.0f M ops/s)", bbElapsed, 1e3 / bbElapsed))

    // Comparison
    print("\n--- Field Comparison ---")
    print(String(format: "  BN254 Fr (256-bit): ~2500 ns/mul"))
    print(String(format: "  Goldilocks (64-bit): %.0f ns/mul", glElapsed))
    print(String(format: "  BabyBear  (31-bit): %.0f ns/mul", bbElapsed))

    print("\nMulti-field benchmark complete.")
}
