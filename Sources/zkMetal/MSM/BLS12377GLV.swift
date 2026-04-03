// BLS12-377 GLV endomorphism for MSM acceleration
// φ(x,y) = (β·x, y) where β is a cube root of unity in Fq
// k = k1 + k2·λ (mod r) where λ is a cube root of unity in Fr
//
// Lattice vectors: v1 = (a1, 1), v2 = (-1, a1-1)
// where a1 = x0^2 = 91893752504881257701523279626832445441
// and det(v1,v2) = r

import Foundation

public enum BLS12377GLV {
    // λ (cube root of unity in Fr, standard form)
    // λ = 8444461749428370424248824938781546531284005582649182570233710176290576793600
    public static let LAMBDA: [UInt64] = [
        0x0000000000000000, 0x14885f3240000000,
        0x60b44d1e5c37b001, 0x12ab655e9a2ca556
    ]

    // β (cube root of unity in Fq, standard form)
    // β = 258664426012969093929703085429980814127835149614277183275038967946009968870203535512256352201271898244626862047231
    public static let BETA: [UInt64] = [
        0xffffffffffffffff, 0xd1e945779fffffff,
        0x59064ee822fb5bff, 0xb8882a75cc9bc8e3,
        0xbc8756ba8f8c524e, 0x01ae3a4617c510ea
    ]

    // β in Montgomery form (β * R mod q)
    // Computed as fq377Mul(Fq377.from64(BETA), Fq377.from64(Fq377.R2_MOD_Q))
    public static var betaMontgomery: Fq377 {
        fq377Mul(Fq377.from64(BETA), Fq377.from64(Fq377.R2_MOD_Q))
    }

    // a1 = 91893752504881257701523279626832445441 = x0^2
    // In hex: 0x452217cc900000010a11800000000001
    public static let A1: [UInt64] = [
        0x0a11800000000001, 0x452217cc90000001
    ]

    // a1 - 1 = 91893752504881257701523279626832445440
    public static let A1_MINUS_1: [UInt64] = [
        0x0a11800000000000, 0x452217cc90000001
    ]

    // Fr order r (for reduction)
    public static let R_ORDER: [UInt64] = [
        0x0a11800000000001, 0x59aa76fed0000001,
        0x60b44d1e5c37b001, 0x12ab655e9a2ca556
    ]

    // half_r = (r - 1) / 2 + 1
    public static let HALF_R: [UInt64] = [
        0x0508c00000000001, 0x2cd53b7f68000000,
        0xb05a268f2e1bd801, 0x0955b2af4d1652ab
    ]

    /// GLV decomposition: k → (k1, k2, neg1, neg2)
    /// where k ≡ k1 + k2·λ (mod r), with |k1|, |k2| ≈ √r
    ///
    /// Since b1=1 and a2=-1:
    ///   c1 = round(k · (a1-1) / r)
    ///   c2 = round(-k / r) = 0  (since 0 ≤ k < r)
    ///   k1 = k - c1·a1
    ///   k2 = -c1
    public static func decompose(_ k: [UInt32]) -> (k1: [UInt32], k2: [UInt32], neg1: Bool, neg2: Bool) {
        // Convert k to 4×64-bit
        let k64: [UInt64] = [
            UInt64(k[0]) | (UInt64(k[1]) << 32),
            UInt64(k[2]) | (UInt64(k[3]) << 32),
            UInt64(k[4]) | (UInt64(k[5]) << 32),
            UInt64(k[6]) | (UInt64(k[7]) << 32)
        ]

        // c1 = round(k · (a1-1) / r)
        // We compute k · (a1-1) as a wide product, then divide by r
        // Since (a1-1)/r ≈ 10^-38 and k < r ≈ 10^75, c1 ≈ 10^37, fits in 128 bits
        let c1 = computeC1(k64)

        // k1 = k - c1 · a1
        let c1a1 = mul128x128(c1, A1)  // 256-bit result
        var k1 = sub256(k64, c1a1)

        // k2 = -c1 (negative of c1)
        // We store |k2| = c1 and set neg2 = true
        var k2 = [c1.0, c1.1, UInt64(0), UInt64(0)]
        var neg2 = (c1.0 != 0 || c1.1 != 0)  // k2 is negative when c1 > 0

        // Reduce k1: if k1 > half_r, negate
        var neg1 = false
        if gte256(k1, HALF_R) {
            k1 = sub256mod(R_ORDER, k1)
            neg1 = true
        }

        // Convert back to 8×32-bit
        let k1_32 = to32(k1)
        let k2_32 = to32(k2)

        return (k1_32, k2_32, neg1, neg2)
    }

    // Compute c1 = round(k · (a1-1) / r)
    // We use the approach: multiply k (256-bit) by (a1-1) (128-bit),
    // getting a 384-bit product, then divide by r (253-bit)
    private static func computeC1(_ k: [UInt64]) -> (UInt64, UInt64) {
        // Full 256×128 multiply: k * (a1-1)
        let a = A1_MINUS_1
        var prod = [UInt64](repeating: 0, count: 6) // 384-bit product

        for i in 0..<4 {
            var carry: UInt64 = 0
            for j in 0..<2 {
                let (hi, lo) = k[i].multipliedFullWidth(by: a[j])
                let (s1, c1) = prod[i + j].addingReportingOverflow(lo)
                let (s2, c2) = s1.addingReportingOverflow(carry)
                prod[i + j] = s2
                carry = hi &+ (c1 ? 1 : 0) &+ (c2 ? 1 : 0)
            }
            prod[i + 2] = prod[i + 2] &+ carry
        }

        // Now prod contains k * (a1-1) as a 384-bit number.
        // We need to divide by r (253 bits).
        //
        // Approximate approach: since (a1-1) ≈ 2^127 and r ≈ 2^253,
        // the quotient c1 = prod / r ≈ k * 2^127 / 2^253 = k * 2^(-126)
        // So c1 ≈ k >> 126 which is at most 2^127.
        //
        // For exact computation, we do schoolbook division.
        // prod is at most 2^(253+128) = 2^381, r is 253 bits.
        // Quotient is at most 2^128.
        return divideByR(prod)
    }

    // Divide a 384-bit number by r, returning the 128-bit quotient
    // Uses trial division with the top limbs
    private static func divideByR(_ n: [UInt64]) -> (UInt64, UInt64) {
        // r as 256-bit
        let r = R_ORDER

        // The quotient is at most 128 bits. We compute it by repeated subtraction
        // of r shifted left. More efficient: use multi-precision division.

        var remainder = n // 384-bit
        var quotient: [UInt64] = [0, 0] // 128-bit result

        // Since quotient fits in 128 bits, we need at most 128 shift-subtract steps.
        // But this is slow. Instead, use the approximation:
        //
        // c1 ≈ (prod[5:2]) >> (253 - 256) ... actually, let me use a different approach.
        //
        // Simpler: compute g = floor(2^384 / r) once, then c1 = (n * g) >> 384
        // But that requires another wide multiply.
        //
        // Simplest correct approach for CPU: convert to arbitrary precision and divide.
        // For now, use iterative approach since this is CPU-only.

        // Try: shift r into 384-bit, aligned to top of n
        var rShifted = [UInt64](repeating: 0, count: 6)

        // n has at most 381 bits, r has 253 bits.
        // Max shift = 381 - 253 = 128 bits.
        // We process from shift=128 down to shift=0.

        for shift in stride(from: 128, through: 0, by: -1) {
            // Shift r left by `shift` bits into rShifted (384-bit)
            let limbShift = shift / 64
            let bitShift = shift % 64

            for i in 0..<6 { rShifted[i] = 0 }
            for i in 0..<4 {
                let destIdx = i + limbShift
                if destIdx < 6 {
                    rShifted[destIdx] |= r[i] << bitShift
                }
                if bitShift > 0 && destIdx + 1 < 6 {
                    rShifted[destIdx + 1] |= r[i] >> (64 - bitShift)
                }
            }

            // Compare remainder >= rShifted
            var ge = true
            for i in stride(from: 5, through: 0, by: -1) {
                if remainder[i] > rShifted[i] { break }
                if remainder[i] < rShifted[i] { ge = false; break }
            }

            if ge {
                // remainder -= rShifted
                var borrow: UInt64 = 0
                for i in 0..<6 {
                    let (s1, b1) = remainder[i].subtractingReportingOverflow(rShifted[i])
                    let (s2, b2) = s1.subtractingReportingOverflow(borrow)
                    remainder[i] = s2
                    borrow = (b1 ? 1 : 0) + (b2 ? 1 : 0)
                }

                // quotient |= 1 << shift
                let qLimb = shift / 64
                let qBit = shift % 64
                if qLimb < 2 {
                    quotient[qLimb] |= UInt64(1) << qBit
                }
            }
        }

        return (quotient[0], quotient[1])
    }

    // 128×128 multiply → 256-bit result
    private static func mul128x128(_ a: (UInt64, UInt64), _ b: [UInt64]) -> [UInt64] {
        var r = [UInt64](repeating: 0, count: 4)
        let av = [a.0, a.1]
        for i in 0..<2 {
            var carry: UInt64 = 0
            for j in 0..<2 {
                let (hi, lo) = av[i].multipliedFullWidth(by: b[j])
                let (s1, c1) = r[i + j].addingReportingOverflow(lo)
                let (s2, c2) = s1.addingReportingOverflow(carry)
                r[i + j] = s2
                carry = hi &+ (c1 ? 1 : 0) &+ (c2 ? 1 : 0)
            }
            r[i + 2] = carry
        }
        return r
    }

    // 256-bit subtraction (a - b), assumes a >= b
    private static func sub256(_ a: [UInt64], _ b: [UInt64]) -> [UInt64] {
        var r = [UInt64](repeating: 0, count: 4)
        var borrow: UInt64 = 0
        for i in 0..<4 {
            let (s1, b1) = a[i].subtractingReportingOverflow(b[i])
            let (s2, b2) = s1.subtractingReportingOverflow(borrow)
            r[i] = s2
            borrow = (b1 ? 1 : 0) + (b2 ? 1 : 0)
        }
        // If borrow, add r back
        if borrow != 0 {
            var carry: UInt64 = 0
            for i in 0..<4 {
                let (s1, c1) = r[i].addingReportingOverflow(R_ORDER[i])
                let (s2, c2) = s1.addingReportingOverflow(carry)
                r[i] = s2
                carry = (c1 ? 1 : 0) + (c2 ? 1 : 0)
            }
        }
        return r
    }

    private static func sub256mod(_ a: [UInt64], _ b: [UInt64]) -> [UInt64] {
        var r = [UInt64](repeating: 0, count: 4)
        var borrow: UInt64 = 0
        for i in 0..<4 {
            let (s1, b1) = a[i].subtractingReportingOverflow(b[i])
            let (s2, b2) = s1.subtractingReportingOverflow(borrow)
            r[i] = s2
            borrow = (b1 ? 1 : 0) + (b2 ? 1 : 0)
        }
        return r
    }

    private static func gte256(_ a: [UInt64], _ b: [UInt64]) -> Bool {
        for i in stride(from: 3, through: 0, by: -1) {
            if a[i] > b[i] { return true }
            if a[i] < b[i] { return false }
        }
        return true
    }

    private static func to32(_ v: [UInt64]) -> [UInt32] {
        var r = [UInt32](repeating: 0, count: 8)
        for i in 0..<4 {
            r[2*i] = UInt32(v[i] & 0xFFFFFFFF)
            r[2*i+1] = UInt32(v[i] >> 32)
        }
        return r
    }
}
