// secp256k1 GLV endomorphism for MSM acceleration
// φ(x,y) = (β·x, y) where β is a cube root of unity in Fp
// k = k1 + k2·λ (mod n) where λ is a cube root of unity in Fr
//
// Babai rounding: c_i = floor(k · g_i / 2^384)
// Lattice basis: (a1, -minus_b1), (a2, b2)

import Foundation

public enum Secp256k1GLV {
    // Curve order n
    public static let N: [UInt64] = [
        0xBFD25E8CD0364141, 0xBAAEDCE6AF48A03B,
        0xFFFFFFFFFFFFFFFE, 0xFFFFFFFFFFFFFFFF
    ]

    public static let HALF_N: [UInt64] = [
        (0xBFD25E8CD0364141 >> 1) | (0xBAAEDCE6AF48A03B << 63),
        (0xBAAEDCE6AF48A03B >> 1) | (0xFFFFFFFFFFFFFFFE << 63),
        (0xFFFFFFFFFFFFFFFE >> 1) | (0xFFFFFFFFFFFFFFFF << 63),
        0xFFFFFFFFFFFFFFFF >> 1
    ]

    // β (cube root of unity in Fp, Montgomery form, 8×32 LE)
    public static let BETA_MONT: [UInt32] = [
        0x8e81894e, 0x58a4361c, 0x1c4b80af, 0x03fde163,
        0xd02e3905, 0xf8e98978, 0xbcbb3d53, 0x7a4a36ae
    ]

    // Babai rounding constants: g_i = floor(2^384 * basis / n)
    static let G1: [UInt64] = [
        0xe893209a45dbb031, 0x3daa8a1471e8ca7f,
        0xe86c90e49284eb15, 0x3086d221a7d46bcd
    ]
    static let G2: [UInt64] = [
        0x1571b4ae8ac47f71, 0x221208ac9df506c6,
        0x6f547fa90abfe4c4, 0xe4437ed6010e8828
    ]

    // Lattice basis (short form)
    static let A1: (UInt64, UInt64) = (0xe86c90e49284eb15, 0x3086d221a7d46bcd)
    static let MINUS_B1: (UInt64, UInt64) = (0x6f547fa90abfe4c3, 0xe4437ed6010e8828)
    // a2 is 129 bits
    static let A2: (UInt64, UInt64, UInt64) = (0x57c1108d9d44cfd8, 0x14ca50f7a8e2f3f6, 0x1)
    // b2 = a1
    static let B2: (UInt64, UInt64) = (0xe86c90e49284eb15, 0x3086d221a7d46bcd)

    /// Decompose a 256-bit scalar k into (k1, k2) such that k ≡ k1 + k2·λ (mod n)
    public static func decompose(_ scalar: [UInt32]) -> (k1: [UInt32], k2: [UInt32], neg1: Bool, neg2: Bool) {
        var kr: (UInt64, UInt64, UInt64, UInt64) = (
            UInt64(scalar[0]) | (UInt64(scalar[1]) << 32),
            UInt64(scalar[2]) | (UInt64(scalar[3]) << 32),
            UInt64(scalar[4]) | (UInt64(scalar[5]) << 32),
            UInt64(scalar[6]) | (UInt64(scalar[7]) << 32)
        )

        // Reduce mod n
        let nT = (N[0], N[1], N[2], N[3])
        while gte256(kr, nT) {
            (kr, _) = sub256(kr, nT)
        }

        // c1 = floor(k * g1 / 2^384)
        let c1 = mulTop128(kr, G1)
        // c2 = floor(k * g2 / 2^384)
        let c2 = mulTop128(kr, G2)

        // k1 = k - c1*a1 - c2*a2 (mod n)
        // c2*a2 can be up to 257 bits, so compute sum = c1*a1 + c2*a2 first
        let c1a1 = mul128x128(c1, A1)
        let c2a2 = mul128x192(c2, A2)

        var borrow: Bool
        // sum = c1*a1 + c2*a2 (wide addition including 5th limb)
        var sum: (UInt64, UInt64, UInt64, UInt64) = (0, 0, 0, 0)
        var sumCarry: UInt64 = 0
        do {
            let (s0, c0) = c1a1.0.addingReportingOverflow(c2a2.0)
            sum.0 = s0; sumCarry = c0 ? 1 : 0
            let (s1a, c1a) = c1a1.1.addingReportingOverflow(c2a2.1)
            let (s1b, c1b) = s1a.addingReportingOverflow(sumCarry)
            sum.1 = s1b; sumCarry = (c1a ? 1 : 0) + (c1b ? 1 : 0)
            let (s2a, c2a) = c1a1.2.addingReportingOverflow(c2a2.2)
            let (s2b, c2b) = s2a.addingReportingOverflow(sumCarry)
            sum.2 = s2b; sumCarry = (c2a ? 1 : 0) + (c2b ? 1 : 0)
            let (s3a, c3a) = c1a1.3.addingReportingOverflow(c2a2.3)
            let (s3b, c3b) = s3a.addingReportingOverflow(sumCarry)
            sum.3 = s3b; sumCarry = (c3a ? 1 : 0) + (c3b ? 1 : 0) + c2a2.4
        }

        // k1 = k - sum (mod n)
        var k1: (UInt64, UInt64, UInt64, UInt64)
        (k1, borrow) = sub256(kr, sum)
        if borrow || sumCarry > 0 {
            (k1, _) = add256(k1, nT)
            if sumCarry > 1 { (k1, _) = add256(k1, nT) }
        }

        // k2 = c1*minus_b1 - c2*b2
        let c1mb1 = mul128x128(c1, MINUS_B1)
        let c2b2 = mul128x128(c2, B2)

        var k2_neg = false
        var k2: (UInt64, UInt64, UInt64, UInt64)
        (k2, borrow) = sub256(c1mb1, c2b2)
        if borrow {
            k2_neg = true
            k2 = neg256(k2)
        }

        // Center k1: if k1 > n/2, negate
        var neg1 = false
        let halfN = (HALF_N[0], HALF_N[1], HALF_N[2], HALF_N[3])
        if gte256(k1, halfN) {
            (k1, _) = sub256(nT, k1)
            neg1 = true
        }

        let k1_32 = to32(k1)
        let k2_32 = to32(k2)
        return (k1_32, k2_32, neg1, k2_neg)
    }

    /// Apply the endomorphism: φ(x, y) = (β·x, y)
    public static func applyEndomorphism(_ p: SecpPointAffine) -> SecpPointAffine {
        let beta = SecpFp(v: (BETA_MONT[0], BETA_MONT[1], BETA_MONT[2], BETA_MONT[3],
                               BETA_MONT[4], BETA_MONT[5], BETA_MONT[6], BETA_MONT[7]))
        return SecpPointAffine(x: secpMul(beta, p.x), y: p.y)
    }

    // MARK: - Arithmetic helpers

    // 256×256 multiply, return top 128 bits (floor(product / 2^384))
    private static func mulTop128(_ k: (UInt64, UInt64, UInt64, UInt64), _ g: [UInt64]) -> (UInt64, UInt64) {
        var prod = [UInt64](repeating: 0, count: 8)
        let kv = [k.0, k.1, k.2, k.3]
        for i in 0..<4 {
            var carry: UInt64 = 0
            for j in 0..<4 {
                let (hi, lo) = kv[i].multipliedFullWidth(by: g[j])
                let (s1, c1) = prod[i+j].addingReportingOverflow(lo)
                let (s2, c2) = s1.addingReportingOverflow(carry)
                prod[i+j] = s2
                carry = hi &+ (c1 ? 1 : 0) &+ (c2 ? 1 : 0)
            }
            prod[i+4] &+= carry
        }
        return (prod[6], prod[7])
    }

    private static func mul128x128(_ a: (UInt64, UInt64), _ b: (UInt64, UInt64)) -> (UInt64, UInt64, UInt64, UInt64) {
        let (h00, l00) = a.0.multipliedFullWidth(by: b.0)
        let (h01, l01) = a.0.multipliedFullWidth(by: b.1)
        let (h10, l10) = a.1.multipliedFullWidth(by: b.0)
        let (h11, l11) = a.1.multipliedFullWidth(by: b.1)

        let r0 = l00
        let (s1a, c1a) = l01.addingReportingOverflow(h00)
        let (s1b, c1b) = s1a.addingReportingOverflow(l10)
        let r1 = s1b
        let (s2a, c2a) = h01.addingReportingOverflow(h10)
        let (s2b, c2b) = s2a.addingReportingOverflow(l11)
        let (s2c, c2c) = s2b.addingReportingOverflow((c1a ? 1 : 0) &+ (c1b ? 1 : 0))
        let r2 = s2c
        let r3 = h11 &+ (c2a ? 1 : 0) &+ (c2b ? 1 : 0) &+ (c2c ? 1 : 0)
        return (r0, r1, r2, r3)
    }

    // 128 × 192 (a2 is 129 bits stored as 3 limbs) -> 320-bit
    private static func mul128x192(_ c: (UInt64, UInt64), _ a2: (UInt64, UInt64, UInt64)) -> (UInt64, UInt64, UInt64, UInt64, UInt64) {
        var prod = [UInt64](repeating: 0, count: 5)
        let cv = [c.0, c.1]
        let av = [a2.0, a2.1, a2.2]
        for i in 0..<2 {
            var carry: UInt64 = 0
            for j in 0..<3 {
                let (hi, lo) = cv[i].multipliedFullWidth(by: av[j])
                let (s1, c1) = prod[i+j].addingReportingOverflow(lo)
                let (s2, c2) = s1.addingReportingOverflow(carry)
                prod[i+j] = s2
                carry = hi &+ (c1 ? 1 : 0) &+ (c2 ? 1 : 0)
            }
            prod[i+3] &+= carry
        }
        return (prod[0], prod[1], prod[2], prod[3], prod[4])
    }

    private static func gte256(_ a: (UInt64, UInt64, UInt64, UInt64),
                                _ b: (UInt64, UInt64, UInt64, UInt64)) -> Bool {
        if a.3 != b.3 { return a.3 > b.3 }
        if a.2 != b.2 { return a.2 > b.2 }
        if a.1 != b.1 { return a.1 > b.1 }
        return a.0 >= b.0
    }

    private static func sub256(_ a: (UInt64, UInt64, UInt64, UInt64),
                                _ b: (UInt64, UInt64, UInt64, UInt64)) -> ((UInt64, UInt64, UInt64, UInt64), Bool) {
        var br: UInt64 = 0
        let (d0, b0) = a.0.subtractingReportingOverflow(b.0)
        let (d0b, b0b) = d0.subtractingReportingOverflow(br)
        br = (b0 ? 1 : 0) + (b0b ? 1 : 0)
        let (d1, b1) = a.1.subtractingReportingOverflow(b.1)
        let (d1b, b1b) = d1.subtractingReportingOverflow(br)
        br = (b1 ? 1 : 0) + (b1b ? 1 : 0)
        let (d2, b2) = a.2.subtractingReportingOverflow(b.2)
        let (d2b, b2b) = d2.subtractingReportingOverflow(br)
        br = (b2 ? 1 : 0) + (b2b ? 1 : 0)
        let (d3, b3) = a.3.subtractingReportingOverflow(b.3)
        let (d3b, b3b) = d3.subtractingReportingOverflow(br)
        let borrow = (b3 ? 1 : 0) + (b3b ? 1 : 0) > 0
        return ((d0b, d1b, d2b, d3b), borrow)
    }

    private static func add256(_ a: (UInt64, UInt64, UInt64, UInt64),
                                _ b: (UInt64, UInt64, UInt64, UInt64)) -> ((UInt64, UInt64, UInt64, UInt64), Bool) {
        var c: UInt64 = 0
        let (s0, c0) = a.0.addingReportingOverflow(b.0)
        let (s0c, c0c) = s0.addingReportingOverflow(c)
        c = (c0 ? 1 : 0) + (c0c ? 1 : 0)
        let (s1, c1) = a.1.addingReportingOverflow(b.1)
        let (s1c, c1c) = s1.addingReportingOverflow(c)
        c = (c1 ? 1 : 0) + (c1c ? 1 : 0)
        let (s2, c2) = a.2.addingReportingOverflow(b.2)
        let (s2c, c2c) = s2.addingReportingOverflow(c)
        c = (c2 ? 1 : 0) + (c2c ? 1 : 0)
        let (s3, c3) = a.3.addingReportingOverflow(b.3)
        let (s3c, c3c) = s3.addingReportingOverflow(c)
        let carry = (c3 ? 1 : 0) + (c3c ? 1 : 0) > 0
        return ((s0c, s1c, s2c, s3c), carry)
    }

    private static func neg256(_ a: (UInt64, UInt64, UInt64, UInt64)) -> (UInt64, UInt64, UInt64, UInt64) {
        let n0 = ~a.0 &+ 1
        let c0: UInt64 = (a.0 == 0) ? 1 : 0
        let n1 = ~a.1 &+ c0
        let c1: UInt64 = (a.1 == 0 && c0 == 1) ? 1 : 0
        let n2 = ~a.2 &+ c1
        let c2: UInt64 = (a.2 == 0 && c1 == 1) ? 1 : 0
        let n3 = ~a.3 &+ c2
        return (n0, n1, n2, n3)
    }

    private static func to32(_ v: (UInt64, UInt64, UInt64, UInt64)) -> [UInt32] {
        var r = [UInt32](repeating: 0, count: 8)
        r[0] = UInt32(v.0 & 0xFFFFFFFF); r[1] = UInt32(v.0 >> 32)
        r[2] = UInt32(v.1 & 0xFFFFFFFF); r[3] = UInt32(v.1 >> 32)
        r[4] = UInt32(v.2 & 0xFFFFFFFF); r[5] = UInt32(v.2 >> 32)
        r[6] = UInt32(v.3 & 0xFFFFFFFF); r[7] = UInt32(v.3 >> 32)
        return r
    }
}
