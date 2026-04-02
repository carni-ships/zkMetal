// GLV Endomorphism for BN254
// Scalar decomposition: k -> (k1, k2) where k = k1 + k2*lambda (mod r)
// Endomorphism: phi(P) = (beta*x, y)

import Foundation

// BN254 scalar field order r
public let FR_ORDER: [UInt64] = [
    0x43e1f593f0000001, 0x2833e84879b97091,
    0xb85045b68181585d, 0x30644e72e131a029
]

// Cube root of unity β in base field Fp (Montgomery form)
public let FP_BETA = Fp.from64([
    0x71930c11d782e155, 0xa6bb947cffbe3323,
    0xaa303344d4741444, 0x2c3b3f0d26594943
])

// GLV lattice constants for scalar decomposition
let GLV_A1: UInt64 = 0x89d3256894d213e3
let GLV_MINUS_B1: (UInt64, UInt64) = (0x8211bbeb7d4f1128, 0x6f4d8248eeb859fc)
let GLV_A2: (UInt64, UInt64) = (0x0be4e1541221250b, 0x6f4d8248eeb859fd)
let GLV_B2: UInt64 = 0x89d3256894d213e3
let GLV_G1: (UInt64, UInt64, UInt64) = (0x7a7bd9d4391eb18d, 0x4ccef014a773d2cf, 0x2)
let GLV_G2: (UInt64, UInt64) = (0xd91d232ec7e0b3d7, 0x2)

let FR_ORDER_T: U256 = (FR_ORDER[0], FR_ORDER[1], FR_ORDER[2], FR_ORDER[3])
let HALF_R_T: U256 = (
    FR_ORDER[0] >> 1 | (FR_ORDER[1] << 63),
    FR_ORDER[1] >> 1 | (FR_ORDER[2] << 63),
    FR_ORDER[2] >> 1 | (FR_ORDER[3] << 63),
    FR_ORDER[3] >> 1
)

/// Decompose a 256-bit scalar k into (k1, k2) such that k ≡ k1 + k2·λ (mod r)
/// where |k1|, |k2| < 2^128. Writes directly into output pointers (zero allocation).
public func glvDecompose(_ scalar: UnsafePointer<UInt32>,
                  k1Out: UnsafeMutablePointer<UInt32>,
                  k2Out: UnsafeMutablePointer<UInt32>) -> (neg1: Bool, neg2: Bool) {
    var kr: U256 = (
        UInt64(scalar[0]) | (UInt64(scalar[1]) << 32),
        UInt64(scalar[2]) | (UInt64(scalar[3]) << 32),
        UInt64(scalar[4]) | (UInt64(scalar[5]) << 32),
        UInt64(scalar[6]) | (UInt64(scalar[7]) << 32)
    )

    while gte256t(kr, FR_ORDER_T) {
        (kr, _) = sub256t(kr, FR_ORDER_T)
    }

    let c1 = mulScalarByG1(kr)
    let c2 = mulScalarByG2(kr)

    let (c2a1_hi, c2a1_lo) = c2.multipliedFullWidth(by: GLV_A1)
    let c1a2 = mul128x128(c1, (GLV_A2.0, GLV_A2.1))

    var borrow: Bool
    var k1: U256
    (k1, borrow) = sub256t(kr, (c2a1_lo, c2a1_hi, 0, 0))
    if borrow { (k1, _) = add256t(k1, FR_ORDER_T) }
    (k1, borrow) = sub256t(k1, c1a2)
    if borrow { (k1, _) = add256t(k1, FR_ORDER_T) }

    let c2mb1 = mul64x128_inline(c2, (GLV_MINUS_B1.0, GLV_MINUS_B1.1))
    let c1b2 = mul128x64(c1, GLV_B2)

    var k2_neg = false
    var k2: U256
    (k2, borrow) = sub256t((c2mb1.0, c2mb1.1, c2mb1.2, 0), (c1b2.0, c1b2.1, c1b2.2, 0))
    if borrow {
        k2_neg = true
        k2 = neg256t(k2)
    }

    var neg1 = false
    if gte256t(k1, HALF_R_T) {
        (k1, _) = sub256t(FR_ORDER_T, k1)
        neg1 = true
    }

    k1Out[0] = UInt32(k1.0 & 0xFFFFFFFF); k1Out[1] = UInt32(k1.0 >> 32)
    k1Out[2] = UInt32(k1.1 & 0xFFFFFFFF); k1Out[3] = UInt32(k1.1 >> 32)
    k1Out[4] = UInt32(k1.2 & 0xFFFFFFFF); k1Out[5] = UInt32(k1.2 >> 32)
    k1Out[6] = UInt32(k1.3 & 0xFFFFFFFF); k1Out[7] = UInt32(k1.3 >> 32)

    k2Out[0] = UInt32(k2.0 & 0xFFFFFFFF); k2Out[1] = UInt32(k2.0 >> 32)
    k2Out[2] = UInt32(k2.1 & 0xFFFFFFFF); k2Out[3] = UInt32(k2.1 >> 32)
    k2Out[4] = UInt32(k2.2 & 0xFFFFFFFF); k2Out[5] = UInt32(k2.2 >> 32)
    k2Out[6] = UInt32(k2.3 & 0xFFFFFFFF); k2Out[7] = UInt32(k2.3 >> 32)

    return (neg1, k2_neg)
}

// Legacy wrapper for GLV test
public func glvDecompose(_ scalar: [UInt32]) -> (k1: [UInt32], neg1: Bool, k2: [UInt32], neg2: Bool) {
    var k1 = [UInt32](repeating: 0, count: 8)
    var k2 = [UInt32](repeating: 0, count: 8)
    let (neg1, neg2) = scalar.withUnsafeBufferPointer { sp in
        k1.withUnsafeMutableBufferPointer { k1p in
            k2.withUnsafeMutableBufferPointer { k2p in
                glvDecompose(sp.baseAddress!, k1Out: k1p.baseAddress!, k2Out: k2p.baseAddress!)
            }
        }
    }
    return (k1, neg1, k2, neg2)
}

// Apply the BN254 endomorphism: φ(x, y) = (β·x, y)
public func applyEndomorphism(_ p: PointAffine) -> PointAffine {
    return PointAffine(x: fpMul(FP_BETA, p.x), y: p.y)
}

// MARK: - Wide Multiply Helpers

@inline(__always)
func u256at(_ v: U256, _ i: Int) -> UInt64 {
    switch i { case 0: return v.0; case 1: return v.1; case 2: return v.2; default: return v.3 }
}

func mulScalarByG1(_ k: U256) -> (UInt64, UInt64) {
    let g = (GLV_G1.0, GLV_G1.1, GLV_G1.2)
    return withUnsafeTemporaryAllocation(of: UInt64.self, capacity: 7) { prod in
        for i in 0..<7 { prod[i] = 0 }
        for i in 0..<4 {
            let ki = u256at(k, i)
            var carry: UInt64 = 0
            for j in 0..<3 {
                let gj = j == 0 ? g.0 : (j == 1 ? g.1 : g.2)
                let (hi, lo) = ki.multipliedFullWidth(by: gj)
                let (s1, c1) = prod[i+j].addingReportingOverflow(lo)
                let (s2, c2) = s1.addingReportingOverflow(carry)
                prod[i+j] = s2
                carry = hi &+ (c1 ? 1 : 0) &+ (c2 ? 1 : 0)
            }
            prod[i+3] &+= carry
        }
        return (prod[4], prod[5])
    }
}

func mulScalarByG2(_ k: U256) -> UInt64 {
    let g = (GLV_G2.0, GLV_G2.1)
    return withUnsafeTemporaryAllocation(of: UInt64.self, capacity: 6) { prod in
        for i in 0..<6 { prod[i] = 0 }
        for i in 0..<4 {
            let ki = u256at(k, i)
            var carry: UInt64 = 0
            for j in 0..<2 {
                let gj = j == 0 ? g.0 : g.1
                let (hi, lo) = ki.multipliedFullWidth(by: gj)
                let (s1, c1) = prod[i+j].addingReportingOverflow(lo)
                let (s2, c2) = s1.addingReportingOverflow(carry)
                prod[i+j] = s2
                carry = hi &+ (c1 ? 1 : 0) &+ (c2 ? 1 : 0)
            }
            prod[i+2] &+= carry
        }
        return prod[4]
    }
}

@inline(__always)
func mul128x128(_ a: (UInt64, UInt64), _ b: (UInt64, UInt64)) -> (UInt64, UInt64, UInt64, UInt64) {
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

@inline(__always)
func mul64x128_inline(_ a: UInt64, _ b: (UInt64, UInt64)) -> (UInt64, UInt64, UInt64) {
    let (h0, l0) = a.multipliedFullWidth(by: b.0)
    let (h1, l1) = a.multipliedFullWidth(by: b.1)
    let (s1, c1) = l1.addingReportingOverflow(h0)
    return (l0, s1, h1 &+ (c1 ? 1 : 0))
}

@inline(__always)
func mul128x64(_ a: (UInt64, UInt64), _ b: UInt64) -> (UInt64, UInt64, UInt64) {
    let (h0, l0) = a.0.multipliedFullWidth(by: b)
    let (h1, l1) = a.1.multipliedFullWidth(by: b)
    let (s1, c1) = l1.addingReportingOverflow(h0)
    return (l0, s1, h1 &+ (c1 ? 1 : 0))
}

// Schoolbook multiply a[] (na limbs) × b[] (nb limbs) → prod[] (na+nb limbs)
public func mulSchoolbook(_ a: [UInt64], _ na: Int, _ b: [UInt64], _ nb: Int) -> [UInt64] {
    var prod = [UInt64](repeating: 0, count: na + nb)
    for i in 0..<na {
        var carry: UInt64 = 0
        for j in 0..<nb {
            let (hi, lo) = a[i].multipliedFullWidth(by: b[j])
            let (s1, c1) = prod[i+j].addingReportingOverflow(lo)
            let (s2, c2) = s1.addingReportingOverflow(carry)
            prod[i+j] = s2
            carry = hi
            let extra: UInt64 = (c1 ? 1 : 0) &+ (c2 ? 1 : 0)
            let (cn, co) = carry.addingReportingOverflow(extra)
            carry = cn
            if co {
                var idx = i + j + 1
                while idx < na + nb {
                    let (v, o) = prod[idx].addingReportingOverflow(1)
                    prod[idx] = v
                    if !o { break }
                    idx += 1
                }
            }
        }
        prod[i + nb] = prod[i + nb] &+ carry
    }
    return prod
}
