// EVM Precompile Bridge — BN254 (EIP-196/197) + BLS12-381 (EIP-2537)
//
// Implements the exact EVM ABI for Ethereum elliptic curve precompiles,
// backed by zkMetal's C-accelerated CIOS Montgomery field arithmetic.
//
// Encoding conventions:
//   BN254:     coordinates are 32-byte big-endian (256-bit words, zero-padded)
//   BLS12-381: Fp elements are 64-byte big-endian (padded from 48-byte field)
//              Fp2 elements are 128 bytes (c0 || c1, each 64 bytes)
//
// All functions return nil on invalid input (not on curve, wrong length, etc.)

import Foundation
import NeonFieldOps

// MARK: - BN254 Field Constants

/// BN254 base field modulus p
private let BN254_P: [UInt64] = [
    0x3c208c16d87cfd47, 0x97816a916871ca8d,
    0xb85045b68181585d, 0x30644e72e131a029
]

/// BN254 scalar field modulus r
private let BN254_R: [UInt64] = [
    0x43e1f593f0000001, 0x2833e84879b97091,
    0xb85045b68181585d, 0x30644e72e131a029
]

/// Montgomery R^2 mod p for BN254 Fp (for converting integer -> Montgomery)
private let BN254_FP_R2: [UInt64] = [
    0xf32cfc5b538afa89, 0xb5e71911d44501fb,
    0x47ab1eff0a417ff6, 0x06d89f71cab8351f
]

/// Montgomery form of 1 in BN254 Fp
private let BN254_FP_ONE: [UInt64] = [
    0xd35d438dc58f0d9d, 0x0a78eb28f5c70b3d,
    0x666ea36f7879462c, 0x0e0a77c19a07df2f
]

/// BN254 curve: y^2 = x^3 + 3
/// Montgomery form of b=3 in Fp
private let BN254_B_MONT: [UInt64] = [
    0x7a17caa950ad28d7, 0x1f6ac17ae15521b9,
    0x334bea4e696bd284, 0x2a1f6744ce179d8e
]

// BN254 G2 twist: y^2 = x^3 + 3/(9+u) = x^3 + b'
// b' = 3 * (9+u)^{-1} in Fp2, where (9+u)^{-1} = (9-u)/82
// b'_c0 (real) and b'_c1 (imaginary, coefficient of u) in Montgomery form
private let BN254_G2_B_C0: [UInt64] = [
    0x3bf938e377b802a8, 0x020b1b273633535d,
    0x26b7edf049755260, 0x2514c6324384a86d
]
private let BN254_G2_B_C1: [UInt64] = [
    0x38e7ecccd1dcff67, 0x65f0b37d93ce0d3e,
    0xd749d0dd22ac00aa, 0x0141b9ce4a688d4d
]

// MARK: - BLS12-381 Field Constants

/// BLS12-381 base field modulus
private let BLS12_381_P: [UInt64] = [
    0xb9feffffffffaaab, 0x1eabfffeb153ffff,
    0x6730d2a0f6b0f624, 0x64774b84f38512bf,
    0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a
]

/// Montgomery R^2 mod p for BLS12-381 Fp
private let BLS12_381_FP_R2: [UInt64] = [
    0xf4df1f341c341746, 0x0a76e6a609d104f1,
    0x8de5476c4c95b6d5, 0x67eb88a9939d83c0,
    0x9a793e85b519952d, 0x11988fe592cae3aa
]

/// Montgomery form of 1 in BLS12-381 Fp
private let BLS12_381_FP_ONE: [UInt64] = [
    0x760900000002fffd, 0xebf4000bc40c0002,
    0x5f48985753c758ba, 0x77ce585370525745,
    0x5c071a97a256ec6d, 0x15f65ec3fa80e493
]

/// BLS12-381 curve: y^2 = x^3 + 4
/// Montgomery form of b=4 in Fp
private let BLS12_381_B_MONT: [UInt64] = [
    0xaa270000000cfff3, 0x53cc0032fc34000a,
    0x478fe97a6b0a807f, 0xb1d37ebee6ba24d7,
    0x8ec9733bbf78ab2f, 0x09d645513d83de7e
]

// BLS12-381 G2 twist: y^2 = x^3 + 4*(1+u)
// b' = 4*(1+u) in Fp2 -> b'_c0 = 4, b'_c1 = 4 (both in Montgomery form)
private let BLS12_381_G2_B_C0: [UInt64] = [
    0xaa270000000cfff3, 0x53cc0032fc34000a,
    0x478fe97a6b0a807f, 0xb1d37ebee6ba24d7,
    0x8ec9733bbf78ab2f, 0x09d645513d83de7e
]
private let BLS12_381_G2_B_C1: [UInt64] = [
    0xaa270000000cfff3, 0x53cc0032fc34000a,
    0x478fe97a6b0a807f, 0xb1d37ebee6ba24d7,
    0x8ec9733bbf78ab2f, 0x09d645513d83de7e
]

// MARK: - Byte <-> Limb Conversions

/// Convert 32 big-endian bytes to 4 little-endian uint64 limbs
private func bytes32ToLimbs4(_ data: [UInt8]) -> [UInt64] {
    assert(data.count == 32)
    var limbs = [UInt64](repeating: 0, count: 4)
    for i in 0..<4 {
        var val: UInt64 = 0
        for j in 0..<8 {
            val = (val << 8) | UInt64(data[31 - i * 8 - (7 - j)])
        }
        limbs[i] = val
    }
    return limbs
}

/// Convert 4 little-endian uint64 limbs to 32 big-endian bytes
private func limbs4ToBytes32(_ limbs: [UInt64]) -> [UInt8] {
    var out = [UInt8](repeating: 0, count: 32)
    for i in 0..<4 {
        let val = limbs[i]
        for j in 0..<8 {
            out[31 - i * 8 - j] = UInt8((val >> (j * 8)) & 0xFF)
        }
    }
    return out
}

/// Convert 48 big-endian bytes to 6 little-endian uint64 limbs
private func bytes48ToLimbs6(_ data: [UInt8]) -> [UInt64] {
    assert(data.count == 48)
    var limbs = [UInt64](repeating: 0, count: 6)
    for i in 0..<6 {
        var val: UInt64 = 0
        for j in 0..<8 {
            val = (val << 8) | UInt64(data[47 - i * 8 - (7 - j)])
        }
        limbs[i] = val
    }
    return limbs
}

/// Convert 6 little-endian uint64 limbs to 48 big-endian bytes
private func limbs6ToBytes48(_ limbs: [UInt64]) -> [UInt8] {
    var out = [UInt8](repeating: 0, count: 48)
    for i in 0..<6 {
        let val = limbs[i]
        for j in 0..<8 {
            out[47 - i * 8 - j] = UInt8((val >> (j * 8)) & 0xFF)
        }
    }
    return out
}

/// Convert 64 big-endian bytes (EVM Fp for BLS12-381: 16 leading zero bytes + 48 field bytes)
/// to 6 uint64 limbs. Returns nil if the top 16 bytes are non-zero.
private func bytes64ToFp381Limbs(_ data: [UInt8]) -> [UInt64]? {
    assert(data.count == 64)
    // Top 16 bytes must be zero
    for i in 0..<16 {
        if data[i] != 0 { return nil }
    }
    return bytes48ToLimbs6(Array(data[16..<64]))
}

/// Convert 6 uint64 limbs to 64 big-endian bytes (16 zero padding + 48 field bytes)
private func fp381LimbsToBytes64(_ limbs: [UInt64]) -> [UInt8] {
    var out = [UInt8](repeating: 0, count: 64)
    let raw = limbs6ToBytes48(limbs)
    out[16..<64] = raw[0..<48]
    return out
}

// MARK: - Montgomery Conversions

/// Convert integer limbs to Montgomery form: mont = integer * R mod p
/// Computed as CIOS(integer, R^2) since CIOS(a, b) = a * b * R^{-1} mod p
private func bn254FpToMont(_ integer: [UInt64]) -> [UInt64] {
    return swiftBN254FpMul(integer, BN254_FP_R2)
}

/// Convert from Montgomery form to integer: integer = mont * 1 mod p
private func bn254FpFromMont(_ mont: [UInt64]) -> [UInt64] {
    let one: [UInt64] = [1, 0, 0, 0]
    return swiftBN254FpMul(mont, one)
}

/// Swift implementation of CIOS Montgomery multiplication for BN254 Fp (4-limb)
private func swiftBN254FpMul(_ a: [UInt64], _ b: [UInt64]) -> [UInt64] {
    let pInv: UInt64 = 0x87d20782e4866389
    var t0: UInt64 = 0, t1: UInt64 = 0, t2: UInt64 = 0, t3: UInt64 = 0, t4: UInt64 = 0

    for i in 0..<4 {
        var c: UInt64 = 0
        var w: (high: UInt64, low: UInt64)

        w = a[i].multipliedFullWidth(by: b[0])
        var sum = UInt64.addingFullWidth(w.low, t0)
        t0 = sum.low; c = sum.high &+ w.high

        w = a[i].multipliedFullWidth(by: b[1])
        sum = UInt64.addingFullWidth(w.low, t1)
        let sum2 = UInt64.addingFullWidth(sum.low, c)
        t1 = sum2.low; c = sum.high &+ sum2.high &+ w.high

        w = a[i].multipliedFullWidth(by: b[2])
        sum = UInt64.addingFullWidth(w.low, t2)
        let sum3 = UInt64.addingFullWidth(sum.low, c)
        t2 = sum3.low; c = sum.high &+ sum3.high &+ w.high

        w = a[i].multipliedFullWidth(by: b[3])
        sum = UInt64.addingFullWidth(w.low, t3)
        let sum4 = UInt64.addingFullWidth(sum.low, c)
        t3 = sum4.low; c = sum.high &+ sum4.high &+ w.high

        t4 = t4 &+ c

        let m = t0 &* pInv
        w = m.multipliedFullWidth(by: BN254_P[0])
        sum = UInt64.addingFullWidth(w.low, t0)
        c = sum.high &+ w.high

        w = m.multipliedFullWidth(by: BN254_P[1])
        sum = UInt64.addingFullWidth(w.low, t1)
        let s5 = UInt64.addingFullWidth(sum.low, c)
        t0 = s5.low; c = sum.high &+ s5.high &+ w.high

        w = m.multipliedFullWidth(by: BN254_P[2])
        sum = UInt64.addingFullWidth(w.low, t2)
        let s6 = UInt64.addingFullWidth(sum.low, c)
        t1 = s6.low; c = sum.high &+ s6.high &+ w.high

        w = m.multipliedFullWidth(by: BN254_P[3])
        sum = UInt64.addingFullWidth(w.low, t3)
        let s7 = UInt64.addingFullWidth(sum.low, c)
        t2 = s7.low; c = sum.high &+ s7.high &+ w.high

        t3 = t4 &+ c
        t4 = 0
    }

    // Conditional subtraction
    var borrow: UInt64 = 0
    var tmp = [UInt64](repeating: 0, count: 4)
    (tmp[0], borrow) = t0.subtractingWithBorrow(BN254_P[0], borrow: 0)
    (tmp[1], borrow) = t1.subtractingWithBorrow(BN254_P[1], borrow: borrow)
    (tmp[2], borrow) = t2.subtractingWithBorrow(BN254_P[2], borrow: borrow)
    (tmp[3], borrow) = t3.subtractingWithBorrow(BN254_P[3], borrow: borrow)

    if borrow == 0 {
        return tmp
    } else {
        return [t0, t1, t2, t3]
    }
}

/// Swift CIOS Montgomery multiplication for BLS12-381 Fp (6-limb)
private func swiftBLS12FpMul(_ a: [UInt64], _ b: [UInt64]) -> [UInt64] {
    let pInv: UInt64 = 0x89f3fffcfffcfffd
    var t = [UInt64](repeating: 0, count: 7)

    for i in 0..<6 {
        // Multiply: t += a[i] * b
        var carry: UInt64 = 0
        for j in 0..<6 {
            let w = a[i].multipliedFullWidth(by: b[j])
            let s1 = UInt64.addingFullWidth(w.low, t[j])
            let s2 = UInt64.addingFullWidth(s1.low, carry)
            t[j] = s2.low
            carry = w.high &+ s1.high &+ s2.high
        }
        t[6] = t[6] &+ carry

        // Reduce
        let m = t[0] &* pInv
        let w0 = m.multipliedFullWidth(by: BLS12_381_P[0])
        let s0 = UInt64.addingFullWidth(w0.low, t[0])
        carry = w0.high &+ s0.high
        for j in 1..<6 {
            let w = m.multipliedFullWidth(by: BLS12_381_P[j])
            let s1 = UInt64.addingFullWidth(w.low, t[j])
            let s2 = UInt64.addingFullWidth(s1.low, carry)
            t[j-1] = s2.low
            carry = w.high &+ s1.high &+ s2.high
        }
        t[5] = t[6] &+ carry
        t[6] = 0
    }

    // Conditional subtraction
    var borrow: UInt64 = 0
    var tmp = [UInt64](repeating: 0, count: 6)
    for i in 0..<6 {
        (tmp[i], borrow) = t[i].subtractingWithBorrow(BLS12_381_P[i], borrow: borrow)
    }
    return borrow == 0 ? tmp : Array(t[0..<6])
}

/// Convert integer to BLS12-381 Fp Montgomery form
private func bls12FpToMont(_ integer: [UInt64]) -> [UInt64] {
    return swiftBLS12FpMul(integer, BLS12_381_FP_R2)
}

/// Convert BLS12-381 Fp Montgomery form to integer
private func bls12FpFromMont(_ mont: [UInt64]) -> [UInt64] {
    let one: [UInt64] = [1, 0, 0, 0, 0, 0]
    return swiftBLS12FpMul(mont, one)
}

// MARK: - UInt64 Arithmetic Helpers

private extension UInt64 {
    /// Returns (low, high) of full-width addition: a + b
    static func addingFullWidth(_ a: UInt64, _ b: UInt64) -> (low: UInt64, high: UInt64) {
        let (low, overflow) = a.addingReportingOverflow(b)
        return (low, overflow ? 1 : 0)
    }

    /// Returns (result, borrow) of a - b - borrow_in
    func subtractingWithBorrow(_ rhs: UInt64, borrow: UInt64) -> (UInt64, UInt64) {
        let (r1, b1) = self.subtractingReportingOverflow(rhs)
        let (r2, b2) = r1.subtractingReportingOverflow(borrow)
        return (r2, (b1 ? 1 : 0) + (b2 ? 1 : 0))
    }
}

// MARK: - Field Comparison

/// Check if limbs represent a value >= modulus (for BN254 Fp, 4 limbs)
private func bn254FpGEP(_ a: [UInt64]) -> Bool {
    for i in stride(from: 3, through: 0, by: -1) {
        if a[i] < BN254_P[i] { return false }
        if a[i] > BN254_P[i] { return true }
    }
    return true // equal
}

/// Check if limbs represent a value >= modulus (for BLS12-381 Fp, 6 limbs)
private func bls12FpGEP(_ a: [UInt64]) -> Bool {
    for i in stride(from: 5, through: 0, by: -1) {
        if a[i] < BLS12_381_P[i] { return false }
        if a[i] > BLS12_381_P[i] { return true }
    }
    return true
}

/// Check if all limbs are zero
private func isZero4(_ a: [UInt64]) -> Bool {
    return a[0] == 0 && a[1] == 0 && a[2] == 0 && a[3] == 0
}

private func isZero6(_ a: [UInt64]) -> Bool {
    return a[0] == 0 && a[1] == 0 && a[2] == 0 && a[3] == 0 && a[4] == 0 && a[5] == 0
}

// MARK: - Point on Curve Checks

/// Check if (x, y) in Montgomery form is on BN254: y^2 = x^3 + 3
private func bn254IsOnCurve(_ xMont: [UInt64], _ yMont: [UInt64]) -> Bool {
    let y2 = swiftBN254FpMul(yMont, yMont)
    let x2 = swiftBN254FpMul(xMont, xMont)
    let x3 = swiftBN254FpMul(x2, xMont)
    let rhs = bn254FpAdd(x3, BN254_B_MONT)
    return y2 == rhs
}

/// BN254 Fp addition: (a + b) mod p
private func bn254FpAdd(_ a: [UInt64], _ b: [UInt64]) -> [UInt64] {
    var r = [UInt64](repeating: 0, count: 4)
    var carry: UInt64 = 0
    for i in 0..<4 {
        let sum = UInt64.addingFullWidth(a[i], b[i])
        let sum2 = UInt64.addingFullWidth(sum.low, carry)
        r[i] = sum2.low
        carry = sum.high + sum2.high
    }
    // Conditional subtraction
    var borrow: UInt64 = 0
    var tmp = [UInt64](repeating: 0, count: 4)
    for i in 0..<4 {
        (tmp[i], borrow) = r[i].subtractingWithBorrow(BN254_P[i], borrow: borrow)
    }
    if carry > 0 || borrow == 0 {
        return tmp
    }
    return r
}

/// BN254 Fp2 multiplication: (a0+a1*u)(b0+b1*u) = (a0*b0 - a1*b1) + (a0*b1 + a1*b0)*u
/// where u^2 = -1
private func bn254Fp2Mul(_ a: ([UInt64], [UInt64]), _ b: ([UInt64], [UInt64])) -> ([UInt64], [UInt64]) {
    let a0b0 = swiftBN254FpMul(a.0, b.0)
    let a1b1 = swiftBN254FpMul(a.1, b.1)
    let a0b1 = swiftBN254FpMul(a.0, b.1)
    let a1b0 = swiftBN254FpMul(a.1, b.0)
    let c0 = bn254FpSub(a0b0, a1b1)
    let c1 = bn254FpAdd(a0b1, a1b0)
    return (c0, c1)
}

/// BN254 Fp2 squaring
private func bn254Fp2Sqr(_ a: ([UInt64], [UInt64])) -> ([UInt64], [UInt64]) {
    return bn254Fp2Mul(a, a)
}

/// BN254 Fp subtraction
private func bn254FpSub(_ a: [UInt64], _ b: [UInt64]) -> [UInt64] {
    var r = [UInt64](repeating: 0, count: 4)
    var borrow: UInt64 = 0
    for i in 0..<4 {
        (r[i], borrow) = a[i].subtractingWithBorrow(b[i], borrow: borrow)
    }
    if borrow > 0 {
        var carry: UInt64 = 0
        for i in 0..<4 {
            let sum = UInt64.addingFullWidth(r[i], BN254_P[i])
            let sum2 = UInt64.addingFullWidth(sum.low, carry)
            r[i] = sum2.low
            carry = sum.high + sum2.high
        }
    }
    return r
}

/// Check if (x, y) as Fp2 (Montgomery) is on BN254 G2 twist: y^2 = x^3 + b'
private func bn254G2IsOnCurve(_ x: ([UInt64], [UInt64]), _ y: ([UInt64], [UInt64])) -> Bool {
    let y2 = bn254Fp2Sqr(y)
    let x2 = bn254Fp2Mul(x, x)
    let x3 = bn254Fp2Mul(x2, x)
    let bPrime = (BN254_G2_B_C0, BN254_G2_B_C1)
    let rhs = (bn254FpAdd(x3.0, bPrime.0), bn254FpAdd(x3.1, bPrime.1))
    return y2.0 == rhs.0 && y2.1 == rhs.1
}

/// Check if (x, y) in Montgomery form is on BLS12-381 G1: y^2 = x^3 + 4
private func bls12G1IsOnCurve(_ xMont: [UInt64], _ yMont: [UInt64]) -> Bool {
    let y2 = swiftBLS12FpMul(yMont, yMont)
    let x2 = swiftBLS12FpMul(xMont, xMont)
    let x3 = swiftBLS12FpMul(x2, xMont)
    let rhs = bls12FpAdd(x3, BLS12_381_B_MONT)
    return y2 == rhs
}

/// BLS12-381 Fp addition
private func bls12FpAdd(_ a: [UInt64], _ b: [UInt64]) -> [UInt64] {
    var r = [UInt64](repeating: 0, count: 6)
    var carry: UInt64 = 0
    for i in 0..<6 {
        let sum = UInt64.addingFullWidth(a[i], b[i])
        let sum2 = UInt64.addingFullWidth(sum.low, carry)
        r[i] = sum2.low
        carry = sum.high + sum2.high
    }
    var borrow: UInt64 = 0
    var tmp = [UInt64](repeating: 0, count: 6)
    for i in 0..<6 {
        (tmp[i], borrow) = r[i].subtractingWithBorrow(BLS12_381_P[i], borrow: borrow)
    }
    if carry > 0 || borrow == 0 {
        return tmp
    }
    return r
}

/// BLS12-381 Fp subtraction
private func bls12FpSub(_ a: [UInt64], _ b: [UInt64]) -> [UInt64] {
    var r = [UInt64](repeating: 0, count: 6)
    var borrow: UInt64 = 0
    for i in 0..<6 {
        (r[i], borrow) = a[i].subtractingWithBorrow(b[i], borrow: borrow)
    }
    if borrow > 0 {
        var carry: UInt64 = 0
        for i in 0..<6 {
            let sum = UInt64.addingFullWidth(r[i], BLS12_381_P[i])
            let sum2 = UInt64.addingFullWidth(sum.low, carry)
            r[i] = sum2.low
            carry = sum.high + sum2.high
        }
    }
    return r
}

/// BLS12-381 Fp2 multiplication: u^2 = -1
private func bls12Fp2Mul(_ a: ([UInt64], [UInt64]), _ b: ([UInt64], [UInt64])) -> ([UInt64], [UInt64]) {
    let a0b0 = swiftBLS12FpMul(a.0, b.0)
    let a1b1 = swiftBLS12FpMul(a.1, b.1)
    let a0b1 = swiftBLS12FpMul(a.0, b.1)
    let a1b0 = swiftBLS12FpMul(a.1, b.0)
    let c0 = bls12FpSub(a0b0, a1b1)
    let c1 = bls12FpAdd(a0b1, a1b0)
    return (c0, c1)
}

private func bls12Fp2Sqr(_ a: ([UInt64], [UInt64])) -> ([UInt64], [UInt64]) {
    return bls12Fp2Mul(a, a)
}

/// Check if (x, y) as Fp2 is on BLS12-381 G2: y^2 = x^3 + 4*(1+u)
private func bls12G2IsOnCurve(_ x: ([UInt64], [UInt64]), _ y: ([UInt64], [UInt64])) -> Bool {
    let y2 = bls12Fp2Sqr(y)
    let x2 = bls12Fp2Mul(x, x)
    let x3 = bls12Fp2Mul(x2, x)
    let bPrime = (BLS12_381_G2_B_C0, BLS12_381_G2_B_C1)
    let rhs_c0 = bls12FpAdd(x3.0, bPrime.0)
    let rhs_c1 = bls12FpAdd(x3.1, bPrime.1)
    return y2.0 == rhs_c0 && y2.1 == rhs_c1
}

// MARK: - EVM Precompile 0x06: BN254 ecAdd (EIP-196)

/// BN254 point addition. Input: 128 bytes (two uncompressed affine points).
/// Output: 64 bytes (result affine point). Returns nil on invalid input.
public func EVMPrecompile06_ecAdd(input: [UInt8]) -> [UInt8]? {
    // Pad input to 128 bytes (EVM pads with zeros)
    var data = input
    if data.count < 128 {
        data.append(contentsOf: [UInt8](repeating: 0, count: 128 - data.count))
    }

    // Parse point 1: (x1, y1)
    let x1Int = bytes32ToLimbs4(Array(data[0..<32]))
    let y1Int = bytes32ToLimbs4(Array(data[32..<64]))
    // Parse point 2: (x2, y2)
    let x2Int = bytes32ToLimbs4(Array(data[64..<96]))
    let y2Int = bytes32ToLimbs4(Array(data[96..<128]))

    // Validate: coordinates must be < p
    if bn254FpGEP(x1Int) || bn254FpGEP(y1Int) { return nil }
    if bn254FpGEP(x2Int) || bn254FpGEP(y2Int) { return nil }

    let p1IsInfinity = isZero4(x1Int) && isZero4(y1Int)
    let p2IsInfinity = isZero4(x2Int) && isZero4(y2Int)

    // Convert to Montgomery
    let x1Mont = bn254FpToMont(x1Int)
    let y1Mont = bn254FpToMont(y1Int)
    let x2Mont = bn254FpToMont(x2Int)
    let y2Mont = bn254FpToMont(y2Int)

    // Point-on-curve check (skip for infinity)
    if !p1IsInfinity && !bn254IsOnCurve(x1Mont, y1Mont) { return nil }
    if !p2IsInfinity && !bn254IsOnCurve(x2Mont, y2Mont) { return nil }

    // Handle infinity cases
    if p1IsInfinity && p2IsInfinity {
        return [UInt8](repeating: 0, count: 64)
    }
    if p1IsInfinity {
        return limbs4ToBytes32(x2Int) + limbs4ToBytes32(y2Int)
    }
    if p2IsInfinity {
        return limbs4ToBytes32(x1Int) + limbs4ToBytes32(y1Int)
    }

    // Build projective points (Montgomery form): [x, y, z=1]
    var p1 = x1Mont + y1Mont + BN254_FP_ONE
    var p2 = x2Mont + y2Mont + BN254_FP_ONE

    // Call C point_add
    var result = [UInt64](repeating: 0, count: 12)
    p1.withUnsafeBufferPointer { p1P in
        p2.withUnsafeBufferPointer { p2P in
            result.withUnsafeMutableBufferPointer { rP in
                bn254_point_add(p1P.baseAddress!, p2P.baseAddress!, rP.baseAddress!)
            }
        }
    }

    // Convert projective -> affine
    var affine = [UInt64](repeating: 0, count: 8)
    result.withUnsafeBufferPointer { rP in
        affine.withUnsafeMutableBufferPointer { aP in
            bn254_projective_to_affine(rP.baseAddress!, aP.baseAddress!)
        }
    }

    // Check for identity (z was zero -> affine returns (0,0))
    let xAff = Array(affine[0..<4])
    let yAff = Array(affine[4..<8])

    // Convert from Montgomery to integer
    let xOut = bn254FpFromMont(xAff)
    let yOut = bn254FpFromMont(yAff)

    return limbs4ToBytes32(xOut) + limbs4ToBytes32(yOut)
}

// MARK: - EVM Precompile 0x07: BN254 ecMul (EIP-196)

/// BN254 scalar multiplication. Input: 96 bytes (point + scalar).
/// Output: 64 bytes (result affine point). Returns nil on invalid input.
public func EVMPrecompile07_ecMul(input: [UInt8]) -> [UInt8]? {
    var data = input
    if data.count < 96 {
        data.append(contentsOf: [UInt8](repeating: 0, count: 96 - data.count))
    }

    let xInt = bytes32ToLimbs4(Array(data[0..<32]))
    let yInt = bytes32ToLimbs4(Array(data[32..<64]))
    let sInt = bytes32ToLimbs4(Array(data[64..<96]))

    // Validate coordinate < p
    if bn254FpGEP(xInt) || bn254FpGEP(yInt) { return nil }

    let isInfinity = isZero4(xInt) && isZero4(yInt)

    let xMont = bn254FpToMont(xInt)
    let yMont = bn254FpToMont(yInt)

    if !isInfinity && !bn254IsOnCurve(xMont, yMont) { return nil }

    // Scalar = 0 or point at infinity -> return infinity
    if isInfinity || isZero4(sInt) {
        return [UInt8](repeating: 0, count: 64)
    }

    // Build projective point
    var proj = xMont + yMont + BN254_FP_ONE

    // Convert scalar to uint32 limbs (the C function expects 8 x uint32, little-endian)
    var scalar = [UInt32](repeating: 0, count: 8)
    for i in 0..<4 {
        scalar[i * 2] = UInt32(sInt[i] & 0xFFFFFFFF)
        scalar[i * 2 + 1] = UInt32(sInt[i] >> 32)
    }

    var result = [UInt64](repeating: 0, count: 12)
    proj.withUnsafeBufferPointer { pP in
        scalar.withUnsafeBufferPointer { sP in
            result.withUnsafeMutableBufferPointer { rP in
                bn254_point_scalar_mul(pP.baseAddress!, sP.baseAddress!, rP.baseAddress!)
            }
        }
    }

    // Projective -> affine
    var affine = [UInt64](repeating: 0, count: 8)
    result.withUnsafeBufferPointer { rP in
        affine.withUnsafeMutableBufferPointer { aP in
            bn254_projective_to_affine(rP.baseAddress!, aP.baseAddress!)
        }
    }

    let xOut = bn254FpFromMont(Array(affine[0..<4]))
    let yOut = bn254FpFromMont(Array(affine[4..<8]))

    return limbs4ToBytes32(xOut) + limbs4ToBytes32(yOut)
}

// MARK: - EVM Precompile 0x08: BN254 ecPairing (EIP-197)

/// BN254 pairing check. Input: N*192 bytes. Output: 32 bytes (0x00..01 or 0x00..00).
/// Returns nil on invalid input.
public func EVMPrecompile08_ecPairing(input: [UInt8]) -> [UInt8]? {
    // Empty input is valid: trivial pairing = 1
    if input.isEmpty {
        var result = [UInt8](repeating: 0, count: 32)
        result[31] = 1
        return result
    }

    // Input must be multiple of 192 bytes
    if input.count % 192 != 0 { return nil }
    let n = input.count / 192

    // Parse and validate all points, collecting non-trivial pairs.
    // Skip pairs where either G1 or G2 is the point at infinity (e(O, Q) = e(P, O) = 1).
    // C format: interleaved [g1_affine[8], g2_affine[16]] per pair (24 uint64 per pair)
    var validPairs = [UInt64]()

    for i in 0..<n {
        let off = i * 192

        // G1 point: bytes [off..off+64)
        let x1Int = bytes32ToLimbs4(Array(input[off..<off+32]))
        let y1Int = bytes32ToLimbs4(Array(input[off+32..<off+64]))

        if bn254FpGEP(x1Int) || bn254FpGEP(y1Int) { return nil }

        let p1IsInf = isZero4(x1Int) && isZero4(y1Int)

        if !p1IsInf {
            let x1Mont = bn254FpToMont(x1Int)
            let y1Mont = bn254FpToMont(y1Int)
            if !bn254IsOnCurve(x1Mont, y1Mont) { return nil }
        }

        // G2 point: bytes [off+64..off+192)
        // EVM encoding for G2: x_imaginary (32B) || x_real (32B) || y_imaginary (32B) || y_real (32B)
        let x2ImInt = bytes32ToLimbs4(Array(input[off+64..<off+96]))
        let x2ReInt = bytes32ToLimbs4(Array(input[off+96..<off+128]))
        let y2ImInt = bytes32ToLimbs4(Array(input[off+128..<off+160]))
        let y2ReInt = bytes32ToLimbs4(Array(input[off+160..<off+192]))

        if bn254FpGEP(x2ImInt) || bn254FpGEP(x2ReInt) { return nil }
        if bn254FpGEP(y2ImInt) || bn254FpGEP(y2ReInt) { return nil }

        let p2IsInf = isZero4(x2ImInt) && isZero4(x2ReInt) && isZero4(y2ImInt) && isZero4(y2ReInt)

        if !p2IsInf {
            let x2Re = bn254FpToMont(x2ReInt)
            let x2Im = bn254FpToMont(x2ImInt)
            let y2Re = bn254FpToMont(y2ReInt)
            let y2Im = bn254FpToMont(y2ImInt)
            if !bn254G2IsOnCurve((x2Re, x2Im), (y2Re, y2Im)) { return nil }
        }

        // Skip identity pairs: e(O, Q) = e(P, O) = 1, contributes nothing to product
        if p1IsInf || p2IsInf { continue }

        // Both points are non-trivial, add to pairs
        let x1Mont = bn254FpToMont(x1Int)
        let y1Mont = bn254FpToMont(y1Int)
        let x2Re = bn254FpToMont(x2ReInt)
        let x2Im = bn254FpToMont(x2ImInt)
        let y2Re = bn254FpToMont(y2ReInt)
        let y2Im = bn254FpToMont(y2ImInt)

        validPairs.append(contentsOf: x1Mont)
        validPairs.append(contentsOf: y1Mont)
        validPairs.append(contentsOf: x2Re)       // x.c0
        validPairs.append(contentsOf: x2Im)        // x.c1
        validPairs.append(contentsOf: y2Re)        // y.c0
        validPairs.append(contentsOf: y2Im)        // y.c1
    }

    // If all pairs were trivial, product = 1
    let validN = validPairs.count / 24
    if validN == 0 {
        var result = [UInt8](repeating: 0, count: 32)
        result[31] = 1
        return result
    }

    // Call C pairing check
    let checkResult = validPairs.withUnsafeBufferPointer { pP in
        bn254_pairing_check(pP.baseAddress!, Int32(validN))
    }

    var result = [UInt8](repeating: 0, count: 32)
    result[31] = checkResult == 1 ? 1 : 0
    return result
}

// MARK: - EVM Precompile 0x0A: BLS12-381 G1 Add (EIP-2537)

/// BLS12-381 G1 point addition. Input: 256 bytes (two points, each 128 bytes).
/// Each point: x (64 bytes, padded Fp) || y (64 bytes, padded Fp).
/// Output: 128 bytes. Returns nil on invalid input.
public func EVMPrecompile0A_bls12381G1Add(input: [UInt8]) -> [UInt8]? {
    if input.count != 256 { return nil }

    guard let x1Int = bytes64ToFp381Limbs(Array(input[0..<64])),
          let y1Int = bytes64ToFp381Limbs(Array(input[64..<128])),
          let x2Int = bytes64ToFp381Limbs(Array(input[128..<192])),
          let y2Int = bytes64ToFp381Limbs(Array(input[192..<256])) else { return nil }

    if bls12FpGEP(x1Int) || bls12FpGEP(y1Int) { return nil }
    if bls12FpGEP(x2Int) || bls12FpGEP(y2Int) { return nil }

    let p1IsInf = isZero6(x1Int) && isZero6(y1Int)
    let p2IsInf = isZero6(x2Int) && isZero6(y2Int)

    let x1Mont = bls12FpToMont(x1Int)
    let y1Mont = bls12FpToMont(y1Int)
    let x2Mont = bls12FpToMont(x2Int)
    let y2Mont = bls12FpToMont(y2Int)

    if !p1IsInf && !bls12G1IsOnCurve(x1Mont, y1Mont) { return nil }
    if !p2IsInf && !bls12G1IsOnCurve(x2Mont, y2Mont) { return nil }

    if p1IsInf && p2IsInf {
        return [UInt8](repeating: 0, count: 128)
    }
    if p1IsInf {
        return fp381LimbsToBytes64(x2Int) + fp381LimbsToBytes64(y2Int)
    }
    if p2IsInf {
        return fp381LimbsToBytes64(x1Int) + fp381LimbsToBytes64(y1Int)
    }

    // Build projective points
    var p1 = x1Mont + y1Mont + BLS12_381_FP_ONE  // 18 limbs
    var p2 = x2Mont + y2Mont + BLS12_381_FP_ONE

    var result = [UInt64](repeating: 0, count: 18)
    p1.withUnsafeBufferPointer { p1P in
        p2.withUnsafeBufferPointer { p2P in
            result.withUnsafeMutableBufferPointer { rP in
                bls12_381_g1_point_add(p1P.baseAddress!, p2P.baseAddress!, rP.baseAddress!)
            }
        }
    }

    // Projective -> affine manually (no exported function)
    let zMont = Array(result[12..<18])
    if isZero6(zMont) {
        return [UInt8](repeating: 0, count: 128)
    }

    // z_inv = z^(p-2) mod p
    var zInv = [UInt64](repeating: 0, count: 6)
    zMont.withUnsafeBufferPointer { zP in
        zInv.withUnsafeMutableBufferPointer { rP in
            bls12_381_fp_inv_ext(zP.baseAddress!, rP.baseAddress!)
        }
    }

    let zInv2 = swiftBLS12FpMul(zInv, zInv)
    let zInv3 = swiftBLS12FpMul(zInv2, zInv)
    let xAff = swiftBLS12FpMul(Array(result[0..<6]), zInv2)
    let yAff = swiftBLS12FpMul(Array(result[6..<12]), zInv3)

    let xOut = bls12FpFromMont(xAff)
    let yOut = bls12FpFromMont(yAff)

    return fp381LimbsToBytes64(xOut) + fp381LimbsToBytes64(yOut)
}

// MARK: - EVM Precompile 0x0B: BLS12-381 G1 Scalar Mul (EIP-2537)

/// BLS12-381 G1 scalar multiplication. Input: 160 bytes (128 byte point + 32 byte scalar).
/// Output: 128 bytes. Returns nil on invalid input.
public func EVMPrecompile0B_bls12381G1Mul(input: [UInt8]) -> [UInt8]? {
    if input.count != 160 { return nil }

    guard let xInt = bytes64ToFp381Limbs(Array(input[0..<64])),
          let yInt = bytes64ToFp381Limbs(Array(input[64..<128])) else { return nil }

    // Scalar is 32 bytes big-endian (fits in 256 bits)
    let sInt = bytes32ToLimbs4(Array(input[128..<160]))

    if bls12FpGEP(xInt) || bls12FpGEP(yInt) { return nil }

    let isInf = isZero6(xInt) && isZero6(yInt)
    let xMont = bls12FpToMont(xInt)
    let yMont = bls12FpToMont(yInt)

    if !isInf && !bls12G1IsOnCurve(xMont, yMont) { return nil }

    if isInf || isZero4(sInt) {
        return [UInt8](repeating: 0, count: 128)
    }

    // Build projective point
    var proj = xMont + yMont + BLS12_381_FP_ONE
    var scalar = sInt  // bls12_381_g1_scalar_mul expects 4 x uint64 integer form

    var result = [UInt64](repeating: 0, count: 18)
    proj.withUnsafeBufferPointer { pP in
        scalar.withUnsafeBufferPointer { sP in
            result.withUnsafeMutableBufferPointer { rP in
                bls12_381_g1_scalar_mul(pP.baseAddress!, sP.baseAddress!, rP.baseAddress!)
            }
        }
    }

    // Projective -> affine
    let zMont = Array(result[12..<18])
    if isZero6(zMont) {
        return [UInt8](repeating: 0, count: 128)
    }

    var zInv = [UInt64](repeating: 0, count: 6)
    zMont.withUnsafeBufferPointer { zP in
        zInv.withUnsafeMutableBufferPointer { rP in
            bls12_381_fp_inv_ext(zP.baseAddress!, rP.baseAddress!)
        }
    }

    let zInv2 = swiftBLS12FpMul(zInv, zInv)
    let zInv3 = swiftBLS12FpMul(zInv2, zInv)
    let xAff = swiftBLS12FpMul(Array(result[0..<6]), zInv2)
    let yAff = swiftBLS12FpMul(Array(result[6..<12]), zInv3)

    let xOut = bls12FpFromMont(xAff)
    let yOut = bls12FpFromMont(yAff)

    return fp381LimbsToBytes64(xOut) + fp381LimbsToBytes64(yOut)
}

// MARK: - EVM Precompile 0x10: BLS12-381 Pairing Check (EIP-2537)

/// BLS12-381 pairing check. Input: N*384 bytes.
/// Each pair: G1 (128 bytes) || G2 (256 bytes).
/// G2 encoding: x_c0 (64B) || x_c1 (64B) || y_c0 (64B) || y_c1 (64B)
/// Output: 32 bytes (0 or 1). Returns nil on invalid input.
public func EVMPrecompile10_bls12381Pairing(input: [UInt8]) -> [UInt8]? {
    if input.isEmpty {
        var result = [UInt8](repeating: 0, count: 32)
        result[31] = 1
        return result
    }

    if input.count % 384 != 0 { return nil }
    let n = input.count / 384

    // Parse, validate, and collect non-trivial pairs.
    // Skip pairs where either G1 or G2 is identity (e(O,Q) = e(P,O) = 1).
    // C format: interleaved [g1_affine[12], g2_affine[24]] = 36 uint64 per pair
    var validPairs = [UInt64]()

    for i in 0..<n {
        let off = i * 384

        // G1: x (64B) || y (64B)
        guard let x1Int = bytes64ToFp381Limbs(Array(input[off..<off+64])),
              let y1Int = bytes64ToFp381Limbs(Array(input[off+64..<off+128])) else { return nil }

        if bls12FpGEP(x1Int) || bls12FpGEP(y1Int) { return nil }

        let p1IsInf = isZero6(x1Int) && isZero6(y1Int)

        if !p1IsInf {
            let x1Mont = bls12FpToMont(x1Int)
            let y1Mont = bls12FpToMont(y1Int)
            if !bls12G1IsOnCurve(x1Mont, y1Mont) { return nil }
        }

        // G2: x_c0 (64B) || x_c1 (64B) || y_c0 (64B) || y_c1 (64B)
        guard let x2c0Int = bytes64ToFp381Limbs(Array(input[off+128..<off+192])),
              let x2c1Int = bytes64ToFp381Limbs(Array(input[off+192..<off+256])),
              let y2c0Int = bytes64ToFp381Limbs(Array(input[off+256..<off+320])),
              let y2c1Int = bytes64ToFp381Limbs(Array(input[off+320..<off+384])) else { return nil }

        if bls12FpGEP(x2c0Int) || bls12FpGEP(x2c1Int) { return nil }
        if bls12FpGEP(y2c0Int) || bls12FpGEP(y2c1Int) { return nil }

        let p2IsInf = isZero6(x2c0Int) && isZero6(x2c1Int) && isZero6(y2c0Int) && isZero6(y2c1Int)

        if !p2IsInf {
            let x2c0 = bls12FpToMont(x2c0Int)
            let x2c1 = bls12FpToMont(x2c1Int)
            let y2c0 = bls12FpToMont(y2c0Int)
            let y2c1 = bls12FpToMont(y2c1Int)
            if !bls12G2IsOnCurve((x2c0, x2c1), (y2c0, y2c1)) { return nil }
        }

        // Skip identity pairs
        if p1IsInf || p2IsInf { continue }

        // Both non-trivial, add to pairs
        let x1Mont = bls12FpToMont(x1Int)
        let y1Mont = bls12FpToMont(y1Int)
        let x2c0 = bls12FpToMont(x2c0Int)
        let x2c1 = bls12FpToMont(x2c1Int)
        let y2c0 = bls12FpToMont(y2c0Int)
        let y2c1 = bls12FpToMont(y2c1Int)

        validPairs.append(contentsOf: x1Mont)
        validPairs.append(contentsOf: y1Mont)
        validPairs.append(contentsOf: x2c0)
        validPairs.append(contentsOf: x2c1)
        validPairs.append(contentsOf: y2c0)
        validPairs.append(contentsOf: y2c1)
    }

    let validN = validPairs.count / 36
    if validN == 0 {
        var result = [UInt8](repeating: 0, count: 32)
        result[31] = 1
        return result
    }

    let checkResult = validPairs.withUnsafeBufferPointer { pP in
        bls12_381_pairing_check(pP.baseAddress!, Int32(validN))
    }

    var result = [UInt8](repeating: 0, count: 32)
    result[31] = checkResult == 1 ? 1 : 0
    return result
}

// MARK: - EVM Precompile 0x0C: BLS12-381 G2 Scalar Mul (EIP-2537)

/// BLS12-381 G2 scalar multiplication. Input: 256 bytes (192 byte G2 point + 32 byte scalar).
/// G2 point is encoded as: x_c0 (64B) || x_c1 (64B) || y_c0 (64B) || y_c1 (64B)
/// Output: 256 bytes (G2 point). Returns nil on invalid input.
public func EVMPrecompile0C_bls12381G2Mul(input: [UInt8]) -> [UInt8]? {
    if input.count != 256 { return nil }

    // G2 point: x_c0, x_c1, y_c0, y_c1 (each 64 bytes -> 48 bytes field element)
    guard let xc0 = bytes64ToFp381Limbs(Array(input[0..<64])),
          let xc1 = bytes64ToFp381Limbs(Array(input[64..<128])),
          let yc0 = bytes64ToFp381Limbs(Array(input[128..<192])),
          let yc1 = bytes64ToFp381Limbs(Array(input[192..<256])) else { return nil }

    // Scalar: 32 bytes big-endian
    let sInt = bytes32ToLimbs4(Array(input[224..<256]))

    // Validate coordinates are in field
    if bls12FpGEP(xc0) || bls12FpGEP(xc1) { return nil }
    if bls12FpGEP(yc0) || bls12FpGEP(yc1) { return nil }

    // Check if point is identity
    let isInf = isZero6(xc0) && isZero6(xc1) && isZero6(yc0) && isZero6(yc1)

    if isInf || isZero4(sInt) {
        return [UInt8](repeating: 0, count: 256)
    }

    // Convert to Montgomery and verify on curve
    let xc0Mont = bls12FpToMont(xc0)
    let xc1Mont = bls12FpToMont(xc1)
    let yc0Mont = bls12FpToMont(yc0)
    let yc1Mont = bls12FpToMont(yc1)

    if !bls12G2IsOnCurve((xc0Mont, xc1Mont), (yc0Mont, yc1Mont)) { return nil }

    // Build projective G2 point: (x, y, 1) in Fp2 coordinates
    var projPoint = xc0Mont + xc1Mont + yc0Mont + yc1Mont + BLS12_381_FP_ONE
    var scalar = sInt

    var result = [UInt64](repeating: 0, count: 36)
    projPoint.withUnsafeBufferPointer { pP in
        scalar.withUnsafeBufferPointer { sP in
            result.withUnsafeMutableBufferPointer { rP in
                bls12_381_g2_scalar_mul(pP.baseAddress!, sP.baseAddress!, rP.baseAddress!)
            }
        }
    }

    // Projective -> affine
    let zMont = Array(result[24..<36])
    if isZero6(zMont) {
        return [UInt8](repeating: 0, count: 256)
    }

    // z_inv = z^(p-2) mod p
    var zInv = [UInt64](repeating: 0, count: 6)
    zMont.withUnsafeBufferPointer { zP in
        zInv.withUnsafeMutableBufferPointer { rP in
            bls12_381_fp_inv_ext(zP.baseAddress!, rP.baseAddress!)
        }
    }

    // Fp2 inversion
    var zInvFp2 = [UInt64](repeating: 0, count: 12)
    zInvFp2.withUnsafeMutableBufferPointer { zFp2P in
        zInv.withUnsafeBufferPointer { zP in
            bls12_381_fp2_inv(zP.baseAddress!, zFp2P.baseAddress!)
        }
    }

    // x_aff = x * z_inv^2, y_aff = y * z_inv^3
    let zInv2 = swiftBLS12FpMul(zInv, zInv)
    let zInv3 = swiftBLS12FpMul(zInv2, zInv)
    let xc0Aff = swiftBLS12FpMul(Array(result[0..<6]), zInv2)
    let xc1Aff = swiftBLS12FpMul(Array(result[6..<12]), zInv2)
    let yc0Aff = swiftBLS12FpMul(Array(result[12..<18]), zInv3)
    let yc1Aff = swiftBLS12FpMul(Array(result[18..<24]), zInv3)

    let xc0Out = bls12FpFromMont(xc0Aff)
    let xc1Out = bls12FpFromMont(xc1Aff)
    let yc0Out = bls12FpFromMont(yc0Aff)
    let yc1Out = bls12FpFromMont(yc1Aff)

    return fp381LimbsToBytes64(xc0Out) + fp381LimbsToBytes64(xc1Out) +
           fp381LimbsToBytes64(yc0Out) + fp381LimbsToBytes64(yc1Out)
}

// MARK: - EVM Precompile 0x0D: BLS12-381 G1+G2 Pairing Check (EIP-2537)

/// BLS12-381 G1 x G2 pairing check (variant for G1 in Fp, G2 in Fp2).
/// Input: 384 bytes (G1 (128B) || G2 (256B)).
/// Output: 32 bytes (0 or 1). Returns nil on invalid input.
public func EVMPrecompile0D_pairing12(input: [UInt8]) -> [UInt8]? {
    // Identical input format to 0x10, but different gas cost (handled by caller)
    // Reuse the same pairing check logic
    return EVMPrecompile10_bls12381Pairing(input: input)
}

// MARK: - EVM Precompile 0x0E: BLS12-381 G1 Multi-Exp (EIP-2537)

/// BLS12-381 G1 multi-exponential. Input: variable (N * 160 bytes).
/// Each item: G1 point (128B) || scalar (32B).
/// Output: 128 bytes (G1 point). Returns nil on invalid input.
public func EVMPrecompile0E_bls12381G1MultiExp(input: [UInt8]) -> [UInt8]? {
    if input.isEmpty || input.count % 160 != 0 { return nil }
    let n = input.count / 160

    // Parse points and scalars
    var flatScalars = [UInt32]()
    var points = [UInt64]()

    for i in 0..<n {
        let off = i * 160

        // G1 point
        guard let xInt = bytes64ToFp381Limbs(Array(input[off..<off+64])),
              let yInt = bytes64ToFp381Limbs(Array(input[off+64..<off+128])) else { return nil }

        if bls12FpGEP(xInt) || bls12FpGEP(yInt) { return nil }
        let isInf = isZero6(xInt) && isZero6(yInt)

        if !isInf {
            let xMont = bls12FpToMont(xInt)
            let yMont = bls12FpToMont(yInt)
            if !bls12G1IsOnCurve(xMont, yMont) { return nil }
            points.append(contentsOf: xMont)
            points.append(contentsOf: yMont)
            points.append(contentsOf: BLS12_381_FP_ONE)
        } else {
            // Identity point - add zero point
            points.append(contentsOf: [UInt64](repeating: 0, count: 18))
        }

        // Scalar: 32 bytes big-endian -> 8 UInt32
        let scalarBytes = Array(input[off+128..<off+160])
        for j in 0..<8 {
            let w = UInt32(scalarBytes[28-j*4]) |
                    (UInt32(scalarBytes[29-j*4]) << 8) |
                    (UInt32(scalarBytes[30-j*4]) << 16) |
                    (UInt32(scalarBytes[31-j*4]) << 24)
            flatScalars.append(w)
        }
    }

    var result = [UInt64](repeating: 0, count: 18)
    flatScalars.withUnsafeBufferPointer { sP in
        points.withUnsafeBufferPointer { pP in
            result.withUnsafeMutableBufferPointer { rP in
                bls12_381_g1_pippenger_msm(pP.baseAddress!, sP.baseAddress!, Int32(n), rP.baseAddress!)
            }
        }
    }

    // Convert to affine and return
    let zMont = Array(result[12..<18])
    if isZero6(zMont) {
        return [UInt8](repeating: 0, count: 128)
    }

    var zInv = [UInt64](repeating: 0, count: 6)
    zMont.withUnsafeBufferPointer { zP in
        zInv.withUnsafeMutableBufferPointer { rP in
            bls12_381_fp_inv_ext(zP.baseAddress!, rP.baseAddress!)
        }
    }

    let zInv2 = swiftBLS12FpMul(zInv, zInv)
    let zInv3 = swiftBLS12FpMul(zInv2, zInv)
    let xAff = swiftBLS12FpMul(Array(result[0..<6]), zInv2)
    let yAff = swiftBLS12FpMul(Array(result[6..<12]), zInv3)

    let xOut = bls12FpFromMont(xAff)
    let yOut = bls12FpFromMont(yAff)

    return fp381LimbsToBytes64(xOut) + fp381LimbsToBytes64(yOut)
}

// MARK: - BLS12-381 Fp Square Root

/// Compute square root of BLS12-381 Fp element using NeonFieldOps C implementation.
/// Returns (sqrt, exists) where exists=false if no square root exists.
private func bls12FpSqrt(_ a: [UInt64]) -> ([UInt64], Bool) {
    var aCopy = a
    var r = [UInt64](repeating: 0, count: 6)
    let ok = bls12_381_fp_sqrt(&aCopy, &r)
    return (r, ok != 0)
}

/// Compute modular inverse of BLS12-381 Fp element using Fermat's little theorem
private func bls12FpInverse(_ a: [UInt64]) -> [UInt64] {
    // p-2 constant in Montgomery form (R^(p-2) mod p)
    let pMinus2Mont: [UInt64] = [
        0x9e9f6d5d9b82e31f, 0x3ab6c2d14d8b4a9,
        0xeb6d7f0eb3d8c8d, 0x64774b84f38512be,
        0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a
    ]
    var result = BLS12_381_FP_ONE
    var base = a
    for i in 0..<6 {
        var bit = pMinus2Mont[i]
        while bit != 0 {
            if (bit & 1) != 0 {
                result = swiftBLS12FpMul(result, base)
            }
            base = swiftBLS12FpMul(base, base)
            bit >>= 1
        }
    }
    return result
}

// MARK: - EVM Precompile 0x11: BLS12-381 Map Fp to G1 (EIP-2537)

/// BLS12-381 map field element to G1 point using Simplified SWU.
/// Input: 64 bytes (field element).
/// Output: 128 bytes (G1 point). Returns nil on invalid input.
///
/// EIP-2537 Simplified SWU map for G1 (y^2 = x^3 + 4):
///   x = (-b / (1 + u^2 * (u^4 - 1)^2)) - a    where a=0, b=4
///   y = sqrt(x^3 + 4)
///
/// The sign of y is adjusted so that sgn0(y) == sgn0(x) (parity matching per EIP-2537).
public func EVMPrecompile11_bls12381MapG1(input: [UInt8]) -> [UInt8]? {
    guard input.count == 64 else { return nil }

    // Parse and validate input as Fp element
    guard let u = bytes64ToFp381Limbs(input) else { return nil }

    // Convert to Montgomery form for arithmetic
    let uMont = bls12FpToMont(u)

    // Simplified SWU for G1 (a=0, b=4):
    // x = -4 / (1 + u^2 * (u^4 - 1)^2)
    //   = -4 / (1 + u^6 - 2*u^4 + u^2)
    //   = -4 / (u^6 - 2*u^4 + u^2 + 1)
    //
    // Simplified: denom = 1 + u^2 * (u^4 - 1)^2

    // Compute u^2
    let u2 = swiftBLS12FpMul(uMont, uMont)

    // Compute u^4
    let u4 = swiftBLS12FpMul(u2, u2)

    // Compute (u^4 - 1)
    let u4Minus1 = bls12FpSub(u4, BLS12_381_FP_ONE)

    // Compute (u^4 - 1)^2
    let u4Minus1Sq = swiftBLS12FpMul(u4Minus1, u4Minus1)

    // Compute u^2 * (u^4 - 1)^2
    let u2Times = swiftBLS12FpMul(u2, u4Minus1Sq)

    // Compute denominator = 1 + u^2 * (u^4 - 1)^2
    let denom = bls12FpAdd(BLS12_381_FP_ONE, u2Times)

    // Compute x = -4 / denom = (-4) * denom^(-1)
    // -4 in Montgomery form: R * (-4) mod p = p - 4
    var neg4Int: [UInt64] = BLS12_381_P
    neg4Int[0] = neg4Int[0] &- 4  // p - 4 (limb 0 with borrow)
    let neg4Mont = bls12FpToMont(neg4Int)

    let denomInv = bls12FpInverse(denom)
    let xMont = swiftBLS12FpMul(neg4Mont, denomInv)

    // Compute y^2 = x^3 + 4
    let x2 = swiftBLS12FpMul(xMont, xMont)
    let x3 = swiftBLS12FpMul(x2, xMont)
    let y2 = bls12FpAdd(x3, BLS12_381_B_MONT)

    // Compute y = sqrt(y^2)
    let (yMont, yExists) = bls12FpSqrt(y2)
    guard yExists else { return nil }

    // Sign fixup per EIP-2537: sgn0(y) must equal sgn0(x)
    // Both x and y are in Montgomery; to check integer parity we convert out
    let xInt = bls12FpFromMont(xMont)
    let yInt = bls12FpFromMont(yMont)

    let xParity = (xInt[0] & 1)
    let yParity = (yInt[0] & 1)

    var finalY = yMont
    if xParity != yParity {
        // Negate y: y = p - y (in Montgomery terms: -yMont mod p)
        var negYMont: [UInt64] = [0, 0, 0, 0, 0, 0]
        var borrow: UInt64 = 0
        for i in 0..<6 {
            let diff = BLS12_381_P[i] &- yMont[i] &- borrow
            negYMont[i] = diff
            borrow = (yMont[i] > BLS12_381_P[i] &- borrow) ? 1 : 0
        }
        finalY = negYMont
    }

    // Convert back to integer for output
    let xOut = bls12FpFromMont(xMont)
    let yOut = bls12FpFromMont(finalY)

    return fp381LimbsToBytes64(xOut) + fp381LimbsToBytes64(yOut)
}

// MARK: - EVM Precompile 0x12: BLS12-381 Map Fp2 to G2 (EIP-2537)

/// BLS12-381 map Fp2 element to G2 point using Simplified SWU.
/// Input: 128 bytes (Fp2 element: c0 (64B) || c1 (64B)).
/// Output: 256 bytes (G2 point).
///
/// EIP-2537 specifies an isogeny-based map from Fp2 to G2 curve.
/// This uses the same simplified SWU methodology as G1 but with
/// Fp2 arithmetic and a 3-isogeny to move from the helper curve E' to G2.
public func EVMPrecompile12_bls12381MapG2(input: [UInt8]) -> [UInt8]? {
    guard input.count == 128 else { return nil }

    // Parse Fp2 element (c0 || c1)
    guard let c0 = bytes64ToFp381Limbs(Array(input[0..<64])),
          let c1 = bytes64ToFp381Limbs(Array(input[64..<128])) else { return nil }

    // Convert to Montgomery form
    let c0Mont = bls12FpToMont(c0)
    let c1Mont = bls12FpToMont(c1)
    let u = (c0Mont, c1Mont)

    // Helper curve E': y^2 = x^3 + A'*x + B'
    // A' = 240 * u (Fp2: real=0, imag=240)
    // B' = 1012 * (1+u) (Fp2: real=1012, imag=1012)
    // Z = -(2+u) = (-2, -1) in Fp2

    let aPrime = (
        [UInt64](repeating: 0, count: 6),  // c0 = 0
        bls12FpToMont([240, 0, 0, 0, 0, 0])  // c1 = 240
    )
    let bPrime = (
        bls12FpToMont([1012, 0, 0, 0, 0, 0]),  // c0 = 1012
        bls12FpToMont([1012, 0, 0, 0, 0, 0])    // c1 = 1012
    )
    let zVal = (
        bls12FpToMont([BLS12_381_P[0] &- 2, 0, 0, 0, 0, BLS12_381_P[5]]),  // c0 = -2
        bls12FpToMont([BLS12_381_P[0] &- 1, 0, 0, 0, 0, BLS12_381_P[5]])    // c1 = -1
    )

    let oneFp2: ([UInt64], [UInt64]) = (BLS12_381_FP_ONE, BLS12_381_FP_ONE)

    // SWU: compute x' on helper curve E'
    // tv1 = u^2
    let u2 = bls12Fp2Mul(u, u)

    // tv1 = Z * u^2
    let z_u2 = bls12Fp2Mul(zVal, u2)

    // tv2 = tv1^2
    let tv1_sq = bls12Fp2Sqr(z_u2)

    // tv2 = tv2 + tv1 = z_u2^2 + z_u2
    let tv2_plus_tv1 = bls12Fp2Add(tv1_sq, z_u2)

    // tv3 = tv2 + 1
    let tv3 = bls12Fp2Add(tv2_plus_tv1, oneFp2)

    // tv3 = B' * tv3
    let tv3_scaled = bls12Fp2Mul(bPrime, tv3)

    // tv4 = (tv2 == 0) ? Z : -tv2
    let isZero = bls12Fp2IsZero(tv1_sq)
    let negTv1_sq = bls12Fp2Neg(tv1_sq)
    let tv4 = isZero ? zVal : negTv1_sq

    // tv4 = A' * tv4
    let tv4_scaled = bls12Fp2Mul(aPrime, tv4)

    // x_num = tv4_scaled * tv4_scaled (for rational function denominator check)
    let x_num_den = bls12Fp2Mul(tv4_scaled, tv4_scaled)

    // TV2 = -A'/B' * (1 + Z*u^2 / (1 + u^2*(u^4-1)^2))
    // Simplified: compute (1 + z*u^2) / (tv3 + tv4*x_num_den)
    // Actually EIP-2537 uses: x' = -A'/B' * (1 + Z*u^2 / (1 + u^2*(u^4-1)^2))
    // Let's follow RFC 9380 Appendix F more directly

    // Compute u^4
    let u4 = bls12Fp2Sqr(bls12Fp2Sqr(u))

    // tv1 = u^4 - 1
    let u4_minus_1 = bls12Fp2Sub(u4, oneFp2)

    // tv1 = (u^4 - 1)^2
    let tv1 = bls12Fp2Sqr(u4_minus_1)

    // tv1 = u^2 * (u^4 - 1)^2
    let tv1_denom = bls12Fp2Mul(u2, tv1)

    // tv1 = 1 + u^2 * (u^4 - 1)^2
    let tv1_plus_1 = bls12Fp2Add(oneFp2, tv1_denom)

    // tv1 = Z * u^2
    let z_u2_v2 = bls12Fp2Mul(zVal, u2)

    // TV2 (numerator for the fraction) = Z * u^2 / (1 + u^2*(u^4-1)^2)
    // We need to divide z_u2_v2 by tv1_plus_1
    let z_u2_div = bls12Fp2Inv(tv1_plus_1)
    let frac = bls12Fp2Mul(z_u2_v2, z_u2_div)

    // x' = -A'/B' * (1 + frac)
    // A' = (0, 240), B' = (1012, 1012)
    // -A'/B' = -A' * B'^-1, but since B' = 1012*(1+u), the -A'/B' computation
    // For the helper curve, x' = (-A' * (1 + frac)) / B'

    // 1 + frac
    let one_frac = bls12Fp2Add(oneFp2, frac)

    // -A' * (1 + frac) = A' * (-1) * (1 + frac) where -1 = (p-1, p-1) in Montgomery
    let negOne: ([UInt64], [UInt64]) = (
        bls12FpSub(BLS12_381_FP_ONE, [UInt64(1), 0, 0, 0, 0, 0]),  // p-1 in Mont
        bls12FpSub(BLS12_381_FP_ONE, [UInt64(1), 0, 0, 0, 0, 0])
    )
    let negA = bls12Fp2Mul(aPrime, negOne)  // -A' in Fp2
    let negA_times_frac = bls12Fp2Mul(negA, one_frac)

    // x' = (-A' * (1 + frac)) / B' = negA_times_frac * B'^-1
    let bPrimeInv = bls12Fp2Inv(bPrime)
    let xPrime = bls12Fp2Mul(negA_times_frac, bPrimeInv)

    // Compute y'^2 = x'^3 + A'*x' + B'
    let xPrime_sq = bls12Fp2Sqr(xPrime)
    let xPrime_cu = bls12Fp2Mul(xPrime_sq, xPrime)
    let aPrime_x = bls12Fp2Mul(aPrime, xPrime)
    let yPrime_sq = bls12Fp2Add(bls12Fp2Add(xPrime_cu, aPrime_x), bPrime)

    // y' = sqrt(y'^2) with sign adjustment
    let (yPrime, yExists) = bls12Fp2Sqrt(yPrime_sq)
    guard yExists else { return nil }

    // Sign adjustment per EIP-2537: sgn0(y) must equal sgn0(u)
    // This ensures deterministic mapping
    let signY = sgn0Fp2(yPrime)
    let signU = sgn0Fp2(u)

    var yAdjusted = yPrime
    if signY != signU {
        // y = -y (negate in Fp2: (p - c0, p - c1) in Montgomery form)
        yAdjusted = bls12Fp2Neg(yPrime)
    }

    // Apply 3-isogeny map to get (x, y) on G2
    let (xG2, yG2) = applyIso3Map(xPrime, yAdjusted)

    // Convert from Montgomery form to integer
    let xG2Int0 = bls12FpFromMont(xG2.0)
    let xG2Int1 = bls12FpFromMont(xG2.1)
    let yG2Int0 = bls12FpFromMont(yG2.0)
    let yG2Int1 = bls12FpFromMont(yG2.1)

    // Output: xG2 (c0 || c1) || yG2 (c0 || c1), each as 64-byte big-endian
    return fp381LimbsToBytes64(xG2Int0) + fp381LimbsToBytes64(xG2Int1) +
           fp381LimbsToBytes64(yG2Int0) + fp381LimbsToBytes64(yG2Int1)
}

// MARK: - Fp2 Helpers for Map G2

/// Fp2 addition
private func bls12Fp2Add(_ a: ([UInt64], [UInt64]), _ b: ([UInt64], [UInt64])) -> ([UInt64], [UInt64]) {
    return (bls12FpAdd(a.0, b.0), bls12FpAdd(a.1, b.1))
}

/// Fp2 subtraction
private func bls12Fp2Sub(_ a: ([UInt64], [UInt64]), _ b: ([UInt64], [UInt64])) -> ([UInt64], [UInt64]) {
    return (bls12FpSub(a.0, b.0), bls12FpSub(a.1, b.1))
}

/// Fp2 negation
private func bls12Fp2Neg(_ a: ([UInt64], [UInt64])) -> ([UInt64], [UInt64]) {
    return (bls12FpSub(BLS12_381_FP_ONE, a.0), bls12FpSub(BLS12_381_FP_ONE, a.1))
}

/// Fp2 zero check
private func bls12Fp2IsZero(_ a: ([UInt64], [UInt64])) -> Bool {
    return isZero6(a.0) && isZero6(a.1)
}

/// Fp2 copy
private func bls12Fp2Copy(_ a: ([UInt64], [UInt64])) -> ([UInt64], [UInt64]) {
    return a
}

/// Fp2 inverse via Fermat's little theorem: a^(p^2 - 2) mod p
/// Note: Fp2 inverse is expensive; we use the formula: a^-1 = (a0 - a1*u) / (a0^2 + a1^2)
/// which requires only one Fp inverse instead of full exponentiation
private func bls12Fp2Inv(_ a: ([UInt64], [UInt64])) -> ([UInt64], [UInt64]) {
    // norm = a0^2 + a1^2 in Fp
    let a0Sq = swiftBLS12FpMul(a.0, a.0)
    let a1Sq = swiftBLS12FpMul(a.1, a.1)
    let norm = bls12FpAdd(a0Sq, a1Sq)

    // norm^-1
    let normInv = bls12FpInverse(norm)

    // result c0 = a0 * norm^-1, c1 = -a1 * norm^-1
    let c0 = swiftBLS12FpMul(a.0, normInv)
    let negA1 = bls12FpSub(BLS12_381_FP_ONE, a.1)  // -a1 in Montgomery = p - a1
    let c1 = swiftBLS12FpMul(negA1, normInv)

    return (c0, c1)
}

/// Fp2 square root using Tonelli-Shanks algorithm for Fp2
/// Returns (sqrt, exists) where exists=false if no square root exists
private func bls12Fp2Sqrt(_ a: ([UInt64], [UInt64])) -> (([UInt64], [UInt64]), Bool) {
    if bls12Fp2IsZero(a) {
        return (([UInt64](repeating: 0, count: 6), [UInt64](repeating: 0, count: 6)), true)
    }

    // norm = c0^2 + c1^2
    let c0Sq = swiftBLS12FpMul(a.0, a.0)
    let c1Sq = swiftBLS12FpMul(a.1, a.1)
    let norm = bls12FpAdd(c0Sq, c1Sq)

    // sqrt(norm) in Fp
    let (normSqrt, normExists) = bls12FpSqrt(norm)
    guard normExists else { return ((a.0, a.1), false) }

    // two_inv = 1/2 in Fp Montgomery
    let two: [UInt64] = [2, 0, 0, 0, 0, 0]
    let twoMont = bls12FpToMont(two)
    // Compute 1/2 = (p+1)/2 via bls12FpInverse of 2
    let oneMont: [UInt64] = [1, 0, 0, 0, 0, 0]
    let twoInv = swiftBLS12FpMul(oneMont, bls12FpInverse(twoMont))

    // Try branch 1: t^2 = (c0 + norm_sqrt) / 2
    let c0PlusNorm = bls12FpAdd(a.0, normSqrt)
    let cand = swiftBLS12FpMul(c0PlusNorm, twoInv)
    let (t1, t1Exists) = bls12FpSqrt(cand)

    if t1Exists && !isZero6(t1) {
        // Compute c1 / (2*t1)
        let twoT = swiftBLS12FpMul(twoMont, t1)
        let twoTInv = bls12FpInverse(twoT)
        let c1Part = swiftBLS12FpMul(a.1, twoTInv)

        // Result is (t1, c1_part)
        let result = (t1, c1Part)
        // Verify
        let verify = bls12Fp2Sqr(result)
        if verify.0 == a.0 && verify.1 == a.1 {
            return (result, true)
        }
        // Try negated
        let negT1 = bls12FpSub(BLS12_381_FP_ONE, t1)
        let negC1Part = bls12FpSub(BLS12_381_FP_ONE, c1Part)
        let result2 = (negT1, negC1Part)
        let verify2 = bls12Fp2Sqr(result2)
        if verify2.0 == a.0 && verify2.1 == a.1 {
            return (result2, true)
        }
    }

    // Try branch 2: t^2 = (c0 - norm_sqrt) / 2
    let c0MinusNorm = bls12FpSub(a.0, normSqrt)
    let cand2 = swiftBLS12FpMul(c0MinusNorm, twoInv)
    let (t2, t2Exists) = bls12FpSqrt(cand2)

    if t2Exists && !isZero6(t2) {
        let twoT2 = swiftBLS12FpMul(twoMont, t2)
        let twoT2Inv = bls12FpInverse(twoT2)
        let c1Part2 = swiftBLS12FpMul(a.1, twoT2Inv)

        let result = (t2, c1Part2)
        let verify = bls12Fp2Sqr(result)
        if verify.0 == a.0 && verify.1 == a.1 {
            return (result, true)
        }
        let negT2 = bls12FpSub(BLS12_381_FP_ONE, t2)
        let negC1Part2 = bls12FpSub(BLS12_381_FP_ONE, c1Part2)
        let result2 = (negT2, negC1Part2)
        let verify2 = bls12Fp2Sqr(result2)
        if verify2.0 == a.0 && verify2.1 == a.1 {
            return (result2, true)
        }
    }

    return ((a.0, a.1), false)
}

/// Sign function for Fp2 per RFC 9380: sgn0(x) = 0 if x.c0 != 0 and x.c0 mod 2 == 0,
/// or if x.c0 == 0 and x.c1 mod 2 == 0; otherwise 1
private func sgn0Fp2(_ x: ([UInt64], [UInt64])) -> Int {
    // Get integer form of c0 (multiply by 1 in Montgomery to convert)
    let c0Int = bls12FpFromMont(x.0)
    let c1Int = bls12FpFromMont(x.1)

    let c0Parity = Int(c0Int[0] & 1)
    let c0Zero = isZero6(x.0) ? 1 : 0
    let c1Parity = Int(c1Int[0] & 1)

    return c0Parity | (c0Zero & c1Parity)
}

// MARK: - 3-Isogeny Map Coefficients for G2
// From RFC 9380 Appendix E.3 for BLS12-381 G2

/// x_num coefficients: [k3, k2, k1, k0] for x_num(x) = k3*x^3 + k2*x^2 + k1*x + k0
private struct Iso3XNum {
    static let k3_c0: [UInt64] = [
        0xbb3e04691b4814c0, 0x8b6a4d177f4ec647,
        0x2c52d39fd3a042a8, 0xb5b7a9a47d7ed853,
        0x5c759507e8e333eb, 0x0000000000000000
    ]
    static let k3_c1: [UInt64] = k3_c0  // Same values

    static let k2_c0: [UInt64] = [0, 0, 0, 0, 0, 0]
    static let k2_c1: [UInt64] = [
        0x26a9ffffffffc71a, 0x1472aaa9cb8d5555,
        0x9a208c6b4f20a418, 0x984f87adf7ae0c7f,
        0x32126fced787c88f, 0x11560bf17baa99bc
    ]

    static let k1_c0: [UInt64] = [
        0x26a9ffffffffc71e, 0x1472aaa9cb8d5555,
        0x9a208c6b4f20a418, 0x984f87adf7ae0c7f,
        0x32126fced787c88f, 0x11560bf17baa99bc
    ]
    static let k1_c1: [UInt64] = [
        0x6e0affffffffc71c, 0xe0f6bdb5db82d0a7,
        0x090c6e55a0226f65, 0x674d812049b7d4a1,
        0x90937e76bc3e447d, 0x8ab05f8bdd54cde1
    ]

    static let k0_c0: [UInt64] = [
        0xab2aaaaaaab71cb, 0xb4dfdb27fdaff85c,
        0x1b12c47c21a1be30, 0xd09bfd206a1aca1e,
        0x71d6541fa38ccfae, 0x0000000000000001
    ]
    static let k0_c1: [UInt64] = [0, 0, 0, 0, 0, 0]
}

/// x_den coefficients: [1, k1, k0] for x_den(x) = x^2 + k1*x + k0
private struct Iso3XDen {
    static let k1_c0: [UInt64] = [0x12, 0, 0, 0, 0, 0]
    static let k1_c1: [UInt64] = [
        0xb9feffffffffaa63, 0x1eabfffeb153ffff,
        0x6730d2a0f6b0f624, 0x64774b84f38512bf,
        0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a
    ]

    static let k0_c0: [UInt64] = [0xc, 0, 0, 0, 0, 0]
    static let k0_c1: [UInt64] = [
        0xb9feffffffffaa9f, 0x1eabfffeb153ffff,
        0x6730d2a0f6b0f624, 0x64774b84f38512bf,
        0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a
    ]
}

/// y_num coefficients: [k3, k2, k1, k0] for y_num(x) = k3*x^3 + k2*x^2 + k1*x + k0
private struct Iso3YNum {
    static let k3_c0: [UInt64] = [
        0x4fd20b93d5a2f921, 0xb11ab8e6358a17f4,
        0xf5446d21d8fd8e24, 0x0f7da5d4a07f649b,
        0x59a4c18b076d1193, 0x1530477c7ab4113b
    ]
    static let k3_c1: [UInt64] = k3_c0  // Same values

    static let k2_c0: [UInt64] = [0, 0, 0, 0, 0, 0]
    static let k2_c1: [UInt64] = [
        0xbb3e04691b4814c0, 0x8b6a4d177f4ec647,
        0x2c52d39fd3a042a8, 0xb5b7a9a47d7ed853,
        0x5c759507e8e333eb, 0x0000000000000000
    ]

    static let k1_c0: [UInt64] = [
        0x26a9ffffffffc71c, 0x1472aaa9cb8d5555,
        0x9a208c6b4f20a418, 0x984f87adf7ae0c7f,
        0x32126fced787c88f, 0x11560bf17baa99bc
    ]
    static let k1_c1: [UInt64] = [
        0x6e0affffffffc71c, 0xe0f6bdb5db82d0a7,
        0x090c6e55a0226f65, 0x674d812049b7d4a1,
        0x90937e76bc3e447d, 0x8ab05f8bdd54cde1
    ]

    static let k0_c0: [UInt64] = [
        0x06d7f7e29ef67ccc, 0xd096233c18df0cfe,
        0xb0e977999978dc5f, 0x761b0f37a1e26286,
        0xfbf7043de3811ad0, 0x124c9ad43b6cf79b
    ]
    static let k0_c1: [UInt64] = k0_c0  // Same values
}

/// y_den coefficients: [1, k2, k1, k0] for y_den(x) = x^3 + k2*x^2 + k1*x + k0
private struct Iso3YDen {
    static let k2_c0: [UInt64] = [
        0xb9feffffffffa8fb, 0x1eabfffeb153ffff,
        0x6730d2a0f6b0f624, 0x64774b84f38512bf,
        0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a
    ]
    static let k2_c1: [UInt64] = k2_c0  // Same values

    static let k1_c0: [UInt64] = [0, 0, 0, 0, 0, 0]
    static let k1_c1: [UInt64] = [
        0xb9feffffffffa9d3, 0x1eabfffeb153ffff,
        0x6730d2a0f6b0f624, 0x64774b84f38512bf,
        0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a
    ]

    static let k0_c0: [UInt64] = [0x12, 0, 0, 0, 0, 0]
    static let k0_c1: [UInt64] = [
        0xb9feffffffffaa99, 0x1eabfffeb153ffff,
        0x6730d2a0f6b0f624, 0x64774b84f38512bf,
        0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a
    ]
}

/// Helper to create Fp2 constant from limb arrays (non-Montgomery -> Montgomery)
private func makeFp2Const(_ c0: [UInt64], _ c1: [UInt64]) -> ([UInt64], [UInt64]) {
    return (bls12FpToMont(c0), bls12FpToMont(c1))
}

/// Evaluate x_num(x) using Horner's method: ((k3*x + k2)*x + k1)*x + k0
private func evalXNum(_ x: ([UInt64], [UInt64])) -> ([UInt64], [UInt64]) {
    let k3 = makeFp2Const(Iso3XNum.k3_c0, Iso3XNum.k3_c1)
    let k2 = makeFp2Const(Iso3XNum.k2_c0, Iso3XNum.k2_c1)
    let k1 = makeFp2Const(Iso3XNum.k1_c0, Iso3XNum.k1_c1)
    let k0 = makeFp2Const(Iso3XNum.k0_c0, Iso3XNum.k0_c1)

    var result = k3
    result = bls12Fp2Mul(result, x)
    result = bls12Fp2Add(result, k2)
    result = bls12Fp2Mul(result, x)
    result = bls12Fp2Add(result, k1)
    result = bls12Fp2Mul(result, x)
    result = bls12Fp2Add(result, k0)
    return result
}

/// Evaluate x_den(x) using Horner's method: (x + k1)*x + k0 (leading coeff = 1)
private func evalXDen(_ x: ([UInt64], [UInt64])) -> ([UInt64], [UInt64]) {
    let k1 = makeFp2Const(Iso3XDen.k1_c0, Iso3XDen.k1_c1)
    let k0 = makeFp2Const(Iso3XDen.k0_c0, Iso3XDen.k0_c1)

    var result = x  // Start with x
    result = bls12Fp2Add(result, k1)
    result = bls12Fp2Mul(result, x)
    result = bls12Fp2Add(result, k0)
    return result
}

/// Evaluate y_num(x) using Horner's method: ((k3*x + k2)*x + k1)*x + k0
private func evalYNum(_ x: ([UInt64], [UInt64])) -> ([UInt64], [UInt64]) {
    let k3 = makeFp2Const(Iso3YNum.k3_c0, Iso3YNum.k3_c1)
    let k2 = makeFp2Const(Iso3YNum.k2_c0, Iso3YNum.k2_c1)
    let k1 = makeFp2Const(Iso3YNum.k1_c0, Iso3YNum.k1_c1)
    let k0 = makeFp2Const(Iso3YNum.k0_c0, Iso3YNum.k0_c1)

    var result = k3
    result = bls12Fp2Mul(result, x)
    result = bls12Fp2Add(result, k2)
    result = bls12Fp2Mul(result, x)
    result = bls12Fp2Add(result, k1)
    result = bls12Fp2Mul(result, x)
    result = bls12Fp2Add(result, k0)
    return result
}

/// Evaluate y_den(x) using Horner's method: ((x + k2)*x + k1)*x + k0 (leading coeff = 1)
private func evalYDen(_ x: ([UInt64], [UInt64])) -> ([UInt64], [UInt64]) {
    let k2 = makeFp2Const(Iso3YDen.k2_c0, Iso3YDen.k2_c1)
    let k1 = makeFp2Const(Iso3YDen.k1_c0, Iso3YDen.k1_c1)
    let k0 = makeFp2Const(Iso3YDen.k0_c0, Iso3YDen.k0_c1)

    var result = x  // Start with x
    result = bls12Fp2Add(result, k2)
    result = bls12Fp2Mul(result, x)
    result = bls12Fp2Add(result, k1)
    result = bls12Fp2Mul(result, x)
    result = bls12Fp2Add(result, k0)
    return result
}

/// Apply 3-isogeny map to an affine point (x, y) on E' to get a point on G2
private func applyIso3Map(_ x: ([UInt64], [UInt64]), _ y: ([UInt64], [UInt64])) -> (([UInt64], [UInt64]), ([UInt64], [UInt64])) {
    // x_aff = x, y_aff = y (already affine at this point)
    // x_result = x_num(x) / x_den(x)
    // y_result = y * y_num(x) / y_den(x)

    let xNum = evalXNum(x)
    let xDen = evalXDen(x)
    let yNum = evalYNum(x)
    let yDen = evalYDen(x)

    let xDenInv = bls12Fp2Inv(xDen)
    let yDenInv = bls12Fp2Inv(yDen)

    let resultX = bls12Fp2Mul(xNum, xDenInv)
    let yTimesYNumerator = bls12Fp2Mul(y, yNum)
    let resultY = bls12Fp2Mul(yTimesYNumerator, yDenInv)

    return (resultX, resultY)
}

