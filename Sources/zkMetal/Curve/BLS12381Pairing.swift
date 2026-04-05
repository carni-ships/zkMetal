// BLS12-381 optimal ate pairing implementation — C accelerated
// e: G1 x G2 -> GT (subgroup of Fp12*)
//
// The ate pairing for BLS12-381 uses parameter x = -0xd201000000010000
// Miller loop iterates over the bits of |x| = 0xd201000000010000
// Since x is negative, we conjugate the result at the end of the Miller loop.
//
// Tower: Fp2 = Fp[u]/(u^2+1), Fp6 = Fp2[v]/(v^3 - (1+u)), Fp12 = Fp6[w]/(w^2 - v)
// BLS12-381 uses a D-type sextic twist: E': y^2 = x^3 + 4(1+u)

import Foundation
import NeonFieldOps

// MARK: - Fp12/G1/G2 <-> flat uint64_t conversions

/// Convert G1Affine381 to flat [UInt64] (12 limbs: x[6], y[6])
private func g1AffineToFlat(_ p: G1Affine381) -> [UInt64] {
    p.x.to64() + p.y.to64()
}

/// Convert G2Affine381 to flat [UInt64] (24 limbs: x.c0[6], x.c1[6], y.c0[6], y.c1[6])
private func g2AffineToFlat(_ q: G2Affine381) -> [UInt64] {
    q.x.c0.to64() + q.x.c1.to64() + q.y.c0.to64() + q.y.c1.to64()
}

/// Convert flat [UInt64] (72 limbs) to Fp12_381
private func fp12FromFlat(_ f: [UInt64]) -> Fp12_381 {
    // Fp12 = [c0[36], c1[36]]
    // Fp6 = [c0[12], c1[12], c2[12]]
    // Fp2 = [c0[6], c1[6]]
    func fp(_ start: Int) -> Fp381 {
        Fp381.from64(Array(f[start..<start+6]))
    }
    func fp2(_ start: Int) -> Fp2_381 {
        Fp2_381(c0: fp(start), c1: fp(start+6))
    }
    func fp6(_ start: Int) -> Fp6_381 {
        Fp6_381(c0: fp2(start), c1: fp2(start+12), c2: fp2(start+24))
    }
    return Fp12_381(c0: fp6(0), c1: fp6(36))
}

/// Convert Fp12_381 to flat [UInt64] (72 limbs)
private func fp12ToFlat(_ f: Fp12_381) -> [UInt64] {
    func fp2Flat(_ a: Fp2_381) -> [UInt64] { a.c0.to64() + a.c1.to64() }
    func fp6Flat(_ a: Fp6_381) -> [UInt64] { fp2Flat(a.c0) + fp2Flat(a.c1) + fp2Flat(a.c2) }
    return fp6Flat(f.c0) + fp6Flat(f.c1)
}

// MARK: - C-Accelerated Miller Loop

/// Compute the Miller loop for the optimal ate pairing — C accelerated.
public func millerLoop381(_ p: G1Affine381, _ q: G2Affine381) -> Fp12_381 {
    var pFlat = g1AffineToFlat(p)
    var qFlat = g2AffineToFlat(q)
    var result = [UInt64](repeating: 0, count: 72)
    bls12_381_miller_loop(&pFlat, &qFlat, &result)
    return fp12FromFlat(result)
}

// MARK: - C-Accelerated Final Exponentiation

/// Final exponentiation: f^((p^12 - 1) / r) — C accelerated.
public func finalExponentiation381(_ f: Fp12_381) -> Fp12_381 {
    var fFlat = fp12ToFlat(f)
    var result = [UInt64](repeating: 0, count: 72)
    bls12_381_final_exp(&fFlat, &result)
    return fp12FromFlat(result)
}

// MARK: - Public API

/// Compute the optimal ate pairing e(P, Q) for BLS12-381 — C accelerated.
/// P is a G1 point, Q is a G2 point. Returns an element of GT (subgroup of Fp12*).
public func bls12381Pairing(_ p: G1Affine381, _ q: G2Affine381) -> Fp12_381 {
    var pFlat = g1AffineToFlat(p)
    var qFlat = g2AffineToFlat(q)
    var result = [UInt64](repeating: 0, count: 72)
    bls12_381_pairing(&pFlat, &qFlat, &result)
    return fp12FromFlat(result)
}

/// Pairing check: verify that the product of pairings equals 1 — C accelerated.
/// Returns true if prod_i e(P_i, Q_i) = 1 in GT.
public func bls12381PairingCheck(_ pairs: [(G1Affine381, G2Affine381)]) -> Bool {
    let n = pairs.count
    // Build interleaved flat array: [p0[12], q0[24], p1[12], q1[24], ...]
    var flat = [UInt64]()
    flat.reserveCapacity(n * 36)
    for (p, q) in pairs {
        flat.append(contentsOf: g1AffineToFlat(p))
        flat.append(contentsOf: g2AffineToFlat(q))
    }
    return bls12_381_pairing_check(&flat, Int32(n)) != 0
}

/// Check equality of two Fp12 elements
public func fp12_381Equal(_ a: Fp12_381, _ b: Fp12_381) -> Bool {
    let diff = fp12_381Sub(a, b)
    return diff.c0.c0.c0.isZero && diff.c0.c0.c1.isZero &&
           diff.c0.c1.c0.isZero && diff.c0.c1.c1.isZero &&
           diff.c0.c2.c0.isZero && diff.c0.c2.c1.isZero &&
           diff.c1.c0.c0.isZero && diff.c1.c0.c1.isZero &&
           diff.c1.c1.c0.isZero && diff.c1.c1.c1.isZero &&
           diff.c1.c2.c0.isZero && diff.c1.c2.c1.isZero
}

// MARK: - C-Accelerated Fp12 Operations (for use in other modules)

/// Fp12 multiplication — C accelerated
public func fp12_381Mul_C(_ a: Fp12_381, _ b: Fp12_381) -> Fp12_381 {
    var aFlat = fp12ToFlat(a)
    var bFlat = fp12ToFlat(b)
    var result = [UInt64](repeating: 0, count: 72)
    bls12_381_fp12_mul(&aFlat, &bFlat, &result)
    return fp12FromFlat(result)
}

/// Fp12 squaring — C accelerated
public func fp12_381Sqr_C(_ a: Fp12_381) -> Fp12_381 {
    var aFlat = fp12ToFlat(a)
    var result = [UInt64](repeating: 0, count: 72)
    bls12_381_fp12_sqr(&aFlat, &result)
    return fp12FromFlat(result)
}

// MARK: - BLS Signature Verification

/// BLS signature verification: e(pk, H(m)) = e(G1, sig)
/// pk: public key (G1 point)
/// message: hash-to-curve output (G2 point)
/// signature: BLS signature (G2 point)
///
/// Note: This checks e(pk, message) == e(G1_gen, signature)
/// which is equivalent to checking e(pk, message) * e(-G1_gen, signature) == 1
public func bls12381BLSVerify(
    pubkey: G1Affine381,
    message: G2Affine381,
    signature: G2Affine381
) -> Bool {
    let gen = bls12381G1Generator()
    let negGen = g1_381NegateAffine(gen)
    return bls12381PairingCheck([(pubkey, message), (negGen, signature)])
}
