// BLS12-381 Engine: unified API for curve operations and pairing
//
// Provides G1/G2 operations, pairing computation, and BLS signature verification.
// CPU-first implementation -- GPU acceleration of field operations for MSM/NTT
// can use the Fr381 Metal shader separately.

import Foundation

public class BLS12381Engine {

    public init() {}

    // MARK: - G1 Operations

    /// Add two G1 points
    public func g1Add(_ a: G1Projective381, _ b: G1Projective381) -> G1Projective381 {
        g1_381Add(a, b)
    }

    /// Double a G1 point
    public func g1Double(_ p: G1Projective381) -> G1Projective381 {
        g1_381Double(p)
    }

    /// Scalar multiply: [s]P on G1
    public func g1ScalarMul(_ p: G1Projective381, _ s: [UInt64]) -> G1Projective381 {
        g1_381ScalarMul(p, s)
    }

    /// Negate a G1 point
    public func g1Neg(_ p: G1Projective381) -> G1Projective381 {
        g1_381Negate(p)
    }

    /// Multi-scalar multiplication on G1 using Pippenger (C accelerated, multi-threaded)
    public func g1MSM(points: [G1Affine381], scalars: [[UInt32]]) -> G1Projective381 {
        g1_381PippengerMSM(points: points, scalars: scalars)
    }

    /// MSM with pre-flattened scalars
    public func g1MSMFlat(points: [G1Affine381], flatScalars: [UInt32]) -> G1Projective381 {
        g1_381PippengerMSMFlat(points: points, flatScalars: flatScalars)
    }

    /// Convert G1 projective to affine
    public func g1ToAffine(_ p: G1Projective381) -> G1Affine381? {
        g1_381ToAffine(p)
    }

    /// G1 generator
    public func g1Generator() -> G1Affine381 {
        bls12381G1Generator()
    }

    /// Identity element of G1
    public func g1Identity() -> G1Projective381 {
        g1_381Identity()
    }

    /// Check if G1 point is on curve: y^2 = x^3 + 4
    public func g1IsOnCurve(_ p: G1Affine381) -> Bool {
        let y2 = fp381Sqr(p.y)
        let x3 = fp381Mul(fp381Sqr(p.x), p.x)
        let four = fp381FromInt(4)
        let rhs = fp381Add(x3, four)
        return fp381ToInt(y2) == fp381ToInt(rhs)
    }

    // MARK: - G2 Operations

    /// Add two G2 points
    public func g2Add(_ a: G2Projective381, _ b: G2Projective381) -> G2Projective381 {
        g2_381Add(a, b)
    }

    /// Double a G2 point
    public func g2Double(_ p: G2Projective381) -> G2Projective381 {
        g2_381Double(p)
    }

    /// Scalar multiply: [s]P on G2
    public func g2ScalarMul(_ p: G2Projective381, _ s: [UInt64]) -> G2Projective381 {
        g2_381ScalarMul(p, s)
    }

    /// Negate a G2 point
    public func g2Neg(_ p: G2Projective381) -> G2Projective381 {
        g2_381Negate(p)
    }

    /// Convert G2 projective to affine
    public func g2ToAffine(_ p: G2Projective381) -> G2Affine381? {
        g2_381ToAffine(p)
    }

    /// G2 identity
    public func g2Identity() -> G2Projective381 {
        g2_381Identity()
    }

    /// Check if G2 point is on twist curve: y^2 = x^3 + 4(1+u)
    public func g2IsOnCurve(_ p: G2Affine381) -> Bool {
        let y2 = fp2_381Sqr(p.y)
        let x2 = fp2_381Sqr(p.x)
        let x3 = fp2_381Mul(x2, p.x)
        let bPrime = Fp2_381(c0: fp381FromInt(4), c1: fp381FromInt(4))
        let rhs = fp2_381Add(x3, bPrime)
        // Compare c0 and c1 components
        return fp381ToInt(y2.c0) == fp381ToInt(rhs.c0) &&
               fp381ToInt(y2.c1) == fp381ToInt(rhs.c1)
    }

    // MARK: - Pairing

    /// Compute the ate pairing e(P, Q)
    public func pair(_ p: G1Affine381, _ q: G2Affine381) -> Fp12_381 {
        bls12381Pairing(p, q)
    }

    /// Pairing check: verify prod_i e(P_i, Q_i) == 1
    public func pairingCheck(_ pairs: [(G1Affine381, G2Affine381)]) -> Bool {
        bls12381PairingCheck(pairs)
    }

    // MARK: - BLS Signatures

    /// Verify a BLS signature
    /// pubkey: public key in G1
    /// message: hash-to-curve output in G2
    /// signature: BLS signature in G2
    public func blsVerify(pubkey: G1Affine381, message: G2Affine381, signature: G2Affine381) -> Bool {
        bls12381BLSVerify(pubkey: pubkey, message: message, signature: signature)
    }

    // MARK: - Field Tower Operations

    /// Fp arithmetic
    public func fpAdd(_ a: Fp381, _ b: Fp381) -> Fp381 { fp381Add(a, b) }
    public func fpSub(_ a: Fp381, _ b: Fp381) -> Fp381 { fp381Sub(a, b) }
    public func fpMul(_ a: Fp381, _ b: Fp381) -> Fp381 { fp381Mul(a, b) }
    public func fpSqr(_ a: Fp381) -> Fp381 { fp381Sqr(a) }
    public func fpInv(_ a: Fp381) -> Fp381 { fp381Inverse(a) }
    public func fpNeg(_ a: Fp381) -> Fp381 { fp381Neg(a) }

    /// Fp2 arithmetic
    public func fp2Add(_ a: Fp2_381, _ b: Fp2_381) -> Fp2_381 { fp2_381Add(a, b) }
    public func fp2Sub(_ a: Fp2_381, _ b: Fp2_381) -> Fp2_381 { fp2_381Sub(a, b) }
    public func fp2Mul(_ a: Fp2_381, _ b: Fp2_381) -> Fp2_381 { fp2_381Mul(a, b) }
    public func fp2Sqr(_ a: Fp2_381) -> Fp2_381 { fp2_381Sqr(a) }
    public func fp2Inv(_ a: Fp2_381) -> Fp2_381 { fp2_381Inverse(a) }
    public func fp2Neg(_ a: Fp2_381) -> Fp2_381 { fp2_381Neg(a) }
    public func fp2Conj(_ a: Fp2_381) -> Fp2_381 { fp2_381Conjugate(a) }

    /// Fp6 arithmetic
    public func fp6Add(_ a: Fp6_381, _ b: Fp6_381) -> Fp6_381 { fp6_381Add(a, b) }
    public func fp6Sub(_ a: Fp6_381, _ b: Fp6_381) -> Fp6_381 { fp6_381Sub(a, b) }
    public func fp6Mul(_ a: Fp6_381, _ b: Fp6_381) -> Fp6_381 { fp6_381Mul(a, b) }
    public func fp6Sqr(_ a: Fp6_381) -> Fp6_381 { fp6_381Sqr(a) }
    public func fp6Inv(_ a: Fp6_381) -> Fp6_381 { fp6_381Inverse(a) }
    public func fp6Neg(_ a: Fp6_381) -> Fp6_381 { fp6_381Neg(a) }

    /// Fp12 arithmetic
    public func fp12Add(_ a: Fp12_381, _ b: Fp12_381) -> Fp12_381 { fp12_381Add(a, b) }
    public func fp12Sub(_ a: Fp12_381, _ b: Fp12_381) -> Fp12_381 { fp12_381Sub(a, b) }
    public func fp12Mul(_ a: Fp12_381, _ b: Fp12_381) -> Fp12_381 { fp12_381Mul(a, b) }
    public func fp12Sqr(_ a: Fp12_381) -> Fp12_381 { fp12_381Sqr(a) }
    public func fp12Inv(_ a: Fp12_381) -> Fp12_381 { fp12_381Inverse(a) }
    public func fp12Conj(_ a: Fp12_381) -> Fp12_381 { fp12_381Conjugate(a) }

    /// Frobenius endomorphism: x -> x^p, x^(p^2), x^(p^3) on Fp12
    public func fp12Frobenius(_ a: Fp12_381) -> Fp12_381 { fp12_381Frobenius(a) }
    public func fp12Frobenius2(_ a: Fp12_381) -> Fp12_381 { fp12_381Frobenius2(a) }
    public func fp12Frobenius3(_ a: Fp12_381) -> Fp12_381 { fp12_381Frobenius3(a) }

    /// Cyclotomic squaring (for elements in the cyclotomic subgroup after easy part of final exp)
    public func fp12CyclotomicSqr(_ a: Fp12_381) -> Fp12_381 { fp12_381CyclotomicSqr(a) }

    /// Fr (scalar field) arithmetic
    public func frAdd(_ a: Fr381, _ b: Fr381) -> Fr381 { fr381Add(a, b) }
    public func frSub(_ a: Fr381, _ b: Fr381) -> Fr381 { fr381Sub(a, b) }
    public func frMul(_ a: Fr381, _ b: Fr381) -> Fr381 { fr381Mul(a, b) }
    public func frSqr(_ a: Fr381) -> Fr381 { fr381Sqr(a) }
    public func frInv(_ a: Fr381) -> Fr381 { fr381Inverse(a) }
    public func frNeg(_ a: Fr381) -> Fr381 { fr381Neg(a) }
}
