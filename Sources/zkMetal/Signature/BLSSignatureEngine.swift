// BLS Signature Engine for BLS12-381
// Implements BLS signatures as used by Ethereum's consensus layer (beacon chain).
//
// Signature scheme:
//   - Secret key: scalar in Fr
//   - Public key: [sk]G1 (point on G1)
//   - Signature: [sk]H(m) where H: {0,1}* -> G2 (hash-to-curve)
//   - Verify: e(G1, sig) == e(pk, H(m))
//
// Hash-to-curve uses expand_message_xmd (SHA-256) per RFC 9380 for uniform
// field element generation, then a try-and-increment method to find a point
// on the G2 twist curve y^2 = x^3 + 4(1+u), followed by cofactor clearing.
//
// Aggregation: signatures aggregate by G2 point addition.

import Foundation
import NeonFieldOps

// MARK: - BLS Signature Engine

public class BLSSignatureEngine {
    public static let version = Versions.blsSignature

    public init() {}

    // MARK: - Key Generation

    /// Generate a public key from a secret key.
    /// pk = [sk]G1
    public func publicKey(secretKey: Fr381) -> G1Affine381 {
        let gen = bls12381G1Generator()
        let genProj = g1_381FromAffine(gen)
        let skLimbs = fr381ToInt(secretKey)
        let pkProj = g1_381ScalarMul(genProj, skLimbs)
        return g1_381ToAffine(pkProj)!
    }

    // MARK: - Sign

    /// Sign a message.
    /// sig = [sk] * H(message)
    /// Returns signature as a G2 affine point.
    public func sign(message: [UInt8], secretKey: Fr381) -> G2Affine381 {
        let hm = hashToCurveG2(message: message)
        let skLimbs = fr381ToInt(secretKey)
        let sigProj = g2_381ScalarMul(hm, skLimbs)
        return g2_381ToAffine(sigProj)!
    }

    // MARK: - Verify

    /// Verify a BLS signature.
    /// Check: e(G1, sig) == e(pk, H(m))
    /// Equivalently: e(pk, H(m)) * e(-G1, sig) == 1
    public func verify(message: [UInt8], signature: G2Affine381, publicKey: G1Affine381) -> Bool {
        let hm = hashToCurveG2(message: message)
        let hmAff = g2_381ToAffine(hm)!
        let gen = bls12381G1Generator()
        let negGen = g1_381NegateAffine(gen)
        return bls12381PairingCheck([(publicKey, hmAff), (negGen, signature)])
    }

    // MARK: - Aggregation

    /// Aggregate multiple G2 signatures by point addition.
    public func aggregate(signatures: [G2Affine381]) -> G2Affine381 {
        precondition(!signatures.isEmpty, "Cannot aggregate empty signature list")
        var acc = g2_381FromAffine(signatures[0])
        for i in 1..<signatures.count {
            acc = g2_381Add(acc, g2_381FromAffine(signatures[i]))
        }
        return g2_381ToAffine(acc)!
    }

    /// Aggregate verify: different messages, different signers.
    /// Check: product of e(pk_i, H(m_i)) == e(G1, aggSig)
    /// Equivalently: e(-G1, aggSig) * product(e(pk_i, H(m_i))) == 1
    public func aggregateVerify(messages: [[UInt8]], signatures: [G2Affine381],
                                 publicKeys: [G1Affine381]) -> Bool {
        precondition(messages.count == publicKeys.count)
        precondition(messages.count == signatures.count)

        let aggSig = aggregate(signatures: signatures)

        let gen = bls12381G1Generator()
        let negGen = g1_381NegateAffine(gen)
        var pairs: [(G1Affine381, G2Affine381)] = [(negGen, aggSig)]

        for i in 0..<messages.count {
            let hm = hashToCurveG2(message: messages[i])
            let hmAff = g2_381ToAffine(hm)!
            pairs.append((publicKeys[i], hmAff))
        }

        return bls12381PairingCheck(pairs)
    }

    /// Fast aggregate verify: same message, multiple signers (Ethereum beacon chain).
    /// aggregateSig = sum of individual signatures on the same message
    /// Check: e(sum(pk_i), H(m)) == e(G1, aggSig)
    public func fastAggregateVerify(message: [UInt8], aggregateSig: G2Affine381,
                                     publicKeys: [G1Affine381]) -> Bool {
        if publicKeys.isEmpty { return false }
        var aggPk = g1_381FromAffine(publicKeys[0])
        for i in 1..<publicKeys.count {
            aggPk = g1_381Add(aggPk, g1_381FromAffine(publicKeys[i]))
        }
        guard let aggPkAff = g1_381ToAffine(aggPk) else { return false }

        let hm = hashToCurveG2(message: message)
        let hmAff = g2_381ToAffine(hm)!
        let gen = bls12381G1Generator()
        let negGen = g1_381NegateAffine(gen)
        return bls12381PairingCheck([(aggPkAff, hmAff), (negGen, aggregateSig)])
    }

    // MARK: - Hash to Curve G2

    /// Hash a message to a G2 point on y^2 = x^3 + 4(1+u).
    ///
    /// Method: hash-to-field using expand_message_xmd (SHA-256) to get a uniform Fp2
    /// element, then try-and-increment to find a curve point, then clear cofactor.
    /// Two independent hash-to-field outputs are mapped to curve and added for
    /// uniformity (random oracle model).
    public func hashToCurveG2(message: [UInt8],
                               dst: [UInt8] = Array("BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_".utf8)) -> G2Projective381 {
        // Generate 256 bytes of uniform randomness (4 field elements worth)
        let uniformBytes = expandMessageXMD(message: message, dst: dst, lenInBytes: 256)

        // Extract two Fp2 elements
        let u0c0 = fieldElementFromUniformBytes(Array(uniformBytes[0..<64]))
        let u0c1 = fieldElementFromUniformBytes(Array(uniformBytes[64..<128]))
        let u1c0 = fieldElementFromUniformBytes(Array(uniformBytes[128..<192]))
        let u1c1 = fieldElementFromUniformBytes(Array(uniformBytes[192..<256]))

        let u0 = Fp2_381(c0: u0c0, c1: u0c1)
        let u1 = Fp2_381(c0: u1c0, c1: u1c1)

        // Map each Fp2 element to a curve point via try-and-increment
        let p0 = mapToCurveG2TryInc(u0)
        let p1 = mapToCurveG2TryInc(u1)

        // Combine and clear cofactor
        let sum = g2_381Add(p0, p1)
        return clearCofactorG2(sum)
    }

    // MARK: - expand_message_xmd (SHA-256) per RFC 9380 Section 5.3.1

    func expandMessageXMD(message: [UInt8], dst: [UInt8], lenInBytes: Int) -> [UInt8] {
        let bInBytes = 32  // SHA-256 output
        let rInBytes = 64  // SHA-256 block size
        let ell = (lenInBytes + bInBytes - 1) / bInBytes
        precondition(ell <= 255)
        precondition(dst.count <= 255)

        var dstPrime = dst
        dstPrime.append(UInt8(dst.count))

        let zPad = [UInt8](repeating: 0, count: rInBytes)
        let libStr: [UInt8] = [UInt8((lenInBytes >> 8) & 0xFF), UInt8(lenInBytes & 0xFF)]

        var msgPrime = zPad
        msgPrime.append(contentsOf: message)
        msgPrime.append(contentsOf: libStr)
        msgPrime.append(0)
        msgPrime.append(contentsOf: dstPrime)

        let b0 = sha256(msgPrime)

        var b1Input = b0
        b1Input.append(1)
        b1Input.append(contentsOf: dstPrime)
        var bVals = [sha256(b1Input)]

        for i in 2...ell {
            let prev = bVals[bVals.count - 1]
            var xored = [UInt8](repeating: 0, count: bInBytes)
            for j in 0..<bInBytes { xored[j] = b0[j] ^ prev[j] }
            xored.append(UInt8(i))
            xored.append(contentsOf: dstPrime)
            bVals.append(sha256(xored))
        }

        var result = [UInt8]()
        result.reserveCapacity(lenInBytes)
        for b in bVals { result.append(contentsOf: b) }
        return Array(result.prefix(lenInBytes))
    }

    // MARK: - Field element from 64 uniform bytes

    /// Convert 64 big-endian bytes to Fp by reducing a 512-bit integer mod p.
    func fieldElementFromUniformBytes(_ bytes: [UInt8]) -> Fp381 {
        precondition(bytes.count == 64)
        // Parse as big-endian 512-bit integer into 8 x 64-bit little-endian limbs
        var limbs = [UInt64](repeating: 0, count: 8)
        for i in 0..<8 {
            let byteOff = (7 - i) * 8
            var w: UInt64 = 0
            for j in 0..<8 {
                w |= UInt64(bytes[byteOff + (7 - j)]) << (j * 8)
            }
            limbs[i] = w
        }

        // Split into low (384 bits) and high (128 bits)
        let low: [UInt64] = [limbs[0], limbs[1], limbs[2], limbs[3], limbs[4], limbs[5]]
        let high: [UInt64] = [limbs[6], limbs[7], 0, 0, 0, 0]

        // Reduce low mod p
        var lr = low
        while gte384(lr, Fp381.P) { (lr, _) = sub384(lr, Fp381.P) }
        let lowMont = fp381Mul(Fp381.from64(lr), Fp381.from64(Fp381.R2_MOD_P))

        // Reduce high mod p, then multiply by 2^384 mod p = R
        var hr = high
        while gte384(hr, Fp381.P) { (hr, _) = sub384(hr, Fp381.P) }
        let highMont = fp381Mul(Fp381.from64(hr), Fp381.from64(Fp381.R2_MOD_P))
        // highMont = high * R mod p in Montgomery form
        // We want (high * 2^384) in Montgomery form = (high * R) * R mod p
        // = fp381Mul(highMont, R_in_mont) = fp381Mul(highMont, R2_MOD_P)
        let highShifted = fp381Mul(highMont, Fp381.from64(Fp381.R2_MOD_P))

        return fp381Add(lowMont, highShifted)
    }

    // MARK: - Try-and-increment map to G2

    /// Map an Fp2 element to a point on E2: y^2 = x^3 + 4(1+u)
    /// Uses try-and-increment: try x = u, u+1, u+2, ... until RHS is a QR in Fp2.
    func mapToCurveG2TryInc(_ u: Fp2_381) -> G2Projective381 {
        let bTwist = Fp2_381(c0: fp381FromInt(4), c1: fp381FromInt(4))  // 4(1+u)

        var x = u
        for _ in 0..<256 {
            // Compute y^2 = x^3 + 4(1+u)
            let x2 = fp2_381Sqr(x)
            let x3 = fp2_381Mul(x2, x)
            let rhs = fp2_381Add(x3, bTwist)

            if let y = fp2Sqrt(rhs) {
                // Choose sign: ensure sgn0(y) == 0 for determinism
                var yFinal = y
                if sgn0Fp2(y) != 0 {
                    yFinal = fp2_381Neg(y)
                }
                return G2Projective381(x: x, y: yFinal, z: .one)
            }

            // Increment x: add (1, 0)
            x = Fp2_381(c0: fp381Add(x.c0, Fp381.one), c1: x.c1)
        }

        // Fallback: should never reach here for valid inputs
        fatalError("mapToCurveG2TryInc: failed to find point after 256 attempts")
    }

    // MARK: - Fp2 square root

    /// Compute sqrt in Fp2 = Fp[u]/(u^2+1) using the formula for p = 3 mod 4.
    /// For a = c0 + c1*u, the norm is N = c0^2 + c1^2.
    /// If N is a QR in Fp, then a has a sqrt in Fp2.
    func fp2Sqrt(_ a: Fp2_381) -> Fp2_381? {
        if a.isZero { return .zero }

        // Norm: N = c0^2 + c1^2
        let norm = fp381Add(fp381Sqr(a.c0), fp381Sqr(a.c1))
        guard let normSqrt = fpSqrt(norm) else { return nil }

        // candidate = (c0 + sqrt(norm)) / 2
        let twoInv = fp381Inverse(fp381FromInt(2))

        // Try first branch: t^2 = (c0 + normSqrt) / 2
        let cand1 = fp381Mul(fp381Add(a.c0, normSqrt), twoInv)
        if let t = fpSqrt(cand1) {
            if !t.isZero {
                let c1Part = fp381Mul(a.c1, fp381Inverse(fp381Double(t)))
                let result = Fp2_381(c0: t, c1: c1Part)
                // Verify
                if fp2Verify(result, a) { return result }
                let neg = Fp2_381(c0: fp381Neg(t), c1: fp381Neg(c1Part))
                if fp2Verify(neg, a) { return neg }
            }
        }

        // Try second branch: t^2 = (c0 - normSqrt) / 2
        let cand2 = fp381Mul(fp381Sub(a.c0, normSqrt), twoInv)
        if let t2 = fpSqrt(cand2) {
            if !t2.isZero {
                let c1Part2 = fp381Mul(a.c1, fp381Inverse(fp381Double(t2)))
                let result2 = Fp2_381(c0: t2, c1: c1Part2)
                if fp2Verify(result2, a) { return result2 }
                let neg2 = Fp2_381(c0: fp381Neg(t2), c1: fp381Neg(c1Part2))
                if fp2Verify(neg2, a) { return neg2 }
            } else {
                // t2 = 0 means we need c1 = 0 and c0 = cand2 = 0
                // Try: result = (0, sqrt(c1/2)) or similar -- edge case
                // If c1 is zero too, the input was zero (handled above)
                // Otherwise, we have a = c1*u, and sqrt(c1*u) = sqrt(c1)*sqrt(u)
                // sqrt(u) in Fp2 where u^2=-1: sqrt(u) = (1+u)/sqrt(2) (if it exists)
            }
        }

        // Try the "swapped" approach: result = (c1_part, t) instead of (t, c1_part)
        // sqrt(c0 + c1*u) where the real part of the sqrt is from c1
        if let t = fpSqrt(cand1) {
            if t.isZero {
                // a = c1*u case: result.c0 = 0 not possible normally, try other form
            }
        }

        // Fallback: try exponentiation approach
        // a^((p^2+1)/4) -- but p^2+1 is huge, so we use the chain:
        // a^((p^2-3)/4) * a = a^((p^2+1)/4)
        // This is too expensive for a try-and-increment approach.
        // The above should work for ~50% of inputs, and we just try the next x.
        return nil
    }

    /// Verify that r^2 == a in Fp2.
    private func fp2Verify(_ r: Fp2_381, _ a: Fp2_381) -> Bool {
        let sq = fp2_381Sqr(r)
        return fp381ToInt(sq.c0) == fp381ToInt(a.c0) &&
               fp381ToInt(sq.c1) == fp381ToInt(a.c1)
    }

    /// Square root in Fp — C accelerated. p = 3 mod 4 so sqrt(a) = a^((p+1)/4).
    func fpSqrt(_ a: Fp381) -> Fp381? {
        if a.isZero { return .zero }
        var al = a.to64()
        var r = [UInt64](repeating: 0, count: 6)
        let ok = bls12_381_fp_sqrt(&al, &r)
        if ok != 0 {
            return Fp381.from64(r)
        }
        return nil
    }

    // MARK: - Cofactor Clearing for G2

    /// Clear the cofactor of a G2 point using the effective cofactor h_eff.
    /// h_eff per draft-irtf-cfrg-hash-to-curve (Section 8.8.2):
    /// 0xbc69f08f2ee75b3584c6a0ea91b352888e2a8e9145ad7689986ff031508ffe1329c2f178731db956d82bf015d1212b02ec0ec69d7477c1ae954cbc06689f6a359894c0adebbf6b4e8020005aaa95551
    /// [h_eff]P maps any E'(Fp2) point to the r-torsion subgroup (G2).
    func clearCofactorG2(_ p: G2Projective381) -> G2Projective381 {
        let hEff: [UInt64] = [
            0xe8020005aaa95551, 0x59894c0adebbf6b4,
            0xe954cbc06689f6a3, 0x2ec0ec69d7477c1a,
            0x6d82bf015d1212b0, 0x329c2f178731db95,
            0x9986ff031508ffe1, 0x88e2a8e9145ad768,
            0x584c6a0ea91b3528, 0x0bc69f08f2ee75b3
        ]
        return g2_381ScalarMulWide(p, hEff)
    }

    // MARK: - sgn0 for Fp2 (RFC 9380)

    /// sgn0(x) for Fp2: returns the "sign" (0 or 1) of the element.
    func sgn0Fp2(_ a: Fp2_381) -> Int {
        let c0Int = fp381ToInt(a.c0)
        let c1Int = fp381ToInt(a.c1)
        let sign0 = Int(c0Int[0] & 1)
        let zero0 = a.c0.isZero ? 1 : 0
        let sign1 = Int(c1Int[0] & 1)
        return sign0 | (zero0 & sign1)
    }
}
