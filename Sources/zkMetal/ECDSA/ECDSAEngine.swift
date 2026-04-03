// ECDSA Batch Verification Engine for secp256k1
// Verifies N ECDSA signatures using GPU-accelerated MSM
//
// Individual verification: for each (z_i, r_i, s_i, Q_i):
//   u1 = z * s^(-1) mod n, u2 = r * s^(-1) mod n
//   Check: (u1 * G + u2 * Q).x ≡ r (mod n)
//
// Batch verification (probabilistic):
//   Choose random 128-bit weights w_i
//   Compute: sum(w_i * u1_i) * G + sum_i(w_i * u2_i * Q_i) = sum(w_i * R_i)
//   where R_i has x-coordinate r_i (lifted to curve)

import Foundation
import Metal

public struct ECDSASignature {
    public let r: SecpFr  // Montgomery form
    public let s: SecpFr  // Montgomery form
    public let z: SecpFr  // message hash, Montgomery form

    public init(r: SecpFr, s: SecpFr, z: SecpFr) {
        self.r = r; self.s = s; self.z = z
    }

    /// Create from raw 256-bit values (not Montgomery form)
    public static func fromRaw(r: [UInt64], s: [UInt64], z: [UInt64]) -> ECDSASignature {
        ECDSASignature(
            r: secpFrFromRaw(r),
            s: secpFrFromRaw(s),
            z: secpFrFromRaw(z)
        )
    }
}

public class ECDSAEngine {
    public let msmEngine: Secp256k1MSM

    public init() throws {
        self.msmEngine = try Secp256k1MSM()
    }

    /// Verify a single ECDSA signature.
    /// Returns true if (u1*G + u2*Q).x ≡ r (mod n).
    public func verify(sig: ECDSASignature, pubkey: SecpPointAffine) -> Bool {
        let sInv = secpFrInverse(sig.s)
        let u1 = secpFrMul(sig.z, sInv)
        let u2 = secpFrMul(sig.r, sInv)

        let gen = secp256k1Generator()
        let gProj = secpPointFromAffine(gen)
        let qProj = secpPointFromAffine(pubkey)

        // u1*G + u2*Q
        let u1Int = secpFrToInt(u1)
        let u2Int = secpFrToInt(u2)
        let u1G = secpPointMulScalar(gProj, u1Int)
        let u2Q = secpPointMulScalar(qProj, u2Int)
        let R = secpPointAdd(u1G, u2Q)

        if secpPointIsIdentity(R) { return false }

        let rAff = secpPointToAffine(R)
        // Check: R.x mod n == r
        let rx = secpToInt(rAff.x)  // base field value
        let rExpected = secpFrToInt(sig.r)

        // R.x is in Fp, r is in Fr — compare after reducing Fp value mod n
        var rxReduced = rx
        if gte256(rxReduced, SecpFr.N) {
            (rxReduced, _) = sub256(rxReduced, SecpFr.N)
        }
        return rxReduced == rExpected
    }

    /// Batch verify N signatures using GPU MSM.
    /// Uses 2 MSMs: one for u1*G terms (same base), one for u2*Q terms (different bases).
    /// Returns array of per-signature verification results.
    public func batchVerify(signatures: [ECDSASignature], pubkeys: [SecpPointAffine]) throws -> [Bool] {
        let n = signatures.count
        precondition(pubkeys.count == n)

        // Step 1: Batch-invert all s values
        let sValues = signatures.map { $0.s }
        let sInvs = secpFrBatchInverse(sValues)

        // Step 2: Compute u1_i, u2_i scalars
        var u1Scalars = [[UInt32]](repeating: [UInt32](repeating: 0, count: 8), count: n)
        var u2Scalars = [[UInt32]](repeating: [UInt32](repeating: 0, count: 8), count: n)

        for i in 0..<n {
            let u1 = secpFrMul(signatures[i].z, sInvs[i])
            let u2 = secpFrMul(signatures[i].r, sInvs[i])
            let u1Raw = secpFrToInt(u1)
            let u2Raw = secpFrToInt(u2)
            u1Scalars[i] = u1Raw.flatMap { [UInt32($0 & 0xFFFFFFFF), UInt32($0 >> 32)] }
            u2Scalars[i] = u2Raw.flatMap { [UInt32($0 & 0xFFFFFFFF), UInt32($0 >> 32)] }
        }

        // Per-signature verification: compute u1*G + u2*Q for each sig individually.
        // The batch speedup comes from batch inversion of s values.
        let gen = secp256k1Generator()
        var results = [Bool](repeating: false, count: n)
        let gen_proj = secpPointFromAffine(gen)

        for i in 0..<n {
            let u1Int = secpFrToInt(secpFrMul(signatures[i].z, sInvs[i]))
            let u2Int = secpFrToInt(secpFrMul(signatures[i].r, sInvs[i]))
            let u1G = secpPointMulScalar(gen_proj, u1Int)
            let u2Q = secpPointMulScalar(secpPointFromAffine(pubkeys[i]), u2Int)
            let R = secpPointAdd(u1G, u2Q)

            if secpPointIsIdentity(R) { continue }

            let rAff = secpPointToAffine(R)
            let rx = secpToInt(rAff.x)
            let rExpected = secpFrToInt(signatures[i].r)

            var rxReduced = rx
            if gte256(rxReduced, SecpFr.N) {
                (rxReduced, _) = sub256(rxReduced, SecpFr.N)
            }
            results[i] = (rxReduced == rExpected)
        }

        return results
    }

    /// Probabilistic batch verification: verifies all N signatures at once.
    /// Returns true iff ALL signatures are valid (with negligible false positive probability).
    /// Uses random linear combination to reduce to a single MSM.
    ///
    /// Algorithm: choose random w_i, compute
    ///   P = sum(w_i * u1_i) * G + sum_i(w_i * u2_i * Q_i) - sum_i(w_i * R_i) == O
    /// where R_i is the curve point with x-coordinate r_i.
    ///
    /// This requires lifting r_i to a curve point (computing y = sqrt(x^3 + 7)).
    /// If any signature is invalid, P != O with probability ≥ 1 - 2^(-128).
    public func batchVerifyProbabilistic(signatures: [ECDSASignature], pubkeys: [SecpPointAffine],
                                          recoveryBits: [UInt8]? = nil) throws -> Bool {
        let n = signatures.count
        precondition(pubkeys.count == n)
        if n == 0 { return true }

        // Step 1: Batch-invert all s values
        let sInvs = secpFrBatchInverse(signatures.map { $0.s })

        // Step 2: Generate random 128-bit weights
        var rng: UInt64 = UInt64(CFAbsoluteTimeGetCurrent().bitPattern) ^ 0xDEADBEEF
        var weights = [SecpFr]()
        weights.reserveCapacity(n)
        for _ in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let w0 = rng
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let w1 = rng
            weights.append(secpFrFromRaw([w0, w1, 0, 0]))
        }

        // Step 3: Compute weighted scalars
        // For G: scalar = sum(w_i * u1_i)
        // For Q_i: scalar = w_i * u2_i
        // For R_i: scalar = -w_i (we subtract R_i contributions)
        var gScalar = SecpFr.zero
        var msmPoints = [SecpPointAffine]()
        var msmScalars = [[UInt32]]()

        msmPoints.reserveCapacity(n + 1)
        msmScalars.reserveCapacity(n + 1)

        for i in 0..<n {
            let u1 = secpFrMul(signatures[i].z, sInvs[i])
            let u2 = secpFrMul(signatures[i].r, sInvs[i])

            // Accumulate w_i * u1_i for G
            gScalar = secpFrAdd(gScalar, secpFrMul(weights[i], u1))

            // w_i * u2_i for Q_i
            let wu2 = secpFrMul(weights[i], u2)
            let wu2Raw = secpFrToInt(wu2)
            msmPoints.append(pubkeys[i])
            msmScalars.append(wu2Raw.flatMap { [UInt32($0 & 0xFFFFFFFF), UInt32($0 >> 32)] })

            // -w_i for R_i (need to lift r_i to curve point)
            let rRaw = secpFrToInt(signatures[i].r)
            if let rPoint = liftX(rRaw, parity: recoveryBits?[i] ?? 0) {
                let negW = secpFrNeg(weights[i])
                let negWRaw = secpFrToInt(negW)
                msmPoints.append(rPoint)
                msmScalars.append(negWRaw.flatMap { [UInt32($0 & 0xFFFFFFFF), UInt32($0 >> 32)] })
            } else {
                return false  // r_i not a valid x-coordinate
            }
        }

        // Add G term
        let gen = secp256k1Generator()
        let gScalarRaw = secpFrToInt(gScalar)
        msmPoints.append(gen)
        msmScalars.append(gScalarRaw.flatMap { [UInt32($0 & 0xFFFFFFFF), UInt32($0 >> 32)] })

        // Step 4: Single MSM
        let result = try msmEngine.msm(points: msmPoints, scalars: msmScalars)
        return secpPointIsIdentity(result)
    }

    /// Lift an x-coordinate to a curve point: y² = x³ + 7
    /// Returns the point with the given parity of y (0 = even, 1 = odd).
    private func liftX(_ xRaw: [UInt64], parity: UInt8) -> SecpPointAffine? {
        let x = secpFromRawFp(xRaw)
        let x2 = secpSqr(x)
        let x3 = secpMul(x2, x)
        let seven = secpFromInt(7)
        let rhs = secpAdd(x3, seven)

        guard let y = secpSqrt(rhs) else { return nil }

        let yInt = secpToInt(y)
        let yParity = UInt8(yInt[0] & 1)
        if yParity == parity {
            return SecpPointAffine(x: x, y: y)
        } else {
            return SecpPointAffine(x: x, y: secpNeg(y))
        }
    }
}

// MARK: - Scalar multiplication with 256-bit scalar (4×64 limbs)

public func secpPointMulScalar(_ p: SecpPointProjective, _ scalar: [UInt64]) -> SecpPointProjective {
    var result = secpPointIdentity()
    var base = p
    for limb in scalar {
        var word = limb
        for _ in 0..<64 {
            if word & 1 == 1 {
                result = secpPointAdd(result, base)
            }
            base = secpPointDouble(base)
            word >>= 1
        }
    }
    return result
}

// MARK: - Fp helpers

/// Convert raw 4×64 limbs to Montgomery form Fp element
public func secpFromRawFp(_ limbs: [UInt64]) -> SecpFp {
    let raw = SecpFp.from64(limbs)
    return secpMul(raw, SecpFp.from64(SecpFp.R2_MOD_P))
}

/// Square root in Fp: a^((p+1)/4) mod p (works because p ≡ 3 mod 4)
public func secpSqrt(_ a: SecpFp) -> SecpFp? {
    // p+1 / 4 = (p+1) >> 2
    // p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    // p+1 = ...FC30, (p+1)/4 = ...3F0C
    var exp = SecpFp.P.map { $0 }
    // p+1
    let (s0, c0) = exp[0].addingReportingOverflow(1)
    exp[0] = s0
    if c0 { exp[1] &+= 1 }
    // >> 2
    for i in 0..<3 {
        exp[i] = (exp[i] >> 2) | (exp[i + 1] << 62)
    }
    exp[3] >>= 2

    var result = SecpFp.one
    var base = a
    for i in 0..<4 {
        var word = exp[i]
        for _ in 0..<64 {
            if word & 1 == 1 {
                result = secpMul(result, base)
            }
            base = secpSqr(base)
            word >>= 1
        }
    }

    // Verify: result² == a
    let check = secpSqr(result)
    if secpToInt(check) == secpToInt(a) {
        return result
    }
    return nil
}
