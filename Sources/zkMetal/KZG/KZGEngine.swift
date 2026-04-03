// KZG Polynomial Commitment Engine
// Composes MSM + polynomial operations for commit() and open().
// SRS (Structured Reference String) is provided externally.

import Foundation
import Metal

public struct KZGProof {
    public let evaluation: Fr      // p(z)
    public let witness: PointProjective  // [q(s)] where q(x) = (p(x) - p(z)) / (x - z)
}

public class KZGEngine {
    public let msmEngine: MetalMSM
    public let polyEngine: PolyEngine

    /// SRS points: [G, sG, s^2 G, ..., s^(d-1) G] in affine form
    public private(set) var srs: [PointAffine]

    public init(srs: [PointAffine]) throws {
        self.msmEngine = try MetalMSM()
        self.polyEngine = try PolyEngine()
        self.srs = srs
    }

    /// Generate a toy SRS for testing (NOT secure — uses known secret).
    /// secret: the toxic waste scalar s
    /// size: number of SRS points (max polynomial degree + 1)
    /// generator: base point G in affine form
    public static func generateTestSRS(secret: [UInt32], size: Int, generator: PointAffine) -> [PointAffine] {
        let gProj = pointFromAffine(generator)
        var points = [PointProjective]()
        points.reserveCapacity(size)
        var sPow = Fr.one  // s^0 = 1
        let sFr = frFromLimbs(secret)

        for _ in 0..<size {
            // Compute s^i * G via scalar multiplication
            let scalar = frToLimbs(sPow)
            var acc = pointIdentity()
            var base = gProj
            for limb in scalar {
                var word = limb
                for _ in 0..<32 {
                    if word & 1 == 1 {
                        acc = pointAdd(acc, base)
                    }
                    base = pointDouble(base)
                    word >>= 1
                }
            }
            points.append(acc)
            sPow = frMul(sPow, sFr)
        }
        return batchToAffine(points)
    }

    /// Commit to a polynomial: C = MSM(SRS[0..deg], coefficients)
    public func commit(_ coeffs: [Fr]) throws -> PointProjective {
        let n = coeffs.count
        guard n <= srs.count else {
            throw MSMError.invalidInput
        }
        let srsSlice = Array(srs.prefix(n))
        let scalars = coeffs.map { frToLimbs($0) }
        return try msmEngine.msm(points: srsSlice, scalars: scalars)
    }

    /// Open a polynomial at point z: compute evaluation and witness proof.
    /// Returns (p(z), proof_point) where proof_point = MSM(SRS, quotient_coeffs)
    /// and quotient = (p(x) - p(z)) / (x - z).
    public func open(_ coeffs: [Fr], at z: Fr) throws -> KZGProof {
        // Evaluate p(z)
        let evals = try polyEngine.evaluate(coeffs, at: [z])
        let pz = evals[0]

        // Compute quotient polynomial q(x) = (p(x) - p(z)) / (x - z)
        // Using synthetic division: if p(x) = a_n x^n + ... + a_0,
        // then q(x) = b_{n-1} x^{n-1} + ... + b_0
        // where b_{n-1} = a_n, b_{i-1} = a_i + z * b_i
        let n = coeffs.count
        guard n >= 2 else {
            // Constant polynomial: quotient is zero, witness is identity
            return KZGProof(evaluation: pz, witness: pointIdentity())
        }

        var quotient = [Fr](repeating: Fr.zero, count: n - 1)
        quotient[n - 2] = coeffs[n - 1]
        for i in stride(from: n - 3, through: 0, by: -1) {
            quotient[i] = frAdd(coeffs[i + 1], frMul(z, quotient[i + 1]))
        }

        // Witness = MSM(SRS[0..n-1], quotient)
        let srsSlice = Array(srs.prefix(n - 1))
        let scalars = quotient.map { frToLimbs($0) }
        let witness = try msmEngine.msm(points: srsSlice, scalars: scalars)

        return KZGProof(evaluation: pz, witness: witness)
    }
}

// MARK: - Fr <-> [UInt32] limb conversion helpers

/// Convert Fr (Montgomery form) to raw [UInt32] limbs (8 limbs, little-endian)
public func frToLimbs(_ a: Fr) -> [UInt32] {
    let raw = frToInt(a)  // [UInt64] in standard form
    return [
        UInt32(raw[0] & 0xFFFFFFFF), UInt32(raw[0] >> 32),
        UInt32(raw[1] & 0xFFFFFFFF), UInt32(raw[1] >> 32),
        UInt32(raw[2] & 0xFFFFFFFF), UInt32(raw[2] >> 32),
        UInt32(raw[3] & 0xFFFFFFFF), UInt32(raw[3] >> 32),
    ]
}

/// Convert raw [UInt32] limbs (8 limbs, little-endian) to Fr (Montgomery form)
public func frFromLimbs(_ limbs: [UInt32]) -> Fr {
    let raw: [UInt64] = [
        UInt64(limbs[0]) | (UInt64(limbs[1]) << 32),
        UInt64(limbs[2]) | (UInt64(limbs[3]) << 32),
        UInt64(limbs[4]) | (UInt64(limbs[5]) << 32),
        UInt64(limbs[6]) | (UInt64(limbs[7]) << 32),
    ]
    let a = Fr.from64(raw)
    return frMul(a, Fr.from64(Fr.R2_MOD_R))  // Convert to Montgomery form
}
