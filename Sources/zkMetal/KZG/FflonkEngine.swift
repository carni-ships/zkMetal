// Fflonk Polynomial Commitment Scheme
//
// Batches k polynomial evaluations into a single KZG opening by encoding them
// at roots of unity, reducing proof size and verifier cost.
//
// Given polynomials p_0, ..., p_{k-1} each of degree < d, the combined polynomial is:
//   P(X) = sum_{i=0}^{k-1} p_i(X^k) * X^i
//
// This interleaves the polynomials so that evaluating P at omega^j * z recovers
// p_j(z^k) where omega is a primitive k-th root of unity.
//
// The opening proof is a single KZG witness for P at the k points
// {z, omega*z, omega^2*z, ..., omega^{k-1}*z}, which share vanishing polynomial
// Z(X) = X^k - z^k, enabling a single-witness opening.
//
// Supported batch sizes: 1, 2, 4, 8 (power-of-two).
//
// Reference: "fflonk: a Fast-Fourier inspired verifier efficient version of PlonK"
//            (Gabizon, Khovratovich, 2021)

import Foundation
import NeonFieldOps

// MARK: - Fflonk SRS

/// Structured reference string for fflonk (wraps a standard KZG SRS).
/// The SRS must be large enough to commit to the combined polynomial P(X)
/// of degree k*d - 1, where k is the batch size and d is the max polynomial degree.
public struct FflonkSRS {
    public let points: [PointAffine]   // [G, sG, s^2 G, ..., s^{N-1} G]
    public let secret: [UInt32]?       // toxic waste (testing only)

    public init(points: [PointAffine], secret: [UInt32]? = nil) {
        self.points = points
        self.secret = secret
    }

    /// Generate a test SRS (NOT secure).
    public static func generateTest(secret: [UInt32], size: Int, generator: PointAffine) -> FflonkSRS {
        let pts = KZGEngine.generateTestSRS(secret: secret, size: size, generator: generator)
        return FflonkSRS(points: pts, secret: secret)
    }
}

// MARK: - Fflonk Commitment

/// A commitment to k polynomials via a single group element.
public struct FflonkCommitment {
    /// The KZG commitment to the combined polynomial P(X).
    public let point: PointProjective
    /// Number of batched polynomials.
    public let batchSize: Int

    public init(point: PointProjective, batchSize: Int) {
        self.point = point
        self.batchSize = batchSize
    }
}

// MARK: - Fflonk Opening Proof

/// Proof that polynomials p_0,...,p_{k-1} evaluate to claimed values at z.
public struct FflonkOpeningProof {
    /// Single KZG witness point [q(s)] where q = (P(X) - R(X)) / Z(X).
    public let witness: PointProjective
    /// Evaluations p_i(z^k) for i in 0..<k.
    public let evaluations: [Fr]
    /// The evaluation point z used.
    public let point: Fr
    /// The batch size k.
    public let batchSize: Int

    public init(witness: PointProjective, evaluations: [Fr], point: Fr, batchSize: Int) {
        self.witness = witness
        self.evaluations = evaluations
        self.point = point
        self.batchSize = batchSize
    }
}

// MARK: - Fflonk Engine

public class FflonkEngine {
    public let kzg: KZGEngine

    public init(kzg: KZGEngine) {
        self.kzg = kzg
    }

    /// Convenience init from an FflonkSRS.
    public convenience init(srs: FflonkSRS) throws {
        let kzg = try KZGEngine(srs: srs.points)
        self.init(kzg: kzg)
    }

    // MARK: - Combined Polynomial Construction

    /// Build the combined polynomial P(X) = sum_{i=0}^{k-1} p_i(X^k) * X^i.
    ///
    /// If p_i has coefficients [a_0, a_1, ..., a_{d-1}], then p_i(X^k) has
    /// non-zero coefficients at positions 0, k, 2k, .... Multiplying by X^i shifts
    /// them to positions i, k+i, 2k+i, ....
    ///
    /// The resulting polynomial has degree k*d - 1 where d = max degree of input polys.
    public static func buildCombinedPoly(_ polynomials: [[Fr]], batchSize k: Int) -> [Fr] {
        guard !polynomials.isEmpty else { return [] }

        let d = polynomials.map { $0.count }.max()!
        let combinedDeg = k * d
        var combined = [Fr](repeating: Fr.zero, count: combinedDeg)

        for i in 0..<min(k, polynomials.count) {
            let poly = polynomials[i]
            for j in 0..<poly.count {
                // Coefficient j of p_i goes to position k*j + i in combined
                let idx = k * j + i
                if idx < combinedDeg {
                    combined[idx] = poly[j]
                }
            }
        }

        return combined
    }

    /// Compute the primitive k-th root of unity in Fr (BN254 scalar field).
    /// k must be a power of 2 and k <= 2^TWO_ADICITY.
    public static func rootOfUnity(k: Int) -> Fr {
        precondition(k > 0 && (k & (k - 1)) == 0, "k must be a power of 2")
        let logK = Int(log2(Double(k)))
        return frRootOfUnity(logN: logK)
    }

    // MARK: - Commitment

    /// Commit to k polynomials as a single group element.
    /// The polynomials are interleaved into P(X) and committed via KZG.
    public func commit(_ polynomials: [[Fr]]) throws -> FflonkCommitment {
        let k = nextPowerOf2(polynomials.count)
        precondition(k >= 1 && k <= 8, "Batch size must be 1, 2, 4, or 8")

        // Pad to power-of-2 batch size with zero polynomials
        var padded = polynomials
        while padded.count < k {
            padded.append([Fr.zero])
        }

        let combined = FflonkEngine.buildCombinedPoly(padded, batchSize: k)
        let point = try kzg.commit(combined)
        return FflonkCommitment(point: point, batchSize: k)
    }

    // MARK: - Opening (Prover)

    /// Open the batched commitment at evaluation point z.
    ///
    /// Computes evaluations y_i = p_i(z^k) for each polynomial, then constructs
    /// the remainder polynomial R(X) that interpolates P at {omega^j * z} to y_j,
    /// and produces a single KZG witness for (P(X) - R(X)) / Z(X) where
    /// Z(X) = X^k - z^k.
    ///
    /// The key insight: evaluating P at omega^j * z gives
    ///   P(omega^j * z) = sum_{i=0}^{k-1} p_i((omega^j * z)^k) * (omega^j * z)^i
    ///                   = sum_{i=0}^{k-1} p_i(z^k) * omega^{ij} * z^i
    /// which is a DFT of the vector [p_0(z^k)*z^0, p_1(z^k)*z^1, ...] evaluated at omega^j.
    public func open(_ polynomials: [[Fr]], at z: Fr) throws -> FflonkOpeningProof {
        let k = nextPowerOf2(polynomials.count)
        precondition(k >= 1 && k <= 8, "Batch size must be 1, 2, 4, or 8")

        var padded = polynomials
        while padded.count < k {
            padded.append([Fr.zero])
        }

        let combined = FflonkEngine.buildCombinedPoly(padded, batchSize: k)

        // z^k
        let zk = frPowBig(z, UInt64(k))

        // Evaluate each sub-polynomial at z^k
        var evaluations = [Fr]()
        evaluations.reserveCapacity(k)
        for i in 0..<k {
            evaluations.append(cEvaluate(padded[i], at: zk))
        }

        // Build the remainder polynomial R(X) of degree < k that satisfies:
        //   R(omega^j * z) = P(omega^j * z) for j = 0,...,k-1
        //
        // P(omega^j * z) = sum_{i=0}^{k-1} p_i(z^k) * omega^{ij} * z^i
        //
        // R(X) is the unique polynomial of degree < k interpolating these values
        // at the k points {z, omega*z, ..., omega^{k-1}*z}.
        let omega = FflonkEngine.rootOfUnity(k: k)
        let remainder = buildRemainderPoly(evaluations: evaluations, z: z, omega: omega, k: k)

        // Compute the numerator N(X) = P(X) - R(X)
        var numerator = combined
        for i in 0..<min(remainder.count, numerator.count) {
            numerator[i] = frSub(numerator[i], remainder[i])
        }

        // Divide by vanishing polynomial Z(X) = X^k - z^k
        // Z(X) has coefficients: [-z^k, 0, ..., 0, 1] (degree k)
        let quotient = divideByVanishing(numerator, zk: zk, k: k)

        // Commit to quotient as the witness
        let witness = try kzg.commit(quotient)

        return FflonkOpeningProof(
            witness: witness,
            evaluations: evaluations,
            point: z,
            batchSize: k
        )
    }

    // MARK: - Verification

    /// Verify an fflonk opening proof using the SRS secret (testing).
    ///
    /// Checks: [P(s)] - [R(s)] == (s^k - z^k) * [q(s)]
    /// i.e., commitment - R_commitment == (s^k - z^k) * witness
    ///
    /// In production, this is a single pairing check:
    ///   e(C - [R(s)], [1]_2) == e(W, [s^k]_2 - [z^k]_2)
    public func verify(
        commitment: FflonkCommitment,
        proof: FflonkOpeningProof,
        srsSecret: Fr
    ) -> Bool {
        let k = proof.batchSize
        let z = proof.point
        let omega = FflonkEngine.rootOfUnity(k: k)

        // Rebuild remainder polynomial from evaluations
        let remainder = buildRemainderPoly(
            evaluations: proof.evaluations, z: z, omega: omega, k: k
        )

        // Evaluate remainder at the SRS secret s
        let rAtS = cEvaluate(remainder, at: srsSecret)

        // LHS: C - [R(s)] * G
        let g1 = pointFromAffine(kzg.srs[0])
        let rG = cPointScalarMul(g1, rAtS)
        let lhs = pointAdd(commitment.point, pointNeg(rG))

        // RHS: (s^k - z^k) * W
        let sk = frPowBig(srsSecret, UInt64(k))
        let zk = frPowBig(z, UInt64(k))
        let vanishAtS = frSub(sk, zk)
        let rhs = cPointScalarMul(proof.witness, vanishAtS)

        // Compare
        return pointsEqual(lhs, rhs)
    }

    /// Verify using re-computation (no SRS secret needed, but expensive).
    /// Re-opens the polynomials and checks the proof matches.
    public func verifyByReopen(
        polynomials: [[Fr]],
        commitment: FflonkCommitment,
        proof: FflonkOpeningProof
    ) throws -> Bool {
        let recomputed = try open(polynomials, at: proof.point)

        // Check evaluations match
        guard recomputed.evaluations.count == proof.evaluations.count else { return false }
        for i in 0..<proof.evaluations.count {
            if frToInt(recomputed.evaluations[i]) != frToInt(proof.evaluations[i]) {
                return false
            }
        }

        // Check witness matches
        return pointsEqual(recomputed.witness, proof.witness)
    }

    // MARK: - Internal Helpers

    /// Evaluate polynomial at a point using C Horner.
    private func cEvaluate(_ coeffs: [Fr], at z: Fr) -> Fr {
        if coeffs.isEmpty { return Fr.zero }
        var result = Fr.zero
        coeffs.withUnsafeBytes { cBuf in
            withUnsafeBytes(of: z) { zBuf in
                withUnsafeMutableBytes(of: &result) { rBuf in
                    bn254_fr_horner_eval(
                        cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(coeffs.count),
                        zBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                    )
                }
            }
        }
        return result
    }

    /// Compute a^n mod r using frPow (for small n).
    private func frPowBig(_ a: Fr, _ n: UInt64) -> Fr {
        return frPow(a, n)
    }

    /// Build the remainder polynomial R(X) of degree < k that interpolates
    /// the values {P(omega^j * z)} at points {omega^j * z} for j=0,...,k-1.
    ///
    /// Uses Lagrange interpolation over the k evaluation points.
    private func buildRemainderPoly(evaluations: [Fr], z: Fr, omega: Fr, k: Int) -> [Fr] {
        // Evaluation points: x_j = omega^j * z for j = 0,...,k-1
        var points = [Fr]()
        points.reserveCapacity(k)
        var omegaPow = Fr.one
        for _ in 0..<k {
            points.append(frMul(omegaPow, z))
            omegaPow = frMul(omegaPow, omega)
        }

        // Values at these points: P(x_j) = sum_i y_i * omega^{ij} * z^i
        var values = [Fr]()
        values.reserveCapacity(k)
        for j in 0..<k {
            var val = Fr.zero
            var omegaIJ = Fr.one  // omega^{i*j}
            var zPow = Fr.one     // z^i
            for i in 0..<k {
                // y_i * omega^{ij} * z^i
                val = frAdd(val, frMul(evaluations[i], frMul(omegaIJ, zPow)))
                omegaIJ = frMul(omegaIJ, frPow(omega, UInt64(j)))
                zPow = frMul(zPow, z)
            }
            values.append(val)
        }

        // Lagrange interpolation to get R(X) of degree < k
        return lagrangeInterpolate(points: points, values: values)
    }

    /// Standard Lagrange interpolation: given (x_i, y_i) pairs, compute the
    /// unique polynomial of degree < n passing through all points.
    /// Returns coefficient form [a_0, a_1, ..., a_{n-1}].
    private func lagrangeInterpolate(points: [Fr], values: [Fr]) -> [Fr] {
        let n = points.count
        if n == 0 { return [] }
        if n == 1 { return [values[0]] }

        var result = [Fr](repeating: Fr.zero, count: n)

        // Precompute all Lagrange denominators and batch-invert
        var denoms = [Fr](repeating: Fr.one, count: n)
        for i in 0..<n {
            for j in 0..<n where j != i {
                denoms[i] = frMul(denoms[i], frSub(points[i], points[j]))
            }
        }
        var denomInvs = [Fr](repeating: Fr.zero, count: n)
        denoms.withUnsafeBytes { src in
            denomInvs.withUnsafeMutableBytes { dst in
                bn254_fr_batch_inverse(
                    src.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(n),
                    dst.baseAddress!.assumingMemoryBound(to: UInt64.self))
            }
        }

        for i in 0..<n {
            let coeff = frMul(values[i], denomInvs[i])

            // Build the numerator polynomial: prod_{j!=i} (X - x_j)
            // Start with [1] and multiply by (X - x_j) for each j != i
            var basis = [Fr](repeating: Fr.zero, count: n)
            basis[0] = Fr.one
            var deg = 0

            for j in 0..<n {
                if j == i { continue }
                // Multiply current basis by (X - x_j)
                let negXj = frNeg(points[j])
                // Expand in reverse to avoid overwriting
                for d in stride(from: deg + 1, through: 1, by: -1) {
                    basis[d] = frAdd(basis[d - 1], frMul(basis[d], negXj))
                }
                basis[0] = frMul(basis[0], negXj)
                deg += 1
            }

            // Accumulate coeff * basis into result
            for d in 0..<n {
                result[d] = frAdd(result[d], frMul(coeff, basis[d]))
            }
        }

        return result
    }

    /// Divide polynomial N(X) by Z(X) = X^k - z^k.
    /// Z(X) = -z^k + 0*X + ... + 0*X^{k-1} + X^k
    ///
    /// Uses long division. The numerator should be exactly divisible (remainder = 0).
    private func divideByVanishing(_ numerator: [Fr], zk: Fr, k: Int) -> [Fr] {
        let n = numerator.count
        if n <= k { return [] }

        let quotientDeg = n - k
        var remainder = numerator  // working copy

        var quotient = [Fr](repeating: Fr.zero, count: quotientDeg)

        // Long division: divide by X^k - z^k (leading coeff = 1 at degree k)
        for i in stride(from: n - 1, through: k, by: -1) {
            let qi = i - k  // quotient index
            let c = remainder[i]
            quotient[qi] = c

            // Subtract c * Z(X) shifted by qi:
            // Z(X) = X^k - z^k, so shifted = c*X^{qi+k} - c*z^k*X^{qi}
            // remainder[qi + k] -= c * 1  (but that's remainder[i], set to 0)
            remainder[i] = Fr.zero
            // remainder[qi] -= c * (-z^k) = remainder[qi] + c * z^k
            remainder[qi] = frAdd(remainder[qi], frMul(c, zk))
        }

        return quotient
    }

    /// Compare two projective points for equality.
    private func pointsEqual(_ a: PointProjective, _ b: PointProjective) -> Bool {
        if pointIsIdentity(a) && pointIsIdentity(b) { return true }
        if pointIsIdentity(a) || pointIsIdentity(b) { return false }
        let aAff = batchToAffine([a])
        let bAff = batchToAffine([b])
        return fpToInt(aAff[0].x) == fpToInt(bAff[0].x) &&
               fpToInt(aAff[0].y) == fpToInt(bAff[0].y)
    }

    /// Smallest power of 2 >= n.
    private func nextPowerOf2(_ n: Int) -> Int {
        if n <= 1 { return 1 }
        var v = n - 1
        v |= v >> 1
        v |= v >> 2
        v |= v >> 4
        v |= v >> 8
        v |= v >> 16
        return v + 1
    }
}
