// GPUKZGMultiOpenEngine — GPU-accelerated KZG multi-point opening
//
// Opens a single polynomial (or batch of polynomials) at multiple evaluation
// points simultaneously, producing a compact proof.
//
// For a single polynomial f(X) opened at points S = {z_0, ..., z_{k-1}}:
//   1. Compute evaluations y_i = f(z_i)
//   2. Build interpolation polynomial I(X) with I(z_i) = y_i
//   3. Build vanishing polynomial Z_S(X) = prod_i (X - z_i)
//   4. Witness polynomial W(X) = (f(X) - I(X)) / Z_S(X)
//   5. Commit to W(X) via GPU MSM
//
// Batch mode: N polynomials each at their own point sets, combined via
// random linear combination with Fiat-Shamir challenge.
//
// All MSM operations go through MetalMSM (GPU).

import Foundation
import Metal
import NeonFieldOps

// MARK: - Multi-point opening proof structures

/// Proof for opening a single polynomial at multiple points.
public struct GPUMultiPointProof {
    /// Commitment to the polynomial
    public let commitment: PointProjective
    /// Evaluations at each opening point
    public let evaluations: [Fr]
    /// Witness commitment [W(s)]_1
    public let witness: PointProjective

    public init(commitment: PointProjective, evaluations: [Fr], witness: PointProjective) {
        self.commitment = commitment
        self.evaluations = evaluations
        self.witness = witness
    }
}

/// Proof for batch multi-point opening (N polynomials, each at multiple points).
public struct GPUBatchMultiPointProof {
    /// Commitments to each polynomial
    public let commitments: [PointProjective]
    /// For each polynomial: evaluations at the opening points
    public let evaluations: [[Fr]]
    /// Opening points for each polynomial
    public let points: [[Fr]]
    /// Combined witness commitment
    public let witness: PointProjective
    /// Batching challenge
    public let gamma: Fr

    public init(commitments: [PointProjective], evaluations: [[Fr]],
                points: [[Fr]], witness: PointProjective, gamma: Fr) {
        self.commitments = commitments
        self.evaluations = evaluations
        self.points = points
        self.witness = witness
        self.gamma = gamma
    }
}

// MARK: - GPU KZG Multi-Open Engine

public class GPUKZGMultiOpenEngine {
    public static let version = Versions.gpuKZGMultiOpen

    public let gpuKZG: GPUKZGEngine

    public init(srs: [PointAffine]) throws {
        self.gpuKZG = try GPUKZGEngine(srs: srs)
    }

    public init(gpuKZG: GPUKZGEngine) {
        self.gpuKZG = gpuKZG
    }

    // MARK: - Single polynomial, multiple points

    /// Open polynomial f(X) at points S = {z_0, ..., z_{k-1}}.
    ///
    /// Computes witness W(X) = (f(X) - I(X)) / Z_S(X) where:
    ///   I(X) is the Lagrange interpolation polynomial with I(z_i) = f(z_i)
    ///   Z_S(X) = prod_i (X - z_i)
    ///
    /// - Returns: GPUMultiPointProof with commitment, evaluations, and witness
    public func openMultiPoint(
        polynomial: [Fr],
        points: [Fr]
    ) throws -> GPUMultiPointProof {
        let n = polynomial.count
        guard n >= 1, !points.isEmpty else { throw MSMError.invalidInput }
        guard points.count < n else { throw MSMError.invalidInput }

        // Commit to f(X)
        let commitment = try gpuKZG.commit(polynomial)

        // Evaluate f at each point
        var evaluations = [Fr]()
        evaluations.reserveCapacity(points.count)
        for z in points {
            evaluations.append(hornerEval(polynomial, at: z))
        }

        // Build vanishing polynomial Z_S(X) = prod_i (X - z_i)
        let vanishing = buildVanishingPoly(roots: points)

        // Build interpolation polynomial I(X) with I(z_i) = y_i
        let interp = lagrangeInterpolation(points: points, values: evaluations)

        // Compute f(X) - I(X)
        let maxLen = max(polynomial.count, interp.count)
        var numerator = [Fr](repeating: Fr.zero, count: maxLen)
        for i in 0..<polynomial.count {
            numerator[i] = polynomial[i]
        }
        for i in 0..<interp.count {
            numerator[i] = frSub(numerator[i], interp[i])
        }

        // Divide by Z_S(X) — exact division
        let witness_coeffs = polyExactDivide(numerator, by: vanishing)

        // Commit to W(X) via GPU MSM
        let witnessCommitment: PointProjective
        if witness_coeffs.isEmpty || witness_coeffs.allSatisfy({ isZeroFr($0) }) {
            witnessCommitment = pointIdentity()
        } else {
            witnessCommitment = try gpuKZG.commit(witness_coeffs)
        }

        return GPUMultiPointProof(
            commitment: commitment,
            evaluations: evaluations,
            witness: witnessCommitment
        )
    }

    // MARK: - Batch multi-point opening

    /// Open N polynomials, each at their own set of points.
    ///
    /// Uses random linear combination with Fiat-Shamir:
    ///   Combined witness = sum_i gamma^i * W_i(X)
    /// where W_i is the witness for polynomial i.
    public func batchOpenMultiPoint(
        polynomials: [[Fr]],
        pointSets: [[Fr]],
        transcript: Transcript
    ) throws -> GPUBatchMultiPointProof {
        let numPolys = polynomials.count
        guard numPolys > 0, numPolys == pointSets.count else { throw MSMError.invalidInput }

        // Commit to all polynomials
        var commitments = [PointProjective]()
        commitments.reserveCapacity(numPolys)
        for poly in polynomials {
            commitments.append(try gpuKZG.commit(poly))
        }

        // Absorb commitments
        for c in commitments {
            absorbPoint(c, into: transcript)
        }

        // Evaluate and absorb
        var allEvals = [[Fr]]()
        allEvals.reserveCapacity(numPolys)
        for i in 0..<numPolys {
            var evals = [Fr]()
            evals.reserveCapacity(pointSets[i].count)
            for z in pointSets[i] {
                let y = hornerEval(polynomials[i], at: z)
                evals.append(y)
                transcript.absorb(y)
            }
            allEvals.append(evals)
        }

        // Squeeze combination challenge gamma
        let gamma = transcript.squeeze()

        // Compute combined witness W(X) = sum_i gamma^i * W_i(X)
        var maxWitnessDeg = 0
        var witnessPolys = [[Fr]]()
        witnessPolys.reserveCapacity(numPolys)

        for i in 0..<numPolys {
            let poly = polynomials[i]
            let pts = pointSets[i]
            let evals = allEvals[i]

            let vanishing = buildVanishingPoly(roots: pts)
            let interp = lagrangeInterpolation(points: pts, values: evals)

            let maxLen = max(poly.count, interp.count)
            var numerator = [Fr](repeating: Fr.zero, count: maxLen)
            for j in 0..<poly.count {
                numerator[j] = poly[j]
            }
            for j in 0..<interp.count {
                numerator[j] = frSub(numerator[j], interp[j])
            }

            let wi = polyExactDivide(numerator, by: vanishing)
            witnessPolys.append(wi)
            if wi.count > maxWitnessDeg { maxWitnessDeg = wi.count }
        }

        // Combine: sum_i gamma^i * W_i(X)
        var combined = [Fr](repeating: Fr.zero, count: maxWitnessDeg)
        var gammaPow = Fr.one
        for i in 0..<numPolys {
            let wi = witnessPolys[i]
            wi.withUnsafeBytes { pBuf in
                combined.withUnsafeMutableBytes { cBuf in
                    withUnsafeBytes(of: gammaPow) { gBuf in
                        bn254_fr_batch_mac_neon(
                            cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            gBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(wi.count))
                    }
                }
            }
            if i < numPolys - 1 {
                gammaPow = frMul(gammaPow, gamma)
            }
        }

        // Single GPU MSM for combined witness
        let witnessCommitment: PointProjective
        if combined.isEmpty || combined.allSatisfy({ isZeroFr($0) }) {
            witnessCommitment = pointIdentity()
        } else {
            witnessCommitment = try gpuKZG.commit(combined)
        }

        return GPUBatchMultiPointProof(
            commitments: commitments,
            evaluations: allEvals,
            points: pointSets,
            witness: witnessCommitment,
            gamma: gamma
        )
    }

    // MARK: - Verification (SRS-secret testing mode)

    /// Verify a single multi-point opening proof using the SRS secret.
    ///
    /// Check: [f(s)] - [I(s)] == [Z_S(s)] * [W(s)]
    /// Equivalently: C - I(s)*G == Z_S(s) * W
    public func verifyMultiPoint(
        proof: GPUMultiPointProof,
        points: [Fr],
        srsSecret: Fr
    ) -> Bool {
        guard proof.evaluations.count == points.count else { return false }

        let g1 = pointFromAffine(gpuKZG.srs[0])

        // Evaluate vanishing polynomial at secret
        var zsAtS = Fr.one
        for z in points {
            zsAtS = frMul(zsAtS, frSub(srsSecret, z))
        }

        // Evaluate interpolation polynomial at secret
        let interp = lagrangeInterpolation(points: points, values: proof.evaluations)
        let iAtS = hornerEval(interp, at: srsSecret)

        // LHS = C - I(s)*G
        let iG = cPointScalarMul(g1, iAtS)
        let lhs = pointAdd(proof.commitment, pointNeg(iG))

        // RHS = Z_S(s) * W
        let rhs = cPointScalarMul(proof.witness, zsAtS)

        return gpuPointsEqual(lhs, rhs)
    }

    /// Verify a batch multi-point opening proof using the SRS secret.
    public func verifyBatchMultiPoint(
        proof: GPUBatchMultiPointProof,
        transcript: Transcript,
        srsSecret: Fr
    ) -> Bool {
        let numPolys = proof.commitments.count
        guard numPolys == proof.evaluations.count,
              numPolys == proof.points.count else { return false }

        // Reconstruct transcript state
        for c in proof.commitments {
            absorbPoint(c, into: transcript)
        }
        for i in 0..<numPolys {
            for eval in proof.evaluations[i] {
                transcript.absorb(eval)
            }
        }
        let gamma = transcript.squeeze()

        let g1 = pointFromAffine(gpuKZG.srs[0])

        // Verify: sum_i gamma^i * W_i(s) == W(s)
        // W_i(s) = (f_i(s) - I_i(s)) / Z_{S_i}(s)
        var expectedW = pointIdentity()
        var gammaPow = Fr.one

        for i in 0..<numPolys {
            let pts = proof.points[i]
            let evals = proof.evaluations[i]

            // Z_S(s) for this polynomial's point set
            var zsAtS = Fr.one
            for z in pts {
                zsAtS = frMul(zsAtS, frSub(srsSecret, z))
            }
            let zsInv = frInverse(zsAtS)

            // I(s) for this polynomial's interpolation
            let interp = lagrangeInterpolation(points: pts, values: evals)
            let iAtS = hornerEval(interp, at: srsSecret)

            // W_i(s) * G = (C_i - I_i(s)*G) / Z_{S_i}(s)
            let iG = cPointScalarMul(g1, iAtS)
            let numerator = pointAdd(proof.commitments[i], pointNeg(iG))
            let wiPoint = cPointScalarMul(numerator, zsInv)

            expectedW = pointAdd(expectedW, cPointScalarMul(wiPoint, gammaPow))
            if i < numPolys - 1 {
                gammaPow = frMul(gammaPow, gamma)
            }
        }

        return gpuPointsEqual(proof.witness, expectedW)
    }

    // MARK: - Polynomial helpers

    /// Horner evaluation of polynomial at a point.
    private func hornerEval(_ coeffs: [Fr], at z: Fr) -> Fr {
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

    /// Build vanishing polynomial Z_S(X) = prod_i (X - z_i).
    /// Result in ascending coefficient order.
    private func buildVanishingPoly(roots: [Fr]) -> [Fr] {
        // Start with Z(X) = 1
        var result: [Fr] = [Fr.one]

        for root in roots {
            // Multiply by (X - root): shift up and subtract root * current
            var newResult = [Fr](repeating: Fr.zero, count: result.count + 1)
            // X * result
            for i in 0..<result.count {
                newResult[i + 1] = frAdd(newResult[i + 1], result[i])
            }
            // -root * result
            for i in 0..<result.count {
                newResult[i] = frSub(newResult[i], frMul(root, result[i]))
            }
            result = newResult
        }
        return result
    }

    /// Lagrange interpolation: given points (x_i, y_i), return polynomial coefficients.
    private func lagrangeInterpolation(points: [Fr], values: [Fr]) -> [Fr] {
        let k = points.count
        guard k > 0 else { return [] }
        if k == 1 { return [values[0]] }

        var result = [Fr](repeating: Fr.zero, count: k)

        for i in 0..<k {
            // Compute Lagrange basis polynomial L_i(X)
            // L_i(X) = prod_{j!=i} (X - x_j) / (x_i - x_j)

            // Compute denominator: prod_{j!=i} (x_i - x_j)
            var denom = Fr.one
            for j in 0..<k {
                if j != i {
                    denom = frMul(denom, frSub(points[i], points[j]))
                }
            }
            let denomInv = frInverse(denom)
            let scalar = frMul(values[i], denomInv)

            // Build numerator polynomial prod_{j!=i} (X - x_j)
            var basis: [Fr] = [Fr.one]
            for j in 0..<k {
                if j != i {
                    var newBasis = [Fr](repeating: Fr.zero, count: basis.count + 1)
                    for m in 0..<basis.count {
                        newBasis[m + 1] = frAdd(newBasis[m + 1], basis[m])
                        newBasis[m] = frSub(newBasis[m], frMul(points[j], basis[m]))
                    }
                    basis = newBasis
                }
            }

            // Accumulate scalar * basis into result
            for m in 0..<basis.count {
                if m < result.count {
                    result[m] = frAdd(result[m], frMul(scalar, basis[m]))
                }
            }
        }
        return result
    }

    /// Exact polynomial division: numerator / divisor (assumes exact division).
    /// Uses long division in coefficient form.
    private func polyExactDivide(_ numerator: [Fr], by divisor: [Fr]) -> [Fr] {
        let nDeg = polyDegree(numerator)
        let dDeg = polyDegree(divisor)
        if nDeg < dDeg { return [] }
        if dDeg < 0 { return [] }  // divisor is zero

        var rem = Array(numerator)
        let qLen = nDeg - dDeg + 1
        var quotient = [Fr](repeating: Fr.zero, count: qLen)
        let leadInv = frInverse(divisor[dDeg])

        for i in stride(from: nDeg, through: dDeg, by: -1) {
            if isZeroFr(rem[i]) { continue }
            let coeff = frMul(rem[i], leadInv)
            let qIdx = i - dDeg
            quotient[qIdx] = coeff
            for j in 0...dDeg {
                rem[qIdx + j] = frSub(rem[qIdx + j], frMul(coeff, divisor[j]))
            }
        }
        return quotient
    }

    /// Degree of polynomial (index of highest non-zero coefficient, or -1 if zero).
    private func polyDegree(_ p: [Fr]) -> Int {
        for i in stride(from: p.count - 1, through: 0, by: -1) {
            if !isZeroFr(p[i]) { return i }
        }
        return -1
    }

    private func isZeroFr(_ a: Fr) -> Bool {
        return frToInt(a) == frToInt(Fr.zero)
    }

    // MARK: - Point helpers

    private func absorbPoint(_ p: PointProjective, into transcript: Transcript) {
        if pointIsIdentity(p) {
            transcript.absorb(Fr.zero)
            transcript.absorb(Fr.zero)
        } else {
            let aff = batchToAffine([p])
            let xLimbs = fpToInt(aff[0].x)
            let yLimbs = fpToInt(aff[0].y)
            transcript.absorb(Fr.from64(xLimbs))
            transcript.absorb(Fr.from64(yLimbs))
        }
    }

    private func gpuPointsEqual(_ a: PointProjective, _ b: PointProjective) -> Bool {
        if pointIsIdentity(a) && pointIsIdentity(b) { return true }
        if pointIsIdentity(a) || pointIsIdentity(b) { return false }
        let aAff = batchToAffine([a])
        let bAff = batchToAffine([b])
        return fpToInt(aAff[0].x) == fpToInt(bAff[0].x) &&
               fpToInt(aAff[0].y) == fpToInt(bAff[0].y)
    }
}
