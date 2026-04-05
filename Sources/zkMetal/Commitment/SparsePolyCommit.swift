// Sparse Polynomial Commitment Scheme
//
// Optimized commitment for polynomials with few non-zero coefficients,
// common in R1CS matrices and selector polynomials in Plonk.
//
// Key optimization: only perform MSM over non-zero terms, selecting
// the corresponding SRS points by index. For a polynomial of degree N
// with k non-zero coefficients, commitment costs O(k) instead of O(N).

import Foundation
import NeonFieldOps

// MARK: - Sparse Polynomial Representation

/// A sparse polynomial represented as (index, coefficient) pairs.
/// Coefficients are in Montgomery form (Fr).
/// Invariant: indices are sorted in ascending order, no duplicates, all coefficients non-zero.
public struct SparsePoly {
    /// Non-zero terms as (index, coefficient) pairs, sorted by index.
    public let terms: [(index: Int, coeff: Fr)]
    /// The degree bound (one past the highest possible index).
    /// The polynomial is treated as having degree < degreeBound.
    public let degreeBound: Int

    /// Create a sparse polynomial from (index, coeff) pairs.
    /// Filters out zero coefficients, sorts by index, and merges duplicates.
    public init(terms: [(index: Int, coeff: Fr)], degreeBound: Int) {
        var merged = [Int: Fr]()
        for (idx, c) in terms {
            if frToInt(c) == frToInt(Fr.zero) { continue }
            if let existing = merged[idx] {
                let sum = frAdd(existing, c)
                if frToInt(sum) == frToInt(Fr.zero) {
                    merged.removeValue(forKey: idx)
                } else {
                    merged[idx] = sum
                }
            } else {
                merged[idx] = c
            }
        }
        self.terms = merged.sorted { $0.key < $1.key }.map { (index: $0.key, coeff: $0.value) }
        self.degreeBound = degreeBound
    }

    /// Create from a dense coefficient array, extracting non-zero entries.
    public init(dense coeffs: [Fr]) {
        var pairs = [(index: Int, coeff: Fr)]()
        pairs.reserveCapacity(coeffs.count / 4) // heuristic for sparse
        for (i, c) in coeffs.enumerated() {
            if frToInt(c) != frToInt(Fr.zero) {
                pairs.append((index: i, coeff: c))
            }
        }
        self.terms = pairs
        self.degreeBound = coeffs.count
    }

    /// Number of non-zero coefficients.
    public var nnz: Int { terms.count }

    /// Density: fraction of non-zero coefficients.
    public var density: Double {
        guard degreeBound > 0 else { return 0 }
        return Double(nnz) / Double(degreeBound)
    }

    /// Convert to dense coefficient array.
    public func toDense() -> [Fr] {
        var coeffs = [Fr](repeating: Fr.zero, count: degreeBound)
        for (idx, c) in terms {
            coeffs[idx] = c
        }
        return coeffs
    }

    /// Evaluate the polynomial at a point z using Horner-like sparse evaluation.
    /// Computes sum_i coeff_i * z^index_i.
    public func evaluate(at z: Fr) -> Fr {
        if terms.isEmpty { return Fr.zero }

        // Compute powers of z incrementally
        var result = Fr.zero
        var zPow = Fr.one  // z^0
        var prevIdx = 0

        for (idx, c) in terms {
            // Advance zPow from z^prevIdx to z^idx
            let gap = idx - prevIdx
            for _ in 0..<gap {
                zPow = frMul(zPow, z)
            }
            prevIdx = idx
            result = frAdd(result, frMul(c, zPow))
        }
        return result
    }

    /// Add two sparse polynomials.
    public func add(_ other: SparsePoly) -> SparsePoly {
        var combined = [(index: Int, coeff: Fr)]()
        combined.reserveCapacity(self.nnz + other.nnz)
        var i = 0, j = 0
        while i < self.terms.count && j < other.terms.count {
            let a = self.terms[i], b = other.terms[j]
            if a.index < b.index {
                combined.append(a); i += 1
            } else if a.index > b.index {
                combined.append(b); j += 1
            } else {
                let s = frAdd(a.coeff, b.coeff)
                if frToInt(s) != frToInt(Fr.zero) {
                    combined.append((index: a.index, coeff: s))
                }
                i += 1; j += 1
            }
        }
        while i < self.terms.count { combined.append(self.terms[i]); i += 1 }
        while j < other.terms.count { combined.append(other.terms[j]); j += 1 }
        let bound = max(self.degreeBound, other.degreeBound)
        // Already sorted and merged, bypass init processing
        var result = SparsePoly(terms: [], degreeBound: bound)
        result = SparsePoly(_sortedTerms: combined, degreeBound: bound)
        return result
    }

    /// Scale all coefficients by a scalar.
    public func scale(by s: Fr) -> SparsePoly {
        if frToInt(s) == frToInt(Fr.zero) {
            return SparsePoly(terms: [], degreeBound: degreeBound)
        }
        let scaled = terms.map { (index: $0.index, coeff: frMul($0.coeff, s)) }
        return SparsePoly(_sortedTerms: scaled, degreeBound: degreeBound)
    }

    /// Internal init that trusts terms are already sorted, deduped, and non-zero.
    internal init(_sortedTerms: [(index: Int, coeff: Fr)], degreeBound: Int) {
        self.terms = _sortedTerms
        self.degreeBound = degreeBound
    }
}

// MARK: - Sparse KZG Commitment

/// KZG commitment engine optimized for sparse polynomials.
/// Commit cost is O(k) MSM where k = number of non-zero coefficients,
/// instead of O(N) for the dense polynomial degree.
public class SparseKZG {
    public let srs: [PointAffine]

    public init(srs: [PointAffine]) {
        self.srs = srs
    }

    /// Commit to a sparse polynomial: C = sum_i coeff_i * SRS[index_i]
    /// Only touches k SRS points where k = poly.nnz.
    public func commit(_ poly: SparsePoly) -> PointProjective {
        let k = poly.nnz
        if k == 0 { return pointIdentity() }

        // Gather the SRS points and scalars for non-zero terms
        var points = [PointAffine]()
        points.reserveCapacity(k)
        var flatScalars = [UInt32]()
        flatScalars.reserveCapacity(k * 8)

        for (idx, coeff) in poly.terms {
            guard idx < srs.count else { continue }
            points.append(srs[idx])
            let limbs = frToLimbs(coeff)
            flatScalars.append(contentsOf: limbs)
        }

        if points.isEmpty { return pointIdentity() }
        return cPippengerMSMFlat(points: points, flatScalars: flatScalars)
    }

    /// Commit to a dense polynomial (convenience, delegates to standard MSM).
    public func commitDense(_ coeffs: [Fr]) -> PointProjective {
        let n = min(coeffs.count, srs.count)
        if n == 0 { return pointIdentity() }
        let points = Array(srs.prefix(n))
        var flatScalars = [UInt32](repeating: 0, count: n * 8)
        coeffs.prefix(n).withUnsafeBytes { src in
            flatScalars.withUnsafeMutableBufferPointer { dst in
                bn254_fr_batch_to_limbs(
                    src.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    dst.baseAddress!,
                    Int32(n)
                )
            }
        }
        return cPippengerMSMFlat(points: points, flatScalars: flatScalars)
    }
}

// MARK: - Sparse IPA Commitment

/// Inner product argument adapted for sparse polynomials.
/// The commitment is a Pedersen commitment using only the generators
/// at non-zero indices, reducing MSM cost from O(N) to O(k).
public class SparseIPA {
    /// Full generator set G_0, ..., G_{N-1}.
    public let generators: [PointAffine]
    /// Binding generator Q for inner product.
    public let Q: PointAffine

    public init(generators: [PointAffine], Q: PointAffine) {
        self.generators = generators
        self.Q = Q
    }

    /// Commit to a sparse polynomial: C = sum_i coeff_i * G[index_i]
    /// Cost is O(k) MSM where k = poly.nnz.
    public func commit(_ poly: SparsePoly) -> PointProjective {
        let k = poly.nnz
        if k == 0 { return pointIdentity() }

        var points = [PointAffine]()
        points.reserveCapacity(k)
        var flatScalars = [UInt32]()
        flatScalars.reserveCapacity(k * 8)

        for (idx, coeff) in poly.terms {
            guard idx < generators.count else { continue }
            points.append(generators[idx])
            flatScalars.append(contentsOf: frToLimbs(coeff))
        }

        if points.isEmpty { return pointIdentity() }
        return cPippengerMSMFlat(points: points, flatScalars: flatScalars)
    }

    /// Commit with binding: C = sum_i coeff_i * G[index_i] + ip * Q
    /// where ip is the inner product of the sparse vector with a public vector.
    public func commitWithBinding(_ poly: SparsePoly, innerProduct ip: Fr) -> PointProjective {
        let base = commit(poly)
        let qProj = pointFromAffine(Q)
        let ipQ = cPointScalarMul(qProj, ip)
        return pointAdd(base, ipQ)
    }
}

// MARK: - Batch Sparse Commitment

/// Batch commitment for multiple sparse polynomials that may share structure
/// (e.g., R1CS matrices with similar sparsity patterns, or selector polynomials).
public class BatchSparseCommit {
    public let srs: [PointAffine]

    public init(srs: [PointAffine]) {
        self.srs = srs
    }

    /// Commit to multiple sparse polynomials independently.
    /// Returns one commitment per polynomial.
    public func commitAll(_ polys: [SparsePoly]) -> [PointProjective] {
        let engine = SparseKZG(srs: srs)
        return polys.map { engine.commit($0) }
    }

    /// Commit to multiple sparse polynomials using shared-index optimization.
    /// When polynomials share non-zero indices, the SRS point is loaded once
    /// and accumulated across all polynomials. This reduces memory traffic.
    ///
    /// Returns one commitment per polynomial.
    public func commitShared(_ polys: [SparsePoly]) -> [PointProjective] {
        if polys.isEmpty { return [] }
        if polys.count == 1 {
            return [SparseKZG(srs: srs).commit(polys[0])]
        }

        // Build union of all non-zero indices
        var indexSet = Set<Int>()
        for poly in polys {
            for (idx, _) in poly.terms {
                if idx < srs.count { indexSet.insert(idx) }
            }
        }
        let sortedIndices = indexSet.sorted()
        let indexMap = Dictionary(uniqueKeysWithValues: sortedIndices.enumerated().map { ($1, $0) })

        // Gather SRS points at union indices
        let sharedPoints = sortedIndices.map { srs[$0] }
        let k = sharedPoints.count

        // For each polynomial, build scalar vector over the shared indices
        var results = [PointProjective]()
        results.reserveCapacity(polys.count)

        for poly in polys {
            var flatScalars = [UInt32](repeating: 0, count: k * 8)
            for (idx, coeff) in poly.terms {
                guard let pos = indexMap[idx] else { continue }
                let limbs = frToLimbs(coeff)
                for j in 0..<8 {
                    flatScalars[pos * 8 + j] = limbs[j]
                }
            }
            results.append(cPippengerMSMFlat(points: sharedPoints, flatScalars: flatScalars))
        }

        return results
    }

    /// Random linear combination commitment: C = sum_i gamma^i * C_i
    /// Useful for batch verification with a single pairing check.
    public func commitLinearCombination(_ polys: [SparsePoly], gamma: Fr) -> PointProjective {
        if polys.isEmpty { return pointIdentity() }

        // Combine into a single sparse polynomial: h(x) = sum_i gamma^i * p_i(x)
        var combined = SparsePoly(terms: [], degreeBound: 0)
        var gammaPow = Fr.one
        for poly in polys {
            let scaled = poly.scale(by: gammaPow)
            combined = combined.add(scaled)
            gammaPow = frMul(gammaPow, gamma)
        }

        return SparseKZG(srs: srs).commit(combined)
    }
}

// MARK: - Sparse Opening Proof

/// Opening proof for a sparse polynomial at a single evaluation point.
public struct SparseOpeningProof {
    /// The evaluation p(z).
    public let evaluation: Fr
    /// The witness point [q(s)] where q(x) = (p(x) - p(z)) / (x - z).
    public let witness: PointProjective
}

/// Sparse opening engine: prove and verify evaluations of sparse polynomials.
/// Exploits sparsity in both the evaluation and quotient computation.
public class SparseOpening {
    public let srs: [PointAffine]

    public init(srs: [PointAffine]) {
        self.srs = srs
    }

    /// Open a sparse polynomial at point z.
    ///
    /// Algorithm:
    ///   1. Evaluate p(z) using sparse Horner (O(k) muls)
    ///   2. Compute quotient q(x) = (p(x) - p(z)) / (x - z) via dense synthetic division
    ///   3. Commit to quotient using dense MSM (quotient is generally dense)
    ///
    /// For very sparse polynomials with large degree, converting to dense for the
    /// quotient is unavoidable since (p(x) - v) / (x - z) fills in the gaps.
    public func open(_ poly: SparsePoly, at z: Fr) -> SparseOpeningProof {
        // 1. Sparse evaluation
        let pz = poly.evaluate(at: z)

        // 2. Compute quotient: (p(x) - p(z)) / (x - z)
        //    Convert to dense, subtract evaluation, then synthetic divide
        var dense = poly.toDense()
        if dense.count < 2 {
            return SparseOpeningProof(evaluation: pz, witness: pointIdentity())
        }
        dense[0] = frSub(dense[0], pz)

        let quotient = syntheticDivide(dense, by: z)

        // 3. Commit to quotient
        let n = quotient.count
        guard n > 0, n <= srs.count else {
            return SparseOpeningProof(evaluation: pz, witness: pointIdentity())
        }

        let pts = Array(srs.prefix(n))
        var flatScalars = [UInt32](repeating: 0, count: n * 8)
        quotient.withUnsafeBytes { src in
            flatScalars.withUnsafeMutableBufferPointer { dst in
                bn254_fr_batch_to_limbs(
                    src.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    dst.baseAddress!,
                    Int32(n)
                )
            }
        }
        let witness = cPippengerMSMFlat(points: pts, flatScalars: flatScalars)

        return SparseOpeningProof(evaluation: pz, witness: witness)
    }

    /// Verify a sparse opening proof using the SRS secret (for testing).
    ///
    /// Checks: C == [p(z)] * G + [s - z] * witness
    /// where C is the commitment, s is the SRS secret.
    public func verify(
        commitment: PointProjective,
        point z: Fr,
        proof: SparseOpeningProof,
        srsSecret: Fr
    ) -> Bool {
        let g1 = pointFromAffine(srs[0])

        // [p(z)] * G
        let evalG = cPointScalarMul(g1, proof.evaluation)

        // [s - z] * witness
        let sMz = frSub(srsSecret, z)
        let szWitness = cPointScalarMul(proof.witness, sMz)

        // expected = evalG + szWitness
        let expected = pointAdd(evalG, szWitness)

        return pointEqual(commitment, expected)
    }

    // MARK: - Private Helpers

    /// Synthetic division of polynomial by (x - z).
    /// Input: coefficients of p(x) where p(z) has already been subtracted from constant term.
    /// Output: coefficients of q(x) = p(x) / (x - z).
    private func syntheticDivide(_ coeffs: [Fr], by z: Fr) -> [Fr] {
        let n = coeffs.count
        if n < 2 { return [] }

        var quotient = [Fr](repeating: Fr.zero, count: n - 1)
        coeffs.withUnsafeBytes { cBuf in
            withUnsafeBytes(of: z) { zBuf in
                quotient.withUnsafeMutableBytes { qBuf in
                    bn254_fr_synthetic_div(
                        cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        zBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n),
                        qBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                    )
                }
            }
        }
        return quotient
    }
}

// MARK: - Sparse Verification Utilities

/// Verification utilities that exploit sparsity for faster checks.
public struct SparseVerification {
    /// Verify that a sparse commitment matches a dense commitment for the same polynomial.
    /// Useful as a sanity check during testing.
    public static func commitmentConsistency(
        sparse: PointProjective,
        dense: PointProjective
    ) -> Bool {
        return pointEqual(sparse, dense)
    }

    /// Batch verify multiple sparse opening proofs with random linear combination.
    ///
    /// Instead of k individual checks, combines into a single MSM check:
    ///   sum_i gamma^i * (C_i - [y_i]*G - [s-z_i]*W_i) == 0
    ///
    /// Parameters:
    ///   - commitments: the polynomial commitments C_i
    ///   - points: evaluation points z_i
    ///   - proofs: opening proofs (evaluation, witness)
    ///   - gamma: random challenge for batching
    ///   - srs: the SRS (first element is G)
    ///   - srsSecret: the SRS toxic waste (for testing)
    /// Returns: true if all openings verify
    public static func batchVerify(
        commitments: [PointProjective],
        points: [Fr],
        proofs: [SparseOpeningProof],
        gamma: Fr,
        srs: [PointAffine],
        srsSecret: Fr
    ) -> Bool {
        let n = commitments.count
        guard n == points.count, n == proofs.count, n > 0 else { return false }

        let g1 = pointFromAffine(srs[0])
        var accumulated = pointIdentity()
        var gammaPow = Fr.one

        for i in 0..<n {
            // term_i = C_i - [y_i]*G - [s-z_i]*W_i
            let evalG = cPointScalarMul(g1, proofs[i].evaluation)
            let sMz = frSub(srsSecret, points[i])
            let szW = cPointScalarMul(proofs[i].witness, sMz)
            let expected = pointAdd(evalG, szW)

            // diff = C_i - expected
            // Negate expected: negate y coordinate
            let negExpected = pointNeg(expected)
            let diff = pointAdd(commitments[i], negExpected)

            // Accumulate gamma^i * diff
            let scaled = cPointScalarMul(diff, gammaPow)
            accumulated = pointAdd(accumulated, scaled)

            gammaPow = frMul(gammaPow, gamma)
        }

        return pointIsIdentity(accumulated)
    }
}

