// MultilinearCommitment — Polynomial commitment schemes for multilinear polynomials
//
// Provides a unified protocol for multilinear PCS and concrete implementations:
//
// 1. HyraxCommitment — Pedersen-based multilinear commitment (row-wise MSM).
//    Splits 2^n evaluations into a sqrt(2^n) x sqrt(2^n) matrix and commits to
//    each row independently via Pedersen (MSM). Verification uses inner product argument.
//    Reference: Wahby et al., "Doubly-efficient zkSNARKs without trusted setup" (S&P 2018)
//
// 2. GeminiAdapter — Reduces multilinear evaluation claims to univariate claims,
//    then delegates to an existing univariate PCS (KZG or IPA).
//    Reference: Bünz, Maller, Mishra, Tyagi, Vesely — "Gemini" (EUROCRYPT 2023)

import Foundation
import NeonFieldOps

// MARK: - Protocol

/// A polynomial commitment scheme for multilinear polynomials.
///
/// Generic over:
/// - Commitment: the commitment type (e.g., curve point, Merkle root)
/// - OpeningProof: the proof type for opening at a point
public protocol MultilinearPCS {
    associatedtype Commitment
    associatedtype OpeningProof

    /// Commit to a multilinear polynomial.
    func commit(_ poly: MultilinearPoly) throws -> Commitment

    /// Open the polynomial at a point: produce a proof that poly(point) = value.
    func open(_ poly: MultilinearPoly, at point: [Fr], value: Fr?) throws -> (value: Fr, proof: OpeningProof)

    /// Verify that the commitment opens to the claimed value at the given point.
    func verify(commitment: Commitment, point: [Fr], value: Fr, proof: OpeningProof) throws -> Bool
}

// MARK: - Hyrax Commitment

/// Hyrax commitment for multilinear polynomials.
/// Reshapes 2^n evaluations into a 2^(n/2) x 2^(n/2) matrix and commits
/// to each row via Pedersen (MSM over generator points).
///
/// Commit:  C_i = MSM(G, row_i)  for i = 0,...,2^(n/2)-1
///
/// Open at point r = (r_L, r_R) where r_L has n/2 vars (row selector) and
/// r_R has n/2 vars (column selector):
///   1. Compute eq(r_L, i) for all row indices i
///   2. Compute v_R = sum_i eq(r_L, i) * row_i  (a vector of length 2^(n/2))
///   3. value = <v_R, eq(r_R, .)>
///   4. Proof = v_R (the verifier checks consistency with row commitments)
///
/// Verification:
///   1. Compute eq(r_L, i) for each row i
///   2. Check MSM(C_i, eq(r_L, i)) == MSM(G, v_R)
///   3. Check <v_R, eq(r_R, .)> == claimed value

public struct HyraxCommitment {
    /// Per-row Pedersen commitments
    public let rowCommitments: [PointProjective]
}

public struct HyraxOpeningProof {
    /// The inner vector v_R = sum_i eq(r_L, i) * row_i
    public let innerVector: [Fr]
    /// The claimed evaluation value
    public let value: Fr
}

public class HyraxPCS: MultilinearPCS {
    public typealias Commitment = HyraxCommitment
    public typealias OpeningProof = HyraxOpeningProof

    /// Generator points for Pedersen commitment (one per column)
    public let generators: [PointAffine]
    /// MSM engine for GPU-accelerated commitment
    public let msmEngine: MetalMSM

    /// Number of columns = 2^colVars
    public let colVars: Int
    public let numCols: Int

    /// Initialize with generator points.
    /// The number of generators determines the column dimension.
    /// For an n-variable MLP, use 2^(ceil(n/2)) generators.
    public init(generators: [PointAffine], msmEngine: MetalMSM? = nil) throws {
        precondition(!generators.isEmpty && (generators.count & (generators.count - 1)) == 0,
                     "Generator count must be a power of 2")
        self.generators = generators
        self.msmEngine = try msmEngine ?? MetalMSM()

        var cv = 0
        var s = generators.count
        while s > 1 { s >>= 1; cv += 1 }
        self.colVars = cv
        self.numCols = generators.count
    }

    /// Convenience: generate deterministic test generators from a base point.
    public static func generateTestGenerators(count: Int, base: PointAffine) -> [PointAffine] {
        var points = [PointProjective]()
        points.reserveCapacity(count)
        var current = pointFromAffine(base)
        for _ in 0..<count {
            points.append(current)
            // Double to get next generator (deterministic, NOT secure for production)
            current = pointDouble(current)
        }
        return batchToAffine(points)
    }

    // MARK: - Fr to MSM scalar conversion

    /// Convert an array of Fr elements to [[UInt32]] limbs suitable for MSM.
    private func frToMSMScalars(_ scalars: [Fr]) -> [[UInt32]] {
        scalars.map { fr in
            let limbs = fr.v
            return [limbs.0, limbs.1, limbs.2, limbs.3, limbs.4, limbs.5, limbs.6, limbs.7]
        }
    }

    // MARK: - Commit

    /// Commit to a multilinear polynomial by reshaping into a matrix and
    /// committing to each row via Pedersen MSM.
    public func commit(_ poly: MultilinearPoly) throws -> HyraxCommitment {
        let n = poly.numVars
        let rowVars = n - colVars
        precondition(rowVars >= 0, "Polynomial has too few variables (\(n)) for \(colVars) column variables")

        let numRows = 1 << rowVars

        var rowCommitments = [PointProjective]()
        rowCommitments.reserveCapacity(numRows)

        // Commit each row: C_i = MSM(G, row_i)
        for i in 0..<numRows {
            let rowStart = i * numCols
            let row = Array(poly.evals[rowStart..<(rowStart + numCols)])
            let scalars = frToMSMScalars(row)
            let commitment = try msmEngine.msm(points: generators, scalars: scalars)
            rowCommitments.append(commitment)
        }

        return HyraxCommitment(rowCommitments: rowCommitments)
    }

    // MARK: - Open

    /// Open at a multilinear evaluation point.
    /// Point is split into (r_L, r_R) where r_L selects the row and r_R selects the column.
    public func open(_ poly: MultilinearPoly, at point: [Fr], value: Fr? = nil) throws -> (value: Fr, proof: HyraxOpeningProof) {
        let n = poly.numVars
        precondition(point.count == n, "Point dimension mismatch")
        let rowVars = n - colVars

        let rL = Array(point[0..<rowVars])  // row selector
        let rR = Array(point[rowVars..<n])  // column selector

        // Compute eq(r_L, i) for all row indices
        let eqL = MultilinearPoly.eqPolyC(point: rL)

        // Compute inner vector: v_R = sum_i eq(r_L, i) * row_i
        var innerVector = [Fr](repeating: Fr.zero, count: numCols)
        let numRows = 1 << rowVars
        for i in 0..<numRows {
            if eqL.evals[i].isZero { continue }
            let rowStart = i * numCols
            withUnsafeBytes(of: eqL.evals[i]) { sPtr in
                poly.evals.withUnsafeBytes { pBuf in
                    innerVector.withUnsafeMutableBytes { rBuf in
                        bn254_fr_batch_mac_neon(
                            rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self) + rowStart * 4,
                            sPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(numCols))
                    }
                }
            }
        }

        // Compute evaluation: value = <v_R, eq(r_R, .)>
        let eqR = MultilinearPoly.eqPolyC(point: rR)
        var resultLimbs = [UInt64](repeating: 0, count: 4)
        innerVector.withUnsafeBytes { aBuf in
            eqR.evals.withUnsafeBytes { bBuf in
                bn254_fr_inner_product(
                    aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(numCols), &resultLimbs)
            }
        }
        let computedValue = Fr.from64(resultLimbs)

        // Sanity check if value provided
        if let v = value {
            precondition(frEq(v, computedValue), "Provided value does not match computed evaluation")
        }

        let proof = HyraxOpeningProof(innerVector: innerVector, value: computedValue)
        return (computedValue, proof)
    }

    // MARK: - Verify

    /// Verify a Hyrax opening proof.
    /// Checks:
    /// 1. MSM over row commitments with eq(r_L) weights equals MSM(G, v_R)
    /// 2. Inner product <v_R, eq(r_R)> equals claimed value
    public func verify(commitment: HyraxCommitment, point: [Fr], value: Fr, proof: HyraxOpeningProof) throws -> Bool {
        let n = point.count
        let rowVars = n - colVars

        let rL = Array(point[0..<rowVars])
        let rR = Array(point[rowVars..<n])

        // Check 1: value = <v_R, eq(r_R)>
        let eqR = MultilinearPoly.eqPolyC(point: rR)
        var computedValue = Fr.zero
        for j in 0..<numCols {
            computedValue = frAdd(computedValue, frMul(proof.innerVector[j], eqR.evals[j]))
        }
        guard frEq(computedValue, value) else {
            return false
        }

        // Check 2: sum_i eq(r_L, i) * C_i == MSM(G, v_R)
        let eqL = MultilinearPoly.eqPolyC(point: rL)
        let numRows = commitment.rowCommitments.count
        precondition(numRows == (1 << rowVars), "Row commitment count mismatch")

        // LHS: linear combination of row commitments weighted by eq(r_L)
        let rowAffine = batchToAffine(commitment.rowCommitments)
        let lhsScalars = frToMSMScalars(eqL.evals)
        let lhs = try msmEngine.msm(points: rowAffine, scalars: lhsScalars)

        // RHS: commitment to the inner vector
        let rhsScalars = frToMSMScalars(proof.innerVector)
        let rhs = try msmEngine.msm(points: generators, scalars: rhsScalars)

        // Compare projective points directly (no affine conversion needed)
        return pointEqual(lhs, rhs)
    }
}

// MARK: - Gemini-style adapter (KZG -> Multilinear)

/// Adapter that converts a univariate KZG commitment into a multilinear PCS
/// using the Zeromorph/Gemini reduction.
///
/// The idea: embed 2^n MLE evaluations as coefficients of a univariate polynomial,
/// then use the Zeromorph identity to reduce the multilinear evaluation claim to
/// a univariate claim at a single point.
///
/// This wraps the existing ZeromorphEngine for a clean MultilinearPCS interface.
public struct GeminiZeromorphProof {
    public let zeromorphProof: ZeromorphProof
}

public class GeminiAdapter: MultilinearPCS {
    public typealias Commitment = PointProjective
    public typealias OpeningProof = GeminiZeromorphProof

    public let zeromorph: ZeromorphEngine
    /// SRS secret for testing verification (in production, use pairing check)
    public var srsSecret: Fr?

    public init(zeromorph: ZeromorphEngine, srsSecret: Fr? = nil) {
        self.zeromorph = zeromorph
        self.srsSecret = srsSecret
    }

    /// Convenience: create from an existing KZG engine.
    public convenience init(kzg: KZGEngine, srsSecret: Fr? = nil) {
        self.init(zeromorph: ZeromorphEngine(kzg: kzg), srsSecret: srsSecret)
    }

    /// Commit to a multilinear polynomial (delegates to Zeromorph/KZG).
    public func commit(_ poly: MultilinearPoly) throws -> PointProjective {
        try zeromorph.commit(evaluations: poly.evals)
    }

    /// Open at a point using the Zeromorph reduction.
    public func open(_ poly: MultilinearPoly, at point: [Fr], value: Fr? = nil) throws -> (value: Fr, proof: GeminiZeromorphProof) {
        let proof = try zeromorph.open(evaluations: poly.evals, point: point, value: value)
        return (proof.claimedValue, GeminiZeromorphProof(zeromorphProof: proof))
    }

    /// Verify a Gemini/Zeromorph opening proof.
    /// Note: requires srsSecret for test verification (production would use pairings).
    public func verify(commitment: PointProjective, point: [Fr], value: Fr, proof: GeminiZeromorphProof) throws -> Bool {
        guard let secret = srsSecret else {
            // Without srsSecret, can only check claimed value matches
            return frEq(proof.zeromorphProof.claimedValue, value)
        }
        return zeromorph.verify(
            commitment: commitment,
            point: point,
            value: value,
            proof: proof.zeromorphProof,
            srsSecret: secret
        )
    }
}

// MARK: - Basefold adapter (transparent multilinear PCS)

/// Adapter wrapping BasefoldEngine as a MultilinearPCS.
/// Basefold is a transparent (no trusted setup) PCS for multilinear polynomials
/// using Poseidon2 Merkle commitments and sumcheck-style folding.
public struct BasefoldMultilinearProof {
    public let basefoldProof: BasefoldProof
    public let commitment: BasefoldCommitment
}

public class BasefoldAdapter: MultilinearPCS {
    public typealias Commitment = BasefoldCommitment
    public typealias OpeningProof = BasefoldMultilinearProof

    public let basefold: BasefoldEngine

    public init() throws {
        self.basefold = try BasefoldEngine()
    }

    public init(basefold: BasefoldEngine) {
        self.basefold = basefold
    }

    /// Commit to a multilinear polynomial via Poseidon2 Merkle tree.
    public func commit(_ poly: MultilinearPoly) throws -> BasefoldCommitment {
        try basefold.commit(evaluations: poly.evals)
    }

    /// Open at a point using Basefold sumcheck-style folding.
    public func open(_ poly: MultilinearPoly, at point: [Fr], value: Fr? = nil) throws -> (value: Fr, proof: BasefoldMultilinearProof) {
        let commitment = try basefold.commit(evaluations: poly.evals)
        let proof = try basefold.open(commitment: commitment, point: point)
        let computedValue = poly.evaluateC(at: point)
        return (computedValue, BasefoldMultilinearProof(basefoldProof: proof, commitment: commitment))
    }

    /// Verify a Basefold opening proof.
    public func verify(commitment: BasefoldCommitment, point: [Fr], value: Fr, proof: BasefoldMultilinearProof) throws -> Bool {
        basefold.verify(root: commitment.root, point: point, claimedValue: value, proof: proof.basefoldProof)
    }
}
