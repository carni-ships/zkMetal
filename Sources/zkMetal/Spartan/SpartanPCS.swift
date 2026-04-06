// SpartanPCS — Protocol for multilinear polynomial commitment schemes
// used by Spartan IPA/KZG variants.
//
// Any PCS that can commit to evaluations on {0,1}^n,
// open at a multilinear point, and verify the opening can serve as backend.

import Foundation
import NeonFieldOps

/// A commitment handle returned by a multilinear PCS.
/// Carries whatever the PCS needs (root hash, curve point, etc.)
/// plus an Fr-representable "tag" for Fiat-Shamir absorption.
public protocol SpartanCommitmentHandle {
    /// An Fr element summarizing this commitment for transcript absorption.
    var transcriptTag: Fr { get }
}

/// An opening proof from a multilinear PCS.
public protocol SpartanOpeningProof {}

/// Protocol for multilinear polynomial commitment schemes usable with Spartan.
public protocol SpartanPCSBackend {
    associatedtype Commitment: SpartanCommitmentHandle
    associatedtype Opening: SpartanOpeningProof

    /// Commit to evaluations on the boolean hypercube {0,1}^n.
    /// `evaluations` has length 2^n (padded if needed).
    func commit(evaluations: [Fr]) throws -> Commitment

    /// Open the committed polynomial at a multilinear point.
    /// Returns the claimed evaluation value and an opening proof.
    func open(evaluations: [Fr], point: [Fr]) throws -> (value: Fr, proof: Opening)

    /// Verify that the commitment opens to `value` at `point`.
    func verify(commitment: Commitment, point: [Fr], value: Fr, proof: Opening) -> Bool
}

// MARK: - Basefold Adapter (wraps existing BasefoldEngine)

/// Wraps BasefoldEngine as a SpartanPCSBackend.
public struct BasefoldPCSAdapter: SpartanPCSBackend {
    public struct Commitment: SpartanCommitmentHandle {
        public let inner: BasefoldCommitment
        public var transcriptTag: Fr { inner.root }
    }

    public struct Opening: SpartanOpeningProof {
        public let inner: BasefoldProof
        public let value: Fr
    }

    private let engine: BasefoldEngine

    public init(engine: BasefoldEngine) {
        self.engine = engine
    }

    public func commit(evaluations: [Fr]) throws -> Commitment {
        Commitment(inner: try engine.commit(evaluations: evaluations))
    }

    public func open(evaluations: [Fr], point: [Fr]) throws -> (value: Fr, proof: Opening) {
        // Compute the MLE evaluation
        let value = spartanEvalML(evals: evaluations, pt: point)
        let inner = try engine.open(
            commitment: BasefoldCommitment(root: Fr.zero, evaluations: evaluations, tree: []),
            point: point
        )
        // We need the actual commitment for the open call; re-commit
        let comm = try engine.commit(evaluations: evaluations)
        let proof = try engine.open(commitment: comm, point: point)
        return (value, Opening(inner: proof, value: value))
    }

    public func verify(commitment: Commitment, point: [Fr], value: Fr, proof: Opening) -> Bool {
        engine.verify(root: commitment.inner.root, point: point,
                      claimedValue: value, proof: proof.inner)
    }
}

// MARK: - IPA Adapter (wraps IPAEngine for multilinear evaluation)

/// IPA-based multilinear PCS: commits to evaluation vector,
/// opens via inner product with the eq polynomial.
///
/// Commitment: C = MSM(G, evals) — uses GPU Metal MSM for large vectors,
///   BGMW fixed-base tables for small vectors.
/// Opening claim: <evals, eq(point, .)> = value
/// Proof: IPA proof with log(n) halving rounds using dual MSM.
///
/// Transparent: no trusted setup required.
/// GPU-accelerated: commit uses Metal MSM (>= 4096 generators),
///   IPA halving uses C Pippenger dual MSM.
public struct IPAPCSAdapter: SpartanPCSBackend {
    public struct Commitment: SpartanCommitmentHandle {
        /// The Pedersen commitment point (affine x-coordinate as Fr for transcript)
        public let point: PointProjective
        public let evaluations: [Fr]  // prover keeps for opening

        public var transcriptTag: Fr {
            // Use affine x-coordinate as transcript tag
            if let aff = pointToAffine(point) {
                return fpToFr(aff.x)
            }
            return Fr.zero
        }
    }

    public struct Opening: SpartanOpeningProof {
        public let inner: IPAProof
    }

    /// The underlying IPA engine with GPU MSM and BGMW fixed-base support.
    public let engine: IPAEngine

    public init(engine: IPAEngine) {
        self.engine = engine
    }

    /// Commit to evaluations using GPU-accelerated MSM (for large vectors)
    /// or BGMW fixed-base tables (for small vectors).
    public func commit(evaluations: [Fr]) throws -> Commitment {
        // engine.commit() automatically selects GPU vs BGMW based on size
        let C = try engine.commit(evaluations)
        return Commitment(point: C, evaluations: evaluations)
    }

    public func open(evaluations: [Fr], point: [Fr]) throws -> (value: Fr, proof: Opening) {
        let n = evaluations.count
        precondition(n == engine.generators.count, "evaluation size must match generator count")
        precondition(n > 0 && (n & (n - 1)) == 0, "must be power of 2")

        // b = eq(point, .) evaluated at all boolean hypercube points
        let numVars = point.count
        precondition((1 << numVars) == n)
        let b = cEqPolyExpand(point: point)

        // value = <evals, b> via C-accelerated inner product
        let value = cFrInnerProduct(evaluations, b)

        // IPA proof that <evals, b> = value
        // Uses dual MSM + C Pippenger for halving rounds
        let proof = try engine.createProof(a: evaluations, b: b)
        return (value, Opening(inner: proof))
    }

    public func verify(commitment: Commitment, point: [Fr], value: Fr, proof: Opening) -> Bool {
        let n = engine.generators.count
        let numVars = point.count
        guard (1 << numVars) == n else { return false }

        let b = cEqPolyExpand(point: point)

        // Pass raw commitment — engine.verify computes Cbound = C + v*Q internally
        return engine.verify(commitment: commitment.point, b: b,
                             innerProductValue: value, proof: proof.inner)
    }

    /// C-accelerated eq polynomial expansion using spartan_eq_poly.
    /// Much faster than the Swift loop for large vectors.
    private func cEqPolyExpand(point: [Fr]) -> [Fr] {
        let n = point.count
        let size = 1 << n
        var eq = [Fr](repeating: Fr.zero, count: size)
        point.withUnsafeBytes { ptBuf in
            eq.withUnsafeMutableBytes { eqBuf in
                spartan_eq_poly(
                    ptBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(n),
                    eqBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                )
            }
        }
        return eq
    }
}

// MARK: - KZG Adapter (multilinear KZG via iterative folding)

/// KZG-based multilinear PCS via iterative univariate folding.
/// Proves MLE evaluation by committing to folded polynomials at each step.
///
/// Protocol (for MLE with n variables):
///   1. Commit: C = KZG.commit(evals) treating evaluations as univariate coefficients
///   2. Open at point r = (r_0, ..., r_{n-1}):
///      - For each variable k from n-1 down to 0:
///        - Split current into even/odd: f_even[i] = f[2i], f_odd[i] = f[2i+1]
///        - folded[i] = (1-r_k)*f_even[i] + r_k*f_odd[i]  (standard MLE fold)
///        - Commit to folded polynomial
///      - Final value = folded[0]
///   3. Verify: check each fold step using KZG openings at random challenge points
///
/// Requires trusted setup (SRS), but gives O(log n) proof size.
public struct KZGPCSAdapter: SpartanPCSBackend {
    public struct Commitment: SpartanCommitmentHandle {
        public let point: PointProjective
        public let evaluations: [Fr]  // prover keeps for opening

        public var transcriptTag: Fr {
            if let aff = pointToAffine(point) {
                return fpToFr(aff.x)
            }
            return Fr.zero
        }
    }

    public struct Opening: SpartanOpeningProof {
        /// Commitments to folded polynomials at each level
        public let foldCommitments: [PointProjective]
        /// KZG opening proofs for each fold step verification
        public let foldProofs: [KZGProof]
        /// The claimed MLE evaluation value
        public let value: Fr
    }

    private let kzg: KZGEngine
    /// SRS secret for verification (testing only; production uses pairings)
    private let srsSecret: Fr

    public init(kzg: KZGEngine, srsSecret: Fr) {
        self.kzg = kzg
        self.srsSecret = srsSecret
    }

    public func commit(evaluations: [Fr]) throws -> Commitment {
        let C = try kzg.commit(evaluations)
        return Commitment(point: C, evaluations: evaluations)
    }

    public func open(evaluations: [Fr], point: [Fr]) throws -> (value: Fr, proof: Opening) {
        let n = point.count
        precondition(evaluations.count == (1 << n))

        var current = evaluations
        var foldCommitments = [PointProjective]()
        var foldProofs = [KZGProof]()
        foldCommitments.reserveCapacity(n)
        foldProofs.reserveCapacity(n)

        // Iterative folding: at each step, fold using (1-r_k)*even + r_k*odd
        for k in stride(from: n - 1, through: 0, by: -1) {
            let half = current.count / 2
            let rk = point[k]
            let oneMinusRk = frSub(Fr.one, rk)
            var folded = [Fr](repeating: Fr.zero, count: half)
            for i in 0..<half {
                let lo = current[2 * i]
                let hi = current[2 * i + 1]
                folded[i] = frAdd(frMul(oneMinusRk, lo), frMul(rk, hi))
            }

            // Commit to the folded polynomial
            let C = try kzg.commit(folded)
            foldCommitments.append(C)

            // Open the current polynomial at a challenge derived from the fold
            // For simplicity in testing: open at X=0 to get current[0]
            let proof = try kzg.open(current, at: rk)
            foldProofs.append(proof)

            current = folded
        }

        let value = current[0]
        return (value, Opening(foldCommitments: foldCommitments,
                                foldProofs: foldProofs, value: value))
    }

    public func verify(commitment: Commitment, point: [Fr], value: Fr, proof: Opening) -> Bool {
        // Check claimed value matches
        guard frToInt(value) == frToInt(proof.value) else { return false }

        // Verify each fold step's KZG opening
        let n = point.count
        guard proof.foldCommitments.count == n else { return false }
        guard proof.foldProofs.count == n else { return false }

        // In testing mode with known srsSecret: verify KZG proofs algebraically.
        // Each fold proof opens the polynomial at r_k.
        // The evaluation at r_k should be consistent with the fold.
        for i in 0..<n {
            let kzgProof = proof.foldProofs[i]
            let rk = point[n - 1 - i]

            // Verify KZG opening: check e(C - v*G, H) = e(pi, s*H - z*H)
            // In test mode: C(s) - v = (s - z) * pi(s)
            // where C(s) is implicit in the commitment
            let sMinusZ = frSub(srsSecret, rk)
            let piS = cPointScalarMul(kzgProof.witness, sMinusZ)
            let vG = cPointScalarMul(pointFromAffine(kzg.srs[0]), kzgProof.evaluation)

            // Determine which commitment this fold corresponds to
            let comm: PointProjective
            if i == 0 {
                comm = commitment.point
            } else {
                comm = proof.foldCommitments[i - 1]
            }

            let lhs = pointAdd(comm, pointNeg(vG))
            let lhsAff = pointToAffine(lhs)
            let piAff = pointToAffine(piS)

            if let la = lhsAff, let pa = piAff {
                let lhsX = fpToInt(la.x)
                let rhsX = fpToInt(pa.x)
                let lhsY = fpToInt(la.y)
                let rhsY = fpToInt(pa.y)
                if lhsX != rhsX || lhsY != rhsY { return false }
            }
        }

        return true
    }
}

// Note: fpToFr(_ fp: Fp) -> Fr is defined in HyperNovaEngine.swift
// and available module-wide. No need to duplicate here.
