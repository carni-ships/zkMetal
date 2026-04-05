// CommittedInstance — Unified committed instance type for HyperNova folding
//
// Wraps both "fresh" (CCCS) and "running/relaxed" (LCCCS) instances into a
// single type that the multi-fold API can operate on uniformly.
//
// A CommittedCCSInstance carries:
//   - Pedersen commitment to witness
//   - Public input vector
//   - Relaxation scalar u (1 for fresh, accumulates via folding)
//   - Random evaluation point r (log(m)-dimensional)
//   - Claimed MLE evaluations v_i = MLE(M_i * z)(r)
//   - Error terms (for multi-fold cross-term tracking)

import Foundation
import NeonFieldOps

// MARK: - Committed CCS Instance (Unified)

/// A committed (possibly relaxed) CCS instance. Unifies CCCS and LCCCS.
///
/// Fresh instances have u=1, isRelaxed=false.
/// After folding, u != 1, isRelaxed=true, and v/r are populated.
public struct CommittedCCSInstance {
    public let commitment: PointProjective   // Pedersen commitment to witness
    public let publicInput: [Fr]             // Public input x
    public let u: Fr                         // Relaxation scalar (1 for fresh)
    public let r: [Fr]                       // MLE evaluation point (empty for fresh CCCS)
    public let v: [Fr]                       // MLE evaluations v_i (empty for fresh CCCS)
    public let isRelaxed: Bool               // true if this is a folded/LCCCS instance

    // Cached affine coordinates of commitment (avoids repeated projective-to-affine)
    public let cachedAffineX: Fr?
    public let cachedAffineY: Fr?

    /// Create a fresh (non-relaxed) committed instance from a CCCS.
    public init(commitment: PointProjective, publicInput: [Fr],
                affineX: Fr? = nil, affineY: Fr? = nil) {
        self.commitment = commitment
        self.publicInput = publicInput
        self.u = Fr.one
        self.r = []
        self.v = []
        self.isRelaxed = false
        self.cachedAffineX = affineX
        self.cachedAffineY = affineY
    }

    /// Create a relaxed (folded) committed instance from an LCCCS.
    public init(commitment: PointProjective, publicInput: [Fr],
                u: Fr, r: [Fr], v: [Fr],
                affineX: Fr? = nil, affineY: Fr? = nil) {
        self.commitment = commitment
        self.publicInput = publicInput
        self.u = u
        self.r = r
        self.v = v
        self.isRelaxed = true
        self.cachedAffineX = affineX
        self.cachedAffineY = affineY
    }

    /// Convert from existing CCCS type.
    public init(from cccs: CCCS) {
        self.commitment = cccs.commitment
        self.publicInput = cccs.publicInput
        self.u = Fr.one
        self.r = []
        self.v = []
        self.isRelaxed = false
        self.cachedAffineX = cccs.cachedAffineX
        self.cachedAffineY = cccs.cachedAffineY
    }

    /// Convert from existing LCCCS type.
    public init(from lcccs: LCCCS) {
        self.commitment = lcccs.commitment
        self.publicInput = lcccs.publicInput
        self.u = lcccs.u
        self.r = lcccs.r
        self.v = lcccs.v
        self.isRelaxed = true
        self.cachedAffineX = lcccs.cachedAffineX
        self.cachedAffineY = lcccs.cachedAffineY
    }

    /// Convert to LCCCS (requires isRelaxed == true).
    public func toLCCCS() -> LCCCS {
        precondition(isRelaxed, "Cannot convert non-relaxed instance to LCCCS")
        if let ax = cachedAffineX, let ay = cachedAffineY {
            return LCCCS(commitment: commitment, publicInput: publicInput,
                         u: u, r: r, v: v, affineX: ax, affineY: ay)
        }
        return LCCCS(commitment: commitment, publicInput: publicInput,
                     u: u, r: r, v: v)
    }

    /// Convert to CCCS (for non-relaxed instances).
    public func toCCCS() -> CCCS {
        if let ax = cachedAffineX, let ay = cachedAffineY {
            return CCCS(commitment: commitment, publicInput: publicInput,
                        affineX: ax, affineY: ay)
        }
        return CCCS(commitment: commitment, publicInput: publicInput)
    }
}

// MARK: - Multi-fold Proof

/// Proof for a multi-instance fold step (N instances -> 1).
/// Extends the basic FoldingProof with data for N-1 cross-terms.
public struct MultiFoldProof {
    /// Per-instance sigma evaluations: sigmas[i][j] = MLE(M_j * z_i)(r)
    public let sigmas: [[Fr]]
    /// Per-instance theta evaluations (for fresh instances)
    public let thetas: [[Fr]]
    /// Sumcheck proof for the multi-fold cross-term reduction
    public let sumcheckProof: SumcheckFoldProof
    /// Number of instances that were folded
    public let instanceCount: Int
}
