// Zeromorph Proof and Verification Key Structures
// Supporting types for the Zeromorph multilinear PCS (eprint 2023/917).
//
// The proof consists of:
//   - Quotient commitments [q^(0)], ..., [q^(n-1)] in G1
//   - A single batched KZG opening proof (witness in G1)
//   - The claimed evaluation value v
//
// The verification key contains [tau]_2 in G2 (from the KZG SRS).

import Foundation

// MARK: - Verification Key

/// Zeromorph verification key: the G2 elements needed for pairing-based verification.
///
/// In standard KZG, the SRS includes [tau^i]_1 in G1 and [tau]_2 in G2.
/// Zeromorph verification only needs [1]_2 and [tau]_2 from G2.
public struct ZeromorphVK {
    /// [1]_2: G2 generator
    public let g2Generator: G2AffinePoint
    /// [tau]_2: tau times the G2 generator
    public let tauG2: G2AffinePoint

    public init(g2Generator: G2AffinePoint, tauG2: G2AffinePoint) {
        self.g2Generator = g2Generator
        self.tauG2 = tauG2
    }

    /// Generate a test VK from a known secret (NOT secure).
    public static func generateTestVK(secret: [UInt64]) -> ZeromorphVK {
        let g2 = bn254G2Generator()
        let g2Proj = g2FromAffine(g2)
        let tauG2Proj = g2ScalarMul(g2Proj, secret)
        guard let tauG2Aff = g2ToAffine(tauG2Proj) else {
            fatalError("G2 scalar mul produced identity")
        }
        return ZeromorphVK(g2Generator: g2, tauG2: tauG2Aff)
    }
}

// MARK: - Proof

/// A Zeromorph proof for opening a multilinear polynomial at a point.
///
/// Given f(x_1,...,x_n) evaluated on the boolean hypercube {0,1}^n,
/// and an evaluation point u = (u_1,...,u_n), the proof demonstrates that
/// the ZM-fold evaluation equals the claimed value v.
///
/// The proof is verified via a single pairing check that combines
/// all n quotient commitments using a random challenge.
public struct ZeromorphOpeningProof {
    /// Commitments to quotient polynomials [q^(0)], ..., [q^(n-1)] in G1
    public let quotientCommitments: [PointProjective]

    /// The claimed evaluation value v = ZMFold(f, u)
    public let claimedValue: Fr

    /// KZG opening proof witness for the linearized polynomial L(X) at zeta.
    /// L(X) = f(X) - v - sum_s phi_s * q^(s)(X) where phi_s = zeta^{2^s} - u_{n-1-s}.
    public let kzgWitness: PointProjective

    /// Evaluation of L at zeta: L(zeta) = f(zeta) - v - sum_s phi_s * q^(s)(zeta)
    public let linearizationEval: Fr

    /// The Fiat-Shamir challenge zeta
    public let zeta: Fr

    /// Number of variables (= number of quotient commitments)
    public var numVariables: Int { quotientCommitments.count }

    /// Proof size: n G1 points for quotients + 1 G1 for KZG witness + 1 Fr for value
    public var sizeInG1Points: Int { quotientCommitments.count + 1 }

    public init(quotientCommitments: [PointProjective], claimedValue: Fr,
                kzgWitness: PointProjective, linearizationEval: Fr, zeta: Fr) {
        self.quotientCommitments = quotientCommitments
        self.claimedValue = claimedValue
        self.kzgWitness = kzgWitness
        self.linearizationEval = linearizationEval
        self.zeta = zeta
    }
}
