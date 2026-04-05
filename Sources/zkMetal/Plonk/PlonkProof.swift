// PlonkProof — Proof structure for Plonk protocol
//
// Contains all commitments, evaluations, and opening proofs generated
// during the 5-round Plonk proving protocol:
//   - Round 1: Wire polynomial commitments [a], [b], [c]
//   - Round 2: Permutation accumulator commitment [z]
//   - Round 3: Quotient polynomial commitments [t_lo], [t_mid], [t_hi], ...
//   - Round 4: Polynomial evaluations at challenge zeta
//   - Round 5: KZG opening proofs at zeta and zeta*omega

import Foundation
import NeonFieldOps

public struct PlonkProof {
    // Round 1: Wire polynomial commitments
    public let aCommit: PointProjective
    public let bCommit: PointProjective
    public let cCommit: PointProjective

    // Round 2: Permutation accumulator commitment
    public let zCommit: PointProjective

    // Round 3: Quotient polynomial commitments (split into degree-n chunks)
    public let tLoCommit: PointProjective        // t_0(x)
    public let tMidCommit: PointProjective       // t_1(x)
    public let tHiCommit: PointProjective        // t_2(x)
    public let tExtraCommits: [PointProjective]  // additional chunks for high-degree custom gates

    // Round 4: Polynomial evaluations at challenge zeta
    public let aEval: Fr                         // a(zeta)
    public let bEval: Fr                         // b(zeta)
    public let cEval: Fr                         // c(zeta)
    public let sigma1Eval: Fr                    // sigma1(zeta)
    public let sigma2Eval: Fr                    // sigma2(zeta)
    public let zOmegaEval: Fr                    // z(zeta * omega)

    // Round 5: KZG opening proofs
    public let openingProof: PointProjective         // W_zeta
    public let shiftedOpeningProof: PointProjective  // W_{zeta*omega}

    // Public inputs (included for verifier convenience)
    public let publicInputs: [Fr]

    public init(aCommit: PointProjective, bCommit: PointProjective, cCommit: PointProjective,
                zCommit: PointProjective,
                tLoCommit: PointProjective, tMidCommit: PointProjective, tHiCommit: PointProjective,
                tExtraCommits: [PointProjective],
                aEval: Fr, bEval: Fr, cEval: Fr,
                sigma1Eval: Fr, sigma2Eval: Fr,
                zOmegaEval: Fr,
                openingProof: PointProjective,
                shiftedOpeningProof: PointProjective,
                publicInputs: [Fr] = []) {
        self.aCommit = aCommit
        self.bCommit = bCommit
        self.cCommit = cCommit
        self.zCommit = zCommit
        self.tLoCommit = tLoCommit
        self.tMidCommit = tMidCommit
        self.tHiCommit = tHiCommit
        self.tExtraCommits = tExtraCommits
        self.aEval = aEval
        self.bEval = bEval
        self.cEval = cEval
        self.sigma1Eval = sigma1Eval
        self.sigma2Eval = sigma2Eval
        self.zOmegaEval = zOmegaEval
        self.openingProof = openingProof
        self.shiftedOpeningProof = shiftedOpeningProof
        self.publicInputs = publicInputs
    }
}

// MARK: - Proving Key

/// Preprocessed proving key containing all circuit-dependent data.
/// Generated once by PlonkPreprocessor.setup(), reused for every proof.
public typealias PlonkProvingKey = PlonkSetup

// MARK: - Verification Key

/// Verification key: subset of PlonkSetup needed for verification.
/// Contains only commitments and domain parameters (no polynomial evaluations).
public struct PlonkVerificationKey {
    public let selectorCommitments: [PointProjective]
    public let permutationCommitments: [PointProjective]
    public let omega: Fr
    public let n: Int
    public let k1: Fr
    public let k2: Fr
    public let srs: [PointAffine]
    public let srsSecret: Fr  // test-only; production would use pairing
    public let lookupTables: [PlonkLookupTable]

    /// Extract verification key from a full setup
    public init(from setup: PlonkSetup) {
        self.selectorCommitments = setup.selectorCommitments
        self.permutationCommitments = setup.permutationCommitments
        self.omega = setup.omega
        self.n = setup.n
        self.k1 = setup.k1
        self.k2 = setup.k2
        self.srs = setup.srs
        self.srsSecret = setup.srsSecret
        self.lookupTables = setup.lookupTables
    }
}
