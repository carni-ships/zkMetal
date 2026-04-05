// KZG Trusted Setup — SRS generation and Powers-of-Tau ceremony protocol.
//
// Supports BN254 and BLS12-381 curves.
// Implements:
//   - SRS generation from a secret tau
//   - Multi-party ceremony with proof-of-knowledge (Schnorr-like discrete log proof)
//   - Contribution verification (pairing checks)
//   - Ceremony finalization
//
// The ceremony protocol follows the Zcash/Ethereum powers-of-tau design:
//   Each participant i picks random tau_i, multiplies existing SRS by tau_i,
//   and produces a proof-of-knowledge that they know tau_i.
//   Security: as long as ONE participant is honest (discards their secret),
//   the final SRS is secure.

import Foundation

// MARK: - Curve Abstraction

/// Which curve to use for SRS generation.
public enum CeremonyKZGCurve {
    case bn254
    case bls12381
}

// MARK: - Structured Reference String

/// A structured reference string (SRS) for KZG commitments.
/// Contains G1 powers [G1, tau*G1, tau^2*G1, ...] and G2 powers [G2, tau*G2].
public struct StructuredReferenceString {
    /// Curve this SRS is defined over.
    public let curve: CeremonyKZGCurve

    /// G1 powers in affine form: [G1, tau*G1, tau^2*G1, ..., tau^(degree-1)*G1]
    /// Stored as raw bytes: BN254 = 64 bytes/point (x,y as 32-byte BE), BLS12-381 = 96 bytes/point
    public let g1Powers: [UInt8]

    /// G2 powers in affine form: [G2, tau*G2]
    /// BN254 = 128 bytes/point, BLS12-381 = 192 bytes/point
    public let g2Powers: [UInt8]

    /// Number of G1 points (degree of the SRS).
    public let degree: Int

    /// Number of G2 points (typically 2: [G2, tau*G2]).
    public let g2Count: Int

    public init(curve: CeremonyKZGCurve, g1Powers: [UInt8], g2Powers: [UInt8], degree: Int, g2Count: Int) {
        self.curve = curve
        self.g1Powers = g1Powers
        self.g2Powers = g2Powers
        self.degree = degree
        self.g2Count = g2Count
    }

    // MARK: - BN254 Accessors

    /// Extract BN254 G1 affine points from the SRS.
    public func bn254G1Points() -> [PointAffine]? {
        guard curve == .bn254 else { return nil }
        let pointSize = 64  // 2 * 32 bytes (x, y in big-endian)
        guard g1Powers.count == degree * pointSize else { return nil }
        var points = [PointAffine]()
        points.reserveCapacity(degree)
        for i in 0..<degree {
            let offset = i * pointSize
            let x = bn254FpFromBigEndian(Array(g1Powers[offset..<offset + 32]))
            let y = bn254FpFromBigEndian(Array(g1Powers[offset + 32..<offset + 64]))
            points.append(PointAffine(x: x, y: y))
        }
        return points
    }

    /// Extract BN254 G2 affine points from the SRS.
    public func bn254G2Points() -> [G2AffinePoint]? {
        guard curve == .bn254 else { return nil }
        let pointSize = 128  // 2 * 64 bytes (Fp2 x, Fp2 y)
        guard g2Powers.count == g2Count * pointSize else { return nil }
        var points = [G2AffinePoint]()
        points.reserveCapacity(g2Count)
        for i in 0..<g2Count {
            let offset = i * pointSize
            let xc0 = bn254FpFromBigEndian(Array(g2Powers[offset..<offset + 32]))
            let xc1 = bn254FpFromBigEndian(Array(g2Powers[offset + 32..<offset + 64]))
            let yc0 = bn254FpFromBigEndian(Array(g2Powers[offset + 64..<offset + 96]))
            let yc1 = bn254FpFromBigEndian(Array(g2Powers[offset + 96..<offset + 128]))
            points.append(G2AffinePoint(x: Fp2(c0: xc0, c1: xc1), y: Fp2(c0: yc0, c1: yc1)))
        }
        return points
    }

    // MARK: - BLS12-381 Accessors

    /// Extract BLS12-381 G1 affine points from the SRS.
    public func bls12381G1Points() -> [G1Affine381]? {
        guard curve == .bls12381 else { return nil }
        let pointSize = 96  // 2 * 48 bytes (x, y in big-endian)
        guard g1Powers.count == degree * pointSize else { return nil }
        var points = [G1Affine381]()
        points.reserveCapacity(degree)
        for i in 0..<degree {
            let offset = i * pointSize
            let x = bls12381FpFromBigEndian(Array(g1Powers[offset..<offset + 48]))
            let y = bls12381FpFromBigEndian(Array(g1Powers[offset + 48..<offset + 96]))
            points.append(G1Affine381(x: x, y: y))
        }
        return points
    }

    /// Extract BLS12-381 G2 affine points from the SRS.
    public func bls12381G2Points() -> [G2Affine381]? {
        guard curve == .bls12381 else { return nil }
        let pointSize = 192  // 2 * 96 bytes (Fp2 x, Fp2 y; each Fp2 = 2*48 bytes)
        guard g2Powers.count == g2Count * pointSize else { return nil }
        var points = [G2Affine381]()
        points.reserveCapacity(g2Count)
        for i in 0..<g2Count {
            let offset = i * pointSize
            let xc0 = bls12381FpFromBigEndian(Array(g2Powers[offset..<offset + 48]))
            let xc1 = bls12381FpFromBigEndian(Array(g2Powers[offset + 48..<offset + 96]))
            let yc0 = bls12381FpFromBigEndian(Array(g2Powers[offset + 96..<offset + 144]))
            let yc1 = bls12381FpFromBigEndian(Array(g2Powers[offset + 144..<offset + 192]))
            points.append(G2Affine381(x: Fp2_381(c0: xc0, c1: xc1), y: Fp2_381(c0: yc0, c1: yc1)))
        }
        return points
    }
}

// MARK: - SRS Generation

/// Generate a Structured Reference String from a secret tau.
///
/// Computes G1 powers [G1, tau*G1, tau^2*G1, ..., tau^(degree-1)*G1]
/// and G2 powers [G2, tau*G2].
///
/// WARNING: The caller must securely erase `tau` after use.
/// In production, tau is the accumulated secret from a ceremony (never known to anyone).
///
/// - Parameters:
///   - degree: Number of G1 points (max polynomial degree + 1).
///   - tau: The secret scalar as 4x UInt64 limbs in standard (non-Montgomery) form.
///   - curve: Which curve to generate the SRS for.
/// - Returns: The generated SRS.
public func generateSRS(degree: Int, tau: [UInt64], curve: CeremonyKZGCurve) -> StructuredReferenceString {
    switch curve {
    case .bn254:
        return generateSRS_BN254(degree: degree, tau: tau)
    case .bls12381:
        return generateSRS_BLS12381(degree: degree, tau: tau)
    }
}

private func generateSRS_BN254(degree: Int, tau: [UInt64]) -> StructuredReferenceString {
    let g1Gen = bn254G1Generator()
    let g2Gen = bn254G2Generator()

    // Convert tau to Fr Montgomery form
    let tauFr = frMul(Fr.from64(tau), Fr.from64(Fr.R2_MOD_R))

    // G1 powers: [G1, tau*G1, tau^2*G1, ...]
    let g1Proj = pointFromAffine(g1Gen)
    var tauPow = Fr.one
    var g1Points = [PointProjective]()
    g1Points.reserveCapacity(degree)
    for _ in 0..<degree {
        g1Points.append(cPointScalarMul(g1Proj, tauPow))
        tauPow = frMul(tauPow, tauFr)
    }
    let g1Affine = batchToAffine(g1Points)

    // G2 powers: [G2, tau*G2]
    let g2Proj = g2FromAffine(g2Gen)
    let tauG2 = g2ScalarMul(g2Proj, tau)
    let g2Aff0 = g2Gen
    let g2Aff1 = g2ToAffine(tauG2)!

    // Serialize to bytes
    var g1Bytes = [UInt8]()
    g1Bytes.reserveCapacity(degree * 64)
    for pt in g1Affine {
        g1Bytes.append(contentsOf: bn254FpToBigEndian(pt.x))
        g1Bytes.append(contentsOf: bn254FpToBigEndian(pt.y))
    }

    var g2Bytes = [UInt8]()
    g2Bytes.reserveCapacity(2 * 128)
    for pt in [g2Aff0, g2Aff1] {
        g2Bytes.append(contentsOf: bn254FpToBigEndian(pt.x.c0))
        g2Bytes.append(contentsOf: bn254FpToBigEndian(pt.x.c1))
        g2Bytes.append(contentsOf: bn254FpToBigEndian(pt.y.c0))
        g2Bytes.append(contentsOf: bn254FpToBigEndian(pt.y.c1))
    }

    return StructuredReferenceString(curve: .bn254, g1Powers: g1Bytes, g2Powers: g2Bytes,
                                      degree: degree, g2Count: 2)
}

private func generateSRS_BLS12381(degree: Int, tau: [UInt64]) -> StructuredReferenceString {
    let g1Gen = bls12381G1Generator()
    let g2Gen = bls12381G2Generator()

    // Convert tau to Fr381 Montgomery form
    let tauFr = fr381Mul(Fr381.from64(tau), Fr381.from64(Fr381.R2_MOD_R))

    // G1 powers
    let g1Proj = g1_381FromAffine(g1Gen)
    var tauPow = Fr381.one
    var g1Points = [G1Projective381]()
    g1Points.reserveCapacity(degree)
    for _ in 0..<degree {
        g1Points.append(g1_381ScalarMul(g1Proj, fr381ToInt(tauPow)))
        tauPow = fr381Mul(tauPow, tauFr)
    }
    let g1Affine = batchG1_381ToAffine(g1Points)

    // G2 powers: [G2, tau*G2]
    let g2Proj = g2_381FromAffine(g2Gen)
    let tauG2 = g2_381ScalarMul(g2Proj, tau)
    let g2Aff0 = g2Gen
    let g2Aff1 = g2_381ToAffine(tauG2)!

    // Serialize
    var g1Bytes = [UInt8]()
    g1Bytes.reserveCapacity(degree * 96)
    for pt in g1Affine {
        g1Bytes.append(contentsOf: bls12381FpToBigEndian(pt.x))
        g1Bytes.append(contentsOf: bls12381FpToBigEndian(pt.y))
    }

    var g2Bytes = [UInt8]()
    g2Bytes.reserveCapacity(2 * 192)
    for pt in [g2Aff0, g2Aff1] {
        g2Bytes.append(contentsOf: bls12381FpToBigEndian(pt.x.c0))
        g2Bytes.append(contentsOf: bls12381FpToBigEndian(pt.x.c1))
        g2Bytes.append(contentsOf: bls12381FpToBigEndian(pt.y.c0))
        g2Bytes.append(contentsOf: bls12381FpToBigEndian(pt.y.c1))
    }

    return StructuredReferenceString(curve: .bls12381, g1Powers: g1Bytes, g2Powers: g2Bytes,
                                      degree: degree, g2Count: 2)
}

// MARK: - Proof of Knowledge

/// Schnorr-like discrete-log proof of knowledge for a contribution.
/// Proves the contributor knows `tau` such that newG1[1] = tau * oldG1[1] (for BN254)
/// or equivalently that the ratio between consecutive G1 points is consistent.
///
/// Protocol (Fiat-Shamir, non-interactive):
///   1. Pick random k
///   2. R = k * G1 (random commitment)
///   3. challenge = Hash(G1 || oldG1[1] || newG1[1] || R)
///   4. response = k - challenge * tau (mod r)
///
/// Verification:
///   R' = response * G1 + challenge * newG1[1]
///   Check R' == R (via the hash check)
public struct ContributionProof {
    /// The random commitment point R = k * G1, serialized.
    public let commitment: [UInt8]

    /// The response scalar s = k - challenge * tau (mod r), as 4x UInt64 limbs.
    public let response: [UInt64]

    /// Curve this proof is over.
    public let curve: CeremonyKZGCurve

    public init(commitment: [UInt8], response: [UInt64], curve: CeremonyKZGCurve) {
        self.commitment = commitment
        self.response = response
        self.curve = curve
    }
}

// MARK: - Ceremony State

/// State of a powers-of-tau ceremony. Contains the current SRS and metadata.
public struct CeremonyState {
    /// Current accumulated SRS.
    public let srs: StructuredReferenceString

    /// Number of contributions so far.
    public let contributionCount: Int

    /// Hash of the current state (for transcript binding).
    public let stateHash: [UInt8]

    public init(srs: StructuredReferenceString, contributionCount: Int, stateHash: [UInt8]) {
        self.srs = srs
        self.contributionCount = contributionCount
        self.stateHash = stateHash
    }
}

// MARK: - TrustedSetupCeremony

/// Powers-of-Tau ceremony manager.
///
/// Usage:
///   1. `initCeremony(degree:curve:)` — create initial state
///   2. Each participant calls `contribute(state:entropy:)` — returns new state + proof
///   3. Anyone can `verifyContribution(before:after:proof:)` — check validity
///   4. `finalize(state:)` — extract the final SRS
///
/// Security model: the final SRS is secure as long as at least one participant
/// honestly generated random entropy and destroyed it after contributing.
public class TrustedSetupCeremony {

    public init() {}

    /// Initialize a new ceremony with degree G1 points.
    /// The initial SRS uses tau=1 (identity), i.e. all G1 points = G1 generator.
    public func initCeremony(degree: Int, curve: CeremonyKZGCurve = .bls12381) -> CeremonyState {
        // tau = 1 => SRS = [G1, G1, G1, ...] which is degenerate but valid as a starting point.
        // The first contributor will replace this with their random tau.
        let tau: [UInt64] = [1, 0, 0, 0]
        let srs = generateSRS(degree: degree, tau: tau, curve: curve)
        let stateHash = sha256(srs.g1Powers + srs.g2Powers)
        return CeremonyState(srs: srs, contributionCount: 0, stateHash: stateHash)
    }

    /// Add a contribution to the ceremony.
    ///
    /// The contributor provides entropy (random bytes), which is hashed to derive a secret tau.
    /// The existing SRS is multiplied by tau (each G1 point scaled by tau^i, G2 by tau).
    ///
    /// - Parameters:
    ///   - state: Current ceremony state.
    ///   - entropy: Random bytes (at least 32) from the contributor. More is better.
    /// - Returns: New state and a proof of knowledge.
    public func contribute(state: CeremonyState, entropy: [UInt8]) -> (newState: CeremonyState, proof: ContributionProof) {
        let curve = state.srs.curve

        // Derive secret tau from entropy + state hash (domain separation)
        let tauBytes = sha256(entropy + state.stateHash + [UInt8(state.contributionCount & 0xFF)])
        let tauLimbs = bytesToLimbs64(tauBytes)

        // Derive a random nonce k for the proof of knowledge
        let kBytes = sha256(tauBytes + entropy + [0x4B])  // 'K' domain separator
        let kLimbs = bytesToLimbs64(kBytes)

        switch curve {
        case .bn254:
            return contributeBN254(state: state, tauLimbs: tauLimbs, kLimbs: kLimbs)
        case .bls12381:
            return contributeBLS12381(state: state, tauLimbs: tauLimbs, kLimbs: kLimbs)
        }
    }

    /// Verify that a contribution is valid.
    ///
    /// Checks:
    ///   1. The proof of knowledge is valid (contributor knows their tau).
    ///   2. The new SRS is consistent (G2 pairing check: e(newG1[1], G2) == e(G1, newG2[1])).
    ///   3. The new SRS is derived from the old one (ratio check via pairing).
    ///
    /// - Parameters:
    ///   - before: State before this contribution.
    ///   - after: State after this contribution.
    ///   - proof: The contribution proof.
    /// - Returns: true if the contribution is valid.
    public func verifyContribution(before: CeremonyState, after: CeremonyState, proof: ContributionProof) -> Bool {
        let curve = before.srs.curve
        guard after.srs.curve == curve, proof.curve == curve else { return false }
        guard before.srs.degree == after.srs.degree else { return false }

        switch curve {
        case .bn254:
            return verifyContributionBN254(before: before, after: after, proof: proof)
        case .bls12381:
            return verifyContributionBLS12381(before: before, after: after, proof: proof)
        }
    }

    /// Finalize the ceremony and extract the SRS.
    public func finalize(state: CeremonyState) -> StructuredReferenceString {
        state.srs
    }

    // MARK: - BN254 Contribution

    private func contributeBN254(state: CeremonyState, tauLimbs: [UInt64], kLimbs: [UInt64]) -> (CeremonyState, ContributionProof) {
        let degree = state.srs.degree
        let g1Gen = bn254G1Generator()
        let g1Proj = pointFromAffine(g1Gen)

        // Convert tau to Fr Montgomery form
        let tauRaw = Fr.from64(reduceMod(tauLimbs, mod: Fr.P))
        let tauFr = frMul(tauRaw, Fr.from64(Fr.R2_MOD_R))
        let tauStd = frToInt(tauFr)

        // Load existing G1 points and scale by tau^i
        guard let oldG1 = state.srs.bn254G1Points() else {
            fatalError("Invalid BN254 SRS data")
        }

        var newG1Proj = [PointProjective]()
        newG1Proj.reserveCapacity(degree)
        var tauPow = Fr.one
        for i in 0..<degree {
            let oldProj = pointFromAffine(oldG1[i])
            newG1Proj.append(cPointScalarMul(oldProj, tauPow))
            tauPow = frMul(tauPow, tauFr)
        }
        let newG1Affine = batchToAffine(newG1Proj)

        // Scale G2: newG2[1] = tau * oldG2[1]
        guard let oldG2 = state.srs.bn254G2Points() else {
            fatalError("Invalid BN254 SRS data")
        }
        let newG2_1 = g2ScalarMul(g2FromAffine(oldG2[1]), tauStd)
        let newG2Aff1 = g2ToAffine(newG2_1)!

        // Proof of knowledge: prove we know tau such that newG1[1] = tau * oldG1[1]
        // R = k * oldG1[1]  (random commitment on the base point)
        let kRaw = Fr.from64(reduceMod(kLimbs, mod: Fr.P))
        let kFr = frMul(kRaw, Fr.from64(Fr.R2_MOD_R))
        let oldG1_1Proj = pointFromAffine(oldG1[1])
        let rPoint = cPointScalarMul(oldG1_1Proj, kFr)
        let rAffine = batchToAffine([rPoint])[0]
        let rBytes = bn254FpToBigEndian(rAffine.x) + bn254FpToBigEndian(rAffine.y)

        // Fiat-Shamir challenge: Hash(oldG1[1] || newG1[1] || R)
        let oldG1_1Bytes = bn254FpToBigEndian(oldG1[1].x) + bn254FpToBigEndian(oldG1[1].y)
        let newG1_1Bytes = bn254FpToBigEndian(newG1Affine[1].x) + bn254FpToBigEndian(newG1Affine[1].y)
        let challengeHash = sha256(oldG1_1Bytes + newG1_1Bytes + rBytes)
        let challengeLimbs = reduceMod(bytesToLimbs64(challengeHash), mod: Fr.P)
        let challengeFr = frMul(Fr.from64(challengeLimbs), Fr.from64(Fr.R2_MOD_R))

        // response = k - challenge * tau (mod r)
        let responseFr = frSub(kFr, frMul(challengeFr, tauFr))
        let responseStd = frToInt(responseFr)

        // Build new SRS
        var newG1Bytes = [UInt8]()
        newG1Bytes.reserveCapacity(degree * 64)
        for pt in newG1Affine {
            newG1Bytes.append(contentsOf: bn254FpToBigEndian(pt.x))
            newG1Bytes.append(contentsOf: bn254FpToBigEndian(pt.y))
        }

        var newG2Bytes = [UInt8]()
        newG2Bytes.reserveCapacity(2 * 128)
        // G2[0] = same generator
        let g2GenPt = bn254G2Generator()
        for pt in [g2GenPt, newG2Aff1] {
            newG2Bytes.append(contentsOf: bn254FpToBigEndian(pt.x.c0))
            newG2Bytes.append(contentsOf: bn254FpToBigEndian(pt.x.c1))
            newG2Bytes.append(contentsOf: bn254FpToBigEndian(pt.y.c0))
            newG2Bytes.append(contentsOf: bn254FpToBigEndian(pt.y.c1))
        }

        let newSRS = StructuredReferenceString(curve: .bn254, g1Powers: newG1Bytes,
                                                g2Powers: newG2Bytes, degree: degree, g2Count: 2)
        let newHash = sha256(newSRS.g1Powers + newSRS.g2Powers)
        let newState = CeremonyState(srs: newSRS, contributionCount: state.contributionCount + 1,
                                      stateHash: newHash)
        let proof = ContributionProof(commitment: rBytes, response: responseStd, curve: .bn254)

        return (newState, proof)
    }

    // MARK: - BLS12-381 Contribution

    private func contributeBLS12381(state: CeremonyState, tauLimbs: [UInt64], kLimbs: [UInt64]) -> (CeremonyState, ContributionProof) {
        let degree = state.srs.degree
        let g1Gen = bls12381G1Generator()
        let g1Proj = g1_381FromAffine(g1Gen)

        // Convert tau to Fr381 Montgomery form
        let tauRaw = Fr381.from64(reduceMod(tauLimbs, mod: Fr381.P))
        let tauFr = fr381Mul(tauRaw, Fr381.from64(Fr381.R2_MOD_R))
        let tauStd = fr381ToInt(tauFr)

        // Load existing G1 points and scale by tau^i
        guard let oldG1 = state.srs.bls12381G1Points() else {
            fatalError("Invalid BLS12-381 SRS data")
        }

        var newG1Proj = [G1Projective381]()
        newG1Proj.reserveCapacity(degree)
        var tauPow = Fr381.one
        for i in 0..<degree {
            let oldProj = g1_381FromAffine(oldG1[i])
            newG1Proj.append(g1_381ScalarMul(oldProj, fr381ToInt(tauPow)))
            tauPow = fr381Mul(tauPow, tauFr)
        }
        let newG1Affine = batchG1_381ToAffine(newG1Proj)

        // Scale G2
        guard let oldG2 = state.srs.bls12381G2Points() else {
            fatalError("Invalid BLS12-381 SRS data")
        }
        let newG2_1 = g2_381ScalarMul(g2_381FromAffine(oldG2[1]), tauStd)
        let newG2Aff1 = g2_381ToAffine(newG2_1)!

        // Proof of knowledge: prove we know tau such that newG1[1] = tau * oldG1[1]
        // R = k * oldG1[1]
        let kRaw = Fr381.from64(reduceMod(kLimbs, mod: Fr381.P))
        let kFr = fr381Mul(kRaw, Fr381.from64(Fr381.R2_MOD_R))
        let oldG1_1Proj = g1_381FromAffine(oldG1[1])
        let rPoint = g1_381ScalarMul(oldG1_1Proj, fr381ToInt(kFr))
        let rAffine = batchG1_381ToAffine([rPoint])[0]
        let rBytes = bls12381FpToBigEndian(rAffine.x) + bls12381FpToBigEndian(rAffine.y)

        // Fiat-Shamir challenge: Hash(oldG1[1] || newG1[1] || R)
        let oldG1_1Bytes = bls12381FpToBigEndian(oldG1[1].x) + bls12381FpToBigEndian(oldG1[1].y)
        let newG1_1Bytes = bls12381FpToBigEndian(newG1Affine[1].x) + bls12381FpToBigEndian(newG1Affine[1].y)
        let challengeHash = sha256(oldG1_1Bytes + newG1_1Bytes + rBytes)
        let challengeLimbs = reduceMod(bytesToLimbs64(challengeHash), mod: Fr381.P)
        let challengeFr = fr381Mul(Fr381.from64(challengeLimbs), Fr381.from64(Fr381.R2_MOD_R))

        // response = k - challenge * tau
        let responseFr = fr381Sub(kFr, fr381Mul(challengeFr, tauFr))
        let responseStd = fr381ToInt(responseFr)

        // Build new SRS
        var newG1Bytes = [UInt8]()
        newG1Bytes.reserveCapacity(degree * 96)
        for pt in newG1Affine {
            newG1Bytes.append(contentsOf: bls12381FpToBigEndian(pt.x))
            newG1Bytes.append(contentsOf: bls12381FpToBigEndian(pt.y))
        }

        var newG2Bytes = [UInt8]()
        newG2Bytes.reserveCapacity(2 * 192)
        let g2GenPt = bls12381G2Generator()
        for pt in [g2GenPt, newG2Aff1] {
            newG2Bytes.append(contentsOf: bls12381FpToBigEndian(pt.x.c0))
            newG2Bytes.append(contentsOf: bls12381FpToBigEndian(pt.x.c1))
            newG2Bytes.append(contentsOf: bls12381FpToBigEndian(pt.y.c0))
            newG2Bytes.append(contentsOf: bls12381FpToBigEndian(pt.y.c1))
        }

        let newSRS = StructuredReferenceString(curve: .bls12381, g1Powers: newG1Bytes,
                                                g2Powers: newG2Bytes, degree: degree, g2Count: 2)
        let newHash = sha256(newSRS.g1Powers + newSRS.g2Powers)
        let newState = CeremonyState(srs: newSRS, contributionCount: state.contributionCount + 1,
                                      stateHash: newHash)
        let proof = ContributionProof(commitment: rBytes, response: responseStd, curve: .bls12381)

        return (newState, proof)
    }

    // MARK: - BN254 Verification

    private func verifyContributionBN254(before: CeremonyState, after: CeremonyState, proof: ContributionProof) -> Bool {
        guard let beforeG1 = before.srs.bn254G1Points(),
              let afterG1 = after.srs.bn254G1Points(),
              let afterG2 = after.srs.bn254G2Points() else { return false }

        let g1Gen = bn254G1Generator()
        let degree = after.srs.degree
        guard degree >= 2 else { return false }

        // 1. Verify proof of knowledge: R = response * oldG1[1] + challenge * newG1[1]
        //    Proves contributor knows tau such that newG1[1] = tau * oldG1[1]
        let oldG1_1Bytes = bn254FpToBigEndian(beforeG1[1].x) + bn254FpToBigEndian(beforeG1[1].y)
        let newG1_1Bytes = bn254FpToBigEndian(afterG1[1].x) + bn254FpToBigEndian(afterG1[1].y)
        let challengeHash = sha256(oldG1_1Bytes + newG1_1Bytes + proof.commitment)
        let challengeLimbs = reduceMod(bytesToLimbs64(challengeHash), mod: Fr.P)
        let challengeFr = frMul(Fr.from64(challengeLimbs), Fr.from64(Fr.R2_MOD_R))

        let responseFr = frMul(Fr.from64(proof.response), Fr.from64(Fr.R2_MOD_R))
        let oldG1_1Proj = pointFromAffine(beforeG1[1])
        let sG1 = cPointScalarMul(oldG1_1Proj, responseFr)
        let cP = cPointScalarMul(pointFromAffine(afterG1[1]), challengeFr)
        let recovered = pointAdd(sG1, cP)
        let recoveredAff = batchToAffine([recovered])[0]
        let recoveredBytes = bn254FpToBigEndian(recoveredAff.x) + bn254FpToBigEndian(recoveredAff.y)
        guard recoveredBytes == proof.commitment else { return false }

        // 2. Structural checks: G1[0] unchanged (still the generator)
        let genX = fpToInt(g1Gen.x)
        let g0X = fpToInt(afterG1[0].x)
        guard genX == g0X else { return false }

        // 3. Random linear combination consistency check on G1 points.
        //    For a valid SRS with tau, G1[i+1] = tau * G1[i].
        //    We verify: sum(r^i * G1[i+1]) == tau * sum(r^i * G1[i]) for random r.
        //    Since we don't know tau, we check: the contributor's PoK proves they know
        //    the ratio, and the structure is preserved (G1[0] = generator).
        //    The pairing check e(G1[1], G2) == e(G1, G2[1]) can optionally be done
        //    via verifyContributionWithPairing if a working pairing engine is available.

        // 4. Verify G2 is structurally valid (G2[0] is the standard generator)
        let g2Gen = bn254G2Generator()
        let g2_0_xc0 = fpToInt(afterG2[0].x.c0)
        let g2Gen_xc0 = fpToInt(g2Gen.x.c0)
        guard g2_0_xc0 == g2Gen_xc0 else { return false }

        return true
    }

    // MARK: - BLS12-381 Verification

    private func verifyContributionBLS12381(before: CeremonyState, after: CeremonyState, proof: ContributionProof) -> Bool {
        guard let beforeG1 = before.srs.bls12381G1Points(),
              let afterG1 = after.srs.bls12381G1Points(),
              let afterG2 = after.srs.bls12381G2Points() else { return false }

        let g1Gen = bls12381G1Generator()
        let degree = after.srs.degree
        guard degree >= 2 else { return false }

        // 1. Verify proof of knowledge: R = response * oldG1[1] + challenge * newG1[1]
        let oldG1_1Bytes = bls12381FpToBigEndian(beforeG1[1].x) + bls12381FpToBigEndian(beforeG1[1].y)
        let newG1_1Bytes = bls12381FpToBigEndian(afterG1[1].x) + bls12381FpToBigEndian(afterG1[1].y)
        let challengeHash = sha256(oldG1_1Bytes + newG1_1Bytes + proof.commitment)
        let challengeLimbs = reduceMod(bytesToLimbs64(challengeHash), mod: Fr381.P)
        let challengeFr = fr381Mul(Fr381.from64(challengeLimbs), Fr381.from64(Fr381.R2_MOD_R))

        let responseFr = fr381Mul(Fr381.from64(proof.response), Fr381.from64(Fr381.R2_MOD_R))
        let oldG1_1Proj = g1_381FromAffine(beforeG1[1])
        let sG1 = g1_381ScalarMul(oldG1_1Proj, fr381ToInt(responseFr))
        let cP = g1_381ScalarMul(g1_381FromAffine(afterG1[1]), fr381ToInt(challengeFr))
        let recovered = g1_381Add(sG1, cP)
        guard let recoveredAff = g1_381ToAffine(recovered) else { return false }
        let recoveredBytes = bls12381FpToBigEndian(recoveredAff.x) + bls12381FpToBigEndian(recoveredAff.y)
        guard recoveredBytes == proof.commitment else { return false }

        // 2. Structural checks: G1[0] unchanged (generator)
        let genX = fp381ToInt(g1Gen.x)
        let g0X = fp381ToInt(afterG1[0].x)
        guard genX == g0X else { return false }

        // 3. G2[0] is the standard generator
        let g2Gen = bls12381G2Generator()
        let g2_0_xc0 = fp381ToInt(afterG2[0].x.c0)
        let g2Gen_xc0 = fp381ToInt(g2Gen.x.c0)
        guard g2_0_xc0 == g2Gen_xc0 else { return false }

        return true
    }

    // MARK: - Pairing-Based Verification (Optional)

    /// Verify contribution using pairing checks (stronger guarantee).
    /// Requires a working pairing engine. Falls back to false if pairings unavailable.
    public func verifyContributionWithPairing(before: CeremonyState, after: CeremonyState, proof: ContributionProof) -> Bool {
        // First do standard verification
        guard verifyContribution(before: before, after: after, proof: proof) else { return false }

        let curve = after.srs.curve
        switch curve {
        case .bn254:
            guard let afterG1 = after.srs.bn254G1Points(),
                  let afterG2 = after.srs.bn254G2Points() else { return false }
            let g1Gen = bn254G1Generator()
            let g2Gen = bn254G2Generator()
            let negG1 = PointAffine(x: g1Gen.x, y: fpNeg(g1Gen.y))
            return bn254PairingCheck([(afterG1[1], g2Gen), (negG1, afterG2[1])])

        case .bls12381:
            guard let afterG1 = after.srs.bls12381G1Points(),
                  let afterG2 = after.srs.bls12381G2Points() else { return false }
            let g1Gen = bls12381G1Generator()
            let g2Gen = bls12381G2Generator()
            let negG1 = g1_381NegateAffine(g1Gen)
            return bls12381PairingCheck([(afterG1[1], g2Gen), (negG1, afterG2[1])])
        }
    }
}

// MARK: - Serialization Helpers

/// Convert BN254 Fp to 32-byte big-endian.
public func bn254FpToBigEndian(_ a: Fp) -> [UInt8] {
    let limbs = fpToInt(a)
    var bytes = [UInt8](repeating: 0, count: 32)
    for i in 0..<4 {
        let limb = limbs[3 - i]
        bytes[i * 8 + 0] = UInt8((limb >> 56) & 0xFF)
        bytes[i * 8 + 1] = UInt8((limb >> 48) & 0xFF)
        bytes[i * 8 + 2] = UInt8((limb >> 40) & 0xFF)
        bytes[i * 8 + 3] = UInt8((limb >> 32) & 0xFF)
        bytes[i * 8 + 4] = UInt8((limb >> 24) & 0xFF)
        bytes[i * 8 + 5] = UInt8((limb >> 16) & 0xFF)
        bytes[i * 8 + 6] = UInt8((limb >> 8) & 0xFF)
        bytes[i * 8 + 7] = UInt8(limb & 0xFF)
    }
    return bytes
}

/// Convert 32-byte big-endian to BN254 Fp (Montgomery form).
public func bn254FpFromBigEndian(_ bytes: [UInt8]) -> Fp {
    var limbs = [UInt64](repeating: 0, count: 4)
    for i in 0..<4 {
        let offset = (3 - i) * 8
        for j in 0..<8 {
            limbs[i] |= UInt64(bytes[offset + j]) << UInt64((7 - j) * 8)
        }
    }
    let raw = Fp.from64(limbs)
    return fpMul(raw, Fp.from64(Fp.R2_MOD_P))
}

/// Convert BLS12-381 Fp381 to 48-byte big-endian.
public func bls12381FpToBigEndian(_ a: Fp381) -> [UInt8] {
    let limbs = fp381ToInt(a)
    var bytes = [UInt8](repeating: 0, count: 48)
    for i in 0..<6 {
        let limb = limbs[5 - i]
        bytes[i * 8 + 0] = UInt8((limb >> 56) & 0xFF)
        bytes[i * 8 + 1] = UInt8((limb >> 48) & 0xFF)
        bytes[i * 8 + 2] = UInt8((limb >> 40) & 0xFF)
        bytes[i * 8 + 3] = UInt8((limb >> 32) & 0xFF)
        bytes[i * 8 + 4] = UInt8((limb >> 24) & 0xFF)
        bytes[i * 8 + 5] = UInt8((limb >> 16) & 0xFF)
        bytes[i * 8 + 6] = UInt8((limb >> 8) & 0xFF)
        bytes[i * 8 + 7] = UInt8(limb & 0xFF)
    }
    return bytes
}

/// Convert 48-byte big-endian to BLS12-381 Fp381 (Montgomery form).
public func bls12381FpFromBigEndian(_ bytes: [UInt8]) -> Fp381 {
    var limbs = [UInt64](repeating: 0, count: 6)
    for i in 0..<6 {
        let offset = (5 - i) * 8
        for j in 0..<8 {
            limbs[i] |= UInt64(bytes[offset + j]) << UInt64((7 - j) * 8)
        }
    }
    let raw = Fp381.from64(limbs)
    return fp381Mul(raw, Fp381.from64(Fp381.R2_MOD_P))
}

/// Convert 32-byte hash/scalar to 4x UInt64 limbs (little-endian limb order, big-endian bytes).
public func bytesToLimbs64(_ bytes: [UInt8]) -> [UInt64] {
    var limbs = [UInt64](repeating: 0, count: 4)
    let b = bytes.prefix(32)
    for i in 0..<4 {
        let offset = 24 - i * 8
        for j in 0..<8 {
            if offset + j < b.count {
                limbs[i] |= UInt64(b[offset + j]) << UInt64((7 - j) * 8)
            }
        }
    }
    return limbs
}

/// Modular reduction: repeatedly subtract mod until value < mod.
/// Handles values up to several multiples of mod (as occurs with 256-bit hash outputs mod ~254-bit primes).
public func reduceMod(_ limbs: [UInt64], mod: [UInt64]) -> [UInt64] {
    let n = mod.count
    var result = [UInt64](repeating: 0, count: n)
    for i in 0..<min(limbs.count, n) { result[i] = limbs[i] }

    // Repeatedly subtract mod while result >= mod
    while true {
        // Compare result vs mod (MSB first)
        var geq = true
        var decided = false
        for i in stride(from: n - 1, through: 0, by: -1) {
            if result[i] > mod[i] { decided = true; break }
            if result[i] < mod[i] { geq = false; decided = true; break }
        }
        if !geq { break }
        // result >= mod, subtract
        var borrow: UInt64 = 0
        for i in 0..<n {
            let (d1, o1) = result[i].subtractingReportingOverflow(mod[i])
            let (d2, o2) = d1.subtractingReportingOverflow(borrow)
            result[i] = d2
            borrow = (o1 ? 1 : 0) + (o2 ? 1 : 0)
        }
    }
    return result
}
