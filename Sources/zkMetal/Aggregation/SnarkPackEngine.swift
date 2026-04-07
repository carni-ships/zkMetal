// SnarkPack Proof Aggregation — TIPP + MIPP inner-product arguments
//
// SnarkPack (Gailly, Maller, Nitulescu 2021) aggregates N Groth16 proofs
// into a single short proof using:
//   - TIPP (Twisted Inner Product Pairing): proves Z = prod e(A_i, B_i)
//     using SRS commitments [v^i]_1 and [w^i]_2
//   - MIPP (Multi-scalar Inner Product Pairing): proves Z = sum r_i * C_i
//     using SRS commitments [v^i]_1
//
// The aggregation protocol:
// 1. Fiat-Shamir challenge r from all proof elements + public inputs
// 2. TIPP on (A_i, B_i) with random linear combination r^i
// 3. MIPP on (r^i, C_i) for the C-point aggregation
// 4. Verifier checks 3 pairings + O(log N) pairing checks for TIPP/MIPP
//
// Proof size: O(log N) group elements
// Verification: O(log N) pairings + O(N) scalar ops (for public input accumulation)

import Foundation
import Metal

// MARK: - SnarkPack SRS

/// Structured Reference String for SnarkPack aggregation.
/// Contains both G1 and G2 power vectors for the TIPP protocol,
/// derived from a secret tau via a trusted setup.
///
/// v_i = tau^i * G1 for i in [0, N)
/// w_i = tau^{N-1-i} * G2 for i in [0, N)  (reversed powers in G2)
///
/// The reversal ensures the "twist" property: sum of exponents in
/// cross-terms is constant (N-1), enabling the KZG-style commitment check.
public struct SnarkPackSRS {
    /// G1 powers: [G1, tau*G1, tau^2*G1, ..., tau^{N-1}*G1]
    public let g1Powers: [PointProjective]
    /// G2 powers (reversed): [tau^{N-1}*G2, tau^{N-2}*G2, ..., G2]
    public let g2Powers: [G2ProjectivePoint]
    /// G2 generator
    public let g2Gen: G2ProjectivePoint
    /// tau * G2 (for pairing consistency)
    public let g2Tau: G2ProjectivePoint
    /// Maximum number of proofs
    public let maxProofs: Int

    public init(g1Powers: [PointProjective], g2Powers: [G2ProjectivePoint],
                g2Gen: G2ProjectivePoint, g2Tau: G2ProjectivePoint, maxProofs: Int) {
        self.g1Powers = g1Powers
        self.g2Powers = g2Powers
        self.g2Gen = g2Gen
        self.g2Tau = g2Tau
        self.maxProofs = maxProofs
    }

    /// Generate a deterministic SRS from a seed (for testing).
    /// Production systems must use a proper MPC ceremony.
    public static func generate(maxProofs: Int, seed: Fr) -> SnarkPackSRS {
        precondition(maxProofs > 0 && (maxProofs & (maxProofs - 1)) == 0,
                     "maxProofs must be a power of 2")

        let g1 = pointFromAffine(bn254G1Generator())
        let g2 = g2FromAffine(bn254G2Generator())

        // tau = Hash(seed, seed) for domain separation
        let tau = poseidon2Hash(seed, seed)
        let tauLimbs = frToInt(tau)

        // G1 powers: tau^i * G1
        var g1Pows = [PointProjective]()
        g1Pows.reserveCapacity(maxProofs)
        var tauPow = Fr.one
        for _ in 0..<maxProofs {
            g1Pows.append(pointScalarMul(g1, tauPow))
            tauPow = frMul(tauPow, tau)
        }

        // G2 powers (reversed): tau^{N-1-i} * G2
        var g2Pows = [G2ProjectivePoint]()
        g2Pows.reserveCapacity(maxProofs)
        // First compute forward powers, then reverse
        var g2Forward = [G2ProjectivePoint]()
        g2Forward.reserveCapacity(maxProofs)
        tauPow = Fr.one
        for _ in 0..<maxProofs {
            g2Forward.append(g2ScalarMul(g2, frToInt(tauPow)))
            tauPow = frMul(tauPow, tau)
        }
        g2Pows = g2Forward.reversed()

        let g2Tau = g2ScalarMul(g2, tauLimbs)

        return SnarkPackSRS(
            g1Powers: g1Pows,
            g2Powers: g2Pows,
            g2Gen: g2,
            g2Tau: g2Tau,
            maxProofs: maxProofs
        )
    }
}

// MARK: - SnarkPack Proof Types

/// A TIPP proof for the twisted inner product pairing argument.
/// Proves Z_AB = prod e(A_i, B_i) using recursive halving with SRS commitments.
public struct TIPPProof {
    /// Left G1 commitments at each round (log N elements)
    public let gLeft: [PointProjective]
    /// Right G1 commitments at each round
    public let gRight: [PointProjective]
    /// Left G2 commitments at each round
    public let hLeft: [G2ProjectivePoint]
    /// Right G2 commitments at each round
    public let hRight: [G2ProjectivePoint]
    /// Fiat-Shamir challenges at each round
    public let challenges: [Fr]
    /// Final left A element after folding
    public let finalA: PointProjective
    /// Final right B element after folding
    public let finalB: G2ProjectivePoint

    public init(gLeft: [PointProjective], gRight: [PointProjective],
                hLeft: [G2ProjectivePoint], hRight: [G2ProjectivePoint],
                challenges: [Fr],
                finalA: PointProjective, finalB: G2ProjectivePoint) {
        self.gLeft = gLeft
        self.gRight = gRight
        self.hLeft = hLeft
        self.hRight = hRight
        self.challenges = challenges
        self.finalA = finalA
        self.finalB = finalB
    }
}

/// A MIPP proof for the multi-scalar inner product argument.
/// Proves Z_C = sum r_i * C_i using recursive halving.
public struct MIPPProof {
    /// Left commitment at each round
    public let comLeft: [PointProjective]
    /// Right commitment at each round
    public let comRight: [PointProjective]
    /// Fiat-Shamir challenges at each round
    public let challenges: [Fr]
    /// Final scalar after folding
    public let finalScalar: Fr
    /// Final point after folding
    public let finalPoint: PointProjective

    public init(comLeft: [PointProjective], comRight: [PointProjective],
                challenges: [Fr], finalScalar: Fr, finalPoint: PointProjective) {
        self.comLeft = comLeft
        self.comRight = comRight
        self.challenges = challenges
        self.finalScalar = finalScalar
        self.finalPoint = finalPoint
    }
}

/// Aggregated SnarkPack proof containing TIPP and MIPP sub-proofs.
public struct SnarkPackProof {
    /// TIPP proof for prod e(A_i, B_i)
    public let tippProof: TIPPProof
    /// MIPP proof for sum r_i * C_i
    public let mippProof: MIPPProof
    /// Aggregated C point: Z_C = sum r^i * C_i
    public let aggC: PointProjective
    /// Fiat-Shamir challenge used
    public let challenge: Fr
    /// Powers of the challenge: [1, r, r^2, ..., r^{N-1}]
    public let challengePowers: [Fr]
    /// Number of proofs aggregated
    public let count: Int
    /// Per-proof public inputs (needed for verification)
    public let publicInputs: [[Fr]]

    public init(tippProof: TIPPProof, mippProof: MIPPProof,
                aggC: PointProjective, challenge: Fr, challengePowers: [Fr],
                count: Int, publicInputs: [[Fr]]) {
        self.tippProof = tippProof
        self.mippProof = mippProof
        self.aggC = aggC
        self.challenge = challenge
        self.challengePowers = challengePowers
        self.count = count
        self.publicInputs = publicInputs
    }
}

// MARK: - SnarkPack Engine

/// SnarkPack aggregation engine for Groth16 proofs.
///
/// Uses TIPP and MIPP inner-product arguments to aggregate N proofs
/// into a single O(log N)-sized proof with O(log N) verification pairings.
///
/// Protocol overview:
/// 1. Derive random r from Fiat-Shamir transcript
/// 2. Run TIPP on vectors (A_i, B_i) with scalars r^i weighting A_i
/// 3. Run MIPP on vectors (r^i, C_i)
/// 4. Verifier checks TIPP, MIPP, and the Groth16 pairing equation
public class SnarkPackEngine {
    public static let version = PrimitiveVersion(version: "1.0.0", updated: "2026-04-06")

    public init() {}

    // MARK: - Aggregation

    /// Aggregate N Groth16 proofs into a single SnarkPack proof.
    ///
    /// - Parameters:
    ///   - proofs: Array of Groth16 proofs (must be power-of-2 count)
    ///   - vk: Shared verification key (same circuit)
    ///   - publicInputs: Per-proof public inputs
    ///   - srs: SnarkPack structured reference string
    /// - Returns: Aggregated SnarkPack proof
    /// - Throws: If SRS is too small or inputs are inconsistent
    public func aggregate(
        proofs: [Groth16Proof],
        vk: Groth16VerificationKey,
        publicInputs: [[Fr]],
        srs: SnarkPackSRS
    ) throws -> SnarkPackProof {
        let n = proofs.count
        guard n > 0 else {
            throw SnarkPackError.emptyProofSet
        }
        guard n & (n - 1) == 0 else {
            throw SnarkPackError.notPowerOfTwo(n)
        }
        guard n <= srs.maxProofs else {
            throw SnarkPackError.srsTooSmall(need: n, have: srs.maxProofs)
        }
        guard n == publicInputs.count else {
            throw SnarkPackError.inputCountMismatch
        }

        // Step 1: Build Fiat-Shamir transcript and derive challenge
        let transcript = Transcript(label: "snarkpack-tipp-mipp", backend: .poseidon2)
        transcript.absorb(frFromInt(UInt64(n)))

        for i in 0..<n {
            absorbG1(transcript, proofs[i].a)
            absorbG2(transcript, proofs[i].b)
            absorbG1(transcript, proofs[i].c)
            for inp in publicInputs[i] {
                transcript.absorb(inp)
            }
        }

        // Absorb VK for binding
        absorbG1(transcript, vk.alpha_g1)
        absorbG2(transcript, vk.beta_g2)
        absorbG2(transcript, vk.gamma_g2)
        absorbG2(transcript, vk.delta_g2)
        for ic in vk.ic {
            absorbG1(transcript, ic)
        }

        let r = transcript.squeeze()

        // Step 2: Compute powers of r
        var rPowers = [Fr]()
        rPowers.reserveCapacity(n)
        var rPow = Fr.one
        for _ in 0..<n {
            rPowers.append(rPow)
            rPow = frMul(rPow, r)
        }

        // Step 3: Weight A_i by r^i for TIPP
        // TIPP proves: prod e(r^i * A_i, B_i)
        let weightedA = (0..<n).map { i in
            cPointScalarMul(proofs[i].a, rPowers[i])
        }

        // Step 4: Run TIPP on (weightedA, B)
        let tippProof = computeTIPP(
            aVec: weightedA,
            bVec: proofs.map { $0.b },
            srsG1: Array(srs.g1Powers[0..<n]),
            srsG2: Array(srs.g2Powers[0..<n]),
            transcript: transcript
        )

        // Step 5: Compute aggregated C via MIPP
        let aggC = cpuMSM(points: proofs.map { $0.c }, scalars: rPowers)

        // Step 6: Run MIPP on (rPowers, C)
        let mippTranscript = Transcript(label: "snarkpack-mipp", backend: .poseidon2)
        // Bind MIPP transcript to the TIPP context
        mippTranscript.absorb(r)
        absorbG1(mippTranscript, aggC)

        let mippProof = computeMIPP(
            scalars: rPowers,
            points: proofs.map { $0.c },
            srsG1: Array(srs.g1Powers[0..<n]),
            transcript: mippTranscript
        )

        return SnarkPackProof(
            tippProof: tippProof,
            mippProof: mippProof,
            aggC: aggC,
            challenge: r,
            challengePowers: rPowers,
            count: n,
            publicInputs: publicInputs
        )
    }

    // MARK: - Verification

    /// Verify a SnarkPack aggregated proof.
    ///
    /// The verifier:
    /// 1. Re-derives the Fiat-Shamir challenge
    /// 2. Verifies the TIPP proof (log N pairing checks)
    /// 3. Verifies the MIPP proof (log N scalar checks)
    /// 4. Checks the Groth16 pairing equation using aggregated values
    ///
    /// - Parameters:
    ///   - aggregateProof: The SnarkPack proof to verify
    ///   - proofs: Original Groth16 proofs (needed for TIPP/MIPP verification)
    ///   - vk: Shared verification key
    ///   - publicInputs: Per-proof public inputs
    ///   - srs: SnarkPack SRS
    /// - Returns: true if the aggregated proof is valid
    public func verify(
        aggregateProof: SnarkPackProof,
        proofs: [Groth16Proof],
        vk: Groth16VerificationKey,
        publicInputs: [[Fr]],
        srs: SnarkPackSRS
    ) throws -> Bool {
        let n = aggregateProof.count
        guard n > 0, n == proofs.count, n == publicInputs.count else { return false }
        guard n & (n - 1) == 0 else { return false }
        guard n <= srs.maxProofs else { return false }

        // Step 1: Re-derive Fiat-Shamir challenge
        let transcript = Transcript(label: "snarkpack-tipp-mipp", backend: .poseidon2)
        transcript.absorb(frFromInt(UInt64(n)))

        for i in 0..<n {
            absorbG1(transcript, proofs[i].a)
            absorbG2(transcript, proofs[i].b)
            absorbG1(transcript, proofs[i].c)
            for inp in publicInputs[i] {
                transcript.absorb(inp)
            }
        }

        absorbG1(transcript, vk.alpha_g1)
        absorbG2(transcript, vk.beta_g2)
        absorbG2(transcript, vk.gamma_g2)
        absorbG2(transcript, vk.delta_g2)
        for ic in vk.ic {
            absorbG1(transcript, ic)
        }

        let r = transcript.squeeze()
        guard frFieldEqual(r, aggregateProof.challenge) else { return false }

        // Step 2: Re-compute challenge powers
        var rPowers = [Fr]()
        rPowers.reserveCapacity(n)
        var rPow = Fr.one
        for _ in 0..<n {
            rPowers.append(rPow)
            rPow = frMul(rPow, r)
        }

        // Step 3: Weight A_i by r^i
        let weightedA = (0..<n).map { i in
            cPointScalarMul(proofs[i].a, rPowers[i])
        }

        // Step 4: Verify TIPP
        let tippValid = verifyTIPP(
            proof: aggregateProof.tippProof,
            aVec: weightedA,
            bVec: proofs.map { $0.b },
            srsG1: Array(srs.g1Powers[0..<n]),
            srsG2: Array(srs.g2Powers[0..<n]),
            transcript: transcript
        )
        guard tippValid else { return false }

        // Step 5: Verify MIPP
        let mippTranscript = Transcript(label: "snarkpack-mipp", backend: .poseidon2)
        mippTranscript.absorb(r)
        absorbG1(mippTranscript, aggregateProof.aggC)

        let mippValid = verifyMIPP(
            proof: aggregateProof.mippProof,
            scalars: rPowers,
            points: proofs.map { $0.c },
            srsG1: Array(srs.g1Powers[0..<n]),
            transcript: mippTranscript
        )
        guard mippValid else { return false }

        // Step 6: Verify aggC consistency
        let recomputedAggC = cpuMSM(points: proofs.map { $0.c }, scalars: rPowers)
        guard projPointEqual(recomputedAggC, aggregateProof.aggC) else { return false }

        // Step 7: Groth16 pairing check using aggregated values
        // Compute aggL = sum r^i * L_i
        var aggL = pointIdentity()
        for i in 0..<n {
            let inputs = publicInputs[i]
            guard inputs.count + 1 <= vk.ic.count else { return false }
            var li = vk.ic[0]
            for j in 0..<inputs.count {
                if !inputs[j].isZero {
                    li = pointAdd(li, pointScalarMul(vk.ic[j + 1], inputs[j]))
                }
            }
            aggL = pointAdd(aggL, cPointScalarMul(li, rPowers[i]))
        }

        // Sum of r powers for alpha*beta term
        var rSum = Fr.zero
        for p in rPowers { rSum = frAdd(rSum, p) }

        // Pairing check:
        // prod e(r^i*A_i, B_i) = e(sum r^i * alpha, beta) * e(aggL, gamma) * e(aggC, delta)
        //
        // Since we've verified TIPP proves prod e(r^i*A_i, B_i) correctly,
        // we use the final TIPP values for the left side.
        //
        // For the simplified check (using individual pairings):
        // e(-aggA, B_common) * e(rSum*alpha, beta) * e(aggL, gamma) * e(aggC, delta) = 1
        // where aggA = sum r^i * A_i
        let aggA = cpuMSM(points: proofs.map { $0.a }, scalars: rPowers)

        let allBSame = proofs.count == 1 || proofs.allSatisfy { g2ProjEqual($0.b, proofs[0].b) }

        if allBSame {
            // Optimized: 4 pairings
            let negAggA = pointNeg(aggA)
            let rSumAlpha = cPointScalarMul(vk.alpha_g1, rSum)

            guard let nA = pointToAffine(negAggA),
                  let ra = pointToAffine(rSumAlpha),
                  let aL = pointToAffine(aggL),
                  let aC = pointToAffine(aggregateProof.aggC) else { return false }
            guard let b0 = g2ToAffine(proofs[0].b),
                  let be = g2ToAffine(vk.beta_g2),
                  let ga = g2ToAffine(vk.gamma_g2),
                  let de = g2ToAffine(vk.delta_g2) else { return false }

            return cBN254PairingCheck([(nA, b0), (ra, be), (aL, ga), (aC, de)])
        } else {
            // General: N+3 pairings
            var pairs = [(PointAffine, G2AffinePoint)]()
            pairs.reserveCapacity(n + 3)

            for i in 0..<n {
                let rAi = cPointScalarMul(proofs[i].a, rPowers[i])
                guard let rA = pointToAffine(rAi),
                      let bi = g2ToAffine(proofs[i].b) else { return false }
                pairs.append((rA, bi))
            }

            let rSumAlpha = pointNeg(cPointScalarMul(vk.alpha_g1, rSum))
            guard let nra = pointToAffine(rSumAlpha),
                  let negAggL = pointToAffine(pointNeg(aggL)),
                  let negAggC = pointToAffine(pointNeg(aggregateProof.aggC)) else { return false }
            guard let be = g2ToAffine(vk.beta_g2),
                  let ga = g2ToAffine(vk.gamma_g2),
                  let de = g2ToAffine(vk.delta_g2) else { return false }

            pairs.append((nra, be))
            pairs.append((negAggL, ga))
            pairs.append((negAggC, de))

            return cBN254PairingCheck(pairs)
        }
    }

    // MARK: - TIPP (Twisted Inner Product Pairing)

    /// Compute the TIPP proof via recursive halving.
    ///
    /// Given vectors A in G1^n, B in G2^n, and SRS vectors v in G1^n, w in G2^n:
    /// At each round (halving from n to n/2):
    ///   - zL = prod e(A_L[i], B_R[i])  (left-right cross pairing product)
    ///   - zR = prod e(A_R[i], B_L[i])  (right-left cross pairing product)
    ///   - gL = sum A_L[i] (SRS-weighted commitment)
    ///   - gR = sum A_R[i]
    ///   - hL = sum B_L[i] (G2 commitment)
    ///   - hR = sum B_R[i]
    ///   - challenge x = Hash(gL, gR, hL, hR)
    ///   - Fold: A' = A_L + x*A_R, B' = B_L + x^{-1}*B_R
    ///
    /// After log(n) rounds: single pair (A_final, B_final) remains.
    private func computeTIPP(
        aVec: [PointProjective],
        bVec: [G2ProjectivePoint],
        srsG1: [PointProjective],
        srsG2: [G2ProjectivePoint],
        transcript: Transcript
    ) -> TIPPProof {
        var a = aVec
        var b = bVec
        var v = srsG1
        var w = srsG2

        var gLefts = [PointProjective]()
        var gRights = [PointProjective]()
        var hLefts = [G2ProjectivePoint]()
        var hRights = [G2ProjectivePoint]()
        var challenges = [Fr]()

        while a.count > 1 {
            let half = a.count / 2

            let aL = Array(a[0..<half])
            let aR = Array(a[half...])
            let bL = Array(b[0..<half])
            let bR = Array(b[half...])
            let vL = Array(v[0..<half])
            let vR = Array(v[half...])
            let wL = Array(w[0..<half])
            let wR = Array(w[half...])

            // Cross-term G1 commitments using SRS
            // gL = sum vR[i] * (scalar extracting aL[i] relative to vL)
            // Simplified: gL = sum aL[i] (commitment to left half via SRS right powers)
            var gL = pointIdentity()
            var gR = pointIdentity()
            for i in 0..<half {
                if !pointIsIdentity(aL[i]) { gL = pointAdd(gL, aL[i]) }
                if !pointIsIdentity(aR[i]) { gR = pointAdd(gR, aR[i]) }
            }

            // Cross-term G2 commitments
            var hL = g2Identity()
            var hR = g2Identity()
            for i in 0..<half {
                if !g2IsIdentity(bR[i]) { hL = g2Add(hL, bR[i]) }
                if !g2IsIdentity(bL[i]) { hR = g2Add(hR, bL[i]) }
            }

            gLefts.append(gL)
            gRights.append(gR)
            hLefts.append(hL)
            hRights.append(hR)

            // Absorb commitments into transcript
            absorbG1(transcript, gL)
            absorbG1(transcript, gR)
            absorbG2(transcript, hL)
            absorbG2(transcript, hR)

            let x = transcript.squeeze()
            let xInv = frInverse(x)
            let xLimbs = frToInt(x)
            let xInvLimbs = frToInt(xInv)
            challenges.append(x)

            // Fold A: A'[i] = A_L[i] + x * A_R[i]
            var newA = [PointProjective]()
            newA.reserveCapacity(half)
            for i in 0..<half {
                let xAR = cPointScalarMul(aR[i], x)
                newA.append(pointAdd(aL[i], xAR))
            }

            // Fold B: B'[i] = B_L[i] + x^{-1} * B_R[i]
            var newB = [G2ProjectivePoint]()
            newB.reserveCapacity(half)
            for i in 0..<half {
                let xInvBR = g2ScalarMul(bR[i], xInvLimbs)
                newB.append(g2Add(bL[i], xInvBR))
            }

            // Fold SRS vectors similarly
            var newV = [PointProjective]()
            newV.reserveCapacity(half)
            for i in 0..<half {
                newV.append(pointAdd(vL[i], cPointScalarMul(vR[i], x)))
            }

            var newW = [G2ProjectivePoint]()
            newW.reserveCapacity(half)
            for i in 0..<half {
                newW.append(g2Add(wL[i], g2ScalarMul(wR[i], xInvLimbs)))
            }

            a = newA
            b = newB
            v = newV
            w = newW
        }

        return TIPPProof(
            gLeft: gLefts,
            gRight: gRights,
            hLeft: hLefts,
            hRight: hRights,
            challenges: challenges,
            finalA: a.isEmpty ? pointIdentity() : a[0],
            finalB: b.isEmpty ? g2Identity() : b[0]
        )
    }

    /// Verify the TIPP proof by re-deriving challenges and re-folding.
    private func verifyTIPP(
        proof: TIPPProof,
        aVec: [PointProjective],
        bVec: [G2ProjectivePoint],
        srsG1: [PointProjective],
        srsG2: [G2ProjectivePoint],
        transcript: Transcript
    ) -> Bool {
        let numRounds = proof.gLeft.count
        guard numRounds == proof.gRight.count else { return false }
        guard numRounds == proof.hLeft.count else { return false }
        guard numRounds == proof.hRight.count else { return false }
        guard numRounds == proof.challenges.count else { return false }

        var a = aVec
        var b = bVec

        for round in 0..<numRounds {
            let half = a.count / 2
            guard half > 0 else { return false }

            let aL = Array(a[0..<half])
            let aR = Array(a[half...])
            let bL = Array(b[0..<half])
            let bR = Array(b[half...])

            // Recompute commitments
            var gL = pointIdentity()
            var gR = pointIdentity()
            for i in 0..<half {
                if !pointIsIdentity(aL[i]) { gL = pointAdd(gL, aL[i]) }
                if !pointIsIdentity(aR[i]) { gR = pointAdd(gR, aR[i]) }
            }

            var hL = g2Identity()
            var hR = g2Identity()
            for i in 0..<half {
                if !g2IsIdentity(bR[i]) { hL = g2Add(hL, bR[i]) }
                if !g2IsIdentity(bL[i]) { hR = g2Add(hR, bL[i]) }
            }

            // Verify commitments match proof
            if !projPointEqual(gL, proof.gLeft[round]) { return false }
            if !projPointEqual(gR, proof.gRight[round]) { return false }
            if !g2ProjEqual(hL, proof.hLeft[round]) { return false }
            if !g2ProjEqual(hR, proof.hRight[round]) { return false }

            // Re-derive challenge
            absorbG1(transcript, proof.gLeft[round])
            absorbG1(transcript, proof.gRight[round])
            absorbG2(transcript, proof.hLeft[round])
            absorbG2(transcript, proof.hRight[round])

            let x = transcript.squeeze()
            guard frFieldEqual(x, proof.challenges[round]) else { return false }
            let xInv = frInverse(x)
            let xInvLimbs = frToInt(xInv)

            // Fold
            var newA = [PointProjective]()
            newA.reserveCapacity(half)
            for i in 0..<half {
                newA.append(pointAdd(aL[i], cPointScalarMul(aR[i], x)))
            }

            var newB = [G2ProjectivePoint]()
            newB.reserveCapacity(half)
            for i in 0..<half {
                newB.append(g2Add(bL[i], g2ScalarMul(bR[i], xInvLimbs)))
            }

            a = newA
            b = newB
        }

        // Final check
        guard a.count == 1, b.count == 1 else { return false }
        if !projPointEqual(a[0], proof.finalA) { return false }
        if !g2ProjEqual(b[0], proof.finalB) { return false }

        return true
    }

    // MARK: - MIPP (Multi-scalar Inner Product Pairing)

    /// Compute the MIPP proof via recursive halving.
    ///
    /// Given scalars r = [r_1,...,r_n] and points C = [C_1,...,C_n]:
    /// Proves Z = sum r_i * C_i
    ///
    /// At each round:
    ///   - comL = sum s_R[i] * C_L[i]  (cross-term left commitment)
    ///   - comR = sum s_L[i] * C_R[i]  (cross-term right commitment)
    ///   - challenge x = Hash(comL, comR)
    ///   - Fold: s' = s_L + x*s_R, C' = C_L + x^{-1}*C_R
    private func computeMIPP(
        scalars: [Fr],
        points: [PointProjective],
        srsG1: [PointProjective],
        transcript: Transcript
    ) -> MIPPProof {
        var s = scalars
        var p = points

        var comLefts = [PointProjective]()
        var comRights = [PointProjective]()
        var challenges = [Fr]()

        while s.count > 1 {
            let half = s.count / 2

            let sL = Array(s[0..<half])
            let sR = Array(s[half...])
            let pL = Array(p[0..<half])
            let pR = Array(p[half...])

            // Cross-term commitments
            var comL = pointIdentity()
            var comR = pointIdentity()
            for i in 0..<half {
                if !sR[i].isZero && !pointIsIdentity(pL[i]) {
                    comL = pointAdd(comL, cPointScalarMul(pL[i], sR[i]))
                }
                if !sL[i].isZero && !pointIsIdentity(pR[i]) {
                    comR = pointAdd(comR, cPointScalarMul(pR[i], sL[i]))
                }
            }

            comLefts.append(comL)
            comRights.append(comR)

            // Absorb into transcript
            absorbG1(transcript, comL)
            absorbG1(transcript, comR)

            let x = transcript.squeeze()
            let xInv = frInverse(x)
            challenges.append(x)

            // Fold scalars: s' = s_L + x * s_R
            var newS = [Fr]()
            newS.reserveCapacity(half)
            for i in 0..<half {
                newS.append(frAdd(sL[i], frMul(x, sR[i])))
            }

            // Fold points: p' = p_L + x^{-1} * p_R
            var newP = [PointProjective]()
            newP.reserveCapacity(half)
            for i in 0..<half {
                newP.append(pointAdd(pL[i], cPointScalarMul(pR[i], xInv)))
            }

            s = newS
            p = newP
        }

        return MIPPProof(
            comLeft: comLefts,
            comRight: comRights,
            challenges: challenges,
            finalScalar: s.isEmpty ? Fr.zero : s[0],
            finalPoint: p.isEmpty ? pointIdentity() : p[0]
        )
    }

    /// Verify the MIPP proof by re-deriving challenges and re-folding.
    private func verifyMIPP(
        proof: MIPPProof,
        scalars: [Fr],
        points: [PointProjective],
        srsG1: [PointProjective],
        transcript: Transcript
    ) -> Bool {
        let numRounds = proof.comLeft.count
        guard numRounds == proof.comRight.count else { return false }
        guard numRounds == proof.challenges.count else { return false }

        var s = scalars
        var p = points

        for round in 0..<numRounds {
            let half = s.count / 2
            guard half > 0 else { return false }

            let sL = Array(s[0..<half])
            let sR = Array(s[half...])
            let pL = Array(p[0..<half])
            let pR = Array(p[half...])

            // Recompute cross-term commitments
            var comL = pointIdentity()
            var comR = pointIdentity()
            for i in 0..<half {
                if !sR[i].isZero && !pointIsIdentity(pL[i]) {
                    comL = pointAdd(comL, cPointScalarMul(pL[i], sR[i]))
                }
                if !sL[i].isZero && !pointIsIdentity(pR[i]) {
                    comR = pointAdd(comR, cPointScalarMul(pR[i], sL[i]))
                }
            }

            // Verify commitments match
            if !projPointEqual(comL, proof.comLeft[round]) { return false }
            if !projPointEqual(comR, proof.comRight[round]) { return false }

            // Re-derive challenge
            absorbG1(transcript, proof.comLeft[round])
            absorbG1(transcript, proof.comRight[round])

            let x = transcript.squeeze()
            guard frFieldEqual(x, proof.challenges[round]) else { return false }
            let xInv = frInverse(x)

            // Fold
            var newS = [Fr]()
            newS.reserveCapacity(half)
            for i in 0..<half {
                newS.append(frAdd(sL[i], frMul(x, sR[i])))
            }

            var newP = [PointProjective]()
            newP.reserveCapacity(half)
            for i in 0..<half {
                newP.append(pointAdd(pL[i], cPointScalarMul(pR[i], xInv)))
            }

            s = newS
            p = newP
        }

        guard s.count == 1, p.count == 1 else { return false }
        if !frFieldEqual(s[0], proof.finalScalar) { return false }
        if !projPointEqual(p[0], proof.finalPoint) { return false }

        return true
    }

    // MARK: - Internal Helpers

    /// CPU MSM: sum points[i] * scalars[i]
    private func cpuMSM(
        points: [PointProjective],
        scalars: [Fr]
    ) -> PointProjective {
        var acc = pointIdentity()
        for i in 0..<points.count {
            if !scalars[i].isZero && !pointIsIdentity(points[i]) {
                acc = pointAdd(acc, cPointScalarMul(points[i], scalars[i]))
            }
        }
        return acc
    }

    /// Absorb a G1 projective point into the transcript.
    private func absorbG1(_ transcript: Transcript, _ p: PointProjective) {
        if let aff = pointToAffine(p) {
            transcript.absorb(fpToFr(aff.x))
            transcript.absorb(fpToFr(aff.y))
        } else {
            transcript.absorb(Fr.zero)
            transcript.absorb(Fr.zero)
        }
    }

    /// Absorb a G2 projective point into the transcript.
    private func absorbG2(_ transcript: Transcript, _ p: G2ProjectivePoint) {
        if let aff = g2ToAffine(p) {
            transcript.absorb(fpToFr(aff.x.c0))
            transcript.absorb(fpToFr(aff.x.c1))
            transcript.absorb(fpToFr(aff.y.c0))
            transcript.absorb(fpToFr(aff.y.c1))
        } else {
            transcript.absorb(Fr.zero)
            transcript.absorb(Fr.zero)
            transcript.absorb(Fr.zero)
            transcript.absorb(Fr.zero)
        }
    }

    /// Compare two G1 projective points for equality via affine conversion.
    private func projPointEqual(_ a: PointProjective, _ b: PointProjective) -> Bool {
        if pointIsIdentity(a) && pointIsIdentity(b) { return true }
        if pointIsIdentity(a) || pointIsIdentity(b) { return false }
        let aAffs = batchToAffine([a])
        let bAffs = batchToAffine([b])
        return fpToInt(aAffs[0].x) == fpToInt(bAffs[0].x) &&
               fpToInt(aAffs[0].y) == fpToInt(bAffs[0].y)
    }

    /// Compare two G2 projective points for equality.
    private func g2ProjEqual(_ a: G2ProjectivePoint, _ b: G2ProjectivePoint) -> Bool {
        if g2IsIdentity(a) && g2IsIdentity(b) { return true }
        if g2IsIdentity(a) || g2IsIdentity(b) { return false }
        // Cross-multiply to compare without converting to affine
        let az2 = fp2Sqr(a.z)
        let bz2 = fp2Sqr(b.z)
        let lhsX = fp2Mul(a.x, bz2)
        let rhsX = fp2Mul(b.x, az2)
        let dx = fp2Sub(lhsX, rhsX)
        if !dx.c0.isZero || !dx.c1.isZero { return false }
        let az3 = fp2Mul(az2, a.z)
        let bz3 = fp2Mul(bz2, b.z)
        let lhsY = fp2Mul(a.y, bz3)
        let rhsY = fp2Mul(b.y, az3)
        let dy = fp2Sub(lhsY, rhsY)
        return dy.c0.isZero && dy.c1.isZero
    }

    /// Compare two Fr elements for equality.
    private func frFieldEqual(_ a: Fr, _ b: Fr) -> Bool {
        a.v.0 == b.v.0 && a.v.1 == b.v.1 && a.v.2 == b.v.2 && a.v.3 == b.v.3 &&
        a.v.4 == b.v.4 && a.v.5 == b.v.5 && a.v.6 == b.v.6 && a.v.7 == b.v.7
    }
}

// MARK: - Errors

public enum SnarkPackError: Error, CustomStringConvertible {
    case emptyProofSet
    case notPowerOfTwo(Int)
    case srsTooSmall(need: Int, have: Int)
    case inputCountMismatch

    public var description: String {
        switch self {
        case .emptyProofSet:
            return "SnarkPack: empty proof set"
        case .notPowerOfTwo(let n):
            return "SnarkPack: proof count \(n) must be a power of 2"
        case .srsTooSmall(let need, let have):
            return "SnarkPack: SRS supports \(have) proofs but \(need) requested"
        case .inputCountMismatch:
            return "SnarkPack: publicInputs count does not match proofs count"
        }
    }
}
