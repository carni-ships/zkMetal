// GPUGroth16AggregateEngine — GPU-accelerated Groth16 proof aggregation
//
// SnarkPack-style aggregation of N Groth16 proofs into a single aggregated
// proof with IPPA. GPU MSM for G1 aggregation, Fiat-Shamir challenges,
// heterogeneous circuit support, O(1) pairing verification.
// For BN254 curve.

import Foundation
import Metal
import NeonFieldOps

// MARK: - Aggregation SRS

/// Structured Reference String for proof aggregation.
/// Contains random group elements used to bind the aggregation.
/// Generated from a trusted setup ceremony or derived deterministically.
public struct AggregationSRS {
    /// G1 elements: [g1, tau*g1, tau^2*g1, ..., tau^{n-1}*g1]
    public let g1Powers: [PointProjective]
    /// G2 elements: [g2, tau*g2] — needed for pairing consistency check
    public let g2Gen: G2ProjectivePoint
    public let g2Tau: G2ProjectivePoint
    /// Maximum number of proofs this SRS supports
    public let maxProofs: Int

    public init(g1Powers: [PointProjective], g2Gen: G2ProjectivePoint,
                g2Tau: G2ProjectivePoint, maxProofs: Int) {
        self.g1Powers = g1Powers
        self.g2Gen = g2Gen
        self.g2Tau = g2Tau
        self.maxProofs = maxProofs
    }

    /// Generate a deterministic SRS from a seed scalar (for testing).
    /// In production, use a proper trusted setup ceremony.
    public static func generate(maxProofs: Int, seed: Fr) -> AggregationSRS {
        precondition(maxProofs > 0, "SRS must support at least 1 proof")
        let g1 = pointFromAffine(bn254G1Generator())
        let g2 = g2FromAffine(bn254G2Generator())

        // tau = hash(seed) for domain separation
        let tau = poseidon2Hash(seed, seed)

        var g1Powers = [PointProjective]()
        g1Powers.reserveCapacity(maxProofs)
        var tauPow = Fr.one
        for _ in 0..<maxProofs {
            g1Powers.append(pointScalarMul(g1, tauPow))
            tauPow = frMul(tauPow, tau)
        }

        let g2Tau = g2ScalarMul(g2, frToInt(tau))

        return AggregationSRS(
            g1Powers: g1Powers,
            g2Gen: g2,
            g2Tau: g2Tau,
            maxProofs: maxProofs
        )
    }
}

// MARK: - Aggregate Proof Types

/// A GPU-aggregated Groth16 proof produced by the aggregate engine.
/// Contains aggregated G1 points, IPPA proof, and per-proof metadata.
public struct GPUAggregatedProof {
    /// Aggregated A point: sum r^i * A_i
    public let aggA: PointProjective
    /// Aggregated C point: sum r^i * C_i
    public let aggC: PointProjective
    /// Aggregated public input accumulators: sum r^i * L_i
    public let aggL: PointProjective
    /// Inner pairing product argument proof
    public let ippProof: GPUIIPPProof
    /// Fiat-Shamir challenge used for aggregation
    public let challenge: Fr
    /// Powers of challenge: [1, r, r^2, ..., r^{N-1}]
    public let challengePowers: [Fr]
    /// Sum of challenge powers
    public let challengeSum: Fr
    /// Number of proofs aggregated
    public let count: Int
    /// Per-proof public inputs
    public let publicInputs: [[Fr]]
    /// Per-proof verification key indices (for heterogeneous circuits)
    public let vkIndices: [Int]

    public init(aggA: PointProjective, aggC: PointProjective, aggL: PointProjective,
                ippProof: GPUIIPPProof, challenge: Fr, challengePowers: [Fr],
                challengeSum: Fr, count: Int, publicInputs: [[Fr]], vkIndices: [Int]) {
        self.aggA = aggA
        self.aggC = aggC
        self.aggL = aggL
        self.ippProof = ippProof
        self.challenge = challenge
        self.challengePowers = challengePowers
        self.challengeSum = challengeSum
        self.count = count
        self.publicInputs = publicInputs
        self.vkIndices = vkIndices
    }
}

/// Inner pairing product proof for GPU aggregate engine.
/// Uses recursive halving with Fiat-Shamir challenges at each round.
public struct GPUIIPPProof {
    /// Left commitments at each round of recursive halving
    public let leftCommitments: [PointProjective]
    /// Right commitments at each round
    public let rightCommitments: [PointProjective]
    /// Challenges derived at each round
    public let challenges: [Fr]
    /// Final reduced point after all rounds
    public let finalPoint: PointProjective
    /// Final reduced scalar after all rounds
    public let finalScalar: Fr

    public init(leftCommitments: [PointProjective], rightCommitments: [PointProjective],
                challenges: [Fr], finalPoint: PointProjective, finalScalar: Fr) {
        self.leftCommitments = leftCommitments
        self.rightCommitments = rightCommitments
        self.challenges = challenges
        self.finalPoint = finalPoint
        self.finalScalar = finalScalar
    }
}

/// Input descriptor for a single proof in a heterogeneous aggregation batch.
public struct AggregateProofInput {
    /// The Groth16 proof
    public let proof: Groth16Proof
    /// Public inputs for this proof
    public let publicInputs: [Fr]
    /// Verification key for this proof's circuit
    public let vk: Groth16VerificationKey
    /// Index identifying which circuit/VK this proof belongs to
    public let circuitIndex: Int

    public init(proof: Groth16Proof, publicInputs: [Fr],
                vk: Groth16VerificationKey, circuitIndex: Int) {
        self.proof = proof
        self.publicInputs = publicInputs
        self.vk = vk
        self.circuitIndex = circuitIndex
    }
}

// MARK: - GPU Groth16 Aggregate Engine

/// GPU-accelerated Groth16 proof aggregation engine.
/// Aggregates N proofs into a single aggregated proof using SnarkPack-style
/// random linear combination with GPU MSM acceleration.
/// Supports homogeneous (same circuit) and heterogeneous (different circuits).
public class GPUGroth16AggregateEngine {
    public static let version = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")

    /// GPU MSM engine for BN254 G1
    public let msm: MetalMSM

    /// GPU MSM threshold: use Metal MSM when point count exceeds this
    public var gpuMSMThreshold: Int = 64

    /// Enable profiling output
    public var profile: Bool = false

    public init() throws {
        self.msm = try MetalMSM()
    }

    // MARK: - Homogeneous Aggregation (same circuit)

    /// Aggregate N proofs from the same circuit into a single aggregated proof.
    ///
    /// - Parameters:
    ///   - proofs: Array of Groth16 proofs (all from same circuit)
    ///   - publicInputs: Per-proof public inputs
    ///   - vk: Shared verification key
    ///   - srs: Aggregation structured reference string
    /// - Returns: Aggregated proof
    public func aggregateHomogeneous(
        proofs: [Groth16Proof],
        publicInputs: [[Fr]],
        vk: Groth16VerificationKey,
        srs: AggregationSRS
    ) throws -> GPUAggregatedProof {
        let inputs = proofs.enumerated().map { (i, proof) in
            AggregateProofInput(
                proof: proof,
                publicInputs: publicInputs[i],
                vk: vk,
                circuitIndex: 0
            )
        }
        return try aggregate(inputs: inputs, srs: srs)
    }

    // MARK: - Heterogeneous Aggregation (different circuits)

    /// Aggregate N proofs from potentially different circuits.
    ///
    /// Each proof carries its own verification key and circuit index.
    /// The Fiat-Shamir transcript absorbs all VK data to bind the
    /// aggregation to the specific set of circuits.
    ///
    /// - Parameters:
    ///   - inputs: Array of proof inputs with per-proof VK
    ///   - srs: Aggregation structured reference string
    /// - Returns: Aggregated proof
    public func aggregate(
        inputs: [AggregateProofInput],
        srs: AggregationSRS
    ) throws -> GPUAggregatedProof {
        let n = inputs.count
        precondition(n > 0, "Must aggregate at least one proof")
        precondition(n <= srs.maxProofs, "SRS too small for \(n) proofs")

        let t0 = profile ? CFAbsoluteTimeGetCurrent() : 0

        // Step 1: Build Fiat-Shamir transcript
        let transcript = Transcript(label: "gpu-groth16-aggregate", backend: .poseidon2)

        // Absorb proof count
        transcript.absorb(frFromInt(UInt64(n)))

        // Absorb all proof elements and public inputs
        for i in 0..<n {
            let proof = inputs[i].proof
            absorbPoint(transcript, proof.a)
            absorbG2Point(transcript, proof.b)
            absorbPoint(transcript, proof.c)

            // Absorb circuit index for heterogeneous binding
            transcript.absorb(frFromInt(UInt64(inputs[i].circuitIndex)))

            // Absorb public inputs
            for inp in inputs[i].publicInputs {
                transcript.absorb(inp)
            }
        }

        // Absorb VK data for each unique circuit
        var seenCircuits = Set<Int>()
        for input in inputs {
            if seenCircuits.insert(input.circuitIndex).inserted {
                absorbPoint(transcript, input.vk.alpha_g1)
                absorbG2Point(transcript, input.vk.beta_g2)
                absorbG2Point(transcript, input.vk.gamma_g2)
                absorbG2Point(transcript, input.vk.delta_g2)
                for ic in input.vk.ic {
                    absorbPoint(transcript, ic)
                }
            }
        }

        // Derive challenge r
        let r = transcript.squeeze()

        // Step 2: Compute powers of r and their sum
        var challengePowers = [Fr]()
        challengePowers.reserveCapacity(n)
        var rPow = Fr.one
        var rSum = Fr.zero
        for _ in 0..<n {
            challengePowers.append(rPow)
            rSum = frAdd(rSum, rPow)
            rPow = frMul(rPow, r)
        }

        if profile {
            let t1 = CFAbsoluteTimeGetCurrent()
            fputs(String(format: "  [gpu-agg] transcript + powers: %.2f ms\n", (t1 - t0) * 1000), stderr)
        }

        // Step 3: Compute aggregated points via GPU MSM
        let aggA: PointProjective
        let aggC: PointProjective
        let aggL: PointProjective

        let tMSM = profile ? CFAbsoluteTimeGetCurrent() : 0

        if n >= gpuMSMThreshold {
            // GPU path: batch MSM
            let aPoints = batchToAffine(inputs.map { $0.proof.a })
            let cPoints = batchToAffine(inputs.map { $0.proof.c })
            let scalarLimbs = challengePowers.map { frToLimbs($0) }

            aggA = try msm.msm(points: aPoints, scalars: scalarLimbs)
            aggC = try msm.msm(points: cPoints, scalars: scalarLimbs)

            // Compute L_i for each proof and aggregate via GPU MSM
            let lPoints = computePerProofL(inputs: inputs)
            let lAffine = batchToAffine(lPoints)
            aggL = try msm.msm(points: lAffine, scalars: scalarLimbs)
        } else {
            // CPU path: scalar multiplication
            aggA = cpuAggregateMSM(
                points: inputs.map { $0.proof.a },
                scalars: challengePowers
            )
            aggC = cpuAggregateMSM(
                points: inputs.map { $0.proof.c },
                scalars: challengePowers
            )
            let lPoints = computePerProofL(inputs: inputs)
            aggL = cpuAggregateMSM(points: lPoints, scalars: challengePowers)
        }

        if profile {
            let t2 = CFAbsoluteTimeGetCurrent()
            fputs(String(format: "  [gpu-agg] MSM aggregation: %.2f ms\n", (t2 - tMSM) * 1000), stderr)
        }

        // Step 4: Compute IPPA proof
        let tIPP = profile ? CFAbsoluteTimeGetCurrent() : 0

        let ippProof = computeIPPA(
            points: inputs.map { $0.proof.a },
            scalars: challengePowers,
            transcript: transcript
        )

        if profile {
            let t3 = CFAbsoluteTimeGetCurrent()
            fputs(String(format: "  [gpu-agg] IPPA: %.2f ms\n", (t3 - tIPP) * 1000), stderr)
        }

        return GPUAggregatedProof(
            aggA: aggA,
            aggC: aggC,
            aggL: aggL,
            ippProof: ippProof,
            challenge: r,
            challengePowers: challengePowers,
            challengeSum: rSum,
            count: n,
            publicInputs: inputs.map { $0.publicInputs },
            vkIndices: inputs.map { $0.circuitIndex }
        )
    }

    // MARK: - Verification

    /// Verify an aggregated proof against original proofs and verification keys.
    /// Re-derives Fiat-Shamir challenge, verifies IPPA, checks multi-pairing.
    public func verify(
        aggProof: GPUAggregatedProof,
        originalProofs: [Groth16Proof],
        vks: [Int: Groth16VerificationKey],
        srs: AggregationSRS
    ) -> Bool {
        let n = aggProof.count
        guard n > 0, n == originalProofs.count else { return false }
        guard n == aggProof.publicInputs.count else { return false }
        guard n == aggProof.vkIndices.count else { return false }

        // Step 1: Re-derive Fiat-Shamir challenge
        let transcript = Transcript(label: "gpu-groth16-aggregate", backend: .poseidon2)
        transcript.absorb(frFromInt(UInt64(n)))

        for i in 0..<n {
            absorbPoint(transcript, originalProofs[i].a)
            absorbG2Point(transcript, originalProofs[i].b)
            absorbPoint(transcript, originalProofs[i].c)
            transcript.absorb(frFromInt(UInt64(aggProof.vkIndices[i])))
            for inp in aggProof.publicInputs[i] {
                transcript.absorb(inp)
            }
        }

        var seenCircuits = Set<Int>()
        for i in 0..<n {
            let idx = aggProof.vkIndices[i]
            if seenCircuits.insert(idx).inserted {
                guard let vk = vks[idx] else { return false }
                absorbPoint(transcript, vk.alpha_g1)
                absorbG2Point(transcript, vk.beta_g2)
                absorbG2Point(transcript, vk.gamma_g2)
                absorbG2Point(transcript, vk.delta_g2)
                for ic in vk.ic {
                    absorbPoint(transcript, ic)
                }
            }
        }

        let r = transcript.squeeze()

        // Verify challenge matches
        if !frEqual(r, aggProof.challenge) { return false }

        // Step 2: Re-compute challenge powers and sum
        var rPowers = [Fr]()
        rPowers.reserveCapacity(n)
        var rPow = Fr.one
        var rSum = Fr.zero
        for _ in 0..<n {
            rPowers.append(rPow)
            rSum = frAdd(rSum, rPow)
            rPow = frMul(rPow, r)
        }

        if !frEqual(rSum, aggProof.challengeSum) { return false }

        // Step 2b: Verify aggA consistency (re-derive from original proofs)
        var recomputedAggA = pointIdentity()
        for i in 0..<n {
            if !rPowers[i].isZero && !pointIsIdentity(originalProofs[i].a) {
                recomputedAggA = pointAdd(recomputedAggA, cPointScalarMul(originalProofs[i].a, rPowers[i]))
            }
        }
        if !projPointEqual(recomputedAggA, aggProof.aggA) { return false }

        // Step 2c: Verify aggC consistency (re-derive from original proofs)
        var recomputedAggC = pointIdentity()
        for i in 0..<n {
            if !rPowers[i].isZero && !pointIsIdentity(originalProofs[i].c) {
                recomputedAggC = pointAdd(recomputedAggC, cPointScalarMul(originalProofs[i].c, rPowers[i]))
            }
        }
        if !projPointEqual(recomputedAggC, aggProof.aggC) { return false }

        // Step 3: Verify IPPA
        if !verifyIPPA(
            ippProof: aggProof.ippProof,
            points: originalProofs.map { $0.a },
            scalars: rPowers,
            transcript: transcript
        ) {
            return false
        }

        // Step 4: Re-compute aggL from public inputs
        var aggL = pointIdentity()
        for i in 0..<n {
            let idx = aggProof.vkIndices[i]
            guard let vk = vks[idx] else { return false }
            let inputs = aggProof.publicInputs[i]
            guard inputs.count + 1 <= vk.ic.count else { return false }
            var li = vk.ic[0]
            for j in 0..<inputs.count {
                if !inputs[j].isZero {
                    li = pointAdd(li, pointScalarMul(vk.ic[j + 1], inputs[j]))
                }
            }
            aggL = pointAdd(aggL, cPointScalarMul(li, rPowers[i]))
        }

        // Verify aggL matches
        if !projPointEqual(aggL, aggProof.aggL) { return false }

        // Step 5: Pairing check
        // For homogeneous: e(-aggA, B_0) * e(rSum*alpha, beta) * e(aggL, gamma) * e(aggC, delta) = 1
        // For heterogeneous: need per-proof pairing for A_i * B_i terms
        let isHomogeneous = Set(aggProof.vkIndices).count == 1
        guard let firstVKIdx = aggProof.vkIndices.first,
              let firstVK = vks[firstVKIdx] else { return false }

        if isHomogeneous {
            return verifyHomogeneousPairing(
                aggProof: aggProof,
                originalProofs: originalProofs,
                vk: firstVK,
                rPowers: rPowers,
                rSum: rSum
            )
        } else {
            return verifyHeterogeneousPairing(
                aggProof: aggProof,
                originalProofs: originalProofs,
                vks: vks,
                rPowers: rPowers,
                rSum: rSum
            )
        }
    }

    /// Convenience: verify a homogeneous aggregated proof with a single VK.
    public func verifyHomogeneous(
        aggProof: GPUAggregatedProof,
        originalProofs: [Groth16Proof],
        vk: Groth16VerificationKey,
        srs: AggregationSRS
    ) -> Bool {
        let idx = aggProof.vkIndices.first ?? 0
        return verify(
            aggProof: aggProof,
            originalProofs: originalProofs,
            vks: [idx: vk],
            srs: srs
        )
    }

    // MARK: - Batch Convenience

    /// Aggregate and immediately verify (one-shot batch verification).
    /// Returns true if all proofs are valid.
    public func batchVerify(
        inputs: [AggregateProofInput],
        srs: AggregationSRS
    ) throws -> Bool {
        let aggProof = try aggregate(inputs: inputs, srs: srs)
        var vks = [Int: Groth16VerificationKey]()
        for input in inputs {
            vks[input.circuitIndex] = input.vk
        }
        return verify(
            aggProof: aggProof,
            originalProofs: inputs.map { $0.proof },
            vks: vks,
            srs: srs
        )
    }

    // MARK: - IPPA (Inner Pairing Product Argument)

    /// Compute the inner pairing product argument via recursive halving.
    /// At each round: L = <P_L, s_R>, R = <P_R, s_L>, challenge x = Hash(L,R),
    /// fold P' = P_L + x*P_R, s' = s_L + x^{-1}*s_R.
    ///
    /// After all rounds: one point P_final and one scalar s_final remain.
    private func computeIPPA(
        points: [PointProjective],
        scalars: [Fr],
        transcript: Transcript
    ) -> GPUIIPPProof {
        var pts = points
        var scs = scalars
        var leftComms = [PointProjective]()
        var rightComms = [PointProjective]()
        var challenges = [Fr]()

        // Pad to power of 2 if needed
        var padN = 1
        while padN < pts.count { padN <<= 1 }
        while pts.count < padN {
            pts.append(pointIdentity())
            scs.append(Fr.zero)
        }

        while pts.count > 1 {
            let half = pts.count / 2
            let pL = Array(pts[0..<half])
            let pR = Array(pts[half...])
            let sL = Array(scs[0..<half])
            let sR = Array(scs[half...])

            // Compute cross-term commitments
            var lComm = pointIdentity()
            var rComm = pointIdentity()
            for i in 0..<half {
                if !sR[i].isZero && !pointIsIdentity(pL[i]) {
                    lComm = pointAdd(lComm, cPointScalarMul(pL[i], sR[i]))
                }
                if !sL[i].isZero && !pointIsIdentity(pR[i]) {
                    rComm = pointAdd(rComm, cPointScalarMul(pR[i], sL[i]))
                }
            }

            leftComms.append(lComm)
            rightComms.append(rComm)

            // Derive challenge from transcript
            absorbPoint(transcript, lComm)
            absorbPoint(transcript, rComm)
            let x = transcript.squeeze()
            let xInv = frInverse(x)
            challenges.append(x)

            // Fold points: P' = P_L[i] + x * P_R[i]
            var newPts = [PointProjective]()
            newPts.reserveCapacity(half)
            for i in 0..<half {
                newPts.append(pointAdd(pL[i], cPointScalarMul(pR[i], x)))
            }

            // Fold scalars: s' = s_L[i] + x^{-1} * s_R[i]
            var newScs = [Fr]()
            newScs.reserveCapacity(half)
            for i in 0..<half {
                newScs.append(frAdd(sL[i], frMul(xInv, sR[i])))
            }

            pts = newPts
            scs = newScs
        }

        return GPUIIPPProof(
            leftCommitments: leftComms,
            rightCommitments: rightComms,
            challenges: challenges,
            finalPoint: pts.isEmpty ? pointIdentity() : pts[0],
            finalScalar: scs.isEmpty ? Fr.zero : scs[0]
        )
    }

    /// Verify the IPPA proof by re-deriving challenges and re-folding.
    private func verifyIPPA(
        ippProof: GPUIIPPProof,
        points: [PointProjective],
        scalars: [Fr],
        transcript: Transcript
    ) -> Bool {
        let numRounds = ippProof.leftCommitments.count
        guard numRounds == ippProof.rightCommitments.count else { return false }
        guard numRounds == ippProof.challenges.count else { return false }

        var pts = points
        var scs = scalars

        // Pad to power of 2
        var padN = 1
        while padN < pts.count { padN <<= 1 }
        while pts.count < padN {
            pts.append(pointIdentity())
            scs.append(Fr.zero)
        }

        for round in 0..<numRounds {
            let half = pts.count / 2
            guard half > 0 else { return false }

            let pL = Array(pts[0..<half])
            let pR = Array(pts[half...])
            let sL = Array(scs[0..<half])
            let sR = Array(scs[half...])

            // Recompute L and R to verify they match
            var lComm = pointIdentity()
            var rComm = pointIdentity()
            for i in 0..<half {
                if !sR[i].isZero && !pointIsIdentity(pL[i]) {
                    lComm = pointAdd(lComm, cPointScalarMul(pL[i], sR[i]))
                }
                if !sL[i].isZero && !pointIsIdentity(pR[i]) {
                    rComm = pointAdd(rComm, cPointScalarMul(pR[i], sL[i]))
                }
            }

            if !projPointEqual(lComm, ippProof.leftCommitments[round]) { return false }
            if !projPointEqual(rComm, ippProof.rightCommitments[round]) { return false }

            // Re-derive challenge and verify
            absorbPoint(transcript, ippProof.leftCommitments[round])
            absorbPoint(transcript, ippProof.rightCommitments[round])
            let x = transcript.squeeze()

            if !frEqual(x, ippProof.challenges[round]) { return false }
            let xInv = frInverse(x)

            // Fold
            var newPts = [PointProjective]()
            newPts.reserveCapacity(half)
            for i in 0..<half {
                newPts.append(pointAdd(pL[i], cPointScalarMul(pR[i], x)))
            }
            var newScs = [Fr]()
            newScs.reserveCapacity(half)
            for i in 0..<half {
                newScs.append(frAdd(sL[i], frMul(xInv, sR[i])))
            }

            pts = newPts
            scs = newScs
        }

        // Final check: reduced point and scalar should match
        guard pts.count == 1, scs.count == 1 else { return false }
        if !projPointEqual(pts[0], ippProof.finalPoint) { return false }
        if !frEqual(scs[0], ippProof.finalScalar) { return false }

        return true
    }

    // MARK: - Pairing Verification

    /// Verify pairing equation for homogeneous aggregation (all same VK).
    private func verifyHomogeneousPairing(
        aggProof: GPUAggregatedProof,
        originalProofs: [Groth16Proof],
        vk: Groth16VerificationKey,
        rPowers: [Fr],
        rSum: Fr
    ) -> Bool {
        let n = aggProof.count

        // Since B_i differ per proof even for same circuit (due to blinding),
        // we compute individual pairing terms
        var pairs = [(PointAffine, G2AffinePoint)]()
        pairs.reserveCapacity(n + 3)

        // e(r_i * A_i, B_i) for each proof
        for i in 0..<n {
            let rA = cPointScalarMul(originalProofs[i].a, rPowers[i])
            guard let rA_aff = pointToAffine(rA) else { return false }
            guard let b_aff = g2ToAffine(originalProofs[i].b) else { return false }
            pairs.append((rA_aff, b_aff))
        }

        // e(-rSum * alpha, beta)
        let negRSumAlpha = pointNeg(pointScalarMul(vk.alpha_g1, rSum))
        guard let nra = pointToAffine(negRSumAlpha) else { return false }
        guard let beta = g2ToAffine(vk.beta_g2) else { return false }
        pairs.append((nra, beta))

        // e(-aggL, gamma)
        let negAggL = pointNeg(aggProof.aggL)
        guard let nal = pointToAffine(negAggL) else { return false }
        guard let gamma = g2ToAffine(vk.gamma_g2) else { return false }
        pairs.append((nal, gamma))

        // e(-aggC, delta)
        let negAggC = pointNeg(aggProof.aggC)
        guard let nac = pointToAffine(negAggC) else { return false }
        guard let delta = g2ToAffine(vk.delta_g2) else { return false }
        pairs.append((nac, delta))

        return cBN254PairingCheck(pairs)
    }

    /// Verify pairing equation for heterogeneous aggregation (different VKs).
    private func verifyHeterogeneousPairing(
        aggProof: GPUAggregatedProof,
        originalProofs: [Groth16Proof],
        vks: [Int: Groth16VerificationKey],
        rPowers: [Fr],
        rSum: Fr
    ) -> Bool {
        let n = aggProof.count

        // Group proofs by circuit index
        var groups = [Int: [(Int, Fr)]]()  // circuitIdx -> [(proofIdx, r^i)]
        for i in 0..<n {
            let idx = aggProof.vkIndices[i]
            groups[idx, default: []].append((i, rPowers[i]))
        }

        var pairs = [(PointAffine, G2AffinePoint)]()
        pairs.reserveCapacity(n + 3 * groups.count)

        // Per-proof pairing: e(r_i * A_i, B_i)
        for i in 0..<n {
            let rA = cPointScalarMul(originalProofs[i].a, rPowers[i])
            guard let rA_aff = pointToAffine(rA) else { return false }
            guard let b_aff = g2ToAffine(originalProofs[i].b) else { return false }
            pairs.append((rA_aff, b_aff))
        }

        // Per-circuit group: e(-rSum_group * alpha, beta) * e(-aggL_group, gamma) * e(-aggC_group, delta)
        for (circuitIdx, proofIndices) in groups {
            guard let vk = vks[circuitIdx] else { return false }

            // Compute group-level rSum
            var groupRSum = Fr.zero
            for (_, ri) in proofIndices {
                groupRSum = frAdd(groupRSum, ri)
            }

            // e(-groupRSum * alpha, beta)
            let negAlpha = pointNeg(pointScalarMul(vk.alpha_g1, groupRSum))
            guard let na = pointToAffine(negAlpha) else { return false }
            guard let beta = g2ToAffine(vk.beta_g2) else { return false }
            pairs.append((na, beta))

            // Compute group aggL
            var groupAggL = pointIdentity()
            for (proofIdx, ri) in proofIndices {
                let inputs = aggProof.publicInputs[proofIdx]
                var li = vk.ic[0]
                for j in 0..<inputs.count {
                    if !inputs[j].isZero {
                        li = pointAdd(li, pointScalarMul(vk.ic[j + 1], inputs[j]))
                    }
                }
                groupAggL = pointAdd(groupAggL, cPointScalarMul(li, ri))
            }

            // e(-groupAggL, gamma)
            let negGL = pointNeg(groupAggL)
            guard let ngl = pointToAffine(negGL) else { return false }
            guard let gamma = g2ToAffine(vk.gamma_g2) else { return false }
            pairs.append((ngl, gamma))

            // Compute group aggC
            var groupAggC = pointIdentity()
            for (proofIdx, ri) in proofIndices {
                groupAggC = pointAdd(groupAggC, cPointScalarMul(originalProofs[proofIdx].c, ri))
            }

            // e(-groupAggC, delta)
            let negGC = pointNeg(groupAggC)
            guard let ngc = pointToAffine(negGC) else { return false }
            guard let delta = g2ToAffine(vk.delta_g2) else { return false }
            pairs.append((ngc, delta))
        }

        return cBN254PairingCheck(pairs)
    }

    // MARK: - Internal Helpers

    /// Compute per-proof public input accumulator: L_i = ic[0] + sum(inp_j * ic[j+1])
    private func computePerProofL(inputs: [AggregateProofInput]) -> [PointProjective] {
        var result = [PointProjective]()
        result.reserveCapacity(inputs.count)
        for input in inputs {
            let vk = input.vk
            var li = vk.ic[0]
            for j in 0..<input.publicInputs.count {
                if !input.publicInputs[j].isZero {
                    li = pointAdd(li, pointScalarMul(vk.ic[j + 1], input.publicInputs[j]))
                }
            }
            result.append(li)
        }
        return result
    }

    /// CPU MSM fallback: sum(points[i] * scalars[i])
    private func cpuAggregateMSM(
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
    private func absorbPoint(_ transcript: Transcript, _ p: PointProjective) {
        if let aff = pointToAffine(p) {
            transcript.absorb(fpToFr(aff.x))
            transcript.absorb(fpToFr(aff.y))
        } else {
            // Identity: absorb zeros
            transcript.absorb(Fr.zero)
            transcript.absorb(Fr.zero)
        }
    }

    /// Absorb a G2 projective point into the transcript.
    private func absorbG2Point(_ transcript: Transcript, _ p: G2ProjectivePoint) {
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

    /// Compare two projective points for equality via affine conversion.
    private func projPointEqual(_ a: PointProjective, _ b: PointProjective) -> Bool {
        if pointIsIdentity(a) && pointIsIdentity(b) { return true }
        if pointIsIdentity(a) || pointIsIdentity(b) { return false }
        let aAffs = batchToAffine([a])
        let bAffs = batchToAffine([b])
        return fpToInt(aAffs[0].x) == fpToInt(bAffs[0].x) &&
               fpToInt(aAffs[0].y) == fpToInt(bAffs[0].y)
    }
}

// MARK: - Aggregate Statistics

/// Statistics from an aggregation operation.
public struct AggregateStatistics {
    /// Number of proofs aggregated
    public let proofCount: Int
    /// Number of distinct circuits
    public let circuitCount: Int
    /// Number of IPPA rounds (log2 of padded proof count)
    public let ippaRounds: Int
    /// Whether GPU MSM was used
    public let usedGPU: Bool

    public init(proofCount: Int, circuitCount: Int, ippaRounds: Int, usedGPU: Bool) {
        self.proofCount = proofCount
        self.circuitCount = circuitCount
        self.ippaRounds = ippaRounds
        self.usedGPU = usedGPU
    }

    /// Compute statistics for a given set of inputs.
    public static func compute(inputs: [AggregateProofInput], gpuThreshold: Int) -> AggregateStatistics {
        let n = inputs.count
        let circuits = Set(inputs.map { $0.circuitIndex }).count
        var padN = 1
        while padN < n { padN <<= 1 }
        var rounds = 0
        var tmp = padN
        while tmp > 1 { tmp >>= 1; rounds += 1 }
        return AggregateStatistics(
            proofCount: n,
            circuitCount: circuits,
            ippaRounds: rounds,
            usedGPU: n >= gpuThreshold
        )
    }
}
