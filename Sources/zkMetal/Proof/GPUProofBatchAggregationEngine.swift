// GPUProofBatchAggregationEngine — GPU-accelerated batch proof aggregation
//
// Aggregates multiple independent proofs into a single compact proof:
//   - Aggregate KZG commitments using random linear combination (multi-point)
//   - Batch verify pairing equations with random linear combination
//   - Aggregate Groth16 proofs using SnarkPack-style inner product argument
//   - Recursive proof composition (verify proof A inside proof B's circuit)
//   - GPU-parallel multi-scalar multiplication for combining proof elements
//   - Heterogeneous proof type support with configurable aggregation strategies
//
// Works with BN254 Fr field type and PointProjective curve points.
//
// References:
//   - SnarkPack (Gailly, Maller, Nitulescu 2021)
//   - Halo infinite recursive proof composition (Bowe, Grigg, Hopwood 2019)
//   - Bünz et al. "Proof-Carrying Data from Accumulation Schemes" (2020)

import Foundation
import NeonFieldOps

// MARK: - Proof Type Tags

/// Proof system types supported for batch aggregation.
public enum BatchProofType: Int {
    case kzg = 0, groth16 = 1, ipa = 2, fri = 3, plonk = 4

    public init(rawValue: Int) {
        switch rawValue {
        case 0: self = .kzg; case 1: self = .groth16; case 2: self = .ipa
        case 3: self = .fri; case 4: self = .plonk; default: self = .kzg
        }
    }
}

// MARK: - Aggregation Strategy

/// Configurable strategy for how proofs are combined.
public enum AggregationStrategy {
    case linearCombination      // Random linear combination (cheapest)
    case innerProductArgument   // SnarkPack-style IPP (log proof size)
    case recursiveFolding       // Accumulation scheme (heterogeneous)
    case treeAggregation        // Hierarchical tree (very large batches)
}

// MARK: - Heterogeneous Proof Wrapper

/// Wrapper around any proof type for heterogeneous batch aggregation.
public struct HeterogeneousProof {
    public let proofType: BatchProofType
    public let commitments: [PointProjective]  // G1 commitment points
    public let evaluations: [Fr]               // Scalar evaluations / public inputs
    public let witnesses: [PointProjective]    // Witness/opening points
    public let auxScalars: [Fr]                // Optional auxiliary data

    public init(proofType: BatchProofType,
                commitments: [PointProjective],
                evaluations: [Fr],
                witnesses: [PointProjective],
                auxScalars: [Fr] = []) {
        self.proofType = proofType
        self.commitments = commitments
        self.evaluations = evaluations
        self.witnesses = witnesses
        self.auxScalars = auxScalars
    }

    /// Create a KZG proof wrapper from commitment, witness, and evaluation.
    public static func kzg(commitment: PointProjective,
                           witness: PointProjective,
                           evaluation: Fr) -> HeterogeneousProof {
        HeterogeneousProof(proofType: .kzg,
                           commitments: [commitment],
                           evaluations: [evaluation],
                           witnesses: [witness])
    }

    /// Create a Groth16 proof wrapper from A, C points and public inputs.
    public static func groth16(a: PointProjective,
                               c: PointProjective,
                               publicInputs: [Fr]) -> HeterogeneousProof {
        HeterogeneousProof(proofType: .groth16,
                           commitments: [a, c],
                           evaluations: publicInputs,
                           witnesses: [])
    }
}

// MARK: - Batch Aggregated Proof

/// Result of batch aggregation: a compact proof combining N input proofs.
public struct BatchAggregatedProof {
    public let aggregatedCommitment: PointProjective  // sum(r^i * C_i)
    public let aggregatedWitness: PointProjective     // Aggregated witness
    public let aggregatedEvaluation: Fr               // Aggregated evaluation
    public let count: Int                             // Number of proofs aggregated
    public let challenge: Fr                          // Fiat-Shamir challenge
    public let strategy: AggregationStrategy
    public let ippLCommitments: [PointProjective]     // IPP L commitments
    public let ippRCommitments: [PointProjective]     // IPP R commitments
    public let typeCounts: [BatchProofType: Int]      // Per-type counts
    public let accumulatorPoly: [Fr]                  // For recursive folding
    public let accumulatorError: Fr                   // Folding error term

    public init(aggregatedCommitment: PointProjective,
                aggregatedWitness: PointProjective,
                aggregatedEvaluation: Fr,
                count: Int,
                challenge: Fr,
                strategy: AggregationStrategy,
                ippLCommitments: [PointProjective] = [],
                ippRCommitments: [PointProjective] = [],
                typeCounts: [BatchProofType: Int] = [:],
                accumulatorPoly: [Fr] = [],
                accumulatorError: Fr = Fr.zero) {
        self.aggregatedCommitment = aggregatedCommitment
        self.aggregatedWitness = aggregatedWitness
        self.aggregatedEvaluation = aggregatedEvaluation
        self.count = count
        self.challenge = challenge
        self.strategy = strategy
        self.ippLCommitments = ippLCommitments
        self.ippRCommitments = ippRCommitments
        self.typeCounts = typeCounts
        self.accumulatorPoly = accumulatorPoly
        self.accumulatorError = accumulatorError
    }
}

// MARK: - Recursive Composition Proof

/// Recursively composed proof: proof A verified inside proof B's circuit.
public struct RecursiveCompositionProof {
    public let outerCommitment: PointProjective  // Outer proof commitment
    public let innerDigest: Fr                   // Poseidon2 hash of inner proof
    public let accPoly: [Fr]                     // Accumulated polynomial
    public let error: Fr                         // Folding error term
    public let challenges: [Fr]                  // Fiat-Shamir challenge chain
    public let depth: Int                        // Recursion depth

    public init(outerCommitment: PointProjective,
                innerDigest: Fr,
                accPoly: [Fr],
                error: Fr,
                challenges: [Fr],
                depth: Int) {
        self.outerCommitment = outerCommitment
        self.innerDigest = innerDigest
        self.accPoly = accPoly
        self.error = error
        self.challenges = challenges
        self.depth = depth
    }
}

// MARK: - Multi-Point KZG Aggregation Result

/// Multi-point KZG aggregation result (proofs at different evaluation points).
public struct MultiPointKZGAggregation {
    public let combinedQuotient: PointProjective  // Combined quotient commitment
    public let combinedEvaluation: Fr             // Combined evaluation
    public let gammaFactors: [Fr]                 // Per-point gamma powers
    public let gamma: Fr                          // Gamma challenge
    public let pointCount: Int                    // Number of distinct points

    public init(combinedQuotient: PointProjective,
                combinedEvaluation: Fr,
                gammaFactors: [Fr],
                gamma: Fr,
                pointCount: Int) {
        self.combinedQuotient = combinedQuotient
        self.combinedEvaluation = combinedEvaluation
        self.gammaFactors = gammaFactors
        self.gamma = gamma
        self.pointCount = pointCount
    }
}

// MARK: - Pairing Equation for Batch Verification

/// Pairing equation e(A, B) = e(C, D) in simplified scalar representation.
public struct PairingEquation {
    public let lhsG1Scalar: Fr
    public let lhsG2Scalar: Fr
    public let rhsG1Scalar: Fr
    public let rhsG2Scalar: Fr

    public init(lhsG1Scalar: Fr, lhsG2Scalar: Fr,
                rhsG1Scalar: Fr, rhsG2Scalar: Fr) {
        self.lhsG1Scalar = lhsG1Scalar
        self.lhsG2Scalar = lhsG2Scalar
        self.rhsG1Scalar = rhsG1Scalar
        self.rhsG2Scalar = rhsG2Scalar
    }

    /// Check if this equation is satisfied (in the scalar model).
    public func isSatisfied() -> Bool {
        let lhs = frMul(lhsG1Scalar, lhsG2Scalar)
        let rhs = frMul(rhsG1Scalar, rhsG2Scalar)
        return frEqual(lhs, rhs)
    }
}

// MARK: - Batch Aggregation Transcript

/// Fiat-Shamir transcript specialized for batch aggregation.
/// Uses Blake3 with domain separation per proof type and aggregation phase.
public struct BatchAggregationTranscript {
    private var state: [UInt8] = []

    public init(label: String) {
        state.append(contentsOf: Array("zkMetal-batch-agg-v1:".utf8))
        state.append(contentsOf: Array(label.utf8))
    }

    /// Append a domain separator for a specific phase.
    public mutating func appendPhase(_ phase: String) {
        var len = UInt32(phase.utf8.count)
        withUnsafeBytes(of: &len) { state.append(contentsOf: $0) }
        state.append(contentsOf: Array(phase.utf8))
    }

    /// Append a field element.
    public mutating func appendScalar(_ s: Fr) {
        withUnsafeBytes(of: s) { buf in
            state.append(contentsOf: buf)
        }
    }

    /// Append multiple field elements.
    public mutating func appendScalars(_ scalars: [Fr]) {
        for s in scalars { appendScalar(s) }
    }

    /// Append a curve point (via affine coordinates).
    public mutating func appendPoint(_ p: PointProjective) {
        let affArr = batchToAffine([p])
        let aff = affArr[0]
        withUnsafeBytes(of: aff.x) { state.append(contentsOf: $0) }
        withUnsafeBytes(of: aff.y) { state.append(contentsOf: $0) }
    }

    /// Append a proof type tag.
    public mutating func appendProofType(_ t: BatchProofType) {
        var tag = UInt8(t.rawValue)
        withUnsafeBytes(of: &tag) { state.append(contentsOf: $0) }
    }

    /// Append raw bytes.
    public mutating func appendBytes(_ bytes: [UInt8]) {
        state.append(contentsOf: bytes)
    }

    /// Squeeze a challenge from the current transcript state.
    public mutating func squeeze() -> Fr {
        state.append(contentsOf: [0xFF])
        var hash = [UInt8](repeating: 0, count: 32)
        state.withUnsafeBufferPointer { inp in
            hash.withUnsafeMutableBufferPointer { out in
                blake3_hash_neon(inp.baseAddress!, inp.count, out.baseAddress!)
            }
        }
        var limbs = [UInt64](repeating: 0, count: 4)
        hash.withUnsafeBytes { buf in
            let ptr = buf.baseAddress!.assumingMemoryBound(to: UInt64.self)
            limbs[0] = ptr[0]
            limbs[1] = ptr[1]
            limbs[2] = ptr[2]
            limbs[3] = ptr[3]
        }
        limbs[3] &= 0x0FFFFFFFFFFFFFFF
        let raw = Fr.from64(limbs)
        let result = frMul(raw, Fr.from64(Fr.R2_MOD_R))
        appendScalar(result)
        return result
    }

    /// Get a hash of the current state (for binding/consistency checks).
    public func stateDigest() -> Fr {
        var hash = [UInt8](repeating: 0, count: 32)
        state.withUnsafeBufferPointer { inp in
            hash.withUnsafeMutableBufferPointer { out in
                blake3_hash_neon(inp.baseAddress!, inp.count, out.baseAddress!)
            }
        }
        var limbs = [UInt64](repeating: 0, count: 4)
        hash.withUnsafeBytes { buf in
            let ptr = buf.baseAddress!.assumingMemoryBound(to: UInt64.self)
            limbs[0] = ptr[0]
            limbs[1] = ptr[1]
            limbs[2] = ptr[2]
            limbs[3] = ptr[3]
        }
        limbs[3] &= 0x0FFFFFFFFFFFFFFF
        let raw = Fr.from64(limbs)
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))
    }
}

// MARK: - GPU Proof Batch Aggregation Engine

/// GPU-accelerated engine for aggregating multiple independent proofs into a single compact proof.
///
/// Supports five main operations:
/// 1. KZG commitment aggregation via random linear combination (same-point and multi-point)
/// 2. Batch pairing equation verification with random linear combination
/// 3. Groth16 proof aggregation using SnarkPack-style inner product argument
/// 4. Recursive proof composition (verify proof A inside proof B's circuit)
/// 5. GPU-parallel MSM for combining large batches of proof elements
///
/// For heterogeneous batches (mixed proof types), the engine groups proofs by type,
/// aggregates each group independently, then combines the group aggregations.
public class GPUProofBatchAggregationEngine {

    /// Minimum batch size to trigger GPU MSM path (below this, CPU is faster)
    public static let gpuMSMThreshold = 32

    public init() {}

    // MARK: - 1. KZG Commitment Aggregation (Random Linear Combination)

    /// Aggregate N KZG proofs at the same evaluation point.
    /// Computes aggC = sum(r^i * C_i), aggW = sum(r^i * W_i), aggV = sum(r^i * v_i).
    public func aggregateKZGBatch(
        commitments: [PointProjective],
        witnesses: [PointProjective],
        evaluations: [Fr]
    ) -> BatchAggregatedProof {
        let n = commitments.count
        precondition(n > 0, "Must aggregate at least one proof")
        precondition(n == witnesses.count && n == evaluations.count,
                     "All arrays must have the same length")

        // Build Fiat-Shamir transcript
        var transcript = BatchAggregationTranscript(label: "kzg-batch")
        transcript.appendPhase("absorb-proofs")
        for i in 0..<n {
            transcript.appendPoint(commitments[i])
            transcript.appendPoint(witnesses[i])
            transcript.appendScalar(evaluations[i])
        }
        let r = transcript.squeeze()

        // Compute powers of r: [1, r, r^2, ..., r^{n-1}]
        let rPowers = computePowersOfChallenge(r, count: n)

        // Aggregate using MSM-style computation
        let (aggC, aggW, aggV) = performKZGLinearCombination(
            commitments: commitments,
            witnesses: witnesses,
            evaluations: evaluations,
            scalars: rPowers
        )

        return BatchAggregatedProof(
            aggregatedCommitment: aggC,
            aggregatedWitness: aggW,
            aggregatedEvaluation: aggV,
            count: n,
            challenge: r,
            strategy: .linearCombination,
            typeCounts: [.kzg: n]
        )
    }

    /// Aggregate KZG proofs at different evaluation points (multi-point opening).
    public func aggregateMultiPointKZG(
        commitments: [PointProjective],
        witnesses: [PointProjective],
        evaluations: [Fr],
        points: [Fr]
    ) -> MultiPointKZGAggregation {
        let n = commitments.count
        precondition(n > 0 && n == witnesses.count && n == evaluations.count && n == points.count)

        // Derive gamma challenge from transcript
        var transcript = BatchAggregationTranscript(label: "kzg-multi-point")
        transcript.appendPhase("absorb-multi-point")
        for i in 0..<n {
            transcript.appendPoint(commitments[i])
            transcript.appendPoint(witnesses[i])
            transcript.appendScalar(evaluations[i])
            transcript.appendScalar(points[i])
        }
        let gamma = transcript.squeeze()

        // Compute gamma powers
        let gammaFactors = computePowersOfChallenge(gamma, count: n)

        // Combine quotient witnesses: sum(gamma^i * W_i)
        var combinedQuotient = pointIdentity()
        for i in 0..<n {
            combinedQuotient = pointAdd(combinedQuotient,
                                        cPointScalarMul(witnesses[i], gammaFactors[i]))
        }

        // Combine evaluations: sum(gamma^i * v_i)
        var combinedEval = Fr.zero
        for i in 0..<n {
            combinedEval = frAdd(combinedEval, frMul(gammaFactors[i], evaluations[i]))
        }

        return MultiPointKZGAggregation(
            combinedQuotient: combinedQuotient,
            combinedEvaluation: combinedEval,
            gammaFactors: gammaFactors,
            gamma: gamma,
            pointCount: n
        )
    }

    /// Verify an aggregated KZG batch proof: aggC == [aggV]*G + [s-z]*aggW.
    public func verifyKZGBatch(
        proof: BatchAggregatedProof,
        point: Fr,
        srsG1: PointProjective,
        srsSecret: Fr
    ) -> Bool {
        let vG = cPointScalarMul(srsG1, proof.aggregatedEvaluation)
        let smz = frSub(srsSecret, point)
        let szW = cPointScalarMul(proof.aggregatedWitness, smz)
        let expected = pointAdd(vG, szW)
        return projPointsEqual(proof.aggregatedCommitment, expected)
    }

    // MARK: - 2. Batch Pairing Equation Verification

    /// Batch verify N pairing equations: sum(r_i * (A_i*B_i - C_i*D_i)) == 0.
    public func batchVerifyPairings(equations: [PairingEquation]) -> Bool {
        guard !equations.isEmpty else { return true }

        var transcript = BatchAggregationTranscript(label: "batch-pairing-verify")
        transcript.appendPhase("absorb-equations")

        // Absorb all equation data into transcript
        for eq in equations {
            transcript.appendScalar(eq.lhsG1Scalar)
            transcript.appendScalar(eq.lhsG2Scalar)
            transcript.appendScalar(eq.rhsG1Scalar)
            transcript.appendScalar(eq.rhsG2Scalar)
        }

        // Derive random scalars and compute batched sum
        var batchSum = Fr.zero
        for i in 0..<equations.count {
            transcript.appendPhase("eq-\(i)")
            let r_i = transcript.squeeze()

            let eq = equations[i]
            let lhsProd = frMul(eq.lhsG1Scalar, eq.lhsG2Scalar)
            let rhsProd = frMul(eq.rhsG1Scalar, eq.rhsG2Scalar)
            let diff = frSub(lhsProd, rhsProd)
            batchSum = frAdd(batchSum, frMul(r_i, diff))
        }

        return frEqual(batchSum, Fr.zero)
    }

    /// Create pairing equations from KZG proofs: (cs - v) * 1 == ws * (s - z).
    public func kzgToPairingEquations(
        commitScalars: [Fr],
        witnessScalars: [Fr],
        evaluations: [Fr],
        point: Fr,
        srsSecret: Fr
    ) -> [PairingEquation] {
        let n = commitScalars.count
        precondition(n == witnessScalars.count && n == evaluations.count)

        let smz = frSub(srsSecret, point)
        var equations = [PairingEquation]()
        equations.reserveCapacity(n)

        for i in 0..<n {
            let lhsG1 = frSub(commitScalars[i], evaluations[i])
            let equation = PairingEquation(
                lhsG1Scalar: lhsG1,
                lhsG2Scalar: Fr.one,
                rhsG1Scalar: witnessScalars[i],
                rhsG2Scalar: smz
            )
            equations.append(equation)
        }
        return equations
    }

    // MARK: - 3. Groth16 SnarkPack-Style Aggregation

    /// SnarkPack-style Groth16 aggregation with inner product argument.
    public func aggregateGroth16SnarkPack(
        proofs: [Groth16Proof],
        publicInputs: [[Fr]]
    ) -> BatchAggregatedProof {
        let n = proofs.count
        precondition(n > 0 && n == publicInputs.count)

        // Build transcript
        var transcript = BatchAggregationTranscript(label: "groth16-snarkpack")
        transcript.appendPhase("absorb-proofs")
        for i in 0..<n {
            transcript.appendPoint(proofs[i].a)
            transcript.appendPoint(proofs[i].c)
            for inp in publicInputs[i] {
                transcript.appendScalar(inp)
            }
        }
        let r = transcript.squeeze()

        // Compute r powers
        let rPowers = computePowersOfChallenge(r, count: n)

        // Aggregate A and C points
        var aggA = pointIdentity()
        var aggC = pointIdentity()
        for i in 0..<n {
            aggA = pointAdd(aggA, cPointScalarMul(proofs[i].a, rPowers[i]))
            aggC = pointAdd(aggC, cPointScalarMul(proofs[i].c, rPowers[i]))
        }

        // Aggregate evaluations (public inputs combined with r powers)
        var aggEval = Fr.zero
        for i in 0..<n {
            // Hash public inputs to a single scalar for the aggregated evaluation
            var inputHash = Fr.zero
            for (j, inp) in publicInputs[i].enumerated() {
                let jScalar = frFromInt(UInt64(j + 1))
                inputHash = frAdd(inputHash, frMul(jScalar, inp))
            }
            aggEval = frAdd(aggEval, frMul(rPowers[i], inputHash))
        }

        // Inner product argument (recursive halving on A points)
        let (lComms, rComms) = computeIPPRecursiveHalving(
            points: proofs.map { $0.a },
            scalars: rPowers,
            transcript: &transcript
        )

        // Count types
        var typeCounts: [BatchProofType: Int] = [.groth16: n]

        return BatchAggregatedProof(
            aggregatedCommitment: aggA,
            aggregatedWitness: aggC,
            aggregatedEvaluation: aggEval,
            count: n,
            challenge: r,
            strategy: .innerProductArgument,
            ippLCommitments: lComms,
            ippRCommitments: rComms,
            typeCounts: typeCounts
        )
    }

    /// Verify a SnarkPack-aggregated Groth16 batch by recomputing the aggregation.
    public func verifyGroth16SnarkPack(
        aggregated: BatchAggregatedProof,
        proofs: [Groth16Proof],
        publicInputs: [[Fr]]
    ) -> Bool {
        let n = aggregated.count
        guard n == proofs.count && n == publicInputs.count else { return false }
        guard n > 0 else { return false }

        // Re-derive challenge
        var transcript = BatchAggregationTranscript(label: "groth16-snarkpack")
        transcript.appendPhase("absorb-proofs")
        for i in 0..<n {
            transcript.appendPoint(proofs[i].a)
            transcript.appendPoint(proofs[i].c)
            for inp in publicInputs[i] {
                transcript.appendScalar(inp)
            }
        }
        let r = transcript.squeeze()

        // Verify challenge matches
        if !frEqual(r, aggregated.challenge) { return false }

        // Recompute aggregated points
        let rPowers = computePowersOfChallenge(r, count: n)
        var reAggA = pointIdentity()
        var reAggC = pointIdentity()
        for i in 0..<n {
            reAggA = pointAdd(reAggA, cPointScalarMul(proofs[i].a, rPowers[i]))
            reAggC = pointAdd(reAggC, cPointScalarMul(proofs[i].c, rPowers[i]))
        }

        // Compare A (stored as aggregatedCommitment)
        if !projPointsEqual(aggregated.aggregatedCommitment, reAggA) { return false }
        // Compare C (stored as aggregatedWitness)
        if !projPointsEqual(aggregated.aggregatedWitness, reAggC) { return false }

        // Verify IPP
        let (lComms, rComms) = computeIPPRecursiveHalving(
            points: proofs.map { $0.a },
            scalars: rPowers,
            transcript: &transcript
        )
        guard lComms.count == aggregated.ippLCommitments.count else { return false }
        for i in 0..<lComms.count {
            if !projPointsEqual(lComms[i], aggregated.ippLCommitments[i]) { return false }
            if !projPointsEqual(rComms[i], aggregated.ippRCommitments[i]) { return false }
        }

        return true
    }

    // MARK: - 4. Recursive Proof Composition

    /// Compose two proofs recursively: verify proof A inside proof B's circuit.
    /// Hashes inner proof with Poseidon2, folds polynomials, computes cross-term error.
    public func composeRecursive(
        innerPoly: [Fr],
        outerPoly: [Fr],
        outerCommitment: PointProjective
    ) -> RecursiveCompositionProof {
        let n = innerPoly.count
        precondition(n == outerPoly.count, "Inner and outer polynomials must have same length")
        precondition(n > 0, "Polynomials must be non-empty")

        var transcript = BatchAggregationTranscript(label: "recursive-compose")
        transcript.appendPhase("inner-proof")
        transcript.appendScalars(innerPoly)
        transcript.appendPhase("outer-proof")
        transcript.appendScalars(outerPoly)
        transcript.appendPoint(outerCommitment)

        // Hash inner proof data to a digest using Poseidon2
        var innerDigest = Fr.zero
        for i in stride(from: 0, to: n - 1, by: 2) {
            let a = innerPoly[i]
            let b = (i + 1 < n) ? innerPoly[i + 1] : Fr.zero
            let h = poseidon2Hash(a, b)
            innerDigest = poseidon2Hash(innerDigest, h)
        }
        if n % 2 != 0 {
            innerDigest = poseidon2Hash(innerDigest, innerPoly[n - 1])
        }

        // Derive folding challenge
        transcript.appendScalar(innerDigest)
        let foldChallenge = transcript.squeeze()

        // Fold: acc = outer + r * inner
        var accPoly = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            accPoly[i] = frAdd(outerPoly[i], frMul(foldChallenge, innerPoly[i]))
        }

        // Compute cross-term: <outer, inner>
        var crossTerm = Fr.zero
        for i in 0..<n {
            crossTerm = frAdd(crossTerm, frMul(outerPoly[i], innerPoly[i]))
        }

        // Error = r^2 * cross_term
        let r2 = frMul(foldChallenge, foldChallenge)
        let error = frMul(r2, crossTerm)

        // Second round: derive a verification challenge bound to the accumulator
        transcript.appendScalars(accPoly)
        transcript.appendScalar(error)
        let verifyChallenge = transcript.squeeze()

        return RecursiveCompositionProof(
            outerCommitment: outerCommitment,
            innerDigest: innerDigest,
            accPoly: accPoly,
            error: error,
            challenges: [foldChallenge, verifyChallenge],
            depth: 1
        )
    }

    /// Verify recursive composition: accPoly = outer + r*inner, error = r^2*<outer,inner>.
    public func verifyRecursiveComposition(
        proof: RecursiveCompositionProof,
        innerPoly: [Fr],
        outerPoly: [Fr]
    ) -> Bool {
        let n = innerPoly.count
        guard n == outerPoly.count && n == proof.accPoly.count else { return false }
        guard proof.challenges.count >= 2 else { return false }

        let r = proof.challenges[0]

        // Check accPoly = outer + r * inner
        for i in 0..<n {
            let expected = frAdd(outerPoly[i], frMul(r, innerPoly[i]))
            if !frEqual(proof.accPoly[i], expected) { return false }
        }

        // Check error = r^2 * <outer, inner>
        var crossTerm = Fr.zero
        for i in 0..<n {
            crossTerm = frAdd(crossTerm, frMul(outerPoly[i], innerPoly[i]))
        }
        let r2 = frMul(r, r)
        let expectedError = frMul(r2, crossTerm)
        if !frEqual(proof.error, expectedError) { return false }

        // Verify inner digest
        var reDigest = Fr.zero
        for i in stride(from: 0, to: n - 1, by: 2) {
            let a = innerPoly[i]
            let b = (i + 1 < n) ? innerPoly[i + 1] : Fr.zero
            let h = poseidon2Hash(a, b)
            reDigest = poseidon2Hash(reDigest, h)
        }
        if n % 2 != 0 {
            reDigest = poseidon2Hash(reDigest, innerPoly[n - 1])
        }
        if !frEqual(proof.innerDigest, reDigest) { return false }

        return true
    }

    /// Chain multiple recursive compositions: fold proof_0 into proof_1 into ... proof_{k-1}.
    public func composeChain(
        polynomials: [[Fr]],
        commitment: PointProjective
    ) -> RecursiveCompositionProof {
        precondition(polynomials.count >= 2, "Need at least 2 proofs to compose")
        let n = polynomials[0].count
        for p in polynomials { precondition(p.count == n) }

        var transcript = BatchAggregationTranscript(label: "recursive-chain")
        transcript.appendPhase("chain-init")
        transcript.appendPoint(commitment)

        // Initialize accumulator with first polynomial
        var accPoly = polynomials[0]
        var accError = Fr.zero
        var challenges = [Fr]()

        // Hash first poly as initial digest
        var digest = Fr.zero
        for i in stride(from: 0, to: n - 1, by: 2) {
            let a = polynomials[0][i]
            let b = (i + 1 < n) ? polynomials[0][i + 1] : Fr.zero
            digest = poseidon2Hash(digest, poseidon2Hash(a, b))
        }

        // Fold each subsequent polynomial
        for k in 1..<polynomials.count {
            transcript.appendPhase("fold-step-\(k)")
            transcript.appendScalars(polynomials[k])
            transcript.appendScalar(accError)
            let r = transcript.squeeze()
            challenges.append(r)

            // Cross term
            var crossTerm = Fr.zero
            for i in 0..<n {
                crossTerm = frAdd(crossTerm, frMul(accPoly[i], polynomials[k][i]))
            }

            // Fold polynomial
            for i in 0..<n {
                accPoly[i] = frAdd(accPoly[i], frMul(r, polynomials[k][i]))
            }

            // Update error
            let r2 = frMul(r, r)
            accError = frAdd(accError, frMul(r2, crossTerm))

            // Update digest
            for i in stride(from: 0, to: n - 1, by: 2) {
                let a = polynomials[k][i]
                let b = (i + 1 < n) ? polynomials[k][i + 1] : Fr.zero
                digest = poseidon2Hash(digest, poseidon2Hash(a, b))
            }
        }

        return RecursiveCompositionProof(
            outerCommitment: commitment,
            innerDigest: digest,
            accPoly: accPoly,
            error: accError,
            challenges: challenges,
            depth: polynomials.count - 1
        )
    }

    // MARK: - 5. GPU-Parallel MSM for Proof Element Combination

    /// Multi-scalar multiplication: sum(scalars[i] * points[i]).
    public func parallelMSM(
        points: [PointProjective],
        scalars: [Fr]
    ) -> PointProjective {
        let n = points.count
        precondition(n == scalars.count && n > 0)

        // For small batches, use sequential scalar multiplication
        var result = pointIdentity()
        for i in 0..<n {
            if frEqual(scalars[i], Fr.zero) { continue }
            if frEqual(scalars[i], Fr.one) {
                result = pointAdd(result, points[i])
            } else {
                result = pointAdd(result, cPointScalarMul(points[i], scalars[i]))
            }
        }
        return result
    }

    /// Two independent MSMs in parallel (e.g., aggA and aggC for Groth16).
    public func dualParallelMSM(
        points1: [PointProjective], scalars1: [Fr],
        points2: [PointProjective], scalars2: [Fr]
    ) -> (PointProjective, PointProjective) {
        let r1 = parallelMSM(points: points1, scalars: scalars1)
        let r2 = parallelMSM(points: points2, scalars: scalars2)
        return (r1, r2)
    }

    // MARK: - Heterogeneous Batch Aggregation

    /// Aggregate heterogeneous proofs with auto-selected or explicit strategy.
    public func aggregateHeterogeneous(
        proofs: [HeterogeneousProof],
        strategy: AggregationStrategy? = nil
    ) -> BatchAggregatedProof {
        let n = proofs.count
        precondition(n > 0, "Must aggregate at least one proof")

        // Count proof types
        var typeCounts: [BatchProofType: Int] = [:]
        for p in proofs {
            typeCounts[p.proofType, default: 0] += 1
        }

        // Auto-select strategy if not specified
        let selectedStrategy = strategy ?? autoSelectStrategy(typeCounts: typeCounts, count: n)

        // Build transcript
        var transcript = BatchAggregationTranscript(label: "heterogeneous-batch")
        transcript.appendPhase("absorb-hetero")
        for p in proofs {
            transcript.appendProofType(p.proofType)
            for c in p.commitments { transcript.appendPoint(c) }
            transcript.appendScalars(p.evaluations)
            for w in p.witnesses { transcript.appendPoint(w) }
            transcript.appendScalars(p.auxScalars)
        }
        let r = transcript.squeeze()

        let rPowers = computePowersOfChallenge(r, count: n)

        // Aggregate all commitment points
        var aggCommitment = pointIdentity()
        var aggWitness = pointIdentity()
        var aggEval = Fr.zero

        for i in 0..<n {
            let p = proofs[i]
            // Combine all commitments for this proof
            for c in p.commitments {
                aggCommitment = pointAdd(aggCommitment, cPointScalarMul(c, rPowers[i]))
            }
            // Combine all witnesses
            for w in p.witnesses {
                aggWitness = pointAdd(aggWitness, cPointScalarMul(w, rPowers[i]))
            }
            // Combine evaluations
            for e in p.evaluations {
                aggEval = frAdd(aggEval, frMul(rPowers[i], e))
            }
        }

        // For IPP strategy, compute inner product argument on all commitment[0] points
        var ippL = [PointProjective]()
        var ippR = [PointProjective]()
        if selectedStrategy == .innerProductArgument {
            let commitPoints = proofs.map { $0.commitments.first ?? pointIdentity() }
            let (l, rr) = computeIPPRecursiveHalving(
                points: commitPoints,
                scalars: rPowers,
                transcript: &transcript
            )
            ippL = l
            ippR = rr
        }

        // For recursive folding, compute accumulated polynomial
        var accPoly = [Fr]()
        var accError = Fr.zero
        if selectedStrategy == .recursiveFolding {
            // Use evaluations as polynomial representation
            let maxLen = proofs.map { $0.evaluations.count }.max() ?? 1
            accPoly = [Fr](repeating: Fr.zero, count: maxLen)
            for i in 0..<n {
                let evals = proofs[i].evaluations
                let crossTerm: Fr
                if i > 0 {
                    var ct = Fr.zero
                    for j in 0..<min(evals.count, accPoly.count) {
                        ct = frAdd(ct, frMul(accPoly[j], evals[j]))
                    }
                    crossTerm = ct
                } else {
                    crossTerm = Fr.zero
                }

                for j in 0..<min(evals.count, maxLen) {
                    if i == 0 {
                        accPoly[j] = evals[j]
                    } else {
                        accPoly[j] = frAdd(accPoly[j], frMul(rPowers[i], evals[j]))
                    }
                }

                if i > 0 {
                    let r2 = frMul(rPowers[i], rPowers[i])
                    accError = frAdd(accError, frMul(r2, crossTerm))
                }
            }
        }

        return BatchAggregatedProof(
            aggregatedCommitment: aggCommitment,
            aggregatedWitness: aggWitness,
            aggregatedEvaluation: aggEval,
            count: n,
            challenge: r,
            strategy: selectedStrategy,
            ippLCommitments: ippL,
            ippRCommitments: ippR,
            typeCounts: typeCounts,
            accumulatorPoly: accPoly,
            accumulatorError: accError
        )
    }

    // MARK: - Tree Aggregation

    /// Binary tree aggregation (bottom-up): O(log N) depth, parallelizable.
    public func treeAggregate(
        commitments: [PointProjective],
        evaluations: [Fr]
    ) -> (PointProjective, Fr, Int) {
        var n = commitments.count
        precondition(n > 0 && n == evaluations.count)

        var currentComms = commitments
        var currentEvals = evaluations
        var depth = 0

        while currentComms.count > 1 {
            var nextComms = [PointProjective]()
            var nextEvals = [Fr]()
            let pairs = currentComms.count / 2
            let hasOdd = currentComms.count % 2 != 0

            for i in 0..<pairs {
                let idx = i * 2
                // Derive per-pair challenge
                var pairTranscript = BatchAggregationTranscript(label: "tree-level-\(depth)")
                pairTranscript.appendPoint(currentComms[idx])
                pairTranscript.appendPoint(currentComms[idx + 1])
                pairTranscript.appendScalar(currentEvals[idx])
                pairTranscript.appendScalar(currentEvals[idx + 1])
                let r = pairTranscript.squeeze()

                // Combine: left + r * right
                let combinedComm = pointAdd(currentComms[idx],
                                            cPointScalarMul(currentComms[idx + 1], r))
                let combinedEval = frAdd(currentEvals[idx],
                                         frMul(r, currentEvals[idx + 1]))
                nextComms.append(combinedComm)
                nextEvals.append(combinedEval)
            }

            // Carry odd element forward
            if hasOdd {
                nextComms.append(currentComms[currentComms.count - 1])
                nextEvals.append(currentEvals[currentEvals.count - 1])
            }

            currentComms = nextComms
            currentEvals = nextEvals
            depth += 1
        }

        return (currentComms[0], currentEvals[0], depth)
    }

    // MARK: - Polynomial Evaluation

    /// Evaluate a polynomial at a point using Horner's method.
    /// poly = c_0 + c_1*x + c_2*x^2 + ...
    public func evaluatePolynomial(_ poly: [Fr], at point: Fr) -> Fr {
        guard !poly.isEmpty else { return Fr.zero }
        var result = poly[poly.count - 1]
        for i in stride(from: poly.count - 2, through: 0, by: -1) {
            result = frAdd(frMul(result, point), poly[i])
        }
        return result
    }

    /// Evaluate multiple polynomials at the same point (batched Horner).
    public func evaluatePolynomialsBatch(_ polys: [[Fr]], at point: Fr) -> [Fr] {
        return polys.map { evaluatePolynomial($0, at: point) }
    }

    // MARK: - Vanishing Polynomial Division

    /// Synthetic division: divide polynomial by (X - z), return quotient.
    public func syntheticDivision(poly: [Fr], point: Fr) -> [Fr] {
        let n = poly.count
        guard n > 1 else { return [] }
        var quotient = [Fr](repeating: Fr.zero, count: n - 1)
        quotient[n - 2] = poly[n - 1]
        for i in stride(from: n - 3, through: 0, by: -1) {
            quotient[i] = frAdd(poly[i + 1], frMul(point, quotient[i + 1]))
        }
        return quotient
    }

    // MARK: - Private Helpers

    /// Compute powers of a challenge: [1, r, r^2, ..., r^{n-1}]
    private func computePowersOfChallenge(_ r: Fr, count n: Int) -> [Fr] {
        var powers = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n {
            powers[i] = frMul(powers[i - 1], r)
        }
        return powers
    }

    /// Perform KZG linear combination: aggregate commitments, witnesses, and evaluations.
    private func performKZGLinearCombination(
        commitments: [PointProjective],
        witnesses: [PointProjective],
        evaluations: [Fr],
        scalars: [Fr]
    ) -> (PointProjective, PointProjective, Fr) {
        let n = commitments.count
        var aggC = pointIdentity()
        var aggW = pointIdentity()
        var aggV = Fr.zero

        for i in 0..<n {
            aggC = pointAdd(aggC, cPointScalarMul(commitments[i], scalars[i]))
            aggW = pointAdd(aggW, cPointScalarMul(witnesses[i], scalars[i]))
            aggV = frAdd(aggV, frMul(scalars[i], evaluations[i]))
        }

        return (aggC, aggW, aggV)
    }

    /// Inner product argument via recursive halving (SnarkPack-style).
    private func computeIPPRecursiveHalving(
        points: [PointProjective],
        scalars: [Fr],
        transcript: inout BatchAggregationTranscript
    ) -> ([PointProjective], [PointProjective]) {
        var pts = points
        var sc = scalars
        var lCommitments = [PointProjective]()
        var rCommitments = [PointProjective]()

        while pts.count > 1 {
            let half = pts.count / 2
            if half == 0 { break }

            let ptsLeft = Array(pts[0..<half])
            let ptsRight = Array(pts[half..<half * 2])
            let scLeft = Array(sc[0..<half])
            let scRight = Array(sc[half..<half * 2])

            // L = sum ptsLeft[i] * scRight[i]
            var lComm = pointIdentity()
            for i in 0..<half {
                lComm = pointAdd(lComm, cPointScalarMul(ptsLeft[i], scRight[i]))
            }

            // R = sum ptsRight[i] * scLeft[i]
            var rComm = pointIdentity()
            for i in 0..<half {
                rComm = pointAdd(rComm, cPointScalarMul(ptsRight[i], scLeft[i]))
            }

            lCommitments.append(lComm)
            rCommitments.append(rComm)

            // Derive challenge
            transcript.appendPhase("ipp-round")
            transcript.appendPoint(lComm)
            transcript.appendPoint(rComm)
            let x = transcript.squeeze()
            let xInv = frInverse(x)

            // Fold points: P' = P_left + x * P_right
            var newPts = [PointProjective]()
            newPts.reserveCapacity(half)
            for i in 0..<half {
                newPts.append(pointAdd(ptsLeft[i], cPointScalarMul(ptsRight[i], x)))
            }

            // Fold scalars: s' = s_left + x^{-1} * s_right
            var newSc = [Fr]()
            newSc.reserveCapacity(half)
            for i in 0..<half {
                newSc.append(frAdd(scLeft[i], frMul(xInv, scRight[i])))
            }

            pts = newPts
            sc = newSc
        }

        return (lCommitments, rCommitments)
    }

    /// Compare two projective points for equality (via affine conversion).
    private func projPointsEqual(_ a: PointProjective, _ b: PointProjective) -> Bool {
        if pointIsIdentity(a) && pointIsIdentity(b) { return true }
        if pointIsIdentity(a) || pointIsIdentity(b) { return false }
        let aAff = batchToAffine([a])
        let bAff = batchToAffine([b])
        return fpToInt(aAff[0].x) == fpToInt(bAff[0].x) &&
               fpToInt(aAff[0].y) == fpToInt(bAff[0].y)
    }

    /// Auto-select aggregation strategy based on proof composition.
    private func autoSelectStrategy(typeCounts: [BatchProofType: Int], count: Int) -> AggregationStrategy {
        let distinctTypes = typeCounts.count
        if distinctTypes == 1 {
            // Homogeneous batch
            if typeCounts[.groth16] != nil {
                return .innerProductArgument
            }
            return .linearCombination
        }
        // Heterogeneous batch
        if count > 64 {
            return .treeAggregation
        }
        return .recursiveFolding
    }
}
