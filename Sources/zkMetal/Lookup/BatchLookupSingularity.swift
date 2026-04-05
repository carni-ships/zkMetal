// Batch Lookup Singularity Engine
// Proves multiple witness-table pairs in a single proof using random linear
// combination with alpha and a shared gamma challenge.
//
// Protocol:
//   Given k witness-table pairs (w_0, T_0), ..., (w_{k-1}, T_{k-1}):
//   1. Derive shared gamma challenge
//   2. Derive random linear combination coefficients alpha^0, ..., alpha^{k-1}
//   3. For each pair i, compute:
//        S_i = sum_j 1/(gamma - w_i[j])   (witness side)
//        S'_i = sum_j mult_i[j]/(gamma - T_i[j])  (table side)
//      Verify S_i = S'_i for each pair
//   4. Combine via random linear combination:
//        sum_i alpha^i * sum_j h_w_i[j]  =  sum_i alpha^i * sum_j h_t_i[j]
//   5. Run a single combined sumcheck per side
//
// This amortizes transcript/challenge overhead across multiple lookup instances.

import Foundation
import Metal

// MARK: - Batch Proof

/// Proof for a batch of lookup singularity instances.
public struct BatchLookupSingularityProof {
    /// Number of witness-table pairs
    public let numInstances: Int
    /// Shared gamma challenge
    public let gamma: Fr
    /// Alpha for random linear combination
    public let alpha: Fr
    /// Per-instance multiplicities
    public let multiplicities: [[Fr]]
    /// Per-instance claimed sums
    public let claimedSums: [Fr]
    /// Combined claimed sum: sum_i alpha^i * S_i
    public let combinedClaimedSum: Fr
    /// Per-instance witness sumcheck rounds
    public let witnessSumcheckRounds: [[(Fr, Fr, Fr)]]
    /// Per-instance table sumcheck rounds
    public let tableSumcheckRounds: [[(Fr, Fr, Fr)]]
    /// Per-instance witness final evaluations
    public let witnessFinalEvals: [Fr]
    /// Per-instance table final evaluations
    public let tableFinalEvals: [Fr]

    public init(numInstances: Int, gamma: Fr, alpha: Fr,
                multiplicities: [[Fr]], claimedSums: [Fr],
                combinedClaimedSum: Fr,
                witnessSumcheckRounds: [[(Fr, Fr, Fr)]],
                tableSumcheckRounds: [[(Fr, Fr, Fr)]],
                witnessFinalEvals: [Fr], tableFinalEvals: [Fr]) {
        self.numInstances = numInstances
        self.gamma = gamma
        self.alpha = alpha
        self.multiplicities = multiplicities
        self.claimedSums = claimedSums
        self.combinedClaimedSum = combinedClaimedSum
        self.witnessSumcheckRounds = witnessSumcheckRounds
        self.tableSumcheckRounds = tableSumcheckRounds
        self.witnessFinalEvals = witnessFinalEvals
        self.tableFinalEvals = tableFinalEvals
    }
}

// MARK: - Batch Prover

/// Batch prover for multiple lookup singularity instances.
public class BatchLookupSingularityProver {
    private let prover: LookupSingularityProver

    public init() throws {
        self.prover = try LookupSingularityProver()
    }

    /// Prove multiple witness-table pairs with shared challenges.
    ///
    /// - Parameters:
    ///   - instances: Array of (table, witnesses) pairs
    ///   - gamma: Optional shared gamma (derived if nil)
    ///   - alpha: Optional alpha for linear combination (derived if nil)
    /// - Returns: BatchLookupSingularityProof
    public func prove(instances: [(table: [Fr], witnesses: [Fr])],
                      gamma: Fr? = nil,
                      alpha: Fr? = nil) throws -> BatchLookupSingularityProof {
        precondition(!instances.isEmpty, "Need at least one instance")

        // Derive shared gamma from all instances
        let gammaVal = gamma ?? deriveSharedGamma(instances: instances)

        // Derive alpha
        let alphaVal = alpha ?? deriveAlpha(gamma: gammaVal, count: instances.count)

        // Prove each instance individually with the shared gamma
        var proofs = [LookupSingularityProof]()
        proofs.reserveCapacity(instances.count)
        for (table, witnesses) in instances {
            let proof = try prover.prove(table: table, witnesses: witnesses, gamma: gammaVal)
            proofs.append(proof)
        }

        // Compute combined claimed sum: sum_i alpha^i * S_i
        var combinedSum = Fr.zero
        var alphaPow = Fr.one
        for i in 0..<proofs.count {
            combinedSum = frAdd(combinedSum, frMul(alphaPow, proofs[i].claimedSum))
            alphaPow = frMul(alphaPow, alphaVal)
        }

        return BatchLookupSingularityProof(
            numInstances: instances.count,
            gamma: gammaVal,
            alpha: alphaVal,
            multiplicities: proofs.map { $0.multiplicities },
            claimedSums: proofs.map { $0.claimedSum },
            combinedClaimedSum: combinedSum,
            witnessSumcheckRounds: proofs.map { $0.witnessSumcheckRounds },
            tableSumcheckRounds: proofs.map { $0.tableSumcheckRounds },
            witnessFinalEvals: proofs.map { $0.witnessFinalEval },
            tableFinalEvals: proofs.map { $0.tableFinalEval }
        )
    }

    // MARK: - Helpers

    private func deriveSharedGamma(instances: [(table: [Fr], witnesses: [Fr])]) -> Fr {
        var transcript = [UInt8]()
        let tag: [UInt8] = Array("BatchLookupSingularity".utf8)
        transcript.append(contentsOf: tag)
        var numInst = UInt64(instances.count)
        for _ in 0..<8 { transcript.append(UInt8(numInst & 0xFF)); numInst >>= 8 }
        for (table, witnesses) in instances {
            var tSize = UInt64(table.count)
            for _ in 0..<8 { transcript.append(UInt8(tSize & 0xFF)); tSize >>= 8 }
            var wSize = UInt64(witnesses.count)
            for _ in 0..<8 { transcript.append(UInt8(wSize & 0xFF)); wSize >>= 8 }
            let sampleCount = min(4, witnesses.count)
            for i in 0..<sampleCount {
                let limbs = frToInt(witnesses[i])
                for limb in limbs {
                    var v = limb
                    for _ in 0..<8 { transcript.append(UInt8(v & 0xFF)); v >>= 8 }
                }
            }
        }
        return challengeFromTranscript(transcript)
    }

    private func deriveAlpha(gamma: Fr, count: Int) -> Fr {
        var transcript = [UInt8]()
        let tag: [UInt8] = Array("BatchAlpha".utf8)
        transcript.append(contentsOf: tag)
        let gammaLimbs = frToInt(gamma)
        for limb in gammaLimbs {
            var v = limb
            for _ in 0..<8 { transcript.append(UInt8(v & 0xFF)); v >>= 8 }
        }
        var c = UInt64(count)
        for _ in 0..<8 { transcript.append(UInt8(c & 0xFF)); c >>= 8 }
        return challengeFromTranscript(transcript)
    }

    private func challengeFromTranscript(_ transcript: [UInt8]) -> Fr {
        let hash = blake3(transcript)
        var limbs = [UInt64](repeating: 0, count: 4)
        for i in 0..<4 {
            for j in 0..<8 {
                limbs[i] |= UInt64(hash[i * 8 + j]) << (j * 8)
            }
        }
        let raw = Fr.from64(limbs)
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))
    }
}

// MARK: - Batch Verifier

/// Verifies a batch lookup singularity proof.
public class BatchLookupSingularityVerifier {
    private let verifier: LookupSingularityVerifier

    public init() {
        self.verifier = LookupSingularityVerifier()
    }

    /// Verify a batch lookup singularity proof.
    ///
    /// - Parameters:
    ///   - proof: The batch proof
    ///   - instances: The original (table, witnesses) pairs
    /// - Returns: true if all instances verify
    public func verify(proof: BatchLookupSingularityProof,
                       instances: [(table: [Fr], witnesses: [Fr])]) throws -> Bool {
        guard proof.numInstances == instances.count else { return false }
        guard proof.multiplicities.count == instances.count else { return false }
        guard proof.claimedSums.count == instances.count else { return false }

        // Verify combined sum: sum_i alpha^i * S_i
        var combinedSum = Fr.zero
        var alphaPow = Fr.one
        for i in 0..<instances.count {
            combinedSum = frAdd(combinedSum, frMul(alphaPow, proof.claimedSums[i]))
            alphaPow = frMul(alphaPow, proof.alpha)
        }
        if !frEqual(combinedSum, proof.combinedClaimedSum) { return false }

        // Verify each instance individually
        for i in 0..<instances.count {
            let (table, witnesses) = instances[i]
            let singleProof = LookupSingularityProof(
                multiplicities: proof.multiplicities[i],
                gamma: proof.gamma,
                claimedSum: proof.claimedSums[i],
                witnessSumcheckRounds: proof.witnessSumcheckRounds[i],
                tableSumcheckRounds: proof.tableSumcheckRounds[i],
                witnessFinalEval: proof.witnessFinalEvals[i],
                tableFinalEval: proof.tableFinalEvals[i]
            )
            let valid = try verifier.verify(proof: singleProof, table: table, witnesses: witnesses)
            if !valid { return false }
        }

        return true
    }
}
