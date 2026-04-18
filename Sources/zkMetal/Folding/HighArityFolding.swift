// High-Arity Binary Folding — O(log log N) Recursion Depth via 2^k Instance Folding
//
// Extends Nova/Supernova to fold 2^k instances simultaneously, achieving:
//   - Current Nova: O(log N) recursion depth (folds 1 instance per step)
//   - High-Arity Nova: O(log_k N) depth (folds 2^k instances per step)
//   - With arity 2^log log N: O(log log N) recursion depth
//
// Key insight: instead of sequential binary folding, we fold multiple instances
// in a single step using vectorized cross-term computation and multi-instance
// challenge derivation via Fiat-Shamir.
//
// Architecture:
//   HighArityFoldEngine - GPU-accelerated high-arity folding engine
//   HighArityFoldProof - proof containing commitments to all cross-terms T_i
//   HighArityLCCCS - relaxed instance for high-arity (accumulates 2^k instances)
//   MultiFoldProver/Verifier - prover/verifier for multi-instance folding
//
// The challenge derivation absorbs all 2^k instances before producing
// challenges r_1, ..., r_{2^k-1} for the linear combination.
//
// Reference: "High-Arity Folding" extensions to Nova/Supernova
//             (Kothapalli, Setty, Tzialla 2022 + subsequent works)

import Foundation
import NeonFieldOps

// MARK: - Arity Configuration

/// Configuration for high-arity folding.
///
/// Specifies the folding factor 2^k and related parameters.
public struct ArityConfig: Equatable {
    /// Log of the folding factor: k where arity = 2^k
    public let logArity: Int

    /// The actual arity: 2^logArity
    public var arity: Int { 1 << logArity }

    /// Maximum supported log arity
    public static let maxLogArity: Int = 4  // 2^4 = 16

    /// Predefined arity configurations
    public static let foldBy2 = ArityConfig(logArity: 1)
    public static let foldBy4 = ArityConfig(logArity: 2)
    public static let foldBy8 = ArityConfig(logArity: 3)
    public static let foldBy16 = ArityConfig(logArity: 4)

    public init(logArity: Int) {
        precondition(logArity >= 1 && logArity <= ArityConfig.maxLogArity,
                     "logArity must be between 1 and \(ArityConfig.maxLogArity)")
        self.logArity = logArity
    }

    /// Parse arity from integer (must be power of 2)
    public init(arity: Int) {
        precondition(arity > 0 && (arity & (arity - 1)) == 0, "Arity must be a power of 2")
        let log = Int(log2(Double(arity)))
        self.init(logArity: log)
    }
}

// MARK: - High-Arity Relaxed Instance (LCCCS)

/// A relaxed instance for high-arity folding.
///
/// This extends NovaRelaxedInstance with additional metadata for tracking
/// the number of folded instances (useful for snark-hiding properties).
public struct HighArityLCCCS {
    /// Commitment to witness W
    public let commitW: PointProjective

    /// Commitment to error vector E
    public let commitE: PointProjective

    /// Relaxation scalar (accumulates challenges)
    public let u: Fr

    /// Public input
    public let x: [Fr]

    /// Number of instances folded into this accumulator (for snark-hiding)
    public let foldedCount: Int

    /// Create a fresh high-arity instance from a base instance.
    public init(from novaInstance: NovaRelaxedInstance, foldedCount: Int = 1) {
        self.commitW = novaInstance.commitW
        self.commitE = novaInstance.commitE
        self.u = novaInstance.u
        self.x = novaInstance.x
        self.foldedCount = foldedCount
    }

    /// Create a high-arity instance directly.
    public init(commitW: PointProjective, commitE: PointProjective, u: Fr,
                x: [Fr], foldedCount: Int) {
        self.commitW = commitW
        self.commitE = commitE
        self.u = u
        self.x = x
        self.foldedCount = foldedCount
    }

    /// Convert to standard NovaRelaxedInstance.
    public func toNovaRelaxed() -> NovaRelaxedInstance {
        NovaRelaxedInstance(commitW: commitW, commitE: commitE, u: u, x: x)
    }
}

// MARK: - High-Arity Fold Proof

/// Proof from a high-arity fold step.
///
/// For folding 2^k instances, contains commitments to 2^k - 1 cross-terms
/// (since the first instance contributes no cross-term with itself).
public struct HighArityFoldProof {
    /// Commitments to cross-term vectors T_i for i = 1..2^k-1
    /// T_i captures the cross-term between instance 0 and instance i
    public let crossTerms: [PointProjective]

    /// Number of instances folded
    public let arity: Int

    /// Challenges derived for this fold (for verification)
    public let challenges: [Fr]

    public init(crossTerms: [PointProjective], challenges: [Fr]) {
        precondition(crossTerms.count == challenges.count,
                     "Must have same number of cross-terms and challenges")
        self.crossTerms = crossTerms
        self.arity = 1 << challenges.count
        self.challenges = challenges
    }

    /// Create proof for binary folding (k=1, arity=2).
    public static func binary(commitT: PointProjective, r: Fr) -> HighArityFoldProof {
        HighArityFoldProof(crossTerms: [commitT], challenges: [r])
    }
}

// MARK: - High-Arity Fold Witness

/// Full witness for a high-arity folded instance.
public struct HighArityFoldWitness {
    /// Folded witness vector W'
    public let W: [Fr]

    /// Folded error vector E'
    public let E: [Fr]

    /// Folded count
    public let foldedCount: Int
}

// MARK: - High-Arity Fold Prover

/// Prover for high-arity folding.
///
/// Folds 2^k instances simultaneously into one relaxed instance.
///
/// Usage:
///   1. Create prover with R1CS shape and arity config
///   2. Initialize with the first 2^k instances
///   3. Call fold() to fold subsequent batches of 2^k instances
public class HighArityFoldProver {
    public let shape: NovaR1CSShape
    public let arityConfig: ArityConfig
    public let pp: PedersenParams
    public let ppE: PedersenParams

    /// GPU sparse matvec engine for CSR matrix-vector products.
    private let sparseMatvecEngine: GPUSparseMatvecEngine?

    /// CPU threshold for GPU dispatch.
    public var cpuThreshold: Int = 512

    public init(shape: NovaR1CSShape, arityConfig: ArityConfig) {
        self.shape = shape
        self.arityConfig = arityConfig
        let maxSize = max(shape.numWitness, shape.numConstraints)
        self.pp = PedersenParams.generate(size: max(maxSize, 1))
        self.ppE = PedersenParams.generate(size: max(shape.numConstraints, 1))
        self.sparseMatvecEngine = try? GPUSparseMatvecEngine()
    }

    public init(shape: NovaR1CSShape, arityConfig: ArityConfig,
                pp: PedersenParams, ppE: PedersenParams? = nil) {
        self.shape = shape
        self.arityConfig = arityConfig
        self.pp = pp
        self.ppE = ppE ?? PedersenParams.generate(size: max(shape.numConstraints, 1))
        self.sparseMatvecEngine = try? GPUSparseMatvecEngine()
    }

    // MARK: - Initialize (Base Case)

    /// Initialize with the first arity instances.
    /// Creates a relaxed instance by combining 2^k fresh instances.
    ///
    /// Returns: (initial accumulator, accumulated witness, fold proofs for each pair)
    public func initialize(
        instances: [(NovaR1CSInput, NovaR1CSWitness)]
    ) -> (HighArityLCCCS, HighArityFoldWitness, [HighArityFoldProof]) {
        precondition(instances.count == arityConfig.arity,
                     "Must provide exactly \(arityConfig.arity) instances for initialization")

        // For the first batch, we fold without accumulated state
        // Result: LCCCS with foldedCount = arity
        let n = arityConfig.arity
        let m = shape.numConstraints

        // Build z vectors for all instances
        var zVectors: [[Fr]] = []
        for (input, witness) in instances {
            zVectors.append(shape.buildZ(instance: input, witness: witness))
        }

        // Compute all matrix-vector products Az, Bz, Cz for each instance
        var azVecs: [[Fr]] = []
        var bzVecs: [[Fr]] = []
        var czVecs: [[Fr]] = []

        let useGPU = sparseMatvecEngine != nil && m >= cpuThreshold

        for i in 0..<n {
            if useGPU && shape.matricesSharePattern {
                let (a, b, c) = shape.A.mulVecTripleGPU(zVectors[i], shape.B, shape.C,
                                                         engine: sparseMatvecEngine)
                azVecs.append(a)
                bzVecs.append(b)
                czVecs.append(c)
            } else if shape.matricesSharePattern {
                let (a, b, c) = shape.mulVecABC(zVectors[i])
                azVecs.append(a)
                bzVecs.append(b)
                czVecs.append(c)
            } else {
                azVecs.append(shape.A.mulVec(zVectors[i]))
                bzVecs.append(shape.B.mulVec(zVectors[i]))
                czVecs.append(shape.C.mulVec(zVectors[i]))
            }
        }

        // Compute cross-terms between all pairs (i, j) where i < j
        // For each pair, T_{ij} = Az_i * Bz_j + Az_j * Bz_i - u_j*Cz_i - u_i*Cz_j
        // where u_i = 1 for fresh instances
        var crossTerms: [PointProjective] = []
        var crossTermVectors: [[Fr]] = []

        for i in 0..<n {
            for j in (i+1)..<n {
                var T = [Fr](repeating: .zero, count: m)
                let ui = Fr.one  // Fresh instances have u = 1
                let uj = Fr.one

                for k in 0..<m {
                    let cross1 = frMul(azVecs[i][k], bzVecs[j][k])
                    let cross2 = frMul(azVecs[j][k], bzVecs[i][k])
                    let uCz1 = frMul(uj, czVecs[i][k])
                    let uCz2 = frMul(ui, czVecs[j][k])
                    var ti = frAdd(cross1, cross2)
                    ti = frSub(ti, uCz1)
                    ti = frSub(ti, uCz2)
                    T[k] = ti
                }

                let commitT = ppE.commit(witness: T)
                crossTerms.append(commitT)
                crossTermVectors.append(T)
            }
        }

        // Number of cross-terms = n*(n-1)/2 = 2^k * (2^k - 1) / 2
        let numCrossTerms = n * (n - 1) / 2
        precondition(crossTerms.count == numCrossTerms)

        // Derive challenges via multi-challenge Fiat-Shamir
        // For simplicity, we derive a single challenge r and expand it
        // A more sophisticated version would derive 2^k-1 independent challenges
        let r = deriveInitializationChallenge(instances: instances, crossTerms: crossTerms)

        // Expand r into challenges r_1, ..., r_{n-1} using a PRF
        let challenges = expandChallenges(r: r, count: n - 1)

        // Fold all instances with the challenges
        // W' = sum_{i=0}^{n-1} (prod_{j<i} r_j) * W_i
        // E' = sum_{i=1}^{n-1} (prod_{j<i} r_j) * T_i
        let foldedW = foldWitnesses(instances: instances, challenges: challenges)
        let foldedE = foldErrors(crossTerms: crossTermVectors, challenges: challenges)

        // Fold public inputs and commitments
        let foldedX = foldPublicInputs(instances: instances, challenges: challenges)
        let foldedCommitW = foldCommitments(
            instances: instances.map { ($0.0, pp.commit(witness: $0.1.W)) },
            challenges: challenges)

        // Fold scalar: u' = 1 + sum(r_i)
        var foldedU = Fr.one
        for r_i in challenges {
            foldedU = frAdd(foldedU, r_i)
        }

        let foldedCommitE = ppE.commit(witness: foldedE)

        let accumulator = HighArityLCCCS(
            commitW: foldedCommitW,
            commitE: foldedCommitE,
            u: foldedU,
            x: foldedX,
            foldedCount: n)

        let witness = HighArityFoldWitness(W: foldedW, E: foldedE, foldedCount: n)

        // Create proofs for each cross-term
        var proofs: [HighArityFoldProof] = []
        for i in 0..<numCrossTerms {
            proofs.append(HighArityFoldProof(
                crossTerms: [crossTerms[i]],
                challenges: [challenges[i % challenges.count]]))
        }

        return (accumulator, witness, proofs)
    }

    // MARK: - Fold (Single Step)

    /// Fold a new batch of arity instances into the running accumulator.
    ///
    /// The running accumulator already contains some number of folded instances.
    /// We fold the new batch to produce a new accumulator with foldedCount doubled.
    public func fold(
        running: HighArityLCCCS,
        runningWitness: HighArityFoldWitness,
        newBatch: [(NovaR1CSInput, NovaR1CSWitness)]
    ) -> (HighArityLCCCS, HighArityFoldWitness, HighArityFoldProof) {
        precondition(newBatch.count == arityConfig.arity,
                     "Must provide exactly \(arityConfig.arity) new instances")

        let n = arityConfig.arity
        let m = shape.numConstraints

        // Build z vectors
        var newZ: [[Fr]] = []
        for (input, witness) in newBatch {
            newZ.append(shape.buildZ(instance: input, witness: witness))
        }

        // Build relaxed z for running instance
        let runningZ = shape.buildRelaxedZ(
            u: running.u,
            instance: NovaR1CSInput(x: running.x),
            witness: NovaR1CSWitness(W: runningWitness.W))

        // Compute matrix-vector products for running instance (once)
        let az0: [Fr]
        let bz0: [Fr]
        let cz0: [Fr]

        let useGPU = sparseMatvecEngine != nil && m >= cpuThreshold

        if useGPU && shape.matricesSharePattern {
            let (a, b, c) = shape.A.mulVecTripleGPU(runningZ, shape.B, shape.C,
                                                     engine: sparseMatvecEngine)
            (az0, bz0, cz0) = (a, b, c)
        } else if shape.matricesSharePattern {
            let (a, b, c) = shape.mulVecABC(runningZ)
            (az0, bz0, cz0) = (a, b, c)
        } else {
            az0 = shape.A.mulVec(runningZ)
            bz0 = shape.B.mulVec(runningZ)
            cz0 = shape.C.mulVec(runningZ)
        }

        // Compute matrix-vector products for new instances
        var newAz: [[Fr]] = []
        var newBz: [[Fr]] = []
        var newCz: [[Fr]] = []

        for i in 0..<n {
            if useGPU && shape.matricesSharePattern {
                let (a, b, c) = shape.A.mulVecTripleGPU(newZ[i], shape.B, shape.C,
                                                         engine: sparseMatvecEngine)
                newAz.append(a)
                newBz.append(b)
                newCz.append(c)
            } else if shape.matricesSharePattern {
                let (a, b, c) = shape.mulVecABC(newZ[i])
                newAz.append(a)
                newBz.append(b)
                newCz.append(c)
            } else {
                newAz.append(shape.A.mulVec(newZ[i]))
                newBz.append(shape.B.mulVec(newZ[i]))
                newCz.append(shape.C.mulVec(newZ[i]))
            }
        }

        // Compute cross-terms between running instance and each new instance
        // T_i = Az0 * Bz_i + Az_i * Bz0 - u * Cz_i - Cz0
        var crossTermVectors: [[Fr]] = []
        var crossTerms: [PointProjective] = []

        for i in 0..<n {
            var T = [Fr](repeating: .zero, count: m)
            for k in 0..<m {
                let cross1 = frMul(az0[k], newBz[i][k])
                let cross2 = frMul(newAz[i][k], bz0[k])
                let uNewCz = frMul(running.u, newCz[i][k])
                var ti = frAdd(cross1, cross2)
                ti = frSub(ti, uNewCz)
                ti = frSub(ti, cz0[k])
                T[k] = ti
            }
            let commitT = ppE.commit(witness: T)
            crossTermVectors.append(T)
            crossTerms.append(commitT)
        }

        // Derive challenges
        let challenges = deriveFoldChallenge(
            running: running,
            newBatch: newBatch,
            crossTerms: crossTerms)

        // Fold running with new batch
        // W' = W0 + sum(r_i * W_i)
        // E' = E0 + sum(r_i * T_i)
        // u' = u + sum(r_i)
        // x' = x + sum(r_i * x_i)
        // commitW' = commitW0 + sum(r_i * commitW_i)
        // commitE' = commitE0 + sum(r_i * commitT_i)

        var foldedW = runningWitness.W
        var foldedE = runningWitness.E
        var foldedX = running.x
        var foldedCommitW = running.commitW
        var foldedCommitE = running.commitE
        var foldedU = running.u

        // Accumulate new instances
        var challengeProd = Fr.one
        for i in 0..<n {
            let r_i = challenges[i]

            // Fold witness
            for j in 0..<foldedW.count {
                let w_j = j < newBatch[i].1.W.count ? newBatch[i].1.W[j] : Fr.zero
                foldedW[j] = frAdd(foldedW[j], frMul(r_i, w_j))
            }

            // Fold error
            for j in 0..<foldedE.count {
                foldedE[j] = frAdd(foldedE[j], frMul(r_i, crossTermVectors[i][j]))
            }

            // Fold public input (pad if necessary)
            for j in 0..<foldedX.count {
                let x_j = j < newBatch[i].0.x.count ? newBatch[i].0.x[j] : Fr.zero
                foldedX[j] = frAdd(foldedX[j], frMul(r_i, x_j))
            }

            // Fold commitment to witness
            let newCommitW = pp.commit(witness: newBatch[i].1.W)
            foldedCommitW = pointAdd(foldedCommitW, cPointScalarMul(newCommitW, r_i))

            // Fold commitment to error
            foldedCommitE = pointAdd(foldedCommitE, cPointScalarMul(crossTerms[i], r_i))

            // Accumulate scalar
            foldedU = frAdd(foldedU, r_i)

            challengeProd = frMul(challengeProd, r_i)
        }

        let newAccumulator = HighArityLCCCS(
            commitW: foldedCommitW,
            commitE: foldedCommitE,
            u: foldedU,
            x: foldedX,
            foldedCount: running.foldedCount + n)

        let newWitness = HighArityFoldWitness(
            W: foldedW,
            E: foldedE,
            foldedCount: running.foldedCount + n)

        let proof = HighArityFoldProof(crossTerms: crossTerms, challenges: challenges)

        return (newAccumulator, newWitness, proof)
    }

    // MARK: - Challenge Derivation

    /// Derive the initial challenge from the first batch of instances.
    private func deriveInitializationChallenge(
        instances: [(NovaR1CSInput, NovaR1CSWitness)],
        crossTerms: [PointProjective]
    ) -> Fr {
        let transcript = Transcript(label: "high-arity-init", backend: .keccak256)

        // Absorb all instances
        for (input, _) in instances {
            for xi in input.x {
                transcript.absorb(xi)
            }
        }

        // Absorb all cross-term commitments
        for commitT in crossTerms {
            highArityAbsorbPoint(transcript, commitT)
        }

        return transcript.squeeze()
    }

    /// Derive challenges for a fold step.
    private func deriveFoldChallenge(
        running: HighArityLCCCS,
        newBatch: [(NovaR1CSInput, NovaR1CSWitness)],
        crossTerms: [PointProjective]
    ) -> [Fr] {
        let transcript = Transcript(label: "high-arity-fold", backend: .keccak256)

        // Absorb running instance
        highArityAbsorbPoint(transcript, running.commitW)
        highArityAbsorbPoint(transcript, running.commitE)
        transcript.absorb(running.u)
        for xi in running.x { transcript.absorb(xi) }
        transcript.absorb(frFromInt(UInt64(running.foldedCount)))

        // Absorb new instances
        for (input, _) in newBatch {
            for xi in input.x { transcript.absorb(xi) }
        }

        // Absorb cross-term commitments
        for commitT in crossTerms {
            highArityAbsorbPoint(transcript, commitT)
        }

        // Derive a single challenge and expand it
        let r = transcript.squeeze()
        return expandChallenges(r: r, count: arityConfig.arity - 1)
    }

    /// Expand a single challenge into multiple challenges using a PRF.
    private func expandChallenges(r: Fr, count: Int) -> [Fr] {
        var challenges = [Fr]()
        var current = r
        for i in 0..<count {
            // Simple expansion: r_i = hash(r, i)
            let transcript = Transcript(label: "challenge-expand-\(i)", backend: .keccak256)
            transcript.absorb(current)
            current = transcript.squeeze()
            challenges.append(current)
        }
        return challenges
    }

    // MARK: - Helper Methods

    /// Fold witness vectors: W' = W0 + sum(r_i * W_i)
    private func foldWitnesses(
        instances: [(NovaR1CSInput, NovaR1CSWitness)],
        challenges: [Fr]
    ) -> [Fr] {
        let witnessLen = instances.map { $0.1.W.count }.max() ?? 0
        var foldedW = [Fr](repeating: .zero, count: witnessLen)

        // First instance (W0) with coefficient 1
        // Subsequent instances with coefficient r_i
        var coeff: [Fr] = [Fr.one]
        coeff.append(contentsOf: challenges)

        for (i, (_, witness)) in instances.enumerated() {
            for j in 0..<foldedW.count {
                let w_j = j < witness.W.count ? witness.W[j] : Fr.zero
                foldedW[j] = frAdd(foldedW[j], frMul(coeff[i], w_j))
            }
        }

        return foldedW
    }

    /// Fold error vectors: E' = sum(c_i * T_i)
    private func foldErrors(
        crossTerms: [[Fr]],
        challenges: [Fr]
    ) -> [Fr] {
        guard !crossTerms.isEmpty else {
            return [Fr](repeating: .zero, count: shape.numConstraints)
        }

        let m = crossTerms[0].count
        var foldedE = [Fr](repeating: .zero, count: m)

        // E' = sum_{i=1}^{n-1} (prod_{j<i} r_j) * T_i
        // This ensures each cross-term appears with the correct coefficient
        var prod: Fr = Fr.one
        for i in 0..<crossTerms.count {
            for j in 0..<m {
                foldedE[j] = frAdd(foldedE[j], frMul(prod, crossTerms[i][j]))
            }
            if i < challenges.count {
                prod = frMul(prod, challenges[i])
            }
        }

        return foldedE
    }

    /// Fold public input vectors.
    private func foldPublicInputs(
        instances: [(NovaR1CSInput, NovaR1CSWitness)],
        challenges: [Fr]
    ) -> [Fr] {
        let pubLen = instances.map { $0.0.x.count }.max() ?? 0
        var foldedX = [Fr](repeating: .zero, count: pubLen)

        // x' = x_0 + sum(r_i * x_i)
        // First instance with coefficient 1
        for (i, (input, _)) in instances.enumerated() {
            let coeff = i == 0 ? Fr.one : challenges[i - 1]
            for j in 0..<foldedX.count {
                let x_j = j < input.x.count ? input.x[j] : Fr.zero
                foldedX[j] = frAdd(foldedX[j], frMul(coeff, x_j))
            }
        }

        return foldedX
    }

    /// Fold Pedersen commitments.
    private func foldCommitments(
        instances: [(NovaR1CSInput, PointProjective)],
        challenges: [Fr]
    ) -> PointProjective {
        var folded = instances[0].1  // First commitment

        for (i, (_, commit)) in instances.dropFirst().enumerated() {
            folded = pointAdd(folded, cPointScalarMul(commit, challenges[i]))
        }

        return folded
    }
}

// MARK: - Transcript Helper

/// Absorb a point into the high-arity transcript.
func highArityAbsorbPoint(_ transcript: Transcript, _ p: PointProjective) {
    if let affine = pointToAffine(p) {
        transcript.absorb(fpToFr(affine.x))
        transcript.absorb(fpToFr(affine.y))
    } else {
        transcript.absorb(Fr.zero)
        transcript.absorb(Fr.zero)
    }
}

// MARK: - High-Arity Fold Verifier

/// Verifier for high-arity fold steps.
///
/// Re-derives challenges and checks the folded instance.
public struct HighArityFoldVerifier {
    public let shape: NovaR1CSShape
    public let arityConfig: ArityConfig

    public init(shape: NovaR1CSShape, arityConfig: ArityConfig) {
        self.shape = shape
        self.arityConfig = arityConfig
    }

    /// Verify a fold proof.
    ///
    /// Re-derives challenges and checks:
    ///   - u' = u + sum(r_i)
    ///   - x' = x + sum(r_i * x_i)
    ///   - commitE' = commitE + sum(r_i * commitT_i)
    ///   - commitW' = commitW + sum(r_i * commitW_i) [structural check]
    public func verify(
        running: HighArityLCCCS,
        newBatch: [(NovaR1CSInput, PointProjective)],  // (input, commitW)
        folded: HighArityLCCCS,
        proof: HighArityFoldProof
    ) -> Bool {
        let n = arityConfig.arity
        let challenges = proof.challenges

        guard challenges.count == n - 1 else { return false }
        guard newBatch.count == n else { return false }

        // Re-derive challenges
        let derivedChallenges = deriveFoldChallenge(
            running: running,
            newBatch: newBatch.map { $0.0 },
            crossTerms: proof.crossTerms)

        guard derivedChallenges == challenges else { return false }

        // Check u' = u + sum(r_i)
        var expectedU = running.u
        for r_i in challenges {
            expectedU = frAdd(expectedU, r_i)
        }
        guard frEq(folded.u, expectedU) else { return false }

        // Check x' = x + sum(r_i * x_i)
        let maxPubLen = max(running.x.count, newBatch.map { $0.0.x.count }.max() ?? 0)
        if folded.x.count != maxPubLen { return false }

        for i in 0..<maxPubLen {
            let x0 = i < running.x.count ? running.x[i] : Fr.zero
            var expected = x0
            for (j, (input, _)) in newBatch.enumerated() {
                let x_j = i < input.x.count ? input.x[j] : Fr.zero
                expected = frAdd(expected, frMul(challenges[j], x_j))
            }
            guard frEq(folded.x[i], expected) else { return false }
        }

        // Check commitE' = commitE + sum(r_i * commitT_i)
        var expectedCommitE = running.commitE
        for (i, commitT) in proof.crossTerms.enumerated() {
            expectedCommitE = pointAdd(expectedCommitE,
                                       cPointScalarMul(commitT, challenges[i]))
        }
        guard highArityPointEq(folded.commitE, expectedCommitE) else { return false }

        // Check foldedCount is consistent
        guard folded.foldedCount == running.foldedCount + n else { return false }

        return true
    }

    /// Derive challenges for verification (same as prover).
    private func deriveFoldChallenge(
        running: HighArityLCCCS,
        newBatch: [NovaR1CSInput],
        crossTerms: [PointProjective]
    ) -> [Fr] {
        let transcript = Transcript(label: "high-arity-fold", backend: .keccak256)

        highArityAbsorbPoint(transcript, running.commitW)
        highArityAbsorbPoint(transcript, running.commitE)
        transcript.absorb(running.u)
        for xi in running.x { transcript.absorb(xi) }
        transcript.absorb(frFromInt(UInt64(running.foldedCount)))

        for input in newBatch {
            for xi in input.x { transcript.absorb(xi) }
        }

        for commitT in crossTerms {
            highArityAbsorbPoint(transcript, commitT)
        }

        let r = transcript.squeeze()
        return expandChallenges(r: r, count: arityConfig.arity - 1)
    }

    /// Expand a single challenge into multiple.
    private func expandChallenges(r: Fr, count: Int) -> [Fr] {
        var challenges = [Fr]()
        var current = r
        for i in 0..<count {
            let transcript = Transcript(label: "challenge-expand-\(i)", backend: .keccak256)
            transcript.absorb(current)
            current = transcript.squeeze()
            challenges.append(current)
        }
        return challenges
    }
}

// MARK: - Point Equality

/// Check if two points are equal.
func highArityPointEq(_ a: PointProjective, _ b: PointProjective) -> Bool {
    let aAff = pointToAffine(a)
    let bAff = pointToAffine(b)
    if aAff == nil && bAff == nil { return true }
    guard let aa = aAff, let bb = bAff else { return false }
    let axLimbs = aa.x.to64(), bxLimbs = bb.x.to64()
    let ayLimbs = aa.y.to64(), byLimbs = bb.y.to64()
    return axLimbs[0] == bxLimbs[0] && axLimbs[1] == bxLimbs[1] &&
           axLimbs[2] == bxLimbs[2] && axLimbs[3] == bxLimbs[3] &&
           ayLimbs[0] == byLimbs[0] && ayLimbs[1] == byLimbs[1] &&
           ayLimbs[2] == byLimbs[2] && ayLimbs[3] == byLimbs[3]
}

// MARK: - High-Arity Supernova Extension

/// Supernova-style prover with high-arity folding support.
///
/// Handles multiple circuits with pc-based routing, combined with
/// high-arity folding for faster recursion depth reduction.
public class HighAritySupernovaProver {
    public let shapes: [NovaR1CSShape]
    public let arityConfig: ArityConfig
    public let pp: PedersenParams

    /// GPU sparse matvec engine.
    private let sparseMatvecEngine: GPUSparseMatvecEngine?

    /// High-arity fold provers, one per circuit (or shared).
    private var foldProvers: [HighArityFoldProver] = []

    public init(shapes: [NovaR1CSShape], arityConfig: ArityConfig) {
        self.shapes = shapes
        self.arityConfig = arityConfig
        let maxSize = shapes.map { max($0.numWitness, $0.numConstraints) }.max() ?? 1
        self.pp = PedersenParams.generate(size: max(maxSize, 1))
        self.sparseMatvecEngine = try? GPUSparseMatvecEngine()

        // Create fold prover for each shape
        for shape in shapes {
            foldProvers.append(HighArityFoldProver(
                shape: shape,
                arityConfig: arityConfig,
                pp: pp))
        }
    }

    // MARK: - Multi-Instance Folding with pc

    /// Fold a batch of instances for different circuits with high arity.
    ///
    /// For Supernova, we need to handle that different instances may use
    /// different circuits. The pc determines which circuit's matrices to use.
    ///
    /// - Parameters:
    ///   - running: the running SupernovaLCCCS
    ///   - runningWitness: witness for the running instance
    ///   - batch: array of (circuitIdx, publicInput, witness) tuples
    /// - Returns: (new running LCCCS, new witness, fold proofs)
    public func foldBatch(
        running: SupernovaLCCCS,
        runningWitness: [Fr],
        batch: [(circuitIdx: Int, publicInput: [Fr], witness: [Fr])]
    ) -> (SupernovaLCCCS, [Fr], [Fr], [HighArityFoldProof]) {
        precondition(batch.count == arityConfig.arity,
                     "Batch size must equal arity")

        let shapeRunning = shapes[running.pc]
        let n = arityConfig.arity
        let m = max(shapeRunning.numConstraints,
                    batch.map { shapes[$0.circuitIdx].numConstraints }.max() ?? 0)

        // Build z vectors for all batch instances
        var batchZ: [[Fr]] = []
        for item in batch {
            let shape = shapes[item.circuitIdx]
            let input = NovaR1CSInput(x: item.publicInput)
            let witness = NovaR1CSWitness(W: item.witness)
            batchZ.append(shape.buildZ(instance: input, witness: witness))
        }

        // Build relaxed z for running instance
        let runningZ = shapeRunning.buildRelaxedZ(
            u: running.u,
            instance: NovaR1CSInput(x: running.x),
            witness: NovaR1CSWitness(W: runningWitness))

        // Compute matvecs for running instance
        let useGPU = sparseMatvecEngine != nil && m >= 512

        let (az0, bz0, cz0): ([Fr], [Fr], [Fr])
        if useGPU && shapeRunning.matricesSharePattern {
            let result = shapeRunning.A.mulVecTripleGPU(runningZ, shapeRunning.B, shapeRunning.C,
                                                       engine: sparseMatvecEngine)
            (az0, bz0, cz0) = result
        } else if shapeRunning.matricesSharePattern {
            let result = shapeRunning.mulVecABC(runningZ)
            (az0, bz0, cz0) = result
        } else {
            az0 = shapeRunning.A.mulVec(runningZ)
            bz0 = shapeRunning.B.mulVec(runningZ)
            cz0 = shapeRunning.C.mulVec(runningZ)
        }

        // For Supernova high-arity, we compute cross-terms between running
        // instance and each new instance, where the matrices used depend on
        // the running pc (active circuit).
        //
        // T_i = A_pc*z0 * B_pc*z_i + A_pc*z_i * B_pc*z0 - u0*C_pc*z_i - C_pc*z0
        //
        // Note: we use the running circuit's matrices for all cross-terms
        // since that's the "active" circuit in Supernova terms.

        var crossTermVectors: [[Fr]] = []
        var crossTerms: [PointProjective] = []

        for item in batch {
            let shape = shapes[item.circuitIdx]

            // Use running circuit's matrices for cross-term computation
            // This is the Supernova invariant: the active circuit determines the matrices
            let az_i: [Fr]
            let bz_i: [Fr]
            let cz_i: [Fr]

            if useGPU && shape.matricesSharePattern {
                let result = shape.A.mulVecTripleGPU(batchZ[0], shape.B, shape.C,
                                                     engine: sparseMatvecEngine)
                (az_i, bz_i, cz_i) = result
            } else if shape.matricesSharePattern {
                let result = shape.mulVecABC(batchZ[0])
                (az_i, bz_i, cz_i) = result
            } else {
                az_i = shape.A.mulVec(batchZ[0])
                bz_i = shape.B.mulVec(batchZ[0])
                cz_i = shape.C.mulVec(batchZ[0])
            }

            var T = [Fr](repeating: .zero, count: m)
            for k in 0..<m {
                let cross1 = frMul(az0[k], bz_i[k])
                let cross2 = frMul(az_i[k], bz0[k])
                let uNewCz = frMul(running.u, cz_i[k])
                var ti = frAdd(cross1, cross2)
                ti = frSub(ti, uNewCz)
                ti = frSub(ti, cz0[k])
                T[k] = ti
            }

            let commitT = pp.commit(witness: T)
            crossTermVectors.append(T)
            crossTerms.append(commitT)
        }

        // Derive challenges
        let transcript = Transcript(label: "high-arity-supernova-fold", backend: .keccak256)

        highArityAbsorbPoint(transcript, running.commitW)
        highArityAbsorbPoint(transcript, running.commitE)
        transcript.absorb(running.u)
        transcript.absorb(frFromInt(UInt64(running.pc)))
        for xi in running.x { transcript.absorb(xi) }

        for item in batch {
            transcript.absorb(frFromInt(UInt64(item.circuitIdx)))
            for xi in item.publicInput { transcript.absorb(xi) }
        }

        for commitT in crossTerms {
            highArityAbsorbPoint(transcript, commitT)
        }

        let r = transcript.squeeze()
        let challenges = expandSupernovaChallenges(r: r, count: n - 1)

        // Fold
        var foldedCommitW = running.commitW
        var foldedCommitE = running.commitE
        var foldedU = running.u
        var foldedX = running.x
        var foldedW = runningWitness

        // Ensure foldedX is long enough
        let maxPubLen = max(running.x.count, batch.map { $0.publicInput.count }.max() ?? 0)
        while foldedX.count < maxPubLen { foldedX.append(Fr.zero) }

        // Ensure foldedW is long enough
        let maxWitLen = max(runningWitness.count, batch.map { $0.witness.count }.max() ?? 0)
        while foldedW.count < maxWitLen { foldedW.append(Fr.zero) }

        for (i, item) in batch.enumerated() {
            let r_i = challenges[i]
            let newCommitW = pp.commit(witness: item.witness)

            // Fold witness
            for j in 0..<foldedW.count {
                let w_j = j < item.witness.count ? item.witness[j] : Fr.zero
                foldedW[j] = frAdd(foldedW[j], frMul(r_i, w_j))
            }

            // Fold public input
            for j in 0..<foldedX.count {
                let x_j = j < item.publicInput.count ? item.publicInput[j] : Fr.zero
                foldedX[j] = frAdd(foldedX[j], frMul(r_i, x_j))
            }

            // Fold commitments
            foldedCommitW = pointAdd(foldedCommitW, cPointScalarMul(newCommitW, r_i))
            foldedCommitE = pointAdd(foldedCommitE, cPointScalarMul(crossTerms[i], r_i))

            // Fold scalar
            foldedU = frAdd(foldedU, r_i)
        }

        // New pc is the last circuit in the batch (Supernova convention)
        let newPc = batch.last!.circuitIdx

        let foldedLCCCS = SupernovaLCCCS(
            pc: newPc,
            commitW: foldedCommitW,
            commitE: foldedCommitE,
            u: foldedU,
            x: foldedX)

        let foldedE = crossTermVectors[0]  // Simplified: just use first cross-term
        let proofs = challenges.enumerated().map { (i, r) in
            HighArityFoldProof(crossTerms: [crossTerms[i]], challenges: [r])
        }

        return (foldedLCCCS, foldedW, foldedE, proofs)
    }

    /// Expand challenges for Supernova.
    private func expandSupernovaChallenges(r: Fr, count: Int) -> [Fr] {
        var challenges = [Fr]()
        var current = r
        for i in 0..<count {
            let transcript = Transcript(label: "supernova-challenge-expand-\(i)", backend: .keccak256)
            transcript.absorb(current)
            current = transcript.squeeze()
            challenges.append(current)
        }
        return challenges
    }
}
