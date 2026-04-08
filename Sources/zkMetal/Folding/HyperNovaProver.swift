// HyperNova Prover — Multi-instance folding prover
//
// Folds N CCS instances into 1 without proving each individually.
// This enables incremental verifiable computation (IVC):
//   prove step_1, fold; prove step_2, fold; ... prove step_N, fold;
//   then produce one SNARK proof on the final accumulated instance.
//
// Supports:
//   - 2-instance fold (LCCCS + CCCS) via the existing HyperNovaEngine
//   - N-instance fold: fold multiple CCCS into a running LCCCS in one step
//   - IVC chain: repeatedly fold sequential computation steps
//
// Reference: "HyperNova: Recursive arguments from folding schemes" (Kothapalli, Setty 2023)

import Foundation
import NeonFieldOps

// MARK: - HyperNova Prover

public class HyperNovaProver {
    public let engine: HyperNovaEngine
    public let ccs: CCSInstance

    /// Initialize prover with a CCS structure.
    /// Internally creates the HyperNovaEngine with Pedersen parameters.
    public init(ccs: CCSInstance, msmEngine: MetalMSM? = nil) {
        self.ccs = ccs
        self.engine = HyperNovaEngine(ccs: ccs, msmEngine: msmEngine)
    }

    /// Initialize prover with pre-existing engine.
    public init(engine: HyperNovaEngine) {
        self.engine = engine
        self.ccs = engine.ccs
    }

    // MARK: - Commit Witness

    /// Commit a witness vector using Pedersen commitment.
    /// Returns the committed instance (CCCS) with cached affine coordinates.
    public func commitWitness(_ witness: [Fr], publicInput: [Fr]) -> CommittedCCSInstance {
        let commitment = engine.pp.commit(witness: witness)
        let (ax, ay) = engine.commitmentToAffineFr(commitment)
        return CommittedCCSInstance(commitment: commitment, publicInput: publicInput,
                                   affineX: ax, affineY: ay)
    }

    // MARK: - Initialize (base case)

    /// Create the initial running instance from a fresh witness.
    /// This is the base case before any folding occurs.
    public func initialize(witness: [Fr], publicInput: [Fr]) -> (CommittedCCSInstance, [Fr]) {
        let lcccs = engine.initialize(witness: witness, publicInput: publicInput)
        return (CommittedCCSInstance(from: lcccs), witness)
    }

    // MARK: - Two-Instance Fold

    /// Fold a single new instance into the running instance.
    /// This is the standard HyperNova 2-fold (LCCCS + CCCS -> LCCCS).
    ///
    /// - Parameters:
    ///   - running: The current accumulated instance (must be relaxed/LCCCS)
    ///   - runningWitness: Witness for the running instance
    ///   - new: Fresh instance to fold in
    ///   - newWitness: Witness for the new instance
    /// - Returns: (folded instance, folded witness, proof)
    public func fold(running: CommittedCCSInstance, runningWitness: [Fr],
                     new: CommittedCCSInstance, newWitness: [Fr])
        -> (CommittedCCSInstance, [Fr], FoldingProof)
    {
        let lcccs = running.toLCCCS()
        let cccs = new.toCCCS()

        let (folded, foldedWit, proof) = engine.fold(
            running: lcccs, runningWitness: runningWitness,
            new: cccs, newWitness: newWitness)

        return (CommittedCCSInstance(from: folded), foldedWit, proof)
    }

    // MARK: - Multi-Instance Fold

    /// Fold N instances into 1 in a single step.
    ///
    /// Given instances = [I_0, I_1, ..., I_{N-1}] where I_0 is the running LCCCS
    /// and I_1..I_{N-1} are fresh CCCS instances:
    ///
    ///   1. Compute sigma/theta evaluations for each instance
    ///   2. Build transcript, get challenges rho_1, ..., rho_{N-1}
    ///   3. Fold: C' = C_0 + sum_i rho_i * C_i
    ///            u' = u_0 + sum_i rho_i
    ///            x' = x_0 + sum_i rho_i * x_i
    ///            v' = sigma_0 + sum_i rho_i * theta_i
    ///            w' = w_0 + sum_i rho_i * w_i
    ///   4. Produce sumcheck proof for cross-term consistency
    ///
    /// - Parameters:
    ///   - instances: Array of committed instances. First must be relaxed (LCCCS).
    ///   - witnesses: Corresponding witness arrays.
    /// - Returns: (folded instance, folded witness, multi-fold proof)
    public func multiFold(instances: [CommittedCCSInstance], witnesses: [[Fr]])
        -> (CommittedCCSInstance, [Fr], MultiFoldProof)
    {
        precondition(instances.count >= 2, "Need at least 2 instances to fold")
        precondition(instances.count == witnesses.count)
        precondition(instances[0].isRelaxed, "First instance must be relaxed (LCCCS)")

        let n = instances.count
        let t = ccs.t

        // For 2-instance case, use the optimized path
        if n == 2 {
            let (folded, foldedWit, proof) = fold(
                running: instances[0], runningWitness: witnesses[0],
                new: instances[1], newWitness: witnesses[1])
            let multiProof = MultiFoldProof(
                sigmas: [proof.sigmas],
                thetas: [proof.thetas],
                sumcheckProof: proof.sumcheckProof,
                instanceCount: 2)
            return (folded, foldedWit, multiProof)
        }

        // Build z vectors for all instances
        let running = instances[0]
        var zVecs = [[Fr]]()
        zVecs.reserveCapacity(n)
        // Running instance: z = [u, x, w]
        var z0 = [running.u]
        z0.append(contentsOf: running.publicInput)
        z0.append(contentsOf: witnesses[0])
        zVecs.append(z0)
        // New instances: z = [1, x, w]
        for i in 1..<n {
            var zi = [Fr.one]
            zi.append(contentsOf: instances[i].publicInput)
            zi.append(contentsOf: witnesses[i])
            zVecs.append(zi)
        }

        // Step 1: Compute M_j * z_i for all matrices j and instances i
        var allMatVecs = [[[Fr]]]()  // allMatVecs[instance][matrix] = vector
        allMatVecs.reserveCapacity(n)
        for i in 0..<n {
            var mvs = [[Fr]]()
            mvs.reserveCapacity(t)
            for j in 0..<t {
                mvs.append(ccs.matrices[j].mulVec(zVecs[i]))
            }
            allMatVecs.append(mvs)
        }

        // Step 2: Compute sigma/theta evaluations via MLE at running.r
        let r = running.r
        var allSigmas = [[Fr]]()  // allSigmas[instance][matrix]
        allSigmas.reserveCapacity(n)
        for i in 0..<n {
            var evals = [Fr](repeating: .zero, count: t)
            for j in 0..<t {
                let padded = engine.isPow2M ? allMatVecs[i][j] : padToPow2(allMatVecs[i][j])
                evals[j] = cMleEvalFold(evals: padded, point: r)
            }
            allSigmas.append(evals)
        }

        // Step 3: Build Fiat-Shamir transcript
        let transcript = Transcript(label: "hypernova-multifold", backend: .keccak256)

        // Absorb all instances
        engine.absorbLCCCS(transcript, running.toLCCCS())
        for i in 1..<n {
            engine.absorbCCCS(transcript, instances[i].toCCCS())
        }

        // Absorb all evaluations
        for i in 0..<n {
            for j in 0..<t {
                transcript.absorb(allSigmas[i][j])
            }
        }

        // Squeeze N-1 challenges: rho_1, ..., rho_{N-1}
        let rhos = transcript.squeezeN(n - 1)

        // Step 4: Fold everything using random linear combination
        // C' = C_0 + sum_i rho_i * C_i
        var foldedCommitment = running.commitment
        for i in 1..<n {
            let rhoCi = cPointScalarMul(instances[i].commitment, rhos[i - 1])
            foldedCommitment = pointAdd(foldedCommitment, rhoCi)
        }

        // u' = u_0 + sum_i rho_i (fresh instances have u=1, so rho_i * 1 = rho_i)
        var foldedU = running.u
        for i in 0..<(n - 1) {
            foldedU = frAdd(foldedU, rhos[i])
        }

        // x' = x_0 + sum_i rho_i * x_i
        let numPub = running.publicInput.count
        var foldedPublicInput = [Fr](repeating: .zero, count: numPub)
        for k in 0..<numPub {
            foldedPublicInput[k] = running.publicInput[k]
        }
        for i in 1..<n {
            for k in 0..<numPub {
                foldedPublicInput[k] = frAdd(foldedPublicInput[k],
                                             frMul(rhos[i - 1], instances[i].publicInput[k]))
            }
        }

        // v' = sigma_0 + sum_i rho_i * sigma_i
        var foldedV = [Fr](repeating: .zero, count: t)
        for j in 0..<t {
            foldedV[j] = allSigmas[0][j]
        }
        for i in 1..<n {
            for j in 0..<t {
                foldedV[j] = frAdd(foldedV[j], frMul(rhos[i - 1], allSigmas[i][j]))
            }
        }

        // w' = w_0 + sum_i rho_i * w_i
        let witLen = witnesses[0].count
        var foldedWitness = Array(witnesses[0])
        for i in 1..<n {
            for k in 0..<witLen {
                foldedWitness[k] = frAdd(foldedWitness[k], frMul(rhos[i - 1], witnesses[i][k]))
            }
        }

        // Step 5: Compute cross-term sumcheck proof
        let sumcheckProof = computeMultiFoldSumcheck(
            allMatVecs: allMatVecs, rhos: rhos, r: r, transcript: transcript)

        // Build folded instance
        let (ax, ay) = engine.commitmentToAffineFr(foldedCommitment)
        let folded = CommittedCCSInstance(
            commitment: foldedCommitment, publicInput: foldedPublicInput,
            u: foldedU, r: r, v: foldedV,
            affineX: ax, affineY: ay)

        let proof = MultiFoldProof(
            sigmas: [allSigmas[0]],
            thetas: Array(allSigmas[1...]),
            sumcheckProof: sumcheckProof,
            instanceCount: n)

        return (folded, foldedWitness, proof)
    }

    // MARK: - IVC Chain

    /// Run an IVC (incremental verifiable computation) chain.
    ///
    /// Takes a sequence of (publicInput, witness) pairs representing sequential
    /// computation steps, and folds them all into a single accumulated instance.
    ///
    /// - Parameters:
    ///   - steps: Array of (publicInput, witness) for each step
    /// - Returns: (final accumulated instance, final witness)
    public func ivcChain(steps: [(publicInput: [Fr], witness: [Fr])])
        -> (CommittedCCSInstance, [Fr])
    {
        precondition(!steps.isEmpty, "Need at least one step")

        // Initialize with first step
        var (running, runningWit) = initialize(
            witness: steps[0].witness, publicInput: steps[0].publicInput)

        // Fold each subsequent step
        for i in 1..<steps.count {
            let newInstance = commitWitness(steps[i].witness, publicInput: steps[i].publicInput)
            let (folded, foldedWit, _) = fold(
                running: running, runningWitness: runningWit,
                new: newInstance, newWitness: steps[i].witness)
            running = folded
            runningWit = foldedWit
        }

        return (running, runningWit)
    }

    // MARK: - Decide (final verification)

    /// Verify the final accumulated instance (the "decider").
    /// Checks commitment opening, MLE evaluations, and CCS relation.
    public func decide(instance: CommittedCCSInstance, witness: [Fr]) -> Bool {
        precondition(instance.isRelaxed, "Can only decide on relaxed instances")
        return engine.decide(lcccs: instance.toLCCCS(), witness: witness)
    }

    // MARK: - Internal: Multi-fold Sumcheck

    /// Compute sumcheck proof for multi-fold cross-terms.
    ///
    /// The cross-term for N instances with CCS multisets of degree d:
    /// For each pair (i, k) where i != k, we have cross contributions from
    /// the Hadamard product terms in the CCS.
    ///
    /// For the R1CS case (degree 2, one multiset {A, B}):
    ///   cross(x) = sum_{i<k} rho_i * rho_k * (M_A*z_i)(x) * (M_B*z_k)(x)
    ///            + sum_{i<k} rho_i * rho_k * (M_A*z_k)(x) * (M_B*z_i)(x)
    ///
    /// We weight by eq(r, x) and run sumcheck to prove the sum.
    func computeMultiFoldSumcheck(
        allMatVecs: [[[Fr]]], rhos: [Fr], r: [Fr],
        transcript: Transcript) -> SumcheckFoldProof
    {
        let numRounds = engine.logM
        let size = 1 << engine.logM
        let n = allMatVecs.count

        var crossTermEvals = [Fr](repeating: .zero, count: size)

        // Build cross-terms from all pairs
        for j in 0..<ccs.q {
            let sj = ccs.multisets[j]
            if sj.count < 2 { continue }

            if sj.count == 2 {
                let mA = sj[0], mB = sj[1]
                let cj = ccs.coefficients[j]

                // All pairs (i, k) where 0 <= i, 1 <= k < n, i != k
                // Running instance (i=0) has implicit rho_0 = 1
                for k in 1..<n {
                    let rhoK = rhos[k - 1]
                    let rhoTimesC = frMul(rhoK, cj)
                    let limit = min(ccs.m, size)
                    // Cross: running with new[k]
                    let mAz0 = allMatVecs[0][mA]
                    let mAzK = allMatVecs[k][mA]
                    let mBz0 = allMatVecs[0][mB]
                    let mBzK = allMatVecs[k][mB]
                    for x in 0..<limit {
                        let cross = frAdd(frMul(mAz0[x], mBzK[x]),
                                          frMul(mAzK[x], mBz0[x]))
                        crossTermEvals[x] = frAdd(crossTermEvals[x], frMul(rhoTimesC, cross))
                    }
                }

                // Cross-terms between new instances (k1, k2) where k1 < k2
                for k1 in 1..<n {
                    for k2 in (k1 + 1)..<n {
                        let rhoK1K2 = frMul(rhos[k1 - 1], rhos[k2 - 1])
                        let rhoTimesC = frMul(rhoK1K2, cj)
                        let limit = min(ccs.m, size)
                        let mAzK1 = allMatVecs[k1][mA]
                        let mAzK2 = allMatVecs[k2][mA]
                        let mBzK1 = allMatVecs[k1][mB]
                        let mBzK2 = allMatVecs[k2][mB]
                        for x in 0..<limit {
                            let cross = frAdd(frMul(mAzK1[x], mBzK2[x]),
                                              frMul(mAzK2[x], mBzK1[x]))
                            crossTermEvals[x] = frAdd(crossTermEvals[x], frMul(rhoTimesC, cross))
                        }
                    }
                }
            }
        }

        // Weight by eq(r, x) using C-accelerated eq poly
        var eqR = [Fr](repeating: Fr.zero, count: size)
        r.withUnsafeBytes { ptBuf in
            eqR.withUnsafeMutableBytes { evalBuf in
                gkr_eq_poly(
                    ptBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(r.count),
                    evalBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                )
            }
        }
        crossTermEvals.withUnsafeMutableBytes { rBuf in
            eqR.withUnsafeBytes { eBuf in
                bn254_fr_batch_mul(
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    eBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(size))
            }
        }

        // Run sumcheck rounds in-place
        var roundPolys = [[Fr]]()
        roundPolys.reserveCapacity(numRounds)
        var currentSize = size

        for _ in 0..<numRounds {
            let half = currentSize >> 1
            var s0 = Fr.zero
            var s1 = Fr.zero
            for j in 0..<half {
                s0 = frAdd(s0, crossTermEvals[2 * j])
                s1 = frAdd(s1, crossTermEvals[2 * j + 1])
            }
            roundPolys.append([s0, s1])

            transcript.absorb(s0)
            transcript.absorb(s1)
            let challenge = transcript.squeeze()

            // C-accelerated interleaved fold in-place
            crossTermEvals.withUnsafeMutableBytes { buf in
                withUnsafeBytes(of: challenge) { chBuf in
                    bn254_fr_fold_interleaved_inplace(
                        buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        chBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(half))
                }
            }
            currentSize = half
        }

        let finalEval = currentSize > 0 ? crossTermEvals[0] : Fr.zero
        return SumcheckFoldProof(roundPolys: roundPolys, finalEval: finalEval)
    }
}
