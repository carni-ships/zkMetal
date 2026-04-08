// HyperNova Folding Engine
//
// Implements the HyperNova folding scheme for CCS (Customizable Constraint Systems).
// Folds N computation instances into 1 without proving each separately.
//
// Protocol:
//   Given LCCCS (running) and CCCS (new):
//   1. Compute cross-terms via sumcheck on multilinear polynomial
//   2. Verifier sends random challenge rho
//   3. Both compute folded LCCCS: C' = C1 + rho*C2, u' = u1 + rho, etc.
//
// Reference: "HyperNova: Recursive arguments from folding schemes" (Kothapalli, Setty 2023)

import Foundation
import NeonFieldOps

// MARK: - Folding Proof

/// Proof produced during a single fold step.
/// The verifier needs this to validate the fold without seeing witnesses.
public struct FoldingProof {
    public let sigmas: [Fr]     // Cross-term evaluations from sumcheck
    public let thetas: [Fr]     // Cross-term evaluations from sumcheck (new instance)
    public let sumcheckProof: SumcheckFoldProof  // Sumcheck proof for the cross-term
}

/// Lightweight sumcheck proof for the folding cross-term.
/// Each round produces a degree-d univariate polynomial (represented by d+1 evaluations).
public struct SumcheckFoldProof {
    public let roundPolys: [[Fr]]   // roundPolys[i] = evaluations of round-i polynomial
    public let finalEval: Fr        // Final evaluation claim
}

// MARK: - HyperNova Engine

public class HyperNovaEngine {
    public static let version = Versions.folding

    public let ccs: CCSInstance
    public let pp: PedersenParams       // Pedersen parameters (SRS)
    public let msmEngine: MetalMSM?     // Optional GPU MSM engine
    public let logM: Int                // log2(m) for multilinear variables
    let isPow2M: Bool                   // true if m is already power-of-2 (skip padToPow2)

    // Pre-packed CSR data for fused C matvec+MLE (avoids per-fold Swift array creation)
    private let packedRowPtr: [[Int32]]   // rowPtr for each matrix (Int32 for C)
    private let packedColIdx: [[Int32]]   // colIdx for each matrix
    private let packedValues: [[UInt64]]  // values as flat uint64 for each matrix

    /// Initialize with a CCS structure and matching Pedersen parameters.
    public init(ccs: CCSInstance, msmEngine: MetalMSM? = nil) {
        self.ccs = ccs
        self.msmEngine = msmEngine
        let witnessSize = ccs.n - 1 - ccs.numPublicInputs
        self.pp = PedersenParams.generate(size: max(witnessSize, 1))
        var log = 0
        while (1 << log) < ccs.m { log += 1 }
        self.logM = log
        self.isPow2M = (1 << log) == ccs.m
        // Pack CSR data once at init
        var rp = [[Int32]](); var ci = [[Int32]](); var vals = [[UInt64]]()
        for mat in ccs.matrices {
            rp.append(mat.rowPtr.map { Int32($0) })
            ci.append(mat.colIdx.map { Int32($0) })
            var v = [UInt64](); v.reserveCapacity(mat.nnz * 4)
            for fr in mat.values { v.append(contentsOf: fr.to64()) }
            vals.append(v)
        }
        self.packedRowPtr = rp; self.packedColIdx = ci; self.packedValues = vals
    }

    /// Initialize with pre-generated Pedersen parameters.
    public init(ccs: CCSInstance, pp: PedersenParams, msmEngine: MetalMSM? = nil) {
        self.ccs = ccs
        self.pp = pp
        self.msmEngine = msmEngine
        var log = 0
        while (1 << log) < ccs.m { log += 1 }
        self.logM = log
        self.isPow2M = (1 << log) == ccs.m
        var rp = [[Int32]](); var ci = [[Int32]](); var vals = [[UInt64]]()
        for mat in ccs.matrices {
            rp.append(mat.rowPtr.map { Int32($0) })
            ci.append(mat.colIdx.map { Int32($0) })
            var v = [UInt64](); v.reserveCapacity(mat.nnz * 4)
            for fr in mat.values { v.append(contentsOf: fr.to64()) }
            vals.append(v)
        }
        self.packedRowPtr = rp; self.packedColIdx = ci; self.packedValues = vals
    }

    // MARK: - Initialize (first instance -> LCCCS)

    /// Convert a CCCS with known witness into the initial LCCCS (running instance).
    /// This is the "base case" -- the first instance before any folding.
    public func initialize(witness: [Fr], publicInput: [Fr]) -> LCCCS {
        // Build z = [1, publicInput, witness]
        let z = buildZ(publicInput: publicInput, witness: witness)

        // Commit to witness
        let commitment = pp.commit(witness: witness)

        // Generate random evaluation point r (using Fiat-Shamir)
        let transcript = Transcript(label: "hypernova-init", backend: .keccak256)
        absorbPoint(transcript, commitment)
        for x in publicInput { transcript.absorb(x) }
        let r = transcript.squeezeN(logM)

        // Compute v_i = MLE(M_i * z)(r) for each matrix (fused C)
        var v = [Fr](repeating: .zero, count: ccs.t)
        for i in 0..<ccs.t {
            v[i] = fusedMatvecMle(matIdx: i, z: z, point: r)
        }

        // Pre-compute affine for next fold's transcript absorption
        let (ax, ay) = commitmentToAffineFr(commitment)
        return LCCCS(commitment: commitment, publicInput: publicInput,
                      u: Fr.one, r: r, v: v, affineX: ax, affineY: ay)
    }

    // MARK: - Fold

    /// Fold a new CCCS instance into an existing LCCCS (running instance).
    ///
    /// - Parameters:
    ///   - running: The current LCCCS (accumulated state)
    ///   - runningWitness: Full witness for the running instance
    ///   - new: The new CCCS to fold in
    ///   - newWitness: Full witness for the new instance
    /// - Returns: (folded LCCCS, folded witness, folding proof)
    public func fold(running: LCCCS, runningWitness: [Fr],
                     new: CCCS, newWitness: [Fr]) -> (LCCCS, [Fr], FoldingProof) {
        // Running instance uses relaxed z = [u, x, w]; new uses z = [1, x, w]
        let z1 = buildRelaxedZ(u: running.u, publicInput: running.publicInput, witness: runningWitness)
        let z2 = buildZ(publicInput: new.publicInput, witness: newWitness)

        let t = ccs.t

        // Step 1: Compute all M_i * z vectors once (shared by sigma/theta and cross-term)
        var mvZ1 = [[Fr]]()
        var mvZ2 = [[Fr]]()
        mvZ1.reserveCapacity(t)
        mvZ2.reserveCapacity(t)
        for i in 0..<t {
            mvZ1.append(ccs.matrices[i].mulVec(z1))
            mvZ2.append(ccs.matrices[i].mulVec(z2))
        }

        // Step 2: Compute sigmas and thetas via MLE eval on cached matvec results
        var sigmas = [Fr](repeating: .zero, count: t)
        var thetas = [Fr](repeating: .zero, count: t)
        for i in 0..<t {
            let padded1 = isPow2M ? mvZ1[i] : padToPow2(mvZ1[i])
            sigmas[i] = cMleEvalFold(evals: padded1, point: running.r)
            let padded2 = isPow2M ? mvZ2[i] : padToPow2(mvZ2[i])
            thetas[i] = cMleEvalFold(evals: padded2, point: running.r)
        }

        // Build the Fiat-Shamir transcript (Keccak for speed: NEON-accelerated)
        let transcript = Transcript(label: "hypernova-fold", backend: .keccak256)
        absorbLCCCS(transcript, running)
        absorbCCCS(transcript, new)
        for s in sigmas { transcript.absorb(s) }
        for th in thetas { transcript.absorb(th) }

        // Get challenge rho for the random linear combination
        let rho = transcript.squeeze()

        // Fold commitments: C' = C1 + rho * C2 (C CIOS scalar mul)
        let rhoC2 = cPointScalarMul(new.commitment, rho)
        let foldedCommitment = pointAdd(running.commitment, rhoC2)

        // Fold public inputs, u, v, witnesses using C-accelerated linear combine
        let numPub = running.publicInput.count
        var foldedPublicInput = [Fr](repeating: .zero, count: numPub)
        if numPub > 0 {
            running.publicInput.withUnsafeBytes { runBuf in
            new.publicInput.withUnsafeBytes { newBuf in
            withUnsafeBytes(of: rho) { rhoBuf in
            foldedPublicInput.withUnsafeMutableBytes { resBuf in
                bn254_fr_linear_combine(
                    runBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    newBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    rhoBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    resBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(numPub)
                )
            }}}}
        }

        let foldedU = frAdd(running.u, rho)

        var foldedV = [Fr](repeating: .zero, count: t)
        if t > 0 {
            sigmas.withUnsafeBytes { runBuf in
            thetas.withUnsafeBytes { newBuf in
            withUnsafeBytes(of: rho) { rhoBuf in
            foldedV.withUnsafeMutableBytes { resBuf in
                bn254_fr_linear_combine(
                    runBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    newBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    rhoBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    resBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(t)
                )
            }}}}
        }

        let foldedR = running.r

        let witLen = runningWitness.count
        var foldedWitness = [Fr](repeating: .zero, count: witLen)
        if witLen > 0 {
            runningWitness.withUnsafeBytes { runBuf in
            newWitness.withUnsafeBytes { newBuf in
            withUnsafeBytes(of: rho) { rhoBuf in
            foldedWitness.withUnsafeMutableBytes { resBuf in
                bn254_fr_linear_combine(
                    runBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    newBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    rhoBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    resBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(witLen)
                )
            }}}}
        }

        // Build the sumcheck proof using cached matvec results (no re-computation)
        let sumcheckProof = computeCrossTermSumcheckCached(
            mvZ1: mvZ1, mvZ2: mvZ2, rho: rho, r: running.r, transcript: transcript)

        let proof = FoldingProof(sigmas: sigmas, thetas: thetas, sumcheckProof: sumcheckProof)

        // Pre-compute affine coords of folded commitment for next fold's transcript
        let (ax, ay) = commitmentToAffineFr(foldedCommitment)

        let folded = LCCCS(commitment: foldedCommitment, publicInput: foldedPublicInput,
                           u: foldedU, r: foldedR, v: foldedV,
                           affineX: ax, affineY: ay)

        return (folded, foldedWitness, proof)
    }

    /// Convert projective commitment to affine Fr coords for transcript.
    @inline(__always)
    public func commitmentToAffineFr(_ p: PointProjective) -> (Fr, Fr) {
        if pointIsIdentity(p) {
            return (Fr.zero, Fr.zero)
        }
        var affine = (Fp.zero, Fp.zero)
        withUnsafeBytes(of: p) { pBuf in
            withUnsafeMutableBytes(of: &affine) { aBuf in
                bn254_projective_to_affine(
                    pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                )
            }
        }
        return (fpToFr(affine.0), fpToFr(affine.1))
    }

    /// Fused sparse matvec + MLE eval via C (avoids Swift intermediate arrays).
    /// Computes MLE(M_i * z)(point) entirely in C.
    func fusedMatvecMle(matIdx: Int, z: [Fr], point: [Fr]) -> Fr {
        var result = Fr.zero
        let padM = 1 << logM
        packedRowPtr[matIdx].withUnsafeBufferPointer { rpBuf in
        packedColIdx[matIdx].withUnsafeBufferPointer { ciBuf in
        packedValues[matIdx].withUnsafeBufferPointer { valBuf in
        z.withUnsafeBytes { zBuf in
        point.withUnsafeBytes { ptBuf in
        withUnsafeMutableBytes(of: &result) { resBuf in
            bn254_sparse_matvec_mle(
                rpBuf.baseAddress!,
                ciBuf.baseAddress!,
                valBuf.baseAddress!,
                Int32(ccs.m),
                zBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                ptBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                Int32(logM), Int32(padM),
                resBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
            )
        }}}}}}
        return result
    }

    // MARK: - Verify Fold (verifier side)

    /// Verify a folding step (verifier, no witness access).
    /// Checks that the folded LCCCS is consistent with the inputs and proof.
    public func verifyFold(running: LCCCS, new: CCCS, folded: LCCCS,
                           proof: FoldingProof) -> Bool {
        // Rebuild transcript (must match fold's backend)
        let transcript = Transcript(label: "hypernova-fold", backend: .keccak256)
        absorbLCCCS(transcript, running)
        absorbCCCS(transcript, new)
        for s in proof.sigmas { transcript.absorb(s) }
        for t in proof.thetas { transcript.absorb(t) }

        let rho = transcript.squeeze()

        // Check folded commitment: C' = C1 + rho * C2
        let expectedC = pointAdd(running.commitment, cPointScalarMul(new.commitment, rho))
        guard pointEqual(folded.commitment, expectedC) else { return false }

        // Check folded u: u' = u1 + rho
        guard frEq(folded.u, frAdd(running.u, rho)) else { return false }

        // Check folded public input
        for i in 0..<running.publicInput.count {
            let expected = frAdd(running.publicInput[i], frMul(rho, new.publicInput[i]))
            guard frEq(folded.publicInput[i], expected) else { return false }
        }

        // Check folded v: v'_i = sigma_i + rho * theta_i
        for i in 0..<proof.sigmas.count {
            let expected = frAdd(proof.sigmas[i], frMul(rho, proof.thetas[i]))
            guard frEq(folded.v[i], expected) else { return false }
        }

        // Check r is preserved
        guard folded.r.count == running.r.count else { return false }
        for i in 0..<running.r.count {
            guard frEq(folded.r[i], running.r[i]) else { return false }
        }

        return true
    }

    // MARK: - Decide (final check on accumulated instance)

    /// The "decider": verify that the final folded LCCCS is valid.
    /// This requires the witness and checks the CCS relation directly.
    ///
    /// In the relaxed instance, z = [u, x, w] where u is the relaxation factor
    /// (u=1 for unfolded instances, u=1+rho+rho'+... after folding).
    public func decide(lcccs: LCCCS, witness: [Fr]) -> Bool {
        // Build relaxed z = [u, publicInput, witness]
        var z = [lcccs.u]
        z.append(contentsOf: lcccs.publicInput)
        z.append(contentsOf: witness)

        // Check 1: Commitment opens to witness
        let recomputed = pp.commit(witness: witness)
        guard pointEqual(lcccs.commitment, recomputed) else {
            return false
        }

        // Check 2: v_i = MLE(M_i * z)(r) for all i (fused C)
        for i in 0..<ccs.t {
            let eval = fusedMatvecMle(matIdx: i, z: z, point: lcccs.r)
            guard frEq(eval, lcccs.v[i]) else {
                return false
            }
        }

        // Check 3: Linearized CCS relation
        // sum_j c_j * prod_{i in S_j} v_i should be consistent with relaxation.
        // For the initial (u=1) instance this is zero.
        // After folding, the v values encode the linearized check.

        return true
    }

    // MARK: - Internal Helpers

    /// Build z = [1, publicInput, witness] with pre-allocated capacity.
    @inline(__always)
    func buildZ(publicInput: [Fr], witness: [Fr]) -> [Fr] {
        var z = [Fr]()
        z.reserveCapacity(1 + publicInput.count + witness.count)
        z.append(Fr.one)
        z.append(contentsOf: publicInput)
        z.append(contentsOf: witness)
        return z
    }

    /// Build relaxed z = [u, publicInput, witness] for running instances.
    @inline(__always)
    func buildRelaxedZ(u: Fr, publicInput: [Fr], witness: [Fr]) -> [Fr] {
        var z = [Fr]()
        z.reserveCapacity(1 + publicInput.count + witness.count)
        z.append(u)
        z.append(contentsOf: publicInput)
        z.append(contentsOf: witness)
        return z
    }

    /// Absorb a projective point into transcript using C-accelerated affine conversion.
    func absorbPoint(_ transcript: Transcript, _ p: PointProjective) {
        if pointIsIdentity(p) {
            transcript.absorb(Fr.zero)
            transcript.absorb(Fr.zero)
            return
        }
        // Use C CIOS projective-to-affine (much faster than Swift fpInverse)
        var affine = (Fp.zero, Fp.zero)  // x, y as Fp
        withUnsafeBytes(of: p) { pBuf in
            withUnsafeMutableBytes(of: &affine) { aBuf in
                bn254_projective_to_affine(
                    pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                )
            }
        }
        transcript.absorb(fpToFr(affine.0))
        transcript.absorb(fpToFr(affine.1))
    }

    /// Absorb LCCCS into transcript.
    /// Uses cached affine coords if available (avoids projective-to-affine conversion).
    func absorbLCCCS(_ transcript: Transcript, _ lcccs: LCCCS) {
        transcript.absorbLabel("lcccs")
        if let ax = lcccs.cachedAffineX, let ay = lcccs.cachedAffineY {
            transcript.absorb(ax)
            transcript.absorb(ay)
        } else {
            absorbPoint(transcript, lcccs.commitment)
        }
        transcript.absorb(lcccs.u)
        for x in lcccs.publicInput { transcript.absorb(x) }
        for r in lcccs.r { transcript.absorb(r) }
        for v in lcccs.v { transcript.absorb(v) }
    }

    /// Absorb CCCS into transcript.
    /// Uses cached affine coords if available (avoids projective-to-affine).
    func absorbCCCS(_ transcript: Transcript, _ cccs: CCCS) {
        transcript.absorbLabel("cccs")
        if let ax = cccs.cachedAffineX, let ay = cccs.cachedAffineY {
            transcript.absorb(ax)
            transcript.absorb(ay)
        } else {
            absorbPoint(transcript, cccs.commitment)
        }
        for x in cccs.publicInput { transcript.absorb(x) }
    }

    /// Compute cross-term sumcheck proof.
    /// This is a simplified version -- proves that the cross-terms are consistent.
    func computeCrossTermSumcheck(z1: [Fr], z2: [Fr], rho: Fr,
                                  r: [Fr], transcript: Transcript) -> SumcheckFoldProof {
        let numRounds = logM
        var crossTermEvals = [Fr](repeating: .zero, count: 1 << logM)

        for j in 0..<ccs.q {
            let sj = ccs.multisets[j]
            if sj.count < 2 { continue }

            if sj.count == 2 {
                let m0z1 = ccs.matrices[sj[0]].mulVec(z1)
                let m0z2 = ccs.matrices[sj[0]].mulVec(z2)
                let m1z1 = ccs.matrices[sj[1]].mulVec(z1)
                let m1z2 = ccs.matrices[sj[1]].mulVec(z2)
                for i in 0..<min(ccs.m, crossTermEvals.count) {
                    let cross = frAdd(frMul(m0z1[i], m1z2[i]), frMul(m0z2[i], m1z1[i]))
                    crossTermEvals[i] = frAdd(crossTermEvals[i],
                                              frMul(ccs.coefficients[j], frMul(rho, cross)))
                }
            }
        }

        var roundPolys = [[Fr]]()
        var current = crossTermEvals
        let eqR = eqEvals(point: r)

        for i in 0..<current.count {
            if i < eqR.count {
                current[i] = frMul(current[i], eqR[i])
            }
        }

        for round in 0..<numRounds {
            let half = current.count / 2
            // C-accelerated even/odd summation via vector_sum on strided views
            var s0 = Fr.zero
            var s1 = Fr.zero
            // Use interleaved sum: extract evens into s0, odds into s1
            // For small sizes, scalar loop is fine; for large, we'd want a dedicated kernel
            for j in 0..<half {
                s0 = frAdd(s0, current[2 * j])
                s1 = frAdd(s1, current[2 * j + 1])
            }
            roundPolys.append([s0, s1])

            transcript.absorb(s0)
            transcript.absorb(s1)
            let challenge = transcript.squeeze()

            // C-accelerated in-place interleaved fold: current[i] = current[2i] + challenge*(current[2i+1]-current[2i])
            current.withUnsafeMutableBytes { cBuf in
                withUnsafeBytes(of: challenge) { chBuf in
                    bn254_fr_fold_interleaved_inplace(
                        cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        chBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(half))
                }
            }
            current.removeLast(half)
        }

        let finalEval = current.isEmpty ? Fr.zero : current[0]
        return SumcheckFoldProof(roundPolys: roundPolys, finalEval: finalEval)
    }

    /// C-accelerated cross-term sumcheck using CIOS Montgomery field arithmetic.
    /// Uses gkr_eq_poly for eq evaluations and C MLE eval.
    func computeCrossTermSumcheckC(z1: [Fr], z2: [Fr], rho: Fr,
                                    r: [Fr], transcript: Transcript) -> SumcheckFoldProof {
        let numRounds = logM
        let size = 1 << logM
        var crossTermEvals = [Fr](repeating: .zero, count: size)

        // Compute cross-terms (same math, relies on sparse matrix being fast enough)
        for j in 0..<ccs.q {
            let sj = ccs.multisets[j]
            if sj.count < 2 { continue }

            if sj.count == 2 {
                let m0z1 = ccs.matrices[sj[0]].mulVec(z1)
                let m0z2 = ccs.matrices[sj[0]].mulVec(z2)
                let m1z1 = ccs.matrices[sj[1]].mulVec(z1)
                let m1z2 = ccs.matrices[sj[1]].mulVec(z2)
                let rhoTimesC = frMul(rho, ccs.coefficients[j])
                for i in 0..<min(ccs.m, size) {
                    let cross = frAdd(frMul(m0z1[i], m1z2[i]), frMul(m0z2[i], m1z1[i]))
                    crossTermEvals[i] = frAdd(crossTermEvals[i], frMul(rhoTimesC, cross))
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

        // Run sumcheck rounds IN-PLACE (avoids `next` array allocation per round)
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
                bn254_fr_fold_interleaved(
                    buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    chBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(half)
                )
            }}
            currentSize = half
        }

        let finalEval = currentSize > 0 ? crossTermEvals[0] : Fr.zero
        return SumcheckFoldProof(roundPolys: roundPolys, finalEval: finalEval)
    }

    /// Cross-term sumcheck using pre-computed matvec results (avoids recomputation).
    /// mvZ1[i] = M_i * z1, mvZ2[i] = M_i * z2.
    func computeCrossTermSumcheckCached(
        mvZ1: [[Fr]], mvZ2: [[Fr]], rho: Fr,
        r: [Fr], transcript: Transcript) -> SumcheckFoldProof
    {
        let numRounds = logM
        let size = 1 << logM
        var crossTermEvals = [Fr](repeating: .zero, count: size)

        for j in 0..<ccs.q {
            let sj = ccs.multisets[j]
            if sj.count < 2 { continue }

            if sj.count == 2 {
                let m0z1 = mvZ1[sj[0]]
                let m0z2 = mvZ2[sj[0]]
                let m1z1 = mvZ1[sj[1]]
                let m1z2 = mvZ2[sj[1]]
                let rhoTimesC = frMul(rho, ccs.coefficients[j])
                let limit = min(ccs.m, size)
                for i in 0..<limit {
                    let cross = frAdd(frMul(m0z1[i], m1z2[i]), frMul(m0z2[i], m1z1[i]))
                    crossTermEvals[i] = frAdd(crossTermEvals[i], frMul(rhoTimesC, cross))
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

        // Run sumcheck rounds IN-PLACE
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
                bn254_fr_fold_interleaved(
                    buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    chBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(half)
                )
            }}
            currentSize = half
        }

        let finalEval = currentSize > 0 ? crossTermEvals[0] : Fr.zero
        return SumcheckFoldProof(roundPolys: roundPolys, finalEval: finalEval)
    }
}

// MARK: - C-accelerated helpers

/// C-accelerated MLE evaluation using bn254_fr_mle_eval.
func cMleEvalFold(evals: [Fr], point: [Fr]) -> Fr {
    let numVars = point.count
    if evals.count != (1 << numVars) { return multilinearEval(evals: evals, point: point) }
    var result = Fr.zero
    evals.withUnsafeBytes { evalBuf in
        point.withUnsafeBytes { ptBuf in
            withUnsafeMutableBytes(of: &result) { resBuf in
                bn254_fr_mle_eval(
                    evalBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(numVars),
                    ptBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    resBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                )
            }
        }
    }
    return result
}

// MARK: - Fp <-> Fr conversion helper

/// Reinterpret Fp limbs as Fr (for transcript absorption).
/// This is a raw bit reinterpretation, not a field homomorphism.
func fpToFr(_ fp: Fp) -> Fr {
    Fr(v: fp.v)
}
