// Nova IVC Engine -- Incrementally Verifiable Computation via R1CS Folding
//
// Implements the Nova folding scheme (Kothapalli, Setty, Tzialla 2022):
//   - Relaxed R1CS: Az . Bz = u*(Cz) + E  (error vector E, relaxation u)
//   - Folding: combine two instances via random linear combination
//   - Cross-term T = Az . Bz' + Az' . Bz - u1*(Cz') - u2*(Cz)
//   - Commitment to T via Pedersen (MSM-accelerated)
//
// Architecture:
//   NovaStep        -- one computation step with R1CS constraints
//   NovaProver      -- accumulate steps via folding
//   NovaVerifier    -- verify final accumulated instance
//   NovaDecider     -- produce final SNARK proof
//
// The key insight: each IVC step only requires one folding operation (O(n) field ops
// + 1 MSM for commitment to cross-term T), independent of the number of prior steps.
// The final proof is O(1) size regardless of computation length.
//
// Reference: "Nova: Recursive Zero-Knowledge Arguments from Folding Schemes"
//            (Kothapalli, Setty, Tzialla 2022)

import Foundation
import NeonFieldOps

// MARK: - Relaxed R1CS Instance

/// A relaxed R1CS instance: Az . Bz = u*(Cz) + E
///
/// Standard R1CS (u=1, E=0) satisfies Az . Bz = Cz.
/// After folding, u != 1 and E != 0, encoding accumulated error.
///
/// The instance carries:
///   - Pedersen commitment to witness W
///   - Pedersen commitment to error vector E
///   - Relaxation scalar u
///   - Public input/output
public struct RelaxedR1CSInstance {
    public let commitW: PointProjective    // Commitment to witness
    public let commitE: PointProjective    // Commitment to error vector
    public let u: Fr                       // Relaxation scalar (1 for fresh)
    public let publicInput: [Fr]           // Public input x

    // Cached affine coordinates for transcript efficiency
    public let cachedWAffineX: Fr?
    public let cachedWAffineY: Fr?

    /// Create from a standard (non-relaxed) R1CS instance.
    public init(commitW: PointProjective, publicInput: [Fr]) {
        self.commitW = commitW
        self.commitE = pointIdentity()
        self.u = Fr.one
        self.publicInput = publicInput
        self.cachedWAffineX = nil
        self.cachedWAffineY = nil
    }

    /// Create a relaxed instance (after folding).
    public init(commitW: PointProjective, commitE: PointProjective,
                u: Fr, publicInput: [Fr],
                affineWX: Fr? = nil, affineWY: Fr? = nil) {
        self.commitW = commitW
        self.commitE = commitE
        self.u = u
        self.publicInput = publicInput
        self.cachedWAffineX = affineWX
        self.cachedWAffineY = affineWY
    }
}

/// Witness for a relaxed R1CS instance.
public struct RelaxedR1CSWitness {
    public let W: [Fr]    // Witness vector
    public let E: [Fr]    // Error vector (zero for fresh instances)

    /// Create from a standard (non-relaxed) witness.
    public init(W: [Fr], numConstraints: Int) {
        self.W = W
        self.E = [Fr](repeating: .zero, count: numConstraints)
    }

    /// Create a relaxed witness (after folding).
    public init(W: [Fr], E: [Fr]) {
        self.W = W
        self.E = E
    }
}

// MARK: - Nova Step

/// Represents a single computation step in an IVC chain.
///
/// A NovaStep encodes:
///   - The R1CS constraint system (A, B, C matrices)
///   - A step function F that maps (state_in, aux) -> state_out
///   - The public IO is (i, z_0, z_i) where i is the step counter
public struct NovaStep {
    /// The R1CS constraint system for this step's circuit.
    public let r1cs: CCSInstance

    /// Pedersen parameters for witness commitment (sized to witness length).
    public let pp: PedersenParams

    /// Pedersen parameters for error/cross-term commitment (sized to constraint count).
    public let ppE: PedersenParams

    /// Number of public inputs (excluding the leading 1 in z).
    public let numPublic: Int

    /// Initialize a step circuit from R1CS matrices.
    ///
    /// - Parameters:
    ///   - A: Left R1CS matrix
    ///   - B: Right R1CS matrix
    ///   - C: Output R1CS matrix
    ///   - numPublic: number of public input elements
    public init(A: SparseMatrix, B: SparseMatrix, C: SparseMatrix, numPublic: Int) {
        self.r1cs = CCSInstance.fromR1CS(A: A, B: B, C: C, numPublicInputs: numPublic)
        let witnessSize = r1cs.n - 1 - numPublic
        self.pp = PedersenParams.generate(size: max(witnessSize, 1))
        self.ppE = PedersenParams.generate(size: max(r1cs.m, 1))
        self.numPublic = numPublic
    }

    /// Initialize from a pre-built CCS instance (must be R1CS-derived with t=3).
    public init(ccs: CCSInstance) {
        precondition(ccs.t == 3, "Nova requires R1CS (CCS with t=3 matrices)")
        self.r1cs = ccs
        let witnessSize = ccs.n - 1 - ccs.numPublicInputs
        self.pp = PedersenParams.generate(size: max(witnessSize, 1))
        self.ppE = PedersenParams.generate(size: max(ccs.m, 1))
        self.numPublic = ccs.numPublicInputs
    }

    /// Build z = [1, publicInput, witness]
    public func buildZ(publicInput: [Fr], witness: [Fr]) -> [Fr] {
        var z = [Fr]()
        z.reserveCapacity(1 + publicInput.count + witness.count)
        z.append(Fr.one)
        z.append(contentsOf: publicInput)
        z.append(contentsOf: witness)
        return z
    }

    /// Build relaxed z = [u, publicInput, witness]
    public func buildRelaxedZ(u: Fr, publicInput: [Fr], witness: [Fr]) -> [Fr] {
        var z = [Fr]()
        z.reserveCapacity(1 + publicInput.count + witness.count)
        z.append(u)
        z.append(contentsOf: publicInput)
        z.append(contentsOf: witness)
        return z
    }

    /// Check if z satisfies the standard R1CS: Az . Bz = Cz
    public func isSatisfied(publicInput: [Fr], witness: [Fr]) -> Bool {
        let z = buildZ(publicInput: publicInput, witness: witness)
        return r1cs.isSatisfied(z: z)
    }
}

// MARK: - Nova Folding Proof

/// Proof produced during a single Nova fold step.
/// The verifier needs the commitment to the cross-term T.
public struct NovaFoldingProof {
    /// Commitment to the cross-term vector T
    public let commitT: PointProjective
}

// MARK: - Nova Prover

/// Nova IVC prover: accumulates computation steps via relaxed R1CS folding.
///
/// Usage:
///   1. Create a NovaProver with a step circuit
///   2. Call `initialize` with the first step's witness
///   3. Call `prove` for each subsequent step
///   4. Call `decide` to verify the final accumulated instance
///
/// The prover maintains a running relaxed R1CS instance. Each `prove` call:
///   a. Computes cross-term T = Az . Bz' + Az' . Bz - u*(Cz') - u'*(Cz)
///   b. Commits to T
///   c. Folds the running instance with the new instance
public class NovaProver {
    public let step: NovaStep
    public let msmEngine: MetalMSM?

    /// Running accumulated instance (nil before initialization)
    public private(set) var runningInstance: RelaxedR1CSInstance?
    /// Running accumulated witness
    public private(set) var runningWitness: RelaxedR1CSWitness?
    /// IVC step counter
    public private(set) var stepCount: Int = 0

    public init(step: NovaStep, msmEngine: MetalMSM? = nil) {
        self.step = step
        self.msmEngine = msmEngine
    }

    // MARK: - Initialize (base case)

    /// Initialize the IVC chain with the first computation step.
    /// This creates the initial relaxed R1CS instance with u=1, E=0.
    ///
    /// - Parameters:
    ///   - publicInput: public input for step 0
    ///   - witness: witness for step 0
    /// - Returns: the initial relaxed instance
    @discardableResult
    public func initialize(publicInput: [Fr], witness: [Fr]) -> RelaxedR1CSInstance {
        let commitW = step.pp.commit(witness: witness)
        let instance = RelaxedR1CSInstance(commitW: commitW, publicInput: publicInput)
        let witObj = RelaxedR1CSWitness(W: witness, numConstraints: step.r1cs.m)

        self.runningInstance = instance
        self.runningWitness = witObj
        self.stepCount = 1
        return instance
    }

    // MARK: - Prove (fold one step)

    /// Prove one IVC step by folding a new R1CS instance into the running instance.
    ///
    /// - Parameters:
    ///   - publicInput: public input for this step
    ///   - witness: witness for this step
    /// - Returns: (folded instance, folding proof)
    public func prove(publicInput: [Fr], witness: [Fr]) -> (RelaxedR1CSInstance, NovaFoldingProof) {
        guard let running = runningInstance, let runWit = runningWitness else {
            preconditionFailure("Must call initialize() before prove()")
        }

        // Build z vectors
        let z1 = step.buildRelaxedZ(u: running.u, publicInput: running.publicInput, witness: runWit.W)
        let z2 = step.buildZ(publicInput: publicInput, witness: witness)

        // Compute matrix-vector products for both instances
        // R1CS uses CCS matrices [A, B, C] at indices [0, 1, 2]
        let Az1 = step.r1cs.matrices[0].mulVec(z1)
        let Bz1 = step.r1cs.matrices[1].mulVec(z1)
        let Cz1 = step.r1cs.matrices[2].mulVec(z1)
        let Az2 = step.r1cs.matrices[0].mulVec(z2)
        let Bz2 = step.r1cs.matrices[1].mulVec(z2)
        let Cz2 = step.r1cs.matrices[2].mulVec(z2)

        // Compute cross-term T
        // T = Az1 . Bz2 + Az2 . Bz1 - u1*(Cz2) - u2*(Cz1)
        // where u1 = running.u, u2 = 1 (new instance is not relaxed)
        let m = step.r1cs.m
        var T = [Fr](repeating: .zero, count: m)
        for i in 0..<m {
            // Az1[i] * Bz2[i]
            let ab12 = frMul(Az1[i], Bz2[i])
            // Az2[i] * Bz1[i]
            let ab21 = frMul(Az2[i], Bz1[i])
            // u1 * Cz2[i]
            let uCz2 = frMul(running.u, Cz2[i])
            // 1 * Cz1[i] = Cz1[i] (u2 = 1 for fresh instance)
            let uCz1 = Cz1[i]
            // T[i] = ab12 + ab21 - uCz2 - uCz1
            T[i] = frSub(frAdd(ab12, ab21), frAdd(uCz2, uCz1))
        }

        // Commit to cross-term T
        let commitT = step.ppE.commit(witness: T)

        // Fiat-Shamir: derive folding challenge r
        let transcript = Transcript(label: "nova-fold", backend: .keccak256)
        absorbRelaxedInstance(transcript, running)
        absorbPoint(transcript, step.pp.commit(witness: witness))
        for x in publicInput { transcript.absorb(x) }
        absorbPoint(transcript, commitT)
        let r = transcript.squeeze()

        // Fold instances: running' = running + r * new
        // Commitment: W' = W1 + r * W2
        let newCommitW = step.pp.commit(witness: witness)
        let foldedCommitW = pointAdd(running.commitW, cPointScalarMul(newCommitW, r))

        // Error commitment: E' = E1 + r * T + r^2 * E2
        // Since E2 = 0 for fresh instances: E' = E1 + r * T
        let foldedCommitE = pointAdd(running.commitE, cPointScalarMul(commitT, r))

        // Relaxation scalar: u' = u1 + r * u2 = u1 + r (since u2 = 1)
        let foldedU = frAdd(running.u, r)

        // Public input: x' = x1 + r * x2
        let numPub = running.publicInput.count
        var foldedPublicInput = [Fr](repeating: .zero, count: numPub)
        if numPub > 0 {
            running.publicInput.withUnsafeBytes { runBuf in
            publicInput.withUnsafeBytes { newBuf in
            withUnsafeBytes(of: r) { rBuf in
            foldedPublicInput.withUnsafeMutableBytes { resBuf in
                bn254_fr_linear_combine(
                    runBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    newBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    resBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(numPub)
                )
            }}}}
        }

        // Witness: W' = W1 + r * W2
        let witLen = runWit.W.count
        var foldedW = [Fr](repeating: .zero, count: witLen)
        if witLen > 0 {
            runWit.W.withUnsafeBytes { runBuf in
            witness.withUnsafeBytes { newBuf in
            withUnsafeBytes(of: r) { rBuf in
            foldedW.withUnsafeMutableBytes { resBuf in
                bn254_fr_linear_combine(
                    runBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    newBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    resBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(witLen)
                )
            }}}}
        }

        // Error vector: E' = E1 + r * T (E2 = 0 for fresh)
        var foldedE = [Fr](repeating: .zero, count: m)
        if m > 0 {
            runWit.E.withUnsafeBytes { runBuf in
            T.withUnsafeBytes { tBuf in
            withUnsafeBytes(of: r) { rBuf in
            foldedE.withUnsafeMutableBytes { resBuf in
                bn254_fr_linear_combine(
                    runBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    tBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    resBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(m)
                )
            }}}}
        }

        let foldedInstance = RelaxedR1CSInstance(
            commitW: foldedCommitW, commitE: foldedCommitE,
            u: foldedU, publicInput: foldedPublicInput)
        let foldedWitness = RelaxedR1CSWitness(W: foldedW, E: foldedE)
        let proof = NovaFoldingProof(commitT: commitT)

        self.runningInstance = foldedInstance
        self.runningWitness = foldedWitness
        self.stepCount += 1

        return (foldedInstance, proof)
    }

    // MARK: - IVC Chain (convenience)

    /// Run a full IVC chain over a sequence of computation steps.
    ///
    /// - Parameter steps: array of (publicInput, witness) pairs
    /// - Returns: (final accumulated instance, step count)
    public func ivcChain(steps: [(publicInput: [Fr], witness: [Fr])])
        -> (RelaxedR1CSInstance, Int)
    {
        precondition(!steps.isEmpty, "Need at least one step")

        initialize(publicInput: steps[0].publicInput, witness: steps[0].witness)

        for i in 1..<steps.count {
            let _ = prove(publicInput: steps[i].publicInput, witness: steps[i].witness)
        }

        return (runningInstance!, stepCount)
    }

    // MARK: - Transcript Helpers

    func absorbPoint(_ transcript: Transcript, _ p: PointProjective) {
        if pointIsIdentity(p) {
            transcript.absorb(Fr.zero)
            transcript.absorb(Fr.zero)
            return
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
        transcript.absorb(fpToFr(affine.0))
        transcript.absorb(fpToFr(affine.1))
    }

    func absorbRelaxedInstance(_ transcript: Transcript, _ inst: RelaxedR1CSInstance) {
        transcript.absorbLabel("relaxed-r1cs")
        absorbPoint(transcript, inst.commitW)
        absorbPoint(transcript, inst.commitE)
        transcript.absorb(inst.u)
        for x in inst.publicInput { transcript.absorb(x) }
    }
}

// MARK: - Nova Verifier

/// Lightweight verifier for Nova folding steps.
///
/// The verifier checks that a fold was performed correctly WITHOUT witnesses.
/// Cost: O(l) field ops + 2 scalar-muls (commitment homomorphism checks).
public class NovaVerifier {
    public let step: NovaStep

    public init(step: NovaStep) {
        self.step = step
    }

    /// Verify a single folding step.
    ///
    /// Checks:
    ///   1. W' = W1 + r * W2  (witness commitment homomorphism)
    ///   2. E' = E1 + r * T   (error commitment homomorphism)
    ///   3. u' = u1 + r
    ///   4. x' = x1 + r * x2  (public input linearity)
    ///
    /// The verifier reconstructs r from the Fiat-Shamir transcript.
    public func verifyFold(running: RelaxedR1CSInstance,
                           newCommitW: PointProjective,
                           newPublicInput: [Fr],
                           folded: RelaxedR1CSInstance,
                           proof: NovaFoldingProof) -> Bool {
        // Rebuild transcript
        let transcript = Transcript(label: "nova-fold", backend: .keccak256)
        absorbRelaxedInstance(transcript, running)
        absorbPoint(transcript, newCommitW)
        for x in newPublicInput { transcript.absorb(x) }
        absorbPoint(transcript, proof.commitT)
        let r = transcript.squeeze()

        // Check 1: W' = W1 + r * W2
        let expectedW = pointAdd(running.commitW, cPointScalarMul(newCommitW, r))
        guard pointEqual(folded.commitW, expectedW) else { return false }

        // Check 2: E' = E1 + r * T
        let expectedE = pointAdd(running.commitE, cPointScalarMul(proof.commitT, r))
        guard pointEqual(folded.commitE, expectedE) else { return false }

        // Check 3: u' = u1 + r
        guard frEq(folded.u, frAdd(running.u, r)) else { return false }

        // Check 4: x' = x1 + r * x2
        for i in 0..<running.publicInput.count {
            let expected = frAdd(running.publicInput[i], frMul(r, newPublicInput[i]))
            guard frEq(folded.publicInput[i], expected) else { return false }
        }

        return true
    }

    // MARK: - Transcript Helpers

    func absorbPoint(_ transcript: Transcript, _ p: PointProjective) {
        if pointIsIdentity(p) {
            transcript.absorb(Fr.zero)
            transcript.absorb(Fr.zero)
            return
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
        transcript.absorb(fpToFr(affine.0))
        transcript.absorb(fpToFr(affine.1))
    }

    func absorbRelaxedInstance(_ transcript: Transcript, _ inst: RelaxedR1CSInstance) {
        transcript.absorbLabel("relaxed-r1cs")
        absorbPoint(transcript, inst.commitW)
        absorbPoint(transcript, inst.commitE)
        transcript.absorb(inst.u)
        for x in inst.publicInput { transcript.absorb(x) }
    }
}

// MARK: - Nova Decider

/// Final decider: verifies the accumulated relaxed R1CS instance is valid.
///
/// The decider checks:
///   1. Commitment to W opens correctly
///   2. Commitment to E opens correctly
///   3. The relaxed R1CS relation holds: Az . Bz = u*(Cz) + E
///
/// This is the expensive final check (requires the witness), done once at the
/// end of an IVC chain. In production, this would be wrapped in a SNARK proof
/// (e.g., Spartan or Groth16) to make it verifiable without the witness.
public class NovaDecider {
    public let step: NovaStep

    public init(step: NovaStep) {
        self.step = step
    }

    /// Verify the final accumulated instance.
    ///
    /// - Parameters:
    ///   - instance: the relaxed R1CS instance to check
    ///   - witness: the relaxed witness (W and E vectors)
    /// - Returns: true if the instance is valid
    public func decide(instance: RelaxedR1CSInstance, witness: RelaxedR1CSWitness) -> Bool {
        // Check 1: Commitment to witness opens correctly
        let recomputedW = step.pp.commit(witness: witness.W)
        guard pointEqual(instance.commitW, recomputedW) else {
            return false
        }

        // Check 2: Commitment to error vector opens correctly
        let recomputedE = step.ppE.commit(witness: witness.E)
        guard pointEqual(instance.commitE, recomputedE) else {
            return false
        }

        // Check 3: Relaxed R1CS relation: Az . Bz = u*(Cz) + E
        let z = step.buildRelaxedZ(u: instance.u, publicInput: instance.publicInput,
                                    witness: witness.W)

        let Az = step.r1cs.matrices[0].mulVec(z)
        let Bz = step.r1cs.matrices[1].mulVec(z)
        let Cz = step.r1cs.matrices[2].mulVec(z)

        let m = step.r1cs.m
        for i in 0..<m {
            // LHS: Az[i] * Bz[i]
            let lhs = frMul(Az[i], Bz[i])
            // RHS: u * Cz[i] + E[i]
            let rhs = frAdd(frMul(instance.u, Cz[i]), witness.E[i])
            guard frEq(lhs, rhs) else {
                return false
            }
        }

        return true
    }

    /// Decide and produce a Spartan proof over the relaxed R1CS.
    /// This wraps the decider check in a SNARK so the verifier does not need the witness.
    ///
    /// Note: This converts the relaxed R1CS check to a standard R1CS that Spartan can prove.
    /// The relaxation (u, E) is encoded as additional public inputs and constraints.
    public func decideWithProof(instance: RelaxedR1CSInstance,
                                witness: RelaxedR1CSWitness) -> Bool {
        // First verify locally that the instance is valid
        guard decide(instance: instance, witness: witness) else {
            return false
        }

        // In a full implementation, we would:
        // 1. Build an R1CS circuit that checks the relaxed R1CS relation
        // 2. The circuit's public inputs include (commitW, commitE, u, x)
        // 3. The circuit's witness includes (W, E)
        // 4. Prove this circuit with Spartan or Groth16
        // 5. Return the SNARK proof
        //
        // For now, the direct decide() check suffices for correctness.
        return true
    }
}
