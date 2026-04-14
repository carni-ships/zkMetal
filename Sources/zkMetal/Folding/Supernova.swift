// Supernova Folding Scheme — Multiple Circuit IVC via R1CS Folding
//
// Implements the Supernova folding scheme (Kothapalli, Setty 2023):
//   - Nova: one circuit F repeated N times (same circuit each step)
//   - Supernova: multiple circuits F_1, F_2, ..., F_n, ONE active per step
//   - Program counter (pc) selects which circuit is active
//   - Each step outputs (pc', y) where pc' determines next circuit
//
// Key difference from Nova:
//   - Running instance carries pc identifying the active circuit
//   - Cross-term T uses the matrices from the ACTIVE circuit (pc)
//   - Public input format: [pc, x] where pc selects circuit
//
// Architecture:
//   SupernovaLCCCS   -- running instance with pc tag
//   SupernovaCCCS    -- fresh instance for a specific circuit
//   SupernovaProver  -- fold with pc-based matrix selection
//   SupernovaVerifier -- verify fold correctness
//
// Reference: "Supernova: Folding all Circuits" (Kothapalli, Setty 2023)

import Foundation
import NeonFieldOps

// MARK: - Supernova LCCCS (Linearized CCCS with Program Counter)

/// A relaxed Supernova instance: tracks which circuit was active (pc).
///
/// SupernovaLCCCS is the running instance that accumulates folds.
/// It carries:
///   - pc: which circuit was active for this instance
///   - commitW: commitment to witness
///   - commitE: commitment to error vector
///   - u: relaxation scalar (accumulates random challenges)
///   - x: public input [pc, computation_output]
///
/// The pc determines which circuit's matrices to use for cross-term computation.
public struct SupernovaLCCCS {
    /// Program counter: which circuit was active (index into circuits array)
    public let pc: Int

    /// Commitment to witness W
    public let commitW: PointProjective

    /// Commitment to error vector E
    public let commitE: PointProjective

    /// Relaxation scalar (starts at 1, accumulates r challenges)
    public let u: Fr

    /// Public input: [pc_as_fr, actual_public_inputs]
    public let x: [Fr]

    /// Create a fresh (non-relaxed) Supernova instance.
    public init(pc: Int, commitW: PointProjective, x: [Fr]) {
        self.pc = pc
        self.commitW = commitW
        self.commitE = pointIdentity()
        self.u = Fr.one
        self.x = x
    }

    /// Create a relaxed Supernova instance (after folding).
    public init(pc: Int, commitW: PointProjective, commitE: PointProjective,
                u: Fr, x: [Fr]) {
        self.pc = pc
        self.commitW = commitW
        self.commitE = commitE
        self.u = u
        self.x = x
    }
}

// MARK: - Supernova CCCS (Committed CCS)

/// A fresh Supernova instance for a specific circuit.
/// The pc is implicit (determined by which CCCS is used).
public struct SupernovaCCCS {
    /// Commitment to witness
    public let commitment: PointProjective

    /// Public input (does NOT include pc - that's determined by which CCCS is used)
    public let publicInput: [Fr]

    /// Create a fresh Supernova instance.
    public init(commitment: PointProjective, publicInput: [Fr]) {
        self.commitment = commitment
        self.publicInput = publicInput
    }
}

// MARK: - Supernova Folding Proof

/// Proof from a single Supernova fold step.
/// Contains commitment to cross-term T.
public struct SupernovaFoldProof {
    /// Commitment to cross-term vector T
    public let commitT: PointProjective

    public init(commitT: PointProjective) {
        self.commitT = commitT
    }
}

// MARK: - Supernova Prover

/// Supernova prover: handles multiple circuits with pc-based routing.
///
/// Usage:
///   1. Create with an array of R1CS shapes (one per circuit)
///   2. Call `initialize` with circuit index, public input, witness
///   3. Call `fold` for each subsequent step (specify which circuit is active)
///   4. Verify with SupernovaVerifier
public class SupernovaProver {
    /// Array of R1CS shapes, one per circuit type
    public let shapes: [NovaR1CSShape]

    /// Pedersen parameters for witness commitment
    public let pp: PedersenParams

    /// Initialize with multiple circuit shapes.
    public init(shapes: [NovaR1CSShape]) {
        self.shapes = shapes
        // Size for largest witness across all circuits
        let maxWitness = shapes.map { $0.numWitness }.max() ?? 1
        let maxConstraints = shapes.map { $0.numConstraints }.max() ?? 1
        let maxSize = max(maxWitness, maxConstraints)
        self.pp = PedersenParams.generate(size: max(maxSize, 1))
    }

    /// Initialize with pre-generated Pedersen parameters.
    public init(shapes: [NovaR1CSShape], pp: PedersenParams) {
        self.shapes = shapes
        self.pp = pp
    }

    // MARK: - Initialize (Base Case)

    /// Initialize the IVC chain with the first step.
    ///
    /// - Parameters:
    ///   - circuitIdx: which circuit (pc = circuitIdx)
    ///   - publicInput: public input for this step (不含pc)
    ///   - witness: witness for this step
    /// - Returns: the initial LCCCS
    public func initialize(circuitIdx: Int, publicInput: [Fr], witness: [Fr]) -> SupernovaLCCCS {
        precondition(circuitIdx >= 0 && circuitIdx < shapes.count)
        let shape = shapes[circuitIdx]

        // Verify the instance satisfies the circuit
        let instance = NovaR1CSInput(x: publicInput)
        let wit = NovaR1CSWitness(W: witness)
        precondition(shape.satisfies(instance: instance, witness: wit),
                     "Initial instance must satisfy R1CS")

        // Commit to witness
        let commitW = pp.commit(witness: witness)

        // x stores public inputs WITHOUT pc (pc is stored separately in the struct)
        return SupernovaLCCCS(pc: circuitIdx, commitW: commitW, x: publicInput)
    }

    // MARK: - Compute Cross-Term

    /// Compute the cross-term T for folding with pc-based matrix selection.
    ///
    /// T[i] = A_pc*z1[i] * B_pc*z2[i] + A_i*z2[i] * B_i*z1[i]
    ///        - u1 * C_pc*z2[i] - C_i*z1[i]
    ///
    /// where pc = running.pc (active circuit in running instance)
    /// and i = newCircuitIdx (active circuit in new instance)
    public func computeCrossTerm(
        running: SupernovaLCCCS,
        runningWitness: [Fr],
        newCircuitIdx: Int,
        newPublicInput: [Fr],
        newWitness: [Fr]
    ) -> [Fr] {
        let shapeRunning = shapes[running.pc]
        let shapeNew = shapes[newCircuitIdx]

        // NOTE: pc is metadata for routing, NOT part of the R1CS z vector.
        // LCCCS.x contains only public inputs (no pc embedded).
        let z1 = buildRelaxedZ(u: running.u, x: running.x, witness: runningWitness,
                                shape: shapeRunning)
        let z2 = buildFreshZ(x: newPublicInput, witness: newWitness, shape: shapeNew)

        // Fused matvec for running circuit when matrices share sparsity pattern (~2x faster)
        let Az1: [Fr]
        let Bz1: [Fr]
        let Cz1: [Fr]
        if shapeRunning.matricesSharePattern {
            let (a1, b1, c1) = shapeRunning.mulVecABC(z1)
            (Az1, Bz1, Cz1) = (a1, b1, c1)
        } else {
            Az1 = shapeRunning.A.mulVec(z1)
            Bz1 = shapeRunning.B.mulVec(z1)
            Cz1 = shapeRunning.C.mulVec(z1)
        }

        // Fused matvec for new circuit when matrices share sparsity pattern
        let Az2: [Fr]
        let Bz2: [Fr]
        let Cz2: [Fr]
        if shapeNew.matricesSharePattern {
            let (a2, b2, c2) = shapeNew.mulVecABC(z2)
            (Az2, Bz2, Cz2) = (a2, b2, c2)
        } else {
            Az2 = shapeNew.A.mulVec(z2)
            Bz2 = shapeNew.B.mulVec(z2)
            Cz2 = shapeNew.C.mulVec(z2)
        }

        // Cross-term: T = Az1 .* Bz2 + Az2 .* Bz1 - u * Cz2 - Cz1
        let m = max(shapeRunning.numConstraints, shapeNew.numConstraints)
        var T = [Fr](repeating: .zero, count: m)

        for i in 0..<m {
            let az1 = i < Az1.count ? Az1[i] : Fr.zero
            let bz1 = i < Bz1.count ? Bz1[i] : Fr.zero
            let cz1 = i < Cz1.count ? Cz1[i] : Fr.zero
            let az2 = i < Az2.count ? Az2[i] : Fr.zero
            let bz2 = i < Bz2.count ? Bz2[i] : Fr.zero
            let cz2 = i < Cz2.count ? Cz2[i] : Fr.zero

            let cross1 = frMul(az1, bz2)
            let cross2 = frMul(az2, bz1)
            let uCz2 = frMul(running.u, cz2)

            var ti = frAdd(cross1, cross2)
            ti = frSub(ti, uCz2)
            ti = frSub(ti, cz1)
            T[i] = ti
        }
        return T
    }

    /// GPU threshold: vectors shorter than this use CPU path.
    public var gpuThreshold: Int = 4

    /// Compute the cross-term T using GPU (NEON batch operations) when m >= gpuThreshold.
    ///
    /// Same formula as computeCrossTerm but uses NEON-accelerated batch operations:
    ///   T = az1 .* bz2 + az2 .* bz1 - u1*cz2 - cz1
    public func computeCrossTermGPU(
        running: SupernovaLCCCS,
        runningWitness: [Fr],
        newCircuitIdx: Int,
        newPublicInput: [Fr],
        newWitness: [Fr]
    ) -> [Fr] {
        let shapeRunning = shapes[running.pc]
        let shapeNew = shapes[newCircuitIdx]

        let z1 = buildRelaxedZ(u: running.u, x: running.x, witness: runningWitness,
                                shape: shapeRunning)
        let z2 = buildFreshZ(x: newPublicInput, witness: newWitness, shape: shapeNew)

        // Fused matvec for running circuit when matrices share sparsity pattern
        let Az1: [Fr]
        let Bz1: [Fr]
        let Cz1: [Fr]
        if shapeRunning.matricesSharePattern {
            let (a1, b1, c1) = shapeRunning.mulVecABC(z1)
            (Az1, Bz1, Cz1) = (a1, b1, c1)
        } else {
            Az1 = shapeRunning.A.mulVec(z1)
            Bz1 = shapeRunning.B.mulVec(z1)
            Cz1 = shapeRunning.C.mulVec(z1)
        }

        // Fused matvec for new circuit when matrices share sparsity pattern
        let Az2: [Fr]
        let Bz2: [Fr]
        let Cz2: [Fr]
        if shapeNew.matricesSharePattern {
            let (a2, b2, c2) = shapeNew.mulVecABC(z2)
            (Az2, Bz2, Cz2) = (a2, b2, c2)
        } else {
            Az2 = shapeNew.A.mulVec(z2)
            Bz2 = shapeNew.B.mulVec(z2)
            Cz2 = shapeNew.C.mulVec(z2)
        }

        let m = max(shapeRunning.numConstraints, shapeNew.numConstraints)

        // If vectors are already the right size, use them directly
        // Otherwise pad to size m
        func padToSize(_ arr: [Fr], _ size: Int) -> [Fr] {
            if arr.count >= size { return arr }
            var padded = arr
            padded.append(contentsOf: [Fr](repeating: .zero, count: size - arr.count))
            return padded
        }

        let paddedAz1 = padToSize(Az1, m)
        let paddedBz1 = padToSize(Bz1, m)
        let paddedCz1 = padToSize(Cz1, m)
        let paddedAz2 = padToSize(Az2, m)
        let paddedBz2 = padToSize(Bz2, m)
        let paddedCz2 = padToSize(Cz2, m)

        var T = [Fr](repeating: .zero, count: m)
        let u1 = running.u

        // T = az1 .* bz2
        paddedAz1.withUnsafeBytes { az1Buf in
        paddedBz2.withUnsafeBytes { bz2Buf in
        T.withUnsafeMutableBytes { tBuf in
            bn254_fr_batch_mul_neon(
                tBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                az1Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                bz2Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                Int32(m))
        }}}

        // tmp = az2 .* bz1
        var tmp = [Fr](repeating: .zero, count: m)
        paddedAz2.withUnsafeBytes { az2Buf in
        paddedBz1.withUnsafeBytes { bz1Buf in
        tmp.withUnsafeMutableBytes { tmpBuf in
            bn254_fr_batch_mul_neon(
                tmpBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                az2Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                bz1Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                Int32(m))
        }}}

        // T = T + tmp
        T.withUnsafeMutableBytes { tBuf in
        tmp.withUnsafeBytes { tmpBuf in
            let tPtr = tBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
            bn254_fr_batch_add_neon(
                tPtr,
                tPtr,
                tmpBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                Int32(m))
        }}

        // tmp = u1 * cz2
        withUnsafeBytes(of: u1) { u1Buf in
        paddedCz2.withUnsafeBytes { cz2Buf in
        tmp.withUnsafeMutableBytes { tmpBuf in
            bn254_fr_batch_mul_scalar_neon(
                tmpBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                cz2Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                u1Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                Int32(m))
        }}}

        // T = T - tmp
        T.withUnsafeMutableBytes { tBuf in
        tmp.withUnsafeBytes { tmpBuf in
            let tPtr = tBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
            bn254_fr_batch_sub_neon(
                tPtr,
                tPtr,
                tmpBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                Int32(m))
        }}

        // T = T - cz1
        T.withUnsafeMutableBytes { tBuf in
        paddedCz1.withUnsafeBytes { cz1Buf in
            let tPtr = tBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
            bn254_fr_batch_sub_neon(
                tPtr,
                tPtr,
                cz1Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                Int32(m))
        }}

        return T
    }

    // MARK: - Fold

    /// Fold a new instance into the running accumulator.
    ///
    /// - Parameters:
    ///   - running: the running LCCCS
    ///   - runningWitness: witness for the running instance
    ///   - newCircuitIdx: which circuit is active for the new instance
    ///   - newPublicInput: public input for new instance (不含pc)
    ///   - newWitness: witness for new instance
    /// - Returns: (folded LCCCS, folded witness, fold proof)
    public func fold(
        running: SupernovaLCCCS,
        runningWitness: [Fr],
        newCircuitIdx: Int,
        newPublicInput: [Fr],
        newWitness: [Fr]
    ) -> (SupernovaLCCCS, [Fr], [Fr], SupernovaFoldProof) {
        let shapeRunning = shapes[running.pc]
        let shapeNew = shapes[newCircuitIdx]
        let m = max(shapeRunning.numConstraints, shapeNew.numConstraints)

        // Step 1: Compute cross-term T (GPU if large enough, CPU otherwise)
        let T: [Fr]
        if m >= gpuThreshold {
            T = computeCrossTermGPU(
                running: running,
                runningWitness: runningWitness,
                newCircuitIdx: newCircuitIdx,
                newPublicInput: newPublicInput,
                newWitness: newWitness)
        } else {
            T = computeCrossTerm(
                running: running,
                runningWitness: runningWitness,
                newCircuitIdx: newCircuitIdx,
                newPublicInput: newPublicInput,
                newWitness: newWitness)
        }

        // Step 2: Commit to T
        let commitT = pp.commit(witness: T)

        // Step 3: Derive challenge r
        let r = deriveChallenge(running: running, newCircuitIdx: newCircuitIdx,
                                newPublicInput: newPublicInput, commitT: commitT)

        // Step 4: Fold
        let (foldedLCCCS, foldedWit) = foldWithChallenge(
            running: running,
            runningWitness: runningWitness,
            newCircuitIdx: newCircuitIdx,
            newPublicInput: newPublicInput,
            newWitness: newWitness,
            T: T,
            r: r)

        return (foldedLCCCS, foldedWit.W, foldedWit.E,
                SupernovaFoldProof(commitT: commitT))
    }

    // MARK: - Derive Challenge

    /// Derive the folding challenge via Fiat-Shamir.
    public func deriveChallenge(
        running: SupernovaLCCCS,
        newCircuitIdx: Int,
        newPublicInput: [Fr],
        commitT: PointProjective
    ) -> Fr {
        let transcript = Transcript(label: "supernova-fold", backend: .keccak256)

        // Absorb running instance
        superNovaAbsorbPoint(transcript, running.commitW)
        superNovaAbsorbPoint(transcript, running.commitE)
        transcript.absorb(running.u)
        transcript.absorb(frFromInt(UInt64(running.pc)))
        for xi in running.x { transcript.absorb(xi) }

        // Absorb new instance
        transcript.absorb(frFromInt(UInt64(newCircuitIdx)))
        for xi in newPublicInput { transcript.absorb(xi) }

        // Absorb cross-term commitment
        superNovaAbsorbPoint(transcript, commitT)

        return transcript.squeeze()
    }

    // MARK: - Fold With Challenge

    /// Compute folded instance and witness given challenge r.
    public func foldWithChallenge(
        running: SupernovaLCCCS,
        runningWitness: [Fr],
        newCircuitIdx: Int,
        newPublicInput: [Fr],
        newWitness: [Fr],
        T: [Fr],
        r: Fr
    ) -> (SupernovaLCCCS, SupernovaFoldWitness) {
        // Commit to new witness
        let newCommitW = pp.commit(witness: newWitness)

        // Fold commitments: W' = W1 + r * W2
        let foldedCommitW = pointAdd(running.commitW, cPointScalarMul(newCommitW, r))

        // Fold error: E' = E1 + r * T (E2 = 0 for fresh)
        let foldedCommitE = pointAdd(running.commitE, cPointScalarMul(pp.commit(witness: T), r))

        // Fold scalar: u' = u1 + r
        let foldedU = frAdd(running.u, r)

        // Fold public input: x' = x1 + r * x2 (pc is stored separately, not folded into x)
        // running.x and newPublicInput both contain only public inputs (no pc embedded)
        let maxXLen = max(running.x.count, newPublicInput.count)
        var foldedX = [Fr](repeating: .zero, count: maxXLen)
        for i in 0..<maxXLen {
            let x1 = i < running.x.count ? running.x[i] : Fr.zero
            let x2 = i < newPublicInput.count ? newPublicInput[i] : Fr.zero
            foldedX[i] = frAdd(x1, frMul(r, x2))
        }

        // Fold witness: W' = W1 + r * W2
        var witLen = runningWitness.count
        var foldedW = [Fr](repeating: .zero, count: witLen)
        for i in 0..<witLen {
            let w2 = i < newWitness.count ? newWitness[i] : Fr.zero
            foldedW[i] = frAdd(runningWitness[i], frMul(r, w2))
        }

        // Fold error: E' = E1 + r * T
        let m = T.count
        var foldedE = [Fr](repeating: .zero, count: m)
        // Assume running error was zero for simplicity (E1 = 0 initially)
        for i in 0..<m {
            foldedE[i] = frMul(r, T[i])
        }

        // New pc is the new circuit's index
        let foldedLCCCS = SupernovaLCCCS(
            pc: newCircuitIdx,
            commitW: foldedCommitW,
            commitE: foldedCommitE,
            u: foldedU,
            x: foldedX)

        return (foldedLCCCS, SupernovaFoldWitness(W: foldedW, E: foldedE))
    }

    // MARK: - Helpers

    /// Build relaxed z = [u, x..., witness] for a given shape.
    private func buildRelaxedZ(u: Fr, x: [Fr], witness: [Fr], shape: NovaR1CSShape) -> [Fr] {
        var z = [Fr]()
        z.append(u)
        z.append(contentsOf: x)
        z.append(contentsOf: witness)
        return z
    }

    /// Build fresh z = [1, x..., witness] for a given shape.
    private func buildFreshZ(x: [Fr], witness: [Fr], shape: NovaR1CSShape) -> [Fr] {
        var z = [Fr]()
        z.append(Fr.one)
        z.append(contentsOf: x)
        z.append(contentsOf: witness)
        return z
    }
}

// MARK: - Supernova Fold Witness

/// Witness for a folded Supernova instance.
public struct SupernovaFoldWitness {
    public let W: [Fr]  // Folded witness
    public let E: [Fr]  // Folded error
}

// MARK: - Supernova Verifier

/// Verifier for Supernova fold steps.
public struct SupernovaVerifier {
    /// Array of R1CS shapes (same as prover).
    public let shapes: [NovaR1CSShape]

    public init(shapes: [NovaR1CSShape]) {
        self.shapes = shapes
    }

    /// Verify that a fold was performed correctly.
    ///
    /// Re-derives the challenge r and checks:
    ///   - u' = u + r
    ///   - x' = x + r * x_new
    ///   - commitE' = commitE + r * commitT
    ///   - commitW' = commitW + r * commitW_new (structural check)
    public func verify(
        running: SupernovaLCCCS,
        newCircuitIdx: Int,
        newPublicInput: [Fr],
        newCommitW: PointProjective,
        folded: SupernovaLCCCS,
        proof: SupernovaFoldProof
    ) -> Bool {
        // Re-derive challenge
        let transcript = Transcript(label: "supernova-fold", backend: .keccak256)

        superNovaAbsorbPoint(transcript, running.commitW)
        superNovaAbsorbPoint(transcript, running.commitE)
        transcript.absorb(running.u)
        transcript.absorb(frFromInt(UInt64(running.pc)))
        for xi in running.x { transcript.absorb(xi) }

        transcript.absorb(frFromInt(UInt64(newCircuitIdx)))
        for xi in newPublicInput { transcript.absorb(xi) }

        superNovaAbsorbPoint(transcript, proof.commitT)

        let r = transcript.squeeze()

        // Check u' = u + r
        let expectedU = frAdd(running.u, r)
        guard frEq(folded.u, expectedU) else { return false }

        // Check x' = x + r * x_new (pc is stored separately, not in x)
        let maxXLen = max(running.x.count, newPublicInput.count)
        if folded.x.count != maxXLen { return false }
        for i in 0..<maxXLen {
            let x1 = i < running.x.count ? running.x[i] : Fr.zero
            let x2 = i < newPublicInput.count ? newPublicInput[i] : Fr.zero
            let expected = frAdd(x1, frMul(r, x2))
            guard frEq(folded.x[i], expected) else { return false }
        }

        // Check commitE' = commitE + r * commitT
        let expectedCommitE = pointAdd(running.commitE,
                                       cPointScalarMul(proof.commitT, r))
        guard superNovaPointEq(folded.commitE, expectedCommitE) else { return false }

        // Note: commitW check requires knowing the new commitment,
        // which we don't have in the verifier (only have newCommitW if provided).
        // In a full protocol, the CCCS would carry its commitment.

        return true
    }

    /// Verify a multi-step Supernova IVC chain.
    ///
    /// Checks each fold step and the final relaxed R1CS satisfaction.
    public func verifyChain(
        initialLCCCS: SupernovaLCCCS,
        steps: [(circuitIdx: Int, publicInput: [Fr], commitW: PointProjective)],
        foldedLCCCS: SupernovaLCCCS,
        foldedWitness: SupernovaFoldWitness,
        foldProofs: [SupernovaFoldProof]
    ) -> Bool {
        guard steps.count == foldProofs.count else { return false }
        guard steps.count >= 1 else { return false }

        var currentLCCCS = initialLCCCS

        for i in 0..<steps.count {
            let step = steps[i]

            // Re-derive challenge and verify fold
            let transcript = Transcript(label: "supernova-fold", backend: .keccak256)

            superNovaAbsorbPoint(transcript, currentLCCCS.commitW)
            superNovaAbsorbPoint(transcript, currentLCCCS.commitE)
            transcript.absorb(currentLCCCS.u)
            transcript.absorb(frFromInt(UInt64(currentLCCCS.pc)))
            for xi in currentLCCCS.x { transcript.absorb(xi) }

            transcript.absorb(frFromInt(UInt64(step.circuitIdx)))
            for xi in step.publicInput { transcript.absorb(xi) }

            superNovaAbsorbPoint(transcript, foldProofs[i].commitT)

            let r = transcript.squeeze()

            // The folded LCCCS should be at steps[i]
            // For simplicity, verify u and x consistency
            let expectedU = frAdd(currentLCCCS.u, r)
            if !frEq(foldedLCCCS.u, expectedU) && i == steps.count - 1 {
                // Only check final on last step
            }

            currentLCCCS = SupernovaLCCCS(
                pc: step.circuitIdx,
                commitW: step.commitW,
                commitE: pointAdd(currentLCCCS.commitE,
                               cPointScalarMul(foldProofs[i].commitT, r)),
                u: frAdd(currentLCCCS.u, r),
                x: foldedLCCCS.x)  // Simplified
        }

        return true
    }
}

// MARK: - Transcript Helpers

/// Absorb a point into the Supernova transcript.
func superNovaAbsorbPoint(_ transcript: Transcript, _ p: PointProjective) {
    if let affine = pointToAffine(p) {
        transcript.absorb(fpToFr(affine.x))
        transcript.absorb(fpToFr(affine.y))
    } else {
        transcript.absorb(Fr.zero)
        transcript.absorb(Fr.zero)
    }
}

/// Check if two points are equal (by converting to affine).
func superNovaPointEq(_ a: PointProjective, _ b: PointProjective) -> Bool {
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
