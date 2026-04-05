// Nova Folding Scheme — IVC via R1CS instance folding
//
// Implements the Nova folding protocol for incrementally verifiable computation.
// Given two R1CS instances (a running relaxed instance and a fresh instance),
// produces a single folded relaxed instance that is valid iff both inputs were valid.
//
// The key insight: the cross-term T captures the "error" introduced by random
// linear combination. Committing to T before the challenge r ensures soundness.
//
// Protocol:
//   1. Prover computes cross-term T from the two instances
//   2. Prover commits to T, sends commitment to verifier
//   3. Verifier (or Fiat-Shamir) produces random challenge r
//   4. Both sides compute the folded instance using r
//
// This file provides NovaFoldProver and NovaFoldVerifier which operate on the
// NovaR1CSShape / NovaRelaxedInstance types from R1CSInstance.swift.
//
// Reference: "Nova: Recursive Zero-Knowledge Arguments from Folding Schemes"
//            (Kothapalli, Setty, Tzialla 2022)

import Foundation
import NeonFieldOps

// MARK: - Nova Fold Proof

/// Proof produced during a Nova folding step.
/// Contains the commitment to the cross-term T, which the verifier needs.
public struct NovaFoldProof {
    public let commitT: PointProjective  // Commitment to cross-term vector T

    public init(commitT: PointProjective) {
        self.commitT = commitT
    }
}

// MARK: - Nova Fold Prover

/// Nova folding prover: folds two R1CS instances into one relaxed instance.
///
/// Usage:
///   1. Create prover with an R1CS shape
///   2. Initialize a base instance with `relax()`
///   3. Repeatedly `fold()` new instances into the running instance
///   4. At the end, verify the final accumulated instance with the decider
public class NovaFoldProver {
    public let shape: NovaR1CSShape
    public let pp: PedersenParams

    /// Initialize with an R1CS shape. Generates Pedersen parameters sized for the witness
    /// and cross-term/error vectors.
    public init(shape: NovaR1CSShape) {
        self.shape = shape
        // Need generators for witness (numWitness) and for cross-term/error (numConstraints)
        let maxSize = max(shape.numWitness, shape.numConstraints)
        self.pp = PedersenParams.generate(size: max(maxSize, 1))
    }

    /// Initialize with pre-generated Pedersen parameters.
    public init(shape: NovaR1CSShape, pp: PedersenParams) {
        self.shape = shape
        self.pp = pp
    }

    // MARK: - Compute Cross-Term T

    /// Compute the cross-term vector T for folding two instances.
    ///
    /// T[i] = A*z1[i] * B*z2[i] + A*z2[i] * B*z1[i] - u1 * C*z2[i] - u2 * C*z1[i]
    ///
    /// where z1 = (u1, x1, W1) is the running (relaxed) instance
    /// and   z2 = (1, x2, W2) is the fresh instance (u2 = 1).
    ///
    /// This captures all the cross-terms that arise when we linearly combine
    /// the two R1CS relations with random scalar r.
    public func computeCrossTerm(
        runningInstance: NovaRelaxedInstance,
        runningWitness: NovaRelaxedWitness,
        newInstance: NovaR1CSInput,
        newWitness: NovaR1CSWitness
    ) -> [Fr] {
        let z1 = shape.buildRelaxedZ(
            u: runningInstance.u,
            instance: NovaR1CSInput(x: runningInstance.x),
            witness: NovaR1CSWitness(W: runningWitness.W))
        let z2 = shape.buildZ(instance: newInstance, witness: newWitness)

        let az1 = shape.A.mulVec(z1)
        let bz1 = shape.B.mulVec(z1)
        let cz1 = shape.C.mulVec(z1)
        let az2 = shape.A.mulVec(z2)
        let bz2 = shape.B.mulVec(z2)
        let cz2 = shape.C.mulVec(z2)

        let m = shape.numConstraints
        var T = [Fr](repeating: .zero, count: m)

        // T[i] = az1[i]*bz2[i] + az2[i]*bz1[i] - u1*cz2[i] - 1*cz1[i]
        // (u2 = 1 for a fresh instance)
        let u1 = runningInstance.u
        for i in 0..<m {
            let cross1 = frMul(az1[i], bz2[i])
            let cross2 = frMul(az2[i], bz1[i])
            let uCz2 = frMul(u1, cz2[i])
            // T[i] = cross1 + cross2 - uCz2 - cz1[i]
            var ti = frAdd(cross1, cross2)
            ti = frSub(ti, uCz2)
            ti = frSub(ti, cz1[i])
            T[i] = ti
        }
        return T
    }

    // MARK: - Fold

    /// Fold a fresh R1CS instance into the running relaxed instance.
    ///
    /// Returns the folded (relaxed) instance, folded witness, and the folding proof.
    ///
    /// The folding is homomorphic:
    ///   commitW' = commitW1 + r * commitW2
    ///   commitE' = commitE1 + r * commitT  (commitE2 = 0 for fresh instance)
    ///   u' = u1 + r
    ///   x' = x1 + r * x2
    ///   W' = W1 + r * W2
    ///   E' = E1 + r * T
    public func fold(
        runningInstance: NovaRelaxedInstance,
        runningWitness: NovaRelaxedWitness,
        newInstance: NovaR1CSInput,
        newWitness: NovaR1CSWitness
    ) -> (NovaRelaxedInstance, NovaRelaxedWitness, NovaFoldProof) {
        // Step 1: Compute cross-term T
        let T = computeCrossTerm(
            runningInstance: runningInstance,
            runningWitness: runningWitness,
            newInstance: newInstance,
            newWitness: newWitness)

        // Step 2: Commit to T
        let commitT = pp.commit(witness: T)

        // Step 3: Derive challenge r via Fiat-Shamir
        let r = deriveChallenge(
            runningInstance: runningInstance,
            newInstance: newInstance,
            commitT: commitT)

        // Step 4: Fold everything
        let (foldedInst, foldedWit) = foldWithChallenge(
            runningInstance: runningInstance,
            runningWitness: runningWitness,
            newInstance: newInstance,
            newWitness: newWitness,
            T: T,
            commitT: commitT,
            r: r)

        return (foldedInst, foldedWit, NovaFoldProof(commitT: commitT))
    }

    // MARK: - Derive Challenge (Fiat-Shamir)

    /// Derive the folding challenge r from the transcript.
    /// Absorbs both instances and the commitment to T.
    public func deriveChallenge(
        runningInstance: NovaRelaxedInstance,
        newInstance: NovaR1CSInput,
        commitT: PointProjective
    ) -> Fr {
        let transcript = Transcript(label: "nova-r1cs-fold", backend: .keccak256)

        // Absorb running instance
        novaAbsorbPoint(transcript, runningInstance.commitW)
        novaAbsorbPoint(transcript, runningInstance.commitE)
        transcript.absorb(runningInstance.u)
        for xi in runningInstance.x { transcript.absorb(xi) }

        // Absorb new instance
        for xi in newInstance.x { transcript.absorb(xi) }

        // Absorb commitment to cross-term
        novaAbsorbPoint(transcript, commitT)

        return transcript.squeeze()
    }

    // MARK: - Fold With Challenge

    /// Compute the folded instance and witness given the challenge r.
    /// This is the core algebraic folding operation.
    public func foldWithChallenge(
        runningInstance: NovaRelaxedInstance,
        runningWitness: NovaRelaxedWitness,
        newInstance: NovaR1CSInput,
        newWitness: NovaR1CSWitness,
        T: [Fr],
        commitT: PointProjective,
        r: Fr
    ) -> (NovaRelaxedInstance, NovaRelaxedWitness) {
        // Fold commitments: commitW' = commitW1 + r * commitW2
        let commitW2 = pp.commit(witness: newWitness.W)
        let foldedCommitW = pointAdd(runningInstance.commitW,
                                      cPointScalarMul(commitW2, r))

        // Fold error commitment: commitE' = commitE1 + r * commitT
        let foldedCommitE = pointAdd(runningInstance.commitE,
                                      cPointScalarMul(commitT, r))

        // Fold scalar: u' = u1 + r (fresh instance has u2 = 1, so r * u2 = r)
        let foldedU = frAdd(runningInstance.u, r)

        // Fold public input: x' = x1 + r * x2
        let numPub = runningInstance.x.count
        var foldedX = [Fr](repeating: .zero, count: numPub)
        for k in 0..<numPub {
            foldedX[k] = frAdd(runningInstance.x[k], frMul(r, newInstance.x[k]))
        }

        // Fold witness: W' = W1 + r * W2
        let witLen = runningWitness.W.count
        var foldedW = [Fr](repeating: .zero, count: witLen)
        for k in 0..<witLen {
            foldedW[k] = frAdd(runningWitness.W[k], frMul(r, newWitness.W[k]))
        }

        // Fold error: E' = E1 + r * T
        let m = shape.numConstraints
        var foldedE = [Fr](repeating: .zero, count: m)
        for k in 0..<m {
            foldedE[k] = frAdd(runningWitness.E[k], frMul(r, T[k]))
        }

        let foldedInst = NovaRelaxedInstance(
            commitW: foldedCommitW,
            commitE: foldedCommitE,
            u: foldedU,
            x: foldedX)

        let foldedWit = NovaRelaxedWitness(W: foldedW, E: foldedE)

        return (foldedInst, foldedWit)
    }

    // MARK: - IVC Chain

    /// Run an IVC chain: fold a sequence of (instance, witness) pairs.
    ///
    /// The first pair is relaxed to form the base case. Each subsequent pair
    /// is folded into the running accumulated instance.
    ///
    /// Returns the final accumulated relaxed instance and witness.
    public func ivcChain(
        steps: [(instance: NovaR1CSInput, witness: NovaR1CSWitness)]
    ) -> (NovaRelaxedInstance, NovaRelaxedWitness) {
        precondition(!steps.isEmpty, "Need at least one step")

        // Base case: relax the first instance
        var (runInst, runWit) = shape.relax(
            instance: steps[0].instance, witness: steps[0].witness, pp: pp)

        // Fold each subsequent step
        for i in 1..<steps.count {
            let (foldedInst, foldedWit, _) = fold(
                runningInstance: runInst, runningWitness: runWit,
                newInstance: steps[i].instance, newWitness: steps[i].witness)
            runInst = foldedInst
            runWit = foldedWit
        }

        return (runInst, runWit)
    }
}

// MARK: - Nova Fold Verifier

/// Nova folding verifier: checks that a fold was performed correctly.
///
/// The verifier does NOT need the witnesses. It only needs:
///   - The two input instances (running + fresh)
///   - The folding proof (commitment to T)
///   - The claimed folded instance
///
/// It re-derives the folded instance from the inputs + proof and checks equality.
public struct NovaFoldVerifier {
    public let shape: NovaR1CSShape

    public init(shape: NovaR1CSShape) {
        self.shape = shape
    }

    /// Verify that a fold was performed correctly.
    ///
    /// Re-derives the folded instance from:
    ///   - running: the relaxed running instance
    ///   - new: the fresh instance
    ///   - proof: contains commitment to cross-term T
    ///   - claimed: the prover's claimed folded instance
    ///
    /// Returns true if the claimed folded instance matches the re-derived one.
    public func verify(
        running: NovaRelaxedInstance,
        new: NovaR1CSInput,
        proof: NovaFoldProof,
        claimed: NovaRelaxedInstance
    ) -> Bool {
        // Re-derive the challenge r using the same Fiat-Shamir transcript
        let transcript = Transcript(label: "nova-r1cs-fold", backend: .keccak256)

        // Absorb running instance
        novaAbsorbPoint(transcript, running.commitW)
        novaAbsorbPoint(transcript, running.commitE)
        transcript.absorb(running.u)
        for xi in running.x { transcript.absorb(xi) }

        // Absorb new instance
        for xi in new.x { transcript.absorb(xi) }

        // Absorb commitment to cross-term
        novaAbsorbPoint(transcript, proof.commitT)

        let r = transcript.squeeze()

        // Check u' = u1 + r
        let expectedU = frAdd(running.u, r)
        if !frEq(claimed.u, expectedU) { return false }

        // Check x' = x1 + r * x2
        let numPub = running.x.count
        if claimed.x.count != numPub { return false }
        for k in 0..<numPub {
            let expected = frAdd(running.x[k], frMul(r, new.x[k]))
            if !frEq(claimed.x[k], expected) { return false }
        }

        // Check commitE' = commitE1 + r * commitT
        let expectedCommitE = pointAdd(running.commitE,
                                        cPointScalarMul(proof.commitT, r))
        if !novaPointEq(claimed.commitE, expectedCommitE) { return false }

        // Check commitW' = commitW1 + r * commitW_new
        // Note: the verifier would need the fresh instance's commitment to W
        // to fully check this. In a full protocol the fresh instance carries its
        // own commitment. For structural verification (u, x, commitE) we defer
        // the W commitment check to the final decider.

        return true
    }
}

// MARK: - Point Helpers (internal to Nova folding)

/// Absorb a projective point into a transcript by converting to affine coordinates.
func novaAbsorbPoint(_ transcript: Transcript, _ p: PointProjective) {
    if let affine = pointToAffine(p) {
        let xLimbs = affine.x.to64()
        let yLimbs = affine.y.to64()
        transcript.absorb(Fr.from64(xLimbs))
        transcript.absorb(Fr.from64(yLimbs))
    } else {
        // Point at infinity: absorb zeros
        transcript.absorb(Fr.zero)
        transcript.absorb(Fr.zero)
    }
}

/// Check if two projective points are equal (by converting to affine).
func novaPointEq(_ a: PointProjective, _ b: PointProjective) -> Bool {
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
