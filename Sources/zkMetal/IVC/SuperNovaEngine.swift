// SuperNova Engine -- Non-Uniform IVC via Multi-Circuit R1CS Folding
//
// Extends Nova to support non-uniform computation: different circuits per step.
// Each step selects which circuit to execute, maintaining one running instance
// per circuit type.
//
// Architecture:
//   SuperNovaProver  -- accumulate steps with circuit selection
//   CircuitSelector  -- pick which circuit to execute at each step
//   Multiple running instances (one per circuit type), folded independently
//
// The key insight: at each IVC step, only ONE circuit's running instance is
// updated (folded with the new step). The other circuits' running instances
// remain unchanged. This is sound because the circuit selector is committed
// to in the Fiat-Shamir transcript.
//
// Reference: "SuperNova: Proving universal machine execution" (Kothapalli, Setty 2022)

import Foundation
import NeonFieldOps

// MARK: - Circuit Selector

/// Selects which circuit to execute at each IVC step.
///
/// In non-uniform IVC, the computation can switch between different circuits
/// (e.g., different VM opcodes, different branches of execution).
public struct CircuitSelector {
    /// The available circuit types, indexed 0..<count
    public let count: Int

    /// Select which circuit to run for a given step.
    /// The selector function takes (step_index, current_state) and returns a circuit index.
    public let select: (_ step: Int, _ state: [Fr]) -> Int

    /// Create a selector with a fixed number of circuit types and a selection function.
    public init(count: Int, select: @escaping (_ step: Int, _ state: [Fr]) -> Int) {
        precondition(count > 0, "Must have at least one circuit type")
        self.count = count
        self.select = select
    }

    /// Create a round-robin selector (cycles through circuits in order).
    public static func roundRobin(count: Int) -> CircuitSelector {
        CircuitSelector(count: count) { step, _ in step % count }
    }

    /// Create a constant selector (always uses the same circuit).
    public static func constant(index: Int, count: Int) -> CircuitSelector {
        CircuitSelector(count: count) { _, _ in index }
    }
}

// MARK: - SuperNova Running State

/// The running state for a SuperNova IVC chain.
///
/// Maintains one relaxed R1CS instance per circuit type, plus the current
/// public state (step counter, initial input, current output).
public struct SuperNovaState {
    /// Running relaxed R1CS instances, one per circuit type.
    /// runningInstances[i] accumulates all steps that used circuit i.
    public var runningInstances: [RelaxedR1CSInstance]

    /// Running witnesses for each circuit type.
    public var runningWitnesses: [RelaxedR1CSWitness]

    /// Which circuit was used at the most recent step.
    public var lastCircuitIndex: Int

    /// Total number of IVC steps completed.
    public var stepCount: Int

    /// The current public state (carried across steps).
    public var currentState: [Fr]
}

// MARK: - SuperNova Folding Proof

/// Proof for a single SuperNova fold step.
/// Includes the circuit index and the Nova folding proof for that circuit.
public struct SuperNovaFoldingProof {
    /// Which circuit was used at this step.
    public let circuitIndex: Int

    /// The Nova folding proof for the selected circuit.
    public let novaProof: NovaFoldingProof

    /// Commitment to the circuit selector (for non-interactive soundness).
    public let selectorCommitment: Fr
}

// MARK: - SuperNova Prover

/// SuperNova IVC prover: extends Nova for non-uniform computation.
///
/// Supports K different circuit types. At each step:
///   1. The circuit selector picks which circuit to run
///   2. The prover generates a witness for that circuit
///   3. The new instance is folded into that circuit's running instance
///   4. All other circuits' running instances remain unchanged
///
/// Usage:
///   1. Create with K step circuits and a selector
///   2. Call `initialize` with the first step
///   3. Call `prove` for each subsequent step
///   4. Call `decide` to verify all accumulated instances
public class SuperNovaProver {
    /// The circuit definitions, one per circuit type.
    public let steps: [NovaStep]

    /// Circuit selector function.
    public let selector: CircuitSelector

    /// GPU MSM engine (optional).
    public let msmEngine: MetalMSM?

    /// Current running state.
    public private(set) var state: SuperNovaState?

    /// Initialize with multiple circuit types and a selector.
    ///
    /// - Parameters:
    ///   - steps: array of NovaStep, one per circuit type (must match selector.count)
    ///   - selector: circuit selector function
    ///   - msmEngine: optional GPU MSM engine
    public init(steps: [NovaStep], selector: CircuitSelector, msmEngine: MetalMSM? = nil) {
        precondition(steps.count == selector.count,
                     "Number of steps (\(steps.count)) must match selector count (\(selector.count))")
        self.steps = steps
        self.selector = selector
        self.msmEngine = msmEngine
    }

    // MARK: - Initialize (base case)

    /// Initialize the SuperNova IVC chain.
    ///
    /// Creates initial relaxed R1CS instances for all circuit types.
    /// Only the selected circuit's instance gets a real witness; others get
    /// trivial (zero) instances that will be populated on first use.
    ///
    /// - Parameters:
    ///   - publicInput: public input for the first step
    ///   - witness: witness for the first step
    ///   - initialState: initial public state to carry across steps
    /// - Returns: the initial SuperNova state
    @discardableResult
    public func initialize(publicInput: [Fr], witness: [Fr],
                           initialState: [Fr]) -> SuperNovaState {
        let circuitIdx = selector.select(0, initialState)
        precondition(circuitIdx >= 0 && circuitIdx < steps.count,
                     "Invalid circuit index \(circuitIdx)")

        let selectedStep = steps[circuitIdx]

        // Create running instances for all circuit types
        var instances = [RelaxedR1CSInstance]()
        var witnesses = [RelaxedR1CSWitness]()
        instances.reserveCapacity(steps.count)
        witnesses.reserveCapacity(steps.count)

        for i in 0..<steps.count {
            if i == circuitIdx {
                // Selected circuit: commit the real witness
                let commitW = selectedStep.pp.commit(witness: witness)
                instances.append(RelaxedR1CSInstance(commitW: commitW, publicInput: publicInput))
                witnesses.append(RelaxedR1CSWitness(W: witness, numConstraints: selectedStep.r1cs.m))
            } else {
                // Other circuits: trivial instance (identity commitment, zero witness)
                let trivialWitSize = steps[i].r1cs.n - 1 - steps[i].numPublic
                let trivialWit = [Fr](repeating: .zero, count: max(trivialWitSize, 1))
                let commitW = steps[i].pp.commit(witness: trivialWit)
                let trivialPub = [Fr](repeating: .zero, count: steps[i].numPublic)
                instances.append(RelaxedR1CSInstance(commitW: commitW, publicInput: trivialPub))
                witnesses.append(RelaxedR1CSWitness(W: trivialWit, numConstraints: steps[i].r1cs.m))
            }
        }

        let newState = SuperNovaState(
            runningInstances: instances,
            runningWitnesses: witnesses,
            lastCircuitIndex: circuitIdx,
            stepCount: 1,
            currentState: initialState
        )
        self.state = newState
        return newState
    }

    // MARK: - Prove (fold one step)

    /// Prove one SuperNova IVC step.
    ///
    /// - Parameters:
    ///   - publicInput: public input for this step
    ///   - witness: witness for this step
    ///   - newState: updated public state after this step
    /// - Returns: (updated SuperNova state, folding proof)
    public func prove(publicInput: [Fr], witness: [Fr],
                      newState: [Fr]) -> (SuperNovaState, SuperNovaFoldingProof) {
        guard var current = state else {
            preconditionFailure("Must call initialize() before prove()")
        }

        // Select circuit for this step
        let circuitIdx = selector.select(current.stepCount, current.currentState)
        precondition(circuitIdx >= 0 && circuitIdx < steps.count,
                     "Invalid circuit index \(circuitIdx)")

        let selectedStep = steps[circuitIdx]
        let running = current.runningInstances[circuitIdx]
        let runWit = current.runningWitnesses[circuitIdx]

        // Build z vectors
        let z1 = selectedStep.buildRelaxedZ(u: running.u, publicInput: running.publicInput,
                                             witness: runWit.W)
        let z2 = selectedStep.buildZ(publicInput: publicInput, witness: witness)

        // Compute matrix-vector products
        let Az1 = selectedStep.r1cs.matrices[0].mulVec(z1)
        let Bz1 = selectedStep.r1cs.matrices[1].mulVec(z1)
        let Cz1 = selectedStep.r1cs.matrices[2].mulVec(z1)
        let Az2 = selectedStep.r1cs.matrices[0].mulVec(z2)
        let Bz2 = selectedStep.r1cs.matrices[1].mulVec(z2)
        let Cz2 = selectedStep.r1cs.matrices[2].mulVec(z2)

        // Compute cross-term T
        let m = selectedStep.r1cs.m
        var T = [Fr](repeating: .zero, count: m)
        for i in 0..<m {
            let ab12 = frMul(Az1[i], Bz2[i])
            let ab21 = frMul(Az2[i], Bz1[i])
            let uCz2 = frMul(running.u, Cz2[i])
            let uCz1 = Cz1[i]
            T[i] = frSub(frAdd(ab12, ab21), frAdd(uCz2, uCz1))
        }

        // Commit to cross-term
        let commitT = selectedStep.ppE.commit(witness: T)

        // Derive circuit selector commitment (hash of circuit index for Fiat-Shamir)
        let selectorCommitment = frMul(Fr.from64([UInt64(circuitIdx + 1), 0, 0, 0]), Fr.one)

        // Fiat-Shamir transcript
        let transcript = Transcript(label: "supernova-fold", backend: .keccak256)
        transcript.absorb(selectorCommitment)
        absorbRelaxedInstance(transcript, running)
        absorbPoint(transcript, selectedStep.pp.commit(witness: witness))
        for x in publicInput { transcript.absorb(x) }
        absorbPoint(transcript, commitT)
        let r = transcript.squeeze()

        // Fold the selected circuit's running instance
        let newCommitW = selectedStep.pp.commit(witness: witness)
        let foldedCommitW = pointAdd(running.commitW, cPointScalarMul(newCommitW, r))
        let foldedCommitE = pointAdd(running.commitE, cPointScalarMul(commitT, r))
        let foldedU = frAdd(running.u, r)

        // Fold public input using C-accelerated linear combine
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

        // Fold witness
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

        // Fold error vector
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

        // Update only the selected circuit's running instance
        current.runningInstances[circuitIdx] = RelaxedR1CSInstance(
            commitW: foldedCommitW, commitE: foldedCommitE,
            u: foldedU, publicInput: foldedPublicInput)
        current.runningWitnesses[circuitIdx] = RelaxedR1CSWitness(W: foldedW, E: foldedE)
        current.lastCircuitIndex = circuitIdx
        current.stepCount += 1
        current.currentState = newState

        let proof = SuperNovaFoldingProof(
            circuitIndex: circuitIdx,
            novaProof: NovaFoldingProof(commitT: commitT),
            selectorCommitment: selectorCommitment)

        self.state = current
        return (current, proof)
    }

    // MARK: - IVC Chain (convenience)

    /// Run a full SuperNova IVC chain.
    ///
    /// - Parameters:
    ///   - initialState: initial public state
    ///   - steps: array of (publicInput, witness, newState) per step
    /// - Returns: (final state, step count)
    public func ivcChain(initialState: [Fr],
                         steps: [(publicInput: [Fr], witness: [Fr], newState: [Fr])])
        -> (SuperNovaState, Int)
    {
        precondition(!steps.isEmpty, "Need at least one step")

        initialize(publicInput: steps[0].publicInput, witness: steps[0].witness,
                   initialState: initialState)

        for i in 1..<steps.count {
            let _ = prove(publicInput: steps[i].publicInput,
                          witness: steps[i].witness,
                          newState: steps[i].newState)
        }

        return (state!, state!.stepCount)
    }

    // MARK: - Decide

    /// Verify all accumulated instances (the "decider").
    ///
    /// Checks that EVERY circuit type's running instance satisfies its
    /// relaxed R1CS relation. All must pass for the IVC chain to be valid.
    ///
    /// - Returns: true if all running instances are valid
    public func decide() -> Bool {
        guard let current = state else { return false }

        for i in 0..<steps.count {
            let decider = NovaDecider(step: steps[i])
            guard decider.decide(instance: current.runningInstances[i],
                                 witness: current.runningWitnesses[i]) else {
                return false
            }
        }
        return true
    }

    // MARK: - SuperNova Verifier

    /// Verify a single SuperNova fold step (no witness access).
    ///
    /// Checks the fold for the selected circuit and ensures all other
    /// circuits' running instances are unchanged.
    public func verifyFold(prevState: SuperNovaState,
                           newCommitW: PointProjective,
                           newPublicInput: [Fr],
                           nextState: SuperNovaState,
                           proof: SuperNovaFoldingProof) -> Bool {
        let circuitIdx = proof.circuitIndex
        guard circuitIdx >= 0 && circuitIdx < steps.count else { return false }

        // Verify the Nova fold for the selected circuit
        let verifier = NovaVerifier(step: steps[circuitIdx])
        guard verifier.verifyFold(
            running: prevState.runningInstances[circuitIdx],
            newCommitW: newCommitW,
            newPublicInput: newPublicInput,
            folded: nextState.runningInstances[circuitIdx],
            proof: proof.novaProof) else {
            return false
        }

        // Verify all other circuits are unchanged
        for i in 0..<steps.count {
            if i == circuitIdx { continue }
            guard pointEqual(prevState.runningInstances[i].commitW,
                             nextState.runningInstances[i].commitW) else { return false }
            guard pointEqual(prevState.runningInstances[i].commitE,
                             nextState.runningInstances[i].commitE) else { return false }
            guard frEq(prevState.runningInstances[i].u,
                       nextState.runningInstances[i].u) else { return false }
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
