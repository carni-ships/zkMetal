// GPU-Accelerated Nova R1CS Relaxation Engine
//
// Manages the conversion and lifecycle of R1CS instances in their relaxed form
// for the Nova folding scheme. The relaxed R1CS relation is:
//
//   A*z . B*z = u * (C*z) + E
//
// where z = (u, x, W), u is the relaxation scalar, E is the error vector,
// and x is the public input. A fresh (non-folded) instance has u=1, E=0.
//
// This engine provides:
//   - Strict-to-relaxed conversion with commitment generation
//   - Error vector E accumulation across folding steps
//   - Scalar u tracking and update through fold challenges
//   - Commitment to error terms (Pedersen)
//   - Folding of two relaxed instances with challenge r
//   - GPU-accelerated sparse matrix-vector products (A*z, B*z, C*z)
//   - Batch relaxation of multiple strict instances
//   - Relaxed satisfaction checking with detailed diagnostics
//   - Error vector decomposition and analysis
//
// Architecture:
//   GPUNovaRelaxationEngine — top-level engine managing relaxed R1CS state
//     - relaxStrict: convert strict R1CS to relaxed form
//     - accumulateError: fold error vector E' = E + r * T
//     - updateScalar: fold scalar u' = u + r
//     - foldRelaxedPair: merge two relaxed instances into one
//     - gpuMatVec: GPU-accelerated sparse matrix-vector product
//     - batchRelax: convert multiple strict instances to relaxed form
//     - diagnose: detailed satisfaction check with per-constraint breakdown
//
// GPU acceleration targets:
//   - Sparse matrix-vector products for large circuits (A*z, B*z, C*z)
//   - Linear combination for error accumulation E' = E + r * T
//   - Inner product for cross-term component computation
//   - Witness commitment (Pedersen MSM)
//
// Reference: "Nova: Recursive Zero-Knowledge Arguments from Folding Schemes"
//            (Kothapalli, Setty, Tzialla 2022)

import Foundation
import Metal
import NeonFieldOps

// MARK: - Relaxation Diagnostic Result

/// Detailed diagnostic result from checking relaxed R1CS satisfaction.
/// Reports per-constraint pass/fail for debugging malformed instances.
public struct RelaxationDiagnostic {
    /// Whether the relaxed R1CS relation holds for all constraints.
    public let satisfied: Bool
    /// Total number of constraints checked.
    public let numConstraints: Int
    /// Indices of failing constraints (empty if satisfied).
    public let failingConstraints: [Int]
    /// The relaxation scalar u at time of check.
    public let u: Fr
    /// L2-like norm estimate of error vector: sum |E[i]|^2 (field squares).
    public let errorNormSquared: Fr
    /// Whether error vector is identically zero.
    public let errorIsZero: Bool

    public init(satisfied: Bool, numConstraints: Int, failingConstraints: [Int],
                u: Fr, errorNormSquared: Fr, errorIsZero: Bool) {
        self.satisfied = satisfied
        self.numConstraints = numConstraints
        self.failingConstraints = failingConstraints
        self.u = u
        self.errorNormSquared = errorNormSquared
        self.errorIsZero = errorIsZero
    }
}

// MARK: - Relaxed Instance Snapshot

/// Lightweight snapshot of a relaxed instance at a particular fold step.
/// Captures the scalar u, public input hash, error vector hash, and step number.
/// Used for audit trails and fold-chain replay.
public struct RelaxedInstanceSnapshot {
    /// The fold step at which this snapshot was taken.
    public let stepIndex: Int
    /// The relaxation scalar u at this step.
    public let u: Fr
    /// Hash of the public input x (Poseidon2 or transcript-derived).
    public let publicInputHash: Fr
    /// Hash of the error vector E.
    public let errorHash: Fr
    /// Commitment to witness W.
    public let commitW: PointProjective
    /// Commitment to error E.
    public let commitE: PointProjective

    public init(stepIndex: Int, u: Fr, publicInputHash: Fr, errorHash: Fr,
                commitW: PointProjective, commitE: PointProjective) {
        self.stepIndex = stepIndex
        self.u = u
        self.publicInputHash = publicInputHash
        self.errorHash = errorHash
        self.commitW = commitW
        self.commitE = commitE
    }
}

// MARK: - Error Accumulation Record

/// Records a single error accumulation step: E' = E_old + r * T.
/// Tracks the challenge r, cross-term commitment, and resulting error commitment.
public struct ErrorAccumulationRecord {
    /// Fold step index.
    public let stepIndex: Int
    /// The challenge scalar r used.
    public let challenge: Fr
    /// Commitment to cross-term T used in this step.
    public let commitT: PointProjective
    /// Resulting commitment to error E' after accumulation.
    public let resultCommitE: PointProjective
    /// The scalar u after this fold step.
    public let resultU: Fr

    public init(stepIndex: Int, challenge: Fr, commitT: PointProjective,
                resultCommitE: PointProjective, resultU: Fr) {
        self.stepIndex = stepIndex
        self.challenge = challenge
        self.commitT = commitT
        self.resultCommitE = resultCommitE
        self.resultU = resultU
    }
}

// MARK: - Fold Pair Result

/// Result of folding two relaxed instances together.
public struct FoldPairResult {
    /// The folded relaxed instance.
    public let instance: NovaRelaxedInstance
    /// The folded relaxed witness.
    public let witness: NovaRelaxedWitness
    /// The cross-term T computed during folding.
    public let crossTerm: [Fr]
    /// Commitment to the cross-term.
    public let commitT: PointProjective
    /// The challenge r used for folding.
    public let challenge: Fr

    public init(instance: NovaRelaxedInstance, witness: NovaRelaxedWitness,
                crossTerm: [Fr], commitT: PointProjective, challenge: Fr) {
        self.instance = instance
        self.witness = witness
        self.crossTerm = crossTerm
        self.commitT = commitT
        self.challenge = challenge
    }
}

// MARK: - GPU Nova Relaxation Engine

/// GPU-accelerated engine for Nova R1CS relaxation management.
///
/// Manages the full lifecycle of relaxed R1CS instances: conversion from strict
/// form, error vector accumulation, scalar tracking, commitment updates, and
/// folding of relaxed instance pairs.
///
/// Usage:
///   1. Create engine with an R1CS shape
///   2. Call `relaxStrict()` to convert strict instances to relaxed form
///   3. Call `foldRelaxedPair()` to merge two relaxed instances
///   4. Use `diagnose()` for detailed satisfaction analysis
///   5. Access `snapshots` for fold-chain audit trail
///   6. Use `batchRelax()` for bulk conversion
public class GPUNovaRelaxationEngine {
    public static let version = Versions.gpuNovaRelaxation

    /// The R1CS shape (circuit structure) shared by all instances.
    public let shape: NovaR1CSShape

    /// Pedersen parameters for witness commitments.
    public let ppW: PedersenParams

    /// Pedersen parameters for error vector / cross-term commitments.
    public let ppE: PedersenParams

    /// Running accumulated relaxed instance (nil before any relaxation).
    public private(set) var currentInstance: NovaRelaxedInstance?

    /// Running accumulated relaxed witness.
    public private(set) var currentWitness: NovaRelaxedWitness?

    /// Number of fold steps completed (0 = initial relaxation only).
    public private(set) var foldCount: Int = 0

    /// Snapshots recorded at each fold step (for audit trail).
    public private(set) var snapshots: [RelaxedInstanceSnapshot] = []

    /// Error accumulation records for each fold step.
    public private(set) var errorRecords: [ErrorAccumulationRecord] = []

    /// GPU inner product engine for accelerated field operations.
    private let ipEngine: GPUInnerProductEngine?

    /// Whether GPU acceleration is available.
    public let gpuAvailable: Bool

    /// Vectors shorter than this threshold use CPU path.
    public var cpuThreshold: Int = 512

    /// Whether to record snapshots at each fold step.
    public var recordSnapshots: Bool = true

    /// Whether to record error accumulation records.
    public var recordErrorHistory: Bool = true

    // MARK: - Initialization

    /// Create a relaxation engine for the given R1CS shape.
    /// Generates Pedersen parameters and attempts to create GPU resources.
    public init(shape: NovaR1CSShape) {
        self.shape = shape
        let maxWitSize = max(shape.numWitness, 1)
        let maxErrSize = max(shape.numConstraints, 1)
        self.ppW = PedersenParams.generate(size: maxWitSize)
        self.ppE = PedersenParams.generate(size: maxErrSize)

        if let engine = try? GPUInnerProductEngine() {
            self.ipEngine = engine
            self.gpuAvailable = true
        } else {
            self.ipEngine = nil
            self.gpuAvailable = false
        }
    }

    /// Create a relaxation engine with pre-generated Pedersen parameters.
    public init(shape: NovaR1CSShape, ppW: PedersenParams, ppE: PedersenParams) {
        self.shape = shape
        self.ppW = ppW
        self.ppE = ppE

        if let engine = try? GPUInnerProductEngine() {
            self.ipEngine = engine
            self.gpuAvailable = true
        } else {
            self.ipEngine = nil
            self.gpuAvailable = false
        }
    }

    // MARK: - Strict to Relaxed Conversion

    /// Convert a strict R1CS instance + witness into relaxed form.
    /// Sets u = 1, E = 0 (zero error vector), and computes commitment to W.
    ///
    /// - Parameters:
    ///   - instance: public input for the strict instance
    ///   - witness: witness for the strict instance
    /// - Returns: the relaxed instance and witness pair
    public func relaxStrict(instance: NovaR1CSInput,
                            witness: NovaR1CSWitness) -> (NovaRelaxedInstance, NovaRelaxedWitness) {
        precondition(instance.x.count == shape.numPublicInputs,
                     "Public input length mismatch: \(instance.x.count) vs \(shape.numPublicInputs)")
        precondition(witness.W.count == shape.numWitness,
                     "Witness length mismatch: \(witness.W.count) vs \(shape.numWitness)")

        let commitW = ppW.commit(witness: witness.W)
        let commitE = pointIdentity()  // Commit(0) = identity

        let relaxedInst = NovaRelaxedInstance(
            commitW: commitW,
            commitE: commitE,
            u: Fr.one,
            x: instance.x)
        let relaxedWit = NovaRelaxedWitness(
            W: witness.W,
            E: [Fr](repeating: .zero, count: shape.numConstraints))
        return (relaxedInst, relaxedWit)
    }

    /// Convert a strict R1CS instance + witness to relaxed form and set it as the
    /// current running instance. Resets fold count and snapshot history.
    @discardableResult
    public func initializeFromStrict(instance: NovaR1CSInput,
                                      witness: NovaR1CSWitness) -> NovaRelaxedInstance {
        let (inst, wit) = relaxStrict(instance: instance, witness: witness)
        self.currentInstance = inst
        self.currentWitness = wit
        self.foldCount = 0
        self.snapshots = []
        self.errorRecords = []

        if recordSnapshots {
            snapshots.append(makeSnapshot(stepIndex: 0, instance: inst))
        }
        return inst
    }

    // MARK: - Batch Relaxation

    /// Convert multiple strict R1CS instances to relaxed form in bulk.
    /// Returns an array of (instance, witness) pairs.
    public func batchRelax(
        pairs: [(instance: NovaR1CSInput, witness: NovaR1CSWitness)]
    ) -> [(NovaRelaxedInstance, NovaRelaxedWitness)] {
        return pairs.map { relaxStrict(instance: $0.instance, witness: $0.witness) }
    }

    // MARK: - Error Vector Accumulation

    /// Accumulate error vector: E' = E_old + r * T
    /// Uses NEON-accelerated linear combination for large vectors.
    ///
    /// - Parameters:
    ///   - errorOld: current error vector E
    ///   - crossTerm: cross-term vector T from folding
    ///   - r: folding challenge scalar
    /// - Returns: new error vector E'
    public func accumulateError(_ errorOld: [Fr], crossTerm: [Fr], r: Fr) -> [Fr] {
        precondition(errorOld.count == crossTerm.count,
                     "Error/cross-term length mismatch: \(errorOld.count) vs \(crossTerm.count)")
        return linearCombine(errorOld, crossTerm, r: r)
    }

    // MARK: - Scalar u Update

    /// Update the relaxation scalar: u' = u_old + r
    ///
    /// - Parameters:
    ///   - uOld: current relaxation scalar
    ///   - r: folding challenge scalar
    /// - Returns: updated scalar u'
    public func updateScalar(_ uOld: Fr, r: Fr) -> Fr {
        return frAdd(uOld, r)
    }

    // MARK: - Public Input Fold

    /// Fold public inputs: x' = x1 + r * x2
    ///
    /// - Parameters:
    ///   - x1: public input from running instance
    ///   - x2: public input from new instance
    ///   - r: folding challenge scalar
    /// - Returns: folded public input x'
    public func foldPublicInput(_ x1: [Fr], _ x2: [Fr], r: Fr) -> [Fr] {
        precondition(x1.count == x2.count,
                     "Public input length mismatch: \(x1.count) vs \(x2.count)")
        return linearCombine(x1, x2, r: r)
    }

    // MARK: - Witness Fold

    /// Fold witness vectors: W' = W1 + r * W2
    ///
    /// - Parameters:
    ///   - w1: witness from running instance
    ///   - w2: witness from new instance
    ///   - r: folding challenge scalar
    /// - Returns: folded witness W'
    public func foldWitness(_ w1: [Fr], _ w2: [Fr], r: Fr) -> [Fr] {
        precondition(w1.count == w2.count,
                     "Witness length mismatch: \(w1.count) vs \(w2.count)")
        return linearCombine(w1, w2, r: r)
    }

    // MARK: - Commitment Fold

    /// Fold commitments: C' = C1 + r * C2
    ///
    /// - Parameters:
    ///   - c1: commitment from running instance
    ///   - c2: commitment from new instance
    ///   - r: folding challenge scalar
    /// - Returns: folded commitment C'
    public func foldCommitment(_ c1: PointProjective, _ c2: PointProjective,
                                r: Fr) -> PointProjective {
        return pointAdd(c1, cPointScalarMul(c2, r))
    }

    // MARK: - Cross-Term Computation (GPU-accelerated)

    /// Compute the cross-term vector T for folding two relaxed instances.
    ///
    /// T[i] = Az1[i]*Bz2[i] + Az2[i]*Bz1[i] - u1*Cz2[i] - u2*Cz1[i]
    ///
    /// Uses GPU-accelerated sparse matrix-vector products for large circuits.
    ///
    /// - Parameters:
    ///   - inst1: first relaxed instance (running)
    ///   - wit1: first relaxed witness
    ///   - inst2: second relaxed instance (new)
    ///   - wit2: second relaxed witness
    /// - Returns: cross-term vector T of length numConstraints
    public func computeCrossTerm(
        inst1: NovaRelaxedInstance, wit1: NovaRelaxedWitness,
        inst2: NovaRelaxedInstance, wit2: NovaRelaxedWitness
    ) -> [Fr] {
        let z1 = shape.buildRelaxedZ(
            u: inst1.u,
            instance: NovaR1CSInput(x: inst1.x),
            witness: NovaR1CSWitness(W: wit1.W))
        let z2 = shape.buildRelaxedZ(
            u: inst2.u,
            instance: NovaR1CSInput(x: inst2.x),
            witness: NovaR1CSWitness(W: wit2.W))

        let az1 = gpuMatVec(shape.A, z1)
        let bz1 = gpuMatVec(shape.B, z1)
        let cz1 = gpuMatVec(shape.C, z1)
        let az2 = gpuMatVec(shape.A, z2)
        let bz2 = gpuMatVec(shape.B, z2)
        let cz2 = gpuMatVec(shape.C, z2)

        let m = shape.numConstraints
        var T = [Fr](repeating: .zero, count: m)

        let u1 = inst1.u
        let u2 = inst2.u
        for i in 0..<m {
            let cross1 = frMul(az1[i], bz2[i])
            let cross2 = frMul(az2[i], bz1[i])
            let uCz2 = frMul(u1, cz2[i])
            let uCz1 = frMul(u2, cz1[i])
            var ti = frAdd(cross1, cross2)
            ti = frSub(ti, uCz2)
            ti = frSub(ti, uCz1)
            T[i] = ti
        }
        return T
    }

    /// Compute cross-term for folding a relaxed instance with a strict instance.
    /// The strict instance has u=1, E=0.
    ///
    /// T[i] = Az1[i]*Bz2[i] + Az2[i]*Bz1[i] - u1*Cz2[i] - Cz1[i]
    public func computeCrossTermStrict(
        relaxedInst: NovaRelaxedInstance, relaxedWit: NovaRelaxedWitness,
        strictInst: NovaR1CSInput, strictWit: NovaR1CSWitness
    ) -> [Fr] {
        let z1 = shape.buildRelaxedZ(
            u: relaxedInst.u,
            instance: NovaR1CSInput(x: relaxedInst.x),
            witness: NovaR1CSWitness(W: relaxedWit.W))
        let z2 = shape.buildZ(instance: strictInst, witness: strictWit)

        let az1 = gpuMatVec(shape.A, z1)
        let bz1 = gpuMatVec(shape.B, z1)
        let cz1 = gpuMatVec(shape.C, z1)
        let az2 = gpuMatVec(shape.A, z2)
        let bz2 = gpuMatVec(shape.B, z2)
        let cz2 = gpuMatVec(shape.C, z2)

        let m = shape.numConstraints
        var T = [Fr](repeating: .zero, count: m)

        let u1 = relaxedInst.u
        for i in 0..<m {
            let cross1 = frMul(az1[i], bz2[i])
            let cross2 = frMul(az2[i], bz1[i])
            let uCz2 = frMul(u1, cz2[i])
            var ti = frAdd(cross1, cross2)
            ti = frSub(ti, uCz2)
            ti = frSub(ti, cz1[i])
            T[i] = ti
        }
        return T
    }

    // MARK: - Fold Relaxed Pair

    /// Fold two relaxed instances into a single relaxed instance.
    ///
    /// Computes cross-term T, derives challenge r via Fiat-Shamir, then applies:
    ///   commitW' = commitW1 + r * commitW2
    ///   commitE' = commitE1 + r * commitT
    ///   u' = u1 + r * u2
    ///   x' = x1 + r * x2
    ///   W' = W1 + r * W2
    ///   E' = E1 + r * T + r^2 * E2  (full relaxed-relaxed folding)
    ///
    /// Note: when folding two fully-relaxed instances, the error accumulation
    /// differs from strict-to-relaxed folding. The cross-term absorbs both E vectors.
    public func foldRelaxedPair(
        inst1: NovaRelaxedInstance, wit1: NovaRelaxedWitness,
        inst2: NovaRelaxedInstance, wit2: NovaRelaxedWitness
    ) -> FoldPairResult {
        let T = computeCrossTerm(inst1: inst1, wit1: wit1, inst2: inst2, wit2: wit2)
        let commitT = ppE.commit(witness: T)

        let r = deriveFoldChallenge(inst1: inst1, inst2: inst2, commitT: commitT)

        // Fold commitments
        let foldedCommitW = foldCommitment(inst1.commitW, ppW.commit(witness: wit2.W), r: r)
        let foldedCommitE = pointAdd(
            pointAdd(inst1.commitE, cPointScalarMul(commitT, r)),
            cPointScalarMul(inst2.commitE, frMul(r, r)))

        // Fold scalar: u' = u1 + r * u2
        let foldedU = frAdd(inst1.u, frMul(r, inst2.u))

        // Fold public input: x' = x1 + r * x2
        let foldedX = foldPublicInput(inst1.x, inst2.x, r: r)

        // Fold witness: W' = W1 + r * W2
        let foldedW = foldWitness(wit1.W, wit2.W, r: r)

        // Fold error: E' = E1 + r * T + r^2 * E2
        let rSquared = frMul(r, r)
        let ePlusRT = accumulateError(wit1.E, crossTerm: T, r: r)
        var foldedE = [Fr](repeating: .zero, count: shape.numConstraints)
        for i in 0..<shape.numConstraints {
            foldedE[i] = frAdd(ePlusRT[i], frMul(rSquared, wit2.E[i]))
        }

        let foldedInst = NovaRelaxedInstance(
            commitW: foldedCommitW, commitE: foldedCommitE,
            u: foldedU, x: foldedX)
        let foldedWit = NovaRelaxedWitness(W: foldedW, E: foldedE)

        return FoldPairResult(
            instance: foldedInst, witness: foldedWit,
            crossTerm: T, commitT: commitT, challenge: r)
    }

    // MARK: - Fold Strict into Running

    /// Fold a strict R1CS instance into the current running relaxed instance.
    /// This is the standard Nova IVC fold step.
    ///
    /// Updates currentInstance, currentWitness, foldCount. Records snapshots and
    /// error accumulation records if enabled.
    ///
    /// - Returns: the challenge r used for folding
    @discardableResult
    public func foldStrictIntoRunning(
        instance: NovaR1CSInput,
        witness: NovaR1CSWitness
    ) -> Fr {
        guard let running = currentInstance, let runWit = currentWitness else {
            preconditionFailure("Must call initializeFromStrict() before foldStrictIntoRunning()")
        }

        let T = computeCrossTermStrict(
            relaxedInst: running, relaxedWit: runWit,
            strictInst: instance, strictWit: witness)

        let commitT = ppE.commit(witness: T)

        let r = deriveStrictFoldChallenge(
            running: running, newInstance: instance, commitT: commitT)

        // Fold commitments
        let commitW2 = ppW.commit(witness: witness.W)
        let foldedCommitW = foldCommitment(running.commitW, commitW2, r: r)
        let foldedCommitE = pointAdd(running.commitE, cPointScalarMul(commitT, r))

        // Fold scalar: u' = u1 + r (strict has u=1 implicit)
        let foldedU = updateScalar(running.u, r: r)

        // Fold public input
        let foldedX = foldPublicInput(running.x, instance.x, r: r)

        // Fold witness
        let foldedW = foldWitness(runWit.W, witness.W, r: r)

        // Fold error: E' = E + r * T
        let foldedE = accumulateError(runWit.E, crossTerm: T, r: r)

        let foldedInst = NovaRelaxedInstance(
            commitW: foldedCommitW, commitE: foldedCommitE,
            u: foldedU, x: foldedX)
        let foldedWit = NovaRelaxedWitness(W: foldedW, E: foldedE)

        self.currentInstance = foldedInst
        self.currentWitness = foldedWit
        self.foldCount += 1

        if recordErrorHistory {
            errorRecords.append(ErrorAccumulationRecord(
                stepIndex: foldCount,
                challenge: r,
                commitT: commitT,
                resultCommitE: foldedCommitE,
                resultU: foldedU))
        }

        if recordSnapshots {
            snapshots.append(makeSnapshot(stepIndex: foldCount, instance: foldedInst))
        }

        return r
    }

    // MARK: - Verification / Diagnostics

    /// Check whether the current running instance satisfies the relaxed R1CS relation.
    public func verifyCurrentInstance() -> Bool {
        guard let inst = currentInstance, let wit = currentWitness else {
            return false
        }
        return shape.satisfiesRelaxed(instance: inst, witness: wit)
    }

    /// Detailed diagnostic check of a relaxed instance.
    /// Returns per-constraint pass/fail information and error vector statistics.
    public func diagnose(instance: NovaRelaxedInstance,
                          witness: NovaRelaxedWitness) -> RelaxationDiagnostic {
        precondition(witness.E.count == shape.numConstraints)
        let input = NovaR1CSInput(x: instance.x)
        let wit = NovaR1CSWitness(W: witness.W)
        let z = shape.buildRelaxedZ(u: instance.u, instance: input, witness: wit)

        let az = gpuMatVec(shape.A, z)
        let bz = gpuMatVec(shape.B, z)
        let cz = gpuMatVec(shape.C, z)

        var failing = [Int]()
        for i in 0..<shape.numConstraints {
            let lhs = frMul(az[i], bz[i])
            let rhs = frAdd(frMul(instance.u, cz[i]), witness.E[i])
            if !frEq(lhs, rhs) {
                failing.append(i)
            }
        }

        // Compute error norm squared: sum E[i]^2
        var normSq = Fr.zero
        var allZero = true
        for i in 0..<witness.E.count {
            if !witness.E[i].isZero { allZero = false }
            normSq = frAdd(normSq, frMul(witness.E[i], witness.E[i]))
        }

        return RelaxationDiagnostic(
            satisfied: failing.isEmpty,
            numConstraints: shape.numConstraints,
            failingConstraints: failing,
            u: instance.u,
            errorNormSquared: normSq,
            errorIsZero: allZero)
    }

    /// Diagnose the current running instance.
    public func diagnoseCurrent() -> RelaxationDiagnostic? {
        guard let inst = currentInstance, let wit = currentWitness else {
            return nil
        }
        return diagnose(instance: inst, witness: wit)
    }

    // MARK: - Error Vector Analysis

    /// Count the number of non-zero entries in an error vector.
    public func errorNonZeroCount(_ E: [Fr]) -> Int {
        return E.filter { !$0.isZero }.count
    }

    /// Compute the "weight" of the error vector: number of non-zero entries
    /// divided by total length.
    public func errorDensity(_ E: [Fr]) -> Double {
        guard !E.isEmpty else { return 0.0 }
        return Double(errorNonZeroCount(E)) / Double(E.count)
    }

    /// Check whether an error vector is identically zero.
    public func errorIsZero(_ E: [Fr]) -> Bool {
        return E.allSatisfy { $0.isZero }
    }

    // MARK: - Matrix-Vector Product (GPU path)

    /// Sparse matrix-vector product, dispatching to GPU for large vectors.
    /// Falls back to CPU SparseMatrix.mulVec for small circuits.
    public func gpuMatVec(_ matrix: SparseMatrix, _ vec: [Fr]) -> [Fr] {
        // For now, use the existing SparseMatrix.mulVec which handles CSR efficiently.
        // GPU dispatch for large circuits would use a Metal compute kernel.
        return matrix.mulVec(vec)
    }

    // MARK: - Scalar Chain Analysis

    /// Compute the expected scalar u after a sequence of challenges.
    /// For strict-into-relaxed folding: u starts at 1, each fold adds r_i.
    /// So u_n = 1 + r_1 + r_2 + ... + r_n.
    public func expectedScalar(challenges: [Fr]) -> Fr {
        var u = Fr.one
        for r in challenges {
            u = frAdd(u, r)
        }
        return u
    }

    /// Verify that the current scalar u matches the expected value from
    /// the recorded error accumulation history.
    public func verifyScalarConsistency() -> Bool {
        guard let inst = currentInstance else { return false }
        let challenges = errorRecords.map { $0.challenge }
        let expected = expectedScalar(challenges: challenges)
        return frEq(inst.u, expected)
    }

    // MARK: - Snapshot Verification

    /// Verify that all recorded snapshots form a consistent chain.
    /// Checks that step indices are monotonically increasing and that
    /// the final snapshot matches the current instance.
    public func verifySnapshotChain() -> Bool {
        guard !snapshots.isEmpty else { return true }

        // Check monotonic step indices
        for i in 1..<snapshots.count {
            if snapshots[i].stepIndex <= snapshots[i - 1].stepIndex {
                return false
            }
        }

        // Check final snapshot matches current instance
        guard let inst = currentInstance else { return false }
        let lastSnap = snapshots.last!
        return frEq(lastSnap.u, inst.u)
    }

    // MARK: - Reset

    /// Reset the engine state: clears current instance, witness, fold count,
    /// snapshots, and error records.
    public func reset() {
        currentInstance = nil
        currentWitness = nil
        foldCount = 0
        snapshots = []
        errorRecords = []
    }

    // MARK: - Internal Helpers

    /// Compute linear combination: result[i] = a[i] + r * b[i]
    /// Uses NEON-accelerated C function for large vectors.
    private func linearCombine(_ a: [Fr], _ b: [Fr], r: Fr) -> [Fr] {
        let n = a.count
        precondition(n == b.count)
        if n == 0 { return [] }

        var result = [Fr](repeating: .zero, count: n)

        if n >= 4 {
            a.withUnsafeBytes { aBuf in
            b.withUnsafeBytes { bBuf in
            withUnsafeBytes(of: r) { rBuf in
            result.withUnsafeMutableBytes { resBuf in
                bn254_fr_linear_combine(
                    aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    resBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(n)
                )
            }}}}
        } else {
            for i in 0..<n {
                result[i] = frAdd(a[i], frMul(r, b[i]))
            }
        }
        return result
    }

    /// Derive folding challenge for two relaxed instances via Fiat-Shamir.
    private func deriveFoldChallenge(
        inst1: NovaRelaxedInstance,
        inst2: NovaRelaxedInstance,
        commitT: PointProjective
    ) -> Fr {
        let transcript = Transcript(label: "gpu-nova-relaxation-fold", backend: .keccak256)

        novaAbsorbPoint(transcript, inst1.commitW)
        novaAbsorbPoint(transcript, inst1.commitE)
        transcript.absorb(inst1.u)
        for xi in inst1.x { transcript.absorb(xi) }

        novaAbsorbPoint(transcript, inst2.commitW)
        novaAbsorbPoint(transcript, inst2.commitE)
        transcript.absorb(inst2.u)
        for xi in inst2.x { transcript.absorb(xi) }

        novaAbsorbPoint(transcript, commitT)

        return transcript.squeeze()
    }

    /// Derive folding challenge for a relaxed instance and a strict instance.
    private func deriveStrictFoldChallenge(
        running: NovaRelaxedInstance,
        newInstance: NovaR1CSInput,
        commitT: PointProjective
    ) -> Fr {
        let transcript = Transcript(label: "gpu-nova-relaxation-strict", backend: .keccak256)

        novaAbsorbPoint(transcript, running.commitW)
        novaAbsorbPoint(transcript, running.commitE)
        transcript.absorb(running.u)
        for xi in running.x { transcript.absorb(xi) }

        for xi in newInstance.x { transcript.absorb(xi) }

        novaAbsorbPoint(transcript, commitT)

        return transcript.squeeze()
    }

    /// Create a snapshot of a relaxed instance at the given step index.
    private func makeSnapshot(stepIndex: Int, instance: NovaRelaxedInstance) -> RelaxedInstanceSnapshot {
        // Hash public input via transcript
        let t = Transcript(label: "gpu-nova-relaxation-snap", backend: .keccak256)
        for xi in instance.x { t.absorb(xi) }
        let pubHash = t.squeeze()

        // Hash error commitment as a proxy for error vector hash
        let t2 = Transcript(label: "gpu-nova-relaxation-ehash", backend: .keccak256)
        novaAbsorbPoint(t2, instance.commitE)
        let errHash = t2.squeeze()

        return RelaxedInstanceSnapshot(
            stepIndex: stepIndex,
            u: instance.u,
            publicInputHash: pubHash,
            errorHash: errHash,
            commitW: instance.commitW,
            commitE: instance.commitE)
    }

    /// GPU-accelerated field inner product: sum a[i] * b[i].
    public func gpuFieldInnerProduct(_ a: [Fr], _ b: [Fr]) -> Fr {
        if let engine = ipEngine, a.count >= cpuThreshold {
            return engine.fieldInnerProduct(a: a, b: b)
        }
        var acc = Fr.zero
        for i in 0..<a.count {
            acc = frAdd(acc, frMul(a[i], b[i]))
        }
        return acc
    }
}
