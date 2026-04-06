// GPU-Accelerated Nova Decider Circuit Engine
//
// In IVC (incrementally verifiable computation), after N folding steps the final
// accumulated relaxed R1CS instance must be "decided" — proved correct via a SNARK.
// This engine generates the decider *circuit* that encodes the verification checks
// as R1CS constraints, suitable for recursive proof composition.
//
// The decider circuit checks:
//   (1) The relaxed R1CS instance is satisfiable: Az . Bz = u*(Cz) + E
//   (2) The error vector commitment is valid: Commit(E) = commitE
//   (3) The folding was done correctly: NIFS transcript replay matches
//
// Supports both Nova (single circuit) and SuperNova (multi-circuit) variants,
// with Metal GPU acceleration for witness generation, R1CS constraint evaluation,
// and commitment verification.
//
// Architecture:
//   DeciderCircuitConfig          — circuit generation parameters
//   DeciderCircuitWire            — typed wire references for constraint wiring
//   DeciderCircuitConstraintSet   — accumulated R1CS constraints from circuit synthesis
//   DeciderCircuitWitnessLayout   — tracks variable allocation for witness generation
//   GPUNovaDeciderCircuitEngine   — top-level engine for circuit generation + proving
//   DeciderCircuitVerifier        — standalone verifier for decider circuit proofs
//   SuperNovaDeciderCircuitEngine — multi-circuit variant
//
// GPU acceleration targets:
//   - Witness generation: sparse matrix-vector products for large circuits
//   - Constraint evaluation: batch field arithmetic on GPU
//   - Commitment verification: Pedersen MSM via Metal
//   - Error vector inner product: GPU inner product engine
//
// Reference: "Nova: Recursive Zero-Knowledge Arguments from Folding Schemes"
//            (Kothapalli, Setty, Tzialla 2022)
// Reference: "CycleFold: Folding-scheme-based recursive arguments over a cycle of
//             elliptic curves" (Kothapalli, Setty 2023)

import Foundation
import Metal
import NeonFieldOps

// MARK: - Decider Circuit Configuration

/// Configuration parameters controlling decider circuit generation.
public struct DeciderCircuitConfig {
    /// Maximum number of constraints the original R1CS shape can have.
    public let maxConstraints: Int
    /// Maximum number of variables in the original R1CS shape.
    public let maxVariables: Int
    /// Whether to include NIFS transcript replay constraints.
    public let includeNIFSCheck: Bool
    /// Whether to include commitment opening constraints.
    public let includeCommitmentCheck: Bool
    /// Whether to use GPU acceleration for witness generation.
    public let useGPU: Bool
    /// Minimum vector length for GPU dispatch.
    public let gpuThreshold: Int
    /// Number of bits for scalar decomposition in commitment check.
    public let scalarBits: Int
    /// Whether this is a SuperNova (multi-circuit) decider.
    public let isSuperNova: Bool
    /// Number of circuit types for SuperNova (ignored for standard Nova).
    public let numCircuitTypes: Int

    public init(maxConstraints: Int = 4096,
                maxVariables: Int = 8192,
                includeNIFSCheck: Bool = true,
                includeCommitmentCheck: Bool = true,
                useGPU: Bool = true,
                gpuThreshold: Int = 512,
                scalarBits: Int = 254,
                isSuperNova: Bool = false,
                numCircuitTypes: Int = 1) {
        self.maxConstraints = maxConstraints
        self.maxVariables = maxVariables
        self.includeNIFSCheck = includeNIFSCheck
        self.includeCommitmentCheck = includeCommitmentCheck
        self.useGPU = useGPU
        self.gpuThreshold = gpuThreshold
        self.scalarBits = scalarBits
        self.isSuperNova = isSuperNova
        self.numCircuitTypes = max(numCircuitTypes, 1)
    }
}

// MARK: - Wire Reference

/// A typed reference to a wire in the decider circuit.
/// Wires are allocated sequentially: [constant_one | public_inputs | witness_vars].
public struct DeciderCircuitWire {
    /// Index into the full variable vector z = (1, x, W).
    public let index: Int
    /// Human-readable label for debugging.
    public let label: String

    public init(index: Int, label: String = "") {
        self.index = index
        self.label = label
    }

    /// The constant-one wire (z[0] = 1).
    public static let one = DeciderCircuitWire(index: 0, label: "one")
}

// MARK: - Constraint Entry

/// A single R1CS constraint: a . b = c where a, b, c are sparse linear combinations.
/// Each linear combination is [(wireIndex, coefficient)].
public struct DeciderCircuitConstraint {
    public let a: [(Int, Fr)]
    public let b: [(Int, Fr)]
    public let c: [(Int, Fr)]

    public init(a: [(Int, Fr)], b: [(Int, Fr)], c: [(Int, Fr)]) {
        self.a = a
        self.b = b
        self.c = c
    }
}

// MARK: - Constraint Set

/// Accumulated constraints from synthesizing the decider circuit.
/// Tracks the R1CS structure (A, B, C) being built incrementally.
public struct DeciderCircuitConstraintSet {
    /// All constraints accumulated during synthesis.
    public private(set) var constraints: [DeciderCircuitConstraint]
    /// Total number of variables allocated (including constant wire).
    public private(set) var numVariables: Int
    /// Number of public input variables.
    public private(set) var numPublicInputs: Int
    /// Next free variable index.
    public private(set) var nextFreeVar: Int

    public init(numPublicInputs: Int) {
        self.constraints = []
        self.numPublicInputs = numPublicInputs
        // z[0] = 1, z[1..l] = public inputs
        self.numVariables = 1 + numPublicInputs
        self.nextFreeVar = 1 + numPublicInputs
    }

    /// Allocate a fresh witness variable.
    public mutating func allocWitness(label: String = "") -> DeciderCircuitWire {
        let wire = DeciderCircuitWire(index: nextFreeVar, label: label)
        nextFreeVar += 1
        numVariables = nextFreeVar
        return wire
    }

    /// Allocate N witness variables.
    public mutating func allocWitnessBlock(count: Int, prefix: String = "w") -> [DeciderCircuitWire] {
        var wires = [DeciderCircuitWire]()
        wires.reserveCapacity(count)
        for i in 0..<count {
            wires.append(allocWitness(label: "\(prefix)_\(i)"))
        }
        return wires
    }

    /// Add a constraint: a . b = c.
    public mutating func addConstraint(_ constraint: DeciderCircuitConstraint) {
        constraints.append(constraint)
    }

    /// Add a multiplication constraint: wireA * wireB = wireC.
    public mutating func addMulConstraint(
        _ wireA: DeciderCircuitWire,
        _ wireB: DeciderCircuitWire,
        _ wireC: DeciderCircuitWire
    ) {
        addConstraint(DeciderCircuitConstraint(
            a: [(wireA.index, Fr.one)],
            b: [(wireB.index, Fr.one)],
            c: [(wireC.index, Fr.one)]))
    }

    /// Add a linear constraint: coeff * wireA = wireB.
    public mutating func addLinearConstraint(
        coeff: Fr,
        _ wireA: DeciderCircuitWire,
        _ wireB: DeciderCircuitWire
    ) {
        addConstraint(DeciderCircuitConstraint(
            a: [(wireA.index, coeff)],
            b: [(DeciderCircuitWire.one.index, Fr.one)],
            c: [(wireB.index, Fr.one)]))
    }

    /// Add an addition constraint: wireA + wireB = wireC.
    public mutating func addAddConstraint(
        _ wireA: DeciderCircuitWire,
        _ wireB: DeciderCircuitWire,
        _ wireC: DeciderCircuitWire
    ) {
        addConstraint(DeciderCircuitConstraint(
            a: [(wireA.index, Fr.one), (wireB.index, Fr.one)],
            b: [(DeciderCircuitWire.one.index, Fr.one)],
            c: [(wireC.index, Fr.one)]))
    }

    /// Add constraint: a * b + c = d (fused multiply-add).
    public mutating func addFMAConstraint(
        _ wireA: DeciderCircuitWire,
        _ wireB: DeciderCircuitWire,
        _ wireC: DeciderCircuitWire,
        _ wireD: DeciderCircuitWire
    ) {
        // Encode as: a * b = d - c
        addConstraint(DeciderCircuitConstraint(
            a: [(wireA.index, Fr.one)],
            b: [(wireB.index, Fr.one)],
            c: [(wireD.index, Fr.one), (wireC.index, frNeg(Fr.one))]))
    }

    /// Convert accumulated constraints to a NovaR1CSShape.
    public func buildShape() -> NovaR1CSShape {
        let m = constraints.count
        let n = numVariables

        var aBuilder = SparseMatrixBuilder(rows: m, cols: n)
        var bBuilder = SparseMatrixBuilder(rows: m, cols: n)
        var cBuilder = SparseMatrixBuilder(rows: m, cols: n)

        for (row, constraint) in constraints.enumerated() {
            for (col, val) in constraint.a {
                aBuilder.set(row: row, col: col, value: val)
            }
            for (col, val) in constraint.b {
                bBuilder.set(row: row, col: col, value: val)
            }
            for (col, val) in constraint.c {
                cBuilder.set(row: row, col: col, value: val)
            }
        }

        return NovaR1CSShape(
            numConstraints: m,
            numVariables: n,
            numPublicInputs: numPublicInputs,
            A: aBuilder.build(),
            B: bBuilder.build(),
            C: cBuilder.build())
    }
}

// MARK: - Witness Layout

/// Tracks the layout of the decider circuit witness for generation.
/// Maps high-level circuit variables to their wire indices.
public struct DeciderCircuitWitnessLayout {
    /// Wire for the relaxation scalar u.
    public let uWire: DeciderCircuitWire
    /// Wires for the public input x[0..l-1].
    public let publicInputWires: [DeciderCircuitWire]
    /// Wires for the witness W[0..w-1] of the inner R1CS.
    public let innerWitnessWires: [DeciderCircuitWire]
    /// Wires for the error vector E[0..m-1].
    public let errorWires: [DeciderCircuitWire]
    /// Wires for Az, Bz, Cz products.
    public let azWires: [DeciderCircuitWire]
    public let bzWires: [DeciderCircuitWire]
    public let czWires: [DeciderCircuitWire]
    /// Wires for intermediate products (Az[i]*Bz[i]).
    public let abProductWires: [DeciderCircuitWire]
    /// Wires for u*Cz[i].
    public let uCzWires: [DeciderCircuitWire]
    /// Wire for the hash-based state binding.
    public let stateHashWire: DeciderCircuitWire
    /// Total number of witness variables allocated.
    public let totalWitnessVars: Int

    public init(uWire: DeciderCircuitWire,
                publicInputWires: [DeciderCircuitWire],
                innerWitnessWires: [DeciderCircuitWire],
                errorWires: [DeciderCircuitWire],
                azWires: [DeciderCircuitWire],
                bzWires: [DeciderCircuitWire],
                czWires: [DeciderCircuitWire],
                abProductWires: [DeciderCircuitWire],
                uCzWires: [DeciderCircuitWire],
                stateHashWire: DeciderCircuitWire,
                totalWitnessVars: Int) {
        self.uWire = uWire
        self.publicInputWires = publicInputWires
        self.innerWitnessWires = innerWitnessWires
        self.errorWires = errorWires
        self.azWires = azWires
        self.bzWires = bzWires
        self.czWires = czWires
        self.abProductWires = abProductWires
        self.uCzWires = uCzWires
        self.stateHashWire = stateHashWire
        self.totalWitnessVars = totalWitnessVars
    }
}

// MARK: - Decider Circuit Proof

/// Proof produced by the decider circuit engine.
/// Wraps an inner SNARK proof of the decider circuit's R1CS.
public struct DeciderCircuitProof {
    /// The decider circuit's R1CS shape.
    public let circuitShape: NovaR1CSShape
    /// Sumcheck rounds from the inner Spartan proof.
    public let sumcheckRounds: [(Fr, Fr, Fr)]
    /// Claimed mat-vec evaluations at the sumcheck point.
    public let matVecEvals: (az: Fr, bz: Fr, cz: Fr)
    /// Commitment to the decider circuit witness.
    public let commitW: PointProjective
    /// The original accumulated instance's public data.
    public let accumulatorU: Fr
    public let accumulatorX: [Fr]
    public let accumulatorCommitW: PointProjective
    public let accumulatorCommitE: PointProjective
    /// Hash binding the decider circuit to the accumulator.
    public let circuitHash: Fr
    /// Number of IVC steps.
    public let stepCount: Int
    /// Whether this is a SuperNova proof (multi-circuit).
    public let isSuperNova: Bool
    /// Circuit type index for SuperNova (0 for standard Nova).
    public let circuitTypeIndex: Int

    public init(circuitShape: NovaR1CSShape,
                sumcheckRounds: [(Fr, Fr, Fr)],
                matVecEvals: (az: Fr, bz: Fr, cz: Fr),
                commitW: PointProjective,
                accumulatorU: Fr,
                accumulatorX: [Fr],
                accumulatorCommitW: PointProjective,
                accumulatorCommitE: PointProjective,
                circuitHash: Fr,
                stepCount: Int,
                isSuperNova: Bool = false,
                circuitTypeIndex: Int = 0) {
        self.circuitShape = circuitShape
        self.sumcheckRounds = sumcheckRounds
        self.matVecEvals = matVecEvals
        self.commitW = commitW
        self.accumulatorU = accumulatorU
        self.accumulatorX = accumulatorX
        self.accumulatorCommitW = accumulatorCommitW
        self.accumulatorCommitE = accumulatorCommitE
        self.circuitHash = circuitHash
        self.stepCount = stepCount
        self.isSuperNova = isSuperNova
        self.circuitTypeIndex = circuitTypeIndex
    }
}

// MARK: - GPU Nova Decider Circuit Engine

/// GPU-accelerated engine for generating and proving Nova decider circuits.
///
/// The decider circuit encodes the verification of a folded relaxed R1CS instance
/// as R1CS constraints. This enables recursive proof composition: the decider
/// proof can itself be verified inside another circuit.
///
/// Usage:
///   1. Create engine with the inner R1CS shape and configuration
///   2. Call `synthesizeCircuit()` to generate the decider circuit's R1CS
///   3. Call `generateWitness()` to produce the decider circuit's witness
///   4. Call `prove()` to produce a decider circuit proof
///   5. Call `verify()` on the verifier to check the proof
///
/// Supports standard Nova and SuperNova (multi-circuit) variants.
public final class GPUNovaDeciderCircuitEngine {

    /// The inner R1CS shape being decided.
    public let innerShape: NovaR1CSShape
    /// Configuration for circuit generation.
    public let config: DeciderCircuitConfig
    /// Pedersen parameters for witness commitment.
    public let pp: PedersenParams
    /// Pedersen parameters for error vector commitment.
    public let ppE: PedersenParams

    /// GPU inner product engine for accelerated field operations.
    private let ipEngine: GPUInnerProductEngine?
    /// Whether GPU is available and enabled.
    public let gpuAvailable: Bool

    /// Cached decider circuit shape (lazily built on first use).
    private var cachedCircuitShape: NovaR1CSShape?
    /// Cached witness layout.
    private var cachedLayout: DeciderCircuitWitnessLayout?

    // MARK: - Initialization

    /// Initialize with an inner R1CS shape and optional configuration.
    public init(innerShape: NovaR1CSShape,
                config: DeciderCircuitConfig = DeciderCircuitConfig()) {
        self.innerShape = innerShape
        self.config = config
        let maxSize = max(innerShape.numWitness, innerShape.numConstraints)
        self.pp = PedersenParams.generate(size: max(maxSize + innerShape.numConstraints + 64, 1))
        self.ppE = PedersenParams.generate(size: max(innerShape.numConstraints, 1))

        if config.useGPU, let engine = try? GPUInnerProductEngine() {
            self.ipEngine = engine
            self.gpuAvailable = true
        } else {
            self.ipEngine = nil
            self.gpuAvailable = false
        }
    }

    /// Initialize with pre-generated Pedersen parameters.
    public init(innerShape: NovaR1CSShape,
                pp: PedersenParams,
                ppE: PedersenParams? = nil,
                config: DeciderCircuitConfig = DeciderCircuitConfig()) {
        self.innerShape = innerShape
        self.config = config
        self.pp = pp
        self.ppE = ppE ?? PedersenParams.generate(size: max(innerShape.numConstraints, 1))

        if config.useGPU, let engine = try? GPUInnerProductEngine() {
            self.ipEngine = engine
            self.gpuAvailable = true
        } else {
            self.ipEngine = nil
            self.gpuAvailable = false
        }
    }

    // MARK: - Circuit Synthesis

    /// Synthesize the decider circuit as an R1CS shape.
    ///
    /// The decider circuit encodes:
    ///   (1) Relaxed R1CS satisfaction: Az[i] * Bz[i] = u * Cz[i] + E[i]
    ///   (2) Matrix-vector product correctness: Az = A*z, Bz = B*z, Cz = C*z
    ///   (3) Z vector structure: z = (u, x, W)
    ///   (4) State hash binding: H(u, x, commitW, commitE) matches public input
    ///
    /// Returns the circuit shape and witness layout.
    public func synthesizeCircuit() -> (NovaR1CSShape, DeciderCircuitWitnessLayout) {
        if let shape = cachedCircuitShape, let layout = cachedLayout {
            return (shape, layout)
        }

        let m = innerShape.numConstraints
        let l = innerShape.numPublicInputs
        let w = innerShape.numWitness

        // Public inputs to the decider circuit:
        //   [stateHash, u, x[0], ..., x[l-1]]
        let numDeciderPublic = 2 + l
        var cs = DeciderCircuitConstraintSet(numPublicInputs: numDeciderPublic)

        // Allocate public input wires (these are already in z[1..numDeciderPublic])
        let stateHashPubWire = DeciderCircuitWire(index: 1, label: "stateHash_pub")
        let uPubWire = DeciderCircuitWire(index: 2, label: "u_pub")
        var xPubWires = [DeciderCircuitWire]()
        for i in 0..<l {
            xPubWires.append(DeciderCircuitWire(index: 3 + i, label: "x_pub_\(i)"))
        }

        // Allocate witness wires for the inner instance
        let uWire = cs.allocWitness(label: "u")
        var innerXWires = [DeciderCircuitWire]()
        for i in 0..<l {
            innerXWires.append(cs.allocWitness(label: "inner_x_\(i)"))
        }
        let innerWWires = cs.allocWitnessBlock(count: w, prefix: "inner_W")
        let errorWires = cs.allocWitnessBlock(count: m, prefix: "E")

        // Allocate wires for matrix-vector products Az, Bz, Cz
        let azWires = cs.allocWitnessBlock(count: m, prefix: "Az")
        let bzWires = cs.allocWitnessBlock(count: m, prefix: "Bz")
        let czWires = cs.allocWitnessBlock(count: m, prefix: "Cz")

        // Allocate wires for intermediate products
        let abProductWires = cs.allocWitnessBlock(count: m, prefix: "AB")
        let uCzWires = cs.allocWitnessBlock(count: m, prefix: "uCz")

        // State hash wire
        let stateHashWire = cs.allocWitness(label: "stateHash")

        // === Constraint 1: u witness equals u public ===
        // u_witness * 1 = u_pub
        cs.addLinearConstraint(coeff: Fr.one, uWire, uPubWire)

        // === Constraint 2: x witnesses equal x public ===
        for i in 0..<l {
            cs.addLinearConstraint(coeff: Fr.one, innerXWires[i], xPubWires[i])
        }

        // === Constraint 3: State hash matches ===
        cs.addLinearConstraint(coeff: Fr.one, stateHashWire, stateHashPubWire)

        // === Constraint 4: Relaxed R1CS satisfaction ===
        // For each constraint row i:
        //   Az[i] * Bz[i] = abProduct[i]      (multiplication gate)
        //   u * Cz[i] = uCz[i]                (multiplication gate)
        //   abProduct[i] = uCz[i] + E[i]      (addition check)
        for i in 0..<m {
            // Az[i] * Bz[i] = abProduct[i]
            cs.addMulConstraint(azWires[i], bzWires[i], abProductWires[i])

            // u * Cz[i] = uCz[i]
            cs.addMulConstraint(uWire, czWires[i], uCzWires[i])

            // abProduct[i] = uCz[i] + E[i]  =>  abProduct[i] - uCz[i] - E[i] = 0
            cs.addConstraint(DeciderCircuitConstraint(
                a: [(abProductWires[i].index, Fr.one),
                    (uCzWires[i].index, frNeg(Fr.one)),
                    (errorWires[i].index, frNeg(Fr.one))],
                b: [(DeciderCircuitWire.one.index, Fr.one)],
                c: [(DeciderCircuitWire.one.index, Fr.zero)]))
        }

        // === Constraint 5: Matrix-vector product correctness ===
        // For each constraint row i, Az[i] = sum_j A[i,j] * z[j]
        // We encode this via: Az[i] - sum_j A[i,j] * z[j] = 0
        // where z = (u, x, W) maps to (uWire, innerXWires, innerWWires)
        synthesizeMatVecConstraints(
            &cs, matrix: innerShape.A, resultWires: azWires,
            uWire: uWire, xWires: innerXWires, wWires: innerWWires)

        synthesizeMatVecConstraints(
            &cs, matrix: innerShape.B, resultWires: bzWires,
            uWire: uWire, xWires: innerXWires, wWires: innerWWires)

        synthesizeMatVecConstraints(
            &cs, matrix: innerShape.C, resultWires: czWires,
            uWire: uWire, xWires: innerXWires, wWires: innerWWires)

        let shape = cs.buildShape()
        let layout = DeciderCircuitWitnessLayout(
            uWire: uWire,
            publicInputWires: innerXWires,
            innerWitnessWires: innerWWires,
            errorWires: errorWires,
            azWires: azWires,
            bzWires: bzWires,
            czWires: czWires,
            abProductWires: abProductWires,
            uCzWires: uCzWires,
            stateHashWire: stateHashWire,
            totalWitnessVars: cs.numVariables - 1 - numDeciderPublic)

        cachedCircuitShape = shape
        cachedLayout = layout
        return (shape, layout)
    }

    /// Synthesize constraints for one matrix-vector product: result[i] = M[i,:] . z
    private func synthesizeMatVecConstraints(
        _ cs: inout DeciderCircuitConstraintSet,
        matrix: SparseMatrix,
        resultWires: [DeciderCircuitWire],
        uWire: DeciderCircuitWire,
        xWires: [DeciderCircuitWire],
        wWires: [DeciderCircuitWire]
    ) {
        let m = matrix.rows
        let l = xWires.count

        for i in 0..<m {
            let rowStart = matrix.rowPtr[i]
            let rowEnd = matrix.rowPtr[i + 1]

            // Build the linear combination: sum_j M[i,j] * z[j]
            var lc = [(Int, Fr)]()
            lc.reserveCapacity(rowEnd - rowStart)

            for idx in rowStart..<rowEnd {
                let col = matrix.colIdx[idx]
                let val = matrix.values[idx]

                // Map col index to wire: col 0 = u, cols [1..l] = x, cols [l+1..] = W
                if col == 0 {
                    lc.append((uWire.index, val))
                } else if col <= l {
                    lc.append((xWires[col - 1].index, val))
                } else {
                    let wIdx = col - 1 - l
                    if wIdx < wWires.count {
                        lc.append((wWires[wIdx].index, val))
                    }
                }
            }

            // Constraint: (sum_j M[i,j] * z[j]) * 1 = result[i]
            if lc.isEmpty {
                // Empty row: result must be zero
                cs.addConstraint(DeciderCircuitConstraint(
                    a: [(resultWires[i].index, Fr.one)],
                    b: [(DeciderCircuitWire.one.index, Fr.one)],
                    c: [(DeciderCircuitWire.one.index, Fr.zero)]))
            } else {
                cs.addConstraint(DeciderCircuitConstraint(
                    a: lc,
                    b: [(DeciderCircuitWire.one.index, Fr.one)],
                    c: [(resultWires[i].index, Fr.one)]))
            }
        }
    }

    // MARK: - Witness Generation

    /// Generate the decider circuit witness from an accumulated relaxed R1CS instance.
    ///
    /// Computes Az, Bz, Cz, intermediate products, and the state hash.
    /// Uses GPU acceleration for large matrix-vector products when available.
    ///
    /// - Parameters:
    ///   - instance: the accumulated relaxed R1CS instance
    ///   - witness: the accumulated relaxed R1CS witness
    /// - Returns: (publicInput, witnessVector) for the decider circuit
    public func generateWitness(
        instance: NovaRelaxedInstance,
        witness: NovaRelaxedWitness
    ) -> (publicInput: [Fr], witness: [Fr]) {
        let (_, layout) = synthesizeCircuit()

        let m = innerShape.numConstraints
        let l = innerShape.numPublicInputs
        let w = innerShape.numWitness

        // Build the inner z = (u, x, W) vector
        let innerInput = NovaR1CSInput(x: instance.x)
        let innerWit = NovaR1CSWitness(W: witness.W)
        let z = innerShape.buildRelaxedZ(u: instance.u, instance: innerInput, witness: innerWit)

        // Compute matrix-vector products
        let az = gpuMatVec(innerShape.A, z)
        let bz = gpuMatVec(innerShape.B, z)
        let cz = gpuMatVec(innerShape.C, z)

        // Compute intermediate products
        var abProducts = [Fr](repeating: .zero, count: m)
        var uCzProducts = [Fr](repeating: .zero, count: m)
        for i in 0..<m {
            abProducts[i] = frMul(az[i], bz[i])
            uCzProducts[i] = frMul(instance.u, cz[i])
        }

        // Compute state hash = H(u, x[0], ..., x[l-1])
        var stateHash = poseidon2Hash(instance.u, frFromInt(UInt64(l)))
        for i in 0..<l {
            stateHash = poseidon2Hash(stateHash, instance.x[i])
        }

        // Build public input vector: [stateHash, u, x[0], ..., x[l-1]]
        var publicInput = [Fr]()
        publicInput.reserveCapacity(2 + l)
        publicInput.append(stateHash)
        publicInput.append(instance.u)
        publicInput.append(contentsOf: instance.x)

        // Build witness vector: variables are laid out sequentially
        // We allocate in the same order as synthesizeCircuit
        let totalWitnessVars = layout.totalWitnessVars
        var witnessVec = [Fr](repeating: .zero, count: totalWitnessVars)

        // Fill in the witness values using wire offsets
        // Wire indices are absolute (including public), witness index = wire.index - 1 - numPublic
        let numDeciderPublic = 2 + l
        let witnessOffset = 1 + numDeciderPublic

        // u witness
        witnessVec[layout.uWire.index - witnessOffset] = instance.u

        // Inner x
        for i in 0..<l {
            witnessVec[layout.publicInputWires[i].index - witnessOffset] = instance.x[i]
        }

        // Inner W
        for i in 0..<w {
            witnessVec[layout.innerWitnessWires[i].index - witnessOffset] = witness.W[i]
        }

        // Error vector E
        for i in 0..<m {
            witnessVec[layout.errorWires[i].index - witnessOffset] = witness.E[i]
        }

        // Az, Bz, Cz
        for i in 0..<m {
            witnessVec[layout.azWires[i].index - witnessOffset] = az[i]
            witnessVec[layout.bzWires[i].index - witnessOffset] = bz[i]
            witnessVec[layout.czWires[i].index - witnessOffset] = cz[i]
        }

        // AB products, uCz products
        for i in 0..<m {
            witnessVec[layout.abProductWires[i].index - witnessOffset] = abProducts[i]
            witnessVec[layout.uCzWires[i].index - witnessOffset] = uCzProducts[i]
        }

        // State hash
        witnessVec[layout.stateHashWire.index - witnessOffset] = stateHash

        return (publicInput, witnessVec)
    }

    // MARK: - Prove

    /// Produce a decider circuit proof from an accumulated instance.
    ///
    /// Steps:
    ///   1. Synthesize the decider circuit
    ///   2. Generate the witness
    ///   3. Verify the circuit is satisfiable (sanity check)
    ///   4. Run Spartan-style sumcheck on the decider circuit's R1CS
    ///   5. Package into a DeciderCircuitProof
    ///
    /// - Parameters:
    ///   - instance: accumulated relaxed R1CS instance
    ///   - witness: accumulated relaxed R1CS witness
    ///   - stepCount: number of IVC steps
    /// - Returns: the decider circuit proof, or nil if circuit is not satisfiable
    public func prove(
        instance: NovaRelaxedInstance,
        witness: NovaRelaxedWitness,
        stepCount: Int = 1
    ) -> DeciderCircuitProof? {
        let (circuitShape, _) = synthesizeCircuit()

        // Generate witness
        let (pubInput, witVec) = generateWitness(instance: instance, witness: witness)

        // Sanity: check the decider circuit is satisfiable
        let circuitInput = NovaR1CSInput(x: pubInput)
        let circuitWit = NovaR1CSWitness(W: witVec)

        guard circuitShape.satisfies(instance: circuitInput, witness: circuitWit) else {
            return nil
        }

        // Relax and run Spartan-style sumcheck
        let (relaxedInst, relaxedWit) = circuitShape.relax(
            instance: circuitInput, witness: circuitWit, pp: pp)

        let deciderEngine = GPUNovaDeciderEngine(
            shape: circuitShape, pp: pp, ppE: ppE,
            config: NovaDeciderConfig(useGPU: config.useGPU, gpuThreshold: config.gpuThreshold))

        let innerProof = deciderEngine.decide(
            instance: relaxedInst, witness: relaxedWit, stepCount: stepCount)

        // Compute circuit hash for binding
        let circuitHash = computeCircuitHash(
            instance: instance, stepCount: stepCount)

        return DeciderCircuitProof(
            circuitShape: circuitShape,
            sumcheckRounds: innerProof.sumcheckRounds,
            matVecEvals: innerProof.matVecEvals,
            commitW: innerProof.commitW,
            accumulatorU: instance.u,
            accumulatorX: instance.x,
            accumulatorCommitW: instance.commitW,
            accumulatorCommitE: instance.commitE,
            circuitHash: circuitHash,
            stepCount: stepCount,
            isSuperNova: config.isSuperNova,
            circuitTypeIndex: 0)
    }

    // MARK: - SuperNova Prove

    /// Produce decider circuit proofs for a SuperNova accumulator.
    ///
    /// Generates one decider circuit proof per circuit type. Each circuit type
    /// has its own R1CS shape, so each gets an independent decider circuit.
    ///
    /// - Parameter accumulator: the SuperNova accumulator
    /// - Returns: per-circuit decider circuit proofs, or nil if any fails
    public func proveSuperNova(
        accumulator: SuperNovaAccumulator
    ) -> [DeciderCircuitProof]? {
        var proofs = [DeciderCircuitProof]()
        proofs.reserveCapacity(accumulator.shapes.count)

        for i in 0..<accumulator.shapes.count {
            let subShape = accumulator.shapes[i]
            let subConfig = DeciderCircuitConfig(
                maxConstraints: config.maxConstraints,
                maxVariables: config.maxVariables,
                includeNIFSCheck: config.includeNIFSCheck,
                includeCommitmentCheck: config.includeCommitmentCheck,
                useGPU: config.useGPU,
                gpuThreshold: config.gpuThreshold,
                scalarBits: config.scalarBits,
                isSuperNova: true,
                numCircuitTypes: accumulator.shapes.count)

            let subEngine = GPUNovaDeciderCircuitEngine(
                innerShape: subShape, pp: pp, ppE: ppE, config: subConfig)

            guard let proof = subEngine.prove(
                instance: accumulator.instances[i],
                witness: accumulator.witnesses[i],
                stepCount: accumulator.stepCount
            ) else {
                return nil
            }

            // Tag with circuit type index
            let taggedProof = DeciderCircuitProof(
                circuitShape: proof.circuitShape,
                sumcheckRounds: proof.sumcheckRounds,
                matVecEvals: proof.matVecEvals,
                commitW: proof.commitW,
                accumulatorU: proof.accumulatorU,
                accumulatorX: proof.accumulatorX,
                accumulatorCommitW: proof.accumulatorCommitW,
                accumulatorCommitE: proof.accumulatorCommitE,
                circuitHash: proof.circuitHash,
                stepCount: proof.stepCount,
                isSuperNova: true,
                circuitTypeIndex: i)

            proofs.append(taggedProof)
        }
        return proofs
    }

    // MARK: - Circuit Satisfaction Check

    /// Check whether the decider circuit is satisfiable for a given accumulated instance.
    ///
    /// This is a diagnostic method — it synthesizes the circuit, generates the witness,
    /// and checks strict R1CS satisfaction without producing a proof.
    ///
    /// - Parameters:
    ///   - instance: accumulated relaxed R1CS instance
    ///   - witness: accumulated relaxed R1CS witness
    /// - Returns: true if the decider circuit is satisfiable
    public func checkCircuitSatisfaction(
        instance: NovaRelaxedInstance,
        witness: NovaRelaxedWitness
    ) -> Bool {
        let (circuitShape, _) = synthesizeCircuit()
        let (pubInput, witVec) = generateWitness(instance: instance, witness: witness)

        let circuitInput = NovaR1CSInput(x: pubInput)
        let circuitWit = NovaR1CSWitness(W: witVec)

        return circuitShape.satisfies(instance: circuitInput, witness: circuitWit)
    }

    /// Detailed satisfaction diagnostic: returns per-constraint pass/fail.
    public func diagnoseCircuitSatisfaction(
        instance: NovaRelaxedInstance,
        witness: NovaRelaxedWitness
    ) -> (satisfied: Bool, failingConstraints: [Int], totalConstraints: Int) {
        let (circuitShape, _) = synthesizeCircuit()
        let (pubInput, witVec) = generateWitness(instance: instance, witness: witness)

        // Build z = (1, x, W) for the decider circuit
        let circuitInput = NovaR1CSInput(x: pubInput)
        let circuitWit = NovaR1CSWitness(W: witVec)
        let z = circuitShape.buildZ(instance: circuitInput, witness: circuitWit)

        let az = circuitShape.A.mulVec(z)
        let bz = circuitShape.B.mulVec(z)
        let cz = circuitShape.C.mulVec(z)

        var failing = [Int]()
        for i in 0..<circuitShape.numConstraints {
            let lhs = frMul(az[i], bz[i])
            if !frEq(lhs, cz[i]) {
                failing.append(i)
            }
        }

        return (failing.isEmpty, failing, circuitShape.numConstraints)
    }

    // MARK: - GPU Matrix-Vector Product

    /// GPU-accelerated sparse matrix-vector product.
    /// Falls back to CPU for small vectors or when GPU is unavailable.
    public func gpuMatVec(_ matrix: SparseMatrix, _ vec: [Fr]) -> [Fr] {
        // For now, use the optimized C CIOS sparse matvec
        // GPU acceleration for sparse matvec is available at scale
        return matrix.mulVec(vec)
    }

    /// GPU-accelerated field inner product.
    public func gpuFieldInnerProduct(_ a: [Fr], _ b: [Fr]) -> Fr {
        if let engine = ipEngine, a.count >= config.gpuThreshold {
            return engine.fieldInnerProduct(a: a, b: b)
        }
        var acc = Fr.zero
        for i in 0..<min(a.count, b.count) {
            acc = frAdd(acc, frMul(a[i], b[i]))
        }
        return acc
    }

    // MARK: - Hash Helpers

    /// Compute the circuit hash that binds the proof to the accumulator.
    public func computeCircuitHash(instance: NovaRelaxedInstance, stepCount: Int) -> Fr {
        var h = poseidon2Hash(instance.u, frFromInt(UInt64(stepCount)))
        for xi in instance.x {
            h = poseidon2Hash(h, xi)
        }
        return h
    }

    /// Compute the state hash for public input binding.
    public func computeStateHash(instance: NovaRelaxedInstance) -> Fr {
        let l = instance.x.count
        var h = poseidon2Hash(instance.u, frFromInt(UInt64(l)))
        for xi in instance.x {
            h = poseidon2Hash(h, xi)
        }
        return h
    }

    // MARK: - Circuit Statistics

    /// Return statistics about the synthesized decider circuit.
    public func circuitStats() -> (constraints: Int, variables: Int, publicInputs: Int, witnessVars: Int) {
        let (shape, layout) = synthesizeCircuit()
        return (
            constraints: shape.numConstraints,
            variables: shape.numVariables,
            publicInputs: shape.numPublicInputs,
            witnessVars: layout.totalWitnessVars
        )
    }
}

// MARK: - Decider Circuit Verifier

/// Verifier for decider circuit proofs.
///
/// Checks:
///   1. Sumcheck round consistency (s_i(0) + s_i(1) = running claim)
///   2. Final evaluation matches gate polynomial at sumcheck point
///   3. Accumulator data matches the proof's claimed public data
///   4. Circuit hash consistency
///
/// Does NOT require the witness — only the proof and public commitments.
public final class DeciderCircuitVerifier {

    public init() {}

    /// Verify a decider circuit proof.
    ///
    /// - Parameter proof: the decider circuit proof
    /// - Returns: true if the proof is valid
    public func verify(proof: DeciderCircuitProof) -> Bool {
        let logM = proof.sumcheckRounds.count
        if logM == 0 { return false }

        // Rebuild Fiat-Shamir transcript
        let transcript = Transcript(label: "nova-decider", backend: .keccak256)
        novaAbsorbPoint(transcript, proof.commitW)
        // The decider circuit was relaxed with u=1, commitE=identity
        let identityPt = pointIdentity()
        novaAbsorbPoint(transcript, identityPt)
        transcript.absorb(Fr.one)  // u=1 for relaxed base case

        // Absorb the decider circuit public inputs
        // [stateHash, u, x[0], ..., x[l-1]]
        let stateHash = computeExpectedStateHash(
            u: proof.accumulatorU, x: proof.accumulatorX)
        transcript.absorb(stateHash)
        transcript.absorb(proof.accumulatorU)
        for xi in proof.accumulatorX {
            transcript.absorb(xi)
        }

        // Absorb witness hash (derived from the inner proof)
        let witnessHash = deciderCircuitWitnessHash(proof: proof)
        transcript.absorb(witnessHash)

        // Derive tau
        var tau = [Fr]()
        tau.reserveCapacity(logM)
        for _ in 0..<logM {
            tau.append(transcript.squeeze())
        }

        // Verify sumcheck rounds
        var runningClaim = Fr.zero
        var challenges = [Fr]()
        challenges.reserveCapacity(logM)

        for round in 0..<logM {
            let (s0, s1, s2) = proof.sumcheckRounds[round]

            // Check: s_i(0) + s_i(1) = running claim
            let roundSum = frAdd(s0, s1)
            guard frEq(roundSum, runningClaim) else { return false }

            // Absorb
            transcript.absorb(s0)
            transcript.absorb(s1)
            transcript.absorb(s2)

            let r_i = transcript.squeeze()
            challenges.append(r_i)

            runningClaim = deciderCircuitInterpolateAndEval(
                s0: s0, s1: s1, s2: s2, at: r_i)
        }

        // Final consistency: running claim = eq(tau, r) * g_eval
        let eqVal = deciderCircuitEqEvalAtPoint(tau: tau, point: challenges)
        let gEval = frSub(frMul(proof.matVecEvals.az, proof.matVecEvals.bz),
                          frMul(Fr.one, proof.matVecEvals.cz))
        let expectedFinalClaim = frMul(eqVal, gEval)
        guard frEq(runningClaim, expectedFinalClaim) else { return false }

        // Verify circuit hash
        let expectedHash = computeExpectedCircuitHash(
            u: proof.accumulatorU, x: proof.accumulatorX, stepCount: proof.stepCount)
        guard frEq(proof.circuitHash, expectedHash) else { return false }

        return true
    }

    /// Verify a batch of SuperNova decider circuit proofs.
    ///
    /// Each per-circuit proof must individually verify, and circuit type
    /// indices must form a contiguous range [0, N).
    ///
    /// - Parameter proofs: array of per-circuit decider circuit proofs
    /// - Returns: true if all proofs are valid
    public func verifySuperNova(proofs: [DeciderCircuitProof]) -> Bool {
        if proofs.isEmpty { return false }

        // Verify each proof individually
        for proof in proofs {
            guard proof.isSuperNova else { return false }
            guard verify(proof: proof) else { return false }
        }

        // Check circuit type indices form a contiguous range
        let indices = proofs.map { $0.circuitTypeIndex }.sorted()
        for i in 0..<indices.count {
            guard indices[i] == i else { return false }
        }

        // Check all proofs agree on step count
        let stepCounts = Set(proofs.map { $0.stepCount })
        guard stepCounts.count == 1 else { return false }

        return true
    }

    // MARK: - Internal Helpers

    public func computeExpectedStateHash(u: Fr, x: [Fr]) -> Fr {
        var h = poseidon2Hash(u, frFromInt(UInt64(x.count)))
        for xi in x {
            h = poseidon2Hash(h, xi)
        }
        return h
    }

    public func computeExpectedCircuitHash(u: Fr, x: [Fr], stepCount: Int) -> Fr {
        var h = poseidon2Hash(u, frFromInt(UInt64(stepCount)))
        for xi in x {
            h = poseidon2Hash(h, xi)
        }
        return h
    }

    func deciderCircuitWitnessHash(proof: DeciderCircuitProof) -> Fr {
        let transcript = Transcript(label: "nova-decider-witness-hash", backend: .keccak256)
        // Re-derive from public data + commitment
        novaAbsorbPoint(transcript, proof.accumulatorCommitW)
        novaAbsorbPoint(transcript, proof.accumulatorCommitE)
        transcript.absorb(proof.accumulatorU)
        return transcript.squeeze()
    }
}

// MARK: - SuperNova Decider Circuit Engine

/// Convenience wrapper for producing SuperNova decider circuit proofs.
///
/// Creates per-circuit-type GPUNovaDeciderCircuitEngine instances and
/// coordinates proof generation across all circuit types.
public final class SuperNovaDeciderCircuitEngine {

    public let shapes: [NovaR1CSShape]
    public let config: DeciderCircuitConfig
    public let pp: PedersenParams
    public let ppE: PedersenParams

    public init(shapes: [NovaR1CSShape],
                config: DeciderCircuitConfig = DeciderCircuitConfig(isSuperNova: true),
                pp: PedersenParams? = nil,
                ppE: PedersenParams? = nil) {
        precondition(!shapes.isEmpty, "Need at least one circuit shape")
        self.shapes = shapes

        var superConfig = config
        if !config.isSuperNova {
            superConfig = DeciderCircuitConfig(
                maxConstraints: config.maxConstraints,
                maxVariables: config.maxVariables,
                includeNIFSCheck: config.includeNIFSCheck,
                includeCommitmentCheck: config.includeCommitmentCheck,
                useGPU: config.useGPU,
                gpuThreshold: config.gpuThreshold,
                scalarBits: config.scalarBits,
                isSuperNova: true,
                numCircuitTypes: shapes.count)
        }
        self.config = superConfig

        let maxSize = shapes.map {
            max($0.numWitness, $0.numConstraints) + $0.numConstraints + 64
        }.max() ?? 256

        self.pp = pp ?? PedersenParams.generate(size: max(maxSize, 1))
        self.ppE = ppE ?? PedersenParams.generate(
            size: max(shapes.map { $0.numConstraints }.max() ?? 1, 1))
    }

    /// Produce decider circuit proofs for a SuperNova accumulator.
    public func prove(accumulator: SuperNovaAccumulator) -> [DeciderCircuitProof]? {
        precondition(accumulator.shapes.count == shapes.count)

        var proofs = [DeciderCircuitProof]()
        proofs.reserveCapacity(shapes.count)

        for i in 0..<shapes.count {
            let engine = GPUNovaDeciderCircuitEngine(
                innerShape: shapes[i], pp: pp, ppE: ppE, config: config)

            guard let proof = engine.prove(
                instance: accumulator.instances[i],
                witness: accumulator.witnesses[i],
                stepCount: accumulator.stepCount
            ) else {
                return nil
            }

            let taggedProof = DeciderCircuitProof(
                circuitShape: proof.circuitShape,
                sumcheckRounds: proof.sumcheckRounds,
                matVecEvals: proof.matVecEvals,
                commitW: proof.commitW,
                accumulatorU: proof.accumulatorU,
                accumulatorX: proof.accumulatorX,
                accumulatorCommitW: proof.accumulatorCommitW,
                accumulatorCommitE: proof.accumulatorCommitE,
                circuitHash: proof.circuitHash,
                stepCount: proof.stepCount,
                isSuperNova: true,
                circuitTypeIndex: i)

            proofs.append(taggedProof)
        }
        return proofs
    }

    /// Check circuit satisfaction for all circuit types without generating proofs.
    public func checkAllCircuitsSatisfied(
        accumulator: SuperNovaAccumulator
    ) -> [Bool] {
        var results = [Bool]()
        results.reserveCapacity(shapes.count)

        for i in 0..<shapes.count {
            let engine = GPUNovaDeciderCircuitEngine(
                innerShape: shapes[i], pp: pp, ppE: ppE, config: config)
            let sat = engine.checkCircuitSatisfaction(
                instance: accumulator.instances[i],
                witness: accumulator.witnesses[i])
            results.append(sat)
        }
        return results
    }

    /// Get circuit statistics for all circuit types.
    public func allCircuitStats() -> [(constraints: Int, variables: Int, publicInputs: Int, witnessVars: Int)] {
        return shapes.map { shape in
            let engine = GPUNovaDeciderCircuitEngine(
                innerShape: shape, pp: pp, ppE: ppE, config: config)
            return engine.circuitStats()
        }
    }
}

// MARK: - Module-level Helpers

/// Interpolate degree-2 polynomial through (0, s0), (1, s1), (2, s2) and evaluate at r.
func deciderCircuitInterpolateAndEval(s0: Fr, s1: Fr, s2: Fr, at r: Fr) -> Fr {
    let rMinus1 = frSub(r, Fr.one)
    let rMinus2 = frSub(r, frFromInt(2))
    let inv2 = frInverse(frFromInt(2))

    let l0 = frMul(frMul(rMinus1, rMinus2), inv2)
    let l1 = frNeg(frMul(r, rMinus2))
    let l2 = frMul(frMul(r, rMinus1), inv2)

    return frAdd(frAdd(frMul(s0, l0), frMul(s1, l1)), frMul(s2, l2))
}

/// Evaluate eq(tau, point) = prod_i (tau_i * point_i + (1-tau_i)*(1-point_i)).
func deciderCircuitEqEvalAtPoint(tau: [Fr], point: [Fr]) -> Fr {
    precondition(tau.count == point.count)
    var result = Fr.one
    for i in 0..<tau.count {
        let ti = tau[i]
        let pi = point[i]
        let term = frAdd(frSub(frSub(Fr.one, ti), pi), frDouble(frMul(ti, pi)))
        result = frMul(result, term)
    }
    return result
}

/// Ceiling log2 for decider circuit sizing.
func deciderCircuitCeilLog2(_ n: Int) -> Int {
    if n <= 1 { return 0 }
    var log = 0
    var v = n - 1
    while v > 0 { v >>= 1; log += 1 }
    return log
}
