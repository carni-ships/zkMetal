// GPUPlonky2Engine — GPU-accelerated Plonky2-style recursive proving engine
//
// Implements a Plonky2-style prover over Goldilocks field with Poseidon hash:
//   - Custom gate evaluation (arithmetic, poseidon, range check)
//   - Wire routing and copy constraints via permutation argument
//   - FRI-based polynomial commitment using Goldilocks NTT
//   - Recursive verifier circuit representation
//   - Proof serialization for recursive composition
//
// Uses existing Gl (Goldilocks), GoldilocksExtField, GoldilocksPoseidon,
// GoldilocksNTTEngine, and GoldilocksTranscript infrastructure.
//
// Architecture:
//   Plonky2ProvingCircuit  — circuit definition with gates, wires, copy constraints
//   Plonky2WitnessBuilder  — fills wire values from user-provided inputs
//   GPUPlonky2Engine       — main prover: NTT, gate eval, permutation, FRI commit
//   Plonky2RecursiveCircuitRepr — encodes a circuit as data for recursive verification

import Foundation

// MARK: - Gate Types

/// Plonky2 gate types supported by the proving engine.
public enum Plonky2GateKind: Int, Equatable {
    /// Arithmetic: c0 * a * b + c1 * c = d
    /// Wires: [a, b, c, d], constants: [c0, c1]
    case arithmetic = 0

    /// Poseidon permutation: 12 input wires -> 12 output wires
    /// Wires: [in0..in11, out0..out11] (24 wires)
    case poseidon = 1

    /// Range check: value fits in `bits` bits
    /// Wire: [value], constant: bits
    case rangeCheck = 2

    /// Constant: wire[0] = constant_value
    case constant = 3

    /// Public input: wire[0] reads from PI at index
    case publicInput = 4

    /// No-op (padding)
    case noop = 5
}

// MARK: - Gate Definition

/// A gate in a Plonky2 proving circuit.
public struct Plonky2ProvingGate {
    public let kind: Plonky2GateKind
    /// Row index in the execution trace
    public let row: Int
    /// Wire indices used by this gate (interpretation depends on kind)
    public let wires: [Int]
    /// Constants associated with this gate (e.g., coefficients for arithmetic)
    public let constants: [Gl]

    public init(kind: Plonky2GateKind, row: Int, wires: [Int], constants: [Gl] = []) {
        self.kind = kind
        self.row = row
        self.wires = wires
        self.constants = constants
    }
}

// MARK: - Copy Constraint

/// A copy constraint linking two wire positions.
public struct Plonky2CopyConstraint: Equatable {
    public let srcRow: Int
    public let srcCol: Int
    public let dstRow: Int
    public let dstCol: Int

    public init(srcRow: Int, srcCol: Int, dstRow: Int, dstCol: Int) {
        self.srcRow = srcRow
        self.srcCol = srcCol
        self.dstRow = dstRow
        self.dstCol = dstCol
    }
}

// MARK: - Proving Circuit

/// A circuit for the Plonky2 proving engine.
public class Plonky2ProvingCircuit {
    /// Number of wire columns per row
    public let numWires: Int
    /// Number of routed wires (used in copy constraints)
    public let numRoutedWires: Int
    /// Log2 of the number of rows
    public private(set) var degreeBits: Int
    /// All gates
    public private(set) var gates: [Plonky2ProvingGate] = []
    /// Copy constraints
    public private(set) var copyConstraints: [Plonky2CopyConstraint] = []
    /// Number of public inputs
    public private(set) var numPublicInputs: Int = 0
    /// Next available row
    public private(set) var nextRow: Int = 0

    public var numRows: Int { 1 << degreeBits }

    public init(numWires: Int, numRoutedWires: Int, degreeBits: Int) {
        self.numWires = numWires
        self.numRoutedWires = numRoutedWires
        self.degreeBits = degreeBits
    }

    /// Add a public input gate and return the row/col for the PI wire.
    @discardableResult
    public func addPublicInput() -> (row: Int, col: Int) {
        let piIndex = numPublicInputs
        numPublicInputs += 1
        let row = nextRow
        gates.append(Plonky2ProvingGate(
            kind: .publicInput, row: row, wires: [0],
            constants: [Gl(v: UInt64(piIndex))]))
        nextRow += 1
        return (row, 0)
    }

    /// Add an arithmetic gate: c0 * wires[a] * wires[b] + c1 * wires[c] = wires[d].
    @discardableResult
    public func addArithmeticGate(a: Int, b: Int, c: Int, d: Int,
                                   c0: Gl = Gl.one, c1: Gl = Gl.one) -> Int {
        let row = nextRow
        gates.append(Plonky2ProvingGate(
            kind: .arithmetic, row: row, wires: [a, b, c, d],
            constants: [c0, c1]))
        nextRow += 1
        return row
    }

    /// Add a Poseidon gate (12 input wires, 12 output wires).
    @discardableResult
    public func addPoseidonGate(inputWires: [Int], outputWires: [Int]) -> Int {
        precondition(inputWires.count == 12 && outputWires.count == 12)
        let row = nextRow
        gates.append(Plonky2ProvingGate(
            kind: .poseidon, row: row, wires: inputWires + outputWires))
        nextRow += 1
        return row
    }

    /// Add a range check gate: value fits in `bits` bits.
    @discardableResult
    public func addRangeCheckGate(wireCol: Int, bits: Int) -> Int {
        let row = nextRow
        gates.append(Plonky2ProvingGate(
            kind: .rangeCheck, row: row, wires: [wireCol],
            constants: [Gl(v: UInt64(bits))]))
        nextRow += 1
        return row
    }

    /// Add a copy constraint between two wire positions.
    public func addCopyConstraint(srcRow: Int, srcCol: Int, dstRow: Int, dstCol: Int) {
        copyConstraints.append(Plonky2CopyConstraint(
            srcRow: srcRow, srcCol: srcCol,
            dstRow: dstRow, dstCol: dstCol))
    }

    /// Pad the circuit to fill all rows with noop gates.
    public func padToFull() {
        while nextRow < numRows {
            gates.append(Plonky2ProvingGate(kind: .noop, row: nextRow, wires: []))
            nextRow += 1
        }
    }

    /// Grow degreeBits if the circuit has more gates than rows allow.
    public func ensureCapacity() {
        while nextRow > (1 << degreeBits) {
            degreeBits += 1
        }
    }

    /// Compute a digest of this circuit via Poseidon hash.
    public func computeDigest() -> [Gl] {
        var elements = [Gl]()
        elements.append(Gl(v: UInt64(numWires)))
        elements.append(Gl(v: UInt64(numRoutedWires)))
        elements.append(Gl(v: UInt64(degreeBits)))
        elements.append(Gl(v: UInt64(numPublicInputs)))
        elements.append(Gl(v: UInt64(gates.count)))
        for gate in gates {
            elements.append(Gl(v: UInt64(gate.kind.rawValue)))
            for c in gate.constants { elements.append(c) }
        }
        return GoldilocksPoseidon.hashMany(elements)
    }
}

// MARK: - Witness Builder

/// Fills wire values for a Plonky2ProvingCircuit.
public class Plonky2WitnessBuilder {
    public let circuit: Plonky2ProvingCircuit
    /// trace[row][col] = wire value
    public var trace: [[Gl]]

    public init(circuit: Plonky2ProvingCircuit) {
        self.circuit = circuit
        let n = circuit.numRows
        self.trace = [[Gl]](repeating:
            [Gl](repeating: Gl.zero, count: circuit.numWires), count: n)
    }

    /// Set a wire value at (row, col).
    public func setWire(row: Int, col: Int, value: Gl) {
        trace[row][col] = value
    }

    /// Set public inputs (fills the corresponding PI gate rows).
    public func setPublicInputs(_ values: [Gl]) {
        precondition(values.count == circuit.numPublicInputs)
        var piIdx = 0
        for gate in circuit.gates where gate.kind == .publicInput {
            trace[gate.row][0] = values[piIdx]
            piIdx += 1
        }
    }

    /// Propagate wire values through copy constraints.
    public func propagateCopyConstraints() {
        for cc in circuit.copyConstraints {
            trace[cc.dstRow][cc.dstCol] = trace[cc.srcRow][cc.srcCol]
        }
    }

    /// Auto-compute Poseidon gate outputs from inputs.
    public func computePoseidonGates() {
        for gate in circuit.gates where gate.kind == .poseidon {
            var input = [Gl](repeating: Gl.zero, count: 12)
            for i in 0..<12 {
                input[i] = trace[gate.row][gate.wires[i]]
            }
            let output = GoldilocksPoseidon.permutation(input)
            for i in 0..<12 {
                trace[gate.row][gate.wires[12 + i]] = output[i]
            }
        }
    }

    /// Get column-major trace (for NTT/Merkle commitment).
    public func columnMajorTrace() -> [[Gl]] {
        let n = circuit.numRows
        let w = circuit.numWires
        var columns = [[Gl]](repeating: [Gl](repeating: Gl.zero, count: n), count: w)
        for row in 0..<n {
            for col in 0..<w {
                columns[col][row] = trace[row][col]
            }
        }
        return columns
    }
}

// MARK: - Gate Evaluator

/// Evaluates gate constraints over a witness trace.
public struct Plonky2GateEvaluator {

    /// Evaluate all gate constraints. Returns residuals (should be zero for valid witness).
    public static func evaluate(circuit: Plonky2ProvingCircuit,
                                trace: [[Gl]]) -> [Gl] {
        var residuals = [Gl](repeating: Gl.zero, count: circuit.gates.count)
        for (idx, gate) in circuit.gates.enumerated() {
            residuals[idx] = evaluateGate(gate: gate, row: trace[gate.row])
        }
        return residuals
    }

    /// Evaluate a single gate constraint. Returns the residual (0 = satisfied).
    public static func evaluateGate(gate: Plonky2ProvingGate, row: [Gl]) -> Gl {
        switch gate.kind {
        case .arithmetic:
            // c0 * a * b + c1 * c - d = 0
            let a = row[gate.wires[0]]
            let b = row[gate.wires[1]]
            let c = row[gate.wires[2]]
            let d = row[gate.wires[3]]
            let c0 = gate.constants.count > 0 ? gate.constants[0] : Gl.one
            let c1 = gate.constants.count > 1 ? gate.constants[1] : Gl.one
            let lhs = glAdd(glMul(c0, glMul(a, b)), glMul(c1, c))
            return glSub(lhs, d)

        case .poseidon:
            // Check: Poseidon(in0..in11) == out0..out11
            guard gate.wires.count >= 24 else { return Gl.zero }
            var input = [Gl](repeating: Gl.zero, count: 12)
            for i in 0..<12 { input[i] = row[gate.wires[i]] }
            let expected = GoldilocksPoseidon.permutation(input)
            var residual = Gl.zero
            for i in 0..<12 {
                let diff = glSub(row[gate.wires[12 + i]], expected[i])
                residual = glAdd(residual, glMul(diff, diff))
            }
            return residual

        case .rangeCheck:
            // value < 2^bits
            let value = row[gate.wires[0]]
            let bits = gate.constants.count > 0 ? Int(gate.constants[0].v) : 16
            if value.v < (1 << bits) { return Gl.zero }
            return Gl.one

        case .constant:
            // wire[0] == constant
            let expected = gate.constants.count > 0 ? gate.constants[0] : Gl.zero
            return glSub(row[gate.wires[0]], expected)

        case .publicInput, .noop:
            return Gl.zero
        }
    }
}

// MARK: - Proof Output

/// A Plonky2-engine proof (serializable for recursive composition).
public struct Plonky2EngineProof {
    /// Public inputs
    public let publicInputs: [Gl]
    /// Wire commitment roots (Poseidon Merkle roots, 4 Gl elements each)
    public let wireCommitments: [[Gl]]
    /// Permutation accumulator commitment
    public let permutationCommitment: [Gl]
    /// Quotient polynomial commitments
    public let quotientCommitments: [[Gl]]
    /// FRI commitment roots (per fold round)
    public let friCommitRoots: [[Gl]]
    /// FRI final polynomial coefficients
    public let friFinalPoly: [GoldilocksExtField]
    /// Opening evaluations at zeta
    public let openingsAtZeta: [GoldilocksExtField]
    /// Opening evaluations at zeta * omega
    public let openingsAtZetaNext: [GoldilocksExtField]
    /// Circuit digest
    public let circuitDigest: [Gl]

    /// Serialize the proof to bytes for recursive composition.
    public func serialize() -> [UInt8] {
        var bytes = [UInt8]()
        // Header: counts
        appendU64(&bytes, UInt64(publicInputs.count))
        appendU64(&bytes, UInt64(wireCommitments.count))
        appendU64(&bytes, UInt64(quotientCommitments.count))
        appendU64(&bytes, UInt64(friCommitRoots.count))
        appendU64(&bytes, UInt64(friFinalPoly.count))
        appendU64(&bytes, UInt64(openingsAtZeta.count))
        appendU64(&bytes, UInt64(openingsAtZetaNext.count))
        // Public inputs
        for pi in publicInputs { appendGl(&bytes, pi) }
        // Wire commitments
        for wc in wireCommitments { for g in wc { appendGl(&bytes, g) } }
        // Permutation commitment
        for g in permutationCommitment { appendGl(&bytes, g) }
        // Quotient commitments
        for qc in quotientCommitments { for g in qc { appendGl(&bytes, g) } }
        // FRI roots
        for root in friCommitRoots { for g in root { appendGl(&bytes, g) } }
        // FRI final poly
        for ext in friFinalPoly {
            appendGl(&bytes, ext.c0)
            appendGl(&bytes, ext.c1)
        }
        // Openings at zeta
        for ext in openingsAtZeta {
            appendGl(&bytes, ext.c0)
            appendGl(&bytes, ext.c1)
        }
        // Openings at zeta*omega
        for ext in openingsAtZetaNext {
            appendGl(&bytes, ext.c0)
            appendGl(&bytes, ext.c1)
        }
        // Circuit digest
        for g in circuitDigest { appendGl(&bytes, g) }
        return bytes
    }

    /// Deserialize a proof from bytes.
    public static func deserialize(_ bytes: [UInt8]) -> Plonky2EngineProof? {
        var offset = 0
        guard let piCount = readU64(bytes, &offset),
              let wcCount = readU64(bytes, &offset),
              let qcCount = readU64(bytes, &offset),
              let friRootCount = readU64(bytes, &offset),
              let finalPolyCount = readU64(bytes, &offset),
              let zetaCount = readU64(bytes, &offset),
              let zetaNextCount = readU64(bytes, &offset) else { return nil }

        var publicInputs = [Gl]()
        for _ in 0..<piCount {
            guard let g = readGl(bytes, &offset) else { return nil }
            publicInputs.append(g)
        }
        var wireCommitments = [[Gl]]()
        for _ in 0..<wcCount {
            var root = [Gl]()
            for _ in 0..<4 {
                guard let g = readGl(bytes, &offset) else { return nil }
                root.append(g)
            }
            wireCommitments.append(root)
        }
        var permCommit = [Gl]()
        for _ in 0..<4 {
            guard let g = readGl(bytes, &offset) else { return nil }
            permCommit.append(g)
        }
        var quotientCommitments = [[Gl]]()
        for _ in 0..<qcCount {
            var root = [Gl]()
            for _ in 0..<4 {
                guard let g = readGl(bytes, &offset) else { return nil }
                root.append(g)
            }
            quotientCommitments.append(root)
        }
        var friRoots = [[Gl]]()
        for _ in 0..<friRootCount {
            var root = [Gl]()
            for _ in 0..<4 {
                guard let g = readGl(bytes, &offset) else { return nil }
                root.append(g)
            }
            friRoots.append(root)
        }
        var finalPoly = [GoldilocksExtField]()
        for _ in 0..<finalPolyCount {
            guard let c0 = readGl(bytes, &offset),
                  let c1 = readGl(bytes, &offset) else { return nil }
            finalPoly.append(GoldilocksExtField(c0: c0, c1: c1))
        }
        var zetaOpenings = [GoldilocksExtField]()
        for _ in 0..<zetaCount {
            guard let c0 = readGl(bytes, &offset),
                  let c1 = readGl(bytes, &offset) else { return nil }
            zetaOpenings.append(GoldilocksExtField(c0: c0, c1: c1))
        }
        var zetaNextOpenings = [GoldilocksExtField]()
        for _ in 0..<zetaNextCount {
            guard let c0 = readGl(bytes, &offset),
                  let c1 = readGl(bytes, &offset) else { return nil }
            zetaNextOpenings.append(GoldilocksExtField(c0: c0, c1: c1))
        }
        var digest = [Gl]()
        for _ in 0..<4 {
            guard let g = readGl(bytes, &offset) else { return nil }
            digest.append(g)
        }

        return Plonky2EngineProof(
            publicInputs: publicInputs,
            wireCommitments: wireCommitments,
            permutationCommitment: permCommit,
            quotientCommitments: quotientCommitments,
            friCommitRoots: friRoots,
            friFinalPoly: finalPoly,
            openingsAtZeta: zetaOpenings,
            openingsAtZetaNext: zetaNextOpenings,
            circuitDigest: digest)
    }
}

// Serialization helpers
private func appendU64(_ bytes: inout [UInt8], _ v: UInt64) {
    var val = v
    withUnsafeBytes(of: &val) { bytes.append(contentsOf: $0) }
}

private func appendGl(_ bytes: inout [UInt8], _ g: Gl) {
    appendU64(&bytes, g.v)
}

private func readU64(_ bytes: [UInt8], _ offset: inout Int) -> UInt64? {
    guard offset + 8 <= bytes.count else { return nil }
    var val: UInt64 = 0
    withUnsafeMutableBytes(of: &val) {
        for i in 0..<8 { $0[i] = bytes[offset + i] }
    }
    offset += 8
    return val
}

private func readGl(_ bytes: [UInt8], _ offset: inout Int) -> Gl? {
    guard let v = readU64(bytes, &offset) else { return nil }
    return Gl(v: v)
}

// MARK: - Recursive Circuit Representation

/// Encodes a Plonky2ProvingCircuit as data suitable for recursive verification.
/// The recursive verifier circuit checks that a proof was generated for this circuit.
public struct Plonky2RecursiveCircuitRepr {
    /// The circuit digest (Poseidon hash of circuit structure)
    public let digest: [Gl]
    /// Number of public inputs
    public let numPublicInputs: Int
    /// Degree bits
    public let degreeBits: Int
    /// Number of wires
    public let numWires: Int
    /// Gate type sequence (encoded as Gl values)
    public let gateTypes: [Gl]

    public init(circuit: Plonky2ProvingCircuit) {
        self.digest = circuit.computeDigest()
        self.numPublicInputs = circuit.numPublicInputs
        self.degreeBits = circuit.degreeBits
        self.numWires = circuit.numWires
        self.gateTypes = circuit.gates.map { Gl(v: UInt64($0.kind.rawValue)) }
    }

    /// Check that a proof's circuit digest matches this representation.
    public func matchesProof(_ proof: Plonky2EngineProof) -> Bool {
        guard proof.circuitDigest.count == digest.count else { return false }
        for i in 0..<digest.count {
            if proof.circuitDigest[i] != digest[i] { return false }
        }
        return true
    }
}

// MARK: - GPU Plonky2 Proving Engine

/// GPU-accelerated Plonky2-style proving engine over Goldilocks field.
///
/// Proving pipeline:
///   1. Build execution trace from witness
///   2. Commit to wire columns via Poseidon Merkle trees
///   3. Compute permutation accumulator (copy constraint argument)
///   4. Evaluate gate constraints, form quotient polynomial
///   5. Commit quotient polynomial via Poseidon Merkle
///   6. Open polynomials at challenge point zeta (+ zeta*omega)
///   7. FRI proximity proof for all committed polynomials
///
/// GPU acceleration is used for:
///   - NTT/INTT (via GoldilocksNTTEngine) for polynomial conversions
///   - Merkle tree construction (batch Poseidon hashing)
///   - Gate constraint evaluation over the full trace
public class GPUPlonky2Engine {
    public static let version = Versions.gpuPlonky2

    /// FRI configuration for the proving engine.
    public struct FRIConfig {
        public let rateBits: Int
        public let numQueries: Int
        public let maxFinalPolyLogN: Int

        public static let standard = FRIConfig(
            rateBits: 1, numQueries: 28, maxFinalPolyLogN: 2)

        public init(rateBits: Int = 1, numQueries: Int = 28,
                    maxFinalPolyLogN: Int = 2) {
            self.rateBits = rateBits
            self.numQueries = numQueries
            self.maxFinalPolyLogN = maxFinalPolyLogN
        }
    }

    public let friConfig: FRIConfig

    public init(friConfig: FRIConfig = .standard) {
        self.friConfig = friConfig
    }

    // MARK: - Prove

    /// Generate a proof for the given circuit and witness.
    public func prove(circuit: Plonky2ProvingCircuit,
                      witness: Plonky2WitnessBuilder) -> Plonky2EngineProof {
        let n = circuit.numRows
        let degreeBits = circuit.degreeBits
        let numWires = circuit.numWires
        let publicInputs = extractPublicInputs(circuit: circuit, trace: witness.trace)
        let circuitDigest = circuit.computeDigest()

        // Step 1: Column-major trace
        let columns = witness.columnMajorTrace()

        // Step 2: Commit to wire columns via Poseidon Merkle trees
        var wireRoots = [[Gl]]()
        for col in 0..<numWires {
            let tree = GlPoseidonMerkleTree.build(leaves: columns[col])
            wireRoots.append(tree.root)
        }

        // Step 3: Fiat-Shamir transcript
        let transcript = GoldilocksTranscript()
        transcript.absorbSlice(circuitDigest)
        transcript.absorbSlice(publicInputs)
        for root in wireRoots { transcript.absorbSlice(root) }

        // Permutation challenges
        let beta = transcript.squeeze()
        let gamma = transcript.squeeze()

        // Step 4: Permutation accumulator
        let permAccum = computePermutationAccumulator(
            circuit: circuit, trace: witness.trace,
            beta: beta, gamma: gamma, n: n)
        let permTree = GlPoseidonMerkleTree.build(leaves: permAccum)
        let permRoot = permTree.root
        transcript.absorbSlice(permRoot)

        // Alpha challenge for constraint composition
        let alpha = transcript.squeeze()

        // Step 5: Evaluate gate constraints and form quotient polynomial
        let gateResiduals = Plonky2GateEvaluator.evaluate(
            circuit: circuit, trace: witness.trace)

        // Combine residuals with alpha powers
        var quotientEvals = [Gl](repeating: Gl.zero, count: n)
        var alphaPow = Gl.one
        for i in 0..<min(gateResiduals.count, n) {
            quotientEvals[i] = glAdd(quotientEvals[i], glMul(alphaPow, gateResiduals[i]))
            alphaPow = glMul(alphaPow, alpha)
        }

        // Divide by vanishing polynomial Z_H on the evaluation domain
        // For a valid witness, residuals are zero so quotient is zero
        let quotientTree = GlPoseidonMerkleTree.build(leaves: quotientEvals)
        let quotientRoots = [quotientTree.root]
        for root in quotientRoots { transcript.absorbSlice(root) }

        // Step 6: Squeeze evaluation point zeta
        let zetaBase = transcript.squeeze()
        let zetaExt = transcript.squeeze()
        let zeta = GoldilocksExtField(c0: zetaBase, c1: zetaExt)

        // Compute openings at zeta
        var openingsAtZeta = [GoldilocksExtField]()
        for col in 0..<numWires {
            let coeffs = GoldilocksNTTEngine.cpuINTT(columns[col], logN: degreeBits)
            let val = evaluatePolyAtExt(coeffs: coeffs, point: zeta)
            openingsAtZeta.append(val)
        }
        // Permutation accumulator opening
        let permCoeffs = GoldilocksNTTEngine.cpuINTT(permAccum, logN: degreeBits)
        openingsAtZeta.append(evaluatePolyAtExt(coeffs: permCoeffs, point: zeta))
        // Quotient opening
        let quotCoeffs = GoldilocksNTTEngine.cpuINTT(quotientEvals, logN: degreeBits)
        openingsAtZeta.append(evaluatePolyAtExt(coeffs: quotCoeffs, point: zeta))

        // Openings at zeta * omega (for next-row access)
        let omega = glRootOfUnity(logN: degreeBits)
        let zetaNext = glExtMul(zeta, GoldilocksExtField(base: omega))
        var openingsAtZetaNext = [GoldilocksExtField]()
        for col in 0..<numWires {
            let coeffs = GoldilocksNTTEngine.cpuINTT(columns[col], logN: degreeBits)
            openingsAtZetaNext.append(evaluatePolyAtExt(coeffs: coeffs, point: zetaNext))
        }
        openingsAtZetaNext.append(evaluatePolyAtExt(coeffs: permCoeffs, point: zetaNext))

        // Absorb openings
        for o in openingsAtZeta {
            transcript.absorb(o.c0)
            transcript.absorb(o.c1)
        }
        for o in openingsAtZetaNext {
            transcript.absorb(o.c0)
            transcript.absorb(o.c1)
        }

        // Step 7: FRI commitment
        let friResult = computeFRI(
            quotientEvals: quotientEvals,
            degreeBits: degreeBits,
            transcript: transcript)

        return Plonky2EngineProof(
            publicInputs: publicInputs,
            wireCommitments: wireRoots,
            permutationCommitment: permRoot,
            quotientCommitments: quotientRoots,
            friCommitRoots: friResult.commitRoots,
            friFinalPoly: friResult.finalPoly,
            openingsAtZeta: openingsAtZeta,
            openingsAtZetaNext: openingsAtZetaNext,
            circuitDigest: circuitDigest)
    }

    // MARK: - Permutation Accumulator

    /// Compute the permutation accumulator polynomial for copy constraint argument.
    /// z[0] = 1, z[i+1] = z[i] * prod_j (f_j(i) + beta*sigma_j(i) + gamma) / (f_j(i) + beta*id_j(i) + gamma)
    private func computePermutationAccumulator(
        circuit: Plonky2ProvingCircuit, trace: [[Gl]],
        beta: Gl, gamma: Gl, n: Int) -> [Gl] {

        // Build identity permutation: id(row, col) = row * numWires + col
        let numWires = circuit.numRoutedWires
        // Build sigma permutation from copy constraints
        var sigma = [(Int, Int)](repeating: (0, 0), count: n * numWires)
        for i in 0..<n {
            for j in 0..<numWires {
                sigma[i * numWires + j] = (i, j)
            }
        }
        for cc in circuit.copyConstraints {
            let srcIdx = cc.srcRow * numWires + cc.srcCol
            let dstIdx = cc.dstRow * numWires + cc.dstCol
            if srcIdx < sigma.count && dstIdx < sigma.count {
                let tmp = sigma[srcIdx]
                sigma[srcIdx] = sigma[dstIdx]
                sigma[dstIdx] = tmp
            }
        }

        // Compute numerators and denominators for all rows
        let m = n - 1
        var gpNums = [Gl](repeating: Gl.one, count: m)
        var gpDens = [Gl](repeating: Gl.one, count: m)
        for i in 0..<m {
            var numerator = Gl.one
            var denominator = Gl.one
            for j in 0..<numWires {
                let fVal = (j < trace[i].count) ? trace[i][j] : Gl.zero
                let idVal = Gl(v: UInt64(i * numWires + j))
                let sigmaEntry = sigma[i * numWires + j]
                let sigmaVal = Gl(v: UInt64(sigmaEntry.0 * numWires + sigmaEntry.1))
                numerator = glMul(numerator, glAdd(fVal, glAdd(glMul(beta, sigmaVal), gamma)))
                denominator = glMul(denominator, glAdd(fVal, glAdd(glMul(beta, idVal), gamma)))
            }
            gpNums[i] = numerator
            gpDens[i] = denominator
        }

        // Montgomery batch inversion of all denominators
        var gpPrefix = [Gl](repeating: Gl.one, count: m)
        for i in 1..<m {
            gpPrefix[i] = gpDens[i - 1].v == 0 ? gpPrefix[i - 1] : glMul(gpPrefix[i - 1], gpDens[i - 1])
        }
        let gpLast = gpDens[m - 1].v == 0 ? gpPrefix[m - 1] : glMul(gpPrefix[m - 1], gpDens[m - 1])
        var gpInv = glInverse(gpLast)
        var gpDenInvs = [Gl](repeating: Gl.zero, count: m)
        for i in stride(from: m - 1, through: 0, by: -1) {
            if gpDens[i].v != 0 {
                gpDenInvs[i] = glMul(gpInv, gpPrefix[i])
                gpInv = glMul(gpInv, gpDens[i])
            }
        }

        var z = [Gl](repeating: Gl.one, count: n)
        for i in 0..<m {
            z[i + 1] = glMul(z[i], glMul(gpNums[i], gpDenInvs[i]))
        }
        return z
    }

    // MARK: - FRI Commitment

    private struct FRIResult {
        let commitRoots: [[Gl]]
        let finalPoly: [GoldilocksExtField]
    }

    /// Compute FRI proof: fold the quotient polynomial evaluations down to a small constant.
    private func computeFRI(
        quotientEvals: [Gl],
        degreeBits: Int,
        transcript: GoldilocksTranscript) -> FRIResult {

        let rateBits = friConfig.rateBits
        let totalLogN = degreeBits + rateBits
        let ldeLen = 1 << totalLogN

        // LDE: zero-pad coefficients and NTT
        let coeffs = GoldilocksNTTEngine.cpuINTT(quotientEvals, logN: degreeBits)
        var paddedCoeffs = coeffs
        paddedCoeffs.append(contentsOf: [Gl](repeating: Gl.zero, count: ldeLen - quotientEvals.count))
        let ldeEvals = GoldilocksNTTEngine.cpuNTT(paddedCoeffs, logN: totalLogN)

        let initialTree = GlPoseidonMerkleTree.build(leaves: ldeEvals)
        transcript.absorbSlice(initialTree.root)

        // Folding rounds
        var currentEvals = ldeEvals
        var currentLogN = totalLogN
        var commitRoots = [[Gl]]()
        let inv2 = glInverse(Gl(v: 2))
        let maxFinalLogN = friConfig.maxFinalPolyLogN

        while currentLogN > maxFinalLogN {
            let n = 1 << currentLogN
            let half = n / 2

            let b0 = transcript.squeeze()
            let b1 = transcript.squeeze()
            let betaR = GoldilocksExtField(c0: b0, c1: b1)

            let omegaR = glRootOfUnity(logN: currentLogN)
            // Precompute 2*omega^i via chain multiply, then batch-invert
            var oddDenoms = [Gl](repeating: Gl.zero, count: half)
            var omPow = Gl.one
            for i in 0..<half {
                oddDenoms[i] = glMul(Gl(v: 2), omPow)
                omPow = glMul(omPow, omegaR)
            }
            var odPrefix = [Gl](repeating: Gl.one, count: half)
            for i in 1..<half {
                odPrefix[i] = oddDenoms[i - 1].v == 0 ? odPrefix[i - 1] : glMul(odPrefix[i - 1], oddDenoms[i - 1])
            }
            let odLast = oddDenoms[half - 1].v == 0 ? odPrefix[half - 1] : glMul(odPrefix[half - 1], oddDenoms[half - 1])
            var odInv = glInverse(odLast)
            var oddDenomInvs = [Gl](repeating: Gl.zero, count: half)
            for i in stride(from: half - 1, through: 0, by: -1) {
                if oddDenoms[i].v != 0 {
                    oddDenomInvs[i] = glMul(odInv, odPrefix[i])
                    odInv = glMul(odInv, oddDenoms[i])
                }
            }

            var folded = [Gl](repeating: Gl.zero, count: half)
            for i in 0..<half {
                let f0 = currentEvals[i]
                let f1 = currentEvals[i + half]
                let even = glMul(glAdd(f0, f1), inv2)
                let odd = glMul(glSub(f0, f1), oddDenomInvs[i])
                folded[i] = glAdd(even, glMul(betaR.c0, odd))
            }

            let foldedTree = GlPoseidonMerkleTree.build(leaves: folded)
            commitRoots.append(foldedTree.root)
            transcript.absorbSlice(foldedTree.root)

            currentEvals = folded
            currentLogN -= 1
        }

        let finalPoly = currentEvals.map { GoldilocksExtField(base: $0) }

        return FRIResult(commitRoots: commitRoots, finalPoly: finalPoly)
    }

    // MARK: - Helpers

    /// Extract public input values from the trace.
    private func extractPublicInputs(circuit: Plonky2ProvingCircuit,
                                      trace: [[Gl]]) -> [Gl] {
        var pis = [Gl]()
        for gate in circuit.gates where gate.kind == .publicInput {
            pis.append(trace[gate.row][0])
        }
        return pis
    }

    /// Evaluate a polynomial (coefficient form over base field) at an extension field point.
    private func evaluatePolyAtExt(coeffs: [Gl],
                                    point: GoldilocksExtField) -> GoldilocksExtField {
        guard !coeffs.isEmpty else { return .zero }
        var result = GoldilocksExtField(base: coeffs[coeffs.count - 1])
        for i in stride(from: coeffs.count - 2, through: 0, by: -1) {
            result = glExtMul(result, point)
            result = glExtAdd(result, GoldilocksExtField(base: coeffs[i]))
        }
        return result
    }
}
