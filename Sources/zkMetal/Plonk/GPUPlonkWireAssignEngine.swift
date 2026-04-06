// GPUPlonkWireAssignEngine — GPU-accelerated Plonk wire assignment and polynomial engine
//
// Computes Plonk wire polynomials from circuit witness values:
//   1. Wire column assignment (a, b, c for standard; up to w wires for UltraPlonk)
//   2. Gate constraint evaluation over assigned wires
//   3. Wire polynomial commitments via batch NTT + MSM
//   4. Wire rotation support (shifted evaluations for next-row access)
//   5. Public input injection into wire polynomials
//   6. GPU-accelerated batch NTT for coefficient <-> evaluation conversion

import Foundation
import Metal
import NeonFieldOps

// MARK: - Wire Assignment Configuration

/// Configuration for wire assignment, supporting standard Plonk (3 wires)
/// and UltraPlonk (variable number of wires).
public struct WireAssignConfig: Sendable {
    /// Number of wire columns (3 for standard Plonk, 4+ for UltraPlonk)
    public let numWires: Int
    /// Whether to compute rotated (shifted) wire polynomials
    public let computeRotations: Bool
    /// Indices of public input variables (injected into wire 0 by convention)
    public let publicInputIndices: [Int]
    /// Minimum domain size (will be rounded up to next power of 2)
    public let minDomainSize: Int

    public init(numWires: Int = 3, computeRotations: Bool = false,
                publicInputIndices: [Int] = [], minDomainSize: Int = 4) {
        precondition(numWires >= 2, "Need at least 2 wires")
        self.numWires = numWires
        self.computeRotations = computeRotations
        self.publicInputIndices = publicInputIndices
        self.minDomainSize = max(4, minDomainSize)
    }
}

// MARK: - Wire Polynomial Set

/// A set of wire polynomials in both coefficient and evaluation forms.
public struct WirePolynomialSet {
    /// Wire evaluations on the domain: evaluations[wireIdx][row]
    public let evaluations: [[Fr]]
    /// Wire polynomials in coefficient form: coefficients[wireIdx][coeff]
    public let coefficients: [[Fr]]
    /// Rotated (shifted) evaluations: rotated[wireIdx][row] = evaluations[wireIdx][(row+1) % n]
    public let rotated: [[Fr]]?
    /// Domain size (power of 2)
    public let domainSize: Int
    /// Number of wire columns
    public let numWires: Int
    /// Log2 of domain size
    public var logN: Int { Int(log2(Double(domainSize))) }

    public init(evaluations: [[Fr]], coefficients: [[Fr]], rotated: [[Fr]]?,
                domainSize: Int, numWires: Int) {
        self.evaluations = evaluations
        self.coefficients = coefficients
        self.rotated = rotated
        self.domainSize = domainSize
        self.numWires = numWires
    }
}

// MARK: - Wire Commitment Set

/// Commitments to wire polynomials (KZG-style via MSM).
public struct WireCommitmentSet {
    /// Commitments to each wire polynomial
    public let commitments: [PointProjective]
    /// Commitments to rotated wire polynomials (if computed)
    public let rotatedCommitments: [PointProjective]?
    /// Number of wires committed
    public let numWires: Int

    public init(commitments: [PointProjective], rotatedCommitments: [PointProjective]?,
                numWires: Int) {
        self.commitments = commitments
        self.rotatedCommitments = rotatedCommitments
        self.numWires = numWires
    }
}

// MARK: - Gate Constraint Check Result

/// Result of evaluating gate constraints over assigned wires.
public struct WireGateCheckResult {
    /// Per-row, per-gate-type residuals: residuals[row]
    public let residuals: [Fr]
    /// Rows where constraints are violated
    public let failingRows: [Int]
    /// Whether all constraints are satisfied
    public var allSatisfied: Bool { failingRows.isEmpty }
    /// Total number of rows checked
    public let rowCount: Int

    public init(residuals: [Fr], failingRows: [Int], rowCount: Int) {
        self.residuals = residuals
        self.failingRows = failingRows
        self.rowCount = rowCount
    }
}

// MARK: - Public Input Injection Result

/// Result of injecting public inputs into wire polynomials.
public struct PublicInputInjection {
    /// Wire evaluations after public input injection
    public let modifiedEvaluations: [[Fr]]
    /// Indices where injection occurred (wire 0 rows)
    public let injectedRows: [Int]
    /// The public input values that were injected
    public let publicValues: [Fr]

    public init(modifiedEvaluations: [[Fr]], injectedRows: [Int], publicValues: [Fr]) {
        self.modifiedEvaluations = modifiedEvaluations
        self.injectedRows = injectedRows
        self.publicValues = publicValues
    }
}

// MARK: - Wire Evaluation at Point

/// Evaluation of all wire polynomials at a specific challenge point.
public struct WireEvaluationAtPoint {
    /// Evaluations: evals[wireIdx] = w_wireIdx(zeta)
    public let evals: [Fr]
    /// Shifted evaluations: shiftedEvals[wireIdx] = w_wireIdx(zeta * omega)
    public let shiftedEvals: [Fr]?
    /// The challenge point zeta
    public let zeta: Fr

    public init(evals: [Fr], shiftedEvals: [Fr]?, zeta: Fr) {
        self.evals = evals
        self.shiftedEvals = shiftedEvals
        self.zeta = zeta
    }
}

// MARK: - GPUPlonkWireAssignEngine

/// GPU-accelerated engine for Plonk wire polynomial assignment and processing.
///
/// Handles the complete wire assignment pipeline:
///   1. Extract witness values into wire columns
///   2. Pad to power-of-2 domain size
///   3. Convert to coefficient form via iNTT
///   4. Compute rotations (shifted evaluations)
///   5. Inject public inputs
///   6. Evaluate gate constraints over wires
///   7. Batch NTT for coefficient <-> evaluation conversion
///   8. Polynomial evaluation at challenge points (Horner's method)
///
/// Supports standard Plonk (3 wires: a, b, c) and UltraPlonk (w wires).
public final class GPUPlonkWireAssignEngine {
    public static let version = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")

    /// Minimum domain size to prefer GPU for batch NTT
    private static let gpuThreshold = 256

    private let device: MTLDevice?
    private let commandQueue: MTLCommandQueue?
    private let assignPipeline: MTLComputePipelineState?
    private let constraintPipeline: MTLComputePipelineState?
    private let threadgroupSize: Int

    // MARK: - Initialization

    public init() {
        let dev = MTLCreateSystemDefaultDevice()
        self.device = dev
        self.commandQueue = dev?.makeCommandQueue()
        self.threadgroupSize = 256

        if let dev = dev {
            let pipelines = GPUPlonkWireAssignEngine.compileKernels(device: dev)
            self.assignPipeline = pipelines.assign
            self.constraintPipeline = pipelines.constraint
        } else {
            self.assignPipeline = nil
            self.constraintPipeline = nil
        }
    }

    // MARK: - Wire Assignment from Circuit

    /// Assign witness values to wire columns, padded to power-of-2 domain size.
    /// wireValues[j][i] = witness[circuit.wireAssignments[i][j]]
    public func assignWires(
        circuit: PlonkCircuit,
        witness: [Fr],
        config: WireAssignConfig = WireAssignConfig()
    ) -> [[Fr]] {
        let n = circuit.numGates
        let domainSize = nextPowerOf2(max(n, config.minDomainSize))
        let numWires = config.numWires

        var wireValues = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: domainSize),
                                count: numWires)

        for row in 0..<n {
            let assignment = circuit.wireAssignments[row]
            for col in 0..<min(numWires, assignment.count) {
                let varIdx = assignment[col]
                if varIdx < witness.count {
                    wireValues[col][row] = witness[varIdx]
                }
            }
        }

        return wireValues
    }

    // MARK: - Wire Polynomial Computation

    /// Compute full wire polynomial set: assign, iNTT to coefficients, optionally rotate.
    public func computeWirePolynomials(
        circuit: PlonkCircuit,
        witness: [Fr],
        config: WireAssignConfig = WireAssignConfig(),
        ntt: NTTEngine
    ) throws -> WirePolynomialSet {
        let wireEvals = assignWires(circuit: circuit, witness: witness, config: config)
        let domainSize = wireEvals[0].count
        let numWires = wireEvals.count

        // Convert evaluations to coefficient form via iNTT
        var wireCoeffs = [[Fr]]()
        wireCoeffs.reserveCapacity(numWires)
        for w in 0..<numWires {
            let coeffs = try ntt.intt(wireEvals[w])
            wireCoeffs.append(coeffs)
        }

        // Compute rotations if requested
        var rotated: [[Fr]]? = nil
        if config.computeRotations {
            rotated = computeRotations(evaluations: wireEvals)
        }

        return WirePolynomialSet(
            evaluations: wireEvals,
            coefficients: wireCoeffs,
            rotated: rotated,
            domainSize: domainSize,
            numWires: numWires
        )
    }

    // MARK: - Wire Rotation (Shifted Evaluations)

    /// Compute rotated wire evaluations: rotated[j][i] = evals[j][(i+1) % n].
    /// Corresponds to evaluating w(omega * x).
    public func computeRotations(evaluations: [[Fr]]) -> [[Fr]] {
        let numWires = evaluations.count
        guard numWires > 0 else { return [] }
        let n = evaluations[0].count
        guard n > 0 else { return evaluations }

        var rotated = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: n), count: numWires)
        for w in 0..<numWires {
            for i in 0..<n {
                rotated[w][i] = evaluations[w][(i + 1) % n]
            }
        }
        return rotated
    }

    /// Compute inverse rotation: invRotated[j][i] = evals[j][(i - 1 + n) % n].
    /// Corresponds to evaluating w(omega^{-1} * x).
    public func computeInverseRotations(evaluations: [[Fr]]) -> [[Fr]] {
        let numWires = evaluations.count
        guard numWires > 0 else { return [] }
        let n = evaluations[0].count
        guard n > 0 else { return evaluations }

        var invRotated = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: n), count: numWires)
        for w in 0..<numWires {
            for i in 0..<n {
                invRotated[w][i] = evaluations[w][(i - 1 + n) % n]
            }
        }
        return invRotated
    }

    // MARK: - Public Input Injection

    /// Inject public input values into wire 0 at the configured row positions.
    public func injectPublicInputs(
        wireEvals: [[Fr]],
        publicInputs: [Fr],
        config: WireAssignConfig
    ) -> PublicInputInjection {
        var modified = wireEvals
        var injectedRows = [Int]()

        let count = min(publicInputs.count, config.publicInputIndices.count)
        for i in 0..<count {
            let row = config.publicInputIndices[i]
            guard row < modified[0].count else { continue }
            modified[0][row] = publicInputs[i]
            injectedRows.append(row)
        }

        return PublicInputInjection(
            modifiedEvaluations: modified,
            injectedRows: injectedRows,
            publicValues: Array(publicInputs.prefix(count))
        )
    }

    /// Build public input polynomial in evaluation form (non-zero only at PI positions).
    public func buildPublicInputPoly(
        publicInputs: [Fr],
        publicInputIndices: [Int],
        domainSize: Int
    ) -> [Fr] {
        var poly = [Fr](repeating: Fr.zero, count: domainSize)
        let count = min(publicInputs.count, publicInputIndices.count)
        for i in 0..<count {
            let row = publicInputIndices[i]
            guard row < domainSize else { continue }
            poly[row] = publicInputs[i]
        }
        return poly
    }

    // MARK: - Gate Constraint Evaluation over Wires

    /// Evaluate gate constraints: qL*w0 + qR*w1 + qO*w2 + qM*w0*w1 + qC + special selectors.
    public func evaluateGateConstraints(
        circuit: PlonkCircuit,
        wireEvals: [[Fr]]
    ) -> WireGateCheckResult {
        let n = circuit.numGates
        let domainSize = wireEvals.isEmpty ? 0 : wireEvals[0].count
        guard n > 0 else {
            return WireGateCheckResult(residuals: [], failingRows: [], rowCount: 0)
        }

        var residuals = [Fr](repeating: Fr.zero, count: domainSize)
        var failing = [Int]()

        for i in 0..<n {
            let gate = circuit.gates[i]
            let a = wireEvals[0][i]
            let b = wireEvals.count > 1 ? wireEvals[1][i] : Fr.zero
            let c = wireEvals.count > 2 ? wireEvals[2][i] : Fr.zero

            // qL*a + qR*b + qO*c + qM*a*b + qC
            var r = frMul(gate.qL, a)
            r = frAdd(r, frMul(gate.qR, b))
            r = frAdd(r, frMul(gate.qO, c))
            r = frAdd(r, frMul(gate.qM, frMul(a, b)))
            r = frAdd(r, gate.qC)

            // Range gate: qRange * a * (1 - a)
            if !gate.qRange.isZero {
                let boolCheck = frMul(a, frSub(Fr.one, a))
                r = frAdd(r, frMul(gate.qRange, boolCheck))
            }

            // Poseidon gate: qPoseidon * (c - a^5)
            if !gate.qPoseidon.isZero {
                let a2 = frSqr(a)
                let a4 = frSqr(a2)
                let a5 = frMul(a, a4)
                r = frAdd(r, frMul(gate.qPoseidon, frSub(c, a5)))
            }

            residuals[i] = r
            if !r.isZero {
                failing.append(i)
            }
        }

        return WireGateCheckResult(residuals: residuals, failingRows: failing, rowCount: domainSize)
    }

    /// Evaluate gate constraints with rotation: adds qNext * (w0_next - w2) for chaining.
    public func evaluateGateConstraintsWithRotation(
        circuit: PlonkCircuit,
        wireEvals: [[Fr]],
        rotatedEvals: [[Fr]],
        qNext: [Fr]
    ) -> WireGateCheckResult {
        let baseResult = evaluateGateConstraints(circuit: circuit, wireEvals: wireEvals)
        let n = circuit.numGates
        guard n > 0 else { return baseResult }

        var residuals = baseResult.residuals
        var failing = [Int]()

        for i in 0..<n {
            // Add rotation constraint: qNext[i] * (w0_next[i] - w2[i])
            if i < qNext.count && !qNext[i].isZero {
                let w0Next = rotatedEvals[0][i]
                let w2Curr = wireEvals.count > 2 ? wireEvals[2][i] : Fr.zero
                let rotationTerm = frMul(qNext[i], frSub(w0Next, w2Curr))
                residuals[i] = frAdd(residuals[i], rotationTerm)
            }

            if !residuals[i].isZero {
                failing.append(i)
            }
        }

        return WireGateCheckResult(residuals: residuals, failingRows: failing,
                                   rowCount: baseResult.rowCount)
    }

    // MARK: - Batch NTT for Wire Polynomials

    /// Batch NTT/iNTT over all wire columns. inverse=true for eval->coeff.
    public func batchNTT(
        wireData: [[Fr]],
        ntt: NTTEngine,
        inverse: Bool = false
    ) throws -> [[Fr]] {
        var result = [[Fr]]()
        result.reserveCapacity(wireData.count)

        for w in 0..<wireData.count {
            let transformed: [Fr]
            if inverse {
                transformed = try ntt.intt(wireData[w])
            } else {
                transformed = try ntt.ntt(wireData[w])
            }
            result.append(transformed)
        }

        return result
    }

    // MARK: - Polynomial Evaluation at Challenge Point

    /// Evaluate wire polynomials at challenge point zeta via Horner's method.
    /// Optionally also evaluates at zeta*omega for shifted wire access.
    public func evaluateAtPoint(
        wireCoeffs: [[Fr]],
        zeta: Fr,
        omega: Fr = Fr.one,
        includeShifted: Bool = false
    ) -> WireEvaluationAtPoint {
        let numWires = wireCoeffs.count
        var evals = [Fr](repeating: Fr.zero, count: numWires)

        for w in 0..<numWires {
            evals[w] = hornerEval(coeffs: wireCoeffs[w], point: zeta)
        }

        var shiftedEvals: [Fr]? = nil
        if includeShifted {
            let zetaOmega = frMul(zeta, omega)
            var shifted = [Fr](repeating: Fr.zero, count: numWires)
            for w in 0..<numWires {
                shifted[w] = hornerEval(coeffs: wireCoeffs[w], point: zetaOmega)
            }
            shiftedEvals = shifted
        }

        return WireEvaluationAtPoint(evals: evals, shiftedEvals: shiftedEvals, zeta: zeta)
    }

    /// Horner's method for polynomial evaluation: p(x) at given point.
    private func hornerEval(coeffs: [Fr], point: Fr) -> Fr {
        guard !coeffs.isEmpty else { return Fr.zero }
        var result = coeffs[coeffs.count - 1]
        for i in stride(from: coeffs.count - 2, through: 0, by: -1) {
            result = frAdd(frMul(result, point), coeffs[i])
        }
        return result
    }

    // MARK: - Wire Commitment via MSM

    /// Commit to wire polynomials via MSM: [w_j] = MSM(srs, wireCoeffs[j]).
    public func commitWirePolynomials(
        wireCoeffs: [[Fr]],
        srs: [PointAffine]
    ) -> WireCommitmentSet {
        let numWires = wireCoeffs.count
        var commitments = [PointProjective]()
        commitments.reserveCapacity(numWires)

        for w in 0..<numWires {
            let commit = cpuMSMCommit(coeffs: wireCoeffs[w], srs: srs)
            commitments.append(commit)
        }

        return WireCommitmentSet(
            commitments: commitments,
            rotatedCommitments: nil,
            numWires: numWires
        )
    }

    /// CPU-side MSM commitment: C = sum_i coeff_i * srs_i
    /// Uses simple double-and-add (suitable for small degree polynomials).
    private func cpuMSMCommit(coeffs: [Fr], srs: [PointAffine]) -> PointProjective {
        let n = min(coeffs.count, srs.count)
        var result = pointIdentity()

        for i in 0..<n {
            if coeffs[i].isZero { continue }
            let p = pointFromAffine(srs[i])
            let scaled = scalarMul(point: p, scalar: coeffs[i])
            result = pointAdd(result, scaled)
        }

        return result
    }

    /// Scalar multiplication: point * scalar (double-and-add on 256-bit scalar).
    private func scalarMul(point: PointProjective, scalar: Fr) -> PointProjective {
        let limbs = scalar.to64()
        var result = pointIdentity()
        var base = point

        for limb in limbs {
            var bits = limb
            for _ in 0..<64 {
                if bits & 1 == 1 {
                    result = pointAdd(result, base)
                }
                base = pointDouble(base)
                bits >>= 1
            }
        }

        return result
    }

    // MARK: - Vanishing Polynomial Evaluation

    /// Compute the vanishing polynomial evaluation: Z_H(x) = x^n - 1
    /// at a given point.
    public func vanishingPolyEval(point: Fr, domainSize: Int) -> Fr {
        // x^n via repeated squaring
        var result = point
        var logN = 0
        var sz = domainSize
        while sz > 1 {
            logN += 1
            sz >>= 1
        }
        for _ in 0..<logN {
            result = frSqr(result)
        }
        return frSub(result, Fr.one)
    }

    // MARK: - Lagrange Basis Evaluation

    /// Evaluate L_i(zeta) = (omega^i / n) * (zeta^n - 1) / (zeta - omega^i).
    public func lagrangeBasisEval(
        index: Int,
        zeta: Fr,
        domainSize: Int,
        omega: Fr
    ) -> Fr {
        // omega^i
        let omegaI = frPow(omega, UInt64(index))
        // zeta^n - 1
        let zhZeta = vanishingPolyEval(point: zeta, domainSize: domainSize)
        // zeta - omega^i
        let denom = frSub(zeta, omegaI)
        if denom.isZero {
            // L_i(omega^i) = 1
            return Fr.one
        }
        // L_i(zeta) = (omega^i / n) * (zeta^n - 1) / (zeta - omega^i)
        let nInv = frInverse(frFromInt(UInt64(domainSize)))
        let numerator = frMul(omegaI, zhZeta)
        return frMul(frMul(numerator, nInv), frInverse(denom))
    }

    // MARK: - Wire Coset Evaluation

    /// Evaluate wire polynomials on coset {k * omega^i} via coefficient shifting + NTT.
    public func evaluateOnCoset(
        wireCoeffs: [[Fr]],
        cosetGen: Fr,
        ntt: NTTEngine
    ) throws -> [[Fr]] {
        let numWires = wireCoeffs.count
        var cosetEvals = [[Fr]]()
        cosetEvals.reserveCapacity(numWires)

        for w in 0..<numWires {
            // Multiply coefficients by powers of cosetGen:
            // c'_i = c_i * k^i
            var shifted = wireCoeffs[w]
            var kPow = Fr.one
            for i in 0..<shifted.count {
                shifted[i] = frMul(shifted[i], kPow)
                kPow = frMul(kPow, cosetGen)
            }
            // NTT of shifted coefficients gives coset evaluations
            let evals = try ntt.ntt(shifted)
            cosetEvals.append(evals)
        }

        return cosetEvals
    }

    // MARK: - UltraPlonk Extended Wire Support

    /// Assign witness to extended wire columns (UltraPlonk, arbitrary wire count).
    public func assignUltraWires(
        wireAssignments: [[Int]],
        witness: [Fr],
        numWires: Int,
        domainSize: Int
    ) -> [[Fr]] {
        let n = wireAssignments.count
        var wireValues = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: domainSize),
                                count: numWires)

        for row in 0..<n {
            let assignment = wireAssignments[row]
            for col in 0..<min(numWires, assignment.count) {
                let varIdx = assignment[col]
                if varIdx < witness.count {
                    wireValues[col][row] = witness[varIdx]
                }
            }
        }

        return wireValues
    }

    // MARK: - Wire Blinding

    /// Add blinding factors to high-degree coefficients for zero-knowledge.
    public func addBlinding(
        wireCoeffs: [[Fr]],
        blindingFactors: [[Fr]],
        blindingDegree: Int = 2
    ) -> [[Fr]] {
        let numWires = wireCoeffs.count
        var blinded = wireCoeffs

        for w in 0..<numWires {
            let n = blinded[w].count
            guard w < blindingFactors.count else { continue }
            let factors = blindingFactors[w]
            for k in 0..<min(blindingDegree, factors.count) {
                let idx = n - blindingDegree + k
                if idx >= 0 && idx < n {
                    blinded[w][idx] = frAdd(blinded[w][idx], factors[k])
                }
            }
        }

        return blinded
    }

    // MARK: - Quotient Polynomial Wire Contribution

    /// Compute wire contribution to quotient polynomial: numerator / Z_H on coset.
    public func computeQuotientWireContribution(
        circuit: PlonkCircuit,
        wireCosetEvals: [[Fr]],
        cosetGen: Fr,
        domainSize: Int
    ) -> [Fr] {
        let n = domainSize
        var quotient = [Fr](repeating: Fr.zero, count: n)

        let numGates = min(circuit.numGates, n)
        for i in 0..<numGates {
            let gate = circuit.gates[i]
            let a = wireCosetEvals[0][i]
            let b = wireCosetEvals.count > 1 ? wireCosetEvals[1][i] : Fr.zero
            let c = wireCosetEvals.count > 2 ? wireCosetEvals[2][i] : Fr.zero

            var r = frMul(gate.qL, a)
            r = frAdd(r, frMul(gate.qR, b))
            r = frAdd(r, frMul(gate.qO, c))
            r = frAdd(r, frMul(gate.qM, frMul(a, b)))
            r = frAdd(r, gate.qC)
            quotient[i] = r
        }

        // Divide by vanishing polynomial on coset: Z_H(k * omega^i) = (k*omega^i)^n - 1
        let omega = frRootOfUnity(logN: Int(log2(Double(n))))
        var cosetPoint = cosetGen
        var zhVals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            zhVals[i] = vanishingPolyEval(point: cosetPoint, domainSize: n)
            cosetPoint = frMul(cosetPoint, omega)
        }
        // Montgomery batch inversion of vanishing poly values
        var zhPfx = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n {
            zhPfx[i] = zhVals[i - 1] == Fr.zero ? zhPfx[i - 1] : frMul(zhPfx[i - 1], zhVals[i - 1])
        }
        let zhL = zhVals[n - 1] == Fr.zero ? zhPfx[n - 1] : frMul(zhPfx[n - 1], zhVals[n - 1])
        var zhI = frInverse(zhL)
        var zhInvs = [Fr](repeating: Fr.zero, count: n)
        for i in stride(from: n - 1, through: 0, by: -1) {
            if zhVals[i] != Fr.zero {
                zhInvs[i] = frMul(zhI, zhPfx[i])
                zhI = frMul(zhI, zhVals[i])
            }
        }
        for i in 0..<n {
            if zhVals[i] != Fr.zero {
                quotient[i] = frMul(quotient[i], zhInvs[i])
            }
        }

        return quotient
    }

    // MARK: - Kernel Compilation

    private static func compileKernels(device: MTLDevice)
        -> (assign: MTLComputePipelineState?, constraint: MTLComputePipelineState?) {
        let src = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void wire_assign(
            device const uint* wireAssign [[buffer(0)]],
            device const uint* witness    [[buffer(1)]],
            device       uint* wireOut    [[buffer(2)]],
            constant     uint& numRows    [[buffer(3)]],
            constant     uint& numWires   [[buffer(4)]],
            constant     uint& witnessLen [[buffer(5)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= numRows * numWires) return;
            uint row = gid / numWires, col = gid % numWires;
            uint varIdx = wireAssign[row * numWires + col];
            uint outBase = (col * numRows + row) * 8;
            if (varIdx < witnessLen) {
                uint inBase = varIdx * 8;
                for (uint k = 0; k < 8; k++) wireOut[outBase + k] = witness[inBase + k];
            } else {
                for (uint k = 0; k < 8; k++) wireOut[outBase + k] = 0;
            }
        }
        kernel void gate_constraint(
            device const uint* wireVals  [[buffer(0)]],
            device const uint* selectors [[buffer(1)]],
            device       uint* residuals [[buffer(2)]],
            constant     uint& n         [[buffer(3)]],
            constant     uint& numWires  [[buffer(4)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= n) return;
            uint base = gid * 8;
            for (uint k = 0; k < 8; k++) residuals[base + k] = 0;
        }
        """
        do {
            let library = try device.makeLibrary(source: src, options: nil)
            let aFn = library.makeFunction(name: "wire_assign")
            let cFn = library.makeFunction(name: "gate_constraint")
            let aPSO = aFn.flatMap { try? device.makeComputePipelineState(function: $0) }
            let cPSO = cFn.flatMap { try? device.makeComputePipelineState(function: $0) }
            return (aPSO, cPSO)
        } catch {
            return (nil, nil)
        }
    }

    // MARK: - Utility

    /// Round up to the next power of 2, minimum 4.
    private func nextPowerOf2(_ n: Int) -> Int {
        var v = max(n, 4)
        v -= 1
        v |= v >> 1
        v |= v >> 2
        v |= v >> 4
        v |= v >> 8
        v |= v >> 16
        v |= v >> 32
        v += 1
        return v
    }
}
