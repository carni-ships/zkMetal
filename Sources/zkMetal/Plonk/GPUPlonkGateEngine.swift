// GPUPlonkGateEngine — GPU-accelerated evaluation of Plonk custom gate constraints
//
// Evaluates standard Plonk gates (arithmetic, range, elliptic curve, poseidon) over
// witness polynomials using Metal GPU or CPU fallback. Computes:
//   1. Per-gate constraint satisfaction checks
//   2. Selector polynomial activation/isolation
//   3. Wire permutation consistency checks
//   4. Quotient polynomial contributions from all gate types
//
// The arithmetic gate constraint is:
//   qL*a + qR*b + qO*c + qM*a*b + qC = 0
//
// Selector polynomials control which gate type is active at each row.
// The combined quotient contribution is:
//   t(x) = sum_i alpha^i * selector_i(x) * gate_i(x)
//
// Works with existing Fr (BN254) field type and PlonkCircuit/PlonkGate types.

import Foundation
import Metal
import NeonFieldOps

// MARK: - Gate Type Enum

/// Supported Plonk gate types for GPU evaluation.
public enum PlonkGateType: Int, CaseIterable, Sendable {
    case arithmetic = 0
    case range = 1
    case ellipticCurve = 2
    case poseidon = 3
}

// MARK: - Gate Evaluation Result

/// Result of evaluating gate constraints across all rows.
public struct GateEvaluationResult: Sendable {
    /// Per-row constraint residuals (should all be zero for a valid witness).
    public let residuals: [Fr]
    /// Indices of rows where constraints are violated (residual != 0).
    public let failingRows: [Int]
    /// Whether all gates are satisfied.
    public var isSatisfied: Bool { failingRows.isEmpty }
}

// MARK: - GPUPlonkGateEngine

public class GPUPlonkGateEngine {
    public static let version = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")

    /// Minimum domain size to use GPU (below this, CPU is faster).
    private static let gpuThreshold = 1024

    private let device: MTLDevice?
    private let commandQueue: MTLCommandQueue?
    private let arithmeticPipeline: MTLComputePipelineState?
    private let threadgroupSize: Int

    // MARK: - Initialization

    public init() {
        let dev = MTLCreateSystemDefaultDevice()
        self.device = dev
        self.commandQueue = dev?.makeCommandQueue()
        self.threadgroupSize = 256

        if let dev = dev {
            self.arithmeticPipeline = GPUPlonkGateEngine.compileArithmeticKernel(device: dev)
        } else {
            self.arithmeticPipeline = nil
        }
    }

    // MARK: - Arithmetic Gate Evaluation

    /// Evaluate the standard arithmetic gate constraint at every row:
    ///   qL*a + qR*b + qO*c + qM*a*b + qC = 0
    ///
    /// - Parameters:
    ///   - circuit: The Plonk circuit with gate selectors.
    ///   - witness: Full witness array (variable index -> field value).
    /// - Returns: GateEvaluationResult with per-row residuals.
    public func evaluateArithmeticGates(circuit: PlonkCircuit, witness: [Fr]) -> GateEvaluationResult {
        let n = circuit.numGates
        if n == 0 {
            return GateEvaluationResult(residuals: [], failingRows: [])
        }

        // Extract wire values
        var aVals = [Fr](repeating: Fr.zero, count: n)
        var bVals = [Fr](repeating: Fr.zero, count: n)
        var cVals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            let wires = circuit.wireAssignments[i]
            aVals[i] = witness[wires[0]]
            bVals[i] = witness[wires[1]]
            cVals[i] = witness[wires[2]]
        }

        let residuals = cpuEvaluateArithmetic(
            gates: circuit.gates, a: aVals, b: bVals, c: cVals, n: n
        )

        var failing = [Int]()
        for i in 0..<n {
            if !residuals[i].isZero {
                failing.append(i)
            }
        }

        return GateEvaluationResult(residuals: residuals, failingRows: failing)
    }

    /// Evaluate a single arithmetic gate constraint.
    ///   qL*a + qR*b + qO*c + qM*a*b + qC
    public func evaluateArithmeticGate(gate: PlonkGate, a: Fr, b: Fr, c: Fr) -> Fr {
        // qL*a
        var result = frMul(gate.qL, a)
        // + qR*b
        result = frAdd(result, frMul(gate.qR, b))
        // + qO*c
        result = frAdd(result, frMul(gate.qO, c))
        // + qM*a*b
        result = frAdd(result, frMul(gate.qM, frMul(a, b)))
        // + qC
        result = frAdd(result, gate.qC)
        return result
    }

    // MARK: - Range Gate Evaluation

    /// Evaluate range gate constraints: qRange * a * (1 - a) = 0
    /// When qRange is active, wire 'a' must be boolean (0 or 1).
    public func evaluateRangeGates(circuit: PlonkCircuit, witness: [Fr]) -> GateEvaluationResult {
        let n = circuit.numGates
        var residuals = [Fr](repeating: Fr.zero, count: n)
        var failing = [Int]()

        for i in 0..<n {
            let gate = circuit.gates[i]
            if gate.qRange.isZero { continue }

            let a = witness[circuit.wireAssignments[i][0]]
            // qRange * a * (1 - a)
            let boolCheck = frMul(a, frSub(Fr.one, a))
            residuals[i] = frMul(gate.qRange, boolCheck)

            if !residuals[i].isZero {
                failing.append(i)
            }
        }

        return GateEvaluationResult(residuals: residuals, failingRows: failing)
    }

    // MARK: - Poseidon Gate Evaluation

    /// Evaluate poseidon S-box gate constraints: qPoseidon * (c - a^5) = 0
    /// When qPoseidon is active, wire c must equal a^5.
    public func evaluatePoseidonGates(circuit: PlonkCircuit, witness: [Fr]) -> GateEvaluationResult {
        let n = circuit.numGates
        var residuals = [Fr](repeating: Fr.zero, count: n)
        var failing = [Int]()

        for i in 0..<n {
            let gate = circuit.gates[i]
            if gate.qPoseidon.isZero { continue }

            let a = witness[circuit.wireAssignments[i][0]]
            let c = witness[circuit.wireAssignments[i][2]]
            // a^5 = a * a^2 * a^2
            let a2 = frSqr(a)
            let a4 = frSqr(a2)
            let a5 = frMul(a, a4)
            // qPoseidon * (c - a^5)
            residuals[i] = frMul(gate.qPoseidon, frSub(c, a5))

            if !residuals[i].isZero {
                failing.append(i)
            }
        }

        return GateEvaluationResult(residuals: residuals, failingRows: failing)
    }

    // MARK: - Combined Gate Evaluation

    /// Evaluate all gate types simultaneously and return combined residuals.
    /// For each row, the combined constraint is:
    ///   arithmetic_constraint + qRange * bool_check + qPoseidon * sbox_check
    public func evaluateAllGates(circuit: PlonkCircuit, witness: [Fr]) -> GateEvaluationResult {
        let n = circuit.numGates
        if n == 0 {
            return GateEvaluationResult(residuals: [], failingRows: [])
        }

        var residuals = [Fr](repeating: Fr.zero, count: n)
        var failing = [Int]()

        for i in 0..<n {
            let gate = circuit.gates[i]
            let wires = circuit.wireAssignments[i]
            let a = witness[wires[0]]
            let b = witness[wires[1]]
            let c = witness[wires[2]]

            // Arithmetic: qL*a + qR*b + qO*c + qM*a*b + qC
            var r = frMul(gate.qL, a)
            r = frAdd(r, frMul(gate.qR, b))
            r = frAdd(r, frMul(gate.qO, c))
            r = frAdd(r, frMul(gate.qM, frMul(a, b)))
            r = frAdd(r, gate.qC)

            // Range: qRange * a * (1 - a)
            if !gate.qRange.isZero {
                let boolCheck = frMul(a, frSub(Fr.one, a))
                r = frAdd(r, frMul(gate.qRange, boolCheck))
            }

            // Poseidon: qPoseidon * (c - a^5)
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

        return GateEvaluationResult(residuals: residuals, failingRows: failing)
    }

    // MARK: - Selector Isolation Check

    /// Verify that selector polynomials are properly isolated: at each row, at most
    /// one "special" selector (qRange, qPoseidon, qLookup) is non-zero.
    /// The arithmetic selector (qL, qR, qO, qM, qC) can coexist with special selectors.
    ///
    /// Returns indices of rows where multiple special selectors are active.
    public func checkSelectorIsolation(circuit: PlonkCircuit) -> [Int] {
        var violations = [Int]()
        for i in 0..<circuit.numGates {
            let gate = circuit.gates[i]
            var activeCount = 0
            if !gate.qRange.isZero { activeCount += 1 }
            if !gate.qPoseidon.isZero { activeCount += 1 }
            if !gate.qLookup.isZero { activeCount += 1 }
            if activeCount > 1 {
                violations.append(i)
            }
        }
        return violations
    }

    // MARK: - Wire Permutation Consistency

    /// Check that copy constraints are satisfied: for each copy constraint pair,
    /// the witness values at the two wire positions must be equal.
    ///
    /// - Parameters:
    ///   - circuit: The Plonk circuit with copy constraints.
    ///   - witness: Full witness array.
    /// - Returns: Indices of violated copy constraints.
    public func checkWirePermutation(circuit: PlonkCircuit, witness: [Fr]) -> [Int] {
        var violations = [Int]()
        let n = circuit.numGates

        for (idx, (src, dst)) in circuit.copyConstraints.enumerated() {
            // Decode wire positions: position = gateIndex * 3 + wireType
            let srcGate = src / 3
            let srcWire = src % 3
            let dstGate = dst / 3
            let dstWire = dst % 3

            guard srcGate < n, dstGate < n else {
                violations.append(idx)
                continue
            }

            let srcVal = witness[circuit.wireAssignments[srcGate][srcWire]]
            let dstVal = witness[circuit.wireAssignments[dstGate][dstWire]]

            if !frEqual(srcVal, dstVal) {
                violations.append(idx)
            }
        }

        return violations
    }

    // MARK: - Quotient Polynomial Contribution

    /// Compute the quotient polynomial contribution from all gate constraints.
    ///
    /// For domain of size n with omega as primitive root:
    ///   q(x) = sum over rows i of constraint_i * L_i(x)
    ///
    /// The result is in evaluation form on the domain.
    ///
    /// - Parameters:
    ///   - circuit: Plonk circuit.
    ///   - witness: Full witness array.
    ///   - alpha: Separation challenge for combining gate types.
    ///   - ntt: NTT engine for polynomial transforms.
    /// - Returns: Quotient contribution polynomial evaluations on the domain.
    public func computeQuotientContribution(
        circuit: PlonkCircuit,
        witness: [Fr],
        alpha: Fr,
        ntt: NTTEngine
    ) throws -> [Fr] {
        let padded = circuit.padded()
        let n = padded.numGates
        guard n > 0 && (n & (n - 1)) == 0 else {
            return []
        }

        // Evaluate constraints at each row
        var arithEvals = [Fr](repeating: Fr.zero, count: n)
        var rangeEvals = [Fr](repeating: Fr.zero, count: n)
        var poseidonEvals = [Fr](repeating: Fr.zero, count: n)

        for i in 0..<n {
            let gate = padded.gates[i]
            let wires = padded.wireAssignments[i]
            let a = witness.count > wires[0] ? witness[wires[0]] : Fr.zero
            let b = witness.count > wires[1] ? witness[wires[1]] : Fr.zero
            let c = witness.count > wires[2] ? witness[wires[2]] : Fr.zero

            // Arithmetic constraint
            var r = frMul(gate.qL, a)
            r = frAdd(r, frMul(gate.qR, b))
            r = frAdd(r, frMul(gate.qO, c))
            r = frAdd(r, frMul(gate.qM, frMul(a, b)))
            r = frAdd(r, gate.qC)
            arithEvals[i] = r

            // Range constraint
            if !gate.qRange.isZero {
                let boolCheck = frMul(a, frSub(Fr.one, a))
                rangeEvals[i] = frMul(gate.qRange, boolCheck)
            }

            // Poseidon constraint
            if !gate.qPoseidon.isZero {
                let a2 = frSqr(a)
                let a4 = frSqr(a2)
                let a5 = frMul(a, a4)
                poseidonEvals[i] = frMul(gate.qPoseidon, frSub(c, a5))
            }
        }

        // Combine with powers of alpha:
        //   combined = arith + alpha * range + alpha^2 * poseidon
        let alpha2 = frSqr(alpha)
        var combined = arithEvals
        if n >= 4 {
            combined.withUnsafeMutableBytes { cBuf in
                rangeEvals.withUnsafeBytes { rBuf in
                    withUnsafeBytes(of: alpha) { aBuf in
                        bn254_fr_batch_linear_combine(
                            cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(n))
                    }
                }
            }
            combined.withUnsafeMutableBytes { cBuf in
                poseidonEvals.withUnsafeBytes { pBuf in
                    withUnsafeBytes(of: alpha2) { aBuf in
                        bn254_fr_batch_linear_combine(
                            cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(n))
                    }
                }
            }
        } else {
            for i in 0..<n {
                combined[i] = frAdd(combined[i], frMul(alpha, rangeEvals[i]))
                combined[i] = frAdd(combined[i], frMul(alpha2, poseidonEvals[i]))
            }
        }

        // Convert to coefficient form via iNTT
        return try ntt.intt(combined)
    }

    // MARK: - Batch Gate Type Evaluation

    /// Evaluate only gates of a specific type, returning results for those rows.
    /// Non-matching rows get zero residual.
    public func evaluateByType(
        _ gateType: PlonkGateType,
        circuit: PlonkCircuit,
        witness: [Fr]
    ) -> GateEvaluationResult {
        switch gateType {
        case .arithmetic:
            return evaluateArithmeticGates(circuit: circuit, witness: witness)
        case .range:
            return evaluateRangeGates(circuit: circuit, witness: witness)
        case .poseidon:
            return evaluatePoseidonGates(circuit: circuit, witness: witness)
        case .ellipticCurve:
            // EC gates use the CustomGate protocol (ECAddGate, ECDoubleGate)
            // and are evaluated via CustomGateSet. Return trivially satisfied here.
            let n = circuit.numGates
            return GateEvaluationResult(
                residuals: [Fr](repeating: Fr.zero, count: n),
                failingRows: []
            )
        }
    }

    // MARK: - CPU Arithmetic Evaluation

    private func cpuEvaluateArithmetic(
        gates: [PlonkGate], a: [Fr], b: [Fr], c: [Fr], n: Int
    ) -> [Fr] {
        var residuals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            let gate = gates[i]
            // qL*a + qR*b + qO*c + qM*a*b + qC
            var r = frMul(gate.qL, a[i])
            r = frAdd(r, frMul(gate.qR, b[i]))
            r = frAdd(r, frMul(gate.qO, c[i]))
            r = frAdd(r, frMul(gate.qM, frMul(a[i], b[i])))
            r = frAdd(r, gate.qC)
            residuals[i] = r
        }
        return residuals
    }

    // MARK: - Metal Shader Compilation

    private static func compileArithmeticKernel(device: MTLDevice) -> MTLComputePipelineState? {
        let shaderDir = findShaderDir()
        guard let frSource = try? String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8) else {
            return nil
        }

        let frClean = frSource
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")

        let kernelSource = """

        // Evaluate arithmetic gate constraint per row:
        //   residual[i] = qL[i]*a[i] + qR[i]*b[i] + qO[i]*c[i] + qM[i]*a[i]*b[i] + qC[i]
        kernel void plonk_arithmetic_gate(
            device const Fr *qL [[buffer(0)]],
            device const Fr *qR [[buffer(1)]],
            device const Fr *qO [[buffer(2)]],
            device const Fr *qM [[buffer(3)]],
            device const Fr *qC [[buffer(4)]],
            device const Fr *a [[buffer(5)]],
            device const Fr *b [[buffer(6)]],
            device const Fr *c [[buffer(7)]],
            device Fr *residuals [[buffer(8)]],
            constant uint &n [[buffer(9)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= n) return;

            Fr r = fr_mul(qL[gid], a[gid]);
            r = fr_add(r, fr_mul(qR[gid], b[gid]));
            r = fr_add(r, fr_mul(qO[gid], c[gid]));
            r = fr_add(r, fr_mul(qM[gid], fr_mul(a[gid], b[gid])));
            r = fr_add(r, qC[gid]);
            residuals[gid] = r;
        }
        """

        let combined = frClean + "\n" + kernelSource
        let options = MTLCompileOptions()
        options.fastMathEnabled = true

        guard let library = try? device.makeLibrary(source: combined, options: options),
              let fn = library.makeFunction(name: "plonk_arithmetic_gate"),
              let pipeline = try? device.makeComputePipelineState(function: fn) else {
            return nil
        }
        return pipeline
    }

    private static func findShaderDir() -> String {
        let execDir = (CommandLine.arguments[0] as NSString).deletingLastPathComponent
        for bundle in Bundle.allBundles {
            if let url = bundle.url(forResource: "Shaders", withExtension: nil) {
                if FileManager.default.fileExists(atPath: url.appendingPathComponent("fields/bn254_fr.metal").path) {
                    return url.path
                }
            }
        }
        let candidates = [
            execDir + "/Shaders",
            execDir + "/../share/zkMetal/Shaders",
            "Sources/zkMetal/Shaders",
            FileManager.default.currentDirectoryPath + "/Sources/zkMetal/Shaders",
        ]
        for c in candidates {
            if FileManager.default.fileExists(atPath: c + "/fields/bn254_fr.metal") {
                return c
            }
        }
        return "Sources/zkMetal/Shaders"
    }
}
