// Groth16GPUWitness — GPU-accelerated R1CS witness generation for Groth16
//
// Takes R1CS matrices (A, B, C) and a partial witness (public inputs + hints),
// solves layer-by-layer using GPU parallel field arithmetic.
//
// Architecture:
//   1. Convert R1CS to R1CSConstraint format
//   2. Build WaveScheduler to find parallel layers of independent constraints
//   3. For each wave:
//      a. CPU: evaluate known LC parts, determine solve mode, extract coefficients
//      b. CPU: batch Montgomery inversion of denominators (1 inversion + O(n) muls)
//      c. GPU: parallel solve using pre-computed inverses (3 field muls per constraint)
//   4. GPU sparse mat-vec for large constraint evaluation (A*z, B*z, C*z)
//
// Falls back to CPU for small waves (< gpuWaveThreshold constraints).

import Foundation
import Metal
import NeonFieldOps

// MARK: - GPU Solve Info (mirrors Metal struct layout)

/// Packed solve information for one constraint, matching the Metal R1CSSolveInfo struct.
/// Must be memory-layout compatible with the GPU struct.
struct R1CSSolveInfo {
    var knownA: Fr
    var knownB: Fr
    var knownC: Fr
    var targetCoeff: Fr
    var precomputedInv: Fr
    var targetVar: UInt32
    var solveMode: UInt32   // 0=C, 1=A, 2=B, 3=skip
    var _pad0: UInt32 = 0
    var _pad1: UInt32 = 0
}

// MARK: - Groth16 GPU Witness Generator

/// GPU-accelerated Groth16 witness generator.
///
/// Uses Metal compute shaders to parallelize R1CS constraint solving.
/// Constraints are grouped into dependency waves via WaveScheduler;
/// within each wave, all constraints are independent and can be solved
/// in parallel on the GPU.
///
/// The heavy field arithmetic (BN254 Fr multiply, subtract) runs on GPU,
/// while the wave scheduling and batch inversion run on CPU. This hybrid
/// approach avoids GPU launch overhead for small waves while getting
/// parallelism for large ones.
public class Groth16GPUWitness {
    public static let version = Versions.groth16GPUWitness

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // Pipeline states
    private let solvePipeline: MTLComputePipelineState
    private let matvecPipeline: MTLComputePipelineState

    /// Minimum wave size for GPU dispatch. Smaller waves use CPU.
    public var gpuWaveThreshold: Int = 64

    /// Minimum number of total constraints to use GPU at all.
    public var gpuMinConstraints: Int = 128

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        let library = try Groth16GPUWitness.compileShaders(device: device)

        guard let solveFn = library.makeFunction(name: "r1cs_witness_solve_bn254"),
              let matvecFn = library.makeFunction(name: "r1cs_sparse_matvec_bn254") else {
            throw MSMError.missingKernel
        }

        self.solvePipeline = try device.makeComputePipelineState(function: solveFn)
        self.matvecPipeline = try device.makeComputePipelineState(function: matvecFn)
    }

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let frSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let r1csSource = try String(contentsOfFile: shaderDir + "/witness/r1cs_witness_solve.metal", encoding: .utf8)

        // Strip include guards for inline compilation
        let frClean = frSource
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")

        let r1csClean = r1csSource.split(separator: "\n")
            .filter { !$0.contains("#include") }
            .joined(separator: "\n")

        let combined = frClean + "\n" + r1csClean

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    // MARK: - Public API

    /// Generate a complete witness for a Groth16 R1CS system.
    ///
    /// Takes R1CS matrices (A, B, C) in sparse format and partial inputs,
    /// solves layer-by-layer using GPU parallel field arithmetic for large waves.
    ///
    /// - Parameters:
    ///   - r1cs: The R1CS constraint system
    ///   - publicInputs: Public input values (placed at indices 1..numPublic in z)
    ///   - hints: Optional pre-assigned witness values (variable index -> value)
    /// - Returns: Complete z vector [1, publicInputs..., witness...]
    public func generateWitness(r1cs: R1CSInstance,
                                publicInputs: [Fr],
                                hints: [Int: Fr] = [:]) -> [Fr] {
        let constraints = r1csToConstraints(r1cs: r1cs)

        // Build initial assignment
        var assignment = [Fr](repeating: Fr.zero, count: r1cs.numVars)
        assignment[0] = Fr.one
        for (i, val) in publicInputs.enumerated() { assignment[i + 1] = val }
        for (idx, val) in hints { assignment[idx] = val }

        // Determine known variables
        var known = Set<Int>()
        known.insert(0)
        for i in 0..<publicInputs.count { known.insert(i + 1) }
        for idx in hints.keys { known.insert(idx) }

        // Schedule constraints into waves
        let scheduler = WaveScheduler(constraints: constraints, knownVariables: known,
                                       numVariables: r1cs.numVars)

        let useGPU = r1cs.numConstraints >= gpuMinConstraints

        // Process each wave
        for wave in scheduler.waves {
            if useGPU && wave.count >= gpuWaveThreshold {
                solveWaveGPU(wave: wave, constraints: constraints,
                             produced: scheduler.producedVariable,
                             assignment: &assignment)
            } else {
                solveWaveCPU(wave: wave, constraints: constraints,
                             produced: scheduler.producedVariable,
                             assignment: &assignment)
            }
        }

        return assignment
    }

    /// Generate witness and return timing breakdown.
    /// Useful for benchmarking GPU vs CPU performance.
    public func generateWitnessWithProfile(r1cs: R1CSInstance,
                                            publicInputs: [Fr],
                                            hints: [Int: Fr] = [:]) -> (witness: [Fr], scheduleMs: Double, solveMs: Double, wavesGPU: Int, wavesCPU: Int) {
        let constraints = r1csToConstraints(r1cs: r1cs)

        var assignment = [Fr](repeating: Fr.zero, count: r1cs.numVars)
        assignment[0] = Fr.one
        for (i, val) in publicInputs.enumerated() { assignment[i + 1] = val }
        for (idx, val) in hints { assignment[idx] = val }

        var known = Set<Int>()
        known.insert(0)
        for i in 0..<publicInputs.count { known.insert(i + 1) }
        for idx in hints.keys { known.insert(idx) }

        let t0 = CFAbsoluteTimeGetCurrent()
        let scheduler = WaveScheduler(constraints: constraints, knownVariables: known,
                                       numVariables: r1cs.numVars)
        let t1 = CFAbsoluteTimeGetCurrent()

        let useGPU = r1cs.numConstraints >= gpuMinConstraints
        var gpuWaves = 0, cpuWaves = 0

        for wave in scheduler.waves {
            if useGPU && wave.count >= gpuWaveThreshold {
                solveWaveGPU(wave: wave, constraints: constraints,
                             produced: scheduler.producedVariable,
                             assignment: &assignment)
                gpuWaves += 1
            } else {
                solveWaveCPU(wave: wave, constraints: constraints,
                             produced: scheduler.producedVariable,
                             assignment: &assignment)
                cpuWaves += 1
            }
        }
        let t2 = CFAbsoluteTimeGetCurrent()

        return (assignment, (t1 - t0) * 1000, (t2 - t1) * 1000, gpuWaves, cpuWaves)
    }

    // MARK: - GPU Sparse Matrix-Vector Product

    /// Compute sparse matrix-vector product on GPU using CSR format.
    /// Returns result[i] = sum_j (matrix[i,j] * vec[j]) for each row.
    public func gpuSparseMatVec(entries: [R1CSEntry], numRows: Int, vec: [Fr]) -> [Fr]? {
        if numRows == 0 { return [Fr]() }

        // Convert COO to CSR
        var rowCounts = [Int](repeating: 0, count: numRows)
        for e in entries { rowCounts[e.row] += 1 }
        var rowStarts = [UInt32](repeating: 0, count: numRows + 1)
        for i in 0..<numRows { rowStarts[i + 1] = rowStarts[i] + UInt32(rowCounts[i]) }

        let nnz = entries.count
        var cols = [UInt32](repeating: 0, count: nnz)
        var vals = [Fr](repeating: Fr.zero, count: nnz)
        var offsets = [Int](repeating: 0, count: numRows)

        for e in entries {
            let pos = Int(rowStarts[e.row]) + offsets[e.row]
            cols[pos] = UInt32(e.col)
            vals[pos] = e.val
            offsets[e.row] += 1
        }

        let frStride = MemoryLayout<Fr>.stride
        let u32Stride = MemoryLayout<UInt32>.stride

        guard let resultBuf = device.makeBuffer(length: numRows * frStride, options: .storageModeShared),
              let vecBuf = device.makeBuffer(bytes: vec, length: vec.count * frStride, options: .storageModeShared),
              let colsBuf = device.makeBuffer(bytes: cols, length: max(1, nnz) * u32Stride, options: .storageModeShared),
              let valsBuf = device.makeBuffer(bytes: vals, length: max(1, nnz) * frStride, options: .storageModeShared),
              let rowStartsBuf = device.makeBuffer(bytes: rowStarts, length: (numRows + 1) * u32Stride, options: .storageModeShared),
              let cmdBuf = commandQueue.makeCommandBuffer() else {
            return nil
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(matvecPipeline)
        enc.setBuffer(resultBuf, offset: 0, index: 0)
        enc.setBuffer(vecBuf, offset: 0, index: 1)
        enc.setBuffer(colsBuf, offset: 0, index: 2)
        enc.setBuffer(valsBuf, offset: 0, index: 3)
        enc.setBuffer(rowStartsBuf, offset: 0, index: 4)
        var numRowsU32 = UInt32(numRows)
        enc.setBytes(&numRowsU32, length: 4, index: 5)

        let tg = min(256, Int(matvecPipeline.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(
            MTLSize(width: numRows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Read results
        let ptr = resultBuf.contents().bindMemory(to: Fr.self, capacity: numRows)
        return Array(UnsafeBufferPointer(start: ptr, count: numRows))
    }

    // MARK: - Wave Solving

    /// Solve a wave of constraints on CPU with batch Montgomery inversion.
    private func solveWaveCPU(wave: [Int],
                              constraints: [R1CSConstraint],
                              produced: [Int?],
                              assignment: inout [Fr]) {
        let count = wave.count

        // Phase 1: Collect solve info
        struct CPUSolveInfo {
            var knownA: Fr = Fr.zero
            var knownB: Fr = Fr.zero
            var knownC: Fr = Fr.zero
            var targetCoeff: Fr = Fr.zero
            var targetVar: Int = 0
            var mode: Int = 3  // 0=C, 1=A, 2=B, 3=skip
        }

        var infos = [CPUSolveInfo](repeating: CPUSolveInfo(), count: count)

        for i in 0..<count {
            let ci = wave[i]
            guard let targetVar = produced[ci] else { continue }
            let c = constraints[ci]
            let aHas = c.a.variables.contains(targetVar)
            let bHas = c.b.variables.contains(targetVar)
            let cHas = c.c.variables.contains(targetVar)

            infos[i].targetVar = targetVar

            if cHas && !aHas && !bHas {
                infos[i].knownA = c.a.evaluate(assignment: assignment)
                infos[i].knownB = c.b.evaluate(assignment: assignment)
                let (knownSum, coeff) = splitLC(lc: c.c, targetVar: targetVar, assignment: assignment)
                infos[i].knownC = knownSum
                infos[i].targetCoeff = coeff
                infos[i].mode = 0
            } else if aHas && !bHas && !cHas {
                infos[i].knownB = c.b.evaluate(assignment: assignment)
                infos[i].knownC = c.c.evaluate(assignment: assignment)
                let (knownSum, coeff) = splitLC(lc: c.a, targetVar: targetVar, assignment: assignment)
                infos[i].knownA = knownSum
                infos[i].targetCoeff = coeff
                infos[i].mode = 1
            } else if bHas && !aHas && !cHas {
                infos[i].knownA = c.a.evaluate(assignment: assignment)
                infos[i].knownC = c.c.evaluate(assignment: assignment)
                let (knownSum, coeff) = splitLC(lc: c.b, targetVar: targetVar, assignment: assignment)
                infos[i].knownB = knownSum
                infos[i].targetCoeff = coeff
                infos[i].mode = 2
            }
        }

        // Phase 2: Batch inversion of denominators
        var toInvert = [Fr]()
        var invertMap = [Int](repeating: -1, count: count)

        for i in 0..<count {
            let info = infos[i]
            switch info.mode {
            case 0:
                toInvert.append(info.targetCoeff)
                invertMap[i] = toInvert.count - 1
            case 1:
                toInvert.append(frMul(info.knownB, info.targetCoeff))
                invertMap[i] = toInvert.count - 1
            case 2:
                toInvert.append(frMul(info.knownA, info.targetCoeff))
                invertMap[i] = toInvert.count - 1
            default:
                break
            }
        }

        let inverses = batchInverse(toInvert)

        // Phase 3: Solve
        for i in 0..<count {
            let info = infos[i]
            let invIdx = invertMap[i]
            guard invIdx >= 0 else { continue }

            switch info.mode {
            case 0:
                let product = frMul(info.knownA, info.knownB)
                let diff = frSub(product, info.knownC)
                assignment[info.targetVar] = frMul(diff, inverses[invIdx])
            case 1:
                let cMinusAB = frSub(info.knownC, frMul(info.knownA, info.knownB))
                assignment[info.targetVar] = frMul(cMinusAB, inverses[invIdx])
            case 2:
                let cMinusAB = frSub(info.knownC, frMul(info.knownA, info.knownB))
                assignment[info.targetVar] = frMul(cMinusAB, inverses[invIdx])
            default:
                break
            }
        }
    }

    /// Solve a wave of constraints on GPU using pre-computed batch inverses.
    private func solveWaveGPU(wave: [Int],
                              constraints: [R1CSConstraint],
                              produced: [Int?],
                              assignment: inout [Fr]) {
        let count = wave.count

        // Phase 1: Evaluate known parts in parallel on CPU using GCD
        var infos = [R1CSSolveInfo](repeating: R1CSSolveInfo(
            knownA: Fr.zero, knownB: Fr.zero, knownC: Fr.zero,
            targetCoeff: Fr.zero, precomputedInv: Fr.zero,
            targetVar: 0, solveMode: 3
        ), count: count)

        DispatchQueue.concurrentPerform(iterations: count) { i in
            let ci = wave[i]
            guard let targetVar = produced[ci] else { return }
            let c = constraints[ci]
            let aHas = c.a.variables.contains(targetVar)
            let bHas = c.b.variables.contains(targetVar)
            let cHas = c.c.variables.contains(targetVar)

            infos[i].targetVar = UInt32(targetVar)

            if cHas && !aHas && !bHas {
                infos[i].knownA = c.a.evaluate(assignment: assignment)
                infos[i].knownB = c.b.evaluate(assignment: assignment)
                let (knownSum, coeff) = self.splitLC(lc: c.c, targetVar: targetVar, assignment: assignment)
                infos[i].knownC = knownSum
                infos[i].targetCoeff = coeff
                infos[i].solveMode = 0
            } else if aHas && !bHas && !cHas {
                infos[i].knownB = c.b.evaluate(assignment: assignment)
                infos[i].knownC = c.c.evaluate(assignment: assignment)
                let (knownSum, coeff) = self.splitLC(lc: c.a, targetVar: targetVar, assignment: assignment)
                infos[i].knownA = knownSum
                infos[i].targetCoeff = coeff
                infos[i].solveMode = 1
            } else if bHas && !aHas && !cHas {
                infos[i].knownA = c.a.evaluate(assignment: assignment)
                infos[i].knownC = c.c.evaluate(assignment: assignment)
                let (knownSum, coeff) = self.splitLC(lc: c.b, targetVar: targetVar, assignment: assignment)
                infos[i].knownB = knownSum
                infos[i].targetCoeff = coeff
                infos[i].solveMode = 2
            }
        }

        // Phase 2: Batch inversion of denominators on CPU
        var toInvert = [Fr]()
        var invertMap = [Int](repeating: -1, count: count)

        for i in 0..<count {
            let info = infos[i]
            switch info.solveMode {
            case 0:
                toInvert.append(info.targetCoeff)
                invertMap[i] = toInvert.count - 1
            case 1:
                toInvert.append(frMul(info.knownB, info.targetCoeff))
                invertMap[i] = toInvert.count - 1
            case 2:
                toInvert.append(frMul(info.knownA, info.targetCoeff))
                invertMap[i] = toInvert.count - 1
            default:
                break
            }
        }

        let inverses = batchInverse(toInvert)

        // Store pre-computed inverses into solve infos
        for i in 0..<count {
            let invIdx = invertMap[i]
            if invIdx >= 0 {
                infos[i].precomputedInv = inverses[invIdx]
            }
        }

        // Phase 3: GPU dispatch
        let infoStride = MemoryLayout<R1CSSolveInfo>.stride
        let frStride = MemoryLayout<Fr>.stride

        // Create witness buffer from current assignment
        guard let witnessBuf = device.makeBuffer(bytes: assignment, length: assignment.count * frStride, options: .storageModeShared),
              let infoBuf = device.makeBuffer(bytes: infos, length: count * infoStride, options: .storageModeShared),
              let cmdBuf = commandQueue.makeCommandBuffer() else {
            // GPU failed, fall back to CPU
            solveWaveCPU(wave: wave, constraints: constraints, produced: produced, assignment: &assignment)
            return
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(solvePipeline)
        enc.setBuffer(witnessBuf, offset: 0, index: 0)
        enc.setBuffer(infoBuf, offset: 0, index: 1)
        var numConstraints = UInt32(count)
        enc.setBytes(&numConstraints, length: 4, index: 2)

        let tg = min(256, Int(solvePipeline.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Read back solved values from GPU witness buffer
        let ptr = witnessBuf.contents().bindMemory(to: Fr.self, capacity: assignment.count)
        for i in 0..<count {
            let info = infos[i]
            if info.solveMode < 3 {
                let tv = Int(info.targetVar)
                assignment[tv] = ptr[tv]
            }
        }
    }

    // MARK: - Helpers

    /// Split a linear combination into (knownSum, targetCoefficient).
    private func splitLC(lc: LinearCombination, targetVar: Int, assignment: [Fr]) -> (Fr, Fr) {
        var knownSum = Fr.zero
        var coefficient = Fr.zero
        for (idx, coeff) in lc.terms {
            if idx == targetVar {
                coefficient = frAdd(coefficient, coeff)
            } else {
                knownSum = frAdd(knownSum, frMul(coeff, assignment[idx]))
            }
        }
        return (knownSum, coefficient)
    }

    /// Batch Montgomery inversion: 1 field inversion + 3(n-1) multiplications.
    private func batchInverse(_ elements: [Fr]) -> [Fr] {
        let n = elements.count
        if n == 0 { return [] }
        if n == 1 { return [frInverse(elements[0])] }
        var result = [Fr](repeating: Fr.zero, count: n)
        elements.withUnsafeBytes { aBuf in
            result.withUnsafeMutableBytes { rBuf in
                bn254_fr_batch_inverse_safe(
                    aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(n),
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                )
            }
        }
        return result
    }

    /// Convert R1CSInstance sparse entries to R1CSConstraint array.
    private func r1csToConstraints(r1cs: R1CSInstance) -> [R1CSConstraint] {
        var aByRow = [[R1CSEntry]](repeating: [], count: r1cs.numConstraints)
        var bByRow = [[R1CSEntry]](repeating: [], count: r1cs.numConstraints)
        var cByRow = [[R1CSEntry]](repeating: [], count: r1cs.numConstraints)
        for e in r1cs.aEntries { aByRow[e.row].append(e) }
        for e in r1cs.bEntries { bByRow[e.row].append(e) }
        for e in r1cs.cEntries { cByRow[e.row].append(e) }

        var constraints = [R1CSConstraint]()
        constraints.reserveCapacity(r1cs.numConstraints)
        for i in 0..<r1cs.numConstraints {
            let aLC = LinearCombination(aByRow[i].map { ($0.col, $0.val) })
            let bLC = LinearCombination(bByRow[i].map { ($0.col, $0.val) })
            let cLC = LinearCombination(cByRow[i].map { ($0.col, $0.val) })
            constraints.append(R1CSConstraint(a: aLC, b: bLC, c: cLC))
        }
        return constraints
    }
}
