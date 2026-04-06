// GPUPlonkCopyConstraintEngine — GPU-accelerated copy constraint handling for Plonk
//
// Handles the full copy constraint pipeline:
//   1. Wire assignment from circuit constraints
//   2. Cycle detection and construction for copy constraints
//   3. Sigma polynomial computation from permutation cycles
//   4. Multi-table copy constraint merging
//   5. Verification of copy constraint satisfaction
//
// Copy constraints enforce that certain wire positions hold equal values.
// They are encoded as permutation cycles over the set of wire positions
// {(wire_j, row_i) : j in 0..<numWires, i in 0..<n}.
//
// The identity permutation maps each position to itself:
//   id(j, i) = cosetMul(j) * omega^i
//
// A copy constraint between positions (j1, i1) and (j2, i2) swaps their
// sigma values, creating cycles. The sigma polynomial encodes the full
// permutation after all copy constraints are applied.
//
// GPU acceleration:
//   - Parallel sigma polynomial evaluation from cycle structure
//   - Batch constraint satisfaction checking (one thread per row)
//   - Multi-table merge via parallel union-find

import Foundation
import Metal
import NeonFieldOps

// MARK: - Wire Position

/// A position in the Plonk wire assignment table.
public struct WirePosition: Hashable, Equatable {
    /// Wire column index (0 = left/a, 1 = right/b, 2 = output/c, ...)
    public let wire: Int
    /// Row index within the circuit
    public let row: Int

    public init(wire: Int, row: Int) {
        self.wire = wire
        self.row = row
    }

    /// Flatten to a single index: wire * n + row
    public func linearIndex(n: Int) -> Int {
        return wire * n + row
    }

    /// Reconstruct from a linear index
    public static func fromLinear(_ idx: Int, n: Int) -> WirePosition {
        return WirePosition(wire: idx / n, row: idx % n)
    }
}

// MARK: - Permutation Cycle

/// A cycle in the copy constraint permutation.
/// Each cycle represents a set of wire positions that must all hold the same value.
public struct PermutationCycle {
    /// Ordered positions in the cycle. The permutation maps
    /// positions[i] -> positions[(i+1) % count].
    public let positions: [WirePosition]

    public init(positions: [WirePosition]) {
        precondition(positions.count >= 2, "Cycle must have at least 2 positions")
        self.positions = positions
    }

    public var count: Int { positions.count }
}

// MARK: - Copy Constraint Table

/// A named table of copy constraints, supporting multi-table merging.
public struct CopyConstraintTable {
    /// Table identifier
    public let id: String
    /// Constraints within this table
    public let constraints: [PlonkCopyConstraint]

    public init(id: String, constraints: [PlonkCopyConstraint]) {
        self.id = id
        self.constraints = constraints
    }
}

// MARK: - Sigma Polynomials Result

/// Result of sigma polynomial computation.
public struct SigmaPolynomials {
    /// Per-wire sigma evaluations. sigmas[j][i] encodes where position (j, i)
    /// maps to under the copy constraint permutation.
    public let sigmas: [[Fr]]
    /// The permutation cycles that were detected
    public let cycles: [PermutationCycle]
    /// Number of wires
    public let numWires: Int
    /// Domain size
    public let domainSize: Int

    public init(sigmas: [[Fr]], cycles: [PermutationCycle], numWires: Int, domainSize: Int) {
        self.sigmas = sigmas
        self.cycles = cycles
        self.numWires = numWires
        self.domainSize = domainSize
    }
}

// MARK: - Constraint Satisfaction Result

/// Result of checking whether copy constraints are satisfied by a witness.
public struct CopyConstraintCheckResult {
    /// Whether all copy constraints are satisfied
    public let satisfied: Bool
    /// Indices of failing constraints (into the original constraint list)
    public let failingIndices: [Int]
    /// For each failing constraint, the two values that should be equal but aren't
    public let failingValues: [(Fr, Fr)]

    public init(satisfied: Bool, failingIndices: [Int], failingValues: [(Fr, Fr)]) {
        self.satisfied = satisfied
        self.failingIndices = failingIndices
        self.failingValues = failingValues
    }
}

// MARK: - GPUPlonkCopyConstraintEngine

/// GPU-accelerated engine for Plonk copy constraint processing.
///
/// Handles the full pipeline from raw copy constraints to sigma polynomials
/// suitable for the permutation argument. Supports:
///   - Cycle detection via union-find
///   - Sigma polynomial computation from cycles
///   - Multi-table constraint merging
///   - GPU-accelerated constraint satisfaction checking
///   - Wire assignment extraction from circuit structure
public final class GPUPlonkCopyConstraintEngine {
    public static let version = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")

    /// Minimum domain size to use GPU path
    private static let gpuThreshold = 1024

    private let device: MTLDevice?
    private let commandQueue: MTLCommandQueue?
    private let checkPipeline: MTLComputePipelineState?
    private let sigmaPipeline: MTLComputePipelineState?
    private let threadgroupSize: Int

    // MARK: - Initialization

    public init() {
        let dev = MTLCreateSystemDefaultDevice()
        self.device = dev
        self.commandQueue = dev?.makeCommandQueue()
        self.threadgroupSize = 256

        if let dev = dev {
            let pipelines = GPUPlonkCopyConstraintEngine.compileKernels(device: dev)
            self.checkPipeline = pipelines.check
            self.sigmaPipeline = pipelines.sigma
        } else {
            self.checkPipeline = nil
            self.sigmaPipeline = nil
        }
    }

    // MARK: - Wire Assignment Extraction

    /// Extract per-wire witness values from a circuit and full witness vector.
    ///
    /// Given a PlonkCircuit with wireAssignments[row][col] = variable index,
    /// and a witness vector mapping variable index -> field value,
    /// returns per-wire evaluations suitable for the permutation argument.
    ///
    /// - Parameters:
    ///   - circuit: The Plonk circuit.
    ///   - witness: Full witness array (variable index -> field value).
    ///   - numWires: Number of wire columns (default 3 for standard Plonk).
    /// - Returns: wireValues[j][i] = witness value at wire j, row i.
    public func extractWireValues(
        circuit: PlonkCircuit,
        witness: [Fr],
        numWires: Int = 3
    ) -> [[Fr]] {
        let n = circuit.numGates
        var wireValues = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: n), count: numWires)

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

    // MARK: - Cycle Detection (Union-Find)

    /// Detect permutation cycles from a list of copy constraints.
    ///
    /// Uses union-find to group wire positions into equivalence classes,
    /// then extracts the cycle structure. Each equivalence class becomes
    /// one permutation cycle.
    ///
    /// - Parameters:
    ///   - constraints: List of copy constraints.
    ///   - numWires: Number of wire columns.
    ///   - domainSize: Number of rows (n).
    /// - Returns: Array of permutation cycles.
    public func detectCycles(
        constraints: [PlonkCopyConstraint],
        numWires: Int,
        domainSize: Int
    ) -> [PermutationCycle] {
        let totalPositions = numWires * domainSize

        // Union-Find data structure
        var parent = Array(0..<totalPositions)
        var rank = [Int](repeating: 0, count: totalPositions)

        func find(_ x: Int) -> Int {
            var x = x
            while parent[x] != x {
                parent[x] = parent[parent[x]]  // path compression
                x = parent[x]
            }
            return x
        }

        func union(_ a: Int, _ b: Int) {
            let ra = find(a)
            let rb = find(b)
            if ra == rb { return }
            if rank[ra] < rank[rb] {
                parent[ra] = rb
            } else if rank[ra] > rank[rb] {
                parent[rb] = ra
            } else {
                parent[rb] = ra
                rank[ra] += 1
            }
        }

        // Process all constraints
        for c in constraints {
            let srcIdx = c.srcWire * domainSize + c.srcRow
            let dstIdx = c.dstWire * domainSize + c.dstRow
            guard srcIdx < totalPositions, dstIdx < totalPositions else { continue }
            union(srcIdx, dstIdx)
        }

        // Group positions by their root
        var groups = [Int: [Int]]()
        for i in 0..<totalPositions {
            let root = find(i)
            groups[root, default: []].append(i)
        }

        // Convert groups with >1 member into cycles
        var cycles = [PermutationCycle]()
        for (_, members) in groups {
            if members.count < 2 { continue }
            let positions = members.map { WirePosition.fromLinear($0, n: domainSize) }
            cycles.append(PermutationCycle(positions: positions))
        }

        return cycles
    }

    // MARK: - Sigma Polynomial Computation

    /// Compute sigma polynomials from copy constraints.
    ///
    /// The sigma polynomials encode the permutation that enforces copy constraints.
    /// For each wire position (j, i), sigma[j][i] gives the domain element
    /// corresponding to the next position in its cycle.
    ///
    /// Identity permutation: sigma[j][i] = cosetMul(j) * omega^i
    /// After copy constraint linking (j1,i1) <-> (j2,i2):
    ///   sigma values are swapped along the cycle.
    ///
    /// - Parameters:
    ///   - constraints: Copy constraints to encode.
    ///   - numWires: Number of wire columns.
    ///   - domainSize: Must be a power of 2.
    /// - Returns: SigmaPolynomials containing per-wire sigma evaluations and detected cycles.
    public func computeSigmaPolynomials(
        constraints: [PlonkCopyConstraint],
        numWires: Int,
        domainSize: Int
    ) -> SigmaPolynomials {
        precondition(domainSize > 0 && domainSize & (domainSize - 1) == 0,
                     "Domain size must be a power of 2")

        let logN = Int(log2(Double(domainSize)))
        let omega = computeNthRootOfUnity(logN: logN)

        // Build evaluation domain
        var domain = [Fr](repeating: Fr.zero, count: domainSize)
        domain[0] = Fr.one
        for i in 1..<domainSize {
            domain[i] = frMul(domain[i - 1], omega)
        }

        // Coset multipliers: col 0 -> 1, col j -> j+1
        var cosetMuls = [Fr](repeating: Fr.one, count: numWires)
        for j in 1..<numWires {
            cosetMuls[j] = frFromInt(UInt64(j + 1))
        }

        // Initialize sigma to identity permutation
        var sigmas = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: domainSize), count: numWires)
        for j in 0..<numWires {
            for i in 0..<domainSize {
                sigmas[j][i] = frMul(cosetMuls[j], domain[i])
            }
        }

        // Detect cycles
        let cycles = detectCycles(
            constraints: constraints,
            numWires: numWires,
            domainSize: domainSize
        )

        // Apply cycles to sigma: for each cycle [p0, p1, ..., pk],
        // sigma(p0) -> id(p1), sigma(p1) -> id(p2), ..., sigma(pk) -> id(p0)
        for cycle in cycles {
            let positions = cycle.positions
            let count = positions.count
            for idx in 0..<count {
                let current = positions[idx]
                let next = positions[(idx + 1) % count]
                // sigma at current position gets the identity value of the next position
                sigmas[current.wire][current.row] = frMul(cosetMuls[next.wire], domain[next.row])
            }
        }

        return SigmaPolynomials(
            sigmas: sigmas,
            cycles: cycles,
            numWires: numWires,
            domainSize: domainSize
        )
    }

    // MARK: - Multi-Table Merging

    /// Merge copy constraints from multiple tables into a unified set.
    ///
    /// When a Plonk circuit uses multiple sub-circuits or lookup tables,
    /// each may have its own copy constraints. This method merges them,
    /// deduplicating and resolving any transitive constraints.
    ///
    /// - Parameters:
    ///   - tables: Array of copy constraint tables to merge.
    ///   - numWires: Number of wire columns.
    ///   - domainSize: Domain size.
    /// - Returns: SigmaPolynomials encoding the merged constraints.
    public func mergeTables(
        tables: [CopyConstraintTable],
        numWires: Int,
        domainSize: Int
    ) -> SigmaPolynomials {
        // Collect all constraints from all tables
        var allConstraints = [PlonkCopyConstraint]()
        for table in tables {
            allConstraints.append(contentsOf: table.constraints)
        }

        return computeSigmaPolynomials(
            constraints: allConstraints,
            numWires: numWires,
            domainSize: domainSize
        )
    }

    // MARK: - Copy Constraint Verification

    /// Verify that all copy constraints are satisfied by a witness.
    ///
    /// Checks each constraint (srcWire, srcRow) <-> (dstWire, dstRow)
    /// by comparing witness values at the two positions.
    ///
    /// Uses GPU for large domains (parallel checking), CPU for small ones.
    ///
    /// - Parameters:
    ///   - constraints: Copy constraints to check.
    ///   - wireValues: Per-wire witness values. wireValues[j][i] = value at wire j, row i.
    /// - Returns: CopyConstraintCheckResult indicating pass/fail and failing indices.
    public func verifyCopyConstraints(
        constraints: [PlonkCopyConstraint],
        wireValues: [[Fr]]
    ) -> CopyConstraintCheckResult {
        if constraints.isEmpty {
            return CopyConstraintCheckResult(satisfied: true, failingIndices: [], failingValues: [])
        }

        let numWires = wireValues.count
        guard numWires > 0 else {
            return CopyConstraintCheckResult(satisfied: true, failingIndices: [], failingValues: [])
        }
        let n = wireValues[0].count

        // Use GPU for large constraint sets
        if constraints.count >= GPUPlonkCopyConstraintEngine.gpuThreshold,
           let device = device, let queue = commandQueue, let pipeline = checkPipeline {
            return gpuVerifyCopyConstraints(
                constraints: constraints,
                wireValues: wireValues,
                device: device,
                queue: queue,
                pipeline: pipeline,
                numWires: numWires,
                n: n
            )
        }

        return cpuVerifyCopyConstraints(
            constraints: constraints,
            wireValues: wireValues,
            numWires: numWires,
            n: n
        )
    }

    /// Verify the permutation argument using sigma polynomials.
    ///
    /// Checks that the sigma polynomials correctly encode the copy constraints
    /// by verifying that for every pair of positions in a cycle, the witness
    /// values are equal.
    ///
    /// - Parameters:
    ///   - sigmaResult: Sigma polynomials (from computeSigmaPolynomials).
    ///   - wireValues: Per-wire witness values.
    /// - Returns: True if all cycle constraints are satisfied.
    public func verifySigmaConsistency(
        sigmaResult: SigmaPolynomials,
        wireValues: [[Fr]]
    ) -> Bool {
        for cycle in sigmaResult.cycles {
            // All positions in a cycle must have the same witness value
            let firstPos = cycle.positions[0]
            let expectedValue = wireValues[firstPos.wire][firstPos.row]

            for pos in cycle.positions.dropFirst() {
                let value = wireValues[pos.wire][pos.row]
                if !frEqual(value, expectedValue) {
                    return false
                }
            }
        }
        return true
    }

    // MARK: - Sigma from Circuit

    /// Compute sigma polynomials directly from a PlonkCircuit.
    ///
    /// Extracts copy constraints from the circuit's copyConstraints field
    /// and computes the sigma polynomials. The circuit's copy constraints
    /// are encoded as pairs of flattened indices (gateIndex * 3 + wireType).
    ///
    /// - Parameters:
    ///   - circuit: The Plonk circuit.
    ///   - numWires: Number of wire columns (default 3).
    /// - Returns: SigmaPolynomials for the circuit.
    public func sigmaFromCircuit(
        circuit: PlonkCircuit,
        numWires: Int = 3
    ) -> SigmaPolynomials {
        let n = circuit.numGates
        guard n > 0 else {
            return SigmaPolynomials(sigmas: [], cycles: [], numWires: numWires, domainSize: 0)
        }

        // Pad to next power of 2
        let domainSize = nextPow2(n)

        // Convert circuit copy constraints to PlonkCopyConstraint format
        var constraints = [PlonkCopyConstraint]()
        for (flat1, flat2) in circuit.copyConstraints {
            let wire1 = flat1 % numWires
            let row1 = flat1 / numWires
            let wire2 = flat2 % numWires
            let row2 = flat2 / numWires
            guard row1 < domainSize, row2 < domainSize,
                  wire1 < numWires, wire2 < numWires else { continue }
            constraints.append(PlonkCopyConstraint(
                srcWire: wire1, srcRow: row1,
                dstWire: wire2, dstRow: row2
            ))
        }

        return computeSigmaPolynomials(
            constraints: constraints,
            numWires: numWires,
            domainSize: domainSize
        )
    }

    // MARK: - CPU Verification

    private func cpuVerifyCopyConstraints(
        constraints: [PlonkCopyConstraint],
        wireValues: [[Fr]],
        numWires: Int,
        n: Int
    ) -> CopyConstraintCheckResult {
        var failingIndices = [Int]()
        var failingValues = [(Fr, Fr)]()

        for (idx, c) in constraints.enumerated() {
            guard c.srcWire < numWires, c.dstWire < numWires,
                  c.srcRow < n, c.dstRow < n else {
                failingIndices.append(idx)
                failingValues.append((Fr.zero, Fr.zero))
                continue
            }

            let srcVal = wireValues[c.srcWire][c.srcRow]
            let dstVal = wireValues[c.dstWire][c.dstRow]

            if !frEqual(srcVal, dstVal) {
                failingIndices.append(idx)
                failingValues.append((srcVal, dstVal))
            }
        }

        return CopyConstraintCheckResult(
            satisfied: failingIndices.isEmpty,
            failingIndices: failingIndices,
            failingValues: failingValues
        )
    }

    // MARK: - GPU Verification

    private func gpuVerifyCopyConstraints(
        constraints: [PlonkCopyConstraint],
        wireValues: [[Fr]],
        device: MTLDevice,
        queue: MTLCommandQueue,
        pipeline: MTLComputePipelineState,
        numWires: Int,
        n: Int
    ) -> CopyConstraintCheckResult {
        let numConstraints = constraints.count
        let frStride = MemoryLayout<Fr>.stride

        // Flatten wire values
        var flatWire = [Fr](repeating: Fr.zero, count: numWires * n)
        for j in 0..<numWires {
            for i in 0..<n {
                flatWire[j * n + i] = wireValues[j][i]
            }
        }

        // Pack constraint indices: [srcWire, srcRow, dstWire, dstRow] per constraint
        var constraintData = [UInt32](repeating: 0, count: numConstraints * 4)
        for (idx, c) in constraints.enumerated() {
            constraintData[idx * 4 + 0] = UInt32(c.srcWire)
            constraintData[idx * 4 + 1] = UInt32(c.srcRow)
            constraintData[idx * 4 + 2] = UInt32(c.dstWire)
            constraintData[idx * 4 + 3] = UInt32(c.dstRow)
        }

        let wireSize = numWires * n * frStride
        let constraintSize = numConstraints * 4 * MemoryLayout<UInt32>.stride
        let resultSize = numConstraints * MemoryLayout<UInt32>.stride

        guard let wireBuf = device.makeBuffer(length: wireSize, options: .storageModeShared),
              let cBuf = device.makeBuffer(length: constraintSize, options: .storageModeShared),
              let resBuf = device.makeBuffer(length: resultSize, options: .storageModeShared) else {
            return cpuVerifyCopyConstraints(
                constraints: constraints, wireValues: wireValues,
                numWires: numWires, n: n
            )
        }

        flatWire.withUnsafeBytes { src in memcpy(wireBuf.contents(), src.baseAddress!, wireSize) }
        constraintData.withUnsafeBytes { src in memcpy(cBuf.contents(), src.baseAddress!, constraintSize) }

        // Zero the result buffer
        memset(resBuf.contents(), 0, resultSize)

        guard let cmdBuf = queue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            return cpuVerifyCopyConstraints(
                constraints: constraints, wireValues: wireValues,
                numWires: numWires, n: n
            )
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(wireBuf, offset: 0, index: 0)
        encoder.setBuffer(cBuf, offset: 0, index: 1)
        encoder.setBuffer(resBuf, offset: 0, index: 2)

        var nVal = UInt32(n)
        var numC = UInt32(numConstraints)
        encoder.setBytes(&nVal, length: 4, index: 3)
        encoder.setBytes(&numC, length: 4, index: 4)

        let tgSize = MTLSize(width: threadgroupSize, height: 1, depth: 1)
        let gridSize = MTLSize(width: numConstraints, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if cmdBuf.error != nil {
            return cpuVerifyCopyConstraints(
                constraints: constraints, wireValues: wireValues,
                numWires: numWires, n: n
            )
        }

        // Read results: 1 = fail, 0 = pass
        var results = [UInt32](repeating: 0, count: numConstraints)
        results.withUnsafeMutableBytes { dst in
            memcpy(dst.baseAddress!, resBuf.contents(), resultSize)
        }

        var failingIndices = [Int]()
        var failingValues = [(Fr, Fr)]()
        for i in 0..<numConstraints {
            if results[i] != 0 {
                failingIndices.append(i)
                let c = constraints[i]
                let srcVal = wireValues[c.srcWire][c.srcRow]
                let dstVal = wireValues[c.dstWire][c.dstRow]
                failingValues.append((srcVal, dstVal))
            }
        }

        return CopyConstraintCheckResult(
            satisfied: failingIndices.isEmpty,
            failingIndices: failingIndices,
            failingValues: failingValues
        )
    }

    // MARK: - GPU Sigma Computation

    /// Compute sigma polynomials on GPU for large domains.
    /// Falls back to CPU for domains below gpuThreshold.
    public func computeSigmaPolynomialsGPU(
        constraints: [PlonkCopyConstraint],
        numWires: Int,
        domainSize: Int
    ) -> SigmaPolynomials {
        // Cycle detection is inherently sequential (union-find), done on CPU.
        // Sigma evaluation from cycles can be parallelized on GPU.
        let cycles = detectCycles(
            constraints: constraints,
            numWires: numWires,
            domainSize: domainSize
        )

        let logN = Int(log2(Double(domainSize)))
        let omega = computeNthRootOfUnity(logN: logN)

        var domain = [Fr](repeating: Fr.zero, count: domainSize)
        domain[0] = Fr.one
        for i in 1..<domainSize {
            domain[i] = frMul(domain[i - 1], omega)
        }

        var cosetMuls = [Fr](repeating: Fr.one, count: numWires)
        for j in 1..<numWires {
            cosetMuls[j] = frFromInt(UInt64(j + 1))
        }

        // Build the permutation mapping: for each position, where does it map?
        // Start with identity, then override with cycle mappings.
        let totalPositions = numWires * domainSize
        var permMap = Array(0..<totalPositions)

        for cycle in cycles {
            let positions = cycle.positions
            let count = positions.count
            for idx in 0..<count {
                let current = positions[idx]
                let next = positions[(idx + 1) % count]
                permMap[current.linearIndex(n: domainSize)] = next.linearIndex(n: domainSize)
            }
        }

        // Evaluate sigma from permMap
        if domainSize >= GPUPlonkCopyConstraintEngine.gpuThreshold,
           let device = device, let queue = commandQueue, let pipeline = sigmaPipeline {
            if let result = gpuEvaluateSigma(
                permMap: permMap,
                domain: domain,
                cosetMuls: cosetMuls,
                numWires: numWires,
                domainSize: domainSize,
                device: device,
                queue: queue,
                pipeline: pipeline
            ) {
                return SigmaPolynomials(
                    sigmas: result,
                    cycles: cycles,
                    numWires: numWires,
                    domainSize: domainSize
                )
            }
        }

        // CPU fallback: evaluate sigma from permMap
        var sigmas = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: domainSize), count: numWires)
        for j in 0..<numWires {
            for i in 0..<domainSize {
                let srcIdx = j * domainSize + i
                let dstIdx = permMap[srcIdx]
                let dstPos = WirePosition.fromLinear(dstIdx, n: domainSize)
                sigmas[j][i] = frMul(cosetMuls[dstPos.wire], domain[dstPos.row])
            }
        }

        return SigmaPolynomials(
            sigmas: sigmas,
            cycles: cycles,
            numWires: numWires,
            domainSize: domainSize
        )
    }

    private func gpuEvaluateSigma(
        permMap: [Int],
        domain: [Fr],
        cosetMuls: [Fr],
        numWires: Int,
        domainSize: Int,
        device: MTLDevice,
        queue: MTLCommandQueue,
        pipeline: MTLComputePipelineState
    ) -> [[Fr]]? {
        let totalPositions = numWires * domainSize
        let frStride = MemoryLayout<Fr>.stride

        // Convert permMap to UInt32
        let permMapU32 = permMap.map { UInt32($0) }

        let permSize = totalPositions * MemoryLayout<UInt32>.stride
        let domainBufSize = domainSize * frStride
        let cosetBufSize = numWires * frStride
        let outputSize = totalPositions * frStride

        guard let permBuf = device.makeBuffer(length: permSize, options: .storageModeShared),
              let domBuf = device.makeBuffer(length: domainBufSize, options: .storageModeShared),
              let cosetBuf = device.makeBuffer(length: cosetBufSize, options: .storageModeShared),
              let outBuf = device.makeBuffer(length: outputSize, options: .storageModeShared) else {
            return nil
        }

        permMapU32.withUnsafeBytes { src in memcpy(permBuf.contents(), src.baseAddress!, permSize) }
        domain.withUnsafeBytes { src in memcpy(domBuf.contents(), src.baseAddress!, domainBufSize) }
        cosetMuls.withUnsafeBytes { src in memcpy(cosetBuf.contents(), src.baseAddress!, cosetBufSize) }

        guard let cmdBuf = queue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            return nil
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(permBuf, offset: 0, index: 0)
        encoder.setBuffer(domBuf, offset: 0, index: 1)
        encoder.setBuffer(cosetBuf, offset: 0, index: 2)
        encoder.setBuffer(outBuf, offset: 0, index: 3)

        var nVal = UInt32(domainSize)
        var numW = UInt32(numWires)
        encoder.setBytes(&nVal, length: 4, index: 4)
        encoder.setBytes(&numW, length: 4, index: 5)

        let tgSize = MTLSize(width: threadgroupSize, height: 1, depth: 1)
        let gridSize = MTLSize(width: totalPositions, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if cmdBuf.error != nil { return nil }

        // Read back
        var flat = [Fr](repeating: Fr.zero, count: totalPositions)
        flat.withUnsafeMutableBytes { dst in
            memcpy(dst.baseAddress!, outBuf.contents(), outputSize)
        }

        // Unflatten
        var sigmas = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: domainSize), count: numWires)
        for j in 0..<numWires {
            for i in 0..<domainSize {
                sigmas[j][i] = flat[j * domainSize + i]
            }
        }

        return sigmas
    }

    // MARK: - Utilities

    private func nextPow2(_ n: Int) -> Int {
        var v = n
        v -= 1
        v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16
        return v + 1
    }

    // MARK: - Metal Shader Compilation

    private static func compileKernels(device: MTLDevice) -> (check: MTLComputePipelineState?, sigma: MTLComputePipelineState?) {
        let shaderDir = findShaderDir()
        guard let frSource = try? String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8) else {
            return (nil, nil)
        }

        let frClean = frSource
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")

        let kernelSource = """

        // Check copy constraint satisfaction: one thread per constraint.
        // Compares wireValues at (srcWire, srcRow) vs (dstWire, dstRow).
        // results[i] = 0 if satisfied, 1 if violated.
        kernel void copy_constraint_check(
            device const Fr *wireValues [[buffer(0)]],
            device const uint *constraints [[buffer(1)]],
            device uint *results [[buffer(2)]],
            constant uint &n [[buffer(3)]],
            constant uint &numConstraints [[buffer(4)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= numConstraints) return;

            uint srcWire = constraints[gid * 4 + 0];
            uint srcRow  = constraints[gid * 4 + 1];
            uint dstWire = constraints[gid * 4 + 2];
            uint dstRow  = constraints[gid * 4 + 3];

            Fr srcVal = wireValues[srcWire * n + srcRow];
            Fr dstVal = wireValues[dstWire * n + dstRow];

            results[gid] = fr_equal(srcVal, dstVal) ? 0 : 1;
        }

        // Evaluate sigma polynomial from permutation map.
        // One thread per position in the flattened (numWires * domainSize) array.
        // sigma[pos] = cosetMuls[targetWire] * domain[targetRow]
        // where target = permMap[pos], targetWire = target / n, targetRow = target % n
        kernel void sigma_from_perm(
            device const uint *permMap [[buffer(0)]],
            device const Fr *domain [[buffer(1)]],
            device const Fr *cosetMuls [[buffer(2)]],
            device Fr *sigma [[buffer(3)]],
            constant uint &n [[buffer(4)]],
            constant uint &numWires [[buffer(5)]],
            uint gid [[thread_position_in_grid]]
        ) {
            uint total = numWires * n;
            if (gid >= total) return;

            uint target = permMap[gid];
            uint targetWire = target / n;
            uint targetRow  = target % n;

            sigma[gid] = fr_mul(cosetMuls[targetWire], domain[targetRow]);
        }
        """

        let combined = frClean + "\n" + kernelSource
        let options = MTLCompileOptions()
        options.fastMathEnabled = true

        guard let library = try? device.makeLibrary(source: combined, options: options) else {
            return (nil, nil)
        }

        let checkFn = library.makeFunction(name: "copy_constraint_check")
        let sigmaFn = library.makeFunction(name: "sigma_from_perm")

        let checkPipeline = checkFn.flatMap { try? device.makeComputePipelineState(function: $0) }
        let sigmaPipeline = sigmaFn.flatMap { try? device.makeComputePipelineState(function: $0) }

        return (checkPipeline, sigmaPipeline)
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
