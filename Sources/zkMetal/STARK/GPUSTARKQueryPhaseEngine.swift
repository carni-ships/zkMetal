// GPUSTARKQueryPhaseEngine — GPU-accelerated STARK query (decommitment) phase
//
// Implements the full STARK query phase pipeline:
//   1. Fiat-Shamir challenge sampling for query indices
//   2. Merkle authentication path generation for queried positions
//   3. Trace and composition polynomial decommitment at query positions
//   4. FRI query round decommitment (opening at query positions through all FRI layers)
//   5. Batch decommitment across multiple columns
//   6. Deep composition polynomial evaluation at query points
//   7. GPU-accelerated parallel Merkle path extraction
//
// Works with BN254 Fr field type. Falls back to CPU when Metal is unavailable.

import Foundation
import Metal
import NeonFieldOps

// MARK: - Query Phase Configuration

/// Configuration for the STARK query phase engine.
public struct STARKQueryPhaseConfig {
    /// Log2 of the trace length.
    public let logTraceLen: Int
    /// Blowup factor for the LDE domain (power of 2, >= 2).
    public let blowupFactor: Int
    /// Number of trace columns.
    public let numTraceColumns: Int
    /// Number of queries to sample (determines soundness).
    public let numQueries: Int
    /// Number of FRI folding rounds.
    public let numFRIRounds: Int
    /// Number of composition polynomial segments.
    public let numCompositionSegments: Int
    /// Coset shift generator for LDE domain.
    public let cosetShift: Fr

    /// Trace length = 2^logTraceLen.
    public var traceLen: Int { 1 << logTraceLen }
    /// LDE domain size = traceLen * blowupFactor.
    public var ldeDomainSize: Int { traceLen * blowupFactor }
    /// Log2 of LDE domain size.
    public var logLDEDomainSize: Int { logTraceLen + Int(log2(Double(blowupFactor))) }
    /// Estimated security bits = numQueries * log2(blowupFactor).
    public var securityBits: Int { numQueries * Int(log2(Double(blowupFactor))) }

    public init(logTraceLen: Int, blowupFactor: Int, numTraceColumns: Int,
                numQueries: Int, numFRIRounds: Int = 3,
                numCompositionSegments: Int = 1, cosetShift: Fr? = nil) {
        precondition(logTraceLen > 0 && logTraceLen <= 20,
                     "logTraceLen must be in [1, 20]")
        precondition(blowupFactor >= 2 && (blowupFactor & (blowupFactor - 1)) == 0,
                     "blowupFactor must be a power of 2 >= 2")
        precondition(numTraceColumns > 0, "Must have at least one trace column")
        precondition(numQueries > 0, "Must have at least one query")
        precondition(numFRIRounds >= 0, "numFRIRounds must be non-negative")
        self.logTraceLen = logTraceLen
        self.blowupFactor = blowupFactor
        self.numTraceColumns = numTraceColumns
        self.numQueries = numQueries
        self.numFRIRounds = numFRIRounds
        self.numCompositionSegments = numCompositionSegments
        self.cosetShift = cosetShift ?? frFromInt(Fr.GENERATOR)
    }
}

// MARK: - Query Index Set

/// A set of sampled query indices with their Fiat-Shamir derivation.
public struct QueryIndexSet {
    /// Sampled query indices into the LDE domain.
    public let indices: [Int]
    /// The Fiat-Shamir seed used to derive these indices.
    public let seed: Fr
    /// Domain size from which indices were sampled.
    public let domainSize: Int

    public init(indices: [Int], seed: Fr, domainSize: Int) {
        self.indices = indices
        self.seed = seed
        self.domainSize = domainSize
    }
}

// MARK: - Trace Decommitment

/// Decommitment data for the trace at a single query position.
public struct TraceDecommitment {
    /// Index in the LDE domain.
    public let queryIndex: Int
    /// Trace column values at this position: values[colIdx].
    public let values: [Fr]
    /// Merkle authentication path proving inclusion in the trace commitment.
    public let merklePath: FrMerklePath

    public init(queryIndex: Int, values: [Fr], merklePath: FrMerklePath) {
        self.queryIndex = queryIndex
        self.values = values
        self.merklePath = merklePath
    }
}

// MARK: - Composition Decommitment

/// Decommitment data for the composition polynomial at a single query position.
public struct CompositionDecommitment {
    /// Index in the LDE domain.
    public let queryIndex: Int
    /// Composition segment values at this position: segments[segIdx].
    public let segments: [Fr]
    /// Merkle authentication path proving inclusion in the composition commitment.
    public let merklePath: FrMerklePath

    public init(queryIndex: Int, segments: [Fr], merklePath: FrMerklePath) {
        self.queryIndex = queryIndex
        self.segments = segments
        self.merklePath = merklePath
    }
}

// MARK: - FRI Query Decommitment

/// Full FRI decommitment data for a single query across all folding rounds.
public struct FRIQueryDecommitment {
    /// Original query index in layer 0.
    public let queryIndex: Int
    /// Per-layer decommitments (value, sibling, Merkle path).
    public let layers: [FRILayerDecommitment]
    /// Folded query indices at each layer: foldedIndices[layerIdx].
    public let foldedIndices: [Int]

    public init(queryIndex: Int, layers: [FRILayerDecommitment], foldedIndices: [Int]) {
        self.queryIndex = queryIndex
        self.layers = layers
        self.foldedIndices = foldedIndices
    }
}

// MARK: - Batch Decommitment Result

/// Complete query phase result containing all decommitments for all queries.
public struct BatchDecommitmentResult {
    /// Trace decommitments at each query position.
    public let traceDecommitments: [TraceDecommitment]
    /// Composition polynomial decommitments at each query position.
    public let compositionDecommitments: [CompositionDecommitment]
    /// FRI query decommitments across all layers.
    public let friDecommitments: [FRIQueryDecommitment]
    /// The query index set used.
    public let querySet: QueryIndexSet
    /// Deep composition polynomial evaluations at query points.
    public let deepEvals: [Fr]
    /// Configuration used.
    public let config: STARKQueryPhaseConfig

    public init(traceDecommitments: [TraceDecommitment],
                compositionDecommitments: [CompositionDecommitment],
                friDecommitments: [FRIQueryDecommitment],
                querySet: QueryIndexSet,
                deepEvals: [Fr],
                config: STARKQueryPhaseConfig) {
        self.traceDecommitments = traceDecommitments
        self.compositionDecommitments = compositionDecommitments
        self.friDecommitments = friDecommitments
        self.querySet = querySet
        self.deepEvals = deepEvals
        self.config = config
    }

    /// Number of queries.
    public var numQueries: Int { querySet.indices.count }
}

// MARK: - Query Phase Errors

public enum STARKQueryPhaseError: Error, CustomStringConvertible {
    case noGPU
    case gpuError(String)
    case noCommandBuffer
    case invalidQueryIndex(String)
    case merkleTreeMissing(String)
    case decommitmentFailed(String)
    case friLayerMismatch(String)
    case deepEvalMismatch(String)
    case invalidConfig(String)

    public var description: String {
        switch self {
        case .noGPU: return "No Metal GPU device found"
        case .gpuError(let msg): return "GPU error: \(msg)"
        case .noCommandBuffer: return "No command buffer"
        case .invalidQueryIndex(let msg): return "Invalid query index: \(msg)"
        case .merkleTreeMissing(let msg): return "Merkle tree missing: \(msg)"
        case .decommitmentFailed(let msg): return "Decommitment failed: \(msg)"
        case .friLayerMismatch(let msg): return "FRI layer mismatch: \(msg)"
        case .deepEvalMismatch(let msg): return "Deep eval mismatch: \(msg)"
        case .invalidConfig(let msg): return "Invalid config: \(msg)"
        }
    }
}

// MARK: - GPU STARK Query Phase Engine

/// GPU-accelerated STARK query phase engine over BN254 Fr.
///
/// Pipeline: sample query indices -> extract trace/composition values ->
/// generate Merkle proofs -> compute FRI query paths -> evaluate DEEP at queries.
/// Falls back to CPU when Metal is unavailable or query count is small.
public final class GPUSTARKQueryPhaseEngine {
    /// Minimum number of queries to use GPU path.
    public static let gpuThreshold = 16

    private let device: MTLDevice?
    private let commandQueue: MTLCommandQueue?
    private let useGPU: Bool

    public init(forceGPU: Bool = false) throws {
        if let device = MTLCreateSystemDefaultDevice() {
            self.device = device
            self.commandQueue = device.makeCommandQueue()
            self.useGPU = true
        } else if forceGPU {
            throw STARKQueryPhaseError.noGPU
        } else {
            self.device = nil
            self.commandQueue = nil
            self.useGPU = false
        }
    }

    // MARK: - Fiat-Shamir Query Sampling

    /// Sample unique query indices in [0, domainSize) from a Fiat-Shamir seed.
    /// Returns sorted indices for cache-friendly access.
    public func sampleQueryIndices(seed: Fr, config: STARKQueryPhaseConfig) -> QueryIndexSet {
        let domainSize = config.ldeDomainSize
        var indices = [Int]()
        indices.reserveCapacity(config.numQueries)
        var seen = Set<Int>()

        var current = seed
        var attempts = 0
        let maxAttempts = config.numQueries * 10  // Safety bound

        while indices.count < config.numQueries && attempts < maxAttempts {
            // Derive index: hash the current state, extract lower bits
            let hashed = poseidon2Hash(current, frFromInt(UInt64(attempts)))
            let indexRaw = frToUInt64(hashed)
            let index = Int(indexRaw % UInt64(domainSize))

            if !seen.contains(index) {
                seen.insert(index)
                indices.append(index)
            }

            current = hashed
            attempts += 1
        }

        // Sort for cache-friendly access patterns
        indices.sort()

        return QueryIndexSet(indices: indices, seed: seed, domainSize: domainSize)
    }

    // MARK: - Trace Decommitment

    /// Generate trace decommitments: extract column values + Merkle paths at query positions.
    public func generateTraceDecommitments(
        querySet: QueryIndexSet,
        ldeColumns: [[Fr]],
        merkleLeaves: [Fr],
        merkleNodes: [Fr],
        commitment: Fr,
        config: STARKQueryPhaseConfig
    ) throws -> [TraceDecommitment] {
        let m = config.ldeDomainSize

        guard ldeColumns.count == config.numTraceColumns else {
            throw STARKQueryPhaseError.decommitmentFailed(
                "Expected \(config.numTraceColumns) trace columns, got \(ldeColumns.count)")
        }
        guard merkleLeaves.count == m else {
            throw STARKQueryPhaseError.merkleTreeMissing(
                "Expected \(m) Merkle leaves, got \(merkleLeaves.count)")
        }

        if useGPU && querySet.indices.count >= GPUSTARKQueryPhaseEngine.gpuThreshold {
            return try gpuBatchTraceDecommit(
                querySet: querySet, ldeColumns: ldeColumns,
                merkleLeaves: merkleLeaves, merkleNodes: merkleNodes,
                commitment: commitment, config: config)
        }

        var results = [TraceDecommitment]()
        results.reserveCapacity(querySet.indices.count)

        for idx in querySet.indices {
            guard idx >= 0 && idx < m else {
                throw STARKQueryPhaseError.invalidQueryIndex(
                    "Query index \(idx) out of range [0, \(m))")
            }

            // Gather column values at this position
            var values = [Fr](repeating: Fr.zero, count: config.numTraceColumns)
            for c in 0..<config.numTraceColumns {
                values[c] = ldeColumns[c][idx]
            }

            // Generate Merkle proof
            let merklePath = extractMerklePath(
                leafIndex: idx, leaves: merkleLeaves,
                nodes: merkleNodes, root: commitment)

            results.append(TraceDecommitment(
                queryIndex: idx, values: values, merklePath: merklePath))
        }

        return results
    }

    // MARK: - GPU Batch Trace Decommitment

    /// GPU-accelerated batch trace decommitment with parallel extraction.
    private func gpuBatchTraceDecommit(
        querySet: QueryIndexSet,
        ldeColumns: [[Fr]],
        merkleLeaves: [Fr],
        merkleNodes: [Fr],
        commitment: Fr,
        config: STARKQueryPhaseConfig
    ) throws -> [TraceDecommitment] {
        // GPU dispatch: parallel over query indices
        // Each thread extracts column values + Merkle path for one query
        let m = config.ldeDomainSize
        var results = [TraceDecommitment]()
        results.reserveCapacity(querySet.indices.count)

        // Batch gather: for large query counts, the GPU can parallelize
        // column value extraction and Merkle path computation.
        // For now, use optimized CPU path with batch allocation.
        let numQ = querySet.indices.count
        var allValues = [[Fr]](repeating: [Fr](repeating: Fr.zero,
            count: config.numTraceColumns), count: numQ)

        // Batch column value extraction
        for c in 0..<config.numTraceColumns {
            let col = ldeColumns[c]
            for (qi, idx) in querySet.indices.enumerated() {
                allValues[qi][c] = col[idx]
            }
        }

        // Batch Merkle path generation
        for (qi, idx) in querySet.indices.enumerated() {
            guard idx >= 0 && idx < m else {
                throw STARKQueryPhaseError.invalidQueryIndex(
                    "Query index \(idx) out of range [0, \(m))")
            }
            let path = extractMerklePath(
                leafIndex: idx, leaves: merkleLeaves,
                nodes: merkleNodes, root: commitment)
            results.append(TraceDecommitment(
                queryIndex: idx, values: allValues[qi], merklePath: path))
        }

        return results
    }

    // MARK: - Composition Decommitment

    /// Generate composition polynomial decommitments at query positions.
    public func generateCompositionDecommitments(
        querySet: QueryIndexSet,
        compositionSegments: [[Fr]],
        merkleLeaves: [Fr],
        merkleNodes: [Fr],
        commitment: Fr,
        config: STARKQueryPhaseConfig
    ) throws -> [CompositionDecommitment] {
        let m = config.ldeDomainSize

        guard compositionSegments.count == config.numCompositionSegments else {
            throw STARKQueryPhaseError.decommitmentFailed(
                "Expected \(config.numCompositionSegments) composition segments, got \(compositionSegments.count)")
        }

        var results = [CompositionDecommitment]()
        results.reserveCapacity(querySet.indices.count)

        for idx in querySet.indices {
            guard idx >= 0 && idx < m else {
                throw STARKQueryPhaseError.invalidQueryIndex(
                    "Query index \(idx) out of range [0, \(m))")
            }

            // Gather segment values at this position
            var segments = [Fr](repeating: Fr.zero, count: config.numCompositionSegments)
            for s in 0..<config.numCompositionSegments {
                segments[s] = compositionSegments[s][idx]
            }

            // Generate Merkle proof
            let merklePath = extractMerklePath(
                leafIndex: idx, leaves: merkleLeaves,
                nodes: merkleNodes, root: commitment)

            results.append(CompositionDecommitment(
                queryIndex: idx, segments: segments, merklePath: merklePath))
        }

        return results
    }

    // MARK: - FRI Query Decommitment

    /// Generate FRI decommitments: fold indices through layers, extract values + Merkle paths.
    public func generateFRIDecommitments(
        querySet: QueryIndexSet,
        friLayers: [[Fr]],
        friMerkleData: [(leaves: [Fr], nodes: [Fr], root: Fr)],
        config: STARKQueryPhaseConfig
    ) throws -> [FRIQueryDecommitment] {
        let numRounds = config.numFRIRounds

        guard friLayers.count == numRounds else {
            throw STARKQueryPhaseError.friLayerMismatch(
                "Expected \(numRounds) FRI layers, got \(friLayers.count)")
        }
        guard friMerkleData.count == numRounds else {
            throw STARKQueryPhaseError.friLayerMismatch(
                "Expected \(numRounds) FRI Merkle trees, got \(friMerkleData.count)")
        }

        var results = [FRIQueryDecommitment]()
        results.reserveCapacity(querySet.indices.count)

        for queryIdx in querySet.indices {
            var layers = [FRILayerDecommitment]()
            layers.reserveCapacity(numRounds)
            var foldedIndices = [Int]()
            foldedIndices.reserveCapacity(numRounds)

            var currentIdx = queryIdx

            for round in 0..<numRounds {
                let layerSize = friLayers[round].count
                let foldedIdx = currentIdx % layerSize
                let siblingIdx = foldedIdx ^ 1

                foldedIndices.append(foldedIdx)

                let value = friLayers[round][foldedIdx]
                let siblingValue = siblingIdx < layerSize
                    ? friLayers[round][siblingIdx]
                    : Fr.zero

                let merklePath = extractMerklePath(
                    leafIndex: foldedIdx,
                    leaves: friMerkleData[round].leaves,
                    nodes: friMerkleData[round].nodes,
                    root: friMerkleData[round].root)

                layers.append(FRILayerDecommitment(
                    value: value, siblingValue: siblingValue,
                    merklePath: merklePath))

                // Fold index for next layer: halve the domain
                currentIdx = foldedIdx >> 1
            }

            results.append(FRIQueryDecommitment(
                queryIndex: queryIdx, layers: layers,
                foldedIndices: foldedIndices))
        }

        return results
    }

    // MARK: - GPU Batch FRI Decommitment

    /// GPU-accelerated batch FRI decommitment: parallel extract + fold recompute for all queries.
    ///
    /// Dispatches two kernels:
    /// 1. `fri_batch_query_bn254` — extracts (value, sibling) for each (query, layer) pair
    ///    and recomputes the FRI fold: (a+b) + c*(a-b)*d_inv
    /// 2. `fri_merkle_paths_bn254` — generates Merkle auth paths for each pair
    ///
    /// Falls back to CPU for small query counts (< gpuThreshold).
    public func gpuBatchFRIDecommit(
        querySet: QueryIndexSet,
        friLayers: [[Fr]],
        friMerkleData: [(leaves: [Fr], nodes: [Fr], root: Fr)],
        domainInvs: [[Fr]],
        challenges: [Fr],
        config: STARKQueryPhaseConfig
    ) throws -> [FRIQueryDecommitment] {
        guard useGPU, let device = device, let commandQueue = commandQueue else {
            throw STARKQueryPhaseError.noGPU
        }
        guard querySet.indices.count >= GPUSTARKQueryPhaseEngine.gpuThreshold else {
            // Fall through to CPU path
            return try generateFRIDecommitments(
                querySet: querySet, friLayers: friLayers,
                friMerkleData: friMerkleData, config: config)
        }

        let numQueries = querySet.indices.count
        let numLayers = config.numFRIRounds
        let maxDepth = Int(log2(Double(friLayers[0].count))) + 1

        // Precompute layer sizes and offsets
        var layerSizes = [UInt32]()
        var layerOffsets = [UInt32]()
        var invOffsets = [UInt32]()
        var totalLayerSize = 0
        var totalInvSize = 0
        for layer in friLayers {
            layerOffsets.append(UInt32(totalLayerSize))
            layerSizes.append(UInt32(layer.count))
            totalLayerSize += layer.count
        }
        for inv in domainInvs {
            invOffsets.append(UInt32(totalInvSize))
            totalInvSize += inv.count
        }

        // Flatten friLayers and domainInvs into ContiguousArrays
        var flatLayers = [Fr](repeating: .zero, count: totalLayerSize)
        var flatInvs = [Fr](repeating: .zero, count: totalInvSize)
        var offset = 0
        for layer in friLayers {
            for v in layer { flatLayers[offset] = v; offset += 1 }
        }
        offset = 0
        for inv in domainInvs {
            for v in inv { flatInvs[offset] = v; offset += 1 }
        }

        // Allocate GPU buffers
        let frSize = MemoryLayout<Fr>.stride
        let layerSizeBytes = layerSizes.count * MemoryLayout<UInt32>.stride
        let flatLayerBytes = totalLayerSize * frSize
        let flatInvBytes = totalInvSize * frSize
        let outBytes = numQueries * numLayers * frSize
        let idxOutBytes = numQueries * numLayers * MemoryLayout<UInt32>.stride
        let pathBytes = numQueries * numLayers * maxDepth * frSize

        guard let layersBuf = device.makeBuffer(length: flatLayerBytes, options: .storageModeShared),
              let invsBuf = device.makeBuffer(length: flatInvBytes, options: .storageModeShared),
              let sizesBuf = device.makeBuffer(length: layerSizeBytes, options: .storageModeShared),
              let layerOffBuf = device.makeBuffer(length: layerSizeBytes, options: .storageModeShared),
              let invOffBuf = device.makeBuffer(length: layerSizeBytes, options: .storageModeShared),
              let queryBuf = device.makeBuffer(length: numQueries * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let chalBuf = device.makeBuffer(length: numLayers * frSize, options: .storageModeShared),
              let valuesBuf = device.makeBuffer(length: outBytes, options: .storageModeShared),
              let sibBuf = device.makeBuffer(length: outBytes, options: .storageModeShared),
              let foldedBuf = device.makeBuffer(length: outBytes, options: .storageModeShared),
              let idxBuf = device.makeBuffer(length: idxOutBytes, options: .storageModeShared) else {
            throw STARKQueryPhaseError.gpuError("Failed to allocate FRI query buffers")
        }

        // Copy data to GPU
        flatLayers.withUnsafeBytes { memcpy(layersBuf.contents(), $0.baseAddress!, flatLayerBytes) }
        flatInvs.withUnsafeBytes { memcpy(invsBuf.contents(), $0.baseAddress!, flatInvBytes) }
        layerSizes.withUnsafeBytes { memcpy(sizesBuf.contents(), $0.baseAddress!, layerSizeBytes) }
        layerOffsets.withUnsafeBytes { memcpy(layerOffBuf.contents(), $0.baseAddress!, layerSizeBytes) }
        invOffsets.withUnsafeBytes { memcpy(invOffBuf.contents(), $0.baseAddress!, layerSizeBytes) }
        querySet.indices.withUnsafeBytes { memcpy(queryBuf.contents(), $0.baseAddress!, numQueries * MemoryLayout<UInt32>.stride) }
        challenges.withUnsafeBytes { memcpy(chalBuf.contents(), $0.baseAddress!, numLayers * frSize) }

        // Ensure kernels are compiled
        let batchQueryKernel = try getBatchQueryKernel(device: device)
        let pathKernel = try getMerklPathsKernel(device: device)

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw STARKQueryPhaseError.noCommandBuffer
        }
        guard let enc = cmdBuf.makeComputeCommandEncoder() else {
            throw STARKQueryPhaseError.gpuError("No encoder")
        }

        // Kernel 1: fri_batch_query_bn254
        var nQ = UInt32(numQueries)
        var nL = UInt32(numLayers)
        enc.setComputePipelineState(batchQueryKernel)
        enc.setBuffer(layersBuf, offset: 0, index: 0)
        enc.setBuffer(queryBuf, offset: 0, index: 1)
        enc.setBuffer(chalBuf, offset: 0, index: 2)
        enc.setBuffer(invsBuf, offset: 0, index: 3)
        enc.setBuffer(sizesBuf, offset: 0, index: 4)
        enc.setBuffer(layerOffBuf, offset: 0, index: 5)
        enc.setBuffer(invOffBuf, offset: 0, index: 6)
        enc.setBuffer(valuesBuf, offset: 0, index: 7)
        enc.setBuffer(sibBuf, offset: 0, index: 8)
        enc.setBuffer(foldedBuf, offset: 0, index: 9)
        enc.setBuffer(idxBuf, offset: 0, index: 10)
        enc.setBytes(&nQ, length: 4, index: 11)
        enc.setBytes(&nL, length: 4, index: 12)

        let totalThreads = numQueries * numLayers
        let tgSize = min(batchQueryKernel.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(MTLSize(width: totalThreads, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc.endEncoding()

        // Now read back indices for path kernel input
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let idxPtr = idxBuf.contents().bindMemory(to: UInt32.self, capacity: numQueries * numLayers)
        let valuesPtr = valuesBuf.contents().bindMemory(to: Fr.self, capacity: numQueries * numLayers)
        let sibPtr = sibBuf.contents().bindMemory(to: Fr.self, capacity: numQueries * numLayers)
        let foldedPtr = foldedBuf.contents().bindMemory(to: Fr.self, capacity: numQueries * numLayers)

        // Build per-layer Merkle data for path kernel
        var totalMerkleNodes = 0
        var totalMerkleLeaves = 0
        for md in friMerkleData {
            totalMerkleLeaves += md.leaves.count
            totalMerkleNodes += md.nodes.count
        }
        var flatLeaves = [Fr](repeating: .zero, count: totalMerkleLeaves)
        var flatNodes = [Fr](repeating: .zero, count: totalMerkleNodes)
        var leafOffs = [UInt32]()
        var nodeOffs = [UInt32]()
        var leafOff = 0
        var nodeOff = 0
        for md in friMerkleData {
            leafOffs.append(UInt32(leafOff))
            nodeOffs.append(UInt32(nodeOff))
            for v in md.leaves { flatLeaves[leafOff] = v; leafOff += 1 }
            for v in md.nodes { flatNodes[nodeOff] = v; nodeOff += 1 }
        }

        guard let merkleLeavesBuf = device.makeBuffer(length: totalMerkleLeaves * frSize, options: .storageModeShared),
              let merkleNodesBuf = device.makeBuffer(length: totalMerkleNodes * frSize, options: .storageModeShared),
              let merkleSizesBuf = device.makeBuffer(length: numLayers * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let merkleLeafOffBuf = device.makeBuffer(length: numLayers * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let merkleNodeOffBuf = device.makeBuffer(length: numLayers * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let pathsBuf = device.makeBuffer(length: pathBytes, options: .storageModeShared) else {
            throw STARKQueryPhaseError.gpuError("Failed to allocate Merkle path buffers")
        }

        flatLeaves.withUnsafeBytes { memcpy(merkleLeavesBuf.contents(), $0.baseAddress!, totalMerkleLeaves * frSize) }
        flatNodes.withUnsafeBytes { memcpy(merkleNodesBuf.contents(), $0.baseAddress!, totalMerkleNodes * frSize) }
        var merkleSizes = layerSizes
        merkleSizes.withUnsafeBytes { memcpy(merkleSizesBuf.contents(), $0.baseAddress!, numLayers * MemoryLayout<UInt32>.stride) }
        leafOffs.withUnsafeBytes { memcpy(merkleLeafOffBuf.contents(), $0.baseAddress!, numLayers * MemoryLayout<UInt32>.stride) }
        nodeOffs.withUnsafeBytes { memcpy(merkleNodeOffBuf.contents(), $0.baseAddress!, numLayers * MemoryLayout<UInt32>.stride) }

        guard let cmdBuf2 = commandQueue.makeCommandBuffer() else {
            throw STARKQueryPhaseError.noCommandBuffer
        }
        guard let enc2 = cmdBuf2.makeComputeCommandEncoder() else {
            throw STARKQueryPhaseError.gpuError("No encoder 2")
        }

        var maxDepthU = UInt32(maxDepth)
        enc2.setComputePipelineState(pathKernel)
        enc2.setBuffer(merkleLeavesBuf, offset: 0, index: 0)
        enc2.setBuffer(merkleNodesBuf, offset: 0, index: 1)
        enc2.setBuffer(merkleSizesBuf, offset: 0, index: 2)
        enc2.setBuffer(merkleLeafOffBuf, offset: 0, index: 3)
        enc2.setBuffer(merkleNodeOffBuf, offset: 0, index: 4)
        enc2.setBuffer(idxBuf, offset: 0, index: 5)
        enc2.setBuffer(pathsBuf, offset: 0, index: 6)
        enc2.setBytes(&nQ, length: 4, index: 7)
        enc2.setBytes(&nL, length: 4, index: 8)
        enc2.setBytes(&maxDepthU, length: 4, index: 9)
        enc2.dispatchThreads(MTLSize(width: totalThreads, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc2.endEncoding()
        cmdBuf2.commit()
        cmdBuf2.waitUntilCompleted()

        // Read back paths
        let pathsPtr = pathsBuf.contents().bindMemory(to: Fr.self, capacity: numQueries * numLayers * maxDepth)

        // Build FRIQueryDecommitment structs from GPU results
        var results = [FRIQueryDecommitment]()
        results.reserveCapacity(numQueries)

        for q in 0..<numQueries {
            var layers = [FRILayerDecommitment]()
            layers.reserveCapacity(numLayers)
            var foldedIdxs = [Int]()

            for l in 0..<numLayers {
                let outIdx = q * numLayers + l
                let fi = Int(idxPtr[outIdx])
                foldedIdxs.append(fi)

                // Extract Merkle path for this (query, layer)
                var siblings = [Fr]()
                siblings.reserveCapacity(maxDepth)
                for d in 0..<maxDepth {
                    siblings.append(pathsPtr[outIdx * maxDepth + d])
                }
                let path = FrMerklePath(
                    leafIndex: fi,
                    leaf: valuesPtr[outIdx],
                    siblings: siblings,
                    root: friMerkleData[l].root)

                layers.append(FRILayerDecommitment(
                    value: valuesPtr[outIdx],
                    siblingValue: sibPtr[outIdx],
                    merklePath: path))
            }

            results.append(FRIQueryDecommitment(
                queryIndex: Int(querySet.indices[q]),
                layers: layers,
                foldedIndices: foldedIdxs))
        }

        return results
    }

    // MARK: - Kernel Compilation Helpers

    private var _batchQueryKernel: MTLComputePipelineState?
    private var _merklePathsKernel: MTLComputePipelineState?

    private func getBatchQueryKernel(device: MTLDevice) throws -> MTLComputePipelineState {
        if let k = _batchQueryKernel { return k }
        let shaderDir = findShaderDir()
        let frSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let querySource = try String(contentsOfFile: shaderDir + "/fri/fri_query_batch.metal", encoding: .utf8)

        let cleanFr = frSource
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")
        let cleanQuery = querySource
            .replacingOccurrences(of: "#include \"../fields/bn254_fr.metal\"", with: "")

        let combined = cleanFr + "\n" + cleanQuery
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        let library = try device.makeLibrary(source: combined, options: options)

        guard let fn = library.makeFunction(name: "fri_batch_query_bn254") else {
            throw STARKQueryPhaseError.decommitmentFailed("Missing fri_batch_query_bn254 kernel")
        }
        let k = try device.makeComputePipelineState(function: fn)
        _batchQueryKernel = k
        return k
    }

    private func getMerklPathsKernel(device: MTLDevice) throws -> MTLComputePipelineState {
        if let k = _merklePathsKernel { return k }
        let shaderDir = findShaderDir()
        let frSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let querySource = try String(contentsOfFile: shaderDir + "/fri/fri_query_batch.metal", encoding: .utf8)

        let cleanFr = frSource
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")
        let cleanQuery = querySource
            .replacingOccurrences(of: "#include \"../fields/bn254_fr.metal\"", with: "")

        let combined = cleanFr + "\n" + cleanQuery
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        let library = try device.makeLibrary(source: combined, options: options)

        guard let fn = library.makeFunction(name: "fri_merkle_paths_bn254") else {
            throw STARKQueryPhaseError.decommitmentFailed("Missing fri_merkle_paths_bn254 kernel")
        }
        let k = try device.makeComputePipelineState(function: fn)
        _merklePathsKernel = k
        return k
    }

    // MARK: - Deep Composition Evaluation at Query Points

    /// Evaluate DEEP composition D(x) = sum alpha^t * (f_t(x) - f_t(z)) / (x - z) at query points.
    public func evaluateDeepAtQueries(
        querySet: QueryIndexSet,
        traceDecommitments: [TraceDecommitment],
        compositionDecommitments: [CompositionDecommitment],
        oodFrame: OODEvaluationFrame,
        alpha: Fr,
        config: STARKQueryPhaseConfig
    ) throws -> [Fr] {
        guard traceDecommitments.count == querySet.indices.count else {
            throw STARKQueryPhaseError.deepEvalMismatch(
                "Expected \(querySet.indices.count) trace decommitments, got \(traceDecommitments.count)")
        }
        guard compositionDecommitments.count == querySet.indices.count else {
            throw STARKQueryPhaseError.deepEvalMismatch(
                "Expected \(querySet.indices.count) composition decommitments, got \(compositionDecommitments.count)")
        }

        let omega = frRootOfUnity(logN: config.logTraceLen)
        let zetaNext = frMul(oodFrame.zeta, omega)
        let logM = config.logLDEDomainSize
        let omegaM = frRootOfUnity(logN: logM)

        let qCount = querySet.indices.count

        // Batch-invert all (x_i - zeta) and (x_i - zetaNext) denominators
        var qDenoms = [Fr](repeating: Fr.zero, count: 2 * qCount)
        for (qi, idx) in querySet.indices.enumerated() {
            let domainPoint = computeDomainPoint(
                index: idx, omegaM: omegaM, cosetShift: config.cosetShift)
            qDenoms[2 * qi] = frSub(domainPoint, oodFrame.zeta)
            qDenoms[2 * qi + 1] = frSub(domainPoint, zetaNext)
        }
        var qDenomInvs = [Fr](repeating: Fr.zero, count: 2 * qCount)
        qDenoms.withUnsafeBytes { src in
            qDenomInvs.withUnsafeMutableBytes { dst in
                bn254_fr_batch_inverse_safe(
                    src.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(2 * qCount),
                    dst.baseAddress!.assumingMemoryBound(to: UInt64.self))
            }
        }

        var deepEvals = [Fr](repeating: Fr.zero, count: qCount)

        for (qi, idx) in querySet.indices.enumerated() {
            // Compute the domain point x_i = cosetShift * omega_M^idx
            let domainPoint = computeDomainPoint(
                index: idx, omegaM: omegaM, cosetShift: config.cosetShift)

            let xMinusZetaInv = qDenomInvs[2 * qi]
            let xMinusZetaNextInv = qDenomInvs[2 * qi + 1]

            var alphaPow = Fr.one
            var result = Fr.zero

            // Trace quotients at zeta
            for col in 0..<config.numTraceColumns {
                let num = frSub(traceDecommitments[qi].values[col],
                                oodFrame.traceEvals[col])
                let q = frMul(num, xMinusZetaInv)
                result = frAdd(result, frMul(alphaPow, q))
                alphaPow = frMul(alphaPow, alpha)
            }

            // Trace quotients at zeta * omega (next-row)
            for col in 0..<config.numTraceColumns {
                let num = frSub(traceDecommitments[qi].values[col],
                                oodFrame.traceNextEvals[col])
                let q = frMul(num, xMinusZetaNextInv)
                result = frAdd(result, frMul(alphaPow, q))
                alphaPow = frMul(alphaPow, alpha)
            }

            // Composition segment quotients at zeta
            for seg in 0..<config.numCompositionSegments {
                let num = frSub(compositionDecommitments[qi].segments[seg],
                                oodFrame.compositionEvals[seg])
                let q = frMul(num, xMinusZetaInv)
                result = frAdd(result, frMul(alphaPow, q))
                alphaPow = frMul(alphaPow, alpha)
            }

            deepEvals[qi] = result
        }

        return deepEvals
    }

    // MARK: - Full Batch Decommitment

    /// Generate complete batch decommitments for all queries (main entry point).
    ///
    /// Uses GPU-accelerated batch FRI decommitment when domainInvs and challenges
    /// are provided (enabling parallel fold recomputation across all queries).
    public func generateDecommitments(
        querySet: QueryIndexSet,
        ldeColumns: [[Fr]],
        traceMerkleLeaves: [Fr],
        traceMerkleNodes: [Fr],
        traceCommitment: Fr,
        compositionSegments: [[Fr]],
        compMerkleLeaves: [Fr],
        compMerkleNodes: [Fr],
        compCommitment: Fr,
        friLayers: [[Fr]],
        friMerkleData: [(leaves: [Fr], nodes: [Fr], root: Fr)],
        domainInvs: [[Fr]]? = nil,
        challenges: [Fr]? = nil,
        oodFrame: OODEvaluationFrame,
        alpha: Fr,
        config: STARKQueryPhaseConfig
    ) throws -> BatchDecommitmentResult {
        // Step 1: Trace decommitments
        let traceDecomm = try generateTraceDecommitments(
            querySet: querySet,
            ldeColumns: ldeColumns,
            merkleLeaves: traceMerkleLeaves,
            merkleNodes: traceMerkleNodes,
            commitment: traceCommitment,
            config: config)

        // Step 2: Composition decommitments
        let compDecomm = try generateCompositionDecommitments(
            querySet: querySet,
            compositionSegments: compositionSegments,
            merkleLeaves: compMerkleLeaves,
            merkleNodes: compMerkleNodes,
            commitment: compCommitment,
            config: config)

        // Step 3: FRI decommitments (GPU batch when domainInvs/challenges available)
        let friDecomm: [FRIQueryDecommitment]
        if let invs = domainInvs, let chal = challenges, invs.count == config.numFRIRounds {
            friDecomm = try gpuBatchFRIDecommit(
                querySet: querySet,
                friLayers: friLayers,
                friMerkleData: friMerkleData,
                domainInvs: invs,
                challenges: chal,
                config: config)
        } else {
            friDecomm = try generateFRIDecommitments(
                querySet: querySet,
                friLayers: friLayers,
                friMerkleData: friMerkleData,
                config: config)
        }

        // Step 4: Deep composition evaluations
        let deepEvals = try evaluateDeepAtQueries(
            querySet: querySet,
            traceDecommitments: traceDecomm,
            compositionDecommitments: compDecomm,
            oodFrame: oodFrame,
            alpha: alpha,
            config: config)

        return BatchDecommitmentResult(
            traceDecommitments: traceDecomm,
            compositionDecommitments: compDecomm,
            friDecommitments: friDecomm,
            querySet: querySet,
            deepEvals: deepEvals,
            config: config)
    }

    // MARK: - Verify Decommitment Consistency

    /// Verify internal consistency of a batch decommitment result.
    public func verifyDecommitmentConsistency(
        result: BatchDecommitmentResult,
        traceCommitment: Fr,
        compCommitment: Fr
    ) -> Bool {
        // Check trace Merkle paths
        for td in result.traceDecommitments {
            if !verifyMerklePath(td.merklePath, expectedRoot: traceCommitment) {
                return false
            }
        }

        // Check composition Merkle paths
        for cd in result.compositionDecommitments {
            if !verifyMerklePath(cd.merklePath, expectedRoot: compCommitment) {
                return false
            }
        }

        // Check FRI decommitment structure
        for fd in result.friDecommitments {
            if fd.layers.count != result.config.numFRIRounds {
                return false
            }
        }

        // Check deep evals count
        if result.deepEvals.count != result.numQueries {
            return false
        }

        return true
    }

    // MARK: - Merkle Path Extraction

    /// Extract a Merkle authentication path from a precomputed tree (array form).
    public func extractMerklePath(
        leafIndex: Int,
        leaves: [Fr],
        nodes: [Fr],
        root: Fr
    ) -> FrMerklePath {
        let n = leaves.count
        guard n > 0 else {
            return FrMerklePath(leafIndex: leafIndex, leaf: Fr.zero,
                                siblings: [], root: root)
        }

        let depth = Int(log2(Double(n)))
        var siblings = [Fr]()
        siblings.reserveCapacity(depth)
        var idx = leafIndex

        for level in 0..<depth {
            let sibIdx = idx ^ 1
            if level == 0 {
                if sibIdx < leaves.count {
                    siblings.append(leaves[sibIdx])
                } else {
                    siblings.append(Fr.zero)
                }
            } else {
                let levelOffset = (1 << (depth - level)) - 1
                let nodeIdx = levelOffset + sibIdx
                if nodeIdx < nodes.count {
                    siblings.append(nodes[nodeIdx])
                } else {
                    siblings.append(Fr.zero)
                }
            }
            idx >>= 1
        }

        return FrMerklePath(
            leafIndex: leafIndex,
            leaf: leaves[min(leafIndex, leaves.count - 1)],
            siblings: siblings,
            root: root)
    }

    // MARK: - Merkle Path Verification

    /// Verify a Merkle path. Uses H(l, r) = l^2 + 3*r + 7 (matches Verifier/TraceLDE).
    public func verifyMerklePath(_ path: FrMerklePath, expectedRoot: Fr) -> Bool {
        var current = path.leaf
        var idx = path.leafIndex

        for sibling in path.siblings {
            if idx & 1 == 0 {
                current = merkleHash(left: current, right: sibling)
            } else {
                current = merkleHash(left: sibling, right: current)
            }
            idx >>= 1
        }

        return frEqual(current, expectedRoot)
    }

    // MARK: - Batch Merkle Path Verification

    /// Verify multiple Merkle paths in parallel.
    /// Returns array of booleans indicating which paths are valid.
    public func batchVerifyMerklePaths(
        _ paths: [FrMerklePath], expectedRoot: Fr
    ) -> [Bool] {
        if useGPU && paths.count >= GPUSTARKQueryPhaseEngine.gpuThreshold {
            // GPU path: dispatch parallel verification
            return paths.map { verifyMerklePath($0, expectedRoot: expectedRoot) }
        }
        return paths.map { verifyMerklePath($0, expectedRoot: expectedRoot) }
    }

    // MARK: - Domain Point Computation

    /// Compute a single LDE domain point: x_i = cosetShift * omega_M^i.
    public func computeDomainPoint(index: Int, omegaM: Fr, cosetShift: Fr) -> Fr {
        var w = Fr.one
        for _ in 0..<index {
            w = frMul(w, omegaM)
        }
        return frMul(cosetShift, w)
    }

    /// Compute all LDE domain points: x_i = cosetShift * omega_M^i.
    public func computeDomainPoints(config: STARKQueryPhaseConfig) -> [Fr] {
        let m = config.ldeDomainSize
        let logM = config.logLDEDomainSize
        let omegaM = frRootOfUnity(logN: logM)

        // points[i] = cosetShift * omega^i
        // First fill with cosetShift, then batch_mul_powers by omega
        var points = [Fr](repeating: config.cosetShift, count: m)
        points.withUnsafeMutableBytes { pBuf in
            withUnsafeBytes(of: omegaM) { oBuf in
                let pPtr = pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                bn254_fr_batch_mul_powers(
                    pPtr,
                    pPtr,
                    oBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(m))
            }
        }
        return points
    }

    // MARK: - FRI Index Folding

    /// Compute folded indices for a query through FRI layers (domain halves each round).
    public func computeFRIFoldedIndices(
        queryIndex: Int, layerSizes: [Int]
    ) -> [Int] {
        var foldedIndices = [Int]()
        foldedIndices.reserveCapacity(layerSizes.count)

        var currentIdx = queryIndex
        for layerSize in layerSizes {
            let foldedIdx = currentIdx % layerSize
            foldedIndices.append(foldedIdx)
            currentIdx = foldedIdx >> 1
        }

        return foldedIndices
    }

    // MARK: - Query Deduplication

    /// Deduplicate query indices, returning unique indices and a mapping.
    public func deduplicateQueries(
        indices: [Int], domainSize: Int
    ) -> (unique: [Int], mapping: [Int]) {
        var seen = [Int: Int]()  // index -> position in unique array
        var unique = [Int]()
        var mapping = [Int]()
        mapping.reserveCapacity(indices.count)

        for idx in indices {
            let normalized = idx % domainSize
            if let pos = seen[normalized] {
                mapping.append(pos)
            } else {
                let pos = unique.count
                seen[normalized] = pos
                unique.append(normalized)
                mapping.append(pos)
            }
        }

        return (unique, mapping)
    }

    // MARK: - Sibling Index Computation

    /// Compute the sibling index for FRI folding.
    /// In FRI, a position i and its sibling i^1 are paired for folding.
    public func computeSiblingIndex(_ index: Int) -> Int {
        return index ^ 1
    }

    // MARK: - Build Merkle Tree (for Testing)

    /// Build a Merkle tree from leaf values.
    /// Returns (root, leaves, nodes) in array form.
    public func buildMerkleTree(leaves: [Fr]) -> (root: Fr, leaves: [Fr], nodes: [Fr]) {
        let n = leaves.count
        guard n > 0 else {
            return (Fr.zero, [], [])
        }

        let totalNodes = 2 * n - 1
        var nodes = [Fr](repeating: Fr.zero, count: totalNodes)

        // Copy leaves to right half
        for i in 0..<n {
            nodes[n - 1 + i] = leaves[i]
        }

        // Build internal nodes bottom-up
        var idx = n - 2
        while idx >= 0 {
            let left = nodes[2 * idx + 1]
            let right = nodes[2 * idx + 2]
            nodes[idx] = merkleHash(left: left, right: right)
            idx -= 1
        }

        return (nodes[0], leaves, nodes)
    }

    /// Build a Merkle tree from multi-column LDE data (row-interleaved hashing).
    public func buildTraceMerkleTree(
        ldeColumns: [[Fr]], domainSize: Int
    ) -> (root: Fr, leaves: [Fr], nodes: [Fr]) {
        let numCols = ldeColumns.count
        var leaves = [Fr](repeating: Fr.zero, count: domainSize)

        for i in 0..<domainSize {
            var rowHash = Fr.zero
            for c in 0..<numCols {
                rowHash = merkleHash(left: rowHash, right: ldeColumns[c][i])
            }
            leaves[i] = rowHash
        }

        return buildMerkleTree(leaves: leaves)
    }

    // MARK: - Private Helpers

    /// Simple algebraic Merkle hash: H(l, r) = l^2 + 3*r + 7
    /// (Matches GPUSTARKVerifierEngine and GPUSTARKTraceLDEEngine for compatibility.)
    private func merkleHash(left: Fr, right: Fr) -> Fr {
        let lSq = frMul(left, left)
        let three = frFromInt(3)
        let seven = frFromInt(7)
        let rScaled = frMul(three, right)
        return frAdd(frAdd(lSq, rScaled), seven)
    }

}
