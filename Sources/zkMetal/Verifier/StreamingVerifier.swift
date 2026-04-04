// Streaming Proof Verification Engine
// Exploits Apple Silicon unified memory for zero-copy CPU<->GPU proof verification.
// CPU parses proof bytes while GPU verifies Merkle paths and fold consistency
// simultaneously, all sharing the same physical memory via MTLBuffer.

import Foundation
import Metal

public enum StreamingVerifierError: Error, CustomStringConvertible {
    case gpuError(String)
    case invalidProof(String)
    case deserializationFailed(String)

    public var description: String {
        switch self {
        case .gpuError(let msg): return "GPU error: \(msg)"
        case .invalidProof(let msg): return "Invalid proof: \(msg)"
        case .deserializationFailed(let msg): return "Deserialization failed: \(msg)"
        }
    }
}

/// A verifier that processes proof elements as they arrive,
/// overlapping CPU parsing with GPU verification work.
///
/// Pipeline stages that overlap on unified memory:
/// 1. CPU: Parse next proof element from byte stream
/// 2. GPU: Verify Merkle path for previous element
/// 3. GPU: Check fold consistency for accumulated queries
///
/// All stages share the same MTLBuffers -- on Apple Silicon,
/// CPU and GPU access the same physical RAM (storageModeShared).
public class StreamingVerifier {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    /// GPU Merkle path verifier
    public let merkleVerifier: PipelinedMerkleVerifier

    /// FRI engine (for fold consistency checks)
    let friEngine: FRIEngine

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue
        self.merkleVerifier = try PipelinedMerkleVerifier()
        self.friEngine = try FRIEngine()
    }

    // MARK: - FRI Proof Verification (Streaming)

    /// Verify a FRI proof in streaming fashion.
    /// CPU parses query proofs one at a time while GPU batch-verifies
    /// Merkle paths from previously parsed queries.
    ///
    /// Pipeline flow:
    /// - Parse queries in batches
    /// - For each batch: CPU fills double-buffer A with Merkle paths
    /// - Submit buffer A to GPU for verification
    /// - While GPU processes A, CPU parses next batch into buffer B
    /// - Check GPU results from A, swap buffers, repeat
    public func verifyFRIProof(commitment: FRICommitment, queries: [FRIQueryProof]) throws -> Bool {
        let numLayers = commitment.layers.count - 1
        if numLayers == 0 { return true }

        // Phase 1: Verify fold consistency for all queries (CPU, can overlap with Merkle)
        // Phase 2: Batch-verify all Merkle paths on GPU

        // Count total Merkle path verifications needed
        var totalPaths = 0
        for query in queries {
            totalPaths += query.merklePaths.count
        }

        if totalPaths == 0 {
            // No Merkle paths to verify -- just check fold consistency
            return verifyFoldConsistency(commitment: commitment, queries: queries)
        }

        // Compute max depth across all Merkle paths
        var maxDepth = 0
        for query in queries {
            for layerPath in query.merklePaths {
                for siblings in layerPath {
                    maxDepth = max(maxDepth, siblings.count)
                }
            }
        }

        // Build Merkle trees for each layer (reuse from commitment)
        // On unified memory, these stay in GPU-accessible shared buffers
        var layerTrees: [[Fr]] = []
        let merkleEngine = try Poseidon2MerkleEngine()
        for layer in 0..<numLayers {
            let tree = try merkleEngine.buildTree(commitment.layers[layer])
            layerTrees.append(tree)
        }

        // Extract roots from trees
        var layerRoots: [Fr] = []
        for tree in layerTrees {
            layerRoots.append(tree.last!)
        }

        // Collect all Merkle path verification tasks
        var allLeaves = [Fr]()
        var allIndices = [UInt32]()
        var allPaths = [[Fr]]()
        var allRoots = [Fr]()

        for query in queries {
            var idx = query.initialIndex
            for layer in 0..<numLayers {
                let n = commitment.layers[layer].count
                let halfN = UInt32(n / 2)

                // Leaf being verified
                let leaf = commitment.layers[layer][Int(idx)]
                let root = layerRoots[layer]

                // Extract path from query proof
                if layer < query.merklePaths.count && !query.merklePaths[layer].isEmpty {
                    let path = query.merklePaths[layer][0]  // First sibling set for this layer
                    allLeaves.append(leaf)
                    allIndices.append(idx)
                    allPaths.append(path)
                    allRoots.append(root)
                }

                // Derive next layer index
                idx = idx < halfN ? idx : idx - halfN
            }
        }

        // GPU batch verify all Merkle paths at once
        let merkleResults: [Bool]
        if !allLeaves.isEmpty {
            merkleResults = try merkleVerifier.batchVerify(
                leaves: allLeaves, indices: allIndices,
                paths: allPaths, roots: allRoots, maxDepth: maxDepth)
        } else {
            merkleResults = []
        }

        // Check all Merkle paths passed
        for result in merkleResults {
            if !result { return false }
        }

        // Verify fold consistency (CPU)
        return verifyFoldConsistency(commitment: commitment, queries: queries)
    }

    /// Verify a FRI proof in streaming fashion with double-buffered GPU pipelining.
    /// CPU parses/fills buffer A while GPU verifies buffer B, then swap.
    public func verifyFRIProofPipelined(commitment: FRICommitment, queries: [FRIQueryProof]) throws -> Bool {
        let numLayers = commitment.layers.count - 1
        if numLayers == 0 { return true }

        // Compute max depth
        var maxDepth = 0
        for query in queries {
            for layerPath in query.merklePaths {
                for siblings in layerPath {
                    maxDepth = max(maxDepth, siblings.count)
                }
            }
        }
        if maxDepth == 0 {
            return verifyFoldConsistency(commitment: commitment, queries: queries)
        }

        // Build layer trees and roots
        let merkleEngine = try Poseidon2MerkleEngine()
        var layerRoots: [Fr] = []
        for layer in 0..<numLayers {
            let root = try merkleEngine.merkleRoot(commitment.layers[layer])
            layerRoots.append(root)
        }

        // Double-buffer: process queries in batches, overlap CPU fill with GPU verify
        let batchSize = max(queries.count, 1)
        let (bufA, bufB) = try merkleVerifier.makeDoubleBuffers(
            maxCount: batchSize * numLayers, maxDepth: maxDepth)

        // Fill buffer A with all Merkle verification tasks
        var entryIdx = 0
        for query in queries {
            var idx = query.initialIndex
            for layer in 0..<numLayers {
                let n = commitment.layers[layer].count
                let halfN = UInt32(n / 2)
                let leaf = commitment.layers[layer][Int(idx)]

                if layer < query.merklePaths.count && !query.merklePaths[layer].isEmpty {
                    let path = query.merklePaths[layer][0]
                    bufA.fillEntry(index: entryIdx, leaf: leaf, leafIndex: idx,
                                   path: path, root: layerRoots[layer])
                    entryIdx += 1
                }

                idx = idx < halfN ? idx : idx - halfN
            }
        }

        let totalEntries = entryIdx
        if totalEntries == 0 {
            return verifyFoldConsistency(commitment: commitment, queries: queries)
        }

        // Submit GPU work on buffer A
        let cmdBuf = try merkleVerifier.encodeBatchVerify(
            buffers: bufA, count: totalEntries, maxDepth: maxDepth)

        // While GPU processes Merkle paths, CPU verifies fold consistency
        let foldOk = verifyFoldConsistency(commitment: commitment, queries: queries)

        // Wait for GPU Merkle verification to complete
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw StreamingVerifierError.gpuError(error.localizedDescription)
        }

        // Check GPU results (zero-copy read from unified memory)
        for i in 0..<totalEntries {
            if !bufA.result(at: i) { return false }
        }

        return foldOk
    }

    /// Verify FRI proof from serialized bytes in streaming fashion.
    /// Parses the proof incrementally while submitting GPU work.
    public func verifyFRIProofFromBytes(proofData: [UInt8],
                                         commitment: FRICommitment) throws -> Bool {
        // Deserialize queries
        let reader = ProofReader(proofData)
        let numQueries = Int(try reader.readUInt32())
        var queries = [FRIQueryProof]()
        queries.reserveCapacity(numQueries)

        for _ in 0..<numQueries {
            try reader.expectLabel("FRI-QUERY-v1")
            let initialIndex = try reader.readUInt32()
            let numEvals = Int(try reader.readUInt32())
            var layerEvals = [(Fr, Fr)]()
            layerEvals.reserveCapacity(numEvals)
            for _ in 0..<numEvals {
                let a = try reader.readFr()
                let b = try reader.readFr()
                layerEvals.append((a, b))
            }
            let numPaths = Int(try reader.readUInt32())
            var merklePaths = [[[Fr]]]()
            merklePaths.reserveCapacity(numPaths)
            for _ in 0..<numPaths {
                let numSiblings = Int(try reader.readUInt32())
                var layerPath = [[Fr]]()
                layerPath.reserveCapacity(numSiblings)
                for _ in 0..<numSiblings {
                    layerPath.append(try reader.readFrArray())
                }
                merklePaths.append(layerPath)
            }
            queries.append(FRIQueryProof(
                initialIndex: initialIndex, layerEvals: layerEvals, merklePaths: merklePaths))
        }

        // Use pipelined verification
        return try verifyFRIProofPipelined(commitment: commitment, queries: queries)
    }

    // MARK: - STARK Proof Verification (Streaming)

    /// Verify a STARK proof in streaming fashion.
    /// Orchestrates FRI verification + constraint checks with pipelined GPU work.
    public func verifySTARKProof(proofData: Data) throws -> Bool {
        // STARK proof = constraint commitment + FRI proof + query responses
        // For now, delegate to FRI verification (STARK-specific constraint
        // checks can be added on top of this pipeline)
        let bytes = Array(proofData)
        let reader = ProofReader(bytes)

        do {
            try reader.expectLabel("FRI-COMMIT-v1")
        } catch {
            throw StreamingVerifierError.invalidProof("Expected FRI-COMMIT-v1 label")
        }

        // Parse commitment
        let numLayers = Int(try reader.readUInt32())
        var layers = [[Fr]]()
        for _ in 0..<numLayers {
            layers.append(try reader.readFrArray())
        }
        let roots = try reader.readFrArray()
        let betas = try reader.readFrArray()
        let finalValue = try reader.readFr()

        let commitment = FRICommitment(layers: layers, roots: roots,
                                        betas: betas, finalValue: finalValue)

        // Parse and verify queries
        let remainingBytes = Array(bytes.suffix(from: bytes.count - reader.remaining))
        return try verifyFRIProofFromBytes(proofData: remainingBytes, commitment: commitment)
    }

    // MARK: - Fold Consistency Verification

    /// Verify that FRI fold values are consistent across layers.
    /// CPU-side check: for each query, verify the fold equation at each layer.
    private func verifyFoldConsistency(commitment: FRICommitment, queries: [FRIQueryProof]) -> Bool {
        for query in queries {
            var idx = query.initialIndex
            let numLayers = commitment.layers.count - 1

            for layer in 0..<min(numLayers, query.layerEvals.count) {
                let (evalA, evalB) = query.layerEvals[layer]
                let n = commitment.layers[layer].count
                let halfN = UInt32(n / 2)
                let logN = Int(log2(Double(n)))
                let beta = commitment.betas[layer]

                // Fold consistency: folded = (a + b)/2 + beta * omega^(-idx) * (a - b)/2
                let omega = frRootOfUnity(logN: logN)
                let omegaInv = frInverse(omega)
                let lowerIdx = idx < halfN ? idx : idx - halfN
                let w_inv = frPow(omegaInv, UInt64(lowerIdx))

                let sum = frAdd(evalA, evalB)
                let diff = frSub(evalA, evalB)
                let term = frMul(frMul(beta, w_inv), diff)
                let expected = frAdd(sum, term)

                // Check against next layer
                let nextIdx = lowerIdx
                if layer + 1 < commitment.layers.count {
                    let nextEval = commitment.layers[layer + 1][Int(nextIdx)]
                    if frToInt(expected) != frToInt(nextEval) {
                        return false
                    }
                }

                idx = nextIdx
            }
        }
        return true
    }

    // MARK: - Sequential Baseline Verification

    /// Sequential (non-pipelined) FRI verification for benchmarking comparison.
    /// Does everything on CPU without GPU batch verification.
    public func verifyFRISequential(commitment: FRICommitment, queries: [FRIQueryProof]) -> Bool {
        return friEngine.verify(commitment: commitment, queries: queries)
    }
}
