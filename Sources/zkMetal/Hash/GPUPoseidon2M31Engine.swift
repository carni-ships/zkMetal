// GPUPoseidon2M31Engine.swift — ANE/GPU-accelerated Poseidon2-M31 for Merkle tree operations
//
// This module bridges the ANE Poseidon2 batch API to enable fast Merkle tree construction.
// The ANE processes multiple Poseidon2 permutations in a single GPU dispatch.
//
// Optimization: Batch Poseidon2 across multiple trees at each level to maximize ANE throughput.

import Foundation
import ANEOps

// MARK: - Poseidon2 Execution Mode

/// Controls which hardware path to use for Poseidon2-M31 batch hashing
public enum Poseidon2ExecutionMode: String, CaseIterable {
    case cpu = "CPU-only"
    case gpuOnly = "GPU-only"
    case aneOnly = "ANE-only"
    case gpuANE = "GPU+ANE"

    public var description: String { rawValue }
}

/// Minimum batch size to justify ANE/GPU dispatch overhead
public let poseidon2MinBatchSize: Int = 256

// MARK: - Round Constants Conversion

/// Convert Swift [[M31]] round constants to UInt32 array for ANE
func poseidon2M31RoundConstantsToUInt32(_ rc: [[M31]]) -> [UInt32] {
    var result = [UInt32]()
    result.reserveCapacity(rc.count * rc[0].count)
    for round in rc {
        for val in round {
            result.append(val.v)
        }
    }
    return result
}

/// Cached ANE round constants (converted once)
private let aneM31RoundConstants: [UInt32] = {
    poseidon2M31RoundConstantsToUInt32(POSEIDON2_M31_ROUND_CONSTANTS)
}()

// MARK: - Batch Internal Node Hashing

/// Batch hash internal Merkle nodes using ANE Poseidon2.
/// Processes n_pairs of (leftDigest, rightDigest) → outputDigest in a single GPU dispatch.
///
/// - Parameters:
///   - pairs: Array of (left, right) digest pairs to hash
///   - output: Output array (must have capacity for n_pairs digests)
/// - Returns: Number of pairs processed, or -1 on error
@discardableResult
public func anePoseidon2BatchHashInternalNodes(
    _ pairs: [(M31Digest, M31Digest)],
    output: inout [M31Digest]
) -> Int {
    // Ensure ANE is initialized before first use
    anePoseidon2PreWarm()

    let nPairs = pairs.count
    guard nPairs > 0 else { return 0 }

    // Convert pairs to flat uint32 array for ANE
    // Layout: [l0, l1, ..., l7, r0, r1, ..., r7] per pair
    var flatInput = [UInt32]()
    flatInput.reserveCapacity(nPairs * 16)
    for pair in pairs {
        for i in 0..<8 {
            flatInput.append(pair.0.values[i].v)
        }
        for i in 0..<8 {
            flatInput.append(pair.1.values[i].v)
        }
    }

    // Allocate output array (ANE writes 16 elements per permutation)
    var flatOutput = [UInt32](repeating: 0, count: nPairs * 16)

    // Call ANE batch permutation
    flatInput.withUnsafeBufferPointer { inputPtr in
        flatOutput.withUnsafeMutableBufferPointer { outputPtr in
            m31_poseidon2_permutation_batch_ane(
                inputPtr.baseAddress,
                Int32(nPairs),
                aneM31RoundConstants,
                outputPtr.baseAddress
            )
        }
    }

    // Convert output back to M31Digest
    // ANE returns 16 elements per permutation, we only need first 8
    output.reserveCapacity(nPairs)
    for i in 0..<nPairs {
        var values = [M31]()
        values.reserveCapacity(8)
        for j in 0..<8 {
            values.append(M31(v: flatOutput[i * 16 + j]))
        }
        output.append(M31Digest(values: values))
    }

    return nPairs
}

// MARK: - CPU-only Batch Hashing

/// Batch hash internal Merkle nodes using CPU Poseidon2 (pure scalar).
public func cpuPoseidon2BatchHashInternalNodes(
    _ pairs: [(M31Digest, M31Digest)],
    output: inout [M31Digest]
) -> Int {
    let nPairs = pairs.count
    guard nPairs > 0 else { return 0 }

    output.reserveCapacity(nPairs)
    for pair in pairs {
        let hashResult = poseidon2M31Hash(left: pair.0.values, right: pair.1.values)
        output.append(M31Digest(values: hashResult))
    }
    return nPairs
}

// MARK: - GPU-only Batch Hashing (Metal shader without ANE)

/// Batch hash internal Merkle nodes using GPU Metal shader only (no ANE).
/// Uses Poseidon2M31Engine directly for maximum GPU control.
private var gpuEngine: Poseidon2M31Engine?
private let gpuEngineLock = NSLock()

public func gpuOnlyPoseidon2BatchHashInternalNodes(
    _ pairs: [(M31Digest, M31Digest)],
    output: inout [M31Digest]
) -> Int {
    let nPairs = pairs.count
    guard nPairs > 0 else { return 0 }

    // Ensure GPU engine is initialized
    gpuEngineLock.lock()
    if gpuEngine == nil {
        do {
            gpuEngine = try Poseidon2M31Engine()
        } catch {
            gpuEngineLock.unlock()
            return -1
        }
    }
    gpuEngineLock.unlock()

    // Convert pairs to flat array for GPU
    var flatInput = [M31]()
    flatInput.reserveCapacity(nPairs * 16)
    for pair in pairs {
        for i in 0..<8 {
            flatInput.append(pair.0.values[i])
        }
        for i in 0..<8 {
            flatInput.append(pair.1.values[i])
        }
    }

    do {
        // Use GPU engine to hash pairs
        let gpuResults = try gpuEngine!.hashPairs(flatInput)
        output.reserveCapacity(nPairs)
        for i in 0..<nPairs {
            var values = [M31]()
            values.reserveCapacity(8)
            for j in 0..<8 {
                values.append(gpuResults[i * 8 + j])
            }
            output.append(M31Digest(values: values))
        }
        return nPairs
    } catch {
        return -1
    }
}

// MARK: - Mode-aware Batch Hashing

/// Hash internal Merkle nodes using the specified execution mode.
public func poseidon2BatchHashInternalNodes(
    _ pairs: [(M31Digest, M31Digest)],
    output: inout [M31Digest],
    mode: Poseidon2ExecutionMode
) -> Int {
    let nPairs = pairs.count

    // Fallback to CPU for small batches where GPU/ANE overhead dominates
    if nPairs < poseidon2MinBatchSize {
        return cpuPoseidon2BatchHashInternalNodes(pairs, output: &output)
    }

    switch mode {
    case .cpu:
        return cpuPoseidon2BatchHashInternalNodes(pairs, output: &output)
    case .gpuOnly:
        return gpuOnlyPoseidon2BatchHashInternalNodes(pairs, output: &output)
    case .aneOnly, .gpuANE:
        return anePoseidon2BatchHashInternalNodes(pairs, output: &output)
    }
}

// MARK: - Batched Poseidon2 Merkle Tree Building (Multi-Tree)

/// Input for batched Merkle tree building
public struct BatchedMerkleInput {
    /// Leaf values for each tree (all trees must have same leaf count)
    public let leafValues: [[M31]]
    /// Number of leaves per tree (all must be same)
    public let leafCount: Int

    public init(leafValues: [[M31]], leafCount: Int) {
        self.leafValues = leafValues
        self.leafCount = leafCount
    }
}

/// Result of batched Merkle tree building
public struct BatchedMerkleResult {
    /// Merkle roots for each tree (in input order)
    public let roots: [M31Digest]
    /// Merkle trees for each tree (in input order)
    public let trees: [[M31Digest]]
}

/// Build multiple Poseidon2-M31 Merkle trees in a single batch pass.
///
/// Batches all trees at each level to maximize throughput:
/// - Level 0: batch all trees' first-level internal nodes
/// - Level 1: batch all trees' second-level internal nodes
/// - etc.
///
/// This reduces calls from T × log(n) to log(n) where T = number of trees.
///
/// Optimizations:
/// - CPU fallback: uses CPU for small batches (< 256 pairs) where GPU overhead dominates
/// - Buffer pooling: reuses Metal buffers for reduced allocation overhead
///
/// - Parameters:
///   - inputs: Array of leaf value arrays (one per tree)
///   - leafCount: Number of leaves per tree (must be power of 2)
///   - mode: Execution mode (CPU, GPU-only, ANE-only, GPU+ANE)
/// - Returns: Roots and trees for all inputs
public func poseidon2BatchBuildMerkleTrees(
    _ inputs: [[M31]], leafCount n: Int,
    mode: Poseidon2ExecutionMode = .gpuANE
) -> BatchedMerkleResult {
    let numTrees = inputs.count
    precondition(numTrees > 0, "Must have at least one tree")
    precondition(n > 0 && (n & (n - 1)) == 0, "n must be a power of 2")

    // Pre-compute all leaf hashes sequentially (CPU - fast, Poseidon2 is quick)
    var leafHashes = [[M31Digest]]()
    leafHashes.reserveCapacity(numTrees)
    for treeIdx in 0..<numTrees {
        var hashes = [M31Digest](repeating: .zero, count: n)
        let values = inputs[treeIdx]
        for i in 0..<n {
            let val = i < values.count ? values[i] : M31.zero
            let leafInput = [val, M31(v: UInt32(i)), M31.zero, M31.zero,
                            M31.zero, M31.zero, M31.zero, M31.zero]
            let digest = poseidon2M31HashSingle(leafInput)
            hashes[i] = M31Digest(values: digest)
        }
        leafHashes.append(hashes)
    }

    // Build trees bottom-up, batching all trees at each level
    // currentLevel[treeIdx] = array of digests at current level for each tree
    var currentLevel = leafHashes
    var levelSize = n
    var levels = [[[M31Digest]]]()  // levels[levelIdx][treeIdx] = digest array

    while levelSize > 1 {
        let parentSize = levelSize / 2

        // Collect all pairs from all trees at this level, interleaved
        // Layout: [tree0_pair0, tree0_pair1, ..., tree0_pairN, tree1_pair0, ...]
        // Standard consecutive pairing: (0,1), (2,3), (4,5), (6,7) at each level
        var allPairs = [(M31Digest, M31Digest)]()
        allPairs.reserveCapacity(numTrees * parentSize)

        for treeIdx in 0..<numTrees {
            let treeLevel = currentLevel[treeIdx]
            for i in 0..<parentSize {
                let left = treeLevel[2 * i]
                let right = treeLevel[2 * i + 1]
                allPairs.append((left, right))
            }
        }

        // Batch hash all pairs at this level using the specified mode
        // Note: poseidon2BatchHashInternalNodes auto-falls back to CPU for small batches
        var allResults = [M31Digest]()
        allResults.reserveCapacity(numTrees * parentSize)
        let pairsProcessed = poseidon2BatchHashInternalNodes(allPairs, output: &allResults, mode: mode)
        if pairsProcessed != allPairs.count {
            // Fallback to CPU if batching fails
            allResults.removeAll()
            for pair in allPairs {
                let hashResult = poseidon2M31Hash(left: pair.0.values, right: pair.1.values)
                allResults.append(M31Digest(values: hashResult))
            }
        }

        // Extract parent digests for each tree
        var nextLevel = [[M31Digest]](repeating: [], count: numTrees)
        for treeIdx in 0..<numTrees {
            var treeParents = [M31Digest](repeating: .zero, count: parentSize)
            for i in 0..<parentSize {
                let resultIdx = treeIdx * parentSize + i
                treeParents[i] = allResults[resultIdx]
            }
            nextLevel[treeIdx] = treeParents
        }

        levels.append(currentLevel)
        currentLevel = nextLevel
        levelSize = parentSize
    }

    // Reconstruct individual trees from levels
    var trees = [[M31Digest]]()
    trees.reserveCapacity(numTrees)
    for treeIdx in 0..<numTrees {
        var tree = [M31Digest]()
        // Level 0 (leaves) to root
        for levelIdx in 0..<levels.count {
            tree.append(contentsOf: levels[levelIdx][treeIdx])
        }
        // Root
        tree.append(currentLevel[treeIdx][0])
        trees.append(tree)
    }

    // Extract roots
    var roots = [M31Digest]()
    roots.reserveCapacity(numTrees)
    for treeIdx in 0..<numTrees {
        roots.append(currentLevel[treeIdx][0])
    }

    return BatchedMerkleResult(roots: roots, trees: trees)
}

// MARK: - Single Tree Building

/// Build Poseidon2-M31 Merkle tree using the specified execution mode.
/// Leaves are hashed individually (CPU), internal nodes are batched per mode.
///
/// - Parameters:
///   - values: Leaf values
///   - n: Number of leaves (power of 2)
///   - mode: Execution mode (CPU, GPU-only, ANE-only, GPU+ANE)
/// - Returns: Merkle tree as array of digests
public func buildPoseidon2M31MerkleTree(
    _ values: [M31], count n: Int,
    mode: Poseidon2ExecutionMode = .gpuANE
) -> [M31Digest] {
    let result = poseidon2BatchBuildMerkleTrees([values], leafCount: n, mode: mode)
    return result.trees[0]
}

/// Legacy alias for backward compatibility
public func buildPoseidon2M31MerkleTreeGPU(_ values: [M31], count n: Int) -> [M31Digest] {
    return buildPoseidon2M31MerkleTree(values, count: n, mode: .gpuANE)
}

/// Pre-initialize ANE Poseidon2 GPU (call once before benchmarking)
public func anePoseidon2PreWarm() {
    _ = ane_poseidon2_init()
}

/// Check if ANE Poseidon2 GPU is available
public var anePoseidon2Available: Bool {
    return ane_poseidon2_gpu_available()
}

/// Debug: get pipeline status bitmask
public func anePoseidon2DebugPipelineStatus() -> Int {
    return Int(ane_poseidon2_debug_pipeline_status())
}
