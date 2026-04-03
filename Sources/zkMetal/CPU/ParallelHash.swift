// Parallel CPU batch hashing — trivially parallel per-element hashing
// Does NOT replace vanilla poseidon2Hash/keccak256/blake3 (those remain as references).

import Foundation

// MARK: - Parallel Poseidon2 batch hash

/// Hash n pairs of field elements in parallel using Poseidon2 (2-to-1 compression).
public func parallelPoseidon2Batch(_ pairs: [(Fr, Fr)]) -> [Fr] {
    var results = [Fr](repeating: Fr.zero, count: pairs.count)
    DispatchQueue.concurrentPerform(iterations: pairs.count) { i in
        results[i] = poseidon2Hash(pairs[i].0, pairs[i].1)
    }
    return results
}

// MARK: - Parallel Keccak-256 batch hash

/// Hash n inputs in parallel using Keccak-256.
public func parallelKeccak256Batch(_ inputs: [[UInt8]]) -> [[UInt8]] {
    var results = [[UInt8]](repeating: [], count: inputs.count)
    DispatchQueue.concurrentPerform(iterations: inputs.count) { i in
        results[i] = keccak256(inputs[i])
    }
    return results
}

// MARK: - Parallel Blake3 batch hash

/// Hash n inputs in parallel using Blake3.
public func parallelBlake3Batch(_ inputs: [[UInt8]]) -> [[UInt8]] {
    var results = [[UInt8]](repeating: [], count: inputs.count)
    DispatchQueue.concurrentPerform(iterations: inputs.count) { i in
        results[i] = blake3(inputs[i])
    }
    return results
}

// MARK: - Parallel Poseidon2 Merkle tree

/// Build a Poseidon2 Merkle tree in parallel, level by level.
/// Returns all nodes (leaves at indices n..2n-1, root at index 1, index 0 unused).
public func parallelPoseidon2Merkle(_ leaves: [Fr]) -> [Fr] {
    let n = leaves.count
    precondition(n > 0 && (n & (n - 1)) == 0, "leaf count must be power of 2")
    var tree = [Fr](repeating: Fr.zero, count: 2 * n)
    // Copy leaves
    for i in 0..<n {
        tree[n + i] = leaves[i]
    }
    // Build bottom-up, each level in parallel
    var levelSize = n / 2
    var offset = n / 2
    while levelSize >= 1 {
        DispatchQueue.concurrentPerform(iterations: levelSize) { i in
            let idx = offset + i
            tree[idx] = poseidon2Hash(tree[2 * idx], tree[2 * idx + 1])
        }
        levelSize /= 2
        offset /= 2
    }
    return tree
}

// MARK: - Parallel Keccak-256 Merkle tree

/// Build a Keccak-256 Merkle tree in parallel, level by level.
/// Leaves and nodes are 32-byte hashes. Returns flat array [unused, root, ...internal..., leaves].
public func parallelKeccak256Merkle(_ leaves: [[UInt8]]) -> [[UInt8]] {
    let n = leaves.count
    precondition(n > 0 && (n & (n - 1)) == 0, "leaf count must be power of 2")
    var tree = [[UInt8]](repeating: [UInt8](repeating: 0, count: 32), count: 2 * n)
    for i in 0..<n {
        tree[n + i] = leaves[i]
    }
    var levelSize = n / 2
    var offset = n / 2
    while levelSize >= 1 {
        DispatchQueue.concurrentPerform(iterations: levelSize) { i in
            let idx = offset + i
            tree[idx] = keccak256(tree[2 * idx] + tree[2 * idx + 1])
        }
        levelSize /= 2
        offset /= 2
    }
    return tree
}

// MARK: - Parallel Blake3 Merkle tree

/// Build a Blake3 Merkle tree in parallel, level by level.
public func parallelBlake3Merkle(_ leaves: [[UInt8]]) -> [[UInt8]] {
    let n = leaves.count
    precondition(n > 0 && (n & (n - 1)) == 0, "leaf count must be power of 2")
    var tree = [[UInt8]](repeating: [UInt8](repeating: 0, count: 32), count: 2 * n)
    for i in 0..<n {
        tree[n + i] = leaves[i]
    }
    var levelSize = n / 2
    var offset = n / 2
    while levelSize >= 1 {
        DispatchQueue.concurrentPerform(iterations: levelSize) { i in
            let idx = offset + i
            tree[idx] = blake3(tree[2 * idx] + tree[2 * idx + 1])
        }
        levelSize /= 2
        offset /= 2
    }
    return tree
}
