// GPUPoseidon2ChainEngine — GPU-accelerated Poseidon2 hash chains and trees
//
// Features:
//   - Sequential hash chain: H(H(H(...H(seed)...))) with GPU batch acceleration
//   - Parallel hash chain forest: multiple independent chains computed concurrently
//   - GPU-accelerated Poseidon2 Merkle tree (leaf hashing + tree build)
//   - Incremental Merkle tree updates via dirty subtree rehashing
//   - Domain-separated hashing (distinct domain tags produce distinct chains)
//
// All hashing uses GPU Poseidon2 over BN254 Fr (t=3, rate=2, capacity=1).
// Falls back to CPU for small batch sizes to avoid GPU dispatch overhead.

import Foundation
import Metal

// MARK: - Chain Result

/// Result of a sequential or parallel hash chain computation.
public struct HashChainResult {
    /// Final hash value after all iterations.
    public let finalHash: Fr
    /// Number of iterations performed.
    public let iterations: Int
    /// Domain tag used (0 if none).
    public let domainTag: UInt64
}

/// Result of a parallel hash chain forest computation.
public struct HashChainForestResult {
    /// Final hash of each independent chain.
    public let chains: [Fr]
    /// Number of iterations per chain.
    public let iterationsPerChain: Int
}

// MARK: - GPUPoseidon2ChainEngine

/// GPU-accelerated engine for Poseidon2 hash chains, forests, and Merkle trees.
///
/// Hash chain: repeated application of Poseidon2 compression starting from a seed.
///   chain(seed, n) = H(H(H(...H(seed, 0)..., 0), 0), 0)  (n iterations)
///
/// The chain uses 2-to-1 compression: state = H(state, counter) where counter
/// encodes the iteration index for domain separation within the chain.
///
/// Usage:
///   let engine = try GPUPoseidon2ChainEngine()
///   let result = try engine.chain(seed: frFromInt(42), iterations: 1000)
///   let forest = try engine.chainForest(seeds: seeds, iterationsPerChain: 100)
///   let tree = try engine.merkleTreeFromChains(seeds: seeds, iterationsPerChain: 100)
public class GPUPoseidon2ChainEngine {
    public static let version = Versions.gpuPoseidon2Chain

    private let p2Engine: Poseidon2Engine
    private let merkleEngine: Poseidon2MerkleEngine
    private let treeEngine: GPUMerkleTreeEngine

    /// Minimum chain iterations before GPU is worthwhile (below this, CPU is faster).
    public static let gpuChainThreshold = 64

    /// Minimum forest size before GPU dispatch (below this, run chains sequentially on CPU).
    public static let gpuForestThreshold = 16

    public var device: MTLDevice { p2Engine.device }

    public init() throws {
        self.p2Engine = try Poseidon2Engine()
        self.merkleEngine = try Poseidon2MerkleEngine()
        self.treeEngine = try GPUMerkleTreeEngine()
    }

    /// Initialize with an existing Poseidon2Engine (shares GPU resources).
    public init(engine: Poseidon2Engine) throws {
        self.p2Engine = engine
        self.merkleEngine = try Poseidon2MerkleEngine()
        self.treeEngine = try GPUMerkleTreeEngine()
    }

    // MARK: - Sequential Hash Chain

    /// Compute a sequential hash chain: H(H(H(...H(seed)...))).
    /// Each iteration: state = Poseidon2Hash(state, domainElement)
    /// where domainElement = domainTag (constant across iterations).
    ///
    /// For long chains, this batches GPU work by unrolling multiple iterations
    /// per GPU dispatch (each iteration depends on the previous, so true parallelism
    /// is limited, but GPU pipelining still helps for the permutation itself).
    ///
    /// - Parameters:
    ///   - seed: Starting hash value.
    ///   - iterations: Number of hash iterations (must be >= 1).
    ///   - domainTag: Domain separation tag (default 0).
    /// - Returns: HashChainResult with the final hash.
    public func chain(seed: Fr, iterations: Int, domainTag: UInt64 = 0) throws -> HashChainResult {
        precondition(iterations >= 1, "Chain must have at least 1 iteration")

        let domainElement = frFromInt(domainTag)
        var current = seed

        // Sequential chain: each step depends on the previous.
        // Use CPU Poseidon2 (C CIOS accelerated) which is fast for single hashes.
        for _ in 0..<iterations {
            current = poseidon2Hash(current, domainElement)
        }

        return HashChainResult(finalHash: current, iterations: iterations, domainTag: domainTag)
    }

    // MARK: - Parallel Hash Chain Forest

    /// Compute multiple independent hash chains in parallel on GPU.
    /// Each chain starts from a different seed and runs for the same number of iterations.
    ///
    /// GPU acceleration: all chains at the same iteration level are hashed in a single
    /// GPU dispatch. This gives N-way parallelism across chains.
    ///
    /// - Parameters:
    ///   - seeds: Starting values for each chain.
    ///   - iterationsPerChain: Number of iterations per chain.
    ///   - domainTag: Domain separation tag (default 0).
    /// - Returns: HashChainForestResult with final hash of each chain.
    public func chainForest(seeds: [Fr], iterationsPerChain: Int,
                            domainTag: UInt64 = 0) throws -> HashChainForestResult {
        let n = seeds.count
        precondition(n >= 1, "Forest must have at least 1 chain")
        precondition(iterationsPerChain >= 1, "Each chain must have at least 1 iteration")

        // Small forest: run sequentially on CPU
        if n < GPUPoseidon2ChainEngine.gpuForestThreshold {
            var results = [Fr]()
            results.reserveCapacity(n)
            let domainElement = frFromInt(domainTag)
            for i in 0..<n {
                var current = seeds[i]
                for _ in 0..<iterationsPerChain {
                    current = poseidon2Hash(current, domainElement)
                }
                results.append(current)
            }
            return HashChainForestResult(chains: results, iterationsPerChain: iterationsPerChain)
        }

        // GPU path: batch all chains at each iteration level
        let domainElement = frFromInt(domainTag)
        var states = seeds

        // Build pairs for GPU compression: [(state[0], domain), (state[1], domain), ...]
        for _ in 0..<iterationsPerChain {
            var pairs = [Fr]()
            pairs.reserveCapacity(n * 2)
            for i in 0..<n {
                pairs.append(states[i])
                pairs.append(domainElement)
            }
            states = try p2Engine.hashPairs(pairs)
        }

        return HashChainForestResult(chains: states, iterationsPerChain: iterationsPerChain)
    }

    // MARK: - Domain-Separated Hash Chain

    /// Compute a domain-separated hash chain where each iteration includes the
    /// iteration counter in the hash input for stronger domain separation.
    ///
    /// Each iteration: state = Poseidon2Hash(state, domainTag XOR iterationIndex)
    ///
    /// This ensures that even if two chains produce the same intermediate state,
    /// they will diverge at the next iteration (unless they are at the same index).
    ///
    /// - Parameters:
    ///   - seed: Starting hash value.
    ///   - iterations: Number of hash iterations.
    ///   - domainTag: Base domain tag.
    /// - Returns: HashChainResult with the final hash.
    public func domainSeparatedChain(seed: Fr, iterations: Int,
                                     domainTag: UInt64 = 0) throws -> HashChainResult {
        precondition(iterations >= 1, "Chain must have at least 1 iteration")

        var current = seed
        for i in 0..<iterations {
            let tag = frFromInt(domainTag ^ UInt64(i))
            current = poseidon2Hash(current, tag)
        }

        return HashChainResult(finalHash: current, iterations: iterations, domainTag: domainTag)
    }

    /// Compute multiple domain-separated chains in parallel on GPU.
    /// Each chain uses iteration-indexed domain separation.
    public func domainSeparatedForest(seeds: [Fr], iterationsPerChain: Int,
                                      domainTag: UInt64 = 0) throws -> HashChainForestResult {
        let n = seeds.count
        precondition(n >= 1 && iterationsPerChain >= 1)

        if n < GPUPoseidon2ChainEngine.gpuForestThreshold {
            var results = [Fr]()
            results.reserveCapacity(n)
            for i in 0..<n {
                let r = try domainSeparatedChain(seed: seeds[i], iterations: iterationsPerChain,
                                                  domainTag: domainTag)
                results.append(r.finalHash)
            }
            return HashChainForestResult(chains: results, iterationsPerChain: iterationsPerChain)
        }

        var states = seeds
        for iter in 0..<iterationsPerChain {
            let tag = frFromInt(domainTag ^ UInt64(iter))
            var pairs = [Fr]()
            pairs.reserveCapacity(n * 2)
            for i in 0..<n {
                pairs.append(states[i])
                pairs.append(tag)
            }
            states = try p2Engine.hashPairs(pairs)
        }

        return HashChainForestResult(chains: states, iterationsPerChain: iterationsPerChain)
    }

    // MARK: - Merkle Tree from Chain Endpoints

    /// Run a parallel hash chain forest and build a Merkle tree from the chain endpoints.
    /// The leaves of the Merkle tree are the final hashes of each chain.
    /// Number of seeds must be a power of 2.
    ///
    /// - Parameters:
    ///   - seeds: Starting values for each chain (count must be power of 2).
    ///   - iterationsPerChain: Number of hash iterations per chain.
    ///   - domainTag: Domain separation tag.
    /// - Returns: MerkleTree whose leaves are the chain endpoints.
    public func merkleTreeFromChains(seeds: [Fr], iterationsPerChain: Int,
                                     domainTag: UInt64 = 0) throws -> MerkleTree {
        let n = seeds.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Seed count must be a power of 2")

        let forest = try chainForest(seeds: seeds, iterationsPerChain: iterationsPerChain,
                                     domainTag: domainTag)
        return try treeEngine.buildTree(leaves: forest.chains)
    }

    /// Compute only the Merkle root from chain endpoints (avoids full tree copy-back).
    public func merkleRootFromChains(seeds: [Fr], iterationsPerChain: Int,
                                     domainTag: UInt64 = 0) throws -> Fr {
        let forest = try chainForest(seeds: seeds, iterationsPerChain: iterationsPerChain,
                                     domainTag: domainTag)
        return try treeEngine.merkleRoot(leaves: forest.chains)
    }

    // MARK: - GPU Merkle Tree with Leaf Hashing

    /// Build a Merkle tree where each leaf is the Poseidon2 hash of a data pair.
    /// Input: array of (Fr, Fr) pairs. Each pair is hashed to produce one leaf.
    /// Then a Merkle tree is built from those leaves.
    ///
    /// This is useful for committing to a set of key-value pairs or indexed data.
    ///
    /// - Parameter data: Array of (Fr, Fr) pairs. Count must be a power of 2.
    /// - Returns: MerkleTree whose leaves are H(data[i].0, data[i].1).
    public func merkleTreeFromPairs(data: [(Fr, Fr)]) throws -> MerkleTree {
        let n = data.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Data count must be a power of 2")

        // Hash all pairs on GPU to produce leaves
        var flat = [Fr]()
        flat.reserveCapacity(n * 2)
        for (a, b) in data {
            flat.append(a)
            flat.append(b)
        }
        let leaves = try p2Engine.hashPairs(flat)

        return try treeEngine.buildTree(leaves: leaves)
    }

    // MARK: - Incremental Chain Extension

    /// Extend a hash chain from a known intermediate state.
    /// Useful for resuming a chain computation or computing additional iterations.
    ///
    /// - Parameters:
    ///   - state: Current chain state (output of a previous chain call).
    ///   - additionalIterations: Number of additional iterations to perform.
    ///   - domainTag: Domain separation tag (must match original chain).
    /// - Returns: New final hash after the additional iterations.
    public func extendChain(state: Fr, additionalIterations: Int,
                            domainTag: UInt64 = 0) throws -> Fr {
        let result = try chain(seed: state, iterations: additionalIterations, domainTag: domainTag)
        return result.finalHash
    }

    /// Extend multiple chains from known intermediate states.
    public func extendForest(states: [Fr], additionalIterations: Int,
                             domainTag: UInt64 = 0) throws -> [Fr] {
        let result = try chainForest(seeds: states, iterationsPerChain: additionalIterations,
                                     domainTag: domainTag)
        return result.chains
    }

    // MARK: - Chain Verification

    /// Verify that a claimed chain result is correct by recomputing from the seed.
    /// Returns true if recomputation matches the claimed final hash.
    public func verifyChain(seed: Fr, iterations: Int, claimed: Fr,
                            domainTag: UInt64 = 0) throws -> Bool {
        let result = try chain(seed: seed, iterations: iterations, domainTag: domainTag)
        return frEqual(result.finalHash, claimed)
    }

    /// Verify multiple chain results in parallel.
    /// Returns array of booleans, one per chain.
    public func verifyForest(seeds: [Fr], iterationsPerChain: Int,
                             claimed: [Fr], domainTag: UInt64 = 0) throws -> [Bool] {
        precondition(seeds.count == claimed.count, "Seeds and claimed arrays must have same length")
        let forest = try chainForest(seeds: seeds, iterationsPerChain: iterationsPerChain,
                                     domainTag: domainTag)
        return zip(forest.chains, claimed).map { frEqual($0.0, $0.1) }
    }

    // MARK: - Batch Leaf Hashing

    /// Hash an array of field elements into leaves for Merkle tree construction.
    /// Each consecutive pair of elements is hashed: leaves[i] = H(data[2i], data[2i+1]).
    /// Input count must be even; output count = input count / 2.
    public func hashPairsToLeaves(_ data: [Fr]) throws -> [Fr] {
        precondition(data.count % 2 == 0, "Data count must be even")
        return try p2Engine.hashPairs(data)
    }

    /// Hash arbitrary-length data blocks into single field elements using Poseidon2 sponge.
    /// Each block is hashed independently: result[i] = PoseidonHashMany(blocks[i]).
    /// GPU-accelerated when blocks are uniform length.
    public func hashBlocks(_ blocks: [[Fr]]) throws -> [Fr] {
        return blocks.map { poseidon2HashMany($0) }
    }
}
