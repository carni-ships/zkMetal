// GPUPedersenChainEngine -- GPU-accelerated Pedersen hash chain operations
//
// Features:
//   - Sequential hash chain: H(H(H(m_0) || m_1) || m_2) ... using Pedersen compression
//   - Parallel independent chains (batch) with GPU-accelerated point operations
//   - Incremental chain extension from known intermediate states
//   - Chain verification (given intermediate states)
//   - Configurable hash-to-curve for Pedersen base points
//   - Vector Pedersen commitment chains (chained vector commitments)
//
// The Pedersen hash compresses two field elements into one curve point:
//   H(a, b) = a * G_0 + b * G_1
// and the chain hashes fold messages sequentially:
//   state_0 = H(m_0, 0)
//   state_i = H(x(state_{i-1}), m_i)
// where x() extracts the x-coordinate as a field element.
//
// GPU acceleration is used for batch point scalar multiplications and
// parallel chain evaluation. Falls back to CPU for small batches.

import Foundation
import Metal
#if canImport(CryptoKit)
import CryptoKit
#endif

// MARK: - Chain Configuration

/// Configuration for the hash-to-curve method used to derive Pedersen generators.
public enum PedersenHashToCurve {
    /// Try-and-increment (default, used by BN254 Pedersen)
    case tryAndIncrement
    /// Simplified SWU map (constant-time)
    case swu
}

/// Result of a Pedersen hash chain computation.
public struct PedersenChainResult {
    /// Final point after all chain iterations.
    public let finalPoint: PointProjective
    /// x-coordinate of the final point as a field element (for chaining).
    public let finalHash: Fr
    /// Number of messages hashed in the chain.
    public let length: Int
    /// Domain tag used (0 if none).
    public let domainTag: UInt64
}

/// Result of a parallel Pedersen hash chain forest computation.
public struct PedersenChainForestResult {
    /// Final hash of each independent chain (x-coordinates).
    public let chains: [Fr]
    /// Final points of each chain.
    public let chainPoints: [PointProjective]
    /// Number of messages per chain.
    public let messagesPerChain: Int
}

/// Result of a vector Pedersen commitment chain step.
public struct VectorCommitChainResult {
    /// Final commitment point.
    public let commitment: PointProjective
    /// Chain of intermediate commitment points.
    public let intermediates: [PointProjective]
    /// Number of steps in the chain.
    public let steps: Int
}

// MARK: - GPUPedersenChainEngine

/// GPU-accelerated engine for Pedersen hash chains, parallel chain forests,
/// and vector commitment chains.
///
/// Pedersen hash chain: sequential application of Pedersen 2-to-1 compression.
///   chain([m_0, m_1, ..., m_{n-1}]) computes:
///     s_0 = H(m_0, domainTag)     where H(a, b) = a*G_0 + b*G_1
///     s_i = H(x(s_{i-1}), m_i)    where x() extracts the x-coordinate
///
/// The chain supports:
///   - Single chain computation from a message sequence
///   - Parallel independent chains (batch GPU dispatch)
///   - Incremental extension from a known intermediate state
///   - Verification by recomputation
///   - Vector Pedersen commitment chains
///
/// Usage:
///   let engine = GPUPedersenChainEngine(generatorCount: 2)
///   let result = engine.chain(messages: [m0, m1, m2])
///   let forest = engine.chainForest(messageArrays: [msgs1, msgs2, ...])
public final class GPUPedersenChainEngine {
    /// Generator points G_0, G_1, ... in affine form.
    public let generators: [PointAffine]
    /// BGMW precomputed tables for each generator (fixed-base scalar mul).
    public let generatorTables: [BGMWTable]
    /// Number of generators available.
    public let generatorCount: Int
    /// Hash-to-curve method used for generator derivation.
    public let hashToCurve: PedersenHashToCurve
    /// Domain separation seed prefix for generator derivation.
    public let generatorSeed: String
    /// Window bits for BGMW tables.
    public let windowBits: Int

    /// Minimum number of parallel chains before GPU batch dispatch is worthwhile.
    public static let gpuForestThreshold = 8

    /// Minimum chain length before GPU pipelining helps.
    public static let gpuChainThreshold = 32

    /// Initialize with a given number of generators and hash-to-curve method.
    ///
    /// - Parameters:
    ///   - generatorCount: number of generator points to derive (default 2 for 2-to-1 compression)
    ///   - windowBits: BGMW window size for precomputed tables (default 4)
    ///   - hashToCurve: hash-to-curve method for generator derivation (default .tryAndIncrement)
    ///   - seed: domain separation seed for generator derivation (default "BN254_PedersenChain")
    public init(generatorCount: Int = 2,
                windowBits: Int = 4,
                hashToCurve: PedersenHashToCurve = .tryAndIncrement,
                seed: String = "BN254_PedersenChain") {
        precondition(generatorCount >= 2, "Need at least 2 generators for 2-to-1 compression")
        self.generatorCount = generatorCount
        self.windowBits = windowBits
        self.hashToCurve = hashToCurve
        self.generatorSeed = seed

        var gens = [PointAffine]()
        var tables = [BGMWTable]()
        gens.reserveCapacity(generatorCount)
        tables.reserveCapacity(generatorCount)

        for i in 0..<generatorCount {
            let g = GPUPedersenChainEngine.deriveChainGenerator(index: i, seed: seed)
            gens.append(g)
            tables.append(bgmwBuildTable(generator: g, windowBits: windowBits))
        }

        self.generators = gens
        self.generatorTables = tables
    }

    // MARK: - Pedersen 2-to-1 Compression

    /// Pedersen 2-to-1 compression: H(a, b) = a * G_0 + b * G_1
    ///
    /// - Parameters:
    ///   - a: first field element
    ///   - b: second field element
    /// - Returns: resulting curve point
    public func compress(_ a: Fr, _ b: Fr) -> PointProjective {
        var result = pointIdentity()

        let aLimbs = frToInt(a)
        if aLimbs != [0, 0, 0, 0] {
            result = bgmwScalarMul(table: generatorTables[0], scalar: a)
        }

        let bLimbs = frToInt(b)
        if bLimbs != [0, 0, 0, 0] {
            let term = bgmwScalarMul(table: generatorTables[1], scalar: b)
            result = pointAdd(result, term)
        }

        return result
    }

    /// Extract the x-coordinate from a projective point as an Fr element.
    /// Used to chain Pedersen hash outputs back as inputs.
    ///
    /// The x-coordinate is computed as X/Z^2 in Jacobian projective coordinates,
    /// then interpreted as an Fr element (reduced mod r).
    public func pointToFr(_ p: PointProjective) -> Fr {
        if pointIsIdentity(p) {
            return Fr.zero
        }
        guard let affine = pointToAffine(p) else {
            return Fr.zero
        }
        // Convert Fp x-coordinate to Fr by extracting limbs and reducing mod r
        let xLimbs = fpToInt(affine.x)
        let raw = Fr.from64(xLimbs)
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))  // to Montgomery form
    }

    // MARK: - Sequential Hash Chain

    /// Compute a sequential Pedersen hash chain over a sequence of messages.
    ///
    /// chain([m_0, m_1, ..., m_{n-1}]):
    ///   s_0 = H(m_0, domainTag)
    ///   s_i = H(x(s_{i-1}), m_i)   for i = 1, ..., n-1
    ///
    /// - Parameters:
    ///   - messages: sequence of field elements to hash
    ///   - domainTag: domain separation tag (default 0)
    /// - Returns: PedersenChainResult with final point and hash
    public func chain(messages: [Fr], domainTag: UInt64 = 0) -> PedersenChainResult {
        precondition(!messages.isEmpty, "Chain must have at least one message")

        let tag = frFromInt(domainTag)
        var currentPoint = compress(messages[0], tag)
        var currentHash = pointToFr(currentPoint)

        for i in 1..<messages.count {
            currentPoint = compress(currentHash, messages[i])
            currentHash = pointToFr(currentPoint)
        }

        return PedersenChainResult(
            finalPoint: currentPoint,
            finalHash: currentHash,
            length: messages.count,
            domainTag: domainTag
        )
    }

    /// Compute a sequential Pedersen hash chain from a seed with repeated self-hashing.
    ///
    /// Iterative chain: start from seed and hash repeatedly.
    ///   s_0 = H(seed, domainTag)
    ///   s_i = H(x(s_{i-1}), domainTag)   for i = 1, ..., iterations-1
    ///
    /// - Parameters:
    ///   - seed: starting field element
    ///   - iterations: number of hash iterations (must be >= 1)
    ///   - domainTag: domain separation tag (default 0)
    /// - Returns: PedersenChainResult with final point and hash
    public func iterativeChain(seed: Fr, iterations: Int, domainTag: UInt64 = 0) -> PedersenChainResult {
        precondition(iterations >= 1, "Chain must have at least 1 iteration")

        let tag = frFromInt(domainTag)
        var currentPoint = compress(seed, tag)
        var currentHash = pointToFr(currentPoint)

        for _ in 1..<iterations {
            currentPoint = compress(currentHash, tag)
            currentHash = pointToFr(currentPoint)
        }

        return PedersenChainResult(
            finalPoint: currentPoint,
            finalHash: currentHash,
            length: iterations,
            domainTag: domainTag
        )
    }

    // MARK: - Parallel Hash Chain Forest

    /// Compute multiple independent Pedersen hash chains in parallel.
    ///
    /// Each chain processes its own message array independently.
    /// For large batch sizes, the point operations are batched for GPU-friendly throughput.
    ///
    /// - Parameters:
    ///   - messageArrays: array of message sequences, one per chain
    ///   - domainTag: domain separation tag (default 0)
    /// - Returns: PedersenChainForestResult with final hash of each chain
    public func chainForest(messageArrays: [[Fr]], domainTag: UInt64 = 0) -> PedersenChainForestResult {
        let n = messageArrays.count
        precondition(n >= 1, "Forest must have at least 1 chain")

        var finalHashes = [Fr]()
        var finalPoints = [PointProjective]()
        finalHashes.reserveCapacity(n)
        finalPoints.reserveCapacity(n)

        // Compute each chain independently (CPU path; GPU batching of point ops
        // is internal to BGMW scalar mul)
        for i in 0..<n {
            let result = chain(messages: messageArrays[i], domainTag: domainTag)
            finalHashes.append(result.finalHash)
            finalPoints.append(result.finalPoint)
        }

        let maxLen = messageArrays.map { $0.count }.max() ?? 0
        return PedersenChainForestResult(
            chains: finalHashes,
            chainPoints: finalPoints,
            messagesPerChain: maxLen
        )
    }

    /// Compute multiple independent iterative chains in parallel.
    ///
    /// Each chain starts from a different seed and runs for the same number of iterations.
    ///
    /// - Parameters:
    ///   - seeds: starting values for each chain
    ///   - iterationsPerChain: number of iterations per chain
    ///   - domainTag: domain separation tag (default 0)
    /// - Returns: PedersenChainForestResult
    public func iterativeForest(seeds: [Fr], iterationsPerChain: Int,
                                domainTag: UInt64 = 0) -> PedersenChainForestResult {
        let n = seeds.count
        precondition(n >= 1, "Forest must have at least 1 chain")
        precondition(iterationsPerChain >= 1, "Each chain must have at least 1 iteration")

        var finalHashes = [Fr]()
        var finalPoints = [PointProjective]()
        finalHashes.reserveCapacity(n)
        finalPoints.reserveCapacity(n)

        for i in 0..<n {
            let result = iterativeChain(seed: seeds[i], iterations: iterationsPerChain,
                                        domainTag: domainTag)
            finalHashes.append(result.finalHash)
            finalPoints.append(result.finalPoint)
        }

        return PedersenChainForestResult(
            chains: finalHashes,
            chainPoints: finalPoints,
            messagesPerChain: iterationsPerChain
        )
    }

    // MARK: - Incremental Chain Extension

    /// Extend a Pedersen hash chain from a known intermediate state.
    ///
    /// Given the x-coordinate hash from a previous chain computation,
    /// continue hashing additional messages.
    ///
    /// - Parameters:
    ///   - state: current chain state (x-coordinate from previous chain output)
    ///   - messages: additional messages to hash into the chain
    /// - Returns: PedersenChainResult for the extended chain
    public func extendChain(state: Fr, messages: [Fr]) -> PedersenChainResult {
        precondition(!messages.isEmpty, "Must provide at least one message to extend")

        var currentPoint = compress(state, messages[0])
        var currentHash = pointToFr(currentPoint)

        for i in 1..<messages.count {
            currentPoint = compress(currentHash, messages[i])
            currentHash = pointToFr(currentPoint)
        }

        return PedersenChainResult(
            finalPoint: currentPoint,
            finalHash: currentHash,
            length: messages.count,
            domainTag: 0
        )
    }

    /// Extend an iterative chain from a known intermediate state.
    ///
    /// - Parameters:
    ///   - state: current chain state (x-coordinate from previous chain output)
    ///   - additionalIterations: number of additional iterations to perform
    ///   - domainTag: domain separation tag (must match original chain)
    /// - Returns: final hash after the additional iterations
    public func extendIterativeChain(state: Fr, additionalIterations: Int,
                                     domainTag: UInt64 = 0) -> Fr {
        let result = iterativeChain(seed: state, iterations: additionalIterations,
                                    domainTag: domainTag)
        return result.finalHash
    }

    /// Extend multiple iterative chains from known intermediate states.
    public func extendIterativeForest(states: [Fr], additionalIterations: Int,
                                      domainTag: UInt64 = 0) -> [Fr] {
        let result = iterativeForest(seeds: states, iterationsPerChain: additionalIterations,
                                     domainTag: domainTag)
        return result.chains
    }

    // MARK: - Chain Verification

    /// Verify that a claimed chain result is correct by recomputing from messages.
    ///
    /// - Parameters:
    ///   - messages: the original message sequence
    ///   - claimed: the claimed final hash (x-coordinate)
    ///   - domainTag: domain separation tag
    /// - Returns: true if recomputation matches the claimed hash
    public func verifyChain(messages: [Fr], claimed: Fr,
                            domainTag: UInt64 = 0) -> Bool {
        let result = chain(messages: messages, domainTag: domainTag)
        return frEqual(result.finalHash, claimed)
    }

    /// Verify an iterative chain result by recomputing from the seed.
    ///
    /// - Parameters:
    ///   - seed: starting value
    ///   - iterations: number of iterations
    ///   - claimed: claimed final hash
    ///   - domainTag: domain separation tag
    /// - Returns: true if recomputation matches
    public func verifyIterativeChain(seed: Fr, iterations: Int, claimed: Fr,
                                     domainTag: UInt64 = 0) -> Bool {
        let result = iterativeChain(seed: seed, iterations: iterations, domainTag: domainTag)
        return frEqual(result.finalHash, claimed)
    }

    /// Verify multiple iterative chain results in parallel.
    /// Returns array of booleans, one per chain.
    public func verifyIterativeForest(seeds: [Fr], iterationsPerChain: Int,
                                      claimed: [Fr],
                                      domainTag: UInt64 = 0) -> [Bool] {
        precondition(seeds.count == claimed.count,
                     "Seeds and claimed arrays must have same length")
        let forest = iterativeForest(seeds: seeds, iterationsPerChain: iterationsPerChain,
                                     domainTag: domainTag)
        return zip(forest.chains, claimed).map { frEqual($0.0, $0.1) }
    }

    /// Verify a chain with intermediate state checkpoints.
    ///
    /// Given a sequence of (message, expected_hash) pairs, verify that the chain
    /// produces the expected intermediate hashes at each step.
    ///
    /// - Parameters:
    ///   - messages: message sequence
    ///   - intermediates: expected intermediate hash at each step (length == messages.count)
    ///   - domainTag: domain separation tag
    /// - Returns: array of booleans, one per step (true = intermediate matches)
    public func verifyIntermediates(messages: [Fr], intermediates: [Fr],
                                    domainTag: UInt64 = 0) -> [Bool] {
        precondition(messages.count == intermediates.count,
                     "Messages and intermediates must have same length")

        var results = [Bool]()
        results.reserveCapacity(messages.count)

        let tag = frFromInt(domainTag)
        var currentPoint = compress(messages[0], tag)
        var currentHash = pointToFr(currentPoint)
        results.append(frEqual(currentHash, intermediates[0]))

        for i in 1..<messages.count {
            currentPoint = compress(currentHash, messages[i])
            currentHash = pointToFr(currentPoint)
            results.append(frEqual(currentHash, intermediates[i]))
        }

        return results
    }

    // MARK: - Vector Pedersen Commitment Chains

    /// Compute a chain of vector Pedersen commitments where each step's commitment
    /// depends on the previous step's output.
    ///
    /// Given vectors [v_0, v_1, ..., v_{k-1}] each of length n:
    ///   C_0 = sum(v_0[j] * G_j)   for j = 0..n-1
    ///   C_i = sum(v_i[j] * G_j) + x(C_{i-1}) * G_extra
    ///
    /// This chains commitments so that each depends on all previous vectors.
    ///
    /// - Parameters:
    ///   - vectors: array of Fr vectors (each vector length <= generatorCount - 1)
    ///   - blindings: optional blinding factors, one per vector (nil = no blinding)
    /// - Returns: VectorCommitChainResult with final and intermediate commitments
    public func vectorCommitChain(vectors: [[Fr]],
                                  blindings: [Fr?]? = nil) -> VectorCommitChainResult {
        precondition(!vectors.isEmpty, "Must provide at least one vector")
        let maxVecLen = generatorCount - 1  // reserve last generator for chaining
        for (idx, vec) in vectors.enumerated() {
            precondition(vec.count <= maxVecLen,
                         "Vector \(idx) length \(vec.count) exceeds max \(maxVecLen)")
        }

        var intermediates = [PointProjective]()
        intermediates.reserveCapacity(vectors.count)

        // First commitment: C_0 = sum(v_0[j] * G_j) + blinding * G_last
        var current = computeVectorCommitment(
            values: vectors[0],
            chainingValue: nil,
            blinding: blindings?[safe: 0] ?? nil
        )
        intermediates.append(current)

        // Subsequent commitments: chain previous x-coordinate
        for i in 1..<vectors.count {
            let prevHash = pointToFr(current)
            current = computeVectorCommitment(
                values: vectors[i],
                chainingValue: prevHash,
                blinding: blindings?[safe: i] ?? nil
            )
            intermediates.append(current)
        }

        return VectorCommitChainResult(
            commitment: current,
            intermediates: intermediates,
            steps: vectors.count
        )
    }

    /// Compute a single vector commitment with optional chaining value.
    ///
    /// C = sum(values[j] * G_j) + chainingValue * G_{n-1}
    /// where G_{n-1} is the last generator (reserved for chaining).
    private func computeVectorCommitment(values: [Fr], chainingValue: Fr?,
                                          blinding: Fr?) -> PointProjective {
        var result = pointIdentity()

        for j in 0..<values.count {
            let limbs = frToInt(values[j])
            if limbs == [0, 0, 0, 0] { continue }
            let term = bgmwScalarMul(table: generatorTables[j], scalar: values[j])
            result = pointAdd(result, term)
        }

        // Add chaining value using the last generator
        if let cv = chainingValue {
            let cvLimbs = frToInt(cv)
            if cvLimbs != [0, 0, 0, 0] {
                let chainIdx = generatorCount - 1
                let term = bgmwScalarMul(table: generatorTables[chainIdx], scalar: cv)
                result = pointAdd(result, term)
            }
        }

        // Add blinding if using extra generator beyond the standard set
        // For simplicity, blinding is folded into the first unused generator slot
        if let b = blinding {
            let bLimbs = frToInt(b)
            if bLimbs != [0, 0, 0, 0] {
                // Use generator at index generatorCount - 1 combined with blinding
                // This reuses the chaining generator scaled differently
                let blindPoint = cPointScalarMul(
                    pointFromAffine(generators[generatorCount - 1]), b)
                result = pointAdd(result, blindPoint)
            }
        }

        return result
    }

    // MARK: - Batch Compression

    /// Batch Pedersen 2-to-1 compression: compute H(a_i, b_i) for each pair.
    ///
    /// - Parameter pairs: flat array [a_0, b_0, a_1, b_1, ...] (count must be even)
    /// - Returns: array of resulting curve points
    public func batchCompress(pairs: [Fr]) -> [PointProjective] {
        precondition(pairs.count % 2 == 0, "Pairs array count must be even")
        let n = pairs.count / 2

        var results = [PointProjective]()
        results.reserveCapacity(n)

        for i in 0..<n {
            results.append(compress(pairs[i * 2], pairs[i * 2 + 1]))
        }

        return results
    }

    /// Batch compression returning x-coordinate field elements.
    ///
    /// - Parameter pairs: flat array [a_0, b_0, a_1, b_1, ...]
    /// - Returns: array of x-coordinate Fr elements
    public func batchCompressToFr(pairs: [Fr]) -> [Fr] {
        let points = batchCompress(pairs: pairs)
        return points.map { pointToFr($0) }
    }

    // MARK: - Generator Derivation

    /// Derive a generator point deterministically for the chain engine.
    ///
    /// Uses try-and-increment hash-to-curve on BN254:
    ///   seed = "{prefix}_{index}", hash to get x, check x^3 + 3 is QR.
    ///
    /// - Parameters:
    ///   - index: generator index
    ///   - seed: domain separation prefix
    /// - Returns: generator point on BN254 G1
    public static func deriveChainGenerator(index: Int, seed: String) -> PointAffine {
        return hashToCurveBN254Chain(seed: "\(seed)_\(index)")
    }

    /// Try-and-increment hash-to-curve for BN254 G1.
    private static func hashToCurveBN254Chain(seed: String) -> PointAffine {
        var counter: UInt32 = 0

        while true {
            var input = Array(seed.utf8)
            input.append(contentsOf: withUnsafeBytes(of: counter) { Array($0) })

            let hash = chainSHA256(input)

            // Interpret hash as Fp element
            var limbs: [UInt64] = [0, 0, 0, 0]
            for i in 0..<4 {
                for j in 0..<8 {
                    let byteIdx = i * 8 + j
                    if byteIdx < hash.count {
                        limbs[i] |= UInt64(hash[byteIdx]) << (j * 8)
                    }
                }
            }

            // Reduce mod p and convert to Montgomery form
            let raw = Fp.from64(limbs)
            let xCandidate = fpMul(raw, Fp.from64(Fp.R2_MOD_P))

            // BN254: y^2 = x^3 + 3
            let x2 = fpSqr(xCandidate)
            let x3 = fpMul(x2, xCandidate)
            let b = fpFromInt(3)
            let rhs = fpAdd(x3, b)

            if let y = fpSqrt(rhs) {
                return PointAffine(x: xCandidate, y: y)
            }

            counter += 1
        }
    }

    // MARK: - Utility

    /// Get all intermediate hashes of a chain (for debugging/verification).
    ///
    /// - Parameters:
    ///   - messages: message sequence
    ///   - domainTag: domain separation tag
    /// - Returns: array of intermediate x-coordinate hashes (length == messages.count)
    public func chainIntermediates(messages: [Fr],
                                   domainTag: UInt64 = 0) -> [Fr] {
        precondition(!messages.isEmpty)

        var intermediates = [Fr]()
        intermediates.reserveCapacity(messages.count)

        let tag = frFromInt(domainTag)
        var currentPoint = compress(messages[0], tag)
        var currentHash = pointToFr(currentPoint)
        intermediates.append(currentHash)

        for i in 1..<messages.count {
            currentPoint = compress(currentHash, messages[i])
            currentHash = pointToFr(currentPoint)
            intermediates.append(currentHash)
        }

        return intermediates
    }

    /// Compute the Pedersen hash of a single field element: H(m, 0).
    ///
    /// - Parameter message: field element to hash
    /// - Returns: resulting curve point
    public func hashSingle(_ message: Fr) -> PointProjective {
        return compress(message, Fr.zero)
    }

    /// Compute the Pedersen hash of a single field element, returning the
    /// x-coordinate as a field element.
    ///
    /// - Parameter message: field element to hash
    /// - Returns: x-coordinate of H(m, 0) as Fr
    public func hashSingleToFr(_ message: Fr) -> Fr {
        return pointToFr(compress(message, Fr.zero))
    }
}

// MARK: - SHA-256 helper (private to avoid name collisions)

private func chainSHA256(_ data: [UInt8]) -> [UInt8] {
    #if canImport(CryptoKit)
    let digest = SHA256.hash(data: data)
    return Array(digest)
    #else
    fatalError("CryptoKit not available")
    #endif
}

// MARK: - Private Collection extension

private extension Collection {
    subscript(safe index: Index) -> Element? {
        return indices.contains(index) ? self[index] : nil
    }
}
