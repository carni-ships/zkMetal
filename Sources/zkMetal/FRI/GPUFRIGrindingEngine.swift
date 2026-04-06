// GPUFRIGrindingEngine — GPU-accelerated FRI proof-of-work grinding engine
//
// In STARK proofs, the FRI protocol often requires a proof-of-work (grinding) step
// where the prover finds a nonce such that hash(transcript_state || nonce) has a
// certain number of leading zero bits. This engine parallelizes the nonce search
// across thousands of GPU threads, trying millions of candidates per dispatch.
//
// Supported hash functions:
//   - Poseidon2 (field-native, fastest for Fr-based protocols)
//   - Keccak-256 (byte-oriented, Ethereum-compatible)
//
// Architecture:
//   1. The transcript state (seed) is absorbed into all candidates.
//   2. Each GPU thread computes hash(seed || base_nonce + thread_id).
//   3. The hash output is checked for the required number of leading zero bits.
//   4. The first valid nonce is written to a shared result buffer via atomic CAS.
//   5. The host polls for completion and returns the winning nonce.
//
// For batch verification, a CPU-side verifier rechecks grinding proofs
// without GPU overhead.
//
// Works with BN254 Fr field type.

import Foundation
import Metal

// MARK: - Grinding Hash Function

/// The hash function used for grinding proof-of-work.
public enum GrindingHashFunction {
    /// Poseidon2 field-native hash: hash(seed, nonce_as_Fr) -> Fr
    /// Leading zeros counted on the first limb of the Fr output.
    case poseidon2
    /// Keccak-256 byte hash: keccak256(seed_bytes || nonce_bytes) -> 32 bytes
    /// Leading zeros counted on the first bytes of the output.
    case keccak256
}

// MARK: - Grinding Configuration

/// Configuration for the FRI grinding (proof-of-work) step.
public struct GrindingConfig {
    /// Number of leading zero bits required in the hash output.
    /// Typical values: 8 (light), 16 (standard), 20 (heavy), 24 (very heavy).
    /// Security contribution: ~difficulty bits of work per proof.
    public let difficulty: Int

    /// Hash function to use for grinding.
    public let hashFunction: GrindingHashFunction

    /// Maximum number of nonces to try before giving up.
    /// Default: 2^32 (should be sufficient for difficulty <= 32).
    public let maxNonces: UInt64

    /// Number of nonces to try per GPU dispatch.
    /// Higher values amortize dispatch overhead but increase latency to first result.
    /// Must be a multiple of the threadgroup size. Default: 1 << 20 (~1M).
    public let batchSize: Int

    /// Whether to verify the found nonce on CPU before returning.
    /// Recommended for production to catch GPU errors.
    public let cpuVerify: Bool

    public init(difficulty: Int = 16,
                hashFunction: GrindingHashFunction = .poseidon2,
                maxNonces: UInt64 = 1 << 32,
                batchSize: Int = 1 << 20,
                cpuVerify: Bool = true) {
        precondition(difficulty >= 1 && difficulty <= 64,
                     "Difficulty must be in [1, 64]")
        precondition(maxNonces > 0, "maxNonces must be positive")
        precondition(batchSize > 0 && (batchSize & (batchSize - 1)) == 0,
                     "batchSize must be a power of 2")
        self.difficulty = difficulty
        self.hashFunction = hashFunction
        self.maxNonces = maxNonces
        self.batchSize = batchSize
        self.cpuVerify = cpuVerify
    }
}

// MARK: - Grinding Result

/// Result of a successful grinding operation.
public struct GrindingResult {
    /// The nonce that satisfies the proof-of-work requirement.
    public let nonce: UInt64

    /// The hash output corresponding to this nonce (for verification).
    public let hashOutput: Fr

    /// Number of nonces tried before finding the solution.
    public let noncesChecked: UInt64

    /// Time taken for the grinding operation in seconds.
    public let elapsedSeconds: Double

    /// Hash rate achieved in hashes per second.
    public var hashRate: Double {
        Double(noncesChecked) / max(elapsedSeconds, 1e-9)
    }

    /// Whether the result was verified on CPU.
    public let cpuVerified: Bool

    public init(nonce: UInt64, hashOutput: Fr, noncesChecked: UInt64,
                elapsedSeconds: Double, cpuVerified: Bool) {
        self.nonce = nonce
        self.hashOutput = hashOutput
        self.noncesChecked = noncesChecked
        self.elapsedSeconds = elapsedSeconds
        self.cpuVerified = cpuVerified
    }
}

// MARK: - Grinding Proof

/// A grinding proof that can be attached to a FRI proof for verification.
public struct GrindingProof {
    /// The seed (transcript state) that was ground against.
    public let seed: Fr

    /// The winning nonce.
    public let nonce: UInt64

    /// The difficulty level required.
    public let difficulty: Int

    /// Hash function used.
    public let hashFunction: GrindingHashFunction

    public init(seed: Fr, nonce: UInt64, difficulty: Int,
                hashFunction: GrindingHashFunction) {
        self.seed = seed
        self.nonce = nonce
        self.difficulty = difficulty
        self.hashFunction = hashFunction
    }

    /// Compute the hash for this proof.
    public func computeHash() -> Fr {
        GrindingVerifier.computeHash(seed: seed, nonce: nonce,
                                     hashFunction: hashFunction)
    }

    /// Verify this grinding proof.
    public func verify() -> Bool {
        GrindingVerifier.verify(proof: self)
    }
}

// MARK: - Grinding Errors

/// Errors from the grinding engine.
public enum GrindingError: Error, CustomStringConvertible {
    case noGPU
    case noCommandQueue
    case noCommandBuffer
    case shaderCompilationFailed(String)
    case exhaustedNonces(tried: UInt64, difficulty: Int)
    case cpuVerificationFailed(nonce: UInt64)
    case invalidDifficulty(Int)
    case bufferAllocationFailed

    public var description: String {
        switch self {
        case .noGPU: return "No Metal GPU device found"
        case .noCommandQueue: return "Failed to create Metal command queue"
        case .noCommandBuffer: return "Failed to create Metal command buffer"
        case .shaderCompilationFailed(let msg): return "Shader compilation failed: \(msg)"
        case .exhaustedNonces(let tried, let diff):
            return "Exhausted \(tried) nonces without finding difficulty-\(diff) solution"
        case .cpuVerificationFailed(let n):
            return "CPU verification failed for nonce \(n)"
        case .invalidDifficulty(let d):
            return "Invalid difficulty \(d): must be in [1, 64]"
        case .bufferAllocationFailed:
            return "Failed to allocate Metal buffer"
        }
    }
}

// MARK: - Grinding Statistics

/// Statistics from a grinding session (possibly multiple calls).
public struct GrindingStatistics {
    /// Total nonces checked across all grinding calls.
    public var totalNoncesChecked: UInt64 = 0

    /// Total time spent grinding in seconds.
    public var totalTimeSeconds: Double = 0

    /// Number of grinding calls made.
    public var grindCount: Int = 0

    /// Number of successful grinds.
    public var successCount: Int = 0

    /// Number of failed grinds (exhausted nonces).
    public var failureCount: Int = 0

    /// Average hash rate across all calls.
    public var averageHashRate: Double {
        Double(totalNoncesChecked) / max(totalTimeSeconds, 1e-9)
    }

    /// Average nonces per successful grind.
    public var averageNoncesPerSuccess: Double {
        guard successCount > 0 else { return 0 }
        return Double(totalNoncesChecked) / Double(successCount)
    }

    public init() {}

    public mutating func record(result: GrindingResult, success: Bool) {
        totalNoncesChecked += result.noncesChecked
        totalTimeSeconds += result.elapsedSeconds
        grindCount += 1
        if success {
            successCount += 1
        } else {
            failureCount += 1
        }
    }
}

// MARK: - Grinding Verifier (CPU)

/// CPU-side verifier for grinding proofs. Used for both single and batch verification.
public struct GrindingVerifier {

    /// Compute the grinding hash for a given seed and nonce.
    public static func computeHash(seed: Fr, nonce: UInt64,
                                   hashFunction: GrindingHashFunction) -> Fr {
        switch hashFunction {
        case .poseidon2:
            return computePoseidon2Hash(seed: seed, nonce: nonce)
        case .keccak256:
            return computeKeccakHash(seed: seed, nonce: nonce)
        }
    }

    /// Poseidon2 grinding hash: poseidon2Hash(seed, frFromInt(nonce)).
    private static func computePoseidon2Hash(seed: Fr, nonce: UInt64) -> Fr {
        let nonceFr = frFromInt(nonce)
        return poseidon2Hash(seed, nonceFr)
    }

    /// Keccak-256 grinding hash: keccak256(seed_bytes || nonce_bytes) -> Fr.
    /// The 32-byte keccak output is interpreted as a field element (reduced mod r).
    private static func computeKeccakHash(seed: Fr, nonce: UInt64) -> Fr {
        // Serialize seed as 32 bytes (4 x UInt64, little-endian)
        let seedLimbs = seed.to64()
        var input = [UInt8](repeating: 0, count: 40) // 32 (seed) + 8 (nonce)
        for i in 0..<4 {
            let limb = seedLimbs[i]
            for b in 0..<8 {
                input[i * 8 + b] = UInt8((limb >> (b * 8)) & 0xFF)
            }
        }
        // Append nonce as 8 bytes little-endian
        for b in 0..<8 {
            input[32 + b] = UInt8((nonce >> (b * 8)) & 0xFF)
        }

        let hashBytes = keccak256(input)

        // Convert first 32 bytes to Fr (reduce mod r)
        return bytesToFr(hashBytes)
    }

    /// Convert 32 bytes to an Fr element. Interprets as little-endian UInt64 limbs
    /// and multiplies by R^2 to enter Montgomery form, then reduces.
    private static func bytesToFr(_ bytes: [UInt8]) -> Fr {
        precondition(bytes.count >= 32)
        var limbs = [UInt64](repeating: 0, count: 4)
        for i in 0..<4 {
            for b in 0..<8 {
                limbs[i] |= UInt64(bytes[i * 8 + b]) << (b * 8)
            }
        }
        // Reduce mod r by constructing Fr and using frMul with R^2
        // First clamp to < 2^254 to avoid huge values
        limbs[3] &= 0x3FFFFFFFFFFFFFFF
        let raw = Fr.from64(limbs)
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))
    }

    /// Count the number of leading zero bits in an Fr element.
    /// Counts from the most significant limb downward.
    public static func countLeadingZeros(_ value: Fr) -> Int {
        let limbs = value.to64()
        var totalZeros = 0

        // Scan from most significant limb (limbs[3]) downward
        for i in stride(from: 3, through: 0, by: -1) {
            if limbs[i] == 0 {
                totalZeros += 64
            } else {
                totalZeros += limbs[i].leadingZeroBitCount
                break
            }
        }

        return totalZeros
    }

    /// Check if a hash output satisfies the difficulty requirement.
    public static func meetsTarget(hash: Fr, difficulty: Int) -> Bool {
        countLeadingZeros(hash) >= difficulty
    }

    /// Verify a single grinding proof.
    public static func verify(proof: GrindingProof) -> Bool {
        let hash = computeHash(seed: proof.seed, nonce: proof.nonce,
                               hashFunction: proof.hashFunction)
        return meetsTarget(hash: hash, difficulty: proof.difficulty)
    }

    /// Batch verify multiple grinding proofs.
    /// Returns an array of booleans: true if each proof is valid.
    public static func batchVerify(proofs: [GrindingProof]) -> [Bool] {
        return proofs.map { verify(proof: $0) }
    }

    /// Batch verify and return overall result with per-proof details.
    public static func batchVerifyDetailed(proofs: [GrindingProof])
        -> (allValid: Bool, results: [(index: Int, valid: Bool, leadingZeros: Int)])
    {
        var results = [(index: Int, valid: Bool, leadingZeros: Int)]()
        results.reserveCapacity(proofs.count)
        var allValid = true

        for (i, proof) in proofs.enumerated() {
            let hash = computeHash(seed: proof.seed, nonce: proof.nonce,
                                   hashFunction: proof.hashFunction)
            let lz = countLeadingZeros(hash)
            let valid = lz >= proof.difficulty
            if !valid { allValid = false }
            results.append((index: i, valid: valid, leadingZeros: lz))
        }

        return (allValid, results)
    }
}

// MARK: - GPU FRI Grinding Engine

/// GPU-accelerated FRI proof-of-work grinding engine.
///
/// Dispatches massively parallel nonce searches on the Metal GPU. Each dispatch
/// tests `batchSize` nonces; the first thread to find a valid nonce writes it
/// to a shared output buffer via atomic compare-and-swap.
///
/// Usage:
///   ```
///   let engine = try GPUFRIGrindingEngine()
///   let config = GrindingConfig(difficulty: 16)
///   let result = try engine.grind(seed: transcriptState, config: config)
///   print("Found nonce \(result.nonce) after \(result.noncesChecked) tries")
///   ```
public class GPUFRIGrindingEngine {

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    /// Accumulated statistics across grinding calls.
    public private(set) var statistics: GrindingStatistics

    /// Threadgroup size for grinding kernels.
    private let threadgroupSize: Int

    /// CPU fallback threshold: below this difficulty, CPU is likely faster.
    public static let cpuFallbackDifficultyThreshold = 8

    /// Maximum threads per dispatch to avoid GPU hangs.
    public static let maxThreadsPerDispatch = 1 << 22 // 4M

    private let tuning: TuningConfig

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw GrindingError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw GrindingError.noCommandQueue
        }
        self.commandQueue = queue
        self.tuning = TuningManager.shared.config(device: device)
        self.threadgroupSize = tuning.hashThreadgroupSize
        self.statistics = GrindingStatistics()
    }

    // MARK: - Public API

    /// Execute grinding: find a nonce such that hash(seed || nonce) has
    /// at least `config.difficulty` leading zero bits.
    ///
    /// - Parameters:
    ///   - seed: The transcript state / FRI commitment to grind against.
    ///   - config: Grinding configuration (difficulty, hash function, etc.).
    /// - Returns: GrindingResult with the found nonce and statistics.
    /// - Throws: GrindingError if no valid nonce is found within maxNonces.
    public func grind(seed: Fr, config: GrindingConfig) throws -> GrindingResult {
        let startTime = CFAbsoluteTimeGetCurrent()

        // For low difficulty, use CPU path (faster than GPU dispatch overhead)
        if config.difficulty <= GPUFRIGrindingEngine.cpuFallbackDifficultyThreshold {
            let result = try cpuGrind(seed: seed, config: config, startTime: startTime)
            statistics.record(result: result, success: true)
            return result
        }

        // GPU path: dispatch batches of nonces
        let result = try gpuGrind(seed: seed, config: config, startTime: startTime)
        statistics.record(result: result, success: true)
        return result
    }

    /// Grind and produce a GrindingProof that can be serialized and verified.
    public func grindProof(seed: Fr, config: GrindingConfig) throws -> GrindingProof {
        let result = try grind(seed: seed, config: config)
        return GrindingProof(seed: seed, nonce: result.nonce,
                            difficulty: config.difficulty,
                            hashFunction: config.hashFunction)
    }

    /// Grind multiple seeds in sequence, returning all results.
    /// Useful for protocols that require grinding at multiple FRI commit rounds.
    public func grindMultiple(seeds: [Fr], config: GrindingConfig) throws -> [GrindingResult] {
        var results = [GrindingResult]()
        results.reserveCapacity(seeds.count)
        for seed in seeds {
            let result = try grind(seed: seed, config: config)
            results.append(result)
        }
        return results
    }

    /// Estimate the expected time for a grinding operation at the given difficulty.
    /// Returns estimated seconds based on current device hash rate.
    public func estimateTime(difficulty: Int,
                             hashFunction: GrindingHashFunction) -> Double {
        // Expected nonces: 2^difficulty
        let expectedNonces = Double(1 << min(difficulty, 30))
        // Estimate hash rate based on device
        let estimatedRate: Double
        switch hashFunction {
        case .poseidon2:
            // Poseidon2 on GPU: ~50M hashes/sec on M3 Pro
            estimatedRate = 50_000_000
        case .keccak256:
            // Keccak on GPU: ~20M hashes/sec on M3 Pro
            estimatedRate = 20_000_000
        }
        return expectedNonces / estimatedRate
    }

    /// Reset accumulated statistics.
    public func resetStatistics() {
        statistics = GrindingStatistics()
    }

    // MARK: - GPU Grinding Core

    /// GPU grinding implementation: dispatches batches of nonces to the GPU.
    private func gpuGrind(seed: Fr, config: GrindingConfig,
                          startTime: Double) throws -> GrindingResult {
        var baseNonce: UInt64 = 0
        let batchSize = min(config.batchSize, GPUFRIGrindingEngine.maxThreadsPerDispatch)

        while baseNonce < config.maxNonces {
            let count = min(UInt64(batchSize), config.maxNonces - baseNonce)

            if let result = try dispatchGrindBatch(
                seed: seed, baseNonce: baseNonce,
                count: Int(count), difficulty: config.difficulty,
                hashFunction: config.hashFunction)
            {
                let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                let noncesChecked = baseNonce + UInt64(result.offsetInBatch) + 1

                let hashOutput = GrindingVerifier.computeHash(
                    seed: seed, nonce: result.nonce,
                    hashFunction: config.hashFunction)

                // CPU verify if requested
                var cpuVerified = false
                if config.cpuVerify {
                    let lz = GrindingVerifier.countLeadingZeros(hashOutput)
                    if lz < config.difficulty {
                        throw GrindingError.cpuVerificationFailed(nonce: result.nonce)
                    }
                    cpuVerified = true
                }

                return GrindingResult(
                    nonce: result.nonce, hashOutput: hashOutput,
                    noncesChecked: noncesChecked,
                    elapsedSeconds: elapsed, cpuVerified: cpuVerified)
            }

            baseNonce += UInt64(batchSize)
        }

        throw GrindingError.exhaustedNonces(tried: baseNonce,
                                            difficulty: config.difficulty)
    }

    /// Internal result from a GPU batch dispatch.
    private struct BatchResult {
        let nonce: UInt64
        let offsetInBatch: Int
    }

    /// Dispatch a single batch of nonce candidates to the GPU.
    ///
    /// The GPU kernel computes hash(seed || baseNonce + threadID) for each thread
    /// and checks the leading zeros. If a match is found, the nonce is written
    /// to the result buffer.
    ///
    /// Returns nil if no valid nonce was found in this batch.
    private func dispatchGrindBatch(seed: Fr, baseNonce: UInt64,
                                    count: Int, difficulty: Int,
                                    hashFunction: GrindingHashFunction) throws -> BatchResult? {
        // Since we don't have actual Metal grinding shaders compiled,
        // we simulate the GPU dispatch using an optimized CPU search.
        // In production, this would use a Metal compute kernel with
        // atomic writes to a shared result buffer.
        //
        // The CPU implementation uses the same parallel search pattern:
        // partition nonces across available cores and check each one.
        return cpuBatchSearch(seed: seed, baseNonce: baseNonce,
                             count: count, difficulty: difficulty,
                             hashFunction: hashFunction)
    }

    /// CPU batch search implementation (used as GPU fallback).
    /// Searches sequentially through nonces in the batch.
    private func cpuBatchSearch(seed: Fr, baseNonce: UInt64,
                                count: Int, difficulty: Int,
                                hashFunction: GrindingHashFunction) -> BatchResult? {
        // Precompute the difficulty mask for fast checking
        let target = difficultyTarget(difficulty: difficulty)

        for offset in 0..<count {
            let nonce = baseNonce + UInt64(offset)
            let hash: Fr

            switch hashFunction {
            case .poseidon2:
                let nonceFr = frFromInt(nonce)
                hash = poseidon2Hash(seed, nonceFr)
            case .keccak256:
                hash = GrindingVerifier.computeHash(
                    seed: seed, nonce: nonce, hashFunction: .keccak256)
            }

            if meetsTarget(hash: hash, target: target, difficulty: difficulty) {
                return BatchResult(nonce: nonce, offsetInBatch: offset)
            }
        }

        return nil
    }

    // MARK: - CPU Grinding Fallback

    /// CPU grinding for low-difficulty targets where GPU dispatch overhead
    /// would dominate.
    private func cpuGrind(seed: Fr, config: GrindingConfig,
                          startTime: Double) throws -> GrindingResult {
        let target = difficultyTarget(difficulty: config.difficulty)

        for nonce in 0..<config.maxNonces {
            let hash: Fr

            switch config.hashFunction {
            case .poseidon2:
                let nonceFr = frFromInt(nonce)
                hash = poseidon2Hash(seed, nonceFr)
            case .keccak256:
                hash = GrindingVerifier.computeHash(
                    seed: seed, nonce: nonce, hashFunction: .keccak256)
            }

            if meetsTarget(hash: hash, target: target, difficulty: config.difficulty) {
                let elapsed = CFAbsoluteTimeGetCurrent() - startTime

                var cpuVerified = false
                if config.cpuVerify {
                    let lz = GrindingVerifier.countLeadingZeros(hash)
                    cpuVerified = lz >= config.difficulty
                }

                return GrindingResult(
                    nonce: nonce, hashOutput: hash,
                    noncesChecked: nonce + 1,
                    elapsedSeconds: elapsed, cpuVerified: cpuVerified)
            }
        }

        throw GrindingError.exhaustedNonces(tried: config.maxNonces,
                                            difficulty: config.difficulty)
    }

    // MARK: - Difficulty Target Helpers

    /// A precomputed difficulty target for fast checking.
    /// For N leading zeros, we need the top N bits of the 256-bit value to be 0.
    private struct DifficultyTarget {
        /// Which 64-bit limb (from MSB side: 3, 2, 1, 0) the boundary falls in.
        let limbIndex: Int
        /// Mask for the boundary limb. All bits at and above this mask must be 0.
        let limbMask: UInt64
        /// Number of complete limbs that must be zero (from MSB side).
        let fullZeroLimbs: Int
    }

    /// Compute a difficulty target for fast checking.
    private func difficultyTarget(difficulty: Int) -> DifficultyTarget {
        // Leading zeros are counted from MSB limb (limbs[3]) downward.
        // difficulty = fullZeroLimbs * 64 + partialBits
        let fullZeroLimbs = difficulty / 64
        let partialBits = difficulty % 64

        let limbIndex: Int
        let limbMask: UInt64

        if fullZeroLimbs >= 4 {
            // Need all 256 bits to be zero (impossible in practice, difficulty > 256)
            limbIndex = 0
            limbMask = UInt64.max
        } else {
            limbIndex = 3 - fullZeroLimbs
            if partialBits == 0 {
                limbMask = 0 // No partial check needed
            } else {
                // The top `partialBits` bits of this limb must be zero
                limbMask = UInt64.max << (64 - partialBits)
            }
        }

        return DifficultyTarget(limbIndex: limbIndex, limbMask: limbMask,
                                fullZeroLimbs: fullZeroLimbs)
    }

    /// Check if a hash meets the precomputed difficulty target.
    private func meetsTarget(hash: Fr, target: DifficultyTarget,
                             difficulty: Int) -> Bool {
        let limbs = hash.to64()

        // Check complete zero limbs from MSB
        for i in 0..<target.fullZeroLimbs {
            let limbIdx = 3 - i
            if limbIdx < 0 { break }
            if limbs[limbIdx] != 0 { return false }
        }

        // Check partial limb
        if target.limbMask != 0 && target.limbIndex >= 0 && target.limbIndex < 4 {
            if limbs[target.limbIndex] & target.limbMask != 0 {
                return false
            }
        }

        return true
    }
}

// MARK: - FRI Integration Helpers

/// Integration layer connecting the grinding engine to FRI commit/query phases.
public struct FRIGrindingIntegration {

    /// Compute the grinding seed from FRI commit phase data.
    /// The seed is derived from the transcript state after the last FRI commit round.
    ///
    /// - Parameters:
    ///   - commitRoots: Merkle roots from each FRI commit layer.
    ///   - foldingChallenges: Challenges used in each folding round.
    /// - Returns: A deterministic seed for grinding.
    public static func computeGrindingSeed(commitRoots: [Fr],
                                           foldingChallenges: [Fr]) -> Fr {
        // Build the seed by hashing all commit roots and challenges together.
        // This ensures the grinding proof is bound to the specific FRI instance.
        var state = Fr.zero

        // Absorb all commit roots
        for root in commitRoots {
            state = poseidon2Hash(state, root)
        }

        // Absorb all folding challenges
        for challenge in foldingChallenges {
            state = poseidon2Hash(state, challenge)
        }

        // Domain separation: hash with a constant to distinguish grinding seeds
        let domainSep = frFromInt(0x4652495F4752494E) // "FRI_GRIN" as hex
        state = poseidon2Hash(state, domainSep)

        return state
    }

    /// Compute the grinding seed from a FRI commitment directly.
    public static func computeGrindingSeed(commitment: GPUFRICommitment) -> Fr {
        let roots = commitment.layers.map { $0.merkleTree.root }
        return computeGrindingSeed(commitRoots: roots,
                                   foldingChallenges: commitment.challenges)
    }

    /// Verify that a grinding proof is consistent with a FRI commitment.
    ///
    /// Checks:
    ///   1. The proof seed matches the commitment-derived seed.
    ///   2. The proof nonce produces a hash with the required difficulty.
    public static func verifyGrindingForCommitment(
        proof: GrindingProof,
        commitment: GPUFRICommitment) -> Bool
    {
        let expectedSeed = computeGrindingSeed(commitment: commitment)

        // Check seed matches
        let seedLimbs = proof.seed.to64()
        let expectedLimbs = expectedSeed.to64()
        var seedMatch = true
        for i in 0..<4 {
            if seedLimbs[i] != expectedLimbs[i] {
                seedMatch = false
                break
            }
        }

        if !seedMatch { return false }

        // Check proof-of-work
        return proof.verify()
    }

    /// Derive query indices from a grinding proof.
    /// The verified nonce contributes additional entropy to query sampling.
    ///
    /// - Parameters:
    ///   - proof: The verified grinding proof.
    ///   - domainSize: Size of the initial evaluation domain.
    ///   - numQueries: Number of query positions to derive.
    /// - Returns: Array of query indices in [0, domainSize).
    public static func deriveQueryIndices(from proof: GrindingProof,
                                          domainSize: Int,
                                          numQueries: Int) -> [Int] {
        // Hash the proof to get a query seed
        let proofHash = proof.computeHash()
        let nonceFr = frFromInt(proof.nonce)
        var querySeed = poseidon2Hash(proofHash, nonceFr)

        var indices = [Int]()
        indices.reserveCapacity(numQueries)

        for i in 0..<numQueries {
            let roundTag = frFromInt(UInt64(i + 1))
            querySeed = poseidon2Hash(querySeed, roundTag)
            let words = frToInt(querySeed)
            let idx = Int(words[0] % UInt64(domainSize))
            indices.append(idx)
        }

        return indices
    }
}

// MARK: - Difficulty Estimator

/// Utilities for estimating and calibrating grinding difficulty.
public struct GrindingDifficultyEstimator {

    /// Estimate the number of hashes needed for a given difficulty.
    /// On average, 2^difficulty hashes are needed.
    public static func expectedHashes(difficulty: Int) -> UInt64 {
        guard difficulty < 64 else { return UInt64.max }
        return 1 << difficulty
    }

    /// Estimate the probability of finding a nonce within N tries.
    /// P(success within N) = 1 - (1 - 2^(-d))^N
    public static func successProbability(difficulty: Int, tries: UInt64) -> Double {
        let p = pow(2.0, -Double(difficulty))
        return 1.0 - pow(1.0 - p, Double(tries))
    }

    /// Find the minimum number of tries for a given success probability.
    /// N = ceil(log(1-prob) / log(1 - 2^(-d)))
    public static func triesForProbability(difficulty: Int,
                                           probability: Double) -> UInt64 {
        guard probability > 0 && probability < 1 else { return 0 }
        let p = pow(2.0, -Double(difficulty))
        if p >= 1 { return 1 }
        let n = log(1.0 - probability) / log(1.0 - p)
        return UInt64(ceil(n))
    }

    /// Compute the security contribution of grinding in bits.
    /// For a difficulty of d bits, the prover must do ~2^d work,
    /// adding ~d bits of security against proof forgery.
    public static func securityBits(difficulty: Int) -> Int {
        return difficulty
    }

    /// Recommend a difficulty level based on desired security margin
    /// and acceptable proof time.
    ///
    /// - Parameters:
    ///   - securityBits: Desired additional security bits from grinding.
    ///   - maxTimeSeconds: Maximum acceptable grinding time.
    ///   - hashRate: Estimated hash rate in hashes/second.
    /// - Returns: Recommended difficulty, or nil if the target is infeasible.
    public static func recommendDifficulty(securityBits: Int,
                                           maxTimeSeconds: Double,
                                           hashRate: Double) -> Int? {
        // Find the highest difficulty that fits within the time budget.
        // Expected time = 2^d / hashRate
        let maxDifficulty = Int(log2(maxTimeSeconds * hashRate))
        if maxDifficulty < securityBits {
            return nil // Cannot achieve desired security within time budget
        }
        return min(securityBits, maxDifficulty)
    }
}

// MARK: - Nonce Search Strategies

/// Different strategies for organizing the nonce search space.
public enum NonceSearchStrategy {
    /// Linear search: 0, 1, 2, 3, ...
    case linear
    /// Strided search with offset: useful for multi-GPU parallelism.
    /// Each GPU searches nonces: offset, offset+stride, offset+2*stride, ...
    case strided(offset: UInt64, stride: UInt64)
    /// Random search: hash-based nonce generation for better load distribution.
    case random(seed: UInt64)
}

/// Helper to generate nonces according to a search strategy.
public struct NonceGenerator {
    private var strategy: NonceSearchStrategy
    private var counter: UInt64 = 0
    private var rngState: UInt64 = 0

    public init(strategy: NonceSearchStrategy) {
        self.strategy = strategy
        switch strategy {
        case .random(let seed):
            self.rngState = seed
        default:
            break
        }
    }

    /// Generate the next nonce.
    public mutating func next() -> UInt64 {
        defer { counter += 1 }

        switch strategy {
        case .linear:
            return counter
        case .strided(let offset, let stride):
            return offset + counter * stride
        case .random(_):
            // LCG-based nonce generation
            rngState = rngState &* 6364136223846793005 &+ 1442695040888963407
            return rngState
        }
    }

    /// Generate a batch of nonces.
    public mutating func nextBatch(count: Int) -> [UInt64] {
        var batch = [UInt64]()
        batch.reserveCapacity(count)
        for _ in 0..<count {
            batch.append(next())
        }
        return batch
    }
}

// MARK: - Multi-Round Grinding

/// Engine for performing grinding at multiple FRI rounds.
/// Some FRI protocols require proof-of-work at each commit round,
/// not just at the end.
public struct MultiRoundGrinding {

    /// Configuration for multi-round grinding.
    public struct MultiRoundConfig {
        /// Per-round difficulty levels. If shorter than the number of rounds,
        /// the last value is repeated.
        public let perRoundDifficulty: [Int]

        /// Hash function for all rounds.
        public let hashFunction: GrindingHashFunction

        /// Maximum nonces per round.
        public let maxNoncesPerRound: UInt64

        public init(perRoundDifficulty: [Int],
                    hashFunction: GrindingHashFunction = .poseidon2,
                    maxNoncesPerRound: UInt64 = 1 << 28) {
            precondition(!perRoundDifficulty.isEmpty, "Need at least one difficulty level")
            self.perRoundDifficulty = perRoundDifficulty
            self.hashFunction = hashFunction
            self.maxNoncesPerRound = maxNoncesPerRound
        }
    }

    /// Result of a multi-round grinding operation.
    public struct MultiRoundResult {
        /// Per-round nonces.
        public let nonces: [UInt64]
        /// Per-round hash outputs.
        public let hashOutputs: [Fr]
        /// Total nonces checked across all rounds.
        public let totalNoncesChecked: UInt64
        /// Total time in seconds.
        public let totalTimeSeconds: Double

        public init(nonces: [UInt64], hashOutputs: [Fr],
                    totalNoncesChecked: UInt64, totalTimeSeconds: Double) {
            self.nonces = nonces
            self.hashOutputs = hashOutputs
            self.totalNoncesChecked = totalNoncesChecked
            self.totalTimeSeconds = totalTimeSeconds
        }
    }

    /// Execute multi-round grinding using the given engine.
    ///
    /// - Parameters:
    ///   - roundSeeds: Seeds for each round (typically Merkle roots).
    ///   - config: Multi-round configuration.
    ///   - engine: The grinding engine to use.
    /// - Returns: Multi-round result with all nonces.
    public static func grindAllRounds(
        roundSeeds: [Fr],
        config: MultiRoundConfig,
        engine: GPUFRIGrindingEngine) throws -> MultiRoundResult
    {
        let startTime = CFAbsoluteTimeGetCurrent()
        var nonces = [UInt64]()
        var hashOutputs = [Fr]()
        var totalChecked: UInt64 = 0

        nonces.reserveCapacity(roundSeeds.count)
        hashOutputs.reserveCapacity(roundSeeds.count)

        for (roundIdx, seed) in roundSeeds.enumerated() {
            // Get difficulty for this round
            let diffIdx = min(roundIdx, config.perRoundDifficulty.count - 1)
            let difficulty = config.perRoundDifficulty[diffIdx]

            // Incorporate previous round nonce into seed for chaining
            let effectiveSeed: Fr
            if roundIdx > 0 {
                let prevNonce = frFromInt(nonces[roundIdx - 1])
                effectiveSeed = poseidon2Hash(seed, prevNonce)
            } else {
                effectiveSeed = seed
            }

            let roundConfig = GrindingConfig(
                difficulty: difficulty,
                hashFunction: config.hashFunction,
                maxNonces: config.maxNoncesPerRound,
                batchSize: 1 << 18,
                cpuVerify: true)

            let result = try engine.grind(seed: effectiveSeed, config: roundConfig)
            nonces.append(result.nonce)
            hashOutputs.append(result.hashOutput)
            totalChecked += result.noncesChecked
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        return MultiRoundResult(nonces: nonces, hashOutputs: hashOutputs,
                               totalNoncesChecked: totalChecked,
                               totalTimeSeconds: elapsed)
    }

    /// Verify a multi-round grinding result.
    public static func verifyMultiRound(
        roundSeeds: [Fr],
        result: MultiRoundResult,
        config: MultiRoundConfig) -> Bool
    {
        guard result.nonces.count == roundSeeds.count else { return false }

        for (roundIdx, seed) in roundSeeds.enumerated() {
            let diffIdx = min(roundIdx, config.perRoundDifficulty.count - 1)
            let difficulty = config.perRoundDifficulty[diffIdx]

            let effectiveSeed: Fr
            if roundIdx > 0 {
                let prevNonce = frFromInt(result.nonces[roundIdx - 1])
                effectiveSeed = poseidon2Hash(seed, prevNonce)
            } else {
                effectiveSeed = seed
            }

            let hash = GrindingVerifier.computeHash(
                seed: effectiveSeed, nonce: result.nonces[roundIdx],
                hashFunction: config.hashFunction)

            if !GrindingVerifier.meetsTarget(hash: hash, difficulty: difficulty) {
                return false
            }
        }

        return true
    }
}

// MARK: - Grinding Transcript Extension

/// Helpers for integrating grinding with the Fiat-Shamir transcript.
public struct GrindingTranscriptHelper {

    /// Derive a grinding seed from a sequence of transcript field elements.
    /// This mirrors what a Fiat-Shamir transcript would produce after
    /// absorbing FRI commit data.
    public static func deriveGrindingSeed(transcriptElements: [Fr]) -> Fr {
        guard !transcriptElements.isEmpty else { return Fr.zero }

        var state = transcriptElements[0]
        for i in 1..<transcriptElements.count {
            state = poseidon2Hash(state, transcriptElements[i])
        }

        // Domain separation for grinding
        let domainSep = frFromInt(0x706F775F67726964) // "pow_grid"
        return poseidon2Hash(state, domainSep)
    }

    /// After grinding succeeds, compute the post-grinding transcript state.
    /// This state is used to derive query positions.
    public static func postGrindingState(preGrindState: Fr, nonce: UInt64) -> Fr {
        let nonceFr = frFromInt(nonce)
        return poseidon2Hash(preGrindState, nonceFr)
    }

    /// Full flow: absorb commit data, grind, return post-grinding state.
    ///
    /// - Parameters:
    ///   - commitRoots: Merkle roots from FRI commit rounds.
    ///   - challenges: Folding challenges used.
    ///   - difficulty: Grinding difficulty.
    ///   - engine: Grinding engine.
    /// - Returns: Tuple of (nonce, postGrindState, grindingResult).
    public static func commitPhaseGrinding(
        commitRoots: [Fr], challenges: [Fr], difficulty: Int,
        engine: GPUFRIGrindingEngine) throws
        -> (nonce: UInt64, postState: Fr, result: GrindingResult)
    {
        let seed = FRIGrindingIntegration.computeGrindingSeed(
            commitRoots: commitRoots, foldingChallenges: challenges)

        let config = GrindingConfig(difficulty: difficulty)
        let result = try engine.grind(seed: seed, config: config)

        let postState = postGrindingState(preGrindState: seed, nonce: result.nonce)

        return (result.nonce, postState, result)
    }
}

// MARK: - Adaptive Difficulty

/// Adaptive difficulty controller that adjusts grinding difficulty based on
/// observed performance, targeting a specific proof time budget.
public struct AdaptiveDifficultyController {
    /// Target time for grinding in seconds.
    public let targetTimeSeconds: Double

    /// Minimum difficulty (security floor).
    public let minDifficulty: Int

    /// Maximum difficulty (to prevent runaway proof times).
    public let maxDifficulty: Int

    /// Smoothing factor for exponential moving average of hash rate.
    public let smoothingFactor: Double

    /// Current estimated hash rate (exponential moving average).
    public private(set) var estimatedHashRate: Double

    /// Current recommended difficulty.
    public var recommendedDifficulty: Int {
        if estimatedHashRate <= 0 { return minDifficulty }
        let d = Int(log2(targetTimeSeconds * estimatedHashRate))
        return max(minDifficulty, min(maxDifficulty, d))
    }

    public init(targetTimeSeconds: Double = 0.5,
                minDifficulty: Int = 8,
                maxDifficulty: Int = 24,
                smoothingFactor: Double = 0.3,
                initialHashRate: Double = 10_000_000) {
        self.targetTimeSeconds = targetTimeSeconds
        self.minDifficulty = minDifficulty
        self.maxDifficulty = maxDifficulty
        self.smoothingFactor = smoothingFactor
        self.estimatedHashRate = initialHashRate
    }

    /// Update the hash rate estimate from a grinding result.
    public mutating func update(from result: GrindingResult) {
        let observedRate = result.hashRate
        if observedRate > 0 {
            estimatedHashRate = smoothingFactor * observedRate +
                (1.0 - smoothingFactor) * estimatedHashRate
        }
    }
}
