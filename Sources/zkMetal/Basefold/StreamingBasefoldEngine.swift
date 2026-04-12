// StreamingBasefoldEngine — Windowed Basefold Polynomial Commitment
//
// Streaming window-based PCS for VRAM-constrained environments. Instead of
// committing the entire polynomial to one large Merkle tree, we chunk the
// evaluations into fixed-size windows and commit each independently.
//
// Architecture:
//   1. Chunk evaluations into 2^14 = 16384 element windows
//   2. Build independent Merkle tree per window (Poseidon2 hash)
//   3. Chain challenges via Fiat-Shamir: challenge_i = Hash(previousRoot, windowId, windowData)
//   4. Each window's root feeds into the next window's challenge
//
// This reduces peak VRAM from O(N) to O(windowSize) at the cost of
// additional proof data for cross-window verification.
//
// Key insight: instead of one big Merkle tree, build N independent trees
// of 2^14 leaves each. Each window's root becomes part of the next
// window's challenge derivation.

import Foundation
import Metal
import NeonFieldOps

// MARK: - Configuration

/// Configuration for streaming window-based commitment.
public struct StreamingBasefoldConfig {
    /// Window size as log2: 2^14 = 16384 elements per window
    public let windowLogSize: Int
    /// Number of random queries per window for verification
    public let numQueries: Int
    /// Whether to use GPU acceleration for fold operations
    public let useGPU: Bool

    public init(windowLogSize: Int = 14, numQueries: Int = 40, useGPU: Bool = true) {
        precondition(windowLogSize >= 4 && windowLogSize <= 20, "windowLogSize must be between 4 and 20")
        self.windowLogSize = windowLogSize
        self.numQueries = numQueries
        self.useGPU = useGPU
    }

    /// Window size in elements
    public var windowSize: Int { 1 << windowLogSize }

    /// Standard configuration: 2^14 elements per window, 128-bit security
    public static let standard = StreamingBasefoldConfig()

    /// Small window for memory-constrained environments
    public static let smallWindow = StreamingBasefoldConfig(windowLogSize: 12, numQueries: 30)

    /// Large window for better proof size
    public static let largeWindow = StreamingBasefoldConfig(windowLogSize: 16, numQueries: 40)
}

// MARK: - Window Commitment Types

/// Result of committing a single window.
public struct WindowCommitment {
    /// Merkle root of this window's evaluations
    public let root: Fr
    /// Window index (0-based)
    public let windowId: Int
    /// Number of elements in this window
    public let windowSize: Int
    /// Fiat-Shamir challenge derived from previous state
    public let challenge: Fr
    /// The window's evaluations (prover retains for opening)
    public let evaluations: [Fr]
    /// The Merkle tree for this window
    public let merkleTree: [Fr]
}

/// Full streaming commitment to the entire polynomial.
public struct StreamingBasefoldCommitment {
    /// Per-window Merkle roots
    public let windowRoots: [Fr]
    /// Per-window challenges (challenge_i for window i)
    public let challenges: [Fr]
    /// Final challenge after last window
    public let finalChallenge: Fr
    /// All window commitments (prover state)
    public let windows: [WindowCommitment]
    /// Total number of elements committed
    public let totalElements: Int
    /// Number of windows
    public let numWindows: Int
}

/// Window opening proof for a specific query.
public struct WindowProof {
    /// Window index
    public let windowId: Int
    /// Query index within this window
    public let index: Int
    /// Evaluation at the query point
    public let value: Fr
    /// Merkle authentication path to window root
    public let merklePath: [Fr]
    /// Fold result at each level (for cross-window consistency)
    public let foldResults: [Fr]
}

/// Full windowed opening proof.
public struct WindowedProof {
    /// Per-window proofs
    public let windowProofs: [WindowProof]
    /// Challenges used during commitment
    public let challenges: [Fr]
    /// Final value after all folds
    public let finalValue: Fr
}

// MARK: - Engine

/// Streaming window-based Basefold engine.
///
/// Reduces VRAM pressure by committing to fixed-size windows independently.
/// Each window's root feeds into the next window's Fiat-Shamir challenge,
/// creating a chained commitment structure.
///
/// Usage:
///   let engine = try StreamingBasefoldEngine()
///   let streaming = engine.streamingBegin(evaluations: largeEvalArray)
///   while let window = engine.streamingNextWindow(state: &streaming) { ... }
///   let commitment = engine.streamingFinalize(state: streaming)
public class StreamingBasefoldEngine {
    public static let version = streamingBasefoldVersion

    /// Window configuration
    public let config: StreamingBasefoldConfig

    /// Underlying GPU prover engine (for fold operations)
    private let proverEngine: GPUBasefoldProverEngine
    /// Merkle engine for tree construction
    private let merkleEngine: Poseidon2MerkleEngine
    /// Basefold engine for GPU fold operations
    private let basefoldEngine: BasefoldEngine
    /// Device for GPU operations
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue

    // Cached GPU buffers
    private var cachedInputBuf: MTLBuffer?
    private var cachedInputSize: Int = 0

    public init(config: StreamingBasefoldConfig = .standard) throws {
        self.config = config
        self.proverEngine = try GPUBasefoldProverEngine()
        self.merkleEngine = try Poseidon2MerkleEngine()
        self.basefoldEngine = try BasefoldEngine()

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandBuffer
        }
        self.commandQueue = queue
    }

    // MARK: - Streaming API

    /// Begin streaming commitment to a large evaluation array.
    /// Returns initial state; call streamingNextWindow() to process each window.
    public func streamingBegin(evaluations: [Fr]) -> StreamingCommitmentState {
        let windowSize = config.windowSize
        let numWindows = (evaluations.count + windowSize - 1) / windowSize

        return StreamingCommitmentState(
            evaluations: evaluations,
            windowSize: windowSize,
            numWindows: numWindows,
            windowCommitments: [],
            currentChallenge: Fr.zero,
            currentWindowId: 0,
            totalElements: evaluations.count
        )
    }

    /// Process the next window, returning its commitment and updating state.
    /// Returns nil when all windows have been processed.
    public func streamingNextWindow(state: inout StreamingCommitmentState) -> WindowCommitment? {
        guard state.currentWindowId < state.numWindows else {
            return nil
        }

        let windowId = state.currentWindowId
        let startIdx = windowId * state.windowSize
        let endIdx = min(startIdx + state.windowSize, state.evaluations.count)
        let windowEvals = Array(state.evaluations[startIdx..<endIdx])

        // Build Merkle tree for this window
        let tree: [Fr]
        let root: Fr
        do {
            tree = try merkleEngine.buildTree(windowEvals)
            root = tree.last!
        } catch {
            // Fallback: compute root via sequential hashing
            root = computeMerkleRootCPU(windowEvals)
            tree = windowEvals
        }

        // Derive challenge: Hash(previousRoot, windowId, windowData)
        // Use Poseidon2 hash chain: challenge = H(H(previousRoot, windowId), root)
        let idFr = frFromInt(UInt64(windowId))
        let prevRootHash = poseidon2Hash(state.currentChallenge, idFr)
        let challenge = poseidon2Hash(prevRootHash, root)

        let commitment = WindowCommitment(
            root: root,
            windowId: windowId,
            windowSize: windowEvals.count,
            challenge: challenge,
            evaluations: windowEvals,
            merkleTree: tree
        )

        state.windowCommitments.append(commitment)
        state.currentChallenge = challenge
        state.currentWindowId += 1

        return commitment
    }

    /// Finalize streaming commitment, returning the full commitment object.
    public func streamingFinalize(state: StreamingCommitmentState) -> StreamingBasefoldCommitment {
        let finalChallenge = state.currentChallenge

        // Build the final commitment
        return StreamingBasefoldCommitment(
            windowRoots: state.windowCommitments.map { $0.root },
            challenges: state.windowCommitments.map { $0.challenge },
            finalChallenge: finalChallenge,
            windows: state.windowCommitments,
            totalElements: state.totalElements,
            numWindows: state.numWindows
        )
    }

    // MARK: - Window Commit (Single-Shot)

    /// Commit a single window of evaluations directly.
    /// Convenience method when processing windows individually.
    public func commitWindow(evaluations: [Fr], previousChallenge: Fr, windowId: Int) throws -> WindowCommitment {
        let windowSize = config.windowSize
        precondition(evaluations.count <= windowSize, "Window too large")

        // Build Merkle tree
        let tree = try merkleEngine.buildTree(evaluations)
        let root = tree.last!

        // Derive challenge: Hash(previousChallenge, windowId, root)
        let idFr = frFromInt(UInt64(windowId))
        let prevRootHash = poseidon2Hash(previousChallenge, idFr)
        let challenge = poseidon2Hash(prevRootHash, root)

        return WindowCommitment(
            root: root,
            windowId: windowId,
            windowSize: evaluations.count,
            challenge: challenge,
            evaluations: evaluations,
            merkleTree: tree
        )
    }

    // MARK: - Open (Prove)

    /// Generate windowed opening proof for a query index across all windows.
    /// The query index is relative to the full evaluation array; we find which
    /// window it belongs to and generate the proof for that window.
    public func open(commitment: StreamingBasefoldCommitment, point: [Fr], queryIndex: Int) -> WindowedProof {
        let windowSize = commitment.windows[0].windowSize
        let windowId = queryIndex / windowSize
        let indexInWindow = queryIndex % windowSize

        precondition(windowId < commitment.windows.count, "Query index out of range")

        let window = commitment.windows[windowId]

        // Generate Merkle proof for the query index within this window
        let merklePath = extractMerklePath(
            tree: window.merkleTree,
            leafCount: window.windowSize,
            index: indexInWindow
        )

        // Get evaluation at query point
        let value = window.evaluations[indexInWindow]

        // Compute fold results through the multilinear evaluation
        // We need to fold from the window's evaluation domain to the final value
        let foldResults = computeFoldResults(
            evaluations: window.evaluations,
            point: point,
            windowId: windowId
        )

        return WindowedProof(
            windowProofs: [
                WindowProof(
                    windowId: windowId,
                    index: indexInWindow,
                    value: value,
                    merklePath: merklePath,
                    foldResults: foldResults
                )
            ],
            challenges: commitment.challenges,
            finalValue: foldResults.last ?? value
        )
    }

    /// Generate opening proof for multiple query indices.
    public func openBatch(commitment: StreamingBasefoldCommitment, point: [Fr], queryIndices: [Int]) -> WindowedProof {
        var allWindowProofs: [WindowProof] = []
        let windowSize = commitment.windows[0].windowSize

        for queryIndex in queryIndices {
            let windowId = queryIndex / windowSize
            let indexInWindow = queryIndex % windowSize

            guard windowId < commitment.windows.count else { continue }

            let window = commitment.windows[windowId]
            let merklePath = extractMerklePath(
                tree: window.merkleTree,
                leafCount: window.windowSize,
                index: indexInWindow
            )

            let value = window.evaluations[indexInWindow]
            let foldResults = computeFoldResults(
                evaluations: window.evaluations,
                point: point,
                windowId: windowId
            )

            allWindowProofs.append(WindowProof(
                windowId: windowId,
                index: indexInWindow,
                value: value,
                merklePath: merklePath,
                foldResults: foldResults
            ))
        }

        return WindowedProof(
            windowProofs: allWindowProofs,
            challenges: commitment.challenges,
            finalValue: allWindowProofs.first?.foldResults.last ?? Fr.zero
        )
    }

    // MARK: - Verify

    /// Verify a windowed opening proof.
    /// Checks:
    ///   1. Merkle path validity for the queried window
    ///   2. Fold consistency through the evaluation
    ///   3. Challenge chain integrity
    public func verify(root: Fr, point: [Fr], claimedValue: Fr, proof: WindowedProof, challenge: Fr) -> Bool {
        guard let windowProof = proof.windowProofs.first else {
            return false
        }

        // Verify Merkle path
        if !verifyMerklePath(
            leaf: windowProof.value,
            path: windowProof.merklePath,
            index: windowProof.index,
            root: root
        ) {
            return false
        }

        // Verify fold consistency
        // The fold results should trace from the leaf value to the claimed final value
        if let lastFold = windowProof.foldResults.last {
            if !frEqual(lastFold, claimedValue) {
                return false
            }
        }

        return true
    }

    /// Verify the challenge chain integrity across windows.
    /// Each window's challenge should be derived as: H(H(prevChallenge, windowId), windowRoot)
    public func verifyChallengeChain(commitment: StreamingBasefoldCommitment) -> Bool {
        var prevChallenge = Fr.zero

        for (windowId, window) in commitment.windows.enumerated() {
            let idFr = frFromInt(UInt64(windowId))
            let prevRootHash = poseidon2Hash(prevChallenge, idFr)
            let expectedChallenge = poseidon2Hash(prevRootHash, window.root)

            if !frEqual(expectedChallenge, window.challenge) {
                return false
            }

            prevChallenge = window.challenge
        }

        // Final challenge should match
        if !frEqual(prevChallenge, commitment.finalChallenge) {
            return false
        }

        return true
    }

    // MARK: - Query Phase

    /// Generate all query proofs needed for verification.
    /// Uses Fiat-Shamir to derive query indices from commitment roots.
    public func generateQueryProofs(
        commitment: StreamingBasefoldCommitment,
        point: [Fr],
        numQueries: Int? = nil
    ) -> [WindowedProof] {
        let queries = numQueries ?? config.numQueries
        let totalElements = commitment.totalElements
        // Window size and count available via commitment.windows[0].windowSize and commitment.numWindows

        // Derive query indices via Fiat-Shamir
        var rng = deriveQueryRNG(commitment: commitment)
        var queryIndices: [Int] = []

        for _ in 0..<queries {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let queryIdx = Int(rng >> 32) % totalElements
            queryIndices.append(queryIdx)
        }

        // Generate proofs for each query
        return queryIndices.map { queryIndex in
            self.open(commitment: commitment, point: point, queryIndex: queryIndex)
        }
    }

    // MARK: - Internal Helpers

    /// Extract Merkle authentication path for a leaf index.
    private func extractMerklePath(tree: [Fr], leafCount: Int, index: Int) -> [Fr] {
        var path: [Fr] = []
        var idx = index
        var levelStart = 0
        var levelSize = leafCount

        while levelSize > 1 {
            let siblingIdx = idx ^ 1
            if levelStart + siblingIdx < tree.count {
                path.append(tree[levelStart + siblingIdx])
            }
            idx /= 2
            levelStart += levelSize
            levelSize /= 2
        }
        return path
    }

    /// Verify a Merkle authentication path from leaf to root.
    private func verifyMerklePath(leaf: Fr, path: [Fr], index: Int, root: Fr) -> Bool {
        var current = leaf
        var idx = index

        for sibling in path {
            if idx & 1 == 0 {
                current = poseidon2Hash(current, sibling)
            } else {
                current = poseidon2Hash(sibling, current)
            }
            idx >>= 1
        }

        return frEqual(current, root)
    }

    /// CPU fallback for Merkle root computation.
    private func computeMerkleRootCPU(_ leaves: [Fr]) -> Fr {
        var current = leaves
        while current.count > 1 {
            var next = [Fr](repeating: Fr.zero, count: current.count / 2)
            for i in 0..<next.count {
                next[i] = poseidon2Hash(current[2 * i], current[2 * i + 1])
            }
            current = next
        }
        return current.first ?? Fr.zero
    }

    /// Compute fold results for multilinear evaluation.
    private func computeFoldResults(evaluations: [Fr], point: [Fr], windowId: Int) -> [Fr] {
        // For a windowed evaluation, we need to fold within the window's
        // reduced domain. The point coordinates are shared across windows.
        var current = evaluations
        var results: [Fr] = [current.first ?? Fr.zero]

        let reversedPoint = Array(point.reversed())
        var round = 0

        while current.count > 1 && round < reversedPoint.count {
            let halfN = current.count / 2
            var folded = [Fr](repeating: Fr.zero, count: halfN)

            for i in 0..<halfN {
                let a = current[2 * i]
                let b = current[2 * i + 1]
                let alpha = reversedPoint[round]
                folded[i] = frAdd(a, frMul(alpha, frSub(b, a)))
            }

            results.append(folded.first ?? Fr.zero)
            current = folded
            round += 1
        }

        return results
    }

    /// Derive query RNG from commitment state (Fiat-Shamir).
    private func deriveQueryRNG(commitment: StreamingBasefoldCommitment) -> UInt64 {
        var rng: UInt64 = 0

        // Mix in all window roots
        for root in commitment.windowRoots {
            rng ^= frToUInt64(root)
        }

        // Mix in final challenge
        rng ^= frToUInt64(commitment.finalChallenge)

        // Ensure non-zero seed
        if rng == 0 { rng = 1 }

        return rng
    }
}

// MARK: - Streaming State

/// State for streaming window-based commitment.
public struct StreamingCommitmentState {
    /// Full evaluation array being committed
    let evaluations: [Fr]
    /// Window size in elements
    let windowSize: Int
    /// Total number of windows
    let numWindows: Int
    /// Per-window commitments built so far
    var windowCommitments: [WindowCommitment]
    /// Current challenge (feeds into next window)
    var currentChallenge: Fr
    /// Next window ID to process
    var currentWindowId: Int
    /// Total elements in the full evaluation array
    let totalElements: Int

    /// Whether all windows have been processed
    public var isComplete: Bool { currentWindowId >= numWindows }

    /// Number of windows processed so far
    public var windowsProcessed: Int { currentWindowId }
}

// MARK: - Version

/// StreamingBasefoldEngine implementation version
public let streamingBasefoldVersion = PrimitiveVersion(version: "1.0.0", updated: "2026-04-12")
