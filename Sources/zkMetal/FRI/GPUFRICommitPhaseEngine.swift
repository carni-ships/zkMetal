// GPUFRICommitPhaseEngine — GPU-accelerated FRI commit phase engine
//
// Implements the commit phase of the FRI (Fast Reed-Solomon IOP of Proximity)
// protocol with GPU acceleration for both folding and Merkle tree construction.
//
// The commit phase iteratively:
//   1. Splits polynomial evaluations: f(x) = f_even(x^2) + x * f_odd(x^2)
//   2. Folds using a random challenge: f'(x^2) = f_even(x^2) + alpha * f_odd(x^2)
//   3. Builds a Merkle commitment over the folded evaluations
//   4. Halves the evaluation domain
//   5. Repeats until the polynomial is below a degree threshold
//
// Supports configurable folding factors (2, 4, 8, 16) and both standard
// multiplicative and circle FRI domains. Uses GPUFRIFoldEngine for
// GPU-accelerated folding and GPUMerkleTreeEngine for Poseidon2-based
// Merkle commitments.
//
// Works with BN254 Fr field type.

import Foundation
import Metal
import NeonFieldOps

// MARK: - FRI Domain Types

/// The type of evaluation domain used in the FRI commit phase.
public enum FRICommitDomainType {
    /// Standard multiplicative domain: coset of roots of unity {g * omega^i}.
    case multiplicative
    /// Circle FRI domain: points on a circle group (used in Stwo/Circle STARKs).
    case circle
}

// MARK: - FRI Commit Phase Configuration

/// Configuration for the FRI commit phase engine.
public struct FRICommitPhaseConfig {
    /// Folding factor per round: 2 (standard), 4, 8, or 16.
    /// Higher factors produce fewer rounds but more work per round.
    public let foldingFactor: Int

    /// Log2 of the folding factor (1 for factor 2, 2 for 4, 3 for 8, 4 for 16).
    public var foldingBits: Int {
        foldingFactor.trailingZeroBitCount
    }

    /// Blowup factor: ratio of evaluation domain to polynomial degree.
    public let blowupFactor: Int

    /// Maximum degree of the final (remainder) polynomial.
    public let finalPolyMaxDegree: Int

    /// Domain type: multiplicative or circle.
    public let domainType: FRICommitDomainType

    /// Whether to keep all intermediate layer evaluations in memory.
    /// If false, only the Merkle roots are retained (saves memory for large polynomials).
    public let retainLayerEvals: Bool

    /// Number of cosets for coset-based FRI (usually 1; >1 for batched FRI).
    public let numCosets: Int

    public init(foldingFactor: Int = 2,
                blowupFactor: Int = 4,
                finalPolyMaxDegree: Int = 7,
                domainType: FRICommitDomainType = .multiplicative,
                retainLayerEvals: Bool = true,
                numCosets: Int = 1) {
        precondition(foldingFactor == 2 || foldingFactor == 4 ||
                     foldingFactor == 8 || foldingFactor == 16,
                     "Folding factor must be 2, 4, 8, or 16")
        precondition(blowupFactor >= 2 && (blowupFactor & (blowupFactor - 1)) == 0,
                     "Blowup factor must be a power of 2")
        precondition(finalPolyMaxDegree >= 0, "Final poly max degree must be non-negative")
        precondition(numCosets >= 1, "Need at least one coset")
        self.foldingFactor = foldingFactor
        self.blowupFactor = blowupFactor
        self.finalPolyMaxDegree = finalPolyMaxDegree
        self.domainType = domainType
        self.retainLayerEvals = retainLayerEvals
        self.numCosets = numCosets
    }
}

// MARK: - FRI Commit Phase Layer

/// A single committed layer in the FRI commit phase.
public struct FRICommitPhaseLayer {
    /// Evaluations at this layer (nil if retainLayerEvals is false for non-final layers).
    public let evaluations: [Fr]?
    /// Merkle root of this layer's evaluations.
    public let merkleRoot: Fr
    /// The full Merkle tree (for proof generation).
    public let merkleTree: MerkleTree
    /// Log2 of the evaluation domain size at this layer.
    public let logDomainSize: Int
    /// The folding challenge used to produce this layer (nil for the initial layer).
    public let foldingChallenge: Fr?

    public init(evaluations: [Fr]?, merkleRoot: Fr, merkleTree: MerkleTree,
                logDomainSize: Int, foldingChallenge: Fr?) {
        self.evaluations = evaluations
        self.merkleRoot = merkleRoot
        self.merkleTree = merkleTree
        self.logDomainSize = logDomainSize
        self.foldingChallenge = foldingChallenge
    }
}

// MARK: - FRI Commit Phase Result

/// Complete result of the FRI commit phase.
public struct FRICommitPhaseResult {
    /// All committed layers (layer 0 = initial polynomial, layer N = remainder).
    public let layers: [FRICommitPhaseLayer]
    /// Folding challenges used at each round transition.
    public let challenges: [Fr]
    /// The remainder polynomial coefficients (low degree).
    public let remainderPoly: [Fr]
    /// Configuration used.
    public let config: FRICommitPhaseConfig
    /// Number of folding rounds performed.
    public let numRounds: Int
    /// Total time for the commit phase in seconds.
    public let commitTimeSeconds: Double

    public init(layers: [FRICommitPhaseLayer], challenges: [Fr],
                remainderPoly: [Fr], config: FRICommitPhaseConfig,
                numRounds: Int, commitTimeSeconds: Double) {
        self.layers = layers
        self.challenges = challenges
        self.remainderPoly = remainderPoly
        self.config = config
        self.numRounds = numRounds
        self.commitTimeSeconds = commitTimeSeconds
    }

    /// Merkle roots of all committed layers.
    public var merkleRoots: [Fr] {
        layers.map { $0.merkleRoot }
    }
}

// MARK: - Circle Domain Point

/// A point on the circle domain for circle FRI.
/// Represents (x, y) where x^2 + y^2 = 1 over the field.
public struct CircleDomainPoint {
    public let x: Fr
    public let y: Fr

    public init(x: Fr, y: Fr) {
        self.x = x
        self.y = y
    }

    /// Double the point on the circle: (x, y) -> (2x^2 - 1, 2xy).
    public func doubled() -> CircleDomainPoint {
        let x2 = frMul(x, x)
        let two = frFromInt(2)
        let newX = frSub(frMul(two, x2), Fr.one)
        let newY = frMul(two, frMul(x, y))
        return CircleDomainPoint(x: newX, y: newY)
    }

    /// Conjugate: (x, y) -> (x, -y).
    public func conjugate() -> CircleDomainPoint {
        let negY = frSub(Fr.zero, y)
        return CircleDomainPoint(x: x, y: negY)
    }
}

// MARK: - GPUFRICommitPhaseEngine

/// GPU-accelerated FRI commit phase engine.
///
/// Orchestrates the iterative folding + commitment loop that forms the core
/// of the FRI protocol. Each round:
///   1. Splits the polynomial into even/odd parts
///   2. Folds with a random challenge (alpha):
///      f'(x^2) = f_even(x^2) + alpha * f_odd(x^2)
///   3. Commits via Poseidon2 Merkle tree
///   4. Derives the next challenge (Fiat-Shamir from the Merkle root)
///
/// Supports folding by 2, 4, 8, or 16 per round.
///
/// Usage:
///   let engine = try GPUFRICommitPhaseEngine()
///   let result = try engine.commitPhase(evaluations: evals, config: config)
public final class GPUFRICommitPhaseEngine {
    public static let version = Versions.fri

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let foldEngine: GPUFRIFoldEngine
    private let merkleEngine: GPUMerkleTreeEngine
    private let tuning: TuningConfig

    /// CPU fallback threshold: below this size, folding happens on CPU.
    public static let cpuFoldThreshold = 512

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue
        self.tuning = TuningManager.shared.config(device: device)
        self.foldEngine = try GPUFRIFoldEngine()
        self.merkleEngine = try GPUMerkleTreeEngine()
    }

    // MARK: - Main Commit Phase

    /// Execute the full FRI commit phase.
    ///
    /// - Parameters:
    ///   - evaluations: Polynomial evaluations on the blowup domain (size must be power of 2).
    ///   - challenges: Optional pre-determined folding challenges. If nil, challenges
    ///                 are derived via Fiat-Shamir from Merkle roots.
    ///   - config: Commit phase configuration.
    /// - Returns: FRICommitPhaseResult with all layers, challenges, and the remainder polynomial.
    public func commitPhase(evaluations: [Fr],
                            challenges: [Fr]? = nil,
                            config: FRICommitPhaseConfig) throws -> FRICommitPhaseResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        let n = evaluations.count
        precondition(n > 1 && (n & (n - 1)) == 0, "Evaluation count must be a power of 2")

        let logN = n.trailingZeroBitCount
        let foldBits = config.foldingBits

        // Compute number of rounds
        let targetSize = max((config.finalPolyMaxDegree + 1) * config.blowupFactor,
                             1 << foldBits)
        var numRounds = 0
        var sz = n
        while sz > targetSize {
            sz >>= foldBits
            numRounds += 1
        }

        if numRounds == 0 {
            // Already at target — just commit the initial evals
            let tree = try merkleEngine.buildTree(leaves: padToPow2(evaluations))
            let layer = FRICommitPhaseLayer(
                evaluations: evaluations, merkleRoot: tree.root,
                merkleTree: tree, logDomainSize: logN, foldingChallenge: nil)
            let remainder = extractRemainderPoly(evaluations: evaluations, logN: logN)
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            return FRICommitPhaseResult(
                layers: [layer], challenges: [], remainderPoly: remainder,
                config: config, numRounds: 0, commitTimeSeconds: elapsed)
        }

        var layers = [FRICommitPhaseLayer]()
        var usedChallenges = [Fr]()
        var currentEvals = evaluations
        var currentLogN = logN

        // Layer 0: commit to initial evaluations
        let tree0 = try merkleEngine.buildTree(leaves: padToPow2(currentEvals))
        layers.append(FRICommitPhaseLayer(
            evaluations: config.retainLayerEvals ? currentEvals : nil,
            merkleRoot: tree0.root, merkleTree: tree0,
            logDomainSize: currentLogN, foldingChallenge: nil))

        for round in 0..<numRounds {
            // Obtain or derive the folding challenge
            let challenge: Fr
            if let ch = challenges, round < ch.count {
                challenge = ch[round]
            } else {
                challenge = deriveChallenge(from: layers.last!.merkleRoot, round: round)
            }
            usedChallenges.append(challenge)

            // Perform the fold based on folding factor
            switch config.foldingFactor {
            case 2:
                currentEvals = try foldByTwo(evals: currentEvals, logN: currentLogN,
                                             challenge: challenge, config: config)
                currentLogN -= 1

            case 4:
                currentEvals = try foldByFour(evals: currentEvals, logN: currentLogN,
                                              challenge: challenge, config: config)
                currentLogN -= 2

            case 8:
                currentEvals = try foldByEight(evals: currentEvals, logN: currentLogN,
                                               challenge: challenge, config: config)
                currentLogN -= 3

            case 16:
                currentEvals = try foldBySixteen(evals: currentEvals, logN: currentLogN,
                                                 challenge: challenge, config: config)
                currentLogN -= 4

            default:
                fatalError("Unsupported folding factor \(config.foldingFactor)")
            }

            // Commit the folded evaluations
            let tree = try merkleEngine.buildTree(leaves: padToPow2(currentEvals))
            layers.append(FRICommitPhaseLayer(
                evaluations: config.retainLayerEvals ? currentEvals : nil,
                merkleRoot: tree.root, merkleTree: tree,
                logDomainSize: currentLogN, foldingChallenge: challenge))
        }

        // Extract the remainder polynomial
        let remainder = extractRemainderPoly(evaluations: currentEvals, logN: currentLogN)

        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        return FRICommitPhaseResult(
            layers: layers, challenges: usedChallenges,
            remainderPoly: remainder, config: config,
            numRounds: numRounds, commitTimeSeconds: elapsed)
    }

    // MARK: - Circle FRI Commit Phase

    /// Execute the FRI commit phase on a circle domain.
    ///
    /// Circle FRI folds by mapping circle domain points via the doubling map:
    ///   (x, y) -> (2x^2 - 1, 2xy)
    /// and the folding formula uses the y-coordinate inverse.
    ///
    /// - Parameters:
    ///   - evaluations: Evaluations on the circle domain.
    ///   - domainPoints: Circle domain points corresponding to evaluations.
    ///   - challenges: Optional pre-determined challenges.
    ///   - config: Configuration (domainType should be .circle).
    /// - Returns: FRICommitPhaseResult.
    public func commitPhaseCircle(
        evaluations: [Fr],
        domainPoints: [CircleDomainPoint],
        challenges: [Fr]? = nil,
        config: FRICommitPhaseConfig
    ) throws -> FRICommitPhaseResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        let n = evaluations.count
        precondition(n > 1 && (n & (n - 1)) == 0, "Size must be power of 2")
        precondition(domainPoints.count == n, "Domain points must match evaluations")

        let logN = n.trailingZeroBitCount
        let targetSize = max((config.finalPolyMaxDegree + 1) * config.blowupFactor, 2)
        var numRounds = 0
        var sz = n
        while sz > targetSize { sz >>= 1; numRounds += 1 }

        var layers = [FRICommitPhaseLayer]()
        var usedChallenges = [Fr]()
        var currentEvals = evaluations
        var currentDomain = domainPoints
        var currentLogN = logN

        // Initial commitment
        let tree0 = try merkleEngine.buildTree(leaves: padToPow2(currentEvals))
        layers.append(FRICommitPhaseLayer(
            evaluations: config.retainLayerEvals ? currentEvals : nil,
            merkleRoot: tree0.root, merkleTree: tree0,
            logDomainSize: currentLogN, foldingChallenge: nil))

        for round in 0..<numRounds {
            let challenge: Fr
            if let ch = challenges, round < ch.count {
                challenge = ch[round]
            } else {
                challenge = deriveChallenge(from: layers.last!.merkleRoot, round: round)
            }
            usedChallenges.append(challenge)

            // Circle fold: f'(2x^2-1) = f_even(x) + alpha/y * f_odd(x)
            let half = currentEvals.count / 2
            // Batch-invert all y-coordinates
            var yVals = [Fr](repeating: Fr.zero, count: half)
            for i in 0..<half { yVals[i] = currentDomain[i].y }
            var yPfx = [Fr](repeating: Fr.one, count: half)
            for i in 1..<half { yPfx[i] = frMul(yPfx[i - 1], yVals[i - 1]) }
            var yAcc = frInverse(frMul(yPfx[half - 1], yVals[half - 1]))
            var yInvs = [Fr](repeating: Fr.zero, count: half)
            for i in Swift.stride(from: half - 1, through: 0, by: -1) {
                yInvs[i] = frMul(yAcc, yPfx[i])
                yAcc = frMul(yAcc, yVals[i])
            }
            var folded = [Fr](repeating: Fr.zero, count: half)
            for i in 0..<half {
                let a = currentEvals[i]
                let b = currentEvals[i + half]
                let sum = frAdd(a, b)
                let diff = frSub(a, b)
                let term = frMul(challenge, frMul(diff, yInvs[i]))
                folded[i] = frAdd(sum, term)
            }
            currentEvals = folded

            // Double the domain points for the next round
            var nextDomain = [CircleDomainPoint]()
            nextDomain.reserveCapacity(half)
            for i in 0..<half {
                nextDomain.append(currentDomain[i].doubled())
            }
            currentDomain = nextDomain
            currentLogN -= 1

            let tree = try merkleEngine.buildTree(leaves: padToPow2(currentEvals))
            layers.append(FRICommitPhaseLayer(
                evaluations: config.retainLayerEvals ? currentEvals : nil,
                merkleRoot: tree.root, merkleTree: tree,
                logDomainSize: currentLogN, foldingChallenge: challenge))
        }

        let remainder = extractRemainderPoly(evaluations: currentEvals, logN: currentLogN)
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        return FRICommitPhaseResult(
            layers: layers, challenges: usedChallenges,
            remainderPoly: remainder, config: config,
            numRounds: numRounds, commitTimeSeconds: elapsed)
    }

    // MARK: - Batched Commit Phase

    /// Execute the commit phase for multiple polynomials sharing the same domain.
    ///
    /// The polynomials are combined via random linear combination before folding,
    /// yielding a single FRI commit. This is the standard batching approach used
    /// in STARK provers (Plonky2, Plonky3, Stwo).
    ///
    /// - Parameters:
    ///   - polynomials: Array of evaluation vectors (all same size, power of 2).
    ///   - batchCoeffs: Random linear combination coefficients (one per polynomial).
    ///   - challenges: Optional folding challenges.
    ///   - config: Configuration.
    /// - Returns: FRICommitPhaseResult for the combined polynomial.
    public func commitPhaseBatched(
        polynomials: [[Fr]],
        batchCoeffs: [Fr],
        challenges: [Fr]? = nil,
        config: FRICommitPhaseConfig
    ) throws -> FRICommitPhaseResult {
        precondition(!polynomials.isEmpty, "Need at least one polynomial")
        precondition(batchCoeffs.count == polynomials.count,
                     "Batch coefficients count must match polynomials count")

        let n = polynomials[0].count
        for poly in polynomials {
            precondition(poly.count == n, "All polynomials must have the same size")
        }

        // Combine: combined[i] = sum_j (batchCoeffs[j] * polynomials[j][i])
        var combined = [Fr](repeating: Fr.zero, count: n)
        for j in 0..<polynomials.count {
            var coeff = batchCoeffs[j]
            let poly = polynomials[j]
            combined.withUnsafeMutableBytes { cBuf in
                poly.withUnsafeBytes { pBuf in
                    withUnsafeBytes(of: &coeff) { sBuf in
                        bn254_fr_batch_mac_neon(
                            cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            sBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(n))
                    }
                }
            }
        }

        return try commitPhase(evaluations: combined, challenges: challenges, config: config)
    }

    // MARK: - Streaming Commit Phase

    /// Execute the commit phase in a streaming fashion, yielding each layer
    /// as it is produced. Useful for very large polynomials where holding all
    /// layer evaluations in memory is prohibitive.
    ///
    /// - Parameters:
    ///   - evaluations: Initial evaluations.
    ///   - config: Configuration (retainLayerEvals is ignored; layers are yielded individually).
    ///   - onLayer: Callback invoked for each committed layer.
    /// - Returns: The remainder polynomial and challenges.
    public func commitPhaseStreaming(
        evaluations: [Fr],
        config: FRICommitPhaseConfig,
        onLayer: (FRICommitPhaseLayer, Int) throws -> Void
    ) throws -> (remainderPoly: [Fr], challenges: [Fr]) {
        let n = evaluations.count
        precondition(n > 1 && (n & (n - 1)) == 0, "Size must be power of 2")

        let logN = n.trailingZeroBitCount
        let foldBits = config.foldingBits
        let targetSize = max((config.finalPolyMaxDegree + 1) * config.blowupFactor,
                             1 << foldBits)
        var numRounds = 0
        var sz = n
        while sz > targetSize { sz >>= foldBits; numRounds += 1 }

        var usedChallenges = [Fr]()
        var currentEvals = evaluations
        var currentLogN = logN
        var lastRoot = Fr.zero

        // Initial layer
        let tree0 = try merkleEngine.buildTree(leaves: padToPow2(currentEvals))
        let layer0 = FRICommitPhaseLayer(
            evaluations: currentEvals, merkleRoot: tree0.root,
            merkleTree: tree0, logDomainSize: currentLogN, foldingChallenge: nil)
        try onLayer(layer0, 0)
        lastRoot = tree0.root

        for round in 0..<numRounds {
            let challenge = deriveChallenge(from: lastRoot, round: round)
            usedChallenges.append(challenge)

            currentEvals = try foldByFactor(evals: currentEvals, logN: currentLogN,
                                            challenge: challenge, config: config)
            currentLogN -= foldBits

            let tree = try merkleEngine.buildTree(leaves: padToPow2(currentEvals))
            let layer = FRICommitPhaseLayer(
                evaluations: currentEvals, merkleRoot: tree.root,
                merkleTree: tree, logDomainSize: currentLogN, foldingChallenge: challenge)
            try onLayer(layer, round + 1)
            lastRoot = tree.root
        }

        let remainder = extractRemainderPoly(evaluations: currentEvals, logN: currentLogN)
        return (remainder, usedChallenges)
    }

    // MARK: - Query Proof Generation

    /// Generate authentication paths for a set of query positions across all layers.
    ///
    /// - Parameters:
    ///   - result: The commit phase result.
    ///   - queryIndices: Positions in the initial domain to query.
    /// - Returns: Array of (layerIndex, leafIndex, authPath) tuples per query.
    public func generateQueryProofs(
        result: FRICommitPhaseResult,
        queryIndices: [Int]
    ) -> [[(layerIndex: Int, leafIndex: Int, authPath: MerkleAuthPath)]] {
        var allProofs = [[(layerIndex: Int, leafIndex: Int, authPath: MerkleAuthPath)]]()
        allProofs.reserveCapacity(queryIndices.count)

        let foldBits = result.config.foldingBits

        for queryIdx in queryIndices {
            var proofs = [(layerIndex: Int, leafIndex: Int, authPath: MerkleAuthPath)]()
            var currentIdx = queryIdx

            for layerIdx in 0..<result.layers.count {
                let layer = result.layers[layerIdx]
                let domainSize = 1 << layer.logDomainSize
                let idx = currentIdx % domainSize
                let authPath = layer.merkleTree.proof(forLeafAt: idx)
                proofs.append((layerIndex: layerIdx, leafIndex: idx, authPath: authPath))

                // Compute index in the next (folded) layer
                currentIdx = idx % (domainSize >> foldBits)
            }

            allProofs.append(proofs)
        }

        return allProofs
    }

    // MARK: - Remainder Polynomial Verification

    /// Verify that the remainder polynomial has degree at most `maxDegree`.
    public func verifyRemainderDegree(result: FRICommitPhaseResult) -> Bool {
        let maxDeg = result.config.finalPolyMaxDegree
        let poly = result.remainderPoly
        if poly.count <= maxDeg + 1 { return true }
        for i in (maxDeg + 1)..<poly.count {
            if !poly[i].isZero { return false }
        }
        return true
    }

    /// Evaluate the remainder polynomial at a given point (Horner's method).
    public func evaluateRemainder(_ result: FRICommitPhaseResult, at point: Fr) -> Fr {
        let coeffs = result.remainderPoly
        guard !coeffs.isEmpty else { return Fr.zero }
        var acc = coeffs[coeffs.count - 1]
        for i in Swift.stride(from: coeffs.count - 2, through: 0, by: -1) {
            acc = frAdd(frMul(acc, point), coeffs[i])
        }
        return acc
    }

    // MARK: - Folding Implementations

    /// Fold by 2: standard FRI fold.
    /// f(x) = f_even(x^2) + x * f_odd(x^2)
    /// folded(x^2) = f_even(x^2) + alpha * f_odd(x^2)
    private func foldByTwo(evals: [Fr], logN: Int, challenge: Fr,
                           config: FRICommitPhaseConfig) throws -> [Fr] {
        let n = evals.count
        precondition(n == 1 << logN)
        let stride = MemoryLayout<Fr>.stride

        if n >= GPUFRICommitPhaseEngine.cpuFoldThreshold {
            // GPU fold via the fold engine
            let evalsBuf = device.makeBuffer(
                bytes: evals, length: n * stride,
                options: .storageModeShared)!
            let resultBuf = try foldEngine.fold(evals: evalsBuf, logN: logN, challenge: challenge)
            let half = n / 2
            let ptr = resultBuf.contents().bindMemory(to: Fr.self, capacity: half)
            return Array(UnsafeBufferPointer(start: ptr, count: half))
        }

        // CPU fallback
        return cpuFoldOnce(evals: evals, logN: logN, challenge: challenge)
    }

    /// GPU-resident multi-fold: chains k fold-by-2 rounds in a single command buffer.
    /// Avoids intermediate CPU↔GPU copies between sub-fold rounds.
    private func foldMultiRound(evals: [Fr], logN: Int, challenge: Fr,
                                 numSubFolds: Int) throws -> [Fr] {
        let n = evals.count
        precondition(n == 1 << logN && numSubFolds <= logN)
        let stride = MemoryLayout<Fr>.stride

        // Build derived challenges: ch, ch², ch³, ...
        var challenges = [Fr]()
        var ch = challenge
        for _ in 0..<numSubFolds {
            challenges.append(ch)
            ch = frMul(ch, challenge)
        }

        if n >= GPUFRICommitPhaseEngine.cpuFoldThreshold {
            let evalsBuf = device.makeBuffer(
                bytes: evals, length: n * stride,
                options: .storageModeShared)!
            let resultBuf = try foldEngine.foldMultiRound(
                evals: evalsBuf, logN: logN, challenges: challenges)
            let outN = n >> numSubFolds
            let ptr = resultBuf.contents().bindMemory(to: Fr.self, capacity: outN)
            return Array(UnsafeBufferPointer(start: ptr, count: outN))
        }

        // CPU fallback: chain individual folds
        var current = evals
        var curLogN = logN
        for i in 0..<numSubFolds {
            current = cpuFoldOnce(evals: current, logN: curLogN, challenge: challenges[i])
            curLogN -= 1
        }
        return current
    }

    /// Fold by 4: two successive folds with derived challenges.
    private func foldByFour(evals: [Fr], logN: Int, challenge: Fr,
                            config: FRICommitPhaseConfig) throws -> [Fr] {
        precondition(logN >= 2, "Need logN >= 2 for fold-by-4")
        return try foldMultiRound(evals: evals, logN: logN, challenge: challenge, numSubFolds: 2)
    }

    /// Fold by 8: three successive folds in a single GPU command buffer.
    private func foldByEight(evals: [Fr], logN: Int, challenge: Fr,
                             config: FRICommitPhaseConfig) throws -> [Fr] {
        precondition(logN >= 3, "Need logN >= 3 for fold-by-8")
        return try foldMultiRound(evals: evals, logN: logN, challenge: challenge, numSubFolds: 3)
    }

    /// Fold by 16: four successive folds in a single GPU command buffer.
    private func foldBySixteen(evals: [Fr], logN: Int, challenge: Fr,
                               config: FRICommitPhaseConfig) throws -> [Fr] {
        precondition(logN >= 4, "Need logN >= 4 for fold-by-16")
        return try foldMultiRound(evals: evals, logN: logN, challenge: challenge, numSubFolds: 4)
    }

    /// Generic fold dispatcher based on config.
    private func foldByFactor(evals: [Fr], logN: Int, challenge: Fr,
                              config: FRICommitPhaseConfig) throws -> [Fr] {
        switch config.foldingFactor {
        case 2:  return try foldByTwo(evals: evals, logN: logN, challenge: challenge, config: config)
        case 4:  return try foldByFour(evals: evals, logN: logN, challenge: challenge, config: config)
        case 8:  return try foldByEight(evals: evals, logN: logN, challenge: challenge, config: config)
        case 16: return try foldBySixteen(evals: evals, logN: logN, challenge: challenge, config: config)
        default: fatalError("Unsupported folding factor")
        }
    }

    // MARK: - CPU Fold Reference

    /// CPU-only single fold round for small domains or verification.
    private func cpuFoldOnce(evals: [Fr], logN: Int, challenge: Fr) -> [Fr] {
        let n = evals.count
        let half = n / 2

        // Build domain inverses
        let invTwiddles = precomputeInverseTwiddles(logN: logN)

        var folded = [Fr](repeating: Fr.zero, count: half)
        evals.withUnsafeBytes { eBuf in
            invTwiddles.withUnsafeBytes { tBuf in
                folded.withUnsafeMutableBytes { fBuf in
                    withUnsafeBytes(of: challenge) { cBuf in
                        bn254_fr_fri_fold(
                            eBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            tBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            fBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(half))
                    }
                }
            }
        }
        return folded
    }

    // MARK: - Polynomial Splitting

    /// Split polynomial evaluations into even and odd parts.
    /// Given f(x) = f_even(x^2) + x * f_odd(x^2), compute f_even and f_odd
    /// from evaluations on a domain {omega^i}.
    ///
    /// This is the algebraic decomposition that underlies FRI folding.
    public func splitEvenOdd(evaluations: [Fr], logN: Int) -> (even: [Fr], odd: [Fr]) {
        let n = evaluations.count
        precondition(n == 1 << logN)
        let half = n / 2

        // Build domain
        let invTwiddles = precomputeInverseTwiddles(logN: logN)

        var even = [Fr](repeating: Fr.zero, count: half)
        var odd = [Fr](repeating: Fr.zero, count: half)
        let twoInv = frInverse(frFromInt(2))

        for i in 0..<half {
            let a = evaluations[i]
            let b = evaluations[i + half]
            // f_even(omega^{2i}) = (f(omega^i) + f(-omega^i)) / 2
            //                    = (f(omega^i) + f(omega^{i+n/2})) / 2
            let sum = frAdd(a, b)
            even[i] = frMul(sum, twoInv)

            // f_odd(omega^{2i}) = (f(omega^i) - f(-omega^i)) / (2 * omega^i)
            //                   = (f(omega^i) - f(omega^{i+n/2})) / (2 * omega^i)
            let diff = frSub(a, b)
            let twoOmegaInv = frMul(twoInv, invTwiddles[i])
            odd[i] = frMul(diff, twoOmegaInv)
        }

        return (even, odd)
    }

    // MARK: - Domain Halving

    /// Compute the halved domain: given domain {omega^i} of size n,
    /// produce domain {omega^{2i}} of size n/2. This is the "squaring map"
    /// that halves the multiplicative domain at each FRI round.
    public func halveDomain(domain: [Fr]) -> [Fr] {
        let n = domain.count
        let half = n / 2
        var halved = [Fr](repeating: Fr.zero, count: half)
        for i in 0..<half {
            halved[i] = frMul(domain[i], domain[i])
        }
        return halved
    }

    /// Build the full multiplicative domain of size 2^logN.
    public func buildDomain(logN: Int) -> [Fr] {
        let n = 1 << logN
        let invTwiddles = precomputeInverseTwiddles(logN: logN)
        let omega = frInverse(invTwiddles[1])
        var domain = [Fr](repeating: Fr.one, count: n)
        var w = Fr.one
        for i in 0..<n {
            domain[i] = w
            w = frMul(w, omega)
        }
        return domain
    }

    /// Build a coset domain: {g * omega^i} where g is the coset generator.
    public func buildCosetDomain(logN: Int, cosetGenerator: Fr) -> [Fr] {
        let n = 1 << logN
        let invTwiddles = precomputeInverseTwiddles(logN: logN)
        let omega = frInverse(invTwiddles[1])
        var domain = [Fr](repeating: Fr.zero, count: n)
        var w = cosetGenerator
        for i in 0..<n {
            domain[i] = w
            w = frMul(w, omega)
        }
        return domain
    }

    // MARK: - Helpers

    /// Derive a folding challenge from a Merkle root via Fiat-Shamir.
    private func deriveChallenge(from root: Fr, round: Int) -> Fr {
        let roundField = frFromInt(UInt64(round + 1))
        return poseidon2Hash(root, roundField)
    }

    /// Extract remainder polynomial coefficients via inverse NTT.
    private func extractRemainderPoly(evaluations: [Fr], logN: Int) -> [Fr] {
        let n = evaluations.count
        if n <= 1 { return evaluations }

        let invTwiddles = precomputeInverseTwiddles(logN: logN)
        var coeffs = evaluations

        // Gentleman-Sande inverse NTT (natural order → natural order)
        var m = n
        while m > 1 {
            let halfM = m >> 1
            let twiddleStep = n / m
            for k in Swift.stride(from: 0, to: n, by: m) {
                for j in 0..<halfM {
                    let u = coeffs[k + j]
                    let v = coeffs[k + j + halfM]
                    coeffs[k + j] = frAdd(u, v)
                    let diff = frSub(u, v)
                    coeffs[k + j + halfM] = frMul(diff, invTwiddles[j * twiddleStep])
                }
            }
            m = halfM
        }

        // Bit-reversal permutation
        for i in 0..<n {
            var x = i, r = 0
            for _ in 0..<logN { r = (r << 1) | (x & 1); x >>= 1 }
            if i < r {
                let tmp = coeffs[i]
                coeffs[i] = coeffs[r]
                coeffs[r] = tmp
            }
        }

        // Multiply by 1/n
        let nInv = frInverse(frFromInt(UInt64(n)))
        coeffs.withUnsafeMutableBytes { rBuf in
            withUnsafeBytes(of: nInv) { sBuf in
                bn254_fr_batch_mul_scalar(
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    sBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(n))
            }
        }

        return coeffs
    }

    /// Pad array to next power of 2 for Merkle tree.
    private func padToPow2(_ arr: [Fr]) -> [Fr] {
        let n = arr.count
        if n & (n - 1) == 0 { return arr }
        var next = 1
        while next < n { next <<= 1 }
        var padded = arr
        padded.append(contentsOf: [Fr](repeating: Fr.zero, count: next - n))
        return padded
    }

    /// Clear internal engine caches.
    public func clearCache() {
        foldEngine.clearCache()
    }
}
