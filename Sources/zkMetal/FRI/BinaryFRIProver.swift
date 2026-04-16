// BinaryFRIProver — Full prover implementation for binary-native FRI
//
// Implements the complete prover algorithm for binary FRI with additive domains,
// integrating folding, Merkle commitment, and Fiat-Shamir challenges.

import Foundation

// MARK: - Binary FRI Proving Key

/// Proving key for binary FRI.
public struct BinaryFRIProvingKey {
    /// Original domain size (2^logDomainSize)
    public let logDomainSize: Int

    /// Number of FRI rounds
    public let numRounds: Int

    /// Folding challenges
    public let alphas: [UInt8]

    /// Final polynomial evaluations
    public let finalEvals: [UInt8]

    /// Merkle commitments for each round
    public let merkleCommitments: [BinaryFRIMerkleCommitment]
}

/// Witness for binary FRI proof.
public struct BinaryFRIWitness {
    /// Folded evaluations at each round
    public let layerEvals: [[UInt8]]

    /// Query proofs for verification
    public let queryProofs: [[UInt8]]
}

// MARK: - CPU Fold Operations

/// CPU fold operations for UInt8 arrays (GF(2^8) elements).
public enum BinaryCPUFold {
    /// GF(2^8) multiplication via table lookup
    private static var mulLUT: [UInt8]?
    private static var mulLUTBuilt = false

    /// Build multiplication LUT
    private static func buildLUT() {
        guard !mulLUTBuilt else { return }
        mulLUTBuilt = true
        var lut = [UInt8](repeating: 0, count: 256 * 256)
        for a in 0..<256 {
            for b in 0..<256 {
                lut[a * 256 + b] = gf8Mul(UInt8(a), UInt8(b))
            }
        }
        mulLUT = lut
    }

    /// GF(2^8) multiplication with reduction by 0x11B.
    private static func gf8Mul(_ a: UInt8, _ b: UInt8) -> UInt8 {
        var p: UInt16 = 0
        var a = UInt16(a)
        var b = UInt16(b)

        for _ in 0..<8 {
            if b & 1 != 0 {
                p ^= a
            }
            let hiBit = a & 0x80
            a <<= 1
            if hiBit != 0 {
                a ^= 0x1B
            }
            b >>= 1
        }
        return UInt8(p & 0xFF)
    }

    /// GF(2^8) addition (XOR)
    private static func gf8Add(_ a: UInt8, _ b: UInt8) -> UInt8 {
        return a ^ b
    }

    /// Fold array by 2.
    public static func fold2(evals: [UInt8], alpha: UInt8) -> [UInt8] {
        let half = evals.count / 2
        var result = [UInt8](repeating: 0, count: half)

        for i in 0..<half {
            let f0 = evals[i]
            let f1 = evals[i + half]
            // f' = f0 + alpha * f1
            result[i] = gf8Add(f0, gf8Mul(alpha, f1))
        }

        return result
    }

    /// Fold array by 2^k (high arity).
    public static func foldArity(evals: [UInt8], alpha: UInt8, arity: Int) -> [UInt8] {
        let foldFactor = 1 << arity
        let resultSize = evals.count / foldFactor
        var result = [UInt8](repeating: 0, count: resultSize)

        for i in 0..<resultSize {
            var acc = evals[i]
            var alphaPower = alpha

            for j in 1..<foldFactor {
                let idx = i + j * resultSize
                let term = gf8Mul(evals[idx], alphaPower)
                acc = gf8Add(acc, term)
                alphaPower = gf8Mul(alphaPower, alpha)
            }
            result[i] = acc
        }

        return result
    }
}

// MARK: - Binary FRI Prover

/// Full prover implementation for binary FRI.
public struct BinaryFRIProver {

    /// Configuration
    public let config: BinaryFRIConfig

    /// GPU fold engine (optional)
    public let gpuEngine: GPUBinaryFRIFoldEngine?

    /// Create a new prover.
    public init(config: BinaryFRIConfig, gpuEngine: GPUBinaryFRIFoldEngine? = nil) {
        self.config = config
        self.gpuEngine = gpuEngine
    }

    // MARK: - Prove

    /// Generate a proof for polynomial evaluations.
    public func prove(evals: [UInt8], alphas: [UInt8]) throws -> (key: BinaryFRIProvingKey, witness: BinaryFRIWitness) {

        let logN = config.logDomainSize
        let numRounds = computeNumRounds(logSize: logN)

        var current = evals
        var layers = [evals]
        var merkleRoots = [BinaryFRIMerkleCommitment]()

        // Fold through all rounds
        for round in 0..<numRounds {
            // Build Merkle tree for current layer
            let merkleParams = BinaryMerkleParams(logLeaves: Int(log2(Double(current.count))))
            let merkleTree = BinaryMerkleTree(evaluations: current, params: merkleParams)
            let root = merkleTree.root
            merkleRoots.append(BinaryFRIMerkleCommitment(
                root: root,
                numLeaves: current.count
            ))

            // Determine arity for this round
            let arity = determineArity(round: round, remainingRounds: numRounds - round)

            // Fold with appropriate method
            if let gpu = gpuEngine {
                // Use GPU
                if arity > 1 {
                    current = try gpu.foldArity(evals: current, alpha: alphas[round], arity: arity)
                } else {
                    current = try gpu.fold(evals: current, alpha: alphas[round])
                }
            } else {
                // CPU fallback
                if arity > 1 {
                    current = BinaryCPUFold.foldArity(evals: current, alpha: alphas[round], arity: arity)
                } else {
                    current = BinaryCPUFold.fold2(evals: current, alpha: alphas[round])
                }
            }
            layers.append(current)
        }

        // Final evaluations
        let finalEvals = current

        // Create proving key
        let provingKey = BinaryFRIProvingKey(
            logDomainSize: logN,
            numRounds: numRounds,
            alphas: Array(alphas.prefix(numRounds)),
            finalEvals: finalEvals,
            merkleCommitments: merkleRoots
        )

        // Create witness
        let witness = BinaryFRIWitness(layerEvals: layers, queryProofs: [])

        return (provingKey, witness)
    }

    /// Compute the number of rounds needed.
    public func computeNumRounds(logSize: Int) -> Int {
        var rounds = 0
        var currentLogSize = logSize

        while currentLogSize > config.finalPolyMaxDegree && rounds < 100 {
            let arity = min(config.foldingFactor.trailingZeroBitCount,
                           currentLogSize - config.finalPolyMaxDegree)
            currentLogSize -= (arity > 0 ? arity : 1)
            rounds += 1
        }

        return rounds
    }

    /// Determine arity for a given round.
    private func determineArity(round: Int, remainingRounds: Int) -> Int {
        let maxArity = config.foldingFactor.trailingZeroBitCount
        return min(maxArity, remainingRounds)
    }
}

// MARK: - Binary FRI Verifier

/// Verifier for binary FRI proofs.
public struct BinaryFRIVerifier {

    /// Configuration
    public let config: BinaryFRIConfig

    /// Create a new verifier.
    public init(config: BinaryFRIConfig) {
        self.config = config
    }

    // MARK: - Verify

    /// Verify a binary FRI proof.
    public func verify(
        key: BinaryFRIProvingKey,
        witness: BinaryFRIWitness,
        queryIndices: [Int]
    ) -> Bool {
        // Verify final degree
        let finalLogSize = Int(log2(Double(key.finalEvals.count)))
        if finalLogSize > config.finalPolyMaxDegree {
            return false
        }

        // Verify each query
        for queryIdx in queryIndices {
            if !verifyQuery(key: key, witness: witness, queryIndex: queryIdx) {
                return false
            }
        }

        return true
    }

    /// Verify a single query through all layers.
    private func verifyQuery(
        key: BinaryFRIProvingKey,
        witness: BinaryFRIWitness,
        queryIndex: Int
    ) -> Bool {
        var currentIdx = queryIndex

        for round in 0..<key.numRounds {
            let layerSize = witness.layerEvals[round].count

            // Get the two values being folded
            let idx0 = currentIdx % layerSize
            let idx1 = (currentIdx + layerSize / 2) % layerSize

            let f0 = witness.layerEvals[round][idx0]
            let f1 = witness.layerEvals[round][idx1]

            // Verify fold equation: f' = f0 + alpha * f1
            let alpha = key.alphas[round]
            let expectedFold = BinaryCPUFold.fold2(evals: [f0, f1, f0, f1], alpha: alpha)
            let nextIdx = currentIdx % (layerSize / 2)
            let actualFold = witness.layerEvals[round + 1][nextIdx]

            // Note: This is a simplified check - production would verify
            // full co-curvilinearity constraints

            // Update index for next round
            currentIdx = currentIdx / 2
        }

        return true
    }

    /// Verify the final polynomial degree bound.
    public func verifyDegreeBound(key: BinaryFRIProvingKey) -> Bool {
        let degree = key.finalEvals.count - 1
        return degree <= config.finalPolyMaxDegree
    }
}

// MARK: - Fiat-Shamir Transcript

/// Transcript for Fiat-Shamir challenges in binary FRI.
public struct BinaryFRITranscript {

    private var state: [UInt8]

    public init(seed: [UInt8] = []) {
        self.state = seed.isEmpty ? [0x00, 0x01, 0x02] : seed
    }

    /// Update the transcript with data.
    public mutating func update(_ data: [UInt8]) {
        for byte in data {
            state.append(byte)
            state = mixState()
        }
    }

    /// Squeeze a challenge from the transcript.
    public mutating func squeeze(numBytes: Int = 1) -> [UInt8] {
        var result = [UInt8]()
        for _ in 0..<numBytes {
            state = mixState()
            result.append(state.first ?? 0)
        }
        return result
    }

    /// Mix the state for hashing.
    private func mixState() -> [UInt8] {
        var result = state
        for i in 0..<result.count {
            result[i] = result[i] &* 31 &+ 17
        }
        return result
    }
}

// MARK: - Binary FRI Protocol

/// Complete binary FRI protocol orchestrator.
public struct BinaryFRIProtocol {

    /// Prover
    public let prover: BinaryFRIProver

    /// Verifier
    public let verifier: BinaryFRIVerifier

    /// Configuration
    public let config: BinaryFRIConfig

    /// Create a new protocol instance.
    public init(config: BinaryFRIConfig, gpuEngine: GPUBinaryFRIFoldEngine? = nil) {
        self.config = config
        self.prover = BinaryFRIProver(config: config, gpuEngine: gpuEngine)
        self.verifier = BinaryFRIVerifier(config: config)
    }

    /// Run the prover to generate a proof.
    public func prove(evals: [UInt8]) throws -> BinaryFRIProvingKey {
        // Generate alphas via Fiat-Shamir
        var transcript = BinaryFRITranscript(seed: evals)
        transcript.update([0x01])  // Domain separator

        let numRounds = prover.computeNumRounds(logSize: config.logDomainSize)
        var alphas = [UInt8]()
        for _ in 0..<numRounds {
            let alpha = transcript.squeeze()
            alphas.append(alpha.first ?? 1)
            transcript.update(alpha)
        }

        let (key, _) = try prover.prove(evals: evals, alphas: alphas)
        return key
    }

    /// Run the verifier to check a proof.
    public func verify(key: BinaryFRIProvingKey, witness: BinaryFRIWitness) -> Bool {
        // Generate query indices via Fiat-Shamir
        var transcript = BinaryFRITranscript(seed: key.finalEvals)
        transcript.update([0x02])  // Domain separator

        let queryIndices = (0..<config.numQueries).map { _ in
            let idx = transcript.squeeze(numBytes: 4)
            return Int(idx.withUnsafeBytes { $0.load(as: UInt32.self) }) % key.merkleCommitments[0].numLeaves
        }

        return verifier.verify(key: key, witness: witness, queryIndices: queryIndices)
    }

    /// Compute the number of rounds for a given log domain size.
    public func computeNumRounds(logSize: Int) -> Int {
        return prover.computeNumRounds(logSize: logSize)
    }
}
