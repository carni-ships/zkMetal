// Circle STARK Configuration — parameters controlling proof generation and verification
//
// Blowup factor: ratio of evaluation domain to trace domain (2 or 4 typical)
// Num queries: number of FRI query points (30 for ~100-bit, 50 for ~128-bit security)
// Extension degree: 4 for QM31 (128-bit security over M31)

import Foundation

/// Configuration for Circle STARK proof generation and verification.
public struct CircleSTARKConfig {
    /// Log2 of blowup factor (1 = 2x, 2 = 4x, 3 = 8x, 4 = 16x)
    public let logBlowup: Int

    /// Blowup factor: evaluation domain size / trace domain size
    public var blowupFactor: Int { 1 << logBlowup }

    /// Number of FRI query points for soundness
    public let numQueries: Int

    /// Extension field degree for challenges (4 = QM31 for 128-bit security)
    public let extensionDegree: Int

    /// Hash function used for Merkle commitments
    public let hashFunction: STARKHashFunction

    /// Grinding bits for proof-of-work (0 = disabled)
    public let grindingBits: Int

    /// Whether to use GPU acceleration when available
    public let useGPU: Bool

    /// Default configuration: 16x blowup, 30 queries, QM31 extension, Keccak hash
    public static let `default` = CircleSTARKConfig(
        logBlowup: 4,
        numQueries: 30,
        extensionDegree: 4,
        hashFunction: .keccak256,
        grindingBits: 0,
        useGPU: true
    )

    /// Fast configuration for testing: 4x blowup, 10 queries
    public static let fast = CircleSTARKConfig(
        logBlowup: 2,
        numQueries: 10,
        extensionDegree: 4,
        hashFunction: .keccak256,
        grindingBits: 0,
        useGPU: true
    )

    /// High security configuration: 16x blowup, 50 queries
    public static let highSecurity = CircleSTARKConfig(
        logBlowup: 4,
        numQueries: 50,
        extensionDegree: 4,
        hashFunction: .keccak256,
        grindingBits: 8,
        useGPU: true
    )

    public init(logBlowup: Int = 4, numQueries: Int = 30, extensionDegree: Int = 4,
                hashFunction: STARKHashFunction = .keccak256, grindingBits: Int = 0,
                useGPU: Bool = true) {
        precondition(logBlowup >= 1 && logBlowup <= 8, "logBlowup must be in [1, 8]")
        precondition(numQueries >= 1 && numQueries <= 200, "numQueries must be in [1, 200]")
        precondition(extensionDegree == 1 || extensionDegree == 2 || extensionDegree == 4,
                     "extensionDegree must be 1, 2, or 4")
        self.logBlowup = logBlowup
        self.numQueries = numQueries
        self.extensionDegree = extensionDegree
        self.hashFunction = hashFunction
        self.grindingBits = grindingBits
        self.useGPU = useGPU
    }

    /// Compute security bits: each query eliminates ~logBlowup bits of cheating probability
    public var securityBits: Int {
        numQueries * logBlowup + grindingBits
    }
}

/// Hash function choices for STARK Merkle commitments
public enum STARKHashFunction {
    case keccak256
    case poseidon2M31
    case blake3
    case sha256
}
