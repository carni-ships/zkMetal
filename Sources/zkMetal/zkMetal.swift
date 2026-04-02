// zkMetal — GPU-accelerated ZK cryptography primitives for Apple Silicon
//
// Primitives:
// - Multi-Scalar Multiplication (Pippenger + signed-digit + GLV)
// - Number Theoretic Transform (four-step FFT, BN254/Goldilocks/BabyBear)
// - Poseidon2 hash (BN254 Fr, t=3)
// - Keccak-256 hash
// - Merkle trees (Poseidon2 and Keccak backends)
// - FRI folding (fused multi-round)
// - Sumcheck protocol
// - Polynomial operations (evaluation, interpolation, subproduct trees)
// - GPU radix sort

import Foundation
import Metal

/// Top-level namespace for zkMetal library.
public enum ZKMetal {
    /// Library version.
    public static let version = "0.3.0"

    /// Create a new MSM engine instance.
    public static func createMSMEngine() throws -> MetalMSM {
        try MetalMSM()
    }
}
