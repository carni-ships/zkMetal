// zkMetal — GPU-accelerated ZK cryptography primitives for Apple Silicon
//
// Currently implements:
// - BN254 base field Fp arithmetic (Montgomery form)
// - BN254 elliptic curve point operations (Jacobian projective)
// - GLV endomorphism for scalar decomposition
// - Multi-Scalar Multiplication (Pippenger's bucket method)

import Foundation
import Metal

/// Top-level namespace for zkMetal library.
public enum ZKMetal {
    /// Library version.
    public static let version = "0.2.0"

    /// Create a new MSM engine instance.
    public static func createMSMEngine() throws -> MetalMSM {
        try MetalMSM()
    }
}
