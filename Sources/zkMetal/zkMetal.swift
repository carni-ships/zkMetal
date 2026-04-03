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

/// Find the Shaders directory. Checks (in order):
/// 1. ZKMETAL_SHADER_DIR environment variable
/// 2. Bundle resources
/// 3. Executable-relative paths
/// 4. Current working directory
public func findShaderDir() -> String {
    // 1. Environment variable (for FFI / non-Swift callers)
    if let envDir = ProcessInfo.processInfo.environment["ZKMETAL_SHADER_DIR"] {
        if FileManager.default.fileExists(atPath: "\(envDir)/fields/bn254_fp.metal") ||
           FileManager.default.fileExists(atPath: "\(envDir)/fields/bn254_fr.metal") {
            return envDir
        }
    }

    // 2. Bundle resources
    for bundle in Bundle.allBundles {
        if let url = bundle.url(forResource: "Shaders", withExtension: nil) {
            let frPath = url.appendingPathComponent("fields/bn254_fr.metal").path
            let fpPath = url.appendingPathComponent("fields/bn254_fp.metal").path
            if FileManager.default.fileExists(atPath: frPath) ||
               FileManager.default.fileExists(atPath: fpPath) {
                return url.path
            }
        }
    }

    // 3. Executable-relative
    let execPath = CommandLine.arguments[0]
    let execDir = (execPath as NSString).deletingLastPathComponent
    let candidates = [
        "\(execDir)/../Sources/Shaders",
        "./Sources/Shaders",
    ]
    for path in candidates {
        if FileManager.default.fileExists(atPath: "\(path)/fields/bn254_fp.metal") ||
           FileManager.default.fileExists(atPath: "\(path)/fields/bn254_fr.metal") {
            return path
        }
    }
    return "./Sources/Shaders"
}
