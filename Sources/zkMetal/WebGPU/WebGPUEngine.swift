// WebGPUEngine.swift — WebGPU dispatch abstraction for zkMetal WGSL shaders
//
// Provides the structure for compiling and dispatching WGSL compute shaders
// via WebGPU. On macOS native there is no WebGPU runtime, so this module:
// 1. Generates correct WGSL shader source via WGSLCodegen
// 2. Provides dispatch parameter computation (workgroup counts, buffer layouts)
// 3. Defines the buffer binding layout matching the WGSL @group/@binding annotations
//
// In a browser environment (via wasm-bindgen or wgpu-native), the generated WGSL
// and dispatch parameters can be used directly with the WebGPU API.

import Foundation

/// Represents a compiled WGSL shader module ready for dispatch.
public struct WGSLShaderModule {
    /// The shader entry point name (e.g., "ntt_butterfly", "poseidon2_permute").
    public let entryPoint: String

    /// The WGSL source code.
    public let source: String

    /// Workgroup size used in the shader (matches @workgroup_size in WGSL).
    public let workgroupSize: Int

    /// Buffer binding layout: maps binding index to (group, binding, usage).
    public let bindings: [BufferBinding]

    public struct BufferBinding {
        public let group: Int
        public let binding: Int
        public let usage: BufferUsage
        public let label: String

        public enum BufferUsage {
            case storage       // read-write
            case readOnly      // read-only storage
            case uniform       // uniform buffer
        }
    }
}

/// Dispatch parameters for a WebGPU compute pass.
public struct WebGPUDispatchParams {
    /// Number of workgroups in each dimension.
    public let workgroupCountX: Int
    public let workgroupCountY: Int
    public let workgroupCountZ: Int

    /// Total threads = workgroupCount * workgroupSize.
    public var totalThreads: Int {
        workgroupCountX * workgroupCountY * workgroupCountZ * 256
    }
}

/// Engine for managing WGSL shader compilation and compute dispatch.
///
/// This is a structural abstraction: it generates correct WGSL source and
/// dispatch parameters but does not call WebGPU APIs directly (no WebGPU
/// runtime exists on macOS native). Use with wgpu-native or browser WebGPU.
public final class WebGPUEngine {

    /// All compiled shader modules, keyed by entry point name.
    public private(set) var modules: [String: WGSLShaderModule] = [:]

    /// Workgroup size used across all kernels.
    public static let workgroupSize = 256

    public init() {}

    // MARK: - Shader Compilation

    /// Compile a WGSL shader source and register it by entry point name.
    /// Returns the shader module for dispatch parameter queries.
    @discardableResult
    public func compileShader(entryPoint: String, wgsl source: String, bindings: [WGSLShaderModule.BufferBinding]) -> WGSLShaderModule {
        let module = WGSLShaderModule(
            entryPoint: entryPoint,
            source: source,
            workgroupSize: Self.workgroupSize,
            bindings: bindings
        )
        modules[entryPoint] = module
        return module
    }

    /// Load and compile all pre-generated WGSL shaders.
    public func loadAllShaders() {
        loadNTTShaders()
        loadPoseidon2Shaders()
        loadMSMShaders()
    }

    // MARK: - NTT Dispatch

    /// Compute dispatch parameters for an NTT butterfly stage.
    /// - Parameters:
    ///   - n: Transform size (must be power of 2).
    ///   - stage: Current butterfly stage index.
    /// - Returns: Dispatch parameters for the butterfly kernel.
    public func dispatchNTT(n: Int, stage: Int) -> WebGPUDispatchParams {
        let numButterflies = n / 2
        let workgroups = (numButterflies + Self.workgroupSize - 1) / Self.workgroupSize
        return WebGPUDispatchParams(
            workgroupCountX: workgroups,
            workgroupCountY: 1,
            workgroupCountZ: 1
        )
    }

    /// Generate the full NTT dispatch sequence (all stages).
    /// - Parameters:
    ///   - n: Transform size.
    ///   - direction: .forward for NTT, .inverse for iNTT.
    /// - Returns: Array of (entryPoint, dispatchParams) for each stage.
    public func nttDispatchSequence(n: Int, direction: NTTDirection) -> [(String, WebGPUDispatchParams)] {
        let logN = Int(log2(Double(n)))
        let entryPoint = direction == .forward ? "ntt_butterfly" : "intt_butterfly"
        return (0..<logN).map { stage in
            (entryPoint, dispatchNTT(n: n, stage: stage))
        }
    }

    public enum NTTDirection {
        case forward
        case inverse
    }

    // MARK: - Poseidon2 Dispatch

    /// Compute dispatch parameters for Poseidon2 batch permutation.
    /// - Parameter count: Number of independent permutations.
    public func dispatchPoseidon2(count: Int) -> WebGPUDispatchParams {
        let workgroups = (count + Self.workgroupSize - 1) / Self.workgroupSize
        return WebGPUDispatchParams(
            workgroupCountX: workgroups,
            workgroupCountY: 1,
            workgroupCountZ: 1
        )
    }

    /// Compute dispatch for Poseidon2 2-to-1 hash pairs.
    /// - Parameter count: Number of pairs to hash.
    public func dispatchPoseidon2HashPairs(count: Int) -> WebGPUDispatchParams {
        return dispatchPoseidon2(count: count)
    }

    // MARK: - MSM Dispatch

    /// Compute dispatch parameters for MSM signed-digit extraction.
    /// - Parameter nPoints: Number of scalar/point pairs.
    public func dispatchMSMExtract(nPoints: Int) -> WebGPUDispatchParams {
        let workgroups = (nPoints + Self.workgroupSize - 1) / Self.workgroupSize
        return WebGPUDispatchParams(
            workgroupCountX: workgroups,
            workgroupCountY: 1,
            workgroupCountZ: 1
        )
    }

    /// Compute dispatch parameters for MSM bucket sum.
    /// - Parameters:
    ///   - nSegments: Number of segments per window.
    ///   - nWindows: Number of scalar windows.
    public func dispatchMSMBucketSum(nSegments: Int, nWindows: Int) -> WebGPUDispatchParams {
        let total = nSegments * nWindows
        let workgroups = (total + Self.workgroupSize - 1) / Self.workgroupSize
        return WebGPUDispatchParams(
            workgroupCountX: workgroups,
            workgroupCountY: 1,
            workgroupCountZ: 1
        )
    }

    /// Full MSM dispatch sequence.
    /// - Parameters:
    ///   - nPoints: Number of points.
    ///   - windowBits: Window size in bits.
    /// - Returns: Array of (entryPoint, dispatchParams) for the full MSM pipeline.
    public func msmDispatchSequence(nPoints: Int, windowBits: Int) -> [(String, WebGPUDispatchParams)] {
        let nWindows = (256 + windowBits - 1) / windowBits
        let nBuckets = 1 << windowBits
        let nSegments = max(1, nBuckets / Self.workgroupSize)

        return [
            ("signed_digit_extract", dispatchMSMExtract(nPoints: nPoints)),
            ("msm_bucket_sum_direct", dispatchMSMBucketSum(nSegments: nSegments, nWindows: nWindows)),
        ]
    }

    // MARK: - Buffer Layout Helpers

    /// Size in bytes of a BN254 Fr element (8 × u32 = 32 bytes).
    public static let frElementSize = 32

    /// Size in bytes of a BN254 Fp element (8 × u32 = 32 bytes).
    public static let fpElementSize = 32

    /// Size in bytes of a projective point (3 × Fp = 96 bytes).
    public static let pointProjectiveSize = 96

    /// Size in bytes of an affine point (2 × Fp = 64 bytes).
    public static let pointAffineSize = 64

    /// Compute the storage buffer size for NTT data.
    /// - Parameter n: Transform size.
    /// - Returns: Size in bytes.
    public static func nttBufferSize(n: Int) -> Int {
        return n * frElementSize
    }

    /// Compute the storage buffer size for Poseidon2 input/output.
    /// - Parameter count: Number of permutations.
    /// - Returns: Size in bytes for the state buffer (3 Fr per permutation).
    public static func poseidon2BufferSize(count: Int) -> Int {
        return count * 3 * frElementSize
    }

    // MARK: - Private Shader Loading

    private func loadNTTShaders() {
        let source = WGSLCodegen.generateNTT()
        compileShader(
            entryPoint: "ntt_butterfly",
            wgsl: source,
            bindings: [
                .init(group: 0, binding: 0, usage: .storage, label: "data"),
                .init(group: 0, binding: 1, usage: .readOnly, label: "twiddles"),
                .init(group: 0, binding: 2, usage: .uniform, label: "params"),
            ]
        )
        compileShader(
            entryPoint: "intt_butterfly",
            wgsl: source,
            bindings: [
                .init(group: 0, binding: 0, usage: .storage, label: "data"),
                .init(group: 0, binding: 1, usage: .readOnly, label: "twiddles_inv"),
                .init(group: 0, binding: 2, usage: .uniform, label: "params"),
            ]
        )
    }

    private func loadPoseidon2Shaders() {
        let source = WGSLCodegen.generatePoseidon2()
        compileShader(
            entryPoint: "poseidon2_permute",
            wgsl: source,
            bindings: [
                .init(group: 0, binding: 0, usage: .readOnly, label: "input"),
                .init(group: 0, binding: 1, usage: .storage, label: "output"),
                .init(group: 0, binding: 2, usage: .readOnly, label: "round_constants"),
                .init(group: 0, binding: 3, usage: .uniform, label: "count"),
            ]
        )
        compileShader(
            entryPoint: "poseidon2_hash_pairs",
            wgsl: source,
            bindings: [
                .init(group: 0, binding: 0, usage: .readOnly, label: "input"),
                .init(group: 0, binding: 1, usage: .storage, label: "output"),
                .init(group: 0, binding: 2, usage: .readOnly, label: "round_constants"),
                .init(group: 0, binding: 3, usage: .uniform, label: "count"),
            ]
        )
    }

    private func loadMSMShaders() {
        let source = WGSLCodegen.generateMSMBucket()
        compileShader(
            entryPoint: "signed_digit_extract",
            wgsl: source,
            bindings: [
                .init(group: 0, binding: 0, usage: .readOnly, label: "scalars"),
                .init(group: 0, binding: 1, usage: .storage, label: "digits"),
                .init(group: 0, binding: 2, usage: .uniform, label: "extract_params"),
            ]
        )
        compileShader(
            entryPoint: "msm_bucket_sum_direct",
            wgsl: source,
            bindings: [
                .init(group: 1, binding: 0, usage: .readOnly, label: "buckets"),
                .init(group: 1, binding: 1, usage: .storage, label: "segment_results"),
                .init(group: 1, binding: 2, usage: .uniform, label: "msm_params"),
                .init(group: 1, binding: 3, usage: .uniform, label: "seg_params"),
            ]
        )
    }
}
