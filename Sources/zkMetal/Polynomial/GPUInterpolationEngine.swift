// GPU-accelerated polynomial interpolation engine
import NeonFieldOps
//
// Recovers polynomial coefficients from evaluation points {(x_i, y_i)}.
//
// Three modes:
//   1. interpolate(points:values:) — general Lagrange interpolation, GPU-accelerated
//      barycentric weight computation for the O(n^2) denominator products
//   2. interpolateOnDomain(evals:domain:) — when domain points are known separately
//   3. interpolateNTT(evals:) — when domain is roots of unity, uses GPU iNTT (O(n log n))
//
// CPU fallback for small inputs (n < 64).

import Foundation
import Metal

// MARK: - GPUInterpolationEngine

public class GPUInterpolationEngine {
    public static let version = Versions.interpolation

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // GPU pipelines for Lagrange interpolation
    private let denomPipeline: MTLComputePipelineState
    private let scalePipeline: MTLComputePipelineState
    private let evalPipeline: MTLComputePipelineState

    // NTT engine for roots-of-unity interpolation (lazy)
    private var nttEngine: NTTEngine?

    // CPU fallback threshold
    public static let cpuThreshold = 64

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        let library = try GPUInterpolationEngine.compileShaders(device: device)

        guard let denomFn = library.makeFunction(name: "lagrange_denom_bn254"),
              let scaleFn = library.makeFunction(name: "lagrange_scale_bn254"),
              let evalFn = library.makeFunction(name: "lagrange_eval_bn254") else {
            throw MSMError.missingKernel
        }

        self.denomPipeline = try device.makeComputePipelineState(function: denomFn)
        self.scalePipeline = try device.makeComputePipelineState(function: scaleFn)
        self.evalPipeline = try device.makeComputePipelineState(function: evalFn)
    }

    // MARK: - Shader compilation

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let fieldSrc = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let interpSrc = try String(contentsOfFile: shaderDir + "/poly/lagrange_interp.metal", encoding: .utf8)

        func clean(_ src: String) -> String {
            src.split(separator: "\n")
                .filter { line in
                    if line.contains("#include") || line.contains("#ifndef") || line.contains("#endif") { return false }
                    if line.contains("#define") {
                        let trimmed = line.trimmingCharacters(in: .whitespaces)
                        let parts = trimmed.split(separator: " ", maxSplits: 3)
                        return parts.count >= 3
                    }
                    return true
                }
                .joined(separator: "\n")
        }

        let combined = clean(fieldSrc) + "\n" + clean(interpSrc)

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    // MARK: - General Lagrange interpolation

    /// Recover polynomial coefficients from evaluation points.
    /// Given {(x_i, y_i)} for i=0..n-1, returns coefficients [c_0, c_1, ..., c_{n-1}]
    /// such that p(x) = c_0 + c_1*x + ... + c_{n-1}*x^{n-1} and p(x_i) = y_i.
    ///
    /// GPU-accelerated for n >= cpuThreshold; CPU fallback for small n.
    public func interpolate(points: [Fr], values: [Fr]) throws -> [Fr] {
        let n = points.count
        guard n == values.count, n > 0 else {
            throw MSMError.invalidInput
        }
        if n == 1 {
            return [values[0]]
        }

        if n < GPUInterpolationEngine.cpuThreshold {
            return cpuLagrangeInterpolate(points: points, values: values)
        }

        return try gpuLagrangeInterpolate(points: points, values: values)
    }

    /// Convenience: interpolate from (point, value) tuples.
    public func interpolate(pointValuePairs: [(Fr, Fr)]) throws -> [Fr] {
        let points = pointValuePairs.map { $0.0 }
        let values = pointValuePairs.map { $0.1 }
        return try interpolate(points: points, values: values)
    }

    // MARK: - Domain interpolation

    /// Interpolate on a known domain: given evaluations y_i at domain points x_i.
    /// Equivalent to interpolate(points: domain, values: evals).
    public func interpolateOnDomain(evals: [Fr], domain: [Fr]) throws -> [Fr] {
        return try interpolate(points: domain, values: evals)
    }

    // MARK: - NTT-based interpolation (roots of unity)

    /// Fast interpolation when the domain is n-th roots of unity.
    /// evals[i] = p(omega^i), returns polynomial coefficients via inverse NTT.
    /// n must be a power of 2 and n <= 2^28 (BN254 TWO_ADICITY).
    public func interpolateNTT(evals: [Fr]) throws -> [Fr] {
        let n = evals.count
        guard n > 0 && (n & (n - 1)) == 0 else {
            throw MSMError.invalidInput
        }
        let logN = Int(log2(Double(n)))
        guard logN <= Fr.TWO_ADICITY else {
            throw MSMError.invalidInput
        }

        // Lazy init NTT engine
        if nttEngine == nil {
            nttEngine = try NTTEngine()
        }

        // iNTT converts evaluations at roots of unity back to coefficients
        return try nttEngine!.intt(evals)
    }

    // MARK: - GPU Lagrange interpolation

    /// GPU-accelerated Lagrange interpolation.
    /// Phase 1: GPU computes denominator products denom[i] = prod_{j!=i}(x_i - x_j) in parallel
    /// Phase 2: CPU batch-inverts denominators (single inversion + O(n) muls)
    /// Phase 3: CPU accumulates Lagrange basis polynomials with scaled weights
    private func gpuLagrangeInterpolate(points: [Fr], values: [Fr]) throws -> [Fr] {
        let n = points.count
        let elemSize = 32 // 8 x UInt32 = 32 bytes per Fr

        // Allocate GPU buffers — pass Fr bytes directly, no flatMap copy
        guard let pointsBuf = points.withUnsafeBytes({ buf in
                  device.makeBuffer(bytes: buf.baseAddress!, length: n * elemSize,
                                    options: .storageModeShared)
              }),
              let denomsBuf = device.makeBuffer(length: n * elemSize,
                                                 options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate interpolation buffers")
        }

        // Phase 1: GPU compute denominator products
        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(denomPipeline)
        enc.setBuffer(pointsBuf, offset: 0, index: 0)
        enc.setBuffer(denomsBuf, offset: 0, index: 1)
        var nVal = UInt32(n)
        enc.setBytes(&nVal, length: 4, index: 2)
        let tg = min(256, Int(denomPipeline.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        // Read back denominators — direct memory bind, no per-element reconstruction
        let denomFrPtr = denomsBuf.contents().bindMemory(to: Fr.self, capacity: n)
        let denoms = Array(UnsafeBufferPointer(start: denomFrPtr, count: n))

        // Phase 2: CPU batch inversion via C kernel
        var invDenoms = [Fr](repeating: .zero, count: n)
        denoms.withUnsafeBytes { dBuf in
            invDenoms.withUnsafeMutableBytes { iBuf in
                bn254_fr_batch_inverse(
                    dBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(n),
                    iBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
            }
        }

        // Phase 3: CPU accumulate Lagrange basis polynomials
        // For each i, compute basis polynomial L_i(x) = prod_{j!=i}(x - x_j) / denom_i
        // then add values[i] * invDenoms[i] * [polynomial with roots at all x_j, j!=i]
        return accumulateLagrangeBasis(points: points, values: values, invDenoms: invDenoms)
    }

    /// Accumulate Lagrange basis polynomials to get final coefficients.
    /// For each i: scale = y_i / prod_{j!=i}(x_i - x_j), then add scale * prod_{j!=i}(x - x_j).
    /// The product polynomial is built incrementally in O(n) per basis.
    /// Total: O(n^2) — same as naive but with GPU-computed denominators.
    private func accumulateLagrangeBasis(points: [Fr], values: [Fr], invDenoms: [Fr]) -> [Fr] {
        let n = points.count
        var result = [Fr](repeating: .zero, count: n)

        for i in 0..<n {
            // Build basis polynomial: prod_{j!=i}(x - x_j)
            var basis = [Fr](repeating: .zero, count: n)
            basis[0] = .one
            var deg = 0

            for j in 0..<n {
                if j == i { continue }
                deg += 1
                // Multiply basis by (x - x_j): shift up and subtract x_j * old
                for d in stride(from: deg, through: 1, by: -1) {
                    basis[d] = frSub(basis[d - 1], frMul(points[j], basis[d]))
                }
                basis[0] = frSub(.zero, frMul(points[j], basis[0]))
            }

            // Scale by y_i / denom_i and accumulate
            let scale = frMul(values[i], invDenoms[i])
            for d in 0..<n {
                result[d] = frAdd(result[d], frMul(scale, basis[d]))
            }
        }
        return result
    }

    // MARK: - CPU fallback

    /// CPU Lagrange interpolation for small n. O(n^2).
    public static func cpuInterpolate(points: [Fr], values: [Fr]) -> [Fr] {
        let n = points.count
        guard n > 0, n == values.count else { return [] }
        if n == 1 { return [values[0]] }

        var result = [Fr](repeating: .zero, count: n)

        for i in 0..<n {
            var basis = [Fr](repeating: .zero, count: n)
            basis[0] = .one
            var denom = Fr.one
            var basisDeg = 0

            for j in 0..<n {
                if j == i { continue }
                denom = frMul(denom, frSub(points[i], points[j]))
                basisDeg += 1
                for d in stride(from: basisDeg, through: 1, by: -1) {
                    basis[d] = frSub(basis[d - 1], frMul(points[j], basis[d]))
                }
                basis[0] = frSub(.zero, frMul(points[j], basis[0]))
            }

            let scale = frMul(values[i], frInverse(denom))
            for d in 0..<n {
                result[d] = frAdd(result[d], frMul(scale, basis[d]))
            }
        }
        return result
    }

    // MARK: - Evaluate interpolated polynomial at a point

    /// Evaluate the interpolated polynomial at point z using barycentric formula.
    /// Avoids computing full coefficient vector — O(n) after weights are known.
    /// Returns p(z) directly.
    public func barycentricEval(points: [Fr], values: [Fr], at z: Fr) -> Fr {
        let n = points.count
        guard n > 0, n == values.count else { return .zero }

        // Compute barycentric weights w_i = 1 / prod_{j!=i}(x_i - x_j)
        var denoms = [Fr](repeating: Fr.one, count: n)
        for i in 0..<n {
            for j in 0..<n {
                if j == i { continue }
                denoms[i] = frMul(denoms[i], frSub(points[i], points[j]))
            }
        }
        var weights = [Fr](repeating: .zero, count: n)
        denoms.withUnsafeBytes { dBuf in
            weights.withUnsafeMutableBytes { wBuf in
                bn254_fr_batch_inverse(
                    dBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(n),
                    wBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
            }
        }

        // Barycentric formula: p(z) = [sum y_i * w_i / (z - x_i)] / [sum w_i / (z - x_i)]
        // Check if z equals any x_i
        for i in 0..<n {
            if points[i] == z { return values[i] }
        }

        // Batch compute diffs: diffs[i] = z - points[i]
        var diffs = [Fr](repeating: .zero, count: n)
        var zCopy = z
        points.withUnsafeBytes { pBuf in
            diffs.withUnsafeMutableBytes { dBuf in
                withUnsafeBytes(of: &zCopy) { zBuf in
                    bn254_fr_batch_scalar_sub_neon(
                        dBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        zBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n))
                }
            }
        }

        // Batch invert diffs
        var invDiffs = [Fr](repeating: .zero, count: n)
        diffs.withUnsafeBytes { dBuf in
            invDiffs.withUnsafeMutableBytes { iBuf in
                bn254_fr_batch_inverse(
                    dBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(n),
                    iBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
            }
        }

        // Batch compute wOverDiff[i] = weights[i] * invDiffs[i]
        var wOverDiff = [Fr](repeating: .zero, count: n)
        weights.withUnsafeBytes { wBuf in
            invDiffs.withUnsafeBytes { iBuf in
                wOverDiff.withUnsafeMutableBytes { oBuf in
                    bn254_fr_batch_mul_neon(
                        oBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        wBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        iBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n))
                }
            }
        }

        // Compute numer = sum(values[i] * wOverDiff[i]) via inner product
        var numerResult = Fr.zero
        values.withUnsafeBytes { vBuf in
            wOverDiff.withUnsafeBytes { wBuf in
                withUnsafeMutableBytes(of: &numerResult) { rBuf in
                    bn254_fr_inner_product(
                        vBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        wBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
                }
            }
        }

        // Compute denom = sum(wOverDiff[i]) via vector sum
        var denomResult = Fr.zero
        wOverDiff.withUnsafeBytes { wBuf in
            withUnsafeMutableBytes(of: &denomResult) { rBuf in
                bn254_fr_vector_sum(
                    wBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(n),
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
            }
        }

        return frMul(numerResult, frInverse(denomResult))
    }

    // MARK: - Helpers

    /// CPU Lagrange interpolation (private, delegates to static method).
    private func cpuLagrangeInterpolate(points: [Fr], values: [Fr]) -> [Fr] {
        return GPUInterpolationEngine.cpuInterpolate(points: points, values: values)
    }
}
