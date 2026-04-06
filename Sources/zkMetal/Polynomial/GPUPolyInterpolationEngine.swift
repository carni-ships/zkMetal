// GPU-accelerated polynomial interpolation engine (advanced)
//
// Provides multiple interpolation methods beyond basic Lagrange:
//
//   1. lagrangeInterpolate(points:values:)     — general Lagrange, GPU-accelerated O(n^2)
//   2. barycentricInterpolate(weights:points:values:at:) — O(n) eval with precomputed weights
//   3. batchInterpolate(points:valueSets:)     — interpolate multiple polynomials on same domain
//   4. subgroupInterpolate(evals:logN:)        — NTT-based for roots-of-unity domains
//   5. newtonInterpolate(points:values:)       — Newton form via divided differences
//
// BN254 Fr field (256-bit Montgomery). CPU fallback for small inputs (n < 64).

import Foundation
import Metal

// MARK: - GPUPolyInterpolationEngine

public final class GPUPolyInterpolationEngine {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // GPU pipelines for Lagrange denominator computation
    private let denomPipeline: MTLComputePipelineState
    private let scalePipeline: MTLComputePipelineState
    private let evalPipeline: MTLComputePipelineState

    // NTT engine for subgroup interpolation (lazy)
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

        let library = try GPUPolyInterpolationEngine.compileShaders(device: device)

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
                .filter { !$0.contains("#include") && !$0.contains("#ifndef") &&
                         !$0.contains("#define") && !$0.contains("#endif") }
                .joined(separator: "\n")
        }

        let combined = clean(fieldSrc) + "\n" + clean(interpSrc)

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    // MARK: - 1. General Lagrange Interpolation

    /// Recover polynomial coefficients from evaluation points.
    /// Given {(x_i, y_i)}, returns [c_0, ..., c_{n-1}] with p(x_i) = y_i.
    /// GPU-accelerated for n >= cpuThreshold.
    public func lagrangeInterpolate(points: [Fr], values: [Fr]) throws -> [Fr] {
        let n = points.count
        guard n == values.count, n > 0 else {
            throw MSMError.invalidInput
        }
        if n == 1 {
            return [values[0]]
        }

        if n < GPUPolyInterpolationEngine.cpuThreshold {
            return cpuLagrangeInterpolate(points: points, values: values)
        }

        return try gpuLagrangeInterpolate(points: points, values: values)
    }

    // MARK: - 2. Barycentric Interpolation

    /// Precomputed barycentric weights for a fixed set of points.
    /// w_i = 1 / prod_{j!=i}(x_i - x_j)
    /// Once computed, evaluation at any new point z is O(n).
    public struct BarycentricWeights {
        public let points: [Fr]
        public let weights: [Fr]
    }

    /// Compute barycentric weights for a set of evaluation points.
    /// GPU-accelerated denominator computation for large n.
    public func computeBarycentricWeights(points: [Fr]) throws -> BarycentricWeights {
        let n = points.count
        guard n > 0 else { throw MSMError.invalidInput }

        if n == 1 {
            return BarycentricWeights(points: points, weights: [Fr.one])
        }

        var denoms: [Fr]
        if n >= GPUPolyInterpolationEngine.cpuThreshold {
            denoms = try gpuComputeDenominators(points: points)
        } else {
            denoms = cpuComputeDenominators(points: points)
        }

        // Batch invert to get weights
        let weights = frBatchInverse(denoms)
        return BarycentricWeights(points: points, weights: weights)
    }

    /// Evaluate the interpolated polynomial at point z using precomputed barycentric weights.
    /// O(n) field operations.
    /// Formula: p(z) = [sum y_i * w_i / (z - x_i)] / [sum w_i / (z - x_i)]
    public func barycentricEval(weights: BarycentricWeights, values: [Fr], at z: Fr) -> Fr {
        let n = weights.points.count
        guard n == values.count, n > 0 else { return .zero }

        // Check if z equals any evaluation point
        for i in 0..<n {
            if weights.points[i] == z { return values[i] }
        }

        // Batch-invert all (z - x_i) denominators
        var beDiffs = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { beDiffs[i] = frSub(z, weights.points[i]) }
        var bePrefix = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n {
            bePrefix[i] = beDiffs[i - 1] == Fr.zero ? bePrefix[i - 1] : frMul(bePrefix[i - 1], beDiffs[i - 1])
        }
        let beLast = beDiffs[n - 1] == Fr.zero ? bePrefix[n - 1] : frMul(bePrefix[n - 1], beDiffs[n - 1])
        var beInv = frInverse(beLast)
        var beDiffInvs = [Fr](repeating: Fr.zero, count: n)
        for i in stride(from: n - 1, through: 0, by: -1) {
            if beDiffs[i] != Fr.zero {
                beDiffInvs[i] = frMul(beInv, bePrefix[i])
                beInv = frMul(beInv, beDiffs[i])
            }
        }

        var numer = Fr.zero
        var denom = Fr.zero
        for i in 0..<n {
            let wOverDiff = frMul(weights.weights[i], beDiffInvs[i])
            numer = frAdd(numer, frMul(values[i], wOverDiff))
            denom = frAdd(denom, wOverDiff)
        }

        return frMul(numer, frInverse(denom))
    }

    /// Evaluate the interpolated polynomial at point z without precomputed weights.
    /// Convenience that computes weights on the fly — O(n^2) for weight computation + O(n) eval.
    public func barycentricEval(points: [Fr], values: [Fr], at z: Fr) throws -> Fr {
        let bw = try computeBarycentricWeights(points: points)
        return barycentricEval(weights: bw, values: values, at: z)
    }

    // MARK: - 3. Batch Interpolation

    /// Interpolate multiple polynomials that share the same evaluation domain.
    /// Given points [x_0,...,x_{n-1}] and k value sets, returns k coefficient vectors.
    /// More efficient than separate calls because barycentric weights / denominators
    /// are computed once and reused.
    public func batchInterpolate(points: [Fr], valueSets: [[Fr]]) throws -> [[Fr]] {
        let n = points.count
        guard n > 0 else { throw MSMError.invalidInput }
        for vs in valueSets {
            guard vs.count == n else { throw MSMError.invalidInput }
        }

        if valueSets.isEmpty { return [] }

        if n == 1 {
            return valueSets.map { [$0[0]] }
        }

        // Compute inverse denominators once
        var invDenoms: [Fr]
        if n >= GPUPolyInterpolationEngine.cpuThreshold {
            let denoms = try gpuComputeDenominators(points: points)
            invDenoms = frBatchInverse(denoms)
        } else {
            let denoms = cpuComputeDenominators(points: points)
            invDenoms = frBatchInverse(denoms)
        }

        // Interpolate each value set using shared inverse denominators
        var results = [[Fr]]()
        results.reserveCapacity(valueSets.count)

        for values in valueSets {
            let coeffs = accumulateLagrangeBasis(points: points, values: values, invDenoms: invDenoms)
            results.append(coeffs)
        }

        return results
    }

    // MARK: - 4. Subgroup Interpolation (NTT-based)

    /// Fast interpolation when domain is n-th roots of unity (subgroup of Fr*).
    /// evals[i] = p(omega^i), returns polynomial coefficients via inverse NTT.
    /// n must be a power of 2, n <= 2^28 (BN254 TWO_ADICITY).
    /// This is O(n log n) vs O(n^2) for general Lagrange.
    public func subgroupInterpolate(evals: [Fr], logN: Int) throws -> [Fr] {
        let n = evals.count
        guard n > 0 && n == (1 << logN) else {
            throw MSMError.invalidInput
        }
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

    /// Convenience: auto-detect logN from array length.
    public func subgroupInterpolate(evals: [Fr]) throws -> [Fr] {
        let n = evals.count
        guard n > 0 && (n & (n - 1)) == 0 else {
            throw MSMError.invalidInput
        }
        let logN = Int(log2(Double(n)))
        return try subgroupInterpolate(evals: evals, logN: logN)
    }

    /// Evaluate the subgroup polynomial at an arbitrary point z.
    /// Uses NTT to get coefficients, then Horner evaluation.
    public func subgroupEvalAt(evals: [Fr], logN: Int, at z: Fr) throws -> Fr {
        let coeffs = try subgroupInterpolate(evals: evals, logN: logN)
        return hornerEval(coeffs, at: z)
    }

    // MARK: - 5. Newton Form Interpolation

    /// Newton form interpolation using divided differences.
    /// Returns Newton coefficients [c_0, c_1, ..., c_{n-1}] such that:
    ///   p(x) = c_0 + c_1*(x-x_0) + c_2*(x-x_0)*(x-x_1) + ...
    ///
    /// The divided difference table is computed in O(n^2) field operations.
    /// Newton form is useful when points are added incrementally.
    public func newtonInterpolate(points: [Fr], values: [Fr]) throws -> NewtonPoly {
        let n = points.count
        guard n == values.count, n > 0 else {
            throw MSMError.invalidInput
        }

        // Build divided difference table
        // dd[i][j] = f[x_i, x_{i+1}, ..., x_{i+j}]
        var dd = values

        for j in 1..<n {
            for i in stride(from: n - 1, through: j, by: -1) {
                let diff = frSub(points[i], points[i - j])
                dd[i] = frMul(frSub(dd[i], dd[i - 1]), frInverse(diff))
            }
        }

        // dd[i] now holds the i-th divided difference coefficient
        return NewtonPoly(points: points, coeffs: dd)
    }

    /// Convert Newton form to standard monomial (coefficient) form.
    /// Uses Horner-like expansion: expand from highest term down.
    public func newtonToMonomial(_ np: NewtonPoly) -> [Fr] {
        let n = np.coeffs.count
        if n == 0 { return [] }
        if n == 1 { return [np.coeffs[0]] }

        // Start from the highest coefficient and multiply by (x - x_{k-1}) going down
        var result = [Fr](repeating: .zero, count: n)
        result[0] = np.coeffs[n - 1]

        for k in stride(from: n - 2, through: 0, by: -1) {
            // Multiply current polynomial by (x - x_k): shift up and subtract x_k * old
            for d in stride(from: min(n - 1, n - 1 - k + 1), through: 1, by: -1) {
                result[d] = frSub(result[d - 1], frMul(np.points[k], result[d]))
            }
            result[0] = frSub(np.coeffs[k], frMul(np.points[k], result[0]))
        }

        return result
    }

    /// Newton polynomial: stores points and divided difference coefficients.
    public struct NewtonPoly {
        public let points: [Fr]
        public let coeffs: [Fr]

        /// Evaluate Newton polynomial at z using Horner's method.
        /// p(z) = c_0 + c_1*(z-x_0) + c_2*(z-x_0)*(z-x_1) + ...
        public func evaluate(at z: Fr) -> Fr {
            let n = coeffs.count
            if n == 0 { return .zero }

            var result = coeffs[n - 1]
            for k in stride(from: n - 2, through: 0, by: -1) {
                result = frAdd(coeffs[k], frMul(result, frSub(z, points[k])))
            }
            return result
        }

        /// Add a new point to the Newton polynomial incrementally.
        /// Returns a new NewtonPoly with degree one higher.
        public func addPoint(x: Fr, y: Fr) -> NewtonPoly {
            let n = points.count

            // Evaluate current polynomial at x
            let px = evaluate(at: x)

            // New divided difference coefficient
            // c_n = (y - p(x)) / prod_{i=0}^{n-1}(x - x_i)
            var prodDiff = Fr.one
            for i in 0..<n {
                prodDiff = frMul(prodDiff, frSub(x, points[i]))
            }
            let newCoeff = frMul(frSub(y, px), frInverse(prodDiff))

            var newPoints = points
            newPoints.append(x)
            var newCoeffs = coeffs
            newCoeffs.append(newCoeff)

            return NewtonPoly(points: newPoints, coeffs: newCoeffs)
        }
    }

    // MARK: - GPU Lagrange Interpolation (internal)

    /// GPU-accelerated Lagrange interpolation.
    /// Phase 1: GPU computes denominator products denom[i] = prod_{j!=i}(x_i - x_j)
    /// Phase 2: CPU batch-inverts denominators
    /// Phase 3: CPU accumulates Lagrange basis polynomials
    private func gpuLagrangeInterpolate(points: [Fr], values: [Fr]) throws -> [Fr] {
        let denoms = try gpuComputeDenominators(points: points)
        let invDenoms = frBatchInverse(denoms)
        return accumulateLagrangeBasis(points: points, values: values, invDenoms: invDenoms)
    }

    /// GPU compute denominator products: denom[i] = prod_{j!=i}(x_i - x_j)
    private func gpuComputeDenominators(points: [Fr]) throws -> [Fr] {
        let n = points.count
        let elemSize = 32 // 8 x UInt32 = 32 bytes per Fr

        let pointWords = flattenFr(points)

        guard let pointsBuf = device.makeBuffer(bytes: pointWords, length: n * elemSize,
                                                 options: .storageModeShared),
              let denomsBuf = device.makeBuffer(length: n * elemSize,
                                                 options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate interpolation buffers")
        }

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

        let denomPtr = denomsBuf.contents().bindMemory(to: UInt32.self, capacity: n * 8)
        var denoms = [Fr](repeating: .zero, count: n)
        for i in 0..<n {
            let base = i * 8
            denoms[i] = Fr(v: (denomPtr[base], denomPtr[base+1], denomPtr[base+2], denomPtr[base+3],
                               denomPtr[base+4], denomPtr[base+5], denomPtr[base+6], denomPtr[base+7]))
        }
        return denoms
    }

    /// CPU compute denominators for small n.
    private func cpuComputeDenominators(points: [Fr]) -> [Fr] {
        let n = points.count
        var denoms = [Fr](repeating: Fr.one, count: n)
        for i in 0..<n {
            for j in 0..<n {
                if j == i { continue }
                denoms[i] = frMul(denoms[i], frSub(points[i], points[j]))
            }
        }
        return denoms
    }

    /// Accumulate Lagrange basis polynomials to get final coefficients.
    private func accumulateLagrangeBasis(points: [Fr], values: [Fr], invDenoms: [Fr]) -> [Fr] {
        let n = points.count
        var result = [Fr](repeating: .zero, count: n)

        for i in 0..<n {
            var basis = [Fr](repeating: .zero, count: n)
            basis[0] = .one
            var deg = 0

            for j in 0..<n {
                if j == i { continue }
                deg += 1
                for d in stride(from: deg, through: 1, by: -1) {
                    basis[d] = frSub(basis[d - 1], frMul(points[j], basis[d]))
                }
                basis[0] = frSub(.zero, frMul(points[j], basis[0]))
            }

            let scale = frMul(values[i], invDenoms[i])
            for d in 0..<n {
                result[d] = frAdd(result[d], frMul(scale, basis[d]))
            }
        }
        return result
    }

    // MARK: - CPU Lagrange Interpolation (fallback)

    /// CPU Lagrange interpolation for small n. O(n^2).
    public static func cpuInterpolate(points: [Fr], values: [Fr]) -> [Fr] {
        let n = points.count
        guard n > 0, n == values.count else { return [] }
        if n == 1 { return [values[0]] }

        var result = [Fr](repeating: .zero, count: n)

        // Precompute all denominators and basis polynomials, then batch-invert
        var cpuDenoms = [Fr](repeating: Fr.one, count: n)
        var cpuBases = [[Fr]](repeating: [Fr](repeating: .zero, count: n), count: n)
        for i in 0..<n {
            cpuBases[i][0] = .one
            var basisDeg = 0
            for j in 0..<n where j != i {
                cpuDenoms[i] = frMul(cpuDenoms[i], frSub(points[i], points[j]))
                basisDeg += 1
                for d in stride(from: basisDeg, through: 1, by: -1) {
                    cpuBases[i][d] = frSub(cpuBases[i][d - 1], frMul(points[j], cpuBases[i][d]))
                }
                cpuBases[i][0] = frSub(.zero, frMul(points[j], cpuBases[i][0]))
            }
        }
        var cpuPfx = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n {
            cpuPfx[i] = cpuDenoms[i - 1] == Fr.zero ? cpuPfx[i - 1] : frMul(cpuPfx[i - 1], cpuDenoms[i - 1])
        }
        let cpuLst = cpuDenoms[n - 1] == Fr.zero ? cpuPfx[n - 1] : frMul(cpuPfx[n - 1], cpuDenoms[n - 1])
        var cpuInvR = frInverse(cpuLst)
        var cpuDenomInvs = [Fr](repeating: Fr.zero, count: n)
        for i in stride(from: n - 1, through: 0, by: -1) {
            if cpuDenoms[i] != Fr.zero {
                cpuDenomInvs[i] = frMul(cpuInvR, cpuPfx[i])
                cpuInvR = frMul(cpuInvR, cpuDenoms[i])
            }
        }

        for i in 0..<n {
            let scale = frMul(values[i], cpuDenomInvs[i])
            for d in 0..<n {
                result[d] = frAdd(result[d], frMul(scale, cpuBases[i][d]))
            }
        }
        return result
    }

    // MARK: - CPU Lagrange (private instance method)

    private func cpuLagrangeInterpolate(points: [Fr], values: [Fr]) -> [Fr] {
        return GPUPolyInterpolationEngine.cpuInterpolate(points: points, values: values)
    }

    // MARK: - Horner Evaluation

    /// Evaluate polynomial in coefficient form at point z.
    public func hornerEval(_ coeffs: [Fr], at z: Fr) -> Fr {
        if coeffs.isEmpty { return .zero }
        var result = coeffs[coeffs.count - 1]
        for i in stride(from: coeffs.count - 2, through: 0, by: -1) {
            result = frAdd(frMul(result, z), coeffs[i])
        }
        return result
    }

    // MARK: - Helpers

    /// Flatten Fr array to UInt32 array for GPU buffers.
    private func flattenFr(_ arr: [Fr]) -> [UInt32] {
        var words = [UInt32](repeating: 0, count: arr.count * 8)
        for i in 0..<arr.count {
            let base = i * 8
            words[base]   = arr[i].v.0
            words[base+1] = arr[i].v.1
            words[base+2] = arr[i].v.2
            words[base+3] = arr[i].v.3
            words[base+4] = arr[i].v.4
            words[base+5] = arr[i].v.5
            words[base+6] = arr[i].v.6
            words[base+7] = arr[i].v.7
        }
        return words
    }
}
