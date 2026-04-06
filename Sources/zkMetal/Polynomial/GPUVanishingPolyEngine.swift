// GPU-accelerated vanishing polynomial engine
//
// Operations:
// 1. Vanishing polynomial Z_H(X) = X^n - 1 evaluation for multiplicative subgroups
// 2. Evaluation of Z_H over coset domains (batch GPU-accelerated)
// 3. Polynomial division by vanishing polynomial (coefficient form)
// 4. Sparse vanishing polynomial for custom domains: prod(X - d_i)
// 5. Batch evaluation and batch division pipelines
//
// Supports BN254 Fr (256-bit Montgomery) and BabyBear (32-bit Barrett).
// CPU fallback for small inputs.

import Foundation
import Metal

// MARK: - GPUVanishingPolyEngine

public class GPUVanishingPolyEngine {
    public static let version = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // Vanishing eval kernels
    private let zhEvalBn254: MTLComputePipelineState
    private let zhEvalBb: MTLComputePipelineState

    // Batch inverse kernels
    private let batchInvBn254: MTLComputePipelineState
    private let batchInvBb: MTLComputePipelineState

    // Element-wise multiply kernels (for division in eval domain)
    private let elemMulBn254: MTLComputePipelineState
    private let elemMulBb: MTLComputePipelineState

    /// CPU fallback threshold
    private let cpuThreshold = 512

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        let library = try GPUVanishingPolyEngine.compileShaders(device: device)

        guard let fn1 = library.makeFunction(name: "vanishing_zh_eval_bn254"),
              let fn2 = library.makeFunction(name: "vanishing_zh_eval_babybear"),
              let fn3 = library.makeFunction(name: "vanishing_batch_inv_bn254"),
              let fn4 = library.makeFunction(name: "vanishing_batch_inv_babybear"),
              let fn5 = library.makeFunction(name: "vanishing_elem_mul_bn254"),
              let fn6 = library.makeFunction(name: "vanishing_elem_mul_babybear") else {
            throw MSMError.missingKernel
        }

        self.zhEvalBn254 = try device.makeComputePipelineState(function: fn1)
        self.zhEvalBb = try device.makeComputePipelineState(function: fn2)
        self.batchInvBn254 = try device.makeComputePipelineState(function: fn3)
        self.batchInvBb = try device.makeComputePipelineState(function: fn4)
        self.elemMulBn254 = try device.makeComputePipelineState(function: fn5)
        self.elemMulBb = try device.makeComputePipelineState(function: fn6)
    }

    // MARK: - Shader Compilation

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let frSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let bbSource = try String(contentsOfFile: shaderDir + "/fields/babybear.metal", encoding: .utf8)
        let divSource = try String(contentsOfFile: shaderDir + "/poly/poly_division.metal", encoding: .utf8)

        func clean(_ src: String) -> String {
            src.split(separator: "\n")
                .filter { !$0.contains("#include") && !$0.contains("#ifndef") &&
                         !$0.contains("#define") && !$0.contains("#endif") }
                .joined(separator: "\n")
        }

        // Build inline kernels for vanishing poly operations
        let vanishingKernels = """

        // --- Vanishing polynomial Z_H evaluation: BN254 ---
        kernel void vanishing_zh_eval_bn254(
            device const Fr* points        [[buffer(0)]],
            device Fr* zh_out               [[buffer(1)]],
            constant uint& count            [[buffer(2)]],
            constant uint& subgroup_log     [[buffer(3)]],
            uint gid                        [[thread_position_in_grid]]
        ) {
            if (gid >= count) return;
            Fr x = points[gid];
            for (uint i = 0; i < subgroup_log; i++) {
                x = fr_sqr(x);
            }
            zh_out[gid] = fr_sub(x, fr_one());
        }

        // --- Vanishing polynomial Z_H evaluation: BabyBear ---
        kernel void vanishing_zh_eval_babybear(
            device const uint* points      [[buffer(0)]],
            device uint* zh_out            [[buffer(1)]],
            constant uint& count           [[buffer(2)]],
            constant uint& subgroup_log    [[buffer(3)]],
            uint gid                       [[thread_position_in_grid]]
        ) {
            if (gid >= count) return;
            uint x = points[gid];
            for (uint i = 0; i < subgroup_log; i++) {
                x = bb_mul(x, x);
            }
            zh_out[gid] = bb_sub(x, 1u);
        }

        // --- Batch inverse (Montgomery's trick): BN254 ---
        // Each threadgroup processes one chunk
        kernel void vanishing_batch_inv_bn254(
            device const Fr* input         [[buffer(0)]],
            device Fr* output              [[buffer(1)]],
            constant uint& count           [[buffer(2)]],
            uint gid                       [[threadgroup_position_in_grid]]
        ) {
            const uint CHUNK = 256;
            uint start = gid * CHUNK;
            uint end = min(start + CHUNK, count);
            if (start >= count) return;

            Fr prefix[256];
            prefix[0] = input[start];
            for (uint i = 1; i < end - start; i++) {
                prefix[i] = fr_mul(prefix[i-1], input[start + i]);
            }

            Fr inv = fr_inverse(prefix[end - start - 1]);
            for (uint i = end - start; i > 1; i--) {
                output[start + i - 1] = fr_mul(inv, prefix[i - 2]);
                inv = fr_mul(inv, input[start + i - 1]);
            }
            output[start] = inv;
        }

        // --- Batch inverse: BabyBear ---
        kernel void vanishing_batch_inv_babybear(
            device const uint* input       [[buffer(0)]],
            device uint* output            [[buffer(1)]],
            constant uint& count           [[buffer(2)]],
            uint gid                       [[threadgroup_position_in_grid]]
        ) {
            const uint CHUNK = 1024;
            uint start = gid * CHUNK;
            uint end = min(start + CHUNK, count);
            if (start >= count) return;

            uint prefix[1024];
            prefix[0] = input[start];
            for (uint i = 1; i < end - start; i++) {
                prefix[i] = bb_mul(prefix[i-1], input[start + i]);
            }

            uint inv = bb_inverse(prefix[end - start - 1]);
            for (uint i = end - start; i > 1; i--) {
                output[start + i - 1] = bb_mul(inv, prefix[i - 2]);
                inv = bb_mul(inv, input[start + i - 1]);
            }
            output[start] = inv;
        }

        // --- Element-wise multiply: BN254 ---
        kernel void vanishing_elem_mul_bn254(
            device const Fr* a             [[buffer(0)]],
            device const Fr* b             [[buffer(1)]],
            device Fr* out                 [[buffer(2)]],
            constant uint& count           [[buffer(3)]],
            uint gid                       [[thread_position_in_grid]]
        ) {
            if (gid >= count) return;
            out[gid] = fr_mul(a[gid], b[gid]);
        }

        // --- Element-wise multiply: BabyBear ---
        kernel void vanishing_elem_mul_babybear(
            device const uint* a           [[buffer(0)]],
            device const uint* b           [[buffer(1)]],
            device uint* out               [[buffer(2)]],
            constant uint& count           [[buffer(3)]],
            uint gid                       [[thread_position_in_grid]]
        ) {
            if (gid >= count) return;
            out[gid] = bb_mul(a[gid], b[gid]);
        }
        """

        let combined = clean(frSource) + "\n" + clean(bbSource) + "\n" +
                        clean(divSource) + "\n" + vanishingKernels
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    private static func findShaderDir() -> String {
        let execDir = (CommandLine.arguments[0] as NSString).deletingLastPathComponent
        for bundle in Bundle.allBundles {
            if let url = bundle.url(forResource: "Shaders", withExtension: nil) {
                if FileManager.default.fileExists(atPath: url.appendingPathComponent("fields/bn254_fr.metal").path) {
                    return url.path
                }
            }
        }
        for path in ["\(execDir)/../Sources/Shaders", "./Sources/Shaders"] {
            if FileManager.default.fileExists(atPath: "\(path)/fields/bn254_fr.metal") { return path }
        }
        return "./Sources/Shaders"
    }

    // MARK: - Element size

    private func elementSize(for field: FieldType) -> Int {
        switch field {
        case .bn254: return 32
        case .babybear: return 4
        case .goldilocks: return 8
        }
    }

    // MARK: - Evaluate Z_H(x) = x^n - 1

    /// Evaluate the vanishing polynomial Z_H(x) = x^n - 1 at a single point.
    /// n = 2^logSubgroup (size of the multiplicative subgroup).
    ///
    /// - Parameters:
    ///   - point: field element in Montgomery form (BN254) or raw (BabyBear)
    ///   - logSubgroup: log2 of subgroup size n
    ///   - field: .bn254 or .babybear
    /// - Returns: Z_H(point) as [UInt32] words
    public func evaluateZH(point: [UInt32], logSubgroup: Int, field: FieldType) -> [UInt32] {
        switch field {
        case .bn254:
            return evaluateZHBn254(point: point, logSubgroup: logSubgroup)
        case .babybear:
            return evaluateZHBb(point: point, logSubgroup: logSubgroup)
        case .goldilocks:
            fatalError("Goldilocks not supported for vanishing poly")
        }
    }

    private func evaluateZHBn254(point: [UInt32], logSubgroup: Int) -> [UInt32] {
        var x = Fr(v: (point[0], point[1], point[2], point[3],
                       point[4], point[5], point[6], point[7]))
        // x^n via repeated squaring where n = 2^logSubgroup
        for _ in 0..<logSubgroup {
            x = frSqr(x)
        }
        let result = frSub(x, Fr.one)
        return [result.v.0, result.v.1, result.v.2, result.v.3,
                result.v.4, result.v.5, result.v.6, result.v.7]
    }

    private func evaluateZHBb(point: [UInt32], logSubgroup: Int) -> [UInt32] {
        var x = Bb(v: point[0])
        for _ in 0..<logSubgroup {
            x = bbSqr(x)
        }
        let result = bbSub(x, Bb.one)
        return [result.v]
    }

    // MARK: - Batch Evaluate Z_H over Coset Domain (GPU)

    /// Batch evaluate Z_H(x) = x^n - 1 over a coset domain {g * omega^i}.
    /// Returns array of Z_H values, one per domain point.
    ///
    /// - Parameters:
    ///   - logDomain: log2(domain size N)
    ///   - logSubgroup: log2(subgroup size n)
    ///   - cosetGen: coset generator g (Montgomery form for BN254, raw for BabyBear)
    ///   - field: .bn254 or .babybear
    /// - Returns: array of Z_H evaluations as flat [UInt32]
    public func batchEvaluateZH(logDomain: Int, logSubgroup: Int,
                                 cosetGen: [UInt32], field: FieldType) throws -> [UInt32] {
        let domainSize = 1 << logDomain

        if domainSize < cpuThreshold {
            return cpuBatchEvaluateZH(logDomain: logDomain, logSubgroup: logSubgroup,
                                       cosetGen: cosetGen, field: field)
        }

        return try gpuBatchEvaluateZH(domainSize: domainSize, logDomain: logDomain,
                                       logSubgroup: logSubgroup, cosetGen: cosetGen, field: field)
    }

    private func gpuBatchEvaluateZH(domainSize: Int, logDomain: Int, logSubgroup: Int,
                                     cosetGen: [UInt32], field: FieldType) throws -> [UInt32] {
        let elemSize = elementSize(for: field)
        let elemWords = elemSize / 4

        // Precompute coset points on CPU
        let pointsBuf = try precomputeCosetPoints(domainSize: domainSize, logDomain: logDomain,
                                                   cosetGen: cosetGen, field: field)

        let zhBuf = device.makeBuffer(length: domainSize * elemSize, options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var countU32 = UInt32(domainSize)
        var logSubU32 = UInt32(logSubgroup)

        let enc = cmdBuf.makeComputeCommandEncoder()!
        let pipeline = field == .bn254 ? zhEvalBn254 : zhEvalBb
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(pointsBuf, offset: 0, index: 0)
        enc.setBuffer(zhBuf, offset: 0, index: 1)
        enc.setBytes(&countU32, length: 4, index: 2)
        enc.setBytes(&logSubU32, length: 4, index: 3)
        let tg = min(256, Int(pipeline.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: domainSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = zhBuf.contents().bindMemory(to: UInt32.self, capacity: domainSize * elemWords)
        return Array(UnsafeBufferPointer(start: ptr, count: domainSize * elemWords))
    }

    private func cpuBatchEvaluateZH(logDomain: Int, logSubgroup: Int,
                                     cosetGen: [UInt32], field: FieldType) -> [UInt32] {
        let domainSize = 1 << logDomain

        switch field {
        case .bn254:
            let g = Fr(v: (cosetGen[0], cosetGen[1], cosetGen[2], cosetGen[3],
                           cosetGen[4], cosetGen[5], cosetGen[6], cosetGen[7]))
            let omega = frRootOfUnity(logN: logDomain)
            var result = [UInt32]()
            result.reserveCapacity(domainSize * 8)
            var current = g
            for _ in 0..<domainSize {
                var x = current
                for _ in 0..<logSubgroup {
                    x = frSqr(x)
                }
                let zh = frSub(x, Fr.one)
                result.append(contentsOf: [zh.v.0, zh.v.1, zh.v.2, zh.v.3,
                                           zh.v.4, zh.v.5, zh.v.6, zh.v.7])
                current = frMul(current, omega)
            }
            return result

        case .babybear:
            let g = Bb(v: cosetGen[0])
            let omega = bbRootOfUnity(logN: logDomain)
            var result = [UInt32]()
            result.reserveCapacity(domainSize)
            var current = g
            for _ in 0..<domainSize {
                var x = current
                for _ in 0..<logSubgroup {
                    x = bbSqr(x)
                }
                let zh = bbSub(x, Bb.one)
                result.append(zh.v)
                current = bbMul(current, omega)
            }
            return result

        case .goldilocks:
            fatalError("Goldilocks not supported for vanishing poly")
        }
    }

    // MARK: - Division by Vanishing Polynomial (Evaluation Domain)

    /// Divide evaluations by Z_H on a coset domain.
    /// Given f(x_i) for x_i in coset, returns f(x_i) / Z_H(x_i).
    ///
    /// - Parameters:
    ///   - evals: flat [UInt32] of polynomial evaluations over coset
    ///   - logDomain: log2(domain size)
    ///   - logSubgroup: log2(subgroup size)
    ///   - cosetGen: coset generator (Montgomery/raw)
    ///   - field: .bn254 or .babybear
    /// - Returns: quotient evaluations as flat [UInt32]
    public func divideByVanishingEval(evals: [UInt32], logDomain: Int, logSubgroup: Int,
                                       cosetGen: [UInt32], field: FieldType) throws -> [UInt32] {
        let domainSize = 1 << logDomain
        let elemWords = elementSize(for: field) / 4

        guard evals.count >= domainSize * elemWords else {
            throw MSMError.invalidInput
        }

        if domainSize < cpuThreshold {
            return cpuDivideByVanishingEval(evals: evals, logDomain: logDomain,
                                             logSubgroup: logSubgroup, cosetGen: cosetGen, field: field)
        }

        return try gpuDivideByVanishingEval(evals: evals, domainSize: domainSize, logDomain: logDomain,
                                             logSubgroup: logSubgroup, cosetGen: cosetGen, field: field)
    }

    private func gpuDivideByVanishingEval(evals: [UInt32], domainSize: Int, logDomain: Int,
                                           logSubgroup: Int, cosetGen: [UInt32],
                                           field: FieldType) throws -> [UInt32] {
        let elemSize = elementSize(for: field)
        let elemWords = elemSize / 4

        // Step 1: Precompute coset points
        let pointsBuf = try precomputeCosetPoints(domainSize: domainSize, logDomain: logDomain,
                                                   cosetGen: cosetGen, field: field)

        // Step 2: Evaluate Z_H
        let zhBuf = device.makeBuffer(length: domainSize * elemSize, options: .storageModeShared)!

        // Step 3: Batch inverse
        let zhInvBuf = device.makeBuffer(length: domainSize * elemSize, options: .storageModeShared)!

        // Step 4: Multiply evals * zhInv
        let evalsBuf = device.makeBuffer(bytes: evals, length: domainSize * elemSize, options: .storageModeShared)!
        let outBuf = device.makeBuffer(length: domainSize * elemSize, options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var countU32 = UInt32(domainSize)
        var logSubU32 = UInt32(logSubgroup)

        // Encode Z_H evaluation
        let enc1 = cmdBuf.makeComputeCommandEncoder()!
        let zhPipeline = field == .bn254 ? zhEvalBn254 : zhEvalBb
        enc1.setComputePipelineState(zhPipeline)
        enc1.setBuffer(pointsBuf, offset: 0, index: 0)
        enc1.setBuffer(zhBuf, offset: 0, index: 1)
        enc1.setBytes(&countU32, length: 4, index: 2)
        enc1.setBytes(&logSubU32, length: 4, index: 3)
        let tg1 = min(256, Int(zhPipeline.maxTotalThreadsPerThreadgroup))
        enc1.dispatchThreads(MTLSize(width: domainSize, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tg1, height: 1, depth: 1))
        enc1.endEncoding()

        // Encode batch inverse
        let enc2 = cmdBuf.makeComputeCommandEncoder()!
        let invPipeline = field == .bn254 ? batchInvBn254 : batchInvBb
        let chunkSize = field == .bn254 ? 256 : 1024
        let numChunks = (domainSize + chunkSize - 1) / chunkSize
        enc2.setComputePipelineState(invPipeline)
        enc2.setBuffer(zhBuf, offset: 0, index: 0)
        enc2.setBuffer(zhInvBuf, offset: 0, index: 1)
        enc2.setBytes(&countU32, length: 4, index: 2)
        enc2.dispatchThreadgroups(MTLSize(width: numChunks, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc2.endEncoding()

        // Encode element-wise multiply
        let enc3 = cmdBuf.makeComputeCommandEncoder()!
        let mulPipeline = field == .bn254 ? elemMulBn254 : elemMulBb
        enc3.setComputePipelineState(mulPipeline)
        enc3.setBuffer(evalsBuf, offset: 0, index: 0)
        enc3.setBuffer(zhInvBuf, offset: 0, index: 1)
        enc3.setBuffer(outBuf, offset: 0, index: 2)
        enc3.setBytes(&countU32, length: 4, index: 3)
        let tg3 = min(256, Int(mulPipeline.maxTotalThreadsPerThreadgroup))
        enc3.dispatchThreads(MTLSize(width: domainSize, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tg3, height: 1, depth: 1))
        enc3.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = outBuf.contents().bindMemory(to: UInt32.self, capacity: domainSize * elemWords)
        return Array(UnsafeBufferPointer(start: ptr, count: domainSize * elemWords))
    }

    private func cpuDivideByVanishingEval(evals: [UInt32], logDomain: Int, logSubgroup: Int,
                                            cosetGen: [UInt32], field: FieldType) -> [UInt32] {
        let domainSize = 1 << logDomain

        switch field {
        case .bn254:
            let g = Fr(v: (cosetGen[0], cosetGen[1], cosetGen[2], cosetGen[3],
                           cosetGen[4], cosetGen[5], cosetGen[6], cosetGen[7]))
            let omega = frRootOfUnity(logN: logDomain)
            var result = [UInt32]()
            result.reserveCapacity(domainSize * 8)
            var current = g
            for i in 0..<domainSize {
                var x = current
                for _ in 0..<logSubgroup {
                    x = frSqr(x)
                }
                let zh = frSub(x, Fr.one)
                let zhInv = frInverse(zh)
                let base = i * 8
                let eval = Fr(v: (evals[base], evals[base+1], evals[base+2], evals[base+3],
                                  evals[base+4], evals[base+5], evals[base+6], evals[base+7]))
                let q = frMul(eval, zhInv)
                result.append(contentsOf: [q.v.0, q.v.1, q.v.2, q.v.3,
                                           q.v.4, q.v.5, q.v.6, q.v.7])
                current = frMul(current, omega)
            }
            return result

        case .babybear:
            let g = Bb(v: cosetGen[0])
            let omega = bbRootOfUnity(logN: logDomain)
            var result = [UInt32]()
            result.reserveCapacity(domainSize)
            var current = g
            for i in 0..<domainSize {
                var x = current
                for _ in 0..<logSubgroup {
                    x = bbSqr(x)
                }
                let zh = bbSub(x, Bb.one)
                let zhInv = bbInverse(zh)
                let eval = Bb(v: evals[i])
                result.append(bbMul(eval, zhInv).v)
                current = bbMul(current, omega)
            }
            return result

        case .goldilocks:
            fatalError("Goldilocks not supported for vanishing poly")
        }
    }

    // MARK: - Division by Vanishing Polynomial (Coefficient Form)

    /// Divide a polynomial f(x) by Z_H(x) = x^n - 1 in coefficient form.
    /// If f(x) = Z_H(x) * q(x) + r(x), returns (quotient, remainder).
    /// f has degree >= n, quotient has degree deg(f) - n, remainder has degree < n.
    ///
    /// - Parameters:
    ///   - coeffs: coefficients of f in ascending order [c0, c1, ..., c_d]
    ///   - logSubgroup: log2(subgroup size n) where Z_H(x) = x^n - 1
    ///   - field: .bn254 or .babybear
    /// - Returns: (quotient coefficients, remainder coefficients) both ascending
    public func divideByVanishingCoeff(coeffs: [UInt32], logSubgroup: Int,
                                        field: FieldType) throws -> ([UInt32], [UInt32]) {
        let n = 1 << logSubgroup
        let elemWords = elementSize(for: field) / 4
        let numCoeffs = coeffs.count / elemWords

        guard numCoeffs > n else {
            // Polynomial degree < n means quotient is 0, remainder is the polynomial
            return ([], coeffs)
        }

        switch field {
        case .bn254:
            return divideByVanishingCoeffBn254(coeffs: coeffs, n: n, numCoeffs: numCoeffs)
        case .babybear:
            return divideByVanishingCoeffBb(coeffs: coeffs, n: n, numCoeffs: numCoeffs)
        case .goldilocks:
            throw MSMError.invalidInput
        }
    }

    /// CPU coefficient division for BN254: f(x) = (x^n - 1) * q(x) + r(x)
    /// Algorithm: process from highest degree down. q[i-n] = f[i] + q[i] (when i >= n).
    private func divideByVanishingCoeffBn254(coeffs: [UInt32], n: Int, numCoeffs: Int) -> ([UInt32], [UInt32]) {
        var c = [Fr]()
        c.reserveCapacity(numCoeffs)
        for i in 0..<numCoeffs {
            let base = i * 8
            c.append(Fr(v: (coeffs[base], coeffs[base+1], coeffs[base+2], coeffs[base+3],
                            coeffs[base+4], coeffs[base+5], coeffs[base+6], coeffs[base+7])))
        }

        let quotDeg = numCoeffs - n
        var q = [Fr](repeating: Fr.zero, count: quotDeg)

        // Division: dividing by x^n - 1 means for each coefficient from top down:
        // q[i] = c[i+n], then c[i] += q[i] (propagating the -(-1) = +1 term)
        for i in stride(from: quotDeg - 1, through: 0, by: -1) {
            q[i] = c[i + n]
            c[i] = frAdd(c[i], q[i])
        }

        // Remainder is c[0..n-1]
        var quotWords = [UInt32]()
        quotWords.reserveCapacity(quotDeg * 8)
        for i in 0..<quotDeg {
            quotWords.append(contentsOf: [q[i].v.0, q[i].v.1, q[i].v.2, q[i].v.3,
                                          q[i].v.4, q[i].v.5, q[i].v.6, q[i].v.7])
        }

        var remWords = [UInt32]()
        remWords.reserveCapacity(n * 8)
        for i in 0..<n {
            remWords.append(contentsOf: [c[i].v.0, c[i].v.1, c[i].v.2, c[i].v.3,
                                         c[i].v.4, c[i].v.5, c[i].v.6, c[i].v.7])
        }

        return (quotWords, remWords)
    }

    private func divideByVanishingCoeffBb(coeffs: [UInt32], n: Int, numCoeffs: Int) -> ([UInt32], [UInt32]) {
        var c = coeffs
        let quotDeg = numCoeffs - n
        var q = [UInt32](repeating: 0, count: quotDeg)

        for i in stride(from: quotDeg - 1, through: 0, by: -1) {
            q[i] = c[i + n]
            c[i] = bbAdd(Bb(v: c[i]), Bb(v: q[i])).v
        }

        return (q, Array(c[0..<n]))
    }

    // MARK: - Sparse Vanishing Polynomial

    /// Build the sparse vanishing polynomial V(x) = prod_{d in domain} (x - d).
    /// Returns coefficients in ascending order.
    ///
    /// - Parameters:
    ///   - domain: array of field elements [d0, d1, ..., d_{k-1}] as flat [UInt32]
    ///   - field: .bn254 or .babybear
    /// - Returns: coefficients of V(x) in ascending order (degree k polynomial)
    public func sparseVanishing(domain: [UInt32], field: FieldType) -> [UInt32] {
        switch field {
        case .bn254:
            return sparseVanishingBn254(domain: domain)
        case .babybear:
            return sparseVanishingBb(domain: domain)
        case .goldilocks:
            fatalError("Goldilocks not supported for vanishing poly")
        }
    }

    /// Build vanishing polynomial for a set of BN254 field elements.
    /// Uses iterative multiplication: start with (x - d0), multiply by (x - d_i).
    private func sparseVanishingBn254(domain: [UInt32]) -> [UInt32] {
        let k = domain.count / 8
        guard k > 0 else { return frToWords(Fr.one) }

        // Start: polynomial = (x - d0) = [-d0, 1]
        let d0 = Fr(v: (domain[0], domain[1], domain[2], domain[3],
                        domain[4], domain[5], domain[6], domain[7]))
        var poly = [frNeg(d0), Fr.one]

        for i in 1..<k {
            let base = i * 8
            let di = Fr(v: (domain[base], domain[base+1], domain[base+2], domain[base+3],
                            domain[base+4], domain[base+5], domain[base+6], domain[base+7]))
            // Multiply poly by (x - di)
            var newPoly = [Fr](repeating: Fr.zero, count: poly.count + 1)
            // poly * x
            for j in 0..<poly.count {
                newPoly[j + 1] = frAdd(newPoly[j + 1], poly[j])
            }
            // poly * (-di)
            let negDi = frNeg(di)
            for j in 0..<poly.count {
                newPoly[j] = frAdd(newPoly[j], frMul(poly[j], negDi))
            }
            poly = newPoly
        }

        var result = [UInt32]()
        result.reserveCapacity(poly.count * 8)
        for coeff in poly {
            result.append(contentsOf: frToWords(coeff))
        }
        return result
    }

    private func sparseVanishingBb(domain: [UInt32]) -> [UInt32] {
        let k = domain.count
        guard k > 0 else { return [1] }

        let p = Bb.P
        // (x - d0) = [p - d0, 1]
        var poly: [UInt32] = [bbNeg(Bb(v: domain[0])).v, 1]

        for i in 1..<k {
            let negDi = bbNeg(Bb(v: domain[i])).v
            var newPoly = [UInt32](repeating: 0, count: poly.count + 1)
            // poly * x
            for j in 0..<poly.count {
                newPoly[j + 1] = bbAdd(Bb(v: newPoly[j + 1]), Bb(v: poly[j])).v
            }
            // poly * (-di)
            for j in 0..<poly.count {
                newPoly[j] = bbAdd(Bb(v: newPoly[j]), bbMul(Bb(v: poly[j]), Bb(v: negDi))).v
            }
            poly = newPoly
        }

        return poly
    }

    // MARK: - Evaluate Sparse Vanishing at Point

    /// Evaluate V(x) = prod(x - d_i) at a single point.
    ///
    /// - Parameters:
    ///   - point: evaluation point as [UInt32]
    ///   - domain: custom domain elements as flat [UInt32]
    ///   - field: .bn254 or .babybear
    /// - Returns: V(point) as [UInt32]
    public func evaluateSparseVanishing(point: [UInt32], domain: [UInt32],
                                         field: FieldType) -> [UInt32] {
        switch field {
        case .bn254:
            let p = Fr(v: (point[0], point[1], point[2], point[3],
                           point[4], point[5], point[6], point[7]))
            let k = domain.count / 8
            var result = Fr.one
            for i in 0..<k {
                let base = i * 8
                let di = Fr(v: (domain[base], domain[base+1], domain[base+2], domain[base+3],
                                domain[base+4], domain[base+5], domain[base+6], domain[base+7]))
                result = frMul(result, frSub(p, di))
            }
            return frToWords(result)

        case .babybear:
            let p = Bb(v: point[0])
            var result = Bb.one
            for i in 0..<domain.count {
                result = bbMul(result, bbSub(p, Bb(v: domain[i])))
            }
            return [result.v]

        case .goldilocks:
            fatalError("Goldilocks not supported")
        }
    }

    // MARK: - Batch Evaluate Z_H at Multiple Points (GPU)

    /// Evaluate Z_H(x) = x^n - 1 at multiple arbitrary points (not necessarily on a coset).
    /// Uses GPU for parallelism over points.
    ///
    /// - Parameters:
    ///   - points: flat [UInt32] of evaluation points
    ///   - logSubgroup: log2(subgroup size n)
    ///   - field: .bn254 or .babybear
    /// - Returns: Z_H values at each point as flat [UInt32]
    public func batchEvaluateZHAtPoints(points: [UInt32], logSubgroup: Int,
                                         field: FieldType) throws -> [UInt32] {
        let elemWords = elementSize(for: field) / 4
        let numPoints = points.count / elemWords

        guard numPoints > 0 else { return [] }

        if numPoints < cpuThreshold {
            return cpuBatchEvaluateZHAtPoints(points: points, logSubgroup: logSubgroup,
                                               field: field, numPoints: numPoints)
        }

        return try gpuBatchEvaluateZHAtPoints(points: points, logSubgroup: logSubgroup,
                                               field: field, numPoints: numPoints)
    }

    private func gpuBatchEvaluateZHAtPoints(points: [UInt32], logSubgroup: Int,
                                             field: FieldType, numPoints: Int) throws -> [UInt32] {
        let elemSize = elementSize(for: field)
        let elemWords = elemSize / 4

        let pointsBuf = device.makeBuffer(bytes: points, length: numPoints * elemSize, options: .storageModeShared)!
        let zhBuf = device.makeBuffer(length: numPoints * elemSize, options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var countU32 = UInt32(numPoints)
        var logSubU32 = UInt32(logSubgroup)

        let enc = cmdBuf.makeComputeCommandEncoder()!
        let pipeline = field == .bn254 ? zhEvalBn254 : zhEvalBb
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(pointsBuf, offset: 0, index: 0)
        enc.setBuffer(zhBuf, offset: 0, index: 1)
        enc.setBytes(&countU32, length: 4, index: 2)
        enc.setBytes(&logSubU32, length: 4, index: 3)
        let tg = min(256, Int(pipeline.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: numPoints, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = zhBuf.contents().bindMemory(to: UInt32.self, capacity: numPoints * elemWords)
        return Array(UnsafeBufferPointer(start: ptr, count: numPoints * elemWords))
    }

    private func cpuBatchEvaluateZHAtPoints(points: [UInt32], logSubgroup: Int,
                                             field: FieldType, numPoints: Int) -> [UInt32] {
        var result = [UInt32]()
        let elemWords = elementSize(for: field) / 4
        result.reserveCapacity(numPoints * elemWords)

        for i in 0..<numPoints {
            let base = i * elemWords
            let pt = Array(points[base..<(base + elemWords)])
            let zh = evaluateZH(point: pt, logSubgroup: logSubgroup, field: field)
            result.append(contentsOf: zh)
        }
        return result
    }

    // MARK: - Batch Division at Multiple Points

    /// Divide multiple polynomial evaluations by Z_H evaluated at the same points.
    /// Equivalent to out[i] = evals[i] / Z_H(points[i]).
    ///
    /// - Parameters:
    ///   - evals: polynomial evaluation values as flat [UInt32]
    ///   - points: evaluation points as flat [UInt32]
    ///   - logSubgroup: log2(subgroup size)
    ///   - field: .bn254 or .babybear
    /// - Returns: division results as flat [UInt32]
    public func batchDivideByZH(evals: [UInt32], points: [UInt32], logSubgroup: Int,
                                 field: FieldType) throws -> [UInt32] {
        let elemWords = elementSize(for: field) / 4
        let numPoints = points.count / elemWords

        guard evals.count >= numPoints * elemWords else {
            throw MSMError.invalidInput
        }

        if numPoints < cpuThreshold {
            return cpuBatchDivideByZH(evals: evals, points: points, logSubgroup: logSubgroup,
                                       field: field, numPoints: numPoints)
        }

        return try gpuBatchDivideByZH(evals: evals, points: points, logSubgroup: logSubgroup,
                                       field: field, numPoints: numPoints)
    }

    private func gpuBatchDivideByZH(evals: [UInt32], points: [UInt32], logSubgroup: Int,
                                     field: FieldType, numPoints: Int) throws -> [UInt32] {
        let elemSize = elementSize(for: field)
        let elemWords = elemSize / 4

        // Step 1: Evaluate Z_H at all points
        let pointsBuf = device.makeBuffer(bytes: points, length: numPoints * elemSize, options: .storageModeShared)!
        let zhBuf = device.makeBuffer(length: numPoints * elemSize, options: .storageModeShared)!

        // Step 2: Batch inverse
        let zhInvBuf = device.makeBuffer(length: numPoints * elemSize, options: .storageModeShared)!

        // Step 3: Element-wise multiply
        let evalsBuf = device.makeBuffer(bytes: evals, length: numPoints * elemSize, options: .storageModeShared)!
        let outBuf = device.makeBuffer(length: numPoints * elemSize, options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var countU32 = UInt32(numPoints)
        var logSubU32 = UInt32(logSubgroup)

        // Z_H evaluation
        let enc1 = cmdBuf.makeComputeCommandEncoder()!
        let zhPipeline = field == .bn254 ? zhEvalBn254 : zhEvalBb
        enc1.setComputePipelineState(zhPipeline)
        enc1.setBuffer(pointsBuf, offset: 0, index: 0)
        enc1.setBuffer(zhBuf, offset: 0, index: 1)
        enc1.setBytes(&countU32, length: 4, index: 2)
        enc1.setBytes(&logSubU32, length: 4, index: 3)
        let tg1 = min(256, Int(zhPipeline.maxTotalThreadsPerThreadgroup))
        enc1.dispatchThreads(MTLSize(width: numPoints, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tg1, height: 1, depth: 1))
        enc1.endEncoding()

        // Batch inverse
        let enc2 = cmdBuf.makeComputeCommandEncoder()!
        let invPipeline = field == .bn254 ? batchInvBn254 : batchInvBb
        let chunkSize = field == .bn254 ? 256 : 1024
        let numChunks = (numPoints + chunkSize - 1) / chunkSize
        enc2.setComputePipelineState(invPipeline)
        enc2.setBuffer(zhBuf, offset: 0, index: 0)
        enc2.setBuffer(zhInvBuf, offset: 0, index: 1)
        enc2.setBytes(&countU32, length: 4, index: 2)
        enc2.dispatchThreadgroups(MTLSize(width: numChunks, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc2.endEncoding()

        // Element-wise multiply
        let enc3 = cmdBuf.makeComputeCommandEncoder()!
        let mulPipeline = field == .bn254 ? elemMulBn254 : elemMulBb
        enc3.setComputePipelineState(mulPipeline)
        enc3.setBuffer(evalsBuf, offset: 0, index: 0)
        enc3.setBuffer(zhInvBuf, offset: 0, index: 1)
        enc3.setBuffer(outBuf, offset: 0, index: 2)
        enc3.setBytes(&countU32, length: 4, index: 3)
        let tg3 = min(256, Int(mulPipeline.maxTotalThreadsPerThreadgroup))
        enc3.dispatchThreads(MTLSize(width: numPoints, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tg3, height: 1, depth: 1))
        enc3.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = outBuf.contents().bindMemory(to: UInt32.self, capacity: numPoints * elemWords)
        return Array(UnsafeBufferPointer(start: ptr, count: numPoints * elemWords))
    }

    private func cpuBatchDivideByZH(evals: [UInt32], points: [UInt32], logSubgroup: Int,
                                     field: FieldType, numPoints: Int) -> [UInt32] {
        let elemWords = elementSize(for: field) / 4
        var result = [UInt32]()
        result.reserveCapacity(numPoints * elemWords)

        switch field {
        case .bn254:
            for i in 0..<numPoints {
                let base = i * 8
                let pt = Fr(v: (points[base], points[base+1], points[base+2], points[base+3],
                                points[base+4], points[base+5], points[base+6], points[base+7]))
                var x = pt
                for _ in 0..<logSubgroup { x = frSqr(x) }
                let zh = frSub(x, Fr.one)
                let zhInv = frInverse(zh)
                let eval = Fr(v: (evals[base], evals[base+1], evals[base+2], evals[base+3],
                                  evals[base+4], evals[base+5], evals[base+6], evals[base+7]))
                let q = frMul(eval, zhInv)
                result.append(contentsOf: [q.v.0, q.v.1, q.v.2, q.v.3,
                                           q.v.4, q.v.5, q.v.6, q.v.7])
            }

        case .babybear:
            for i in 0..<numPoints {
                var x = Bb(v: points[i])
                for _ in 0..<logSubgroup { x = bbSqr(x) }
                let zh = bbSub(x, Bb.one)
                let zhInv = bbInverse(zh)
                let eval = Bb(v: evals[i])
                result.append(bbMul(eval, zhInv).v)
            }

        case .goldilocks:
            fatalError("Goldilocks not supported")
        }

        return result
    }

    // MARK: - Vanishing Polynomial Coefficients

    /// Return the coefficients of Z_H(x) = x^n - 1 for subgroup of size n = 2^logSubgroup.
    /// Coefficients: [-1, 0, 0, ..., 0, 1] (degree n).
    ///
    /// - Parameters:
    ///   - logSubgroup: log2(subgroup size)
    ///   - field: .bn254 or .babybear
    /// - Returns: n+1 coefficients in ascending order as flat [UInt32]
    public func vanishingCoefficients(logSubgroup: Int, field: FieldType) -> [UInt32] {
        let n = 1 << logSubgroup

        switch field {
        case .bn254:
            var result = [UInt32](repeating: 0, count: (n + 1) * 8)
            // c0 = -1 (in Montgomery form)
            let negOne = frNeg(Fr.one)
            let base0 = 0
            result[base0] = negOne.v.0; result[base0+1] = negOne.v.1
            result[base0+2] = negOne.v.2; result[base0+3] = negOne.v.3
            result[base0+4] = negOne.v.4; result[base0+5] = negOne.v.5
            result[base0+6] = negOne.v.6; result[base0+7] = negOne.v.7
            // c_n = 1
            let baseN = n * 8
            result[baseN] = Fr.one.v.0; result[baseN+1] = Fr.one.v.1
            result[baseN+2] = Fr.one.v.2; result[baseN+3] = Fr.one.v.3
            result[baseN+4] = Fr.one.v.4; result[baseN+5] = Fr.one.v.5
            result[baseN+6] = Fr.one.v.6; result[baseN+7] = Fr.one.v.7
            return result

        case .babybear:
            var result = [UInt32](repeating: 0, count: n + 1)
            result[0] = Bb.P - 1  // -1 mod P
            result[n] = 1
            return result

        case .goldilocks:
            fatalError("Goldilocks not supported")
        }
    }

    // MARK: - Check if Point is Root of Z_H

    /// Check whether Z_H(point) = 0, i.e., point^n = 1.
    /// Returns true if point is in the multiplicative subgroup of order n.
    public func isRootOfVanishing(point: [UInt32], logSubgroup: Int, field: FieldType) -> Bool {
        let zh = evaluateZH(point: point, logSubgroup: logSubgroup, field: field)
        return zh.allSatisfy { $0 == 0 }
    }

    // MARK: - Coset Generator for Non-vanishing Coset

    /// Return a coset generator g such that Z_H(g) != 0, suitable for coset evaluation.
    /// For BN254, uses the multiplicative generator raised to a power not in the subgroup.
    /// For BabyBear, uses a known non-residue.
    public func defaultCosetGenerator(logSubgroup: Int, field: FieldType) -> [UInt32] {
        switch field {
        case .bn254:
            // Use generator^1 = 5 in Montgomery form
            // This is not a root of unity for any power-of-two subgroup
            let gen = frFromInt(Fr.GENERATOR)
            return frToWords(gen)

        case .babybear:
            // BabyBear multiplicative generator
            let gen = Bb(v: 31)  // primitive root of BabyBear
            return [gen.v]

        case .goldilocks:
            fatalError("Goldilocks not supported")
        }
    }

    // MARK: - Precompute Coset Points

    private func precomputeCosetPoints(domainSize: Int, logDomain: Int,
                                        cosetGen: [UInt32], field: FieldType) throws -> MTLBuffer {
        let elemSize = elementSize(for: field)
        let buf = device.makeBuffer(length: domainSize * elemSize, options: .storageModeShared)!

        switch field {
        case .bn254:
            let g = Fr(v: (cosetGen[0], cosetGen[1], cosetGen[2], cosetGen[3],
                           cosetGen[4], cosetGen[5], cosetGen[6], cosetGen[7]))
            let omega = frRootOfUnity(logN: logDomain)
            let ptr = buf.contents().bindMemory(to: UInt32.self, capacity: domainSize * 8)
            var current = g
            for i in 0..<domainSize {
                let base = i * 8
                ptr[base]   = current.v.0; ptr[base+1] = current.v.1
                ptr[base+2] = current.v.2; ptr[base+3] = current.v.3
                ptr[base+4] = current.v.4; ptr[base+5] = current.v.5
                ptr[base+6] = current.v.6; ptr[base+7] = current.v.7
                current = frMul(current, omega)
            }

        case .babybear:
            let g = Bb(v: cosetGen[0])
            let omega = bbRootOfUnity(logN: logDomain)
            let ptr = buf.contents().bindMemory(to: UInt32.self, capacity: domainSize)
            var current = g
            for i in 0..<domainSize {
                ptr[i] = current.v
                current = bbMul(current, omega)
            }

        case .goldilocks:
            throw MSMError.invalidInput
        }

        return buf
    }

    // MARK: - Helpers

    private func frToWords(_ f: Fr) -> [UInt32] {
        [f.v.0, f.v.1, f.v.2, f.v.3, f.v.4, f.v.5, f.v.6, f.v.7]
    }

    private func frNeg(_ a: Fr) -> Fr {
        if a.isZero { return a }
        return frSub(Fr.zero, a)
    }
}
