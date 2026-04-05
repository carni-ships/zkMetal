// GPU-accelerated polynomial division engine
//
// Operations:
// 1. Division by vanishing polynomial Z_H(x) = x^n - 1 in coset evaluation domain
// 2. Division by linear factor (X - a) via synthetic division
// 3. Batch division by multiple linear factors
//
// Supports BN254 Fr (256-bit Montgomery) and BabyBear (32-bit Barrett).
// CPU fallback for small inputs.

import Foundation
import Metal

// MARK: - GPUPolyDivEngine

public class GPUPolyDivEngine {
    public static let version = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // Vanishing polynomial kernels
    private let zhEvalBn254: MTLComputePipelineState
    private let zhEvalBb: MTLComputePipelineState
    private let divVanishBn254: MTLComputePipelineState
    private let divVanishBb: MTLComputePipelineState
    private let batchInvBn254: MTLComputePipelineState
    private let batchInvBb: MTLComputePipelineState

    // Linear division kernels
    private let divLinearBn254: MTLComputePipelineState
    private let divLinearBb: MTLComputePipelineState

    // Batch division kernels
    private let batchDivBn254: MTLComputePipelineState
    private let batchDivBb: MTLComputePipelineState

    /// CPU fallback threshold for linear division
    private let cpuLinearThreshold = 256
    /// CPU fallback threshold for vanishing division (domain size)
    private let cpuVanishThreshold = 1024

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        let library = try GPUPolyDivEngine.compileShaders(device: device)

        guard let fn1 = library.makeFunction(name: "poly_zh_eval_bn254"),
              let fn2 = library.makeFunction(name: "poly_zh_eval_babybear"),
              let fn3 = library.makeFunction(name: "poly_div_by_vanishing_bn254"),
              let fn4 = library.makeFunction(name: "poly_div_by_vanishing_babybear"),
              let fn5 = library.makeFunction(name: "poly_div_batch_inverse_bn254"),
              let fn6 = library.makeFunction(name: "poly_div_batch_inverse_babybear"),
              let fn7 = library.makeFunction(name: "poly_div_by_linear_bn254"),
              let fn8 = library.makeFunction(name: "poly_div_by_linear_babybear"),
              let fn9 = library.makeFunction(name: "poly_batch_div_bn254"),
              let fn10 = library.makeFunction(name: "poly_batch_div_babybear") else {
            throw MSMError.missingKernel
        }

        self.zhEvalBn254 = try device.makeComputePipelineState(function: fn1)
        self.zhEvalBb = try device.makeComputePipelineState(function: fn2)
        self.divVanishBn254 = try device.makeComputePipelineState(function: fn3)
        self.divVanishBb = try device.makeComputePipelineState(function: fn4)
        self.batchInvBn254 = try device.makeComputePipelineState(function: fn5)
        self.batchInvBb = try device.makeComputePipelineState(function: fn6)
        self.divLinearBn254 = try device.makeComputePipelineState(function: fn7)
        self.divLinearBb = try device.makeComputePipelineState(function: fn8)
        self.batchDivBn254 = try device.makeComputePipelineState(function: fn9)
        self.batchDivBb = try device.makeComputePipelineState(function: fn10)
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

        let combined = clean(frSource) + "\n" + clean(bbSource) + "\n" + clean(divSource)
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

    // MARK: - Division by Vanishing Polynomial

    /// Divide constraint polynomial evaluations by Z_H(x) = x^n - 1 on a coset domain.
    ///
    /// - Parameters:
    ///   - evals: MTLBuffer containing evaluations over coset {g * omega^i}
    ///   - logDomain: log2(evaluation domain size N)
    ///   - logSubgroup: log2(subgroup size n), where Z_H(x) = x^n - 1
    ///   - cosetGen: coset generator g (in Montgomery form for BN254, raw for BabyBear)
    ///   - field: .bn254 or .babybear
    /// - Returns: MTLBuffer containing quotient evaluations
    public func divideByVanishing(evals: MTLBuffer, logDomain: Int, logSubgroup: Int,
                                   cosetGen: [UInt32], field: FieldType) throws -> MTLBuffer {
        let domainSize = 1 << logDomain
        let elemSize = elementSize(for: field)

        guard evals.length >= domainSize * elemSize else {
            throw MSMError.invalidInput
        }

        // CPU fallback for small domains
        if domainSize < cpuVanishThreshold {
            return try cpuDivideByVanishing(evals: evals, logDomain: logDomain,
                                             logSubgroup: logSubgroup, cosetGen: cosetGen, field: field)
        }

        return try gpuDivideByVanishing(evals: evals, domainSize: domainSize, logDomain: logDomain,
                                         logSubgroup: logSubgroup, cosetGen: cosetGen, field: field)
    }

    private func gpuDivideByVanishing(evals: MTLBuffer, domainSize: Int, logDomain: Int,
                                       logSubgroup: Int,
                                       cosetGen: [UInt32], field: FieldType) throws -> MTLBuffer {
        let elemSize = elementSize(for: field)

        // Step 1: Precompute coset points g * omega^i
        let gPowersBuf = try precomputeCosetPoints(domainSize: domainSize, logDomain: logDomain,
                                                    cosetGen: cosetGen, field: field)

        // Step 2: Compute Z_H values on GPU
        let zhBuf = device.makeBuffer(length: domainSize * elemSize, options: .storageModeShared)!

        // Step 3: Batch inverse Z_H values
        let zhInvBuf = device.makeBuffer(length: domainSize * elemSize, options: .storageModeShared)!

        // Step 4: Element-wise multiply
        let outBuf = device.makeBuffer(length: domainSize * elemSize, options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var domainU32 = UInt32(domainSize)
        var subgroupLogU32 = UInt32(logSubgroup)

        // Encode Z_H evaluation
        let enc1 = cmdBuf.makeComputeCommandEncoder()!
        let zhPipeline = field == .bn254 ? zhEvalBn254 : zhEvalBb
        enc1.setComputePipelineState(zhPipeline)
        enc1.setBuffer(gPowersBuf, offset: 0, index: 0)
        enc1.setBuffer(zhBuf, offset: 0, index: 1)
        enc1.setBytes(&domainU32, length: 4, index: 2)
        enc1.setBytes(&subgroupLogU32, length: 4, index: 3)
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
        enc2.setBytes(&domainU32, length: 4, index: 2)
        enc2.dispatchThreadgroups(MTLSize(width: numChunks, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc2.endEncoding()

        // Encode element-wise division
        let enc3 = cmdBuf.makeComputeCommandEncoder()!
        let divPipeline = field == .bn254 ? divVanishBn254 : divVanishBb
        enc3.setComputePipelineState(divPipeline)
        enc3.setBuffer(evals, offset: 0, index: 0)
        enc3.setBuffer(zhInvBuf, offset: 0, index: 1)
        enc3.setBuffer(outBuf, offset: 0, index: 2)
        enc3.setBytes(&domainU32, length: 4, index: 3)
        let tg3 = min(256, Int(divPipeline.maxTotalThreadsPerThreadgroup))
        enc3.dispatchThreads(MTLSize(width: domainSize, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tg3, height: 1, depth: 1))
        enc3.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        return outBuf
    }

    /// Precompute coset points: g * omega^i for i in [0, domainSize)
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
            throw MSMError.invalidInput  // Not supported for vanishing div
        }

        return buf
    }

    // MARK: - CPU Fallback for Vanishing Division

    private func cpuDivideByVanishing(evals: MTLBuffer, logDomain: Int, logSubgroup: Int,
                                       cosetGen: [UInt32], field: FieldType) throws -> MTLBuffer {
        let domainSize = 1 << logDomain
        let elemSize = elementSize(for: field)
        let outBuf = device.makeBuffer(length: domainSize * elemSize, options: .storageModeShared)!

        switch field {
        case .bn254:
            let g = Fr(v: (cosetGen[0], cosetGen[1], cosetGen[2], cosetGen[3],
                           cosetGen[4], cosetGen[5], cosetGen[6], cosetGen[7]))
            let omega = frRootOfUnity(logN: logDomain)

            let inPtr = evals.contents().bindMemory(to: UInt32.self, capacity: domainSize * 8)
            let outPtr = outBuf.contents().bindMemory(to: UInt32.self, capacity: domainSize * 8)

            var cosetPt = g
            for i in 0..<domainSize {
                // Compute Z_H = cosetPt^n - 1
                var zh = cosetPt
                for _ in 0..<logSubgroup {
                    zh = frSqr(zh)
                }
                zh = frSub(zh, Fr.one)
                let zhInv = frInverse(zh)

                let base = i * 8
                let eval = Fr(v: (inPtr[base], inPtr[base+1], inPtr[base+2], inPtr[base+3],
                                  inPtr[base+4], inPtr[base+5], inPtr[base+6], inPtr[base+7]))
                let result = frMul(eval, zhInv)
                outPtr[base]   = result.v.0; outPtr[base+1] = result.v.1
                outPtr[base+2] = result.v.2; outPtr[base+3] = result.v.3
                outPtr[base+4] = result.v.4; outPtr[base+5] = result.v.5
                outPtr[base+6] = result.v.6; outPtr[base+7] = result.v.7

                cosetPt = frMul(cosetPt, omega)
            }

        case .babybear:
            let g = Bb(v: cosetGen[0])
            let omega = bbRootOfUnity(logN: logDomain)

            let inPtr = evals.contents().bindMemory(to: UInt32.self, capacity: domainSize)
            let outPtr = outBuf.contents().bindMemory(to: UInt32.self, capacity: domainSize)

            var cosetPt = g
            for i in 0..<domainSize {
                var zh = cosetPt
                for _ in 0..<logSubgroup {
                    zh = bbSqr(zh)
                }
                zh = bbSub(zh, Bb.one)
                let zhInv = bbInverse(zh)
                let eval = Bb(v: inPtr[i])
                outPtr[i] = bbMul(eval, zhInv).v
                cosetPt = bbMul(cosetPt, omega)
            }

        case .goldilocks:
            throw MSMError.invalidInput
        }

        return outBuf
    }

    // MARK: - Division by Linear Factor

    /// Divide polynomial by (X - root) using synthetic division.
    /// Returns (quotient coefficients, remainder).
    /// Coefficients are in ascending order: c0 + c1*x + c2*x^2 + ...
    public func divideByLinear(coeffs: [UInt32], root: [UInt32], field: FieldType) throws -> ([UInt32], [UInt32]) {
        let elemWords = elementSize(for: field) / 4
        let degree = coeffs.count / elemWords

        guard degree >= 2 else {
            throw MSMError.invalidInput
        }

        // CPU fallback for small polynomials
        if degree < cpuLinearThreshold {
            return cpuDivideByLinear(coeffs: coeffs, root: root, field: field, degree: degree)
        }

        return try gpuDivideByLinear(coeffs: coeffs, root: root, field: field, degree: degree)
    }

    private func gpuDivideByLinear(coeffs: [UInt32], root: [UInt32], field: FieldType,
                                    degree: Int) throws -> ([UInt32], [UInt32]) {
        let elemSize = elementSize(for: field)
        let elemWords = elemSize / 4

        let coeffsBuf = device.makeBuffer(bytes: coeffs, length: degree * elemSize, options: .storageModeShared)!
        let quotBuf = device.makeBuffer(length: (degree - 1) * elemSize, options: .storageModeShared)!
        let remBuf = device.makeBuffer(length: elemSize, options: .storageModeShared)!
        let rootBuf = device.makeBuffer(bytes: root, length: elemSize, options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        let pipeline = field == .bn254 ? divLinearBn254 : divLinearBb
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(coeffsBuf, offset: 0, index: 0)
        enc.setBuffer(quotBuf, offset: 0, index: 1)
        enc.setBuffer(remBuf, offset: 0, index: 2)
        enc.setBuffer(rootBuf, offset: 0, index: 3)
        var deg = UInt32(degree)
        var numPolys: UInt32 = 1
        enc.setBytes(&deg, length: 4, index: 4)
        enc.setBytes(&numPolys, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let quotWords = (degree - 1) * elemWords
        let qPtr = quotBuf.contents().bindMemory(to: UInt32.self, capacity: quotWords)
        let quotient = Array(UnsafeBufferPointer(start: qPtr, count: quotWords))

        let rPtr = remBuf.contents().bindMemory(to: UInt32.self, capacity: elemWords)
        let remainder = Array(UnsafeBufferPointer(start: rPtr, count: elemWords))

        return (quotient, remainder)
    }

    // MARK: - CPU Fallback for Linear Division

    private func cpuDivideByLinear(coeffs: [UInt32], root: [UInt32], field: FieldType,
                                    degree: Int) -> ([UInt32], [UInt32]) {
        switch field {
        case .bn254:
            return cpuDivLinearBn254(coeffs: coeffs, root: root, degree: degree)
        case .babybear:
            return cpuDivLinearBb(coeffs: coeffs, root: root, degree: degree)
        case .goldilocks:
            fatalError("Goldilocks not supported for linear division")
        }
    }

    private func cpuDivLinearBn254(coeffs: [UInt32], root: [UInt32], degree: Int) -> ([UInt32], [UInt32]) {
        let r = Fr(v: (root[0], root[1], root[2], root[3],
                       root[4], root[5], root[6], root[7]))

        // Convert to Fr array
        var c = [Fr]()
        c.reserveCapacity(degree)
        for i in 0..<degree {
            let b = i * 8
            c.append(Fr(v: (coeffs[b], coeffs[b+1], coeffs[b+2], coeffs[b+3],
                            coeffs[b+4], coeffs[b+5], coeffs[b+6], coeffs[b+7])))
        }

        // Synthetic division
        var carry = c[degree - 1]
        var quotient = [Fr](repeating: Fr.zero, count: degree - 1)
        quotient[degree - 2] = carry

        for k in stride(from: degree - 2, to: 0, by: -1) {
            carry = frAdd(c[k], frMul(r, carry))
            quotient[k - 1] = carry
        }

        let remainder = frAdd(c[0], frMul(r, carry))

        // Convert back
        var qOut = [UInt32]()
        qOut.reserveCapacity((degree - 1) * 8)
        for q in quotient {
            qOut.append(q.v.0); qOut.append(q.v.1); qOut.append(q.v.2); qOut.append(q.v.3)
            qOut.append(q.v.4); qOut.append(q.v.5); qOut.append(q.v.6); qOut.append(q.v.7)
        }

        let rOut: [UInt32] = [remainder.v.0, remainder.v.1, remainder.v.2, remainder.v.3,
                               remainder.v.4, remainder.v.5, remainder.v.6, remainder.v.7]
        return (qOut, rOut)
    }

    private func cpuDivLinearBb(coeffs: [UInt32], root: [UInt32], degree: Int) -> ([UInt32], [UInt32]) {
        let r = Bb(v: root[0])

        var carry = Bb(v: coeffs[degree - 1])
        var quotient = [UInt32](repeating: 0, count: degree - 1)
        quotient[degree - 2] = carry.v

        for k in stride(from: degree - 2, to: 0, by: -1) {
            carry = bbAdd(Bb(v: coeffs[k]), bbMul(r, carry))
            quotient[k - 1] = carry.v
        }

        let remainder = bbAdd(Bb(v: coeffs[0]), bbMul(r, carry))
        return (quotient, [remainder.v])
    }

    // MARK: - Batch Division by Multiple Linear Factors

    /// Divide one polynomial by multiple linear factors (X - r_j) simultaneously.
    /// Returns array of (quotient, remainder) pairs, one per root.
    public func batchDivideByLinear(coeffs: [UInt32], roots: [[UInt32]],
                                    field: FieldType) throws -> [([UInt32], [UInt32])] {
        let elemWords = elementSize(for: field) / 4
        let degree = coeffs.count / elemWords
        let numRoots = roots.count

        guard degree >= 2 && numRoots >= 1 else {
            throw MSMError.invalidInput
        }

        // CPU fallback for small inputs
        if degree < cpuLinearThreshold || numRoots < 4 {
            return roots.map { root in
                cpuDivideByLinear(coeffs: coeffs, root: root, field: field, degree: degree)
            }
        }

        return try gpuBatchDivideByLinear(coeffs: coeffs, roots: roots, field: field,
                                           degree: degree, numRoots: numRoots)
    }

    private func gpuBatchDivideByLinear(coeffs: [UInt32], roots: [[UInt32]], field: FieldType,
                                         degree: Int, numRoots: Int) throws -> [([UInt32], [UInt32])] {
        let elemSize = elementSize(for: field)
        let elemWords = elemSize / 4

        let coeffsBuf = device.makeBuffer(bytes: coeffs, length: degree * elemSize, options: .storageModeShared)!
        let quotBuf = device.makeBuffer(length: numRoots * (degree - 1) * elemSize, options: .storageModeShared)!
        let remBuf = device.makeBuffer(length: numRoots * elemSize, options: .storageModeShared)!

        // Pack roots contiguously
        var packedRoots = [UInt32]()
        packedRoots.reserveCapacity(numRoots * elemWords)
        for r in roots { packedRoots.append(contentsOf: r) }
        let rootsBuf = device.makeBuffer(bytes: packedRoots, length: numRoots * elemSize, options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        let pipeline = field == .bn254 ? batchDivBn254 : batchDivBb
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(coeffsBuf, offset: 0, index: 0)
        enc.setBuffer(quotBuf, offset: 0, index: 1)
        enc.setBuffer(remBuf, offset: 0, index: 2)
        enc.setBuffer(rootsBuf, offset: 0, index: 3)
        var deg = UInt32(degree)
        var nRoots = UInt32(numRoots)
        enc.setBytes(&deg, length: 4, index: 4)
        enc.setBytes(&nRoots, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: numRoots, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        // Unpack results
        let quotWords = (degree - 1) * elemWords
        let qPtr = quotBuf.contents().bindMemory(to: UInt32.self, capacity: numRoots * quotWords)
        let rPtr = remBuf.contents().bindMemory(to: UInt32.self, capacity: numRoots * elemWords)

        var results = [([UInt32], [UInt32])]()
        results.reserveCapacity(numRoots)
        for i in 0..<numRoots {
            let q = Array(UnsafeBufferPointer(start: qPtr + i * quotWords, count: quotWords))
            let r = Array(UnsafeBufferPointer(start: rPtr + i * elemWords, count: elemWords))
            results.append((q, r))
        }
        return results
    }
}
