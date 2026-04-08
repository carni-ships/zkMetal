// Coset LDE Engine — GPU-accelerated Low-Degree Extension over coset domains
//
// Coset LDE is the most time-critical operation in STARK provers.
// Given polynomial p(x) of degree < N, compute evaluations of p(x) over
// a coset domain {g*w^i : i in [0, M)} where M = blowupFactor * N.
//
// Algorithm:
//   1. iNTT(p) to recover coefficients (if input is in evaluation form)
//   2. Zero-pad coefficients from N to M = blowupFactor * N
//   3. Multiply coefficient[i] by g^i (coset shift)
//   4. Forward NTT of size M
//
// GPU path fuses steps 2+3 into a single kernel dispatch, minimizing round-trips.

import Foundation
import Metal
import NeonFieldOps

// MARK: - Field type enum for CosetLDE

public enum CosetLDEField {
    case babyBear
    case goldilocks
    case bn254Fr
    case mersenne31
}

// MARK: - CosetLDEEngine

public class CosetLDEEngine {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // GPU kernels for zero-pad + coset shift
    private let zeroPadCosetShiftBb: MTLComputePipelineState
    private let zeroPadCosetShiftGl: MTLComputePipelineState
    private let zeroPadCosetShiftFr: MTLComputePipelineState
    private let zeroPadCosetShiftM31: MTLComputePipelineState
    private let batchCosetShiftBb: MTLComputePipelineState
    private let batchCosetShiftGl: MTLComputePipelineState

    // NTT engines (lazily initialized per field)
    private var bbNTTEngine: BabyBearNTTEngine?
    private var glNTTEngine: GoldilocksNTTEngine?
    private var frNTTEngine: NTTEngine?

    // Coset generator power caches: key = (logM, field)
    private var bbCosetPowersCache: [Int: MTLBuffer] = [:]
    private var glCosetPowersCache: [Int: MTLBuffer] = [:]
    private var frCosetPowersCache: [Int: MTLBuffer] = [:]
    private var m31CosetPowersCache: [Int: MTLBuffer] = [:]

    private let tuning: TuningConfig

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue
        self.tuning = TuningManager.shared.config(device: device)

        let library = try CosetLDEEngine.compileShaders(device: device)

        guard let zeroPadCosetBb = library.makeFunction(name: "zero_pad_coset_shift_bb"),
              let zeroPadCosetGl = library.makeFunction(name: "zero_pad_coset_shift_gl"),
              let zeroPadCosetFr = library.makeFunction(name: "zero_pad_coset_shift_fr"),
              let zeroPadCosetM31 = library.makeFunction(name: "zero_pad_coset_shift_m31"),
              let batchBb = library.makeFunction(name: "batch_coset_shift_bb"),
              let batchGl = library.makeFunction(name: "batch_coset_shift_gl") else {
            throw MSMError.missingKernel
        }

        self.zeroPadCosetShiftBb = try device.makeComputePipelineState(function: zeroPadCosetBb)
        self.zeroPadCosetShiftGl = try device.makeComputePipelineState(function: zeroPadCosetGl)
        self.zeroPadCosetShiftFr = try device.makeComputePipelineState(function: zeroPadCosetFr)
        self.zeroPadCosetShiftM31 = try device.makeComputePipelineState(function: zeroPadCosetM31)
        self.batchCosetShiftBb = try device.makeComputePipelineState(function: batchBb)
        self.batchCosetShiftGl = try device.makeComputePipelineState(function: batchGl)
    }

    // MARK: - Shader compilation

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let fieldBb = try String(contentsOfFile: shaderDir + "/fields/babybear.metal", encoding: .utf8)
        let fieldGl = try String(contentsOfFile: shaderDir + "/fields/goldilocks.metal", encoding: .utf8)
        let fieldFr = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let fieldM31 = try String(contentsOfFile: shaderDir + "/fields/mersenne31.metal", encoding: .utf8)
        let cosetSrc = try String(contentsOfFile: shaderDir + "/poly/coset_lde.metal", encoding: .utf8)

        // Strip include guards and #include directives
        func clean(_ src: String) -> String {
            src.split(separator: "\n")
                .filter { !$0.contains("#include") && !$0.contains("#ifndef") &&
                         !$0.contains("#define") && !$0.contains("#endif") }
                .joined(separator: "\n")
        }

        let combined = clean(fieldBb) + "\n" + clean(fieldGl) + "\n" + clean(fieldFr) + "\n" +
                        clean(fieldM31) + "\n" + clean(cosetSrc)

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    private static func findShaderDir() -> String {
        let execPath = CommandLine.arguments[0]
        let execDir = (execPath as NSString).deletingLastPathComponent
        for bundle in Bundle.allBundles {
            if let url = bundle.url(forResource: "Shaders", withExtension: nil) {
                let path = url.appendingPathComponent("fields/babybear.metal").path
                if FileManager.default.fileExists(atPath: path) {
                    return url.path
                }
            }
        }
        let candidates = [
            "\(execDir)/../Sources/Shaders",
            "./Sources/Shaders",
        ]
        for path in candidates {
            if FileManager.default.fileExists(atPath: "\(path)/fields/babybear.metal") {
                return path
            }
        }
        return "./Sources/Shaders"
    }

    // MARK: - NTT engine accessors

    private func getBbNTTEngine() throws -> BabyBearNTTEngine {
        if let e = bbNTTEngine { return e }
        let e = try BabyBearNTTEngine()
        bbNTTEngine = e
        return e
    }

    private func getGlNTTEngine() throws -> GoldilocksNTTEngine {
        if let e = glNTTEngine { return e }
        let e = try GoldilocksNTTEngine()
        glNTTEngine = e
        return e
    }

    private func getFrNTTEngine() throws -> NTTEngine {
        if let e = frNTTEngine { return e }
        let e = try NTTEngine()
        frNTTEngine = e
        return e
    }

    // MARK: - Coset generator power precomputation

    /// Coset generator for BabyBear: g = GENERATOR (a multiplicative generator of Fr*)
    /// The coset domain is {g * omega^i} where omega is the M-th root of unity.
    /// We shift coefficients: c'[i] = c[i] * g^i before forward NTT.
    private func getBbCosetPowers(logM: Int) -> MTLBuffer {
        if let cached = bbCosetPowersCache[logM] { return cached }
        let m = 1 << logM
        var powers = [Bb](repeating: Bb.one, count: m)
        let g = Bb(v: Bb.GENERATOR)
        for i in 1..<m {
            powers[i] = bbMul(powers[i - 1], g)
        }
        let buf = device.makeBuffer(bytes: &powers, length: m * MemoryLayout<Bb>.stride, options: .storageModeShared)!
        bbCosetPowersCache[logM] = buf
        return buf
    }

    private func getGlCosetPowers(logM: Int) -> MTLBuffer {
        if let cached = glCosetPowersCache[logM] { return cached }
        let m = 1 << logM
        var powers = [Gl](repeating: Gl.one, count: m)
        let g = Gl(v: Gl.GENERATOR)
        for i in 1..<m {
            powers[i] = glMul(powers[i - 1], g)
        }
        let buf = device.makeBuffer(bytes: &powers, length: m * MemoryLayout<Gl>.stride, options: .storageModeShared)!
        glCosetPowersCache[logM] = buf
        return buf
    }

    private func getFrCosetPowers(logM: Int) -> MTLBuffer {
        if let cached = frCosetPowersCache[logM] { return cached }
        let m = 1 << logM
        // BN254 Fr multiplicative generator = 5
        let g = frFromInt(Fr.GENERATOR)
        var powers = [Fr](repeating: Fr.one, count: m)
        for i in 1..<m {
            powers[i] = frMul(powers[i - 1], g)
        }
        let buf = device.makeBuffer(bytes: &powers, length: m * MemoryLayout<Fr>.stride, options: .storageModeShared)!
        frCosetPowersCache[logM] = buf
        return buf
    }

    private func getM31CosetPowers(logM: Int) -> MTLBuffer {
        if let cached = m31CosetPowersCache[logM] { return cached }
        let m = 1 << logM
        // M31 generator: use 5 as multiplicative generator
        let g = M31(v: 5)
        var powers = [M31](repeating: M31.one, count: m)
        for i in 1..<m {
            powers[i] = m31Mul(powers[i - 1], g)
        }
        let buf = device.makeBuffer(bytes: &powers, length: m * MemoryLayout<M31>.stride, options: .storageModeShared)!
        m31CosetPowersCache[logM] = buf
        return buf
    }

    // MARK: - BabyBear Coset LDE

    /// Compute coset LDE for a BabyBear polynomial.
    /// Input: polynomial evaluations of size N (power of 2).
    /// Output: evaluations over coset domain of size blowupFactor * N.
    /// blowupFactor must be 2, 4, or 8.
    public func cosetLDE(poly: [Bb], blowupFactor: Int) throws -> [Bb] {
        precondition(blowupFactor == 2 || blowupFactor == 4 || blowupFactor == 8,
                     "blowupFactor must be 2, 4, or 8")
        let n = poly.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Polynomial size must be power of 2")
        let logN = Int(log2(Double(n)))
        let m = n * blowupFactor
        let logM = logN + Int(log2(Double(blowupFactor)))

        // CPU path for small polynomials
        if n <= 256 {
            return try cpuCosetLDEBb(poly: poly, blowupFactor: blowupFactor)
        }

        let engine = try getBbNTTEngine()

        // Step 1: iNTT to get coefficients
        let coeffs = try engine.intt(poly)

        // Step 2+3: Fused zero-pad + coset shift on GPU
        let cosetPowers = getBbCosetPowers(logM: logM)

        let inputBuf = device.makeBuffer(bytes: coeffs, length: n * MemoryLayout<Bb>.stride, options: .storageModeShared)!
        let outputBuf = device.makeBuffer(length: m * MemoryLayout<Bb>.stride, options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(zeroPadCosetShiftBb)
        enc.setBuffer(inputBuf, offset: 0, index: 0)
        enc.setBuffer(outputBuf, offset: 0, index: 1)
        enc.setBuffer(cosetPowers, offset: 0, index: 2)
        var nOrig = UInt32(n)
        var nExt = UInt32(m)
        enc.setBytes(&nOrig, length: 4, index: 3)
        enc.setBytes(&nExt, length: 4, index: 4)
        let tg = min(256, Int(zeroPadCosetShiftBb.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: m, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        // Step 4: Forward NTT of size M
        // Copy outputBuf data to NTT engine's expected format
        let ptr = outputBuf.contents().bindMemory(to: Bb.self, capacity: m)
        let shifted = Array(UnsafeBufferPointer(start: ptr, count: m))
        return try engine.ntt(shifted)
    }

    // MARK: - Goldilocks Coset LDE

    public func cosetLDE(poly: [Gl], blowupFactor: Int) throws -> [Gl] {
        precondition(blowupFactor == 2 || blowupFactor == 4 || blowupFactor == 8,
                     "blowupFactor must be 2, 4, or 8")
        let n = poly.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Polynomial size must be power of 2")
        let logN = Int(log2(Double(n)))
        let m = n * blowupFactor
        let logM = logN + Int(log2(Double(blowupFactor)))

        if n <= 256 {
            return try cpuCosetLDEGl(poly: poly, blowupFactor: blowupFactor)
        }

        let engine = try getGlNTTEngine()
        let coeffs = try engine.intt(poly)

        let cosetPowers = getGlCosetPowers(logM: logM)
        let inputBuf = device.makeBuffer(bytes: coeffs, length: n * MemoryLayout<Gl>.stride, options: .storageModeShared)!
        let outputBuf = device.makeBuffer(length: m * MemoryLayout<Gl>.stride, options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(zeroPadCosetShiftGl)
        enc.setBuffer(inputBuf, offset: 0, index: 0)
        enc.setBuffer(outputBuf, offset: 0, index: 1)
        enc.setBuffer(cosetPowers, offset: 0, index: 2)
        var nOrig = UInt32(n)
        var nExt = UInt32(m)
        enc.setBytes(&nOrig, length: 4, index: 3)
        enc.setBytes(&nExt, length: 4, index: 4)
        let tg = min(256, Int(zeroPadCosetShiftGl.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: m, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = outputBuf.contents().bindMemory(to: Gl.self, capacity: m)
        let shifted = Array(UnsafeBufferPointer(start: ptr, count: m))
        return try engine.ntt(shifted)
    }

    // MARK: - BN254 Fr Coset LDE

    public func cosetLDE(poly: [Fr], blowupFactor: Int) throws -> [Fr] {
        precondition(blowupFactor == 2 || blowupFactor == 4 || blowupFactor == 8,
                     "blowupFactor must be 2, 4, or 8")
        let n = poly.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Polynomial size must be power of 2")
        let logN = Int(log2(Double(n)))
        let m = n * blowupFactor
        let logM = logN + Int(log2(Double(blowupFactor)))

        if n <= 64 {
            return try cpuCosetLDEFr(poly: poly, blowupFactor: blowupFactor)
        }

        let engine = try getFrNTTEngine()
        let coeffs = try engine.intt(poly)

        let cosetPowers = getFrCosetPowers(logM: logM)
        let inputBuf = device.makeBuffer(bytes: coeffs, length: n * MemoryLayout<Fr>.stride, options: .storageModeShared)!
        let outputBuf = device.makeBuffer(length: m * MemoryLayout<Fr>.stride, options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(zeroPadCosetShiftFr)
        enc.setBuffer(inputBuf, offset: 0, index: 0)
        enc.setBuffer(outputBuf, offset: 0, index: 1)
        enc.setBuffer(cosetPowers, offset: 0, index: 2)
        var nOrig = UInt32(n)
        var nExt = UInt32(m)
        enc.setBytes(&nOrig, length: 4, index: 3)
        enc.setBytes(&nExt, length: 4, index: 4)
        let tg = min(256, Int(zeroPadCosetShiftFr.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: m, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = outputBuf.contents().bindMemory(to: Fr.self, capacity: m)
        let shifted = Array(UnsafeBufferPointer(start: ptr, count: m))
        return try engine.ntt(shifted)
    }

    // MARK: - Batch Coset LDE (multiple columns in one dispatch)

    /// Batch coset LDE for multiple BabyBear columns.
    /// All columns must have the same length N.
    /// Returns array of M-length evaluation arrays, one per column.
    public func batchCosetLDE(polys: [[Bb]], blowupFactor: Int) throws -> [[Bb]] {
        precondition(blowupFactor == 2 || blowupFactor == 4 || blowupFactor == 8,
                     "blowupFactor must be 2, 4, or 8")
        guard let first = polys.first else { return [] }
        let n = first.count
        let numCols = polys.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Polynomial size must be power of 2")
        for p in polys { precondition(p.count == n, "All columns must have equal length") }

        let logN = Int(log2(Double(n)))
        let m = n * blowupFactor
        let logM = logN + Int(log2(Double(blowupFactor)))
        let engine = try getBbNTTEngine()

        // Step 1: iNTT each column to get coefficients
        var allCoeffs = [[Bb]]()
        allCoeffs.reserveCapacity(numCols)
        for col in polys {
            allCoeffs.append(try engine.intt(col))
        }

        // Step 2+3: Fused zero-pad + coset shift on GPU for all columns
        let cosetPowers = getBbCosetPowers(logM: logM)

        // Pack all columns contiguously: [col0_coeff0..col0_coeffN-1, col1_coeff0..., ...]
        var packedInput = [Bb]()
        packedInput.reserveCapacity(n * numCols)
        for col in allCoeffs { packedInput.append(contentsOf: col) }

        let packedInputBuf = device.makeBuffer(bytes: &packedInput,
                                                length: n * numCols * MemoryLayout<Bb>.stride,
                                                options: .storageModeShared)!
        let packedOutputBuf = device.makeBuffer(length: m * numCols * MemoryLayout<Bb>.stride,
                                                 options: .storageModeShared)!

        // Dispatch fused zero-pad + coset shift per column
        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!

        for c in 0..<numCols {
            enc.setComputePipelineState(zeroPadCosetShiftBb)
            enc.setBuffer(packedInputBuf, offset: c * n * MemoryLayout<Bb>.stride, index: 0)
            enc.setBuffer(packedOutputBuf, offset: c * m * MemoryLayout<Bb>.stride, index: 1)
            enc.setBuffer(cosetPowers, offset: 0, index: 2)
            var nOrig = UInt32(n)
            var nExt = UInt32(m)
            enc.setBytes(&nOrig, length: 4, index: 3)
            enc.setBytes(&nExt, length: 4, index: 4)
            let tg = min(256, Int(zeroPadCosetShiftBb.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreads(MTLSize(width: m, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            if c < numCols - 1 {
                enc.memoryBarrier(scope: .buffers)
            }
        }
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        // Step 4: Forward NTT each column
        var results = [[Bb]]()
        results.reserveCapacity(numCols)
        let outPtr = packedOutputBuf.contents().bindMemory(to: Bb.self, capacity: m * numCols)
        for c in 0..<numCols {
            let colSlice = Array(UnsafeBufferPointer(start: outPtr + c * m, count: m))
            results.append(try engine.ntt(colSlice))
        }
        return results
    }

    /// Batch coset LDE for multiple Goldilocks columns.
    public func batchCosetLDE(polys: [[Gl]], blowupFactor: Int) throws -> [[Gl]] {
        precondition(blowupFactor == 2 || blowupFactor == 4 || blowupFactor == 8,
                     "blowupFactor must be 2, 4, or 8")
        guard let first = polys.first else { return [] }
        let n = first.count
        let numCols = polys.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Polynomial size must be power of 2")
        for p in polys { precondition(p.count == n, "All columns must have equal length") }

        let logN = Int(log2(Double(n)))
        let m = n * blowupFactor
        let logM = logN + Int(log2(Double(blowupFactor)))
        let engine = try getGlNTTEngine()

        var allCoeffs = [[Gl]]()
        allCoeffs.reserveCapacity(numCols)
        for col in polys {
            allCoeffs.append(try engine.intt(col))
        }

        let cosetPowers = getGlCosetPowers(logM: logM)

        var packedInput = [Gl]()
        packedInput.reserveCapacity(n * numCols)
        for col in allCoeffs { packedInput.append(contentsOf: col) }

        let packedInputBuf = device.makeBuffer(bytes: &packedInput,
                                                length: n * numCols * MemoryLayout<Gl>.stride,
                                                options: .storageModeShared)!
        let packedOutputBuf = device.makeBuffer(length: m * numCols * MemoryLayout<Gl>.stride,
                                                 options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!

        for c in 0..<numCols {
            enc.setComputePipelineState(zeroPadCosetShiftGl)
            enc.setBuffer(packedInputBuf, offset: c * n * MemoryLayout<Gl>.stride, index: 0)
            enc.setBuffer(packedOutputBuf, offset: c * m * MemoryLayout<Gl>.stride, index: 1)
            enc.setBuffer(cosetPowers, offset: 0, index: 2)
            var nOrig = UInt32(n)
            var nExt = UInt32(m)
            enc.setBytes(&nOrig, length: 4, index: 3)
            enc.setBytes(&nExt, length: 4, index: 4)
            let tg = min(256, Int(zeroPadCosetShiftGl.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreads(MTLSize(width: m, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            if c < numCols - 1 {
                enc.memoryBarrier(scope: .buffers)
            }
        }
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        var results = [[Gl]]()
        results.reserveCapacity(numCols)
        let outPtr = packedOutputBuf.contents().bindMemory(to: Gl.self, capacity: m * numCols)
        for c in 0..<numCols {
            let colSlice = Array(UnsafeBufferPointer(start: outPtr + c * m, count: m))
            results.append(try engine.ntt(colSlice))
        }
        return results
    }

    /// Batch coset LDE for multiple BN254 Fr columns.
    public func batchCosetLDE(polys: [[Fr]], blowupFactor: Int) throws -> [[Fr]] {
        precondition(blowupFactor == 2 || blowupFactor == 4 || blowupFactor == 8,
                     "blowupFactor must be 2, 4, or 8")
        guard let first = polys.first else { return [] }
        let n = first.count
        let numCols = polys.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Polynomial size must be power of 2")
        for p in polys { precondition(p.count == n, "All columns must have equal length") }

        let logN = Int(log2(Double(n)))
        let m = n * blowupFactor
        let logM = logN + Int(log2(Double(blowupFactor)))
        let engine = try getFrNTTEngine()

        var allCoeffs = [[Fr]]()
        allCoeffs.reserveCapacity(numCols)
        for col in polys {
            allCoeffs.append(try engine.intt(col))
        }

        let cosetPowers = getFrCosetPowers(logM: logM)

        var packedInput = [Fr]()
        packedInput.reserveCapacity(n * numCols)
        for col in allCoeffs { packedInput.append(contentsOf: col) }

        let packedInputBuf = device.makeBuffer(bytes: &packedInput,
                                                length: n * numCols * MemoryLayout<Fr>.stride,
                                                options: .storageModeShared)!
        let packedOutputBuf = device.makeBuffer(length: m * numCols * MemoryLayout<Fr>.stride,
                                                 options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!

        for c in 0..<numCols {
            enc.setComputePipelineState(zeroPadCosetShiftFr)
            enc.setBuffer(packedInputBuf, offset: c * n * MemoryLayout<Fr>.stride, index: 0)
            enc.setBuffer(packedOutputBuf, offset: c * m * MemoryLayout<Fr>.stride, index: 1)
            enc.setBuffer(cosetPowers, offset: 0, index: 2)
            var nOrig = UInt32(n)
            var nExt = UInt32(m)
            enc.setBytes(&nOrig, length: 4, index: 3)
            enc.setBytes(&nExt, length: 4, index: 4)
            let tg = min(256, Int(zeroPadCosetShiftFr.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreads(MTLSize(width: m, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            if c < numCols - 1 {
                enc.memoryBarrier(scope: .buffers)
            }
        }
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        var results = [[Fr]]()
        results.reserveCapacity(numCols)
        let outPtr = packedOutputBuf.contents().bindMemory(to: Fr.self, capacity: m * numCols)
        for c in 0..<numCols {
            let colSlice = Array(UnsafeBufferPointer(start: outPtr + c * m, count: m))
            results.append(try engine.ntt(colSlice))
        }
        return results
    }

    // MARK: - CPU reference implementations

    /// CPU coset LDE for BabyBear (small polynomials).
    public func cpuCosetLDEBb(poly: [Bb], blowupFactor: Int) throws -> [Bb] {
        let n = poly.count
        let logN = Int(log2(Double(n)))
        let m = n * blowupFactor
        let logM = logN + Int(log2(Double(blowupFactor)))

        // Step 1: iNTT to get coefficients
        let coeffs = BabyBearNTTEngine.cpuINTT(poly, logN: logN)

        // Step 2: Zero-pad to size M
        var padded = [Bb](repeating: Bb.zero, count: m)
        for i in 0..<n { padded[i] = coeffs[i] }

        // Step 3: Coset shift: padded[i] *= g^i
        let g = Bb(v: Bb.GENERATOR)
        var gPow = Bb.one
        for i in 0..<m {
            padded[i] = bbMul(padded[i], gPow)
            gPow = bbMul(gPow, g)
        }

        // Step 4: Forward NTT
        return BabyBearNTTEngine.cpuNTT(padded, logN: logM)
    }

    /// CPU coset LDE for Goldilocks.
    public func cpuCosetLDEGl(poly: [Gl], blowupFactor: Int) throws -> [Gl] {
        let n = poly.count
        let logN = Int(log2(Double(n)))
        let m = n * blowupFactor
        let logM = logN + Int(log2(Double(blowupFactor)))

        let coeffs = GoldilocksNTTEngine.cpuINTT(poly, logN: logN)

        var padded = [Gl](repeating: Gl.zero, count: m)
        for i in 0..<n { padded[i] = coeffs[i] }

        let g = Gl(v: Gl.GENERATOR)
        var gPow = Gl.one
        for i in 0..<m {
            padded[i] = glMul(padded[i], gPow)
            gPow = glMul(gPow, g)
        }

        return GoldilocksNTTEngine.cpuNTT(padded, logN: logM)
    }

    /// CPU coset LDE for BN254 Fr.
    public func cpuCosetLDEFr(poly: [Fr], blowupFactor: Int) throws -> [Fr] {
        let n = poly.count
        let logN = Int(log2(Double(n)))
        let m = n * blowupFactor
        let logM = logN + Int(log2(Double(blowupFactor)))

        let coeffs = NTTEngine.cpuINTT(poly, logN: logN)

        var padded = [Fr](repeating: Fr.zero, count: m)
        for i in 0..<n { padded[i] = coeffs[i] }

        let g = frFromInt(Fr.GENERATOR)
        padded.withUnsafeMutableBytes { rBuf in
            withUnsafeBytes(of: g) { gBuf in
                bn254_fr_batch_mul_powers(
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    gBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(m))
            }
        }

        return NTTEngine.cpuNTT(padded, logN: logM)
    }
}
