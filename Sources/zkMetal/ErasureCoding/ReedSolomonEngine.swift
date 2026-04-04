// Reed-Solomon Erasure Coding Engine
// Two backends:
//   1. NTT-based (BabyBear prime field) - uses existing GPU NTT for O(n log n) encode/decode
//   2. GF(2^16) matrix-based - GPU-accelerated via log/antilog tables, classic RS
//
// NTT-based encoding: interpret data as polynomial coefficients, NTT to get evaluations
// NTT-based decoding: inverse NTT from evaluations to recover coefficients
//
// GF(2^16) encoding: systematic via generator matrix (Vandermonde)
// GF(2^16) decoding: invert submatrix of Vandermonde for available shards

import Foundation
import Metal

public enum RSError: Error {
    case noGPU
    case noCommandQueue
    case noCommandBuffer
    case invalidInput
    case insufficientShards
    case gpuError(String)
    case missingKernel
}

// MARK: - NTT-Based RS Engine (BabyBear)

/// Reed-Solomon encoder/decoder using NTT over BabyBear field.
/// Data shards = k polynomial coefficients.
/// Total shards = n = k * expansionFactor evaluation points (power of 2).
/// Encoding = forward NTT; decoding = erasure recovery via INTT.
public class ReedSolomonNTTEngine {
    public static let version = Versions.reedSolomon
    public let nttEngine: BabyBearNTTEngine

    public init() throws {
        self.nttEngine = try BabyBearNTTEngine()
    }

    /// Encode k data elements into n = nextPow2(k * expansion) evaluation shards.
    /// Returns all n shards (data is embedded in frequency domain).
    public func encode(data: [Bb], expansionFactor: Int = 2) throws -> [Bb] {
        precondition(expansionFactor >= 2, "Expansion factor must be >= 2")
        let k = data.count
        let n = nextPow2(k * expansionFactor)
        precondition(n <= (1 << Bb.TWO_ADICITY), "NTT size exceeds BabyBear 2-adicity")

        // Pad data to n with zeros (higher coefficients = 0)
        var padded = [Bb](repeating: .zero, count: n)
        for i in 0..<k { padded[i] = data[i] }

        // Forward NTT: evaluates polynomial at n roots of unity
        return try nttEngine.ntt(padded)
    }

    /// Decode from any k-of-n shards back to original k data elements.
    /// shards: array of (index, value) where index is the evaluation point index (0..<n).
    /// originalK: number of original data elements.
    /// totalN: total number of shards from encoding.
    public func decode(shards: [(index: Int, value: Bb)], originalK: Int, totalN: Int) throws -> [Bb] {
        let k = shards.count
        guard k >= originalK else {
            throw RSError.insufficientShards
        }

        let logN = Int(log2(Double(totalN)))
        precondition(1 << logN == totalN, "totalN must be power of 2")

        // We have evaluations at specific roots of unity.
        // To recover, use Lagrange interpolation on k points, then read coefficients.
        // For efficiency, build full evaluation array with zeros at missing positions,
        // then use erasure decoding.

        // Simple approach: CPU Lagrange interpolation for the k points
        let omega = bbRootOfUnity(logN: logN)
        let points = shards.map { shard -> Bb in
            bbPow(omega, UInt32(shard.index))
        }
        let values = shards.map { $0.value }

        // Lagrange interpolation to get polynomial coefficients
        let coeffs = lagrangeInterpolate(points: Array(points.prefix(originalK)),
                                          values: Array(values.prefix(originalK)))
        return Array(coeffs.prefix(originalK))
    }

    /// CPU Lagrange interpolation: given (x_i, y_i), recover polynomial coefficients.
    private func lagrangeInterpolate(points: [Bb], values: [Bb]) -> [Bb] {
        let n = points.count
        var result = [Bb](repeating: .zero, count: n)

        for i in 0..<n {
            // Compute Lagrange basis polynomial L_i evaluated contribution
            var basis = [Bb](repeating: .zero, count: n)
            basis[0] = .one

            var denom = Bb.one
            var basisDeg = 0

            for j in 0..<n {
                if j == i { continue }
                // Multiply basis by (x - x_j)
                denom = bbMul(denom, bbSub(points[i], points[j]))
                // Shift basis up and subtract x_j * basis
                basisDeg += 1
                for d in stride(from: basisDeg, through: 1, by: -1) {
                    basis[d] = bbSub(basis[d - 1], bbMul(points[j], basis[d]))
                }
                basis[0] = bbSub(.zero, bbMul(points[j], basis[0]))
            }

            // Scale by y_i / denom
            let scale = bbMul(values[i], bbInverse(denom))
            for d in 0..<n {
                result[d] = bbAdd(result[d], bbMul(scale, basis[d]))
            }
        }
        return result
    }

    /// Verify a shard against its expected evaluation.
    /// Recomputes the evaluation from coefficients and checks equality.
    public func verifyShard(coeffs: [Bb], shardIndex: Int, shardValue: Bb, totalN: Int) -> Bool {
        let logN = Int(log2(Double(totalN)))
        let omega = bbRootOfUnity(logN: logN)
        let point = bbPow(omega, UInt32(shardIndex))

        // Horner evaluation
        var acc = Bb.zero
        for i in stride(from: coeffs.count - 1, through: 0, by: -1) {
            acc = bbAdd(bbMul(acc, point), coeffs[i])
        }
        return acc.v == shardValue.v
    }
}

// MARK: - GF(2^16) RS Engine (GPU Matrix-Based)

/// Reed-Solomon encoder/decoder over GF(2^16) using GPU matrix operations.
/// Systematic encoding: data shards are preserved, parity shards appended.
/// Uses Vandermonde matrix for encoding and its inverse for decoding.
public class ReedSolomonGF16Engine {
    public static let version = Versions.reedSolomon
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    let encodeFunction: MTLComputePipelineState
    let decodeFunction: MTLComputePipelineState
    let batchMulFunction: MTLComputePipelineState
    let batchAddFunction: MTLComputePipelineState
    let polyEvalFunction: MTLComputePipelineState
    let logTableBuf: MTLBuffer
    let antilogTableBuf: MTLBuffer

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw RSError.noGPU
        }
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw RSError.noCommandQueue
        }
        self.commandQueue = queue

        // Compile shaders
        let library = try ReedSolomonGF16Engine.compileShaders(device: device)

        guard let encodeFn = library.makeFunction(name: "rs_encode_systematic"),
              let decodeFn = library.makeFunction(name: "rs_decode_matrix"),
              let batchMulFn = library.makeFunction(name: "gf16_batch_mul"),
              let batchAddFn = library.makeFunction(name: "gf16_batch_add"),
              let polyEvalFn = library.makeFunction(name: "gf16_poly_eval") else {
            throw RSError.missingKernel
        }

        self.encodeFunction = try device.makeComputePipelineState(function: encodeFn)
        self.decodeFunction = try device.makeComputePipelineState(function: decodeFn)
        self.batchMulFunction = try device.makeComputePipelineState(function: batchMulFn)
        self.batchAddFunction = try device.makeComputePipelineState(function: batchAddFn)
        self.polyEvalFunction = try device.makeComputePipelineState(function: polyEvalFn)

        // Upload log/antilog tables to GPU
        let tables = gf16Tables
        let logSize = 65536 * MemoryLayout<UInt16>.stride
        guard let logBuf = device.makeBuffer(length: logSize, options: .storageModeShared) else {
            throw RSError.gpuError("Failed to create log table buffer")
        }
        tables.logTable.withUnsafeBytes { src in
            memcpy(logBuf.contents(), src.baseAddress!, logSize)
        }
        self.logTableBuf = logBuf

        let antilogSize = 131070 * MemoryLayout<UInt16>.stride
        guard let alogBuf = device.makeBuffer(length: antilogSize, options: .storageModeShared) else {
            throw RSError.gpuError("Failed to create antilog table buffer")
        }
        tables.antilogTable.withUnsafeBytes { src in
            memcpy(alogBuf.contents(), src.baseAddress!, antilogSize)
        }
        self.antilogTableBuf = alogBuf
    }

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let source = try String(contentsOfFile: shaderDir + "/erasure/reed_solomon.metal", encoding: .utf8)
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: source, options: options)
    }

    // MARK: - Systematic Encoding

    /// Encode k data shards into k data + (n-k) parity shards.
    /// Returns array of n = k + parityCount elements.
    /// First k elements are the original data (systematic).
    public func encode(data: [GF16], parityCount: Int) throws -> [GF16] {
        let k = data.count
        let n = k + parityCount
        precondition(n <= 65535, "Total shards must be <= 65535 for GF(2^16)")

        // Build generator matrix: (parityCount x k) Vandermonde submatrix
        // eval points: g^k, g^(k+1), ..., g^(n-1) for parity rows
        // generator[i][j] = evalPoint[i]^j
        let evalPoints = gf16EvalPoints(n)
        var generator = [UInt16](repeating: 0, count: parityCount * k)
        for i in 0..<parityCount {
            let alpha = evalPoints[k + i]
            var alphaPow = GF16.one
            for j in 0..<k {
                generator[i * k + j] = alphaPow.value
                alphaPow = gf16Mul(alphaPow, alpha)
            }
        }

        // Upload to GPU
        let dataSize = k * MemoryLayout<UInt16>.stride
        guard let dataBuf = device.makeBuffer(length: dataSize, options: .storageModeShared) else {
            throw RSError.gpuError("Failed to create data buffer")
        }
        data.withUnsafeBytes { src in
            memcpy(dataBuf.contents(), src.baseAddress!, dataSize)
        }

        let paritySize = parityCount * MemoryLayout<UInt16>.stride
        guard let parityBuf = device.makeBuffer(length: paritySize, options: .storageModeShared) else {
            throw RSError.gpuError("Failed to create parity buffer")
        }

        let genSize = generator.count * MemoryLayout<UInt16>.stride
        guard let genBuf = device.makeBuffer(bytes: generator, length: genSize, options: .storageModeShared) else {
            throw RSError.gpuError("Failed to create generator buffer")
        }

        var kVal = UInt32(k)

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw RSError.noCommandBuffer
        }
        guard let enc = cmdBuf.makeComputeCommandEncoder() else {
            throw RSError.gpuError("Failed to create compute encoder")
        }

        enc.setComputePipelineState(encodeFunction)
        enc.setBuffer(dataBuf, offset: 0, index: 0)
        enc.setBuffer(parityBuf, offset: 0, index: 1)
        enc.setBuffer(genBuf, offset: 0, index: 2)
        enc.setBuffer(logTableBuf, offset: 0, index: 3)
        enc.setBuffer(antilogTableBuf, offset: 0, index: 4)
        enc.setBytes(&kVal, length: 4, index: 5)

        let tgSize = min(parityCount, encodeFunction.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(MTLSize(width: parityCount, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw RSError.gpuError(error.localizedDescription)
        }

        // Combine: data + parity
        var result = data
        let parityPtr = parityBuf.contents().bindMemory(to: GF16.self, capacity: parityCount)
        result.append(contentsOf: UnsafeBufferPointer(start: parityPtr, count: parityCount))
        return result
    }

    // MARK: - Decoding

    /// Decode original k data shards from any k available shards.
    /// shards: array of (index, value) where index is the shard position (0..<n).
    /// originalK: number of original data shards.
    /// totalN: total number of shards from encoding.
    public func decode(shards: [(index: Int, value: GF16)], originalK: Int, totalN: Int) throws -> [GF16] {
        let k = originalK
        guard shards.count >= k else {
            throw RSError.insufficientShards
        }

        let availableShards = Array(shards.prefix(k))

        // Build Vandermonde submatrix for available shard positions
        let evalPoints = gf16EvalPoints(totalN)
        var matrix = [UInt16](repeating: 0, count: k * k)
        for i in 0..<k {
            let alpha = evalPoints[availableShards[i].index]
            var alphaPow = GF16.one
            for j in 0..<k {
                matrix[i * k + j] = alphaPow.value
                alphaPow = gf16Mul(alphaPow, alpha)
            }
        }

        // Invert the matrix (CPU, O(k^3))
        let invMatrix = try gf16MatrixInverse(matrix, size: k)

        // GPU matrix-vector multiply: recovered = invMatrix * shardValues
        let shardValues = availableShards.map { $0.value }

        let shardSize = k * MemoryLayout<UInt16>.stride
        guard let shardBuf = device.makeBuffer(length: shardSize, options: .storageModeShared) else {
            throw RSError.gpuError("Failed to create shard buffer")
        }
        shardValues.withUnsafeBytes { src in
            memcpy(shardBuf.contents(), src.baseAddress!, shardSize)
        }

        guard let outBuf = device.makeBuffer(length: shardSize, options: .storageModeShared) else {
            throw RSError.gpuError("Failed to create output buffer")
        }

        let invSize = invMatrix.count * MemoryLayout<UInt16>.stride
        guard let invBuf = device.makeBuffer(bytes: invMatrix, length: invSize, options: .storageModeShared) else {
            throw RSError.gpuError("Failed to create inverse matrix buffer")
        }

        var kVal = UInt32(k)

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw RSError.noCommandBuffer
        }
        guard let enc = cmdBuf.makeComputeCommandEncoder() else {
            throw RSError.gpuError("Failed to create compute encoder")
        }

        enc.setComputePipelineState(decodeFunction)
        enc.setBuffer(shardBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBuffer(invBuf, offset: 0, index: 2)
        enc.setBuffer(logTableBuf, offset: 0, index: 3)
        enc.setBuffer(antilogTableBuf, offset: 0, index: 4)
        enc.setBytes(&kVal, length: 4, index: 5)

        let tgSize = min(k, decodeFunction.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(MTLSize(width: k, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw RSError.gpuError(error.localizedDescription)
        }

        let ptr = outBuf.contents().bindMemory(to: GF16.self, capacity: k)
        return Array(UnsafeBufferPointer(start: ptr, count: k))
    }

    // MARK: - GPU Batch Operations

    /// GPU batch multiply: out[i] = a[i] * b[i] in GF(2^16)
    public func batchMul(_ a: [GF16], _ b: [GF16]) throws -> [GF16] {
        let n = a.count
        precondition(n == b.count)

        let size = n * MemoryLayout<UInt16>.stride
        guard let aBuf = device.makeBuffer(length: size, options: .storageModeShared),
              let bBuf = device.makeBuffer(length: size, options: .storageModeShared),
              let outBuf = device.makeBuffer(length: size, options: .storageModeShared) else {
            throw RSError.gpuError("Failed to create buffers")
        }
        a.withUnsafeBytes { src in memcpy(aBuf.contents(), src.baseAddress!, size) }
        b.withUnsafeBytes { src in memcpy(bBuf.contents(), src.baseAddress!, size) }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else { throw RSError.noCommandBuffer }
        guard let enc = cmdBuf.makeComputeCommandEncoder() else { throw RSError.gpuError("No encoder") }

        enc.setComputePipelineState(batchMulFunction)
        enc.setBuffer(aBuf, offset: 0, index: 0)
        enc.setBuffer(bBuf, offset: 0, index: 1)
        enc.setBuffer(outBuf, offset: 0, index: 2)
        enc.setBuffer(logTableBuf, offset: 0, index: 3)
        enc.setBuffer(antilogTableBuf, offset: 0, index: 4)

        let tg = min(n, batchMulFunction.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let ptr = outBuf.contents().bindMemory(to: GF16.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    /// GPU polynomial evaluation at multiple points in GF(2^16)
    public func polyEval(coeffs: [GF16], points: [GF16]) throws -> [GF16] {
        let nPts = points.count
        let deg = coeffs.count

        let coeffSize = deg * MemoryLayout<UInt16>.stride
        let ptsSize = nPts * MemoryLayout<UInt16>.stride

        guard let coeffBuf = device.makeBuffer(length: coeffSize, options: .storageModeShared),
              let ptsBuf = device.makeBuffer(length: ptsSize, options: .storageModeShared),
              let outBuf = device.makeBuffer(length: ptsSize, options: .storageModeShared) else {
            throw RSError.gpuError("Failed to create buffers")
        }
        coeffs.withUnsafeBytes { src in memcpy(coeffBuf.contents(), src.baseAddress!, coeffSize) }
        points.withUnsafeBytes { src in memcpy(ptsBuf.contents(), src.baseAddress!, ptsSize) }

        var degVal = UInt32(deg)

        guard let cmdBuf = commandQueue.makeCommandBuffer() else { throw RSError.noCommandBuffer }
        guard let enc = cmdBuf.makeComputeCommandEncoder() else { throw RSError.gpuError("No encoder") }

        enc.setComputePipelineState(polyEvalFunction)
        enc.setBuffer(coeffBuf, offset: 0, index: 0)
        enc.setBuffer(ptsBuf, offset: 0, index: 1)
        enc.setBuffer(outBuf, offset: 0, index: 2)
        enc.setBuffer(logTableBuf, offset: 0, index: 3)
        enc.setBuffer(antilogTableBuf, offset: 0, index: 4)
        enc.setBytes(&degVal, length: 4, index: 5)

        let tg = min(nPts, polyEvalFunction.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(MTLSize(width: nPts, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let ptr = outBuf.contents().bindMemory(to: GF16.self, capacity: nPts)
        return Array(UnsafeBufferPointer(start: ptr, count: nPts))
    }
}

// MARK: - GF(2^16) Matrix Inverse (Gauss-Jordan, CPU)

/// Invert a k x k matrix over GF(2^16) using Gauss-Jordan elimination.
/// Matrix is stored row-major as flat UInt16 array.
func gf16MatrixInverse(_ matrix: [UInt16], size k: Int) throws -> [UInt16] {
    // Augmented matrix [A | I]
    var aug = [UInt16](repeating: 0, count: k * 2 * k)
    for i in 0..<k {
        for j in 0..<k {
            aug[i * 2 * k + j] = matrix[i * k + j]
        }
        aug[i * 2 * k + k + i] = 1  // Identity
    }

    let tables = gf16Tables

    for col in 0..<k {
        // Find pivot
        var pivotRow = -1
        for row in col..<k {
            if aug[row * 2 * k + col] != 0 {
                pivotRow = row
                break
            }
        }
        guard pivotRow >= 0 else {
            throw RSError.invalidInput  // Singular matrix
        }

        // Swap rows
        if pivotRow != col {
            for j in 0..<(2 * k) {
                let tmp = aug[col * 2 * k + j]
                aug[col * 2 * k + j] = aug[pivotRow * 2 * k + j]
                aug[pivotRow * 2 * k + j] = tmp
            }
        }

        // Scale pivot row so pivot = 1
        let pivotVal = aug[col * 2 * k + col]
        if pivotVal != 1 {
            let logInv = 65535 - Int(tables.logTable[Int(pivotVal)])
            let inv = tables.antilogTable[logInv < 0 ? logInv + 65535 : logInv]
            for j in 0..<(2 * k) {
                let v = aug[col * 2 * k + j]
                if v != 0 {
                    let s = Int(tables.logTable[Int(v)]) + Int(tables.logTable[Int(inv)])
                    aug[col * 2 * k + j] = tables.antilogTable[s >= 65535 ? s - 65535 : s]
                }
            }
        }

        // Eliminate column in all other rows
        for row in 0..<k {
            if row == col { continue }
            let factor = aug[row * 2 * k + col]
            if factor == 0 { continue }
            let logFactor = Int(tables.logTable[Int(factor)])
            for j in 0..<(2 * k) {
                let v = aug[col * 2 * k + j]
                if v != 0 {
                    let s = Int(tables.logTable[Int(v)]) + logFactor
                    aug[row * 2 * k + j] ^= tables.antilogTable[s >= 65535 ? s - 65535 : s]
                }
            }
        }
    }

    // Extract inverse from right half
    var result = [UInt16](repeating: 0, count: k * k)
    for i in 0..<k {
        for j in 0..<k {
            result[i * k + j] = aug[i * 2 * k + k + j]
        }
    }
    return result
}

// MARK: - Helpers

// nextPow2 defined in BrakedownEngine.swift (module-level)
