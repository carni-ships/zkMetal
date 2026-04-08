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
import NeonFieldOps

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

        // Precompute all Lagrange denominators prod_{j!=i}(x_i - x_j) and batch-invert
        var rsDenoms = [Bb](repeating: Bb.one, count: n)
        var rsBases = [[Bb]](repeating: [Bb](repeating: .zero, count: n), count: n)
        for i in 0..<n {
            rsBases[i][0] = .one
            var basisDeg = 0
            for j in 0..<n where j != i {
                rsDenoms[i] = bbMul(rsDenoms[i], bbSub(points[i], points[j]))
                basisDeg += 1
                for d in stride(from: basisDeg, through: 1, by: -1) {
                    rsBases[i][d] = bbSub(rsBases[i][d - 1], bbMul(points[j], rsBases[i][d]))
                }
                rsBases[i][0] = bbSub(.zero, bbMul(points[j], rsBases[i][0]))
            }
        }
        var rsPrefix = [Bb](repeating: Bb.one, count: n)
        for i in 1..<n {
            rsPrefix[i] = rsDenoms[i - 1].v == 0 ? rsPrefix[i - 1] : bbMul(rsPrefix[i - 1], rsDenoms[i - 1])
        }
        let rsLast = rsDenoms[n - 1].v == 0 ? rsPrefix[n - 1] : bbMul(rsPrefix[n - 1], rsDenoms[n - 1])
        var rsInv = bbInverse(rsLast)
        var rsDenomInvs = [Bb](repeating: Bb.zero, count: n)
        for i in stride(from: n - 1, through: 0, by: -1) {
            if rsDenoms[i].v != 0 {
                rsDenomInvs[i] = bbMul(rsInv, rsPrefix[i])
                rsInv = bbMul(rsInv, rsDenoms[i])
            }
        }

        for i in 0..<n {
            let scale = bbMul(values[i], rsDenomInvs[i])
            for d in 0..<n {
                result[d] = bbAdd(result[d], bbMul(scale, rsBases[i][d]))
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

    /// Encode k data shards into n = k + parityCount evaluation shards.
    /// Treats data as polynomial coefficients and evaluates at n distinct points.
    /// All shards are evaluations (not systematic -- enables clean Vandermonde decode).
    public func encode(data: [GF16], parityCount: Int) throws -> [GF16] {
        let k = data.count
        let n = k + parityCount
        precondition(n <= 65535, "Total shards must be <= 65535 for GF(2^16)")

        // Evaluate data polynomial at n points using GPU
        let evalPoints = gf16EvalPoints(n)
        return try polyEval(coeffs: data, points: evalPoints)
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

// MARK: - BN254 Fr RS Engine (GPU NTT)

/// KZG proof for data availability sampling.
/// Contains the opening proof (witness commitment) and the claimed evaluation.
public struct DAKZGProof {
    public let evaluation: Fr
    public let witness: PointProjective

    public init(evaluation: Fr, witness: PointProjective) {
        self.evaluation = evaluation
        self.witness = witness
    }
}

/// Reed-Solomon encoder/decoder over BN254 Fr using GPU NTT.
/// RS encoding = forward NTT (evaluation on roots of unity).
/// RS decoding = erasure recovery via Lagrange interpolation + iNTT.
/// KZG proofs for data availability sampling via existing KZGEngine.
public class ReedSolomonBN254Engine {
    public static let version = Versions.reedSolomon
    public let nttEngine: NTTEngine
    public var kzgEngine: KZGEngine?

    /// Initialize with GPU NTT. Optionally provide KZG engine for DA proofs.
    public init(kzgEngine: KZGEngine? = nil) throws {
        self.nttEngine = try NTTEngine()
        self.kzgEngine = kzgEngine
    }

    // MARK: - Encode

    /// Encode data as RS codeword: pad polynomial coefficients to n = nextPow2(data.count * redundancyFactor),
    /// then forward NTT to evaluate on n-th roots of unity.
    /// Returns all n evaluations (the codeword).
    public func encode(data: [Fr], redundancyFactor: Int = 2) throws -> [Fr] {
        precondition(redundancyFactor >= 2, "Redundancy factor must be >= 2")
        let k = data.count
        let n = nextPow2(k * redundancyFactor)
        let logN = Int(log2(Double(n)))
        precondition(logN <= Fr.TWO_ADICITY, "NTT size exceeds BN254 Fr 2-adicity (\(Fr.TWO_ADICITY))")

        // Pad polynomial to n coefficients (higher-degree coefficients = 0)
        var padded = [Fr](repeating: .zero, count: n)
        for i in 0..<k { padded[i] = data[i] }

        // Forward NTT = RS encode: evaluates poly at n-th roots of unity
        return try nttEngine.ntt(padded)
    }

    // MARK: - Decode (erasure recovery)

    /// Decode original data from a subset of received evaluations.
    /// received: array of (index, value) pairs where index is position in the codeword (0..<n).
    /// originalSize: number of original data coefficients (k).
    /// Returns the original k polynomial coefficients.
    ///
    /// Algorithm: Lagrange interpolation over roots-of-unity subgroup points,
    /// exploiting the structure omega^i for efficient computation.
    public func decode(received: [(index: Int, value: Fr)], originalSize: Int) throws -> [Fr] {
        guard received.count >= originalSize else {
            throw RSError.insufficientShards
        }

        let k = originalSize
        let usable = Array(received.prefix(k))

        // Determine the codeword length from max index
        let maxIdx = usable.map { $0.index }.max()! + 1
        let n = nextPow2(max(maxIdx, k * 2))
        let logN = Int(log2(Double(n)))
        let omega = frRootOfUnity(logN: logN)

        // Compute evaluation points: x_i = omega^(index_i)
        let points = usable.map { shard -> Fr in
            frPow(omega, UInt64(shard.index))
        }
        let values = usable.map { $0.value }

        // Lagrange interpolation to recover polynomial coefficients
        let coeffs = lagrangeInterpolateBN254(points: points, values: values)
        return Array(coeffs.prefix(k))
    }

    // MARK: - KZG DA Proofs

    /// Generate a KZG opening proof for a single codeword position.
    /// data: original polynomial coefficients.
    /// index: position in the codeword (evaluation at omega^index).
    /// Requires kzgEngine to be set.
    public func generateProof(data: [Fr], index: Int) throws -> DAKZGProof {
        guard let kzg = kzgEngine else {
            throw RSError.gpuError("KZG engine not configured")
        }

        let n = nextPow2(data.count * 2)
        let logN = Int(log2(Double(n)))
        let omega = frRootOfUnity(logN: logN)
        let evalPoint = frPow(omega, UInt64(index))

        let proof = try kzg.open(data, at: evalPoint)
        return DAKZGProof(evaluation: proof.evaluation, witness: proof.witness)
    }

    /// Verify a KZG opening proof for a codeword position.
    /// commitment: KZG commitment to the polynomial [p(s)]_1.
    /// index: codeword position.
    /// value: claimed evaluation p(omega^index).
    /// proof: the KZG witness point.
    /// srsSecret: toxic waste (for non-pairing verification; test only).
    public func verifyProof(commitment: PointProjective, index: Int, value: Fr,
                            proof: DAKZGProof, codewordSize: Int, srsSecret: Fr) -> Bool {
        guard let kzg = kzgEngine else { return false }

        let logN = Int(log2(Double(codewordSize)))
        let omega = frRootOfUnity(logN: logN)
        let evalPoint = frPow(omega, UInt64(index))

        // Verify: e(C - [v]G, H) == e(proof, [s - z]H)
        // Using SRS secret directly (test-mode verification):
        // C == [v]*G + [s - z]*proof
        let g1 = pointFromAffine(kzg.srs[0])
        let vG = cPointScalarMul(g1, value)
        let sMz = frSub(srsSecret, evalPoint)
        let szProof = cPointScalarMul(proof.witness, sMz)
        let expected = pointAdd(vG, szProof)

        let cAff = batchToAffine([commitment])
        let eAff = batchToAffine([expected])
        return fpToInt(cAff[0].x) == fpToInt(eAff[0].x) &&
               fpToInt(cAff[0].y) == fpToInt(eAff[0].y)
    }

    /// Commit to a polynomial (convenience wrapper).
    public func commit(data: [Fr]) throws -> PointProjective {
        guard let kzg = kzgEngine else {
            throw RSError.gpuError("KZG engine not configured")
        }
        return try kzg.commit(data)
    }

    // MARK: - Private Helpers

    /// Lagrange interpolation over BN254 Fr: given (x_i, y_i), recover polynomial coefficients.
    /// O(k^2) algorithm suitable for moderate k.
    private func lagrangeInterpolateBN254(points: [Fr], values: [Fr]) -> [Fr] {
        let n = points.count
        var result = [Fr](repeating: .zero, count: n)

        // Precompute all Lagrange denominators and basis polynomials
        var frDenoms = [Fr](repeating: Fr.one, count: n)
        var frBases = [[Fr]](repeating: [Fr](repeating: .zero, count: n), count: n)
        for i in 0..<n {
            frBases[i][0] = .one
            var basisDeg = 0
            for j in 0..<n where j != i {
                frDenoms[i] = frMul(frDenoms[i], frSub(points[i], points[j]))
                basisDeg += 1
                for d in stride(from: basisDeg, through: 1, by: -1) {
                    frBases[i][d] = frSub(frBases[i][d - 1], frMul(points[j], frBases[i][d]))
                }
                frBases[i][0] = frSub(.zero, frMul(points[j], frBases[i][0]))
            }
        }

        // Montgomery batch inversion of all denominators
        var frPfx = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n {
            frPfx[i] = frDenoms[i - 1] == Fr.zero ? frPfx[i - 1] : frMul(frPfx[i - 1], frDenoms[i - 1])
        }
        let frLst = frDenoms[n - 1] == Fr.zero ? frPfx[n - 1] : frMul(frPfx[n - 1], frDenoms[n - 1])
        var frInvR = frInverse(frLst)
        var frDenomInvs = [Fr](repeating: Fr.zero, count: n)
        for i in stride(from: n - 1, through: 0, by: -1) {
            if frDenoms[i] != Fr.zero {
                frDenomInvs[i] = frMul(frInvR, frPfx[i])
                frInvR = frMul(frInvR, frDenoms[i])
            }
        }

        for i in 0..<n {
            let scale = frMul(values[i], frDenomInvs[i])
            for d in 0..<n {
                result[d] = frAdd(result[d], frMul(scale, frBases[i][d]))
            }
        }
        return result
    }
}

// MARK: - BLS12-381 Fr RS Engine (CPU NTT, Danksharding-compatible)

/// KZG proof for BLS12-381 data availability (Danksharding).
/// Uses G1 points on BLS12-381 curve.
public struct DAKZG381Proof {
    public let evaluation: Fr381
    public let witness: G1Projective381

    public init(evaluation: Fr381, witness: G1Projective381) {
        self.evaluation = evaluation
        self.witness = witness
    }
}

/// Reed-Solomon encoder/decoder over BLS12-381 Fr using CPU NTT.
/// Designed for Ethereum Danksharding data availability sampling.
///
/// BLS12-381 Fr has TWO_ADICITY=32, supporting NTT up to 2^32 = 4 billion slots.
/// Danksharding uses 4096 slots (logN=12), well within range.
///
/// RS encoding = forward NTT (evaluation at roots of unity).
/// RS decoding = erasure recovery via Lagrange interpolation.
/// KZG proofs use BLS12-381 G1 for DA sampling.
public class ReedSolomon381Engine {
    public static let version = Versions.reedSolomon

    /// SRS points for KZG: [G, sG, s^2 G, ..., s^(d-1) G] on BLS12-381 G1.
    public private(set) var srs: [G1Affine381]

    /// Initialize with optional SRS for KZG proofs.
    /// If no SRS is provided, encode/decode still work but proof generation will fail.
    public init(srs: [G1Affine381] = []) {
        self.srs = srs
    }

    /// Generate a test SRS for BLS12-381 KZG (NOT secure -- uses known secret).
    public static func generateTestSRS(secret: [UInt64], size: Int) -> [G1Affine381] {
        let gen = bls12381G1Generator()
        let genProj = g1_381FromAffine(gen)
        var points = [G1Projective381]()
        points.reserveCapacity(size)
        var sPow = fr381ToInt(Fr381.one) // s^0 = 1 in standard form
        let sStd = secret

        // Compute s^0, s^1, ..., s^(size-1) and multiply generator
        var sPowMont = Fr381.one
        let sFr = fr381Mul(Fr381.from64(secret), Fr381.from64(Fr381.R2_MOD_R))
        for _ in 0..<size {
            let scalarStd = fr381ToInt(sPowMont)
            points.append(g1_381ScalarMul(genProj, scalarStd))
            sPowMont = fr381Mul(sPowMont, sFr)
        }
        return batchG1_381ToAffine(points)
    }

    // MARK: - Encode

    /// Encode data as RS codeword over BLS12-381 Fr.
    /// Pad polynomial to n = nextPow2(data.count * redundancyFactor), then CPU NTT.
    public func encode(data: [Fr381], redundancyFactor: Int = 2) -> [Fr381] {
        precondition(redundancyFactor >= 2)
        let k = data.count
        let n = nextPow2(k * redundancyFactor)
        let logN = Int(log2(Double(n)))
        precondition(logN <= Fr381.TWO_ADICITY, "NTT size exceeds BLS12-381 Fr 2-adicity (\(Fr381.TWO_ADICITY))")

        var padded = [Fr381](repeating: .zero, count: n)
        for i in 0..<k { padded[i] = data[i] }

        // CPU forward NTT (Cooley-Tukey DIT)
        return cpuNTT381(padded, logN: logN, inverse: false)
    }

    // MARK: - Decode (erasure recovery)

    /// Decode original data from a subset of received evaluations.
    /// received: array of (index, value) pairs.
    /// originalSize: number of original data coefficients.
    public func decode(received: [(index: Int, value: Fr381)], originalSize: Int) throws -> [Fr381] {
        guard received.count >= originalSize else {
            throw RSError.insufficientShards
        }

        let k = originalSize
        let usable = Array(received.prefix(k))

        let maxIdx = usable.map { $0.index }.max()! + 1
        let n = nextPow2(max(maxIdx, k * 2))
        let logN = Int(log2(Double(n)))
        let omega = fr381RootOfUnity(logN: logN)

        // Evaluation points: x_i = omega^(index_i)
        let points = usable.map { shard -> Fr381 in
            fr381Pow(omega, UInt64(shard.index))
        }
        let values = usable.map { $0.value }

        let coeffs = lagrangeInterpolate381(points: points, values: values)
        return Array(coeffs.prefix(k))
    }

    // MARK: - KZG DA Proofs (BLS12-381)

    /// Generate a KZG opening proof at codeword position `index`.
    /// data: polynomial coefficients in Fr381.
    /// Returns proof with evaluation and witness point.
    public func generateProof(data: [Fr381], index: Int) throws -> DAKZG381Proof {
        guard !srs.isEmpty else {
            throw RSError.gpuError("SRS not configured")
        }

        let n = nextPow2(data.count * 2)
        let logN = Int(log2(Double(n)))
        let omega = fr381RootOfUnity(logN: logN)
        let evalPoint = fr381Pow(omega, UInt64(index))

        // Evaluate polynomial at evalPoint using Horner's method
        let pz = hornerEval381(data, at: evalPoint)

        // Compute quotient q(x) = (p(x) - p(z)) / (x - z) via synthetic division
        var shifted = data
        shifted[0] = fr381Sub(shifted[0], pz)
        let quotient = syntheticDiv381(shifted, z: evalPoint)

        // Witness = MSM(SRS[0..deg-1], quotient)
        let witness = msm381(points: Array(srs.prefix(quotient.count)), scalars: quotient)

        return DAKZG381Proof(evaluation: pz, witness: witness)
    }

    /// Verify a KZG opening proof.
    /// commitment: [p(s)]_1 on G1.
    /// index: codeword position.
    /// value: claimed evaluation.
    /// proof: the KZG witness.
    /// srsSecret: toxic waste scalar (test-mode verification without pairing).
    public func verifyProof(commitment: G1Projective381, index: Int, value: Fr381,
                            proof: DAKZG381Proof, codewordSize: Int, srsSecret: [UInt64]) -> Bool {
        let logN = Int(log2(Double(codewordSize)))
        let omega = fr381RootOfUnity(logN: logN)
        let evalPoint = fr381Pow(omega, UInt64(index))

        // Verify: C == [v]*G + [s - z]*proof
        let gen = g1_381FromAffine(bls12381G1Generator())
        let vG = g1_381ScalarMul(gen, fr381ToInt(value))

        let sFr = fr381Mul(Fr381.from64(srsSecret), Fr381.from64(Fr381.R2_MOD_R))
        let sMz = fr381Sub(sFr, evalPoint)
        let szProof = g1_381ScalarMul(proof.witness, fr381ToInt(sMz))

        let expected = g1_381Add(vG, szProof)

        guard let cAff = g1_381ToAffine(commitment),
              let eAff = g1_381ToAffine(expected) else {
            // If either is identity, both must be
            return g1_381IsIdentity(commitment) && g1_381IsIdentity(expected)
        }
        return fp381ToInt(cAff.x) == fp381ToInt(eAff.x) &&
               fp381ToInt(cAff.y) == fp381ToInt(eAff.y)
    }

    /// Commit to a polynomial using MSM on BLS12-381 G1.
    public func commit(data: [Fr381]) throws -> G1Projective381 {
        guard srs.count >= data.count else {
            throw RSError.gpuError("SRS too small for polynomial degree")
        }
        return msm381(points: Array(srs.prefix(data.count)), scalars: data)
    }

    // MARK: - CPU NTT over Fr381

    /// NTT / iNTT over Fr381 via C CIOS implementation.
    private func cpuNTT381(_ input: [Fr381], logN: Int, inverse: Bool) -> [Fr381] {
        let n = 1 << logN
        precondition(input.count == n)
        var data = input
        data.withUnsafeMutableBytes { buf in
            let ptr = buf.baseAddress!.assumingMemoryBound(to: UInt64.self)
            if inverse {
                bls12_381_fr_intt(ptr, Int32(logN))
            } else {
                bls12_381_fr_ntt(ptr, Int32(logN))
            }
        }
        return data
    }

    /// Bit-reverse an integer.
    private func bitReverse381(_ x: Int, bits: Int) -> Int {
        var result = 0
        var val = x
        for _ in 0..<bits {
            result = (result << 1) | (val & 1)
            val >>= 1
        }
        return result
    }

    /// Horner evaluation of polynomial at a point.
    private func hornerEval381(_ coeffs: [Fr381], at z: Fr381) -> Fr381 {
        var acc = Fr381.zero
        for i in stride(from: coeffs.count - 1, through: 0, by: -1) {
            acc = fr381Add(fr381Mul(acc, z), coeffs[i])
        }
        return acc
    }

    /// Synthetic division: (p(x) - p(z)) / (x - z) assuming p(z) has been subtracted from constant term.
    private func syntheticDiv381(_ coeffs: [Fr381], z: Fr381) -> [Fr381] {
        let n = coeffs.count
        if n < 2 { return [] }
        var quotient = [Fr381](repeating: .zero, count: n - 1)
        quotient[n - 2] = coeffs[n - 1]
        for i in stride(from: n - 3, through: 0, by: -1) {
            quotient[i] = fr381Add(coeffs[i + 1], fr381Mul(z, quotient[i + 1]))
        }
        return quotient
    }

    /// CPU MSM on BLS12-381 G1 using Pippenger.
    private func msm381(points: [G1Affine381], scalars: [Fr381]) -> G1Projective381 {
        let n = points.count
        precondition(scalars.count == n)
        if n == 0 { return g1_381Identity() }

        // Convert Fr381 scalars to flat UInt32 limbs
        var flatScalars = [UInt32]()
        flatScalars.reserveCapacity(n * 8)
        for s in scalars {
            let std = fr381ToInt(s) // [UInt64] in standard form
            flatScalars.append(UInt32(std[0] & 0xFFFFFFFF))
            flatScalars.append(UInt32(std[0] >> 32))
            flatScalars.append(UInt32(std[1] & 0xFFFFFFFF))
            flatScalars.append(UInt32(std[1] >> 32))
            flatScalars.append(UInt32(std[2] & 0xFFFFFFFF))
            flatScalars.append(UInt32(std[2] >> 32))
            flatScalars.append(UInt32(std[3] & 0xFFFFFFFF))
            flatScalars.append(UInt32(std[3] >> 32))
        }
        return g1_381PippengerMSMFlat(points: points, flatScalars: flatScalars)
    }

    /// Lagrange interpolation over Fr381.
    private func lagrangeInterpolate381(points: [Fr381], values: [Fr381]) -> [Fr381] {
        let n = points.count
        var result = [Fr381](repeating: .zero, count: n)

        for i in 0..<n {
            var basis = [Fr381](repeating: .zero, count: n)
            basis[0] = .one
            var denom = Fr381.one
            var basisDeg = 0

            for j in 0..<n {
                if j == i { continue }
                denom = fr381Mul(denom, fr381Sub(points[i], points[j]))
                basisDeg += 1
                for d in stride(from: basisDeg, through: 1, by: -1) {
                    basis[d] = fr381Sub(basis[d - 1], fr381Mul(points[j], basis[d]))
                }
                basis[0] = fr381Sub(.zero, fr381Mul(points[j], basis[0]))
            }

            let scale = fr381Mul(values[i], fr381Inverse(denom))
            for d in 0..<n {
                result[d] = fr381Add(result[d], fr381Mul(scale, basis[d]))
            }
        }
        return result
    }
}

// MARK: - Helpers

// nextPow2 defined in BrakedownEngine.swift (module-level)
