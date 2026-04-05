// GPU-accelerated sumcheck engine for multilinear polynomials
// Supports BN254 Fr (8x uint32 Montgomery), BabyBear (uint32), Goldilocks (uint64).
//
// The core sumcheck operation per round:
// 1. Compute round polynomial: s0 = sum f(0,x), s1 = sum f(1,x) over all x
// 2. Fold the table: table[i] = table[i] + r*(table[i+half] - table[i])
//
// For tables >= 1024 elements, these run on Metal GPU; below that, CPU fallback.

import Foundation
import Metal
import NeonFieldOps

// MARK: - Field Type

public enum FieldType {
    case bn254
    case babybear
    case goldilocks
}

// MARK: - GPU Sumcheck Engine

public class GPUSumcheckEngine {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // BN254 pipelines
    private let reduceBN254: MTLComputePipelineState
    private let roundPolyBN254: MTLComputePipelineState
    private let fusedBN254: MTLComputePipelineState

    // BabyBear pipelines
    private let reduceBabyBear: MTLComputePipelineState
    private let roundPolyBabyBear: MTLComputePipelineState

    // Goldilocks pipelines
    private let reduceGoldilocks: MTLComputePipelineState
    private let roundPolyGoldilocks: MTLComputePipelineState

    // Cached buffers (BN254)
    private var evalBufA: MTLBuffer?
    private var evalBufB: MTLBuffer?
    private var evalBufCapacity: Int = 0  // in bytes
    private var partialBuf: MTLBuffer?
    private var partialBufCapacity: Int = 0  // in bytes

    // CPU fallback threshold
    private static let gpuThreshold = 1024

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        let library = try GPUSumcheckEngine.compileShaders(device: device)

        guard let rBN = library.makeFunction(name: "sumcheck_reduce_bn254"),
              let rpBN = library.makeFunction(name: "sumcheck_round_poly_bn254"),
              let fBN = library.makeFunction(name: "sumcheck_fused_round_reduce_bn254"),
              let rBB = library.makeFunction(name: "sumcheck_reduce_babybear"),
              let rpBB = library.makeFunction(name: "sumcheck_round_poly_babybear"),
              let rGL = library.makeFunction(name: "sumcheck_reduce_goldilocks"),
              let rpGL = library.makeFunction(name: "sumcheck_round_poly_goldilocks") else {
            throw MSMError.missingKernel
        }

        self.reduceBN254 = try device.makeComputePipelineState(function: rBN)
        self.roundPolyBN254 = try device.makeComputePipelineState(function: rpBN)
        self.fusedBN254 = try device.makeComputePipelineState(function: fBN)
        self.reduceBabyBear = try device.makeComputePipelineState(function: rBB)
        self.roundPolyBabyBear = try device.makeComputePipelineState(function: rpBB)
        self.reduceGoldilocks = try device.makeComputePipelineState(function: rGL)
        self.roundPolyGoldilocks = try device.makeComputePipelineState(function: rpGL)
    }

    // MARK: - Shader Compilation

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let frSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let cleanFr = frSource
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")

        let reduceSource = try String(contentsOfFile: shaderDir + "/sumcheck/sumcheck_reduce.metal", encoding: .utf8)
        let cleanReduce = reduceSource.split(separator: "\n").filter { !$0.contains("#include") }.joined(separator: "\n")

        let combined = cleanFr + "\n" + cleanReduce
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    private static func findShaderDir() -> String {
        let execPath = CommandLine.arguments[0]
        let execDir = (execPath as NSString).deletingLastPathComponent
        for bundle in Bundle.allBundles {
            if let url = bundle.url(forResource: "Shaders", withExtension: nil) {
                let path = url.appendingPathComponent("fields/bn254_fr.metal").path
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
            if FileManager.default.fileExists(atPath: "\(path)/fields/bn254_fr.metal") {
                return path
            }
        }
        return "./Sources/Shaders"
    }

    // MARK: - Buffer Management

    private func ensureEvalBuffers(byteCount: Int) {
        if evalBufCapacity >= byteCount { return }
        evalBufA = device.makeBuffer(length: byteCount, options: .storageModeShared)
        evalBufB = device.makeBuffer(length: byteCount, options: .storageModeShared)
        evalBufCapacity = byteCount
    }

    private func ensurePartialBuffer(byteCount: Int) {
        if partialBufCapacity >= byteCount { return }
        partialBuf = device.makeBuffer(length: byteCount, options: .storageModeShared)
        partialBufCapacity = byteCount
    }

    // MARK: - BN254 Prove Round (GPU)

    /// Prove one sumcheck round on BN254 Fr multilinear evaluations stored in an MTLBuffer.
    /// Returns (s0, s1) round polynomial coefficients and a new MTLBuffer with the folded table.
    public func proveRoundBN254(table: MTLBuffer, logSize: Int, challenge: Fr) throws -> (s0: Fr, s1: Fr, newTable: MTLBuffer) {
        let n = 1 << logSize
        let halfN = n / 2
        let stride = MemoryLayout<Fr>.stride

        if n < GPUSumcheckEngine.gpuThreshold {
            return try proveRoundBN254CPU(table: table, n: n, challenge: challenge)
        }

        let tgSize = 256
        let numGroups = max(1, (halfN + tgSize - 1) / tgSize)

        // Ensure output + partial buffers
        let outBytes = halfN * stride
        ensureEvalBuffers(byteCount: outBytes)
        let partialBytes = numGroups * 2 * stride
        ensurePartialBuffer(byteCount: partialBytes)

        guard let outputBuf = evalBufA,
              let pBuf = partialBuf,
              let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        // Fused kernel: round poly + reduce in one pass
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(fusedBN254)
        enc.setBuffer(table, offset: 0, index: 0)
        enc.setBuffer(outputBuf, offset: 0, index: 1)
        enc.setBuffer(pBuf, offset: 0, index: 2)
        var chal = challenge
        enc.setBytes(&chal, length: stride, index: 3)
        var halfNVal = UInt32(halfN)
        enc.setBytes(&halfNVal, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        // CPU-side final reduction of partial sums
        let ptr = pBuf.contents().bindMemory(to: Fr.self, capacity: numGroups * 2)
        var s0 = Fr.zero
        var s1 = Fr.zero
        for g in 0..<numGroups {
            s0 = frAdd(s0, ptr[g * 2])
            s1 = frAdd(s1, ptr[g * 2 + 1])
        }

        return (s0: s0, s1: s1, newTable: outputBuf)
    }

    private func proveRoundBN254CPU(table: MTLBuffer, n: Int, challenge: Fr) throws -> (s0: Fr, s1: Fr, newTable: MTLBuffer) {
        let halfN = n / 2
        let stride = MemoryLayout<Fr>.stride
        let ptr = table.contents().bindMemory(to: Fr.self, capacity: n)

        var s0 = Fr.zero
        var s1 = Fr.zero
        var result = [Fr](repeating: Fr.zero, count: halfN)

        let oneMinusR = frSub(Fr.one, challenge)
        for i in 0..<halfN {
            let a = ptr[i]
            let b = ptr[i + halfN]
            s0 = frAdd(s0, a)
            s1 = frAdd(s1, b)
            // fold: a + r*(b - a) = (1-r)*a + r*b
            let diff = frSub(b, a)
            let rDiff = frMul(challenge, diff)
            result[i] = frAdd(a, rDiff)
        }

        guard let outBuf = device.makeBuffer(length: halfN * stride, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to create CPU output buffer")
        }
        result.withUnsafeBytes { src in
            memcpy(outBuf.contents(), src.baseAddress!, halfN * stride)
        }

        return (s0: s0, s1: s1, newTable: outBuf)
    }

    // MARK: - Full Sumcheck Protocol (BN254)

    /// Run full sumcheck: given a multilinear polynomial (as evaluation table), execute all rounds.
    /// The prover computes round polynomials and folds the table using Fiat-Shamir challenges.
    ///
    /// Parameters:
    ///   - evals: evaluation table of a multilinear polynomial with numVars variables
    ///   - transcript: Fiat-Shamir transcript for challenge generation
    ///
    /// Returns: (rounds: [(s0, s1)] per round, challenges: Fr[], finalEval: Fr)
    public func fullSumcheckBN254(
        evals: [Fr],
        transcript: Transcript
    ) throws -> (rounds: [(Fr, Fr)], challenges: [Fr], finalEval: Fr) {
        let n = evals.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Table size must be a power of 2")
        var numVars = 0
        var tmp = n
        while tmp > 1 { tmp >>= 1; numVars += 1 }

        let stride = MemoryLayout<Fr>.stride
        guard let tableBuf = device.makeBuffer(length: n * stride, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to create table buffer")
        }
        evals.withUnsafeBytes { src in
            memcpy(tableBuf.contents(), src.baseAddress!, n * stride)
        }

        var currentTable = tableBuf
        var currentLogSize = numVars
        var rounds: [(Fr, Fr)] = []
        var challenges: [Fr] = []
        rounds.reserveCapacity(numVars)
        challenges.reserveCapacity(numVars)

        for _ in 0..<numVars {
            // Step 1: compute round polynomial (s0, s1)
            let (s0, s1) = try computeRoundPolyBN254(table: currentTable, logSize: currentLogSize)

            rounds.append((s0, s1))

            // Step 2: absorb into transcript and get challenge
            transcript.absorb(s0)
            transcript.absorb(s1)
            let challenge = transcript.squeeze()
            challenges.append(challenge)

            // Step 3: fold the table
            let newTable = try reduceBN254Table(table: currentTable, logSize: currentLogSize, challenge: challenge)
            currentTable = newTable
            currentLogSize -= 1
        }

        // Final evaluation is the single remaining element
        let finalPtr = currentTable.contents().bindMemory(to: Fr.self, capacity: 1)
        let finalEval = finalPtr[0]

        return (rounds: rounds, challenges: challenges, finalEval: finalEval)
    }

    /// Compute only the round polynomial (s0, s1) without folding.
    public func computeRoundPolyBN254(table: MTLBuffer, logSize: Int) throws -> (Fr, Fr) {
        let n = 1 << logSize
        let halfN = n / 2
        let stride = MemoryLayout<Fr>.stride

        if n < GPUSumcheckEngine.gpuThreshold {
            let ptr = table.contents().bindMemory(to: Fr.self, capacity: n)
            var s0 = Fr.zero
            var s1 = Fr.zero
            for i in 0..<halfN {
                s0 = frAdd(s0, ptr[i])
                s1 = frAdd(s1, ptr[i + halfN])
            }
            return (s0, s1)
        }

        let tgSize = 256
        let numGroups = max(1, (halfN + tgSize - 1) / tgSize)
        let partialBytes = numGroups * 2 * stride
        ensurePartialBuffer(byteCount: partialBytes)

        guard let pBuf = partialBuf,
              let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(roundPolyBN254)
        enc.setBuffer(table, offset: 0, index: 0)
        enc.setBuffer(pBuf, offset: 0, index: 1)
        var halfNVal = UInt32(halfN)
        enc.setBytes(&halfNVal, length: 4, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = pBuf.contents().bindMemory(to: Fr.self, capacity: numGroups * 2)
        var s0 = Fr.zero
        var s1 = Fr.zero
        for g in 0..<numGroups {
            s0 = frAdd(s0, ptr[g * 2])
            s1 = frAdd(s1, ptr[g * 2 + 1])
        }

        return (s0, s1)
    }

    /// Fold the table by challenge (reduce), returning a new MTLBuffer with half the elements.
    public func reduceBN254Table(table: MTLBuffer, logSize: Int, challenge: Fr) throws -> MTLBuffer {
        let n = 1 << logSize
        let halfN = n / 2
        let stride = MemoryLayout<Fr>.stride

        if n < GPUSumcheckEngine.gpuThreshold {
            let ptr = table.contents().bindMemory(to: Fr.self, capacity: n)
            var result = [Fr](repeating: Fr.zero, count: halfN)
            for i in 0..<halfN {
                let a = ptr[i]
                let b = ptr[i + halfN]
                let diff = frSub(b, a)
                let rDiff = frMul(challenge, diff)
                result[i] = frAdd(a, rDiff)
            }
            guard let outBuf = device.makeBuffer(length: halfN * stride, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to create reduce output buffer")
            }
            result.withUnsafeBytes { src in
                memcpy(outBuf.contents(), src.baseAddress!, halfN * stride)
            }
            return outBuf
        }

        let outBytes = halfN * stride
        ensureEvalBuffers(byteCount: outBytes)

        guard let outputBuf = evalBufA,
              let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(reduceBN254)
        enc.setBuffer(table, offset: 0, index: 0)
        enc.setBuffer(outputBuf, offset: 0, index: 1)
        var chal = challenge
        enc.setBytes(&chal, length: stride, index: 2)
        var halfNVal = UInt32(halfN)
        enc.setBytes(&halfNVal, length: 4, index: 3)

        let tgSize = min(256, Int(reduceBN254.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: halfN, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        return outputBuf
    }

    // MARK: - BabyBear Operations

    /// Compute BabyBear round polynomial (s0, s1).
    public func computeRoundPolyBabyBear(table: MTLBuffer, logSize: Int) throws -> (UInt32, UInt32) {
        let n = 1 << logSize
        let halfN = n / 2
        let elemSize = MemoryLayout<UInt32>.stride

        if n < GPUSumcheckEngine.gpuThreshold {
            let ptr = table.contents().bindMemory(to: UInt32.self, capacity: n)
            let p: UInt64 = 0x78000001
            var s0: UInt64 = 0
            var s1: UInt64 = 0
            for i in 0..<halfN {
                s0 = (s0 + UInt64(ptr[i])) % p
                s1 = (s1 + UInt64(ptr[i + halfN])) % p
            }
            return (UInt32(s0), UInt32(s1))
        }

        let tgSize = 256
        let numGroups = max(1, (halfN + tgSize - 1) / tgSize)
        let partialBytes = numGroups * 2 * elemSize
        ensurePartialBuffer(byteCount: partialBytes)

        guard let pBuf = partialBuf,
              let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(roundPolyBabyBear)
        enc.setBuffer(table, offset: 0, index: 0)
        enc.setBuffer(pBuf, offset: 0, index: 1)
        var halfNVal = UInt32(halfN)
        enc.setBytes(&halfNVal, length: 4, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = pBuf.contents().bindMemory(to: UInt32.self, capacity: numGroups * 2)
        let p: UInt64 = 0x78000001
        var s0: UInt64 = 0
        var s1: UInt64 = 0
        for g in 0..<numGroups {
            s0 = (s0 + UInt64(ptr[g * 2])) % p
            s1 = (s1 + UInt64(ptr[g * 2 + 1])) % p
        }
        return (UInt32(s0), UInt32(s1))
    }

    /// Fold BabyBear table.
    public func reduceBabyBearTable(table: MTLBuffer, logSize: Int, challenge: UInt32) throws -> MTLBuffer {
        let n = 1 << logSize
        let halfN = n / 2
        let elemSize = MemoryLayout<UInt32>.stride

        if n < GPUSumcheckEngine.gpuThreshold {
            let p: UInt64 = 0x78000001
            let ptr = table.contents().bindMemory(to: UInt32.self, capacity: n)
            var result = [UInt32](repeating: 0, count: halfN)
            let r64 = UInt64(challenge)
            for i in 0..<halfN {
                let a = UInt64(ptr[i])
                let b = UInt64(ptr[i + halfN])
                let diff = (b + p - a) % p
                let rDiff = (r64 * diff) % p
                result[i] = UInt32((a + rDiff) % p)
            }
            guard let outBuf = device.makeBuffer(length: halfN * elemSize, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to create BabyBear output buffer")
            }
            result.withUnsafeBytes { src in
                memcpy(outBuf.contents(), src.baseAddress!, halfN * elemSize)
            }
            return outBuf
        }

        let outBytes = halfN * elemSize
        guard let outputBuf = device.makeBuffer(length: outBytes, options: .storageModeShared),
              let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(reduceBabyBear)
        enc.setBuffer(table, offset: 0, index: 0)
        enc.setBuffer(outputBuf, offset: 0, index: 1)
        var chal = challenge
        enc.setBytes(&chal, length: elemSize, index: 2)
        var halfNVal = UInt32(halfN)
        enc.setBytes(&halfNVal, length: 4, index: 3)

        let tgSize = min(256, Int(reduceBabyBear.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: halfN, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        return outputBuf
    }

    // MARK: - Goldilocks Operations

    /// Compute Goldilocks round polynomial (s0, s1).
    public func computeRoundPolyGoldilocks(table: MTLBuffer, logSize: Int) throws -> (UInt64, UInt64) {
        let n = 1 << logSize
        let halfN = n / 2
        let elemSize = MemoryLayout<UInt64>.stride

        if n < GPUSumcheckEngine.gpuThreshold {
            let p: UInt64 = 0xFFFFFFFF00000001
            let ptr = table.contents().bindMemory(to: UInt64.self, capacity: n)
            var s0: UInt64 = 0
            var s1: UInt64 = 0
            for i in 0..<halfN {
                s0 = glAddCPU(s0, ptr[i], p)
                s1 = glAddCPU(s1, ptr[i + halfN], p)
            }
            return (s0, s1)
        }

        let tgSize = 256
        let numGroups = max(1, (halfN + tgSize - 1) / tgSize)
        let partialBytes = numGroups * 2 * elemSize
        ensurePartialBuffer(byteCount: partialBytes)

        guard let pBuf = partialBuf,
              let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(roundPolyGoldilocks)
        enc.setBuffer(table, offset: 0, index: 0)
        enc.setBuffer(pBuf, offset: 0, index: 1)
        var halfNVal = UInt32(halfN)
        enc.setBytes(&halfNVal, length: 4, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let p: UInt64 = 0xFFFFFFFF00000001
        let ptr = pBuf.contents().bindMemory(to: UInt64.self, capacity: numGroups * 2)
        var s0: UInt64 = 0
        var s1: UInt64 = 0
        for g in 0..<numGroups {
            s0 = glAddCPU(s0, ptr[g * 2], p)
            s1 = glAddCPU(s1, ptr[g * 2 + 1], p)
        }
        return (s0, s1)
    }

    /// Fold Goldilocks table.
    public func reduceGoldilocksTable(table: MTLBuffer, logSize: Int, challenge: UInt64) throws -> MTLBuffer {
        let n = 1 << logSize
        let halfN = n / 2
        let elemSize = MemoryLayout<UInt64>.stride
        let p: UInt64 = 0xFFFFFFFF00000001

        if n < GPUSumcheckEngine.gpuThreshold {
            let ptr = table.contents().bindMemory(to: UInt64.self, capacity: n)
            var result = [UInt64](repeating: 0, count: halfN)
            for i in 0..<halfN {
                let a = ptr[i]
                let b = ptr[i + halfN]
                let diff = glSubCPU(b, a, p)
                let rDiff = glMulCPU(challenge, diff, p)
                result[i] = glAddCPU(a, rDiff, p)
            }
            guard let outBuf = device.makeBuffer(length: halfN * elemSize, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to create Goldilocks output buffer")
            }
            result.withUnsafeBytes { src in
                memcpy(outBuf.contents(), src.baseAddress!, halfN * elemSize)
            }
            return outBuf
        }

        let outBytes = halfN * elemSize
        guard let outputBuf = device.makeBuffer(length: outBytes, options: .storageModeShared),
              let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(reduceGoldilocks)
        enc.setBuffer(table, offset: 0, index: 0)
        enc.setBuffer(outputBuf, offset: 0, index: 1)
        var chal = challenge
        enc.setBytes(&chal, length: elemSize, index: 2)
        var halfNVal = UInt32(halfN)
        enc.setBytes(&halfNVal, length: 4, index: 3)

        let tgSize = min(256, Int(reduceGoldilocks.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: halfN, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        return outputBuf
    }

    // MARK: - CPU Goldilocks helpers

    private func glAddCPU(_ a: UInt64, _ b: UInt64, _ p: UInt64) -> UInt64 {
        let (sum, overflow) = a.addingReportingOverflow(b)
        if overflow {
            return sum &+ 0xFFFFFFFF  // eps = 2^32 - 1
        }
        return sum >= p ? sum - p : sum
    }

    private func glSubCPU(_ a: UInt64, _ b: UInt64, _ p: UInt64) -> UInt64 {
        if a >= b { return a - b }
        return a &+ p &- b
    }

    private func glMulCPU(_ a: UInt64, _ b: UInt64, _ p: UInt64) -> UInt64 {
        // Use 128-bit product via (hi, lo) decomposition
        let a0 = UInt64(UInt32(a & 0xFFFFFFFF))
        let a1 = UInt64(UInt32(a >> 32))
        let b0 = UInt64(UInt32(b & 0xFFFFFFFF))
        let b1 = UInt64(UInt32(b >> 32))

        let t0 = a0 * b0
        let t1 = a0 * b1
        let t2 = a1 * b0
        let t3 = a1 * b1

        let (mid, midOverflow) = t1.addingReportingOverflow(t2)

        let (lo, loOverflow) = t0.addingReportingOverflow(mid << 32)

        var hi = t3 &+ (mid >> 32) &+ (loOverflow ? 1 : 0)
        if midOverflow { hi &+= (1 << 32) }

        // Reduce: result = lo + hi_lo * eps - hi_hi (mod p)
        let hiLo = UInt64(UInt32(hi & 0xFFFFFFFF))
        let hiHi = UInt64(UInt32(hi >> 32))

        let hiLoEps = (hiLo << 32) &- hiLo
        let (s, c1) = lo.addingReportingOverflow(hiLoEps)
        var r = s &- hiHi
        let b2 = s < hiHi

        if c1 { r &+= 0xFFFFFFFF }
        if b2 { r &+= p }

        return r >= p ? r - p : r
    }

    // MARK: - Unified Prove Round API

    /// Prove one sumcheck round, dispatching to the correct field.
    /// Returns round polynomial coefficients as [UInt32] arrays and a new MTLBuffer.
    public func proveRound(
        table: MTLBuffer,
        logSize: Int,
        field: FieldType
    ) throws -> (roundPoly: [UInt32], newTable: MTLBuffer) {
        switch field {
        case .bn254:
            // Generate a deterministic challenge for standalone use
            let challenge = Fr.one  // caller should provide real challenge
            let (s0, s1, newT) = try proveRoundBN254(table: table, logSize: logSize, challenge: challenge)
            // Pack s0, s1 as 16 uint32s
            var result = [UInt32](repeating: 0, count: 16)
            withUnsafeBytes(of: s0) { src in
                for i in 0..<8 { result[i] = src.load(fromByteOffset: i * 4, as: UInt32.self) }
            }
            withUnsafeBytes(of: s1) { src in
                for i in 0..<8 { result[8 + i] = src.load(fromByteOffset: i * 4, as: UInt32.self) }
            }
            return (roundPoly: result, newTable: newT)

        case .babybear:
            let (s0, s1) = try computeRoundPolyBabyBear(table: table, logSize: logSize)
            let newT = try reduceBabyBearTable(table: table, logSize: logSize, challenge: 1)
            return (roundPoly: [s0, s1], newTable: newT)

        case .goldilocks:
            let (s0, s1) = try computeRoundPolyGoldilocks(table: table, logSize: logSize)
            let newT = try reduceGoldilocksTable(table: table, logSize: logSize, challenge: 1)
            return (roundPoly: [UInt32(s0 & 0xFFFFFFFF), UInt32(s0 >> 32),
                               UInt32(s1 & 0xFFFFFFFF), UInt32(s1 >> 32)],
                    newTable: newT)
        }
    }

    // MARK: - CPU Reference (BN254)

    /// CPU sumcheck round polynomial for verification.
    public static func cpuRoundPoly(evals: [Fr]) -> (Fr, Fr) {
        let n = evals.count
        let halfN = n / 2
        var s0 = Fr.zero
        var s1 = Fr.zero
        for i in 0..<halfN {
            s0 = frAdd(s0, evals[i])
            s1 = frAdd(s1, evals[i + halfN])
        }
        return (s0, s1)
    }

    /// CPU reduce (fold) for verification.
    public static func cpuReduce(evals: [Fr], challenge: Fr) -> [Fr] {
        let n = evals.count
        let halfN = n / 2
        var result = [Fr](repeating: Fr.zero, count: halfN)
        for i in 0..<halfN {
            let a = evals[i]
            let b = evals[i + halfN]
            let diff = frSub(b, a)
            let rDiff = frMul(challenge, diff)
            result[i] = frAdd(a, rDiff)
        }
        return result
    }
}
