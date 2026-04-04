// Sumcheck Engine — GPU-accelerated sumcheck protocol
// Core primitive for STARK/GKR proof systems
// Operates on multilinear polynomials represented as evaluations over the boolean hypercube.

import Foundation
import Metal

public class SumcheckEngine {
    public static let version = Versions.sumcheck
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    let reduceFunction: MTLComputePipelineState
    let roundPartialFunction: MTLComputePipelineState
    let roundReduceFusedFunction: MTLComputePipelineState
    let roundReduceCoalescedFunction: MTLComputePipelineState
    let partialFinalReduceFunction: MTLComputePipelineState
    let fusedMultiroundFunction: MTLComputePipelineState
    let fused4CoalescedFunction: MTLComputePipelineState
    let fused2CoalescedFunction: MTLComputePipelineState
    let fused2StridedFunction: MTLComputePipelineState
    let highDegRoundReduceFunction: MTLComputePipelineState
    let highDegRoundReduceSmallFunction: MTLComputePipelineState

    // Cached buffers for high-degree sumcheck
    private var hdEvalPointsBuf: MTLBuffer?
    private var hdEvalPointsCount: Int = 0
    private var hdPolyBufs: [MTLBuffer] = []  // ping-pong poly buffers
    private var hdPolyBufSize: Int = 0
    private var hdPartialBuf: MTLBuffer?
    private var hdPartialBufSize: Int = 0

    // Cached ping-pong buffers for full sumcheck
    private var scEvalBufA: MTLBuffer?
    private var scEvalBufB: MTLBuffer?
    private var scEvalBufSize: Int = 0
    // Cached partial sum buffers pool (reused across calls)
    private var scPartialBufs: [MTLBuffer] = []
    private var scPartialBufsMaxGroups: Int = 0  // max numGroups seen
    // Cached fused batch partial buffer
    private var scFusedPartialBuf: MTLBuffer?
    private var scFusedPartialBufSize: Int = 0
    // Cached input buffer for fullSumcheck
    private var scInputBuf: MTLBuffer?
    private var scInputBufElements: Int = 0
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

        let library = try SumcheckEngine.compileShaders(device: device)
        let hdLibrary = try SumcheckEngine.compileHighDegShaders(device: device)

        guard let reduceFn = library.makeFunction(name: "sumcheck_reduce"),
              let roundFn = library.makeFunction(name: "sumcheck_round_partial"),
              let fusedFn = library.makeFunction(name: "sumcheck_round_reduce_fused"),
              let coalescedFn = library.makeFunction(name: "sumcheck_round_reduce_coalesced"),
              let finalReduceFn = library.makeFunction(name: "sumcheck_partial_final_reduce"),
              let fusedMultiroundFn = library.makeFunction(name: "sumcheck_fused_multiround"),
              let fused4Fn = library.makeFunction(name: "sumcheck_fused4_coalesced"),
              let fused2Fn = library.makeFunction(name: "sumcheck_fused2_coalesced"),
              let fused2StridedFn = library.makeFunction(name: "sumcheck_fused2_strided") else {
            throw MSMError.missingKernel
        }
        guard let hdRoundReduceFn = hdLibrary.makeFunction(name: "sumcheck_highdeg_round_reduce"),
              let hdRoundReduceSmallFn = hdLibrary.makeFunction(name: "sumcheck_highdeg_round_reduce_small") else {
            throw MSMError.missingKernel
        }

        self.reduceFunction = try device.makeComputePipelineState(function: reduceFn)
        self.roundPartialFunction = try device.makeComputePipelineState(function: roundFn)
        self.roundReduceFusedFunction = try device.makeComputePipelineState(function: fusedFn)
        self.roundReduceCoalescedFunction = try device.makeComputePipelineState(function: coalescedFn)
        self.partialFinalReduceFunction = try device.makeComputePipelineState(function: finalReduceFn)
        self.fusedMultiroundFunction = try device.makeComputePipelineState(function: fusedMultiroundFn)
        self.fused4CoalescedFunction = try device.makeComputePipelineState(function: fused4Fn)
        self.fused2CoalescedFunction = try device.makeComputePipelineState(function: fused2Fn)
        self.fused2StridedFunction = try device.makeComputePipelineState(function: fused2StridedFn)
        self.highDegRoundReduceFunction = try device.makeComputePipelineState(function: hdRoundReduceFn)
        self.highDegRoundReduceSmallFunction = try device.makeComputePipelineState(function: hdRoundReduceSmallFn)
        self.tuning = TuningManager.shared.config(device: device)
    }

    private static func cleanFrSource(_ shaderDir: String) throws -> String {
        let frSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        return frSource
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")
    }

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let cleanFr = try cleanFrSource(shaderDir)
        let scSource = try String(contentsOfFile: shaderDir + "/sumcheck/sumcheck_kernels.metal", encoding: .utf8)

        let cleanSC = scSource.split(separator: "\n").filter { !$0.contains("#include") }.joined(separator: "\n")

        let combined = cleanFr + "\n" + cleanSC
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    private static func compileHighDegShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let cleanFr = try cleanFrSource(shaderDir)
        let hdSource = try String(contentsOfFile: shaderDir + "/sumcheck/sumcheck_highdeg_kernels.metal", encoding: .utf8)

        let cleanHD = hdSource.split(separator: "\n").filter { !$0.contains("#include") }.joined(separator: "\n")

        let combined = cleanFr + "\n" + cleanHD
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

    // MARK: - Sumcheck reduce

    /// Fix one variable to challenge r, reducing evaluations from n to n/2.
    public func reduce(evals: MTLBuffer, output: MTLBuffer, challenge: Fr, n: Int) throws {
        let halfN = UInt32(n / 2)

        guard let challengeBuf = createFrBuffer([challenge]),
              let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var halfNVal = halfN
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(reduceFunction)
        enc.setBuffer(evals, offset: 0, index: 0)
        enc.setBuffer(output, offset: 0, index: 1)
        enc.setBuffer(challengeBuf, offset: 0, index: 2)
        enc.setBytes(&halfNVal, length: 4, index: 3)
        let tg = min(tuning.sumcheckPerRoundTGSize, Int(reduceFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: Int(halfN), height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    /// Compute round polynomial evaluations at X=0, X=1, X=2.
    /// Returns (S(0), S(1), S(2)) as three Fr elements.
    public func computeRoundPoly(evals: MTLBuffer, n: Int) throws -> (Fr, Fr, Fr) {
        let halfN = n / 2
        let tgSize = 256
        let numGroups = max(1, (halfN + tgSize - 1) / tgSize)

        guard let partialBuf = device.makeBuffer(length: numGroups * 3 * MemoryLayout<Fr>.stride,
                                                   options: .storageModeShared),
              let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var halfNVal = UInt32(halfN)
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(roundPartialFunction)
        enc.setBuffer(evals, offset: 0, index: 0)
        enc.setBuffer(partialBuf, offset: 0, index: 1)
        enc.setBytes(&halfNVal, length: 4, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        // CPU-side reduction of partial sums
        let ptr = partialBuf.contents().bindMemory(to: Fr.self, capacity: numGroups * 3)
        var s0 = Fr.zero, s1 = Fr.zero, s2 = Fr.zero
        for g in 0..<numGroups {
            s0 = frAdd(s0, ptr[g * 3])
            s1 = frAdd(s1, ptr[g * 3 + 1])
            s2 = frAdd(s2, ptr[g * 3 + 2])
        }

        return (s0, s1, s2)
    }

    // MARK: - High-level API

    /// High-level reduce: array in, array out.
    public func reduce(evals: [Fr], challenge: Fr) throws -> [Fr] {
        let n = evals.count
        precondition(n >= 2 && (n & (n - 1)) == 0)
        let halfN = n / 2

        guard let evalsBuf = createFrBuffer(evals),
              let outBuf = device.makeBuffer(length: halfN * MemoryLayout<Fr>.stride,
                                              options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to create buffers")
        }

        try reduce(evals: evalsBuf, output: outBuf, challenge: challenge, n: n)

        let ptr = outBuf.contents().bindMemory(to: Fr.self, capacity: halfN)
        return Array(UnsafeBufferPointer(start: ptr, count: halfN))
    }

    /// Run full sumcheck protocol simulation.
    /// Returns: array of round polynomials [(S(0), S(1), S(2))] and the final evaluation.
    /// Optimized: fused multi-round kernel batches up to 8 rounds per dispatch (shared memory),
    /// falls back to per-round kernel for small sizes. Single command buffer.
    public func fullSumcheck(evals: [Fr], challenges: [Fr]) throws
        -> (rounds: [(Fr, Fr, Fr)], finalEval: Fr)
    {
        let numVars = challenges.count
        precondition(evals.count == (1 << numVars))

        let stride = MemoryLayout<Fr>.stride
        let fusedChunkSize = tuning.sumcheckPerRoundTGSize
        let fusedTGSize = tuning.sumcheckFusedTGSize
        let maxFusedRounds = 8     // max rounds per fused dispatch
        let perRoundTGSize = tuning.sumcheckPerRoundTGSize

        // Ensure cached eval ping-pong buffers are large enough
        let halfMax = evals.count / 2
        if scEvalBufSize < halfMax {
            guard let a = device.makeBuffer(length: halfMax * stride, options: .storageModeShared),
                  let b = device.makeBuffer(length: halfMax * stride, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to allocate sumcheck eval buffers")
            }
            scEvalBufA = a
            scEvalBufB = b
            scEvalBufSize = halfMax
        }

        // Plan: fused 2-round dispatches for early rounds (halves dispatch count),
        // single per-round when odd round remains, then fused multi-round for final rounds.
        enum DispatchInfo {
            case single(round: Int, halfN: Int, numGroups: Int, partialBuf: MTLBuffer)
            case fused2(startRound: Int, quarterN: Int, numGroups: Int, partialBuf: MTLBuffer)
        }
        struct FusedBatch {
            let startRound: Int
            let numRounds: Int
            let inputN: Int
            let partialBuf: MTLBuffer
        }

        var dispatchInfos: [DispatchInfo] = []
        var fusedBatch: FusedBatch? = nil
        var currentN = evals.count
        var roundIdx = 0

        // Count dispatches and max groups needed for buffer pool
        let maxNumGroups = max(1, (evals.count / 4 + perRoundTGSize - 1) / perRoundTGSize)
        let maxPartialBytes = maxNumGroups * 6 * stride  // 2 rounds * 3 entries per group
        var dispatchCount = 0
        var tmpN = evals.count
        while tmpN > fusedChunkSize {
            if tmpN / 2 > fusedChunkSize {
                dispatchCount += 1  // fused2 counts as 1 dispatch
                tmpN /= 4
            } else {
                dispatchCount += 1  // single round
                tmpN /= 2
            }
        }
        if scPartialBufs.count < dispatchCount || scPartialBufsMaxGroups < maxNumGroups {
            scPartialBufs = []
            for _ in 0..<max(dispatchCount, 1) {
                guard let buf = device.makeBuffer(length: maxPartialBytes, options: .storageModeShared) else {
                    throw MSMError.gpuError("Failed to allocate per-round partial buffer")
                }
                scPartialBufs.append(buf)
            }
            scPartialBufsMaxGroups = maxNumGroups
        }

        // Phase 1: fused 2-round or single-round for large sizes (n > fusedChunkSize)
        while roundIdx < numVars && currentN > fusedChunkSize {
            let remainingLarge = {
                var n = currentN; var count = 0
                while n > fusedChunkSize { n /= 2; count += 1 }
                return count
            }()
            if remainingLarge >= 2 && roundIdx + 1 < numVars {
                // Fused 2-round: process 2 rounds at once
                let quarterN = currentN / 4
                let numGroups = max(1, (quarterN + perRoundTGSize - 1) / perRoundTGSize)
                dispatchInfos.append(.fused2(startRound: roundIdx, quarterN: quarterN,
                                              numGroups: numGroups, partialBuf: scPartialBufs[dispatchInfos.count]))
                currentN = quarterN
                roundIdx += 2
            } else {
                // Single round
                let halfN = currentN / 2
                let numGroups = max(1, (halfN + perRoundTGSize - 1) / perRoundTGSize)
                dispatchInfos.append(.single(round: roundIdx, halfN: halfN,
                                              numGroups: numGroups, partialBuf: scPartialBufs[dispatchInfos.count]))
                currentN = halfN
                roundIdx += 1
            }
        }

        // Phase 2: fused multi-round for final rounds (n <= fusedChunkSize = 256)
        if roundIdx < numVars && currentN <= fusedChunkSize && currentN >= 2 {
            let remaining = numVars - roundIdx
            let numRounds = min(maxFusedRounds, remaining)
            let numTGroups = max(1, currentN / fusedChunkSize)
            let fusedBytes = numRounds * numTGroups * 3 * stride
            if scFusedPartialBufSize < fusedBytes {
                guard let buf = device.makeBuffer(length: fusedBytes, options: .storageModeShared) else {
                    throw MSMError.gpuError("Failed to allocate fused partial buffer")
                }
                scFusedPartialBuf = buf
                scFusedPartialBufSize = fusedBytes
            }
            fusedBatch = FusedBatch(startRound: roundIdx, numRounds: numRounds,
                                     inputN: currentN, partialBuf: scFusedPartialBuf!)
            currentN >>= numRounds
            roundIdx += numRounds
        }

        // Any remaining rounds after fused batch (shouldn't happen with maxFusedRounds=8)
        while roundIdx < numVars {
            let halfN = currentN / 2
            let numGroups = max(1, (halfN + perRoundTGSize - 1) / perRoundTGSize)
            guard let pBuf = device.makeBuffer(length: numGroups * 3 * stride,
                                                options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to allocate per-round partial buffer")
            }
            dispatchInfos.append(.single(round: roundIdx, halfN: halfN,
                                          numGroups: numGroups, partialBuf: pBuf))
            currentN = halfN
            roundIdx += 1
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        // Reuse cached input buffer
        let evalsN = evals.count
        if evalsN > scInputBufElements {
            guard let buf = device.makeBuffer(length: evalsN * stride, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to create input buffer")
            }
            scInputBuf = buf
            scInputBufElements = evalsN
        }
        let inputBufInit = scInputBuf!
        evals.withUnsafeBytes { src in
            memcpy(inputBufInit.contents(), src.baseAddress!, evalsN * stride)
        }
        var inputBuf = inputBufInit
        var useA = true
        let enc = cmdBuf.makeComputeCommandEncoder()!

        // Phase 1: dispatches for large sizes (fused 2-round or single)
        for (idx, info) in dispatchInfos.enumerated() {
            let outputBuf = useA ? scEvalBufA! : scEvalBufB!

            switch info {
            case .fused2(let startRound, let quarterN, let numGroups, let partialBuf):
                let chal0 = challenges[startRound]
                let chal1 = challenges[startRound + 1]
                let chalBuf = device.makeBuffer(length: 2 * stride, options: .storageModeShared)!
                [chal0, chal1].withUnsafeBytes { src in
                    memcpy(chalBuf.contents(), src.baseAddress!, 2 * stride)
                }

                enc.setComputePipelineState(fused2StridedFunction)
                enc.setBuffer(inputBuf, offset: 0, index: 0)
                enc.setBuffer(outputBuf, offset: 0, index: 1)
                enc.setBuffer(partialBuf, offset: 0, index: 2)
                enc.setBuffer(chalBuf, offset: 0, index: 3)
                var qnVal = UInt32(quarterN)
                enc.setBytes(&qnVal, length: 4, index: 4)
                var ngVal = UInt32(numGroups)
                enc.setBytes(&ngVal, length: 4, index: 5)
                enc.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: perRoundTGSize, height: 1, depth: 1))

            case .single(let round, let halfN, let numGroups, let partialBuf):
                enc.setComputePipelineState(roundReduceFusedFunction)
                enc.setBuffer(inputBuf, offset: 0, index: 0)
                enc.setBuffer(outputBuf, offset: 0, index: 1)
                enc.setBuffer(partialBuf, offset: 0, index: 2)
                var challenge = challenges[round]
                enc.setBytes(&challenge, length: stride, index: 3)
                var halfNVal = UInt32(halfN)
                enc.setBytes(&halfNVal, length: 4, index: 4)
                enc.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: perRoundTGSize, height: 1, depth: 1))
            }

            if idx < dispatchInfos.count - 1 || fusedBatch != nil {
                enc.memoryBarrier(scope: .buffers)
            }

            inputBuf = outputBuf
            useA = !useA
        }

        // Phase 2: fused multi-round for final rounds
        if let batch = fusedBatch {
            let outputBuf = useA ? scEvalBufA! : scEvalBufB!

            let batchChallenges = Array(challenges[batch.startRound..<batch.startRound + batch.numRounds])
            let chalBuf = device.makeBuffer(length: batch.numRounds * stride, options: .storageModeShared)!
            batchChallenges.withUnsafeBytes { src in
                memcpy(chalBuf.contents(), src.baseAddress!, batch.numRounds * stride)
            }

            enc.setComputePipelineState(fusedMultiroundFunction)
            enc.setBuffer(inputBuf, offset: 0, index: 0)
            enc.setBuffer(outputBuf, offset: 0, index: 1)
            enc.setBuffer(batch.partialBuf, offset: 0, index: 2)
            enc.setBuffer(chalBuf, offset: 0, index: 3)
            var currentNVal = UInt32(batch.inputN)
            enc.setBytes(&currentNVal, length: 4, index: 4)
            var numRoundsVal = UInt32(batch.numRounds)
            enc.setBytes(&numRoundsVal, length: 4, index: 5)
            let numTGroups = max(1, batch.inputN / fusedChunkSize)
            enc.dispatchThreadgroups(MTLSize(width: numTGroups, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: fusedTGSize, height: 1, depth: 1))

            inputBuf = outputBuf
            useA = !useA
        }

        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        // CPU-side final reduction of partial sums
        var rounds: [(Fr, Fr, Fr)] = []

        // Per-round partial sums (phase 1)
        for info in dispatchInfos {
            switch info {
            case .fused2(_, _, let numGroups, let partialBuf):
                // 2 rounds × numGroups × 3 entries
                let ptr = partialBuf.contents().bindMemory(to: Fr.self, capacity: numGroups * 6)
                for r in 0..<2 {
                    var s0 = Fr.zero, s1 = Fr.zero, s2 = Fr.zero
                    for g in 0..<numGroups {
                        let base = (r * numGroups + g) * 3
                        s0 = frAdd(s0, ptr[base])
                        s1 = frAdd(s1, ptr[base + 1])
                        s2 = frAdd(s2, ptr[base + 2])
                    }
                    rounds.append((s0, s1, s2))
                }
            case .single(_, _, let numGroups, let partialBuf):
                let ptr = partialBuf.contents().bindMemory(to: Fr.self, capacity: numGroups * 3)
                var s0 = Fr.zero, s1 = Fr.zero, s2 = Fr.zero
                for g in 0..<numGroups {
                    s0 = frAdd(s0, ptr[g * 3])
                    s1 = frAdd(s1, ptr[g * 3 + 1])
                    s2 = frAdd(s2, ptr[g * 3 + 2])
                }
                rounds.append((s0, s1, s2))
            }
        }

        // Fused batch partial sums (phase 2)
        if let batch = fusedBatch {
            let numTG = max(1, batch.inputN / fusedChunkSize)
            let ptr = batch.partialBuf.contents().bindMemory(to: Fr.self, capacity: batch.numRounds * numTG * 3)
            for r in 0..<batch.numRounds {
                var s0 = Fr.zero, s1 = Fr.zero, s2 = Fr.zero
                for g in 0..<numTG {
                    let base = (r * numTG + g) * 3
                    s0 = frAdd(s0, ptr[base])
                    s1 = frAdd(s1, ptr[base + 1])
                    s2 = frAdd(s2, ptr[base + 2])
                }
                rounds.append((s0, s1, s2))
            }
        }

        let finalPtr = inputBuf.contents().bindMemory(to: Fr.self, capacity: 1)
        return (rounds, finalPtr[0])
    }

    // MARK: - CPU Reference

    /// CPU sumcheck reduce for verification.
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

    /// CPU round polynomial at X=0,1,2.
    public static func cpuRoundPoly(evals: [Fr]) -> (Fr, Fr, Fr) {
        let n = evals.count
        let halfN = n / 2
        var s0 = Fr.zero, s1 = Fr.zero, s2 = Fr.zero
        for i in 0..<halfN {
            let a = evals[i]
            let b = evals[i + halfN]
            s0 = frAdd(s0, a)
            s1 = frAdd(s1, b)
            let twoB = frAdd(b, b)
            let f2 = frSub(twoB, a)
            s2 = frAdd(s2, f2)
        }
        return (s0, s1, s2)
    }

    // MARK: - High-Degree Sumcheck

    /// Precompute evaluation points [0, 1, 2, ..., degree] in Montgomery form.
    private static func evalPoints(degree: Int) -> [Fr] {
        return (0...degree).map { frFromInt(UInt64($0)) }
    }

    /// Run high-degree sumcheck protocol for k multilinear polynomials.
    /// The round polynomial has degree k (= numPolys) and is evaluated at k+1 points.
    ///
    /// Parameters:
    ///   - polynomials: array of k polynomial evaluation vectors, each of length 2^numVars
    ///   - challenges: numVars Fiat-Shamir challenges (one per round)
    ///
    /// Returns: array of round polynomial evaluations [S(0), S(1), ..., S(degree)] per round,
    ///          plus the final evaluation (product of the single remaining values).
    public func proveHighDegree(
        polynomials: [[Fr]],
        challenges: [Fr]
    ) throws -> (rounds: [[Fr]], finalEval: Fr) {
        let numPolys = polynomials.count
        precondition(numPolys >= 2 && numPolys <= 32, "Supported: 2-32 polynomials")
        let numVars = challenges.count
        let n = polynomials[0].count
        precondition(n == (1 << numVars), "Polynomial length must be 2^numVars")
        for p in polynomials {
            precondition(p.count == n, "All polynomials must have same length")
        }

        let degree = numPolys  // degree of round polynomial
        let numEvalPts = degree + 1
        let stride = MemoryLayout<Fr>.stride

        // Precompute eval points in Montgomery form
        let evalPts = SumcheckEngine.evalPoints(degree: degree)
        if hdEvalPointsCount < numEvalPts {
            guard let buf = createFrBuffer(evalPts) else {
                throw MSMError.gpuError("Failed to create eval points buffer")
            }
            hdEvalPointsBuf = buf
            hdEvalPointsCount = numEvalPts
        } else {
            evalPts.withUnsafeBytes { src in
                memcpy(hdEvalPointsBuf!.contents(), src.baseAddress!, numEvalPts * stride)
            }
        }

        // Create interleaved poly buffer: poly0_evals || poly1_evals || ... || polyk_evals
        let totalElements = numPolys * n
        let polyBufBytes = totalElements * stride

        // Ensure ping-pong buffers are large enough
        if hdPolyBufSize < polyBufBytes {
            guard let a = device.makeBuffer(length: polyBufBytes, options: .storageModeShared),
                  let b = device.makeBuffer(length: polyBufBytes, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to allocate high-degree poly buffers")
            }
            hdPolyBufs = [a, b]
            hdPolyBufSize = polyBufBytes
        }

        // Copy polynomial data into input buffer
        let inputBuf = hdPolyBufs[0]
        for (j, poly) in polynomials.enumerated() {
            poly.withUnsafeBytes { src in
                memcpy(inputBuf.contents() + j * n * stride, src.baseAddress!, n * stride)
            }
        }

        let tgSize = min(256, Int(highDegRoundReduceFunction.maxTotalThreadsPerThreadgroup))
        var currentN = n
        var rounds: [[Fr]] = []
        var useFirst = true

        for round in 0..<numVars {
            let halfN = currentN / 2
            let numGroups = max(1, (halfN + tgSize - 1) / tgSize)

            // Ensure partial buffer is large enough
            let partialBytes = numGroups * numEvalPts * stride
            if hdPartialBufSize < partialBytes {
                guard let buf = device.makeBuffer(length: partialBytes, options: .storageModeShared) else {
                    throw MSMError.gpuError("Failed to allocate high-degree partial buffer")
                }
                hdPartialBuf = buf
                hdPartialBufSize = partialBytes
            }

            let srcBuf = useFirst ? hdPolyBufs[0] : hdPolyBufs[1]
            let dstBuf = useFirst ? hdPolyBufs[1] : hdPolyBufs[0]

            guard let chalBuf = createFrBuffer([challenges[round]]),
                  let cmdBuf = commandQueue.makeCommandBuffer() else {
                throw MSMError.noCommandBuffer
            }

            var halfNVal = UInt32(halfN)
            var numPolysVal = UInt32(numPolys)
            var numEvalPtsVal = UInt32(numEvalPts)

            let pipelineState = numPolys <= 8 ? highDegRoundReduceSmallFunction : highDegRoundReduceFunction

            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipelineState)
            enc.setBuffer(srcBuf, offset: 0, index: 0)
            enc.setBuffer(dstBuf, offset: 0, index: 1)
            enc.setBuffer(hdPartialBuf, offset: 0, index: 2)
            enc.setBuffer(hdEvalPointsBuf, offset: 0, index: 3)
            enc.setBuffer(chalBuf, offset: 0, index: 4)
            enc.setBytes(&halfNVal, length: 4, index: 5)
            enc.setBytes(&numPolysVal, length: 4, index: 6)
            enc.setBytes(&numEvalPtsVal, length: 4, index: 7)
            enc.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
            enc.endEncoding()

            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            if let error = cmdBuf.error {
                throw MSMError.gpuError(error.localizedDescription)
            }

            // CPU-side reduction of partial sums
            let ptr = hdPartialBuf!.contents().bindMemory(to: Fr.self, capacity: numGroups * numEvalPts)
            var roundEvals = [Fr](repeating: Fr.zero, count: numEvalPts)
            for g in 0..<numGroups {
                for t in 0..<numEvalPts {
                    roundEvals[t] = frAdd(roundEvals[t], ptr[g * numEvalPts + t])
                }
            }
            rounds.append(roundEvals)

            currentN = halfN
            useFirst = !useFirst
        }

        // Final evaluation: product of the single remaining value from each polynomial
        let finalBuf = useFirst ? hdPolyBufs[0] : hdPolyBufs[1]
        let finalPtr = finalBuf.contents().bindMemory(to: Fr.self, capacity: numPolys)
        var finalEval = Fr.one
        for j in 0..<numPolys {
            finalEval = frMul(finalEval, finalPtr[j])
        }

        return (rounds, finalEval)
    }

    // MARK: - High-Degree CPU Reference

    /// CPU reference for high-degree sumcheck reduce step.
    /// Fixes one variable to challenge r for all k polynomials.
    public static func cpuReduceHighDegree(polynomials: [[Fr]], challenge: Fr) -> [[Fr]] {
        let k = polynomials.count
        let n = polynomials[0].count
        let halfN = n / 2
        var result = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: halfN), count: k)
        for j in 0..<k {
            for i in 0..<halfN {
                let a = polynomials[j][i]
                let b = polynomials[j][i + halfN]
                let diff = frSub(b, a)
                result[j][i] = frAdd(a, frMul(challenge, diff))
            }
        }
        return result
    }

    /// CPU reference for high-degree round polynomial evaluation.
    /// Returns evaluations [S(0), S(1), ..., S(degree)] where degree = number of polynomials.
    public static func cpuRoundPolyHighDegree(polynomials: [[Fr]]) -> [Fr] {
        let k = polynomials.count
        let n = polynomials[0].count
        let halfN = n / 2
        let numEvalPts = k + 1
        let evalPts = evalPoints(degree: k)

        var sums = [Fr](repeating: Fr.zero, count: numEvalPts)
        for i in 0..<halfN {
            for t in 0..<numEvalPts {
                var product = Fr.one
                for j in 0..<k {
                    let a = polynomials[j][i]
                    let b = polynomials[j][i + halfN]
                    let diff = frSub(b, a)
                    let pjt = frAdd(a, frMul(evalPts[t], diff))
                    product = frMul(product, pjt)
                }
                sums[t] = frAdd(sums[t], product)
            }
        }
        return sums
    }

    /// CPU reference: full high-degree sumcheck protocol.
    public static func cpuFullHighDegree(
        polynomials: [[Fr]],
        challenges: [Fr]
    ) -> (rounds: [[Fr]], finalEval: Fr) {
        let numVars = challenges.count
        var currentPolys = polynomials
        var rounds: [[Fr]] = []

        for round in 0..<numVars {
            let roundPoly = cpuRoundPolyHighDegree(polynomials: currentPolys)
            rounds.append(roundPoly)
            currentPolys = cpuReduceHighDegree(polynomials: currentPolys, challenge: challenges[round])
        }

        // Final eval: product of single remaining values
        var finalEval = Fr.one
        for j in 0..<polynomials.count {
            finalEval = frMul(finalEval, currentPolys[j][0])
        }

        return (rounds, finalEval)
    }

    /// Verify high-degree sumcheck: checks that S(0) + S(1) = running sum for each round.
    /// For degree-d sumcheck, the sum of round poly at 0 and 1 equals the claimed sum.
    public static func verifyHighDegree(
        claimedSum: Fr,
        rounds: [[Fr]],
        challenges: [Fr],
        finalEval: Fr
    ) -> Bool {
        let numRounds = rounds.count
        guard numRounds == challenges.count else { return false }
        let degree = rounds[0].count - 1  // number of eval points - 1

        var currentSum = claimedSum

        for i in 0..<numRounds {
            let roundPoly = rounds[i]
            // Verify: S(0) + S(1) = currentSum
            let s01 = frAdd(roundPoly[0], roundPoly[1])
            let s01Limbs = frToInt(s01)
            let curLimbs = frToInt(currentSum)
            if s01Limbs != curLimbs {
                return false
            }

            // Compute S(r) using Lagrange interpolation over points 0,1,...,degree
            let r = challenges[i]
            currentSum = lagrangeInterpolateEval(
                points: evalPoints(degree: degree),
                values: roundPoly,
                at: r
            )
        }

        // Final check: S(r_last) should equal the final evaluation
        let finalLimbs = frToInt(currentSum)
        let expectedLimbs = frToInt(finalEval)
        return finalLimbs == expectedLimbs
    }

    /// Lagrange interpolation: given (x_i, y_i) pairs, evaluate the polynomial at point z.
    private static func lagrangeInterpolateEval(points: [Fr], values: [Fr], at z: Fr) -> Fr {
        let n = points.count
        var result = Fr.zero
        for i in 0..<n {
            // Compute Lagrange basis L_i(z) = product_{j != i} (z - x_j) / (x_i - x_j)
            var num = Fr.one
            var den = Fr.one
            for j in 0..<n {
                if j == i { continue }
                num = frMul(num, frSub(z, points[j]))
                den = frMul(den, frSub(points[i], points[j]))
            }
            let basis = frMul(num, frInverse(den))
            result = frAdd(result, frMul(values[i], basis))
        }
        return result
    }

    // MARK: - Internal

    private func createFrBuffer(_ data: [Fr]) -> MTLBuffer? {
        let byteCount = data.count * MemoryLayout<Fr>.stride
        guard let buf = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            return nil
        }
        data.withUnsafeBytes { src in
            memcpy(buf.contents(), src.baseAddress!, byteCount)
        }
        return buf
    }
}
