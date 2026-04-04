// Circle STARK Prover — GPU-accelerated STARK proofs using circle group over M31
//
// Protocol:
// 1. Trace generation + LDE via GPU Circle NTT on evaluation domain (blowup factor)
// 2. Commit trace columns via Keccak-256 Merkle trees
// 3. Fiat-Shamir challenge alpha, GPU constraint evaluation
// 4. Commit composition polynomial
// 5. GPU Circle FRI to prove low degree
// 6. Query phase: open trace + composition at FRI query positions

import Foundation
import Metal

// MARK: - Proof Data Structures

/// A single query response: opened values and Merkle authentication paths
public struct CircleSTARKQueryResponse {
    /// Trace values at this query position: [column] of M31
    public let traceValues: [M31]
    /// Trace Merkle authentication paths: [column] of path (each path is array of 32-byte hashes)
    public let tracePaths: [[[UInt8]]]
    /// Composition polynomial value at this query position
    public let compositionValue: M31
    /// Composition Merkle authentication path
    public let compositionPath: [[UInt8]]
    /// The query index in the evaluation domain
    public let queryIndex: Int
}

/// Circle FRI proof data
public struct CircleFRIProofData {
    /// Per-round data: commitment root + query responses
    public let rounds: [CircleFRIRound]
    /// Final constant value after all folding rounds
    public let finalValue: M31
    /// Query indices used
    public let queryIndices: [Int]
}

/// One round of Circle FRI
public struct CircleFRIRound {
    /// Merkle root of the folded polynomial evaluations
    public let commitment: [UInt8]
    /// For each query: (value at query, value at sibling, Merkle path)
    public let queryResponses: [(M31, M31, [[UInt8]])]
}

/// Complete Circle STARK proof
public struct CircleSTARKProof {
    /// Merkle roots of trace column LDEs
    public let traceCommitments: [[UInt8]]
    /// Merkle root of composition polynomial
    public let compositionCommitment: [UInt8]
    /// FRI proof for the composition polynomial
    public let friProof: CircleFRIProofData
    /// Query responses: trace + composition openings at query positions
    public let queryResponses: [CircleSTARKQueryResponse]
    /// Random alpha used for constraint batching
    public let alpha: M31
    /// Proof metadata
    public let traceLength: Int
    public let numColumns: Int
    public let logBlowup: Int

    public init(traceCommitments: [[UInt8]], compositionCommitment: [UInt8],
                friProof: CircleFRIProofData, queryResponses: [CircleSTARKQueryResponse],
                alpha: M31, traceLength: Int, numColumns: Int, logBlowup: Int) {
        self.traceCommitments = traceCommitments
        self.compositionCommitment = compositionCommitment
        self.friProof = friProof
        self.queryResponses = queryResponses
        self.alpha = alpha
        self.traceLength = traceLength
        self.numColumns = numColumns
        self.logBlowup = logBlowup
    }
}

// MARK: - Prover

public class CircleSTARKProver {
    public static let version = Versions.circleSTARK

    public let logBlowup: Int
    public let numQueries: Int

    private var nttEng: CircleNTTEngine?
    private var friEng: CircleFRIEngine?
    private var merkleEng: KeccakMerkleEngine?
    private var witnessEng: WitnessEngine?
    private var cstPipeline: MTLComputePipelineState?
    private var fusedSepColPipeline: MTLComputePipelineState?
    /// Cached domain_y buffer for fused constraint eval
    private var cachedDomainYBuf: MTLBuffer?
    private var cachedDomainYLogN: Int = 0

    public init(logBlowup: Int = 4, numQueries: Int = 30) {
        self.logBlowup = logBlowup
        self.numQueries = numQueries
    }

    private func ensureWitness() throws -> WitnessEngine {
        if let e = witnessEng { return e }
        let e = try WitnessEngine()
        witnessEng = e
        return e
    }

    /// GPU-accelerated trace generation for FibonacciAIR.
    /// Falls back to CPU generateTrace() for non-Fibonacci AIRs.
    public func generateTraceGPU<A: CircleAIR>(air: A) throws -> [[M31]] {
        if let fibAIR = air as? FibonacciAIR {
            let witness = try ensureWitness()
            let (colA, colB) = try witness.generateFibonacciTrace(
                a0: fibAIR.a0, b0: fibAIR.b0, numRows: fibAIR.traceLength
            )
            return [colA, colB]
        }
        return air.generateTrace()
    }

    private func ensureNTT() throws -> CircleNTTEngine {
        if let e = nttEng { return e }
        let e = try CircleNTTEngine()
        nttEng = e
        return e
    }

    private func ensureFRI() throws -> CircleFRIEngine {
        if let e = friEng { return e }
        let e = try CircleFRIEngine()
        friEng = e
        return e
    }

    private func ensureMerkle() throws -> KeccakMerkleEngine {
        if let e = merkleEng { return e }
        let e = try KeccakMerkleEngine()
        merkleEng = e
        return e
    }

    private func ensureConstraintPipeline() throws -> MTLComputePipelineState {
        if let p = cstPipeline { return p }
        let dev = try ensureNTT().device
        let sd = CircleSTARKProver.shaderDir()
        let fSrc = try String(contentsOfFile: sd + "/fields/mersenne31.metal", encoding: .utf8)
        let cSrc = try String(contentsOfFile: sd + "/constraint/circle_constraint_m31.metal", encoding: .utf8)
        let cClean = cSrc.split(separator: "\n").filter { !$0.contains("#include") }.joined(separator: "\n")
        let fClean = fSrc
            .replacingOccurrences(of: "#ifndef MERSENNE31_METAL", with: "")
            .replacingOccurrences(of: "#define MERSENNE31_METAL", with: "")
            .replacingOccurrences(of: "#endif // MERSENNE31_METAL", with: "")
        let opts = MTLCompileOptions()
        opts.fastMathEnabled = true
        let lib = try dev.makeLibrary(source: fClean + "\n" + cClean, options: opts)
        guard let fn = lib.makeFunction(name: "circle_fibonacci_constraint_eval") else {
            throw MSMError.missingKernel
        }
        let p = try dev.makeComputePipelineState(function: fn)
        cstPipeline = p
        return p
    }

    private func ensureFusedSepColPipeline() throws -> MTLComputePipelineState {
        if let p = fusedSepColPipeline { return p }
        let dev = try ensureNTT().device
        let sd = CircleSTARKProver.shaderDir()
        let fSrc = try String(contentsOfFile: sd + "/fields/mersenne31.metal", encoding: .utf8)
        let cSrc = try String(contentsOfFile: sd + "/constraint/fused_circle_ntt_constraint.metal", encoding: .utf8)
        let cClean = cSrc.split(separator: "\n").filter { !$0.contains("#include") }.joined(separator: "\n")
        let fClean = fSrc
            .replacingOccurrences(of: "#ifndef MERSENNE31_METAL", with: "")
            .replacingOccurrences(of: "#define MERSENNE31_METAL", with: "")
            .replacingOccurrences(of: "#endif // MERSENNE31_METAL", with: "")
        let opts = MTLCompileOptions()
        opts.fastMathEnabled = true
        let lib = try dev.makeLibrary(source: fClean + "\n" + cClean, options: opts)
        guard let fn = lib.makeFunction(name: "circle_fib_constraint_separate_cols") else {
            throw MSMError.missingKernel
        }
        let p = try dev.makeComputePipelineState(function: fn)
        fusedSepColPipeline = p
        return p
    }

    private func getDomainYBuffer(logN: Int) throws -> MTLBuffer {
        if logN == cachedDomainYLogN, let buf = cachedDomainYBuf { return buf }
        let dev = try ensureNTT().device
        let n = 1 << logN
        let dom = circleCosetDomain(logN: logN)
        let yVals = dom.map { $0.y.v }
        guard let buf = dev.makeBuffer(bytes: yVals, length: n * MemoryLayout<UInt32>.stride, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to create domain_y buffer")
        }
        cachedDomainYBuf = buf
        cachedDomainYLogN = logN
        return buf
    }

    private static func shaderDir() -> String {
        let execDir = (CommandLine.arguments[0] as NSString).deletingLastPathComponent
        for b in Bundle.allBundles {
            if let url = b.url(forResource: "Shaders", withExtension: nil) {
                if FileManager.default.fileExists(atPath: url.appendingPathComponent("fields/mersenne31.metal").path) {
                    return url.path
                }
            }
        }
        for p in ["\(execDir)/../Sources/Shaders", "./Sources/Shaders"] {
            if FileManager.default.fileExists(atPath: "\(p)/fields/mersenne31.metal") { return p }
        }
        return "./Sources/Shaders"
    }

    public var profileProve = false

    /// Prove that the given AIR is satisfied. GPU-accelerated.
    public func prove<A: CircleAIR>(air: A) throws -> CircleSTARKProof {
        let traceLen = air.traceLength
        let logTrace = air.logTraceLength
        let logEval = logTrace + logBlowup
        let evalLen = 1 << logEval

        let proveT0 = CFAbsoluteTimeGetCurrent()
        // Step 1: Generate trace (GPU-accelerated for supported AIRs)
        let trace = try generateTraceGPU(air: air)
        precondition(trace.count == air.numColumns)
        for col in trace { precondition(col.count == traceLen) }
        let traceT = CFAbsoluteTimeGetCurrent()

        // Step 2: LDE via GPU Circle NTT
        let ntt = try ensureNTT()
        var traceLDEs = [[M31]]()
        traceLDEs.reserveCapacity(air.numColumns)
        for col in trace {
            let coeffs = try ntt.intt(col)
            var padded = [M31](repeating: M31.zero, count: evalLen)
            for i in 0..<traceLen { padded[i] = coeffs[i] }
            let evals = try ntt.ntt(padded)
            traceLDEs.append(evals)
        }

        let ldeT = CFAbsoluteTimeGetCurrent()

        // Step 3: Commit trace columns via GPU Keccak Merkle trees
        let merkle = try ensureMerkle()
        var traceFlatTrees = [[UInt8]]()
        var traceCommitments = [[UInt8]]()
        for col in traceLDEs {
            let vals = col.map { $0.v }
            let flatTree = try merkle.buildTreeFromM31(vals, count: evalLen)
            traceCommitments.append(KeccakMerkleEngine.rootFromFlat(flatTree, n: evalLen))
            traceFlatTrees.append(flatTree)
        }

        let commitTraceT = CFAbsoluteTimeGetCurrent()

        // Step 4: Fiat-Shamir
        var transcript = CircleSTARKTranscript()
        transcript.absorbLabel("circle-stark-v1")
        for root in traceCommitments { transcript.absorbBytes(root) }
        let alpha = transcript.squeezeM31()

        let fsT = CFAbsoluteTimeGetCurrent()

        // Step 5: GPU constraint evaluation
        let compositionEvals = try gpuConstraintEval(
            air: air, traceLDEs: traceLDEs, alpha: alpha,
            logTrace: logTrace, logEval: logEval
        )

        let constraintT = CFAbsoluteTimeGetCurrent()

        // Step 6: Commit composition polynomial (GPU Merkle)
        let compVals = compositionEvals.map { $0.v }
        let compFlatTree = try merkle.buildTreeFromM31(compVals, count: evalLen)
        let compositionCommitment = KeccakMerkleEngine.rootFromFlat(compFlatTree, n: evalLen)

        let commitCompT = CFAbsoluteTimeGetCurrent()

        // Step 7: CPU FRI (matches verifier's fold formula exactly)
        transcript.absorbBytes(compositionCommitment)
        let friProof = try cpuFRI(
            evals: compositionEvals, logN: logEval,
            numQueries: numQueries, transcript: &transcript
        )

        let friT = CFAbsoluteTimeGetCurrent()

        // Step 8: Query phase (using flat GPU tree layout)
        var queryResponses = [CircleSTARKQueryResponse]()
        for qi in friProof.queryIndices {
            guard qi < evalLen else { continue }
            var traceVals = [M31]()
            var tracePaths = [[[UInt8]]]()
            for colIdx in 0..<traceLDEs.count {
                traceVals.append(traceLDEs[colIdx][qi])
                tracePaths.append(KeccakMerkleEngine.merkleProofFlat(traceFlatTrees[colIdx], n: evalLen, index: qi))
            }
            queryResponses.append(CircleSTARKQueryResponse(
                traceValues: traceVals, tracePaths: tracePaths,
                compositionValue: compositionEvals[qi],
                compositionPath: KeccakMerkleEngine.merkleProofFlat(compFlatTree, n: evalLen, index: qi),
                queryIndex: qi
            ))
        }

        let queryT = CFAbsoluteTimeGetCurrent()

        if profileProve {
            let fmt = { (label: String, t0: Double, t1: Double) -> String in
                String(format: "  %-20s %7.1f ms", (label as NSString).utf8String!, (t1 - t0) * 1000)
            }
            fputs("Circle STARK prove profile (2^\(logTrace)):\n", stderr)
            fputs(fmt("trace gen", proveT0, traceT) + "\n", stderr)
            fputs(fmt("LDE (NTT)", traceT, ldeT) + "\n", stderr)
            fputs(fmt("commit trace", ldeT, commitTraceT) + "\n", stderr)
            fputs(fmt("Fiat-Shamir", commitTraceT, fsT) + "\n", stderr)
            fputs(fmt("constraint eval", fsT, constraintT) + "\n", stderr)
            fputs(fmt("commit comp", constraintT, commitCompT) + "\n", stderr)
            fputs(fmt("CPU FRI", commitCompT, friT) + "\n", stderr)
            fputs(fmt("query phase", friT, queryT) + "\n", stderr)
            fputs(String(format: "  %-20s %7.1f ms\n", ("TOTAL" as NSString).utf8String!, (queryT - proveT0) * 1000), stderr)
        }

        return CircleSTARKProof(
            traceCommitments: traceCommitments,
            compositionCommitment: compositionCommitment,
            friProof: friProof, queryResponses: queryResponses,
            alpha: alpha, traceLength: traceLen,
            numColumns: air.numColumns, logBlowup: logBlowup
        )
    }

    // MARK: - GPU Constraint Evaluation

    private func gpuConstraintEval<A: CircleAIR>(
        air: A, traceLDEs: [[M31]], alpha: M31,
        logTrace: Int, logEval: Int
    ) throws -> [M31] {
        let evalLen = 1 << logEval
        let sz = MemoryLayout<UInt32>.stride
        let dev = try ensureNTT().device
        let queue = try ensureNTT().commandQueue
        let pipe = try ensureConstraintPipeline()

        guard let aBuf = dev.makeBuffer(length: evalLen * sz, options: .storageModeShared),
              let bBuf = dev.makeBuffer(length: evalLen * sz, options: .storageModeShared),
              let yBuf = dev.makeBuffer(length: evalLen * sz, options: .storageModeShared),
              let oBuf = dev.makeBuffer(length: evalLen * sz, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate constraint eval buffers")
        }

        let pA = aBuf.contents().bindMemory(to: UInt32.self, capacity: evalLen)
        let pB = bBuf.contents().bindMemory(to: UInt32.self, capacity: evalLen)
        for i in 0..<evalLen { pA[i] = traceLDEs[0][i].v; pB[i] = traceLDEs[1][i].v }

        let dom = circleCosetDomain(logN: logEval)
        let pY = yBuf.contents().bindMemory(to: UInt32.self, capacity: evalLen)
        for i in 0..<evalLen { pY[i] = dom[i].y.v }

        guard let cb = queue.makeCommandBuffer() else { throw MSMError.noCommandBuffer }
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipe)
        enc.setBuffer(aBuf, offset: 0, index: 0)
        enc.setBuffer(bBuf, offset: 0, index: 1)
        enc.setBuffer(yBuf, offset: 0, index: 2)
        enc.setBuffer(oBuf, offset: 0, index: 3)

        var av = alpha.v; enc.setBytes(&av, length: sz, index: 4)
        var a0: UInt32 = 0; var b0: UInt32 = 0
        for bc in air.boundaryConstraints {
            if bc.column == 0 { a0 = bc.value.v }
            if bc.column == 1 { b0 = bc.value.v }
        }
        enc.setBytes(&a0, length: sz, index: 5)
        enc.setBytes(&b0, length: sz, index: 6)
        var el = UInt32(evalLen); enc.setBytes(&el, length: sz, index: 7)
        var tl = UInt32(1 << logTrace); enc.setBytes(&tl, length: sz, index: 8)
        var lt = UInt32(logTrace); enc.setBytes(&lt, length: sz, index: 9)

        let tg = min(256, Int(pipe.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: evalLen, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { throw MSMError.gpuError(err.localizedDescription) }

        let oP = oBuf.contents().bindMemory(to: UInt32.self, capacity: evalLen)
        var result = [M31](repeating: M31.zero, count: evalLen)
        for i in 0..<evalLen { result[i] = M31(v: oP[i]) }
        return result
    }

    // MARK: - Fused LDE + Constraint Evaluation (single command buffer)

    /// Fused LDE + constraint evaluation: INTT -> zero-pad -> NTT -> constraint eval
    /// all in a single command buffer. Returns (compositionEvals, traceLDE_a, traceLDE_b).
    /// Avoids host round-trips between NTT and constraint evaluation.
    public func fusedLDEAndConstraintEval<A: CircleAIR>(
        air: A, trace: [[M31]], alpha: M31,
        logTrace: Int, logEval: Int
    ) throws -> (composition: [M31], ldeA: [M31], ldeB: [M31]) {
        let traceLen = 1 << logTrace
        let evalLen = 1 << logEval
        let sz = MemoryLayout<UInt32>.stride
        let ntt = try ensureNTT()
        let dev = ntt.device
        let queue = ntt.commandQueue
        let pipe = try ensureFusedSepColPipeline()
        let domainYBuf = try getDomainYBuffer(logN: logEval)

        // Allocate GPU buffers for both columns at eval domain size
        guard let bufA = dev.makeBuffer(length: evalLen * sz, options: .storageModeShared),
              let bufB = dev.makeBuffer(length: evalLen * sz, options: .storageModeShared),
              let oBuf = dev.makeBuffer(length: evalLen * sz, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate fused LDE+constraint buffers")
        }

        // Copy trace data into buffers
        let pA = bufA.contents().bindMemory(to: UInt32.self, capacity: evalLen)
        let pB = bufB.contents().bindMemory(to: UInt32.self, capacity: evalLen)
        for i in 0..<traceLen { pA[i] = trace[0][i].v; pB[i] = trace[1][i].v }
        // Zero-pad will happen after INTT

        // Phase 1: INTT both columns (trace -> coefficients)
        // Use separate command buffer for INTT since we need to zero-pad after
        guard let cb1 = queue.makeCommandBuffer() else { throw MSMError.noCommandBuffer }
        ntt.encodeINTT(data: bufA, logN: logTrace, cmdBuf: cb1)
        ntt.encodeINTT(data: bufB, logN: logTrace, cmdBuf: cb1)
        cb1.commit()
        cb1.waitUntilCompleted()
        if let err = cb1.error { throw MSMError.gpuError("INTT error: \(err.localizedDescription)") }

        // Zero-pad from traceLen to evalLen (coefficients beyond traceLen are zero)
        for i in traceLen..<evalLen { pA[i] = 0; pB[i] = 0 }

        // Phase 2: Single command buffer — NTT both columns + constraint eval
        guard let cb2 = queue.makeCommandBuffer() else { throw MSMError.noCommandBuffer }

        // NTT column A
        ntt.encodeNTT(data: bufA, logN: logEval, cmdBuf: cb2)
        // NTT column B
        ntt.encodeNTT(data: bufB, logN: logEval, cmdBuf: cb2)

        // Constraint evaluation (reads NTT'd data directly, no host round-trip)
        var a0: UInt32 = 0; var b0: UInt32 = 0
        for bc in air.boundaryConstraints {
            if bc.column == 0 { a0 = bc.value.v }
            if bc.column == 1 { b0 = bc.value.v }
        }

        let enc = cb2.makeComputeCommandEncoder()!
        enc.memoryBarrier(scope: .buffers)
        enc.setComputePipelineState(pipe)
        enc.setBuffer(bufA, offset: 0, index: 0)
        enc.setBuffer(bufB, offset: 0, index: 1)
        enc.setBuffer(oBuf, offset: 0, index: 2)
        enc.setBuffer(domainYBuf, offset: 0, index: 3)
        var av = alpha.v; enc.setBytes(&av, length: sz, index: 4)
        enc.setBytes(&a0, length: sz, index: 5)
        enc.setBytes(&b0, length: sz, index: 6)
        var el = UInt32(evalLen); enc.setBytes(&el, length: sz, index: 7)
        var tl = UInt32(traceLen); enc.setBytes(&tl, length: sz, index: 8)
        var lt = UInt32(logTrace); enc.setBytes(&lt, length: sz, index: 9)

        let tg = min(256, Int(pipe.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: evalLen, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cb2.commit()
        cb2.waitUntilCompleted()
        if let err = cb2.error { throw MSMError.gpuError("Fused NTT+constraint error: \(err.localizedDescription)") }

        // Read back results
        let oP = oBuf.contents().bindMemory(to: UInt32.self, capacity: evalLen)
        var composition = [M31](repeating: M31.zero, count: evalLen)
        for i in 0..<evalLen { composition[i] = M31(v: oP[i]) }

        // Also read back LDE evaluations (needed for Merkle commitments and queries)
        var ldeA = [M31](repeating: M31.zero, count: evalLen)
        var ldeB = [M31](repeating: M31.zero, count: evalLen)
        for i in 0..<evalLen { ldeA[i] = M31(v: pA[i]); ldeB[i] = M31(v: pB[i]) }

        return (composition, ldeA, ldeB)
    }

    /// Prove using fused LDE + constraint evaluation (single command buffer path).
    /// Eliminates host round-trip between NTT and constraint evaluation.
    public func proveFused<A: CircleAIR>(air: A) throws -> CircleSTARKProof {
        let traceLen = air.traceLength
        let logTrace = air.logTraceLength
        let logEval = logTrace + logBlowup
        let evalLen = 1 << logEval

        let proveT0 = CFAbsoluteTimeGetCurrent()

        // Step 1: Generate trace (GPU-accelerated for supported AIRs)
        let trace = try generateTraceGPU(air: air)
        precondition(trace.count == air.numColumns)
        for col in trace { precondition(col.count == traceLen) }
        let traceT = CFAbsoluteTimeGetCurrent()

        // Step 2+5 Fused: LDE + constraint evaluation in single command buffer
        let merkle = try ensureMerkle()

        var transcript = CircleSTARKTranscript()
        transcript.absorbLabel("circle-stark-v1")

        // We need alpha before constraint eval, but alpha depends on trace commitments.
        // So we do LDE first (needed for commitments), then fuse NTT + constraint.
        // Alternative: do INTT + pad + NTT, commit, then do constraint eval in a second pass.
        // The key fused step is: NTT output stays on GPU -> constraint eval reads it directly.

        // Step 2: LDE via GPU Circle NTT (INTT -> pad -> NTT)
        let ntt = try ensureNTT()
        let sz = MemoryLayout<UInt32>.stride
        let dev = ntt.device
        let queue = ntt.commandQueue

        // Allocate GPU buffers at eval domain size
        guard let bufA = dev.makeBuffer(length: evalLen * sz, options: .storageModeShared),
              let bufB = dev.makeBuffer(length: evalLen * sz, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate LDE buffers")
        }

        // Copy trace into buffers
        let pA = bufA.contents().bindMemory(to: UInt32.self, capacity: evalLen)
        let pB = bufB.contents().bindMemory(to: UInt32.self, capacity: evalLen)
        for i in 0..<traceLen { pA[i] = trace[0][i].v; pB[i] = trace[1][i].v }

        // INTT both columns to get coefficients
        guard let cb1 = queue.makeCommandBuffer() else { throw MSMError.noCommandBuffer }
        ntt.encodeINTT(data: bufA, logN: logTrace, cmdBuf: cb1)
        ntt.encodeINTT(data: bufB, logN: logTrace, cmdBuf: cb1)
        cb1.commit()
        cb1.waitUntilCompleted()
        if let err = cb1.error { throw MSMError.gpuError("INTT error: \(err.localizedDescription)") }

        // Zero-pad to eval domain
        for i in traceLen..<evalLen { pA[i] = 0; pB[i] = 0 }

        // NTT to get LDE evaluations
        guard let cb2 = queue.makeCommandBuffer() else { throw MSMError.noCommandBuffer }
        ntt.encodeNTT(data: bufA, logN: logEval, cmdBuf: cb2)
        ntt.encodeNTT(data: bufB, logN: logEval, cmdBuf: cb2)
        cb2.commit()
        cb2.waitUntilCompleted()
        if let err = cb2.error { throw MSMError.gpuError("NTT LDE error: \(err.localizedDescription)") }

        // Read LDE results for commitments
        var traceLDEs = [[M31]](repeating: [M31](), count: 2)
        traceLDEs[0] = [M31](repeating: M31.zero, count: evalLen)
        traceLDEs[1] = [M31](repeating: M31.zero, count: evalLen)
        for i in 0..<evalLen { traceLDEs[0][i] = M31(v: pA[i]); traceLDEs[1][i] = M31(v: pB[i]) }

        let ldeT = CFAbsoluteTimeGetCurrent()

        // Step 3: Commit trace columns
        var traceFlatTrees = [[UInt8]]()
        var traceCommitments = [[UInt8]]()
        for col in traceLDEs {
            let vals = col.map { $0.v }
            let flatTree = try merkle.buildTreeFromM31(vals, count: evalLen)
            traceCommitments.append(KeccakMerkleEngine.rootFromFlat(flatTree, n: evalLen))
            traceFlatTrees.append(flatTree)
        }

        let commitTraceT = CFAbsoluteTimeGetCurrent()

        // Step 4: Fiat-Shamir
        for root in traceCommitments { transcript.absorbBytes(root) }
        let alpha = transcript.squeezeM31()
        let fsT = CFAbsoluteTimeGetCurrent()

        // Step 5: FUSED constraint evaluation — NTT output already on GPU in bufA/bufB
        // Dispatch constraint eval reading directly from the GPU NTT output buffers
        let pipe = try ensureFusedSepColPipeline()
        let domainYBuf = try getDomainYBuffer(logN: logEval)

        guard let oBuf = dev.makeBuffer(length: evalLen * sz, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate constraint output buffer")
        }

        var a0: UInt32 = 0; var b0: UInt32 = 0
        for bc in air.boundaryConstraints {
            if bc.column == 0 { a0 = bc.value.v }
            if bc.column == 1 { b0 = bc.value.v }
        }

        guard let cb3 = queue.makeCommandBuffer() else { throw MSMError.noCommandBuffer }
        let enc = cb3.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipe)
        enc.setBuffer(bufA, offset: 0, index: 0)
        enc.setBuffer(bufB, offset: 0, index: 1)
        enc.setBuffer(oBuf, offset: 0, index: 2)
        enc.setBuffer(domainYBuf, offset: 0, index: 3)
        var av = alpha.v; enc.setBytes(&av, length: sz, index: 4)
        enc.setBytes(&a0, length: sz, index: 5)
        enc.setBytes(&b0, length: sz, index: 6)
        var el = UInt32(evalLen); enc.setBytes(&el, length: sz, index: 7)
        var tl = UInt32(traceLen); enc.setBytes(&tl, length: sz, index: 8)
        var lt = UInt32(logTrace); enc.setBytes(&lt, length: sz, index: 9)

        let tg = min(256, Int(pipe.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: evalLen, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cb3.commit()
        cb3.waitUntilCompleted()
        if let err = cb3.error { throw MSMError.gpuError("Fused constraint eval error: \(err.localizedDescription)") }

        let oP = oBuf.contents().bindMemory(to: UInt32.self, capacity: evalLen)
        var compositionEvals = [M31](repeating: M31.zero, count: evalLen)
        for i in 0..<evalLen { compositionEvals[i] = M31(v: oP[i]) }

        let constraintT = CFAbsoluteTimeGetCurrent()

        // Step 6: Commit composition polynomial
        let compVals = compositionEvals.map { $0.v }
        let compFlatTree = try merkle.buildTreeFromM31(compVals, count: evalLen)
        let compositionCommitment = KeccakMerkleEngine.rootFromFlat(compFlatTree, n: evalLen)
        let commitCompT = CFAbsoluteTimeGetCurrent()

        // Step 7: CPU FRI
        transcript.absorbBytes(compositionCommitment)
        let friProof = try cpuFRI(
            evals: compositionEvals, logN: logEval,
            numQueries: numQueries, transcript: &transcript
        )
        let friT = CFAbsoluteTimeGetCurrent()

        // Step 8: Query phase
        var queryResponses = [CircleSTARKQueryResponse]()
        for qi in friProof.queryIndices {
            guard qi < evalLen else { continue }
            var traceVals = [M31]()
            var tracePaths = [[[UInt8]]]()
            for colIdx in 0..<traceLDEs.count {
                traceVals.append(traceLDEs[colIdx][qi])
                tracePaths.append(KeccakMerkleEngine.merkleProofFlat(traceFlatTrees[colIdx], n: evalLen, index: qi))
            }
            queryResponses.append(CircleSTARKQueryResponse(
                traceValues: traceVals, tracePaths: tracePaths,
                compositionValue: compositionEvals[qi],
                compositionPath: KeccakMerkleEngine.merkleProofFlat(compFlatTree, n: evalLen, index: qi),
                queryIndex: qi
            ))
        }
        let queryT = CFAbsoluteTimeGetCurrent()

        if profileProve {
            let fmt = { (label: String, t0: Double, t1: Double) -> String in
                String(format: "  %-20s %7.1f ms", (label as NSString).utf8String!, (t1 - t0) * 1000)
            }
            fputs("Circle STARK prove FUSED profile (2^\(logTrace)):\n", stderr)
            fputs(fmt("trace gen", proveT0, traceT) + "\n", stderr)
            fputs(fmt("LDE (NTT)", traceT, ldeT) + "\n", stderr)
            fputs(fmt("commit trace", ldeT, commitTraceT) + "\n", stderr)
            fputs(fmt("Fiat-Shamir", commitTraceT, fsT) + "\n", stderr)
            fputs(fmt("constraint eval", fsT, constraintT) + " [FUSED]\n", stderr)
            fputs(fmt("commit comp", constraintT, commitCompT) + "\n", stderr)
            fputs(fmt("CPU FRI", commitCompT, friT) + "\n", stderr)
            fputs(fmt("query phase", friT, queryT) + "\n", stderr)
            fputs(String(format: "  %-20s %7.1f ms\n", ("TOTAL" as NSString).utf8String!, (queryT - proveT0) * 1000), stderr)
        }

        return CircleSTARKProof(
            traceCommitments: traceCommitments,
            compositionCommitment: compositionCommitment,
            friProof: friProof, queryResponses: queryResponses,
            alpha: alpha, traceLength: traceLen,
            numColumns: air.numColumns, logBlowup: logBlowup
        )
    }

    // MARK: - CPU FRI (matches verifier fold formula exactly)

    private func cpuFRI(
        evals: [M31], logN: Int, numQueries: Int,
        transcript: inout CircleSTARKTranscript
    ) throws -> CircleFRIProofData {
        var currentEvals = evals
        var currentLogN = logN
        var rounds = [CircleFRIRound]()
        var friProfileFold = 0.0, friProfileMerkle = 0.0, friProfileQuery = 0.0

        transcript.absorbLabel("fri-queries")
        var queryIndices = [Int]()
        let evalLen = 1 << logN
        for _ in 0..<numQueries {
            queryIndices.append(Int(transcript.squeezeM31().v) % (evalLen / 2))
        }

        var currentQueryIndices = queryIndices
        let inv2 = m31Inverse(M31(v: 2))

        while currentLogN > 1 {
            let foldT0 = CFAbsoluteTimeGetCurrent()
            let n = 1 << currentLogN
            let half = n / 2
            let foldAlpha = transcript.squeezeM31()

            let domain = circleCosetDomain(logN: currentLogN)
            var twiddles = [M31](repeating: M31.zero, count: half)
            if rounds.isEmpty {
                for i in 0..<half { twiddles[i] = domain[i].y }
            } else {
                for i in 0..<half { twiddles[i] = domain[i].x }
            }

            // Batch-invert twiddles using Montgomery's trick: O(n) muls + 1 inv
            var invTwiddles = [M31](repeating: M31.zero, count: half)
            if half > 0 {
                var partials = [M31](repeating: M31.zero, count: half)
                partials[0] = twiddles[0]
                for i in 1..<half {
                    partials[i] = m31Mul(partials[i - 1], twiddles[i])
                }
                var acc = m31Inverse(partials[half - 1])
                for i in stride(from: half - 1, through: 1, by: -1) {
                    invTwiddles[i] = m31Mul(acc, partials[i - 1])
                    acc = m31Mul(acc, twiddles[i])
                }
                invTwiddles[0] = acc
            }

            var folded = [M31](repeating: M31.zero, count: half)
            for i in 0..<half {
                let fi = currentEvals[i]
                let fih = currentEvals[i + half]
                let sum = m31Mul(m31Add(fi, fih), inv2)
                let invTw2 = m31Mul(inv2, invTwiddles[i])
                let diff = m31Mul(m31Sub(fi, fih), invTw2)
                folded[i] = m31Add(sum, m31Mul(foldAlpha, diff))
            }
            let foldT1 = CFAbsoluteTimeGetCurrent()
            friProfileFold += foldT1 - foldT0

            let merkleT0 = CFAbsoluteTimeGetCurrent()
            let merkle = try ensureMerkle()
            let foldedVals = folded.map { $0.v }
            let foldedFlatTree = try merkle.buildTreeFromM31(foldedVals, count: half)
            let commitment = KeccakMerkleEngine.rootFromFlat(foldedFlatTree, n: half)
            transcript.absorbBytes(commitment)
            friProfileMerkle += CFAbsoluteTimeGetCurrent() - merkleT0

            let queryT0 = CFAbsoluteTimeGetCurrent()
            var roundQueries = [(M31, M31, [[UInt8]])]()
            for qi in currentQueryIndices {
                let idx = qi % half
                roundQueries.append((currentEvals[idx], currentEvals[idx + half],
                                     KeccakMerkleEngine.merkleProofFlat(foldedFlatTree, n: half, index: idx)))
            }
            friProfileQuery += CFAbsoluteTimeGetCurrent() - queryT0

            rounds.append(CircleFRIRound(commitment: commitment, queryResponses: roundQueries))
            currentEvals = folded
            currentLogN -= 1
            currentQueryIndices = currentQueryIndices.map { $0 % max(half / 2, 1) }
        }

        if profileProve {
            fputs(String(format: "  FRI breakdown: fold %.1fms, merkle %.1fms, query %.1fms\n",
                        friProfileFold * 1000, friProfileMerkle * 1000, friProfileQuery * 1000), stderr)
        }

        return CircleFRIProofData(
            rounds: rounds, finalValue: currentEvals[0], queryIndices: queryIndices
        )
    }
}

// MARK: - Circle Domain Utilities

public func circleVanishing(point: CirclePoint, logDomainSize: Int) -> M31 {
    var v = point.y
    for _ in 0..<logDomainSize {
        v = m31Sub(m31Add(m31Sqr(v), m31Sqr(v)), M31.one)
    }
    return v
}

// MARK: - Keccak Merkle Tree Utilities

@inline(__always)
func m31ToBytes(_ v: M31) -> [UInt8] {
    var val = v.v
    return withUnsafeBytes(of: &val) { Array($0) }
}

func buildKeccakMerkle(_ leafHashes: [[UInt8]]) -> [[UInt8]] {
    let n = leafHashes.count
    precondition(n > 0 && (n & (n - 1)) == 0, "leaf count must be power of 2")
    var tree = [[UInt8]](repeating: [UInt8](repeating: 0, count: 32), count: 2 * n)
    for i in 0..<n { tree[n + i] = leafHashes[i] }
    var levelSize = n / 2
    var offset = n / 2
    while levelSize >= 1 {
        for i in 0..<levelSize {
            let idx = offset + i
            tree[idx] = keccak256(tree[2 * idx] + tree[2 * idx + 1])
        }
        levelSize /= 2
        offset /= 2
    }
    return tree
}

func merkleProof(tree: [[UInt8]], index: Int) -> [[UInt8]] {
    let n = tree.count / 2
    var path = [[UInt8]]()
    var idx = n + index
    while idx > 1 {
        path.append(tree[idx ^ 1])
        idx /= 2
    }
    return path
}

func verifyMerkleProof(leafHash: [UInt8], path: [[UInt8]], index: Int, root: [UInt8]) -> Bool {
    var current = leafHash
    var idx = index
    for sibling in path {
        if idx & 1 == 0 {
            current = keccak256(current + sibling)
        } else {
            current = keccak256(sibling + current)
        }
        idx /= 2
    }
    return current == root
}

// MARK: - Lightweight Circle STARK Transcript (M31-native)

public struct CircleSTARKTranscript {
    private var state: [UInt8]

    public init() {
        self.state = [UInt8](repeating: 0, count: 32)
    }

    public mutating func absorbLabel(_ label: String) {
        let bytes = Array(label.utf8)
        var len = UInt32(bytes.count)
        let lenBytes = withUnsafeBytes(of: &len) { Array($0) }
        absorbBytes(lenBytes + bytes)
    }

    public mutating func absorbBytes(_ data: [UInt8]) {
        state = keccak256(state + data)
    }

    public mutating func absorbM31(_ v: M31) {
        absorbBytes(m31ToBytes(v))
    }

    public mutating func squeezeM31() -> M31 {
        state = keccak256(state + [0x01])
        let raw = UInt32(state[0]) | (UInt32(state[1]) << 8) |
                  (UInt32(state[2]) << 16) | (UInt32(state[3]) << 24)
        let val = raw & M31.P
        return M31(v: val == M31.P ? 0 : val)
    }
}
