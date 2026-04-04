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
    private var cstPipeline: MTLComputePipelineState?

    public init(logBlowup: Int = 4, numQueries: Int = 30) {
        self.logBlowup = logBlowup
        self.numQueries = numQueries
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

    /// Prove that the given AIR is satisfied. GPU-accelerated.
    public func prove<A: CircleAIR>(air: A) throws -> CircleSTARKProof {
        let traceLen = air.traceLength
        let logTrace = air.logTraceLength
        let logEval = logTrace + logBlowup
        let evalLen = 1 << logEval

        // Step 1: Generate trace
        let trace = air.generateTrace()
        precondition(trace.count == air.numColumns)
        for col in trace { precondition(col.count == traceLen) }

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

        // Step 3: Commit trace columns via Keccak Merkle trees
        var traceTrees = [[[UInt8]]]()
        var traceCommitments = [[UInt8]]()
        for col in traceLDEs {
            let leafHashes = col.map { keccak256(m31ToBytes($0)) }
            let tree = buildKeccakMerkle(leafHashes)
            traceCommitments.append(tree[1])
            traceTrees.append(tree)
        }

        // Step 4: Fiat-Shamir
        var transcript = CircleSTARKTranscript()
        transcript.absorbLabel("circle-stark-v1")
        for root in traceCommitments { transcript.absorbBytes(root) }
        let alpha = transcript.squeezeM31()

        // Step 5: GPU constraint evaluation
        let compositionEvals = try gpuConstraintEval(
            air: air, traceLDEs: traceLDEs, alpha: alpha,
            logTrace: logTrace, logEval: logEval
        )

        // Step 6: Commit composition polynomial
        let compLeafHashes = compositionEvals.map { keccak256(m31ToBytes($0)) }
        let compTree = buildKeccakMerkle(compLeafHashes)
        let compositionCommitment = compTree[1]

        // Step 7: GPU FRI
        transcript.absorbBytes(compositionCommitment)
        let friProof = try gpuFRI(
            evals: compositionEvals, logN: logEval,
            numQueries: numQueries, transcript: &transcript
        )

        // Step 8: Query phase
        var queryResponses = [CircleSTARKQueryResponse]()
        for qi in friProof.queryIndices {
            guard qi < evalLen else { continue }
            var traceVals = [M31]()
            var tracePaths = [[[UInt8]]]()
            for colIdx in 0..<traceLDEs.count {
                traceVals.append(traceLDEs[colIdx][qi])
                tracePaths.append(merkleProof(tree: traceTrees[colIdx], index: qi))
            }
            queryResponses.append(CircleSTARKQueryResponse(
                traceValues: traceVals, tracePaths: tracePaths,
                compositionValue: compositionEvals[qi],
                compositionPath: merkleProof(tree: compTree, index: qi),
                queryIndex: qi
            ))
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

    // MARK: - GPU FRI

    private func gpuFRI(
        evals: [M31], logN: Int, numQueries: Int,
        transcript: inout CircleSTARKTranscript
    ) throws -> CircleFRIProofData {
        let fri = try ensureFRI()
        var currentEvals = evals
        var currentLogN = logN
        var rounds = [CircleFRIRound]()

        transcript.absorbLabel("fri-queries")
        var queryIndices = [Int]()
        let evalLen = 1 << logN
        for _ in 0..<numQueries {
            queryIndices.append(Int(transcript.squeezeM31().v) % (evalLen / 2))
        }

        var currentQueryIndices = queryIndices
        var xFoldRound = 0

        while currentLogN > 1 {
            let n = 1 << currentLogN
            let half = n / 2
            let foldAlpha = transcript.squeezeM31()
            let isFirst = rounds.isEmpty

            let folded = try fri.fold(
                evals: currentEvals, alpha: foldAlpha,
                logN: currentLogN, isFirstFold: isFirst,
                xFoldRound: isFirst ? 0 : xFoldRound
            )
            if !isFirst { xFoldRound += 1 }

            let foldedLeaves = folded.map { keccak256(m31ToBytes($0)) }
            let foldedTree = buildKeccakMerkle(foldedLeaves)
            let commitment = foldedTree[1]
            transcript.absorbBytes(commitment)

            var roundQueries = [(M31, M31, [[UInt8]])]()
            for qi in currentQueryIndices {
                let idx = qi % half
                roundQueries.append((currentEvals[idx], currentEvals[idx + half],
                                     merkleProof(tree: foldedTree, index: idx)))
            }

            rounds.append(CircleFRIRound(commitment: commitment, queryResponses: roundQueries))
            currentEvals = folded
            currentLogN -= 1
            currentQueryIndices = currentQueryIndices.map { $0 % max(half / 2, 1) }
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
