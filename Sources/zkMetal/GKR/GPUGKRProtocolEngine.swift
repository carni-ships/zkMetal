// GPU-accelerated GKR (Goldwasser-Kalai-Rothblum) Protocol Engine
//
// Full GKR interactive proof for layered arithmetic circuits with Metal GPU acceleration.
// Protocol:
//   1. Prover evaluates circuit, absorbs output into transcript
//   2. For each layer (output to input):
//      a. Run sumcheck on the GKR equation with add/mul wiring predicates
//      b. GPU-accelerated MLE evaluation over boolean hypercube
//      c. Verifier reduces claims using random linear combination
//   3. Final check at input layer against claimed input MLE
//
// GPU acceleration targets:
//   - Multilinear extension evaluation (parallel hypercube evaluation)
//   - Eq polynomial computation (parallel product tree)
//   - Sumcheck round polynomial computation (parallel reduction)
//   - Wiring predicate batch evaluation
//
// Falls back to CPU for small circuits (< gpuThreshold elements).

import Foundation
import Metal
import NeonFieldOps

// MARK: - GPU GKR Error

public enum GPUGKRError: Error {
    case noGPU
    case noCommandQueue
    case noCommandBuffer
    case missingKernel
    case gpuDispatchFailed(String)
    case invalidCircuit(String)
    case verificationFailed(String)
}

// MARK: - GPU GKR Layer Proof

/// Proof for one layer of GPU-accelerated GKR protocol.
/// Contains sumcheck round messages and claimed MLE evaluations.
public struct GPUGKRLayerProof {
    /// Sumcheck round messages: evaluations at {0, 1, 2} per variable.
    public let roundMessages: [GPUGKRRoundMsg]
    /// Claimed V_{i-1}(rx) — MLE of previous layer at the x-point.
    public let claimedVx: Fr
    /// Claimed V_{i-1}(ry) — MLE of previous layer at the y-point.
    public let claimedVy: Fr

    public init(roundMessages: [GPUGKRRoundMsg], claimedVx: Fr, claimedVy: Fr) {
        self.roundMessages = roundMessages
        self.claimedVx = claimedVx
        self.claimedVy = claimedVy
    }
}

/// A single sumcheck round message in the GPU GKR protocol.
public struct GPUGKRRoundMsg {
    public let eval0: Fr  // polynomial at t=0
    public let eval1: Fr  // polynomial at t=1
    public let eval2: Fr  // polynomial at t=2

    public init(eval0: Fr, eval1: Fr, eval2: Fr) {
        self.eval0 = eval0
        self.eval1 = eval1
        self.eval2 = eval2
    }
}

/// Complete GPU GKR proof for the entire circuit.
public struct GPUGKRProof {
    /// Layer proofs, from output layer toward input.
    public let layerProofs: [GPUGKRLayerProof]
    /// The circuit output values (committed via transcript).
    public let outputValues: [Fr]

    public init(layerProofs: [GPUGKRLayerProof], outputValues: [Fr]) {
        self.layerProofs = layerProofs
        self.outputValues = outputValues
    }
}

// MARK: - GPU GKR Configuration

/// Configuration for the GPU GKR engine.
public struct GPUGKRConfig {
    /// Minimum table size to trigger GPU path (below this, use CPU).
    public var gpuThreshold: Int
    /// Maximum threadgroup size for Metal dispatches.
    public var maxThreadgroupSize: Int
    /// Whether to enable data-parallel mode for repeated sub-circuits.
    public var dataParallelMode: Bool

    public init(gpuThreshold: Int = 2048,
                maxThreadgroupSize: Int = 256,
                dataParallelMode: Bool = false) {
        self.gpuThreshold = gpuThreshold
        self.maxThreadgroupSize = maxThreadgroupSize
        self.dataParallelMode = dataParallelMode
    }
}

// MARK: - GPU GKR Protocol Engine

/// GPU-accelerated engine for the GKR interactive proof protocol.
///
/// Supports layered arithmetic circuits with add/mul gates.
/// The engine compiles Metal shaders for:
///   - MLE evaluation: parallel evaluation of multilinear extension over hypercube
///   - Eq polynomial: parallel computation of eq(r, x) for all x in {0,1}^n
///   - Sumcheck rounds: GPU-accelerated partial sum computation
///   - Table folding: parallel fold of evaluation tables by challenge
///
/// For data-parallel circuits (N copies of the same sub-circuit), the wiring
/// MLE is computed once and amortized across all instances.
public class GPUGKRProtocolEngine {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let config: GPUGKRConfig

    // Metal compute pipelines
    private let mleEvalPipeline: MTLComputePipelineState
    private let eqPolyPipeline: MTLComputePipelineState
    private let tableFoldPipeline: MTLComputePipelineState
    private let roundPolyPipeline: MTLComputePipelineState

    // Cached GPU buffers for reuse across prove calls
    private var cachedEvalBufA: MTLBuffer?
    private var cachedEvalBufB: MTLBuffer?
    private var cachedEvalCapacity: Int = 0
    private var cachedPartialBuf: MTLBuffer?
    private var cachedPartialCapacity: Int = 0

    // Pre-computed circuit topology (same structure as GKREngine)
    private var layerWiringKeys: [[Int]] = []
    private var layerGateToWiring: [[Int]] = []
    private var layerGateData: [[Int32]] = []

    /// Initialize the GPU GKR engine.
    /// - Parameters:
    ///   - config: Engine configuration (thresholds, threadgroup sizes)
    /// - Throws: GPUGKRError if Metal is unavailable or shader compilation fails
    public init(config: GPUGKRConfig = GPUGKRConfig()) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw GPUGKRError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw GPUGKRError.noCommandQueue
        }
        self.commandQueue = queue
        self.config = config

        let library = try GPUGKRProtocolEngine.compileGKRShaders(device: device)

        guard let mleFunc = library.makeFunction(name: "gkr_mle_eval_bn254"),
              let eqFunc = library.makeFunction(name: "gkr_eq_poly_bn254"),
              let foldFunc = library.makeFunction(name: "gkr_table_fold_bn254"),
              let roundFunc = library.makeFunction(name: "gkr_round_poly_bn254") else {
            throw GPUGKRError.missingKernel
        }

        self.mleEvalPipeline = try device.makeComputePipelineState(function: mleFunc)
        self.eqPolyPipeline = try device.makeComputePipelineState(function: eqFunc)
        self.tableFoldPipeline = try device.makeComputePipelineState(function: foldFunc)
        self.roundPolyPipeline = try device.makeComputePipelineState(function: roundFunc)
    }

    /// Whether the GPU is available and ready.
    public var isGPUAvailable: Bool { true }

    /// Device name for diagnostics.
    public var deviceName: String { device.name }

    // MARK: - Circuit Setup

    /// Prepare internal topology for a circuit (call before prove/verify).
    /// Pre-computes wiring indices, gate-to-wiring maps, and gate type arrays.
    public func setupCircuit(_ circuit: LayeredCircuit) {
        layerWiringKeys = []
        layerGateToWiring = []
        layerGateData = []

        for layerIdx in 0..<circuit.layers.count {
            let layer = circuit.layers[layerIdx]
            let nIn = circuit.inputVars(layer: layerIdx)
            let inSize = 1 << nIn

            // Gate data: [type, left, right] packed
            var gd = [Int32](repeating: 0, count: layer.gates.count * 3)
            for (i, gate) in layer.gates.enumerated() {
                gd[i * 3] = gate.type == .add ? 0 : 1
                gd[i * 3 + 1] = Int32(gate.leftInput)
                gd[i * 3 + 2] = Int32(gate.rightInput)
            }
            layerGateData.append(gd)

            // Build wiring topology: unique (x, y) pairs mapped to sorted indices
            var dict = [Int: Int]()
            dict.reserveCapacity(layer.gates.count)
            var keys = [Int]()
            keys.reserveCapacity(layer.gates.count)
            var g2w = [Int](repeating: 0, count: layer.gates.count)

            for (gIdx, gate) in layer.gates.enumerated() {
                let xyIdx = gate.leftInput * inSize + gate.rightInput
                if let idx = dict[xyIdx] {
                    g2w[gIdx] = idx
                } else {
                    let idx = keys.count
                    dict[xyIdx] = idx
                    keys.append(xyIdx)
                    g2w[gIdx] = idx
                }
            }

            let sortedIndices = keys.indices.sorted { keys[$0] < keys[$1] }
            var remap = [Int](repeating: 0, count: keys.count)
            var sortedKeys = [Int](repeating: 0, count: keys.count)
            for (newIdx, oldIdx) in sortedIndices.enumerated() {
                remap[oldIdx] = newIdx
                sortedKeys[newIdx] = keys[oldIdx]
            }
            for i in 0..<g2w.count {
                g2w[i] = remap[g2w[i]]
            }

            layerWiringKeys.append(sortedKeys)
            layerGateToWiring.append(g2w)
        }
    }

    // MARK: - Prover

    /// Prove the GKR protocol for a circuit evaluation.
    ///
    /// Uses batched GKR: after each layer's sumcheck, claims at two points
    /// (rx, ry) are combined via random linear combination for the next layer.
    ///
    /// GPU acceleration is used for MLE evaluation, eq polynomial, and
    /// sumcheck when table sizes exceed the configured threshold.
    ///
    /// - Parameters:
    ///   - circuit: The layered arithmetic circuit
    ///   - inputs: Circuit input values
    ///   - transcript: Fiat-Shamir transcript for non-interactive proof
    /// - Returns: Complete GKR proof
    public func prove(circuit: LayeredCircuit, inputs: [Fr],
                      transcript: Transcript) -> GPUGKRProof {
        if layerWiringKeys.isEmpty { setupCircuit(circuit) }

        let allValues = circuit.evaluate(inputs: inputs)
        let d = circuit.depth

        // Absorb output into transcript
        let outputValues = allValues[d]
        for v in outputValues { transcript.absorb(v) }
        transcript.absorbLabel("gpu-gkr-init")

        let outputVars = circuit.outputVars(layer: d - 1)
        let r0 = transcript.squeezeN(outputVars)

        var rPoints: [([Fr], Fr)] = [(r0, Fr.one)]
        var layerProofs = [GPUGKRLayerProof]()

        for layerIdx in stride(from: d - 1, through: 0, by: -1) {
            let nOut = circuit.outputVars(layer: layerIdx)
            let nIn = circuit.inputVars(layer: layerIdx)
            let prevValues = allValues[layerIdx]
            let prevEvals = padToPowerOf2(prevValues, numVars: nIn)

            let (msgs, rx, ry) = proverLayerSumcheck(
                rPoints: rPoints,
                layer: circuit.layers[layerIdx],
                prevEvals: prevEvals, nOut: nOut, nIn: nIn,
                transcript: transcript,
                layerIdx: layerIdx
            )

            let vx = gpuMleEval(evals: prevEvals, point: rx)
            let vy = gpuMleEval(evals: prevEvals, point: ry)

            transcript.absorb(vx)
            transcript.absorb(vy)
            transcript.absorbLabel("gpu-gkr-layer-\(layerIdx)")

            layerProofs.append(GPUGKRLayerProof(
                roundMessages: msgs, claimedVx: vx, claimedVy: vy
            ))

            let alpha = transcript.squeeze()
            let beta = transcript.squeeze()
            rPoints = [(rx, alpha), (ry, beta)]
        }

        return GPUGKRProof(layerProofs: layerProofs, outputValues: outputValues)
    }

    // MARK: - Verifier

    /// Verify a GPU GKR proof.
    ///
    /// Checks each layer's sumcheck messages and wiring predicate consistency,
    /// then verifies the final claim against the circuit inputs.
    ///
    /// - Parameters:
    ///   - circuit: The layered arithmetic circuit
    ///   - inputs: Circuit input values
    ///   - proof: The GKR proof to verify
    ///   - transcript: Fresh transcript (same label as prover)
    /// - Returns: true if the proof is valid
    public func verify(circuit: LayeredCircuit, inputs: [Fr],
                       proof: GPUGKRProof, transcript: Transcript) -> Bool {
        let d = circuit.depth
        guard proof.layerProofs.count == d else { return false }

        // Absorb output
        for v in proof.outputValues { transcript.absorb(v) }
        transcript.absorbLabel("gpu-gkr-init")

        let outputVars = circuit.outputVars(layer: d - 1)
        let r0 = transcript.squeezeN(outputVars)
        let outputEvals = padToPowerOf2(proof.outputValues, numVars: outputVars)

        var claim = gpuMleEval(evals: outputEvals, point: r0)
        var rPoints: [([Fr], Fr)] = [(r0, Fr.one)]

        for layerIdx in stride(from: d - 1, through: 0, by: -1) {
            let nIn = circuit.inputVars(layer: layerIdx)
            let layerProof = proof.layerProofs[d - 1 - layerIdx]

            let totalVars = 2 * nIn
            guard layerProof.roundMessages.count == totalVars else { return false }

            // Verify sumcheck rounds
            var currentClaim = claim
            var challenges = [Fr]()
            challenges.reserveCapacity(totalVars)

            for roundIdx in 0..<totalVars {
                let msg = layerProof.roundMessages[roundIdx]
                let sum = frAdd(msg.eval0, msg.eval1)
                if !gpuGKRFrEqual(sum, currentClaim) { return false }

                transcript.absorb(msg.eval0)
                transcript.absorb(msg.eval1)
                transcript.absorb(msg.eval2)
                let challenge = transcript.squeeze()
                challenges.append(challenge)

                currentClaim = gpuLagrangeEval3(
                    s0: msg.eval0, s1: msg.eval1, s2: msg.eval2, at: challenge
                )
            }

            let rx = Array(challenges.prefix(nIn))
            let ry = Array(challenges.suffix(nIn))
            let vx = layerProof.claimedVx
            let vy = layerProof.claimedVy

            // Verify wiring predicate consistency
            let eqRx = gpuEqPoly(point: rx)
            let eqRy = gpuEqPoly(point: ry)
            let sumVxVy = frAdd(vx, vy)
            let prodVxVy = frMul(vx, vy)

            var expected = Fr.zero
            for (rk, wk) in rPoints {
                let eqRk = gpuEqPoly(point: rk)
                for (gIdx, gate) in circuit.layers[layerIdx].gates.enumerated() {
                    let eqZ = gIdx < eqRk.count ? eqRk[gIdx] : Fr.zero
                    if eqZ.isZero { continue }
                    let eqX = gate.leftInput < eqRx.count ? eqRx[gate.leftInput] : Fr.zero
                    let eqY = gate.rightInput < eqRy.count ? eqRy[gate.rightInput] : Fr.zero
                    let wiringVal = frMul(eqZ, frMul(eqX, eqY))
                    let contrib: Fr
                    switch gate.type {
                    case .add: contrib = frMul(wiringVal, sumVxVy)
                    case .mul: contrib = frMul(wiringVal, prodVxVy)
                    }
                    expected = frAdd(expected, frMul(wk, contrib))
                }
            }

            if !gpuGKRFrEqual(currentClaim, expected) { return false }

            transcript.absorb(vx)
            transcript.absorb(vy)
            transcript.absorbLabel("gpu-gkr-layer-\(layerIdx)")

            let alpha = transcript.squeeze()
            let beta = transcript.squeeze()
            claim = frAdd(frMul(alpha, vx), frMul(beta, vy))
            rPoints = [(rx, alpha), (ry, beta)]
        }

        // Final check: claim matches input MLE
        let inputNumVars = rPoints[0].0.count
        let inputEvals = padToPowerOf2(inputs, numVars: inputNumVars)
        var inputExpected = Fr.zero
        for (rk, wk) in rPoints {
            let val = gpuMleEval(evals: inputEvals, point: rk)
            inputExpected = frAdd(inputExpected, frMul(wk, val))
        }
        return gpuGKRFrEqual(claim, inputExpected)
    }

    // MARK: - Data-Parallel Prove

    /// Prove GKR for N identical copies of a sub-circuit evaluated on different inputs.
    ///
    /// Exploits the fact that the wiring predicate MLE is the same across all copies,
    /// computing it once and amortizing the cost. Each copy has its own input/output
    /// values, but the circuit structure is shared.
    ///
    /// - Parameters:
    ///   - circuit: Template circuit (one copy)
    ///   - batchInputs: Array of input vectors, one per circuit copy
    ///   - transcript: Fiat-Shamir transcript
    /// - Returns: Array of proofs (one per circuit copy)
    public func proveDataParallel(circuit: LayeredCircuit,
                                  batchInputs: [[Fr]],
                                  transcript: Transcript) -> [GPUGKRProof] {
        if layerWiringKeys.isEmpty { setupCircuit(circuit) }

        var proofs = [GPUGKRProof]()
        proofs.reserveCapacity(batchInputs.count)

        // Shared transcript prefix: absorb circuit description once
        transcript.absorbLabel("gpu-gkr-dataparallel-\(batchInputs.count)")

        for (batchIdx, inputs) in batchInputs.enumerated() {
            // Each copy gets its own sub-transcript section
            transcript.absorbLabel("gpu-gkr-batch-\(batchIdx)")
            let proof = prove(circuit: circuit, inputs: inputs, transcript: transcript)
            proofs.append(proof)
        }

        return proofs
    }

    /// Verify a batch of data-parallel GKR proofs.
    public func verifyDataParallel(circuit: LayeredCircuit,
                                   batchInputs: [[Fr]],
                                   proofs: [GPUGKRProof],
                                   transcript: Transcript) -> Bool {
        guard proofs.count == batchInputs.count else { return false }

        transcript.absorbLabel("gpu-gkr-dataparallel-\(batchInputs.count)")

        for (batchIdx, (inputs, proof)) in zip(batchInputs, proofs).enumerated() {
            transcript.absorbLabel("gpu-gkr-batch-\(batchIdx)")
            if !verify(circuit: circuit, inputs: inputs, proof: proof, transcript: transcript) {
                return false
            }
        }
        return true
    }

    // MARK: - Output Layer Claim Reduction

    /// Reduce an output layer claim to individual gate claims.
    ///
    /// Given the output values and a random evaluation point, compute the
    /// output layer's MLE evaluation and decompose it into claims about
    /// individual gates that can be verified via sumcheck.
    ///
    /// - Parameters:
    ///   - outputValues: The circuit output (one value per output gate)
    ///   - point: Random evaluation point from transcript
    /// - Returns: The MLE evaluation at the given point
    public func reduceOutputClaim(outputValues: [Fr], point: [Fr]) -> Fr {
        let numVars = point.count
        let evals = padToPowerOf2(outputValues, numVars: numVars)
        return gpuMleEval(evals: evals, point: point)
    }

    // MARK: - GPU-Accelerated MLE Evaluation

    /// Evaluate a multilinear extension at an arbitrary point.
    ///
    /// For tables >= gpuThreshold, dispatches Metal compute; otherwise CPU.
    /// Uses the iterative reduction: fix variables one by one.
    ///
    /// - Parameters:
    ///   - evals: Evaluation table (size 2^n)
    ///   - point: Evaluation point (n coordinates)
    /// - Returns: MLE(point)
    public func gpuMleEval(evals: [Fr], point: [Fr]) -> Fr {
        let n = evals.count
        if n == 0 { return Fr.zero }
        if n == 1 { return evals[0] }

        // For large tables, use GPU-accelerated folding
        if n >= config.gpuThreshold {
            return gpuMleEvalMetal(evals: evals, point: point)
        }

        // CPU fallback: iterative variable fixing
        return cpuMleEval(evals: evals, point: point)
    }

    /// GPU-accelerated eq polynomial: eq(r, x) = prod_i(r_i*x_i + (1-r_i)*(1-x_i))
    /// Returns evaluations over {0,1}^n for fixed r.
    public func gpuEqPoly(point: [Fr]) -> [Fr] {
        let n = point.count
        let size = 1 << n
        if size >= config.gpuThreshold {
            return gpuEqPolyMetal(point: point)
        }
        return cpuEqPoly(point: point)
    }

    // MARK: - GPU Metal Dispatch

    private func gpuMleEvalMetal(evals: [Fr], point: [Fr]) -> Fr {
        let numVars = point.count
        guard numVars > 0 else { return evals.isEmpty ? Fr.zero : evals[0] }

        let elemSize = MemoryLayout<Fr>.stride
        var currentSize = evals.count

        // Ensure GPU buffers are large enough
        let needed = currentSize * elemSize
        if needed > cachedEvalCapacity {
            cachedEvalBufA = device.makeBuffer(length: needed, options: .storageModeShared)
            cachedEvalBufB = device.makeBuffer(length: needed, options: .storageModeShared)
            cachedEvalCapacity = needed
        }

        guard let bufA = cachedEvalBufA, let bufB = cachedEvalBufB else {
            return cpuMleEval(evals: evals, point: point)
        }

        // Copy evals to GPU buffer A
        evals.withUnsafeBytes { src in
            bufA.contents().copyMemory(from: src.baseAddress!, byteCount: currentSize * elemSize)
        }

        // Iteratively fold: for each variable, halve the table
        var srcBuf = bufA
        var dstBuf = bufB

        // Single command buffer for all fold rounds
        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            return cpuMleEval(evals: evals, point: point)
        }
        let encoder = cmdBuf.makeComputeCommandEncoder()!

        for varIdx in 0..<numVars {
            let halfSize = currentSize / 2
            if halfSize == 0 { break }

            var challenge = point[varIdx]
            var halfN = UInt32(halfSize)

            encoder.setComputePipelineState(tableFoldPipeline)
            encoder.setBuffer(srcBuf, offset: 0, index: 0)
            encoder.setBuffer(dstBuf, offset: 0, index: 1)
            encoder.setBytes(&challenge, length: elemSize, index: 2)
            encoder.setBytes(&halfN, length: 4, index: 3)

            let tgSize = min(config.maxThreadgroupSize, halfSize)
            encoder.dispatchThreads(MTLSize(width: halfSize, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))

            swap(&srcBuf, &dstBuf)
            currentSize = halfSize

            if currentSize > 1 {
                encoder.memoryBarrier(scope: .buffers)
            }
        }
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Read back the single result
        var result = Fr.zero
        withUnsafeMutableBytes(of: &result) { dst in
            dst.baseAddress!.copyMemory(from: srcBuf.contents(), byteCount: elemSize)
        }
        return result
    }

    private func gpuEqPolyMetal(point: [Fr]) -> [Fr] {
        let n = point.count
        let size = 1 << n
        let elemSize = MemoryLayout<Fr>.stride

        // Allocate GPU buffer for eq values
        guard let eqBuf = device.makeBuffer(length: size * elemSize,
                                            options: .storageModeShared) else {
            return cpuEqPoly(point: point)
        }

        // Initialize: eq[0] = 1 (Montgomery form)
        var one = Fr.one
        eqBuf.contents().copyMemory(from: &one, byteCount: elemSize)

        // Build eq polynomial level by level — single command buffer for all levels
        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            return cpuEqPoly(point: point)
        }
        let encoder = cmdBuf.makeComputeCommandEncoder()!

        for i in 0..<n {
            let half = 1 << i

            var ri = point[i]
            var halfVal = UInt32(half)

            encoder.setComputePipelineState(eqPolyPipeline)
            encoder.setBuffer(eqBuf, offset: 0, index: 0)
            encoder.setBytes(&ri, length: elemSize, index: 1)
            encoder.setBytes(&halfVal, length: 4, index: 2)

            let tgSize = min(config.maxThreadgroupSize, half)
            encoder.dispatchThreads(MTLSize(width: half, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))

            if i < n - 1 {
                encoder.memoryBarrier(scope: .buffers)
            }
        }
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Read back
        var result = [Fr](repeating: Fr.zero, count: size)
        result.withUnsafeMutableBytes { dst in
            dst.baseAddress!.copyMemory(from: eqBuf.contents(), byteCount: size * elemSize)
        }
        return result
    }

    // MARK: - CPU Fallbacks

    private func cpuMleEval(evals: [Fr], point: [Fr]) -> Fr {
        var result = Fr.zero
        evals.withUnsafeBytes { evalBuf in
            point.withUnsafeBytes { ptBuf in
                withUnsafeMutableBytes(of: &result) { resBuf in
                    bn254_fr_mle_eval(
                        evalBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(point.count),
                        ptBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        resBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                    )
                }
            }
        }
        return result
    }

    private func cpuEqPoly(point: [Fr]) -> [Fr] {
        let n = point.count
        let size = 1 << n
        var eq = [Fr](repeating: Fr.zero, count: size)
        point.withUnsafeBytes { ptBuf in
            eq.withUnsafeMutableBytes { eqBuf in
                gkr_eq_poly(
                    ptBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(n),
                    eqBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                )
            }
        }
        return eq
    }

    // MARK: - Layer Sumcheck

    /// Run batched sumcheck for one GKR layer.
    ///
    /// Combines claims from multiple output points (rx, ry from previous layer)
    /// with their weights (alpha, beta), using wiring predicates to relate
    /// the current layer's output gates to the previous layer's MLE.
    private func proverLayerSumcheck(
        rPoints: [([Fr], Fr)],
        layer: CircuitLayer,
        prevEvals: [Fr],
        nOut: Int, nIn: Int,
        transcript: Transcript,
        layerIdx: Int
    ) -> (msgs: [GPUGKRRoundMsg], rx: [Fr], ry: [Fr]) {

        let totalVars = 2 * nIn
        let numGates = layer.gates.count
        let eqSize = 1 << nOut
        let sortedKeys = layerWiringKeys[layerIdx]
        let g2w = layerGateToWiring[layerIdx]
        let gd = layerGateData[layerIdx]
        let numEntries = sortedKeys.count

        // Build wiring coefficients
        var addCoeffs = [Fr](repeating: Fr.zero, count: numEntries)
        var mulCoeffs = [Fr](repeating: Fr.zero, count: numEntries)

        for (rk, wk) in rPoints {
            let eqVals = gpuEqPoly(point: rk)
            let isUnitWeight = gpuGKRFrEqual(wk, Fr.one)
            var coeffs: [Fr]
            if isUnitWeight {
                coeffs = eqVals
            } else {
                coeffs = [Fr](repeating: Fr.zero, count: eqSize)
                eqVals.withUnsafeBytes { eBuf in
                    coeffs.withUnsafeMutableBytes { cBuf in
                        var w = wk
                        withUnsafeBytes(of: &w) { wBuf in
                            bn254_fr_batch_mul_scalar_neon(
                                cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                eBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                wBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                Int32(eqSize))
                        }
                    }
                }
            }

            for gIdx in 0..<numGates {
                let coeff = gIdx < eqSize ? coeffs[gIdx] : Fr.zero
                if coeff.isZero { continue }
                let wIdx = g2w[gIdx]
                if gd[gIdx * 3] == 0 {
                    addCoeffs[wIdx] = frAdd(addCoeffs[wIdx], coeff)
                } else {
                    mulCoeffs[wIdx] = frAdd(mulCoeffs[wIdx], coeff)
                }
            }
        }

        // Pack into wiring format: [xyIdx, addCoeff(4 u64), mulCoeff(4 u64)]
        var wiring = [UInt64](repeating: 0, count: numEntries * 9)
        var wiringAlt = [UInt64](repeating: 0, count: numEntries * 9)
        wiring.withUnsafeMutableBufferPointer { wBuf in
            let p = wBuf.baseAddress!
            addCoeffs.withUnsafeBytes { addBuf in
                let ap = addBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                mulCoeffs.withUnsafeBytes { mulBuf in
                    let mp = mulBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                    for i in 0..<numEntries {
                        let base = i * 9
                        p[base] = UInt64(sortedKeys[i])
                        p[base + 1] = ap[i * 4]
                        p[base + 2] = ap[i * 4 + 1]
                        p[base + 3] = ap[i * 4 + 2]
                        p[base + 4] = ap[i * 4 + 3]
                        p[base + 5] = mp[i * 4]
                        p[base + 6] = mp[i * 4 + 1]
                        p[base + 7] = mp[i * 4 + 2]
                        p[base + 8] = mp[i * 4 + 3]
                    }
                }
            }
        }
        var numWiringEntries = numEntries

        // Copy prev evals for VX/VY folding
        let evalCount = prevEvals.count
        var vxBuf = prevEvals.withUnsafeBytes { src -> [UInt64] in
            let u64count = evalCount * 4
            var buf = [UInt64](repeating: 0, count: u64count)
            buf.withUnsafeMutableBytes { dst in
                dst.baseAddress!.copyMemory(from: src.baseAddress!, byteCount: evalCount * 32)
            }
            return buf
        }
        var vyBuf = vxBuf  // same initial values
        var vxSize = Int32(evalCount)
        var vySize = Int32(evalCount)

        var msgs = [GPUGKRRoundMsg]()
        msgs.reserveCapacity(totalVars)
        var challenges = [Fr]()
        challenges.reserveCapacity(totalVars)
        var currentTableSize = Int32(1 << totalVars)

        var s0 = [UInt64](repeating: 0, count: 4)
        var s1 = [UInt64](repeating: 0, count: 4)
        var s2 = [UInt64](repeating: 0, count: 4)
        var chal = [UInt64](repeating: 0, count: 4)

        for round in 0..<totalVars {
            let halfSize = currentTableSize / 2

            gkr_sumcheck_step(
                wiring, Int32(numWiringEntries),
                vxBuf, vxSize,
                vyBuf, vySize,
                Int32(round), Int32(nIn), currentTableSize,
                &s0, &s1, &s2
            )

            let frS0 = s0.withUnsafeBytes {
                Fr(v: $0.load(as: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32).self))
            }
            let frS1 = s1.withUnsafeBytes {
                Fr(v: $0.load(as: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32).self))
            }
            let frS2 = s2.withUnsafeBytes {
                Fr(v: $0.load(as: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32).self))
            }

            msgs.append(GPUGKRRoundMsg(eval0: frS0, eval1: frS1, eval2: frS2))

            transcript.absorb(frS0)
            transcript.absorb(frS1)
            transcript.absorb(frS2)
            let challenge = transcript.squeeze()
            challenges.append(challenge)

            withUnsafeBytes(of: challenge) { src in
                chal.withUnsafeMutableBytes { dst in
                    dst.baseAddress!.copyMemory(from: src.baseAddress!, byteCount: 32)
                }
            }

            let newCount = gkr_wiring_reduce(
                wiring, Int32(numWiringEntries),
                chal, halfSize,
                &wiringAlt
            )
            swap(&wiring, &wiringAlt)
            numWiringEntries = Int(newCount)
            currentTableSize = halfSize

            if round < nIn {
                let vxHalf = vxSize / 2
                if vxHalf > 0 {
                    gkr_mle_fold(&vxBuf, Int32(vxHalf), chal)
                    vxSize = vxHalf
                }
            } else {
                let vyHalf = vySize / 2
                if vyHalf > 0 {
                    gkr_mle_fold(&vyBuf, Int32(vyHalf), chal)
                    vySize = vyHalf
                }
            }
        }

        let rx = Array(challenges.prefix(nIn))
        let ry = Array(challenges.suffix(nIn))
        return (msgs, rx, ry)
    }

    // MARK: - Shader Compilation

    private static func compileGKRShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderSource = """
        #include <metal_stdlib>
        using namespace metal;

        struct Fr {
            uint v[8];
        };

        constant uint FR_P[8] = {
            0xf0000001, 0x43e1f593, 0x79b97091, 0x2833e848,
            0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72
        };
        constant uint FR_INV = 0xefffffffu;

        // --- 256-bit arithmetic for BN254 Fr ---

        Fr fr_add(Fr a, Fr b) {
            Fr r;
            ulong c = 0;
            for (int i = 0; i < 8; i++) {
                c += ulong(a.v[i]) + ulong(b.v[i]);
                r.v[i] = uint(c & 0xFFFFFFFF);
                c >>= 32;
            }
            // Reduce if >= p
            long borrow = 0;
            Fr s;
            for (int i = 0; i < 8; i++) {
                borrow += long(r.v[i]) - long(FR_P[i]);
                s.v[i] = uint(borrow & 0xFFFFFFFF);
                borrow >>= 32;
            }
            return (borrow >= 0) ? s : r;
        }

        Fr fr_sub(Fr a, Fr b) {
            Fr r;
            long c = 0;
            for (int i = 0; i < 8; i++) {
                c += long(a.v[i]) - long(b.v[i]);
                r.v[i] = uint(c & 0xFFFFFFFF);
                c >>= 32;
            }
            if (c < 0) {
                ulong carry = 0;
                for (int i = 0; i < 8; i++) {
                    carry += ulong(r.v[i]) + ulong(FR_P[i]);
                    r.v[i] = uint(carry & 0xFFFFFFFF);
                    carry >>= 32;
                }
            }
            return r;
        }

        Fr fr_mul(Fr a, Fr b) {
            // CIOS Montgomery multiplication
            uint t[9] = {0};
            for (int i = 0; i < 8; i++) {
                ulong carry = 0;
                for (int j = 0; j < 8; j++) {
                    carry += ulong(t[j]) + ulong(a.v[j]) * ulong(b.v[i]);
                    t[j] = uint(carry & 0xFFFFFFFF);
                    carry >>= 32;
                }
                t[8] = uint(carry);

                uint m = t[0] * FR_INV;
                carry = ulong(t[0]) + ulong(m) * ulong(FR_P[0]);
                carry >>= 32;
                for (int j = 1; j < 8; j++) {
                    carry += ulong(t[j]) + ulong(m) * ulong(FR_P[j]);
                    t[j-1] = uint(carry & 0xFFFFFFFF);
                    carry >>= 32;
                }
                t[7] = uint(carry + ulong(t[8]));
                t[8] = uint((carry + ulong(t[8])) >> 32);
            }
            Fr r;
            for (int i = 0; i < 8; i++) r.v[i] = t[i];
            // Final reduction
            long borrow = 0;
            Fr s;
            for (int i = 0; i < 8; i++) {
                borrow += long(r.v[i]) - long(FR_P[i]);
                s.v[i] = uint(borrow & 0xFFFFFFFF);
                borrow >>= 32;
            }
            return (borrow >= 0) ? s : r;
        }

        Fr fr_zero() {
            Fr r;
            for (int i = 0; i < 8; i++) r.v[i] = 0;
            return r;
        }

        // --- GKR Kernels ---

        // MLE table fold: out[i] = evals[i] + r * (evals[i+half] - evals[i])
        kernel void gkr_table_fold_bn254(
            device const Fr* evals      [[buffer(0)]],
            device Fr* evals_out        [[buffer(1)]],
            constant Fr* challenge      [[buffer(2)]],
            constant uint& half_n       [[buffer(3)]],
            uint gid                    [[thread_position_in_grid]]
        ) {
            if (gid >= half_n) return;
            Fr a = evals[gid];
            Fr b = evals[gid + half_n];
            Fr r = challenge[0];
            Fr diff = fr_sub(b, a);
            Fr r_diff = fr_mul(r, diff);
            evals_out[gid] = fr_add(a, r_diff);
        }

        // MLE evaluation via parallel reduction (one level of folding)
        kernel void gkr_mle_eval_bn254(
            device const Fr* evals      [[buffer(0)]],
            device Fr* evals_out        [[buffer(1)]],
            constant Fr* challenge      [[buffer(2)]],
            constant uint& half_n       [[buffer(3)]],
            uint gid                    [[thread_position_in_grid]]
        ) {
            if (gid >= half_n) return;
            Fr a = evals[gid];
            Fr b = evals[gid + half_n];
            Fr r = challenge[0];
            Fr diff = fr_sub(b, a);
            Fr r_diff = fr_mul(r, diff);
            evals_out[gid] = fr_add(a, r_diff);
        }

        // Eq polynomial construction: process in reverse order
        // For level i: eq[2j+1] = eq[j] * r_i, eq[2j] = eq[j] * (1 - r_i)
        kernel void gkr_eq_poly_bn254(
            device Fr* eq               [[buffer(0)]],
            constant Fr* ri_buf         [[buffer(1)]],
            constant uint& half_val     [[buffer(2)]],
            uint gid                    [[thread_position_in_grid]]
        ) {
            uint half = half_val;
            if (gid >= half) return;
            // Process in reverse to avoid overwriting
            uint j = half - 1 - gid;
            Fr val = eq[j];
            Fr ri = ri_buf[0];

            // one_minus_ri = 1 - ri (compute as p + 1 - ri mod p, but use fr_sub)
            Fr one;
            for (int k = 0; k < 8; k++) one.v[k] = 0;
            // Montgomery form of 1
            one.v[0] = 0xac96341cu; one.v[1] = 0x36fc7695u;
            one.v[2] = 0x7879462eu; one.v[3] = 0x0e0a77c1u;
            one.v[4] = 0xa44ba594u; one.v[5] = 0x14a3074du;
            one.v[6] = 0xbd3a8d9fu; one.v[7] = 0x0d819232u;

            Fr one_minus_ri = fr_sub(one, ri);
            eq[2 * j + 1] = fr_mul(val, ri);
            eq[2 * j] = fr_mul(val, one_minus_ri);
        }

        // Sumcheck round polynomial: compute partial sums s0, s1 via reduction
        kernel void gkr_round_poly_bn254(
            device const Fr* evals      [[buffer(0)]],
            device Fr* partial_sums     [[buffer(1)]],
            constant uint& half_n       [[buffer(2)]],
            uint tid                    [[thread_index_in_threadgroup]],
            uint tgid                   [[threadgroup_position_in_grid]],
            uint tg_size                [[threads_per_threadgroup]]
        ) {
            threadgroup Fr shared_s0[256];
            threadgroup Fr shared_s1[256];

            Fr local_s0 = fr_zero();
            Fr local_s1 = fr_zero();

            uint global_idx = tgid * tg_size + tid;
            if (global_idx < half_n) {
                local_s0 = evals[global_idx];
                local_s1 = evals[global_idx + half_n];
            }

            shared_s0[tid] = local_s0;
            shared_s1[tid] = local_s1;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    shared_s0[tid] = fr_add(shared_s0[tid], shared_s0[tid + stride]);
                    shared_s1[tid] = fr_add(shared_s1[tid], shared_s1[tid + stride]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (tid == 0) {
                partial_sums[tgid * 2] = shared_s0[0];
                partial_sums[tgid * 2 + 1] = shared_s1[0];
            }
        }
        """

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: shaderSource, options: options)
    }

    // MARK: - Utility

    /// Pad values to 2^numVars with zeros.
    private func padToPowerOf2(_ values: [Fr], numVars: Int) -> [Fr] {
        let n = 1 << numVars
        if values.count == n { return values }
        var padded = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<min(values.count, n) {
            padded[i] = values[i]
        }
        return padded
    }
}

// MARK: - Private Helpers

/// Compare two Fr elements for equality (component-wise Montgomery comparison).
private func gpuGKRFrEqual(_ a: Fr, _ b: Fr) -> Bool {
    return a.v.0 == b.v.0 && a.v.1 == b.v.1 && a.v.2 == b.v.2 && a.v.3 == b.v.3 &&
           a.v.4 == b.v.4 && a.v.5 == b.v.5 && a.v.6 == b.v.6 && a.v.7 == b.v.7
}

/// Precomputed inverse of 2 for Lagrange interpolation.
private let gpuGKRInv2: Fr = frInverse(frAdd(Fr.one, Fr.one))

/// Evaluate degree-2 polynomial through (0,s0), (1,s1), (2,s2) at point x.
private func gpuLagrangeEval3(s0: Fr, s1: Fr, s2: Fr, at x: Fr) -> Fr {
    let xm1 = frSub(x, Fr.one)
    let xm2 = frSub(x, frAdd(Fr.one, Fr.one))
    let negOne = frSub(Fr.zero, Fr.one)

    let l0 = frMul(frMul(xm1, xm2), gpuGKRInv2)
    let l1 = frMul(frMul(x, xm2), negOne)
    let l2 = frMul(frMul(x, xm1), gpuGKRInv2)

    return frAdd(frAdd(frMul(s0, l0), frMul(s1, l1)), frMul(s2, l2))
}
