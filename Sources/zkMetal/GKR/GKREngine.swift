// GKR (Goldwasser-Kalai-Rothblum) Interactive Proof Engine
// Efficiently proves layered arithmetic circuit evaluations using sumcheck at each layer.
// No FFT/NTT needed — purely algebraic protocol based on multilinear extensions.
//
// Protocol outline (Fiat-Shamir non-interactive):
// 1. Prover evaluates circuit, commits output
// 2. For each layer from output to input:
//    a. Run sumcheck on the GKR equation to reduce the claim
//    b. Prover sends evaluations of the previous layer's MLE at the sumcheck output points
//    c. Verifier combines claims using random linear combination (batched)
// 3. At the input layer, verifier checks the MLE evaluation directly
//
// Key insight: after each layer's sumcheck, we get two claims V(rx) and V(ry).
// For multi-variable layers, we CANNOT merge them into a single point because V is
// multilinear (not linear). Instead we batch: the next layer proves
//   claim = alpha * V(rx) + beta * V(ry)
// by building a combined sumcheck table using both output points.

import Foundation
import NeonFieldOps

// MARK: - GKR Proof Types

/// A single round of the sumcheck sub-protocol within GKR.
/// Stores the polynomial evaluations at 0, 1, 2 for each sumcheck variable.
public struct SumcheckRoundMsg {
    public let s0: Fr  // polynomial evaluated at 0
    public let s1: Fr  // polynomial evaluated at 1
    public let s2: Fr  // polynomial evaluated at 2

    public init(s0: Fr, s1: Fr, s2: Fr) {
        self.s0 = s0
        self.s1 = s1
        self.s2 = s2
    }
}

/// Proof for one layer of GKR: sumcheck messages + claimed evaluations at the random points.
public struct GKRLayerProof {
    public let sumcheckMsgs: [SumcheckRoundMsg]
    public let claimedVx: Fr   // V_{i-1}(rx)
    public let claimedVy: Fr   // V_{i-1}(ry)
}

/// Complete GKR proof for an entire circuit.
public struct GKRProof {
    public let layerProofs: [GKRLayerProof]
}

// MARK: - GKR Engine

public class GKREngine {
    public static let version = Versions.gkr
    public let circuit: LayeredCircuit

    // Pre-computed gate data in C format: [type(0=add/1=mul), left, right] per gate
    private let layerGateData: [[Int32]]
    // Pre-computed sorted unique wiring indices per layer
    private let layerWiringKeys: [[Int]]
    // Pre-computed: for each wiring entry, list of (gateIndex, isAdd) pairs
    private let layerWiringGates: [[(gateIdx: Int, isAdd: Bool)]]
    // Flat gate-to-wiring-entry mapping
    private let layerGateToWiring: [[Int]]

    // Cached buffers for reuse across prove calls
    private var _cachedWiring: [UInt64] = []
    private var _cachedWiringAlt: [UInt64] = []
    private var _cachedVx: [UInt64] = []
    private var _cachedVy: [UInt64] = []

    public init(circuit: LayeredCircuit) {
        self.circuit = circuit

        var gateData = [[Int32]]()
        var wiringKeys = [[Int]]()
        var wiringGates = [[(gateIdx: Int, isAdd: Bool)]]()
        var gateToWiring = [[Int]]()

        for layerIdx in 0..<circuit.layers.count {
            let layer = circuit.layers[layerIdx]
            let nIn = circuit.inputVars(layer: layerIdx)
            let inSize = 1 << nIn

            // Gate data in C format
            var gd = [Int32](repeating: 0, count: layer.gates.count * 3)
            for (i, gate) in layer.gates.enumerated() {
                gd[i * 3] = gate.type == .add ? 0 : 1
                gd[i * 3 + 1] = Int32(gate.leftInput)
                gd[i * 3 + 2] = Int32(gate.rightInput)
            }
            gateData.append(gd)

            // Build wiring topology
            var dict = [Int: Int]()  // xyIdx -> wiring entry index
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

            // Sort keys and build remapping
            let sortedIndices = keys.indices.sorted { keys[$0] < keys[$1] }
            var remap = [Int](repeating: 0, count: keys.count)
            var sortedKeys = [Int](repeating: 0, count: keys.count)
            for (newIdx, oldIdx) in sortedIndices.enumerated() {
                remap[oldIdx] = newIdx
                sortedKeys[newIdx] = keys[oldIdx]
            }

            // Remap gate-to-wiring
            for i in 0..<g2w.count {
                g2w[i] = remap[g2w[i]]
            }

            wiringKeys.append(sortedKeys)
            gateToWiring.append(g2w)

            // Build per-entry gate lists (not used in hot path but useful for debugging)
            wiringGates.append([])
        }

        self.layerGateData = gateData
        self.layerWiringKeys = wiringKeys
        self.layerWiringGates = wiringGates
        self.layerGateToWiring = gateToWiring
    }

    // MARK: - Prover

    /// Prove that the circuit, on the given inputs, produces the claimed output.
    ///
    /// Uses batched GKR: after each layer, we get claims at two points (rx, ry).
    /// The next layer's sumcheck proves alpha*V(rx) + beta*V(ry) using both output points.
    public func prove(inputs: [Fr], transcript: Transcript) -> GKRProof {
        let allValues = circuit.evaluate(inputs: inputs)
        let d = circuit.depth

        // Absorb output
        let outputValues = allValues[d]
        for v in outputValues { transcript.absorb(v) }
        transcript.absorbLabel("gkr-init")

        let outputVars = circuit.outputVars(layer: d - 1)
        let r0 = transcript.squeezeN(outputVars)

        let outputEvals = MultilinearPoly(numVars: outputVars, values: outputValues).evals
        _ = cMleEval(evals: outputEvals, point: r0, numVars: outputVars)

        // For the first layer, we have a single claim at one point with weight 1.
        var rPoints: [([Fr], Fr)] = [(r0, Fr.one)]

        var layerProofs = [GKRLayerProof]()

        for layerIdx in stride(from: d - 1, through: 0, by: -1) {
            let nOut = circuit.outputVars(layer: layerIdx)
            let nIn = circuit.inputVars(layer: layerIdx)
            let prevValues = allValues[layerIdx]
            let prevMLE = MultilinearPoly(numVars: nIn, values: prevValues)

            let (msgs, rx, ry) = proverBatchedSumcheck(
                rPoints: rPoints,
                layer: circuit.layers[layerIdx],
                prevMLE: prevMLE, nOut: nOut, nIn: nIn,
                transcript: transcript,
                layerIdx: layerIdx
            )

            let vx = cMleEval(evals: prevMLE.evals, point: rx, numVars: nIn)
            let vy = cMleEval(evals: prevMLE.evals, point: ry, numVars: nIn)

            transcript.absorb(vx)
            transcript.absorb(vy)
            transcript.absorbLabel("gkr-layer-\(layerIdx)")

            layerProofs.append(GKRLayerProof(sumcheckMsgs: msgs, claimedVx: vx, claimedVy: vy))

            // Get batching coefficients for next layer
            let alpha = transcript.squeeze()
            let beta = transcript.squeeze()

            rPoints = [(rx, alpha), (ry, beta)]
        }

        return GKRProof(layerProofs: layerProofs)
    }

    /// Batched sumcheck for one GKR layer using sparse wiring predicates.
    /// C-accelerated: eq polynomial, sumcheck rounds, wiring reduction, and MLE folding
    /// all use CIOS Montgomery C code. Pre-computed wiring topology eliminates Dictionary
    /// construction and sorting. Cached buffers eliminate per-call allocations.
    private func proverBatchedSumcheck(
        rPoints: [([Fr], Fr)],  // [(output_point, weight)]
        layer: CircuitLayer,
        prevMLE: MultilinearPoly,
        nOut: Int, nIn: Int,
        transcript: Transcript,
        layerIdx: Int
    ) -> (msgs: [SumcheckRoundMsg], rx: [Fr], ry: [Fr]) {

        let totalVars = 2 * nIn
        let numGates = layer.gates.count
        let eqSize = 1 << nOut
        let sortedKeys = layerWiringKeys[layerIdx]
        let g2w = layerGateToWiring[layerIdx]
        let gd = layerGateData[layerIdx]
        let numEntries = sortedKeys.count

        // Build wiring coefficients using pre-computed topology
        // addCoeffs[i] and mulCoeffs[i] correspond to sortedKeys[i]
        var addCoeffs = [Fr](repeating: Fr.zero, count: numEntries)
        var mulCoeffs = [Fr](repeating: Fr.zero, count: numEntries)

        for (rk, wk) in rPoints {
            var eqVals = [Fr](repeating: Fr.zero, count: eqSize)
            rk.withUnsafeBytes { rkBuf in
                eqVals.withUnsafeMutableBytes { eqBuf in
                    gkr_eq_poly(
                        rkBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(nOut),
                        eqBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                    )
                }
            }
            // When weight = 1, skip batch multiply (identity in Montgomery form)
            let isUnitWeight = gkrFrEqual(wk, Fr.one)
            var coeffs: [Fr]
            if isUnitWeight {
                coeffs = eqVals
            } else {
                coeffs = [Fr](repeating: Fr.zero, count: eqSize)
                eqVals.withUnsafeBytes { eqBuf in
                    withUnsafeBytes(of: wk) { wkBuf in
                        coeffs.withUnsafeMutableBytes { outBuf in
                            bn254_fr_batch_mul_scalar_neon(
                                outBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                eqBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                wkBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                Int32(eqSize)
                            )
                        }
                    }
                }
            }

            // Accumulate using pre-computed gate-to-wiring mapping (no Dictionary)
            for gIdx in 0..<numGates {
                let coeff = gIdx < eqSize ? coeffs[gIdx] : Fr.zero
                if coeff.isZero { continue }
                let wIdx = g2w[gIdx]
                if gd[gIdx * 3] == 0 { // add
                    addCoeffs[wIdx] = frAdd(addCoeffs[wIdx], coeff)
                } else { // mul
                    mulCoeffs[wIdx] = frAdd(mulCoeffs[wIdx], coeff)
                }
            }
        }

        // Pack into C wiring format using cached buffers (already sorted)
        let wiringSize = numEntries * 9
        if _cachedWiring.count < wiringSize {
            _cachedWiring = [UInt64](repeating: 0, count: max(wiringSize, 1024 * 9))
            _cachedWiringAlt = [UInt64](repeating: 0, count: max(wiringSize, 1024 * 9))
        }

        _cachedWiring.withUnsafeMutableBufferPointer { wBuf in
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

        // Set up MLE evaluations using cached buffers
        let evalCount = prevMLE.evals.count
        let evalU64Count = evalCount * 4
        if _cachedVx.count < evalU64Count {
            _cachedVx = [UInt64](repeating: 0, count: evalU64Count)
            _cachedVy = [UInt64](repeating: 0, count: evalU64Count)
        }
        prevMLE.evals.withUnsafeBytes { src in
            _cachedVx.withUnsafeMutableBytes { dst in
                dst.baseAddress!.copyMemory(from: src.baseAddress!, byteCount: evalCount * 32)
            }
            _cachedVy.withUnsafeMutableBytes { dst in
                dst.baseAddress!.copyMemory(from: src.baseAddress!, byteCount: evalCount * 32)
            }
        }
        var vxSize = Int32(evalCount)
        var vySize = Int32(evalCount)

        var msgs = [SumcheckRoundMsg]()
        msgs.reserveCapacity(totalVars)
        var challenges = [Fr]()
        challenges.reserveCapacity(totalVars)

        var currentTableSize = Int32(1 << totalVars)

        // Pre-allocate round buffers outside the loop
        var s0 = [UInt64](repeating: 0, count: 4)
        var s1 = [UInt64](repeating: 0, count: 4)
        var s2 = [UInt64](repeating: 0, count: 4)
        var chal = [UInt64](repeating: 0, count: 4)

        for round in 0..<totalVars {
            let halfSize = currentTableSize / 2

            gkr_sumcheck_step(
                _cachedWiring, Int32(numWiringEntries),
                _cachedVx, vxSize,
                _cachedVy, vySize,
                Int32(round), Int32(nIn), currentTableSize,
                &s0, &s1, &s2
            )

            let frS0 = s0.withUnsafeBytes { Fr(v: $0.load(as: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32).self)) }
            let frS1 = s1.withUnsafeBytes { Fr(v: $0.load(as: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32).self)) }
            let frS2 = s2.withUnsafeBytes { Fr(v: $0.load(as: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32).self)) }

            msgs.append(SumcheckRoundMsg(s0: frS0, s1: frS1, s2: frS2))

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
                _cachedWiring, Int32(numWiringEntries),
                chal, halfSize,
                &_cachedWiringAlt
            )
            swap(&_cachedWiring, &_cachedWiringAlt)
            numWiringEntries = Int(newCount)
            currentTableSize = halfSize

            if round < nIn {
                let vxHalf = vxSize / 2
                if vxHalf > 0 {
                    gkr_mle_fold(&_cachedVx, Int32(vxHalf), chal)
                    vxSize = vxHalf
                }
            } else {
                let vyHalf = vySize / 2
                if vyHalf > 0 {
                    gkr_mle_fold(&_cachedVy, Int32(vyHalf), chal)
                    vySize = vyHalf
                }
            }
        }

        let rx = Array(challenges.prefix(nIn))
        let ry = Array(challenges.suffix(nIn))
        return (msgs, rx, ry)
    }

    // MARK: - Verifier

    /// Verify a GKR proof.
    public func verify(inputs: [Fr], output: [Fr], proof: GKRProof, transcript: Transcript) -> Bool {
        let d = circuit.depth
        guard proof.layerProofs.count == d else { return false }

        for v in output { transcript.absorb(v) }
        transcript.absorbLabel("gkr-init")

        let outputVars = circuit.outputVars(layer: d - 1)
        let r0 = transcript.squeezeN(outputVars)

        var claim = cMleEval(evals: output.count == (1 << outputVars) ? output :
            MultilinearPoly(numVars: outputVars, values: output).evals, point: r0, numVars: outputVars)

        var rPoints: [([Fr], Fr)] = [(r0, Fr.one)]

        for layerIdx in stride(from: d - 1, through: 0, by: -1) {
            let nIn = circuit.inputVars(layer: layerIdx)
            let layerProof = proof.layerProofs[d - 1 - layerIdx]

            let totalVars = 2 * nIn
            guard layerProof.sumcheckMsgs.count == totalVars else { return false }

            var currentClaim = claim
            var challenges = [Fr]()
            challenges.reserveCapacity(totalVars)

            for roundIdx in 0..<totalVars {
                let msg = layerProof.sumcheckMsgs[roundIdx]

                let sum = frAdd(msg.s0, msg.s1)
                if !gkrFrEqual(sum, currentClaim) {
                    return false
                }

                transcript.absorb(msg.s0)
                transcript.absorb(msg.s1)
                transcript.absorb(msg.s2)
                let challenge = transcript.squeeze()
                challenges.append(challenge)

                currentClaim = lagrangeEval3(s0: msg.s0, s1: msg.s1, s2: msg.s2, at: challenge)
            }

            let rx = Array(challenges.prefix(nIn))
            let ry = Array(challenges.suffix(nIn))
            let vx = layerProof.claimedVx
            let vy = layerProof.claimedVy

            // Verify: expected = sum_k w_k * [add(r_k, rx, ry)*(vx+vy) + mul(r_k, rx, ry)*vx*vy]
            // C-accelerated eq polynomial computation
            let eqRx = cEqPoly(point: rx)
            let eqRy = cEqPoly(point: ry)
            let sumVxVy = frAdd(vx, vy)
            let prodVxVy = frMul(vx, vy)

            var expected = Fr.zero
            for (rk, wk) in rPoints {
                let eqRk = cEqPoly(point: rk)
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

            if !gkrFrEqual(currentClaim, expected) {
                return false
            }

            transcript.absorb(vx)
            transcript.absorb(vy)
            transcript.absorbLabel("gkr-layer-\(layerIdx)")

            let alpha = transcript.squeeze()
            let beta = transcript.squeeze()

            claim = frAdd(frMul(alpha, vx), frMul(beta, vy))
            rPoints = [(rx, alpha), (ry, beta)]
        }

        // Final check: claim = sum_k w_k * V_input(r_k)
        let inputNumVars = rPoints[0].0.count
        let inputEvals = MultilinearPoly(numVars: inputNumVars, values: inputs).evals
        var inputExpected = Fr.zero
        for (rk, wk) in rPoints {
            let val = cMleEval(evals: inputEvals, point: rk, numVars: inputNumVars)
            inputExpected = frAdd(inputExpected, frMul(wk, val))
        }
        return gkrFrEqual(claim, inputExpected)
    }
}

// MARK: - Helpers

/// Compare two Fr elements for equality (direct Montgomery comparison, no reduction).
private func gkrFrEqual(_ a: Fr, _ b: Fr) -> Bool {
    return a.v.0 == b.v.0 && a.v.1 == b.v.1 && a.v.2 == b.v.2 && a.v.3 == b.v.3 &&
           a.v.4 == b.v.4 && a.v.5 == b.v.5 && a.v.6 == b.v.6 && a.v.7 == b.v.7
}

/// Precomputed inverse of 2 in Montgomery form (avoids frInverse per call).
private let gkrInv2: Fr = frInverse(frAdd(Fr.one, Fr.one))

/// Evaluate the degree-2 polynomial passing through (0, s0), (1, s1), (2, s2) at point x.
private func lagrangeEval3(s0: Fr, s1: Fr, s2: Fr, at x: Fr) -> Fr {
    let xm1 = frSub(x, Fr.one)
    let xm2 = frSub(x, frAdd(Fr.one, Fr.one))
    let negOne = frSub(Fr.zero, Fr.one)

    let l0 = frMul(frMul(xm1, xm2), gkrInv2)
    let l1 = frMul(frMul(x, xm2), negOne)
    let l2 = frMul(frMul(x, xm1), gkrInv2)

    return frAdd(frAdd(frMul(s0, l0), frMul(s1, l1)), frMul(s2, l2))
}

/// C-accelerated MLE evaluation.
private func cMleEval(evals: [Fr], point: [Fr], numVars: Int) -> Fr {
    var result = Fr.zero
    evals.withUnsafeBytes { evalBuf in
        point.withUnsafeBytes { ptBuf in
            withUnsafeMutableBytes(of: &result) { resBuf in
                bn254_fr_mle_eval(
                    evalBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(numVars),
                    ptBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    resBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                )
            }
        }
    }
    return result
}

/// C-accelerated eq polynomial computation.
private func cEqPoly(point: [Fr]) -> [Fr] {
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
