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

// MARK: - Sparse wiring entry (contiguous struct for cache-friendly iteration)

private struct WiringEntry {
    var idx: Int
    var addCoeff: Fr
    var mulCoeff: Fr
}

// MARK: - GKR Engine

public class GKREngine {
    public static let version = Versions.gkr
    public let circuit: LayeredCircuit

    public init(circuit: LayeredCircuit) {
        self.circuit = circuit
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

        let outputMLE = MultilinearPoly(numVars: outputVars, values: outputValues)
        _ = outputMLE.evaluate(at: r0)  // claim absorbed via transcript consistency

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
                transcript: transcript
            )

            let vx = prevMLE.evaluate(at: rx)
            let vy = prevMLE.evaluate(at: ry)

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
    ///
    /// Given multiple (output-point, weight) pairs, the function being summed is:
    ///   g(a, b) = sum_k w_k * [add_k(a,b) * (V(a)+V(b)) + mul_k(a,b) * V(a)*V(b)]
    ///
    /// Key optimization: wiring predicates are extremely sparse (numGates nonzero entries
    /// out of 2^(2*nIn) total). We track only nonzero entries in a sorted contiguous array,
    /// reducing each sumcheck round from O(2^(2*nIn)) to O(numGates) field operations.
    /// Uses sorted-array merge instead of Dictionary for cache-friendly access and zero
    /// hashing overhead. Pairing, evaluation, and reduction are fused into a single pass.
    private func proverBatchedSumcheck(
        rPoints: [([Fr], Fr)],  // [(output_point, weight)]
        layer: CircuitLayer,
        prevMLE: MultilinearPoly,
        nOut: Int, nIn: Int,
        transcript: Transcript
    ) -> (msgs: [SumcheckRoundMsg], rx: [Fr], ry: [Fr]) {

        let totalVars = 2 * nIn
        let inSize = 1 << nIn

        // Build sparse wiring from gate structure -- O(numGates) not O(2^(nOut+2*nIn)).
        // Use dictionary only once for initial construction, then convert to sorted array.
        var sparseWiringDict = [Int: (Fr, Fr)]()
        sparseWiringDict.reserveCapacity(layer.gates.count)

        for (rk, wk) in rPoints {
            let eqVals = MultilinearPoly.eqPoly(point: rk)

            for (gIdx, gate) in layer.gates.enumerated() {
                let eqVal = gIdx < eqVals.count ? eqVals[gIdx] : Fr.zero
                if eqVal.isZero { continue }
                let coeff = frMul(wk, eqVal)
                let xyIdx = gate.leftInput * inSize + gate.rightInput

                var entry = sparseWiringDict[xyIdx] ?? (Fr.zero, Fr.zero)
                switch gate.type {
                case .add: entry.0 = frAdd(entry.0, coeff)
                case .mul: entry.1 = frAdd(entry.1, coeff)
                }
                sparseWiringDict[xyIdx] = entry
            }
        }

        // Convert to sorted array for cache-friendly merge-based operations
        var wiring = [WiringEntry]()
        wiring.reserveCapacity(sparseWiringDict.count)
        for (idx, coeffs) in sparseWiringDict {
            if !coeffs.0.isZero || !coeffs.1.isZero {
                wiring.append(WiringEntry(idx: idx, addCoeff: coeffs.0, mulCoeff: coeffs.1))
            }
        }
        wiring.sort { $0.idx < $1.idx }

        // In-place reduction: we overwrite the first half each round, tracking effective size.
        var curVx = prevMLE.evals
        var curVy = prevMLE.evals
        var vxSize = curVx.count
        var vySize = curVy.count

        var msgs = [SumcheckRoundMsg]()
        msgs.reserveCapacity(totalVars)
        var challenges = [Fr]()
        challenges.reserveCapacity(totalVars)

        var currentTableSize = 1 << totalVars
        var newWiring = [WiringEntry]()
        newWiring.reserveCapacity(wiring.count)

        for round in 0..<totalVars {
            let halfSize = currentTableSize / 2
            let isXPhase = round < nIn

            // Fused pair + eval: single pass over sorted wiring array using two-pointer merge.
            let splitPos = gkrLowerBound(wiring, halfSize)

            var s0 = Fr.zero
            var s1 = Fr.zero
            var s2 = Fr.zero

            var li = 0
            var hi = splitPos

            if isXPhase {
                // X-phase: vy is fixed per entry, vx varies with low/high.
                // Optimization: g = a*(vx+vy) + m*(vx*vy) = (a + m*vy)*vx + a*vy
                // Precompute combined = a + m*vy and avy = a*vy per coeff level,
                // then g = combined*vx + avy (1 frMul instead of 3).
                let vxHalf = vxSize / 2

                curVx.withUnsafeBufferPointer { vxBuf in
                    curVy.withUnsafeBufferPointer { vyBuf in
                        let yMask = vySize - 1
                        while li < splitPos || hi < wiring.count {
                            let lowIdx = li < splitPos ? wiring[li].idx : Int.max
                            let highIdx = hi < wiring.count ? wiring[hi].idx &- halfSize : Int.max

                            var a0 = Fr.zero, a1 = Fr.zero, m0 = Fr.zero, m1 = Fr.zero
                            let mergedIdx: Int

                            if lowIdx <= highIdx {
                                mergedIdx = lowIdx
                                a0 = wiring[li].addCoeff
                                m0 = wiring[li].mulCoeff
                                li += 1
                                if lowIdx == highIdx {
                                    a1 = wiring[hi].addCoeff
                                    m1 = wiring[hi].mulCoeff
                                    hi += 1
                                }
                            } else {
                                mergedIdx = highIdx
                                a1 = wiring[hi].addCoeff
                                m1 = wiring[hi].mulCoeff
                                hi += 1
                            }

                            let yIdx = mergedIdx & yMask
                            let xIdx = mergedIdx >> nIn
                            let vx0 = xIdx < vxHalf ? vxBuf[xIdx] : Fr.zero
                            let vx1 = (xIdx &+ vxHalf) < vxSize ? vxBuf[xIdx &+ vxHalf] : Fr.zero
                            let vyVal = yIdx < vySize ? vyBuf[yIdx] : Fr.zero

                            // Precompute: combined_i = a_i + m_i*vy, avy_i = a_i*vy
                            let m0vy = frMul(m0, vyVal)
                            let m1vy = frMul(m1, vyVal)
                            let c0 = frAdd(a0, m0vy)      // a0 + m0*vy
                            let c1 = frAdd(a1, m1vy)      // a1 + m1*vy
                            let a0vy = frMul(a0, vyVal)
                            let a1vy = frMul(a1, vyVal)

                            // g0 = c0*vx0 + a0vy  (7 frMul total instead of 9)
                            s0 = frAdd(s0, frAdd(frMul(c0, vx0), a0vy))
                            // g1 = c1*vx1 + a1vy
                            s1 = frAdd(s1, frAdd(frMul(c1, vx1), a1vy))
                            // g2 via extrapolation: c2=2c1-c0, a2vy=2*a1vy-a0vy, vx2=2vx1-vx0
                            let c2 = frSub(frAdd(c1, c1), c0)
                            let a2vy = frSub(frAdd(a1vy, a1vy), a0vy)
                            let vx2 = frSub(frAdd(vx1, vx1), vx0)
                            s2 = frAdd(s2, frAdd(frMul(c2, vx2), a2vy))
                        }
                    }
                }
            } else {
                // Y-phase: vx is a single scalar, vy varies.
                // Optimization: g = a*(vx+vy) + m*(vx*vy) = (a + m*vx)*vy + a*vx
                // Precompute combined = a + m*vx and avx = a*vx, then g = combined*vy + avx
                let vxScalar = vxSize > 0 ? curVx[0] : Fr.zero
                let vyHalf = vySize / 2

                curVy.withUnsafeBufferPointer { vyBuf in
                    while li < splitPos || hi < wiring.count {
                        let lowIdx = li < splitPos ? wiring[li].idx : Int.max
                        let highIdx = hi < wiring.count ? wiring[hi].idx &- halfSize : Int.max

                        var a0 = Fr.zero, a1 = Fr.zero, m0 = Fr.zero, m1 = Fr.zero
                        let mergedIdx: Int

                        if lowIdx <= highIdx {
                            mergedIdx = lowIdx
                            a0 = wiring[li].addCoeff
                            m0 = wiring[li].mulCoeff
                            li += 1
                            if lowIdx == highIdx {
                                a1 = wiring[hi].addCoeff
                                m1 = wiring[hi].mulCoeff
                                hi += 1
                            }
                        } else {
                            mergedIdx = highIdx
                            a1 = wiring[hi].addCoeff
                            m1 = wiring[hi].mulCoeff
                            hi += 1
                        }

                        let vy0 = mergedIdx < vyHalf ? vyBuf[mergedIdx] : Fr.zero
                        let vy1 = (mergedIdx &+ vyHalf) < vySize ? vyBuf[mergedIdx &+ vyHalf] : Fr.zero

                        // Precompute: combined_i = a_i + m_i*vx, avx_i = a_i*vx
                        let m0vx = frMul(m0, vxScalar)
                        let m1vx = frMul(m1, vxScalar)
                        let c0 = frAdd(a0, m0vx)
                        let c1 = frAdd(a1, m1vx)
                        let a0vx = frMul(a0, vxScalar)
                        let a1vx = frMul(a1, vxScalar)

                        // g0 = c0*vy0 + a0vx
                        s0 = frAdd(s0, frAdd(frMul(c0, vy0), a0vx))
                        // g1 = c1*vy1 + a1vx
                        s1 = frAdd(s1, frAdd(frMul(c1, vy1), a1vx))
                        // g2: c2=2c1-c0, a2vx=2*a1vx-a0vx, vy2=2vy1-vy0
                        let c2 = frSub(frAdd(c1, c1), c0)
                        let a2vx = frSub(frAdd(a1vx, a1vx), a0vx)
                        let vy2 = frSub(frAdd(vy1, vy1), vy0)
                        s2 = frAdd(s2, frAdd(frMul(c2, vy2), a2vx))
                    }
                }
            }

            let msg = SumcheckRoundMsg(s0: s0, s1: s1, s2: s2)
            msgs.append(msg)

            transcript.absorb(s0)
            transcript.absorb(s1)
            transcript.absorb(s2)
            let challenge = transcript.squeeze()
            challenges.append(challenge)

            // Reduce sparse wiring: merge low/high with challenge into newWiring (sorted)
            let oneMinusC = frSub(Fr.one, challenge)

            li = 0
            hi = splitPos
            newWiring.removeAll(keepingCapacity: true)

            while li < splitPos || hi < wiring.count {
                let lowIdx = li < splitPos ? wiring[li].idx : Int.max
                let highIdx = hi < wiring.count ? wiring[hi].idx &- halfSize : Int.max

                if lowIdx < highIdx {
                    newWiring.append(WiringEntry(
                        idx: lowIdx,
                        addCoeff: frMul(oneMinusC, wiring[li].addCoeff),
                        mulCoeff: frMul(oneMinusC, wiring[li].mulCoeff)))
                    li += 1
                } else if highIdx < lowIdx {
                    newWiring.append(WiringEntry(
                        idx: highIdx,
                        addCoeff: frMul(challenge, wiring[hi].addCoeff),
                        mulCoeff: frMul(challenge, wiring[hi].mulCoeff)))
                    hi += 1
                } else {
                    newWiring.append(WiringEntry(
                        idx: lowIdx,
                        addCoeff: frAdd(frMul(oneMinusC, wiring[li].addCoeff),
                                        frMul(challenge, wiring[hi].addCoeff)),
                        mulCoeff: frAdd(frMul(oneMinusC, wiring[li].mulCoeff),
                                        frMul(challenge, wiring[hi].mulCoeff))))
                    li += 1
                    hi += 1
                }
            }
            swap(&wiring, &newWiring)
            currentTableSize = halfSize

            // In-place reduction: overwrite first half, update size tracker
            if isXPhase {
                let vxHalf = vxSize / 2
                if vxHalf > 0 {
                    for j in 0..<vxHalf {
                        curVx[j] = frAdd(frMul(oneMinusC, curVx[j]), frMul(challenge, curVx[j + vxHalf]))
                    }
                    vxSize = vxHalf
                }
            } else {
                let vyHalf = vySize / 2
                if vyHalf > 0 {
                    for j in 0..<vyHalf {
                        curVy[j] = frAdd(frMul(oneMinusC, curVy[j]), frMul(challenge, curVy[j + vyHalf]))
                    }
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

        let outputMLE = MultilinearPoly(numVars: outputVars, values: output)
        var claim = outputMLE.evaluate(at: r0)

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
            // Sparse evaluation: for each gate g at gIdx with inputs (left, right),
            // wiring(rk, rx, ry) = eq(rk, gIdx) * eq(rx, left) * eq(ry, right)
            let eqRx = MultilinearPoly.eqPoly(point: rx)
            let eqRy = MultilinearPoly.eqPoly(point: ry)
            let sumVxVy = frAdd(vx, vy)
            let prodVxVy = frMul(vx, vy)

            var expected = Fr.zero
            for (rk, wk) in rPoints {
                let eqRk = MultilinearPoly.eqPoly(point: rk)
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
        let inputMLE = MultilinearPoly(numVars: rPoints[0].0.count, values: inputs)
        var inputExpected = Fr.zero
        for (rk, wk) in rPoints {
            inputExpected = frAdd(inputExpected, frMul(wk, inputMLE.evaluate(at: rk)))
        }
        return gkrFrEqual(claim, inputExpected)
    }
}

// MARK: - Helpers

/// Binary search: find first index in sorted wiring where idx >= target.
@inline(__always)
private func gkrLowerBound(_ wiring: [WiringEntry], _ target: Int) -> Int {
    var lo = 0, hi = wiring.count
    while lo < hi {
        let mid = (lo + hi) / 2
        if wiring[mid].idx < target { lo = mid + 1 } else { hi = mid }
    }
    return lo
}

/// Compare two Fr elements for equality.
private func gkrFrEqual(_ a: Fr, _ b: Fr) -> Bool {
    let diff = frSub(a, b)
    let limbs = frToInt(diff)
    return limbs[0] == 0 && limbs[1] == 0 && limbs[2] == 0 && limbs[3] == 0
}

/// Evaluate the degree-2 polynomial passing through (0, s0), (1, s1), (2, s2) at point x.
private func lagrangeEval3(s0: Fr, s1: Fr, s2: Fr, at x: Fr) -> Fr {
    let one = Fr.one
    let two = frAdd(one, one)
    let inv2 = frInverse(two)

    let xm1 = frSub(x, one)
    let xm2 = frSub(x, two)
    let negOne = frSub(Fr.zero, one)

    let l0 = frMul(frMul(xm1, xm2), inv2)
    let l1 = frMul(frMul(x, xm2), negOne)
    let l2 = frMul(frMul(x, xm1), inv2)

    return frAdd(frAdd(frMul(s0, l0), frMul(s1, l1)), frMul(s2, l2))
}
