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
    /// out of 2^(2*nIn) total). We track only nonzero (index, value) pairs, reducing
    /// each sumcheck round from O(2^(2*nIn)) to O(numGates) field operations.
    private func proverBatchedSumcheck(
        rPoints: [([Fr], Fr)],  // [(output_point, weight)]
        layer: CircuitLayer,
        prevMLE: MultilinearPoly,
        nOut: Int, nIn: Int,
        transcript: Transcript
    ) -> (msgs: [SumcheckRoundMsg], rx: [Fr], ry: [Fr]) {

        let totalVars = 2 * nIn
        let inSize = 1 << nIn

        // Build sparse wiring directly from gate structure — O(numGates) not O(2^(nOut+2*nIn)).
        // For each gate g at index gIdx: the wiring entry at (leftInput, rightInput) in the
        // 2^(2*nIn) table gets coefficient sum_k w_k * eq(rk, gIdx).
        var sparseWiringDict = [Int: (Fr, Fr)]()  // xyIdx -> (addCoeff, mulCoeff)

        for (rk, wk) in rPoints {
            // Compute eq polynomial: eq(rk, gIdx) for each gIdx
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

        var sparseWiring = sparseWiringDict.compactMap { (idx, coeffs) -> (Int, Fr, Fr)? in
            if coeffs.0.isZero && coeffs.1.isZero { return nil }
            return (idx, coeffs.0, coeffs.1)
        }

        var curVx = prevMLE.evals
        var curVy = prevMLE.evals

        var msgs = [SumcheckRoundMsg]()
        msgs.reserveCapacity(totalVars)
        var challenges = [Fr]()
        challenges.reserveCapacity(totalVars)

        var currentTableSize = 1 << totalVars

        for round in 0..<totalVars {
            let halfSize = currentTableSize / 2
            let isXPhase = round < nIn

            // Pair sparse entries by their low-half index.
            // For index j in low half (j < halfSize): paired with j + halfSize in high half.
            var paired = [Int: (Fr, Fr, Fr, Fr)]()  // lowIdx -> (a0, a1, m0, m1)
            for (idx, addCoeff, mulCoeff) in sparseWiring {
                let lowIdx = idx % halfSize
                let isHigh = idx >= halfSize
                var entry = paired[lowIdx] ?? (Fr.zero, Fr.zero, Fr.zero, Fr.zero)
                if !isHigh {
                    entry.0 = frAdd(entry.0, addCoeff)
                    entry.2 = frAdd(entry.2, mulCoeff)
                } else {
                    entry.1 = frAdd(entry.1, addCoeff)
                    entry.3 = frAdd(entry.3, mulCoeff)
                }
                paired[lowIdx] = entry
            }

            var s0 = Fr.zero
            var s1 = Fr.zero
            var s2 = Fr.zero

            if isXPhase {
                let vxHalf = curVx.count / 2
                let ySize = curVy.count
                let yMask = ySize - 1

                for (j, coeffs) in paired {
                    let (a0, a1, m0, m1) = coeffs
                    let yIdx = j & yMask
                    let xIdx = j >> nIn

                    let vx0 = xIdx < vxHalf ? curVx[xIdx] : Fr.zero
                    let vx1 = (xIdx + vxHalf) < curVx.count ? curVx[xIdx + vxHalf] : Fr.zero
                    let vyVal = yIdx < curVy.count ? curVy[yIdx] : Fr.zero

                    let g0 = frAdd(frMul(a0, frAdd(vx0, vyVal)), frMul(m0, frMul(vx0, vyVal)))
                    s0 = frAdd(s0, g0)

                    let g1 = frAdd(frMul(a1, frAdd(vx1, vyVal)), frMul(m1, frMul(vx1, vyVal)))
                    s1 = frAdd(s1, g1)

                    let a2 = frSub(frAdd(a1, a1), a0)
                    let m2 = frSub(frAdd(m1, m1), m0)
                    let vx2 = frSub(frAdd(vx1, vx1), vx0)
                    let g2 = frAdd(frMul(a2, frAdd(vx2, vyVal)), frMul(m2, frMul(vx2, vyVal)))
                    s2 = frAdd(s2, g2)
                }
            } else {
                let vxScalar = curVx.count > 0 ? curVx[0] : Fr.zero
                let vyHalf = curVy.count / 2

                for (j, coeffs) in paired {
                    let (a0, a1, m0, m1) = coeffs

                    let vy0 = j < vyHalf ? curVy[j] : Fr.zero
                    let vy1 = (j + vyHalf) < curVy.count ? curVy[j + vyHalf] : Fr.zero

                    let g0 = frAdd(frMul(a0, frAdd(vxScalar, vy0)), frMul(m0, frMul(vxScalar, vy0)))
                    s0 = frAdd(s0, g0)

                    let g1 = frAdd(frMul(a1, frAdd(vxScalar, vy1)), frMul(m1, frMul(vxScalar, vy1)))
                    s1 = frAdd(s1, g1)

                    let a2 = frSub(frAdd(a1, a1), a0)
                    let m2 = frSub(frAdd(m1, m1), m0)
                    let vy2 = frSub(frAdd(vy1, vy1), vy0)
                    let g2 = frAdd(frMul(a2, frAdd(vxScalar, vy2)), frMul(m2, frMul(vxScalar, vy2)))
                    s2 = frAdd(s2, g2)
                }
            }

            let msg = SumcheckRoundMsg(s0: s0, s1: s1, s2: s2)
            msgs.append(msg)

            transcript.absorb(s0)
            transcript.absorb(s1)
            transcript.absorb(s2)
            let challenge = transcript.squeeze()
            challenges.append(challenge)

            // Reduce sparse wiring: merge low/high halves with challenge
            let oneMinusC = frSub(Fr.one, challenge)
            sparseWiring = reduceSparseWiring(sparseWiring, halfSize: halfSize, challenge: challenge, oneMinusC: oneMinusC)
            currentTableSize = halfSize

            if isXPhase {
                let vxHalf = curVx.count / 2
                if vxHalf > 0 {
                    var newVx = [Fr](repeating: Fr.zero, count: vxHalf)
                    for j in 0..<vxHalf {
                        newVx[j] = frAdd(frMul(oneMinusC, curVx[j]), frMul(challenge, curVx[j + vxHalf]))
                    }
                    curVx = newVx
                }
            } else {
                let vyHalf = curVy.count / 2
                if vyHalf > 0 {
                    var newVy = [Fr](repeating: Fr.zero, count: vyHalf)
                    for j in 0..<vyHalf {
                        newVy[j] = frAdd(frMul(oneMinusC, curVy[j]), frMul(challenge, curVy[j + vyHalf]))
                    }
                    curVy = newVy
                }
            }
        }

        let rx = Array(challenges.prefix(nIn))
        let ry = Array(challenges.suffix(nIn))
        return (msgs, rx, ry)
    }

    /// Reduce sparse wiring after fixing one variable to challenge value.
    /// For each entry at index j: if j < halfSize it's in the low half (coeff *= (1-c)),
    /// if j >= halfSize it's in the high half (new index = j - halfSize, coeff *= c).
    /// Entries at the same reduced index are summed.
    private func reduceSparseWiring(_ wiring: [(Int, Fr, Fr)], halfSize: Int, challenge: Fr, oneMinusC: Fr) -> [(Int, Fr, Fr)] {
        var dict = [Int: (Fr, Fr)]()
        for (idx, addCoeff, mulCoeff) in wiring {
            let newIdx: Int
            let scale: Fr
            if idx < halfSize {
                newIdx = idx
                scale = oneMinusC
            } else {
                newIdx = idx - halfSize
                scale = challenge
            }
            let newAdd = frMul(scale, addCoeff)
            let newMul = frMul(scale, mulCoeff)
            if let existing = dict[newIdx] {
                dict[newIdx] = (frAdd(existing.0, newAdd), frAdd(existing.1, newMul))
            } else {
                dict[newIdx] = (newAdd, newMul)
            }
        }
        return dict.compactMap { (idx, coeffs) in
            if coeffs.0.isZero && coeffs.1.isZero { return nil }
            return (idx, coeffs.0, coeffs.1)
        }
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
            let nOut = circuit.outputVars(layer: layerIdx)
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
