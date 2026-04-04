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

            let addMLE = circuit.addMLEForLayer(layerIdx)
            let mulMLE = circuit.mulMLEForLayer(layerIdx)

            let (msgs, rx, ry) = proverBatchedSumcheck(
                rPoints: rPoints,
                addMLE: addMLE, mulMLE: mulMLE,
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

    /// Batched sumcheck for one GKR layer.
    ///
    /// Given multiple (output-point, weight) pairs, the function being summed is:
    ///   g(a, b) = sum_k w_k * [add_k(a,b) * (V(a)+V(b)) + mul_k(a,b) * V(a)*V(b)]
    ///
    /// where add_k(a,b) = add(r_k, a, b) is the add wiring MLE with z fixed to r_k.
    ///
    /// This is degree 2 in each variable, so we need evaluations at t=0,1,2.
    private func proverBatchedSumcheck(
        rPoints: [([Fr], Fr)],  // [(output_point, weight)]
        addMLE: MultilinearPoly, mulMLE: MultilinearPoly,
        prevMLE: MultilinearPoly,
        nOut: Int, nIn: Int,
        transcript: Transcript
    ) -> (msgs: [SumcheckRoundMsg], rx: [Fr], ry: [Fr]) {

        let totalVars = 2 * nIn

        // Build combined wiring tables: sum_k w_k * add(r_k, ., .) and sum_k w_k * mul(r_k, ., .)
        let xySize = 1 << totalVars
        var curAdd = [Fr](repeating: Fr.zero, count: xySize)
        var curMul = [Fr](repeating: Fr.zero, count: xySize)

        for (rk, wk) in rPoints {
            var addFixed = addMLE
            for i in 0..<nOut { addFixed = addFixed.fixVariable(rk[i]) }
            var mulFixed = mulMLE
            for i in 0..<nOut { mulFixed = mulFixed.fixVariable(rk[i]) }

            for j in 0..<xySize {
                curAdd[j] = frAdd(curAdd[j], frMul(wk, addFixed.evals[j]))
                curMul[j] = frAdd(curMul[j], frMul(wk, mulFixed.evals[j]))
            }
        }

        var curVx = prevMLE.evals
        var curVy = prevMLE.evals

        var msgs = [SumcheckRoundMsg]()
        msgs.reserveCapacity(totalVars)
        var challenges = [Fr]()
        challenges.reserveCapacity(totalVars)

        for round in 0..<totalVars {
            let wiringsSize = curAdd.count
            let halfWirings = wiringsSize / 2
            let isXPhase = round < nIn

            var s0 = Fr.zero
            var s1 = Fr.zero
            var s2 = Fr.zero

            if isXPhase {
                let vxHalf = curVx.count / 2
                let ySize = curVy.count

                for j in 0..<halfWirings {
                    let a0 = curAdd[j]
                    let a1 = curAdd[j + halfWirings]
                    let m0 = curMul[j]
                    let m1 = curMul[j + halfWirings]

                    let yIdx = j & (ySize - 1)
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

                for j in 0..<halfWirings {
                    let a0 = curAdd[j]
                    let a1 = curAdd[j + halfWirings]
                    let m0 = curMul[j]
                    let m1 = curMul[j + halfWirings]

                    let yIdx = j

                    let vy0 = yIdx < vyHalf ? curVy[yIdx] : Fr.zero
                    let vy1 = (yIdx + vyHalf) < curVy.count ? curVy[yIdx + vyHalf] : Fr.zero

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

            let oneMinusC = frSub(Fr.one, challenge)
            var newAdd = [Fr](repeating: Fr.zero, count: halfWirings)
            var newMul = [Fr](repeating: Fr.zero, count: halfWirings)
            for j in 0..<halfWirings {
                newAdd[j] = frAdd(frMul(oneMinusC, curAdd[j]), frMul(challenge, curAdd[j + halfWirings]))
                newMul[j] = frAdd(frMul(oneMinusC, curMul[j]), frMul(challenge, curMul[j + halfWirings]))
            }
            curAdd = newAdd
            curMul = newMul

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
            let addMLE = circuit.addMLEForLayer(layerIdx)
            let mulMLE = circuit.mulMLEForLayer(layerIdx)

            var expected = Fr.zero
            for (rk, wk) in rPoints {
                let fullPoint = rk + rx + ry
                let addVal = addMLE.evaluate(at: fullPoint)
                let mulVal = mulMLE.evaluate(at: fullPoint)
                let contribution = frAdd(frMul(addVal, frAdd(vx, vy)), frMul(mulVal, frMul(vx, vy)))
                expected = frAdd(expected, frMul(wk, contribution))
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
