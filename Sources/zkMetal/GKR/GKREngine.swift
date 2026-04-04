// GKR (Goldwasser-Kalai-Rothblum) Interactive Proof Engine
// Efficiently proves layered arithmetic circuit evaluations using sumcheck at each layer.
// No FFT/NTT needed — purely algebraic protocol based on multilinear extensions.
//
// Protocol outline (Fiat-Shamir non-interactive):
// 1. Prover evaluates circuit, commits output
// 2. For each layer from output to input:
//    a. Run sumcheck on the GKR equation to reduce the claim
//    b. Prover sends evaluations of the previous layer's MLE at the sumcheck output points
//    c. Verifier combines claims using random linear combination
// 3. At the input layer, verifier checks the MLE evaluation directly

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
    public func prove(inputs: [Fr], transcript: Transcript) -> GKRProof {
        let allValues = circuit.evaluate(inputs: inputs)
        let d = circuit.depth

        // Absorb output
        let outputValues = allValues[d]
        for v in outputValues { transcript.absorb(v) }
        transcript.absorbLabel("gkr-init")

        let outputVars = circuit.outputVars(layer: d - 1)
        var r = transcript.squeezeN(outputVars)

        let outputMLE = MultilinearPoly(numVars: outputVars, values: outputValues)
        var claim = outputMLE.evaluate(at: r)
        _ = claim  // used implicitly via transcript consistency

        var layerProofs = [GKRLayerProof]()

        for layerIdx in stride(from: d - 1, through: 0, by: -1) {
            let nOut = circuit.outputVars(layer: layerIdx)
            let nIn = circuit.inputVars(layer: layerIdx)
            let prevValues = allValues[layerIdx]
            let prevMLE = MultilinearPoly(numVars: nIn, values: prevValues)

            let addMLE = circuit.addMLEForLayer(layerIdx)
            let mulMLE = circuit.mulMLEForLayer(layerIdx)

            let (msgs, rx, ry) = proverSumcheckForLayer(
                r: r, addMLE: addMLE, mulMLE: mulMLE,
                prevMLE: prevMLE, nOut: nOut, nIn: nIn,
                transcript: transcript
            )

            let vx = prevMLE.evaluate(at: rx)
            let vy = prevMLE.evaluate(at: ry)

            transcript.absorb(vx)
            transcript.absorb(vy)
            transcript.absorbLabel("gkr-layer-\(layerIdx)")

            layerProofs.append(GKRLayerProof(sumcheckMsgs: msgs, claimedVx: vx, claimedVy: vy))

            let beta = transcript.squeeze()
            var newR = [Fr]()
            newR.reserveCapacity(nIn)
            for i in 0..<nIn {
                newR.append(frAdd(rx[i], frMul(beta, frSub(ry[i], rx[i]))))
            }
            r = newR
            claim = frAdd(vx, frMul(beta, frSub(vy, vx)))
        }

        return GKRProof(layerProofs: layerProofs)
    }

    /// Sumcheck for one GKR layer.
    ///
    /// The function being summed is:
    ///   g(x, y) = add_r(x,y) * (V(x) + V(y)) + mul_r(x,y) * V(x) * V(y)
    ///
    /// This is degree 2 in each variable (due to the mul*V*V product), so we need
    /// evaluations at 3 points (t=0,1,2) per round. We maintain separate bookkeeping
    /// tables for add, mul, and the V polynomial evaluations.
    private func proverSumcheckForLayer(
        r: [Fr],
        addMLE: MultilinearPoly, mulMLE: MultilinearPoly,
        prevMLE: MultilinearPoly,
        nOut: Int, nIn: Int,
        transcript: Transcript
    ) -> (msgs: [SumcheckRoundMsg], rx: [Fr], ry: [Fr]) {

        // Fix the output variables in wiring predicates
        var addFixed = addMLE
        for i in 0..<nOut { addFixed = addFixed.fixVariable(r[i]) }
        var mulFixed = mulMLE
        for i in 0..<nOut { mulFixed = mulFixed.fixVariable(r[i]) }

        let totalVars = 2 * nIn
        let inSize = 1 << nIn

        // Bookkeeping tables (indexed over the 2*nIn xy-space)
        var curAdd = addFixed.evals   // 2^(2*nIn) entries
        var curMul = mulFixed.evals
        // V tables: for x-phase, we reduce the V(x) table; V(y) stays full
        //           for y-phase, V(x) is a scalar, we reduce V(y)
        var curVx = prevMLE.evals     // 2^nIn entries, reduced during x-phase
        var curVy = prevMLE.evals     // 2^nIn entries, reduced during y-phase

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
                // Current variable is an x-variable (x_{round})
                // For the wiring tables, half = halfWirings
                // For V(x), half = curVx.count / 2
                let vxHalf = curVx.count / 2
                let ySize = curVy.count  // V(y) table is full

                // The remaining free variables after the current one:
                // x-vars: round+1..nIn-1 (nIn-round-1 bits)
                // y-vars: 0..nIn-1 (nIn bits)
                // total remaining after this round: nIn-round-1 + nIn bits
                // halfWirings = 2^(totalVars - round - 1) / 1... actually halfWirings = wiringsSize/2

                // Index decomposition for the wiring tables:
                // The wiring table has totalVars-round variables remaining.
                // Current variable is the MSB. Index j in [0, halfWirings) represents
                // the remaining (totalVars-round-1) variables.
                // Of these, the first (nIn-round-1) are remaining x-vars, the last nIn are y-vars.

                let remainingXBits = nIn - round - 1
                let yBits = nIn

                for j in 0..<halfWirings {
                    let a0 = curAdd[j]
                    let a1 = curAdd[j + halfWirings]
                    let m0 = curMul[j]
                    let m1 = curMul[j + halfWirings]

                    // Extract x-index and y-index from j
                    let yIdx = j & (ySize - 1)  // lower nIn bits
                    let xIdx = j >> yBits        // upper (nIn-round-1) bits

                    // V(x) at current variable = 0 and = 1
                    let vx0 = xIdx < vxHalf ? curVx[xIdx] : Fr.zero
                    let vx1 = (xIdx + vxHalf) < curVx.count ? curVx[xIdx + vxHalf] : Fr.zero

                    // V(y) doesn't depend on x-variables
                    let vyVal = yIdx < curVy.count ? curVy[yIdx] : Fr.zero

                    // g(t, ...) = add_t * (Vx_t + Vy) + mul_t * Vx_t * Vy
                    // where add_t = (1-t)*a0 + t*a1, mul_t = (1-t)*m0 + t*m1, Vx_t = (1-t)*vx0 + t*vx1

                    // At t=0:
                    let g0 = frAdd(frMul(a0, frAdd(vx0, vyVal)), frMul(m0, frMul(vx0, vyVal)))
                    s0 = frAdd(s0, g0)

                    // At t=1:
                    let g1 = frAdd(frMul(a1, frAdd(vx1, vyVal)), frMul(m1, frMul(vx1, vyVal)))
                    s1 = frAdd(s1, g1)

                    // At t=2: extrapolate each component linearly, then combine
                    let a2 = frSub(frAdd(a1, a1), a0)   // 2*a1 - a0
                    let m2 = frSub(frAdd(m1, m1), m0)
                    let vx2 = frSub(frAdd(vx1, vx1), vx0)
                    let g2 = frAdd(frMul(a2, frAdd(vx2, vyVal)), frMul(m2, frMul(vx2, vyVal)))
                    s2 = frAdd(s2, g2)
                }
            } else {
                // Current variable is a y-variable (y_{round - nIn})
                // V(x) is fully reduced to a scalar
                let vxScalar = curVx.count > 0 ? curVx[0] : Fr.zero
                let vyHalf = curVy.count / 2

                let remainingYBits = nIn - (round - nIn) - 1

                for j in 0..<halfWirings {
                    let a0 = curAdd[j]
                    let a1 = curAdd[j + halfWirings]
                    let m0 = curMul[j]
                    let m1 = curMul[j + halfWirings]

                    // Extract y-index from j (the remaining y-bits)
                    let yIdx = j  // j indexes the remaining (nIn-(round-nIn)-1) y-vars

                    // V(y) at current variable = 0 and = 1
                    let vy0 = yIdx < vyHalf ? curVy[yIdx] : Fr.zero
                    let vy1 = (yIdx + vyHalf) < curVy.count ? curVy[yIdx + vyHalf] : Fr.zero

                    // g(t, ...) = add_t * (Vx + Vy_t) + mul_t * Vx * Vy_t
                    // At t=0:
                    let g0 = frAdd(frMul(a0, frAdd(vxScalar, vy0)), frMul(m0, frMul(vxScalar, vy0)))
                    s0 = frAdd(s0, g0)

                    // At t=1:
                    let g1 = frAdd(frMul(a1, frAdd(vxScalar, vy1)), frMul(m1, frMul(vxScalar, vy1)))
                    s1 = frAdd(s1, g1)

                    // At t=2:
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

            // Reduce wiring tables
            let oneMinusC = frSub(Fr.one, challenge)
            var newAdd = [Fr](repeating: Fr.zero, count: halfWirings)
            var newMul = [Fr](repeating: Fr.zero, count: halfWirings)
            for j in 0..<halfWirings {
                newAdd[j] = frAdd(frMul(oneMinusC, curAdd[j]), frMul(challenge, curAdd[j + halfWirings]))
                newMul[j] = frAdd(frMul(oneMinusC, curMul[j]), frMul(challenge, curMul[j + halfWirings]))
            }
            curAdd = newAdd
            curMul = newMul

            // Reduce V tables
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
        var r = transcript.squeezeN(outputVars)

        let outputMLE = MultilinearPoly(numVars: outputVars, values: output)
        var claim = outputMLE.evaluate(at: r)

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

            // Verify: final sumcheck claim = add(r,rx,ry)*(vx+vy) + mul(r,rx,ry)*vx*vy
            let addMLE = circuit.addMLEForLayer(layerIdx)
            let mulMLE = circuit.mulMLEForLayer(layerIdx)
            let fullPoint = r + rx + ry
            let addVal = addMLE.evaluate(at: fullPoint)
            let mulVal = mulMLE.evaluate(at: fullPoint)

            let expected = frAdd(frMul(addVal, frAdd(vx, vy)), frMul(mulVal, frMul(vx, vy)))
            if !gkrFrEqual(currentClaim, expected) {
                return false
            }

            transcript.absorb(vx)
            transcript.absorb(vy)
            transcript.absorbLabel("gkr-layer-\(layerIdx)")

            let beta = transcript.squeeze()
            var newR = [Fr]()
            newR.reserveCapacity(nIn)
            for i in 0..<nIn {
                newR.append(frAdd(rx[i], frMul(beta, frSub(ry[i], rx[i]))))
            }
            r = newR
            claim = frAdd(vx, frMul(beta, frSub(vy, vx)))
        }

        // Final check: claim = MLE of inputs at r
        let inputMLE = MultilinearPoly(numVars: r.count, values: inputs)
        let inputEval = inputMLE.evaluate(at: r)
        return gkrFrEqual(claim, inputEval)
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
