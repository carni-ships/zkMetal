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
    /// Uses Fiat-Shamir transcript for non-interactive challenges.
    public func prove(inputs: [Fr], transcript: Transcript) -> GKRProof {
        // Evaluate the full circuit
        let allValues = circuit.evaluate(inputs: inputs)
        // allValues[0] = inputs, allValues[1] = output of layer 0, ...

        let d = circuit.depth

        // Absorb the output into the transcript
        let outputValues = allValues[d]
        for v in outputValues {
            transcript.absorb(v)
        }
        transcript.absorbLabel("gkr-init")

        // Get random point for the output layer
        let outputVars = circuit.outputVars(layer: d - 1)
        var r = transcript.squeezeN(outputVars)

        // The initial claim: V_d(r)
        let outputMLE = MultilinearPoly(numVars: outputVars, values: outputValues)
        var claim = outputMLE.evaluate(at: r)

        var layerProofs = [GKRLayerProof]()

        // Process layers from output (d-1) down to input (0)
        for layerIdx in stride(from: d - 1, through: 0, by: -1) {
            let nOut = circuit.outputVars(layer: layerIdx)
            let nIn = circuit.inputVars(layer: layerIdx)

            // Get the values at the input to this layer
            let prevValues = allValues[layerIdx]
            let prevMLE = MultilinearPoly(numVars: nIn, values: prevValues)

            // Build wiring predicate MLEs
            let addMLE = circuit.addMLEForLayer(layerIdx)
            let mulMLE = circuit.mulMLEForLayer(layerIdx)

            // Run sumcheck for this layer using the full-table approach
            let (msgs, rx, ry) = proverSumcheckFullTable(
                r: r, addMLE: addMLE, mulMLE: mulMLE,
                prevMLE: prevMLE, nOut: nOut, nIn: nIn,
                transcript: transcript
            )

            // Evaluate V_{i-1} at the random points rx, ry
            let vx = prevMLE.evaluate(at: rx)
            let vy = prevMLE.evaluate(at: ry)

            // Absorb evaluations
            transcript.absorb(vx)
            transcript.absorb(vy)
            transcript.absorbLabel("gkr-layer-\(layerIdx)")

            layerProofs.append(GKRLayerProof(sumcheckMsgs: msgs, claimedVx: vx, claimedVy: vy))

            // Combine rx and ry for next layer
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

    /// Sumcheck for one GKR layer using full-table materialization.
    ///
    /// The function being summed is:
    ///   g(x, y) = add_r(x,y) * (V(x) + V(y)) + mul_r(x,y) * V(x) * V(y)
    /// where add_r, mul_r have the output variable fixed to r.
    ///
    /// We materialize the full table of g over {0,1}^{2*nIn} and iteratively reduce.
    private func proverSumcheckFullTable(
        r: [Fr],
        addMLE: MultilinearPoly, mulMLE: MultilinearPoly,
        prevMLE: MultilinearPoly,
        nOut: Int, nIn: Int,
        transcript: Transcript
    ) -> (msgs: [SumcheckRoundMsg], rx: [Fr], ry: [Fr]) {

        // Fix the output variables in wiring predicates
        var addFixed = addMLE
        for i in 0..<nOut {
            addFixed = addFixed.fixVariable(r[i])
        }
        var mulFixed = mulMLE
        for i in 0..<nOut {
            mulFixed = mulFixed.fixVariable(r[i])
        }

        let totalVars = 2 * nIn
        let tableSize = 1 << totalVars
        let inSize = 1 << nIn

        // Materialize the full sumcheck table:
        // table[idx] = g(x, y) where idx = x * inSize + y
        var table = [Fr](repeating: Fr.zero, count: tableSize)
        let prevEvals = prevMLE.evals
        let addEvals = addFixed.evals
        let mulEvals = mulFixed.evals

        for idx in 0..<tableSize {
            let xIdx = idx >> nIn  // upper nIn bits = x
            let yIdx = idx & (inSize - 1)  // lower nIn bits = y

            let vx = xIdx < prevEvals.count ? prevEvals[xIdx] : Fr.zero
            let vy = yIdx < prevEvals.count ? prevEvals[yIdx] : Fr.zero
            let aVal = idx < addEvals.count ? addEvals[idx] : Fr.zero
            let mVal = idx < mulEvals.count ? mulEvals[idx] : Fr.zero

            // g = add(x,y) * (V(x) + V(y)) + mul(x,y) * V(x) * V(y)
            let gAdd = frMul(aVal, frAdd(vx, vy))
            let gMul = frMul(mVal, frMul(vx, vy))
            table[idx] = frAdd(gAdd, gMul)
        }

        // Now run standard sumcheck on the materialized table
        var msgs = [SumcheckRoundMsg]()
        msgs.reserveCapacity(totalVars)
        var challenges = [Fr]()
        challenges.reserveCapacity(totalVars)

        var curTable = table

        for _ in 0..<totalVars {
            let currentSize = curTable.count
            let halfSize = currentSize / 2

            var s0 = Fr.zero
            var s1 = Fr.zero
            var s2 = Fr.zero

            for j in 0..<halfSize {
                let f0 = curTable[j]           // variable = 0
                let f1 = curTable[j + halfSize] // variable = 1
                s0 = frAdd(s0, f0)
                s1 = frAdd(s1, f1)
                // f(2) = 2*f(1) - f(0) for linear interpolation
                s2 = frAdd(s2, frSub(frAdd(f1, f1), f0))
            }

            let msg = SumcheckRoundMsg(s0: s0, s1: s1, s2: s2)
            msgs.append(msg)

            // Absorb and get challenge
            transcript.absorb(s0)
            transcript.absorb(s1)
            transcript.absorb(s2)
            let challenge = transcript.squeeze()
            challenges.append(challenge)

            // Reduce: fix this variable to the challenge
            let oneMinusC = frSub(Fr.one, challenge)
            var newTable = [Fr](repeating: Fr.zero, count: halfSize)
            for j in 0..<halfSize {
                newTable[j] = frAdd(frMul(oneMinusC, curTable[j]),
                                   frMul(challenge, curTable[j + halfSize]))
            }
            curTable = newTable
        }

        let rx = Array(challenges.prefix(nIn))
        let ry = Array(challenges.suffix(nIn))

        return (msgs, rx, ry)
    }

    // MARK: - Verifier

    /// Verify a GKR proof for the given output.
    /// The verifier needs: the circuit structure, claimed output, proof, and inputs for final check.
    public func verify(inputs: [Fr], output: [Fr], proof: GKRProof, transcript: Transcript) -> Bool {
        let d = circuit.depth
        guard proof.layerProofs.count == d else { return false }

        // Absorb output
        for v in output {
            transcript.absorb(v)
        }
        transcript.absorbLabel("gkr-init")

        // Get random point for the output layer
        let outputVars = circuit.outputVars(layer: d - 1)
        var r = transcript.squeezeN(outputVars)

        // Initial claim: MLE of output evaluated at r
        let outputMLE = MultilinearPoly(numVars: outputVars, values: output)
        var claim = outputMLE.evaluate(at: r)

        for layerIdx in stride(from: d - 1, through: 0, by: -1) {
            let nOut = circuit.outputVars(layer: layerIdx)
            let nIn = circuit.inputVars(layer: layerIdx)
            let layerProof = proof.layerProofs[d - 1 - layerIdx]

            let totalVars = 2 * nIn
            guard layerProof.sumcheckMsgs.count == totalVars else { return false }

            // Verify sumcheck rounds
            var currentClaim = claim
            var challenges = [Fr]()
            challenges.reserveCapacity(totalVars)

            for roundIdx in 0..<totalVars {
                let msg = layerProof.sumcheckMsgs[roundIdx]

                // Check: s(0) + s(1) = current claim
                let sum = frAdd(msg.s0, msg.s1)
                if !gkrFrEqual(sum, currentClaim) {
                    return false
                }

                // Absorb and get challenge
                transcript.absorb(msg.s0)
                transcript.absorb(msg.s1)
                transcript.absorb(msg.s2)
                let challenge = transcript.squeeze()
                challenges.append(challenge)

                // Next claim = s(challenge) via Lagrange interpolation
                currentClaim = lagrangeEval3(s0: msg.s0, s1: msg.s1, s2: msg.s2, at: challenge)
            }

            let rx = Array(challenges.prefix(nIn))
            let ry = Array(challenges.suffix(nIn))

            let vx = layerProof.claimedVx
            let vy = layerProof.claimedVy

            // Evaluate wiring predicates at (r, rx, ry)
            let addMLE = circuit.addMLEForLayer(layerIdx)
            let mulMLE = circuit.mulMLEForLayer(layerIdx)
            let fullPoint = r + rx + ry
            let addVal = addMLE.evaluate(at: fullPoint)
            let mulVal = mulMLE.evaluate(at: fullPoint)

            // Check: final claim = add(r,rx,ry) * (vx + vy) + mul(r,rx,ry) * vx * vy
            let expected = frAdd(frMul(addVal, frAdd(vx, vy)), frMul(mulVal, frMul(vx, vy)))
            if !gkrFrEqual(currentClaim, expected) {
                return false
            }

            // Absorb claimed evaluations
            transcript.absorb(vx)
            transcript.absorb(vy)
            transcript.absorbLabel("gkr-layer-\(layerIdx)")

            // Combine for next layer
            let beta = transcript.squeeze()
            var newR = [Fr]()
            newR.reserveCapacity(nIn)
            for i in 0..<nIn {
                newR.append(frAdd(rx[i], frMul(beta, frSub(ry[i], rx[i]))))
            }
            r = newR
            claim = frAdd(vx, frMul(beta, frSub(vy, vx)))
        }

        // Final check: claim should equal the MLE of the inputs evaluated at r
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

    // L0(x) = (x-1)(x-2)/2
    let l0 = frMul(frMul(xm1, xm2), inv2)
    // L1(x) = -x(x-2)
    let l1 = frMul(frMul(x, xm2), negOne)
    // L2(x) = x(x-1)/2
    let l2 = frMul(frMul(x, xm1), inv2)

    return frAdd(frAdd(frMul(s0, l0), frMul(s1, l1)), frMul(s2, l2))
}
