// DataParallelVerifier — Verifies a data-parallel GKR proof
//
// Same cost as single-instance GKR verification + O(N) field ops for combining.
// The verifier reconstructs the wiring check from the template circuit (shared structure)
// and checks the factored eq constraints without materializing the full combined circuit.
//
// Verification equation per layer:
//   claim == eq(r_inst, x_inst) * eq(x_inst, y_inst) *
//            [add(r_circ, x_circ, y_circ) * (vx + vy) +
//             mul(r_circ, x_circ, y_circ) * vx * vy]
//
// The verifier only needs the template wiring (not N copies), so verification cost
// is independent of N (beyond the O(N) work to build the combined output/input MLEs).

import Foundation

// MARK: - DataParallelVerifier

public class DataParallelVerifier {

    /// Precomputed inverse of 2 for Lagrange interpolation.
    private let inv2: Fr = frInverse(frAdd(Fr.one, Fr.one))

    public init() {}

    /// Verify a data-parallel proof.
    ///
    /// - Parameters:
    ///   - template: The shared circuit topology (same as used by prover).
    ///   - numInstances: Number of parallel instances (N).
    ///   - inputs: Per-instance input vectors.
    ///   - proof: The data-parallel proof to verify.
    ///   - transcript: Fiat-Shamir transcript (must match prover's).
    /// - Returns: true if the proof is valid.
    public func verify(
        template: LayeredCircuit,
        numInstances: Int,
        inputs: [[Fr]],
        proof: DataParallelGKRProof,
        transcript: Transcript
    ) -> Bool {
        let d = template.depth
        guard proof.layerProofs.count == d else { return false }
        guard proof.allOutputs.count == numInstances else { return false }

        let instBits = numInstances <= 1 ? 0 : Int(ceil(log2(Double(numInstances))))
        let padN = 1 << instBits

        // Reconstruct combined output
        let outputPadSize = template.layers[d - 1].paddedSize
        let combinedOutput = buildCombined(
            values: proof.allOutputs, padSize: outputPadSize, padN: padN)

        // Absorb output (must match prover)
        for v in combinedOutput { transcript.absorb(v) }
        transcript.absorbLabel("dp-gkr-init")

        let outputCircuitVars = template.outputVars(layer: d - 1)
        let totalOutputVars = instBits + outputCircuitVars
        var r = transcript.squeezeN(totalOutputVars)

        let outputMLE = MultilinearPoly(numVars: totalOutputVars, values: combinedOutput)
        var claim = outputMLE.evaluate(at: r)

        // Verify each layer
        for layerIdx in stride(from: d - 1, through: 0, by: -1) {
            let nOutCircuit = template.outputVars(layer: layerIdx)
            let nInCircuit = verifierInputVars(template: template, layerIdx: layerIdx)

            let rInstance = Array(r.prefix(instBits))
            let rCircuit = Array(r.suffix(nOutCircuit))

            let layerProof = proof.layerProofs[d - 1 - layerIdx]
            let totalVars = 2 * instBits + 2 * nInCircuit
            guard layerProof.sumcheckMsgs.count == totalVars else { return false }

            // Replay sumcheck
            var currentClaim = claim
            var challenges = [Fr]()
            challenges.reserveCapacity(totalVars)

            for roundIdx in 0..<totalVars {
                let msg = layerProof.sumcheckMsgs[roundIdx]

                // Check: s(0) + s(1) == currentClaim
                let sum = frAdd(msg.s0, msg.s1)
                if !dpvFrEqual(sum, currentClaim) {
                    print("  [DBG] Layer \(layerIdx) round \(roundIdx): sumcheck consistency fail")
                    print("  [DBG]   sum=\(frToInt(sum)) vs claim=\(frToInt(currentClaim))")
                    return false
                }

                transcript.absorb(msg.s0)
                transcript.absorb(msg.s1)
                transcript.absorb(msg.s2)
                let challenge = transcript.squeeze()
                challenges.append(challenge)

                // Update claim via Lagrange interpolation at the challenge point
                currentClaim = lagrangeEval3(s0: msg.s0, s1: msg.s1, s2: msg.s2, at: challenge)
            }

            // Extract challenge points
            let xiChallenges = Array(challenges[0..<instBits])
            let xcChallenges = Array(challenges[instBits..<(instBits + nInCircuit)])
            let yiChallenges = Array(challenges[(instBits + nInCircuit)..<(2 * instBits + nInCircuit)])
            let ycChallenges = Array(challenges[(2 * instBits + nInCircuit)...])

            let vx = layerProof.claimedVx
            let vy = layerProof.claimedVy

            // Verify the GKR equation with factored wiring:
            // expected = eq(r_inst, x_inst) * eq(x_inst, y_inst) *
            //            [add(r_circ, x_circ, y_circ) * (vx + vy) +
            //             mul(r_circ, x_circ, y_circ) * vx * vy]

            // 1. eq(r_inst, x_inst) — instance selector
            let eqRInst = evaluateEq(rInstance, xiChallenges)

            // 2. eq(x_inst, y_inst) — cross-instance independence constraint
            let eqXiYi = evaluateEq(xiChallenges, yiChallenges)

            // 3. Wiring MLEs from template (shared, computed once)
            let addMLE = buildWiringMLE(template: template, layerIdx: layerIdx, type: .add)
            let mulMLE = buildWiringMLE(template: template, layerIdx: layerIdx, type: .mul)

            // Evaluate wiring at (r_circ, x_circ, y_circ)
            let fullCircuitPoint = rCircuit + xcChallenges + ycChallenges
            let addVal = addMLE.evaluate(at: fullCircuitPoint)
            let mulVal = mulMLE.evaluate(at: fullCircuitPoint)

            // Compute expected value
            let expected = frMul(
                frMul(eqRInst, eqXiYi),
                frAdd(
                    frMul(addVal, frAdd(vx, vy)),
                    frMul(mulVal, frMul(vx, vy))
                )
            )

            if !dpvFrEqual(currentClaim, expected) {
                print("  [DBG] Layer \(layerIdx): GKR equation mismatch")
                print("  [DBG]   currentClaim = \(frToInt(currentClaim))")
                print("  [DBG]   expected     = \(frToInt(expected))")
                print("  [DBG]   eqRInst=\(frToInt(eqRInst)), eqXiYi=\(frToInt(eqXiYi))")
                print("  [DBG]   addVal=\(frToInt(addVal)), mulVal=\(frToInt(mulVal))")
                print("  [DBG]   vx=\(frToInt(layerProof.claimedVx)), vy=\(frToInt(layerProof.claimedVy))")
                return false
            }

            // Absorb claimed evaluations
            transcript.absorb(vx)
            transcript.absorb(vy)
            transcript.absorbLabel("dp-layer-\(layerIdx)")

            // Combine claims for next layer
            let beta = transcript.squeeze()
            let totalInVars = instBits + nInCircuit
            var newR = [Fr]()
            newR.reserveCapacity(totalInVars)
            for i in 0..<totalInVars {
                let rx_i = (i < xiChallenges.count + xcChallenges.count)
                    ? (i < xiChallenges.count ? xiChallenges[i] : xcChallenges[i - xiChallenges.count])
                    : Fr.zero
                let ry_i = (i < yiChallenges.count + ycChallenges.count)
                    ? (i < yiChallenges.count ? yiChallenges[i] : ycChallenges[i - yiChallenges.count])
                    : Fr.zero
                newR.append(frAdd(rx_i, frMul(beta, frSub(ry_i, rx_i))))
            }
            r = newR
            claim = frAdd(vx, frMul(beta, frSub(vy, vx)))
        }

        // Final check: claim == MLE(combined_inputs)(r)
        let inputSize = inputSizeFromTemplate(template)
        let inputCircuitVars = inputSize <= 1 ? 0 : Int(ceil(log2(Double(inputSize))))
        let paddedInputSize = 1 << inputCircuitVars
        let combinedInput = buildCombined(values: inputs, padSize: paddedInputSize, padN: padN)
        let totalInputVars = instBits + inputCircuitVars
        let inputMLE = MultilinearPoly(numVars: totalInputVars, values: combinedInput)
        let inputEval = inputMLE.evaluate(at: r)
        if !dpvFrEqual(claim, inputEval) {
            print("  [DBG] Final input check FAIL: claim=\(frToInt(claim)), inputEval=\(frToInt(inputEval))")
            return false
        }
        return true
    }

    // MARK: - Helpers

    /// Build combined array from per-instance values.
    private func buildCombined(values: [[Fr]], padSize: Int, padN: Int) -> [Fr] {
        var combined = [Fr](repeating: Fr.zero, count: padN * padSize)
        for (inst, vals) in values.enumerated() {
            for (g, v) in vals.prefix(padSize).enumerated() {
                combined[inst * padSize + g] = v
            }
        }
        return combined
    }

    /// Compute input variables for a given layer.
    private func verifierInputVars(template: LayeredCircuit, layerIdx: Int) -> Int {
        if layerIdx == 0 {
            var maxIdx = 0
            for gate in template.layers[0].gates {
                maxIdx = max(maxIdx, gate.leftInput, gate.rightInput)
            }
            return (maxIdx + 1) <= 1 ? 0 : Int(ceil(log2(Double(maxIdx + 1))))
        } else {
            return template.layers[layerIdx - 1].numVars
        }
    }

    /// Get input size from template.
    private func inputSizeFromTemplate(_ template: LayeredCircuit) -> Int {
        guard !template.layers.isEmpty else { return 0 }
        var maxIdx = 0
        for g in template.layers[0].gates {
            maxIdx = max(maxIdx, g.leftInput, g.rightInput)
        }
        return maxIdx + 1
    }

    /// Build wiring MLE for the template circuit.
    private func buildWiringMLE(template: LayeredCircuit, layerIdx: Int, type: GateType) -> MultilinearPoly {
        let nOut = template.outputVars(layer: layerIdx)
        let nIn = verifierInputVars(template: template, layerIdx: layerIdx)
        let totalVars = nOut + 2 * nIn
        let totalSize = 1 << totalVars
        let inSize = 1 << nIn

        var evals = [Fr](repeating: Fr.zero, count: totalSize)
        for (gIdx, gate) in template.layers[layerIdx].gates.enumerated() {
            guard gate.type == type else { continue }
            let idx = gIdx * inSize * inSize + gate.leftInput * inSize + gate.rightInput
            if idx < totalSize {
                evals[idx] = Fr.one
            }
        }
        return MultilinearPoly(numVars: totalVars, evals: evals)
    }

    /// Evaluate eq(a, b) = product_i (a_i * b_i + (1 - a_i) * (1 - b_i)).
    private func evaluateEq(_ a: [Fr], _ b: [Fr]) -> Fr {
        precondition(a.count == b.count)
        var result = Fr.one
        for i in 0..<a.count {
            let term = frAdd(
                frMul(a[i], b[i]),
                frMul(frSub(Fr.one, a[i]), frSub(Fr.one, b[i]))
            )
            result = frMul(result, term)
        }
        return result
    }

    /// Lagrange interpolation: evaluate degree-2 polynomial through (0,s0),(1,s1),(2,s2) at x.
    private func lagrangeEval3(s0: Fr, s1: Fr, s2: Fr, at x: Fr) -> Fr {
        let one = Fr.one
        let two = frAdd(one, one)
        let xm1 = frSub(x, one)
        let xm2 = frSub(x, two)
        let negOne = frSub(Fr.zero, one)

        let l0 = frMul(frMul(xm1, xm2), inv2)
        let l1 = frMul(frMul(x, xm2), negOne)
        let l2 = frMul(frMul(x, xm1), inv2)

        return frAdd(frAdd(frMul(s0, l0), frMul(s1, l1)), frMul(s2, l2))
    }
}

/// Compare two Fr elements for equality via subtraction.
private func dpvFrEqual(_ a: Fr, _ b: Fr) -> Bool {
    let diff = frSub(a, b)
    let limbs = frToInt(diff)
    return limbs[0] == 0 && limbs[1] == 0 && limbs[2] == 0 && limbs[3] == 0
}
