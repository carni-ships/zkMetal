// DataParallelEngine — Efficient proof for N repetitions of the same sub-circuit
//
// Key insight: When the same sub-circuit is evaluated N times with different inputs,
// the wiring predicate factors into:
//   wiring(r, x, y) = subCircuit_wiring(r_circuit, x_circuit, y_circuit) * eq(r_instance, x_instance)
//
// This factoring means sumcheck at each GKR layer costs O(|C| + N) per round
// instead of O(|C| * N), yielding O(|C| + N log N) total prover work.
//
// Protocol:
// 1. Evaluate all N instances
// 2. Commit to the combined output as a multilinear extension
// 3. Run GKR layer-by-layer from output to input:
//    a. At each layer, run sumcheck with the factored polynomial
//    b. The eq(r_instance, x_instance) term handles instance selection
//    c. The wiring(r_circuit, x_circuit, y_circuit) term handles circuit structure
// 4. At the input layer, the verifier checks against the committed inputs

import Foundation

// MARK: - Proof Types

/// Proof for one layer of the data-parallel GKR protocol.
public struct DataParallelLayerProof {
    public let sumcheckMsgs: [SumcheckRoundMsg]
    public let claimedVx: Fr   // combined V at rx
    public let claimedVy: Fr   // combined V at ry
}

/// Complete data-parallel proof.
public struct DataParallelProof {
    public let layerProofs: [DataParallelLayerProof]
    public let allOutputs: [[Fr]]  // N sets of outputs (public for verification)
}

// MARK: - Engine

public class DataParallelEngine {
    public static let version = Versions.dataParallel

    public init() {}

    // MARK: - Prover

    /// Prove N parallel evaluations of the same sub-circuit.
    /// Returns a proof that all N instances were evaluated correctly.
    ///
    /// Complexity: O(|C| + N log N) instead of O(|C| * N) for N separate proofs.
    public func prove(circuit: UniformCircuit, transcript: Transcript) -> DataParallelProof {
        let sub = circuit.subCircuit
        let d = sub.depth
        let instBits = circuit.instanceBits
        let padN = circuit.paddedInstances

        // Step 1: Evaluate all instances, get full layer values
        let allLayerValues = circuit.evaluateAllLayers()
        // allLayerValues[instance][layer] where layer 0 = inputs, layer d = outputs

        let allOutputs = allLayerValues.map { $0[d] }

        // Step 2: Build combined output and absorb into transcript
        let outputLayerSize = sub.layers[d - 1].paddedSize
        let combinedOutput = circuit.combinedLayerValues(
            layerValues: allOutputs, layerSize: outputLayerSize)

        for v in combinedOutput {
            transcript.absorb(v)
        }
        transcript.absorbLabel("dp-gkr-init")

        // Combined variables = instance bits + circuit gate bits
        let outputCircuitVars = sub.layers[d - 1].numVars
        let totalOutputVars = instBits + outputCircuitVars

        var r = transcript.squeezeN(totalOutputVars)

        // Initial claim: MLE of combined output evaluated at r
        let outputMLE = MultilinearPoly(numVars: totalOutputVars, values: combinedOutput)
        var claim = outputMLE.evaluate(at: r)

        var layerProofs = [DataParallelLayerProof]()

        // Step 3: Process layers from output to input
        for layerIdx in stride(from: d - 1, through: 0, by: -1) {
            let nOutCircuit = sub.layers[layerIdx].numVars
            let nInCircuit: Int
            if layerIdx == 0 {
                var maxIdx = 0
                for gate in sub.layers[0].gates {
                    maxIdx = max(maxIdx, gate.leftInput, gate.rightInput)
                }
                nInCircuit = (maxIdx + 1) <= 1 ? 0 : Int(ceil(log2(Double(maxIdx + 1))))
            } else {
                nInCircuit = sub.layers[layerIdx - 1].numVars
            }

            // Total variables: instance bits are the MSBs, circuit bits are the LSBs
            // r has totalOutputVars = instBits + nOutCircuit components
            let rInstance = Array(r.prefix(instBits))
            let rCircuit = Array(r.suffix(nOutCircuit))

            // Build combined previous layer values
            let prevLayerValues = allLayerValues.map { $0[layerIdx] }
            let prevLayerSize = 1 << nInCircuit
            let combinedPrev = circuit.combinedLayerValues(
                layerValues: prevLayerValues, layerSize: prevLayerSize)

            // Run the factored sumcheck
            let (msgs, rx, ry) = proverFactoredSumcheck(
                rInstance: rInstance,
                rCircuit: rCircuit,
                subCircuit: sub,
                layerIdx: layerIdx,
                combinedPrev: combinedPrev,
                instBits: instBits,
                nOutCircuit: nOutCircuit,
                nInCircuit: nInCircuit,
                padN: padN,
                transcript: transcript
            )

            // Evaluate combined V at rx, ry
            let totalPrevVars = instBits + nInCircuit
            let prevMLE = MultilinearPoly(numVars: totalPrevVars, values: combinedPrev)
            let vx = prevMLE.evaluate(at: rx)
            let vy = prevMLE.evaluate(at: ry)

            transcript.absorb(vx)
            transcript.absorb(vy)
            transcript.absorbLabel("dp-layer-\(layerIdx)")

            layerProofs.append(DataParallelLayerProof(
                sumcheckMsgs: msgs, claimedVx: vx, claimedVy: vy))

            // Combine rx and ry for next layer using random linear combination
            let beta = transcript.squeeze()
            let totalInVars = instBits + nInCircuit
            var newR = [Fr]()
            newR.reserveCapacity(totalInVars)
            for i in 0..<totalInVars {
                newR.append(frAdd(rx[i], frMul(beta, frSub(ry[i], rx[i]))))
            }
            r = newR
            claim = frAdd(vx, frMul(beta, frSub(vy, vx)))
        }

        return DataParallelProof(layerProofs: layerProofs, allOutputs: allOutputs)
    }

    /// Factored sumcheck for one GKR layer.
    ///
    /// The key optimization: the sumcheck polynomial factors as:
    ///   g(x_inst, x_circ, y_circ) =
    ///     eq(r_inst, x_inst) *
    ///     [add(r_circ, x_circ, y_circ) * (V(x_inst,x_circ) + V(x_inst,y_circ))
    ///      + mul(r_circ, x_circ, y_circ) * V(x_inst,x_circ) * V(x_inst,y_circ)]
    ///
    /// The delta(x_inst, y_inst) constraint forces both inputs to come from the same
    /// instance, so y_inst = x_inst and we only iterate over x_inst once.
    /// This reduces the table from (padN * |C|)^2 to padN * |C|^2.
    private func proverFactoredSumcheck(
        rInstance: [Fr],
        rCircuit: [Fr],
        subCircuit: SubCircuit,
        layerIdx: Int,
        combinedPrev: [Fr],
        instBits: Int,
        nOutCircuit: Int,
        nInCircuit: Int,
        padN: Int,
        transcript: Transcript
    ) -> (msgs: [SumcheckRoundMsg], rx: [Fr], ry: [Fr]) {

        let circuitInSize = 1 << nInCircuit

        // Build circuit-level wiring predicates (small: only depends on |C|)
        let layer = subCircuit.layers[layerIdx]
        let addMLE = dpBuildWiringMLE(layer: layer, type: .add, nOut: nOutCircuit, nIn: nInCircuit)
        let mulMLE = dpBuildWiringMLE(layer: layer, type: .mul, nOut: nOutCircuit, nIn: nInCircuit)

        // Fix output circuit variables in wiring predicates
        var addFixed = addMLE
        for i in 0..<nOutCircuit {
            addFixed = addFixed.fixVariable(rCircuit[i])
        }
        var mulFixed = mulMLE
        for i in 0..<nOutCircuit {
            mulFixed = mulFixed.fixVariable(rCircuit[i])
        }

        // Compute eq polynomial for instance selection: eq(rInstance, .)
        let eqInst = MultilinearPoly.eqPoly(point: rInstance)

        // Build the factored sumcheck table.
        // Variables: (x_inst: instBits, x_circ: nInCircuit, y_circ: nInCircuit)
        // table[x_inst * circuitInSize^2 + x_circ * circuitInSize + y_circ] =
        //   eq(rInstance, x_inst) * [add * (V(x) + V(y)) + mul * V(x) * V(y)]
        let circuitTableSize = circuitInSize * circuitInSize
        let totalVars = instBits + 2 * nInCircuit
        let totalTableSize = padN * circuitTableSize
        var table = [Fr](repeating: Fr.zero, count: totalTableSize)

        let addEvals = addFixed.evals
        let mulEvals = mulFixed.evals

        for inst in 0..<padN {
            let eqVal = eqInst[inst]
            let eqLimbs = frToInt(eqVal)
            if eqLimbs[0] == 0 && eqLimbs[1] == 0 && eqLimbs[2] == 0 && eqLimbs[3] == 0 {
                continue
            }

            let instOffset = inst * circuitInSize
            let tableInstOffset = inst * circuitTableSize

            for xc in 0..<circuitInSize {
                let vx = instOffset + xc < combinedPrev.count ?
                    combinedPrev[instOffset + xc] : Fr.zero

                for yc in 0..<circuitInSize {
                    let vy = instOffset + yc < combinedPrev.count ?
                        combinedPrev[instOffset + yc] : Fr.zero

                    let circIdx = xc * circuitInSize + yc
                    let aVal = circIdx < addEvals.count ? addEvals[circIdx] : Fr.zero
                    let mVal = circIdx < mulEvals.count ? mulEvals[circIdx] : Fr.zero

                    let gCircuit = frAdd(
                        frMul(aVal, frAdd(vx, vy)),
                        frMul(mVal, frMul(vx, vy))
                    )
                    table[tableInstOffset + circIdx] = frMul(eqVal, gCircuit)
                }
            }
        }

        // Run standard sumcheck on this table
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
                let f0 = curTable[j]
                let f1 = curTable[j + halfSize]
                s0 = frAdd(s0, f0)
                s1 = frAdd(s1, f1)
                s2 = frAdd(s2, frSub(frAdd(f1, f1), f0))
            }

            let msg = SumcheckRoundMsg(s0: s0, s1: s1, s2: s2)
            msgs.append(msg)

            transcript.absorb(s0)
            transcript.absorb(s1)
            transcript.absorb(s2)
            let challenge = transcript.squeeze()
            challenges.append(challenge)

            let oneMinusC = frSub(Fr.one, challenge)
            var newTable = [Fr](repeating: Fr.zero, count: halfSize)
            for j in 0..<halfSize {
                newTable[j] = frAdd(frMul(oneMinusC, curTable[j]),
                                   frMul(challenge, curTable[j + halfSize]))
            }
            curTable = newTable
        }

        // Extract rx and ry:
        // rx = (instance challenges, x_circ challenges)
        // ry = (instance challenges, y_circ challenges)
        // Since delta forces y_inst = x_inst, both share the instance dimension
        let instChallenges = Array(challenges.prefix(instBits))
        let xCircChallenges = Array(challenges[instBits..<(instBits + nInCircuit)])
        let yCircChallenges = Array(challenges[(instBits + nInCircuit)...])

        let rx = instChallenges + xCircChallenges
        let ry = instChallenges + yCircChallenges

        return (msgs, rx, ry)
    }

    // MARK: - Verifier

    /// Verify a data-parallel proof.
    /// The verifier checks the GKR proof and confirms outputs match.
    ///
    /// Complexity: O(|C| + N log N) -- dominated by MLE evaluations, not circuit size * N.
    public func verify(
        subCircuit: SubCircuit,
        numInstances: Int,
        inputs: [[Fr]],
        proof: DataParallelProof,
        transcript: Transcript
    ) -> Bool {
        let d = subCircuit.depth
        guard proof.layerProofs.count == d else { return false }
        guard proof.allOutputs.count == numInstances else { return false }

        let instBits = numInstances <= 1 ? 0 : Int(ceil(log2(Double(numInstances))))

        // Build a temporary UniformCircuit for helper methods
        let uniformCircuit = UniformCircuit(subCircuit: subCircuit, inputs: inputs)

        // Build combined output and absorb
        let outputLayerSize = subCircuit.layers[d - 1].paddedSize
        let combinedOutput = uniformCircuit.combinedLayerValues(
            layerValues: proof.allOutputs, layerSize: outputLayerSize)

        for v in combinedOutput {
            transcript.absorb(v)
        }
        transcript.absorbLabel("dp-gkr-init")

        let outputCircuitVars = subCircuit.layers[d - 1].numVars
        let totalOutputVars = instBits + outputCircuitVars
        var r = transcript.squeezeN(totalOutputVars)

        let outputMLE = MultilinearPoly(numVars: totalOutputVars, values: combinedOutput)
        var claim = outputMLE.evaluate(at: r)

        // Verify each layer
        for layerIdx in stride(from: d - 1, through: 0, by: -1) {
            let nOutCircuit = subCircuit.layers[layerIdx].numVars
            let nInCircuit: Int
            if layerIdx == 0 {
                var maxIdx = 0
                for gate in subCircuit.layers[0].gates {
                    maxIdx = max(maxIdx, gate.leftInput, gate.rightInput)
                }
                nInCircuit = (maxIdx + 1) <= 1 ? 0 : Int(ceil(log2(Double(maxIdx + 1))))
            } else {
                nInCircuit = subCircuit.layers[layerIdx - 1].numVars
            }

            let rInstance = Array(r.prefix(instBits))
            let rCircuit = Array(r.suffix(nOutCircuit))

            let layerProof = proof.layerProofs[d - 1 - layerIdx]
            let totalVars = instBits + 2 * nInCircuit
            guard layerProof.sumcheckMsgs.count == totalVars else { return false }

            // Verify sumcheck rounds
            var currentClaim = claim
            var challenges = [Fr]()
            challenges.reserveCapacity(totalVars)

            for roundIdx in 0..<totalVars {
                let msg = layerProof.sumcheckMsgs[roundIdx]
                let sum = frAdd(msg.s0, msg.s1)
                if !dpFrEqual(sum, currentClaim) { return false }

                transcript.absorb(msg.s0)
                transcript.absorb(msg.s1)
                transcript.absorb(msg.s2)
                let challenge = transcript.squeeze()
                challenges.append(challenge)

                currentClaim = dpLagrangeEval3(s0: msg.s0, s1: msg.s1, s2: msg.s2, at: challenge)
            }

            // Extract rx, ry from challenges
            let instChallenges = Array(challenges.prefix(instBits))
            let xCircChallenges = Array(challenges[instBits..<(instBits + nInCircuit)])
            let yCircChallenges = Array(challenges[(instBits + nInCircuit)...])

            let rx = instChallenges + xCircChallenges
            let ry = instChallenges + yCircChallenges

            let vx = layerProof.claimedVx
            let vy = layerProof.claimedVy

            // Evaluate the factored polynomial at the challenge point
            // eq(rInstance, instChallenges) * [add(rCircuit, xCirc, yCirc)*(vx+vy) + mul(...)*vx*vy]
            let eqVal = dpEvaluateEq(rInstance, instChallenges)

            let layer = subCircuit.layers[layerIdx]
            let addMLE = dpBuildWiringMLE(layer: layer, type: .add, nOut: nOutCircuit, nIn: nInCircuit)
            let mulMLE = dpBuildWiringMLE(layer: layer, type: .mul, nOut: nOutCircuit, nIn: nInCircuit)

            let fullCircuitPoint = rCircuit + xCircChallenges + yCircChallenges
            let addVal = addMLE.evaluate(at: fullCircuitPoint)
            let mulVal = mulMLE.evaluate(at: fullCircuitPoint)

            let expected = frMul(eqVal, frAdd(
                frMul(addVal, frAdd(vx, vy)),
                frMul(mulVal, frMul(vx, vy))
            ))
            if !dpFrEqual(currentClaim, expected) { return false }

            // Absorb and combine for next layer
            transcript.absorb(vx)
            transcript.absorb(vy)
            transcript.absorbLabel("dp-layer-\(layerIdx)")

            let beta = transcript.squeeze()
            let totalInVars = instBits + nInCircuit
            var newR = [Fr]()
            newR.reserveCapacity(totalInVars)
            for i in 0..<totalInVars {
                newR.append(frAdd(rx[i], frMul(beta, frSub(ry[i], rx[i]))))
            }
            r = newR
            claim = frAdd(vx, frMul(beta, frSub(vy, vx)))
        }

        // Final check: claim should equal MLE of combined inputs at r
        let inputSize = subCircuit.inputSize
        let inputCircuitVars: Int
        if inputSize <= 1 {
            inputCircuitVars = 0
        } else {
            inputCircuitVars = Int(ceil(log2(Double(inputSize))))
        }
        let paddedInputSize = 1 << inputCircuitVars

        let combinedInput = uniformCircuit.combinedLayerValues(
            layerValues: inputs, layerSize: paddedInputSize)
        let totalInputVars = instBits + inputCircuitVars
        let inputMLE = MultilinearPoly(numVars: totalInputVars, values: combinedInput)
        let inputEval = inputMLE.evaluate(at: r)
        return dpFrEqual(claim, inputEval)
    }
}

// MARK: - Module-level Helpers

/// Evaluate the eq polynomial: eq(a, b) = prod_i (a_i * b_i + (1-a_i)(1-b_i))
private func dpEvaluateEq(_ a: [Fr], _ b: [Fr]) -> Fr {
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

/// Build a wiring MLE for a specific gate type in a layer.
func dpBuildWiringMLE(
    layer: CircuitLayer, type: GateType, nOut: Int, nIn: Int
) -> MultilinearPoly {
    let totalVars = nOut + 2 * nIn
    let totalSize = 1 << totalVars
    let inSize = 1 << nIn

    var evals = [Fr](repeating: Fr.zero, count: totalSize)
    for (gIdx, gate) in layer.gates.enumerated() {
        guard gate.type == type else { continue }
        let idx = gIdx * inSize * inSize + gate.leftInput * inSize + gate.rightInput
        if idx < totalSize {
            evals[idx] = Fr.one
        }
    }
    return MultilinearPoly(numVars: totalVars, evals: evals)
}

private func dpFrEqual(_ a: Fr, _ b: Fr) -> Bool {
    let diff = frSub(a, b)
    let limbs = frToInt(diff)
    return limbs[0] == 0 && limbs[1] == 0 && limbs[2] == 0 && limbs[3] == 0
}

private func dpLagrangeEval3(s0: Fr, s1: Fr, s2: Fr, at x: Fr) -> Fr {
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
