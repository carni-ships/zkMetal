// DataParallelProver — Proves N instances of the same circuit efficiently
//
// Batched GKR: runs sumcheck once with a combined MLE across all instances.
// The wiring predicates are shared (computed once from the template) and amortized.
//
// Protocol:
// 1. Prover evaluates all N instances, absorbs combined output into transcript
// 2. Verifier sends random challenge r = (r_inst, r_circ) for output layer
// 3. For each layer (output to input):
//    a. Run sumcheck on the data-parallel GKR equation:
//       sum_{x,y} eq(r_inst, x_inst) * eq(x_inst, y_inst) *
//                  [add(r_circ, x_circ, y_circ)*(V(x)+V(y)) +
//                   mul(r_circ, x_circ, y_circ)*V(x)*V(y)]
//    b. The factored structure means wiring predicates are evaluated once (not N times)
//    c. Prover sends V(rx), V(ry) at the sumcheck output points
// 4. Final claim checked against combined input MLE
//
// Cost: O(d * N * |C_layer|) total, vs O(d * N * |C_layer| * log(N*|C|)) for flat GKR

import Foundation

// MARK: - Proof Types

/// Proof for one layer of the data-parallel GKR protocol.
public struct DataParallelLayerProof {
    public let sumcheckMsgs: [SumcheckRoundMsg]
    public let claimedVx: Fr   // V_{prev}(rx)
    public let claimedVy: Fr   // V_{prev}(ry)

    public init(sumcheckMsgs: [SumcheckRoundMsg], claimedVx: Fr, claimedVy: Fr) {
        self.sumcheckMsgs = sumcheckMsgs
        self.claimedVx = claimedVx
        self.claimedVy = claimedVy
    }
}

/// Complete data-parallel proof for all layers.
public struct DataParallelGKRProof {
    public let layerProofs: [DataParallelLayerProof]
    public let allOutputs: [[Fr]]   // per-instance outputs for the verifier

    public init(layerProofs: [DataParallelLayerProof], allOutputs: [[Fr]]) {
        self.layerProofs = layerProofs
        self.allOutputs = allOutputs
    }
}

// MARK: - DataParallelProver

public class DataParallelProver {
    public static let version = Versions.dataParallel

    /// Cached inverse of 2 for Lagrange interpolation.
    private let inv2: Fr = frInverse(frAdd(Fr.one, Fr.one))

    public init() {}

    /// Prove all N instances of the data-parallel circuit.
    /// The circuit must have been evaluated (evaluateAll) before calling this.
    public func prove(circuit: inout DataParallelCircuit, transcript: Transcript) -> DataParallelGKRProof {
        // Evaluate if not already done
        if circuit.allLayerValues == nil {
            circuit.evaluateAll()
        }

        let d = circuit.template.depth
        let instBits = circuit.instanceBits
        let padN = circuit.paddedInstances

        // Get per-instance outputs
        let allOutputs = circuit.instanceOutputs!

        // Build combined output and absorb
        let outputPadSize = circuit.template.layers[d - 1].paddedSize
        let combinedOutput = circuit.combinedOutputValues()

        for v in combinedOutput { transcript.absorb(v) }
        transcript.absorbLabel("dp-gkr-init")

        // Initial random point: (r_inst, r_circ) for the output layer
        let outputCircuitVars = circuit.outputVarsForLayer(d - 1)
        let totalOutputVars = instBits + outputCircuitVars
        var r = transcript.squeezeN(totalOutputVars)

        let outputMLE = MultilinearPoly(numVars: totalOutputVars, values: combinedOutput)
        var claim = outputMLE.evaluate(at: r)

        var layerProofs = [DataParallelLayerProof]()
        layerProofs.reserveCapacity(d)

        // Process layers from output to input
        for layerIdx in stride(from: d - 1, through: 0, by: -1) {
            let nOutCircuit = circuit.outputVarsForLayer(layerIdx)
            let nInCircuit = circuit.inputVarsForLayer(layerIdx)

            let rInstance = Array(r.prefix(instBits))
            let rCircuit = Array(r.suffix(nOutCircuit))

            // Build combined previous-layer values
            let combinedPrev = circuit.combinedValues(layerIndex: layerIdx)

            // Get shared wiring MLEs (computed once, cached)
            let addMLE = circuit.addWiringMLE(layer: layerIdx)
            let mulMLE = circuit.mulWiringMLE(layer: layerIdx)

            // Run sumcheck with factored wiring
            let (msgs, rx, ry) = sumcheckLayer(
                rInstance: rInstance, rCircuit: rCircuit,
                addMLE: addMLE, mulMLE: mulMLE,
                layer: circuit.template.layers[layerIdx],
                combinedPrev: combinedPrev,
                instBits: instBits, nOutCircuit: nOutCircuit,
                nInCircuit: nInCircuit, padN: padN,
                transcript: transcript
            )

            // Evaluate previous-layer MLE at sumcheck output points
            let totalPrevVars = instBits + nInCircuit
            let prevMLE = MultilinearPoly(numVars: totalPrevVars, values: combinedPrev)
            let vx = prevMLE.evaluate(at: rx)
            let vy = prevMLE.evaluate(at: ry)

            transcript.absorb(vx)
            transcript.absorb(vy)
            transcript.absorbLabel("dp-layer-\(layerIdx)")

            layerProofs.append(DataParallelLayerProof(
                sumcheckMsgs: msgs, claimedVx: vx, claimedVy: vy))

            // Combine claims for next layer: r = rx + beta * (ry - rx)
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

        return DataParallelGKRProof(layerProofs: layerProofs, allOutputs: allOutputs)
    }

    // MARK: - Sumcheck for One Layer

    /// Run sumcheck for one GKR layer with factored wiring predicates.
    ///
    /// The sumcheck polynomial is:
    ///   g(x_inst, x_circ, y_inst, y_circ) =
    ///     eq(r_inst, x_inst) * eq(x_inst, y_inst) *
    ///     [add_fixed(x_circ, y_circ) * (V(x) + V(y)) +
    ///      mul_fixed(x_circ, y_circ) * V(x) * V(y)]
    ///
    /// where add_fixed/mul_fixed have r_circ already fixed (computed once from template).
    ///
    /// Variable order: x_inst, x_circ, y_inst, y_circ
    private func sumcheckLayer(
        rInstance: [Fr], rCircuit: [Fr],
        addMLE: MultilinearPoly, mulMLE: MultilinearPoly,
        layer: CircuitLayer,
        combinedPrev: [Fr],
        instBits: Int, nOutCircuit: Int, nInCircuit: Int, padN: Int,
        transcript: Transcript
    ) -> (msgs: [SumcheckRoundMsg], rx: [Fr], ry: [Fr]) {

        let circuitInSize = 1 << nInCircuit

        // Fix output circuit variables in wiring MLEs (shared computation)
        var addFixed = addMLE
        for i in 0..<nOutCircuit { addFixed = addFixed.fixVariable(rCircuit[i]) }
        var mulFixed = mulMLE
        for i in 0..<nOutCircuit { mulFixed = mulFixed.fixVariable(rCircuit[i]) }

        // Precompute eq(r_inst, *) over the boolean hypercube
        let eqInst = MultilinearPoly.eqPoly(point: rInstance)

        // Build the sumcheck table
        // Variables: x_inst (instBits), x_circ (nInCircuit), y_inst (instBits), y_circ (nInCircuit)
        let totalVars = 2 * instBits + 2 * nInCircuit
        let xSize = padN * circuitInSize
        let ySize = padN * circuitInSize
        let totalTableSize = xSize * ySize

        var table = [Fr](repeating: Fr.zero, count: totalTableSize)
        let addEvals = addFixed.evals
        let mulEvals = mulFixed.evals

        // Exploit structure: eq(x_inst, y_inst) = 1 only when x_inst == y_inst on {0,1}^instBits
        // So we only iterate over the diagonal (xi == yi), saving factor of padN
        for xi in 0..<padN {
            let eqR = eqInst[xi]
            if eqR.isZero { continue }

            let yi = xi  // eq(xi, yi) = 1 only on diagonal
            for xc in 0..<circuitInSize {
                let xIdx = xi * circuitInSize + xc
                let vxVal = xIdx < combinedPrev.count ? combinedPrev[xIdx] : Fr.zero

                for yc in 0..<circuitInSize {
                    let yIdx = yi * circuitInSize + yc
                    let vyVal = yIdx < combinedPrev.count ? combinedPrev[yIdx] : Fr.zero

                    let circIdx = xc * circuitInSize + yc
                    let aVal = circIdx < addEvals.count ? addEvals[circIdx] : Fr.zero
                    let mVal = circIdx < mulEvals.count ? mulEvals[circIdx] : Fr.zero

                    // g_circuit = add_fixed(xc,yc) * (Vx + Vy) + mul_fixed(xc,yc) * Vx * Vy
                    let gCircuit = frAdd(
                        frMul(aVal, frAdd(vxVal, vyVal)),
                        frMul(mVal, frMul(vxVal, vyVal)))

                    // Full value: eq(r_inst, xi) * eq(xi, yi) * g_circuit
                    // eq(xi, yi) = 1 on diagonal
                    let tableIdx = xIdx * ySize + yIdx
                    if tableIdx < totalTableSize {
                        table[tableIdx] = frMul(eqR, gCircuit)
                    }
                }
            }
        }

        // Standard sumcheck reduction on the table
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
                // s2 = f(2) = 2*f1 - f0 (degree-2 extrapolation)
                s2 = frAdd(s2, frSub(frAdd(f1, f1), f0))
            }

            let msg = SumcheckRoundMsg(s0: s0, s1: s1, s2: s2)
            msgs.append(msg)

            transcript.absorb(s0)
            transcript.absorb(s1)
            transcript.absorb(s2)
            let challenge = transcript.squeeze()
            challenges.append(challenge)

            // Fold: newTable[j] = (1-c)*curTable[j] + c*curTable[j + halfSize]
            let oneMinusC = frSub(Fr.one, challenge)
            var newTable = [Fr](repeating: Fr.zero, count: halfSize)
            for j in 0..<halfSize {
                newTable[j] = frAdd(frMul(oneMinusC, curTable[j]),
                                    frMul(challenge, curTable[j + halfSize]))
            }
            curTable = newTable
        }

        // Extract rx = (x_inst, x_circ) and ry = (y_inst, y_circ) from challenges
        let xiChallenges = Array(challenges[0..<instBits])
        let xcChallenges = Array(challenges[instBits..<(instBits + nInCircuit)])
        let yiChallenges = Array(challenges[(instBits + nInCircuit)..<(2 * instBits + nInCircuit)])
        let ycChallenges = Array(challenges[(2 * instBits + nInCircuit)...])

        let rx = xiChallenges + xcChallenges
        let ry = yiChallenges + ycChallenges

        return (msgs, rx, ry)
    }
}
