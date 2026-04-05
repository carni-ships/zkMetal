// Grand Product via GKR Protocol
//
// Proves that the product of a vector of field elements equals a claimed value,
// using the GKR protocol on a binary multiplication tree circuit.
//
// Circuit structure (2-layer for n elements):
//   Layer 0 (input):  v[0], v[1], ..., v[n-1]
//   Layer 1:          v[0]*v[1], v[2]*v[3], ...
//   ...
//   Layer log(n):     final product
//
// Each layer halves the number of values by multiplying adjacent pairs.
// GKR reduces the claim about the output to claims about the input layer,
// using sumcheck at each intermediate layer.
//
// Batch grand product: proves multiple independent products simultaneously
// by sharing the same GKR structure with separate value polynomials.
//
// References: Thaler (2013), Setty (2020), Lasso/Jolt (2024)

import Foundation
import NeonFieldOps

// MARK: - Grand Product Proof Types

/// Proof that the product of a vector equals a claimed value.
public struct GrandProductProof {
    /// GKR layer proofs for each level of the multiplication tree.
    public let layerProofs: [GrandProductLayerProof]
    /// The claimed product (output of the tree).
    public let claimedProduct: Fr
}

/// Single layer proof within the grand product GKR.
public struct GrandProductLayerProof {
    /// Sumcheck round messages for this layer.
    public let sumcheckMsgs: [SumcheckRoundMsg]
    /// Claimed evaluations of the previous layer's MLE at the random points.
    public let claimedVLeft: Fr
    public let claimedVRight: Fr
}

/// Batch grand product proof: multiple independent grand products proved together.
public struct BatchGrandProductProof {
    /// Individual grand product proofs, one per product.
    public let proofs: [GrandProductProof]
    /// Number of products in the batch.
    public let batchSize: Int
}

// MARK: - Grand Product Engine

/// Engine for proving and verifying grand products using the GKR protocol.
///
/// The multiplication tree is structured as a layered circuit where each layer
/// multiplies adjacent pairs. For n input elements (padded to power of 2):
///   - depth = log2(n) layers
///   - layer i has n / 2^(i+1) multiplication gates
///   - layer i gate j computes: prev[2j] * prev[2j+1]
///
/// The GKR protocol runs sumcheck at each layer to reduce the output claim
/// to input claims, avoiding the need to commit to intermediate values.
public class GrandProductEngine {
    public static let version = Versions.gkr

    // MARK: - Prover

    /// Prove that the product of `values` equals the claimed product.
    ///
    /// - Parameters:
    ///   - values: Input field elements (will be padded to power of 2 with Fr.one)
    ///   - transcript: Fiat-Shamir transcript for challenge generation
    /// - Returns: Grand product proof
    public static func prove(values: [Fr], transcript: Transcript) -> GrandProductProof {
        let n = values.count
        precondition(n > 0, "Cannot compute grand product of empty vector")

        // Pad to power of 2 with multiplicative identity (1)
        let logN = ceilLog2(n)
        let paddedN = 1 << logN
        var padded = [Fr](repeating: Fr.one, count: paddedN)
        for i in 0..<n { padded[i] = values[i] }

        // Build the multiplication tree bottom-up
        let depth = logN
        var layers = [[Fr]]()
        layers.reserveCapacity(depth + 1)
        layers.append(padded)

        var current = padded
        for _ in 0..<depth {
            let half = current.count / 2
            var next = [Fr](repeating: Fr.zero, count: half)
            for j in 0..<half {
                next[j] = frMul(current[2 * j], current[2 * j + 1])
            }
            layers.append(next)
            current = next
        }

        // The final product
        let product = layers[depth][0]

        // Absorb the product
        transcript.absorb(product)
        transcript.absorbLabel("grand-product-init")

        // GKR proving: from output (product) to input (values)
        // The output layer has 1 element, so 0 variables. The verifier's initial
        // random point is empty, and the claim is the product itself.
        var layerProofs = [GrandProductLayerProof]()

        // For each layer from output toward input, run sumcheck
        // Layer i has 2^i elements, layer i+1 has 2^(i+1) elements
        // Gate j in layer i: output[j] = input[2j] * input[2j+1]
        var rPoint = [Fr]()  // random point from previous layer
        var claim = product

        for layerIdx in stride(from: depth - 1, through: 0, by: -1) {
            // This layer has 2^(depth - layerIdx - 1) gates
            // Input to this layer is layers[layerIdx] with 2^(depth - layerIdx) elements
            let inputSize = layers[layerIdx].count
            let inputVars = ceilLog2(inputSize)

            let (msgs, vLeft, vRight, newR) = proveMulLayer(
                claim: claim,
                rPoint: rPoint,
                inputValues: layers[layerIdx],
                inputVars: inputVars,
                transcript: transcript,
                layerIdx: layerIdx
            )

            transcript.absorb(vLeft)
            transcript.absorb(vRight)
            transcript.absorbLabel("grand-product-layer-\(layerIdx)")

            layerProofs.append(GrandProductLayerProof(
                sumcheckMsgs: msgs,
                claimedVLeft: vLeft,
                claimedVRight: vRight
            ))

            // Combine claims for next layer using random linear combination
            let alpha = transcript.squeeze()
            claim = frAdd(frMul(alpha, vLeft), frMul(frSub(Fr.one, alpha), vRight))
            rPoint = newR
        }

        return GrandProductProof(layerProofs: layerProofs, claimedProduct: product)
    }

    /// Prove one multiplication layer of the grand product tree.
    ///
    /// Each gate j computes output[j] = left[2j] * left[2j+1].
    /// The sumcheck proves:
    ///   claim = sum_{x in {0,1}^v} eq(r, x) * left(x, 0) * left(x, 1)
    /// where left(x, b) indexes into the input layer with the last bit selecting
    /// left (b=0) or right (b=1) child.
    private static func proveMulLayer(
        claim: Fr,
        rPoint: [Fr],
        inputValues: [Fr],
        inputVars: Int,
        transcript: Transcript,
        layerIdx: Int
    ) -> (msgs: [SumcheckRoundMsg], vLeft: Fr, vRight: Fr, newR: [Fr]) {
        let gateVars = inputVars - 1  // number of variables to index gates
        let gateCount = 1 << gateVars

        // Build eq polynomial for the output point
        var eqVals: [Fr]
        if rPoint.isEmpty {
            // First layer (single output): eq is just [1]
            eqVals = [Fr.one]
        } else {
            eqVals = MultilinearPoly.eqPoly(point: rPoint)
        }
        // Pad eq to gateCount
        while eqVals.count < gateCount {
            eqVals.append(Fr.zero)
        }

        // Split input into left children (even indices) and right children (odd indices)
        var leftVals = [Fr](repeating: Fr.zero, count: gateCount)
        var rightVals = [Fr](repeating: Fr.zero, count: gateCount)
        for j in 0..<min(gateCount, inputValues.count / 2) {
            leftVals[j] = inputValues[2 * j]
            rightVals[j] = inputValues[2 * j + 1]
        }

        // Sumcheck over gateVars variables:
        // f(x) = eq(r, x) * left(x) * right(x)
        var msgs = [SumcheckRoundMsg]()
        msgs.reserveCapacity(gateVars)
        var challenges = [Fr]()
        challenges.reserveCapacity(gateVars)

        var eqCurrent = eqVals
        var leftCurrent = leftVals
        var rightCurrent = rightVals
        var currentSize = gateCount

        for _ in 0..<gateVars {
            let half = currentSize / 2

            // Compute s(0), s(1), s(2) for the current variable
            var s0 = Fr.zero
            var s1 = Fr.zero
            var s2 = Fr.zero

            for j in 0..<half {
                // At t=0: use values at index j (first half)
                let eq0 = eqCurrent[j]
                let l0 = leftCurrent[j]
                let r0 = rightCurrent[j]
                let prod0 = frMul(eq0, frMul(l0, r0))
                s0 = frAdd(s0, prod0)

                // At t=1: use values at index j+half (second half)
                let eq1 = eqCurrent[j + half]
                let l1 = leftCurrent[j + half]
                let r1 = rightCurrent[j + half]
                let prod1 = frMul(eq1, frMul(l1, r1))
                s1 = frAdd(s1, prod1)

                // At t=2: linear extrapolation
                // f(2) = 2*f(1) - f(0) for linear functions,
                // but eq, left, right are each linear in this variable, so f is degree 3.
                // eq(2) = 2*eq1 - eq0, left(2) = 2*l1 - l0, right(2) = 2*r1 - r0
                let eq2 = frSub(frAdd(eq1, eq1), eq0)
                let l2 = frSub(frAdd(l1, l1), l0)
                let r2 = frSub(frAdd(r1, r1), r0)
                let prod2 = frMul(eq2, frMul(l2, r2))
                s2 = frAdd(s2, prod2)
            }

            msgs.append(SumcheckRoundMsg(s0: s0, s1: s1, s2: s2))

            transcript.absorb(s0)
            transcript.absorb(s1)
            transcript.absorb(s2)
            let challenge = transcript.squeeze()
            challenges.append(challenge)

            // Fold the bookkeeping tables
            let oneMinusC = frSub(Fr.one, challenge)
            var newEq = [Fr](repeating: Fr.zero, count: half)
            var newLeft = [Fr](repeating: Fr.zero, count: half)
            var newRight = [Fr](repeating: Fr.zero, count: half)

            for j in 0..<half {
                newEq[j] = frAdd(frMul(oneMinusC, eqCurrent[j]),
                                 frMul(challenge, eqCurrent[j + half]))
                newLeft[j] = frAdd(frMul(oneMinusC, leftCurrent[j]),
                                   frMul(challenge, leftCurrent[j + half]))
                newRight[j] = frAdd(frMul(oneMinusC, rightCurrent[j]),
                                    frMul(challenge, rightCurrent[j + half]))
            }

            eqCurrent = newEq
            leftCurrent = newLeft
            rightCurrent = newRight
            currentSize = half
        }

        // After sumcheck, evaluate the input MLE at the resulting point
        let vLeft = cMleEvalGP(evals: leftVals, point: challenges, numVars: gateVars)
        let vRight = cMleEvalGP(evals: rightVals, point: challenges, numVars: gateVars)

        return (msgs, vLeft, vRight, challenges)
    }

    // MARK: - Verifier

    /// Verify a grand product proof.
    ///
    /// - Parameters:
    ///   - values: The input elements whose product is claimed
    ///   - proof: The grand product proof
    ///   - transcript: Fiat-Shamir transcript (must match prover's)
    /// - Returns: true if the proof is valid
    public static func verify(values: [Fr], proof: GrandProductProof, transcript: Transcript) -> Bool {
        let n = values.count
        guard n > 0 else { return false }

        let logN = ceilLog2(n)
        let paddedN = 1 << logN
        let depth = logN

        guard proof.layerProofs.count == depth else { return false }

        // Absorb the claimed product
        transcript.absorb(proof.claimedProduct)
        transcript.absorbLabel("grand-product-init")

        // Verify: compute the actual product and check it matches
        var actualProduct = Fr.one
        for v in values { actualProduct = frMul(actualProduct, v) }
        // Pad with ones
        // (ones don't change the product, so actualProduct is the correct full product)

        guard gpFrEqual(actualProduct, proof.claimedProduct) else { return false }

        var rPoint = [Fr]()
        var claim = proof.claimedProduct

        for (proofIdx, layerProof) in proof.layerProofs.enumerated() {
            let layerIdx = depth - 1 - proofIdx
            let inputSize = 1 << (depth - layerIdx)
            let inputVars = ceilLog2(inputSize)
            let gateVars = inputVars - 1

            guard layerProof.sumcheckMsgs.count == gateVars else { return false }

            // Verify sumcheck
            var currentClaim = claim
            var challenges = [Fr]()
            challenges.reserveCapacity(gateVars)

            for msg in layerProof.sumcheckMsgs {
                let sum = frAdd(msg.s0, msg.s1)
                guard gpFrEqual(sum, currentClaim) else { return false }

                transcript.absorb(msg.s0)
                transcript.absorb(msg.s1)
                transcript.absorb(msg.s2)
                let challenge = transcript.squeeze()
                challenges.append(challenge)

                currentClaim = gpLagrangeEval3(s0: msg.s0, s1: msg.s1, s2: msg.s2, at: challenge)
            }

            // Check final claim: eq(r, challenges) * vLeft * vRight should equal currentClaim
            let vLeft = layerProof.claimedVLeft
            let vRight = layerProof.claimedVRight

            var eqVal: Fr
            if rPoint.isEmpty {
                eqVal = Fr.one
            } else {
                eqVal = evalEqAtPoint(rPoint, challenges)
            }

            let expected = frMul(eqVal, frMul(vLeft, vRight))
            guard gpFrEqual(currentClaim, expected) else { return false }

            transcript.absorb(vLeft)
            transcript.absorb(vRight)
            transcript.absorbLabel("grand-product-layer-\(layerIdx)")

            let alpha = transcript.squeeze()
            claim = frAdd(frMul(alpha, vLeft), frMul(frSub(Fr.one, alpha), vRight))
            rPoint = challenges
        }

        // Final check: the claim should match the input MLE evaluation
        var padded = [Fr](repeating: Fr.one, count: paddedN)
        for i in 0..<n { padded[i] = values[i] }

        // Split into left (even) and right (odd) for the bottom layer
        let bottomGateVars = ceilLog2(paddedN) - 1
        let bottomGateCount = 1 << bottomGateVars
        var leftVals = [Fr](repeating: Fr.zero, count: bottomGateCount)
        var rightVals = [Fr](repeating: Fr.zero, count: bottomGateCount)
        for j in 0..<bottomGateCount {
            leftVals[j] = padded[2 * j]
            rightVals[j] = padded[2 * j + 1]
        }

        let evalLeft = cMleEvalGP(evals: leftVals, point: rPoint, numVars: bottomGateVars)
        let evalRight = cMleEvalGP(evals: rightVals, point: rPoint, numVars: bottomGateVars)

        // Reconstruct what the claim should be from the last alpha
        // claim = alpha * evalLeft + (1 - alpha) * evalRight
        let expectedFinal = claim
        let lastAlpha = transcript.squeeze()  // This was already squeezed; we recompute
        // Actually, we already computed claim from the last layer's alpha.
        // We just need to verify claim matches alpha * evalLeft + (1-alpha) * evalRight
        // But claim was set by the prover's vLeft/vRight. We need to check those match input.
        // The last layer's vLeft/vRight are evaluations of the input halves.
        let lastProof = proof.layerProofs.last!
        guard gpFrEqual(lastProof.claimedVLeft, evalLeft) else { return false }
        guard gpFrEqual(lastProof.claimedVRight, evalRight) else { return false }

        return true
    }

    // MARK: - Batch Grand Product

    /// Prove multiple independent grand products in a batch.
    ///
    /// Each product is proved independently but shares the same transcript
    /// for efficiency. In a full implementation, the sumcheck rounds would
    /// be batched across products using random linear combinations.
    ///
    /// - Parameters:
    ///   - valueSets: Array of value vectors, one per product
    ///   - transcript: Shared Fiat-Shamir transcript
    /// - Returns: Batch proof containing all individual proofs
    public static func proveBatch(
        valueSets: [[Fr]],
        transcript: Transcript
    ) -> BatchGrandProductProof {
        let batchSize = valueSets.count
        precondition(batchSize > 0, "Empty batch")

        transcript.absorbLabel("batch-grand-product")
        transcript.absorb(frFromInt(UInt64(batchSize)))

        var proofs = [GrandProductProof]()
        proofs.reserveCapacity(batchSize)

        for values in valueSets {
            let proof = prove(values: values, transcript: transcript)
            proofs.append(proof)
        }

        return BatchGrandProductProof(proofs: proofs, batchSize: batchSize)
    }

    /// Verify a batch of grand product proofs.
    public static func verifyBatch(
        valueSets: [[Fr]],
        proof: BatchGrandProductProof,
        transcript: Transcript
    ) -> Bool {
        guard valueSets.count == proof.batchSize else { return false }
        guard proof.proofs.count == proof.batchSize else { return false }

        transcript.absorbLabel("batch-grand-product")
        transcript.absorb(frFromInt(UInt64(proof.batchSize)))

        for (values, subproof) in zip(valueSets, proof.proofs) {
            guard verify(values: values, proof: subproof, transcript: transcript) else {
                return false
            }
        }
        return true
    }
}

// MARK: - Helpers

/// Ceiling log2 of n. Returns 0 for n <= 1.
private func ceilLog2(_ n: Int) -> Int {
    guard n > 1 else { return n <= 0 ? 0 : 0 }
    return Int(ceil(log2(Double(n))))
}

/// Compare two Fr elements.
private func gpFrEqual(_ a: Fr, _ b: Fr) -> Bool {
    return a.v.0 == b.v.0 && a.v.1 == b.v.1 && a.v.2 == b.v.2 && a.v.3 == b.v.3 &&
           a.v.4 == b.v.4 && a.v.5 == b.v.5 && a.v.6 == b.v.6 && a.v.7 == b.v.7
}

/// MLE evaluation using iterative folding.
private func cMleEvalGP(evals: [Fr], point: [Fr], numVars: Int) -> Fr {
    guard numVars > 0 else { return evals.isEmpty ? Fr.zero : evals[0] }

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

/// Evaluate the eq polynomial at a single point: eq(a, b) = prod_i (a_i * b_i + (1-a_i)*(1-b_i))
private func evalEqAtPoint(_ a: [Fr], _ b: [Fr]) -> Fr {
    precondition(a.count == b.count)
    var result = Fr.one
    for i in 0..<a.count {
        let ai = a[i]
        let bi = b[i]
        let term = frAdd(frMul(ai, bi),
                         frMul(frSub(Fr.one, ai), frSub(Fr.one, bi)))
        result = frMul(result, term)
    }
    return result
}

/// Precomputed inverse of 2 in Montgomery form.
private let gpInv2: Fr = frInverse(frAdd(Fr.one, Fr.one))

/// Evaluate the degree-2 polynomial passing through (0, s0), (1, s1), (2, s2) at point x.
private func gpLagrangeEval3(s0: Fr, s1: Fr, s2: Fr, at x: Fr) -> Fr {
    let xm1 = frSub(x, Fr.one)
    let xm2 = frSub(x, frAdd(Fr.one, Fr.one))
    let negOne = frSub(Fr.zero, Fr.one)

    let l0 = frMul(frMul(xm1, xm2), gpInv2)
    let l1 = frMul(frMul(x, xm2), negOne)
    let l2 = frMul(frMul(x, xm1), gpInv2)

    return frAdd(frAdd(frMul(s0, l0), frMul(s1, l1)), frMul(s2, l2))
}
