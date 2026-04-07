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

/// Round message for degree-3 sumcheck in the grand product.
/// The polynomial eq(r,x)*left(x)*right(x) is degree 3 in each variable,
/// requiring 4 evaluation points (at t=0,1,2,3) per round.
public struct GPSumcheckRoundMsg {
    public let s0: Fr  // polynomial evaluated at 0
    public let s1: Fr  // polynomial evaluated at 1
    public let s2: Fr  // polynomial evaluated at 2
    public let s3: Fr  // polynomial evaluated at 3

    public init(s0: Fr, s1: Fr, s2: Fr, s3: Fr) {
        self.s0 = s0
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
    }
}

/// Single layer proof within the grand product GKR.
public struct GrandProductLayerProof {
    /// Sumcheck round messages for this layer (degree-3 polynomial).
    public let sumcheckMsgs: [GPSumcheckRoundMsg]
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
            current.withUnsafeBytes { cBuf in
                next.withUnsafeMutableBytes { rBuf in
                    bn254_fr_batch_mul_adjacent(
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(half))
                }
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
            // The combined claim equals V(newR, 1-alpha), so append (1-alpha) to the random point
            rPoint = newR + [frSub(Fr.one, alpha)]
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
    ) -> (msgs: [GPSumcheckRoundMsg], vLeft: Fr, vRight: Fr, newR: [Fr]) {
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
        // f(x) = eq(r, x) * left(x) * right(x)  [degree 3 in each variable]
        var msgs = [GPSumcheckRoundMsg]()
        msgs.reserveCapacity(gateVars)
        var challenges = [Fr]()
        challenges.reserveCapacity(gateVars)

        var eqCurrent = eqVals
        var leftCurrent = leftVals
        var rightCurrent = rightVals
        var currentSize = gateCount

        for _ in 0..<gateVars {
            let half = currentSize / 2

            // Compute s(0), s(1), s(2), s(3) via fused C kernel
            var s0Limbs = [UInt64](repeating: 0, count: 4)
            var s1Limbs = [UInt64](repeating: 0, count: 4)
            var s2Limbs = [UInt64](repeating: 0, count: 4)
            var s3Limbs = [UInt64](repeating: 0, count: 4)

            eqCurrent.withUnsafeBytes { eqBuf in
                leftCurrent.withUnsafeBytes { lBuf in
                    rightCurrent.withUnsafeBytes { rBuf in
                        bn254_fr_gp_sumcheck_round(
                            eqBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            lBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(half), &s0Limbs, &s1Limbs, &s2Limbs, &s3Limbs)
                    }
                }
            }
            let s0 = Fr.from64(s0Limbs)
            let s1 = Fr.from64(s1Limbs)
            let s2 = Fr.from64(s2Limbs)
            let s3 = Fr.from64(s3Limbs)

            msgs.append(GPSumcheckRoundMsg(s0: s0, s1: s1, s2: s2, s3: s3))

            transcript.absorb(s0)
            transcript.absorb(s1)
            transcript.absorb(s2)
            transcript.absorb(s3)
            let challenge = transcript.squeeze()
            challenges.append(challenge)

            // Fold the bookkeeping tables: newX[j] = X[j] + c*(X[j+half] - X[j])
            var newEq = [Fr](repeating: Fr.zero, count: half)
            var newLeft = [Fr](repeating: Fr.zero, count: half)
            var newRight = [Fr](repeating: Fr.zero, count: half)

            withUnsafeBytes(of: challenge) { cPtr in
                let cP = cPtr.baseAddress!.assumingMemoryBound(to: UInt64.self)
                eqCurrent.withUnsafeBytes { eqBuf in
                    newEq.withUnsafeMutableBytes { rBuf in
                        bn254_fr_sumcheck_reduce(
                            eqBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            cP, rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(half))
                    }
                }
                leftCurrent.withUnsafeBytes { lBuf in
                    newLeft.withUnsafeMutableBytes { rBuf in
                        bn254_fr_sumcheck_reduce(
                            lBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            cP, rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(half))
                    }
                }
                rightCurrent.withUnsafeBytes { rBuf in
                    newRight.withUnsafeMutableBytes { oBuf in
                        bn254_fr_sumcheck_reduce(
                            rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            cP, oBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(half))
                    }
                }
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
                transcript.absorb(msg.s3)
                let challenge = transcript.squeeze()
                challenges.append(challenge)

                currentClaim = gpLagrangeEval4(s0: msg.s0, s1: msg.s1, s2: msg.s2, s3: msg.s3, at: challenge)
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
            // The combined claim equals V(challenges, 1-alpha), so append (1-alpha) to the random point
            rPoint = challenges + [frSub(Fr.one, alpha)]
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

        // The last layer's vLeft/vRight are evaluations of the input halves.
        // We verify they match the actual MLE evaluations of the input layer.
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

/// Precomputed inverse of 6 in Montgomery form (for degree-3 Lagrange).
private let gpInv6: Fr = frInverse(frAdd(frAdd(Fr.one, Fr.one), frAdd(frAdd(Fr.one, Fr.one), frAdd(Fr.one, Fr.one))))

/// Evaluate the degree-3 polynomial passing through (0, s0), (1, s1), (2, s2), (3, s3) at point x.
///
/// Uses Lagrange basis:
///   l0(x) = (x-1)(x-2)(x-3) / (-6)
///   l1(x) = x(x-2)(x-3) / 2
///   l2(x) = x(x-1)(x-3) / (-2)
///   l3(x) = x(x-1)(x-2) / 6
private func gpLagrangeEval4(s0: Fr, s1: Fr, s2: Fr, s3: Fr, at x: Fr) -> Fr {
    let two = frAdd(Fr.one, Fr.one)
    let three = frAdd(two, Fr.one)
    let negOne = frSub(Fr.zero, Fr.one)

    let xm1 = frSub(x, Fr.one)
    let xm2 = frSub(x, two)
    let xm3 = frSub(x, three)

    // l0 = (x-1)(x-2)(x-3) / (0-1)(0-2)(0-3) = (x-1)(x-2)(x-3) / (-6)
    let l0 = frMul(frMul(frMul(xm1, xm2), xm3), frMul(negOne, gpInv6))
    // l1 = x(x-2)(x-3) / (1-0)(1-2)(1-3) = x(x-2)(x-3) / (1*-1*-2) = x(x-2)(x-3) / 2
    let l1 = frMul(frMul(frMul(x, xm2), xm3), gpInv2)
    // l2 = x(x-1)(x-3) / (2-0)(2-1)(2-3) = x(x-1)(x-3) / (2*1*-1) = x(x-1)(x-3) / (-2)
    let l2 = frMul(frMul(frMul(x, xm1), xm3), frMul(negOne, gpInv2))
    // l3 = x(x-1)(x-2) / (3-0)(3-1)(3-2) = x(x-1)(x-2) / (3*2*1) = x(x-1)(x-2) / 6
    let l3 = frMul(frMul(frMul(x, xm1), xm2), gpInv6)

    return frAdd(frAdd(frMul(s0, l0), frMul(s1, l1)),
                 frAdd(frMul(s2, l2), frMul(s3, l3)))
}
