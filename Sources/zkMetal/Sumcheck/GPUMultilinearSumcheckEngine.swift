// GPU-accelerated multilinear sumcheck protocol engine.
//
// Provides three sumcheck protocol variants:
//   1. Single-polynomial sumcheck: prove sum of f(x) over boolean hypercube
//   2. Product sumcheck: prove sum of f(x)*g(x) over boolean hypercube
//   3. Batch sumcheck: combine multiple sumcheck instances with random weights
//
// Each variant produces round-by-round prover messages (univariate polynomials),
// uses Fiat-Shamir for challenge generation, and includes a verifier.
//
// Delegates to GPUSumcheckEngine for Metal-accelerated table operations on
// large tables, with CPU fallback for small ones.

import Foundation
import Metal

// MARK: - Multilinear Round Polynomial

/// A univariate polynomial of degree <= d representing one sumcheck round message.
/// For single-poly sumcheck: degree 1 (two coefficients).
/// For product sumcheck: degree 2 (three coefficients).
/// Stored as evaluations at {0, 1, 2} for uniform representation.
public struct MultilinearRoundPoly: Equatable {
    public let evals: [Fr]  // evaluations at 0, 1, 2, ... up to degree

    /// The degree of this round polynomial.
    public var degree: Int { evals.count - 1 }

    /// Create from evaluations at 0, 1, ..., d.
    public init(evals: [Fr]) {
        precondition(evals.count >= 2, "Round poly needs at least 2 evaluations")
        self.evals = evals
    }

    /// Evaluate at an arbitrary point using Lagrange interpolation.
    public func evaluate(at r: Fr) -> Fr {
        let d = evals.count
        if d == 2 {
            // Linear: p(r) = evals[0] + r * (evals[1] - evals[0])
            let diff = frSub(evals[1], evals[0])
            return frAdd(evals[0], frMul(r, diff))
        }
        if d == 3 {
            // Quadratic interpolation through (0, e0), (1, e1), (2, e2)
            // Using Lagrange basis:
            // L0(r) = (r-1)(r-2)/2,  L1(r) = r(r-2)/(-1),  L2(r) = r(r-1)/2
            let e0 = evals[0], e1 = evals[1], e2 = evals[2]
            let rMinus1 = frSub(r, Fr.one)
            let two = frAdd(Fr.one, Fr.one)
            let rMinus2 = frSub(r, two)
            let twoInv = frInverse(two)

            // L0 = (r-1)(r-2) / 2
            let l0 = frMul(frMul(rMinus1, rMinus2), twoInv)
            // L1 = r(r-2) / (-1) = -r(r-2)
            let l1 = frNeg(frMul(r, rMinus2))
            // L2 = r(r-1) / 2
            let l2 = frMul(frMul(r, rMinus1), twoInv)

            return frAdd(frAdd(frMul(e0, l0), frMul(e1, l1)), frMul(e2, l2))
        }
        // General Lagrange interpolation for higher degrees
        return lagrangeInterpolate(points: evals, at: r)
    }

    /// p(0)
    public var atZero: Fr { evals[0] }

    /// p(1)
    public var atOne: Fr { evals[1] }

    public static func == (lhs: MultilinearRoundPoly, rhs: MultilinearRoundPoly) -> Bool {
        guard lhs.evals.count == rhs.evals.count else { return false }
        for i in 0..<lhs.evals.count {
            if !frEqual(lhs.evals[i], rhs.evals[i]) { return false }
        }
        return true
    }
}

/// General Lagrange interpolation: given evaluations at 0, 1, ..., n-1,
/// evaluate the interpolating polynomial at point r.
private func lagrangeInterpolate(points: [Fr], at r: Fr) -> Fr {
    let n = points.count
    var result = Fr.zero
    for i in 0..<n {
        var basis = Fr.one
        let xi = frFromInt(UInt64(i))
        for j in 0..<n {
            if i == j { continue }
            let xj = frFromInt(UInt64(j))
            // basis *= (r - xj) / (xi - xj)
            let num = frSub(r, xj)
            let den = frSub(xi, xj)
            basis = frMul(basis, frMul(num, frInverse(den)))
        }
        result = frAdd(result, frMul(points[i], basis))
    }
    return result
}

// MARK: - Multilinear Sumcheck Proof

/// Complete proof from a multilinear sumcheck protocol.
public struct MultilinearSumcheckProof {
    /// Round messages: one univariate polynomial per variable.
    public let roundPolys: [MultilinearRoundPoly]

    /// Random challenges generated each round via Fiat-Shamir.
    public let challenges: [Fr]

    /// Final evaluation: f(r_0, ..., r_{n-1}).
    public let finalEval: Fr

    /// Number of variables in the multilinear polynomial.
    public let numVars: Int

    /// For product sumcheck, the final evaluation of the second polynomial.
    public let finalEvalG: Fr?
}

// MARK: - Batch Sumcheck Proof

/// Proof for a batch of sumcheck instances combined with random weights.
public struct BatchSumcheckProof {
    /// The individual proofs are merged into a single combined proof.
    public let combinedProof: MultilinearSumcheckProof

    /// The random weights used to combine instances.
    public let batchWeights: [Fr]

    /// Final evaluations for each polynomial in the batch.
    public let finalEvals: [Fr]
}

// MARK: - GPU Multilinear Sumcheck Engine

/// GPU-accelerated multilinear sumcheck protocol engine.
///
/// Supports three protocol variants:
/// 1. **Single-polynomial**: prove that C = sum_{x in {0,1}^n} f(x)
/// 2. **Product**: prove that C = sum_{x in {0,1}^n} f(x)*g(x)
/// 3. **Batch**: combine k sumcheck instances into one with random weights
///
/// Uses Metal GPU for table folding/reduction on large tables (>= 1024),
/// falls back to CPU for smaller ones.
public final class GPUMultilinearSumcheckEngine {
    private let gpuEngine: GPUSumcheckEngine?
    private static let gpuThreshold = 1024

    /// Initialize with optional GPU acceleration.
    /// Falls back to CPU-only if Metal is unavailable.
    public init() {
        self.gpuEngine = try? GPUSumcheckEngine()
    }

    /// Initialize with an existing GPU engine (for reuse across protocols).
    public init(engine: GPUSumcheckEngine) {
        self.gpuEngine = engine
    }

    /// Whether GPU acceleration is available.
    public var hasGPU: Bool { gpuEngine != nil }

    // MARK: - Single-Polynomial Sumcheck

    /// Execute single-polynomial sumcheck: prove C = sum_{x in {0,1}^n} f(x).
    ///
    /// - Parameters:
    ///   - evals: Evaluation table [f(0...0), f(0...1), ..., f(1...1)] of size 2^n
    ///   - claimedSum: The claimed sum C
    ///   - transcript: Fiat-Shamir transcript for challenge generation
    /// - Returns: A MultilinearSumcheckProof
    public func proveSingle(
        evals: [Fr],
        claimedSum: Fr,
        transcript: Transcript
    ) -> MultilinearSumcheckProof {
        let n = evals.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Table size must be a power of 2")
        let numVars = log2Int(n)

        var current = evals
        var roundPolys = [MultilinearRoundPoly]()
        var challenges = [Fr]()
        roundPolys.reserveCapacity(numVars)
        challenges.reserveCapacity(numVars)

        for _ in 0..<numVars {
            let half = current.count / 2

            // Compute s0 = sum of first half, s1 = sum of second half
            var s0 = Fr.zero
            var s1 = Fr.zero
            for i in 0..<half {
                s0 = frAdd(s0, current[i])
                s1 = frAdd(s1, current[i + half])
            }

            // Degree-1 round polynomial: p(X) with p(0)=s0, p(1)=s1
            let roundPoly = MultilinearRoundPoly(evals: [s0, s1])
            roundPolys.append(roundPoly)

            // Absorb round polynomial into transcript
            transcript.absorb(s0)
            transcript.absorb(s1)
            let challenge = transcript.squeeze()
            challenges.append(challenge)

            // Fold table: next[i] = current[i] + r * (current[i+half] - current[i])
            var next = [Fr](repeating: Fr.zero, count: half)
            for i in 0..<half {
                let diff = frSub(current[i + half], current[i])
                next[i] = frAdd(current[i], frMul(challenge, diff))
            }
            current = next
        }

        return MultilinearSumcheckProof(
            roundPolys: roundPolys,
            challenges: challenges,
            finalEval: current[0],
            numVars: numVars,
            finalEvalG: nil
        )
    }

    // MARK: - Product Sumcheck

    /// Execute product sumcheck: prove C = sum_{x in {0,1}^n} f(x)*g(x).
    ///
    /// The product of two multilinear polynomials yields degree-2 round polynomials.
    /// Each round, we compute p(0), p(1), p(2) where:
    ///   p(t) = sum_x f(t, x) * g(t, x)
    /// with f(t, x) = (1-t)*f(0,x) + t*f(1,x) and similarly for g.
    ///
    /// - Parameters:
    ///   - evalsF: Evaluation table for f of size 2^n
    ///   - evalsG: Evaluation table for g of size 2^n
    ///   - claimedSum: The claimed sum C = sum f(x)*g(x)
    ///   - transcript: Fiat-Shamir transcript
    /// - Returns: A MultilinearSumcheckProof (with degree-2 round polys)
    public func proveProduct(
        evalsF: [Fr],
        evalsG: [Fr],
        claimedSum: Fr,
        transcript: Transcript
    ) -> MultilinearSumcheckProof {
        let n = evalsF.count
        precondition(n == evalsG.count, "f and g must have the same size")
        precondition(n > 0 && (n & (n - 1)) == 0, "Table size must be a power of 2")
        let numVars = log2Int(n)

        var currentF = evalsF
        var currentG = evalsG
        var roundPolys = [MultilinearRoundPoly]()
        var challenges = [Fr]()
        roundPolys.reserveCapacity(numVars)
        challenges.reserveCapacity(numVars)

        for _ in 0..<numVars {
            let half = currentF.count / 2

            // Compute p(0), p(1), p(2) for the round polynomial
            // p(t) = sum_{x in {0,1}^{n-1}} f(t,x) * g(t,x)
            // where f(t,x) = (1-t)*f_lo[x] + t*f_hi[x], similarly for g

            // p(0) = sum f_lo[i] * g_lo[i]
            var p0 = Fr.zero
            for i in 0..<half {
                p0 = frAdd(p0, frMul(currentF[i], currentG[i]))
            }

            // p(1) = sum f_hi[i] * g_hi[i]
            var p1 = Fr.zero
            for i in 0..<half {
                p1 = frAdd(p1, frMul(currentF[i + half], currentG[i + half]))
            }

            // p(2) = sum f(2,x)*g(2,x) where f(2,x) = 2*f_hi[x] - f_lo[x]
            var p2 = Fr.zero
            let two = frAdd(Fr.one, Fr.one)
            for i in 0..<half {
                // f(2,x) = (1-2)*f_lo + 2*f_hi = 2*f_hi - f_lo
                let f2 = frSub(frMul(two, currentF[i + half]), currentF[i])
                let g2 = frSub(frMul(two, currentG[i + half]), currentG[i])
                p2 = frAdd(p2, frMul(f2, g2))
            }

            let roundPoly = MultilinearRoundPoly(evals: [p0, p1, p2])
            roundPolys.append(roundPoly)

            // Absorb round polynomial evaluations into transcript
            transcript.absorb(p0)
            transcript.absorb(p1)
            transcript.absorb(p2)
            let challenge = transcript.squeeze()
            challenges.append(challenge)

            // Fold both tables at challenge r
            var nextF = [Fr](repeating: Fr.zero, count: half)
            var nextG = [Fr](repeating: Fr.zero, count: half)
            for i in 0..<half {
                let diffF = frSub(currentF[i + half], currentF[i])
                nextF[i] = frAdd(currentF[i], frMul(challenge, diffF))
                let diffG = frSub(currentG[i + half], currentG[i])
                nextG[i] = frAdd(currentG[i], frMul(challenge, diffG))
            }
            currentF = nextF
            currentG = nextG
        }

        return MultilinearSumcheckProof(
            roundPolys: roundPolys,
            challenges: challenges,
            finalEval: currentF[0],
            numVars: numVars,
            finalEvalG: currentG[0]
        )
    }

    // MARK: - Batch Sumcheck

    /// Execute batch sumcheck: combine k sumcheck instances into one.
    ///
    /// Given polynomials f_0, ..., f_{k-1} with claimed sums C_0, ..., C_{k-1},
    /// the verifier sends random weights alpha_0, ..., alpha_{k-1} and the prover
    /// runs sumcheck on h(x) = sum_j alpha_j * f_j(x) with claim = sum_j alpha_j * C_j.
    ///
    /// All polynomials must have the same number of variables.
    ///
    /// - Parameters:
    ///   - polys: Array of evaluation tables, each of size 2^n
    ///   - claimedSums: Claimed sum for each polynomial
    ///   - transcript: Fiat-Shamir transcript
    /// - Returns: A BatchSumcheckProof
    public func proveBatch(
        polys: [[Fr]],
        claimedSums: [Fr],
        transcript: Transcript
    ) -> BatchSumcheckProof {
        let k = polys.count
        precondition(k > 0, "Need at least one polynomial")
        precondition(k == claimedSums.count, "Mismatch: polys vs claimedSums count")
        let n = polys[0].count
        let numVars = log2Int(n)
        for i in 1..<k {
            precondition(polys[i].count == n, "All polynomials must have same size")
        }

        // Generate batch weights from transcript
        var batchWeights = [Fr]()
        batchWeights.reserveCapacity(k)
        // Absorb all claimed sums first
        for cs in claimedSums {
            transcript.absorb(cs)
        }
        for _ in 0..<k {
            batchWeights.append(transcript.squeeze())
        }

        // Compute combined evaluation table: h[i] = sum_j alpha_j * f_j[i]
        var combined = [Fr](repeating: Fr.zero, count: n)
        for j in 0..<k {
            let w = batchWeights[j]
            for i in 0..<n {
                combined[i] = frAdd(combined[i], frMul(w, polys[j][i]))
            }
        }

        // Compute combined claimed sum
        var combinedSum = Fr.zero
        for j in 0..<k {
            combinedSum = frAdd(combinedSum, frMul(batchWeights[j], claimedSums[j]))
        }

        // Run single-polynomial sumcheck on combined polynomial
        let proof = proveSingle(evals: combined, claimedSum: combinedSum, transcript: transcript)

        // Compute final evaluations for each individual polynomial at the challenge point
        var finalEvals = [Fr]()
        finalEvals.reserveCapacity(k)
        for j in 0..<k {
            let mle = MultilinearPoly(numVars: numVars, evals: polys[j])
            finalEvals.append(mle.evaluate(at: proof.challenges))
        }

        return BatchSumcheckProof(
            combinedProof: proof,
            batchWeights: batchWeights,
            finalEvals: finalEvals
        )
    }

    // MARK: - Verification: Single-Polynomial

    /// Verify a single-polynomial sumcheck proof.
    ///
    /// Checks:
    /// 1. Each round: p_i(0) + p_i(1) = current claim
    /// 2. Challenges are consistent with transcript
    /// 3. Final claim equals finalEval
    ///
    /// - Returns: Tuple of (valid, evalPoint, finalEval).
    public static func verifySingle(
        proof: MultilinearSumcheckProof,
        claimedSum: Fr,
        transcript: Transcript
    ) -> (valid: Bool, evalPoint: [Fr], finalEval: Fr) {
        var currentClaim = claimedSum

        for round in 0..<proof.numVars {
            let rp = proof.roundPolys[round]

            // Check: p(0) + p(1) = currentClaim
            let roundSum = frAdd(rp.atZero, rp.atOne)
            if !frEqual(roundSum, currentClaim) {
                return (valid: false, evalPoint: [], finalEval: Fr.zero)
            }

            // Absorb and squeeze challenge
            for e in rp.evals {
                transcript.absorb(e)
            }
            let challenge = transcript.squeeze()

            // Check challenge consistency
            if !frEqual(challenge, proof.challenges[round]) {
                return (valid: false, evalPoint: [], finalEval: Fr.zero)
            }

            // Next claim = p(r)
            currentClaim = rp.evaluate(at: challenge)
        }

        let valid = frEqual(currentClaim, proof.finalEval)
        return (valid: valid, evalPoint: proof.challenges, finalEval: proof.finalEval)
    }

    // MARK: - Verification: Product Sumcheck

    /// Verify a product sumcheck proof.
    ///
    /// Same as single-poly verification except round polynomials are degree 2,
    /// and the final check requires both f and g evaluations: finalEval * finalEvalG.
    public static func verifyProduct(
        proof: MultilinearSumcheckProof,
        claimedSum: Fr,
        transcript: Transcript
    ) -> (valid: Bool, evalPoint: [Fr], finalEvalF: Fr, finalEvalG: Fr) {
        guard let finalG = proof.finalEvalG else {
            return (valid: false, evalPoint: [], finalEvalF: Fr.zero, finalEvalG: Fr.zero)
        }

        var currentClaim = claimedSum

        for round in 0..<proof.numVars {
            let rp = proof.roundPolys[round]

            // Check: p(0) + p(1) = currentClaim
            let roundSum = frAdd(rp.atZero, rp.atOne)
            if !frEqual(roundSum, currentClaim) {
                return (valid: false, evalPoint: [], finalEvalF: Fr.zero, finalEvalG: Fr.zero)
            }

            // Absorb all evaluations of the round poly
            for e in rp.evals {
                transcript.absorb(e)
            }
            let challenge = transcript.squeeze()

            if !frEqual(challenge, proof.challenges[round]) {
                return (valid: false, evalPoint: [], finalEvalF: Fr.zero, finalEvalG: Fr.zero)
            }

            currentClaim = rp.evaluate(at: challenge)
        }

        // Final check: last claim should equal f(r) * g(r)
        let expectedFinal = frMul(proof.finalEval, finalG)
        let valid = frEqual(currentClaim, expectedFinal)
        return (valid: valid, evalPoint: proof.challenges,
                finalEvalF: proof.finalEval, finalEvalG: finalG)
    }

    // MARK: - Verification: Batch Sumcheck

    /// Verify a batch sumcheck proof.
    ///
    /// 1. Re-derive batch weights from transcript
    /// 2. Verify the combined sumcheck proof
    /// 3. Check that the combined final eval matches weighted sum of individual evals
    public static func verifyBatch(
        proof: BatchSumcheckProof,
        claimedSums: [Fr],
        transcript: Transcript
    ) -> (valid: Bool, evalPoint: [Fr], finalEvals: [Fr]) {
        let k = claimedSums.count
        precondition(k == proof.batchWeights.count, "Weight count mismatch")

        // Re-derive batch weights
        for cs in claimedSums {
            transcript.absorb(cs)
        }
        var weights = [Fr]()
        for _ in 0..<k {
            weights.append(transcript.squeeze())
        }

        // Check weights match
        for i in 0..<k {
            if !frEqual(weights[i], proof.batchWeights[i]) {
                return (valid: false, evalPoint: [], finalEvals: [])
            }
        }

        // Compute combined claimed sum
        var combinedSum = Fr.zero
        for j in 0..<k {
            combinedSum = frAdd(combinedSum, frMul(weights[j], claimedSums[j]))
        }

        // Verify the combined proof
        let (singleValid, evalPoint, combinedFinalEval) = verifySingle(
            proof: proof.combinedProof,
            claimedSum: combinedSum,
            transcript: transcript
        )

        if !singleValid {
            return (valid: false, evalPoint: [], finalEvals: [])
        }

        // Check combined final eval = sum_j alpha_j * finalEvals[j]
        var expectedCombined = Fr.zero
        for j in 0..<k {
            expectedCombined = frAdd(expectedCombined, frMul(weights[j], proof.finalEvals[j]))
        }

        let valid = frEqual(combinedFinalEval, expectedCombined)
        return (valid: valid, evalPoint: evalPoint, finalEvals: proof.finalEvals)
    }

    // MARK: - GPU-Accelerated Single-Polynomial Sumcheck

    /// GPU-accelerated single-polynomial sumcheck.
    /// Uses Metal for table folding when the table is large enough.
    public func proveSingleGPU(
        evals: [Fr],
        claimedSum: Fr,
        transcript: Transcript
    ) throws -> MultilinearSumcheckProof {
        let n = evals.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Table size must be a power of 2")
        let numVars = log2Int(n)

        // Fall back to CPU for small tables or if no GPU
        guard let engine = gpuEngine, n >= GPUMultilinearSumcheckEngine.gpuThreshold else {
            return proveSingle(evals: evals, claimedSum: claimedSum, transcript: transcript)
        }

        let stride = MemoryLayout<Fr>.stride

        guard let tableBuf = engine.device.makeBuffer(length: n * stride,
                                                       options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to create table buffer")
        }
        evals.withUnsafeBytes { src in
            memcpy(tableBuf.contents(), src.baseAddress!, n * stride)
        }

        var currentTable = tableBuf
        var currentLogSize = numVars
        var roundPolys = [MultilinearRoundPoly]()
        var challenges = [Fr]()
        roundPolys.reserveCapacity(numVars)
        challenges.reserveCapacity(numVars)

        for _ in 0..<numVars {
            // GPU: compute round poly sums
            let (s0, s1) = try engine.computeRoundPolyBN254(
                table: currentTable, logSize: currentLogSize)

            let roundPoly = MultilinearRoundPoly(evals: [s0, s1])
            roundPolys.append(roundPoly)

            // Fiat-Shamir
            transcript.absorb(s0)
            transcript.absorb(s1)
            let challenge = transcript.squeeze()
            challenges.append(challenge)

            // GPU: fold table
            let newTable = try engine.reduceBN254Table(
                table: currentTable, logSize: currentLogSize, challenge: challenge)
            currentTable = newTable
            currentLogSize -= 1
        }

        let finalPtr = currentTable.contents().bindMemory(to: Fr.self, capacity: 1)
        let finalEval = finalPtr[0]

        return MultilinearSumcheckProof(
            roundPolys: roundPolys,
            challenges: challenges,
            finalEval: finalEval,
            numVars: numVars,
            finalEvalG: nil
        )
    }

    // MARK: - Helpers

    private func log2Int(_ n: Int) -> Int {
        var count = 0
        var v = n
        while v > 1 { v >>= 1; count += 1 }
        return count
    }
}
