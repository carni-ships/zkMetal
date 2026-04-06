// GPU-accelerated sumcheck protocol engine.
//
// Full sumcheck protocol implementation supporting:
//   1. Standard sumcheck: prove sum of f(x) over boolean hypercube
//   2. Product sumcheck (GKR-style): prove sum of f(x)*g(x)*...
//   3. Combined sumcheck: batch multiple claims with random weights
//   4. Configurable degree bound per round
//   5. GPU-accelerated partial evaluation over boolean hypercube
//
// Round-by-round univariate polynomial generation with challenge binding
// and partial evaluation. Prover + verifier in one engine.

import Foundation
import Metal

// MARK: - Sumcheck Claim

/// A single sumcheck claim: a polynomial and its claimed sum over {0,1}^n.
public struct SumcheckClaim {
    /// Evaluation table of the multilinear polynomial (size 2^numVars).
    public let evals: [Fr]

    /// Number of variables (log2 of evals.count).
    public let numVars: Int

    /// Claimed sum: sum_{x in {0,1}^n} f(x).
    public let claimedSum: Fr

    public init(evals: [Fr], claimedSum: Fr) {
        let n = evals.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Table size must be power of 2")
        self.evals = evals
        self.numVars = Self.log2Int(n)
        self.claimedSum = claimedSum
    }

    /// Convenience: compute claimed sum from evaluations.
    public static func withComputedSum(evals: [Fr]) -> SumcheckClaim {
        var s = Fr.zero
        for e in evals { s = frAdd(s, e) }
        return SumcheckClaim(evals: evals, claimedSum: s)
    }

    private static func log2Int(_ n: Int) -> Int {
        var c = 0; var v = n
        while v > 1 { v >>= 1; c += 1 }
        return c
    }
}

// MARK: - Round Univariate

/// A univariate polynomial for one sumcheck round, stored as evaluations at 0, 1, ..., d.
/// Degree is evals.count - 1.
public struct RoundUnivariate: Equatable {
    /// Evaluations at points 0, 1, ..., degree.
    public let evals: [Fr]

    /// Degree of the polynomial.
    public var degree: Int { evals.count - 1 }

    /// Maximum allowed degree for this round.
    public let degreeBound: Int

    public init(evals: [Fr], degreeBound: Int) {
        precondition(evals.count >= 2, "Need at least 2 evaluations")
        precondition(evals.count - 1 <= degreeBound,
                     "Polynomial degree \(evals.count - 1) exceeds bound \(degreeBound)")
        self.evals = evals
        self.degreeBound = degreeBound
    }

    /// p(0)
    public var atZero: Fr { evals[0] }

    /// p(1)
    public var atOne: Fr { evals[1] }

    /// Evaluate at arbitrary point via Lagrange interpolation over {0, 1, ..., d}.
    public func evaluate(at r: Fr) -> Fr {
        let d = evals.count
        if d == 2 {
            // Linear: p(r) = e0 + r*(e1 - e0)
            return frAdd(evals[0], frMul(r, frSub(evals[1], evals[0])))
        }
        if d == 3 {
            // Quadratic through (0,e0), (1,e1), (2,e2)
            let e0 = evals[0], e1 = evals[1], e2 = evals[2]
            let rM1 = frSub(r, Fr.one)
            let two = frAdd(Fr.one, Fr.one)
            let rM2 = frSub(r, two)
            let twoInv = frInverse(two)
            let l0 = frMul(frMul(rM1, rM2), twoInv)
            let l1 = frNeg(frMul(r, rM2))
            let l2 = frMul(frMul(r, rM1), twoInv)
            return frAdd(frAdd(frMul(e0, l0), frMul(e1, l1)), frMul(e2, l2))
        }
        // General Lagrange
        return generalLagrange(points: evals, at: r)
    }

    public static func == (lhs: RoundUnivariate, rhs: RoundUnivariate) -> Bool {
        guard lhs.evals.count == rhs.evals.count else { return false }
        for i in 0..<lhs.evals.count {
            if !frEqual(lhs.evals[i], rhs.evals[i]) { return false }
        }
        return true
    }
}

/// General Lagrange interpolation over evaluation points {0, 1, ..., n-1}.
private func generalLagrange(points: [Fr], at r: Fr) -> Fr {
    let n = points.count
    var result = Fr.zero
    for i in 0..<n {
        var basis = Fr.one
        let xi = frFromInt(UInt64(i))
        for j in 0..<n {
            if i == j { continue }
            let xj = frFromInt(UInt64(j))
            let num = frSub(r, xj)
            let den = frSub(xi, xj)
            basis = frMul(basis, frMul(num, frInverse(den)))
        }
        result = frAdd(result, frMul(points[i], basis))
    }
    return result
}

// MARK: - Protocol Proof

/// Complete proof from the sumcheck protocol engine.
public struct SumcheckProtocolProof {
    /// Round univariate polynomials (one per variable).
    public let roundPolys: [RoundUnivariate]

    /// Challenges generated each round via Fiat-Shamir.
    public let challenges: [Fr]

    /// Final evaluation f(r_0, ..., r_{n-1}).
    public let finalEval: Fr

    /// Number of variables.
    public let numVars: Int

    /// For product sumcheck: final evaluations of each factor polynomial.
    public let factorFinalEvals: [Fr]?

    /// Degree bound used per round.
    public let degreeBound: Int

    public init(roundPolys: [RoundUnivariate], challenges: [Fr], finalEval: Fr,
                numVars: Int, factorFinalEvals: [Fr]?, degreeBound: Int) {
        self.roundPolys = roundPolys
        self.challenges = challenges
        self.finalEval = finalEval
        self.numVars = numVars
        self.factorFinalEvals = factorFinalEvals
        self.degreeBound = degreeBound
    }
}

// MARK: - Combined Sumcheck Proof

/// Proof for combined (batched) sumcheck over multiple claims.
public struct CombinedSumcheckProof {
    /// The merged sumcheck proof over the combined polynomial.
    public let innerProof: SumcheckProtocolProof

    /// Random weights used to combine claims.
    public let combinationWeights: [Fr]

    /// Final evaluations of each individual polynomial at the challenge point.
    public let individualFinalEvals: [Fr]

    public init(innerProof: SumcheckProtocolProof, combinationWeights: [Fr],
                individualFinalEvals: [Fr]) {
        self.innerProof = innerProof
        self.combinationWeights = combinationWeights
        self.individualFinalEvals = individualFinalEvals
    }
}

// MARK: - Sumcheck Protocol Configuration

/// Configuration for the sumcheck protocol engine.
public struct SumcheckProtocolConfig {
    /// Maximum degree bound per round. For standard multilinear sumcheck this is 1.
    /// For product sumcheck of k polynomials, this is k.
    public let degreeBound: Int

    /// GPU threshold: use GPU for tables with >= this many elements.
    public let gpuThreshold: Int

    /// Whether to enable GPU acceleration.
    public let useGPU: Bool

    public init(degreeBound: Int = 1, gpuThreshold: Int = 1024, useGPU: Bool = true) {
        precondition(degreeBound >= 1, "Degree bound must be >= 1")
        self.degreeBound = degreeBound
        self.gpuThreshold = gpuThreshold
        self.useGPU = useGPU
    }

    /// Standard config for single-polynomial sumcheck (degree 1).
    public static let standard = SumcheckProtocolConfig(degreeBound: 1)

    /// Config for product sumcheck of 2 polynomials (degree 2).
    public static let product2 = SumcheckProtocolConfig(degreeBound: 2)

    /// Config for product sumcheck of 3 polynomials (degree 3).
    public static let product3 = SumcheckProtocolConfig(degreeBound: 3)

    /// CPU-only config (no GPU).
    public static let cpuOnly = SumcheckProtocolConfig(degreeBound: 1, useGPU: false)
}

// MARK: - GPU Sumcheck Protocol Engine

/// GPU-accelerated sumcheck protocol engine.
///
/// Implements a complete interactive sumcheck protocol (prover + verifier)
/// with Metal GPU acceleration for partial evaluation over the boolean hypercube.
///
/// Supports:
/// - Single-polynomial sumcheck (degree-1 round polys)
/// - Product sumcheck for k polynomials (degree-k round polys, GKR-style)
/// - Combined/batched sumcheck (multiple claims merged with random weights)
/// - Configurable degree bound per round
///
/// The prover executes n rounds (one per variable). Each round:
/// 1. Computes a univariate round polynomial p_i(X) of degree <= degreeBound
/// 2. Sends p_i to verifier (via Fiat-Shamir transcript)
/// 3. Receives challenge r_i
/// 4. Partially evaluates (folds) the table at r_i
///
/// GPU acceleration is used for the partial evaluation (folding) step
/// when the table is large enough.
public final class GPUSumcheckProtocolEngine {
    private let gpuEngine: GPUSumcheckEngine?
    private let config: SumcheckProtocolConfig

    /// Initialize with a configuration. Falls back to CPU if Metal unavailable.
    public init(config: SumcheckProtocolConfig = .standard) {
        self.config = config
        if config.useGPU {
            self.gpuEngine = try? GPUSumcheckEngine()
        } else {
            self.gpuEngine = nil
        }
    }

    /// Initialize with an existing GPU engine and configuration.
    public init(engine: GPUSumcheckEngine, config: SumcheckProtocolConfig = .standard) {
        self.gpuEngine = engine
        self.config = config
    }

    /// Whether GPU acceleration is available.
    public var hasGPU: Bool { gpuEngine != nil }

    // MARK: - Standard Sumcheck (Single Polynomial)

    /// Execute standard sumcheck: prove C = sum_{x in {0,1}^n} f(x).
    ///
    /// Produces degree-1 round polynomials (one per variable).
    ///
    /// - Parameters:
    ///   - claim: The sumcheck claim (polynomial + claimed sum)
    ///   - transcript: Fiat-Shamir transcript
    /// - Returns: A SumcheckProtocolProof
    public func proveStandard(
        claim: SumcheckClaim,
        transcript: Transcript
    ) -> SumcheckProtocolProof {
        let numVars = claim.numVars
        var current = claim.evals
        var roundPolys = [RoundUnivariate]()
        var challenges = [Fr]()
        roundPolys.reserveCapacity(numVars)
        challenges.reserveCapacity(numVars)

        for _ in 0..<numVars {
            let half = current.count / 2

            // s0 = sum of first half (x_i = 0), s1 = sum of second half (x_i = 1)
            var s0 = Fr.zero
            var s1 = Fr.zero
            for i in 0..<half {
                s0 = frAdd(s0, current[i])
                s1 = frAdd(s1, current[i + half])
            }

            let roundPoly = RoundUnivariate(evals: [s0, s1], degreeBound: max(config.degreeBound, 1))
            roundPolys.append(roundPoly)

            // Absorb round polynomial into transcript
            transcript.absorb(s0)
            transcript.absorb(s1)
            let challenge = transcript.squeeze()
            challenges.append(challenge)

            // Fold: next[i] = current[i] + r*(current[i+half] - current[i])
            current = foldTable(current, challenge: challenge)
        }

        return SumcheckProtocolProof(
            roundPolys: roundPolys,
            challenges: challenges,
            finalEval: current[0],
            numVars: numVars,
            factorFinalEvals: nil,
            degreeBound: max(config.degreeBound, 1)
        )
    }

    // MARK: - Product Sumcheck (GKR-style)

    /// Execute product sumcheck: prove C = sum_{x in {0,1}^n} prod_j f_j(x).
    ///
    /// For k factor polynomials, produces degree-k round polynomials.
    /// Each round computes p(t) for t = 0, 1, ..., k by evaluating
    /// the product of interpolated factor values at t.
    ///
    /// - Parameters:
    ///   - factors: Array of evaluation tables (each of size 2^n)
    ///   - claimedSum: Claimed sum of the product
    ///   - transcript: Fiat-Shamir transcript
    /// - Returns: A SumcheckProtocolProof with degree-k round polys
    public func proveProduct(
        factors: [[Fr]],
        claimedSum: Fr,
        transcript: Transcript
    ) -> SumcheckProtocolProof {
        let k = factors.count
        precondition(k >= 2, "Product sumcheck needs >= 2 factors")
        let n = factors[0].count
        precondition(n > 0 && (n & (n - 1)) == 0, "Table size must be power of 2")
        let numVars = log2Int(n)
        for i in 1..<k {
            precondition(factors[i].count == n, "All factors must have same size")
        }

        let degreeBound = k  // product of k linear forms has degree k
        var currentFactors = factors
        var roundPolys = [RoundUnivariate]()
        var challenges = [Fr]()
        roundPolys.reserveCapacity(numVars)
        challenges.reserveCapacity(numVars)

        for _ in 0..<numVars {
            let half = currentFactors[0].count / 2

            // Compute p(t) for t = 0, 1, ..., k
            // p(t) = sum_{x in {0,1}^{n-1}} prod_j f_j(t, x)
            // where f_j(t, x) = (1-t)*f_j_lo[x] + t*f_j_hi[x]
            var pEvals = [Fr]()
            pEvals.reserveCapacity(degreeBound + 1)

            for t in 0...degreeBound {
                let tFr = frFromInt(UInt64(t))
                let oneMinusT = frSub(Fr.one, tFr)
                var pT = Fr.zero

                for x in 0..<half {
                    // Compute product of all factors at (t, x)
                    var prod = Fr.one
                    for j in 0..<k {
                        // f_j(t, x) = (1-t)*lo + t*hi
                        let lo = currentFactors[j][x]
                        let hi = currentFactors[j][x + half]
                        let val = frAdd(frMul(oneMinusT, lo), frMul(tFr, hi))
                        prod = frMul(prod, val)
                    }
                    pT = frAdd(pT, prod)
                }
                pEvals.append(pT)
            }

            let roundPoly = RoundUnivariate(evals: pEvals, degreeBound: degreeBound)
            roundPolys.append(roundPoly)

            // Absorb all evaluations
            for e in pEvals {
                transcript.absorb(e)
            }
            let challenge = transcript.squeeze()
            challenges.append(challenge)

            // Fold each factor table
            var nextFactors = [[Fr]]()
            nextFactors.reserveCapacity(k)
            for j in 0..<k {
                nextFactors.append(foldTable(currentFactors[j], challenge: challenge))
            }
            currentFactors = nextFactors
        }

        // Collect final evaluations for each factor
        let factorFinals = currentFactors.map { $0[0] }

        // The final combined eval is the product of individual factor evals
        var finalProduct = Fr.one
        for fe in factorFinals {
            finalProduct = frMul(finalProduct, fe)
        }

        return SumcheckProtocolProof(
            roundPolys: roundPolys,
            challenges: challenges,
            finalEval: finalProduct,
            numVars: numVars,
            factorFinalEvals: factorFinals,
            degreeBound: degreeBound
        )
    }

    // MARK: - Combined Sumcheck (Batched Claims)

    /// Execute combined sumcheck: batch multiple claims into one protocol run.
    ///
    /// Given k claims (f_0, C_0), ..., (f_{k-1}, C_{k-1}), the verifier
    /// sends random combination weights alpha_j and the prover runs sumcheck
    /// on h(x) = sum_j alpha_j * f_j(x) with claim = sum_j alpha_j * C_j.
    ///
    /// All claims must have the same number of variables.
    ///
    /// - Parameters:
    ///   - claims: Array of sumcheck claims
    ///   - transcript: Fiat-Shamir transcript
    /// - Returns: A CombinedSumcheckProof
    public func proveCombined(
        claims: [SumcheckClaim],
        transcript: Transcript
    ) -> CombinedSumcheckProof {
        let k = claims.count
        precondition(k >= 1, "Need at least one claim")
        let numVars = claims[0].numVars
        let n = 1 << numVars
        for i in 1..<k {
            precondition(claims[i].numVars == numVars, "All claims must have same numVars")
        }

        // Generate combination weights from transcript
        for c in claims {
            transcript.absorb(c.claimedSum)
        }
        var weights = [Fr]()
        weights.reserveCapacity(k)
        for _ in 0..<k {
            weights.append(transcript.squeeze())
        }

        // Compute combined polynomial: h[i] = sum_j alpha_j * f_j[i]
        var combined = [Fr](repeating: Fr.zero, count: n)
        for j in 0..<k {
            let w = weights[j]
            for i in 0..<n {
                combined[i] = frAdd(combined[i], frMul(w, claims[j].evals[i]))
            }
        }

        // Compute combined claimed sum
        var combinedSum = Fr.zero
        for j in 0..<k {
            combinedSum = frAdd(combinedSum, frMul(weights[j], claims[j].claimedSum))
        }

        // Run standard sumcheck on combined polynomial
        let combinedClaim = SumcheckClaim(evals: combined, claimedSum: combinedSum)
        let proof = proveStandard(claim: combinedClaim, transcript: transcript)

        // Compute individual final evaluations at the challenge point
        var individualFinals = [Fr]()
        individualFinals.reserveCapacity(k)
        for j in 0..<k {
            let mle = MultilinearPoly(numVars: numVars, evals: claims[j].evals)
            individualFinals.append(mle.evaluate(at: proof.challenges))
        }

        return CombinedSumcheckProof(
            innerProof: proof,
            combinationWeights: weights,
            individualFinalEvals: individualFinals
        )
    }

    // MARK: - GPU-Accelerated Standard Sumcheck

    /// GPU-accelerated standard sumcheck using Metal compute for table folding.
    ///
    /// Falls back to CPU path for small tables or when GPU is unavailable.
    /// The GPU path dispatches table reduction to the Metal compute shader
    /// for each round's partial evaluation step.
    ///
    /// - Parameters:
    ///   - claim: The sumcheck claim
    ///   - transcript: Fiat-Shamir transcript
    /// - Returns: A SumcheckProtocolProof
    public func proveStandardGPU(
        claim: SumcheckClaim,
        transcript: Transcript
    ) throws -> SumcheckProtocolProof {
        let n = claim.evals.count

        // Fall back to CPU for small tables or if no GPU
        guard let engine = gpuEngine, n >= config.gpuThreshold else {
            return proveStandard(claim: claim, transcript: transcript)
        }

        let numVars = claim.numVars
        let stride = MemoryLayout<Fr>.stride

        guard let tableBuf = engine.device.makeBuffer(length: n * stride,
                                                       options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to create sumcheck table buffer")
        }
        claim.evals.withUnsafeBytes { src in
            memcpy(tableBuf.contents(), src.baseAddress!, n * stride)
        }

        var currentTable = tableBuf
        var currentLogSize = numVars
        var roundPolys = [RoundUnivariate]()
        var challenges = [Fr]()
        roundPolys.reserveCapacity(numVars)
        challenges.reserveCapacity(numVars)

        for _ in 0..<numVars {
            // GPU: compute round poly sums
            let (s0, s1) = try engine.computeRoundPolyBN254(
                table: currentTable, logSize: currentLogSize)

            let roundPoly = RoundUnivariate(evals: [s0, s1], degreeBound: max(config.degreeBound, 1))
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

        return SumcheckProtocolProof(
            roundPolys: roundPolys,
            challenges: challenges,
            finalEval: finalEval,
            numVars: numVars,
            factorFinalEvals: nil,
            degreeBound: max(config.degreeBound, 1)
        )
    }

    // MARK: - Verification: Standard Sumcheck

    /// Verify a standard sumcheck proof.
    ///
    /// Checks:
    /// 1. Each round: p_i(0) + p_i(1) = current claim
    /// 2. Degree of each round poly <= degreeBound
    /// 3. Challenges are consistent with transcript
    /// 4. Final claim equals finalEval
    ///
    /// - Returns: (valid, evalPoint, finalEval)
    public static func verifyStandard(
        proof: SumcheckProtocolProof,
        claimedSum: Fr,
        transcript: Transcript
    ) -> (valid: Bool, evalPoint: [Fr], finalEval: Fr) {
        var currentClaim = claimedSum

        for round in 0..<proof.numVars {
            let rp = proof.roundPolys[round]

            // Degree check
            if rp.degree > proof.degreeBound {
                return (valid: false, evalPoint: [], finalEval: Fr.zero)
            }

            // Sum check: p(0) + p(1) = currentClaim
            let roundSum = frAdd(rp.atZero, rp.atOne)
            if !frEqual(roundSum, currentClaim) {
                return (valid: false, evalPoint: [], finalEval: Fr.zero)
            }

            // Absorb and squeeze
            for e in rp.evals {
                transcript.absorb(e)
            }
            let challenge = transcript.squeeze()

            // Challenge consistency
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
    /// Same as standard verification except:
    /// - Round polys may have degree > 1 (up to number of factors)
    /// - Final check: last claim = product of factorFinalEvals
    ///
    /// - Returns: (valid, evalPoint, factorFinalEvals)
    public static func verifyProduct(
        proof: SumcheckProtocolProof,
        claimedSum: Fr,
        transcript: Transcript
    ) -> (valid: Bool, evalPoint: [Fr], factorFinalEvals: [Fr]) {
        guard let factorEvals = proof.factorFinalEvals else {
            return (valid: false, evalPoint: [], factorFinalEvals: [])
        }

        var currentClaim = claimedSum

        for round in 0..<proof.numVars {
            let rp = proof.roundPolys[round]

            // Degree check
            if rp.degree > proof.degreeBound {
                return (valid: false, evalPoint: [], factorFinalEvals: [])
            }

            // Sum check: p(0) + p(1) = currentClaim
            let roundSum = frAdd(rp.atZero, rp.atOne)
            if !frEqual(roundSum, currentClaim) {
                return (valid: false, evalPoint: [], factorFinalEvals: [])
            }

            // Absorb all evaluations
            for e in rp.evals {
                transcript.absorb(e)
            }
            let challenge = transcript.squeeze()

            if !frEqual(challenge, proof.challenges[round]) {
                return (valid: false, evalPoint: [], factorFinalEvals: [])
            }

            currentClaim = rp.evaluate(at: challenge)
        }

        // Final check: currentClaim should equal product of factor evals
        var expectedProduct = Fr.one
        for fe in factorEvals {
            expectedProduct = frMul(expectedProduct, fe)
        }

        let valid = frEqual(currentClaim, expectedProduct)
        return (valid: valid, evalPoint: proof.challenges, factorFinalEvals: factorEvals)
    }

    // MARK: - Verification: Combined Sumcheck

    /// Verify a combined (batched) sumcheck proof.
    ///
    /// 1. Re-derive combination weights from transcript
    /// 2. Verify the inner proof as a standard sumcheck
    /// 3. Check that combined final eval = sum_j alpha_j * individualFinalEvals[j]
    ///
    /// - Returns: (valid, evalPoint, individualFinalEvals)
    public static func verifyCombined(
        proof: CombinedSumcheckProof,
        claimedSums: [Fr],
        transcript: Transcript
    ) -> (valid: Bool, evalPoint: [Fr], individualFinalEvals: [Fr]) {
        let k = claimedSums.count
        precondition(k == proof.combinationWeights.count, "Weight count mismatch")

        // Re-derive weights
        for cs in claimedSums {
            transcript.absorb(cs)
        }
        var weights = [Fr]()
        for _ in 0..<k {
            weights.append(transcript.squeeze())
        }

        // Check weights
        for i in 0..<k {
            if !frEqual(weights[i], proof.combinationWeights[i]) {
                return (valid: false, evalPoint: [], individualFinalEvals: [])
            }
        }

        // Compute combined claimed sum
        var combinedSum = Fr.zero
        for j in 0..<k {
            combinedSum = frAdd(combinedSum, frMul(weights[j], claimedSums[j]))
        }

        // Verify the inner standard proof
        let (innerValid, evalPoint, combinedFinalEval) = verifyStandard(
            proof: proof.innerProof,
            claimedSum: combinedSum,
            transcript: transcript
        )

        if !innerValid {
            return (valid: false, evalPoint: [], individualFinalEvals: [])
        }

        // Check: combined final eval = sum_j alpha_j * individualFinalEvals[j]
        var expectedCombined = Fr.zero
        for j in 0..<k {
            expectedCombined = frAdd(expectedCombined, frMul(weights[j], proof.individualFinalEvals[j]))
        }

        let valid = frEqual(combinedFinalEval, expectedCombined)
        return (valid: valid, evalPoint: evalPoint, individualFinalEvals: proof.individualFinalEvals)
    }

    // MARK: - Partial Evaluation Helpers

    /// Fold (partially evaluate) a table at a challenge point.
    /// Given table of size 2*half, produces table of size half:
    ///   next[i] = table[i] + r * (table[i + half] - table[i])
    private func foldTable(_ table: [Fr], challenge: Fr) -> [Fr] {
        let half = table.count / 2
        var next = [Fr](repeating: Fr.zero, count: half)
        for i in 0..<half {
            let diff = frSub(table[i + half], table[i])
            next[i] = frAdd(table[i], frMul(challenge, diff))
        }
        return next
    }

    /// Compute the sum of evaluations in a table.
    public static func computeSum(_ evals: [Fr]) -> Fr {
        var s = Fr.zero
        for e in evals { s = frAdd(s, e) }
        return s
    }

    /// Evaluate a multilinear polynomial (given as evaluation table) at a point.
    /// Uses sequential partial evaluation (folding).
    public static func evaluateMultilinear(evals: [Fr], at point: [Fr]) -> Fr {
        var current = evals
        for r in point {
            let half = current.count / 2
            var next = [Fr](repeating: Fr.zero, count: half)
            for i in 0..<half {
                let diff = frSub(current[i + half], current[i])
                next[i] = frAdd(current[i], frMul(r, diff))
            }
            current = next
        }
        return current[0]
    }

    /// Compute the equality multilinear eq(x, r) = prod_i (r_i * x_i + (1-r_i)*(1-x_i)).
    /// Returns evaluations over {0,1}^n.
    public static func eqPoly(point: [Fr]) -> [Fr] {
        let numVars = point.count
        let n = 1 << numVars
        var evals = [Fr](repeating: Fr.zero, count: n)
        evals[0] = Fr.one

        for i in 0..<numVars {
            let step = 1 << i
            let ri = point[i]
            let oneMinusRi = frSub(Fr.one, ri)
            // Process in reverse to avoid overwriting values we still need
            for j in stride(from: step - 1, through: 0, by: -1) {
                evals[2 * j + 1] = frMul(evals[j], ri)
                evals[2 * j] = frMul(evals[j], oneMinusRi)
            }
        }
        return evals
    }

    // MARK: - Utility

    private func log2Int(_ n: Int) -> Int {
        var c = 0; var v = n
        while v > 1 { v >>= 1; c += 1 }
        return c
    }
}
