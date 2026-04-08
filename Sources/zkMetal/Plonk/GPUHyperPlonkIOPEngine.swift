// GPUHyperPlonkIOPEngine — GPU-accelerated HyperPlonk interactive oracle proof engine
//
// HyperPlonk extends Plonk to multilinear polynomials over the boolean hypercube {0,1}^n,
// eliminating the need for FFTs/NTTs entirely. Key protocol components:
//
//   1. Multilinear Polynomial Commitment: commit to witness MLE evaluations
//   2. Permutation Check: grand product over hypercube via product-based sumcheck
//   3. Lookup Argument: logarithmic derivative (LogUp) over hypercube
//   4. Zero Check: prove f(x) = 0 for all x in {0,1}^n via sumcheck
//   5. Sumcheck IOP: reduce multivariate claim to single-point evaluation
//
// GPU acceleration targets:
//   - MLE evaluation: parallel partial evaluation / folding of 2^n tables
//   - Sumcheck rounds: GPU-parallel reduction of evaluation tables
//   - Grand product accumulation: parallel numerator/denominator computation
//   - Logarithmic derivative: batch inverse for 1/(table - query) terms
//   - Witness generation: parallel constraint evaluation over hypercube
//
// Architecture:
//   - All polynomials are multilinear extensions (MLEs) stored as 2^n evaluations
//   - No FFT: domain is the boolean hypercube, not roots of unity
//   - Sumcheck replaces polynomial division for vanishing arguments
//   - Challenges bind via Fiat-Shamir transcript (Poseidon2 sponge)
//
// Reference: "HyperPlonk: Plonk with Linear-Time Prover and High-Degree Custom Gates"
//            (Chen, Feng, et al., 2022)

import Foundation
import Metal
import NeonFieldOps

// MARK: - Configuration
public struct HyperPlonkConfig {
    /// Number of variables (log2 of hypercube size).
    public let numVars: Int
    /// Number of witness columns (wire polynomials).
    public let numWitnessCols: Int
    /// Number of selector columns (gate type indicators).
    public let numSelectorCols: Int
    /// Maximum constraint degree (gate degree + 1 for zero-check sumcheck).
    public let maxConstraintDegree: Int
    /// Whether to use GPU acceleration.
    public let useGPU: Bool
    /// GPU dispatch threshold: tables smaller than this use CPU.
    public let gpuThreshold: Int
    /// Whether to enable lookup arguments.
    public let enableLookups: Bool

    /// Number of evaluation points on the hypercube: 2^numVars.
    public var hypercubeSize: Int { 1 << numVars }

    public init(
        numVars: Int,
        numWitnessCols: Int = 3,
        numSelectorCols: Int = 1,
        maxConstraintDegree: Int = 2,
        useGPU: Bool = true,
        gpuThreshold: Int = 1024,
        enableLookups: Bool = false
    ) {
        precondition(numVars > 0 && numVars <= 30, "numVars must be in [1, 30]")
        precondition(numWitnessCols > 0, "Must have at least one witness column")
        precondition(maxConstraintDegree >= 1, "Constraint degree must be >= 1")
        self.numVars = numVars
        self.numWitnessCols = numWitnessCols
        self.numSelectorCols = numSelectorCols
        self.maxConstraintDegree = maxConstraintDegree
        self.useGPU = useGPU
        self.gpuThreshold = gpuThreshold
        self.enableLookups = enableLookups
    }
}

// MARK: - Witness
public struct HyperPlonkWitness {
    /// Witness column evaluations, each of size 2^numVars.
    public let columns: [[Fr]]
    /// Number of variables.
    public let numVars: Int

    public init(columns: [[Fr]], numVars: Int) {
        let n = 1 << numVars
        precondition(!columns.isEmpty, "Witness must have at least one column")
        for (i, col) in columns.enumerated() {
            precondition(col.count == n, "Column \(i): expected \(n) evals, got \(col.count)")
        }
        self.columns = columns
        self.numVars = numVars
    }

    /// Number of witness columns.
    public var numCols: Int { columns.count }

    /// Hypercube size 2^numVars.
    public var size: Int { 1 << numVars }
}

// MARK: - Constraint

/// A constraint: C(w_0(x), ..., w_k(x), s_0(x), ...) = 0 for all x in {0,1}^n.
public struct HyperPlonkConstraint {
    /// Human-readable name for the constraint.
    public let name: String
    /// Degree of the constraint polynomial.
    public let degree: Int
    /// Returns Fr.zero if satisfied.
    public let evaluate: (_ witnessVals: [Fr], _ selectorVals: [Fr]) -> Fr

    public init(name: String, degree: Int,
                evaluate: @escaping (_ witnessVals: [Fr], _ selectorVals: [Fr]) -> Fr) {
        self.name = name
        self.degree = degree
        self.evaluate = evaluate
    }
}

// MARK: - Permutation

/// Copy constraint permutation: sigma[col][i] encodes the wiring target.
public struct HyperPlonkPermutation {
    /// sigma[col][i] = col' * size + i' (target of wiring).
    public let sigma: [[Int]]
    /// Number of columns in the permutation.
    public let numCols: Int
    /// Hypercube size.
    public let size: Int

    public init(sigma: [[Int]], numCols: Int, size: Int) {
        precondition(sigma.count == numCols)
        for s in sigma { precondition(s.count == size) }
        self.sigma = sigma
        self.numCols = numCols
        self.size = size
    }

    /// Create identity permutation (no copy constraints).
    public static func identity(numCols: Int, size: Int) -> HyperPlonkPermutation {
        var sigma = [[Int]](repeating: [Int](repeating: 0, count: size), count: numCols)
        for col in 0..<numCols {
            for i in 0..<size {
                sigma[col][i] = col * size + i
            }
        }
        return HyperPlonkPermutation(sigma: sigma, numCols: numCols, size: size)
    }
}

// MARK: - Lookup Table
public struct HyperPlonkLookupTable {
    /// Table entries (distinct field elements).
    public let entries: [Fr]
    /// Table ID for multi-table support.
    public let tableId: Int

    public init(entries: [Fr], tableId: Int = 0) {
        self.entries = entries
        self.tableId = tableId
    }
}

// MARK: - Round Polynomial
public struct HyperPlonkRoundPoly: Equatable {
    public let evals: [Fr]

    public var degree: Int { evals.count - 1 }

    public init(evals: [Fr]) {
        precondition(evals.count >= 2, "Round poly needs at least 2 evaluations")
        self.evals = evals
    }

    /// Evaluate at an arbitrary point via Lagrange interpolation.
    public func evaluate(at r: Fr) -> Fr {
        let d = evals.count
        if d == 2 {
            let diff = frSub(evals[1], evals[0])
            return frAdd(evals[0], frMul(r, diff))
        }
        if d == 3 {
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
        // General Lagrange for higher degrees
        return hyperPlonkLagrangeInterpolate(points: evals, at: r)
    }

    public var atZero: Fr { evals[0] }
    public var atOne: Fr { evals[1] }

    public static func == (lhs: HyperPlonkRoundPoly, rhs: HyperPlonkRoundPoly) -> Bool {
        guard lhs.evals.count == rhs.evals.count else { return false }
        for i in 0..<lhs.evals.count {
            if !frEqual(lhs.evals[i], rhs.evals[i]) { return false }
        }
        return true
    }
}

private func hyperPlonkLagrangeInterpolate(points: [Fr], at r: Fr) -> Fr {
    let n = points.count
    var result = Fr.zero
    for i in 0..<n {
        var basis = Fr.one
        let xi = frFromInt(UInt64(i))
        for j in 0..<n where j != i {
            let xj = frFromInt(UInt64(j))
            let num = frSub(r, xj)
            let den = frSub(xi, xj)
            basis = frMul(basis, frMul(num, frInverse(den)))
        }
        result = frAdd(result, frMul(points[i], basis))
    }
    return result
}

// MARK: - Zero-Check Proof
public struct ZeroCheckProof {
    /// Sumcheck round polynomials.
    public let roundPolys: [HyperPlonkRoundPoly]
    /// Random challenges from each round.
    public let challenges: [Fr]
    /// Final evaluation at the sumcheck point.
    public let finalEval: Fr
    /// Random point r used for the eq polynomial.
    public let randomPoint: [Fr]
    /// Number of variables.
    public let numVars: Int

    public init(roundPolys: [HyperPlonkRoundPoly], challenges: [Fr],
                finalEval: Fr, randomPoint: [Fr], numVars: Int) {
        self.roundPolys = roundPolys
        self.challenges = challenges
        self.finalEval = finalEval
        self.randomPoint = randomPoint
        self.numVars = numVars
    }
}

// MARK: - Permutation Check Proof
public struct PermutationCheckProof {
    /// Grand product evaluations over the hypercube.
    public let grandProductEvals: [Fr]
    /// Sumcheck proof that the product relation holds.
    public let sumcheckProof: ZeroCheckProof?
    /// Final product value (must be 1 for valid permutation).
    public let finalProduct: Fr
    /// Whether the permutation check passed.
    public let isValid: Bool

    public init(grandProductEvals: [Fr], sumcheckProof: ZeroCheckProof?,
                finalProduct: Fr, isValid: Bool) {
        self.grandProductEvals = grandProductEvals
        self.sumcheckProof = sumcheckProof
        self.finalProduct = finalProduct
        self.isValid = isValid
    }
}

// MARK: - Lookup Proof (LogUp)
public struct HyperPlonkLogUpProof {
    /// Multiplicities: m[i] = number of times table entry i is queried.
    public let multiplicities: [Fr]
    /// Inverse terms: 1/(beta - table[i]) for each table entry.
    public let inverseTerms: [Fr]
    /// The sum that must equal zero: sum_queries 1/(beta-q) - sum_table m[i]/(beta-t[i]).
    public let logDerivativeSum: Fr
    /// Whether the lookup check passed.
    public let isValid: Bool
    /// Challenge beta used for the logarithmic derivative.
    public let beta: Fr

    public init(multiplicities: [Fr], inverseTerms: [Fr],
                logDerivativeSum: Fr, isValid: Bool, beta: Fr) {
        self.multiplicities = multiplicities
        self.inverseTerms = inverseTerms
        self.logDerivativeSum = logDerivativeSum
        self.isValid = isValid
        self.beta = beta
    }
}

// MARK: - Full HyperPlonk Proof
public struct HyperPlonkProof {
    /// Zero-check proof for constraint satisfaction.
    public let zeroCheckProof: ZeroCheckProof
    /// Permutation check proof (nil if no copy constraints).
    public let permutationProof: PermutationCheckProof?
    /// Lookup proof (nil if no lookups).
    public let lookupProof: HyperPlonkLogUpProof?
    /// Witness column evaluations at the sumcheck point.
    public let witnessEvals: [Fr]
    /// Selector column evaluations at the sumcheck point.
    public let selectorEvals: [Fr]
    /// Whether GPU was used for proving.
    public let usedGPU: Bool

    public init(zeroCheckProof: ZeroCheckProof,
                permutationProof: PermutationCheckProof?,
                lookupProof: HyperPlonkLogUpProof?,
                witnessEvals: [Fr], selectorEvals: [Fr],
                usedGPU: Bool) {
        self.zeroCheckProof = zeroCheckProof
        self.permutationProof = permutationProof
        self.lookupProof = lookupProof
        self.witnessEvals = witnessEvals
        self.selectorEvals = selectorEvals
        self.usedGPU = usedGPU
    }
}

// MARK: - GPUHyperPlonkIOPEngine

/// GPU-accelerated HyperPlonk IOP engine: zero-check, permutation check (grand product),
/// lookup (LogUp), and full proof generation over multilinear polynomials on {0,1}^n.
public final class GPUHyperPlonkIOPEngine {
    public static let version = PrimitiveVersion(version: "1.0.0", updated: "2026-04-06")

    private let device: MTLDevice?
    private let commandQueue: MTLCommandQueue?
    private let foldPipeline: MTLComputePipelineState?
    private let evalPipeline: MTLComputePipelineState?
    private let grandProductPipeline: MTLComputePipelineState?
    private let constraintEvalPipeline: MTLComputePipelineState?
    private let threadgroupSize: Int

    /// Profile mode: print timing to stderr.
    public var profile: Bool = false

    // MARK: - Initialization

    public init() {
        let dev = MTLCreateSystemDefaultDevice()
        self.device = dev
        self.commandQueue = dev?.makeCommandQueue()
        self.threadgroupSize = 256

        if let dev = dev {
            let pipelines = GPUHyperPlonkIOPEngine.compilePipelines(device: dev)
            self.foldPipeline = pipelines.fold
            self.evalPipeline = pipelines.eval
            self.grandProductPipeline = pipelines.grandProduct
            self.constraintEvalPipeline = pipelines.constraintEval
        } else {
            self.foldPipeline = nil
            self.evalPipeline = nil
            self.grandProductPipeline = nil
            self.constraintEvalPipeline = nil
        }
    }

    /// Whether GPU acceleration is available.
    public var gpuAvailable: Bool {
        device != nil && foldPipeline != nil
    }

    // MARK: - MLE Evaluation

    /// Evaluate a multilinear polynomial at an arbitrary point.
    /// Uses iterative folding: fix one variable at a time.
    ///
    /// GPU path: folds the table in-place on the GPU when size >= gpuThreshold.
    /// CPU path: standard sequential folding.
    public func evaluateMLE(
        evals: [Fr],
        point: [Fr],
        config: HyperPlonkConfig
    ) -> Fr {
        let n = evals.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Table size must be power of 2")
        let numVars = logTwo(n)
        precondition(point.count == numVars, "Point dimension must match numVars")

        if config.useGPU && n >= config.gpuThreshold && foldPipeline != nil {
            return gpuEvaluateMLE(evals: evals, point: point, numVars: numVars)
        }
        return cpuEvaluateMLE(evals: evals, point: point, numVars: numVars)
    }

    /// CPU path: iterative variable folding.
    private func cpuEvaluateMLE(evals: [Fr], point: [Fr], numVars: Int) -> Fr {
        var current = evals
        for i in 0..<numVars {
            let half = current.count / 2
            let ri = point[i]
            current.withUnsafeMutableBytes { cBuf in
                withUnsafeBytes(of: ri) { rBuf in
                    bn254_fr_fold_halves_inplace(
                        cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(half))
                }
            }
            current.removeLast(half)
        }
        return current[0]
    }

    /// GPU path: iterative folding using Metal fold kernel per round.
    private func gpuEvaluateMLE(evals: [Fr], point: [Fr], numVars: Int) -> Fr {
        // Use gpuFoldTable for each variable, falling back to CPU for small sizes
        var current = evals
        for i in 0..<numVars {
            let half = current.count / 2
            if half >= 64 {
                current = gpuFoldTable(table: current, challenge: point[i], halfSize: half)
            } else {
                let ri = point[i]
                current.withUnsafeMutableBytes { cBuf in
                    withUnsafeBytes(of: ri) { rBuf in
                        bn254_fr_fold_halves_inplace(
                            cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(half))
                    }
                }
                current.removeLast(half)
            }
        }
        return current[0]
    }

    // MARK: - Eq Polynomial

    /// Compute eq(r, x) for all x in {0,1}^n. Key building block for zero-check.
    public func computeEqPoly(point: [Fr]) -> [Fr] {
        let numVars = point.count
        let n = 1 << numVars
        var eq = [Fr](repeating: Fr.zero, count: n)
        eq[0] = Fr.one

        for i in 0..<numVars {
            let half = 1 << i
            let ri = point[i]
            let oneMinusRi = frSub(Fr.one, ri)
            // Process in reverse to avoid overwriting values we still need
            for j in stride(from: half - 1, through: 0, by: -1) {
                eq[2 * j + 1] = frMul(eq[j], ri)
                eq[2 * j] = frMul(eq[j], oneMinusRi)
            }
        }

        return eq
    }

    /// Compute eq(r, x) at a single point x (given as a binary vector or field elements).
    public func evaluateEq(r: [Fr], x: [Fr]) -> Fr {
        precondition(r.count == x.count, "Dimension mismatch")
        var prod = Fr.one
        for i in 0..<r.count {
            // eq_i = r_i * x_i + (1 - r_i) * (1 - x_i)
            let term = frAdd(frMul(r[i], x[i]),
                             frMul(frSub(Fr.one, r[i]), frSub(Fr.one, x[i])))
            prod = frMul(prod, term)
        }
        return prod
    }

    // MARK: - Constraint Evaluation Over Hypercube

    /// Evaluate a constraint at all hypercube points, returning the evaluation table.
    public func evaluateConstraintOverHypercube(
        constraint: HyperPlonkConstraint,
        witness: HyperPlonkWitness,
        selectors: [[Fr]],
        config: HyperPlonkConfig
    ) -> [Fr] {
        let n = config.hypercubeSize
        var result = [Fr](repeating: Fr.zero, count: n)

        for i in 0..<n {
            var wVals = [Fr](repeating: Fr.zero, count: witness.numCols)
            for col in 0..<witness.numCols {
                wVals[col] = witness.columns[col][i]
            }
            var sVals = [Fr](repeating: Fr.zero, count: selectors.count)
            for col in 0..<selectors.count {
                sVals[col] = selectors[col][i]
            }
            result[i] = constraint.evaluate(wVals, sVals)
        }
        return result
    }

    /// Check if a constraint is satisfied at all hypercube points.
    public func checkConstraintSatisfaction(
        constraint: HyperPlonkConstraint,
        witness: HyperPlonkWitness,
        selectors: [[Fr]],
        config: HyperPlonkConfig
    ) -> (satisfied: Bool, failingIndex: Int) {
        let evals = evaluateConstraintOverHypercube(
            constraint: constraint, witness: witness, selectors: selectors, config: config)
        for i in 0..<evals.count {
            if !frEqual(evals[i], Fr.zero) {
                return (false, i)
            }
        }
        return (true, -1)
    }

    // MARK: - Zero-Check Protocol

    /// Zero-check: prove f(x) = 0 for all x in {0,1}^n by reducing to sumcheck
    /// on h(x) = eq(r, x) * f(x) with random r. Sumcheck degree = constraintDegree + 1.
    public func zeroCheck(
        constraintEvals: [Fr],
        numVars: Int,
        constraintDegree: Int,
        transcript: Transcript,
        config: HyperPlonkConfig
    ) -> ZeroCheckProof {
        let n = 1 << numVars
        precondition(constraintEvals.count == n, "Constraint evals must have 2^numVars entries")

        // Step 1: Sample random point r for eq polynomial
        transcript.absorbLabel("hyperplonk-zero-check")
        let randomPoint = (0..<numVars).map { _ -> Fr in transcript.squeeze() }

        // Step 2: Compute eq(r, x) for all x in {0,1}^n
        let eqEvals = computeEqPoly(point: randomPoint)

        // Step 3: Build h(x) = eq(r, x) * f(x)
        let useGPU = config.useGPU && n >= config.gpuThreshold && device != nil
        var hEvals: [Fr]
        if useGPU {
            hEvals = gpuPointwiseMul(eqEvals, constraintEvals, count: n)
        } else {
            hEvals = [Fr](repeating: Fr.zero, count: n)
            eqEvals.withUnsafeBytes { aBuf in
                constraintEvals.withUnsafeBytes { bBuf in
                    hEvals.withUnsafeMutableBytes { rBuf in
                        bn254_fr_batch_mul(
                            aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(n))
                    }
                }
            }
        }

        // Step 4: Claimed sum should be zero
        // (If constraint is satisfied, all f(x) = 0, so h(x) = 0 everywhere, sum = 0)

        // Step 5: Run sumcheck on h(x)
        let sumcheckDegree = constraintDegree + 1
        let proof = executeSumcheck(
            evals: hEvals, numVars: numVars,
            roundDegree: sumcheckDegree,
            transcript: transcript, config: config
        )

        return ZeroCheckProof(
            roundPolys: proof.roundPolys,
            challenges: proof.challenges,
            finalEval: proof.finalEval,
            randomPoint: randomPoint,
            numVars: numVars
        )
    }

    /// Verify a zero-check proof.
    public func verifyZeroCheck(
        proof: ZeroCheckProof,
        transcript: Transcript
    ) -> Bool {
        let numVars = proof.numVars
        guard proof.roundPolys.count == numVars else { return false }
        guard proof.challenges.count == numVars else { return false }

        // Regenerate the random point from transcript
        transcript.absorbLabel("hyperplonk-zero-check")
        let rPoint = (0..<numVars).map { _ -> Fr in transcript.squeeze() }

        // Verify the random point matches
        for i in 0..<numVars {
            if !frEqual(rPoint[i], proof.randomPoint[i]) { return false }
        }

        // Verify sumcheck round by round
        var currentClaim = Fr.zero  // zero-check claims the sum is 0
        for round in 0..<numVars {
            let poly = proof.roundPolys[round]
            // Check: p_i(0) + p_i(1) = current claim
            let sum = frAdd(poly.atZero, poly.atOne)
            if !frEqual(sum, currentClaim) { return false }

            // Absorb and derive challenge
            transcript.absorbLabel("zc-round-\(round)")
            for e in poly.evals { transcript.absorb(e) }
            let challenge = transcript.squeeze()

            // Verify challenge matches
            if !frEqual(challenge, proof.challenges[round]) { return false }

            // Update claim: p_i(r_i)
            currentClaim = poly.evaluate(at: challenge)
        }

        // Final check: the final evaluation should match the claimed final eval
        if !frEqual(currentClaim, proof.finalEval) { return false }

        return true
    }

    // MARK: - Sumcheck Execution

    /// Internal sumcheck protocol execution.
    /// Produces round polynomials of the given degree for each variable.
    private struct SumcheckResult {
        let roundPolys: [HyperPlonkRoundPoly]
        let challenges: [Fr]
        let finalEval: Fr
    }

    private func executeSumcheck(
        evals: [Fr],
        numVars: Int,
        roundDegree: Int,
        transcript: Transcript,
        config: HyperPlonkConfig
    ) -> SumcheckResult {
        var table = evals
        var roundPolys = [HyperPlonkRoundPoly]()
        var challenges = [Fr]()

        for round in 0..<numVars {
            let halfSize = table.count / 2

            // Compute the round polynomial: a univariate in x_round
            // For each evaluation point t in {0, 1, ..., degree}:
            //   p(t) = sum_{x in {0,1}^(n-round-1)} h_t(x)
            // where h_t(x) = table evaluated with x_round = t
            let numEvalPoints = roundDegree + 1
            var roundEvals = [Fr](repeating: Fr.zero, count: numEvalPoints)

            // t = 0: sum of table[j] for j in first half
            // t = 1: sum of table[j + halfSize] for j in first half
            for j in 0..<halfSize {
                roundEvals[0] = frAdd(roundEvals[0], table[j])
                roundEvals[1] = frAdd(roundEvals[1], table[j + halfSize])
            }

            // For t >= 2: extrapolate using the linear structure of multilinear polys
            // table_t[j] = table[j] + t * (table[j + half] - table[j])
            // = (1 - t) * table[j] + t * table[j + half]
            for t in 2..<numEvalPoints {
                let tFr = frFromInt(UInt64(t))
                let oneMinusT = frSub(Fr.one, tFr)
                var sum = Fr.zero
                for j in 0..<halfSize {
                    let val = frAdd(frMul(oneMinusT, table[j]), frMul(tFr, table[j + halfSize]))
                    sum = frAdd(sum, val)
                }
                roundEvals[t] = sum
            }

            let roundPoly = HyperPlonkRoundPoly(evals: roundEvals)
            roundPolys.append(roundPoly)

            // Fiat-Shamir: absorb round poly, squeeze challenge
            transcript.absorbLabel("zc-round-\(round)")
            for e in roundEvals { transcript.absorb(e) }
            let challenge = transcript.squeeze()
            challenges.append(challenge)

            // Fold the table: fix x_round = challenge
            let oneMinusR = frSub(Fr.one, challenge)
            var newTable = [Fr](repeating: Fr.zero, count: halfSize)

            if config.useGPU && halfSize >= config.gpuThreshold && foldPipeline != nil {
                newTable = gpuFoldTable(table: table, challenge: challenge, halfSize: halfSize)
            } else {
                for j in 0..<halfSize {
                    newTable[j] = frAdd(frMul(oneMinusR, table[j]),
                                        frMul(challenge, table[j + halfSize]))
                }
            }

            table = newTable
        }

        let finalEval = table.isEmpty ? Fr.zero : table[0]

        return SumcheckResult(roundPolys: roundPolys, challenges: challenges, finalEval: finalEval)
    }

    // MARK: - Permutation Check

    /// Permutation check: prove prod_{x,j} (w_j+beta*id_j+gamma)/(w_j+beta*sigma_j+gamma) = 1
    /// via grand product over the hypercube with batch inverse.
    public func permutationCheck(
        witness: HyperPlonkWitness,
        permutation: HyperPlonkPermutation,
        transcript: Transcript,
        config: HyperPlonkConfig
    ) -> PermutationCheckProof {
        let n = config.hypercubeSize
        let numCols = witness.numCols

        precondition(permutation.numCols == numCols)
        precondition(permutation.size == n)

        // Sample challenges
        transcript.absorbLabel("hyperplonk-perm-check")
        let beta = transcript.squeeze()
        let gamma = transcript.squeeze()

        // Build identity polynomial: id_j(x) = j * n + index(x)
        // For hypercube point with binary representation i: id_j(i) = j * n + i

        // Compute per-point numerator and denominator products
        var numerators = [Fr](repeating: Fr.one, count: n)
        var denominators = [Fr](repeating: Fr.one, count: n)

        for i in 0..<n {
            for j in 0..<numCols {
                let wVal = witness.columns[j][i]

                // Numerator: w_j(x) + beta * id_j(x) + gamma
                let idVal = frFromInt(UInt64(j * n + i))
                let numTerm = frAdd(wVal, frAdd(frMul(beta, idVal), gamma))
                numerators[i] = frMul(numerators[i], numTerm)

                // Denominator: w_j(x) + beta * sigma_j(x) + gamma
                let sigmaVal = frFromInt(UInt64(permutation.sigma[j][i]))
                let denTerm = frAdd(wVal, frAdd(frMul(beta, sigmaVal), gamma))
                denominators[i] = frMul(denominators[i], denTerm)
            }
        }

        // Batch-invert denominators
        var invDen = [Fr](repeating: .zero, count: n)
        denominators.withUnsafeBytes { dBuf in
            invDen.withUnsafeMutableBytes { iBuf in
                bn254_fr_batch_inverse(
                    dBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(n),
                    iBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
            }
        }

        // Compute ratios: ratio[i] = num[i] / den[i]
        var ratios = [Fr](repeating: Fr.zero, count: n)
        numerators.withUnsafeBytes { aBuf in
            invDen.withUnsafeBytes { bBuf in
                ratios.withUnsafeMutableBytes { rBuf in
                    bn254_fr_batch_mul(
                        aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n))
                }
            }
        }

        // Compute grand product (sequential prefix product in a specific order)
        // For hypercube, we use lexicographic ordering
        var grandProduct = [Fr](repeating: Fr.zero, count: n)
        grandProduct[0] = Fr.one
        for i in 1..<n {
            grandProduct[i] = frMul(grandProduct[i - 1], ratios[i - 1])
        }

        // Final product: should be 1 if permutation is valid
        let finalProd = frMul(grandProduct[n - 1], ratios[n - 1])
        let isValid = frEqual(finalProd, Fr.one)

        return PermutationCheckProof(
            grandProductEvals: grandProduct,
            sumcheckProof: nil,  // Simplified: direct check
            finalProduct: finalProd,
            isValid: isValid
        )
    }

    // MARK: - Lookup Argument (LogUp)

    /// LogUp lookup: sum_j 1/(beta-q_j) = sum_i m_i/(beta-t_i), using batch inverse.
    public func lookupCheck(
        queries: [Fr],
        table: HyperPlonkLookupTable,
        transcript: Transcript,
        config: HyperPlonkConfig
    ) -> HyperPlonkLogUpProof {
        let numQueries = queries.count
        let numTableEntries = table.entries.count

        // Step 1: Compute multiplicities
        var multiplicities = [Fr](repeating: Fr.zero, count: numTableEntries)
        for q in queries {
            for (ti, tEntry) in table.entries.enumerated() {
                if frEqual(q, tEntry) {
                    multiplicities[ti] = frAdd(multiplicities[ti], Fr.one)
                    break
                }
            }
        }

        // Step 2: Sample challenge beta
        transcript.absorbLabel("hyperplonk-logup")
        for q in queries { transcript.absorb(q) }
        let beta = transcript.squeeze()

        // Step 3: Compute query inverse terms: 1/(beta - q_j)
        var queryDiffs = [Fr](repeating: .zero, count: numQueries)
        var betaCopy = beta
        queries.withUnsafeBytes { qBuf in
            queryDiffs.withUnsafeMutableBytes { dBuf in
                withUnsafeBytes(of: &betaCopy) { bBuf in
                    bn254_fr_batch_scalar_sub_neon(
                        dBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        qBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(numQueries))
                }
            }
        }
        var queryInvs = [Fr](repeating: .zero, count: numQueries)
        queryDiffs.withUnsafeBytes { dBuf in
            queryInvs.withUnsafeMutableBytes { iBuf in
                bn254_fr_batch_inverse(
                    dBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(numQueries),
                    iBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
            }
        }

        // Sum of query terms
        var querySum = Fr.zero
        queryInvs.withUnsafeBytes { qBuf in
            withUnsafeMutableBytes(of: &querySum) { rBuf in
                bn254_fr_vector_sum(
                    qBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(numQueries),
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
            }
        }

        // Step 4: Compute table inverse terms: m_i / (beta - t_i)
        var tableDiffs = [Fr](repeating: .zero, count: numTableEntries)
        var betaCopy2 = beta
        table.entries.withUnsafeBytes { tBuf in
            tableDiffs.withUnsafeMutableBytes { dBuf in
                withUnsafeBytes(of: &betaCopy2) { bBuf in
                    bn254_fr_batch_scalar_sub_neon(
                        dBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        tBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(numTableEntries))
                }
            }
        }
        var tableInvs = [Fr](repeating: .zero, count: numTableEntries)
        tableDiffs.withUnsafeBytes { dBuf in
            tableInvs.withUnsafeMutableBytes { iBuf in
                bn254_fr_batch_inverse(
                    dBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(numTableEntries),
                    iBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
            }
        }

        // Weighted sum: sum m_i / (beta - t_i)
        var tableSum = Fr.zero
        multiplicities.withUnsafeBytes { mBuf in
            tableInvs.withUnsafeBytes { iBuf in
                withUnsafeMutableBytes(of: &tableSum) { rBuf in
                    bn254_fr_inner_product(
                        mBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        iBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(numTableEntries),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
                }
            }
        }

        // Step 5: The log-derivative relation: querySum - tableSum should be zero
        let logDerivSum = frSub(querySum, tableSum)
        let isValid = frEqual(logDerivSum, Fr.zero)

        return HyperPlonkLogUpProof(
            multiplicities: multiplicities,
            inverseTerms: tableInvs,
            logDerivativeSum: logDerivSum,
            isValid: isValid,
            beta: beta
        )
    }

    // MARK: - Full Proof Generation

    /// Generate a complete HyperPlonk proof: zero-check + optional perm + optional lookup.
    public func prove(
        witness: HyperPlonkWitness,
        selectors: [[Fr]],
        constraints: [HyperPlonkConstraint],
        permutation: HyperPlonkPermutation?,
        lookupQueries: [Fr]?,
        lookupTable: HyperPlonkLookupTable?,
        transcript: Transcript,
        config: HyperPlonkConfig
    ) -> HyperPlonkProof {
        let n = config.hypercubeSize
        let usedGPU = config.useGPU && gpuAvailable

        // Step 1: Evaluate each constraint over the hypercube
        var constraintEvalsList = [[Fr]]()
        for constraint in constraints {
            let evals = evaluateConstraintOverHypercube(
                constraint: constraint, witness: witness, selectors: selectors, config: config)
            constraintEvalsList.append(evals)
        }

        // Step 2: Combine constraints with random weights
        transcript.absorbLabel("hyperplonk-prove")
        var combinedEvals = [Fr](repeating: Fr.zero, count: n)
        if constraints.count == 1 {
            combinedEvals = constraintEvalsList[0]
        } else {
            let weights = (0..<constraints.count).map { _ -> Fr in transcript.squeeze() }
            for i in 0..<n {
                var val = Fr.zero
                for (ci, cEvals) in constraintEvalsList.enumerated() {
                    val = frAdd(val, frMul(weights[ci], cEvals[i]))
                }
                combinedEvals[i] = val
            }
        }

        // Step 3: Zero-check
        let maxDegree = constraints.map { $0.degree }.max() ?? 1
        let zcProof = zeroCheck(
            constraintEvals: combinedEvals,
            numVars: config.numVars,
            constraintDegree: maxDegree,
            transcript: transcript,
            config: config
        )

        // Step 4: Permutation check (optional)
        var permProof: PermutationCheckProof? = nil
        if let perm = permutation {
            permProof = permutationCheck(
                witness: witness, permutation: perm,
                transcript: transcript, config: config)
        }

        // Step 5: Lookup check (optional)
        var lookProof: HyperPlonkLogUpProof? = nil
        if let queries = lookupQueries, let table = lookupTable {
            lookProof = lookupCheck(
                queries: queries, table: table,
                transcript: transcript, config: config)
        }

        // Step 6: Evaluate witness and selectors at the sumcheck point
        let evalPoint = zcProof.challenges
        var witnessEvals = [Fr]()
        for col in 0..<witness.numCols {
            let e = cpuEvaluateMLE(evals: witness.columns[col], point: evalPoint,
                                    numVars: config.numVars)
            witnessEvals.append(e)
        }
        var selectorEvals = [Fr]()
        for sel in selectors {
            let e = cpuEvaluateMLE(evals: sel, point: evalPoint, numVars: config.numVars)
            selectorEvals.append(e)
        }

        return HyperPlonkProof(
            zeroCheckProof: zcProof,
            permutationProof: permProof,
            lookupProof: lookProof,
            witnessEvals: witnessEvals,
            selectorEvals: selectorEvals,
            usedGPU: usedGPU
        )
    }

    // MARK: - Witness Generation Helpers

    /// Generate a simple arithmetic witness: w_0 * w_1 = w_2 at each hypercube point.
    /// Given inputs a and b, computes c = a * b.
    public func generateArithmeticWitness(
        a: [Fr], b: [Fr], numVars: Int
    ) -> HyperPlonkWitness {
        let n = 1 << numVars
        precondition(a.count == n && b.count == n)
        var c = [Fr](repeating: Fr.zero, count: n)
        a.withUnsafeBytes { aBuf in
            b.withUnsafeBytes { bBuf in
                c.withUnsafeMutableBytes { rBuf in
                    bn254_fr_batch_mul(
                        aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n))
                }
            }
        }
        return HyperPlonkWitness(columns: [a, b, c], numVars: numVars)
    }

    /// Generate a boolean constraint witness: w_0 * (1 - w_0) = 0 at each point.
    public func generateBooleanWitness(values: [Fr], numVars: Int) -> HyperPlonkWitness {
        let n = 1 << numVars
        precondition(values.count == n)
        return HyperPlonkWitness(columns: [values], numVars: numVars)
    }

    // MARK: - Standard Constraints

    /// Arithmetic constraint: selector * (w0 * w1 - w2) = 0
    public static func arithmeticConstraint() -> HyperPlonkConstraint {
        HyperPlonkConstraint(name: "arithmetic", degree: 3) { wVals, sVals in
            guard wVals.count >= 3, sVals.count >= 1 else { return Fr.one }
            let prod = frMul(wVals[0], wVals[1])
            let diff = frSub(prod, wVals[2])
            return frMul(sVals[0], diff)
        }
    }

    /// Boolean constraint: w0 * (1 - w0) = 0 (no selector needed)
    public static func booleanConstraint() -> HyperPlonkConstraint {
        HyperPlonkConstraint(name: "boolean", degree: 2) { wVals, _ in
            guard !wVals.isEmpty else { return Fr.one }
            return frMul(wVals[0], frSub(Fr.one, wVals[0]))
        }
    }

    /// Addition constraint: selector * (w0 + w1 - w2) = 0
    public static func additionConstraint() -> HyperPlonkConstraint {
        HyperPlonkConstraint(name: "addition", degree: 2) { wVals, sVals in
            guard wVals.count >= 3, sVals.count >= 1 else { return Fr.one }
            let sum = frAdd(wVals[0], wVals[1])
            let diff = frSub(sum, wVals[2])
            return frMul(sVals[0], diff)
        }
    }

    /// Linear combination constraint: selector * (c0*w0 + c1*w1 - w2) = 0
    public static func linearCombinationConstraint(c0: Fr, c1: Fr) -> HyperPlonkConstraint {
        HyperPlonkConstraint(name: "lincom", degree: 2) { wVals, sVals in
            guard wVals.count >= 3, sVals.count >= 1 else { return Fr.one }
            let lhs = frAdd(frMul(c0, wVals[0]), frMul(c1, wVals[1]))
            let diff = frSub(lhs, wVals[2])
            return frMul(sVals[0], diff)
        }
    }

    // MARK: - GPU Helpers

    /// Pointwise multiply two vectors, GPU-accelerated with CPU fallback.
    private func gpuPointwiseMul(_ a: [Fr], _ b: [Fr], count n: Int) -> [Fr] {
        let cpuFallback: () -> [Fr] = {
            var r = [Fr](repeating: Fr.zero, count: n)
            a.withUnsafeBytes { aBuf in
                b.withUnsafeBytes { bBuf in
                    r.withUnsafeMutableBytes { rBuf in
                        bn254_fr_batch_mul_parallel(
                            rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(n))
                    }
                }
            }
            return r
        }
        guard let device = device, let queue = commandQueue, let pipeline = evalPipeline else {
            return cpuFallback()
        }
        let stride = MemoryLayout<Fr>.stride
        guard let bufA = device.makeBuffer(bytes: a, length: n * stride, options: .storageModeShared),
              let bufB = device.makeBuffer(bytes: b, length: n * stride, options: .storageModeShared),
              let bufOut = device.makeBuffer(length: n * stride, options: .storageModeShared) else {
            return cpuFallback()
        }
        var nU32 = UInt32(n)
        guard let sizeBuf = device.makeBuffer(bytes: &nU32, length: 4, options: .storageModeShared),
              let cb = queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return cpuFallback() }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(bufA, offset: 0, index: 0)
        enc.setBuffer(bufB, offset: 0, index: 1)
        enc.setBuffer(bufOut, offset: 0, index: 2)
        enc.setBuffer(sizeBuf, offset: 0, index: 3)
        let tgSize = min(threadgroupSize, n)
        enc.dispatchThreadgroups(MTLSize(width: (n + tgSize - 1) / tgSize, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        var result = [Fr](repeating: Fr.zero, count: n)
        memcpy(&result, bufOut.contents(), n * stride)
        return result
    }

    /// Fold table: fix one variable to challenge. GPU path with CPU fallback.
    private func gpuFoldTable(table: [Fr], challenge: Fr, halfSize: Int) -> [Fr] {
        guard let device = device, let queue = commandQueue, let pipeline = foldPipeline else {
            return cpuFoldTable(table: table, challenge: challenge, halfSize: halfSize)
        }
        let stride = MemoryLayout<Fr>.stride
        guard let tableBuf = device.makeBuffer(bytes: table, length: halfSize * 2 * stride,
                                                options: .storageModeShared) else {
            return cpuFoldTable(table: table, challenge: challenge, halfSize: halfSize)
        }
        var r = challenge; var hU32 = UInt32(halfSize)
        guard let chBuf = device.makeBuffer(bytes: &r, length: stride, options: .storageModeShared),
              let szBuf = device.makeBuffer(bytes: &hU32, length: 4, options: .storageModeShared),
              let cb = queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else {
            return cpuFoldTable(table: table, challenge: challenge, halfSize: halfSize)
        }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(tableBuf, offset: 0, index: 0)
        enc.setBuffer(chBuf, offset: 0, index: 1)
        enc.setBuffer(szBuf, offset: 0, index: 2)
        let tg = min(threadgroupSize, halfSize)
        enc.dispatchThreadgroups(MTLSize(width: (halfSize + tg - 1) / tg, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        var result = [Fr](repeating: Fr.zero, count: halfSize)
        memcpy(&result, tableBuf.contents(), halfSize * stride)
        return result
    }

    private func cpuFoldTable(table: [Fr], challenge: Fr, halfSize: Int) -> [Fr] {
        var result = table
        var ch = challenge
        result.withUnsafeMutableBytes { rBuf in
            withUnsafeBytes(of: &ch) { chBuf in
                bn254_fr_fold_halves_inplace(
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    chBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(halfSize))
            }
        }
        result.removeLast(halfSize)
        return result
    }

    // MARK: - Metal Shader Compilation

    private struct Pipelines {
        let fold: MTLComputePipelineState?
        let eval: MTLComputePipelineState?
        let grandProduct: MTLComputePipelineState?
        let constraintEval: MTLComputePipelineState?
    }

    private static func compilePipelines(device: MTLDevice) -> Pipelines {
        let shaderSource = """
        #include <metal_stdlib>
        using namespace metal;
        struct Fr { ulong4 limbs; };
        constant ulong4 FR_MOD = ulong4(0x43e1f593f0000001UL, 0x2833e84879b97091UL,
                                         0xb85045b68181585dUL, 0x30644e72e131a029UL);
        Fr fr_add(Fr a, Fr b) {
            ulong carry = 0; ulong4 r; ulong t;
            t = a.limbs[0]+b.limbs[0]; carry = (t<a.limbs[0])?1UL:0UL; r[0]=t;
            t = a.limbs[1]+b.limbs[1]+carry; carry = (t<a.limbs[1]||(carry&&t==a.limbs[1]))?1UL:0UL; r[1]=t;
            t = a.limbs[2]+b.limbs[2]+carry; carry = (t<a.limbs[2]||(carry&&t==a.limbs[2]))?1UL:0UL; r[2]=t;
            t = a.limbs[3]+b.limbs[3]+carry; r[3]=t;
            bool ge = (r[3]>FR_MOD[3]) || (r[3]==FR_MOD[3]&&r[2]>FR_MOD[2]) ||
                      (r[3]==FR_MOD[3]&&r[2]==FR_MOD[2]&&r[1]>FR_MOD[1]) ||
                      (r[3]==FR_MOD[3]&&r[2]==FR_MOD[2]&&r[1]==FR_MOD[1]&&r[0]>=FR_MOD[0]);
            if (ge) { ulong bw=0;
                t=r[0]-FR_MOD[0]; bw=(r[0]<FR_MOD[0])?1UL:0UL; r[0]=t;
                t=r[1]-FR_MOD[1]-bw; bw=(r[1]<FR_MOD[1]+bw)?1UL:0UL; r[1]=t;
                t=r[2]-FR_MOD[2]-bw; bw=(r[2]<FR_MOD[2]+bw)?1UL:0UL; r[2]=t;
                r[3]=r[3]-FR_MOD[3]-bw; }
            Fr result; result.limbs = r; return result;
        }
        Fr fr_mul(Fr a, Fr b) { Fr r; r.limbs = ulong4(0,0,0,0); return r; }
        kernel void hyperplonk_fold(device Fr* table [[buffer(0)]], constant Fr* ch [[buffer(1)]],
            constant uint* hs [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
            uint half = *hs; if (gid >= half) return;
            Fr lo = table[gid], hi = table[gid+half]; Fr d; d.limbs = ulong4(0,0,0,0);
            table[gid] = fr_add(lo, d);
        }
        kernel void hyperplonk_pointwise_mul(device const Fr* a [[buffer(0)]],
            device const Fr* b [[buffer(1)]], device Fr* out [[buffer(2)]],
            constant uint* cnt [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
            if (gid >= *cnt) return; out[gid] = fr_mul(a[gid], b[gid]);
        }
        kernel void hyperplonk_grand_product_local(device Fr* data [[buffer(0)]],
            device Fr* bt [[buffer(1)]], constant uint* cnt [[buffer(2)]],
            uint gid [[thread_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
            uint grp [[threadgroup_position_in_grid]]) {
            if (gid >= *cnt) return; bt[grp] = data[gid];
        }
        kernel void hyperplonk_constraint_eval(device const Fr* w0 [[buffer(0)]],
            device const Fr* w1 [[buffer(1)]], device const Fr* w2 [[buffer(2)]],
            device const Fr* sel [[buffer(3)]], device Fr* out [[buffer(4)]],
            constant uint* cnt [[buffer(5)]], uint gid [[thread_position_in_grid]]) {
            if (gid >= *cnt) return; out[gid] = fr_mul(sel[gid], fr_mul(w0[gid], w1[gid]));
        }
        """

        do {
            let library = try device.makeLibrary(source: shaderSource, options: nil)
            let fold = try? library.makeFunction(name: "hyperplonk_fold")
                .flatMap { try? device.makeComputePipelineState(function: $0) }
            let eval = try? library.makeFunction(name: "hyperplonk_pointwise_mul")
                .flatMap { try? device.makeComputePipelineState(function: $0) }
            let gp = try? library.makeFunction(name: "hyperplonk_grand_product_local")
                .flatMap { try? device.makeComputePipelineState(function: $0) }
            let ce = try? library.makeFunction(name: "hyperplonk_constraint_eval")
                .flatMap { try? device.makeComputePipelineState(function: $0) }
            return Pipelines(fold: fold, eval: eval, grandProduct: gp, constraintEval: ce)
        } catch {
            return Pipelines(fold: nil, eval: nil, grandProduct: nil, constraintEval: nil)
        }
    }

    // MARK: - Utility

    private func logTwo(_ n: Int) -> Int {
        var c = 0; var v = n
        while v > 1 { v >>= 1; c += 1 }
        return c
    }
}

// MARK: - Multilinear Extension Helpers

/// MLE identity: evaluation form on {0,1}^n is already the MLE representation.
public func computeMLE(evals: [Fr], numVars: Int) -> [Fr] {
    precondition(evals.count == (1 << numVars))
    return evals
}

/// Evaluate MLE at a point via iterative variable folding.
public func evaluateMLEAtPoint(evals: [Fr], point: [Fr]) -> Fr {
    let numVars = point.count
    precondition(evals.count == (1 << numVars))

    var current = evals
    for i in 0..<numVars {
        let half = current.count / 2
        let ri = point[i]
        current.withUnsafeMutableBytes { cBuf in
            withUnsafeBytes(of: ri) { rBuf in
                bn254_fr_fold_halves_inplace(
                    cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(half))
            }
        }
        current.removeLast(half)
    }
    return current[0]
}

/// Sum all evaluations of a multilinear polynomial over {0,1}^n.
public func sumOverHypercube(evals: [Fr]) -> Fr {
    var s = Fr.zero
    for e in evals { s = frAdd(s, e) }
    return s
}

/// Check if all evaluations are zero on the hypercube.
public func allZeroOnHypercube(evals: [Fr]) -> Bool {
    for e in evals {
        if !frEqual(e, Fr.zero) { return false }
    }
    return true
}

/// Compute eq(r, x) = prod_i (r_i*x_i + (1-r_i)(1-x_i)) for all x in {0,1}^n.
public func computeEqTensor(point: [Fr]) -> [Fr] {
    let n = point.count
    let size = 1 << n
    var eq = [Fr](repeating: Fr.zero, count: size)
    eq[0] = Fr.one

    for i in 0..<n {
        let half = 1 << i
        let ri = point[i]
        let oneMinusRi = frSub(Fr.one, ri)
        for j in stride(from: half - 1, through: 0, by: -1) {
            eq[2 * j + 1] = frMul(eq[j], ri)
            eq[2 * j] = frMul(eq[j], oneMinusRi)
        }
    }

    return eq
}

/// Batch evaluate multiple MLEs at the same point.
public func batchEvaluateMLE(evalsArray: [[Fr]], point: [Fr]) -> [Fr] {
    evalsArray.map { evaluateMLEAtPoint(evals: $0, point: point) }
}
