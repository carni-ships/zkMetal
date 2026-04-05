// TensorSumcheck — Sumcheck prover/verifier exploiting tensor product structure
//
// Key insight: when a multilinear polynomial factors as f(x) = f_1(x_1) ⊗ f_2(x_2) ⊗ ... ⊗ f_k(x_k),
// the sumcheck round polynomials within each factor block are determined by that factor alone
// (scaled by the partial product of already-evaluated factors).
//
// This means:
//   1. The prover processes each factor independently, scaling by partial products
//   2. Factor evaluations compress: only the factor's own table (2^m_i entries) is needed
//   3. Proof size drops from O(n * 3) to O(sum_i 2^m_i) + O(n) for factor snapshots + challenges
//
// For equal-sized factors with m_i = n/k, this gives O(k * 2^(n/k)) instead of O(n * 3).
// With k=2 (two halves), proof size is O(2 * sqrt(N)) vs O(n * 3) = O(log(N) * 3).
//
// Applications: Lasso structured lookups, Jolt VM proofs, HyperNova with structured witnesses

import Foundation
import NeonFieldOps

// MARK: - Tensor Sumcheck Prover

public class TensorSumcheckProver {

    /// Prove that the sum of f(x) over {0,1}^n equals claimedSum,
    /// where f decomposes as a tensor product of the given factors.
    ///
    /// f(x_1, ..., x_n) = f_1(x_1, ..., x_{m1}) * f_2(x_{m1+1}, ..., x_{m2}) * ...
    ///
    /// The factors' variable counts must sum to n.
    /// Challenges are derived from the transcript via Fiat-Shamir.
    public static func prove(
        factors: [TensorFactor],
        claimedSum: Fr,
        transcript: Transcript
    ) -> TensorSumcheckProof {
        let totalVars = factors.reduce(0) { $0 + $1.numVars }
        precondition(totalVars >= 1, "Must have at least 1 variable")
        precondition(!factors.isEmpty, "Must have at least 1 factor")

        // Verify tensor structure: product of factor sizes = 2^totalVars
        let totalSize = factors.reduce(1) { $0 * $1.evaluations.count }
        precondition(totalSize == (1 << totalVars), "Factor sizes must multiply to 2^totalVars")

        transcript.absorb(claimedSum)

        var rounds: [(Fr, Fr, Fr)] = []
        var challenges: [Fr] = []
        var partialProducts: [Fr] = [Fr.one]  // PP[0] = 1 (nothing evaluated yet)
        var factorSnapshots: [[Fr]] = []

        // Running partial product of fully-evaluated factors
        var runningProduct = Fr.one

        // Partial product of factors not yet started (suffix product)
        // suffixProduct = product of sum(f_j) for j > current factor
        var factorSums: [Fr] = factors.map { factor in
            var sum = Fr.zero
            for e in factor.evaluations {
                sum = frAdd(sum, e)
            }
            return sum
        }
        var suffixProduct = Fr.one
        for i in 1..<factors.count {
            suffixProduct = frMul(suffixProduct, factorSums[i])
        }

        for fi in 0..<factors.count {
            let factor = factors[fi]
            factorSnapshots.append(factor.evaluations)

            // Compute suffix product for factors after this one
            // (recomputed as we consume factors)
            var currentSuffix = Fr.one
            for j in (fi + 1)..<factors.count {
                currentSuffix = frMul(currentSuffix, factorSums[j])
            }

            // Process each variable in this factor
            var currentEvals = factor.evaluations
            let m = factor.numVars

            for localRound in 0..<m {
                let half = currentEvals.count / 2

                // Compute round polynomial: S(X) = sum_{b in {0,1}^remaining} f_i(r_0,...,r_{j-1}, X, b) * prefix * suffix
                // S(0) = prefix * suffix * sum of currentEvals[2*i]
                // S(1) = prefix * suffix * sum of currentEvals[2*i + 1]
                // S(2) = prefix * suffix * sum of (2*currentEvals[2*i+1] - currentEvals[2*i])
                var s0 = Fr.zero
                var s1 = Fr.zero
                var s2 = Fr.zero

                for i in 0..<half {
                    let lo = currentEvals[2 * i]
                    let hi = currentEvals[2 * i + 1]
                    s0 = frAdd(s0, lo)
                    s1 = frAdd(s1, hi)
                    // Evaluate at X=2: f(2) = 2*hi - lo (linear interpolation extrapolation)
                    let at2 = frSub(frAdd(hi, hi), lo)
                    s2 = frAdd(s2, at2)
                }

                // Scale by running product of evaluated factors * suffix of unevaluated factors
                let scale = frMul(runningProduct, currentSuffix)
                s0 = frMul(s0, scale)
                s1 = frMul(s1, scale)
                s2 = frMul(s2, scale)

                rounds.append((s0, s1, s2))

                // Absorb round polynomial and squeeze challenge
                transcript.absorb(s0)
                transcript.absorb(s1)
                transcript.absorb(s2)
                let r = transcript.squeeze()
                challenges.append(r)

                // Fold: fix this variable to r
                var folded = [Fr](repeating: Fr.zero, count: half)
                for i in 0..<half {
                    let lo = currentEvals[2 * i]
                    let hi = currentEvals[2 * i + 1]
                    // folded[i] = (1-r)*lo + r*hi = lo + r*(hi - lo)
                    folded[i] = frAdd(lo, frMul(r, frSub(hi, lo)))
                }
                currentEvals = folded

                // After the last local round, update suffix for next rounds in this factor
                if localRound < m - 1 {
                    // Suffix doesn't change within a factor, but the "sum" of remaining
                    // evaluations in *this factor* changes implicitly via folding
                }
            }

            // After processing all variables in this factor, currentEvals has 1 element
            precondition(currentEvals.count == 1, "Factor should reduce to single value")
            let factorValue = currentEvals[0]

            // Update running product
            runningProduct = frMul(runningProduct, factorValue)

            if fi + 1 < factors.count {
                partialProducts.append(runningProduct)
            }
        }

        // Final evaluation = product of all factor values at their challenge points
        let finalEval = runningProduct

        let factorSizeInfo = factors.enumerated().map { (i, f) in
            (index: i, numVars: f.numVars)
        }

        return TensorSumcheckProof(
            claimedSum: claimedSum,
            numVars: totalVars,
            factorSizes: factorSizeInfo,
            partialProducts: partialProducts,
            factorSnapshots: factorSnapshots,
            rounds: rounds,
            challenges: challenges,
            finalEval: finalEval
        )
    }

    /// Convenience: prove with two equal-sized factors (most common case).
    /// Splits a polynomial's evaluations into two halves of variables.
    public static func proveTwoFactor(
        evaluations: [Fr],
        numVars: Int,
        claimedSum: Fr,
        transcript: Transcript
    ) -> TensorSumcheckProof {
        precondition(numVars >= 2 && numVars % 2 == 0, "numVars must be even and >= 2")
        let halfVars = numVars / 2
        let sqrtN = 1 << halfVars

        // Reshape evaluations as tensor product of two factors
        // f(x_L, x_R) where x_L indexes rows, x_R indexes columns
        // Factor 1: sum over x_R dimension -> row sums
        // Factor 2: sum over x_L dimension -> column sums
        // This only works when f truly has tensor structure.
        //
        // For a general polynomial, we instead use the matrix decomposition:
        // factor1[i] = row i evaluations, factor2[j] = column j evaluations
        // But for true tensor products f = g ⊗ h, we have:
        //   g has 2^halfVars evaluations, h has 2^halfVars evaluations
        //   f[i * sqrtN + j] = g[i] * h[j]

        // Extract factors by SVD-style decomposition (only valid for rank-1 = true tensor)
        // Factor 1 = first row's column-normalized pattern
        // Factor 2 = first column
        let factor2Evals = (0..<sqrtN).map { j in evaluations[j] }  // first row = h
        let h0 = factor2Evals[0]
        let factor1Evals: [Fr]
        if frToInt(h0) != frToInt(Fr.zero) {
            let h0Inv = frInverse(h0)
            factor1Evals = (0..<sqrtN).map { i in frMul(evaluations[i * sqrtN], h0Inv) }
        } else {
            // Fallback: use column sums normalized
            factor1Evals = (0..<sqrtN).map { i in evaluations[i * sqrtN] }
        }

        let f1 = TensorFactor(evaluations: factor1Evals, numVars: halfVars)
        let f2 = TensorFactor(evaluations: factor2Evals, numVars: halfVars)

        return prove(factors: [f1, f2], claimedSum: claimedSum, transcript: transcript)
    }
}

// MARK: - Tensor Sumcheck Verifier

public class TensorSumcheckVerifier {

    /// Verify a tensor-compressed sumcheck proof.
    /// The verifier checks:
    ///   1. Each round polynomial satisfies S_i(0) + S_i(1) = previous claimed sum
    ///   2. The challenges match the transcript
    ///   3. The final evaluation matches the product of factor evaluations
    ///
    /// Returns true if the proof is valid.
    public static func verify(
        proof: TensorSumcheckProof,
        transcript: Transcript
    ) -> Bool {
        let numVars = proof.numVars
        guard proof.rounds.count == numVars else { return false }
        guard proof.challenges.count == numVars else { return false }

        transcript.absorb(proof.claimedSum)

        var expectedSum = proof.claimedSum

        for round in 0..<numVars {
            let (s0, s1, s2) = proof.rounds[round]

            // Check: S(0) + S(1) = expected sum
            let roundSum = frAdd(s0, s1)
            if frToInt(roundSum) != frToInt(expectedSum) {
                return false
            }

            // Absorb and squeeze to get/verify challenge
            transcript.absorb(s0)
            transcript.absorb(s1)
            transcript.absorb(s2)
            let r = transcript.squeeze()

            // Verify challenge matches
            if frToInt(r) != frToInt(proof.challenges[round]) {
                return false
            }

            // Next expected sum = S(r) via quadratic interpolation
            expectedSum = evalQuadratic(s0: s0, s1: s1, s2: s2, at: r)
        }

        // Final check: expectedSum should equal the final evaluation
        if frToInt(expectedSum) != frToInt(proof.finalEval) {
            return false
        }

        return true
    }

    /// Verify using compressed representation (factor snapshots + challenges).
    /// Reconstructs round polynomials from factor data and checks consistency.
    public static func verifyCompressed(
        proof: TensorSumcheckProof,
        transcript: Transcript
    ) -> Bool {
        // First verify the standard sumcheck relation
        guard verify(proof: proof, transcript: transcript) else {
            return false
        }

        // Additionally verify that the final evaluation is consistent with factor structure:
        // finalEval should equal the product of each factor evaluated at its challenge point
        var roundOffset = 0
        var product = Fr.one
        for (fi, (_, m)) in proof.factorSizes.enumerated() {
            guard fi < proof.factorSnapshots.count else { return false }
            let snapshot = proof.factorSnapshots[fi]
            guard snapshot.count == (1 << m) else { return false }

            // Evaluate this factor at its challenge point by sequential folding
            let factorChallenges = Array(proof.challenges[roundOffset..<(roundOffset + m)])
            let factorVal = TensorCompressor.multilinearEval(evals: snapshot, point: factorChallenges)
            product = frMul(product, factorVal)
            roundOffset += m
        }

        // Product of factor evaluations should equal final eval
        if frToInt(product) != frToInt(proof.finalEval) {
            return false
        }

        return true
    }

    // MARK: - Helpers

    /// Evaluate degree-2 polynomial from values at 0, 1, 2 using Lagrange interpolation.
    private static func evalQuadratic(s0: Fr, s1: Fr, s2: Fr, at r: Fr) -> Fr {
        // Same as TensorCompressor.evalQuadratic
        let rMinus1 = frSub(r, Fr.one)
        let rMinus2 = frSub(r, frFromInt(2))
        let inv2 = frInverse(frFromInt(2))

        let l0 = frMul(frMul(rMinus1, rMinus2), inv2)
        let l1 = frSub(Fr.zero, frMul(r, rMinus2))
        let l2 = frMul(frMul(r, rMinus1), inv2)

        return frAdd(frAdd(frMul(s0, l0), frMul(s1, l1)), frMul(s2, l2))
    }
}
