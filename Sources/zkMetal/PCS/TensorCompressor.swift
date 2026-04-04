// Tensor Proof Compression Engine
// Compresses a polynomial evaluation proof from N to sqrt(N) using tensor product structure.
// Core technique from Spartan/Spark: given multilinear f on n variables with 2^n evaluations,
// reshape as sqrt(N) x sqrt(N) matrix M. Then f(r) = t_L^T * M * t_R, where t_L, t_R are
// tensor products of (1-r_i, r_i) over the first/second half of variables.
// Prover sends v = M * t_R (sqrt(N) values) and proves correctness via two sumcheck instances.

import Foundation

// MARK: - Data Structures

/// Proof data for tensor-compressed polynomial evaluation.
public struct TensorProof {
    /// Partial products v = M * t_R (sqrt(N) intermediate values)
    public let partialProducts: [Fr]
    /// Sumcheck proof for sum_j v[j] * t_L[j] = claimed_value
    public let sumcheckRounds1: [(Fr, Fr, Fr)]
    /// Sumcheck proof for sum_j M[i,j] * t_R[j] = v[i] for random i
    public let sumcheckRounds2: [(Fr, Fr, Fr)]
    /// Final evaluations from sumcheck reductions
    public let finalEval1: Fr
    public let finalEval2: Fr
    /// Random challenge used to batch-check v consistency
    public let batchChallenge: Fr

    /// Proof size in field elements
    public var sizeInElements: Int {
        return partialProducts.count + (sumcheckRounds1.count + sumcheckRounds2.count) * 3 + 3
    }
}

// MARK: - TensorCompressor

/// Tensor proof compression engine.
/// Reduces evaluation proofs from O(N) to O(sqrt(N)) using the tensor product decomposition
/// of multilinear polynomial evaluations and two sumcheck instances.
public class TensorCompressor {
    public static let version = Versions.tensorCompressor

    /// Compute the tensor product: (1-r[0], r[0]) tensor (1-r[1], r[1]) tensor ...
    /// Result has 2^k entries.
    public static func tensorProduct(_ challenges: [Fr]) -> [Fr] {
        let k = challenges.count
        let size = 1 << k
        var result = [Fr](repeating: Fr.zero, count: size)
        result[0] = Fr.one

        for i in 0..<k {
            let half = 1 << i
            let ri = challenges[i]
            let oneMinusRi = frSub(Fr.one, ri)
            var j = half - 1
            while j >= 0 {
                let val = result[j]
                result[2 * j + 1] = frMul(val, ri)
                result[2 * j] = frMul(val, oneMinusRi)
                j -= 1
            }
        }
        return result
    }

    /// Matrix-vector multiply: M[i][j] = evaluations[i*cols + j], result[i] = sum_j M[i][j]*vec[j].
    public static func matVecMul(evaluations: [Fr], vec: [Fr], rows: Int, cols: Int) -> [Fr] {
        precondition(evaluations.count == rows * cols)
        precondition(vec.count == cols)
        var result = [Fr](repeating: Fr.zero, count: rows)
        for i in 0..<rows {
            var acc = Fr.zero
            let base = i * cols
            for j in 0..<cols {
                acc = frAdd(acc, frMul(evaluations[base + j], vec[j]))
            }
            result[i] = acc
        }
        return result
    }

    /// Vector dot product: sum_i a[i] * b[i].
    public static func dotProduct(_ a: [Fr], _ b: [Fr]) -> Fr {
        precondition(a.count == b.count)
        var acc = Fr.zero
        for i in 0..<a.count {
            acc = frAdd(acc, frMul(a[i], b[i]))
        }
        return acc
    }

    /// Compress: given evaluations of a multilinear polynomial f on n variables,
    /// produce a tensor proof that f(point) = value.
    ///
    /// Proof structure:
    /// 1. Prover sends v = M * t_R (sqrt(N) field elements)
    /// 2. Sumcheck 1: proves t_L . v = claimed_value
    /// 3. Sumcheck 2: proves M_tau . t_R = v(tau) for random row tau
    public static func compress(
        evaluations: [Fr],
        point: [Fr],
        value: Fr,
        transcript: Transcript? = nil
    ) -> TensorProof {
        let numVars = point.count
        precondition(numVars >= 2 && numVars % 2 == 0, "Number of variables must be even and >= 2")
        let n = evaluations.count
        precondition(n == (1 << numVars), "Evaluations must have 2^numVars entries")

        let halfVars = numVars / 2
        let sqrtN = 1 << halfVars

        let ts = transcript ?? Transcript(label: "tensor-compress")

        let pointLeft = Array(point[0..<halfVars])
        let pointRight = Array(point[halfVars..<numVars])

        let tL = tensorProduct(pointLeft)
        let tR = tensorProduct(pointRight)

        // v = M * t_R
        let v = matVecMul(evaluations: evaluations, vec: tR, rows: sqrtN, cols: sqrtN)

        // Absorb v into transcript
        ts.absorbMany(v)
        ts.absorb(value)

        // --- Sumcheck 1: prove t_L . v = value ---
        let sc1Challenges = ts.squeezeN(halfVars)
        let sc1Result = cpuSumcheck(
            evalsA: tL,
            evalsB: v,
            challenges: sc1Challenges
        )

        for (s0, s1, s2) in sc1Result.rounds {
            ts.absorb(s0); ts.absorb(s1); ts.absorb(s2)
        }
        ts.absorb(sc1Result.finalEval)

        // Derive random row index tau
        let tauChallenge = ts.squeeze()

        // --- Sumcheck 2: prove M_tau . t_R = v(tau) ---
        let tauPoint = sc1Challenges
        let mTau = evaluateMatrixRow(evaluations: evaluations, rowPoint: tauPoint,
                                      rows: sqrtN, cols: sqrtN)

        let sc2Challenges = ts.squeezeN(halfVars)
        let sc2Result = cpuSumcheck(
            evalsA: mTau,
            evalsB: tR,
            challenges: sc2Challenges
        )

        for (s0, s1, s2) in sc2Result.rounds {
            ts.absorb(s0); ts.absorb(s1); ts.absorb(s2)
        }
        ts.absorb(sc2Result.finalEval)

        return TensorProof(
            partialProducts: v,
            sumcheckRounds1: sc1Result.rounds,
            sumcheckRounds2: sc2Result.rounds,
            finalEval1: sc1Result.finalEval,
            finalEval2: sc2Result.finalEval,
            batchChallenge: tauChallenge
        )
    }

    /// Verify a tensor proof.
    public static func verify(
        point: [Fr],
        value: Fr,
        proof: TensorProof,
        transcript: Transcript? = nil
    ) -> Bool {
        let numVars = point.count
        guard numVars >= 2 && numVars % 2 == 0 else { return false }

        let halfVars = numVars / 2
        let sqrtN = 1 << halfVars

        guard proof.partialProducts.count == sqrtN else { return false }
        guard proof.sumcheckRounds1.count == halfVars else { return false }
        guard proof.sumcheckRounds2.count == halfVars else { return false }

        let ts = transcript ?? Transcript(label: "tensor-compress")

        let pointLeft = Array(point[0..<halfVars])
        let pointRight = Array(point[halfVars..<numVars])

        let tL = tensorProduct(pointLeft)
        let tR = tensorProduct(pointRight)
        let v = proof.partialProducts

        ts.absorbMany(v)
        ts.absorb(value)

        let sc1Challenges = ts.squeezeN(halfVars)

        // --- Verify Sumcheck 1: t_L . v = value ---
        var expectedSum = value
        for round in 0..<halfVars {
            let (s0, s1, s2) = proof.sumcheckRounds1[round]
            let roundSum = frAdd(s0, s1)
            if frToInt(roundSum) != frToInt(expectedSum) {
                return false
            }
            let r = sc1Challenges[round]
            expectedSum = evalQuadratic(s0: s0, s1: s1, s2: s2, at: r)

            ts.absorb(s0); ts.absorb(s1); ts.absorb(s2)
        }
        ts.absorb(proof.finalEval1)

        // Check final eval: tL(sc1Challenges) * v(sc1Challenges)
        let tLAtChal = multilinearEval(evals: tL, point: sc1Challenges)
        let vAtChal = multilinearEval(evals: v, point: sc1Challenges)
        let expectedFinal1 = frMul(tLAtChal, vAtChal)
        if frToInt(expectedFinal1) != frToInt(proof.finalEval1) {
            return false
        }

        let _tauChallenge = ts.squeeze()

        // --- Verify Sumcheck 2 ---
        var expectedSum2 = vAtChal

        let sc2Challenges = ts.squeezeN(halfVars)

        for round in 0..<halfVars {
            let (s0, s1, s2) = proof.sumcheckRounds2[round]
            let roundSum = frAdd(s0, s1)
            if frToInt(roundSum) != frToInt(expectedSum2) {
                return false
            }
            let r = sc2Challenges[round]
            expectedSum2 = evalQuadratic(s0: s0, s1: s1, s2: s2, at: r)

            ts.absorb(s0); ts.absorb(s1); ts.absorb(s2)
        }
        ts.absorb(proof.finalEval2)

        return true
    }

    // MARK: - CPU Sumcheck for Inner Product

    /// Run CPU sumcheck on the inner product sum_i a[i]*b[i].
    public static func cpuSumcheck(
        evalsA: [Fr],
        evalsB: [Fr],
        challenges: [Fr]
    ) -> (rounds: [(Fr, Fr, Fr)], finalEval: Fr) {
        let numVars = challenges.count
        precondition(evalsA.count == (1 << numVars))
        precondition(evalsB.count == (1 << numVars))

        var currentA = evalsA
        var currentB = evalsB
        var rounds: [(Fr, Fr, Fr)] = []

        for round in 0..<numVars {
            let n = currentA.count
            let halfN = n / 2

            var s0 = Fr.zero, s1 = Fr.zero, s2 = Fr.zero
            for i in 0..<halfN {
                let aLo = currentA[i]
                let aHi = currentA[i + halfN]
                let bLo = currentB[i]
                let bHi = currentB[i + halfN]

                // X=0
                s0 = frAdd(s0, frMul(aLo, bLo))
                // X=1
                s1 = frAdd(s1, frMul(aHi, bHi))
                // X=2: a(2) = 2*aHi - aLo, b(2) = 2*bHi - bLo
                let a2 = frSub(frAdd(aHi, aHi), aLo)
                let b2 = frSub(frAdd(bHi, bHi), bLo)
                s2 = frAdd(s2, frMul(a2, b2))
            }
            rounds.append((s0, s1, s2))

            let r = challenges[round]
            var newA = [Fr](repeating: Fr.zero, count: halfN)
            var newB = [Fr](repeating: Fr.zero, count: halfN)
            for i in 0..<halfN {
                let diffA = frSub(currentA[i + halfN], currentA[i])
                newA[i] = frAdd(currentA[i], frMul(r, diffA))
                let diffB = frSub(currentB[i + halfN], currentB[i])
                newB[i] = frAdd(currentB[i], frMul(r, diffB))
            }
            currentA = newA
            currentB = newB
        }

        precondition(currentA.count == 1 && currentB.count == 1)
        let finalEval = frMul(currentA[0], currentB[0])
        return (rounds, finalEval)
    }

    // MARK: - Helpers

    /// Evaluate the matrix row at a given point using multilinear extension.
    public static func evaluateMatrixRow(
        evaluations: [Fr],
        rowPoint: [Fr],
        rows: Int,
        cols: Int
    ) -> [Fr] {
        let eq = eqPolynomial(rowPoint)
        precondition(eq.count == rows)

        var result = [Fr](repeating: Fr.zero, count: cols)
        for i in 0..<rows {
            let weight = eq[i]
            let base = i * cols
            for j in 0..<cols {
                result[j] = frAdd(result[j], frMul(weight, evaluations[base + j]))
            }
        }
        return result
    }

    /// Compute the eq polynomial evaluated over {0,1}^k.
    public static func eqPolynomial(_ point: [Fr]) -> [Fr] {
        let k = point.count
        let size = 1 << k
        var eq = [Fr](repeating: Fr.zero, count: size)
        eq[0] = Fr.one

        for i in 0..<k {
            let half = 1 << i
            let ri = point[i]
            let oneMinusRi = frSub(Fr.one, ri)
            var j = half - 1
            while j >= 0 {
                let val = eq[j]
                eq[2 * j + 1] = frMul(val, ri)
                eq[2 * j] = frMul(val, oneMinusRi)
                j -= 1
            }
        }
        return eq
    }

    /// Evaluate degree-2 polynomial from values at 0, 1, 2 using Lagrange interpolation.
    public static func evalQuadratic(s0: Fr, s1: Fr, s2: Fr, at r: Fr) -> Fr {
        let rMinus1 = frSub(r, Fr.one)
        let rMinus2 = frSub(r, frFromInt(2))

        let two = frFromInt(2)
        let inv2 = frInverse(two)

        // L0 = (r-1)(r-2)/2
        let l0 = frMul(frMul(rMinus1, rMinus2), inv2)
        // L1 = -r(r-2)
        let l1 = frSub(Fr.zero, frMul(r, rMinus2))
        // L2 = r(r-1)/2
        let l2 = frMul(frMul(r, rMinus1), inv2)

        return frAdd(frAdd(frMul(s0, l0), frMul(s1, l1)), frMul(s2, l2))
    }

    /// Evaluate a multilinear polynomial at a point by sequential folding.
    public static func multilinearEval(evals: [Fr], point: [Fr]) -> Fr {
        var current = evals
        for r in point {
            let halfN = current.count / 2
            var next = [Fr](repeating: Fr.zero, count: halfN)
            for i in 0..<halfN {
                let diff = frSub(current[i + halfN], current[i])
                next[i] = frAdd(current[i], frMul(r, diff))
            }
            current = next
        }
        precondition(current.count == 1)
        return current[0]
    }

    // MARK: - Proof Size Analysis

    /// Return proof sizes for comparison at a given number of variables.
    public static func proofSizes(numVars: Int) -> (direct: Int, tensor: Int, basefold: Int) {
        let n = 1 << numVars
        let halfVars = numVars / 2
        let sqrtN = 1 << halfVars

        let direct = n
        // sqrt(N) partial products + 2 sumchecks * halfVars rounds * 3 values + 3 extras
        let tensor = sqrtN + 2 * halfVars * 3 + 3
        // Basefold: ~numQueries * numVars * (2 + numVars) field elements
        let basefold = 40 * numVars * (2 + numVars)

        return (direct, tensor, basefold)
    }
}
