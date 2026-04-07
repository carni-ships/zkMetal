// MultilinearPolynomial — Comprehensive multilinear polynomial engine
//
// Foundation for sumcheck-based protocols: Spartan, GKR, Lasso, HyperNova.
// Multilinear polynomials are represented as evaluations over the boolean hypercube {0,1}^n.
//
// The multilinear extension (MLE) of f: {0,1}^n -> F is the unique polynomial of degree <= 1
// in each variable such that MLE(x) = f(x) for all x in {0,1}^n:
//   f~(r_0,...,r_{n-1}) = sum_{x in {0,1}^n} f(x) * prod_i (r_i * x_i + (1 - r_i)(1 - x_i))
//
// Index convention: index i encodes the boolean point whose binary representation is i,
// with MSB = variable 0 (consistent with SumcheckEngine and GKR).

import Foundation
import NeonFieldOps

// MARK: - MultilinearPoly Extensions

extension MultilinearPoly {

    // MARK: - Evaluate at arbitrary point (C-accelerated)

    /// Evaluate the MLE at an arbitrary field point using C CIOS Montgomery arithmetic.
    /// This is the primary evaluation path — O(2^n) time, O(2^n) space.
    public func evaluateC(at point: [Fr]) -> Fr {
        precondition(point.count == numVars, "Point dimension \(point.count) != numVars \(numVars)")
        if numVars == 0 { return evals[0] }
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

    // MARK: - Fix variable at arbitrary index

    /// Fix variable at position `index` to `value`, returning a polynomial with numVars-1 variables.
    /// Index 0 = MSB (first variable). This generalizes `fixVariable(_:)` which always fixes variable 0.
    ///
    /// For fixing variable j in an n-variable MLP:
    /// - Variables 0..j-1 remain, variables j+1..n-1 become j..n-2
    /// - For each assignment of the remaining variables, interpolate along variable j
    public func fixVariable(index: Int, value: Fr) -> MultilinearPoly {
        precondition(numVars > 0, "Cannot fix variable in 0-variable polynomial")
        precondition(index >= 0 && index < numVars, "Variable index \(index) out of range [0, \(numVars))")

        if index == 0 {
            // Fast path: fixing MSB is the standard case
            return fixVariable(value)
        }

        let n = numVars
        let newSize = 1 << (n - 1)
        let oneMinusV = frSub(Fr.one, value)

        // The stride for variable `index` is 2^(n-1-index)
        let stride = 1 << (n - 1 - index)
        // Block size (number of consecutive indices before variable j toggles)
        let blockSize = stride
        // Number of blocks of 2*stride that tile the domain
        let numBlocks = size / (2 * stride)

        var result = [Fr](repeating: Fr.zero, count: newSize)
        var outIdx = 0
        for block in 0..<numBlocks {
            let base = block * 2 * stride
            for i in 0..<blockSize {
                let lo = evals[base + i]           // variable j = 0
                let hi = evals[base + stride + i]  // variable j = 1
                result[outIdx] = frAdd(frMul(oneMinusV, lo), frMul(value, hi))
                outIdx += 1
            }
        }

        return MultilinearPoly(numVars: n - 1, evals: result)
    }

    // MARK: - Create from evaluation table

    /// Create a multilinear polynomial from an evaluation table.
    /// Infers numVars from the table size (must be a power of 2).
    public static func extend(fromEvals evaluations: [Fr]) -> MultilinearPoly {
        precondition(!evaluations.isEmpty, "Empty evaluation table")
        let n = evaluations.count
        precondition(n & (n - 1) == 0, "Evaluation table size \(n) must be a power of 2")
        var numVars = 0
        var s = n
        while s > 1 { s >>= 1; numVars += 1 }
        return MultilinearPoly(numVars: numVars, evals: evaluations)
    }

    // MARK: - Arithmetic operations

    /// Add two multilinear polynomials with the same number of variables.
    public static func add(_ a: MultilinearPoly, _ b: MultilinearPoly) -> MultilinearPoly {
        precondition(a.numVars == b.numVars, "Cannot add MLPs with different numVars: \(a.numVars) vs \(b.numVars)")
        let n = a.size
        var result = [Fr](repeating: Fr.zero, count: n)

        // Use C batch add for performance
        a.evals.withUnsafeBytes { aBuf in
            b.evals.withUnsafeBytes { bBuf in
                result.withUnsafeMutableBytes { rBuf in
                    bn254_fr_batch_add_neon(
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n)
                    )
                }
            }
        }
        return MultilinearPoly(numVars: a.numVars, evals: result)
    }

    /// Subtract b from a (pointwise). Uses C batch sub for performance.
    public static func sub(_ a: MultilinearPoly, _ b: MultilinearPoly) -> MultilinearPoly {
        precondition(a.numVars == b.numVars, "Cannot subtract MLPs with different numVars")
        let n = a.size
        var result = [Fr](repeating: Fr.zero, count: n)
        a.evals.withUnsafeBytes { aBuf in
            b.evals.withUnsafeBytes { bBuf in
                result.withUnsafeMutableBytes { rBuf in
                    bn254_fr_batch_sub_neon(
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n)
                    )
                }
            }
        }
        return MultilinearPoly(numVars: a.numVars, evals: result)
    }

    /// Hadamard (pointwise) product of two MLPs (batch C kernel).
    /// Note: the result is NOT multilinear in general — it has degree 2 in each variable.
    /// However, its evaluations over {0,1}^n are correct (since x^2 = x over {0,1}).
    public static func mul(_ a: MultilinearPoly, _ b: MultilinearPoly) -> MultilinearPoly {
        precondition(a.numVars == b.numVars, "Cannot multiply MLPs with different numVars")
        let n = a.size
        var result = [Fr](repeating: Fr.zero, count: n)
        a.evals.withUnsafeBytes { aBuf in
            b.evals.withUnsafeBytes { bBuf in
                result.withUnsafeMutableBytes { rBuf in
                    bn254_fr_batch_mul_neon(
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n))
                }
            }
        }
        return MultilinearPoly(numVars: a.numVars, evals: result)
    }

    /// Scale all evaluations by a constant.
    public static func scale(_ poly: MultilinearPoly, by scalar: Fr) -> MultilinearPoly {
        let n = poly.size
        var result = [Fr](repeating: Fr.zero, count: n)

        // Use C batch scalar multiply
        poly.evals.withUnsafeBytes { pBuf in
            withUnsafeBytes(of: scalar) { sBuf in
                result.withUnsafeMutableBytes { rBuf in
                    bn254_fr_batch_mul_scalar_neon(
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        sBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n)
                    )
                }
            }
        }
        return MultilinearPoly(numVars: poly.numVars, evals: result)
    }

    // MARK: - Equality polynomial (C-accelerated)

    /// Compute the equality polynomial eq(r, x) = prod_i (r_i * x_i + (1 - r_i)(1 - x_i))
    /// evaluated at all x in {0,1}^n, using C CIOS Montgomery arithmetic.
    ///
    /// This is the key building block for sumcheck: the "selector" that picks out
    /// a specific point r from the boolean hypercube.
    public static func eqPolyC(point: [Fr]) -> MultilinearPoly {
        let n = point.count
        let size = 1 << n
        var eq = [Fr](repeating: Fr.zero, count: size)
        point.withUnsafeBytes { ptBuf in
            eq.withUnsafeMutableBytes { eqBuf in
                gkr_eq_poly(
                    ptBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(n),
                    eqBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                )
            }
        }
        return MultilinearPoly(numVars: n, evals: eq)
    }

    // MARK: - Tensor product

    /// Tensor product of two MLPs: if a has m variables and b has n variables,
    /// the result has m+n variables where result(x, y) = a(x) * b(y).
    ///
    /// Evaluations: result[i * 2^n + j] = a[i] * b[j]
    public static func tensor(_ a: MultilinearPoly, _ b: MultilinearPoly) -> MultilinearPoly {
        let newVars = a.numVars + b.numVars
        let newSize = a.size * b.size
        var result = [Fr](repeating: Fr.zero, count: newSize)

        for i in 0..<a.size {
            if a.evals[i].isZero { continue }
            let base = i * b.size
            // Use batch scalar multiply for the inner loop
            b.evals.withUnsafeBytes { bBuf in
                withUnsafeBytes(of: a.evals[i]) { sBuf in
                    result.withUnsafeMutableBytes { rBuf in
                        let rPtr = rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                            .advanced(by: base * 4) // 4 uint64 per Fr
                        bn254_fr_batch_mul_scalar_neon(
                            rPtr,
                            bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            sBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(b.size)
                        )
                    }
                }
            }
        }

        return MultilinearPoly(numVars: newVars, evals: result)
    }

    // MARK: - Batch evaluation

    /// Evaluate multiple MLPs at the same point, sharing the eq polynomial computation.
    /// This is O(n * 2^n + k * 2^n) where n = numVars and k = number of polynomials,
    /// compared to O(k * 2^n) for k independent evaluations — but the eq polynomial
    /// is computed once and reused.
    ///
    /// All polynomials must have the same numVars.
    public static func batchEvaluate(polys: [MultilinearPoly], point: [Fr]) -> [Fr] {
        guard let first = polys.first else { return [] }
        let n = first.numVars
        precondition(point.count == n, "Point dimension mismatch")
        for p in polys {
            precondition(p.numVars == n, "All polynomials must have the same numVars")
        }

        if polys.count == 1 {
            return [first.evaluateC(at: point)]
        }

        // Compute eq(point, x) for all x in {0,1}^n
        let eq = eqPolyC(point: point)
        let size = 1 << n

        // For each polynomial, compute inner product <evals, eq>
        var results = [Fr](repeating: Fr.zero, count: polys.count)
        for (k, poly) in polys.enumerated() {
            var sum = Fr.zero
            for i in 0..<size {
                if eq.evals[i].isZero || poly.evals[i].isZero { continue }
                sum = frAdd(sum, frMul(poly.evals[i], eq.evals[i]))
            }
            results[k] = sum
        }

        return results
    }

    // MARK: - Random MLP for testing

    /// Generate a random multilinear polynomial for testing.
    /// Uses arc4random for non-cryptographic randomness.
    public static func randomize(numVars: Int) -> MultilinearPoly {
        let size = 1 << numVars
        var evals = [Fr](repeating: Fr.zero, count: size)
        for i in 0..<size {
            // Generate random Fr element: 8 random UInt32 limbs, then reduce mod p
            let limbs: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32) = (
                arc4random(), arc4random(), arc4random(), arc4random(),
                arc4random(), arc4random(), arc4random(), arc4random()
            )
            // Convert to Montgomery form by multiplying raw value by R^2
            let raw = Fr(v: limbs)
            evals[i] = frMul(raw, Fr.from64(Fr.R2_MOD_R))
        }
        return MultilinearPoly(numVars: numVars, evals: evals)
    }

    // MARK: - Linear combination

    /// Compute a linear combination: result = sum_i coeffs[i] * polys[i]
    /// All polynomials must have the same numVars.
    public static func linearCombination(coeffs: [Fr], polys: [MultilinearPoly]) -> MultilinearPoly {
        precondition(coeffs.count == polys.count, "Coefficients and polynomials count mismatch")
        guard let first = polys.first else {
            return MultilinearPoly(numVars: 0, evals: [Fr.zero])
        }
        let n = first.numVars
        let size = first.size
        for p in polys { precondition(p.numVars == n) }

        var result = [Fr](repeating: Fr.zero, count: size)
        for (k, poly) in polys.enumerated() {
            let c = coeffs[k]
            if c.isZero { continue }
            for i in 0..<size {
                result[i] = frAdd(result[i], frMul(c, poly.evals[i]))
            }
        }
        return MultilinearPoly(numVars: n, evals: result)
    }

    // MARK: - Conversion helpers

    /// Convert to dense evaluation array (identity — evals are already dense).
    public func toDenseEvals() -> [Fr] {
        return evals
    }

    /// Convert from sparse representation to dense multilinear polynomial.
    public static func fromSparse(_ sparse: SparseMultilinearPoly) -> MultilinearPoly {
        var evals = [Fr](repeating: Fr.zero, count: sparse.domainSize)
        for entry in sparse.entries {
            evals[entry.idx] = entry.val
        }
        return MultilinearPoly(numVars: sparse.numVars, evals: evals)
    }

    /// Check if this polynomial is the zero polynomial.
    public var isZeroPoly: Bool {
        for e in evals {
            if !e.isZero { return false }
        }
        return true
    }

    // MARK: - Partial evaluation (multi-variable)

    /// Fix multiple variables at once, starting from variable 0 (MSB).
    /// Equivalent to calling fixVariable repeatedly but potentially more efficient.
    public func partialEvaluate(values: [Fr]) -> MultilinearPoly {
        precondition(values.count <= numVars, "Too many values: \(values.count) > \(numVars)")
        var current = self
        for v in values {
            current = current.fixVariable(v)
        }
        return current
    }

    // MARK: - Bind and sum (sumcheck building block)

    /// For sumcheck round: compute the univariate polynomial obtained by
    /// summing over all remaining variables except the first (MSB).
    /// Returns (s0, s1) where s0 = sum over x_1..x_{n-1} of f(0, x_1, ..., x_{n-1})
    /// and s1 = sum over x_1..x_{n-1} of f(1, x_1, ..., x_{n-1}).
    public func sumcheckRoundSums() -> (Fr, Fr) {
        precondition(numVars > 0)
        let half = size / 2
        var s0 = Fr.zero
        var s1 = Fr.zero
        for j in 0..<half {
            s0 = frAdd(s0, evals[j])
            s1 = frAdd(s1, evals[j + half])
        }
        return (s0, s1)
    }
}
