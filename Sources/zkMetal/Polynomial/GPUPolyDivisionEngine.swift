// GPU-accelerated polynomial division engine (high-level Fr API)
//
// Operations on BN254 Fr polynomials:
//   - longDivide(num, den)          -- full polynomial long division (quotient + remainder)
//   - divideByVanishing(poly, n)    -- division by Z_H(X) = X^n - 1
//   - divideByLinearFactor(poly, r) -- division by (X - r) via synthetic division
//   - batchDivideByLinear(polys, r) -- divide multiple polys by same (X - r)
//   - syntheticDivide(poly, r)      -- synthetic division returning (quotient, remainder)
//
// All polynomials are represented as [Fr] in ascending degree order:
//   p(X) = coeffs[0] + coeffs[1]*X + coeffs[2]*X^2 + ...
//
// CPU-only implementation; leverages NEON-accelerated Fr arithmetic.

import Foundation
import NeonFieldOps

public class GPUPolyDivisionEngine {
    public static let version = Versions.gpuPolyDivision

    // MARK: - Initialization

    public init() {}

    // MARK: - Polynomial Long Division

    /// Divide numerator by denominator: num = den * quotient + remainder.
    /// Returns (quotient, remainder) with deg(remainder) < deg(denominator).
    /// Both polynomials in ascending degree order.
    public func longDivide(_ numerator: [Fr], by denominator: [Fr]) -> ([Fr], [Fr]) {
        guard !denominator.isEmpty else {
            fatalError("GPUPolyDivisionEngine: division by zero polynomial")
        }

        // Strip leading zeros from denominator
        let den = stripLeadingZeros(denominator)
        let num = stripLeadingZeros(numerator)

        let denDeg = den.count - 1
        let numDeg = num.count - 1

        // If numerator degree < denominator degree, quotient is 0
        if numDeg < denDeg {
            return ([Fr.zero], num.isEmpty ? [Fr.zero] : num)
        }

        // Leading coefficient of denominator and its inverse
        let leadInv = frInverse(den[denDeg])

        let quotDeg = numDeg - denDeg
        var quotient = [Fr](repeating: Fr.zero, count: quotDeg + 1)

        // Working copy of numerator (we subtract from it)
        var rem = num

        // Long division: work from highest degree down
        for i in stride(from: quotDeg, through: 0, by: -1) {
            let remIdx = i + denDeg
            guard remIdx < rem.count else { continue }

            // Quotient coefficient: rem[remIdx] / den[denDeg]
            let qCoeff = frMul(rem[remIdx], leadInv)
            quotient[i] = qCoeff

            // Subtract qCoeff * den from remainder
            for j in 0...denDeg {
                let idx = i + j
                if idx < rem.count {
                    rem[idx] = frSub(rem[idx], frMul(qCoeff, den[j]))
                }
            }
        }

        // Trim remainder to degree < denDeg
        var remainder = Array(rem.prefix(denDeg))
        remainder = stripLeadingZeros(remainder)
        if remainder.isEmpty { remainder = [Fr.zero] }

        return (quotient, remainder)
    }

    // MARK: - Division by Vanishing Polynomial

    /// Divide polynomial by vanishing polynomial Z_H(X) = X^n - 1.
    /// The polynomial must be exactly divisible (remainder must be zero).
    /// n must be a power of 2.
    /// Returns the quotient polynomial.
    public func divideByVanishing(_ poly: [Fr], subgroupSize n: Int) -> ([Fr], [Fr]) {
        precondition(n > 0 && (n & (n - 1)) == 0, "subgroupSize must be a power of 2")

        let polyLen = poly.count

        // If polynomial degree < n, the vanishing polynomial has higher degree
        if polyLen <= n {
            // quotient is zero, remainder is the original polynomial
            return ([Fr.zero], poly.isEmpty ? [Fr.zero] : poly)
        }

        // Division by X^n - 1 can be done efficiently:
        // If p(X) = q(X) * (X^n - 1) + r(X) where deg(r) < n, then:
        //   For coefficients at position i >= n: q[i-n] gets contributions
        //   We process from highest degree down.
        //
        // Efficient algorithm: fold coefficients in blocks of n.
        // p = a_0 + a_1*X^n + a_2*X^{2n} + ... (where each a_i is a polynomial of degree < n)
        // p / (X^n - 1) uses the fact that X^n = 1 + (X^n - 1).

        let quotLen = polyLen - n
        var quotient = [Fr](repeating: Fr.zero, count: quotLen)
        var remainder = [Fr](repeating: Fr.zero, count: n)

        // Start from highest degree coefficient and work down
        // Using synthetic-style approach for X^n - 1
        // p(X) = sum c_i X^i
        // We compute q, r such that p = q * (X^n - 1) + r
        //
        // Working from top: for i from (polyLen-1) down to n:
        //   q[i - n] = accumulated[i]
        //   accumulated[i - n] += q[i - n]  (because -(−1) * q = +q at position i-n)
        var work = poly
        for i in stride(from: polyLen - 1, through: n, by: -1) {
            let qCoeff = work[i]
            quotient[i - n] = qCoeff
            // X^n - 1 means we add qCoeff back at position i - n
            work[i - n] = frAdd(work[i - n], qCoeff)
        }

        // Remainder is the first n coefficients after folding
        for i in 0..<n {
            remainder[i] = work[i]
        }

        return (stripLeadingZeros(quotient.isEmpty ? [Fr.zero] : quotient),
                stripLeadingZeros(remainder))
    }

    // MARK: - Division by Linear Factor (X - r)

    /// Divide polynomial by (X - r) using synthetic division.
    /// Returns (quotient, remainder) where remainder is a single Fr element (as [Fr]).
    /// Remainder equals p(r) by the Remainder Theorem.
    public func divideByLinearFactor(_ poly: [Fr], root r: Fr) -> ([Fr], [Fr]) {
        let (q, rem) = syntheticDivide(poly, root: r)
        return (q, [rem])
    }

    // MARK: - Synthetic Division

    /// Synthetic division of polynomial by (X - r).
    /// Returns (quotient coefficients, remainder scalar).
    /// Uses Horner-like scheme: process from highest degree to lowest.
    public func syntheticDivide(_ poly: [Fr], root r: Fr) -> ([Fr], Fr) {
        let coeffs = stripLeadingZeros(poly)
        let n = coeffs.count

        if n <= 1 {
            return ([Fr.zero], coeffs.isEmpty ? Fr.zero : coeffs[0])
        }

        var quotient = [Fr](repeating: Fr.zero, count: n - 1)
        coeffs.withUnsafeBytes { cBuf in
            withUnsafeBytes(of: r) { zBuf in
                quotient.withUnsafeMutableBytes { qBuf in
                    bn254_fr_synthetic_div(
                        cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        zBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n),
                        qBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                    )
                }
            }
        }
        let remainder = frAdd(coeffs[0], frMul(r, quotient[0]))

        return (quotient, remainder)
    }

    // MARK: - Batch Division (multiple polys, same divisor)

    /// Divide multiple polynomials by the same linear factor (X - r).
    /// Returns array of (quotient, remainder) pairs.
    public func batchDivideByLinear(_ polys: [[Fr]], root r: Fr) -> [([Fr], [Fr])] {
        return polys.map { poly in
            divideByLinearFactor(poly, root: r)
        }
    }

    /// Divide a single polynomial by multiple linear factors.
    /// Returns array of (quotient, remainder) pairs, one per root.
    public func divideByMultipleRoots(_ poly: [Fr], roots: [Fr]) -> [([Fr], [Fr])] {
        return roots.map { r in
            divideByLinearFactor(poly, root: r)
        }
    }

    // MARK: - Utility: Evaluate via Remainder Theorem

    /// Evaluate polynomial at a point using synthetic division (Remainder Theorem).
    /// p(r) = remainder of p(X) / (X - r).
    public func evaluate(_ poly: [Fr], at point: Fr) -> Fr {
        let (_, rem) = syntheticDivide(poly, root: point)
        return rem
    }

    // MARK: - Helpers

    /// Strip trailing zero coefficients (leading in degree sense) from ascending-order polynomial.
    private func stripLeadingZeros(_ poly: [Fr]) -> [Fr] {
        var p = poly
        while p.count > 1 && p.last! == Fr.zero {
            p.removeLast()
        }
        return p
    }
}
