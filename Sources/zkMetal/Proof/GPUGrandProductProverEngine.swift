// GPUGrandProductProverEngine — GPU-accelerated grand product argument prover
//
// Higher-level prover engine for grand product arguments used in:
//   - Plonk/UltraHonk permutation arguments
//   - Plookup/LogUp lookup arguments
//   - GKR grand product layers
//   - Custom gate product constraints
//
// Building blocks:
//   1. Parallel prefix product computation (delegates to GPUGrandProductEngine)
//   2. Grand product polynomial construction (for permutation and lookup arguments)
//   3. Batch grand products for multiple columns
//   4. Fractional grand product (numerator/denominator separation)
//   5. Product check verification (wraps to 1)
//   6. Multilinear extension of grand product polynomials
//
// Works with BN254 Fr field type. GPU-accelerated via GPUGrandProductEngine
// with CPU fallback for all operations.

import Foundation
import NeonFieldOps

// MARK: - Grand Product Polynomial

/// A grand product polynomial z(X) with associated metadata.
/// z[0] = 1, z[i] = prod(values[0..i-1]), and the product check
/// verifies that z[n] wraps back to 1 (for valid permutations).
public struct GrandProductPolynomial {
    /// The polynomial evaluations: z[0] = 1, z[i] = prod of first i values.
    public let evaluations: [Fr]
    /// The final product z[n] = prod(all values). Should be 1 for valid permutation.
    public let finalProduct: Fr
    /// Number of elements in the original values array.
    public let domainSize: Int

    public init(evaluations: [Fr], finalProduct: Fr, domainSize: Int) {
        self.evaluations = evaluations
        self.finalProduct = finalProduct
        self.domainSize = domainSize
    }
}

// MARK: - Fractional Grand Product

/// A fractional grand product separating numerator and denominator accumulations.
/// Used in lookup arguments where the grand product is a ratio of two products.
///
/// numeratorPoly[i]   = prod(numerators[0..i-1])
/// denominatorPoly[i] = prod(denominators[0..i-1])
/// The lookup argument checks: numeratorPoly[n] == denominatorPoly[n].
public struct FractionalGrandProduct {
    /// Prefix products of numerators: numPoly[0] = 1, numPoly[i] = prod(num[0..i-1])
    public let numeratorPoly: [Fr]
    /// Prefix products of denominators: denPoly[0] = 1, denPoly[i] = prod(den[0..i-1])
    public let denominatorPoly: [Fr]
    /// Combined ratio polynomial: z[i] = numPoly[i] * inv(denPoly[i])
    public let ratioPoly: [Fr]
    /// Whether the final numerator product equals the final denominator product.
    public let isBalanced: Bool

    public init(numeratorPoly: [Fr], denominatorPoly: [Fr], ratioPoly: [Fr], isBalanced: Bool) {
        self.numeratorPoly = numeratorPoly
        self.denominatorPoly = denominatorPoly
        self.ratioPoly = ratioPoly
        self.isBalanced = isBalanced
    }
}

// MARK: - Batch Grand Product Result

/// Result of batch grand product computation over multiple columns.
public struct BatchGrandProductResult {
    /// One GrandProductPolynomial per column.
    public let polynomials: [GrandProductPolynomial]
    /// Combined product check: true iff all columns wrap to 1.
    public let allProductsValid: Bool

    public init(polynomials: [GrandProductPolynomial], allProductsValid: Bool) {
        self.polynomials = polynomials
        self.allProductsValid = allProductsValid
    }
}

// MARK: - Multilinear Extension

/// Multilinear extension of a grand product polynomial over the Boolean hypercube.
/// For a domain of size n = 2^k, the MLE is a k-variate multilinear polynomial
/// such that MLE(b_1, ..., b_k) = z[index(b_1,...,b_k)] for all binary inputs.
public struct GrandProductMLE {
    /// Evaluations over the Boolean hypercube {0,1}^k, length 2^k.
    public let evaluations: [Fr]
    /// Number of variables k (log2 of domain size).
    public let numVariables: Int

    public init(evaluations: [Fr], numVariables: Int) {
        self.evaluations = evaluations
        self.numVariables = numVariables
    }
}

// MARK: - GPU Grand Product Prover Engine

/// GPU-accelerated grand product argument prover.
///
/// Provides high-level operations for constructing grand product arguments:
///   - Grand product polynomial construction from raw values
///   - Fractional grand product for lookup arguments
///   - Batch grand products for multiple witness columns
///   - Product check verification
///   - Multilinear extension for sumcheck-based protocols
///
/// Delegates low-level prefix product and batch inverse to GPUGrandProductEngine.
/// Falls back to CPU when GPU is unavailable or for small inputs.
public final class GPUGrandProductProverEngine {

    /// The underlying GPU grand product engine (nil if GPU unavailable).
    private let gpuEngine: GPUGrandProductEngine?

    /// Threshold below which we use CPU path directly.
    public var cpuThreshold: Int = 1024

    /// Whether GPU acceleration is available.
    public var isGPUAvailable: Bool { gpuEngine != nil }

    /// Create engine, attempting GPU initialization. Falls back to CPU-only if GPU unavailable.
    public init() {
        self.gpuEngine = try? GPUGrandProductEngine()
    }

    // MARK: - Grand Product Polynomial Construction

    /// Construct the grand product polynomial z(X) from a vector of field elements.
    ///
    /// z[0] = 1
    /// z[i] = values[0] * values[1] * ... * values[i-1]  for i in 1..n
    ///
    /// The output has length n+1 (includes z[n] = product of all values).
    /// For a valid permutation argument, z[n] should equal 1.
    ///
    /// - Parameter values: Field elements to accumulate.
    /// - Returns: Grand product polynomial with evaluations and final product.
    public func constructGrandProductPolynomial(values: [Fr]) -> GrandProductPolynomial {
        let n = values.count
        guard n > 0 else {
            return GrandProductPolynomial(evaluations: [Fr.one], finalProduct: Fr.one, domainSize: 0)
        }

        // Compute prefix products z[0..n-1] using GPU or CPU
        let prefixProds = computePrefixProducts(values)

        // z[n] = z[n-1] * values[n-1] = product of all values
        let finalProduct = frMul(prefixProds[n - 1], values[n - 1])

        // Build evaluations array of length n+1: [z[0], z[1], ..., z[n]]
        var evaluations = [Fr](repeating: Fr.zero, count: n + 1)
        for i in 0..<n {
            evaluations[i] = prefixProds[i]
        }
        evaluations[n] = finalProduct

        return GrandProductPolynomial(
            evaluations: evaluations,
            finalProduct: finalProduct,
            domainSize: n
        )
    }

    /// Construct the grand product polynomial for a permutation argument.
    ///
    /// Given sigma(i) and id(i) representations plus a random challenge beta, gamma:
    ///   values[i] = (w_i + beta * id(i) + gamma) / (w_i + beta * sigma(i) + gamma)
    ///
    /// This method takes pre-computed numerator and denominator arrays.
    ///
    /// - Parameters:
    ///   - numerators: (w_i + beta * id(i) + gamma) for each i
    ///   - denominators: (w_i + beta * sigma(i) + gamma) for each i
    /// - Returns: Grand product polynomial of the ratios.
    public func constructPermutationPolynomial(
        numerators: [Fr],
        denominators: [Fr]
    ) -> GrandProductPolynomial {
        precondition(numerators.count == denominators.count,
                     "numerators and denominators must have equal length")
        let n = numerators.count
        guard n > 0 else {
            return GrandProductPolynomial(evaluations: [Fr.one], finalProduct: Fr.one, domainSize: 0)
        }

        // Compute ratios = numerators[i] / denominators[i]
        let ratios = computeRatios(numerators: numerators, denominators: denominators)

        // Build grand product polynomial from ratios
        return constructGrandProductPolynomial(values: ratios)
    }

    // MARK: - Fractional Grand Product

    /// Compute a fractional grand product, keeping numerator and denominator
    /// prefix products separate. Used in lookup arguments (Plookup, LogUp).
    ///
    /// numeratorPoly[i]   = prod(numerators[0..i-1])
    /// denominatorPoly[i] = prod(denominators[0..i-1])
    /// ratioPoly[i]       = numeratorPoly[i] / denominatorPoly[i]
    ///
    /// The lookup is valid iff numeratorPoly[n] == denominatorPoly[n].
    ///
    /// - Parameters:
    ///   - numerators: Numerator field elements.
    ///   - denominators: Denominator field elements (must be non-zero).
    /// - Returns: FractionalGrandProduct with separate accumulations and balance check.
    public func computeFractionalGrandProduct(
        numerators: [Fr],
        denominators: [Fr]
    ) -> FractionalGrandProduct {
        precondition(numerators.count == denominators.count,
                     "numerators and denominators must have equal length")
        let n = numerators.count
        guard n > 0 else {
            return FractionalGrandProduct(
                numeratorPoly: [Fr.one],
                denominatorPoly: [Fr.one],
                ratioPoly: [Fr.one],
                isBalanced: true
            )
        }

        // Compute separate prefix products for numerators and denominators
        let numPrefix = computePrefixProducts(numerators)
        let denPrefix = computePrefixProducts(denominators)

        // Final products
        let numFinal = frMul(numPrefix[n - 1], numerators[n - 1])
        let denFinal = frMul(denPrefix[n - 1], denominators[n - 1])
        let isBalanced = frEqual(numFinal, denFinal)

        // Compute ratio polynomial: ratioPoly[i] = numPrefix[i] * inv(denPrefix[i])
        let ratioPoly = computeRatioPoly(numPrefix: numPrefix, denPrefix: denPrefix)

        return FractionalGrandProduct(
            numeratorPoly: numPrefix,
            denominatorPoly: denPrefix,
            ratioPoly: ratioPoly,
            isBalanced: isBalanced
        )
    }

    // MARK: - Batch Grand Products

    /// Compute grand product polynomials for multiple columns simultaneously.
    ///
    /// Each column produces its own grand product polynomial. This is used when
    /// a protocol requires multiple independent grand product arguments (e.g.,
    /// multi-column permutation in UltraHonk).
    ///
    /// - Parameter columns: Array of value vectors, one per column.
    /// - Returns: BatchGrandProductResult with per-column polynomials and validity check.
    public func batchGrandProducts(columns: [[Fr]]) -> BatchGrandProductResult {
        guard !columns.isEmpty else {
            return BatchGrandProductResult(polynomials: [], allProductsValid: true)
        }

        var polynomials = [GrandProductPolynomial]()
        polynomials.reserveCapacity(columns.count)
        var allValid = true

        for col in columns {
            let poly = constructGrandProductPolynomial(values: col)
            if !frEqual(poly.finalProduct, Fr.one) {
                allValid = false
            }
            polynomials.append(poly)
        }

        return BatchGrandProductResult(polynomials: polynomials, allProductsValid: allValid)
    }

    /// Compute batch grand products for permutation arguments across multiple columns.
    ///
    /// Each column has its own numerator/denominator arrays. The combined product
    /// across all columns must equal 1 for the permutation to be valid.
    ///
    /// - Parameters:
    ///   - numeratorColumns: Array of numerator vectors, one per column.
    ///   - denominatorColumns: Array of denominator vectors, one per column.
    /// - Returns: BatchGrandProductResult with per-column polynomials.
    public func batchPermutationProducts(
        numeratorColumns: [[Fr]],
        denominatorColumns: [[Fr]]
    ) -> BatchGrandProductResult {
        precondition(numeratorColumns.count == denominatorColumns.count,
                     "Must have same number of numerator and denominator columns")

        var polynomials = [GrandProductPolynomial]()
        polynomials.reserveCapacity(numeratorColumns.count)
        var combinedProduct = Fr.one

        for i in 0..<numeratorColumns.count {
            let poly = constructPermutationPolynomial(
                numerators: numeratorColumns[i],
                denominators: denominatorColumns[i]
            )
            combinedProduct = frMul(combinedProduct, poly.finalProduct)
            polynomials.append(poly)
        }

        let allValid = frEqual(combinedProduct, Fr.one)
        return BatchGrandProductResult(polynomials: polynomials, allProductsValid: allValid)
    }

    // MARK: - Product Check Verification

    /// Verify that a grand product polynomial wraps to 1: z[0] = z[n] = 1.
    ///
    /// This is the core soundness check for permutation arguments:
    /// if the permutation is valid, the product of all ratios is 1.
    ///
    /// - Parameter polynomial: The grand product polynomial to check.
    /// - Returns: true if z[0] == 1 and z[n] == 1.
    public func verifyProductCheck(_ polynomial: GrandProductPolynomial) -> Bool {
        guard !polynomial.evaluations.isEmpty else { return true }
        let z0 = polynomial.evaluations[0]
        let zn = polynomial.finalProduct
        return frEqual(z0, Fr.one) && frEqual(zn, Fr.one)
    }

    /// Verify that a grand product polynomial is consistent with its values.
    ///
    /// Checks the recurrence relation: z[i+1] = z[i] * values[i] for all i.
    ///
    /// - Parameters:
    ///   - polynomial: The grand product polynomial.
    ///   - values: The original values used to construct it.
    /// - Returns: true if all recurrence checks pass.
    public func verifyRecurrence(polynomial: GrandProductPolynomial, values: [Fr]) -> Bool {
        let evals = polynomial.evaluations
        let n = values.count
        guard evals.count == n + 1 else { return false }
        guard frEqual(evals[0], Fr.one) else { return false }

        for i in 0..<n {
            let expected = frMul(evals[i], values[i])
            if !frEqual(evals[i + 1], expected) {
                return false
            }
        }
        return true
    }

    /// Verify a fractional grand product: numerator and denominator final products match.
    ///
    /// - Parameter fractional: The fractional grand product to check.
    /// - Returns: true if the products are balanced.
    public func verifyFractionalBalance(_ fractional: FractionalGrandProduct) -> Bool {
        return fractional.isBalanced
    }

    // MARK: - Multilinear Extension

    /// Compute the multilinear extension (MLE) of a grand product polynomial.
    ///
    /// Given z[0..2^k-1], produces the unique multilinear polynomial MLE such that
    /// MLE(b_1, ..., b_k) = z[index(b_1,...,b_k)] for all (b_1,...,b_k) in {0,1}^k.
    ///
    /// The input evaluations must have power-of-two length. If the grand product
    /// polynomial has length n+1 (from n values), this pads or truncates to 2^k.
    ///
    /// - Parameter polynomial: Grand product polynomial to extend.
    /// - Returns: GrandProductMLE with evaluations over the Boolean hypercube.
    public func multilinearExtension(of polynomial: GrandProductPolynomial) -> GrandProductMLE {
        let evals = polynomial.evaluations
        let n = evals.count
        guard n > 0 else {
            return GrandProductMLE(evaluations: [Fr.one], numVariables: 0)
        }

        // Pad to next power of two
        let k = ceilLog2(n)
        let size = 1 << k

        var padded = [Fr](repeating: Fr.zero, count: size)
        for i in 0..<min(n, size) {
            padded[i] = evals[i]
        }

        return GrandProductMLE(evaluations: padded, numVariables: k)
    }

    /// Evaluate the multilinear extension at an arbitrary point in F^k.
    ///
    /// Uses the standard MLE evaluation algorithm:
    ///   For each variable j from k-1 down to 0:
    ///     For each i in 0..<(1 << j):
    ///       table[i] = table[2*i] * (1 - point[j]) + table[2*i+1] * point[j]
    ///
    /// - Parameters:
    ///   - mle: The multilinear extension.
    ///   - point: Evaluation point, must have length == mle.numVariables.
    /// - Returns: MLE evaluated at the given point.
    public func evaluateMLE(_ mle: GrandProductMLE, at point: [Fr]) -> Fr {
        let k = mle.numVariables
        precondition(point.count == k, "Point dimension must match number of variables")

        guard k > 0 else {
            return mle.evaluations.isEmpty ? Fr.zero : mle.evaluations[0]
        }

        // Work buffer: start with the hypercube evaluations
        var table = mle.evaluations
        let size = table.count
        precondition(size == (1 << k))

        // Iteratively collapse dimensions
        var halfSize = size >> 1
        for j in 0..<k {
            let rj = point[j]
            let oneMinusRj = frSub(Fr.one, rj)
            for i in 0..<halfSize {
                // table[i] = table[2i] * (1 - r_j) + table[2i+1] * r_j
                let lo = frMul(table[2 * i], oneMinusRj)
                let hi = frMul(table[2 * i + 1], rj)
                table[i] = frAdd(lo, hi)
            }
            halfSize >>= 1
        }

        return table[0]
    }

    /// Compute the MLE of a grand product polynomial and evaluate it at a point.
    ///
    /// Convenience method combining multilinearExtension + evaluateMLE.
    ///
    /// - Parameters:
    ///   - polynomial: Grand product polynomial.
    ///   - point: Evaluation point in F^k where k = ceil(log2(domainSize + 1)).
    /// - Returns: MLE evaluation at the given point.
    public func evaluateGrandProductMLE(
        polynomial: GrandProductPolynomial,
        at point: [Fr]
    ) -> Fr {
        let mle = multilinearExtension(of: polynomial)
        return evaluateMLE(mle, at: point)
    }

    // MARK: - Internal: Prefix Products

    /// Compute exclusive prefix products using GPU engine or CPU fallback.
    /// Returns array of length n where result[0] = 1, result[i] = prod(values[0..i-1]).
    private func computePrefixProducts(_ values: [Fr]) -> [Fr] {
        let n = values.count
        guard n > 0 else { return [] }

        // Try GPU path for large inputs
        if let engine = gpuEngine, n >= cpuThreshold {
            return engine.partialProducts(values: values)
        }

        // CPU fallback
        return cpuPrefixProducts(values)
    }

    /// CPU prefix product: result[0] = 1, result[i] = prod(values[0..i-1]).
    private func cpuPrefixProducts(_ values: [Fr]) -> [Fr] {
        let n = values.count
        var result = [Fr](repeating: Fr.zero, count: n)
        result[0] = Fr.one
        for i in 1..<n {
            result[i] = frMul(result[i - 1], values[i - 1])
        }
        return result
    }

    // MARK: - Internal: Ratio Computation

    /// Compute element-wise ratios numerators[i] / denominators[i].
    private func computeRatios(numerators: [Fr], denominators: [Fr]) -> [Fr] {
        let n = numerators.count
        guard n > 0 else { return [] }

        // Batch invert denominators using Montgomery's trick
        let invDen = batchInverse(denominators)

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
        return ratios
    }

    /// Compute ratio polynomial from separate numerator and denominator prefix products.
    /// ratioPoly[i] = numPrefix[i] * inv(denPrefix[i])
    private func computeRatioPoly(numPrefix: [Fr], denPrefix: [Fr]) -> [Fr] {
        let n = numPrefix.count
        let invDen = batchInverse(denPrefix)
        var result = [Fr](repeating: Fr.zero, count: n)
        numPrefix.withUnsafeBytes { aBuf in
            invDen.withUnsafeBytes { bBuf in
                result.withUnsafeMutableBytes { rBuf in
                    bn254_fr_batch_mul(
                        aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n))
                }
            }
        }
        return result
    }

    /// Batch inverse using Montgomery's trick.
    /// For zero elements, returns zero (not mathematically correct, but safe).
    private func batchInverse(_ values: [Fr]) -> [Fr] {
        let n = values.count
        guard n > 0 else { return [] }
        var result = [Fr](repeating: Fr.zero, count: n)
        values.withUnsafeBytes { aBuf in
            result.withUnsafeMutableBytes { rBuf in
                bn254_fr_batch_inverse_safe(
                    aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(n),
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                )
            }
        }
        return result
    }

    // MARK: - Internal: Utilities

    /// Ceiling of log2(n). Returns 0 for n <= 1.
    private func ceilLog2(_ n: Int) -> Int {
        guard n > 1 else { return 0 }
        var k = 0
        var v = n - 1
        while v > 0 {
            v >>= 1
            k += 1
        }
        return k
    }
}
