// GPUGrandProductProverTests — Tests for GPU-accelerated grand product prover engine
//
// Verifies correctness of:
//   - Grand product polynomial construction
//   - Permutation polynomial construction
//   - Fractional grand product (numerator/denominator separation)
//   - Batch grand products for multiple columns
//   - Product check verification
//   - Recurrence verification
//   - Multilinear extension computation and evaluation
//   - CPU fallback and GPU paths

import Foundation
import zkMetal

public func runGPUGrandProductProverTests() {
    let engine = GPUGrandProductProverEngine()

    // ================================================================
    // MARK: - Grand Product Polynomial Construction
    // ================================================================

    suite("GPUGrandProductProver — constructGrandProductPolynomial")

    // Empty input
    do {
        let poly = engine.constructGrandProductPolynomial(values: [])
        expectEqual(poly.evaluations.count, 1, "Empty: evaluations has 1 element")
        expect(frEqual(poly.evaluations[0], Fr.one), "Empty: z[0] == 1")
        expect(frEqual(poly.finalProduct, Fr.one), "Empty: finalProduct == 1")
        expectEqual(poly.domainSize, 0, "Empty: domainSize == 0")
    }

    // Single element
    do {
        let three = frFromInt(3)
        let poly = engine.constructGrandProductPolynomial(values: [three])
        expectEqual(poly.evaluations.count, 2, "Single: evaluations has 2 elements")
        expect(frEqual(poly.evaluations[0], Fr.one), "Single: z[0] == 1")
        expect(frEqual(poly.evaluations[1], three), "Single: z[1] == 3")
        expect(frEqual(poly.finalProduct, three), "Single: finalProduct == 3")
        expectEqual(poly.domainSize, 1, "Single: domainSize == 1")
    }

    // [2, 3, 5] -> z = [1, 2, 6, 30]
    do {
        let vals = [frFromInt(2), frFromInt(3), frFromInt(5)]
        let poly = engine.constructGrandProductPolynomial(values: vals)
        expectEqual(poly.evaluations.count, 4, "[2,3,5]: 4 evaluations")
        expect(frEqual(poly.evaluations[0], Fr.one), "z[0] == 1")
        expect(frEqual(poly.evaluations[1], frFromInt(2)), "z[1] == 2")
        expect(frEqual(poly.evaluations[2], frFromInt(6)), "z[2] == 6")
        expect(frEqual(poly.evaluations[3], frFromInt(30)), "z[3] == 30")
        expect(frEqual(poly.finalProduct, frFromInt(30)), "finalProduct == 30")
    }

    // All ones -> z = [1, 1, 1, ..., 1]
    do {
        let n = 16
        let vals = [Fr](repeating: Fr.one, count: n)
        let poly = engine.constructGrandProductPolynomial(values: vals)
        expectEqual(poly.evaluations.count, n + 1, "All ones: n+1 evaluations")
        for i in 0...n {
            expect(frEqual(poly.evaluations[i], Fr.one), "All ones: z[\(i)] == 1")
        }
        expect(frEqual(poly.finalProduct, Fr.one), "All ones: finalProduct == 1")
    }

    // ================================================================
    // MARK: - Permutation Polynomial Construction
    // ================================================================

    suite("GPUGrandProductProver — constructPermutationPolynomial")

    // Identity permutation: numerators == denominators -> z = all 1s
    do {
        let n = 8
        var vals = [Fr]()
        for i in 1...n { vals.append(frFromInt(UInt64(i))) }
        let poly = engine.constructPermutationPolynomial(numerators: vals, denominators: vals)
        expectEqual(poly.evaluations.count, n + 1, "Identity perm: n+1 evaluations")
        for i in 0...n {
            expect(frEqual(poly.evaluations[i], Fr.one), "Identity perm: z[\(i)] == 1")
        }
        expect(frEqual(poly.finalProduct, Fr.one), "Identity perm: wraps to 1")
    }

    // Simple ratio: nums = [6, 10], dens = [2, 5] -> ratios = [3, 2]
    // z = [1, 3, 6]
    do {
        let nums = [frFromInt(6), frFromInt(10)]
        let dens = [frFromInt(2), frFromInt(5)]
        let poly = engine.constructPermutationPolynomial(numerators: nums, denominators: dens)
        expectEqual(poly.evaluations.count, 3, "Ratio perm: 3 evaluations")
        expect(frEqual(poly.evaluations[0], Fr.one), "Ratio perm: z[0] == 1")
        expect(frEqual(poly.evaluations[1], frFromInt(3)), "Ratio perm: z[1] == 3")
        expect(frEqual(poly.evaluations[2], frFromInt(6)), "Ratio perm: z[2] == 6")
    }

    // Empty inputs
    do {
        let poly = engine.constructPermutationPolynomial(numerators: [], denominators: [])
        expectEqual(poly.evaluations.count, 1, "Empty perm: 1 evaluation")
        expect(frEqual(poly.finalProduct, Fr.one), "Empty perm: finalProduct == 1")
    }

    // ================================================================
    // MARK: - Fractional Grand Product
    // ================================================================

    suite("GPUGrandProductProver — computeFractionalGrandProduct")

    // Balanced: product of nums == product of dens
    do {
        // nums = [2, 3], dens = [3, 2] -> both products = 6
        let nums = [frFromInt(2), frFromInt(3)]
        let dens = [frFromInt(3), frFromInt(2)]
        let frac = engine.computeFractionalGrandProduct(numerators: nums, denominators: dens)

        expect(frac.isBalanced, "Balanced fractional: isBalanced")
        expectEqual(frac.numeratorPoly.count, 2, "Balanced: numPoly count")
        expectEqual(frac.denominatorPoly.count, 2, "Balanced: denPoly count")
        expectEqual(frac.ratioPoly.count, 2, "Balanced: ratioPoly count")

        // numPoly = [1, 2], denPoly = [1, 3]
        expect(frEqual(frac.numeratorPoly[0], Fr.one), "numPoly[0] == 1")
        expect(frEqual(frac.numeratorPoly[1], frFromInt(2)), "numPoly[1] == 2")
        expect(frEqual(frac.denominatorPoly[0], Fr.one), "denPoly[0] == 1")
        expect(frEqual(frac.denominatorPoly[1], frFromInt(3)), "denPoly[1] == 3")

        // ratioPoly[0] = 1/1 = 1, ratioPoly[1] = 2/3
        expect(frEqual(frac.ratioPoly[0], Fr.one), "ratioPoly[0] == 1")
        let twoThirds = frMul(frFromInt(2), frInverse(frFromInt(3)))
        expect(frEqual(frac.ratioPoly[1], twoThirds), "ratioPoly[1] == 2/3")
    }

    // Unbalanced: product of nums != product of dens
    do {
        let nums = [frFromInt(2), frFromInt(3)]
        let dens = [frFromInt(1), frFromInt(1)]
        let frac = engine.computeFractionalGrandProduct(numerators: nums, denominators: dens)
        expect(!frac.isBalanced, "Unbalanced fractional: not balanced")
    }

    // Empty inputs
    do {
        let frac = engine.computeFractionalGrandProduct(numerators: [], denominators: [])
        expect(frac.isBalanced, "Empty fractional: balanced")
        expectEqual(frac.ratioPoly.count, 1, "Empty fractional: ratioPoly has 1 element")
    }

    // ================================================================
    // MARK: - Batch Grand Products
    // ================================================================

    suite("GPUGrandProductProver — batchGrandProducts")

    // Empty columns
    do {
        let result = engine.batchGrandProducts(columns: [])
        expectEqual(result.polynomials.count, 0, "Empty batch: no polynomials")
        expect(result.allProductsValid, "Empty batch: all valid")
    }

    // Single column with all ones
    do {
        let col = [Fr](repeating: Fr.one, count: 4)
        let result = engine.batchGrandProducts(columns: [col])
        expectEqual(result.polynomials.count, 1, "Single column: 1 polynomial")
        expect(result.allProductsValid, "All ones: valid")
    }

    // Two columns, one wraps to 1, one does not
    do {
        let col1 = [Fr](repeating: Fr.one, count: 4)      // product = 1
        let col2 = [frFromInt(2), frFromInt(3)]             // product = 6
        let result = engine.batchGrandProducts(columns: [col1, col2])
        expectEqual(result.polynomials.count, 2, "Two columns: 2 polynomials")
        expect(!result.allProductsValid, "Mixed columns: not all valid")
    }

    // Multiple columns all wrapping to 1
    do {
        let n = 4
        var columns = [[Fr]]()
        for _ in 0..<3 {
            columns.append([Fr](repeating: Fr.one, count: n))
        }
        let result = engine.batchGrandProducts(columns: columns)
        expectEqual(result.polynomials.count, 3, "3 columns: 3 polynomials")
        expect(result.allProductsValid, "All-ones columns: all valid")
    }

    // ================================================================
    // MARK: - Batch Permutation Products
    // ================================================================

    suite("GPUGrandProductProver — batchPermutationProducts")

    // Identity permutation across columns
    do {
        let n = 4
        var vals = [Fr]()
        for i in 1...n { vals.append(frFromInt(UInt64(i))) }
        let result = engine.batchPermutationProducts(
            numeratorColumns: [vals, vals],
            denominatorColumns: [vals, vals]
        )
        expectEqual(result.polynomials.count, 2, "Batch perm: 2 polynomials")
        expect(result.allProductsValid, "Identity batch perm: all valid")
    }

    // ================================================================
    // MARK: - Product Check Verification
    // ================================================================

    suite("GPUGrandProductProver — verifyProductCheck")

    // Valid product check: z[0] = 1, z[n] = 1
    do {
        let poly = GrandProductPolynomial(
            evaluations: [Fr.one, frFromInt(3), frFromInt(6), Fr.one],
            finalProduct: Fr.one,
            domainSize: 3
        )
        expect(engine.verifyProductCheck(poly), "Valid product check passes")
    }

    // Invalid: z[n] != 1
    do {
        let poly = GrandProductPolynomial(
            evaluations: [Fr.one, frFromInt(2), frFromInt(6)],
            finalProduct: frFromInt(6),
            domainSize: 2
        )
        expect(!engine.verifyProductCheck(poly), "Invalid product check: z[n] != 1")
    }

    // Invalid: z[0] != 1
    do {
        let poly = GrandProductPolynomial(
            evaluations: [frFromInt(2), frFromInt(4), Fr.one],
            finalProduct: Fr.one,
            domainSize: 2
        )
        expect(!engine.verifyProductCheck(poly), "Invalid product check: z[0] != 1")
    }

    // Empty polynomial
    do {
        let poly = GrandProductPolynomial(evaluations: [], finalProduct: Fr.one, domainSize: 0)
        expect(engine.verifyProductCheck(poly), "Empty polynomial: trivially valid")
    }

    // ================================================================
    // MARK: - Recurrence Verification
    // ================================================================

    suite("GPUGrandProductProver — verifyRecurrence")

    // Valid recurrence: z[i+1] = z[i] * values[i]
    do {
        let vals = [frFromInt(2), frFromInt(3), frFromInt(5)]
        let poly = engine.constructGrandProductPolynomial(values: vals)
        expect(engine.verifyRecurrence(polynomial: poly, values: vals),
               "Recurrence valid for constructed polynomial")
    }

    // Invalid recurrence: tampered evaluations
    do {
        let vals = [frFromInt(2), frFromInt(3)]
        let tampered = GrandProductPolynomial(
            evaluations: [Fr.one, frFromInt(2), frFromInt(99)],
            finalProduct: frFromInt(99),
            domainSize: 2
        )
        expect(!engine.verifyRecurrence(polynomial: tampered, values: vals),
               "Tampered recurrence fails verification")
    }

    // Mismatched sizes
    do {
        let vals = [frFromInt(2)]
        let poly = GrandProductPolynomial(
            evaluations: [Fr.one, frFromInt(2), frFromInt(6)],
            finalProduct: frFromInt(6),
            domainSize: 2
        )
        expect(!engine.verifyRecurrence(polynomial: poly, values: vals),
               "Size mismatch fails verification")
    }

    // ================================================================
    // MARK: - Fractional Balance Verification
    // ================================================================

    suite("GPUGrandProductProver — verifyFractionalBalance")

    do {
        let balanced = FractionalGrandProduct(
            numeratorPoly: [Fr.one], denominatorPoly: [Fr.one],
            ratioPoly: [Fr.one], isBalanced: true
        )
        expect(engine.verifyFractionalBalance(balanced), "Balanced passes")

        let unbalanced = FractionalGrandProduct(
            numeratorPoly: [Fr.one], denominatorPoly: [Fr.one],
            ratioPoly: [Fr.one], isBalanced: false
        )
        expect(!engine.verifyFractionalBalance(unbalanced), "Unbalanced fails")
    }

    // ================================================================
    // MARK: - Multilinear Extension
    // ================================================================

    suite("GPUGrandProductProver — multilinearExtension")

    // MLE of 4-element polynomial: z = [1, 2, 6, 30] -> pad to 4 (already pow2)
    do {
        let vals = [frFromInt(2), frFromInt(3), frFromInt(5)]
        let poly = engine.constructGrandProductPolynomial(values: vals)
        // evaluations = [1, 2, 6, 30], length 4 = 2^2
        let mle = engine.multilinearExtension(of: poly)
        expectEqual(mle.numVariables, 2, "MLE of 4 elements: 2 variables")
        expectEqual(mle.evaluations.count, 4, "MLE: 4 evaluations")

        // Check that MLE at binary points recovers the original evaluations
        // MLE(0,0) = evals[0] = 1
        let v00 = engine.evaluateMLE(mle, at: [Fr.zero, Fr.zero])
        expect(frEqual(v00, Fr.one), "MLE(0,0) == z[0] = 1")

        // MLE(1,0) = evals[1] = 2
        let v10 = engine.evaluateMLE(mle, at: [Fr.one, Fr.zero])
        expect(frEqual(v10, frFromInt(2)), "MLE(1,0) == z[1] = 2")

        // MLE(0,1) = evals[2] = 6
        let v01 = engine.evaluateMLE(mle, at: [Fr.zero, Fr.one])
        expect(frEqual(v01, frFromInt(6)), "MLE(0,1) == z[2] = 6")

        // MLE(1,1) = evals[3] = 30
        let v11 = engine.evaluateMLE(mle, at: [Fr.one, Fr.one])
        expect(frEqual(v11, frFromInt(30)), "MLE(1,1) == z[3] = 30")
    }

    // MLE evaluation at non-binary point (interpolation)
    do {
        // 2 elements -> pad to 2 = 2^1, so 1 variable
        let poly = GrandProductPolynomial(
            evaluations: [frFromInt(3), frFromInt(7)],
            finalProduct: frFromInt(7),
            domainSize: 1
        )
        let mle = engine.multilinearExtension(of: poly)
        expectEqual(mle.numVariables, 1, "MLE of 2 elements: 1 variable")

        // MLE(r) = 3*(1-r) + 7*r = 3 + 4r
        // At r=0: 3, r=1: 7
        let v0 = engine.evaluateMLE(mle, at: [Fr.zero])
        expect(frEqual(v0, frFromInt(3)), "MLE(0) == 3")
        let v1 = engine.evaluateMLE(mle, at: [Fr.one])
        expect(frEqual(v1, frFromInt(7)), "MLE(1) == 7")

        // At r=2: 3 + 4*2 = 11
        let v2 = engine.evaluateMLE(mle, at: [frFromInt(2)])
        expect(frEqual(v2, frFromInt(11)), "MLE(2) == 11")
    }

    // MLE with padding (non-power-of-two input)
    do {
        // 3 evaluations -> pad to 4 = 2^2
        let poly = GrandProductPolynomial(
            evaluations: [frFromInt(1), frFromInt(2), frFromInt(3)],
            finalProduct: frFromInt(3),
            domainSize: 2
        )
        let mle = engine.multilinearExtension(of: poly)
        expectEqual(mle.numVariables, 2, "MLE of 3 elements: 2 variables (padded)")
        expectEqual(mle.evaluations.count, 4, "MLE: padded to 4")
        // evals[3] should be zero (padding)
        expect(frEqual(mle.evaluations[3], Fr.zero), "MLE: padding is zero")
    }

    // Convenience method: evaluateGrandProductMLE
    do {
        let vals = [frFromInt(2), frFromInt(3), frFromInt(5)]
        let poly = engine.constructGrandProductPolynomial(values: vals)
        let result = engine.evaluateGrandProductMLE(polynomial: poly, at: [Fr.zero, Fr.zero])
        expect(frEqual(result, Fr.one), "evaluateGrandProductMLE(0,0) == 1")
    }

    // ================================================================
    // MARK: - Larger Inputs (exercises GPU path if available)
    // ================================================================

    suite("GPUGrandProductProver — larger inputs")

    // Larger prefix product correctness
    do {
        let n = 2048
        var vals = [Fr]()
        var rng: UInt64 = 0xDEAD_BEEF_CAFE_1234
        for _ in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            var limbs: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32) = (0,0,0,0,0,0,0,0)
            withUnsafeMutableBytes(of: &limbs) { buf in
                let ptr = buf.bindMemory(to: UInt64.self)
                for j in 0..<4 {
                    rng = rng &* 6364136223846793005 &+ 1442695040888963407
                    ptr[j] = rng
                }
            }
            limbs.7 &= 0x0FFFFFFF
            var elem = Fr(v: limbs)
            if elem.isZero { elem = Fr.one }
            vals.append(elem)
        }

        let poly = engine.constructGrandProductPolynomial(values: vals)
        expectEqual(poly.evaluations.count, n + 1, "Large: n+1 evaluations")
        expect(frEqual(poly.evaluations[0], Fr.one), "Large: z[0] == 1")

        // Verify recurrence at sample points
        var ok = true
        let checks = [0, 1, 10, 100, 500, 1000, n - 1]
        for i in checks {
            let expected = frMul(poly.evaluations[i], vals[i])
            if !frEqual(poly.evaluations[i + 1], expected) {
                ok = false
            }
        }
        expect(ok, "Large: recurrence valid at sample points")

        // Full recurrence check
        expect(engine.verifyRecurrence(polynomial: poly, values: vals),
               "Large: full recurrence verification passes")
    }

    // Larger permutation product (identity permutation)
    do {
        let n = 1024
        var vals = [Fr]()
        for i in 1...n { vals.append(frFromInt(UInt64(i))) }
        let poly = engine.constructPermutationPolynomial(numerators: vals, denominators: vals)
        expect(frEqual(poly.finalProduct, Fr.one), "Large identity perm: wraps to 1")
        expect(engine.verifyProductCheck(poly), "Large identity perm: product check passes")
    }

    // Larger fractional grand product (balanced)
    do {
        let n = 512
        var nums = [Fr]()
        var dens = [Fr]()
        // Create balanced pairs: nums is a permutation of dens
        for i in 1...n { nums.append(frFromInt(UInt64(i))) }
        // Reverse order
        for i in stride(from: n, through: 1, by: -1) { dens.append(frFromInt(UInt64(i))) }

        let frac = engine.computeFractionalGrandProduct(numerators: nums, denominators: dens)
        expect(frac.isBalanced, "Large balanced fractional: isBalanced")
    }

    // Batch with larger columns
    do {
        let n = 256
        let numCols = 4
        var numCols_ = [[Fr]]()
        var denCols_ = [[Fr]]()
        for _ in 0..<numCols {
            var vals = [Fr]()
            for i in 1...n { vals.append(frFromInt(UInt64(i))) }
            numCols_.append(vals)
            denCols_.append(vals)  // identity permutation
        }
        let result = engine.batchPermutationProducts(
            numeratorColumns: numCols_,
            denominatorColumns: denCols_
        )
        expectEqual(result.polynomials.count, numCols, "Batch large: correct column count")
        expect(result.allProductsValid, "Batch large identity: all valid")
    }

    // ================================================================
    // MARK: - GPU availability
    // ================================================================

    suite("GPUGrandProductProver — GPU availability")

    do {
        // Just report GPU status; engine works either way
        if engine.isGPUAvailable {
            print("  [INFO] GPU acceleration available")
        } else {
            print("  [INFO] CPU-only mode (no Metal device)")
        }
        expect(true, "Engine initialized successfully")
    }
}
