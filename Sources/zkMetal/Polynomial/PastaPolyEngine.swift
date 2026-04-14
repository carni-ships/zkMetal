// Pasta polynomial operations — CPU-side Swift wrappers for C CIOS kernels.
// Supports both Pallas Fr (= Vesta Fp) and Vesta Fr (= Pallas Fp) scalar fields.
// Used for commitment evaluation, opening proofs, and Kimchi/Pickles prover.

import Foundation
import NeonFieldOps

// MARK: - Pallas Fr polynomial operations (scalar field of Pallas = base field of Vesta)
// PallasFr is represented as VestaFp (cycle property)

public enum PallasPolyEngine {

    // MARK: Horner evaluation

    /// Evaluate polynomial at a point using Horner's method.
    /// p(z) = coeffs[0] + coeffs[1]*z + ... + coeffs[n-1]*z^(n-1)
    public static func evaluate(_ coeffs: [VestaFp], at z: VestaFp) -> VestaFp {
        let n = coeffs.count
        if n == 0 { return VestaFp.zero }
        var result = VestaFp.zero
        coeffs.withUnsafeBytes { cBuf in
            withUnsafeBytes(of: z) { zBuf in
                withUnsafeMutableBytes(of: &result) { rBuf in
                    pallas_fr_horner_eval(
                        cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n),
                        zBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                    )
                }
            }
        }
        return result
    }

    // MARK: Synthetic division

    /// Synthetic division: q(x) = (p(x) - p(z)) / (x - z)
    /// Returns quotient polynomial of degree n-2 (n-1 coefficients).
    public static func syntheticDiv(_ coeffs: [VestaFp], z: VestaFp) -> [VestaFp] {
        let n = coeffs.count
        if n < 2 { return [] }
        var quotient = [VestaFp](repeating: VestaFp.zero, count: n - 1)
        coeffs.withUnsafeBytes { cBuf in
            withUnsafeBytes(of: z) { zBuf in
                quotient.withUnsafeMutableBytes { qBuf in
                    pallas_fr_synthetic_div(
                        cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        zBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n),
                        qBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                    )
                }
            }
        }
        return quotient
    }

    // MARK: Fused eval + div

    /// Fused evaluation and synthetic division in a single pass.
    /// Returns (p(z), quotient) where quotient = (p(x) - p(z)) / (x - z).
    public static func evalAndDiv(_ coeffs: [VestaFp], z: VestaFp) -> (VestaFp, [VestaFp]) {
        let n = coeffs.count
        if n == 0 { return (VestaFp.zero, []) }
        if n == 1 { return (coeffs[0], []) }
        var eval = VestaFp.zero
        var quotient = [VestaFp](repeating: VestaFp.zero, count: n - 1)
        coeffs.withUnsafeBytes { cBuf in
            withUnsafeBytes(of: z) { zBuf in
                withUnsafeMutableBytes(of: &eval) { eBuf in
                    quotient.withUnsafeMutableBytes { qBuf in
                        pallas_fr_eval_and_div(
                            cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(n),
                            zBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            eBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            qBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                        )
                    }
                }
            }
        }
        return (eval, quotient)
    }

    // MARK: Batch operations

    /// Batch scalar multiply (in-place): data[i] *= scalar
    public static func batchMulScalar(_ data: inout [VestaFp], scalar: VestaFp) {
        let n = data.count
        if n == 0 { return }
        data.withUnsafeMutableBytes { dBuf in
            withUnsafeBytes(of: scalar) { sBuf in
                pallas_fr_batch_mul_scalar(
                    dBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    sBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(n)
                )
            }
        }
    }

    /// Batch add: result[i] = a[i] + b[i]
    public static func batchAdd(_ a: [VestaFp], _ b: [VestaFp]) -> [VestaFp] {
        let n = min(a.count, b.count)
        if n == 0 { return [] }
        var result = [VestaFp](repeating: VestaFp.zero, count: n)
        a.withUnsafeBytes { aBuf in
            b.withUnsafeBytes { bBuf in
                result.withUnsafeMutableBytes { rBuf in
                    pallas_fr_batch_add(
                        aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n)
                    )
                }
            }
        }
        return result
    }

    /// Batch sub: result[i] = a[i] - b[i]
    public static func batchSub(_ a: [VestaFp], _ b: [VestaFp]) -> [VestaFp] {
        let n = min(a.count, b.count)
        if n == 0 { return [] }
        var result = [VestaFp](repeating: VestaFp.zero, count: n)
        a.withUnsafeBytes { aBuf in
            b.withUnsafeBytes { bBuf in
                result.withUnsafeMutableBytes { rBuf in
                    pallas_fr_batch_sub(
                        aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n)
                    )
                }
            }
        }
        return result
    }

    /// Inner product: sum(a[i] * b[i])
    public static func innerProduct(_ a: [VestaFp], _ b: [VestaFp]) -> VestaFp {
        let n = min(a.count, b.count)
        if n == 0 { return VestaFp.zero }
        var result = VestaFp.zero
        a.withUnsafeBytes { aBuf in
            b.withUnsafeBytes { bBuf in
                withUnsafeMutableBytes(of: &result) { rBuf in
                    pallas_fr_inner_product(
                        aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                    )
                }
            }
        }
        return result
    }

    // MARK: Polynomial multiplication (naive for small degrees)

    /// Naive polynomial multiplication: c(x) = a(x) * b(x)
    /// O(n*m) — suitable for small polynomials.
    public static func multiply(_ a: [VestaFp], _ b: [VestaFp]) -> [VestaFp] {
        let na = a.count, nb = b.count
        if na == 0 || nb == 0 { return [] }
        var result = [VestaFp](repeating: VestaFp.zero, count: na + nb - 1)
        for i in 0..<na {
            for j in 0..<nb {
                let prod = vestaMul(a[i], b[j])
                result[i + j] = vestaAdd(result[i + j], prod)
            }
        }
        return result
    }

    // MARK: Lagrange interpolation

    /// Lagrange interpolation over distinct points.
    /// Given (x_i, y_i) pairs, returns polynomial coefficients [c_0, c_1, ..., c_{n-1}].
    public static func lagrangeInterpolation(xs: [VestaFp], ys: [VestaFp]) -> [VestaFp] {
        let n = xs.count
        precondition(n == ys.count && n > 0, "xs and ys must be same nonzero length")
        if n == 1 { return [ys[0]] }

        var result = [VestaFp](repeating: VestaFp.zero, count: n)

        for i in 0..<n {
            // Compute Lagrange basis polynomial l_i(x) = product_{j!=i} (x - x_j) / (x_i - x_j)
            // First compute denominator: product_{j!=i} (x_i - x_j)
            var denom = VestaFp.one
            for j in 0..<n {
                if j == i { continue }
                denom = vestaMul(denom, vestaSub(xs[i], xs[j]))
            }
            let denomInv = vestaInverse(denom)
            let scale = vestaMul(ys[i], denomInv)

            // Build numerator polynomial: product_{j!=i} (x - x_j)
            // Start with [1] and multiply by (x - x_j) = [-x_j, 1] for each j != i
            var basis: [VestaFp] = [VestaFp.one]
            for j in 0..<n {
                if j == i { continue }
                let negXj = vestaNeg(xs[j])
                // Multiply basis by (x - x_j): new[k] = basis[k-1] + negXj * basis[k]
                var newBasis = [VestaFp](repeating: VestaFp.zero, count: basis.count + 1)
                for k in 0..<basis.count {
                    newBasis[k] = vestaAdd(newBasis[k], vestaMul(negXj, basis[k]))
                    newBasis[k + 1] = vestaAdd(newBasis[k + 1], basis[k])
                }
                basis = newBasis
            }

            // Add scale * basis to result
            for k in 0..<basis.count {
                result[k] = vestaAdd(result[k], vestaMul(scale, basis[k]))
            }
        }

        return result
    }
}

// MARK: - Vesta Fr polynomial operations (scalar field of Vesta = base field of Pallas)
// VestaFr is represented as PallasFp (cycle property)

public enum VestaPolyEngine {

    // MARK: Horner evaluation

    /// Evaluate polynomial at a point using Horner's method.
    public static func evaluate(_ coeffs: [PallasFp], at z: PallasFp) -> PallasFp {
        let n = coeffs.count
        if n == 0 { return PallasFp.zero }
        var result = PallasFp.zero
        coeffs.withUnsafeBytes { cBuf in
            withUnsafeBytes(of: z) { zBuf in
                withUnsafeMutableBytes(of: &result) { rBuf in
                    vesta_fr_horner_eval(
                        cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n),
                        zBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                    )
                }
            }
        }
        return result
    }

    // MARK: Synthetic division

    /// Synthetic division: q(x) = (p(x) - p(z)) / (x - z)
    public static func syntheticDiv(_ coeffs: [PallasFp], z: PallasFp) -> [PallasFp] {
        let n = coeffs.count
        if n < 2 { return [] }
        var quotient = [PallasFp](repeating: PallasFp.zero, count: n - 1)
        coeffs.withUnsafeBytes { cBuf in
            withUnsafeBytes(of: z) { zBuf in
                quotient.withUnsafeMutableBytes { qBuf in
                    vesta_fr_synthetic_div(
                        cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        zBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n),
                        qBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                    )
                }
            }
        }
        return quotient
    }

    // MARK: Fused eval + div

    /// Fused evaluation and synthetic division.
    public static func evalAndDiv(_ coeffs: [PallasFp], z: PallasFp) -> (PallasFp, [PallasFp]) {
        let n = coeffs.count
        if n == 0 { return (PallasFp.zero, []) }
        if n == 1 { return (coeffs[0], []) }
        var eval = PallasFp.zero
        var quotient = [PallasFp](repeating: PallasFp.zero, count: n - 1)
        coeffs.withUnsafeBytes { cBuf in
            withUnsafeBytes(of: z) { zBuf in
                withUnsafeMutableBytes(of: &eval) { eBuf in
                    quotient.withUnsafeMutableBytes { qBuf in
                        vesta_fr_eval_and_div(
                            cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(n),
                            zBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            eBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            qBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                        )
                    }
                }
            }
        }
        return (eval, quotient)
    }

    // MARK: Batch operations

    /// Batch scalar multiply (in-place): data[i] *= scalar
    public static func batchMulScalar(_ data: inout [PallasFp], scalar: PallasFp) {
        let n = data.count
        if n == 0 { return }
        data.withUnsafeMutableBytes { dBuf in
            withUnsafeBytes(of: scalar) { sBuf in
                vesta_fr_batch_mul_scalar(
                    dBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    sBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(n)
                )
            }
        }
    }

    /// Batch add: result[i] = a[i] + b[i]
    public static func batchAdd(_ a: [PallasFp], _ b: [PallasFp]) -> [PallasFp] {
        let n = min(a.count, b.count)
        if n == 0 { return [] }
        var result = [PallasFp](repeating: PallasFp.zero, count: n)
        a.withUnsafeBytes { aBuf in
            b.withUnsafeBytes { bBuf in
                result.withUnsafeMutableBytes { rBuf in
                    vesta_fr_batch_add(
                        aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n)
                    )
                }
            }
        }
        return result
    }

    /// Batch sub: result[i] = a[i] - b[i]
    public static func batchSub(_ a: [PallasFp], _ b: [PallasFp]) -> [PallasFp] {
        let n = min(a.count, b.count)
        if n == 0 { return [] }
        var result = [PallasFp](repeating: PallasFp.zero, count: n)
        a.withUnsafeBytes { aBuf in
            b.withUnsafeBytes { bBuf in
                result.withUnsafeMutableBytes { rBuf in
                    vesta_fr_batch_sub(
                        aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n)
                    )
                }
            }
        }
        return result
    }

    /// Inner product: sum(a[i] * b[i])
    public static func innerProduct(_ a: [PallasFp], _ b: [PallasFp]) -> PallasFp {
        let n = min(a.count, b.count)
        if n == 0 { return PallasFp.zero }
        var result = PallasFp.zero
        a.withUnsafeBytes { aBuf in
            b.withUnsafeBytes { bBuf in
                withUnsafeMutableBytes(of: &result) { rBuf in
                    vesta_fr_inner_product(
                        aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                    )
                }
            }
        }
        return result
    }

    // MARK: Polynomial multiplication (naive for small degrees)

    /// Naive polynomial multiplication: c(x) = a(x) * b(x)
    public static func multiply(_ a: [PallasFp], _ b: [PallasFp]) -> [PallasFp] {
        let na = a.count, nb = b.count
        if na == 0 || nb == 0 { return [] }
        var result = [PallasFp](repeating: PallasFp.zero, count: na + nb - 1)
        for i in 0..<na {
            for j in 0..<nb {
                let prod = pallasMul(a[i], b[j])
                result[i + j] = pallasAdd(result[i + j], prod)
            }
        }
        return result
    }

    // MARK: Lagrange interpolation

    /// Lagrange interpolation over distinct points.
    public static func lagrangeInterpolation(xs: [PallasFp], ys: [PallasFp]) -> [PallasFp] {
        let n = xs.count
        precondition(n == ys.count && n > 0, "xs and ys must be same nonzero length")
        if n == 1 { return [ys[0]] }

        var result = [PallasFp](repeating: PallasFp.zero, count: n)

        for i in 0..<n {
            var denom = PallasFp.one
            for j in 0..<n {
                if j == i { continue }
                denom = pallasMul(denom, pallasSub(xs[i], xs[j]))
            }
            let denomInv = pallasInverse(denom)
            let scale = pallasMul(ys[i], denomInv)

            var basis: [PallasFp] = [PallasFp.one]
            for j in 0..<n {
                if j == i { continue }
                let negXj = pallasNeg(xs[j])
                var newBasis = [PallasFp](repeating: PallasFp.zero, count: basis.count + 1)
                for k in 0..<basis.count {
                    newBasis[k] = pallasAdd(newBasis[k], pallasMul(negXj, basis[k]))
                    newBasis[k + 1] = pallasAdd(newBasis[k + 1], basis[k])
                }
                basis = newBasis
            }

            for k in 0..<basis.count {
                result[k] = pallasAdd(result[k], pallasMul(scale, basis[k]))
            }
        }

        return result
    }
}
