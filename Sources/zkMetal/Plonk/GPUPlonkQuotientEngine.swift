// GPUPlonkQuotientEngine — GPU-accelerated Plonk quotient polynomial computation
//
// Computes the quotient polynomial t(X) for the Plonk protocol:
//   t(X) * Z_H(X) = gate_identity(X) + alpha * perm_argument(X) + alpha^2 * boundary(X)
//
// The gate identity at each row is:
//   qL*a + qR*b + qO*c + qM*a*b + qC = 0
//
// The permutation argument contribution uses the grand product accumulator Z(X):
//   (a + beta*id1 + gamma)(b + beta*id2 + gamma)(c + beta*id3 + gamma) * Z(X)
//   - (a + beta*sigma1 + gamma)(b + beta*sigma2 + gamma)(c + beta*sigma3 + gamma) * Z(omega*X)
//
// The boundary constraint enforces Z(1) = 1:
//   (Z(X) - 1) * L_1(X)
//
// After computing the numerator on a coset (to avoid vanishing polynomial zeros),
// we divide by Z_H(X) = X^n - 1 and split the result into degree-n chunks:
//   t(X) = t_lo(X) + X^n * t_mid(X) + X^{2n} * t_hi(X) + ...
//
// GPU acceleration:
//   - Metal compute kernel evaluates the full constraint numerator at each coset point
//   - Parallel vanishing polynomial inversion via batch inverse
//   - Coset NTT for fast polynomial evaluation on shifted domain
//   - CPU fallback for small domains where Metal dispatch overhead dominates
//
// Custom gate contributions (evaluated in parallel):
//   - Range: qRange * a * (1 - a)
//   - Lookup: qLookup * prod(a - t_i) for each table value t_i
//   - Poseidon: qPoseidon * (c - a^5) where a^5 = a*a*a*a*a

import Foundation
import Metal
import NeonFieldOps

// MARK: - Quotient Configuration

/// Configuration for quotient polynomial computation, controlling GPU dispatch
/// thresholds, coset generator selection, and custom gate activation.
public struct QuotientConfig {
    /// Domain size (must be power of 2, >= 4)
    public let domainSize: Int
    /// Log2 of domain size
    public let logN: Int
    /// Coset generator (multiplicative shift for evaluation domain)
    public let cosetGenerator: Fr
    /// Number of quotient chunks (typically 3 for standard Plonk)
    public let numChunks: Int
    /// Whether to include range gate contributions
    public let enableRangeGates: Bool
    /// Whether to include lookup gate contributions
    public let enableLookupGates: Bool
    /// Whether to include Poseidon S-box gate contributions
    public let enablePoseidonGates: Bool

    public init(domainSize: Int, cosetGenerator: Fr? = nil, numChunks: Int = 3,
                enableRangeGates: Bool = true, enableLookupGates: Bool = true,
                enablePoseidonGates: Bool = true) {
        precondition(domainSize >= 4 && domainSize & (domainSize - 1) == 0,
                     "Domain size must be a power of 2, >= 4")
        self.domainSize = domainSize
        self.logN = Int(log2(Double(domainSize)))
        // Default coset generator: a small generator that avoids the n-th roots
        self.cosetGenerator = cosetGenerator ?? frFromInt(7)
        self.numChunks = numChunks
        self.enableRangeGates = enableRangeGates
        self.enableLookupGates = enableLookupGates
        self.enablePoseidonGates = enablePoseidonGates
    }
}

// MARK: - Quotient Result

/// Result of quotient polynomial computation.
public struct QuotientResult {
    /// Quotient polynomial chunks in coefficient form.
    /// chunks[0] = t_lo, chunks[1] = t_mid, chunks[2] = t_hi, etc.
    /// Each chunk has at most domainSize coefficients.
    public let chunks: [[Fr]]
    /// Full quotient polynomial in coefficient form (before splitting).
    public let fullQuotient: [Fr]
    /// Gate constraint evaluations on the coset (for debugging).
    public let gateEvals: [Fr]
    /// Permutation constraint evaluations on the coset (for debugging).
    public let permEvals: [Fr]
    /// Whether computation used the GPU path.
    public let usedGPU: Bool

    public init(chunks: [[Fr]], fullQuotient: [Fr], gateEvals: [Fr],
                permEvals: [Fr], usedGPU: Bool) {
        self.chunks = chunks
        self.fullQuotient = fullQuotient
        self.gateEvals = gateEvals
        self.permEvals = permEvals
        self.usedGPU = usedGPU
    }
}

// MARK: - Coset Domain

/// Precomputed coset domain: {g * omega^0, g * omega^1, ..., g * omega^{n-1}}
/// where g is the coset generator and omega is the n-th root of unity.
public struct CosetDomainData {
    /// Coset points in order
    public let points: [Fr]
    /// Vanishing polynomial evaluations Z_H(x) = x^n - 1 at each coset point
    public let vanishingEvals: [Fr]
    /// Inverse of vanishing polynomial at each coset point (for division)
    public let vanishingInvs: [Fr]
    /// The n-th root of unity
    public let omega: Fr
    /// The coset generator
    public let generator: Fr
    /// Domain size
    public let n: Int

    public init(points: [Fr], vanishingEvals: [Fr], vanishingInvs: [Fr],
                omega: Fr, generator: Fr, n: Int) {
        self.points = points
        self.vanishingEvals = vanishingEvals
        self.vanishingInvs = vanishingInvs
        self.omega = omega
        self.generator = generator
        self.n = n
    }
}

// MARK: - Constraint Numerator Components

/// All polynomial evaluations needed on the coset for constraint numerator computation.
public struct CosetConstraintInputs {
    /// Wire polynomial evaluations on coset: a(gw^i), b(gw^i), c(gw^i)
    public let aEvals: [Fr]
    public let bEvals: [Fr]
    public let cEvals: [Fr]
    /// Selector polynomial evaluations on coset
    public let qLEvals: [Fr]
    public let qREvals: [Fr]
    public let qOEvals: [Fr]
    public let qMEvals: [Fr]
    public let qCEvals: [Fr]
    public let qRangeEvals: [Fr]
    public let qLookupEvals: [Fr]
    public let qPoseidonEvals: [Fr]
    /// Permutation polynomial evaluations on coset
    public let sigma1Evals: [Fr]
    public let sigma2Evals: [Fr]
    public let sigma3Evals: [Fr]
    /// Grand product Z(X) evaluations on coset
    public let zEvals: [Fr]
    /// Shifted grand product Z(omega*X) evaluations on coset
    public let zShiftedEvals: [Fr]
    /// First Lagrange basis L_1(X) evaluations on coset
    public let l1Evals: [Fr]
    /// Identity permutation evaluations id_1, id_2, id_3 on coset
    public let id1Evals: [Fr]
    public let id2Evals: [Fr]
    public let id3Evals: [Fr]

    public init(aEvals: [Fr], bEvals: [Fr], cEvals: [Fr],
                qLEvals: [Fr], qREvals: [Fr], qOEvals: [Fr],
                qMEvals: [Fr], qCEvals: [Fr],
                qRangeEvals: [Fr], qLookupEvals: [Fr], qPoseidonEvals: [Fr],
                sigma1Evals: [Fr], sigma2Evals: [Fr], sigma3Evals: [Fr],
                zEvals: [Fr], zShiftedEvals: [Fr], l1Evals: [Fr],
                id1Evals: [Fr], id2Evals: [Fr], id3Evals: [Fr]) {
        self.aEvals = aEvals; self.bEvals = bEvals; self.cEvals = cEvals
        self.qLEvals = qLEvals; self.qREvals = qREvals; self.qOEvals = qOEvals
        self.qMEvals = qMEvals; self.qCEvals = qCEvals
        self.qRangeEvals = qRangeEvals; self.qLookupEvals = qLookupEvals
        self.qPoseidonEvals = qPoseidonEvals
        self.sigma1Evals = sigma1Evals; self.sigma2Evals = sigma2Evals
        self.sigma3Evals = sigma3Evals
        self.zEvals = zEvals; self.zShiftedEvals = zShiftedEvals
        self.l1Evals = l1Evals
        self.id1Evals = id1Evals; self.id2Evals = id2Evals; self.id3Evals = id3Evals
    }
}

// MARK: - GPUPlonkQuotientEngine

/// GPU-accelerated engine for computing the Plonk quotient polynomial.
///
/// The quotient polynomial t(X) encodes all constraint satisfaction information:
///   t(X) = [gate(X) + alpha*perm(X) + alpha^2*boundary(X)] / Z_H(X)
///
/// This engine evaluates the full numerator on a coset domain (avoiding Z_H roots),
/// divides pointwise by Z_H, then converts back to coefficient form via iNTT.
public class GPUPlonkQuotientEngine {
    public static let version = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")

    /// Minimum domain size to dispatch to GPU (below this, CPU is faster).
    private static let gpuThreshold = 512

    private let device: MTLDevice?
    private let commandQueue: MTLCommandQueue?
    private let constraintPipeline: MTLComputePipelineState?
    private let threadgroupSize: Int

    // MARK: - Initialization

    public init() {
        let dev = MTLCreateSystemDefaultDevice()
        self.device = dev
        self.commandQueue = dev?.makeCommandQueue()
        self.threadgroupSize = 256

        if let dev = dev {
            self.constraintPipeline = GPUPlonkQuotientEngine.compileConstraintKernel(device: dev)
        } else {
            self.constraintPipeline = nil
        }
    }

    // MARK: - Coset Domain Construction

    /// Build a coset evaluation domain with precomputed vanishing polynomial values.
    ///
    /// The coset is {g * omega^0, g * omega^1, ..., g * omega^{n-1}} where g is
    /// the coset generator. The vanishing polynomial Z_H(X) = X^n - 1 is nonzero
    /// on this coset (since g is not an n-th root of unity), allowing safe division.
    ///
    /// - Parameter config: Quotient configuration with domain size and coset generator.
    /// - Returns: Precomputed coset domain data.
    public func buildCosetDomain(config: QuotientConfig) -> CosetDomainData {
        let n = config.domainSize
        let omega = frRootOfUnity(logN: config.logN)
        let g = config.cosetGenerator

        // Build coset points: g, g*omega, g*omega^2, ..., g*omega^{n-1}
        var points = [Fr](repeating: Fr.zero, count: n)
        points[0] = g
        for i in 1..<n {
            points[i] = frMul(points[i - 1], omega)
        }

        // Vanishing polynomial: Z_H(x) = x^n - 1 at each coset point
        // Since (g*omega^i)^n = g^n * (omega^n)^i = g^n * 1 = g^n,
        // all vanishing evals are the same: g^n - 1
        let gN = frPow(g, UInt64(n))
        let zhVal = frSub(gN, Fr.one)
        let vanishingEvals = [Fr](repeating: zhVal, count: n)

        // Batch invert: all inverses are the same, but compute properly
        let zhInv = frInverse(zhVal)
        let vanishingInvs = [Fr](repeating: zhInv, count: n)

        return CosetDomainData(
            points: points, vanishingEvals: vanishingEvals,
            vanishingInvs: vanishingInvs,
            omega: omega, generator: g, n: n
        )
    }

    // MARK: - Evaluate Polynomials on Coset

    /// Evaluate a polynomial (in coefficient form) at all coset points.
    ///
    /// Uses the coset NTT trick: to evaluate f(X) at {g*omega^i}, we can
    /// scale coefficients f_k by g^k, then apply a standard NTT.
    /// For small domains, uses direct Horner evaluation.
    ///
    /// - Parameters:
    ///   - coeffs: Polynomial coefficients [f_0, f_1, ..., f_{n-1}].
    ///   - coset: Precomputed coset domain.
    /// - Returns: Evaluations [f(g), f(g*omega), ..., f(g*omega^{n-1})].
    public func evaluateOnCoset(coeffs: [Fr], coset: CosetDomainData) -> [Fr] {
        let n = coset.n
        // Pad or truncate coefficients to domain size
        var padded = [Fr](repeating: Fr.zero, count: n)
        let copyLen = min(coeffs.count, n)
        for i in 0..<copyLen { padded[i] = coeffs[i] }

        // Coset NTT: multiply c_k by g^k, then NTT
        padded.withUnsafeMutableBytes { rBuf in
            withUnsafeBytes(of: coset.generator) { gBuf in
                bn254_fr_batch_mul_powers(
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    gBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(n))
            }
        }

        // Use CPU C NTT for the forward transform
        padded.withUnsafeMutableBytes { buf in
            bn254_fr_ntt(buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                         Int32(coset.n.trailingZeroBitCount))
        }

        return padded
    }

    // MARK: - Coset iNTT (Inverse)

    /// Convert evaluations on a coset back to coefficient form.
    ///
    /// Applies iNTT then divides each coefficient by g^k to undo coset shift.
    ///
    /// - Parameters:
    ///   - evals: Evaluations on the coset domain.
    ///   - coset: Precomputed coset domain.
    /// - Returns: Polynomial coefficients.
    public func cosetINTT(evals: [Fr], coset: CosetDomainData) -> [Fr] {
        let n = coset.n
        var data = evals
        if data.count < n {
            data += [Fr](repeating: Fr.zero, count: n - data.count)
        }

        // iNTT
        data.withUnsafeMutableBytes { buf in
            bn254_fr_intt(buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                          Int32(n.trailingZeroBitCount))
        }

        // Undo coset shift: divide c_k by g^k
        let gInv = frInverse(coset.generator)
        data.withUnsafeMutableBytes { rBuf in
            withUnsafeBytes(of: gInv) { gBuf in
                bn254_fr_batch_mul_powers(
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    gBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(n))
            }
        }

        return data
    }

    // MARK: - Gate Constraint Evaluation

    /// Evaluate the arithmetic gate identity at each coset point:
    ///   gate(x) = qL(x)*a(x) + qR(x)*b(x) + qO(x)*c(x) + qM(x)*a(x)*b(x) + qC(x)
    ///
    /// Includes custom gate contributions when enabled:
    ///   + qRange(x) * a(x) * (1 - a(x))         [range check / boolean constraint]
    ///   + qLookup(x) * prod(a(x) - t_i)          [lookup argument]
    ///   + qPoseidon(x) * (c(x) - a(x)^5)         [Poseidon S-box]
    ///
    /// - Parameters:
    ///   - inputs: All polynomial evaluations on the coset.
    ///   - config: Quotient configuration (controls which custom gates are active).
    ///   - lookupTables: Lookup table values for lookup gate contributions.
    /// - Returns: Gate constraint evaluations at each coset point.
    public func evaluateGateConstraint(
        inputs: CosetConstraintInputs,
        config: QuotientConfig,
        lookupTables: [PlonkLookupTable] = []
    ) -> [Fr] {
        let n = config.domainSize
        var result = [Fr](repeating: Fr.zero, count: n)

        if n >= GPUPlonkQuotientEngine.gpuThreshold,
           let _ = constraintPipeline {
            // GPU path for large domains
            gpuEvaluateGateConstraint(inputs: inputs, config: config,
                                      lookupTables: lookupTables, result: &result)
        } else {
            cpuEvaluateGateConstraint(inputs: inputs, config: config,
                                      lookupTables: lookupTables, result: &result)
        }

        return result
    }

    // MARK: - Permutation Argument Evaluation

    /// Evaluate the permutation argument contribution at each coset point:
    ///   perm(x) = (a + beta*id1 + gamma)(b + beta*id2 + gamma)(c + beta*id3 + gamma) * Z(x)
    ///           - (a + beta*sigma1 + gamma)(b + beta*sigma2 + gamma)(c + beta*sigma3 + gamma) * Z(omega*x)
    ///
    /// - Parameters:
    ///   - inputs: All polynomial evaluations on the coset.
    ///   - beta: Permutation challenge.
    ///   - gamma: Permutation challenge.
    /// - Returns: Permutation constraint evaluations at each coset point.
    public func evaluatePermutationArgument(
        inputs: CosetConstraintInputs, beta: Fr, gamma: Fr
    ) -> [Fr] {
        let n = inputs.aEvals.count
        var result = [Fr](repeating: Fr.zero, count: n)

        for i in 0..<n {
            // Numerator: (a + beta*id1 + gamma)(b + beta*id2 + gamma)(c + beta*id3 + gamma) * z
            let numA = frAdd(frAdd(inputs.aEvals[i], frMul(beta, inputs.id1Evals[i])), gamma)
            let numB = frAdd(frAdd(inputs.bEvals[i], frMul(beta, inputs.id2Evals[i])), gamma)
            let numC = frAdd(frAdd(inputs.cEvals[i], frMul(beta, inputs.id3Evals[i])), gamma)
            let numProd = frMul(frMul(numA, numB), frMul(numC, inputs.zEvals[i]))

            // Denominator: (a + beta*sigma1 + gamma)(b + beta*sigma2 + gamma)(c + beta*sigma3 + gamma) * z_shifted
            let denA = frAdd(frAdd(inputs.aEvals[i], frMul(beta, inputs.sigma1Evals[i])), gamma)
            let denB = frAdd(frAdd(inputs.bEvals[i], frMul(beta, inputs.sigma2Evals[i])), gamma)
            let denC = frAdd(frAdd(inputs.cEvals[i], frMul(beta, inputs.sigma3Evals[i])), gamma)
            let denProd = frMul(frMul(denA, denB), frMul(denC, inputs.zShiftedEvals[i]))

            result[i] = frSub(numProd, denProd)
        }

        return result
    }

    // MARK: - Boundary Constraint Evaluation

    /// Evaluate the boundary constraint: (Z(x) - 1) * L_1(x)
    /// This enforces that the permutation accumulator starts at 1.
    ///
    /// - Parameter inputs: Constraint inputs containing Z and L_1 evaluations.
    /// - Returns: Boundary constraint evaluations at each coset point.
    public func evaluateBoundaryConstraint(inputs: CosetConstraintInputs) -> [Fr] {
        let n = inputs.zEvals.count
        // result[i] = (zEvals[i] - 1) * l1Evals[i]
        // Step 1: tmp[i] = zEvals[i] - 1 = (-1) + zEvals[i]
        let negOne = frNeg(Fr.one)
        var tmp = [Fr](repeating: Fr.zero, count: n)
        withUnsafeBytes(of: negOne) { sBuf in
            inputs.zEvals.withUnsafeBytes { zBuf in
                tmp.withUnsafeMutableBytes { tBuf in
                    bn254_fr_batch_add_scalar_neon(
                        tBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        sBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        zBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n))
                }
            }
        }
        // Step 2: result[i] = tmp[i] * l1Evals[i]
        var result = [Fr](repeating: Fr.zero, count: n)
        tmp.withUnsafeBytes { tBuf in
            inputs.l1Evals.withUnsafeBytes { lBuf in
                result.withUnsafeMutableBytes { rBuf in
                    bn254_fr_batch_mul(
                        tBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        lBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n))
                }
            }
        }
        return result
    }

    // MARK: - Full Quotient Computation

    /// Compute the full quotient polynomial t(X) and split into degree-n chunks.
    ///
    /// Combines gate, permutation, and boundary constraints with challenge powers:
    ///   numerator(X) = gate(X) + alpha * perm(X) + alpha^2 * boundary(X)
    ///   t(X) = numerator(X) / Z_H(X)
    ///
    /// The division is performed pointwise on the coset (where Z_H is nonzero),
    /// then converted back to coefficient form via coset iNTT.
    ///
    /// - Parameters:
    ///   - inputs: All polynomial evaluations on the coset.
    ///   - coset: Precomputed coset domain with vanishing polynomial inverses.
    ///   - config: Quotient configuration.
    ///   - alpha: Challenge for combining constraint types.
    ///   - beta: Permutation challenge.
    ///   - gamma: Permutation challenge.
    ///   - lookupTables: Lookup tables for custom gate evaluation.
    /// - Returns: QuotientResult with coefficient-form chunks and debug evaluations.
    public func computeQuotient(
        inputs: CosetConstraintInputs,
        coset: CosetDomainData,
        config: QuotientConfig,
        alpha: Fr, beta: Fr, gamma: Fr,
        lookupTables: [PlonkLookupTable] = []
    ) -> QuotientResult {
        let n = config.domainSize

        // Step 1: Evaluate each constraint type on the coset
        let gateEvals = evaluateGateConstraint(inputs: inputs, config: config,
                                                lookupTables: lookupTables)
        let permEvals = evaluatePermutationArgument(inputs: inputs, beta: beta, gamma: gamma)
        let boundaryEvals = evaluateBoundaryConstraint(inputs: inputs)

        // Step 2: Combine with challenge powers using C batch kernels
        let alpha2 = frSqr(alpha)
        // numerator[i] = gateEvals[i] + alpha * permEvals[i]
        var numeratorEvals = [Fr](repeating: Fr.zero, count: n)
        gateEvals.withUnsafeBytes { gBuf in
            withUnsafeBytes(of: alpha) { aBuf in
                permEvals.withUnsafeBytes { pBuf in
                    numeratorEvals.withUnsafeMutableBytes { rBuf in
                        bn254_fr_batch_linear_combine(
                            gBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(n))
                    }
                }
            }
        }
        // numerator[i] += alpha2 * boundaryEvals[i]
        withUnsafeBytes(of: alpha2) { a2Buf in
            boundaryEvals.withUnsafeBytes { bBuf in
                numeratorEvals.withUnsafeMutableBytes { rBuf in
                    bn254_fr_batch_fma_scalar(
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        a2Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n))
                }
            }
        }

        // Step 3: Divide by Z_H pointwise (multiply by precomputed inverses)
        var quotientEvals = [Fr](repeating: Fr.zero, count: n)
        numeratorEvals.withUnsafeBytes { nBuf in
            coset.vanishingInvs.withUnsafeBytes { vBuf in
                quotientEvals.withUnsafeMutableBytes { rBuf in
                    bn254_fr_batch_mul(
                        nBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        vBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n))
                }
            }
        }

        // Step 4: Convert back to coefficient form via coset iNTT
        let fullQuotient = cosetINTT(evals: quotientEvals, coset: coset)

        // Step 5: Split into degree-n chunks
        let numChunks = config.numChunks
        var chunks = [[Fr]]()
        for c in 0..<numChunks {
            let start = c * n
            if start < fullQuotient.count {
                let chunk = Array(fullQuotient.dropFirst(start).prefix(n))
                let padded = chunk + [Fr](repeating: Fr.zero, count: max(0, n - chunk.count))
                chunks.append(padded)
            } else {
                chunks.append([Fr](repeating: Fr.zero, count: n))
            }
        }

        let usedGPU = n >= GPUPlonkQuotientEngine.gpuThreshold && constraintPipeline != nil

        return QuotientResult(
            chunks: chunks, fullQuotient: fullQuotient,
            gateEvals: gateEvals, permEvals: permEvals,
            usedGPU: usedGPU
        )
    }

    // MARK: - Quotient Reconstruction

    /// Reconstruct the full quotient polynomial from its chunks at a given evaluation point.
    ///
    /// t(zeta) = t_lo(zeta) + zeta^n * t_mid(zeta) + zeta^{2n} * t_hi(zeta) + ...
    ///
    /// This is used during the opening phase (Round 4) of the Plonk protocol.
    ///
    /// - Parameters:
    ///   - chunks: Quotient polynomial chunks in coefficient form.
    ///   - at: Evaluation point (typically the challenge zeta).
    ///   - domainSize: The domain size n (for computing zeta^n, zeta^{2n}, etc.).
    /// - Returns: The reconstructed quotient evaluation t(zeta).
    public func evaluateQuotientFromChunks(chunks: [[Fr]], at zeta: Fr,
                                           domainSize: Int) -> Fr {
        if chunks.isEmpty { return Fr.zero }

        let zetaN = frPow(zeta, UInt64(domainSize))
        var result = Fr.zero
        var zetaPow = Fr.one  // zeta^{k*n}

        for chunk in chunks {
            let chunkEval = polyEval(chunk, at: zeta)
            result = frAdd(result, frMul(chunkEval, zetaPow))
            zetaPow = frMul(zetaPow, zetaN)
        }

        return result
    }

    // MARK: - Vanishing Polynomial Division (Coefficient Form)

    /// Divide a polynomial by the vanishing polynomial Z_H(X) = X^n - 1 in coefficient form.
    ///
    /// This is an alternative to the coset-based approach: compute the numerator
    /// directly in coefficient form, then perform polynomial long division.
    ///
    /// - Parameters:
    ///   - numeratorCoeffs: Numerator polynomial in coefficient form.
    ///   - n: Domain size (degree of Z_H).
    /// - Returns: Quotient polynomial coefficients.
    public func divideByVanishing(numeratorCoeffs: [Fr], n: Int) -> [Fr] {
        return polyDivideByVanishing(numeratorCoeffs, n: n)
    }

    // MARK: - Quotient Verification

    /// Verify that the quotient polynomial satisfies the constraint identity.
    ///
    /// Checks: t(X) * Z_H(X) == gate(X) + alpha*perm(X) + alpha^2*boundary(X)
    /// at a random evaluation point (not on the domain).
    ///
    /// - Parameters:
    ///   - quotientChunks: The quotient polynomial chunks.
    ///   - gateCoeffs: Gate constraint polynomial coefficients.
    ///   - permCoeffs: Permutation constraint polynomial coefficients.
    ///   - boundaryCoeffs: Boundary constraint polynomial coefficients.
    ///   - alpha: Challenge for combining constraints.
    ///   - domainSize: Domain size n.
    ///   - evalPoint: Point at which to verify (should not be on domain).
    /// - Returns: True if the identity holds at the evaluation point.
    public func verifyQuotientIdentity(
        quotientChunks: [[Fr]],
        gateCoeffs: [Fr], permCoeffs: [Fr], boundaryCoeffs: [Fr],
        alpha: Fr, domainSize: Int, evalPoint: Fr
    ) -> Bool {
        let n = domainSize

        // LHS: t(zeta) * Z_H(zeta)
        let tZeta = evaluateQuotientFromChunks(chunks: quotientChunks, at: evalPoint,
                                                domainSize: n)
        let zetaN = frPow(evalPoint, UInt64(n))
        let zhZeta = frSub(zetaN, Fr.one)
        let lhs = frMul(tZeta, zhZeta)

        // RHS: gate(zeta) + alpha*perm(zeta) + alpha^2*boundary(zeta)
        let gateVal = polyEval(gateCoeffs, at: evalPoint)
        let permVal = polyEval(permCoeffs, at: evalPoint)
        let boundVal = polyEval(boundaryCoeffs, at: evalPoint)
        let alpha2 = frSqr(alpha)
        let rhs = frAdd(frAdd(gateVal, frMul(alpha, permVal)), frMul(alpha2, boundVal))

        return frEqual(lhs, rhs)
    }

    // MARK: - Simplified Gate-Only Quotient

    /// Compute the quotient polynomial for gate constraints only (no permutation or boundary).
    ///
    /// This is useful for testing and for protocols that handle the permutation
    /// argument separately. The result satisfies:
    ///   t(X) * Z_H(X) = qL*a + qR*b + qO*c + qM*a*b + qC
    ///
    /// - Parameters:
    ///   - circuit: The Plonk circuit.
    ///   - witness: Witness assignment.
    ///   - config: Quotient configuration.
    /// - Returns: Quotient polynomial chunks and full coefficient form.
    public func computeGateOnlyQuotient(
        circuit: PlonkCircuit, witness: [Fr], config: QuotientConfig
    ) -> QuotientResult {
        let n = config.domainSize
        precondition(circuit.numGates <= n, "Circuit too large for domain")

        // Build wire evaluations (pad to domain size)
        var aEvals = [Fr](repeating: Fr.zero, count: n)
        var bEvals = [Fr](repeating: Fr.zero, count: n)
        var cEvals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<circuit.numGates {
            let wires = circuit.wireAssignments[i]
            aEvals[i] = witness[wires[0]]
            bEvals[i] = witness[wires[1]]
            cEvals[i] = witness[wires[2]]
        }

        // Build selector evaluations
        var qLEvals = [Fr](repeating: Fr.zero, count: n)
        var qREvals = [Fr](repeating: Fr.zero, count: n)
        var qOEvals = [Fr](repeating: Fr.zero, count: n)
        var qMEvals = [Fr](repeating: Fr.zero, count: n)
        var qCEvals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<circuit.numGates {
            let g = circuit.gates[i]
            qLEvals[i] = g.qL; qREvals[i] = g.qR; qOEvals[i] = g.qO
            qMEvals[i] = g.qM; qCEvals[i] = g.qC
        }

        // Convert all to coefficient form via iNTT
        let logN = config.logN
        let aCoeffs = cINTT_Fr(aEvals, logN: logN)
        let bCoeffs = cINTT_Fr(bEvals, logN: logN)
        let cCoeffs = cINTT_Fr(cEvals, logN: logN)
        let qLCoeffs = cINTT_Fr(qLEvals, logN: logN)
        let qRCoeffs = cINTT_Fr(qREvals, logN: logN)
        let qOCoeffs = cINTT_Fr(qOEvals, logN: logN)
        let qMCoeffs = cINTT_Fr(qMEvals, logN: logN)
        let qCCoeffs = cINTT_Fr(qCEvals, logN: logN)

        // Build the coset domain
        let coset = buildCosetDomain(config: config)

        // Evaluate all polynomials on the coset
        let aCoset = evaluateOnCoset(coeffs: aCoeffs, coset: coset)
        let bCoset = evaluateOnCoset(coeffs: bCoeffs, coset: coset)
        let cCoset = evaluateOnCoset(coeffs: cCoeffs, coset: coset)
        let qLCoset = evaluateOnCoset(coeffs: qLCoeffs, coset: coset)
        let qRCoset = evaluateOnCoset(coeffs: qRCoeffs, coset: coset)
        let qOCoset = evaluateOnCoset(coeffs: qOCoeffs, coset: coset)
        let qMCoset = evaluateOnCoset(coeffs: qMCoeffs, coset: coset)
        let qCCoset = evaluateOnCoset(coeffs: qCCoeffs, coset: coset)

        // Gate constraint on coset: qL*a + qR*b + qO*c + qM*a*b + qC
        // Decompose into batch ops: 5 batch_mul + 4 batch_add
        var qLa = [Fr](repeating: Fr.zero, count: n)
        var qRb = [Fr](repeating: Fr.zero, count: n)
        var qOc = [Fr](repeating: Fr.zero, count: n)
        var ab = [Fr](repeating: Fr.zero, count: n)
        var qMab = [Fr](repeating: Fr.zero, count: n)
        // Batch multiplies (auto-parallel for n >= 4096)
        qLCoset.withUnsafeBytes { a in aCoset.withUnsafeBytes { b in qLa.withUnsafeMutableBytes { r in
            bn254_fr_batch_mul(a.baseAddress!.assumingMemoryBound(to: UInt64.self),
                               b.baseAddress!.assumingMemoryBound(to: UInt64.self),
                               r.baseAddress!.assumingMemoryBound(to: UInt64.self), Int32(n))
        }}}
        qRCoset.withUnsafeBytes { a in bCoset.withUnsafeBytes { b in qRb.withUnsafeMutableBytes { r in
            bn254_fr_batch_mul(a.baseAddress!.assumingMemoryBound(to: UInt64.self),
                               b.baseAddress!.assumingMemoryBound(to: UInt64.self),
                               r.baseAddress!.assumingMemoryBound(to: UInt64.self), Int32(n))
        }}}
        qOCoset.withUnsafeBytes { a in cCoset.withUnsafeBytes { b in qOc.withUnsafeMutableBytes { r in
            bn254_fr_batch_mul(a.baseAddress!.assumingMemoryBound(to: UInt64.self),
                               b.baseAddress!.assumingMemoryBound(to: UInt64.self),
                               r.baseAddress!.assumingMemoryBound(to: UInt64.self), Int32(n))
        }}}
        aCoset.withUnsafeBytes { a in bCoset.withUnsafeBytes { b in ab.withUnsafeMutableBytes { r in
            bn254_fr_batch_mul(a.baseAddress!.assumingMemoryBound(to: UInt64.self),
                               b.baseAddress!.assumingMemoryBound(to: UInt64.self),
                               r.baseAddress!.assumingMemoryBound(to: UInt64.self), Int32(n))
        }}}
        qMCoset.withUnsafeBytes { a in ab.withUnsafeBytes { b in qMab.withUnsafeMutableBytes { r in
            bn254_fr_batch_mul(a.baseAddress!.assumingMemoryBound(to: UInt64.self),
                               b.baseAddress!.assumingMemoryBound(to: UInt64.self),
                               r.baseAddress!.assumingMemoryBound(to: UInt64.self), Int32(n))
        }}}
        // Sum: gateEvals = qLa + qRb + qOc + qMab + qC
        var gateEvals = [Fr](repeating: Fr.zero, count: n)
        qLa.withUnsafeBytes { a in qRb.withUnsafeBytes { b in gateEvals.withUnsafeMutableBytes { r in
            bn254_fr_batch_add(a.baseAddress!.assumingMemoryBound(to: UInt64.self),
                               b.baseAddress!.assumingMemoryBound(to: UInt64.self),
                               r.baseAddress!.assumingMemoryBound(to: UInt64.self), Int32(n))
        }}}
        qOc.withUnsafeBytes { a in gateEvals.withUnsafeMutableBytes { r in
            bn254_fr_batch_add(r.baseAddress!.assumingMemoryBound(to: UInt64.self),
                               a.baseAddress!.assumingMemoryBound(to: UInt64.self),
                               r.baseAddress!.assumingMemoryBound(to: UInt64.self), Int32(n))
        }}
        qMab.withUnsafeBytes { a in gateEvals.withUnsafeMutableBytes { r in
            bn254_fr_batch_add(r.baseAddress!.assumingMemoryBound(to: UInt64.self),
                               a.baseAddress!.assumingMemoryBound(to: UInt64.self),
                               r.baseAddress!.assumingMemoryBound(to: UInt64.self), Int32(n))
        }}
        qCCoset.withUnsafeBytes { a in gateEvals.withUnsafeMutableBytes { r in
            bn254_fr_batch_add(r.baseAddress!.assumingMemoryBound(to: UInt64.self),
                               a.baseAddress!.assumingMemoryBound(to: UInt64.self),
                               r.baseAddress!.assumingMemoryBound(to: UInt64.self), Int32(n))
        }}

        // Divide by Z_H pointwise
        var quotientEvals = [Fr](repeating: Fr.zero, count: n)
        gateEvals.withUnsafeBytes { gBuf in
            coset.vanishingInvs.withUnsafeBytes { vBuf in
                quotientEvals.withUnsafeMutableBytes { rBuf in
                    bn254_fr_batch_mul(
                        gBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        vBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n))
                }
            }
        }

        // Convert back to coefficient form
        let fullQuotient = cosetINTT(evals: quotientEvals, coset: coset)

        // Split into chunks
        let numChunks = config.numChunks
        var chunks = [[Fr]]()
        for c in 0..<numChunks {
            let start = c * n
            if start < fullQuotient.count {
                let chunk = Array(fullQuotient.dropFirst(start).prefix(n))
                chunks.append(chunk + [Fr](repeating: Fr.zero, count: max(0, n - chunk.count)))
            } else {
                chunks.append([Fr](repeating: Fr.zero, count: n))
            }
        }

        return QuotientResult(
            chunks: chunks, fullQuotient: fullQuotient,
            gateEvals: gateEvals, permEvals: [],
            usedGPU: false
        )
    }

    // MARK: - CPU Constraint Evaluation

    private func cpuEvaluateGateConstraint(
        inputs: CosetConstraintInputs, config: QuotientConfig,
        lookupTables: [PlonkLookupTable], result: inout [Fr]
    ) {
        let n = config.domainSize
        for i in 0..<n {
            let a = inputs.aEvals[i]
            let b = inputs.bEvals[i]
            let c = inputs.cEvals[i]

            // Arithmetic: qL*a + qR*b + qO*c + qM*a*b + qC
            let qLa = frMul(inputs.qLEvals[i], a)
            let qRb = frMul(inputs.qREvals[i], b)
            let qOc = frMul(inputs.qOEvals[i], c)
            let qMab = frMul(inputs.qMEvals[i], frMul(a, b))
            var val = frAdd(frAdd(frAdd(qLa, qRb), frAdd(qOc, qMab)), inputs.qCEvals[i])

            // Range: qRange * a * (1 - a)
            if config.enableRangeGates {
                let aMinusASq = frSub(a, frSqr(a))
                val = frAdd(val, frMul(inputs.qRangeEvals[i], aMinusASq))
            }

            // Lookup: qLookup * prod(a - t_i) for each table
            if config.enableLookupGates && !lookupTables.isEmpty {
                for table in lookupTables {
                    if table.values.isEmpty { continue }
                    var prod = Fr.one
                    for tVal in table.values {
                        prod = frMul(prod, frSub(a, tVal))
                    }
                    val = frAdd(val, frMul(inputs.qLookupEvals[i], prod))
                }
            }

            // Poseidon S-box: qPoseidon * (c - a^5)
            if config.enablePoseidonGates {
                let a2 = frSqr(a)
                let a4 = frSqr(a2)
                let a5 = frMul(a4, a)
                let sboxDiff = frSub(c, a5)
                val = frAdd(val, frMul(inputs.qPoseidonEvals[i], sboxDiff))
            }

            result[i] = val
        }
    }

    // MARK: - GPU Constraint Evaluation

    private func gpuEvaluateGateConstraint(
        inputs: CosetConstraintInputs, config: QuotientConfig,
        lookupTables: [PlonkLookupTable], result: inout [Fr]
    ) {
        guard let device = device,
              let queue = commandQueue,
              let pipeline = constraintPipeline else {
            // Fallback to CPU
            cpuEvaluateGateConstraint(inputs: inputs, config: config,
                                      lookupTables: lookupTables, result: &result)
            return
        }

        let n = config.domainSize
        let elemSize = MemoryLayout<Fr>.stride

        // Create Metal buffers for all inputs
        guard let aBuf = device.makeBuffer(bytes: inputs.aEvals, length: n * elemSize),
              let bBuf = device.makeBuffer(bytes: inputs.bEvals, length: n * elemSize),
              let cBuf = device.makeBuffer(bytes: inputs.cEvals, length: n * elemSize),
              let qLBuf = device.makeBuffer(bytes: inputs.qLEvals, length: n * elemSize),
              let qRBuf = device.makeBuffer(bytes: inputs.qREvals, length: n * elemSize),
              let qOBuf = device.makeBuffer(bytes: inputs.qOEvals, length: n * elemSize),
              let qMBuf = device.makeBuffer(bytes: inputs.qMEvals, length: n * elemSize),
              let qCBuf = device.makeBuffer(bytes: inputs.qCEvals, length: n * elemSize),
              let outBuf = device.makeBuffer(length: n * elemSize) else {
            cpuEvaluateGateConstraint(inputs: inputs, config: config,
                                      lookupTables: lookupTables, result: &result)
            return
        }

        guard let cmdBuf = queue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            cpuEvaluateGateConstraint(inputs: inputs, config: config,
                                      lookupTables: lookupTables, result: &result)
            return
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(aBuf, offset: 0, index: 0)
        encoder.setBuffer(bBuf, offset: 0, index: 1)
        encoder.setBuffer(cBuf, offset: 0, index: 2)
        encoder.setBuffer(qLBuf, offset: 0, index: 3)
        encoder.setBuffer(qRBuf, offset: 0, index: 4)
        encoder.setBuffer(qOBuf, offset: 0, index: 5)
        encoder.setBuffer(qMBuf, offset: 0, index: 6)
        encoder.setBuffer(qCBuf, offset: 0, index: 7)
        encoder.setBuffer(outBuf, offset: 0, index: 8)

        var count = UInt32(n)
        encoder.setBytes(&count, length: 4, index: 9)

        let gridSize = MTLSize(width: n, height: 1, depth: 1)
        let tgSize = MTLSize(width: min(threadgroupSize, pipeline.maxTotalThreadsPerThreadgroup),
                             height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Read results back
        let outPtr = outBuf.contents().bindMemory(to: Fr.self, capacity: n)
        for i in 0..<n {
            result[i] = outPtr[i]
        }

        // Custom gates still on CPU (Metal kernel only handles arithmetic)
        if config.enableRangeGates || config.enableLookupGates || config.enablePoseidonGates {
            for i in 0..<n {
                let a = inputs.aEvals[i]
                let c = inputs.cEvals[i]

                if config.enableRangeGates {
                    let aMinusASq = frSub(a, frSqr(a))
                    result[i] = frAdd(result[i], frMul(inputs.qRangeEvals[i], aMinusASq))
                }
                if config.enableLookupGates && !lookupTables.isEmpty {
                    for table in lookupTables {
                        if table.values.isEmpty { continue }
                        var prod = Fr.one
                        for tVal in table.values {
                            prod = frMul(prod, frSub(a, tVal))
                        }
                        result[i] = frAdd(result[i], frMul(inputs.qLookupEvals[i], prod))
                    }
                }
                if config.enablePoseidonGates {
                    let a2 = frSqr(a)
                    let a4 = frSqr(a2)
                    let a5 = frMul(a4, a)
                    result[i] = frAdd(result[i], frMul(inputs.qPoseidonEvals[i], frSub(c, a5)))
                }
            }
        }
    }

    // MARK: - Identity Permutation Construction

    /// Build identity permutation evaluations for 3-wire Plonk on a given domain.
    ///
    /// id_1(omega^i) = omega^i
    /// id_2(omega^i) = k1 * omega^i
    /// id_3(omega^i) = k2 * omega^i
    ///
    /// - Parameters:
    ///   - domain: The domain points {omega^0, ..., omega^{n-1}}.
    ///   - k1: First coset multiplier.
    ///   - k2: Second coset multiplier.
    /// - Returns: Triple of identity permutation evaluation arrays.
    public func buildIdentityPermutation(
        domain: [Fr], k1: Fr, k2: Fr
    ) -> (id1: [Fr], id2: [Fr], id3: [Fr]) {
        let n = domain.count
        var id1 = [Fr](repeating: Fr.zero, count: n)
        var id2 = [Fr](repeating: Fr.zero, count: n)
        var id3 = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            id1[i] = domain[i]
            id2[i] = frMul(k1, domain[i])
            id3[i] = frMul(k2, domain[i])
        }
        return (id1, id2, id3)
    }

    // MARK: - First Lagrange Basis

    /// Compute the first Lagrange basis polynomial L_1(X) in evaluation form on a coset.
    ///
    /// L_1(omega^i) = 1 if i=0, else 0 in the original domain.
    /// On the coset, we evaluate the interpolated polynomial.
    ///
    /// - Parameters:
    ///   - coset: The coset domain.
    ///   - config: Quotient config.
    /// - Returns: L_1 evaluations on the coset.
    public func buildL1OnCoset(coset: CosetDomainData, config: QuotientConfig) -> [Fr] {
        let n = config.domainSize
        // L_1 in evaluation form: 1 at index 0, 0 elsewhere
        var l1Evals = [Fr](repeating: Fr.zero, count: n)
        l1Evals[0] = Fr.one
        // Convert to coefficient form
        let l1Coeffs = cINTT_Fr(l1Evals, logN: config.logN)
        // Evaluate on coset
        return evaluateOnCoset(coeffs: l1Coeffs, coset: coset)
    }

    // MARK: - Metal Kernel Compilation

    private static func compileConstraintKernel(device: MTLDevice) -> MTLComputePipelineState? {
        // BN254 Fr arithmetic gate evaluation kernel
        // Computes qL*a + qR*b + qO*c + qM*a*b + qC per thread
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        // BN254 Fr modulus limbs (low to high)
        constant uint Fr_P[8] = {
            0xf0000001u, 0x43e1f593u, 0x79b97091u, 0x2833e848u,
            0x8181585du, 0xb85045b6u, 0xe131a029u, 0x30644e72u
        };

        // Montgomery multiplication for BN254 Fr (schoolbook, 8x32-bit limbs)
        // Simplified for GPU: uses 64-bit intermediates
        void fr_mul_mont(thread const uint* a, thread const uint* b, thread uint* r) {
            ulong t[9] = {0,0,0,0,0,0,0,0,0};
            constant uint INV0 = 0xefffffffu; // -P^{-1} mod 2^32

            for (int i = 0; i < 8; i++) {
                ulong carry = 0;
                for (int j = 0; j < 8; j++) {
                    ulong prod = ulong(a[i]) * ulong(b[j]) + t[j] + carry;
                    t[j] = prod & 0xFFFFFFFF;
                    carry = prod >> 32;
                }
                t[8] += carry;

                uint m = uint(t[0]) * INV0;
                carry = 0;
                for (int j = 0; j < 8; j++) {
                    ulong prod = ulong(m) * ulong(Fr_P[j]) + t[j] + carry;
                    if (j > 0) t[j-1] = prod & 0xFFFFFFFF;
                    else { /* shift */ }
                    carry = prod >> 32;
                }
                // Shift: t[0..6] = t[1..7], t[7] = t[8] + carry
                for (int j = 0; j < 7; j++) t[j] = t[j+1];
                t[7] = t[8] + carry;
                t[8] = 0;
            }

            // Final reduction
            bool gte = false;
            for (int j = 7; j >= 0; j--) {
                if (t[j] > Fr_P[j]) { gte = true; break; }
                if (t[j] < Fr_P[j]) { break; }
            }
            if (gte) {
                ulong borrow = 0;
                for (int j = 0; j < 8; j++) {
                    ulong diff = t[j] - ulong(Fr_P[j]) - borrow;
                    t[j] = diff & 0xFFFFFFFF;
                    borrow = (diff >> 63) & 1;
                }
            }
            for (int j = 0; j < 8; j++) r[j] = uint(t[j]);
        }

        void fr_add(thread const uint* a, thread const uint* b, thread uint* r) {
            ulong carry = 0;
            for (int j = 0; j < 8; j++) {
                ulong s = ulong(a[j]) + ulong(b[j]) + carry;
                r[j] = uint(s & 0xFFFFFFFF);
                carry = s >> 32;
            }
            // Reduce if >= P
            bool gte = (carry != 0);
            if (!gte) {
                for (int j = 7; j >= 0; j--) {
                    if (r[j] > Fr_P[j]) { gte = true; break; }
                    if (r[j] < Fr_P[j]) { break; }
                }
            }
            if (gte) {
                ulong borrow = 0;
                for (int j = 0; j < 8; j++) {
                    ulong diff = ulong(r[j]) - ulong(Fr_P[j]) - borrow;
                    r[j] = uint(diff & 0xFFFFFFFF);
                    borrow = (diff >> 63) & 1;
                }
            }
        }

        // Gate constraint kernel: qL*a + qR*b + qO*c + qM*a*b + qC
        kernel void plonk_gate_constraint(
            device const uint* a_vals   [[buffer(0)]],
            device const uint* b_vals   [[buffer(1)]],
            device const uint* c_vals   [[buffer(2)]],
            device const uint* qL_vals  [[buffer(3)]],
            device const uint* qR_vals  [[buffer(4)]],
            device const uint* qO_vals  [[buffer(5)]],
            device const uint* qM_vals  [[buffer(6)]],
            device const uint* qC_vals  [[buffer(7)]],
            device uint* out            [[buffer(8)]],
            device const uint& count    [[buffer(9)]],
            uint tid [[thread_position_in_grid]]
        ) {
            if (tid >= count) return;

            uint off = tid * 8;
            uint a[8], b[8], c[8], qL[8], qR[8], qO[8], qM[8], qC[8];
            for (int j = 0; j < 8; j++) {
                a[j]  = a_vals[off+j];
                b[j]  = b_vals[off+j];
                c[j]  = c_vals[off+j];
                qL[j] = qL_vals[off+j];
                qR[j] = qR_vals[off+j];
                qO[j] = qO_vals[off+j];
                qM[j] = qM_vals[off+j];
                qC[j] = qC_vals[off+j];
            }

            // qL*a
            uint t1[8]; fr_mul_mont(qL, a, t1);
            // qR*b
            uint t2[8]; fr_mul_mont(qR, b, t2);
            // qO*c
            uint t3[8]; fr_mul_mont(qO, c, t3);
            // a*b
            uint ab[8]; fr_mul_mont(a, b, ab);
            // qM*(a*b)
            uint t4[8]; fr_mul_mont(qM, ab, t4);

            // Sum: t1 + t2
            uint s1[8]; fr_add(t1, t2, s1);
            // s1 + t3
            uint s2[8]; fr_add(s1, t3, s2);
            // s2 + t4
            uint s3[8]; fr_add(s2, t4, s3);
            // s3 + qC
            uint s4[8]; fr_add(s3, qC, s4);

            for (int j = 0; j < 8; j++) {
                out[off+j] = s4[j];
            }
        }
        """

        do {
            let library = try device.makeLibrary(source: source, options: nil)
            guard let fn = library.makeFunction(name: "plonk_gate_constraint") else { return nil }
            return try device.makeComputePipelineState(function: fn)
        } catch {
            return nil
        }
    }
}

// MARK: - Standalone Helpers

/// Compute frNeg for the quotient engine (negate in BN254 Fr).
/// Returns -a mod p, i.e. p - a for nonzero a.
private func quotientFrNeg(_ a: Fr) -> Fr {
    return frSub(Fr.zero, a)
}
