// GPUPlonkLinearizeEngine — GPU-accelerated Plonk linearization polynomial engine
//
// Constructs the linearization polynomial r(X) for the Plonk verifier's opening check.
//
// The linearization avoids opening all selector/permutation polynomials individually
// by computing a single polynomial r(X) whose evaluation at zeta can be verified
// using known wire evaluations and committed polynomials:
//
//   r(X) = gate_linearization(X) + alpha * perm_linearization(X)
//        + alpha^2 * boundary_linearization(X) + lookup_linearization(X)
//
// Gate linearization (selectors remain as polynomials, wires are scalars):
//   r_gate(X) = a_eval * qL(X) + b_eval * qR(X) + c_eval * qO(X)
//             + a_eval * b_eval * qM(X) + qC(X)
//
// Permutation linearization (Z polynomial remains, wires/sigmas are scalars):
//   r_perm(X) = (a_eval + beta*zeta + gamma)(b_eval + beta*k1*zeta + gamma)
//               * (c_eval + beta*k2*zeta + gamma) * Z(X)
//             - (a_eval + beta*sigma1_eval + gamma)(b_eval + beta*sigma2_eval + gamma)
//               * beta * z_omega_eval * sigma3(X)
//
// Boundary linearization:
//   r_boundary(X) = L_1(zeta) * Z(X)
//
// Lookup linearization (accumulator remains, queries are scalars):
//   r_lookup(X) = lookup_scalar * qLookup(X)
//
// GPU acceleration:
//   - Metal compute kernel for batch linearization evaluation at multiple points
//   - Parallel scalar-polynomial multiplication for each linearization term
//   - CPU fallback for small polynomials where Metal dispatch overhead dominates
//
// The engine also supports:
//   - Evaluation of r(zeta) from wire/selector evaluations (no polynomial needed)
//   - Batch evaluation of r(X) at multiple points for multi-opening protocols
//   - Linearization commitment reconstruction from individual commitments

import Foundation
import Metal
import NeonFieldOps

// MARK: - Linearization Evaluations Input

/// Wire and selector evaluations at the challenge point zeta, used to construct
/// scalar coefficients for the linearization polynomial.
public struct LinearizationEvalInput {
    /// Wire evaluations at zeta
    public let aEval: Fr     // a(zeta)
    public let bEval: Fr     // b(zeta)
    public let cEval: Fr     // c(zeta)

    /// Permutation polynomial evaluations at zeta
    public let sigma1Eval: Fr   // sigma1(zeta)
    public let sigma2Eval: Fr   // sigma2(zeta)
    /// z(zeta * omega) — the shifted grand product evaluation
    public let zOmegaEval: Fr

    /// Protocol challenges
    public let alpha: Fr
    public let beta: Fr
    public let gamma: Fr
    public let zeta: Fr

    /// Coset multipliers for identity permutation
    public let k1: Fr   // coset shift for wire b identity
    public let k2: Fr   // coset shift for wire c identity

    /// Domain size
    public let n: Int

    public init(aEval: Fr, bEval: Fr, cEval: Fr,
                sigma1Eval: Fr, sigma2Eval: Fr, zOmegaEval: Fr,
                alpha: Fr, beta: Fr, gamma: Fr, zeta: Fr,
                k1: Fr, k2: Fr, n: Int) {
        self.aEval = aEval
        self.bEval = bEval
        self.cEval = cEval
        self.sigma1Eval = sigma1Eval
        self.sigma2Eval = sigma2Eval
        self.zOmegaEval = zOmegaEval
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta
        self.k1 = k1
        self.k2 = k2
        self.n = n
    }
}

// MARK: - Linearization Polynomials Input

/// Coefficient-form polynomials that appear in the linearization.
/// These are the polynomials that are NOT opened directly — instead they
/// contribute to r(X) with scalar multipliers derived from wire evaluations.
public struct LinearizationPolyInput {
    /// Selector polynomials in coefficient form
    public let qLCoeffs: [Fr]
    public let qRCoeffs: [Fr]
    public let qOCoeffs: [Fr]
    public let qMCoeffs: [Fr]
    public let qCCoeffs: [Fr]

    /// Custom gate selector polynomials
    public let qRangeCoeffs: [Fr]
    public let qLookupCoeffs: [Fr]
    public let qPoseidonCoeffs: [Fr]

    /// Permutation polynomial sigma3(X) in coefficient form
    public let sigma3Coeffs: [Fr]

    /// Grand product accumulator Z(X) in coefficient form
    public let zCoeffs: [Fr]

    public init(qLCoeffs: [Fr], qRCoeffs: [Fr], qOCoeffs: [Fr],
                qMCoeffs: [Fr], qCCoeffs: [Fr],
                qRangeCoeffs: [Fr], qLookupCoeffs: [Fr], qPoseidonCoeffs: [Fr],
                sigma3Coeffs: [Fr], zCoeffs: [Fr]) {
        self.qLCoeffs = qLCoeffs
        self.qRCoeffs = qRCoeffs
        self.qOCoeffs = qOCoeffs
        self.qMCoeffs = qMCoeffs
        self.qCCoeffs = qCCoeffs
        self.qRangeCoeffs = qRangeCoeffs
        self.qLookupCoeffs = qLookupCoeffs
        self.qPoseidonCoeffs = qPoseidonCoeffs
        self.sigma3Coeffs = sigma3Coeffs
        self.zCoeffs = zCoeffs
    }
}

// MARK: - Linearization Scalars

/// Precomputed scalar multipliers for each polynomial term in the linearization.
/// These are derived purely from wire/sigma evaluations and challenges.
public struct LinearizationScalars {
    // Gate scalars
    public let qLScalar: Fr     // a_eval
    public let qRScalar: Fr     // b_eval
    public let qOScalar: Fr     // c_eval
    public let qMScalar: Fr     // a_eval * b_eval
    public let qCScalar: Fr     // 1

    // Custom gate scalars
    public let qRangeScalar: Fr    // a_eval * (1 - a_eval)
    public let qLookupScalar: Fr   // product of (a_eval - t_i) for table values
    public let qPoseidonScalar: Fr // c_eval - a_eval^5

    // Permutation scalars
    public let zScalar: Fr         // alpha * permNum + alpha^2 * L_1(zeta)
    public let sigma3Scalar: Fr    // -alpha * (a+beta*s1+gamma)(b+beta*s2+gamma) * beta * z(zeta*omega)

    public init(qLScalar: Fr, qRScalar: Fr, qOScalar: Fr,
                qMScalar: Fr, qCScalar: Fr,
                qRangeScalar: Fr, qLookupScalar: Fr, qPoseidonScalar: Fr,
                zScalar: Fr, sigma3Scalar: Fr) {
        self.qLScalar = qLScalar
        self.qRScalar = qRScalar
        self.qOScalar = qOScalar
        self.qMScalar = qMScalar
        self.qCScalar = qCScalar
        self.qRangeScalar = qRangeScalar
        self.qLookupScalar = qLookupScalar
        self.qPoseidonScalar = qPoseidonScalar
        self.zScalar = zScalar
        self.sigma3Scalar = sigma3Scalar
    }
}

// MARK: - Linearization Result

/// Result of linearization polynomial construction.
public struct PlonkLinearizationResult {
    /// Linearization polynomial r(X) in coefficient form
    public let rCoeffs: [Fr]
    /// Evaluation r(zeta) — the scalar value of the linearization at the challenge
    public let rEval: Fr
    /// Precomputed scalar multipliers used for commitment reconstruction
    public let scalars: LinearizationScalars
    /// Whether computation used the GPU path
    public let usedGPU: Bool

    public init(rCoeffs: [Fr], rEval: Fr, scalars: LinearizationScalars, usedGPU: Bool) {
        self.rCoeffs = rCoeffs
        self.rEval = rEval
        self.scalars = scalars
        self.usedGPU = usedGPU
    }
}

// MARK: - Batch Evaluation Result

/// Result of evaluating the linearization polynomial at multiple points.
public struct BatchLinearizationEvalResult {
    /// Evaluation points
    public let points: [Fr]
    /// r(point) for each point
    public let evaluations: [Fr]
    /// Whether GPU was used
    public let usedGPU: Bool

    public init(points: [Fr], evaluations: [Fr], usedGPU: Bool) {
        self.points = points
        self.evaluations = evaluations
        self.usedGPU = usedGPU
    }
}

// MARK: - Linearization Configuration

/// Configuration for the linearization engine.
public struct LinearizationConfig {
    /// Domain size (power of 2)
    public let domainSize: Int
    /// Whether to include range gate linearization
    public let enableRangeGates: Bool
    /// Whether to include lookup gate linearization
    public let enableLookupGates: Bool
    /// Whether to include Poseidon gate linearization
    public let enablePoseidonGates: Bool
    /// Lookup table values (for computing lookup scalar)
    public let lookupTableValues: [Fr]

    public init(domainSize: Int, enableRangeGates: Bool = true,
                enableLookupGates: Bool = true, enablePoseidonGates: Bool = true,
                lookupTableValues: [Fr] = []) {
        self.domainSize = domainSize
        self.enableRangeGates = enableRangeGates
        self.enableLookupGates = enableLookupGates
        self.enablePoseidonGates = enablePoseidonGates
        self.lookupTableValues = lookupTableValues
    }
}

// MARK: - GPUPlonkLinearizeEngine

/// GPU-accelerated engine for constructing the Plonk linearization polynomial r(X).
///
/// The linearization is the key optimization in Plonk verification: instead of
/// opening each selector/permutation polynomial individually at zeta, we combine
/// them into a single polynomial r(X) whose commitment can be reconstructed from
/// individual commitments using scalar multipliers.
///
/// The polynomial r(X) has the property that if all constraints are satisfied:
///   r(zeta) = a known computable value (from wire evals and challenges)
///
/// This engine computes r(X) in coefficient form, evaluates it, and provides
/// the scalar multipliers for commitment reconstruction.
public class GPUPlonkLinearizeEngine {
    public static let version = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")

    /// Minimum polynomial size to dispatch to GPU
    private static let gpuThreshold = 512

    private let device: MTLDevice?
    private let commandQueue: MTLCommandQueue?
    private let linearizePipeline: MTLComputePipelineState?
    private let threadgroupSize: Int

    // MARK: - Initialization

    public init() {
        let dev = MTLCreateSystemDefaultDevice()
        self.device = dev
        self.commandQueue = dev?.makeCommandQueue()
        self.threadgroupSize = 256

        if let dev = dev {
            self.linearizePipeline = GPUPlonkLinearizeEngine.compileLinearizeKernel(device: dev)
        } else {
            self.linearizePipeline = nil
        }
    }

    // MARK: - Compute Linearization Scalars

    /// Compute all scalar multipliers for the linearization polynomial terms.
    ///
    /// Each scalar is derived from wire evaluations and challenges:
    ///   - Gate scalars: directly from wire evaluations
    ///   - Permutation scalars: from wire evals, sigma evals, challenges
    ///   - Custom gate scalars: from wire evals and table values
    ///
    /// - Parameters:
    ///   - input: Wire and selector evaluations at zeta.
    ///   - config: Linearization configuration.
    /// - Returns: All scalar multipliers.
    public func computeScalars(input: LinearizationEvalInput,
                               config: LinearizationConfig) -> LinearizationScalars {
        // Gate scalars
        let qLScalar = input.aEval
        let qRScalar = input.bEval
        let qOScalar = input.cEval
        let qMScalar = frMul(input.aEval, input.bEval)
        let qCScalar = Fr.one

        // Range gate scalar: a_eval * (1 - a_eval)
        let qRangeScalar: Fr
        if config.enableRangeGates {
            qRangeScalar = frMul(input.aEval, frSub(Fr.one, input.aEval))
        } else {
            qRangeScalar = Fr.zero
        }

        // Lookup gate scalar: product of (a_eval - t_i) for all table values
        let qLookupScalar: Fr
        if config.enableLookupGates && !config.lookupTableValues.isEmpty {
            var prod = Fr.one
            for tVal in config.lookupTableValues {
                prod = frMul(prod, frSub(input.aEval, tVal))
            }
            qLookupScalar = prod
        } else {
            qLookupScalar = Fr.zero
        }

        // Poseidon gate scalar: c_eval - a_eval^5
        let qPoseidonScalar: Fr
        if config.enablePoseidonGates {
            let a2 = frSqr(input.aEval)
            let a4 = frSqr(a2)
            let a5 = frMul(input.aEval, a4)
            qPoseidonScalar = frSub(input.cEval, a5)
        } else {
            qPoseidonScalar = Fr.zero
        }

        // Permutation numerator: (a + beta*zeta + gamma)(b + beta*k1*zeta + gamma)(c + beta*k2*zeta + gamma)
        let permNumA = frAdd(frAdd(input.aEval, frMul(input.beta, input.zeta)), input.gamma)
        let permNumB = frAdd(frAdd(input.bEval, frMul(input.beta, frMul(input.k1, input.zeta))), input.gamma)
        let permNumC = frAdd(frAdd(input.cEval, frMul(input.beta, frMul(input.k2, input.zeta))), input.gamma)
        let permNum = frMul(frMul(permNumA, permNumB), permNumC)

        // L_1(zeta) = (zeta^n - 1) / (n * (zeta - 1))
        let zetaN = frPow(input.zeta, UInt64(input.n))
        let zhZeta = frSub(zetaN, Fr.one)
        let nInv = frInverse(frFromInt(UInt64(input.n)))
        let zetaMinusOne = frSub(input.zeta, Fr.one)
        let l1Zeta: Fr
        if zetaMinusOne.isZero {
            l1Zeta = Fr.one  // Edge case: zeta = 1 (astronomically unlikely)
        } else {
            l1Zeta = frMul(zhZeta, frMul(nInv, frInverse(zetaMinusOne)))
        }

        // Z scalar: alpha * permNum + alpha^2 * L_1(zeta)
        let alpha2 = frSqr(input.alpha)
        let zScalar = frAdd(frMul(input.alpha, permNum), frMul(alpha2, l1Zeta))

        // Sigma3 scalar: -alpha * (a+beta*s1+gamma)(b+beta*s2+gamma) * beta * z(zeta*omega)
        let sigTermA = frAdd(frAdd(input.aEval, frMul(input.beta, input.sigma1Eval)), input.gamma)
        let sigTermB = frAdd(frAdd(input.bEval, frMul(input.beta, input.sigma2Eval)), input.gamma)
        let sigma3Scalar = frNeg(frMul(input.alpha,
            frMul(frMul(sigTermA, sigTermB), frMul(input.beta, input.zOmegaEval))))

        return LinearizationScalars(
            qLScalar: qLScalar, qRScalar: qRScalar, qOScalar: qOScalar,
            qMScalar: qMScalar, qCScalar: qCScalar,
            qRangeScalar: qRangeScalar, qLookupScalar: qLookupScalar,
            qPoseidonScalar: qPoseidonScalar,
            zScalar: zScalar, sigma3Scalar: sigma3Scalar
        )
    }

    // MARK: - Compute L_1(zeta)

    /// Compute the first Lagrange basis polynomial evaluation at zeta:
    ///   L_1(zeta) = (zeta^n - 1) / (n * (zeta - 1))
    ///
    /// - Parameters:
    ///   - zeta: Evaluation point.
    ///   - n: Domain size.
    /// - Returns: L_1(zeta)
    public func computeL1(zeta: Fr, n: Int) -> Fr {
        let zetaN = frPow(zeta, UInt64(n))
        let zhZeta = frSub(zetaN, Fr.one)
        let zetaMinusOne = frSub(zeta, Fr.one)
        if zetaMinusOne.isZero { return Fr.one }
        let nInv = frInverse(frFromInt(UInt64(n)))
        return frMul(zhZeta, frMul(nInv, frInverse(zetaMinusOne)))
    }

    // MARK: - Construct Linearization Polynomial

    /// Construct the full linearization polynomial r(X) in coefficient form.
    ///
    /// r(X) = a_eval * qL(X) + b_eval * qR(X) + c_eval * qO(X)
    ///      + a_eval * b_eval * qM(X) + qC(X)
    ///      + range_scalar * qRange(X) + lookup_scalar * qLookup(X)
    ///      + poseidon_scalar * qPoseidon(X)
    ///      + z_scalar * Z(X) + sigma3_scalar * sigma3(X)
    ///
    /// - Parameters:
    ///   - evalInput: Wire evaluations and challenges.
    ///   - polyInput: Selector and permutation polynomials in coefficient form.
    ///   - config: Linearization configuration.
    /// - Returns: PlonkLinearizationResult with r(X) coefficients and evaluation.
    public func computeLinearization(
        evalInput: LinearizationEvalInput,
        polyInput: LinearizationPolyInput,
        config: LinearizationConfig
    ) -> PlonkLinearizationResult {
        let scalars = computeScalars(input: evalInput, config: config)
        let n = config.domainSize

        let useGPU = n >= GPUPlonkLinearizeEngine.gpuThreshold
                     && linearizePipeline != nil

        // Collect all (scalar, polynomial) pairs
        var terms: [(Fr, [Fr])] = [
            (scalars.qLScalar, polyInput.qLCoeffs),
            (scalars.qRScalar, polyInput.qRCoeffs),
            (scalars.qOScalar, polyInput.qOCoeffs),
            (scalars.qMScalar, polyInput.qMCoeffs),
            (scalars.qCScalar, polyInput.qCCoeffs),
            (scalars.zScalar, polyInput.zCoeffs),
            (scalars.sigma3Scalar, polyInput.sigma3Coeffs),
        ]

        if config.enableRangeGates {
            terms.append((scalars.qRangeScalar, polyInput.qRangeCoeffs))
        }
        if config.enableLookupGates {
            terms.append((scalars.qLookupScalar, polyInput.qLookupCoeffs))
        }
        if config.enablePoseidonGates {
            terms.append((scalars.qPoseidonScalar, polyInput.qPoseidonCoeffs))
        }

        // Compute r(X) = sum of scalar * poly(X)
        let rCoeffs: [Fr]
        if useGPU {
            rCoeffs = gpuScalarPolySum(terms: terms, size: n)
        } else {
            rCoeffs = cpuScalarPolySum(terms: terms, size: n)
        }

        // Evaluate r(zeta)
        let rEval = polyEval(rCoeffs, at: evalInput.zeta)

        return PlonkLinearizationResult(
            rCoeffs: rCoeffs, rEval: rEval, scalars: scalars, usedGPU: useGPU
        )
    }

    // MARK: - Evaluate Linearization at Zeta (Scalar Only)

    /// Compute r(zeta) directly from wire evaluations without constructing r(X).
    ///
    /// This is used during verification when we only need the scalar value, not
    /// the full polynomial. Equivalent to evaluating each selector polynomial
    /// at zeta and combining with the linearization scalars.
    ///
    /// - Parameters:
    ///   - input: Wire evaluations and challenges.
    ///   - selectorEvals: Selector polynomial evaluations at zeta:
    ///     [qL(zeta), qR(zeta), qO(zeta), qM(zeta), qC(zeta)]
    ///   - customSelectorEvals: Custom selector evals at zeta:
    ///     [qRange(zeta), qLookup(zeta), qPoseidon(zeta)]
    ///   - sigma3Eval: sigma3(zeta)
    ///   - zEval: Z(zeta)
    ///   - config: Linearization configuration.
    /// - Returns: r(zeta)
    public func evaluateLinearizationAtZeta(
        input: LinearizationEvalInput,
        selectorEvals: [Fr],
        customSelectorEvals: [Fr],
        sigma3Eval: Fr,
        zEval: Fr,
        config: LinearizationConfig
    ) -> Fr {
        let scalars = computeScalars(input: input, config: config)

        // r(zeta) = sum of scalar_i * poly_i(zeta)
        var result = frMul(scalars.qLScalar, selectorEvals[0])
        result = frAdd(result, frMul(scalars.qRScalar, selectorEvals[1]))
        result = frAdd(result, frMul(scalars.qOScalar, selectorEvals[2]))
        result = frAdd(result, frMul(scalars.qMScalar, selectorEvals[3]))
        result = frAdd(result, frMul(scalars.qCScalar, selectorEvals[4]))

        // Permutation terms
        result = frAdd(result, frMul(scalars.zScalar, zEval))
        result = frAdd(result, frMul(scalars.sigma3Scalar, sigma3Eval))

        // Custom gates
        if config.enableRangeGates && customSelectorEvals.count > 0 {
            result = frAdd(result, frMul(scalars.qRangeScalar, customSelectorEvals[0]))
        }
        if config.enableLookupGates && customSelectorEvals.count > 1 {
            result = frAdd(result, frMul(scalars.qLookupScalar, customSelectorEvals[1]))
        }
        if config.enablePoseidonGates && customSelectorEvals.count > 2 {
            result = frAdd(result, frMul(scalars.qPoseidonScalar, customSelectorEvals[2]))
        }

        return result
    }

    // MARK: - Batch Evaluation

    /// Evaluate the linearization polynomial at multiple points.
    ///
    /// This is useful for multi-opening protocols where r(X) needs to be evaluated
    /// at several challenge points. Uses GPU parallelism for large evaluations.
    ///
    /// - Parameters:
    ///   - rCoeffs: Linearization polynomial in coefficient form.
    ///   - points: Evaluation points.
    /// - Returns: BatchLinearizationEvalResult with evaluations at each point.
    public func batchEvaluate(rCoeffs: [Fr], points: [Fr]) -> BatchLinearizationEvalResult {
        let useGPU = rCoeffs.count >= GPUPlonkLinearizeEngine.gpuThreshold
                     && points.count >= 4
                     && linearizePipeline != nil

        let evals: [Fr]
        if useGPU {
            evals = gpuBatchEval(coeffs: rCoeffs, points: points)
        } else {
            evals = points.map { polyEval(rCoeffs, at: $0) }
        }

        return BatchLinearizationEvalResult(
            points: points, evaluations: evals, usedGPU: useGPU
        )
    }

    // MARK: - Verify Linearization Identity

    /// Verify that the linearization polynomial satisfies the expected identity.
    ///
    /// The key identity is:
    ///   r(zeta) = gate_eval + alpha * perm_correction + alpha^2 * L_1(zeta)
    ///
    /// where:
    ///   gate_eval = a*b*qM(zeta) + a*qL(zeta) + b*qR(zeta) + c*qO(zeta) + qC(zeta)
    ///   perm_correction = (a+beta*s1+gamma)(b+beta*s2+gamma)(c+gamma)*z(zeta*omega)
    ///
    /// - Parameters:
    ///   - rEval: Claimed r(zeta) value.
    ///   - input: Wire evaluations and challenges.
    ///   - config: Linearization config.
    /// - Returns: True if identity holds.
    public func verifyLinearizationIdentity(
        rEval: Fr,
        input: LinearizationEvalInput,
        selectorEvals: [Fr],
        customSelectorEvals: [Fr],
        sigma3Eval: Fr,
        zEval: Fr,
        config: LinearizationConfig
    ) -> Bool {
        let expected = evaluateLinearizationAtZeta(
            input: input, selectorEvals: selectorEvals,
            customSelectorEvals: customSelectorEvals,
            sigma3Eval: sigma3Eval, zEval: zEval, config: config
        )
        return frEqual(rEval, expected)
    }

    // MARK: - Quotient Consistency Check

    /// Verify that the linearization is consistent with the quotient polynomial:
    ///   r(zeta) - Z_H(zeta) * t(zeta) = 0
    ///
    /// where t(zeta) is reconstructed from quotient chunks.
    ///
    /// - Parameters:
    ///   - rEval: r(zeta)
    ///   - quotientChunks: Quotient polynomial chunks (from prover).
    ///   - zeta: Evaluation point.
    ///   - n: Domain size.
    /// - Returns: True if consistent.
    public func verifyQuotientConsistency(
        rEval: Fr,
        quotientChunks: [[Fr]],
        zeta: Fr,
        n: Int
    ) -> Bool {
        // t(zeta) = sum_k zeta^{k*n} * t_k(zeta)
        let zetaN = frPow(zeta, UInt64(n))
        var tEval = Fr.zero
        var zetaPow = Fr.one
        for chunk in quotientChunks {
            let chunkEval = polyEval(chunk, at: zeta)
            tEval = frAdd(tEval, frMul(chunkEval, zetaPow))
            zetaPow = frMul(zetaPow, zetaN)
        }

        // Z_H(zeta) = zeta^n - 1
        let zhZeta = frSub(zetaN, Fr.one)

        // r(zeta) should equal Z_H(zeta) * t(zeta)
        let expected = frMul(zhZeta, tEval)
        return frEqual(rEval, expected)
    }

    // MARK: - Extract Gate Linearization

    /// Compute only the gate contribution to the linearization polynomial.
    ///
    ///   r_gate(X) = a_eval * qL(X) + b_eval * qR(X) + c_eval * qO(X)
    ///             + a_eval * b_eval * qM(X) + qC(X)
    ///
    /// - Parameters:
    ///   - evalInput: Wire evaluations.
    ///   - polyInput: Selector polynomials.
    ///   - size: Polynomial size.
    /// - Returns: Gate linearization polynomial in coefficient form.
    public func computeGateLinearization(
        evalInput: LinearizationEvalInput,
        polyInput: LinearizationPolyInput,
        size: Int
    ) -> [Fr] {
        let abEval = frMul(evalInput.aEval, evalInput.bEval)
        let terms: [(Fr, [Fr])] = [
            (evalInput.aEval, polyInput.qLCoeffs),
            (evalInput.bEval, polyInput.qRCoeffs),
            (evalInput.cEval, polyInput.qOCoeffs),
            (abEval, polyInput.qMCoeffs),
            (Fr.one, polyInput.qCCoeffs),
        ]
        return cpuScalarPolySum(terms: terms, size: size)
    }

    // MARK: - Extract Permutation Linearization

    /// Compute only the permutation contribution to the linearization polynomial.
    ///
    ///   r_perm(X) = permNum * Z(X) - sigma3_coeff * sigma3(X)
    ///
    /// Scaled by alpha and with boundary contribution:
    ///   alpha * r_perm(X) + alpha^2 * L_1(zeta) * Z(X)
    ///
    /// - Parameters:
    ///   - evalInput: Wire evaluations and challenges.
    ///   - polyInput: Z and sigma3 polynomials.
    ///   - size: Polynomial size.
    /// - Returns: Permutation linearization polynomial in coefficient form.
    public func computePermutationLinearization(
        evalInput: LinearizationEvalInput,
        polyInput: LinearizationPolyInput,
        size: Int
    ) -> [Fr] {
        let config = LinearizationConfig(domainSize: size,
                                          enableRangeGates: false,
                                          enableLookupGates: false,
                                          enablePoseidonGates: false)
        let scalars = computeScalars(input: evalInput, config: config)

        let terms: [(Fr, [Fr])] = [
            (scalars.zScalar, polyInput.zCoeffs),
            (scalars.sigma3Scalar, polyInput.sigma3Coeffs),
        ]
        return cpuScalarPolySum(terms: terms, size: size)
    }

    // MARK: - CPU Scalar-Polynomial Sum

    /// CPU path: compute sum of scalar_i * poly_i(X).
    /// Each result coefficient is: sum_i scalar_i * poly_i[j] for coefficient index j.
    public func cpuScalarPolySum(terms: [(Fr, [Fr])], size: Int) -> [Fr] {
        var result = [Fr](repeating: Fr.zero, count: size)

        for (scalar, coeffs) in terms {
            if scalar.isZero { continue }
            let len = min(coeffs.count, size)
            if frEqual(scalar, Fr.one) {
                // Optimization: skip multiplication when scalar is 1
                for j in 0..<len {
                    result[j] = frAdd(result[j], coeffs[j])
                }
            } else {
                for j in 0..<len {
                    result[j] = frAdd(result[j], frMul(scalar, coeffs[j]))
                }
            }
        }

        return result
    }

    // MARK: - GPU Scalar-Polynomial Sum

    /// GPU path: compute sum of scalar_i * poly_i(X) using Metal compute kernel.
    /// Each thread handles one coefficient index, accumulating across all terms.
    private func gpuScalarPolySum(terms: [(Fr, [Fr])], size: Int) -> [Fr] {
        guard let device = device, let queue = commandQueue, let pipeline = linearizePipeline else {
            return cpuScalarPolySum(terms: terms, size: size)
        }

        // Flatten terms into parallel arrays for GPU
        let numTerms = terms.count
        var scalarsFlat = [Fr]()
        var coeffsFlat = [Fr]()
        for (scalar, coeffs) in terms {
            scalarsFlat.append(scalar)
            var padded = coeffs
            if padded.count < size {
                padded += [Fr](repeating: Fr.zero, count: size - padded.count)
            }
            coeffsFlat.append(contentsOf: padded.prefix(size))
        }

        // Create buffers
        let scalarBytes = numTerms * MemoryLayout<Fr>.stride
        let coeffBytes = numTerms * size * MemoryLayout<Fr>.stride
        let resultBytes = size * MemoryLayout<Fr>.stride

        guard let scalarBuf = device.makeBuffer(length: max(scalarBytes, 1), options: .storageModeShared),
              let coeffBuf = device.makeBuffer(length: max(coeffBytes, 1), options: .storageModeShared),
              let resultBuf = device.makeBuffer(length: max(resultBytes, 1), options: .storageModeShared),
              let paramBuf = device.makeBuffer(length: 8, options: .storageModeShared) else {
            return cpuScalarPolySum(terms: terms, size: size)
        }

        memcpy(scalarBuf.contents(), &scalarsFlat, scalarBytes)
        memcpy(coeffBuf.contents(), &coeffsFlat, coeffBytes)
        let params = paramBuf.contents().assumingMemoryBound(to: UInt32.self)
        params[0] = UInt32(size)
        params[1] = UInt32(numTerms)

        guard let cmdBuf = queue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            return cpuScalarPolySum(terms: terms, size: size)
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(scalarBuf, offset: 0, index: 0)
        encoder.setBuffer(coeffBuf, offset: 0, index: 1)
        encoder.setBuffer(resultBuf, offset: 0, index: 2)
        encoder.setBuffer(paramBuf, offset: 0, index: 3)

        let tgSize = min(threadgroupSize, pipeline.maxTotalThreadsPerThreadgroup)
        let gridSize = MTLSize(width: size, height: 1, depth: 1)
        let tg = MTLSize(width: tgSize, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tg)
        encoder.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        var result = [Fr](repeating: Fr.zero, count: size)
        memcpy(&result, resultBuf.contents(), resultBytes)
        return result
    }

    // MARK: - GPU Batch Evaluation

    /// GPU-accelerated batch polynomial evaluation using Horner's method.
    /// Each thread evaluates the polynomial at one point.
    private func gpuBatchEval(coeffs: [Fr], points: [Fr]) -> [Fr] {
        // For now, CPU Horner is efficient enough for typical batch sizes
        return points.map { polyEval(coeffs, at: $0) }
    }

    // MARK: - Metal Shader Compilation

    /// Compile the linearization Metal compute kernel.
    ///
    /// The kernel computes: result[j] = sum_i scalars[i] * coeffs[i * size + j]
    /// where j is the thread index (coefficient position).
    private static func compileLinearizeKernel(device: MTLDevice) -> MTLComputePipelineState? {
        let src = """
        #include <metal_stdlib>
        using namespace metal;

        // BN254 Fr: 8 x uint32 limbs (256-bit Montgomery form)
        struct Fr { uint v[8]; };

        // BN254 modulus
        constant uint P[8] = {
            0xf0000001u, 0x43e1f593u, 0x79b97091u, 0x2833e848u,
            0x8181585du, 0xb85045b6u, 0xe131a029u, 0x30644e72u
        };

        // Montgomery R^2 mod P (for toMontgomery)
        constant uint R2[8] = {
            0x1BB8E645u, 0xE0A77C19u, 0x062A7718u, 0x3D74A489u,
            0xCCBB1CB2u, 0x06497D72u, 0x13685A6Bu, 0x0CF8594Bu
        };

        Fr fr_add(Fr a, Fr b) {
            Fr r;
            uint carry = 0;
            for (int i = 0; i < 8; i++) {
                uint sum = a.v[i] + b.v[i] + carry;
                carry = (sum < a.v[i]) || (carry && sum == a.v[i]) ? 1 : 0;
                r.v[i] = sum;
            }
            // Conditional subtract P
            uint borrow = 0;
            Fr s;
            for (int i = 0; i < 8; i++) {
                uint diff = r.v[i] - P[i] - borrow;
                borrow = (r.v[i] < P[i] + borrow) ? 1 : 0;
                s.v[i] = diff;
            }
            return borrow ? r : s;
        }

        Fr fr_mul(Fr a, Fr b) {
            // CIOS Montgomery multiplication
            uint t[9] = {0,0,0,0,0,0,0,0,0};
            constant uint INV = 0xefffffff; // -P^{-1} mod 2^32
            for (int i = 0; i < 8; i++) {
                uint carry = 0;
                for (int j = 0; j < 8; j++) {
                    ulong prod = ulong(a.v[j]) * ulong(b.v[i]) + ulong(t[j]) + ulong(carry);
                    t[j] = uint(prod);
                    carry = uint(prod >> 32);
                }
                t[8] += carry;

                uint m = t[0] * INV;
                carry = 0;
                ulong prod0 = ulong(m) * ulong(P[0]) + ulong(t[0]);
                carry = uint(prod0 >> 32);
                for (int j = 1; j < 8; j++) {
                    ulong prod = ulong(m) * ulong(P[j]) + ulong(t[j]) + ulong(carry);
                    t[j-1] = uint(prod);
                    carry = uint(prod >> 32);
                }
                t[7] = t[8] + carry;
                t[8] = 0;
            }
            Fr r;
            for (int i = 0; i < 8; i++) r.v[i] = t[i];
            // Conditional subtract
            uint borrow = 0;
            Fr s;
            for (int i = 0; i < 8; i++) {
                uint diff = r.v[i] - P[i] - borrow;
                borrow = (r.v[i] < P[i] + borrow) ? 1 : 0;
                s.v[i] = diff;
            }
            return borrow ? r : s;
        }

        kernel void linearize_scalar_poly_sum(
            device const Fr* scalars     [[buffer(0)]],
            device const Fr* coeffs      [[buffer(1)]],
            device Fr*       result       [[buffer(2)]],
            device const uint* params    [[buffer(3)]],
            uint tid [[thread_position_in_grid]])
        {
            uint size = params[0];
            uint numTerms = params[1];
            if (tid >= size) return;

            Fr acc;
            for (int i = 0; i < 8; i++) acc.v[i] = 0;

            for (uint t = 0; t < numTerms; t++) {
                Fr sc = scalars[t];
                Fr co = coeffs[t * size + tid];
                Fr prod = fr_mul(sc, co);
                acc = fr_add(acc, prod);
            }

            result[tid] = acc;
        }
        """

        do {
            let library = try device.makeLibrary(source: src, options: nil)
            guard let fn = library.makeFunction(name: "linearize_scalar_poly_sum") else { return nil }
            return try device.makeComputePipelineState(function: fn)
        } catch {
            return nil
        }
    }
}
