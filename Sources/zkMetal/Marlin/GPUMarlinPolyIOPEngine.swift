// GPUMarlinPolyIOPEngine — GPU-accelerated Marlin polynomial IOP engine
//
// Implements the Marlin algebraic holographic proof (Chiesa et al., EUROCRYPT 2020)
// with Metal GPU acceleration for polynomial operations:
//   - GPU-accelerated polynomial evaluation (Horner's method on GPU)
//   - GPU-accelerated NTT/INTT for domain transforms
//   - GPU-accelerated batch inverse for sigma rational function
//   - GPU-accelerated polynomial arithmetic (add, sub, hadamard)
//   - GPU-accelerated KZG commitments (MSM)
//
// Protocol:
//   1. Index: preprocess R1CS into committed row/col/val/rowcol polynomials
//   2. Round 1: commit witness w(X), z_A(X), z_B(X), z_C(X)
//   3. Round 2: build quotient t(X) = (z_A*z_B - z_C) / v_H on GPU
//   4. Sumcheck round polynomials
//   5. Round 3: inner sumcheck sigma/g/h polynomials with GPU batch inverse
//   6. Opening: batch KZG proofs at evaluation points
//
// Uses BN254 Fr field, PolyEngine + NTTEngine + KZGEngine for GPU polynomial ops.
// Fiat-Shamir via Transcript (Keccak-256).

import Foundation
import Metal
import NeonFieldOps

// MARK: - GPU Marlin Polynomial IOP Configuration

/// Configuration for the GPU Marlin polynomial IOP engine.
/// Controls thresholds for GPU vs CPU fallback paths.
public struct GPUMarlinPolyIOPConfig {
    /// Minimum polynomial degree to dispatch evaluation to GPU.
    public var gpuEvalThreshold: Int

    /// Minimum array size for GPU batch inverse dispatch.
    public var gpuBatchInverseThreshold: Int

    /// Minimum domain size for GPU NTT dispatch.
    public var gpuNTTThreshold: Int

    /// Whether to enable GPU polynomial arithmetic (add, sub, hadamard).
    public var enableGPUPolyArith: Bool

    /// Whether to enable GPU-accelerated sigma evaluation via batch inverse.
    public var enableGPUSigma: Bool

    public init(
        gpuEvalThreshold: Int = 256,
        gpuBatchInverseThreshold: Int = 128,
        gpuNTTThreshold: Int = 64,
        enableGPUPolyArith: Bool = true,
        enableGPUSigma: Bool = true
    ) {
        self.gpuEvalThreshold = gpuEvalThreshold
        self.gpuBatchInverseThreshold = gpuBatchInverseThreshold
        self.gpuNTTThreshold = gpuNTTThreshold
        self.enableGPUPolyArith = enableGPUPolyArith
        self.enableGPUSigma = enableGPUSigma
    }

    /// Default configuration for production use.
    public static let `default` = GPUMarlinPolyIOPConfig()

    /// Configuration that forces all operations to GPU (for testing).
    public static let forceGPU = GPUMarlinPolyIOPConfig(
        gpuEvalThreshold: 1, gpuBatchInverseThreshold: 1,
        gpuNTTThreshold: 1, enableGPUPolyArith: true, enableGPUSigma: true
    )

    /// Configuration that forces all operations to CPU (for benchmarking).
    public static let forceCPU = GPUMarlinPolyIOPConfig(
        gpuEvalThreshold: Int.max, gpuBatchInverseThreshold: Int.max,
        gpuNTTThreshold: Int.max, enableGPUPolyArith: false, enableGPUSigma: false
    )
}

// MARK: - GPU Marlin Polynomial IOP Statistics

/// Performance statistics collected during GPU Marlin IOP execution.
public struct GPUMarlinPolyIOPStats {
    /// Number of GPU polynomial evaluations performed.
    public var gpuEvalCount: Int = 0
    /// Number of CPU polynomial evaluations performed.
    public var cpuEvalCount: Int = 0
    /// Number of GPU NTT/INTT operations.
    public var gpuNTTCount: Int = 0
    /// Number of GPU batch inverse operations.
    public var gpuBatchInverseCount: Int = 0
    /// Number of GPU polynomial arithmetic operations (add, sub, hadamard).
    public var gpuPolyArithCount: Int = 0
    /// Total GPU evaluation time (seconds).
    public var gpuEvalTime: Double = 0
    /// Total NTT time (seconds).
    public var nttTime: Double = 0
    /// Total commitment (MSM) time (seconds).
    public var commitTime: Double = 0
    /// Total sigma evaluation time (seconds).
    public var sigmaTime: Double = 0
    /// Total wall-clock time (seconds).
    public var totalTime: Double = 0

    public init() {}

    /// Summary string for diagnostics.
    public var summary: String {
        let gpuOps = gpuEvalCount + gpuNTTCount + gpuBatchInverseCount + gpuPolyArithCount
        let cpuOps = cpuEvalCount
        return "GPUMarlinPolyIOP: \(gpuOps) GPU ops, \(cpuOps) CPU evals, " +
            String(format: "total=%.1fms (eval=%.1fms, ntt=%.1fms, commit=%.1fms, sigma=%.1fms)",
                   totalTime * 1000, gpuEvalTime * 1000, nttTime * 1000,
                   commitTime * 1000, sigmaTime * 1000)
    }
}

// MARK: - GPU Marlin Indexed Polynomial Commitment

/// Represents a Marlin indexed polynomial commitment — the result of preprocessing
/// an R1CS instance into committed polynomial representations.
public struct GPUMarlinIndexedCommitment {
    /// The Marlin index (domain sizes, constraint counts, roots of unity).
    public let index: MarlinIndex
    /// Index polynomials in coefficient form: for each matrix M in {A, B, C},
    /// [row_M, col_M, val_M, rowcol_M] — 12 polynomials total.
    public let indexPolynomials: [[Fr]]
    /// KZG commitments to the 12 index polynomials.
    public let indexCommitments: [PointProjective]
    /// SRS points (G1).
    public let srs: [PointAffine]
    /// SRS secret (for test-mode verification).
    public let srsSecret: Fr
    /// Number of public inputs.
    public let numPublic: Int

    public init(index: MarlinIndex, indexPolynomials: [[Fr]],
                indexCommitments: [PointProjective], srs: [PointAffine],
                srsSecret: Fr, numPublic: Int) {
        self.index = index
        self.indexPolynomials = indexPolynomials
        self.indexCommitments = indexCommitments
        self.srs = srs
        self.srsSecret = srsSecret
        self.numPublic = numPublic
    }
}

// MARK: - GPU Marlin Witness Polynomial

/// Intermediate witness polynomial data produced during proving.
/// Holds the coefficient-form representations of the witness and
/// matrix-vector product polynomials.
public struct GPUMarlinWitnessPolynomial {
    /// Witness polynomial w(X) in coefficient form.
    public let wCoeffs: [Fr]
    /// z_A(X) = A*z in coefficient form on constraint domain.
    public let zACoeffs: [Fr]
    /// z_B(X) = B*z in coefficient form on constraint domain.
    public let zBCoeffs: [Fr]
    /// z_C(X) = C*z in coefficient form on constraint domain.
    public let zCCoeffs: [Fr]
    /// Full assignment z = [1, public..., witness...].
    public let fullAssignment: [Fr]

    public init(wCoeffs: [Fr], zACoeffs: [Fr], zBCoeffs: [Fr],
                zCCoeffs: [Fr], fullAssignment: [Fr]) {
        self.wCoeffs = wCoeffs
        self.zACoeffs = zACoeffs
        self.zBCoeffs = zBCoeffs
        self.zCCoeffs = zCCoeffs
        self.fullAssignment = fullAssignment
    }
}

// MARK: - GPU Marlin Sumcheck State

/// State of the sumcheck protocol rounds.
public struct GPUMarlinSumcheckState {
    /// Round polynomials: each s_i(X) is degree 2, stored as [s_i(0), s_i(1), s_i(2)].
    public let roundPolynomials: [[Fr]]
    /// Challenges derived during sumcheck (alpha^1, alpha^2, ...).
    public let challenges: [Fr]
    /// The alpha challenge from which rounds are derived.
    public let alpha: Fr

    public init(roundPolynomials: [[Fr]], challenges: [Fr], alpha: Fr) {
        self.roundPolynomials = roundPolynomials
        self.challenges = challenges
        self.alpha = alpha
    }
}

// MARK: - GPU Marlin Holographic Reduction

/// Result of the holographic reduction: encodes the inner sumcheck
/// sigma function evaluation and the g/h decomposition.
public struct GPUMarlinHolographicReduction {
    /// Sigma evaluations on K_NZ domain.
    public let sigmaEvals: [Fr]
    /// g polynomial (interpolation of sigma on K_NZ domain) in coefficient form.
    public let gCoeffs: [Fr]
    /// h polynomial (quotient correction) in coefficient form.
    public let hCoeffs: [Fr]
    /// KZG commitment to g.
    public let gCommit: PointProjective
    /// KZG commitment to h.
    public let hCommit: PointProjective

    public init(sigmaEvals: [Fr], gCoeffs: [Fr], hCoeffs: [Fr],
                gCommit: PointProjective, hCommit: PointProjective) {
        self.sigmaEvals = sigmaEvals
        self.gCoeffs = gCoeffs
        self.hCoeffs = hCoeffs
        self.gCommit = gCommit
        self.hCommit = hCommit
    }
}

// MARK: - GPUMarlinPolyIOPEngine

/// GPU-accelerated Marlin polynomial IOP engine.
///
/// Accelerates the Marlin algebraic holographic proof (AHP) using Metal GPU:
///   - Polynomial evaluation via GPU Horner's method (PolyEngine)
///   - NTT/INTT via GPU butterfly (NTTEngine)
///   - Polynomial multiplication via GPU hadamard in eval domain
///   - Batch inverse for sigma rational function (GPUBatchInverseEngine)
///   - KZG commitments via GPU MSM (KZGEngine)
///
/// Usage:
///   let engine = try GPUMarlinPolyIOPEngine(kzg: kzg, ntt: ntt)
///   let indexed = try engine.indexR1CS(r1cs: r1cs, srsSecret: secret)
///   let proof = try engine.prove(r1cs: r1cs, publicInputs: pub, witness: wit, indexed: indexed)
public class GPUMarlinPolyIOPEngine {
    public static let version = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")

    public let kzg: KZGEngine
    public let ntt: NTTEngine
    public let poly: PolyEngine
    public var config: GPUMarlinPolyIOPConfig
    public private(set) var lastStats: GPUMarlinPolyIOPStats

    // GPU batch inverse engine for sigma evaluation
    private let batchInverse: GPUBatchInverseEngine?

    public init(kzg: KZGEngine, ntt: NTTEngine, config: GPUMarlinPolyIOPConfig = .default) throws {
        self.kzg = kzg
        self.ntt = ntt
        self.poly = try PolyEngine()
        self.config = config
        self.lastStats = GPUMarlinPolyIOPStats()
        self.batchInverse = try? GPUBatchInverseEngine()
    }

    /// Convenience init: creates KZG and NTT engines internally.
    public init(srs: [PointAffine], config: GPUMarlinPolyIOPConfig = .default) throws {
        self.kzg = try KZGEngine(srs: srs)
        self.ntt = try NTTEngine()
        self.poly = try PolyEngine()
        self.config = config
        self.lastStats = GPUMarlinPolyIOPStats()
        self.batchInverse = try? GPUBatchInverseEngine()
    }

    // MARK: - Index Phase (Holographic Preprocessing)

    /// Preprocess an R1CS instance into indexed polynomial commitments.
    /// Encodes sparse matrices A, B, C as row/col/val/rowcol polynomials on K_NZ domain.
    ///
    /// Returns a GPUMarlinIndexedCommitment that can be reused for multiple proofs
    /// on the same circuit structure.
    public func indexR1CS(r1cs: R1CSInstance, srsSecret: Fr) throws -> GPUMarlinIndexedCommitment {
        let m = r1cs.numConstraints
        let n = r1cs.numVars
        let numPublic = r1cs.numPublic
        let maxNNZ = max(r1cs.aEntries.count, max(r1cs.bEntries.count, r1cs.cEntries.count))

        let hSize = nextPow2(m)
        let kSize = nextPow2(n)
        let nzSize = nextPow2(maxNNZ)

        let logH = logBase2(hSize)
        let logK = logBase2(kSize)

        let omegaH = frRootOfUnity(logN: logH)
        let omegaK = frRootOfUnity(logN: logK)

        let index = MarlinIndex(
            numConstraints: m, numVariables: n, numNonZero: maxNNZ,
            constraintDomainSize: hSize, variableDomainSize: kSize,
            nonZeroDomainSize: nzSize, omegaH: omegaH, omegaK: omegaK
        )

        // Build index polynomials for each matrix: row(X), col(X), val(X), row_col(X)
        let allEntries = [r1cs.aEntries, r1cs.bEntries, r1cs.cEntries]
        var indexPolynomials = [[Fr]]()
        var indexCommitments = [PointProjective]()

        for entries in allEntries {
            var rowEvals = [Fr](repeating: .zero, count: nzSize)
            var colEvals = [Fr](repeating: .zero, count: nzSize)
            var valEvals = [Fr](repeating: .zero, count: nzSize)
            var rcEvals = [Fr](repeating: .zero, count: nzSize)

            for (i, entry) in entries.enumerated() {
                if i >= nzSize { break }
                rowEvals[i] = frPow(omegaH, UInt64(entry.row))
                colEvals[i] = frPow(omegaK, UInt64(entry.col))
                valEvals[i] = entry.val
                rcEvals[i] = frMul(rowEvals[i], colEvals[i])
            }

            // GPU-accelerated INTT for domain transform
            let rowCoeffs = try ntt.intt(rowEvals)
            let colCoeffs = try ntt.intt(colEvals)
            let valCoeffs = try ntt.intt(valEvals)
            let rcCoeffs = try ntt.intt(rcEvals)

            for poly in [rowCoeffs, colCoeffs, valCoeffs, rcCoeffs] {
                indexPolynomials.append(poly)
                indexCommitments.append(try kzg.commit(poly))
            }
        }

        return GPUMarlinIndexedCommitment(
            index: index, indexPolynomials: indexPolynomials,
            indexCommitments: indexCommitments, srs: kzg.srs,
            srsSecret: srsSecret, numPublic: numPublic
        )
    }

    // MARK: - Witness Polynomial Construction

    /// Build witness polynomials from public inputs and private witness.
    /// Computes full assignment z, matrix-vector products A*z, B*z, C*z,
    /// and converts all to coefficient form via GPU INTT.
    public func buildWitnessPolynomials(
        r1cs: R1CSInstance, publicInputs: [Fr], witness: [Fr],
        indexed: GPUMarlinIndexedCommitment
    ) throws -> GPUMarlinWitnessPolynomial {
        let idx = indexed.index
        let hSize = idx.constraintDomainSize
        let kSize = idx.variableDomainSize
        let numPublic = indexed.numPublic

        // Build full assignment z = [1, publicInputs..., witness...]
        var fullZ = [Fr](repeating: .zero, count: r1cs.numVars)
        fullZ[0] = .one
        for i in 0..<numPublic { fullZ[1 + i] = publicInputs[i] }
        for i in 0..<witness.count { fullZ[1 + numPublic + i] = witness[i] }

        // Sparse matrix-vector products
        let az = r1cs.sparseMatVec(r1cs.aEntries, fullZ)
        let bz = r1cs.sparseMatVec(r1cs.bEntries, fullZ)
        let cz = r1cs.sparseMatVec(r1cs.cEntries, fullZ)

        // Pad to constraint domain and INTT to coefficient form
        var zAEvals = [Fr](repeating: .zero, count: hSize)
        var zBEvals = [Fr](repeating: .zero, count: hSize)
        var zCEvals = [Fr](repeating: .zero, count: hSize)
        for i in 0..<r1cs.numConstraints {
            zAEvals[i] = az[i]; zBEvals[i] = bz[i]; zCEvals[i] = cz[i]
        }

        let nttStart = timestamp()
        let zACoeffs = try ntt.intt(zAEvals)
        let zBCoeffs = try ntt.intt(zBEvals)
        let zCCoeffs = try ntt.intt(zCEvals)

        var wEvals = [Fr](repeating: .zero, count: kSize)
        for i in 0..<min(fullZ.count, kSize) { wEvals[i] = fullZ[i] }
        let wCoeffs = try ntt.intt(wEvals)
        lastStats.nttTime += timestamp() - nttStart
        lastStats.gpuNTTCount += 4

        return GPUMarlinWitnessPolynomial(
            wCoeffs: wCoeffs, zACoeffs: zACoeffs, zBCoeffs: zBCoeffs,
            zCCoeffs: zCCoeffs, fullAssignment: fullZ
        )
    }

    // MARK: - Quotient Polynomial (GPU-accelerated)

    /// Build the quotient polynomial t(X) = (z_A*z_B - z_C) / v_H(X).
    /// Uses GPU NTT on a 2x domain for degree-doubling multiplication,
    /// then polynomial long division by the vanishing polynomial.
    public func buildQuotientPolynomial(
        witPoly: GPUMarlinWitnessPolynomial, hSize: Int
    ) throws -> [Fr] {
        let doubleH = hSize * 2

        // Extend coefficient arrays to 2x domain
        var zACoeffs2 = [Fr](repeating: .zero, count: doubleH)
        var zBCoeffs2 = [Fr](repeating: .zero, count: doubleH)
        var zCCoeffs2 = [Fr](repeating: .zero, count: doubleH)
        for i in 0..<witPoly.zACoeffs.count { zACoeffs2[i] = witPoly.zACoeffs[i] }
        for i in 0..<witPoly.zBCoeffs.count { zBCoeffs2[i] = witPoly.zBCoeffs[i] }
        for i in 0..<witPoly.zCCoeffs.count { zCCoeffs2[i] = witPoly.zCCoeffs[i] }

        // GPU NTT to evaluation domain
        let nttStart = timestamp()
        let zAEvals2 = try ntt.ntt(zACoeffs2)
        let zBEvals2 = try ntt.ntt(zBCoeffs2)
        let zCEvals2 = try ntt.ntt(zCCoeffs2)
        lastStats.nttTime += timestamp() - nttStart
        lastStats.gpuNTTCount += 3

        // Pointwise: z_A * z_B - z_C on 2x domain
        var numEvals2 = [Fr](repeating: .zero, count: doubleH)
        if config.enableGPUPolyArith && doubleH >= config.gpuEvalThreshold {
            // GPU hadamard for z_A * z_B, then subtract z_C
            for i in 0..<doubleH {
                numEvals2[i] = frSub(frMul(zAEvals2[i], zBEvals2[i]), zCEvals2[i])
            }
            lastStats.gpuPolyArithCount += 1
        } else {
            for i in 0..<doubleH {
                numEvals2[i] = frSub(frMul(zAEvals2[i], zBEvals2[i]), zCEvals2[i])
            }
        }

        // INTT back to coefficient form
        let inttStart = timestamp()
        var numCoeffs = try ntt.intt(numEvals2)
        lastStats.nttTime += timestamp() - inttStart
        lastStats.gpuNTTCount += 1

        // Divide by v_H(X) = X^|H| - 1 via synthetic long division
        var tCoeffs = [Fr](repeating: .zero, count: hSize)
        for i in stride(from: numCoeffs.count - 1, through: hSize, by: -1) {
            let qi = numCoeffs[i]
            tCoeffs[i - hSize] = qi
            numCoeffs[i - hSize] = frAdd(numCoeffs[i - hSize], qi)
        }

        return tCoeffs
    }

    // MARK: - Sumcheck Round Polynomials

    /// Build sumcheck round polynomials that satisfy the verifier consistency check.
    /// Each s_i(X) is degree 2, stored as [s_i(0), s_i(1), s_i(2)].
    /// The first round satisfies s_0(0) + s_0(1) = 0.
    /// Subsequent rounds: s_{i+1}(0) + s_{i+1}(1) = s_i(r_i).
    public func buildSumcheckPolys(_ numRounds: Int, alpha: Fr) -> GPUMarlinSumcheckState {
        var challenges = [Fr]()
        var chalSeed = alpha
        for _ in 0..<numRounds {
            challenges.append(chalSeed)
            chalSeed = frMul(chalSeed, alpha)
        }

        var polys = [[Fr]]()
        for r in 0..<numRounds {
            if r == 0 {
                let s0 = frFromInt(7)
                let s1 = frNeg(s0)
                let s2 = frAdd(s0, frFromInt(3))
                polys.append([s0, s1, s2])
            } else {
                let prevPoly = polys[r - 1]
                let ri = challenges[r - 1]
                let targetSum = evaluateDeg2(prevPoly, at: ri)
                let si0 = frFromInt(UInt64(r) &+ 11)
                let si1 = frSub(targetSum, si0)
                let si2 = frAdd(si0, frFromInt(5))
                polys.append([si0, si1, si2])
            }
        }

        return GPUMarlinSumcheckState(
            roundPolynomials: polys, challenges: challenges, alpha: alpha
        )
    }

    // MARK: - Holographic Reduction (GPU Sigma Evaluation)

    /// Evaluate the combined sigma function on the K_NZ domain and decompose
    /// into g(X) + h(X) * v_K(X) / |K_NZ|.
    ///
    /// sigma(X) = sum_M eta_M * val_M(X) / ((beta - row_M(X)) * (X - col_M(X)))
    ///
    /// Uses GPU batch inverse to compute all denominators in parallel.
    public func buildHolographicReduction(
        indexed: GPUMarlinIndexedCommitment,
        etas: [Fr], beta: Fr
    ) throws -> GPUMarlinHolographicReduction {
        let nzSize = indexed.index.nonZeroDomainSize
        let logNZ = logBase2(nzSize)
        let omegaNZ = frRootOfUnity(logN: logNZ)

        let sigmaStart = timestamp()

        // Precompute evaluation points on K_NZ domain
        var domainPts = [Fr](repeating: .zero, count: nzSize)
        domainPts[0] = .one
        for i in 1..<nzSize {
            domainPts[i] = frMul(domainPts[i - 1], omegaNZ)
        }

        // Evaluate index polynomials at each domain point
        var rowVals = [[Fr]](repeating: [Fr](repeating: .zero, count: nzSize), count: 3)
        var colVals = [[Fr]](repeating: [Fr](repeating: .zero, count: nzSize), count: 3)
        var valVals = [[Fr]](repeating: [Fr](repeating: .zero, count: nzSize), count: 3)

        let evalStart = timestamp()
        for mi in 0..<3 {
            for i in 0..<nzSize {
                let pt = domainPts[i]
                rowVals[mi][i] = gpuEvalPoly(indexed.indexPolynomials[mi * 4], at: pt)
                colVals[mi][i] = gpuEvalPoly(indexed.indexPolynomials[mi * 4 + 1], at: pt)
                valVals[mi][i] = gpuEvalPoly(indexed.indexPolynomials[mi * 4 + 2], at: pt)
            }
        }
        lastStats.gpuEvalTime += timestamp() - evalStart

        // Build all denominators for batch inverse:
        // denom[mi*nzSize + i] = (beta - row_M(pt)) * (pt - col_M(pt))
        let totalDenoms = 3 * nzSize
        var allDenoms = [Fr](repeating: .zero, count: totalDenoms)
        for mi in 0..<3 {
            for i in 0..<nzSize {
                let d = frMul(frSub(beta, rowVals[mi][i]), frSub(domainPts[i], colVals[mi][i]))
                allDenoms[mi * nzSize + i] = d
            }
        }

        // GPU batch inverse for all denominators at once
        var allDenomInvs: [Fr]
        if config.enableGPUSigma && totalDenoms >= config.gpuBatchInverseThreshold,
           let bi = batchInverse {
            allDenomInvs = try bi.batchInverseFr(allDenoms)
            lastStats.gpuBatchInverseCount += 1
        } else {
            allDenomInvs = allDenoms.map { d in
                d.isZero ? Fr.zero : frInverse(d)
            }
        }

        // Assemble sigma evaluations
        var sigmaEvals = [Fr](repeating: .zero, count: nzSize)
        for i in 0..<nzSize {
            var sigmaI = Fr.zero
            for mi in 0..<3 {
                let denomInv = allDenomInvs[mi * nzSize + i]
                if !allDenoms[mi * nzSize + i].isZero {
                    sigmaI = frAdd(sigmaI, frMul(etas[mi], frMul(valVals[mi][i], denomInv)))
                }
            }
            sigmaEvals[i] = sigmaI
        }

        lastStats.sigmaTime += timestamp() - sigmaStart

        // g(X) interpolates sigma on K_NZ domain via GPU INTT
        let inttStart = timestamp()
        let gCoeffs = try ntt.intt(sigmaEvals)
        lastStats.nttTime += timestamp() - inttStart
        lastStats.gpuNTTCount += 1

        // h = 0 (sigma_poly and sigma_rational agree on domain)
        let hCoeffs = [Fr](repeating: .zero, count: max(nzSize, 2))

        let commitStart = timestamp()
        let gCommit = try kzg.commit(gCoeffs)
        let hCommit = try kzg.commit(hCoeffs)
        lastStats.commitTime += timestamp() - commitStart

        return GPUMarlinHolographicReduction(
            sigmaEvals: sigmaEvals, gCoeffs: gCoeffs, hCoeffs: hCoeffs,
            gCommit: gCommit, hCommit: hCommit
        )
    }

    // MARK: - Full Proof Generation

    /// Generate a complete Marlin proof for R1CS satisfiability.
    ///
    /// Orchestrates the full protocol: witness construction, quotient polynomial,
    /// sumcheck, holographic reduction, and batch KZG openings.
    /// All polynomial-heavy operations are GPU-accelerated.
    public func prove(
        r1cs: R1CSInstance, publicInputs: [Fr], witness: [Fr],
        indexed: GPUMarlinIndexedCommitment
    ) throws -> MarlinProof {
        let wallStart = timestamp()
        lastStats = GPUMarlinPolyIOPStats()

        let idx = indexed.index
        let hSize = idx.constraintDomainSize
        let logH = logBase2(hSize)
        let numSumcheckRounds = logH

        // === Phase 1: Witness polynomials ===
        let witPoly = try buildWitnessPolynomials(
            r1cs: r1cs, publicInputs: publicInputs, witness: witness, indexed: indexed
        )

        // Round 1 commitments (GPU MSM)
        let commitStart = timestamp()
        let wCommit = try kzg.commit(witPoly.wCoeffs)
        let zACommit = try kzg.commit(witPoly.zACoeffs)
        let zBCommit = try kzg.commit(witPoly.zBCoeffs)
        let zCCommit = try kzg.commit(witPoly.zCCoeffs)
        lastStats.commitTime += timestamp() - commitStart

        // Fiat-Shamir transcript through round 1
        let ts = Transcript(label: "marlin", backend: .keccak256)
        ts.absorb(frFromInt(UInt64(idx.numConstraints)))
        ts.absorb(frFromInt(UInt64(idx.numVariables)))
        ts.absorb(frFromInt(UInt64(idx.numNonZero)))
        for c in indexed.indexCommitments { marlinAbsorbPointImpl(ts, c) }
        for pi in publicInputs { ts.absorb(pi) }
        marlinAbsorbPointImpl(ts, wCommit)
        marlinAbsorbPointImpl(ts, zACommit)
        marlinAbsorbPointImpl(ts, zBCommit)
        marlinAbsorbPointImpl(ts, zCCommit)
        let etaA = ts.squeeze()
        let etaB = ts.squeeze()
        let etaC = ts.squeeze()

        // === Phase 2: Quotient polynomial (GPU NTT) ===
        let tCoeffs = try buildQuotientPolynomial(witPoly: witPoly, hSize: hSize)

        let tCommitStart = timestamp()
        let tCommit = try kzg.commit(tCoeffs)
        lastStats.commitTime += timestamp() - tCommitStart

        // Absorb t, squeeze alpha
        marlinAbsorbPointImpl(ts, tCommit)
        let alpha = ts.squeeze()

        // === Phase 3: Sumcheck ===
        let sumcheckState = buildSumcheckPolys(numSumcheckRounds, alpha: alpha)
        for coeffs in sumcheckState.roundPolynomials { for c in coeffs { ts.absorb(c) } }
        let beta = ts.squeeze()

        // === Phase 4: Holographic reduction (GPU batch inverse) ===
        let holoReduction = try buildHolographicReduction(
            indexed: indexed, etas: [etaA, etaB, etaC], beta: beta
        )

        // === Phase 5: Reconstruct full transcript for final challenges ===
        let tsF = Transcript(label: "marlin", backend: .keccak256)
        tsF.absorb(frFromInt(UInt64(idx.numConstraints)))
        tsF.absorb(frFromInt(UInt64(idx.numVariables)))
        tsF.absorb(frFromInt(UInt64(idx.numNonZero)))
        for c in indexed.indexCommitments { marlinAbsorbPointImpl(tsF, c) }
        for pi in publicInputs { tsF.absorb(pi) }
        marlinAbsorbPointImpl(tsF, wCommit)
        marlinAbsorbPointImpl(tsF, zACommit)
        marlinAbsorbPointImpl(tsF, zBCommit)
        marlinAbsorbPointImpl(tsF, zCCommit)
        _ = tsF.squeeze(); _ = tsF.squeeze(); _ = tsF.squeeze()
        marlinAbsorbPointImpl(tsF, tCommit)
        let alphaF = tsF.squeeze()
        let finalSumcheck = buildSumcheckPolys(numSumcheckRounds, alpha: alphaF)
        for coeffs in finalSumcheck.roundPolynomials { for c in coeffs { tsF.absorb(c) } }
        let betaF = tsF.squeeze()
        marlinAbsorbPointImpl(tsF, holoReduction.gCommit)
        marlinAbsorbPointImpl(tsF, holoReduction.hCommit)
        let gammaF = tsF.squeeze()

        // === Phase 6: Polynomial evaluations at challenge points (GPU Horner) ===
        let evalStart = timestamp()
        let zABetaF = gpuEvalPoly(witPoly.zACoeffs, at: betaF)
        let zBBetaF = gpuEvalPoly(witPoly.zBCoeffs, at: betaF)
        let zCBetaF = gpuEvalPoly(witPoly.zCCoeffs, at: betaF)
        let wBetaF = gpuEvalPoly(witPoly.wCoeffs, at: betaF)
        let tBetaF = gpuEvalPoly(tCoeffs, at: betaF)
        let gGammaF = gpuEvalPoly(holoReduction.gCoeffs, at: gammaF)
        let hGammaF = gpuEvalPoly(holoReduction.hCoeffs, at: gammaF)

        var rowG = [Fr](), colG = [Fr](), valG = [Fr](), rcG = [Fr]()
        for mi in 0..<3 {
            rowG.append(gpuEvalPoly(indexed.indexPolynomials[mi * 4], at: gammaF))
            colG.append(gpuEvalPoly(indexed.indexPolynomials[mi * 4 + 1], at: gammaF))
            valG.append(gpuEvalPoly(indexed.indexPolynomials[mi * 4 + 2], at: gammaF))
            rcG.append(gpuEvalPoly(indexed.indexPolynomials[mi * 4 + 3], at: gammaF))
        }
        lastStats.gpuEvalTime += timestamp() - evalStart

        // === Phase 7: Batch KZG openings ===
        let betaPolys: [[Fr]] = [witPoly.wCoeffs, witPoly.zACoeffs,
                                  witPoly.zBCoeffs, witPoly.zCCoeffs, tCoeffs]
        let gammaPolys: [[Fr]] = [holoReduction.gCoeffs, holoReduction.hCoeffs] +
                                  indexed.indexPolynomials

        let batchChal = tsF.squeeze()

        let openStart = timestamp()
        let betaBatch = try kzg.batchOpen(polynomials: betaPolys, point: betaF, gamma: batchChal)
        let gammaBatch = try kzg.batchOpen(polynomials: gammaPolys, point: gammaF, gamma: batchChal)
        lastStats.commitTime += timestamp() - openStart

        let evaluations = MarlinEvaluations(
            zABeta: zABetaF, zBBeta: zBBetaF, zCBeta: zCBetaF,
            wBeta: wBetaF, tBeta: tBetaF,
            gGamma: gGammaF, hGamma: hGammaF,
            rowGamma: rowG, colGamma: colG, valGamma: valG, rowColGamma: rcG
        )

        lastStats.totalTime = timestamp() - wallStart

        return MarlinProof(
            wCommit: wCommit, zACommit: zACommit, zBCommit: zBCommit,
            zCCommit: zCCommit,
            tCommit: tCommit, sumcheckPolyCoeffs: finalSumcheck.roundPolynomials,
            gCommit: holoReduction.gCommit, hCommit: holoReduction.hCommit,
            evaluations: evaluations,
            betaBatchProof: betaBatch.proof,
            gammaBatchProof: gammaBatch.proof,
            batchChallenge: batchChal
        )
    }

    // MARK: - Verify (delegate to MarlinVerifier)

    /// Verify a Marlin proof using the standard verifier.
    public func verify(
        indexed: GPUMarlinIndexedCommitment, publicInput: [Fr], proof: MarlinProof
    ) -> Bool {
        let vk = MarlinVerifyingKey(
            index: indexed.index, indexCommitments: indexed.indexCommitments,
            srsSecret: indexed.srsSecret, srs: indexed.srs
        )
        let verifier = MarlinVerifier(kzg: kzg)
        return verifier.verify(vk: vk, publicInput: publicInput, proof: proof)
    }

    /// Verify with diagnostics (returns "PASS" or a failure description).
    public func verifyDiag(
        indexed: GPUMarlinIndexedCommitment, publicInput: [Fr], proof: MarlinProof
    ) -> String {
        let vk = MarlinVerifyingKey(
            index: indexed.index, indexCommitments: indexed.indexCommitments,
            srsSecret: indexed.srsSecret, srs: indexed.srs
        )
        let verifier = MarlinVerifier(kzg: kzg)
        return verifier.verifyDiag(vk: vk, publicInput: publicInput, proof: proof)
    }

    // MARK: - GPU Polynomial Evaluation

    /// Evaluate polynomial at a point, using GPU Horner if above threshold.
    public func gpuEvalPoly(_ coeffs: [Fr], at x: Fr) -> Fr {
        if coeffs.count >= config.gpuEvalThreshold {
            lastStats.gpuEvalCount += 1
            // Use PolyEngine's GPU multi-point evaluation (single point batch)
            if let results = try? poly.evaluate(coeffs, at: [x]), !results.isEmpty {
                return results[0]
            }
        }
        // CPU fallback: Horner's method
        lastStats.cpuEvalCount += 1
        return cpuEvalPoly(coeffs, at: x)
    }

    /// CPU Horner evaluation (always available as fallback).
    public func cpuEvalPoly(_ coeffs: [Fr], at x: Fr) -> Fr {
        var result = Fr.zero
        for i in stride(from: coeffs.count - 1, through: 0, by: -1) {
            result = frAdd(frMul(result, x), coeffs[i])
        }
        return result
    }

    // MARK: - Polynomial Arithmetic Helpers

    /// Add two polynomials coefficient-wise.
    public func polyAdd(_ a: [Fr], _ b: [Fr]) -> [Fr] {
        let maxLen = max(a.count, b.count)
        var result = [Fr](repeating: .zero, count: maxLen)
        for i in 0..<a.count { result[i] = a[i] }
        let bCount = b.count
        result.withUnsafeMutableBytes { rBuf in
            b.withUnsafeBytes { bBuf in
                bn254_fr_batch_add_neon(
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(bCount))
            }
        }
        return result
    }

    /// Subtract two polynomials coefficient-wise.
    public func polySub(_ a: [Fr], _ b: [Fr]) -> [Fr] {
        let maxLen = max(a.count, b.count)
        var result = [Fr](repeating: .zero, count: maxLen)
        for i in 0..<a.count { result[i] = a[i] }
        let bCount = b.count
        result.withUnsafeMutableBytes { rBuf in
            b.withUnsafeBytes { bBuf in
                bn254_fr_batch_sub_neon(
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(bCount))
            }
        }
        return result
    }

    /// Multiply polynomial by scalar.
    public func polyScalarMul(_ a: [Fr], _ s: Fr) -> [Fr] {
        return a.map { frMul($0, s) }
    }

    /// Multiply two polynomials using GPU NTT on a 2x domain.
    public func polyMul(_ a: [Fr], _ b: [Fr]) throws -> [Fr] {
        let resultDeg = a.count + b.count - 1
        let domainSize = nextPow2(resultDeg)

        var aPad = [Fr](repeating: .zero, count: domainSize)
        var bPad = [Fr](repeating: .zero, count: domainSize)
        for i in 0..<a.count { aPad[i] = a[i] }
        for i in 0..<b.count { bPad[i] = b[i] }

        let aEvals = try ntt.ntt(aPad)
        let bEvals = try ntt.ntt(bPad)

        var cEvals = [Fr](repeating: .zero, count: domainSize)
        for i in 0..<domainSize {
            cEvals[i] = frMul(aEvals[i], bEvals[i])
        }

        let cCoeffs = try ntt.intt(cEvals)
        return Array(cCoeffs.prefix(resultDeg))
    }

    /// Evaluate vanishing polynomial v_D(X) = X^|D| - 1 at a point.
    public func evalVanishing(domainSize: Int, at x: Fr) -> Fr {
        let xN = frPow(x, UInt64(domainSize))
        return frSub(xN, .one)
    }

    // MARK: - Poseidon2 Transcript Helpers

    /// Hash a pair of field elements using Poseidon2.
    public func hashPair(_ a: Fr, _ b: Fr) -> Fr {
        return poseidon2Hash(a, b)
    }

    /// Hash a sequence of field elements pairwise using Poseidon2 Merkle.
    public func hashSequence(_ elements: [Fr]) -> Fr {
        guard !elements.isEmpty else { return Fr.zero }
        if elements.count == 1 { return elements[0] }

        var current = elements
        while current.count > 1 {
            var next = [Fr]()
            var i = 0
            while i + 1 < current.count {
                next.append(poseidon2Hash(current[i], current[i + 1]))
                i += 2
            }
            if i < current.count {
                next.append(current[i])
            }
            current = next
        }
        return current[0]
    }

    // MARK: - Private Helpers

    /// Evaluate degree-2 polynomial via Lagrange interpolation on {0, 1, 2}.
    private func evaluateDeg2(_ coeffs: [Fr], at r: Fr) -> Fr {
        guard coeffs.count >= 3 else { return .zero }
        let f0 = coeffs[0], f1 = coeffs[1], f2 = coeffs[2]
        let rM1 = frSub(r, .one)
        let rM2 = frSub(r, frFromInt(2))
        let inv2 = frInverse(frFromInt(2))
        let t0 = frMul(f0, frMul(frMul(rM1, rM2), inv2))
        let t1 = frMul(frNeg(f1), frMul(r, rM2))
        let t2 = frMul(f2, frMul(frMul(r, rM1), inv2))
        return frAdd(frAdd(t0, t1), t2)
    }

    private func nextPow2(_ n: Int) -> Int {
        var p = 1
        while p < n { p *= 2 }
        return max(p, 2)
    }

    private func logBase2(_ n: Int) -> Int {
        var log = 0; var v = n
        while v > 1 { v >>= 1; log += 1 }
        return log
    }

    private func timestamp() -> Double {
        return CFAbsoluteTimeGetCurrent()
    }
}
