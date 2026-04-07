// GPUR1CSToQAPEngine — GPU-accelerated R1CS to QAP conversion
//
// Converts R1CS constraint systems to Quadratic Arithmetic Programs (QAP)
// using NTT-based polynomial arithmetic. Supports both Groth16 and Marlin
// QAP formats.
//
// Pipeline:
//   1. Sparse matrix evaluation: A*z, B*z, C*z
//   2. IFFT to get coefficient polynomials a(x), b(x), c(x)
//   3. Coset evaluation: evaluate on shifted domain for division
//   4. Quotient computation: H(x) = (A(x)*B(x) - C(x)) / Z_H(x)
//   5. IFFT back to coefficient form
//
// Supports:
//   - Batch processing for large circuits (chunked sparse mat-vec)
//   - Sparse matrix handling for efficient R1CS representation
//   - Groth16 QAP format (H polynomial for proof element)
//   - Marlin QAP format (indexed polynomials with domain metadata)
//
// Public API:
//   convert(r1cs:witness:)                — full R1CS -> QAP conversion (Groth16)
//   convertMarlin(r1cs:witness:)          — full R1CS -> QAP conversion (Marlin)
//   computeQuotient(r1cs:witness:)        — compute H(x) quotient polynomial
//   evaluateOnCoset(coeffs:cosetGen:)     — evaluate polynomial on coset domain
//   sparseMatVecBatch(entries:witness:numRows:batchSize:) — batched sparse mat-vec

import Foundation
import NeonFieldOps
import Metal

// MARK: - QAP Result Types

/// Result of R1CS-to-QAP conversion for Groth16.
public struct Groth16QAPResult {
    /// Quotient polynomial H(x) coefficients: H = (A*B - C) / Z_H
    public let hPoly: [Fr]
    /// A polynomial evaluations at witness: A(x) * z
    public let aEvals: [Fr]
    /// B polynomial evaluations at witness: B(x) * z
    public let bEvals: [Fr]
    /// C polynomial evaluations at witness: C(x) * z
    public let cEvals: [Fr]
    /// Domain size (power of 2 >= numConstraints)
    public let domainSize: Int
    /// Log2 of domain size
    public let logDomainSize: Int
}

/// Result of R1CS-to-QAP conversion for Marlin.
public struct MarlinQAPResult {
    /// z_A polynomial coefficients (IFFT of A*z)
    public let zACoeffs: [Fr]
    /// z_B polynomial coefficients (IFFT of B*z)
    public let zBCoeffs: [Fr]
    /// z_C polynomial coefficients (IFFT of C*z)
    public let zCCoeffs: [Fr]
    /// Quotient polynomial t(x) = (z_A * z_B - z_C) / v_H
    public let tCoeffs: [Fr]
    /// Constraint domain size |H|
    public let constraintDomainSize: Int
    /// Variable domain size |K|
    public let variableDomainSize: Int
    /// Root of unity for constraint domain
    public let omegaH: Fr
}

/// Coset evaluation result.
public struct CosetEvalResult {
    /// Polynomial evaluations on coset {g * omega^i}
    public let evaluations: [Fr]
    /// Coset generator used
    public let cosetGenerator: Fr
    /// Domain size
    public let domainSize: Int
}

// MARK: - GPU R1CS to QAP Engine

public final class GPUR1CSToQAPEngine {

    /// Metal device (nil if GPU unavailable, uses CPU fallback).
    public let device: MTLDevice?
    public let commandQueue: MTLCommandQueue?

    /// NTT engine for GPU-accelerated transforms.
    private var nttEngine: NTTEngine?

    /// Threshold below which we use CPU NTT instead of GPU.
    public var gpuNTTThreshold: Int = 64

    /// Batch size for chunked sparse matrix-vector products.
    public var sparseBatchSize: Int = 65536

    /// Whether to use concurrent dispatch for large sparse mat-vec.
    public var concurrentMatVec: Bool = true

    /// Minimum constraint count to trigger concurrent dispatch.
    public var concurrentThreshold: Int = 1024

    // MARK: - Initialization

    /// Create engine. Falls back to CPU if GPU is unavailable.
    public init() {
        if let device = MTLCreateSystemDefaultDevice(),
           let queue = device.makeCommandQueue() {
            self.device = device
            self.commandQueue = queue
            do {
                self.nttEngine = try NTTEngine()
            } catch {
                self.nttEngine = nil
            }
        } else {
            self.device = nil
            self.commandQueue = nil
            self.nttEngine = nil
        }
    }

    /// Create engine with explicit NTT engine (for reuse / testing).
    public init(nttEngine: NTTEngine?) {
        self.nttEngine = nttEngine
        if let ntt = nttEngine {
            self.device = ntt.device
            self.commandQueue = ntt.commandQueue
        } else {
            self.device = nil
            self.commandQueue = nil
        }
    }

    // MARK: - Full Groth16 QAP Conversion

    /// Convert R1CS to Groth16 QAP: computes A*z, B*z, C*z evaluations and H(x) quotient.
    ///
    /// - Parameters:
    ///   - r1cs: The R1CS constraint system.
    ///   - witness: Full witness vector [1, pub_1..pub_n, priv_1..priv_m].
    /// - Returns: Groth16QAPResult with H polynomial and evaluation vectors.
    public func convert(r1cs: R1CSInstance, witness: [Fr]) throws -> Groth16QAPResult {
        let m = r1cs.numConstraints
        var domainN = 1; var logN = 0
        while domainN < m { domainN <<= 1; logN += 1 }

        // Step 1: Sparse matrix-vector products
        let (az, bz, cz) = computeMatVecTriple(r1cs: r1cs, witness: witness)

        // Step 2: Pad to domain size
        var aEvals = padToDomain(az, domainSize: domainN)
        var bEvals = padToDomain(bz, domainSize: domainN)
        var cEvals = padToDomain(cz, domainSize: domainN)

        // Step 3: Compute H(x) quotient polynomial
        let hPoly = try computeQuotientInternal(
            aEvals: &aEvals, bEvals: &bEvals, cEvals: &cEvals,
            domainN: domainN, logN: logN
        )

        return Groth16QAPResult(
            hPoly: hPoly, aEvals: az, bEvals: bz, cEvals: cz,
            domainSize: domainN, logDomainSize: logN
        )
    }

    // MARK: - Full Marlin QAP Conversion

    /// Convert R1CS to Marlin QAP format with indexed polynomial structure.
    ///
    /// - Parameters:
    ///   - r1cs: The R1CS constraint system.
    ///   - witness: Full witness vector [1, pub_1..pub_n, priv_1..priv_m].
    /// - Returns: MarlinQAPResult with z_A, z_B, z_C coefficient polynomials and quotient t(x).
    public func convertMarlin(r1cs: R1CSInstance, witness: [Fr]) throws -> MarlinQAPResult {
        let m = r1cs.numConstraints
        let n = r1cs.numVars
        var hSize = 1; var logH = 0
        while hSize < m { hSize <<= 1; logH += 1 }
        var kSize = 1
        while kSize < n { kSize <<= 1 }

        // Step 1: Sparse matrix-vector products
        let (az, bz, cz) = computeMatVecTriple(r1cs: r1cs, witness: witness)

        // Step 2: Pad evaluations to constraint domain size
        var aEvals = padToDomain(az, domainSize: hSize)
        var bEvals = padToDomain(bz, domainSize: hSize)
        var cEvals = padToDomain(cz, domainSize: hSize)

        // Step 3: IFFT to get coefficient polynomials z_A, z_B, z_C
        let zACoeffs = try performINTT(aEvals, logN: logH)
        let zBCoeffs = try performINTT(bEvals, logN: logH)
        let zCCoeffs = try performINTT(cEvals, logN: logH)

        // Step 4: Compute quotient t(x) = (z_A * z_B - z_C) / v_H on doubled domain
        let doubleH = hSize * 2
        let logDoubleH = logH + 1

        var zA2 = [Fr](repeating: .zero, count: doubleH)
        var zB2 = [Fr](repeating: .zero, count: doubleH)
        var zC2 = [Fr](repeating: .zero, count: doubleH)
        for i in 0..<zACoeffs.count { zA2[i] = zACoeffs[i] }
        for i in 0..<zBCoeffs.count { zB2[i] = zBCoeffs[i] }
        for i in 0..<zCCoeffs.count { zC2[i] = zCCoeffs[i] }

        // NTT to evaluation form on doubled domain
        let zAE2 = try performNTT(zA2, logN: logDoubleH)
        let zBE2 = try performNTT(zB2, logN: logDoubleH)
        let zCE2 = try performNTT(zC2, logN: logDoubleH)

        // Pointwise: numerator = z_A * z_B - z_C
        var numEvals = [Fr](repeating: .zero, count: doubleH)
        zAE2.withUnsafeBytes { aBuf in
            zBE2.withUnsafeBytes { bBuf in
                numEvals.withUnsafeMutableBytes { rBuf in
                    bn254_fr_batch_mul_parallel(
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(doubleH))
                }
            }
        }
        numEvals.withUnsafeMutableBytes { rBuf in
            zCE2.withUnsafeBytes { cBuf in
                bn254_fr_batch_sub_parallel(
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(doubleH))
            }
        }

        // INTT back to coefficient form
        var numCoeffs = try performINTT(numEvals, logN: logDoubleH)

        // Divide by vanishing polynomial v_H(X) = X^|H| - 1 via synthetic division
        var tCoeffs = [Fr](repeating: .zero, count: hSize)
        for i in stride(from: numCoeffs.count - 1, through: hSize, by: -1) {
            let qi = numCoeffs[i]
            tCoeffs[i - hSize] = qi
            numCoeffs[i - hSize] = frAdd(numCoeffs[i - hSize], qi)
        }

        let omegaH = frRootOfUnity(logN: logH)

        return MarlinQAPResult(
            zACoeffs: zACoeffs, zBCoeffs: zBCoeffs, zCCoeffs: zCCoeffs,
            tCoeffs: tCoeffs,
            constraintDomainSize: hSize, variableDomainSize: kSize,
            omegaH: omegaH
        )
    }

    // MARK: - Quotient Polynomial Computation

    /// Compute the quotient polynomial H(x) = (A*B - C) / Z_H from an R1CS and witness.
    ///
    /// This is the core operation for Groth16 proving. The result is a polynomial of
    /// degree < numConstraints whose evaluations encode the circuit satisfiability.
    public func computeQuotient(r1cs: R1CSInstance, witness: [Fr]) throws -> [Fr] {
        let m = r1cs.numConstraints
        var domainN = 1; var logN = 0
        while domainN < m { domainN <<= 1; logN += 1 }

        let (az, bz, cz) = computeMatVecTriple(r1cs: r1cs, witness: witness)

        var aEvals = padToDomain(az, domainSize: domainN)
        var bEvals = padToDomain(bz, domainSize: domainN)
        var cEvals = padToDomain(cz, domainSize: domainN)

        return try computeQuotientInternal(
            aEvals: &aEvals, bEvals: &bEvals, cEvals: &cEvals,
            domainN: domainN, logN: logN
        )
    }

    // MARK: - Coset Evaluation

    /// Evaluate polynomial (given as coefficients) on coset domain {g * omega^i}.
    ///
    /// This multiplies each coefficient c[i] by g^i, then performs NTT.
    /// Used in STARK and PLONK provers for constraint evaluation on shifted domains.
    ///
    /// - Parameters:
    ///   - coeffs: Polynomial coefficients (length must be power of 2).
    ///   - cosetGen: Coset generator g.
    /// - Returns: CosetEvalResult with evaluations on {g * omega^i}.
    public func evaluateOnCoset(coeffs: [Fr], cosetGen: Fr) throws -> CosetEvalResult {
        let n = coeffs.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Coefficient count must be a power of 2")
        let logN = Int(log2(Double(n)))

        // Multiply coefficients by coset shift powers: c[i] *= g^i
        var shifted = [Fr](repeating: .zero, count: n)
        var cg = cosetGen
        coeffs.withUnsafeBytes { cBuf in
            shifted.withUnsafeMutableBytes { sBuf in
                withUnsafeBytes(of: &cg) { bBuf in
                    bn254_fr_batch_mul_powers(
                        sBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n))
                }
            }
        }

        // NTT to get evaluations on coset
        let evals = try performNTT(shifted, logN: logN)

        return CosetEvalResult(evaluations: evals, cosetGenerator: cosetGen, domainSize: n)
    }

    /// Inverse coset evaluation: recover coefficients from coset evaluations.
    ///
    /// Performs INTT then divides each coefficient by g^i.
    ///
    /// - Parameters:
    ///   - evals: Evaluations on coset domain.
    ///   - cosetGen: Coset generator g that was used.
    /// - Returns: Recovered polynomial coefficients.
    public func inverseCosetEval(evals: [Fr], cosetGen: Fr) throws -> [Fr] {
        let n = evals.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Evaluation count must be a power of 2")
        let logN = Int(log2(Double(n)))

        // INTT
        var coeffs = try performINTT(evals, logN: logN)

        // Unshift: c[i] /= g^i
        var gInv = frInverse(cosetGen)
        coeffs.withUnsafeMutableBytes { cBuf in
            let ptr = cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
            withUnsafeBytes(of: &gInv) { bBuf in
                bn254_fr_batch_mul_powers(ptr, ptr,
                    bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(n))
            }
        }

        return coeffs
    }

    // MARK: - Sparse Matrix-Vector Products (Batched)

    /// Batched sparse matrix-vector product for large circuits.
    ///
    /// Processes entries in chunks of `batchSize` to limit memory usage and
    /// improve cache locality.
    ///
    /// - Parameters:
    ///   - entries: Sparse matrix entries (row, col, val).
    ///   - witness: Witness vector.
    ///   - numRows: Number of rows in the result.
    ///   - batchSize: Maximum entries per batch (0 = no batching).
    /// - Returns: Result vector of length numRows.
    public func sparseMatVecBatch(entries: [R1CSEntry], witness: [Fr],
                                   numRows: Int, batchSize: Int = 0) -> [Fr] {
        let effectiveBatch = batchSize > 0 ? batchSize : sparseBatchSize
        if entries.count <= effectiveBatch {
            return cpuSparseMatVec(entries, witness, numRows: numRows)
        }

        var result = [Fr](repeating: .zero, count: numRows)
        var offset = 0
        while offset < entries.count {
            let end = min(offset + effectiveBatch, entries.count)
            let chunk = Array(entries[offset..<end])
            let partial = cpuSparseMatVec(chunk, witness, numRows: numRows)
            for i in 0..<numRows {
                result[i] = frAdd(result[i], partial[i])
            }
            offset = end
        }
        return result
    }

    // MARK: - Vanishing Polynomial

    /// Evaluate the vanishing polynomial Z_H(x) = x^n - 1 at a point.
    ///
    /// - Parameters:
    ///   - point: Evaluation point.
    ///   - domainSize: Size of the domain H (must be a power of 2).
    /// - Returns: Z_H(point) = point^domainSize - 1.
    public func evaluateVanishing(point: Fr, domainSize: Int) -> Fr {
        let xn = frPow(point, UInt64(domainSize))
        return frSub(xn, Fr.one)
    }

    /// Compute vanishing polynomial evaluations on a set of points.
    ///
    /// - Parameters:
    ///   - points: Evaluation points.
    ///   - domainSize: Size of the domain H.
    /// - Returns: Array of Z_H(point[i]) values.
    public func evaluateVanishingBatch(points: [Fr], domainSize: Int) -> [Fr] {
        return points.map { evaluateVanishing(point: $0, domainSize: domainSize) }
    }

    // MARK: - Domain Utilities

    /// Compute the smallest power-of-2 domain size that fits `n` elements.
    public static func domainSizeFor(n: Int) -> (domainSize: Int, logN: Int) {
        var d = 1; var logN = 0
        while d < n { d <<= 1; logN += 1 }
        return (d, logN)
    }

    /// Generate the evaluation domain: [omega^0, omega^1, ..., omega^(n-1)].
    public func generateDomain(logN: Int) -> [Fr] {
        let n = 1 << logN
        let omega = frRootOfUnity(logN: logN)
        var domain = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n {
            domain[i] = frMul(domain[i - 1], omega)
        }
        return domain
    }

    /// Generate coset domain: [g*omega^0, g*omega^1, ..., g*omega^(n-1)].
    public func generateCosetDomain(logN: Int, cosetGen: Fr) -> [Fr] {
        let n = 1 << logN
        let omega = frRootOfUnity(logN: logN)
        var domain = [Fr](repeating: cosetGen, count: n)
        for i in 1..<n {
            domain[i] = frMul(domain[i - 1], omega)
        }
        return domain
    }

    // MARK: - R1CS Density Analysis

    /// Analyze sparse matrix density: useful for choosing batch sizes and GPU thresholds.
    public struct DensityInfo {
        /// Total number of non-zero entries across A, B, C.
        public let totalNonZero: Int
        /// Non-zero entries per matrix.
        public let aNonZero: Int
        public let bNonZero: Int
        public let cNonZero: Int
        /// Average entries per constraint row.
        public let avgEntriesPerRow: Double
        /// Maximum entries in a single row.
        public let maxEntriesPerRow: Int
    }

    /// Analyze the density of an R1CS instance.
    public func analyzeDensity(r1cs: R1CSInstance) -> DensityInfo {
        var rowCounts = [Int](repeating: 0, count: r1cs.numConstraints)
        for e in r1cs.aEntries { rowCounts[e.row] += 1 }
        for e in r1cs.bEntries { rowCounts[e.row] += 1 }
        for e in r1cs.cEntries { rowCounts[e.row] += 1 }

        let total = r1cs.aEntries.count + r1cs.bEntries.count + r1cs.cEntries.count
        let maxPerRow = rowCounts.max() ?? 0
        let avg = r1cs.numConstraints > 0
            ? Double(total) / Double(r1cs.numConstraints)
            : 0.0

        return DensityInfo(
            totalNonZero: total,
            aNonZero: r1cs.aEntries.count,
            bNonZero: r1cs.bEntries.count,
            cNonZero: r1cs.cEntries.count,
            avgEntriesPerRow: avg,
            maxEntriesPerRow: maxPerRow
        )
    }

    // MARK: - Internal: Sparse Matrix-Vector

    /// CPU sparse matrix-vector multiply: result[row] += val * witness[col].
    private func cpuSparseMatVec(_ entries: [R1CSEntry], _ witness: [Fr], numRows: Int) -> [Fr] {
        var result = [Fr](repeating: .zero, count: numRows)
        for e in entries {
            guard e.col < witness.count else { continue }
            result[e.row] = frAdd(result[e.row], frMul(e.val, witness[e.col]))
        }
        return result
    }

    /// Compute all three sparse mat-vec products (A*z, B*z, C*z) with optional concurrency.
    private func computeMatVecTriple(r1cs: R1CSInstance, witness: [Fr]) -> (az: [Fr], bz: [Fr], cz: [Fr]) {
        let m = r1cs.numConstraints

        if concurrentMatVec && m >= concurrentThreshold {
            let results = UnsafeMutablePointer<[Fr]>.allocate(capacity: 3)
            results.initialize(repeating: [], count: 3)
            DispatchQueue.concurrentPerform(iterations: 3) { idx in
                switch idx {
                case 0: results[0] = self.sparseMatVecBatch(entries: r1cs.aEntries, witness: witness, numRows: m)
                case 1: results[1] = self.sparseMatVecBatch(entries: r1cs.bEntries, witness: witness, numRows: m)
                default: results[2] = self.sparseMatVecBatch(entries: r1cs.cEntries, witness: witness, numRows: m)
                }
            }
            let az = results[0]; let bz = results[1]; let cz = results[2]
            results.deinitialize(count: 3); results.deallocate()
            return (az, bz, cz)
        } else {
            let az = sparseMatVecBatch(entries: r1cs.aEntries, witness: witness, numRows: m)
            let bz = sparseMatVecBatch(entries: r1cs.bEntries, witness: witness, numRows: m)
            let cz = sparseMatVecBatch(entries: r1cs.cEntries, witness: witness, numRows: m)
            return (az, bz, cz)
        }
    }

    // MARK: - Internal: NTT Wrappers

    /// Perform forward NTT (GPU if available and large enough, else CPU).
    private func performNTT(_ input: [Fr], logN: Int) throws -> [Fr] {
        let n = input.count
        if let ntt = nttEngine, n > gpuNTTThreshold {
            return try ntt.ntt(input)
        } else {
            return cNTT_Fr(input, logN: logN)
        }
    }

    /// Perform inverse NTT (GPU if available and large enough, else CPU).
    private func performINTT(_ input: [Fr], logN: Int) throws -> [Fr] {
        let n = input.count
        if let ntt = nttEngine, n > gpuNTTThreshold {
            return try ntt.intt(input)
        } else {
            return cINTT_Fr(input, logN: logN)
        }
    }

    // MARK: - Internal: Quotient Computation

    /// Core quotient polynomial computation: H = (A*B - C) / Z_H.
    ///
    /// Takes evaluation vectors (already padded to domainN), performs:
    ///   1. IFFT to coefficients
    ///   2. Zero-pad to 2*domainN, FFT to doubled evaluation domain
    ///   3. Pointwise A*B - C
    ///   4. IFFT back to coefficients
    ///   5. Divide by vanishing polynomial Z_H(x) = x^domainN - 1
    private func computeQuotientInternal(
        aEvals: inout [Fr], bEvals: inout [Fr], cEvals: inout [Fr],
        domainN: Int, logN: Int
    ) throws -> [Fr] {
        // Step 1: IFFT: evaluation -> coefficients
        let aCoeffs = try performINTT(aEvals, logN: logN)
        let bCoeffs = try performINTT(bEvals, logN: logN)
        let cCoeffs = try performINTT(cEvals, logN: logN)

        // Step 2: Zero-pad to 2*domainN
        let bigN = domainN * 2
        let logBigN = logN + 1

        var aPad = [Fr](repeating: .zero, count: bigN)
        var bPad = [Fr](repeating: .zero, count: bigN)
        var cPad = [Fr](repeating: .zero, count: bigN)
        for i in 0..<domainN {
            aPad[i] = aCoeffs[i]
            bPad[i] = bCoeffs[i]
            cPad[i] = cCoeffs[i]
        }

        // Step 3: FFT to evaluation form on doubled domain
        let aE = try performNTT(aPad, logN: logBigN)
        let bE = try performNTT(bPad, logN: logBigN)
        let cE = try performNTT(cPad, logN: logBigN)

        // Step 4: Pointwise: p[i] = a[i]*b[i] - c[i]
        var pE = [Fr](repeating: .zero, count: bigN)
        aE.withUnsafeBytes { aBuf in
            bE.withUnsafeBytes { bBuf in
                pE.withUnsafeMutableBytes { rBuf in
                    bn254_fr_batch_mul_parallel(
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(bigN))
                }
            }
        }
        pE.withUnsafeMutableBytes { rBuf in
            cE.withUnsafeBytes { cBuf in
                bn254_fr_batch_sub_parallel(
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(bigN))
            }
        }

        // Step 5: IFFT back to coefficient form
        let pCoeffs = try performINTT(pE, logN: logBigN)

        // Step 6: Divide by vanishing polynomial Z_H(x) = x^domainN - 1
        // For a polynomial p(x) = Z_H(x) * h(x), we have:
        //   p(x) = x^domainN * h(x) - h(x)
        // So: h[i] = p[i + domainN] and we propagate: p[i] += h[i]
        // This is equivalent to synthetic division by (x^n - 1).
        var hCoeffs = [Fr](repeating: .zero, count: domainN)
        var pMut = pCoeffs
        for i in stride(from: pMut.count - 1, through: domainN, by: -1) {
            let qi = pMut[i]
            hCoeffs[i - domainN] = qi
            pMut[i - domainN] = frAdd(pMut[i - domainN], qi)
        }

        return hCoeffs
    }

    // MARK: - Internal: Padding

    /// Pad array to a target domain size with zeros.
    private func padToDomain(_ arr: [Fr], domainSize: Int) -> [Fr] {
        if arr.count >= domainSize { return Array(arr.prefix(domainSize)) }
        var padded = [Fr](repeating: .zero, count: domainSize)
        for i in 0..<arr.count { padded[i] = arr[i] }
        return padded
    }
}
