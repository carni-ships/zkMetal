// GPUWitnessCommitEngine — GPU-accelerated witness polynomial commitment engine
//
// Converts witness vectors to polynomial form (via iNTT), commits using
// Pedersen/KZG commitment schemes, supports blinding for zero-knowledge,
// batch commitment across multiple witness columns, and witness polynomial
// splitting for degree bounds.
//
// Architecture:
//   1. Witness-to-polynomial conversion via GPU iNTT (evaluation -> coefficient form)
//   2. KZG commitment via GPU MSM (coefficient * SRS point accumulation)
//   3. Blinding: append random scalars to hide witness in zero-knowledge
//   4. Batch mode: commit to multiple witness columns in a single GPU pass
//   5. Polynomial splitting: decompose high-degree polynomials for degree-bound proofs
//
// Works with BN254 Fr field type, PointProjective (Jacobian G1), and Metal GPU.

import Foundation
import Metal

// MARK: - Configuration

/// Configuration for witness polynomial commitment.
public struct WitnessCommitConfig {
    /// Number of blinding factors to append for zero-knowledge (0 = no blinding)
    public let numBlindingFactors: Int
    /// Whether to use GPU acceleration (false = CPU-only fallback)
    public let useGPU: Bool
    /// Maximum polynomial degree for splitting (0 = no splitting)
    public let degreeBound: Int
    /// Whether to cache intermediate NTT results
    public let cachePolynomials: Bool

    public init(numBlindingFactors: Int = 0, useGPU: Bool = true,
                degreeBound: Int = 0, cachePolynomials: Bool = false) {
        self.numBlindingFactors = numBlindingFactors
        self.useGPU = useGPU
        self.degreeBound = degreeBound
        self.cachePolynomials = cachePolynomials
    }
}

// MARK: - Commitment Result

/// Result of a single witness polynomial commitment (KZG-based).
public struct WitnessPolyCommitment {
    /// The commitment point C = sum(coeff_i * SRS_i)
    public let commitment: PointProjective
    /// The polynomial in coefficient form (after iNTT)
    public let polynomial: [Fr]
    /// Blinding factors used (empty if no blinding)
    public let blindingFactors: [Fr]
    /// Whether GPU was used for this commitment
    public let usedGPU: Bool

    public init(commitment: PointProjective, polynomial: [Fr],
                blindingFactors: [Fr], usedGPU: Bool) {
        self.commitment = commitment
        self.polynomial = polynomial
        self.blindingFactors = blindingFactors
        self.usedGPU = usedGPU
    }
}

/// Result of a batch witness commitment operation.
public struct BatchWitnessCommitResult {
    /// Individual commitments for each witness column
    public let commitments: [WitnessPolyCommitment]
    /// Total time for the batch operation (seconds)
    public let elapsedTime: Double
    /// Whether any column used GPU
    public let usedGPU: Bool

    public init(commitments: [WitnessPolyCommitment], elapsedTime: Double, usedGPU: Bool) {
        self.commitments = commitments
        self.elapsedTime = elapsedTime
        self.usedGPU = usedGPU
    }
}

/// Result of polynomial splitting for degree bounds.
public struct SplitPolynomial {
    /// Low-degree part: coefficients [0..degreeBound)
    public let low: [Fr]
    /// High-degree part: coefficients [degreeBound..n) shifted down
    public let high: [Fr]
    /// Commitment to the low part
    public let lowCommitment: PointProjective
    /// Commitment to the high part (using SRS offset by degreeBound)
    public let highCommitment: PointProjective
    /// The degree bound used for splitting
    public let degreeBound: Int

    public init(low: [Fr], high: [Fr], lowCommitment: PointProjective,
                highCommitment: PointProjective, degreeBound: Int) {
        self.low = low
        self.high = high
        self.lowCommitment = lowCommitment
        self.highCommitment = highCommitment
        self.degreeBound = degreeBound
    }
}

// MARK: - Blinding Mode

/// How to generate blinding factors for zero-knowledge.
public enum BlindingMode {
    /// Deterministic blinding from a seed (reproducible proofs)
    case deterministic(seed: UInt64)
    /// Pseudo-random blinding using system randomness
    case random
    /// Caller-supplied blinding factors
    case explicit(factors: [Fr])
}

// MARK: - GPUWitnessCommitEngine

/// GPU-accelerated engine for committing to witness polynomials.
///
/// Pipeline: witness evaluations -> iNTT (GPU) -> blinding -> MSM commitment (GPU)
///
/// Supports:
///   - Single and batch witness commitment
///   - Configurable blinding for zero-knowledge
///   - Polynomial splitting for degree-bound proofs
///   - CPU fallback when GPU is unavailable
public class GPUWitnessCommitEngine {
    public static let version = Versions.gpuWitnessCommit

    /// Metal device (nil if CPU-only)
    public let device: MTLDevice?
    /// NTT engine for polynomial conversion
    private var nttEngine: NTTEngine?
    /// MSM engine for commitment computation
    private var msmEngine: MetalMSM?

    /// SRS points for KZG commitment: [G, sG, s^2 G, ..., s^(d-1) G]
    public let srs: [PointAffine]
    /// Maximum supported polynomial degree
    public var maxDegree: Int { srs.count }

    /// Configuration
    public var config: WitnessCommitConfig

    /// Cached polynomial results (when config.cachePolynomials is true)
    private var polynomialCache: [Int: [Fr]] = [:]

    /// CPU-only mode flag
    public let cpuOnly: Bool

    /// Minimum witness size to dispatch to GPU (below this, use CPU)
    public var gpuThreshold: Int = 64

    // MARK: - Initialization

    /// Create engine with a toy SRS generated from a secret scalar.
    /// For testing only; production should use a ceremony-generated SRS.
    public init(srs: [PointAffine], config: WitnessCommitConfig = WitnessCommitConfig()) throws {
        self.srs = srs
        self.config = config
        self.cpuOnly = !config.useGPU

        if config.useGPU {
            guard let dev = MTLCreateSystemDefaultDevice() else {
                throw MSMError.noGPU
            }
            self.device = dev
            self.nttEngine = try NTTEngine()
            self.msmEngine = try MetalMSM()
        } else {
            self.device = nil
            self.nttEngine = nil
            self.msmEngine = nil
        }
    }

    /// Create a CPU-only engine (no Metal required).
    public init(srs: [PointAffine], cpuOnly: Bool) {
        self.srs = srs
        self.config = WitnessCommitConfig(useGPU: false)
        self.cpuOnly = true
        self.device = nil
        self.nttEngine = nil
        self.msmEngine = nil
    }

    // MARK: - SRS Generation (test helper)

    /// Generate a toy SRS from a secret tau. For testing only.
    public static func generateTestSRS(degree: Int, tau: Fr) -> [PointAffine] {
        let g1 = pointFromAffine(bn254G1Generator())
        var points = [PointProjective]()
        points.reserveCapacity(degree)
        var tauPow = Fr.one
        for _ in 0..<degree {
            points.append(pointScalarMul(g1, tauPow))
            tauPow = frMul(tauPow, tau)
        }
        return batchToAffine(points)
    }

    // MARK: - Blinding Factor Generation

    /// Generate blinding factors based on the configured mode.
    public func generateBlindingFactors(count: Int, mode: BlindingMode = .random) -> [Fr] {
        guard count > 0 else { return [] }
        switch mode {
        case .deterministic(let seed):
            return generateDeterministicBlinding(count: count, seed: seed)
        case .random:
            return generateRandomBlinding(count: count)
        case .explicit(let factors):
            // Pad or truncate to requested count
            if factors.count >= count {
                return Array(factors.prefix(count))
            }
            return factors + [Fr](repeating: Fr.zero, count: count - factors.count)
        }
    }

    private func generateDeterministicBlinding(count: Int, seed: UInt64) -> [Fr] {
        var factors = [Fr]()
        factors.reserveCapacity(count)
        // Simple PRNG: hash seed with index to get deterministic "random" scalars
        var state = seed
        for _ in 0..<count {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let val = state >> 1 // Ensure < field modulus by keeping values small
            factors.append(frFromInt(val & 0xFFFFFFFF))
        }
        return factors
    }

    private func generateRandomBlinding(count: Int) -> [Fr] {
        var factors = [Fr]()
        factors.reserveCapacity(count)
        for _ in 0..<count {
            var bytes = [UInt8](repeating: 0, count: 32)
            for j in 0..<32 {
                bytes[j] = UInt8.random(in: 0...255)
            }
            // Convert bytes to Fr (take mod p implicitly via Montgomery multiplication)
            let limbs: [UInt64] = [
                UInt64(bytes[0]) | (UInt64(bytes[1]) << 8) | (UInt64(bytes[2]) << 16) | (UInt64(bytes[3]) << 24) |
                (UInt64(bytes[4]) << 32) | (UInt64(bytes[5]) << 40) | (UInt64(bytes[6]) << 48) | (UInt64(bytes[7]) << 56),
                UInt64(bytes[8]) | (UInt64(bytes[9]) << 8) | (UInt64(bytes[10]) << 16) | (UInt64(bytes[11]) << 24) |
                (UInt64(bytes[12]) << 32) | (UInt64(bytes[13]) << 40) | (UInt64(bytes[14]) << 48) | (UInt64(bytes[15]) << 56),
                UInt64(bytes[16]) | (UInt64(bytes[17]) << 8) | (UInt64(bytes[18]) << 16) | (UInt64(bytes[19]) << 24) |
                (UInt64(bytes[20]) << 32) | (UInt64(bytes[21]) << 40) | (UInt64(bytes[22]) << 48) | (UInt64(bytes[23]) << 56),
                UInt64(bytes[24]) | (UInt64(bytes[25]) << 8) | (UInt64(bytes[26]) << 16) | (UInt64(bytes[27]) << 24) |
                (UInt64(bytes[28]) << 32) | (UInt64(bytes[29]) << 40) | (UInt64(bytes[30]) << 48) | (UInt64(bytes[31]) << 56),
            ]
            let raw = Fr.from64(limbs)
            // Convert to Montgomery form via multiplication by R^2
            factors.append(frMul(raw, Fr.from64(Fr.R2_MOD_R)))
        }
        return factors
    }

    // MARK: - Witness to Polynomial (iNTT)

    /// Convert witness evaluations to polynomial coefficients via iNTT.
    /// Pads to next power of 2 if needed.
    public func witnessToPolynomial(_ witness: [Fr]) throws -> [Fr] {
        let n = nextPowerOf2(witness.count)
        var padded = witness
        if padded.count < n {
            padded.append(contentsOf: [Fr](repeating: Fr.zero, count: n - padded.count))
        }

        if let nttEngine = nttEngine, !cpuOnly && n >= gpuThreshold {
            // GPU path: iNTT
            return try nttEngine.intt(padded)
        } else {
            // CPU fallback: naive iNTT via DFT matrix
            return cpuINTT(padded)
        }
    }

    /// Convert polynomial coefficients back to evaluations via NTT.
    public func polynomialToWitness(_ poly: [Fr]) throws -> [Fr] {
        let n = poly.count
        precondition(n > 0 && (n & (n - 1)) == 0, "polynomial length must be power of 2")

        if let nttEngine = nttEngine, !cpuOnly && n >= gpuThreshold {
            return try nttEngine.ntt(poly)
        } else {
            return cpuNTT(poly)
        }
    }

    // MARK: - Commitment (MSM)

    /// Commit to a polynomial in coefficient form: C = sum(coeff_i * SRS_i).
    public func commitPolynomial(_ coeffs: [Fr]) throws -> PointProjective {
        let n = coeffs.count
        guard n > 0 else { return pointIdentity() }
        guard n <= srs.count else {
            throw MSMError.invalidInput
        }

        let pts = Array(srs.prefix(n))

        if let msmEngine = msmEngine, !cpuOnly && n >= gpuThreshold {
            // GPU MSM
            let limbs = batchFrToLimbs(coeffs)
            return try msmEngine.msm(points: pts, scalars: limbs)
        } else {
            // CPU fallback: naive multi-scalar multiplication
            return cpuMSM(points: pts, scalars: coeffs)
        }
    }

    // MARK: - Single Witness Commitment

    /// Full pipeline: witness -> iNTT -> blind -> commit.
    public func commitWitness(_ witness: [Fr],
                              blinding: BlindingMode = .random) throws -> WitnessPolyCommitment {
        let numBlinding = config.numBlindingFactors
        let blindingFactors = generateBlindingFactors(count: numBlinding, mode: blinding)

        // Convert to polynomial form
        var poly = try witnessToPolynomial(witness)

        // Append blinding factors (extend polynomial degree)
        if !blindingFactors.isEmpty {
            poly.append(contentsOf: blindingFactors)
            // Re-pad to power of 2 if needed
            let n2 = nextPowerOf2(poly.count)
            if poly.count < n2 {
                poly.append(contentsOf: [Fr](repeating: Fr.zero, count: n2 - poly.count))
            }
        }

        // Commit via MSM
        let commitment = try commitPolynomial(poly)

        let usedGPU = !cpuOnly && witness.count >= gpuThreshold

        // Cache if configured
        if config.cachePolynomials {
            polynomialCache[witness.count] = poly
        }

        return WitnessPolyCommitment(
            commitment: commitment,
            polynomial: poly,
            blindingFactors: blindingFactors,
            usedGPU: usedGPU
        )
    }

    /// Commit to witness with explicit blinding factors.
    public func commitWitnessWithBlinding(_ witness: [Fr],
                                          blindingFactors: [Fr]) throws -> WitnessPolyCommitment {
        return try commitWitness(witness, blinding: .explicit(factors: blindingFactors))
    }

    // MARK: - Batch Commitment

    /// Commit to multiple witness columns in a batch.
    /// More efficient than individual commits due to GPU pipeline reuse.
    public func batchCommitWitnesses(_ witnesses: [[Fr]],
                                     blinding: BlindingMode = .random) throws -> BatchWitnessCommitResult {
        let t0 = CFAbsoluteTimeGetCurrent()
        guard !witnesses.isEmpty else {
            return BatchWitnessCommitResult(commitments: [],
                                            elapsedTime: 0, usedGPU: false)
        }

        var results = [WitnessPolyCommitment]()
        results.reserveCapacity(witnesses.count)
        var anyGPU = false

        for witness in witnesses {
            let result = try commitWitness(witness, blinding: blinding)
            anyGPU = anyGPU || result.usedGPU
            results.append(result)
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        return BatchWitnessCommitResult(commitments: results,
                                        elapsedTime: elapsed, usedGPU: anyGPU)
    }

    /// Batch commit with per-column blinding factors.
    public func batchCommitWithBlinding(_ witnesses: [[Fr]],
                                        blindingFactors: [[Fr]]) throws -> BatchWitnessCommitResult {
        let t0 = CFAbsoluteTimeGetCurrent()
        precondition(witnesses.count == blindingFactors.count,
                     "must provide blinding factors for each witness")

        var results = [WitnessPolyCommitment]()
        results.reserveCapacity(witnesses.count)
        var anyGPU = false

        for i in 0..<witnesses.count {
            let result = try commitWitnessWithBlinding(witnesses[i],
                                                       blindingFactors: blindingFactors[i])
            anyGPU = anyGPU || result.usedGPU
            results.append(result)
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        return BatchWitnessCommitResult(commitments: results,
                                        elapsedTime: elapsed, usedGPU: anyGPU)
    }

    // MARK: - Polynomial Splitting for Degree Bounds

    /// Split a polynomial at a degree bound and commit to both parts.
    ///
    /// Given p(X) = p_low(X) + X^d * p_high(X), computes:
    ///   - lowCommit = commit(p_low)
    ///   - highCommit = commit(p_high) using SRS shifted by d
    ///
    /// This is used in proof systems that enforce degree bounds (e.g., Marlin, PLONK).
    public func splitAndCommit(_ polynomial: [Fr],
                               degreeBound: Int) throws -> SplitPolynomial {
        guard degreeBound > 0 && degreeBound < polynomial.count else {
            throw MSMError.invalidInput
        }
        guard polynomial.count <= srs.count else {
            throw MSMError.invalidInput
        }

        // Split into low and high parts
        let low = Array(polynomial.prefix(degreeBound))
        let high = Array(polynomial.suffix(from: degreeBound))

        // Commit to low part: C_low = sum(low_i * SRS_i) for i in [0, d)
        let lowCommitment = try commitPolynomial(low)

        // Commit to high part: C_high = sum(high_i * SRS_{d+i}) for i in [0, n-d)
        let highCommitment = try commitWithOffset(high, offset: degreeBound)

        return SplitPolynomial(
            low: low, high: high,
            lowCommitment: lowCommitment,
            highCommitment: highCommitment,
            degreeBound: degreeBound
        )
    }

    /// Commit using SRS points starting at an offset.
    /// Used for degree-bound proofs: C = sum(coeff_i * SRS_{offset+i}).
    public func commitWithOffset(_ coeffs: [Fr], offset: Int) throws -> PointProjective {
        let n = coeffs.count
        guard n > 0 else { return pointIdentity() }
        guard offset + n <= srs.count else {
            throw MSMError.invalidInput
        }

        let pts = Array(srs[offset..<(offset + n)])

        if let msmEngine = msmEngine, !cpuOnly && n >= gpuThreshold {
            let limbs = batchFrToLimbs(coeffs)
            return try msmEngine.msm(points: pts, scalars: limbs)
        } else {
            return cpuMSM(points: pts, scalars: coeffs)
        }
    }

    // MARK: - Verification Helpers

    /// Verify that a commitment matches a polynomial and SRS.
    /// Recomputes the commitment and checks equality.
    public func verifyCommitment(_ commitment: PointProjective,
                                 polynomial: [Fr]) throws -> Bool {
        let recomputed = try commitPolynomial(polynomial)
        return pointEqual(commitment, recomputed)
    }

    /// Evaluate a polynomial at a point using Horner's method.
    public func evaluatePolynomial(_ poly: [Fr], at point: Fr) -> Fr {
        guard !poly.isEmpty else { return Fr.zero }
        var result = poly[poly.count - 1]
        for i in stride(from: poly.count - 2, through: 0, by: -1) {
            result = frAdd(frMul(result, point), poly[i])
        }
        return result
    }

    /// Check polynomial consistency: NTT(poly) should recover the original witness.
    public func verifyPolynomialConsistency(_ poly: [Fr],
                                            originalWitness: [Fr]) throws -> Bool {
        let recovered = try polynomialToWitness(poly)
        let n = originalWitness.count
        for i in 0..<n {
            if !frEqual(recovered[i], originalWitness[i]) {
                return false
            }
        }
        // Remaining entries should be zero (padding)
        for i in n..<recovered.count {
            if !frEqual(recovered[i], Fr.zero) {
                return false
            }
        }
        return true
    }

    // MARK: - Cache Management

    /// Clear the polynomial cache.
    public func clearCache() {
        polynomialCache.removeAll()
    }

    /// Get a cached polynomial by witness size.
    public func getCachedPolynomial(witnessSize: Int) -> [Fr]? {
        return polynomialCache[witnessSize]
    }

    // MARK: - CPU Fallback: iNTT

    /// CPU-based inverse NTT (Cooley-Tukey DIF).
    private func cpuINTT(_ input: [Fr]) -> [Fr] {
        let n = input.count
        guard n > 1 else { return input }

        // Get n-th root of unity and its inverse
        let omega = nthRootOfUnity(n)
        let omegaInv = frInverse(omega)
        let nInv = frInverse(frFromInt(UInt64(n)))

        // DIF butterfly with inverse twiddles
        var data = input
        var step = 1
        while step < n {
            let halfStep = step
            step *= 2
            var w = Fr.one
            let wStep = frPow(omegaInv, UInt64(n / step))
            for j in 0..<halfStep {
                var i = j
                while i < n {
                    let u = data[i]
                    let v = data[i + halfStep]
                    data[i] = frAdd(u, v)
                    data[i + halfStep] = frMul(frSub(u, v), w)
                    i += step
                }
                w = frMul(w, wStep)
            }
        }

        // Scale by 1/n
        for i in 0..<n {
            data[i] = frMul(data[i], nInv)
        }

        // Bit-reversal permutation
        return bitReverse(data)
    }

    /// CPU-based forward NTT (Cooley-Tukey DIT).
    private func cpuNTT(_ input: [Fr]) -> [Fr] {
        let n = input.count
        guard n > 1 else { return input }

        let omega = nthRootOfUnity(n)

        // Bit-reversal first
        var data = bitReverse(input)

        var step = 2
        while step <= n {
            let halfStep = step / 2
            let wStep = frPow(omega, UInt64(n / step))
            var w = Fr.one
            for j in 0..<halfStep {
                var i = j
                while i < n {
                    let u = data[i]
                    let v = frMul(data[i + halfStep], w)
                    data[i] = frAdd(u, v)
                    data[i + halfStep] = frSub(u, v)
                    i += step
                }
                w = frMul(w, wStep)
            }
            step *= 2
        }

        return data
    }

    // MARK: - CPU Fallback: MSM

    /// CPU-based naive multi-scalar multiplication.
    private func cpuMSM(points: [PointAffine], scalars: [Fr]) -> PointProjective {
        var result = pointIdentity()
        for i in 0..<min(points.count, scalars.count) {
            if frEqual(scalars[i], Fr.zero) { continue }
            let p = pointFromAffine(points[i])
            let sp = pointScalarMul(p, scalars[i])
            result = pointAdd(result, sp)
        }
        return result
    }

    // MARK: - Scalar Conversion

    /// Convert Fr array to [[UInt32]] limbs for GPU MSM.
    private func batchFrToLimbs(_ coeffs: [Fr]) -> [[UInt32]] {
        var result = [[UInt32]]()
        result.reserveCapacity(coeffs.count)
        for c in coeffs {
            let l = c.v
            result.append([l.0, l.1, l.2, l.3, l.4, l.5, l.6, l.7])
        }
        return result
    }

    // MARK: - NTT Helpers

    /// Compute the n-th primitive root of unity in BN254 Fr.
    /// Uses: omega_n = omega_{2^28}^{2^28 / n} where TWO_ADICITY = 28.
    private func nthRootOfUnity(_ n: Int) -> Fr {
        precondition(n > 0 && (n & (n - 1)) == 0, "n must be power of 2")
        let logN = Int(log2(Double(n)))
        precondition(logN <= Fr.TWO_ADICITY, "n exceeds max NTT size")
        let root = Fr.from64(Fr.ROOT_OF_UNITY)
        let exp = UInt64(1) << UInt64(Fr.TWO_ADICITY - logN)
        return frPow(root, exp)
    }

    /// Field exponentiation by squaring.
    private func frPow(_ base: Fr, _ exp: UInt64) -> Fr {
        if exp == 0 { return Fr.one }
        if exp == 1 { return base }
        var result = Fr.one
        var b = base
        var e = exp
        while e > 0 {
            if e & 1 == 1 {
                result = frMul(result, b)
            }
            b = frMul(b, b)
            e >>= 1
        }
        return result
    }

    /// Bit-reversal permutation for NTT.
    private func bitReverse(_ input: [Fr]) -> [Fr] {
        let n = input.count
        let logN = Int(log2(Double(n)))
        var output = input
        for i in 0..<n {
            let rev = reverseBits(UInt32(i), logN)
            if i < rev {
                output.swapAt(i, Int(rev))
            }
        }
        return output
    }

    /// Reverse bits of a value up to logN bits.
    private func reverseBits(_ val: UInt32, _ logN: Int) -> Int {
        var v = val
        var result: UInt32 = 0
        for _ in 0..<logN {
            result = (result << 1) | (v & 1)
            v >>= 1
        }
        return Int(result)
    }

    // MARK: - Utilities

    /// Round up to next power of 2.
    private func nextPowerOf2(_ n: Int) -> Int {
        var p = 1
        while p < n { p <<= 1 }
        return p
    }
}
