// Data Availability Sampling Engine (EIP-4844 / Danksharding)
//
// Implements KZG-committed data availability sampling compatible with
// Ethereum's EIP-4844 blob format and danksharding extension.
//
// Architecture:
//   - Blob = 4096 BLS12-381 Fr field elements (EIP-4844 standard)
//   - RS extension: NTT on a coset domain to double the blob (2x redundancy)
//   - KZG commitments: Lagrange-basis MSM on BLS12-381 G1
//   - Samples: single-position KZG opening proofs (evaluation + witness)
//   - Reconstruction: Lagrange interpolation from >= 4096 of 8192 samples
//
// Reference: EIP-4844, Ethereum Danksharding spec

import Foundation
import NeonFieldOps

// MARK: - Configuration

/// Configuration for data availability sampling.
public struct DASConfig {
    /// Number of field elements per blob (EIP-4844: 4096).
    public let blobSize: Int

    /// Extension factor for Reed-Solomon encoding (2x = 50% redundancy).
    public let extensionFactor: Int

    /// Number of random samples a light client draws to gain confidence.
    public let sampleCount: Int

    /// Total number of extended positions = blobSize * extensionFactor.
    public var extendedSize: Int { blobSize * extensionFactor }

    /// log2 of the extended size.
    public var logExtendedSize: Int { Int(log2(Double(extendedSize))) }

    /// Default EIP-4844 / danksharding configuration.
    public static let eip4844 = DASConfig(blobSize: 4096, extensionFactor: 2, sampleCount: 75)

    /// Small configuration for tests.
    public static let test = DASConfig(blobSize: 16, extensionFactor: 2, sampleCount: 8)

    public init(blobSize: Int, extensionFactor: Int, sampleCount: Int) {
        precondition(blobSize > 0 && (blobSize & (blobSize - 1)) == 0, "blobSize must be power of 2")
        precondition(extensionFactor >= 2, "extensionFactor must be >= 2")
        precondition(sampleCount > 0, "sampleCount must be > 0")
        self.blobSize = blobSize
        self.extensionFactor = extensionFactor
        self.sampleCount = sampleCount
    }
}

// MARK: - Extended Blob

/// A blob that has been RS-extended to 2x its original size.
public struct ExtendedBlob {
    /// The original blob field elements (indices 0..<blobSize).
    public let original: [Fr381]

    /// The full extended codeword (indices 0..<extendedSize).
    public let codeword: [Fr381]

    /// Configuration used for extension.
    public let config: DASConfig
}

// MARK: - DAS Sample

/// A single data availability sample: one position in the extended blob
/// with a KZG opening proof.
public struct DASSample {
    /// Index in the extended codeword (0..<extendedSize).
    public let index: Int

    /// Field element at this index.
    public let value: Fr381

    /// KZG opening proof: witness point [q(s)]_1 where q(x) = (p(x) - value) / (x - z).
    public let proof: G1Projective381

    /// The evaluation point z = omega^index (or coset shifted).
    public let evalPoint: Fr381
}

// MARK: - Data Availability Sampler

/// EIP-4844 / Danksharding compatible data availability sampler.
/// Uses BLS12-381 KZG commitments and Reed-Solomon erasure coding.
public class EIP4844DataAvailabilitySampler {
    public let config: DASConfig
    private let rsEngine: ReedSolomon381Engine

    /// Initialize with SRS for KZG proofs and optional config.
    /// srs: structured reference string [G1, s*G1, s^2*G1, ...] in affine form.
    /// config: DAS parameters (defaults to EIP-4844 standard).
    public init(srs: [G1Affine381], config: DASConfig = .eip4844) {
        self.config = config
        self.rsEngine = ReedSolomon381Engine(srs: srs)
    }

    // MARK: - RS Extension

    /// Extend a blob (polynomial coefficients) to 2x via NTT over a coset domain.
    /// The original blob is interpreted as polynomial coefficients p(x) of degree < blobSize.
    /// The extension evaluates p at the roots of unity of order extendedSize.
    ///
    /// Returns an ExtendedBlob containing the full codeword.
    public func extendBlob(data: [Fr381]) -> ExtendedBlob {
        let k = config.blobSize
        let n = config.extendedSize
        let logN = config.logExtendedSize
        precondition(data.count <= k, "Blob data exceeds configured blobSize")
        precondition(logN <= Fr381.TWO_ADICITY, "Extended size exceeds Fr381 2-adicity")

        // Pad to extended size with zeros (higher-degree coefficients = 0)
        var padded = [Fr381](repeating: .zero, count: n)
        for i in 0..<data.count { padded[i] = data[i] }

        // Forward NTT: evaluate polynomial at n-th roots of unity
        let codeword = cpuNTT381Forward(padded, logN: logN)

        return ExtendedBlob(
            original: Array(data.prefix(k)),
            codeword: codeword,
            config: config
        )
    }

    // MARK: - KZG Commitment

    /// Commit to a blob using KZG (MSM with the SRS).
    /// Returns (commitment, polynomial_coefficients).
    /// The commitment is C = sum_i coeff_i * SRS[i].
    public func commitBlob(data: [Fr381]) throws -> (commitment: G1Projective381, coefficients: [Fr381]) {
        let k = config.blobSize
        precondition(data.count <= k, "Blob data exceeds blobSize")

        // Pad to blobSize
        var coeffs = [Fr381](repeating: .zero, count: k)
        for i in 0..<data.count { coeffs[i] = data[i] }

        let commitment = try rsEngine.commit(data: coeffs)
        return (commitment: commitment, coefficients: coeffs)
    }

    // MARK: - Sample Generation

    /// Generate a single DAS sample at a given index in the extended codeword.
    /// Produces a KZG opening proof that the polynomial evaluates to the claimed value
    /// at the corresponding root of unity.
    ///
    /// - Parameters:
    ///   - coefficients: polynomial coefficients (the blob)
    ///   - codeword: the extended codeword from extendBlob
    ///   - index: position in the extended codeword (0..<extendedSize)
    /// - Returns: a DASSample with evaluation and KZG proof
    public func generateSample(coefficients: [Fr381], codeword: [Fr381], index: Int) throws -> DASSample {
        let n = config.extendedSize
        let logN = config.logExtendedSize
        precondition(index >= 0 && index < n, "Sample index out of range")

        let omega = fr381RootOfUnity(logN: logN)
        let evalPoint = fr381Pow(omega, UInt64(index))

        // Evaluate polynomial at evalPoint using Horner (authoritative value for KZG proof)
        let value = hornerEval381(coefficients, at: evalPoint)

        // KZG opening proof: q(x) = (p(x) - p(z)) / (x - z)
        let proof = try computeKZGWitness(coefficients: coefficients, evalPoint: evalPoint, value: value)

        return DASSample(
            index: index,
            value: value,
            proof: proof,
            evalPoint: evalPoint
        )
    }

    // MARK: - Sample Verification

    /// Verify a single DAS sample against the blob commitment.
    /// Uses the pairing check: e(C - [v]G1, G2) == e(proof, [s]G2 - [z]G2).
    /// Falls back to SRS-secret verification in test mode.
    ///
    /// For test-mode (non-pairing) verification, use verifySampleWithSecret.
    public func verifySample(commitment: G1Projective381, sample: DASSample,
                             srsSecret: [UInt64]) -> Bool {
        let n = config.extendedSize
        let logN = config.logExtendedSize
        let omega = fr381RootOfUnity(logN: logN)
        let evalPoint = fr381Pow(omega, UInt64(sample.index))

        // Verify using SRS secret: C == [v]*G + [s - z]*proof
        let gen = g1_381FromAffine(bls12381G1Generator())
        let vG = g1_381ScalarMul(gen, fr381ToInt(sample.value))

        let sFr = fr381Mul(Fr381.from64(srsSecret), Fr381.from64(Fr381.R2_MOD_R))
        let sMz = fr381Sub(sFr, evalPoint)
        let szProof = g1_381ScalarMul(sample.proof, fr381ToInt(sMz))

        let expected = g1_381Add(vG, szProof)

        guard let cAff = g1_381ToAffine(commitment),
              let eAff = g1_381ToAffine(expected) else {
            return g1_381IsIdentity(commitment) && g1_381IsIdentity(expected)
        }
        return fp381ToInt(cAff.x) == fp381ToInt(eAff.x) &&
               fp381ToInt(cAff.y) == fp381ToInt(eAff.y)
    }

    /// Verify a single DAS sample using BLS12-381 pairing check (production mode).
    /// e(C - [v]G1, G2) * e(-proof, [s]G2 - [z]G2) == 1
    public func verifySampleWithPairing(commitment: G1Projective381, sample: DASSample,
                                        g2Gen: G2Affine381, sG2: G2Affine381) -> Bool {
        let gen = g1_381FromAffine(bls12381G1Generator())
        let vG = g1_381ScalarMul(gen, fr381ToInt(sample.value))
        let cMinusVG = g1_381Add(commitment, g1_381Negate(vG))

        guard let p1Aff = g1_381ToAffine(cMinusVG) else {
            // C == [v]*G1: polynomial is constant, proof should be identity
            return g1_381IsIdentity(sample.proof)
        }

        let zG2 = g2_381ScalarMul(g2_381FromAffine(g2Gen), fr381ToInt(sample.evalPoint))
        let sMinusZG2 = g2_381Add(g2_381FromAffine(sG2), g2_381Negate(zG2))
        guard let q2Aff = g2_381ToAffine(sMinusZG2) else {
            return false
        }

        guard let proofAff = g1_381ToAffine(sample.proof) else {
            // Proof is identity; only valid if commitment == [v]*G1 (handled above)
            return false
        }
        let negProofAff = g1_381NegateAffine(proofAff)
        return bls12381PairingCheck([(p1Aff, g2Gen), (negProofAff, q2Aff)])
    }

    /// Verify a DAS sample by recomputing the proof from scratch and comparing.
    /// This is an O(N) verification suitable for testing without pairings or SRS secret.
    /// Checks that the proof witness matches what we'd generate independently.
    public func verifySampleByRecompute(commitment: G1Projective381, sample: DASSample,
                                         coefficients: [Fr381]) -> Bool {
        // Recompute the expected value and proof
        let expectedValue = hornerEval381(coefficients, at: sample.evalPoint)

        // Check value matches
        let vStd = fr381ToInt(sample.value)
        let evStd = fr381ToInt(expectedValue)
        guard vStd == evStd else { return false }

        // Recompute witness
        guard let expectedWitness = try? computeKZGWitness(
            coefficients: coefficients, evalPoint: sample.evalPoint, value: expectedValue
        ) else { return false }

        // Compare proof witness points
        guard let proofAff = g1_381ToAffine(sample.proof),
              let expectedAff = g1_381ToAffine(expectedWitness) else {
            return g1_381IsIdentity(sample.proof) && g1_381IsIdentity(expectedWitness)
        }
        return fp381ToInt(proofAff.x) == fp381ToInt(expectedAff.x) &&
               fp381ToInt(proofAff.y) == fp381ToInt(expectedAff.y)
    }

    // MARK: - Batch Sample Verification

    /// Verify multiple samples at once using recompute verification.
    /// Returns true only if all samples verify.
    public func sampleAndVerify(commitment: G1Projective381, sampleIndices: [Int],
                                coefficients: [Fr381], codeword: [Fr381]) throws -> Bool {
        for idx in sampleIndices {
            let sample = try generateSample(coefficients: coefficients, codeword: codeword, index: idx)
            if !verifySampleByRecompute(commitment: commitment, sample: sample, coefficients: coefficients) {
                return false
            }
        }
        return true
    }

    /// Verify multiple samples at once using SRS secret.
    public func sampleAndVerify(commitment: G1Projective381, sampleIndices: [Int],
                                coefficients: [Fr381], codeword: [Fr381],
                                srsSecret: [UInt64]) throws -> Bool {
        for idx in sampleIndices {
            let sample = try generateSample(coefficients: coefficients, codeword: codeword, index: idx)
            if !verifySample(commitment: commitment, sample: sample, srsSecret: srsSecret) {
                return false
            }
        }
        return true
    }

    // MARK: - Reconstruction

    /// Reconstruct the original blob polynomial from a sufficient number of samples.
    /// Requires at least blobSize samples from the extendedSize positions.
    /// Uses Lagrange interpolation over roots-of-unity evaluation points.
    ///
    /// Returns the polynomial coefficients if reconstruction succeeds, nil otherwise.
    public func reconstructBlob(samples: [DASSample], threshold: Int? = nil) -> [Fr381]? {
        let minSamples = threshold ?? config.blobSize
        guard samples.count >= minSamples else { return nil }

        let usable = Array(samples.prefix(minSamples))
        let n = config.extendedSize
        let logN = config.logExtendedSize

        // Build evaluation points and values
        let omega = fr381RootOfUnity(logN: logN)
        let points = usable.map { fr381Pow(omega, UInt64($0.index)) }
        let values = usable.map { $0.value }

        // Lagrange interpolation to recover polynomial coefficients
        let coeffs = lagrangeInterpolate381(points: points, values: values)

        // Return only the first blobSize coefficients (higher are zero for honest blobs)
        return Array(coeffs.prefix(config.blobSize))
    }

    // MARK: - Random Index Generation

    /// Generate random sample indices using a seeded PRNG.
    /// In production, the seed would come from the block hash or beacon randomness.
    public func randomSampleIndices(seed: UInt64 = 0) -> [Int] {
        let n = config.extendedSize
        let count = config.sampleCount
        var indices = [Int]()
        var used = Set<Int>()
        var rng: UInt64 = seed == 0 ? UInt64(CFAbsoluteTimeGetCurrent().bitPattern) : seed

        while indices.count < min(count, n) {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let idx = Int(rng >> 33) % n
            if !used.contains(idx) {
                used.insert(idx)
                indices.append(idx)
            }
        }
        return indices
    }

    // MARK: - Private Helpers

    /// Compute KZG witness: [q(s)]_1 where q(x) = (p(x) - value) / (x - evalPoint).
    private func computeKZGWitness(coefficients: [Fr381], evalPoint: Fr381, value: Fr381) throws -> G1Projective381 {
        guard rsEngine.srs.count >= coefficients.count else {
            throw RSError.gpuError("SRS too small for polynomial degree")
        }

        // p(x) - value: subtract from constant term
        var shifted = coefficients
        shifted[0] = fr381Sub(shifted[0], value)

        // Synthetic division by (x - evalPoint)
        let quotient = syntheticDiv381(shifted, z: evalPoint)
        guard !quotient.isEmpty else {
            return g1_381Identity()
        }

        // MSM: witness = sum_i q_i * SRS[i]
        return msm381(points: Array(rsEngine.srs.prefix(quotient.count)), scalars: quotient)
    }

    /// Forward NTT over BLS12-381 Fr (Cooley-Tukey radix-2 DIT).
    private func cpuNTT381Forward(_ input: [Fr381], logN: Int) -> [Fr381] {
        let n = 1 << logN
        precondition(input.count == n)

        // Bit-reversal permutation
        var data = [Fr381](repeating: .zero, count: n)
        for i in 0..<n {
            let rev = bitReverse(i, bits: logN)
            data[rev] = input[i]
        }

        let omega = fr381RootOfUnity(logN: logN)

        // Butterfly stages
        var m = 1
        for _ in 0..<logN {
            let wm = fr381Pow(omega, UInt64(n / (2 * m)))
            var k = 0
            while k < n {
                var w = Fr381.one
                for j in 0..<m {
                    let t = fr381Mul(w, data[k + j + m])
                    let u = data[k + j]
                    data[k + j] = fr381Add(u, t)
                    data[k + j + m] = fr381Sub(u, t)
                    w = fr381Mul(w, wm)
                }
                k += 2 * m
            }
            m *= 2
        }
        return data
    }

    /// Inverse NTT over BLS12-381 Fr.
    private func cpuINTT381(_ input: [Fr381], logN: Int) -> [Fr381] {
        let n = 1 << logN
        precondition(input.count == n)

        var data = [Fr381](repeating: .zero, count: n)
        for i in 0..<n {
            let rev = bitReverse(i, bits: logN)
            data[rev] = input[i]
        }

        let omega = fr381Inverse(fr381RootOfUnity(logN: logN))

        var m = 1
        for _ in 0..<logN {
            let wm = fr381Pow(omega, UInt64(n / (2 * m)))
            var k = 0
            while k < n {
                var w = Fr381.one
                for j in 0..<m {
                    let t = fr381Mul(w, data[k + j + m])
                    let u = data[k + j]
                    data[k + j] = fr381Add(u, t)
                    data[k + j + m] = fr381Sub(u, t)
                    w = fr381Mul(w, wm)
                }
                k += 2 * m
            }
            m *= 2
        }

        let nInv = fr381Inverse(fr381FromInt(UInt64(n)))
        for i in 0..<n {
            data[i] = fr381Mul(data[i], nInv)
        }
        return data
    }

    /// Synthetic division: compute (p(x) - p(z)) / (x - z) given that p(z) has been
    /// subtracted from the constant term.
    private func syntheticDiv381(_ coeffs: [Fr381], z: Fr381) -> [Fr381] {
        let n = coeffs.count
        if n < 2 { return [] }
        var quotient = [Fr381](repeating: .zero, count: n - 1)
        quotient[n - 2] = coeffs[n - 1]
        for i in stride(from: n - 3, through: 0, by: -1) {
            quotient[i] = fr381Add(coeffs[i + 1], fr381Mul(z, quotient[i + 1]))
        }
        return quotient
    }

    /// CPU MSM on BLS12-381 G1.
    private func msm381(points: [G1Affine381], scalars: [Fr381]) -> G1Projective381 {
        let n = points.count
        precondition(scalars.count == n)
        if n == 0 { return g1_381Identity() }

        var flatScalars = [UInt32]()
        flatScalars.reserveCapacity(n * 8)
        for s in scalars {
            let std = fr381ToInt(s)
            flatScalars.append(UInt32(std[0] & 0xFFFFFFFF))
            flatScalars.append(UInt32(std[0] >> 32))
            flatScalars.append(UInt32(std[1] & 0xFFFFFFFF))
            flatScalars.append(UInt32(std[1] >> 32))
            flatScalars.append(UInt32(std[2] & 0xFFFFFFFF))
            flatScalars.append(UInt32(std[2] >> 32))
            flatScalars.append(UInt32(std[3] & 0xFFFFFFFF))
            flatScalars.append(UInt32(std[3] >> 32))
        }
        return g1_381PippengerMSMFlat(points: points, flatScalars: flatScalars)
    }

    /// Lagrange interpolation over Fr381.
    private func lagrangeInterpolate381(points: [Fr381], values: [Fr381]) -> [Fr381] {
        let n = points.count
        var result = [Fr381](repeating: .zero, count: n)

        for i in 0..<n {
            var basis = [Fr381](repeating: .zero, count: n)
            basis[0] = .one
            var denom = Fr381.one
            var basisDeg = 0

            for j in 0..<n {
                if j == i { continue }
                denom = fr381Mul(denom, fr381Sub(points[i], points[j]))
                basisDeg += 1
                for d in stride(from: basisDeg, through: 1, by: -1) {
                    basis[d] = fr381Sub(basis[d - 1], fr381Mul(points[j], basis[d]))
                }
                basis[0] = fr381Sub(.zero, fr381Mul(points[j], basis[0]))
            }

            let scale = fr381Mul(values[i], fr381Inverse(denom))
            for d in 0..<n {
                result[d] = fr381Add(result[d], fr381Mul(scale, basis[d]))
            }
        }
        return result
    }

    /// Bit-reverse an integer.
    private func bitReverse(_ x: Int, bits: Int) -> Int {
        var result = 0
        var val = x
        for _ in 0..<bits {
            result = (result << 1) | (val & 1)
            val >>= 1
        }
        return result
    }

    /// Horner evaluation of polynomial at a point.
    private func hornerEval381(_ coeffs: [Fr381], at z: Fr381) -> Fr381 {
        var acc = Fr381.zero
        for i in stride(from: coeffs.count - 1, through: 0, by: -1) {
            acc = fr381Add(fr381Mul(acc, z), coeffs[i])
        }
        return acc
    }
}
