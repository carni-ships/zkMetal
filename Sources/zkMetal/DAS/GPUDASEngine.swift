// GPU-Accelerated Data Availability Sampling Engine
//
// Enhances the EIP-4844 / Danksharding DAS pipeline with Metal GPU acceleration
// for KZG proof generation. The main bottleneck in DAS proof generation is the
// multi-scalar multiplication (MSM) used to compute KZG commitment and witness
// polynomials. This engine routes those MSMs through the BLS12-381 Metal GPU
// pipeline (Pippenger bucket method) for significant speedup on Apple Silicon.
//
// Architecture:
//   - Blob encoding: raw bytes -> Fr381 polynomial coefficients
//   - KZG commitment: GPU MSM on Lagrange or monomial SRS
//   - KZG proofs: batched witness polynomial computation via GPU MSM
//   - Verification: pairing-based or recompute-based
//   - Random sampling: PRNG-driven index selection + batch verification

import Foundation

// MARK: - DAS Blob

/// A committed blob ready for DAS sampling.
/// Contains the polynomial, its KZG commitment, and the RS-extended codeword.
public struct DASBlob {
    /// Polynomial coefficients (the blob data as Fr381 elements).
    public let coefficients: [Fr381]

    /// KZG commitment to the polynomial.
    public let commitment: G1Projective381

    /// Reed-Solomon extended codeword (evaluations at roots of unity).
    public let codeword: [Fr381]

    /// DAS configuration used.
    public let config: DASConfig
}

// MARK: - KZG Proof

/// A KZG opening proof at a single evaluation point (BLS12-381).
public struct KZGProof381 {
    /// The evaluation point z (root of unity for the index).
    public let evalPoint: Fr381

    /// The claimed polynomial evaluation p(z).
    public let value: Fr381

    /// The witness: [q(s)]_1 where q(x) = (p(x) - p(z)) / (x - z).
    public let witness: G1Projective381

    /// Index in the extended codeword this proof corresponds to.
    public let index: Int

    public init(evalPoint: Fr381, value: Fr381, witness: G1Projective381, index: Int) {
        self.evalPoint = evalPoint
        self.value = value
        self.witness = witness
        self.index = index
    }
}

// MARK: - GPU DAS Engine

/// GPU-accelerated Data Availability Sampling engine.
///
/// Uses Metal GPU MSM for KZG commitment and proof generation, with CPU fallback
/// for small inputs or when GPU is unavailable.
///
/// Usage:
///   let engine = try GPUDASEngine(srs: srsPoints)
///   let blob = try engine.encodeBlob(data: rawBytes)
///   let proofs = try engine.generateProofs(blob: blob, indices: [0, 5, 20])
///   let valid = engine.verifyProof(commitment: blob.commitment, proof: proofs[0])
///   let allSamplesOk = try engine.sampleAndVerify(blob: blob, numSamples: 75)
public class GPUDASEngine {
    public let config: DASConfig

    /// SRS points for KZG (monomial basis): [G, sG, s^2 G, ..., s^(d-1) G].
    private let srs: [G1Affine381]

    /// GPU MSM engine (nil if GPU unavailable -- falls back to CPU).
    private let gpuMSM: BLS12381MSM?

    /// GPU MSM threshold: use GPU for MSMs with this many points or more.
    private let gpuThreshold: Int

    /// Initialize with SRS and optional configuration.
    /// - Parameters:
    ///   - srs: Structured reference string in affine form.
    ///   - config: DAS parameters (defaults to EIP-4844 standard).
    ///   - gpuThreshold: Minimum number of points to use GPU MSM (default 256).
    public init(srs: [G1Affine381], config: DASConfig = .eip4844, gpuThreshold: Int = 256) {
        self.srs = srs
        self.config = config
        self.gpuThreshold = gpuThreshold
        self.gpuMSM = try? BLS12381MSM()
    }

    /// Whether GPU acceleration is available.
    public var hasGPU: Bool { gpuMSM != nil }

    // MARK: - Blob Encoding

    /// Encode raw byte data into a DAS blob.
    ///
    /// Interprets the data as polynomial coefficients (padding with zeros to blobSize),
    /// computes the KZG commitment via GPU MSM, and RS-extends the polynomial.
    ///
    /// - Parameter data: Raw data as bytes. Each 32-byte chunk becomes one Fr381 element.
    ///                   If fewer than blobSize elements, the polynomial is zero-padded.
    /// - Returns: A committed DASBlob ready for proof generation and sampling.
    public func encodeBlob(data: [UInt8]) -> DASBlob {
        let k = config.blobSize

        // Convert bytes to field elements (32 bytes per element, big-endian)
        var coefficients = [Fr381](repeating: .zero, count: k)
        let numElements = min(data.count / 32, k)
        for i in 0..<numElements {
            let start = i * 32
            let end = min(start + 32, data.count)
            var chunk = [UInt8](repeating: 0, count: 32)
            for j in start..<end { chunk[j - start] = data[j] }
            coefficients[i] = bytesToFr381(chunk)
        }
        // Handle partial last element
        if data.count % 32 != 0 && numElements < k {
            var chunk = [UInt8](repeating: 0, count: 32)
            let start = numElements * 32
            for j in start..<data.count { chunk[j - start] = data[j] }
            if numElements < k {
                coefficients[numElements] = bytesToFr381(chunk)
            }
        }

        // Compute KZG commitment via GPU MSM
        let commitment = gpuCommit(coefficients)

        // RS extend via NTT
        let codeword = cpuNTT381Forward(coefficients)

        return DASBlob(
            coefficients: coefficients,
            commitment: commitment,
            codeword: codeword,
            config: config
        )
    }

    /// Encode pre-computed Fr381 field elements into a DAS blob.
    /// Useful when the polynomial is already in field element form.
    public func encodeFrBlob(data: [Fr381]) -> DASBlob {
        let k = config.blobSize
        var coefficients = [Fr381](repeating: .zero, count: k)
        for i in 0..<min(data.count, k) { coefficients[i] = data[i] }

        let commitment = gpuCommit(coefficients)
        let codeword = cpuNTT381Forward(coefficients)

        return DASBlob(
            coefficients: coefficients,
            commitment: commitment,
            codeword: codeword,
            config: config
        )
    }

    // MARK: - Proof Generation

    /// Batch generate KZG opening proofs at multiple indices using GPU MSM.
    ///
    /// For each index i in the extended codeword, computes:
    ///   - z_i = omega^i (evaluation point)
    ///   - v_i = p(z_i) (polynomial evaluation via Horner)
    ///   - q_i(x) = (p(x) - v_i) / (x - z_i) (quotient polynomial)
    ///   - proof_i = MSM(SRS, q_i) (KZG witness via GPU MSM)
    ///
    /// - Parameters:
    ///   - blob: A committed DASBlob.
    ///   - indices: Positions in the extended codeword to generate proofs for.
    /// - Returns: Array of KZGProof381, one per index.
    public func generateProofs(blob: DASBlob, indices: [Int]) throws -> [KZGProof381] {
        let n = config.extendedSize
        let logN = config.logExtendedSize
        let omega = fr381RootOfUnity(logN: logN)

        var proofs = [KZGProof381]()
        proofs.reserveCapacity(indices.count)

        for index in indices {
            precondition(index >= 0 && index < n, "Index out of range for extended codeword")

            let evalPoint = fr381Pow(omega, UInt64(index))
            let value = hornerEval381(blob.coefficients, at: evalPoint)

            // Compute quotient q(x) = (p(x) - v) / (x - z)
            var shifted = blob.coefficients
            shifted[0] = fr381Sub(shifted[0], value)
            let quotient = syntheticDiv381(shifted, z: evalPoint)

            // GPU MSM for witness
            let witness = try gpuMSMWitness(quotient)

            proofs.append(KZGProof381(
                evalPoint: evalPoint,
                value: value,
                witness: witness,
                index: index
            ))
        }

        return proofs
    }

    /// Generate a single KZG proof at an index.
    public func generateProof(blob: DASBlob, index: Int) throws -> KZGProof381 {
        return try generateProofs(blob: blob, indices: [index])[0]
    }

    // MARK: - Verification

    /// Verify a KZG proof by recomputing the witness and comparing.
    /// This is O(N) verification suitable for testing without pairings.
    ///
    /// - Parameters:
    ///   - commitment: The KZG commitment to the blob polynomial.
    ///   - proof: The KZG opening proof to verify.
    ///   - coefficients: The polynomial coefficients (needed for recompute verification).
    /// - Returns: true if the proof is valid.
    public func verifyProof(commitment: G1Projective381, proof: KZGProof381,
                             coefficients: [Fr381]) -> Bool {
        // Recompute expected value
        let expectedValue = hornerEval381(coefficients, at: proof.evalPoint)
        let vStd = fr381ToInt(proof.value)
        let evStd = fr381ToInt(expectedValue)
        guard vStd == evStd else { return false }

        // Recompute witness
        var shifted = coefficients
        shifted[0] = fr381Sub(shifted[0], expectedValue)
        let quotient = syntheticDiv381(shifted, z: proof.evalPoint)
        guard let expectedWitness = try? gpuMSMWitness(quotient) else { return false }

        // Compare witness points
        guard let proofAff = g1_381ToAffine(proof.witness),
              let expectedAff = g1_381ToAffine(expectedWitness) else {
            return g1_381IsIdentity(proof.witness) && g1_381IsIdentity(expectedWitness)
        }
        return fp381ToInt(proofAff.x) == fp381ToInt(expectedAff.x) &&
               fp381ToInt(proofAff.y) == fp381ToInt(expectedAff.y)
    }

    /// Verify a KZG proof using the SRS secret (test mode, no pairings needed).
    ///
    /// Checks: C == [v]*G + [s - z]*witness
    public func verifyProof(commitment: G1Projective381, proof: KZGProof381,
                             srsSecret: [UInt64]) -> Bool {
        let gen = g1_381FromAffine(bls12381G1Generator())
        let vG = g1_381ScalarMul(gen, fr381ToInt(proof.value))

        let sFr = fr381Mul(Fr381.from64(srsSecret), Fr381.from64(Fr381.R2_MOD_R))
        let sMz = fr381Sub(sFr, proof.evalPoint)
        let szProof = g1_381ScalarMul(proof.witness, fr381ToInt(sMz))

        let expected = g1_381Add(vG, szProof)

        guard let cAff = g1_381ToAffine(commitment),
              let eAff = g1_381ToAffine(expected) else {
            return g1_381IsIdentity(commitment) && g1_381IsIdentity(expected)
        }
        return fp381ToInt(cAff.x) == fp381ToInt(eAff.x) &&
               fp381ToInt(cAff.y) == fp381ToInt(eAff.y)
    }

    /// Verify a KZG proof using BLS12-381 pairings (production mode).
    /// e(C - [v]G1, G2) * e(-proof, [s]G2 - [z]G2) == 1
    public func verifyProofWithPairing(commitment: G1Projective381, proof: KZGProof381,
                                        g2Gen: G2Affine381, sG2: G2Affine381) -> Bool {
        let gen = g1_381FromAffine(bls12381G1Generator())
        let vG = g1_381ScalarMul(gen, fr381ToInt(proof.value))
        let cMinusVG = g1_381Add(commitment, g1_381Negate(vG))

        guard let p1Aff = g1_381ToAffine(cMinusVG) else {
            return g1_381IsIdentity(proof.witness)
        }

        let zG2 = g2_381ScalarMul(g2_381FromAffine(g2Gen), fr381ToInt(proof.evalPoint))
        let sMinusZG2 = g2_381Add(g2_381FromAffine(sG2), g2_381Negate(zG2))
        guard let q2Aff = g2_381ToAffine(sMinusZG2) else { return false }

        guard let proofAff = g1_381ToAffine(proof.witness) else { return false }
        let negProofAff = g1_381NegateAffine(proofAff)
        return bls12381PairingCheck([(p1Aff, g2Gen), (negProofAff, q2Aff)])
    }

    // MARK: - Random Sampling

    /// Perform random data availability sampling: select random indices, generate proofs,
    /// and verify all proofs.
    ///
    /// This is the core DAS protocol: a light client draws numSamples random positions
    /// in the extended codeword and verifies KZG proofs at those positions. If all verify,
    /// the client gains statistical confidence that the blob data is available.
    ///
    /// - Parameters:
    ///   - blob: The committed DASBlob to sample.
    ///   - numSamples: Number of random samples to draw (default: config.sampleCount).
    ///   - seed: PRNG seed (default: 0 uses current time).
    /// - Returns: true if all samples verify correctly.
    public func sampleAndVerify(blob: DASBlob, numSamples: Int? = nil, seed: UInt64 = 0) throws -> Bool {
        let count = numSamples ?? config.sampleCount
        let indices = randomSampleIndices(count: count, seed: seed)
        let proofs = try generateProofs(blob: blob, indices: indices)

        for proof in proofs {
            if !verifyProof(commitment: blob.commitment, proof: proof,
                            coefficients: blob.coefficients) {
                return false
            }
        }
        return true
    }

    /// Perform random DAS with SRS-secret verification (test mode).
    public func sampleAndVerify(blob: DASBlob, numSamples: Int? = nil,
                                 srsSecret: [UInt64], seed: UInt64 = 0) throws -> Bool {
        let count = numSamples ?? config.sampleCount
        let indices = randomSampleIndices(count: count, seed: seed)
        let proofs = try generateProofs(blob: blob, indices: indices)

        for proof in proofs {
            if !verifyProof(commitment: blob.commitment, proof: proof, srsSecret: srsSecret) {
                return false
            }
        }
        return true
    }

    // MARK: - Random Index Generation

    /// Generate random sample indices using a seeded PRNG.
    public func randomSampleIndices(count: Int, seed: UInt64 = 0) -> [Int] {
        let n = config.extendedSize
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

    // MARK: - Private: GPU MSM

    /// Compute KZG commitment via GPU MSM: C = sum_i coeff_i * SRS[i].
    private func gpuCommit(_ coefficients: [Fr381]) -> G1Projective381 {
        let n = min(coefficients.count, srs.count)

        // Filter out zero scalars to avoid Pippenger zero-scalar bug
        let (filteredPoints, filteredScalars) = filterZeroScalars(
            points: Array(srs.prefix(n)), scalars: Array(coefficients.prefix(n)))
        guard !filteredPoints.isEmpty else { return g1_381Identity() }

        if let gpu = gpuMSM, filteredPoints.count >= gpuThreshold {
            if let result = try? gpu.msmFr(points: filteredPoints, scalars: filteredScalars) {
                return result
            }
        }
        // CPU fallback
        return cpuMSM381(points: filteredPoints, scalars: filteredScalars)
    }

    /// Compute KZG witness via GPU MSM: W = sum_i q_i * SRS[i].
    private func gpuMSMWitness(_ quotient: [Fr381]) throws -> G1Projective381 {
        guard !quotient.isEmpty else { return g1_381Identity() }
        guard srs.count >= quotient.count else {
            throw RSError.gpuError("SRS too small for polynomial degree")
        }

        // Filter out zero scalars to avoid Pippenger zero-scalar bug
        let (filteredPoints, filteredScalars) = filterZeroScalars(
            points: Array(srs.prefix(quotient.count)), scalars: quotient)
        guard !filteredPoints.isEmpty else { return g1_381Identity() }

        if let gpu = gpuMSM, filteredPoints.count >= gpuThreshold {
            if let result = try? gpu.msmFr(points: filteredPoints, scalars: filteredScalars) {
                return result
            }
        }
        // CPU fallback
        return cpuMSM381(points: filteredPoints, scalars: filteredScalars)
    }

    /// Filter out zero scalars from MSM inputs.
    /// Works around a Pippenger MSM bug where zero scalars paired with
    /// non-trivial points can produce incorrect contributions to the sum.
    private func filterZeroScalars(points: [G1Affine381], scalars: [Fr381]) -> ([G1Affine381], [Fr381]) {
        var filteredPoints = [G1Affine381]()
        var filteredScalars = [Fr381]()
        filteredPoints.reserveCapacity(scalars.count)
        filteredScalars.reserveCapacity(scalars.count)
        let zeroLimbs: [UInt64] = [0, 0, 0, 0]
        for i in 0..<scalars.count {
            if fr381ToInt(scalars[i]) != zeroLimbs {
                filteredPoints.append(points[i])
                filteredScalars.append(scalars[i])
            }
        }
        return (filteredPoints, filteredScalars)
    }

    // MARK: - Private: CPU Helpers

    /// CPU MSM on BLS12-381 G1.
    private func cpuMSM381(points: [G1Affine381], scalars: [Fr381]) -> G1Projective381 {
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

    /// Forward NTT over BLS12-381 Fr (Cooley-Tukey radix-2 DIT).
    private func cpuNTT381Forward(_ input: [Fr381]) -> [Fr381] {
        let n = config.extendedSize
        let logN = config.logExtendedSize
        precondition(logN <= Fr381.TWO_ADICITY, "Extended size exceeds Fr381 2-adicity")

        // Pad to extended size
        var padded = [Fr381](repeating: .zero, count: n)
        for i in 0..<min(input.count, n) { padded[i] = input[i] }

        // Bit-reversal permutation
        var data = [Fr381](repeating: .zero, count: n)
        for i in 0..<n {
            let rev = bitReverse(i, bits: logN)
            data[rev] = padded[i]
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

    /// Synthetic division: (p(x) - value) / (x - z), assuming value already subtracted from p[0].
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

    /// Horner evaluation of polynomial at a point.
    private func hornerEval381(_ coeffs: [Fr381], at z: Fr381) -> Fr381 {
        var acc = Fr381.zero
        for i in stride(from: coeffs.count - 1, through: 0, by: -1) {
            acc = fr381Add(fr381Mul(acc, z), coeffs[i])
        }
        return acc
    }

    /// Convert 32-byte big-endian chunk to Fr381 (reduce mod r).
    private func bytesToFr381(_ bytes: [UInt8]) -> Fr381 {
        precondition(bytes.count == 32)
        var limbs: [UInt64] = [0, 0, 0, 0]
        for i in 0..<4 {
            let byteOffset = 24 - i * 8
            for j in 0..<8 {
                if byteOffset + j < bytes.count {
                    limbs[i] |= UInt64(bytes[byteOffset + j]) << UInt64((7 - j) * 8)
                }
            }
        }
        // Reduce mod r if needed
        if !isLessThanFr381Modulus(limbs) {
            var borrow: UInt64 = 0
            for i in 0..<4 {
                let (diff, o1) = limbs[i].subtractingReportingOverflow(Fr381.P[i])
                let (diff2, o2) = diff.subtractingReportingOverflow(borrow)
                limbs[i] = diff2
                borrow = (o1 ? 1 : 0) + (o2 ? 1 : 0)
            }
        }
        let raw = Fr381.from64(limbs)
        return fr381Mul(raw, Fr381.from64(Fr381.R2_MOD_R))
    }

    /// Check if limbs < Fr381 modulus.
    private func isLessThanFr381Modulus(_ limbs: [UInt64]) -> Bool {
        for i in stride(from: 3, through: 0, by: -1) {
            if limbs[i] < Fr381.P[i] { return true }
            if limbs[i] > Fr381.P[i] { return false }
        }
        return false
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
}
