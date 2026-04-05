// EIP-4844 Blob KZG Verifier
// Implements Ethereum blob KZG proof verification per the EIP-4844 specification.
// Uses BLS12-381 pairing, Fr381 scalars, and the Ethereum trusted setup format.
//
// Reference: https://eips.ethereum.org/EIPS/eip-4844
//            https://github.com/ethereum/consensus-specs/blob/dev/specs/deneb/polynomial-commitments.md

import Foundation
import NeonFieldOps

// MARK: - EIP-4844 Constants

/// Ethereum blob field elements per blob (4096 = 2^12).
public let FIELD_ELEMENTS_PER_BLOB = 4096

/// Bytes per field element in blob encoding (32 bytes, big-endian, reduced mod BLS_MODULUS).
public let BYTES_PER_FIELD_ELEMENT = 32

/// Total bytes per blob: 4096 * 32 = 131072.
public let BYTES_PER_BLOB = FIELD_ELEMENTS_PER_BLOB * BYTES_PER_FIELD_ELEMENT

/// BLS12-381 scalar field modulus r (the "BLS_MODULUS" in EIP-4844).
/// r = 52435875175126190479447740508185965837690552500527637822603658699938581184513
public let BLS_MODULUS: [UInt64] = Fr381.P

/// Domain separator for Fiat-Shamir challenge in EIP-4844.
private let FIAT_SHAMIR_PROTOCOL_DOMAIN = "FSBLOBVERIFY_V1_"

/// Domain separator for computing the challenge z in verify_blob_kzg_proof.
private let RANDOM_CHALLENGE_KZG_BATCH_DOMAIN = "RCKZGBATCH___V1_"

// MARK: - EIP-4844 Trusted Setup (SRS)

/// Structured Reference String for EIP-4844 blob KZG.
/// Contains G1 Lagrange points (4096) and the G2 monomial points (needed for pairing checks).
public struct EIP4844TrustedSetup {
    /// G1 points in Lagrange basis: [L_0(s), L_1(s), ..., L_{4095}(s)] where L_i are Lagrange polys
    /// over the roots of unity domain.
    public let g1Lagrange: [G1Affine381]

    /// G2 monomial points: at minimum [G2, s*G2] for the pairing verification.
    /// g2Monomial[0] = G2 generator, g2Monomial[1] = [s]*G2.
    public let g2Monomial: [G2Affine381]

    public init(g1Lagrange: [G1Affine381], g2Monomial: [G2Affine381]) {
        precondition(g1Lagrange.count == FIELD_ELEMENTS_PER_BLOB,
                     "G1 Lagrange basis must have \(FIELD_ELEMENTS_PER_BLOB) points")
        precondition(g2Monomial.count >= 2,
                     "G2 monomial basis must have at least 2 points ([G2], [s]G2)")
        self.g1Lagrange = g1Lagrange
        self.g2Monomial = g2Monomial
    }

    /// Load trusted setup from the Ethereum KZG ceremony output files.
    /// g1LagrangeBytes: concatenated compressed G1 points (48 bytes each, 4096 points).
    /// g2MonomialBytes: concatenated compressed G2 points (96 bytes each, at least 2 points).
    public static func load(g1LagrangeBytes: [UInt8], g2MonomialBytes: [UInt8]) -> EIP4844TrustedSetup? {
        let g1Count = g1LagrangeBytes.count / 48
        guard g1Count == FIELD_ELEMENTS_PER_BLOB else { return nil }
        let g2Count = g2MonomialBytes.count / 96
        guard g2Count >= 2 else { return nil }

        var g1Points = [G1Affine381]()
        g1Points.reserveCapacity(g1Count)
        for i in 0..<g1Count {
            let start = i * 48
            let compressed = Array(g1LagrangeBytes[start..<start + 48])
            guard let proj = bls12381G1Decompress(compressed),
                  let aff = g1_381ToAffine(proj) else {
                return nil
            }
            g1Points.append(aff)
        }

        var g2Points = [G2Affine381]()
        g2Points.reserveCapacity(g2Count)
        for i in 0..<g2Count {
            let start = i * 96
            let compressed = Array(g2MonomialBytes[start..<start + 96])
            guard let proj = bls12381G2Decompress(compressed),
                  let aff = g2_381ToAffine(proj) else {
                return nil
            }
            g2Points.append(aff)
        }

        return EIP4844TrustedSetup(g1Lagrange: g1Points, g2Monomial: g2Points)
    }

    /// Generate a test trusted setup from a known secret (NOT secure, for testing only).
    /// The Lagrange basis is computed from the monomial basis via inverse NTT on the SRS.
    public static func generateTestSetup(secret: Fr381) -> EIP4844TrustedSetup {
        let n = FIELD_ELEMENTS_PER_BLOB
        let g1Gen = bls12381G1Generator()
        let g2Gen = bls12381G2Generator()

        // Monomial SRS: [G1, s*G1, s^2*G1, ..., s^(n-1)*G1]
        var g1Monomial = [G1Projective381]()
        g1Monomial.reserveCapacity(n)
        let g1Proj = g1_381FromAffine(g1Gen)
        var sPow = Fr381.one
        let sLimbs = fr381ToInt(secret)
        for _ in 0..<n {
            let pt = g1_381ScalarMul(g1Proj, fr381ToInt(sPow))
            g1Monomial.append(pt)
            sPow = fr381Mul(sPow, secret)
        }

        // Convert monomial SRS to Lagrange basis using INTT on the exponents.
        // L_i(s) = sum_j (omega^(-ij) / n) * s^j * G1
        // This is equivalent to performing an INTT of the monomial points.
        let g1Lagrange = monomialToLagrange381(g1Monomial)

        // G2 monomial: [G2, s*G2]
        let g2Proj = g2_381FromAffine(g2Gen)
        let sG2 = g2_381ScalarMul(g2Proj, sLimbs)
        let g2Aff0 = g2Gen
        let g2Aff1 = g2_381ToAffine(sG2)!

        let g1Affine = batchG1_381ToAffine(g1Lagrange)
        return EIP4844TrustedSetup(g1Lagrange: g1Affine, g2Monomial: [g2Aff0, g2Aff1])
    }
}

// MARK: - Blob to Polynomial Conversion

/// Convert a 131072-byte blob to 4096 BLS12-381 scalar field elements.
/// Each 32-byte chunk is interpreted as a big-endian integer, reduced mod BLS_MODULUS.
/// The spec requires that each element is already < BLS_MODULUS; returns nil if not.
public func blobToFieldElements(_ blob: [UInt8]) -> [Fr381]? {
    guard blob.count == BYTES_PER_BLOB else { return nil }
    var elements = [Fr381]()
    elements.reserveCapacity(FIELD_ELEMENTS_PER_BLOB)
    for i in 0..<FIELD_ELEMENTS_PER_BLOB {
        let start = i * BYTES_PER_FIELD_ELEMENT
        let chunk = Array(blob[start..<start + BYTES_PER_FIELD_ELEMENT])
        // Big-endian to 4x64-bit little-endian limbs
        let limbs = bigEndianBytesToLimbs64(chunk)
        // Validate: element must be < BLS_MODULUS
        if !isLessThanModulus381(limbs) { return nil }
        // Convert to Montgomery form
        let raw = Fr381.from64(limbs)
        let mont = fr381Mul(raw, Fr381.from64(Fr381.R2_MOD_R))
        elements.append(mont)
    }
    return elements
}

/// Convert 32 big-endian bytes to 4x UInt64 limbs (little-endian limb order).
private func bigEndianBytesToLimbs64(_ bytes: [UInt8]) -> [UInt64] {
    precondition(bytes.count == 32)
    var limbs = [UInt64](repeating: 0, count: 4)
    for i in 0..<4 {
        // Limb i holds bytes [24-8i..31-8i] in big-endian byte order
        let byteOffset = 24 - i * 8
        for j in 0..<8 {
            limbs[i] |= UInt64(bytes[byteOffset + j]) << UInt64((7 - j) * 8)
        }
    }
    return limbs
}

/// Check if 4x64 limbs represent a value < BLS12-381 Fr modulus.
private func isLessThanModulus381(_ limbs: [UInt64]) -> Bool {
    for i in stride(from: 3, through: 0, by: -1) {
        if limbs[i] < Fr381.P[i] { return true }
        if limbs[i] > Fr381.P[i] { return false }
    }
    return false  // Equal to modulus is not valid
}

/// Convert field elements back to blob bytes (big-endian, 32 bytes each).
public func fieldElementsToBlob(_ elements: [Fr381]) -> [UInt8] {
    precondition(elements.count == FIELD_ELEMENTS_PER_BLOB)
    var blob = [UInt8](repeating: 0, count: BYTES_PER_BLOB)
    for i in 0..<FIELD_ELEMENTS_PER_BLOB {
        let limbs = fr381ToInt(elements[i])
        let start = i * BYTES_PER_FIELD_ELEMENT
        for li in 0..<4 {
            let byteOffset = 24 - li * 8
            for j in 0..<8 {
                blob[start + byteOffset + j] = UInt8((limbs[li] >> UInt64((7 - j) * 8)) & 0xFF)
            }
        }
    }
    return blob
}

// MARK: - Bit-Reversal Permutation

/// Apply bit-reversal permutation to an array of length 2^logN.
/// Required for mapping between coefficient and evaluation representations.
public func bitReversalPermutation<T>(_ arr: inout [T], logN: Int) {
    let n = arr.count
    precondition(n == (1 << logN))
    for i in 0..<n {
        let j = reverseBits(i, logN: logN)
        if i < j {
            arr.swapAt(i, j)
        }
    }
}

/// Reverse the lower logN bits of an integer.
private func reverseBits(_ x: Int, logN: Int) -> Int {
    var result = 0
    var v = x
    for _ in 0..<logN {
        result = (result << 1) | (v & 1)
        v >>= 1
    }
    return result
}

// MARK: - Polynomial Evaluation (Lagrange Basis)

/// Evaluate a polynomial in Lagrange basis at an arbitrary point z.
/// Given evaluations [f(omega^0), f(omega^1), ..., f(omega^{n-1})] and a point z,
/// computes f(z) using the barycentric formula:
///   f(z) = (z^n - 1) / n * sum_{i=0}^{n-1} [ f(omega^i) * omega^i / (z - omega^i) ]
public func evaluatePolynomialInLagrangeBasis(_ evals: [Fr381], at z: Fr381) -> Fr381 {
    let n = evals.count
    precondition(n == FIELD_ELEMENTS_PER_BLOB)
    let logN = 12  // log2(4096)

    // Precompute omega (primitive n-th root of unity)
    let omega = fr381RootOfUnity(logN: logN)

    // Compute z^n
    var zn = z
    for _ in 0..<logN {
        zn = fr381Sqr(zn)
    }

    // If z^n == 1, then z is a root of unity; find which one and return directly
    let znMinusOne = fr381Sub(zn, Fr381.one)
    if fr381ToInt(znMinusOne) == [0, 0, 0, 0] {
        // z is a root of unity; find the index
        var omegaI = Fr381.one
        for i in 0..<n {
            if fr381ToInt(fr381Sub(z, omegaI)) == [0, 0, 0, 0] {
                return evals[i]
            }
            omegaI = fr381Mul(omegaI, omega)
        }
    }

    // Barycentric interpolation: f(z) = (z^n - 1)/n * sum_i f(omega^i) * omega^i / (z - omega^i)
    // Using the identity: L_i(z) = (z^n - 1) / (n * (z - omega^i)) * omega^i
    // Note: omega^(-i) in the standard formula, but since omega^n = 1, omega^(-i) = omega^(n-i).
    // The Ethereum spec uses: f(z) = (z^n - 1)/n * sum_i [ y_i / (z - omega^i) ]
    // with the roots in bit-reversed order. However, the Lagrange SRS already handles the ordering.

    let nInv = fr381Inverse(fr381FromInt(UInt64(n)))
    let factor = fr381Mul(znMinusOne, nInv)

    var sum = Fr381.zero
    var omegaI = Fr381.one  // omega^i
    for i in 0..<n {
        let diff = fr381Sub(z, omegaI)
        let diffInv = fr381Inverse(diff)
        let term = fr381Mul(fr381Mul(evals[i], omegaI), diffInv)
        sum = fr381Add(sum, term)
        omegaI = fr381Mul(omegaI, omega)
    }

    return fr381Mul(factor, sum)
}

// MARK: - Fiat-Shamir Challenge

/// Compute the Fiat-Shamir challenge point z for verify_blob_kzg_proof.
/// hash = SHA-256(FIAT_SHAMIR_PROTOCOL_DOMAIN || degree_separator || blob || commitment)
/// z = hash mod BLS_MODULUS (interpreted as big-endian).
public func computeBlobChallenge(blob: [UInt8], commitment: [UInt8]) -> Fr381 {
    precondition(blob.count == BYTES_PER_BLOB)
    precondition(commitment.count == 48)

    // Build input: domain || poly_degree (uint64 LE) || blob || commitment
    var input = [UInt8]()
    input.reserveCapacity(16 + 8 + BYTES_PER_BLOB + 48)
    input.append(contentsOf: Array(FIAT_SHAMIR_PROTOCOL_DOMAIN.utf8))
    // Polynomial degree = FIELD_ELEMENTS_PER_BLOB as uint64 little-endian
    var degree = UInt64(FIELD_ELEMENTS_PER_BLOB)
    withUnsafeBytes(of: &degree) { input.append(contentsOf: $0) }
    input.append(contentsOf: blob)
    input.append(contentsOf: commitment)

    let hash = sha256(input)
    return hashToFr381(hash)
}

/// Compute the random challenge r for batch verification.
/// hash = SHA-256(RANDOM_CHALLENGE_KZG_BATCH_DOMAIN || poly_degree || blob_count || ...commitments || ...proofs || ...blobs)
public func computeBatchChallenge(blobs: [[UInt8]], commitments: [[UInt8]], proofs: [[UInt8]]) -> Fr381 {
    let n = blobs.count
    var input = [UInt8]()
    input.append(contentsOf: Array(RANDOM_CHALLENGE_KZG_BATCH_DOMAIN.utf8))
    // Polynomial degree
    var degree = UInt64(FIELD_ELEMENTS_PER_BLOB)
    withUnsafeBytes(of: &degree) { input.append(contentsOf: $0) }
    // Number of blobs
    var count = UInt64(n)
    withUnsafeBytes(of: &count) { input.append(contentsOf: $0) }
    // Commitments
    for c in commitments { input.append(contentsOf: c) }
    // Proofs
    for p in proofs { input.append(contentsOf: p) }
    // Blobs
    for b in blobs { input.append(contentsOf: b) }

    let hash = sha256(input)
    return hashToFr381(hash)
}

/// Convert 32-byte hash to Fr381: interpret as big-endian integer, reduce mod r.
private func hashToFr381(_ hash: [UInt8]) -> Fr381 {
    precondition(hash.count == 32)
    let limbs = bigEndianBytesToLimbs64(hash)
    // Reduce mod r by converting to Montgomery form (handles reduction implicitly)
    // Since hash < 2^256 and r ~ 2^255, we need an explicit reduction.
    // Simple approach: if limbs >= r, subtract r.
    var reduced = limbs
    if !isLessThanModulus381(reduced) {
        // Subtract modulus
        var borrow: UInt64 = 0
        for i in 0..<4 {
            let (diff, overflow1) = reduced[i].subtractingReportingOverflow(Fr381.P[i])
            let (diff2, overflow2) = diff.subtractingReportingOverflow(borrow)
            reduced[i] = diff2
            borrow = (overflow1 ? 1 : 0) + (overflow2 ? 1 : 0)
        }
    }
    let raw = Fr381.from64(reduced)
    return fr381Mul(raw, Fr381.from64(Fr381.R2_MOD_R))
}

// SHA-256: uses the existing sha256() from SHA256Engine.swift

// MARK: - Monomial to Lagrange Basis Conversion (INTT on G1 points)

/// Convert monomial-basis G1 SRS to Lagrange-basis via inverse NTT.
/// Given [G1, s*G1, s^2*G1, ...], produces [L_0(s)*G1, L_1(s)*G1, ...].
private func monomialToLagrange381(_ monomial: [G1Projective381]) -> [G1Projective381] {
    let n = monomial.count
    let logN = Int(log2(Double(n)))
    precondition(1 << logN == n)

    var points = monomial

    // Bit-reversal permutation
    for i in 0..<n {
        let j = reverseBits(i, logN: logN)
        if i < j {
            points.swapAt(i, j)
        }
    }

    // Cooley-Tukey butterfly INTT using inverse roots of unity
    let omegaInv = fr381Inverse(fr381RootOfUnity(logN: logN))

    for s in 0..<logN {
        let m = 1 << (s + 1)
        let halfM = m / 2
        // twiddle = omegaInv^(n/m)
        var twiddle = Fr381.one
        let step = n / m
        var baseOmega = Fr381.one
        // Compute omegaInv^step
        for _ in 0..<step {
            baseOmega = fr381Mul(baseOmega, omegaInv)
        }

        for j in 0..<halfM {
            for k in stride(from: j, to: n, by: m) {
                let u = points[k]
                let v = g1_381ScalarMul(points[k + halfM], fr381ToInt(twiddle))
                points[k] = g1_381Add(u, v)
                points[k + halfM] = g1_381Add(u, g1_381Negate(v))
            }
            twiddle = fr381Mul(twiddle, baseOmega)
        }
    }

    // Multiply each element by 1/n
    let nInv = fr381Inverse(fr381FromInt(UInt64(n)))
    let nInvLimbs = fr381ToInt(nInv)
    for i in 0..<n {
        points[i] = g1_381ScalarMul(points[i], nInvLimbs)
    }

    return points
}

// MARK: - BlobKZGVerifier

/// Ethereum EIP-4844 blob KZG proof verifier.
/// Verifies that a KZG commitment opens to a blob's polynomial at a challenge point z,
/// using BLS12-381 pairings.
public class BlobKZGVerifier {
    public let trustedSetup: EIP4844TrustedSetup

    public init(trustedSetup: EIP4844TrustedSetup) {
        self.trustedSetup = trustedSetup
    }

    // MARK: - Blob to KZG Commitment

    /// Compute the KZG commitment to a blob.
    /// C = sum_{i=0}^{4095} f_i * [L_i(s)]_1
    /// where f_i are the blob's field elements and [L_i(s)]_1 are the Lagrange SRS points.
    public func blobToKZGCommitment(blob: [UInt8]) -> G1Projective381? {
        guard let elements = blobToFieldElements(blob) else { return nil }
        return commitToPolynomial(elements)
    }

    /// Commit to field elements using the Lagrange SRS.
    public func commitToPolynomial(_ elements: [Fr381]) -> G1Projective381 {
        let n = elements.count
        precondition(n <= FIELD_ELEMENTS_PER_BLOB)

        // MSM: sum_i scalar_i * point_i
        var flatScalars = [UInt32]()
        flatScalars.reserveCapacity(n * 8)
        for e in elements {
            let limbs = fr381ToInt(e)
            flatScalars.append(UInt32(limbs[0] & 0xFFFFFFFF))
            flatScalars.append(UInt32(limbs[0] >> 32))
            flatScalars.append(UInt32(limbs[1] & 0xFFFFFFFF))
            flatScalars.append(UInt32(limbs[1] >> 32))
            flatScalars.append(UInt32(limbs[2] & 0xFFFFFFFF))
            flatScalars.append(UInt32(limbs[2] >> 32))
            flatScalars.append(UInt32(limbs[3] & 0xFFFFFFFF))
            flatScalars.append(UInt32(limbs[3] >> 32))
        }

        let points = Array(trustedSetup.g1Lagrange.prefix(n))
        return g1_381PippengerMSMFlat(points: points, flatScalars: flatScalars)
    }

    // MARK: - Single Blob KZG Proof Verification

    /// Verify a KZG proof for a single blob.
    ///
    /// Checks that the commitment C opens to the blob's polynomial at challenge point z,
    /// where z is derived via Fiat-Shamir from the blob and commitment.
    ///
    /// The verification equation (pairing check):
    ///   e(C - [y]G1, G2) == e(pi, [s]G2 - [z]G2)
    /// equivalently:
    ///   e(C - [y]G1, G2) * e(-pi, [s]G2 - [z]G2) == 1
    ///
    /// - Parameters:
    ///   - blob: 131072-byte Ethereum blob
    ///   - commitment: 48-byte compressed G1 point (KZG commitment)
    ///   - proof: 48-byte compressed G1 point (KZG opening proof at z)
    /// - Returns: true if the proof verifies
    public func verifyBlobKZGProof(blob: [UInt8], commitment: [UInt8], proof: [UInt8]) -> Bool {
        guard blob.count == BYTES_PER_BLOB,
              commitment.count == 48,
              proof.count == 48 else {
            return false
        }

        // Deserialize commitment and proof
        guard let commitProj = bls12381G1Decompress(commitment),
              let commitAff = g1_381ToAffine(commitProj),
              let proofProj = bls12381G1Decompress(proof),
              let proofAff = g1_381ToAffine(proofProj) else {
            return false
        }

        // Convert blob to polynomial (field elements)
        guard let polyEvals = blobToFieldElements(blob) else {
            return false
        }

        // Compute Fiat-Shamir challenge z
        let z = computeBlobChallenge(blob: blob, commitment: commitment)

        // Evaluate polynomial at z (barycentric interpolation)
        let y = evaluatePolynomialInLagrangeBasis(polyEvals, at: z)

        // Verify the KZG opening: pairing check
        return verifyKZGProofInternal(commitment: commitAff, z: z, y: y, proof: proofAff)
    }

    /// Verify a KZG opening proof given the commitment, evaluation point, claimed value, and proof.
    ///
    /// Pairing check:  e(C - [y]G1, G2) == e(pi, [s]G2 - [z]G2)
    /// Rearranged:     e(C - [y]G1, G2) * e(-pi, [s]G2 - [z]G2) == 1
    public func verifyKZGProof(commitment: G1Affine381, z: Fr381, y: Fr381, proof: G1Affine381) -> Bool {
        verifyKZGProofInternal(commitment: commitment, z: z, y: y, proof: proof)
    }

    private func verifyKZGProofInternal(commitment: G1Affine381, z: Fr381, y: Fr381, proof: G1Affine381) -> Bool {
        let g1Gen = bls12381G1Generator()
        let g2Gen = trustedSetup.g2Monomial[0]
        let sG2 = trustedSetup.g2Monomial[1]

        // P1 = C - [y]*G1
        let yG1 = g1_381ScalarMul(g1_381FromAffine(g1Gen), fr381ToInt(y))
        let cMinusYG1 = g1_381Add(g1_381FromAffine(commitment), g1_381Negate(yG1))
        guard let p1Aff = g1_381ToAffine(cMinusYG1) else {
            // C - [y]*G1 is the identity: commitment equals [y]*G1
            // This means the polynomial is constant y, and any proof should be identity
            return g1_381IsIdentity(g1_381FromAffine(proof))
        }

        // Q2 = [s]*G2 - [z]*G2
        let zG2 = g2_381ScalarMul(g2_381FromAffine(g2Gen), fr381ToInt(z))
        let sMinusZG2 = g2_381Add(g2_381FromAffine(sG2), g2_381Negate(zG2))
        guard let q2Aff = g2_381ToAffine(sMinusZG2) else {
            return false
        }

        // Pairing check: e(P1, G2) * e(-proof, Q2) == 1
        let negProof = g1_381NegateAffine(proof)
        return bls12381PairingCheck([(p1Aff, g2Gen), (negProof, q2Aff)])
    }

    // MARK: - Batch Blob KZG Proof Verification

    /// Verify multiple blob KZG proofs via random linear combination.
    ///
    /// Uses a random challenge r to combine N individual verification equations
    /// into a single pairing check, reducing N pairings to 2.
    ///
    /// For each blob i:
    ///   - z_i = challenge(blob_i, commitment_i)
    ///   - y_i = eval(poly_i, z_i)
    ///
    /// Combined check with random r:
    ///   e( sum_i r^i * (C_i - [y_i]*G1 + [z_i]*pi_i), G2 ) == e( sum_i r^i * pi_i, [s]*G2 )
    ///
    /// - Parameters:
    ///   - blobs: array of 131072-byte blobs
    ///   - commitments: array of 48-byte compressed G1 commitments
    ///   - proofs: array of 48-byte compressed G1 proofs
    /// - Returns: true if all proofs verify
    public func verifyBlobKZGProofBatch(blobs: [[UInt8]], commitments: [[UInt8]], proofs: [[UInt8]]) -> Bool {
        let n = blobs.count
        guard n == commitments.count, n == proofs.count else { return false }
        if n == 0 { return true }
        if n == 1 {
            return verifyBlobKZGProof(blob: blobs[0], commitment: commitments[0], proof: proofs[0])
        }

        // Deserialize all commitments and proofs
        var commitAffs = [G1Affine381]()
        commitAffs.reserveCapacity(n)
        var proofAffs = [G1Affine381]()
        proofAffs.reserveCapacity(n)
        for i in 0..<n {
            guard let cProj = bls12381G1Decompress(commitments[i]),
                  let cAff = g1_381ToAffine(cProj),
                  let pProj = bls12381G1Decompress(proofs[i]),
                  let pAff = g1_381ToAffine(pProj) else {
                return false
            }
            commitAffs.append(cAff)
            proofAffs.append(pAff)
        }

        // Compute individual challenges and evaluations
        var zValues = [Fr381]()
        var yValues = [Fr381]()
        zValues.reserveCapacity(n)
        yValues.reserveCapacity(n)
        for i in 0..<n {
            guard let polyEvals = blobToFieldElements(blobs[i]) else { return false }
            let z = computeBlobChallenge(blob: blobs[i], commitment: commitments[i])
            let y = evaluatePolynomialInLagrangeBasis(polyEvals, at: z)
            zValues.append(z)
            yValues.append(y)
        }

        // Compute random challenge r for batch combination
        let r = computeBatchChallenge(blobs: blobs, commitments: commitments, proofs: proofs)

        // Build combined points using random linear combination
        let g1Gen = bls12381G1Generator()
        let g2Gen = trustedSetup.g2Monomial[0]
        let sG2 = trustedSetup.g2Monomial[1]

        // LHS = sum_i r^i * (C_i - [y_i]*G1 + [z_i]*pi_i)
        // RHS_proof = sum_i r^i * pi_i
        var lhsAccum = g1_381Identity()
        var rhsAccum = g1_381Identity()
        var rPow = Fr381.one

        for i in 0..<n {
            let rPowLimbs = fr381ToInt(rPow)

            // C_i * r^i
            let rC = g1_381ScalarMul(g1_381FromAffine(commitAffs[i]), rPowLimbs)

            // -[y_i * r^i]*G1
            let rY = fr381Mul(rPow, yValues[i])
            let rYG1 = g1_381ScalarMul(g1_381FromAffine(g1Gen), fr381ToInt(rY))

            // [z_i * r^i]*pi_i
            let rZ = fr381Mul(rPow, zValues[i])
            let rZPi = g1_381ScalarMul(g1_381FromAffine(proofAffs[i]), fr381ToInt(rZ))

            // LHS += r^i * C_i - r^i * y_i * G1 + r^i * z_i * pi_i
            lhsAccum = g1_381Add(lhsAccum, rC)
            lhsAccum = g1_381Add(lhsAccum, g1_381Negate(rYG1))
            lhsAccum = g1_381Add(lhsAccum, rZPi)

            // RHS += r^i * pi_i
            let rPi = g1_381ScalarMul(g1_381FromAffine(proofAffs[i]), rPowLimbs)
            rhsAccum = g1_381Add(rhsAccum, rPi)

            rPow = fr381Mul(rPow, r)
        }

        // Pairing check: e(LHS, G2) == e(RHS, [s]*G2)
        // => e(LHS, G2) * e(-RHS, [s]*G2) == 1
        guard let lhsAff = g1_381ToAffine(lhsAccum) else {
            // LHS is identity => check RHS is also identity
            return g1_381IsIdentity(rhsAccum)
        }
        guard let rhsAff = g1_381ToAffine(rhsAccum) else {
            // RHS is identity => check e(LHS, G2) == 1 (only if LHS is also identity)
            return false
        }

        let negRhsAff = g1_381NegateAffine(rhsAff)
        return bls12381PairingCheck([(lhsAff, g2Gen), (negRhsAff, sG2)])
    }
}
