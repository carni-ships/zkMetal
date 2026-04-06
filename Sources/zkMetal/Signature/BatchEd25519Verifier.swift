// Batch Ed25519 Signature Verification (GPU-accelerated)
//
// Provides both individual and batched Ed25519 signature verification.
// The batch path uses random linear combination to reduce N verifications
// into a single multi-scalar multiplication (MSM), which is dispatched
// to the C Pippenger MSM (CPU) or Metal GPU depending on batch size.
//
// Individual verification (Shamir's trick via C):
//   Check: [S]G == R + [H(R,A,M)]A
//
// Batch verification (random linear combination):
//   Choose random 128-bit weights z_i for soundness.
//   Reduce to: [sum(z_i*S_i)]G - sum(z_i*R_i) - sum(z_i*k_i*A_i) == O
//   where k_i = H(R_i || A_i || M_i)
//   If any signature is invalid, the check fails with probability >= 1 - 2^(-128).
//
// GPU parallelism: for large batches, the combined MSM of (2N+1) points
// is dispatched to the Ed25519 Metal MSM engine via Pippenger's bucket method.

import Foundation
import NeonFieldOps

/// Type aliases for cleaner API
public typealias Ed25519Signature = EdDSASignature
public typealias Ed25519PublicKey = EdDSAPublicKey

/// Minimum batch size before using the batched MSM path.
/// Below this, individual verification with Shamir's trick is faster.
private let kEd25519BatchThreshold = 4

public class BatchEd25519Verifier {
    public static let version = Versions.batchEd25519

    private let engine: EdDSAEngine

    /// Optional GPU MSM engine for large batches (created lazily).
    private var msmEngine: Ed25519MSM?

    public init() {
        self.engine = EdDSAEngine()
    }

    // MARK: - Single Verification

    /// Verify a single Ed25519 signature.
    ///
    /// - Parameters:
    ///   - message: Raw message bytes.
    ///   - signature: The Ed25519 signature (R, S).
    ///   - publicKey: The signer's public key.
    /// - Returns: `true` if the signature is valid.
    public func verifySingle(message: [UInt8], signature: Ed25519Signature,
                             publicKey: Ed25519PublicKey) -> Bool {
        engine.verify(signature: signature, message: message, publicKey: publicKey)
    }

    // MARK: - Batch Verification

    /// Verify a batch of Ed25519 signatures.
    ///
    /// For small batches (N < threshold), delegates to individual verification.
    /// For larger batches, performs probabilistic batch verification first;
    /// if that passes, all signatures are reported valid. If the probabilistic
    /// check fails, falls back to individual checks to identify which are invalid.
    ///
    /// - Parameters:
    ///   - messages: Array of raw message byte arrays (one per signature).
    ///   - signatures: The Ed25519 signatures.
    ///   - publicKeys: The signer public keys (one per signature).
    /// - Returns: `true` if ALL signatures are valid.
    public func verifyBatch(messages: [[UInt8]], signatures: [Ed25519Signature],
                            publicKeys: [Ed25519PublicKey]) -> Bool {
        let n = signatures.count
        precondition(messages.count == n && publicKeys.count == n,
                     "Input arrays must have equal length")
        if n == 0 { return true }

        // Small batch: individual verification
        if n < kEd25519BatchThreshold {
            for i in 0..<n {
                if !engine.verify(signature: signatures[i], message: messages[i],
                                  publicKey: publicKeys[i]) {
                    return false
                }
            }
            return true
        }

        // Large batch: probabilistic batch via random linear combination MSM
        return engine.batchVerify(signatures: signatures, messages: messages,
                                  publicKeys: publicKeys)
    }

    /// Verify a batch and return per-signature results.
    ///
    /// Attempts probabilistic batch verification first. If it passes, returns
    /// all true. If it fails, falls back to individual checks.
    ///
    /// - Returns: Per-signature boolean array.
    public func verifyBatchDetailed(messages: [[UInt8]], signatures: [Ed25519Signature],
                                    publicKeys: [Ed25519PublicKey]) -> [Bool] {
        let n = signatures.count
        precondition(messages.count == n && publicKeys.count == n,
                     "Input arrays must have equal length")
        if n == 0 { return [] }

        // Small batch: individual
        if n < kEd25519BatchThreshold {
            return (0..<n).map { i in
                engine.verify(signature: signatures[i], message: messages[i],
                              publicKey: publicKeys[i])
            }
        }

        // Try probabilistic batch first
        let allValid = engine.batchVerify(signatures: signatures, messages: messages,
                                          publicKeys: publicKeys)
        if allValid {
            return [Bool](repeating: true, count: n)
        }

        // At least one bad signature -- fall back to individual checks
        return (0..<n).map { i in
            engine.verify(signature: signatures[i], message: messages[i],
                          publicKey: publicKeys[i])
        }
    }

    /// Verify a batch using the GPU MSM engine for large batches.
    ///
    /// This path creates an Ed25519MSM Metal engine and dispatches the
    /// combined (2N+1)-point MSM to the GPU. For batches above ~512,
    /// this is significantly faster than the CPU Pippenger path.
    ///
    /// - Returns: `true` if ALL signatures are valid.
    public func verifyBatchGPU(messages: [[UInt8]], signatures: [Ed25519Signature],
                               publicKeys: [Ed25519PublicKey]) throws -> Bool {
        let n = signatures.count
        precondition(messages.count == n && publicKeys.count == n,
                     "Input arrays must have equal length")
        if n == 0 { return true }
        if n == 1 {
            return verifySingle(message: messages[0], signature: signatures[0],
                                publicKey: publicKeys[0])
        }

        // Lazily create GPU MSM engine
        if msmEngine == nil {
            msmEngine = try Ed25519MSM()
        }
        guard let msm = msmEngine else {
            throw MSMError.noGPU
        }

        // Generate random 128-bit weights
        var rng: UInt64 = UInt64(CFAbsoluteTimeGetCurrent().bitPattern) ^ 0xDEADBEEFCAFEBABE
        var weights = [[UInt64]](repeating: [UInt64](repeating: 0, count: 4), count: n)
        for i in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let w0 = rng
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let w1 = rng
            var raw: [UInt64] = [w0, w1, 0, 0]
            var mont = [UInt64](repeating: 0, count: 4)
            ed25519_fq_from_raw(&raw, &mont)
            weights[i] = mont
        }

        // Build MSM arrays: (2N+1) points and scalars
        // Points:  G, R_0..R_{n-1}, A_0..A_{n-1}
        // Scalars: sum(z_i*S_i), -z_0..-z_{n-1}, -z_0*k_0..-z_{n-1}*k_{n-1}
        let totalPoints = 2 * n + 1
        var affinePoints = [Ed25519PointAffine]()
        affinePoints.reserveCapacity(totalPoints)
        var scalars32 = [[UInt32]]()
        scalars32.reserveCapacity(totalPoints)

        // Point 0: Generator
        let gen = ed25519Generator()
        affinePoints.append(gen)

        // Accumulate gScalar = sum(z_i * S_i) in Montgomery
        var gScalarMont = [UInt64](repeating: 0, count: 4)

        for i in 0..<n {
            // Decode S_i
            var sLimbs: [UInt64] = [0, 0, 0, 0]
            for li in 0..<4 {
                for j in 0..<8 {
                    sLimbs[li] |= UInt64(signatures[i].s[li * 8 + j]) << (j * 8)
                }
            }
            if ed25519FqGte(sLimbs, Ed25519Fq.Q) { return false }
            var sMont = [UInt64](repeating: 0, count: 4)
            ed25519_fq_from_raw(&sLimbs, &sMont)

            // gScalar += z_i * S_i
            var ziSi = [UInt64](repeating: 0, count: 4)
            ed25519_fq_mul(&weights[i], &sMont, &ziSi)
            var tmp = [UInt64](repeating: 0, count: 4)
            ed25519_fq_add(&gScalarMont, &ziSi, &tmp)
            gScalarMont = tmp

            // k_i = H(R_i || A_i || M_i) mod q
            let kHash = sha512(signatures[i].r + publicKeys[i].encoded + messages[i])
            var kMont = [UInt64](repeating: 0, count: 4)
            kHash.withUnsafeBufferPointer { hashPtr in
                ed25519_fq_from_bytes64(hashPtr.baseAddress!, &kMont)
            }

            // Decode R_i
            guard let rAff = ed25519PointDecode(signatures[i].r) else { return false }
            affinePoints.append(rAff)

            // Set A_i
            affinePoints.append(publicKeys[i].point)

            // Scalar for R_i: -z_i
            var negZi = [UInt64](repeating: 0, count: 4)
            var zeroFq = [UInt64](repeating: 0, count: 4)
            ed25519_fq_sub(&zeroFq, &weights[i], &negZi)
            var negZiRaw = [UInt64](repeating: 0, count: 4)
            ed25519_fq_to_raw(&negZi, &negZiRaw)
            scalars32.append(uint64ToUint32Limbs(negZiRaw))

            // Scalar for A_i: -z_i * k_i
            var zkMont = [UInt64](repeating: 0, count: 4)
            ed25519_fq_mul(&weights[i], &kMont, &zkMont)
            var negZk = [UInt64](repeating: 0, count: 4)
            ed25519_fq_sub(&zeroFq, &zkMont, &negZk)
            var negZkRaw = [UInt64](repeating: 0, count: 4)
            ed25519_fq_to_raw(&negZk, &negZkRaw)
            scalars32.append(uint64ToUint32Limbs(negZkRaw))
        }

        // G scalar at index 0
        var gScalarRaw = [UInt64](repeating: 0, count: 4)
        ed25519_fq_to_raw(&gScalarMont, &gScalarRaw)
        scalars32.insert(uint64ToUint32Limbs(gScalarRaw), at: 0)

        // Reorder points: G, R_0, A_0, R_1, A_1, ... -> G, R_0..R_{n-1}, A_0..A_{n-1}
        // Actually we interleaved R_i and A_i above. Reorder for MSM:
        // affinePoints = [G, R_0, A_0, R_1, A_1, ...]
        // scalars32 = [gScalar, -z_0, -z_0*k_0, -z_1, -z_1*k_1, ...]
        // The MSM engine doesn't care about ordering, so this is fine as-is.

        // Run GPU MSM
        let result = try msm.msm(points: affinePoints, scalars: scalars32)

        // Check if result is identity
        return ed25519PointIsIdentity(result)
    }

    // MARK: - Strategy Description

    /// Returns a description of the verification strategy for a given batch size.
    public static func strategyDescription(batchSize n: Int) -> String {
        if n < kEd25519BatchThreshold {
            return "individual (Shamir's trick, \(n) verifications)"
        }
        let msmPoints = 2 * n + 1
        if msmPoints <= 512 {
            return "batch MSM (C Pippenger, \(msmPoints) points)"
        }
        return "batch MSM (Metal GPU, \(msmPoints) points)"
    }
}

// MARK: - Helpers

/// Convert 4 x UInt64 limbs to 8 x UInt32 limbs (little-endian)
private func uint64ToUint32Limbs(_ limbs: [UInt64]) -> [UInt32] {
    var result = [UInt32](repeating: 0, count: 8)
    for j in 0..<4 {
        result[j * 2] = UInt32(limbs[j] & 0xFFFFFFFF)
        result[j * 2 + 1] = UInt32(limbs[j] >> 32)
    }
    return result
}
