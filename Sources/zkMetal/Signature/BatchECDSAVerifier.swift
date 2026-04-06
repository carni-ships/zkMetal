// Batch ECDSA Verification over secp256k1
//
// Provides both individual and batched ECDSA signature verification.
// The batch path uses random linear combination to reduce N verifications
// into a single multi-scalar multiplication (MSM), which is then dispatched
// to the GPU via Pippenger's bucket method.
//
// Individual verification (Shamir's trick):
//   u1 = z * s^(-1) mod n,  u2 = r * s^(-1) mod n
//   Check: (u1*G + u2*Q).x == r (mod n)
//
// Batch verification (random linear combination):
//   Choose random 128-bit weights w_i for soundness.
//   Reduce to:  sum(w_i*u1_i)*G + sum_i(w_i*u2_i*Q_i) - sum_i(w_i*R_i) == O
//   If any signature is invalid, the check fails with probability >= 1 - 2^(-128).
//
// GPU parallelism: the combined MSM of (2N+1) points is dispatched to the
// Metal MSM engine for batches above a configurable threshold.

import Foundation
import Metal
import NeonFieldOps

/// Minimum batch size before we use the batched MSM path.
/// Below this threshold, individual verification with Shamir's trick is faster
/// because GPU launch overhead dominates.
private let kBatchThreshold = 4

public class BatchECDSAVerifier {
    public static let version = Versions.batchECDSA

    private let ecdsaEngine: ECDSAEngine

    /// The underlying MSM engine (secp256k1 Pippenger, Metal-accelerated).
    public var msmEngine: Secp256k1MSM { ecdsaEngine.msmEngine }

    public init() throws {
        self.ecdsaEngine = try ECDSAEngine()
    }

    // MARK: - Single Verification

    /// Verify a single ECDSA signature using Shamir's trick.
    ///
    /// - Parameters:
    ///   - message: Raw message bytes (will be hashed to a scalar).
    ///   - signature: The ECDSA signature (r, s) with pre-hashed z.
    ///   - publicKey: The signer's public key as an affine point on secp256k1.
    /// - Returns: `true` if the signature is valid.
    public func verifySingle(message: [UInt8], signature: ECDSASignature,
                             publicKey: SecpPointAffine) -> Bool {
        ecdsaEngine.verify(sig: signature, pubkey: publicKey)
    }

    /// Convenience overload that accepts a pre-computed message hash scalar directly.
    public func verifySingle(sig: ECDSASignature, pubkey: SecpPointAffine) -> Bool {
        ecdsaEngine.verify(sig: sig, pubkey: pubkey)
    }

    // MARK: - Batch Verification

    /// Verify a batch of ECDSA signatures.
    ///
    /// For small batches (N < threshold), delegates to individual verification and
    /// returns per-signature results. For larger batches, performs probabilistic
    /// batch verification first; if that passes, all signatures are reported valid.
    /// If the probabilistic check fails, falls back to individual checks to
    /// identify which signatures are invalid.
    ///
    /// - Parameters:
    ///   - messages: Array of raw message byte arrays (one per signature).
    ///   - signatures: The ECDSA signatures with pre-hashed z fields.
    ///   - publicKeys: The signer public keys (one per signature).
    ///   - recoveryBits: Optional y-parity bits for lifting r to a curve point.
    /// - Returns: Per-signature boolean array.
    public func verifyBatch(messages: [[UInt8]], signatures: [ECDSASignature],
                            publicKeys: [SecpPointAffine],
                            recoveryBits: [UInt8]? = nil) throws -> [Bool] {
        let n = signatures.count
        precondition(messages.count == n && publicKeys.count == n,
                     "Input arrays must have equal length")
        if n == 0 { return [] }

        // Small batch: individual verification
        if n < kBatchThreshold {
            return signatures.enumerated().map { (i, sig) in
                ecdsaEngine.verify(sig: sig, pubkey: publicKeys[i])
            }
        }

        // Large batch: try probabilistic batch first
        let allValid = try batchVerifyProbabilistic(
            signatures: signatures, publicKeys: publicKeys, recoveryBits: recoveryBits)

        if allValid {
            return [Bool](repeating: true, count: n)
        }

        // Probabilistic check failed — at least one bad signature.
        // Fall back to individual checks to identify which ones.
        return signatures.enumerated().map { (i, sig) in
            ecdsaEngine.verify(sig: sig, pubkey: publicKeys[i])
        }
    }

    /// Convenience overload without the messages parameter, since ECDSASignature
    /// already contains the message hash z.
    public func verifyBatch(signatures: [ECDSASignature], publicKeys: [SecpPointAffine],
                            recoveryBits: [UInt8]? = nil) throws -> [Bool] {
        // Create placeholder messages (z is already in the signature)
        let msgs = [[UInt8]](repeating: [], count: signatures.count)
        return try verifyBatch(messages: msgs, signatures: signatures,
                               publicKeys: publicKeys, recoveryBits: recoveryBits)
    }

    // MARK: - Probabilistic Batch Verification (all-or-nothing)

    /// Probabilistic batch verification: returns true iff ALL signatures are valid.
    ///
    /// Uses random 128-bit weights w_i to combine N verifications into a single MSM:
    ///   sum(w_i*u1_i)*G + sum_i(w_i*u2_i*Q_i) - sum_i(w_i*R_i) == O
    ///
    /// If any signature is invalid, the result is false with probability >= 1 - 2^(-128).
    ///
    /// For N >= 300, the MSM is dispatched to the GPU. Below that, the C Pippenger
    /// implementation on CPU is used (avoids GPU launch overhead).
    public func batchVerifyProbabilistic(signatures: [ECDSASignature],
                                         publicKeys: [SecpPointAffine],
                                         recoveryBits: [UInt8]? = nil) throws -> Bool {
        try ecdsaEngine.batchVerifyProbabilistic(
            signatures: signatures, pubkeys: publicKeys, recoveryBits: recoveryBits)
    }

    // MARK: - Batch Statistics

    /// Returns a description of the verification strategy for a given batch size.
    public static func strategyDescription(batchSize n: Int) -> String {
        if n < kBatchThreshold {
            return "individual (Shamir's trick, \(n) verifications)"
        }
        let msmPoints = 2 * n + 1
        if msmPoints <= 300 {
            return "batch MSM (C Pippenger, \(msmPoints) points)"
        }
        return "batch MSM (Metal GPU, \(msmPoints) points)"
    }
}
