// GPU-Accelerated ECDSA Batch Signature Verification Engine
//
// Batch ECDSA verification over secp256k1 using random linear combination
// and GPU-accelerated multi-scalar multiplication (MSM).
//
// The engine provides:
//   - Single signature verification (Shamir's trick)
//   - Batch verification with Fiat-Shamir derived challenges
//   - GPU-accelerated MSM for the batch check
//   - Signature malleability detection (high-s values)
//   - Message hash-to-scalar conversion (keccak256-based)
//
// Batch verification algorithm:
//   For N signatures (z_i, r_i, s_i, Q_i), choose deterministic weights w_i
//   via Fiat-Shamir transcript. Reduce N verifications to one MSM:
//     sum(w_i * u1_i) * G + sum_i(w_i * u2_i * Q_i) - sum_i(w_i * R_i) == O
//   If any signature is invalid, the check fails with overwhelming probability.
//
// Fiat-Shamir challenge derivation:
//   transcript = keccak256("ecdsa-batch-verify" || n || r_0 || s_0 || z_0 || Q_0 || ...)
//   w_i = keccak256(transcript || i) truncated to 128 bits
//   This makes the batch check deterministic and non-interactive.
//
// Performance characteristics:
//   - Single verify: ~0.3ms (Shamir's trick via C CIOS)
//   - Batch N=16: ~2ms (CPU Pippenger, 33-point MSM)
//   - Batch N=1000: dispatched to Metal GPU MSM (Pippenger bucket method)
//   - Batch inverse: O(N) field ops via Montgomery's trick
//   - Fiat-Shamir overhead: O(N) keccak256 hashes for challenge derivation
//
// Security:
//   - 128-bit challenge weights provide 2^(-128) soundness error
//   - Fiat-Shamir transcript binds all signature data (r, s, z, Q)
//   - Optional BIP-62 / EIP-2 malleability enforcement (low-s check)

import Foundation
import Metal
import NeonFieldOps

// MARK: - Malleability Detection

/// The half-order of secp256k1: n/2.
/// Signatures with s > n/2 are considered malleable (BIP-62 / EIP-2).
/// A valid low-s signature has s in [1, halfN].
private let kSecpHalfN: [UInt64] = {
    // n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    // halfN = (n - 1) / 2 + 1 = (n + 1) / 2
    // Actually: halfN = n >> 1
    // n >> 1 = 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A0
    return [
        0xDFE92F46681B20A0,
        0x5D576E7357A4501D,
        0xFFFFFFFFFFFFFFFF,
        0x7FFFFFFFFFFFFFFF
    ]
}()

/// Result of a malleability check on a batch of signatures.
public struct MalleabilityReport {
    /// Indices of signatures with high-s values (s > n/2).
    public let malleableIndices: [Int]

    /// True if no signatures are malleable.
    public var allCanonical: Bool { malleableIndices.isEmpty }

    /// Number of malleable signatures found.
    public var malleableCount: Int { malleableIndices.count }

    public init(malleableIndices: [Int]) {
        self.malleableIndices = malleableIndices
    }
}

// MARK: - Batch Verify Configuration

/// Configuration for the GPU ECDSA batch verification engine.
public struct GPUECDSABatchConfig {
    /// Minimum batch size for GPU MSM dispatch.
    /// Below this, CPU Pippenger is used (avoids GPU launch overhead).
    public var gpuMSMThreshold: Int

    /// Minimum batch size for batch (RLC) path vs individual verification.
    /// Below this, individual Shamir's trick verification is used.
    public var batchThreshold: Int

    /// Whether to enforce low-s (canonical) signatures.
    /// If true, signatures with s > n/2 are rejected as invalid.
    public var enforceLowS: Bool

    /// Whether to use Fiat-Shamir deterministic challenges (true)
    /// or cryptographic random challenges (false).
    public var useFiatShamirChallenges: Bool

    public init(gpuMSMThreshold: Int = 300,
                batchThreshold: Int = 4,
                enforceLowS: Bool = false,
                useFiatShamirChallenges: Bool = true) {
        self.gpuMSMThreshold = gpuMSMThreshold
        self.batchThreshold = batchThreshold
        self.enforceLowS = enforceLowS
        self.useFiatShamirChallenges = useFiatShamirChallenges
    }

    /// Default configuration.
    public static let `default` = GPUECDSABatchConfig()

    /// Strict configuration: enforces low-s, uses Fiat-Shamir.
    public static let strict = GPUECDSABatchConfig(enforceLowS: true,
                                                    useFiatShamirChallenges: true)
}

// MARK: - Batch Verify Result

/// Detailed result from a batch verification operation.
public struct GPUECDSABatchResult {
    /// Per-signature verification results.
    public let results: [Bool]

    /// Whether the probabilistic batch check passed (nil if not used).
    public let batchCheckPassed: Bool?

    /// Malleability report (nil if not checked).
    public let malleabilityReport: MalleabilityReport?

    /// Number of valid signatures.
    public var validCount: Int { results.filter { $0 }.count }

    /// Number of invalid signatures.
    public var invalidCount: Int { results.filter { !$0 }.count }

    /// True if all signatures are valid.
    public var allValid: Bool { results.allSatisfy { $0 } }

    /// Strategy used for verification.
    public let strategy: String

    public init(results: [Bool], batchCheckPassed: Bool?,
                malleabilityReport: MalleabilityReport?, strategy: String) {
        self.results = results
        self.batchCheckPassed = batchCheckPassed
        self.malleabilityReport = malleabilityReport
        self.strategy = strategy
    }
}

// MARK: - GPU ECDSA Batch Verify Engine

/// GPU-accelerated ECDSA batch signature verification engine for secp256k1.
///
/// Uses random linear combination to reduce N signature verifications into
/// a single multi-scalar multiplication, dispatched to Metal GPU for large
/// batches. Challenges are derived via Fiat-Shamir transcript for
/// deterministic, non-interactive batch proofs.
public class GPUECDSABatchVerifyEngine {
    public static let version = Versions.batchECDSA

    /// The underlying ECDSA engine (provides Shamir's trick single verify).
    private let ecdsaEngine: ECDSAEngine

    /// Configuration for batch verification behavior.
    public var config: GPUECDSABatchConfig

    /// Access to the underlying MSM engine for diagnostics.
    public var msmEngine: Secp256k1MSM { ecdsaEngine.msmEngine }

    /// Statistics: total signatures verified.
    public private(set) var totalVerified: Int = 0

    /// Statistics: total batch operations performed.
    public private(set) var totalBatchOps: Int = 0

    public init(config: GPUECDSABatchConfig = .default) throws {
        self.ecdsaEngine = try ECDSAEngine()
        self.config = config
    }

    // MARK: - Message Hash-to-Scalar Conversion

    /// Convert a raw message to a secp256k1 scalar field element.
    /// Uses keccak256 hash, then reduces modulo n (the secp256k1 group order).
    /// This follows the Ethereum ecrecover convention.
    ///
    /// - Parameter message: Raw message bytes.
    /// - Returns: Scalar in Montgomery form suitable for ECDSA verification.
    public func messageToScalar(_ message: [UInt8]) -> SecpFr {
        let hash = keccak256(message)
        return hashBytesToScalar(hash)
    }

    /// Convert a 32-byte hash to a secp256k1 scalar field element.
    /// The hash is interpreted as a big-endian 256-bit integer and reduced mod n.
    ///
    /// - Parameter hash: 32-byte hash value.
    /// - Returns: Scalar in Montgomery form.
    public func hashBytesToScalar(_ hash: [UInt8]) -> SecpFr {
        precondition(hash.count == 32, "Hash must be 32 bytes")
        // Interpret as big-endian 256-bit integer -> 4x64 LE limbs
        var limbs = [UInt64](repeating: 0, count: 4)
        for i in 0..<4 {
            let offset = 24 - i * 8  // big-endian: first byte is MSB
            for j in 0..<8 {
                limbs[i] |= UInt64(hash[offset + 7 - j]) << (j * 8)
            }
        }
        // Reduce mod n if necessary
        if gte256(limbs, SecpFr.N) {
            (limbs, _) = sub256(limbs, SecpFr.N)
        }
        // Convert to Montgomery form
        return secpFrFromRaw(limbs)
    }

    // MARK: - Malleability Detection

    /// Check a batch of signatures for malleability (high-s values).
    /// A signature is malleable if s > n/2, meaning (r, n-s) is also valid.
    ///
    /// - Parameter signatures: Array of ECDSA signatures to check.
    /// - Returns: Report indicating which signatures are malleable.
    public func checkMalleability(_ signatures: [ECDSASignature]) -> MalleabilityReport {
        var malleable = [Int]()
        for (i, sig) in signatures.enumerated() {
            let sInt = secpFrToInt(sig.s)
            if isHighS(sInt) {
                malleable.append(i)
            }
        }
        return MalleabilityReport(malleableIndices: malleable)
    }

    /// Check if a single s value is "high" (s > n/2).
    private func isHighS(_ sLimbs: [UInt64]) -> Bool {
        // s > halfN means the signature is malleable
        for i in stride(from: 3, through: 0, by: -1) {
            if sLimbs[i] > kSecpHalfN[i] { return true }
            if sLimbs[i] < kSecpHalfN[i] { return false }
        }
        return false  // s == halfN is not high
    }

    /// Normalize a malleable signature by replacing s with n - s.
    /// This converts a high-s signature to its canonical low-s form.
    ///
    /// - Parameter sig: The original signature.
    /// - Returns: Canonical signature with s <= n/2.
    public func normalizeSignature(_ sig: ECDSASignature) -> ECDSASignature {
        let sInt = secpFrToInt(sig.s)
        if isHighS(sInt) {
            let normalizedS = secpFrNeg(sig.s)
            return ECDSASignature(r: sig.r, s: normalizedS, z: sig.z)
        }
        return sig
    }

    // MARK: - Single Verification

    /// Verify a single ECDSA signature using Shamir's trick.
    ///
    /// - Parameters:
    ///   - signature: The ECDSA signature (r, s, z) in Montgomery form.
    ///   - publicKey: The signer's public key as an affine point.
    /// - Returns: `true` if the signature is valid.
    public func verifySingle(signature: ECDSASignature,
                             publicKey: SecpPointAffine) -> Bool {
        if config.enforceLowS {
            let sInt = secpFrToInt(signature.s)
            if isHighS(sInt) { return false }
        }
        totalVerified += 1
        return ecdsaEngine.verify(sig: signature, pubkey: publicKey)
    }

    /// Verify a single signature with message hash conversion.
    ///
    /// - Parameters:
    ///   - message: Raw message bytes (will be hashed via keccak256).
    ///   - r: Signature r value in Montgomery form.
    ///   - s: Signature s value in Montgomery form.
    ///   - publicKey: The signer's public key.
    /// - Returns: `true` if the signature is valid.
    public func verifySingleMessage(message: [UInt8], r: SecpFr, s: SecpFr,
                                    publicKey: SecpPointAffine) -> Bool {
        let z = messageToScalar(message)
        let sig = ECDSASignature(r: r, s: s, z: z)
        return verifySingle(signature: sig, publicKey: publicKey)
    }

    // MARK: - Batch Verification

    /// Full batch verification with detailed results.
    ///
    /// Strategy selection:
    ///   - N < batchThreshold: individual Shamir's trick
    ///   - N >= batchThreshold: probabilistic batch check first;
    ///     if it fails, fall back to individual for per-signature results.
    ///
    /// - Parameters:
    ///   - signatures: Array of ECDSA signatures.
    ///   - publicKeys: Corresponding public keys.
    ///   - recoveryBits: Optional y-parity bits for lifting r to curve point.
    /// - Returns: Detailed batch verification result.
    public func verifyBatch(signatures: [ECDSASignature],
                            publicKeys: [SecpPointAffine],
                            recoveryBits: [UInt8]? = nil) throws -> GPUECDSABatchResult {
        let n = signatures.count
        precondition(publicKeys.count == n, "signatures and publicKeys must have equal length")
        totalBatchOps += 1

        if n == 0 {
            return GPUECDSABatchResult(results: [], batchCheckPassed: nil,
                                       malleabilityReport: nil, strategy: "empty")
        }

        // Malleability check if configured
        var malleabilityReport: MalleabilityReport? = nil
        if config.enforceLowS {
            malleabilityReport = checkMalleability(signatures)
            if !malleabilityReport!.allCanonical {
                // Mark malleable signatures as invalid, verify the rest individually
                var results = [Bool](repeating: false, count: n)
                let malleableSet = Set(malleabilityReport!.malleableIndices)
                for i in 0..<n {
                    if !malleableSet.contains(i) {
                        results[i] = ecdsaEngine.verify(sig: signatures[i],
                                                        pubkey: publicKeys[i])
                    }
                }
                totalVerified += n
                return GPUECDSABatchResult(
                    results: results, batchCheckPassed: false,
                    malleabilityReport: malleabilityReport,
                    strategy: "individual (malleable signatures detected)")
            }
        }

        // Small batch: individual verification
        if n < config.batchThreshold {
            let results = (0..<n).map { i in
                ecdsaEngine.verify(sig: signatures[i], pubkey: publicKeys[i])
            }
            totalVerified += n
            return GPUECDSABatchResult(
                results: results, batchCheckPassed: nil,
                malleabilityReport: malleabilityReport,
                strategy: "individual (Shamir's trick, \(n) sigs)")
        }

        // Large batch: probabilistic batch check
        let batchOk: Bool
        if config.useFiatShamirChallenges {
            batchOk = try batchVerifyFiatShamir(
                signatures: signatures, publicKeys: publicKeys,
                recoveryBits: recoveryBits)
        } else {
            batchOk = try ecdsaEngine.batchVerifyProbabilistic(
                signatures: signatures, pubkeys: publicKeys,
                recoveryBits: recoveryBits)
        }

        if batchOk {
            totalVerified += n
            let msmPoints = 2 * n + 1
            let msmKind = msmPoints <= config.gpuMSMThreshold ? "CPU Pippenger" : "Metal GPU"
            return GPUECDSABatchResult(
                results: [Bool](repeating: true, count: n),
                batchCheckPassed: true,
                malleabilityReport: malleabilityReport,
                strategy: "batch RLC (\(msmKind), \(msmPoints) points)")
        }

        // Batch failed: fall back to individual to identify bad signatures
        let results = (0..<n).map { i in
            ecdsaEngine.verify(sig: signatures[i], pubkey: publicKeys[i])
        }
        totalVerified += n
        return GPUECDSABatchResult(
            results: results, batchCheckPassed: false,
            malleabilityReport: malleabilityReport,
            strategy: "batch RLC failed, fallback to individual")
    }

    /// Simple batch verification returning only per-signature booleans.
    public func verifyBatchSimple(signatures: [ECDSASignature],
                                  publicKeys: [SecpPointAffine],
                                  recoveryBits: [UInt8]? = nil) throws -> [Bool] {
        let result = try verifyBatch(signatures: signatures,
                                     publicKeys: publicKeys,
                                     recoveryBits: recoveryBits)
        return result.results
    }

    // MARK: - Fiat-Shamir Batch Verification

    /// Batch verification using Fiat-Shamir derived challenges.
    ///
    /// The transcript is built from all signature data to derive deterministic
    /// 128-bit challenge weights. This makes the batch verification
    /// non-interactive and reproducible.
    ///
    /// Algorithm:
    ///   1. Build transcript: domain separator || n || for each i: r_i || s_i || z_i || Q_i
    ///   2. Derive seed = keccak256(transcript)
    ///   3. For each i: w_i = keccak256(seed || i) truncated to 128 bits
    ///   4. Compute weighted sums and perform MSM check
    ///
    /// - Returns: true iff ALL signatures are valid.
    public func batchVerifyFiatShamir(signatures: [ECDSASignature],
                                      publicKeys: [SecpPointAffine],
                                      recoveryBits: [UInt8]? = nil) throws -> Bool {
        let n = signatures.count
        precondition(publicKeys.count == n)
        if n == 0 { return true }

        // Step 1: Build Fiat-Shamir transcript
        let challenges = deriveFiatShamirChallenges(
            signatures: signatures, publicKeys: publicKeys, count: n)

        // Step 2: Compute weighted scalars using batch inversion
        let sValues = signatures.map { $0.s }
        let sInvs = secpFrBatchInverse(sValues)

        // Step 3: Build MSM points and scalars
        let totalMSMPoints = 2 * n + 1
        var msmPoints = [SecpPointAffine]()
        msmPoints.reserveCapacity(totalMSMPoints)
        var msmScalars = [[UInt32]]()
        msmScalars.reserveCapacity(totalMSMPoints)

        // Accumulate weighted u1 sum for generator: sum(w_i * u1_i)
        var u1Sum = SecpFr.zero
        for i in 0..<n {
            let u1 = secpFrMul(signatures[i].z, sInvs[i])
            let weightedU1 = secpFrMul(challenges[i], u1)
            u1Sum = secpFrAdd(u1Sum, weightedU1)
        }

        // First point: generator with scalar sum(w_i * u1_i)
        let gen = secp256k1Generator()
        msmPoints.append(gen)
        msmScalars.append(secpFrToInt(u1Sum).flatMap {
            [UInt32($0 & 0xFFFFFFFF), UInt32($0 >> 32)]
        })

        // For each signature: add w_i*u2_i * Q_i term
        for i in 0..<n {
            let u2 = secpFrMul(signatures[i].r, sInvs[i])
            let weightedU2 = secpFrMul(challenges[i], u2)
            msmPoints.append(publicKeys[i])
            msmScalars.append(secpFrToInt(weightedU2).flatMap {
                [UInt32($0 & 0xFFFFFFFF), UInt32($0 >> 32)]
            })
        }

        // For each signature: subtract w_i * R_i (negate the scalar)
        let recov = recoveryBits ?? [UInt8](repeating: 0, count: n)
        for i in 0..<n {
            let rPoint = liftRToPoint(sig: signatures[i], parity: recov[i])
            guard let rp = rPoint else { return false }
            let negWeight = secpFrNeg(challenges[i])
            msmPoints.append(rp)
            msmScalars.append(secpFrToInt(negWeight).flatMap {
                [UInt32($0 & 0xFFFFFFFF), UInt32($0 >> 32)]
            })
        }

        // Step 4: Perform MSM and check result is identity
        if totalMSMPoints <= config.gpuMSMThreshold {
            let result = cSecpPippengerMSM(points: msmPoints, scalars: msmScalars)
            return secpPointIsIdentity(result)
        }

        let result = try msmEngine.msm(points: msmPoints, scalars: msmScalars)
        return secpPointIsIdentity(result)
    }

    // MARK: - Fiat-Shamir Challenge Derivation

    /// Derive deterministic 128-bit challenge weights from a Fiat-Shamir transcript.
    ///
    /// transcript = keccak256("ecdsa-batch-verify" || le64(n) || sig_data...)
    /// w_i = keccak256(transcript_seed || le32(i)) with top 128 bits zeroed
    private func deriveFiatShamirChallenges(signatures: [ECDSASignature],
                                             publicKeys: [SecpPointAffine],
                                             count n: Int) -> [SecpFr] {
        // Build transcript data
        var transcriptData = [UInt8]()
        let domainSep = Array("ecdsa-batch-verify".utf8)
        transcriptData.append(contentsOf: domainSep)

        // Append count as little-endian 8 bytes
        var nLE = UInt64(n).littleEndian
        withUnsafeBytes(of: &nLE) { transcriptData.append(contentsOf: $0) }

        // Append each signature's data: r || s || z || Q.x || Q.y
        for i in 0..<n {
            appendFrToTranscript(&transcriptData, signatures[i].r)
            appendFrToTranscript(&transcriptData, signatures[i].s)
            appendFrToTranscript(&transcriptData, signatures[i].z)
            appendFpToTranscript(&transcriptData, publicKeys[i].x)
            appendFpToTranscript(&transcriptData, publicKeys[i].y)
        }

        // Hash transcript to get seed
        let seed = keccak256(transcriptData)

        // Derive per-signature challenges
        var challenges = [SecpFr]()
        challenges.reserveCapacity(n)

        for i in 0..<n {
            var challengeInput = seed
            var iLE = UInt32(i).littleEndian
            withUnsafeBytes(of: &iLE) { challengeInput.append(contentsOf: $0) }
            let hash = keccak256(challengeInput)

            // Use first 16 bytes (128 bits) as challenge scalar
            var limbs = [UInt64](repeating: 0, count: 4)
            for j in 0..<8 {
                limbs[0] |= UInt64(hash[j]) << (j * 8)
            }
            for j in 0..<8 {
                limbs[1] |= UInt64(hash[8 + j]) << (j * 8)
            }
            // limbs[2] and limbs[3] stay zero (128-bit challenge)

            // Ensure non-zero challenge (vanishingly unlikely but handle it)
            if limbs[0] == 0 && limbs[1] == 0 {
                limbs[0] = 1
            }

            challenges.append(secpFrFromRaw(limbs))
        }

        return challenges
    }

    /// Append a SecpFr value to transcript data as 32 raw bytes (LE).
    private func appendFrToTranscript(_ data: inout [UInt8], _ fr: SecpFr) {
        let limbs = secpFrToInt(fr)
        for limb in limbs {
            var le = limb.littleEndian
            withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
        }
    }

    /// Append a SecpFp value to transcript data as 32 raw bytes (LE).
    private func appendFpToTranscript(_ data: inout [UInt8], _ fp: SecpFp) {
        let limbs = fp.to64()
        for limb in limbs {
            var le = limb.littleEndian
            withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
        }
    }

    // MARK: - Helper: Lift r to Curve Point

    /// Lift an r value (x-coordinate in Fr) to a secp256k1 curve point.
    /// Computes y from y^2 = x^3 + 7 with the given parity.
    private func liftRToPoint(sig: ECDSASignature, parity: UInt8) -> SecpPointAffine? {
        let rInt = secpFrToInt(sig.r)
        return liftX(rInt, parity: parity)
    }

    /// Lift a 256-bit x-coordinate to a curve point: y^2 = x^3 + 7.
    /// Returns the point with the given y-parity (0 = even, 1 = odd).
    private func liftX(_ xRaw: [UInt64], parity: UInt8) -> SecpPointAffine? {
        let x = secpFromRawFp(xRaw)
        let x2 = secpSqr(x)
        let x3 = secpMul(x2, x)
        let seven = secpFromInt(7)
        let rhs = secpAdd(x3, seven)

        guard let y = secpSqrt(rhs) else { return nil }

        let yInt = secpToInt(y)
        let yParity = UInt8(yInt[0] & 1)
        if yParity == parity {
            return SecpPointAffine(x: x, y: y)
        } else {
            return SecpPointAffine(x: x, y: secpNeg(y))
        }
    }

    // MARK: - Strategy Description

    /// Returns a human-readable description of the verification strategy
    /// that would be used for a given batch size.
    public func strategyDescription(batchSize n: Int) -> String {
        if n == 0 { return "empty (no signatures)" }
        if n < config.batchThreshold {
            return "individual (Shamir's trick, \(n) verifications)"
        }
        let msmPoints = 2 * n + 1
        if msmPoints <= config.gpuMSMThreshold {
            return "batch RLC (CPU Pippenger, \(msmPoints) points)"
        }
        return "batch RLC (Metal GPU, \(msmPoints) points)"
    }

    /// Reset verification statistics.
    public func resetStats() {
        totalVerified = 0
        totalBatchOps = 0
    }
}

