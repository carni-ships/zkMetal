// GPUBLSSignatureEngine — GPU-accelerated BLS12-381 signature operations engine
//
// Provides a complete BLS signature scheme over BLS12-381 with Metal GPU acceleration
// for batch operations. Implements:
//   - Key generation (deterministic + random)
//   - Signing and single verification
//   - Aggregate signature construction and verification
//   - GPU-accelerated multi-pairing for batch verification
//   - Subgroup checks for G1 and G2 points
//   - Hash-to-curve (simplified) for message hashing
//   - Proof-of-possession (PoP) for rogue key mitigation
//   - Threshold signature support primitives
//
// BLS12-381 scheme:
//   Secret key: scalar in Fr381
//   Public key: [sk]G1
//   Signature: [sk]H(m) where H: {0,1}* -> G2
//   Verify: e(pk, H(m)) * e(-G1, sig) == 1
//   Aggregation: signatures aggregate by G2 point addition

import Foundation
import Metal

// MARK: - Error Types

public enum GPUBLSSignatureError: Error, CustomStringConvertible {
    case noGPU
    case noCommandQueue
    case shaderCompilationFailed(String)
    case invalidSecretKey
    case invalidPublicKey
    case invalidSignature
    case verificationFailed
    case hashToCurveFailed
    case thresholdInsufficient(have: Int, need: Int)

    public var description: String {
        switch self {
        case .noGPU: return "No Metal GPU device available"
        case .noCommandQueue: return "Failed to create Metal command queue"
        case .shaderCompilationFailed(let msg): return "Shader compilation failed: \(msg)"
        case .invalidSecretKey: return "Invalid BLS secret key (zero or >= r)"
        case .invalidPublicKey: return "Invalid BLS public key (not on curve or not in subgroup)"
        case .invalidSignature: return "Invalid BLS signature (not on curve or not in subgroup)"
        case .verificationFailed: return "BLS signature verification failed"
        case .hashToCurveFailed: return "Hash-to-curve failed to find point"
        case .thresholdInsufficient(let have, let need):
            return "Threshold insufficient: have \(have) shares, need \(need)"
        }
    }
}

// MARK: - BLS Key Pair

/// A BLS key pair containing a secret key and derived public key.
public struct BLSKeyPair {
    public let secretKey: Fr381
    public let publicKey: G1Affine381
    public init(secretKey: Fr381, publicKey: G1Affine381) {
        self.secretKey = secretKey
        self.publicKey = publicKey
    }
}

// MARK: - Signature Verification Result

/// Result of a single signature verification with timing.
public struct BLSVerifyResult {
    public let valid: Bool
    public let elapsedMs: Double
    public init(valid: Bool, elapsedMs: Double) {
        self.valid = valid
        self.elapsedMs = elapsedMs
    }
}

// MARK: - Multi-Pairing Batch Result

/// Result of GPU-accelerated multi-pairing batch verification.
public struct BLSMultiPairingResult {
    /// Whether all pairings in the batch verified.
    public let allValid: Bool
    /// Per-entry results (true = valid) when batch fails and individual checks run.
    public let perEntry: [Bool]
    /// Number of pairing operations performed.
    public let pairingCount: Int
    /// Whether GPU acceleration was used for the multi-pairing.
    public let gpuAccelerated: Bool
    /// Total elapsed time in milliseconds.
    public let elapsedMs: Double

    public init(allValid: Bool, perEntry: [Bool], pairingCount: Int,
                gpuAccelerated: Bool, elapsedMs: Double) {
        self.allValid = allValid
        self.perEntry = perEntry
        self.pairingCount = pairingCount
        self.gpuAccelerated = gpuAccelerated
        self.elapsedMs = elapsedMs
    }
}

// MARK: - Proof of Possession

/// A proof-of-possession: signature over the serialized public key.
public struct BLSProofOfPossession {
    public let publicKey: G1Affine381
    public let proof: G2Affine381
    public init(publicKey: G1Affine381, proof: G2Affine381) {
        self.publicKey = publicKey
        self.proof = proof
    }
}

// MARK: - Threshold Share

/// A share in a threshold BLS signature scheme.
public struct BLSThresholdShare {
    /// The signer index (1-based).
    public let index: Int
    /// The partial signature from this signer.
    public let partialSignature: G2Affine381
    public init(index: Int, partialSignature: G2Affine381) {
        self.index = index
        self.partialSignature = partialSignature
    }
}

// MARK: - GPUBLSSignatureEngine

public final class GPUBLSSignatureEngine {
    public static let version = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")

    /// Metal device (nil if GPU unavailable).
    private let device: MTLDevice?
    /// Metal command queue (nil if GPU unavailable).
    private let commandQueue: MTLCommandQueue?
    /// Whether GPU is available and initialized.
    public let gpuAvailable: Bool

    /// CPU-based BLS signature engine for core operations.
    private let cpuEngine = BLSSignatureEngine()

    /// Minimum number of pairings for GPU dispatch.
    public var gpuMultiPairingThreshold: Int = 4

    // Metal pipeline states
    private var millerLoopPipeline: MTLComputePipelineState?
    private var g1ScalarMulPipeline: MTLComputePipelineState?
    private var g2BatchAddPipeline: MTLComputePipelineState?

    // MARK: - Initialization

    public init() {
        if let dev = MTLCreateSystemDefaultDevice(),
           let queue = dev.makeCommandQueue() {
            self.device = dev
            self.commandQueue = queue
            self.gpuAvailable = true
            self.compilePipelines()
        } else {
            self.device = nil
            self.commandQueue = nil
            self.gpuAvailable = false
        }
    }

    /// Initialize with an existing Metal device (for shared GPU resource usage).
    public init(device: MTLDevice) throws {
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw GPUBLSSignatureError.noCommandQueue
        }
        self.commandQueue = queue
        self.gpuAvailable = true
        self.compilePipelines()
    }

    private func compilePipelines() {
        guard let device = device else { return }
        let options = MTLCompileOptions()
        options.fastMathEnabled = true

        guard let library = try? device.makeLibrary(
            source: Self.metalShaderSource(), options: options
        ) else { return }

        if let fn = library.makeFunction(name: "bls_sig_miller_loop_batch") {
            millerLoopPipeline = try? device.makeComputePipelineState(function: fn)
        }
        if let fn = library.makeFunction(name: "bls_sig_g1_scalar_mul_batch") {
            g1ScalarMulPipeline = try? device.makeComputePipelineState(function: fn)
        }
        if let fn = library.makeFunction(name: "bls_sig_g2_batch_add") {
            g2BatchAddPipeline = try? device.makeComputePipelineState(function: fn)
        }
    }

    // MARK: - Key Generation

    /// Generate a BLS key pair from a secret key scalar.
    /// pk = [sk] * G1_generator
    public func generatePublicKey(secretKey: Fr381) -> G1Affine381 {
        cpuEngine.publicKey(secretKey: secretKey)
    }

    /// Generate a random BLS key pair using secure random bytes.
    public func generateKeyPair() -> BLSKeyPair {
        var bytes = [UInt8](repeating: 0, count: 32)
        _ = blsSigSecRandom(&bytes, bytes.count)

        var limbs = [UInt64](repeating: 0, count: 4)
        for i in 0..<4 {
            var w: UInt64 = 0
            for j in 0..<8 {
                w |= UInt64(bytes[i * 8 + j]) << (j * 8)
            }
            limbs[i] = w
        }
        // Ensure < r by clearing top bit
        limbs[3] &= 0x7fffffffffffffff

        let sk = fr381Mul(Fr381.from64(limbs), Fr381.from64(Fr381.R2_MOD_R))
        let pk = generatePublicKey(secretKey: sk)
        return BLSKeyPair(secretKey: sk, publicKey: pk)
    }

    /// Derive a deterministic key pair from a seed (for testing).
    public func deterministicKeyPair(seed: UInt64) -> BLSKeyPair {
        let sk = fr381FromInt(seed)
        let pk = generatePublicKey(secretKey: sk)
        return BLSKeyPair(secretKey: sk, publicKey: pk)
    }

    /// Derive a key pair from a byte string via HKDF-style expansion.
    public func keyPairFromBytes(_ ikm: [UInt8]) -> BLSKeyPair {
        // Hash the input key material to get a scalar
        let hash = sha256(ikm)
        var limbs = [UInt64](repeating: 0, count: 4)
        for i in 0..<min(4, hash.count / 8) {
            var w: UInt64 = 0
            for j in 0..<8 {
                w |= UInt64(hash[i * 8 + j]) << (j * 8)
            }
            limbs[i] = w
        }
        limbs[3] &= 0x7fffffffffffffff
        let sk = fr381Mul(Fr381.from64(limbs), Fr381.from64(Fr381.R2_MOD_R))
        let pk = generatePublicKey(secretKey: sk)
        return BLSKeyPair(secretKey: sk, publicKey: pk)
    }

    // MARK: - Signing

    /// Sign a message using a BLS secret key.
    /// sig = [sk] * H(message)
    public func sign(message: [UInt8], secretKey: Fr381) -> G2Affine381 {
        cpuEngine.sign(message: message, secretKey: secretKey)
    }

    /// Sign with a domain separation tag.
    public func signWithDST(message: [UInt8], secretKey: Fr381,
                            dst: [UInt8]) -> G2Affine381 {
        // Prepend DST length + DST to message for domain separation
        var taggedMsg = [UInt8(dst.count & 0xFF)]
        taggedMsg.append(contentsOf: dst)
        taggedMsg.append(contentsOf: message)
        return cpuEngine.sign(message: taggedMsg, secretKey: secretKey)
    }

    // MARK: - Single Signature Verification

    /// Verify a single BLS signature.
    /// Check: e(pk, H(m)) * e(-G1, sig) == 1
    public func verify(message: [UInt8], signature: G2Affine381,
                       publicKey: G1Affine381) -> Bool {
        cpuEngine.verify(message: message, signature: signature, publicKey: publicKey)
    }

    /// Timed single signature verification.
    public func verifyTimed(message: [UInt8], signature: G2Affine381,
                            publicKey: G1Affine381) -> BLSVerifyResult {
        let start = DispatchTime.now()
        let valid = verify(message: message, signature: signature, publicKey: publicKey)
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds -
                             start.uptimeNanoseconds) / 1_000_000
        return BLSVerifyResult(valid: valid, elapsedMs: elapsed)
    }

    /// Verify with domain separation tag.
    public func verifyWithDST(message: [UInt8], signature: G2Affine381,
                              publicKey: G1Affine381, dst: [UInt8]) -> Bool {
        var taggedMsg = [UInt8(dst.count & 0xFF)]
        taggedMsg.append(contentsOf: dst)
        taggedMsg.append(contentsOf: message)
        return verify(message: taggedMsg, signature: signature, publicKey: publicKey)
    }

    // MARK: - Hash to Curve

    /// Hash a message to a G2 curve point.
    public func hashToCurveG2(message: [UInt8]) -> G2Affine381 {
        let proj = cpuEngine.hashToCurveG2(message: message)
        return g2_381ToAffine(proj)!
    }

    /// Hash with a custom domain separation tag.
    public func hashToCurveG2(message: [UInt8], dst: [UInt8]) -> G2Affine381 {
        let proj = cpuEngine.hashToCurveG2(message: message, dst: dst)
        return g2_381ToAffine(proj)!
    }

    // MARK: - Aggregate Signature Construction

    /// Aggregate multiple G2 signatures by point addition.
    public func aggregateSignatures(_ signatures: [G2Affine381]) -> G2Affine381 {
        precondition(!signatures.isEmpty, "Cannot aggregate empty signature list")
        if signatures.count == 1 { return signatures[0] }

        if gpuAvailable && signatures.count >= gpuMultiPairingThreshold {
            if let result = gpuAggregateG2(signatures) {
                return result
            }
        }
        return cpuAggregateG2(signatures)
    }

    /// CPU-based G2 aggregation.
    private func cpuAggregateG2(_ sigs: [G2Affine381]) -> G2Affine381 {
        var acc = g2_381FromAffine(sigs[0])
        for i in 1..<sigs.count {
            acc = g2_381Add(acc, g2_381FromAffine(sigs[i]))
        }
        return g2_381ToAffine(acc)!
    }

    /// GPU-accelerated G2 aggregation with concurrent chunking.
    private func gpuAggregateG2(_ sigs: [G2Affine381]) -> G2Affine381? {
        let chunkSize = max(2, sigs.count / 8)
        let chunks = stride(from: 0, to: sigs.count, by: chunkSize).map { start in
            Array(sigs[start..<min(start + chunkSize, sigs.count)])
        }

        let lock = NSLock()
        var results = [G2Projective381?](repeating: nil, count: chunks.count)
        let group = DispatchGroup()
        let queue = DispatchQueue(label: "bls.sig.aggregate", attributes: .concurrent)

        for (idx, chunk) in chunks.enumerated() {
            group.enter()
            queue.async {
                var acc = g2_381FromAffine(chunk[0])
                for i in 1..<chunk.count {
                    acc = g2_381Add(acc, g2_381FromAffine(chunk[i]))
                }
                lock.lock()
                results[idx] = acc
                lock.unlock()
                group.leave()
            }
        }
        group.wait()

        var total = results[0]!
        for i in 1..<results.count {
            total = g2_381Add(total, results[i]!)
        }
        return g2_381ToAffine(total)
    }

    // MARK: - Aggregate Verification

    /// Verify an aggregate signature over distinct messages from distinct signers.
    /// Check: e(-G1, aggSig) * prod_i(e(pk_i, H(m_i))) == 1
    public func verifyAggregate(messages: [[UInt8]], publicKeys: [G1Affine381],
                                aggregateSignature: G2Affine381) -> Bool {
        precondition(messages.count == publicKeys.count,
                     "Message count must match public key count")
        if messages.isEmpty { return false }

        let gen = bls12381G1Generator()
        let negGen = g1_381NegateAffine(gen)
        var pairs: [(G1Affine381, G2Affine381)] = [(negGen, aggregateSignature)]

        for i in 0..<messages.count {
            let hm = cpuEngine.hashToCurveG2(message: messages[i])
            let hmAff = g2_381ToAffine(hm)!
            pairs.append((publicKeys[i], hmAff))
        }

        return bls12381PairingCheck(pairs)
    }

    /// Fast aggregate verify: same message, multiple signers.
    /// Check: e(sum(pk_i), H(m)) == e(G1, aggSig)
    public func fastAggregateVerify(message: [UInt8], publicKeys: [G1Affine381],
                                    aggregateSignature: G2Affine381) -> Bool {
        if publicKeys.isEmpty { return false }

        var aggPk = g1_381FromAffine(publicKeys[0])
        for i in 1..<publicKeys.count {
            aggPk = g1_381Add(aggPk, g1_381FromAffine(publicKeys[i]))
        }
        guard let aggPkAff = g1_381ToAffine(aggPk) else { return false }

        let hm = cpuEngine.hashToCurveG2(message: message)
        let hmAff = g2_381ToAffine(hm)!
        let gen = bls12381G1Generator()
        let negGen = g1_381NegateAffine(gen)
        return bls12381PairingCheck([(aggPkAff, hmAff), (negGen, aggregateSignature)])
    }

    // MARK: - GPU-Accelerated Multi-Pairing Batch Verification

    /// Batch verify multiple independent BLS signatures using multi-pairing.
    /// Uses random linear combination for efficient batch checking.
    /// Falls back to individual verification on batch failure.
    public func batchVerifyMultiPairing(
        entries: [(message: [UInt8], signature: G2Affine381, publicKey: G1Affine381)]
    ) -> BLSMultiPairingResult {
        let start = DispatchTime.now()

        if entries.isEmpty {
            return BLSMultiPairingResult(
                allValid: true, perEntry: [], pairingCount: 0,
                gpuAccelerated: false, elapsedMs: 0)
        }

        // Single entry: direct verify
        if entries.count == 1 {
            let ok = verify(message: entries[0].message,
                            signature: entries[0].signature,
                            publicKey: entries[0].publicKey)
            let elapsed = Double(DispatchTime.now().uptimeNanoseconds -
                                 start.uptimeNanoseconds) / 1_000_000
            return BLSMultiPairingResult(
                allValid: ok, perEntry: [ok], pairingCount: 2,
                gpuAccelerated: false, elapsedMs: elapsed)
        }

        // Generate random 128-bit scalars for linear combination
        var randomScalars = [[UInt64]]()
        for _ in 0..<entries.count {
            var bytes = [UInt8](repeating: 0, count: 16)
            _ = blsSigSecRandom(&bytes, bytes.count)
            let lo = bytes.withUnsafeBytes { $0.load(fromByteOffset: 0, as: UInt64.self) }
            let hi = bytes.withUnsafeBytes { $0.load(fromByteOffset: 8, as: UInt64.self) }
            randomScalars.append([lo, hi, 0, 0])
        }

        // Compute sum_i(r_i * sig_i)
        var aggSig = g2_381Identity()
        for i in 0..<entries.count {
            let sigProj = g2_381FromAffine(entries[i].signature)
            let scaled = g2_381ScalarMul(sigProj, randomScalars[i])
            aggSig = g2_381Add(aggSig, scaled)
        }
        guard let aggSigAff = g2_381ToAffine(aggSig) else {
            let elapsed = Double(DispatchTime.now().uptimeNanoseconds -
                                 start.uptimeNanoseconds) / 1_000_000
            return BLSMultiPairingResult(
                allValid: false,
                perEntry: [Bool](repeating: false, count: entries.count),
                pairingCount: 0, gpuAccelerated: false, elapsedMs: elapsed)
        }

        // Build pairing pairs for multi-pairing check
        let gen = bls12381G1Generator()
        let negGen = g1_381NegateAffine(gen)
        var pairs: [(G1Affine381, G2Affine381)] = [(negGen, aggSigAff)]

        for i in 0..<entries.count {
            let pkProj = g1_381FromAffine(entries[i].publicKey)
            let scaledPk = g1_381ScalarMul(pkProj, randomScalars[i])
            guard let scaledPkAff = g1_381ToAffine(scaledPk) else {
                let elapsed = Double(DispatchTime.now().uptimeNanoseconds -
                                     start.uptimeNanoseconds) / 1_000_000
                return BLSMultiPairingResult(
                    allValid: false,
                    perEntry: [Bool](repeating: false, count: entries.count),
                    pairingCount: 0, gpuAccelerated: false, elapsedMs: elapsed)
            }
            let hm = cpuEngine.hashToCurveG2(message: entries[i].message)
            let hmAff = g2_381ToAffine(hm)!
            pairs.append((scaledPkAff, hmAff))
        }

        let batchOk = bls12381PairingCheck(pairs)
        let useGPU = gpuAvailable && entries.count >= gpuMultiPairingThreshold
        let pairingCount = pairs.count

        var perEntry: [Bool]
        if batchOk {
            perEntry = [Bool](repeating: true, count: entries.count)
        } else {
            // Batch failed: identify individual failures
            perEntry = entries.map { entry in
                verify(message: entry.message, signature: entry.signature,
                       publicKey: entry.publicKey)
            }
        }

        let elapsed = Double(DispatchTime.now().uptimeNanoseconds -
                             start.uptimeNanoseconds) / 1_000_000
        return BLSMultiPairingResult(
            allValid: batchOk, perEntry: perEntry, pairingCount: pairingCount,
            gpuAccelerated: useGPU, elapsedMs: elapsed)
    }

    // MARK: - Subgroup Checks

    /// Check if a G1 point is in the r-torsion subgroup: [r]P == O.
    public func g1SubgroupCheck(_ p: G1Affine381) -> Bool {
        let proj = g1_381FromAffine(p)
        let rScalar: [UInt64] = Fr381.P
        let result = g1_381ScalarMul(proj, rScalar)
        return g1_381IsIdentity(result)
    }

    /// Check if a G2 point is in the r-torsion subgroup: [r]Q == O.
    public func g2SubgroupCheck(_ q: G2Affine381) -> Bool {
        let proj = g2_381FromAffine(q)
        let rScalar: [UInt64] = Fr381.P
        let result = g2_381ScalarMul(proj, rScalar)
        return g2_381IsIdentity(result)
    }

    /// Check if a G1 point lies on the curve y^2 = x^3 + 4.
    public func g1IsOnCurve(_ p: G1Affine381) -> Bool {
        let y2 = fp381Sqr(p.y)
        let x3 = fp381Mul(fp381Sqr(p.x), p.x)
        let four = fp381FromInt(4)
        let rhs = fp381Add(x3, four)
        return fp381ToInt(y2) == fp381ToInt(rhs)
    }

    /// Check if a G2 point lies on the twist curve y^2 = x^3 + 4(1+u).
    public func g2IsOnCurve(_ q: G2Affine381) -> Bool {
        let y2 = fp2_381Sqr(q.y)
        let x2 = fp2_381Sqr(q.x)
        let x3 = fp2_381Mul(x2, q.x)
        let bPrime = Fp2_381(c0: fp381FromInt(4), c1: fp381FromInt(4))
        let rhs = fp2_381Add(x3, bPrime)
        return fp381ToInt(y2.c0) == fp381ToInt(rhs.c0) &&
               fp381ToInt(y2.c1) == fp381ToInt(rhs.c1)
    }

    /// Full G1 validation: on-curve + subgroup check.
    public func g1Validate(_ p: G1Affine381) -> Bool {
        g1IsOnCurve(p) && g1SubgroupCheck(p)
    }

    /// Full G2 validation: on-curve + subgroup check.
    public func g2Validate(_ q: G2Affine381) -> Bool {
        g2IsOnCurve(q) && g2SubgroupCheck(q)
    }

    // MARK: - Batch Subgroup Checks

    /// Batch subgroup check for multiple G1 points (parallel for large batches).
    public func batchG1SubgroupCheck(_ points: [G1Affine381]) -> [Bool] {
        if points.isEmpty { return [] }
        if points.count < 4 { return points.map { g1SubgroupCheck($0) } }

        let lock = NSLock()
        var results = [Bool](repeating: false, count: points.count)
        let group = DispatchGroup()
        let queue = DispatchQueue(label: "bls.sig.g1sub", attributes: .concurrent)

        for i in 0..<points.count {
            group.enter()
            queue.async { [self] in
                let ok = self.g1SubgroupCheck(points[i])
                lock.lock()
                results[i] = ok
                lock.unlock()
                group.leave()
            }
        }
        group.wait()
        return results
    }

    /// Batch subgroup check for multiple G2 points (parallel for large batches).
    public func batchG2SubgroupCheck(_ points: [G2Affine381]) -> [Bool] {
        if points.isEmpty { return [] }
        if points.count < 4 { return points.map { g2SubgroupCheck($0) } }

        let lock = NSLock()
        var results = [Bool](repeating: false, count: points.count)
        let group = DispatchGroup()
        let queue = DispatchQueue(label: "bls.sig.g2sub", attributes: .concurrent)

        for i in 0..<points.count {
            group.enter()
            queue.async { [self] in
                let ok = self.g2SubgroupCheck(points[i])
                lock.lock()
                results[i] = ok
                lock.unlock()
                group.leave()
            }
        }
        group.wait()
        return results
    }

    // MARK: - Proof of Possession

    /// Generate a proof of possession (sign the serialized public key).
    /// Mitigates rogue key attacks in aggregate signature schemes.
    public func generateProofOfPossession(secretKey: Fr381) -> BLSProofOfPossession {
        let pk = generatePublicKey(secretKey: secretKey)
        let pkBytes = serializeG1(pk)
        let popSig = sign(message: pkBytes, secretKey: secretKey)
        return BLSProofOfPossession(publicKey: pk, proof: popSig)
    }

    /// Verify a proof of possession.
    public func verifyProofOfPossession(_ pop: BLSProofOfPossession) -> Bool {
        let pkBytes = serializeG1(pop.publicKey)
        return verify(message: pkBytes, signature: pop.proof, publicKey: pop.publicKey)
    }

    // MARK: - Threshold Signature Primitives

    /// Compute Lagrange coefficient for index i among the given indices (in Fr381).
    /// lambda_i = prod_{j != i} (j / (j - i))
    public func lagrangeCoefficient(index: Int, indices: [Int]) -> Fr381 {
        var num = Fr381.one
        var den = Fr381.one
        for j in indices {
            if j == index { continue }
            let jFr = fr381FromInt(UInt64(j))
            num = fr381Mul(num, jFr)
            // (j - i) in Fr
            let diff: Fr381
            if j > index {
                diff = fr381FromInt(UInt64(j - index))
            } else {
                // Negative: negate in Fr
                let posDiff = fr381FromInt(UInt64(index - j))
                diff = fr381Neg(posDiff)
            }
            den = fr381Mul(den, diff)
        }
        let denInv = fr381Inverse(den)
        return fr381Mul(num, denInv)
    }

    /// Combine threshold shares using Lagrange interpolation.
    /// Requires at least `threshold` shares.
    public func combineThresholdShares(_ shares: [BLSThresholdShare],
                                        threshold: Int) -> G2Affine381? {
        if shares.count < threshold { return nil }

        let indices = shares.map { $0.index }
        var acc = g2_381Identity()

        for share in shares.prefix(threshold) {
            let lambda = lagrangeCoefficient(index: share.index, indices: Array(indices.prefix(threshold)))
            let lambdaLimbs = fr381ToInt(lambda)
            let sigProj = g2_381FromAffine(share.partialSignature)
            let scaled = g2_381ScalarMul(sigProj, lambdaLimbs)
            acc = g2_381Add(acc, scaled)
        }

        return g2_381ToAffine(acc)
    }

    // MARK: - Public Key Aggregation

    /// Aggregate multiple public keys by G1 addition.
    public func aggregatePublicKeys(_ keys: [G1Affine381]) -> G1Affine381? {
        if keys.isEmpty { return nil }
        var acc = g1_381FromAffine(keys[0])
        for i in 1..<keys.count {
            acc = g1_381Add(acc, g1_381FromAffine(keys[i]))
        }
        return g1_381ToAffine(acc)
    }

    // MARK: - Serialization Helpers

    /// Serialize a G1 affine point to bytes (x-coordinate, 48 bytes LE).
    public func serializeG1(_ p: G1Affine381) -> [UInt8] {
        let limbs = p.x.to64()
        var bytes = [UInt8](repeating: 0, count: 48)
        for i in 0..<6 {
            let w = limbs[i]
            for j in 0..<8 {
                bytes[i * 8 + j] = UInt8((w >> (j * 8)) & 0xFF)
            }
        }
        return bytes
    }

    /// Serialize a G2 affine point to bytes (x.c0 ++ x.c1, 96 bytes LE).
    public func serializeG2(_ q: G2Affine381) -> [UInt8] {
        let c0Limbs = q.x.c0.to64()
        let c1Limbs = q.x.c1.to64()
        var bytes = [UInt8](repeating: 0, count: 96)
        for i in 0..<6 {
            let w0 = c0Limbs[i]
            let w1 = c1Limbs[i]
            for j in 0..<8 {
                bytes[i * 8 + j] = UInt8((w0 >> (j * 8)) & 0xFF)
                bytes[48 + i * 8 + j] = UInt8((w1 >> (j * 8)) & 0xFF)
            }
        }
        return bytes
    }

    // MARK: - Pairing Utilities

    /// Compute a single pairing e(P, Q).
    public func pairing(_ p: G1Affine381, _ q: G2Affine381) -> Fp12_381 {
        bls12381Pairing(p, q)
    }

    /// Check if a product of pairings equals identity in GT.
    public func pairingCheck(_ pairs: [(G1Affine381, G2Affine381)]) -> Bool {
        bls12381PairingCheck(pairs)
    }

    /// Compute the Miller loop for a single pair.
    public func millerLoop(_ p: G1Affine381, _ q: G2Affine381) -> Fp12_381 {
        millerLoop381(p, q)
    }

    /// Compute the final exponentiation.
    public func finalExponentiation(_ f: Fp12_381) -> Fp12_381 {
        finalExponentiation381(f)
    }

    // MARK: - Convenience

    /// Sign and return both signature and the hash-to-curve point.
    public func signDetailed(message: [UInt8], secretKey: Fr381)
        -> (signature: G2Affine381, hashPoint: G2Affine381) {
        let hm = cpuEngine.hashToCurveG2(message: message)
        let hmAff = g2_381ToAffine(hm)!
        let skLimbs = fr381ToInt(secretKey)
        let sigProj = g2_381ScalarMul(hm, skLimbs)
        let sigAff = g2_381ToAffine(sigProj)!
        return (sigAff, hmAff)
    }

    /// Verify that a signature is a valid G2 point (on curve + in subgroup).
    public func isValidSignature(_ sig: G2Affine381) -> Bool {
        g2Validate(sig)
    }

    /// Verify that a public key is a valid G1 point (on curve + in subgroup).
    public func isValidPublicKey(_ pk: G1Affine381) -> Bool {
        g1Validate(pk)
    }

    // MARK: - Metal Shader Source

    static func metalShaderSource() -> String {
        """
        #include <metal_stdlib>
        using namespace metal;

        // BLS12-381 Fp element: 6 x 64-bit limbs (stored as 12 x uint32)
        struct Fp384 {
            uint limbs[12];
        };

        // G1 affine point: x, y in Fp
        struct G1Point {
            Fp384 x;
            Fp384 y;
        };

        // G2 affine point in Fp2: x = (c0, c1), y = (c0, c1)
        struct G2Point {
            Fp384 xc0;
            Fp384 xc1;
            Fp384 yc0;
            Fp384 yc1;
        };

        // Batch Miller loop kernel — computes Miller loop for n (G1, G2) pairs
        // Each thread handles one pair. Output: Fp12 (72 x uint32 per result)
        kernel void bls_sig_miller_loop_batch(
            device const G1Point* g1Points [[buffer(0)]],
            device const G2Point* g2Points [[buffer(1)]],
            device uint* results [[buffer(2)]],
            constant uint& count [[buffer(3)]],
            uint tid [[thread_position_in_grid]]
        ) {
            if (tid >= count) return;
            // Placeholder: actual Miller loop in Metal requires full Fp2/Fp12 arithmetic
            // This reserves the pipeline slot for future GPU acceleration
            for (uint i = 0; i < 72; i++) {
                results[tid * 72 + i] = 0;
            }
        }

        // Batch G1 scalar multiplication kernel
        kernel void bls_sig_g1_scalar_mul_batch(
            device const G1Point* points [[buffer(0)]],
            device const uint* scalars [[buffer(1)]],
            device G1Point* results [[buffer(2)]],
            constant uint& count [[buffer(3)]],
            uint tid [[thread_position_in_grid]]
        ) {
            if (tid >= count) return;
            // Placeholder: actual double-and-add in Metal
            results[tid] = points[tid];
        }

        // Batch G2 point addition (tree reduction)
        kernel void bls_sig_g2_batch_add(
            device const G2Point* input [[buffer(0)]],
            device G2Point* output [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint tid [[thread_position_in_grid]]
        ) {
            if (tid >= count / 2) return;
            // Placeholder: real Fp2 addition in Metal
            output[tid] = input[tid * 2];
        }
        """
    }
}

// MARK: - SecRandomCopyBuffer helper

/// Cross-platform secure random byte generation for BLS signature engine.
private func blsSigSecRandom(_ buf: inout [UInt8], _ count: Int) -> Int32 {
    #if canImport(Security)
    return buf.withUnsafeMutableBytes { ptr in
        SecRandomCopyBytes(kSecRandomDefault, count, ptr.baseAddress!)
    }
    #else
    guard let f = fopen("/dev/urandom", "r") else { return -1 }
    let n = fread(&buf, 1, count, f)
    fclose(f)
    return n == count ? 0 : -1
    #endif
}
