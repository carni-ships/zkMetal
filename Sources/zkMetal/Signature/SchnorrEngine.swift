// BIP 340 Schnorr Signatures over secp256k1
// Implements the Bitcoin Taproot signature scheme.
//
// Key properties:
//   - x-only public keys (32 bytes): even Y coordinate convention
//   - Tagged hashing: SHA256(SHA256(tag) || SHA256(tag) || msg)
//   - Deterministic nonce: k = tagged_hash("BIP0340/nonce", rand || pk || msg)
//   - Signature: (R.x, s) where R = k*G with even Y, s = k + e*sk mod n
//   - Verification: s*G == R + e*P (no field inversion needed)

import Foundation
import NeonFieldOps

// MARK: - Schnorr Signature Types

public struct SchnorrSignature: Equatable {
    /// x-coordinate of R (32 bytes, big-endian)
    public let rx: [UInt8]
    /// Scalar s (32 bytes, big-endian)
    public let s: [UInt8]

    public init(rx: [UInt8], s: [UInt8]) {
        precondition(rx.count == 32 && s.count == 32)
        self.rx = rx
        self.s = s
    }

    /// 64-byte serialization: rx || s
    public var bytes: [UInt8] { rx + s }

    public static func fromBytes(_ data: [UInt8]) -> SchnorrSignature? {
        guard data.count == 64 else { return nil }
        return SchnorrSignature(rx: Array(data[0..<32]), s: Array(data[32..<64]))
    }
}

/// x-only public key (32 bytes, big-endian)
public struct SchnorrPublicKey: Equatable {
    public let bytes: [UInt8]

    public init(bytes: [UInt8]) {
        precondition(bytes.count == 32)
        self.bytes = bytes
    }
}

// MARK: - Schnorr Engine

public class SchnorrEngine {
    public static let version = Versions.schnorr
    public let msmEngine: Secp256k1MSM?

    /// Initialize with optional MSM engine for batch verification.
    /// Pass nil to skip GPU-accelerated batch verify.
    public init(withMSM: Bool = false) throws {
        if withMSM {
            self.msmEngine = try Secp256k1MSM()
        } else {
            self.msmEngine = nil
        }
    }

    // MARK: - Tagged Hashing (BIP 340)

    /// BIP 340 tagged hash: SHA256(SHA256(tag) || SHA256(tag) || msg)
    public static func taggedHash(tag: String, data: [UInt8]) -> [UInt8] {
        let tagHash = sha256(Array(tag.utf8))
        return sha256(tagHash + tagHash + data)
    }

    // MARK: - Key Generation

    /// Generate a public key from a 32-byte secret key.
    /// Returns x-only public key (even Y convention).
    public func publicKey(secretKey: [UInt8]) -> SchnorrPublicKey? {
        guard secretKey.count == 32 else { return nil }
        let skLimbs = bytesToLimbs(secretKey)

        // sk must be in [1, n-1]
        if isZero256(skLimbs) { return nil }
        if gte256(skLimbs, SecpFr.N) { return nil }

        let gen = secp256k1Generator()
        let gProj = secpPointFromAffine(gen)
        let pkProj = secpPointMulScalar(gProj, skLimbs)
        let pkAff = secpPointToAffine(pkProj)

        return SchnorrPublicKey(bytes: fpToBytes(pkAff.x))
    }

    /// Generate a key pair: (secretKey, publicKey).
    /// Uses system random for the secret key.
    public func keyGen() -> (secretKey: [UInt8], publicKey: SchnorrPublicKey)? {
        var sk = [UInt8](repeating: 0, count: 32)
        for attempt in 0..<100 {
            _ = attempt
            // Generate random secret key
            for i in 0..<32 {
                var r: UInt8 = 0
                arc4random_buf(&r, 1)
                sk[i] = r
            }
            if let pk = publicKey(secretKey: sk) {
                return (sk, pk)
            }
        }
        return nil
    }

    // MARK: - Sign

    /// BIP 340 Schnorr sign.
    /// secretKey: 32 bytes, message: arbitrary length.
    /// Optional auxRand: 32 bytes of auxiliary randomness (default: zeros).
    public func sign(message: [UInt8], secretKey: [UInt8], auxRand: [UInt8]? = nil) -> SchnorrSignature? {
        guard secretKey.count == 32 else { return nil }
        let skLimbs = bytesToLimbs(secretKey)
        if isZero256(skLimbs) { return nil }
        if gte256(skLimbs, SecpFr.N) { return nil }

        // Compute P = sk * G
        let gen = secp256k1Generator()
        let gProj = secpPointFromAffine(gen)
        let pProj = secpPointMulScalar(gProj, skLimbs)
        let pAff = secpPointToAffine(pProj)

        // Negate secret key if P.y is odd (ensure even Y convention)
        let pyInt = secpToInt(pAff.y)
        let pYIsOdd = (pyInt[0] & 1) == 1
        let d: [UInt64]
        if pYIsOdd {
            let (negated, _) = sub256(SecpFr.N, skLimbs)
            d = negated
        } else {
            d = skLimbs
        }

        let pkBytes = fpToBytes(pAff.x)

        // Compute nonce
        let aux = auxRand ?? [UInt8](repeating: 0, count: 32)
        precondition(aux.count == 32)

        // t = d XOR tagged_hash("BIP0340/aux", a)
        let auxHash = SchnorrEngine.taggedHash(tag: "BIP0340/aux", data: aux)
        let dBytes = limbsToBytes(d)
        var t = [UInt8](repeating: 0, count: 32)
        for i in 0..<32 { t[i] = dBytes[i] ^ auxHash[i] }

        // k' = tagged_hash("BIP0340/nonce", t || pk || msg) mod n
        let nonceHash = SchnorrEngine.taggedHash(tag: "BIP0340/nonce", data: t + pkBytes + message)
        var kLimbs = bytesToLimbs(nonceHash)
        reduceMod(&kLimbs, SecpFr.N)

        if isZero256(kLimbs) { return nil }

        // R = k' * G
        let rProj = secpPointMulScalar(gProj, kLimbs)
        let rAff = secpPointToAffine(rProj)

        // Negate k if R.y is odd
        let ryInt = secpToInt(rAff.y)
        let rYIsOdd = (ryInt[0] & 1) == 1
        let k: [UInt64]
        if rYIsOdd {
            let (negated, _) = sub256(SecpFr.N, kLimbs)
            k = negated
        } else {
            k = kLimbs
        }

        let rxBytes = fpToBytes(rAff.x)

        // e = tagged_hash("BIP0340/challenge", R.x || P.x || msg) mod n
        let eHash = SchnorrEngine.taggedHash(tag: "BIP0340/challenge", data: rxBytes + pkBytes + message)
        var eLimbs = bytesToLimbs(eHash)
        reduceMod(&eLimbs, SecpFr.N)

        // s = (k + e * d) mod n
        let eMont = secpFrFromRaw(eLimbs)
        let dMont = secpFrFromRaw(d)
        let kMont = secpFrFromRaw(k)
        let ed = secpFrMul(eMont, dMont)
        let sMont = secpFrAdd(kMont, ed)
        let sLimbs = secpFrToInt(sMont)
        let sBytes = limbsToBytes(sLimbs)

        return SchnorrSignature(rx: rxBytes, s: sBytes)
    }

    // MARK: - Verify

    /// BIP 340 Schnorr verify.
    /// Returns true if the signature is valid for the given message and public key.
    public func verify(message: [UInt8], signature: SchnorrSignature, publicKey: SchnorrPublicKey) -> Bool {
        // Lift public key x-coordinate to curve point with even Y
        guard let pAff = liftX(publicKey.bytes) else { return false }

        // Parse R.x from signature
        let rxLimbs = bytesToLimbs(signature.rx)
        // R.x must be < p
        if gte256(rxLimbs, SecpFp.P) { return false }

        // Parse s from signature
        let sLimbs = bytesToLimbs(signature.s)
        // s must be < n
        if gte256(sLimbs, SecpFr.N) { return false }

        // e = tagged_hash("BIP0340/challenge", R.x || P.x || msg) mod n
        let eHash = SchnorrEngine.taggedHash(tag: "BIP0340/challenge", data: signature.rx + publicKey.bytes + message)
        var eLimbs = bytesToLimbs(eHash)
        reduceMod(&eLimbs, SecpFr.N)

        // Verify: s*G == R + e*P
        // Equivalently: R = s*G - e*P = s*G + (-e)*P
        // Use Shamir's trick for ~25% faster verification (single double-and-add scan)
        let gen = secp256k1Generator()
        let gProj = secpPointFromAffine(gen)
        let pProj = secpPointFromAffine(pAff)

        // Negate e mod n: -e = n - e
        let negE: [UInt64]
        if isZero256(eLimbs) {
            negE = eLimbs
        } else {
            let (neg, _) = sub256(SecpFr.N, eLimbs)
            negE = neg
        }

        // R' = s*G + (-e)*P via Shamir's trick
        let rPrime = secpShamirDoubleMul(gProj, sLimbs, pProj, negE)

        // R' must not be the point at infinity
        if secpPointIsIdentity(rPrime) { return false }

        let rPrimeAff = secpPointToAffine(rPrime)

        // R'.y must be even
        let ryInt = secpToInt(rPrimeAff.y)
        if (ryInt[0] & 1) != 0 { return false }

        // R'.x must equal R.x from the signature
        let rxComputed = secpToInt(rPrimeAff.x)
        return rxComputed == rxLimbs
    }

    // MARK: - Batch Verify

    /// Batch verification using random linear combination.
    /// Returns true iff ALL signatures are valid.
    /// Uses Strauss/Shamir multi-scalar multiplication approach.
    ///
    /// Algorithm: check sum_i(a_i * s_i) * G - sum_i(a_i * e_i) * P_i - sum_i(a_i) * R_i == O
    /// where a_i are random 128-bit weights (a_0 = 1).
    public func batchVerify(messages: [[UInt8]], signatures: [SchnorrSignature],
                            publicKeys: [SchnorrPublicKey]) -> Bool {
        let n = messages.count
        precondition(signatures.count == n && publicKeys.count == n)
        if n == 0 { return true }
        if n == 1 { return verify(message: messages[0], signature: signatures[0], publicKey: publicKeys[0]) }

        // Parse all inputs first
        var pAffines = [SecpPointAffine]()
        var rAffines = [SecpPointAffine]()
        var sLimbsArr = [[UInt64]]()
        var eLimbsArr = [[UInt64]]()

        for i in 0..<n {
            guard let pAff = liftX(publicKeys[i].bytes) else { return false }
            guard let rAff = liftX(signatures[i].rx) else { return false }

            let sLimbs = bytesToLimbs(signatures[i].s)
            if gte256(sLimbs, SecpFr.N) { return false }

            let eHash = SchnorrEngine.taggedHash(
                tag: "BIP0340/challenge",
                data: signatures[i].rx + publicKeys[i].bytes + messages[i])
            var eLimbs = bytesToLimbs(eHash)
            reduceMod(&eLimbs, SecpFr.N)

            pAffines.append(pAff)
            rAffines.append(rAff)
            sLimbsArr.append(sLimbs)
            eLimbsArr.append(eLimbs)
        }

        // Generate random weights (a_0 = 1)
        var rng: UInt64 = UInt64(CFAbsoluteTimeGetCurrent().bitPattern) ^ 0xCAFEBABE
        var weights = [SecpFr]()
        weights.reserveCapacity(n)
        weights.append(SecpFr.one) // a_0 = 1

        for _ in 1..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let w0 = rng
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let w1 = rng
            weights.append(secpFrFromRaw([w0, w1, 0, 0]))
        }

        // Compute: sum(a_i * s_i) for G scalar
        var gScalar = SecpFr.zero
        var msmPoints = [SecpPointAffine]()
        var msmScalars = [[UInt32]]()

        for i in 0..<n {
            let siMont = secpFrFromRaw(sLimbsArr[i])
            let aiSi = secpFrMul(weights[i], siMont)
            gScalar = secpFrAdd(gScalar, aiSi)

            // -a_i * e_i for P_i
            let eiMont = secpFrFromRaw(eLimbsArr[i])
            let negAiEi = secpFrNeg(secpFrMul(weights[i], eiMont))
            let negAiEiRaw = secpFrToInt(negAiEi)
            msmPoints.append(pAffines[i])
            msmScalars.append(negAiEiRaw.flatMap { [UInt32($0 & 0xFFFFFFFF), UInt32($0 >> 32)] })

            // -a_i for R_i
            let negAi = secpFrNeg(weights[i])
            let negAiRaw = secpFrToInt(negAi)
            msmPoints.append(rAffines[i])
            msmScalars.append(negAiRaw.flatMap { [UInt32($0 & 0xFFFFFFFF), UInt32($0 >> 32)] })
        }

        // Add G term
        let gen = secp256k1Generator()
        let gScalarRaw = secpFrToInt(gScalar)
        msmPoints.append(gen)
        msmScalars.append(gScalarRaw.flatMap { [UInt32($0 & 0xFFFFFFFF), UInt32($0 >> 32)] })

        // Use GPU MSM if available, otherwise CPU
        if let msm = msmEngine {
            do {
                let result = try msm.msm(points: msmPoints, scalars: msmScalars)
                return secpPointIsIdentity(result)
            } catch {
                // Fall through to CPU
            }
        }

        // CPU fallback using Pippenger
        let result = cSecpPippengerMSM(points: msmPoints, scalars: msmScalars)
        return secpPointIsIdentity(result)
    }

    // MARK: - Helpers

    /// Lift an x-coordinate (32 bytes big-endian) to a curve point with even Y.
    /// Returns nil if x is not a valid coordinate on secp256k1.
    private func liftX(_ xBytes: [UInt8]) -> SecpPointAffine? {
        let xLimbs = bytesToLimbs(xBytes)
        if gte256(xLimbs, SecpFp.P) { return nil }

        let x = secpFromRawFp(xLimbs)
        let x2 = secpSqr(x)
        let x3 = secpMul(x2, x)
        let seven = secpFromInt(7)
        let rhs = secpAdd(x3, seven)

        guard let y = secpSqrt(rhs) else { return nil }

        let yInt = secpToInt(y)
        let yIsOdd = (yInt[0] & 1) == 1
        if yIsOdd {
            return SecpPointAffine(x: x, y: secpNeg(y))
        } else {
            return SecpPointAffine(x: x, y: y)
        }
    }
}

// MARK: - Byte/Limb Conversion Utilities

/// Convert 32-byte big-endian to 4x64-bit little-endian limbs
private func bytesToLimbs(_ bytes: [UInt8]) -> [UInt64] {
    precondition(bytes.count == 32)
    var limbs = [UInt64](repeating: 0, count: 4)
    for i in 0..<4 {
        let offset = 24 - i * 8  // big-endian: bytes[24..31] -> limbs[0]
        for j in 0..<8 {
            limbs[i] |= UInt64(bytes[offset + j]) << (56 - j * 8)
        }
    }
    return limbs
}

/// Convert 4x64-bit little-endian limbs to 32-byte big-endian
private func limbsToBytes(_ limbs: [UInt64]) -> [UInt8] {
    var bytes = [UInt8](repeating: 0, count: 32)
    for i in 0..<4 {
        let offset = 24 - i * 8
        for j in 0..<8 {
            bytes[offset + j] = UInt8((limbs[i] >> (56 - j * 8)) & 0xFF)
        }
    }
    return bytes
}

/// Convert Fp element to 32-byte big-endian
private func fpToBytes(_ fp: SecpFp) -> [UInt8] {
    let limbs = secpToInt(fp)
    return limbsToBytes(limbs)
}

/// Check if 256-bit value is zero
private func isZero256(_ a: [UInt64]) -> Bool {
    a[0] == 0 && a[1] == 0 && a[2] == 0 && a[3] == 0
}

/// Reduce a 256-bit value modulo n (in-place). Simple subtraction loop.
private func reduceMod(_ a: inout [UInt64], _ n: [UInt64]) {
    while gte256(a, n) {
        (a, _) = sub256(a, n)
    }
}
