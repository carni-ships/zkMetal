// EdDSA (Ed25519) Sign/Verify Engine
// Implements RFC 8032 Ed25519 signature scheme
//
// Sign: (R, S) where R = r*G, S = (r + H(R,A,M)*a) mod q
// Verify: [8][S]G = [8]R + [8][H(R,A,M)]A
//
// Uses SHA-512 for hashing (via CommonCrypto).

import Foundation
#if canImport(CryptoKit)
import CryptoKit
#endif

// MARK: - EdDSA Types

public struct EdDSAPublicKey {
    public let point: Ed25519PointAffine
    public let encoded: [UInt8]  // 32-byte compressed point

    public init(point: Ed25519PointAffine) {
        self.point = point
        self.encoded = ed25519PointEncode(point)
    }

    public init?(encoded: [UInt8]) {
        guard encoded.count == 32,
              let pt = ed25519PointDecode(encoded) else { return nil }
        self.point = pt
        self.encoded = encoded
    }
}

public struct EdDSASecretKey {
    public let seed: [UInt8]  // 32-byte seed
    public let scalar: Ed25519Fq  // Derived scalar a
    public let nonceSeed: [UInt8]  // Lower 32 bytes of SHA-512(seed) for nonce
    public let publicKey: EdDSAPublicKey

    public init(seed: [UInt8]) {
        precondition(seed.count == 32)
        self.seed = seed
        let h = sha512(seed)
        // Clamp: clear lowest 3 bits, set bit 254, clear bit 255
        var a = Array(h[0..<32])
        a[0] &= 248
        a[31] &= 127
        a[31] |= 64
        self.nonceSeed = Array(h[32..<64])
        // Convert clamped scalar to field element
        var limbs: [UInt64] = [0, 0, 0, 0]
        for i in 0..<4 {
            for j in 0..<8 {
                limbs[i] |= UInt64(a[i * 8 + j]) << (j * 8)
            }
        }
        self.scalar = ed25519FqFromRaw(limbs)

        // Public key = a * G
        let gen = ed25519Generator()
        let genExt = ed25519PointFromAffine(gen)
        let pubExt = ed25519PointMulScalar(genExt, ed25519FqToInt(self.scalar))
        let pubAff = ed25519PointToAffine(pubExt)
        self.publicKey = EdDSAPublicKey(point: pubAff)
    }
}

public struct EdDSASignature {
    public let r: [UInt8]  // 32-byte encoded R point
    public let s: [UInt8]  // 32-byte scalar S

    public init(r: [UInt8], s: [UInt8]) {
        precondition(r.count == 32 && s.count == 32)
        self.r = r
        self.s = s
    }

    public func toBytes() -> [UInt8] { r + s }

    public init?(fromBytes bytes: [UInt8]) {
        guard bytes.count == 64 else { return nil }
        self.r = Array(bytes[0..<32])
        self.s = Array(bytes[32..<64])
    }
}

// MARK: - EdDSA Engine

public class EdDSAEngine {
    public static let version = Versions.eddsa

    public init() {}

    /// Sign a message using Ed25519 (RFC 8032)
    public func sign(message: [UInt8], secretKey: EdDSASecretKey) -> EdDSASignature {
        // r = SHA-512(nonceSeed || message) mod q
        let rHash = sha512(secretKey.nonceSeed + message)
        let r = ed25519FqFromBytes64(rHash)

        // R = r * G
        let gen = ed25519Generator()
        let genExt = ed25519PointFromAffine(gen)
        let rPoint = ed25519PointMulScalar(genExt, ed25519FqToInt(r))
        let rAff = ed25519PointToAffine(rPoint)
        let rEncoded = ed25519PointEncode(rAff)

        // k = SHA-512(R || A || M) mod q
        let kHash = sha512(rEncoded + secretKey.publicKey.encoded + message)
        let k = ed25519FqFromBytes64(kHash)

        // S = (r + k * a) mod q
        let s = ed25519FqAdd(r, ed25519FqMul(k, secretKey.scalar))
        let sBytes = ed25519FqToBytes(s)

        return EdDSASignature(r: rEncoded, s: sBytes)
    }

    /// Verify an Ed25519 signature (RFC 8032)
    /// Check: [S]G = R + [k]A
    public func verify(signature: EdDSASignature, message: [UInt8], publicKey: EdDSAPublicKey) -> Bool {
        // Decode R
        guard let rPoint = ed25519PointDecode(signature.r) else { return false }

        // Decode S as scalar (must be < q)
        var sLimbs: [UInt64] = [0, 0, 0, 0]
        for i in 0..<4 {
            for j in 0..<8 {
                sLimbs[i] |= UInt64(signature.s[i * 8 + j]) << (j * 8)
            }
        }
        // S must be < q
        if ed25519FqGte(sLimbs, Ed25519Fq.Q) { return false }

        // k = SHA-512(R || A || M) mod q
        let kHash = sha512(signature.r + publicKey.encoded + message)
        let k = ed25519FqFromBytes64(kHash)

        // Verify: [S]G == R + [k]A
        let gen = ed25519Generator()
        let genExt = ed25519PointFromAffine(gen)
        let sG = ed25519PointMulScalar(genExt, sLimbs)

        let rExt = ed25519PointFromAffine(rPoint)
        let aExt = ed25519PointFromAffine(publicKey.point)
        let kA = ed25519PointMulScalar(aExt, ed25519FqToInt(k))
        let rPlusKA = ed25519PointAdd(rExt, kA)

        // Compare by converting to affine
        let lhs = ed25519PointToAffine(sG)
        let rhs = ed25519PointToAffine(rPlusKA)
        return ed25519FpToInt(lhs.x) == ed25519FpToInt(rhs.x) &&
               ed25519FpToInt(lhs.y) == ed25519FpToInt(rhs.y)
    }

    /// Batch verify N signatures using random linear combination
    /// Returns true iff ALL signatures are valid (with negligible false positive probability ~2^-128)
    ///
    /// Algorithm: choose random z_i, check:
    ///   [sum(z_i * S_i)]G - sum(z_i * R_i) - sum(z_i * k_i * A_i) == identity
    public func batchVerify(signatures: [EdDSASignature], messages: [[UInt8]],
                            publicKeys: [EdDSAPublicKey]) -> Bool {
        let n = signatures.count
        precondition(messages.count == n && publicKeys.count == n)
        if n == 0 { return true }

        // Generate random 128-bit weights
        var rng: UInt64 = UInt64(CFAbsoluteTimeGetCurrent().bitPattern) ^ 0xDEADBEEFCAFEBABE
        var weights = [Ed25519Fq]()
        for _ in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let w0 = rng
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let w1 = rng
            weights.append(ed25519FqFromRaw([w0, w1, 0, 0]))
        }

        // Accumulate: sum(z_i * S_i) for G scalar
        var gScalar = Ed25519Fq.zero

        // Sum up: result = sum(z_i * S_i) * G - sum(z_i * R_i) - sum(z_i * k_i * A_i)
        var rAccum = ed25519PointIdentity()
        var aAccum = ed25519PointIdentity()

        for i in 0..<n {
            // Decode S_i
            var sLimbs: [UInt64] = [0, 0, 0, 0]
            for li in 0..<4 {
                for j in 0..<8 {
                    sLimbs[li] |= UInt64(signatures[i].s[li * 8 + j]) << (j * 8)
                }
            }
            if ed25519FqGte(sLimbs, Ed25519Fq.Q) { return false }
            let sMont = ed25519FqFromRaw(sLimbs)

            // k_i = H(R_i || A_i || M_i)
            let kHash = sha512(signatures[i].r + publicKeys[i].encoded + messages[i])
            let k = ed25519FqFromBytes64(kHash)

            // Decode R_i
            guard let rPoint = ed25519PointDecode(signatures[i].r) else { return false }

            // Accumulate: gScalar += z_i * S_i
            gScalar = ed25519FqAdd(gScalar, ed25519FqMul(weights[i], sMont))

            // rAccum += z_i * R_i
            let rExt = ed25519PointFromAffine(rPoint)
            let ziRi = ed25519PointMulScalar(rExt, ed25519FqToInt(weights[i]))
            rAccum = ed25519PointAdd(rAccum, ziRi)

            // aAccum += z_i * k_i * A_i
            let zk = ed25519FqMul(weights[i], k)
            let aExt = ed25519PointFromAffine(publicKeys[i].point)
            let zkA = ed25519PointMulScalar(aExt, ed25519FqToInt(zk))
            aAccum = ed25519PointAdd(aAccum, zkA)
        }

        // Final check: gScalar * G - rAccum - aAccum == identity
        let gen = ed25519Generator()
        let genExt = ed25519PointFromAffine(gen)
        let sG = ed25519PointMulScalar(genExt, ed25519FqToInt(gScalar))
        let result = ed25519PointAdd(sG, ed25519PointNeg(ed25519PointAdd(rAccum, aAccum)))

        return ed25519PointIsIdentity(result)
    }
}

// MARK: - SHA-512 helper

func sha512(_ data: [UInt8]) -> [UInt8] {
    #if canImport(CryptoKit)
    let digest = SHA512.hash(data: data)
    return Array(digest)
    #else
    // Fallback: use CommonCrypto
    var hash = [UInt8](repeating: 0, count: 64)
    data.withUnsafeBytes { ptr in
        CC_SHA512(ptr.baseAddress, CC_LONG(data.count), &hash)
    }
    return hash
    #endif
}
