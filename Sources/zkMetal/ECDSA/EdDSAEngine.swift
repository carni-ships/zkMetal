// EdDSA (Ed25519) Sign/Verify Engine
// Implements RFC 8032 Ed25519 signature scheme
//
// Sign: (R, S) where R = r*G, S = (r + H(R,A,M)*a) mod q
// Verify: [8][S]G = [8]R + [8][H(R,A,M)]A
//
// Uses SHA-512 for hashing (via CommonCrypto).

import Foundation
import NeonFieldOps
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
        // Convert clamped scalar to raw limbs
        var rawLimbs: [UInt64] = [0, 0, 0, 0]
        for i in 0..<4 {
            for j in 0..<8 {
                rawLimbs[i] |= UInt64(a[i * 8 + j]) << (j * 8)
            }
        }
        // Store in Montgomery form for signing math (mod q operations)
        self.scalar = ed25519FqFromRaw(rawLimbs)

        // Public key = a * G (use raw scalar for point multiplication)
        let gen = ed25519Generator()
        let genExt = ed25519PointFromAffine(gen)
        let pubExt = ed25519PointMulScalar(genExt, rawLimbs)
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

    /// Sign a message using Ed25519 (RFC 8032) — C-accelerated
    public func sign(message: [UInt8], secretKey: EdDSASecretKey) -> EdDSASignature {
        // r = SHA-512(nonceSeed || message) mod q
        let rHash = sha512(secretKey.nonceSeed + message)

        // Use C for hash-to-scalar
        var rMont = [UInt64](repeating: 0, count: 4)
        rHash.withUnsafeBufferPointer { hashPtr in
            ed25519_fq_from_bytes64(hashPtr.baseAddress!, &rMont)
        }

        // Get r as raw integer for scalar mul
        var rRaw = [UInt64](repeating: 0, count: 4)
        ed25519_fq_to_raw(&rMont, &rRaw)

        // R = r * G using C scalar mul
        let gen = ed25519Generator()
        var genExt = ed25519PointToExtLimbs(ed25519PointFromAffine(gen))
        var rPointExt = [UInt64](repeating: 0, count: 16)
        ed25519_eddsa_sign_compute_r(&genExt, &rRaw, &rPointExt)

        // Convert R to affine and encode
        var rAff = [UInt64](repeating: 0, count: 8)
        ed25519_point_to_affine(&rPointExt, &rAff)
        let rAffine = ed25519AffineFromLimbs(rAff)
        let rEncoded = ed25519PointEncode(rAffine)

        // k = SHA-512(R || A || M) mod q using C
        let kHash = sha512(rEncoded + secretKey.publicKey.encoded + message)
        var kMont = [UInt64](repeating: 0, count: 4)
        kHash.withUnsafeBufferPointer { hashPtr in
            ed25519_fq_from_bytes64(hashPtr.baseAddress!, &kMont)
        }

        // S = (r + k * a) mod q using C
        var aMont = secretKey.scalar.toLimbs()
        var sMont = [UInt64](repeating: 0, count: 4)
        ed25519_eddsa_sign_compute_s(&rMont, &kMont, &aMont, &sMont)

        // Convert S to bytes
        var sBytes = [UInt8](repeating: 0, count: 32)
        ed25519_fq_to_bytes(&sMont, &sBytes)

        return EdDSASignature(r: rEncoded, s: sBytes)
    }

    /// Verify an Ed25519 signature (RFC 8032) — C-accelerated with Shamir's trick
    /// Check: s*G == R + h*A (equivalently, s*G + h*(-A) == R)
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

        // k = SHA-512(R || A || M) mod q using C
        let kHash = sha512(signature.r + publicKey.encoded + message)
        var kMont = [UInt64](repeating: 0, count: 4)
        kHash.withUnsafeBufferPointer { hashPtr in
            ed25519_fq_from_bytes64(hashPtr.baseAddress!, &kMont)
        }
        var hRaw = [UInt64](repeating: 0, count: 4)
        ed25519_fq_to_raw(&kMont, &hRaw)

        // Convert points to C format (16 x uint64 extended coords)
        let gen = ed25519Generator()
        var genExt = ed25519PointToExtLimbs(ed25519PointFromAffine(gen))
        var rExt = ed25519PointToExtLimbs(ed25519PointFromAffine(rPoint))
        var aExt = ed25519PointToExtLimbs(ed25519PointFromAffine(publicKey.point))

        // C verify: Shamir's trick s*G + h*(-A) == R
        let valid = ed25519_eddsa_verify(&genExt, &sLimbs, &rExt, &hRaw, &aExt)
        return valid != 0
    }

    /// Batch verify N signatures using random linear combination — C-accelerated
    /// Returns true iff ALL signatures are valid (with negligible false positive probability ~2^-128)
    ///
    /// Algorithm: choose random z_i, check:
    ///   [sum(z_i * S_i)]G - sum(z_i * R_i) - sum(z_i * k_i * A_i) == identity
    ///
    /// Uses the C Pippenger MSM for the multi-scalar multiplication.
    public func batchVerify(signatures: [EdDSASignature], messages: [[UInt8]],
                            publicKeys: [EdDSAPublicKey]) -> Bool {
        let n = signatures.count
        precondition(messages.count == n && publicKeys.count == n)
        if n == 0 { return true }
        if n == 1 { return verify(signature: signatures[0], message: messages[0], publicKey: publicKeys[0]) }

        // Generate random 128-bit weights using C Fq ops
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

        // Build MSM: we need 2*n+1 points and scalars
        // Points:  G, R_0, ..., R_{n-1}, A_0, ..., A_{n-1}
        // Scalars: sum(z_i*S_i), -z_0, ..., -z_{n-1}, -z_0*k_0, ..., -z_{n-1}*k_{n-1}
        //
        // Final: sum(z_i*S_i)*G - sum(z_i*R_i) - sum(z_i*k_i*A_i) == identity
        //
        // For efficiency, collect all points and scalars for the C Pippenger MSM.

        let totalPoints = 2 * n + 1
        var affPoints = [UInt64](repeating: 0, count: totalPoints * 8)
        var scalars32 = [UInt32](repeating: 0, count: totalPoints * 8)

        // Point 0: Generator G
        let gen = ed25519Generator()
        let genFp = ed25519PointToAffineLimbs(gen)
        for j in 0..<8 { affPoints[j] = genFp[j] }

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

            // k_i = H(R_i || A_i || M_i)
            let kHash = sha512(signatures[i].r + publicKeys[i].encoded + messages[i])
            var kMont = [UInt64](repeating: 0, count: 4)
            kHash.withUnsafeBufferPointer { hashPtr in
                ed25519_fq_from_bytes64(hashPtr.baseAddress!, &kMont)
            }

            // Decode R_i and set as point (1 + i)
            guard let rAff = ed25519PointDecode(signatures[i].r) else { return false }
            let rLimbs = ed25519PointToAffineLimbs(rAff)
            let rIdx = (1 + i) * 8
            for j in 0..<8 { affPoints[rIdx + j] = rLimbs[j] }

            // Scalar for R_i: negate z_i (we subtract R_i contribution)
            // In Fq: neg(z_i) = q - z_i
            var negZi = [UInt64](repeating: 0, count: 4)
            var zeroFq = [UInt64](repeating: 0, count: 4)
            ed25519_fq_sub(&zeroFq, &weights[i], &negZi)
            var negZiRaw = [UInt64](repeating: 0, count: 4)
            ed25519_fq_to_raw(&negZi, &negZiRaw)
            let rSIdx = (1 + i) * 8
            for j in 0..<4 {
                scalars32[rSIdx + j * 2] = UInt32(negZiRaw[j] & 0xFFFFFFFF)
                scalars32[rSIdx + j * 2 + 1] = UInt32(negZiRaw[j] >> 32)
            }

            // Set A_i as point (1 + n + i)
            let aLimbs = ed25519PointToAffineLimbs(publicKeys[i].point)
            let aIdx = (1 + n + i) * 8
            for j in 0..<8 { affPoints[aIdx + j] = aLimbs[j] }

            // Scalar for A_i: negate z_i * k_i
            var zkMont = [UInt64](repeating: 0, count: 4)
            ed25519_fq_mul(&weights[i], &kMont, &zkMont)
            var negZk = [UInt64](repeating: 0, count: 4)
            ed25519_fq_sub(&zeroFq, &zkMont, &negZk)
            var negZkRaw = [UInt64](repeating: 0, count: 4)
            ed25519_fq_to_raw(&negZk, &negZkRaw)
            let aSIdx = (1 + n + i) * 8
            for j in 0..<4 {
                scalars32[aSIdx + j * 2] = UInt32(negZkRaw[j] & 0xFFFFFFFF)
                scalars32[aSIdx + j * 2 + 1] = UInt32(negZkRaw[j] >> 32)
            }
        }

        // Set G scalar (index 0)
        var gScalarRaw = [UInt64](repeating: 0, count: 4)
        ed25519_fq_to_raw(&gScalarMont, &gScalarRaw)
        for j in 0..<4 {
            scalars32[j * 2] = UInt32(gScalarRaw[j] & 0xFFFFFFFF)
            scalars32[j * 2 + 1] = UInt32(gScalarRaw[j] >> 32)
        }

        // Run C Pippenger MSM
        var result = [UInt64](repeating: 0, count: 16)
        ed25519_pippenger_msm(&affPoints, &scalars32, Int32(totalPoints), &result)

        // Check if result is identity
        let resultExt = ed25519ExtFromLimbs(result)
        return ed25519PointIsIdentity(resultExt)
    }
}

// MARK: - Swift <-> C limb conversion helpers

/// Convert Swift extended point to 16 x UInt64 limb array for C
func ed25519PointToExtLimbs(_ p: Ed25519PointExtended) -> [UInt64] {
    [p.x.v.0, p.x.v.1, p.x.v.2, p.x.v.3,
     p.y.v.0, p.y.v.1, p.y.v.2, p.y.v.3,
     p.z.v.0, p.z.v.1, p.z.v.2, p.z.v.3,
     p.t.v.0, p.t.v.1, p.t.v.2, p.t.v.3]
}

/// Convert 16 x UInt64 limb array back to Swift extended point
func ed25519ExtFromLimbs(_ l: [UInt64]) -> Ed25519PointExtended {
    Ed25519PointExtended(
        x: Ed25519Fp(v: (l[0], l[1], l[2], l[3])),
        y: Ed25519Fp(v: (l[4], l[5], l[6], l[7])),
        z: Ed25519Fp(v: (l[8], l[9], l[10], l[11])),
        t: Ed25519Fp(v: (l[12], l[13], l[14], l[15]))
    )
}

/// Convert Swift affine point to 8 x UInt64 limb array for C
func ed25519PointToAffineLimbs(_ p: Ed25519PointAffine) -> [UInt64] {
    [p.x.v.0, p.x.v.1, p.x.v.2, p.x.v.3,
     p.y.v.0, p.y.v.1, p.y.v.2, p.y.v.3]
}

/// Convert 8 x UInt64 limb array to Swift affine point
func ed25519AffineFromLimbs(_ l: [UInt64]) -> Ed25519PointAffine {
    Ed25519PointAffine(
        x: Ed25519Fp(v: (l[0], l[1], l[2], l[3])),
        y: Ed25519Fp(v: (l[4], l[5], l[6], l[7]))
    )
}

// MARK: - SHA-512 helper

public func sha512(_ data: [UInt8]) -> [UInt8] {
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
