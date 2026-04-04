// EdDSA over BabyJubjub (used by Circom's eddsa-babyjubjub)
//
// Sign: (R, S) where R = r*G, S = r + H(R,A,M)*sk (mod q)
// Verify: S*G == R + H(R,A,M)*A
//
// Uses Poseidon2 hash for the Fiat-Shamir challenge (standard in Circom).
// All arithmetic is in BN254 Fr (the base field of BabyJubjub).

import Foundation
#if canImport(CryptoKit)
import CryptoKit
#endif

// MARK: - Key Types

public struct BJJPublicKey {
    public let point: BJJPointAffine
    public let encoded: [UInt8]  // 64-byte uncompressed (x, y)

    public init(point: BJJPointAffine) {
        self.point = point
        self.encoded = bjjPointEncode(point)
    }

    public init?(encoded: [UInt8]) {
        guard let pt = bjjPointDecode(encoded) else { return nil }
        self.point = pt
        self.encoded = encoded
    }
}

public struct BJJSecretKey {
    public let seed: [UInt8]  // 32-byte seed
    public let scalar: [UInt64]  // Raw scalar (mod subgroup order q)
    public let publicKey: BJJPublicKey

    public init(seed: [UInt8]) {
        precondition(seed.count == 32)
        self.seed = seed

        // Derive scalar from seed using SHA-256 and reducing mod q
        let h = sha256Bytes(seed)
        var rawLimbs: [UInt64] = [0, 0, 0, 0]
        for i in 0..<4 {
            for j in 0..<8 {
                let byteIdx = i * 8 + j
                if byteIdx < h.count {
                    rawLimbs[i] |= UInt64(h[byteIdx]) << (j * 8)
                }
            }
        }
        // Reduce mod subgroup order q
        self.scalar = BJJSecretKey.reduceMod(rawLimbs, BJJ_SUBGROUP_ORDER)

        // Public key = scalar * G
        let gen = bjjGenerator()
        let genExt = bjjPointFromAffine(gen)
        let pubExt = bjjPointMulScalar(genExt, self.scalar)
        let pubAff = bjjPointToAffine(pubExt)
        self.publicKey = BJJPublicKey(point: pubAff)
    }

    /// Create from known scalar (for testing)
    public init(scalarValue: UInt64) {
        self.seed = [UInt8](repeating: 0, count: 32)
        self.scalar = [scalarValue, 0, 0, 0]
        let gen = bjjGenerator()
        let genExt = bjjPointFromAffine(gen)
        let pubExt = bjjPointMulScalar(genExt, self.scalar)
        let pubAff = bjjPointToAffine(pubExt)
        self.publicKey = BJJPublicKey(point: pubAff)
    }

    /// Simple modular reduction: if a >= m, subtract m repeatedly (for values near m)
    /// For proper reduction of arbitrary values, use Barrett or Montgomery
    static func reduceMod(_ a: [UInt64], _ m: [UInt64]) -> [UInt64] {
        var r = a
        while gte256(r, m) {
            (r, _) = sub256(r, m)
        }
        return r
    }
}

public struct BJJSignature {
    public let rx: Fr  // R point x-coordinate (Montgomery form)
    public let ry: Fr  // R point y-coordinate (Montgomery form)
    public let s: [UInt64]  // S scalar (raw, non-Montgomery)

    public init(rx: Fr, ry: Fr, s: [UInt64]) {
        self.rx = rx
        self.ry = ry
        self.s = s
    }
}

// MARK: - EdDSA Engine for BabyJubjub

public class BabyJubjubEdDSA {
    public static let version = Versions.bjjEdDSA

    public init() {}

    /// Sign a message (given as Fr elements) using EdDSA over BabyJubjub
    /// Uses Poseidon2 for the Fiat-Shamir challenge
    public func sign(message: [Fr], secretKey: BJJSecretKey) -> BJJSignature {
        // r = Poseidon2(sk_scalar_as_Fr, message...) — deterministic nonce
        let skFr = frFromInt(secretKey.scalar[0])  // simplified; full scalar for production
        var nonceInput = [skFr]
        nonceInput.append(contentsOf: message)
        let rHash = poseidon2HashMany(nonceInput)
        // Reduce hash to a scalar mod q
        let rLimbs = frToInt(rHash)
        let rScalar = BJJSecretKey.reduceMod(rLimbs, BJJ_SUBGROUP_ORDER)

        // R = r * G
        let gen = bjjGenerator()
        let genExt = bjjPointFromAffine(gen)
        let rPoint = bjjPointMulScalar(genExt, rScalar)
        let rAff = bjjPointToAffine(rPoint)

        // k = Poseidon2(R.x, R.y, A.x, A.y, message...)
        var challengeInput = [rAff.x, rAff.y,
                              secretKey.publicKey.point.x, secretKey.publicKey.point.y]
        challengeInput.append(contentsOf: message)
        let kHash = poseidon2HashMany(challengeInput)
        let kLimbs = frToInt(kHash)
        let kScalar = BJJSecretKey.reduceMod(kLimbs, BJJ_SUBGROUP_ORDER)

        // S = (r + k * sk) mod q
        let s = addModQ(rScalar, mulModQ(kScalar, secretKey.scalar))

        return BJJSignature(rx: rAff.x, ry: rAff.y, s: s)
    }

    /// Verify an EdDSA signature over BabyJubjub
    /// Check: S*G == R + H(R,A,M)*A
    public func verify(signature: BJJSignature, message: [Fr], publicKey: BJJPublicKey) -> Bool {
        // Reconstruct R
        let rAff = BJJPointAffine(x: signature.rx, y: signature.ry)
        guard bjjPointOnCurve(rAff) else { return false }

        // Check S < q
        if gte256(signature.s, BJJ_SUBGROUP_ORDER) { return false }

        // k = Poseidon2(R.x, R.y, A.x, A.y, message...)
        var challengeInput = [rAff.x, rAff.y,
                              publicKey.point.x, publicKey.point.y]
        challengeInput.append(contentsOf: message)
        let kHash = poseidon2HashMany(challengeInput)
        let kLimbs = frToInt(kHash)
        let kScalar = BJJSecretKey.reduceMod(kLimbs, BJJ_SUBGROUP_ORDER)

        // LHS: S * G
        let gen = bjjGenerator()
        let genExt = bjjPointFromAffine(gen)
        let sG = bjjPointMulScalar(genExt, signature.s)

        // RHS: R + k * A
        let rExt = bjjPointFromAffine(rAff)
        let aExt = bjjPointFromAffine(publicKey.point)
        let kA = bjjPointMulScalar(aExt, kScalar)
        let rPlusKA = bjjPointAdd(rExt, kA)

        // Compare
        let lhs = bjjPointToAffine(sG)
        let rhs = bjjPointToAffine(rPlusKA)
        return frToInt(lhs.x) == frToInt(rhs.x) &&
               frToInt(lhs.y) == frToInt(rhs.y)
    }

    /// Batch verify N signatures using random linear combination
    public func batchVerify(signatures: [BJJSignature], messages: [[Fr]],
                            publicKeys: [BJJPublicKey]) -> Bool {
        let n = signatures.count
        precondition(messages.count == n && publicKeys.count == n)
        if n == 0 { return true }
        if n == 1 { return verify(signature: signatures[0], message: messages[0], publicKey: publicKeys[0]) }

        // Random weights for batch verification
        var rng: UInt64 = UInt64(CFAbsoluteTimeGetCurrent().bitPattern) ^ 0xDEADBEEFCAFEBABE
        var weights = [[UInt64]]()
        for _ in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let w0 = rng
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let w1 = rng
            weights.append([w0, w1, 0, 0])
        }

        // Check: sum(w_i * S_i) * G == sum(w_i * R_i) + sum(w_i * k_i * A_i)
        var gScalar: [UInt64] = [0, 0, 0, 0]
        var rAccum = bjjPointIdentity()
        var aAccum = bjjPointIdentity()

        for i in 0..<n {
            let sig = signatures[i]

            // Validate
            let rAff = BJJPointAffine(x: sig.rx, y: sig.ry)
            guard bjjPointOnCurve(rAff) else { return false }
            if gte256(sig.s, BJJ_SUBGROUP_ORDER) { return false }

            // k_i
            var challengeInput = [rAff.x, rAff.y,
                                  publicKeys[i].point.x, publicKeys[i].point.y]
            challengeInput.append(contentsOf: messages[i])
            let kHash = poseidon2HashMany(challengeInput)
            let kLimbs = frToInt(kHash)
            let kScalar = BJJSecretKey.reduceMod(kLimbs, BJJ_SUBGROUP_ORDER)

            // gScalar += w_i * S_i (mod q)
            let wS = mulModQ(weights[i], sig.s)
            gScalar = addModQ(gScalar, wS)

            // rAccum += w_i * R_i
            let rExt = bjjPointFromAffine(rAff)
            let wR = bjjPointMulScalar(rExt, weights[i])
            rAccum = bjjPointAdd(rAccum, wR)

            // aAccum += w_i * k_i * A_i
            let wk = mulModQ(weights[i], kScalar)
            let aExt = bjjPointFromAffine(publicKeys[i].point)
            let wkA = bjjPointMulScalar(aExt, wk)
            aAccum = bjjPointAdd(aAccum, wkA)
        }

        // Final: gScalar * G == rAccum + aAccum
        let gen = bjjGenerator()
        let genExt = bjjPointFromAffine(gen)
        let sG = bjjPointMulScalar(genExt, gScalar)
        let rhs = bjjPointAdd(rAccum, aAccum)

        let lhsAff = bjjPointToAffine(sG)
        let rhsAff = bjjPointToAffine(rhs)
        return frToInt(lhsAff.x) == frToInt(rhsAff.x) &&
               frToInt(lhsAff.y) == frToInt(rhsAff.y)
    }
}

// MARK: - Modular arithmetic on subgroup order q

/// Multiplication mod q via double-and-add on scalars
/// Both a, b < q < 2^252. Uses O(252) additions.
func mulModQ(_ a: [UInt64], _ b: [UInt64]) -> [UInt64] {
    var result: [UInt64] = [0, 0, 0, 0]
    var acc = a  // will be doubled each iteration
    // Ensure acc < q
    while gte256(acc, BJJ_SUBGROUP_ORDER) {
        (acc, _) = sub256(acc, BJJ_SUBGROUP_ORDER)
    }
    for i in 0..<4 {
        var word = b[i]
        for _ in 0..<64 {
            if word & 1 == 1 {
                result = addModQ(result, acc)
            }
            acc = addModQ(acc, acc)  // double
            word >>= 1
        }
    }
    return result
}

/// Addition mod q
func addModQ(_ a: [UInt64], _ b: [UInt64]) -> [UInt64] {
    var (result, carry) = add256(a, b)
    if carry != 0 || gte256(result, BJJ_SUBGROUP_ORDER) {
        (result, _) = sub256(result, BJJ_SUBGROUP_ORDER)
    }
    return result
}
