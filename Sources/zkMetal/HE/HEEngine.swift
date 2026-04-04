// HEEngine — GPU-accelerated BFV Homomorphic Encryption
// Uses RNS representation with GPU batch NTT for polynomial arithmetic.
// Core operations: keygen, encrypt, decrypt, add, multiply.
//
// BFV scheme: ciphertext = (c0, c1) where each ci is an RNS polynomial.
// Homomorphic add: component-wise polynomial addition
// Homomorphic multiply: tensor product + relinearization

import Foundation
import Metal

// MARK: - HE Parameter Set

public struct HEParams {
    public let logN: Int             // polynomial degree = 2^logN
    public let moduli: [UInt32]      // RNS moduli chain q1, ..., qL
    public let plainModulus: UInt32   // plaintext modulus t (for BFV)

    public var degree: Int { 1 << logN }
    public var numLimbs: Int { moduli.count }

    /// Standard BFV parameters for 128-bit security
    /// N=4096, L=3 moduli (~90-bit Q), t=65537
    public static let bfv128_N4096 = HEParams(
        logN: 12,
        moduli: [],  // will be filled by heVerifiedPrimes
        plainModulus: 65537
    )

    public init(logN: Int, moduli: [UInt32], plainModulus: UInt32) {
        self.logN = logN
        if moduli.isEmpty {
            self.moduli = heVerifiedPrimes(count: 3, logN: logN)
        } else {
            self.moduli = moduli
        }
        self.plainModulus = plainModulus
    }
}

// MARK: - Key Types

public struct SecretKey {
    /// Secret polynomial s with small coefficients ({-1, 0, 1})
    public let poly: RNSPoly  // in NTT domain
}

public struct PublicKey {
    /// pk = (pk0, pk1) where pk0 = -(a*s + e), pk1 = a
    public let pk0: RNSPoly  // in NTT domain
    public let pk1: RNSPoly  // in NTT domain
}

public struct RelinKey {
    /// Relinearization key for squaring the secret key
    public let rlk0: RNSPoly  // in NTT domain
    public let rlk1: RNSPoly  // in NTT domain
}

public struct Ciphertext {
    /// BFV ciphertext: (c0, c1, ...) — usually 2 components
    public var components: [RNSPoly]  // in NTT domain

    public var c0: RNSPoly { components[0] }
    public var c1: RNSPoly { components[1] }
}

// MARK: - HE Engine

public class HEEngine {
    public let params: HEParams
    public let nttEngine: RNSNTTEngine

    private var rng: UInt64 = 0xCAFE_BABE_DEAD_BEEF

    public init(params: HEParams) throws {
        self.params = params
        self.nttEngine = try RNSNTTEngine(logN: params.logN, moduli: params.moduli)
    }

    // MARK: - Key Generation

    /// Generate secret key (ternary distribution: coefficients in {-1, 0, 1})
    public func keyGen() throws -> (PublicKey, SecretKey, RelinKey) {
        let n = params.degree
        let moduli = params.moduli

        // Secret key: small polynomial with ternary coefficients
        var sCoeffs = [Int64](repeating: 0, count: n)
        for i in 0..<n {
            let r = nextRandom() % 4
            if r == 0 { sCoeffs[i] = -1 }
            else if r == 1 { sCoeffs[i] = 1 }
            // else 0
        }
        var sPoly = RNSPoly(coefficients: sCoeffs, moduli: moduli)
        try nttEngine.forwardNTT(&sPoly)
        let sk = SecretKey(poly: sPoly)

        // Public key: pk = (-(a*s + e), a)
        let a = randomRNSPoly()
        var aNTT = a
        try nttEngine.forwardNTT(&aNTT)

        // e: small error polynomial (discrete Gaussian, approximated by centered binomial)
        let e = smallErrorPoly()
        var eNTT = e
        try nttEngine.forwardNTT(&eNTT)

        // pk0 = -(a*s + e) in NTT domain
        let as_prod = try nttEngine.multiply(aNTT, sk.poly)
        let as_plus_e = try nttEngine.add(as_prod, eNTT)
        // Negate: for each limb, negate mod qi
        var pk0 = as_plus_e
        for li in 0..<moduli.count {
            let q = moduli[li]
            for j in 0..<n {
                let v = pk0.limbs[li][j]
                pk0.limbs[li][j] = v == 0 ? 0 : q - v
            }
        }

        let pk = PublicKey(pk0: pk0, pk1: aNTT)

        // Relinearization key (simplified): rlk encrypts s^2
        // rlk = (-(a'*s + e') + s^2, a')
        let a2 = randomRNSPoly()
        var a2NTT = a2
        try nttEngine.forwardNTT(&a2NTT)
        let e2 = smallErrorPoly()
        var e2NTT = e2
        try nttEngine.forwardNTT(&e2NTT)

        let s2 = try nttEngine.multiply(sk.poly, sk.poly)  // s^2 in NTT domain
        let a2s = try nttEngine.multiply(a2NTT, sk.poly)
        let a2s_e2 = try nttEngine.add(a2s, e2NTT)
        var rlk0 = a2s_e2
        for li in 0..<moduli.count {
            let q = moduli[li]
            for j in 0..<n {
                let v = rlk0.limbs[li][j]
                rlk0.limbs[li][j] = v == 0 ? 0 : q - v
            }
        }
        // rlk0 = -(a'*s + e') + s^2
        rlk0 = try nttEngine.add(rlk0, s2)

        let rlk = RelinKey(rlk0: rlk0, rlk1: a2NTT)

        return (pk, sk, rlk)
    }

    // MARK: - Encryption

    /// Encrypt a plaintext polynomial (BFV scheme)
    /// plaintext coefficients should be in [0, plainModulus)
    public func encrypt(plaintext: [Int64], pk: PublicKey) throws -> Ciphertext {
        let n = params.degree
        let moduli = params.moduli
        let t = params.plainModulus

        // u: small random polynomial (ternary)
        var uCoeffs = [Int64](repeating: 0, count: n)
        for i in 0..<n {
            let r = nextRandom() % 4
            if r == 0 { uCoeffs[i] = -1 }
            else if r == 1 { uCoeffs[i] = 1 }
        }
        var u = RNSPoly(coefficients: uCoeffs, moduli: moduli)
        try nttEngine.forwardNTT(&u)

        // e1, e2: small error polynomials
        let e1 = smallErrorPoly()
        var e1NTT = e1
        try nttEngine.forwardNTT(&e1NTT)
        let e2 = smallErrorPoly()
        var e2NTT = e2
        try nttEngine.forwardNTT(&e2NTT)

        // Encode plaintext: delta * m where delta = floor(Q/t) for each RNS modulus
        // In RNS, delta_i = floor(qi/t) for each qi
        var mPoly = RNSPoly(degree: n, moduli: moduli)
        for li in 0..<moduli.count {
            let q = moduli[li]
            let delta = q / t  // floor(qi/t)
            for j in 0..<min(n, plaintext.count) {
                var coeff = plaintext[j] % Int64(t)
                if coeff < 0 { coeff += Int64(t) }
                mPoly.limbs[li][j] = UInt32((UInt64(delta) * UInt64(coeff)) % UInt64(q))
            }
        }
        var mNTT = mPoly
        try nttEngine.forwardNTT(&mNTT)

        // c0 = pk0 * u + e1 + delta*m
        let pk0u = try nttEngine.multiply(pk.pk0, u)
        let pk0u_e1 = try nttEngine.add(pk0u, e1NTT)
        let c0 = try nttEngine.add(pk0u_e1, mNTT)

        // c1 = pk1 * u + e2
        let pk1u = try nttEngine.multiply(pk.pk1, u)
        let c1 = try nttEngine.add(pk1u, e2NTT)

        return Ciphertext(components: [c0, c1])
    }

    // MARK: - Decryption

    /// Decrypt a BFV ciphertext
    public func decrypt(ciphertext ct: Ciphertext, sk: SecretKey) throws -> [Int64] {
        let n = params.degree
        let moduli = params.moduli
        let t = params.plainModulus

        // Compute c0 + c1 * s (in NTT domain)
        let c1s = try nttEngine.multiply(ct.c1, sk.poly)
        var noisy = try nttEngine.add(ct.c0, c1s)

        // Inverse NTT to get coefficient form
        try nttEngine.inverseNTT(&noisy)

        // Decode: round(t * noisy / qi) mod t for each coefficient
        // Use first modulus for simplicity (proper CRT would combine all)
        let q = UInt64(moduli[0])
        let halfQ = q / 2

        var result = [Int64](repeating: 0, count: n)
        for j in 0..<n {
            let v = UInt64(noisy.limbs[0][j])
            // Compute round(t * v / q) mod t
            // = floor((t * v + q/2) / q) mod t
            let scaled = (UInt64(t) * v + halfQ) / q
            result[j] = Int64(scaled % UInt64(t))
        }
        return result
    }

    // MARK: - Homomorphic Operations

    /// Homomorphic addition: component-wise add
    public func add(_ a: Ciphertext, _ b: Ciphertext) throws -> Ciphertext {
        precondition(a.components.count == b.components.count)
        var result = [RNSPoly]()
        for i in 0..<a.components.count {
            let sum = try nttEngine.add(a.components[i], b.components[i])
            result.append(sum)
        }
        return Ciphertext(components: result)
    }

    /// Homomorphic multiplication (without relinearization)
    /// Result is a degree-3 ciphertext (3 components)
    public func multiplyRaw(_ a: Ciphertext, _ b: Ciphertext) throws -> Ciphertext {
        // Tensor product: (a0, a1) x (b0, b1) = (a0*b0, a0*b1 + a1*b0, a1*b1)
        let d0 = try nttEngine.multiply(a.c0, b.c0)
        let a0b1 = try nttEngine.multiply(a.c0, b.c1)
        let a1b0 = try nttEngine.multiply(a.c1, b.c0)
        let d1 = try nttEngine.add(a0b1, a1b0)
        let d2 = try nttEngine.multiply(a.c1, b.c1)

        return Ciphertext(components: [d0, d1, d2])
    }

    /// Homomorphic multiplication with relinearization
    public func multiply(_ a: Ciphertext, _ b: Ciphertext, rlk: RelinKey) throws -> Ciphertext {
        let raw = try multiplyRaw(a, b)

        // Relinearize: convert 3-component ciphertext to 2-component
        // c0' = d0 + d2 * rlk0
        // c1' = d1 + d2 * rlk1
        let d2_rlk0 = try nttEngine.multiply(raw.components[2], rlk.rlk0)
        let d2_rlk1 = try nttEngine.multiply(raw.components[2], rlk.rlk1)
        let c0 = try nttEngine.add(raw.c0, d2_rlk0)
        let c1 = try nttEngine.add(raw.components[1], d2_rlk1)

        return Ciphertext(components: [c0, c1])
    }

    // MARK: - Internal helpers

    /// Generate a random RNS polynomial (uniform random coefficients mod qi)
    private func randomRNSPoly() -> RNSPoly {
        var poly = RNSPoly(degree: params.degree, moduli: params.moduli)
        for li in 0..<params.moduli.count {
            let q = params.moduli[li]
            for j in 0..<params.degree {
                poly.limbs[li][j] = UInt32(nextRandom() % UInt64(q))
            }
        }
        return poly
    }

    /// Small error polynomial (centered binomial distribution, width ~6)
    private func smallErrorPoly() -> RNSPoly {
        let n = params.degree
        let moduli = params.moduli
        var coeffs = [Int64](repeating: 0, count: n)
        for i in 0..<n {
            // Centered binomial: sum of 3 random bits minus sum of 3 random bits
            var sum: Int64 = 0
            for _ in 0..<3 {
                sum += Int64(nextRandom() & 1)
            }
            for _ in 0..<3 {
                sum -= Int64(nextRandom() & 1)
            }
            coeffs[i] = sum
        }
        return RNSPoly(coefficients: coeffs, moduli: moduli)
    }

    /// Simple PRNG (xorshift64)
    private func nextRandom() -> UInt64 {
        rng ^= rng << 13
        rng ^= rng >> 7
        rng ^= rng << 17
        return rng
    }
}
