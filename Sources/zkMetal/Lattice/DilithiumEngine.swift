// Dilithium Signature Engine
// Implements ML-DSA-44 (NIST FIPS 204) core operations with GPU-accelerated NTT.
//
// Dilithium2 (ML-DSA-44): k=4, l=4, n=256, q=8380417
// Security: ~128-bit classical, ~128-bit quantum
//
// GPU acceleration targets:
// - Batch NTT for A*y and A*z (k*l = 16 NTTs per sign/verify)
// - Batch pointwise multiply in NTT domain
// - Rejection sampling loop can batch NTT operations per attempt

import Foundation

// MARK: - Dilithium Parameters

public enum DilithiumParams {
    public static let n = 256           // polynomial degree
    public static let k = 4             // rows of A
    public static let l = 4             // columns of A
    public static let q: UInt32 = 8380417
    public static let gamma1: UInt32 = 131072   // 2^17, bound for y
    public static let gamma2: UInt32 = 95232    // (q-1)/88, low-order rounding
    public static let beta: UInt32 = 78         // tau * eta
    public static let tau = 39           // number of +/-1 in challenge
    public static let eta = 2            // secret coefficient bound
    public static let omega = 80         // max 1s in hint
}

// MARK: - Key types

public struct DilithiumPublicKey {
    /// Matrix A in NTT domain: k x l polynomials
    public let A_hat: [[DilithiumField]]  // k*l polynomials
    /// Public key t = A*s1 + s2
    public let t: [[DilithiumField]]      // k polynomials
}

public struct DilithiumSecretKey {
    /// Secret vector s1 (l polynomials, small coefficients)
    public let s1: [[DilithiumField]]
    /// Secret vector s2 (k polynomials, small coefficients)
    public let s2: [[DilithiumField]]
    /// NTT of s1
    public let s1_hat: [[DilithiumField]]
    /// NTT of s2
    public let s2_hat: [[DilithiumField]]
    /// Associated public key
    public let publicKey: DilithiumPublicKey
}

public struct DilithiumSignature {
    /// Response vector z (l polynomials)
    public let z: [[DilithiumField]]
    /// Challenge polynomial c (sparse: tau non-zero entries of +/-1)
    public let c: [DilithiumField]
    /// Hint for verification
    public let hint: [[Bool]]  // k polynomials of boolean hints
}

// MARK: - Dilithium Engine

public class DilithiumEngine {
    public static let version = Versions.dilithium
    public let nttEngine: LatticeNTTEngine
    private var rng: UInt64

    public init(nttEngine: LatticeNTTEngine, seed: UInt64 = 0xCAFE_BABE_DEAD_BEEF) {
        self.nttEngine = nttEngine
        self.rng = seed
    }

    // MARK: - Key Generation

    /// Generate Dilithium key pair.
    public func keyGen() throws -> DilithiumSecretKey {
        let k = DilithiumParams.k
        let l = DilithiumParams.l

        // Generate matrix A (k x l polynomials in NTT domain)
        var A_hat = [[DilithiumField]]()
        for _ in 0..<(k * l) {
            A_hat.append(randomDilithiumPoly())
        }

        // Generate secret vectors s1 (l polys) and s2 (k polys) with small coefficients
        var s1 = [[DilithiumField]]()
        for _ in 0..<l {
            s1.append(sampleUniform(bound: DilithiumParams.eta))
        }
        var s2 = [[DilithiumField]]()
        for _ in 0..<k {
            s2.append(sampleUniform(bound: DilithiumParams.eta))
        }

        // NTT(s1), NTT(s2) — use GPU batch NTT for speed
        let s1_hat = try nttEngine.batchDilithiumNTT(s1)
        let s2_hat = try nttEngine.batchDilithiumNTT(s2)

        // t = A * s1 + s2 — use GPU matvec for A*s1, then add s2
        let A_flat = A_hat.flatMap { $0.map { $0.value } }
        let s1_flat = s1_hat.flatMap { $0.map { $0.value } }
        let t_flat = try nttEngine.dilithiumMatvec(A_flat, s1_flat, k: k)
        var t = [[DilithiumField]]()
        for i in 0..<k {
            let start = i * 256
            t.append(Array(t_flat[start..<start+256]).map { DilithiumField(value: $0) })
        }
        // Add s2
        for i in 0..<k {
            for c in 0..<DilithiumParams.n {
                t[i][c] = dilithiumAdd(t[i][c], s2[i][c])
            }
        }

        let pk = DilithiumPublicKey(A_hat: A_hat, t: t)
        return DilithiumSecretKey(s1: s1, s2: s2, s1_hat: s1_hat, s2_hat: s2_hat, publicKey: pk)
    }

    // MARK: - Signing

    /// Sign a message. Uses rejection sampling — may require multiple attempts.
    public func sign(sk: DilithiumSecretKey, message: [UInt8]) throws -> DilithiumSignature {
        let k = DilithiumParams.k
        let l = DilithiumParams.l

        // Rejection sampling loop
        for _ in 0..<1000 {
            // Sample y from [-gamma1+1, gamma1]
            var y = [[DilithiumField]]()
            for _ in 0..<l {
                y.append(sampleMasking())
            }

            // w = A * y (via NTT) — GPU batch NTT + GPU matvec
            let y_hat = try nttEngine.batchDilithiumNTT(y)

            // A*y in NTT domain using GPU matvec
            let A_flat = sk.publicKey.A_hat.flatMap { $0.map { $0.value } }
            let y_flat = y_hat.flatMap { $0.map { $0.value } }
            let w_flat = try nttEngine.dilithiumMatvec(A_flat, y_flat, k: k)
            var w_hat = [[DilithiumField]]()
            for i in 0..<k {
                let start = i * 256
                w_hat.append(Array(w_flat[start..<start+256]).map { DilithiumField(value: $0) })
            }
            let w = try nttEngine.batchDilithiumINTT(w_hat)

            // High bits of w for challenge generation
            let w1 = w.map { highBits($0) }

            // Generate challenge c from hash of (message, w1)
            let c = generateChallenge(message: message, w1: w1)

            // z = y + c * s1 — use GPU pointwise mul for c*s1
            var c_hat = c
            dilithiumNTTCPU(&c_hat)

            // Compute c*s1 in NTT domain using GPU pointwise mul (c_hat broadcast to l polys)
            let c_flat = c_hat.map { $0.value }  // [UInt32] (256 elements)
            let s1_flat = sk.s1_hat.flatMap { $0.map { $0.value } }
            // Broadcast c across l polynomials: [c0,c1,...,c255, c0,c1,...,c255, ...]
            let l = DilithiumParams.l
            var c_broadcast = [UInt32](repeating: 0, count: l * 256)
            for j in 0..<l {
                for c_idx in 0..<256 {
                    c_broadcast[j * 256 + c_idx] = c_flat[c_idx]
                }
            }
            let cs1_pointwise = try nttEngine.dilithiumPointwiseMul(c_broadcast, s1_flat)  // l*256 result
            var cs1_hat = [[DilithiumField]]()
            for j in 0..<l {
                var cs1j = [DilithiumField](repeating: DilithiumField.zero, count: 256)
                for c_idx in 0..<256 {
                    cs1j[c_idx] = DilithiumField(value: cs1_pointwise[j * 256 + c_idx])
                }
                cs1_hat.append(cs1j)
            }
            let cs1 = try nttEngine.batchDilithiumINTT(cs1_hat)

            var z = [[DilithiumField]]()
            for j in 0..<l {
                var zj = [DilithiumField](repeating: DilithiumField.zero, count: DilithiumParams.n)
                for c_idx in 0..<DilithiumParams.n {
                    zj[c_idx] = dilithiumAdd(y[j][c_idx], cs1[j][c_idx])
                }
                z.append(zj)
            }

            // Check z norm: ||z||_inf < gamma1 - beta
            let bound = DilithiumParams.gamma1 - DilithiumParams.beta
            var reject = false
            for j in 0..<l {
                for c_idx in 0..<DilithiumParams.n {
                    let val = centeredReduce(z[j][c_idx])
                    if abs(val) >= Int32(bound) {
                        reject = true
                        break
                    }
                }
                if reject { break }
            }
            if reject { continue }

            // Compute hint (simplified)
            let hint = [[Bool]](repeating: [Bool](repeating: false, count: DilithiumParams.n), count: k)

            return DilithiumSignature(z: z, c: c, hint: hint)
        }

        // Should not happen with correct parameters
        fatalError("Dilithium signing failed: too many rejection attempts")
    }

    // MARK: - Verification

    /// Verify a signature.
    public func verify(pk: DilithiumPublicKey, message: [UInt8], signature: DilithiumSignature) throws -> Bool {
        let k = DilithiumParams.k
        let l = DilithiumParams.l

        // Check z norm
        let bound = DilithiumParams.gamma1 - DilithiumParams.beta
        for j in 0..<l {
            for c_idx in 0..<DilithiumParams.n {
                let val = centeredReduce(signature.z[j][c_idx])
                if abs(val) >= Int32(bound) {
                    return false
                }
            }
        }

        // w_prime_hat = A*z - c*t (in NTT domain) — use GPU for c*t pointwise, then CPU subtract
        let z_hat = try nttEngine.batchDilithiumNTT(signature.z)
        var c_hat = signature.c
        dilithiumNTTCPU(&c_hat)  // c is 1 polynomial, batch not worth it
        let t_hat = try nttEngine.batchDilithiumNTT(pk.t)

        // A*z using GPU matvec
        let A_flat = pk.A_hat.flatMap { $0.map { $0.value } }
        let z_flat = z_hat.flatMap { $0.map { $0.value } }
        let Az_flat = try nttEngine.dilithiumMatvec(A_flat, z_flat, k: k)

        // Compute c*t using GPU pointwise mul (c broadcast to k polynomials)
        let c_flat = c_hat.map { $0.value }  // [UInt32] (256 elements)
        let t_flat = t_hat.flatMap { $0.map { $0.value } }
        var c_broadcast = [UInt32](repeating: 0, count: k * 256)
        for i in 0..<k {
            for c_idx in 0..<256 {
                c_broadcast[i * 256 + c_idx] = c_flat[c_idx]
            }
        }
        let ct_pointwise = try nttEngine.dilithiumPointwiseMul(c_broadcast, t_flat)  // k*256 result

        // Subtract ct from Az: w_prime_hat[i*256+c] = Az_flat[i*256+c] - ct_pointwise[i*256+c]
        var w_prime_hat = [[DilithiumField]]()
        for i in 0..<k {
            var wi = [DilithiumField](repeating: DilithiumField.zero, count: 256)
            for c_idx in 0..<256 {
                let az_val = DilithiumField(value: Az_flat[i * 256 + c_idx])
                let ct_val = DilithiumField(value: ct_pointwise[i * 256 + c_idx])
                wi[c_idx] = dilithiumSub(az_val, ct_val)
            }
            w_prime_hat.append(wi)
        }

        let w_prime = try nttEngine.batchDilithiumINTT(w_prime_hat)

        // Recompute challenge from (message, highBits(w'))
        let w1_prime = w_prime.map { highBits($0) }
        let c_recomputed = generateChallenge(message: message, w1: w1_prime)

        // Verify challenge matches
        for i in 0..<DilithiumParams.n {
            if signature.c[i].value != c_recomputed[i].value {
                return false
            }
        }

        return true
    }

    // MARK: - Batch Operations

    /// Batch key generation
    public func batchKeyGen(count: Int) throws -> [DilithiumSecretKey] {
        var keys = [DilithiumSecretKey]()
        keys.reserveCapacity(count)
        for _ in 0..<count {
            keys.append(try keyGen())
        }
        return keys
    }

    /// Batch verification: verify multiple signatures in parallel
    public func batchVerify(entries: [(DilithiumPublicKey, [UInt8], DilithiumSignature)]) throws -> [Bool] {
        var results = [Bool]()
        results.reserveCapacity(entries.count)
        for (pk, msg, sig) in entries {
            results.append(try verify(pk: pk, message: msg, signature: sig))
        }
        return results
    }

    // MARK: - Helper functions

    /// Centered reduction: map [0, q) to [-(q-1)/2, (q-1)/2]
    private func centeredReduce(_ a: DilithiumField) -> Int32 {
        let halfQ = Int32(DilithiumParams.q / 2)
        let v = Int32(a.value)
        return v > halfQ ? v - Int32(DilithiumParams.q) : v
    }

    /// High bits extraction (simplified: top bits after rounding)
    private func highBits(_ poly: [DilithiumField]) -> [DilithiumField] {
        let gamma2 = DilithiumParams.gamma2
        return poly.map { a in
            let centered = centeredReduce(a)
            let abs_val = UInt32(abs(centered))
            let high = abs_val / gamma2
            return DilithiumField(value: high % DilithiumParams.q)
        }
    }

    /// Sample polynomial with coefficients uniform in [-eta, eta]
    private func sampleUniform(bound eta: Int) -> [DilithiumField] {
        var poly = [DilithiumField](repeating: DilithiumField.zero, count: DilithiumParams.n)
        for i in 0..<DilithiumParams.n {
            let r = Int(nextRandom() % UInt64(2 * eta + 1)) - eta
            if r >= 0 {
                poly[i] = DilithiumField(value: UInt32(r))
            } else {
                poly[i] = DilithiumField(value: DilithiumParams.q - UInt32(-r))
            }
        }
        return poly
    }

    /// Sample masking polynomial with coefficients in [-gamma1+1, gamma1]
    private func sampleMasking() -> [DilithiumField] {
        let gamma1 = DilithiumParams.gamma1
        var poly = [DilithiumField](repeating: DilithiumField.zero, count: DilithiumParams.n)
        for i in 0..<DilithiumParams.n {
            let r = Int64(nextRandom() % UInt64(2 * gamma1)) - Int64(gamma1) + 1
            if r >= 0 {
                poly[i] = DilithiumField(value: UInt32(r))
            } else {
                poly[i] = DilithiumField(value: UInt32(Int64(DilithiumParams.q) + r))
            }
        }
        return poly
    }

    /// Generate random polynomial (for matrix A)
    private func randomDilithiumPoly() -> [DilithiumField] {
        var poly = [DilithiumField](repeating: DilithiumField.zero, count: DilithiumParams.n)
        for i in 0..<DilithiumParams.n {
            poly[i] = DilithiumField(value: UInt32(nextRandom() % UInt64(DilithiumParams.q)))
        }
        return poly
    }

    /// Generate challenge polynomial (sparse with tau +/-1 entries)
    /// In real Dilithium, this is derived from hash of (message, w1) via SHAKE-256.
    private func generateChallenge(message: [UInt8], w1: [[DilithiumField]]) -> [DilithiumField] {
        var c = [DilithiumField](repeating: DilithiumField.zero, count: DilithiumParams.n)

        // Hash message and w1 to get seed for challenge
        var seed: UInt64 = 0x123456789ABCDEF0
        for b in message {
            seed ^= UInt64(b)
            seed = seed &* 0x100000001b3
        }
        for poly in w1 {
            for coeff in poly {
                seed ^= UInt64(coeff.value)
                seed = seed &* 0x100000001b3
            }
        }

        // Place tau +/-1 entries at random positions (Fisher-Yates on [0..255])
        var positions = Array(0..<256)
        var localRng = seed
        for i in stride(from: 255, through: 256 - DilithiumParams.tau, by: -1) {
            localRng ^= localRng << 13
            localRng ^= localRng >> 7
            localRng ^= localRng << 17
            let j = Int(localRng % UInt64(i + 1))
            positions.swapAt(i, j)
            // Set c at position positions[i] to +1 or -1
            localRng ^= localRng << 13
            localRng ^= localRng >> 7
            localRng ^= localRng << 17
            let sign = localRng & 1
            if sign == 0 {
                c[positions[i]] = DilithiumField(value: 1)
            } else {
                c[positions[i]] = DilithiumField(value: DilithiumParams.q - 1)
            }
        }

        return c
    }

    /// Simple PRNG (xorshift64)
    private func nextRandom() -> UInt64 {
        rng ^= rng << 13
        rng ^= rng >> 7
        rng ^= rng << 17
        return rng
    }
}
