// RNS (Residue Number System) field for Homomorphic Encryption
// HE uses ciphertext modulus Q = q1 * q2 * ... * qL where each qi is ~30 bits.
// Each qi is chosen as qi = ki * 2^d + 1 (NTT-friendly) where 2^d >= 2N.
// This enables efficient NTT on each residue independently.

import Foundation

// MARK: - HE-friendly NTT primes

/// Standard HE-friendly 30-bit primes: qi = ki * 2^17 + 1 (supports NTT up to N=2^16)
/// These satisfy qi < 2^30 so that qi * qi < 2^60 fits in UInt64.
/// Each has 2-adicity >= 17, supporting polynomial degrees up to 2^16.
public let heDefaultModuli: [UInt32] = [
    0x3FFE0001,  // 1073610753 = 8191 * 2^17 + 1
    0x3FFC0001,  // 1073479681 = 8190 * 2^17 + 1 (not prime, will verify)
    0x3FF80001,  // 1073348609
    0x3FEC0001,  // 1072562177
    0x3FE40001,  // 1072037889
]

/// Verified HE primes: each is prime, < 2^30, and = k * 2^17 + 1
/// Pre-selected for common HE parameter sets (N=4096, 8192, 16384)
public func heVerifiedPrimes(count: Int, logN: Int) -> [UInt32] {
    // We need primes q = k * 2^d + 1 where d >= logN + 1
    // For logN <= 16, d = 17 works.
    // Search for valid primes
    let d: UInt32 = max(UInt32(logN + 1), 17)
    let step = UInt32(1) << d
    var primes: [UInt32] = []
    // Start from a large k so primes are close to 2^30
    var k = (UInt32(1) << 30) / step
    while primes.count < count && k > 1 {
        let candidate = k &* step &+ 1
        if candidate > (1 << 30) { k -= 1; continue }
        if isPrime32(candidate) {
            primes.append(candidate)
        }
        k -= 1
    }
    return primes
}

/// Simple primality test for 32-bit numbers
public func isPrime32(_ n: UInt32) -> Bool {
    if n < 2 { return false }
    if n < 4 { return true }
    if n % 2 == 0 || n % 3 == 0 { return false }
    var i: UInt32 = 5
    while i * i <= n {
        if n % i == 0 || n % (i + 2) == 0 { return false }
        i += 6
    }
    return true
}

// MARK: - RNS Limb (single residue mod qi)

public struct RNSLimb {
    public var value: UInt32
    public let modulus: UInt32

    public init(value: UInt32, modulus: UInt32) {
        self.value = value % modulus
        self.modulus = modulus
    }

    @inline(__always)
    public static func add(_ a: RNSLimb, _ b: RNSLimb) -> RNSLimb {
        assert(a.modulus == b.modulus)
        let s = UInt64(a.value) + UInt64(b.value)
        let q = UInt64(a.modulus)
        return RNSLimb(value: UInt32(s >= q ? s - q : s), modulus: a.modulus)
    }

    @inline(__always)
    public static func sub(_ a: RNSLimb, _ b: RNSLimb) -> RNSLimb {
        assert(a.modulus == b.modulus)
        if a.value >= b.value {
            return RNSLimb(value: a.value - b.value, modulus: a.modulus)
        }
        return RNSLimb(value: a.value &+ a.modulus &- b.value, modulus: a.modulus)
    }

    @inline(__always)
    public static func mul(_ a: RNSLimb, _ b: RNSLimb) -> RNSLimb {
        assert(a.modulus == b.modulus)
        let prod = UInt64(a.value) * UInt64(b.value)
        return RNSLimb(value: UInt32(prod % UInt64(a.modulus)), modulus: a.modulus)
    }

    public static func pow(_ base: RNSLimb, _ exp: UInt32) -> RNSLimb {
        if exp == 0 { return RNSLimb(value: 1, modulus: base.modulus) }
        var result = RNSLimb(value: 1, modulus: base.modulus)
        var b = base
        var e = exp
        while e > 0 {
            if e & 1 == 1 { result = mul(result, b) }
            b = mul(b, b)
            e >>= 1
        }
        return result
    }

    public static func inverse(_ a: RNSLimb) -> RNSLimb {
        return pow(a, a.modulus - 2)
    }
}

// MARK: - Barrett constant

/// Compute Barrett constant for modulus q: floor(2^62 / q)
public func barrettConstant(_ q: UInt32) -> UInt32 {
    // floor(2^62 / q)
    // 2^62 = 4611686018427387904
    let k = UInt64(1) << 62
    return UInt32(k / UInt64(q))
}

// MARK: - RNS Polynomial

/// RNS representation of a polynomial: coefficients stored as residues mod q1,...,qL
/// Memory layout: limbs[i][j] = j-th coefficient mod qi
/// GPU layout (packed): [limb0_coeff0, limb0_coeff1, ..., limb0_coeffN-1, limb1_coeff0, ...]
public struct RNSPoly {
    public let degree: Int       // N (polynomial degree, power of 2)
    public let moduli: [UInt32]  // q1, ..., qL
    public var limbs: [[UInt32]] // limbs[i] = coefficients mod qi, length = degree

    public var numLimbs: Int { moduli.count }

    public init(degree: Int, moduli: [UInt32]) {
        precondition(degree > 0 && (degree & (degree - 1)) == 0, "Degree must be power of 2")
        self.degree = degree
        self.moduli = moduli
        self.limbs = Array(repeating: [UInt32](repeating: 0, count: degree), count: moduli.count)
    }

    /// Create from integer coefficients (reduce mod each qi)
    public init(coefficients: [Int64], moduli: [UInt32]) {
        let n = coefficients.count
        // Round up to power of 2
        var deg = 1
        while deg < n { deg <<= 1 }
        self.degree = deg
        self.moduli = moduli
        self.limbs = Array(repeating: [UInt32](repeating: 0, count: deg), count: moduli.count)
        for (li, q) in moduli.enumerated() {
            let qI64 = Int64(q)
            for (ci, c) in coefficients.enumerated() {
                var r = c % qI64
                if r < 0 { r += qI64 }
                limbs[li][ci] = UInt32(r)
            }
        }
    }

    /// Pack into flat array for GPU: [limb0 coeffs | limb1 coeffs | ...]
    public func packed() -> [UInt32] {
        var result = [UInt32]()
        result.reserveCapacity(numLimbs * degree)
        for l in limbs {
            result.append(contentsOf: l)
        }
        return result
    }

    /// Unpack from flat GPU array
    public mutating func unpack(_ data: [UInt32]) {
        precondition(data.count == numLimbs * degree)
        for i in 0..<numLimbs {
            let start = i * degree
            limbs[i] = Array(data[start..<start + degree])
        }
    }
}

// MARK: - Primitive root of unity

/// Find a primitive 2^k-th root of unity mod q where q = m * 2^d + 1
/// Uses generator approach: find g such that g^((q-1)/2^k) has order 2^k
public func primitiveRootOfUnity(modulus q: UInt32, logN: Int) -> UInt32 {
    let n = UInt32(1 << logN)
    let qm1 = q - 1
    // Check 2-adicity: find largest d such that 2^d | (q-1)
    var twoAdicity = 0
    var tmp = qm1
    while tmp & 1 == 0 {
        twoAdicity += 1
        tmp >>= 1
    }
    precondition(logN <= twoAdicity, "Modulus does not have sufficient 2-adicity for N=2^\(logN)")

    // Find a generator of the multiplicative group
    let exp = qm1 / n
    // Try small values as generators
    for g: UInt32 in 2..<100 {
        let limb = RNSLimb(value: g, modulus: q)
        let omega = RNSLimb.pow(limb, exp)
        // Verify it has order exactly n: omega^(n/2) != 1
        let half = RNSLimb.pow(omega, n / 2)
        if half.value != 1 {
            return omega.value
        }
    }
    fatalError("Could not find primitive root of unity for q=\(q), logN=\(logN)")
}

/// Precompute twiddle factors for NTT mod q: [omega^0, omega^1, ..., omega^(N/2-1)]
/// in bit-reversed order for Cooley-Tukey DIT
public func precomputeTwiddles(modulus q: UInt32, logN: Int) -> [UInt32] {
    let n = 1 << logN
    let half = n / 2
    let omega = primitiveRootOfUnity(modulus: q, logN: logN)

    // Compute powers of omega
    var twiddles = [UInt32](repeating: 0, count: half)
    var w: UInt64 = 1
    for i in 0..<half {
        twiddles[i] = UInt32(w)
        w = (w * UInt64(omega)) % UInt64(q)
    }
    return twiddles
}

/// Precompute inverse twiddle factors for iNTT
public func precomputeInverseTwiddles(modulus q: UInt32, logN: Int) -> [UInt32] {
    let n = 1 << logN
    let half = n / 2
    let omega = primitiveRootOfUnity(modulus: q, logN: logN)
    let omegaInv = RNSLimb.inverse(RNSLimb(value: omega, modulus: q)).value

    var twiddles = [UInt32](repeating: 0, count: half)
    var w: UInt64 = 1
    for i in 0..<half {
        twiddles[i] = UInt32(w)
        w = (w * UInt64(omegaInv)) % UInt64(q)
    }
    return twiddles
}

// MARK: - CPU Reference NTT

/// CPU NTT: Cooley-Tukey DIT, in-place, mod q
/// Input must be in bit-reversed order. Output in natural order.
public func cpuNTT(_ data: inout [UInt32], modulus q: UInt32, logN: Int) {
    let n = 1 << logN
    precondition(data.count == n)

    // Bit-reverse permutation
    for i in 0..<n {
        let j = bitReverse(UInt32(i), bits: logN)
        if i < Int(j) {
            data.swapAt(i, Int(j))
        }
    }

    let twiddles = precomputeTwiddles(modulus: q, logN: logN)

    // Butterfly stages
    for stage in 0..<logN {
        let halfBlock = 1 << stage
        let blockSize = halfBlock << 1
        let stride = n / blockSize

        for blockStart in Swift.stride(from: 0, to: n, by: blockSize) {
            for j in 0..<halfBlock {
                let i0 = blockStart + j
                let i1 = i0 + halfBlock
                let twIdx = j * stride

                let a = UInt64(data[i0])
                let b = UInt64(data[i1])
                let w = UInt64(twiddles[twIdx])
                let wb = (w * b) % UInt64(q)

                data[i0] = UInt32((a + wb) % UInt64(q))
                data[i1] = UInt32((a + UInt64(q) - wb) % UInt64(q))
            }
        }
    }
}

/// CPU inverse NTT: Gentleman-Sande DIF, in-place, mod q
public func cpuINTT(_ data: inout [UInt32], modulus q: UInt32, logN: Int) {
    let n = 1 << logN
    precondition(data.count == n)

    let invTwiddles = precomputeInverseTwiddles(modulus: q, logN: logN)

    // DIF butterfly stages (reverse order)
    for si in 0..<logN {
        let stage = logN - 1 - si
        let halfBlock = 1 << stage
        let blockSize = halfBlock << 1
        let stride = n / blockSize

        for blockStart in Swift.stride(from: 0, to: n, by: blockSize) {
            for j in 0..<halfBlock {
                let i0 = blockStart + j
                let i1 = i0 + halfBlock
                let twIdx = j * stride

                let a = UInt64(data[i0])
                let b = UInt64(data[i1])
                let w = UInt64(invTwiddles[twIdx])
                let sum = (a + b) % UInt64(q)
                let diff = (a + UInt64(q) - b) % UInt64(q)

                data[i0] = UInt32(sum)
                data[i1] = UInt32((diff * w) % UInt64(q))
            }
        }
    }

    // Bit-reverse permutation
    for i in 0..<n {
        let j = bitReverse(UInt32(i), bits: logN)
        if i < Int(j) {
            data.swapAt(i, Int(j))
        }
    }

    // Scale by 1/N
    let invN = RNSLimb.inverse(RNSLimb(value: UInt32(n), modulus: q)).value
    let invN64 = UInt64(invN)
    let q64 = UInt64(q)
    for i in 0..<n {
        data[i] = UInt32((UInt64(data[i]) * invN64) % q64)
    }
}

/// Bit-reverse a value with given number of bits
private func bitReverse(_ x: UInt32, bits: Int) -> UInt32 {
    var v = x
    var r: UInt32 = 0
    for _ in 0..<bits {
        r = (r << 1) | (v & 1)
        v >>= 1
    }
    return r
}
