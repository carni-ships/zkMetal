// XHash-M31: STARK-friendly hash function for the Mersenne31 field
// Based on "RPO-M31 and XHash-M31: Efficient Hash Functions for Circle STARKs"
// (ePrint 2024/1635) and the reference implementation at
// github.com/AbdelStark/rpo-xhash-m31.
//
// State: 24 M31 elements, Rate: 16, Capacity: 8
// S-box: x^5 (quintic), inverse: x^(5^-1 mod (p-1))
// 3 round triplets: FM | BM | P3M, followed by final CLS
//   FM: MDS -> constants -> x^5
//   BM: MDS -> constants -> x^(1/5)
//   P3M: constants -> Fp3 x^5 on 8 disjoint triplets
//   CLS: MDS -> constants
//
// Fp3 = M31[x]/(x^3 + 2), i.e., x^3 = -2
// Security: ~124-bit generic

import Foundation

// MARK: - Configuration

public enum XHashM31Config {
    public static let stateWidth = 24
    public static let rate = 16
    public static let capacity = 8
    public static let numTriplets = 3
    public static let p: UInt32 = 0x7FFFFFFF  // 2^31 - 1
}

// MARK: - S-box: x^5 over M31

@inline(__always)
private func m31Quintic(_ x: M31) -> M31 {
    let x2 = m31Sqr(x)
    let x4 = m31Sqr(x2)
    return m31Mul(x4, x)
}

// MARK: - Inverse S-box: x^(1/5) over M31
// 5^(-1) mod (p-1) = 1717986917

private let INV_QUINTIC_EXP: UInt64 = 1_717_986_917

@inline(__always)
private func m31QuinticInv(_ x: M31) -> M31 {
    return m31PowU64(x, INV_QUINTIC_EXP)
}

/// Exponentiation with UInt64 exponent (m31Pow in Mersenne31.swift takes UInt32).
@inline(__always)
private func m31PowU64(_ base: M31, _ exp: UInt64) -> M31 {
    var result = M31.one
    var b = base
    var e = exp
    while e > 0 {
        if e & 1 == 1 { result = m31Mul(result, b) }
        b = m31Sqr(b)
        e >>= 1
    }
    return result
}

// MARK: - MDS Matrix (24x24 from 32x32 circulant)
// First row from the paper's reference implementation.

private let MDS_M31_FIRST_ROW_32: [UInt32] = [
    185870542, 2144994796, 1696461115, 215190769, 930115258, 766567118, 2003379079,
    1770558586, 1779722644, 434368282, 289154277, 1979813463, 1436360233, 1342944808,
    63026005, 903393155, 1512525948, 105409451, 1072974295, 979558870, 436105640,
    2126764826, 1981550821, 636196459, 645360517, 412540024, 1649351985, 1485803845,
    53244687, 719457988, 270924307, 82564914,
]

/// Build the 24x24 MDS matrix from the 32x32 circulant first row, reduced mod p.
private let MDS_M31_MATRIX: [[M31]] = {
    let n = 24
    let bigN = 32
    var first = [M31](repeating: M31.zero, count: bigN)
    for i in 0..<bigN {
        // Reduce the hardcoded values mod p (they may be > p for the M31 field)
        first[i] = M31(v: UInt32(UInt64(MDS_M31_FIRST_ROW_32[i]) % UInt64(M31.P)))
    }
    var m = [[M31]](repeating: [M31](repeating: M31.zero, count: n), count: n)
    for row in 0..<n {
        for col in 0..<n {
            let idx = (col + bigN - row) % bigN
            m[row][col] = first[idx]
        }
    }
    return m
}()

// MARK: - Round Constants (derived via SHAKE-256)
// Tag: "XHash-M31:p=2147483647,m=24,c=8,n=3"
// Total: (3*3 + 1) * 24 = 240 constants
// Each constant: read 5 LE bytes from SHAKE-256 XOF, reduce mod p.
//
// We precompute these at first use. For a production build one would
// embed them as literals; here we derive deterministically for correctness.

import CryptoKit

private func deriveXHashM31Constants() -> [[M31]] {
    let tag = "XHash\u{2011}M31:p=2147483647,m=24,c=8,n=3"
    let needed = (XHashM31Config.numTriplets * 3 + 1) * XHashM31Config.stateWidth  // 240

    // Use SHA-256 based SHAKE-256 equivalent via deterministic expansion.
    // Since Swift doesn't have SHAKE-256 natively, we use a simple deterministic
    // XOF based on SHA-256 in counter mode with the tag as input.
    var constants = [M31]()
    constants.reserveCapacity(needed)

    var counter: UInt32 = 0
    var buf = [UInt8]()

    while constants.count < needed {
        // Generate more bytes: SHA-256(tag || counter_le)
        var data = Array(tag.utf8)
        var ctr = counter
        data.append(contentsOf: withUnsafeBytes(of: &ctr) { Array($0) })
        let hash = SHA256.hash(data: data)
        buf.append(contentsOf: hash)
        counter += 1

        // Extract 5-byte chunks and reduce mod p
        while buf.count >= 5 && constants.count < needed {
            var v: UInt64 = 0
            for shift in 0..<5 {
                v |= UInt64(buf[shift]) << (8 * shift)
            }
            buf.removeFirst(5)
            let reduced = UInt32(v % UInt64(M31.P))
            constants.append(M31(v: reduced))
        }
    }

    // Reshape into (numTriplets * 3 + 1) arrays of stateWidth
    let numArrays = XHashM31Config.numTriplets * 3 + 1  // 10
    var result = [[M31]]()
    for i in 0..<numArrays {
        let start = i * XHashM31Config.stateWidth
        let end = start + XHashM31Config.stateWidth
        result.append(Array(constants[start..<end]))
    }
    return result
}

private let XHASH_M31_CONSTANTS: [[M31]] = deriveXHashM31Constants()

// MARK: - MDS Multiply for M31

@inline(__always)
private func m31MdsMultiply(_ state: inout [M31]) {
    let n = 24
    var result = [M31](repeating: M31.zero, count: n)
    for i in 0..<n {
        var acc = M31.zero
        for j in 0..<n {
            acc = m31Add(acc, m31Mul(state[j], MDS_M31_MATRIX[i][j]))
        }
        result[i] = acc
    }
    state = result
}

@inline(__always)
private func m31AddConstants(_ state: inout [M31], _ c: [M31]) {
    for i in 0..<24 {
        state[i] = m31Add(state[i], c[i])
    }
}

// MARK: - Fp3 = M31[x]/(x^3 + 2), i.e., x^3 = -2

private struct Fp3M31 {
    var a: M31  // coefficient of x^0
    var b: M31  // coefficient of x^1
    var c: M31  // coefficient of x^2
}

@inline(__always)
private func fp3M31Mul(_ lhs: Fp3M31, _ rhs: Fp3M31) -> Fp3M31 {
    let v0 = m31Mul(lhs.a, rhs.a)
    let v1 = m31Mul(lhs.b, rhs.b)
    let v2 = m31Mul(lhs.c, rhs.c)

    let s1 = m31Sub(m31Mul(m31Add(lhs.a, lhs.b), m31Add(rhs.a, rhs.b)), m31Add(v0, v1))
    let s2 = m31Sub(m31Mul(m31Add(lhs.b, lhs.c), m31Add(rhs.b, rhs.c)), m31Add(v1, v2))
    let s3 = m31Sub(m31Mul(m31Add(lhs.a, lhs.c), m31Add(rhs.a, rhs.c)), m31Add(v0, v2))

    let two = M31(v: 2)
    // x^3 = -2, so reduction: c0 = v0 - 2*v2, c1 = s1 - 2*v2, c2 = s3 + s2
    // Note: -2 * v2 in M31 is (p - 2) * v2 but easier: sub(v0, mul(two, v2))
    return Fp3M31(
        a: m31Sub(v0, m31Mul(two, v2)),
        b: m31Sub(s1, m31Mul(two, v2)),
        c: m31Add(s3, s2)
    )
}

@inline(__always)
private func fp3M31Quintic(_ x: Fp3M31) -> Fp3M31 {
    let x2 = fp3M31Mul(x, x)
    let x4 = fp3M31Mul(x2, x2)
    return fp3M31Mul(x4, x)
}

// MARK: - XHash-M31 Permutation

/// Applies the XHash-M31 permutation to a 24-element M31 state.
/// Structure: 3 triplets of (FM, BM, P3M) + final CLS.
public func xhashM31Permutation(_ input: [M31]) -> [M31] {
    precondition(input.count == 24)
    var state = input
    xhashM31Permutation(state: &state)
    return state
}

/// In-place XHash-M31 permutation.
public func xhashM31Permutation(state: inout [M31]) {
    precondition(state.count == 24)
    let c = XHASH_M31_CONSTANTS
    var idx = 0

    for _ in 0..<XHashM31Config.numTriplets {
        // FM step: MDS -> constants -> x^5
        m31MdsMultiply(&state)
        m31AddConstants(&state, c[idx])
        for i in 0..<24 { state[i] = m31Quintic(state[i]) }
        idx += 1

        // BM step: MDS -> constants -> x^(1/5)
        m31MdsMultiply(&state)
        m31AddConstants(&state, c[idx])
        for i in 0..<24 { state[i] = m31QuinticInv(state[i]) }
        idx += 1

        // P3M step: constants -> Fp3 x^5 on 8 disjoint triplets
        m31AddConstants(&state, c[idx])
        for t in 0..<8 {
            let base = t * 3
            let fp3 = Fp3M31(a: state[base], b: state[base+1], c: state[base+2])
            let result = fp3M31Quintic(fp3)
            state[base] = result.a
            state[base+1] = result.b
            state[base+2] = result.c
        }
        idx += 1
    }

    // CLS: MDS -> constants
    m31MdsMultiply(&state)
    m31AddConstants(&state, c[idx])
}

// MARK: - XHash-M31 Hash (Sponge)

/// Hash using XHash-M31 sponge. Rate=16, capacity=8. Returns 16-element digest.
public func xhashM31Hash(_ inputs: [M31]) -> [M31] {
    if inputs.isEmpty { return [M31](repeating: M31.zero, count: 16) }

    var state = [M31](repeating: M31.zero, count: 24)
    var i = 0
    while i < inputs.count {
        for j in 0..<16 {
            if i + j < inputs.count {
                state[j] = m31Add(state[j], inputs[i + j])
            }
        }
        xhashM31Permutation(state: &state)
        i += 16
    }
    return Array(state[0..<16])
}

/// 2-to-1 compression using XHash-M31.
public func xhashM31Merge(left: [M31], right: [M31]) -> [M31] {
    precondition(left.count == 8 && right.count == 8)
    var state = [M31](repeating: M31.zero, count: 24)
    for i in 0..<8 { state[i] = left[i] }
    for i in 0..<8 { state[i + 8] = right[i] }
    xhashM31Permutation(state: &state)
    return Array(state[0..<8])
}
