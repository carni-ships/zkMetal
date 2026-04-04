// Poseidon2 hash function for BabyBear (p = 0x78000001 = 2013265921)
// t=16 (rate=8, capacity=8), d=7 (x^7 S-box)
// rounds_f=8 (4 + 4 external), rounds_p=13 (internal)
// Parameters match SP1/Plonky3 exactly for full compatibility.
//
// Poseidon2 permutation structure:
// 1. Initial linear layer (external matrix)
// 2. First half of full rounds (4 rounds)
// 3. Partial rounds (13 rounds)
// 4. Second half of full rounds (4 rounds)
//
// Full round: AddRC -> S-box (all elements) -> Linear layer (external)
// Partial round: AddRC (first element only) -> S-box (first element only) -> Linear layer (internal)

import Foundation

// MARK: - Poseidon2 BabyBear Configuration

public enum Poseidon2BabyBearConfig {
    public static let t = 16
    public static let rate = 8
    public static let capacity = 8
    public static let alpha = 7         // S-box exponent x^7
    public static let roundsF = 8       // 4 + 4 full rounds
    public static let roundsP = 13      // partial rounds
    public static let totalRounds = 21  // 8 + 13
}

// MARK: - Internal Diagonal Constants (Plonky3 width-16)

// V = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/4, 1/8, 1/2^27, -1/2^8, -1/16, -1/2^27]
// Converted to canonical BabyBear field elements mod p
private let POSEIDON2_BB_INTERNAL_DIAG: [UInt32] = [
    0x77ffffff, 0x00000001, 0x00000002, 0x3c000001,
    0x00000003, 0x00000004, 0x3c000000, 0x77fffffe,
    0x77fffffd, 0x77880001, 0x5a000001, 0x69000001,
    0x77fffff2, 0x00780000, 0x07800000, 0x0000000f
]

// MARK: - Round Constants (from Plonky3 source, Grain LFSR generated)

// External initial round constants (4 rounds x 16 elements)
// External final round constants (4 rounds x 16 elements)
// Internal round constants (13 rounds x 1 element, stored as 16 with padding)
private func generateBabyBearRoundConstants() -> [[Bb]] {
    let t = Poseidon2BabyBearConfig.t
    let total = Poseidon2BabyBearConfig.totalRounds

    // Plonky3 exact round constants for BabyBear width-16
    // External initial (4 rounds x 16)
    let extInit: [[UInt32]] = [
        [0x69cbb6af, 0x46ad93f9, 0x60a00f4e, 0x6b1297cd,
         0x23189afe, 0x732e7bef, 0x72c246de, 0x2c941900,
         0x0557eede, 0x1580496f, 0x3a3ea77b, 0x54f3f271,
         0x0f49b029, 0x47872fe1, 0x221e2e36, 0x1ab7202e],
        [0x487779a6, 0x3851c9d8, 0x38dc17c0, 0x209f8849,
         0x268dcee8, 0x350c48da, 0x5b9ad32e, 0x0523272b,
         0x3f89055b, 0x01e894b2, 0x13ddedde, 0x1b2ef334,
         0x7507d8b4, 0x6ceeb94e, 0x52eb6ba2, 0x50642905],
        [0x05453f3f, 0x06349efc, 0x6922787c, 0x04bfff9c,
         0x768c714a, 0x3e9ff21a, 0x15737c9c, 0x2229c807,
         0x0d47f88c, 0x097e0ecc, 0x27eadba0, 0x2d7d29e4,
         0x3502aaa0, 0x0f475fd7, 0x29fbda49, 0x018afffd],
        [0x0315b618, 0x6d4497d1, 0x1b171d9e, 0x52861abd,
         0x2e5d0501, 0x3ec8646c, 0x6e5f250a, 0x148ae8e6,
         0x17f5fa4a, 0x3e66d284, 0x0051aa3b, 0x483f7913,
         0x2cfe5f15, 0x023427ca, 0x2cc78315, 0x1e36ea47],
    ]

    // Internal round constants (13 rounds, only first element used)
    let intRC: [UInt32] = [
        0x5a8053c0, 0x693be639, 0x3858867d, 0x19334f6b,
        0x128f0fd8, 0x4e2b1ccb, 0x61210ce0, 0x3c318939,
        0x0b5b2f22, 0x2edb11d5, 0x213effdf, 0x0cac4606,
        0x241af16d
    ]

    // External final (4 rounds x 16)
    let extFinal: [[UInt32]] = [
        [0x7290a80d, 0x6f7e5329, 0x598ec8a8, 0x76a859a0,
         0x6559e868, 0x657b83af, 0x13271d3f, 0x1f876063,
         0x0aeeae37, 0x706e9ca6, 0x46400cee, 0x72a05c26,
         0x2c589c9e, 0x20bd37a7, 0x6a2d3d10, 0x20523767],
        [0x5b8fe9c4, 0x2aa501d6, 0x1e01ac3e, 0x1448bc54,
         0x5ce5ad1c, 0x4918a14d, 0x2c46a83f, 0x4fcf6876,
         0x61d8d5c8, 0x6ddf4ff9, 0x11fda4d3, 0x02933a8f,
         0x170eaf81, 0x5a9c314f, 0x49a12590, 0x35ec52a1],
        [0x58eb1611, 0x5e481e65, 0x367125c9, 0x0eba33ba,
         0x1fc28ded, 0x066399ad, 0x0cbec0ea, 0x75fd1af0,
         0x50f5bf4e, 0x643d5f41, 0x6f4fe718, 0x5b3cbbde,
         0x1e3afb3e, 0x296fb027, 0x45e1547b, 0x4a8db2ab],
        [0x59986d19, 0x30bcdfa3, 0x1db63932, 0x1d7c2824,
         0x53b33681, 0x0673b747, 0x038a98a3, 0x2c5bce60,
         0x351979cd, 0x5008fb73, 0x547bca78, 0x711af481,
         0x3f93bf64, 0x644d987b, 0x3c8bcd87, 0x608758b8],
    ]

    // Build flat array: 21 rounds x 16 elements
    var constants = [[Bb]](repeating: [Bb](repeating: Bb.zero, count: t), count: total)

    // Rounds 0-3: external initial
    for r in 0..<4 {
        for i in 0..<t {
            constants[r][i] = Bb(v: extInit[r][i])
        }
    }

    // Rounds 4-16: internal (only first element matters, rest are 0)
    for r in 0..<13 {
        constants[4 + r][0] = Bb(v: intRC[r])
    }

    // Rounds 17-20: external final
    for r in 0..<4 {
        for i in 0..<t {
            constants[17 + r][i] = Bb(v: extFinal[r][i])
        }
    }

    return constants
}

/// Cached round constants
public let POSEIDON2_BB_ROUND_CONSTANTS: [[Bb]] = generateBabyBearRoundConstants()

// MARK: - M4 Circulant Matrix

@inline(__always)
private func bbM4Apply(_ s: inout [Bb], _ offset: Int) {
    let s0 = s[offset], s1 = s[offset+1], s2 = s[offset+2], s3 = s[offset+3]
    let t0 = bbAdd(s0, s1)
    let t1 = bbAdd(s2, s3)
    let t2 = bbAdd(bbAdd(s1, s1), t1)
    let t3 = bbAdd(bbAdd(s3, s3), t0)
    s[offset]   = bbAdd(t0, t3)
    s[offset+1] = bbAdd(t1, t2)
    s[offset+2] = bbAdd(t0, t2)
    s[offset+3] = bbAdd(t1, t3)
}

// MARK: - External Linear Layer

@inline(__always)
private func bbExternalLinearLayer(_ state: inout [Bb]) {
    bbM4Apply(&state, 0)
    bbM4Apply(&state, 4)
    bbM4Apply(&state, 8)
    bbM4Apply(&state, 12)

    for i in 0..<4 {
        let sum = bbAdd(bbAdd(state[i], state[i+4]), bbAdd(state[i+8], state[i+12]))
        state[i]    = bbAdd(state[i], sum)
        state[i+4]  = bbAdd(state[i+4], sum)
        state[i+8]  = bbAdd(state[i+8], sum)
        state[i+12] = bbAdd(state[i+12], sum)
    }
}

// MARK: - Internal Linear Layer

@inline(__always)
private func bbInternalLinearLayer(_ state: inout [Bb]) {
    var sum = Bb.zero
    for i in 0..<16 {
        sum = bbAdd(sum, state[i])
    }

    for i in 0..<16 {
        let diag = POSEIDON2_BB_INTERNAL_DIAG[i]
        let prod = bbMul(state[i], Bb(v: diag))
        state[i] = bbAdd(prod, sum)
    }
}

// MARK: - S-box

/// S-box: x -> x^7
@inline(__always)
private func bbSbox(_ x: Bb) -> Bb {
    let x2 = bbSqr(x)
    let x3 = bbMul(x2, x)
    let x6 = bbSqr(x3)
    return bbMul(x6, x)
}

// MARK: - Poseidon2 BabyBear Permutation

/// Full Poseidon2 permutation on a state of 16 BabyBear elements.
public func poseidon2BbPermutation(_ input: [Bb]) -> [Bb] {
    precondition(input.count == 16)
    let rc = POSEIDON2_BB_ROUND_CONSTANTS

    var state = input

    // Initial external linear layer
    bbExternalLinearLayer(&state)

    // First half of full rounds (rounds 0..3)
    for r in 0..<4 {
        for i in 0..<16 { state[i] = bbAdd(state[i], rc[r][i]) }
        for i in 0..<16 { state[i] = bbSbox(state[i]) }
        bbExternalLinearLayer(&state)
    }

    // Partial rounds (rounds 4..16)
    for r in 4..<17 {
        state[0] = bbAdd(state[0], rc[r][0])
        state[0] = bbSbox(state[0])
        bbInternalLinearLayer(&state)
    }

    // Second half of full rounds (rounds 17..20)
    for r in 17..<21 {
        for i in 0..<16 { state[i] = bbAdd(state[i], rc[r][i]) }
        for i in 0..<16 { state[i] = bbSbox(state[i]) }
        bbExternalLinearLayer(&state)
    }

    return state
}

/// In-place Poseidon2 permutation.
public func poseidon2BbPermutation(state: inout [Bb]) {
    precondition(state.count == 16)
    let rc = POSEIDON2_BB_ROUND_CONSTANTS

    bbExternalLinearLayer(&state)

    for r in 0..<4 {
        for i in 0..<16 { state[i] = bbAdd(state[i], rc[r][i]) }
        for i in 0..<16 { state[i] = bbSbox(state[i]) }
        bbExternalLinearLayer(&state)
    }

    for r in 4..<17 {
        state[0] = bbAdd(state[0], rc[r][0])
        state[0] = bbSbox(state[0])
        bbInternalLinearLayer(&state)
    }

    for r in 17..<21 {
        for i in 0..<16 { state[i] = bbAdd(state[i], rc[r][i]) }
        for i in 0..<16 { state[i] = bbSbox(state[i]) }
        bbExternalLinearLayer(&state)
    }
}

// MARK: - Poseidon2 BabyBear Hash

/// Hash two 8-element BabyBear arrays using Poseidon2 sponge (2-to-1 compression for Merkle trees).
public func poseidon2BbHash(left: [Bb], right: [Bb]) -> [Bb] {
    precondition(left.count == 8 && right.count == 8)
    var state = [Bb](repeating: Bb.zero, count: 16)
    for i in 0..<8 { state[i] = left[i] }
    for i in 0..<8 { state[i + 8] = right[i] }

    poseidon2BbPermutation(state: &state)

    return Array(state[0..<8])
}

/// Hash a single block of 8 BabyBear elements.
public func poseidon2BbHashSingle(_ input: [Bb]) -> [Bb] {
    precondition(input.count == 8)
    var state = [Bb](repeating: Bb.zero, count: 16)
    for i in 0..<8 { state[i] = input[i] }
    poseidon2BbPermutation(state: &state)
    return Array(state[0..<8])
}

/// Hash a variable-length array of BabyBear elements using Poseidon2 sponge.
public func poseidon2BbHashMany(_ inputs: [Bb]) -> [Bb] {
    var state = [Bb](repeating: Bb.zero, count: 16)

    var i = 0
    while i < inputs.count {
        for j in 0..<8 {
            if i + j < inputs.count {
                state[j] = bbAdd(state[j], inputs[i + j])
            }
        }
        poseidon2BbPermutation(state: &state)
        i += 8
    }

    return Array(state[0..<8])
}
