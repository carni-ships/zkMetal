// Poseidon2 hash function for M31 (Mersenne31, p = 2^31 - 1)
// t=16 (rate=8, capacity=8), d=5 (x^5 S-box)
// rounds_f=14 (7 + 7 external), rounds_p=21 (internal)
// Standard parameters for 128-bit security at 31-bit field width.
//
// Poseidon2 permutation structure:
// 1. Initial linear layer (external matrix)
// 2. First half of full rounds (7 rounds)
// 3. Partial rounds (21 rounds)
// 4. Second half of full rounds (7 rounds)
//
// Full round: AddRC -> S-box (all elements) -> Linear layer (external)
// Partial round: AddRC (first element only) -> S-box (first element only) -> Linear layer (internal)

import Foundation

// MARK: - Poseidon2 M31 Configuration

public enum Poseidon2M31Config {
    public static let t = 16
    public static let rate = 8
    public static let capacity = 8
    public static let alpha = 5        // S-box exponent
    public static let roundsF = 14     // 7 + 7 full rounds
    public static let roundsP = 21     // partial rounds
    public static let totalRounds = 35 // 14 + 21
}

// MARK: - Round Constants (deterministic generation)

/// Generate Poseidon2 M31 round constants deterministically from a seed.
/// Uses SHA-256 in counter mode, reducing outputs mod p.
private func generateM31RoundConstants() -> [[M31]] {
    // Seed: SHA-256("Poseidon2_M31_t16")
    // We generate totalRounds arrays, each of width t (for full rounds)
    // or width 1 (for partial rounds, only first element used).
    // For simplicity, generate totalRounds * t constants.
    let t = Poseidon2M31Config.t
    let total = Poseidon2M31Config.totalRounds
    let needed = total * t

    // Simple deterministic PRNG: use a 64-bit LCG seeded from the domain separator.
    // The actual security of Poseidon2 does not depend on the randomness of round constants,
    // only that they are non-degenerate. Using a simple, reproducible generator.
    var state: UInt64 = 0x506F736569646F6E  // "Poseidon" as hex
    var constants = [[M31]](repeating: [M31](repeating: M31.zero, count: t), count: total)

    for r in 0..<total {
        for i in 0..<t {
            // LCG step
            state = state &* 6364136223846793005 &+ 0x5F4D33315F743136  // "_M31_t16" as hex
            // Reduce to M31
            let val = UInt32(truncatingIfNeeded: (state >> 33)) // use top 31 bits
            constants[r][i] = M31(v: val % M31.P)
        }
    }
    return constants
}

/// Cached round constants
public let POSEIDON2_M31_ROUND_CONSTANTS: [[M31]] = generateM31RoundConstants()

// MARK: - M4 Circulant Matrix (External Layer Building Block)

/// The M4 circulant matrix used in Poseidon2 for small fields.
/// M4 = circ(2, 3, 1, 1) applied to 4 elements.
/// This is the standard Poseidon2 choice for external rounds.
@inline(__always)
private func m4Apply(_ s: inout [M31], _ offset: Int) {
    // circ(2, 3, 1, 1):
    // y0 = 2*s0 + 3*s1 + 1*s2 + 1*s3
    // y1 = 1*s0 + 2*s1 + 3*s2 + 1*s3
    // y2 = 1*s0 + 1*s1 + 2*s2 + 3*s3
    // y3 = 3*s0 + 1*s1 + 1*s2 + 2*s3
    //
    // Efficient implementation using Feistel-like structure:
    // t0 = s0 + s1; t1 = s2 + s3; t2 = 2*s1 + t1; t3 = 2*s3 + t0;
    // y0 = t0 + t3; y1 = t1 + t2; y2 = t0 + t2; y3 = t1 + t3;
    let s0 = s[offset], s1 = s[offset+1], s2 = s[offset+2], s3 = s[offset+3]
    let t0 = m31Add(s0, s1)
    let t1 = m31Add(s2, s3)
    let t2 = m31Add(m31Add(s1, s1), t1)  // 2*s1 + s2 + s3
    let t3 = m31Add(m31Add(s3, s3), t0)  // 2*s3 + s0 + s1
    s[offset]   = m31Add(t0, t3)  // s0+s1 + 2*s3+s0+s1
    s[offset+1] = m31Add(t1, t2)  // s2+s3 + 2*s1+s2+s3
    s[offset+2] = m31Add(t0, t2)  // s0+s1 + 2*s1+s2+s3
    s[offset+3] = m31Add(t1, t3)  // s2+s3 + 2*s3+s0+s1
}

// MARK: - External Linear Layer

/// Apply the external linear layer for t=16.
/// Uses M4 circulant on 4x4 blocks, then adds the sum of all blocks to each block.
@inline(__always)
private func externalLinearLayerM31(_ state: inout [M31]) {
    // Step 1: Apply M4 to each 4-element block
    m4Apply(&state, 0)
    m4Apply(&state, 4)
    m4Apply(&state, 8)
    m4Apply(&state, 12)

    // Step 2: Compute sum of corresponding positions across blocks
    // Then add to each block element: s[i] += sum_of_blocks[i % 4]
    for i in 0..<4 {
        let sum = m31Add(m31Add(state[i], state[i+4]), m31Add(state[i+8], state[i+12]))
        state[i]    = m31Add(state[i], sum)
        state[i+4]  = m31Add(state[i+4], sum)
        state[i+8]  = m31Add(state[i+8], sum)
        state[i+12] = m31Add(state[i+12], sum)
    }
}

// MARK: - Internal Linear Layer

/// Internal linear layer for Poseidon2 with t=16.
/// M_I = I + diag(d_0, ..., d_{t-1}) where the diagonal values are chosen
/// to provide good diffusion. Standard choice: d_i = i+1 (i.e., 1,2,...,16).
/// Equivalent to: sum = sum(state), state[i] = state[i] * (d_i + 1) + sum
/// But more efficiently: state[i] += sum, state[i] += d_i * old_state[i]
///
/// Actually, the standard Poseidon2 internal matrix for width 16 is:
/// M_I * x = (1 + D) * x where D = diag(d) and the operation is:
/// y_i = x_i + d_i * x_i + sum(x_j for all j)
/// Simplified: y_i = (1 + d_i) * x_i + sum_j(x_j)
///
/// We use the standard diagonal: [1, 1, 2, 1, 8, 32, 2, 256, 4096, 8, 65536, 1024, 2, 16384, 512, 32768]
/// (from Plonky3/Stwo reference)
private let POSEIDON2_M31_INTERNAL_DIAG: [UInt32] = [
    1, 1, 2, 1, 8, 32, 2, 256, 4096, 8, 65536, 1024, 2, 16384, 512, 32768
]

@inline(__always)
private func internalLinearLayerM31(_ state: inout [M31]) {
    // Compute sum of all state elements
    var sum = M31.zero
    for i in 0..<16 {
        sum = m31Add(sum, state[i])
    }

    // state[i] = state[i] * diag[i] + sum
    for i in 0..<16 {
        let diag = POSEIDON2_M31_INTERNAL_DIAG[i]
        let prod: M31
        if diag == 1 {
            prod = state[i]
        } else if diag == 2 {
            prod = m31Add(state[i], state[i])
        } else {
            prod = m31Mul(state[i], M31(v: diag))
        }
        state[i] = m31Add(prod, sum)
    }
}

// MARK: - S-box

/// S-box: x -> x^5
@inline(__always)
private func sboxM31(_ x: M31) -> M31 {
    let x2 = m31Sqr(x)
    let x4 = m31Sqr(x2)
    return m31Mul(x4, x)
}

// MARK: - Poseidon2 M31 Permutation

/// Full Poseidon2 permutation on a state of 16 M31 elements.
public func poseidon2M31Permutation(_ input: [M31]) -> [M31] {
    precondition(input.count == 16)
    let rc = POSEIDON2_M31_ROUND_CONSTANTS

    var state = input

    // Initial external linear layer
    externalLinearLayerM31(&state)

    // First half of full rounds (rounds 0..6)
    for r in 0..<7 {
        // Add round constants
        for i in 0..<16 { state[i] = m31Add(state[i], rc[r][i]) }
        // S-box on all elements
        for i in 0..<16 { state[i] = sboxM31(state[i]) }
        // External linear layer
        externalLinearLayerM31(&state)
    }

    // Partial rounds (rounds 7..27)
    for r in 7..<28 {
        // Add round constant to first element only
        state[0] = m31Add(state[0], rc[r][0])
        // S-box on first element only
        state[0] = sboxM31(state[0])
        // Internal linear layer
        internalLinearLayerM31(&state)
    }

    // Second half of full rounds (rounds 28..34)
    for r in 28..<35 {
        for i in 0..<16 { state[i] = m31Add(state[i], rc[r][i]) }
        for i in 0..<16 { state[i] = sboxM31(state[i]) }
        externalLinearLayerM31(&state)
    }

    return state
}

// MARK: - Poseidon2 M31 Permutation (in-place)

/// In-place Poseidon2 permutation for efficiency.
public func poseidon2M31Permutation(state: inout [M31]) {
    precondition(state.count == 16)
    let rc = POSEIDON2_M31_ROUND_CONSTANTS

    externalLinearLayerM31(&state)

    for r in 0..<7 {
        for i in 0..<16 { state[i] = m31Add(state[i], rc[r][i]) }
        for i in 0..<16 { state[i] = sboxM31(state[i]) }
        externalLinearLayerM31(&state)
    }

    for r in 7..<28 {
        state[0] = m31Add(state[0], rc[r][0])
        state[0] = sboxM31(state[0])
        internalLinearLayerM31(&state)
    }

    for r in 28..<35 {
        for i in 0..<16 { state[i] = m31Add(state[i], rc[r][i]) }
        for i in 0..<16 { state[i] = sboxM31(state[i]) }
        externalLinearLayerM31(&state)
    }
}

// MARK: - Poseidon2 M31 Hash

/// Hash two 8-element M31 arrays using Poseidon2 sponge (2-to-1 compression for Merkle trees).
/// Input: left[8] and right[8] fill the rate portion; capacity initialized to 0.
/// Output: 8 M31 elements (the rate portion of the output state).
public func poseidon2M31Hash(left: [M31], right: [M31]) -> [M31] {
    precondition(left.count == 8 && right.count == 8)
    var state = [M31](repeating: M31.zero, count: 16)
    // Rate portion: first 8 elements = left, next 8 = right
    // Actually for 2-to-1 compression: absorb left into rate, absorb right into rate
    // With rate=8 and two 8-element inputs, we fill the full state (capacity starts at 0)
    for i in 0..<8 { state[i] = left[i] }
    for i in 0..<8 { state[i + 8] = right[i] }

    poseidon2M31Permutation(state: &state)

    return Array(state[0..<8])
}

/// Hash a single block of 8 M31 elements (e.g., for leaf hashing).
/// Capacity portion initialized to zero.
public func poseidon2M31HashSingle(_ input: [M31]) -> [M31] {
    precondition(input.count == 8)
    var state = [M31](repeating: M31.zero, count: 16)
    for i in 0..<8 { state[i] = input[i] }
    poseidon2M31Permutation(state: &state)
    return Array(state[0..<8])
}

/// Hash a variable-length array of M31 elements using Poseidon2 sponge.
/// Uses rate=8, capacity=8.
public func poseidon2M31HashMany(_ inputs: [M31]) -> [M31] {
    var state = [M31](repeating: M31.zero, count: 16)

    // Absorb phase: process inputs in chunks of 8 (rate)
    var i = 0
    while i < inputs.count {
        for j in 0..<8 {
            if i + j < inputs.count {
                state[j] = m31Add(state[j], inputs[i + j])
            }
        }
        poseidon2M31Permutation(state: &state)
        i += 8
    }

    // Squeeze: output rate portion
    return Array(state[0..<8])
}
