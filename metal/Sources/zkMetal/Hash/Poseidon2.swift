// Poseidon2 hash function for BN254 Fr
// t=3 (rate=2, capacity=1), d=5, rounds_f=8, rounds_p=56
//
// Poseidon2 permutation structure:
// 1. Initial linear layer (external matrix)
// 2. First half of full rounds (4 rounds)
// 3. Partial rounds (56 rounds)
// 4. Second half of full rounds (4 rounds)
//
// Full round: AddRC -> S-box (all elements) -> Linear layer (external)
// Partial round: AddRC (first element only) -> S-box (first element only) -> Linear layer (internal)

import Foundation

// MARK: - Poseidon2 Permutation (CPU)

/// Apply the external linear layer (circulant matrix [2,1,1] for t=3).
/// M_E * [a,b,c] = [2a+b+c, a+2b+c, a+b+2c]
@inline(__always)
private func externalLinearLayer(_ state: inout [Fr]) {
    let sum = frAdd(frAdd(state[0], state[1]), state[2])
    state[0] = frAdd(state[0], sum)
    state[1] = frAdd(state[1], sum)
    state[2] = frAdd(state[2], sum)
}

/// Apply the internal linear layer.
/// M_I = [[2,1,1],[1,2,1],[1,1,3]]
/// M_I * [a,b,c] = [2a+b+c, a+2b+c, a+b+3c]
@inline(__always)
private func internalLinearLayer(_ state: inout [Fr]) {
    let sum = frAdd(frAdd(state[0], state[1]), state[2])
    state[0] = frAdd(state[0], sum)    // a + (a+b+c) = 2a+b+c
    state[1] = frAdd(state[1], sum)    // b + (a+b+c) = a+2b+c
    // state[2] gets diag=2 extra: c + (a+b+c) + c = a+b+3c
    state[2] = frAdd(frAdd(state[2], sum), state[2])
}

/// S-box: x -> x^5
@inline(__always)
private func sbox(_ x: Fr) -> Fr {
    let x2 = frSqr(x)
    let x4 = frSqr(x2)
    return frMul(x4, x)
}

/// Full Poseidon2 permutation on a state of 3 Fr elements.
public func poseidon2Permutation(_ input: [Fr]) -> [Fr] {
    precondition(input.count == 3)
    let rc = POSEIDON2_ROUND_CONSTANTS

    var state = input

    // Initial external linear layer
    externalLinearLayer(&state)

    // First half of full rounds (rounds 0..3)
    for r in 0..<4 {
        // Add round constants
        for i in 0..<3 { state[i] = frAdd(state[i], rc[r][i]) }
        // S-box on all elements
        for i in 0..<3 { state[i] = sbox(state[i]) }
        // External linear layer
        externalLinearLayer(&state)
    }

    // Partial rounds (rounds 4..59)
    for r in 4..<60 {
        // Add round constant to first element only
        state[0] = frAdd(state[0], rc[r][0])
        // S-box on first element only
        state[0] = sbox(state[0])
        // Internal linear layer
        internalLinearLayer(&state)
    }

    // Second half of full rounds (rounds 60..63)
    for r in 60..<64 {
        for i in 0..<3 { state[i] = frAdd(state[i], rc[r][i]) }
        for i in 0..<3 { state[i] = sbox(state[i]) }
        externalLinearLayer(&state)
    }

    return state
}

// MARK: - Poseidon2 Hash (Sponge)

/// Hash two field elements using Poseidon2 sponge (2-to-1 compression).
/// Input: [a, b], capacity element = 0
/// Output: state[0] after permutation
public func poseidon2Hash(_ a: Fr, _ b: Fr) -> Fr {
    let state = [a, b, Fr.zero]
    let result = poseidon2Permutation(state)
    return result[0]
}

/// Hash a variable-length array of field elements using Poseidon2 sponge.
/// Uses rate=2, capacity=1.
public func poseidon2HashMany(_ inputs: [Fr]) -> Fr {
    var state = [Fr.zero, Fr.zero, Fr.zero]

    // Absorb phase: process inputs in chunks of 2 (rate)
    var i = 0
    while i < inputs.count {
        state[0] = frAdd(state[0], inputs[i])
        if i + 1 < inputs.count {
            state[1] = frAdd(state[1], inputs[i + 1])
        }
        state = poseidon2Permutation(state)
        i += 2
    }

    // Squeeze: output first element
    return state[0]
}
