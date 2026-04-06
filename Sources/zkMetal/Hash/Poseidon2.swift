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
import NeonFieldOps

// MARK: - Poseidon2 Permutation (CPU) — C CIOS accelerated

/// Full Poseidon2 permutation on a state of 3 Fr elements.
/// Delegates to C CIOS Montgomery implementation for ~10-30x speedup.
public func poseidon2Permutation(_ input: [Fr]) -> [Fr] {
    precondition(input.count == 3)
    var result = [Fr](repeating: Fr.zero, count: 3)
    input.withUnsafeBytes { inPtr in
        result.withUnsafeMutableBytes { outPtr in
            poseidon2_permutation_cpu(
                inPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                outPtr.baseAddress!.assumingMemoryBound(to: UInt64.self)
            )
        }
    }
    return result
}

/// In-place Poseidon2 permutation on a 3-element tuple — zero allocation.
/// Use this in hot loops (sponge, transcript) to avoid array allocation overhead.
@inline(__always)
public func poseidon2PermuteInPlace(_ s0: inout Fr, _ s1: inout Fr, _ s2: inout Fr) {
    // Build contiguous input/output on stack (no heap allocation)
    var input = (s0, s1, s2)
    var output: (Fr, Fr, Fr) = (Fr.zero, Fr.zero, Fr.zero)
    withUnsafeBytes(of: &input) { inBuf in
        withUnsafeMutableBytes(of: &output) { outBuf in
            poseidon2_permutation_cpu(
                inBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                outBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
            )
        }
    }
    s0 = output.0
    s1 = output.1
    s2 = output.2
}

// MARK: - Poseidon2 Hash (Sponge) — C CIOS accelerated

/// Hash two field elements using Poseidon2 sponge (2-to-1 compression).
/// Input: [a, b], capacity element = 0
/// Output: state[0] after permutation
public func poseidon2Hash(_ a: Fr, _ b: Fr) -> Fr {
    var result = Fr.zero
    withUnsafePointer(to: a) { aPtr in
        withUnsafePointer(to: b) { bPtr in
            withUnsafeMutablePointer(to: &result) { rPtr in
                poseidon2_hash_cpu(
                    UnsafeRawPointer(aPtr).assumingMemoryBound(to: UInt64.self),
                    UnsafeRawPointer(bPtr).assumingMemoryBound(to: UInt64.self),
                    UnsafeMutableRawPointer(rPtr).assumingMemoryBound(to: UInt64.self)
                )
            }
        }
    }
    return result
}

/// Hash a variable-length array of field elements using Poseidon2 sponge.
/// Uses rate=2, capacity=1. Uses C CIOS for the permutation.
/// Zero-allocation inner loop: uses stack-based tuple permutation.
public func poseidon2HashMany(_ inputs: [Fr]) -> Fr {
    var s0 = Fr.zero, s1 = Fr.zero, s2 = Fr.zero

    var i = 0
    while i < inputs.count {
        // Absorb: add inputs to rate portion of state
        s0 = frAdd(s0, inputs[i])
        if i + 1 < inputs.count {
            s1 = frAdd(s1, inputs[i + 1])
        }

        // Permute (zero-allocation, stack-based)
        poseidon2PermuteInPlace(&s0, &s1, &s2)
        i += 2
    }

    return s0
}
