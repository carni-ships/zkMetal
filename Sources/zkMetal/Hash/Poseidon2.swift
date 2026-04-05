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
public func poseidon2HashMany(_ inputs: [Fr]) -> Fr {
    var s0 = Fr.zero, s1 = Fr.zero, s2 = Fr.zero

    var i = 0
    while i < inputs.count {
        // Absorb: add inputs to rate portion of state
        withUnsafePointer(to: s0) { s0Ptr in
            withUnsafePointer(to: inputs[i]) { inPtr in
                var r = Fr.zero
                withUnsafeMutablePointer(to: &r) { rPtr in
                    bn254_fr_add(
                        UnsafeRawPointer(s0Ptr).assumingMemoryBound(to: UInt64.self),
                        UnsafeRawPointer(inPtr).assumingMemoryBound(to: UInt64.self),
                        UnsafeMutableRawPointer(rPtr).assumingMemoryBound(to: UInt64.self)
                    )
                }
                s0 = r
            }
        }
        if i + 1 < inputs.count {
            withUnsafePointer(to: s1) { s1Ptr in
                withUnsafePointer(to: inputs[i + 1]) { inPtr in
                    var r = Fr.zero
                    withUnsafeMutablePointer(to: &r) { rPtr in
                        bn254_fr_add(
                            UnsafeRawPointer(s1Ptr).assumingMemoryBound(to: UInt64.self),
                            UnsafeRawPointer(inPtr).assumingMemoryBound(to: UInt64.self),
                            UnsafeMutableRawPointer(rPtr).assumingMemoryBound(to: UInt64.self)
                        )
                    }
                    s1 = r
                }
            }
        }

        // Permute
        var state = [s0, s1, s2]
        var result = [Fr](repeating: Fr.zero, count: 3)
        state.withUnsafeBytes { inPtr in
            result.withUnsafeMutableBytes { outPtr in
                poseidon2_permutation_cpu(
                    inPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    outPtr.baseAddress!.assumingMemoryBound(to: UInt64.self)
                )
            }
        }
        s0 = result[0]; s1 = result[1]; s2 = result[2]
        i += 2
    }

    return s0
}
