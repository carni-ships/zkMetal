// Poseidon hash for Pasta curves (Mina Kimchi variant) — CPU implementation
// Delegates to C CIOS Montgomery arithmetic for performance.
// 55 full rounds, x^7 S-box, full MDS, width=3, rate=2, capacity=1
//
// Reference: o1-labs/proof-systems/poseidon

import Foundation
import NeonFieldOps

// MARK: - Pallas Poseidon (CPU)

/// Full Pallas Poseidon permutation on a state of 3 PallasFp elements.
public func pallasPoseidonPermutation(_ input: [PallasFp]) -> [PallasFp] {
    precondition(input.count == 3)
    var result = [PallasFp](repeating: PallasFp.zero, count: 3)
    input.withUnsafeBytes { inPtr in
        result.withUnsafeMutableBytes { outPtr in
            pallas_poseidon_permutation_cpu(
                inPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                outPtr.baseAddress!.assumingMemoryBound(to: UInt64.self)
            )
        }
    }
    return result
}

/// In-place Pallas Poseidon permutation — zero allocation.
@inline(__always)
public func pallasPoseidonPermuteInPlace(_ s0: inout PallasFp, _ s1: inout PallasFp, _ s2: inout PallasFp) {
    var input = (s0, s1, s2)
    var output: (PallasFp, PallasFp, PallasFp) = (PallasFp.zero, PallasFp.zero, PallasFp.zero)
    withUnsafeBytes(of: &input) { inBuf in
        withUnsafeMutableBytes(of: &output) { outBuf in
            pallas_poseidon_permutation_cpu(
                inBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                outBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
            )
        }
    }
    s0 = output.0
    s1 = output.1
    s2 = output.2
}

/// Hash two Pallas field elements using Poseidon sponge (2-to-1 compression).
/// hash(a, b) = permute([a, b, 0])[0]
public func pallasPoseidonHash(_ a: PallasFp, _ b: PallasFp) -> PallasFp {
    var result = PallasFp.zero
    withUnsafePointer(to: a) { aPtr in
        withUnsafePointer(to: b) { bPtr in
            withUnsafeMutablePointer(to: &result) { rPtr in
                pallas_poseidon_hash_cpu(
                    UnsafeRawPointer(aPtr).assumingMemoryBound(to: UInt64.self),
                    UnsafeRawPointer(bPtr).assumingMemoryBound(to: UInt64.self),
                    UnsafeMutableRawPointer(rPtr).assumingMemoryBound(to: UInt64.self)
                )
            }
        }
    }
    return result
}

/// Hash a variable-length array of Pallas field elements using Poseidon sponge.
/// Uses rate=2, capacity=1. Zero-allocation inner loop.
public func pallasPoseidonHashMany(_ inputs: [PallasFp]) -> PallasFp {
    var result = PallasFp.zero
    inputs.withUnsafeBytes { inPtr in
        withUnsafeMutablePointer(to: &result) { rPtr in
            pallas_poseidon_hash_many_cpu(
                inPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                Int32(inputs.count),
                UnsafeMutableRawPointer(rPtr).assumingMemoryBound(to: UInt64.self)
            )
        }
    }
    return result
}

// MARK: - Vesta Poseidon (CPU)

/// Full Vesta Poseidon permutation on a state of 3 VestaFp elements.
public func vestaPoseidonPermutation(_ input: [VestaFp]) -> [VestaFp] {
    precondition(input.count == 3)
    var result = [VestaFp](repeating: VestaFp.zero, count: 3)
    input.withUnsafeBytes { inPtr in
        result.withUnsafeMutableBytes { outPtr in
            vesta_poseidon_permutation_cpu(
                inPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                outPtr.baseAddress!.assumingMemoryBound(to: UInt64.self)
            )
        }
    }
    return result
}

/// In-place Vesta Poseidon permutation — zero allocation.
@inline(__always)
public func vestaPoseidonPermuteInPlace(_ s0: inout VestaFp, _ s1: inout VestaFp, _ s2: inout VestaFp) {
    var input = (s0, s1, s2)
    var output: (VestaFp, VestaFp, VestaFp) = (VestaFp.zero, VestaFp.zero, VestaFp.zero)
    withUnsafeBytes(of: &input) { inBuf in
        withUnsafeMutableBytes(of: &output) { outBuf in
            vesta_poseidon_permutation_cpu(
                inBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                outBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
            )
        }
    }
    s0 = output.0
    s1 = output.1
    s2 = output.2
}

/// Hash two Vesta field elements using Poseidon sponge (2-to-1 compression).
public func vestaPoseidonHash(_ a: VestaFp, _ b: VestaFp) -> VestaFp {
    var result = VestaFp.zero
    withUnsafePointer(to: a) { aPtr in
        withUnsafePointer(to: b) { bPtr in
            withUnsafeMutablePointer(to: &result) { rPtr in
                vesta_poseidon_hash_cpu(
                    UnsafeRawPointer(aPtr).assumingMemoryBound(to: UInt64.self),
                    UnsafeRawPointer(bPtr).assumingMemoryBound(to: UInt64.self),
                    UnsafeMutableRawPointer(rPtr).assumingMemoryBound(to: UInt64.self)
                )
            }
        }
    }
    return result
}

/// Hash a variable-length array of Vesta field elements using Poseidon sponge.
public func vestaPoseidonHashMany(_ inputs: [VestaFp]) -> VestaFp {
    var result = VestaFp.zero
    inputs.withUnsafeBytes { inPtr in
        withUnsafeMutablePointer(to: &result) { rPtr in
            vesta_poseidon_hash_many_cpu(
                inPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                Int32(inputs.count),
                UnsafeMutableRawPointer(rPtr).assumingMemoryBound(to: UInt64.self)
            )
        }
    }
    return result
}
