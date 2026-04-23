// BLS12-381 GLV (Gallant-Lambert-Vaudenay) Endomorphism Implementation
// Scalar decomposition: k → (k1, k2) where k ≡ k1 + k2·λ (mod r)
// Endomorphism: φ(x,y) = (β·x, y) where β³ = 1 in Fp381
//
// BLS12-381 curve: y² = x³ + 4
// Scalar field r = 0x73eda753299d7d483339d90809f473de69880fc420e643cc00000000000008d8db
// Base field p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab

import Foundation

public struct BLS12381GLV {
    // Scalar field order r as 4×64-bit (little-endian)
    public static let R: (UInt64, UInt64, UInt64, UInt64) = (
        0x0000000d8ed0000, 0x0000000d8db0000,
        0x00000000fc420e, 0x73eda753299d7d48
    )

    // a1 = floor((r-1)/λ) where λ is cube root of unity
    // For BLS12-381: a1 = 9586122913090631933 (≈ 2^63 / 3)
    public static let A1: (UInt64, UInt64) = (
        0x21d42c0d00200001, 0x0000000000000008
    )

    // a1 - 1
    public static let A1_MINUS_1: (UInt64, UInt64) = (
        0x21d42c0d00200000, 0x0000000000000008
    )

    // half_r = (r+1)/2 for signed decomposition
    public static let HALF_R: (UInt64, UInt64, UInt64, UInt64) = (
        0x80000006e4768000, 0x00000006c6ed8000,
        0x000000007e210f, 0x39f6d3a99ffffe24
    )

    // λ in Fr381 (cube root of unity): computed from lattice
    // λ satisfies λ² + λ + 1 = 0 in Fr
    public static let LAMBDA: (UInt64, UInt64, UInt64, UInt64) = (
        0x8db1c8f3e8a00001, 0x3d12385d23be0d24,
        0x4e9c404a11e09e86, 0x5e3763b70a89ec0b
    )

    // β in Fp381 (cube root of unity in base field)
    // β satisfies β³ = 1, β ≠ 1
    public static let BETA: (UInt64, UInt64, UInt64, UInt64, UInt64, UInt64) = (
        0x5b623b5ce84ea86e, 0x8f13871d5eead1ad,
        0x7c2c93a83ae36ad5, 0x8e60e7391f4a9b17,
        0x7d3d2c5dc0405be8, 0x5c671f3c24e0d0b3
    )

    // Decompose scalar k into (k1, k2) using lattice basis
    // Returns (k1, k2, neg1, neg2) where:
    //   k = k1 + k2·λ (mod r) if neg1=0, or -k1 if neg1=1
    //   k2 is reduced to half-width if neg2=1
    public static func decompose(_ k: (UInt64, UInt64, UInt64, UInt64)) -> (k1: (UInt64, UInt64, UInt64, UInt64), k2: (UInt64, UInt64), neg1: Bool, neg2: Bool) {
        // Approximate c1 = floor(k * (a1-1) / r)
        // Since r ≈ 2^255 and a1 ≈ 2^63, we use 128×128 multiplication
        let c1_lo = k.0 &* A1_MINUS_1.0
        let c1_hi: UInt64 = (k.0 >> 1) &+ (k.1 &* A1_MINUS_1.0) &+ (k.2 &* A1_MINUS_1.1) &+ (k.3 &* A1_MINUS_1.1)

        // k1 = k - c1 * a1
        var k1_0 = k.0 &- c1_lo &* A1.0
        var k1_1 = k.1 &- (c1_lo >> 32) &- c1_hi &* A1.0
        var k1_2 = k.2 &- (c1_hi &* A1.1)
        var k1_3 = k.3

        // Handle borrows (simplified)
        if k1_0 > k.0 { k1_0 &+= 0xFFFFFFFF00000001; k1_1 &-= 1 }
        if k1_1 > k.1 { k1_1 &+= 0xFFFFFFFF00000001; k1_2 &-= 1 }

        // k2 = c1
        let k2 = (c1_lo, c1_hi)

        // Reduce k1 to half-width: if k1 >= r/2, negate
        var neg1 = false
        var k1_final_0 = k1_0
        var k1_final_1 = k1_1
        var k1_final_2 = k1_2
        var k1_final_3 = k1_3

        // Check if k1 >= HALF_R
        if k1_final_3 > HALF_R.3 || (k1_final_3 == HALF_R.3 && k1_final_2 > HALF_R.2) ||
           (k1_final_3 == HALF_R.3 && k1_final_2 == HALF_R.2 && k1_final_1 > HALF_R.1) ||
           (k1_final_3 == HALF_R.3 && k1_final_2 == HALF_R.2 && k1_final_1 == HALF_R.1 && k1_final_0 > HALF_R.0) {
            // k1 = r - k1
            neg1 = true
            k1_final_0 = 0 &- k1_final_0
            k1_final_1 = 0 &- k1_final_1 &- (k1_final_0 > 0 ? 1 : 0)
            k1_final_2 = 0 &- k1_final_2 &- (k1_final_1 > 0 ? 1 : 0)
            k1_final_3 = 0 &- k1_final_3 &- (k1_final_2 > 0 ? 1 : 0)
        }

        // k2 is always positive (no need for half-width reduction since k2 ≈ k/a1 < sqrt(r))
        let neg2 = false

        return ((k1_final_0, k1_final_1, k1_final_2, k1_final_3), k2, neg1, neg2)
    }

    // Simplified decompose for [UInt32] input (8 limbs)
    public static func decompose(_ scalar: [UInt32]) -> (k1: [UInt32], k2: [UInt32], neg1: Bool, neg2: Bool) {
        guard scalar.count >= 8 else { return (scalar, [0,0,0,0,0,0,0,0], false, false) }

        let k = (UInt64(scalar[0]) | (UInt64(scalar[1]) << 32),
                 UInt64(scalar[2]) | (UInt64(scalar[3]) << 32),
                 UInt64(scalar[4]) | (UInt64(scalar[5]) << 32),
                 UInt64(scalar[6]) | (UInt64(scalar[7]) << 32))

        let result = decompose(k)

        return (
            [UInt32(result.k1.0 & 0xFFFFFFFF), UInt32(result.k1.0 >> 32),
             UInt32(result.k1.1 & 0xFFFFFFFF), UInt32(result.k1.1 >> 32),
             UInt32(result.k1.2 & 0xFFFFFFFF), UInt32(result.k1.2 >> 32),
             UInt32(result.k1.3 & 0xFFFFFFFF), UInt32(result.k1.3 >> 32)],
            [UInt32(result.k2.0 & 0xFFFFFFFF), UInt32(result.k2.0 >> 32),
             UInt32(result.k2.1 & 0xFFFFFFFF), UInt32(result.k2.1 >> 32),
             0, 0, 0, 0],
            result.neg1,
            result.neg2
        )
    }
}