// ProofSerializer — Unified proof serialization for cross-system interoperability.
// BN254 and BLS12-381 point compression/decompression (SEC1/ZCash conventions),
// Ethereum blob KZG compressed commitment format, and high-level serialize/deserialize API.
//
// Compressed point format:
//   BN254 G1:  33 bytes — 0x02/0x03 prefix + 32-byte big-endian x coordinate
//   BLS12-381 G1: 48 bytes — high bit of first byte = compression flag (1),
//                             second-highest bit = y odd flag, rest = 48-byte big-endian x
//   BLS12-381 G2: 96 bytes — same convention over Fp2 (c1 || c0, each 48 bytes)
//   Identity:  BN254 = 0x00 * 33, BLS12-381 G1 = 0xC0 + 0x00*47, G2 = 0xC0 + 0x00*95

import Foundation
import NeonFieldOps

// MARK: - BN254 Point Compression (SEC1 / Ethereum style)

/// Compress a BN254 G1 projective point to 33 bytes (SEC1 compressed).
/// Converts Jacobian projective to affine, serializes x as 32-byte big-endian,
/// prefixes with 0x02 (y even) or 0x03 (y odd).
public func bn254G1Compress(_ p: PointProjective) -> [UInt8] {
    if pointIsIdentity(p) {
        return [UInt8](repeating: 0, count: 33)
    }
    guard let aff = pointToAffine(p) else {
        return [UInt8](repeating: 0, count: 33)
    }
    return bn254G1CompressAffine(aff)
}

/// Compress a BN254 G1 affine point to 33 bytes.
public func bn254G1CompressAffine(_ aff: PointAffine) -> [UInt8] {
    let xLimbs = fpToInt(aff.x)
    let yLimbs = fpToInt(aff.y)
    let yOdd = yLimbs[0] & 1 == 1
    var result = [UInt8](repeating: 0, count: 33)
    result[0] = yOdd ? 0x03 : 0x02
    // x in big-endian
    for i in 0..<4 {
        let limb = xLimbs[3 - i]
        result[1 + i * 8 + 0] = UInt8((limb >> 56) & 0xFF)
        result[1 + i * 8 + 1] = UInt8((limb >> 48) & 0xFF)
        result[1 + i * 8 + 2] = UInt8((limb >> 40) & 0xFF)
        result[1 + i * 8 + 3] = UInt8((limb >> 32) & 0xFF)
        result[1 + i * 8 + 4] = UInt8((limb >> 24) & 0xFF)
        result[1 + i * 8 + 5] = UInt8((limb >> 16) & 0xFF)
        result[1 + i * 8 + 6] = UInt8((limb >> 8) & 0xFF)
        result[1 + i * 8 + 7] = UInt8(limb & 0xFF)
    }
    return result
}

/// Decompress a BN254 G1 point from 33 bytes (SEC1 compressed).
/// Returns nil if the point is not on the curve or the prefix is invalid.
public func bn254G1Decompress(_ data: [UInt8]) -> PointProjective? {
    guard data.count == 33 else { return nil }
    let prefix = data[0]

    // Identity
    if prefix == 0x00 {
        return pointIdentity()
    }

    guard prefix == 0x02 || prefix == 0x03 else { return nil }
    let wantOdd = prefix == 0x03

    // Parse x from big-endian
    var xLimbs = [UInt64](repeating: 0, count: 4)
    for i in 0..<4 {
        let base = 1 + (3 - i) * 8
        xLimbs[i] = (UInt64(data[base]) << 56) | (UInt64(data[base + 1]) << 48) |
                     (UInt64(data[base + 2]) << 40) | (UInt64(data[base + 3]) << 32) |
                     (UInt64(data[base + 4]) << 24) | (UInt64(data[base + 5]) << 16) |
                     (UInt64(data[base + 6]) << 8) | UInt64(data[base + 7])
    }

    // Convert to Montgomery form
    let xRaw = Fp.from64(xLimbs)
    let x = fpMul(xRaw, Fp.from64(Fp.R2_MOD_P))

    // y^2 = x^3 + 3 (BN254 curve equation)
    let x2 = fpSqr(x)
    let x3 = fpMul(x2, x)
    let three = fpFromInt(3)
    let rhs = fpAdd(x3, three)

    guard var y = fpSqrt(rhs) else { return nil }

    // Check parity and negate if needed
    let yLimbs = fpToInt(y)
    let isOdd = yLimbs[0] & 1 == 1
    if isOdd != wantOdd {
        y = fpNeg(y)
    }

    return PointProjective(x: x, y: y, z: .one)
}

// MARK: - BLS12-381 Point Compression (ZCash / Ethereum blob KZG style)

/// Compress a BLS12-381 G1 projective point to 48 bytes (ZCash compressed format).
/// High bit = 1 (compressed), second-highest bit = y odd flag, third bit = infinity flag.
public func bls12381G1Compress(_ p: G1Projective381) -> [UInt8] {
    if g1_381IsIdentity(p) {
        // Infinity: 0xC0 followed by 47 zero bytes
        var result = [UInt8](repeating: 0, count: 48)
        result[0] = 0xC0
        return result
    }
    guard let aff = g1_381ToAffine(p) else {
        var result = [UInt8](repeating: 0, count: 48)
        result[0] = 0xC0
        return result
    }
    return bls12381G1CompressAffine(aff)
}

/// Compress a BLS12-381 G1 affine point to 48 bytes.
public func bls12381G1CompressAffine(_ aff: G1Affine381) -> [UInt8] {
    let xLimbs = fp381ToInt(aff.x)
    let yLimbs = fp381ToInt(aff.y)
    let yOdd = yLimbs[0] & 1 == 1

    // Serialize x as 48-byte big-endian
    var result = fp381ToBE48(xLimbs)

    // Set flags in high byte: bit 7 = compressed (1), bit 6 = infinity (0), bit 5 = sort (y parity)
    result[0] |= 0x80  // compressed flag
    if yOdd {
        result[0] |= 0x20  // y odd flag (sort bit)
    }
    return result
}

/// Decompress a BLS12-381 G1 point from 48 bytes (ZCash compressed format).
public func bls12381G1Decompress(_ data: [UInt8]) -> G1Projective381? {
    guard data.count == 48 else { return nil }
    let flags = data[0]

    // Must be compressed
    guard flags & 0x80 != 0 else { return nil }

    // Infinity check
    if flags & 0x40 != 0 {
        return g1_381Identity()
    }

    let wantOdd = flags & 0x20 != 0

    // Parse x: clear flag bits from first byte
    var cleaned = data
    cleaned[0] &= 0x1F
    let xLimbs = be48ToFp381Limbs(cleaned)

    // Convert to Montgomery form
    let xRaw = Fp381.from64(xLimbs)
    let x = fp381Mul(xRaw, Fp381.from64(Fp381.R2_MOD_P))

    // y^2 = x^3 + 4 (BLS12-381 G1 curve equation)
    let x2 = fp381Sqr(x)
    let x3 = fp381Mul(x2, x)
    let four = fp381FromInt(4)
    let rhs = fp381Add(x3, four)

    // Compute sqrt using C function
    var rhsLimbs = rhs.to64()
    var yLimbs = [UInt64](repeating: 0, count: 6)
    let ok = bls12_381_fp_sqrt(&rhsLimbs, &yLimbs)
    guard ok != 0 else { return nil }

    var y = Fp381.from64(yLimbs)

    // Check parity and negate if needed
    let yInt = fp381ToInt(y)
    let isOdd = yInt[0] & 1 == 1
    if isOdd != wantOdd {
        y = fp381Neg(y)
    }

    return G1Projective381(x: x, y: y, z: .one)
}

/// Compress a BLS12-381 G2 projective point to 96 bytes (ZCash compressed format).
/// Fp2 is serialized as c1 (48 bytes) || c0 (48 bytes), big-endian, with flags on first byte.
public func bls12381G2Compress(_ p: G2Projective381) -> [UInt8] {
    if g2_381IsIdentity(p) {
        var result = [UInt8](repeating: 0, count: 96)
        result[0] = 0xC0
        return result
    }
    guard let aff = g2_381ToAffine(p) else {
        var result = [UInt8](repeating: 0, count: 96)
        result[0] = 0xC0
        return result
    }
    return bls12381G2CompressAffine(aff)
}

/// Compress a BLS12-381 G2 affine point to 96 bytes.
public func bls12381G2CompressAffine(_ aff: G2Affine381) -> [UInt8] {
    // For G2 y-parity, use lexicographic ordering on Fp2: compare c1 first, then c0
    let yc1 = fp381ToInt(aff.y.c1)
    let yc0 = fp381ToInt(aff.y.c0)
    let yOdd = fp2_381IsLexLarger(yc1, yc0)

    // x.c1 (48 bytes big-endian) || x.c0 (48 bytes big-endian)
    let xc1Limbs = fp381ToInt(aff.x.c1)
    let xc0Limbs = fp381ToInt(aff.x.c0)
    var result = fp381ToBE48(xc1Limbs) + fp381ToBE48(xc0Limbs)

    // Flags on first byte
    result[0] |= 0x80  // compressed
    if yOdd {
        result[0] |= 0x20  // sort bit
    }
    return result
}

/// Decompress a BLS12-381 G2 point from 96 bytes.
public func bls12381G2Decompress(_ data: [UInt8]) -> G2Projective381? {
    guard data.count == 96 else { return nil }
    let flags = data[0]
    guard flags & 0x80 != 0 else { return nil }

    if flags & 0x40 != 0 {
        return g2_381Identity()
    }

    let wantLarger = flags & 0x20 != 0

    // Parse x.c1 from first 48 bytes, x.c0 from next 48 bytes
    var c1Bytes = Array(data[0..<48])
    c1Bytes[0] &= 0x1F  // clear flags
    let c0Bytes = Array(data[48..<96])

    let xc1Limbs = be48ToFp381Limbs(c1Bytes)
    let xc0Limbs = be48ToFp381Limbs(c0Bytes)

    let xc1 = fp381Mul(Fp381.from64(xc1Limbs), Fp381.from64(Fp381.R2_MOD_P))
    let xc0 = fp381Mul(Fp381.from64(xc0Limbs), Fp381.from64(Fp381.R2_MOD_P))
    let x = Fp2_381(c0: xc0, c1: xc1)

    // y^2 = x^3 + B' where B' = 4(1 + u) for BLS12-381 G2
    let x2 = fp2_381Sqr(x)
    let x3 = fp2_381Mul(x2, x)
    let bPrime = Fp2_381(c0: fp381FromInt(4), c1: fp381FromInt(4))
    let rhs = fp2_381Add(x3, bPrime)

    // Fp2 sqrt: try rhs^((p^2+7)/16) approach
    // For BLS12-381, p = 3 mod 4, so Fp2 sqrt can use the Cipolla/direct method
    guard let y = fp2_381Sqrt(rhs) else { return nil }

    // Check lexicographic ordering and negate if needed
    let yc1Int = fp381ToInt(y.c1)
    let yc0Int = fp381ToInt(y.c0)
    let isLarger = fp2_381IsLexLarger(yc1Int, yc0Int)
    let finalY = isLarger != wantLarger ? fp2_381Neg(y) : y

    return G2Projective381(x: x, y: finalY, z: .one)
}

// MARK: - Fp2_381 Square Root (for G2 decompression)

/// Compute sqrt in Fp2 = Fp[u]/(u^2 + 1) using the algorithm from
/// "Square root computation over even extension fields" (Adj et al.).
/// For BLS12-381, p = 3 mod 4, so we use: sqrt(a) = a^((p^2+7)/16) adjusted.
/// Returns nil if no square root exists.
private func fp2_381Sqrt(_ a: Fp2_381) -> Fp2_381? {
    if a.isZero { return .zero }

    // If c1 == 0, sqrt is just sqrt of c0 (in Fp) if it exists, with c1 = 0
    if a.c1.isZero {
        var c0Limbs = a.c0.to64()
        var rLimbs = [UInt64](repeating: 0, count: 6)
        let ok = bls12_381_fp_sqrt(&c0Limbs, &rLimbs)
        if ok != 0 {
            return Fp2_381(c0: Fp381.from64(rLimbs), c1: .zero)
        }
        // Try sqrt(-c0) * u: if -c0 is a QR, then sqrt(c0) = sqrt(-c0) * u
        var negC0Limbs = fp381Neg(a.c0).to64()
        let ok2 = bls12_381_fp_sqrt(&negC0Limbs, &rLimbs)
        if ok2 != 0 {
            return Fp2_381(c0: .zero, c1: Fp381.from64(rLimbs))
        }
        return nil
    }

    // General case: use the norm-based approach
    // norm = c0^2 + c1^2 (since u^2 = -1)
    let c0sq = fp381Sqr(a.c0)
    let c1sq = fp381Sqr(a.c1)
    let norm = fp381Add(c0sq, c1sq)

    // sqrt(norm) in Fp — if this fails, no Fp2 sqrt exists
    var normLimbs = norm.to64()
    var sqrtNormLimbs = [UInt64](repeating: 0, count: 6)
    let ok = bls12_381_fp_sqrt(&normLimbs, &sqrtNormLimbs)
    guard ok != 0 else { return nil }
    let sqrtNorm = Fp381.from64(sqrtNormLimbs)

    // alpha = (c0 + sqrtNorm) / 2
    let two = fp381FromInt(2)
    let twoInv = fp381Inverse(two)
    var alpha = fp381Mul(fp381Add(a.c0, sqrtNorm), twoInv)

    // Try sqrt(alpha) in Fp
    var alphaLimbs = alpha.to64()
    var sqrtAlphaLimbs = [UInt64](repeating: 0, count: 6)
    var alphaOk = bls12_381_fp_sqrt(&alphaLimbs, &sqrtAlphaLimbs)

    if alphaOk == 0 {
        // Try the other root: alpha = (c0 - sqrtNorm) / 2
        alpha = fp381Mul(fp381Sub(a.c0, sqrtNorm), twoInv)
        alphaLimbs = alpha.to64()
        alphaOk = bls12_381_fp_sqrt(&alphaLimbs, &sqrtAlphaLimbs)
        guard alphaOk != 0 else { return nil }
    }

    let x0 = Fp381.from64(sqrtAlphaLimbs)
    // x1 = c1 / (2 * x0)
    let x1 = fp381Mul(a.c1, fp381Inverse(fp381Double(x0)))

    let candidate = Fp2_381(c0: x0, c1: x1)

    // Verify: candidate^2 == a
    let check = fp2_381Sqr(candidate)
    let diff = fp2_381Sub(check, a)
    guard diff.isZero else { return nil }

    return candidate
}

// MARK: - Fp2 Lexicographic Ordering

/// Returns true if the Fp2 element (c1, c0) is lexicographically larger than its negation.
/// ZCash convention: compare c1 first (as unsigned big integer), then c0.
/// "Larger" means the element is in the upper half of the field.
private func fp2_381IsLexLarger(_ c1Limbs: [UInt64], _ c0Limbs: [UInt64]) -> Bool {
    // If c1 != 0, compare c1 against (p-1)/2
    let c1IsZero = c1Limbs.allSatisfy { $0 == 0 }
    if !c1IsZero {
        return fp381LimbsGtHalfP(c1Limbs)
    }
    // c1 == 0, compare c0
    return fp381LimbsGtHalfP(c0Limbs)
}

/// Returns true if the value (as standard integer limbs) is > (p-1)/2.
private func fp381LimbsGtHalfP(_ limbs: [UInt64]) -> Bool {
    // (p-1)/2 for BLS12-381, computed from Fp381.P
    // p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
    // (p-1)/2 = 0x0d0088f51cbff34d258dd3db21a5d66bb23ba5c279c2895fb398695076d87b120f55ffff58a9ffffdcff7fffffffd555
    let halfP: [UInt64] = [
        0xdcff7fffffffd555, 0x0f55ffff58a9ffff,
        0xb398695076d87b12, 0xb23ba5c279c2895f,
        0x258dd3db21a5d66b, 0x0d0088f51cbff34d
    ]
    // Compare from most significant limb
    for i in stride(from: 5, through: 0, by: -1) {
        if limbs[i] > halfP[i] { return true }
        if limbs[i] < halfP[i] { return false }
    }
    return false  // exactly equal = not larger
}

// MARK: - Byte Conversion Helpers

/// Convert 6 x UInt64 limbs (little-endian) to 48-byte big-endian representation.
private func fp381ToBE48(_ limbs: [UInt64]) -> [UInt8] {
    var bytes = [UInt8](repeating: 0, count: 48)
    for i in 0..<6 {
        let limb = limbs[5 - i]
        bytes[i * 8 + 0] = UInt8((limb >> 56) & 0xFF)
        bytes[i * 8 + 1] = UInt8((limb >> 48) & 0xFF)
        bytes[i * 8 + 2] = UInt8((limb >> 40) & 0xFF)
        bytes[i * 8 + 3] = UInt8((limb >> 32) & 0xFF)
        bytes[i * 8 + 4] = UInt8((limb >> 24) & 0xFF)
        bytes[i * 8 + 5] = UInt8((limb >> 16) & 0xFF)
        bytes[i * 8 + 6] = UInt8((limb >> 8) & 0xFF)
        bytes[i * 8 + 7] = UInt8(limb & 0xFF)
    }
    return bytes
}

/// Parse a 48-byte big-endian representation into 6 x UInt64 limbs (little-endian).
private func be48ToFp381Limbs(_ bytes: [UInt8]) -> [UInt64] {
    var limbs = [UInt64](repeating: 0, count: 6)
    for i in 0..<6 {
        let base = (5 - i) * 8
        limbs[i] = (UInt64(bytes[base]) << 56) | (UInt64(bytes[base + 1]) << 48) |
                   (UInt64(bytes[base + 2]) << 40) | (UInt64(bytes[base + 3]) << 32) |
                   (UInt64(bytes[base + 4]) << 24) | (UInt64(bytes[base + 5]) << 16) |
                   (UInt64(bytes[base + 6]) << 8) | UInt64(bytes[base + 7])
    }
    return limbs
}

// MARK: - KZG Commitment Compressed Serialization (Ethereum blob KZG format)

/// Serialize a KZG commitment (G1 point) in compressed format for Ethereum blob KZG.
/// Returns 48 bytes in ZCash/Ethereum compressed format.
public func kzgCommitmentCompress(_ commitment: PointProjective) -> [UInt8] {
    // BN254 KZG commitments use 33-byte SEC1 compressed format
    bn254G1Compress(commitment)
}

/// Deserialize a KZG commitment from compressed format.
public func kzgCommitmentDecompress(_ data: [UInt8]) -> PointProjective? {
    bn254G1Decompress(data)
}

/// Serialize a KZG commitment on BLS12-381 in compressed format (Ethereum blob KZG).
/// Returns 48 bytes matching the EIP-4844 blob commitment format.
public func kzgCommitmentCompress381(_ commitment: G1Projective381) -> [UInt8] {
    bls12381G1Compress(commitment)
}

/// Deserialize a KZG commitment on BLS12-381 from compressed format.
public func kzgCommitmentDecompress381(_ data: [UInt8]) -> G1Projective381? {
    bls12381G1Decompress(data)
}

// MARK: - KZG Proof Compressed Serialization

extension KZGProof {
    /// Serialize KZG proof in compressed format:
    ///   - 32 bytes: evaluation (Fr, big-endian standard form)
    ///   - 33 bytes: witness commitment (BN254 compressed G1)
    /// Total: 65 bytes.
    public func serializeCompressed() -> [UInt8] {
        var data = [UInt8]()
        data.reserveCapacity(65)
        // Evaluation as 32-byte big-endian (standard form, not Montgomery)
        let evalLimbs = frToInt(evaluation)
        for i in 0..<4 {
            let limb = evalLimbs[3 - i]
            data.append(UInt8((limb >> 56) & 0xFF))
            data.append(UInt8((limb >> 48) & 0xFF))
            data.append(UInt8((limb >> 40) & 0xFF))
            data.append(UInt8((limb >> 32) & 0xFF))
            data.append(UInt8((limb >> 24) & 0xFF))
            data.append(UInt8((limb >> 16) & 0xFF))
            data.append(UInt8((limb >> 8) & 0xFF))
            data.append(UInt8(limb & 0xFF))
        }
        // Witness as compressed G1
        data.append(contentsOf: bn254G1Compress(witness))
        return data
    }

    /// Deserialize KZG proof from compressed format (65 bytes).
    public static func deserializeCompressed(_ data: [UInt8]) -> KZGProof? {
        guard data.count == 65 else { return nil }
        // Parse evaluation from big-endian
        var evalLimbs = [UInt64](repeating: 0, count: 4)
        for i in 0..<4 {
            let base = (3 - i) * 8
            evalLimbs[i] = (UInt64(data[base]) << 56) | (UInt64(data[base + 1]) << 48) |
                           (UInt64(data[base + 2]) << 40) | (UInt64(data[base + 3]) << 32) |
                           (UInt64(data[base + 4]) << 24) | (UInt64(data[base + 5]) << 16) |
                           (UInt64(data[base + 6]) << 8) | UInt64(data[base + 7])
        }
        let evalRaw = Fr.from64(evalLimbs)
        let evaluation = frMul(evalRaw, Fr.from64(Fr.R2_MOD_R))

        // Parse witness
        let witnessBytes = Array(data[32..<65])
        guard let witness = bn254G1Decompress(witnessBytes) else { return nil }

        return KZGProof(evaluation: evaluation, witness: witness)
    }
}

// MARK: - STARK / FRI Proof Binary Serialization (Compact, Length-Prefixed)

/// Compact binary serialization for STARK proofs combining FRI commitment + query proofs.
/// Format: "STARK-v1" label, FRI commitment, query count, query proofs (all length-prefixed).
public struct STARKProofBinary {
    public let commitment: FRICommitment
    public let queries: [FRIQueryProof]
    public let publicInputs: [Fr]

    public init(commitment: FRICommitment, queries: [FRIQueryProof], publicInputs: [Fr]) {
        self.commitment = commitment
        self.queries = queries
        self.publicInputs = publicInputs
    }

    /// Serialize to compact binary format (length-prefixed sections).
    public func serialize() -> [UInt8] {
        let w = ProofWriter()
        w.writeLabel("STARK-v1")

        // FRI commitment (inline, not separately length-prefixed — saves overhead)
        w.writeUInt32(UInt32(commitment.layers.count))
        for layer in commitment.layers {
            w.writeFrArray(layer)
        }
        w.writeFrArray(commitment.roots)
        w.writeFrArray(commitment.betas)
        w.writeFr(commitment.finalValue)

        // Query proofs
        w.writeUInt32(UInt32(queries.count))
        for query in queries {
            w.writeUInt32(query.initialIndex)
            w.writeUInt32(UInt32(query.layerEvals.count))
            for (a, b) in query.layerEvals {
                w.writeFr(a)
                w.writeFr(b)
            }
            w.writeUInt32(UInt32(query.merklePaths.count))
            for layerPath in query.merklePaths {
                w.writeUInt32(UInt32(layerPath.count))
                for siblings in layerPath {
                    w.writeFrArray(siblings)
                }
            }
        }

        // Public inputs
        w.writeFrArray(publicInputs)

        return w.finalize()
    }

    /// Deserialize from compact binary format.
    public static func deserialize(_ data: [UInt8]) throws -> STARKProofBinary {
        let r = ProofReader(data)
        try r.expectLabel("STARK-v1")

        // FRI commitment
        let numLayers = Int(try r.readUInt32())
        var layers = [[Fr]]()
        layers.reserveCapacity(numLayers)
        for _ in 0..<numLayers {
            layers.append(try r.readFrArray())
        }
        let roots = try r.readFrArray()
        let betas = try r.readFrArray()
        let finalValue = try r.readFr()
        let commitment = FRICommitment(layers: layers, roots: roots, betas: betas, finalValue: finalValue)

        // Query proofs
        let numQueries = Int(try r.readUInt32())
        var queries = [FRIQueryProof]()
        queries.reserveCapacity(numQueries)
        for _ in 0..<numQueries {
            let initialIndex = try r.readUInt32()
            let numEvals = Int(try r.readUInt32())
            var layerEvals = [(Fr, Fr)]()
            layerEvals.reserveCapacity(numEvals)
            for _ in 0..<numEvals {
                let a = try r.readFr()
                let b = try r.readFr()
                layerEvals.append((a, b))
            }
            let numPaths = Int(try r.readUInt32())
            var merklePaths = [[[Fr]]]()
            merklePaths.reserveCapacity(numPaths)
            for _ in 0..<numPaths {
                let numSiblings = Int(try r.readUInt32())
                var layerPath = [[Fr]]()
                layerPath.reserveCapacity(numSiblings)
                for _ in 0..<numSiblings {
                    layerPath.append(try r.readFrArray())
                }
                merklePaths.append(layerPath)
            }
            queries.append(FRIQueryProof(initialIndex: initialIndex, layerEvals: layerEvals, merklePaths: merklePaths))
        }

        let publicInputs = try r.readFrArray()

        return STARKProofBinary(commitment: commitment, queries: queries, publicInputs: publicInputs)
    }
}

// MARK: - ProofSerializer (Unified Facade)

/// High-level proof serialization interface for cross-system interoperability.
/// Provides format-specific serialize/deserialize methods for all supported proof types.
public enum ProofSerializer {

    // MARK: Groth16 (snarkjs JSON)

    /// Serialize a Groth16 proof to snarkjs-compatible JSON bytes.
    public static func groth16ToJSON(_ proof: Groth16Proof, prettyPrint: Bool = true) -> Data? {
        proof.toSnarkjsJSON(prettyPrint: prettyPrint)
    }

    /// Deserialize a Groth16 proof from snarkjs-compatible JSON bytes.
    public static func groth16FromJSON(_ data: Data) -> Groth16Proof? {
        Groth16Proof.fromSnarkjsJSON(data)
    }

    /// Serialize a Groth16 verification key to snarkjs-compatible JSON bytes.
    public static func groth16VKToJSON(_ vk: Groth16VerificationKey, prettyPrint: Bool = true) -> Data? {
        vk.toSnarkjsJSON(prettyPrint: prettyPrint)
    }

    /// Deserialize a Groth16 verification key from snarkjs-compatible JSON bytes.
    public static func groth16VKFromJSON(_ data: Data) -> Groth16VerificationKey? {
        Groth16VerificationKey.fromSnarkjsJSON(data)
    }

    /// Serialize a Groth16 proof for Ethereum ABI (on-chain verification).
    public static func groth16ToABI(_ proof: Groth16Proof, publicInputs: [Fr]) -> [UInt8]? {
        EthereumABIEncoder.encodeProof(proof: proof, publicInputs: publicInputs)
    }

    // MARK: FRI / STARK (compact binary)

    /// Serialize a STARK proof (FRI commitment + queries) to compact binary.
    public static func starkToBytes(_ commitment: FRICommitment, queries: [FRIQueryProof],
                                     publicInputs: [Fr]) -> [UInt8] {
        STARKProofBinary(commitment: commitment, queries: queries, publicInputs: publicInputs).serialize()
    }

    /// Deserialize a STARK proof from compact binary.
    public static func starkFromBytes(_ data: [UInt8]) throws -> STARKProofBinary {
        try STARKProofBinary.deserialize(data)
    }

    /// Serialize a standalone FRI commitment to binary.
    public static func friCommitmentToBytes(_ commitment: FRICommitment) -> [UInt8] {
        commitment.serialize()
    }

    /// Deserialize a standalone FRI commitment from binary.
    public static func friCommitmentFromBytes(_ data: [UInt8]) throws -> FRICommitment {
        try FRICommitment.deserialize(data)
    }

    // MARK: KZG (compressed points)

    /// Serialize a KZG proof in compressed format (BN254, 65 bytes).
    public static func kzgProofToCompressed(_ proof: KZGProof) -> [UInt8] {
        proof.serializeCompressed()
    }

    /// Deserialize a KZG proof from compressed format (BN254, 65 bytes).
    public static func kzgProofFromCompressed(_ data: [UInt8]) -> KZGProof? {
        KZGProof.deserializeCompressed(data)
    }

    /// Serialize a KZG proof in the existing binary format (labeled, length-prefixed).
    public static func kzgProofToBytes(_ proof: KZGProof) -> [UInt8] {
        proof.serialize()
    }

    /// Deserialize a KZG proof from binary format.
    public static func kzgProofFromBytes(_ data: [UInt8]) throws -> KZGProof {
        try KZGProof.deserialize(data)
    }

    // MARK: Point Compression (BN254)

    /// Compress a BN254 G1 point to 33 bytes (SEC1 compressed).
    public static func compressBN254G1(_ point: PointProjective) -> [UInt8] {
        bn254G1Compress(point)
    }

    /// Decompress a BN254 G1 point from 33 bytes.
    public static func decompressBN254G1(_ data: [UInt8]) -> PointProjective? {
        bn254G1Decompress(data)
    }

    // MARK: Point Compression (BLS12-381)

    /// Compress a BLS12-381 G1 point to 48 bytes (ZCash/Ethereum blob KZG format).
    public static func compressBLS12381G1(_ point: G1Projective381) -> [UInt8] {
        bls12381G1Compress(point)
    }

    /// Decompress a BLS12-381 G1 point from 48 bytes.
    public static func decompressBLS12381G1(_ data: [UInt8]) -> G1Projective381? {
        bls12381G1Decompress(data)
    }

    /// Compress a BLS12-381 G2 point to 96 bytes (ZCash compressed format).
    public static func compressBLS12381G2(_ point: G2Projective381) -> [UInt8] {
        bls12381G2Compress(point)
    }

    /// Decompress a BLS12-381 G2 point from 96 bytes.
    public static func decompressBLS12381G2(_ data: [UInt8]) -> G2Projective381? {
        bls12381G2Decompress(data)
    }

    // MARK: Envelope (generic proof container)

    /// Wrap any proof in a ProofEnvelope for routing and versioning.
    public static func toEnvelope(scheme: ProofScheme, curve: CurveIdentifier,
                                   proofBytes: [UInt8], publicInputs: [Fr],
                                   metadata: [String: String]? = nil) -> ProofEnvelope {
        ProofEnvelope.binary(scheme: scheme, proofBytes: proofBytes,
                             publicInputs: publicInputs, curve: curve, metadata: metadata)
    }

    /// Wrap a Groth16 proof in a ProofEnvelope with snarkjs-format payload.
    public static func groth16ToEnvelope(_ proof: Groth16Proof, publicInputs: [Fr],
                                          curve: CurveIdentifier = .bn254,
                                          metadata: [String: String]? = nil) -> ProofEnvelope {
        ProofEnvelope.groth16(proof: proof, publicInputs: publicInputs,
                              curve: curve, metadata: metadata)
    }
}
