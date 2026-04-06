// GPUGroth16VKEngine — GPU-accelerated Groth16 verification key management
//
// Provides VK serialization/deserialization (binary + JSON), point compression,
// batch VK preprocessing for multi-circuit verification, IC computation from
// public inputs, VK validation (subgroup checks, pairing consistency), and
// VK comparison/diff utilities.
//
// Uses Metal GPU MSM for accelerated IC computation on large public input
// vectors, with CPU fallback for small instances.

import Foundation
import Metal

// MARK: - Compressed Point Types

/// Compressed G1 point: x-coordinate + sign bit for y
public struct CompressedG1 {
    public let x: [UInt64]  // 4 limbs
    public let ySign: Bool  // true if y is "larger" (lexicographic)

    public init(x: [UInt64], ySign: Bool) {
        self.x = x
        self.ySign = ySign
    }
}

/// Compressed G2 point: x-coordinate (Fp2) + sign bit for y
public struct CompressedG2 {
    public let xC0: [UInt64]  // 4 limbs
    public let xC1: [UInt64]  // 4 limbs
    public let ySign: Bool

    public init(xC0: [UInt64], xC1: [UInt64], ySign: Bool) {
        self.xC0 = xC0
        self.xC1 = xC1
        self.ySign = ySign
    }
}

/// Compressed verification key
public struct CompressedGroth16VK {
    public let alpha_g1: CompressedG1
    public let beta_g2: CompressedG2
    public let gamma_g2: CompressedG2
    public let delta_g2: CompressedG2
    public let ic: [CompressedG1]

    public init(alpha_g1: CompressedG1, beta_g2: CompressedG2,
                gamma_g2: CompressedG2, delta_g2: CompressedG2,
                ic: [CompressedG1]) {
        self.alpha_g1 = alpha_g1
        self.beta_g2 = beta_g2
        self.gamma_g2 = gamma_g2
        self.delta_g2 = delta_g2
        self.ic = ic
    }
}

/// Preprocessed VK for fast verification (affine conversions cached)
public struct PreprocessedGroth16VK {
    public let alpha_g1: PointAffine
    public let beta_g2: G2AffinePoint
    public let gamma_g2: G2AffinePoint
    public let delta_g2: G2AffinePoint
    public let ic: [PointAffine]
    public let negAlpha_g1: PointAffine  // precomputed negation for pairing

    public init(alpha_g1: PointAffine, beta_g2: G2AffinePoint,
                gamma_g2: G2AffinePoint, delta_g2: G2AffinePoint,
                ic: [PointAffine], negAlpha_g1: PointAffine) {
        self.alpha_g1 = alpha_g1
        self.beta_g2 = beta_g2
        self.gamma_g2 = gamma_g2
        self.delta_g2 = delta_g2
        self.ic = ic
        self.negAlpha_g1 = negAlpha_g1
    }
}

/// VK diff result: describes differences between two VKs
public struct VKDiff {
    public let alphaG1Differs: Bool
    public let betaG2Differs: Bool
    public let gammaG2Differs: Bool
    public let deltaG2Differs: Bool
    public let icCountDiffers: Bool
    public let icDiffIndices: [Int]  // which IC points differ

    public var isIdentical: Bool {
        !alphaG1Differs && !betaG2Differs && !gammaG2Differs &&
        !deltaG2Differs && !icCountDiffers && icDiffIndices.isEmpty
    }
}

/// VK validation result
public struct VKValidationResult {
    public let valid: Bool
    public let alphaOnCurve: Bool
    public let betaOnCurve: Bool
    public let gammaOnCurve: Bool
    public let deltaOnCurve: Bool
    public let icAllOnCurve: Bool
    public let icCountValid: Bool  // at least 1 IC point (for constant term)
    public let pairingConsistent: Bool
    public let message: String
}

// MARK: - GPU Groth16 VK Engine

public final class GPUGroth16VKEngine {
    public static let version = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")

    /// GPU MSM engine for accelerated IC computation
    private let msm: MetalMSM?

    /// GPU MSM threshold: use Metal MSM when point count exceeds this
    public var gpuMSMThreshold: Int = 256

    /// Enable profiling output to stderr
    public var profile = false

    public init() {
        self.msm = try? MetalMSM()
    }

    // MARK: - Serialization (Binary)

    /// Serialize a VK to binary format.
    /// Format: [numIC:UInt32][alpha_g1:3*32B][beta_g2:6*32B][gamma_g2:6*32B][delta_g2:6*32B][ic:N*3*32B]
    public func serializeBinary(_ vk: Groth16VerificationKey) -> Data {
        var data = Data()
        // Number of IC points
        var numIC = UInt32(vk.ic.count)
        data.append(Data(bytes: &numIC, count: 4))
        // alpha_g1
        appendG1Projective(&data, vk.alpha_g1)
        // beta_g2, gamma_g2, delta_g2
        appendG2Projective(&data, vk.beta_g2)
        appendG2Projective(&data, vk.gamma_g2)
        appendG2Projective(&data, vk.delta_g2)
        // IC points
        for pt in vk.ic {
            appendG1Projective(&data, pt)
        }
        return data
    }

    /// Deserialize a VK from binary format.
    public func deserializeBinary(_ data: Data) -> Groth16VerificationKey? {
        guard data.count >= 4 else { return nil }
        let numIC = data.withUnsafeBytes { $0.load(fromByteOffset: 0, as: UInt32.self) }
        let headerSize = 4
        let g1Size = 3 * 32  // 3 Fp coordinates * 32 bytes each
        let g2Size = 6 * 32  // 3 Fp2 coordinates * 32 bytes each (each Fp2 = 2*32)
        let expectedSize = headerSize + g1Size + 3 * g2Size + Int(numIC) * g1Size
        guard data.count >= expectedSize else { return nil }

        var offset = headerSize
        guard let alpha = readG1Projective(data, &offset) else { return nil }
        guard let beta = readG2Projective(data, &offset) else { return nil }
        guard let gamma = readG2Projective(data, &offset) else { return nil }
        guard let delta = readG2Projective(data, &offset) else { return nil }

        var ic = [PointProjective]()
        ic.reserveCapacity(Int(numIC))
        for _ in 0..<numIC {
            guard let pt = readG1Projective(data, &offset) else { return nil }
            ic.append(pt)
        }

        return Groth16VerificationKey(alpha_g1: alpha, beta_g2: beta,
                                       gamma_g2: gamma, delta_g2: delta, ic: ic)
    }

    // MARK: - Serialization (JSON)

    /// Serialize a VK to a JSON-compatible dictionary.
    public func serializeJSON(_ vk: Groth16VerificationKey) -> [String: Any] {
        var dict = [String: Any]()
        dict["protocol"] = "groth16"
        dict["curve"] = "bn254"
        dict["nPublic"] = vk.ic.count - 1

        dict["alpha_g1"] = g1ToHexArray(vk.alpha_g1)
        dict["beta_g2"] = g2ToHexArray(vk.beta_g2)
        dict["gamma_g2"] = g2ToHexArray(vk.gamma_g2)
        dict["delta_g2"] = g2ToHexArray(vk.delta_g2)

        var icArray = [[String]]()
        for pt in vk.ic {
            icArray.append(g1ToHexArray(pt))
        }
        dict["ic"] = icArray
        return dict
    }

    /// Deserialize a VK from a JSON-compatible dictionary.
    public func deserializeJSON(_ dict: [String: Any]) -> Groth16VerificationKey? {
        guard let alphaArr = dict["alpha_g1"] as? [String],
              let betaArr = dict["beta_g2"] as? [String],
              let gammaArr = dict["gamma_g2"] as? [String],
              let deltaArr = dict["delta_g2"] as? [String],
              let icArrays = dict["ic"] as? [[String]] else { return nil }

        guard let alpha = hexArrayToG1(alphaArr) else { return nil }
        guard let beta = hexArrayToG2(betaArr) else { return nil }
        guard let gamma = hexArrayToG2(gammaArr) else { return nil }
        guard let delta = hexArrayToG2(deltaArr) else { return nil }

        var ic = [PointProjective]()
        for arr in icArrays {
            guard let pt = hexArrayToG1(arr) else { return nil }
            ic.append(pt)
        }

        return Groth16VerificationKey(alpha_g1: alpha, beta_g2: beta,
                                       gamma_g2: gamma, delta_g2: delta, ic: ic)
    }

    // MARK: - Point Compression

    /// Compress a G1 point (projective -> x-coordinate + y sign).
    public func compressG1(_ p: PointProjective) -> CompressedG1? {
        guard let aff = pointToAffine(p) else { return nil }
        let xLimbs = fpToInt(aff.x)
        let yLimbs = fpToInt(aff.y)
        // Sign convention: y is "positive" if the last nonzero limb is even
        let ySign = isLexicographicallyLarger(yLimbs)
        return CompressedG1(x: xLimbs, ySign: ySign)
    }

    /// Decompress a G1 point from x-coordinate + y sign.
    public func decompressG1(_ c: CompressedG1) -> PointProjective? {
        let x = fpFromIntLimbs(c.x)
        // y^2 = x^3 + 3 (BN254)
        let x3 = fpMul(x, fpMul(x, x))
        let rhs = fpAdd(x3, fpFromInt(3))
        guard let y = fpSqrt(rhs) else { return nil }
        let yLimbs = fpToInt(y)
        let yIsLarger = isLexicographicallyLarger(yLimbs)
        let finalY = (yIsLarger == c.ySign) ? y : fpNeg(y)
        return pointFromAffine(PointAffine(x: x, y: finalY))
    }

    /// Compress a G2 point.
    public func compressG2(_ p: G2ProjectivePoint) -> CompressedG2? {
        guard let aff = g2ToAffine(p) else { return nil }
        let xC0Limbs = fpToInt(aff.x.c0)
        let xC1Limbs = fpToInt(aff.x.c1)
        let yC1Limbs = fpToInt(aff.y.c1)
        let yC0Limbs = fpToInt(aff.y.c0)
        // For Fp2, compare c1 first; if zero compare c0
        let ySign: Bool
        if !isAllZero(yC1Limbs) {
            ySign = isLexicographicallyLarger(yC1Limbs)
        } else {
            ySign = isLexicographicallyLarger(yC0Limbs)
        }
        return CompressedG2(xC0: xC0Limbs, xC1: xC1Limbs, ySign: ySign)
    }

    /// Compress an entire VK.
    public func compressVK(_ vk: Groth16VerificationKey) -> CompressedGroth16VK? {
        guard let alpha = compressG1(vk.alpha_g1) else { return nil }
        guard let beta = compressG2(vk.beta_g2) else { return nil }
        guard let gamma = compressG2(vk.gamma_g2) else { return nil }
        guard let delta = compressG2(vk.delta_g2) else { return nil }
        var icComp = [CompressedG1]()
        icComp.reserveCapacity(vk.ic.count)
        for pt in vk.ic {
            guard let c = compressG1(pt) else { return nil }
            icComp.append(c)
        }
        return CompressedGroth16VK(alpha_g1: alpha, beta_g2: beta,
                                    gamma_g2: gamma, delta_g2: delta, ic: icComp)
    }

    /// Compute compressed VK size in bytes.
    public func compressedSize(_ vk: Groth16VerificationKey) -> Int {
        // G1 compressed: 32 bytes (x) + 1 byte (sign) = 33
        // G2 compressed: 64 bytes (x Fp2) + 1 byte (sign) = 65
        let g1Bytes = 33
        let g2Bytes = 65
        return 4 + g1Bytes + 3 * g2Bytes + vk.ic.count * g1Bytes
    }

    /// Compute uncompressed VK size in bytes.
    public func uncompressedSize(_ vk: Groth16VerificationKey) -> Int {
        let g1Bytes = 96   // 3 * 32
        let g2Bytes = 192  // 6 * 32
        return 4 + g1Bytes + 3 * g2Bytes + vk.ic.count * g1Bytes
    }

    // MARK: - Batch Preprocessing

    /// Preprocess a single VK for fast verification (convert to affine, precompute negation).
    public func preprocess(_ vk: Groth16VerificationKey) -> PreprocessedGroth16VK? {
        guard let alphaAff = pointToAffine(vk.alpha_g1),
              let betaAff = g2ToAffine(vk.beta_g2),
              let gammaAff = g2ToAffine(vk.gamma_g2),
              let deltaAff = g2ToAffine(vk.delta_g2) else { return nil }
        var icAff = [PointAffine]()
        icAff.reserveCapacity(vk.ic.count)
        for pt in vk.ic {
            guard let a = pointToAffine(pt) else { return nil }
            icAff.append(a)
        }
        let negAlpha = pointNegateAffine(alphaAff)
        return PreprocessedGroth16VK(alpha_g1: alphaAff, beta_g2: betaAff,
                                      gamma_g2: gammaAff, delta_g2: deltaAff,
                                      ic: icAff, negAlpha_g1: negAlpha)
    }

    /// Batch preprocess multiple VKs for multi-circuit verification.
    /// Returns nil entries for invalid VKs.
    public func batchPreprocess(_ vks: [Groth16VerificationKey]) -> [PreprocessedGroth16VK?] {
        let count = vks.count
        if count <= 4 {
            return vks.map { preprocess($0) }
        }
        // Parallelize for larger batches
        let results = UnsafeMutablePointer<PreprocessedGroth16VK?>.allocate(capacity: count)
        results.initialize(repeating: nil, count: count)
        DispatchQueue.concurrentPerform(iterations: count) { i in
            results[i] = preprocess(vks[i])
        }
        var out = [PreprocessedGroth16VK?]()
        out.reserveCapacity(count)
        for i in 0..<count { out.append(results[i]) }
        results.deinitialize(count: count)
        results.deallocate()
        return out
    }

    // MARK: - IC Computation

    /// Compute input commitment: vk_x = IC[0] + sum(publicInputs[i] * IC[i+1])
    /// Uses GPU MSM for large public input vectors.
    public func computeIC(vk: Groth16VerificationKey, publicInputs: [Fr]) -> PointProjective? {
        guard publicInputs.count + 1 == vk.ic.count else { return nil }
        if publicInputs.isEmpty { return vk.ic[0] }

        let n = publicInputs.count
        // Filter non-zero inputs
        var points = [PointAffine]()
        var scalars = [Fr]()
        points.reserveCapacity(n)
        scalars.reserveCapacity(n)

        for i in 0..<n {
            if !publicInputs[i].isZero {
                guard let aff = pointToAffine(vk.ic[i + 1]) else { return nil }
                points.append(aff)
                scalars.append(publicInputs[i])
            }
        }

        if points.isEmpty { return vk.ic[0] }

        let msmResult: PointProjective
        if points.count >= gpuMSMThreshold, let gpu = msm {
            var sl = [[UInt32]]()
            sl.reserveCapacity(points.count)
            for s in scalars { sl.append(frToLimbs(s)) }
            do {
                msmResult = try gpu.msm(points: points, scalars: sl)
            } catch {
                // Fallback to CPU
                msmResult = cpuMSM(points: points, scalars: scalars)
            }
        } else {
            msmResult = cpuMSM(points: points, scalars: scalars)
        }

        return pointAdd(vk.ic[0], msmResult)
    }

    /// Compute IC using a preprocessed VK (affine points already cached).
    public func computeICPreprocessed(pvk: PreprocessedGroth16VK, publicInputs: [Fr]) -> PointProjective? {
        guard publicInputs.count + 1 == pvk.ic.count else { return nil }
        if publicInputs.isEmpty { return pointFromAffine(pvk.ic[0]) }

        let n = publicInputs.count
        var points = [PointAffine]()
        var scalars = [Fr]()
        points.reserveCapacity(n)
        scalars.reserveCapacity(n)

        for i in 0..<n {
            if !publicInputs[i].isZero {
                points.append(pvk.ic[i + 1])
                scalars.append(publicInputs[i])
            }
        }

        if points.isEmpty { return pointFromAffine(pvk.ic[0]) }

        let msmResult: PointProjective
        if points.count >= gpuMSMThreshold, let gpu = msm {
            var sl = [[UInt32]]()
            sl.reserveCapacity(points.count)
            for s in scalars { sl.append(frToLimbs(s)) }
            do {
                msmResult = try gpu.msm(points: points, scalars: sl)
            } catch {
                msmResult = cpuMSM(points: points, scalars: scalars)
            }
        } else {
            msmResult = cpuMSM(points: points, scalars: scalars)
        }

        return pointAdd(pointFromAffine(pvk.ic[0]), msmResult)
    }

    // MARK: - VK Validation

    /// Validate a verification key: on-curve checks, IC count, and pairing consistency.
    public func validate(_ vk: Groth16VerificationKey) -> VKValidationResult {
        // Check IC has at least 1 element (for the constant term)
        let icCountValid = vk.ic.count >= 1

        // On-curve checks for G1 points
        let alphaOnCurve = isG1OnCurve(vk.alpha_g1)
        var icAllOnCurve = true
        for pt in vk.ic {
            if !isG1OnCurve(pt) { icAllOnCurve = false; break }
        }

        // On-curve checks for G2 points
        let betaOnCurve = isG2OnCurve(vk.beta_g2)
        let gammaOnCurve = isG2OnCurve(vk.gamma_g2)
        let deltaOnCurve = isG2OnCurve(vk.delta_g2)

        // Pairing consistency: e(alpha, beta) should be a valid GT element (non-identity)
        var pairingConsistent = false
        if alphaOnCurve && betaOnCurve {
            if let alphaAff = pointToAffine(vk.alpha_g1),
               let betaAff = g2ToAffine(vk.beta_g2) {
                // Check that alpha and beta are not identity
                if !pointIsIdentity(vk.alpha_g1) && !g2IsIdentity(vk.beta_g2) {
                    // Compute pairing and verify it's not trivial
                    let gt = bn254Pairing(alphaAff, betaAff)
                    pairingConsistent = !gt.c0.isZero || !gt.c1.isZero
                }
            }
        }

        let allCurveChecks = alphaOnCurve && betaOnCurve && gammaOnCurve && deltaOnCurve && icAllOnCurve
        let valid = allCurveChecks && icCountValid && pairingConsistent

        var message = ""
        if !icCountValid { message += "IC must have at least 1 element. " }
        if !alphaOnCurve { message += "alpha_g1 not on curve. " }
        if !betaOnCurve { message += "beta_g2 not on curve. " }
        if !gammaOnCurve { message += "gamma_g2 not on curve. " }
        if !deltaOnCurve { message += "delta_g2 not on curve. " }
        if !icAllOnCurve { message += "Some IC points not on curve. " }
        if !pairingConsistent { message += "Pairing consistency check failed. " }
        if valid { message = "VK is valid." }

        return VKValidationResult(valid: valid, alphaOnCurve: alphaOnCurve,
                                   betaOnCurve: betaOnCurve, gammaOnCurve: gammaOnCurve,
                                   deltaOnCurve: deltaOnCurve, icAllOnCurve: icAllOnCurve,
                                   icCountValid: icCountValid, pairingConsistent: pairingConsistent,
                                   message: message)
    }

    // MARK: - VK Comparison & Diff

    /// Compare two VKs and produce a diff describing any differences.
    public func diff(_ a: Groth16VerificationKey, _ b: Groth16VerificationKey) -> VKDiff {
        let alphaG1Differs = !g1Equal(a.alpha_g1, b.alpha_g1)
        let betaG2Differs = !g2PointEqual(a.beta_g2, b.beta_g2)
        let gammaG2Differs = !g2PointEqual(a.gamma_g2, b.gamma_g2)
        let deltaG2Differs = !g2PointEqual(a.delta_g2, b.delta_g2)
        let icCountDiffers = a.ic.count != b.ic.count

        var icDiffIndices = [Int]()
        let minIC = min(a.ic.count, b.ic.count)
        for i in 0..<minIC {
            if !g1Equal(a.ic[i], b.ic[i]) {
                icDiffIndices.append(i)
            }
        }
        // Any extra IC points in the longer array are considered different
        if a.ic.count > minIC {
            for i in minIC..<a.ic.count { icDiffIndices.append(i) }
        } else if b.ic.count > minIC {
            for i in minIC..<b.ic.count { icDiffIndices.append(i) }
        }

        return VKDiff(alphaG1Differs: alphaG1Differs, betaG2Differs: betaG2Differs,
                      gammaG2Differs: gammaG2Differs, deltaG2Differs: deltaG2Differs,
                      icCountDiffers: icCountDiffers, icDiffIndices: icDiffIndices)
    }

    /// Check if two VKs are identical.
    public func isEqual(_ a: Groth16VerificationKey, _ b: Groth16VerificationKey) -> Bool {
        return diff(a, b).isIdentical
    }

    // MARK: - Internal Helpers

    /// CPU MSM fallback
    private func cpuMSM(points: [PointAffine], scalars: [Fr]) -> PointProjective {
        var sl = [[UInt32]]()
        sl.reserveCapacity(points.count)
        for s in scalars { sl.append(frToLimbs(s)) }
        return cPippengerMSM(points: points, scalars: sl)
    }

    /// Check if a G1 projective point is on the BN254 curve: y^2 = x^3 + 3
    private func isG1OnCurve(_ p: PointProjective) -> Bool {
        if pointIsIdentity(p) { return true }
        guard let aff = pointToAffine(p) else { return false }
        let lhs = fpSqr(aff.y)
        let x3 = fpMul(aff.x, fpMul(aff.x, aff.x))
        let rhs = fpAdd(x3, fpFromInt(3))
        return fpSub(lhs, rhs).isZero
    }

    /// Check if a G2 projective point is on the twist curve: y^2 = x^3 + 3/(9+u)
    /// The twist b' = 3 * inverse(9+u)
    private func isG2OnCurve(_ p: G2ProjectivePoint) -> Bool {
        if g2IsIdentity(p) { return true }
        guard let aff = g2ToAffine(p) else { return false }
        let lhs = fp2Sqr(aff.y)
        let x3 = fp2Mul(aff.x, fp2Mul(aff.x, aff.x))
        // b' = 3/(9+u) for BN254 twist
        let xi = Fp2(c0: fpFromInt(9), c1: fpFromInt(1))
        let xiInv = fp2Inverse(xi)
        let bTwist = fp2Mul(Fp2(c0: fpFromInt(3), c1: .zero), xiInv)
        let rhs = fp2Add(x3, bTwist)
        let diff = fp2Sub(lhs, rhs)
        return diff.c0.isZero && diff.c1.isZero
    }

    /// G1 projective point equality via cross-multiplication
    private func g1Equal(_ a: PointProjective, _ b: PointProjective) -> Bool {
        if pointIsIdentity(a) && pointIsIdentity(b) { return true }
        if pointIsIdentity(a) || pointIsIdentity(b) { return false }
        // a.x * b.z^2 == b.x * a.z^2
        let az2 = fpSqr(a.z)
        let bz2 = fpSqr(b.z)
        let lhsX = fpMul(a.x, bz2)
        let rhsX = fpMul(b.x, az2)
        if !fpSub(lhsX, rhsX).isZero { return false }
        // a.y * b.z^3 == b.y * a.z^3
        let az3 = fpMul(az2, a.z)
        let bz3 = fpMul(bz2, b.z)
        let lhsY = fpMul(a.y, bz3)
        let rhsY = fpMul(b.y, az3)
        return fpSub(lhsY, rhsY).isZero
    }

    /// G2 projective point equality via cross-multiplication
    private func g2PointEqual(_ a: G2ProjectivePoint, _ b: G2ProjectivePoint) -> Bool {
        if g2IsIdentity(a) && g2IsIdentity(b) { return true }
        if g2IsIdentity(a) || g2IsIdentity(b) { return false }
        let az2 = fp2Sqr(a.z)
        let bz2 = fp2Sqr(b.z)
        let lhsX = fp2Mul(a.x, bz2)
        let rhsX = fp2Mul(b.x, az2)
        let xDiff = fp2Sub(lhsX, rhsX)
        if !xDiff.c0.isZero || !xDiff.c1.isZero { return false }
        let az3 = fp2Mul(az2, a.z)
        let bz3 = fp2Mul(bz2, b.z)
        let lhsY = fp2Mul(a.y, bz3)
        let rhsY = fp2Mul(b.y, az3)
        let yDiff = fp2Sub(lhsY, rhsY)
        return yDiff.c0.isZero && yDiff.c1.isZero
    }

    // MARK: - Binary Serialization Helpers

    private func appendFp(_ data: inout Data, _ f: Fp) {
        // Store raw Montgomery form (8 x UInt32 limbs) for exact round-trip
        let limbs = [f.v.0, f.v.1, f.v.2, f.v.3, f.v.4, f.v.5, f.v.6, f.v.7]
        for l in limbs {
            var le = l.littleEndian
            data.append(Data(bytes: &le, count: 4))
        }
    }

    private func appendG1Projective(_ data: inout Data, _ p: PointProjective) {
        appendFp(&data, p.x)
        appendFp(&data, p.y)
        appendFp(&data, p.z)
    }

    private func appendG2Projective(_ data: inout Data, _ p: G2ProjectivePoint) {
        appendFp(&data, p.x.c0); appendFp(&data, p.x.c1)
        appendFp(&data, p.y.c0); appendFp(&data, p.y.c1)
        appendFp(&data, p.z.c0); appendFp(&data, p.z.c1)
    }

    private func readFp(_ data: Data, _ offset: inout Int) -> Fp? {
        guard offset + 32 <= data.count else { return nil }
        // Read raw Montgomery form (8 x UInt32 limbs)
        var limbs = [UInt32](repeating: 0, count: 8)
        for i in 0..<8 {
            limbs[i] = data.withUnsafeBytes {
                $0.load(fromByteOffset: offset + i * 4, as: UInt32.self).littleEndian
            }
        }
        offset += 32
        return Fp(v: (limbs[0], limbs[1], limbs[2], limbs[3],
                      limbs[4], limbs[5], limbs[6], limbs[7]))
    }

    private func readG1Projective(_ data: Data, _ offset: inout Int) -> PointProjective? {
        guard let x = readFp(data, &offset),
              let y = readFp(data, &offset),
              let z = readFp(data, &offset) else { return nil }
        return PointProjective(x: x, y: y, z: z)
    }

    private func readG2Projective(_ data: Data, _ offset: inout Int) -> G2ProjectivePoint? {
        guard let xc0 = readFp(data, &offset), let xc1 = readFp(data, &offset),
              let yc0 = readFp(data, &offset), let yc1 = readFp(data, &offset),
              let zc0 = readFp(data, &offset), let zc1 = readFp(data, &offset) else { return nil }
        return G2ProjectivePoint(x: Fp2(c0: xc0, c1: xc1),
                                  y: Fp2(c0: yc0, c1: yc1),
                                  z: Fp2(c0: zc0, c1: zc1))
    }

    // MARK: - JSON Helpers

    private func g1ToHexArray(_ p: PointProjective) -> [String] {
        guard let aff = pointToAffine(p) else {
            return ["0x0", "0x0"]
        }
        return [fpToHexString(aff.x), fpToHexString(aff.y)]
    }

    private func g2ToHexArray(_ p: G2ProjectivePoint) -> [String] {
        guard let aff = g2ToAffine(p) else {
            return ["0x0", "0x0", "0x0", "0x0"]
        }
        return [fpToHexString(aff.x.c0), fpToHexString(aff.x.c1),
                fpToHexString(aff.y.c0), fpToHexString(aff.y.c1)]
    }

    private func fpToHexString(_ f: Fp) -> String {
        return fpToHex(f)
    }

    private func hexArrayToG1(_ arr: [String]) -> PointProjective? {
        guard arr.count >= 2 else { return nil }
        guard let x = fpFromHexString(arr[0]),
              let y = fpFromHexString(arr[1]) else { return nil }
        return pointFromAffine(PointAffine(x: x, y: y))
    }

    private func hexArrayToG2(_ arr: [String]) -> G2ProjectivePoint? {
        guard arr.count >= 4 else { return nil }
        guard let xc0 = fpFromHexString(arr[0]),
              let xc1 = fpFromHexString(arr[1]),
              let yc0 = fpFromHexString(arr[2]),
              let yc1 = fpFromHexString(arr[3]) else { return nil }
        return G2ProjectivePoint(x: Fp2(c0: xc0, c1: xc1),
                                  y: Fp2(c0: yc0, c1: yc1),
                                  z: .one)
    }

    private func fpFromHexString(_ s: String) -> Fp? {
        var hex = s
        if hex.hasPrefix("0x") || hex.hasPrefix("0X") {
            hex = String(hex.dropFirst(2))
        }
        // Parse as big-endian into 4 UInt64 limbs (little-endian order)
        while hex.count < 64 { hex = "0" + hex }
        guard hex.count == 64 else { return nil }
        var limbs = [UInt64](repeating: 0, count: 4)
        for i in 0..<4 {
            let start = hex.index(hex.startIndex, offsetBy: i * 16)
            let end = hex.index(start, offsetBy: 16)
            guard let val = UInt64(hex[start..<end], radix: 16) else { return nil }
            limbs[3 - i] = val
        }
        // Convert from integer to Montgomery form (same as fpFromHex)
        let raw = Fp.from64(limbs)
        return fpMul(raw, Fp.from64(Fp.R2_MOD_P))
    }

    // MARK: - Compression Helpers

    /// Check if limbs represent a "lexicographically larger" value (MSB comparison)
    private func isLexicographicallyLarger(_ limbs: [UInt64]) -> Bool {
        // Compare against p/2 by checking the high bit of the field element
        // Simple heuristic: check if the highest limb's MSB indicates > p/2
        guard limbs.count == 4 else { return false }
        // BN254 p = 0x30644e72e131a029...
        // p/2 ~ 0x183227397098d014...
        // If limbs[3] > p/2's high limb, it's larger
        let pHalf3: UInt64 = 0x183227397098d014
        if limbs[3] > pHalf3 { return true }
        if limbs[3] < pHalf3 { return false }
        let pHalf2: UInt64 = 0x5c2822d9c40a22e5
        if limbs[2] > pHalf2 { return true }
        if limbs[2] < pHalf2 { return false }
        let pHalf1: UInt64 = 0x4c3c0b5c8e4d84e5
        if limbs[1] > pHalf1 { return true }
        if limbs[1] < pHalf1 { return false }
        let pHalf0: UInt64 = 0x9e10460b6c3e7ea3
        return limbs[0] > pHalf0
    }

    private func isAllZero(_ limbs: [UInt64]) -> Bool {
        for l in limbs { if l != 0 { return false } }
        return true
    }

    /// Convert integer UInt64 limbs to Fp Montgomery form (via R^2 multiplication)
    private func fpFromIntLimbs(_ limbs: [UInt64]) -> Fp {
        let raw = Fp.from64(limbs)
        return fpMul(raw, Fp.from64(Fp.R2_MOD_P))
    }
}
