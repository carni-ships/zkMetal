// Committed CCS (CCCS) and Linearized CCCS (LCCCS) for HyperNova folding
//
// CCCS: CCS instance with Pedersen-committed witness
// LCCCS: Relaxed/linearized form that can absorb folded results
//
// The key insight: folding transforms checking a degree-d constraint into
// checking a degree-1 (linear) constraint, deferring the actual proof.

import Foundation
import NeonFieldOps

// MARK: - Multilinear Extension

/// Evaluate the multilinear extension of a vector at point r.
/// Given f: {0,1}^s -> Fr with evaluations evals[0..2^s-1],
/// compute f(r_0, ..., r_{s-1}) via successive linear interpolation.
public func multilinearEval(evals: [Fr], point: [Fr]) -> Fr {
    let s = point.count
    precondition(evals.count == (1 << s), "evals length must be 2^s")

    var current = evals
    for i in 0..<s {
        let half = current.count / 2
        let ri = point[i]
        current.withUnsafeMutableBytes { cBuf in
            withUnsafeBytes(of: ri) { rBuf in
                bn254_fr_fold_interleaved_inplace(
                    cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(half))
            }
        }
        current.removeLast(half)
    }
    return current[0]
}

/// Compute the multilinear extension evaluations of a sparse matrix row
/// when the column variable is bound to z.
/// Returns M*z as a vector (the MLE over the row variable).
/// For MLE over both row and column, we evaluate at a point.
public func matrixMleEval(matrix: SparseMatrix, z: [Fr], rowPoint: [Fr]) -> Fr {
    // M(r, y) = sum_{i,j} M[i][j] * eq(r, i) * y[j]
    // Since we want sum_y M(r, y) * z(y), this equals:
    // sum_i eq(r, i) * (M * z)[i]
    let mv = matrix.mulVec(z)
    return multilinearEval(evals: padToPow2(mv), point: rowPoint)
}

/// Pad a vector to the next power of 2 with zeros.
public func padToPow2(_ v: [Fr]) -> [Fr] {
    let n = v.count
    if n == 0 { return [Fr.zero] }
    var logN = 0
    while (1 << logN) < n { logN += 1 }
    let padded = (1 << logN)
    if padded == n { return v }
    var result = [Fr](repeating: .zero, count: padded)
    result.withUnsafeMutableBytes { rBuf in
        v.withUnsafeBytes { vBuf in
            memcpy(rBuf.baseAddress!, vBuf.baseAddress!, n * MemoryLayout<Fr>.stride)
        }
    }
    return result
}

/// Compute eq(r, x) for all x in {0,1}^s.
/// eq(r, x) = prod_{i=0}^{s-1} (r_i * x_i + (1 - r_i) * (1 - x_i))
public func eqEvals(point: [Fr]) -> [Fr] {
    let s = point.count
    let n = 1 << s
    var evals = [Fr](repeating: Fr.zero, count: n)
    evals[0] = Fr.one

    for i in 0..<s {
        let ri = point[i]
        let oneMinusR = frSub(Fr.one, ri)
        let half = 1 << i
        // Process in reverse to avoid overwriting
        for j in stride(from: half - 1, through: 0, by: -1) {
            evals[2 * j + 1] = frMul(evals[j], ri)
            evals[2 * j] = frMul(evals[j], oneMinusR)
        }
    }
    return evals
}

// MARK: - Committed CCS (CCCS)

/// A CCS instance whose witness is committed via Pedersen commitment (MSM).
/// The prover knows the full witness z; the verifier only sees the commitment C.
public struct CCCS {
    public let commitment: PointProjective  // Pedersen commitment to witness portion
    public let publicInput: [Fr]            // Public input x (portion of z)
    public let ccsRef: Int                  // Index/tag identifying the CCS structure
    public let cachedAffineX: Fr?           // Cached affine x for transcript
    public let cachedAffineY: Fr?           // Cached affine y for transcript

    public init(commitment: PointProjective, publicInput: [Fr], ccsRef: Int = 0) {
        self.commitment = commitment
        self.publicInput = publicInput
        self.ccsRef = ccsRef
        self.cachedAffineX = nil
        self.cachedAffineY = nil
    }

    /// Init with pre-computed affine coordinates.
    public init(commitment: PointProjective, publicInput: [Fr], ccsRef: Int = 0,
                affineX: Fr, affineY: Fr) {
        self.commitment = commitment
        self.publicInput = publicInput
        self.ccsRef = ccsRef
        self.cachedAffineX = affineX
        self.cachedAffineY = affineY
    }
}

// MARK: - Linearized CCCS (LCCCS)

/// A relaxed/linearized CCS instance. The "running instance" in folding.
///
/// An LCCCS satisfies if there exists witness z such that:
///   1. C = Commit(w) where w is the witness portion of z
///   2. For all i in {1,...,t}: v_i = MLE(M_i * z)(r)
///   3. sum_j c_j * prod_{i in S_j} v_i = u * 0  (the linearized check)
///
/// The relaxation factor u allows folding: after folding, u != 1.
public struct LCCCS {
    public let commitment: PointProjective  // Pedersen commitment
    public let publicInput: [Fr]            // Public input
    public let u: Fr                        // Relaxation factor (initially 1)
    public let r: [Fr]                      // Random evaluation point (log(m)-dimensional)
    public let v: [Fr]                      // Claimed evaluations: v_i = MLE(M_i * z)(r)
    public let ccsRef: Int                  // Reference to CCS structure
    // Cached affine coordinates of commitment (avoids repeated projective-to-affine in transcript)
    public let cachedAffineX: Fr?
    public let cachedAffineY: Fr?

    public init(commitment: PointProjective, publicInput: [Fr],
                u: Fr, r: [Fr], v: [Fr], ccsRef: Int = 0) {
        self.commitment = commitment
        self.publicInput = publicInput
        self.u = u
        self.r = r
        self.v = v
        self.ccsRef = ccsRef
        self.cachedAffineX = nil
        self.cachedAffineY = nil
    }

    /// Init with pre-computed affine coords (avoids repeated projective-to-affine).
    public init(commitment: PointProjective, publicInput: [Fr],
                u: Fr, r: [Fr], v: [Fr], ccsRef: Int = 0,
                affineX: Fr, affineY: Fr) {
        self.commitment = commitment
        self.publicInput = publicInput
        self.u = u
        self.r = r
        self.v = v
        self.ccsRef = ccsRef
        self.cachedAffineX = affineX
        self.cachedAffineY = affineY
    }
}

// MARK: - Pedersen Commitment

/// Pedersen commitment: C = sum_i w_i * G_i where G_i are generator points.
/// Uses MSM for efficient computation.
public struct PedersenParams {
    public let generators: [PointAffine]    // G_1, ..., G_n (SRS)
    public let blinding: PointAffine        // H (blinding generator)

    /// Generate deterministic SRS from a seed point by repeated hashing/doubling.
    public static func generate(size: Int) -> PedersenParams {
        let gx = fpFromInt(1)
        let gy = fpFromInt(2)
        let g = pointFromAffine(PointAffine(x: gx, y: gy))

        var projPoints = [PointProjective]()
        projPoints.reserveCapacity(size + 1)
        var acc = g
        for _ in 0..<(size + 1) {
            projPoints.append(acc)
            acc = pointDouble(pointAdd(acc, g))
        }
        let affinePoints = batchToAffine(projPoints)
        return PedersenParams(
            generators: Array(affinePoints.prefix(size)),
            blinding: affinePoints[size]
        )
    }

    /// Commit to witness vector w: C = sum_i w_i * G_i
    /// Uses C CIOS field arithmetic for fast scalar multiplication.
    public func commit(witness: [Fr]) -> PointProjective {
        precondition(witness.count <= generators.count,
                     "Witness too large for SRS: \(witness.count) > \(generators.count)")
        let n = witness.count
        if n <= 16 {
            // For small n, direct C scalar-mul accumulation (no Pippenger overhead)
            var result = pointIdentity()
            for i in 0..<n {
                let p = pointFromAffine(generators[i])
                let sp = cPointScalarMul(p, witness[i])
                result = pointAdd(result, sp)
            }
            return result
        }
        let pts = Array(generators.prefix(n))
        let scalars = witness.map { scalar -> [UInt32] in
            frToLimbs(scalar)
        }
        return cPippengerMSM(points: pts, scalars: scalars)
    }

    /// Commit using GPU MSM engine for larger witnesses.
    public func commitGPU(witness: [Fr], engine: MetalMSM) throws -> PointProjective {
        let n = witness.count
        let pts = Array(generators.prefix(n))
        let scalars = witness.map { scalar -> [UInt32] in
            let limbs = scalar.to64()
            return [
                UInt32(limbs[0] & 0xFFFFFFFF), UInt32(limbs[0] >> 32),
                UInt32(limbs[1] & 0xFFFFFFFF), UInt32(limbs[1] >> 32),
                UInt32(limbs[2] & 0xFFFFFFFF), UInt32(limbs[2] >> 32),
                UInt32(limbs[3] & 0xFFFFFFFF), UInt32(limbs[3] >> 32),
            ]
        }
        return try engine.msm(points: pts, scalars: scalars)
    }
}

// MARK: - CPU MSM (small, for commitment)

/// Simple Pippenger CPU MSM for small inputs.
func cpuMSM(points: [PointAffine], scalars: [[UInt32]]) -> PointProjective {
    var result = pointIdentity()
    for i in 0..<points.count {
        let p = pointFromAffine(points[i])
        let s = Fr(v: (scalars[i][0], scalars[i][1], scalars[i][2], scalars[i][3],
                       scalars[i][4], scalars[i][5], scalars[i][6], scalars[i][7]))
        result = pointAdd(result, pointScalarMul(p, s))
    }
    return result
}
