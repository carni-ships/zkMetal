// Groth16 SNARK types and R1CS representation for BN254
import Foundation

// MARK: - R1CS

public struct R1CSEntry {
    public let row: Int
    public let col: Int
    public let val: Fr
    public init(row: Int, col: Int, val: Fr) { self.row = row; self.col = col; self.val = val }
}

public struct R1CSInstance {
    public let numConstraints: Int
    public let numVars: Int
    public let numPublic: Int  // number of public inputs (not counting the 1)
    public let aEntries: [R1CSEntry]
    public let bEntries: [R1CSEntry]
    public let cEntries: [R1CSEntry]

    public init(numConstraints: Int, numVars: Int, numPublic: Int,
                aEntries: [R1CSEntry], bEntries: [R1CSEntry], cEntries: [R1CSEntry]) {
        self.numConstraints = numConstraints
        self.numVars = numVars
        self.numPublic = numPublic
        self.aEntries = aEntries
        self.bEntries = bEntries
        self.cEntries = cEntries
    }

    public func sparseMatVec(_ entries: [R1CSEntry], _ z: [Fr]) -> [Fr] {
        let m = numConstraints
        var result = [Fr](repeating: .zero, count: m)
        let n = entries.count
        if n < 8192 {
            for e in entries {
                result[e.row] = frAdd(result[e.row], frMul(e.val, z[e.col]))
            }
            return result
        }
        // Parallel: each thread accumulates into its own result array, then merge
        let nThreads = min(8, ProcessInfo.processInfo.activeProcessorCount)
        let chunkSize = (n + nThreads - 1) / nThreads
        var partials = [[Fr]](repeating: [Fr](repeating: .zero, count: m), count: nThreads)
        DispatchQueue.concurrentPerform(iterations: nThreads) { t in
            let start = t * chunkSize
            let end = min(start + chunkSize, n)
            for idx in start..<end {
                let e = entries[idx]
                partials[t][e.row] = frAdd(partials[t][e.row], frMul(e.val, z[e.col]))
            }
        }
        // Merge thread-local results
        for t in 0..<nThreads {
            for i in 0..<m {
                result[i] = frAdd(result[i], partials[t][i])
            }
        }
        return result
    }

    public func isSatisfied(z: [Fr]) -> Bool {
        let az = sparseMatVec(aEntries, z)
        let bz = sparseMatVec(bEntries, z)
        let cz = sparseMatVec(cEntries, z)
        for i in 0..<numConstraints {
            if !frEq(frMul(az[i], bz[i]), cz[i]) { return false }
        }
        return true
    }
}

// MARK: - Groth16 Keys and Proof

public struct Groth16ProvingKey {
    public let alpha_g1: PointProjective
    public let beta_g1: PointProjective
    public let beta_g2: G2ProjectivePoint
    public let delta_g1: PointProjective
    public let delta_g2: G2ProjectivePoint
    public let ic: [PointProjective]       // public input verification points
    public let a_query: [PointProjective]
    public let b_g1_query: [PointProjective]
    public let b_g2_query: [G2ProjectivePoint]
    public let h_query: [PointProjective]
    public let l_query: [PointProjective]  // witness portion
    // Cached affine conversions (computed once at setup time, reused across proofs)
    public let a_query_affine: [PointAffine]
    public let b_g1_query_affine: [PointAffine]
    public let h_query_affine: [PointAffine]
    public let l_query_affine: [PointAffine]

    public init(alpha_g1: PointProjective, beta_g1: PointProjective, beta_g2: G2ProjectivePoint,
                delta_g1: PointProjective, delta_g2: G2ProjectivePoint,
                ic: [PointProjective], a_query: [PointProjective], b_g1_query: [PointProjective],
                b_g2_query: [G2ProjectivePoint], h_query: [PointProjective], l_query: [PointProjective],
                a_query_affine: [PointAffine], b_g1_query_affine: [PointAffine],
                h_query_affine: [PointAffine], l_query_affine: [PointAffine]) {
        self.alpha_g1 = alpha_g1; self.beta_g1 = beta_g1; self.beta_g2 = beta_g2
        self.delta_g1 = delta_g1; self.delta_g2 = delta_g2
        self.ic = ic; self.a_query = a_query; self.b_g1_query = b_g1_query
        self.b_g2_query = b_g2_query; self.h_query = h_query; self.l_query = l_query
        self.a_query_affine = a_query_affine; self.b_g1_query_affine = b_g1_query_affine
        self.h_query_affine = h_query_affine; self.l_query_affine = l_query_affine
    }
}

public struct Groth16VerificationKey {
    public let alpha_g1: PointProjective
    public let beta_g2: G2ProjectivePoint
    public let gamma_g2: G2ProjectivePoint
    public let delta_g2: G2ProjectivePoint
    public let ic: [PointProjective]

    public init(alpha_g1: PointProjective, beta_g2: G2ProjectivePoint,
                gamma_g2: G2ProjectivePoint, delta_g2: G2ProjectivePoint,
                ic: [PointProjective]) {
        self.alpha_g1 = alpha_g1; self.beta_g2 = beta_g2
        self.gamma_g2 = gamma_g2; self.delta_g2 = delta_g2; self.ic = ic
    }
}

public struct Groth16Proof {
    public let a: PointProjective   // G1
    public let b: G2ProjectivePoint // G2
    public let c: PointProjective   // G1

    public init(a: PointProjective, b: G2ProjectivePoint, c: PointProjective) {
        self.a = a; self.b = b; self.c = c
    }
}
