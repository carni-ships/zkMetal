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
        var result = [Fr](repeating: .zero, count: numConstraints)
        for e in entries {
            result[e.row] = frAdd(result[e.row], frMul(e.val, z[e.col]))
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
}

public struct Groth16VerificationKey {
    public let alpha_g1: PointProjective
    public let beta_g2: G2ProjectivePoint
    public let gamma_g2: G2ProjectivePoint
    public let delta_g2: G2ProjectivePoint
    public let ic: [PointProjective]
}

public struct Groth16Proof {
    public let a: PointProjective   // G1
    public let b: G2ProjectivePoint // G2
    public let c: PointProjective   // G1
}
