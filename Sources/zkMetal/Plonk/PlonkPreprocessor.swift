// PlonkPreprocessor — One-time circuit preprocessing for Plonk
//
// Computes selector and permutation polynomials from the circuit,
// commits them via KZG, and produces a PlonkSetup for reuse across proofs.
//
// Permutation construction:
//   For 3n wires (a_0..a_{n-1}, b_0..b_{n-1}, c_0..c_{n-1}), we define
//   three permutation polynomials sigma_1, sigma_2, sigma_3 that encode
//   which wire positions must hold equal values (copy constraints).
//
//   Wire position encoding: position i in column k (k=0,1,2) maps to
//   k*n + i, and the coset element is omega^i * {1, k1, k2} for columns 0,1,2.

import Foundation
import NeonFieldOps

// MARK: - Setup result

public struct PlonkSetup {
    public let selectorCommitments: [PointProjective]   // [qL], [qR], [qO], [qM], [qC], [qRange], [qLookup], [qPoseidon]
    public let permutationCommitments: [PointProjective] // [sigma1], [sigma2], [sigma3]
    public let selectorPolys: [[Fr]]       // coefficient form (after iNTT)
    public let permutationPolys: [[Fr]]    // coefficient form
    public let selectorEvals: [[Fr]]       // evaluation form (NTT domain)
    public let permutationEvals: [[Fr]]    // evaluation form
    public let domain: [Fr]                // omega^0, omega^1, ..., omega^{n-1}
    public let omega: Fr                   // primitive n-th root of unity
    public let n: Int                      // domain size (power of 2)
    public let srs: [PointAffine]
    public let k1: Fr                      // coset generator for b-column
    public let k2: Fr                      // coset generator for c-column
    public let srsSecret: Fr               // toxic waste (for test verification only)
    public let lookupTables: [PlonkLookupTable]  // lookup tables for lookup gates
}

// MARK: - Preprocessor

public class PlonkPreprocessor {
    public let kzg: KZGEngine
    public let ntt: NTTEngine

    public init(kzg: KZGEngine, ntt: NTTEngine) {
        self.kzg = kzg
        self.ntt = ntt
    }

    /// One-time circuit preprocessing. Circuit must be padded to power of 2.
    public func setup(circuit: PlonkCircuit, srsSecret: Fr) throws -> PlonkSetup {
        let n = circuit.numGates
        precondition(n > 0 && (n & (n - 1)) == 0, "Circuit must be padded to power of 2")

        // 1. Compute domain: omega = primitive n-th root of unity
        let logN = Int(log2(Double(n)))
        let omega = computeNthRootOfUnity(logN: logN)
        var domain = [Fr](repeating: Fr.zero, count: n)
        domain[0] = Fr.one
        for i in 1..<n {
            domain[i] = frMul(domain[i - 1], omega)
        }

        // 2. Build selector evaluation vectors (values at omega^i)
        var qLEvals = [Fr](repeating: Fr.zero, count: n)
        var qREvals = [Fr](repeating: Fr.zero, count: n)
        var qOEvals = [Fr](repeating: Fr.zero, count: n)
        var qMEvals = [Fr](repeating: Fr.zero, count: n)
        var qCEvals = [Fr](repeating: Fr.zero, count: n)
        var qRangeEvals = [Fr](repeating: Fr.zero, count: n)
        var qLookupEvals = [Fr](repeating: Fr.zero, count: n)
        var qPoseidonEvals = [Fr](repeating: Fr.zero, count: n)

        for i in 0..<circuit.numGates {
            let g = circuit.gates[i]
            qLEvals[i] = g.qL
            qREvals[i] = g.qR
            qOEvals[i] = g.qO
            qMEvals[i] = g.qM
            qCEvals[i] = g.qC
            qRangeEvals[i] = g.qRange
            qLookupEvals[i] = g.qLookup
            qPoseidonEvals[i] = g.qPoseidon
        }

        // 3. iNTT to get coefficient form
        let qLCoeffs = try ntt.intt(qLEvals)
        let qRCoeffs = try ntt.intt(qREvals)
        let qOCoeffs = try ntt.intt(qOEvals)
        let qMCoeffs = try ntt.intt(qMEvals)
        let qCCoeffs = try ntt.intt(qCEvals)
        let qRangeCoeffs = try ntt.intt(qRangeEvals)
        let qLookupCoeffs = try ntt.intt(qLookupEvals)
        let qPoseidonCoeffs = try ntt.intt(qPoseidonEvals)

        let selectorPolys = [qLCoeffs, qRCoeffs, qOCoeffs, qMCoeffs, qCCoeffs,
                             qRangeCoeffs, qLookupCoeffs, qPoseidonCoeffs]
        let selectorEvals = [qLEvals, qREvals, qOEvals, qMEvals, qCEvals,
                             qRangeEvals, qLookupEvals, qPoseidonEvals]

        // 4. Commit to selector polynomials
        var selectorCommitments = [PointProjective]()
        for poly in selectorPolys {
            selectorCommitments.append(try kzg.commit(poly))
        }

        // 5. Coset generators k1, k2 (must be quadratic non-residues mod r)
        //    Using small constants that are not n-th roots of unity
        let k1 = frFromInt(2)
        let k2 = frFromInt(3)

        // 6. Build permutation from copy constraints
        //    Initial identity permutation: sigma[k][i] = omega^i * {1, k1, k2}
        var sigma = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: n), count: 3)
        for i in 0..<n {
            sigma[0][i] = domain[i]                          // omega^i
            sigma[1][i] = frMul(k1, domain[i])               // k1 * omega^i
            sigma[2][i] = frMul(k2, domain[i])               // k2 * omega^i
        }

        // Apply copy constraints using union-find on variables
        let numVars = circuit.wireAssignments.flatMap { $0 }.max().map { $0 + 1 } ?? 0
        var parent = Array(0..<numVars)

        func find(_ x: Int) -> Int {
            var x = x
            while parent[x] != x { parent[x] = parent[parent[x]]; x = parent[x] }
            return x
        }
        func union(_ a: Int, _ b: Int) {
            let ra = find(a), rb = find(b)
            if ra != rb { parent[ra] = rb }
        }

        // Union all copy constraints (explicit)
        for (a, b) in circuit.copyConstraints {
            union(a, b)
        }

        // Union implicit copy constraints from wire assignments
        // If the same variable appears in multiple wire positions, they must be equal
        var varPositions: [Int: [(col: Int, row: Int)]] = [:]
        for i in 0..<n {
            if i < circuit.wireAssignments.count {
                let wires = circuit.wireAssignments[i]
                for col in 0..<3 {
                    let v = wires[col]
                    varPositions[v, default: []].append((col: col, row: i))
                }
            }
        }

        // For each variable appearing in multiple positions, create a permutation cycle
        for (_, positions) in varPositions where positions.count > 1 {
            // Create a cycle: pos[0] -> pos[1] -> ... -> pos[last] -> pos[0]
            for k in 0..<positions.count {
                let src = positions[k]
                let dst = positions[(k + 1) % positions.count]

                // sigma[src.col][src.row] should map to the coset element of dst
                let cosetMul: Fr
                switch dst.col {
                case 0: cosetMul = Fr.one
                case 1: cosetMul = k1
                case 2: cosetMul = k2
                default: cosetMul = Fr.one
                }
                sigma[src.col][src.row] = frMul(cosetMul, domain[dst.row])
            }
        }

        // 7. iNTT permutation evals to get coefficient form
        let sigma1Coeffs = try ntt.intt(sigma[0])
        let sigma2Coeffs = try ntt.intt(sigma[1])
        let sigma3Coeffs = try ntt.intt(sigma[2])

        let permutationPolys = [sigma1Coeffs, sigma2Coeffs, sigma3Coeffs]
        let permutationEvals = [sigma[0], sigma[1], sigma[2]]

        // 8. Commit to permutation polynomials
        var permutationCommitments = [PointProjective]()
        for poly in permutationPolys {
            permutationCommitments.append(try kzg.commit(poly))
        }

        return PlonkSetup(
            selectorCommitments: selectorCommitments,
            permutationCommitments: permutationCommitments,
            selectorPolys: selectorPolys,
            permutationPolys: permutationPolys,
            selectorEvals: selectorEvals,
            permutationEvals: permutationEvals,
            domain: domain,
            omega: omega,
            n: n,
            srs: kzg.srs,
            k1: k1,
            k2: k2,
            srsSecret: srsSecret,
            lookupTables: circuit.lookupTables
        )
    }
}

// MARK: - Root of unity helpers

/// Compute omega = g^(2^(TWO_ADICITY - logN)) where g is the primitive 2^TWO_ADICITY root
public func computeNthRootOfUnity(logN: Int) -> Fr {
    precondition(logN <= Fr.TWO_ADICITY, "Domain too large for BN254 Fr (max 2^\(Fr.TWO_ADICITY))")
    var omega = Fr.from64(Fr.ROOT_OF_UNITY)
    // Square down from 2^TWO_ADICITY to 2^logN
    for _ in 0..<(Fr.TWO_ADICITY - logN) {
        omega = frSqr(omega)
    }
    return omega
}
