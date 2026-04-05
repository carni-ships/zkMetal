// Halo2Permutation — Halo2-style generalized permutation argument engine
//
// Implements the permutation argument from the Halo2 protocol, which uses a
// generalized permutation (not just copy constraints between 3 wires) to express
// arbitrary wiring between cells in different columns and rows.
//
// Architecture:
//   1. Halo2PermutationAssembly — builds permutation cycles from equality constraints
//      using union-find, then generates sigma polynomials
//   2. Halo2PermutationProver — computes the grand product Z(x) and commits
//   3. Halo2PermutationVerifier — checks Z(omega^0)=1, Z(omega^n)=1, and
//      the per-row transition identity
//
// The permutation argument proves:
//   Z(omega^0) = 1
//   Z(omega^{i+1}) = Z(omega^i) * prod_j (f_j(i) + beta*delta^j*omega^i + gamma)
//                                       / (f_j(i) + beta*sigma_j(i) + gamma)
//   Z(omega^{n-1}) -> final ratio gives 1
//
// where delta^j is the coset multiplier for column j (generalizing Plonk's k1, k2).

import Foundation
import NeonFieldOps

// MARK: - Union-Find for Permutation Cycles

/// Weighted union-find with path compression for building permutation cycles.
/// Each element is a flat position index: flatCol * domainSize + row.
private struct UnionFind {
    private var parent: [Int]
    private var rank: [Int]

    init(size: Int) {
        parent = Array(0..<size)
        rank = [Int](repeating: 0, count: size)
    }

    mutating func find(_ x: Int) -> Int {
        var x = x
        while parent[x] != x {
            parent[x] = parent[parent[x]]  // path halving
            x = parent[x]
        }
        return x
    }

    mutating func union(_ a: Int, _ b: Int) {
        let ra = find(a), rb = find(b)
        if ra == rb { return }
        if rank[ra] < rank[rb] {
            parent[ra] = rb
        } else if rank[ra] > rank[rb] {
            parent[rb] = ra
        } else {
            parent[rb] = ra
            rank[ra] += 1
        }
    }
}

// MARK: - Permutation Assembly

/// Builds the Halo2-style generalized permutation from equality constraints.
///
/// Usage:
/// ```swift
/// var assembly = Halo2PermutationAssembly(numColumns: 4, domainSize: 8)
/// assembly.addEquality(a: (col: 0, row: 0), b: (col: 2, row: 3))
/// assembly.addEquality(a: (col: 0, row: 0), b: (col: 1, row: 5))
/// let sigma = assembly.buildSigmaPolynomials(domain: domain)
/// ```
///
/// Internally uses union-find to build equivalence classes from pairwise
/// equality constraints, then constructs a cyclic permutation within each class.
public struct Halo2PermutationAssembly {

    /// Number of columns participating in the permutation.
    public let numColumns: Int

    /// Domain size (number of rows, must be power of 2).
    public let domainSize: Int

    /// Coset multipliers: delta^0=1, delta^1, delta^2, ...
    /// Column j uses delta^j to separate its identity permutation from other columns.
    public let cosetMultipliers: [Fr]

    /// Accumulated equality constraints as (flatPosA, flatPosB) pairs.
    private var equalities: [(Int, Int)] = []

    /// Initialize with the given number of columns and domain size.
    ///
    /// - Parameters:
    ///   - numColumns: Number of columns in the permutation.
    ///   - domainSize: Number of rows (must be power of 2).
    ///   - cosetMultipliers: Optional custom coset multipliers. If nil, uses [1, 2, 3, ...].
    public init(numColumns: Int, domainSize: Int, cosetMultipliers: [Fr]? = nil) {
        self.numColumns = numColumns
        self.domainSize = domainSize

        if let cm = cosetMultipliers {
            precondition(cm.count == numColumns, "Need exactly numColumns coset multipliers")
            self.cosetMultipliers = cm
        } else {
            // Default: column j uses (j+1) as coset multiplier
            // Column 0 -> 1, Column 1 -> 2, Column 2 -> 3, etc.
            var mults = [Fr]()
            for j in 0..<numColumns {
                mults.append(frFromInt(UInt64(j + 1)))
            }
            self.cosetMultipliers = mults
        }
    }

    /// Constrain two cells to have equal values.
    ///
    /// Both cells are identified by (column index, row index).
    /// The assembly builds permutation cycles so that all cells in
    /// the same equivalence class map to each other cyclically in sigma.
    ///
    /// - Parameters:
    ///   - a: First cell as (col: column index, row: row index).
    ///   - b: Second cell as (col: column index, row: row index).
    public mutating func addEquality(a: (col: Int, row: Int), b: (col: Int, row: Int)) {
        precondition(a.col >= 0 && a.col < numColumns, "Column \(a.col) out of range")
        precondition(b.col >= 0 && b.col < numColumns, "Column \(b.col) out of range")
        precondition(a.row >= 0 && a.row < domainSize, "Row \(a.row) out of range")
        precondition(b.row >= 0 && b.row < domainSize, "Row \(b.row) out of range")

        let posA = a.col * domainSize + a.row
        let posB = b.col * domainSize + b.row
        equalities.append((posA, posB))
    }

    /// The number of equality constraints added so far.
    public var equalityCount: Int { equalities.count }

    /// Build the sigma permutation polynomials from accumulated equality constraints.
    ///
    /// Returns `numColumns` arrays, each of length `domainSize`. sigma[j][i] contains
    /// the coset element that position (j, i) maps to under the permutation.
    ///
    /// For the identity permutation (no equalities), sigma[j][i] = delta^j * omega^i.
    /// Equality constraints modify sigma to create cyclic permutations within each
    /// equivalence class.
    ///
    /// - Parameter domain: Evaluation domain [omega^0, omega^1, ..., omega^{n-1}].
    /// - Returns: Array of numColumns sigma polynomial evaluations.
    public func buildSigmaPolynomials(domain: [Fr]) -> [[Fr]] {
        let n = domainSize
        precondition(domain.count == n, "Domain size mismatch")

        // Step 1: Initialize identity permutation
        // sigma[j][i] = cosetMultipliers[j] * omega^i
        var sigma = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: n), count: numColumns)
        for j in 0..<numColumns {
            let delta_j = cosetMultipliers[j]
            for i in 0..<n {
                sigma[j][i] = frMul(delta_j, domain[i])
            }
        }

        // Step 2: Build equivalence classes via union-find
        let totalPositions = numColumns * n
        var uf = UnionFind(size: totalPositions)

        for (posA, posB) in equalities {
            uf.union(posA, posB)
        }

        // Step 3: Group positions by equivalence class
        var classes = [Int: [Int]]()
        for pos in 0..<totalPositions {
            let root = uf.find(pos)
            classes[root, default: []].append(pos)
        }

        // Step 4: For each class with >1 member, create a cyclic permutation
        // Position pos maps to (col, row) where col = pos / n, row = pos % n
        for (_, members) in classes where members.count > 1 {
            for k in 0..<members.count {
                let srcPos = members[k]
                let dstPos = members[(k + 1) % members.count]

                let srcCol = srcPos / n
                let srcRow = srcPos % n
                let dstCol = dstPos / n
                let dstRow = dstPos % n

                // sigma[srcCol][srcRow] = cosetMultipliers[dstCol] * omega^dstRow
                sigma[srcCol][srcRow] = frMul(cosetMultipliers[dstCol], domain[dstRow])
            }
        }

        return sigma
    }

    /// Build the identity permutation polynomials (no wiring).
    /// id[j][i] = cosetMultipliers[j] * omega^i
    ///
    /// - Parameter domain: Evaluation domain.
    /// - Returns: Array of numColumns identity polynomial evaluations.
    public func buildIdentityPolynomials(domain: [Fr]) -> [[Fr]] {
        let n = domainSize
        precondition(domain.count == n)
        var id = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: n), count: numColumns)
        for j in 0..<numColumns {
            let delta_j = cosetMultipliers[j]
            for i in 0..<n {
                id[j][i] = frMul(delta_j, domain[i])
            }
        }
        return id
    }
}

// MARK: - Permutation Prover

/// Proves the Halo2 permutation argument by computing the grand product polynomial Z(x).
///
/// The grand product accumulates:
///   Z(omega^0) = 1
///   Z(omega^{i+1}) = Z(omega^i) * prod_j (f_j(i) + beta * delta^j * omega^i + gamma)
///                                       / (f_j(i) + beta * sigma_j(i) + gamma)
///
/// If the permutation is satisfied (all constrained cells hold equal values),
/// then Z(omega^n) = 1 (the product telescopes).
public struct Halo2PermutationProver {

    /// Number of columns in the permutation.
    public let numColumns: Int

    /// Coset multipliers for the identity permutation.
    public let cosetMultipliers: [Fr]

    public init(numColumns: Int, cosetMultipliers: [Fr]) {
        precondition(cosetMultipliers.count == numColumns)
        self.numColumns = numColumns
        self.cosetMultipliers = cosetMultipliers
    }

    /// Convenience initializer from a permutation assembly.
    public init(assembly: Halo2PermutationAssembly) {
        self.numColumns = assembly.numColumns
        self.cosetMultipliers = assembly.cosetMultipliers
    }

    /// Compute the grand product polynomial Z(x) in evaluation form.
    ///
    /// - Parameters:
    ///   - witness: Per-column witness evaluations. witness[j] has n elements for column j.
    ///   - sigma: Per-column sigma permutation evaluations from the assembly.
    ///   - beta: Random challenge beta (separates permutation from gate constraints).
    ///   - gamma: Random challenge gamma (prevents zero denominators).
    ///   - domain: Evaluation domain [omega^0, ..., omega^{n-1}].
    /// - Returns: Grand product polynomial Z(x) in evaluation form, length n.
    public func computeGrandProduct(
        witness: [[Fr]],
        sigma: [[Fr]],
        beta: Fr,
        gamma: Fr,
        domain: [Fr]
    ) -> [Fr] {
        let n = domain.count
        precondition(witness.count == numColumns, "Need \(numColumns) witness columns")
        precondition(sigma.count == numColumns, "Need \(numColumns) sigma columns")

        // Step 1: Compute per-row numerator and denominator products
        var numerators = [Fr](repeating: Fr.one, count: n)
        var denominators = [Fr](repeating: Fr.one, count: n)

        for i in 0..<n {
            for j in 0..<numColumns {
                // Identity element for column j at row i: delta^j * omega^i
                let idVal = frMul(cosetMultipliers[j], domain[i])

                // numerator: f_j(i) + beta * id_j(i) + gamma
                let numTerm = frAdd(frAdd(witness[j][i], frMul(beta, idVal)), gamma)
                numerators[i] = frMul(numerators[i], numTerm)

                // denominator: f_j(i) + beta * sigma_j(i) + gamma
                let denTerm = frAdd(frAdd(witness[j][i], frMul(beta, sigma[j][i])), gamma)
                denominators[i] = frMul(denominators[i], denTerm)
            }
        }

        // Step 2: Batch invert denominators
        var invDenominators = [Fr](repeating: Fr.zero, count: n)
        denominators.withUnsafeBytes { denBuf in
            invDenominators.withUnsafeMutableBytes { invBuf in
                bn254_fr_batch_inverse(
                    denBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(n),
                    invBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                )
            }
        }

        // Step 3: Running product Z[0] = 1, Z[i+1] = Z[i] * num[i] / den[i]
        var zEvals = [Fr](repeating: Fr.zero, count: n)
        zEvals[0] = Fr.one
        for i in 0..<(n - 1) {
            let ratio = frMul(numerators[i], invDenominators[i])
            zEvals[i + 1] = frMul(zEvals[i], ratio)
        }

        return zEvals
    }

    /// Compute the final grand product value (what Z(omega^n) would be).
    ///
    /// If the permutation is valid, this should equal Fr.one.
    /// This computes Z[n-1] * (num[n-1] / den[n-1]).
    ///
    /// - Parameters:
    ///   - zEvals: The grand product evaluations from computeGrandProduct.
    ///   - witness: Per-column witness evaluations.
    ///   - sigma: Per-column sigma permutation evaluations.
    ///   - beta: Random challenge beta.
    ///   - gamma: Random challenge gamma.
    ///   - domain: Evaluation domain.
    /// - Returns: The closing value of the grand product (should be 1 for valid permutation).
    public func grandProductClosingValue(
        zEvals: [Fr],
        witness: [[Fr]],
        sigma: [[Fr]],
        beta: Fr,
        gamma: Fr,
        domain: [Fr]
    ) -> Fr {
        let n = domain.count
        let lastIdx = n - 1

        var finalNum = Fr.one
        var finalDen = Fr.one
        for j in 0..<numColumns {
            let idVal = frMul(cosetMultipliers[j], domain[lastIdx])
            finalNum = frMul(finalNum, frAdd(frAdd(witness[j][lastIdx], frMul(beta, idVal)), gamma))
            finalDen = frMul(finalDen, frAdd(frAdd(witness[j][lastIdx], frMul(beta, sigma[j][lastIdx])), gamma))
        }

        return frMul(zEvals[lastIdx], frMul(finalNum, frInverse(finalDen)))
    }
}

// MARK: - Permutation Verifier

/// Verifies the Halo2 permutation argument.
///
/// Checks:
///   1. Z(omega^0) = 1 (boundary constraint)
///   2. Z(omega^n) = 1 (closing constraint — the grand product telescopes)
///   3. Per-row transition: Z(omega^{i+1}) * prod_j(f_j + beta*sigma_j + gamma)
///                        = Z(omega^i) * prod_j(f_j + beta*id_j + gamma)
public struct Halo2PermutationVerifier {

    /// Number of columns in the permutation.
    public let numColumns: Int

    /// Coset multipliers for the identity permutation.
    public let cosetMultipliers: [Fr]

    public init(numColumns: Int, cosetMultipliers: [Fr]) {
        precondition(cosetMultipliers.count == numColumns)
        self.numColumns = numColumns
        self.cosetMultipliers = cosetMultipliers
    }

    /// Convenience initializer from a permutation assembly.
    public init(assembly: Halo2PermutationAssembly) {
        self.numColumns = assembly.numColumns
        self.cosetMultipliers = assembly.cosetMultipliers
    }

    /// Verify the permutation argument from evaluation-form polynomials.
    ///
    /// Checks all three conditions on the evaluation domain:
    ///   1. Z[0] = 1
    ///   2. For each i: Z[i+1] * den_prod[i] = Z[i] * num_prod[i]
    ///   3. The closing value (Z[n-1] * last_ratio) = 1
    ///
    /// - Parameters:
    ///   - zEvals: Grand product polynomial evaluations, length n.
    ///   - witness: Per-column witness evaluations.
    ///   - sigma: Per-column sigma permutation evaluations.
    ///   - beta: Random challenge beta.
    ///   - gamma: Random challenge gamma.
    ///   - domain: Evaluation domain [omega^0, ..., omega^{n-1}].
    /// - Returns: True if all permutation checks pass.
    public func verify(
        zEvals: [Fr],
        witness: [[Fr]],
        sigma: [[Fr]],
        beta: Fr,
        gamma: Fr,
        domain: [Fr]
    ) -> Bool {
        let n = domain.count
        precondition(zEvals.count == n)
        precondition(witness.count == numColumns)
        precondition(sigma.count == numColumns)

        // Check 1: Z[0] = 1
        guard zEvals[0] == Fr.one else { return false }

        // Check 2: Transition identity at each row
        for i in 0..<(n - 1) {
            var numProd = Fr.one
            var denProd = Fr.one

            for j in 0..<numColumns {
                let idVal = frMul(cosetMultipliers[j], domain[i])
                numProd = frMul(numProd, frAdd(frAdd(witness[j][i], frMul(beta, idVal)), gamma))
                denProd = frMul(denProd, frAdd(frAdd(witness[j][i], frMul(beta, sigma[j][i])), gamma))
            }

            // Z[i+1] * denProd should equal Z[i] * numProd
            let lhs = frMul(zEvals[i + 1], denProd)
            let rhs = frMul(zEvals[i], numProd)
            guard frSub(lhs, rhs).isZero else { return false }
        }

        // Check 3: Closing value = 1
        var finalNum = Fr.one
        var finalDen = Fr.one
        for j in 0..<numColumns {
            let idVal = frMul(cosetMultipliers[j], domain[n - 1])
            finalNum = frMul(finalNum, frAdd(frAdd(witness[j][n - 1], frMul(beta, idVal)), gamma))
            finalDen = frMul(finalDen, frAdd(frAdd(witness[j][n - 1], frMul(beta, sigma[j][n - 1])), gamma))
        }

        let closingValue = frMul(zEvals[n - 1], frMul(finalNum, frInverse(finalDen)))
        guard closingValue == Fr.one else { return false }

        return true
    }

    /// Verify the permutation at a single evaluation point (for IOP-based verification).
    ///
    /// Checks:
    ///   1. L_1(zeta) * (Z(zeta) - 1) = 0  (boundary at omega^0)
    ///   2. Z(zeta) * prod_j(f_j(zeta) + beta*id_j(zeta) + gamma)
    ///      = Z(zeta*omega) * prod_j(f_j(zeta) + beta*sigma_j(zeta) + gamma)
    ///
    /// - Parameters:
    ///   - zEval: Z(zeta)
    ///   - zOmegaEval: Z(zeta * omega)
    ///   - witnessEvals: [f_j(zeta)] for each column j
    ///   - sigmaEvals: [sigma_j(zeta)] for each column j
    ///   - beta: Permutation challenge
    ///   - gamma: Permutation challenge
    ///   - zeta: Evaluation point
    ///   - l1Zeta: L_1(zeta) = (zeta^n - 1) / (n * (zeta - 1))
    /// - Returns: True if permutation identities hold at zeta.
    public func verifyAtPoint(
        zEval: Fr,
        zOmegaEval: Fr,
        witnessEvals: [Fr],
        sigmaEvals: [Fr],
        beta: Fr,
        gamma: Fr,
        zeta: Fr,
        l1Zeta: Fr
    ) -> Bool {
        precondition(witnessEvals.count == numColumns)
        precondition(sigmaEvals.count == numColumns)

        // Check 1: L_1(zeta) * (Z(zeta) - 1) = 0
        let boundaryCheck = frMul(l1Zeta, frSub(zEval, Fr.one))
        guard boundaryCheck.isZero else { return false }

        // Check 2: Transition identity at zeta
        var numProd = Fr.one
        var denProd = Fr.one

        for j in 0..<numColumns {
            let idVal = frMul(cosetMultipliers[j], zeta)
            numProd = frMul(numProd, frAdd(frAdd(witnessEvals[j], frMul(beta, idVal)), gamma))
            denProd = frMul(denProd, frAdd(frAdd(witnessEvals[j], frMul(beta, sigmaEvals[j])), gamma))
        }

        let lhs = frMul(zEval, numProd)
        let rhs = frMul(zOmegaEval, denProd)

        return frSub(lhs, rhs).isZero
    }
}
