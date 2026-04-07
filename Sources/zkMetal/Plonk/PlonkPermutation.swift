// PlonkPermutation — Standalone permutation argument for Plonk
//
// Implements the grand product check proving that two multisets are equal,
// the core of Plonk's copy constraint enforcement. Can be used independently
// or within the full Plonk prover.
//
// The permutation argument proves:
//   Z(omega^0) = 1
//   Z(omega^{i+1}) = Z(omega^i) * prod_j (w_j(i) + beta*id_j(i) + gamma) /
//                                          (w_j(i) + beta*sigma_j(i) + gamma)
//   Z(omega^{n-1}) = 1  (grand product equals 1)
//
// where id_j(i) = omega^i * {1, k1, k2, ...} is the identity permutation
// and sigma_j(i) encodes the actual wire routing.

import Foundation
import NeonFieldOps

// MARK: - Copy Constraint Representation

/// Explicit copy constraint: two wire positions that must hold equal values.
///
/// A wire position is identified by (wire column index, row index).
/// For standard 3-wire Plonk: column 0 = a (left), 1 = b (right), 2 = c (output).
public struct PlonkCopyConstraint: Equatable {
    /// Source wire column (0-based)
    public let srcWire: Int
    /// Source row
    public let srcRow: Int
    /// Destination wire column (0-based)
    public let dstWire: Int
    /// Destination row
    public let dstRow: Int

    public init(srcWire: Int, srcRow: Int, dstWire: Int, dstRow: Int) {
        self.srcWire = srcWire
        self.srcRow = srcRow
        self.dstWire = dstWire
        self.dstRow = dstRow
    }

    /// Create from variable-level copy constraints and wire assignments.
    /// Given that variable `varA` and `varB` must be equal, and they appear
    /// in the wire assignment table, this creates the corresponding position-level constraint.
    public static func fromVariableEquality(
        varA: Int, varB: Int,
        wireAssignments: [[Int]],
        numWires: Int
    ) -> [PlonkCopyConstraint] {
        // Find all positions where each variable appears
        var posA = [(wire: Int, row: Int)]()
        var posB = [(wire: Int, row: Int)]()

        for row in 0..<wireAssignments.count {
            let wires = wireAssignments[row]
            for col in 0..<min(numWires, wires.count) {
                if wires[col] == varA { posA.append((wire: col, row: row)) }
                if wires[col] == varB { posB.append((wire: col, row: row)) }
            }
        }

        // Create cross-product constraints (typically just one pair)
        var result = [PlonkCopyConstraint]()
        if let a = posA.first, let b = posB.first {
            result.append(PlonkCopyConstraint(
                srcWire: a.wire, srcRow: a.row,
                dstWire: b.wire, dstRow: b.row
            ))
        }
        return result
    }
}

// MARK: - Permutation Argument

/// The Plonk permutation argument: proves multiset equality via a grand product accumulator.
///
/// Supports multi-wire permutations (3+ wires). The number of wires determines
/// the number of sigma polynomials and coset generators needed.
public struct PermutationArgument {

    /// Number of wires (columns) in the permutation. Standard Plonk uses 3.
    public let numWires: Int

    /// Coset generators: one per wire column beyond the first.
    /// For 3-wire Plonk: [k1, k2] where column 0 uses 1, column 1 uses k1, column 2 uses k2.
    public let cosetGenerators: [Fr]

    public init(numWires: Int = 3, cosetGenerators: [Fr]? = nil) {
        self.numWires = numWires
        if let generators = cosetGenerators {
            precondition(generators.count == numWires - 1,
                         "Need numWires-1 coset generators")
            self.cosetGenerators = generators
        } else {
            // Default: k_i = i+2 for standard Plonk
            var gens = [Fr]()
            for i in 0..<(numWires - 1) {
                gens.append(frFromInt(UInt64(i + 2)))
            }
            self.cosetGenerators = gens
        }
    }

    /// Get the coset multiplier for a given wire column.
    /// Column 0 -> Fr.one, Column j -> cosetGenerators[j-1]
    public func cosetMultiplier(forWire col: Int) -> Fr {
        col == 0 ? Fr.one : cosetGenerators[col - 1]
    }

    // MARK: - Grand Product Computation

    /// Compute the permutation grand product polynomial Z(x) in evaluation form.
    ///
    /// Z is defined on domain {omega^0, ..., omega^{n-1}} where:
    ///   Z(omega^0) = 1
    ///   Z(omega^{i+1}) = Z(omega^i) * numerator(i) / denominator(i)
    ///
    /// numerator(i) = prod_j (witness_j[i] + beta * id_j(omega^i) + gamma)
    /// denominator(i) = prod_j (witness_j[i] + beta * sigma_j[i] + gamma)
    ///
    /// - Parameters:
    ///   - witness: Per-wire witness evaluations. witness[j] has n elements for wire j.
    ///   - sigma: Per-wire sigma permutation evaluations. sigma[j] has n elements.
    ///   - beta: Permutation challenge beta.
    ///   - gamma: Permutation challenge gamma.
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
        precondition(witness.count == numWires, "Need \(numWires) witness columns")
        precondition(sigma.count == numWires, "Need \(numWires) sigma columns")
        for j in 0..<numWires {
            precondition(witness[j].count == n, "Witness column \(j) length mismatch")
            precondition(sigma[j].count == n, "Sigma column \(j) length mismatch")
        }

        // Use the optimized C path for 3-wire case if available
        if numWires == 3 {
            return computeGrandProduct3Wire(
                witness: witness, sigma: sigma,
                beta: beta, gamma: gamma, domain: domain
            )
        }

        // Generic multi-wire path
        return computeGrandProductGeneric(
            witness: witness, sigma: sigma,
            beta: beta, gamma: gamma, domain: domain
        )
    }

    /// Optimized 3-wire grand product using C batch operations.
    private func computeGrandProduct3Wire(
        witness: [[Fr]],
        sigma: [[Fr]],
        beta: Fr,
        gamma: Fr,
        domain: [Fr]
    ) -> [Fr] {
        let n = domain.count
        let k1 = cosetGenerators[0]
        let k2 = cosetGenerators[1]

        var zEvals = [Fr](repeating: Fr.zero, count: n)

        witness[0].withUnsafeBytes { aBuf in
            witness[1].withUnsafeBytes { bBuf in
                witness[2].withUnsafeBytes { cBuf in
                    sigma[0].withUnsafeBytes { s1Buf in
                        sigma[1].withUnsafeBytes { s2Buf in
                            sigma[2].withUnsafeBytes { s3Buf in
                                domain.withUnsafeBytes { dBuf in
                                    withUnsafeBytes(of: beta) { betaBuf in
                                        withUnsafeBytes(of: gamma) { gammaBuf in
                                            withUnsafeBytes(of: k1) { k1Buf in
                                                withUnsafeBytes(of: k2) { k2Buf in
                                                    zEvals.withUnsafeMutableBytes { zBuf in
                                                        plonk_compute_z_accumulator(
                                                            aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                                            bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                                            cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                                            s1Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                                            s2Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                                            s3Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                                            dBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                                            betaBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                                            gammaBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                                            k1Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                                            k2Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                                            Int32(n),
                                                            zBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                                                        )
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return zEvals
    }

    /// Generic multi-wire grand product (any number of wires).
    private func computeGrandProductGeneric(
        witness: [[Fr]],
        sigma: [[Fr]],
        beta: Fr,
        gamma: Fr,
        domain: [Fr]
    ) -> [Fr] {
        let n = domain.count

        // Compute numerator/denominator ratios, then running product
        // ratio[i] = prod_j (w_j[i] + beta*id_j(i) + gamma) /
        //                    (w_j[i] + beta*sigma_j[i] + gamma)

        // Step 1: Compute all numerators and denominators
        var numerators = [Fr](repeating: Fr.one, count: n)
        var denominators = [Fr](repeating: Fr.one, count: n)

        for i in 0..<n {
            for j in 0..<numWires {
                let kj = cosetMultiplier(forWire: j)
                // id_j(omega^i) = kj * omega^i
                let idVal = frMul(kj, domain[i])

                // numerator term: w_j[i] + beta * id_j(i) + gamma
                let numTerm = frAdd(frAdd(witness[j][i], frMul(beta, idVal)), gamma)
                numerators[i] = frMul(numerators[i], numTerm)

                // denominator term: w_j[i] + beta * sigma_j[i] + gamma
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

        // Step 3: Batch multiply numerators * invDenominators to get ratios
        var ratios = [Fr](repeating: Fr.zero, count: n)
        numerators.withUnsafeBytes { numBuf in
            invDenominators.withUnsafeBytes { invBuf in
                ratios.withUnsafeMutableBytes { ratBuf in
                    bn254_fr_batch_mul_parallel(
                        ratBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        numBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        invBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n)
                    )
                }
            }
        }

        // Step 4: Running product (sequential dependency, cannot batch)
        var zEvals = [Fr](repeating: Fr.zero, count: n)
        zEvals[0] = Fr.one
        for i in 0..<(n - 1) {
            zEvals[i + 1] = frMul(zEvals[i], ratios[i])
        }

        return zEvals
    }

    // MARK: - Quotient Polynomial Contribution

    /// Evaluate the permutation's contribution to the quotient polynomial.
    ///
    /// The permutation adds two terms to the quotient numerator:
    ///   1. alpha * [Z(x) * prod_j(w_j + beta*id_j + gamma) - Z(omega*x) * prod_j(w_j + beta*sigma_j + gamma)]
    ///   2. alpha^2 * (Z(x) - 1) * L_1(x)
    ///
    /// This returns the combined numerator in evaluation form on a coset domain
    /// (before dividing by Z_H(x)).
    ///
    /// - Parameters:
    ///   - Z: Grand product polynomial evaluations (length n).
    ///   - witness: Per-wire witness evaluations (numWires arrays, each length n).
    ///   - sigma: Per-wire sigma polynomial evaluations (numWires arrays, each length n).
    ///   - beta: Permutation challenge.
    ///   - gamma: Permutation challenge.
    ///   - alpha: Quotient separation challenge.
    ///   - domain: Evaluation domain [omega^0, ..., omega^{n-1}].
    /// - Returns: Permutation contribution to the quotient numerator, length n.
    public func evaluatePermutationQuotient(
        Z: [Fr],
        witness: [[Fr]],
        sigma: [[Fr]],
        beta: Fr,
        gamma: Fr,
        alpha: Fr,
        domain: [Fr]
    ) -> [Fr] {
        let n = domain.count
        let alpha2 = frSqr(alpha)

        var result = [Fr](repeating: Fr.zero, count: n)

        for i in 0..<n {
            // Term 1: Z(omega^i) * prod_j(w_j[i] + beta*id_j(i) + gamma)
            //       - Z(omega^{i+1}) * prod_j(w_j[i] + beta*sigma_j[i] + gamma)
            var numProd = Fr.one
            var denProd = Fr.one

            for j in 0..<numWires {
                let kj = cosetMultiplier(forWire: j)
                let idVal = frMul(kj, domain[i])
                numProd = frMul(numProd, frAdd(frAdd(witness[j][i], frMul(beta, idVal)), gamma))
                denProd = frMul(denProd, frAdd(frAdd(witness[j][i], frMul(beta, sigma[j][i])), gamma))
            }

            let iNext = (i + 1) % n
            let permTerm = frSub(frMul(Z[i], numProd), frMul(Z[iNext], denProd))

            // Term 2: (Z(omega^i) - 1) * L_1(omega^i)
            // L_1(omega^i) = 1 if i==0, else 0  (on the evaluation domain)
            let boundaryTerm: Fr
            if i == 0 {
                boundaryTerm = frSub(Z[0], Fr.one)
            } else {
                boundaryTerm = Fr.zero
            }

            result[i] = frAdd(frMul(alpha, permTerm), frMul(alpha2, boundaryTerm))
        }

        return result
    }

    // MARK: - Verification

    /// Verify the permutation argument from polynomial evaluations.
    ///
    /// Checks the two key identities at a random evaluation point zeta:
    ///   1. L_1(zeta) * (Z(zeta) - 1) = 0
    ///   2. Z(zeta) * prod_j(w_j(zeta) + beta*id_j(zeta) + gamma)
    ///      = Z(zeta*omega) * prod_j(w_j(zeta) + beta*sigma_j(zeta) + gamma)
    ///
    /// Note: In a full Plonk verifier, these checks are folded into the
    /// linearization polynomial and verified via KZG opening. This standalone
    /// check is useful for testing and modular verification.
    ///
    /// - Parameters:
    ///   - zEval: Z(zeta)
    ///   - zOmegaEval: Z(zeta * omega)
    ///   - witnessEvals: [w_j(zeta)] for each wire j
    ///   - sigmaEvals: [sigma_j(zeta)] for each wire j
    ///   - beta: Permutation challenge
    ///   - gamma: Permutation challenge
    ///   - zeta: Evaluation point
    ///   - l1Zeta: L_1(zeta) = (zeta^n - 1) / (n * (zeta - 1))
    /// - Returns: True if permutation identities hold at zeta.
    public func verifyPermutation(
        zEval: Fr,
        zOmegaEval: Fr,
        witnessEvals: [Fr],
        sigmaEvals: [Fr],
        beta: Fr,
        gamma: Fr,
        zeta: Fr,
        l1Zeta: Fr
    ) -> Bool {
        precondition(witnessEvals.count == numWires)
        precondition(sigmaEvals.count == numWires)

        // Check 1: L_1(zeta) * (Z(zeta) - 1) = 0
        let boundaryCheck = frMul(l1Zeta, frSub(zEval, Fr.one))
        if !boundaryCheck.isZero {
            return false
        }

        // Check 2: Z(zeta) * prod_j(w_j(zeta) + beta*id_j(zeta) + gamma)
        //         == Z(zeta*omega) * prod_j(w_j(zeta) + beta*sigma_j(zeta) + gamma)
        var numProd = Fr.one
        var denProd = Fr.one

        for j in 0..<numWires {
            let kj = cosetMultiplier(forWire: j)
            let idVal = frMul(kj, zeta)
            numProd = frMul(numProd, frAdd(frAdd(witnessEvals[j], frMul(beta, idVal)), gamma))
            denProd = frMul(denProd, frAdd(frAdd(witnessEvals[j], frMul(beta, sigmaEvals[j])), gamma))
        }

        let lhs = frMul(zEval, numProd)
        let rhs = frMul(zOmegaEval, denProd)

        return frSub(lhs, rhs).isZero
    }
}

// MARK: - Build Permutation from Copy Constraints

/// Construct sigma permutation polynomials from explicit copy constraints.
///
/// This is the standalone version of the permutation builder. It takes
/// `PlonkCopyConstraint` objects and produces sigma polynomial evaluations.
///
/// For each wire position (col, row), the identity permutation maps it to
/// the coset element: cosetMultiplier(col) * omega^row.
/// Copy constraints modify this mapping to create permutation cycles
/// among positions that must hold equal values.
///
/// - Parameters:
///   - copies: Explicit copy constraints (position-level).
///   - numWires: Number of wire columns (typically 3).
///   - domainSize: Domain size n (must be power of 2).
///   - domain: Evaluation domain [omega^0, ..., omega^{n-1}].
///   - cosetGenerators: Coset generators [k1, k2, ...] for columns 1, 2, ...
///                      Pass nil for defaults (k_i = i+2).
/// - Returns: numWires sigma evaluation arrays, each of length domainSize.
public func buildPermutationFromCopyConstraints(
    copies: [PlonkCopyConstraint],
    numWires: Int = 3,
    domainSize: Int,
    domain: [Fr],
    cosetGenerators: [Fr]? = nil
) -> [[Fr]] {
    let n = domainSize
    precondition(domain.count == n)

    let generators: [Fr]
    if let gens = cosetGenerators {
        generators = gens
    } else {
        generators = (0..<(numWires - 1)).map { frFromInt(UInt64($0 + 2)) }
    }

    func cosetMul(_ col: Int) -> Fr {
        col == 0 ? Fr.one : generators[col - 1]
    }

    // Initialize identity permutation using batch C calls
    var sigma = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: n), count: numWires)
    domain.withUnsafeBytes { domBuf in
        let domPtr = domBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
        for col in 0..<numWires {
            var km = cosetMul(col)
            sigma[col].withUnsafeMutableBytes { sigBuf in
                withUnsafeBytes(of: &km) { kmBuf in
                    bn254_fr_batch_mul_scalar_parallel(
                        sigBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        domPtr,
                        kmBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n)
                    )
                }
            }
        }
    }

    // Build equivalence classes from copy constraints using union-find
    // Position encoding: col * n + row
    let totalPositions = numWires * n
    var parent = Array(0..<totalPositions)
    var rank = [Int](repeating: 0, count: totalPositions)

    func find(_ x: Int) -> Int {
        var x = x
        while parent[x] != x {
            parent[x] = parent[parent[x]]  // path compression
            x = parent[x]
        }
        return x
    }

    func union(_ a: Int, _ b: Int) {
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

    for cc in copies {
        let posA = cc.srcWire * n + cc.srcRow
        let posB = cc.dstWire * n + cc.dstRow
        union(posA, posB)
    }

    // Group positions by equivalence class
    var classes = [Int: [Int]]()
    for pos in 0..<totalPositions {
        let root = find(pos)
        classes[root, default: []].append(pos)
    }

    // For each equivalence class with >1 member, create a permutation cycle
    for (_, members) in classes where members.count > 1 {
        for k in 0..<members.count {
            let srcPos = members[k]
            let dstPos = members[(k + 1) % members.count]

            let srcCol = srcPos / n
            let srcRow = srcPos % n
            let dstCol = dstPos / n
            let dstRow = dstPos % n

            sigma[srcCol][srcRow] = frMul(cosetMul(dstCol), domain[dstRow])
        }
    }

    return sigma
}

/// Build sigma permutations from variable-level wire assignments and copy constraints.
///
/// This is the higher-level API matching the PlonkPreprocessor interface:
/// given wire assignments (variable indices per gate) and variable equality
/// constraints, build the sigma permutation evaluations.
///
/// - Parameters:
///   - wireAssignments: Per-gate wire variable indices. wireAssignments[row][col].
///   - variableCopies: Pairs of variable indices that must be equal.
///   - numWires: Number of wire columns (typically 3).
///   - domainSize: Domain size n.
///   - domain: Evaluation domain.
///   - cosetGenerators: Optional coset generators.
/// - Returns: numWires sigma evaluation arrays.
public func buildPermutationFromVariables(
    wireAssignments: [[Int]],
    variableCopies: [(Int, Int)] = [],
    numWires: Int = 3,
    domainSize: Int,
    domain: [Fr],
    cosetGenerators: [Fr]? = nil
) -> [[Fr]] {
    // This replicates SigmaPermutationBuilder.buildSigmaEvals logic
    // but supports arbitrary wire counts

    let n = domainSize
    let generators: [Fr]
    if let gens = cosetGenerators {
        generators = gens
    } else {
        generators = (0..<(numWires - 1)).map { frFromInt(UInt64($0 + 2)) }
    }

    func cosetMul(_ col: Int) -> Fr {
        col == 0 ? Fr.one : generators[col - 1]
    }

    // Initialize identity permutation using batch C calls
    var sigma = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: n), count: numWires)
    domain.withUnsafeBytes { domBuf in
        let domPtr = domBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
        for col in 0..<numWires {
            var km = cosetMul(col)
            sigma[col].withUnsafeMutableBytes { sigBuf in
                withUnsafeBytes(of: &km) { kmBuf in
                    bn254_fr_batch_mul_scalar_parallel(
                        sigBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        domPtr,
                        kmBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n)
                    )
                }
            }
        }
    }

    // Build variable -> position mapping
    var varPositions = [Int: [(col: Int, row: Int)]]()
    for row in 0..<min(n, wireAssignments.count) {
        let wires = wireAssignments[row]
        for col in 0..<min(numWires, wires.count) {
            varPositions[wires[col], default: []].append((col: col, row: row))
        }
    }

    // For each variable appearing in multiple positions, create a permutation cycle
    for (_, positions) in varPositions where positions.count > 1 {
        for k in 0..<positions.count {
            let src = positions[k]
            let dst = positions[(k + 1) % positions.count]
            sigma[src.col][src.row] = frMul(cosetMul(dst.col), domain[dst.row])
        }
    }

    return sigma
}
