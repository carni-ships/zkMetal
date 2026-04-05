// PlonkLookupGate -- Plookup-style permutation argument for table lookups within Plonk
//
// Implements the lookup argument from the Plookup paper (Gabizon-Williamson 2020)
// integrated into the Plonk proof system.
//
// Protocol overview:
//   Given a table T[0..N-1] and lookup values f[0..m-1] where each f[i] in T:
//
//   1. Sort the concatenation (f, T) into a vector s of length m+N.
//   2. Compute the grand product accumulator Z_lookup(X):
//      Z_lookup(omega^0) = 1
//      Z_lookup(omega^{i+1}) = Z_lookup(omega^i) *
//        (1 + beta) * (gamma + f_i) * (gamma*(1+beta) + s_i + beta*s_{i+1})
//        / ((gamma*(1+beta) + t_i + beta*t_{i+1}) * (gamma*(1+beta) + s_i + beta*s_{i+1}))
//   3. The prover commits to Z_lookup and proves Z_lookup(omega^{n-1}) = 1.
//
// This is more efficient than the vanishing product approach for large tables,
// as the constraint degree is constant (independent of table size).
//
// Integration with existing LogUp/Lasso engines:
//   - For small tables (< 2^10): use PlonkLookupGate (Plookup, constant degree)
//   - For medium tables (2^10 - 2^20): use LookupEngine (LogUp, sumcheck-based)
//   - For large structured tables: use LassoEngine (decomposition)
//   - For large unstructured tables: use CQEngine (cached quotients, sublinear prover)

import Foundation
import NeonFieldOps

// MARK: - Plookup Argument

/// Plookup-style lookup argument within the Plonk framework.
///
/// The argument proves that a set of lookup values all belong to a fixed table.
/// It adds a grand product accumulator Z_lookup(X) to the Plonk proof, similar
/// to how the permutation argument uses Z(X).
public class PlonkLookupArgument {
    /// The lookup table values (must be sorted for the Plookup protocol)
    public let table: [Fr]
    /// Domain size
    public let n: Int

    public init(table: [Fr], n: Int) {
        precondition(!table.isEmpty, "Lookup table must not be empty")
        self.table = table
        self.n = n
    }

    // MARK: - Sorted vector computation

    /// Compute the sorted concatenation of (f, t) as required by Plookup.
    ///
    /// The sorted vector s has the property that:
    ///   - Every element of f appears in s
    ///   - Every element of t appears in s
    ///   - s is sorted according to the table ordering
    ///   - Adjacent equal elements in s correspond to lookup matches
    ///
    /// Returns: sorted vector s of length n (padded to domain size).
    public func computeSortedVector(lookupValues: [Fr]) -> [Fr] {
        // Build a map from table value to its position in the table
        var tablePos = [FrKey: Int]()
        for (i, t) in table.enumerated() {
            tablePos[FrKey(t)] = i
        }

        // Count multiplicities
        var mult = [Int](repeating: 0, count: table.count)
        for f in lookupValues {
            guard let idx = tablePos[FrKey(f)] else {
                preconditionFailure("Lookup value not in table")
            }
            mult[idx] += 1
        }

        // Build sorted vector: for each table entry, emit the table value
        // (mult[i] + 1) times if it was looked up, or once if not.
        // This interleaves table and lookup values in table order.
        var sorted = [Fr]()
        sorted.reserveCapacity(lookupValues.count + table.count)
        for i in 0..<table.count {
            // Always include the table entry at least once
            sorted.append(table[i])
            // Add extra copies for each lookup
            for _ in 0..<mult[i] {
                sorted.append(table[i])
            }
        }

        // Pad to domain size n with the last table value
        let padValue = table.last!
        while sorted.count < n {
            sorted.append(padValue)
        }
        // Truncate if too long (shouldn't happen with correct padding)
        if sorted.count > n {
            sorted = Array(sorted.prefix(n))
        }

        return sorted
    }

    // MARK: - Grand product accumulator

    /// Compute the grand product accumulator Z_lookup(X) for the Plookup argument.
    ///
    /// Z_lookup(omega^0) = 1
    /// Z_lookup(omega^{i+1}) = Z_lookup(omega^i) *
    ///   ((1+beta) * (gamma + f_i) * (gamma*(1+beta) + t_i + beta*t_{i+1}))
    ///   / (gamma*(1+beta) + s_i + beta*s_{i+1})^2
    ///
    /// - Parameters:
    ///   - lookupEvals: Lookup values f(omega^i) at each row (zero for non-lookup rows)
    ///   - tableEvals: Table polynomial evaluations t(omega^i) at each row
    ///   - sortedEvals: Sorted vector s(omega^i) evaluations
    ///   - beta: Random challenge
    ///   - gamma: Random challenge
    /// - Returns: Z_lookup evaluations at each domain point
    public func computeGrandProduct(
        lookupEvals: [Fr], tableEvals: [Fr], sortedEvals: [Fr],
        beta: Fr, gamma: Fr
    ) -> [Fr] {
        let nn = lookupEvals.count
        precondition(nn == n, "Lookup evals must match domain size")

        var zEvals = [Fr](repeating: Fr.zero, count: nn)
        zEvals[0] = Fr.one

        let onePlusBeta = frAdd(Fr.one, beta)
        let gammaOnePlusBeta = frMul(gamma, onePlusBeta)

        for i in 0..<(nn - 1) {
            // Numerator: (1+beta) * (gamma + f_i) * (gamma*(1+beta) + t_i + beta*t_{i+1})
            let num1 = onePlusBeta
            let num2 = frAdd(gamma, lookupEvals[i])
            let tNext = tableEvals[(i + 1) % nn]
            let num3 = frAdd(gammaOnePlusBeta, frAdd(tableEvals[i], frMul(beta, tNext)))
            let numerator = frMul(frMul(num1, num2), num3)

            // Denominator: (gamma*(1+beta) + s_i + beta*s_{i+1})^2
            // Note: the squared denominator comes from the two copies of s in the protocol
            // (one for the lookup side, one for the table side).
            // Simplified: we use a single denominator factor for each side.
            let sNext = sortedEvals[(i + 1) % nn]
            let den1 = frAdd(gammaOnePlusBeta, frAdd(sortedEvals[i], frMul(beta, sNext)))

            // For the Plookup grand product, we need den1 * den2 where
            // den2 handles the second half of the sorted vector.
            // In the simplified single-accumulator form:
            let denominator = frSqr(den1)

            // Z(omega^{i+1}) = Z(omega^i) * numerator / denominator
            let denInv = frInverse(denominator)
            zEvals[i + 1] = frMul(zEvals[i], frMul(numerator, denInv))
        }

        return zEvals
    }

    // MARK: - Quotient polynomial contribution

    /// Compute the lookup argument's contribution to the quotient polynomial.
    ///
    /// The lookup argument adds three constraints to the quotient:
    ///   1. Grand product initialization: L_1(x) * (Z_lookup(x) - 1) = 0
    ///   2. Grand product transition (the recursive relation)
    ///   3. Grand product finalization: L_{n-1}(x) * (Z_lookup(x) - 1) = 0
    ///
    /// - Parameters:
    ///   - zLookupCoeffs: Grand product polynomial Z_lookup in coefficient form
    ///   - lookupCoeffs: Lookup values polynomial f(x) in coefficient form
    ///   - tableCoeffs: Table polynomial t(x) in coefficient form
    ///   - sortedCoeffs: Sorted vector polynomial s(x) in coefficient form
    ///   - beta: Challenge
    ///   - gamma: Challenge
    ///   - alpha: Separation challenge (different power for each sub-constraint)
    ///   - omega: Root of unity
    ///   - ntt: NTT engine
    /// - Returns: Combined lookup constraint polynomial in coefficient form
    public func quotientContribution(
        zLookupCoeffs: [Fr], lookupCoeffs: [Fr],
        tableCoeffs: [Fr], sortedCoeffs: [Fr],
        beta: Fr, gamma: Fr, alpha: Fr, omega: Fr,
        ntt: NTTEngine
    ) throws -> [Fr] {
        let nn = n

        // --- Constraint 1: L_1(x) * (Z_lookup(x) - 1) = 0 ---
        var l1Evals = [Fr](repeating: Fr.zero, count: nn)
        l1Evals[0] = Fr.one
        let l1Coeffs = try ntt.intt(l1Evals)
        var zMinus1 = zLookupCoeffs
        if zMinus1.isEmpty {
            zMinus1 = [frSub(Fr.zero, Fr.one)]
        } else {
            zMinus1[0] = frSub(zMinus1[0], Fr.one)
        }
        let initConstraint = try polyMulNTT(l1Coeffs, zMinus1, ntt: ntt)

        // --- Constraint 2: Grand product transition ---
        // numerator: (1+beta) * (gamma + f(x)) * (gamma*(1+beta) + t(x) + beta*t(omega*x))
        // = Z_lookup(omega*x) * (gamma*(1+beta) + s(x) + beta*s(omega*x))^2
        // Equivalently: Z_lookup(x) * num_factors - Z_lookup(omega*x) * den_factors = 0

        let onePlusBeta = frAdd(Fr.one, beta)
        let gammaOnePlusBeta = frMul(gamma, onePlusBeta)
        let gammaConst = [gamma]
        let gopbConst = [gammaOnePlusBeta]

        // (gamma + f(x))
        let gammaF = polyAddCoeffs(lookupCoeffs, gammaConst)
        // (1+beta) * (gamma + f(x))
        let numFactor1 = polyScaleCoeffs(gammaF, onePlusBeta)

        // t(omega*x) = shift of t(x)
        let tShifted = polyShift(tableCoeffs, omega: omega)
        // gamma*(1+beta) + t(x) + beta*t(omega*x)
        let numFactor2 = polyAddCoeffs(
            polyAddCoeffs(gopbConst, tableCoeffs),
            polyScaleCoeffs(tShifted, beta))

        // Full numerator for transition: Z_lookup(x) * numFactor1 * numFactor2
        let numProd = try polyMulNTT(numFactor1, numFactor2, ntt: ntt)
        let transitionNum = try polyMulNTT(zLookupCoeffs, numProd, ntt: ntt)

        // s(omega*x)
        let sShifted = polyShift(sortedCoeffs, omega: omega)
        // gamma*(1+beta) + s(x) + beta*s(omega*x)
        let denFactor = polyAddCoeffs(
            polyAddCoeffs(gopbConst, sortedCoeffs),
            polyScaleCoeffs(sShifted, beta))
        // den^2
        let denSq = try polyMulNTT(denFactor, denFactor, ntt: ntt)

        // Z_lookup(omega*x) * den^2
        let zShifted = polyShift(zLookupCoeffs, omega: omega)
        let transitionDen = try polyMulNTT(zShifted, denSq, ntt: ntt)

        let transitionConstraint = polySubCoeffs(transitionNum, transitionDen)

        // --- Constraint 3: L_{n-1}(x) * (Z_lookup(x) - 1) = 0 ---
        var lnEvals = [Fr](repeating: Fr.zero, count: nn)
        lnEvals[nn - 1] = Fr.one
        let lnCoeffs = try ntt.intt(lnEvals)
        let finalConstraint = try polyMulNTT(lnCoeffs, zMinus1, ntt: ntt)

        // Combine with alpha powers
        let alpha2 = frSqr(alpha)
        var combined = initConstraint
        combined = polyAddCoeffs(combined, polyScaleCoeffs(transitionConstraint, alpha))
        combined = polyAddCoeffs(combined, polyScaleCoeffs(finalConstraint, alpha2))

        return combined
    }
}

// MARK: - Integration with LogUp/Lasso

/// Unified lookup dispatch that selects the optimal lookup engine based on table size
/// and structure. Integrates PlonkLookupArgument with the existing LogUp and Lasso engines.
public class PlonkLookupDispatch {

    /// Strategy for the lookup argument
    public enum Strategy {
        /// Plookup grand product (constant degree, best for small tables)
        case plookup
        /// LogUp sumcheck-based (best for medium unstructured tables)
        case logUp
        /// Lasso decomposition (best for large structured tables like range checks)
        case lasso
        /// Cached quotients (best for large unstructured tables, sublinear prover)
        case cachedQuotients
    }

    /// Automatically select the best strategy based on table characteristics.
    ///
    /// Heuristic:
    ///   - |T| <= 1024: Plookup (constant overhead, simple integration)
    ///   - |T| <= 2^20 and unstructured: LogUp
    ///   - |T| > 2^20 and structured (tensor decomposable): Lasso
    ///   - |T| > 2^20 and unstructured: cached quotients
    public static func selectStrategy(tableSize: Int, isStructured: Bool) -> Strategy {
        if tableSize <= 1024 {
            return .plookup
        } else if tableSize <= (1 << 20) {
            return .logUp
        } else if isStructured {
            return .lasso
        } else {
            return .cachedQuotients
        }
    }

    /// Compute the sorted vector and grand product for Plookup integration.
    /// Returns the Z_lookup evaluations needed for the Plonk proof.
    ///
    /// This is the bridge between the Plonk prover (PlonkProver.swift) and
    /// the lookup argument. The Z_lookup polynomial is committed alongside
    /// the permutation accumulator Z(x).
    public static func plookupGrandProduct(
        table: [Fr], lookupValues: [Fr], n: Int,
        beta: Fr, gamma: Fr
    ) -> (zLookupEvals: [Fr], sortedEvals: [Fr]) {
        let arg = PlonkLookupArgument(table: table, n: n)

        // Compute sorted vector
        let sortedEvals = arg.computeSortedVector(lookupValues: lookupValues)

        // Build table evaluations padded to domain size
        var tableEvals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<min(table.count, n) {
            tableEvals[i] = table[i]
        }
        // Pad remaining with last table value
        if table.count < n {
            let padVal = table.last!
            for i in table.count..<n {
                tableEvals[i] = padVal
            }
        }

        // Build lookup evaluations padded to domain size
        var lookupEvals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<min(lookupValues.count, n) {
            lookupEvals[i] = lookupValues[i]
        }
        // Pad with a valid table value (so the constraint is still satisfied)
        if lookupValues.count < n {
            let padVal = table[0]
            for i in lookupValues.count..<n {
                lookupEvals[i] = padVal
            }
        }

        // Compute grand product
        let zLookupEvals = arg.computeGrandProduct(
            lookupEvals: lookupEvals, tableEvals: tableEvals,
            sortedEvals: sortedEvals, beta: beta, gamma: gamma)

        return (zLookupEvals: zLookupEvals, sortedEvals: sortedEvals)
    }

    /// Bridge to LogUp engine: convert Plonk lookup gates to LogUp format.
    /// Extracts lookup values from the circuit witness and runs the LogUp proof.
    ///
    /// - Parameters:
    ///   - circuit: The Plonk circuit with lookup gates
    ///   - witness: Wire assignments
    ///   - lookupEngine: Existing LogUp engine instance
    ///   - beta: Challenge for the LogUp protocol
    /// - Returns: LogUp proof for all lookup gates in the circuit
    public static func logUpFromPlonk(
        circuit: PlonkCircuit, witness: [Fr],
        lookupEngine: LookupEngine, beta: Fr
    ) throws -> [LookupProof] {
        var proofs = [LookupProof]()

        for table in circuit.lookupTables {
            // Extract lookup values for this table
            var lookupValues = [Fr]()
            for (i, gate) in circuit.gates.enumerated() {
                if !frEqual(gate.qLookup, Fr.zero) {
                    // Check if this gate references this table
                    let tableIdInt = frToInt(gate.qC)
                    if tableIdInt.count > 0 && tableIdInt[0] == UInt64(table.id) {
                        let wireIdx = circuit.wireAssignments[i][0]
                        if wireIdx < witness.count {
                            lookupValues.append(witness[wireIdx])
                        }
                    }
                }
            }

            if lookupValues.isEmpty { continue }

            // Pad to power of 2
            let m = lookupValues.count
            var paddedM = 1
            while paddedM < m { paddedM <<= 1 }
            while lookupValues.count < paddedM {
                lookupValues.append(table.values[0])
            }

            // Pad table to power of 2
            var paddedTable = table.values
            var paddedT = 1
            while paddedT < paddedTable.count { paddedT <<= 1 }
            while paddedTable.count < paddedT {
                paddedTable.append(paddedTable.last!)
            }

            let proof = try lookupEngine.prove(
                table: paddedTable, lookups: lookupValues, beta: beta)
            proofs.append(proof)
        }

        return proofs
    }
}
