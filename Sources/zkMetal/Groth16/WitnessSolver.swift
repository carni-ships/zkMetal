// Groth16 witness solver: computes full witness from R1CS constraints + partial inputs
//
// Given R1CS constraints A*B=C in sparse format and a partial witness (public inputs
// + known private inputs), iteratively solves for all remaining witness variables.
//
// Strategy:
//   1. Build per-constraint sparse representations
//   2. Forward propagation: for linear constraints (one side constant), solve directly
//   3. Back-substitution: iterate until all variables resolved or no progress
//   4. Detect unsolvable systems (circular deps without enough hints)

import Foundation

// MARK: - R1CS Constraint Set (sparse per-constraint representation)

/// Sparse term: coefficient * variable
public struct SparseTerm {
    public let varIdx: Int
    public let coeff: Fr
    public init(varIdx: Int, coeff: Fr) { self.varIdx = varIdx; self.coeff = coeff }
}

/// A single R1CS constraint in per-constraint sparse format: A . w * B . w = C . w
public struct SparseConstraint {
    public let a: [SparseTerm]
    public let b: [SparseTerm]
    public let c: [SparseTerm]
}

/// Stores R1CS A, B, C matrices in per-constraint sparse format + metadata
public struct R1CSConstraintSet {
    public let constraints: [SparseConstraint]
    public let numVars: Int
    public let numPublic: Int  // number of public inputs (not counting wire 0 = "one")

    /// Build from an R1CSInstance (flat sparse entries grouped by row)
    public init(from r1cs: R1CSInstance) {
        self.numVars = r1cs.numVars
        self.numPublic = r1cs.numPublic
        let n = r1cs.numConstraints

        // Group entries by row
        var aByRow = [[SparseTerm]](repeating: [], count: n)
        var bByRow = [[SparseTerm]](repeating: [], count: n)
        var cByRow = [[SparseTerm]](repeating: [], count: n)

        for e in r1cs.aEntries { aByRow[e.row].append(SparseTerm(varIdx: e.col, coeff: e.val)) }
        for e in r1cs.bEntries { bByRow[e.row].append(SparseTerm(varIdx: e.col, coeff: e.val)) }
        for e in r1cs.cEntries { cByRow[e.row].append(SparseTerm(varIdx: e.col, coeff: e.val)) }

        var result = [SparseConstraint]()
        result.reserveCapacity(n)
        for i in 0..<n {
            result.append(SparseConstraint(a: aByRow[i], b: bByRow[i], c: cByRow[i]))
        }
        self.constraints = result
    }

    /// Build from R1CSFileConstraint array (Circom parser output)
    public init(from fileConstraints: [R1CSFileConstraint], numVars: Int, numPublic: Int) {
        self.numVars = numVars
        self.numPublic = numPublic

        var result = [SparseConstraint]()
        result.reserveCapacity(fileConstraints.count)
        for fc in fileConstraints {
            let a = fc.a.terms.map { SparseTerm(varIdx: Int($0.wireId), coeff: $0.coeff) }
            let b = fc.b.terms.map { SparseTerm(varIdx: Int($0.wireId), coeff: $0.coeff) }
            let c = fc.c.terms.map { SparseTerm(varIdx: Int($0.wireId), coeff: $0.coeff) }
            result.append(SparseConstraint(a: a, b: b, c: c))
        }
        self.constraints = result
    }
}

// MARK: - Witness Solver Result

public struct WitnessSolverResult {
    /// The complete witness vector: [one, pub_1, ..., pub_n, priv_1, ..., internal_m]
    public let witness: [Fr]
    /// Number of iterations taken to solve
    public let iterations: Int
    /// Indices of variables that could not be solved (empty if fully solved)
    public let unsolvedIndices: [Int]
    /// Whether all variables were successfully resolved
    public var isFullySolved: Bool { unsolvedIndices.isEmpty }
}

// MARK: - Witness Solver

public struct WitnessSolver {

    /// Maximum iterations before giving up
    public var maxIterations: Int

    public init(maxIterations: Int = 1000) {
        self.maxIterations = maxIterations
    }

    /// Solve witness from R1CS constraints and partial inputs.
    ///
    /// - Parameters:
    ///   - constraintSet: The R1CS constraints in sparse per-constraint format
    ///   - knownValues: Dictionary mapping variable index -> known Fr value.
    ///                  Must include index 0 (the "one" wire) and all public inputs.
    /// - Returns: WitnessSolverResult with the full witness vector
    public func solve(constraintSet: R1CSConstraintSet,
                      knownValues: [Int: Fr]) -> WitnessSolverResult {
        let numVars = constraintSet.numVars
        var witness = [Fr](repeating: .zero, count: numVars)
        var solved = [Bool](repeating: false, count: numVars)

        // Initialize known values
        for (idx, val) in knownValues {
            guard idx >= 0 && idx < numVars else { continue }
            witness[idx] = val
            solved[idx] = true
        }

        // Ensure wire 0 = 1
        witness[0] = Fr.one
        solved[0] = true

        var iterations = 0
        var progress = true

        while progress && iterations < maxIterations {
            progress = false
            iterations += 1

            for constraint in constraintSet.constraints {
                // Try to solve this constraint: A.w * B.w = C.w
                if tryLinearSolve(constraint: constraint, witness: &witness,
                                  solved: &solved) {
                    progress = true
                }
            }
        }

        // Collect unsolved
        var unsolved = [Int]()
        for i in 0..<numVars {
            if !solved[i] { unsolved.append(i) }
        }

        return WitnessSolverResult(witness: witness, iterations: iterations,
                                   unsolvedIndices: unsolved)
    }

    /// Convenience: solve from R1CSInstance + public inputs + optional private input hints
    public func solve(r1cs: R1CSInstance,
                      publicInputs: [Fr],
                      privateHints: [Int: Fr] = [:]) -> WitnessSolverResult {
        let cs = R1CSConstraintSet(from: r1cs)
        var known = [Int: Fr]()
        // Wire 0 = 1
        known[0] = Fr.one
        // Public inputs are wires 1..numPublic
        for i in 0..<min(publicInputs.count, r1cs.numPublic) {
            known[1 + i] = publicInputs[i]
        }
        // Additional private hints
        for (k, v) in privateHints {
            known[k] = v
        }
        return solve(constraintSet: cs, knownValues: known)
    }

    // MARK: - Core Solving Logic

    /// Evaluate a linear combination sum(coeff_i * w_i) using known values.
    /// Returns (value, unknownCount, lastUnknownIdx, lastUnknownCoeff).
    /// If unknownCount == 0, value is the full evaluation.
    /// If unknownCount == 1, we can solve for the single unknown.
    private func evalLC(_ terms: [SparseTerm], _ witness: [Fr], _ solved: [Bool])
        -> (value: Fr, unknownCount: Int, unknownIdx: Int, unknownCoeff: Fr) {
        var value = Fr.zero
        var unknownCount = 0
        var unknownIdx = -1
        var unknownCoeff = Fr.zero

        for term in terms {
            if solved[term.varIdx] {
                value = frAdd(value, frMul(term.coeff, witness[term.varIdx]))
            } else {
                unknownCount += 1
                unknownIdx = term.varIdx
                unknownCoeff = term.coeff
            }
        }
        return (value, unknownCount, unknownIdx, unknownCoeff)
    }

    /// Try to solve a single constraint A.w * B.w = C.w for an unknown variable.
    /// Returns true if progress was made.
    private func tryLinearSolve(constraint: SparseConstraint,
                                witness: inout [Fr],
                                solved: inout [Bool]) -> Bool {
        let evalA = evalLC(constraint.a, witness, solved)
        let evalB = evalLC(constraint.b, witness, solved)
        let evalC = evalLC(constraint.c, witness, solved)

        // Case 1: A fully known, B fully known, C has exactly 1 unknown
        // a_val * b_val = c_known + coeff * w_unknown
        // => w_unknown = (a_val * b_val - c_known) / coeff
        if evalA.unknownCount == 0 && evalB.unknownCount == 0 && evalC.unknownCount == 1 {
            let ab = frMul(evalA.value, evalB.value)
            let diff = frSub(ab, evalC.value)
            if !evalC.unknownCoeff.isZero {
                let inv = frInverse(evalC.unknownCoeff)
                witness[evalC.unknownIdx] = frMul(diff, inv)
                solved[evalC.unknownIdx] = true
                return true
            }
        }

        // Case 2: A has 1 unknown, B fully known (nonzero), C fully known
        // (a_known + coeff * w_unknown) * b_val = c_val
        // => coeff * w_unknown * b_val = c_val - a_known * b_val
        // => w_unknown = (c_val - a_known * b_val) / (coeff * b_val)
        if evalA.unknownCount == 1 && evalB.unknownCount == 0 && evalC.unknownCount == 0 {
            let bVal = evalB.value
            if !bVal.isZero {
                let rhs = frSub(evalC.value, frMul(evalA.value, bVal))
                let denom = frMul(evalA.unknownCoeff, bVal)
                if !denom.isZero {
                    let inv = frInverse(denom)
                    witness[evalA.unknownIdx] = frMul(rhs, inv)
                    solved[evalA.unknownIdx] = true
                    return true
                }
            }
        }

        // Case 3: B has 1 unknown, A fully known (nonzero), C fully known
        // a_val * (b_known + coeff * w_unknown) = c_val
        // => coeff * w_unknown * a_val = c_val - a_val * b_known
        // => w_unknown = (c_val - a_val * b_known) / (coeff * a_val)
        if evalB.unknownCount == 1 && evalA.unknownCount == 0 && evalC.unknownCount == 0 {
            let aVal = evalA.value
            if !aVal.isZero {
                let rhs = frSub(evalC.value, frMul(aVal, evalB.value))
                let denom = frMul(evalB.unknownCoeff, aVal)
                if !denom.isZero {
                    let inv = frInverse(denom)
                    witness[evalB.unknownIdx] = frMul(rhs, inv)
                    solved[evalB.unknownIdx] = true
                    return true
                }
            }
        }

        // Case 4: A has 1 unknown, B has 1 unknown (same var), C fully known
        // This handles quadratic constraints like x*x = c where x is the single unknown
        // (a_known + ca * x) * (b_known + cb * x) = c_val
        // Only solvable if the known parts make it linear in x
        if evalA.unknownCount == 1 && evalB.unknownCount == 1 &&
           evalA.unknownIdx == evalB.unknownIdx && evalC.unknownCount == 0 {
            // (a_known + ca*x)(b_known + cb*x) = c_val
            // If a_known == 0 and b_known == 0: ca*cb*x^2 = c_val (quadratic, skip)
            // If one is zero: ca*x*b_known = c_val - a_known*b_known, solve linearly
            let aKnown = evalA.value
            let bKnown = evalB.value

            if aKnown.isZero && !bKnown.isZero {
                // ca*x * b_known = c_val
                let denom = frMul(evalA.unknownCoeff, bKnown)
                if !denom.isZero {
                    let inv = frInverse(denom)
                    witness[evalA.unknownIdx] = frMul(evalC.value, inv)
                    solved[evalA.unknownIdx] = true
                    return true
                }
            } else if bKnown.isZero && !aKnown.isZero {
                // a_known * cb*x = c_val
                let denom = frMul(evalB.unknownCoeff, aKnown)
                if !denom.isZero {
                    let inv = frInverse(denom)
                    witness[evalB.unknownIdx] = frMul(evalC.value, inv)
                    solved[evalB.unknownIdx] = true
                    return true
                }
            }
        }

        return false
    }
}
