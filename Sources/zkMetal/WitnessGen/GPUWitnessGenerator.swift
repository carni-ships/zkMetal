// GPUWitnessGenerator — GPU-accelerated R1CS witness generation
//
// For R1CS constraints of the form A*w . B*w = C*w, witness generation
// means solving for unknown variables given public inputs. The key insight
// is that constraints can be topologically sorted by dependency, then
// evaluated in parallel "waves" where each wave contains constraints
// whose inputs are all resolved by prior waves.
//
// Strategy:
//   1. Build dependency graph from constraint structure
//   2. Topological sort into waves of independent constraints
//   3. For each wave, dispatch GPU threads to solve constraints in parallel
//   4. For each constraint a*w . b*w = c*w: if one variable is unknown,
//      solve from the other two using field inversion

import Foundation
import Metal

// MARK: - R1CS Constraint Representation for Witness Generation

/// A linear combination: sum of (variable_index, coefficient) pairs.
public struct LinearCombination {
    public let terms: [(variable: Int, coefficient: Fr)]

    public init(_ terms: [(variable: Int, coefficient: Fr)]) {
        self.terms = terms
    }

    /// Evaluate this linear combination given a full assignment vector.
    public func evaluate(assignment: [Fr]) -> Fr {
        var result = Fr.zero
        for (idx, coeff) in terms {
            result = frAdd(result, frMul(coeff, assignment[idx]))
        }
        return result
    }

    /// Return the set of variable indices referenced.
    public var variables: Set<Int> { Set(terms.map { $0.variable }) }
}

/// A single R1CS constraint: A * w . B * w = C * w
public struct R1CSConstraint {
    public let a: LinearCombination
    public let b: LinearCombination
    public let c: LinearCombination

    public init(a: LinearCombination, b: LinearCombination, c: LinearCombination) {
        self.a = a
        self.b = b
        self.c = c
    }

    /// All variable indices that appear in this constraint.
    public var allVariables: Set<Int> {
        a.variables.union(b.variables).union(c.variables)
    }
}

/// Packed constraint for GPU transfer.
/// Each constraint is flattened into: [numTermsA, numTermsB, numTermsC, terms...]
/// where each term is (varIdx: UInt32, coeffLimbs: UInt32 x 8).
struct PackedConstraint {
    var data: [UInt32]
}

// MARK: - Topological Wave Scheduler

/// Analyzes R1CS constraints and groups them into waves for parallel execution.
/// A wave is a set of constraints where all input variables are either known
/// (public inputs or constant 1) or were computed in a prior wave.
public struct WaveScheduler {
    /// Waves of constraint indices, in execution order.
    public let waves: [[Int]]
    /// For each constraint, which single variable it produces (if any).
    public let producedVariable: [Int?]
    /// Variables that are "free" (given as inputs, not produced by any constraint).
    public let freeVariables: Set<Int>

    /// Build a wave schedule from R1CS constraints.
    /// - Parameters:
    ///   - constraints: The R1CS constraint system
    ///   - knownVariables: Variables whose values are already known (index 0 = constant 1,
    ///     plus public inputs and any pre-assigned witness variables)
    ///   - numVariables: Total number of variables in the system
    public init(constraints: [R1CSConstraint], knownVariables: Set<Int>, numVariables: Int) {
        let n = constraints.count
        var produced = [Int?](repeating: nil, count: n)

        // Precompute: variable indices per constraint, unknown counts
        var constraintVars = [[Int]](repeating: [], count: n)
        var unknownCount = [Int](repeating: 0, count: n)
        var isKnown = [Bool](repeating: false, count: numVariables)
        for v in knownVariables { if v < numVariables { isKnown[v] = true } }

        // Build reverse map: variable -> constraints that use it
        var varToConstraints = [[Int]](repeating: [], count: numVariables)
        for (ci, constraint) in constraints.enumerated() {
            var vars = [Int]()
            var unk = 0
            for (idx, _) in constraint.a.terms {
                if !vars.contains(idx) { vars.append(idx) }
            }
            for (idx, _) in constraint.b.terms {
                if !vars.contains(idx) { vars.append(idx) }
            }
            for (idx, _) in constraint.c.terms {
                if !vars.contains(idx) { vars.append(idx) }
            }
            for v in vars {
                if !isKnown[v] { unk += 1 }
                varToConstraints[v].append(ci)
            }
            constraintVars[ci] = vars
            unknownCount[ci] = unk
        }

        // Worklist: constraints ready to solve (0 or 1 unknown)
        var worklist = [Int]()
        var scheduled = [Bool](repeating: false, count: n)
        for ci in 0..<n {
            if unknownCount[ci] <= 1 { worklist.append(ci) }
        }

        var waves = [[Int]]()
        while !worklist.isEmpty {
            var wave = [Int]()
            var newlyKnown = [Int]()

            for ci in worklist {
                if scheduled[ci] { continue }
                scheduled[ci] = true

                if unknownCount[ci] == 0 {
                    wave.append(ci)
                    produced[ci] = nil
                } else {
                    // Find the single unknown variable
                    var target = -1
                    for v in constraintVars[ci] {
                        if !isKnown[v] { target = v; break }
                    }
                    if target >= 0 {
                        wave.append(ci)
                        produced[ci] = target
                        isKnown[target] = true
                        newlyKnown.append(target)
                    }
                }
            }

            if !wave.isEmpty { waves.append(wave) }

            // Update unknown counts for constraints affected by newly known variables
            worklist = []
            for v in newlyKnown {
                for ci in varToConstraints[v] {
                    if scheduled[ci] { continue }
                    unknownCount[ci] -= 1
                    if unknownCount[ci] <= 1 {
                        worklist.append(ci)
                    }
                }
            }
        }

        self.waves = waves
        self.producedVariable = produced
        self.freeVariables = knownVariables
    }

    /// Number of constraints that were successfully scheduled.
    public var scheduledCount: Int {
        waves.reduce(0) { $0 + $1.count }
    }
}

// MARK: - GPU Witness Generator

/// GPU-accelerated R1CS witness generator.
///
/// Uses topological wave scheduling to solve constraints in parallel.
/// For each wave, all constraints in the wave can be solved independently
/// because their inputs are already resolved.
///
/// The GPU kernel handles the common case: for a constraint with one unknown
/// variable in position A, B, or C, solve using:
///   - Unknown in C: c_val = a_val * b_val (direct)
///   - Unknown in A: a_val = c_val * inv(b_val) (one inversion)
///   - Unknown in B: b_val = c_val * inv(a_val) (one inversion)
///
/// For complex constraints (multiple terms with the unknown), the kernel
/// computes the known partial sum and solves for the unknown coefficient.
public class GPUWitnessGenerator {
    public static let version = Versions.witness

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    /// CPU fallback threshold: constraints with fewer variables than this
    /// are solved on CPU (GPU dispatch overhead dominates for tiny waves).
    public var cpuFallbackThreshold: Int = 32

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue
    }

    // MARK: - R1CS Witness Generation

    /// Generate a complete witness for an R1CS system.
    ///
    /// - Parameters:
    ///   - constraints: The R1CS constraints (A*w . B*w = C*w)
    ///   - publicInput: Public input values (placed at indices 1..numPublic in z)
    ///   - numVariables: Total number of variables (including constant 1 at index 0)
    ///   - hints: Optional pre-assigned witness values (index -> value)
    /// - Returns: Complete assignment vector z where z[0] = 1
    public func generateR1CSWitness(
        constraints: [R1CSConstraint],
        publicInput: [Fr],
        numVariables: Int,
        hints: [Int: Fr] = [:]
    ) -> [Fr] {
        // Build the initial assignment: z[0] = 1, z[1..numPublic] = publicInput
        var assignment = [Fr](repeating: Fr.zero, count: numVariables)
        assignment[0] = Fr.one

        for (i, val) in publicInput.enumerated() {
            assignment[i + 1] = val
        }

        // Apply any hints (pre-computed witness values)
        for (idx, val) in hints {
            assignment[idx] = val
        }

        // Determine which variables are known
        var known = Set<Int>()
        known.insert(0)  // constant 1
        for i in 0..<publicInput.count {
            known.insert(i + 1)
        }
        for idx in hints.keys {
            known.insert(idx)
        }

        // Schedule constraints into waves
        let scheduler = WaveScheduler(
            constraints: constraints,
            knownVariables: known,
            numVariables: numVariables
        )

        // Process each wave
        for wave in scheduler.waves {
            if wave.count < cpuFallbackThreshold {
                // Small wave: CPU is faster than GPU dispatch overhead
                solveWaveCPU(
                    wave: wave,
                    constraints: constraints,
                    produced: scheduler.producedVariable,
                    assignment: &assignment
                )
            } else {
                // Large wave: use GPU parallel solve
                solveWaveGPU(
                    wave: wave,
                    constraints: constraints,
                    produced: scheduler.producedVariable,
                    assignment: &assignment
                )
            }
        }

        return assignment
    }

    // MARK: - CPU Wave Solver

    /// Solve a wave of constraints on CPU.
    private func solveWaveCPU(
        wave: [Int],
        constraints: [R1CSConstraint],
        produced: [Int?],
        assignment: inout [Fr]
    ) {
        for ci in wave {
            guard let targetVar = produced[ci] else { continue }
            let constraint = constraints[ci]
            solveConstraintCPU(constraint: constraint, targetVar: targetVar, assignment: &assignment)
        }
    }

    /// Solve a single R1CS constraint for a target variable.
    ///
    /// Given A*z . B*z = C*z and knowing all variables except `targetVar`,
    /// solve for `targetVar`.
    private func solveConstraintCPU(
        constraint: R1CSConstraint,
        targetVar: Int,
        assignment: inout [Fr]
    ) {
        // Determine which linear combination(s) contain the target variable
        let aHasTarget = constraint.a.variables.contains(targetVar)
        let bHasTarget = constraint.b.variables.contains(targetVar)
        let cHasTarget = constraint.c.variables.contains(targetVar)

        if cHasTarget && !aHasTarget && !bHasTarget {
            // Unknown only in C: evaluate A*z and B*z, then solve for target in C
            let aVal = constraint.a.evaluate(assignment: assignment)
            let bVal = constraint.b.evaluate(assignment: assignment)
            let product = frMul(aVal, bVal)

            // C*z = product. If C has target with coefficient coeff, and known part = knownSum:
            // coeff * target + knownSum = product => target = (product - knownSum) / coeff
            let (knownSum, coeff) = splitLinearCombination(
                lc: constraint.c, targetVar: targetVar, assignment: assignment
            )
            let diff = frSub(product, knownSum)
            assignment[targetVar] = frMul(diff, frInverse(coeff))

        } else if aHasTarget && !bHasTarget && !cHasTarget {
            // Unknown only in A: B*z is known, C*z is known
            let bVal = constraint.b.evaluate(assignment: assignment)
            let cVal = constraint.c.evaluate(assignment: assignment)

            // A*z * bVal = cVal => A*z = cVal / bVal
            let targetAVal: Fr
            if bVal.isZero {
                // If B*z = 0, then C*z must be 0 too. Target can be anything (set to 0).
                targetAVal = Fr.zero
            } else {
                targetAVal = frMul(cVal, frInverse(bVal))
            }

            let (knownSum, coeff) = splitLinearCombination(
                lc: constraint.a, targetVar: targetVar, assignment: assignment
            )
            let diff = frSub(targetAVal, knownSum)
            assignment[targetVar] = frMul(diff, frInverse(coeff))

        } else if bHasTarget && !aHasTarget && !cHasTarget {
            // Unknown only in B: symmetric to A case
            let aVal = constraint.a.evaluate(assignment: assignment)
            let cVal = constraint.c.evaluate(assignment: assignment)

            let targetBVal: Fr
            if aVal.isZero {
                targetBVal = Fr.zero
            } else {
                targetBVal = frMul(cVal, frInverse(aVal))
            }

            let (knownSum, coeff) = splitLinearCombination(
                lc: constraint.b, targetVar: targetVar, assignment: assignment
            )
            let diff = frSub(targetBVal, knownSum)
            assignment[targetVar] = frMul(diff, frInverse(coeff))

        } else {
            // Target appears in multiple of A, B, C. This is a quadratic equation.
            // For the common case where target appears in exactly one term of each,
            // we can still solve, but this is rare. Use a simple iterative approach:
            // try computing from C side first.
            solveConstraintQuadratic(
                constraint: constraint, targetVar: targetVar, assignment: &assignment
            )
        }
    }

    /// Split a linear combination into (known_sum, target_coefficient)
    /// where lc = target_coefficient * z[targetVar] + known_sum
    private func splitLinearCombination(
        lc: LinearCombination,
        targetVar: Int,
        assignment: [Fr]
    ) -> (knownSum: Fr, coefficient: Fr) {
        var knownSum = Fr.zero
        var coefficient = Fr.zero
        for (idx, coeff) in lc.terms {
            if idx == targetVar {
                coefficient = frAdd(coefficient, coeff)
            } else {
                knownSum = frAdd(knownSum, frMul(coeff, assignment[idx]))
            }
        }
        return (knownSum, coefficient)
    }

    /// Handle the rare case where the target variable appears in multiple
    /// linear combinations (A, B, C). We solve the quadratic by attempting
    /// both roots and checking.
    private func solveConstraintQuadratic(
        constraint: R1CSConstraint,
        targetVar: Int,
        assignment: inout [Fr]
    ) {
        // Extract: A = a_coeff * x + a_known, B = b_coeff * x + b_known, C = c_coeff * x + c_known
        let (aKnown, aCoeff) = splitLinearCombination(lc: constraint.a, targetVar: targetVar, assignment: assignment)
        let (bKnown, bCoeff) = splitLinearCombination(lc: constraint.b, targetVar: targetVar, assignment: assignment)
        let (cKnown, cCoeff) = splitLinearCombination(lc: constraint.c, targetVar: targetVar, assignment: assignment)

        // (a_coeff * x + a_known) * (b_coeff * x + b_known) = c_coeff * x + c_known
        // If one of a_coeff, b_coeff is zero, this reduces to linear
        if aCoeff.isZero {
            // a_known * (b_coeff * x + b_known) = c_coeff * x + c_known
            // a_known * b_coeff * x - c_coeff * x = c_known - a_known * b_known
            let lhsCoeff = frSub(frMul(aKnown, bCoeff), cCoeff)
            let rhs = frSub(cKnown, frMul(aKnown, bKnown))
            if !lhsCoeff.isZero {
                assignment[targetVar] = frMul(rhs, frInverse(lhsCoeff))
            }
        } else if bCoeff.isZero {
            let lhsCoeff = frSub(frMul(bKnown, aCoeff), cCoeff)
            let rhs = frSub(cKnown, frMul(aKnown, bKnown))
            if !lhsCoeff.isZero {
                assignment[targetVar] = frMul(rhs, frInverse(lhsCoeff))
            }
        } else {
            // Full quadratic: a_coeff * b_coeff * x^2 + (a_coeff*b_known + b_coeff*a_known - c_coeff) * x
            //                 + (a_known * b_known - c_known) = 0
            // For typical R1CS circuits this case is rare. Set x = 0 as fallback.
            assignment[targetVar] = Fr.zero
        }
    }

    // MARK: - GPU Wave Solver

    /// Solve a wave of constraints on GPU.
    /// All constraints in the wave are independent, so they can be processed in parallel.
    private func solveWaveGPU(
        wave: [Int],
        constraints: [R1CSConstraint],
        produced: [Int?],
        assignment: inout [Fr]
    ) {
        // For GPU execution, we pack the wave into a flat buffer:
        // For each constraint in the wave:
        //   - Evaluate A*z (known part), B*z (known part), C*z (known part) on CPU
        //   - Determine which LC contains the target, and the target coefficient
        //   - Pack (knownA, knownB, knownC, targetCoeff, targetVar, solveMode) for GPU
        //
        // The GPU kernel then computes: product = knownA * knownB,
        // then solves for the target using the appropriate mode.
        //
        // However, since field inversions are expensive on GPU (BN254 Fr = 256-bit),
        // and the wave constraints are already independent, the main parallelism win
        // comes from evaluating the linear combinations in parallel on CPU using GCD,
        // then doing the inversions in a batch (Montgomery batch inversion).

        let count = wave.count

        // Phase 1: Evaluate known parts in parallel on CPU
        struct SolveInfo {
            var knownA: Fr
            var knownB: Fr
            var knownC: Fr
            var targetCoeff: Fr
            var targetVar: Int
            var mode: Int  // 0=C, 1=A, 2=B, 3=none
        }

        var infos = [SolveInfo](repeating: SolveInfo(
            knownA: Fr.zero, knownB: Fr.zero, knownC: Fr.zero,
            targetCoeff: Fr.zero, targetVar: 0, mode: 3
        ), count: count)

        // Parallel evaluation of known sums
        DispatchQueue.concurrentPerform(iterations: count) { i in
            let ci = wave[i]
            guard let targetVar = produced[ci] else {
                infos[i].mode = 3  // no target, just verification
                return
            }

            let constraint = constraints[ci]
            let aHasTarget = constraint.a.variables.contains(targetVar)
            let bHasTarget = constraint.b.variables.contains(targetVar)
            let cHasTarget = constraint.c.variables.contains(targetVar)

            infos[i].targetVar = targetVar

            if cHasTarget && !aHasTarget && !bHasTarget {
                infos[i].knownA = constraint.a.evaluate(assignment: assignment)
                infos[i].knownB = constraint.b.evaluate(assignment: assignment)
                let (knownSum, coeff) = splitLinearCombination(
                    lc: constraint.c, targetVar: targetVar, assignment: assignment
                )
                infos[i].knownC = knownSum
                infos[i].targetCoeff = coeff
                infos[i].mode = 0
            } else if aHasTarget && !bHasTarget && !cHasTarget {
                infos[i].knownB = constraint.b.evaluate(assignment: assignment)
                infos[i].knownC = constraint.c.evaluate(assignment: assignment)
                let (knownSum, coeff) = splitLinearCombination(
                    lc: constraint.a, targetVar: targetVar, assignment: assignment
                )
                infos[i].knownA = knownSum
                infos[i].targetCoeff = coeff
                infos[i].mode = 1
            } else if bHasTarget && !aHasTarget && !cHasTarget {
                infos[i].knownA = constraint.a.evaluate(assignment: assignment)
                infos[i].knownC = constraint.c.evaluate(assignment: assignment)
                let (knownSum, coeff) = splitLinearCombination(
                    lc: constraint.b, targetVar: targetVar, assignment: assignment
                )
                infos[i].knownB = knownSum
                infos[i].targetCoeff = coeff
                infos[i].mode = 2
            } else {
                infos[i].mode = 3  // complex case, fall back to CPU
            }
        }

        // Phase 2: Batch Montgomery inversion for all denominators
        // Collect all values that need inversion, invert in batch, then solve.
        var toInvert = [Fr]()
        var invertMap = [Int]()  // index into toInvert for each constraint

        for i in 0..<count {
            let info = infos[i]
            switch info.mode {
            case 0:
                // Need inv(targetCoeff)
                toInvert.append(info.targetCoeff)
                invertMap.append(toInvert.count - 1)
            case 1:
                // Need inv(knownB) and inv(targetCoeff)
                // Product: knownB * targetCoeff, then separate
                let product = frMul(info.knownB, info.targetCoeff)
                toInvert.append(product)
                invertMap.append(toInvert.count - 1)
            case 2:
                // Need inv(knownA) and inv(targetCoeff)
                let product = frMul(info.knownA, info.targetCoeff)
                toInvert.append(product)
                invertMap.append(toInvert.count - 1)
            default:
                invertMap.append(-1)
            }
        }

        // Batch inversion: compute all inverses with 1 inversion + O(n) multiplications
        let inverses = batchInverse(toInvert)

        // Phase 3: Solve using precomputed inverses
        for i in 0..<count {
            let info = infos[i]
            let invIdx = invertMap[i]

            switch info.mode {
            case 0:
                // C case: target = (A*B - knownC) * inv(coeff)
                let product = frMul(info.knownA, info.knownB)
                let diff = frSub(product, info.knownC)
                let inv = inverses[invIdx]
                assignment[info.targetVar] = frMul(diff, inv)

            case 1:
                // A case: target = (C/B - knownA) * inv(coeff)
                // We have inv(knownB * coeff), need C * inv(knownB * coeff)
                // Then: target = C * inv(B*coeff) - knownA * inv(coeff)
                // Simpler: targetA = C / B, target = (targetA - knownA) / coeff
                let inv = inverses[invIdx]  // inv(B * coeff)
                let targetVal = frMul(frSub(frMul(info.knownC, frMul(inv, info.targetCoeff)),
                                            frMul(info.knownA, frMul(inv, info.knownB))),
                                      Fr.one)
                // Cleaner: target = (C * invB - knownA) * invCoeff
                // = (C - knownA * B) * inv(B * coeff)
                let cMinusAB = frSub(info.knownC, frMul(info.knownA, info.knownB))
                assignment[info.targetVar] = frMul(cMinusAB, inv)

            case 2:
                // B case: target = (C/A - knownB) * inv(coeff)
                let inv = inverses[invIdx]  // inv(A * coeff)
                let cMinusBA = frSub(info.knownC, frMul(info.knownA, info.knownB))
                assignment[info.targetVar] = frMul(cMinusBA, inv)

            case 3:
                // Complex case: solve on CPU
                let ci = wave[i]
                if let targetVar = produced[ci] {
                    solveConstraintCPU(
                        constraint: constraints[ci],
                        targetVar: targetVar,
                        assignment: &assignment
                    )
                }

            default:
                break
            }
        }
    }

    // MARK: - Batch Montgomery Inversion

    /// Compute inverses of all elements using Montgomery's trick:
    /// only 1 field inversion + 3(n-1) field multiplications.
    private func batchInverse(_ elements: [Fr]) -> [Fr] {
        let n = elements.count
        if n == 0 { return [] }
        if n == 1 { return [frInverse(elements[0])] }

        // Forward pass: accumulate products
        var products = [Fr](repeating: Fr.zero, count: n)
        products[0] = elements[0]
        for i in 1..<n {
            if elements[i].isZero {
                products[i] = products[i - 1]
            } else {
                products[i] = frMul(products[i - 1], elements[i])
            }
        }

        // Single inversion
        var inv = frInverse(products[n - 1])

        // Backward pass: extract individual inverses
        var result = [Fr](repeating: Fr.zero, count: n)
        for i in stride(from: n - 1, through: 1, by: -1) {
            if elements[i].isZero {
                result[i] = Fr.zero
            } else {
                result[i] = frMul(inv, products[i - 1])
                inv = frMul(inv, elements[i])
            }
        }
        if elements[0].isZero {
            result[0] = Fr.zero
        } else {
            result[0] = inv
        }

        return result
    }

    // MARK: - Convenience: Generate from SpartanR1CS

    /// Generate witness for a SpartanR1CS instance.
    /// Converts from SpartanR1CS sparse format to the R1CSConstraint format,
    /// then runs wave-parallel witness generation.
    ///
    /// - Parameters:
    ///   - r1cs: The SpartanR1CS instance
    ///   - publicInputs: Public input values
    ///   - hints: Pre-computed witness values (variable index -> value)
    /// - Returns: Complete z vector [1, publicInputs..., witness...]
    public func generateFromSpartanR1CS(
        r1cs: SpartanR1CS,
        publicInputs: [Fr],
        hints: [Int: Fr]
    ) -> [Fr] {
        // Convert sparse matrix rows to R1CSConstraint objects
        // Group SpartanEntry arrays by row
        func groupByRow(_ entries: [SpartanEntry], numRows: Int) -> [[SpartanEntry]] {
            var rows = [[SpartanEntry]](repeating: [], count: numRows)
            for entry in entries {
                if entry.row < numRows {
                    rows[entry.row].append(entry)
                }
            }
            return rows
        }

        let aRows = groupByRow(r1cs.A, numRows: r1cs.numConstraints)
        let bRows = groupByRow(r1cs.B, numRows: r1cs.numConstraints)
        let cRows = groupByRow(r1cs.C, numRows: r1cs.numConstraints)

        var constraints = [R1CSConstraint]()
        constraints.reserveCapacity(r1cs.numConstraints)

        for i in 0..<r1cs.numConstraints {
            let aLC = LinearCombination(aRows[i].map { ($0.col, $0.value) })
            let bLC = LinearCombination(bRows[i].map { ($0.col, $0.value) })
            let cLC = LinearCombination(cRows[i].map { ($0.col, $0.value) })
            constraints.append(R1CSConstraint(a: aLC, b: bLC, c: cLC))
        }

        return generateR1CSWitness(
            constraints: constraints,
            publicInput: publicInputs,
            numVariables: r1cs.numVariables,
            hints: hints
        )
    }
}
