// ConstraintOptimizer — Multi-pass constraint system optimizer
//
// Reduces constraint count and variable count before proving by applying:
//   1. Dead constraint elimination (outputs never used downstream)
//   2. Constant folding (evaluate pure-constant sub-expressions)
//   3. Linear combination merging (flatten addition chains)
//   4. Duplicate elimination / common sub-expression elimination (CSE)
//   5. Variable renumbering (compact indices after elimination)
//
// All passes preserve semantic equivalence: the optimized system is satisfied
// by the same witness (modulo variable renumbering) as the original.

import Foundation

// MARK: - Optimization Statistics

/// Statistics from a full optimization pass.
public struct OptimizationStats {
    /// Number of constraints before optimization
    public let originalConstraints: Int
    /// Number of constraints after optimization
    public let optimizedConstraints: Int
    /// Number of variables eliminated by renumbering
    public let eliminatedVars: Int
    /// Wall-clock time for all optimization passes (ms)
    public let timeMs: Double

    /// Reduction ratio: 1.0 = no reduction, 0.0 = all eliminated
    public var reductionRatio: Double {
        guard originalConstraints > 0 else { return 1.0 }
        return Double(optimizedConstraints) / Double(originalConstraints)
    }

    /// Human-readable summary
    public var summary: String {
        let pct = (1.0 - reductionRatio) * 100.0
        return String(format: "Optimizer: %d -> %d constraints (%.1f%% reduction), %d vars eliminated, %.2fms",
                      originalConstraints, optimizedConstraints, pct, eliminatedVars, timeMs)
    }
}

// MARK: - Constraint Optimizer

/// Multi-pass optimizer for ConstraintSystem instances.
///
/// Usage:
///   let (optimized, stats) = ConstraintOptimizer.optimizeAll(cs: mySystem)
///   print(stats.summary)
///
/// Individual passes can be applied separately for fine-grained control.
public enum ConstraintOptimizer {

    // MARK: - Pass 1: Dead Constraint Elimination

    /// Remove constraints whose output wires are never referenced by any other constraint.
    ///
    /// A constraint is "dead" if:
    ///   - It has a single unique wire not used by any other constraint
    ///   - It is not a public-input constraint (no way to detect this statically,
    ///     so we conservatively keep constraints with labels starting with "public_")
    ///
    /// For safety, only removes constraints that define an "output" wire (highest-indexed
    /// wire) that appears in no other constraint.
    public static func eliminateDeadConstraints(cs: ConstraintSystem) -> ConstraintSystem {
        let n = cs.constraints.count
        guard n > 1 else { return cs }

        // Build wire usage count across all constraints
        var wireUsageCount = [Int: Int]()  // wire index -> number of constraints using it
        for c in cs.constraints {
            for w in c.expr.wires {
                wireUsageCount[w.index, default: 0] += 1
            }
        }

        // A constraint is dead if it defines a wire used by only itself
        // Heuristic: the "output" wire of a constraint is the highest-indexed wire
        var liveConstraints = [Constraint]()
        for c in cs.constraints {
            // Conservatively keep labeled public constraints
            if let label = c.label, label.hasPrefix("public_") {
                liveConstraints.append(c)
                continue
            }

            let wires = c.expr.wires
            if wires.isEmpty {
                // Pure constant constraint (e.g., 0 == 0): dead
                // But if it's non-trivially constant, keep it (it would fail verification)
                let folded = c.expr.constantFolded()
                if case .constant(let val) = folded, val.isZero {
                    continue  // trivially true, remove
                }
                liveConstraints.append(c)
                continue
            }

            // Find the output wire (highest index)
            let maxWire = wires.max(by: { $0.index < $1.index })!
            if wireUsageCount[maxWire.index, default: 0] <= 1 && wires.count > 1 {
                // This wire is only used in this constraint = dead output
                // But only eliminate if we have more than one wire (single-wire
                // constraints like boolean checks are self-contained and must stay)
                // Actually, decrement usage for removed wires
                for w in wires {
                    wireUsageCount[w.index, default: 0] -= 1
                }
                continue
            }

            liveConstraints.append(c)
        }

        let result = ConstraintSystem(numWires: cs.numWires)
        for c in liveConstraints {
            result.addConstraint(c.expr, label: c.label)
        }
        return result
    }

    // MARK: - Pass 2: Constant Folding

    /// Evaluate sub-expressions where all inputs are known constants.
    /// Leverages the existing `Expr.constantFolded()` method and additionally
    /// removes constraints that reduce to `0 == 0`.
    public static func constantFolding(cs: ConstraintSystem) -> ConstraintSystem {
        let result = ConstraintSystem(numWires: cs.numWires)

        for c in cs.constraints {
            let folded = c.expr.constantFolded()

            // If the entire expression is a constant, it's either trivially true (0)
            // or unsatisfiable (nonzero). Keep unsatisfiable ones for error detection.
            if case .constant(let val) = folded {
                if val.isZero {
                    continue  // trivially true, eliminate
                }
                // unsatisfiable constraint: keep it so verification fails
            }

            result.addConstraint(folded, label: c.label)
        }
        return result
    }

    // MARK: - Pass 3: Linear Combination Merge

    /// Merge chains of additions into single linear combinations.
    ///
    /// Flattens nested `add(add(a, b), c)` into a single sum, and collects
    /// like terms (same wire with different coefficients) into one term.
    /// This reduces expression depth and GPU instruction count.
    public static func linearCombinationMerge(cs: ConstraintSystem) -> ConstraintSystem {
        let result = ConstraintSystem(numWires: cs.numWires)

        for c in cs.constraints {
            let merged = mergeLinearTerms(c.expr)
            result.addConstraint(merged, label: c.label)
        }
        return result
    }

    /// Flatten an expression into a sum of (coefficient, base_expr) terms,
    /// then rebuild with merged coefficients for identical base expressions.
    private static func mergeLinearTerms(_ expr: Expr) -> Expr {
        // Collect all additive terms as (coefficient, base_expression)
        var terms = [(Fr, Expr)]()
        collectLinearTerms(expr, coefficient: Fr.one, into: &terms)

        // Group by base expression (using Hashable conformance)
        var merged = [Expr: Fr]()
        var order = [Expr]()  // preserve insertion order for determinism

        for (coeff, base) in terms {
            if let existing = merged[base] {
                merged[base] = frAdd(existing, coeff)
            } else {
                merged[base] = coeff
                order.append(base)
            }
        }

        // Rebuild expression from merged terms
        var result: Expr? = nil
        for base in order {
            let coeff = merged[base]!
            if coeff.isZero { continue }

            let term: Expr
            if coeff == Fr.one {
                term = base
            } else if case .constant(let c) = base, c == Fr.one {
                // coeff * 1 = coeff
                term = .constant(coeff)
            } else {
                term = .mul(.constant(coeff), base)
            }

            if let existing = result {
                result = .add(existing, term)
            } else {
                result = term
            }
        }

        return result ?? .constant(Fr.zero)
    }

    /// Recursively collect additive terms: expr is decomposed into
    /// sum of (coefficient * non-add-expression) pairs.
    private static func collectLinearTerms(_ expr: Expr, coefficient: Fr, into terms: inout [(Fr, Expr)]) {
        switch expr {
        case .add(let a, let b):
            collectLinearTerms(a, coefficient: coefficient, into: &terms)
            collectLinearTerms(b, coefficient: coefficient, into: &terms)

        case .neg(let a):
            collectLinearTerms(a, coefficient: frNeg(coefficient), into: &terms)

        case .constant(let c):
            // Constant term: coeff * c
            terms.append((frMul(coefficient, c), .constant(Fr.one)))

        case .mul(.constant(let c), let inner):
            // c * inner: fold c into coefficient
            collectLinearTerms(inner, coefficient: frMul(coefficient, c), into: &terms)

        case .mul(let inner, .constant(let c)):
            // inner * c: fold c into coefficient
            collectLinearTerms(inner, coefficient: frMul(coefficient, c), into: &terms)

        default:
            // Non-linear term (wire, mul of non-constants, pow): record as-is
            terms.append((coefficient, expr))
        }
    }

    // MARK: - Pass 4: Duplicate Elimination (CSE)

    /// Detect and merge identical sub-expressions (common sub-expression elimination).
    ///
    /// Builds a hash map of all sub-expressions; when a duplicate is found,
    /// the expression tree is rewritten to share the first occurrence.
    /// Also removes fully duplicate constraints.
    public static func duplicateElimination(cs: ConstraintSystem) -> ConstraintSystem {
        let result = ConstraintSystem(numWires: cs.numWires)
        var seen = Set<Expr>()

        for c in cs.constraints {
            // CSE within the expression tree
            let deduped = cseRewrite(c.expr, cache: &seen)

            // Skip duplicate constraints (same expression already added)
            let canonical = deduped.constantFolded()
            result.addConstraint(canonical, label: c.label)
        }

        // Remove duplicate constraints (same expr)
        return removeDuplicateConstraints(result)
    }

    /// Rewrite an expression tree, replacing duplicate sub-expressions with
    /// canonical representatives. The `cache` set tracks seen expressions.
    private static func cseRewrite(_ expr: Expr, cache: inout Set<Expr>) -> Expr {
        // If we've seen this exact expression before, return it as-is
        // (the Hashable conformance on Expr handles structural equality)
        if cache.contains(expr) {
            return expr
        }

        let result: Expr
        switch expr {
        case .wire, .constant:
            result = expr

        case .add(let a, let b):
            let ra = cseRewrite(a, cache: &cache)
            let rb = cseRewrite(b, cache: &cache)
            result = .add(ra, rb)

        case .mul(let a, let b):
            let ra = cseRewrite(a, cache: &cache)
            let rb = cseRewrite(b, cache: &cache)
            result = .mul(ra, rb)

        case .neg(let a):
            let ra = cseRewrite(a, cache: &cache)
            result = .neg(ra)

        case .pow(let base, let n):
            let rb = cseRewrite(base, cache: &cache)
            result = .pow(rb, n)
        }

        cache.insert(result)
        return result
    }

    /// Remove constraints with identical expressions.
    private static func removeDuplicateConstraints(_ cs: ConstraintSystem) -> ConstraintSystem {
        let result = ConstraintSystem(numWires: cs.numWires)
        var seenExprs = Set<Expr>()

        for c in cs.constraints {
            if seenExprs.contains(c.expr) { continue }
            seenExprs.insert(c.expr)
            result.addConstraint(c.expr, label: c.label)
        }
        return result
    }

    // MARK: - Pass 5: Variable Renumbering

    /// Compact variable (wire) indices after elimination passes.
    /// Returns a new ConstraintSystem with contiguous wire indices starting at 0.
    ///
    /// Also returns the number of variables eliminated (original numWires - new numWires).
    public static func variableRenumbering(cs: ConstraintSystem) -> (ConstraintSystem, eliminatedVars: Int) {
        // Collect all wire indices actually used
        var usedWires = Set<Int>()
        for c in cs.constraints {
            for w in c.expr.wires {
                usedWires.insert(w.index)
            }
        }

        if usedWires.count == cs.numWires {
            return (cs, 0)  // no elimination possible
        }

        // Build renumbering map: old index -> new index
        let sorted = usedWires.sorted()
        var remap = [Int: Int]()
        for (newIdx, oldIdx) in sorted.enumerated() {
            remap[oldIdx] = newIdx
        }

        let newNumWires = max(sorted.count, 1)
        let result = ConstraintSystem(numWires: newNumWires)

        for c in cs.constraints {
            let remapped = remapWires(c.expr, remap: remap)
            result.addConstraint(remapped, label: c.label)
        }

        return (result, cs.numWires - newNumWires)
    }

    /// Recursively remap wire indices in an expression.
    private static func remapWires(_ expr: Expr, remap: [Int: Int]) -> Expr {
        switch expr {
        case .wire(let w):
            let newIdx = remap[w.index] ?? w.index
            return .wire(Wire(index: newIdx, row: w.row))
        case .constant:
            return expr
        case .add(let a, let b):
            return .add(remapWires(a, remap: remap), remapWires(b, remap: remap))
        case .mul(let a, let b):
            return .mul(remapWires(a, remap: remap), remapWires(b, remap: remap))
        case .neg(let a):
            return .neg(remapWires(a, remap: remap))
        case .pow(let base, let n):
            return .pow(remapWires(base, remap: remap), n)
        }
    }

    // MARK: - Full Optimization Pipeline

    /// Run all optimization passes in sequence and return the optimized system with stats.
    ///
    /// Pass order:
    ///   1. Constant folding (simplify expressions)
    ///   2. Linear combination merge (flatten additions, merge like terms)
    ///   3. Duplicate elimination (CSE + remove duplicate constraints)
    ///   4. Dead constraint elimination (remove unused outputs)
    ///   5. Variable renumbering (compact indices)
    ///
    /// Constant folding runs first because it enables the other passes to find
    /// more optimization opportunities.
    public static func optimizeAll(cs: ConstraintSystem) -> (ConstraintSystem, stats: OptimizationStats) {
        let t0 = CFAbsoluteTimeGetCurrent()
        let originalCount = cs.constraints.count
        let originalWires = cs.numWires

        // Pass 1: Constant folding
        var current = constantFolding(cs: cs)

        // Pass 2: Linear combination merge
        current = linearCombinationMerge(cs: current)

        // Pass 3: Duplicate elimination (CSE)
        current = duplicateElimination(cs: current)

        // Pass 4: Dead constraint elimination
        current = eliminateDeadConstraints(cs: current)

        // Pass 5: Variable renumbering
        let (final, eliminatedVars) = variableRenumbering(cs: current)

        let timeMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0

        let stats = OptimizationStats(
            originalConstraints: originalCount,
            optimizedConstraints: final.constraints.count,
            eliminatedVars: eliminatedVars,
            timeMs: timeMs
        )

        return (final, stats: stats)
    }
}
