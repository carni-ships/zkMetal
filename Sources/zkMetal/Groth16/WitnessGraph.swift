// Witness dependency graph: DAG analysis for optimal constraint solving order
//
// Builds a dependency graph from R1CS constraints, performs topological sort
// for optimal solving order, detects circular dependencies, and groups
// independent constraints into layers for potential batch/parallel solving.

import Foundation

// MARK: - Witness Graph

public struct WitnessGraph {

    /// Edge: constraint index that connects variables
    public struct Edge {
        public let constraintIdx: Int
        public let variables: [Int]  // all variable indices involved in this constraint
    }

    /// Per-variable: which constraints reference this variable
    public let variableConstraints: [[Int]]
    /// Per-constraint: which variables it references (union of A, B, C terms)
    public let constraintVariables: [[Int]]
    /// Total number of variables
    public let numVars: Int
    /// Total number of constraints
    public let numConstraints: Int

    /// Build dependency graph from R1CSConstraintSet
    public init(constraintSet: R1CSConstraintSet) {
        self.numVars = constraintSet.numVars
        self.numConstraints = constraintSet.constraints.count

        var varToConstraints = [[Int]](repeating: [], count: constraintSet.numVars)
        var constraintToVars = [[Int]]()
        constraintToVars.reserveCapacity(constraintSet.constraints.count)

        for (i, constraint) in constraintSet.constraints.enumerated() {
            var vars = Set<Int>()
            for t in constraint.a { vars.insert(t.varIdx) }
            for t in constraint.b { vars.insert(t.varIdx) }
            for t in constraint.c { vars.insert(t.varIdx) }

            let varList = Array(vars).sorted()
            constraintToVars.append(varList)
            for v in varList {
                varToConstraints[v].append(i)
            }
        }

        self.variableConstraints = varToConstraints
        self.constraintVariables = constraintToVars
    }

    // MARK: - Topological Sort (constraint ordering)

    /// Compute a solving order for constraints given a set of initially known variables.
    /// Returns constraint indices in the order they should be attempted.
    ///
    /// Strategy: BFS from constraints whose unknowns are minimized.
    /// A constraint is "ready" when it has at most 1 unknown variable.
    public func topologicalOrder(knownVariables: Set<Int>) -> [Int] {
        var known = knownVariables
        // Always include wire 0
        known.insert(0)

        var unknownCount = [Int](repeating: 0, count: numConstraints)
        var processed = [Bool](repeating: false, count: numConstraints)

        // Count unknowns per constraint
        for i in 0..<numConstraints {
            var count = 0
            var seen = Set<Int>()
            for v in constraintVariables[i] {
                if !known.contains(v) && !seen.contains(v) {
                    count += 1
                    seen.insert(v)
                }
            }
            unknownCount[i] = count
        }

        var order = [Int]()
        order.reserveCapacity(numConstraints)
        var progress = true

        while progress {
            progress = false

            // Find constraints with <= 1 unknown (solvable)
            for i in 0..<numConstraints {
                if !processed[i] && unknownCount[i] <= 1 {
                    order.append(i)
                    processed[i] = true
                    progress = true

                    // Mark all variables in this constraint as known
                    for v in constraintVariables[i] {
                        if !known.contains(v) {
                            known.insert(v)
                            // Update unknown counts for other constraints referencing this var
                            for ci in variableConstraints[v] {
                                if !processed[ci] {
                                    unknownCount[ci] = max(0, unknownCount[ci] - 1)
                                }
                            }
                        }
                    }
                }
            }
        }

        // Append remaining unprocessed constraints (may be unsolvable)
        for i in 0..<numConstraints {
            if !processed[i] {
                order.append(i)
            }
        }

        return order
    }

    // MARK: - Circular Dependency Detection

    /// Detect groups of constraints that form circular dependencies (not solvable
    /// without additional hints). Returns arrays of constraint indices that form cycles.
    public func detectCircularDependencies(knownVariables: Set<Int>) -> [[Int]] {
        var known = knownVariables
        known.insert(0)

        var unknownCount = [Int](repeating: 0, count: numConstraints)
        var processed = [Bool](repeating: false, count: numConstraints)

        for i in 0..<numConstraints {
            var seen = Set<Int>()
            for v in constraintVariables[i] {
                if !known.contains(v) && !seen.contains(v) {
                    unknownCount[i] += 1
                    seen.insert(v)
                }
            }
        }

        // Propagate solvable constraints
        var progress = true
        while progress {
            progress = false
            for i in 0..<numConstraints {
                if !processed[i] && unknownCount[i] <= 1 {
                    processed[i] = true
                    progress = true
                    for v in constraintVariables[i] {
                        if !known.contains(v) {
                            known.insert(v)
                            for ci in variableConstraints[v] {
                                if !processed[ci] {
                                    unknownCount[ci] = max(0, unknownCount[ci] - 1)
                                }
                            }
                        }
                    }
                }
            }
        }

        // Remaining unprocessed constraints are in cycles
        // Group connected components via shared unknown variables
        var remaining = [Int]()
        for i in 0..<numConstraints {
            if !processed[i] { remaining.append(i) }
        }

        if remaining.isEmpty { return [] }

        // Union-Find to group connected constraints
        var parent = [Int: Int]()
        for i in remaining { parent[i] = i }

        func find(_ x: Int) -> Int {
            var r = x
            while parent[r] != r { r = parent[r]! }
            var c = x
            while c != r { let n = parent[c]!; parent[c] = r; c = n }
            return r
        }

        func union(_ a: Int, _ b: Int) {
            let ra = find(a), rb = find(b)
            if ra != rb { parent[ra] = rb }
        }

        // Connect constraints that share unknown variables
        var varToRemaining = [Int: Int]()  // var -> first remaining constraint using it
        for ci in remaining {
            for v in constraintVariables[ci] {
                if !known.contains(v) || !knownVariables.contains(v) {
                    if let prev = varToRemaining[v] {
                        union(ci, prev)
                    } else {
                        varToRemaining[v] = ci
                    }
                }
            }
        }

        // Group by root
        var groups = [Int: [Int]]()
        for ci in remaining {
            let root = find(ci)
            groups[root, default: []].append(ci)
        }

        return Array(groups.values)
    }

    // MARK: - Layer-Based Parallelism

    /// Group constraints into layers where all constraints within a layer are independent
    /// (can be solved in parallel). Layer i depends only on layers 0..<i.
    ///
    /// Returns arrays of constraint indices, one array per layer.
    public func parallelLayers(knownVariables: Set<Int>) -> [[Int]] {
        var known = knownVariables
        known.insert(0)

        var unknownCount = [Int](repeating: 0, count: numConstraints)
        var processed = [Bool](repeating: false, count: numConstraints)

        for i in 0..<numConstraints {
            var seen = Set<Int>()
            for v in constraintVariables[i] {
                if !known.contains(v) && !seen.contains(v) {
                    unknownCount[i] += 1
                    seen.insert(v)
                }
            }
        }

        var layers = [[Int]]()
        var progress = true

        while progress {
            progress = false
            var layer = [Int]()
            var newlyKnown = Set<Int>()

            // Collect all solvable constraints in this layer
            for i in 0..<numConstraints {
                if !processed[i] && unknownCount[i] <= 1 {
                    layer.append(i)
                    processed[i] = true
                    progress = true

                    // Gather newly solved variables
                    for v in constraintVariables[i] {
                        if !known.contains(v) {
                            newlyKnown.insert(v)
                        }
                    }
                }
            }

            if !layer.isEmpty {
                layers.append(layer)

                // Update counts for next layer
                for v in newlyKnown {
                    known.insert(v)
                    for ci in variableConstraints[v] {
                        if !processed[ci] {
                            unknownCount[ci] = max(0, unknownCount[ci] - 1)
                        }
                    }
                }
            }
        }

        return layers
    }

    // MARK: - Statistics

    /// Returns (solvableCount, unsolvableCount) given initial known variables
    public func solvabilityStats(knownVariables: Set<Int>) -> (solvable: Int, unsolvable: Int) {
        let order = topologicalOrder(knownVariables: knownVariables)
        var known = knownVariables
        known.insert(0)

        var solvable = 0
        for ci in order {
            var unknowns = 0
            for v in constraintVariables[ci] {
                if !known.contains(v) { unknowns += 1 }
            }
            if unknowns <= 1 {
                solvable += 1
                for v in constraintVariables[ci] { known.insert(v) }
            }
        }
        return (solvable, numConstraints - solvable)
    }
}
