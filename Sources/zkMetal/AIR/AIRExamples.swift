// AIR Examples — demonstrations of the AIR constraint DSL
//
// These examples show how to define complete STARKs using the fluent DSL,
// from simple Fibonacci sequences to cryptographic hash constraints.
// Each compiles to a CircleAIR compatible with the Circle STARK prover.

import Foundation

// MARK: - Fibonacci AIR (DSL version)

/// Build a Fibonacci AIR using the constraint DSL.
///
/// Trace layout:
///   Column "a": first element of each pair
///   Column "b": second element of each pair
///
/// Transition constraints:
///   a' = b        (next row's a equals current b)
///   b' = a + b    (next row's b equals current sum)
///
/// Boundary constraints:
///   a[0] = a0     (initial value)
///   b[0] = b0     (initial value)
public func buildFibonacciAIR(logTraceLength: Int,
                               a0: M31 = M31.one,
                               b0: M31 = M31.one) throws -> CompiledAIR {
    let builder = AIRBuilder(logTraceLength: logTraceLength)

    builder.addColumn(name: "a")
    builder.addColumn(name: "b")
    builder.addPublicInput(name: "a0", value: a0)
    builder.addPublicInput(name: "b0", value: b0)

    // Boundary: a[0] == a0, b[0] == b0
    builder.boundary(at: .first) { ctx in ctx.col("a") - ctx.pub("a0") }
    builder.boundary(at: .first) { ctx in ctx.col("b") - ctx.pub("b0") }

    // Transition: a' = b, b' = a + b
    builder.transition { ctx in ctx.next("a") - ctx.col("b") }
    builder.transition { ctx in ctx.next("b") - (ctx.col("a") + ctx.col("b")) }

    // Trace generator
    builder.traceGen { inputs in
        let n = 1 << logTraceLength
        let initA = inputs["a0"]!
        let initB = inputs["b0"]!
        var colA = [M31](repeating: M31.zero, count: n)
        var colB = [M31](repeating: M31.zero, count: n)
        colA[0] = initA
        colB[0] = initB
        for i in 1..<n {
            colA[i] = colB[i - 1]
            colB[i] = m31Add(colA[i - 1], colB[i - 1])
        }
        return [colA, colB]
    }

    return try builder.compile()
}

// MARK: - Rescue Hash AIR

/// Build a Rescue-Prime-like hash AIR using the constraint DSL.
///
/// Rescue-Prime is an algebraic hash function designed for efficient STARK proving.
/// It operates on a state of `width` field elements, applying alternating
/// S-box layers (x^alpha and x^(1/alpha)) with round constant additions.
///
/// Simplified version: single S-box x^3 (cube) with additive round constants.
///
/// Trace layout:
///   Columns "s0", "s1", ..., "s{width-1}": state elements
///   Periodic column "rc0", "rc1", ...: round constants (repeating per round)
///
/// Transition constraints (per state element i):
///   s_i' = (s_i + rc_i)^3    [simplified Rescue round]
///
/// Boundary constraints:
///   s_i[0] = input_i          (hash input)
public func buildRescueHashAIR(logTraceLength: Int,
                                width: Int = 4,
                                input: [M31]? = nil) throws -> CompiledAIR {
    precondition(width >= 2, "Rescue needs at least 2 state elements")
    let n = 1 << logTraceLength

    let builder = AIRBuilder(logTraceLength: logTraceLength)

    // State columns
    for i in 0..<width {
        builder.addColumn(name: "s\(i)")
    }

    // Public inputs: hash input values
    let hashInput = input ?? (0..<width).map { M31(v: UInt32($0 + 1)) }
    precondition(hashInput.count == width, "Input must have \(width) elements")
    for i in 0..<width {
        builder.addPublicInput(name: "in\(i)", value: hashInput[i])
    }

    // Round constants (periodic) — simple deterministic constants for demonstration
    // In a real Rescue implementation, these come from a secure CSPRNG seeded by nothing-up-my-sleeve values
    var roundConstants = [[M31]]()
    for i in 0..<width {
        // Generate a period's worth of round constants
        // Period = trace length (each row gets unique constants, repeating if trace > period)
        let period = min(n, 16) // 16 rounds max, then repeat
        var rc = [M31]()
        for r in 0..<period {
            // Deterministic "random-looking" constants: (i+1) * (r+1) * 7 + 3 mod p
            let val = UInt32((i + 1) * (r + 1) * 7 + 3) % M31.P
            rc.append(M31(v: val))
        }
        builder.addPeriodicColumn(name: "rc\(i)", values: rc)
        roundConstants.append(rc)
    }

    // Boundary: initial state = input
    for i in 0..<width {
        builder.boundary(at: .first) { ctx in ctx.col("s\(i)") - ctx.pub("in\(i)") }
    }

    // Transition: s_i' = (s_i + rc_i)^3  (simplified Rescue S-box)
    for i in 0..<width {
        builder.transition { ctx in
            let sWithRC = ctx.col("s\(i)") + ctx.periodic("rc\(i)")
            let cubed = sWithRC ** 3
            return ctx.next("s\(i)") - cubed
        }
    }

    // Trace generator
    let capturedWidth = width
    let capturedRC = roundConstants
    builder.traceGen { inputs in
        let traceLen = 1 << logTraceLength
        var state = (0..<capturedWidth).map { inputs["in\($0)"]! }
        var columns = (0..<capturedWidth).map { _ in [M31](repeating: M31.zero, count: traceLen) }

        // Write initial state
        for i in 0..<capturedWidth {
            columns[i][0] = state[i]
        }

        // Apply Rescue rounds
        for row in 1..<traceLen {
            for i in 0..<capturedWidth {
                let rcIdx = (row - 1) % capturedRC[i].count
                let withRC = m31Add(state[i], capturedRC[i][rcIdx])
                state[i] = m31Mul(m31Mul(withRC, withRC), withRC) // x^3
            }
            for i in 0..<capturedWidth {
                columns[i][row] = state[i]
            }
        }

        return columns
    }

    return try builder.compile()
}

// MARK: - Range Check AIR (DSL version)

/// Build a range check AIR using the constraint DSL.
///
/// Proves that all values in a sorted list are in [0, bound).
/// Uses the sorted-value approach: the trace contains the values in sorted order,
/// and the transition constraint enforces monotonicity.
///
/// Trace layout:
///   Column "val": the sorted values
///
/// Transition constraint:
///   diff = val' - val
///   diff * (bound - diff) must be non-negative (meaning diff is in [0, bound))
///
/// Boundary constraint:
///   val[0] = smallest value
public func buildRangeCheckAIR(logTraceLength: Int,
                                values: [M31],
                                bound: UInt32 = 65536) throws -> CompiledAIR {
    let n = 1 << logTraceLength
    precondition(values.count <= n, "Too many values for trace length")

    let builder = AIRBuilder(logTraceLength: logTraceLength)

    builder.addColumn(name: "val")

    // Sort values for the boundary constraint
    let sorted = values.sorted { $0.v < $1.v }
    builder.addPublicInput(name: "min_val", value: sorted.first ?? M31.zero)

    // Boundary: first value is the minimum
    builder.boundary(at: .first) { ctx in ctx.col("val") - ctx.pub("min_val") }

    // Transition: diff * (bound - diff) >= 0, i.e., diff is in [0, bound)
    // where diff = val' - val
    builder.transition { ctx in
        let diff = ctx.next("val") - ctx.col("val")
        let boundExpr = AIRExpression.constant(M31(v: bound))
        return diff * (boundExpr - diff)
    }

    // Trace generator
    let capturedValues = values
    builder.traceGen { inputs in
        let traceLen = 1 << logTraceLength
        var s = capturedValues.sorted { $0.v < $1.v }
        // Pad with the last value
        while s.count < traceLen {
            s.append(s.last ?? M31.zero)
        }
        return [s]
    }

    return try builder.compile()
}

// MARK: - Collatz Sequence AIR

/// Build a Collatz (3n+1) sequence AIR using the constraint DSL.
///
/// The Collatz conjecture states that starting from any positive integer,
/// repeatedly applying n -> n/2 (if even) or n -> 3n+1 (if odd) eventually
/// reaches 1. This AIR proves a valid Collatz sequence execution.
///
/// Trace layout:
///   Column "val": the sequence value at each step
///   Column "is_odd": 1 if val is odd, 0 if even (auxiliary witness)
///   Column "half": val/2 when even, (3*val+1)/2 when odd (auxiliary)
///
/// Transition constraints:
///   is_odd * (is_odd - 1) = 0          (boolean constraint)
///   is_odd * (val' - 3*val - 1) + (1 - is_odd) * (2*val' - val) = 0
///
/// Boundary:
///   val[0] = start_value
public func buildCollatzAIR(logTraceLength: Int,
                             startValue: UInt32) throws -> CompiledAIR {
    let n = 1 << logTraceLength

    let builder = AIRBuilder(logTraceLength: logTraceLength)

    builder.addColumn(name: "val")
    builder.addColumn(name: "is_odd")

    builder.addPublicInput(name: "start", value: M31(v: startValue))

    // Boundary: val[0] == start
    builder.boundary(at: .first) { ctx in ctx.col("val") - ctx.pub("start") }

    // Transition 1: is_odd is boolean
    builder.transition { ctx in
        ctx.col("is_odd") * (ctx.col("is_odd") - 1)
    }

    // Transition 2: combines the two Collatz rules into one constraint:
    //   if odd:  val' = 3*val + 1   =>  val' - 3*val - 1 = 0
    //   if even: val' = val/2        =>  2*val' - val = 0
    // Combined: is_odd * (val' - 3*val - 1) + (1 - is_odd) * (2*val' - val) = 0
    builder.transition { ctx in
        let v = ctx.col("val")
        let vNext = ctx.next("val")
        let odd = ctx.col("is_odd")
        let one = AIRExpression.constant(M31.one)
        let oddConstraint = vNext - v * 3 - one       // val' - 3*val - 1
        let evenConstraint = vNext * 2 - v             // 2*val' - val
        return odd * oddConstraint + (one - odd) * evenConstraint
    }

    // Trace generator
    builder.traceGen { inputs in
        let traceLen = 1 << logTraceLength
        let start = inputs["start"]!.v
        var vals = [M31](repeating: M31.zero, count: traceLen)
        var odds = [M31](repeating: M31.zero, count: traceLen)

        var current = UInt64(start)
        for i in 0..<traceLen {
            vals[i] = M31(v: UInt32(current % UInt64(M31.P)))
            odds[i] = M31(v: UInt32(current & 1))
            if current & 1 == 1 {
                current = 3 * current + 1
            } else {
                current = current / 2
            }
        }
        return [vals, odds]
    }

    return try builder.compile()
}
