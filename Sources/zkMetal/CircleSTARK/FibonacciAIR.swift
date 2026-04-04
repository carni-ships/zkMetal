// Fibonacci AIR — Simplest demonstration of Circle STARK proving
// Trace: 2 columns (a, b), transition: a' = b, b' = a + b
// Boundary: a[0] = 1, b[0] = 1

import Foundation

public struct FibonacciAIR: CircleAIR {
    public let logTraceLength: Int
    public let numColumns: Int = 2
    public let numConstraints: Int = 2
    public let constraintDegrees: [Int] = [1, 1]  // linear constraints

    /// Initial values for the Fibonacci sequence
    public let a0: M31
    public let b0: M31

    public var boundaryConstraints: [(column: Int, row: Int, value: M31)] {
        [(0, 0, a0), (1, 0, b0)]
    }

    public init(logTraceLength: Int, a0: M31 = M31.one, b0: M31 = M31.one) {
        precondition(logTraceLength >= 2, "Need at least 4 rows for Fibonacci")
        self.logTraceLength = logTraceLength
        self.a0 = a0
        self.b0 = b0
    }

    public func generateTrace() -> [[M31]] {
        let n = traceLength
        var colA = [M31](repeating: M31.zero, count: n)
        var colB = [M31](repeating: M31.zero, count: n)
        colA[0] = a0
        colB[0] = b0
        for i in 1..<n {
            colA[i] = colB[i - 1]
            colB[i] = m31Add(colA[i - 1], colB[i - 1])
        }
        return [colA, colB]
    }

    /// Transition constraints:
    ///   C0: a_next - b = 0
    ///   C1: b_next - (a + b) = 0
    public func evaluateConstraints(current: [M31], next: [M31]) -> [M31] {
        let c0 = m31Sub(next[0], current[1])                          // a' - b
        let c1 = m31Sub(next[1], m31Add(current[0], current[1]))      // b' - (a + b)
        return [c0, c1]
    }
}
