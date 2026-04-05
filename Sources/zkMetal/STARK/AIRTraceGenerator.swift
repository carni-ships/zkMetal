// AIRTraceGenerator — Plonky3-compatible execution trace generation engine
//
// Produces column-major execution traces for common AIR programs over BabyBear.
// Each generator returns [[Bb]] where the outer array is columns and the inner
// array is rows, automatically padded to power-of-2 length.
//
// Supported trace types:
//   - Fibonacci sequence (2 columns: a, b)
//   - Range check decomposition (bit decomposition columns)
//   - Permutation argument (original, permuted, grand product accumulator)
//   - Hash chain (Poseidon2 state columns across rounds)
//
// Compatible with Plonky3's BabyBearPoseidon2 configuration (SP1).

import Foundation

// MARK: - AIR Trace Generator

/// Produces execution traces for common AIR constraint programs over BabyBear.
///
/// All traces are returned in column-major format: `[[Bb]]` where `result[col][row]`
/// gives the field element at a specific column and row. Traces are automatically
/// padded to the next power of 2 in length.
public struct AIRTraceGenerator {

    public init() {}

    // MARK: - Padding Utility

    /// Pad a column-major trace so each column has length equal to the next power of 2.
    /// Padding rows replicate the last valid row (or zero if the trace is empty).
    public static func padToPowerOfTwo(_ trace: [[Bb]]) -> [[Bb]] {
        guard !trace.isEmpty else { return trace }
        let rowCount = trace[0].count
        guard rowCount > 0 else { return trace }

        let padded = nextPowerOfTwo(rowCount)
        if padded == rowCount { return trace }

        return trace.map { column in
            var col = column
            let lastVal = col.last ?? Bb.zero
            col.append(contentsOf: [Bb](repeating: lastVal, count: padded - rowCount))
            return col
        }
    }

    /// Compute the next power of 2 >= n.
    private static func nextPowerOfTwo(_ n: Int) -> Int {
        guard n > 0 else { return 1 }
        var v = n - 1
        v |= v >> 1
        v |= v >> 2
        v |= v >> 4
        v |= v >> 8
        v |= v >> 16
        v |= v >> 32
        return v + 1
    }

    // MARK: - Fibonacci Trace

    /// Generate a Fibonacci sequence trace with columns [a, b].
    ///
    /// Each row transitions as: a' = b, b' = a + b.
    /// Starting values are `a0` and `a1` (row 0: a=a0, b=a1).
    ///
    /// - Parameters:
    ///   - steps: Number of Fibonacci steps (rows in the trace before padding)
    ///   - a0: Initial value for column a (first Fibonacci element)
    ///   - a1: Initial value for column b (second Fibonacci element)
    /// - Returns: Column-major trace [[colA], [colB]], padded to power-of-2 length
    public func generateFibonacciTrace(steps: Int, a0: Bb, a1: Bb) -> [[Bb]] {
        precondition(steps >= 2, "Fibonacci trace needs at least 2 steps")

        var colA = [Bb](repeating: Bb.zero, count: steps)
        var colB = [Bb](repeating: Bb.zero, count: steps)

        colA[0] = a0
        colB[0] = a1

        for i in 1..<steps {
            colA[i] = colB[i - 1]
            colB[i] = bbAdd(colA[i - 1], colB[i - 1])
        }

        return AIRTraceGenerator.padToPowerOfTwo([colA, colB])
    }

    // MARK: - Range Check Trace

    /// Generate a range check decomposition trace.
    ///
    /// Each value is decomposed into bits to prove it lies in [0, bound).
    /// The trace has columns: [value, bit_0, bit_1, ..., bit_{k-1}]
    /// where k = ceil(log2(bound)) and value = sum(bit_i * 2^i).
    ///
    /// The transition constraint verifies that the reconstruction from bits
    /// equals the original value and each bit is boolean.
    ///
    /// - Parameters:
    ///   - values: Array of field elements to range-check
    ///   - bound: Upper bound (exclusive); must be a power of 2
    /// - Returns: Column-major trace, padded to power-of-2 length
    public func generateRangeCheckTrace(values: [Bb], bound: UInt64) -> [[Bb]] {
        precondition(bound > 0, "Bound must be positive")
        precondition(values.count > 0, "Need at least one value")

        // Number of bits needed: ceil(log2(bound))
        let numBits: Int
        if bound == 1 {
            numBits = 1
        } else {
            numBits = 64 - (bound - 1).leadingZeroBitCount
        }

        let rowCount = values.count

        // Column 0: the value itself
        var colValue = [Bb](repeating: Bb.zero, count: rowCount)
        // Columns 1..numBits: bit decomposition
        var bitColumns = (0..<numBits).map { _ in [Bb](repeating: Bb.zero, count: rowCount) }

        for (row, val) in values.enumerated() {
            colValue[row] = val
            let v = UInt64(val.v)
            for bit in 0..<numBits {
                let bitVal = (v >> bit) & 1
                bitColumns[bit][row] = Bb(v: UInt32(bitVal))
            }
        }

        var columns = [[Bb]]()
        columns.append(colValue)
        columns.append(contentsOf: bitColumns)

        return AIRTraceGenerator.padToPowerOfTwo(columns)
    }

    // MARK: - Permutation Trace

    /// Generate a permutation argument trace with grand product accumulator.
    ///
    /// Proves that `permuted` is a permutation of `original` using a random
    /// challenge-based grand product argument. The trace has columns:
    ///   [original, permuted, accumulator]
    /// where accumulator[i] = prod_{j<=i} (original[j] + beta) / (permuted[j] + beta)
    /// and beta is a deterministic challenge derived from the inputs.
    ///
    /// The grand product should equal 1 at the final row if `permuted` is
    /// indeed a permutation of `original`.
    ///
    /// - Parameters:
    ///   - original: Original values
    ///   - permuted: Claimed permutation of original
    /// - Returns: Column-major trace [original, permuted, accumulator], padded to power-of-2
    public func generatePermutationTrace(original: [Bb], permuted: [Bb]) -> [[Bb]] {
        precondition(original.count == permuted.count, "Original and permuted must have same length")
        precondition(original.count > 0, "Need at least one element")

        let n = original.count

        // Derive a deterministic challenge beta from the inputs.
        // In a real STARK, this comes from the Fiat-Shamir transcript after
        // committing to original and permuted columns.
        let beta = deriveBetaChallenge(original: original, permuted: permuted)

        // Build the grand product accumulator column
        var colOriginal = [Bb](repeating: Bb.zero, count: n)
        var colPermuted = [Bb](repeating: Bb.zero, count: n)
        var colAccumulator = [Bb](repeating: Bb.zero, count: n)

        var runningProduct = Bb.one

        for i in 0..<n {
            colOriginal[i] = original[i]
            colPermuted[i] = permuted[i]

            // accumulator[i] = running_product * (original[i] + beta) / (permuted[i] + beta)
            let numerator = bbAdd(original[i], beta)
            let denominator = bbAdd(permuted[i], beta)
            let denomInv = bbInverse(denominator)
            runningProduct = bbMul(runningProduct, bbMul(numerator, denomInv))

            colAccumulator[i] = runningProduct
        }

        return AIRTraceGenerator.padToPowerOfTwo([colOriginal, colPermuted, colAccumulator])
    }

    /// Derive a deterministic beta challenge from original and permuted values.
    /// Uses a simple hash: sum of all values scaled by position, reduced mod p.
    private func deriveBetaChallenge(original: [Bb], permuted: [Bb]) -> Bb {
        var acc: UInt64 = 7  // seed with a nonzero value
        for (i, val) in original.enumerated() {
            acc = acc &+ UInt64(val.v) &* UInt64(i &+ 1)
        }
        for (i, val) in permuted.enumerated() {
            acc = acc &+ UInt64(val.v) &* UInt64(i &+ original.count &+ 1)
        }
        // Reduce and ensure nonzero
        let reduced = UInt32(acc % UInt64(Bb.P))
        return Bb(v: reduced == 0 ? 1 : reduced)
    }

    // MARK: - Hash Chain Trace

    /// Generate a hash chain trace for Poseidon2 (or any hash function).
    ///
    /// Computes a sequential hash chain: h_0 = hash(inputs[0]),
    /// h_1 = hash(h_0 || inputs[1]), etc. The trace records the state at each step
    /// so the STARK can verify the entire chain.
    ///
    /// The trace columns are the hash state elements at each step.
    /// If the hash function produces `w` output elements, the trace has `w` columns
    /// and `inputs.count` rows (padded to power-of-2).
    ///
    /// - Parameters:
    ///   - inputs: Array of input blocks, each block is an array of field elements
    ///   - hashFn: Hash function mapping [Bb] -> [Bb] (e.g., Poseidon2 sponge)
    /// - Returns: Column-major trace of hash states, padded to power-of-2
    public func generateHashChainTrace(inputs: [[Bb]], hashFn: ([Bb]) -> [Bb]) -> [[Bb]] {
        precondition(inputs.count > 0, "Need at least one input block")

        // Compute the first hash to determine output width
        let firstHash = hashFn(inputs[0])
        let stateWidth = firstHash.count
        precondition(stateWidth > 0, "Hash function must produce at least one output element")

        let n = inputs.count
        // Initialize columns: one column per state element
        var columns = (0..<stateWidth).map { _ in [Bb](repeating: Bb.zero, count: n) }

        // Row 0: hash of first input
        var currentHash = firstHash
        for col in 0..<stateWidth {
            columns[col][0] = currentHash[col]
        }

        // Subsequent rows: chain the hash
        for row in 1..<n {
            // Concatenate previous hash output with current input block
            var hashInput = currentHash
            hashInput.append(contentsOf: inputs[row])
            currentHash = hashFn(hashInput)

            for col in 0..<stateWidth {
                columns[col][row] = currentHash[col]
            }
        }

        return AIRTraceGenerator.padToPowerOfTwo(columns)
    }
}
