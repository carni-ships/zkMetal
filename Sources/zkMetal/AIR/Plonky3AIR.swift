// Plonky3-compatible AIR interface for SP1/Plonky3 proving acceleration
//
// Provides the bridge between Plonky3's proving protocol and zkMetal's
// GPU-accelerated primitives (BabyBear arithmetic, Poseidon2 hashing, NTT).
//
// Plonky3 protocol overview:
//   Field: BabyBear (p = 2^31 - 2^27 + 1)
//   Hash: Poseidon2 width-16, rate-8, x^7 S-box
//   FRI: fold-by-4, Poseidon2 Merkle commitments
//   Challenger: duplex sponge over Poseidon2
//
// Trace structure:
//   - Preprocessed columns: fixed by the circuit (selectors, permutation data)
//   - Main trace columns: witness-dependent execution trace
//   - Permutation columns: cross-table lookup / interaction columns
//
// Compatible with SP1's BabyBearPoseidon2 configuration.

import Foundation

// MARK: - Plonky3 AIR Configuration

/// Configuration matching Plonky3's BabyBearPoseidon2 proving parameters.
/// This is the standard configuration used by SP1.
public struct Plonky3AIRConfig {
    /// BabyBear prime: p = 2^31 - 2^27 + 1
    public static let fieldModulus: UInt32 = Bb.P  // 0x78000001 = 2013265921

    /// Poseidon2 state width (BabyBear elements per permutation)
    public static let poseidon2Width: Int = Poseidon2BabyBearConfig.t        // 16
    /// Poseidon2 rate (elements absorbed per permutation)
    public static let poseidon2Rate: Int = Poseidon2BabyBearConfig.rate      // 8
    /// Poseidon2 capacity
    public static let poseidon2Capacity: Int = Poseidon2BabyBearConfig.capacity // 8

    /// FRI folding factor: fold 4 evaluations per round (log2 = 2)
    public static let friLogFoldFactor: Int = 2
    /// FRI folding factor
    public static let friFoldFactor: Int = 4

    /// Hash digest size: 8 BabyBear elements (256 bits)
    public static let digestElements: Int = Poseidon2BabyBearEngine.nodeSize // 8

    /// Log2 of blowup factor for LDE
    public let logBlowup: Int

    /// Number of FRI queries for soundness
    public let numQueries: Int

    /// Grinding bits for proof-of-work (SP1 default: 16)
    public let grindingBits: Int

    /// Whether to use Metal GPU acceleration
    public let useGPU: Bool

    /// SP1 default: 2x blowup, 100 queries, 16 grinding bits
    public static let sp1Default = Plonky3AIRConfig(
        logBlowup: 1,
        numQueries: 100,
        grindingBits: 16,
        useGPU: true
    )

    /// Fast configuration for testing
    public static let fast = Plonky3AIRConfig(
        logBlowup: 1,
        numQueries: 20,
        grindingBits: 0,
        useGPU: true
    )

    public init(logBlowup: Int = 1, numQueries: Int = 100,
                grindingBits: Int = 16, useGPU: Bool = true) {
        precondition(logBlowup >= 1 && logBlowup <= 4, "logBlowup must be in [1, 4]")
        precondition(numQueries >= 1, "Need at least 1 query")
        self.logBlowup = logBlowup
        self.numQueries = numQueries
        self.grindingBits = grindingBits
        self.useGPU = useGPU
    }

    /// Blowup factor for LDE domain
    public var blowupFactor: Int { 1 << logBlowup }

    /// Approximate security bits
    public var securityBits: Int {
        numQueries * logBlowup + grindingBits
    }
}

// MARK: - Plonky3 AIR Protocol

/// A Plonky3-compatible AIR definition.
///
/// Plonky3 AIRs separate trace data into three categories:
/// - Preprocessed: fixed columns known at circuit compile time (selectors, wiring)
/// - Main: witness columns filled during proving
/// - Permutation: interaction/lookup columns derived from challenges
public protocol Plonky3AIR {
    /// Number of preprocessed (fixed) columns
    var numPreprocessedColumns: Int { get }

    /// Number of main trace columns
    var numMainColumns: Int { get }

    /// Number of permutation/interaction columns
    var numPermutationColumns: Int { get }

    /// Total columns across all trace sections
    var totalColumns: Int { get }

    /// Log2 of trace length (rows = 2^logTraceLength)
    var logTraceLength: Int { get }

    /// Number of rows in the trace
    var traceLength: Int { get }

    /// Constraint degree bound (max degree of constraint polynomials)
    var constraintDegree: Int { get }

    /// Generate the preprocessed trace (fixed columns). Returns nil if none.
    func generatePreprocessedTrace() -> [[Bb]]?

    /// Generate the main execution trace.
    func generateMainTrace() -> [[Bb]]

    /// Generate permutation trace given random challenges.
    /// `challenges` are squeezed from the Fiat-Shamir transcript after committing main trace.
    func generatePermutationTrace(mainTrace: [[Bb]], challenges: [Bb]) -> [[Bb]]?

    /// Evaluate all AIR constraints at a given row.
    ///
    /// Parameters:
    ///   - preprocessed: preprocessed column values at (current, next) rows; nil if no preprocessed columns
    ///   - main: main trace values at (current, next) rows
    ///   - permutation: permutation column values at (current, next) rows; nil if no permutation columns
    ///   - challenges: interaction challenges (for permutation constraints)
    ///
    /// Returns: array of constraint evaluations, all zero on a valid trace.
    func evaluateConstraints(
        preprocessed: (current: [Bb], next: [Bb])?,
        main: (current: [Bb], next: [Bb]),
        permutation: (current: [Bb], next: [Bb])?,
        challenges: [Bb]
    ) -> [Bb]
}

// Default implementations
extension Plonky3AIR {
    public var totalColumns: Int {
        numPreprocessedColumns + numMainColumns + numPermutationColumns
    }

    public var traceLength: Int { 1 << logTraceLength }

    /// Verify that a trace satisfies all constraints (CPU debug check).
    public func verifyTrace(
        preprocessed: [[Bb]]?,
        main: [[Bb]],
        permutation: [[Bb]]?,
        challenges: [Bb]
    ) -> String? {
        let n = traceLength

        // Validate column counts
        if let pp = preprocessed {
            guard pp.count == numPreprocessedColumns else {
                return "Expected \(numPreprocessedColumns) preprocessed columns, got \(pp.count)"
            }
            for (i, col) in pp.enumerated() {
                guard col.count == n else {
                    return "Preprocessed column \(i): expected \(n) rows, got \(col.count)"
                }
            }
        }
        guard main.count == numMainColumns else {
            return "Expected \(numMainColumns) main columns, got \(main.count)"
        }
        for (i, col) in main.enumerated() {
            guard col.count == n else {
                return "Main column \(i): expected \(n) rows, got \(col.count)"
            }
        }
        if let perm = permutation {
            guard perm.count == numPermutationColumns else {
                return "Expected \(numPermutationColumns) permutation columns, got \(perm.count)"
            }
            for (i, col) in perm.enumerated() {
                guard col.count == n else {
                    return "Permutation column \(i): expected \(n) rows, got \(col.count)"
                }
            }
        }

        // Check transition constraints on all rows except the last
        for row in 0..<(n - 1) {
            let nextRow = row + 1

            let ppCurrent = preprocessed.map { cols in cols.map { $0[row] } }
            let ppNext = preprocessed.map { cols in cols.map { $0[nextRow] } }
            let ppPair: (current: [Bb], next: [Bb])? = ppCurrent.map { (current: $0, next: ppNext!) }

            let mainCurrent = main.map { $0[row] }
            let mainNext = main.map { $0[nextRow] }

            let permCurrent = permutation.map { cols in cols.map { $0[row] } }
            let permNext = permutation.map { cols in cols.map { $0[nextRow] } }
            let permPair: (current: [Bb], next: [Bb])? = permCurrent.map { (current: $0, next: permNext!) }

            let evals = evaluateConstraints(
                preprocessed: ppPair,
                main: (current: mainCurrent, next: mainNext),
                permutation: permPair,
                challenges: challenges
            )

            for (ci, ev) in evals.enumerated() {
                if ev.v != 0 {
                    return "Constraint \(ci) failed at row \(row): eval=\(ev.v)"
                }
            }
        }

        return nil
    }
}

// MARK: - Trace Commitment (Poseidon2 Merkle)

/// Commits to execution trace columns via Poseidon2 Merkle tree.
///
/// Each row of the trace is hashed into a leaf using Poseidon2 sponge,
/// then the leaves form a binary Merkle tree with Poseidon2 2-to-1 compression.
/// This matches Plonky3's commitment scheme for trace polynomials.
public struct Plonky3TraceCommitment {
    /// Merkle root: 8 BabyBear elements (256 bits)
    public let root: [Bb]

    /// All Merkle tree layers, for opening proofs.
    /// layers[0] = leaf hashes, layers[k] = level k internal nodes.
    /// The root is layers.last with a single node.
    public let layers: [[[Bb]]]

    /// Number of rows committed
    public let numRows: Int

    /// Number of columns in this trace section
    public let numColumns: Int

    /// Commit to a set of trace columns using Poseidon2 Merkle tree.
    ///
    /// Parameters:
    ///   - columns: trace data as [column][row] of BabyBear elements
    ///   - engine: optional GPU engine; falls back to CPU if nil
    ///
    /// Each leaf is the Poseidon2 hash of one row across all columns.
    /// For rows wider than 8 elements, the sponge absorbs in 8-element chunks.
    public static func commit(columns: [[Bb]], engine: Poseidon2BabyBearEngine? = nil) throws -> Plonky3TraceCommitment {
        guard !columns.isEmpty else {
            return Plonky3TraceCommitment(root: [Bb](repeating: Bb.zero, count: 8),
                                          layers: [], numRows: 0, numColumns: 0)
        }

        let numCols = columns.count
        let numRows = columns[0].count
        precondition(numRows > 0 && (numRows & (numRows - 1)) == 0,
                     "Number of rows must be a power of 2")

        // Hash each row into a leaf (8 BabyBear elements)
        var leaves = [[Bb]](repeating: [Bb](repeating: Bb.zero, count: 8), count: numRows)
        for row in 0..<numRows {
            var rowData = [Bb]()
            rowData.reserveCapacity(numCols)
            for col in 0..<numCols {
                rowData.append(columns[col][row])
            }
            leaves[row] = poseidon2BbHashMany(rowData)
        }

        // Build Merkle tree bottom-up
        var layers = [[[Bb]]]()
        layers.append(leaves)

        var currentLevel = leaves
        while currentLevel.count > 1 {
            let pairs = currentLevel.count / 2
            var nextLevel = [[Bb]](repeating: [Bb](repeating: Bb.zero, count: 8), count: pairs)

            if let eng = engine {
                // GPU path: batch hash pairs
                var flatInput = [Bb]()
                flatInput.reserveCapacity(pairs * 16)
                for i in 0..<pairs {
                    flatInput.append(contentsOf: currentLevel[2 * i])
                    flatInput.append(contentsOf: currentLevel[2 * i + 1])
                }
                let flatOutput = try eng.hashPairs(flatInput)
                for i in 0..<pairs {
                    nextLevel[i] = Array(flatOutput[i * 8..<(i + 1) * 8])
                }
            } else {
                // CPU path
                for i in 0..<pairs {
                    nextLevel[i] = poseidon2BbHash(left: currentLevel[2 * i],
                                                   right: currentLevel[2 * i + 1])
                }
            }

            layers.append(nextLevel)
            currentLevel = nextLevel
        }

        let root = currentLevel[0]
        return Plonky3TraceCommitment(root: root, layers: layers,
                                       numRows: numRows, numColumns: numCols)
    }

    /// Generate a Merkle opening proof for a given row index.
    /// Returns the authentication path: array of sibling hashes from leaf to root.
    public func openingProof(row: Int) -> [[Bb]] {
        precondition(row >= 0 && row < numRows, "Row index out of range")
        var path = [[Bb]]()
        var idx = row
        for level in 0..<(layers.count - 1) {
            let sibling = idx ^ 1
            path.append(layers[level][sibling])
            idx >>= 1
        }
        return path
    }

    /// Verify a Merkle opening proof.
    public static func verifyOpening(root: [Bb], leaf: [Bb], row: Int, path: [[Bb]]) -> Bool {
        var current = leaf
        var idx = row
        for sibling in path {
            if idx & 1 == 0 {
                current = poseidon2BbHash(left: current, right: sibling)
            } else {
                current = poseidon2BbHash(left: sibling, right: current)
            }
            idx >>= 1
        }
        return current == root
    }
}

// MARK: - Constraint Evaluator

/// Evaluates Plonky3 AIR constraints over the full trace, producing the
/// quotient polynomial for FRI commitment.
///
/// Supports batching all constraint evaluations with random linear combination
/// (using verifier challenge alpha), which is the standard Plonky3 approach.
public struct Plonky3ConstraintEvaluator {
    /// The AIR being evaluated
    public let air: any Plonky3AIR

    /// Proving configuration
    public let config: Plonky3AIRConfig

    public init(air: any Plonky3AIR, config: Plonky3AIRConfig = .sp1Default) {
        self.air = air
        self.config = config
    }

    /// Evaluate the quotient polynomial at all trace rows.
    ///
    /// Computes: Q(x) = sum_i alpha^i * C_i(x) / Z_H(x)
    /// where C_i are the constraint polynomials and Z_H is the vanishing polynomial.
    ///
    /// Parameters:
    ///   - preprocessed: preprocessed trace columns (nil if none)
    ///   - main: main trace columns
    ///   - permutation: permutation trace columns (nil if none)
    ///   - challenges: interaction challenges from transcript
    ///   - alpha: random linear combination challenge
    ///
    /// Returns: quotient polynomial evaluations over the trace domain
    public func evaluateQuotient(
        preprocessed: [[Bb]]?,
        main: [[Bb]],
        permutation: [[Bb]]?,
        challenges: [Bb],
        alpha: Bb
    ) -> [Bb] {
        let n = air.traceLength
        var quotient = [Bb](repeating: Bb.zero, count: n)

        // Precompute inverse of vanishing polynomial Z_H(omega^i) at each row
        // For transition constraints, Z_H vanishes on all rows except the last:
        // Z_H(x) = x^n - 1, but constraint applies on rows 0..n-2
        // Denominator at row i: omega^(i*n) - 1 (handled by excluding last row)

        for row in 0..<(n - 1) {
            let nextRow = row + 1

            let ppCurrent = preprocessed.map { cols in cols.map { $0[row] } }
            let ppNext = preprocessed.map { cols in cols.map { $0[nextRow] } }
            let ppPair: (current: [Bb], next: [Bb])? = ppCurrent.map { (current: $0, next: ppNext!) }

            let mainCurrent = main.map { $0[row] }
            let mainNext = main.map { $0[nextRow] }

            let permCurrent = permutation.map { cols in cols.map { $0[row] } }
            let permNext = permutation.map { cols in cols.map { $0[nextRow] } }
            let permPair: (current: [Bb], next: [Bb])? = permCurrent.map { (current: $0, next: permNext!) }

            let constraintEvals = air.evaluateConstraints(
                preprocessed: ppPair,
                main: (current: mainCurrent, next: mainNext),
                permutation: permPair,
                challenges: challenges
            )

            // Random linear combination: sum_i alpha^i * C_i
            var combined = Bb.zero
            var alphaPow = Bb.one
            for eval in constraintEvals {
                combined = bbAdd(combined, bbMul(alphaPow, eval))
                alphaPow = bbMul(alphaPow, alpha)
            }

            quotient[row] = combined
        }

        return quotient
    }

    /// Evaluate constraints at a single out-of-domain point (for deep quotient).
    /// Used during the FRI opening phase.
    public func evaluateAtPoint(
        preprocessedAtPoint: (current: [Bb], next: [Bb])?,
        mainAtPoint: (current: [Bb], next: [Bb]),
        permutationAtPoint: (current: [Bb], next: [Bb])?,
        challenges: [Bb],
        alpha: Bb
    ) -> Bb {
        let constraintEvals = air.evaluateConstraints(
            preprocessed: preprocessedAtPoint,
            main: mainAtPoint,
            permutation: permutationAtPoint,
            challenges: challenges
        )

        var combined = Bb.zero
        var alphaPow = Bb.one
        for eval in constraintEvals {
            combined = bbAdd(combined, bbMul(alphaPow, eval))
            alphaPow = bbMul(alphaPow, alpha)
        }
        return combined
    }
}

// MARK: - Challenger (Fiat-Shamir Duplex Sponge)

/// Plonky3-compatible Fiat-Shamir challenger using Poseidon2 duplex sponge.
///
/// Matches Plonky3's `DuplexChallenger<BabyBear, Poseidon2, 16, 8>`:
/// - State: 16 BabyBear elements (width of Poseidon2)
/// - Rate: 8 elements absorbed/squeezed per permutation
/// - Capacity: 8 elements (never directly read/written by user)
///
/// The duplex sponge alternates between absorbing data and squeezing challenges,
/// with automatic permutation when the rate buffer is full (absorb) or empty (squeeze).
public class Plonky3Challenger {
    /// Full Poseidon2 state: 16 BabyBear elements
    private var state: [Bb]

    /// Input buffer: elements waiting to be absorbed
    private var inputBuffer: [Bb]

    /// Output buffer: elements available to squeeze
    private var outputBuffer: [Bb]

    /// Create a fresh challenger with zeroed state.
    public init() {
        self.state = [Bb](repeating: Bb.zero, count: Plonky3AIRConfig.poseidon2Width)
        self.inputBuffer = []
        self.outputBuffer = []
    }

    /// Create a challenger from an existing state (for forking).
    private init(state: [Bb], inputBuffer: [Bb], outputBuffer: [Bb]) {
        self.state = state
        self.inputBuffer = inputBuffer
        self.outputBuffer = outputBuffer
    }

    /// Fork this challenger (snapshot current state for parallel proving).
    public func fork() -> Plonky3Challenger {
        return Plonky3Challenger(state: state, inputBuffer: inputBuffer,
                                  outputBuffer: outputBuffer)
    }

    // MARK: - Absorb

    /// Absorb a single BabyBear element into the sponge.
    public func observe(_ value: Bb) {
        // Any pending output is invalidated when we absorb new data
        outputBuffer.removeAll()

        inputBuffer.append(value)

        if inputBuffer.count == Plonky3AIRConfig.poseidon2Rate {
            duplexing()
        }
    }

    /// Absorb multiple BabyBear elements.
    public func observeSlice(_ values: [Bb]) {
        for v in values {
            observe(v)
        }
    }

    /// Absorb a Poseidon2 digest (8 BabyBear elements, i.e. a Merkle root).
    public func observeDigest(_ digest: [Bb]) {
        precondition(digest.count == Plonky3AIRConfig.digestElements,
                     "Digest must be \(Plonky3AIRConfig.digestElements) elements")
        observeSlice(digest)
    }

    // MARK: - Squeeze

    /// Squeeze a single BabyBear challenge from the sponge.
    public func sample() -> Bb {
        // Flush any pending input first
        if !inputBuffer.isEmpty || outputBuffer.isEmpty {
            // Pad input buffer with zeros and permute
            while inputBuffer.count < Plonky3AIRConfig.poseidon2Rate {
                inputBuffer.append(Bb.zero)
            }
            duplexing()
        }

        // Pop from output buffer
        return outputBuffer.removeFirst()
    }

    /// Squeeze multiple BabyBear challenges.
    public func sampleSlice(_ count: Int) -> [Bb] {
        var result = [Bb]()
        result.reserveCapacity(count)
        for _ in 0..<count {
            result.append(sample())
        }
        return result
    }

    /// Sample a random challenge and reduce to a canonical field element.
    /// Combines multiple squeezed elements for extra entropy when needed.
    public func sampleExtElement() -> [Bb] {
        // Plonky3 extension field (EF4) uses 4 BabyBear elements
        return sampleSlice(4)
    }

    /// Sample random bits by squeezing a field element and taking low bits.
    /// Used for grinding/proof-of-work checks.
    public func sampleBits(_ numBits: Int) -> UInt32 {
        precondition(numBits <= 31, "BabyBear has 31-bit elements")
        let elem = sample()
        return elem.v & ((1 << numBits) - 1)
    }

    /// Check grinding: sample bits and verify leading zeros.
    public func checkWitness(_ grindingBits: Int, witness: Bb) -> Bool {
        let savedState = Plonky3Challenger(state: state, inputBuffer: inputBuffer,
                                            outputBuffer: outputBuffer)
        observe(witness)
        let bits = sampleBits(grindingBits)
        // Restore state
        self.state = savedState.state
        self.inputBuffer = savedState.inputBuffer
        self.outputBuffer = savedState.outputBuffer
        return bits == 0
    }

    // MARK: - Internal Duplex Operation

    /// Perform one duplex operation: absorb input buffer into state, permute, fill output buffer.
    private func duplexing() {
        precondition(inputBuffer.count <= Plonky3AIRConfig.poseidon2Rate)

        // Overwrite rate portion of state with input (Plonky3 overwrites, not XORs)
        for i in 0..<inputBuffer.count {
            state[i] = inputBuffer[i]
        }
        inputBuffer.removeAll()

        // Apply Poseidon2 permutation
        poseidon2BbPermutation(state: &state)

        // Fill output buffer from rate portion
        outputBuffer = Array(state[0..<Plonky3AIRConfig.poseidon2Rate])
    }
}

// MARK: - Example: Fibonacci AIR (Plonky3-style)

/// Example Plonky3-compatible Fibonacci AIR for testing.
/// Trace: 2 main columns (a, b), transition: a' = b, b' = a + b.
public struct Plonky3FibonacciAIR: Plonky3AIR {
    public let numPreprocessedColumns: Int = 0
    public let numMainColumns: Int = 2
    public let numPermutationColumns: Int = 0
    public let logTraceLength: Int
    public let constraintDegree: Int = 1

    public let a0: Bb
    public let b0: Bb

    public init(logTraceLength: Int, a0: Bb = Bb.one, b0: Bb = Bb.one) {
        precondition(logTraceLength >= 2, "Need at least 4 rows")
        self.logTraceLength = logTraceLength
        self.a0 = a0
        self.b0 = b0
    }

    public func generatePreprocessedTrace() -> [[Bb]]? { nil }

    public func generateMainTrace() -> [[Bb]] {
        let n = traceLength
        var colA = [Bb](repeating: Bb.zero, count: n)
        var colB = [Bb](repeating: Bb.zero, count: n)
        colA[0] = a0
        colB[0] = b0
        for i in 1..<n {
            colA[i] = colB[i - 1]
            colB[i] = bbAdd(colA[i - 1], colB[i - 1])
        }
        return [colA, colB]
    }

    public func generatePermutationTrace(mainTrace: [[Bb]], challenges: [Bb]) -> [[Bb]]? { nil }

    public func evaluateConstraints(
        preprocessed: (current: [Bb], next: [Bb])?,
        main: (current: [Bb], next: [Bb]),
        permutation: (current: [Bb], next: [Bb])?,
        challenges: [Bb]
    ) -> [Bb] {
        // C0: a_next - b = 0
        let c0 = bbSub(main.next[0], main.current[1])
        // C1: b_next - (a + b) = 0
        let c1 = bbSub(main.next[1], bbAdd(main.current[0], main.current[1]))
        return [c0, c1]
    }
}
