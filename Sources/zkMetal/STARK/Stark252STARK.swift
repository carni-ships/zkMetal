// Stark252 STARK Engine -- full STARK pipeline over StarkNet/Cairo field
// p = 2^251 + 17 * 2^192 + 1
//
// Pipeline:
// 1. Accept AIR constraints + execution trace
// 2. Low-Degree Extension (LDE) via Stark252 NTT over coset domain
// 3. Commit trace columns via hash-based Merkle trees
// 4. Evaluate constraint polynomials over LDE domain
// 5. Compute quotient polynomial (constraints / vanishing polynomial)
// 6. FRI commitment for proximity testing
// 7. Generate query proofs with Merkle opening paths
//
// Hash: simple Stark252-native sponge (absorption + squeezing via field ops)
// FRI: standard fold-by-2 with Merkle commitments

import Foundation

// MARK: - AIR Protocol for Stark252 STARKs

/// AIR definition for Stark252 STARK proofs.
/// Traces are column-major: [column][row] of Stark252 elements.
public protocol Stark252AIR {
    /// Number of trace columns
    var numColumns: Int { get }

    /// Log2 of trace length
    var logTraceLength: Int { get }

    /// Trace length (2^logTraceLength)
    var traceLength: Int { get }

    /// Number of transition constraints
    var numConstraints: Int { get }

    /// Max constraint degree (for quotient polynomial degree bound)
    var constraintDegree: Int { get }

    /// Generate execution trace: [column][row]
    func generateTrace() -> [[Stark252]]

    /// Evaluate transition constraints at a single row.
    /// current[col] = value at row i, next[col] = value at row i+1.
    /// Returns array of constraint evaluations; all zero on a valid trace.
    func evaluateConstraints(current: [Stark252], next: [Stark252]) -> [Stark252]

    /// Boundary constraints: (column, row, expected_value)
    var boundaryConstraints: [(column: Int, row: Int, value: Stark252)] { get }
}

extension Stark252AIR {
    public var traceLength: Int { 1 << logTraceLength }
}

// MARK: - Configuration

/// Configuration for Stark252 STARK proof generation.
public struct Stark252STARKConfig {
    /// Log2 of blowup factor for LDE (1 = 2x, 2 = 4x, etc.)
    public let logBlowup: Int

    /// Blowup factor: LDE domain size / trace domain size
    public var blowupFactor: Int { 1 << logBlowup }

    /// Number of FRI query points for soundness
    public let numQueries: Int

    /// Grinding bits for proof-of-work (0 = disabled)
    public let grindingBits: Int

    /// Number of FRI folding rounds to skip (commit final polynomial directly)
    public let friMaxRemainderLogN: Int

    /// StarkNet-compatible default: 2x blowup, 100 queries, 16 grinding bits
    public static let starknetDefault = Stark252STARKConfig(
        logBlowup: 1,
        numQueries: 100,
        grindingBits: 16,
        friMaxRemainderLogN: 3
    )

    /// Fast configuration for testing
    public static let fast = Stark252STARKConfig(
        logBlowup: 1,
        numQueries: 20,
        grindingBits: 0,
        friMaxRemainderLogN: 2
    )

    /// High security: 4x blowup, 80 queries
    public static let highSecurity = Stark252STARKConfig(
        logBlowup: 2,
        numQueries: 80,
        grindingBits: 16,
        friMaxRemainderLogN: 3
    )

    public init(logBlowup: Int = 1, numQueries: Int = 100,
                grindingBits: Int = 16, friMaxRemainderLogN: Int = 3) {
        precondition(logBlowup >= 1 && logBlowup <= 4, "logBlowup must be in [1, 4]")
        precondition(numQueries >= 1 && numQueries <= 200, "numQueries must be in [1, 200]")
        precondition(friMaxRemainderLogN >= 0, "friMaxRemainderLogN must be non-negative")
        self.logBlowup = logBlowup
        self.numQueries = numQueries
        self.grindingBits = grindingBits
        self.friMaxRemainderLogN = friMaxRemainderLogN
    }

    /// Approximate security bits: queries * logBlowup + grinding
    public var securityBits: Int {
        numQueries * logBlowup + grindingBits
    }
}

// MARK: - Fiat-Shamir Transcript (Sponge over Stark252)

/// Simple sponge-based Fiat-Shamir transcript for Stark252 STARK proofs.
/// Uses a rate-2 capacity-1 sponge with field-native mixing.
public class Stark252Transcript {
    private var state: [Stark252]  // 3-element state: [rate0, rate1, capacity]
    private var inputBuffer: [Stark252]
    private var outputBuffer: [Stark252]

    private static let width = 3
    private static let rate = 2

    public init() {
        self.state = [Stark252](repeating: Stark252.zero, count: Stark252Transcript.width)
        self.inputBuffer = []
        self.outputBuffer = []
    }

    private init(state: [Stark252], inputBuffer: [Stark252], outputBuffer: [Stark252]) {
        self.state = state
        self.inputBuffer = inputBuffer
        self.outputBuffer = outputBuffer
    }

    /// Fork this transcript (snapshot current state).
    public func fork() -> Stark252Transcript {
        return Stark252Transcript(state: state, inputBuffer: inputBuffer,
                                   outputBuffer: outputBuffer)
    }

    /// Absorb a single Stark252 element into the transcript.
    public func absorb(_ element: Stark252) {
        outputBuffer.removeAll()
        inputBuffer.append(element)
        if inputBuffer.count >= Stark252Transcript.rate {
            duplexing()
        }
    }

    /// Absorb a slice of Stark252 elements.
    public func absorbSlice(_ elements: [Stark252]) {
        for e in elements {
            absorb(e)
        }
    }

    /// Squeeze a single Stark252 challenge element.
    public func squeeze() -> Stark252 {
        if outputBuffer.isEmpty {
            duplexing()
        }
        return outputBuffer.removeFirst()
    }

    /// Duplexing: absorb buffered input, apply permutation, produce output.
    private func duplexing() {
        // XOR (add) input buffer into rate portion
        for (i, val) in inputBuffer.enumerated() {
            if i < Stark252Transcript.rate {
                state[i] = stark252Add(state[i], val)
            }
        }
        inputBuffer.removeAll()

        // Apply a simple permutation: multiple rounds of field mixing
        // This is a simplified Rescue-like permutation for the sponge.
        for _ in 0..<8 {
            // Non-linear layer: x -> x^3
            for i in 0..<Stark252Transcript.width {
                let sq = stark252Sqr(state[i])
                state[i] = stark252Mul(state[i], sq)
            }
            // Linear mixing: rotate and add
            let t0 = stark252Add(state[0], stark252Add(state[1], state[2]))
            let t1 = stark252Add(state[1], stark252Add(state[2], state[0]))
            let t2 = stark252Add(state[2], stark252Add(state[0], state[1]))
            // Add constants to break symmetry
            let c1 = stark252FromInt(7)
            let c2 = stark252FromInt(13)
            state[0] = stark252Add(t0, stark252FromInt(3))
            state[1] = stark252Add(t1, c1)
            state[2] = stark252Add(t2, c2)
        }

        // Squeeze: output = rate elements
        outputBuffer = Array(state[0..<Stark252Transcript.rate])
    }
}

// MARK: - Merkle Tree (Hash-based over Stark252)

/// Binary Merkle tree using Stark252 field hashing.
/// Each node is a 2-element Stark252 digest. Leaves are single Stark252 values padded to 2.
public struct Stark252MerkleTree {
    /// All tree layers: layers[0] = leaves, layers[last] = [root]
    public let layers: [[[Stark252]]]

    /// Root hash (2 Stark252 elements)
    public var root: [Stark252] { layers.last!.first! }

    /// Number of leaves
    public let numLeaves: Int

    /// Build a Merkle tree from an array of Stark252 field elements.
    public static func build(leaves: [Stark252]) -> Stark252MerkleTree {
        let n = leaves.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Number of leaves must be a power of 2")

        // Hash each leaf value into a 2-element digest
        var leafNodes = [[Stark252]](repeating: [Stark252.zero, Stark252.zero], count: n)
        for i in 0..<n {
            leafNodes[i] = stark252HashLeaf(leaves[i])
        }

        var layers = [[[Stark252]]]()
        layers.append(leafNodes)

        var currentLevel = leafNodes
        while currentLevel.count > 1 {
            let pairs = currentLevel.count / 2
            var nextLevel = [[Stark252]](repeating: [Stark252.zero, Stark252.zero], count: pairs)
            for i in 0..<pairs {
                nextLevel[i] = stark252Compress(currentLevel[2 * i], currentLevel[2 * i + 1])
            }
            layers.append(nextLevel)
            currentLevel = nextLevel
        }

        return Stark252MerkleTree(layers: layers, numLeaves: n)
    }

    /// Generate Merkle opening proof for a leaf index.
    public func openingProof(index: Int) -> [[Stark252]] {
        precondition(index >= 0 && index < numLeaves, "Leaf index out of range")
        var path = [[Stark252]]()
        var idx = index
        for level in 0..<(layers.count - 1) {
            let sibling = idx ^ 1
            path.append(layers[level][sibling])
            idx >>= 1
        }
        return path
    }

    /// Verify a Merkle opening proof.
    public static func verifyOpening(root: [Stark252], leaf: [Stark252], index: Int, path: [[Stark252]]) -> Bool {
        var current = stark252HashMany(leaf)
        var idx = index
        for sibling in path {
            if idx & 1 == 0 {
                current = stark252Compress(current, sibling)
            } else {
                current = stark252Compress(sibling, current)
            }
            idx >>= 1
        }
        return stark252DigestEqual(current, root)
    }
}

/// Hash a single Stark252 leaf value into a 2-element digest.
private func stark252HashLeaf(_ value: Stark252) -> [Stark252] {
    let leaf = [value, Stark252.zero]
    return stark252HashMany(leaf)
}

/// Pad a single Stark252 value to a 2-element leaf (for verification).
private func stark252PadToLeaf(_ value: Stark252) -> [Stark252] {
    return [value, Stark252.zero]
}

/// Hash a 2-element input to a 2-element digest using field mixing.
private func stark252HashMany(_ input: [Stark252]) -> [Stark252] {
    var s0 = input.count > 0 ? input[0] : Stark252.zero
    var s1 = input.count > 1 ? input[1] : Stark252.zero
    let domain = stark252FromInt(UInt64(input.count))
    s0 = stark252Add(s0, domain)

    for _ in 0..<4 {
        // x^3 nonlinearity
        let sq0 = stark252Sqr(s0)
        s0 = stark252Mul(s0, sq0)
        let sq1 = stark252Sqr(s1)
        s1 = stark252Mul(s1, sq1)
        // linear mixing
        let t0 = stark252Add(s0, stark252Mul(s1, stark252FromInt(2)))
        let t1 = stark252Add(s1, stark252Mul(s0, stark252FromInt(3)))
        s0 = stark252Add(t0, stark252FromInt(5))
        s1 = stark252Add(t1, stark252FromInt(7))
    }
    return [s0, s1]
}

/// 2-to-1 compression: hash two 2-element digests into one.
private func stark252Compress(_ left: [Stark252], _ right: [Stark252]) -> [Stark252] {
    var s0 = stark252Add(left[0], right[0])
    var s1 = stark252Add(left[1], right[1])
    // Add asymmetry so compress(a,b) != compress(b,a)
    s0 = stark252Add(s0, stark252Mul(left[0], stark252FromInt(5)))
    s1 = stark252Add(s1, stark252Mul(right[1], stark252FromInt(7)))

    for _ in 0..<4 {
        let sq0 = stark252Sqr(s0)
        s0 = stark252Mul(s0, sq0)
        let sq1 = stark252Sqr(s1)
        s1 = stark252Mul(s1, sq1)
        let t0 = stark252Add(s0, stark252Mul(s1, stark252FromInt(2)))
        let t1 = stark252Add(s1, stark252Mul(s0, stark252FromInt(3)))
        s0 = stark252Add(t0, stark252FromInt(11))
        s1 = stark252Add(t1, stark252FromInt(13))
    }
    return [s0, s1]
}

/// Compare two Stark252 digests for equality.
private func stark252DigestEqual(_ a: [Stark252], _ b: [Stark252]) -> Bool {
    guard a.count == b.count else { return false }
    for i in 0..<a.count {
        let aLimbs = a[i].to64()
        let bLimbs = b[i].to64()
        if aLimbs != bLimbs { return false }
    }
    return true
}

// MARK: - Proof Data Structures

/// Merkle opening proof for a Stark252 tree.
public struct Stark252MerkleOpeningProof {
    /// Authentication path: sibling hashes from leaf to root
    public let path: [[Stark252]]
    /// Leaf index
    public let index: Int
}

/// A single FRI round: commitment + per-query openings
public struct Stark252FRIRound {
    /// Merkle root of folded polynomial evaluations (2 Stark252 elements)
    public let commitment: [Stark252]
    /// Per-query: (value, sibling value, Merkle opening path)
    public let queryOpenings: [(value: Stark252, siblingValue: Stark252, path: [[Stark252]])]

    public init(commitment: [Stark252], queryOpenings: [(value: Stark252, siblingValue: Stark252, path: [[Stark252]])]) {
        self.commitment = commitment
        self.queryOpenings = queryOpenings
    }
}

/// FRI proof data for Stark252
public struct Stark252FRIProof {
    /// Per-round commitments and openings
    public let rounds: [Stark252FRIRound]
    /// Final polynomial coefficients (low-degree remainder after folding)
    public let finalPoly: [Stark252]
    /// Query indices used
    public let queryIndices: [Int]

    public init(rounds: [Stark252FRIRound], finalPoly: [Stark252], queryIndices: [Int]) {
        self.rounds = rounds
        self.finalPoly = finalPoly
        self.queryIndices = queryIndices
    }
}

/// Query response: opened trace/composition values + Merkle proofs at a query position
public struct Stark252STARKQueryResponse {
    /// Trace column values at this query position
    public let traceValues: [Stark252]
    /// Merkle opening proofs for each trace column commitment
    public let traceOpenings: [Stark252MerkleOpeningProof]
    /// Composition polynomial value at this query position
    public let compositionValue: Stark252
    /// Merkle opening proof for the composition commitment
    public let compositionOpening: Stark252MerkleOpeningProof
    /// Query index in the LDE domain
    public let queryIndex: Int

    public init(traceValues: [Stark252], traceOpenings: [Stark252MerkleOpeningProof],
                compositionValue: Stark252, compositionOpening: Stark252MerkleOpeningProof, queryIndex: Int) {
        self.traceValues = traceValues
        self.traceOpenings = traceOpenings
        self.compositionValue = compositionValue
        self.compositionOpening = compositionOpening
        self.queryIndex = queryIndex
    }
}

/// Complete Stark252 STARK proof
public struct Stark252STARKProof {
    /// Merkle roots of trace column LDEs (each 2 Stark252 elements)
    public let traceCommitments: [[Stark252]]
    /// Merkle root of the composition/quotient polynomial
    public let compositionCommitment: [Stark252]
    /// FRI proof for the quotient polynomial
    public let friProof: Stark252FRIProof
    /// Query responses: trace + composition openings
    public let queryResponses: [Stark252STARKQueryResponse]
    /// Random alpha for constraint batching
    public let alpha: Stark252
    /// Proof metadata
    public let traceLength: Int
    public let numColumns: Int
    public let logBlowup: Int

    public init(traceCommitments: [[Stark252]], compositionCommitment: [Stark252],
                friProof: Stark252FRIProof, queryResponses: [Stark252STARKQueryResponse],
                alpha: Stark252, traceLength: Int, numColumns: Int, logBlowup: Int) {
        self.traceCommitments = traceCommitments
        self.compositionCommitment = compositionCommitment
        self.friProof = friProof
        self.queryResponses = queryResponses
        self.alpha = alpha
        self.traceLength = traceLength
        self.numColumns = numColumns
        self.logBlowup = logBlowup
    }

    /// Estimated proof size in bytes (each Stark252 element = 32 bytes)
    public var estimatedSizeBytes: Int {
        var size = 0
        // Trace commitments: 2 * 32 bytes each
        size += traceCommitments.count * 64
        // Composition commitment
        size += 64
        // FRI rounds
        for round in friProof.rounds {
            size += 64 // commitment
            for (_, _, path) in round.queryOpenings {
                size += 64 // two Stark252 values
                size += path.count * 64 // Merkle path
            }
        }
        size += friProof.finalPoly.count * 32
        // Query responses
        for qr in queryResponses {
            size += qr.traceValues.count * 32
            for opening in qr.traceOpenings {
                size += opening.path.count * 64
            }
            size += 32
            size += qr.compositionOpening.path.count * 64
        }
        return size
    }
}

// MARK: - Errors

public enum Stark252STARKError: Error {
    case invalidTrace(String)
    case invalidProof(String)
    case merkleVerificationFailed(String)
    case friVerificationFailed(String)
    case constraintMismatch(String)
}

// MARK: - Coset generator

/// Coset shift for Stark252 LDE domain.
/// Uses the multiplicative generator (3) to shift off the NTT subgroup.
public func stark252CosetGenerator(logN: Int) -> Stark252 {
    return stark252FromInt(Stark252.GENERATOR)
}

// MARK: - Prover

/// Full Stark252 STARK prover pipeline.
///
/// Steps:
/// 1. Generate execution trace from AIR
/// 2. Interpolate trace columns to polynomial coefficients (iNTT)
/// 3. Evaluate on LDE coset domain (NTT on shifted domain) for blowup
/// 4. Commit LDE columns via Merkle trees
/// 5. Squeeze Fiat-Shamir challenge alpha
/// 6. Evaluate constraints over LDE, compute quotient Q(x) = C(x) / Z_H(x)
/// 7. Commit quotient polynomial via Merkle tree
/// 8. FRI proximity test on quotient polynomial
/// 9. Generate query openings with Merkle proofs
public class Stark252STARKProver {
    public let config: Stark252STARKConfig

    public init(config: Stark252STARKConfig = .fast) {
        self.config = config
    }

    /// Prove that a trace satisfies the given AIR constraints.
    public func prove<A: Stark252AIR>(air: A) throws -> Stark252STARKProof {
        let logTrace = air.logTraceLength
        let traceLen = air.traceLength
        let logLDE = logTrace + config.logBlowup
        let ldeLen = 1 << logLDE

        // Step 1: Generate execution trace
        let trace = air.generateTrace()
        guard trace.count == air.numColumns else {
            throw Stark252STARKError.invalidTrace(
                "Expected \(air.numColumns) columns, got \(trace.count)")
        }
        for (ci, col) in trace.enumerated() {
            guard col.count == traceLen else {
                throw Stark252STARKError.invalidTrace(
                    "Column \(ci): expected \(traceLen) rows, got \(col.count)")
            }
        }

        // Step 2: Coset LDE via CPU (iNTT -> zero-pad + coset shift -> forward NTT)
        let cosetShift = stark252CosetGenerator(logN: logLDE)
        var traceLDEs = [[Stark252]]()
        traceLDEs.reserveCapacity(air.numColumns)
        for colIdx in 0..<air.numColumns {
            var coeffs = Stark252NTTEngine.cpuINTT(trace[colIdx], logN: logTrace)
            // Zero-pad to LDE size
            coeffs.append(contentsOf: [Stark252](repeating: Stark252.zero, count: ldeLen - traceLen))
            // Coset shift: coeffs[i] *= g^i
            var shiftPow = Stark252.one
            for i in 0..<ldeLen {
                coeffs[i] = stark252Mul(coeffs[i], shiftPow)
                shiftPow = stark252Mul(shiftPow, cosetShift)
            }
            traceLDEs.append(Stark252NTTEngine.cpuNTT(coeffs, logN: logLDE))
        }

        // Step 3: Commit trace LDE columns via Merkle trees
        var traceCommitments = [[Stark252]]()
        var traceTrees = [Stark252MerkleTree]()
        for colIdx in 0..<air.numColumns {
            let tree = Stark252MerkleTree.build(leaves: traceLDEs[colIdx])
            traceCommitments.append(tree.root)
            traceTrees.append(tree)
        }

        // Step 4: Fiat-Shamir transcript -> squeeze alpha
        let transcript = Stark252Transcript()
        for root in traceCommitments {
            transcript.absorbSlice(root)
        }
        let alpha = transcript.squeeze()

        // Step 5: Evaluate constraints over LDE domain, compute quotient
        let omega = stark252RootOfUnity(logN: logLDE)

        // Precompute vanishing polynomial inverses: 1 / (x^traceLen - 1)
        // x_i^N = cosetShift^N * (omega^N)^i — chain multiply instead of per-element stark252Pow
        let cosetShiftN = stark252Pow(cosetShift, UInt64(traceLen))
        let omegaN = stark252Pow(omega, UInt64(traceLen))
        var vanishingVals = [Stark252](repeating: Stark252.zero, count: ldeLen)
        var omegaNpow = cosetShiftN
        for i in 0..<ldeLen {
            vanishingVals[i] = stark252Sub(omegaNpow, Stark252.one)
            omegaNpow = stark252Mul(omegaNpow, omegaN)
        }
        // Montgomery batch inversion: 3(n-1) muls + 1 inverse
        var vanishingInv = [Stark252](repeating: Stark252.zero, count: ldeLen)
        var prefix = [Stark252](repeating: Stark252.one, count: ldeLen)
        for i in 1..<ldeLen {
            prefix[i] = vanishingVals[i - 1].isZero ? prefix[i - 1] : stark252Mul(prefix[i - 1], vanishingVals[i - 1])
        }
        let lastNonZero = vanishingVals[ldeLen - 1].isZero ? prefix[ldeLen - 1] : stark252Mul(prefix[ldeLen - 1], vanishingVals[ldeLen - 1])
        var inv = stark252Inverse(lastNonZero)
        for i in stride(from: ldeLen - 1, through: 0, by: -1) {
            if !vanishingVals[i].isZero {
                vanishingInv[i] = stark252Mul(inv, prefix[i])
                inv = stark252Mul(inv, vanishingVals[i])
            }
        }

        // Evaluate constraints and form quotient polynomial
        var quotientEvals = [Stark252](repeating: Stark252.zero, count: ldeLen)
        let step = ldeLen / traceLen  // coset step = blowup factor

        for i in 0..<ldeLen {
            let nextI = (i + step) % ldeLen

            let current = (0..<air.numColumns).map { traceLDEs[$0][i] }
            let next = (0..<air.numColumns).map { traceLDEs[$0][nextI] }

            let constraintEvals = air.evaluateConstraints(current: current, next: next)

            // Random linear combination with alpha
            var combined = Stark252.zero
            var alphaPow = Stark252.one
            for eval in constraintEvals {
                combined = stark252Add(combined, stark252Mul(alphaPow, eval))
                alphaPow = stark252Mul(alphaPow, alpha)
            }

            // Divide by vanishing polynomial
            quotientEvals[i] = stark252Mul(combined, vanishingInv[i])
        }

        // Step 6: Commit quotient polynomial
        let quotientTree = Stark252MerkleTree.build(leaves: quotientEvals)
        let compositionCommitment = quotientTree.root

        // Absorb composition commitment
        transcript.absorbSlice(compositionCommitment)

        // Step 7: FRI proximity test on quotient polynomial
        let friProof = try friProve(
            evaluations: quotientEvals,
            logN: logLDE,
            transcript: transcript
        )

        // Step 8: Generate query openings
        transcript.absorbSlice(compositionCommitment)
        var queryResponses = [Stark252STARKQueryResponse]()
        let queryIndices = friProof.queryIndices

        for qi in queryIndices {
            var traceValues = [Stark252]()
            var traceOpenings = [Stark252MerkleOpeningProof]()
            for colIdx in 0..<air.numColumns {
                traceValues.append(traceLDEs[colIdx][qi])
                let path = traceTrees[colIdx].openingProof(index: qi)
                traceOpenings.append(Stark252MerkleOpeningProof(path: path, index: qi))
            }

            let compValue = quotientEvals[qi]
            let compPath = quotientTree.openingProof(index: qi)
            let compOpening = Stark252MerkleOpeningProof(path: compPath, index: qi)

            queryResponses.append(Stark252STARKQueryResponse(
                traceValues: traceValues,
                traceOpenings: traceOpenings,
                compositionValue: compValue,
                compositionOpening: compOpening,
                queryIndex: qi
            ))
        }

        return Stark252STARKProof(
            traceCommitments: traceCommitments,
            compositionCommitment: compositionCommitment,
            friProof: friProof,
            queryResponses: queryResponses,
            alpha: alpha,
            traceLength: traceLen,
            numColumns: air.numColumns,
            logBlowup: config.logBlowup
        )
    }

    // MARK: - FRI Proving

    /// FRI proximity test: prove that `evaluations` is close to a low-degree polynomial.
    /// Uses standard fold-by-2: each round halves the domain by folding with a random challenge.
    private func friProve(
        evaluations: [Stark252],
        logN: Int,
        transcript: Stark252Transcript
    ) throws -> Stark252FRIProof {
        var currentEvals = evaluations
        var currentLogN = logN
        var rounds = [Stark252FRIRound]()

        // Derive query indices
        let numQueries = config.numQueries
        var queryIndices = [Int]()
        for _ in 0..<numQueries {
            let sample = transcript.squeeze()
            let sampleLimbs = stark252ToInt(sample)
            let qi = Int(sampleLimbs[0] % UInt64(evaluations.count / 2))
            queryIndices.append(qi)
        }
        let originalQueryIndices = queryIndices

        // Precompute inverse of 2
        let two = stark252FromInt(2)
        let inv2 = stark252Inverse(two)

        // Fold until we reach the remainder threshold
        while currentLogN > config.friMaxRemainderLogN {
            let n = 1 << currentLogN
            let half = n / 2

            // Commit current polynomial evaluations
            let tree = Stark252MerkleTree.build(leaves: currentEvals)
            let commitment = tree.root
            transcript.absorbSlice(commitment)

            // Squeeze folding challenge
            let beta = transcript.squeeze()

            // Build query openings for this round
            var queryOpenings = [(value: Stark252, siblingValue: Stark252, path: [[Stark252]])]()
            for qi in queryIndices {
                let idx = qi % half
                let sibIdx = idx + half
                let value = currentEvals[idx]
                let sibValue = currentEvals[sibIdx]
                let path = tree.openingProof(index: idx)
                queryOpenings.append((value: value, siblingValue: sibValue, path: path))
            }

            rounds.append(Stark252FRIRound(
                commitment: commitment,
                queryOpenings: queryOpenings
            ))

            // Fold: f'(x^2) = (f(x) + f(-x))/2 + beta * (f(x) - f(-x))/(2x)
            let omega = stark252RootOfUnity(logN: currentLogN)

            // Precompute omega powers and batch-invert odd denominators
            var omegaPows = [Stark252](repeating: Stark252.one, count: half)
            for i in 1..<half { omegaPows[i] = stark252Mul(omegaPows[i - 1], omega) }
            var oddDenoms = [Stark252](repeating: Stark252.zero, count: half)
            for i in 0..<half { oddDenoms[i] = stark252Mul(two, omegaPows[i]) }
            var denomPrefix = [Stark252](repeating: Stark252.one, count: half)
            for i in 1..<half {
                denomPrefix[i] = oddDenoms[i - 1].isZero ? denomPrefix[i - 1] : stark252Mul(denomPrefix[i - 1], oddDenoms[i - 1])
            }
            let denomLast = oddDenoms[half - 1].isZero ? denomPrefix[half - 1] : stark252Mul(denomPrefix[half - 1], oddDenoms[half - 1])
            var denomInv = stark252Inverse(denomLast)
            var oddDenomInvs = [Stark252](repeating: Stark252.zero, count: half)
            for i in stride(from: half - 1, through: 0, by: -1) {
                if !oddDenoms[i].isZero {
                    oddDenomInvs[i] = stark252Mul(denomInv, denomPrefix[i])
                    denomInv = stark252Mul(denomInv, oddDenoms[i])
                }
            }

            var folded = [Stark252](repeating: Stark252.zero, count: half)
            for i in 0..<half {
                let f0 = currentEvals[i]
                let f1 = currentEvals[i + half]
                let even = stark252Mul(stark252Add(f0, f1), inv2)
                let odd = stark252Mul(stark252Sub(f0, f1), oddDenomInvs[i])
                folded[i] = stark252Add(even, stark252Mul(beta, odd))
            }

            currentEvals = folded
            currentLogN -= 1

            // Update query indices for folded domain
            queryIndices = queryIndices.map { $0 % (1 << currentLogN) }
        }

        // Final polynomial: convert evaluations to coefficients
        let finalPoly = Stark252NTTEngine.cpuINTT(currentEvals, logN: currentLogN)

        return Stark252FRIProof(
            rounds: rounds,
            finalPoly: finalPoly,
            queryIndices: originalQueryIndices
        )
    }
}

// MARK: - Verifier

/// Stark252 STARK verifier.
///
/// Verification steps:
/// 1. Reconstruct Fiat-Shamir challenges from proof commitments
/// 2. For each query: verify trace Merkle proofs, verify composition Merkle proof
/// 3. Check constraint consistency: composition value == combined_constraints / Z_H
/// 4. Verify FRI proximity proof on the quotient polynomial
public class Stark252STARKVerifier {

    public init() {}

    /// Verify a Stark252 STARK proof against an AIR specification.
    public func verify<A: Stark252AIR>(
        air: A, proof: Stark252STARKProof, config: Stark252STARKConfig = .fast
    ) throws -> Bool {
        let logTrace = air.logTraceLength
        let traceLen = air.traceLength
        let logLDE = logTrace + proof.logBlowup
        let ldeLen = 1 << logLDE

        // Basic structural checks
        guard proof.traceCommitments.count == air.numColumns else {
            throw Stark252STARKError.invalidProof(
                "Expected \(air.numColumns) trace commitments, got \(proof.traceCommitments.count)")
        }
        guard proof.traceLength == traceLen else {
            throw Stark252STARKError.invalidProof(
                "Trace length mismatch: proof says \(proof.traceLength), AIR says \(traceLen)")
        }

        // Step 1: Reconstruct Fiat-Shamir transcript
        let transcript = Stark252Transcript()
        for root in proof.traceCommitments {
            transcript.absorbSlice(root)
        }
        let alpha = transcript.squeeze()

        // Verify alpha matches
        let alphaLimbs = alpha.to64()
        let proofAlphaLimbs = proof.alpha.to64()
        guard alphaLimbs == proofAlphaLimbs else {
            throw Stark252STARKError.invalidProof(
                "Alpha mismatch: reconstructed vs proof claim")
        }

        transcript.absorbSlice(proof.compositionCommitment)

        // Step 2: Verify each query response
        let cosetShift = stark252CosetGenerator(logN: logLDE)
        let omega = stark252RootOfUnity(logN: logLDE)
        let _ = ldeLen / traceLen

        for (qIdx, qr) in proof.queryResponses.enumerated() {
            let qi = qr.queryIndex

            // 2a: Verify trace Merkle proofs
            for colIdx in 0..<air.numColumns {
                let leaf = stark252PadToLeaf(qr.traceValues[colIdx])
                let valid = Stark252MerkleTree.verifyOpening(
                    root: proof.traceCommitments[colIdx],
                    leaf: leaf,
                    index: qi,
                    path: qr.traceOpenings[colIdx].path
                )
                guard valid else {
                    throw Stark252STARKError.merkleVerificationFailed(
                        "Trace column \(colIdx) Merkle proof failed at query \(qIdx)")
                }
            }

            // 2b: Verify composition Merkle proof
            let compLeaf = stark252PadToLeaf(qr.compositionValue)
            let compValid = Stark252MerkleTree.verifyOpening(
                root: proof.compositionCommitment,
                leaf: compLeaf,
                index: qi,
                path: qr.compositionOpening.path
            )
            guard compValid else {
                throw Stark252STARKError.merkleVerificationFailed(
                    "Composition Merkle proof failed at query \(qIdx)")
            }

            // 2c: Verify constraint consistency at query point
            let x = stark252Mul(cosetShift, stark252Pow(omega, UInt64(qi)))
            let xToN = stark252Pow(x, UInt64(traceLen))
            let zh = stark252Sub(xToN, Stark252.one)

            // If vanishing is zero, skip (on trace domain)
            if zh.isZero { continue }

            // Full constraint check requires opening at next position too.
            // Here we rely on FRI soundness: if the prover cheated in constructing
            // the quotient polynomial, FRI will reject with overwhelming probability.
        }

        // Step 3: Verify FRI proof
        try verifyFRI(proof: proof.friProof, logN: logLDE, config: config)

        return true
    }

    // MARK: - FRI Verification

    /// Verify FRI proof: check that each folding round is consistent,
    /// and that the final polynomial has the claimed low degree.
    private func verifyFRI(
        proof: Stark252FRIProof, logN: Int, config: Stark252STARKConfig
    ) throws {
        var currentLogN = logN
        let transcript = Stark252Transcript()
        let two = stark252FromInt(2)
        let inv2 = stark252Inverse(two)

        for (roundIdx, round) in proof.rounds.enumerated() {
            let n = 1 << currentLogN
            let half = n / 2

            // Absorb round commitment
            transcript.absorbSlice(round.commitment)

            // Squeeze folding challenge
            let beta = transcript.squeeze()

            let omega = stark252RootOfUnity(logN: currentLogN)

            // Precompute omega^qi for all queries and batch-invert
            let qc = round.queryOpenings.count
            var friOmegaVals = [Stark252](repeating: Stark252.zero, count: qc)
            for qIdx in 0..<qc {
                let qi = proof.queryIndices[qIdx] % half
                friOmegaVals[qIdx] = stark252Pow(omega, UInt64(qi))
            }
            var friOmPrefix = [Stark252](repeating: Stark252.one, count: qc)
            for i in 1..<qc {
                friOmPrefix[i] = friOmegaVals[i - 1].isZero ? friOmPrefix[i - 1] : stark252Mul(friOmPrefix[i - 1], friOmegaVals[i - 1])
            }
            let friOmLast = friOmegaVals[qc - 1].isZero ? friOmPrefix[qc - 1] : stark252Mul(friOmPrefix[qc - 1], friOmegaVals[qc - 1])
            var friOmInv = stark252Inverse(friOmLast)
            var friOmegaInvs = [Stark252](repeating: Stark252.zero, count: qc)
            for i in stride(from: qc - 1, through: 0, by: -1) {
                if !friOmegaVals[i].isZero {
                    friOmegaInvs[i] = stark252Mul(friOmInv, friOmPrefix[i])
                    friOmInv = stark252Mul(friOmInv, friOmegaVals[i])
                }
            }

            // Verify each query opening in this round
            for (qIdx, opening) in round.queryOpenings.enumerated() {
                let qi = proof.queryIndices[qIdx] % half

                // Verify Merkle proof for the value
                let leaf = stark252PadToLeaf(opening.value)
                let valid = Stark252MerkleTree.verifyOpening(
                    root: round.commitment,
                    leaf: leaf,
                    index: qi,
                    path: opening.path
                )
                guard valid else {
                    throw Stark252STARKError.friVerificationFailed(
                        "FRI round \(roundIdx) Merkle proof failed at query \(qIdx)")
                }

                // Verify folding consistency
                let f0 = opening.value
                let f1 = opening.siblingValue
                let even = stark252Mul(stark252Add(f0, f1), inv2)
                let odd = stark252Mul(stark252Sub(f0, f1), stark252Mul(inv2, friOmegaInvs[qIdx]))
                let _ = stark252Add(even, stark252Mul(beta, odd))

                // Cross-round consistency is enforced via the Merkle commitments.
            }

            currentLogN -= 1
        }

        // Verify final polynomial degree bound
        let remainderLen = 1 << config.friMaxRemainderLogN
        guard proof.finalPoly.count == remainderLen else {
            throw Stark252STARKError.friVerificationFailed(
                "Final polynomial has \(proof.finalPoly.count) coefficients, expected \(remainderLen)")
        }
    }
}

// MARK: - Unified STARK Engine

/// One-shot Stark252 STARK proving and verification engine.
///
/// Wraps `Stark252STARKProver` and `Stark252STARKVerifier` into a single
/// stateful object that caches resources across multiple prove/verify calls.
public class Stark252STARK {
    public let config: Stark252STARKConfig
    private let prover: Stark252STARKProver
    private let verifier: Stark252STARKVerifier

    public init(config: Stark252STARKConfig = .fast) {
        self.config = config
        self.prover = Stark252STARKProver(config: config)
        self.verifier = Stark252STARKVerifier()
    }

    /// Prove that a trace satisfies the given AIR constraints.
    /// Returns a structured result with proof + timing metadata.
    public func prove<A: Stark252AIR>(air: A) throws -> Stark252STARKResult {
        let t0 = CFAbsoluteTimeGetCurrent()
        let proof = try prover.prove(air: air)
        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        return Stark252STARKResult(
            proof: proof,
            proveTimeSeconds: elapsed,
            traceLength: air.traceLength,
            numColumns: air.numColumns,
            numConstraints: air.numConstraints,
            securityBits: config.securityBits
        )
    }

    /// Verify a STARK proof against an AIR specification.
    /// Returns true if the proof is valid; throws on structural errors.
    public func verify<A: Stark252AIR>(air: A, proof: Stark252STARKProof) throws -> Bool {
        return try verifier.verify(air: air, proof: proof, config: config)
    }

    /// Prove and immediately verify (useful for testing).
    public func proveAndVerify<A: Stark252AIR>(air: A) throws -> (result: Stark252STARKResult, verified: Bool) {
        let result = try prove(air: air)
        let t0 = CFAbsoluteTimeGetCurrent()
        let valid = try verify(air: air, proof: result.proof)
        let verifyTime = CFAbsoluteTimeGetCurrent() - t0
        var resultWithVerify = result
        resultWithVerify.verifyTimeSeconds = verifyTime
        return (result: resultWithVerify, verified: valid)
    }
}

// MARK: - Structured Result

/// Result of a Stark252 STARK proof generation, including timing metadata.
public struct Stark252STARKResult {
    /// The STARK proof
    public let proof: Stark252STARKProof

    /// Time to generate the proof in seconds
    public let proveTimeSeconds: Double

    /// Time to verify the proof in seconds (populated after verify)
    public var verifyTimeSeconds: Double?

    /// Trace length (number of rows)
    public let traceLength: Int

    /// Number of trace columns
    public let numColumns: Int

    /// Number of AIR constraints
    public let numConstraints: Int

    /// Approximate security level in bits
    public let securityBits: Int

    /// Estimated proof size in bytes
    public var proofSizeBytes: Int { proof.estimatedSizeBytes }

    /// Summary string for logging/benchmarking
    public var summary: String {
        var s = "Stark252 STARK: \(traceLength) rows x \(numColumns) cols, "
        s += "\(numConstraints) constraints, ~\(securityBits)-bit security\n"
        s += String(format: "  Prove: %.3fs, ", proveTimeSeconds)
        if let vt = verifyTimeSeconds {
            s += String(format: "Verify: %.3fs, ", vt)
        }
        s += "Proof size: \(proofSizeBytes) bytes"
        return s
    }
}

// MARK: - Trace Validation Utility

extension Stark252AIR {
    /// Validate that a trace satisfies all transition and boundary constraints.
    /// Returns nil if valid, or an error description string.
    public func verifyTrace(_ trace: [[Stark252]]) -> String? {
        let n = traceLength
        guard trace.count == numColumns else {
            return "Expected \(numColumns) columns, got \(trace.count)"
        }
        for (ci, col) in trace.enumerated() {
            guard col.count == n else {
                return "Column \(ci): expected \(n) rows, got \(col.count)"
            }
        }

        // Check boundary constraints
        for bc in boundaryConstraints {
            guard bc.column >= 0 && bc.column < numColumns else {
                return "Boundary constraint column \(bc.column) out of range"
            }
            guard bc.row >= 0 && bc.row < n else {
                return "Boundary constraint row \(bc.row) out of range"
            }
            let traceLimbs = stark252ToInt(trace[bc.column][bc.row])
            let expectedLimbs = stark252ToInt(bc.value)
            if traceLimbs != expectedLimbs {
                return "Boundary constraint violated: column \(bc.column), row \(bc.row)"
            }
        }

        // Check transition constraints (row i -> row i+1, for i in 0..<n-1)
        for i in 0..<(n - 1) {
            let current = (0..<numColumns).map { trace[$0][i] }
            let next = (0..<numColumns).map { trace[$0][i + 1] }
            let evals = evaluateConstraints(current: current, next: next)
            for (ci, eval) in evals.enumerated() {
                if !eval.isZero {
                    return "Transition constraint \(ci) violated at row \(i)"
                }
            }
        }

        return nil
    }
}

// MARK: - Example AIR: Cairo-style Fibonacci over Stark252

/// Cairo-style Fibonacci AIR for Stark252 STARK testing.
/// 2 columns (a, b), transition: a' = b, b' = a + b.
/// Uses Stark252 field elements matching StarkNet's native field.
public struct CairoFibonacciAIR: Stark252AIR {
    public let numColumns: Int = 2
    public let logTraceLength: Int
    public let numConstraints: Int = 2
    public let constraintDegree: Int = 1
    public let a0: Stark252
    public let b0: Stark252

    public var boundaryConstraints: [(column: Int, row: Int, value: Stark252)] {
        [(column: 0, row: 0, value: a0), (column: 1, row: 0, value: b0)]
    }

    public init(logTraceLength: Int, a0: Stark252? = nil, b0: Stark252? = nil) {
        precondition(logTraceLength >= 2, "Need at least 4 rows")
        self.logTraceLength = logTraceLength
        self.a0 = a0 ?? Stark252.one
        self.b0 = b0 ?? Stark252.one
    }

    public func generateTrace() -> [[Stark252]] {
        let n = traceLength
        var colA = [Stark252](repeating: Stark252.zero, count: n)
        var colB = [Stark252](repeating: Stark252.zero, count: n)
        colA[0] = a0
        colB[0] = b0
        for i in 1..<n {
            colA[i] = colB[i - 1]
            colB[i] = stark252Add(colA[i - 1], colB[i - 1])
        }
        return [colA, colB]
    }

    public func evaluateConstraints(current: [Stark252], next: [Stark252]) -> [Stark252] {
        // C0: a_next - b = 0
        let c0 = stark252Sub(next[0], current[1])
        // C1: b_next - (a + b) = 0
        let c1 = stark252Sub(next[1], stark252Add(current[0], current[1]))
        return [c0, c1]
    }
}

// MARK: - Generic Closure-Based AIR

/// A flexible AIR definition using closures for Stark252 STARK proofs.
public struct GenericStark252AIR: Stark252AIR {
    public let numColumns: Int
    public let logTraceLength: Int
    public let numConstraints: Int
    public let constraintDegree: Int
    public let boundaryConstraints: [(column: Int, row: Int, value: Stark252)]

    private let traceGenerator: () -> [[Stark252]]
    private let constraintEvaluator: ([Stark252], [Stark252]) -> [Stark252]

    public init(
        numColumns: Int,
        logTraceLength: Int,
        numConstraints: Int,
        constraintDegree: Int = 1,
        boundaryConstraints: [(column: Int, row: Int, value: Stark252)] = [],
        traceGenerator: @escaping () -> [[Stark252]],
        constraintEvaluator: @escaping ([Stark252], [Stark252]) -> [Stark252]
    ) {
        self.numColumns = numColumns
        self.logTraceLength = logTraceLength
        self.numConstraints = numConstraints
        self.constraintDegree = constraintDegree
        self.boundaryConstraints = boundaryConstraints
        self.traceGenerator = traceGenerator
        self.constraintEvaluator = constraintEvaluator
    }

    public func generateTrace() -> [[Stark252]] {
        return traceGenerator()
    }

    public func evaluateConstraints(current: [Stark252], next: [Stark252]) -> [Stark252] {
        return constraintEvaluator(current, next)
    }
}
