// Circle STARK Prover — Produces STARK proofs using circle group over M31
//
// Protocol:
// 1. Trace generation + LDE via Circle NTT on evaluation domain (blowup factor)
// 2. Commit trace columns via Keccak-256 Merkle trees
// 3. Fiat-Shamir challenge alpha, evaluate composition polynomial
// 4. Commit composition polynomial
// 5. Circle FRI to prove low degree
// 6. Query phase: open trace + composition at FRI query positions

import Foundation

// MARK: - Proof Data Structures

/// A single query response: opened values and Merkle authentication paths
public struct CircleSTARKQueryResponse {
    /// Trace values at this query position: [column] of M31
    public let traceValues: [M31]
    /// Trace Merkle authentication paths: [column] of path (each path is array of 32-byte hashes)
    public let tracePaths: [[[UInt8]]]
    /// Composition polynomial value at this query position
    public let compositionValue: M31
    /// Composition Merkle authentication path
    public let compositionPath: [[UInt8]]
    /// The query index in the evaluation domain
    public let queryIndex: Int
}

/// Circle FRI proof data (self-contained CPU implementation)
public struct CircleFRIProofData {
    /// Per-round data: commitment root + query responses
    public let rounds: [CircleFRIRound]
    /// Final constant value after all folding rounds
    public let finalValue: M31
    /// Query indices used
    public let queryIndices: [Int]
}

/// One round of Circle FRI
public struct CircleFRIRound {
    /// Merkle root of the folded polynomial evaluations
    public let commitment: [UInt8]
    /// For each query: (value at query, value at sibling, Merkle path)
    public let queryResponses: [(M31, M31, [[UInt8]])]
}

/// Complete Circle STARK proof
public struct CircleSTARKProof {
    /// Merkle roots of trace column LDEs
    public let traceCommitments: [[UInt8]]
    /// Merkle root of composition polynomial
    public let compositionCommitment: [UInt8]
    /// FRI proof for the composition polynomial
    public let friProof: CircleFRIProofData
    /// Query responses: trace + composition openings at query positions
    public let queryResponses: [CircleSTARKQueryResponse]
    /// Random alpha used for constraint batching
    public let alpha: M31
    /// Proof metadata
    public let traceLength: Int
    public let numColumns: Int
    public let logBlowup: Int

    public init(traceCommitments: [[UInt8]], compositionCommitment: [UInt8],
                friProof: CircleFRIProofData, queryResponses: [CircleSTARKQueryResponse],
                alpha: M31, traceLength: Int, numColumns: Int, logBlowup: Int) {
        self.traceCommitments = traceCommitments
        self.compositionCommitment = compositionCommitment
        self.friProof = friProof
        self.queryResponses = queryResponses
        self.alpha = alpha
        self.traceLength = traceLength
        self.numColumns = numColumns
        self.logBlowup = logBlowup
    }
}

// MARK: - Prover

public class CircleSTARKProver {
    public static let version = Versions.circleSTARK

    /// Log2 of blowup factor (2 means 4x blowup, 4 means 16x)
    public let logBlowup: Int
    /// Number of FRI queries for soundness
    public let numQueries: Int

    public init(logBlowup: Int = 4, numQueries: Int = 30) {
        self.logBlowup = logBlowup
        self.numQueries = numQueries
    }

    /// Prove that the given AIR is satisfied.
    /// Returns a CircleSTARKProof that can be verified independently.
    public func prove<A: CircleAIR>(air: A) throws -> CircleSTARKProof {
        let traceLen = air.traceLength
        let logTrace = air.logTraceLength
        let logEval = logTrace + logBlowup
        let evalLen = 1 << logEval

        // Step 1: Generate trace
        let trace = air.generateTrace()
        precondition(trace.count == air.numColumns)
        for col in trace {
            precondition(col.count == traceLen)
        }

        // Step 2: LDE - extend trace columns to evaluation domain via Circle NTT
        // For each column: INTT to get coefficients, pad to eval domain size, NTT to get evaluations
        var traceLDEs = [[M31]]()
        traceLDEs.reserveCapacity(air.numColumns)

        for col in trace {
            let coeffs = CircleNTTEngine.cpuINTT(col, logN: logTrace)
            var padded = [M31](repeating: M31.zero, count: evalLen)
            for i in 0..<traceLen {
                padded[i] = coeffs[i]
            }
            let evals = CircleNTTEngine.cpuNTT(padded, logN: logEval)
            traceLDEs.append(evals)
        }

        // Step 3: Commit trace columns via Keccak Merkle trees
        var traceTrees = [[[UInt8]]]()
        var traceCommitments = [[UInt8]]()
        for col in traceLDEs {
            let leafHashes = col.map { keccak256(m31ToBytes($0)) }
            let tree = buildKeccakMerkle(leafHashes)
            traceCommitments.append(tree[1])  // root at index 1
            traceTrees.append(tree)
        }

        // Step 4: Fiat-Shamir - derive alpha from trace commitments
        var transcript = CircleSTARKTranscript()
        transcript.absorbLabel("circle-stark-v1")
        for root in traceCommitments {
            transcript.absorbBytes(root)
        }
        let alpha = transcript.squeezeM31()

        // Step 5: Evaluate composition polynomial on evaluation domain
        // C(P) = sum_i alpha^i * C_i(trace(P)) / Z(P)
        let compositionEvals = evaluateComposition(
            air: air, traceLDEs: traceLDEs, alpha: alpha,
            logTrace: logTrace, logEval: logEval
        )

        // Step 6: Commit composition polynomial
        let compLeafHashes = compositionEvals.map { keccak256(m31ToBytes($0)) }
        let compTree = buildKeccakMerkle(compLeafHashes)
        let compositionCommitment = compTree[1]

        // Step 7: FRI on composition polynomial
        transcript.absorbBytes(compositionCommitment)
        let friProof = try circleFRI(
            evals: compositionEvals, logN: logEval,
            numQueries: numQueries, transcript: &transcript
        )

        // Step 8: Query phase - open trace + composition at FRI query positions
        var queryResponses = [CircleSTARKQueryResponse]()
        for qi in friProof.queryIndices {
            guard qi < evalLen else { continue }
            var traceVals = [M31]()
            var tracePaths = [[[UInt8]]]()
            for colIdx in 0..<traceLDEs.count {
                traceVals.append(traceLDEs[colIdx][qi])
                tracePaths.append(merkleProof(tree: traceTrees[colIdx], index: qi))
            }
            let compVal = compositionEvals[qi]
            let compPath = merkleProof(tree: compTree, index: qi)

            queryResponses.append(CircleSTARKQueryResponse(
                traceValues: traceVals,
                tracePaths: tracePaths,
                compositionValue: compVal,
                compositionPath: compPath,
                queryIndex: qi
            ))
        }

        return CircleSTARKProof(
            traceCommitments: traceCommitments,
            compositionCommitment: compositionCommitment,
            friProof: friProof,
            queryResponses: queryResponses,
            alpha: alpha,
            traceLength: traceLen,
            numColumns: air.numColumns,
            logBlowup: logBlowup
        )
    }

    // MARK: - Composition Polynomial Evaluation

    /// Evaluate the composition polynomial on the evaluation domain.
    /// Composition = sum_i alpha^i * constraint_i(trace) / vanishing(domain)
    private func evaluateComposition<A: CircleAIR>(
        air: A, traceLDEs: [[M31]], alpha: M31,
        logTrace: Int, logEval: Int
    ) -> [M31] {
        let evalLen = 1 << logEval
        let traceLen = 1 << logTrace

        // Precompute evaluation domain
        let evalDomain = circleCosetDomain(logN: logEval)

        // Precompute vanishing polynomial evaluations
        // v_0(P) = P.y, v_{k+1}(P) = 2*v_k(P)^2 - 1 (Chebyshev doubling)
        var vanishing = [M31](repeating: M31.zero, count: evalLen)
        for i in 0..<evalLen {
            vanishing[i] = circleVanishing(point: evalDomain[i], logDomainSize: logTrace)
        }

        var composition = [M31](repeating: M31.zero, count: evalLen)

        for i in 0..<evalLen {
            // "Next" point in LDE corresponds to shift by trace domain generator
            let nextIdx = (i + (evalLen / traceLen)) % evalLen

            var current = [M31]()
            var next = [M31]()
            for col in traceLDEs {
                current.append(col[i])
                next.append(col[nextIdx])
            }

            let constraintVals = air.evaluateConstraints(current: current, next: next)

            let vz = vanishing[i]
            if vz.v == 0 {
                // On trace domain: constraints are zero, quotient defined by continuity
                composition[i] = M31.zero
                continue
            }

            let invVz = m31Inverse(vz)
            var acc = M31.zero
            var alphaPow = M31.one
            for cv in constraintVals {
                let term = m31Mul(alphaPow, m31Mul(cv, invVz))
                acc = m31Add(acc, term)
                alphaPow = m31Mul(alphaPow, alpha)
            }

            // Boundary constraint contributions
            for bc in air.boundaryConstraints {
                let bcNum = m31Sub(traceLDEs[bc.column][i], bc.value)
                let bcTerm = m31Mul(alphaPow, m31Mul(bcNum, invVz))
                acc = m31Add(acc, bcTerm)
                alphaPow = m31Mul(alphaPow, alpha)
            }

            composition[i] = acc
        }

        return composition
    }

    // MARK: - Circle FRI (CPU Implementation)

    /// Circle FRI: prove that a polynomial (given as evaluations) has low degree.
    /// Uses the circle folding formula:
    ///   g[i] = (f[i] + f[i+n/2])/2 + alpha * (f[i] - f[i+n/2]) / (2 * twiddle[i])
    /// Layer 0 uses y-coordinate twiddles, subsequent layers use x-coordinates.
    private func circleFRI(
        evals: [M31], logN: Int, numQueries: Int,
        transcript: inout CircleSTARKTranscript
    ) throws -> CircleFRIProofData {
        var currentEvals = evals
        var currentLogN = logN
        var rounds = [CircleFRIRound]()

        // Generate initial query indices from transcript
        transcript.absorbLabel("fri-queries")
        var queryIndices = [Int]()
        let evalLen = 1 << logN
        for _ in 0..<numQueries {
            let qi = Int(transcript.squeezeM31().v) % (evalLen / 2)
            queryIndices.append(qi)
        }

        var currentQueryIndices = queryIndices

        // Fold until we reach 2 elements
        let targetLogN = 1
        while currentLogN > targetLogN {
            let n = 1 << currentLogN
            let half = n / 2

            let foldAlpha = transcript.squeezeM31()

            // Compute twiddles for this fold
            let domain = circleCosetDomain(logN: currentLogN)
            var twiddles = [M31](repeating: M31.zero, count: half)
            if rounds.isEmpty {
                // First fold uses y-coordinates
                for i in 0..<half { twiddles[i] = domain[i].y }
            } else {
                // Subsequent folds use x-coordinates
                for i in 0..<half { twiddles[i] = domain[i].x }
            }

            // Fold
            let inv2 = m31Inverse(M31(v: 2))
            var folded = [M31](repeating: M31.zero, count: half)
            for i in 0..<half {
                let fi = currentEvals[i]
                let fih = currentEvals[i + half]
                let sum = m31Mul(m31Add(fi, fih), inv2)
                let invTw2 = m31Mul(inv2, m31Inverse(twiddles[i]))
                let diff = m31Mul(m31Sub(fi, fih), invTw2)
                folded[i] = m31Add(sum, m31Mul(foldAlpha, diff))
            }

            // Commit folded evaluations
            let foldedLeaves = folded.map { keccak256(m31ToBytes($0)) }
            let foldedTree = buildKeccakMerkle(foldedLeaves)
            let commitment = foldedTree[1]
            transcript.absorbBytes(commitment)

            // Query responses for this round
            var roundQueries = [(M31, M31, [[UInt8]])]()
            for qi in currentQueryIndices {
                let idx = qi % half
                let val = currentEvals[idx]
                let sibVal = currentEvals[idx + half]
                let path = merkleProof(tree: foldedTree, index: idx)
                roundQueries.append((val, sibVal, path))
            }

            rounds.append(CircleFRIRound(
                commitment: commitment,
                queryResponses: roundQueries
            ))

            currentEvals = folded
            currentLogN -= 1
            currentQueryIndices = currentQueryIndices.map { $0 % max(half / 2, 1) }
        }

        let finalValue = currentEvals[0]

        return CircleFRIProofData(
            rounds: rounds,
            finalValue: finalValue,
            queryIndices: queryIndices
        )
    }
}

// MARK: - Circle Domain Utilities

/// Evaluate the vanishing polynomial for a circle domain of size 2^logDomainSize at a point.
/// Applies the fold/squaring map logDomainSize times:
///   v_0 = point.y, v_{k+1} = 2 * v_k^2 - 1 (Chebyshev doubling)
public func circleVanishing(point: CirclePoint, logDomainSize: Int) -> M31 {
    var v = point.y
    for _ in 0..<logDomainSize {
        v = m31Sub(m31Add(m31Sqr(v), m31Sqr(v)), M31.one)
    }
    return v
}

// MARK: - Keccak Merkle Tree Utilities

/// Serialize an M31 element to 4 bytes (little-endian)
@inline(__always)
func m31ToBytes(_ v: M31) -> [UInt8] {
    var val = v.v
    return withUnsafeBytes(of: &val) { Array($0) }
}

/// Build a Keccak-256 Merkle tree from leaf hashes.
/// Returns flat array: [unused_slot, root, ...internal..., leaves]
/// Index 1 = root, indices n..2n-1 = leaves
func buildKeccakMerkle(_ leafHashes: [[UInt8]]) -> [[UInt8]] {
    let n = leafHashes.count
    precondition(n > 0 && (n & (n - 1)) == 0, "leaf count must be power of 2")
    var tree = [[UInt8]](repeating: [UInt8](repeating: 0, count: 32), count: 2 * n)
    for i in 0..<n {
        tree[n + i] = leafHashes[i]
    }
    var levelSize = n / 2
    var offset = n / 2
    while levelSize >= 1 {
        for i in 0..<levelSize {
            let idx = offset + i
            tree[idx] = keccak256(tree[2 * idx] + tree[2 * idx + 1])
        }
        levelSize /= 2
        offset /= 2
    }
    return tree
}

/// Get Merkle authentication path for leaf at `index`.
func merkleProof(tree: [[UInt8]], index: Int) -> [[UInt8]] {
    let n = tree.count / 2
    var path = [[UInt8]]()
    var idx = n + index
    while idx > 1 {
        let sibling = idx ^ 1
        path.append(tree[sibling])
        idx /= 2
    }
    return path
}

/// Verify a Merkle proof: given leaf hash, path, index, and expected root.
func verifyMerkleProof(leafHash: [UInt8], path: [[UInt8]], index: Int, root: [UInt8]) -> Bool {
    var current = leafHash
    var idx = index
    for sibling in path {
        if idx & 1 == 0 {
            current = keccak256(current + sibling)
        } else {
            current = keccak256(sibling + current)
        }
        idx /= 2
    }
    return current == root
}

// MARK: - Lightweight Circle STARK Transcript (M31-native)

/// A simple Fiat-Shamir transcript for Circle STARKs, operating over M31.
/// Uses Keccak-256 internally for hashing.
public struct CircleSTARKTranscript {
    private var state: [UInt8]

    public init() {
        self.state = [UInt8](repeating: 0, count: 32)
    }

    public mutating func absorbLabel(_ label: String) {
        let bytes = Array(label.utf8)
        var len = UInt32(bytes.count)
        let lenBytes = withUnsafeBytes(of: &len) { Array($0) }
        absorbBytes(lenBytes + bytes)
    }

    public mutating func absorbBytes(_ data: [UInt8]) {
        state = keccak256(state + data)
    }

    public mutating func absorbM31(_ v: M31) {
        absorbBytes(m31ToBytes(v))
    }

    /// Squeeze an M31 challenge from the transcript
    public mutating func squeezeM31() -> M31 {
        state = keccak256(state + [0x01])  // domain-separate squeeze
        let raw = UInt32(state[0]) | (UInt32(state[1]) << 8) |
                  (UInt32(state[2]) << 16) | (UInt32(state[3]) << 24)
        let val = raw & M31.P  // reduce to [0, p)
        return M31(v: val == M31.P ? 0 : val)
    }
}
