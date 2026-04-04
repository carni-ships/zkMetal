// Sparse Sumcheck — Exploits sparsity in multilinear polynomials
// When most evaluations over {0,1}^k are zero, operations take O(s) instead of O(2^k)
// where s is the number of non-zero entries.
//
// Use cases: GKR inner products, Lasso subtable lookups, sparse constraint polynomials
//
// Layout convention: same as dense SumcheckEngine (MSB-first variable elimination).
// Index i corresponds to evaluation at the binary point (bit k-1 of i, ..., bit 0 of i).
// First half (indices 0..2^(k-1)-1) has MSB=0, second half has MSB=1.
//
// Optimization: uses sorted arrays of SparseEntry instead of Dictionary for
// cache-friendly sequential iteration and two-pointer merge (no hash overhead).

import Foundation

/// A single non-zero entry in a sparse multilinear polynomial.
public struct SparseEntry {
    public var idx: Int
    public var val: Fr

    @inline(__always)
    public init(idx: Int, val: Fr) {
        self.idx = idx
        self.val = val
    }
}

/// A multilinear polynomial in sparse representation.
/// Stores only non-zero evaluations over the boolean hypercube {0,1}^numVars
/// as a sorted array of (index, value) pairs for cache-friendly access.
public struct SparseMultilinearPoly {
    public let numVars: Int
    /// Non-zero entries sorted by idx (ascending). Invariant: no duplicate indices, no zero values.
    public var entries: [SparseEntry]

    /// Total number of evaluation points (2^numVars)
    public var domainSize: Int { 1 << numVars }

    /// Number of non-zero entries
    public var nnz: Int { entries.count }

    /// Sparsity ratio (fraction of zeros)
    public var sparsity: Double { 1.0 - Double(nnz) / Double(domainSize) }

    /// Create from pre-sorted entries array. Caller must ensure sorted by idx, no duplicates, no zeros.
    public init(numVars: Int, sortedEntries: [SparseEntry]) {
        self.numVars = numVars
        self.entries = sortedEntries
    }

    /// Create from dictionary (convenience, converts to sorted array).
    public init(numVars: Int, entries dict: [Int: Fr]) {
        self.numVars = numVars
        var arr = [SparseEntry]()
        arr.reserveCapacity(dict.count)
        for (idx, val) in dict {
            if !sparseIsZero(val) {
                arr.append(SparseEntry(idx: idx, val: val))
            }
        }
        arr.sort { $0.idx < $1.idx }
        self.entries = arr
    }

    /// Create from dense evaluations (drops zeros)
    public init(dense: [Fr]) {
        let n = dense.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Size must be power of 2")
        self.numVars = Int(log2(Double(n)))
        var arr = [SparseEntry]()
        for i in 0..<n {
            if !sparseIsZero(dense[i]) {
                arr.append(SparseEntry(idx: i, val: dense[i]))
            }
        }
        // Already sorted since we iterate in order
        self.entries = arr
    }

    /// Convert to dense evaluations
    public func toDense() -> [Fr] {
        var result = [Fr](repeating: Fr.zero, count: domainSize)
        for e in entries {
            result[e.idx] = e.val
        }
        return result
    }

    /// Get value at index (returns zero for missing entries).
    /// Uses binary search: O(log nnz).
    public subscript(index: Int) -> Fr {
        let pos = sparseLowerBound(entries, index)
        if pos < entries.count && entries[pos].idx == index {
            return entries[pos].val
        }
        return Fr.zero
    }
}

// MARK: - Sparse Sumcheck Operations

extension SparseMultilinearPoly {

    /// Compute the round polynomial S(X) = (S(0), S(1), S(2)) for the MSB variable.
    /// S(t) = Σ_{x ∈ {0,1}^(k-1)} f(x_1, ..., x_{k-1}, t) extrapolated to t=0,1,2.
    /// Only iterates over non-zero entries: O(nnz) work.
    ///
    /// Uses two-pointer merge over the sorted array: entries in [0, halfN) are the
    /// "low" half (MSB=0), entries in [halfN, 2*halfN) are the "high" half (MSB=1).
    /// Pairs are matched by pairIdx = idx % halfN.
    public func roundPoly() -> (Fr, Fr, Fr) {
        let halfN = domainSize / 2
        var s0 = Fr.zero  // S(0): sum of entries in first half
        var s1 = Fr.zero  // S(1): sum of entries in second half
        var s2 = Fr.zero  // S(2): extrapolation

        // Find the split point: first entry with idx >= halfN
        let splitPos = sparseLowerBound(entries, halfN)

        // Two-pointer merge over low [0..splitPos) and high [splitPos..count)
        var li = 0
        var hi = splitPos

        entries.withUnsafeBufferPointer { buf in
            while li < splitPos || hi < buf.count {
                let lowPairIdx = li < splitPos ? buf[li].idx : Int.max
                let highPairIdx = hi < buf.count ? (buf[hi].idx - halfN) : Int.max

                if lowPairIdx < highPairIdx {
                    // Only low half has this pair index
                    let a = buf[li].val
                    s0 = frAdd(s0, a)
                    // f(2) = 2*b - a = -a when b=0
                    s2 = frSub(s2, a)
                    li += 1
                } else if highPairIdx < lowPairIdx {
                    // Only high half has this pair index
                    let b = buf[hi].val
                    s1 = frAdd(s1, b)
                    // f(2) = 2*b - a = 2*b when a=0
                    let twoB = frAdd(b, b)
                    s2 = frAdd(s2, twoB)
                    hi += 1
                } else {
                    // Both halves have this pair index
                    let a = buf[li].val
                    let b = buf[hi].val
                    s0 = frAdd(s0, a)
                    s1 = frAdd(s1, b)
                    let twoB = frAdd(b, b)
                    s2 = frAdd(s2, frSub(twoB, a))
                    li += 1
                    hi += 1
                }
            }
        }

        return (s0, s1, s2)
    }

    /// Reduce by fixing the MSB variable to `challenge`.
    /// result[i] = (1-r)*self[i] + r*self[i + halfN] for i in 0..<halfN
    /// Returns a new sparse polynomial with numVars-1 variables.
    /// Only iterates over non-zero entries: O(nnz) work.
    ///
    /// Uses two-pointer merge over sorted array, emitting new sorted entries directly.
    public func reduce(challenge: Fr) -> SparseMultilinearPoly {
        precondition(numVars > 0)
        let halfN = domainSize / 2
        let oneMinusR = frSub(Fr.one, challenge)

        let splitPos = sparseLowerBound(entries, halfN)

        var newEntries = [SparseEntry]()
        newEntries.reserveCapacity(entries.count) // upper bound

        var li = 0
        var hi = splitPos

        entries.withUnsafeBufferPointer { buf in
            while li < splitPos || hi < buf.count {
                let lowPairIdx = li < splitPos ? buf[li].idx : Int.max
                let highPairIdx = hi < buf.count ? (buf[hi].idx - halfN) : Int.max

                let pairIdx: Int
                let result: Fr

                if lowPairIdx < highPairIdx {
                    // Only low: result = (1-r)*a
                    pairIdx = lowPairIdx
                    result = frMul(oneMinusR, buf[li].val)
                    li += 1
                } else if highPairIdx < lowPairIdx {
                    // Only high: result = r*b
                    pairIdx = highPairIdx
                    result = frMul(challenge, buf[hi].val)
                    hi += 1
                } else {
                    // Both: result = (1-r)*a + r*b
                    pairIdx = lowPairIdx
                    result = frAdd(frMul(oneMinusR, buf[li].val), frMul(challenge, buf[hi].val))
                    li += 1
                    hi += 1
                }

                if !sparseIsZero(result) {
                    newEntries.append(SparseEntry(idx: pairIdx, val: result))
                }
            }
        }

        return SparseMultilinearPoly(numVars: numVars - 1, sortedEntries: newEntries)
    }

    /// Compute the total sum: Σ f(x) over all x ∈ {0,1}^numVars.
    /// O(nnz) work — sequential scan over contiguous array.
    public func totalSum() -> Fr {
        var sum = Fr.zero
        for e in entries {
            sum = frAdd(sum, e.val)
        }
        return sum
    }
}

// MARK: - Full Sparse Sumcheck Protocol

/// Run a complete sumcheck protocol on a sparse multilinear polynomial.
/// Returns round polynomials, final evaluation, and challenges used.
/// Challenge derivation uses Fiat-Shamir (Blake3).
public func sparseSumcheck(
    poly: SparseMultilinearPoly,
    transcript: inout [UInt8]
) -> (rounds: [(Fr, Fr, Fr)], finalEval: Fr, challenges: [Fr]) {
    var current = poly
    var rounds = [(Fr, Fr, Fr)]()
    var challenges = [Fr]()
    rounds.reserveCapacity(poly.numVars)
    challenges.reserveCapacity(poly.numVars)

    for _ in 0..<poly.numVars {
        let roundPoly = current.roundPoly()
        rounds.append(roundPoly)

        // Fiat-Shamir: derive challenge from transcript + round poly
        appendFrToTranscript(&transcript, roundPoly.0)
        appendFrToTranscript(&transcript, roundPoly.1)
        appendFrToTranscript(&transcript, roundPoly.2)
        let challenge = deriveSumcheckChallenge(transcript)
        challenges.append(challenge)
        appendFrToTranscript(&transcript, challenge)

        current = current.reduce(challenge: challenge)
    }

    precondition(current.domainSize == 1)
    let finalEval = current[0]
    return (rounds, finalEval, challenges)
}

/// Verify a sparse sumcheck proof.
/// Checks round polynomial consistency and final evaluation.
public func verifySumcheckProof(
    rounds: [(Fr, Fr, Fr)],
    claimedSum: Fr,
    finalEval: Fr,
    transcript: inout [UInt8]
) -> (valid: Bool, challenges: [Fr]) {
    let numRounds = rounds.count
    var challenges = [Fr]()
    challenges.reserveCapacity(numRounds)

    // Check round 0: S(0) + S(1) = claimed sum
    let (s0_0, s1_0, _) = rounds[0]
    if !frEqual(frAdd(s0_0, s1_0), claimedSum) {
        return (false, [])
    }

    // Derive first challenge
    appendFrToTranscript(&transcript, s0_0)
    appendFrToTranscript(&transcript, s1_0)
    appendFrToTranscript(&transcript, rounds[0].2)
    let c0 = deriveSumcheckChallenge(transcript)
    challenges.append(c0)
    appendFrToTranscript(&transcript, c0)

    // Check subsequent rounds
    for k in 1..<numRounds {
        let prevEval = evaluateQuadraticPoly(rounds[k-1], at: challenges[k-1])
        let (s0, s1, _) = rounds[k]
        if !frEqual(frAdd(s0, s1), prevEval) {
            return (false, challenges)
        }

        appendFrToTranscript(&transcript, s0)
        appendFrToTranscript(&transcript, s1)
        appendFrToTranscript(&transcript, rounds[k].2)
        let c = deriveSumcheckChallenge(transcript)
        challenges.append(c)
        appendFrToTranscript(&transcript, c)
    }

    // Check final evaluation
    let lastEval = evaluateQuadraticPoly(rounds[numRounds - 1], at: challenges[numRounds - 1])
    if !frEqual(lastEval, finalEval) {
        return (false, challenges)
    }

    return (true, challenges)
}

// MARK: - Sorted array helpers

/// Binary search: find first index in sorted entries where idx >= target.
@inline(__always)
private func sparseLowerBound(_ entries: [SparseEntry], _ target: Int) -> Int {
    var lo = 0, hi = entries.count
    while lo < hi {
        let mid = (lo + hi) >> 1
        if entries[mid].idx < target { lo = mid + 1 } else { hi = mid }
    }
    return lo
}

/// Check if an Fr element is zero (in Montgomery form)
private func sparseIsZero(_ a: Fr) -> Bool {
    let limbs = frToInt(a)
    return limbs[0] == 0 && limbs[1] == 0 && limbs[2] == 0 && limbs[3] == 0
}

// MARK: - Transcript helpers (module-level)

private func appendFrToTranscript(_ transcript: inout [UInt8], _ v: Fr) {
    let vInt = frToInt(v)
    for limb in vInt {
        for byte in 0..<8 {
            transcript.append(UInt8((limb >> (byte * 8)) & 0xFF))
        }
    }
}

private func deriveSumcheckChallenge(_ transcript: [UInt8]) -> Fr {
    let hash = blake3(transcript)
    var limbs = [UInt64](repeating: 0, count: 4)
    for i in 0..<4 {
        for j in 0..<8 {
            limbs[i] |= UInt64(hash[i * 8 + j]) << (j * 8)
        }
    }
    let raw = Fr.from64(limbs)
    return frMul(raw, Fr.from64(Fr.R2_MOD_R))
}

/// Evaluate degree-2 polynomial given by (S(0), S(1), S(2)) at point x.
private func evaluateQuadraticPoly(_ triple: (Fr, Fr, Fr), at x: Fr) -> Fr {
    let (s0, s1, s2) = triple
    let one = Fr.one
    let two = frAdd(one, one)
    let xm1 = frSub(x, one)
    let xm2 = frSub(x, two)
    let inv2 = frInverse(two)
    let negOne = frSub(Fr.zero, one)

    let l0 = frMul(frMul(xm1, xm2), inv2)
    let l1 = frMul(frMul(x, xm2), negOne)
    let l2 = frMul(frMul(x, xm1), inv2)

    return frAdd(frAdd(frMul(s0, l0), frMul(s1, l1)), frMul(s2, l2))
}
