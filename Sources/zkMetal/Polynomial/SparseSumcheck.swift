// Sparse Sumcheck — Exploits sparsity in multilinear polynomials
// When most evaluations over {0,1}^k are zero, operations take O(s) instead of O(2^k)
// where s is the number of non-zero entries.
//
// Use cases: GKR inner products, Lasso subtable lookups, sparse constraint polynomials
//
// Layout convention: same as dense SumcheckEngine (MSB-first variable elimination).
// Index i corresponds to evaluation at the binary point (bit k-1 of i, ..., bit 0 of i).
// First half (indices 0..2^(k-1)-1) has MSB=0, second half has MSB=1.

import Foundation

/// A multilinear polynomial in sparse representation.
/// Stores only non-zero evaluations over the boolean hypercube {0,1}^numVars.
public struct SparseMultilinearPoly {
    public let numVars: Int
    /// Non-zero entries: maps hypercube index → field element value
    public var entries: [Int: Fr]

    /// Total number of evaluation points (2^numVars)
    public var domainSize: Int { 1 << numVars }

    /// Number of non-zero entries
    public var nnz: Int { entries.count }

    /// Sparsity ratio (fraction of zeros)
    public var sparsity: Double { 1.0 - Double(nnz) / Double(domainSize) }

    public init(numVars: Int, entries: [Int: Fr]) {
        self.numVars = numVars
        self.entries = entries
    }

    /// Create from dense evaluations (drops zeros)
    public init(dense: [Fr]) {
        let n = dense.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Size must be power of 2")
        self.numVars = Int(log2(Double(n)))
        var e = [Int: Fr]()
        for i in 0..<n {
            if !isZero(dense[i]) {
                e[i] = dense[i]
            }
        }
        self.entries = e
    }

    /// Convert to dense evaluations
    public func toDense() -> [Fr] {
        var result = [Fr](repeating: Fr.zero, count: domainSize)
        for (idx, val) in entries {
            result[idx] = val
        }
        return result
    }

    /// Get value at index (returns zero for missing entries)
    public subscript(index: Int) -> Fr {
        entries[index] ?? Fr.zero
    }
}

// MARK: - Sparse Sumcheck Operations

extension SparseMultilinearPoly {

    /// Compute the round polynomial S(X) = (S(0), S(1), S(2)) for the MSB variable.
    /// S(t) = Σ_{x ∈ {0,1}^(k-1)} f(x_1, ..., x_{k-1}, t) extrapolated to t=0,1,2.
    /// Only iterates over non-zero entries: O(nnz) work.
    public func roundPoly() -> (Fr, Fr, Fr) {
        let halfN = domainSize / 2
        var s0 = Fr.zero  // S(0): sum of entries in first half
        var s1 = Fr.zero  // S(1): sum of entries in second half
        var s2 = Fr.zero  // S(2): extrapolation

        // Track which pair indices have been visited from either side
        var pairContributions = [Int: (Fr, Fr)]()  // pairIdx → (a, b)

        for (idx, val) in entries {
            let pairIdx = idx % halfN
            var (a, b) = pairContributions[pairIdx] ?? (Fr.zero, Fr.zero)
            if idx < halfN {
                a = val
            } else {
                b = val
            }
            pairContributions[pairIdx] = (a, b)
        }

        for (_, (a, b)) in pairContributions {
            s0 = frAdd(s0, a)
            s1 = frAdd(s1, b)
            // f(2) = 2*b - a (linear extrapolation from f(0)=a, f(1)=b)
            let twoB = frAdd(b, b)
            s2 = frAdd(s2, frSub(twoB, a))
        }

        return (s0, s1, s2)
    }

    /// Reduce by fixing the MSB variable to `challenge`.
    /// result[i] = (1-r)*self[i] + r*self[i + halfN] for i in 0..<halfN
    /// Returns a new sparse polynomial with numVars-1 variables.
    /// Only iterates over non-zero entries: O(nnz) work.
    public func reduce(challenge: Fr) -> SparseMultilinearPoly {
        precondition(numVars > 0)
        let halfN = domainSize / 2
        let oneMinusR = frSub(Fr.one, challenge)

        // Collect pair indices that have at least one non-zero entry
        var pairEntries = [Int: (Fr, Fr)]()
        for (idx, val) in entries {
            let pairIdx = idx % halfN
            var (a, b) = pairEntries[pairIdx] ?? (Fr.zero, Fr.zero)
            if idx < halfN {
                a = val
            } else {
                b = val
            }
            pairEntries[pairIdx] = (a, b)
        }

        var newEntries = [Int: Fr]()
        for (pairIdx, (a, b)) in pairEntries {
            // result = (1-r)*a + r*b
            let result = frAdd(frMul(oneMinusR, a), frMul(challenge, b))
            if !isZero(result) {
                newEntries[pairIdx] = result
            }
        }

        return SparseMultilinearPoly(numVars: numVars - 1, entries: newEntries)
    }

    /// Compute the total sum: Σ f(x) over all x ∈ {0,1}^numVars.
    /// O(nnz) work.
    public func totalSum() -> Fr {
        var sum = Fr.zero
        for (_, val) in entries {
            sum = frAdd(sum, val)
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

/// Check if an Fr element is zero (in Montgomery form)
private func isZero(_ a: Fr) -> Bool {
    let limbs = frToInt(a)
    return limbs[0] == 0 && limbs[1] == 0 && limbs[2] == 0 && limbs[3] == 0
}
