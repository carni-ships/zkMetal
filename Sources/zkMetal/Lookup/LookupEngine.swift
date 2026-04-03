// LogUp Lookup Argument Engine
// Proves that every element in a lookup vector exists in a table using the LogUp protocol.
//
// Protocol (LogUp / logarithmic derivative):
//   Given table T[0..N-1] and lookups f[0..m-1] where each f[i] ∈ T:
//   1. Prover computes multiplicities mult[j] = #{i : f[i] = T[j]}
//   2. Verifier sends random challenge β
//   3. Prove: Σᵢ 1/(β + f[i]) = Σⱼ mult[j]/(β + T[j])
//   4. Both sides proven via sumcheck on multilinear extensions
//
// References: LogUp (Haböck 2022), Lasso (Setty et al. 2023)

import Foundation
import Metal

/// Proof that all lookups exist in the table
public struct LookupProof {
    /// Multiplicities: mult[j] = number of times T[j] appears in f
    public let multiplicities: [Fr]
    /// The random challenge β used in the protocol
    public let beta: Fr
    /// Sumcheck rounds for the lookup side: Σ 1/(β + f[i])
    public let lookupSumcheckRounds: [(Fr, Fr, Fr)]
    /// Sumcheck rounds for the table side: Σ mult[j]/(β + T[j])
    public let tableSumcheckRounds: [(Fr, Fr, Fr)]
    /// Claimed sum value S = Σ 1/(β + f[i])
    public let claimedSum: Fr
    /// Final evaluation of lookup inverse polynomial at random point
    public let lookupFinalEval: Fr
    /// Final evaluation of table inverse polynomial at random point
    public let tableFinalEval: Fr

    public init(multiplicities: [Fr], beta: Fr,
                lookupSumcheckRounds: [(Fr, Fr, Fr)],
                tableSumcheckRounds: [(Fr, Fr, Fr)],
                claimedSum: Fr, lookupFinalEval: Fr, tableFinalEval: Fr) {
        self.multiplicities = multiplicities
        self.beta = beta
        self.lookupSumcheckRounds = lookupSumcheckRounds
        self.tableSumcheckRounds = tableSumcheckRounds
        self.claimedSum = claimedSum
        self.lookupFinalEval = lookupFinalEval
        self.tableFinalEval = tableFinalEval
    }
}

public class LookupEngine {
    public let polyEngine: PolyEngine
    public let sumcheckEngine: SumcheckEngine

    public init() throws {
        self.polyEngine = try PolyEngine()
        self.sumcheckEngine = try SumcheckEngine()
    }

    /// Compute multiplicities: for each table entry T[j], count how many times it appears in f.
    /// Returns array of length table.count with multiplicity values as Fr elements.
    public static func computeMultiplicities(table: [Fr], lookups: [Fr]) -> [Fr] {
        // Build map from table value → index
        var tableIndex = [FrKey: Int]()
        for j in 0..<table.count {
            tableIndex[FrKey(table[j])] = j
        }

        var mult = [UInt64](repeating: 0, count: table.count)
        for i in 0..<lookups.count {
            guard let idx = tableIndex[FrKey(lookups[i])] else {
                preconditionFailure("Lookup value not in table at index \(i)")
            }
            mult[idx] += 1
        }

        return mult.map { frFromInt($0) }
    }

    /// Create a LogUp lookup proof.
    /// Proves that every element in `lookups` exists in `table`.
    /// The `beta` challenge would normally come from Fiat-Shamir; here it's passed explicitly.
    public func prove(table: [Fr], lookups: [Fr], beta: Fr) throws -> LookupProof {
        let m = lookups.count
        let N = table.count
        precondition(m > 0 && (m & (m - 1)) == 0, "Lookup count must be power of 2")
        precondition(N > 0 && (N & (N - 1)) == 0, "Table size must be power of 2")

        // Step 1: Compute multiplicities
        let mult = LookupEngine.computeMultiplicities(table: table, lookups: lookups)

        // Step 2: Compute h_f[i] = 1/(β + f[i]) for the lookup side
        var betaPlusF = [Fr](repeating: Fr.zero, count: m)
        for i in 0..<m {
            betaPlusF[i] = frAdd(beta, lookups[i])
        }
        let hf = try polyEngine.batchInverse(betaPlusF)

        // Step 3: Compute h_t[j] = mult[j]/(β + T[j]) for the table side
        var betaPlusT = [Fr](repeating: Fr.zero, count: N)
        for i in 0..<N {
            betaPlusT[i] = frAdd(beta, table[i])
        }
        let invBetaPlusT = try polyEngine.batchInverse(betaPlusT)
        let ht = try polyEngine.hadamard(mult, invBetaPlusT)

        // Step 4: Compute the claimed sum S = Σ h_f[i]
        var sum = Fr.zero
        for i in 0..<m {
            sum = frAdd(sum, hf[i])
        }

        // Verify table side matches (sanity check)
        var tableSum = Fr.zero
        for j in 0..<N {
            tableSum = frAdd(tableSum, ht[j])
        }
        precondition(frEqual(sum, tableSum), "LogUp sum mismatch — lookup values not all in table")

        // Step 5: Run sumcheck on h_f (lookup side)
        // Use GPU for large inputs, CPU for small
        var transcript = [UInt8]()
        appendFr(&transcript, beta)
        appendFr(&transcript, sum)

        let logM = Int(log2(Double(m)))
        let (lookupRounds, lookupFinalEval, lookupChallenges) = try runSumcheck(
            evals: hf, numVars: logM, transcript: &transcript)

        // Step 6: Run sumcheck on h_t (table side)
        let logN = Int(log2(Double(N)))
        let (tableRounds, tableFinalEval, tableChallenges) = try runSumcheck(
            evals: ht, numVars: logN, transcript: &transcript)
        _ = tableChallenges  // used implicitly via transcript

        return LookupProof(
            multiplicities: mult,
            beta: beta,
            lookupSumcheckRounds: lookupRounds,
            tableSumcheckRounds: tableRounds,
            claimedSum: sum,
            lookupFinalEval: lookupFinalEval,
            tableFinalEval: tableFinalEval
        )
    }

    /// Verify a LogUp lookup proof.
    /// Requires the original table and lookups (or their commitments in a real system).
    public func verify(proof: LookupProof, table: [Fr], lookups: [Fr]) throws -> Bool {
        let m = lookups.count
        let N = table.count
        let logM = Int(log2(Double(m)))
        let logN = Int(log2(Double(N)))

        guard proof.lookupSumcheckRounds.count == logM else { return false }
        guard proof.tableSumcheckRounds.count == logN else { return false }
        guard proof.multiplicities.count == N else { return false }

        // Reconstruct transcript and challenges (must match prover's derivation)
        var transcript = [UInt8]()
        appendFr(&transcript, proof.beta)
        appendFr(&transcript, proof.claimedSum)

        // Verify lookup-side sumcheck and reconstruct challenges
        let lookupChallenges: [Fr]
        let useGPULookup = m >= 256
        if useGPULookup {
            // GPU path: challenges derived before seeing round polys
            var lc = [Fr]()
            for _ in 0..<logM {
                let c = deriveChallenge(transcript)
                lc.append(c)
                appendFr(&transcript, c)
            }
            lookupChallenges = lc
        } else {
            // CPU path: challenges derived after each round poly
            var lc = [Fr]()
            for k in 0..<logM {
                let (s0, s1, s2) = proof.lookupSumcheckRounds[k]
                appendFr(&transcript, s0)
                appendFr(&transcript, s1)
                appendFr(&transcript, s2)
                let c = deriveChallenge(transcript)
                lc.append(c)
                appendFr(&transcript, c)
            }
            lookupChallenges = lc
        }

        // Check round 0: S(0) + S(1) = claimed sum
        let (s0_0, s1_0, _) = proof.lookupSumcheckRounds[0]
        if !frEqual(frAdd(s0_0, s1_0), proof.claimedSum) { return false }

        // Check subsequent rounds
        for k in 1..<logM {
            let (s0, s1, _) = proof.lookupSumcheckRounds[k]
            let prevEval = evaluateQuadratic(proof.lookupSumcheckRounds[k-1], at: lookupChallenges[k-1])
            if !frEqual(frAdd(s0, s1), prevEval) { return false }
        }

        // Check final eval
        let lastLookupEval = evaluateQuadratic(
            proof.lookupSumcheckRounds[logM - 1], at: lookupChallenges[logM - 1])
        if !frEqual(lastLookupEval, proof.lookupFinalEval) { return false }

        // Verify table-side sumcheck
        let tableChallenges: [Fr]
        let useGPUTable = N >= 256
        if useGPUTable {
            var tc = [Fr]()
            for _ in 0..<logN {
                let c = deriveChallenge(transcript)
                tc.append(c)
                appendFr(&transcript, c)
            }
            tableChallenges = tc
        } else {
            var tc = [Fr]()
            for k in 0..<logN {
                let (s0, s1, s2) = proof.tableSumcheckRounds[k]
                appendFr(&transcript, s0)
                appendFr(&transcript, s1)
                appendFr(&transcript, s2)
                let c = deriveChallenge(transcript)
                tc.append(c)
                appendFr(&transcript, c)
            }
            tableChallenges = tc
        }

        let (t0_0, t1_0, _) = proof.tableSumcheckRounds[0]
        if !frEqual(frAdd(t0_0, t1_0), proof.claimedSum) { return false }

        for k in 1..<logN {
            let (s0, s1, _) = proof.tableSumcheckRounds[k]
            let prevEval = evaluateQuadratic(proof.tableSumcheckRounds[k-1], at: tableChallenges[k-1])
            if !frEqual(frAdd(s0, s1), prevEval) { return false }
        }

        let lastTableEval = evaluateQuadratic(
            proof.tableSumcheckRounds[logN - 1], at: tableChallenges[logN - 1])
        if !frEqual(lastTableEval, proof.tableFinalEval) { return false }

        // Verify final evaluations against the actual polynomials
        // (In a real system, these would be polynomial commitment openings)
        let hfEvals = try computeHfEvals(lookups: lookups, beta: proof.beta)
        let hfAtR = evaluateMLE(hfEvals, at: lookupChallenges)
        if !frEqual(hfAtR, proof.lookupFinalEval) { return false }

        let htEvals = try computeHtEvals(table: table, multiplicities: proof.multiplicities, beta: proof.beta)
        let htAtR = evaluateMLE(htEvals, at: tableChallenges)
        if !frEqual(htAtR, proof.tableFinalEval) { return false }

        return true
    }

    // MARK: - Sumcheck execution

    /// Run a sumcheck protocol with round-by-round Fiat-Shamir challenge derivation.
    /// Uses GPU for sizes ≥ 256, CPU for smaller.
    /// Returns: (round polynomials, final evaluation, challenges used)
    private func runSumcheck(evals: [Fr], numVars: Int,
                             transcript: inout [UInt8]) throws
        -> (rounds: [(Fr, Fr, Fr)], finalEval: Fr, challenges: [Fr])
    {
        let useGPU = evals.count >= 256
        var rounds = [(Fr, Fr, Fr)]()
        var challenges = [Fr]()
        rounds.reserveCapacity(numVars)
        challenges.reserveCapacity(numVars)

        if useGPU {
            // Generate all challenges first (Fiat-Shamir from transcript seed)
            for _ in 0..<numVars {
                let c = deriveChallenge(transcript)
                challenges.append(c)
                appendFr(&transcript, c)
            }
            let (gpuRounds, finalEval) = try sumcheckEngine.fullSumcheck(
                evals: evals, challenges: challenges)
            return (gpuRounds, finalEval, challenges)
        } else {
            // CPU round-by-round with proper Fiat-Shamir
            var current = evals
            for _ in 0..<numVars {
                let roundPoly = SumcheckEngine.cpuRoundPoly(evals: current)
                rounds.append(roundPoly)

                // Derive challenge from transcript including this round's polynomial
                appendFr(&transcript, roundPoly.0)
                appendFr(&transcript, roundPoly.1)
                appendFr(&transcript, roundPoly.2)
                let challenge = deriveChallenge(transcript)
                challenges.append(challenge)
                appendFr(&transcript, challenge)

                current = SumcheckEngine.cpuReduce(evals: current, challenge: challenge)
            }
            precondition(current.count == 1)
            return (rounds, current[0], challenges)
        }
    }

    // MARK: - Helpers

    /// Compute h_f[i] = 1/(β + f[i])
    private func computeHfEvals(lookups: [Fr], beta: Fr) throws -> [Fr] {
        var betaPlusF = [Fr](repeating: Fr.zero, count: lookups.count)
        for i in 0..<lookups.count {
            betaPlusF[i] = frAdd(beta, lookups[i])
        }
        return try polyEngine.batchInverse(betaPlusF)
    }

    /// Compute h_t[j] = mult[j]/(β + T[j])
    private func computeHtEvals(table: [Fr], multiplicities: [Fr], beta: Fr) throws -> [Fr] {
        var betaPlusT = [Fr](repeating: Fr.zero, count: table.count)
        for i in 0..<table.count {
            betaPlusT[i] = frAdd(beta, table[i])
        }
        let inv = try polyEngine.batchInverse(betaPlusT)
        return try polyEngine.hadamard(multiplicities, inv)
    }

    /// Evaluate a degree-2 polynomial given by (S(0), S(1), S(2)) at point x.
    /// Uses Lagrange interpolation over {0, 1, 2}.
    private func evaluateQuadratic(_ triple: (Fr, Fr, Fr), at x: Fr) -> Fr {
        let (s0, s1, s2) = triple
        // L_0(x) = (x-1)(x-2)/2, L_1(x) = x(x-2)/(-1), L_2(x) = x(x-1)/2
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

    /// Evaluate multilinear extension at a point using repeated folding.
    /// evals: evaluations over {0,1}^k, point: [r_0, ..., r_{k-1}]
    private func evaluateMLE(_ evals: [Fr], at point: [Fr]) -> Fr {
        var current = evals
        for r in point {
            let half = current.count / 2
            var next = [Fr](repeating: Fr.zero, count: half)
            for i in 0..<half {
                // f(r) = (1-r)*f(0) + r*f(1)
                let oneMinusR = frSub(Fr.one, r)
                next[i] = frAdd(frMul(oneMinusR, current[i]), frMul(r, current[half + i]))
            }
            current = next
        }
        precondition(current.count == 1)
        return current[0]
    }

    // MARK: - Fiat-Shamir

    private func appendFr(_ transcript: inout [UInt8], _ v: Fr) {
        let vInt = frToInt(v)
        for limb in vInt {
            for byte in 0..<8 {
                transcript.append(UInt8((limb >> (byte * 8)) & 0xFF))
            }
        }
    }

    private func deriveChallenge(_ transcript: [UInt8]) -> Fr {
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
}

/// Hashable wrapper for Fr elements (used in multiplicity computation)
struct FrKey: Hashable {
    let limbs: [UInt64]
    init(_ fr: Fr) {
        self.limbs = frToInt(fr)
    }
    static func == (lhs: FrKey, rhs: FrKey) -> Bool {
        lhs.limbs == rhs.limbs
    }
    func hash(into hasher: inout Hasher) {
        for l in limbs { hasher.combine(l) }
    }
}

/// Check if two Fr elements are equal
public func frEqual(_ a: Fr, _ b: Fr) -> Bool {
    frToInt(a) == frToInt(b)
}
