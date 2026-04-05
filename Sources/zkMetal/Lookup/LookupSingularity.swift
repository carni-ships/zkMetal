// Lookup Singularity Engine
// Proves that every witness value appears in a lookup table using the
// Lookup Singularity protocol -- O(n) prover time, no sorting required.
//
// Protocol (Lookup Singularity):
//   Given table T[0..N-1] and witness values w[0..m-1] where each w[i] in T:
//   1. Prover computes multiplicities: mult[j] = #{i : w[i] = T[j]}
//   2. Verifier sends random challenge gamma
//   3. Prove the rational identity via sumcheck over the boolean hypercube:
//        sum_{x in {0,1}^logM} 1/(gamma - w(x)) = sum_{x in {0,1}^logN} mult(x)/(gamma - T(x))
//   4. The sumcheck reduces to multilinear polynomial evaluations at a random point
//
// Key advantage over LogUp: no sorting step, O(n) prover with small constants.
// The sumcheck is over multilinear extensions on {0,1}^n, using existing GPU engine.
//
// References: Lookup Singularity (Gabizon 2023), LogUp-GKR (Haboeck 2022)

import Foundation
import Metal

// MARK: - Proof

/// Proof produced by the Lookup Singularity protocol.
public struct LookupSingularityProof {
    /// Multiplicities: mult[j] = number of times T[j] appears in w
    public let multiplicities: [Fr]
    /// Random challenge gamma used in the protocol
    public let gamma: Fr
    /// Claimed sum: S = sum_i 1/(gamma - w_i) = sum_j mult_j/(gamma - T_j)
    public let claimedSum: Fr
    /// Sumcheck round polynomials for the witness side
    public let witnessSumcheckRounds: [(Fr, Fr, Fr)]
    /// Sumcheck round polynomials for the table side
    public let tableSumcheckRounds: [(Fr, Fr, Fr)]
    /// Final evaluation of the witness inverse polynomial at the random point
    public let witnessFinalEval: Fr
    /// Final evaluation of the table weighted-inverse polynomial at the random point
    public let tableFinalEval: Fr

    public init(multiplicities: [Fr], gamma: Fr, claimedSum: Fr,
                witnessSumcheckRounds: [(Fr, Fr, Fr)],
                tableSumcheckRounds: [(Fr, Fr, Fr)],
                witnessFinalEval: Fr, tableFinalEval: Fr) {
        self.multiplicities = multiplicities
        self.gamma = gamma
        self.claimedSum = claimedSum
        self.witnessSumcheckRounds = witnessSumcheckRounds
        self.tableSumcheckRounds = tableSumcheckRounds
        self.witnessFinalEval = witnessFinalEval
        self.tableFinalEval = tableFinalEval
    }
}

// MARK: - Prover

/// Proves that all witness values exist in a lookup table using the
/// Lookup Singularity approach: sumcheck over multilinear inverse polynomials.
public class LookupSingularityProver {
    public let polyEngine: PolyEngine
    public let sumcheckEngine: SumcheckEngine

    public init() throws {
        self.polyEngine = try PolyEngine()
        self.sumcheckEngine = try SumcheckEngine()
    }

    /// Create a Lookup Singularity proof.
    ///
    /// - Parameters:
    ///   - table: The lookup table values (will be padded to power of 2)
    ///   - witnesses: The witness values to prove membership for (padded to power of 2)
    ///   - gamma: Random challenge (in a real system, derived via Fiat-Shamir)
    /// - Returns: A LookupSingularityProof
    public func prove(table: [Fr], witnesses: [Fr], gamma: Fr? = nil) throws -> LookupSingularityProof {
        // Pad to power of 2
        let paddedTable = padToPow2(table, padWith: table[0])
        let paddedWitnesses = padToPow2(witnesses, padWith: witnesses[0])
        let m = paddedWitnesses.count
        let N = paddedTable.count

        precondition(m > 0 && (m & (m - 1)) == 0, "Witness count must be power of 2")
        precondition(N > 0 && (N & (N - 1)) == 0, "Table size must be power of 2")

        // Step 1: Derive gamma via Fiat-Shamir if not provided
        let gammaVal = gamma ?? deriveGamma(witnesses: paddedWitnesses, table: paddedTable)

        // Step 2: Compute multiplicities
        let mult = LookupEngine.computeMultiplicities(table: paddedTable, lookups: paddedWitnesses)

        // Step 3: Compute h_w[i] = 1/(gamma - w[i]) -- witness inverse polynomial
        var gammaMinusW = [Fr](repeating: Fr.zero, count: m)
        for i in 0..<m {
            gammaMinusW[i] = frSub(gammaVal, paddedWitnesses[i])
        }
        let hw = try polyEngine.batchInverse(gammaMinusW)

        // Step 4: Compute h_t[j] = mult[j]/(gamma - T[j]) -- table weighted-inverse polynomial
        var gammaMinusT = [Fr](repeating: Fr.zero, count: N)
        for j in 0..<N {
            gammaMinusT[j] = frSub(gammaVal, paddedTable[j])
        }
        let invT = try polyEngine.batchInverse(gammaMinusT)
        let ht = try polyEngine.hadamard(mult, invT)

        // Step 5: Compute claimed sum S = sum_i h_w[i]
        var witnessSum = Fr.zero
        for i in 0..<m {
            witnessSum = frAdd(witnessSum, hw[i])
        }

        // Verify table side matches (soundness check)
        var tableSum = Fr.zero
        for j in 0..<N {
            tableSum = frAdd(tableSum, ht[j])
        }
        precondition(frEqual(witnessSum, tableSum),
                     "Lookup Singularity sum mismatch -- witness values not all in table")

        // Step 6: Run sumcheck on witness side
        var transcript = [UInt8]()
        appendFrToTranscript(&transcript, gammaVal)
        appendFrToTranscript(&transcript, witnessSum)

        let logM = intLog2(m)
        let (witnessRounds, witnessFinalEval, _) = try runSumcheck(
            evals: hw, numVars: logM, transcript: &transcript)

        // Step 7: Run sumcheck on table side
        let logN = intLog2(N)
        let (tableRounds, tableFinalEval, _) = try runSumcheck(
            evals: ht, numVars: logN, transcript: &transcript)

        return LookupSingularityProof(
            multiplicities: mult,
            gamma: gammaVal,
            claimedSum: witnessSum,
            witnessSumcheckRounds: witnessRounds,
            tableSumcheckRounds: tableRounds,
            witnessFinalEval: witnessFinalEval,
            tableFinalEval: tableFinalEval
        )
    }

    // MARK: - Sumcheck

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
            for _ in 0..<numVars {
                let c = challengeFromTranscript(transcript)
                challenges.append(c)
                appendFrToTranscript(&transcript, c)
            }
            let (gpuRounds, finalEval) = try sumcheckEngine.fullSumcheck(
                evals: evals, challenges: challenges)
            return (gpuRounds, finalEval, challenges)
        } else {
            var current = evals
            for _ in 0..<numVars {
                let roundPoly = SumcheckEngine.cpuRoundPoly(evals: current)
                rounds.append(roundPoly)

                appendFrToTranscript(&transcript, roundPoly.0)
                appendFrToTranscript(&transcript, roundPoly.1)
                appendFrToTranscript(&transcript, roundPoly.2)
                let challenge = challengeFromTranscript(transcript)
                challenges.append(challenge)
                appendFrToTranscript(&transcript, challenge)

                current = SumcheckEngine.cpuReduce(evals: current, challenge: challenge)
            }
            precondition(current.count == 1)
            return (rounds, current[0], challenges)
        }
    }

    // MARK: - Helpers

    private func padToPow2(_ arr: [Fr], padWith: Fr) -> [Fr] {
        let n = arr.count
        var p = 1
        while p < n { p <<= 1 }
        if p == n { return arr }
        var result = arr
        while result.count < p { result.append(padWith) }
        return result
    }

    private func intLog2(_ n: Int) -> Int {
        var v = n, k = 0
        while v > 1 { v >>= 1; k += 1 }
        return k
    }

    private func deriveGamma(witnesses: [Fr], table: [Fr]) -> Fr {
        var transcript = [UInt8]()
        transcript.reserveCapacity(128)
        // Domain separator
        let tag: [UInt8] = Array("LookupSingularity".utf8)
        transcript.append(contentsOf: tag)
        // Hash table size + witness size + first few values
        var tSize = UInt64(table.count)
        for _ in 0..<8 { transcript.append(UInt8(tSize & 0xFF)); tSize >>= 8 }
        var wSize = UInt64(witnesses.count)
        for _ in 0..<8 { transcript.append(UInt8(wSize & 0xFF)); wSize >>= 8 }
        let sampleCount = min(16, witnesses.count)
        for i in 0..<sampleCount {
            let limbs = frToInt(witnesses[i])
            for limb in limbs {
                var v = limb
                for _ in 0..<8 { transcript.append(UInt8(v & 0xFF)); v >>= 8 }
            }
        }
        return challengeFromTranscript(transcript)
    }

    private func appendFrToTranscript(_ transcript: inout [UInt8], _ v: Fr) {
        let limbs = frToInt(v)
        for limb in limbs {
            for byte in 0..<8 {
                transcript.append(UInt8((limb >> (byte * 8)) & 0xFF))
            }
        }
    }

    private func challengeFromTranscript(_ transcript: [UInt8]) -> Fr {
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

// MARK: - Verifier

/// Verifies a Lookup Singularity proof.
public class LookupSingularityVerifier {

    public init() {}

    /// Verify a LookupSingularityProof against the original table and witnesses.
    ///
    /// In a real system the final evaluations would be checked via polynomial commitment
    /// openings. Here we recompute the inverse polynomials and evaluate their MLEs.
    public func verify(proof: LookupSingularityProof,
                       table: [Fr], witnesses: [Fr]) throws -> Bool {
        let paddedTable = padToPow2(table, padWith: table[0])
        let paddedWitnesses = padToPow2(witnesses, padWith: witnesses[0])
        let m = paddedWitnesses.count
        let N = paddedTable.count
        let logM = intLog2(m)
        let logN = intLog2(N)

        guard proof.witnessSumcheckRounds.count == logM else { return false }
        guard proof.tableSumcheckRounds.count == logN else { return false }
        guard proof.multiplicities.count == N else { return false }

        // Reconstruct transcript and challenges
        var transcript = [UInt8]()
        appendFrToTranscript(&transcript, proof.gamma)
        appendFrToTranscript(&transcript, proof.claimedSum)

        // Verify witness-side sumcheck
        let witnessChallenges: [Fr]
        let useGPUWitness = m >= 256
        if useGPUWitness {
            var wc = [Fr]()
            for _ in 0..<logM {
                let c = challengeFromTranscript(transcript)
                wc.append(c)
                appendFrToTranscript(&transcript, c)
            }
            witnessChallenges = wc
        } else {
            var wc = [Fr]()
            for k in 0..<logM {
                let (s0, s1, s2) = proof.witnessSumcheckRounds[k]
                appendFrToTranscript(&transcript, s0)
                appendFrToTranscript(&transcript, s1)
                appendFrToTranscript(&transcript, s2)
                let c = challengeFromTranscript(transcript)
                wc.append(c)
                appendFrToTranscript(&transcript, c)
            }
            witnessChallenges = wc
        }

        // Check round 0: S(0) + S(1) = claimed sum
        let (w0_0, w1_0, _) = proof.witnessSumcheckRounds[0]
        if !frEqual(frAdd(w0_0, w1_0), proof.claimedSum) { return false }

        // Check subsequent rounds
        for k in 1..<logM {
            let (s0, s1, _) = proof.witnessSumcheckRounds[k]
            let prevEval = evaluateQuadratic(proof.witnessSumcheckRounds[k-1], at: witnessChallenges[k-1])
            if !frEqual(frAdd(s0, s1), prevEval) { return false }
        }

        // Check final evaluation
        let lastWitnessEval = evaluateQuadratic(
            proof.witnessSumcheckRounds[logM - 1], at: witnessChallenges[logM - 1])
        if !frEqual(lastWitnessEval, proof.witnessFinalEval) { return false }

        // Verify table-side sumcheck
        let tableChallenges: [Fr]
        let useGPUTable = N >= 256
        if useGPUTable {
            var tc = [Fr]()
            for _ in 0..<logN {
                let c = challengeFromTranscript(transcript)
                tc.append(c)
                appendFrToTranscript(&transcript, c)
            }
            tableChallenges = tc
        } else {
            var tc = [Fr]()
            for k in 0..<logN {
                let (s0, s1, s2) = proof.tableSumcheckRounds[k]
                appendFrToTranscript(&transcript, s0)
                appendFrToTranscript(&transcript, s1)
                appendFrToTranscript(&transcript, s2)
                let c = challengeFromTranscript(transcript)
                tc.append(c)
                appendFrToTranscript(&transcript, c)
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

        // Verify final evaluations by recomputing the inverse polynomials
        // and evaluating their MLEs at the sumcheck random points.
        // h_w[i] = 1/(gamma - w[i]) for i in {0,1}^logM
        var hwEvals = [Fr](repeating: Fr.zero, count: m)
        for i in 0..<m {
            hwEvals[i] = frInverse(frSub(proof.gamma, paddedWitnesses[i]))
        }
        let hwAtR = evaluateMLE(hwEvals, at: witnessChallenges)
        if !frEqual(hwAtR, proof.witnessFinalEval) { return false }

        // h_t[j] = mult[j]/(gamma - T[j]) for j in {0,1}^logN
        var htEvals = [Fr](repeating: Fr.zero, count: N)
        for j in 0..<N {
            let inv = frInverse(frSub(proof.gamma, paddedTable[j]))
            htEvals[j] = frMul(proof.multiplicities[j], inv)
        }
        let htAtR = evaluateMLE(htEvals, at: tableChallenges)
        if !frEqual(htAtR, proof.tableFinalEval) { return false }

        return true
    }

    // MARK: - Helpers

    private func padToPow2(_ arr: [Fr], padWith: Fr) -> [Fr] {
        let n = arr.count
        var p = 1
        while p < n { p <<= 1 }
        if p == n { return arr }
        var result = arr
        while result.count < p { result.append(padWith) }
        return result
    }

    private func intLog2(_ n: Int) -> Int {
        var v = n, k = 0
        while v > 1 { v >>= 1; k += 1 }
        return k
    }

    private func evaluateQuadratic(_ triple: (Fr, Fr, Fr), at x: Fr) -> Fr {
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

    private func evaluateMLE(_ evals: [Fr], at point: [Fr]) -> Fr {
        var current = evals
        for r in point {
            let half = current.count / 2
            var next = [Fr](repeating: Fr.zero, count: half)
            let oneMinusR = frSub(Fr.one, r)
            for i in 0..<half {
                next[i] = frAdd(frMul(oneMinusR, current[i]), frMul(r, current[half + i]))
            }
            current = next
        }
        precondition(current.count == 1)
        return current[0]
    }

    private func appendFrToTranscript(_ transcript: inout [UInt8], _ v: Fr) {
        let limbs = frToInt(v)
        for limb in limbs {
            for byte in 0..<8 {
                transcript.append(UInt8((limb >> (byte * 8)) & 0xFF))
            }
        }
    }

    private func challengeFromTranscript(_ transcript: [UInt8]) -> Fr {
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
