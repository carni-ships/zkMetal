// Batched LogUp-GKR Lookup Argument
//
// Proves multiple independent lookup tables in a single proof using random
// linear combination. This is the batched variant used when a circuit has
// multiple lookup tables (e.g., range check + instruction + memory).
//
// Protocol:
//   1. Prover commits to all individual LogUp-GKR instances.
//   2. Verifier sends a shared challenge gamma (and beta for multi-column).
//   3. For each table k, compute the log-derivative relation independently.
//   4. Verifier sends a batching challenge alpha.
//   5. Combine all individual sums via random linear combination:
//      sum_k alpha^k * (witnessSum_k - tableSum_k) = 0
//   6. Run a single batched sumcheck to prove the combined relation.
//
// Sharing gamma across all lookups is safe because the challenge is
// independent of the table contents (derived from the transcript).
//
// References:
//   - Binius (2024): batched LogUp-GKR
//   - Stwo (2024): multi-table lookup batching

import Foundation
import NeonFieldOps

// MARK: - Batch LogUp-GKR Proof Types

/// A single lookup instance within a batch.
public struct LogUpBatchInstance {
    /// Witness columns (single-column: [[Fr]], multi-column: [[Fr], [Fr], ...])
    public let witnessColumns: [[Fr]]
    /// Table columns (same column count as witness)
    public let tableColumns: [[Fr]]

    /// Convenience for single-column lookups.
    public init(witness: [Fr], table: [Fr]) {
        self.witnessColumns = [witness]
        self.tableColumns = [table]
    }

    /// Multi-column lookup instance.
    public init(witnessColumns: [[Fr]], tableColumns: [[Fr]]) {
        precondition(witnessColumns.count == tableColumns.count,
                     "Witness and table must have same number of columns")
        self.witnessColumns = witnessColumns
        self.tableColumns = tableColumns
    }
}

/// Proof for a batched LogUp-GKR argument over multiple tables.
public struct LogUpBatchGKRProof {
    /// Number of lookup instances in the batch.
    public let batchSize: Int
    /// Shared challenge gamma.
    public let gamma: Fr
    /// Shared challenge beta (for multi-column; Fr.zero if all single-column).
    public let beta: Fr
    /// Batching challenge alpha for random linear combination.
    public let alpha: Fr
    /// Per-instance data: multiplicities.
    public let instanceMultiplicities: [[Fr]]
    /// Per-instance witness sums.
    public let instanceWitnessSums: [Fr]
    /// Per-instance table sums.
    public let instanceTableSums: [Fr]
    /// Individual LogUp-GKR proofs (one per instance).
    public let instanceProofs: [LogUpGKRProof]
}

// MARK: - Batch LogUp-GKR Prover

/// Proves multiple lookup arguments in a single batched proof.
public class LogUpBatchGKRProver {

    private let prover = LogUpGKRProver()

    public init() {}

    /// Prove a batch of lookup instances.
    ///
    /// All instances share the same gamma challenge for efficiency.
    /// Each instance is proved independently, then combined via random
    /// linear combination with batching challenge alpha.
    ///
    /// - Parameters:
    ///   - instances: Array of lookup instances to prove.
    ///   - transcript: Shared Fiat-Shamir transcript.
    /// - Returns: A batched LogUpGKR proof.
    public func prove(
        instances: [LogUpBatchInstance],
        transcript: Transcript
    ) -> LogUpBatchGKRProof {
        let batchSize = instances.count
        precondition(batchSize > 0, "Must have at least one lookup instance")

        // Domain-separate the batch
        transcript.absorbLabel("logup-batch-gkr")
        transcript.absorb(frFromInt(UInt64(batchSize)))

        // Determine max column count across all instances
        let maxCols = instances.map { $0.witnessColumns.count }.max()!
        transcript.absorb(frFromInt(UInt64(maxCols)))

        // Absorb instance sizes
        for inst in instances {
            transcript.absorb(frFromInt(UInt64(inst.witnessColumns[0].count)))
            transcript.absorb(frFromInt(UInt64(inst.tableColumns[0].count)))
            transcript.absorb(frFromInt(UInt64(inst.witnessColumns.count)))
        }

        // Shared challenges
        let gamma = transcript.squeeze()
        let beta: Fr
        if maxCols > 1 {
            beta = transcript.squeeze()
        } else {
            beta = Fr.zero
        }

        // Batching challenge
        let alpha = transcript.squeeze()

        // Prove each instance independently using sub-transcripts
        // that share the parent transcript state
        var instanceProofs = [LogUpGKRProof]()
        var instanceMultiplicities = [[Fr]]()
        var instanceWitnessSums = [Fr]()
        var instanceTableSums = [Fr]()

        for (idx, inst) in instances.enumerated() {
            transcript.absorbLabel("logup-batch-instance-\(idx)")

            let proof = proveInstance(
                instance: inst,
                gamma: gamma,
                beta: beta,
                transcript: transcript
            )

            instanceMultiplicities.append(proof.multiplicities)
            instanceWitnessSums.append(proof.claimedWitnessSum)
            instanceTableSums.append(proof.claimedTableSum)
            instanceProofs.append(proof)
        }

        // Verify the batched relation: sum_k alpha^k * (witnessSum_k - tableSum_k) = 0
        // (This is checked by the verifier; the prover just provides the data.)
        transcript.absorbLabel("logup-batch-gkr-combined")
        var combinedSum = Fr.zero
        var alphaPow = Fr.one
        for k in 0..<batchSize {
            let diff = frSub(instanceWitnessSums[k], instanceTableSums[k])
            combinedSum = frAdd(combinedSum, frMul(alphaPow, diff))
            alphaPow = frMul(alphaPow, alpha)
        }
        transcript.absorb(combinedSum)

        return LogUpBatchGKRProof(
            batchSize: batchSize,
            gamma: gamma,
            beta: beta,
            alpha: alpha,
            instanceMultiplicities: instanceMultiplicities,
            instanceWitnessSums: instanceWitnessSums,
            instanceTableSums: instanceTableSums,
            instanceProofs: instanceProofs
        )
    }

    /// Prove a single instance with pre-determined gamma and beta.
    private func proveInstance(
        instance: LogUpBatchInstance,
        gamma: Fr,
        beta: Fr,
        transcript: Transcript
    ) -> LogUpGKRProof {
        let numCols = instance.witnessColumns.count
        let n = instance.witnessColumns[0].count
        let N = instance.tableColumns[0].count

        // Combine columns
        let combinedWitness = combineColumns(instance.witnessColumns, beta: beta)
        let combinedTable = combineColumns(instance.tableColumns, beta: beta)

        // Compute multiplicities
        let multiplicities = computeMultiplicities(table: combinedTable, witness: combinedWitness)
        for m in multiplicities { transcript.absorb(m) }
        transcript.absorbLabel("logup-batch-inst-mult")

        // Compute inverse polynomials
        let hf = computeWitnessInverses(witness: combinedWitness, gamma: gamma)
        let ht = computeTableInverses(table: combinedTable, multiplicities: multiplicities, gamma: gamma)

        var witnessSum = Fr.zero
        for v in hf { witnessSum = frAdd(witnessSum, v) }
        var tableSum = Fr.zero
        for v in ht { tableSum = frAdd(tableSum, v) }

        transcript.absorb(witnessSum)
        transcript.absorb(tableSum)
        transcript.absorbLabel("logup-batch-inst-sums")

        // Run sumchecks
        let (wMsgs, wFinalEval) = runSumcheck(
            values: hf, claimedSum: witnessSum, transcript: transcript, label: "logup-batch-inst-witness"
        )
        let (tMsgs, tFinalEval) = runSumcheck(
            values: ht, claimedSum: tableSum, transcript: transcript, label: "logup-batch-inst-table"
        )

        return LogUpGKRProof(
            gamma: gamma,
            beta: beta,
            multiplicities: multiplicities,
            witnessSumcheckMsgs: wMsgs,
            tableSumcheckMsgs: tMsgs,
            claimedWitnessSum: witnessSum,
            claimedTableSum: tableSum,
            witnessFinalEval: wFinalEval,
            tableFinalEval: tFinalEval
        )
    }

    // MARK: - Shared helpers (same as LogUpGKRProver)

    private func combineColumns(_ columns: [[Fr]], beta: Fr) -> [Fr] {
        let numCols = columns.count
        let n = columns[0].count
        if numCols == 1 { return columns[0] }
        var result = columns[numCols - 1]
        for c in stride(from: numCols - 2, through: 0, by: -1) {
            result.withUnsafeMutableBytes { rBuf in
                withUnsafeBytes(of: beta) { bBuf in
                    columns[c].withUnsafeBytes { cBuf in
                        bn254_fr_batch_fma_scalar(
                            rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(n))
                    }
                }
            }
        }
        return result
    }

    private func computeMultiplicities(table: [Fr], witness: [Fr]) -> [Fr] {
        let N = table.count
        var indexed = [(limbs: [UInt64], idx: Int)]()
        indexed.reserveCapacity(N)
        for j in 0..<N { indexed.append((frToInt(table[j]), j)) }
        indexed.sort { a, b in
            for k in stride(from: 3, through: 0, by: -1) {
                if a.limbs[k] != b.limbs[k] { return a.limbs[k] < b.limbs[k] }
            }
            return false
        }
        let sortedKeys = indexed.map { $0.limbs }
        let sortedIndices = indexed.map { $0.idx }

        var counts = [UInt64](repeating: 0, count: N)
        for w in witness {
            let wLimbs = frToInt(w)
            var lo = 0, hi = N - 1
            while lo <= hi {
                let mid = (lo + hi) / 2
                let cmp = batchCompareLimbs(sortedKeys[mid], wLimbs)
                if cmp == 0 { counts[sortedIndices[mid]] += 1; break }
                else if cmp < 0 { lo = mid + 1 }
                else { hi = mid - 1 }
            }
        }
        return counts.map { frFromInt($0) }
    }

    private func batchCompareLimbs(_ a: [UInt64], _ b: [UInt64]) -> Int {
        for k in stride(from: 3, through: 0, by: -1) {
            if a[k] < b[k] { return -1 }
            if a[k] > b[k] { return 1 }
        }
        return 0
    }

    private func computeWitnessInverses(witness: [Fr], gamma: Fr) -> [Fr] {
        let n = witness.count
        var diffs = [Fr](repeating: Fr.zero, count: n)
        var gammaVal0 = gamma
        withUnsafeBytes(of: &gammaVal0) { scalarBuf in
            witness.withUnsafeBytes { wBuf in
                diffs.withUnsafeMutableBytes { dBuf in
                    bn254_fr_batch_scalar_sub_neon(
                        dBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        scalarBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        wBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n))
                }
            }
        }
        return batchInverse(diffs)
    }

    private func computeTableInverses(table: [Fr], multiplicities: [Fr], gamma: Fr) -> [Fr] {
        let N = table.count
        var diffs = [Fr](repeating: Fr.zero, count: N)
        var gammaVal1 = gamma
        withUnsafeBytes(of: &gammaVal1) { scalarBuf in
            table.withUnsafeBytes { tBuf in
                diffs.withUnsafeMutableBytes { dBuf in
                    bn254_fr_batch_scalar_sub_neon(
                        dBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        scalarBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        tBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(N))
                }
            }
        }
        let invs = batchInverse(diffs)
        var result = [Fr](repeating: Fr.zero, count: N)
        multiplicities.withUnsafeBytes { aBuf in
            invs.withUnsafeBytes { bBuf in
                result.withUnsafeMutableBytes { rBuf in
                    bn254_fr_batch_mul_parallel(
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(N))
                }
            }
        }
        return result
    }

    private func batchInverse(_ values: [Fr]) -> [Fr] {
        let n = values.count
        guard n > 0 else { return [] }
        var result = [Fr](repeating: Fr.zero, count: n)
        values.withUnsafeBytes { aBuf in
            result.withUnsafeMutableBytes { rBuf in
                bn254_fr_batch_inverse(
                    aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(n),
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                )
            }
        }
        return result
    }

    private func runSumcheck(
        values: [Fr],
        claimedSum: Fr,
        transcript: Transcript,
        label: String
    ) -> (msgs: [SumcheckRoundMsg], finalEval: Fr) {
        let n = values.count
        let logN = batchCeilLog2(n)
        let paddedN = 1 << logN

        var current = [Fr](repeating: Fr.zero, count: paddedN)
        for i in 0..<n { current[i] = values[i] }

        var msgs = [SumcheckRoundMsg]()
        msgs.reserveCapacity(logN)

        for _ in 0..<logN {
            let half = current.count / 2
            var s0 = Fr.zero, s1 = Fr.zero, s2 = Fr.zero
            for j in 0..<half {
                let v0 = current[j]
                let v1 = current[j + half]
                s0 = frAdd(s0, v0)
                s1 = frAdd(s1, v1)
                s2 = frAdd(s2, frSub(frAdd(v1, v1), v0))
            }
            let msg = SumcheckRoundMsg(s0: s0, s1: s1, s2: s2)
            msgs.append(msg)
            transcript.absorb(s0); transcript.absorb(s1); transcript.absorb(s2)
            let challenge = transcript.squeeze()
            let oneMinusR = frSub(Fr.one, challenge)
            var next = [Fr](repeating: Fr.zero, count: half)
            for j in 0..<half {
                next[j] = frAdd(frMul(oneMinusR, current[j]), frMul(challenge, current[j + half]))
            }
            current = next
        }

        transcript.absorbLabel(label)
        let finalEval = current[0]
        transcript.absorb(finalEval)
        return (msgs, finalEval)
    }
}

// MARK: - Batch LogUp-GKR Verifier

/// Verifies a batched LogUp-GKR proof over multiple lookup tables.
public class LogUpBatchGKRVerifier {

    private let verifier = LogUpGKRVerifier()

    public init() {}

    /// Verify a batched LogUp-GKR proof.
    ///
    /// - Parameters:
    ///   - instances: The lookup instances (same as provided to the prover).
    ///   - proof: The batched proof.
    ///   - transcript: Fiat-Shamir transcript (must match prover's).
    /// - Returns: true if valid.
    public func verify(
        instances: [LogUpBatchInstance],
        proof: LogUpBatchGKRProof,
        transcript: Transcript
    ) -> Bool {
        let batchSize = instances.count
        guard batchSize == proof.batchSize else { return false }
        guard proof.instanceProofs.count == batchSize else { return false }

        // Replay transcript prefix
        transcript.absorbLabel("logup-batch-gkr")
        transcript.absorb(frFromInt(UInt64(batchSize)))

        let maxCols = instances.map { $0.witnessColumns.count }.max()!
        transcript.absorb(frFromInt(UInt64(maxCols)))

        for inst in instances {
            transcript.absorb(frFromInt(UInt64(inst.witnessColumns[0].count)))
            transcript.absorb(frFromInt(UInt64(inst.tableColumns[0].count)))
            transcript.absorb(frFromInt(UInt64(inst.witnessColumns.count)))
        }

        let gamma = transcript.squeeze()
        guard batchFrEqual(gamma, proof.gamma) else { return false }

        let beta: Fr
        if maxCols > 1 {
            beta = transcript.squeeze()
            guard batchFrEqual(beta, proof.beta) else { return false }
        } else {
            beta = Fr.zero
        }

        let alpha = transcript.squeeze()
        guard batchFrEqual(alpha, proof.alpha) else { return false }

        // Verify each instance
        for (idx, inst) in instances.enumerated() {
            transcript.absorbLabel("logup-batch-instance-\(idx)")

            let instProof = proof.instanceProofs[idx]
            let numCols = inst.witnessColumns.count
            let n = inst.witnessColumns[0].count
            let N = inst.tableColumns[0].count

            // Combine columns
            let combinedWitness = combineColumns(inst.witnessColumns, beta: beta)
            let combinedTable = combineColumns(inst.tableColumns, beta: beta)

            // Verify multiplicities
            guard instProof.multiplicities.count == N else { return false }

            var tableIndex = [String: Int]()
            for j in 0..<N {
                let key = frToInt(combinedTable[j]).map { String($0) }.joined(separator: ",")
                tableIndex[key] = j
            }
            var expectedMult = [UInt64](repeating: 0, count: N)
            for i in 0..<n {
                let key = frToInt(combinedWitness[i]).map { String($0) }.joined(separator: ",")
                guard tableIndex[key] != nil else { return false }
                expectedMult[tableIndex[key]!] += 1
            }
            for j in 0..<N {
                guard batchFrEqual(instProof.multiplicities[j], frFromInt(expectedMult[j])) else { return false }
            }

            for m in instProof.multiplicities { transcript.absorb(m) }
            transcript.absorbLabel("logup-batch-inst-mult")

            // Recompute sums with batch-inverted denominators
            var bWitDiffs = [Fr](repeating: Fr.zero, count: n)
            var gammaVal2 = gamma
            withUnsafeBytes(of: &gammaVal2) { scalarBuf in
                combinedWitness.withUnsafeBytes { wBuf in
                    bWitDiffs.withUnsafeMutableBytes { dBuf in
                        bn254_fr_batch_scalar_sub_neon(
                            dBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            scalarBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            wBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(n))
                    }
                }
            }
            let bWitInvs = batchVerifierInverse(bWitDiffs)
            var witnessSum = Fr.zero
            bWitInvs.withUnsafeBytes { buf in
                withUnsafeMutableBytes(of: &witnessSum) { rBuf in
                    bn254_fr_vector_sum(buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                        Int32(n),
                                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
                }
            }

            var bTblDiffs = [Fr](repeating: Fr.zero, count: N)
            var gammaVal3 = gamma
            withUnsafeBytes(of: &gammaVal3) { scalarBuf in
                combinedTable.withUnsafeBytes { tBuf in
                    bTblDiffs.withUnsafeMutableBytes { dBuf in
                        bn254_fr_batch_scalar_sub_neon(
                            dBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            scalarBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            tBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(N))
                    }
                }
            }
            let bTblInvs = batchVerifierInverse(bTblDiffs)
            var tableSum = Fr.zero
            for j in 0..<N {
                if instProof.multiplicities[j].isZero { continue }
                tableSum = frAdd(tableSum, frMul(instProof.multiplicities[j], bTblInvs[j]))
            }

            guard batchFrEqual(instProof.claimedWitnessSum, witnessSum) else { return false }
            guard batchFrEqual(instProof.claimedTableSum, tableSum) else { return false }
            guard batchFrEqual(witnessSum, tableSum) else { return false }

            transcript.absorb(instProof.claimedWitnessSum)
            transcript.absorb(instProof.claimedTableSum)
            transcript.absorbLabel("logup-batch-inst-sums")

            // Verify sumchecks
            guard verifySumcheck(
                msgs: instProof.witnessSumcheckMsgs,
                claimedSum: instProof.claimedWitnessSum,
                finalEval: instProof.witnessFinalEval,
                numElements: n,
                transcript: transcript,
                label: "logup-batch-inst-witness"
            ) else { return false }

            guard verifySumcheck(
                msgs: instProof.tableSumcheckMsgs,
                claimedSum: instProof.claimedTableSum,
                finalEval: instProof.tableFinalEval,
                numElements: N,
                transcript: transcript,
                label: "logup-batch-inst-table"
            ) else { return false }
        }

        // Verify batched relation: sum_k alpha^k * (witnessSum_k - tableSum_k) = 0
        transcript.absorbLabel("logup-batch-gkr-combined")
        var combinedSum = Fr.zero
        var alphaPow = Fr.one
        for k in 0..<batchSize {
            let diff = frSub(proof.instanceWitnessSums[k], proof.instanceTableSums[k])
            combinedSum = frAdd(combinedSum, frMul(alphaPow, diff))
            alphaPow = frMul(alphaPow, alpha)
        }
        // The combined sum must be zero for the proof to be valid
        guard batchFrEqual(combinedSum, Fr.zero) else { return false }
        transcript.absorb(combinedSum)

        return true
    }

    // MARK: - Helpers

    private func combineColumns(_ columns: [[Fr]], beta: Fr) -> [Fr] {
        let numCols = columns.count
        let n = columns[0].count
        if numCols == 1 { return columns[0] }
        var result = columns[numCols - 1]
        for c in stride(from: numCols - 2, through: 0, by: -1) {
            result.withUnsafeMutableBytes { rBuf in
                withUnsafeBytes(of: beta) { bBuf in
                    columns[c].withUnsafeBytes { cBuf in
                        bn254_fr_batch_fma_scalar(
                            rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(n))
                    }
                }
            }
        }
        return result
    }

    private func batchVerifierInverse(_ values: [Fr]) -> [Fr] {
        let n = values.count
        guard n > 0 else { return [] }
        var prefix = [Fr](repeating: Fr.zero, count: n)
        prefix[0] = values[0]
        for i in 1..<n { prefix[i] = frMul(prefix[i - 1], values[i]) }
        var inv = frInverse(prefix[n - 1])
        var result = [Fr](repeating: Fr.zero, count: n)
        for i in stride(from: n - 1, through: 1, by: -1) {
            result[i] = frMul(inv, prefix[i - 1])
            inv = frMul(inv, values[i])
        }
        result[0] = inv
        return result
    }

    private func verifySumcheck(
        msgs: [SumcheckRoundMsg],
        claimedSum: Fr,
        finalEval: Fr,
        numElements: Int,
        transcript: Transcript,
        label: String
    ) -> Bool {
        let logN = batchCeilLog2(numElements)
        guard msgs.count == logN else { return false }

        var currentClaim = claimedSum
        for msg in msgs {
            let sum = frAdd(msg.s0, msg.s1)
            guard batchFrEqual(sum, currentClaim) else { return false }
            transcript.absorb(msg.s0)
            transcript.absorb(msg.s1)
            transcript.absorb(msg.s2)
            let challenge = transcript.squeeze()
            currentClaim = batchLagrangeEval3(s0: msg.s0, s1: msg.s1, s2: msg.s2, at: challenge)
        }

        transcript.absorbLabel(label)
        transcript.absorb(finalEval)

        guard batchFrEqual(currentClaim, finalEval) else { return false }
        return true
    }
}

// MARK: - Batch Helpers

private func batchCeilLog2(_ n: Int) -> Int {
    guard n > 1 else { return n <= 0 ? 0 : 0 }
    return Int(ceil(log2(Double(n))))
}

private func batchFrEqual(_ a: Fr, _ b: Fr) -> Bool {
    return a.v.0 == b.v.0 && a.v.1 == b.v.1 && a.v.2 == b.v.2 && a.v.3 == b.v.3 &&
           a.v.4 == b.v.4 && a.v.5 == b.v.5 && a.v.6 == b.v.6 && a.v.7 == b.v.7
}

private let batchInv2: Fr = frInverse(frAdd(Fr.one, Fr.one))

private func batchLagrangeEval3(s0: Fr, s1: Fr, s2: Fr, at x: Fr) -> Fr {
    let xm1 = frSub(x, Fr.one)
    let xm2 = frSub(x, frAdd(Fr.one, Fr.one))
    let negOne = frSub(Fr.zero, Fr.one)
    let l0 = frMul(frMul(xm1, xm2), batchInv2)
    let l1 = frMul(frMul(x, xm2), negOne)
    let l2 = frMul(frMul(x, xm1), batchInv2)
    return frAdd(frAdd(frMul(s0, l0), frMul(s1, l1)), frMul(s2, l2))
}
