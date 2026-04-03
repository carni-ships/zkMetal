// Sparse Sumcheck Benchmark & Correctness Tests
import zkMetal
import Foundation

public func runSparseSumcheckBench() {
    fputs("\n=== Sparse Sumcheck ===\n", stderr)

    fputs("\n--- Correctness Tests ---\n", stderr)

    // Test 1: Round poly matches dense
    do {
        let dense: [Fr] = [frFromInt(0), frFromInt(5), frFromInt(3), frFromInt(0),
                           frFromInt(7), frFromInt(0), frFromInt(0), frFromInt(2)]
        let sparse = SparseMultilinearPoly(dense: dense)
        fputs("  Sparse nnz=\(sparse.nnz)/\(sparse.domainSize) (\(String(format: "%.0f", sparse.sparsity * 100))% sparse)\n", stderr)

        let (ss0, ss1, ss2) = sparse.roundPoly()
        let (ds0, ds1, ds2) = SumcheckEngine.cpuRoundPoly(evals: dense)
        let roundMatch = frEqual(ss0, ds0) && frEqual(ss1, ds1) && frEqual(ss2, ds2)
        fputs("  Round poly matches dense: \(roundMatch ? "PASS" : "FAIL")\n", stderr)
    }

    // Test 2: Reduce matches dense
    do {
        let dense: [Fr] = [frFromInt(1), frFromInt(0), frFromInt(0), frFromInt(4),
                           frFromInt(0), frFromInt(3), frFromInt(2), frFromInt(0)]
        let sparse = SparseMultilinearPoly(dense: dense)
        let challenge = frFromInt(42)

        let denseReduced = SumcheckEngine.cpuReduce(evals: dense, challenge: challenge)
        let sparseReduced = sparse.reduce(challenge: challenge)
        let sparseReducedDense = sparseReduced.toDense()

        var reduceMatch = true
        for i in 0..<denseReduced.count {
            if !frEqual(denseReduced[i], sparseReducedDense[i]) {
                reduceMatch = false
                fputs("  Mismatch at index \(i)\n", stderr)
                break
            }
        }
        fputs("  Reduce matches dense: \(reduceMatch ? "PASS" : "FAIL")\n", stderr)
    }

    // Test 3: Total sum matches
    do {
        let dense: [Fr] = [frFromInt(10), frFromInt(0), frFromInt(20), frFromInt(0)]
        let sparse = SparseMultilinearPoly(dense: dense)
        var denseSum = Fr.zero
        for v in dense { denseSum = frAdd(denseSum, v) }
        let sparseSum = sparse.totalSum()
        fputs("  Total sum matches: \(frEqual(denseSum, sparseSum) ? "PASS" : "FAIL")\n", stderr)
    }

    // Test 4: Full sparse sumcheck matches dense sumcheck
    do {
        // Create a sparse polynomial with known entries
        let numVars = 4  // domain size 16
        var entries = [Int: Fr]()
        entries[0] = frFromInt(7)
        entries[3] = frFromInt(11)
        entries[5] = frFromInt(2)
        entries[10] = frFromInt(9)
        entries[15] = frFromInt(3)

        let sparse = SparseMultilinearPoly(numVars: numVars, entries: entries)
        let dense = sparse.toDense()

        // Run sparse sumcheck
        var transcript1 = [UInt8]()
        let (sparseRounds, sparseFinal, sparseChallenges) = sparseSumcheck(
            poly: sparse, transcript: &transcript1)

        // Run dense sumcheck (CPU reference) with same challenge derivation
        var current = dense
        var denseRounds = [(Fr, Fr, Fr)]()
        var denseChallenges = [Fr]()
        var transcript2 = [UInt8]()
        for _ in 0..<numVars {
            let rp = SumcheckEngine.cpuRoundPoly(evals: current)
            denseRounds.append(rp)
            // Same Fiat-Shamir
            let rpBytes0 = frToInt(rp.0)
            let rpBytes1 = frToInt(rp.1)
            let rpBytes2 = frToInt(rp.2)
            for limb in rpBytes0 { for b in 0..<8 { transcript2.append(UInt8((limb >> (b*8)) & 0xFF)) } }
            for limb in rpBytes1 { for b in 0..<8 { transcript2.append(UInt8((limb >> (b*8)) & 0xFF)) } }
            for limb in rpBytes2 { for b in 0..<8 { transcript2.append(UInt8((limb >> (b*8)) & 0xFF)) } }
            let ch = blake3DeriveChallenge(transcript2)
            denseChallenges.append(ch)
            let chBytes = frToInt(ch)
            for limb in chBytes { for b in 0..<8 { transcript2.append(UInt8((limb >> (b*8)) & 0xFF)) } }
            current = SumcheckEngine.cpuReduce(evals: current, challenge: ch)
        }
        let denseFinal = current[0]

        // Compare
        var allRoundsMatch = true
        for k in 0..<numVars {
            let (ss0, ss1, ss2) = sparseRounds[k]
            let (ds0, ds1, ds2) = denseRounds[k]
            if !frEqual(ss0, ds0) || !frEqual(ss1, ds1) || !frEqual(ss2, ds2) {
                allRoundsMatch = false
                fputs("  Round \(k) mismatch\n", stderr)
            }
        }
        fputs("  Full sumcheck rounds match dense: \(allRoundsMatch ? "PASS" : "FAIL")\n", stderr)
        fputs("  Final eval matches: \(frEqual(sparseFinal, denseFinal) ? "PASS" : "FAIL")\n", stderr)
        var challengesMatch = true
        for k in 0..<numVars {
            if !frEqual(sparseChallenges[k], denseChallenges[k]) { challengesMatch = false }
        }
        fputs("  Challenges match: \(challengesMatch ? "PASS" : "FAIL")\n", stderr)
    }

    // Test 5: Verify sparse sumcheck proof
    do {
        let numVars = 3
        var entries = [Int: Fr]()
        entries[1] = frFromInt(5)
        entries[4] = frFromInt(8)
        entries[7] = frFromInt(3)

        let sparse = SparseMultilinearPoly(numVars: numVars, entries: entries)
        let claimedSum = sparse.totalSum()

        var proveTranscript = [UInt8]()
        let (rounds, finalEval, _) = sparseSumcheck(poly: sparse, transcript: &proveTranscript)

        var verifyTranscript = [UInt8]()
        let (valid, _) = verifySumcheckProof(
            rounds: rounds, claimedSum: claimedSum, finalEval: finalEval,
            transcript: &verifyTranscript)
        fputs("  Verify sparse sumcheck proof: \(valid ? "PASS" : "FAIL")\n", stderr)

        // Tamper: wrong sum
        var verifyTranscript2 = [UInt8]()
        let wrongSum = frAdd(claimedSum, Fr.one)
        let (rejected, _) = verifySumcheckProof(
            rounds: rounds, claimedSum: wrongSum, finalEval: finalEval,
            transcript: &verifyTranscript2)
        fputs("  Reject wrong sum: \(!rejected ? "PASS" : "FAIL")\n", stderr)
    }

    // Test 6: Very sparse polynomial (1% density)
    do {
        let numVars = 10  // domain = 1024
        let n = 1 << numVars
        var entries = [Int: Fr]()
        var rng: UInt64 = 0xCAFE_BABE
        let targetNNZ = n / 100  // ~1% density = ~10 entries
        for _ in 0..<max(targetNNZ, 1) {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let idx = Int(rng >> 32) % n
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            entries[idx] = frFromInt(rng >> 32)
        }

        let sparse = SparseMultilinearPoly(numVars: numVars, entries: entries)
        let dense = sparse.toDense()

        // Sparse sumcheck
        var t1 = [UInt8]()
        let (sr, sf, sc) = sparseSumcheck(poly: sparse, transcript: &t1)

        // Dense sumcheck (CPU reference)
        var current = dense
        var dr = [(Fr, Fr, Fr)]()
        var t2 = [UInt8]()
        for _ in 0..<numVars {
            let rp = SumcheckEngine.cpuRoundPoly(evals: current)
            dr.append(rp)
            let rpB0 = frToInt(rp.0); let rpB1 = frToInt(rp.1); let rpB2 = frToInt(rp.2)
            for limb in rpB0 { for b in 0..<8 { t2.append(UInt8((limb >> (b*8)) & 0xFF)) } }
            for limb in rpB1 { for b in 0..<8 { t2.append(UInt8((limb >> (b*8)) & 0xFF)) } }
            for limb in rpB2 { for b in 0..<8 { t2.append(UInt8((limb >> (b*8)) & 0xFF)) } }
            let ch = blake3DeriveChallenge(t2)
            let chB = frToInt(ch)
            for limb in chB { for b in 0..<8 { t2.append(UInt8((limb >> (b*8)) & 0xFF)) } }
            current = SumcheckEngine.cpuReduce(evals: current, challenge: ch)
        }

        var match = true
        for k in 0..<numVars {
            let (ss0, ss1, ss2) = sr[k]
            let (ds0, ds1, ds2) = dr[k]
            if !frEqual(ss0, ds0) || !frEqual(ss1, ds1) || !frEqual(ss2, ds2) { match = false }
        }
        fputs("  1% sparse (2^10, nnz=\(sparse.nnz)): \(match ? "PASS" : "FAIL")\n", stderr)
    }

    // --- Performance comparison ---
    if !skipCPU {
        fputs("\n--- Performance: Sparse vs Dense ---\n", stderr)

        for (numVars, density) in [(14, 0.01), (16, 0.01), (18, 0.01), (16, 0.1)] {
            let n = 1 << numVars
            let targetNNZ = max(Int(Double(n) * density), 1)

            // Build sparse poly
            var entries = [Int: Fr]()
            var rng: UInt64 = UInt64(numVars) &* 0xDEAD_BEEF
            for _ in 0..<targetNNZ {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                let idx = Int(rng >> 32) % n
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                entries[idx] = frFromInt(rng >> 32)
            }
            let sparse = SparseMultilinearPoly(numVars: numVars, entries: entries)
            let dense = sparse.toDense()

            // Time sparse sumcheck
            let t0 = CFAbsoluteTimeGetCurrent()
            var ts = [UInt8]()
            let _ = sparseSumcheck(poly: sparse, transcript: &ts)
            let sparseTime = (CFAbsoluteTimeGetCurrent() - t0) * 1000

            // Time dense sumcheck (CPU reference)
            let t1 = CFAbsoluteTimeGetCurrent()
            var current = dense
            var td = [UInt8]()
            for _ in 0..<numVars {
                let rp = SumcheckEngine.cpuRoundPoly(evals: current)
                let rpB0 = frToInt(rp.0); let rpB1 = frToInt(rp.1); let rpB2 = frToInt(rp.2)
                for limb in rpB0 { for b in 0..<8 { td.append(UInt8((limb >> (b*8)) & 0xFF)) } }
                for limb in rpB1 { for b in 0..<8 { td.append(UInt8((limb >> (b*8)) & 0xFF)) } }
                for limb in rpB2 { for b in 0..<8 { td.append(UInt8((limb >> (b*8)) & 0xFF)) } }
                let ch = blake3DeriveChallenge(td)
                let chB = frToInt(ch)
                for limb in chB { for b in 0..<8 { td.append(UInt8((limb >> (b*8)) & 0xFF)) } }
                current = SumcheckEngine.cpuReduce(evals: current, challenge: ch)
            }
            let denseTime = (CFAbsoluteTimeGetCurrent() - t1) * 1000

            let speedup = denseTime / max(sparseTime, 0.001)
            let pct = density * 100
            fputs("  2^\(numVars) \(String(format: "%.0f", pct))% dense (nnz=\(sparse.nnz)): sparse \(String(format: "%.1f", sparseTime))ms, dense \(String(format: "%.1f", denseTime))ms, \(String(format: "%.1f", speedup))× faster\n", stderr)
        }
    }
}

/// Helper: derive a challenge from transcript using Blake3 (same as other engines)
func blake3DeriveChallenge(_ transcript: [UInt8]) -> Fr {
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
