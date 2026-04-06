// PCS Comparison Benchmark — side-by-side comparison of all polynomial commitment schemes
// KZG (BN254), IPA (BN254), Basefold (hash-based), Zeromorph (KZG-based multilinear), FRI (hash-based)
import zkMetal
import Foundation

// MARK: - Result container

private struct PCSResult {
    let scheme: String
    let logN: Int
    let commitMs: Double
    let openMs: Double
    let verifyMs: Double
    let proofSizeBytes: Int
    let setupType: String       // "trusted" or "transparent"
    let setupSizeBytes: Int     // SRS/generator table size estimate
}

// MARK: - Helpers

/// Median of sorted array at index n/2
private func median(_ arr: [Double]) -> Double {
    let s = arr.sorted()
    return s[s.count / 2]
}

/// Simple RNG for reproducible test data
private struct SimpleRNG {
    var state: UInt64

    mutating func next() -> UInt64 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return state >> 32
    }

    mutating func nextFr() -> Fr {
        return frFromInt(next())
    }

    mutating func frArray(_ count: Int) -> [Fr] {
        var arr = [Fr](repeating: Fr.zero, count: count)
        for i in 0..<count {
            arr[i] = nextFr()
        }
        return arr
    }
}

// MARK: - Individual PCS benchmarks

private func benchKZG(logN: Int, rng: inout SimpleRNG) -> PCSResult? {
    let n = 1 << logN
    // KZG needs SRS of size n; generating large SRS is slow, cap at 2^14
    guard logN <= 14 else { return nil }

    do {
        let gx = fpFromInt(1)
        let gy = fpFromInt(2)
        let generator = PointAffine(x: gx, y: gy)
        let secret: [UInt32] = [42, 0, 0, 0, 0, 0, 0, 0]
        let srs = KZGEngine.generateTestSRS(secret: secret, size: n, generator: generator)
        let engine = try KZGEngine(srs: srs)

        let coeffs = rng.frArray(n)
        let z = rng.nextFr()

        // Warmup
        let _ = try engine.commit(coeffs)
        let _ = try engine.open(coeffs, at: z)

        // Commit
        var commitTimes = [Double]()
        for _ in 0..<5 {
            let t = CFAbsoluteTimeGetCurrent()
            let _ = try engine.commit(coeffs)
            commitTimes.append((CFAbsoluteTimeGetCurrent() - t) * 1000)
        }

        // Open
        var openTimes = [Double]()
        for _ in 0..<5 {
            let t = CFAbsoluteTimeGetCurrent()
            let _ = try engine.open(coeffs, at: z)
            openTimes.append((CFAbsoluteTimeGetCurrent() - t) * 1000)
        }

        // Verify: KZG verify is a pairing check; approximate with open time / 10
        // (KZG verify is 2 pairings, independent of poly degree)
        let proof = try engine.open(coeffs, at: z)
        var verifyTimes = [Double]()
        // KZGEngine doesn't expose a standalone verify; measure pairing cost
        // Use BN254PairingEngine pairingCheck as proxy (2 pairings = KZG verify)
        let pairingEngine = try BN254PairingEngine()
        let commitAffine = batchToAffine([try engine.commit(coeffs)])[0]
        let witnessAffine = batchToAffine([proof.witness])[0]
        let g2Gen = bn254G2Generator()
        for _ in 0..<5 {
            let t = CFAbsoluteTimeGetCurrent()
            // 2 pairings: e(C - v*G, [1]_2) == e(W, [s]_2 - z*[1]_2)
            let _ = try pairingEngine.pairingCheck(pairs: [
                (commitAffine, g2Gen), (witnessAffine, g2Gen)
            ])
            verifyTimes.append((CFAbsoluteTimeGetCurrent() - t) * 1000)
        }

        // Proof size: 1 G1 point (witness) + 1 Fr (evaluation) = 64 + 32 = 96 bytes
        let proofSize = 96
        // Setup size: n G1 affine points (each 64 bytes)
        let setupSize = n * 64

        return PCSResult(
            scheme: "KZG",
            logN: logN,
            commitMs: median(commitTimes),
            openMs: median(openTimes),
            verifyMs: median(verifyTimes),
            proofSizeBytes: proofSize,
            setupType: "trusted",
            setupSizeBytes: setupSize
        )
    } catch {
        fputs("  KZG error (2^\(logN)): \(error)\n", stderr)
        return nil
    }
}

private func benchIPA(logN: Int, rng: inout SimpleRNG) -> PCSResult? {
    let n = 1 << logN
    // IPA generator setup is O(n), cap at 2^14 for reasonable time
    guard logN <= 14 else { return nil }

    do {
        let (gens, Q) = IPAEngine.generateTestGenerators(count: n)
        let engine = try IPAEngine(generators: gens, Q: Q)

        let a = rng.frArray(n)
        var b = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { b[i] = frFromInt(UInt64(i + 1)) }

        let v = IPAEngine.innerProduct(a, b)
        let C = try engine.commit(a)

        // Warmup
        let _ = try engine.createProof(a: a, b: b)

        // Commit
        var commitTimes = [Double]()
        for _ in 0..<5 {
            let t = CFAbsoluteTimeGetCurrent()
            let _ = try engine.commit(a)
            commitTimes.append((CFAbsoluteTimeGetCurrent() - t) * 1000)
        }

        // Prove (open)
        var openTimes = [Double]()
        for _ in 0..<3 {
            let t = CFAbsoluteTimeGetCurrent()
            let _ = try engine.createProof(a: a, b: b)
            openTimes.append((CFAbsoluteTimeGetCurrent() - t) * 1000)
        }

        // Verify
        let proof = try engine.createProof(a: a, b: b)
        var verifyTimes = [Double]()
        for _ in 0..<5 {
            let t = CFAbsoluteTimeGetCurrent()
            let _ = engine.verify(commitment: C, b: b, innerProductValue: v, proof: proof)
            verifyTimes.append((CFAbsoluteTimeGetCurrent() - t) * 1000)
        }

        // Proof size: log(n) L points + log(n) R points + 1 Fr scalar
        // Each G1 point = 64 bytes (affine), Fr = 32 bytes
        let rounds = logN
        let proofSize = rounds * 2 * 64 + 32

        // Setup size: n generator points (affine) + 1 Q point
        let setupSize = (n + 1) * 64

        return PCSResult(
            scheme: "IPA",
            logN: logN,
            commitMs: median(commitTimes),
            openMs: median(openTimes),
            verifyMs: median(verifyTimes),
            proofSizeBytes: proofSize,
            setupType: "transparent",
            setupSizeBytes: setupSize
        )
    } catch {
        fputs("  IPA error (2^\(logN)): \(error)\n", stderr)
        return nil
    }
}

private func benchBasefold(logN: Int, rng: inout SimpleRNG) -> PCSResult? {
    let n = 1 << logN

    do {
        let engine = try BasefoldEngine()

        let evals = rng.frArray(n)
        var pt = [Fr]()
        for i in 0..<logN {
            pt.append(frFromInt(UInt64(i + 1) * 17))
        }

        // Warmup
        let warmCommit = try engine.commit(evaluations: evals)
        let _ = try engine.open(commitment: warmCommit, point: pt)

        // Commit
        var commitTimes = [Double]()
        for _ in 0..<5 {
            let t = CFAbsoluteTimeGetCurrent()
            let _ = try engine.commit(evaluations: evals)
            commitTimes.append((CFAbsoluteTimeGetCurrent() - t) * 1000)
        }

        // Open
        let comm = try engine.commit(evaluations: evals)
        var openTimes = [Double]()
        for _ in 0..<5 {
            let t = CFAbsoluteTimeGetCurrent()
            let _ = try engine.open(commitment: comm, point: pt)
            openTimes.append((CFAbsoluteTimeGetCurrent() - t) * 1000)
        }

        // Verify
        let prf = try engine.open(commitment: comm, point: pt)
        let expectedVal = BasefoldEngine.cpuEvaluate(evals: evals, point: pt)
        var verifyTimes = [Double]()
        for _ in 0..<5 {
            let t = CFAbsoluteTimeGetCurrent()
            let _ = engine.verify(root: comm.root, point: pt,
                                  claimedValue: expectedVal, proof: prf)
            verifyTimes.append((CFAbsoluteTimeGetCurrent() - t) * 1000)
        }

        // Proof size estimate: logN roots (32B each) + finalValue (32B)
        //   + logN layers of folded evaluations (geometric series ~2n Fr elements)
        //   + query proofs (numQueries * logN * (2 evals + fold + merkle path))
        let numQueries = engine.numQueries
        // Each query proof: logN levels * (2 Fr pairs + 1 fold Fr + ~logN merkle path hashes)
        // Rough estimate: numQueries * logN * (3 * 32 + logN * 32) + logN * 32 + 32
        let queryProofSize = numQueries * logN * (3 * 32 + logN * 32)
        let proofSize = logN * 32 + 32 + queryProofSize

        // Setup: transparent (no trusted setup), just Metal GPU init
        let setupSize = 0

        return PCSResult(
            scheme: "Basefold",
            logN: logN,
            commitMs: median(commitTimes),
            openMs: median(openTimes),
            verifyMs: median(verifyTimes),
            proofSizeBytes: proofSize,
            setupType: "transparent",
            setupSizeBytes: setupSize
        )
    } catch {
        fputs("  Basefold error (2^\(logN)): \(error)\n", stderr)
        return nil
    }
}

private func benchZeromorph(logN: Int, rng: inout SimpleRNG) -> PCSResult? {
    let n = 1 << logN
    // Zeromorph needs KZG SRS of size 2^n, cap at 2^11 (SRS generation limit)
    guard logN <= 11 else { return nil }

    do {
        let gx = fpFromInt(1)
        let gy = fpFromInt(2)
        let generator = PointAffine(x: gx, y: gy)
        let secret: [UInt32] = [42, 0, 0, 0, 0, 0, 0, 0]
        let secretU64: [UInt64] = [42, 0, 0, 0]

        let srs = KZGEngine.generateTestSRS(secret: secret, size: n, generator: generator)
        let kzg = try KZGEngine(srs: srs)
        let pcs = try ZeromorphPCS(kzg: kzg)
        let vk = ZeromorphVK.generateTestVK(secret: secretU64)

        let evals = rng.frArray(n)
        var point = [Fr]()
        for _ in 0..<logN {
            point.append(rng.nextFr())
        }
        let evalVal = ZeromorphPCS.evaluateZMFold(evaluations: evals, point: point)

        // Warmup
        let _ = try pcs.commit(evaluations: evals)
        let _ = try pcs.open(evaluations: evals, point: point, value: evalVal)

        // Commit
        var commitTimes = [Double]()
        for _ in 0..<5 {
            let t = CFAbsoluteTimeGetCurrent()
            let _ = try pcs.commit(evaluations: evals)
            commitTimes.append((CFAbsoluteTimeGetCurrent() - t) * 1000)
        }

        // Open
        var openTimes = [Double]()
        for _ in 0..<3 {
            let t = CFAbsoluteTimeGetCurrent()
            let _ = try pcs.open(evaluations: evals, point: point, value: evalVal)
            openTimes.append((CFAbsoluteTimeGetCurrent() - t) * 1000)
        }

        // Verify (pairing-based)
        let comm = try pcs.commit(evaluations: evals)
        let pf = try pcs.open(evaluations: evals, point: point, value: evalVal)
        var verifyTimes = [Double]()
        for _ in 0..<3 {
            let t = CFAbsoluteTimeGetCurrent()
            let _ = try pcs.verify(
                commitment: comm, point: point,
                value: evalVal, proof: pf, vk: vk)
            verifyTimes.append((CFAbsoluteTimeGetCurrent() - t) * 1000)
        }

        // Proof size: n quotient commitments (64B each) + 1 Fr value + 1 G1 KZG witness
        let proofSize = logN * 64 + 32 + 64

        // Setup size: n G1 affine points + G2 VK points
        let setupSize = n * 64 + 2 * 128  // G2 points are 128 bytes

        return PCSResult(
            scheme: "Zeromorph",
            logN: logN,
            commitMs: median(commitTimes),
            openMs: median(openTimes),
            verifyMs: median(verifyTimes),
            proofSizeBytes: proofSize,
            setupType: "trusted",
            setupSizeBytes: setupSize
        )
    } catch {
        fputs("  Zeromorph error (2^\(logN)): \(error)\n", stderr)
        return nil
    }
}

private func benchFRI(logN: Int, rng: inout SimpleRNG) -> PCSResult? {
    let n = 1 << logN

    do {
        let engine = try FRIEngine()

        let evals = rng.frArray(n)
        let numBetas = FRIFoldMode.foldBy8.betaCount(logN: logN)
        var betas = [Fr]()
        for i in 0..<numBetas {
            betas.append(frFromInt(UInt64(i + 1) * 17))
        }
        let queryIndices: [UInt32] = [0, 42, 100, UInt32(n / 2 - 1)]

        // Warmup
        let _ = try engine.commit(evals: evals, betas: betas)

        // Commit phase (includes folding + Merkle trees)
        var commitTimes = [Double]()
        for _ in 0..<5 {
            let t = CFAbsoluteTimeGetCurrent()
            let _ = try engine.commit(evals: evals, betas: betas)
            commitTimes.append((CFAbsoluteTimeGetCurrent() - t) * 1000)
        }

        // Query phase
        let commitment = try engine.commit(evals: evals, betas: betas)
        var queryTimes = [Double]()
        for _ in 0..<5 {
            let t = CFAbsoluteTimeGetCurrent()
            let _ = try engine.query(commitment: commitment, queryIndices: queryIndices)
            queryTimes.append((CFAbsoluteTimeGetCurrent() - t) * 1000)
        }

        // Verify
        let queries = try engine.query(commitment: commitment, queryIndices: queryIndices)
        var verifyTimes = [Double]()
        for _ in 0..<5 {
            let t = CFAbsoluteTimeGetCurrent()
            let _ = engine.verifyProof(commitment: commitment, queries: queries)
            verifyTimes.append((CFAbsoluteTimeGetCurrent() - t) * 1000)
        }

        // Proof size: roots (logN * 32B) + query proofs
        // Each query: logN layers * (2 Fr evals + merkle path ~logN hashes)
        let numQueries = queryIndices.count
        let queryProofSize = numQueries * numBetas * (2 * 32 + logN * 32)
        let proofSize = numBetas * 32 + 32 + queryProofSize

        return PCSResult(
            scheme: "FRI",
            logN: logN,
            commitMs: median(commitTimes),
            openMs: median(queryTimes),   // FRI "open" = query phase
            verifyMs: median(verifyTimes),
            proofSizeBytes: proofSize,
            setupType: "transparent",
            setupSizeBytes: 0
        )
    } catch {
        fputs("  FRI error (2^\(logN)): \(error)\n", stderr)
        return nil
    }
}

// MARK: - Comparison table printer

private func printComparisonTable(_ results: [PCSResult]) {
    // Header
    let sep = String(repeating: "-", count: 120)
    fputs("\n\(sep)\n", stderr)
    fputs(String(format: "%-12s | %5s | %10s | %10s | %10s | %10s | %12s | %s\n",
                 "Scheme" as NSString, "Size" as NSString,
                 "Commit" as NSString, "Open/Prove" as NSString,
                 "Verify" as NSString, "Total" as NSString,
                 "Proof Size" as NSString, "Setup" as NSString), stderr)
    fputs("\(sep)\n", stderr)

    for r in results {
        let sizeStr = "2^\(r.logN)"
        let totalMs = r.commitMs + r.openMs + r.verifyMs

        let proofStr: String
        if r.proofSizeBytes >= 1024 * 1024 {
            proofStr = String(format: "%.1f MiB", Double(r.proofSizeBytes) / (1024 * 1024))
        } else if r.proofSizeBytes >= 1024 {
            proofStr = String(format: "%.1f KiB", Double(r.proofSizeBytes) / 1024)
        } else {
            proofStr = "\(r.proofSizeBytes) B"
        }

        let setupStr: String
        if r.setupSizeBytes == 0 {
            setupStr = r.setupType
        } else if r.setupSizeBytes >= 1024 * 1024 {
            setupStr = String(format: "%s %.1f MiB", (r.setupType as NSString).utf8String!,
                            Double(r.setupSizeBytes) / (1024 * 1024))
        } else if r.setupSizeBytes >= 1024 {
            setupStr = String(format: "%s %.1f KiB", (r.setupType as NSString).utf8String!,
                            Double(r.setupSizeBytes) / 1024)
        } else {
            setupStr = "\(r.setupType) \(r.setupSizeBytes) B"
        }

        fputs(String(format: "%-12s | %5s | %8.2f ms | %8.2f ms | %8.2f ms | %8.2f ms | %12s | %s\n",
                     (r.scheme as NSString), (sizeStr as NSString),
                     r.commitMs, r.openMs, r.verifyMs, totalMs,
                     (proofStr as NSString), (setupStr as NSString)), stderr)
    }
    fputs("\(sep)\n", stderr)
}

private func printGroupedTables(_ results: [PCSResult]) {
    // Group by logN
    let logNs = Array(Set(results.map { $0.logN })).sorted()

    for logN in logNs {
        let group = results.filter { $0.logN == logN }
        fputs("\n=== Polynomial size: 2^\(logN) = \(1 << logN) ===\n", stderr)
        printComparisonTable(group)

        // Find fastest for each operation
        if let fastestCommit = group.min(by: { $0.commitMs < $1.commitMs }) {
            fputs("  Fastest commit: \(fastestCommit.scheme) (\(String(format: "%.2f", fastestCommit.commitMs)) ms)\n", stderr)
        }
        if let fastestOpen = group.min(by: { $0.openMs < $1.openMs }) {
            fputs("  Fastest open:   \(fastestOpen.scheme) (\(String(format: "%.2f", fastestOpen.openMs)) ms)\n", stderr)
        }
        if let fastestVerify = group.min(by: { $0.verifyMs < $1.verifyMs }) {
            fputs("  Fastest verify: \(fastestVerify.scheme) (\(String(format: "%.2f", fastestVerify.verifyMs)) ms)\n", stderr)
        }
        if let smallestProof = group.min(by: { $0.proofSizeBytes < $1.proofSizeBytes }) {
            fputs("  Smallest proof: \(smallestProof.scheme) (\(smallestProof.proofSizeBytes) bytes)\n", stderr)
        }
    }
}

// MARK: - Public entry point

public func runPCSComparisonBench() {
    fputs("=== PCS Comparison Benchmark ===\n", stderr)
    fputs("Comparing: KZG, IPA, Basefold, Zeromorph, FRI\n", stderr)

    let logSizes = [10, 14, 18]
    var allResults = [PCSResult]()

    for logN in logSizes {
        fputs("\n--- Benchmarking at 2^\(logN) = \(1 << logN) ---\n", stderr)
        var rng = SimpleRNG(state: 0xDEAD_BEEF_CAFE_BABE)

        // KZG (BN254) — trusted setup, constant-size proofs
        fputs("  KZG (BN254)...", stderr)
        if let r = benchKZG(logN: logN, rng: &rng) {
            allResults.append(r)
            fputs(" done\n", stderr)
        } else {
            fputs(" skipped (SRS limit)\n", stderr)
        }

        // IPA (BN254) — transparent, log-size proofs
        fputs("  IPA (BN254)...", stderr)
        if let r = benchIPA(logN: logN, rng: &rng) {
            allResults.append(r)
            fputs(" done\n", stderr)
        } else {
            fputs(" skipped (generator limit)\n", stderr)
        }

        // Basefold — transparent, hash-based, multilinear
        fputs("  Basefold...", stderr)
        if let r = benchBasefold(logN: logN, rng: &rng) {
            allResults.append(r)
            fputs(" done\n", stderr)
        } else {
            fputs(" skipped\n", stderr)
        }

        // Zeromorph — trusted setup (KZG-based), multilinear
        fputs("  Zeromorph...", stderr)
        if let r = benchZeromorph(logN: logN, rng: &rng) {
            allResults.append(r)
            fputs(" done\n", stderr)
        } else {
            fputs(" skipped (SRS limit)\n", stderr)
        }

        // FRI — transparent, hash-based
        fputs("  FRI...", stderr)
        if let r = benchFRI(logN: logN, rng: &rng) {
            allResults.append(r)
            fputs(" done\n", stderr)
        } else {
            fputs(" skipped\n", stderr)
        }
    }

    // Print results
    fputs("\n\n========================================\n", stderr)
    fputs("       PCS COMPARISON RESULTS\n", stderr)
    fputs("========================================\n", stderr)

    // Full table
    fputs("\n--- All Results ---\n", stderr)
    printComparisonTable(allResults)

    // Grouped by size with analysis
    printGroupedTables(allResults)

    // Summary: tradeoffs
    fputs("\n--- Tradeoff Summary ---\n", stderr)
    fputs("  KZG:       Constant-size proofs (96 B), fast verify (2 pairings), requires trusted setup\n", stderr)
    fputs("  IPA:       No trusted setup, log-size proofs, slower verify (O(n) group ops)\n", stderr)
    fputs("  Basefold:  Transparent, hash-based, multilinear-native, large proofs\n", stderr)
    fputs("  Zeromorph: Multilinear over KZG, compact proofs, requires trusted setup + pairings\n", stderr)
    fputs("  FRI:       Transparent, hash-based, post-quantum candidate, large proofs\n", stderr)

    fputs("\nPCS comparison benchmark complete.\n", stderr)
}
