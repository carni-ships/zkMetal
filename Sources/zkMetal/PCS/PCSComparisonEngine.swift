// PCSComparisonEngine — Unified benchmark and comparison engine for polynomial commitment schemes
//
// Runs all PCS implementations through the same interface and produces comparison tables.
// Helps users choose the right PCS for their use case.

import Foundation
import NeonFieldOps

// MARK: - Scheme Enum

/// All supported polynomial commitment schemes.
public enum PCSScheme: String, CaseIterable, Sendable, Hashable {
    case kzg       = "KZG"
    case ipa       = "IPA"
    case basefold  = "Basefold"
    case fri       = "FRI"
    case brakedown = "Brakedown"
    case zeromorph = "Zeromorph"
    case stir      = "STIR"
    case whir      = "WHIR"
    case pedersen  = "Pedersen"

    /// Whether this scheme requires a trusted setup ceremony.
    public var requiresTrustedSetup: Bool {
        switch self {
        case .kzg, .zeromorph: return true
        case .ipa, .basefold, .fri, .brakedown, .stir, .whir, .pedersen: return false
        }
    }

    /// Whether the scheme is currently implemented and benchmarkable.
    public var isImplemented: Bool {
        switch self {
        case .kzg, .ipa, .basefold, .fri, .zeromorph, .brakedown, .stir, .whir: return true
        case .pedersen: return false
        }
    }

    /// Maximum log-degree supported for benchmarking (SRS/generator limits).
    public var maxLogDegree: Int {
        switch self {
        case .kzg:       return 14
        case .ipa:       return 14
        case .basefold:  return 20
        case .fri:       return 20
        case .zeromorph: return 11
        case .brakedown: return 16
        case .stir:      return 20
        case .whir:      return 20
        case .pedersen:  return 14
        }
    }

    /// Short description of the scheme's properties.
    public var summary: String {
        switch self {
        case .kzg:       return "Pairing-based, trusted setup, constant-size proofs, O(1) verify"
        case .ipa:       return "Discrete-log, transparent, log-size proofs, O(n) verify"
        case .basefold:  return "Hash-based, transparent, multilinear-native, large proofs"
        case .fri:       return "Hash-based, transparent, post-quantum candidate, large proofs"
        case .zeromorph: return "KZG-based multilinear, trusted setup, compact proofs"
        case .brakedown: return "Linear-code, transparent, linear-time prover, large proofs"
        case .stir:      return "Hash-based, transparent, domain-shifted FRI, O(log^2 n) queries"
        case .whir:      return "Hash-based, transparent, weighted-hash FRI, O(log^2 n) queries"
        case .pedersen:  return "Discrete-log commitment (no opening proof), transparent"
        }
    }

    /// Polynomial type: univariate or multilinear.
    public var polynomialType: String {
        switch self {
        case .kzg, .fri, .stir, .whir: return "univariate"
        case .ipa:                      return "univariate (via inner product)"
        case .basefold, .brakedown, .zeromorph: return "multilinear"
        case .pedersen:                 return "vector"
        }
    }

    /// Asymptotic prover complexity.
    public var proverComplexity: String {
        switch self {
        case .kzg:       return "O(n log n)"
        case .ipa:       return "O(n log n)"
        case .basefold:  return "O(n log n)"
        case .fri:       return "O(n log n)"
        case .zeromorph: return "O(n log n)"
        case .brakedown: return "O(n)"
        case .stir:      return "O(n log n)"
        case .whir:      return "O(n log n)"
        case .pedersen:  return "O(n)"
        }
    }

    /// Asymptotic verifier complexity.
    public var verifierComplexity: String {
        switch self {
        case .kzg:       return "O(1) (pairing)"
        case .ipa:       return "O(n)"
        case .basefold:  return "O(log^2 n)"
        case .fri:       return "O(log^2 n)"
        case .zeromorph: return "O(log n) + 1 pairing"
        case .brakedown: return "O(sqrt(n))"
        case .stir:      return "O(log^2 n)"
        case .whir:      return "O(log^2 n)"
        case .pedersen:  return "O(n)"
        }
    }

    /// Asymptotic proof size.
    public var proofSizeComplexity: String {
        switch self {
        case .kzg:       return "O(1)"
        case .ipa:       return "O(log n)"
        case .basefold:  return "O(log^2 n)"
        case .fri:       return "O(log^2 n)"
        case .zeromorph: return "O(log n)"
        case .brakedown: return "O(sqrt(n))"
        case .stir:      return "O(log^2 n)"
        case .whir:      return "O(log^2 n)"
        case .pedersen:  return "N/A"
        }
    }
}

// MARK: - Benchmark Result

/// Result of benchmarking a single PCS at a given degree.
public struct PCSBenchResult: Sendable {
    /// The scheme that was benchmarked.
    public let scheme: PCSScheme
    /// log2 of the polynomial degree.
    public let logDegree: Int
    /// Polynomial degree (2^logDegree).
    public var degree: Int { 1 << logDegree }
    /// Time to set up public parameters (ms).
    public let setupTimeMs: Double
    /// Time to commit to a polynomial (ms).
    public let commitTimeMs: Double
    /// Time to create an opening proof (ms).
    public let openTimeMs: Double
    /// Time to verify an opening proof (ms).
    public let verifyTimeMs: Double
    /// Proof size in bytes.
    public let proofSizeBytes: Int
    /// Setup/SRS size in bytes (0 for transparent schemes).
    public let setupSizeBytes: Int
    /// Whether a trusted setup is required.
    public let requiresTrustedSetup: Bool

    /// Total prover time: setup + commit + open (ms).
    public var totalProverTimeMs: Double { setupTimeMs + commitTimeMs + openTimeMs }

    /// Total end-to-end time (ms).
    public var totalTimeMs: Double { commitTimeMs + openTimeMs + verifyTimeMs }

    public init(scheme: PCSScheme, logDegree: Int,
                setupTimeMs: Double, commitTimeMs: Double,
                openTimeMs: Double, verifyTimeMs: Double,
                proofSizeBytes: Int, setupSizeBytes: Int,
                requiresTrustedSetup: Bool) {
        self.scheme = scheme
        self.logDegree = logDegree
        self.setupTimeMs = setupTimeMs
        self.commitTimeMs = commitTimeMs
        self.openTimeMs = openTimeMs
        self.verifyTimeMs = verifyTimeMs
        self.proofSizeBytes = proofSizeBytes
        self.setupSizeBytes = setupSizeBytes
        self.requiresTrustedSetup = requiresTrustedSetup
    }
}

// MARK: - Comparison Engine

/// Unified benchmark engine that runs all PCS implementations through the same interface.
public class PCSComparisonEngine {
    public static let version = Versions.pcsComparison

    /// Number of iterations for warm benchmarks (median is taken).
    public var iterations: Int = 5

    /// Deterministic RNG for reproducible test polynomials.
    private struct DetRNG {
        var state: UInt64
        mutating func next() -> UInt64 {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return state >> 32
        }
        mutating func nextFr() -> Fr { frFromInt(next()) }
        mutating func frArray(_ count: Int) -> [Fr] {
            var arr = [Fr](repeating: Fr.zero, count: count)
            for i in 0..<count { arr[i] = nextFr() }
            return arr
        }
    }

    public init() {}

    /// Benchmark a single PCS scheme at the given log-degree.
    /// Returns nil if the scheme is not implemented or the degree exceeds its limit.
    public func benchmark(scheme: PCSScheme, logDegree: Int) -> PCSBenchResult? {
        guard scheme.isImplemented else { return nil }
        guard logDegree <= scheme.maxLogDegree else { return nil }

        var rng = DetRNG(state: 0xDEAD_BEEF_CAFE_BABE)

        switch scheme {
        case .kzg:       return benchmarkKZG(logDegree: logDegree, rng: &rng)
        case .ipa:       return benchmarkIPA(logDegree: logDegree, rng: &rng)
        case .basefold:  return benchmarkBasefold(logDegree: logDegree, rng: &rng)
        case .fri:       return benchmarkFRI(logDegree: logDegree, rng: &rng)
        case .zeromorph: return benchmarkZeromorph(logDegree: logDegree, rng: &rng)
        case .brakedown: return benchmarkBrakedown(logDegree: logDegree, rng: &rng)
        case .stir:      return benchmarkSTIR(logDegree: logDegree, rng: &rng)
        case .whir:      return benchmarkWHIR(logDegree: logDegree, rng: &rng)
        case .pedersen:
            return nil  // Pedersen has no opening proof
        }
    }

    /// Benchmark all implemented PCS schemes at the given log-degree.
    public func compareAll(logDegree: Int) -> [PCSBenchResult] {
        var results = [PCSBenchResult]()
        for scheme in PCSScheme.allCases {
            if let r = benchmark(scheme: scheme, logDegree: logDegree) {
                results.append(r)
            }
        }
        return results
    }

    /// Benchmark across multiple sizes and return all results grouped by size.
    public func compareAllSizes(logDegrees: [Int]) -> [PCSBenchResult] {
        var results = [PCSBenchResult]()
        for logN in logDegrees {
            results.append(contentsOf: compareAll(logDegree: logN))
        }
        return results
    }

    /// Benchmark specific schemes at the given log-degree.
    public func compare(schemes: [PCSScheme], logDegree: Int) -> [PCSBenchResult] {
        var results = [PCSBenchResult]()
        for scheme in schemes {
            if let r = benchmark(scheme: scheme, logDegree: logDegree) {
                results.append(r)
            }
        }
        return results
    }

    /// Format a scheme properties table showing static characteristics.
    public static func formatSchemeProperties() -> String {
        var out = ""
        let sep = String(repeating: "-", count: 110)
        out += sep + "\n"
        out += String(format: "%-10s | %-13s | %-12s | %-13s | %-10s | %-8s | %s\n",
                      "Scheme" as NSString, "Poly Type" as NSString,
                      "Proof Size" as NSString, "Verify" as NSString,
                      "Prover" as NSString, "PQ Safe" as NSString,
                      "Setup" as NSString)
        out += sep + "\n"

        for scheme in PCSScheme.allCases {
            let pq = PCSTradeoffAnalysis.isPostQuantum(scheme: scheme) ? "yes" : "no"
            let setup = scheme.requiresTrustedSetup ? "trusted" : "transparent"
            let impl = scheme.isImplemented ? "" : " (N/A)"
            out += String(format: "%-10s | %-13s | %-12s | %-13s | %-10s | %-8s | %s%s\n",
                          (scheme.rawValue as NSString),
                          (scheme.polynomialType as NSString),
                          (scheme.proofSizeComplexity as NSString),
                          (scheme.verifierComplexity as NSString),
                          (scheme.proverComplexity as NSString),
                          (pq as NSString),
                          (setup as NSString),
                          (impl as NSString))
        }
        out += sep
        return out
    }

    /// Find the best scheme for each metric category at the given log-degree.
    public static func findBestPerCategory(_ results: [PCSBenchResult]) -> [String: PCSScheme] {
        guard !results.isEmpty else { return [:] }
        var best = [String: PCSScheme]()
        if let r = results.min(by: { $0.commitTimeMs < $1.commitTimeMs }) {
            best["fastest_commit"] = r.scheme
        }
        if let r = results.min(by: { $0.openTimeMs < $1.openTimeMs }) {
            best["fastest_open"] = r.scheme
        }
        if let r = results.min(by: { $0.verifyTimeMs < $1.verifyTimeMs }) {
            best["fastest_verify"] = r.scheme
        }
        if let r = results.min(by: { $0.totalTimeMs < $1.totalTimeMs }) {
            best["fastest_total"] = r.scheme
        }
        if let r = results.min(by: { $0.proofSizeBytes < $1.proofSizeBytes }) {
            best["smallest_proof"] = r.scheme
        }
        return best
    }

    // MARK: - KZG Benchmark

    private func benchmarkKZG(logDegree: Int, rng: inout DetRNG) -> PCSBenchResult? {
        let n = 1 << logDegree

        do {
            let generator = PointAffine(x: fpFromInt(1), y: fpFromInt(2))
            let secret: [UInt32] = [42, 0, 0, 0, 0, 0, 0, 0]

            // Setup
            let setupStart = CFAbsoluteTimeGetCurrent()
            let srs = KZGEngine.generateTestSRS(secret: secret, size: n, generator: generator)
            let engine = try KZGEngine(srs: srs)
            let setupMs = (CFAbsoluteTimeGetCurrent() - setupStart) * 1000

            let coeffs = rng.frArray(n)
            let z = rng.nextFr()

            // Warmup
            let _ = try engine.commit(coeffs)
            let _ = try engine.open(coeffs, at: z)

            // Commit
            let commitMs = medianTime(iterations) { try! engine.commit(coeffs) }

            // Open
            let openMs = medianTime(iterations) { try! engine.open(coeffs, at: z) }

            // Verify via pairing check
            let pairingEngine = try BN254PairingEngine()
            let commitAffine = batchToAffine([try engine.commit(coeffs)])[0]
            let proof = try engine.open(coeffs, at: z)
            let witnessAffine = batchToAffine([proof.witness])[0]
            let g2Gen = bn254G2Generator()

            let verifyMs = medianTime(iterations) {
                let _ = try! pairingEngine.pairingCheck(pairs: [
                    (commitAffine, g2Gen), (witnessAffine, g2Gen)
                ])
            }

            // Proof size: 1 G1 witness (64B) + 1 Fr evaluation (32B) = 96 bytes
            let proofSize = 96
            let setupSize = n * 64  // n G1 affine points

            return PCSBenchResult(
                scheme: .kzg, logDegree: logDegree,
                setupTimeMs: setupMs, commitTimeMs: commitMs,
                openTimeMs: openMs, verifyTimeMs: verifyMs,
                proofSizeBytes: proofSize, setupSizeBytes: setupSize,
                requiresTrustedSetup: true
            )
        } catch {
            return nil
        }
    }

    // MARK: - IPA Benchmark

    private func benchmarkIPA(logDegree: Int, rng: inout DetRNG) -> PCSBenchResult? {
        let n = 1 << logDegree

        do {
            // Setup
            let setupStart = CFAbsoluteTimeGetCurrent()
            let (gens, Q) = IPAEngine.generateTestGenerators(count: n)
            let engine = try IPAEngine(generators: gens, Q: Q)
            let setupMs = (CFAbsoluteTimeGetCurrent() - setupStart) * 1000

            let a = rng.frArray(n)
            var b = [Fr](repeating: Fr.zero, count: n)
            for i in 0..<n { b[i] = frFromInt(UInt64(i + 1)) }

            let v = IPAEngine.innerProduct(a, b)
            let C = try engine.commit(a)
            let vQ = cPointScalarMul(pointFromAffine(Q), v)
            let Cbound = pointAdd(C, vQ)

            // Warmup
            let _ = try engine.createProof(a: a, b: b)

            // Commit
            let commitMs = medianTime(iterations) { try! engine.commit(a) }

            // Open (prove)
            let openMs = medianTime(min(iterations, 3)) {
                try! engine.createProof(a: a, b: b)
            }

            // Verify
            let proof = try engine.createProof(a: a, b: b)
            let verifyMs = medianTime(iterations) {
                let _ = engine.verify(commitment: Cbound, b: b,
                                      innerProductValue: v, proof: proof)
            }

            // Proof size: logN L points + logN R points + 1 Fr scalar
            let proofSize = logDegree * 2 * 64 + 32
            let setupSize = (n + 1) * 64  // n generators + Q

            return PCSBenchResult(
                scheme: .ipa, logDegree: logDegree,
                setupTimeMs: setupMs, commitTimeMs: commitMs,
                openTimeMs: openMs, verifyTimeMs: verifyMs,
                proofSizeBytes: proofSize, setupSizeBytes: setupSize,
                requiresTrustedSetup: false
            )
        } catch {
            return nil
        }
    }

    // MARK: - Basefold Benchmark

    private func benchmarkBasefold(logDegree: Int, rng: inout DetRNG) -> PCSBenchResult? {
        let n = 1 << logDegree

        do {
            let setupStart = CFAbsoluteTimeGetCurrent()
            let engine = try BasefoldEngine()
            let setupMs = (CFAbsoluteTimeGetCurrent() - setupStart) * 1000

            let evals = rng.frArray(n)
            var pt = [Fr]()
            for i in 0..<logDegree { pt.append(frFromInt(UInt64(i + 1) * 17)) }

            // Warmup
            let warmCommit = try engine.commit(evaluations: evals)
            let _ = try engine.open(commitment: warmCommit, point: pt)

            // Commit
            let commitMs = medianTime(iterations) {
                try! engine.commit(evaluations: evals)
            }

            // Open
            let comm = try engine.commit(evaluations: evals)
            let openMs = medianTime(iterations) {
                try! engine.open(commitment: comm, point: pt)
            }

            // Verify
            let prf = try engine.open(commitment: comm, point: pt)
            let expectedVal = BasefoldEngine.cpuEvaluate(evals: evals, point: pt)
            let verifyMs = medianTime(iterations) {
                let _ = engine.verify(root: comm.root, point: pt,
                                      claimedValue: expectedVal, proof: prf)
            }

            // Proof size estimate
            let numQueries = engine.numQueries
            let queryProofSize = numQueries * logDegree * (3 * 32 + logDegree * 32)
            let proofSize = logDegree * 32 + 32 + queryProofSize

            return PCSBenchResult(
                scheme: .basefold, logDegree: logDegree,
                setupTimeMs: setupMs, commitTimeMs: commitMs,
                openTimeMs: openMs, verifyTimeMs: verifyMs,
                proofSizeBytes: proofSize, setupSizeBytes: 0,
                requiresTrustedSetup: false
            )
        } catch {
            return nil
        }
    }

    // MARK: - FRI Benchmark

    private func benchmarkFRI(logDegree: Int, rng: inout DetRNG) -> PCSBenchResult? {
        let n = 1 << logDegree

        do {
            let setupStart = CFAbsoluteTimeGetCurrent()
            let engine = try FRIEngine()
            let setupMs = (CFAbsoluteTimeGetCurrent() - setupStart) * 1000

            let evals = rng.frArray(n)
            let numBetas = FRIFoldMode.foldBy8.betaCount(logN: logDegree)
            var betas = [Fr]()
            for i in 0..<numBetas { betas.append(frFromInt(UInt64(i + 1) * 17)) }
            let queryIndices: [UInt32] = [0, 42, 100, UInt32(n / 2 - 1)]

            // Warmup
            let _ = try engine.commit(evals: evals, betas: betas)

            // Commit (includes folding + Merkle trees)
            let commitMs = medianTime(iterations) {
                try! engine.commit(evals: evals, betas: betas)
            }

            // Query phase (open)
            let commitment = try engine.commit(evals: evals, betas: betas)
            let openMs = medianTime(iterations) {
                try! engine.query(commitment: commitment, queryIndices: queryIndices)
            }

            // Verify
            let queries = try engine.query(commitment: commitment, queryIndices: queryIndices)
            let verifyMs = medianTime(iterations) {
                let _ = engine.verifyProof(commitment: commitment, queries: queries)
            }

            // Proof size
            let numQueries = queryIndices.count
            let queryProofSize = numQueries * numBetas * (2 * 32 + logDegree * 32)
            let proofSize = numBetas * 32 + 32 + queryProofSize

            return PCSBenchResult(
                scheme: .fri, logDegree: logDegree,
                setupTimeMs: setupMs, commitTimeMs: commitMs,
                openTimeMs: openMs, verifyTimeMs: verifyMs,
                proofSizeBytes: proofSize, setupSizeBytes: 0,
                requiresTrustedSetup: false
            )
        } catch {
            return nil
        }
    }

    // MARK: - Zeromorph Benchmark

    private func benchmarkZeromorph(logDegree: Int, rng: inout DetRNG) -> PCSBenchResult? {
        let n = 1 << logDegree

        do {
            let generator = PointAffine(x: fpFromInt(1), y: fpFromInt(2))
            let secret: [UInt32] = [42, 0, 0, 0, 0, 0, 0, 0]
            let secretU64: [UInt64] = [42, 0, 0, 0]

            let setupStart = CFAbsoluteTimeGetCurrent()
            let srs = KZGEngine.generateTestSRS(secret: secret, size: n, generator: generator)
            let kzg = try KZGEngine(srs: srs)
            let pcs = try ZeromorphPCS(kzg: kzg)
            let vk = ZeromorphVK.generateTestVK(secret: secretU64)
            let setupMs = (CFAbsoluteTimeGetCurrent() - setupStart) * 1000

            let evals = rng.frArray(n)
            var point = [Fr]()
            for _ in 0..<logDegree { point.append(rng.nextFr()) }
            let evalVal = ZeromorphPCS.evaluateZMFold(evaluations: evals, point: point)

            // Warmup
            let _ = try pcs.commit(evaluations: evals)
            let _ = try pcs.open(evaluations: evals, point: point, value: evalVal)

            // Commit
            let commitMs = medianTime(iterations) {
                try! pcs.commit(evaluations: evals)
            }

            // Open
            let openMs = medianTime(min(iterations, 3)) {
                try! pcs.open(evaluations: evals, point: point, value: evalVal)
            }

            // Verify
            let comm = try pcs.commit(evaluations: evals)
            let pf = try pcs.open(evaluations: evals, point: point, value: evalVal)
            let verifyMs = medianTime(min(iterations, 3)) {
                let _ = try! pcs.verify(commitment: comm, point: point,
                                        value: evalVal, proof: pf, vk: vk)
            }

            // Proof size: logN quotient commitments (64B) + 1 Fr + 1 G1 KZG witness
            let proofSize = logDegree * 64 + 32 + 64
            let setupSize = n * 64 + 2 * 128

            return PCSBenchResult(
                scheme: .zeromorph, logDegree: logDegree,
                setupTimeMs: setupMs, commitTimeMs: commitMs,
                openTimeMs: openMs, verifyTimeMs: verifyMs,
                proofSizeBytes: proofSize, setupSizeBytes: setupSize,
                requiresTrustedSetup: true
            )
        } catch {
            return nil
        }
    }

    // MARK: - Brakedown Benchmark

    private func benchmarkBrakedown(logDegree: Int, rng: inout DetRNG) -> PCSBenchResult? {
        let n = 1 << logDegree
        // Brakedown needs even logN for sqrt matrix decomposition
        guard logDegree >= 4 else { return nil }

        do {
            let setupStart = CFAbsoluteTimeGetCurrent()
            let engine = try BrakedownEngine(params: .fast)
            let setupMs = (CFAbsoluteTimeGetCurrent() - setupStart) * 1000

            let evals = rng.frArray(n)
            var point = [Fr]()
            for i in 0..<logDegree { point.append(frFromInt(UInt64(i + 1) * 17)) }

            // Warmup
            let warmCommit = try engine.commit(evaluations: evals)
            let _ = try engine.open(evaluations: evals, point: point, commitment: warmCommit)

            // Commit
            let commitMs = medianTime(iterations) {
                try! engine.commit(evaluations: evals)
            }

            // Open
            let comm = try engine.commit(evaluations: evals)
            let openMs = medianTime(iterations) {
                try! engine.open(evaluations: evals, point: point, commitment: comm)
            }

            // Verify
            let prf = try engine.open(evaluations: evals, point: point, commitment: comm)
            let expectedVal = BrakedownEngine.cpuEvaluate(evaluations: evals, point: point)
            let verifyMs = medianTime(iterations) {
                let _ = engine.verify(commitment: comm, point: point,
                                      value: expectedVal, proof: prf)
            }

            let proofSize = prf.proofSizeBytes

            return PCSBenchResult(
                scheme: .brakedown, logDegree: logDegree,
                setupTimeMs: setupMs, commitTimeMs: commitMs,
                openTimeMs: openMs, verifyTimeMs: verifyMs,
                proofSizeBytes: proofSize, setupSizeBytes: 0,
                requiresTrustedSetup: false
            )
        } catch {
            return nil
        }
    }

    // MARK: - STIR Benchmark

    private func benchmarkSTIR(logDegree: Int, rng: inout DetRNG) -> PCSBenchResult? {
        let n = 1 << logDegree
        guard logDegree >= 4 else { return nil }

        do {
            let setupStart = CFAbsoluteTimeGetCurrent()
            let prover = try STIRProver(numQueries: 4, reductionFactor: 4)
            let setupMs = (CFAbsoluteTimeGetCurrent() - setupStart) * 1000

            let evals = rng.frArray(n)

            // Warmup
            let _ = try prover.prove(evaluations: evals)

            // Commit (STIR commit is part of prove, time Merkle tree construction)
            let commitMs = medianTime(iterations) {
                let _ = try! prover.commit(evaluations: evals)
            }

            // Prove (open)
            let openMs = medianTime(min(iterations, 3)) {
                let _ = try! prover.prove(evaluations: evals)
            }

            // Verify
            let proof = try prover.prove(evaluations: evals)
            let verifyMs = medianTime(iterations) {
                let _ = prover.verifyFull(proof: proof, evaluations: evals)
            }

            let proofSize = proof.proofSizeBytes

            return PCSBenchResult(
                scheme: .stir, logDegree: logDegree,
                setupTimeMs: setupMs, commitTimeMs: commitMs,
                openTimeMs: openMs, verifyTimeMs: verifyMs,
                proofSizeBytes: proofSize, setupSizeBytes: 0,
                requiresTrustedSetup: false
            )
        } catch {
            return nil
        }
    }

    // MARK: - WHIR Benchmark

    private func benchmarkWHIR(logDegree: Int, rng: inout DetRNG) -> PCSBenchResult? {
        let n = 1 << logDegree
        guard logDegree >= 4 else { return nil }

        do {
            let setupStart = CFAbsoluteTimeGetCurrent()
            let prover = try WHIRProver(numQueries: 4, reductionFactor: 4)
            let setupMs = (CFAbsoluteTimeGetCurrent() - setupStart) * 1000

            let evals = rng.frArray(n)

            // Warmup
            let _ = try prover.prove(evaluations: evals)

            // Commit
            let commitMs = medianTime(iterations) {
                let _ = try! prover.commit(evaluations: evals)
            }

            // Prove (open)
            let openMs = medianTime(min(iterations, 3)) {
                let _ = try! prover.prove(evaluations: evals)
            }

            // Verify
            let proof = try prover.prove(evaluations: evals)
            let verifyMs = medianTime(iterations) {
                let _ = prover.verifyFull(proof: proof, evaluations: evals)
            }

            let proofSize = proof.proofSizeBytes

            return PCSBenchResult(
                scheme: .whir, logDegree: logDegree,
                setupTimeMs: setupMs, commitTimeMs: commitMs,
                openTimeMs: openMs, verifyTimeMs: verifyMs,
                proofSizeBytes: proofSize, setupSizeBytes: 0,
                requiresTrustedSetup: false
            )
        } catch {
            return nil
        }
    }

    // MARK: - Timing Utility

    private func medianTime(_ count: Int, _ block: () -> Void) -> Double {
        var times = [Double]()
        times.reserveCapacity(count)
        for _ in 0..<count {
            let t = CFAbsoluteTimeGetCurrent()
            block()
            times.append((CFAbsoluteTimeGetCurrent() - t) * 1000)
        }
        times.sort()
        return times[times.count / 2]
    }
}

// MARK: - Tradeoff Analysis

/// Analyzes PCS tradeoffs and recommends schemes based on user constraints.
public struct PCSTradeoffAnalysis {

    /// Constraints the user may specify when choosing a PCS.
    public struct Constraints {
        /// If true, exclude schemes requiring trusted setup.
        public var noTrustedSetup: Bool = false
        /// Maximum acceptable proof size in bytes (0 = no limit).
        public var maxProofSizeBytes: Int = 0
        /// Prioritize fast verification (weight 0..1).
        public var verifySpeedWeight: Double = 0.33
        /// Prioritize fast proving (weight 0..1).
        public var proveSpeedWeight: Double = 0.33
        /// Prioritize small proofs (weight 0..1).
        public var proofSizeWeight: Double = 0.34

        public init() {}

        public init(noTrustedSetup: Bool = false,
                    maxProofSizeBytes: Int = 0,
                    verifySpeedWeight: Double = 0.33,
                    proveSpeedWeight: Double = 0.33,
                    proofSizeWeight: Double = 0.34) {
            self.noTrustedSetup = noTrustedSetup
            self.maxProofSizeBytes = maxProofSizeBytes
            self.verifySpeedWeight = verifySpeedWeight
            self.proveSpeedWeight = proveSpeedWeight
            self.proofSizeWeight = proofSizeWeight
        }
    }

    /// Score a benchmark result against the given constraints. Higher = better.
    /// Returns nil if the result does not satisfy hard constraints.
    public static func score(_ result: PCSBenchResult,
                             constraints: Constraints,
                             allResults: [PCSBenchResult]) -> Double? {
        // Hard constraints
        if constraints.noTrustedSetup && result.requiresTrustedSetup { return nil }
        if constraints.maxProofSizeBytes > 0 && result.proofSizeBytes > constraints.maxProofSizeBytes {
            return nil
        }

        // Normalize each metric to [0, 1] where 1 = best in the set
        let sameSize = allResults.filter { $0.logDegree == result.logDegree }
        guard !sameSize.isEmpty else { return nil }

        let maxVerify = sameSize.map(\.verifyTimeMs).max() ?? 1
        let maxProve = sameSize.map(\.totalProverTimeMs).max() ?? 1
        let maxProof = Double(sameSize.map(\.proofSizeBytes).max() ?? 1)

        // Invert: lower is better, so score = 1 - (value / max)
        let verifyScore = maxVerify > 0 ? 1.0 - (result.verifyTimeMs / maxVerify) : 1.0
        let proveScore = maxProve > 0 ? 1.0 - (result.totalProverTimeMs / maxProve) : 1.0
        let sizeScore = maxProof > 0 ? 1.0 - (Double(result.proofSizeBytes) / maxProof) : 1.0

        return constraints.verifySpeedWeight * verifyScore
             + constraints.proveSpeedWeight * proveScore
             + constraints.proofSizeWeight * sizeScore
    }

    /// Recommend the best PCS from a set of benchmark results, given constraints.
    /// Returns results sorted by score (best first), excluding those that fail constraints.
    public static func recommend(from results: [PCSBenchResult],
                                 constraints: Constraints) -> [(PCSBenchResult, Double)] {
        var scored = [(PCSBenchResult, Double)]()
        for r in results {
            if let s = score(r, constraints: constraints, allResults: results) {
                scored.append((r, s))
            }
        }
        scored.sort { $0.1 > $1.1 }
        return scored
    }

    /// Security level estimate for each scheme (bits).
    public static func securityLevel(scheme: PCSScheme) -> Int {
        switch scheme {
        case .kzg:       return 128  // BN254 pairing security
        case .ipa:       return 128  // Discrete log on BN254
        case .basefold:  return 128  // Hash-based (configurable)
        case .fri:       return 128  // Hash-based (configurable)
        case .zeromorph: return 128  // Same as KZG (BN254)
        case .brakedown: return 128  // Hash-based
        case .stir:      return 128  // Hash-based (FRI variant)
        case .whir:      return 128  // Hash-based (FRI variant)
        case .pedersen:  return 128  // Discrete log
        }
    }

    /// Whether the scheme is post-quantum secure.
    public static func isPostQuantum(scheme: PCSScheme) -> Bool {
        switch scheme {
        case .fri, .basefold, .brakedown, .stir, .whir: return true
        case .kzg, .ipa, .zeromorph, .pedersen: return false
        }
    }

    /// Memory usage category for the prover.
    public static func memoryCategory(scheme: PCSScheme, logDegree: Int) -> String {
        let n = 1 << logDegree
        switch scheme {
        case .kzg:       return formatBytes(n * 64)           // SRS
        case .ipa:       return formatBytes((n + 1) * 64)     // generators
        case .basefold:  return formatBytes(n * 32 * 2)       // evals + Merkle
        case .fri:       return formatBytes(n * 32 * 4)       // blowup factor
        case .zeromorph: return formatBytes(n * 64 + n * 32)  // SRS + quotients
        case .brakedown: return formatBytes(n * 32 * 4)       // matrix + encoded
        case .stir:      return formatBytes(n * 32 * 4)       // evals + shifted domains
        case .whir:      return formatBytes(n * 32 * 4)       // evals + weighted hashes
        case .pedersen:  return formatBytes(n * 64)            // generators
        }
    }

    private static func formatBytes(_ bytes: Int) -> String {
        if bytes >= 1024 * 1024 {
            return String(format: "%.1f MiB", Double(bytes) / (1024.0 * 1024.0))
        } else if bytes >= 1024 {
            return String(format: "%.1f KiB", Double(bytes) / 1024.0)
        } else {
            return "\(bytes) B"
        }
    }
}

// MARK: - Pretty-printed Comparison Table

extension PCSComparisonEngine {

    /// Format a human-readable comparison table from benchmark results.
    public static func formatComparisonTable(_ results: [PCSBenchResult]) -> String {
        guard !results.isEmpty else { return "(no results)" }

        var out = ""
        let sep = String(repeating: "-", count: 120)

        out += "\(sep)\n"
        out += String(format: "%-12s | %5s | %10s | %10s | %10s | %10s | %12s | %s\n",
                      "Scheme" as NSString, "Size" as NSString,
                      "Commit" as NSString, "Open" as NSString,
                      "Verify" as NSString, "Total" as NSString,
                      "Proof Size" as NSString, "Setup" as NSString)
        out += "\(sep)\n"

        for r in results {
            let sizeStr = "2^\(r.logDegree)"
            let proofStr = PCSTradeoffAnalysis.formatBytesPublic(r.proofSizeBytes)
            let setupStr: String
            if r.setupSizeBytes == 0 {
                setupStr = r.requiresTrustedSetup ? "trusted" : "transparent"
            } else {
                let kind = r.requiresTrustedSetup ? "trusted" : "transparent"
                setupStr = "\(kind) \(PCSTradeoffAnalysis.formatBytesPublic(r.setupSizeBytes))"
            }

            out += String(format: "%-12s | %5s | %8.2f ms | %8.2f ms | %8.2f ms | %8.2f ms | %12s | %s\n",
                          (r.scheme.rawValue as NSString), (sizeStr as NSString),
                          r.commitTimeMs, r.openTimeMs, r.verifyTimeMs, r.totalTimeMs,
                          (proofStr as NSString), (setupStr as NSString))
        }
        out += sep
        return out
    }

    /// Format a grouped comparison table with per-size analysis.
    public static func formatGroupedTable(_ results: [PCSBenchResult]) -> String {
        guard !results.isEmpty else { return "(no results)" }

        var out = ""
        let logNs = Array(Set(results.map(\.logDegree))).sorted()

        for logN in logNs {
            let group = results.filter { $0.logDegree == logN }
            out += "\n=== Polynomial size: 2^\(logN) = \(1 << logN) ===\n"
            out += formatComparisonTable(group)

            if let fastest = group.min(by: { $0.commitTimeMs < $1.commitTimeMs }) {
                out += String(format: "\n  Fastest commit: %s (%.2f ms)",
                              fastest.scheme.rawValue, fastest.commitTimeMs)
            }
            if let fastest = group.min(by: { $0.openTimeMs < $1.openTimeMs }) {
                out += String(format: "\n  Fastest open:   %s (%.2f ms)",
                              fastest.scheme.rawValue, fastest.openTimeMs)
            }
            if let fastest = group.min(by: { $0.verifyTimeMs < $1.verifyTimeMs }) {
                out += String(format: "\n  Fastest verify: %s (%.2f ms)",
                              fastest.scheme.rawValue, fastest.verifyTimeMs)
            }
            if let smallest = group.min(by: { $0.proofSizeBytes < $1.proofSizeBytes }) {
                out += String(format: "\n  Smallest proof: %s (%d bytes)",
                              smallest.scheme.rawValue, smallest.proofSizeBytes)
            }
            out += "\n"
        }
        return out
    }

    /// Format a recommendation report.
    public static func formatRecommendation(results: [PCSBenchResult],
                                            constraints: PCSTradeoffAnalysis.Constraints) -> String {
        let ranked = PCSTradeoffAnalysis.recommend(from: results, constraints: constraints)
        guard !ranked.isEmpty else { return "No PCS satisfies the given constraints." }

        var out = "=== PCS Recommendation ===\n"
        out += "Constraints:"
        if constraints.noTrustedSetup { out += " no-trusted-setup" }
        if constraints.maxProofSizeBytes > 0 {
            out += " max-proof=\(PCSTradeoffAnalysis.formatBytesPublic(constraints.maxProofSizeBytes))"
        }
        out += String(format: " weights(verify=%.0f%% prove=%.0f%% size=%.0f%%)",
                      constraints.verifySpeedWeight * 100,
                      constraints.proveSpeedWeight * 100,
                      constraints.proofSizeWeight * 100)
        out += "\n\n"

        for (i, (r, s)) in ranked.enumerated() {
            let marker = i == 0 ? " <-- RECOMMENDED" : ""
            out += String(format: "  #%d %s (score: %.3f)%s\n",
                          i + 1, r.scheme.rawValue, s, marker)
            out += "     \(r.scheme.summary)\n"
            out += String(format: "     commit=%.2fms open=%.2fms verify=%.2fms proof=%s\n",
                          r.commitTimeMs, r.openTimeMs, r.verifyTimeMs,
                          PCSTradeoffAnalysis.formatBytesPublic(r.proofSizeBytes))
            let pq = PCSTradeoffAnalysis.isPostQuantum(scheme: r.scheme) ? "yes" : "no"
            out += "     post-quantum: \(pq), memory: \(PCSTradeoffAnalysis.memoryCategory(scheme: r.scheme, logDegree: r.logDegree))\n"
        }
        return out
    }
}

// Public bytes formatter (needed by formatComparisonTable)
extension PCSTradeoffAnalysis {
    public static func formatBytesPublic(_ bytes: Int) -> String {
        return formatBytes(bytes)
    }
}
