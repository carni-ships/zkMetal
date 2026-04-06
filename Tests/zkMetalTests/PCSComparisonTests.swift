// PCSComparisonTests — Tests for PCSComparisonEngine
// Validates all PCS schemes commit+open+verify, comparison output, and recommendation logic.

import zkMetal
import Foundation

public func runPCSComparisonTests() {

    // MARK: - PCSScheme enum tests

    suite("PCSScheme properties")

    expect(PCSScheme.kzg.requiresTrustedSetup, "KZG requires trusted setup")
    expect(PCSScheme.zeromorph.requiresTrustedSetup, "Zeromorph requires trusted setup")
    expect(!PCSScheme.ipa.requiresTrustedSetup, "IPA is transparent")
    expect(!PCSScheme.basefold.requiresTrustedSetup, "Basefold is transparent")
    expect(!PCSScheme.fri.requiresTrustedSetup, "FRI is transparent")
    expect(!PCSScheme.brakedown.requiresTrustedSetup, "Brakedown is transparent")
    expect(!PCSScheme.stir.requiresTrustedSetup, "STIR is transparent")
    expect(!PCSScheme.whir.requiresTrustedSetup, "WHIR is transparent")

    expect(PCSScheme.kzg.isImplemented, "KZG is implemented")
    expect(PCSScheme.ipa.isImplemented, "IPA is implemented")
    expect(PCSScheme.basefold.isImplemented, "Basefold is implemented")
    expect(PCSScheme.fri.isImplemented, "FRI is implemented")
    expect(PCSScheme.zeromorph.isImplemented, "Zeromorph is implemented")
    expect(PCSScheme.brakedown.isImplemented, "Brakedown is implemented")
    expect(PCSScheme.stir.isImplemented, "STIR is implemented")
    expect(PCSScheme.whir.isImplemented, "WHIR is implemented")
    expect(!PCSScheme.pedersen.isImplemented, "Pedersen not yet implemented")

    expectEqual(PCSScheme.allCases.count, 9, "9 PCS schemes in enum")

    // MARK: - Scheme property accessors

    suite("PCSScheme static properties")

    expect(!PCSScheme.kzg.polynomialType.isEmpty, "KZG has poly type")
    expect(!PCSScheme.fri.proverComplexity.isEmpty, "FRI has prover complexity")
    expect(!PCSScheme.ipa.verifierComplexity.isEmpty, "IPA has verifier complexity")
    expect(!PCSScheme.brakedown.proofSizeComplexity.isEmpty, "Brakedown has proof size complexity")
    expectEqual(PCSScheme.brakedown.proverComplexity, "O(n)", "Brakedown is linear-time prover")
    expectEqual(PCSScheme.kzg.proofSizeComplexity, "O(1)", "KZG has constant-size proofs")

    // MARK: - KZG commit + open + verify

    suite("KZG commit + open + verify via engine")
    do {
        let engine = PCSComparisonEngine()
        engine.iterations = 2  // fast for tests

        let result = engine.benchmark(scheme: .kzg, logDegree: 8)
        expect(result != nil, "KZG benchmark at 2^8 should succeed")

        if let r = result {
            expectEqual(r.scheme, .kzg, "scheme is KZG")
            expectEqual(r.logDegree, 8, "logDegree is 8")
            expectEqual(r.degree, 256, "degree is 256")
            expect(r.commitTimeMs > 0, "commit time > 0")
            expect(r.openTimeMs > 0, "open time > 0")
            expect(r.verifyTimeMs > 0, "verify time > 0")
            expectEqual(r.proofSizeBytes, 96, "KZG proof = 96 bytes")
            expect(r.requiresTrustedSetup, "KZG requires trusted setup")
            expect(r.setupSizeBytes > 0, "KZG setup size > 0")
        }
    }

    // KZG exceeding max degree returns nil
    do {
        let engine = PCSComparisonEngine()
        let result = engine.benchmark(scheme: .kzg, logDegree: 20)
        expect(result == nil, "KZG at 2^20 should return nil (exceeds max)")
    }

    // MARK: - IPA commit + open + verify

    suite("IPA commit + open + verify via engine")
    do {
        let engine = PCSComparisonEngine()
        engine.iterations = 2

        let result = engine.benchmark(scheme: .ipa, logDegree: 8)
        expect(result != nil, "IPA benchmark at 2^8 should succeed")

        if let r = result {
            expectEqual(r.scheme, .ipa, "scheme is IPA")
            expectEqual(r.logDegree, 8, "logDegree is 8")
            expect(r.commitTimeMs > 0, "commit time > 0")
            expect(r.openTimeMs > 0, "open time > 0")
            expect(r.verifyTimeMs > 0, "verify time > 0")
            // IPA proof: logN * 2 * 64 + 32 = 8 * 128 + 32 = 1056
            expectEqual(r.proofSizeBytes, 8 * 2 * 64 + 32, "IPA proof size at logN=8")
            expect(!r.requiresTrustedSetup, "IPA is transparent")
        }
    }

    // MARK: - Basefold benchmark

    suite("Basefold benchmark via engine")
    do {
        let engine = PCSComparisonEngine()
        engine.iterations = 2

        let result = engine.benchmark(scheme: .basefold, logDegree: 10)
        expect(result != nil, "Basefold benchmark at 2^10 should succeed")

        if let r = result {
            expectEqual(r.scheme, .basefold, "scheme is Basefold")
            expect(!r.requiresTrustedSetup, "Basefold is transparent")
            expectEqual(r.setupSizeBytes, 0, "Basefold has no SRS")
            expect(r.proofSizeBytes > 0, "proof size > 0")
        }
    }

    // MARK: - FRI benchmark

    suite("FRI benchmark via engine")
    do {
        let engine = PCSComparisonEngine()
        engine.iterations = 2

        let result = engine.benchmark(scheme: .fri, logDegree: 10)
        expect(result != nil, "FRI benchmark at 2^10 should succeed")

        if let r = result {
            expectEqual(r.scheme, .fri, "scheme is FRI")
            expect(!r.requiresTrustedSetup, "FRI is transparent")
            expectEqual(r.setupSizeBytes, 0, "FRI has no SRS")
        }
    }

    // MARK: - Brakedown benchmark

    suite("Brakedown benchmark via engine")
    do {
        let engine = PCSComparisonEngine()
        engine.iterations = 2

        let result = engine.benchmark(scheme: .brakedown, logDegree: 10)
        expect(result != nil, "Brakedown benchmark at 2^10 should succeed")

        if let r = result {
            expectEqual(r.scheme, .brakedown, "scheme is Brakedown")
            expect(!r.requiresTrustedSetup, "Brakedown is transparent")
            expectEqual(r.setupSizeBytes, 0, "Brakedown has no SRS")
            expect(r.proofSizeBytes > 0, "Brakedown proof size > 0")
            expect(r.commitTimeMs > 0, "Brakedown commit time > 0")
            expect(r.openTimeMs > 0, "Brakedown open time > 0")
            expect(r.verifyTimeMs > 0, "Brakedown verify time > 0")
        }
    }

    // MARK: - STIR benchmark

    suite("STIR benchmark via engine")
    do {
        let engine = PCSComparisonEngine()
        engine.iterations = 2

        let result = engine.benchmark(scheme: .stir, logDegree: 10)
        expect(result != nil, "STIR benchmark at 2^10 should succeed")

        if let r = result {
            expectEqual(r.scheme, .stir, "scheme is STIR")
            expect(!r.requiresTrustedSetup, "STIR is transparent")
            expectEqual(r.setupSizeBytes, 0, "STIR has no SRS")
            expect(r.proofSizeBytes > 0, "STIR proof size > 0")
            expect(r.commitTimeMs > 0, "STIR commit time > 0")
            expect(r.openTimeMs > 0, "STIR open time > 0")
        }
    }

    // MARK: - WHIR benchmark

    suite("WHIR benchmark via engine")
    do {
        let engine = PCSComparisonEngine()
        engine.iterations = 2

        let result = engine.benchmark(scheme: .whir, logDegree: 10)
        expect(result != nil, "WHIR benchmark at 2^10 should succeed")

        if let r = result {
            expectEqual(r.scheme, .whir, "scheme is WHIR")
            expect(!r.requiresTrustedSetup, "WHIR is transparent")
            expectEqual(r.setupSizeBytes, 0, "WHIR has no SRS")
            expect(r.proofSizeBytes > 0, "WHIR proof size > 0")
            expect(r.commitTimeMs > 0, "WHIR commit time > 0")
            expect(r.openTimeMs > 0, "WHIR open time > 0")
        }
    }

    // MARK: - Unimplemented scheme returns nil

    suite("Unimplemented schemes return nil")
    do {
        let engine = PCSComparisonEngine()
        expect(engine.benchmark(scheme: .pedersen, logDegree: 10) == nil,
               "Pedersen not implemented")
    }

    // MARK: - compareAll

    suite("compareAll returns multiple results")
    do {
        let engine = PCSComparisonEngine()
        engine.iterations = 2

        let results = engine.compareAll(logDegree: 8)
        // At logDegree 8, KZG/IPA/Basefold/FRI/Zeromorph/Brakedown/STIR/WHIR should all work
        expect(results.count >= 6, "compareAll(8) returns >= 6 results, got \(results.count)")

        // All results should have the same logDegree
        for r in results {
            expectEqual(r.logDegree, 8, "\(r.scheme.rawValue) logDegree = 8")
        }

        // Check schemes are unique
        let schemes = Set(results.map(\.scheme))
        expectEqual(schemes.count, results.count, "all schemes unique")
    }

    // MARK: - compare (subset)

    suite("compare specific schemes")
    do {
        let engine = PCSComparisonEngine()
        engine.iterations = 2

        let results = engine.compare(schemes: [.kzg, .ipa], logDegree: 8)
        expectEqual(results.count, 2, "compare 2 schemes returns 2 results")
        let schemes = results.map(\.scheme)
        expect(schemes.contains(.kzg), "KZG in results")
        expect(schemes.contains(.ipa), "IPA in results")
    }

    // MARK: - findBestPerCategory

    suite("findBestPerCategory")
    do {
        let results = [
            PCSBenchResult(scheme: .kzg, logDegree: 10,
                           setupTimeMs: 100, commitTimeMs: 1.5,
                           openTimeMs: 2.0, verifyTimeMs: 0.5,
                           proofSizeBytes: 96, setupSizeBytes: 65536,
                           requiresTrustedSetup: true),
            PCSBenchResult(scheme: .ipa, logDegree: 10,
                           setupTimeMs: 50, commitTimeMs: 5.0,
                           openTimeMs: 20.0, verifyTimeMs: 15.0,
                           proofSizeBytes: 1312, setupSizeBytes: 65600,
                           requiresTrustedSetup: false),
            PCSBenchResult(scheme: .fri, logDegree: 10,
                           setupTimeMs: 5, commitTimeMs: 3.0,
                           openTimeMs: 1.0, verifyTimeMs: 2.0,
                           proofSizeBytes: 50000, setupSizeBytes: 0,
                           requiresTrustedSetup: false),
        ]

        let best = PCSComparisonEngine.findBestPerCategory(results)
        expectEqual(best["fastest_commit"], .kzg, "KZG fastest commit")
        expectEqual(best["fastest_verify"], .kzg, "KZG fastest verify")
        expectEqual(best["smallest_proof"], .kzg, "KZG smallest proof")
        expectEqual(best["fastest_open"], .fri, "FRI fastest open")
    }

    // MARK: - Comparison table output

    suite("Comparison table formatting")
    do {
        // Create synthetic results for deterministic table testing
        let results = [
            PCSBenchResult(scheme: .kzg, logDegree: 10,
                           setupTimeMs: 100, commitTimeMs: 1.5,
                           openTimeMs: 2.0, verifyTimeMs: 3.0,
                           proofSizeBytes: 96, setupSizeBytes: 65536,
                           requiresTrustedSetup: true),
            PCSBenchResult(scheme: .ipa, logDegree: 10,
                           setupTimeMs: 50, commitTimeMs: 5.0,
                           openTimeMs: 20.0, verifyTimeMs: 15.0,
                           proofSizeBytes: 1312, setupSizeBytes: 65600,
                           requiresTrustedSetup: false),
        ]

        let table = PCSComparisonEngine.formatComparisonTable(results)
        expect(table.contains("KZG"), "table contains KZG")
        expect(table.contains("IPA"), "table contains IPA")
        expect(table.contains("2^10"), "table contains degree")
        expect(table.contains("Commit"), "table has commit header")
        expect(table.contains("Verify"), "table has verify header")
        expect(table.contains("Proof Size"), "table has proof size header")

        let grouped = PCSComparisonEngine.formatGroupedTable(results)
        expect(grouped.contains("Fastest commit"), "grouped table has fastest commit")
        expect(grouped.contains("Smallest proof"), "grouped table has smallest proof")
    }

    // MARK: - Scheme properties table

    suite("Scheme properties table")
    do {
        let propsTable = PCSComparisonEngine.formatSchemeProperties()
        expect(propsTable.contains("KZG"), "props table has KZG")
        expect(propsTable.contains("STIR"), "props table has STIR")
        expect(propsTable.contains("WHIR"), "props table has WHIR")
        expect(propsTable.contains("Brakedown"), "props table has Brakedown")
        expect(propsTable.contains("O(1)"), "props table has O(1) for KZG proof size")
        expect(propsTable.contains("O(n)"), "props table has O(n) for Brakedown prover")
        expect(propsTable.contains("transparent"), "props table has transparent")
        expect(propsTable.contains("trusted"), "props table has trusted")
    }

    // MARK: - Recommendation engine

    suite("PCSTradeoffAnalysis recommendation")
    do {
        let results = [
            PCSBenchResult(scheme: .kzg, logDegree: 10,
                           setupTimeMs: 100, commitTimeMs: 1.5,
                           openTimeMs: 2.0, verifyTimeMs: 0.5,
                           proofSizeBytes: 96, setupSizeBytes: 65536,
                           requiresTrustedSetup: true),
            PCSBenchResult(scheme: .ipa, logDegree: 10,
                           setupTimeMs: 50, commitTimeMs: 5.0,
                           openTimeMs: 20.0, verifyTimeMs: 15.0,
                           proofSizeBytes: 1312, setupSizeBytes: 65600,
                           requiresTrustedSetup: false),
            PCSBenchResult(scheme: .fri, logDegree: 10,
                           setupTimeMs: 5, commitTimeMs: 3.0,
                           openTimeMs: 1.0, verifyTimeMs: 2.0,
                           proofSizeBytes: 50000, setupSizeBytes: 0,
                           requiresTrustedSetup: false),
        ]

        // No constraints: KZG should score well (small proofs, fast verify)
        let defaultConstraints = PCSTradeoffAnalysis.Constraints()
        let ranked = PCSTradeoffAnalysis.recommend(from: results, constraints: defaultConstraints)
        expectEqual(ranked.count, 3, "all 3 schemes eligible with no constraints")
        expect(ranked[0].1 > 0, "top score > 0")

        // No trusted setup: KZG excluded
        var noSetup = PCSTradeoffAnalysis.Constraints()
        noSetup.noTrustedSetup = true
        let noSetupRanked = PCSTradeoffAnalysis.recommend(from: results, constraints: noSetup)
        expectEqual(noSetupRanked.count, 2, "KZG excluded with noTrustedSetup")
        for (r, _) in noSetupRanked {
            expect(r.scheme != .kzg, "KZG excluded from no-trusted-setup results")
        }

        // Max proof size: only KZG fits
        var smallProof = PCSTradeoffAnalysis.Constraints()
        smallProof.maxProofSizeBytes = 200
        let smallRanked = PCSTradeoffAnalysis.recommend(from: results, constraints: smallProof)
        expectEqual(smallRanked.count, 1, "only KZG fits under 200 bytes")
        if let first = smallRanked.first {
            expectEqual(first.0.scheme, .kzg, "KZG is the only small-proof option")
        }

        // Conflicting: no trusted setup + small proofs = nothing
        var impossible = PCSTradeoffAnalysis.Constraints()
        impossible.noTrustedSetup = true
        impossible.maxProofSizeBytes = 100
        let impossibleRanked = PCSTradeoffAnalysis.recommend(from: results, constraints: impossible)
        expectEqual(impossibleRanked.count, 0, "impossible constraints yield empty results")
    }

    // MARK: - Score function

    suite("PCSTradeoffAnalysis score normalization")
    do {
        let results = [
            PCSBenchResult(scheme: .kzg, logDegree: 10,
                           setupTimeMs: 0, commitTimeMs: 1.0,
                           openTimeMs: 1.0, verifyTimeMs: 1.0,
                           proofSizeBytes: 100, setupSizeBytes: 0,
                           requiresTrustedSetup: true),
            PCSBenchResult(scheme: .ipa, logDegree: 10,
                           setupTimeMs: 0, commitTimeMs: 10.0,
                           openTimeMs: 10.0, verifyTimeMs: 10.0,
                           proofSizeBytes: 1000, setupSizeBytes: 0,
                           requiresTrustedSetup: false),
        ]

        let c = PCSTradeoffAnalysis.Constraints()
        let kzgScore = PCSTradeoffAnalysis.score(results[0], constraints: c, allResults: results)
        let ipaScore = PCSTradeoffAnalysis.score(results[1], constraints: c, allResults: results)

        expect(kzgScore != nil, "KZG score not nil")
        expect(ipaScore != nil, "IPA score not nil")
        if let ks = kzgScore, let is_ = ipaScore {
            expect(ks > is_, "KZG scores higher than IPA (faster + smaller proofs)")
        }
    }

    // MARK: - Security / post-quantum properties

    suite("PCS security properties")

    expect(PCSTradeoffAnalysis.isPostQuantum(scheme: .fri), "FRI is post-quantum")
    expect(PCSTradeoffAnalysis.isPostQuantum(scheme: .basefold), "Basefold is post-quantum")
    expect(PCSTradeoffAnalysis.isPostQuantum(scheme: .brakedown), "Brakedown is post-quantum")
    expect(PCSTradeoffAnalysis.isPostQuantum(scheme: .stir), "STIR is post-quantum")
    expect(PCSTradeoffAnalysis.isPostQuantum(scheme: .whir), "WHIR is post-quantum")
    expect(!PCSTradeoffAnalysis.isPostQuantum(scheme: .kzg), "KZG is not post-quantum")
    expect(!PCSTradeoffAnalysis.isPostQuantum(scheme: .ipa), "IPA is not post-quantum")

    expectEqual(PCSTradeoffAnalysis.securityLevel(scheme: .kzg), 128, "KZG 128-bit")
    expectEqual(PCSTradeoffAnalysis.securityLevel(scheme: .fri), 128, "FRI 128-bit")

    // Memory category returns non-empty string for all schemes
    for scheme in PCSScheme.allCases {
        let mem = PCSTradeoffAnalysis.memoryCategory(scheme: scheme, logDegree: 10)
        expect(!mem.isEmpty, "\(scheme.rawValue) memory category non-empty")
    }

    // MARK: - Recommendation report formatting

    suite("Recommendation report formatting")
    do {
        let results = [
            PCSBenchResult(scheme: .kzg, logDegree: 10,
                           setupTimeMs: 100, commitTimeMs: 1.5,
                           openTimeMs: 2.0, verifyTimeMs: 0.5,
                           proofSizeBytes: 96, setupSizeBytes: 65536,
                           requiresTrustedSetup: true),
            PCSBenchResult(scheme: .fri, logDegree: 10,
                           setupTimeMs: 5, commitTimeMs: 3.0,
                           openTimeMs: 1.0, verifyTimeMs: 2.0,
                           proofSizeBytes: 50000, setupSizeBytes: 0,
                           requiresTrustedSetup: false),
        ]

        let report = PCSComparisonEngine.formatRecommendation(
            results: results,
            constraints: PCSTradeoffAnalysis.Constraints()
        )
        expect(report.contains("RECOMMENDED"), "report has recommendation marker")
        expect(report.contains("post-quantum"), "report mentions post-quantum")
        expect(report.contains("memory"), "report mentions memory")
    }

    // MARK: - formatBytesPublic

    suite("Byte formatting")
    expectEqual(PCSTradeoffAnalysis.formatBytesPublic(500), "500 B", "bytes")
    expectEqual(PCSTradeoffAnalysis.formatBytesPublic(2048), "2.0 KiB", "KiB")
    expectEqual(PCSTradeoffAnalysis.formatBytesPublic(1048576), "1.0 MiB", "MiB")

    // MARK: - Version registered

    suite("PCSComparisonEngine version")
    expect(!PCSComparisonEngine.version.version.isEmpty, "version is set")
    expect(PCSComparisonEngine.version.updated == "2026-04-05", "version date is today")
}
