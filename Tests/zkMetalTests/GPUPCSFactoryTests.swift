// GPUPCSFactoryTests — Tests for GPUPCSFactory
// Validates scheme selection, unified commit/open/verify, config filtering, and benchmark mode.

import zkMetal
import Foundation

public func runGPUPCSFactoryTests() {

    // MARK: - GPUPCSType enum

    suite("GPUPCSType properties")

    expectEqual(GPUPCSType.allCases.count, 5, "5 PCS types in enum")
    expect(GPUPCSType.kzg.requiresTrustedSetup, "KZG requires trusted setup")
    expect(GPUPCSType.zeromorph.requiresTrustedSetup, "Zeromorph requires trusted setup")
    expect(!GPUPCSType.ipa.requiresTrustedSetup, "IPA is transparent")
    expect(!GPUPCSType.basefold.requiresTrustedSetup, "Basefold is transparent")
    expect(!GPUPCSType.fri.requiresTrustedSetup, "FRI is transparent")

    expectEqual(GPUPCSType.kzg.polynomialType, .univariate, "KZG is univariate")
    expectEqual(GPUPCSType.ipa.polynomialType, .univariate, "IPA is univariate")
    expectEqual(GPUPCSType.fri.polynomialType, .univariate, "FRI is univariate")
    expectEqual(GPUPCSType.basefold.polynomialType, .multilinear, "Basefold is multilinear")
    expectEqual(GPUPCSType.zeromorph.polynomialType, .multilinear, "Zeromorph is multilinear")

    expect(!GPUPCSType.kzg.proofSizeClass.isEmpty, "KZG has proof size class")
    expect(!GPUPCSType.ipa.verifierCost.isEmpty, "IPA has verifier cost")

    // MARK: - GPUPCSConfig defaults

    suite("GPUPCSConfig defaults")

    let defaultConfig = GPUPCSConfig()
    expectEqual(defaultConfig.securityLevel, .bits128, "default security = 128")
    expect(defaultConfig.allowTrustedSetup, "trusted setup allowed by default")
    expect(defaultConfig.polynomialType == nil, "no forced poly type by default")
    expect(defaultConfig.maxProofSizeBytes == nil, "no proof size limit by default")
    expect(defaultConfig.preferFastProver, "prefer fast prover by default")

    // MARK: - Automatic scheme selection

    suite("GPUPCSFactory scheme selection")

    let factory = GPUPCSFactory()

    // Univariate with trusted setup allowed -> KZG
    let uniScheme = factory.selectScheme(polynomialType: .univariate, logDegree: 10)
    expectEqual(uniScheme, .kzg, "auto-select KZG for univariate + trusted setup")

    // Multilinear with trusted setup allowed -> Zeromorph
    let mlScheme = factory.selectScheme(polynomialType: .multilinear, logDegree: 10)
    expectEqual(mlScheme, .zeromorph, "auto-select Zeromorph for multilinear + trusted setup")

    // Univariate without trusted setup -> IPA
    let noSetupConfig = GPUPCSConfig(allowTrustedSetup: false)
    let noSetupFactory = GPUPCSFactory(config: noSetupConfig)
    let noSetupScheme = noSetupFactory.selectScheme(polynomialType: .univariate, logDegree: 10)
    expectEqual(noSetupScheme, .ipa, "auto-select IPA for univariate without trusted setup")

    // Multilinear without trusted setup -> Basefold
    let noSetupML = noSetupFactory.selectScheme(polynomialType: .multilinear, logDegree: 10)
    expectEqual(noSetupML, .basefold, "auto-select Basefold for multilinear without trusted setup")

    // MARK: - Available schemes filtering

    suite("GPUPCSFactory available schemes")

    let allSchemes = factory.availableSchemes()
    expectEqual(allSchemes.count, 5, "all 5 schemes available with default config")

    let noSetupSchemes = noSetupFactory.availableSchemes()
    expectEqual(noSetupSchemes.count, 3, "3 transparent schemes (IPA, Basefold, FRI)")
    expect(!noSetupSchemes.contains(.kzg), "KZG filtered out when no trusted setup")
    expect(!noSetupSchemes.contains(.zeromorph), "Zeromorph filtered out when no trusted setup")

    let uniConfig = GPUPCSConfig(polynomialType: .univariate)
    let uniFactory = GPUPCSFactory(config: uniConfig)
    let uniSchemes = uniFactory.availableSchemes()
    expectEqual(uniSchemes.count, 3, "3 univariate schemes (KZG, IPA, FRI)")
    expect(!uniSchemes.contains(.basefold), "Basefold filtered out for univariate")
    expect(!uniSchemes.contains(.zeromorph), "Zeromorph filtered out for univariate")

    // MARK: - Recommendation reasons

    suite("GPUPCSFactory recommendation reasons")

    let kzgReason = factory.recommendationReason(for: .kzg)
    expect(kzgReason.contains("trusted setup"), "KZG reason mentions trusted setup")
    expect(kzgReason.contains("univariate"), "KZG reason mentions univariate")

    let friReason = factory.recommendationReason(for: .fri)
    expect(friReason.contains("transparent"), "FRI reason mentions transparent")

    // MARK: - KZG handle commit + open + verify

    suite("GPUPCSFactory KZG handle")
    do {
        let handle = try factory.create(.kzg, srsSize: 256)
        expectEqual(handle.scheme, .kzg, "handle scheme is KZG")

        // Create a small polynomial
        let coeffs = (0..<64).map { i in frFromInt(UInt64(i + 1)) }

        let commitment = try handle.commit(polynomial: coeffs)
        // Commitment should be a non-identity point
        let id = pointIdentity()
        let isIdentity = commitment.x.to64() == id.x.to64() && commitment.y.to64() == id.y.to64() && commitment.z.to64() == id.z.to64()
        expect(!isIdentity, "KZG commitment is not identity")

        let point = frFromInt(42)
        let proof = try handle.open(polynomial: coeffs, at: point, commitment: commitment)
        expectEqual(proof.scheme, .kzg, "proof scheme is KZG")
        expect(proof.sizeBytes > 0, "proof has data")

        let ok = try handle.verify(commitment: commitment, point: point, proof: proof)
        expect(ok, "KZG verify succeeds")
    } catch {
        expect(false, "KZG handle error: \(error)")
    }

    // MARK: - IPA handle commit + open

    suite("GPUPCSFactory IPA handle")
    do {
        let handle = try factory.create(.ipa, srsSize: 256)
        expectEqual(handle.scheme, .ipa, "handle scheme is IPA")

        let coeffs = (0..<128).map { i in frFromInt(UInt64(i + 1)) }
        let commitment = try handle.commit(polynomial: coeffs)
        let id = pointIdentity()
        let isIdentity = commitment.x.to64() == id.x.to64() && commitment.y.to64() == id.y.to64() && commitment.z.to64() == id.z.to64()
        expect(!isIdentity, "IPA commitment is not identity")

        let point = frFromInt(7)
        let proof = try handle.open(polynomial: coeffs, at: point, commitment: commitment)
        expectEqual(proof.scheme, .ipa, "proof scheme is IPA")
        expect(proof.sizeBytes > 0, "IPA proof has data")
    } catch {
        expect(false, "IPA handle error: \(error)")
    }

    // MARK: - Basefold handle commit + open

    suite("GPUPCSFactory Basefold handle")
    do {
        let handle = try factory.create(.basefold, srsSize: 64)
        expectEqual(handle.scheme, .basefold, "handle scheme is Basefold")

        // Basefold works on multilinear evaluations (power-of-2 sized)
        let evals = (0..<64).map { i in frFromInt(UInt64(i + 1)) }
        let commitment = try handle.commit(polynomial: evals)
        expect(true, "Basefold commit succeeded")

        let point = frFromInt(13)
        let proof = try handle.open(polynomial: evals, at: point, commitment: commitment)
        expectEqual(proof.scheme, .basefold, "proof scheme is Basefold")
        expect(proof.sizeBytes > 0, "Basefold proof has data")

        let ok = try handle.verify(commitment: commitment, point: point, proof: proof)
        expect(ok, "Basefold verify succeeds")
    } catch {
        expect(false, "Basefold handle error: \(error)")
    }

    // MARK: - Zeromorph handle commit + open

    suite("GPUPCSFactory Zeromorph handle")
    do {
        let handle = try factory.create(.zeromorph, srsSize: 256)
        expectEqual(handle.scheme, .zeromorph, "handle scheme is Zeromorph")

        let evals = (0..<128).map { i in frFromInt(UInt64(i + 1)) }
        let commitment = try handle.commit(polynomial: evals)
        expect(true, "Zeromorph commit succeeded")

        let point = frFromInt(5)
        let proof = try handle.open(polynomial: evals, at: point, commitment: commitment)
        expectEqual(proof.scheme, .zeromorph, "proof scheme is Zeromorph")
        expect(proof.sizeBytes > 0, "Zeromorph proof has data")
    } catch {
        expect(false, "Zeromorph handle error: \(error)")
    }

    // MARK: - FRI handle commit + open

    suite("GPUPCSFactory FRI handle")
    do {
        let handle = try factory.create(.fri, srsSize: 256)
        expectEqual(handle.scheme, .fri, "handle scheme is FRI")

        let coeffs = (0..<64).map { i in frFromInt(UInt64(i + 1)) }
        let commitment = try handle.commit(polynomial: coeffs)
        expect(true, "FRI commit succeeded")

        let point = frFromInt(99)
        let proof = try handle.open(polynomial: coeffs, at: point, commitment: commitment)
        expectEqual(proof.scheme, .fri, "proof scheme is FRI")
        expect(proof.sizeBytes > 0, "FRI proof has data")

        let ok = try handle.verify(commitment: commitment, point: point, proof: proof)
        expect(ok, "FRI verify succeeds")
    } catch {
        expect(false, "FRI handle error: \(error)")
    }

    // MARK: - Auto-create handle

    suite("GPUPCSFactory auto-create")
    do {
        let autoHandle = try factory.createAuto(polynomialType: .univariate, logDegree: 8)
        expectEqual(autoHandle.scheme, .kzg, "auto-create selects KZG for univariate")

        let mlHandle = try factory.createAuto(polynomialType: .multilinear, logDegree: 7)
        expectEqual(mlHandle.scheme, .zeromorph, "auto-create selects Zeromorph for multilinear")
    } catch {
        expect(false, "auto-create error: \(error)")
    }

    // MARK: - GPUPCSProof container

    suite("GPUPCSProof container")

    let testData: [UInt8] = [1, 2, 3, 4, 5]
    let testEval = frFromInt(42)
    let proof = GPUPCSProof(scheme: .kzg, data: testData, evaluation: testEval)
    expectEqual(proof.scheme, .kzg, "proof scheme")
    expectEqual(proof.sizeBytes, 5, "proof size = 5 bytes")
    expect(frEqual(proof.evaluation, testEval), "proof evaluation matches")

    // MARK: - Benchmark comparison format

    suite("GPUPCSFactory benchmark format")

    let entries = [
        GPUPCSBenchEntry(scheme: .kzg, logDegree: 8, commitMs: 1.5, openMs: 2.0,
                         verifyMs: 0.5, proofBytes: 96, verified: true),
        GPUPCSBenchEntry(scheme: .ipa, logDegree: 8, commitMs: 3.0, openMs: 4.0,
                         verifyMs: 10.0, proofBytes: 1536, verified: true),
    ]
    let table = GPUPCSFactory.formatComparison(entries)
    expect(table.contains("KZG"), "table contains KZG")
    expect(table.contains("IPA"), "table contains IPA")
    expect(!table.isEmpty, "table is not empty")

    let emptyTable = GPUPCSFactory.formatComparison([])
    expectEqual(emptyTable, "(no results)", "empty table shows placeholder")

    // MARK: - Version

    suite("GPUPCSFactory version")

    let v = GPUPCSFactory.version
    expect(!v.version.isEmpty, "version string not empty")
    expect(!v.updated.isEmpty, "updated date not empty")
    expectEqual(v.version, "1.0.0", "version is 1.0.0")
}
