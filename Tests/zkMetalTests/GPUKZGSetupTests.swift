// GPUKZGSetupEngine tests: SRS generation, multi-party ceremony, validation,
// serialization, subgroup extraction, GPU MSM commitment
import zkMetal
import Foundation

public func runGPUKZGSetupTests() {
    // MARK: - BN254 SRS Generation

    suite("GPU KZG Setup - BN254 SRS Generation")

    do {
        let engine = try GPUKZGSetupEngine(curve: .bn254)

        // Test basic SRS generation
        let tau: [UInt64] = [42, 0, 0, 0]
        let srs = engine.generateSRS(degree: 8, tau: tau)
        expect(srs.degree == 8, "BN254 SRS degree == 8")
        expect(srs.g2Count == 2, "BN254 SRS g2Count == 2")
        expect(srs.curve == .bn254, "BN254 SRS curve")

        // G1[0] should be the generator
        if let g1Points = srs.bn254G1Points() {
            expect(g1Points.count == 8, "BN254 G1 point count")
            let gen = bn254G1Generator()
            expect(fpToInt(g1Points[0].x) == fpToInt(gen.x), "BN254 G1[0].x == generator")
            expect(fpToInt(g1Points[0].y) == fpToInt(gen.y), "BN254 G1[0].y == generator")
        } else {
            expect(false, "BN254 G1 extraction failed")
        }

        // G2[0] should be the generator
        if let g2Points = srs.bn254G2Points() {
            expect(g2Points.count == 2, "BN254 G2 point count")
            let g2Gen = bn254G2Generator()
            expect(fpToInt(g2Points[0].x.c0) == fpToInt(g2Gen.x.c0), "BN254 G2[0] is generator")
        } else {
            expect(false, "BN254 G2 extraction failed")
        }
    } catch {
        expect(false, "BN254 engine init failed: \(error)")
    }

    // MARK: - BLS12-381 SRS Generation

    suite("GPU KZG Setup - BLS12-381 SRS Generation")

    do {
        let engine = try GPUKZGSetupEngine(curve: .bls12381)

        let tau: [UInt64] = [7, 0, 0, 0]
        let srs = engine.generateSRS(degree: 4, tau: tau)
        expect(srs.degree == 4, "BLS12-381 SRS degree == 4")
        expect(srs.g2Count == 2, "BLS12-381 SRS g2Count == 2")
        expect(srs.curve == .bls12381, "BLS12-381 SRS curve")

        if let g1Points = srs.bls12381G1Points() {
            expect(g1Points.count == 4, "BLS12-381 G1 point count")
            let gen = bls12381G1Generator()
            expect(fp381ToInt(g1Points[0].x) == fp381ToInt(gen.x), "BLS12-381 G1[0] is generator")
        } else {
            expect(false, "BLS12-381 G1 extraction failed")
        }
    } catch {
        expect(false, "BLS12-381 engine init failed: \(error)")
    }

    // MARK: - Consistency with CPU generateSRS

    suite("GPU KZG Setup - Consistency with CPU Path")

    do {
        let engine = try GPUKZGSetupEngine(curve: .bn254)

        let tau: [UInt64] = [42, 0, 0, 0]
        let gpuSRS = engine.generateSRS(degree: 8, tau: tau)
        let cpuSRS = generateSRS(degree: 8, tau: tau, curve: .bn254)

        expect(gpuSRS.degree == cpuSRS.degree, "Degree matches CPU")
        expect(gpuSRS.g1Powers == cpuSRS.g1Powers, "G1 powers match CPU")
        expect(gpuSRS.g2Powers == cpuSRS.g2Powers, "G2 powers match CPU")
    } catch {
        expect(false, "Consistency test init failed: \(error)")
    }

    // MARK: - BN254 Ceremony

    suite("GPU KZG Setup - BN254 Ceremony")

    do {
        let engine = try GPUKZGSetupEngine(curve: .bn254)

        let initState = engine.initCeremony(degree: 4)
        expect(initState.contributionCount == 0, "Initial contribution count == 0")
        expect(initState.srs.degree == 4, "Initial degree == 4")

        // First contribution
        let entropy1: [UInt8] = Array(repeating: 0xAB, count: 64)
        let (state1, proof1) = engine.contribute(state: initState, entropy: entropy1)
        expect(state1.contributionCount == 1, "After 1st contribution, count == 1")
        expect(proof1.curve == .bn254, "Proof curve matches")

        // Verify first contribution
        let valid1 = engine.verifyContribution(before: initState, after: state1, proof: proof1)
        expect(valid1, "BN254 1st contribution verifies")

        // Second contribution
        let entropy2: [UInt8] = Array(repeating: 0xCD, count: 64)
        let (state2, proof2) = engine.contribute(state: state1, entropy: entropy2)
        expect(state2.contributionCount == 2, "After 2nd contribution, count == 2")

        let valid2 = engine.verifyContribution(before: state1, after: state2, proof: proof2)
        expect(valid2, "BN254 2nd contribution verifies")

        // Finalize
        let finalSRS = engine.finalize(state: state2)
        expect(finalSRS.degree == 4, "Finalized SRS degree == 4")

        // Cross-verify: wrong proof should fail
        let crossValid = engine.verifyContribution(before: initState, after: state2, proof: proof1)
        expect(!crossValid, "Cross-contribution verification fails as expected")

        // Transcript should have 2 entries
        expect(engine.transcript.count == 2, "Transcript has 2 entries")
        expect(engine.transcript[0].index == 0, "Transcript[0] index == 0")
        expect(engine.transcript[1].index == 1, "Transcript[1] index == 1")
    } catch {
        expect(false, "BN254 ceremony init failed: \(error)")
    }

    // MARK: - BLS12-381 Ceremony

    suite("GPU KZG Setup - BLS12-381 Ceremony")

    do {
        let engine = try GPUKZGSetupEngine(curve: .bls12381)

        let initState = engine.initCeremony(degree: 4)
        expect(initState.srs.curve == .bls12381, "BLS12-381 ceremony curve")

        let entropy: [UInt8] = Array(repeating: 0xEF, count: 48)
        let (state1, proof1) = engine.contribute(state: initState, entropy: entropy)
        let valid = engine.verifyContribution(before: initState, after: state1, proof: proof1)
        expect(valid, "BLS12-381 contribution verifies")

        let finalSRS = engine.finalize(state: state1)
        expect(finalSRS.degree == 4, "BLS12-381 finalized degree == 4")
    } catch {
        expect(false, "BLS12-381 ceremony init failed: \(error)")
    }

    // MARK: - SRS Validation

    suite("GPU KZG Setup - SRS Validation")

    do {
        let engine = try GPUKZGSetupEngine(curve: .bn254)

        // Generate a valid SRS
        let tau: [UInt64] = [42, 0, 0, 0]
        let srs = engine.generateSRS(degree: 8, tau: tau)

        // Validate it
        let result = engine.validateSRS(srs, spotCheckCount: 3)
        expect(result.isValid, "Valid SRS passes validation")
        expect(result.failures.isEmpty, "No validation failures")
        expect(result.checks.count > 0, "Some checks were performed")

        // Validate with pairing checks
        let result2 = engine.validateSRS(srs, spotCheckCount: 1)
        expect(result2.isValid, "SRS passes with spotCheckCount=1")
    } catch {
        expect(false, "Validation test init failed: \(error)")
    }

    // MARK: - SRS Validation (Tampered)

    suite("GPU KZG Setup - Tampered SRS Detection")

    do {
        let engine = try GPUKZGSetupEngine(curve: .bn254)

        // Create a valid SRS then tamper with it
        let tau: [UInt64] = [42, 0, 0, 0]
        let srs = engine.generateSRS(degree: 4, tau: tau)

        // Tamper: change one byte in G1 powers (breaks pairing consistency)
        var tampered = srs.g1Powers
        if tampered.count > 70 {
            tampered[70] ^= 0xFF  // flip bits in 2nd point data
        }
        let tamperedSRS = StructuredReferenceString(
            curve: .bn254, g1Powers: tampered, g2Powers: srs.g2Powers,
            degree: srs.degree, g2Count: srs.g2Count
        )

        let result = engine.validateSRS(tamperedSRS, spotCheckCount: 2)
        expect(!result.isValid, "Tampered SRS fails validation")
        expect(!result.failures.isEmpty, "Tampered SRS has failures")
    } catch {
        expect(false, "Tampered SRS test init failed: \(error)")
    }

    // MARK: - SRS Serialization Round-Trip

    suite("GPU KZG Setup - Serialization (.ptau)")

    do {
        let engine = try GPUKZGSetupEngine(curve: .bn254)

        let tau: [UInt64] = [42, 0, 0, 0]
        let srs = engine.generateSRS(degree: 4, tau: tau)

        // Serialize to ptau
        if let data = engine.serialize(srs, format: .ptau) {
            expect(data.count > 0, ".ptau serialization non-empty")

            // Deserialize
            if let reloaded = engine.deserialize(from: data, format: .ptau) {
                expect(reloaded.degree == srs.degree, ".ptau round-trip degree")
                expect(reloaded.g2Count == srs.g2Count, ".ptau round-trip g2Count")
                expect(reloaded.g1Powers == srs.g1Powers, ".ptau round-trip G1 data")
                expect(reloaded.g2Powers == srs.g2Powers, ".ptau round-trip G2 data")
            } else {
                expect(false, ".ptau deserialization failed")
            }
        } else {
            expect(false, ".ptau serialization failed")
        }
    } catch {
        expect(false, "Serialization test init failed: \(error)")
    }

    // MARK: - BLS12-381 Ethereum KZG Format

    suite("GPU KZG Setup - Serialization (Ethereum KZG)")

    do {
        let engine = try GPUKZGSetupEngine(curve: .bls12381)

        let tau: [UInt64] = [7, 0, 0, 0]
        let srs = engine.generateSRS(degree: 4, tau: tau)

        if let data = engine.serialize(srs, format: .ethereumKZG) {
            expect(data.count > 0, "Ethereum KZG serialization non-empty")

            if let reloaded = engine.deserialize(from: data, format: .ethereumKZG) {
                expect(reloaded.degree == srs.degree, "Ethereum KZG round-trip degree")
                expect(reloaded.curve == .bls12381, "Ethereum KZG round-trip curve")
            } else {
                expect(false, "Ethereum KZG deserialization failed")
            }
        } else {
            expect(false, "Ethereum KZG serialization failed")
        }
    } catch {
        expect(false, "Ethereum KZG test init failed: \(error)")
    }

    // MARK: - Subgroup SRS Extraction

    suite("GPU KZG Setup - Subgroup Extraction")

    do {
        let engine = try GPUKZGSetupEngine(curve: .bn254)

        let tau: [UInt64] = [42, 0, 0, 0]
        let srs = engine.generateSRS(degree: 16, tau: tau)

        // Extract a smaller sub-SRS
        if let sub4 = engine.extractSubSRS(srs, degree: 4) {
            expect(sub4.degree == 4, "Sub-SRS degree == 4")
            expect(sub4.g2Count == 2, "Sub-SRS g2Count preserved")
            expect(sub4.curve == .bn254, "Sub-SRS curve preserved")

            // G1 points should be a prefix of the original
            if let subG1 = sub4.bn254G1Points(), let fullG1 = srs.bn254G1Points() {
                for i in 0..<4 {
                    expect(fpToInt(subG1[i].x) == fpToInt(fullG1[i].x), "Sub-SRS G1[\(i)].x matches")
                    expect(fpToInt(subG1[i].y) == fpToInt(fullG1[i].y), "Sub-SRS G1[\(i)].y matches")
                }
            } else {
                expect(false, "Sub-SRS G1 extraction failed")
            }
        } else {
            expect(false, "extractSubSRS failed")
        }

        // Extracting with degree == original should return identical SRS
        if let same = engine.extractSubSRS(srs, degree: 16) {
            expect(same.g1Powers == srs.g1Powers, "Same-degree extraction is identity")
        } else {
            expect(false, "Same-degree extraction failed")
        }

        // Extracting too large should fail
        let tooLarge = engine.extractSubSRS(srs, degree: 32)
        expect(tooLarge == nil, "Too-large extraction returns nil")

        // Extracting 0 should fail
        let zero = engine.extractSubSRS(srs, degree: 0)
        expect(zero == nil, "Zero-degree extraction returns nil")
    } catch {
        expect(false, "Subgroup extraction test init failed: \(error)")
    }

    // MARK: - Extract for Circuit

    suite("GPU KZG Setup - Extract for Circuit")

    do {
        let engine = try GPUKZGSetupEngine(curve: .bn254)

        let tau: [UInt64] = [42, 0, 0, 0]
        let srs = engine.generateSRS(degree: 32, tau: tau)

        // Extract for a circuit of size 5 -> rounds up to 8, needs 9 points
        if let circuitSRS = engine.extractForCircuit(srs, circuitSize: 5) {
            expect(circuitSRS.degree == 9, "Circuit SRS for size 5 has degree 9 (next pow2 + 1)")
        } else {
            expect(false, "Extract for circuit failed")
        }

        // Circuit size exactly power of 2
        if let circuitSRS = engine.extractForCircuit(srs, circuitSize: 8) {
            expect(circuitSRS.degree == 9, "Circuit SRS for size 8 has degree 9")
        } else {
            expect(false, "Extract for power-of-2 circuit failed")
        }

        // Circuit size 1
        if let circuitSRS = engine.extractForCircuit(srs, circuitSize: 1) {
            expect(circuitSRS.degree == 2, "Circuit SRS for size 1 has degree 2")
        } else {
            expect(false, "Extract for size-1 circuit failed")
        }
    } catch {
        expect(false, "Extract for circuit test init failed: \(error)")
    }

    // MARK: - GPU Commit (BN254)

    suite("GPU KZG Setup - BN254 Commit")

    do {
        let engine = try GPUKZGSetupEngine(curve: .bn254)

        let tau: [UInt64] = [42, 0, 0, 0]
        let srs = engine.generateSRS(degree: 8, tau: tau)

        // Commit to polynomial p(x) = 1 + 2x + 3x^2
        let coeffs: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3)]
        if let commitment = engine.commit(srs: srs, coefficients: coeffs) {
            // Verify it's not the point at infinity
            let affPts = batchToAffine([commitment])
            let aff = affPts[0]
            let x = fpToInt(aff.x)
            expect(x != [0, 0, 0, 0], "Commitment is not zero")

            // Commit with single coeff (just the generator scaled by 1)
            let singleCoeffs: [Fr] = [Fr.one]
            if let singleCommit = engine.commit(srs: srs, coefficients: singleCoeffs) {
                let singleAff = batchToAffine([singleCommit])[0]
                let gen = bn254G1Generator()
                expect(fpToInt(singleAff.x) == fpToInt(gen.x), "Commit([1]) == G1")
            } else {
                expect(false, "Single coeff commit failed")
            }
        } else {
            expect(false, "BN254 commit failed")
        }
    } catch {
        expect(false, "BN254 commit test init failed: \(error)")
    }

    // MARK: - Verification with Pairing

    suite("GPU KZG Setup - Pairing Verification")

    do {
        let engine = try GPUKZGSetupEngine(curve: .bn254)

        let initState = engine.initCeremony(degree: 4)
        let entropy: [UInt8] = Array(repeating: 0xAB, count: 64)
        let (state1, proof1) = engine.contribute(state: initState, entropy: entropy)

        // Pairing-based verification (stronger)
        let valid = engine.verifyContributionWithPairing(before: initState, after: state1, proof: proof1)
        expect(valid, "BN254 pairing verification succeeds")
    } catch {
        expect(false, "Pairing verification test init failed: \(error)")
    }

    // MARK: - Transcript Verification

    suite("GPU KZG Setup - Transcript Verification")

    do {
        let engine = try GPUKZGSetupEngine(curve: .bn254)

        let initState = engine.initCeremony(degree: 4)
        let entropy1: [UInt8] = Array(repeating: 0xAB, count: 64)
        let (state1, _) = engine.contribute(state: initState, entropy: entropy1)
        let entropy2: [UInt8] = Array(repeating: 0xCD, count: 64)
        let (state2, _) = engine.contribute(state: state1, entropy: entropy2)

        // Verify full transcript
        let (allValid, failIdx) = engine.verifyTranscript(
            initialState: initState, states: [state1, state2]
        )
        expect(allValid, "Full transcript verification succeeds")
        expect(failIdx == nil, "No failure index")
    } catch {
        expect(false, "Transcript verification test init failed: \(error)")
    }

    // MARK: - Raw Export

    suite("GPU KZG Setup - Raw Export")

    do {
        let engine = try GPUKZGSetupEngine(curve: .bn254)

        let tau: [UInt64] = [42, 0, 0, 0]
        let srs = engine.generateSRS(degree: 4, tau: tau)

        let rawG1 = engine.exportRawG1(srs)
        expect(rawG1.count == 4 * 64, "Raw G1 export: 4 * 64 bytes")

        let rawG2 = engine.exportRawG2(srs)
        expect(rawG2.count == 2 * 128, "Raw G2 export: 2 * 128 bytes")
    } catch {
        expect(false, "Raw export test init failed: \(error)")
    }

    // MARK: - SRS Summary and Byte Size

    suite("GPU KZG Setup - Utilities")

    do {
        let engine = try GPUKZGSetupEngine(curve: .bn254)

        let tau: [UInt64] = [42, 0, 0, 0]
        let srs = engine.generateSRS(degree: 8, tau: tau)

        let size = engine.srsByteSize(srs)
        expect(size == 8 * 64 + 2 * 128, "Byte size = 8*64 + 2*128")

        let summary = engine.srsSummary(srs)
        expect(summary.contains("BN254"), "Summary mentions BN254")
        expect(summary.contains("degree=8"), "Summary mentions degree")
    } catch {
        expect(false, "Utilities test init failed: \(error)")
    }

    // MARK: - Merge SRS

    suite("GPU KZG Setup - Merge SRS")

    do {
        let engine = try GPUKZGSetupEngine(curve: .bn254)

        let tau: [UInt64] = [42, 0, 0, 0]
        let srs8 = engine.generateSRS(degree: 8, tau: tau)
        let srs16 = engine.generateSRS(degree: 16, tau: tau)

        if let merged = engine.mergeSRS(srs8, srs16) {
            expect(merged.degree == 16, "Merge takes larger degree")
        } else {
            expect(false, "Merge failed")
        }

        // Merging with mismatched curves should fail
        let engine381 = try GPUKZGSetupEngine(curve: .bls12381)
        let srs381 = engine381.generateSRS(degree: 4, tau: [7, 0, 0, 0])
        let crossMerge = engine.mergeSRS(srs8, srs381)
        expect(crossMerge == nil, "Cross-curve merge returns nil")
    } catch {
        expect(false, "Merge test init failed: \(error)")
    }

    // MARK: - GPU Availability

    suite("GPU KZG Setup - GPU Status")

    do {
        let engine = try GPUKZGSetupEngine(curve: .bn254)
        // Just check that gpuAvailable doesn't crash
        let _ = engine.gpuAvailable
        expect(true, "GPU availability check runs without crash")
    } catch {
        expect(false, "GPU status test init failed: \(error)")
    }

    // MARK: - BLS12-381 Validation

    suite("GPU KZG Setup - BLS12-381 Validation")

    do {
        let engine = try GPUKZGSetupEngine(curve: .bls12381)

        let tau: [UInt64] = [7, 0, 0, 0]
        let srs = engine.generateSRS(degree: 4, tau: tau)

        let result = engine.validateSRS(srs, spotCheckCount: 2)
        expect(result.isValid, "BLS12-381 SRS validates")
        expect(result.failures.isEmpty, "No BLS12-381 validation failures")
    } catch {
        expect(false, "BLS12-381 validation test init failed: \(error)")
    }

    // MARK: - Curve Mismatch Rejection

    suite("GPU KZG Setup - Curve Mismatch")

    do {
        let engineBN = try GPUKZGSetupEngine(curve: .bn254)
        let engineBLS = try GPUKZGSetupEngine(curve: .bls12381)

        let srsBLS = engineBLS.generateSRS(degree: 4, tau: [7, 0, 0, 0])

        // BN254 engine should reject BLS12-381 SRS validation
        let result = engineBN.validateSRS(srsBLS)
        expect(!result.isValid, "BN254 engine rejects BLS12-381 SRS")

        // BN254 engine should reject BLS12-381 subgroup extraction
        let sub = engineBN.extractSubSRS(srsBLS, degree: 2)
        expect(sub == nil, "BN254 engine rejects BLS12-381 extraction")
    } catch {
        expect(false, "Curve mismatch test init failed: \(error)")
    }
}
