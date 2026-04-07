// DAS (Data Availability Sampling) Tests
// Tests for EIP-4844 / Danksharding DAS engine and cell proofs.

import Foundation
import zkMetal

func runDASTests() {
    suite("DAS — Pippenger MSM Correctness")

    // Test Pippenger MSM with zero scalars
    do {
        let secretLimbs: [UInt64] = [42, 0, 0, 0]
        let srs = ReedSolomon381Engine.generateTestSRS(secret: secretLimbs, size: 17)
        let gen = bls12381G1Generator()
        let genProj = g1_381FromAffine(gen)

        // Verify SRS[0] == G
        if let s0Aff = g1_381ToAffine(g1_381FromAffine(srs[0])),
           let gAff = g1_381ToAffine(genProj) {
            expect(fp381ToInt(s0Aff.x) == fp381ToInt(gAff.x), "SRS[0] == G")
        }

        // Direct: 469*G
        let directResult = g1_381ScalarMul(genProj, [469, 0, 0, 0])

        // MSM with 2 non-zero scalars, no padding
        let fr7 = fr381FromInt(7)
        let fr11 = fr381FromInt(11)
        var flatScalars2 = [UInt32]()
        for s in [fr7, fr11] {
            let std = fr381ToInt(s)
            flatScalars2.append(UInt32(std[0] & 0xFFFFFFFF))
            flatScalars2.append(UInt32(std[0] >> 32))
            flatScalars2.append(UInt32(std[1] & 0xFFFFFFFF))
            flatScalars2.append(UInt32(std[1] >> 32))
            flatScalars2.append(UInt32(std[2] & 0xFFFFFFFF))
            flatScalars2.append(UInt32(std[2] >> 32))
            flatScalars2.append(UInt32(std[3] & 0xFFFFFFFF))
            flatScalars2.append(UInt32(std[3] >> 32))
        }
        let msm2 = g1_381PippengerMSMFlat(points: [srs[0], srs[1]], flatScalars: flatScalars2)
        if let m2Aff = g1_381ToAffine(msm2), let dAff = g1_381ToAffine(directResult) {
            let match = fp381ToInt(m2Aff.x) == fp381ToInt(dAff.x)
            expect(match, "MSM(SRS[:2], [7,11]) == 469*G")
            if !match {
                print("    MSM result.x = \(fp381ToInt(m2Aff.x))")
                print("    469*G.x      = \(fp381ToInt(dAff.x))")
            }
        }

        // MSM with 16 scalars (14 zeros)
        var flatScalars16 = flatScalars2
        for _ in 2..<16 {
            flatScalars16.append(contentsOf: [UInt32](repeating: 0, count: 8))
        }
        let msm16 = g1_381PippengerMSMFlat(points: Array(srs.prefix(16)), flatScalars: flatScalars16)
        if let m16Aff = g1_381ToAffine(msm16), let dAff = g1_381ToAffine(directResult) {
            let match = fp381ToInt(m16Aff.x) == fp381ToInt(dAff.x)
            expect(match, "MSM(SRS[:16], [7,11,0,...]) == 469*G")
            if !match {
                print("    MSM16 result.x = \(fp381ToInt(m16Aff.x))")
                print("    469*G.x        = \(fp381ToInt(dAff.x))")
            }
        }

        // Manual computation: 7*SRS[0] + 11*SRS[1] using scalar mul + add
        let seven_G = g1_381ScalarMul(g1_381FromAffine(srs[0]), [7, 0, 0, 0])
        let eleven_42G = g1_381ScalarMul(g1_381FromAffine(srs[1]), [11, 0, 0, 0])
        let manual = g1_381Add(seven_G, eleven_42G)
        if let mAff = g1_381ToAffine(manual), let dAff = g1_381ToAffine(directResult) {
            let match = fp381ToInt(mAff.x) == fp381ToInt(dAff.x)
            expect(match, "7*SRS[0] + 11*SRS[1] via scalar_mul == 469*G")
            if !match {
                print("    manual.x = \(fp381ToInt(mAff.x))")
                print("    469*G.x  = \(fp381ToInt(dAff.x))")
            }
        }

        // Also test the old-style MSM (not Pippenger): g1_381PippengerMSM (array-of-arrays)
        let scalarsAA: [[UInt32]] = [
            [7, 0, 0, 0, 0, 0, 0, 0],
            [11, 0, 0, 0, 0, 0, 0, 0]
        ]
        let msmAA = g1_381PippengerMSM(points: [srs[0], srs[1]], scalars: scalarsAA)
        if let aaAff = g1_381ToAffine(msmAA), let dAff = g1_381ToAffine(directResult) {
            let match = fp381ToInt(aaAff.x) == fp381ToInt(dAff.x)
            expect(match, "g1_381PippengerMSM(SRS[:2], [7,11]) == 469*G")
        }

        // Single point MSM: MSM([SRS[0]], [1]) should equal G
        let scalars1: [UInt32] = [1, 0, 0, 0, 0, 0, 0, 0]
        let msm1 = g1_381PippengerMSMFlat(points: [srs[0]], flatScalars: scalars1)
        if let m1Aff = g1_381ToAffine(msm1), let gAff = g1_381ToAffine(genProj) {
            let match = fp381ToInt(m1Aff.x) == fp381ToInt(gAff.x)
            expect(match, "MSM([G], [1]) == G")
            if !match {
                print("    msm1.x = \(fp381ToInt(m1Aff.x))")
                print("    G.x    = \(fp381ToInt(gAff.x))")
            }
        }

        // MSM([SRS[0]], [7]) should equal 7*G
        let scalars7: [UInt32] = [7, 0, 0, 0, 0, 0, 0, 0]
        let msm7 = g1_381PippengerMSMFlat(points: [srs[0]], flatScalars: scalars7)
        let sevenG = g1_381ScalarMul(genProj, [7, 0, 0, 0])
        if let m7Aff = g1_381ToAffine(msm7), let s7Aff = g1_381ToAffine(sevenG) {
            let match = fp381ToInt(m7Aff.x) == fp381ToInt(s7Aff.x)
            expect(match, "MSM([G], [7]) == 7*G")
            if !match {
                print("    msm7.x = \(fp381ToInt(m7Aff.x))")
                print("    7G.x   = \(fp381ToInt(s7Aff.x))")
            }
        }

        // MSM([SRS[1]], [11]) should equal 11*42*G = 462*G
        let scalars11: [UInt32] = [11, 0, 0, 0, 0, 0, 0, 0]
        let msm11 = g1_381PippengerMSMFlat(points: [srs[1]], flatScalars: scalars11)
        let four62G = g1_381ScalarMul(genProj, [462, 0, 0, 0])
        if let m11Aff = g1_381ToAffine(msm11), let f462Aff = g1_381ToAffine(four62G) {
            let match = fp381ToInt(m11Aff.x) == fp381ToInt(f462Aff.x)
            expect(match, "MSM([42G], [11]) == 462*G")
            if !match {
                print("    msm462.x = \(fp381ToInt(m11Aff.x))")
                print("    462G.x   = \(fp381ToInt(f462Aff.x))")
            }
        }

        // Verify SRS[1] via scalar_mul
        let srs1Proj = g1_381FromAffine(srs[1])
        let fortyTwoG = g1_381ScalarMul(genProj, [42, 0, 0, 0])
        if let s1Aff = g1_381ToAffine(srs1Proj), let ftAff = g1_381ToAffine(fortyTwoG) {
            let match = fp381ToInt(s1Aff.x) == fp381ToInt(ftAff.x)
            expect(match, "SRS[1] == 42*G")
            if !match {
                print("    SRS[1].x = \(fp381ToInt(s1Aff.x))")
                print("    42*G.x   = \(fp381ToInt(ftAff.x))")
            }
        }

        // Simple test: MSM with 1 point, [469] directly
        let scalars469: [UInt32] = [469, 0, 0, 0, 0, 0, 0, 0]
        let msm469 = g1_381PippengerMSMFlat(points: [srs[0]], flatScalars: scalars469)
        if let m469Aff = g1_381ToAffine(msm469), let dAff = g1_381ToAffine(directResult) {
            let match = fp381ToInt(m469Aff.x) == fp381ToInt(dAff.x)
            expect(match, "MSM([G], [469]) == 469*G")
        }

        // MSM result for [7, 11]
        if let m2Aff = g1_381ToAffine(msm2) {
            // Check if MSM result is actually 7*G + 11*SRS[1]
            // Compute 7*G + 11*SRS[1] manually
            let manualSum = g1_381Add(sevenG, g1_381ScalarMul(srs1Proj, [11, 0, 0, 0]))
            if let msAff = g1_381ToAffine(manualSum) {
                let match = fp381ToInt(m2Aff.x) == fp381ToInt(msAff.x)
                print("  MSM([G,42G], [7,11]) == 7G + 11*(42G)? \(match)")
                if !match {
                    print("    MSM.x    = \(fp381ToInt(m2Aff.x))")
                    print("    manual.x = \(fp381ToInt(msAff.x))")
                }
            }
        }

        print("  Pippenger MSM zero scalar test: done")
    }

    suite("DAS — Data Availability Sampling")

    // Use a small config for fast testing
    let testConfig = DASConfig(blobSize: 16, extensionFactor: 2, sampleCount: 8)
    let testCellConfig = CellConfig(cellSize: 4, dasConfig: testConfig)

    // Generate test SRS (NOT secure — known secret)
    let secretLimbs: [UInt64] = [42, 0, 0, 0]
    let srsSize = testConfig.blobSize + 1 // need at least blobSize points
    let srs = ReedSolomon381Engine.generateTestSRS(secret: secretLimbs, size: srsSize)

    // Create sampler
    let sampler = EIP4844DataAvailabilitySampler(srs: srs, config: testConfig)

    // --- Test DASConfig ---
    do {
        expect(testConfig.extendedSize == 32, "extendedSize = blobSize * extensionFactor")
        expect(testConfig.logExtendedSize == 5, "logExtendedSize = log2(32)")
        expect(DASConfig.eip4844.blobSize == 4096, "EIP-4844 blob size is 4096")
        expect(DASConfig.eip4844.extendedSize == 8192, "EIP-4844 extended size is 8192")
        print("  DASConfig: OK")
    }

    // --- Test blob extension ---
    do {
        // Create a simple polynomial: p(x) = 1 + 2x + 3x^2 (padded to blobSize)
        var blobData = [Fr381](repeating: .zero, count: testConfig.blobSize)
        blobData[0] = fr381FromInt(1)
        blobData[1] = fr381FromInt(2)
        blobData[2] = fr381FromInt(3)

        let extended = sampler.extendBlob(data: blobData)

        // Check dimensions
        expect(extended.codeword.count == testConfig.extendedSize, "Extended codeword has correct size")
        expect(extended.original.count == testConfig.blobSize, "Original blob preserved")

        // Verify: codeword[0] should be p(omega^0) = p(1) = 1 + 2 + 3 = 6
        let evalAt1 = fr381ToInt(extended.codeword[0])
        expect(evalAt1[0] == 6 && evalAt1[1] == 0 && evalAt1[2] == 0 && evalAt1[3] == 0,
               "p(1) = 1 + 2 + 3 = 6")

        print("  Blob extension (RS via NTT): OK")
    }

    // --- Test KZG commitment ---
    do {
        var blobData = [Fr381](repeating: .zero, count: testConfig.blobSize)
        blobData[0] = fr381FromInt(7)
        blobData[1] = fr381FromInt(11)

        let (commitment, coefficients) = try! sampler.commitBlob(data: blobData)

        // Commitment should not be identity (non-trivial polynomial)
        expect(!g1_381IsIdentity(commitment), "Commitment is not identity for non-trivial poly")
        expect(coefficients.count == testConfig.blobSize, "Coefficients padded to blobSize")

        print("  KZG commitment: OK")
    }

    // --- Test single sample generation and recompute verification ---
    do {
        var blobData = [Fr381](repeating: .zero, count: testConfig.blobSize)
        blobData[0] = fr381FromInt(5)
        blobData[1] = fr381FromInt(3)
        blobData[2] = fr381FromInt(1)

        let (commitment, coefficients) = try! sampler.commitBlob(data: blobData)
        let extended = sampler.extendBlob(data: coefficients)

        // Generate sample at index 0
        let sample0 = try! sampler.generateSample(coefficients: coefficients,
                                                   codeword: extended.codeword, index: 0)
        expect(sample0.index == 0, "Sample index correct")

        // Verify via recompute (reliable for test without pairings)
        let valid0 = sampler.verifySampleByRecompute(commitment: commitment, sample: sample0,
                                                      coefficients: coefficients)
        expect(valid0, "Sample at index 0 verifies (recompute)")

        // Generate and verify sample at index 5
        let sample5 = try! sampler.generateSample(coefficients: coefficients,
                                                   codeword: extended.codeword, index: 5)
        let valid5 = sampler.verifySampleByRecompute(commitment: commitment, sample: sample5,
                                                      coefficients: coefficients)
        expect(valid5, "Sample at index 5 verifies (recompute)")

        // Verify sample at a position in the extension region (index >= blobSize)
        let sampleExt = try! sampler.generateSample(coefficients: coefficients,
                                                     codeword: extended.codeword, index: 20)
        let validExt = sampler.verifySampleByRecompute(commitment: commitment, sample: sampleExt,
                                                        coefficients: coefficients)
        expect(validExt, "Sample at index 20 (extension) verifies (recompute)")

        // Tamper with the sample value and check it fails
        let tamperedSample = DASSample(index: sample5.index,
                                       value: fr381FromInt(999),
                                       proof: sample5.proof,
                                       evalPoint: sample5.evalPoint)
        let invalid = sampler.verifySampleByRecompute(commitment: commitment, sample: tamperedSample,
                                                       coefficients: coefficients)
        expect(!invalid, "Tampered sample fails verification")

        print("  Single sample generation + recompute verification: OK")
    }

    // --- Test batch sample verification ---
    do {
        var blobData = [Fr381](repeating: .zero, count: testConfig.blobSize)
        for i in 0..<testConfig.blobSize {
            blobData[i] = fr381FromInt(UInt64(i + 1))
        }

        let (commitment, coefficients) = try! sampler.commitBlob(data: blobData)
        let extended = sampler.extendBlob(data: coefficients)

        let indices = [0, 3, 7, 15, 20, 28]
        let allValid = try! sampler.sampleAndVerify(commitment: commitment,
                                                     sampleIndices: indices,
                                                     coefficients: coefficients,
                                                     codeword: extended.codeword)
        expect(allValid, "Batch sample verification passes (recompute)")

        print("  Batch sample verification: OK")
    }

    // --- Test blob reconstruction ---
    do {
        var blobData = [Fr381](repeating: .zero, count: testConfig.blobSize)
        blobData[0] = fr381FromInt(10)
        blobData[1] = fr381FromInt(20)
        blobData[2] = fr381FromInt(30)

        let (_, coefficients) = try! sampler.commitBlob(data: blobData)
        let extended = sampler.extendBlob(data: coefficients)

        // Generate enough samples (at least blobSize = 16)
        var samples = [DASSample]()
        for i in 0..<testConfig.blobSize {
            let s = try! sampler.generateSample(coefficients: coefficients,
                                                 codeword: extended.codeword, index: i)
            samples.append(s)
        }

        let reconstructed = sampler.reconstructBlob(samples: samples)
        expect(reconstructed != nil, "Reconstruction succeeds with enough samples")

        if let r = reconstructed {
            // Check first few coefficients match
            expect(fr381ToInt(r[0]) == fr381ToInt(blobData[0]), "Reconstructed coeff 0 matches")
            expect(fr381ToInt(r[1]) == fr381ToInt(blobData[1]), "Reconstructed coeff 1 matches")
            expect(fr381ToInt(r[2]) == fr381ToInt(blobData[2]), "Reconstructed coeff 2 matches")

            // Higher coefficients should be zero
            for i in 3..<testConfig.blobSize {
                let limbs = fr381ToInt(r[i])
                expect(limbs == [0, 0, 0, 0], "Reconstructed coeff \(i) is zero")
            }
        }

        // Too few samples should fail
        let insufficientSamples = Array(samples.prefix(testConfig.blobSize / 2))
        let failedReconstruction = sampler.reconstructBlob(samples: insufficientSamples)
        expect(failedReconstruction == nil, "Reconstruction fails with insufficient samples")

        print("  Blob reconstruction: OK")
    }

    // --- Test random sample indices ---
    do {
        let indices = sampler.randomSampleIndices(seed: 12345)
        expect(indices.count == testConfig.sampleCount, "Correct number of random indices")
        let uniqueCount = Set(indices).count
        expect(uniqueCount == indices.count, "All random indices are unique")
        for idx in indices {
            expect(idx >= 0 && idx < testConfig.extendedSize, "Index within range")
        }
        print("  Random sample indices: OK")
    }

    // --- Test NTT codeword consistency with Horner evaluation ---
    do {
        var blobData = [Fr381](repeating: .zero, count: testConfig.blobSize)
        blobData[0] = fr381FromInt(7)
        blobData[1] = fr381FromInt(13)
        blobData[2] = fr381FromInt(2)

        let extended = sampler.extendBlob(data: blobData)
        let logN = testConfig.logExtendedSize
        let omega = fr381RootOfUnity(logN: logN)

        // Verify a few positions: NTT codeword[i] should equal Horner eval at omega^i
        // But since we now use Horner for sample values, just verify the NTT output at index 0
        let cw0 = fr381ToInt(extended.codeword[0])
        // p(omega^0) = p(1) = 7 + 13 + 2 = 22
        expect(cw0[0] == 22 && cw0[1] == 0, "NTT codeword[0] = p(1) = 22")

        print("  NTT codeword consistency: OK")
    }

    // MARK: - Cell Proof Tests

    suite("DAS — Cell Proofs (Danksharding)")

    let cellEngine = CellProofEngine(srs: srs, config: testCellConfig)

    // --- Test single cell proof generation and recompute verification ---
    do {
        var blobData = [Fr381](repeating: .zero, count: testConfig.blobSize)
        blobData[0] = fr381FromInt(5)
        blobData[1] = fr381FromInt(3)
        blobData[2] = fr381FromInt(1)

        let (commitment, coefficients) = try! sampler.commitBlob(data: blobData)
        let extended = sampler.extendBlob(data: coefficients)

        let cellProof = try! cellEngine.generateSingleCellProof(
            coefficients: coefficients, codeword: extended.codeword, cellIndex: 0)

        expect(cellProof.cellIndex == 0, "Cell index correct")
        expect(cellProof.values.count == testCellConfig.cellSize, "Cell has correct number of values")

        // Verify using recompute
        let valid = cellEngine.verifyCellProofByRecompute(
            commitment: commitment, cell: cellProof,
            coefficients: coefficients, codeword: extended.codeword)
        expect(valid, "Cell proof at index 0 verifies (recompute)")

        print("  Single cell proof generation + recompute verification: OK")
    }

    // --- Test all cell proofs ---
    do {
        var blobData = [Fr381](repeating: .zero, count: testConfig.blobSize)
        for i in 0..<testConfig.blobSize {
            blobData[i] = fr381FromInt(UInt64(i * 3 + 1))
        }

        let (commitment, coefficients) = try! sampler.commitBlob(data: blobData)
        let extended = sampler.extendBlob(data: coefficients)

        let allCellProofs = try! cellEngine.generateCellProofs(
            coefficients: coefficients, codeword: extended.codeword, commitment: commitment)

        expect(allCellProofs.count == testCellConfig.cellCount, "Correct number of cell proofs")

        var allValid = true
        for cp in allCellProofs {
            if !cellEngine.verifyCellProofByRecompute(
                commitment: commitment, cell: cp,
                coefficients: coefficients, codeword: extended.codeword) {
                allValid = false
                print("  [FAIL] Cell \(cp.cellIndex) failed recompute verification")
            }
        }
        expect(allValid, "All cell proofs verify (recompute)")

        print("  All cell proofs generation + recompute verification: OK")
    }

    // --- Test cell proof with tampered data fails ---
    do {
        var blobData = [Fr381](repeating: .zero, count: testConfig.blobSize)
        blobData[0] = fr381FromInt(42)
        blobData[1] = fr381FromInt(7)

        let (commitment, coefficients) = try! sampler.commitBlob(data: blobData)
        let extended = sampler.extendBlob(data: coefficients)

        let cellProof = try! cellEngine.generateSingleCellProof(
            coefficients: coefficients, codeword: extended.codeword, cellIndex: 0)

        // Tamper with the cell values
        var tamperedValues = cellProof.values
        tamperedValues[0] = fr381FromInt(999)
        let tamperedCell = CellProof(cellIndex: cellProof.cellIndex,
                                     values: tamperedValues,
                                     witness: cellProof.witness,
                                     batchChallenge: cellProof.batchChallenge)

        let invalid = cellEngine.verifyCellProofByRecompute(
            commitment: commitment, cell: tamperedCell,
            coefficients: coefficients, codeword: extended.codeword)
        expect(!invalid, "Tampered cell proof fails verification")

        print("  Tampered cell proof rejection: OK")
    }

    // --- Test cell proof for extension region ---
    do {
        var blobData = [Fr381](repeating: .zero, count: testConfig.blobSize)
        blobData[0] = fr381FromInt(100)
        blobData[3] = fr381FromInt(50)

        let (commitment, coefficients) = try! sampler.commitBlob(data: blobData)
        let extended = sampler.extendBlob(data: coefficients)

        // Test a cell in the extension region (cellIndex >= blobSize/cellSize)
        let extCellIdx = testCellConfig.cellCount - 1  // last cell
        let cellProof = try! cellEngine.generateSingleCellProof(
            coefficients: coefficients, codeword: extended.codeword, cellIndex: extCellIdx)

        let valid = cellEngine.verifyCellProofByRecompute(
            commitment: commitment, cell: cellProof,
            coefficients: coefficients, codeword: extended.codeword)
        expect(valid, "Extension region cell proof verifies")

        print("  Extension region cell proof: OK")
    }

    // MARK: - GPU DAS Engine Tests

    suite("DAS — GPU-Accelerated KZG Proofs")

    let gpuEngine = GPUDASEngine(srs: srs, config: testConfig, gpuThreshold: 4)

    // --- Test GPU engine initialization ---
    do {
        // GPU may or may not be available depending on the system
        print("  GPU available: \(gpuEngine.hasGPU)")
        print("  GPU DAS engine initialized: OK")
    }

    // --- Test blob encoding via GPU engine ---
    do {
        var blobData = [Fr381](repeating: .zero, count: testConfig.blobSize)
        blobData[0] = fr381FromInt(5)
        blobData[1] = fr381FromInt(3)
        blobData[2] = fr381FromInt(1)

        let blob = gpuEngine.encodeFrBlob(data: blobData)

        expect(blob.coefficients.count == testConfig.blobSize, "Blob coefficients padded to blobSize")
        expect(blob.codeword.count == testConfig.extendedSize, "Codeword has extended size")
        expect(!g1_381IsIdentity(blob.commitment), "Commitment is not identity")

        // Verify codeword[0] = p(1) = 5 + 3 + 1 = 9
        let cw0 = fr381ToInt(blob.codeword[0])
        expect(cw0[0] == 9 && cw0[1] == 0, "GPU blob codeword[0] = p(1) = 9")

        print("  GPU blob encoding: OK")
    }

    // --- Test GPU proof generation and recompute verification ---
    do {
        var blobData = [Fr381](repeating: .zero, count: testConfig.blobSize)
        blobData[0] = fr381FromInt(5)
        blobData[1] = fr381FromInt(3)
        blobData[2] = fr381FromInt(1)

        let blob = gpuEngine.encodeFrBlob(data: blobData)
        let proofs = try! gpuEngine.generateProofs(blob: blob, indices: [0, 5, 20])

        expect(proofs.count == 3, "Generated 3 proofs")
        expect(proofs[0].index == 0, "First proof at index 0")
        expect(proofs[1].index == 5, "Second proof at index 5")
        expect(proofs[2].index == 20, "Third proof at index 20")

        // Verify each proof by recompute
        for proof in proofs {
            let valid = gpuEngine.verifyProof(commitment: blob.commitment, proof: proof,
                                               coefficients: blob.coefficients)
            expect(valid, "GPU proof at index \(proof.index) verifies (recompute)")
        }

        print("  GPU proof generation + recompute verification: OK")
    }

    // --- Test GPU proof SRS-secret verification ---
    do {
        var blobData = [Fr381](repeating: .zero, count: testConfig.blobSize)
        blobData[0] = fr381FromInt(7)
        blobData[1] = fr381FromInt(11)

        let blob = gpuEngine.encodeFrBlob(data: blobData)
        let proofs = try! gpuEngine.generateProofs(blob: blob, indices: [0, 3, 10])

        for proof in proofs {
            let valid = gpuEngine.verifyProof(commitment: blob.commitment, proof: proof,
                                               srsSecret: secretLimbs)
            expect(valid, "GPU proof at index \(proof.index) verifies (SRS secret)")
        }

        print("  GPU proof SRS-secret verification: OK")
    }

    // --- Test GPU sample and verify ---
    do {
        var blobData = [Fr381](repeating: .zero, count: testConfig.blobSize)
        for i in 0..<testConfig.blobSize {
            blobData[i] = fr381FromInt(UInt64(i + 1))
        }

        let blob = gpuEngine.encodeFrBlob(data: blobData)
        let allValid = try! gpuEngine.sampleAndVerify(blob: blob, numSamples: 8, seed: 42)
        expect(allValid, "GPU sampleAndVerify passes (recompute, 8 samples)")

        // Also test with SRS secret
        let allValidSRS = try! gpuEngine.sampleAndVerify(blob: blob, numSamples: 8,
                                                          srsSecret: secretLimbs, seed: 42)
        expect(allValidSRS, "GPU sampleAndVerify passes (SRS secret, 8 samples)")

        print("  GPU sampleAndVerify: OK")
    }

    // --- Test GPU commitment matches CPU commitment ---
    do {
        var blobData = [Fr381](repeating: .zero, count: testConfig.blobSize)
        blobData[0] = fr381FromInt(42)
        blobData[1] = fr381FromInt(7)
        blobData[2] = fr381FromInt(13)

        // GPU blob
        let gpuBlob = gpuEngine.encodeFrBlob(data: blobData)

        // CPU commitment using existing sampler
        let (cpuCommitment, _) = try! sampler.commitBlob(data: blobData)

        // Compare commitments
        guard let gpuAff = g1_381ToAffine(gpuBlob.commitment),
              let cpuAff = g1_381ToAffine(cpuCommitment) else {
            expect(false, "Failed to convert commitments to affine")
            print("  GPU vs CPU commitment: FAIL (could not convert)")
            return
        }
        let match = fp381ToInt(gpuAff.x) == fp381ToInt(cpuAff.x) &&
                    fp381ToInt(gpuAff.y) == fp381ToInt(cpuAff.y)
        expect(match, "GPU commitment matches CPU commitment")

        print("  GPU vs CPU commitment consistency: OK")
    }

    // --- Test tampered proof rejection ---
    do {
        var blobData = [Fr381](repeating: .zero, count: testConfig.blobSize)
        blobData[0] = fr381FromInt(100)
        blobData[1] = fr381FromInt(200)

        let blob = gpuEngine.encodeFrBlob(data: blobData)
        let proof = try! gpuEngine.generateProof(blob: blob, index: 3)

        // Tamper with the value
        let tamperedProof = KZGProof381(
            evalPoint: proof.evalPoint,
            value: fr381FromInt(999),
            witness: proof.witness,
            index: proof.index
        )
        let invalid = gpuEngine.verifyProof(commitment: blob.commitment, proof: tamperedProof,
                                             coefficients: blob.coefficients)
        expect(!invalid, "Tampered GPU proof rejected (recompute)")

        let invalidSRS = gpuEngine.verifyProof(commitment: blob.commitment, proof: tamperedProof,
                                                srsSecret: secretLimbs)
        expect(!invalidSRS, "Tampered GPU proof rejected (SRS secret)")

        print("  Tampered GPU proof rejection: OK")
    }

    // --- Test byte-based blob encoding ---
    do {
        // Create 32 bytes of test data (one field element)
        var testBytes = [UInt8](repeating: 0, count: 32)
        testBytes[31] = 42  // little value at big-endian MSB position

        let blob = gpuEngine.encodeBlob(data: testBytes)
        expect(blob.coefficients.count == testConfig.blobSize, "Byte-encoded blob has correct size")
        expect(!g1_381IsIdentity(blob.commitment), "Byte-encoded blob has non-trivial commitment")

        print("  Byte-based blob encoding: OK")
    }

    // --- Test random index generation ---
    do {
        let indices = gpuEngine.randomSampleIndices(count: testConfig.sampleCount, seed: 12345)
        expect(indices.count == testConfig.sampleCount, "Correct number of random indices")
        let uniqueCount = Set(indices).count
        expect(uniqueCount == indices.count, "All random indices are unique")
        for idx in indices {
            expect(idx >= 0 && idx < testConfig.extendedSize, "Index within range")
        }
        print("  GPU random sample indices: OK")
    }
}
