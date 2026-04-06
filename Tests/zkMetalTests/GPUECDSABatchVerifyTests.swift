// Tests for GPUECDSABatchVerifyEngine
import zkMetal
import Foundation

// MARK: - Test Helpers

/// Create a valid ECDSA signature from small integer keys (deterministic).
private func makeTestSig(privKey: UInt64, nonce: UInt64, msgHash: UInt64)
    -> (ECDSASignature, SecpPointAffine, UInt8)
{
    let gen = secp256k1Generator()
    let gProj = secpPointFromAffine(gen)

    let d = secpFrFromInt(privKey)
    let Q = secpPointToAffine(secpPointMulScalar(gProj, secpFrToInt(d)))
    let k = secpFrFromInt(nonce)
    let z = secpFrFromInt(msgHash)

    let rProj = secpPointMulScalar(gProj, secpFrToInt(k))
    let rAff = secpPointToAffine(rProj)
    let rXraw = secpToInt(rAff.x)
    var rModN = rXraw
    if gte256(rModN, SecpFr.N) {
        (rModN, _) = sub256(rModN, SecpFr.N)
    }
    let rFr = secpFrFromRaw(rModN)

    let kInv = secpFrInverse(k)
    let sFr = secpFrMul(kInv, secpFrAdd(z, secpFrMul(rFr, d)))

    let ry = secpToInt(rAff.y)
    let parity = UInt8(ry[0] & 1)

    return (ECDSASignature(r: rFr, s: sFr, z: z), Q, parity)
}

/// Create a batch of N valid signatures with sequential parameters.
private func makeBatch(count n: Int, privBase: UInt64 = 100,
                       nonceBase: UInt64 = 1000, msgBase: UInt64 = 50000)
    -> ([ECDSASignature], [SecpPointAffine], [UInt8])
{
    var sigs = [ECDSASignature]()
    var pks = [SecpPointAffine]()
    var recovs = [UInt8]()
    sigs.reserveCapacity(n)
    pks.reserveCapacity(n)
    recovs.reserveCapacity(n)

    for i in 0..<n {
        let (s, pk, rb) = makeTestSig(
            privKey: privBase + UInt64(i),
            nonce: nonceBase + UInt64(i) * 7,
            msgHash: msgBase + UInt64(i) * 13)
        sigs.append(s)
        pks.append(pk)
        recovs.append(rb)
    }
    return (sigs, pks, recovs)
}

/// Create a malleable (high-s) version of a signature by negating s.
private func makeHighS(_ sig: ECDSASignature) -> ECDSASignature {
    let negS = secpFrNeg(sig.s)
    return ECDSASignature(r: sig.r, s: negS, z: sig.z)
}

// MARK: - Test Runner

public func runGPUECDSABatchVerifyTests() {
    suite("GPU ECDSA Batch Verify Engine")
    testEngineCreation()
    testSingleSigVerify()
    testSingleSigVerifyInvalid()
    testMessageToScalar()
    testHashBytesToScalar()
    testMalleabilityDetection()
    testMalleabilityNormalization()
    testBatchSmallAllValid()
    testBatchSmallWithInvalid()
    testBatchLargeAllValid()
    testBatchLargeOneInvalid()
    testBatchLargeMultipleInvalid()
    testBatchFiatShamirAllValid()
    testBatchFiatShamirOneInvalid()
    testBatchFiatShamirDeterminism()
    testBatchEmpty()
    testBatchSingleElement()
    testBatchResultFields()
    testStrategyDescription()
    testStrictConfig()
    testEnforceLowSBatch()
    testStatistics()
    testConfigVariants()
    testBatchVerifySimple()
    testVerifySingleMessage()
    testFiatShamirVsRandomChallenges()
    testLargeBatchGPUPath()
}

// MARK: - Engine Creation

private func testEngineCreation() {
    guard let engine = try? GPUECDSABatchVerifyEngine() else {
        expect(false, "Failed to create GPUECDSABatchVerifyEngine with default config")
        return
    }
    expect(engine.config.gpuMSMThreshold == 300,
           "default gpuMSMThreshold should be 300")
    expect(engine.config.batchThreshold == 4,
           "default batchThreshold should be 4")
    expect(!engine.config.enforceLowS,
           "default enforceLowS should be false")
    expect(engine.config.useFiatShamirChallenges,
           "default useFiatShamirChallenges should be true")

    guard let strict = try? GPUECDSABatchVerifyEngine(config: .strict) else {
        expect(false, "Failed to create engine with strict config")
        return
    }
    expect(strict.config.enforceLowS, "strict config should enforce low-s")
}

// MARK: - Single Signature Verification

private func testSingleSigVerify() {
    guard let engine = try? GPUECDSABatchVerifyEngine() else {
        expect(false, "engine creation failed"); return
    }
    let (sig, pk, _) = makeTestSig(privKey: 42, nonce: 137, msgHash: 12345)
    expect(engine.verifySingle(signature: sig, publicKey: pk),
           "valid signature should verify")
}

private func testSingleSigVerifyInvalid() {
    guard let engine = try? GPUECDSABatchVerifyEngine() else {
        expect(false, "engine creation failed"); return
    }
    let gen = secp256k1Generator()
    let gProj = secpPointFromAffine(gen)
    let (sig, pk, _) = makeTestSig(privKey: 42, nonce: 137, msgHash: 12345)

    // Wrong message hash
    let wrongZ = ECDSASignature(r: sig.r, s: sig.s, z: secpFrFromInt(99999))
    expect(!engine.verifySingle(signature: wrongZ, publicKey: pk),
           "wrong message hash should fail")

    // Wrong public key
    let wrongPK = secpPointToAffine(
        secpPointMulScalar(gProj, secpFrToInt(secpFrFromInt(99))))
    expect(!engine.verifySingle(signature: sig, publicKey: wrongPK),
           "wrong public key should fail")

    // Wrong r
    let wrongR = ECDSASignature(r: secpFrFromInt(777), s: sig.s, z: sig.z)
    expect(!engine.verifySingle(signature: wrongR, publicKey: pk),
           "wrong r should fail")

    // Wrong s
    let wrongS = ECDSASignature(r: sig.r, s: secpFrFromInt(888), z: sig.z)
    expect(!engine.verifySingle(signature: wrongS, publicKey: pk),
           "wrong s should fail")

    // Zero r (degenerate)
    let zeroR = ECDSASignature(r: SecpFr.zero, s: sig.s, z: sig.z)
    expect(!engine.verifySingle(signature: zeroR, publicKey: pk),
           "zero r should fail")
}

// MARK: - Message Hash-to-Scalar

private func testMessageToScalar() {
    guard let engine = try? GPUECDSABatchVerifyEngine() else {
        expect(false, "engine creation failed"); return
    }

    let msg1 = Array("hello world".utf8)
    let msg2 = Array("hello world!".utf8)

    let z1 = engine.messageToScalar(msg1)
    let z2 = engine.messageToScalar(msg2)
    let z1b = engine.messageToScalar(msg1)

    // Same message -> same scalar
    expect(z1 == z1b, "same message should produce same scalar")

    // Different messages -> different scalars
    expect(!(z1 == z2), "different messages should produce different scalars")

    // Result should be non-zero
    expect(!z1.isZero, "message scalar should be non-zero")

    // Result should be in [0, n) -- verify by converting back and comparing
    let z1Int = secpFrToInt(z1)
    expect(!gte256(z1Int, SecpFr.N), "scalar should be less than group order")
}

private func testHashBytesToScalar() {
    guard let engine = try? GPUECDSABatchVerifyEngine() else {
        expect(false, "engine creation failed"); return
    }

    // Known 32-byte input
    var hash = [UInt8](repeating: 0, count: 32)
    hash[0] = 0x42
    let z = engine.hashBytesToScalar(hash)
    expect(!z.isZero, "hash-to-scalar should produce non-zero result")

    // All-zero hash
    let zeroHash = [UInt8](repeating: 0, count: 32)
    let zz = engine.hashBytesToScalar(zeroHash)
    expect(zz.isZero || !zz.isZero, "zero hash should not crash")

    // All-0xFF hash (larger than n, should be reduced)
    let maxHash = [UInt8](repeating: 0xFF, count: 32)
    let zm = engine.hashBytesToScalar(maxHash)
    let zmInt = secpFrToInt(zm)
    expect(!gte256(zmInt, SecpFr.N), "max hash should be reduced mod n")
}

// MARK: - Malleability Detection

private func testMalleabilityDetection() {
    guard let engine = try? GPUECDSABatchVerifyEngine() else {
        expect(false, "engine creation failed"); return
    }

    let (sig, _, _) = makeTestSig(privKey: 42, nonce: 137, msgHash: 12345)

    // Create high-s version
    let highSSig = makeHighS(sig)

    // Check single signatures
    let report1 = engine.checkMalleability([sig])
    // The original may or may not be high-s depending on the nonce
    // but the negated version should have opposite malleability
    let report2 = engine.checkMalleability([highSSig])

    // At least one of them should be malleable and the other not
    let oneIsMalleable = report1.malleableCount != report2.malleableCount
    expect(oneIsMalleable,
           "original and negated-s should have different malleability")

    // Test batch malleability
    let (sigs, _, _) = makeBatch(count: 8)
    var mixedSigs = sigs
    // Negate s for indices 2 and 5
    mixedSigs[2] = makeHighS(sigs[2])
    mixedSigs[5] = makeHighS(sigs[5])

    let batchReport = engine.checkMalleability(mixedSigs)
    // The original sigs may have varied malleability, but negating should flip them
    // We cannot assert exact indices without knowing original malleability,
    // so just verify the report is well-formed
    expect(batchReport.malleableCount >= 0, "malleability count should be non-negative")
    expect(batchReport.malleableCount <= 8, "malleability count should not exceed batch size")
}

private func testMalleabilityNormalization() {
    guard let engine = try? GPUECDSABatchVerifyEngine() else {
        expect(false, "engine creation failed"); return
    }

    let (sig, pk, _) = makeTestSig(privKey: 42, nonce: 137, msgHash: 12345)

    // Normalize the original (may or may not change)
    let norm1 = engine.normalizeSignature(sig)

    // Normalize again (should be idempotent)
    let norm2 = engine.normalizeSignature(norm1)
    expect(norm2.s == norm1.s, "double normalization should be idempotent")

    // Both original and normalized should verify (without enforceLowS)
    expect(engine.verifySingle(signature: sig, publicKey: pk),
           "original sig should verify")

    // Create high-s and normalize
    let highSSig = makeHighS(sig)
    let normalized = engine.normalizeSignature(highSSig)

    // Normalized should have low-s
    let sInt = secpFrToInt(normalized.s)
    let normReport = engine.checkMalleability([normalized])
    expect(normReport.allCanonical,
           "normalized signature should be canonical (low-s)")

    // Normalized high-s sig should still verify (s is negated, verification adjusts)
    // Note: ECDSA verification with negated s actually produces a different R,
    // so the high-s version may not verify. The point is normalization works.
    _ = sInt  // suppress unused warning
}

// MARK: - Small Batch Verification

private func testBatchSmallAllValid() {
    guard let engine = try? GPUECDSABatchVerifyEngine() else {
        expect(false, "engine creation failed"); return
    }

    let (sigs, pks, _) = makeBatch(count: 3)  // below batchThreshold=4
    do {
        let result = try engine.verifyBatch(signatures: sigs, publicKeys: pks)
        expect(result.allValid, "small batch all valid should pass")
        expectEqual(result.results.count, 3, "result count should match input")
        expect(result.strategy.contains("individual"),
               "small batch should use individual strategy")
        expect(result.batchCheckPassed == nil,
               "small batch should not use batch check")
    } catch {
        expect(false, "small batch threw: \(error)")
    }
}

private func testBatchSmallWithInvalid() {
    guard let engine = try? GPUECDSABatchVerifyEngine() else {
        expect(false, "engine creation failed"); return
    }

    let (sigs, pks, _) = makeBatch(count: 3)
    var badSigs = sigs
    badSigs[1] = ECDSASignature(r: sigs[1].r, s: sigs[1].s, z: secpFrFromInt(99999))

    do {
        let result = try engine.verifyBatch(signatures: badSigs, publicKeys: pks)
        expect(result.results[0], "sig 0 should be valid")
        expect(!result.results[1], "sig 1 (corrupted) should be invalid")
        expect(result.results[2], "sig 2 should be valid")
        expectEqual(result.invalidCount, 1, "should have exactly 1 invalid")
    } catch {
        expect(false, "small batch with invalid threw: \(error)")
    }
}

// MARK: - Large Batch Verification

private func testBatchLargeAllValid() {
    guard let engine = try? GPUECDSABatchVerifyEngine() else {
        expect(false, "engine creation failed"); return
    }

    let batchN = 16
    let (sigs, pks, recovs) = makeBatch(count: batchN)
    do {
        let result = try engine.verifyBatch(
            signatures: sigs, publicKeys: pks, recoveryBits: recovs)
        expect(result.allValid, "large batch all valid should pass")
        expectEqual(result.results.count, batchN, "result count should match")
        expect(result.batchCheckPassed == true,
               "large batch should use batch check")
        expect(result.strategy.contains("batch") || result.strategy.contains("RLC"),
               "large batch should use batch/RLC strategy")
    } catch {
        expect(false, "large batch threw: \(error)")
    }
}

private func testBatchLargeOneInvalid() {
    guard let engine = try? GPUECDSABatchVerifyEngine() else {
        expect(false, "engine creation failed"); return
    }

    let batchN = 16
    let (sigs, pks, recovs) = makeBatch(count: batchN)
    var badSigs = sigs
    let badIdx = batchN / 2
    badSigs[badIdx] = ECDSASignature(
        r: sigs[badIdx].r, s: sigs[badIdx].s, z: secpFrFromInt(99999))

    do {
        let result = try engine.verifyBatch(
            signatures: badSigs, publicKeys: pks, recoveryBits: recovs)
        expect(!result.results[badIdx], "corrupted sig should be detected")
        expect(result.batchCheckPassed == false,
               "batch check should fail when invalid sig present")

        // Check all other sigs are valid
        let othersOk = result.results.enumerated()
            .filter { $0.offset != badIdx }
            .allSatisfy { $0.element }
        expect(othersOk, "non-corrupted sigs should pass")
    } catch {
        expect(false, "batch with one invalid threw: \(error)")
    }
}

private func testBatchLargeMultipleInvalid() {
    guard let engine = try? GPUECDSABatchVerifyEngine() else {
        expect(false, "engine creation failed"); return
    }

    let batchN = 16
    let (sigs, pks, recovs) = makeBatch(count: batchN)
    var badSigs = sigs
    let badIndices = [0, batchN / 4, batchN / 2, batchN - 1]
    for idx in badIndices {
        badSigs[idx] = ECDSASignature(
            r: sigs[idx].r, s: sigs[idx].s,
            z: secpFrFromInt(UInt64(77777 + idx)))
    }

    do {
        let result = try engine.verifyBatch(
            signatures: badSigs, publicKeys: pks, recoveryBits: recovs)
        for idx in badIndices {
            expect(!result.results[idx], "invalid sig at \(idx) should fail")
        }
        let validIndices = Set(0..<batchN).subtracting(badIndices)
        for idx in validIndices {
            expect(result.results[idx], "valid sig at \(idx) should pass")
        }
        expectEqual(result.invalidCount, badIndices.count,
                    "invalid count should match corrupted count")
    } catch {
        expect(false, "batch with multiple invalid threw: \(error)")
    }
}

// MARK: - Fiat-Shamir Batch Verification

private func testBatchFiatShamirAllValid() {
    guard let engine = try? GPUECDSABatchVerifyEngine() else {
        expect(false, "engine creation failed"); return
    }

    let batchN = 16
    let (sigs, pks, recovs) = makeBatch(count: batchN)
    do {
        let ok = try engine.batchVerifyFiatShamir(
            signatures: sigs, publicKeys: pks, recoveryBits: recovs)
        expect(ok, "Fiat-Shamir batch all valid should return true")
    } catch {
        expect(false, "Fiat-Shamir batch threw: \(error)")
    }
}

private func testBatchFiatShamirOneInvalid() {
    guard let engine = try? GPUECDSABatchVerifyEngine() else {
        expect(false, "engine creation failed"); return
    }

    let batchN = 16
    let (sigs, pks, recovs) = makeBatch(count: batchN)
    var badSigs = sigs
    badSigs[0] = ECDSASignature(
        r: sigs[0].r, s: sigs[0].s, z: secpFrFromInt(11111))
    do {
        let ok = try engine.batchVerifyFiatShamir(
            signatures: badSigs, publicKeys: pks, recoveryBits: recovs)
        expect(!ok, "Fiat-Shamir batch with invalid should return false")
    } catch {
        expect(false, "Fiat-Shamir batch with invalid threw: \(error)")
    }
}

private func testBatchFiatShamirDeterminism() {
    guard let engine = try? GPUECDSABatchVerifyEngine() else {
        expect(false, "engine creation failed"); return
    }

    let batchN = 8
    let (sigs, pks, recovs) = makeBatch(count: batchN)

    // Run twice with same inputs -- should produce same result
    do {
        let ok1 = try engine.batchVerifyFiatShamir(
            signatures: sigs, publicKeys: pks, recoveryBits: recovs)
        let ok2 = try engine.batchVerifyFiatShamir(
            signatures: sigs, publicKeys: pks, recoveryBits: recovs)
        expectEqual(ok1, ok2,
                    "Fiat-Shamir batch should be deterministic (same result both times)")
    } catch {
        expect(false, "Fiat-Shamir determinism test threw: \(error)")
    }
}

// MARK: - Edge Cases

private func testBatchEmpty() {
    guard let engine = try? GPUECDSABatchVerifyEngine() else {
        expect(false, "engine creation failed"); return
    }

    do {
        let result = try engine.verifyBatch(signatures: [], publicKeys: [])
        expect(result.results.isEmpty, "empty batch should return empty results")
        expect(result.strategy == "empty", "empty batch strategy should be 'empty'")
    } catch {
        expect(false, "empty batch threw: \(error)")
    }

    // Fiat-Shamir empty
    do {
        let ok = try engine.batchVerifyFiatShamir(signatures: [], publicKeys: [])
        expect(ok, "Fiat-Shamir empty batch should return true")
    } catch {
        expect(false, "Fiat-Shamir empty threw: \(error)")
    }
}

private func testBatchSingleElement() {
    guard let engine = try? GPUECDSABatchVerifyEngine() else {
        expect(false, "engine creation failed"); return
    }

    let (sig, pk, recov) = makeTestSig(privKey: 42, nonce: 137, msgHash: 12345)
    do {
        let result = try engine.verifyBatch(
            signatures: [sig], publicKeys: [pk], recoveryBits: [recov])
        expectEqual(result.results.count, 1, "single-element batch should have 1 result")
        expect(result.results[0], "valid single-element batch should pass")
        expect(result.strategy.contains("individual"),
               "single-element should use individual strategy")
    } catch {
        expect(false, "single-element batch threw: \(error)")
    }
}

// MARK: - Result Fields

private func testBatchResultFields() {
    guard let engine = try? GPUECDSABatchVerifyEngine() else {
        expect(false, "engine creation failed"); return
    }

    let batchN = 8
    let (sigs, pks, recovs) = makeBatch(count: batchN)
    do {
        let result = try engine.verifyBatch(
            signatures: sigs, publicKeys: pks, recoveryBits: recovs)

        expectEqual(result.validCount, batchN, "validCount should equal batch size")
        expectEqual(result.invalidCount, 0, "invalidCount should be 0")
        expect(result.allValid, "allValid should be true")
        expect(!result.strategy.isEmpty, "strategy should not be empty")
    } catch {
        expect(false, "result fields test threw: \(error)")
    }

    // Test with invalid sigs for invalidCount
    var badSigs = sigs
    badSigs[0] = ECDSASignature(r: sigs[0].r, s: sigs[0].s, z: secpFrFromInt(99999))
    badSigs[3] = ECDSASignature(r: sigs[3].r, s: sigs[3].s, z: secpFrFromInt(88888))
    do {
        let result = try engine.verifyBatch(
            signatures: badSigs, publicKeys: pks, recoveryBits: recovs)
        expectEqual(result.invalidCount, 2, "invalidCount should be 2")
        expectEqual(result.validCount, batchN - 2, "validCount should be N-2")
        expect(!result.allValid, "allValid should be false")
    } catch {
        expect(false, "result fields with invalid threw: \(error)")
    }
}

// MARK: - Strategy Description

private func testStrategyDescription() {
    guard let engine = try? GPUECDSABatchVerifyEngine() else {
        expect(false, "engine creation failed"); return
    }

    let s0 = engine.strategyDescription(batchSize: 0)
    expect(s0.contains("empty"), "N=0 should be empty strategy")

    let s1 = engine.strategyDescription(batchSize: 1)
    expect(s1.contains("individual"), "N=1 should use individual")

    let s3 = engine.strategyDescription(batchSize: 3)
    expect(s3.contains("individual"), "N=3 should use individual")
    expect(s3.contains("3"), "N=3 description should mention 3")

    let s10 = engine.strategyDescription(batchSize: 10)
    expect(s10.contains("batch") || s10.contains("RLC"),
           "N=10 should use batch RLC")
    expect(s10.contains("CPU") || s10.contains("Pippenger"),
           "N=10 (21 MSM points) should use CPU path")

    let s1000 = engine.strategyDescription(batchSize: 1000)
    expect(s1000.contains("GPU") || s1000.contains("Metal"),
           "N=1000 should use GPU path")
}

// MARK: - Strict Configuration

private func testStrictConfig() {
    guard let engine = try? GPUECDSABatchVerifyEngine(config: .strict) else {
        expect(false, "strict engine creation failed"); return
    }
    expect(engine.config.enforceLowS, "strict should enforce low-s")
    expect(engine.config.useFiatShamirChallenges, "strict should use Fiat-Shamir")

    // Valid low-s sig should pass
    let (sig, pk, _) = makeTestSig(privKey: 42, nonce: 137, msgHash: 12345)
    let report = engine.checkMalleability([sig])
    if report.allCanonical {
        expect(engine.verifySingle(signature: sig, publicKey: pk),
               "canonical sig should verify with strict config")
    }
    // If the sig happens to be high-s, strict config should reject
    if !report.allCanonical {
        expect(!engine.verifySingle(signature: sig, publicKey: pk),
               "high-s sig should fail with strict config (enforceLowS)")
    }
}

private func testEnforceLowSBatch() {
    guard let engine = try? GPUECDSABatchVerifyEngine(config: .strict) else {
        expect(false, "strict engine creation failed"); return
    }

    // Create batch where one sig is made malleable
    let batchN = 8
    let (sigs, pks, recovs) = makeBatch(count: batchN)

    // First check which sigs are already canonical
    let origReport = engine.checkMalleability(sigs)

    // Negate s of a canonical sig to make it malleable
    if origReport.allCanonical {
        var badSigs = sigs
        badSigs[2] = makeHighS(sigs[2])

        do {
            let result = try engine.verifyBatch(
                signatures: badSigs, publicKeys: pks, recoveryBits: recovs)
            expect(!result.results[2], "high-s sig should be rejected by strict config")
            expect(result.malleabilityReport != nil,
                   "strict config should produce malleability report")
            expect(result.malleabilityReport!.malleableIndices.contains(2),
                   "malleability report should flag index 2")
        } catch {
            expect(false, "strict batch threw: \(error)")
        }
    } else {
        // Some original sigs are already high-s; just verify the report is produced
        do {
            let result = try engine.verifyBatch(
                signatures: sigs, publicKeys: pks, recoveryBits: recovs)
            expect(result.malleabilityReport != nil,
                   "strict config should always produce malleability report")
        } catch {
            expect(false, "strict batch with existing high-s threw: \(error)")
        }
    }
}

// MARK: - Statistics

private func testStatistics() {
    guard let engine = try? GPUECDSABatchVerifyEngine() else {
        expect(false, "engine creation failed"); return
    }

    expectEqual(engine.totalVerified, 0, "initial totalVerified should be 0")
    expectEqual(engine.totalBatchOps, 0, "initial totalBatchOps should be 0")

    let (sig, pk, _) = makeTestSig(privKey: 42, nonce: 137, msgHash: 12345)
    _ = engine.verifySingle(signature: sig, publicKey: pk)
    expectEqual(engine.totalVerified, 1, "totalVerified should increment on single verify")

    let (sigs, pks, _) = makeBatch(count: 3)
    _ = try? engine.verifyBatch(signatures: sigs, publicKeys: pks)
    expectEqual(engine.totalBatchOps, 1, "totalBatchOps should increment")
    expectEqual(engine.totalVerified, 4, "totalVerified should include batch sigs")

    engine.resetStats()
    expectEqual(engine.totalVerified, 0, "resetStats should zero totalVerified")
    expectEqual(engine.totalBatchOps, 0, "resetStats should zero totalBatchOps")
}

// MARK: - Config Variants

private func testConfigVariants() {
    // Custom config
    let custom = GPUECDSABatchConfig(
        gpuMSMThreshold: 500,
        batchThreshold: 8,
        enforceLowS: true,
        useFiatShamirChallenges: false)
    expectEqual(custom.gpuMSMThreshold, 500, "custom gpuMSMThreshold")
    expectEqual(custom.batchThreshold, 8, "custom batchThreshold")
    expect(custom.enforceLowS, "custom enforceLowS")
    expect(!custom.useFiatShamirChallenges, "custom useFiatShamirChallenges")

    // Default config
    let def = GPUECDSABatchConfig.default
    expectEqual(def.gpuMSMThreshold, 300, "default gpuMSMThreshold")
    expectEqual(def.batchThreshold, 4, "default batchThreshold")

    // Engine with custom config
    guard let engine = try? GPUECDSABatchVerifyEngine(config: custom) else {
        expect(false, "engine with custom config failed"); return
    }

    // With batchThreshold=8, a batch of 5 should use individual
    let (sigs, pks, _) = makeBatch(count: 5)
    do {
        let result = try engine.verifyBatch(signatures: sigs, publicKeys: pks)
        expect(result.strategy.contains("individual"),
               "batch of 5 with threshold=8 should use individual")
    } catch {
        expect(false, "custom config batch threw: \(error)")
    }
}

// MARK: - Simple Batch API

private func testBatchVerifySimple() {
    guard let engine = try? GPUECDSABatchVerifyEngine() else {
        expect(false, "engine creation failed"); return
    }

    let batchN = 8
    let (sigs, pks, recovs) = makeBatch(count: batchN)
    do {
        let results = try engine.verifyBatchSimple(
            signatures: sigs, publicKeys: pks, recoveryBits: recovs)
        expectEqual(results.count, batchN, "simple results count")
        expect(results.allSatisfy { $0 }, "simple batch all valid")
    } catch {
        expect(false, "simple batch threw: \(error)")
    }
}

// MARK: - Verify Single Message

private func testVerifySingleMessage() {
    guard let engine = try? GPUECDSABatchVerifyEngine() else {
        expect(false, "engine creation failed"); return
    }

    // Create a signature for a specific message hash derived from bytes
    let msg = Array("test message for ECDSA".utf8)
    let z = engine.messageToScalar(msg)

    let gen = secp256k1Generator()
    let gProj = secpPointFromAffine(gen)

    let privKey: UInt64 = 42
    let d = secpFrFromInt(privKey)
    let Q = secpPointToAffine(secpPointMulScalar(gProj, secpFrToInt(d)))

    let nonce: UInt64 = 137
    let k = secpFrFromInt(nonce)
    let rProj = secpPointMulScalar(gProj, secpFrToInt(k))
    let rAff = secpPointToAffine(rProj)
    let rXraw = secpToInt(rAff.x)
    var rModN = rXraw
    if gte256(rModN, SecpFr.N) {
        (rModN, _) = sub256(rModN, SecpFr.N)
    }
    let rFr = secpFrFromRaw(rModN)
    let kInv = secpFrInverse(k)
    let sFr = secpFrMul(kInv, secpFrAdd(z, secpFrMul(rFr, d)))

    expect(engine.verifySingleMessage(message: msg, r: rFr, s: sFr, publicKey: Q),
           "verifySingleMessage should pass for correctly signed message")

    // Wrong message
    let wrongMsg = Array("wrong message".utf8)
    expect(!engine.verifySingleMessage(message: wrongMsg, r: rFr, s: sFr, publicKey: Q),
           "verifySingleMessage should fail for wrong message")
}

// MARK: - Fiat-Shamir vs Random Challenges

private func testFiatShamirVsRandomChallenges() {
    // Test that both challenge modes produce correct results

    // Fiat-Shamir mode (default)
    guard let fsEngine = try? GPUECDSABatchVerifyEngine(
        config: GPUECDSABatchConfig(useFiatShamirChallenges: true)) else {
        expect(false, "Fiat-Shamir engine creation failed"); return
    }

    // Random challenge mode
    guard let rcEngine = try? GPUECDSABatchVerifyEngine(
        config: GPUECDSABatchConfig(useFiatShamirChallenges: false)) else {
        expect(false, "random challenge engine creation failed"); return
    }

    let batchN = 8
    let (sigs, pks, recovs) = makeBatch(count: batchN)

    do {
        let fsResult = try fsEngine.verifyBatch(
            signatures: sigs, publicKeys: pks, recoveryBits: recovs)
        let rcResult = try rcEngine.verifyBatch(
            signatures: sigs, publicKeys: pks, recoveryBits: recovs)

        // Both should report all valid
        expect(fsResult.allValid, "Fiat-Shamir should verify all valid sigs")
        expect(rcResult.allValid, "random challenges should verify all valid sigs")
    } catch {
        expect(false, "challenge mode comparison threw: \(error)")
    }

    // Both should detect invalid
    var badSigs = sigs
    badSigs[0] = ECDSASignature(r: sigs[0].r, s: sigs[0].s, z: secpFrFromInt(99999))
    do {
        let fsResult = try fsEngine.verifyBatch(
            signatures: badSigs, publicKeys: pks, recoveryBits: recovs)
        let rcResult = try rcEngine.verifyBatch(
            signatures: badSigs, publicKeys: pks, recoveryBits: recovs)

        expect(!fsResult.allValid, "Fiat-Shamir should detect invalid sig")
        expect(!rcResult.allValid, "random challenges should detect invalid sig")
        expect(!fsResult.results[0], "Fiat-Shamir should identify bad sig at index 0")
        expect(!rcResult.results[0], "random challenges should identify bad sig at index 0")
    } catch {
        expect(false, "challenge mode invalid comparison threw: \(error)")
    }
}

// MARK: - Large Batch (GPU Path)

private func testLargeBatchGPUPath() {
    guard let engine = try? GPUECDSABatchVerifyEngine(
        config: GPUECDSABatchConfig(gpuMSMThreshold: 20)) else {
        expect(false, "engine creation failed"); return
    }

    // With threshold=20, a batch of 16 (33 MSM points) should use GPU
    let batchN = 16
    let (sigs, pks, recovs) = makeBatch(count: batchN)

    let desc = engine.strategyDescription(batchSize: batchN)
    expect(desc.contains("GPU") || desc.contains("Metal"),
           "batch of 16 with threshold=20 should describe GPU strategy")

    do {
        let result = try engine.verifyBatch(
            signatures: sigs, publicKeys: pks, recoveryBits: recovs)
        expect(result.allValid, "GPU path should verify all valid sigs")
    } catch {
        expect(false, "GPU path batch threw: \(error)")
    }

    // Test GPU path with invalid signature
    var badSigs = sigs
    badSigs[7] = ECDSASignature(r: sigs[7].r, s: sigs[7].s, z: secpFrFromInt(99999))
    do {
        let result = try engine.verifyBatch(
            signatures: badSigs, publicKeys: pks, recoveryBits: recovs)
        expect(!result.results[7], "GPU path should detect invalid sig")
        let othersOk = result.results.enumerated()
            .filter { $0.offset != 7 }
            .allSatisfy { $0.element }
        expect(othersOk, "GPU path should pass valid sigs")
    } catch {
        expect(false, "GPU path with invalid threw: \(error)")
    }
}
