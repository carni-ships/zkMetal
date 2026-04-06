// Tests for BatchECDSAVerifier
import zkMetal
import Foundation

// Helper: create a valid ECDSA signature tuple
private func makeTestSignature(privKey: UInt64, nonce: UInt64, msgHash: UInt64)
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

func runBatchECDSATests() {
    suite("BatchECDSAVerifier")

    guard let verifier = try? BatchECDSAVerifier() else {
        expect(false, "Failed to create BatchECDSAVerifier")
        return
    }

    let gen = secp256k1Generator()
    let gProj = secpPointFromAffine(gen)

    // --- Single verification: valid signature ---
    let (validSig, validPK, _) = makeTestSignature(privKey: 42, nonce: 137, msgHash: 12345)
    expect(verifier.verifySingle(sig: validSig, pubkey: validPK),
           "valid signature should verify")

    // --- Single verification: wrong message hash ---
    let wrongZ = ECDSASignature(r: validSig.r, s: validSig.s, z: secpFrFromInt(99999))
    expect(!verifier.verifySingle(sig: wrongZ, pubkey: validPK),
           "wrong message hash should fail")

    // --- Single verification: wrong public key ---
    let wrongPK = secpPointToAffine(secpPointMulScalar(gProj, secpFrToInt(secpFrFromInt(99))))
    expect(!verifier.verifySingle(sig: validSig, pubkey: wrongPK),
           "wrong public key should fail")

    // --- Single verification: wrong r ---
    let wrongR = ECDSASignature(r: secpFrFromInt(777), s: validSig.s, z: validSig.z)
    expect(!verifier.verifySingle(sig: wrongR, pubkey: validPK),
           "wrong r should fail")

    // --- Single verification: wrong s ---
    let wrongS = ECDSASignature(r: validSig.r, s: secpFrFromInt(888), z: validSig.z)
    expect(!verifier.verifySingle(sig: wrongS, pubkey: validPK),
           "wrong s should fail")

    // --- Batch: all valid (below threshold -> individual path) ---
    do {
        let n = 3  // below kBatchThreshold=4
        var sigs = [ECDSASignature]()
        var pks = [SecpPointAffine]()
        for i in 0..<n {
            let (s, pk, _) = makeTestSignature(
                privKey: UInt64(50 + i), nonce: UInt64(500 + i * 3),
                msgHash: UInt64(10000 + i * 17))
            sigs.append(s)
            pks.append(pk)
        }
        let results = try verifier.verifyBatch(signatures: sigs, publicKeys: pks)
        expect(results.allSatisfy { $0 }, "small batch all valid should pass")
    } catch {
        expect(false, "small batch threw: \(error)")
    }

    // --- Batch: all valid (above threshold -> MSM path) ---
    let batchN = 32
    var batchSigs = [ECDSASignature]()
    var batchPKs = [SecpPointAffine]()
    var batchRecov = [UInt8]()
    for i in 0..<batchN {
        let (s, pk, rb) = makeTestSignature(
            privKey: UInt64(100 + i), nonce: UInt64(1000 + i * 7),
            msgHash: UInt64(50000 + i * 13))
        batchSigs.append(s)
        batchPKs.append(pk)
        batchRecov.append(rb)
    }

    do {
        let results = try verifier.verifyBatch(
            signatures: batchSigs, publicKeys: batchPKs, recoveryBits: batchRecov)
        expect(results.count == batchN, "batch result count should match input")
        expect(results.allSatisfy { $0 }, "batch all valid should pass")
    } catch {
        expect(false, "batch verify threw: \(error)")
    }

    // --- Batch: one invalid signature (wrong z) ---
    do {
        var badSigs = batchSigs
        let badIdx = batchN / 2
        badSigs[badIdx] = ECDSASignature(
            r: batchSigs[badIdx].r, s: batchSigs[badIdx].s, z: secpFrFromInt(99999))

        let results = try verifier.verifyBatch(
            signatures: badSigs, publicKeys: batchPKs, recoveryBits: batchRecov)
        expect(!results[badIdx], "corrupted signature should be detected")
        let othersOk = results.enumerated()
            .filter { $0.offset != badIdx }
            .allSatisfy { $0.element }
        expect(othersOk, "other signatures should still pass")
    } catch {
        expect(false, "batch with invalid threw: \(error)")
    }

    // --- Batch: multiple invalid ---
    do {
        var badSigs = batchSigs
        let badIndices = [0, batchN / 4, batchN / 2, batchN - 1]
        for idx in badIndices {
            badSigs[idx] = ECDSASignature(
                r: batchSigs[idx].r, s: batchSigs[idx].s,
                z: secpFrFromInt(UInt64(77777 + idx)))
        }

        let results = try verifier.verifyBatch(
            signatures: badSigs, publicKeys: batchPKs, recoveryBits: batchRecov)
        for idx in badIndices {
            expect(!results[idx], "invalid sig at index \(idx) should be detected")
        }
        let validIndices = Set(0..<batchN).subtracting(badIndices)
        for idx in validIndices {
            expect(results[idx], "valid sig at index \(idx) should pass")
        }
    } catch {
        expect(false, "batch with multiple invalid threw: \(error)")
    }

    // --- Probabilistic: all valid ---
    do {
        let ok = try verifier.batchVerifyProbabilistic(
            signatures: batchSigs, publicKeys: batchPKs, recoveryBits: batchRecov)
        expect(ok, "probabilistic batch all valid should return true")
    } catch {
        expect(false, "probabilistic batch threw: \(error)")
    }

    // --- Probabilistic: one invalid ---
    do {
        var badSigs = batchSigs
        badSigs[0] = ECDSASignature(
            r: batchSigs[0].r, s: batchSigs[0].s, z: secpFrFromInt(11111))
        let ok = try verifier.batchVerifyProbabilistic(
            signatures: badSigs, publicKeys: batchPKs, recoveryBits: batchRecov)
        expect(!ok, "probabilistic batch with invalid should return false")
    } catch {
        expect(false, "probabilistic batch with invalid threw: \(error)")
    }

    // --- Empty batch ---
    do {
        let results = try verifier.verifyBatch(
            signatures: [], publicKeys: [])
        expect(results.isEmpty, "empty batch should return empty results")
    } catch {
        expect(false, "empty batch threw: \(error)")
    }

    // --- Probabilistic empty ---
    do {
        let ok = try verifier.batchVerifyProbabilistic(
            signatures: [], publicKeys: [])
        expect(ok, "probabilistic empty batch should return true")
    } catch {
        expect(false, "probabilistic empty threw: \(error)")
    }

    // --- Strategy descriptions ---
    expect(BatchECDSAVerifier.strategyDescription(batchSize: 1).contains("individual"),
           "N=1 should use individual strategy")
    expect(BatchECDSAVerifier.strategyDescription(batchSize: 3).contains("individual"),
           "N=3 should use individual strategy")
    expect(BatchECDSAVerifier.strategyDescription(batchSize: 4).contains("batch"),
           "N=4 should use batch strategy")
    expect(BatchECDSAVerifier.strategyDescription(batchSize: 1000).contains("GPU"),
           "N=1000 should use GPU strategy")

    // --- Version check ---
    expect(BatchECDSAVerifier.version.version == "1.0.0", "version should be 1.0.0")
}
