import zkMetal
import Foundation

public func runVerkleProofEngineTests() {
    suite("VerkleProofEngine — Single Key Proof")

    // Use width-4 for fast tests (same protocol, smaller vectors)
    let smallIPA = BanderwagonIPAEngine(width: 4)
    let engine = VerkleProofEngine(ipaEngine: smallIPA)
    let tree = engine.createTree()

    // Insert test key-value pairs
    var key1 = [UInt8](repeating: 0, count: 32)
    key1[0] = 1; key1[31] = 0
    let val1 = fr381FromInt(42)

    var key2 = [UInt8](repeating: 0, count: 32)
    key2[0] = 2; key2[31] = 1
    let val2 = fr381FromInt(100)

    var key3 = [UInt8](repeating: 0, count: 32)
    key3[0] = 3; key3[31] = 2
    let val3 = fr381FromInt(200)

    tree.insert(key: key1, value: val1)
    tree.insert(key: key2, value: val2)
    tree.insert(key: key3, value: val3)
    tree.computeCommitments()
    let root = tree.rootCommitment()

    // Test 1: Generate proof for existing key
    let proof1 = engine.generateProof(tree: tree, key: key1)
    expect(proof1.extensionStatus == .present, "Proof key1 is present")
    expect(!proof1.commitments.isEmpty, "Proof key1 has commitments")
    expect(!proof1.ipaProofs.isEmpty, "Proof key1 has IPA proofs")
    expect(proof1.stem == Array(key1.prefix(31)), "Proof key1 stem matches")
    expect(proof1.suffixIndex == key1[31], "Proof key1 suffix matches")

    // Test 2: Generate proof for second key
    let proof2 = engine.generateProof(tree: tree, key: key2)
    expect(proof2.extensionStatus == .present, "Proof key2 is present")

    // Test 3: Verify proofs against root commitment
    // (VerkleTree.verifyProof is the IPA-based verifier)
    let verified1 = engine.verifyProof(root: root, key: key1, value: val1, proof: proof1)
    expect(verified1, "Proof key1 verifies against root")

    let verified2 = engine.verifyProof(root: root, key: key2, value: val2, proof: proof2)
    expect(verified2, "Proof key2 verifies against root")

    // Test 4: Wrong value should be detected (wrong root rejects)
    let wrongRoot = bwDouble(root) // different commitment
    let rejectedWrongRoot = !engine.verifyProof(root: wrongRoot, key: key1, value: val1, proof: proof1)
    expect(rejectedWrongRoot, "Proof rejects wrong root")

    suite("VerkleProofEngine — Proof of Absence")

    // Test 5: Absent key proof
    var keyAbsent = [UInt8](repeating: 0, count: 32)
    keyAbsent[0] = 99; keyAbsent[31] = 0 // never inserted
    let absentProof = engine.generateAbsenceProof(tree: tree, key: keyAbsent)
    expect(absentProof.extensionStatus == .absent || absentProof.extensionStatus == .otherStem,
           "Absent proof status is absent or otherStem")

    // Test 6: Verify absence proof
    let absenceVerified = engine.verifyAbsenceProof(root: root, key: keyAbsent, proof: absentProof)
    expect(absenceVerified, "Absence proof verifies")

    // Test 7: Present key should not have absence status
    let presentProof = engine.generateProof(tree: tree, key: key1)
    expect(presentProof.extensionStatus == .present, "Present key has present status")

    suite("VerkleProofEngine — Multi-Key Batch Proof")

    // Test 8: Generate multi-proof for multiple keys
    let multiProof = engine.generateMultiProof(tree: tree, keys: [key1, key2, key3])
    expect(!multiProof.commitments.isEmpty, "Multi-proof has commitments")
    expect(multiProof.extensionStatuses.count == 3, "Multi-proof has 3 statuses")
    expect(multiProof.stems.count == 3, "Multi-proof has 3 stems")
    expect(multiProof.suffixIndices.count == 3, "Multi-proof has 3 suffixes")

    // Test 9: All multi-proof extension statuses should be present
    for i in 0..<3 {
        expect(multiProof.extensionStatuses[i] == .present,
               "Multi-proof key \(i) status is present")
    }

    // Test 10: Multi-proof IPA proof structure
    let logW = Int(log2(Double(engine.width)))
    expect(multiProof.ipaProof.L.count == logW, "Multi-proof IPA has log(width) L values")
    expect(multiProof.ipaProof.R.count == logW, "Multi-proof IPA has log(width) R values")

    suite("VerkleProofEngine — Tree Update with Proof Maintenance")

    // Test 11: Update a key and verify new proof
    let tree2 = engine.createTree()
    tree2.insert(key: key1, value: val1)
    tree2.insert(key: key2, value: val2)
    tree2.computeCommitments()
    let rootBefore = tree2.rootCommitment()

    // Update key1 to a new value
    let newVal1 = fr381FromInt(999)
    let rootAfter = engine.updateAndRecommit(tree: tree2, key: key1, newValue: newVal1)

    // Root should have changed
    expect(!bwEqual(rootBefore, rootAfter), "Root changed after update")

    // New proof should verify against new root
    let proofAfter = engine.generateProof(tree: tree2, key: key1)
    expect(proofAfter.extensionStatus == .present, "Updated key is still present")
    let verifyAfter = engine.verifyProof(root: rootAfter, key: key1, value: newVal1, proof: proofAfter)
    expect(verifyAfter, "Updated proof verifies against new root")

    // Old root should reject new proof
    let rejectOldRoot = !engine.verifyProof(root: rootBefore, key: key1, value: newVal1, proof: proofAfter)
    expect(rejectOldRoot, "New proof rejects against old root")

    // Test 12: Batch update
    let tree3 = engine.createTree()
    tree3.insert(key: key1, value: val1)
    tree3.computeCommitments()
    let rootPre = tree3.rootCommitment()

    let batchRoot = engine.batchUpdateAndRecommit(tree: tree3, updates: [
        (key2, val2),
        (key3, val3)
    ])
    expect(!bwEqual(rootPre, batchRoot), "Root changed after batch update")

    // Both new keys should be present
    let got2 = tree3.get(key: key2)
    let got3 = tree3.get(key: key3)
    expect(got2 != nil, "key2 present after batch update")
    expect(got3 != nil, "key3 present after batch update")

    suite("VerkleProofEngine — Serialization Round-Trip")

    // Test 13: Single proof serialization round-trip
    let origProof = engine.generateProof(tree: tree, key: key1)
    let serialized = engine.serializeProof(origProof)
    expect(!serialized.isEmpty, "Serialized proof is non-empty")

    if let deserialized = engine.deserializeProof(serialized) {
        expect(deserialized.extensionStatus == origProof.extensionStatus,
               "Deserialized status matches")
        expect(deserialized.stem == origProof.stem, "Deserialized stem matches")
        expect(deserialized.suffixIndex == origProof.suffixIndex, "Deserialized suffix matches")
        expect(deserialized.depth == origProof.depth, "Deserialized depth matches")
        expect(deserialized.commitments.count == origProof.commitments.count,
               "Deserialized commitment count matches")
        expect(deserialized.ipaProofs.count == origProof.ipaProofs.count,
               "Deserialized IPA proof count matches")

        // Verify commitment equality
        for i in 0..<deserialized.commitments.count {
            expect(bwEqual(deserialized.commitments[i], origProof.commitments[i]),
                   "Deserialized commitment \(i) matches")
        }

        // Verify IPA proof structure
        for i in 0..<deserialized.ipaProofs.count {
            expect(deserialized.ipaProofs[i].L.count == origProof.ipaProofs[i].L.count,
                   "Deserialized IPA proof \(i) L count matches")
            expect(deserialized.ipaProofs[i].R.count == origProof.ipaProofs[i].R.count,
                   "Deserialized IPA proof \(i) R count matches")
            expect(deserialized.ipaProofs[i].a == origProof.ipaProofs[i].a,
                   "Deserialized IPA proof \(i) final scalar matches")

            for j in 0..<deserialized.ipaProofs[i].L.count {
                expect(bwEqual(deserialized.ipaProofs[i].L[j], origProof.ipaProofs[i].L[j]),
                       "Deserialized IPA proof \(i) L[\(j)] matches")
                expect(bwEqual(deserialized.ipaProofs[i].R[j], origProof.ipaProofs[i].R[j]),
                       "Deserialized IPA proof \(i) R[\(j)] matches")
            }
        }
    } else {
        expect(false, "Single proof deserialization failed")
    }

    // Test 14: Multi-proof serialization round-trip
    let origMulti = engine.generateMultiProof(tree: tree, keys: [key1, key2])
    let serializedMulti = engine.serializeMultiProof(origMulti)
    expect(!serializedMulti.isEmpty, "Serialized multi-proof is non-empty")

    if let deserializedMulti = engine.deserializeMultiProof(serializedMulti) {
        expect(deserializedMulti.commitments.count == origMulti.commitments.count,
               "Multi-proof deserialized commitment count")
        expect(deserializedMulti.extensionStatuses.count == origMulti.extensionStatuses.count,
               "Multi-proof deserialized status count")
        expect(deserializedMulti.stems.count == origMulti.stems.count,
               "Multi-proof deserialized stem count")
        expect(deserializedMulti.ipaProof.L.count == origMulti.ipaProof.L.count,
               "Multi-proof deserialized IPA L count")
        expect(deserializedMulti.ipaProof.a == origMulti.ipaProof.a,
               "Multi-proof deserialized final scalar matches")

        for i in 0..<deserializedMulti.commitments.count {
            expect(bwEqual(deserializedMulti.commitments[i], origMulti.commitments[i]),
                   "Multi-proof deserialized commitment \(i) matches")
        }

        for i in 0..<deserializedMulti.extensionStatuses.count {
            expect(deserializedMulti.extensionStatuses[i] == origMulti.extensionStatuses[i],
                   "Multi-proof deserialized status \(i)")
            expect(deserializedMulti.stems[i] == origMulti.stems[i],
                   "Multi-proof deserialized stem \(i)")
            expect(deserializedMulti.suffixIndices[i] == origMulti.suffixIndices[i],
                   "Multi-proof deserialized suffix \(i)")
        }
    } else {
        expect(false, "Multi-proof deserialization failed")
    }

    // Test 15: Corrupt data should fail deserialization
    let corrupt = [UInt8](repeating: 0xFF, count: 10)
    let badDeserialize = engine.deserializeProof(corrupt)
    expect(badDeserialize == nil, "Corrupt data fails deserialization")

    suite("VerkleProofEngine — IPA Node Opening")

    // Test 16: Direct node opening proof
    let nodeValues = (0..<engine.width).map { fr381FromInt(UInt64($0 * 10 + 1)) }
    let nodeCommitment = engine.commitNode(values: nodeValues)
    expect(!bwIsIdentity(nodeCommitment.point), "Node commitment is non-trivial")

    for idx in [0, 1, engine.width - 1] {
        let (openProof, openValue) = engine.createNodeOpeningProof(
            nodeCommitment: nodeCommitment, childIndex: idx)
        let expectedVal = bwScalarFromFr381(nodeValues[idx])
        expect(openValue == expectedVal, "Node opening value at index \(idx)")

        let openOk = engine.verifyNodeOpeningProof(
            commitment: nodeCommitment.point,
            childIndex: idx,
            value: openValue,
            proof: openProof)
        expect(openOk, "Node opening proof verifies at index \(idx)")

        // Wrong index should fail
        let wrongIdx = (idx + 1) % engine.width
        let openBad = engine.verifyNodeOpeningProof(
            commitment: nodeCommitment.point,
            childIndex: wrongIdx,
            value: openValue,
            proof: openProof)
        expect(!openBad, "Node opening proof rejects wrong index \(wrongIdx)")
    }

    suite("VerkleProofEngine — Utility Functions")

    // Test 17: Key construction helpers
    let stem = [UInt8](repeating: 0xAB, count: 31)
    let suffix: UInt8 = 0xCD
    let constructed = VerkleProofEngine.makeKey(stem: stem, suffix: suffix)
    expect(constructed.count == 32, "Constructed key is 32 bytes")
    expect(VerkleProofEngine.stemFromKey(constructed) == stem, "Stem extraction")
    expect(VerkleProofEngine.suffixFromKey(constructed) == suffix, "Suffix extraction")

    suite("VerkleProofEngine — Performance (1000-key tree)")

    // Test 18: Build and prove 1000-key tree (width-4 for speed)
    let perfTree = engine.createTree()
    let numKeys = 1000
    var perfKeys = [[UInt8]]()
    perfKeys.reserveCapacity(numKeys)

    let t0 = CFAbsoluteTimeGetCurrent()

    for i in 0..<numKeys {
        var key = [UInt8](repeating: 0, count: 32)
        // Distribute keys across the tree
        key[0] = UInt8(i & 0xFF)
        key[1] = UInt8((i >> 8) & 0xFF)
        key[31] = UInt8(i % 256)
        let val = fr381FromInt(UInt64(i + 1))
        perfTree.insert(key: key, value: val)
        perfKeys.append(key)
    }

    let tInsert = CFAbsoluteTimeGetCurrent()
    perfTree.computeCommitments()
    let tCommit = CFAbsoluteTimeGetCurrent()

    let perfRoot = perfTree.rootCommitment()
    expect(!bwIsIdentity(perfRoot), "1000-key tree has non-trivial root")

    // Generate proof for first key
    let perfProof = engine.generateProof(tree: perfTree, key: perfKeys[0])
    let tProve = CFAbsoluteTimeGetCurrent()
    expect(perfProof.extensionStatus == .present, "1000-key proof is present")

    // Generate multi-proof for 10 keys
    let sampleKeys = Array(perfKeys.prefix(10))
    let perfMulti = engine.generateMultiProof(tree: perfTree, keys: sampleKeys)
    let tMulti = CFAbsoluteTimeGetCurrent()
    expect(perfMulti.extensionStatuses.count == 10, "Multi-proof has 10 statuses")

    // Print timing
    print(String(format: "  1000-key tree: insert=%.1fms commit=%.1fms prove=%.1fms multi(10)=%.1fms",
                 (tInsert - t0) * 1000,
                 (tCommit - tInsert) * 1000,
                 (tProve - tCommit) * 1000,
                 (tMulti - tProve) * 1000))

    // Test 19: Lookups still work on large tree
    for i in stride(from: 0, to: numKeys, by: 100) {
        let got = perfTree.get(key: perfKeys[i])
        expect(got != nil, "Lookup key \(i) in 1000-key tree")
    }
}
