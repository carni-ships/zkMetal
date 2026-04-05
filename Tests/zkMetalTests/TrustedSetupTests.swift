import zkMetal

func runTrustedSetupTests() {
    // MARK: - BN254 SRS Generation

    suite("BN254 SRS Generation")

    let tau254: [UInt64] = [42, 0, 0, 0]
    let srs254 = generateSRS(degree: 8, tau: tau254, curve: .bn254)
    expect(srs254.degree == 8, "BN254 SRS degree == 8")
    expect(srs254.g2Count == 2, "BN254 SRS g2Count == 2")
    expect(srs254.curve == .bn254, "BN254 curve type")

    // G1[0] should be the generator
    if let g1Points = srs254.bn254G1Points() {
        expect(g1Points.count == 8, "BN254 G1 point count")
        let gen = bn254G1Generator()
        let genX = fpToInt(gen.x)
        let g1_0_x = fpToInt(g1Points[0].x)
        expect(genX == g1_0_x, "BN254 G1[0] == generator")
    } else {
        expect(false, "BN254 G1 point extraction failed")
    }

    // G2[0] should be the generator
    if let g2Points = srs254.bn254G2Points() {
        expect(g2Points.count == 2, "BN254 G2 point count")
    } else {
        expect(false, "BN254 G2 point extraction failed")
    }

    // MARK: - BLS12-381 SRS Generation

    suite("BLS12-381 SRS Generation")

    let tau381: [UInt64] = [7, 0, 0, 0]
    let srs381 = generateSRS(degree: 4, tau: tau381, curve: .bls12381)
    expect(srs381.degree == 4, "BLS12-381 SRS degree == 4")
    expect(srs381.g2Count == 2, "BLS12-381 SRS g2Count == 2")
    expect(srs381.curve == .bls12381, "BLS12-381 curve type")

    if let g1Points = srs381.bls12381G1Points() {
        expect(g1Points.count == 4, "BLS12-381 G1 point count")
        let gen = bls12381G1Generator()
        let genX = fp381ToInt(gen.x)
        let g1_0_x = fp381ToInt(g1Points[0].x)
        expect(genX == g1_0_x, "BLS12-381 G1[0] == generator")
    } else {
        expect(false, "BLS12-381 G1 point extraction failed")
    }

    // MARK: - Serialization Round-Trip

    suite("Fp Serialization Round-Trip")

    // BN254 Fp round-trip
    let testFp = fpFromInt(42)
    let fpBytes = bn254FpToBigEndian(testFp)
    let fpBack = bn254FpFromBigEndian(fpBytes)
    expect(fpToInt(testFp) == fpToInt(fpBack), "BN254 Fp big-endian round-trip")

    // BLS12-381 Fp round-trip
    let testFp381v = fp381FromInt(42)
    let fp381Bytes = bls12381FpToBigEndian(testFp381v)
    let fp381Back = bls12381FpFromBigEndian(fp381Bytes)
    expect(fp381ToInt(testFp381v) == fp381ToInt(fp381Back), "BLS12-381 Fp big-endian round-trip")

    // BN254 G1 generator round-trip through SRS encoding
    let gen254v = bn254G1Generator()
    let genBytesV = bn254FpToBigEndian(gen254v.x) + bn254FpToBigEndian(gen254v.y)
    let genXBack = bn254FpFromBigEndian(Array(genBytesV[0..<32]))
    let genYBack = bn254FpFromBigEndian(Array(genBytesV[32..<64]))
    expect(fpToInt(gen254v.x) == fpToInt(genXBack), "BN254 G1 gen x round-trip")
    expect(fpToInt(gen254v.y) == fpToInt(genYBack), "BN254 G1 gen y round-trip")

    // Scalar multiplication sanity check
    suite("Scalar Mul Sanity")
    let g1 = pointFromAffine(bn254G1Generator())
    let twoFr = frFromInt(2)
    let g1x2 = cPointScalarMul(g1, twoFr)
    let g1plus = pointAdd(g1, g1)
    let a2 = batchToAffine([g1x2])[0]
    let aPlus = batchToAffine([g1plus])[0]
    expect(fpToInt(a2.x) == fpToInt(aPlus.x), "2*G = G+G (x)")
    expect(fpToInt(a2.y) == fpToInt(aPlus.y), "2*G = G+G (y)")

    // Proof of knowledge algebra test (BN254)
    suite("PoK Algebra BN254")
    do {
        let g1Base = pointFromAffine(bn254G1Generator())
        let tau = frFromInt(42)
        let k = frFromInt(7)
        let challenge = frFromInt(3)

        // tauG = tau * G
        let tauG = cPointScalarMul(g1Base, tau)
        // R = k * G
        let rPoint = cPointScalarMul(g1Base, k)
        // response = k - challenge * tau
        let response = frSub(k, frMul(challenge, tau))
        // verify: response * G + challenge * tauG = k * G
        let sG = cPointScalarMul(g1Base, response)
        let cTauG = cPointScalarMul(tauG, challenge)
        let recovered = pointAdd(sG, cTauG)

        let rAff = batchToAffine([rPoint])[0]
        let recAff = batchToAffine([recovered])[0]
        expect(fpToInt(rAff.x) == fpToInt(recAff.x), "PoK algebra: R.x matches")
        expect(fpToInt(rAff.y) == fpToInt(recAff.y), "PoK algebra: R.y matches")

        // Now test with response going through frToInt/from64 round-trip (simulating serialization)
        let responseStd = frToInt(response)
        let responseBack = frMul(Fr.from64(responseStd), Fr.from64(Fr.R2_MOD_R))
        let sG2check = cPointScalarMul(g1Base, responseBack)
        let recovered2 = pointAdd(sG2check, cTauG)
        let rec2Aff = batchToAffine([recovered2])[0]
        expect(fpToInt(rAff.x) == fpToInt(rec2Aff.x), "PoK algebra after Fr round-trip: R.x")
        expect(fpToInt(rAff.y) == fpToInt(rec2Aff.y), "PoK algebra after Fr round-trip: R.y")

        // Test with byte serialization round-trip on the point
        let rBytes = bn254FpToBigEndian(rAff.x) + bn254FpToBigEndian(rAff.y)
        let rec2Bytes = bn254FpToBigEndian(rec2Aff.x) + bn254FpToBigEndian(rec2Aff.y)
        expect(rBytes == rec2Bytes, "PoK algebra: byte-level comparison")
    }

    // MARK: - BN254 Ceremony

    suite("BN254 Ceremony")

    let ceremony = TrustedSetupCeremony()
    let initState = ceremony.initCeremony(degree: 4, curve: .bn254)
    expect(initState.contributionCount == 0, "Initial contribution count == 0")
    expect(initState.srs.degree == 4, "Initial degree == 4")

    // First contribution
    let entropy1: [UInt8] = Array(repeating: 0xAB, count: 64)
    let (state1, proof1) = ceremony.contribute(state: initState, entropy: entropy1)
    expect(state1.contributionCount == 1, "After 1st contribution, count == 1")
    expect(proof1.curve == .bn254, "Proof curve matches")

    // Verify first contribution
    let valid1 = ceremony.verifyContribution(before: initState, after: state1, proof: proof1)
    expect(valid1, "BN254 1st contribution verifies")

    // Second contribution
    let entropy2: [UInt8] = Array(repeating: 0xCD, count: 64)
    let (state2, proof2) = ceremony.contribute(state: state1, entropy: entropy2)
    expect(state2.contributionCount == 2, "After 2nd contribution, count == 2")

    let valid2 = ceremony.verifyContribution(before: state1, after: state2, proof: proof2)
    expect(valid2, "BN254 2nd contribution verifies")

    // Finalize
    let finalSRS254 = ceremony.finalize(state: state2)
    expect(finalSRS254.degree == 4, "Finalized SRS degree == 4")

    // Cross-verify: wrong proof should fail
    let crossValid = ceremony.verifyContribution(before: initState, after: state2, proof: proof1)
    // This should fail because state2 is not directly derived from initState with proof1's tau
    // (proof1 was for the transition initState -> state1, not initState -> state2)
    expect(!crossValid, "Cross-contribution verification fails as expected")

    // MARK: - BLS12-381 Ceremony

    suite("BLS12-381 Ceremony")

    let initState381 = ceremony.initCeremony(degree: 4, curve: .bls12381)
    expect(initState381.srs.curve == .bls12381, "BLS12-381 ceremony curve")

    let entropy381: [UInt8] = Array(repeating: 0xEF, count: 48)
    let (state381_1, proof381_1) = ceremony.contribute(state: initState381, entropy: entropy381)
    let valid381 = ceremony.verifyContribution(before: initState381, after: state381_1, proof: proof381_1)
    expect(valid381, "BLS12-381 contribution verifies")

    let final381 = ceremony.finalize(state: state381_1)
    expect(final381.degree == 4, "BLS12-381 finalized degree == 4")

    // MARK: - .ptau Round-Trip

    suite("SRS File Format (.ptau)")

    // Save and reload BN254 SRS
    if let ptauData = saveSRS(finalSRS254, format: .ptau) {
        expect(ptauData.count > 0, ".ptau serialization non-empty")
        if let reloaded = loadSRS(from: ptauData, format: .ptau) {
            expect(reloaded.degree == finalSRS254.degree, ".ptau round-trip degree")
            expect(reloaded.g2Count == finalSRS254.g2Count, ".ptau round-trip g2Count")
            expect(reloaded.curve == .bn254, ".ptau round-trip curve")
            expect(reloaded.g1Powers == finalSRS254.g1Powers, ".ptau round-trip G1 data")
            expect(reloaded.g2Powers == finalSRS254.g2Powers, ".ptau round-trip G2 data")
        } else {
            expect(false, ".ptau deserialization failed")
        }
    } else {
        expect(false, ".ptau serialization failed")
    }

    // Save and reload BLS12-381 SRS
    if let ptauData381 = saveSRS(final381, format: .ptau) {
        if let reloaded = loadSRS(from: ptauData381, format: .ptau) {
            expect(reloaded.curve == .bls12381, ".ptau BLS12-381 round-trip curve")
            expect(reloaded.g1Powers == final381.g1Powers, ".ptau BLS12-381 round-trip G1")
            expect(reloaded.g2Powers == final381.g2Powers, ".ptau BLS12-381 round-trip G2")
        } else {
            expect(false, ".ptau BLS12-381 deserialization failed")
        }
    } else {
        expect(false, ".ptau BLS12-381 serialization failed")
    }

    // MARK: - Ethereum KZG Format Round-Trip

    suite("SRS File Format (Ethereum KZG)")

    // Only BLS12-381 is valid for Ethereum KZG format
    if let ethData = saveSRS(final381, format: .ethereumKZG) {
        expect(ethData.count > 0, "Ethereum KZG serialization non-empty")
        if let reloaded = loadSRS(from: ethData, format: .ethereumKZG) {
            expect(reloaded.degree == final381.degree, "Ethereum KZG round-trip degree")
            expect(reloaded.g2Count == final381.g2Count, "Ethereum KZG round-trip g2Count")
            expect(reloaded.curve == .bls12381, "Ethereum KZG round-trip curve")
            // Points should be identical after decompress/reserialize
            expect(reloaded.g1Powers == final381.g1Powers, "Ethereum KZG round-trip G1 data")
            expect(reloaded.g2Powers == final381.g2Powers, "Ethereum KZG round-trip G2 data")
        } else {
            expect(false, "Ethereum KZG deserialization failed")
        }
    } else {
        expect(false, "Ethereum KZG serialization failed")
    }

    // BN254 should fail for Ethereum KZG format
    let ethBn254 = saveSRS(finalSRS254, format: .ethereumKZG)
    expect(ethBn254 == nil, "Ethereum KZG rejects BN254 SRS")

    // MARK: - SRS Consistency with Existing KZGEngine

    suite("SRS Consistency")

    // Generate SRS the old way (via KZGEngine.generateTestSRS) and the new way, compare
    let testSecret: [UInt32] = [42, 0, 0, 0, 0, 0, 0, 0]
    let oldSRS = KZGEngine.generateTestSRS(secret: testSecret, size: 8, generator: bn254G1Generator())

    let newSRS = generateSRS(degree: 8, tau: [42, 0, 0, 0], curve: .bn254)
    if let newG1 = newSRS.bn254G1Points() {
        expect(newG1.count == 8, "New SRS has 8 points")
        // Compare first point (generator)
        let oldX = fpToInt(oldSRS[0].x)
        let newX = fpToInt(newG1[0].x)
        expect(oldX == newX, "SRS[0].x matches old generateTestSRS")

        // Compare tau*G (second point)
        let oldX1 = fpToInt(oldSRS[1].x)
        let newX1 = fpToInt(newG1[1].x)
        expect(oldX1 == newX1, "SRS[1].x matches old generateTestSRS")

        // Compare last point
        let oldXLast = fpToInt(oldSRS[7].x)
        let newXLast = fpToInt(newG1[7].x)
        expect(oldXLast == newXLast, "SRS[7].x matches old generateTestSRS")
    } else {
        expect(false, "New SRS extraction failed")
    }
}
