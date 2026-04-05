import zkMetal

// MARK: - Pedersen Engine Tests

func runPedersenEngineTests() {
    suite("Pedersen Engine")

    // --- BN254 Tests ---

    // Test 1: BN254 basic commit and verify
    do {
        let params = PedersenEngine.setup(size: 4, curve: .bn254)
        let values: [CurveScalar] = [
            .bn254(frFromInt(3)), .bn254(frFromInt(5)),
            .bn254(frFromInt(7)), .bn254(frFromInt(11))
        ]
        let r: CurveScalar = .bn254(frFromInt(42))
        let c = PedersenEngine.commit(values: values, randomness: r, params: params)
        expect(PedersenEngine.verify(commitment: c, values: values, randomness: r, params: params),
               "BN254 basic commit/verify")
    }

    // Test 2: BN254 wrong values should not verify
    do {
        let params = PedersenEngine.setup(size: 4, curve: .bn254)
        let values: [CurveScalar] = [
            .bn254(frFromInt(3)), .bn254(frFromInt(5)),
            .bn254(frFromInt(7)), .bn254(frFromInt(11))
        ]
        let r: CurveScalar = .bn254(frFromInt(42))
        let c = PedersenEngine.commit(values: values, randomness: r, params: params)
        let wrong: [CurveScalar] = [
            .bn254(frFromInt(3)), .bn254(frFromInt(5)),
            .bn254(frFromInt(7)), .bn254(frFromInt(13))
        ]
        expect(!PedersenEngine.verify(commitment: c, values: wrong, randomness: r, params: params),
               "BN254 wrong values rejected")
    }

    // Test 3: BN254 wrong randomness should not verify
    do {
        let params = PedersenEngine.setup(size: 2, curve: .bn254)
        let values: [CurveScalar] = [.bn254(frFromInt(1)), .bn254(frFromInt(2))]
        let r: CurveScalar = .bn254(frFromInt(42))
        let c = PedersenEngine.commit(values: values, randomness: r, params: params)
        let wrongR: CurveScalar = .bn254(frFromInt(43))
        expect(!PedersenEngine.verify(commitment: c, values: values, randomness: wrongR, params: params),
               "BN254 wrong randomness rejected")
    }

    // Test 4: Additive homomorphism
    do {
        let params = PedersenEngine.setup(size: 3, curve: .bn254)
        let a: [CurveScalar] = [.bn254(frFromInt(2)), .bn254(frFromInt(3)), .bn254(frFromInt(5))]
        let b: [CurveScalar] = [.bn254(frFromInt(7)), .bn254(frFromInt(11)), .bn254(frFromInt(13))]
        let r1: CurveScalar = .bn254(frFromInt(10))
        let r2: CurveScalar = .bn254(frFromInt(20))

        let c1 = PedersenEngine.commit(values: a, randomness: r1, params: params)
        let c2 = PedersenEngine.commit(values: b, randomness: r2, params: params)
        let cSum = PedersenEngine.homomorphicAdd(c1, c2)

        let ab: [CurveScalar] = [
            .bn254(frAdd(frFromInt(2), frFromInt(7))),
            .bn254(frAdd(frFromInt(3), frFromInt(11))),
            .bn254(frAdd(frFromInt(5), frFromInt(13)))
        ]
        let rSum: CurveScalar = .bn254(frAdd(frFromInt(10), frFromInt(20)))
        let cDirect = PedersenEngine.commit(values: ab, randomness: rSum, params: params)
        expect(curvePointEqual(cSum, cDirect), "BN254 additive homomorphism")
    }

    // Test 5: Scalar multiplication homomorphism
    do {
        let params = PedersenEngine.setup(size: 2, curve: .bn254)
        let a: [CurveScalar] = [.bn254(frFromInt(3)), .bn254(frFromInt(7))]
        let r: CurveScalar = .bn254(frFromInt(5))
        let s: CurveScalar = .bn254(frFromInt(4))

        let c = PedersenEngine.commit(values: a, randomness: r, params: params)
        let sc = PedersenEngine.homomorphicScalarMul(commitment: c, scalar: s)

        let sa: [CurveScalar] = [
            .bn254(frMul(frFromInt(3), frFromInt(4))),
            .bn254(frMul(frFromInt(7), frFromInt(4)))
        ]
        let sr: CurveScalar = .bn254(frMul(frFromInt(5), frFromInt(4)))
        let cDirect = PedersenEngine.commit(values: sa, randomness: sr, params: params)
        expect(curvePointEqual(sc, cDirect), "BN254 scalar multiplication homomorphism")
    }

    // Test 6: Zero randomness (non-hiding commitment)
    do {
        let params = PedersenEngine.setup(size: 2, curve: .bn254)
        let values: [CurveScalar] = [.bn254(frFromInt(10)), .bn254(frFromInt(20))]
        let zeroR: CurveScalar = .bn254(.zero)
        let c = PedersenEngine.commit(values: values, randomness: zeroR, params: params)
        expect(PedersenEngine.verify(commitment: c, values: values, randomness: zeroR, params: params),
               "BN254 zero randomness commit/verify")
    }

    // Test 7: Opening round-trip
    do {
        let params = PedersenEngine.setup(size: 2, curve: .bn254)
        let values: [CurveScalar] = [.bn254(frFromInt(10)), .bn254(frFromInt(20))]
        let r: CurveScalar = .bn254(frFromInt(99))
        let c = PedersenEngine.commit(values: values, randomness: r, params: params)
        let opening = PedersenEngine.open(values: values, randomness: r)
        expect(PedersenEngine.verify(commitment: c, values: opening.values,
                                     randomness: opening.randomness, params: params),
               "BN254 opening round-trip")
        expect(opening.curve == .bn254, "Opening curve is BN254")
    }

    // Test 8: Batch commit matches individual commits
    do {
        let params = PedersenEngine.setup(size: 4, curve: .bn254)
        let v1: [CurveScalar] = [.bn254(frFromInt(1)), .bn254(frFromInt(2)),
                                  .bn254(frFromInt(3)), .bn254(frFromInt(4))]
        let v2: [CurveScalar] = [.bn254(frFromInt(5)), .bn254(frFromInt(6)),
                                  .bn254(frFromInt(7)), .bn254(frFromInt(8))]
        let r1: CurveScalar = .bn254(frFromInt(10))
        let r2: CurveScalar = .bn254(frFromInt(20))

        let batch = PedersenEngine.batchCommit(valueVectors: [v1, v2],
                                               randomness: [r1, r2], params: params)
        let single1 = PedersenEngine.commit(values: v1, randomness: r1, params: params)
        let single2 = PedersenEngine.commit(values: v2, randomness: r2, params: params)
        expect(batch.count == 2, "Batch commit returns 2 results")
        expect(curvePointEqual(batch[0], single1), "Batch commit[0] matches single")
        expect(curvePointEqual(batch[1], single2), "Batch commit[1] matches single")
    }

    // --- Pallas Tests ---

    // Test 9: Pallas commit and verify
    do {
        let params = PedersenEngine.setup(size: 3, curve: .pallas)
        let values: [CurveScalar] = [
            .pallas(vestaFromInt(3)), .pallas(vestaFromInt(5)), .pallas(vestaFromInt(7))
        ]
        let r: CurveScalar = .pallas(vestaFromInt(42))
        let c = PedersenEngine.commit(values: values, randomness: r, params: params)
        expect(PedersenEngine.verify(commitment: c, values: values, randomness: r, params: params),
               "Pallas basic commit/verify")
    }

    // Test 10: Pallas additive homomorphism
    do {
        let params = PedersenEngine.setup(size: 2, curve: .pallas)
        let a: [CurveScalar] = [.pallas(vestaFromInt(3)), .pallas(vestaFromInt(7))]
        let b: [CurveScalar] = [.pallas(vestaFromInt(5)), .pallas(vestaFromInt(11))]
        let r1: CurveScalar = .pallas(vestaFromInt(10))
        let r2: CurveScalar = .pallas(vestaFromInt(20))

        let c1 = PedersenEngine.commit(values: a, randomness: r1, params: params)
        let c2 = PedersenEngine.commit(values: b, randomness: r2, params: params)
        let cSum = PedersenEngine.homomorphicAdd(c1, c2)

        let ab: [CurveScalar] = [
            .pallas(vestaAdd(vestaFromInt(3), vestaFromInt(5))),
            .pallas(vestaAdd(vestaFromInt(7), vestaFromInt(11)))
        ]
        let rSum: CurveScalar = .pallas(vestaAdd(vestaFromInt(10), vestaFromInt(20)))
        let cDirect = PedersenEngine.commit(values: ab, randomness: rSum, params: params)
        expect(curvePointEqual(cSum, cDirect), "Pallas additive homomorphism")
    }

    // --- Vesta Tests ---

    // Test 11: Vesta commit and verify
    do {
        let params = PedersenEngine.setup(size: 3, curve: .vesta)
        let values: [CurveScalar] = [
            .vesta(pallasFromInt(3)), .vesta(pallasFromInt(5)), .vesta(pallasFromInt(7))
        ]
        let r: CurveScalar = .vesta(pallasFromInt(42))
        let c = PedersenEngine.commit(values: values, randomness: r, params: params)
        expect(PedersenEngine.verify(commitment: c, values: values, randomness: r, params: params),
               "Vesta basic commit/verify")
    }

    // --- BLS12-381 Tests ---

    // Test 12: BLS12-381 commit and verify
    do {
        let params = PedersenEngine.setup(size: 3, curve: .bls12_381)
        let values: [CurveScalar] = [
            .bls12_381(fr381FromInt(3)), .bls12_381(fr381FromInt(5)), .bls12_381(fr381FromInt(7))
        ]
        let r: CurveScalar = .bls12_381(fr381FromInt(42))
        let c = PedersenEngine.commit(values: values, randomness: r, params: params)
        expect(PedersenEngine.verify(commitment: c, values: values, randomness: r, params: params),
               "BLS12-381 basic commit/verify")
    }

    // Test 13: BLS12-381 wrong values rejected
    do {
        let params = PedersenEngine.setup(size: 2, curve: .bls12_381)
        let values: [CurveScalar] = [.bls12_381(fr381FromInt(3)), .bls12_381(fr381FromInt(5))]
        let r: CurveScalar = .bls12_381(fr381FromInt(42))
        let c = PedersenEngine.commit(values: values, randomness: r, params: params)
        let wrong: [CurveScalar] = [.bls12_381(fr381FromInt(3)), .bls12_381(fr381FromInt(6))]
        expect(!PedersenEngine.verify(commitment: c, values: wrong, randomness: r, params: params),
               "BLS12-381 wrong values rejected")
    }

    // --- Vector Pedersen Tests ---

    // Test 14: Vector Pedersen basic commit/verify (BN254)
    do {
        let vParams = VectorPedersenParams.generate(size: 4, curve: .bn254)
        let vec: [CurveScalar] = [
            .bn254(frFromInt(1)), .bn254(frFromInt(2)),
            .bn254(frFromInt(3)), .bn254(frFromInt(4))
        ]
        let blind: CurveScalar = .bn254(frFromInt(7))
        let c = VectorPedersenCommitment.commit(vector: vec, blinding: blind, params: vParams)
        expect(VectorPedersenCommitment.verify(commitment: c, vector: vec,
                                               blinding: blind, params: vParams),
               "Vector Pedersen basic commit/verify")
    }

    // Test 15: Inner product commitment
    do {
        let vParams = VectorPedersenParams.generate(size: 4, curve: .bn254)
        let a: [CurveScalar] = [
            .bn254(frFromInt(1)), .bn254(frFromInt(2)),
            .bn254(frFromInt(3)), .bn254(frFromInt(4))
        ]
        let b: [CurveScalar] = [
            .bn254(frFromInt(5)), .bn254(frFromInt(6)),
            .bn254(frFromInt(7)), .bn254(frFromInt(8))
        ]
        let blind: CurveScalar = .bn254(frFromInt(99))
        let c = VectorPedersenCommitment.innerProductCommit(a: a, b: b,
                                                             blinding: blind, params: vParams)
        expect(VectorPedersenCommitment.verifyInnerProduct(commitment: c, a: a, b: b,
                                                            blinding: blind, params: vParams),
               "Inner product commitment verify")
    }

    // Test 16: Inner product value check
    do {
        let a: [CurveScalar] = [.bn254(frFromInt(2)), .bn254(frFromInt(3))]
        let b: [CurveScalar] = [.bn254(frFromInt(4)), .bn254(frFromInt(5))]
        let ip = innerProduct(a, b)
        // <a, b> = 2*4 + 3*5 = 8 + 15 = 23
        if case .bn254(let val) = ip {
            expect(frEq(val, frFromInt(23)), "Inner product = 23")
        } else {
            expect(false, "Inner product wrong type")
        }
    }

    // Test 17: Blinding manager
    do {
        let bm = BlindingManager.random(curve: .bn254, seed: 42)
        var bm2 = bm
        bm2.addBlinding(.bn254(frFromInt(10)))
        expect(bm2.curve == .bn254, "Blinding manager operations")
    }

    // Test 18: Pallas vector Pedersen
    do {
        let vParams = VectorPedersenParams.generate(size: 3, curve: .pallas)
        let vec: [CurveScalar] = [
            .pallas(vestaFromInt(10)), .pallas(vestaFromInt(20)), .pallas(vestaFromInt(30))
        ]
        let blind: CurveScalar = .pallas(vestaFromInt(5))
        let c = VectorPedersenCommitment.commit(vector: vec, blinding: blind, params: vParams)
        expect(VectorPedersenCommitment.verify(commitment: c, vector: vec,
                                               blinding: blind, params: vParams),
               "Pallas vector Pedersen commit/verify")
    }
}
