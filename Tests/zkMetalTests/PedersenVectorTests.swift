import zkMetal
import Foundation

// MARK: - Pedersen Vector Commitment + IPA Tests

public func runPedersenVectorTests() {
    suite("Pedersen Vector Commit")

    let engine = PedersenVectorCommitEngine()

    // --- Test 1: Commit and open (basic) ---
    do {
        let params = engine.setup(n: 8)
        let values: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 1)) }
        let r = frFromInt(42)
        let c = engine.commit(values: values, params: params, blinding: r)
        expect(engine.open(values: values, blinding: r, params: params, commitment: c),
               "Commit and open: valid opening accepted")
    }

    // --- Test 2: Wrong values rejected ---
    do {
        let params = engine.setup(n: 4)
        let values: [Fr] = [frFromInt(3), frFromInt(5), frFromInt(7), frFromInt(11)]
        let r = frFromInt(99)
        let c = engine.commit(values: values, params: params, blinding: r)
        let wrong: [Fr] = [frFromInt(3), frFromInt(5), frFromInt(7), frFromInt(13)]
        expect(!engine.open(values: wrong, blinding: r, params: params, commitment: c),
               "Commit and open: wrong values rejected")
    }

    // --- Test 3: Wrong blinding rejected ---
    do {
        let params = engine.setup(n: 4)
        let values: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let r = frFromInt(42)
        let c = engine.commit(values: values, params: params, blinding: r)
        expect(!engine.open(values: values, blinding: frFromInt(43), params: params, commitment: c),
               "Commit and open: wrong blinding rejected")
    }

    // --- Test 4: Homomorphic addition ---
    // Commit(a, r1) + Commit(b, r2) == Commit(a+b, r1+r2)
    do {
        let params = engine.setup(n: 4)
        let a: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let b: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
        let r1 = frFromInt(7)
        let r2 = frFromInt(13)

        let ca = engine.commit(values: a, params: params, blinding: r1)
        let cb = engine.commit(values: b, params: params, blinding: r2)
        let cSum = PedersenVectorCommitEngine.add(ca, cb)

        // a + b element-wise
        let ab: [Fr] = (0..<4).map { frAdd(a[$0], b[$0]) }
        let rSum = frAdd(r1, r2)
        let cDirect = engine.commit(values: ab, params: params, blinding: rSum)

        expect(pointEqual(cSum, cDirect),
               "Homomorphic: commit(a,r1) + commit(b,r2) == commit(a+b, r1+r2)")
    }

    // --- Test 5: Non-hiding commit (zero blinding) ---
    do {
        let params = engine.setup(n: 4)
        let values: [Fr] = [frFromInt(5), frFromInt(10), frFromInt(15), frFromInt(20)]
        let c = engine.commit(values: values, params: params)
        expect(engine.open(values: values, blinding: Fr.zero, params: params, commitment: c),
               "Non-hiding commit (nil blinding) opens with zero blinding")
    }

    // --- Test 6: Batch commit consistency ---
    do {
        let params = engine.setup(n: 8)
        let v1: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 1)) }
        let v2: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 10)) }
        let v3: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 100)) }

        // Batch commit (no blinding for simplicity)
        let batchResults = engine.batchCommit(vectors: [v1, v2, v3], params: params)

        // Individual commits
        let c1 = engine.commit(values: v1, params: params)
        let c2 = engine.commit(values: v2, params: params)
        let c3 = engine.commit(values: v3, params: params)

        expect(pointEqual(batchResults[0], c1), "Batch commit[0] matches individual")
        expect(pointEqual(batchResults[1], c2), "Batch commit[1] matches individual")
        expect(pointEqual(batchResults[2], c3), "Batch commit[2] matches individual")
    }

    // --- Test 7: Scalar multiplication homomorphism ---
    do {
        let params = engine.setup(n: 4)
        let a: [Fr] = [frFromInt(3), frFromInt(5), frFromInt(7), frFromInt(11)]
        let r = frFromInt(17)
        let s = frFromInt(5)

        let ca = engine.commit(values: a, params: params, blinding: r)
        let sCa = PedersenVectorCommitEngine.scalarMul(s, ca)

        // s * a element-wise
        let sa: [Fr] = a.map { frMul(s, $0) }
        let sr = frMul(s, r)
        let cDirect = engine.commit(values: sa, params: params, blinding: sr)

        expect(pointEqual(sCa, cDirect),
               "Scalar mul: s * commit(a,r) == commit(s*a, s*r)")
    }

    // --- IPA Tests ---

    suite("Inner Product Argument")

    // Helper: create test generators (power-of-2 sizes)
    func makeIPA(n: Int) -> (IPAProver, IPAVerifier) {
        let (gens, q) = IPAEngine.generateTestGenerators(count: n)
        return (IPAProver(generators: gens, Q: q),
                IPAVerifier(generators: gens, Q: q))
    }

    // --- IPA Test: n=4 ---
    do {
        let (prover, verifier) = makeIPA(n: 4)
        let v: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let u: [Fr] = [frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8)]
        let t = cFrInnerProduct(v, u)

        let proof = prover.prove(v: v, u: u)

        // Compute commitment: C = MSM(G, v) + t * Q
        let scalarLimbs = v.map { frToLimbs($0) }
        let msmResult = cPippengerMSM(points: prover.generators, scalars: scalarLimbs)
        let tQ = cPointScalarMul(pointFromAffine(prover.Q), t)
        let commitment = pointAdd(msmResult, tQ)

        let valid = verifier.verify(commitment: commitment, u: u, innerProduct: t, proof: proof)
        expect(valid, "IPA n=4: valid proof accepted")
    }

    // --- IPA Test: n=16 ---
    do {
        let (prover, verifier) = makeIPA(n: 16)
        let v: [Fr] = (0..<16).map { frFromInt(UInt64($0 + 1)) }
        let u: [Fr] = (0..<16).map { frFromInt(UInt64($0 + 100)) }
        let t = cFrInnerProduct(v, u)

        let proof = prover.prove(v: v, u: u)

        let scalarLimbs = v.map { frToLimbs($0) }
        let msmResult = cPippengerMSM(points: prover.generators, scalars: scalarLimbs)
        let tQ = cPointScalarMul(pointFromAffine(prover.Q), t)
        let commitment = pointAdd(msmResult, tQ)

        let valid = verifier.verify(commitment: commitment, u: u, innerProduct: t, proof: proof)
        expect(valid, "IPA n=16: valid proof accepted")
    }

    // --- IPA Test: n=64 ---
    do {
        let (prover, verifier) = makeIPA(n: 64)
        let v: [Fr] = (0..<64).map { frFromInt(UInt64($0 + 1)) }
        let u: [Fr] = (0..<64).map { frFromInt(UInt64($0 * 3 + 7)) }
        let t = cFrInnerProduct(v, u)

        let proof = prover.prove(v: v, u: u)

        let scalarLimbs = v.map { frToLimbs($0) }
        let msmResult = cPippengerMSM(points: prover.generators, scalars: scalarLimbs)
        let tQ = cPointScalarMul(pointFromAffine(prover.Q), t)
        let commitment = pointAdd(msmResult, tQ)

        let valid = verifier.verify(commitment: commitment, u: u, innerProduct: t, proof: proof)
        expect(valid, "IPA n=64: valid proof accepted")
    }

    // --- IPA Test: n=256 ---
    do {
        let (prover, verifier) = makeIPA(n: 256)
        let v: [Fr] = (0..<256).map { frFromInt(UInt64($0 + 1)) }
        let u: [Fr] = (0..<256).map { frFromInt(UInt64($0 * 2 + 1)) }
        let t = cFrInnerProduct(v, u)

        let proof = prover.prove(v: v, u: u)

        let scalarLimbs = v.map { frToLimbs($0) }
        let msmResult = cPippengerMSM(points: prover.generators, scalars: scalarLimbs)
        let tQ = cPointScalarMul(pointFromAffine(prover.Q), t)
        let commitment = pointAdd(msmResult, tQ)

        let valid = verifier.verify(commitment: commitment, u: u, innerProduct: t, proof: proof)
        expect(valid, "IPA n=256: valid proof accepted")
    }

    // --- IPA Test: Wrong inner product rejected ---
    do {
        let (prover, verifier) = makeIPA(n: 4)
        let v: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let u: [Fr] = [frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8)]
        let t = cFrInnerProduct(v, u)

        let proof = prover.prove(v: v, u: u)

        // Build commitment with correct inner product
        let scalarLimbs = v.map { frToLimbs($0) }
        let msmResult = cPippengerMSM(points: prover.generators, scalars: scalarLimbs)
        let tQ = cPointScalarMul(pointFromAffine(prover.Q), t)
        let commitment = pointAdd(msmResult, tQ)

        // Try to verify with wrong inner product
        let wrongT = frAdd(t, frFromInt(1))
        let invalid = verifier.verify(commitment: commitment, u: u, innerProduct: wrongT, proof: proof)
        expect(!invalid, "IPA: wrong inner product rejected")
    }

    // --- IPA Test: Wrong vector u rejected ---
    do {
        let (prover, verifier) = makeIPA(n: 4)
        let v: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let u: [Fr] = [frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8)]
        let t = cFrInnerProduct(v, u)

        let proof = prover.prove(v: v, u: u)

        let scalarLimbs = v.map { frToLimbs($0) }
        let msmResult = cPippengerMSM(points: prover.generators, scalars: scalarLimbs)
        let tQ = cPointScalarMul(pointFromAffine(prover.Q), t)
        let commitment = pointAdd(msmResult, tQ)

        // Verify with wrong u
        let wrongU: [Fr] = [frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(9)]
        let invalid = verifier.verify(commitment: commitment, u: wrongU, innerProduct: t, proof: proof)
        expect(!invalid, "IPA: wrong evaluation vector u rejected")
    }

    // --- IPA Test: Proof size is log(n) ---
    do {
        let (prover, _) = makeIPA(n: 64)
        let v: [Fr] = (0..<64).map { frFromInt(UInt64($0 + 1)) }
        let u: [Fr] = (0..<64).map { frFromInt(UInt64($0 + 1)) }
        let proof = prover.prove(v: v, u: u)
        expectEqual(proof.Ls.count, 6, "IPA n=64: proof has log2(64)=6 rounds")
        expectEqual(proof.Rs.count, 6, "IPA n=64: R count matches L count")
    }
}
