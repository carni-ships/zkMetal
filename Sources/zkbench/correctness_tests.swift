// Correctness tests for primitives with gaps
// Covers: secp256k1 MSM, BLS12-377 MSM (full scalar), KZG verify,
//         ParallelHash cross-checks, ParallelMSM full-scalar
import zkMetal
import Foundation

public func runCorrectnessTests() {
    var passed = 0
    var failed = 0

    func check(_ name: String, _ ok: Bool) {
        if ok {
            print("  [pass] \(name)")
            passed += 1
        } else {
            print("  [FAIL] \(name)")
            failed += 1
        }
    }

    // ===== secp256k1 MSM correctness =====
    print("\n--- secp256k1 MSM Correctness ---")
    do {
        let engine = try Secp256k1MSM()
        let gen = secp256k1Generator()
        let gProj = secpPointFromAffine(gen)

        // Test 1: MSM([G], [1]) = G
        let oneScalar: [UInt32] = [1, 0, 0, 0, 0, 0, 0, 0]
        let result1 = try engine.msm(points: [gen], scalars: [oneScalar])
        let r1Aff = secpPointToAffine(result1)
        check("MSM([G], [1]) = G",
              secpToInt(r1Aff.x) == secpToInt(gen.x) &&
              secpToInt(r1Aff.y) == secpToInt(gen.y))

        // Test 2: MSM([G, G], [1, 1]) = 2G
        let twoG = secpPointDouble(gProj)
        let result2 = try engine.msm(points: [gen, gen], scalars: [oneScalar, oneScalar])
        let r2Aff = secpPointToAffine(result2)
        let twoGAff = secpPointToAffine(twoG)
        check("MSM([G,G], [1,1]) = 2G",
              secpToInt(r2Aff.x) == secpToInt(twoGAff.x) &&
              secpToInt(r2Aff.y) == secpToInt(twoGAff.y))

        // Test 3: MSM([G], [5]) = 5G
        let fiveScalar: [UInt32] = [5, 0, 0, 0, 0, 0, 0, 0]
        let fiveG = secpPointMulInt(gProj, 5)
        let result3 = try engine.msm(points: [gen], scalars: [fiveScalar])
        let r3Aff = secpPointToAffine(result3)
        let fiveGAff = secpPointToAffine(fiveG)
        check("MSM([G], [5]) = 5G",
              secpToInt(r3Aff.x) == secpToInt(fiveGAff.x) &&
              secpToInt(r3Aff.y) == secpToInt(fiveGAff.y))

        // Test 4: MSM with 16 points, low scalars — CPU vs GPU cross-check
        let n = 16
        var pts = [SecpPointAffine]()
        var acc = gProj
        for _ in 0..<n {
            pts.append(secpPointToAffine(acc))
            acc = secpPointAdd(acc, gProj)
        }
        var scalars16 = [[UInt32]]()
        for i in 0..<n {
            scalars16.append([UInt32(i + 1), 0, 0, 0, 0, 0, 0, 0])
        }

        // CPU: sum_i (i+1) * (i+1)G = sum_i (i+1)^2 * G
        var cpuSum = 0
        for i in 0..<n { cpuSum += (i + 1) * (i + 1) }
        let cpuExpected = secpPointMulInt(gProj, cpuSum)

        let gpuResult = try engine.msm(points: pts, scalars: scalars16)
        let gpuAff = secpPointToAffine(gpuResult)
        let cpuAff = secpPointToAffine(cpuExpected)
        check("MSM 16 pts (low scalars) CPU=GPU",
              secpToInt(gpuAff.x) == secpToInt(cpuAff.x) &&
              secpToInt(gpuAff.y) == secpToInt(cpuAff.y))

        // Test 5: MSM with random full-range scalars (256 points)
        let nFull = 256
        var fullPts = [SecpPointAffine]()
        acc = gProj
        for _ in 0..<nFull {
            fullPts.append(secpPointToAffine(acc))
            acc = secpPointAdd(acc, gProj)
        }
        var rng: UInt64 = 0xBEEF_CAFE
        var fullScalars = [[UInt32]]()
        for _ in 0..<nFull {
            var limbs = [UInt32](repeating: 0, count: 8)
            for j in 0..<8 {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                limbs[j] = UInt32(truncatingIfNeeded: rng >> 32)
            }
            fullScalars.append(limbs)
        }

        // Run GPU MSM twice — should be deterministic
        let r5a = try engine.msm(points: fullPts, scalars: fullScalars)
        let r5b = try engine.msm(points: fullPts, scalars: fullScalars)
        let r5aAff = secpPointToAffine(r5a)
        let r5bAff = secpPointToAffine(r5b)
        check("MSM 256 pts (full scalars) deterministic",
              secpToInt(r5aAff.x) == secpToInt(r5bAff.x) &&
              secpToInt(r5aAff.y) == secpToInt(r5bAff.y))

        // Result should be on curve
        let x = r5aAff.x
        let y = r5aAff.y
        let y2 = secpSqr(y)
        let x3 = secpMul(secpSqr(x), x)
        let rhs = secpAdd(x3, secpFromInt(7))
        check("MSM 256 pts result on curve", secpToInt(y2) == secpToInt(rhs))
    } catch {
        print("  [FAIL] secp256k1 MSM error: \(error)")
        failed += 1
    }

    // ===== BLS12-377 MSM full-scalar correctness =====
    print("\n--- BLS12-377 MSM Correctness (full scalar) ---")
    do {
        let engine = try BLS12377MSM()
        let gen = bls12377Generator()
        let gProj = point377FromAffine(gen)

        // Test 1: MSM([G], [1]) = G
        let oneScalar: [UInt32] = [1, 0, 0, 0, 0, 0, 0, 0]
        let result1 = try engine.msm(points: [gen], scalars: [oneScalar])
        if let r1Aff = point377ToAffine(result1) {
            check("MSM([G], [1]) = G",
                  fq377ToInt(r1Aff.x) == fq377ToInt(gen.x) &&
                  fq377ToInt(r1Aff.y) == fq377ToInt(gen.y))
        } else {
            check("MSM([G], [1]) = G (not identity)", false)
        }

        // Test 2: MSM([G], [7]) = 7G
        let sevenScalar: [UInt32] = [7, 0, 0, 0, 0, 0, 0, 0]
        let sevenG = point377MulInt(gProj, 7)
        let result2 = try engine.msm(points: [gen], scalars: [sevenScalar])
        if let r2Aff = point377ToAffine(result2), let r2Exp = point377ToAffine(sevenG) {
            check("MSM([G], [7]) = 7G",
                  fq377ToInt(r2Aff.x) == fq377ToInt(r2Exp.x) &&
                  fq377ToInt(r2Aff.y) == fq377ToInt(r2Exp.y))
        } else {
            check("MSM([G], [7]) = 7G", false)
        }

        // Test 3: Deterministic with random full scalars
        let n = 256
        var pts377 = [Point377Affine]()
        var acc377 = gProj
        for _ in 0..<n {
            if let aff = point377ToAffine(acc377) { pts377.append(aff) }
            acc377 = point377Add(acc377, gProj)
        }
        var rng: UInt64 = 0xDEAD_1234
        var scalars377 = [[UInt32]]()
        for _ in 0..<n {
            var limbs = [UInt32](repeating: 0, count: 8)
            for j in 0..<8 {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                limbs[j] = UInt32(truncatingIfNeeded: rng >> 32)
            }
            scalars377.append(limbs)
        }

        let r3a = try engine.msm(points: pts377, scalars: scalars377)
        let r3b = try engine.msm(points: pts377, scalars: scalars377)
        if let aff3a = point377ToAffine(r3a), let aff3b = point377ToAffine(r3b) {
            check("MSM 256 pts (full scalars) deterministic",
                  fq377ToInt(aff3a.x) == fq377ToInt(aff3b.x) &&
                  fq377ToInt(aff3a.y) == fq377ToInt(aff3b.y))
            // On curve
            let y2 = fq377Sqr(aff3a.y)
            let x3 = fq377Mul(fq377Sqr(aff3a.x), aff3a.x)
            let rhs = fq377Add(x3, fq377FromInt(1))
            check("MSM 256 pts result on curve", fq377ToInt(y2) == fq377ToInt(rhs))
        } else {
            check("MSM 256 pts deterministic", point377IsIdentity(r3a) == point377IsIdentity(r3b))
        }
    } catch {
        print("  [FAIL] BLS12-377 MSM error: \(error)")
        failed += 1
    }

    // ===== KZG consistency (no pairing, but algebraic checks) =====
    print("\n--- KZG Consistency Checks ---")
    do {
        let gx = fpFromInt(1)
        let gy = fpFromInt(2)
        let generator = PointAffine(x: gx, y: gy)
        let secret: [UInt32] = [42, 0, 0, 0, 0, 0, 0, 0]
        let srs = KZGEngine.generateTestSRS(secret: secret, size: 256, generator: generator)
        let engine = try KZGEngine(srs: srs)

        // Test 1: Commit to [1] (constant poly) = 1*G = G
        let constPoly = [frFromInt(1)]
        let c1 = try engine.commit(constPoly)
        let c1Aff = batchToAffine([c1])[0]
        check("Commit([1]) = G", fpToInt(c1Aff.x) == fpToInt(gx) && fpToInt(c1Aff.y) == fpToInt(gy))

        // Test 2: Open constant poly at any point returns constant
        let p1 = try engine.open(constPoly, at: frFromInt(999))
        check("Open(const, z=999) eval = 1", frToInt(p1.evaluation)[0] == 1)
        check("Open(const) witness = identity", pointIsIdentity(p1.witness))

        // Test 3: p(x) = x, commit should be s*G = SRS[1]
        let linPoly = [frFromInt(0), frFromInt(1)]  // 0 + 1*x
        let c2 = try engine.commit(linPoly)
        let c2Aff = batchToAffine([c2])[0]
        check("Commit(x) = SRS[1]", fpToInt(c2Aff.x) == fpToInt(srs[1].x) && fpToInt(c2Aff.y) == fpToInt(srs[1].y))

        // Test 4: p(x) = x, p(42) = 42 (since secret = 42)
        let p2 = try engine.open(linPoly, at: frFromInt(42))
        check("Open(x, z=42) eval = 42", frToInt(p2.evaluation)[0] == 42)

        // Test 5: p(x) = x, p(0) = 0
        let p3 = try engine.open(linPoly, at: frFromInt(0))
        check("Open(x, z=0) eval = 0", frToInt(p3.evaluation)[0] == 0)

        // Test 6: Linearity: commit(a*p + b*q) should equal a*commit(p) + b*commit(q) in projective
        let polyA = [frFromInt(1), frFromInt(2), frFromInt(3)]  // 1 + 2x + 3x^2
        let polyB = [frFromInt(4), frFromInt(5), frFromInt(6)]  // 4 + 5x + 6x^2
        let polySum = zip(polyA, polyB).map { frAdd($0, $1) }   // 5 + 7x + 9x^2
        let cA = try engine.commit(polyA)
        let cB = try engine.commit(polyB)
        let cSum = try engine.commit(polySum)
        let cAddedManual = pointAdd(cA, cB)
        let cSumAff = batchToAffine([cSum])[0]
        let cManAff = batchToAffine([cAddedManual])[0]
        check("Commit linearity: C(p+q) = C(p)+C(q)",
              fpToInt(cSumAff.x) == fpToInt(cManAff.x) && fpToInt(cSumAff.y) == fpToInt(cManAff.y))

        // Test 7: Multi-point evaluation consistency
        // p(5) = 1 + 10 + 75 = 86, p(0) = 1, p(1) = 6
        let p = [frFromInt(1), frFromInt(2), frFromInt(3)]
        let open5 = try engine.open(p, at: frFromInt(5))
        let open0 = try engine.open(p, at: frFromInt(0))
        let open1 = try engine.open(p, at: frFromInt(1))
        check("p(5)=86", frToInt(open5.evaluation)[0] == 86)
        check("p(0)=1", frToInt(open0.evaluation)[0] == 1)
        check("p(1)=6", frToInt(open1.evaluation)[0] == 6)
    } catch {
        print("  [FAIL] KZG error: \(error)")
        failed += 1
    }

    // ===== Parallel Hash cross-checks =====
    print("\n--- ParallelHash Cross-Checks ---")

    // Poseidon2 batch: parallel vs sequential
    let p2N = 256
    var p2Pairs = [(Fr, Fr)]()
    for i in 0..<p2N {
        p2Pairs.append((frFromInt(UInt64(i)), frFromInt(UInt64(i + p2N))))
    }
    let seqP2 = p2Pairs.map { poseidon2Hash($0.0, $0.1) }
    let parP2 = parallelPoseidon2Batch(p2Pairs)
    var p2Match = true
    for i in 0..<p2N {
        if frToInt(seqP2[i]) != frToInt(parP2[i]) { p2Match = false; break }
    }
    check("Poseidon2 batch (256): parallel = sequential", p2Match)

    // Keccak batch: parallel vs sequential
    let kN = 256
    var kInputs = [[UInt8]]()
    for i in 0..<kN {
        var block = [UInt8](repeating: 0, count: 64)
        let val = UInt64(i)
        withUnsafeBytes(of: val) { block.replaceSubrange(0..<8, with: $0) }
        kInputs.append(block)
    }
    let seqK = kInputs.map { keccak256($0) }
    let parK = parallelKeccak256Batch(kInputs)
    var kMatch = true
    for i in 0..<kN {
        if seqK[i] != parK[i] { kMatch = false; break }
    }
    check("Keccak batch (256): parallel = sequential", kMatch)

    // Blake3 batch: parallel vs sequential
    let b3N = 256
    var b3Inputs = [[UInt8]]()
    for i in 0..<b3N {
        var block = [UInt8](repeating: 0, count: 64)
        let val = UInt64(i)
        withUnsafeBytes(of: val) { block.replaceSubrange(0..<8, with: $0) }
        b3Inputs.append(block)
    }
    let seqB3 = b3Inputs.map { blake3($0) }
    let parB3 = parallelBlake3Batch(b3Inputs)
    var b3Match = true
    for i in 0..<b3N {
        if seqB3[i] != parB3[i] { b3Match = false; break }
    }
    check("Blake3 batch (256): parallel = sequential", b3Match)

    // ===== Parallel Merkle cross-checks =====
    print("\n--- ParallelMerkle Cross-Checks ---")

    // Poseidon2 Merkle: parallel vs sequential
    let mN = 64
    var leaves = [Fr]()
    for i in 0..<mN { leaves.append(frFromInt(UInt64(i + 1))) }
    let parTree = parallelPoseidon2Merkle(leaves)
    // Manually verify: tree[n+i] = leaves[i], tree[i] = hash(tree[2i], tree[2i+1])
    var merkleOk = true
    for i in 0..<mN {
        if frToInt(parTree[mN + i]) != frToInt(leaves[i]) { merkleOk = false; break }
    }
    // Check internal nodes
    for i in stride(from: mN - 1, through: 1, by: -1) {
        let expected = poseidon2Hash(parTree[2 * i], parTree[2 * i + 1])
        if frToInt(parTree[i]) != frToInt(expected) { merkleOk = false; break }
    }
    check("Poseidon2 Merkle (64 leaves): structure valid", merkleOk)

    // Keccak Merkle
    var kLeaves = [[UInt8]]()
    for i in 0..<mN {
        var leaf = [UInt8](repeating: 0, count: 32)
        leaf[0] = UInt8(i & 0xFF)
        kLeaves.append(leaf)
    }
    let kTree = parallelKeccak256Merkle(kLeaves)
    var kMerkleOk = true
    for i in 0..<mN {
        if kTree[mN + i] != kLeaves[i] { kMerkleOk = false; break }
    }
    for i in stride(from: mN - 1, through: 1, by: -1) {
        let expected = keccak256(kTree[2 * i] + kTree[2 * i + 1])
        if kTree[i] != expected { kMerkleOk = false; break }
    }
    check("Keccak Merkle (64 leaves): structure valid", kMerkleOk)

    // Blake3 Merkle
    let b3Tree = parallelBlake3Merkle(kLeaves)
    var b3MerkleOk = true
    for i in 0..<mN {
        if b3Tree[mN + i] != kLeaves[i] { b3MerkleOk = false; break }
    }
    for i in stride(from: mN - 1, through: 1, by: -1) {
        let expected = blake3(b3Tree[2 * i] + b3Tree[2 * i + 1])
        if b3Tree[i] != expected { b3MerkleOk = false; break }
    }
    check("Blake3 Merkle (64 leaves): structure valid", b3MerkleOk)

    // ===== Parallel MSM full-scalar =====
    print("\n--- ParallelMSM Cross-Checks ---")
    do {
        let gx = fpFromInt(1)
        let gy = fpFromInt(2)
        let gProj = pointFromAffine(PointAffine(x: gx, y: gy))

        // Test 1: Single point, scalar=1
        let r1 = parallelMSM(points: [PointAffine(x: gx, y: gy)],
                              scalars: [[1, 0, 0, 0, 0, 0, 0, 0]])
        check("ParallelMSM([G],[1]) = G", pointEqual(r1, gProj))

        // Test 2: Single point, scalar=5
        let fiveG = pointMulInt(gProj, 5)
        let r2 = parallelMSM(points: [PointAffine(x: gx, y: gy)],
                              scalars: [[5, 0, 0, 0, 0, 0, 0, 0]])
        check("ParallelMSM([G],[5]) = 5G", pointEqual(r2, fiveG))

        // Test 3: 64 points with low scalars, cross-check vs sequential
        let n = 64
        var pts = [PointProjective]()
        var accum = gProj
        for _ in 0..<n {
            pts.append(accum)
            accum = pointAdd(accum, gProj)
        }
        let affPts = batchToAffine(pts)
        var scalars64 = [[UInt32]]()
        for i in 0..<n {
            scalars64.append([UInt32(i + 1), 0, 0, 0, 0, 0, 0, 0])
        }

        // Sequential CPU reference
        var seqResult = pointIdentity()
        for i in 0..<n {
            let scalar = frFromLimbs(scalars64[i])
            seqResult = pointAdd(seqResult, pointScalarMul(pts[i], scalar))
        }
        let parResult = parallelMSM(points: affPts, scalars: scalars64)
        check("ParallelMSM 64 pts (low scalars) = sequential", pointEqual(seqResult, parResult))

        // Test 4: 128 points with random full scalars
        let nFull = 128
        var fullPts = [PointProjective]()
        accum = gProj
        for _ in 0..<nFull {
            fullPts.append(accum)
            accum = pointAdd(accum, gProj)
        }
        let fullAff = batchToAffine(fullPts)
        var rng: UInt64 = 0xABCD_1234
        var fullScalars = [[UInt32]]()
        for _ in 0..<nFull {
            var limbs = [UInt32](repeating: 0, count: 8)
            for j in 0..<8 {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                limbs[j] = UInt32(truncatingIfNeeded: rng >> 32)
            }
            fullScalars.append(limbs)
        }

        // Sequential reference
        var seqFull = pointIdentity()
        for i in 0..<nFull {
            let scalar = frFromLimbs(fullScalars[i])
            seqFull = pointAdd(seqFull, pointScalarMul(fullPts[i], scalar))
        }
        let parFull = parallelMSM(points: fullAff, scalars: fullScalars)
        check("ParallelMSM 128 pts (full scalars) = sequential", pointEqual(seqFull, parFull))

        // Test 5: Result on curve
        let rAff = batchToAffine([parFull])[0]
        let y2 = fpSqr(rAff.y)
        let x3 = fpMul(fpSqr(rAff.x), rAff.x)
        let rhs = fpAdd(x3, fpFromInt(3))  // BN254: y^2 = x^3 + 3
        check("ParallelMSM result on curve", fpToInt(y2) == fpToInt(rhs))
    }

    // ===== Parallel NTT round-trip =====
    print("\n--- ParallelNTT Round-Trip ---")
    let nttN = 1024
    let nttLogN = 10
    var nttInput = [Fr](repeating: Fr.zero, count: nttN)
    for i in 0..<nttN { nttInput[i] = frFromInt(UInt64(i + 1)) }

    let fwd = parallelNTT_Fr(nttInput, logN: nttLogN)
    let recovered = parallelINTT_Fr(fwd, logN: nttLogN)
    var nttOk = true
    for i in 0..<nttN {
        if frToInt(nttInput[i]) != frToInt(recovered[i]) { nttOk = false; break }
    }
    check("ParallelNTT Fr round-trip (2^10)", nttOk)

    // Cross-check against vanilla cpuNTT
    let vanillaNTT = NTTEngine.cpuNTT(nttInput, logN: nttLogN)
    var crossOk = true
    for i in 0..<nttN {
        if frToInt(fwd[i]) != frToInt(vanillaNTT[i]) { crossOk = false; break }
    }
    check("ParallelNTT Fr matches vanilla cpuNTT (2^10)", crossOk)

    // BabyBear round-trip
    var bbInput = [Bb](repeating: Bb.zero, count: nttN)
    for i in 0..<nttN { bbInput[i] = Bb(v: UInt32(i + 1)) }
    let bbFwd = parallelNTT_Bb(bbInput, logN: nttLogN)
    let vanillaBb = BabyBearNTTEngine.cpuNTT(bbInput, logN: nttLogN)
    var bbOk = true
    for i in 0..<nttN { if bbFwd[i].v != vanillaBb[i].v { bbOk = false; break } }
    check("ParallelNTT Bb matches vanilla cpuNTT (2^10)", bbOk)

    // Goldilocks round-trip
    var glInput = [Gl](repeating: Gl.zero, count: nttN)
    for i in 0..<nttN { glInput[i] = Gl(v: UInt64(i + 1)) }
    let glFwd = parallelNTT_Gl(glInput, logN: nttLogN)
    let vanillaGl = GoldilocksNTTEngine.cpuNTT(glInput, logN: nttLogN)
    var glOk = true
    for i in 0..<nttN { if glFwd[i].v != vanillaGl[i].v { glOk = false; break } }
    check("ParallelNTT Gl matches vanilla cpuNTT (2^10)", glOk)

    // ===== Summary =====
    print("\n=== Correctness Summary: \(passed) passed, \(failed) failed ===")
}
