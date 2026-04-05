import zkMetal

func runBatchKZGTests() {
    // Common setup: test SRS
    let gen = PointAffine(x: fpFromInt(1), y: fpFromInt(2))
    let secret: [UInt32] = [0xCAFE, 0xBEEF, 0xDEAD, 0x1234, 0x5678, 0x9ABC, 0xDEF0, 0x0001]
    let srsSecret = frFromLimbs(secret)
    let srs = KZGEngine.generateTestSRS(secret: secret, size: 256, generator: gen)

    suite("Shplonk Batch KZG Opening")
    do {
        let kzg = try KZGEngine(srs: srs)
        let shplonk = ShplonkBatchOpener(kzg: kzg)

        // Test 1: Single polynomial, single point (degenerates to plain KZG)
        do {
            let poly: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
            let commitment = try kzg.commit(poly)
            let z = frFromInt(7)

            let transcript1 = Transcript(label: "shplonk-test-1", backend: .poseidon2)
            let proof = try shplonk.batchOpen(
                commitments: [commitment],
                polynomials: [poly],
                points: [z],
                transcript: transcript1
            )
            expect(proof.claims.count == 1, "Single poly: 1 claim")
            expect(!pointIsIdentity(proof.witness), "Single poly: non-trivial witness")

            // Verify
            let transcript2 = Transcript(label: "shplonk-test-1", backend: .poseidon2)
            let valid = shplonk.batchVerify(
                proof: proof,
                points: [z],
                transcript: transcript2,
                srsSecret: srsSecret
            )
            expect(valid, "Single poly single point: verify")
        }

        // Test 2: Multiple polynomials at same point (Plonk-like)
        do {
            let polys: [[Fr]] = [
                [frFromInt(1), frFromInt(2), frFromInt(3)],
                [frFromInt(4), frFromInt(5), frFromInt(6)],
                [frFromInt(7), frFromInt(8), frFromInt(9), frFromInt(10)],
            ]
            var commitments = [PointProjective]()
            for poly in polys {
                commitments.append(try kzg.commit(poly))
            }
            let z = frFromInt(42)

            let transcript1 = Transcript(label: "shplonk-test-2", backend: .poseidon2)
            let proof = try shplonk.batchOpen(
                commitments: commitments,
                polynomials: polys,
                points: [z],
                transcript: transcript1
            )
            expect(proof.claims.count == 3, "3 polys same point: 3 claims")

            let transcript2 = Transcript(label: "shplonk-test-2", backend: .poseidon2)
            let valid = shplonk.batchVerify(
                proof: proof,
                points: [z],
                transcript: transcript2,
                srsSecret: srsSecret
            )
            expect(valid, "3 polys same point: verify")
        }

        // Test 3: Multiple polynomials at multiple points (full Plonk scenario)
        do {
            let polys: [[Fr]] = [
                [frFromInt(2), frFromInt(3), frFromInt(5), frFromInt(7)],
                [frFromInt(11), frFromInt(13), frFromInt(17)],
                [frFromInt(19), frFromInt(23)],
            ]
            var commitments = [PointProjective]()
            for poly in polys {
                commitments.append(try kzg.commit(poly))
            }
            let z0 = frFromInt(100)
            let z1 = frFromInt(200)

            // All polys at z0, poly 0 and 1 also at z1
            let openingSets: [Int: [Int]] = [0: [0, 1, 2], 1: [0, 1]]

            let transcript1 = Transcript(label: "shplonk-test-3", backend: .poseidon2)
            let proof = try shplonk.batchOpen(
                commitments: commitments,
                polynomials: polys,
                points: [z0, z1],
                openingSets: openingSets,
                transcript: transcript1
            )
            expect(proof.claims.count == 5, "Multi-point: 5 claims (3+2)")

            let transcript2 = Transcript(label: "shplonk-test-3", backend: .poseidon2)
            let valid = shplonk.batchVerify(
                proof: proof,
                points: [z0, z1],
                openingSets: openingSets,
                transcript: transcript2,
                srsSecret: srsSecret
            )
            expect(valid, "Multi-point multi-poly: verify")
        }

        // Test 4: Reject tampered evaluation
        do {
            let poly: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3)]
            let commitment = try kzg.commit(poly)
            let z = frFromInt(5)

            let transcript1 = Transcript(label: "shplonk-test-4", backend: .poseidon2)
            let proof = try shplonk.batchOpen(
                commitments: [commitment],
                polynomials: [poly],
                points: [z],
                transcript: transcript1
            )

            // Tamper: modify the evaluation in the claim
            let tampered = BatchOpeningProof(
                witness: proof.witness,
                claims: [OpeningClaim(
                    polynomialIndex: 0,
                    point: z,
                    evaluation: frFromInt(999)
                )],
                commitments: [commitment]
            )
            let transcript2 = Transcript(label: "shplonk-test-4", backend: .poseidon2)
            let valid = shplonk.batchVerify(
                proof: tampered,
                points: [z],
                transcript: transcript2,
                srsSecret: srsSecret
            )
            expect(!valid, "Tampered evaluation: reject")
        }

        // Test 5: Larger batch (simulating Plonk with ~10 polynomials at 2 points)
        do {
            let numPolys = 10
            var polys = [[Fr]]()
            var commitments = [PointProjective]()
            for i in 0..<numPolys {
                let poly: [Fr] = (0..<16).map { j in frFromInt(UInt64(i * 16 + j + 1)) }
                polys.append(poly)
                commitments.append(try kzg.commit(poly))
            }
            let zeta = frFromInt(12345)
            let zetaOmega = frFromInt(67890)

            // All polys at zeta, first 3 at zetaOmega
            let openingSets: [Int: [Int]] = [
                0: Array(0..<numPolys),
                1: [0, 1, 2]
            ]

            let transcript1 = Transcript(label: "shplonk-plonk", backend: .poseidon2)
            let proof = try shplonk.batchOpen(
                commitments: commitments,
                polynomials: polys,
                points: [zeta, zetaOmega],
                openingSets: openingSets,
                transcript: transcript1
            )
            expect(proof.claims.count == 13, "Plonk-like: 13 claims (10+3)")

            let transcript2 = Transcript(label: "shplonk-plonk", backend: .poseidon2)
            let valid = shplonk.batchVerify(
                proof: proof,
                points: [zeta, zetaOmega],
                openingSets: openingSets,
                transcript: transcript2,
                srsSecret: srsSecret
            )
            expect(valid, "Plonk-like 10 polys at 2 points: verify")
        }

    } catch { expect(false, "Shplonk error: \(error)") }

    suite("Gemini Multilinear Opening")
    do {
        let kzg = try KZGEngine(srs: srs)
        let gemini = GeminiOpener(kzg: kzg)

        // Test 1: Small MLE (n=2, 4 evaluations)
        do {
            // f(x1, x2) with evaluations on {0,1}^2:
            // f(0,0)=1, f(1,0)=2, f(0,1)=3, f(1,1)=4
            let evals: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
            let point: [Fr] = [frFromInt(5), frFromInt(7)]

            // Verify MLE evaluation manually using ZMFold convention:
            // ZMFold folds from variable n-1 down to 0.
            // point = [u0, u1], evals = [1, 2, 3, 4]
            // Step 1 (k=1): fold with u1=7: folded[i] = evals[2i] + u1 * evals[2i+1]
            //   folded[0] = 1 + 7*2 = 15, folded[1] = 3 + 7*4 = 31
            // Step 2 (k=0): fold with u0=5: result = folded[0] + u0 * folded[1]
            //   result = 15 + 5*31 = 170
            let mleValue = GeminiOpener.evaluateMLE(evals, at: point)
            let u0 = frFromInt(5)
            let u1 = frFromInt(7)
            // Manual: evals[0] + u1*evals[1] + u0*(evals[2] + u1*evals[3])
            // = 1 + 7*2 + 5*(3 + 7*4) = 1 + 14 + 5*31 = 15 + 155 = 170
            let fold1_0 = frAdd(frFromInt(1), frMul(u1, frFromInt(2)))  // 15
            let fold1_1 = frAdd(frFromInt(3), frMul(u1, frFromInt(4)))  // 31
            let manual = frAdd(fold1_0, frMul(u0, fold1_1))  // 170
            expect(frToInt(mleValue) == frToInt(manual), "MLE eval n=2 correct")

            // Open
            let transcript1 = Transcript(label: "gemini-test-1", backend: .poseidon2)
            let proof = try gemini.geminiOpen(
                multilinearPoly: evals,
                point: point,
                transcript: transcript1
            )
            expect(proof.foldCommitments.count == 1, "n=2: 1 fold commitment")
            expect(proof.evaluationsAtR.count == 2, "n=2: 2 evals at r")
            expect(proof.evaluationsAtNegR.count == 2, "n=2: 2 evals at -r")
            expect(frToInt(proof.claimedValue) == frToInt(mleValue), "n=2: claimed value matches")

            // Verify
            let commitment = try kzg.commit(evals)
            let transcript2 = Transcript(label: "gemini-test-1", backend: .poseidon2)
            let valid = gemini.geminiVerify(
                commitment: commitment,
                point: point,
                evaluation: mleValue,
                proof: proof,
                transcript: transcript2,
                srsSecret: srsSecret
            )
            expect(valid, "Gemini n=2: verify")
        }

        // Test 2: Larger MLE (n=4, 16 evaluations)
        do {
            let n = 4
            let N = 1 << n
            let evals: [Fr] = (0..<N).map { frFromInt(UInt64($0 + 1)) }
            let point: [Fr] = [frFromInt(3), frFromInt(5), frFromInt(7), frFromInt(11)]

            let mleValue = GeminiOpener.evaluateMLE(evals, at: point)

            let transcript1 = Transcript(label: "gemini-test-2", backend: .poseidon2)
            let proof = try gemini.geminiOpen(
                multilinearPoly: evals,
                point: point,
                transcript: transcript1
            )
            expect(proof.foldCommitments.count == n - 1, "n=4: 3 fold commitments")
            expect(frToInt(proof.claimedValue) == frToInt(mleValue), "n=4: claimed value matches")

            let commitment = try kzg.commit(evals)
            let transcript2 = Transcript(label: "gemini-test-2", backend: .poseidon2)
            let valid = gemini.geminiVerify(
                commitment: commitment,
                point: point,
                evaluation: mleValue,
                proof: proof,
                transcript: transcript2,
                srsSecret: srsSecret
            )
            expect(valid, "Gemini n=4: verify")
        }

        // Test 3: Reject wrong evaluation claim
        do {
            let evals: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
            let point: [Fr] = [frFromInt(2), frFromInt(3)]
            let mleValue = GeminiOpener.evaluateMLE(evals, at: point)

            let transcript1 = Transcript(label: "gemini-test-3", backend: .poseidon2)
            let proof = try gemini.geminiOpen(
                multilinearPoly: evals,
                point: point,
                transcript: transcript1
            )

            // Try to verify with wrong evaluation
            let commitment = try kzg.commit(evals)
            let wrongEval = frFromInt(999)
            let transcript2 = Transcript(label: "gemini-test-3", backend: .poseidon2)
            let valid = gemini.geminiVerify(
                commitment: commitment,
                point: point,
                evaluation: wrongEval,
                proof: proof,
                transcript: transcript2,
                srsSecret: srsSecret
            )
            expect(!valid, "Gemini: reject wrong evaluation")
        }

        // Test 4: n=6 (64 evaluations) - stress test
        do {
            let n = 6
            let N = 1 << n
            let evals: [Fr] = (0..<N).map { frFromInt(UInt64($0 * 3 + 1)) }
            let point: [Fr] = (0..<n).map { frFromInt(UInt64($0 + 2)) }

            let mleValue = GeminiOpener.evaluateMLE(evals, at: point)

            let transcript1 = Transcript(label: "gemini-test-4", backend: .poseidon2)
            let proof = try gemini.geminiOpen(
                multilinearPoly: evals,
                point: point,
                transcript: transcript1
            )

            let commitment = try kzg.commit(evals)
            let transcript2 = Transcript(label: "gemini-test-4", backend: .poseidon2)
            let valid = gemini.geminiVerify(
                commitment: commitment,
                point: point,
                evaluation: mleValue,
                proof: proof,
                transcript: transcript2,
                srsSecret: srsSecret
            )
            expect(valid, "Gemini n=6: verify")
        }

        // Test 5: Folding relation correctness (unit test for even/odd decomposition)
        do {
            // f(x) = 1 + 2x + 3x^2 + 4x^3
            // f_even(y) = 1 + 3y (coeffs at even indices, variable y = x^2)
            // f_odd(y) = 2 + 4y  (coeffs at odd indices, variable y = x^2)
            // f(x) = f_even(x^2) + x * f_odd(x^2)
            //
            // Fold with u: g(y) = f_even(y) + u * f_odd(y)
            // g(y) at y = r^2 should equal: f_even(r^2) + u * f_odd(r^2)
            //   = (f(r) + f(-r))/2 + u * (f(r) - f(-r))/(2r)

            let u = frFromInt(5)
            let r = frFromInt(7)
            let r2 = frMul(r, r)  // r^2 = 49

            // f(r) = 1 + 2*7 + 3*49 + 4*343 = 1 + 14 + 147 + 1372 = 1534
            let fR = frAdd(frAdd(frFromInt(1), frMul(frFromInt(2), r)),
                           frAdd(frMul(frFromInt(3), frMul(r, r)),
                                 frMul(frFromInt(4), frMul(r, frMul(r, r)))))

            // f(-r)
            let negR = frSub(Fr.zero, r)
            let fNegR = frAdd(frAdd(frFromInt(1), frMul(frFromInt(2), negR)),
                              frAdd(frMul(frFromInt(3), frMul(negR, negR)),
                                    frMul(frFromInt(4), frMul(negR, frMul(negR, negR)))))

            // Recover even/odd at r^2
            let twoInv = frInverse(frFromInt(2))
            let evenAtR2 = frMul(frAdd(fR, fNegR), twoInv)
            let oddAtR2 = frMul(frSub(fR, fNegR), frMul(twoInv, frInverse(r)))

            // Folded value at r^2 via recovery
            let foldedViaRecovery = frAdd(evenAtR2, frMul(u, oddAtR2))

            // Direct: g(y) = (1 + 5*2) + (3 + 5*4)*y = 11 + 23*y
            // g(r^2) = 11 + 23*49 = 11 + 1127 = 1138
            let g0 = frAdd(frFromInt(1), frMul(u, frFromInt(2)))  // 11
            let g1 = frAdd(frFromInt(3), frMul(u, frFromInt(4)))  // 23
            let directFolded = frAdd(g0, frMul(g1, r2))

            expect(frToInt(foldedViaRecovery) == frToInt(directFolded),
                   "Folding relation: even/odd recovery at r^2 matches direct fold")
        }

    } catch { expect(false, "Gemini error: \(error)") }
}
