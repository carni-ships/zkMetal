// GPU STARK Verifier Tests — validates GPU-accelerated STARK verification engine
//
// Tests:
//   - AIR constraint evaluation at query points
//   - FRI decommitment verification
//   - Merkle path verification
//   - Invalid proof rejection
//   - Security parameter configuration

import zkMetal

public func runGPUSTARKVerifierTests() {
    suite("GPU STARK Verifier — AIR Constraint Check")
    testAIRConstraintCheck()

    suite("GPU STARK Verifier — FRI Decommitment Verify")
    testFRIDecommitmentVerify()

    suite("GPU STARK Verifier — Merkle Path Verify")
    testMerklePathVerify()

    suite("GPU STARK Verifier — Invalid Proof Rejection")
    testInvalidProofRejection()

    suite("GPU STARK Verifier — Security Parameter Config")
    testSecurityParameterConfig()
}

// MARK: - AIR Constraint Check

func testAIRConstraintCheck() {
    do {
        let engine = try GPUSTARKVerifierEngine(config: .fast)

        // Build a valid Fibonacci-like proof: constraint is next[0] - current[1] == 0
        // current = [a, b], next = [b, a+b]
        let a = frFromInt(5)
        let b = frFromInt(8)
        let aPlusB = frAdd(a, b)

        // Trace evals at OOD point: [a, b]
        let traceEvals = [a, b]
        // Next evals: [b, a+b]
        let traceNextEvals = [b, aPlusB]

        // Composition eval (zero since constraints are satisfied)
        let compositionEval = Fr.zero

        let proof = GPUSTARKVerifierEngine.buildTestProof(
            numColumns: 2,
            traceLength: 8,
            traceEvals: traceEvals,
            traceNextEvals: traceNextEvals,
            compositionEval: compositionEval,
            config: .fast
        )

        // Fibonacci constraint: next[0] = current[1], next[1] = current[0] + current[1]
        let valid = try engine.verify(proof: proof) { current, next -> [Fr] in
            let c0 = frSub(next[0], current[1])
            let c1 = frSub(next[1], frAdd(current[0], current[1]))
            return [c0, c1]
        }
        expect(valid, "Valid Fibonacci AIR constraint should pass verification")
    } catch {
        expect(false, "AIR constraint check failed: \(error)")
    }
}

// MARK: - FRI Decommitment Verify

func testFRIDecommitmentVerify() {
    do {
        let engine = try GPUSTARKVerifierEngine(config: .fast)

        // Build a proof with a trivial FRI structure (no rounds = no decommitments needed)
        let traceEvals = [frFromInt(10), frFromInt(20)]
        let traceNextEvals = [frFromInt(20), frFromInt(30)]
        let compositionEval = Fr.zero

        let proof = GPUSTARKVerifierEngine.buildTestProof(
            numColumns: 2,
            traceLength: 4,
            traceEvals: traceEvals,
            traceNextEvals: traceNextEvals,
            compositionEval: compositionEval,
            config: .fast
        )

        // Trivial constraint: next[0] = current[1], next[1] = current[0] + current[1]
        let valid = try engine.verify(proof: proof) { current, next -> [Fr] in
            let c0 = frSub(next[0], current[1])
            let c1 = frSub(next[1], frAdd(current[0], current[1]))
            return [c0, c1]
        }
        expect(valid, "Proof with trivial FRI should verify (no rounds to check)")

        // Verify FRI proof structure
        expect(proof.friProof.layerCommitments.isEmpty,
               "Trivial FRI should have no layer commitments")
        expect(proof.friProof.finalPoly.count == 1,
               "Final polynomial should be a constant")
    } catch {
        expect(false, "FRI decommitment test failed: \(error)")
    }
}

// MARK: - Merkle Path Verify

func testMerklePathVerify() {
    do {
        let engine = try GPUSTARKVerifierEngine(config: .fast)

        // Test 1: trivial path (leaf == root, no siblings)
        let leaf = frFromInt(42)
        let trivialPath = FrMerklePath(leafIndex: 0, leaf: leaf, siblings: [], root: leaf)
        let valid = engine.verifyMerklePath(trivialPath, expectedRoot: leaf)
        expect(valid, "Trivial Merkle path (no siblings) should verify")

        // Test 2: wrong root should fail
        let wrongRoot = frFromInt(999)
        let invalidPath = FrMerklePath(leafIndex: 0, leaf: leaf, siblings: [], root: wrongRoot)
        let invalid = engine.verifyMerklePath(invalidPath, expectedRoot: wrongRoot)
        // With no siblings, leaf must equal root
        let shouldFail = !engine.verifyMerklePath(invalidPath, expectedRoot: leaf)
        expect(shouldFail, "Wrong root should fail Merkle verification")

        // Test 3: single-level path
        let left = frFromInt(10)
        let right = frFromInt(20)
        // Compute expected root = merkleHash(left, right) = left^2 + 3*right + 7
        let leftSq = frMul(left, left)           // 100
        let threeR = frMul(frFromInt(3), right)   // 60
        let expectedRoot = frAdd(frAdd(leftSq, threeR), frFromInt(7)) // 167

        let leftPath = FrMerklePath(leafIndex: 0, leaf: left, siblings: [right], root: expectedRoot)
        let leftValid = engine.verifyMerklePath(leftPath, expectedRoot: expectedRoot)
        expect(leftValid, "Left child Merkle path should verify")

        // Right child at index 1
        let rightRoot = frAdd(frAdd(frMul(right, right), frMul(frFromInt(3), left)), frFromInt(7))
        let rightPath = FrMerklePath(leafIndex: 1, leaf: right, siblings: [left], root: rightRoot)
        let rightValid = engine.verifyMerklePath(rightPath, expectedRoot: rightRoot)
        expect(rightValid, "Right child Merkle path should verify")

        // Test 4: batch verification
        let paths = [leftPath, trivialPath]
        let batchResults = engine.batchVerifyMerklePaths(paths, expectedRoot: expectedRoot)
        expect(batchResults[0], "Batch: first path should verify against expectedRoot")
        expect(!batchResults[1], "Batch: trivial path should fail against different root")
    } catch {
        expect(false, "Merkle path test failed: \(error)")
    }
}

// MARK: - Invalid Proof Rejection

func testInvalidProofRejection() {
    do {
        let engine = try GPUSTARKVerifierEngine(config: .fast)

        // Test 1: Wrong AIR constraint (evaluates to non-zero)
        let a = frFromInt(5)
        let b = frFromInt(8)

        let proof = GPUSTARKVerifierEngine.buildTestProof(
            numColumns: 2,
            traceLength: 8,
            traceEvals: [a, b],
            traceNextEvals: [b, frAdd(a, b)],
            compositionEval: Fr.zero,
            config: .fast
        )

        // Wrong constraint: expects next[0] = current[0] (not current[1])
        do {
            _ = try engine.verify(proof: proof) { current, next -> [Fr] in
                let c0 = frSub(next[0], current[0])  // Wrong! Should be current[1]
                return [c0]
            }
            expect(false, "Wrong constraint should be rejected")
        } catch {
            expect(true, "Wrong constraint correctly rejected: \(error)")
        }

        // Test 2: Wrong number of columns in OOD evals
        let badProof = FrSTARKProof(
            traceCommitment: frFromInt(1),
            compositionCommitment: frFromInt(2),
            oodPoint: frFromInt(3),
            oodTraceEvals: [a],          // Only 1 eval
            oodTraceNextEvals: [b, a],   // But 2 next evals
            oodCompositionEval: Fr.zero,
            deepCompositionEval: Fr.zero,
            friProof: FrFRIProof(layerCommitments: [], foldingChallenges: [],
                                 decommitments: [], finalPoly: [Fr.one]),
            traceDecommitments: [],
            compositionDecommitments: [],
            numColumns: 2,               // Says 2 columns
            traceLength: 8
        )

        do {
            _ = try engine.verify(proof: badProof) { _, _ in [Fr.zero] }
            expect(false, "Mismatched column count should be rejected")
        } catch {
            expect(true, "Mismatched column count correctly rejected")
        }

        // Test 3: Non-power-of-2 trace length
        let badTraceLenProof = FrSTARKProof(
            traceCommitment: frFromInt(1),
            compositionCommitment: frFromInt(2),
            oodPoint: frFromInt(3),
            oodTraceEvals: [a, b],
            oodTraceNextEvals: [b, frAdd(a, b)],
            oodCompositionEval: Fr.zero,
            deepCompositionEval: Fr.zero,
            friProof: FrFRIProof(layerCommitments: [], foldingChallenges: [],
                                 decommitments: [], finalPoly: [Fr.one]),
            traceDecommitments: [],
            compositionDecommitments: [],
            numColumns: 2,
            traceLength: 7   // Not power of 2!
        )

        do {
            _ = try engine.verify(proof: badTraceLenProof) { _, _ in [Fr.zero] }
            expect(false, "Non-power-of-2 trace length should be rejected")
        } catch {
            expect(true, "Non-power-of-2 trace length correctly rejected")
        }

        // Test 4: Deep composition mismatch
        let deepMismatchProof = GPUSTARKVerifierEngine.buildTestProof(
            numColumns: 2,
            traceLength: 8,
            traceEvals: [a, b],
            traceNextEvals: [b, frAdd(a, b)],
            compositionEval: Fr.zero,
            config: .fast
        )
        // Tamper with deep composition eval
        let tamperedProof = FrSTARKProof(
            traceCommitment: deepMismatchProof.traceCommitment,
            compositionCommitment: deepMismatchProof.compositionCommitment,
            oodPoint: deepMismatchProof.oodPoint,
            oodTraceEvals: deepMismatchProof.oodTraceEvals,
            oodTraceNextEvals: deepMismatchProof.oodTraceNextEvals,
            oodCompositionEval: deepMismatchProof.oodCompositionEval,
            deepCompositionEval: frFromInt(12345),  // Tampered!
            friProof: deepMismatchProof.friProof,
            traceDecommitments: deepMismatchProof.traceDecommitments,
            compositionDecommitments: deepMismatchProof.compositionDecommitments,
            numColumns: deepMismatchProof.numColumns,
            traceLength: deepMismatchProof.traceLength
        )

        do {
            _ = try engine.verify(proof: tamperedProof) { current, next -> [Fr] in
                let c0 = frSub(next[0], current[1])
                let c1 = frSub(next[1], frAdd(current[0], current[1]))
                return [c0, c1]
            }
            expect(false, "Tampered deep composition should be rejected")
        } catch {
            expect(true, "Tampered deep composition correctly rejected")
        }
    } catch {
        expect(false, "Invalid proof rejection test failed: \(error)")
    }
}

// MARK: - Security Parameter Configuration

func testSecurityParameterConfig() {
    // Test fast config
    let fast = STARKSecurityConfig.fast
    expectEqual(fast.numQueries, 8, "Fast: 8 queries")
    expectEqual(fast.blowupFactor, 4, "Fast: blowup 4")
    expectEqual(fast.logBlowup, 2, "Fast: log blowup 2")
    expectEqual(fast.securityBits, 16, "Fast: 16 security bits")
    expectEqual(fast.numFRIRounds, 3, "Fast: 3 FRI rounds")
    expectEqual(fast.maxFinalDegree, 4, "Fast: max final degree 4")

    // Test standard config
    let standard = STARKSecurityConfig.standard
    expectEqual(standard.numQueries, 32, "Standard: 32 queries")
    expectEqual(standard.blowupFactor, 8, "Standard: blowup 8")
    expectEqual(standard.logBlowup, 3, "Standard: log blowup 3")
    expectEqual(standard.securityBits, 96, "Standard: 96 security bits")

    // Test high config
    let high = STARKSecurityConfig.high
    expectEqual(high.numQueries, 64, "High: 64 queries")
    expectEqual(high.blowupFactor, 16, "High: blowup 16")
    expectEqual(high.logBlowup, 4, "High: log blowup 4")
    expectEqual(high.securityBits, 256, "High: 256 security bits")

    // Test custom config
    let custom = STARKSecurityConfig(
        numQueries: 16, blowupFactor: 8, numFRIRounds: 5, maxFinalDegree: 2)
    expectEqual(custom.numQueries, 16, "Custom: 16 queries")
    expectEqual(custom.blowupFactor, 8, "Custom: blowup 8")
    expectEqual(custom.securityBits, 48, "Custom: 48 security bits")
    expectEqual(custom.numFRIRounds, 5, "Custom: 5 FRI rounds")
    expectEqual(custom.maxFinalDegree, 2, "Custom: max final degree 2")

    // Test that engine initializes with each config
    do {
        let _ = try GPUSTARKVerifierEngine(config: .fast)
        expect(true, "Engine initializes with fast config")
        let _ = try GPUSTARKVerifierEngine(config: .standard)
        expect(true, "Engine initializes with standard config")
        let _ = try GPUSTARKVerifierEngine(config: .high)
        expect(true, "Engine initializes with high config")
    } catch {
        expect(false, "Engine initialization failed: \(error)")
    }
}
