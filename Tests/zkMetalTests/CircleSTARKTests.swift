// Circle STARK Tests — end-to-end prove + verify round-trips
// Tests: Fibonacci AIR, Range Check AIR, invalid trace rejection

import zkMetal

func runCircleSTARKTests() {
    suite("Circle STARK — Field Extensions")
    testQM31Arithmetic()

    suite("Circle STARK — AIR Verification")
    testFibonacciAIRTrace()
    testRangeCheckAIRTrace()

    suite("Circle STARK — Prove + Verify")
    testFibonacciProveVerify()
    testFibonacciProveVerifyGPU()
    testRangeCheckProveVerify()
    testGenericAIRProveVerify()

    suite("Circle STARK — Soundness")
    testInvalidTraceRejected()
    testTamperedProofRejected()

    suite("Circle STARK — Config + Proof")
    testConfigSecurityBits()
    testProofSerialization()
}

// MARK: - QM31 Extension Field Tests

func testQM31Arithmetic() {
    // QM31 addition
    let a = QM31(M31(v: 1), M31(v: 2), M31(v: 3), M31(v: 4))
    let b = QM31(M31(v: 5), M31(v: 6), M31(v: 7), M31(v: 8))
    let sum = qm31Add(a, b)
    expectEqual(sum.a.a.v, UInt32(6), "QM31 add c0")
    expectEqual(sum.a.b.v, UInt32(8), "QM31 add c1")
    expectEqual(sum.b.a.v, UInt32(10), "QM31 add c2")
    expectEqual(sum.b.b.v, UInt32(12), "QM31 add c3")

    // QM31 subtraction
    let diff = qm31Sub(b, a)
    expectEqual(diff.a.a.v, UInt32(4), "QM31 sub c0")
    expectEqual(diff.a.b.v, UInt32(4), "QM31 sub c1")

    // QM31 multiplication: 1 * x = x
    let one = QM31.one
    let prod1 = qm31Mul(one, a)
    expectEqual(prod1.a.a.v, a.a.a.v, "QM31 mul identity c0")
    expectEqual(prod1.a.b.v, a.a.b.v, "QM31 mul identity c1")
    expectEqual(prod1.b.a.v, a.b.a.v, "QM31 mul identity c2")
    expectEqual(prod1.b.b.v, a.b.b.v, "QM31 mul identity c3")

    // QM31 inverse: a * a^-1 = 1
    let aInv = qm31Inverse(a)
    let shouldBeOne = qm31Mul(a, aInv)
    expectEqual(shouldBeOne.a.a.v, UInt32(1), "QM31 inverse real")
    expectEqual(shouldBeOne.a.b.v, UInt32(0), "QM31 inverse imag0")
    expectEqual(shouldBeOne.b.a.v, UInt32(0), "QM31 inverse imag1")
    expectEqual(shouldBeOne.b.b.v, UInt32(0), "QM31 inverse imag2")

    // QM31 scalar multiply
    let scaled = qm31ScalarMul(a, M31(v: 3))
    expectEqual(scaled.a.a.v, UInt32(3), "QM31 scalar mul c0")
    expectEqual(scaled.a.b.v, UInt32(6), "QM31 scalar mul c1")

    // QM31 from M31
    let lifted = QM31.from(M31(v: 42))
    expectEqual(lifted.a.a.v, UInt32(42), "QM31 from M31")
    expect(lifted.b.isZero, "QM31 from M31 upper zero")
}

// MARK: - AIR Trace Verification

func testFibonacciAIRTrace() {
    let air = FibonacciAIR(logTraceLength: 4)  // 16 rows
    let trace = air.generateTrace()

    expectEqual(trace.count, 2, "Fib trace 2 columns")
    expectEqual(trace[0].count, 16, "Fib trace 16 rows")

    // Check first values
    expectEqual(trace[0][0].v, UInt32(1), "Fib a[0] = 1")
    expectEqual(trace[1][0].v, UInt32(1), "Fib b[0] = 1")

    // Check Fibonacci relation: a[i+1] = b[i], b[i+1] = a[i] + b[i]
    for i in 0..<15 {
        expectEqual(trace[0][i + 1].v, trace[1][i].v, "Fib a[\(i+1)] = b[\(i)]")
        expectEqual(trace[1][i + 1].v, m31Add(trace[0][i], trace[1][i]).v,
                    "Fib b[\(i+1)] = a[\(i)] + b[\(i)]")
    }

    // Verify trace satisfies AIR
    let err = air.verifyTrace(trace)
    expect(err == nil, "Fib trace valid: \(err ?? "")")
}

func testRangeCheckAIRTrace() {
    let values: [M31] = [M31(v: 100), M31(v: 50), M31(v: 200), M31(v: 0),
                         M31(v: 1000), M31(v: 500), M31(v: 65535), M31(v: 300)]
    let air = RangeCheckAIR(logTraceLength: 3, values: values, bound: 65536)
    let trace = air.generateTrace()

    expectEqual(trace.count, 1, "Range check 1 column")
    expectEqual(trace[0].count, 8, "Range check 8 rows")

    // Trace should be sorted
    for i in 0..<7 {
        expect(trace[0][i].v <= trace[0][i + 1].v,
               "Range check sorted: \(trace[0][i].v) <= \(trace[0][i+1].v)")
    }

    // All values should be < bound
    for i in 0..<8 {
        expect(trace[0][i].v < 65536, "Range check value \(i) < 65536")
    }
}

// MARK: - Prove + Verify Round-Trips

func testFibonacciProveVerify() {
    // CPU path (generic, works for any AIR)
    do {
        let air = FibonacciAIR(logTraceLength: 3)  // 8 rows (small for fast test)
        let prover = CircleSTARKProver(logBlowup: 2, numQueries: 5)
        let proof = try prover.proveCPU(air: air)

        expectEqual(proof.traceLength, 8, "Proof trace length")
        expectEqual(proof.numColumns, 2, "Proof num columns")
        expect(proof.queryResponses.count > 0, "Has query responses")
        expect(proof.friProof.rounds.count > 0, "Has FRI rounds")

        // Verify
        let verifier = CircleSTARKVerifier()
        let valid = try verifier.verify(air: air, proof: proof)
        expect(valid, "Fibonacci CPU prove-verify round trip")
    } catch {
        expect(false, "Fibonacci CPU prove error: \(error)")
    }
}

func testFibonacciProveVerifyGPU() {
    // GPU path (specialized for Fibonacci)
    do {
        let air = FibonacciAIR(logTraceLength: 4)  // 16 rows
        let prover = CircleSTARKProver(logBlowup: 2, numQueries: 5)
        let proof = try prover.prove(air: air)

        expectEqual(proof.traceLength, 16, "GPU proof trace length")

        let verifier = CircleSTARKVerifier()
        let valid = try verifier.verify(air: air, proof: proof)
        expect(valid, "Fibonacci GPU prove-verify round trip")
    } catch {
        expect(false, "Fibonacci GPU prove error: \(error)")
    }
}

func testRangeCheckProveVerify() {
    do {
        let values: [M31] = (0..<8).map { M31(v: UInt32($0) * 100) }
        let air = RangeCheckAIR(logTraceLength: 3, values: values, bound: 65536)
        let prover = CircleSTARKProver(logBlowup: 2, numQueries: 5)
        let proof = try prover.proveCPU(air: air)

        expectEqual(proof.numColumns, 1, "Range check 1 column")
        expect(proof.friProof.rounds.count > 0, "Range check FRI rounds")

        let verifier = CircleSTARKVerifier()
        let valid = try verifier.verify(air: air, proof: proof)
        expect(valid, "Range check prove-verify round trip")
    } catch {
        expect(false, "Range check prove error: \(error)")
    }
}

func testGenericAIRProveVerify() {
    // Simple doubling AIR: a[i+1] = 2 * a[i], with a[0] = 1
    do {
        let logN = 3
        let n = 1 << logN
        let air = GenericAIR(
            numColumns: 1,
            logTraceLength: logN,
            constraintDegrees: [1],
            boundaryConstraints: [(0, 0, M31.one)],
            generateTrace: {
                var col = [M31](repeating: M31.zero, count: n)
                col[0] = M31.one
                for i in 1..<n {
                    col[i] = m31Add(col[i - 1], col[i - 1])  // 2x
                }
                return [col]
            },
            evaluateConstraints: { current, next in
                // next[0] - 2 * current[0] = 0
                let doubled = m31Add(current[0], current[0])
                return [m31Sub(next[0], doubled)]
            }
        )

        // Verify trace is valid
        let trace = air.generateTrace()
        let err = air.verifyTrace(trace)
        expect(err == nil, "Doubling trace valid: \(err ?? "")")

        // Prove and verify
        let prover = CircleSTARKProver(logBlowup: 2, numQueries: 5)
        let proof = try prover.proveCPU(air: air)

        let verifier = CircleSTARKVerifier()
        let valid = try verifier.verify(air: air, proof: proof)
        expect(valid, "Generic doubling AIR prove-verify")
    } catch {
        expect(false, "Generic AIR error: \(error)")
    }
}

// MARK: - Soundness Tests

func testInvalidTraceRejected() {
    // Create a Fibonacci AIR with wrong initial values
    let air = FibonacciAIR(logTraceLength: 3, a0: M31.one, b0: M31.one)
    let trace = air.generateTrace()

    // Tamper with a value
    var badTrace = trace
    badTrace[0][2] = M31(v: 999)  // corrupt a[2]

    let err = air.verifyTrace(badTrace)
    expect(err != nil, "Tampered Fibonacci trace rejected: \(err ?? "nil")")

    // Verify that a trace with wrong boundary is caught
    var badBoundary = trace
    badBoundary[0][0] = M31(v: 42)  // wrong initial value
    let err2 = air.verifyTrace(badBoundary)
    expect(err2 != nil, "Wrong boundary Fibonacci trace rejected: \(err2 ?? "nil")")
}

func testTamperedProofRejected() {
    do {
        let air = FibonacciAIR(logTraceLength: 3)
        let prover = CircleSTARKProver(logBlowup: 2, numQueries: 5)
        let proof = try prover.proveCPU(air: air)

        // Tamper with alpha
        let tamperedProof = CircleSTARKProof(
            traceCommitments: proof.traceCommitments,
            compositionCommitment: proof.compositionCommitment,
            friProof: proof.friProof,
            queryResponses: proof.queryResponses,
            alpha: M31(v: 12345),  // wrong alpha
            traceLength: proof.traceLength,
            numColumns: proof.numColumns,
            logBlowup: proof.logBlowup
        )

        let verifier = CircleSTARKVerifier()
        var rejected = false
        do {
            _ = try verifier.verify(air: air, proof: tamperedProof)
        } catch {
            rejected = true
        }
        expect(rejected, "Tampered proof (wrong alpha) rejected")
    } catch {
        expect(false, "Proof generation error: \(error)")
    }
}

// MARK: - Config + Proof Tests

func testConfigSecurityBits() {
    let cfg = CircleSTARKConfig.default
    expectEqual(cfg.logBlowup, 4, "Default logBlowup")
    expectEqual(cfg.numQueries, 30, "Default numQueries")
    expectEqual(cfg.extensionDegree, 4, "Default extension degree")
    expectEqual(cfg.blowupFactor, 16, "Default blowup factor")
    expectEqual(cfg.securityBits, 120, "Default security bits: 30 * 4 = 120")

    let fast = CircleSTARKConfig.fast
    expectEqual(fast.securityBits, 20, "Fast security bits: 10 * 2 = 20")

    let high = CircleSTARKConfig.highSecurity
    expectEqual(high.securityBits, 208, "High security bits: 50 * 4 + 8 = 208")
}

func testProofSerialization() {
    do {
        let air = FibonacciAIR(logTraceLength: 3)
        let prover = CircleSTARKProver(logBlowup: 2, numQueries: 5)
        let proof = try prover.proveCPU(air: air)

        let bytes = proof.serialize()
        expect(bytes.count > 0, "Serialized proof non-empty")

        // Check magic header
        expectEqual(bytes[0], 0x43, "Magic C")
        expectEqual(bytes[1], 0x53, "Magic S")
        expectEqual(bytes[2], 0x54, "Magic T")
        expectEqual(bytes[3], 0x4B, "Magic K")

        // Check size description
        let desc = proof.proofSizeDescription
        expect(desc.count > 0, "Proof size description: \(desc)")

        let size = proof.estimatedSizeBytes
        expect(size > 0, "Proof size > 0: \(size) bytes")
    } catch {
        expect(false, "Serialization test error: \(error)")
    }
}
