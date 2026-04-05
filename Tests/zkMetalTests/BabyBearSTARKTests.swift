// BabyBear STARK Tests — end-to-end prove + verify round-trips over BabyBear field
// Tests: Fibonacci AIR, boundary constraints, proof generation/verification, soundness, benchmarks

import zkMetal
import Foundation

public func runBabyBearSTARKTests() {
    suite("BabyBear STARK -- Field Sanity")
    bbStarkTestFieldOps()

    suite("BabyBear STARK -- AIR Trace Validation")
    bbStarkTestFibTrace()
    bbStarkTestFibBoundary()
    bbStarkTestGenericDoubling()

    suite("BabyBear STARK -- Prove + Verify")
    bbStarkTestFibProveVerify()
    bbStarkTestFibProveVerifyLarger()
    bbStarkTestGenericProveVerify()

    suite("BabyBear STARK -- Soundness")
    bbStarkTestTamperedTrace()
    bbStarkTestTamperedProof()
    bbStarkTestWrongBoundary()

    suite("BabyBear STARK -- Config + Metadata")
    bbStarkTestConfigSecurity()
    bbStarkTestProofMetadata()

    suite("BabyBear STARK -- Benchmark")
    bbStarkBenchmark()
}

// MARK: - Field Sanity

func bbStarkTestFieldOps() {
    // Addition
    let a = Bb(v: 100)
    let b = Bb(v: 200)
    let sum = bbAdd(a, b)
    expectEqual(sum.v, UInt32(300), "BabyBear add")

    // Subtraction
    let diff = bbSub(b, a)
    expectEqual(diff.v, UInt32(100), "BabyBear sub")

    // Subtraction wrapping
    let diff2 = bbSub(a, b)
    expectEqual(diff2.v, Bb.P - 100, "BabyBear sub wrap")

    // Multiplication
    let prod = bbMul(Bb(v: 1000), Bb(v: 2000))
    expectEqual(prod.v, UInt32(2000000 % Bb.P), "BabyBear mul")

    // Inverse
    let x = Bb(v: 42)
    let xInv = bbInverse(x)
    let shouldBeOne = bbMul(x, xInv)
    expectEqual(shouldBeOne.v, UInt32(1), "BabyBear inverse")

    // Power
    let p3 = bbPow(Bb(v: 3), 10)
    let expected: UInt32 = 59049  // 3^10
    expectEqual(p3.v, expected, "BabyBear pow")

    // Root of unity
    let omega = bbRootOfUnity(logN: 3)
    let omega8 = bbPow(omega, 8)
    expectEqual(omega8.v, UInt32(1), "BabyBear 8th root of unity order")

    // Root of unity is not trivial
    let omega4 = bbPow(omega, 4)
    expect(omega4.v != 1, "BabyBear 8th root not 4th root")
}

// MARK: - AIR Trace Validation

func bbStarkTestFibTrace() {
    let air = BabyBearFibonacciAIR(logTraceLength: 4)  // 16 rows
    let trace = air.generateTrace()

    expectEqual(trace.count, 2, "Fib trace 2 columns")
    expectEqual(trace[0].count, 16, "Fib trace 16 rows")

    // Check initial values
    expectEqual(trace[0][0].v, UInt32(1), "Fib a[0] = 1")
    expectEqual(trace[1][0].v, UInt32(1), "Fib b[0] = 1")

    // Check Fibonacci relation: a[i+1] = b[i], b[i+1] = a[i] + b[i]
    for i in 0..<15 {
        expectEqual(trace[0][i + 1].v, trace[1][i].v, "Fib a[\(i+1)] = b[\(i)]")
        let expectedB = bbAdd(trace[0][i], trace[1][i])
        expectEqual(trace[1][i + 1].v, expectedB.v,
                    "Fib b[\(i+1)] = a[\(i)] + b[\(i)]")
    }

    // Validate trace satisfies all constraints
    let err = air.verifyTrace(trace)
    expect(err == nil, "Fib trace valid: \(err ?? "")")
}

func bbStarkTestFibBoundary() {
    // Test with custom initial values
    let a0 = Bb(v: 3)
    let b0 = Bb(v: 5)
    let air = BabyBearFibonacciAIR(logTraceLength: 3, a0: a0, b0: b0)
    let trace = air.generateTrace()

    // Check boundary constraints
    expectEqual(trace[0][0].v, UInt32(3), "Custom Fib a[0] = 3")
    expectEqual(trace[1][0].v, UInt32(5), "Custom Fib b[0] = 5")

    // Verify Fibonacci sequence: 3, 5, 8, 13, 21, 34, 55, 89
    expectEqual(trace[0][1].v, UInt32(5), "Custom Fib a[1] = 5")
    expectEqual(trace[1][1].v, UInt32(8), "Custom Fib b[1] = 8")
    expectEqual(trace[0][2].v, UInt32(8), "Custom Fib a[2] = 8")
    expectEqual(trace[1][2].v, UInt32(13), "Custom Fib b[2] = 13")

    // Check boundary constraint definition
    let bcs = air.boundaryConstraints
    expectEqual(bcs.count, 2, "Fib has 2 boundary constraints")
    expectEqual(bcs[0].column, 0, "BC 0 column")
    expectEqual(bcs[0].row, 0, "BC 0 row")
    expectEqual(bcs[0].value.v, UInt32(3), "BC 0 value")
    expectEqual(bcs[1].column, 1, "BC 1 column")
    expectEqual(bcs[1].value.v, UInt32(5), "BC 1 value")

    // Full trace validation
    let err = air.verifyTrace(trace)
    expect(err == nil, "Custom Fib trace valid: \(err ?? "")")
}

func bbStarkTestGenericDoubling() {
    let logN = 3
    let n = 1 << logN

    let air = GenericBabyBearAIR(
        numColumns: 1,
        logTraceLength: logN,
        numConstraints: 1,
        constraintDegree: 1,
        boundaryConstraints: [(column: 0, row: 0, value: Bb.one)],
        traceGenerator: {
            var col = [Bb](repeating: Bb.zero, count: n)
            col[0] = Bb.one
            for i in 1..<n {
                col[i] = bbAdd(col[i - 1], col[i - 1])  // doubling
            }
            return [col]
        },
        constraintEvaluator: { current, next in
            // next[0] = 2 * current[0]
            let doubled = bbAdd(current[0], current[0])
            return [bbSub(next[0], doubled)]
        }
    )

    let trace = air.generateTrace()
    expectEqual(trace[0][0].v, UInt32(1), "Doubling a[0] = 1")
    expectEqual(trace[0][1].v, UInt32(2), "Doubling a[1] = 2")
    expectEqual(trace[0][2].v, UInt32(4), "Doubling a[2] = 4")
    expectEqual(trace[0][3].v, UInt32(8), "Doubling a[3] = 8")

    let err = air.verifyTrace(trace)
    expect(err == nil, "Doubling trace valid: \(err ?? "")")
}

// MARK: - Prove + Verify Round-Trips

func bbStarkTestFibProveVerify() {
    do {
        let air = BabyBearFibonacciAIR(logTraceLength: 3)  // 8 rows
        let stark = BabyBearSTARK(config: .fast)
        let (result, verified) = try stark.proveAndVerify(air: air)

        expect(verified, "Fibonacci prove-verify round trip")
        expectEqual(result.proof.traceLength, 8, "Proof trace length")
        expectEqual(result.proof.numColumns, 2, "Proof num columns")
        expect(result.proof.queryResponses.count > 0, "Has query responses")
        expect(result.proof.friProof.rounds.count > 0, "Has FRI rounds")
        expect(result.proveTimeSeconds > 0, "Prove time recorded")
        expect(result.proofSizeBytes > 0, "Proof has nonzero size")
    } catch {
        expect(false, "Fibonacci prove error: \(error)")
    }
}

func bbStarkTestFibProveVerifyLarger() {
    do {
        let air = BabyBearFibonacciAIR(logTraceLength: 5)  // 32 rows
        let stark = BabyBearSTARK(config: .fast)
        let result = try stark.prove(air: air)
        let valid = try stark.verify(air: air, proof: result.proof)

        expect(valid, "Fibonacci 32-row prove-verify")
        expectEqual(result.proof.traceLength, 32, "32-row trace length")
    } catch {
        expect(false, "Fibonacci 32-row error: \(error)")
    }
}

func bbStarkTestGenericProveVerify() {
    do {
        let logN = 3
        let n = 1 << logN

        // Simple tripling AIR: a[i+1] = 3 * a[i] mod p
        let air = GenericBabyBearAIR(
            numColumns: 1,
            logTraceLength: logN,
            numConstraints: 1,
            constraintDegree: 1,
            boundaryConstraints: [(column: 0, row: 0, value: Bb.one)],
            traceGenerator: {
                var col = [Bb](repeating: Bb.zero, count: n)
                col[0] = Bb.one
                for i in 1..<n {
                    col[i] = bbMul(col[i - 1], Bb(v: 3))
                }
                return [col]
            },
            constraintEvaluator: { current, next in
                let tripled = bbMul(current[0], Bb(v: 3))
                return [bbSub(next[0], tripled)]
            }
        )

        // Verify trace first
        let err = air.verifyTrace(air.generateTrace())
        expect(err == nil, "Tripling trace valid: \(err ?? "")")

        let stark = BabyBearSTARK(config: .fast)
        let (result, verified) = try stark.proveAndVerify(air: air)
        expect(verified, "Generic tripling AIR prove-verify")
        expect(result.proofSizeBytes > 0, "Tripling proof has size")
    } catch {
        expect(false, "Generic AIR prove error: \(error)")
    }
}

// MARK: - Soundness Tests

func bbStarkTestTamperedTrace() {
    let air = BabyBearFibonacciAIR(logTraceLength: 3)
    let trace = air.generateTrace()

    // Tamper with a value in the middle
    var badTrace = trace
    badTrace[0][2] = Bb(v: 999)  // corrupt a[2]

    let err = air.verifyTrace(badTrace)
    expect(err != nil, "Tampered Fibonacci trace rejected: \(err ?? "nil")")
}

func bbStarkTestTamperedProof() {
    do {
        let air = BabyBearFibonacciAIR(logTraceLength: 3)
        let stark = BabyBearSTARK(config: .fast)
        let result = try stark.prove(air: air)

        // Tamper with alpha in the proof
        let tamperedProof = BabyBearSTARKProof(
            traceCommitments: result.proof.traceCommitments,
            compositionCommitment: result.proof.compositionCommitment,
            friProof: result.proof.friProof,
            queryResponses: result.proof.queryResponses,
            alpha: Bb(v: 12345),  // wrong alpha
            traceLength: result.proof.traceLength,
            numColumns: result.proof.numColumns,
            logBlowup: result.proof.logBlowup
        )

        var rejected = false
        do {
            _ = try stark.verify(air: air, proof: tamperedProof)
        } catch {
            rejected = true
        }
        expect(rejected, "Tampered proof (wrong alpha) rejected")
    } catch {
        expect(false, "Tampered proof test setup error: \(error)")
    }
}

func bbStarkTestWrongBoundary() {
    // Create a valid trace for (a0=1, b0=1) but check against (a0=2, b0=1) AIR
    let air1 = BabyBearFibonacciAIR(logTraceLength: 3, a0: Bb.one, b0: Bb.one)
    let trace1 = air1.generateTrace()

    // Verify against a different AIR with different boundary
    let air2 = BabyBearFibonacciAIR(logTraceLength: 3, a0: Bb(v: 2), b0: Bb.one)

    // The trace from air1 should fail boundary check for air2
    let err = air2.verifyTrace(trace1)
    expect(err != nil, "Wrong boundary trace rejected: \(err ?? "nil")")
}

// MARK: - Config + Metadata

func bbStarkTestConfigSecurity() {
    let fast = BabyBearSTARKConfig.fast
    expect(fast.securityBits > 0, "Fast config has security bits: \(fast.securityBits)")
    expect(fast.blowupFactor >= 2, "Fast config blowup >= 2")

    let sp1 = BabyBearSTARKConfig.sp1Default
    expect(sp1.securityBits >= 100, "SP1 config >= 100-bit security: \(sp1.securityBits)")
    expect(sp1.numQueries >= 80, "SP1 config >= 80 queries")

    let high = BabyBearSTARKConfig.highSecurity
    expect(high.securityBits >= 150, "High security >= 150 bits: \(high.securityBits)")
}

func bbStarkTestProofMetadata() {
    do {
        let air = BabyBearFibonacciAIR(logTraceLength: 3)
        let stark = BabyBearSTARK(config: .fast)
        let result = try stark.prove(air: air)

        expectEqual(result.traceLength, 8, "Result trace length")
        expectEqual(result.numColumns, 2, "Result num columns")
        expectEqual(result.numConstraints, 2, "Result num constraints")
        expect(result.proofSizeBytes > 0, "Proof size > 0")
        expect(result.proveTimeSeconds > 0, "Prove time > 0")

        // Check summary string is non-empty
        let summary = result.summary
        expect(summary.count > 0, "Summary non-empty")
        expect(summary.contains("BabyBear STARK"), "Summary contains header")
    } catch {
        expect(false, "Proof metadata error: \(error)")
    }
}

// MARK: - Benchmark

func bbStarkBenchmark() {
    do {
        let air = BabyBearFibonacciAIR(logTraceLength: 5)  // 32 rows
        let stark = BabyBearSTARK(config: .fast)

        // Warm up
        _ = try stark.prove(air: air)

        // Timed run
        let t0 = CFAbsoluteTimeGetCurrent()
        let result = try stark.prove(air: air)
        let elapsed = CFAbsoluteTimeGetCurrent() - t0

        print(String(format: "  BabyBear STARK Fibonacci (32 rows): %.3fms, proof %d bytes",
                      elapsed * 1000, result.proofSizeBytes))
        expect(elapsed < 30.0, "Prove completes in < 30s")
    } catch {
        expect(false, "Benchmark error: \(error)")
    }
}
