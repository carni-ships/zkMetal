// Stark252 STARK Tests -- end-to-end prove + verify round-trips over Stark252 field
// Tests: Fibonacci AIR, boundary constraints, proof generation/verification, soundness, benchmarks

import zkMetal
import Foundation

public func runStark252STARKTests() {
    suite("Stark252 STARK -- Field Sanity")
    s252StarkTestFieldOps()

    suite("Stark252 STARK -- AIR Trace Validation")
    s252StarkTestFibTrace()
    s252StarkTestFibBoundary()
    s252StarkTestGenericDoubling()

    suite("Stark252 STARK -- Prove + Verify")
    s252StarkTestFibProveVerify()
    s252StarkTestFibProveVerifyLarger()
    s252StarkTestGenericProveVerify()

    suite("Stark252 STARK -- Soundness")
    s252StarkTestTamperedTrace()
    s252StarkTestTamperedProof()
    s252StarkTestWrongBoundary()

    suite("Stark252 STARK -- Config + Metadata")
    s252StarkTestConfigSecurity()
    s252StarkTestProofMetadata()

    suite("Stark252 STARK -- Benchmark")
    s252StarkBenchmark()
}

// MARK: - Field Sanity

func s252StarkTestFieldOps() {
    // Addition
    let a = stark252FromInt(100)
    let b = stark252FromInt(200)
    let sum = stark252Add(a, b)
    let sumInt = stark252ToInt(sum)
    expectEqual(sumInt[0], UInt64(300), "Stark252 add")

    // Subtraction
    let diff = stark252Sub(b, a)
    let diffInt = stark252ToInt(diff)
    expectEqual(diffInt[0], UInt64(100), "Stark252 sub")

    // Multiplication
    let prod = stark252Mul(stark252FromInt(1000), stark252FromInt(2000))
    let prodInt = stark252ToInt(prod)
    expectEqual(prodInt[0], UInt64(2_000_000), "Stark252 mul")

    // Inverse
    let x = stark252FromInt(42)
    let xInv = stark252Inverse(x)
    let shouldBeOne = stark252Mul(x, xInv)
    let oneInt = stark252ToInt(shouldBeOne)
    expectEqual(oneInt[0], UInt64(1), "Stark252 inverse")
    expectEqual(oneInt[1], UInt64(0), "Stark252 inverse high zero")

    // Power
    let p3 = stark252Pow(stark252FromInt(3), 10)
    let p3Int = stark252ToInt(p3)
    expectEqual(p3Int[0], UInt64(59049), "Stark252 pow 3^10")

    // Root of unity
    let omega = stark252RootOfUnity(logN: 3)
    let omega8 = stark252Pow(omega, 8)
    let omega8Int = stark252ToInt(omega8)
    expectEqual(omega8Int[0], UInt64(1), "Stark252 8th root of unity order")
    expectEqual(omega8Int[1], UInt64(0), "Stark252 8th root high zero")

    // Root of unity is not trivial
    let omega4 = stark252Pow(omega, 4)
    let omega4Int = stark252ToInt(omega4)
    let isTrivial = omega4Int[0] == 1 && omega4Int[1] == 0 && omega4Int[2] == 0 && omega4Int[3] == 0
    expect(!isTrivial, "Stark252 8th root not 4th root")
}

// MARK: - AIR Trace Validation

func s252StarkTestFibTrace() {
    let air = CairoFibonacciAIR(logTraceLength: 4)  // 16 rows
    let trace = air.generateTrace()

    expectEqual(trace.count, 2, "Fib trace 2 columns")
    expectEqual(trace[0].count, 16, "Fib trace 16 rows")

    // Check initial values (1, 1)
    let a0Int = stark252ToInt(trace[0][0])
    let b0Int = stark252ToInt(trace[1][0])
    expectEqual(a0Int[0], UInt64(1), "Fib a[0] = 1")
    expectEqual(b0Int[0], UInt64(1), "Fib b[0] = 1")

    // Check Fibonacci relation: a[i+1] = b[i], b[i+1] = a[i] + b[i]
    for i in 0..<15 {
        let aNext = stark252ToInt(trace[0][i + 1])
        let bCurr = stark252ToInt(trace[1][i])
        expectEqual(aNext, bCurr, "Fib a[\(i+1)] = b[\(i)]")

        let expectedB = stark252Add(trace[0][i], trace[1][i])
        let bNext = stark252ToInt(trace[1][i + 1])
        let expBInt = stark252ToInt(expectedB)
        expectEqual(bNext, expBInt, "Fib b[\(i+1)] = a[\(i)] + b[\(i)]")
    }

    // Validate trace satisfies all constraints
    let err = air.verifyTrace(trace)
    expect(err == nil, "Fib trace valid: \(err ?? "")")
}

func s252StarkTestFibBoundary() {
    // Test with custom initial values
    let a0 = stark252FromInt(3)
    let b0 = stark252FromInt(5)
    let air = CairoFibonacciAIR(logTraceLength: 3, a0: a0, b0: b0)
    let trace = air.generateTrace()

    // Check boundary constraints
    let a0Int = stark252ToInt(trace[0][0])
    let b0Int = stark252ToInt(trace[1][0])
    expectEqual(a0Int[0], UInt64(3), "Custom Fib a[0] = 3")
    expectEqual(b0Int[0], UInt64(5), "Custom Fib b[0] = 5")

    // Verify Fibonacci sequence: 3, 5, 8, 13, 21, 34, 55, 89
    let a1Int = stark252ToInt(trace[0][1])
    let b1Int = stark252ToInt(trace[1][1])
    expectEqual(a1Int[0], UInt64(5), "Custom Fib a[1] = 5")
    expectEqual(b1Int[0], UInt64(8), "Custom Fib b[1] = 8")
    let a2Int = stark252ToInt(trace[0][2])
    let b2Int = stark252ToInt(trace[1][2])
    expectEqual(a2Int[0], UInt64(8), "Custom Fib a[2] = 8")
    expectEqual(b2Int[0], UInt64(13), "Custom Fib b[2] = 13")

    // Check boundary constraint definition
    let bcs = air.boundaryConstraints
    expectEqual(bcs.count, 2, "Fib has 2 boundary constraints")
    expectEqual(bcs[0].column, 0, "BC 0 column")
    expectEqual(bcs[0].row, 0, "BC 0 row")
    expectEqual(bcs[1].column, 1, "BC 1 column")

    // Full trace validation
    let err = air.verifyTrace(trace)
    expect(err == nil, "Custom Fib trace valid: \(err ?? "")")
}

func s252StarkTestGenericDoubling() {
    let logN = 3
    let n = 1 << logN
    let one = Stark252.one
    let two = stark252FromInt(2)

    let air = GenericStark252AIR(
        numColumns: 1,
        logTraceLength: logN,
        numConstraints: 1,
        constraintDegree: 1,
        boundaryConstraints: [(column: 0, row: 0, value: one)],
        traceGenerator: {
            var col = [Stark252](repeating: Stark252.zero, count: n)
            col[0] = Stark252.one
            for i in 1..<n {
                col[i] = stark252Add(col[i - 1], col[i - 1])  // doubling
            }
            return [col]
        },
        constraintEvaluator: { current, next in
            // next[0] = 2 * current[0]
            let doubled = stark252Add(current[0], current[0])
            return [stark252Sub(next[0], doubled)]
        }
    )

    let trace = air.generateTrace()
    let v0 = stark252ToInt(trace[0][0])
    let v1 = stark252ToInt(trace[0][1])
    let v2 = stark252ToInt(trace[0][2])
    let v3 = stark252ToInt(trace[0][3])
    expectEqual(v0[0], UInt64(1), "Doubling a[0] = 1")
    expectEqual(v1[0], UInt64(2), "Doubling a[1] = 2")
    expectEqual(v2[0], UInt64(4), "Doubling a[2] = 4")
    expectEqual(v3[0], UInt64(8), "Doubling a[3] = 8")

    let err = air.verifyTrace(trace)
    expect(err == nil, "Doubling trace valid: \(err ?? "")")
}

// MARK: - Prove + Verify Round-Trips

func s252StarkTestFibProveVerify() {
    do {
        let air = CairoFibonacciAIR(logTraceLength: 3)  // 8 rows
        let stark = Stark252STARK(config: .fast)
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

func s252StarkTestFibProveVerifyLarger() {
    do {
        let air = CairoFibonacciAIR(logTraceLength: 5)  // 32 rows
        let stark = Stark252STARK(config: .fast)
        let result = try stark.prove(air: air)
        let valid = try stark.verify(air: air, proof: result.proof)

        expect(valid, "Fibonacci 32-row prove-verify")
        expectEqual(result.proof.traceLength, 32, "32-row trace length")
    } catch {
        expect(false, "Fibonacci 32-row error: \(error)")
    }
}

func s252StarkTestGenericProveVerify() {
    do {
        let logN = 3
        let n = 1 << logN
        let three = stark252FromInt(3)

        // Simple tripling AIR: a[i+1] = 3 * a[i] mod p
        let air = GenericStark252AIR(
            numColumns: 1,
            logTraceLength: logN,
            numConstraints: 1,
            constraintDegree: 1,
            boundaryConstraints: [(column: 0, row: 0, value: Stark252.one)],
            traceGenerator: {
                var col = [Stark252](repeating: Stark252.zero, count: n)
                col[0] = Stark252.one
                let three = stark252FromInt(3)
                for i in 1..<n {
                    col[i] = stark252Mul(col[i - 1], three)
                }
                return [col]
            },
            constraintEvaluator: { current, next in
                let three = stark252FromInt(3)
                let tripled = stark252Mul(current[0], three)
                return [stark252Sub(next[0], tripled)]
            }
        )

        // Verify trace first
        let err = air.verifyTrace(air.generateTrace())
        expect(err == nil, "Tripling trace valid: \(err ?? "")")

        let stark = Stark252STARK(config: .fast)
        let (result, verified) = try stark.proveAndVerify(air: air)
        expect(verified, "Generic tripling AIR prove-verify")
        expect(result.proofSizeBytes > 0, "Tripling proof has size")
    } catch {
        expect(false, "Generic AIR prove error: \(error)")
    }
}

// MARK: - Soundness Tests

func s252StarkTestTamperedTrace() {
    let air = CairoFibonacciAIR(logTraceLength: 3)
    let trace = air.generateTrace()

    // Tamper with a value in the middle
    var badTrace = trace
    badTrace[0][2] = stark252FromInt(999)  // corrupt a[2]

    let err = air.verifyTrace(badTrace)
    expect(err != nil, "Tampered Fibonacci trace rejected: \(err ?? "nil")")
}

func s252StarkTestTamperedProof() {
    do {
        let air = CairoFibonacciAIR(logTraceLength: 3)
        let stark = Stark252STARK(config: .fast)
        let result = try stark.prove(air: air)

        // Tamper with alpha in the proof
        let tamperedProof = Stark252STARKProof(
            traceCommitments: result.proof.traceCommitments,
            compositionCommitment: result.proof.compositionCommitment,
            friProof: result.proof.friProof,
            queryResponses: result.proof.queryResponses,
            alpha: stark252FromInt(12345),  // wrong alpha
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

func s252StarkTestWrongBoundary() {
    // Create a valid trace for (a0=1, b0=1) but check against (a0=2, b0=1) AIR
    let air1 = CairoFibonacciAIR(logTraceLength: 3)
    let trace1 = air1.generateTrace()

    // Verify against a different AIR with different boundary
    let air2 = CairoFibonacciAIR(logTraceLength: 3, a0: stark252FromInt(2))

    // The trace from air1 should fail boundary check for air2
    let err = air2.verifyTrace(trace1)
    expect(err != nil, "Wrong boundary trace rejected: \(err ?? "nil")")
}

// MARK: - Config + Metadata

func s252StarkTestConfigSecurity() {
    let fast = Stark252STARKConfig.fast
    expect(fast.securityBits > 0, "Fast config has security bits: \(fast.securityBits)")
    expect(fast.blowupFactor >= 2, "Fast config blowup >= 2")

    let starknet = Stark252STARKConfig.starknetDefault
    expect(starknet.securityBits >= 100, "StarkNet config >= 100-bit security: \(starknet.securityBits)")
    expect(starknet.numQueries >= 80, "StarkNet config >= 80 queries")

    let high = Stark252STARKConfig.highSecurity
    expect(high.securityBits >= 150, "High security >= 150 bits: \(high.securityBits)")
}

func s252StarkTestProofMetadata() {
    do {
        let air = CairoFibonacciAIR(logTraceLength: 3)
        let stark = Stark252STARK(config: .fast)
        let result = try stark.prove(air: air)

        expectEqual(result.traceLength, 8, "Result trace length")
        expectEqual(result.numColumns, 2, "Result num columns")
        expectEqual(result.numConstraints, 2, "Result num constraints")
        expect(result.proofSizeBytes > 0, "Proof size > 0")
        expect(result.proveTimeSeconds > 0, "Prove time > 0")

        // Check summary string is non-empty
        let summary = result.summary
        expect(summary.count > 0, "Summary non-empty")
        expect(summary.contains("Stark252 STARK"), "Summary contains header")
    } catch {
        expect(false, "Proof metadata error: \(error)")
    }
}

// MARK: - Benchmark

func s252StarkBenchmark() {
    do {
        let air = CairoFibonacciAIR(logTraceLength: 5)  // 32 rows
        let stark = Stark252STARK(config: .fast)

        // Warm up
        _ = try stark.prove(air: air)

        // Timed run
        let t0 = CFAbsoluteTimeGetCurrent()
        let result = try stark.prove(air: air)
        let elapsed = CFAbsoluteTimeGetCurrent() - t0

        print(String(format: "  Stark252 STARK Fibonacci (32 rows): %.3fms, proof %d bytes",
                      elapsed * 1000, result.proofSizeBytes))
        expect(elapsed < 60.0, "Prove completes in < 60s")
    } catch {
        expect(false, "Benchmark error: \(error)")
    }
}
