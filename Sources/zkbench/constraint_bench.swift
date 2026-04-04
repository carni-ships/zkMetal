// Constraint IR benchmark — compile time, evaluation throughput, GPU vs CPU, correctness
import Foundation
import Metal
import zkMetal

func runConstraintBench() {
    fputs("\n=== Constraint IR Benchmark ===\n", stderr)
    fputs("Version: \(ConstraintEngine.version.description)\n\n", stderr)

    do {
        let engine = try ConstraintEngine()
        fputs("GPU: \(engine.device.name)\n\n", stderr)

        // 1. Compilation benchmarks
        compilationBenchmarks(engine: engine)

        // 2. Correctness tests
        correctnessTests(engine: engine)

        // 3. Evaluation throughput
        evaluationBenchmarks(engine: engine)

    } catch {
        fputs("Error: \(error)\n", stderr)
    }
}

// MARK: - Compilation Benchmarks

private func compilationBenchmarks(engine: ConstraintEngine) {
    fputs("--- Compilation Time ---\n", stderr)

    let sizes = [10, 50, 100, 200]
    for n in sizes {
        let cs = ConstraintSystem.r1cs(numGates: n)
        do {
            let t0 = CFAbsoluteTimeGetCurrent()
            let compiled = try engine.compile(system: cs)
            let dt = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
            fputs("  R1CS \(n) gates: \(String(format: "%.1f", dt))ms compile"
                  + " (\(cs.constraints.count) constraints, \(cs.totalNodeCount) nodes)\n", stderr)
            _ = compiled  // silence unused warning
        } catch {
            fputs("  R1CS \(n) gates: COMPILE FAILED - \(error)\n", stderr)
        }
    }

    // Fibonacci compilation
    for steps in [10, 50, 100] {
        let cs = ConstraintSystem.fibonacci(steps: steps)
        do {
            let compiled = try engine.compile(system: cs)
            fputs("  Fibonacci \(steps) steps: \(String(format: "%.1f", compiled.compileTimeMs))ms compile"
                  + " (\(cs.constraints.count) constraints)\n", stderr)
        } catch {
            fputs("  Fibonacci \(steps): COMPILE FAILED - \(error)\n", stderr)
        }
    }

    // Range check compilation
    for bits in [8, 16, 32] {
        let cs = ConstraintSystem.rangeCheck(wire: 0, bits: bits)
        do {
            let compiled = try engine.compile(system: cs)
            fputs("  RangeCheck \(bits)-bit: \(String(format: "%.1f", compiled.compileTimeMs))ms compile"
                  + " (\(cs.constraints.count) constraints)\n", stderr)
        } catch {
            fputs("  RangeCheck \(bits)-bit: COMPILE FAILED - \(error)\n", stderr)
        }
    }
    fputs("\n", stderr)
}

// MARK: - Correctness Tests

private func correctnessTests(engine: ConstraintEngine) {
    fputs("--- Correctness Tests ---\n", stderr)
    var passed = 0
    var failed = 0

    // Test 1: Fibonacci trace should satisfy Fibonacci constraints
    do {
        let steps = 10
        let cs = ConstraintSystem.fibonacci(steps: steps)
        let compiled = try engine.compile(system: cs)

        // Build a valid Fibonacci trace: single row with fib values
        var fib = [Fr](repeating: Fr.zero, count: steps)
        fib[0] = frFromInt(1)
        fib[1] = frFromInt(1)
        for i in 2..<steps {
            fib[i] = frAdd(fib[i - 1], fib[i - 2])
        }

        let trace = try engine.createTrace([fib])
        let ok = try engine.verify(compiled: compiled, trace: trace, numRows: 1)
        if ok {
            fputs("  PASS: Fibonacci valid trace satisfies constraints\n", stderr)
            passed += 1
        } else {
            fputs("  FAIL: Fibonacci valid trace should satisfy constraints\n", stderr)
            failed += 1
        }
    } catch {
        fputs("  FAIL: Fibonacci test error: \(error)\n", stderr)
        failed += 1
    }

    // Test 2: Invalid Fibonacci trace should fail
    do {
        let steps = 10
        let cs = ConstraintSystem.fibonacci(steps: steps)
        let compiled = try engine.compile(system: cs)

        // Invalid: all ones (1+1 != 1)
        var bad = [Fr](repeating: Fr.one, count: steps)
        bad[2] = frFromInt(42)  // definitely wrong

        let trace = try engine.createTrace([bad])
        let ok = try engine.verify(compiled: compiled, trace: trace, numRows: 1)
        if !ok {
            fputs("  PASS: Fibonacci invalid trace correctly fails\n", stderr)
            passed += 1
        } else {
            fputs("  FAIL: Fibonacci invalid trace should NOT pass\n", stderr)
            failed += 1
        }
    } catch {
        fputs("  FAIL: Fibonacci invalid test error: \(error)\n", stderr)
        failed += 1
    }

    // Test 3: R1CS a*b=c
    do {
        let cs = ConstraintSystem.r1cs(numGates: 1)
        let compiled = try engine.compile(system: cs)

        let a = frFromInt(3)
        let b = frFromInt(7)
        let c = frMul(a, b)  // 21

        let trace = try engine.createTrace([[a, b, c]])
        let ok = try engine.verify(compiled: compiled, trace: trace, numRows: 1)
        if ok {
            fputs("  PASS: R1CS 3*7=21 satisfies constraint\n", stderr)
            passed += 1
        } else {
            fputs("  FAIL: R1CS 3*7=21 should satisfy constraint\n", stderr)
            failed += 1
        }

        // Invalid: a*b != wrong_c
        let wrongC = frFromInt(99)
        let badTrace = try engine.createTrace([[a, b, wrongC]])
        let ok2 = try engine.verify(compiled: compiled, trace: badTrace, numRows: 1)
        if !ok2 {
            fputs("  PASS: R1CS 3*7!=99 correctly fails\n", stderr)
            passed += 1
        } else {
            fputs("  FAIL: R1CS 3*7!=99 should fail\n", stderr)
            failed += 1
        }
    } catch {
        fputs("  FAIL: R1CS test error: \(error)\n", stderr)
        failed += 1
    }

    // Test 4: Boolean constraint
    do {
        let cs = ConstraintSystem(numWires: 1)
        cs.assertBool(Wire.col(0))
        let compiled = try engine.compile(system: cs)

        // 0 is boolean
        let trace0 = try engine.createTrace([[Fr.zero]])
        let ok0 = try engine.verify(compiled: compiled, trace: trace0, numRows: 1)

        // 1 is boolean
        let trace1 = try engine.createTrace([[Fr.one]])
        let ok1 = try engine.verify(compiled: compiled, trace: trace1, numRows: 1)

        // 2 is not boolean
        let trace2 = try engine.createTrace([[frFromInt(2)]])
        let ok2 = try engine.verify(compiled: compiled, trace: trace2, numRows: 1)

        if ok0 && ok1 && !ok2 {
            fputs("  PASS: Boolean constraint (0=ok, 1=ok, 2=fail)\n", stderr)
            passed += 1
        } else {
            fputs("  FAIL: Boolean constraint (0=\(ok0), 1=\(ok1), 2=\(!ok2))\n", stderr)
            failed += 1
        }
    } catch {
        fputs("  FAIL: Boolean test error: \(error)\n", stderr)
        failed += 1
    }

    // Test 5: GPU vs CPU agreement
    do {
        let cs = ConstraintSystem.r1cs(numGates: 3)
        let compiled = try engine.compile(system: cs)

        // Multiple rows
        var rows = [[Fr]]()
        for i: UInt64 in 1...4 {
            let a = frFromInt(i)
            let b = frFromInt(i + 1)
            let c = frMul(a, b)
            rows.append([a, b, c, a, b, c, a, b, c])
        }
        let trace = try engine.createTrace(rows)

        let gpuResults = try engine.evaluateToArray(compiled: compiled, trace: trace, numRows: rows.count)
        let cpuResults = engine.evaluateCPU(system: cs, trace: rows)

        var match = true
        for i in 0..<gpuResults.count {
            let gv = gpuResults[i].v
            let cv = cpuResults[i].v
            if gv.0 != cv.0 || gv.1 != cv.1 || gv.2 != cv.2 || gv.3 != cv.3 ||
               gv.4 != cv.4 || gv.5 != cv.5 || gv.6 != cv.6 || gv.7 != cv.7 {
                match = false
                break
            }
        }
        if match {
            fputs("  PASS: GPU and CPU constraint evaluation match (\(gpuResults.count) values)\n", stderr)
            passed += 1
        } else {
            fputs("  FAIL: GPU and CPU results differ\n", stderr)
            failed += 1
        }
    } catch {
        fputs("  FAIL: GPU/CPU comparison error: \(error)\n", stderr)
        failed += 1
    }

    fputs("  Results: \(passed) passed, \(failed) failed\n\n", stderr)
}

// MARK: - Evaluation Throughput Benchmarks

private func evaluationBenchmarks(engine: ConstraintEngine) {
    fputs("--- Evaluation Throughput ---\n", stderr)

    let gateCount = 20
    let cs = ConstraintSystem.r1cs(numGates: gateCount)

    do {
        let compiled = try engine.compile(system: cs)
        let numWires = cs.numWires
        let numConstraints = cs.constraints.count

        for logRows in [10, 14, 16] {
            let numRows = 1 << logRows

            // Build random-ish trace (valid R1CS: a*b=c)
            var flat = [Fr](repeating: Fr.zero, count: numRows * numWires)
            for row in 0..<numRows {
                for gate in 0..<gateCount {
                    let a = frFromInt(UInt64(row * gateCount + gate + 1))
                    let b = frFromInt(UInt64(gate + 2))
                    let c = frMul(a, b)
                    flat[row * numWires + gate * 3] = a
                    flat[row * numWires + gate * 3 + 1] = b
                    flat[row * numWires + gate * 3 + 2] = c
                }
            }

            let trace = try engine.createTraceFlat(flat, numCols: numWires)

            // Warm up
            _ = try engine.evaluate(compiled: compiled, trace: trace, numRows: numRows)

            // Benchmark GPU
            let iters = logRows <= 14 ? 5 : 3
            var gpuMs = Double.infinity
            for _ in 0..<iters {
                let t0 = CFAbsoluteTimeGetCurrent()
                _ = try engine.evaluate(compiled: compiled, trace: trace, numRows: numRows)
                let dt = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
                gpuMs = min(gpuMs, dt)
            }

            let totalConstraints = Double(numRows) * Double(numConstraints)
            let throughput = totalConstraints / (gpuMs / 1000.0)

            fputs("  2^\(logRows) rows x \(numConstraints) constraints: "
                  + "\(String(format: "%.2f", gpuMs))ms GPU"
                  + " (\(String(format: "%.1f", throughput / 1e6))M constraints/s)\n", stderr)

            // CPU comparison for smaller sizes
            if logRows <= 14 {
                // Convert flat to 2D for CPU eval
                var rows2D = [[Fr]]()
                rows2D.reserveCapacity(numRows)
                for row in 0..<numRows {
                    let start = row * numWires
                    rows2D.append(Array(flat[start..<(start + numWires)]))
                }
                let cpuT0 = CFAbsoluteTimeGetCurrent()
                _ = engine.evaluateCPU(system: cs, trace: rows2D)
                let cpuMs = (CFAbsoluteTimeGetCurrent() - cpuT0) * 1000.0
                let speedup = cpuMs / gpuMs
                fputs("    CPU: \(String(format: "%.1f", cpuMs))ms"
                      + " (GPU \(String(format: "%.1f", speedup))x faster)\n", stderr)
            }
        }

    } catch {
        fputs("  Error: \(error)\n", stderr)
    }

    // Fibonacci throughput
    fputs("\n  Fibonacci constraint throughput:\n", stderr)
    do {
        let steps = 50
        let cs = ConstraintSystem.fibonacci(steps: steps)
        let compiled = try engine.compile(system: cs)
        let numConstraints = cs.constraints.count

        for logRows in [10, 14, 16] {
            let numRows = 1 << logRows
            var flat = [Fr](repeating: Fr.zero, count: numRows * steps)
            for row in 0..<numRows {
                flat[row * steps] = frFromInt(UInt64(row + 1))
                flat[row * steps + 1] = frFromInt(1)
                for i in 2..<steps {
                    flat[row * steps + i] = frAdd(flat[row * steps + i - 1], flat[row * steps + i - 2])
                }
            }

            let trace = try engine.createTraceFlat(flat, numCols: steps)
            _ = try engine.evaluate(compiled: compiled, trace: trace, numRows: numRows)

            var gpuMs = Double.infinity
            for _ in 0..<3 {
                let t0 = CFAbsoluteTimeGetCurrent()
                _ = try engine.evaluate(compiled: compiled, trace: trace, numRows: numRows)
                let dt = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
                gpuMs = min(gpuMs, dt)
            }

            let totalConstraints = Double(numRows) * Double(numConstraints)
            let throughput = totalConstraints / (gpuMs / 1000.0)
            fputs("    2^\(logRows) rows x \(numConstraints) fib constraints: "
                  + "\(String(format: "%.2f", gpuMs))ms"
                  + " (\(String(format: "%.1f", throughput / 1e6))M/s)\n", stderr)
        }
    } catch {
        fputs("  Fibonacci bench error: \(error)\n", stderr)
    }
    fputs("\n", stderr)
}
