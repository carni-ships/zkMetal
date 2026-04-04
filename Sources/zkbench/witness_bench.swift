// Witness generation benchmark — GPU trace evaluation vs CPU reference

import Foundation
import Metal
import zkMetal

// MARK: - CPU reference trace evaluation

private func cpuTraceEval(program: CompiledProgram, inputs: [Fr], numRows: Int) -> [[Fr]] {
    let numCols = program.numCols
    var trace = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: numCols), count: numRows)

    let instrs = program.instructions
    let constants = program.constants
    let inputWidth = program.inputWidth

    for row in 0..<numRows {
        for i in 0..<program.numInstructions {
            let opRaw = instrs[i * 4]
            let dst = Int(instrs[i * 4 + 1])
            let src1 = Int(instrs[i * 4 + 2])
            let src2 = Int(instrs[i * 4 + 3])

            let op = opRaw & 0x7F
            let useConst = (opRaw & 0x80) != 0

            if op == 4 { // LOAD
                trace[row][dst] = inputs[row * inputWidth + src1]
                continue
            }

            let a = trace[row][src1]
            let b = useConst ? constants[Int(src2)] : trace[row][Int(src2)]

            switch op {
            case 0: trace[row][dst] = frAdd(a, b)          // ADD
            case 1: trace[row][dst] = frMul(a, b)          // MUL
            case 2: trace[row][dst] = a                     // COPY
            case 3: trace[row][dst] = frSub(a, b)          // SUB
            case 5: trace[row][dst] = frSqr(a)             // SQR
            case 6: trace[row][dst] = frAdd(a, a)          // DOUBLE
            case 7: trace[row][dst] = frSub(Fr.zero, a)    // NEG
            case 8: // SELECT
                let falseCol = Int(src2 & 0xFFFF)
                let selCol = Int((src2 >> 16) & 0xFFFF)
                let selector = trace[row][selCol]
                trace[row][dst] = selector.isZero ? trace[row][falseCol] : a
            default: break
            }
        }
    }
    return trace
}

// MARK: - Correctness check

private func checkCorrectness(gpu: MTLBuffer, cpu: [[Fr]], numRows: Int, numCols: Int, label: String) -> Bool {
    let ptr = gpu.contents().bindMemory(to: Fr.self, capacity: numRows * numCols)
    var mismatches = 0
    for row in 0..<min(numRows, 64) {  // Check first 64 rows
        for col in 0..<numCols {
            let gpuVal = ptr[row * numCols + col]
            let cpuVal = cpu[row][col]
            if gpuVal.v.0 != cpuVal.v.0 || gpuVal.v.1 != cpuVal.v.1 ||
               gpuVal.v.2 != cpuVal.v.2 || gpuVal.v.3 != cpuVal.v.3 ||
               gpuVal.v.4 != cpuVal.v.4 || gpuVal.v.5 != cpuVal.v.5 ||
               gpuVal.v.6 != cpuVal.v.6 || gpuVal.v.7 != cpuVal.v.7 {
                if mismatches < 3 {
                    fputs("  MISMATCH [\(label)] row=\(row) col=\(col): GPU=\(gpuVal.v.0)... CPU=\(cpuVal.v.0)...\n", stderr)
                }
                mismatches += 1
            }
        }
    }
    if mismatches > 0 {
        fputs("  FAIL [\(label)]: \(mismatches) mismatches in first 64 rows\n", stderr)
        return false
    }
    fputs("  PASS [\(label)]: correctness verified (first 64 rows)\n", stderr)
    return true
}

// MARK: - Benchmarks

func runWitnessBench() {
    fputs("\n=== Witness Generation Benchmark ===\n", stderr)
    fputs("Version: \(TraceEngine.version.description)\n\n", stderr)

    let engine: TraceEngine
    do {
        engine = try TraceEngine()
    } catch {
        fputs("Error: Failed to create TraceEngine: \(error)\n", stderr)
        return
    }
    fputs("Device: \(engine.device.name)\n\n", stderr)

    // --- Test 1: Simple addition chain ---
    do {
        fputs("--- Addition Chain (10 cols, 10 add ops) ---\n", stderr)
        let prog = TraceProgram()
        prog.loadInput(0, 0)  // col 0 = input[0]
        prog.loadInput(1, 1)  // col 1 = input[1]
        // Chain: col[i+2] = col[i] + col[i+1] (Fibonacci-like)
        for i in 0..<8 {
            prog.add(i + 2, i, i + 1)
        }
        let compiled = prog.compile()
        fputs("  Program: \(compiled.numInstructions) instrs, \(compiled.numCols) cols, \(compiled.inputWidth) inputs\n", stderr)

        let sizes = [1 << 14, 1 << 16, 1 << 18, 1 << 20]
        for n in sizes {
            // Generate random inputs: 2 per row
            var inputs = [Fr](repeating: Fr.zero, count: n * 2)
            var rng: UInt64 = 0xDEAD_BEEF_1234_5678
            for i in 0..<inputs.count {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                let v = UInt32(truncatingIfNeeded: rng >> 32)
                inputs[i] = frFromInt(UInt64(v & 0xFFFF))  // small values for easy verification
            }

            // Correctness check on small size
            if n == sizes[0] {
                let cpuResult = cpuTraceEval(program: compiled, inputs: inputs, numRows: n)
                let gpuBuf = try engine.evaluate(program: compiled, inputs: inputs, numRows: n)
                let _ = checkCorrectness(gpu: gpuBuf, cpu: cpuResult, numRows: n, numCols: compiled.numCols, label: "add-chain")
            }

            // Warmup
            let _ = try engine.evaluate(program: compiled, inputs: inputs, numRows: n)

            // Bench GPU
            let iters = n <= (1 << 16) ? 10 : 3
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0..<iters {
                let _ = try engine.evaluate(program: compiled, inputs: inputs, numRows: n)
            }
            let gpuMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0 / Double(iters)

            // Bench CPU
            var cpuMs = 0.0
            if !skipCPU && n <= (1 << 18) {
                let t1 = CFAbsoluteTimeGetCurrent()
                let cpuIters = n <= (1 << 16) ? 3 : 1
                for _ in 0..<cpuIters {
                    let _ = cpuTraceEval(program: compiled, inputs: inputs, numRows: n)
                }
                cpuMs = (CFAbsoluteTimeGetCurrent() - t1) * 1000.0 / Double(cpuIters)
            }

            let throughput = Double(n) * Double(compiled.numCols) / (gpuMs / 1000.0) / 1e6
            var line = String(format: "  n=2^%d (%dk rows): GPU %.2fms", Int(log2(Double(n))), n / 1024, gpuMs)
            if cpuMs > 0 {
                line += String(format: ", CPU %.1fms, speedup %.1fx", cpuMs, cpuMs / gpuMs)
            }
            line += String(format: " [%.1f M cells/s]", throughput)
            fputs(line + "\n", stderr)
        }
    } catch {
        fputs("  Error in addition chain bench: \(error)\n", stderr)
    }

    // --- Test 2: Multiply-heavy trace (simulates constraint evaluation) ---
    do {
        fputs("\n--- Multiply-Heavy Trace (arithmetic circuit, 8 cols) ---\n", stderr)
        let prog = TraceProgram()
        prog.loadInput(0, 0)  // a
        prog.loadInput(1, 1)  // b
        prog.mul(2, 0, 1)     // c = a * b
        prog.sqr(3, 2)        // d = c^2
        prog.addConst(4, 3, Fr.one)  // e = d + 1
        prog.mul(5, 4, 0)     // f = e * a
        prog.sub(6, 5, 1)     // g = f - b
        prog.sqr(7, 6)        // h = g^2
        let compiled = prog.compile()
        fputs("  Program: \(compiled.numInstructions) instrs, \(compiled.numCols) cols\n", stderr)

        let sizes = [1 << 14, 1 << 16, 1 << 18, 1 << 20]
        for n in sizes {
            var inputs = [Fr](repeating: Fr.zero, count: n * 2)
            var rng: UInt64 = 0xCAFE_BABE_DEAD_BEEF
            for i in 0..<inputs.count {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                inputs[i] = frFromInt(UInt64(UInt32(truncatingIfNeeded: rng >> 32) & 0xFFFF))
            }

            if n == sizes[0] {
                let cpuResult = cpuTraceEval(program: compiled, inputs: inputs, numRows: n)
                let gpuBuf = try engine.evaluate(program: compiled, inputs: inputs, numRows: n)
                let _ = checkCorrectness(gpu: gpuBuf, cpu: cpuResult, numRows: n, numCols: compiled.numCols, label: "mul-heavy")
            }

            let _ = try engine.evaluate(program: compiled, inputs: inputs, numRows: n)

            let iters = n <= (1 << 16) ? 10 : 3
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0..<iters {
                let _ = try engine.evaluate(program: compiled, inputs: inputs, numRows: n)
            }
            let gpuMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0 / Double(iters)

            var cpuMs = 0.0
            if !skipCPU && n <= (1 << 18) {
                let t1 = CFAbsoluteTimeGetCurrent()
                let _ = cpuTraceEval(program: compiled, inputs: inputs, numRows: n)
                cpuMs = (CFAbsoluteTimeGetCurrent() - t1) * 1000.0
            }

            let throughput = Double(n) * Double(compiled.numCols) / (gpuMs / 1000.0) / 1e6
            var line = String(format: "  n=2^%d (%dk rows): GPU %.2fms", Int(log2(Double(n))), n / 1024, gpuMs)
            if cpuMs > 0 {
                line += String(format: ", CPU %.1fms, speedup %.1fx", cpuMs, cpuMs / gpuMs)
            }
            line += String(format: " [%.1f M cells/s]", throughput)
            fputs(line + "\n", stderr)
        }
    } catch {
        fputs("  Error in mul-heavy bench: \(error)\n", stderr)
    }

    // --- Test 3: Poseidon2-like trace (many rounds, 3 state elements) ---
    do {
        fputs("\n--- Poseidon2-like Trace (4 full rounds, 3-wide state) ---\n", stderr)
        // Simplified: 4 rounds of add + sqr + mix on 3 state columns
        // State cols: 0, 1, 2. Temp col: 3. RC cols: 4, 5, 6 (loaded per round)
        let prog = TraceProgram()
        prog.loadInput(0, 0)  // state[0]
        prog.loadInput(1, 1)  // state[1]
        prog.loadInput(2, 2)  // state[2]

        // 4 full rounds with constant round constants
        for round in 0..<4 {
            let rc0 = frFromInt(UInt64(round * 3 + 1))
            let rc1 = frFromInt(UInt64(round * 3 + 2))
            let rc2 = frFromInt(UInt64(round * 3 + 3))

            // Add round constants
            prog.addConst(0, 0, rc0)
            prog.addConst(1, 1, rc1)
            prog.addConst(2, 2, rc2)

            // S-box: x^5 on all state elements
            // Using temp col 3 for squaring
            for s in 0..<3 {
                prog.sqr(3, s)           // t = s^2
                prog.sqr(3, 3)           // t = s^4
                prog.mul(s, s, 3)        // s = s * s^4 = s^5
            }

            // Simple mix: t = s0+s1+s2, then s_i += t
            prog.add(3, 0, 1)
            prog.add(3, 3, 2)
            prog.add(0, 0, 3)
            prog.add(1, 1, 3)
            prog.add(2, 2, 3)
        }

        let compiled = prog.compile()
        fputs("  Program: \(compiled.numInstructions) instrs, \(compiled.numCols) cols, \(compiled.constants.count) constants\n", stderr)

        let sizes = [1 << 14, 1 << 16, 1 << 18, 1 << 20]
        for n in sizes {
            var inputs = [Fr](repeating: Fr.zero, count: n * 3)
            var rng: UInt64 = 0x1234_5678_ABCD_EF01
            for i in 0..<inputs.count {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                inputs[i] = frFromInt(UInt64(UInt32(truncatingIfNeeded: rng >> 32) & 0xFFFF))
            }

            if n == sizes[0] {
                let cpuResult = cpuTraceEval(program: compiled, inputs: inputs, numRows: n)
                let gpuBuf = try engine.evaluate(program: compiled, inputs: inputs, numRows: n)
                let _ = checkCorrectness(gpu: gpuBuf, cpu: cpuResult, numRows: n, numCols: compiled.numCols, label: "poseidon2-like")
            }

            let _ = try engine.evaluate(program: compiled, inputs: inputs, numRows: n)

            let iters = n <= (1 << 16) ? 5 : 2
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0..<iters {
                let _ = try engine.evaluate(program: compiled, inputs: inputs, numRows: n)
            }
            let gpuMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0 / Double(iters)

            var cpuMs = 0.0
            if !skipCPU && n <= (1 << 16) {
                let t1 = CFAbsoluteTimeGetCurrent()
                let _ = cpuTraceEval(program: compiled, inputs: inputs, numRows: n)
                cpuMs = (CFAbsoluteTimeGetCurrent() - t1) * 1000.0
            }

            let throughput = Double(n) * Double(compiled.numCols) / (gpuMs / 1000.0) / 1e6
            var line = String(format: "  n=2^%d (%dk rows): GPU %.2fms", Int(log2(Double(n))), n / 1024, gpuMs)
            if cpuMs > 0 {
                line += String(format: ", CPU %.1fms, speedup %.1fx", cpuMs, cpuMs / gpuMs)
            }
            line += String(format: " [%.1f M cells/s]", throughput)
            fputs(line + "\n", stderr)
        }
    } catch {
        fputs("  Error in poseidon2-like bench: \(error)\n", stderr)
    }

    fputs("\n", stderr)

    // ========================================================================
    // M31 Witness Engine — GPU Fibonacci trace generation
    // ========================================================================
    fputs("=== M31 Witness Engine (Fibonacci AIR) ===\n", stderr)

    let m31Engine: WitnessEngine
    do {
        m31Engine = try WitnessEngine()
    } catch {
        fputs("Error: Failed to create WitnessEngine: \(error)\n", stderr)
        return
    }
    fputs("Device: \(m31Engine.device.name)\n\n", stderr)

    // --- Test 4: GPU Fibonacci trace vs CPU FibonacciAIR.generateTrace() ---
    do {
        fputs("--- Fibonacci AIR Trace (matrix-power doubling) ---\n", stderr)

        let a0 = M31.one
        let b0 = M31.one

        let sizes = [1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22]
        for n in sizes {
            // Correctness check on small sizes
            if n <= (1 << 16) {
                let (gpuA, gpuB) = try m31Engine.generateFibonacciTrace(a0: a0, b0: b0, numRows: n)
                // CPU reference
                var cpuA = [M31](repeating: M31.zero, count: n)
                var cpuB = [M31](repeating: M31.zero, count: n)
                cpuA[0] = a0; cpuB[0] = b0
                for i in 1..<n {
                    cpuA[i] = cpuB[i - 1]
                    cpuB[i] = m31Add(cpuA[i - 1], cpuB[i - 1])
                }
                var mismatches = 0
                for i in 0..<n {
                    if gpuA[i].v != cpuA[i].v || gpuB[i].v != cpuB[i].v {
                        if mismatches < 3 {
                            fputs("  MISMATCH row=\(i): GPU=(\(gpuA[i].v),\(gpuB[i].v)) CPU=(\(cpuA[i].v),\(cpuB[i].v))\n", stderr)
                        }
                        mismatches += 1
                    }
                }
                if mismatches > 0 {
                    fputs("  FAIL: \(mismatches) mismatches in \(n) rows\n", stderr)
                } else if n == sizes[0] {
                    fputs("  PASS: correctness verified (\(n) rows)\n", stderr)
                }
            }

            // Warmup
            let _ = try m31Engine.generateFibonacciTrace(a0: a0, b0: b0, numRows: n)

            // Bench GPU
            let iters = n <= (1 << 16) ? 20 : (n <= (1 << 20) ? 5 : 3)
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0..<iters {
                let _ = try m31Engine.generateFibonacciTrace(a0: a0, b0: b0, numRows: n)
            }
            let gpuMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0 / Double(iters)

            // Bench CPU
            var cpuMs = 0.0
            if !skipCPU && n <= (1 << 20) {
                let cpuIters = n <= (1 << 16) ? 10 : 1
                let t1 = CFAbsoluteTimeGetCurrent()
                for _ in 0..<cpuIters {
                    var a = [M31](repeating: M31.zero, count: n)
                    var b = [M31](repeating: M31.zero, count: n)
                    a[0] = a0; b[0] = b0
                    for i in 1..<n {
                        a[i] = b[i - 1]
                        b[i] = m31Add(a[i - 1], b[i - 1])
                    }
                }
                cpuMs = (CFAbsoluteTimeGetCurrent() - t1) * 1000.0 / Double(cpuIters)
            }

            let throughput = Double(n) * 2.0 / (gpuMs / 1000.0) / 1e6
            var line = String(format: "  n=2^%d (%dk rows): GPU %.3fms", Int(log2(Double(n))), n / 1024, gpuMs)
            if cpuMs > 0 {
                line += String(format: ", CPU %.2fms, speedup %.1fx", cpuMs, cpuMs / gpuMs)
            }
            line += String(format: " [%.1f M cells/s]", throughput)
            fputs(line + "\n", stderr)
        }
    } catch {
        fputs("  Error in Fibonacci trace bench: \(error)\n", stderr)
    }

    // --- Test 5: Generic M31 trace (independent rows) ---
    do {
        fputs("\n--- Generic M31 Trace (Poseidon2-like, 4 cols, independent rows) ---\n", stderr)
        let prog = M31TraceProgram()
        prog.loadInput(0, 0)
        prog.loadInput(1, 1)
        prog.mul(2, 0, 1)     // c = a * b
        prog.sqr(3, 2)        // d = c^2
        prog.addConst(0, 3, M31(v: 7))  // a = d + 7
        prog.mul(1, 0, 3)     // b = a * d
        let compiled = prog.compile()
        fputs("  Program: \(compiled.numInstructions) instrs, \(compiled.numCols) cols\n", stderr)

        let sizes = [1 << 14, 1 << 16, 1 << 18, 1 << 20]
        for n in sizes {
            var inputs = [UInt32](repeating: 0, count: n * 2)
            var rng: UInt64 = 0xDEAD_BEEF_1234_5678
            for i in 0..<inputs.count {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                let v = UInt32(truncatingIfNeeded: rng >> 32)
                inputs[i] = v % M31.P
            }

            // Warmup
            let _ = try m31Engine.evaluate(program: compiled, inputs: inputs, numRows: n)

            let iters = n <= (1 << 16) ? 10 : 3
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0..<iters {
                let _ = try m31Engine.evaluate(program: compiled, inputs: inputs, numRows: n)
            }
            let gpuMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0 / Double(iters)

            let throughput = Double(n) * Double(compiled.numCols) / (gpuMs / 1000.0) / 1e6
            let line = String(format: "  n=2^%d (%dk rows): GPU %.3fms [%.1f M cells/s]",
                              Int(log2(Double(n))), n / 1024, gpuMs, throughput)
            fputs(line + "\n", stderr)
        }
    } catch {
        fputs("  Error in generic M31 trace bench: \(error)\n", stderr)
    }

    // --- Test 6: Linear recurrence (3-state) ---
    do {
        fputs("\n--- Linear Recurrence (3-wide state, T^row * state0) ---\n", stderr)
        // Example: 3-state recurrence with constant matrix
        let T: [[M31]] = [
            [M31(v: 0), M31(v: 1), M31(v: 0)],
            [M31(v: 0), M31(v: 0), M31(v: 1)],
            [M31(v: 1), M31(v: 1), M31(v: 1)]
        ]
        let s0: [M31] = [M31(v: 1), M31(v: 0), M31(v: 0)]

        let sizes = [1 << 14, 1 << 16, 1 << 18, 1 << 20]
        for n in sizes {
            // Correctness on small size
            if n == sizes[0] {
                let gpuResult = try m31Engine.generateLinearRecurrence(
                    transferMatrix: T, initialState: s0, numRows: n
                )
                // CPU reference
                var state = s0
                var mismatches = 0
                for row in 0..<min(n, 256) {
                    for col in 0..<3 {
                        if gpuResult[row][col].v != state[col].v {
                            if mismatches < 3 {
                                fputs("  MISMATCH row=\(row) col=\(col): GPU=\(gpuResult[row][col].v) CPU=\(state[col].v)\n", stderr)
                            }
                            mismatches += 1
                        }
                    }
                    // Advance state: state = T * state
                    var next = [M31](repeating: M31.zero, count: 3)
                    for i in 0..<3 {
                        var acc = M31.zero
                        for j in 0..<3 {
                            acc = m31Add(acc, m31Mul(T[i][j], state[j]))
                        }
                        next[i] = acc
                    }
                    state = next
                }
                if mismatches > 0 {
                    fputs("  FAIL: \(mismatches) mismatches in first 256 rows\n", stderr)
                } else {
                    fputs("  PASS: correctness verified (first 256 rows)\n", stderr)
                }
            }

            // Warmup
            let _ = try m31Engine.generateLinearRecurrence(
                transferMatrix: T, initialState: s0, numRows: n
            )

            let iters = n <= (1 << 16) ? 10 : 3
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0..<iters {
                let _ = try m31Engine.generateLinearRecurrence(
                    transferMatrix: T, initialState: s0, numRows: n
                )
            }
            let gpuMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0 / Double(iters)

            let throughput = Double(n) * 3.0 / (gpuMs / 1000.0) / 1e6
            let line = String(format: "  n=2^%d (%dk rows): GPU %.3fms [%.1f M cells/s]",
                              Int(log2(Double(n))), n / 1024, gpuMs, throughput)
            fputs(line + "\n", stderr)
        }
    } catch {
        fputs("  Error in linear recurrence bench: \(error)\n", stderr)
    }

    fputs("\n", stderr)
}
