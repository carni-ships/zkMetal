// Jolt VM Benchmark — Prove/verify instruction execution via Lasso lookups
import zkMetal
import Foundation

public func runJoltBench() {
    fputs("\n=== Jolt VM (Lasso-based zkVM) ===\n", stderr)

    // --- Correctness Tests ---
    fputs("\n--- Correctness Tests ---\n", stderr)

    do {
        let engine = try JoltEngine()

        // Test 1: Simple ADD program (2 instructions, minimum for Lasso)
        let addProg = [
            JoltInstruction(op: .add, rs1: 0, rs2: 0, rd: 1),
            JoltInstruction(op: .add, rs1: 0, rs2: 0, rd: 0),
        ]
        let addTrace = joltExecute(program: addProg)
        let addProof = try engine.prove(trace: addTrace)
        let addValid = engine.verify(proof: addProof, program: addProg)
        fputs("  Simple ADD (2 instrs): \(addValid ? "PASS" : "FAIL")\n", stderr)

        // Test 2: Mixed bitwise ops
        let bitwiseProg: [JoltInstruction] = [
            JoltInstruction(op: .xor_, rs1: 0, rs2: 1, rd: 2),
            JoltInstruction(op: .and_, rs1: 0, rs2: 1, rd: 3),
            JoltInstruction(op: .or_,  rs1: 2, rs2: 3, rd: 4),
            JoltInstruction(op: .xor_, rs1: 4, rs2: 0, rd: 5),
        ]
        let bitwiseTrace = joltExecute(program: bitwiseProg)
        let bitwiseProof = try engine.prove(trace: bitwiseTrace)
        let bitwiseValid = engine.verify(proof: bitwiseProof, program: bitwiseProg)
        fputs("  Mixed bitwise (4 instrs): \(bitwiseValid ? "PASS" : "FAIL")\n", stderr)

        // Test 3: Lasso-verified ops (LT, SHR, SHL, SUB) + algebraic fallback (MUL, EQ)
        let mixedProg: [JoltInstruction] = [
            JoltInstruction(op: .mul, rs1: 0, rs2: 1, rd: 2),
            JoltInstruction(op: .eq,  rs1: 0, rs2: 0, rd: 3),
            JoltInstruction(op: .lt,  rs1: 0, rs2: 1, rd: 4),
            JoltInstruction(op: .shr, rs1: 0, rs2: 1, rd: 5),
            JoltInstruction(op: .shl, rs1: 0, rs2: 1, rd: 6),
            JoltInstruction(op: .sub, rs1: 0, rs2: 1, rd: 7),
        ]
        let mixedTrace = joltExecute(program: mixedProg)
        let mixedProof = try engine.prove(trace: mixedTrace)
        let mixedValid = engine.verify(proof: mixedProof, program: mixedProg)
        fputs("  Lasso+algebraic ops (6 instrs): \(mixedValid ? "PASS" : "FAIL")\n", stderr)

        // Test 4: Random mixed program
        let randomProg = joltRandomProgram(count: 16, numRegisters: 8, seed: 42)
        let randomTrace = joltExecute(program: randomProg, numRegisters: 8)
        let randomProof = try engine.prove(trace: randomTrace)
        let randomValid = engine.verify(proof: randomProof, program: randomProg)
        fputs("  Random mixed (16 instrs, 8 regs): \(randomValid ? "PASS" : "FAIL")\n", stderr)

        // Test 5: Verify tampering is detected
        let tamperedOps = [(frFromInt(999), frFromInt(0), frFromInt(0))] +
                          Array(randomProof.operandCommitments.dropFirst())
        let tamperedProof = JoltProof(
            numInstructions: randomProof.numInstructions,
            opcodeProofs: randomProof.opcodeProofs,
            operandCommitments: tamperedOps
        )
        let rejected = !engine.verify(proof: tamperedProof, program: randomProg)
        fputs("  Reject tampered operands: \(rejected ? "PASS" : "FAIL")\n", stderr)

    } catch {
        fputs("  ERROR: \(error)\n", stderr)
    }

    // --- Performance Benchmarks ---
    fputs("\n--- Performance: Jolt VM prove/verify ---\n", stderr)
    do {
        let engine = try JoltEngine()

        for logN in [8, 10, 12] {
            let n = 1 << logN
            let program = joltRandomProgram(count: n, numRegisters: 32, seed: UInt64(logN) * 1000)

            // Execute
            let execT0 = CFAbsoluteTimeGetCurrent()
            let trace = joltExecute(program: program, numRegisters: 32)
            let execTime = (CFAbsoluteTimeGetCurrent() - execT0) * 1000

            // Warmup
            let _ = try engine.prove(trace: trace)

            // Timed prove
            let runs = 3
            var proveTimes = [Double]()
            var proof: JoltProof!
            for _ in 0..<runs {
                let t0 = CFAbsoluteTimeGetCurrent()
                proof = try engine.prove(trace: trace)
                let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                proveTimes.append(elapsed)
            }
            proveTimes.sort()
            let proveMedian = proveTimes[runs / 2]

            // Timed verify
            var verifyTimes = [Double]()
            for _ in 0..<runs {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = engine.verify(proof: proof, program: program)
                let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                verifyTimes.append(elapsed)
            }
            verifyTimes.sort()
            let verifyMedian = verifyTimes[runs / 2]

            // Count ops by type
            var opCounts = [JoltOp: Int]()
            for instr in program {
                opCounts[instr.op, default: 0] += 1
            }
            var lassoCount = 0
            for lop: JoltOp in [.add, .sub, .and_, .or_, .xor_, .shl, .shr, .lt] {
                lassoCount += opCounts[lop] ?? 0
            }
            let algebraicCount = n - lassoCount

            fputs(String(format: "  2^%-2d = %5d instrs: exec %.1fms, prove %.1fms, verify %.1fms (lasso=%d, algebraic=%d)\n",
                        logN, n, execTime, proveMedian, verifyMedian, lassoCount, algebraicCount), stderr)
        }

    } catch {
        fputs("  ERROR: \(error)\n", stderr)
    }

    // --- Throughput summary ---
    fputs("\n--- Throughput ---\n", stderr)
    do {
        let engine = try JoltEngine()
        let n = 1 << 10
        let program = joltRandomProgram(count: n, numRegisters: 32, seed: 7777)
        let trace = joltExecute(program: program, numRegisters: 32)

        // Warmup
        let _ = try engine.prove(trace: trace)

        let t0 = CFAbsoluteTimeGetCurrent()
        let _ = try engine.prove(trace: trace)
        let elapsed = CFAbsoluteTimeGetCurrent() - t0

        let throughput = Double(n) / elapsed
        fputs(String(format: "  %d instructions in %.1fms = %.0f instrs/sec\n",
                    n, elapsed * 1000, throughput), stderr)
    } catch {
        fputs("  ERROR: \(error)\n", stderr)
    }
}
