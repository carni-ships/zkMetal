// GPUConstraintEvalEngine Tests — GPU vs CPU constraint evaluation correctness
//
// Tests the gate-descriptor-based GPU constraint evaluation engine against
// CPU reference implementations for all supported gate types.

import Foundation
@testable import zkMetal

public func runGPUConstraintEvalTests() {
    suite("GPUConstraintEval")

    testMulGate()
    testAddGate()
    testBoolGate()
    testArithmeticGate()
    testRangeDecompGate()
    testMultipleGates()
    testSatisfiedConstraints()
    testQuotientComputation()
    testLargeDomain()
}

// MARK: - Multiplication Gate: a * b - c = 0

private func testMulGate() {
    do {
        let engine = try GPUConstraintEvalEngine()
        let domainSize = 64

        // 3 columns: a, b, c where c = a * b
        var colA = [Fr]()
        var colB = [Fr]()
        var colC = [Fr]()
        for i in 0..<domainSize {
            let a = frFromInt(UInt64(i + 1))
            let b = frFromInt(UInt64(i + 2))
            let c = frMul(a, b)
            colA.append(a)
            colB.append(b)
            colC.append(c)
        }

        let bufA = try engine.createColumnBuffer(colA)
        let bufB = try engine.createColumnBuffer(colB)
        let bufC = try engine.createColumnBuffer(colC)

        let gates = [GateDescriptor.mul(colA: 0, colB: 1, colC: 2)]

        let output = try engine.evaluateConstraints(
            trace: [bufA, bufB, bufC],
            gates: gates,
            constants: [],
            domainSize: domainSize
        )

        // CPU reference
        let cpuOutput = engine.evaluateCPU(
            trace: [colA, colB, colC],
            gates: gates,
            constants: [],
            domainSize: domainSize
        )

        // All should be zero (constraints satisfied)
        let gpuResults = engine.readBuffer(output, count: domainSize)
        var allZero = true
        for i in 0..<domainSize {
            if !gpuResults[i].isZero {
                allZero = false
                break
            }
        }
        expect(allZero, "mul gate: all satisfied constraints should be zero on GPU")

        // CPU should also be zero
        var cpuAllZero = true
        for i in 0..<domainSize {
            if !cpuOutput[i].isZero {
                cpuAllZero = false
                break
            }
        }
        expect(cpuAllZero, "mul gate: all satisfied constraints should be zero on CPU")
        print("  [OK] mul gate (satisfied, \(domainSize) points)")
    } catch {
        print("  [FAIL] mul gate: \(error)")
        expect(false, "mul gate threw: \(error)")
    }
}

// MARK: - Addition Gate: a + b - c = 0

private func testAddGate() {
    do {
        let engine = try GPUConstraintEvalEngine()
        let domainSize = 32

        var colA = [Fr](), colB = [Fr](), colC = [Fr]()
        for i in 0..<domainSize {
            let a = frFromInt(UInt64(i * 3 + 7))
            let b = frFromInt(UInt64(i * 5 + 11))
            colA.append(a)
            colB.append(b)
            colC.append(frAdd(a, b))
        }

        let bufA = try engine.createColumnBuffer(colA)
        let bufB = try engine.createColumnBuffer(colB)
        let bufC = try engine.createColumnBuffer(colC)

        let gates = [GateDescriptor.add(colA: 0, colB: 1, colC: 2)]

        let output = try engine.evaluateConstraints(
            trace: [bufA, bufB, bufC],
            gates: gates, constants: [], domainSize: domainSize
        )

        let gpuResults = engine.readBuffer(output, count: domainSize)
        let allZero = gpuResults.allSatisfy { $0.isZero }
        expect(allZero, "add gate: satisfied constraints should be zero")
        print("  [OK] add gate (satisfied, \(domainSize) points)")
    } catch {
        print("  [FAIL] add gate: \(error)")
        expect(false, "add gate threw: \(error)")
    }
}

// MARK: - Boolean Gate: a * (1 - a) = 0

private func testBoolGate() {
    do {
        let engine = try GPUConstraintEvalEngine()
        let domainSize = 64

        // First half: valid booleans (0 or 1), second half: invalid (2, 3, ...)
        var col = [Fr]()
        for i in 0..<domainSize {
            if i < domainSize / 2 {
                col.append(frFromInt(UInt64(i % 2)))  // 0 or 1
            } else {
                col.append(frFromInt(UInt64(i)))  // non-boolean values
            }
        }

        let buf = try engine.createColumnBuffer(col)
        let gates = [GateDescriptor.bool(col: 0)]

        let output = try engine.evaluateConstraints(
            trace: [buf], gates: gates, constants: [], domainSize: domainSize
        )

        let gpuResults = engine.readBuffer(output, count: domainSize)
        let cpuResults = engine.evaluateCPU(
            trace: [col], gates: gates, constants: [], domainSize: domainSize
        )

        // First half should be zero (valid booleans)
        var firstHalfOK = true
        for i in 0..<(domainSize / 2) {
            if !gpuResults[i].isZero {
                firstHalfOK = false
                break
            }
        }
        expect(firstHalfOK, "bool gate: valid booleans should satisfy constraint")

        // Second half should be non-zero (except where i=0 or i=1 wraps)
        var someNonZero = false
        for i in (domainSize / 2)..<domainSize {
            if !gpuResults[i].isZero {
                someNonZero = true
                break
            }
        }
        expect(someNonZero, "bool gate: non-boolean values should violate constraint")

        // GPU and CPU should match
        var gpuCpuMatch = true
        for i in 0..<domainSize {
            if gpuResults[i] != cpuResults[i] {
                gpuCpuMatch = false
                break
            }
        }
        expect(gpuCpuMatch, "bool gate: GPU and CPU results should match")
        print("  [OK] bool gate (GPU==CPU, \(domainSize) points)")
    } catch {
        print("  [FAIL] bool gate: \(error)")
        expect(false, "bool gate threw: \(error)")
    }
}

// MARK: - Arithmetic Gate: qL*a + qR*b + qO*c + qM*a*b + qC = 0

private func testArithmeticGate() {
    do {
        let engine = try GPUConstraintEvalEngine()
        let domainSize = 32

        // Standard Plonk addition: 1*a + 1*b + (-1)*c + 0*a*b + 0 = 0
        // i.e., a + b - c = 0
        let qL = Fr.one
        let qR = Fr.one
        let qO = frSub(Fr.zero, Fr.one)  // -1
        let qM = Fr.zero
        let qC = Fr.zero
        let constants = [qL, qR, qO, qM, qC]

        var colA = [Fr](), colB = [Fr](), colC = [Fr]()
        for i in 0..<domainSize {
            let a = frFromInt(UInt64(i + 10))
            let b = frFromInt(UInt64(i + 20))
            colA.append(a)
            colB.append(b)
            colC.append(frAdd(a, b))
        }

        let bufA = try engine.createColumnBuffer(colA)
        let bufB = try engine.createColumnBuffer(colB)
        let bufC = try engine.createColumnBuffer(colC)

        let gates = [GateDescriptor.arithmetic(colA: 0, colB: 1, colC: 2,
                                                constantsBaseIdx: 0)]

        let output = try engine.evaluateConstraints(
            trace: [bufA, bufB, bufC],
            gates: gates, constants: constants, domainSize: domainSize
        )

        let gpuResults = engine.readBuffer(output, count: domainSize)
        let cpuResults = engine.evaluateCPU(
            trace: [colA, colB, colC],
            gates: gates, constants: constants, domainSize: domainSize
        )

        let allZero = gpuResults.allSatisfy { $0.isZero }
        expect(allZero, "arithmetic gate: satisfied addition should be zero")

        var gpuCpuMatch = true
        for i in 0..<domainSize {
            if gpuResults[i] != cpuResults[i] {
                gpuCpuMatch = false
                break
            }
        }
        expect(gpuCpuMatch, "arithmetic gate: GPU and CPU should match")
        print("  [OK] arithmetic gate (satisfied, GPU==CPU)")
    } catch {
        print("  [FAIL] arithmetic gate: \(error)")
        expect(false, "arithmetic gate threw: \(error)")
    }
}

// MARK: - Range Decomposition Gate

private func testRangeDecompGate() {
    do {
        let engine = try GPUConstraintEvalEngine()
        let domainSize = 16
        let numBits = 8

        // Column 0: value, columns 1..8: bit decomposition
        var colValue = [Fr]()
        var bitCols = [[Fr]](repeating: [Fr](), count: numBits)

        for i in 0..<domainSize {
            let val = UInt64(i)
            colValue.append(frFromInt(val))
            for b in 0..<numBits {
                let bit = (val >> UInt64(b)) & 1
                bitCols[b].append(frFromInt(bit))
            }
        }

        var buffers = [try engine.createColumnBuffer(colValue)]
        for b in 0..<numBits {
            buffers.append(try engine.createColumnBuffer(bitCols[b]))
        }

        let gates = [GateDescriptor.rangeDecomp(valueCol: 0, firstBitCol: 1, numBits: UInt32(numBits))]

        let output = try engine.evaluateConstraints(
            trace: buffers, gates: gates, constants: [], domainSize: domainSize
        )

        var trace = [colValue]
        trace.append(contentsOf: bitCols)

        let gpuResults = engine.readBuffer(output, count: domainSize)
        let cpuResults = engine.evaluateCPU(
            trace: trace, gates: gates, constants: [], domainSize: domainSize
        )

        let allZero = gpuResults.allSatisfy { $0.isZero }
        expect(allZero, "range decomp: satisfied decomposition should be zero")

        var gpuCpuMatch = true
        for i in 0..<domainSize {
            if gpuResults[i] != cpuResults[i] {
                gpuCpuMatch = false
                break
            }
        }
        expect(gpuCpuMatch, "range decomp: GPU and CPU should match")
        print("  [OK] range decomposition gate (\(numBits) bits, \(domainSize) points)")
    } catch {
        print("  [FAIL] range decomp: \(error)")
        expect(false, "range decomp threw: \(error)")
    }
}

// MARK: - Multiple Gates Combined

private func testMultipleGates() {
    do {
        let engine = try GPUConstraintEvalEngine()
        let domainSize = 32

        // 4 columns: a, b, c, d
        // Gate 0: a * b - c = 0  (mul)
        // Gate 1: a + b - d = 0  (add, using d instead of c)
        // But we only have 4 columns, so let d = a + b
        var colA = [Fr](), colB = [Fr](), colC = [Fr](), colD = [Fr]()
        for i in 0..<domainSize {
            let a = frFromInt(UInt64(i + 3))
            let b = frFromInt(UInt64(i + 5))
            colA.append(a)
            colB.append(b)
            colC.append(frMul(a, b))
            colD.append(frAdd(a, b))
        }

        let bufA = try engine.createColumnBuffer(colA)
        let bufB = try engine.createColumnBuffer(colB)
        let bufC = try engine.createColumnBuffer(colC)
        let bufD = try engine.createColumnBuffer(colD)

        let gates = [
            GateDescriptor.mul(colA: 0, colB: 1, colC: 2),
            GateDescriptor.add(colA: 0, colB: 1, colC: 3),
        ]

        let output = try engine.evaluateConstraints(
            trace: [bufA, bufB, bufC, bufD],
            gates: gates, constants: [], domainSize: domainSize
        )

        let gpuResults = engine.readBuffer(output, count: domainSize * 2)
        let cpuResults = engine.evaluateCPU(
            trace: [colA, colB, colC, colD],
            gates: gates, constants: [], domainSize: domainSize
        )

        var allZero = true
        for i in 0..<(domainSize * 2) {
            if !gpuResults[i].isZero {
                allZero = false
                break
            }
        }
        expect(allZero, "multiple gates: all satisfied")

        var match = true
        for i in 0..<(domainSize * 2) {
            if gpuResults[i] != cpuResults[i] {
                match = false
                break
            }
        }
        expect(match, "multiple gates: GPU==CPU")
        print("  [OK] multiple gates (2 gates, \(domainSize) points)")
    } catch {
        print("  [FAIL] multiple gates: \(error)")
        expect(false, "multiple gates threw: \(error)")
    }
}

// MARK: - Verify All-Zero on Satisfied System

private func testSatisfiedConstraints() {
    do {
        let engine = try GPUConstraintEvalEngine()
        let domainSize = 128

        // Arithmetic gate: 2*a + 3*b - c = 0, with c = 2a + 3b
        let two = frFromInt(2)
        let three = frFromInt(3)
        let negOne = frSub(Fr.zero, Fr.one)
        let constants: [Fr] = [two, three, negOne, Fr.zero, Fr.zero]

        var colA = [Fr](), colB = [Fr](), colC = [Fr]()
        for i in 0..<domainSize {
            let a = frFromInt(UInt64(i * 7 + 1))
            let b = frFromInt(UInt64(i * 11 + 2))
            colA.append(a)
            colB.append(b)
            // c = 2*a + 3*b
            colC.append(frAdd(frMul(two, a), frMul(three, b)))
        }

        let bufA = try engine.createColumnBuffer(colA)
        let bufB = try engine.createColumnBuffer(colB)
        let bufC = try engine.createColumnBuffer(colC)

        let gates = [GateDescriptor.arithmetic(colA: 0, colB: 1, colC: 2,
                                                constantsBaseIdx: 0)]

        let output = try engine.evaluateConstraints(
            trace: [bufA, bufB, bufC],
            gates: gates, constants: constants, domainSize: domainSize
        )

        let gpuResults = engine.readBuffer(output, count: domainSize)
        let allZero = gpuResults.allSatisfy { $0.isZero }
        expect(allZero, "satisfied system: 2a+3b-c=0 should all be zero")
        print("  [OK] satisfied arithmetic (2a+3b=c, \(domainSize) points)")
    } catch {
        print("  [FAIL] satisfied constraints: \(error)")
        expect(false, "satisfied constraints threw: \(error)")
    }
}

// MARK: - Quotient Polynomial Computation

private func testQuotientComputation() {
    do {
        let engine = try GPUConstraintEvalEngine()
        let domainSize = 64

        // Build a simple mul gate with unsatisfied constraints so quotient is non-trivial
        var colA = [Fr](), colB = [Fr](), colC = [Fr]()
        for i in 0..<domainSize {
            let a = frFromInt(UInt64(i + 1))
            let b = frFromInt(UInt64(i + 2))
            // Intentionally wrong: c = a + b instead of a * b
            colA.append(a)
            colB.append(b)
            colC.append(frAdd(a, b))
        }

        let bufA = try engine.createColumnBuffer(colA)
        let bufB = try engine.createColumnBuffer(colB)
        let bufC = try engine.createColumnBuffer(colC)

        let gates = [GateDescriptor.mul(colA: 0, colB: 1, colC: 2)]

        // First get constraint evaluations
        let evalBuf = try engine.evaluateConstraints(
            trace: [bufA, bufB, bufC],
            gates: gates, constants: [], domainSize: domainSize
        )

        // Create dummy vanishing_inv (all ones — not a real vanishing polynomial,
        // just testing the kernel runs correctly)
        var vanishingInv = [Fr](repeating: Fr.one, count: domainSize)
        let vanBuf = try engine.createColumnBuffer(vanishingInv)

        // Alpha powers
        let alpha = frFromInt(7)
        let alphaBuf = try engine.createAlphaPowers(alpha: alpha, count: 1)

        // Compute quotient
        let quotientBuf = try engine.computeQuotient(
            constraintEvals: evalBuf,
            vanishingPoly: vanBuf,
            alphaPowers: alphaBuf,
            domainSize: domainSize,
            numGates: 1
        )

        let quotientResults = engine.readBuffer(quotientBuf, count: domainSize)

        // Since vanishing_inv = 1, quotient = alpha^0 * constraint_eval = constraint_eval
        let evalResults = engine.readBuffer(evalBuf, count: domainSize)

        var match = true
        for i in 0..<domainSize {
            if quotientResults[i] != evalResults[i] {
                match = false
                break
            }
        }
        expect(match, "quotient: with vanishing_inv=1, quotient should equal constraint evals")

        // Some should be non-zero (unsatisfied constraints)
        let someNonZero = quotientResults.contains { !$0.isZero }
        expect(someNonZero, "quotient: unsatisfied constraints produce non-zero quotient")

        print("  [OK] quotient computation (\(domainSize) points)")
    } catch {
        print("  [FAIL] quotient: \(error)")
        expect(false, "quotient threw: \(error)")
    }
}

// MARK: - Large Domain Test (Performance)

private func testLargeDomain() {
    do {
        let engine = try GPUConstraintEvalEngine()
        let domainSize = 1 << 14  // 16K points

        var colA = [Fr](), colB = [Fr](), colC = [Fr]()
        for i in 0..<domainSize {
            let a = frFromInt(UInt64(i + 1))
            let b = frFromInt(UInt64(i + 2))
            colA.append(a)
            colB.append(b)
            colC.append(frMul(a, b))
        }

        let bufA = try engine.createColumnBuffer(colA)
        let bufB = try engine.createColumnBuffer(colB)
        let bufC = try engine.createColumnBuffer(colC)

        let gates = [
            GateDescriptor.mul(colA: 0, colB: 1, colC: 2),
            GateDescriptor.add(colA: 0, colB: 1, colC: 2),  // intentionally wrong for variety
        ]

        let t0 = CFAbsoluteTimeGetCurrent()
        let output = try engine.evaluateConstraints(
            trace: [bufA, bufB, bufC],
            gates: gates, constants: [], domainSize: domainSize
        )
        let gpuTime = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0

        let t1 = CFAbsoluteTimeGetCurrent()
        let cpuOutput = engine.evaluateCPU(
            trace: [colA, colB, colC],
            gates: gates, constants: [], domainSize: domainSize
        )
        let cpuTime = (CFAbsoluteTimeGetCurrent() - t1) * 1000.0

        let gpuResults = engine.readBuffer(output, count: domainSize * 2)

        // Verify GPU==CPU
        var match = true
        for i in 0..<(domainSize * 2) {
            if gpuResults[i] != cpuOutput[i] {
                match = false
                break
            }
        }
        expect(match, "large domain: GPU and CPU should match for \(domainSize) points")

        let speedup = cpuTime / max(gpuTime, 0.001)
        print("  [OK] large domain: GPU \(String(format: "%.2f", gpuTime))ms"
              + " vs CPU \(String(format: "%.2f", cpuTime))ms"
              + " (\(String(format: "%.1f", speedup))x)")
    } catch {
        print("  [FAIL] large domain: \(error)")
        expect(false, "large domain threw: \(error)")
    }
}
