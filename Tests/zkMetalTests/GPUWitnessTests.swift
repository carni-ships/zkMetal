// GPUWitnessTests — Tests for GPU-accelerated witness generation engine
//
// Validates correctness of witness computation for various circuit types,
// including cross-validation between GPU and CPU paths.

import Foundation
import Metal
@testable import zkMetal

public func runGPUWitnessTests() {
    suite("GPUWitness")

    guard let _ = MTLCreateSystemDefaultDevice() else {
        print("  [SKIP] No Metal device available")
        return
    }

    guard let engine = try? GPUWitnessEngine() else {
        print("  [SKIP] Failed to create GPUWitnessEngine")
        return
    }

    // Save original threshold for restoration
    let savedThreshold = engine.cpuFallbackThreshold

    // ================================================================
    // MARK: - Simple circuit: a * b + c
    // ================================================================
    suite("GPUWitness — Simple a*b+c")

    do {
        // Wire layout: 0=a, 1=b, 2=c, 3=a*b, 4=a*b+c
        let gates = [
            WitnessGate(op: .mul, leftIdx: 0, rightIdx: 1, outIdx: 3),
            WitnessGate(op: .add, leftIdx: 3, rightIdx: 2, outIdx: 4),
        ]
        let circuit = WitnessCircuit(numWires: 5, numInputs: 3, gates: gates)

        // a=3, b=5, c=7 -> a*b=15, a*b+c=22
        let a = frFromInt(3)
        let b = frFromInt(5)
        let c = frFromInt(7)

        // Force CPU path
        engine.cpuFallbackThreshold = 999999
        let result = engine.generateWitness(circuit: circuit, inputs: [a, b, c])

        expectEqual(result[0], a, "wire 0 = a")
        expectEqual(result[1], b, "wire 1 = b")
        expectEqual(result[2], c, "wire 2 = c")
        expectEqual(result[3], frFromInt(15), "wire 3 = a*b = 15")
        expectEqual(result[4], frFromInt(22), "wire 4 = a*b+c = 22")

        // Force GPU path (set threshold to 0)
        engine.cpuFallbackThreshold = 0
        let gpuResult = engine.generateWitness(circuit: circuit, inputs: [a, b, c])

        expectEqual(gpuResult[3], frFromInt(15), "GPU: wire 3 = a*b = 15")
        expectEqual(gpuResult[4], frFromInt(22), "GPU: wire 4 = a*b+c = 22")
    }

    // ================================================================
    // MARK: - Constant and copy operations
    // ================================================================
    suite("GPUWitness — Constant & Copy")

    do {
        // Wire: 0=input, 1=constant(42), 2=copy(0), 3=input+constant
        let gates = [
            WitnessGate(op: .constant(frFromInt(42)), leftIdx: 0, rightIdx: 0, outIdx: 1),
            WitnessGate(op: .copy, leftIdx: 0, rightIdx: 0, outIdx: 2),
            WitnessGate(op: .add, leftIdx: 0, rightIdx: 1, outIdx: 3),
        ]
        let circuit = WitnessCircuit(numWires: 4, numInputs: 1, gates: gates)

        let input = frFromInt(10)
        engine.cpuFallbackThreshold = 999999
        let result = engine.generateWitness(circuit: circuit, inputs: [input])

        expectEqual(result[1], frFromInt(42), "constant gate = 42")
        expectEqual(result[2], input, "copy gate = input")
        expectEqual(result[3], frFromInt(52), "input + 42 = 52")
    }

    // ================================================================
    // MARK: - Linear combination
    // ================================================================
    suite("GPUWitness — Linear Combination")

    do {
        // Wire: 0=a, 1=b, 2=qL*a + qR*b + qC
        // qL=2, qR=3, qC=5: 2*a + 3*b + 5
        let qL = frFromInt(2)
        let qR = frFromInt(3)
        let qC = frFromInt(5)
        let gates = [
            WitnessGate(op: .linearCombination(qL: qL, qR: qR, qC: qC),
                         leftIdx: 0, rightIdx: 1, outIdx: 2),
        ]
        let circuit = WitnessCircuit(numWires: 3, numInputs: 2, gates: gates)

        let a = frFromInt(10)
        let b = frFromInt(7)
        // Expected: 2*10 + 3*7 + 5 = 20 + 21 + 5 = 46

        // CPU path
        engine.cpuFallbackThreshold = 999999
        let cpuResult = engine.generateWitness(circuit: circuit, inputs: [a, b])
        expectEqual(cpuResult[2], frFromInt(46), "CPU: 2*10 + 3*7 + 5 = 46")

        // GPU path
        engine.cpuFallbackThreshold = 0
        let gpuResult = engine.generateWitness(circuit: circuit, inputs: [a, b])
        expectEqual(gpuResult[2], frFromInt(46), "GPU: 2*10 + 3*7 + 5 = 46")
    }

    // ================================================================
    // MARK: - Poseidon2 hash circuit
    // ================================================================
    suite("GPUWitness — Poseidon2 Round")

    do {
        // Wire: 0,1,2 = input state, 3,4,5 = output state after poseidon2 round
        // The poseidon2 op reads from leftIdx (state base), writes to outIdx (state base)
        // First copy input state to output position, then apply poseidon2
        let gates = [
            WitnessGate(op: .copy, leftIdx: 0, rightIdx: 0, outIdx: 3),
            WitnessGate(op: .copy, leftIdx: 1, rightIdx: 0, outIdx: 4),
            WitnessGate(op: .copy, leftIdx: 2, rightIdx: 0, outIdx: 5),
            WitnessGate(op: .poseidon2, leftIdx: 3, rightIdx: 0, outIdx: 3),
        ]
        let circuit = WitnessCircuit(numWires: 6, numInputs: 3, gates: gates)

        let s0 = frFromInt(1)
        let s1 = frFromInt(2)
        let s2 = frFromInt(3)

        // CPU reference
        engine.cpuFallbackThreshold = 999999
        let cpuResult = engine.generateWitness(circuit: circuit, inputs: [s0, s1, s2])

        // GPU
        engine.cpuFallbackThreshold = 0
        let gpuResult = engine.generateWitness(circuit: circuit, inputs: [s0, s1, s2])

        expectEqual(gpuResult[3], cpuResult[3], "Poseidon2 s0 GPU == CPU")
        expectEqual(gpuResult[4], cpuResult[4], "Poseidon2 s1 GPU == CPU")
        expectEqual(gpuResult[5], cpuResult[5], "Poseidon2 s2 GPU == CPU")

        // Verify output is not trivially zero or same as input (S-box should change things)
        expect(cpuResult[3] != s0 || cpuResult[4] != s1 || cpuResult[5] != s2,
               "Poseidon2 output differs from input")
    }

    // ================================================================
    // MARK: - Layer dependency ordering
    // ================================================================
    suite("GPUWitness — Layer Dependencies")

    do {
        // Chain: w3 = w0 + w1, w4 = w3 * w2, w5 = w4 + w0
        // Layer 0: gate 0 (depends only on inputs)
        // Layer 1: gate 1 (depends on w3 from layer 0)
        // Layer 2: gate 2 (depends on w4 from layer 1)
        let gates = [
            WitnessGate(op: .add, leftIdx: 0, rightIdx: 1, outIdx: 3),
            WitnessGate(op: .mul, leftIdx: 3, rightIdx: 2, outIdx: 4),
            WitnessGate(op: .add, leftIdx: 4, rightIdx: 0, outIdx: 5),
        ]
        let circuit = WitnessCircuit(numWires: 6, numInputs: 3, gates: gates)

        // Verify layer assignments
        expectEqual(circuit.layers[0], 0, "gate 0 in layer 0")
        expectEqual(circuit.layers[1], 1, "gate 1 in layer 1")
        expectEqual(circuit.layers[2], 2, "gate 2 in layer 2")
        expectEqual(circuit.numLayers, 3, "3 dependency layers")

        // Compute: a=4, b=6, c=2 -> w3=10, w4=20, w5=24
        let a = frFromInt(4)
        let b = frFromInt(6)
        let c = frFromInt(2)

        engine.cpuFallbackThreshold = 999999
        let result = engine.generateWitness(circuit: circuit, inputs: [a, b, c])
        expectEqual(result[3], frFromInt(10), "w3 = a+b = 10")
        expectEqual(result[4], frFromInt(20), "w4 = w3*c = 20")
        expectEqual(result[5], frFromInt(24), "w5 = w4+a = 24")
    }

    // ================================================================
    // MARK: - Parallel gates in same layer
    // ================================================================
    suite("GPUWitness — Parallel Layer")

    do {
        // Two independent gates in the same layer:
        // w2 = w0 + w1
        // w3 = w0 * w1
        // Both only depend on inputs -> same layer
        let gates = [
            WitnessGate(op: .add, leftIdx: 0, rightIdx: 1, outIdx: 2),
            WitnessGate(op: .mul, leftIdx: 0, rightIdx: 1, outIdx: 3),
        ]
        let circuit = WitnessCircuit(numWires: 4, numInputs: 2, gates: gates)

        expectEqual(circuit.layers[0], 0, "both gates in layer 0")
        expectEqual(circuit.layers[1], 0, "both gates in layer 0")

        let a = frFromInt(3)
        let b = frFromInt(7)
        engine.cpuFallbackThreshold = 999999
        let result = engine.generateWitness(circuit: circuit, inputs: [a, b])
        expectEqual(result[2], frFromInt(10), "w2 = 3+7 = 10")
        expectEqual(result[3], frFromInt(21), "w3 = 3*7 = 21")
    }

    // ================================================================
    // MARK: - Large circuit: 2^14 gates mixed operations
    // ================================================================
    suite("GPUWitness — Large Circuit (2^14 gates)")

    do {
        let numGates = 1 << 14  // 16384 gates
        let numInputs = 4
        let numWires = numInputs + numGates

        var gates = [WitnessGate]()
        gates.reserveCapacity(numGates)

        // Build a circuit with mixed add/mul operations
        // Each gate reads from two earlier wires
        for i in 0..<numGates {
            let outIdx = numInputs + i
            let leftIdx: Int
            let rightIdx: Int

            if i < numInputs {
                // First few gates depend directly on inputs
                leftIdx = i % numInputs
                rightIdx = (i + 1) % numInputs
            } else {
                // Later gates depend on earlier outputs
                leftIdx = numInputs + (i - numInputs) / 2
                rightIdx = numInputs + max(0, (i - numInputs) / 2 + 1)
                // Clamp to valid range
            }

            let clampedLeft = min(leftIdx, outIdx - 1)
            let clampedRight = min(rightIdx, outIdx - 1)

            if i % 2 == 0 {
                gates.append(WitnessGate(op: .add, leftIdx: clampedLeft, rightIdx: clampedRight, outIdx: outIdx))
            } else {
                gates.append(WitnessGate(op: .mul, leftIdx: clampedLeft, rightIdx: clampedRight, outIdx: outIdx))
            }
        }

        let circuit = WitnessCircuit(numWires: numWires, numInputs: numInputs, gates: gates)

        let inputs = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]

        // CPU reference
        engine.cpuFallbackThreshold = 999999
        let t0 = CFAbsoluteTimeGetCurrent()
        let cpuResult = engine.generateWitness(circuit: circuit, inputs: inputs)
        let cpuTime = CFAbsoluteTimeGetCurrent() - t0

        // GPU
        engine.cpuFallbackThreshold = 0
        let t1 = CFAbsoluteTimeGetCurrent()
        let gpuResult = engine.generateWitness(circuit: circuit, inputs: inputs)
        let gpuTime = CFAbsoluteTimeGetCurrent() - t1

        // Cross-validate: check all wires match
        var mismatches = 0
        for i in 0..<numWires {
            if cpuResult[i] != gpuResult[i] {
                mismatches += 1
                if mismatches <= 3 {
                    print("  [DETAIL] Mismatch at wire \(i)")
                }
            }
        }
        expect(mismatches == 0, "Large circuit: \(mismatches) mismatches between GPU and CPU")

        print(String(format: "  [INFO] Large circuit (%d gates): CPU %.1fms, GPU %.1fms",
                      numGates, cpuTime * 1000, gpuTime * 1000))
    }

    // ================================================================
    // MARK: - Cross-validate GPU vs CPU for mixed circuit
    // ================================================================
    suite("GPUWitness — GPU vs CPU Cross-Validation")

    do {
        // Build a circuit with all operation types
        // Wires: 0=a, 1=b, 2=c
        // 3 = a+b, 4 = a*b, 5 = 2*a + 3*b + 7, 6 = const(100), 7 = copy(3)
        // 8 = 4+5, 9 = 7*6, 10 = 8+9
        let qL = frFromInt(2)
        let qR = frFromInt(3)
        let qC = frFromInt(7)

        let gates = [
            WitnessGate(op: .add, leftIdx: 0, rightIdx: 1, outIdx: 3),
            WitnessGate(op: .mul, leftIdx: 0, rightIdx: 1, outIdx: 4),
            WitnessGate(op: .linearCombination(qL: qL, qR: qR, qC: qC),
                         leftIdx: 0, rightIdx: 1, outIdx: 5),
            WitnessGate(op: .constant(frFromInt(100)), leftIdx: 0, rightIdx: 0, outIdx: 6),
            WitnessGate(op: .copy, leftIdx: 3, rightIdx: 0, outIdx: 7),
            WitnessGate(op: .add, leftIdx: 4, rightIdx: 5, outIdx: 8),
            WitnessGate(op: .mul, leftIdx: 7, rightIdx: 6, outIdx: 9),
            WitnessGate(op: .add, leftIdx: 8, rightIdx: 9, outIdx: 10),
        ]
        let circuit = WitnessCircuit(numWires: 11, numInputs: 3, gates: gates)

        let a = frFromInt(5)
        let b = frFromInt(3)
        let c = frFromInt(1)

        // CPU
        engine.cpuFallbackThreshold = 999999
        let cpuResult = engine.generateWitness(circuit: circuit, inputs: [a, b, c])

        // GPU
        engine.cpuFallbackThreshold = 0
        let gpuResult = engine.generateWitness(circuit: circuit, inputs: [a, b, c])

        // Verify known values
        // w3 = 5+3 = 8
        expectEqual(cpuResult[3], frFromInt(8), "CPU: a+b = 8")
        // w4 = 5*3 = 15
        expectEqual(cpuResult[4], frFromInt(15), "CPU: a*b = 15")
        // w5 = 2*5 + 3*3 + 7 = 10 + 9 + 7 = 26
        expectEqual(cpuResult[5], frFromInt(26), "CPU: 2a+3b+7 = 26")
        // w6 = 100
        expectEqual(cpuResult[6], frFromInt(100), "CPU: const = 100")
        // w7 = copy(w3) = 8
        expectEqual(cpuResult[7], frFromInt(8), "CPU: copy(w3) = 8")
        // w8 = w4+w5 = 15+26 = 41
        expectEqual(cpuResult[8], frFromInt(41), "CPU: w4+w5 = 41")
        // w9 = w7*w6 = 8*100 = 800
        expectEqual(cpuResult[9], frFromInt(800), "CPU: w7*w6 = 800")
        // w10 = w8+w9 = 41+800 = 841
        expectEqual(cpuResult[10], frFromInt(841), "CPU: w8+w9 = 841")

        // Cross-validate all wires
        var allMatch = true
        for i in 0..<11 {
            if cpuResult[i] != gpuResult[i] {
                allMatch = false
                print("  [FAIL] Wire \(i) mismatch: CPU vs GPU")
            }
        }
        expect(allMatch, "All wires match between GPU and CPU")
    }

    // Restore threshold
    engine.cpuFallbackThreshold = savedThreshold
}
