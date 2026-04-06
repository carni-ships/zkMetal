// Groth16GPUWitnessTests — Tests for GPU-accelerated R1CS witness generation
//
// Validates correctness of GPU witness solving against known values and
// cross-validates against the CPU WitnessSolver path. Tests range from
// tiny circuits (force CPU fallback) to larger chains (GPU dispatch).

import Foundation
import Metal
import zkMetal

public func runGroth16GPUWitnessTests() {
    suite("Groth16 GPU Witness")

    guard let _ = MTLCreateSystemDefaultDevice() else {
        print("  [SKIP] No Metal device available")
        return
    }

    guard let gpuWitness = try? Groth16GPUWitness() else {
        print("  [SKIP] Failed to create Groth16GPUWitness engine")
        return
    }

    // ================================================================
    // MARK: - Test 1: Simple multiply a * b = c
    // ================================================================
    do {
        let one = Fr.one
        var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

        // Constraint: w1 * w2 = w3
        aE.append(R1CSEntry(row: 0, col: 1, val: one))
        bE.append(R1CSEntry(row: 0, col: 2, val: one))
        cE.append(R1CSEntry(row: 0, col: 3, val: one))

        let r1cs = R1CSInstance(numConstraints: 1, numVars: 4, numPublic: 2,
                                aEntries: aE, bEntries: bE, cEntries: cE)

        let a = frFromInt(3)
        let b = frFromInt(7)

        // Force CPU path for small circuit
        gpuWitness.gpuMinConstraints = 999999
        let z = gpuWitness.generateWitness(r1cs: r1cs, publicInputs: [a, b])

        expect(frEq(z[0], Fr.one), "GPU witness: wire 0 = 1")
        expect(frEq(z[1], a), "GPU witness: wire 1 = a")
        expect(frEq(z[2], b), "GPU witness: wire 2 = b")
        let expected_c = frMul(a, b)
        expect(frEq(z[3], expected_c), "GPU witness: w3 = a*b = 21")
        expect(r1cs.isSatisfied(z: z), "GPU witness: simple multiply R1CS satisfied")
    }

    // ================================================================
    // MARK: - Test 2: x^3 + x + 5 = y (example circuit)
    // ================================================================
    do {
        let r1cs = buildExampleCircuit()

        gpuWitness.gpuMinConstraints = 0  // force GPU path
        gpuWitness.gpuWaveThreshold = 0

        let xVal: UInt64 = 3
        let (pubInputs, _) = computeExampleWitness(x: xVal)

        let z = gpuWitness.generateWitness(r1cs: r1cs, publicInputs: pubInputs)

        expect(frEq(z[0], Fr.one), "x^3+x+5: wire 0 = 1")
        expect(frEq(z[1], pubInputs[0]), "x^3+x+5: wire 1 = x")
        expect(frEq(z[2], pubInputs[1]), "x^3+x+5: wire 2 = y")

        // v1 = x*x = 9
        let x = frFromInt(xVal)
        let v1 = frMul(x, x)
        expect(frEq(z[3], v1), "x^3+x+5: v1 = x*x = 9")

        // v2 = v1*x = 27
        let v2 = frMul(v1, x)
        expect(frEq(z[4], v2), "x^3+x+5: v2 = x^3 = 27")

        expect(r1cs.isSatisfied(z: z), "x^3+x+5: R1CS satisfied")
    }

    // ================================================================
    // MARK: - Test 3: Cross-validate GPU vs CPU WitnessSolver
    // ================================================================
    do {
        let r1cs = buildExampleCircuit()
        let (pubInputs, _) = computeExampleWitness(x: 7)

        // GPU path
        gpuWitness.gpuMinConstraints = 0
        gpuWitness.gpuWaveThreshold = 0
        let gpuZ = gpuWitness.generateWitness(r1cs: r1cs, publicInputs: pubInputs)

        // CPU WitnessSolver path
        let solver = WitnessSolver()
        let cpuResult = solver.solve(r1cs: r1cs, publicInputs: pubInputs)

        expect(cpuResult.isFullySolved, "CPU solver fully solved")
        for i in 0..<r1cs.numVars {
            expect(frEq(gpuZ[i], cpuResult.witness[i]),
                   "Cross-validate wire \(i): GPU matches CPU")
        }
    }

    // ================================================================
    // MARK: - Test 4: Chain circuit (sequential dependencies)
    // ================================================================
    do {
        // Build a 10-constraint chain: w_i = w_{i-1} * x
        let (r1cs, pubInputs, expectedWitness) = buildBenchCircuit(numConstraints: 10)

        gpuWitness.gpuMinConstraints = 0
        gpuWitness.gpuWaveThreshold = 0
        let z = gpuWitness.generateWitness(r1cs: r1cs, publicInputs: pubInputs)

        expect(r1cs.isSatisfied(z: z), "Chain circuit (10): R1CS satisfied")

        // Check witness values match
        let nPub = r1cs.numPublic
        for i in 0..<expectedWitness.count {
            expect(frEq(z[1 + nPub + i], expectedWitness[i]),
                   "Chain circuit: witness[\(i)] matches expected")
        }
    }

    // ================================================================
    // MARK: - Test 5: Larger chain circuit for GPU dispatch
    // ================================================================
    do {
        let n = 200
        let (r1cs, pubInputs, expectedWitness) = buildBenchCircuit(numConstraints: n)

        // Force GPU for small waves too
        gpuWitness.gpuMinConstraints = 0
        gpuWitness.gpuWaveThreshold = 1
        let z = gpuWitness.generateWitness(r1cs: r1cs, publicInputs: pubInputs)

        expect(r1cs.isSatisfied(z: z), "Chain circuit (\(n)): R1CS satisfied")

        let nPub = r1cs.numPublic
        for i in 0..<expectedWitness.count {
            expect(frEq(z[1 + nPub + i], expectedWitness[i]),
                   "Chain circuit (\(n)): witness[\(i)] matches")
        }
    }

    // ================================================================
    // MARK: - Test 6: Circuit with hints
    // ================================================================
    do {
        // Circuit: a * b = c, but b is a private hint (not public input)
        let one = Fr.one
        var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

        // Vars: [1, a, c, b] where a is public, b is hint, c is derived
        aE.append(R1CSEntry(row: 0, col: 1, val: one))  // A: a
        bE.append(R1CSEntry(row: 0, col: 3, val: one))  // B: b
        cE.append(R1CSEntry(row: 0, col: 2, val: one))  // C: c

        let r1cs = R1CSInstance(numConstraints: 1, numVars: 4, numPublic: 1,
                                aEntries: aE, bEntries: bE, cEntries: cE)

        let a = frFromInt(5)
        let b = frFromInt(11)
        let expectedC = frMul(a, b)  // 55

        gpuWitness.gpuMinConstraints = 999999  // CPU path
        let z = gpuWitness.generateWitness(r1cs: r1cs, publicInputs: [a], hints: [3: b])

        expect(frEq(z[2], expectedC), "Hints: c = a*b = 55")
        expect(r1cs.isSatisfied(z: z), "Hints: R1CS satisfied")
    }

    // ================================================================
    // MARK: - Test 7: GPU sparse mat-vec
    // ================================================================
    do {
        // Simple 2x3 matrix * 3-vector
        // A = [[1, 2, 0], [0, 3, 4]]
        // z = [5, 7, 11]
        // Expected: [1*5 + 2*7, 3*7 + 4*11] = [19, 65]
        let entries = [
            R1CSEntry(row: 0, col: 0, val: Fr.one),
            R1CSEntry(row: 0, col: 1, val: frFromInt(2)),
            R1CSEntry(row: 1, col: 1, val: frFromInt(3)),
            R1CSEntry(row: 1, col: 2, val: frFromInt(4)),
        ]
        let vec = [frFromInt(5), frFromInt(7), frFromInt(11)]

        if let result = gpuWitness.gpuSparseMatVec(entries: entries, numRows: 2, vec: vec) {
            expect(frEq(result[0], frFromInt(19)), "Sparse mat-vec: row 0 = 19")
            expect(frEq(result[1], frFromInt(65)), "Sparse mat-vec: row 1 = 65")
        } else {
            expect(false, "Sparse mat-vec: GPU dispatch failed")
        }
    }

    // ================================================================
    // MARK: - Test 8: Parallel circuit (wide, not deep)
    // ================================================================
    do {
        // Build a circuit where all constraints are independent (one wave):
        // Each constraint: x * const_i = w_i (all share input x, each produces w_i)
        let numC = 50
        let numVars = 2 + numC  // [1, x, w_0, w_1, ..., w_{numC-1}]
        var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

        for i in 0..<numC {
            // x * (i+1) = w_i  =>  A: x, B: (i+1)*1, C: w_i
            aE.append(R1CSEntry(row: i, col: 1, val: Fr.one))
            bE.append(R1CSEntry(row: i, col: 0, val: frFromInt(UInt64(i + 1))))
            cE.append(R1CSEntry(row: i, col: 2 + i, val: Fr.one))
        }

        let r1cs = R1CSInstance(numConstraints: numC, numVars: numVars, numPublic: 1,
                                aEntries: aE, bEntries: bE, cEntries: cE)

        let x = frFromInt(13)

        gpuWitness.gpuMinConstraints = 0
        gpuWitness.gpuWaveThreshold = 0
        let z = gpuWitness.generateWitness(r1cs: r1cs, publicInputs: [x])

        expect(r1cs.isSatisfied(z: z), "Parallel circuit: R1CS satisfied")

        // Check all w_i = x * (i+1)
        for i in 0..<numC {
            let expected = frMul(x, frFromInt(UInt64(i + 1)))
            expect(frEq(z[2 + i], expected), "Parallel circuit: w_\(i) = x*\(i+1)")
        }
    }

    // ================================================================
    // MARK: - Test 9: Profile API
    // ================================================================
    do {
        let (r1cs, pubInputs, _) = buildBenchCircuit(numConstraints: 50)

        gpuWitness.gpuMinConstraints = 0
        gpuWitness.gpuWaveThreshold = 1
        let (z, scheduleMs, solveMs, gpuWaves, cpuWaves) =
            gpuWitness.generateWitnessWithProfile(r1cs: r1cs, publicInputs: pubInputs)

        expect(r1cs.isSatisfied(z: z), "Profile API: R1CS satisfied")
        expect(scheduleMs >= 0, "Profile API: schedule time non-negative")
        expect(solveMs >= 0, "Profile API: solve time non-negative")
        expect(gpuWaves + cpuWaves > 0, "Profile API: at least one wave processed")
    }

    // ================================================================
    // MARK: - Test 10: Version entry
    // ================================================================
    do {
        let v = Versions.groth16GPUWitness
        expect(v.version == "1.0.0", "Version: groth16GPUWitness = 1.0.0")
    }

    // Restore defaults
    gpuWitness.gpuMinConstraints = 128
    gpuWitness.gpuWaveThreshold = 64
}
