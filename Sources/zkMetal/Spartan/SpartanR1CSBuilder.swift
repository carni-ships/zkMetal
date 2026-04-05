// SpartanR1CSBuilder — Programmatic R1CS construction helpers
//
// Provides a builder API for constructing R1CS instances suitable for
// Spartan proving, plus common circuit templates for testing.

import Foundation

// MARK: - Extended Builder API

extension SpartanR1CSBuilder {
    /// Allocate a new variable (witness). Returns the variable index.
    /// Variable 0 is always the constant 1; this allocates a witness variable.
    public func addVariable() -> Int {
        return addWitness()
    }

    // MARK: - Common Circuit Templates

    /// Build a simple multiply circuit: x * y = z
    /// Returns (instance, witnessGenerator) where witnessGenerator(x, y) -> (publicInputs, witness)
    public static func buildMultiplyCircuit() -> (SpartanR1CS, (Fr, Fr) -> (publicInputs: [Fr], witness: [Fr])) {
        let b = SpartanR1CSBuilder()
        let z = b.addPublicInput()   // var 1: public output
        let x = b.addWitness()       // var 2: first multiplicand
        let y = b.addWitness()       // var 3: second multiplicand

        // x * y = z
        b.mulGate(a: x, b: y, out: z)

        let instance = b.build()
        let gen: (Fr, Fr) -> (publicInputs: [Fr], witness: [Fr]) = { xVal, yVal in
            let zVal = frMul(xVal, yVal)
            return ([zVal], [xVal, yVal])
        }
        return (instance, gen)
    }

    /// Build an add chain circuit with n addition gates.
    /// Computes: v0 = x, v1 = v0 + x, v2 = v1 + x, ..., v_n = v_{n-1} + x
    /// Public output: v_n
    /// Returns (instance, witnessGenerator) where witnessGenerator(x) -> (publicInputs, witness)
    public static func buildAddChainCircuit(n: Int) -> (SpartanR1CS, (Fr) -> (publicInputs: [Fr], witness: [Fr])) {
        precondition(n >= 1, "Add chain must have at least 1 gate")
        let b = SpartanR1CSBuilder()
        let output = b.addPublicInput()  // var 1: public output
        let x = b.addWitness()           // var 2: input

        // Build chain: v[0] = x, then v[i] = v[i-1] + x for i in 1..n
        var prevVar = x
        var intermediateVars = [Int]()
        for i in 0..<n {
            let out: Int
            if i == n - 1 {
                // Last gate outputs to the public output variable
                out = output
            } else {
                out = b.addWitness()
                intermediateVars.append(out)
            }
            // (prevVar + x) * 1 = out
            b.addGate(a: prevVar, b: x, out: out)
            prevVar = out
        }

        let instance = b.build()
        let gen: (Fr) -> (publicInputs: [Fr], witness: [Fr]) = { xVal in
            // v[0] = x, v[i] = v[i-1] + x = (i+1)*x
            // After n gates: output = (n+1)*x
            var witness = [Fr]()
            witness.append(xVal) // x

            var prev = xVal
            for i in 0..<n {
                let next = frAdd(prev, xVal)
                if i < n - 1 {
                    witness.append(next) // intermediate
                }
                prev = next
            }
            let outputVal = prev // (n+1)*x
            return ([outputVal], witness)
        }
        return (instance, gen)
    }

    /// Build a range check circuit that constrains a value to fit in `bits` bits.
    /// Decomposes the value into individual bits and constrains each bit in {0,1}.
    ///
    /// Circuit structure:
    ///   - Public input: value (the number to range-check)
    ///   - Witness: bit_0, bit_1, ..., bit_{bits-1}
    ///   - Constraints:
    ///     1. For each bit b_i: b_i * (1 - b_i) = 0  (boolean constraint)
    ///     2. sum(b_i * 2^i) = value  (decomposition constraint)
    ///
    /// Returns (instance, witnessGenerator) where witnessGenerator(value) -> (publicInputs, witness)
    public static func buildRangeCheckCircuit(bits: Int) -> (SpartanR1CS, (UInt64) -> (publicInputs: [Fr], witness: [Fr])) {
        precondition(bits >= 1 && bits <= 64, "bits must be in [1, 64]")
        let b = SpartanR1CSBuilder()
        let value = b.addPublicInput()  // var 1: the value being range-checked

        // Allocate bit variables
        var bitVars = [Int]()
        for _ in 0..<bits {
            bitVars.append(b.addWitness())
        }

        // Boolean constraints: b_i * (1 - b_i) = 0
        // Rewrite as: b_i * b_i = b_i (since b_i*(1-b_i)=0 iff b_i^2 = b_i)
        for i in 0..<bits {
            b.mulGate(a: bitVars[i], b: bitVars[i], out: bitVars[i])
        }

        // Decomposition constraint: sum(b_i * 2^i) = value
        // Express as: (sum coeffs) * 1 = value
        var decomp = [(Int, Fr)]()
        for i in 0..<bits {
            let coeff = frFromInt(1 << i)
            decomp.append((bitVars[i], coeff))
        }
        b.addConstraint(a: decomp, b: [(0, Fr.one)], c: [(value, Fr.one)])

        let instance = b.build()
        let gen: (UInt64) -> (publicInputs: [Fr], witness: [Fr]) = { val in
            let valueField = frFromInt(val)
            var witness = [Fr]()
            for i in 0..<bits {
                let bit = (val >> i) & 1
                witness.append(bit == 1 ? Fr.one : Fr.zero)
            }
            return ([valueField], witness)
        }
        return (instance, gen)
    }
}
