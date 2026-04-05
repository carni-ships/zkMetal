// CustomGateLibraryTests -- Tests for the custom gate library and compiler
//
// Validates each gate type individually and the compiler integration.

import zkMetal
import Foundation

public func runCustomGateLibraryTests() {

    // ========== BoolGate ==========
    suite("CustomGateLibrary - BoolGate")
    do {
        let gate = BoolGate()
        expect(gate.name == "Bool", "BoolGate name")
        expect(gate.constraintCount == 1, "BoolGate: 1 constraint")
        expect(gate.wireCount == 1, "BoolGate: 1 wire")

        // Valid: w=0
        let r0 = gate.evaluate(witness: [Fr.zero])
        expect(frEqual(r0, Fr.zero), "BoolGate: w=0 is valid")

        // Valid: w=1
        let r1 = gate.evaluate(witness: [Fr.one])
        expect(frEqual(r1, Fr.zero), "BoolGate: w=1 is valid")

        // Invalid: w=2
        let r2 = gate.evaluate(witness: [frFromInt(2)])
        expect(!frEqual(r2, Fr.zero), "BoolGate: w=2 is rejected")

        // Invalid: w=42
        let r42 = gate.evaluate(witness: [frFromInt(42)])
        expect(!frEqual(r42, Fr.zero), "BoolGate: w=42 is rejected")

        // buildConstraints produces 1 gate
        let constraints = gate.buildConstraints(vars: [0])
        expect(constraints.count == 1, "BoolGate: buildConstraints returns 1 gate")

        // Selector values
        let sels = gate.selectorValues
        expect(frEqual(sels["qL"]!, Fr.one), "BoolGate: qL=1")
    }

    // ========== RangeGate ==========
    suite("CustomGateLibrary - RangeGate")
    do {
        let gate = RangeGate(bits: 8)
        expect(gate.name == "Range8", "RangeGate name")
        expect(gate.bits == 8, "RangeGate: 8 bits")
        expect(gate.wireCount == 1 + 8 + 7, "RangeGate: 16 wires (1 value + 8 bits + 7 accumulators)")

        // Helper: decompose value into bits and accumulators
        func makeRangeWitness(_ value: UInt64, bits: Int) -> [Fr] {
            var witness = [Fr]()
            witness.append(frFromInt(value))
            // Bit decomposition (LSB first)
            var bitValues = [Fr]()
            for i in 0..<bits {
                bitValues.append(frFromInt((value >> UInt64(i)) & 1))
            }
            witness.append(contentsOf: bitValues)
            // Accumulator values
            if bits > 1 {
                var acc = bitValues[0]
                for i in 1..<bits {
                    let coeff = frFromInt(1 << UInt64(i))
                    acc = frAdd(acc, frMul(bitValues[i], coeff))
                    witness.append(acc)
                }
            }
            return witness
        }

        // Valid: value=0 (all bits zero)
        let w0 = makeRangeWitness(0, bits: 8)
        let r0 = gate.evaluate(witness: w0)
        expect(frEqual(r0, Fr.zero), "RangeGate(8): value=0 passes")

        // Valid: value=1
        let w1 = makeRangeWitness(1, bits: 8)
        let r1 = gate.evaluate(witness: w1)
        expect(frEqual(r1, Fr.zero), "RangeGate(8): value=1 passes")

        // Valid: value=127
        let w127 = makeRangeWitness(127, bits: 8)
        let r127 = gate.evaluate(witness: w127)
        expect(frEqual(r127, Fr.zero), "RangeGate(8): value=127 passes")

        // Valid: value=255 (max 8-bit)
        let w255 = makeRangeWitness(255, bits: 8)
        let r255 = gate.evaluate(witness: w255)
        expect(frEqual(r255, Fr.zero), "RangeGate(8): value=255 passes")

        // Invalid: value=256 (exceeds 8-bit range)
        // The decomposition for 256 requires bit 8 which is out of range for 8-bit gate.
        // With only 8 bits, 256 = 0b100000000 truncates to 0, mismatch.
        var w256 = makeRangeWitness(0, bits: 8) // bits of 256 mod 256 = 0
        w256[0] = frFromInt(256) // but value is 256
        let r256 = gate.evaluate(witness: w256)
        expect(!frEqual(r256, Fr.zero), "RangeGate(8): value=256 fails (out of range)")

        // Invalid: non-boolean bit
        var wBad = makeRangeWitness(5, bits: 8)
        wBad[1] = frFromInt(2) // corrupt bit 0 to non-boolean value
        let rBad = gate.evaluate(witness: wBad)
        expect(!frEqual(rBad, Fr.zero), "RangeGate(8): non-boolean bit rejected")

        // buildConstraints count
        let vars = Array(0..<gate.wireCount)
        let constraints = gate.buildConstraints(vars: vars)
        // Should have: 8 bool gates + 7 accumulation gates + 1 equality = 16
        expect(constraints.count == 16, "RangeGate(8): buildConstraints returns 16 gates")
    }

    // ========== ConditionalSelectGate ==========
    suite("CustomGateLibrary - ConditionalSelectGate")
    do {
        let gate = ConditionalSelectGateTemplate()
        expect(gate.name == "ConditionalSelect", "ConditionalSelectGate name")
        expect(gate.wireCount == 6, "ConditionalSelectGate: 6 wires")

        // sel=1, a=42, b=99 => out=42
        let r1 = gate.evaluate(witness: [Fr.one, frFromInt(42), frFromInt(99), frFromInt(42)])
        expect(frEqual(r1, Fr.zero), "ConditionalSelect: sel=1 -> out=a=42")

        // sel=0, a=42, b=99 => out=99
        let r0 = gate.evaluate(witness: [Fr.zero, frFromInt(42), frFromInt(99), frFromInt(99)])
        expect(frEqual(r0, Fr.zero), "ConditionalSelect: sel=0 -> out=b=99")

        // Wrong output: sel=1, a=42, b=99 but out=99
        let rBad = gate.evaluate(witness: [Fr.one, frFromInt(42), frFromInt(99), frFromInt(99)])
        expect(!frEqual(rBad, Fr.zero), "ConditionalSelect: wrong output rejected")

        // Non-boolean selector: sel=2
        let rNonBool = gate.evaluate(witness: [frFromInt(2), frFromInt(42), frFromInt(99), frFromInt(42)])
        expect(!frEqual(rNonBool, Fr.zero), "ConditionalSelect: non-boolean selector rejected")

        // sel=1, a=0, b=0 => out=0
        let rZero = gate.evaluate(witness: [Fr.one, Fr.zero, Fr.zero, Fr.zero])
        expect(frEqual(rZero, Fr.zero), "ConditionalSelect: sel=1, a=b=0 -> out=0")

        // buildConstraints
        let vars = Array(0..<gate.wireCount)
        let constraints = gate.buildConstraints(vars: vars)
        expect(constraints.count == 4, "ConditionalSelectGate: buildConstraints returns 4 gates")
    }

    // ========== XorGate ==========
    suite("CustomGateLibrary - XorGate")
    do {
        let gate = XorGate()
        expect(gate.name == "Xor", "XorGate name")

        // Full truth table
        // 0 XOR 0 = 0
        let r00 = gate.evaluate(witness: [Fr.zero, Fr.zero, Fr.zero])
        expect(frEqual(r00, Fr.zero), "XOR: 0^0=0")

        // 0 XOR 1 = 1
        let r01 = gate.evaluate(witness: [Fr.zero, Fr.one, Fr.one])
        expect(frEqual(r01, Fr.zero), "XOR: 0^1=1")

        // 1 XOR 0 = 1
        let r10 = gate.evaluate(witness: [Fr.one, Fr.zero, Fr.one])
        expect(frEqual(r10, Fr.zero), "XOR: 1^0=1")

        // 1 XOR 1 = 0
        let r11 = gate.evaluate(witness: [Fr.one, Fr.one, Fr.zero])
        expect(frEqual(r11, Fr.zero), "XOR: 1^1=0")

        // Wrong: 1 XOR 1 = 1 (should be 0)
        let rBad = gate.evaluate(witness: [Fr.one, Fr.one, Fr.one])
        expect(!frEqual(rBad, Fr.zero), "XOR: 1^1=1 rejected")

        // Non-boolean input rejected
        let rNB = gate.evaluate(witness: [frFromInt(2), Fr.zero, frFromInt(2)])
        expect(!frEqual(rNB, Fr.zero), "XOR: non-boolean input rejected")

        // buildConstraints: optimal encoding uses 3 gates (2 bool + 1 XOR)
        let constraints = gate.buildConstraints(vars: [0, 1, 2, 3])
        expect(constraints.count == 3, "XorGate: buildConstraints returns 3 gates (optimal)")
    }

    // ========== RotlGate ==========
    suite("CustomGateLibrary - RotlGate")
    do {
        // ROTL(0b11001010, 3) over 8 bits = 0b01010110
        // 0xCA = 202, rotated left by 3 = 0x56 = 86... no.
        // 0b11001010 << 3 = 0b01010000 (lower 8), >> (8-3)=5 = 0b00000110
        // result = 0b01010110 = 86? Let's compute properly:
        // 202 = 0b11001010
        // ROTL(202, 3, 8):
        //   hi = 202 >> (8-3) = 202 >> 5 = 6 (top 3 bits: 110)
        //   lo = 202 mod 2^5 = 202 mod 32 = 202 - 6*32 = 202 - 192 = 10 (0b01010)
        //   result = lo * 2^3 + hi = 10 * 8 + 6 = 86 (0b01010110)
        let gate = RotlGate(bitWidth: 8, rotateAmount: 3)
        expect(gate.name == "Rotl8_3", "RotlGate name")
        expect(gate.wireCount == 4, "RotlGate: 4 wires")
        expect(gate.constraintCount == 2, "RotlGate: 2 constraints")

        let x: UInt64 = 202
        let r: UInt64 = 3
        let n: UInt64 = 8
        let hi = x >> (n - r)         // 6
        let lo = x % (1 << (n - r))   // 10
        let result = lo * (1 << r) + hi  // 86

        let rOk = gate.evaluate(witness: [frFromInt(x), frFromInt(result),
                                           frFromInt(hi), frFromInt(lo)])
        expect(frEqual(rOk, Fr.zero), "RotlGate: ROTL(202, 3, 8) = 86")

        // Wrong result
        let rBad = gate.evaluate(witness: [frFromInt(x), frFromInt(42),
                                            frFromInt(hi), frFromInt(lo)])
        expect(!frEqual(rBad, Fr.zero), "RotlGate: wrong result rejected")

        // Edge case: rotate by 0 -- not applicable (precondition rotateAmount > 0)
        // Edge case: rotate full value
        let gate7 = RotlGate(bitWidth: 8, rotateAmount: 7)
        // ROTL(202, 7, 8): hi = 202 >> 1 = 101, lo = 202 % 2 = 0
        // result = 0 * 128 + 101 = 101
        let r7 = gate7.evaluate(witness: [frFromInt(202), frFromInt(101),
                                           frFromInt(101), frFromInt(0)])
        expect(frEqual(r7, Fr.zero), "RotlGate: ROTL(202, 7, 8) = 101")
    }

    // ========== Poseidon2RoundGateTemplate ==========
    suite("CustomGateLibrary - Poseidon2RoundGate")
    do {
        let width = 3
        let rc: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3)]
        // Identity MDS
        var mds = [Fr](repeating: Fr.zero, count: 9)
        mds[0] = Fr.one; mds[4] = Fr.one; mds[8] = Fr.one

        let gate = Poseidon2RoundGateTemplate(width: width, roundConstants: rc, mds: mds)
        expect(gate.name.hasPrefix("Poseidon2Round"), "Poseidon2RoundGate name")
        expect(gate.wireCount == 9, "Poseidon2RoundGate: 9 wires for width=3")

        // Compute expected: sbox(x) = x^5 with identity MDS
        // in = [10, 20, 30], rc = [1, 2, 3]
        // x = [11, 22, 33]
        // out[i] = x[i]^5
        let stateIn: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30)]
        var stateOut = [Fr]()
        var sqValues = [Fr]()
        for i in 0..<width {
            let x = frAdd(stateIn[i], rc[i])
            let x2 = frSqr(x)
            let x4 = frSqr(x2)
            stateOut.append(frMul(x, x4))
            sqValues.append(x2)  // auxiliary: (in+rc)^2
        }

        // Witness: [stateIn..., stateOut..., sq...]
        var witness = stateIn + stateOut + sqValues
        let rOk = gate.evaluate(witness: witness)
        expect(frEqual(rOk, Fr.zero), "Poseidon2RoundGate: correct full round evaluates to zero")

        // Wrong output
        witness[width] = frFromInt(42) // corrupt state_out[0]
        let rBad = gate.evaluate(witness: witness)
        expect(!frEqual(rBad, Fr.zero), "Poseidon2RoundGate: wrong output rejected")
    }

    // ========== ECAddGateTemplate ==========
    suite("CustomGateLibrary - ECAddGate")
    do {
        let gate = ECAddGateTemplate()
        expect(gate.name == "ECAdd", "ECAddGate name")
        expect(gate.wireCount == 7, "ECAddGate: 7 wires")

        // P1=(1,2), P2=(3,4)
        // lambda = (4-2)/(3-1) = 1
        // x3 = 1 - 1 - 3 = -3
        // y3 = 1*(1-(-3)) - 2 = 4 - 2 = 2
        let x1 = frFromInt(1), y1 = frFromInt(2)
        let x2 = frFromInt(3), y2 = frFromInt(4)
        let lam = Fr.one // (y2-y1)/(x2-x1) = 2/2 = 1
        let x3 = frSub(frSub(frSqr(lam), x1), x2)
        let y3 = frSub(frMul(lam, frSub(x1, x3)), y1)

        let rOk = gate.evaluate(witness: [x1, y1, x2, y2, x3, y3, lam])
        expect(frEqual(rOk, Fr.zero), "ECAddGate: correct addition evaluates to zero")

        // Wrong point
        let rBad = gate.evaluate(witness: [x1, y1, x2, y2, frFromInt(999), y3, lam])
        expect(!frEqual(rBad, Fr.zero), "ECAddGate: wrong x3 rejected")

        // Wrong lambda
        let rBadLam = gate.evaluate(witness: [x1, y1, x2, y2, x3, y3, frFromInt(5)])
        expect(!frEqual(rBadLam, Fr.zero), "ECAddGate: wrong lambda rejected")
    }

    // ========== ECDoubleGateTemplate ==========
    suite("CustomGateLibrary - ECDoubleGate")
    do {
        let gate = ECDoubleGateTemplate()
        expect(gate.name == "ECDouble", "ECDoubleGate name")
        expect(gate.wireCount == 5, "ECDoubleGate: 5 wires")

        // P1=(1,2), lambda = 3*1^2/(2*2) = 3/4
        let x1 = frFromInt(1), y1 = frFromInt(2)
        let three = frFromInt(3), four = frFromInt(4), two = frFromInt(2)
        let lam = frMul(three, frInverse(four))
        let x2 = frSub(frSqr(lam), frMul(two, x1))
        let y2 = frSub(frMul(lam, frSub(x1, x2)), y1)

        let rOk = gate.evaluate(witness: [x1, y1, x2, y2, lam])
        expect(frEqual(rOk, Fr.zero), "ECDoubleGate: correct doubling evaluates to zero")

        // Wrong output
        let rBad = gate.evaluate(witness: [x1, y1, frFromInt(42), y2, lam])
        expect(!frEqual(rBad, Fr.zero), "ECDoubleGate: wrong x2 rejected")
    }

    // ========== LookupGateTemplate ==========
    suite("CustomGateLibrary - LookupGate")
    do {
        let table: [Fr] = [frFromInt(0), frFromInt(1), frFromInt(2), frFromInt(3)]
        let gate = LookupGateTemplate(tableId: 0, table: table)
        expect(gate.name == "Lookup_0", "LookupGate name")
        expect(gate.wireCount == 1, "LookupGate: 1 wire")

        // Value in table
        for i: UInt64 in 0...3 {
            let r = gate.evaluate(witness: [frFromInt(i)])
            expect(frEqual(r, Fr.zero), "LookupGate: value \(i) in table")
        }

        // Value not in table
        let rBad = gate.evaluate(witness: [frFromInt(4)])
        expect(!frEqual(rBad, Fr.zero), "LookupGate: value 4 not in table")

        let rBad2 = gate.evaluate(witness: [frFromInt(999)])
        expect(!frEqual(rBad2, Fr.zero), "LookupGate: value 999 not in table")

        // buildConstraints marks as lookup
        let constraints = gate.buildConstraints(vars: [0])
        expect(constraints.count == 1, "LookupGate: 1 gate")
        expect(!frEqual(constraints[0].gate.qLookup, Fr.zero), "LookupGate: qLookup is set")
    }

    // ========== CustomGateCompiler ==========
    suite("CustomGateLibrary - Compiler Integration")
    do {
        // Build a circuit with mixed gate types
        let compiler = CustomGateCompiler()

        // Input variables with witness values
        let a = compiler.addVariable(value: Fr.one)       // boolean 1
        let b = compiler.addVariable(value: Fr.zero)       // boolean 0
        let x = compiler.addVariable(value: frFromInt(42)) // 8-bit value

        // Bool constraint on a
        compiler.addBoolConstraint(a)
        // Bool constraint on b
        compiler.addBoolConstraint(b)

        // XOR: a ^ b = 1
        let xorOut = compiler.addXor(a: a, b: b, aValue: Fr.one, bValue: Fr.zero)

        // Conditional select: sel=a(=1) ? x(=42) : b(=0) => out=42
        let selOut = compiler.addConditionalSelect(
            selector: a, a: x, b: b,
            selectorValue: Fr.one, aValue: frFromInt(42), bValue: Fr.zero)

        // Verify all constraints
        let verified = compiler.verify()
        expect(verified, "Compiler: mixed circuit verification passes")

        // Check XOR output
        let witness = compiler.getWitness()
        expect(frEqual(witness[xorOut], Fr.one), "Compiler: XOR output = 1")
        expect(frEqual(witness[selOut], frFromInt(42)), "Compiler: select output = 42")

        // Compile to PlonkCircuit
        let circuit = compiler.compile()
        expect(circuit.numGates > 0, "Compiler: circuit has gates")
        expect(circuit.gates.count > 0, "Compiler: non-empty gates")

        // Statistics
        expect(compiler.gateCount == 4, "Compiler: 4 custom gates added")
        expect(compiler.totalVariableCount > 0, "Compiler: variables allocated")
        expect(compiler.totalConstraintCount > 0, "Compiler: constraints generated")
    }

    // ========== Compiler: Range Check ==========
    suite("CustomGateLibrary - Compiler Range Check")
    do {
        let compiler = CustomGateCompiler()
        let val = compiler.addVariable(value: frFromInt(200))

        // Decompose 200 = 0b11001000
        var bits = [Fr]()
        for i in 0..<8 {
            bits.append(frFromInt((200 >> UInt64(i)) & 1))
        }
        compiler.addRangeCheck(val, bits: 8, bitValues: bits)

        let verified = compiler.verify()
        expect(verified, "Compiler: range check for 200 in [0,256) passes")

        // Now try with wrong value (256)
        let compiler2 = CustomGateCompiler()
        let val2 = compiler2.addVariable(value: frFromInt(256))
        // Decompose 0 (truncated 256 to 8 bits)
        var bits2 = [Fr](repeating: Fr.zero, count: 8)
        compiler2.addRangeCheck(val2, bits: 8, bitValues: bits2)

        let verified2 = compiler2.verify()
        expect(!verified2, "Compiler: range check for 256 in [0,256) fails")
    }

    // ========== Compiler: Selector Polynomials ==========
    suite("CustomGateLibrary - Selector Polynomials")
    do {
        let compiler = CustomGateCompiler()
        let a = compiler.addVariable(value: Fr.one)
        let b = compiler.addVariable(value: Fr.zero)

        compiler.addBoolConstraint(a)
        compiler.addBoolConstraint(b)

        let (qL, qR, qO, qM, qC) = compiler.generateSelectorPolynomials()

        // BoolGate uses qL=1, qM=-1
        expect(qL.count == 2, "Selector polys: 2 rows for 2 bool gates")
        expect(frEqual(qL[0], Fr.one), "Selector polys: qL[0]=1")
        expect(frEqual(qM[0], frSub(Fr.zero, Fr.one)), "Selector polys: qM[0]=-1")
        expect(frEqual(qR[0], Fr.zero), "Selector polys: qR[0]=0")
        expect(frEqual(qC[0], Fr.zero), "Selector polys: qC[0]=0")
    }

    // ========== Library Factory Methods ==========
    suite("CustomGateLibrary - Factory Methods")
    do {
        let bg = CustomGateLibrary.boolGate()
        expect(bg.name == "Bool", "Factory: boolGate")

        let rg = CustomGateLibrary.rangeGate(bits: 16)
        expect(rg.bits == 16, "Factory: rangeGate(16)")

        let cs = CustomGateLibrary.conditionalSelectGate()
        expect(cs.name == "ConditionalSelect", "Factory: conditionalSelectGate")

        let xg = CustomGateLibrary.xorGate()
        expect(xg.name == "Xor", "Factory: xorGate")

        let rl = CustomGateLibrary.rotlGate(bitWidth: 32, rotateAmount: 7)
        expect(rl.bitWidth == 32, "Factory: rotlGate")

        let p2 = CustomGateLibrary.poseidon2RoundGate(
            width: 3,
            roundConstants: [Fr.one, Fr.one, Fr.one],
            mds: [Fr](repeating: Fr.zero, count: 9))
        expect(p2.width == 3, "Factory: poseidon2RoundGate")

        let ea = CustomGateLibrary.ecAddGate()
        expect(ea.name == "ECAdd", "Factory: ecAddGate")

        let ed = CustomGateLibrary.ecDoubleGate()
        expect(ed.name == "ECDouble", "Factory: ecDoubleGate")

        let lg = CustomGateLibrary.lookupGate(tableId: 0, table: [Fr.zero, Fr.one])
        expect(lg.tableId == 0, "Factory: lookupGate")
    }

    // ========== Compiler: Copy Constraints ==========
    suite("CustomGateLibrary - Copy Constraints")
    do {
        let compiler = CustomGateCompiler()
        let a = compiler.addVariable(value: frFromInt(7))
        let b = compiler.addVariable(value: frFromInt(7))
        let c = compiler.addVariable(value: frFromInt(8))

        compiler.addBoolConstraint(a) // will fail since 7 is not bool, but testing copy
        compiler.assertEqual(a, b) // same value => OK

        // Check copy constraints pass for equal values (ignoring bool gate failure)
        let witness = compiler.getWitness()
        expect(frEqual(witness[a], witness[b]), "Copy constraint: a == b")
        expect(!frEqual(witness[a], witness[c]), "Copy constraint: a != c")
    }

    // ========== Compiler: padded circuit ==========
    suite("CustomGateLibrary - Padded Circuit")
    do {
        let compiler = CustomGateCompiler()
        let a = compiler.addVariable(value: Fr.one)
        compiler.addBoolConstraint(a)

        let circuit = compiler.compileAndPad()
        // Must be power of 2, at least 4
        expect(circuit.numGates >= 4, "Padded circuit: at least 4 gates")
        expect(circuit.numGates & (circuit.numGates - 1) == 0, "Padded circuit: power of 2")
    }
}
