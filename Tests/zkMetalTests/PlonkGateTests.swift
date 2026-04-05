// PlonkGateTests -- Tests for ECC, Hash, and Binary custom gates
//
// Validates that each gate's evaluate() method returns zero for correct witnesses
// and non-zero for incorrect witnesses.

import zkMetal
import Foundation

func runPlonkGateTests() {
    suite("Plonk Custom Gates - ECC")

    // ========== ECAddGate ==========
    do {
        let gate = ECAddGate()
        expect(gate.name == "ECAdd", "ECAddGate name")
        expect(gate.queriedCells.count == 7, "ECAddGate queries 7 cells")

        // Test with a known EC addition on BN254:
        // P1 = (1, 2), P2 = (1, 2) is degenerate (same point), skip
        // Use simple field values where we control lambda:
        // Let x1=1, y1=2, x2=3, y2=4
        // lambda = (y2-y1)/(x2-x1) = (4-2)/(3-1) = 2/2 = 1
        // x3 = lambda^2 - x1 - x2 = 1 - 1 - 3 = -3
        // y3 = lambda*(x1 - x3) - y1 = 1*(1-(-3)) - 2 = 4 - 2 = 2
        let x1 = frFromInt(1)
        let y1 = frFromInt(2)
        let x2 = frFromInt(3)
        let y2 = frFromInt(4)
        let lam = frFromInt(1) // (4-2)/(3-1) = 1
        let x3 = frSub(frSub(frSqr(lam), x1), x2) // 1 - 1 - 3 = -3
        let y3 = frSub(frMul(lam, frSub(x1, x3)), y1) // 1*(1-(-3))-2 = 2

        var rotations: [ColumnRef: Fr] = [
            ColumnRef(column: 0, rotation: .cur): x1,
            ColumnRef(column: 1, rotation: .cur): y1,
            ColumnRef(column: 2, rotation: .cur): x2,
            ColumnRef(column: 0, rotation: .next): y2,
            ColumnRef(column: 1, rotation: .next): x3,
            ColumnRef(column: 2, rotation: .next): y3,
            ColumnRef(column: 3, rotation: .cur): lam,
        ]

        let result = gate.evaluate(rotations: rotations, challenges: [])
        expect(frEqual(result, Fr.zero), "ECAddGate: correct witness evaluates to zero")

        // Incorrect witness: wrong y3
        rotations[ColumnRef(column: 2, rotation: .next)] = frFromInt(999)
        let badResult = gate.evaluate(rotations: rotations, challenges: [])
        expect(!frEqual(badResult, Fr.zero), "ECAddGate: incorrect witness is non-zero")
    }

    // ========== ECDoubleGate ==========
    do {
        let gate = ECDoubleGate()
        expect(gate.name == "ECDouble", "ECDoubleGate name")
        expect(gate.queriedCells.count == 5, "ECDoubleGate queries 5 cells")

        // P1 = (x1, y1), lambda = 3*x1^2 / (2*y1)
        // Use x1=1, y1=2: lambda = 3/(4) = 3*inv(4)
        let x1 = frFromInt(1)
        let y1 = frFromInt(2)
        let three = frFromInt(3)
        let four = frFromInt(4)
        let lam = frMul(three, frInverse(four))
        let lamSq = frSqr(lam)
        let two = frFromInt(2)
        let x2 = frSub(lamSq, frMul(two, x1))
        let y2 = frSub(frMul(lam, frSub(x1, x2)), y1)

        var rotations: [ColumnRef: Fr] = [
            ColumnRef(column: 0, rotation: .cur): x1,
            ColumnRef(column: 1, rotation: .cur): y1,
            ColumnRef(column: 2, rotation: .cur): lam,
            ColumnRef(column: 0, rotation: .next): x2,
            ColumnRef(column: 1, rotation: .next): y2,
        ]

        let result = gate.evaluate(rotations: rotations, challenges: [])
        expect(frEqual(result, Fr.zero), "ECDoubleGate: correct witness evaluates to zero")

        // Incorrect witness
        rotations[ColumnRef(column: 0, rotation: .next)] = frFromInt(42)
        let badResult = gate.evaluate(rotations: rotations, challenges: [])
        expect(!frEqual(badResult, Fr.zero), "ECDoubleGate: incorrect witness is non-zero")
    }

    // ========== TwistedEdwardsAddGate ==========
    do {
        let gate = TwistedEdwardsAddGate.babyJubjub()
        expect(gate.name == "BabyJubjubAdd", "TwistedEdwardsAddGate babyJubjub name")
        expect(gate.queriedCells.count == 6, "TwistedEdwardsAddGate queries 6 cells")

        // Test with identity point addition on twisted Edwards:
        // Identity is (0, 1). Adding identity to itself should give identity.
        // a*x^2 + y^2 = 1 + d*x^2*y^2
        // For (0,1): a*0 + 1 = 1 + d*0 => 1 = 1 (valid)
        //
        // x3 = (0*1 + 1*0) / (1 + d*0*0*1*1) = 0
        // y3 = (1*1 - a*0*0) / (1 - d*0*0*1*1) = 1
        let x1 = Fr.zero
        let y1 = Fr.one
        let x2 = Fr.zero
        let y2 = Fr.one
        let x3 = Fr.zero
        let y3 = Fr.one

        var rotations: [ColumnRef: Fr] = [
            ColumnRef(column: 0, rotation: .cur): x1,
            ColumnRef(column: 1, rotation: .cur): y1,
            ColumnRef(column: 2, rotation: .cur): x2,
            ColumnRef(column: 0, rotation: .next): y2,
            ColumnRef(column: 1, rotation: .next): x3,
            ColumnRef(column: 2, rotation: .next): y3,
        ]

        let result = gate.evaluate(rotations: rotations, challenges: [])
        expect(frEqual(result, Fr.zero), "TwistedEdwardsAdd: identity + identity = identity")

        // Wrong output
        rotations[ColumnRef(column: 2, rotation: .next)] = frFromInt(5)
        let badResult = gate.evaluate(rotations: rotations, challenges: [])
        expect(!frEqual(badResult, Fr.zero), "TwistedEdwardsAdd: wrong output is non-zero")
    }

    // ========== ECScalarMulGate ==========
    do {
        let scalarMul = ECScalarMulGate(scalarBits: 4)
        expect(scalarMul.scalarBits == 4, "ECScalarMulGate: 4-bit scalar")
        expect(scalarMul.name == "ECScalarMul", "ECScalarMulGate name")
    }

    suite("Plonk Custom Gates - Hash")

    // ========== MiMCRoundGate ==========
    do {
        let rc = frFromInt(7)
        let gate = MiMCRoundGate(roundConstant: rc, roundIndex: 0)
        expect(gate.name == "MiMCRound_0", "MiMCRoundGate name")
        expect(gate.queriedCells.count == 5, "MiMCRoundGate queries 5 cells")

        // input=2, key=3, rc=7 => x = 2+3+7 = 12
        // x^2 = 144, x^4 = 144^2 = 20736, x^7 = 12*144*20736 = 35831808
        let input = frFromInt(2)
        let key = frFromInt(3)
        let x = frFromInt(12) // input + key + rc
        let x2 = frSqr(x)
        let x4 = frSqr(x2)
        let output = frMul(x, frMul(x2, x4)) // x^7

        var rotations: [ColumnRef: Fr] = [
            ColumnRef(column: 0, rotation: .cur): input,
            ColumnRef(column: 1, rotation: .cur): key,
            ColumnRef(column: 2, rotation: .cur): output,
            ColumnRef(column: 3, rotation: .cur): x2,
            ColumnRef(column: 4, rotation: .cur): x4,
        ]

        let result = gate.evaluate(rotations: rotations, challenges: [])
        expect(frEqual(result, Fr.zero), "MiMCRoundGate: correct witness evaluates to zero")

        // Wrong output
        rotations[ColumnRef(column: 2, rotation: .cur)] = frFromInt(42)
        let badResult = gate.evaluate(rotations: rotations, challenges: [])
        expect(!frEqual(badResult, Fr.zero), "MiMCRoundGate: wrong output is non-zero")
    }

    // ========== Poseidon2RoundGate ==========
    do {
        // Test with width=3, identity MDS matrix (simplest case)
        let width = 3
        let rc: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3)]
        // Identity MDS for testing
        var mds = [Fr](repeating: Fr.zero, count: 9)
        mds[0] = Fr.one  // mds[0][0]
        mds[4] = Fr.one  // mds[1][1]
        mds[8] = Fr.one  // mds[2][2]

        let gate = Poseidon2RoundGate(width: width, isFullRound: true,
                                       roundConstants: rc, mds: mds, roundIndex: 0)
        expect(gate.queriedCells.count == 6, "Poseidon2RoundGate queries 6 cells (width=3)")

        // Full round with identity MDS: out[i] = sbox(in[i] + rc[i])
        // sbox(x) = x^5
        // in = [10, 20, 30], rc = [1, 2, 3]
        // x = [11, 22, 33]
        // out[0] = 11^5 = 161051
        // out[1] = 22^5 = 5153632
        // out[2] = 33^5 = 39135393
        let stateIn: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30)]
        var stateOut = [Fr](repeating: Fr.zero, count: 3)
        for i in 0..<3 {
            let x = frAdd(stateIn[i], rc[i])
            let x2 = frSqr(x)
            let x4 = frSqr(x2)
            stateOut[i] = frMul(x, x4)
        }

        var rotations: [ColumnRef: Fr] = [:]
        for i in 0..<width {
            rotations[ColumnRef(column: i, rotation: .cur)] = stateIn[i]
            rotations[ColumnRef(column: i, rotation: .next)] = stateOut[i]
        }

        let result = gate.evaluate(rotations: rotations, challenges: [])
        expect(frEqual(result, Fr.zero), "Poseidon2RoundGate: correct full round evaluates to zero")

        // Wrong output
        rotations[ColumnRef(column: 0, rotation: .next)] = frFromInt(42)
        let badResult = gate.evaluate(rotations: rotations, challenges: [])
        expect(!frEqual(badResult, Fr.zero), "Poseidon2RoundGate: wrong output is non-zero")
    }

    // ========== Poseidon2PermutationGate ==========
    do {
        // Test with minimal config: 1 full + 1 partial + 1 full = 3 rounds
        let width = 3
        var mds = [Fr](repeating: Fr.zero, count: 9)
        mds[0] = Fr.one; mds[4] = Fr.one; mds[8] = Fr.one

        let rcs: [[Fr]] = [
            [frFromInt(1), frFromInt(2), frFromInt(3)],
            [frFromInt(4), frFromInt(5), frFromInt(6)],
            [frFromInt(7), frFromInt(8), frFromInt(9)],
        ]

        let perm = Poseidon2PermutationGate(
            width: width, fullRoundsBegin: 1, partialRounds: 1, fullRoundsEnd: 1,
            allRoundConstants: rcs, mds: mds)
        expect(perm.totalRounds == 3, "Poseidon2Permutation: 3 total rounds")

        let roundGates = perm.expandRoundGates()
        expect(roundGates.count == 3, "Poseidon2Permutation: expands to 3 round gates")
        expect(roundGates[0].isFullRound, "Round 0 is full")
        expect(!roundGates[1].isFullRound, "Round 1 is partial")
        expect(roundGates[2].isFullRound, "Round 2 is full")
    }

    suite("Plonk Custom Gates - Binary")

    // ========== ConditionalSelectGate ==========
    do {
        let gate = ConditionalSelectGate()
        expect(gate.name == "ConditionalSelect", "ConditionalSelectGate name")
        expect(gate.queriedCells.count == 4, "ConditionalSelectGate queries 4 cells")

        // cond=1, a=42, b=99 => out=42
        var rotations: [ColumnRef: Fr] = [
            ColumnRef(column: 0, rotation: .cur): Fr.one,
            ColumnRef(column: 1, rotation: .cur): frFromInt(42),
            ColumnRef(column: 2, rotation: .cur): frFromInt(99),
            ColumnRef(column: 0, rotation: .next): frFromInt(42),
        ]

        var result = gate.evaluate(rotations: rotations, challenges: [])
        expect(frEqual(result, Fr.zero), "ConditionalSelect: cond=1 selects a=42")

        // cond=0, a=42, b=99 => out=99
        rotations[ColumnRef(column: 0, rotation: .cur)] = Fr.zero
        rotations[ColumnRef(column: 0, rotation: .next)] = frFromInt(99)
        result = gate.evaluate(rotations: rotations, challenges: [])
        expect(frEqual(result, Fr.zero), "ConditionalSelect: cond=0 selects b=99")

        // Wrong: cond=1, a=42, b=99 but out=99 (should be 42)
        rotations[ColumnRef(column: 0, rotation: .cur)] = Fr.one
        rotations[ColumnRef(column: 0, rotation: .next)] = frFromInt(99)
        result = gate.evaluate(rotations: rotations, challenges: [])
        expect(!frEqual(result, Fr.zero), "ConditionalSelect: wrong output is non-zero")

        // Non-boolean cond should fail
        rotations[ColumnRef(column: 0, rotation: .cur)] = frFromInt(2)
        rotations[ColumnRef(column: 0, rotation: .next)] = frFromInt(42)
        result = gate.evaluate(rotations: rotations, challenges: [])
        expect(!frEqual(result, Fr.zero), "ConditionalSelect: non-boolean cond is non-zero")
    }

    // ========== BinaryDecomposeGate ==========
    do {
        // Test single bit gate: bit=1 at index 0, accumulator goes from 0 to 1
        let gate = BinaryDecomposeGate(bits: 4, bitIndex: 0)
        expect(gate.name == "BinaryDecompose", "BinaryDecomposeGate name")

        // bit=1, acc_prev=0, acc_cur=1 (0 + 2^0 * 1 = 1)
        var rotations: [ColumnRef: Fr] = [
            ColumnRef(column: 0, rotation: .cur): Fr.one,      // bit=1
            ColumnRef(column: 1, rotation: .cur): Fr.one,      // acc_cur=1
            ColumnRef(column: 1, rotation: .prev): Fr.zero,    // acc_prev=0
        ]

        var result = gate.evaluate(rotations: rotations, challenges: [])
        expect(frEqual(result, Fr.zero), "BinaryDecompose: bit=1, acc 0->1 is valid")

        // bit=0, acc_prev=1, acc_cur=1 at index 1 (1 + 2^1*0 = 1)
        let gate1 = BinaryDecomposeGate(bits: 4, bitIndex: 1)
        rotations = [
            ColumnRef(column: 0, rotation: .cur): Fr.zero,     // bit=0
            ColumnRef(column: 1, rotation: .cur): Fr.one,      // acc_cur=1
            ColumnRef(column: 1, rotation: .prev): Fr.one,     // acc_prev=1
        ]
        result = gate1.evaluate(rotations: rotations, challenges: [])
        expect(frEqual(result, Fr.zero), "BinaryDecompose: bit=0, acc stays at 1")

        // Non-boolean bit should fail
        rotations[ColumnRef(column: 0, rotation: .cur)] = frFromInt(2)
        result = gate1.evaluate(rotations: rotations, challenges: [])
        expect(!frEqual(result, Fr.zero), "BinaryDecompose: non-boolean bit is non-zero")
    }

    // ========== ComparisonGate ==========
    do {
        let gate = ComparisonGate(bits: 8)
        expect(gate.name == "Comparison", "ComparisonGate name")

        // a=10, b=20, diff = 20-10-1 = 9
        var rotations: [ColumnRef: Fr] = [
            ColumnRef(column: 0, rotation: .cur): frFromInt(10),
            ColumnRef(column: 1, rotation: .cur): frFromInt(20),
            ColumnRef(column: 2, rotation: .cur): frFromInt(9),
        ]

        var result = gate.evaluate(rotations: rotations, challenges: [])
        expect(frEqual(result, Fr.zero), "ComparisonGate: a=10 < b=20, diff=9 is valid")

        // Wrong diff
        rotations[ColumnRef(column: 2, rotation: .cur)] = frFromInt(42)
        result = gate.evaluate(rotations: rotations, challenges: [])
        expect(!frEqual(result, Fr.zero), "ComparisonGate: wrong diff is non-zero")
    }

    // ========== BinaryDecomposeHelper ==========
    do {
        let compiler = PlonkConstraintCompiler()
        let valueVar = compiler.addVariable()
        let selectorIdx = compiler.allocateCustomSelector()
        let (bitVars, gateDescs) = BinaryDecomposeHelper.expand(
            compiler: compiler, valueVar: valueVar, bits: 8,
            selectorIndex: selectorIdx)
        expect(bitVars.count == 8, "BinaryDecomposeHelper: 8 bit variables")
        expect(gateDescs.count == 8, "BinaryDecomposeHelper: 8 gate descriptors")
    }
}
