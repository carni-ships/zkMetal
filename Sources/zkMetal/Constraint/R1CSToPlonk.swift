// R1CSToPlonk — Convert R1CS constraint systems to Plonk circuit format
//
// For each R1CS constraint: (sum a_i * z_i) * (sum b_j * z_j) = (sum c_k * z_k)
// we decompose into Plonk arithmetic gates: qL*a + qR*b + qO*c + qM*a*b + qC = 0
//
// Conversion strategies:
//   - Linear constraint (A or B is a single term with coefficient 1 on variable 0):
//     directly emits a linear Plonk gate without multiplication
//   - Quadratic constraint (A and B each have a single term):
//     single Plonk gate with qM
//   - Complex constraint (A or B has multiple terms):
//     introduces auxiliary variables to reduce to simple products

import Foundation

// MARK: - Conversion Statistics

public struct R1CSToPlonkStats {
    public let r1csConstraints: Int
    public let plonkGates: Int
    public let auxiliaryVariables: Int
    public let linearConstraints: Int
    public let quadraticConstraints: Int
    public let complexConstraints: Int

    public var conversionRatio: Double {
        guard r1csConstraints > 0 else { return 0 }
        return Double(plonkGates) / Double(r1csConstraints)
    }
}

// MARK: - R1CS to Plonk Converter

public struct R1CSToPlonkConverter {

    /// Convert a SpartanR1CS into a PlonkCircuit with witness values.
    ///
    /// - Parameters:
    ///   - r1cs: The R1CS instance (sparse matrices A, B, C)
    ///   - z: The full witness vector z = [1, public_inputs..., witness...]
    /// - Returns: A tuple of (PlonkCircuit, witness values indexed by variable, conversion stats)
    public static func convert(
        r1cs: SpartanR1CS,
        z: [Fr]
    ) -> (circuit: PlonkCircuit, witness: [Int: Fr], stats: R1CSToPlonkStats) {
        // Group sparse entries by row for each matrix
        let aByRow = groupByRow(r1cs.A, numRows: r1cs.numConstraints)
        let bByRow = groupByRow(r1cs.B, numRows: r1cs.numConstraints)
        let cByRow = groupByRow(r1cs.C, numRows: r1cs.numConstraints)

        let builder = PlonkCircuitBuilder()
        // Allocate variables matching R1CS variable indices
        // Variable 0 in R1CS is constant 1; we represent it implicitly via qC
        // Variables 1..numVariables-1 get Plonk wire variables
        var r1csToPlonk = [Int: Int]()  // R1CS var index -> Plonk variable
        var witness = [Int: Fr]()       // Plonk variable -> value

        // Pre-allocate Plonk variables for each R1CS variable (skip 0 = constant 1)
        for i in 1..<r1cs.numVariables {
            let pVar = builder.addInput()
            r1csToPlonk[i] = pVar
            if i < z.count {
                witness[pVar] = z[i]
            }
        }

        var linearCount = 0
        var quadraticCount = 0
        var complexCount = 0
        let auxBefore = builder.nextVariable

        for row in 0..<r1cs.numConstraints {
            let aTerms = aByRow[row]
            let bTerms = bByRow[row]
            let cTerms = cByRow[row]

            // Classify: is A or B effectively "just 1" (i.e., single entry on variable 0 with coeff 1)?
            let aIsOne = isConstantOne(aTerms)
            let bIsOne = isConstantOne(bTerms)
            let aIsSingle = nonConstantTerms(aTerms).count <= 1
            let bIsSingle = nonConstantTerms(bTerms).count <= 1

            if aIsOne || bIsOne {
                // Linear constraint: one side is constant 1
                // (sum b_j * z_j) = (sum c_k * z_k) [if A is 1]
                // or (sum a_i * z_i) = (sum c_k * z_k) [if B is 1]
                let lhsTerms = aIsOne ? bTerms : aTerms
                emitLinearEquality(
                    lhs: lhsTerms, rhs: cTerms,
                    builder: builder, r1csToPlonk: &r1csToPlonk,
                    witness: &witness, z: z
                )
                linearCount += 1
            } else if aIsSingle && bIsSingle {
                // Quadratic: single-term A * single-term B = C
                emitQuadraticConstraint(
                    aTerms: aTerms, bTerms: bTerms, cTerms: cTerms,
                    builder: builder, r1csToPlonk: &r1csToPlonk,
                    witness: &witness, z: z
                )
                quadraticCount += 1
            } else {
                // Complex: multi-term A or B, introduce auxiliary variables
                emitComplexConstraint(
                    aTerms: aTerms, bTerms: bTerms, cTerms: cTerms,
                    builder: builder, r1csToPlonk: &r1csToPlonk,
                    witness: &witness, z: z
                )
                complexCount += 1
            }
        }

        // Mark public inputs
        if r1cs.numPublic > 0 {
            for i in 1...r1cs.numPublic {
                if let pVar = r1csToPlonk[i] {
                    builder.addPublicInput(wireIndex: pVar)
                }
            }
        }

        let circuit = builder.build()
        let stats = R1CSToPlonkStats(
            r1csConstraints: r1cs.numConstraints,
            plonkGates: circuit.numGates,
            auxiliaryVariables: builder.nextVariable - auxBefore,
            linearConstraints: linearCount,
            quadraticConstraints: quadraticCount,
            complexConstraints: complexCount
        )

        return (circuit, witness, stats)
    }

    // MARK: - Helpers

    /// Group sparse entries by row
    private static func groupByRow(_ entries: [SpartanEntry], numRows: Int) -> [[SpartanEntry]] {
        var result = [[SpartanEntry]](repeating: [], count: numRows)
        for e in entries {
            if e.row < numRows { result[e.row].append(e) }
        }
        return result
    }

    /// Check if a set of terms equals "just 1" (single entry on col 0 with value 1)
    private static func isConstantOne(_ terms: [SpartanEntry]) -> Bool {
        guard terms.count == 1 else { return false }
        return terms[0].col == 0 && terms[0].value == Fr.one
    }

    /// Return terms that are NOT the constant-1 variable (col != 0)
    private static func nonConstantTerms(_ terms: [SpartanEntry]) -> [SpartanEntry] {
        terms.filter { $0.col != 0 }
    }

    /// Constant coefficient: the entry on col 0 (if any)
    private static func constantCoeff(_ terms: [SpartanEntry]) -> Fr {
        for t in terms where t.col == 0 { return t.value }
        return Fr.zero
    }

    /// Compute the value of a linear combination from z
    private static func evalLC(_ terms: [SpartanEntry], z: [Fr]) -> Fr {
        var result = Fr.zero
        for t in terms {
            if t.col < z.count {
                result = frAdd(result, frMul(t.value, z[t.col]))
            }
        }
        return result
    }

    /// Get or create a Plonk variable for an R1CS variable index.
    /// Variable 0 (constant 1) is never mapped; use qC instead.
    private static func plonkVar(
        for r1csVar: Int,
        builder: PlonkCircuitBuilder,
        r1csToPlonk: inout [Int: Int],
        witness: inout [Int: Fr],
        z: [Fr]
    ) -> Int {
        if let existing = r1csToPlonk[r1csVar] { return existing }
        let v = builder.addInput()
        r1csToPlonk[r1csVar] = v
        if r1csVar < z.count { witness[v] = z[r1csVar] }
        return v
    }

    /// Materialize a linear combination as a single Plonk variable.
    /// Returns the variable holding sum(coeff_i * var_i) + constant.
    private static func materializeLC(
        terms: [SpartanEntry],
        builder: PlonkCircuitBuilder,
        r1csToPlonk: inout [Int: Int],
        witness: inout [Int: Fr],
        z: [Fr]
    ) -> Int {
        let nonConst = nonConstantTerms(terms)
        let constCoeff = constantCoeff(terms)

        if nonConst.isEmpty {
            // Pure constant
            let v = builder.addInput()
            witness[v] = constCoeff.isZero ? Fr.zero : constCoeff
            // Add gate: qO*c + qC = 0 => c = -qC => c = constCoeff
            let dummy = builder.addInput()
            witness[dummy] = Fr.zero
            builder.addGate(
                qL: Fr.zero, qR: Fr.zero,
                qO: frSub(Fr.zero, Fr.one),
                qM: Fr.zero, qC: constCoeff,
                a: dummy, b: dummy, c: v
            )
            return v
        }

        // Start with first term
        var accVar = plonkVar(for: nonConst[0].col, builder: builder,
                              r1csToPlonk: &r1csToPlonk, witness: &witness, z: z)
        var accVal = witness[accVar] ?? Fr.zero

        // If first coefficient is not 1, scale it
        if nonConst[0].value != Fr.one {
            let scaled = builder.addInput()
            let scaledVal = frMul(nonConst[0].value, accVal)
            witness[scaled] = scaledVal
            let dummy = builder.addInput()
            witness[dummy] = Fr.zero
            // qL * a + qO * c = 0 => coeff * a - c = 0 => c = coeff * a
            builder.addGate(
                qL: nonConst[0].value, qR: Fr.zero,
                qO: frSub(Fr.zero, Fr.one),
                qM: Fr.zero, qC: Fr.zero,
                a: accVar, b: dummy, c: scaled
            )
            accVar = scaled
            accVal = scaledVal
        }

        // Accumulate remaining terms
        for i in 1..<nonConst.count {
            let t = nonConst[i]
            let termVar = plonkVar(for: t.col, builder: builder,
                                   r1csToPlonk: &r1csToPlonk, witness: &witness, z: z)
            let termVal = witness[termVar] ?? Fr.zero

            let newAcc = builder.addInput()
            let newAccVal = frAdd(accVal, frMul(t.value, termVal))
            witness[newAcc] = newAccVal

            // qL*acc + qR*term - c = 0 => c = acc + coeff*term
            builder.addGate(
                qL: Fr.one, qR: t.value,
                qO: frSub(Fr.zero, Fr.one),
                qM: Fr.zero, qC: Fr.zero,
                a: accVar, b: termVar, c: newAcc
            )
            accVar = newAcc
            accVal = newAccVal
        }

        // Add constant if nonzero
        if !constCoeff.isZero {
            let newAcc = builder.addInput()
            let newAccVal = frAdd(accVal, constCoeff)
            witness[newAcc] = newAccVal
            let dummy = builder.addInput()
            witness[dummy] = Fr.zero
            // qL*acc + qC - c = 0 => c = acc + const
            builder.addGate(
                qL: Fr.one, qR: Fr.zero,
                qO: frSub(Fr.zero, Fr.one),
                qM: Fr.zero, qC: constCoeff,
                a: accVar, b: dummy, c: newAcc
            )
            accVar = newAcc
        }

        return accVar
    }

    // MARK: - Constraint Emitters

    /// Emit gates for a linear equality: lhs_linear_combination = rhs_linear_combination
    private static func emitLinearEquality(
        lhs: [SpartanEntry], rhs: [SpartanEntry],
        builder: PlonkCircuitBuilder,
        r1csToPlonk: inout [Int: Int],
        witness: inout [Int: Fr],
        z: [Fr]
    ) {
        // Convert lhs - rhs = 0 into Plonk gates
        // Simple case: both sides have 1 non-constant term
        let lhsNC = nonConstantTerms(lhs)
        let rhsNC = nonConstantTerms(rhs)
        let lhsConst = constantCoeff(lhs)
        let rhsConst = constantCoeff(rhs)

        if lhsNC.count <= 1 && rhsNC.count <= 1 {
            // qL*a + qR*b + qC = 0 where a is LHS var, b is RHS var
            let negOne = frSub(Fr.zero, Fr.one)

            let aVar: Int
            let qL: Fr
            if lhsNC.count == 1 {
                aVar = plonkVar(for: lhsNC[0].col, builder: builder,
                                r1csToPlonk: &r1csToPlonk, witness: &witness, z: z)
                qL = lhsNC[0].value
            } else {
                aVar = builder.addInput()
                witness[aVar] = Fr.zero
                qL = Fr.zero
            }

            let bVar: Int
            let qR: Fr
            if rhsNC.count == 1 {
                bVar = plonkVar(for: rhsNC[0].col, builder: builder,
                                r1csToPlonk: &r1csToPlonk, witness: &witness, z: z)
                qR = frSub(Fr.zero, rhsNC[0].value) // negate since rhs
            } else {
                bVar = builder.addInput()
                witness[bVar] = Fr.zero
                qR = Fr.zero
            }

            let dummy = builder.addInput()
            witness[dummy] = Fr.zero

            // qL*a + qR*b + qC = 0
            // qC = lhsConst - rhsConst
            let qC = frSub(lhsConst, rhsConst)
            builder.addGate(
                qL: qL, qR: qR, qO: Fr.zero, qM: Fr.zero, qC: qC,
                a: aVar, b: bVar, c: dummy
            )
            return
        }

        // General case: materialize both sides and assert equal
        let lhsVar = materializeLC(terms: lhs, builder: builder,
                                   r1csToPlonk: &r1csToPlonk, witness: &witness, z: z)
        let rhsVar = materializeLC(terms: rhs, builder: builder,
                                   r1csToPlonk: &r1csToPlonk, witness: &witness, z: z)
        builder.assertEqual(lhsVar, rhsVar)
    }

    /// Emit a single Plonk gate for a quadratic constraint: a_coeff*A_var * b_coeff*B_var = C
    private static func emitQuadraticConstraint(
        aTerms: [SpartanEntry], bTerms: [SpartanEntry], cTerms: [SpartanEntry],
        builder: PlonkCircuitBuilder,
        r1csToPlonk: inout [Int: Int],
        witness: inout [Int: Fr],
        z: [Fr]
    ) {
        let aNC = nonConstantTerms(aTerms)
        let bNC = nonConstantTerms(bTerms)
        let aConst = constantCoeff(aTerms)
        let bConst = constantCoeff(bTerms)

        // A = a_coeff * a_var + a_const, B = b_coeff * b_var + b_const
        // A * B = C => a_coeff*b_coeff * a*b + a_coeff*b_const * a + b_coeff*a_const * b + a_const*b_const = C

        let aVar: Int
        let aCoeff: Fr
        if aNC.count == 1 {
            aVar = plonkVar(for: aNC[0].col, builder: builder,
                            r1csToPlonk: &r1csToPlonk, witness: &witness, z: z)
            aCoeff = aNC[0].value
        } else {
            aVar = builder.addInput()
            witness[aVar] = Fr.zero
            aCoeff = Fr.zero
        }

        let bVar: Int
        let bCoeff: Fr
        if bNC.count == 1 {
            bVar = plonkVar(for: bNC[0].col, builder: builder,
                            r1csToPlonk: &r1csToPlonk, witness: &witness, z: z)
            bCoeff = bNC[0].value
        } else {
            bVar = builder.addInput()
            witness[bVar] = Fr.zero
            bCoeff = Fr.zero
        }

        // C side: materialize or use single variable
        let cNC = nonConstantTerms(cTerms)
        let cConst = constantCoeff(cTerms)

        let cVar: Int
        let qO: Fr
        if cNC.count == 1 {
            cVar = plonkVar(for: cNC[0].col, builder: builder,
                            r1csToPlonk: &r1csToPlonk, witness: &witness, z: z)
            qO = frSub(Fr.zero, cNC[0].value) // -c_coeff since it's on the RHS
        } else if cNC.count == 0 {
            cVar = builder.addInput()
            witness[cVar] = Fr.zero
            qO = Fr.zero
        } else {
            // Multiple terms in C: materialize
            cVar = materializeLC(terms: cTerms, builder: builder,
                                 r1csToPlonk: &r1csToPlonk, witness: &witness, z: z)
            qO = frSub(Fr.zero, Fr.one)
        }

        // Gate: qM*a*b + qL*a + qR*b + qO*c + qC = 0
        let qM = frMul(aCoeff, bCoeff)
        let qL = frMul(aCoeff, bConst)
        let qR = frMul(bCoeff, aConst)
        var qC = frMul(aConst, bConst)
        // Subtract C constant
        if cNC.count <= 1 {
            qC = frSub(qC, cConst)
        }

        builder.addGate(
            qL: qL, qR: qR, qO: qO, qM: qM, qC: qC,
            a: aVar, b: bVar, c: cVar
        )
    }

    /// Emit gates for a complex constraint where A or B has multiple terms.
    /// Introduces auxiliary variable(s) to hold the linear combination, then uses a quadratic gate.
    private static func emitComplexConstraint(
        aTerms: [SpartanEntry], bTerms: [SpartanEntry], cTerms: [SpartanEntry],
        builder: PlonkCircuitBuilder,
        r1csToPlonk: inout [Int: Int],
        witness: inout [Int: Fr],
        z: [Fr]
    ) {
        // Materialize A and B into single variables
        let aVar = materializeLC(terms: aTerms, builder: builder,
                                 r1csToPlonk: &r1csToPlonk, witness: &witness, z: z)
        let aVal = witness[aVar] ?? Fr.zero
        let bVar = materializeLC(terms: bTerms, builder: builder,
                                 r1csToPlonk: &r1csToPlonk, witness: &witness, z: z)
        let bVal = witness[bVar] ?? Fr.zero

        // Materialize C
        let cVar = materializeLC(terms: cTerms, builder: builder,
                                 r1csToPlonk: &r1csToPlonk, witness: &witness, z: z)

        // Now emit: a * b - c = 0
        // But we need to create the product variable
        let prodVar = builder.addInput()
        witness[prodVar] = frMul(aVal, bVal)

        // Gate 1: a * b = prod  =>  qM=1, qO=-1
        builder.addGate(
            qL: Fr.zero, qR: Fr.zero,
            qO: frSub(Fr.zero, Fr.one),
            qM: Fr.one, qC: Fr.zero,
            a: aVar, b: bVar, c: prodVar
        )

        // Gate 2: prod - c = 0
        builder.assertEqual(prodVar, cVar)
    }

    /// Verify that a Plonk circuit is satisfied by the given witness.
    /// Checks each gate: qL*a + qR*b + qO*c + qM*a*b + qC = 0
    public static func verifySatisfaction(
        circuit: PlonkCircuit,
        witness: [Int: Fr]
    ) -> Bool {
        for (i, gate) in circuit.gates.enumerated() {
            let wires = circuit.wireAssignments[i]
            let a = witness[wires[0]] ?? Fr.zero
            let b = witness[wires[1]] ?? Fr.zero
            let c = witness[wires[2]] ?? Fr.zero

            // qL*a + qR*b + qO*c + qM*a*b + qC
            var sum = Fr.zero
            sum = frAdd(sum, frMul(gate.qL, a))
            sum = frAdd(sum, frMul(gate.qR, b))
            sum = frAdd(sum, frMul(gate.qO, c))
            sum = frAdd(sum, frMul(gate.qM, frMul(a, b)))
            sum = frAdd(sum, gate.qC)

            if !sum.isZero { return false }
        }

        // Check copy constraints
        for (varA, varB) in circuit.copyConstraints {
            let valA = witness[varA] ?? Fr.zero
            let valB = witness[varB] ?? Fr.zero
            if valA != valB { return false }
        }

        return true
    }
}
