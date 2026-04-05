// PlonkToR1CS — Convert Plonk circuit format to R1CS constraint system
//
// Each Plonk gate: qL*a + qR*b + qO*c + qM*a*b + qC = 0
// is converted into R1CS form: A*z . B*z = C*z
//
// Strategy:
//   - Gates with qM != 0 (multiplication): split into A*B = C form
//   - Pure linear gates (qM = 0): use the standard (linear)*1 = output trick

import Foundation

// MARK: - Conversion Statistics

public struct PlonkToR1CSStats {
    public let plonkGates: Int
    public let r1csConstraints: Int
    public let totalVariables: Int
    public let auxiliaryVariables: Int
    public let linearGates: Int
    public let multiplicativeGates: Int
}

// MARK: - Plonk to R1CS Converter

public struct PlonkToR1CSConverter {

    /// Convert a PlonkCircuit + witness into a SpartanR1CS + z vector.
    ///
    /// - Parameters:
    ///   - circuit: The Plonk circuit with gates and wire assignments
    ///   - witness: Variable index -> field element mapping
    ///   - numPublic: Number of public inputs (default 0)
    /// - Returns: A tuple of (R1CS instance, z vector, stats)
    public static func convert(
        circuit: PlonkCircuit,
        witness: [Int: Fr],
        numPublic: Int = 0
    ) -> (r1cs: SpartanR1CS, z: [Fr], stats: PlonkToR1CSStats) {
        let builder = SpartanR1CSBuilder()

        // Collect all referenced variable indices
        var allVars = Set<Int>()
        for wires in circuit.wireAssignments {
            for w in wires { allVars.insert(w) }
        }

        // Sort for deterministic ordering
        let sortedVars = allVars.sorted()

        // Map Plonk variable index -> R1CS variable index
        // R1CS layout: z = [1, public_vars..., private_vars...]
        var plonkToR1CS = [Int: Int]()
        let publicSet = Set(circuit.publicInputIndices)

        // Allocate public inputs first
        for pVar in sortedVars where publicSet.contains(pVar) {
            plonkToR1CS[pVar] = builder.addPublicInput()
        }
        // Then private/witness variables
        for pVar in sortedVars where !publicSet.contains(pVar) {
            plonkToR1CS[pVar] = builder.addWitness()
        }

        let baseVarCount = builder.nextVariable
        var linearCount = 0
        var mulCount = 0

        for (i, gate) in circuit.gates.enumerated() {
            let wires = circuit.wireAssignments[i]
            let aR1CS = plonkToR1CS[wires[0]]!
            let bR1CS = plonkToR1CS[wires[1]]!
            let cR1CS = plonkToR1CS[wires[2]]!

            let hasMultiplication = !gate.qM.isZero

            if hasMultiplication {
                mulCount += 1
                // Gate: qL*a + qR*b + qO*c + qM*a*b + qC = 0
                //
                // Rewrite as: qM*a*b = -(qL*a + qR*b + qO*c + qC)
                //
                // If qM = 1 and qL = qR = qC = 0 and qO = -1:
                //   Simple case: a * b = c
                //   A row: a, B row: b, C row: c
                //
                // General case: introduce auxiliary variable aux = qM*a*b
                //   Constraint 1: A=[qM*a_var], B=[b_var], C=[aux]
                //   Constraint 2: aux + qL*a + qR*b + qO*c + qC = 0 (linear, use *1 trick)

                let isSimpleMul = gate.qL.isZero && gate.qR.isZero && gate.qC.isZero
                    && gate.qM == Fr.one
                    && gate.qO == frSub(Fr.zero, Fr.one)

                if isSimpleMul {
                    // a * b = c
                    builder.addConstraint(
                        a: [(aR1CS, Fr.one)],
                        b: [(bR1CS, Fr.one)],
                        c: [(cR1CS, Fr.one)]
                    )
                } else {
                    // General: aux = qM * a * b
                    // Then: aux + qL*a + qR*b + qO*c + qC = 0
                    let aux = builder.addWitness()

                    // Constraint 1: (qM * a) * b = aux
                    builder.addConstraint(
                        a: [(aR1CS, gate.qM)],
                        b: [(bR1CS, Fr.one)],
                        c: [(aux, Fr.one)]
                    )

                    // Constraint 2: aux + qL*a + qR*b + qO*c + qC = 0
                    // => (aux + qL*a + qR*b + qO*c + qC) * 1 = 0
                    // => A = [aux:1, a:qL, b:qR, c:qO, const:qC], B = [1], C = []
                    var aEntries: [(Int, Fr)] = [(aux, Fr.one)]
                    if !gate.qL.isZero { aEntries.append((aR1CS, gate.qL)) }
                    if !gate.qR.isZero { aEntries.append((bR1CS, gate.qR)) }
                    if !gate.qO.isZero { aEntries.append((cR1CS, gate.qO)) }
                    if !gate.qC.isZero { aEntries.append((0, gate.qC)) }

                    builder.addConstraint(
                        a: aEntries,
                        b: [(0, Fr.one)],
                        c: []  // = 0
                    )
                }
            } else {
                linearCount += 1
                // Pure linear: qL*a + qR*b + qO*c + qC = 0
                // Express as: (qL*a + qR*b + qO*c + qC) * 1 = 0
                var aEntries: [(Int, Fr)] = []
                if !gate.qL.isZero { aEntries.append((aR1CS, gate.qL)) }
                if !gate.qR.isZero { aEntries.append((bR1CS, gate.qR)) }
                if !gate.qO.isZero { aEntries.append((cR1CS, gate.qO)) }
                if !gate.qC.isZero { aEntries.append((0, gate.qC)) }

                // If empty (trivially satisfied), skip
                if aEntries.isEmpty { continue }

                builder.addConstraint(
                    a: aEntries,
                    b: [(0, Fr.one)],
                    c: []  // = 0
                )
            }
        }

        // Add copy constraints as equality constraints:
        // For (varA, varB): varA - varB = 0 => (varA - varB) * 1 = 0
        for (varA, varB) in circuit.copyConstraints {
            if let rA = plonkToR1CS[varA], let rB = plonkToR1CS[varB] {
                builder.addConstraint(
                    a: [(rA, Fr.one), (rB, frSub(Fr.zero, Fr.one))],
                    b: [(0, Fr.one)],
                    c: []
                )
            }
        }

        let r1cs = builder.build()

        // Build z vector
        var z = [Fr](repeating: Fr.zero, count: r1cs.numVariables)
        z[0] = Fr.one

        // Fill in variable values from witness
        for (plonkVar, r1csVar) in plonkToR1CS {
            if r1csVar < z.count {
                z[r1csVar] = witness[plonkVar] ?? Fr.zero
            }
        }

        // Fill auxiliary variables (products from general mul gates)
        // We need to compute aux = qM * a * b for each general mul gate
        var auxIdx = baseVarCount
        for (i, gate) in circuit.gates.enumerated() {
            let hasMultiplication = !gate.qM.isZero
            if !hasMultiplication { continue }

            let isSimpleMul = gate.qL.isZero && gate.qR.isZero && gate.qC.isZero
                && gate.qM == Fr.one
                && gate.qO == frSub(Fr.zero, Fr.one)
            if isSimpleMul { continue }

            // General mul: aux = qM * a * b
            let wires = circuit.wireAssignments[i]
            let aVal = witness[wires[0]] ?? Fr.zero
            let bVal = witness[wires[1]] ?? Fr.zero
            let auxVal = frMul(gate.qM, frMul(aVal, bVal))
            if auxIdx < z.count {
                z[auxIdx] = auxVal
            }
            auxIdx += 1
        }

        let stats = PlonkToR1CSStats(
            plonkGates: circuit.numGates,
            r1csConstraints: r1cs.numConstraints,
            totalVariables: r1cs.numVariables,
            auxiliaryVariables: builder.nextVariable - baseVarCount,
            linearGates: linearCount,
            multiplicativeGates: mulCount
        )

        return (r1cs, z, stats)
    }

    /// Build a z vector from a Plonk witness and the variable mapping.
    /// Useful for re-checking satisfaction after round-trip conversion.
    public static func buildZ(
        r1cs: SpartanR1CS,
        plonkWitness: [Int: Fr],
        plonkToR1CS: [Int: Int]
    ) -> [Fr] {
        var z = [Fr](repeating: Fr.zero, count: r1cs.numVariables)
        z[0] = Fr.one
        for (plonkVar, r1csVar) in plonkToR1CS {
            if r1csVar < z.count {
                z[r1csVar] = plonkWitness[plonkVar] ?? Fr.zero
            }
        }
        return z
    }
}
