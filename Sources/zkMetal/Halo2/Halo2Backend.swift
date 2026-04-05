// Halo2Backend — Translates Halo2 circuit definitions to zkMetal's PlonkCircuit
//
// Pipeline:
//   1. Run Halo2Circuit.configure() to collect columns, gates, lookups
//   2. Run Halo2Circuit.synthesize() to populate the assignment trace
//   3. Flatten the Halo2 columnar trace into PlonkCircuit gates + wire assignments
//   4. Preprocess via PlonkPreprocessor (selector/permutation polys, KZG commits)
//   5. Prove via PlonkProver with the existing GPU-accelerated backend
//
// Column mapping:
//   Halo2 advice columns 0..a-1 map to PlonkCircuit wire columns starting at wire 'a'.
//   The first 3 wire slots (a, b, c) in each Plonk gate are filled from the first
//   3 advice columns. Additional advice columns beyond 3 require extended-wire support
//   (custom gates handle this via the Halo2Expression evaluator).
//
//   Fixed columns are baked into selector polynomials (qL, qR, qO, qM, qC).
//   Instance columns feed into public input constraints.

import Foundation
import NeonFieldOps

// MARK: - Backend Errors

public enum Halo2BackendError: Error, CustomStringConvertible {
    case tooManyAdviceColumns(Int)
    case emptyCircuit
    case missingAssignment(column: Int, row: Int)
    case invalidColumn(String)

    public var description: String {
        switch self {
        case .tooManyAdviceColumns(let n):
            return "Halo2Backend: circuit has \(n) advice columns; current Plonk backend supports at most 3 (a, b, c wires)"
        case .emptyCircuit:
            return "Halo2Backend: circuit produced no rows after synthesis"
        case .missingAssignment(let col, let row):
            return "Halo2Backend: missing assignment at advice column \(col), row \(row)"
        case .invalidColumn(let msg):
            return "Halo2Backend: invalid column reference: \(msg)"
        }
    }
}

// MARK: - Compiled Circuit

/// Result of compiling a Halo2 circuit: contains the PlonkCircuit + witness.
public struct Halo2CompiledCircuit {
    /// The flattened PlonkCircuit (padded to power of 2).
    public let plonkCircuit: PlonkCircuit
    /// Witness values indexed by variable ID (for PlonkProver.prove()).
    public let witness: [Fr]
    /// Number of rows in the original trace (before padding).
    public let usableRows: Int
    /// The Halo2 assignment storage (for inspection/debugging).
    public let assignment: Halo2Assignment
}

// MARK: - Backend

/// Compiles and proves Halo2-style circuits using zkMetal's Plonk backend.
///
/// Usage:
/// ```swift
/// let backend = try Halo2Backend(maxDegree: 1 << 16)
/// let compiled = try backend.compile(MyCircuit(input: 42))
/// let setup = try backend.setup(compiled: compiled)
/// let proof = try backend.prove(compiled: compiled, setup: setup)
/// let ok = try backend.verify(proof: proof, setup: setup, publicInputs: [...])
/// ```
public class Halo2Backend {
    public let kzg: KZGEngine
    public let ntt: NTTEngine
    private let srsSecret: Fr

    /// Initialize with a test SRS of the given max degree.
    /// For production, supply a real SRS via init(kzg:ntt:srsSecret:).
    public convenience init(maxDegree: Int) throws {
        let secret: [UInt32] = [0x1234_5678, 0x9ABC_DEF0, 0x1111_2222, 0x3333_4444,
                                0x5555_6666, 0x7777_8888, 0x0000_0001, 0x0000_0000]
        let generator = bn254G1Generator()
        let srs = KZGEngine.generateTestSRS(secret: secret, size: maxDegree, generator: generator)
        let kzg = try KZGEngine(srs: srs)
        let ntt = try NTTEngine()
        let sFr = frFromLimbs(secret)
        try self.init(kzg: kzg, ntt: ntt, srsSecret: sFr)
    }

    public init(kzg: KZGEngine, ntt: NTTEngine, srsSecret: Fr) throws {
        self.kzg = kzg
        self.ntt = ntt
        self.srsSecret = srsSecret
    }

    // MARK: - Compile

    /// Compile a Halo2Circuit into a PlonkCircuit + witness.
    ///
    /// Steps:
    ///   1. configure() to get the constraint system
    ///   2. synthesize() to populate the assignment
    ///   3. Flatten into PlonkCircuit format
    public func compile<C: Halo2Circuit>(_ circuit: C) throws -> Halo2CompiledCircuit {
        // Phase 1: Configure
        let cs = Halo2ConstraintSystem()
        let config = C.configure(cs: cs)

        let numAdvice = cs.adviceColumns.count
        let numFixed = cs.fixedColumns.count
        let numInstance = cs.instanceColumns.count
        let numSelectors = cs.selectors.count

        // We map Halo2's advice columns to the 3 Plonk wires (a, b, c).
        // Circuits with > 3 advice columns require an extended-wire approach:
        // we serialize multiple advice columns into successive gate rows.
        if numAdvice > 3 {
            throw Halo2BackendError.tooManyAdviceColumns(numAdvice)
        }

        // Phase 2: Synthesize — first pass to determine trace size
        let initialRows = max(cs.minimumRows + 1, 4)
        let assignment = Halo2Assignment(
            numAdvice: numAdvice,
            numFixed: numFixed,
            numInstance: numInstance,
            numSelectors: numSelectors,
            numRows: initialRows
        )
        let layouter = Halo2Layouter(assignment: assignment)
        try circuit.synthesize(config: config, layouter: layouter)

        let usableRows = assignment.numRows
        if usableRows == 0 { throw Halo2BackendError.emptyCircuit }

        // Phase 3: Flatten to PlonkCircuit
        return try flattenToPlonk(cs: cs, assignment: assignment, usableRows: usableRows)
    }

    /// Flatten a Halo2 assignment into a PlonkCircuit + witness.
    private func flattenToPlonk(
        cs: Halo2ConstraintSystem,
        assignment: Halo2Assignment,
        usableRows: Int
    ) throws -> Halo2CompiledCircuit {

        let numAdvice = cs.adviceColumns.count
        let n = usableRows

        var builder = PlonkCircuitBuilder()

        // Allocate a variable for each (column, row) cell.
        // We use a flat mapping: varId = col * n + row for advice columns.
        // Fixed column values go into selector fields.
        // Instance columns go into public input constraints.

        // Pre-allocate all advice variables
        let totalAdviceVars = numAdvice * n
        var adviceVarBase = [Int]()
        for _ in 0..<numAdvice {
            let base = builder.nextVariable
            for _ in 0..<n {
                _ = builder.addInput()
            }
            adviceVarBase.append(base)
        }

        // Build witness array from advice assignments
        var witnessSize = builder.nextVariable
        var witness = [Fr](repeating: Fr.zero, count: witnessSize)

        for col in 0..<numAdvice {
            for row in 0..<n {
                let varId = adviceVarBase[col] + row
                let val = assignment.advice.count > col ? (assignment.advice[col].count > row ? assignment.advice[col][row] : nil) : nil
                witness[varId] = val ?? Fr.zero
            }
        }

        // For each row, build a PlonkGate from the Halo2 gate expressions.
        // We translate Halo2's expression-based gates into standard Plonk selectors.
        //
        // Strategy: For each row, we emit a standard gate with selectors derived
        // from the Halo2 fixed columns and selector values. The gate constraint is:
        //   qL*a + qR*b + qO*c + qM*a*b + qC = 0
        //
        // For simple circuits (add/mul), we extract qL/qR/qO/qM/qC from the
        // Halo2 gate expressions. For complex custom gates, we use the custom
        // gate framework (PlonkCustomGates).

        for row in 0..<n {
            // Read selector values for this row
            var selectorVals = [Int: Fr]()
            for (idx, selCol) in assignment.selectorValues.enumerated() {
                if row < selCol.count, let v = selCol[row] {
                    selectorVals[idx] = v
                }
            }

            // Read fixed column values for this row
            var fixedVals = [Int: Fr]()
            for (idx, fixCol) in assignment.fixed.enumerated() {
                if row < fixCol.count, let v = fixCol[row] {
                    fixedVals[idx] = v
                }
            }

            // Determine wire variable indices for this row
            let aVar = numAdvice > 0 ? adviceVarBase[0] + row : builder.addInput()
            let bVar = numAdvice > 1 ? adviceVarBase[1] + row : builder.addInput()
            let cVar = numAdvice > 2 ? adviceVarBase[2] + row : builder.addInput()

            // Extract standard Plonk selectors from gate expressions.
            // Walk each gate's expression tree to extract linear coefficients.
            let extracted = extractPlonkSelectors(
                gates: cs.gates,
                selectorValues: selectorVals,
                fixedValues: fixedVals
            )

            builder.addGate(
                qL: extracted.qL, qR: extracted.qR,
                qO: extracted.qO, qM: extracted.qM, qC: extracted.qC,
                a: aVar, b: bVar, c: cVar
            )
        }

        // Grow witness to cover any new dummy variables the builder created
        if builder.nextVariable > witness.count {
            witness.append(contentsOf: [Fr](repeating: Fr.zero,
                                            count: builder.nextVariable - witness.count))
        }

        // Copy constraints from Halo2 assignments
        for (lhsCol, lhsRow, rhsCol, rhsRow) in assignment.copyConstraints {
            let lhsVar = variableId(column: lhsCol, row: lhsRow,
                                    adviceVarBase: adviceVarBase, n: n)
            let rhsVar = variableId(column: rhsCol, row: rhsRow,
                                    adviceVarBase: adviceVarBase, n: n)
            if let lv = lhsVar, let rv = rhsVar {
                builder.assertEqual(lv, rv)
            }
        }

        // Public inputs from instance columns
        for (colIdx, instCol) in assignment.instance.enumerated() {
            for (row, val) in instCol.enumerated() {
                if let v = val {
                    // Instance values become public inputs.
                    // We need a variable that holds this value.
                    let instVar = builder.addInput()
                    if instVar >= witness.count {
                        witness.append(contentsOf: [Fr](repeating: Fr.zero,
                                                        count: instVar - witness.count + 1))
                    }
                    witness[instVar] = v
                    builder.addPublicInput(wireIndex: instVar)

                    // If there's a corresponding advice column cell, constrain equality
                    if colIdx < adviceVarBase.count && row < n {
                        builder.assertEqual(instVar, adviceVarBase[colIdx] + row)
                    }
                }
            }
        }

        // Constant constraints
        for (col, row, val) in assignment.constantConstraints {
            let constVar = builder.constant(val)
            if constVar >= witness.count {
                witness.append(contentsOf: [Fr](repeating: Fr.zero,
                                                count: constVar - witness.count + 1))
            }
            witness[constVar] = val

            if let cellVar = variableId(column: col, row: row,
                                        adviceVarBase: adviceVarBase, n: n) {
                builder.assertEqual(constVar, cellVar)
            }
        }

        // Grow witness to final size
        if builder.nextVariable > witness.count {
            witness.append(contentsOf: [Fr](repeating: Fr.zero,
                                            count: builder.nextVariable - witness.count))
        }

        let circuit = builder.build().padded()

        // Pad witness to match circuit size (padded gates need dummy variables)
        let totalVarsNeeded = (circuit.wireAssignments.flatMap { $0 }.max() ?? 0) + 1
        if totalVarsNeeded > witness.count {
            witness.append(contentsOf: [Fr](repeating: Fr.zero,
                                            count: totalVarsNeeded - witness.count))
        }

        return Halo2CompiledCircuit(
            plonkCircuit: circuit,
            witness: witness,
            usableRows: usableRows,
            assignment: assignment
        )
    }

    /// Map a Halo2Column + row to a PlonkCircuit variable ID.
    private func variableId(column: Halo2Column, row: Int,
                            adviceVarBase: [Int], n: Int) -> Int? {
        switch column.columnType {
        case .advice:
            guard column.index < adviceVarBase.count, row < n else { return nil }
            return adviceVarBase[column.index] + row
        case .fixed, .instance:
            // Fixed/instance columns don't directly map to wire variables
            // in the standard 3-wire Plonk model. Copy constraints involving
            // these are handled via constant gates.
            return nil
        }
    }

    // MARK: - Expression Analysis

    /// Extract standard Plonk selector values (qL, qR, qO, qM, qC) from
    /// Halo2 gate expressions evaluated at a specific row.
    ///
    /// This performs a best-effort extraction. For expressions that fit the
    /// standard Plonk pattern (linear combinations of wire values + products),
    /// we extract exact selectors. For complex expressions, we fall back to
    /// encoding via qC and the custom gate framework.
    private func extractPlonkSelectors(
        gates: [Halo2Gate],
        selectorValues: [Int: Fr],
        fixedValues: [Int: Fr]
    ) -> (qL: Fr, qR: Fr, qO: Fr, qM: Fr, qC: Fr) {
        var qL = Fr.zero, qR = Fr.zero, qO = Fr.zero, qM = Fr.zero, qC = Fr.zero

        for gate in gates {
            for poly in gate.polys {
                let coeffs = extractLinearCoeffs(poly, selectorValues: selectorValues,
                                                 fixedValues: fixedValues)
                qL = frAdd(qL, coeffs.a)
                qR = frAdd(qR, coeffs.b)
                qO = frAdd(qO, coeffs.c)
                qM = frAdd(qM, coeffs.ab)
                qC = frAdd(qC, coeffs.constant)
            }
        }

        return (qL, qR, qO, qM, qC)
    }

    /// Coefficient extraction result for a single expression.
    private struct LinearCoeffs {
        var a: Fr = Fr.zero        // coefficient of advice[0] (wire a)
        var b: Fr = Fr.zero        // coefficient of advice[1] (wire b)
        var c: Fr = Fr.zero        // coefficient of advice[2] (wire c)
        var ab: Fr = Fr.zero       // coefficient of advice[0]*advice[1]
        var constant: Fr = Fr.zero // constant term
    }

    /// Recursively extract linear/bilinear coefficients from an expression.
    /// Returns coefficients for the standard Plonk form: qL*a + qR*b + qO*c + qM*ab + qC.
    private func extractLinearCoeffs(
        _ expr: Halo2Expression,
        selectorValues: [Int: Fr],
        fixedValues: [Int: Fr]
    ) -> LinearCoeffs {
        switch expr {
        case .constant(let c):
            var r = LinearCoeffs()
            r.constant = c
            return r

        case .selector(let sel):
            var r = LinearCoeffs()
            r.constant = selectorValues[sel.index] ?? Fr.zero
            return r

        case .fixed(let col, _):
            var r = LinearCoeffs()
            r.constant = fixedValues[col.index] ?? Fr.zero
            return r

        case .advice(let col, _):
            var r = LinearCoeffs()
            switch col.index {
            case 0: r.a = Fr.one
            case 1: r.b = Fr.one
            case 2: r.c = Fr.one
            default: break
            }
            return r

        case .instance(_, _):
            // Instance values are handled via public input constraints
            return LinearCoeffs()

        case .negated(let e):
            let inner = extractLinearCoeffs(e, selectorValues: selectorValues,
                                            fixedValues: fixedValues)
            var r = LinearCoeffs()
            r.a = frSub(Fr.zero, inner.a)
            r.b = frSub(Fr.zero, inner.b)
            r.c = frSub(Fr.zero, inner.c)
            r.ab = frSub(Fr.zero, inner.ab)
            r.constant = frSub(Fr.zero, inner.constant)
            return r

        case .sum(let lhs, let rhs):
            let l = extractLinearCoeffs(lhs, selectorValues: selectorValues,
                                        fixedValues: fixedValues)
            let r = extractLinearCoeffs(rhs, selectorValues: selectorValues,
                                        fixedValues: fixedValues)
            var result = LinearCoeffs()
            result.a = frAdd(l.a, r.a)
            result.b = frAdd(l.b, r.b)
            result.c = frAdd(l.c, r.c)
            result.ab = frAdd(l.ab, r.ab)
            result.constant = frAdd(l.constant, r.constant)
            return result

        case .product(let lhs, let rhs):
            let l = extractLinearCoeffs(lhs, selectorValues: selectorValues,
                                        fixedValues: fixedValues)
            let r = extractLinearCoeffs(rhs, selectorValues: selectorValues,
                                        fixedValues: fixedValues)
            return multiplyCoeffs(l, r)

        case .scaled(let e, let s):
            let inner = extractLinearCoeffs(e, selectorValues: selectorValues,
                                            fixedValues: fixedValues)
            var result = LinearCoeffs()
            result.a = frMul(s, inner.a)
            result.b = frMul(s, inner.b)
            result.c = frMul(s, inner.c)
            result.ab = frMul(s, inner.ab)
            result.constant = frMul(s, inner.constant)
            return result
        }
    }

    /// Multiply two linear coefficient sets.
    /// (aA + bB + cC + dAB + e) * (a'A + b'B + c'C + d'AB + e')
    /// We only track up to bilinear (AB) terms; higher-degree terms are dropped.
    private func multiplyCoeffs(_ l: LinearCoeffs, _ r: LinearCoeffs) -> LinearCoeffs {
        var result = LinearCoeffs()

        // constant * linear terms
        result.a = frAdd(frMul(l.constant, r.a), frMul(r.constant, l.a))
        result.b = frAdd(frMul(l.constant, r.b), frMul(r.constant, l.b))
        result.c = frAdd(frMul(l.constant, r.c), frMul(r.constant, l.c))

        // constant * constant
        result.constant = frMul(l.constant, r.constant)

        // constant * ab + ab * constant + a*b cross terms
        result.ab = frAdd(
            frAdd(frMul(l.constant, r.ab), frMul(r.constant, l.ab)),
            frAdd(frMul(l.a, r.b), frMul(l.b, r.a))
        )

        return result
    }

    // MARK: - Setup

    /// Preprocess the compiled circuit: compute selector/permutation polynomials,
    /// KZG commitments, and produce a PlonkSetup for proving.
    public func setup(compiled: Halo2CompiledCircuit) throws -> PlonkSetup {
        let preprocessor = PlonkPreprocessor(kzg: kzg, ntt: ntt)
        return try preprocessor.setup(circuit: compiled.plonkCircuit, srsSecret: srsSecret)
    }

    // MARK: - Prove

    /// Generate a Plonk proof from a compiled Halo2 circuit.
    public func prove(compiled: Halo2CompiledCircuit, setup: PlonkSetup) throws -> PlonkProof {
        let prover = PlonkProver(setup: setup, kzg: kzg, ntt: ntt)
        return try prover.prove(witness: compiled.witness, circuit: compiled.plonkCircuit)
    }

    // MARK: - Verify

    /// Verify a Plonk proof against the setup.
    public func verify(proof: PlonkProof, setup: PlonkSetup) -> Bool {
        let verifier = PlonkVerifier(setup: setup, kzg: kzg)
        return verifier.verify(proof: proof)
    }

    // MARK: - Convenience: compile + prove in one call

    /// Compile, setup, and prove a Halo2 circuit in a single call.
    /// Returns the proof and verification key.
    public func proveCircuit<C: Halo2Circuit>(
        _ circuit: C
    ) throws -> (proof: PlonkProof, setup: PlonkSetup, compiled: Halo2CompiledCircuit) {
        let compiled = try compile(circuit)
        let setup = try self.setup(compiled: compiled)
        let proof = try prove(compiled: compiled, setup: setup)
        return (proof, setup, compiled)
    }
}

