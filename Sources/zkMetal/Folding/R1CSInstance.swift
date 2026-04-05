// R1CS Instance Types for Nova Folding
//
// Defines strict and relaxed R1CS instances used by the Nova folding scheme.
// A strict R1CS: A*z . B*z = C*z where z = (1, x, W)
// A relaxed R1CS: A*z . B*z = u*(C*z) + E where z = (u, x, W)
//
// These types are distinct from the Groth16 R1CSInstance and the IVC NovaEngine types.
// They use the SparseMatrix (CSR) format from CCS.swift for efficient matvec.
//
// Reference: "Nova: Recursive Zero-Knowledge Arguments from Folding Schemes"
//            (Kothapalli, Setty, Tzialla 2022)

import Foundation
import NeonFieldOps

// MARK: - R1CS Shape

/// Defines the structure of an R1CS: matrices A, B, C as CSR sparse matrices, plus dimensions.
/// The shape is shared across all instances of the same circuit.
public struct NovaR1CSShape {
    public let numConstraints: Int   // m: number of constraint rows
    public let numVariables: Int     // n: total variables including 1, public, witness
    public let numPublicInputs: Int  // l: number of public input elements
    public let A: SparseMatrix       // m x n
    public let B: SparseMatrix       // m x n
    public let C: SparseMatrix       // m x n

    public init(numConstraints: Int, numVariables: Int, numPublicInputs: Int,
                A: SparseMatrix, B: SparseMatrix, C: SparseMatrix) {
        precondition(A.rows == numConstraints && A.cols == numVariables)
        precondition(B.rows == numConstraints && B.cols == numVariables)
        precondition(C.rows == numConstraints && C.cols == numVariables)
        precondition(numPublicInputs + 1 < numVariables, "Need room for at least 1 witness element")
        self.numConstraints = numConstraints
        self.numVariables = numVariables
        self.numPublicInputs = numPublicInputs
        self.A = A
        self.B = B
        self.C = C
    }

    /// Number of witness elements: n - 1 - l
    public var numWitness: Int { numVariables - 1 - numPublicInputs }
}

// MARK: - R1CS Instance (public part)

/// The public portion of an R1CS assignment: just the public input vector x.
public struct NovaR1CSInput {
    public let x: [Fr]  // public input (length = shape.numPublicInputs)

    public init(x: [Fr]) {
        self.x = x
    }
}

// MARK: - R1CS Witness (private part)

/// The private witness portion of an R1CS assignment.
public struct NovaR1CSWitness {
    public let W: [Fr]  // witness vector (length = shape.numWitness)

    public init(W: [Fr]) {
        self.W = W
    }
}

// MARK: - R1CS Satisfaction Check

extension NovaR1CSShape {
    /// Build z = (1, x, W) from instance + witness.
    public func buildZ(instance: NovaR1CSInput, witness: NovaR1CSWitness) -> [Fr] {
        precondition(instance.x.count == numPublicInputs)
        precondition(witness.W.count == numWitness)
        var z = [Fr]()
        z.reserveCapacity(numVariables)
        z.append(Fr.one)
        z.append(contentsOf: instance.x)
        z.append(contentsOf: witness.W)
        return z
    }

    /// Build relaxed z = (u, x, W) for a relaxed instance.
    public func buildRelaxedZ(u: Fr, instance: NovaR1CSInput, witness: NovaR1CSWitness) -> [Fr] {
        precondition(instance.x.count == numPublicInputs)
        precondition(witness.W.count == numWitness)
        var z = [Fr]()
        z.reserveCapacity(numVariables)
        z.append(u)
        z.append(contentsOf: instance.x)
        z.append(contentsOf: witness.W)
        return z
    }

    /// Check A*z . B*z == C*z (strict R1CS satisfaction).
    /// z = (1, x, W). Returns true if satisfied.
    public func satisfies(instance: NovaR1CSInput, witness: NovaR1CSWitness) -> Bool {
        let z = buildZ(instance: instance, witness: witness)
        let az = A.mulVec(z)
        let bz = B.mulVec(z)
        let cz = C.mulVec(z)

        // Check: az[i] * bz[i] == cz[i] for all i
        for i in 0..<numConstraints {
            let lhs = frMul(az[i], bz[i])
            if !frEq(lhs, cz[i]) {
                return false
            }
        }
        return true
    }
}

// MARK: - Relaxed R1CS Instance

/// A relaxed R1CS instance for Nova folding: carries commitment to witness W,
/// commitment to error E, scalar u, and public input x.
///
/// Relaxed R1CS relation: A*z . B*z = u*(C*z) + E
/// where z = (u, x, W).
///
/// A fresh (non-folded) instance has E = 0, u = 1.
public struct NovaRelaxedInstance {
    public let commitW: PointProjective  // Commitment to witness W
    public let commitE: PointProjective  // Commitment to error vector E
    public let u: Fr                     // Relaxation scalar
    public let x: [Fr]                   // Public input

    /// Create a relaxed instance from commitments and public data.
    public init(commitW: PointProjective, commitE: PointProjective, u: Fr, x: [Fr]) {
        self.commitW = commitW
        self.commitE = commitE
        self.u = u
        self.x = x
    }
}

// MARK: - Relaxed R1CS Witness

/// Full witness for a relaxed R1CS instance: W vector and E (error) vector.
public struct NovaRelaxedWitness {
    public let W: [Fr]   // Witness vector
    public let E: [Fr]   // Error vector (length = numConstraints)

    public init(W: [Fr], E: [Fr]) {
        self.W = W
        self.E = E
    }
}

// MARK: - Relaxed R1CS Satisfaction Check

extension NovaR1CSShape {
    /// Check the relaxed R1CS relation: A*z . B*z == u*(C*z) + E
    /// where z = (u, x, W).
    public func satisfiesRelaxed(instance: NovaRelaxedInstance,
                                  witness: NovaRelaxedWitness) -> Bool {
        precondition(witness.E.count == numConstraints, "Error vector length mismatch")
        let input = NovaR1CSInput(x: instance.x)
        let wit = NovaR1CSWitness(W: witness.W)
        let z = buildRelaxedZ(u: instance.u, instance: input, witness: wit)

        let az = A.mulVec(z)
        let bz = B.mulVec(z)
        let cz = C.mulVec(z)

        // Check: az[i] * bz[i] == u * cz[i] + E[i] for all i
        for i in 0..<numConstraints {
            let lhs = frMul(az[i], bz[i])
            let rhs = frAdd(frMul(instance.u, cz[i]), witness.E[i])
            if !frEq(lhs, rhs) {
                return false
            }
        }
        return true
    }
}

// MARK: - Relax: Convert strict R1CS to relaxed

extension NovaR1CSShape {
    /// Convert a strict R1CS instance + witness into a relaxed form.
    /// Sets E = 0 (zero error vector) and u = 1.
    public func relax(instance: NovaR1CSInput, witness: NovaR1CSWitness,
                      pp: PedersenParams) -> (NovaRelaxedInstance, NovaRelaxedWitness) {
        let commitW = pp.commit(witness: witness.W)
        let commitE = pointIdentity()  // Commit(0) = identity point
        let relaxedInst = NovaRelaxedInstance(
            commitW: commitW, commitE: commitE, u: Fr.one, x: instance.x)
        let relaxedWit = NovaRelaxedWitness(
            W: witness.W,
            E: [Fr](repeating: .zero, count: numConstraints))
        return (relaxedInst, relaxedWit)
    }
}
