// Spartan Engine — Transparent SNARK using multilinear extensions + sumcheck
//
// Proves R1CS satisfiability: A*z . B*z = C*z without trusted setup.
// Uses sumcheck protocol + Basefold PCS (transparent, multilinear).
//
// Protocol (Fiat-Shamir):
//   1. Commit to witness z_tilde via Basefold
//   2. Derive random tau, compute val_a, val_b, val_c
//   3. Check val_a * val_b = val_c
//   4. Sumcheck on combined inner product with random linear combination
//   5. Open z_tilde at sumcheck output point via Basefold

import Foundation

// MARK: - Proof Structure

public struct SpartanProof {
    public let witnessCommitment: Fr
    public let claimedValues: (Fr, Fr, Fr)
    public let sumcheckRounds: [(Fr, Fr, Fr)]
    public let sumcheckPoint: [Fr]
    public let zEval: Fr
    public let openingProof: BasefoldProof

    public init(witnessCommitment: Fr, claimedValues: (Fr, Fr, Fr),
                sumcheckRounds: [(Fr, Fr, Fr)], sumcheckPoint: [Fr],
                zEval: Fr, openingProof: BasefoldProof) {
        self.witnessCommitment = witnessCommitment
        self.claimedValues = claimedValues
        self.sumcheckRounds = sumcheckRounds
        self.sumcheckPoint = sumcheckPoint
        self.zEval = zEval
        self.openingProof = openingProof
    }
}

// MARK: - Engine

public class SpartanEngine {
    public static let version = Versions.spartan
    private let basefold: BasefoldEngine

    public init() throws {
        self.basefold = try BasefoldEngine()
    }

    // MARK: - Prove

    public func prove(instance: R1CSInstance, publicInputs: [Fr], witness: [Fr]) throws -> SpartanProof {
        let z = R1CSInstance.buildZ(publicInputs: publicInputs, witness: witness)
        precondition(z.count == instance.numVariables)
        precondition(instance.isSatisfied(z: z), "R1CS not satisfied")

        let logM = instance.logM
        let logN = instance.logN
        let paddedN = instance.paddedN

        let zTilde = instance.buildZTilde(z: z)
        let commitment = try basefold.commit(evaluations: zTilde)

        let transcript = Transcript(label: "spartan-r1cs")
        transcript.absorb(commitment.root)
        transcript.absorbLabel("tau")
        let tau = transcript.squeezeN(logM)

        let eqTau = MultilinearPoly.eqPoly(point: tau)
        let valA = R1CSInstance.sparseInnerProduct(matrix: instance.A, eqTau: eqTau, z: z)
        let valB = R1CSInstance.sparseInnerProduct(matrix: instance.B, eqTau: eqTau, z: z)
        let valC = R1CSInstance.sparseInnerProduct(matrix: instance.C, eqTau: eqTau, z: z)

        transcript.absorbLabel("claims")
        transcript.absorb(valA)
        transcript.absorb(valB)
        transcript.absorb(valC)

        transcript.absorbLabel("combine")
        let ra = transcript.squeeze()
        let rb = transcript.squeeze()
        let combinedClaim = frAdd(frAdd(frMul(ra, valA), frMul(rb, valB)), valC)

        // Build combined weight vector
        var wVec = [Fr](repeating: Fr.zero, count: paddedN)
        for entry in instance.A {
            guard entry.row < eqTau.count && entry.col < paddedN else { continue }
            wVec[entry.col] = frAdd(wVec[entry.col], frMul(ra, frMul(entry.value, eqTau[entry.row])))
        }
        for entry in instance.B {
            guard entry.row < eqTau.count && entry.col < paddedN else { continue }
            wVec[entry.col] = frAdd(wVec[entry.col], frMul(rb, frMul(entry.value, eqTau[entry.row])))
        }
        for entry in instance.C {
            guard entry.row < eqTau.count && entry.col < paddedN else { continue }
            wVec[entry.col] = frAdd(wVec[entry.col], frMul(entry.value, eqTau[entry.row]))
        }

        let (rounds, challenges) = spartanCPUSumcheck(
            wEvals: wVec, zEvals: zTilde, numVars: logN,
            claimedSum: combinedClaim, transcript: transcript
        )

        let zEvalAtR = spartanEvalMultilinear(evals: zTilde, point: challenges)
        let openingProof = try basefold.open(commitment: commitment, point: challenges)

        return SpartanProof(
            witnessCommitment: commitment.root,
            claimedValues: (valA, valB, valC),
            sumcheckRounds: rounds,
            sumcheckPoint: challenges,
            zEval: zEvalAtR,
            openingProof: openingProof
        )
    }

    // MARK: - Verify

    public func verify(instance: R1CSInstance, publicInputs: [Fr], proof: SpartanProof) -> Bool {
        let logM = instance.logM
        let logN = instance.logN
        let paddedN = instance.paddedN

        let transcript = Transcript(label: "spartan-r1cs")
        transcript.absorb(proof.witnessCommitment)
        transcript.absorbLabel("tau")
        let tau = transcript.squeezeN(logM)

        let (valA, valB, valC) = proof.claimedValues
        if frToInt(frMul(valA, valB)) != frToInt(valC) { return false }

        transcript.absorbLabel("claims")
        transcript.absorb(valA)
        transcript.absorb(valB)
        transcript.absorb(valC)

        transcript.absorbLabel("combine")
        let ra = transcript.squeeze()
        let rb = transcript.squeeze()
        let combinedClaim = frAdd(frAdd(frMul(ra, valA), frMul(rb, valB)), valC)

        guard proof.sumcheckRounds.count == logN else { return false }
        var currentClaim = combinedClaim
        var challenges = [Fr]()

        transcript.absorbLabel("sumcheck")
        for i in 0..<logN {
            let (s0, s1, s2) = proof.sumcheckRounds[i]
            if frToInt(frAdd(s0, s1)) != frToInt(currentClaim) { return false }

            transcript.absorb(s0)
            transcript.absorb(s1)
            transcript.absorb(s2)
            let r_i = transcript.squeeze()
            challenges.append(r_i)
            currentClaim = spartanInterpolateQuad(s0: s0, s1: s1, s2: s2, at: r_i)
        }

        // Recompute w_tilde(r) from sparse matrices
        let eqTau = MultilinearPoly.eqPoly(point: tau)
        var wVec = [Fr](repeating: Fr.zero, count: paddedN)
        for entry in instance.A {
            guard entry.row < eqTau.count && entry.col < paddedN else { continue }
            wVec[entry.col] = frAdd(wVec[entry.col], frMul(ra, frMul(entry.value, eqTau[entry.row])))
        }
        for entry in instance.B {
            guard entry.row < eqTau.count && entry.col < paddedN else { continue }
            wVec[entry.col] = frAdd(wVec[entry.col], frMul(rb, frMul(entry.value, eqTau[entry.row])))
        }
        for entry in instance.C {
            guard entry.row < eqTau.count && entry.col < paddedN else { continue }
            wVec[entry.col] = frAdd(wVec[entry.col], frMul(entry.value, eqTau[entry.row]))
        }
        let wEvalAtR = spartanEvalMultilinear(evals: wVec, point: challenges)

        if frToInt(frMul(wEvalAtR, proof.zEval)) != frToInt(currentClaim) { return false }

        return basefold.verify(root: proof.witnessCommitment, point: challenges,
                               claimedValue: proof.zEval, proof: proof.openingProof)
    }

    // MARK: - CPU Sumcheck (private)

    private func spartanCPUSumcheck(
        wEvals: [Fr], zEvals: [Fr], numVars: Int,
        claimedSum: Fr, transcript: Transcript
    ) -> (rounds: [(Fr, Fr, Fr)], challenges: [Fr]) {
        var wCur = wEvals, zCur = zEvals
        var rounds = [(Fr, Fr, Fr)]()
        var challenges = [Fr]()

        transcript.absorbLabel("sumcheck")

        for _ in 0..<numVars {
            let halfN = wCur.count / 2
            var s0 = Fr.zero, s1 = Fr.zero, s2 = Fr.zero

            for j in 0..<halfN {
                let wLow = wCur[j], wHigh = wCur[j + halfN]
                let zLow = zCur[j], zHigh = zCur[j + halfN]
                s0 = frAdd(s0, frMul(wLow, zLow))
                s1 = frAdd(s1, frMul(wHigh, zHigh))
                let w2 = frSub(frAdd(wHigh, wHigh), wLow)
                let z2 = frSub(frAdd(zHigh, zHigh), zLow)
                s2 = frAdd(s2, frMul(w2, z2))
            }

            rounds.append((s0, s1, s2))
            transcript.absorb(s0)
            transcript.absorb(s1)
            transcript.absorb(s2)
            let r_i = transcript.squeeze()
            challenges.append(r_i)

            var wNext = [Fr](repeating: Fr.zero, count: halfN)
            var zNext = [Fr](repeating: Fr.zero, count: halfN)
            for j in 0..<halfN {
                wNext[j] = frAdd(wCur[j], frMul(r_i, frSub(wCur[j + halfN], wCur[j])))
                zNext[j] = frAdd(zCur[j], frMul(r_i, frSub(zCur[j + halfN], zCur[j])))
            }
            wCur = wNext
            zCur = zNext
        }
        return (rounds, challenges)
    }
}

// MARK: - Spartan Helpers (prefixed to avoid conflicts)

func spartanEvalMultilinear(evals: [Fr], point: [Fr]) -> Fr {
    var current = evals
    for alpha in point {
        let halfN = current.count / 2
        var next = [Fr](repeating: Fr.zero, count: halfN)
        for j in 0..<halfN {
            next[j] = frAdd(current[j], frMul(alpha, frSub(current[j + halfN], current[j])))
        }
        current = next
    }
    return current[0]
}

func spartanInterpolateQuad(s0: Fr, s1: Fr, s2: Fr, at t: Fr) -> Fr {
    let one = Fr.one
    let two = frAdd(one, one)
    let inv2 = frInverse(two)
    let tMinus1 = frSub(t, one)
    let tMinus2 = frSub(t, two)
    let l0 = frMul(inv2, frMul(tMinus1, tMinus2))
    let l1 = frMul(t, frSub(two, t))
    let l2 = frMul(inv2, frMul(t, tMinus1))
    return frAdd(frAdd(frMul(s0, l0), frMul(s1, l1)), frMul(s2, l2))
}
