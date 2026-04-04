// Spartan Engine — Transparent SNARK (no trusted setup)
// Proves R1CS: Az . Bz = Cz via sumcheck over multilinear extensions.
//
// Protocol:
//   1. Commit z_tilde via Basefold (Poseidon2 hash — no SRS needed)
//   2. Verifier sends tau (via Fiat-Shamir)
//   3. Sumcheck #1: sum_x eq(tau,x)*[(Az)(x)*(Bz)(x) - (Cz)(x)] = 0
//      Degree 3 in each variable. Round poly needs 4 evaluations: s(0),s(1),s(2),s(3).
//   4. Prover claims (Az)(rx), (Bz)(rx), (Cz)(rx) at the sumcheck output point rx.
//   5. Sumcheck #2: verify inner product claims reduce to a single z evaluation.
//      Verifier combines: rr*Az(rx) + rs*Bz(rx) + Cz(rx) = sum_y [rr*A(rx,y)+rs*B(rx,y)+C(rx,y)]*z(y)
//   6. Open z_tilde via Basefold at the output point ry.
//
// Transparency: uses Poseidon2 Merkle commitments — no trusted setup.

import Foundation

// MARK: - Proof

public struct SpartanProof {
    public let witnessCommitment: Fr
    // First sumcheck (degree 3): eq(tau,x)*[az(x)*bz(x) - cz(x)]
    public let sc1Rounds: [(Fr, Fr, Fr, Fr)]  // degree-3 round polys: s(0), s(1), s(2), s(3)
    public let azRx: Fr, bzRx: Fr, czRx: Fr   // claimed (Az)(rx), (Bz)(rx), (Cz)(rx)
    // Second sumcheck (degree 2): combined inner product with z
    public let sc2Rounds: [(Fr, Fr, Fr)]
    public let zEval: Fr
    public let openingProof: BasefoldProof
}

// MARK: - Engine

public class SpartanEngine {
    public static let version = Versions.spartan
    private let basefold: BasefoldEngine

    public init() throws {
        self.basefold = try BasefoldEngine()
    }

    // MARK: - Prover

    public func prove(instance: SpartanR1CS, publicInputs: [Fr], witness: [Fr]) throws -> SpartanProof {
        let z = SpartanR1CS.buildZ(publicInputs: publicInputs, witness: witness)
        precondition(z.count == instance.numVariables)
        precondition(instance.isSatisfied(z: z), "R1CS not satisfied")

        let logM = instance.logM, logN = instance.logN
        let paddedM = instance.paddedM, paddedN = instance.paddedN
        let zTilde = instance.buildZTilde(z: z)

        // Commit z_tilde via Basefold (Poseidon2 Merkle — transparent)
        let commitment = try basefold.commit(evaluations: zTilde)

        let ts = Transcript(label: "spartan-r1cs")
        // Bind to public inputs and commitment
        for p in publicInputs { ts.absorb(p) }
        ts.absorb(commitment.root)
        ts.absorbLabel("tau")
        let tau = ts.squeezeN(logM)

        // Compute (Az), (Bz), (Cz) as vectors of length paddedM
        var azVec = [Fr](repeating: Fr.zero, count: paddedM)
        var bzVec = [Fr](repeating: Fr.zero, count: paddedM)
        var czVec = [Fr](repeating: Fr.zero, count: paddedM)
        for e in instance.A {
            guard e.row < paddedM && e.col < z.count else { continue }
            azVec[e.row] = frAdd(azVec[e.row], frMul(e.value, z[e.col]))
        }
        for e in instance.B {
            guard e.row < paddedM && e.col < z.count else { continue }
            bzVec[e.row] = frAdd(bzVec[e.row], frMul(e.value, z[e.col]))
        }
        for e in instance.C {
            guard e.row < paddedM && e.col < z.count else { continue }
            czVec[e.row] = frAdd(czVec[e.row], frMul(e.value, z[e.col]))
        }

        // eq(tau, .) over {0,1}^logM
        let eqTau = MultilinearPoly.eqPoly(point: tau)

        // ---- Sumcheck #1 ----
        // sum_x eq(tau,x)*[az(x)*bz(x) - cz(x)] = 0
        // Degree 3 per variable: eq is linear, az*bz is degree 2, total degree 3.
        ts.absorbLabel("sc1")
        var eqC = eqTau, azC = azVec, bzC = bzVec, czC = czVec
        var sc1Rounds = [(Fr, Fr, Fr, Fr)]()
        var rx = [Fr]()

        for _ in 0..<logM {
            let h = eqC.count / 2
            var s0 = Fr.zero, s1 = Fr.zero, s2 = Fr.zero, s3 = Fr.zero
            for j in 0..<h {
                let eL = eqC[j], eH = eqC[j + h]
                let aL = azC[j], aH = azC[j + h]
                let bL = bzC[j], bH = bzC[j + h]
                let cL = czC[j], cH = czC[j + h]
                // f(t) = eq(t)*[az(t)*bz(t) - cz(t)]
                // t=0: eL*(aL*bL - cL)
                s0 = frAdd(s0, frMul(eL, frSub(frMul(aL, bL), cL)))
                // t=1: eH*(aH*bH - cH)
                s1 = frAdd(s1, frMul(eH, frSub(frMul(aH, bH), cH)))
                // t=2: linear extrapolation at 2
                let e2 = frSub(frAdd(eH, eH), eL)
                let a2 = frSub(frAdd(aH, aH), aL)
                let b2 = frSub(frAdd(bH, bH), bL)
                let c2 = frSub(frAdd(cH, cH), cL)
                s2 = frAdd(s2, frMul(e2, frSub(frMul(a2, b2), c2)))
                // t=3: linear extrapolation at 3
                let three = frFromInt(3), two = frFromInt(2)
                let e3 = frSub(frMul(three, eH), frMul(two, eL))
                let a3 = frSub(frMul(three, aH), frMul(two, aL))
                let b3 = frSub(frMul(three, bH), frMul(two, bL))
                let c3 = frSub(frMul(three, cH), frMul(two, cL))
                s3 = frAdd(s3, frMul(e3, frSub(frMul(a3, b3), c3)))
            }
            sc1Rounds.append((s0, s1, s2, s3))
            ts.absorb(s0); ts.absorb(s1); ts.absorb(s2); ts.absorb(s3)
            let ri = ts.squeeze()
            rx.append(ri)

            // Reduce: fix variable to ri
            var eN = [Fr](repeating: Fr.zero, count: h)
            var aN = [Fr](repeating: Fr.zero, count: h)
            var bN = [Fr](repeating: Fr.zero, count: h)
            var cN = [Fr](repeating: Fr.zero, count: h)
            for j in 0..<h {
                eN[j] = frAdd(eqC[j], frMul(ri, frSub(eqC[j + h], eqC[j])))
                aN[j] = frAdd(azC[j], frMul(ri, frSub(azC[j + h], azC[j])))
                bN[j] = frAdd(bzC[j], frMul(ri, frSub(bzC[j + h], bzC[j])))
                cN[j] = frAdd(czC[j], frMul(ri, frSub(czC[j + h], czC[j])))
            }
            eqC = eN; azC = aN; bzC = bN; czC = cN
        }

        // After SC1: prover claims (Az)(rx), (Bz)(rx), (Cz)(rx)
        let azAtRx = azC[0], bzAtRx = bzC[0], czAtRx = czC[0]
        ts.absorb(azAtRx); ts.absorb(bzAtRx); ts.absorb(czAtRx)

        // ---- Sumcheck #2 ----
        // Verify inner product: (Az)(rx) = sum_y A_tilde(rx,y)*z(y), etc.
        // Combine: rr*Az(rx) + rs*Bz(rx) + Cz(rx) = sum_y [rr*A(rx,y)+rs*B(rx,y)+C(rx,y)]*z(y)
        ts.absorbLabel("sc2-combine")
        let rr = ts.squeeze(), rs = ts.squeeze()

        // Build combined weight: w[j] = rr*A_rx[j] + rs*B_rx[j] + C_rx[j]
        // where A_rx[j] = sum_i A[i,j]*eq(rx, bin(i))
        let eqRx = MultilinearPoly.eqPoly(point: rx)
        var wVec = [Fr](repeating: Fr.zero, count: paddedN)
        for e in instance.A {
            guard e.row < eqRx.count && e.col < paddedN else { continue }
            wVec[e.col] = frAdd(wVec[e.col], frMul(rr, frMul(e.value, eqRx[e.row])))
        }
        for e in instance.B {
            guard e.row < eqRx.count && e.col < paddedN else { continue }
            wVec[e.col] = frAdd(wVec[e.col], frMul(rs, frMul(e.value, eqRx[e.row])))
        }
        for e in instance.C {
            guard e.row < eqRx.count && e.col < paddedN else { continue }
            wVec[e.col] = frAdd(wVec[e.col], frMul(e.value, eqRx[e.row]))
        }

        // Sumcheck on f(y) = w(y)*z(y), degree 2
        ts.absorbLabel("sc2")
        var wC = wVec, zC2 = zTilde
        var sc2Rounds = [(Fr, Fr, Fr)]()
        var ry = [Fr]()
        for _ in 0..<logN {
            let h = wC.count / 2
            var s0 = Fr.zero, s1 = Fr.zero, s2 = Fr.zero
            for j in 0..<h {
                s0 = frAdd(s0, frMul(wC[j], zC2[j]))
                s1 = frAdd(s1, frMul(wC[j + h], zC2[j + h]))
                let w2 = frSub(frAdd(wC[j + h], wC[j + h]), wC[j])
                let z2 = frSub(frAdd(zC2[j + h], zC2[j + h]), zC2[j])
                s2 = frAdd(s2, frMul(w2, z2))
            }
            sc2Rounds.append((s0, s1, s2))
            ts.absorb(s0); ts.absorb(s1); ts.absorb(s2)
            let ri = ts.squeeze()
            ry.append(ri)
            var wN = [Fr](repeating: Fr.zero, count: h)
            var zN = [Fr](repeating: Fr.zero, count: h)
            for j in 0..<h {
                wN[j] = frAdd(wC[j], frMul(ri, frSub(wC[j + h], wC[j])))
                zN[j] = frAdd(zC2[j], frMul(ri, frSub(zC2[j + h], zC2[j])))
            }
            wC = wN; zC2 = zN
        }

        let zEval = spartanEvalML(evals: zTilde, pt: ry)
        let op = try basefold.open(commitment: commitment, point: ry)

        return SpartanProof(witnessCommitment: commitment.root,
                            sc1Rounds: sc1Rounds, azRx: azAtRx, bzRx: bzAtRx, czRx: czAtRx,
                            sc2Rounds: sc2Rounds, zEval: zEval, openingProof: op)
    }

    // MARK: - Verifier

    public func verify(instance: SpartanR1CS, publicInputs: [Fr], proof: SpartanProof) -> Bool {
        let logM = instance.logM, logN = instance.logN, paddedN = instance.paddedN

        let ts = Transcript(label: "spartan-r1cs")
        // Bind to public inputs and commitment
        for p in publicInputs { ts.absorb(p) }
        ts.absorb(proof.witnessCommitment)
        ts.absorbLabel("tau")
        let tau = ts.squeezeN(logM)

        // ---- Verify SC1 ----
        guard proof.sc1Rounds.count == logM else { return false }
        ts.absorbLabel("sc1")
        var cur = Fr.zero  // claimed sum = 0 (R1CS is satisfied)
        var rx = [Fr]()
        for i in 0..<logM {
            let (s0, s1, s2, s3) = proof.sc1Rounds[i]
            // Check: s(0) + s(1) = current claim
            if !spartanFrEqual(frAdd(s0, s1), cur) { return false }
            ts.absorb(s0); ts.absorb(s1); ts.absorb(s2); ts.absorb(s3)
            let ri = ts.squeeze()
            rx.append(ri)
            cur = spartanInterpCubic(s0: s0, s1: s1, s2: s2, s3: s3, t: ri)
        }

        // After SC1: cur should equal eq(tau,rx)*[azRx*bzRx - czRx]
        let eqTauRx = spartanEvalEq(tau, rx)
        let expected = frMul(eqTauRx, frSub(frMul(proof.azRx, proof.bzRx), proof.czRx))
        if !spartanFrEqual(cur, expected) { return false }

        ts.absorb(proof.azRx); ts.absorb(proof.bzRx); ts.absorb(proof.czRx)

        // ---- Verify SC2 ----
        ts.absorbLabel("sc2-combine")
        let rr = ts.squeeze(), rs = ts.squeeze()
        let combinedClaim = frAdd(frAdd(frMul(rr, proof.azRx), frMul(rs, proof.bzRx)), proof.czRx)

        guard proof.sc2Rounds.count == logN else { return false }
        ts.absorbLabel("sc2")
        var cur2 = combinedClaim
        var ry = [Fr]()
        for i in 0..<logN {
            let (s0, s1, s2) = proof.sc2Rounds[i]
            if !spartanFrEqual(frAdd(s0, s1), cur2) { return false }
            ts.absorb(s0); ts.absorb(s1); ts.absorb(s2)
            let ri = ts.squeeze()
            ry.append(ri)
            cur2 = spartanInterpQuadratic(s0: s0, s1: s1, s2: s2, t: ri)
        }

        // After SC2: cur2 = w(ry)*z(ry)
        // Recompute w(ry) from the instance matrices
        let eqRx = MultilinearPoly.eqPoly(point: rx)
        var wVec = [Fr](repeating: Fr.zero, count: paddedN)
        for e in instance.A {
            guard e.row < eqRx.count && e.col < paddedN else { continue }
            wVec[e.col] = frAdd(wVec[e.col], frMul(rr, frMul(e.value, eqRx[e.row])))
        }
        for e in instance.B {
            guard e.row < eqRx.count && e.col < paddedN else { continue }
            wVec[e.col] = frAdd(wVec[e.col], frMul(rs, frMul(e.value, eqRx[e.row])))
        }
        for e in instance.C {
            guard e.row < eqRx.count && e.col < paddedN else { continue }
            wVec[e.col] = frAdd(wVec[e.col], frMul(e.value, eqRx[e.row]))
        }
        let wRy = spartanEvalML(evals: wVec, pt: ry)
        if !spartanFrEqual(frMul(wRy, proof.zEval), cur2) { return false }

        // Verify Basefold opening
        return basefold.verify(root: proof.witnessCommitment, point: ry,
                               claimedValue: proof.zEval, proof: proof.openingProof)
    }
}

// MARK: - Spartan Helpers (prefixed to avoid conflicts)

/// Evaluate multilinear extension at a point via successive halving.
func spartanEvalML(evals: [Fr], pt: [Fr]) -> Fr {
    var c = evals
    for a in pt {
        let h = c.count / 2
        var n = [Fr](repeating: Fr.zero, count: h)
        for j in 0..<h {
            n[j] = frAdd(c[j], frMul(a, frSub(c[j + h], c[j])))
        }
        c = n
    }
    return c[0]
}

/// Evaluate eq(a,b) = prod_i (a_i*b_i + (1-a_i)*(1-b_i))
func spartanEvalEq(_ a: [Fr], _ b: [Fr]) -> Fr {
    precondition(a.count == b.count)
    var result = Fr.one
    for i in 0..<a.count {
        let ai = a[i], bi = b[i]
        let term = frAdd(frMul(ai, bi), frMul(frSub(Fr.one, ai), frSub(Fr.one, bi)))
        result = frMul(result, term)
    }
    return result
}

/// Fr equality check
func spartanFrEqual(_ a: Fr, _ b: Fr) -> Bool {
    let diff = frSub(a, b)
    let limbs = frToInt(diff)
    return limbs[0] == 0 && limbs[1] == 0 && limbs[2] == 0 && limbs[3] == 0
}

/// Interpolate degree-2 polynomial through (0,s0),(1,s1),(2,s2) at t.
func spartanInterpQuadratic(s0: Fr, s1: Fr, s2: Fr, t: Fr) -> Fr {
    let one = Fr.one, two = frAdd(one, one), inv2 = frInverse(two)
    let tm1 = frSub(t, one), tm2 = frSub(t, two)
    let l0 = frMul(inv2, frMul(tm1, tm2))
    let l1 = frMul(t, frSub(two, t))
    let l2 = frMul(inv2, frMul(t, tm1))
    return frAdd(frAdd(frMul(s0, l0), frMul(s1, l1)), frMul(s2, l2))
}

/// Interpolate degree-3 polynomial through (0,s0),(1,s1),(2,s2),(3,s3) at t.
func spartanInterpCubic(s0: Fr, s1: Fr, s2: Fr, s3: Fr, t: Fr) -> Fr {
    let one = Fr.one, two = frAdd(one, one), three = frAdd(two, one)
    let six = frMul(two, three)
    let inv6 = frInverse(six), inv2 = frInverse(two)
    let tm1 = frSub(t, one), tm2 = frSub(t, two), tm3 = frSub(t, three)
    // L0 = (t-1)(t-2)(t-3)/(-6)
    let l0 = frMul(frMul(frSub(Fr.zero, inv6), frMul(tm1, tm2)), tm3)
    // L1 = t(t-2)(t-3)/2
    let l1 = frMul(frMul(inv2, frMul(t, tm2)), tm3)
    // L2 = t(t-1)(t-3)/(-2)
    let l2 = frMul(frMul(frSub(Fr.zero, inv2), frMul(t, tm1)), tm3)
    // L3 = t(t-1)(t-2)/6
    let l3 = frMul(frMul(inv6, frMul(t, tm1)), tm2)
    return frAdd(frAdd(frMul(s0, l0), frMul(s1, l1)), frAdd(frMul(s2, l2), frMul(s3, l3)))
}
