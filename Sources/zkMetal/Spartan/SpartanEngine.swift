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
//
// Optimizations:
//   - C CIOS Montgomery field arithmetic for all sumcheck inner loops
//   - Keccak-256 transcript (faster absorb than Poseidon2)
//   - C-accelerated eq polynomial, MLE evaluation, sparse matvec
//   - Buffer caching for repeated prove calls

import Foundation
import NeonFieldOps

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

// MARK: - Packed sparse entries for C interop

/// Pack SpartanEntry arrays into C-compatible format: 5 uint64 per entry.
/// Layout: [low32=row | high32=col, value_limb0, value_limb1, value_limb2, value_limb3]
private func packEntries(_ entries: [SpartanEntry]) -> [UInt64] {
    var packed = [UInt64](repeating: 0, count: entries.count * 5)
    for (i, e) in entries.enumerated() {
        let base = i * 5
        packed[base] = UInt64(e.row) | (UInt64(e.col) << 32)
        withUnsafeBytes(of: e.value) { src in
            for j in 0..<4 {
                packed[base + 1 + j] = src.load(fromByteOffset: j * 8, as: UInt64.self)
            }
        }
    }
    return packed
}

// MARK: - Engine

public class SpartanEngine {
    public static let version = Versions.spartan
    private let basefold: BasefoldEngine

    // Buffer caches for repeated prove calls (avoid re-packing)
    private var cachedPackedA: [UInt64]?
    private var cachedPackedB: [UInt64]?
    private var cachedPackedC: [UInt64]?
    private var cachedNumA: Int = 0
    private var cachedNumB: Int = 0
    private var cachedNumC: Int = 0

    public init() throws {
        self.basefold = try BasefoldEngine()
    }

    // MARK: - C-accelerated helpers

    /// C-accelerated eq polynomial computation.
    private func cEqPoly(point: [Fr]) -> [Fr] {
        let n = point.count
        let size = 1 << n
        var eq = [Fr](repeating: Fr.zero, count: size)
        point.withUnsafeBytes { ptBuf in
            eq.withUnsafeMutableBytes { eqBuf in
                spartan_eq_poly(
                    ptBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(n),
                    eqBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                )
            }
        }
        return eq
    }

    /// C-accelerated MLE evaluation.
    private func cMleEval(evals: [Fr], pt: [Fr]) -> Fr {
        let numVars = pt.count
        var result = Fr.zero
        evals.withUnsafeBytes { evalBuf in
            pt.withUnsafeBytes { ptBuf in
                withUnsafeMutableBytes(of: &result) { resBuf in
                    spartan_mle_eval(
                        evalBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(numVars),
                        ptBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        resBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                    )
                }
            }
        }
        return result
    }

    // MARK: - Prover

    public func prove(instance: SpartanR1CS, publicInputs: [Fr], witness: [Fr]) throws -> SpartanProof {
        let z = SpartanR1CS.buildZ(publicInputs: publicInputs, witness: witness)
        precondition(z.count == instance.numVariables)
        precondition(instance.isSatisfied(z: z), "R1CS not satisfied")

        let logM = instance.logM, logN = instance.logN
        let paddedM = instance.paddedM, paddedN = instance.paddedN
        let zTilde = instance.buildZTilde(z: z)

        // Pack sparse entries for C interop (cache across calls)
        if cachedPackedA == nil || cachedNumA != instance.A.count {
            cachedPackedA = packEntries(instance.A)
            cachedNumA = instance.A.count
        }
        if cachedPackedB == nil || cachedNumB != instance.B.count {
            cachedPackedB = packEntries(instance.B)
            cachedNumB = instance.B.count
        }
        if cachedPackedC == nil || cachedNumC != instance.C.count {
            cachedPackedC = packEntries(instance.C)
            cachedNumC = instance.C.count
        }
        let packedA = cachedPackedA!
        let packedB = cachedPackedB!
        let packedC = cachedPackedC!

        // Commit z_tilde via Basefold (Poseidon2 Merkle -- transparent)
        let commitment = try basefold.commit(evaluations: zTilde)

        // Keccak transcript (faster absorb)
        let ts = Transcript(label: "spartan-r1cs", backend: .keccak256)
        for p in publicInputs { ts.absorb(p) }
        ts.absorb(commitment.root)
        ts.absorbLabel("tau")
        let tau = ts.squeezeN(logM)

        // Compute (Az), (Bz), (Cz) via C-accelerated sparse matvec
        var azVec = [Fr](repeating: Fr.zero, count: paddedM)
        var bzVec = [Fr](repeating: Fr.zero, count: paddedM)
        var czVec = [Fr](repeating: Fr.zero, count: paddedM)

        z.withUnsafeBytes { zBuf in
            let zPtr = zBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
            let zLen = Int32(z.count)
            packedA.withUnsafeBufferPointer { aBuf in
                azVec.withUnsafeMutableBytes { azBuf in
                    spartan_sparse_matvec(aBuf.baseAddress!, Int32(instance.A.count),
                                          zPtr, zLen,
                                          azBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                          Int32(paddedM))
                }
            }
            packedB.withUnsafeBufferPointer { bBuf in
                bzVec.withUnsafeMutableBytes { bzBuf in
                    spartan_sparse_matvec(bBuf.baseAddress!, Int32(instance.B.count),
                                          zPtr, zLen,
                                          bzBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                          Int32(paddedM))
                }
            }
            packedC.withUnsafeBufferPointer { cBuf in
                czVec.withUnsafeMutableBytes { czBuf in
                    spartan_sparse_matvec(cBuf.baseAddress!, Int32(instance.C.count),
                                          zPtr, zLen,
                                          czBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                          Int32(paddedM))
                }
            }
        }

        // eq(tau, .) via C-accelerated eq poly
        var eqTau = cEqPoly(point: tau)

        // ---- Sumcheck #1 (C-accelerated, degree 3) ----
        ts.absorbLabel("sc1")
        var sc1Rounds = [(Fr, Fr, Fr, Fr)]()
        sc1Rounds.reserveCapacity(logM)
        var rx = [Fr]()
        rx.reserveCapacity(logM)
        var sc1Size = paddedM  // logical size of working arrays

        for _ in 0..<logM {
            let h = sc1Size / 2
            var s0 = Fr.zero, s1 = Fr.zero, s2 = Fr.zero, s3 = Fr.zero

            eqTau.withUnsafeMutableBytes { eqBuf in
                azVec.withUnsafeMutableBytes { azBuf in
                    bzVec.withUnsafeMutableBytes { bzBuf in
                        czVec.withUnsafeMutableBytes { czBuf in
                            withUnsafeMutableBytes(of: &s0) { s0p in
                                withUnsafeMutableBytes(of: &s1) { s1p in
                                    withUnsafeMutableBytes(of: &s2) { s2p in
                                        withUnsafeMutableBytes(of: &s3) { s3p in
                                            spartan_sc1_round(
                                                eqBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                                azBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                                bzBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                                czBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                                Int32(h),
                                                s0p.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                                s1p.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                                s2p.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                                s3p.baseAddress!.assumingMemoryBound(to: UInt64.self)
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            sc1Rounds.append((s0, s1, s2, s3))
            ts.absorb(s0); ts.absorb(s1); ts.absorb(s2); ts.absorb(s3)
            let ri = ts.squeeze()
            rx.append(ri)

            // Fold all 4 arrays in-place via C (no copy needed, just shrink logical size)
            withUnsafeBytes(of: ri) { riBuf in
                let riPtr = riBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                eqTau.withUnsafeMutableBytes { buf in
                    spartan_fold_array(buf.baseAddress!.assumingMemoryBound(to: UInt64.self), Int32(h), riPtr)
                }
                azVec.withUnsafeMutableBytes { buf in
                    spartan_fold_array(buf.baseAddress!.assumingMemoryBound(to: UInt64.self), Int32(h), riPtr)
                }
                bzVec.withUnsafeMutableBytes { buf in
                    spartan_fold_array(buf.baseAddress!.assumingMemoryBound(to: UInt64.self), Int32(h), riPtr)
                }
                czVec.withUnsafeMutableBytes { buf in
                    spartan_fold_array(buf.baseAddress!.assumingMemoryBound(to: UInt64.self), Int32(h), riPtr)
                }
            }
            sc1Size = h
        }

        let azAtRx = azVec[0], bzAtRx = bzVec[0], czAtRx = czVec[0]
        ts.absorb(azAtRx); ts.absorb(bzAtRx); ts.absorb(czAtRx)

        // ---- Sumcheck #2 (C-accelerated, degree 2) ----
        ts.absorbLabel("sc2-combine")
        let rr = ts.squeeze(), rs = ts.squeeze()

        // Build combined weight via C
        let eqRx = cEqPoly(point: rx)
        var wVec = [Fr](repeating: Fr.zero, count: paddedN)

        eqRx.withUnsafeBytes { eqRxBuf in
            let eqRxPtr = eqRxBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
            let eqRxLen = Int32(eqRx.count)
            withUnsafeBytes(of: rr) { rrBuf in
                packedA.withUnsafeBufferPointer { aBuf in
                    wVec.withUnsafeMutableBytes { wBuf in
                        spartan_build_weight_vec(
                            aBuf.baseAddress!, Int32(instance.A.count),
                            eqRxPtr, eqRxLen,
                            rrBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            wBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(paddedN))
                    }
                }
            }
            withUnsafeBytes(of: rs) { rsBuf in
                packedB.withUnsafeBufferPointer { bBuf in
                    wVec.withUnsafeMutableBytes { wBuf in
                        spartan_build_weight_vec(
                            bBuf.baseAddress!, Int32(instance.B.count),
                            eqRxPtr, eqRxLen,
                            rsBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            wBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(paddedN))
                    }
                }
            }
            var one = Fr.one
            withUnsafeBytes(of: &one) { oneBuf in
                packedC.withUnsafeBufferPointer { cBuf in
                    wVec.withUnsafeMutableBytes { wBuf in
                        spartan_build_weight_vec(
                            cBuf.baseAddress!, Int32(instance.C.count),
                            eqRxPtr, eqRxLen,
                            oneBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            wBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(paddedN))
                    }
                }
            }
        }

        // SC2 sumcheck
        ts.absorbLabel("sc2")
        var zC2 = zTilde
        var sc2Rounds = [(Fr, Fr, Fr)]()
        sc2Rounds.reserveCapacity(logN)
        var ry = [Fr]()
        ry.reserveCapacity(logN)
        var sc2Size = paddedN

        for _ in 0..<logN {
            let h = sc2Size / 2
            var s0 = Fr.zero, s1 = Fr.zero, s2 = Fr.zero

            wVec.withUnsafeMutableBytes { wBuf in
                zC2.withUnsafeMutableBytes { zBuf in
                    withUnsafeMutableBytes(of: &s0) { s0p in
                        withUnsafeMutableBytes(of: &s1) { s1p in
                            withUnsafeMutableBytes(of: &s2) { s2p in
                                spartan_sc2_round(
                                    wBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                    zBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                    Int32(h),
                                    s0p.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                    s1p.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                    s2p.baseAddress!.assumingMemoryBound(to: UInt64.self)
                                )
                            }
                        }
                    }
                }
            }

            sc2Rounds.append((s0, s1, s2))
            ts.absorb(s0); ts.absorb(s1); ts.absorb(s2)
            let ri = ts.squeeze()
            ry.append(ri)

            // Fold both arrays in-place
            withUnsafeBytes(of: ri) { riBuf in
                let riPtr = riBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                wVec.withUnsafeMutableBytes { buf in
                    spartan_fold_array(buf.baseAddress!.assumingMemoryBound(to: UInt64.self), Int32(h), riPtr)
                }
                zC2.withUnsafeMutableBytes { buf in
                    spartan_fold_array(buf.baseAddress!.assumingMemoryBound(to: UInt64.self), Int32(h), riPtr)
                }
            }
            sc2Size = h
        }

        let zEval = cMleEval(evals: zTilde, pt: ry)
        let op = try basefold.open(commitment: commitment, point: ry)

        return SpartanProof(witnessCommitment: commitment.root,
                            sc1Rounds: sc1Rounds, azRx: azAtRx, bzRx: bzAtRx, czRx: czAtRx,
                            sc2Rounds: sc2Rounds, zEval: zEval, openingProof: op)
    }

    // MARK: - Verifier

    public func verify(instance: SpartanR1CS, publicInputs: [Fr], proof: SpartanProof) -> Bool {
        let logM = instance.logM, logN = instance.logN, paddedN = instance.paddedN

        // Must match prover Keccak transcript
        let ts = Transcript(label: "spartan-r1cs", backend: .keccak256)
        for p in publicInputs { ts.absorb(p) }
        ts.absorb(proof.witnessCommitment)
        ts.absorbLabel("tau")
        let tau = ts.squeezeN(logM)

        // ---- Verify SC1 ----
        guard proof.sc1Rounds.count == logM else { return false }
        ts.absorbLabel("sc1")
        var cur = Fr.zero
        var rx = [Fr]()
        for i in 0..<logM {
            let (s0, s1, s2, s3) = proof.sc1Rounds[i]
            if !spartanFrEqual(frAdd(s0, s1), cur) { return false }
            ts.absorb(s0); ts.absorb(s1); ts.absorb(s2); ts.absorb(s3)
            let ri = ts.squeeze()
            rx.append(ri)
            cur = spartanInterpCubic(s0: s0, s1: s1, s2: s2, s3: s3, t: ri)
        }

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

        // Recompute w(ry) using C-accelerated helpers
        let eqRx = cEqPoly(point: rx)
        let packedA = packEntries(instance.A)
        let packedB = packEntries(instance.B)
        let packedC = packEntries(instance.C)
        var wVec = [Fr](repeating: Fr.zero, count: paddedN)

        eqRx.withUnsafeBytes { eqRxBuf in
            let eqRxPtr = eqRxBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
            let eqRxLen = Int32(eqRx.count)
            withUnsafeBytes(of: rr) { rrBuf in
                packedA.withUnsafeBufferPointer { aBuf in
                    wVec.withUnsafeMutableBytes { wBuf in
                        spartan_build_weight_vec(
                            aBuf.baseAddress!, Int32(instance.A.count),
                            eqRxPtr, eqRxLen,
                            rrBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            wBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(paddedN))
                    }
                }
            }
            withUnsafeBytes(of: rs) { rsBuf in
                packedB.withUnsafeBufferPointer { bBuf in
                    wVec.withUnsafeMutableBytes { wBuf in
                        spartan_build_weight_vec(
                            bBuf.baseAddress!, Int32(instance.B.count),
                            eqRxPtr, eqRxLen,
                            rsBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            wBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(paddedN))
                    }
                }
            }
            var one = Fr.one
            withUnsafeBytes(of: &one) { oneBuf in
                packedC.withUnsafeBufferPointer { cBuf in
                    wVec.withUnsafeMutableBytes { wBuf in
                        spartan_build_weight_vec(
                            cBuf.baseAddress!, Int32(instance.C.count),
                            eqRxPtr, eqRxLen,
                            oneBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            wBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(paddedN))
                    }
                }
            }
        }

        let wRy = cMleEval(evals: wVec, pt: ry)
        if !spartanFrEqual(frMul(wRy, proof.zEval), cur2) { return false }

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
    let l0 = frMul(frMul(frSub(Fr.zero, inv6), frMul(tm1, tm2)), tm3)
    let l1 = frMul(frMul(inv2, frMul(t, tm2)), tm3)
    let l2 = frMul(frMul(frSub(Fr.zero, inv2), frMul(t, tm1)), tm3)
    let l3 = frMul(frMul(inv6, frMul(t, tm1)), tm2)
    return frAdd(frAdd(frMul(s0, l0), frMul(s1, l1)), frAdd(frMul(s2, l2), frMul(s3, l3)))
}
