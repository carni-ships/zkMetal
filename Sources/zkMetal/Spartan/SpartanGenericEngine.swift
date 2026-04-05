// SpartanGenericEngine — Spartan SNARK parameterized by PCS backend.
//
// Supports:
//   - Basefold (transparent, Poseidon2 Merkle) — original SpartanEngine
//   - IPA (transparent, Pedersen commitment, no trusted setup)
//   - KZG/Zeromorph (succinct proofs, requires SRS)
//
// Protocol:
//   1. Commit z_tilde via PCS
//   2. Sumcheck #1: eq(tau,x) * [Az(x)*Bz(x) - Cz(x)] = 0
//   3. Sumcheck #2: inner product verification of combined claim
//   4. Open z_tilde at ry via PCS
//
// Uses C-accelerated sparse matvec, eq poly, fold, and sumcheck rounds
// (same hot loops as the Basefold-specific SpartanEngine).

import Foundation
import NeonFieldOps

// MARK: - Generic Proof

/// Spartan proof parameterized by PCS opening proof type.
public struct SpartanGenericProof<PCS: SpartanPCSBackend> {
    /// PCS commitment to the witness polynomial
    public let witnessCommitment: PCS.Commitment
    /// First sumcheck (degree 3): eq(tau,x)*[az(x)*bz(x) - cz(x)]
    public let sc1Rounds: [(Fr, Fr, Fr, Fr)]  // s(0), s(1), s(2), s(3)
    /// Claimed (Az)(rx), (Bz)(rx), (Cz)(rx) at the sumcheck output point
    public let azRx: Fr, bzRx: Fr, czRx: Fr
    /// Second sumcheck (degree 2): combined inner product with z
    public let sc2Rounds: [(Fr, Fr, Fr)]
    /// MLE evaluation z_tilde(ry)
    public let zEval: Fr
    /// PCS opening proof at ry
    public let openingProof: PCS.Opening

    public init(witnessCommitment: PCS.Commitment, sc1Rounds: [(Fr, Fr, Fr, Fr)],
                azRx: Fr, bzRx: Fr, czRx: Fr,
                sc2Rounds: [(Fr, Fr, Fr)], zEval: Fr, openingProof: PCS.Opening) {
        self.witnessCommitment = witnessCommitment
        self.sc1Rounds = sc1Rounds
        self.azRx = azRx; self.bzRx = bzRx; self.czRx = czRx
        self.sc2Rounds = sc2Rounds
        self.zEval = zEval
        self.openingProof = openingProof
    }
}

// MARK: - Engine

/// Generic Spartan engine that works with any multilinear PCS backend.
public class SpartanGenericEngine<PCS: SpartanPCSBackend> {
    private let pcs: PCS

    // Buffer caches for repeated prove calls (avoid re-packing)
    private var cachedPackedA: [UInt64]?
    private var cachedPackedB: [UInt64]?
    private var cachedPackedC: [UInt64]?
    private var cachedNumA: Int = 0
    private var cachedNumB: Int = 0
    private var cachedNumC: Int = 0

    public init(pcs: PCS) {
        self.pcs = pcs
    }

    // MARK: - C-accelerated helpers

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

    public func prove(instance: SpartanR1CS, publicInputs: [Fr], witness: [Fr]) throws -> SpartanGenericProof<PCS> {
        let z = SpartanR1CS.buildZ(publicInputs: publicInputs, witness: witness)
        precondition(z.count == instance.numVariables)
        precondition(instance.isSatisfied(z: z), "R1CS not satisfied")

        let logM = instance.logM, logN = instance.logN
        let paddedM = instance.paddedM, paddedN = instance.paddedN
        let zTilde = instance.buildZTilde(z: z)

        // Pack sparse entries for C interop (cache across calls)
        if cachedPackedA == nil || cachedNumA != instance.A.count {
            cachedPackedA = packEntriesGeneric(instance.A)
            cachedNumA = instance.A.count
        }
        if cachedPackedB == nil || cachedNumB != instance.B.count {
            cachedPackedB = packEntriesGeneric(instance.B)
            cachedNumB = instance.B.count
        }
        if cachedPackedC == nil || cachedNumC != instance.C.count {
            cachedPackedC = packEntriesGeneric(instance.C)
            cachedNumC = instance.C.count
        }
        let packedA = cachedPackedA!
        let packedB = cachedPackedB!
        let packedC = cachedPackedC!

        // Commit z_tilde via PCS
        let commitment = try pcs.commit(evaluations: zTilde)

        // Transcript
        let ts = Transcript(label: "spartan-generic", backend: .keccak256)
        for p in publicInputs { ts.absorb(p) }
        ts.absorb(commitment.transcriptTag)
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
        var sc1Size = paddedM

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

        let (zEval, openProof) = try pcs.open(evaluations: zTilde, point: ry)

        return SpartanGenericProof(
            witnessCommitment: commitment,
            sc1Rounds: sc1Rounds, azRx: azAtRx, bzRx: bzAtRx, czRx: czAtRx,
            sc2Rounds: sc2Rounds, zEval: zEval, openingProof: openProof)
    }

    // MARK: - Verifier

    public func verify(instance: SpartanR1CS, publicInputs: [Fr],
                       proof: SpartanGenericProof<PCS>) -> Bool {
        let logM = instance.logM, logN = instance.logN, paddedN = instance.paddedN

        let ts = Transcript(label: "spartan-generic", backend: .keccak256)
        for p in publicInputs { ts.absorb(p) }
        ts.absorb(proof.witnessCommitment.transcriptTag)
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

        // Recompute w(ry)
        let eqRx = cEqPoly(point: rx)
        let packedA = packEntriesGeneric(instance.A)
        let packedB = packEntriesGeneric(instance.B)
        let packedC = packEntriesGeneric(instance.C)
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

        // Verify PCS opening
        return pcs.verify(commitment: proof.witnessCommitment, point: ry,
                          value: proof.zEval, proof: proof.openingProof)
    }
}

// MARK: - Convenience type aliases

/// Spartan with IPA commitment (transparent, no trusted setup).
public typealias SpartanIPAEngine = SpartanGenericEngine<IPAPCSAdapter>

/// Spartan with KZG/Zeromorph commitment (succinct proofs, requires SRS).
public typealias SpartanKZGEngine = SpartanGenericEngine<KZGPCSAdapter>

/// Spartan proof types for each PCS variant.
public typealias SpartanIPAProof = SpartanGenericProof<IPAPCSAdapter>
public typealias SpartanKZGProof = SpartanGenericProof<KZGPCSAdapter>

// MARK: - Pack entries helper (module-private, avoids conflict with SpartanEngine's)

private func packEntriesGeneric(_ entries: [SpartanEntry]) -> [UInt64] {
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
