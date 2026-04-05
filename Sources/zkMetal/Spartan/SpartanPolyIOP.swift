// SpartanPolyIOP — Spartan Polynomial IOP prover and verifier
//
// Implements the full Spartan prove-verify cycle for R1CS circuits:
//   1. Encode A, B, C matrices as multilinear extensions
//   2. Sumcheck on F(x) = eq(tau, x) * (Az(x) * Bz(x) - Cz(x)) to prove sum = 0
//   3. At each round, send degree-2 univariate to verifier (degree-3 for SC1)
//   4. After sumcheck, open MLE evaluations at sumcheck point
//   5. Use PCS (Basefold / IPA / KZG) to commit and open witness polynomial
//
// This is a clean API layer over the Spartan protocol, using the existing
// C-accelerated sumcheck infrastructure from SpartanEngine / SpartanGenericEngine.

import Foundation
import NeonFieldOps

// MARK: - Proof Struct

/// A complete Spartan Polynomial IOP proof, parameterized by PCS backend.
public struct SpartanPolyIOPProof<PCS: SpartanPCSBackend> {
    /// PCS commitment to the witness polynomial z_tilde
    public let witnessCommitment: PCS.Commitment

    /// Sumcheck #1 round polynomials (degree 3): s(0), s(1), s(2), s(3)
    /// One tuple per round, logM rounds total.
    public let sumcheckRounds: [(Fr, Fr, Fr, Fr)]

    /// Claimed evaluations Az(rx), Bz(rx), Cz(rx) at sumcheck output point rx
    public let azEval: Fr
    public let bzEval: Fr
    public let czEval: Fr

    /// Sumcheck #2 round polynomials (degree 2): s(0), s(1), s(2)
    /// One tuple per round, logN rounds total.
    public let innerProductRounds: [(Fr, Fr, Fr)]

    /// MLE evaluation z_tilde(ry) at sumcheck #2 output point
    public let witnessEval: Fr

    /// PCS opening proof for z_tilde at ry
    public let openingProof: PCS.Opening

    /// Number of sumcheck rounds (= log2(numConstraints padded))
    public var numSumcheckRounds: Int { sumcheckRounds.count }

    public init(witnessCommitment: PCS.Commitment,
                sumcheckRounds: [(Fr, Fr, Fr, Fr)],
                azEval: Fr, bzEval: Fr, czEval: Fr,
                innerProductRounds: [(Fr, Fr, Fr)],
                witnessEval: Fr, openingProof: PCS.Opening) {
        self.witnessCommitment = witnessCommitment
        self.sumcheckRounds = sumcheckRounds
        self.azEval = azEval
        self.bzEval = bzEval
        self.czEval = czEval
        self.innerProductRounds = innerProductRounds
        self.witnessEval = witnessEval
        self.openingProof = openingProof
    }
}

// MARK: - Prover

/// Spartan Polynomial IOP Prover.
///
/// Proves R1CS satisfaction via Spartan protocol:
///   a. Encode A, B, C matrices as multilinear extensions
///   b. Run sumcheck on F(x) = eq(tau, x) * (Az(x) * Bz(x) - Cz(x))
///   c. At each round, send the degree-2/3 univariate to the verifier
///   d. After sumcheck, open the MLE evaluations at the sumcheck point
///   e. Use PCS to commit and open the witness polynomial
public class SpartanPolyIOPProver<PCS: SpartanPCSBackend> {
    private let pcs: PCS

    // Buffer caches for repeated prove calls
    private var cachedPackedA: [UInt64]?
    private var cachedPackedB: [UInt64]?
    private var cachedPackedC: [UInt64]?
    private var cachedNumA: Int = 0
    private var cachedNumB: Int = 0
    private var cachedNumC: Int = 0

    public init(pcs: PCS) {
        self.pcs = pcs
    }

    /// Prove that the given witness satisfies the R1CS instance.
    ///
    /// - Parameters:
    ///   - instance: The R1CS constraint system
    ///   - publicInputs: Public input values
    ///   - witness: Private witness values
    /// - Returns: A SpartanPolyIOPProof that can be verified
    public func prove(instance: SpartanR1CS, publicInputs: [Fr],
                      witness: [Fr]) throws -> SpartanPolyIOPProof<PCS> {
        let z = SpartanR1CS.buildZ(publicInputs: publicInputs, witness: witness)
        precondition(z.count == instance.numVariables, "z length mismatch")
        precondition(instance.isSatisfied(z: z), "R1CS not satisfied")

        let logM = instance.logM, logN = instance.logN
        let paddedM = instance.paddedM, paddedN = instance.paddedN

        // Step (a): Build z_tilde (padded witness as MLE evaluations on {0,1}^logN)
        let zTilde = instance.buildZTilde(z: z)

        // Pack sparse entries for C interop (cache across calls)
        cachePackedEntries(instance: instance)
        let packedA = cachedPackedA!
        let packedB = cachedPackedB!
        let packedC = cachedPackedC!

        // Step (e): Commit z_tilde via PCS
        let commitment = try pcs.commit(evaluations: zTilde)

        // Build Fiat-Shamir transcript
        let ts = Transcript(label: "spartan-polyiop", backend: .keccak256)
        for p in publicInputs { ts.absorb(p) }
        ts.absorb(commitment.transcriptTag)
        ts.absorbLabel("tau")
        let tau = ts.squeezeN(logM)

        // Step (a): Compute Az, Bz, Cz via sparse matrix-vector multiply
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

        // eq(tau, .) polynomial
        var eqTau = polyIOPEqPoly(point: tau)

        // Step (b): Sumcheck #1 — degree 3
        // F(x) = eq(tau, x) * (Az(x) * Bz(x) - Cz(x)), prove sum = 0
        ts.absorbLabel("sc1")
        var sc1Rounds = [(Fr, Fr, Fr, Fr)]()
        sc1Rounds.reserveCapacity(logM)
        var rx = [Fr]()
        rx.reserveCapacity(logM)
        var sc1Size = paddedM

        // Step (c): Each round sends degree-3 univariate s(0), s(1), s(2), s(3)
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

            // Fold all arrays in-place
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

        // Step (d): Open MLE evaluations at rx
        let azAtRx = azVec[0], bzAtRx = bzVec[0], czAtRx = czVec[0]
        ts.absorb(azAtRx); ts.absorb(bzAtRx); ts.absorb(czAtRx)

        // Sumcheck #2: verify inner product claims
        // Combined claim: rr*Az(rx) + rs*Bz(rx) + Cz(rx) = sum_y w(y)*z(y)
        ts.absorbLabel("sc2-combine")
        let rr = ts.squeeze(), rs = ts.squeeze()

        let eqRx = polyIOPEqPoly(point: rx)
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

        // SC2 sumcheck (degree 2)
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

        // Step (e): Open z_tilde at ry via PCS
        let (zEval, openProof) = try pcs.open(evaluations: zTilde, point: ry)

        return SpartanPolyIOPProof(
            witnessCommitment: commitment,
            sumcheckRounds: sc1Rounds,
            azEval: azAtRx, bzEval: bzAtRx, czEval: czAtRx,
            innerProductRounds: sc2Rounds,
            witnessEval: zEval,
            openingProof: openProof)
    }

    // MARK: - Internal helpers

    private func cachePackedEntries(instance: SpartanR1CS) {
        if cachedPackedA == nil || cachedNumA != instance.A.count {
            cachedPackedA = polyIOPPackEntries(instance.A)
            cachedNumA = instance.A.count
        }
        if cachedPackedB == nil || cachedNumB != instance.B.count {
            cachedPackedB = polyIOPPackEntries(instance.B)
            cachedNumB = instance.B.count
        }
        if cachedPackedC == nil || cachedNumC != instance.C.count {
            cachedPackedC = polyIOPPackEntries(instance.C)
            cachedNumC = instance.C.count
        }
    }
}

// MARK: - Verifier

/// Spartan Polynomial IOP Verifier.
///
/// Verifies a SpartanPolyIOPProof against an R1CS instance and public inputs.
public class SpartanPolyIOPVerifier<PCS: SpartanPCSBackend> {
    private let pcs: PCS

    public init(pcs: PCS) {
        self.pcs = pcs
    }

    /// Verify a Spartan Polynomial IOP proof.
    ///
    /// - Parameters:
    ///   - instance: The R1CS constraint system
    ///   - publicInputs: Public input values
    ///   - proof: The proof to verify
    /// - Returns: true if the proof is valid
    public func verify(instance: SpartanR1CS, publicInputs: [Fr],
                       proof: SpartanPolyIOPProof<PCS>) -> Bool {
        let logM = instance.logM, logN = instance.logN, paddedN = instance.paddedN

        // Reconstruct Fiat-Shamir transcript
        let ts = Transcript(label: "spartan-polyiop", backend: .keccak256)
        for p in publicInputs { ts.absorb(p) }
        ts.absorb(proof.witnessCommitment.transcriptTag)
        ts.absorbLabel("tau")
        let tau = ts.squeezeN(logM)

        // Verify Sumcheck #1
        guard proof.sumcheckRounds.count == logM else { return false }
        ts.absorbLabel("sc1")
        var cur = Fr.zero  // sum should be 0
        var rx = [Fr]()
        for i in 0..<logM {
            let (s0, s1, s2, s3) = proof.sumcheckRounds[i]
            // Check: s(0) + s(1) = current running claim
            if !spartanFrEqual(frAdd(s0, s1), cur) { return false }
            ts.absorb(s0); ts.absorb(s1); ts.absorb(s2); ts.absorb(s3)
            let ri = ts.squeeze()
            rx.append(ri)
            // Evaluate degree-3 polynomial at challenge ri
            cur = spartanInterpCubic(s0: s0, s1: s1, s2: s2, s3: s3, t: ri)
        }

        // Check final sumcheck #1 claim: cur = eq(tau, rx) * (az*bz - cz)
        let eqTauRx = spartanEvalEq(tau, rx)
        let expected = frMul(eqTauRx, frSub(frMul(proof.azEval, proof.bzEval), proof.czEval))
        if !spartanFrEqual(cur, expected) { return false }

        ts.absorb(proof.azEval); ts.absorb(proof.bzEval); ts.absorb(proof.czEval)

        // Verify Sumcheck #2
        ts.absorbLabel("sc2-combine")
        let rr = ts.squeeze(), rs = ts.squeeze()
        let combinedClaim = frAdd(frAdd(frMul(rr, proof.azEval), frMul(rs, proof.bzEval)), proof.czEval)

        guard proof.innerProductRounds.count == logN else { return false }
        ts.absorbLabel("sc2")
        var cur2 = combinedClaim
        var ry = [Fr]()
        for i in 0..<logN {
            let (s0, s1, s2) = proof.innerProductRounds[i]
            if !spartanFrEqual(frAdd(s0, s1), cur2) { return false }
            ts.absorb(s0); ts.absorb(s1); ts.absorb(s2)
            let ri = ts.squeeze()
            ry.append(ri)
            cur2 = spartanInterpQuadratic(s0: s0, s1: s1, s2: s2, t: ri)
        }

        // Recompute w(ry) from matrices and verify w(ry) * z(ry) = cur2
        let eqRx = polyIOPEqPoly(point: rx)
        let packedA = polyIOPPackEntries(instance.A)
        let packedB = polyIOPPackEntries(instance.B)
        let packedC = polyIOPPackEntries(instance.C)
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

        let wRy = polyIOPMleEval(evals: wVec, pt: ry)
        if !spartanFrEqual(frMul(wRy, proof.witnessEval), cur2) { return false }

        // Verify PCS opening
        return pcs.verify(commitment: proof.witnessCommitment, point: ry,
                          value: proof.witnessEval, proof: proof.openingProof)
    }
}

// MARK: - Module-private helpers (prefixed to avoid conflicts)

/// C-accelerated eq polynomial computation.
func polyIOPEqPoly(point: [Fr]) -> [Fr] {
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
func polyIOPMleEval(evals: [Fr], pt: [Fr]) -> Fr {
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

/// Pack SpartanEntry arrays into C-compatible format.
func polyIOPPackEntries(_ entries: [SpartanEntry]) -> [UInt64] {
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
