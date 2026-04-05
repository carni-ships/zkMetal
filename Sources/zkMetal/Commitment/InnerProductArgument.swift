// Inner Product Argument (IPA) — Prover and Verifier
//
// Implements the Bulletproofs-style inner product argument protocol:
//   Given a Pedersen commitment C to vector v, prove that <v, u> = t
//   for a public vector u and claimed inner product t.
//
// Protocol (log(n) rounds):
//   1. Prover splits vectors in half, computes cross-term commitments L_i, R_i
//   2. Verifier sends random challenge x_i (Fiat-Shamir in non-interactive mode)
//   3. Both sides fold vectors and generators using x_i
//   4. After log(n) rounds, prover outputs final scalar a_final
//   5. Verifier checks: C_folded == a_final * G_final + (a_final * b_final) * Q
//
// References:
//   - Bulletproofs (Bunz et al. 2018)
//   - Halo (Bowe et al. 2019)

import Foundation
import NeonFieldOps

// MARK: - IPA Proof Structure

/// A proof for the inner product argument.
/// Contains log(n) round commitments (L_i, R_i) and the final scalar.
public struct IPAProofData {
    /// Left commitments per round (log(n) entries).
    public let Ls: [PointProjective]
    /// Right commitments per round (log(n) entries).
    public let Rs: [PointProjective]
    /// Final scalar after all folding rounds.
    public let finalA: Fr

    public init(Ls: [PointProjective], Rs: [PointProjective], finalA: Fr) {
        self.Ls = Ls
        self.Rs = Rs
        self.finalA = finalA
    }
}

// MARK: - IPA Transcript (Fiat-Shamir)

/// Simple transcript for Fiat-Shamir challenge derivation.
/// Appends raw bytes and derives challenges via Blake3 hash.
private struct IPATranscript {
    var data = [UInt8]()

    mutating func appendPoint(_ p: PointProjective) {
        withUnsafeBytes(of: p) { buf in
            let ptr = buf.baseAddress!.assumingMemoryBound(to: UInt8.self)
            data.append(contentsOf: UnsafeBufferPointer(start: ptr, count: 96))
        }
    }

    mutating func appendScalar(_ v: Fr) {
        withUnsafeBytes(of: v) { buf in
            let ptr = buf.baseAddress!.assumingMemoryBound(to: UInt8.self)
            data.append(contentsOf: UnsafeBufferPointer(start: ptr, count: 32))
        }
    }

    /// Derive a challenge Fr element from the current transcript state.
    func deriveChallenge() -> Fr {
        var hash = [UInt8](repeating: 0, count: 32)
        data.withUnsafeBufferPointer { inp in
            hash.withUnsafeMutableBufferPointer { out in
                blake3_hash_neon(inp.baseAddress!, inp.count, out.baseAddress!)
            }
        }
        var limbs = [UInt64](repeating: 0, count: 4)
        hash.withUnsafeBytes { buf in
            let ptr = buf.baseAddress!.assumingMemoryBound(to: UInt64.self)
            limbs[0] = ptr[0]
            limbs[1] = ptr[1]
            limbs[2] = ptr[2]
            limbs[3] = ptr[3]
        }
        let raw = Fr.from64(limbs)
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))  // to Montgomery form
    }
}

// MARK: - IPA Prover

/// Prover for the inner product argument.
///
/// Given:
///   - A Pedersen commitment C to vector v (with generators G[] and binding point Q)
///   - A public vector u such that <v, u> = t
///
/// Produces an IPAProofData proving knowledge of v with the claimed inner product.
public class IPAProver {
    /// Generator points G_0, ..., G_{n-1} in affine form.
    public let generators: [PointAffine]
    /// Inner product binding generator Q.
    public let Q: PointAffine

    /// Create an IPA prover with the given generators and binding point.
    /// Generator count must be a power of 2.
    public init(generators: [PointAffine], Q: PointAffine) {
        precondition(generators.count > 0 && (generators.count & (generators.count - 1)) == 0,
                     "Generator count must be a power of 2")
        self.generators = generators
        self.Q = Q
    }

    /// Create a proof that <v, u> = t where C = MSM(G, v) + t * Q.
    ///
    /// - Parameters:
    ///   - v: the committed vector (witness)
    ///   - u: the public evaluation vector
    /// - Returns: (proof, challenges) for verification
    public func prove(v: [Fr], u: [Fr]) -> IPAProofData {
        let n = v.count
        precondition(n == u.count, "v and u must have equal length")
        precondition(n == generators.count, "Vector length must match generator count")
        precondition(n > 0 && (n & (n - 1)) == 0, "Vector length must be power of 2")

        let qProj = pointFromAffine(Q)
        let logN = Int(log2(Double(n)))

        var Ls = [PointProjective]()
        var Rs = [PointProjective]()
        Ls.reserveCapacity(logN)
        Rs.reserveCapacity(logN)

        // Build initial commitment for transcript: C = MSM(G, v) + <v, u> * Q
        let ip = cFrInnerProduct(v, u)
        let commitment = computeCommitment(scalars: v, ip: ip)

        var transcript = IPATranscript()
        transcript.appendPoint(commitment)
        transcript.appendScalar(ip)

        // Working copies (flat UInt64 buffers for C interop)
        var aFlat = [UInt64](repeating: 0, count: n * 4)
        var bFlat = [UInt64](repeating: 0, count: n * 4)
        var gFlat = [UInt64](repeating: 0, count: n * 12)
        var gFoldBuf = [UInt64](repeating: 0, count: n * 12)

        // Copy v into aFlat
        v.withUnsafeBytes { src in
            aFlat.withUnsafeMutableBytes { dst in
                dst.copyMemory(from: UnsafeRawBufferPointer(start: src.baseAddress, count: n * 32))
            }
        }
        // Copy u into bFlat
        u.withUnsafeBytes { src in
            bFlat.withUnsafeMutableBytes { dst in
                dst.copyMemory(from: UnsafeRawBufferPointer(start: src.baseAddress, count: n * 32))
            }
        }
        // Convert affine generators to projective flat buffer
        let fpOne: [UInt64] = [0xd35d438dc58f0d9d, 0x0a78eb28f5c70b3d, 0x666ea36f7879462c, 0x0e0a77c19a07df2f]
        generators.withUnsafeBytes { src in
            for i in 0..<n {
                gFlat.withUnsafeMutableBytes { dst in
                    dst.baseAddress!.advanced(by: i * 96).copyMemory(
                        from: src.baseAddress!.advanced(by: i * 64), byteCount: 64)
                }
                gFlat[i * 12 + 8] = fpOne[0]
                gFlat[i * 12 + 9] = fpOne[1]
                gFlat[i * 12 + 10] = fpOne[2]
                gFlat[i * 12 + 11] = fpOne[3]
            }
        }

        var halfLen = n / 2
        var aLoLimbs = [UInt32](repeating: 0, count: n * 8)
        var aHiLimbs = [UInt32](repeating: 0, count: n * 8)

        for _ in 0..<logN {
            // Compute cross inner products from flat buffers
            var crossL = Fr.zero
            var crossR = Fr.zero
            aFlat.withUnsafeBufferPointer { aPtr in
                bFlat.withUnsafeBufferPointer { bPtr in
                    withUnsafeMutableBytes(of: &crossL) { clBuf in
                        bn254_fr_inner_product(
                            aPtr.baseAddress!,
                            bPtr.baseAddress! + halfLen * 4,
                            Int32(halfLen),
                            clBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
                    }
                    withUnsafeMutableBytes(of: &crossR) { crBuf in
                        bn254_fr_inner_product(
                            aPtr.baseAddress! + halfLen * 4,
                            bPtr.baseAddress!,
                            Int32(halfLen),
                            crBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
                    }
                }
            }

            // Convert a_lo, a_hi to UInt32 limbs for MSM
            aFlat.withUnsafeBufferPointer { aPtr in
                aLoLimbs.withUnsafeMutableBufferPointer { loBuf in
                    bn254_fr_batch_to_limbs(aPtr.baseAddress!, loBuf.baseAddress!, Int32(halfLen))
                }
                aHiLimbs.withUnsafeMutableBufferPointer { hiBuf in
                    bn254_fr_batch_to_limbs(
                        aPtr.baseAddress! + halfLen * 4, hiBuf.baseAddress!, Int32(halfLen))
                }
            }

            // L = MSM(G_hi, a_lo) + crossL * Q
            // R = MSM(G_lo, a_hi) + crossR * Q
            var msmL = PointProjective(x: .one, y: .one, z: .zero)
            var msmR = PointProjective(x: .one, y: .one, z: .zero)

            gFlat.withUnsafeBufferPointer { gPtr in
                aLoLimbs.withUnsafeBufferPointer { alBuf in
                    aHiLimbs.withUnsafeBufferPointer { ahBuf in
                        withUnsafeMutableBytes(of: &msmL) { lBuf in
                            withUnsafeMutableBytes(of: &msmR) { rBuf in
                                bn254_dual_msm_projective(
                                    UnsafeRawPointer(gPtr.baseAddress! + halfLen * 12)
                                        .assumingMemoryBound(to: UInt64.self),
                                    alBuf.baseAddress!,
                                    Int32(halfLen),
                                    UnsafeRawPointer(gPtr.baseAddress!)
                                        .assumingMemoryBound(to: UInt64.self),
                                    ahBuf.baseAddress!,
                                    Int32(halfLen),
                                    lBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
                            }
                        }
                    }
                }
            }

            let cLQ = cPointScalarMul(qProj, crossL)
            let cRQ = cPointScalarMul(qProj, crossR)
            let L = pointAdd(msmL, cLQ)
            let R = pointAdd(msmR, cRQ)

            Ls.append(L)
            Rs.append(R)

            // Fiat-Shamir challenge
            transcript.appendPoint(L)
            transcript.appendPoint(R)
            let x = transcript.deriveChallenge()

            // Compute x^{-1} using C CIOS
            var xInv = Fr.zero
            withUnsafeBytes(of: x) { xBuf in
                withUnsafeMutableBytes(of: &xInv) { rBuf in
                    bn254_fr_inverse(
                        xBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
                }
            }

            // Fold a, b in-place
            aFlat.withUnsafeMutableBufferPointer { aPtr in
                bFlat.withUnsafeMutableBufferPointer { bPtr in
                    withUnsafeBytes(of: x) { xBuf in
                        withUnsafeBytes(of: xInv) { xiBuf in
                            let xp = xBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                            let xip = xiBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                            bn254_fr_vector_fold(
                                aPtr.baseAddress!, aPtr.baseAddress! + halfLen * 4,
                                xp, xip, Int32(halfLen), aPtr.baseAddress!)
                            bn254_fr_vector_fold(
                                bPtr.baseAddress!, bPtr.baseAddress! + halfLen * 4,
                                xip, xp, Int32(halfLen), bPtr.baseAddress!)
                        }
                    }
                }
            }

            // Fold generators using C multi-threaded implementation
            let xLimbs = frToLimbs(x)
            let xInvLimbs = frToLimbs(xInv)
            gFlat.withUnsafeBufferPointer { glBuf in
                xLimbs.withUnsafeBufferPointer { xBuf in
                    xInvLimbs.withUnsafeBufferPointer { xiBuf in
                        gFoldBuf.withUnsafeMutableBufferPointer { outBuf in
                            bn254_fold_generators(
                                UnsafeRawPointer(glBuf.baseAddress!)
                                    .assumingMemoryBound(to: UInt64.self),
                                UnsafeRawPointer(glBuf.baseAddress! + halfLen * 12)
                                    .assumingMemoryBound(to: UInt64.self),
                                xBuf.baseAddress!,
                                xiBuf.baseAddress!,
                                Int32(halfLen),
                                UnsafeMutableRawPointer(outBuf.baseAddress!)
                                    .assumingMemoryBound(to: UInt64.self))
                        }
                    }
                }
            }
            // Copy folded generators back
            gFoldBuf.withUnsafeBytes { src in
                gFlat.withUnsafeMutableBytes { dst in
                    dst.baseAddress!.copyMemory(from: src.baseAddress!, byteCount: halfLen * 96)
                }
            }

            halfLen /= 2
        }

        // Extract final scalar
        let finalA: Fr = aFlat.withUnsafeBytes { buf in
            buf.load(as: Fr.self)
        }

        return IPAProofData(Ls: Ls, Rs: Rs, finalA: finalA)
    }

    // MARK: - Private Helpers

    /// Compute commitment C = MSM(G, scalars) + ip * Q
    private func computeCommitment(scalars: [Fr], ip: Fr) -> PointProjective {
        let n = scalars.count
        let scalarLimbs = scalars.map { frToLimbs($0) }
        let gens = Array(generators.prefix(n))
        let msmResult = cPippengerMSM(points: gens, scalars: scalarLimbs)
        let ipQ = cPointScalarMul(pointFromAffine(Q), ip)
        return pointAdd(msmResult, ipQ)
    }
}

// MARK: - IPA Verifier

/// Verifier for the inner product argument.
///
/// Checks an IPAProofData against a commitment C, public vector u,
/// and claimed inner product value t = <v, u>.
public class IPAVerifier {
    /// Generator points G_0, ..., G_{n-1} in affine form.
    public let generators: [PointAffine]
    /// Inner product binding generator Q.
    public let Q: PointAffine

    /// Create an IPA verifier with the given generators and binding point.
    public init(generators: [PointAffine], Q: PointAffine) {
        precondition(generators.count > 0 && (generators.count & (generators.count - 1)) == 0,
                     "Generator count must be a power of 2")
        self.generators = generators
        self.Q = Q
    }

    /// Verify an IPA proof.
    ///
    /// Given:
    ///   - commitment C = MSM(G, v) + <v, u> * Q
    ///   - public evaluation vector u
    ///   - claimed inner product value t = <v, u>
    ///   - proof containing (L[], R[], final_a)
    ///
    /// Checks:
    ///   1. Reconstructs challenges from transcript (Fiat-Shamir)
    ///   2. Folds the commitment using L_i, R_i and challenges
    ///   3. Computes folded generator G_final and folded b_final
    ///   4. Checks: C_folded == final_a * G_final + (final_a * b_final) * Q
    ///
    /// - Parameters:
    ///   - commitment: the original Pedersen commitment C
    ///   - u: the public evaluation vector
    ///   - innerProduct: the claimed inner product t = <v, u>
    ///   - proof: the IPA proof data
    /// - Returns: true if the proof verifies
    public func verify(commitment: PointProjective, u: [Fr],
                       innerProduct t: Fr, proof: IPAProofData) -> Bool {
        let n = generators.count
        let logN = Int(log2(Double(n)))
        guard proof.Ls.count == logN, proof.Rs.count == logN else { return false }
        guard u.count == n else { return false }

        let qProj = pointFromAffine(Q)

        // Reconstruct Fiat-Shamir challenges from transcript
        var transcript = IPATranscript()
        transcript.appendPoint(commitment)
        transcript.appendScalar(t)

        var challenges = [Fr]()
        var challengeInvs = [Fr]()
        challenges.reserveCapacity(logN)
        challengeInvs.reserveCapacity(logN)

        for round in 0..<logN {
            transcript.appendPoint(proof.Ls[round])
            transcript.appendPoint(proof.Rs[round])
            let x = transcript.deriveChallenge()
            challenges.append(x)

            var xInv = Fr.zero
            withUnsafeBytes(of: x) { xBuf in
                withUnsafeMutableBytes(of: &xInv) { rBuf in
                    bn254_fr_inverse(
                        xBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
                }
            }
            challengeInvs.append(xInv)
        }

        // Fold commitment: C' = C + sum(x_i^2 * L_i + x_i^{-2} * R_i)
        var Cprime = commitment
        for round in 0..<logN {
            let x = challenges[round]
            let xInv = challengeInvs[round]
            let x2 = frMul(x, x)
            let xInv2 = frMul(xInv, xInv)
            let lTerm = cPointScalarMul(proof.Ls[round], x2)
            let rTerm = cPointScalarMul(proof.Rs[round], xInv2)
            Cprime = pointAdd(Cprime, pointAdd(lTerm, rTerm))
        }

        // Compute s[i] = product of x_j^{+-1} based on bit decomposition
        // s[i] = prod_{j=0}^{logN-1} x_j^{bit(i,logN-1-j)} * x_j^{-(1-bit(i,logN-1-j))}
        var s = [Fr](repeating: Fr.one, count: n)
        for round in 0..<logN {
            let x = challenges[round]
            let xInv = challengeInvs[round]
            for i in 0..<n {
                let bit = (i >> (logN - 1 - round)) & 1
                s[i] = frMul(s[i], bit == 1 ? x : xInv)
            }
        }

        // G_final = MSM(G, s) using C Pippenger
        var sLimbs = [UInt32](repeating: 0, count: n * 8)
        s.withUnsafeBytes { sBuf in
            sLimbs.withUnsafeMutableBufferPointer { lBuf in
                bn254_fr_batch_to_limbs(
                    sBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    lBuf.baseAddress!,
                    Int32(n))
            }
        }
        var gFinal = PointProjective(x: .one, y: .one, z: .zero)
        generators.withUnsafeBytes { ptsBuf in
            sLimbs.withUnsafeBufferPointer { scBuf in
                withUnsafeMutableBytes(of: &gFinal) { resBuf in
                    bn254_pippenger_msm(
                        ptsBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        scBuf.baseAddress!,
                        Int32(n),
                        resBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
                }
            }
        }

        // Fold b (= u) using challenges
        var bFolded = u
        var halfLen = n / 2
        for round in 0..<logN {
            let bL = Array(bFolded.prefix(halfLen))
            let bR = Array(bFolded.suffix(halfLen))
            bFolded = cFrVectorFold(bL, bR, x: challengeInvs[round], xInv: challenges[round])
            halfLen /= 2
        }
        let bFinal = bFolded[0]

        // Final check: C' == proof.finalA * G_final + (proof.finalA * bFinal) * Q
        let aG = cPointScalarMul(gFinal, proof.finalA)
        let ab = frMul(proof.finalA, bFinal)
        let abQ = cPointScalarMul(qProj, ab)
        let expected = pointAdd(aG, abQ)

        return pointEqual(Cprime, expected)
    }
}
