// GPUGroth16ProverEngine — GPU-accelerated Groth16 proof generation engine
//
// High-level engine that wraps the Groth16 proving pipeline with explicit
// GPU acceleration for all computationally intensive phases:
//
//   1. Witness reduction: h(x) = (A*B - C) / Z_H via GPU NTT + pointwise ops + iNTT
//   2. Proof element [A]_1 via GPU MSM
//   3. Proof element [B]_2 via G2 Straus MSM (CPU, no G2 GPU MSM)
//   4. Proof element [C]_1 via GPU MSM
//   5. Random blinding (r, s) for zero-knowledge
//
// Uses MetalMSM for G1 multi-scalar multiplication and NTTEngine for
// forward/inverse NTT on BN254 Fr. Falls back to CPU (C Pippenger, C NTT)
// for small instances where GPU dispatch overhead dominates.

import Foundation
import Metal
import NeonFieldOps

// MARK: - GPU Groth16 Prover Engine

/// GPU-accelerated Groth16 proving engine for BN254.
///
/// Orchestrates proof generation using Metal GPU compute for MSM and NTT,
/// with automatic CPU fallback for small circuits. Produces zero-knowledge
/// proofs with random blinding factors (r, s) applied to [A]_1, [B]_2, [C]_1.
///
/// Usage:
/// ```
/// let engine = try GPUGroth16ProverEngine()
/// let proof = try engine.generateProof(pk: pk, r1cs: r1cs,
///                                       publicInputs: pub, witness: wit)
/// ```
public class GPUGroth16ProverEngine {
    public static let version = Versions.gpuGroth16Prover

    /// Underlying GPU MSM engine (BN254 G1)
    public let msm: MetalMSM

    /// Underlying GPU NTT engine (BN254 Fr)
    public let ntt: NTTEngine

    /// Enable profiling output to stderr
    public var profile = false

    /// GPU NTT threshold: use GPU NTT when 2*domainN exceeds this
    public var gpuNTTThreshold: Int = 4096

    /// GPU MSM threshold: use Metal MSM when point count exceeds this
    public var gpuMSMThreshold: Int = 256

    public init() throws {
        self.msm = try MetalMSM()
        self.ntt = try NTTEngine()
    }

    // MARK: - Public API

    /// Generate a Groth16 proof with random blinding for zero-knowledge.
    ///
    /// - Parameters:
    ///   - pk: Proving key from trusted setup
    ///   - r1cs: R1CS constraint system
    ///   - publicInputs: Public input values (excluding the leading 1)
    ///   - witness: Private witness values
    /// - Returns: Groth16Proof with (A, B, C) group elements
    public func generateProof(pk: Groth16ProvingKey, r1cs: R1CSInstance,
                               publicInputs: [Fr], witness: [Fr]) throws -> Groth16Proof {
        // Build full assignment z = [1, publicInputs..., witness...]
        let nP = r1cs.numPublic
        var z = [Fr](repeating: .zero, count: r1cs.numVars)
        z[0] = .one
        for i in 0..<nP { z[1 + i] = publicInputs[i] }
        for i in 0..<witness.count { z[1 + nP + i] = witness[i] }
        precondition(r1cs.isSatisfied(z: z), "R1CS not satisfied")

        // Sample random blinding factors for zero-knowledge
        let r = groth16RandomFr()
        let s = groth16RandomFr()

        var _t = CFAbsoluteTimeGetCurrent()

        // Phase 1: Compute H(x), [A]_1, [B]_2, [B]_1 in parallel
        var h: [Fr]?
        var proofA: PointProjective?
        var proofB: G2ProjectivePoint?
        var proofBG1: PointProjective?
        var hError: Error?
        var aError: Error?
        var bg1Error: Error?
        let group = DispatchGroup()

        // [B]_2 is CPU-only (G2 has no GPU MSM)
        group.enter()
        DispatchQueue.global(qos: .userInitiated).async { [self] in
            proofB = computeProofB(pk: pk, z: z, s: s)
            group.leave()
        }

        // H(x) uses GPU NTT for large circuits
        group.enter()
        DispatchQueue.global(qos: .userInitiated).async { [self] in
            do { h = try computeWitnessReduction(r1cs: r1cs, z: z) } catch { hError = error }
            group.leave()
        }

        // [A]_1 and [B]_1 via GPU MSM
        let n = r1cs.numVars
        if n >= 1024 {
            group.enter()
            DispatchQueue.global(qos: .userInitiated).async { [self] in
                do { proofA = try computeProofA(pk: pk, z: z, r: r) } catch { aError = error }
                group.leave()
            }
            do { proofBG1 = try computeProofBG1(pk: pk, z: z, s: s) } catch { bg1Error = error }
        } else {
            do { proofA = try computeProofA(pk: pk, z: z, r: r) } catch { aError = error }
            do { proofBG1 = try computeProofBG1(pk: pk, z: z, s: s) } catch { bg1Error = error }
        }

        group.wait()
        if profile {
            let _e = CFAbsoluteTimeGetCurrent()
            fputs(String(format: "  [gpu-groth16] phase1 (H+A+B2+B1): %.2f ms\n", (_e - _t) * 1000), stderr)
            _t = _e
        }

        if let e = hError { throw e }
        if let e = aError { throw e }
        if let e = bg1Error { throw e }

        // Phase 2: Compute [C]_1
        let proofC = try computeProofC(pk: pk, w: witness, h: h!, a: proofA!,
                                        bg1: proofBG1!, r: r, s: s)
        if profile {
            let _e = CFAbsoluteTimeGetCurrent()
            fputs(String(format: "  [gpu-groth16] phase2 (C): %.2f ms\n", (_e - _t) * 1000), stderr)
        }

        return Groth16Proof(a: proofA!, b: proofB!, c: proofC)
    }

    /// Generate proof with automatic witness generation from public inputs.
    public func generateProofWithWitness(pk: Groth16ProvingKey, r1cs: R1CSInstance,
                                          publicInputs: [Fr],
                                          hints: [Int: Fr] = [:]) throws -> Groth16Proof {
        let prover = try Groth16Prover()
        let z = prover.generateWitness(r1cs: r1cs, publicInputs: publicInputs, hints: hints)
        let witness = Array(z[(1 + r1cs.numPublic)...])
        return try generateProof(pk: pk, r1cs: r1cs, publicInputs: publicInputs, witness: witness)
    }

    /// Access blinding factors for testing/debugging (generates fresh each call).
    public func sampleBlindingFactors() -> (r: Fr, s: Fr) {
        return (groth16RandomFr(), groth16RandomFr())
    }

    // MARK: - Witness Reduction: h(x) = (A*B - C) / Z_H

    /// Compute the quotient polynomial h(x) via NTT-based polynomial arithmetic.
    ///
    /// Steps:
    ///   1. Evaluate A*z, B*z, C*z via sparse mat-vec
    ///   2. iNTT to get coefficient form
    ///   3. Zero-pad to 2*domainN, NTT to evaluation form
    ///   4. Pointwise: p(x) = A(x)*B(x) - C(x)
    ///   5. iNTT back to coefficient form
    ///   6. Divide by vanishing polynomial Z_H
    private func computeWitnessReduction(r1cs: R1CSInstance, z: [Fr]) throws -> [Fr] {
        let m = r1cs.numConstraints
        var domainN = 1; var logN = 0
        while domainN < m { domainN <<= 1; logN += 1 }

        // Sparse mat-vec: parallelize for large circuits
        let az: [Fr]; let bz: [Fr]; let cz: [Fr]
        if m >= 1024 {
            let results = UnsafeMutablePointer<[Fr]>.allocate(capacity: 3)
            results.initialize(repeating: [], count: 3)
            DispatchQueue.concurrentPerform(iterations: 3) { idx in
                switch idx {
                case 0: results[0] = r1cs.sparseMatVec(r1cs.aEntries, z)
                case 1: results[1] = r1cs.sparseMatVec(r1cs.bEntries, z)
                default: results[2] = r1cs.sparseMatVec(r1cs.cEntries, z)
                }
            }
            az = results[0]; bz = results[1]; cz = results[2]
            results.deinitialize(count: 3); results.deallocate()
        } else {
            az = r1cs.sparseMatVec(r1cs.aEntries, z)
            bz = r1cs.sparseMatVec(r1cs.bEntries, z)
            cz = r1cs.sparseMatVec(r1cs.cEntries, z)
        }

        // Pad to domain size
        var aEvals = [Fr](repeating: .zero, count: domainN)
        var bEvals = [Fr](repeating: .zero, count: domainN)
        var cEvals = [Fr](repeating: .zero, count: domainN)
        for i in 0..<m { aEvals[i] = az[i]; bEvals[i] = bz[i]; cEvals[i] = cz[i] }

        let bigN = domainN * 2
        let logBigN = logN + 1
        let useCPU = bigN <= gpuNTTThreshold

        // iNTT: evaluation -> coefficient
        let aCoeffs: [Fr]; let bCoeffs: [Fr]; let cCoeffs: [Fr]
        if useCPU {
            aCoeffs = cINTT_Fr(aEvals, logN: logN)
            bCoeffs = cINTT_Fr(bEvals, logN: logN)
            cCoeffs = cINTT_Fr(cEvals, logN: logN)
        } else {
            aCoeffs = try ntt.intt(aEvals)
            bCoeffs = try ntt.intt(bEvals)
            cCoeffs = try ntt.intt(cEvals)
        }

        // Zero-pad to 2*domainN
        var aPad = [Fr](repeating: .zero, count: bigN)
        var bPad = [Fr](repeating: .zero, count: bigN)
        var cPad = [Fr](repeating: .zero, count: bigN)
        for i in 0..<domainN { aPad[i] = aCoeffs[i]; bPad[i] = bCoeffs[i]; cPad[i] = cCoeffs[i] }

        // NTT: coefficient -> evaluation on doubled domain
        let aE: [Fr]; let bE: [Fr]; let cE: [Fr]
        if useCPU {
            aE = cNTT_Fr(aPad, logN: logBigN)
            bE = cNTT_Fr(bPad, logN: logBigN)
            cE = cNTT_Fr(cPad, logN: logBigN)
        } else {
            aE = try ntt.ntt(aPad)
            bE = try ntt.ntt(bPad)
            cE = try ntt.ntt(cPad)
        }

        // Pointwise: p[i] = a[i]*b[i] - c[i] via C NEON
        var pE = [Fr](repeating: .zero, count: bigN)
        aE.withUnsafeBytes { aPtr in
            bE.withUnsafeBytes { bPtr in
                cE.withUnsafeBytes { cPtr in
                    pE.withUnsafeMutableBytes { pPtr in
                        bn254_fr_pointwise_mul_sub(
                            aPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            bPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            cPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            pPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(bigN))
                    }
                }
            }
        }

        // iNTT: evaluation -> coefficient
        let pCoeffs: [Fr]
        if useCPU {
            pCoeffs = cINTT_Fr(pE, logN: logBigN)
        } else {
            pCoeffs = try ntt.intt(pE)
        }

        // Divide by vanishing polynomial Z_H via C
        var hCoeffs = [Fr](repeating: .zero, count: domainN)
        pCoeffs.withUnsafeBytes { pPtr in
            hCoeffs.withUnsafeMutableBytes { hPtr in
                bn254_fr_coeff_div_vanishing(
                    pPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(domainN),
                    hPtr.baseAddress!.assumingMemoryBound(to: UInt64.self))
            }
        }
        return hCoeffs
    }

    // MARK: - Proof Element Computation

    /// Compute [A]_1 = alpha_g1 + MSM(a_query, z) + r * delta_g1
    private func computeProofA(pk: Groth16ProvingKey, z: [Fr], r: Fr) throws -> PointProjective {
        let n = min(z.count, pk.a_query_affine.count)
        let (filtA, filtS) = filterNonZero(affine: pk.a_query_affine, proj: pk.a_query,
                                            scalars: z, count: n)
        let m = try performMSM(affine: filtA, scalars: filtS)
        return pointAdd(pointAdd(pk.alpha_g1, m), pointScalarMul(pk.delta_g1, r))
    }

    /// Compute [B]_2 = beta_g2 + MSM_G2(b_g2_query, z) + s * delta_g2
    /// CPU-only: no GPU MSM for G2 points
    private func computeProofB(pk: Groth16ProvingKey, z: [Fr], s: Fr) -> G2ProjectivePoint {
        let n = min(z.count, pk.b_g2_query.count)
        var pts = [G2ProjectivePoint](); var scs = [[UInt64]]()
        pts.reserveCapacity(n + 1); scs.reserveCapacity(n + 1)
        for i in 0..<n {
            if !z[i].isZero { pts.append(pk.b_g2_query[i]); scs.append(frToInt(z[i])) }
        }
        pts.append(pk.delta_g2); scs.append(frToInt(s))
        let m = g2PippengerMSM(points: pts, scalars: scs)
        return g2Add(pk.beta_g2, m)
    }

    /// Compute [B]_1 = beta_g1 + MSM(b_g1_query, z) + s * delta_g1
    private func computeProofBG1(pk: Groth16ProvingKey, z: [Fr], s: Fr) throws -> PointProjective {
        let n = min(z.count, pk.b_g1_query_affine.count)
        let (filtA, filtS) = filterNonZero(affine: pk.b_g1_query_affine, proj: pk.b_g1_query,
                                            scalars: z, count: n)
        let m = try performMSM(affine: filtA, scalars: filtS)
        return pointAdd(pointAdd(pk.beta_g1, m), pointScalarMul(pk.delta_g1, s))
    }

    /// Compute [C]_1 = MSM(l_query, w) + MSM(h_query, h) + s*A + r*BG1 - r*s*delta_g1
    private func computeProofC(pk: Groth16ProvingKey, w: [Fr], h: [Fr],
                                a: PointProjective, bg1: PointProjective,
                                r: Fr, s: Fr) throws -> PointProjective {
        let lN = min(w.count, pk.l_query_affine.count)
        let (filtLA, filtLS) = filterNonZero(affine: pk.l_query_affine, proj: pk.l_query,
                                              scalars: w, count: lN)
        let lM = try performMSM(affine: filtLA, scalars: filtLS)

        let hN = min(h.count, pk.h_query_affine.count)
        let (filtHA, filtHS) = filterNonZero(affine: pk.h_query_affine, proj: pk.h_query,
                                              scalars: h, count: hN)
        let hM = filtHA.count > 0 ? try performMSM(affine: filtHA, scalars: filtHS) : pointIdentity()

        var res = pointAdd(lM, hM)
        res = pointAdd(res, pointScalarMul(a, s))
        res = pointAdd(res, pointScalarMul(bg1, r))
        return pointAdd(res, pointNeg(pointScalarMul(pk.delta_g1, frMul(r, s))))
    }

    // MARK: - Internal Helpers

    /// Filter out zero scalars and identity points for clean MSM input.
    private func filterNonZero(affine: [PointAffine], proj: [PointProjective],
                                scalars: [Fr], count: Int) -> ([PointAffine], [Fr]) {
        var filtA = [PointAffine](); var filtS = [Fr]()
        filtA.reserveCapacity(count); filtS.reserveCapacity(count)
        for i in 0..<count {
            if !scalars[i].isZero && !pointIsIdentity(proj[i]) {
                filtA.append(affine[i]); filtS.append(scalars[i])
            }
        }
        return (filtA, filtS)
    }

    /// Perform MSM using GPU Metal for large instances, C Pippenger for small.
    private func performMSM(affine: [PointAffine], scalars: [Fr]) throws -> PointProjective {
        let n = affine.count
        if n == 0 { return pointIdentity() }
        var sl = [[UInt32]](); sl.reserveCapacity(n)
        for s in scalars { sl.append(frToLimbs(s)) }
        return n >= gpuMSMThreshold ? try msm.msm(points: affine, scalars: sl)
                                    : cPippengerMSM(points: affine, scalars: sl)
    }
}
