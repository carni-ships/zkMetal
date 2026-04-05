// Groth16 Prover — GPU-accelerated proving for BN254
// Supports GPU MSM for [A]/[B]/[C], GPU NTT for H(x), and GPU witness generation
import Foundation
import Metal
import NeonFieldOps

public class Groth16Prover {
    public let msm: MetalMSM; public let ntt: NTTEngine
    public var profileGroth16 = false

    /// GPU NTT threshold: use GPU NTT when bigN (2*domainN) exceeds this.
    /// Below this, CPU NTT is faster due to GPU dispatch overhead.
    public var gpuNTTThreshold: Int = 4096

    /// GPU MSM threshold: use Metal MSM when point count exceeds this.
    /// Below this, C Pippenger on CPU is faster.
    public var gpuMSMThreshold: Int = 256

    public init() throws {
        self.msm = try MetalMSM()
        self.ntt = try NTTEngine()
    }

    // MARK: - GPU-Accelerated Witness Generation

    /// Generate witness from R1CS + public inputs using wave-parallel constraint solver.
    /// For circuits with independent constraint waves (e.g., data-parallel circuits),
    /// uses GPU-accelerated batch solving. For sequential circuits, uses CPU solver
    /// with batch Montgomery inversion for efficiency.
    /// Returns the full z vector: [1, publicInputs..., witness...]
    public func generateWitness(r1cs: R1CSInstance, publicInputs: [Fr],
                                hints: [Int: Fr] = [:]) -> [Fr] {
        // Convert R1CSInstance sparse entries to R1CSConstraint format
        let constraints = r1csToConstraints(r1cs: r1cs)

        // Use wave-parallel CPU solver with batch inversion
        // (GPU wave solver has dependency issues with sequential constraint chains)
        return cpuGenerateWitness(
            constraints: constraints,
            publicInputs: publicInputs,
            numVariables: r1cs.numVars,
            hints: hints
        )
    }

    /// Prove with automatic witness generation from public inputs + hints.
    /// Uses GPU-accelerated witness solver when available.
    public func proveWithWitnessGen(pk: Groth16ProvingKey, r1cs: R1CSInstance,
                                     publicInputs: [Fr],
                                     hints: [Int: Fr] = [:]) throws -> Groth16Proof {
        var _t = CFAbsoluteTimeGetCurrent()
        let z = generateWitness(r1cs: r1cs, publicInputs: publicInputs, hints: hints)
        if profileGroth16 {
            let _e = CFAbsoluteTimeGetCurrent()
            fputs(String(format: "  [groth16] witness gen: %.2f ms\n", (_e - _t) * 1000), stderr)
            _t = _e
        }

        // Extract witness portion (skip 1 + numPublic entries)
        let witness = Array(z[(1 + r1cs.numPublic)...])
        return try prove(pk: pk, r1cs: r1cs, publicInputs: publicInputs, witness: witness)
    }

    public func prove(pk: Groth16ProvingKey, r1cs: R1CSInstance,
                      publicInputs: [Fr], witness: [Fr]) throws -> Groth16Proof {
        let nP = r1cs.numPublic
        var z = [Fr](repeating: .zero, count: r1cs.numVars)
        z[0] = .one; for i in 0..<nP { z[1+i] = publicInputs[i] }
        for i in 0..<witness.count { z[1+nP+i] = witness[i] }
        precondition(r1cs.isSatisfied(z: z), "R1CS not satisfied")
        let r = groth16RandomFr(); let s = groth16RandomFr()

        // Phase 1: computeH, proofA, proofBG2, proofBG1 are all independent
        var _t = CFAbsoluteTimeGetCurrent()
        var h: [Fr]?; var pA: PointProjective?; var pB: G2ProjectivePoint?; var pBg1: PointProjective?
        var hError: Error?; var aError: Error?; var bg1Error: Error?
        let group = DispatchGroup()

        // proofBG2 is CPU-only (G2 has no GPU MSM) — run on separate thread
        group.enter()
        DispatchQueue.global(qos: .userInitiated).async { [self] in
            pB = proofBG2(pk: pk, z: z, s: s)
            group.leave()
        }

        // computeH uses GPU NTT for large circuits, CPU NTT for small
        group.enter()
        DispatchQueue.global(qos: .userInitiated).async { [self] in
            do { h = try computeH(r1cs: r1cs, z: z) } catch { hError = error }
            group.leave()
        }

        // For large circuits, parallelize proofA and proofBG1 on separate threads
        // since each creates its own GPU command buffer
        let n = r1cs.numVars
        if n >= 1024 {
            // Large circuit: run proofA and proofBG1 concurrently
            group.enter()
            DispatchQueue.global(qos: .userInitiated).async { [self] in
                do { pA = try proofA(pk: pk, z: z, r: r) } catch { aError = error }
                group.leave()
            }
            do { pBg1 = try proofBG1(pk: pk, z: z, s: s) } catch { bg1Error = error }
        } else {
            // Small circuit: sequential to avoid GPU contention overhead
            do { pA = try proofA(pk: pk, z: z, r: r) } catch { aError = error }
            do { pBg1 = try proofBG1(pk: pk, z: z, s: s) } catch { bg1Error = error }
        }

        group.wait()
        if profileGroth16 { let _e = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [groth16] phase1 (H+A+BG2+BG1): %.2f ms\n", (_e - _t) * 1000), stderr); _t = _e }

        if let e = hError { throw e }
        if let e = aError { throw e }
        if let e = bg1Error { throw e }

        let pC = try proofC(pk: pk, w: witness, h: h!, a: pA!, bg1: pBg1!, r: r, s: s)
        if profileGroth16 { let _e = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [groth16] proofC: %.2f ms\n", (_e - _t) * 1000), stderr); _t = _e }
        return Groth16Proof(a: pA!, b: pB!, c: pC)
    }

    private func computeH(r1cs: R1CSInstance, z: [Fr]) throws -> [Fr] {
        let m = r1cs.numConstraints
        var domainN = 1; var logN = 0
        while domainN < m { domainN <<= 1; logN += 1 }

        // For large circuits, parallelize the 3 sparse mat-vecs
        let az: [Fr]; let bz: [Fr]; let cz: [Fr]
        if m >= 1024 {
            // Use UnsafeMutablePointer-based storage for thread-safe concurrent writes
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

        // Use CPU NTT for small sizes to avoid GPU dispatch overhead
        let useCPU = bigN <= gpuNTTThreshold

        let aCoeffs: [Fr]; let bCoeffs: [Fr]; let cCoeffs: [Fr]
        if useCPU {
            aCoeffs = cINTT_Fr(aEvals, logN: logN)
            bCoeffs = cINTT_Fr(bEvals, logN: logN)
            cCoeffs = cINTT_Fr(cEvals, logN: logN)
        } else {
            aCoeffs = try ntt.intt(aEvals); bCoeffs = try ntt.intt(bEvals); cCoeffs = try ntt.intt(cEvals)
        }

        var aPad = [Fr](repeating: .zero, count: bigN)
        var bPad = [Fr](repeating: .zero, count: bigN)
        var cPad = [Fr](repeating: .zero, count: bigN)
        for i in 0..<domainN { aPad[i] = aCoeffs[i]; bPad[i] = bCoeffs[i]; cPad[i] = cCoeffs[i] }

        let aE: [Fr]; let bE: [Fr]; let cE: [Fr]
        if useCPU {
            aE = cNTT_Fr(aPad, logN: logBigN)
            bE = cNTT_Fr(bPad, logN: logBigN)
            cE = cNTT_Fr(cPad, logN: logBigN)
        } else {
            aE = try ntt.ntt(aPad); bE = try ntt.ntt(bPad); cE = try ntt.ntt(cPad)
        }

        // Fused pointwise mul-sub via C: pE[i] = aE[i]*bE[i] - cE[i]
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

        let pCoeffs: [Fr]
        if useCPU {
            pCoeffs = cINTT_Fr(pE, logN: logBigN)
        } else {
            pCoeffs = try ntt.intt(pE)
        }

        // Coefficient division by vanishing polynomial via C
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

    private func proofA(pk: Groth16ProvingKey, z: [Fr], r: Fr) throws -> PointProjective {
        let n = min(z.count, pk.a_query_affine.count)
        let (filtA, filtS) = filterNonZero(affine: pk.a_query_affine, proj: pk.a_query, scalars: z, count: n)
        let m = try doMSMAffine(aff: filtA, sc: filtS)
        return pointAdd(pointAdd(pk.alpha_g1, m), pointScalarMul(pk.delta_g1, r))
    }

    private func proofBG2(pk: Groth16ProvingKey, z: [Fr], s: Fr) -> G2ProjectivePoint {
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

    private func proofBG1(pk: Groth16ProvingKey, z: [Fr], s: Fr) throws -> PointProjective {
        let n = min(z.count, pk.b_g1_query_affine.count)
        let (filtA, filtS) = filterNonZero(affine: pk.b_g1_query_affine, proj: pk.b_g1_query, scalars: z, count: n)
        let m = try doMSMAffine(aff: filtA, sc: filtS)
        return pointAdd(pointAdd(pk.beta_g1, m), pointScalarMul(pk.delta_g1, s))
    }

    private func proofC(pk: Groth16ProvingKey, w: [Fr], h: [Fr],
                         a: PointProjective, bg1: PointProjective, r: Fr, s: Fr) throws -> PointProjective {
        let lN = min(w.count, pk.l_query_affine.count)
        let (filtLA, filtLS) = filterNonZero(affine: pk.l_query_affine, proj: pk.l_query, scalars: w, count: lN)
        let lM = try doMSMAffine(aff: filtLA, sc: filtLS)
        let hN = min(h.count, pk.h_query_affine.count)
        let (filtHA, filtHS) = filterNonZero(affine: pk.h_query_affine, proj: pk.h_query, scalars: h, count: hN)
        let hM = filtHA.count > 0 ? try doMSMAffine(aff: filtHA, sc: filtHS) : pointIdentity()
        var res = pointAdd(lM, hM)
        res = pointAdd(res, pointScalarMul(a, s))
        res = pointAdd(res, pointScalarMul(bg1, r))
        return pointAdd(res, pointNeg(pointScalarMul(pk.delta_g1, frMul(r, s))))
    }

    /// Filter out entries where the scalar is zero or the projective point is identity.
    /// Identity points have invalid affine representations from batchToAffine, so they
    /// must be excluded from MSM to avoid corrupting the result.
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

    /// MSM with pre-computed affine points (avoids batchToAffine per call).
    /// Uses GPU Metal MSM for large instances, C Pippenger for small.
    private func doMSMAffine(aff: [PointAffine], sc: [Fr]) throws -> PointProjective {
        let n = aff.count; if n == 0 { return pointIdentity() }
        var sl = [[UInt32]](); sl.reserveCapacity(n)
        for s in sc { sl.append(frToLimbs(s)) }
        return n >= gpuMSMThreshold ? try msm.msm(points: aff, scalars: sl) : cPippengerMSM(points: aff, scalars: sl)
    }

    // MARK: - R1CS to R1CSConstraint Conversion

    /// Convert R1CSInstance (sparse COO format) to array of R1CSConstraint for witness generation.
    private func r1csToConstraints(r1cs: R1CSInstance) -> [R1CSConstraint] {
        // Group entries by row
        var aByRow = [[R1CSEntry]](repeating: [], count: r1cs.numConstraints)
        var bByRow = [[R1CSEntry]](repeating: [], count: r1cs.numConstraints)
        var cByRow = [[R1CSEntry]](repeating: [], count: r1cs.numConstraints)
        for e in r1cs.aEntries { aByRow[e.row].append(e) }
        for e in r1cs.bEntries { bByRow[e.row].append(e) }
        for e in r1cs.cEntries { cByRow[e.row].append(e) }

        var constraints = [R1CSConstraint]()
        constraints.reserveCapacity(r1cs.numConstraints)
        for i in 0..<r1cs.numConstraints {
            let aLC = LinearCombination(aByRow[i].map { ($0.col, $0.val) })
            let bLC = LinearCombination(bByRow[i].map { ($0.col, $0.val) })
            let cLC = LinearCombination(cByRow[i].map { ($0.col, $0.val) })
            constraints.append(R1CSConstraint(a: aLC, b: bLC, c: cLC))
        }
        return constraints
    }

    /// CPU fallback witness generation using simple sequential constraint solving.
    private func cpuGenerateWitness(constraints: [R1CSConstraint], publicInputs: [Fr],
                                     numVariables: Int, hints: [Int: Fr]) -> [Fr] {
        var assignment = [Fr](repeating: Fr.zero, count: numVariables)
        assignment[0] = Fr.one
        for (i, val) in publicInputs.enumerated() { assignment[i + 1] = val }
        for (idx, val) in hints { assignment[idx] = val }

        var known = Set<Int>()
        known.insert(0)
        for i in 0..<publicInputs.count { known.insert(i + 1) }
        for idx in hints.keys { known.insert(idx) }

        let scheduler = WaveScheduler(constraints: constraints, knownVariables: known,
                                       numVariables: numVariables)

        for wave in scheduler.waves {
            for ci in wave {
                guard let targetVar = scheduler.producedVariable[ci] else { continue }
                let c = constraints[ci]
                let aHas = c.a.variables.contains(targetVar)
                let bHas = c.b.variables.contains(targetVar)
                let cHas = c.c.variables.contains(targetVar)

                if cHas && !aHas && !bHas {
                    let aVal = c.a.evaluate(assignment: assignment)
                    let bVal = c.b.evaluate(assignment: assignment)
                    let product = frMul(aVal, bVal)
                    var knownSum = Fr.zero; var coeff = Fr.zero
                    for (idx, co) in c.c.terms {
                        if idx == targetVar { coeff = frAdd(coeff, co) }
                        else { knownSum = frAdd(knownSum, frMul(co, assignment[idx])) }
                    }
                    assignment[targetVar] = frMul(frSub(product, knownSum), frInverse(coeff))
                } else if aHas && !bHas && !cHas {
                    let bVal = c.b.evaluate(assignment: assignment)
                    let cVal = c.c.evaluate(assignment: assignment)
                    let targetAVal = bVal.isZero ? Fr.zero : frMul(cVal, frInverse(bVal))
                    var knownSum = Fr.zero; var coeff = Fr.zero
                    for (idx, co) in c.a.terms {
                        if idx == targetVar { coeff = frAdd(coeff, co) }
                        else { knownSum = frAdd(knownSum, frMul(co, assignment[idx])) }
                    }
                    assignment[targetVar] = frMul(frSub(targetAVal, knownSum), frInverse(coeff))
                } else if bHas && !aHas && !cHas {
                    let aVal = c.a.evaluate(assignment: assignment)
                    let cVal = c.c.evaluate(assignment: assignment)
                    let targetBVal = aVal.isZero ? Fr.zero : frMul(cVal, frInverse(aVal))
                    var knownSum = Fr.zero; var coeff = Fr.zero
                    for (idx, co) in c.b.terms {
                        if idx == targetVar { coeff = frAdd(coeff, co) }
                        else { knownSum = frAdd(knownSum, frMul(co, assignment[idx])) }
                    }
                    assignment[targetVar] = frMul(frSub(targetBVal, knownSum), frInverse(coeff))
                }
            }
        }
        return assignment
    }
}

// MARK: - G2 Straus MSM

/// Straus (Shamir's trick) multi-scalar multiplication for G2 points.
/// Processes all scalars bit-by-bit from MSB to LSB with a single shared doubling.
/// Much faster than N sequential scalar muls: O(256 doubles + 256*density adds)
/// vs O(N * 256 doubles + N * 128 adds).
public func g2PippengerMSM(points: [G2ProjectivePoint], scalars: [[UInt64]]) -> G2ProjectivePoint {
    let n = points.count
    if n == 0 { return g2Identity() }
    if n == 1 { return g2ScalarMul(points[0], scalars[0]) }

    // Find highest set bit to skip leading zeros
    var maxBit = 0
    for sc in scalars {
        for w in 0..<sc.count where sc[w] != 0 {
            let bit = w * 64 + (63 - sc[w].leadingZeroBitCount)
            if bit > maxBit { maxBit = bit }
        }
    }

    // Straus: scan from highest set bit to LSB
    var result = g2Identity()
    for bit in stride(from: maxBit, through: 0, by: -1) {
        result = g2Double(result)
        let wordIdx = bit / 64
        let bitIdx = bit % 64
        for i in 0..<n {
            let sc = scalars[i]
            if wordIdx < sc.count && (sc[wordIdx] >> bitIdx) & 1 == 1 {
                result = g2Add(result, points[i])
            }
        }
    }
    return result
}
