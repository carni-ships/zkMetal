// GPUIPAEngine — GPU-accelerated IPA polynomial commitment scheme
//
// Implements the Inner Product Argument (IPA) as a polynomial commitment scheme,
// using GPU MSM for the expensive multi-scalar multiplications in both prover
// and verifier. This is the Halo-style IPA where polynomials are committed
// via Pedersen vector commitment (MSM over coefficient vector).
//
// Key features:
//   - Commit to polynomial via Pedersen vector commitment using GPU MSM
//   - IPA opening proof: recursive halving protocol (L, R commitments each round)
//   - IPA verification: reconstruct commitment and check final scalar relation
//   - Batch IPA opening (multiple polynomials at same point via random linear combination)
//
// References:
//   - Bulletproofs (Bunz et al. 2018) — inner product argument
//   - Halo (Bowe et al. 2019) — IPA as polynomial commitment scheme

import Foundation
import Metal
import NeonFieldOps

// MARK: - IPA Opening Proof

/// Proof for an IPA polynomial opening at a single point.
/// Contains log(n) round commitments (L_i, R_i), the final scalar, and the evaluation.
public struct IPAOpeningProof {
    /// Left commitments per round (log(n) entries).
    public let Ls: [PointProjective]
    /// Right commitments per round (log(n) entries).
    public let Rs: [PointProjective]
    /// Final scalar after all folding rounds.
    public let finalA: Fr
    /// The evaluation p(z) at the opening point.
    public let evaluation: Fr

    public init(Ls: [PointProjective], Rs: [PointProjective], finalA: Fr, evaluation: Fr) {
        self.Ls = Ls
        self.Rs = Rs
        self.finalA = finalA
        self.evaluation = evaluation
    }

    /// Number of rounds in the proof (should equal log2(polynomial degree)).
    public var rounds: Int { Ls.count }
}

// MARK: - Batch IPA Opening Proof

/// Proof for a batch IPA opening of multiple polynomials at a single point.
public struct BatchIPAOpeningProof {
    /// Individual evaluations [p_0(z), ..., p_{k-1}(z)]
    public let evaluations: [Fr]
    /// Combined IPA proof for the random linear combination
    public let combinedProof: IPAOpeningProof
    /// The batching challenge gamma
    public let gamma: Fr
    /// The evaluation point z
    public let point: Fr

    public init(evaluations: [Fr], combinedProof: IPAOpeningProof, gamma: Fr, point: Fr) {
        self.evaluations = evaluations
        self.combinedProof = combinedProof
        self.gamma = gamma
        self.point = point
    }
}

// MARK: - IPA Transcript (Fiat-Shamir)

/// Transcript for Fiat-Shamir challenge derivation in IPA proofs.
private struct GPUIPATranscript {
    var data = [UInt8]()

    mutating func appendPoint(_ p: PointProjective) {
        // Normalize to affine to ensure consistent hashing regardless of projective representation
        if pointIsIdentity(p) {
            data.append(contentsOf: [UInt8](repeating: 0, count: 64))
        } else {
            let aff = batchToAffine([p])[0]
            withUnsafeBytes(of: aff) { buf in
                let ptr = buf.baseAddress!.assumingMemoryBound(to: UInt8.self)
                data.append(contentsOf: UnsafeBufferPointer(start: ptr, count: 64))
            }
        }
    }

    mutating func appendScalar(_ v: Fr) {
        withUnsafeBytes(of: v) { buf in
            let ptr = buf.baseAddress!.assumingMemoryBound(to: UInt8.self)
            data.append(contentsOf: UnsafeBufferPointer(start: ptr, count: 32))
        }
    }

    mutating func appendLabel(_ label: String) {
        data.append(contentsOf: Array(label.utf8))
    }

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
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))
    }
}

// MARK: - GPU IPA Engine

/// GPU-accelerated IPA polynomial commitment engine.
///
/// Uses Metal GPU MSM for Pedersen vector commitments and the recursive
/// halving protocol. Falls back to CPU Pippenger for small sizes.
///
/// Usage:
///   let engine = try GPUIPAEngine(maxDegree: 256)
///   let commitment = try engine.commit(poly)
///   let proof = try engine.open(poly, at: z)
///   let valid = engine.verify(commitment: c, point: z, proof: proof)
public class GPUIPAEngine {
    public static let version = Versions.gpuIPAEngine

    /// Generator points G_0, ..., G_{n-1} in affine form.
    public let generators: [PointAffine]
    /// Inner product binding generator Q.
    public let Q: PointAffine
    /// Maximum supported polynomial degree.
    public let maxDegree: Int

    /// Cached GPU MSM engine (lazy-initialized).
    private var _msmEngine: MetalMSM?

    /// GPU MSM threshold: use GPU for vectors of this size or larger.
    public static let gpuThreshold = 64

    /// Create GPU IPA engine supporting polynomials up to the given degree.
    /// Degree must be a power of 2.
    ///
    /// Generators are derived deterministically from the BN254 G1 generator.
    public init(maxDegree: Int) throws {
        precondition(maxDegree > 0 && (maxDegree & (maxDegree - 1)) == 0,
                     "maxDegree must be a power of 2, got \(maxDegree)")
        self.maxDegree = maxDegree

        // Generate n+1 distinct points via iterated double-add from G1
        let gx = fpFromInt(1)
        let gy = fpFromInt(2)
        let g = pointFromAffine(PointAffine(x: gx, y: gy))

        var projPoints = [PointProjective]()
        projPoints.reserveCapacity(maxDegree + 1)
        var acc = g
        for _ in 0..<(maxDegree + 1) {
            projPoints.append(acc)
            acc = pointDouble(pointAdd(acc, g))
        }

        let affinePoints = batchToAffine(projPoints)
        self.generators = Array(affinePoints.prefix(maxDegree))
        self.Q = affinePoints[maxDegree]
    }

    /// Create GPU IPA engine with explicit generators and binding point.
    public init(generators: [PointAffine], Q: PointAffine) {
        precondition(generators.count > 0 && (generators.count & (generators.count - 1)) == 0,
                     "Generator count must be a power of 2")
        self.generators = generators
        self.Q = Q
        self.maxDegree = generators.count
    }

    // MARK: - GPU Engine

    private func getMSMEngine() -> MetalMSM? {
        if _msmEngine == nil { _msmEngine = try? MetalMSM() }
        return _msmEngine
    }

    // MARK: - Scalar conversion

    private func batchFrToLimbs(_ coeffs: [Fr]) -> [[UInt32]] {
        return coeffs.map { frToLimbs($0) }
    }

    // MARK: - Polynomial evaluation (CPU, Horner)

    private func evaluate(_ coeffs: [Fr], at z: Fr) -> Fr {
        var result = Fr.zero
        coeffs.withUnsafeBytes { cBuf in
            withUnsafeBytes(of: z) { zBuf in
                withUnsafeMutableBytes(of: &result) { rBuf in
                    bn254_fr_horner_eval(
                        cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(coeffs.count),
                        zBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                    )
                }
            }
        }
        return result
    }

    // MARK: - Commit (Pedersen vector commitment via MSM)

    /// Commit to a polynomial: C = MSM(G[0..n], coefficients)
    ///
    /// Uses GPU MSM for polynomials >= gpuThreshold, CPU Pippenger otherwise.
    public func commit(_ coeffs: [Fr]) throws -> PointProjective {
        let n = coeffs.count
        guard n > 0 else { return pointIdentity() }
        guard n <= maxDegree else {
            fatalError("Polynomial degree \(n) exceeds maxDegree \(maxDegree)")
        }

        // Pad to power of 2 for generator alignment
        let gens = Array(generators.prefix(n))
        let scalars = batchFrToLimbs(coeffs)

        if n >= GPUIPAEngine.gpuThreshold, let engine = getMSMEngine() {
            return (try? engine.msm(points: gens, scalars: scalars))
                ?? cPippengerMSM(points: gens, scalars: scalars)
        }
        return cPippengerMSM(points: gens, scalars: scalars)
    }

    // MARK: - Build evaluation vector

    /// Build the evaluation vector u = [1, z, z^2, ..., z^{n-1}].
    /// The inner product <coeffs, u> = p(z).
    private func buildEvalVector(point z: Fr, length n: Int) -> [Fr] {
        var u = [Fr](repeating: Fr.zero, count: n)
        u[0] = Fr.one
        if n > 1 {
            u[1] = z
            for i in 2..<n {
                u[i] = frMul(u[i - 1], z)
            }
        }
        return u
    }

    // MARK: - Open (IPA recursive halving)

    /// Open polynomial at point z: produces an IPA proof that p(z) = evaluation.
    ///
    /// Algorithm:
    ///   1. Evaluate p(z) via Horner's method
    ///   2. Build evaluation vector u = [1, z, z^2, ...]
    ///   3. Run recursive halving: at each round, compute cross-term MSMs (L, R),
    ///      derive Fiat-Shamir challenge, fold vectors and generators
    ///   4. Output (L[], R[], final_a) proof
    ///
    /// The heavy MSM computations use GPU when the vector is large enough.
    public func open(_ coeffs: [Fr], at z: Fr) throws -> IPAOpeningProof {
        var paddedCoeffs = coeffs
        // Pad to power of 2
        let targetN = nextPowerOf2(coeffs.count)
        while paddedCoeffs.count < targetN {
            paddedCoeffs.append(Fr.zero)
        }
        let n = paddedCoeffs.count
        guard n <= maxDegree else {
            fatalError("Polynomial degree \(n) exceeds maxDegree \(maxDegree)")
        }

        let pz = evaluate(coeffs, at: z)
        let u = buildEvalVector(point: z, length: n)

        let qProj = pointFromAffine(Q)
        let logN = Int(log2(Double(n)))

        // Build commitment for transcript: C = MSM(G, coeffs)
        let commitment = try commit(paddedCoeffs)
        let ip = cFrInnerProduct(paddedCoeffs, u)

        var transcript = GPUIPATranscript()
        transcript.appendLabel("ipa-open")
        transcript.appendPoint(commitment)
        transcript.appendScalar(ip)

        var Ls = [PointProjective]()
        var Rs = [PointProjective]()
        Ls.reserveCapacity(logN)
        Rs.reserveCapacity(logN)

        // Working copies
        var a = paddedCoeffs
        var b = u
        var gens = Array(generators.prefix(n))

        var halfLen = n / 2

        for _ in 0..<logN {
            let aLo = Array(a.prefix(halfLen))
            let aHi = Array(a.suffix(from: halfLen).prefix(halfLen))
            let bLo = Array(b.prefix(halfLen))
            let bHi = Array(b.suffix(from: halfLen).prefix(halfLen))
            let gLo = Array(gens.prefix(halfLen))
            let gHi = Array(gens.suffix(from: halfLen).prefix(halfLen))

            // Cross inner products
            let crossL = cFrInnerProduct(aLo, bHi)
            let crossR = cFrInnerProduct(aHi, bLo)

            // L = MSM(G_hi, a_lo) + crossL * Q
            // R = MSM(G_lo, a_hi) + crossR * Q
            let aLoLimbs = batchFrToLimbs(aLo)
            let aHiLimbs = batchFrToLimbs(aHi)

            let msmL: PointProjective
            let msmR: PointProjective
            if halfLen >= GPUIPAEngine.gpuThreshold, let engine = getMSMEngine() {
                msmL = (try? engine.msm(points: gHi, scalars: aLoLimbs))
                    ?? cPippengerMSM(points: gHi, scalars: aLoLimbs)
                msmR = (try? engine.msm(points: gLo, scalars: aHiLimbs))
                    ?? cPippengerMSM(points: gLo, scalars: aHiLimbs)
            } else {
                msmL = cPippengerMSM(points: gHi, scalars: aLoLimbs)
                msmR = cPippengerMSM(points: gLo, scalars: aHiLimbs)
            }

            let L = pointAdd(msmL, cPointScalarMul(qProj, crossL))
            let R = pointAdd(msmR, cPointScalarMul(qProj, crossR))

            Ls.append(L)
            Rs.append(R)

            // Fiat-Shamir challenge
            transcript.appendPoint(L)
            transcript.appendPoint(R)
            let x = transcript.deriveChallenge()
            let xInv = frInverse(x)

            // Fold a: a'[i] = a_lo[i] * x + a_hi[i] * x_inv
            a = cFrVectorFold(aLo, aHi, x: x, xInv: xInv)

            // Fold b: b'[i] = b_lo[i] * x_inv + b_hi[i] * x
            b = cFrVectorFold(bLo, bHi, x: xInv, xInv: x)

            // Fold generators: G'[i] = x_inv * G_lo[i] + x * G_hi[i]
            gens = foldGenerators(gLo: gLo, gHi: gHi, x: xInv, xInv: x)

            halfLen /= 2
        }

        return IPAOpeningProof(Ls: Ls, Rs: Rs, finalA: a[0], evaluation: pz)
    }

    // MARK: - Verify

    /// Verify an IPA opening proof.
    ///
    /// Given commitment C, point z, and proof (L[], R[], final_a, evaluation):
    ///   1. Reconstruct Fiat-Shamir challenges from transcript
    ///   2. Fold commitment using L_i, R_i and challenges
    ///   3. Compute folded generator G_final and folded b_final
    ///   4. Check: C_folded == final_a * G_final + (final_a * b_final) * Q
    public func verify(commitment: PointProjective, point z: Fr,
                       proof: IPAOpeningProof) -> Bool {
        let n = maxDegree
        let logN = Int(log2(Double(n)))
        guard proof.Ls.count == logN, proof.Rs.count == logN else { return false }

        let qProj = pointFromAffine(Q)
        let u = buildEvalVector(point: z, length: n)
        let ip = proof.evaluation

        // Reconstruct transcript and challenges
        var transcript = GPUIPATranscript()
        transcript.appendLabel("ipa-open")
        transcript.appendPoint(commitment)
        transcript.appendScalar(ip)

        // But wait: the commitment encodes <coeffs, u> not proof.evaluation directly.
        // We need to verify ip = <coeffs, u> matches. The transcript used ip = <padded_coeffs, u>.
        // For the proof to verify, we reconstruct using ip = proof.evaluation since
        // the prover used the true inner product (which equals p(z) for the padded polynomial).

        var challenges = [Fr]()
        var challengeInvs = [Fr]()
        challenges.reserveCapacity(logN)
        challengeInvs.reserveCapacity(logN)

        for round in 0..<logN {
            transcript.appendPoint(proof.Ls[round])
            transcript.appendPoint(proof.Rs[round])
            let x = transcript.deriveChallenge()
            challenges.append(x)
            challengeInvs.append(frInverse(x))
        }

        // Fold commitment: C' = C + ip*Q + sum(x_i^2 * L_i + x_i^{-2} * R_i)
        var Cprime = pointAdd(commitment, cPointScalarMul(qProj, ip))
        for round in 0..<logN {
            let x2 = frMul(challenges[round], challenges[round])
            let xInv2 = frMul(challengeInvs[round], challengeInvs[round])
            let lTerm = cPointScalarMul(proof.Ls[round], x2)
            let rTerm = cPointScalarMul(proof.Rs[round], xInv2)
            Cprime = pointAdd(Cprime, pointAdd(lTerm, rTerm))
        }

        // Compute s[i] = product of challenge factors based on bit decomposition
        var s = [Fr](repeating: Fr.one, count: n)
        for round in 0..<logN {
            let x = challenges[round]
            let xInv = challengeInvs[round]
            for i in 0..<n {
                let bit = (i >> (logN - 1 - round)) & 1
                s[i] = frMul(s[i], bit == 1 ? x : xInv)
            }
        }

        // G_final = MSM(G, s) using Pippenger
        let sLimbs = batchFrToLimbs(s)
        let gFinal: PointProjective
        if n >= GPUIPAEngine.gpuThreshold, let engine = getMSMEngine() {
            gFinal = (try? engine.msm(points: generators, scalars: sLimbs))
                ?? cPippengerMSM(points: generators, scalars: sLimbs)
        } else {
            gFinal = cPippengerMSM(points: generators, scalars: sLimbs)
        }

        // Fold b (= u) using challenges
        var bFolded = u
        var halfLen = n / 2
        for round in 0..<logN {
            let bL = Array(bFolded.prefix(halfLen))
            let bR = Array(bFolded.suffix(from: halfLen).prefix(halfLen))
            bFolded = cFrVectorFold(bL, bR, x: challengeInvs[round], xInv: challenges[round])
            halfLen /= 2
        }
        let bFinal = bFolded[0]

        // Final check: C' == final_a * G_final + (final_a * b_final) * Q
        let aG = cPointScalarMul(gFinal, proof.finalA)
        let ab = frMul(proof.finalA, bFinal)
        let abQ = cPointScalarMul(qProj, ab)
        let expected = pointAdd(aG, abQ)

        return ipaPointsEqual(Cprime, expected)
    }

    // MARK: - Batch Open

    /// Batch open multiple polynomials at the same point z.
    ///
    /// Algorithm:
    ///   1. Evaluate each polynomial at z
    ///   2. Combine polynomials: h(x) = sum_i gamma^i * p_i(x)
    ///   3. Produce a single IPA opening proof for h(x) at z
    ///
    /// Cost: k evaluations (CPU) + 1 IPA proof (GPU MSM per round)
    public func batchOpen(polys: [[Fr]], at z: Fr, gamma: Fr) throws -> BatchIPAOpeningProof {
        guard !polys.isEmpty else {
            fatalError("batchOpen requires at least one polynomial")
        }

        // Evaluate each polynomial
        var evaluations = [Fr]()
        evaluations.reserveCapacity(polys.count)
        for poly in polys {
            evaluations.append(evaluate(poly, at: z))
        }

        // Combine: h(x) = sum_i gamma^i * p_i(x)
        let maxLen = polys.map { $0.count }.max()!
        var combined = [Fr](repeating: Fr.zero, count: maxLen)
        var gammaPow = Fr.one
        for i in 0..<polys.count {
            let poly = polys[i]
            poly.withUnsafeBytes { pBuf in
                combined.withUnsafeMutableBytes { cBuf in
                    withUnsafeBytes(of: gammaPow) { gBuf in
                        bn254_fr_batch_mac_neon(
                            cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            gBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(poly.count))
                    }
                }
            }
            if i < polys.count - 1 {
                gammaPow = frMul(gammaPow, gamma)
            }
        }

        // Single IPA proof for the combined polynomial
        let combinedProof = try open(combined, at: z)

        return BatchIPAOpeningProof(
            evaluations: evaluations,
            combinedProof: combinedProof,
            gamma: gamma,
            point: z
        )
    }

    // MARK: - Batch Verify

    /// Verify a batch IPA opening proof.
    ///
    /// Reconstructs the combined commitment and verifies the combined IPA proof.
    public func batchVerify(commitments: [PointProjective],
                            proof: BatchIPAOpeningProof) -> Bool {
        guard commitments.count == proof.evaluations.count else { return false }
        let k = commitments.count
        guard k > 0 else { return false }

        // Reconstruct combined commitment: C_combined = sum_i gamma^i * C_i
        var combinedC = pointIdentity()
        var gammaPow = Fr.one
        for i in 0..<k {
            combinedC = pointAdd(combinedC, cPointScalarMul(commitments[i], gammaPow))
            if i < k - 1 { gammaPow = frMul(gammaPow, proof.gamma) }
        }

        // Verify combined IPA proof
        return verify(commitment: combinedC, point: proof.point, proof: proof.combinedProof)
    }

    // MARK: - Private Helpers

    /// Fold generator points: G'[i] = x * G_lo[i] + xInv * G_hi[i]
    private func foldGenerators(gLo: [PointAffine], gHi: [PointAffine],
                                x: Fr, xInv: Fr) -> [PointAffine] {
        let n = gLo.count
        var result = [PointProjective]()
        result.reserveCapacity(n)
        for i in 0..<n {
            let left = cPointScalarMul(pointFromAffine(gLo[i]), x)
            let right = cPointScalarMul(pointFromAffine(gHi[i]), xInv)
            result.append(pointAdd(left, right))
        }
        return batchToAffine(result)
    }

    /// Compare two projective points for equality (via affine conversion).
    private func ipaPointsEqual(_ a: PointProjective, _ b: PointProjective) -> Bool {
        if pointIsIdentity(a) && pointIsIdentity(b) { return true }
        if pointIsIdentity(a) || pointIsIdentity(b) { return false }
        let aAff = batchToAffine([a])
        let bAff = batchToAffine([b])
        return fpToInt(aAff[0].x) == fpToInt(bAff[0].x) &&
               fpToInt(aAff[0].y) == fpToInt(bAff[0].y)
    }

    /// Next power of 2 >= n.
    private func nextPowerOf2(_ n: Int) -> Int {
        var v = n - 1
        v |= v >> 1
        v |= v >> 2
        v |= v >> 4
        v |= v >> 8
        v |= v >> 16
        return v + 1
    }
}
