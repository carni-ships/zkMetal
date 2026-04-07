// GPUPolyCommitOpenEngine — GPU-accelerated polynomial commitment opening engine
//
// Unified opening engine: KZG-style and IPA-style polynomial commitment openings.
// Single-point, multi-point, batch, linearized batch, and IPA recursive halving.
// All MSM operations go through MetalMSM (GPU) with CPU Pippenger fallback.

import Foundation
import Metal
import NeonFieldOps

// MARK: - Opening Scheme

/// Which polynomial commitment opening scheme to use.
public enum OpeningScheme {
    /// KZG-style: quotient polynomial + SRS commitment
    case kzg
    /// IPA-style: recursive halving with Pedersen vector commitment
    case ipa
}

// MARK: - Opening Proof Types

/// Single-point KZG opening proof: f(z) = y, witness W = [q(s)]_1.
public struct SinglePointOpenProof {
    public let commitment: PointProjective
    public let evaluation: Fr
    public let witness: PointProjective
    public let point: Fr

    public init(commitment: PointProjective, evaluation: Fr,
                witness: PointProjective, point: Fr) {
        self.commitment = commitment
        self.evaluation = evaluation
        self.witness = witness
        self.point = point
    }
}

/// Multi-point opening proof for a single polynomial at k points.
public struct MultiPointOpenProof {
    public let commitment: PointProjective
    public let evaluations: [Fr]
    public let points: [Fr]
    public let witness: PointProjective

    public init(commitment: PointProjective, evaluations: [Fr],
                points: [Fr], witness: PointProjective) {
        self.commitment = commitment
        self.evaluations = evaluations
        self.points = points
        self.witness = witness
    }
}

/// Batch opening proof: N polynomials at their own point sets, combined via RLC.
public struct BatchOpenProof {
    public let commitments: [PointProjective]
    public let evaluations: [[Fr]]
    public let pointSets: [[Fr]]
    public let witness: PointProjective
    public let gamma: Fr

    public init(commitments: [PointProjective], evaluations: [[Fr]],
                pointSets: [[Fr]], witness: PointProjective, gamma: Fr) {
        self.commitments = commitments
        self.evaluations = evaluations
        self.pointSets = pointSets
        self.witness = witness
        self.gamma = gamma
    }
}

/// IPA-style opening proof with recursive halving.
public struct IPAStyleOpenProof {
    public let Ls: [PointProjective]
    public let Rs: [PointProjective]
    public let finalScalar: Fr
    public let evaluation: Fr
    public let point: Fr

    public init(Ls: [PointProjective], Rs: [PointProjective],
                finalScalar: Fr, evaluation: Fr, point: Fr) {
        self.Ls = Ls
        self.Rs = Rs
        self.finalScalar = finalScalar
        self.evaluation = evaluation
        self.point = point
    }

    /// Number of rounds (log2 of polynomial degree)
    public var rounds: Int { Ls.count }
}

/// Linearized batch proof: pre-combined commitment for efficient verification.
public struct LinearizedBatchProof {
    public let linearizedCommitment: PointProjective
    public let combinedEvaluation: Fr
    public let witness: PointProjective
    public let point: Fr
    public let batchSize: Int

    public init(linearizedCommitment: PointProjective, combinedEvaluation: Fr,
                witness: PointProjective, point: Fr, batchSize: Int) {
        self.linearizedCommitment = linearizedCommitment
        self.combinedEvaluation = combinedEvaluation
        self.witness = witness
        self.point = point
        self.batchSize = batchSize
    }
}

// MARK: - Fiat-Shamir Transcript (internal)

private struct PCOTranscript {
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

// MARK: - GPU Poly Commit Open Engine

/// GPU-accelerated polynomial commitment opening engine.
/// Supports KZG and IPA schemes with Metal GPU MSM acceleration.
public class GPUPolyCommitOpenEngine {
    public static let version = Versions.gpuPolyCommitOpen

    public let scheme: OpeningScheme
    public let srs: [PointAffine]
    public let generators: [PointAffine]
    public let bindingQ: PointAffine
    public static let gpuThreshold = 64
    private var _msmEngine: MetalMSM?

    /// Create a KZG-mode engine with the given SRS.
    public init(srs: [PointAffine], scheme: OpeningScheme = .kzg) throws {
        precondition(!srs.isEmpty, "SRS must be non-empty")
        self.scheme = scheme
        self.srs = srs
        // Derive IPA generators from SRS for dual-mode support
        self.generators = Array(srs.prefix(min(srs.count, 256)))
        self.bindingQ = srs.count > 1 ? srs[1] : srs[0]
        self._msmEngine = try? MetalMSM()
    }

    // MARK: - Init (IPA)

    /// Create an IPA-mode engine with explicit generators.
    public init(generators: [PointAffine], Q: PointAffine, scheme: OpeningScheme = .ipa) throws {
        precondition(!generators.isEmpty, "Generators must be non-empty")
        precondition(generators.count & (generators.count - 1) == 0,
                     "Generator count must be a power of 2")
        self.scheme = scheme
        self.generators = generators
        self.bindingQ = Q
        // Use generators as SRS for dual-mode
        self.srs = generators
        self._msmEngine = try? MetalMSM()
    }

    private func getMSMEngine() -> MetalMSM? {
        if _msmEngine == nil { _msmEngine = try? MetalMSM() }
        return _msmEngine
    }

    // MARK: - Scalar Conversion

    private func batchFrToLimbs(_ coeffs: [Fr]) -> [[UInt32]] {
        return coeffs.map { frToLimbs($0) }
    }

    // MARK: - MSM Helper

    /// Compute MSM: sum_i scalars[i] * bases[i], using GPU when beneficial.
    private func computeMSM(bases: [PointAffine], scalars: [Fr]) -> PointProjective {
        let n = scalars.count
        guard n > 0 else { return pointIdentity() }
        let limbs = batchFrToLimbs(scalars)
        let pts = Array(bases.prefix(n))
        if n >= GPUPolyCommitOpenEngine.gpuThreshold, let engine = getMSMEngine() {
            return (try? engine.msm(points: pts, scalars: limbs))
                ?? cPippengerMSM(points: pts, scalars: limbs)
        }
        return cPippengerMSM(points: pts, scalars: limbs)
    }

    // MARK: - Polynomial Evaluation (Horner)

    private func hornerEval(_ coeffs: [Fr], at z: Fr) -> Fr {
        guard !coeffs.isEmpty else { return Fr.zero }
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

    // MARK: - Commit

    /// Commit to polynomial using SRS (KZG) or generators (IPA).
    public func commit(_ coeffs: [Fr]) -> PointProjective {
        let bases = scheme == .kzg ? srs : generators
        return computeMSM(bases: bases, scalars: coeffs)
    }

    // MARK: - Single-Point KZG Opening

    /// Open polynomial f(X) at a single point z via quotient q(X) = (f(X) - f(z)) / (X - z).
    public func openSinglePoint(polynomial: [Fr], at z: Fr) throws -> SinglePointOpenProof {
        guard polynomial.count >= 1 else { throw MSMError.invalidInput }

        let commitment = commit(polynomial)
        let evaluation = hornerEval(polynomial, at: z)

        // Compute quotient via synthetic division: q(X) = (f(X) - y) / (X - z)
        var shifted = polynomial
        shifted[0] = frSub(shifted[0], evaluation)
        let quotient = syntheticDivide(shifted, root: z)

        let witness: PointProjective
        if quotient.isEmpty || quotient.allSatisfy({ isZeroFr($0) }) {
            witness = pointIdentity()
        } else {
            witness = computeMSM(bases: srs, scalars: quotient)
        }

        return SinglePointOpenProof(
            commitment: commitment,
            evaluation: evaluation,
            witness: witness,
            point: z
        )
    }

    // MARK: - Multi-Point Opening

    /// Open polynomial at multiple points. W(X) = (f(X) - I(X)) / Z_S(X).
    public func openMultiPoint(polynomial: [Fr], points: [Fr]) throws -> MultiPointOpenProof {
        let n = polynomial.count
        guard n >= 1, !points.isEmpty else { throw MSMError.invalidInput }
        guard points.count < n else { throw MSMError.invalidInput }

        let commitment = commit(polynomial)

        // Evaluate f at each point
        var evaluations = [Fr]()
        evaluations.reserveCapacity(points.count)
        for z in points {
            evaluations.append(hornerEval(polynomial, at: z))
        }

        // Build vanishing poly Z_S(X) = prod_i (X - z_i)
        let vanishing = buildVanishingPoly(roots: points)

        // Build interpolation polynomial I(X)
        let interp = lagrangeInterpolation(points: points, values: evaluations)

        // Numerator = f(X) - I(X)
        let maxLen = max(polynomial.count, interp.count)
        var numerator = [Fr](repeating: Fr.zero, count: maxLen)
        for i in 0..<polynomial.count { numerator[i] = polynomial[i] }
        let interpCount = interp.count
        numerator.withUnsafeMutableBytes { nBuf in
            interp.withUnsafeBytes { iBuf in
                bn254_fr_batch_sub_neon(
                    nBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    nBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    iBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(interpCount))
            }
        }

        // Exact division: W(X) = numerator / Z_S(X)
        let witnessCoeffs = polyExactDivide(numerator, by: vanishing)

        let witness: PointProjective
        if witnessCoeffs.isEmpty || witnessCoeffs.allSatisfy({ isZeroFr($0) }) {
            witness = pointIdentity()
        } else {
            witness = computeMSM(bases: srs, scalars: witnessCoeffs)
        }

        return MultiPointOpenProof(
            commitment: commitment,
            evaluations: evaluations,
            points: points,
            witness: witness
        )
    }

    // MARK: - Batch Opening

    /// Batch open N polynomials at their respective point sets via Fiat-Shamir RLC.
    public func batchOpen(
        polynomials: [[Fr]],
        pointSets: [[Fr]]
    ) throws -> BatchOpenProof {
        let numPolys = polynomials.count
        guard numPolys > 0, numPolys == pointSets.count else {
            throw MSMError.invalidInput
        }

        // Commit to all polynomials
        var commitments = [PointProjective]()
        commitments.reserveCapacity(numPolys)
        for poly in polynomials {
            commitments.append(commit(poly))
        }

        // Build transcript for Fiat-Shamir
        var transcript = PCOTranscript()
        transcript.appendLabel("pco-batch-open")
        for c in commitments { transcript.appendPoint(c) }

        // Evaluate each polynomial at its points
        var allEvals = [[Fr]]()
        allEvals.reserveCapacity(numPolys)
        for i in 0..<numPolys {
            var evals = [Fr]()
            evals.reserveCapacity(pointSets[i].count)
            for z in pointSets[i] {
                let y = hornerEval(polynomials[i], at: z)
                evals.append(y)
                transcript.appendScalar(y)
            }
            allEvals.append(evals)
        }

        let gamma = transcript.deriveChallenge()

        // Compute individual witness polynomials
        var maxWitnessDeg = 0
        var witnessPolys = [[Fr]]()
        witnessPolys.reserveCapacity(numPolys)

        for i in 0..<numPolys {
            let poly = polynomials[i]
            let pts = pointSets[i]
            let evals = allEvals[i]

            let vanishing = buildVanishingPoly(roots: pts)
            let interp = lagrangeInterpolation(points: pts, values: evals)

            let maxLen = max(poly.count, interp.count)
            var numerator = [Fr](repeating: Fr.zero, count: maxLen)
            for j in 0..<poly.count { numerator[j] = poly[j] }
            let interpCount = interp.count
            numerator.withUnsafeMutableBytes { nBuf in
                interp.withUnsafeBytes { iBuf in
                    bn254_fr_batch_sub_neon(
                        nBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        nBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        iBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(interpCount))
                }
            }

            let wi = polyExactDivide(numerator, by: vanishing)
            witnessPolys.append(wi)
            if wi.count > maxWitnessDeg { maxWitnessDeg = wi.count }
        }

        // Combine: W(X) = sum_i gamma^i * W_i(X)
        var combined = [Fr](repeating: Fr.zero, count: max(maxWitnessDeg, 1))
        var gammaPow = Fr.one
        for i in 0..<numPolys {
            let wi = witnessPolys[i]
            for j in 0..<wi.count {
                combined[j] = frAdd(combined[j], frMul(gammaPow, wi[j]))
            }
            if i < numPolys - 1 { gammaPow = frMul(gammaPow, gamma) }
        }

        let witness: PointProjective
        if combined.allSatisfy({ isZeroFr($0) }) {
            witness = pointIdentity()
        } else {
            witness = computeMSM(bases: srs, scalars: combined)
        }

        return BatchOpenProof(
            commitments: commitments,
            evaluations: allEvals,
            pointSets: pointSets,
            witness: witness,
            gamma: gamma
        )
    }

    // MARK: - Linearized Batch Opening

    /// Linearized batch opening: pre-combines commitments for efficient verification.
    public func linearizedBatchOpen(
        polynomials: [[Fr]],
        at z: Fr
    ) throws -> LinearizedBatchProof {
        let numPolys = polynomials.count
        guard numPolys > 0 else { throw MSMError.invalidInput }

        // Commit + evaluate
        var commitments = [PointProjective]()
        var evaluations = [Fr]()
        commitments.reserveCapacity(numPolys)
        evaluations.reserveCapacity(numPolys)
        for poly in polynomials {
            commitments.append(commit(poly))
            evaluations.append(hornerEval(poly, at: z))
        }

        // Fiat-Shamir challenge
        var transcript = PCOTranscript()
        transcript.appendLabel("pco-linearized")
        for c in commitments { transcript.appendPoint(c) }
        for e in evaluations { transcript.appendScalar(e) }
        let gamma = transcript.deriveChallenge()

        // Linearize commitment: C_lin = sum_i gamma^i * C_i
        var linC = pointIdentity()
        var gammaPow = Fr.one
        for i in 0..<numPolys {
            linC = pointAdd(linC, cPointScalarMul(commitments[i], gammaPow))
            if i < numPolys - 1 { gammaPow = frMul(gammaPow, gamma) }
        }

        // Linearize evaluation
        var linEval = Fr.zero
        gammaPow = Fr.one
        for i in 0..<numPolys {
            linEval = frAdd(linEval, frMul(gammaPow, evaluations[i]))
            if i < numPolys - 1 { gammaPow = frMul(gammaPow, gamma) }
        }

        // Combined quotient: q(X) = sum_i gamma^i * (f_i(X) - f_i(z)) / (X - z)
        let maxLen = polynomials.map { $0.count }.max()!
        var combinedQuotient = [Fr](repeating: Fr.zero, count: max(maxLen - 1, 1))
        gammaPow = Fr.one
        for i in 0..<numPolys {
            var shifted = polynomials[i]
            shifted[0] = frSub(shifted[0], evaluations[i])
            let qi = syntheticDivide(shifted, root: z)
            for j in 0..<qi.count {
                if j < combinedQuotient.count {
                    combinedQuotient[j] = frAdd(combinedQuotient[j], frMul(gammaPow, qi[j]))
                }
            }
            if i < numPolys - 1 { gammaPow = frMul(gammaPow, gamma) }
        }

        let witness: PointProjective
        if combinedQuotient.allSatisfy({ isZeroFr($0) }) {
            witness = pointIdentity()
        } else {
            witness = computeMSM(bases: srs, scalars: combinedQuotient)
        }

        return LinearizedBatchProof(
            linearizedCommitment: linC,
            combinedEvaluation: linEval,
            witness: witness,
            point: z,
            batchSize: numPolys
        )
    }

    // MARK: - IPA-Style Opening

    /// Open polynomial at point z using IPA recursive halving. Requires power-of-2 generators.
    public func openIPA(polynomial: [Fr], at z: Fr) throws -> IPAStyleOpenProof {
        let targetN = nextPowerOf2(polynomial.count)
        guard targetN <= generators.count else { throw MSMError.invalidInput }

        var padded = polynomial
        while padded.count < targetN { padded.append(Fr.zero) }
        let n = padded.count
        let logN = Int(log2(Double(n)))

        let evaluation = hornerEval(polynomial, at: z)
        let evalVec = buildEvalVector(point: z, length: n)
        let qProj = pointFromAffine(bindingQ)

        let commitment = computeMSM(bases: generators, scalars: padded)
        let ip = cFrInnerProduct(padded, evalVec)

        var transcript = PCOTranscript()
        transcript.appendLabel("pco-ipa-open")
        transcript.appendPoint(commitment)
        transcript.appendScalar(ip)

        var Ls = [PointProjective]()
        var Rs = [PointProjective]()
        Ls.reserveCapacity(logN)
        Rs.reserveCapacity(logN)

        var a = padded
        var b = evalVec
        var gens = Array(generators.prefix(n))
        var halfLen = n / 2

        for _ in 0..<logN {
            let aLo = Array(a.prefix(halfLen))
            let aHi = Array(a.suffix(from: halfLen).prefix(halfLen))
            let bLo = Array(b.prefix(halfLen))
            let bHi = Array(b.suffix(from: halfLen).prefix(halfLen))
            let gLo = Array(gens.prefix(halfLen))
            let gHi = Array(gens.suffix(from: halfLen).prefix(halfLen))

            let crossL = cFrInnerProduct(aLo, bHi)
            let crossR = cFrInnerProduct(aHi, bLo)

            let msmL = computeMSM(bases: gHi, scalars: aLo)
            let msmR = computeMSM(bases: gLo, scalars: aHi)

            let L = pointAdd(msmL, cPointScalarMul(qProj, crossL))
            let R = pointAdd(msmR, cPointScalarMul(qProj, crossR))

            Ls.append(L)
            Rs.append(R)

            transcript.appendPoint(L)
            transcript.appendPoint(R)
            let x = transcript.deriveChallenge()
            let xInv = frInverse(x)

            a = cFrVectorFold(aLo, aHi, x: x, xInv: xInv)
            b = cFrVectorFold(bLo, bHi, x: xInv, xInv: x)
            gens = foldGenerators(gLo: gLo, gHi: gHi, x: xInv, xInv: x)

            halfLen /= 2
        }

        return IPAStyleOpenProof(
            Ls: Ls, Rs: Rs,
            finalScalar: a[0],
            evaluation: evaluation,
            point: z
        )
    }

    // MARK: - Verification: Single Point

    /// Verify single-point proof: C - y*G == (s - z) * W.
    public func verifySinglePoint(proof: SinglePointOpenProof, srsSecret: Fr) -> Bool {
        let g1 = pointFromAffine(srs[0])

        // LHS = C - y * G
        let yG = cPointScalarMul(g1, proof.evaluation)
        let lhs = pointAdd(proof.commitment, pointNeg(yG))

        // RHS = (s - z) * W
        let sMinusZ = frSub(srsSecret, proof.point)
        let rhs = cPointScalarMul(proof.witness, sMinusZ)

        return pcoPointsEqual(lhs, rhs)
    }

    // MARK: - Verification: Multi-Point

    /// Verify multi-point proof: [f(s)] - [I(s)] == [Z_S(s)] * [W(s)].
    public func verifyMultiPoint(proof: MultiPointOpenProof, srsSecret: Fr) -> Bool {
        guard proof.evaluations.count == proof.points.count else { return false }

        let g1 = pointFromAffine(srs[0])

        // Z_S(s) = prod_i (s - z_i)
        var zsAtS = Fr.one
        for z in proof.points {
            zsAtS = frMul(zsAtS, frSub(srsSecret, z))
        }

        // I(s) via Lagrange interpolation at secret
        let interp = lagrangeInterpolation(points: proof.points, values: proof.evaluations)
        let iAtS = hornerEval(interp, at: srsSecret)

        // LHS = C - I(s)*G
        let iG = cPointScalarMul(g1, iAtS)
        let lhs = pointAdd(proof.commitment, pointNeg(iG))

        // RHS = Z_S(s) * W
        let rhs = cPointScalarMul(proof.witness, zsAtS)

        return pcoPointsEqual(lhs, rhs)
    }

    // MARK: - Verification: Batch

    /// Verify batch proof using the SRS secret.
    public func verifyBatch(proof: BatchOpenProof, srsSecret: Fr) -> Bool {
        let numPolys = proof.commitments.count
        guard numPolys == proof.evaluations.count,
              numPolys == proof.pointSets.count else { return false }

        // Reconstruct gamma from transcript
        var transcript = PCOTranscript()
        transcript.appendLabel("pco-batch-open")
        for c in proof.commitments { transcript.appendPoint(c) }
        for i in 0..<numPolys {
            for eval in proof.evaluations[i] {
                transcript.appendScalar(eval)
            }
        }
        let gamma = transcript.deriveChallenge()

        let g1 = pointFromAffine(srs[0])

        // Verify: W(s) == sum_i gamma^i * (C_i - I_i(s)*G) / Z_{S_i}(s)
        var expectedW = pointIdentity()
        var gammaPow = Fr.one

        for i in 0..<numPolys {
            let pts = proof.pointSets[i]
            let evals = proof.evaluations[i]

            var zsAtS = Fr.one
            for z in pts {
                zsAtS = frMul(zsAtS, frSub(srsSecret, z))
            }
            let zsInv = frInverse(zsAtS)

            let interp = lagrangeInterpolation(points: pts, values: evals)
            let iAtS = hornerEval(interp, at: srsSecret)

            let iG = cPointScalarMul(g1, iAtS)
            let numerator = pointAdd(proof.commitments[i], pointNeg(iG))
            let wiPoint = cPointScalarMul(numerator, zsInv)

            expectedW = pointAdd(expectedW, cPointScalarMul(wiPoint, gammaPow))
            if i < numPolys - 1 { gammaPow = frMul(gammaPow, gamma) }
        }

        return pcoPointsEqual(proof.witness, expectedW)
    }

    // MARK: - Verification: Linearized Batch

    /// Verify linearized batch proof: C_lin - y_lin * G == (s - z) * W.
    public func verifyLinearizedBatch(proof: LinearizedBatchProof, srsSecret: Fr) -> Bool {
        let g1 = pointFromAffine(srs[0])

        let yG = cPointScalarMul(g1, proof.combinedEvaluation)
        let lhs = pointAdd(proof.linearizedCommitment, pointNeg(yG))

        let sMinusZ = frSub(srsSecret, proof.point)
        let rhs = cPointScalarMul(proof.witness, sMinusZ)

        return pcoPointsEqual(lhs, rhs)
    }

    // MARK: - Verification: IPA

    /// Verify an IPA-style opening proof.
    public func verifyIPA(commitment: PointProjective, proof: IPAStyleOpenProof) -> Bool {
        let n = generators.count
        let logN = Int(log2(Double(n)))
        guard proof.Ls.count == logN, proof.Rs.count == logN else { return false }

        let qProj = pointFromAffine(bindingQ)
        let evalVec = buildEvalVector(point: proof.point, length: n)
        let ip = proof.evaluation

        var transcript = PCOTranscript()
        transcript.appendLabel("pco-ipa-open")
        transcript.appendPoint(commitment)
        transcript.appendScalar(ip)

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

        // Fold commitment
        var Cprime = pointAdd(commitment, cPointScalarMul(qProj, ip))
        for round in 0..<logN {
            let x2 = frMul(challenges[round], challenges[round])
            let xInv2 = frMul(challengeInvs[round], challengeInvs[round])
            let lTerm = cPointScalarMul(proof.Ls[round], x2)
            let rTerm = cPointScalarMul(proof.Rs[round], xInv2)
            Cprime = pointAdd(Cprime, pointAdd(lTerm, rTerm))
        }

        // Compute s[i] factors from bit decomposition
        var s = [Fr](repeating: Fr.one, count: n)
        for round in 0..<logN {
            let x = challenges[round]
            let xInv = challengeInvs[round]
            for i in 0..<n {
                let bit = (i >> (logN - 1 - round)) & 1
                s[i] = frMul(s[i], bit == 1 ? x : xInv)
            }
        }

        let gFinal = computeMSM(bases: generators, scalars: s)

        // Fold b (evaluation vector)
        var bFolded = evalVec
        var halfLen = n / 2
        for round in 0..<logN {
            let bL = Array(bFolded.prefix(halfLen))
            let bR = Array(bFolded.suffix(from: halfLen).prefix(halfLen))
            bFolded = cFrVectorFold(bL, bR, x: challengeInvs[round], xInv: challenges[round])
            halfLen /= 2
        }
        let bFinal = bFolded[0]

        // Final check: C' == final_a * G_final + (final_a * b_final) * Q
        let aG = cPointScalarMul(gFinal, proof.finalScalar)
        let ab = frMul(proof.finalScalar, bFinal)
        let abQ = cPointScalarMul(qProj, ab)
        let expected = pointAdd(aG, abQ)

        return pcoPointsEqual(Cprime, expected)
    }

    // MARK: - Polynomial Helpers

    /// Synthetic division: poly / (X - root), assuming exact.
    private func syntheticDivide(_ poly: [Fr], root: Fr) -> [Fr] {
        let n = poly.count
        if n < 2 { return [] }
        var quotient = [Fr](repeating: Fr.zero, count: n - 1)
        // Process from highest degree down
        quotient[n - 2] = poly[n - 1]
        for i in stride(from: n - 3, through: 0, by: -1) {
            quotient[i] = frAdd(poly[i + 1], frMul(root, quotient[i + 1]))
        }
        return quotient
    }

    /// Build vanishing polynomial Z_S(X) = prod_i (X - z_i).
    private func buildVanishingPoly(roots: [Fr]) -> [Fr] {
        var result: [Fr] = [Fr.one]
        for root in roots {
            var newResult = [Fr](repeating: Fr.zero, count: result.count + 1)
            for i in 0..<result.count {
                newResult[i + 1] = frAdd(newResult[i + 1], result[i])
                newResult[i] = frSub(newResult[i], frMul(root, result[i]))
            }
            result = newResult
        }
        return result
    }

    /// Lagrange interpolation: given (x_i, y_i), return polynomial coefficients.
    private func lagrangeInterpolation(points: [Fr], values: [Fr]) -> [Fr] {
        let k = points.count
        guard k > 0 else { return [] }
        if k == 1 { return [values[0]] }

        // Precompute all Lagrange denominators and batch-invert
        var denoms = [Fr](repeating: Fr.one, count: k)
        for i in 0..<k {
            for j in 0..<k where j != i {
                denoms[i] = frMul(denoms[i], frSub(points[i], points[j]))
            }
        }
        var dPfx = [Fr](repeating: Fr.one, count: k)
        for i in 1..<k { dPfx[i] = frMul(dPfx[i - 1], denoms[i - 1]) }
        var dAcc = frInverse(frMul(dPfx[k - 1], denoms[k - 1]))
        var denomInvs = [Fr](repeating: Fr.zero, count: k)
        for i in Swift.stride(from: k - 1, through: 0, by: -1) {
            denomInvs[i] = frMul(dAcc, dPfx[i])
            dAcc = frMul(dAcc, denoms[i])
        }

        var result = [Fr](repeating: Fr.zero, count: k)
        for i in 0..<k {
            let scalar = frMul(values[i], denomInvs[i])

            var basis: [Fr] = [Fr.one]
            for j in 0..<k {
                if j != i {
                    var newBasis = [Fr](repeating: Fr.zero, count: basis.count + 1)
                    for m in 0..<basis.count {
                        newBasis[m + 1] = frAdd(newBasis[m + 1], basis[m])
                        newBasis[m] = frSub(newBasis[m], frMul(points[j], basis[m]))
                    }
                    basis = newBasis
                }
            }

            for m in 0..<basis.count {
                if m < result.count {
                    result[m] = frAdd(result[m], frMul(scalar, basis[m]))
                }
            }
        }
        return result
    }

    /// Exact polynomial division (long division).
    private func polyExactDivide(_ numerator: [Fr], by divisor: [Fr]) -> [Fr] {
        let nDeg = polyDegree(numerator)
        let dDeg = polyDegree(divisor)
        if nDeg < dDeg { return [] }
        if dDeg < 0 { return [] }

        var rem = Array(numerator)
        let qLen = nDeg - dDeg + 1
        var quotient = [Fr](repeating: Fr.zero, count: qLen)
        let leadInv = frInverse(divisor[dDeg])

        for i in stride(from: nDeg, through: dDeg, by: -1) {
            if isZeroFr(rem[i]) { continue }
            let coeff = frMul(rem[i], leadInv)
            let qIdx = i - dDeg
            quotient[qIdx] = coeff
            for j in 0...dDeg {
                rem[qIdx + j] = frSub(rem[qIdx + j], frMul(coeff, divisor[j]))
            }
        }
        return quotient
    }

    private func polyDegree(_ p: [Fr]) -> Int {
        for i in stride(from: p.count - 1, through: 0, by: -1) {
            if !isZeroFr(p[i]) { return i }
        }
        return -1
    }

    private func isZeroFr(_ a: Fr) -> Bool {
        return frToInt(a) == frToInt(Fr.zero)
    }

    /// Build evaluation vector [1, z, z^2, ..., z^{n-1}].
    private func buildEvalVector(point z: Fr, length n: Int) -> [Fr] {
        var u = [Fr](repeating: Fr.zero, count: n)
        u[0] = Fr.one
        if n > 1 {
            u[1] = z
            for i in 2..<n { u[i] = frMul(u[i - 1], z) }
        }
        return u
    }

    /// Fold generator points for IPA halving.
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

    private func pcoPointsEqual(_ a: PointProjective, _ b: PointProjective) -> Bool {
        if pointIsIdentity(a) && pointIsIdentity(b) { return true }
        if pointIsIdentity(a) || pointIsIdentity(b) { return false }
        let aAff = batchToAffine([a])
        let bAff = batchToAffine([b])
        return fpToInt(aAff[0].x) == fpToInt(bAff[0].x) &&
               fpToInt(aAff[0].y) == fpToInt(bAff[0].y)
    }

    private func nextPowerOf2(_ n: Int) -> Int {
        var v = n - 1
        v |= v >> 1; v |= v >> 2; v |= v >> 4
        v |= v >> 8; v |= v >> 16
        return v + 1
    }
}
