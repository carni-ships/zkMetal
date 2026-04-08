// GPUIPAProverEngine — GPU-accelerated Bulletproofs-style IPA prover
//
// Implements the Inner Product Argument (IPA) prover with GPU-accelerated
// multi-scalar multiplications for L_i, R_i computation at each round.
// This is the recursive halving protocol from Bulletproofs (Bunz et al. 2018)
// with Pedersen vector commitment as the base commitment scheme.
//
// Key features:
//   - Pedersen vector commitment: C = <a, G> + v*H
//   - Recursive halving: log(n) rounds producing L_i, R_i commitments
//   - Final output: scalar pair (a, b) with proof (L[], R[])
//   - GPU-accelerated generator folding and MSM for L_i, R_i
//   - Verification of IPA proofs
//   - Batch IPA proving (multiple proofs in parallel)
//
// References:
//   - Bulletproofs (Bunz et al. 2018) — inner product argument
//   - Halo (Bowe et al. 2019) — recursive proof composition

import Foundation
import Metal
import NeonFieldOps

// MARK: - IPA Prover Proof

/// Proof produced by the Bulletproofs-style IPA prover.
/// Contains log(n) round commitments (L_i, R_i), final scalar pair (a, b),
/// and the initial inner product value.
public struct IPAProverProof {
    /// Left commitments per round (log(n) entries).
    public let Ls: [PointProjective]
    /// Right commitments per round (log(n) entries).
    public let Rs: [PointProjective]
    /// Final scalar a after all folding rounds.
    public let finalA: Fr
    /// Final scalar b after all folding rounds.
    public let finalB: Fr
    /// The inner product <a, b> before folding.
    public let innerProduct: Fr

    public init(Ls: [PointProjective], Rs: [PointProjective],
                finalA: Fr, finalB: Fr, innerProduct: Fr) {
        self.Ls = Ls
        self.Rs = Rs
        self.finalA = finalA
        self.finalB = finalB
        self.innerProduct = innerProduct
    }

    /// Number of rounds in the proof (should equal log2(vector length)).
    public var rounds: Int { Ls.count }
}

// MARK: - Batch IPA Prover Proof

/// Proof for a batch of IPA proofs computed in parallel.
/// Each entry corresponds to one inner product argument.
public struct BatchIPAProverProof {
    /// Individual proofs for each inner product argument.
    public let proofs: [IPAProverProof]
    /// Original commitments for each proof (for verification).
    public let commitments: [PointProjective]
    /// Number of proofs in this batch.
    public var count: Int { proofs.count }

    public init(proofs: [IPAProverProof], commitments: [PointProjective]) {
        self.proofs = proofs
        self.commitments = commitments
    }
}

// MARK: - Pedersen Vector Commitment

/// A Pedersen vector commitment: C = <a, G> + v*H
/// where G is a vector of generators, H is the blinding generator,
/// a is the committed vector, and v is the blinding factor.
public struct PedersenVectorCommitment {
    /// The commitment point.
    public let point: PointProjective
    /// The blinding factor used (needed for opening).
    public let blindingFactor: Fr
    /// Length of the committed vector.
    public let vectorLength: Int

    public init(point: PointProjective, blindingFactor: Fr, vectorLength: Int) {
        self.point = point
        self.blindingFactor = blindingFactor
        self.vectorLength = vectorLength
    }
}

// MARK: - IPA Prover Transcript (Fiat-Shamir)

/// Transcript for Fiat-Shamir challenge derivation in the IPA prover.
/// Uses Blake3 for hashing, appending points and scalars in sequence.
private struct IPAProverTranscript {
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

    mutating func appendUInt64(_ v: UInt64) {
        var val = v
        withUnsafeBytes(of: &val) { buf in
            data.append(contentsOf: buf)
        }
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

// MARK: - IPA Prover Configuration

/// Configuration options for the GPU IPA Prover Engine.
public struct IPAProverConfig {
    /// Minimum vector size to use GPU MSM (below this, use CPU Pippenger).
    public var gpuThreshold: Int
    /// Whether to enable parallel batch proving.
    public var enableBatchParallel: Bool
    /// Maximum number of concurrent batch proofs.
    public var maxBatchConcurrency: Int

    public init(gpuThreshold: Int = 64,
                enableBatchParallel: Bool = true,
                maxBatchConcurrency: Int = 4) {
        self.gpuThreshold = gpuThreshold
        self.enableBatchParallel = enableBatchParallel
        self.maxBatchConcurrency = maxBatchConcurrency
    }

    /// Default configuration for Apple Silicon.
    public static let `default` = IPAProverConfig()

    /// Configuration optimized for small vectors (CPU-only).
    public static let cpuOnly = IPAProverConfig(
        gpuThreshold: Int.max,
        enableBatchParallel: false,
        maxBatchConcurrency: 1
    )
}

// MARK: - GPU IPA Prover Engine

/// GPU-accelerated Bulletproofs-style IPA prover engine.
///
/// Uses Metal GPU MSM for the expensive multi-scalar multiplications
/// in each round of the recursive halving protocol. Falls back to CPU
/// Pippenger for small vector sizes.
///
/// The engine maintains a set of generator points G_0, ..., G_{n-1},
/// a blinding generator H, and an inner product binding generator U.
///
/// Usage:
///   let engine = try GPUIPAProverEngine(maxSize: 256)
///   let commitment = try engine.pedersenCommit(vector: a, blindingFactor: r)
///   let proof = try engine.prove(a: a, b: b, commitment: commitment)
///   let valid = engine.verify(proof: proof, commitment: commitment.point)
public class GPUIPAProverEngine {
    public static let version = Versions.gpuIPAProver

    /// Generator points G_0, ..., G_{n-1} for vector commitments.
    public let generators: [PointAffine]
    /// Blinding generator H (distinct from G_i generators).
    public let H: PointAffine
    /// Inner product binding generator U.
    public let U: PointAffine
    /// Maximum supported vector size (must be power of 2).
    public let maxSize: Int
    /// Engine configuration.
    public let config: IPAProverConfig

    /// Cached GPU MSM engine (lazy-initialized).
    private var _msmEngine: MetalMSM?

    /// Total proofs generated (for diagnostics).
    private var _proofCount: Int = 0

    /// Create GPU IPA prover engine supporting vectors up to the given size.
    /// Size must be a power of 2.
    ///
    /// Generators are derived deterministically from the BN254 G1 generator
    /// via iterated double-add to ensure independence.
    public init(maxSize: Int, config: IPAProverConfig = .default) throws {
        precondition(maxSize > 0 && (maxSize & (maxSize - 1)) == 0,
                     "maxSize must be a power of 2, got \(maxSize)")
        self.maxSize = maxSize
        self.config = config

        // Derive n+2 distinct points via iterated double-add from G1
        let gx = fpFromInt(1)
        let gy = fpFromInt(2)
        let g = pointFromAffine(PointAffine(x: gx, y: gy))

        var projPoints = [PointProjective]()
        projPoints.reserveCapacity(maxSize + 2)
        var acc = g
        for _ in 0..<(maxSize + 2) {
            projPoints.append(acc)
            acc = pointDouble(pointAdd(acc, g))
        }

        let affinePoints = batchToAffine(projPoints)
        self.generators = Array(affinePoints.prefix(maxSize))
        self.H = affinePoints[maxSize]
        self.U = affinePoints[maxSize + 1]
    }

    /// Create GPU IPA prover engine with explicit generators and binding points.
    public init(generators: [PointAffine], H: PointAffine, U: PointAffine,
                config: IPAProverConfig = .default) {
        precondition(generators.count > 0 && (generators.count & (generators.count - 1)) == 0,
                     "Generator count must be a power of 2")
        self.generators = generators
        self.H = H
        self.U = U
        self.maxSize = generators.count
        self.config = config
    }

    /// Number of proofs generated by this engine instance.
    public var proofCount: Int { _proofCount }

    // MARK: - GPU Engine

    private func getMSMEngine() -> MetalMSM? {
        if _msmEngine == nil { _msmEngine = try? MetalMSM() }
        return _msmEngine
    }

    // MARK: - Scalar Conversion

    private func batchFrToLimbs(_ coeffs: [Fr]) -> [[UInt32]] {
        return coeffs.map { frToLimbs($0) }
    }

    // MARK: - MSM Helper

    /// Perform MSM with GPU/CPU selection based on threshold.
    private func msmCompute(points: [PointAffine], scalars: [Fr]) -> PointProjective {
        let n = points.count
        let limbs = batchFrToLimbs(scalars)
        if n >= config.gpuThreshold, let engine = getMSMEngine() {
            return (try? engine.msm(points: points, scalars: limbs))
                ?? cPippengerMSM(points: points, scalars: limbs)
        }
        return cPippengerMSM(points: points, scalars: limbs)
    }

    // MARK: - Pedersen Vector Commitment

    /// Compute Pedersen vector commitment: C = <a, G[0..n]> + blindingFactor * H
    ///
    /// The commitment binds to both the vector a and the blinding factor.
    /// Uses GPU MSM for the vector part when above threshold.
    public func pedersenCommit(vector a: [Fr],
                               blindingFactor r: Fr) throws -> PedersenVectorCommitment {
        let n = a.count
        guard n > 0 else {
            // Commitment to empty vector is just r * H
            let rH = cPointScalarMul(pointFromAffine(H), r)
            return PedersenVectorCommitment(point: rH, blindingFactor: r, vectorLength: 0)
        }
        guard n <= maxSize else {
            fatalError("Vector length \(n) exceeds maxSize \(maxSize)")
        }

        let gens = Array(generators.prefix(n))
        let aG = msmCompute(points: gens, scalars: a)
        let rH = cPointScalarMul(pointFromAffine(H), r)
        let C = pointAdd(aG, rH)

        return PedersenVectorCommitment(point: C, blindingFactor: r, vectorLength: n)
    }

    /// Commit without blinding (r = 0). Useful for testing.
    public func commitUnblinded(vector a: [Fr]) throws -> PedersenVectorCommitment {
        return try pedersenCommit(vector: a, blindingFactor: Fr.zero)
    }

    // MARK: - Inner Product

    /// Compute inner product <a, b> using C helper for NEON acceleration.
    private func innerProduct(_ a: [Fr], _ b: [Fr]) -> Fr {
        precondition(a.count == b.count, "Vector length mismatch")
        return cFrInnerProduct(a, b)
    }

    // MARK: - Prove (Recursive Halving IPA)

    /// Generate an IPA proof for the inner product <a, b>.
    ///
    /// Given vectors a, b and commitment C = <a, G> + r*H:
    ///   1. Compute inner product ip = <a, b>
    ///   2. For each round i = 0..log(n)-1:
    ///      - Split a, b, G into halves (lo, hi)
    ///      - Compute cross terms: cL = <a_lo, b_hi>, cR = <a_hi, b_lo>
    ///      - L_i = MSM(G_hi, a_lo) + cL * U
    ///      - R_i = MSM(G_lo, a_hi) + cR * U
    ///      - Derive challenge x_i via Fiat-Shamir
    ///      - Fold: a' = a_lo * x + a_hi * x^{-1}
    ///      - Fold: b' = b_lo * x^{-1} + b_hi * x
    ///      - Fold generators similarly
    ///   3. Output (L[], R[], final_a, final_b, ip)
    public func prove(a inputA: [Fr], b inputB: [Fr],
                      commitment: PedersenVectorCommitment) throws -> IPAProverProof {
        guard inputA.count == inputB.count else {
            fatalError("Vector lengths must match: a=\(inputA.count), b=\(inputB.count)")
        }

        // Pad to power of 2
        let targetN = nextPowerOf2(inputA.count)
        var a = [Fr](repeating: Fr.zero, count: targetN)
        var b = [Fr](repeating: Fr.zero, count: targetN)
        let frStride = MemoryLayout<Fr>.stride
        a.withUnsafeMutableBytes { dst in
            inputA.withUnsafeBytes { src in
                memcpy(dst.baseAddress!, src.baseAddress!, inputA.count * frStride)
            }
        }
        b.withUnsafeMutableBytes { dst in
            inputB.withUnsafeBytes { src in
                memcpy(dst.baseAddress!, src.baseAddress!, inputB.count * frStride)
            }
        }

        let n = a.count
        guard n <= maxSize else {
            fatalError("Padded vector length \(n) exceeds maxSize \(maxSize)")
        }

        let ip = innerProduct(a, b)
        let uProj = pointFromAffine(U)
        let logN = Int(log2(Double(n)))

        // Initialize transcript
        var transcript = IPAProverTranscript()
        transcript.appendLabel("ipa-prover")
        transcript.appendPoint(commitment.point)
        transcript.appendScalar(ip)
        transcript.appendUInt64(UInt64(n))

        var Ls = [PointProjective]()
        var Rs = [PointProjective]()
        Ls.reserveCapacity(logN)
        Rs.reserveCapacity(logN)

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
            let crossL = innerProduct(aLo, bHi)
            let crossR = innerProduct(aHi, bLo)

            // L = MSM(G_hi, a_lo) + crossL * U
            // R = MSM(G_lo, a_hi) + crossR * U
            let msmL = msmCompute(points: gHi, scalars: aLo)
            let msmR = msmCompute(points: gLo, scalars: aHi)

            let L = pointAdd(msmL, cPointScalarMul(uProj, crossL))
            let R = pointAdd(msmR, cPointScalarMul(uProj, crossR))

            Ls.append(L)
            Rs.append(R)

            // Fiat-Shamir challenge
            transcript.appendPoint(L)
            transcript.appendPoint(R)
            let x = transcript.deriveChallenge()
            let xInv = frInverse(x)

            // Fold a: a'[i] = a_lo[i] * x + a_hi[i] * x^{-1}
            a = cFrVectorFold(aLo, aHi, x: x, xInv: xInv)

            // Fold b: b'[i] = b_lo[i] * x^{-1} + b_hi[i] * x
            b = cFrVectorFold(bLo, bHi, x: xInv, xInv: x)

            // Fold generators: G'[i] = x^{-1} * G_lo[i] + x * G_hi[i]
            gens = foldGenerators(gLo: gLo, gHi: gHi, x: xInv, xInv: x)

            halfLen /= 2
        }

        _proofCount += 1

        return IPAProverProof(
            Ls: Ls, Rs: Rs,
            finalA: a[0], finalB: b[0],
            innerProduct: ip
        )
    }

    /// Convenience: prove with an unblinded commitment.
    public func proveUnblinded(a: [Fr], b: [Fr]) throws -> IPAProverProof {
        let commitment = try commitUnblinded(vector: a)
        return try prove(a: a, b: b, commitment: commitment)
    }

    // MARK: - Verify

    /// Verify an IPA prover proof.
    ///
    /// Given commitment C, vectors b, and proof (L[], R[], final_a, final_b, ip):
    ///   1. Reconstruct Fiat-Shamir challenges from transcript
    ///   2. Fold commitment using L_i, R_i and challenges
    ///   3. Compute folded generator G_final and folded b_final
    ///   4. Check: C_folded == final_a * G_final + (final_a * final_b) * U
    ///   5. Check: final inner product consistency
    public func verify(proof: IPAProverProof,
                       commitment: PointProjective,
                       b inputB: [Fr]) -> Bool {
        // Pad b to power of 2
        let targetN = nextPowerOf2(inputB.count)
        var bVec = [Fr](repeating: Fr.zero, count: targetN)
        bVec.withUnsafeMutableBytes { p in
            inputB.withUnsafeBytes { s in
                memcpy(p.baseAddress!, s.baseAddress!, inputB.count * MemoryLayout<Fr>.stride)
            }
        }

        let n = bVec.count
        guard n <= maxSize else { return false }
        let logN = Int(log2(Double(n)))
        guard proof.Ls.count == logN, proof.Rs.count == logN else { return false }

        let uProj = pointFromAffine(U)

        // Reconstruct transcript and challenges
        var transcript = IPAProverTranscript()
        transcript.appendLabel("ipa-prover")
        transcript.appendPoint(commitment)
        transcript.appendScalar(proof.innerProduct)
        transcript.appendUInt64(UInt64(n))

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

        // Fold commitment: C' = C + ip*U + sum(x_i^2 * L_i + x_i^{-2} * R_i)
        var Cprime = pointAdd(commitment, cPointScalarMul(uProj, proof.innerProduct))
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

        // G_final = MSM(G, s)
        let gFinal = msmCompute(points: Array(generators.prefix(n)), scalars: s)

        // Fold b using challenges
        var bFolded = bVec
        var halfLen = n / 2
        for round in 0..<logN {
            var xi = challengeInvs[round]
            var xiInv = challenges[round]
            bFolded.withUnsafeBytes { bBuf in
                let base = bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                var out = [Fr](repeating: Fr.zero, count: halfLen)
                out.withUnsafeMutableBytes { oBuf in
                    withUnsafeBytes(of: &xi) { xBuf in
                        withUnsafeBytes(of: &xiInv) { xiBuf in
                            bn254_fr_vector_fold(
                                base,
                                base.advanced(by: halfLen * 4),
                                xBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                xiBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                Int32(halfLen),
                                oBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
                        }
                    }
                }
                bFolded = out
            }
            halfLen /= 2
        }
        let bFinal = bFolded[0]

        // Check 1: final_b matches folded b
        guard frEqual(proof.finalB, bFinal) else { return false }

        // Check 2: final inner product consistency
        let expectedIP = frMul(proof.finalA, proof.finalB)
        // The folded inner product should match the committed value
        // through the algebraic relation maintained by the protocol.

        // Check 3: C' == final_a * G_final + (final_a * final_b) * U
        let aG = cPointScalarMul(gFinal, proof.finalA)
        let abU = cPointScalarMul(uProj, expectedIP)
        let expected = pointAdd(aG, abU)

        return ipaProverPointsEqual(Cprime, expected)
    }

    // MARK: - Batch Prove

    /// Generate multiple IPA proofs in parallel.
    ///
    /// Each entry in the batch consists of vectors (a_i, b_i) and a commitment.
    /// Returns a BatchIPAProverProof containing all individual proofs.
    ///
    /// When config.enableBatchParallel is true, proofs are computed concurrently
    /// using DispatchQueue for CPU work (GPU MSMs are serialized by Metal).
    public func batchProve(
        items: [(a: [Fr], b: [Fr], commitment: PedersenVectorCommitment)]
    ) throws -> BatchIPAProverProof {
        guard !items.isEmpty else {
            fatalError("batchProve requires at least one item")
        }

        var proofs = [IPAProverProof?](repeating: nil, count: items.count)
        var commitments = [PointProjective]()
        commitments.reserveCapacity(items.count)
        for item in items {
            commitments.append(item.commitment.point)
        }

        if config.enableBatchParallel && items.count > 1 {
            let lock = NSLock()
            let group = DispatchGroup()
            let queue = DispatchQueue(
                label: "zkMetal.ipaProver.batch",
                attributes: .concurrent
            )

            for idx in 0..<items.count {
                group.enter()
                queue.async { [self] in
                    defer { group.leave() }
                    let item = items[idx]
                    if let proof = try? self.prove(a: item.a, b: item.b,
                                                    commitment: item.commitment) {
                        lock.lock()
                        proofs[idx] = proof
                        lock.unlock()
                    }
                }
            }
            group.wait()
        } else {
            for idx in 0..<items.count {
                let item = items[idx]
                proofs[idx] = try prove(a: item.a, b: item.b, commitment: item.commitment)
            }
        }

        // Verify all proofs were generated
        let finalProofs = proofs.compactMap { $0 }
        guard finalProofs.count == items.count else {
            fatalError("batchProve: \(items.count - finalProofs.count) proofs failed")
        }

        return BatchIPAProverProof(proofs: finalProofs, commitments: commitments)
    }

    /// Convenience: batch prove with unblinded commitments.
    public func batchProveUnblinded(
        items: [(a: [Fr], b: [Fr])]
    ) throws -> BatchIPAProverProof {
        var fullItems = [(a: [Fr], b: [Fr], commitment: PedersenVectorCommitment)]()
        fullItems.reserveCapacity(items.count)
        for item in items {
            let commitment = try commitUnblinded(vector: item.a)
            fullItems.append((a: item.a, b: item.b, commitment: commitment))
        }
        return try batchProve(items: fullItems)
    }

    // MARK: - Batch Verify

    /// Verify a batch of IPA proofs.
    ///
    /// Returns true only if ALL proofs in the batch are valid.
    public func batchVerify(batch: BatchIPAProverProof,
                            bVectors: [[Fr]]) -> Bool {
        guard batch.count == bVectors.count else { return false }
        guard batch.count == batch.commitments.count else { return false }

        for idx in 0..<batch.count {
            let valid = verify(
                proof: batch.proofs[idx],
                commitment: batch.commitments[idx],
                b: bVectors[idx]
            )
            if !valid { return false }
        }
        return true
    }

    // MARK: - Weighted Inner Product Prove

    /// Prove a weighted inner product: <a, b * w> where w is a weight vector.
    ///
    /// This is useful for polynomial evaluation proofs where the weight vector
    /// is [1, z, z^2, ...].
    public func proveWeighted(a: [Fr], b: [Fr], weights w: [Fr],
                              commitment: PedersenVectorCommitment) throws -> IPAProverProof {
        precondition(b.count == w.count, "b and weights must have same length")
        // Compute b' = b * w (element-wise)
        var bWeighted = [Fr]()
        bWeighted.reserveCapacity(b.count)
        for i in 0..<b.count {
            bWeighted.append(frMul(b[i], w[i]))
        }
        return try prove(a: a, b: bWeighted, commitment: commitment)
    }

    // MARK: - Polynomial Evaluation Proof

    /// Generate an IPA proof that p(z) = v, where p is defined by coefficients.
    ///
    /// The evaluation vector is u = [1, z, z^2, ..., z^{n-1}].
    /// This proves <coefficients, u> = v.
    public func proveEvaluation(coeffs: [Fr], at z: Fr) throws -> IPAProverProof {
        let n = coeffs.count
        let u = buildPowerVector(base: z, length: n)
        let commitment = try commitUnblinded(vector: coeffs)
        return try prove(a: coeffs, b: u, commitment: commitment)
    }

    /// Verify a polynomial evaluation proof.
    public func verifyEvaluation(proof: IPAProverProof,
                                 commitment: PointProjective,
                                 at z: Fr, length n: Int) -> Bool {
        let u = buildPowerVector(base: z, length: n)
        return verify(proof: proof, commitment: commitment, b: u)
    }

    // MARK: - Statistics

    /// Reset proof counter.
    public func resetStats() {
        _proofCount = 0
    }

    // MARK: - Private Helpers

    /// Build vector [1, base, base^2, ..., base^{n-1}].
    private func buildPowerVector(base z: Fr, length n: Int) -> [Fr] {
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
    private func ipaProverPointsEqual(_ a: PointProjective,
                                       _ b: PointProjective) -> Bool {
        if pointIsIdentity(a) && pointIsIdentity(b) { return true }
        if pointIsIdentity(a) || pointIsIdentity(b) { return false }
        let aAff = batchToAffine([a])
        let bAff = batchToAffine([b])
        return fpToInt(aAff[0].x) == fpToInt(bAff[0].x) &&
               fpToInt(aAff[0].y) == fpToInt(bAff[0].y)
    }

    /// Next power of 2 >= n.
    private func nextPowerOf2(_ n: Int) -> Int {
        guard n > 0 else { return 1 }
        var v = n - 1
        v |= v >> 1
        v |= v >> 2
        v |= v >> 4
        v |= v >> 8
        v |= v >> 16
        return v + 1
    }

    // MARK: - Debug

    /// Dump engine state for diagnostics.
    public func debugDescription() -> String {
        var lines = [String]()
        lines.append("GPUIPAProverEngine v\(Self.version.version)")
        lines.append("  maxSize: \(maxSize)")
        lines.append("  generators: \(generators.count)")
        lines.append("  gpuThreshold: \(config.gpuThreshold)")
        lines.append("  batchParallel: \(config.enableBatchParallel)")
        lines.append("  proofCount: \(_proofCount)")
        lines.append("  gpuAvailable: \(getMSMEngine() != nil)")
        return lines.joined(separator: "\n")
    }
}

// MARK: - IPA Proof Serialization

extension IPAProverProof {
    /// Approximate byte size of this proof.
    public var approximateByteSize: Int {
        // Each PointProjective is 96 bytes (3 * Fp), each Fr is 32 bytes
        let pointSize = 96
        let scalarSize = 32
        return Ls.count * pointSize + Rs.count * pointSize + 3 * scalarSize
    }

    /// Verify structural integrity (lengths match, rounds consistent).
    public var isWellFormed: Bool {
        guard Ls.count == Rs.count else { return false }
        guard Ls.count > 0 else { return false }
        // rounds should be log2 of some power of 2
        let n = 1 << Ls.count
        return n >= 2 && n <= (1 << 30)
    }
}

// MARK: - Batch Proof Statistics

extension BatchIPAProverProof {
    /// Total number of rounds across all proofs.
    public var totalRounds: Int {
        proofs.reduce(0) { $0 + $1.rounds }
    }

    /// Approximate total byte size of the batch proof.
    public var approximateByteSize: Int {
        proofs.reduce(0) { $0 + $1.approximateByteSize }
    }
}
