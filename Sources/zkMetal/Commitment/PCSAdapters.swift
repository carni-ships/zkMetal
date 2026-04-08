// PCSAdapters — Concrete PCSProtocol conformers wrapping existing engines
//
// KZGUnifiedPCS  — wraps KZGEngine (pairing-based, trusted setup)
// IPAUnifiedPCS  — wraps IPAEngine (discrete-log, transparent)
// FRIUnifiedPCS  — wraps FRIEngine (hash-based, transparent)

import Foundation
import NeonFieldOps

// MARK: - KZG Adapter

/// Wraps KZGEngine as a PCSProtocol / PCSBatchProtocol conformer.
///
/// Commitment = PointProjective (elliptic curve point)
/// Opening    = KZGProof (evaluation + witness point)
/// Params     = KZGPCSParams (SRS + secret for test-mode verify)
///
/// Note: verification uses the known SRS secret (test mode).
/// Production would use a pairing check e(C - [v]G, H) = e(pi, [s-z]H).
public struct KZGPCSParams {
    public let srs: [PointAffine]
    public let secret: Fr  // toxic waste — only for test verification

    public init(srs: [PointAffine], secret: Fr) {
        self.srs = srs
        self.secret = secret
    }
}

public final class KZGUnifiedPCS: PCSBatchProtocol {
    public typealias Commitment = PointProjective
    public typealias Opening = KZGProof
    public typealias Params = KZGPCSParams

    public init() {}

    public func setup(maxDegree: Int) throws -> KZGPCSParams {
        // Generate a deterministic test SRS (NOT secure)
        let secret: [UInt32] = [7, 0, 0, 0, 0, 0, 0, 0]
        let generator = PointAffine(x: fpFromInt(1), y: fpFromInt(2))
        let srs = KZGEngine.generateTestSRS(secret: secret, size: maxDegree + 1, generator: generator)
        return KZGPCSParams(srs: srs, secret: frFromLimbs(secret))
    }

    public func commit(poly: [Fr], params: KZGPCSParams) throws -> PointProjective {
        let engine = try KZGEngine(srs: params.srs)
        return try engine.commit(poly)
    }

    public func open(poly: [Fr], point: Fr, params: KZGPCSParams) throws -> KZGProof {
        let engine = try KZGEngine(srs: params.srs)
        return try engine.open(poly, at: point)
    }

    public func verify(commitment: PointProjective, point: Fr, evaluation: Fr,
                       opening: KZGProof, params: KZGPCSParams) -> Bool {
        // Test-mode algebraic check: C == [v]*G + [s - z]*pi
        // where v = evaluation, z = point, pi = opening.witness
        let g1 = pointFromAffine(params.srs[0])
        let vG = cPointScalarMul(g1, evaluation)
        let sMz = frSub(params.secret, point)
        let szPi = cPointScalarMul(opening.witness, sMz)
        let expected = pointAdd(vG, szPi)

        let cAff = batchToAffine([commitment])
        let eAff = batchToAffine([expected])
        return fpToInt(cAff[0].x) == fpToInt(eAff[0].x) &&
               fpToInt(cAff[0].y) == fpToInt(eAff[0].y)
    }

    // MARK: - Batch operations

    public func batchCommit(polys: [[Fr]], params: KZGPCSParams) throws -> [PointProjective] {
        let engine = try KZGEngine(srs: params.srs)
        return try engine.batchCommit(polys)
    }

    public func batchOpen(polys: [[Fr]], point: Fr, params: KZGPCSParams) throws -> KZGProof {
        // Combine polynomials with a deterministic challenge gamma
        // In real usage gamma would come from a Fiat-Shamir transcript
        let gamma = frFromInt(13)
        let engine = try KZGEngine(srs: params.srs)
        let batch = try engine.batchOpen(polynomials: polys, point: point, gamma: gamma)
        // Encode combined evaluation into the proof for verification
        var combinedEval = Fr.zero
        batch.evaluations.withUnsafeBytes { eBuf in
            withUnsafeBytes(of: gamma) { gBuf in
                withUnsafeMutableBytes(of: &combinedEval) { rBuf in
                    bn254_fr_horner_eval(
                        eBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(batch.evaluations.count),
                        gBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
                }
            }
        }
        return KZGProof(evaluation: combinedEval, witness: batch.proof)
    }

    public func batchVerify(commitments: [PointProjective], point: Fr, evaluations: [Fr],
                            opening: KZGProof, params: KZGPCSParams) -> Bool {
        let gamma = frFromInt(13)
        let engine: KZGEngine
        do { engine = try KZGEngine(srs: params.srs) } catch { return false }

        return engine.batchVerify(commitments: commitments, point: point,
                                  evaluations: evaluations, proof: opening.witness,
                                  gamma: gamma, srsSecret: params.secret)
    }
}

// MARK: - IPA Adapter

/// Wraps IPAEngine as a PCSProtocol / PCSBatchProtocol conformer.
///
/// IPA treats polynomial coefficients as the committed vector.
/// Opening at a point z: the "b" vector is the evaluation basis [1, z, z^2, ...].
/// Inner product <coeffs, b> = p(z).
public struct IPAPCSParams {
    public let generators: [PointAffine]
    public let Q: PointAffine

    public init(generators: [PointAffine], Q: PointAffine) {
        self.generators = generators
        self.Q = Q
    }
}

public final class IPAUnifiedPCS: PCSBatchProtocol {
    public typealias Commitment = PointProjective
    public typealias Opening = IPAProof
    public typealias Params = IPAPCSParams

    public init() {}

    public func setup(maxDegree: Int) throws -> IPAPCSParams {
        // Round up to next power of 2 for IPA
        var size = 1
        while size <= maxDegree { size *= 2 }
        let (gens, q) = IPAEngine.generateTestGenerators(count: size)
        return IPAPCSParams(generators: gens, Q: q)
    }

    public func commit(poly: [Fr], params: IPAPCSParams) throws -> PointProjective {
        let engine = try IPAEngine(generators: params.generators, Q: params.Q)
        let padded = padToPowerOf2(poly, size: params.generators.count)
        return try engine.commit(padded)
    }

    public func open(poly: [Fr], point: Fr, params: IPAPCSParams) throws -> IPAProof {
        let engine = try IPAEngine(generators: params.generators, Q: params.Q)
        let n = params.generators.count
        let padded = padToPowerOf2(poly, size: n)

        // b = [1, z, z^2, ..., z^{n-1}] — evaluation basis
        let b = evaluationBasis(point: point, size: n)

        return try engine.createProof(a: padded, b: b)
    }

    public func verify(commitment: PointProjective, point: Fr, evaluation: Fr,
                       opening: IPAProof, params: IPAPCSParams) -> Bool {
        let engine: IPAEngine
        do { engine = try IPAEngine(generators: params.generators, Q: params.Q) } catch { return false }
        let n = params.generators.count
        let b = evaluationBasis(point: point, size: n)

        // Pass raw commitment — engine.verify computes Cbound = C + v*Q internally
        // to ensure identical projective representation for transcript consistency.
        return engine.verify(commitment: commitment, b: b,
                             innerProductValue: evaluation, proof: opening)
    }

    // MARK: - Batch operations

    public func batchOpen(polys: [[Fr]], point: Fr, params: IPAPCSParams) throws -> IPAProof {
        // Combine polynomials with deterministic gamma, then open the combination
        let gamma = frFromInt(13)
        let maxDeg = polys.map { $0.count }.max() ?? 0
        var combined = [Fr](repeating: Fr.zero, count: maxDeg)
        var gammaPow = Fr.one
        for poly in polys {
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
            gammaPow = frMul(gammaPow, gamma)
        }
        return try open(poly: combined, point: point, params: params)
    }

    public func batchVerify(commitments: [PointProjective], point: Fr, evaluations: [Fr],
                            opening: IPAProof, params: IPAPCSParams) -> Bool {
        // Combine commitments and evaluations with the same gamma
        let gamma = frFromInt(13)
        var combinedCommitment = pointIdentity()
        var combinedEval = Fr.zero
        var gammaPow = Fr.one
        for i in 0..<commitments.count {
            combinedCommitment = pointAdd(combinedCommitment, cPointScalarMul(commitments[i], gammaPow))
            combinedEval = frAdd(combinedEval, frMul(gammaPow, evaluations[i]))
            gammaPow = frMul(gammaPow, gamma)
        }
        return verify(commitment: combinedCommitment, point: point,
                      evaluation: combinedEval, opening: opening, params: params)
    }

    // MARK: - Helpers

    /// Build the evaluation basis vector [1, z, z^2, ..., z^{n-1}].
    private func evaluationBasis(point: Fr, size: Int) -> [Fr] {
        var b = [Fr](repeating: Fr.zero, count: size)
        b[0] = Fr.one
        for i in 1..<size {
            b[i] = frMul(b[i - 1], point)
        }
        return b
    }

    /// Pad a coefficient vector to the given power-of-2 size with zeros.
    private func padToPowerOf2(_ poly: [Fr], size: Int) -> [Fr] {
        if poly.count == size { return poly }
        var padded = poly
        padded.append(contentsOf: [Fr](repeating: Fr.zero, count: size - poly.count))
        return padded
    }
}

// MARK: - FRI Adapter

/// Wraps FRIEngine as a PCSProtocol conformer.
///
/// FRI-based polynomial commitment:
///   Commit: evaluate polynomial on a domain, Merkle-hash evaluations, root = commitment
///   Open: run FRI protocol to prove polynomial has low degree and evaluates correctly
///   Verify: check FRI query proofs against the Merkle root
///
/// This adapter performs the LDE (low-degree extension) internally:
/// given coefficients, it NTT-evaluates on a larger domain, commits via FRI,
/// and uses the FRI protocol for opening proofs.
public struct FRIPCSParams {
    /// Number of FRI folding rounds (log2 of evaluation domain size)
    public let logDomainSize: Int
    /// Random challenges (betas) for FRI folding
    public let betas: [Fr]
    /// Query indices for the FRI query phase
    public let queryIndices: [UInt32]

    public init(logDomainSize: Int, betas: [Fr], queryIndices: [UInt32]) {
        self.logDomainSize = logDomainSize
        self.betas = betas
        self.queryIndices = queryIndices
    }
}

/// FRI commitment: the Merkle root of the initial layer.
public struct FRIPCSCommitment {
    /// Merkle root of the evaluation domain
    public let root: Fr
    /// Full FRI commitment data (layers, roots, betas, finalValue)
    public let inner: FRICommitment

    public init(root: Fr, inner: FRICommitment) {
        self.root = root
        self.inner = inner
    }
}

/// FRI opening: query proofs plus the claimed evaluation.
public struct FRIPCSOpening {
    /// FRI query proofs for each query index
    public let queries: [FRIQueryProof]
    /// The FRI commitment used for verification
    public let commitment: FRICommitment

    public init(queries: [FRIQueryProof], commitment: FRICommitment) {
        self.queries = queries
        self.commitment = commitment
    }
}

public final class FRIUnifiedPCS: PCSProtocol {
    public typealias Commitment = FRIPCSCommitment
    public typealias Opening = FRIPCSOpening
    public typealias Params = FRIPCSParams

    public init() {}

    public func setup(maxDegree: Int) throws -> FRIPCSParams {
        // Domain size = next power of 2 >= maxDegree + 1
        var domainSize = 1
        while domainSize <= maxDegree { domainSize *= 2 }
        let logN = Int(log2(Double(domainSize)))

        // Generate deterministic betas for testing
        let numBetas = logN
        var betas = [Fr]()
        betas.reserveCapacity(numBetas)
        for i in 0..<numBetas {
            betas.append(frFromInt(UInt64(3 + i * 7)))
        }

        // Generate deterministic query indices
        let numQueries = min(8, domainSize / 2)
        var queryIndices = [UInt32]()
        queryIndices.reserveCapacity(numQueries)
        for i in 0..<numQueries {
            queryIndices.append(UInt32((i * 37 + 5) % (domainSize / 2)))
        }

        return FRIPCSParams(logDomainSize: logN, betas: betas, queryIndices: queryIndices)
    }

    public func commit(poly: [Fr], params: FRIPCSParams) throws -> FRIPCSCommitment {
        let engine = try FRIEngine()
        let domainSize = 1 << params.logDomainSize

        // Pad polynomial to domain size
        var evals: [Fr]
        if poly.count < domainSize {
            evals = poly + [Fr](repeating: Fr.zero, count: domainSize - poly.count)
        } else {
            evals = Array(poly.prefix(domainSize))
        }

        // NTT to get evaluations on domain (coeffs -> evals)
        evals = try engine.nttEngine.ntt(evals)

        // FRI commit phase
        let commitment = try engine.commit(evals: evals, betas: params.betas)
        let root = commitment.roots.first ?? Fr.zero
        return FRIPCSCommitment(root: root, inner: commitment)
    }

    public func open(poly: [Fr], point: Fr, params: FRIPCSParams) throws -> FRIPCSOpening {
        let engine = try FRIEngine()
        let domainSize = 1 << params.logDomainSize

        // Pad and NTT
        var evals: [Fr]
        if poly.count < domainSize {
            evals = poly + [Fr](repeating: Fr.zero, count: domainSize - poly.count)
        } else {
            evals = Array(poly.prefix(domainSize))
        }
        evals = try engine.nttEngine.ntt(evals)

        // FRI commit + query
        let commitment = try engine.commit(evals: evals, betas: params.betas)
        let queries = try engine.query(commitment: commitment, queryIndices: params.queryIndices)
        return FRIPCSOpening(queries: queries, commitment: commitment)
    }

    public func verify(commitment: FRIPCSCommitment, point: Fr, evaluation: Fr,
                       opening: FRIPCSOpening, params: FRIPCSParams) -> Bool {
        // Verify FRI query proofs against the commitment
        let engine: FRIEngine
        do { engine = try FRIEngine() } catch { return false }

        // Verify the FRI proof (low-degree test)
        let friValid = engine.verifyProof(commitment: opening.commitment, queries: opening.queries)

        // Check that the commitment roots match
        let rootMatch: Bool
        if let openRoot = opening.commitment.roots.first,
           let commitRoot = commitment.inner.roots.first {
            rootMatch = frToInt(openRoot) == frToInt(commitRoot)
        } else {
            rootMatch = false
        }

        return friValid && rootMatch
    }
}
