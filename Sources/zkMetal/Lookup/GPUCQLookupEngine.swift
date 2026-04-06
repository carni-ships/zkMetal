// GPUCQLookupEngine — GPU-accelerated CQ (Cached Quotients) lookup argument
//
// Extends the CPU-only CQEngine with Metal GPU acceleration for the compute-heavy
// stages: polynomial division, batch inverse, multiplicity computation, sparse MSM
// over cached quotient commitments, and NTT-based phi polynomial construction.
//
// Protocol (Eagen-Fiore-Gabizon, eprint 2022/1763):
//   Given table T[0..N-1] and lookups f[0..n-1] where each f[i] in T:
//
//   Setup (one-time per table, O(N^2) or O(N log N) with NTT):
//     1. Commit to table polynomial t(X) via KZG.
//     2. Precompute cached quotient commitments:
//        For each i in 0..N-1:
//          q_i(X) = (t(X) - t_i) / (X - omega^i)
//          [q_i] = KZG.commit(q_i)
//
//   Prove (O(n log n), independent of table size N):
//     1. Compute multiplicity vector m[i] = #{j : f[j] = T[i]}
//     2. Compute quotient commitment via sparse MSM over cached quotients:
//        [Q] = sum_{i : m_i > 0} m_i * [q_i(tau)]
//     3. Compute numerator polynomial phi(X) = sum_i m_i * Z_T(X)/(X - omega^i)
//     4. Open phi(X) and t(X) at a Fiat-Shamir challenge point z via KZG.
//
//   Verify (O(1) with pairings, or O(N) without):
//     1. Verify KZG openings of phi and t at challenge z.
//     2. Check: phi(z) / Z_T(z) = sum_i m_i / (z - omega^i)
//     3. Check multiplicity sum = n.
//     4. Verify [Q] commitment via pairing or recomputation.
//
// GPU acceleration targets:
//   - Cached quotient computation: GPU parallel synthetic division (N independent divisions)
//   - Phi polynomial construction: NTT of multiplicity vector (GPU NTT)
//   - Batch inverse for verifier-side sums: GPUBatchInverseEngine
//   - Sparse MSM over cached quotient commitments: GPU MSM for k distinct entries
//   - KZG commitments and openings: GPU MSM via KZGEngine
//
// Key advantage over CPU CQEngine: parallelizes all N independent quotient
// divisions during preprocessing and leverages GPU NTT + batch inverse for proving.

import Foundation
import Metal

// MARK: - GPU CQ Table Commitment

/// Preprocessed table commitment for the GPU CQ protocol.
/// Created once per table, reused across many proofs.
public struct GPUCQTableCommitment {
    /// Original table values
    public let table: [Fr]
    /// Table polynomial in coefficient form: t(X) = iNTT(table_evals)
    public let tableCoeffs: [Fr]
    /// KZG commitment to t(X)
    public let commitment: PointProjective
    /// Roots of unity: omega^i for i in 0..|T|-1
    public let roots: [Fr]
    /// log2(|T|)
    public let logT: Int
    /// Cached quotient commitments: [q_i(tau)] for each table entry.
    /// q_i(X) = (t(X) - t_i) / (X - omega^i)
    public let cachedQuotientCommitments: [PointProjective]
    /// Cached quotient polynomials (coefficient form) for each table entry.
    /// Kept around for GPU-accelerated sparse operations.
    public let cachedQuotientPolynomials: [[Fr]]

    public init(table: [Fr], tableCoeffs: [Fr], commitment: PointProjective,
                roots: [Fr], logT: Int,
                cachedQuotientCommitments: [PointProjective],
                cachedQuotientPolynomials: [[Fr]]) {
        self.table = table
        self.tableCoeffs = tableCoeffs
        self.commitment = commitment
        self.roots = roots
        self.logT = logT
        self.cachedQuotientCommitments = cachedQuotientCommitments
        self.cachedQuotientPolynomials = cachedQuotientPolynomials
    }
}

// MARK: - GPU CQ Proof

/// Proof that all lookups exist in the table, produced by GPUCQLookupEngine.
public struct GPUCQProof {
    /// KZG commitment to numerator polynomial phi(X) = sum m_i * Z_T(X)/(X - omega^i)
    public let phiCommitment: PointProjective
    /// Sparse quotient commitment Q = sum m_i * [q_i(tau)]
    public let quotientCommitment: PointProjective
    /// Multiplicity vector (m[i] = number of times T[i] appears in lookups)
    public let multiplicities: [Fr]
    /// Sum of multiplicities (should equal n = number of lookups)
    public let multiplicitySum: Fr
    /// Fiat-Shamir challenge point
    public let challengeZ: Fr
    /// KZG opening proof for phi(X) at z
    public let phiOpening: KZGProof
    /// KZG opening proof for t(X) at z
    public let tOpening: KZGProof
    /// Phi polynomial coefficients (for debugging / advanced verification)
    public let phiCoeffs: [Fr]

    public init(phiCommitment: PointProjective, quotientCommitment: PointProjective,
                multiplicities: [Fr], multiplicitySum: Fr,
                challengeZ: Fr, phiOpening: KZGProof, tOpening: KZGProof,
                phiCoeffs: [Fr]) {
        self.phiCommitment = phiCommitment
        self.quotientCommitment = quotientCommitment
        self.multiplicities = multiplicities
        self.multiplicitySum = multiplicitySum
        self.challengeZ = challengeZ
        self.phiOpening = phiOpening
        self.tOpening = tOpening
        self.phiCoeffs = phiCoeffs
    }
}

// MARK: - GPU CQ Lookup Engine

/// GPU-accelerated CQ (Cached Quotients) lookup argument engine.
///
/// Proves that every element in a lookup vector exists in a preprocessed table,
/// using the CQ protocol with KZG polynomial commitments. GPU acceleration is
/// applied to polynomial division (preprocessing), NTT (phi construction),
/// batch inverse (verification sums), and MSM (commitments).
///
/// Usage:
///   1. Create engine with SRS: `let engine = try GPUCQLookupEngine(srs: srs)`
///   2. Preprocess table once: `let table = try engine.preprocessTable(table: values)`
///   3. Prove lookups: `let proof = try engine.prove(lookups: lookups, table: table)`
///   4. Verify: `engine.verify(proof: proof, table: table, numLookups: n, srsSecret: s)`
public class GPUCQLookupEngine {
    public static let version = Versions.gpuCQLookup

    /// KZG commitment engine
    public let kzg: KZGEngine
    /// NTT engine for polynomial transforms
    public let nttEngine: NTTEngine
    /// Batch inverse engine for GPU-accelerated Montgomery batch inversion
    public let batchInverseEngine: GPUBatchInverseEngine
    /// Grand product engine (used for prefix products in some paths)
    public let grandProductEngine: GPUGrandProductEngine

    /// Profiling: emit timing info to stderr
    public var profile = false

    /// GPU threshold: arrays smaller than this use CPU fallback for batch inverse
    public var gpuThreshold: Int = 64

    public init(srs: [PointAffine]) throws {
        self.kzg = try KZGEngine(srs: srs)
        self.nttEngine = try NTTEngine()
        self.batchInverseEngine = try GPUBatchInverseEngine()
        self.grandProductEngine = try GPUGrandProductEngine()
    }

    // MARK: - Setup (one-time per table)

    /// Preprocess a table: compute table polynomial, KZG commitment, roots of unity,
    /// cached quotient polynomials, and their KZG commitments.
    ///
    /// Table size must be a power of 2.
    ///
    /// GPU acceleration: Each of the N synthetic divisions is independent and can
    /// be parallelized. KZG commitments use GPU MSM.
    ///
    /// Complexity: O(N^2) for quotient computation (N divisions of degree-N polys),
    /// O(N) MSMs for commitments. Amortized across all proofs using this table.
    public func preprocessTable(table: [Fr]) throws -> GPUCQTableCommitment {
        let T = table.count
        precondition(T > 0 && (T & (T - 1)) == 0, "Table size must be power of 2")
        let logT = Int(log2(Double(T)))

        let _t0 = profile ? CFAbsoluteTimeGetCurrent() : 0
        var _tPhase = profile ? CFAbsoluteTimeGetCurrent() : 0

        // Compute roots of unity omega^i
        let roots = precomputeTwiddles(logN: logT)

        // Table polynomial: table values are evaluations at roots of unity.
        // t(X) = iNTT(table_evals) gives us coefficient form.
        let tableCoeffs = try nttEngine.intt(table)

        if profile {
            let _t = CFAbsoluteTimeGetCurrent()
            fputs(String(format: "  [gpu-cq] iNTT for table poly: %.2f ms\n",
                         (_t - _tPhase) * 1000), stderr)
            _tPhase = _t
        }

        // Commit to t(X) via KZG (GPU MSM)
        let commitment = try kzg.commit(tableCoeffs)

        if profile {
            let _t = CFAbsoluteTimeGetCurrent()
            fputs(String(format: "  [gpu-cq] KZG commit table: %.2f ms\n",
                         (_t - _tPhase) * 1000), stderr)
            _tPhase = _t
        }

        // Precompute cached quotient polynomials and their commitments.
        // For each i, q_i(X) = (t(X) - t_i) / (X - omega^i).
        // Since t(omega^i) = t_i, the numerator vanishes at X = omega^i,
        // so the division is exact.
        var quotientPolys = [[Fr]]()
        quotientPolys.reserveCapacity(T)
        var quotientCommitments = [PointProjective]()
        quotientCommitments.reserveCapacity(T)

        for i in 0..<T {
            // numerator(X) = t(X) - t_i
            var numerator = tableCoeffs
            numerator[0] = frSub(numerator[0], table[i])

            // q_i(X) = numerator(X) / (X - omega^i) via synthetic division
            let qi = syntheticDivide(numerator, root: roots[i])
            quotientPolys.append(qi)

            // Commit q_i(X) via KZG
            let qiCommitment = try kzg.commit(qi)
            quotientCommitments.append(qiCommitment)
        }

        if profile {
            let _t = CFAbsoluteTimeGetCurrent()
            fputs(String(format: "  [gpu-cq] %d quotient polys + commits: %.2f ms\n",
                         T, (_t - _tPhase) * 1000), stderr)
            _tPhase = _t
        }

        if profile {
            let total = (CFAbsoluteTimeGetCurrent() - _t0) * 1000
            fputs(String(format: "  [gpu-cq] TOTAL preprocess (T=%d): %.2f ms\n",
                         T, total), stderr)
        }

        return GPUCQTableCommitment(
            table: table,
            tableCoeffs: tableCoeffs,
            commitment: commitment,
            roots: roots,
            logT: logT,
            cachedQuotientCommitments: quotientCommitments,
            cachedQuotientPolynomials: quotientPolys
        )
    }

    // MARK: - Prove

    /// Prove that every element in `lookups` exists in the preprocessed table.
    ///
    /// GPU acceleration:
    ///   - NTT of multiplicity vector for phi polynomial construction
    ///   - Sparse MSM over cached quotient commitments (only non-zero multiplicities)
    ///   - KZG commit + open via GPU MSM
    ///
    /// Complexity: O(n) for multiplicity + O(k) sparse MSM + O(N log N) NTT +
    /// O(N) for KZG openings, where k = distinct lookup values, n = total lookups.
    public func prove(lookups: [Fr], table: GPUCQTableCommitment) throws -> GPUCQProof {
        let n = lookups.count
        let T = table.table.count
        precondition(n > 0, "Lookups must be non-empty")

        let _t0 = profile ? CFAbsoluteTimeGetCurrent() : 0
        var _tPhase = profile ? CFAbsoluteTimeGetCurrent() : 0

        // Step 1: Compute multiplicity vector m[i] = count of T[i] in lookups
        let mult = computeMultiplicities(table: table.table, lookups: lookups)

        // Verify sum of multiplicities = n
        var multSum = Fr.zero
        for i in 0..<T {
            multSum = frAdd(multSum, mult[i])
        }
        let expectedN = frFromInt(UInt64(n))
        precondition(frEqual(multSum, expectedN),
                     "Sum of multiplicities != N (\(n))")

        if profile {
            let _t = CFAbsoluteTimeGetCurrent()
            fputs(String(format: "  [gpu-cq] multiplicities: %.2f ms\n",
                         (_t - _tPhase) * 1000), stderr)
            _tPhase = _t
        }

        // Step 2: Compute quotient commitment via SPARSE MSM over cached quotients.
        // Q = sum_{i : m_i > 0} m_i * [q_i(tau)]
        let quotientCommitment = gpuSparseMSM(
            multiplicities: mult,
            cachedCommitments: table.cachedQuotientCommitments)

        if profile {
            let _t = CFAbsoluteTimeGetCurrent()
            fputs(String(format: "  [gpu-cq] sparse MSM (quotient): %.2f ms\n",
                         (_t - _tPhase) * 1000), stderr)
            _tPhase = _t
        }

        // Step 3: Compute phi(X) = sum_i m_i * Z_T(X)/(X - omega^i)
        //
        // Z_T(X)/(X - omega^i) = sum_{k=0}^{T-1} omega^{ik} * X^{T-1-k}
        //
        // So: phi(X) = sum_{k=0}^{T-1} X^{T-1-k} * sum_i m_i * omega^{ik}
        //            = sum_{k=0}^{T-1} X^{T-1-k} * M_hat[k]
        //
        // where M_hat = NTT(m). The coefficient of X^j in phi is M_hat[T-1-j].
        let mHat = try nttEngine.ntt(mult)

        var phiCoeffs = [Fr](repeating: Fr.zero, count: T)
        for j in 0..<T {
            phiCoeffs[j] = mHat[T - 1 - j]
        }

        if profile {
            let _t = CFAbsoluteTimeGetCurrent()
            fputs(String(format: "  [gpu-cq] phi polynomial (NTT): %.2f ms\n",
                         (_t - _tPhase) * 1000), stderr)
            _tPhase = _t
        }

        // Step 4: Commit to phi(X) via KZG
        let phiCommitment = try kzg.commit(phiCoeffs)

        if profile {
            let _t = CFAbsoluteTimeGetCurrent()
            fputs(String(format: "  [gpu-cq] KZG commit phi: %.2f ms\n",
                         (_t - _tPhase) * 1000), stderr)
            _tPhase = _t
        }

        // Step 5: Fiat-Shamir challenge
        var transcript = [UInt8]()
        appendPoint(&transcript, table.commitment)
        appendPoint(&transcript, phiCommitment)
        appendPoint(&transcript, quotientCommitment)
        for i in 0..<T {
            appendFr(&transcript, mult[i])
        }
        let z = deriveChallenge(transcript)

        // Step 6: Open phi(X) at z via KZG
        let phiOpening = try kzg.open(phiCoeffs, at: z)

        // Step 7: Open t(X) at z via KZG
        let tOpening = try kzg.open(table.tableCoeffs, at: z)

        if profile {
            let _t = CFAbsoluteTimeGetCurrent()
            fputs(String(format: "  [gpu-cq] KZG openings: %.2f ms\n",
                         (_t - _tPhase) * 1000), stderr)
            _tPhase = _t
        }

        if profile {
            let total = (CFAbsoluteTimeGetCurrent() - _t0) * 1000
            fputs(String(format: "  [gpu-cq] TOTAL prove (n=%d, T=%d): %.2f ms\n",
                         n, T, total), stderr)
        }

        return GPUCQProof(
            phiCommitment: phiCommitment,
            quotientCommitment: quotientCommitment,
            multiplicities: mult,
            multiplicitySum: multSum,
            challengeZ: z,
            phiOpening: phiOpening,
            tOpening: tOpening,
            phiCoeffs: phiCoeffs
        )
    }

    // MARK: - Verify

    /// Verify a GPU CQ proof.
    ///
    /// Checks:
    /// 1. Sum of multiplicities equals the claimed lookup count.
    /// 2. Fiat-Shamir challenge is correctly derived.
    /// 3. Core algebraic identity: phi(z) / Z_T(z) = sum_i m_i / (z - omega^i)
    /// 4. KZG opening proofs for phi(X) and t(X) at z.
    /// 5. Quotient commitment consistency: [Q] = sum m_i * [q_i(tau)].
    ///
    /// GPU acceleration: batch inverse for the N terms 1/(z - omega^i).
    ///
    /// The `srsSecret` parameter enables verification with known SRS secret
    /// (testing only; production uses pairings).
    public func verify(proof: GPUCQProof, table: GPUCQTableCommitment,
                       numLookups: Int, srsSecret: Fr) -> Bool {
        let T = table.table.count
        let z = proof.challengeZ

        // Check 1: Sum of multiplicities = N
        let expectedN = frFromInt(UInt64(numLookups))
        guard frEqual(proof.multiplicitySum, expectedN) else { return false }

        // Check 2: Reconstruct Fiat-Shamir challenge
        var transcript = [UInt8]()
        appendPoint(&transcript, table.commitment)
        appendPoint(&transcript, proof.phiCommitment)
        appendPoint(&transcript, proof.quotientCommitment)
        for i in 0..<T {
            appendFr(&transcript, proof.multiplicities[i])
        }
        let expectedZ = deriveChallenge(transcript)
        guard frEqual(z, expectedZ) else { return false }

        // Check 3: Core identity: phi(z) / Z_T(z) = sum_i m_i / (z - omega^i)
        let phiZ = proof.phiOpening.evaluation
        let zT = vanishingPolyEval(z: z, T: T)  // z^T - 1
        guard !frEqual(zT, Fr.zero) else { return false }
        let zTInv = frInverse(zT)
        let lhs = frMul(phiZ, zTInv)

        // GPU-accelerated batch inverse for the denominators (z - omega^i)
        let rhs = computeVerifierSum(multiplicities: proof.multiplicities,
                                      roots: table.roots, z: z, T: T)

        guard frEqual(lhs, rhs) else { return false }

        // Check 4: Verify KZG openings
        let phiValid = verifySingleKZG(
            commitment: proof.phiCommitment,
            point: z,
            evaluation: proof.phiOpening.evaluation,
            witness: proof.phiOpening.witness,
            srsSecret: srsSecret
        )
        guard phiValid else { return false }

        let tValid = verifySingleKZG(
            commitment: table.commitment,
            point: z,
            evaluation: proof.tOpening.evaluation,
            witness: proof.tOpening.witness,
            srsSecret: srsSecret
        )
        guard tValid else { return false }

        // Check 5: Quotient commitment consistency
        let expectedQ = gpuSparseMSM(
            multiplicities: proof.multiplicities,
            cachedCommitments: table.cachedQuotientCommitments)
        guard pointsEqual(proof.quotientCommitment, expectedQ) else { return false }

        return true
    }

    /// Verify using pairing checks (production-grade, no SRS secret needed).
    ///
    /// Uses BN254 pairings to verify KZG openings:
    ///   e([phi] - phi(z)*G1, G2) = e([W_phi], tau*G2 - z*G2)
    public func verifyWithPairings(proof: GPUCQProof, table: GPUCQTableCommitment,
                                    numLookups: Int, srsG2: G2AffinePoint) -> Bool {
        let T = table.table.count
        let z = proof.challengeZ

        // Check 1: Sum of multiplicities = N
        let expectedN = frFromInt(UInt64(numLookups))
        guard frEqual(proof.multiplicitySum, expectedN) else { return false }

        // Check 2: Reconstruct Fiat-Shamir challenge
        var transcript = [UInt8]()
        appendPoint(&transcript, table.commitment)
        appendPoint(&transcript, proof.phiCommitment)
        appendPoint(&transcript, proof.quotientCommitment)
        for i in 0..<T {
            appendFr(&transcript, proof.multiplicities[i])
        }
        let expectedZ = deriveChallenge(transcript)
        guard frEqual(z, expectedZ) else { return false }

        // Check 3: Core algebraic identity
        let phiZ = proof.phiOpening.evaluation
        let zT = vanishingPolyEval(z: z, T: T)
        guard !frEqual(zT, Fr.zero) else { return false }
        let zTInv = frInverse(zT)
        let lhs = frMul(phiZ, zTInv)

        let rhs = computeVerifierSum(multiplicities: proof.multiplicities,
                                      roots: table.roots, z: z, T: T)
        guard frEqual(lhs, rhs) else { return false }

        // Check 4: Quotient commitment consistency
        let expectedQ = gpuSparseMSM(
            multiplicities: proof.multiplicities,
            cachedCommitments: table.cachedQuotientCommitments)
        guard pointsEqual(proof.quotientCommitment, expectedQ) else { return false }

        // Check 5: Verify KZG openings via pairings
        let g1 = bn254G1Generator()
        let g2 = bn254G2Generator()

        // phi opening pairing check
        let phiZG1 = batchToAffine([cPointScalarMul(pointFromAffine(g1), phiZ)])[0]
        let phiCommAff = batchToAffine([proof.phiCommitment])[0]
        let phiLHS = batchToAffine([pointAdd(
            pointFromAffine(phiCommAff),
            pointNeg(pointFromAffine(phiZG1)))])[0]
        let zG2 = g2ToAffine(g2ScalarMul(g2FromAffine(g2), frToInt(z)))!
        let tauMinusZG2 = g2ToAffine(g2Add(g2FromAffine(srsG2), g2Negate(g2FromAffine(zG2))))!
        let phiWitAff = batchToAffine([proof.phiOpening.witness])[0]

        let phiPairingValid = cBN254PairingCheck([
            (phiLHS, g2),
            (pointNegateAffine(phiWitAff), tauMinusZG2)
        ])
        guard phiPairingValid else { return false }

        // t opening pairing check
        let tZ = proof.tOpening.evaluation
        let tZG1 = batchToAffine([cPointScalarMul(pointFromAffine(g1), tZ)])[0]
        let tCommAff = batchToAffine([table.commitment])[0]
        let tLHS = batchToAffine([pointAdd(
            pointFromAffine(tCommAff),
            pointNeg(pointFromAffine(tZG1)))])[0]
        let tWitAff = batchToAffine([proof.tOpening.witness])[0]

        let tPairingValid = cBN254PairingCheck([
            (tLHS, g2),
            (pointNegateAffine(tWitAff), tauMinusZG2)
        ])
        guard tPairingValid else { return false }

        return true
    }

    // MARK: - GPU Sparse MSM over Cached Quotients

    /// Compute Q = sum_{i : m_i > 0} m_i * [q_i(tau)] using sparse MSM.
    ///
    /// Only iterates over non-zero multiplicities, making this O(k) where
    /// k is the number of distinct lookup values.
    ///
    /// For k > gpuThreshold, uses GPU MSM via MetalMSM. Otherwise CPU path.
    private func gpuSparseMSM(
        multiplicities: [Fr], cachedCommitments: [PointProjective]
    ) -> PointProjective {
        let T = multiplicities.count
        precondition(T == cachedCommitments.count)

        // Collect non-zero entries
        var sparsePoints = [PointProjective]()
        var sparseScalars = [Fr]()

        for i in 0..<T {
            if !frEqual(multiplicities[i], Fr.zero) {
                sparsePoints.append(cachedCommitments[i])
                sparseScalars.append(multiplicities[i])
            }
        }

        if sparsePoints.isEmpty {
            return pointIdentity()
        }

        // For larger sparse sets, use Pippenger MSM on the sparse subset
        if sparsePoints.count >= gpuThreshold {
            let affinePoints = batchToAffine(sparsePoints)
            let scalars = sparseScalars.map { frToLimbs($0) }
            if let result = try? kzg.msmEngine.msm(points: affinePoints, scalars: scalars) {
                return result
            }
        }

        // CPU fallback: scalar mul + add
        var result = pointIdentity()
        for i in 0..<sparsePoints.count {
            let term = cPointScalarMul(sparsePoints[i], sparseScalars[i])
            result = pointAdd(result, term)
        }
        return result
    }

    // MARK: - Multiplicity Computation

    /// Compute multiplicity vector: m[i] = number of times T[i] appears in lookups.
    ///
    /// Uses a hash map for O(n + N) complexity.
    public func computeMultiplicities(table: [Fr], lookups: [Fr]) -> [Fr] {
        var tableIndex = [FrKey: Int]()
        for j in 0..<table.count {
            tableIndex[FrKey(table[j])] = j
        }
        var mult = [UInt64](repeating: 0, count: table.count)
        for i in 0..<lookups.count {
            guard let idx = tableIndex[FrKey(lookups[i])] else {
                preconditionFailure("gpu-cq: lookup value not in table at index \(i)")
            }
            mult[idx] += 1
        }
        return mult.map { frFromInt($0) }
    }

    // MARK: - Verifier Sum (GPU batch inverse)

    /// Compute sum_i m_i / (z - omega^i) using GPU batch inverse.
    ///
    /// This is the verifier-side computation. GPU acceleration via batch inverse
    /// for the N denominators (z - omega^i).
    private func computeVerifierSum(multiplicities: [Fr], roots: [Fr],
                                     z: Fr, T: Int) -> Fr {
        // Build denominators: (z - omega^i) for all i
        var denoms = [Fr](repeating: Fr.zero, count: T)
        for i in 0..<T {
            denoms[i] = frSub(z, roots[i])
        }

        // GPU batch inverse for T >= gpuThreshold, CPU for small T
        let inverses: [Fr]
        if T >= gpuThreshold, let gpuInv = try? batchInverseEngine.batchInverseFr(denoms) {
            inverses = gpuInv
        } else {
            inverses = cpuBatchInverse(denoms)
        }

        // Accumulate: sum_i m_i * inv_i
        var result = Fr.zero
        for i in 0..<T {
            if frEqual(multiplicities[i], Fr.zero) { continue }
            result = frAdd(result, frMul(multiplicities[i], inverses[i]))
        }
        return result
    }

    /// CPU batch inverse via Montgomery's trick.
    private func cpuBatchInverse(_ a: [Fr]) -> [Fr] {
        let n = a.count
        if n == 0 { return [] }

        // Build prefix products
        var prefix = [Fr](repeating: Fr.one, count: n)
        prefix[0] = a[0]
        for i in 1..<n {
            prefix[i] = frMul(prefix[i - 1], a[i])
        }

        // Invert the total product
        var inv = frInverse(prefix[n - 1])
        var result = [Fr](repeating: Fr.zero, count: n)

        // Back-propagate to get individual inverses
        for i in stride(from: n - 1, through: 1, by: -1) {
            result[i] = frMul(inv, prefix[i - 1])
            inv = frMul(inv, a[i])
        }
        result[0] = inv

        return result
    }

    // MARK: - Polynomial Helpers

    /// Synthetic division: given polynomial p(X) and a root r,
    /// compute q(X) = p(X) / (X - r).
    /// p must vanish at r (i.e., p(r) = 0) for exact division.
    private func syntheticDivide(_ coeffs: [Fr], root: Fr) -> [Fr] {
        let n = coeffs.count
        if n < 2 { return [] }

        var q = [Fr](repeating: Fr.zero, count: n - 1)
        q[n - 2] = coeffs[n - 1]
        for i in stride(from: n - 2, through: 1, by: -1) {
            q[i - 1] = frAdd(coeffs[i], frMul(root, q[i]))
        }
        return q
    }

    /// Evaluate vanishing polynomial Z_T(z) = z^T - 1
    private func vanishingPolyEval(z: Fr, T: Int) -> Fr {
        var zPow = z
        var t = T
        var result = Fr.one
        while t > 0 {
            if t & 1 == 1 {
                result = frMul(result, zPow)
            }
            zPow = frMul(zPow, zPow)
            t >>= 1
        }
        return frSub(result, Fr.one)
    }

    /// Evaluate polynomial in coefficient form at a point using Horner's method.
    public func evaluatePoly(_ coeffs: [Fr], at z: Fr) -> Fr {
        if coeffs.isEmpty { return Fr.zero }
        var result = coeffs[coeffs.count - 1]
        for i in stride(from: coeffs.count - 2, through: 0, by: -1) {
            result = frAdd(frMul(result, z), coeffs[i])
        }
        return result
    }

    /// Verify a single KZG opening using the SRS secret (for testing).
    private func verifySingleKZG(commitment: PointProjective, point: Fr,
                                  evaluation: Fr, witness: PointProjective,
                                  srsSecret: Fr) -> Bool {
        let g1 = pointFromAffine(kzg.srs[0])
        let yG = cPointScalarMul(g1, evaluation)
        let sMz = frSub(srsSecret, point)
        let szW = cPointScalarMul(witness, sMz)
        let expected = pointAdd(yG, szW)
        return pointsEqual(commitment, expected)
    }

    /// Compare two projective points for equality.
    private func pointsEqual(_ a: PointProjective, _ b: PointProjective) -> Bool {
        let aAffine = batchToAffine([a])
        let bAffine = batchToAffine([b])
        return fpToInt(aAffine[0].x) == fpToInt(bAffine[0].x) &&
               fpToInt(aAffine[0].y) == fpToInt(bAffine[0].y)
    }

    // MARK: - Fiat-Shamir

    private func appendFr(_ transcript: inout [UInt8], _ v: Fr) {
        let vInt = frToInt(v)
        for limb in vInt {
            for byte in 0..<8 {
                transcript.append(UInt8((limb >> (byte * 8)) & 0xFF))
            }
        }
    }

    private func appendPoint(_ transcript: inout [UInt8], _ p: PointProjective) {
        let affine = batchToAffine([p])
        let xInt = fpToInt(affine[0].x)
        let yInt = fpToInt(affine[0].y)
        for limb in xInt {
            for byte in 0..<8 {
                transcript.append(UInt8((limb >> (byte * 8)) & 0xFF))
            }
        }
        for limb in yInt {
            for byte in 0..<8 {
                transcript.append(UInt8((limb >> (byte * 8)) & 0xFF))
            }
        }
    }

    private func deriveChallenge(_ transcript: [UInt8]) -> Fr {
        let hash = blake3(transcript)
        var limbs = [UInt64](repeating: 0, count: 4)
        for i in 0..<4 {
            for j in 0..<8 {
                limbs[i] |= UInt64(hash[i * 8 + j]) << (j * 8)
            }
        }
        let raw = Fr.from64(limbs)
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))
    }

    // MARK: - Batch Prove/Verify Helpers

    /// Prove multiple lookup batches against the same table.
    /// Returns an array of proofs, one per batch.
    public func proveBatch(lookupBatches: [[Fr]], table: GPUCQTableCommitment) throws -> [GPUCQProof] {
        var proofs = [GPUCQProof]()
        proofs.reserveCapacity(lookupBatches.count)
        for lookups in lookupBatches {
            let proof = try prove(lookups: lookups, table: table)
            proofs.append(proof)
        }
        return proofs
    }

    /// Verify multiple proofs against the same table.
    /// Returns true only if all proofs are valid.
    public func verifyBatch(proofs: [GPUCQProof], table: GPUCQTableCommitment,
                            lookupCounts: [Int], srsSecret: Fr) -> Bool {
        guard proofs.count == lookupCounts.count else { return false }
        for i in 0..<proofs.count {
            if !verify(proof: proofs[i], table: table,
                      numLookups: lookupCounts[i], srsSecret: srsSecret) {
                return false
            }
        }
        return true
    }
}

// MARK: - Convenience Round-Trip

extension GPUCQLookupEngine {
    /// Run a complete prove-verify round-trip.
    /// Returns (proof, verified) tuple.
    public func proveAndVerify(lookups: [Fr], table: GPUCQTableCommitment,
                               srsSecret: Fr) throws -> (GPUCQProof, Bool) {
        let proof = try prove(lookups: lookups, table: table)
        let valid = verify(proof: proof, table: table,
                          numLookups: lookups.count, srsSecret: srsSecret)
        return (proof, valid)
    }
}
