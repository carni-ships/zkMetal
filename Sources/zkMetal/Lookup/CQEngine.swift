// Cached Quotients (cq) Lookup Argument Engine
// Proves that every element in a lookup vector exists in a table using the cq protocol.
//
// Protocol (Eagen-Fiore-Gabizon, eprint 2022/1763):
//   Given table T[0..|T|-1] and lookups f[0..n-1] where each f[i] in T:
//
//   Setup (O(N log N), one-time per table):
//     1. Commit to table polynomial t(X) via KZG.
//     2. Precompute cached quotient commitments:
//        For each i in 0..N-1:
//          q_i(X) = (t(X) - t_i) / (X - omega^i)
//          [q_i] = KZG.commit(q_i)
//        These are the "cached quotients" — expensive but amortized across proofs.
//
//   Prove (O(n log n), independent of table size N):
//     1. Compute multiplicity vector m[i] = #{j : f[j] = T[i]}
//     2. Compute quotient commitment via sparse MSM over cached quotients:
//        [Q] = Σ_{i : m_i > 0} m_i · [q_i(τ)]
//     3. Compute numerator polynomial phi(X) = Σ_i m_i · Z_T(X)/(X - omega^i)
//     4. Open phi(X) and t(X) at a Fiat-Shamir challenge point z via KZG.
//     5. Verifier checks:
//        phi(z) / Z_T(z) = Σ_i m_i / (z - omega^i)
//
//   Verify (O(1) with pairings, or O(N) without):
//     1. Verify KZG openings of phi and t at challenge z.
//     2. Check the core identity: phi(z) = Z_T(z) · Σ_i m_i/(z - omega^i)
//     3. Check multiplicity sum = n (number of lookups).
//     4. (Pairing mode) Verify [Q] commitment matches via pairing check.
//
// Key advantage: Prover work is O(n log n) regardless of table size N.
// LogUp is O(n + N). cq decouples prover cost from table size via preprocessing.

import Foundation
import Metal

// MARK: - Data Structures

/// Preprocessed table commitment for the cq protocol.
/// Created once per table, reused across many proofs.
public struct CQTableCommitment {
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
    /// Cached quotient commitments: [q_i(τ)] for each table entry.
    /// q_i(X) = (t(X) - t_i) / (X - omega^i)
    /// This is the key precomputation that makes cq sublinear in table size.
    public let cachedQuotientCommitments: [PointProjective]
}

/// Proof that all lookups exist in the table using cached quotients.
public struct CQProof {
    /// KZG commitment to numerator polynomial phi(X) = Σ m_i · Z_T(X)/(X - ω^i)
    public let phiCommitment: PointProjective
    /// Sparse quotient commitment Q = Σ m_i · [q_i(τ)] (from cached quotients)
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

    public init(phiCommitment: PointProjective, quotientCommitment: PointProjective,
                multiplicities: [Fr], multiplicitySum: Fr,
                challengeZ: Fr, phiOpening: KZGProof, tOpening: KZGProof) {
        self.phiCommitment = phiCommitment
        self.quotientCommitment = quotientCommitment
        self.multiplicities = multiplicities
        self.multiplicitySum = multiplicitySum
        self.challengeZ = challengeZ
        self.phiOpening = phiOpening
        self.tOpening = tOpening
    }
}

// MARK: - CQ Engine

public class CQEngine {
    public static let version = Versions.cqLookup
    public let kzg: KZGEngine
    public let nttEngine: NTTEngine

    public init(srs: [PointAffine]) throws {
        self.kzg = try KZGEngine(srs: srs)
        self.nttEngine = try NTTEngine()
    }

    // MARK: - Setup (one-time per table)

    /// Preprocess a table: compute table polynomial, KZG commitment, and cached quotient commitments.
    /// This is done once per table and can be reused for many proofs.
    ///
    /// Table size must be a power of 2.
    ///
    /// Complexity: O(N^2) naive, or O(N log N) with NTT-based quotient computation.
    /// The cached quotient commitments are the expensive part — one MSM per table entry.
    ///
    /// For production use with very large tables, the quotient commitments would be
    /// computed in parallel and stored persistently (e.g., on disk).
    public func preprocessTable(table: [Fr]) throws -> CQTableCommitment {
        let T = table.count
        precondition(T > 0 && (T & (T - 1)) == 0, "Table size must be power of 2")
        let logT = Int(log2(Double(T)))

        // Compute roots of unity omega^i
        let roots = computeRootsOfUnity(logN: logT)

        // Table polynomial: table values are evaluations at roots of unity.
        // t(X) = iNTT(table_evals) gives us coefficient form.
        let tableCoeffs = try nttEngine.intt(table)

        // Commit to t(X) via KZG
        let commitment = try kzg.commit(tableCoeffs)

        // Precompute cached quotient commitments.
        // For each i, q_i(X) = (t(X) - t_i) / (X - omega^i).
        //
        // Since t(omega^i) = t_i (table values are evaluations), the numerator
        // (t(X) - t_i) vanishes at X = omega^i, so the division is exact.
        //
        // We compute q_i in coefficient form via synthetic division:
        //   numerator(X) = t(X) - t_i  (subtract t_i from constant term)
        //   q_i(X) = numerator(X) / (X - omega^i)
        //
        // Then commit each q_i via KZG (MSM).
        let cachedQuotients = try computeCachedQuotientCommitments(
            tableCoeffs: tableCoeffs, table: table, roots: roots)

        return CQTableCommitment(
            table: table,
            tableCoeffs: tableCoeffs,
            commitment: commitment,
            roots: roots,
            logT: logT,
            cachedQuotientCommitments: cachedQuotients
        )
    }

    /// Compute cached quotient commitments for all table entries.
    /// q_i(X) = (t(X) - t_i) / (X - omega^i), then [q_i] = KZG.commit(q_i).
    private func computeCachedQuotientCommitments(
        tableCoeffs: [Fr], table: [Fr], roots: [Fr]
    ) throws -> [PointProjective] {
        let T = table.count
        var commitments = [PointProjective]()
        commitments.reserveCapacity(T)

        for i in 0..<T {
            // numerator(X) = t(X) - t_i
            var numerator = tableCoeffs
            numerator[0] = frSub(numerator[0], table[i])

            // q_i(X) = numerator(X) / (X - omega^i)
            // Synthetic division by (X - omega^i)
            let qi = syntheticDivide(numerator, root: roots[i])

            // Commit q_i(X)
            let qiCommitment = try kzg.commit(qi)
            commitments.append(qiCommitment)
        }

        return commitments
    }

    // MARK: - Prove

    /// Prove that every element in `lookups` exists in the preprocessed table.
    ///
    /// The key optimization of cq: the quotient commitment is computed as a
    /// sparse MSM over cached quotient commitments, touching only entries with
    /// non-zero multiplicity. If the lookup accesses only k distinct table entries,
    /// the MSM has size k (not N).
    ///
    /// Complexity: O(n) for multiplicity computation + O(k) for sparse MSM +
    /// O(N log N) for phi polynomial + O(N) for KZG openings, where k = number
    /// of distinct lookup values and n = number of lookups.
    public func prove(lookups: [Fr], table: CQTableCommitment) throws -> CQProof {
        let N = lookups.count
        let T = table.table.count
        precondition(N > 0, "Lookups must be non-empty")

        // Step 1: Compute multiplicity vector m[i] = count of T[i] in lookups
        let mult = CQEngine.computeMultiplicities(table: table.table, lookups: lookups)

        // Verify sum of multiplicities = N
        var multSum = Fr.zero
        for i in 0..<T {
            multSum = frAdd(multSum, mult[i])
        }
        let expectedN = frFromInt(UInt64(N))
        precondition(frEqual(multSum, expectedN),
                     "Sum of multiplicities (\(frToInt(multSum))) != N (\(N))")

        // Step 2: Compute quotient commitment via SPARSE MSM over cached quotients.
        // Q = Σ_{i : m_i > 0} m_i · [q_i(τ)]
        // This is the core cq optimization: only non-zero multiplicities contribute.
        let quotientCommitment = sparseMSMOverCachedQuotients(
            multiplicities: mult, cachedCommitments: table.cachedQuotientCommitments)

        // Step 3: Compute phi(X) = Σ_i m_i · Z_T(X)/(X - omega^i)
        //
        // Z_T(X)/(X - omega^i) = Σ_{k=0}^{T-1} omega^{ik} · X^{T-1-k}  (geometric series)
        //
        // So: phi(X) = Σ_i m_i · Σ_{k=0}^{T-1} omega^{ik} · X^{T-1-k}
        //            = Σ_{k=0}^{T-1} X^{T-1-k} · Σ_i m_i · omega^{ik}
        //            = Σ_{k=0}^{T-1} X^{T-1-k} · M_hat[k]
        //
        // where M_hat = NTT(m). The coefficient of X^j in phi is M_hat[T-1-j].
        let mHat = try nttEngine.ntt(mult)

        var phiCoeffs = [Fr](repeating: Fr.zero, count: T)
        for j in 0..<T {
            phiCoeffs[j] = mHat[T - 1 - j]
        }

        // Commit to phi(X)
        let phiCommitment = try kzg.commit(phiCoeffs)

        // Step 4: Fiat-Shamir challenge
        var transcript = [UInt8]()
        appendPoint(&transcript, table.commitment)
        appendPoint(&transcript, phiCommitment)
        appendPoint(&transcript, quotientCommitment)
        for i in 0..<T {
            appendFr(&transcript, mult[i])
        }
        let z = deriveChallenge(transcript)

        // Step 5: Open phi(X) at z
        let phiOpening = try kzg.open(phiCoeffs, at: z)

        // Step 6: Open t(X) at z
        let tOpening = try kzg.open(table.tableCoeffs, at: z)

        return CQProof(
            phiCommitment: phiCommitment,
            quotientCommitment: quotientCommitment,
            multiplicities: mult,
            multiplicitySum: multSum,
            challengeZ: z,
            phiOpening: phiOpening,
            tOpening: tOpening
        )
    }

    // MARK: - Verify

    /// Verify a CQ proof.
    ///
    /// Checks:
    /// 1. Sum of multiplicities equals the claimed lookup count.
    /// 2. Fiat-Shamir challenge is correctly derived.
    /// 3. Core algebraic identity: phi(z) / Z_T(z) = Σ_i m_i / (z - omega^i)
    /// 4. KZG opening proofs for phi(X) and t(X) at z.
    /// 5. Quotient commitment consistency: [Q] = Σ m_i · [q_i(τ)].
    ///
    /// The `srsSecret` parameter enables verification using the known SRS secret
    /// (acceptable for testing; production would use pairings via `verifyWithPairings`).
    public func verify(proof: CQProof, table: CQTableCommitment,
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

        // Check 3: Core identity: phi(z) / Z_T(z) = Σ_i m_i / (z - omega^i)
        let phiZ = proof.phiOpening.evaluation
        let zT = vanishingPolyEval(z: z, T: T)  // z^T - 1
        guard !frEqual(zT, Fr.zero) else { return false }  // z must not be a root of unity
        let zTInv = frInverse(zT)
        let lhs = frMul(phiZ, zTInv)

        // Batch-invert (z - omega^i) for all non-zero multiplicity entries
        var cqDenoms = [Fr](repeating: Fr.zero, count: T)
        for i in 0..<T { cqDenoms[i] = frSub(z, table.roots[i]) }
        var cqPrefix = [Fr](repeating: Fr.one, count: T)
        for i in 1..<T {
            cqPrefix[i] = cqDenoms[i - 1] == Fr.zero ? cqPrefix[i - 1] : frMul(cqPrefix[i - 1], cqDenoms[i - 1])
        }
        let cqLast = cqDenoms[T - 1] == Fr.zero ? cqPrefix[T - 1] : frMul(cqPrefix[T - 1], cqDenoms[T - 1])
        var cqInvRunning = frInverse(cqLast)
        var cqDenomInvs = [Fr](repeating: Fr.zero, count: T)
        for i in stride(from: T - 1, through: 0, by: -1) {
            if cqDenoms[i] != Fr.zero {
                cqDenomInvs[i] = frMul(cqInvRunning, cqPrefix[i])
                cqInvRunning = frMul(cqInvRunning, cqDenoms[i])
            }
        }
        var rhs = Fr.zero
        for i in 0..<T {
            let mi = proof.multiplicities[i]
            if frEqual(mi, Fr.zero) { continue }
            rhs = frAdd(rhs, frMul(mi, cqDenomInvs[i]))
        }

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
        // Recompute Q = Σ m_i · [q_i(τ)] and compare
        let expectedQ = sparseMSMOverCachedQuotients(
            multiplicities: proof.multiplicities,
            cachedCommitments: table.cachedQuotientCommitments)
        let qAffine = batchToAffine([proof.quotientCommitment])
        let eqAffine = batchToAffine([expectedQ])
        guard fpToInt(qAffine[0].x) == fpToInt(eqAffine[0].x) &&
              fpToInt(qAffine[0].y) == fpToInt(eqAffine[0].y) else { return false }

        return true
    }

    /// Verify a CQ proof using pairing checks (production-grade, no SRS secret needed).
    ///
    /// Uses BN254 pairings to verify KZG openings:
    ///   e([phi] - [phi(z)]·G1, G2) = e([W_phi], [τ]·G2 - [z]·G2)
    /// where [W_phi] is the KZG witness for phi at z.
    ///
    /// Parameters:
    /// - proof: The CQ proof to verify.
    /// - table: The preprocessed table commitment.
    /// - numLookups: Expected number of lookups.
    /// - srsG2: G2 SRS point [τ]·G2 (second element of SRS in G2).
    ///
    /// Returns true if the proof is valid.
    public func verifyWithPairings(proof: CQProof, table: CQTableCommitment,
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

        // Batch-invert (z - omega^i)
        var cq2Denoms = [Fr](repeating: Fr.zero, count: T)
        for i in 0..<T { cq2Denoms[i] = frSub(z, table.roots[i]) }
        var cq2Prefix = [Fr](repeating: Fr.one, count: T)
        for i in 1..<T {
            cq2Prefix[i] = cq2Denoms[i - 1] == Fr.zero ? cq2Prefix[i - 1] : frMul(cq2Prefix[i - 1], cq2Denoms[i - 1])
        }
        let cq2Last = cq2Denoms[T - 1] == Fr.zero ? cq2Prefix[T - 1] : frMul(cq2Prefix[T - 1], cq2Denoms[T - 1])
        var cq2InvRunning = frInverse(cq2Last)
        var cq2DenomInvs = [Fr](repeating: Fr.zero, count: T)
        for i in stride(from: T - 1, through: 0, by: -1) {
            if cq2Denoms[i] != Fr.zero {
                cq2DenomInvs[i] = frMul(cq2InvRunning, cq2Prefix[i])
                cq2InvRunning = frMul(cq2InvRunning, cq2Denoms[i])
            }
        }
        var rhs = Fr.zero
        for i in 0..<T {
            let mi = proof.multiplicities[i]
            if frEqual(mi, Fr.zero) { continue }
            rhs = frAdd(rhs, frMul(mi, cq2DenomInvs[i]))
        }
        guard frEqual(lhs, rhs) else { return false }

        // Check 4: Quotient commitment consistency
        let expectedQ = sparseMSMOverCachedQuotients(
            multiplicities: proof.multiplicities,
            cachedCommitments: table.cachedQuotientCommitments)
        let qAff = batchToAffine([proof.quotientCommitment])
        let eqAff = batchToAffine([expectedQ])
        guard fpToInt(qAff[0].x) == fpToInt(eqAff[0].x) &&
              fpToInt(qAff[0].y) == fpToInt(eqAff[0].y) else { return false }

        // Check 5: Verify KZG openings via pairings
        // For phi(X) at z: e([phi] - [phi(z)]·G1, G2) = e([W_phi], [τ·G2] - [z·G2])
        let g1 = bn254G1Generator()
        let g2 = bn254G2Generator()

        // phi opening pairing check
        let phiZG1 = batchToAffine([cPointScalarMul(pointFromAffine(g1), phiZ)])[0]
        let phiCommAff = batchToAffine([proof.phiCommitment])[0]
        // [phi] - [phi(z)]·G1
        let phiLHS = batchToAffine([pointAdd(
            pointFromAffine(phiCommAff),
            pointNeg(pointFromAffine(phiZG1)))])[0]
        // [τ]·G2 - [z]·G2
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

    // MARK: - Sparse MSM over cached quotients

    /// Compute Q = Σ_{i : m_i > 0} m_i · [q_i(τ)] using sparse MSM.
    /// Only iterates over non-zero multiplicities, making this O(k) where
    /// k is the number of distinct lookup values (not the full table size N).
    private func sparseMSMOverCachedQuotients(
        multiplicities: [Fr], cachedCommitments: [PointProjective]
    ) -> PointProjective {
        let T = multiplicities.count
        precondition(T == cachedCommitments.count)

        // Collect non-zero entries for sparse MSM
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

        // Perform MSM: Q = Σ m_i · [q_i]
        // Use scalar multiplication + addition for correctness.
        // For large sparse sets, could upgrade to Pippenger on the sparse subset.
        var result = pointIdentity()
        for i in 0..<sparsePoints.count {
            let term = cPointScalarMul(sparsePoints[i], sparseScalars[i])
            result = pointAdd(result, term)
        }

        return result
    }

    // MARK: - Multiplicity Computation

    /// Compute multiplicity vector: m[i] = number of times T[i] appears in lookups.
    public static func computeMultiplicities(table: [Fr], lookups: [Fr]) -> [Fr] {
        var tableIndex = [FrKey: Int]()
        for j in 0..<table.count {
            tableIndex[FrKey(table[j])] = j
        }
        var mult = [UInt64](repeating: 0, count: table.count)
        for i in 0..<lookups.count {
            guard let idx = tableIndex[FrKey(lookups[i])] else {
                preconditionFailure("cq: lookup value not in table at index \(i)")
            }
            mult[idx] += 1
        }
        return mult.map { frFromInt($0) }
    }

    // MARK: - Polynomial Helpers

    /// Synthetic division: given polynomial p(X) and a root r,
    /// compute q(X) = p(X) / (X - r).
    /// p must vanish at r (i.e., p(r) = 0) for exact division.
    /// Returns coefficients of q(X) with degree one less than p(X).
    private func syntheticDivide(_ coeffs: [Fr], root: Fr) -> [Fr] {
        let n = coeffs.count
        if n < 2 { return [] }

        // Standard synthetic division: q[n-2] = a[n-1], then
        // q[i-1] = a[i] + root * q[i] for i = n-2 down to 1
        var q = [Fr](repeating: Fr.zero, count: n - 1)
        q[n - 2] = coeffs[n - 1]
        for i in stride(from: n - 2, through: 1, by: -1) {
            q[i - 1] = frAdd(coeffs[i], frMul(root, q[i]))
        }
        // Remainder should be zero: coeffs[0] + root * q[0] = 0
        // (This holds when p(root) = 0)
        return q
    }

    /// Compute roots of unity omega^0, omega^1, ..., omega^{n-1}
    /// where omega is the principal n-th root of unity for BN254 Fr.
    private func computeRootsOfUnity(logN: Int) -> [Fr] {
        return precomputeTwiddles(logN: logN)
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

    /// Verify a single KZG opening using the SRS secret (for testing).
    /// In production this would be a pairing check.
    private func verifySingleKZG(commitment: PointProjective, point: Fr,
                                  evaluation: Fr, witness: PointProjective,
                                  srsSecret: Fr) -> Bool {
        let g1 = pointFromAffine(kzg.srs[0])
        let yG = cPointScalarMul(g1, evaluation)
        let sMz = frSub(srsSecret, point)
        let szW = cPointScalarMul(witness, sMz)
        let expected = pointAdd(yG, szW)

        let cAffine = batchToAffine([commitment])
        let eAffine = batchToAffine([expected])
        return fpToInt(cAffine[0].x) == fpToInt(eAffine[0].x) &&
               fpToInt(cAffine[0].y) == fpToInt(eAffine[0].y)
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
}

// MARK: - Convenience: Prove + Verify round-trip helpers

extension CQEngine {
    /// Run a complete prove-verify round-trip (useful for testing).
    /// Returns (proof, verified) tuple.
    public func proveAndVerify(lookups: [Fr], table: CQTableCommitment,
                               srsSecret: Fr) throws -> (CQProof, Bool) {
        let proof = try prove(lookups: lookups, table: table)
        let valid = verify(proof: proof, table: table,
                          numLookups: lookups.count, srsSecret: srsSecret)
        return (proof, valid)
    }
}
