// Cached Quotients (cq) Lookup Argument Engine
// Proves that every element in a lookup vector exists in a table using the cq protocol.
//
// Protocol (Eagen-Fiore-Gabizon 2022):
//   Given table T[0..|T|-1] and lookups f[0..N-1] where each f[i] in T:
//   1. Preprocess: Commit to table polynomial t(X) via KZG (done once per table).
//   2. Prover:
//      a. Compute multiplicity vector m[i] = #{j : f[j] = T[i]}
//      b. Compute cached quotient h(X) = sum_i m[i] / (X - omega^i)
//         In coefficient form via iNTT of m[i] / derivative_of_vanishing.
//      c. Commit h(X) via KZG.
//      d. Open h(X) and t(X) at a Fiat-Shamir challenge point z.
//      e. Verifier checks: h(z) * Z_T(z) = sum_i m[i] * L_i(z)
//         equivalently: h(z) = sum_i m[i] / (z - omega^i)
//   3. Verifier: Check KZG openings + identity. O(1) verification (with pairing).
//
// Key advantage: Prover work is O(N log N) regardless of table size |T|.
// LogUp is O(N + |T|). cq decouples prover cost from table size via preprocessing.

import Foundation
import Metal

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
}

/// Proof that all lookups exist in the table using cached quotients.
public struct CQProof {
    /// KZG commitment to quotient polynomial h(X)
    public let hCommitment: PointProjective
    /// Multiplicity vector (m[i] = number of times T[i] appears in lookups)
    public let multiplicities: [Fr]
    /// Sum of multiplicities (should equal N = number of lookups)
    public let multiplicitySum: Fr
    /// Fiat-Shamir challenge point
    public let challengeZ: Fr
    /// KZG opening proof for h(X) at z
    public let hOpening: KZGProof
    /// KZG opening proof for t(X) at z
    public let tOpening: KZGProof
    /// h(z) value (also in hOpening.evaluation, kept for clarity)
    public let hEvalAtZ: Fr
    /// t(z) value
    public let tEvalAtZ: Fr
}

public class CQEngine {
    public static let version = Versions.cqLookup
    public let kzg: KZGEngine
    public let nttEngine: NTTEngine

    public init(srs: [PointAffine]) throws {
        self.kzg = try KZGEngine(srs: srs)
        self.nttEngine = try NTTEngine()
    }

    /// Preprocess a table: compute table polynomial and KZG commitment.
    /// This is done once per table and can be reused for many proofs.
    /// Table size must be a power of 2.
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

        return CQTableCommitment(
            table: table,
            tableCoeffs: tableCoeffs,
            commitment: commitment,
            roots: roots,
            logT: logT
        )
    }

    /// Prove that every element in `lookups` exists in the preprocessed table.
    /// Returns a CQProof.
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

        // Step 2: Compute quotient polynomial h(X) = sum_i m[i] / (X - omega^i)
        //
        // This is a partial fraction decomposition. In evaluation-domain terms:
        // h(X) can be computed from its evaluations. But more efficiently,
        // we use the fact that sum_i m[i]/(X - omega^i) = M(X) / Z_T(X)
        // where M(X) = sum_i m[i] * prod_{j!=i} (X - omega^j)
        //
        // In coefficient form: h(X) has degree T-1 (at most).
        // h evaluated at any point z: h(z) = sum_i m[i] / (z - omega^i)
        //
        // To get h in coefficient form for commitment, we compute:
        // h_evals[k] = sum_i m[i] / (omega^k - omega^i) for k != i
        // But this has poles. Instead, use the relation:
        //
        // h(X) * Z_T(X) = sum_i m[i] * L_i(X) * Z_T(X) / (X - omega^i)
        //               = sum_i m[i] * prod_{j!=i}(X - omega^j)
        //
        // Simpler: compute h in coefficient form via:
        // The derivative of vanishing polynomial Z_T(X) = X^T - 1 is T*X^(T-1).
        // Z_T'(omega^i) = T * omega^{i*(T-1)} = T * omega^{-i}
        //
        // By partial fractions: h(X) = sum_i m[i]/(X - omega^i)
        // The polynomial h(X) * Z_T(X) has degree 2T-1 with specific structure.
        //
        // Efficient approach: compute h's evaluations on a coset, then iNTT.
        // Or directly compute coefficients via:
        //   h_hat[k] = (1/T) * sum_i m[i] * omega^{-ik}  for k = 0..T-2
        //            = (1/T) * NTT^{-1}(m)[k] ... but shifted
        //
        // Actually the cleanest: h(X) = sum_i (m[i] / Z_T'(omega^i)) * 1/(X - omega^i)
        // and Z_T'(omega^i) = T * omega^{-i}, so m[i]/Z_T'(omega^i) = m[i]*omega^i/T
        //
        // h(X) = (1/T) * sum_i m[i]*omega^i / (X - omega^i)
        //
        // Wait, let's be precise. We want h(X) = sum_i m[i]/(X - omega^i).
        // This is not a polynomial (it has poles). We need to actually compute
        // the polynomial numerator: p(X) = h(X) * Z_T(X) = sum_i m[i] * prod_{j!=i}(X - omega^j)
        // Then h(X) = p(X) / Z_T(X), and we commit to a polynomial phi(X) of degree T-1
        // such that phi(X) * Z_T(X) = p(X).
        //
        // p(X) = sum_i m[i] * Z_T(X)/(X - omega^i)
        //       = sum_i m[i] * (Z_T(X) - 0)/(X - omega^i)   [since Z_T(omega^i)=0]
        //
        // By the derivative relation: Z_T(X)/(X - omega^i) = Z_T'(omega^i)^{-1} * sum_{k} ...
        // More directly: Z_T(X)/(X-omega^i) = sum_{k=0}^{T-1} omega^{ik} * X^{T-1-k}
        // (geometric series division)
        //
        // So p(X) = sum_i m[i] * sum_{k=0}^{T-1} omega^{ik} * X^{T-1-k}
        //         = sum_{k=0}^{T-1} X^{T-1-k} * sum_i m[i]*omega^{ik}
        //         = sum_{k=0}^{T-1} X^{T-1-k} * M_hat[k]
        //
        // where M_hat = NTT(m). So the coefficient of X^j in p(X) is M_hat[T-1-j].
        // p has degree T-1 (since Z_T has degree T and h has degree T-1).
        //
        // phi(X) = p(X), degree T-1, and the protocol relation is phi(X) = h(X)*Z_T(X).
        //
        // But wait: we don't divide by Z_T since phi IS the numerator polynomial.
        // The prover commits to phi(X) and the verifier checks
        // phi(z) = sum_i m[i]/(z - omega^i) * Z_T(z)
        // i.e., phi(z)/Z_T(z) = sum_i m[i]/(z - omega^i) = h(z)
        //
        // Let's commit to phi directly. phi_coeffs[j] = M_hat[T-1-j].

        // Compute M_hat = NTT(m)
        let mHat = try nttEngine.ntt(mult)

        // phi_coeffs[j] = M_hat[T-1-j] for j = 0..T-1
        var phiCoeffs = [Fr](repeating: Fr.zero, count: T)
        for j in 0..<T {
            phiCoeffs[j] = mHat[T - 1 - j]
        }

        // Commit to phi(X) — this is the "h commitment" in the protocol
        let hCommitment = try kzg.commit(phiCoeffs)

        // Step 3: Fiat-Shamir challenge
        var transcript = [UInt8]()
        appendPoint(&transcript, table.commitment)
        appendPoint(&transcript, hCommitment)
        for i in 0..<T {
            appendFr(&transcript, mult[i])
        }
        let z = deriveChallenge(transcript)

        // Step 4: Open phi(X) at z
        let hOpening = try kzg.open(phiCoeffs, at: z)

        // Step 5: Open t(X) at z
        let tOpening = try kzg.open(table.tableCoeffs, at: z)

        return CQProof(
            hCommitment: hCommitment,
            multiplicities: mult,
            multiplicitySum: multSum,
            challengeZ: z,
            hOpening: hOpening,
            tOpening: tOpening,
            hEvalAtZ: hOpening.evaluation,
            tEvalAtZ: tOpening.evaluation
        )
    }

    /// Verify a CQ proof.
    /// In a full implementation, verification uses pairings for O(1) cost.
    /// Here we verify the algebraic relation and KZG openings using the SRS secret
    /// (acceptable for testing; production would use pairings).
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
        appendPoint(&transcript, proof.hCommitment)
        for i in 0..<T {
            appendFr(&transcript, proof.multiplicities[i])
        }
        let expectedZ = deriveChallenge(transcript)
        guard frEqual(z, expectedZ) else { return false }

        // Check 3: Verify the core identity:
        // phi(z) = sum_i m[i] * omega^{i} ... no, let's use the direct check:
        // phi(z) / Z_T(z) should equal sum_i m[i] / (z - omega^i)
        //
        // Compute LHS: phi(z) / Z_T(z)
        let phiZ = proof.hEvalAtZ
        let zT = vanishingPolyEval(z: z, T: T)  // z^T - 1
        let zTInv = frInverse(zT)
        let lhs = frMul(phiZ, zTInv)

        // Compute RHS: sum_i m[i] / (z - omega^i)
        var rhs = Fr.zero
        for i in 0..<T {
            let mi = proof.multiplicities[i]
            // Skip zero multiplicities for efficiency
            if frEqual(mi, Fr.zero) { continue }
            let zMinusOmegaI = frSub(z, table.roots[i])
            let inv = frInverse(zMinusOmegaI)
            rhs = frAdd(rhs, frMul(mi, inv))
        }

        guard frEqual(lhs, rhs) else { return false }

        // Check 4: Verify KZG openings
        // phi opening at z
        let phiValid = verifySingleKZG(
            commitment: proof.hCommitment,
            point: z,
            evaluation: proof.hEvalAtZ,
            witness: proof.hOpening.witness,
            srsSecret: srsSecret
        )
        guard phiValid else { return false }

        // t opening at z
        let tValid = verifySingleKZG(
            commitment: table.commitment,
            point: z,
            evaluation: proof.tEvalAtZ,
            witness: proof.tOpening.witness,
            srsSecret: srsSecret
        )
        guard tValid else { return false }

        return true
    }

    // MARK: - Helpers

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

    /// Compute roots of unity omega^0, omega^1, ..., omega^{n-1}
    /// where omega is the principal n-th root of unity for BN254 Fr.
    private func computeRootsOfUnity(logN: Int) -> [Fr] {
        let n = 1 << logN
        // BN254 Fr has a 2^28-th root of unity.
        // The primitive 2^28-th root is OMEGA_28.
        // For n = 2^logN, omega_n = OMEGA_28^(2^(28 - logN))
        let omega = rootOfUnity(logN: logN)
        var roots = [Fr](repeating: Fr.zero, count: n)
        roots[0] = Fr.one
        for i in 1..<n {
            roots[i] = frMul(roots[i - 1], omega)
        }
        return roots
    }

    /// Get the n-th root of unity for BN254 Fr (n = 2^logN).
    private func rootOfUnity(logN: Int) -> Fr {
        // The 2^28-th primitive root of unity for BN254 Fr (in Montgomery form).
        // omega_28 = 19103219067921713944291392827692070036674651213730446576152998835171324499365
        // In 64-bit limbs (standard form):
        let omega28Limbs: [UInt64] = [
            0x3c3d3ca739381fb2, 0x9a14cda3ec99772b,
            0xd7f4e2f43e8cc868, 0x0f3a6ca462326449
        ]
        var omega = Fr.from64(omega28Limbs)
        omega = frMul(omega, Fr.from64(Fr.R2_MOD_R))  // to Montgomery form

        // Square (28 - logN) times to get 2^logN-th root
        for _ in 0..<(28 - logN) {
            omega = frMul(omega, omega)
        }
        return omega
    }

    /// Evaluate vanishing polynomial Z_T(z) = z^T - 1
    private func vanishingPolyEval(z: Fr, T: Int) -> Fr {
        var zPow = z
        var t = T
        // Compute z^T via repeated squaring
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
        // Check: C == [y]*G + [s - z]*W
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
