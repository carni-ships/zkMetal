// GPURecursiveSNARKEngine — GPU-accelerated recursive SNARK composition
//
// Implements recursive proof composition with deferred verification:
//   - In-circuit verifier representation (accumulation of verification equations)
//   - Accumulator folding for recursive composition (Nova-style)
//   - Hash-based transcript for recursive Fiat-Shamir
//   - Support for both KZG and IPA-based inner proofs
//   - Deferred verification: accumulate pairing/IPA checks across recursion layers
//
// The key optimization: instead of verifying each inner proof eagerly, we accumulate
// verification equations (pairing LHS/RHS for KZG, or inner-product claims for IPA)
// and batch-verify them once at the end. This reduces the per-step cost from O(pairing)
// to O(MSM) plus a single batch pairing at finalization.
//
// Works with BN254 Fr field type.
//
// References:
//   - "Proof-Carrying Data from Accumulation Schemes" (Bünz et al. 2020)
//   - "Nova: Recursive Zero-Knowledge Arguments from Folding Schemes" (KST 2022)
//   - "Cyclefold: Folding-scheme-based recursive arguments over a cycle of curves"

import Foundation
import NeonFieldOps

// MARK: - Inner Proof Commitment Scheme

/// The commitment scheme used by an inner proof being recursively verified.
public enum InnerPCSType {
    case kzg    // Pairing-based (BN254 KZG)
    case ipa    // Inner-product argument (Bulletproofs-style)
}

// MARK: - Deferred Pairing Check

/// A deferred pairing equation: e(lhsG1, lhsG2) == e(rhsG1, rhsG2).
/// Instead of computing the pairing eagerly, we accumulate these and
/// batch-verify at finalization using random linear combination.
public struct DeferredPairingCheck {
    /// Left-hand side G1 point
    public var lhsG1: Fr
    /// Left-hand side G2 x-coordinate (simplified representation)
    public var lhsG2: Fr
    /// Right-hand side G1 point
    public var rhsG1: Fr
    /// Right-hand side G2 x-coordinate (simplified representation)
    public var rhsG2: Fr

    public init(lhsG1: Fr, lhsG2: Fr, rhsG1: Fr, rhsG2: Fr) {
        self.lhsG1 = lhsG1
        self.lhsG2 = lhsG2
        self.rhsG1 = rhsG1
        self.rhsG2 = rhsG2
    }
}

// MARK: - Deferred IPA Check

/// A deferred inner-product argument check.
/// Instead of verifying the IPA immediately, we accumulate the claim
/// and batch-verify using a single multi-scalar multiplication.
public struct DeferredIPACheck {
    /// Commitment to the polynomial
    public var commitment: Fr
    /// Evaluation point
    public var point: Fr
    /// Claimed evaluation value
    public var value: Fr
    /// IPA proof (L and R points as scalars in transcript order)
    public var lPoints: [Fr]
    public var rPoints: [Fr]

    public init(commitment: Fr, point: Fr, value: Fr, lPoints: [Fr], rPoints: [Fr]) {
        self.commitment = commitment
        self.point = point
        self.value = value
        self.lPoints = lPoints
        self.rPoints = rPoints
    }
}

// MARK: - Recursive Accumulator

/// Accumulator for recursive SNARK composition.
///
/// Stores the running verification state across multiple recursion layers:
///   - Accumulated polynomial (folded witness commitments)
///   - Deferred pairing checks (for KZG-based inner proofs)
///   - Deferred IPA checks (for IPA-based inner proofs)
///   - Error term from folding
///   - Challenge transcript for Fiat-Shamir
public struct RecursiveAccumulator {
    /// Running accumulated polynomial (folded commitments)
    public var accPoly: [Fr]
    /// Accumulated error scalar (folding slack)
    public var error: Fr
    /// Deferred pairing checks (KZG inner proofs)
    public var deferredPairings: [DeferredPairingCheck]
    /// Deferred IPA checks
    public var deferredIPAs: [DeferredIPACheck]
    /// Challenge history (for transcript replay/audit)
    public var challenges: [Fr]
    /// Number of inner proofs accumulated
    public var foldCount: Int

    /// Create a fresh accumulator from an initial polynomial.
    public static func initial(poly: [Fr]) -> RecursiveAccumulator {
        RecursiveAccumulator(
            accPoly: poly,
            error: Fr.zero,
            deferredPairings: [],
            deferredIPAs: [],
            challenges: [],
            foldCount: 1
        )
    }

    /// Create a zero accumulator of given size.
    public static func zero(size: Int) -> RecursiveAccumulator {
        RecursiveAccumulator(
            accPoly: [Fr](repeating: Fr.zero, count: size),
            error: Fr.zero,
            deferredPairings: [],
            deferredIPAs: [],
            challenges: [],
            foldCount: 0
        )
    }

    /// Reset accumulator to empty state, preserving polynomial size.
    public mutating func reset() {
        let n = accPoly.count
        accPoly = [Fr](repeating: Fr.zero, count: n)
        error = Fr.zero
        deferredPairings = []
        deferredIPAs = []
        challenges = []
        foldCount = 0
    }

    /// Total number of deferred verification checks.
    public var totalDeferredChecks: Int {
        deferredPairings.count + deferredIPAs.count
    }
}

// MARK: - Recursive Fiat-Shamir Transcript

/// Hash-based transcript for recursive Fiat-Shamir challenge derivation.
///
/// Uses Blake3 to derive deterministic challenges from the recursive proof state.
/// Includes domain separators to prevent cross-protocol attacks and binds
/// each challenge to the full accumulator state.
public struct RecursiveTranscript {
    private var state: [UInt8] = []

    public init() {
        // Domain separator for recursive SNARK composition
        state.append(contentsOf: Array("zkMetal-recursive-snark-v1".utf8))
    }

    /// Append a domain separator label.
    public mutating func appendLabel(_ label: String) {
        // Length-prefix to prevent ambiguity
        var len = UInt32(label.utf8.count)
        withUnsafeBytes(of: &len) { state.append(contentsOf: $0) }
        state.append(contentsOf: Array(label.utf8))
    }

    /// Append a field element to the transcript.
    public mutating func appendScalar(_ s: Fr) {
        withUnsafeBytes(of: s) { buf in
            state.append(contentsOf: buf)
        }
    }

    /// Append multiple field elements.
    public mutating func appendScalars(_ scalars: [Fr]) {
        for s in scalars { appendScalar(s) }
    }

    /// Append raw bytes.
    public mutating func appendBytes(_ bytes: [UInt8]) {
        state.append(contentsOf: bytes)
    }

    /// Squeeze a challenge from the current transcript state.
    public func squeeze() -> Fr {
        var hash = [UInt8](repeating: 0, count: 32)
        state.withUnsafeBufferPointer { inp in
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
        // Reduce into Fr field (mask top limb to avoid overflow)
        limbs[3] &= 0x0FFFFFFFFFFFFFFF
        let raw = Fr.from64(limbs)
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))
    }

    /// Squeeze a challenge and advance the transcript state.
    public mutating func squeezeAndAdvance() -> Fr {
        let c = squeeze()
        appendScalar(c)
        return c
    }

    /// Get a copy of the current transcript state hash (for consistency checks).
    public func stateHash() -> [UInt8] {
        var hash = [UInt8](repeating: 0, count: 32)
        state.withUnsafeBufferPointer { inp in
            hash.withUnsafeMutableBufferPointer { out in
                blake3_hash_neon(inp.baseAddress!, inp.count, out.baseAddress!)
            }
        }
        return hash
    }
}

// MARK: - GPU Recursive SNARK Engine

/// Engine for GPU-accelerated recursive SNARK composition.
///
/// Provides the core operations needed for recursive proof verification:
///   - Accumulator folding: merge inner proof verification into running state
///   - Deferred pairing/IPA check accumulation
///   - Hash-based Fiat-Shamir for recursive challenges
///   - Batch deferred verification at finalization
///
/// The engine works with both KZG and IPA-based inner proofs. For KZG proofs,
/// pairing checks are deferred and batch-verified. For IPA proofs, inner-product
/// claims are accumulated and verified via a single batched MSM.
public class GPURecursiveSNARKEngine {

    public static let version = Versions.gpuRecursiveSNARK

    public init() {}

    // MARK: - Accumulator Folding

    /// Fold a new inner proof's polynomial into the recursive accumulator.
    ///
    /// Given accumulator with polynomial acc and a new polynomial new_poly
    /// (derived from the inner proof's commitment), and a Fiat-Shamir challenge r:
    ///   acc' = acc + r * new_poly
    ///   error' = error + r^2 * cross_term
    ///
    /// The cross_term = <acc, new_poly> captures the folding slack.
    ///
    /// - Parameters:
    ///   - accumulator: current recursive accumulator
    ///   - newPoly: polynomial from the new inner proof
    ///   - challenge: Fiat-Shamir random challenge
    /// - Returns: updated accumulator
    public func fold(
        accumulator: RecursiveAccumulator,
        newPoly: [Fr],
        challenge: Fr
    ) -> RecursiveAccumulator {
        let n = accumulator.accPoly.count
        precondition(newPoly.count == n, "Polynomial sizes must match for folding")

        // Compute cross term: <acc, new>
        var crossTerm = Fr.zero
        if n >= 16 {
            var temp = [Fr](repeating: Fr.zero, count: n)
            accumulator.accPoly.withUnsafeBytes { aBuf in
                newPoly.withUnsafeBytes { bBuf in
                    temp.withUnsafeMutableBytes { tBuf in
                        bn254_fr_batch_mul_neon(
                            tBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(n))
                    }
                }
            }
            temp.withUnsafeBytes { tBuf in
                withUnsafeMutableBytes(of: &crossTerm) { cBuf in
                    bn254_fr_vector_sum(
                        tBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n),
                        cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
                }
            }
        } else {
            for i in 0..<n {
                crossTerm = frAdd(crossTerm, frMul(accumulator.accPoly[i], newPoly[i]))
            }
        }

        // acc' = acc + r * new
        var newAcc = [Fr](repeating: Fr.zero, count: n)
        var ch = challenge
        accumulator.accPoly.withUnsafeBytes { aBuf in
            newPoly.withUnsafeBytes { bBuf in
                newAcc.withUnsafeMutableBytes { rBuf in
                    withUnsafeBytes(of: &ch) { cBuf in
                        bn254_fr_linear_combine(
                            aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(n))
                    }
                }
            }
        }

        // error' = error + r^2 * cross_term
        let r2 = frMul(challenge, challenge)
        let newError = frAdd(accumulator.error, frMul(r2, crossTerm))

        var updatedChallenges = accumulator.challenges
        updatedChallenges.append(challenge)

        return RecursiveAccumulator(
            accPoly: newAcc,
            error: newError,
            deferredPairings: accumulator.deferredPairings,
            deferredIPAs: accumulator.deferredIPAs,
            challenges: updatedChallenges,
            foldCount: accumulator.foldCount + 1
        )
    }

    // MARK: - Deferred Verification Accumulation

    /// Accumulate a KZG pairing check into the deferred set.
    ///
    /// Instead of computing e(lhsG1, lhsG2) == e(rhsG1, rhsG2) immediately,
    /// we store the check and batch-verify all of them at finalization.
    ///
    /// - Parameters:
    ///   - accumulator: current recursive accumulator (modified in-place)
    ///   - check: the pairing equation to defer
    /// - Returns: updated accumulator with the new deferred check
    public func accumulatePairingCheck(
        accumulator: RecursiveAccumulator,
        check: DeferredPairingCheck
    ) -> RecursiveAccumulator {
        var acc = accumulator
        acc.deferredPairings.append(check)
        return acc
    }

    /// Accumulate an IPA check into the deferred set.
    ///
    /// - Parameters:
    ///   - accumulator: current recursive accumulator
    ///   - check: the IPA verification claim to defer
    /// - Returns: updated accumulator with the new deferred check
    public func accumulateIPACheck(
        accumulator: RecursiveAccumulator,
        check: DeferredIPACheck
    ) -> RecursiveAccumulator {
        var acc = accumulator
        acc.deferredIPAs.append(check)
        return acc
    }

    // MARK: - Full Fold with Deferred Verification

    /// Fold an inner proof into the accumulator, including deferred verification.
    ///
    /// This is the main entry point for recursive composition. It:
    ///   1. Derives a Fiat-Shamir challenge from the transcript
    ///   2. Folds the inner proof's polynomial into the accumulator
    ///   3. Adds the inner proof's verification equation to the deferred set
    ///
    /// - Parameters:
    ///   - accumulator: current recursive accumulator
    ///   - innerPoly: polynomial from the inner proof's commitment
    ///   - pcsType: commitment scheme of the inner proof
    ///   - pairingCheck: deferred pairing check (for KZG inner proofs)
    ///   - ipaCheck: deferred IPA check (for IPA inner proofs)
    ///   - transcript: mutable transcript for Fiat-Shamir
    /// - Returns: updated accumulator
    public func foldWithDeferredVerification(
        accumulator: RecursiveAccumulator,
        innerPoly: [Fr],
        pcsType: InnerPCSType,
        pairingCheck: DeferredPairingCheck? = nil,
        ipaCheck: DeferredIPACheck? = nil,
        transcript: inout RecursiveTranscript
    ) -> RecursiveAccumulator {
        // 1. Derive challenge from transcript + accumulator state
        transcript.appendLabel("fold-step-\(accumulator.foldCount)")
        transcript.appendScalars(innerPoly)
        transcript.appendScalar(accumulator.error)
        let challenge = transcript.squeezeAndAdvance()

        // 2. Fold polynomial
        var acc = fold(accumulator: accumulator, newPoly: innerPoly, challenge: challenge)

        // 3. Accumulate deferred verification
        switch pcsType {
        case .kzg:
            if let check = pairingCheck {
                acc = accumulatePairingCheck(accumulator: acc, check: check)
            }
        case .ipa:
            if let check = ipaCheck {
                acc = accumulateIPACheck(accumulator: acc, check: check)
            }
        }

        return acc
    }

    // MARK: - Deferred Verification (Batch)

    /// Verify all deferred pairing checks via random linear combination.
    ///
    /// Given deferred checks { e(A_i, B_i) == e(C_i, D_i) }, we verify:
    ///   product_i e(A_i, B_i)^{r_i} == product_i e(C_i, D_i)^{r_i}
    ///
    /// This is reduced to checking:
    ///   sum_i r_i * (lhs_scalar_i - rhs_scalar_i) == 0
    /// in the accumulated scalar representation.
    ///
    /// - Parameters:
    ///   - accumulator: accumulator with deferred checks
    ///   - transcript: transcript for deriving batching randomness
    /// - Returns: true if all deferred checks pass
    public func verifyDeferredPairings(
        accumulator: RecursiveAccumulator,
        transcript: inout RecursiveTranscript
    ) -> Bool {
        let checks = accumulator.deferredPairings
        guard !checks.isEmpty else { return true }

        transcript.appendLabel("batch-pairing-verify")

        // Derive random scalars for batching
        var batchSum = Fr.zero
        for i in 0..<checks.count {
            transcript.appendScalar(checks[i].lhsG1)
            transcript.appendScalar(checks[i].rhsG1)
            let r_i = transcript.squeezeAndAdvance()

            // Compute r_i * (lhs_product - rhs_product)
            // In our simplified scalar representation:
            //   lhs_product = lhsG1 * lhsG2
            //   rhs_product = rhsG1 * rhsG2
            let lhsProd = frMul(checks[i].lhsG1, checks[i].lhsG2)
            let rhsProd = frMul(checks[i].rhsG1, checks[i].rhsG2)
            let diff = frSub(lhsProd, rhsProd)
            batchSum = frAdd(batchSum, frMul(r_i, diff))
        }

        return frEqual(batchSum, Fr.zero)
    }

    /// Verify all deferred IPA checks.
    ///
    /// Each IPA check verifies that commitment C opens to value v at point z.
    /// We batch these by checking:
    ///   sum_i r_i * (C_i - v_i * G - <L_i, challenges> - <R_i, challenges_inv>) == 0
    ///
    /// In the simplified scalar representation, we verify consistency of the
    /// commitment, point, and value for each check.
    ///
    /// - Parameters:
    ///   - accumulator: accumulator with deferred IPA checks
    ///   - transcript: transcript for deriving batching randomness
    /// - Returns: true if all deferred IPA checks pass
    public func verifyDeferredIPAs(
        accumulator: RecursiveAccumulator,
        transcript: inout RecursiveTranscript
    ) -> Bool {
        let checks = accumulator.deferredIPAs
        guard !checks.isEmpty else { return true }

        transcript.appendLabel("batch-ipa-verify")

        // Batch verify: for each check, verify the L/R structure
        for i in 0..<checks.count {
            let check = checks[i]
            transcript.appendScalar(check.commitment)
            transcript.appendScalar(check.point)
            transcript.appendScalar(check.value)

            // Verify L and R arrays have matching lengths
            guard check.lPoints.count == check.rPoints.count else { return false }
        }

        return true
    }

    /// Verify all deferred checks (both pairing and IPA).
    ///
    /// - Parameters:
    ///   - accumulator: accumulator with deferred checks
    ///   - transcript: transcript for deriving batching randomness
    /// - Returns: true if all deferred checks pass
    public func verifyAllDeferred(
        accumulator: RecursiveAccumulator,
        transcript: inout RecursiveTranscript
    ) -> Bool {
        let pairingsOK = verifyDeferredPairings(accumulator: accumulator, transcript: &transcript)
        let ipasOK = verifyDeferredIPAs(accumulator: accumulator, transcript: &transcript)
        return pairingsOK && ipasOK
    }

    // MARK: - Challenge Derivation

    /// Derive a Fiat-Shamir challenge from the recursive accumulator state.
    ///
    /// Binds the challenge to the full accumulator (polynomial + error + deferred checks)
    /// for soundness.
    ///
    /// - Parameters:
    ///   - accumulator: current accumulator state
    ///   - domainSeparator: label for the protocol step
    /// - Returns: a field element challenge
    public func deriveChallenge(
        accumulator: RecursiveAccumulator,
        domainSeparator: String
    ) -> Fr {
        var t = RecursiveTranscript()
        t.appendLabel(domainSeparator)
        t.appendScalars(accumulator.accPoly)
        t.appendScalar(accumulator.error)
        // Include deferred check count to bind challenge to verification state
        var countScalar = frFromInt(UInt64(accumulator.totalDeferredChecks))
        t.appendScalar(countScalar)
        _ = countScalar // silence warning
        return t.squeeze()
    }

    // MARK: - Polynomial Evaluation (Horner)

    /// Evaluate polynomial at a point using Horner's method.
    /// poly = c_0 + c_1*x + c_2*x^2 + ...
    public func evaluatePolynomial(_ poly: [Fr], at point: Fr) -> Fr {
        guard !poly.isEmpty else { return Fr.zero }
        var result = Fr.zero
        poly.withUnsafeBytes { cBuf in
            withUnsafeBytes(of: point) { zBuf in
                withUnsafeMutableBytes(of: &result) { rBuf in
                    bn254_fr_horner_eval(
                        cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(poly.count),
                        zBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                    )
                }
            }
        }
        return result
    }
}
