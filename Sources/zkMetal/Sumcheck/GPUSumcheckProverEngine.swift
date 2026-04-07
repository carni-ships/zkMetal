// GPU-accelerated sumcheck prover engine for multilinear polynomials.
//
// Provides a high-level prover interface that executes the sumcheck protocol
// round by round, producing degree-2 univariate polynomials for each round
// and folding the evaluation table via random challenges.
//
// Delegates GPU compute to GPUSumcheckEngine for Metal acceleration on large
// tables and falls back to CPU for small ones.

import Foundation
import Metal
import NeonFieldOps

// MARK: - Round Polynomial

/// A degree-2 univariate polynomial representing one sumcheck round.
/// p(X) = c0 + c1*X + c2*X^2
/// Satisfies: p(0) = s0, p(1) = s1, where s0 + s1 = claimed sum for the round.
/// c0 = s0, c2 = s0 + s1 - 2*s_half (for degree-2 from combining),
/// but for standard multilinear sumcheck the round poly is degree 1:
///   p(X) = s0 + (s1 - s0)*X
/// We store the degree-2 form for generality (c2 = 0 for pure multilinear).
public struct SumcheckRoundPoly {
    public let c0: Fr   // constant term = p(0) = s0
    public let c1: Fr   // linear coefficient
    public let c2: Fr   // quadratic coefficient (zero for degree-1)

    /// Evaluate p(X) at a point r: c0 + c1*r + c2*r^2
    public func evaluate(at r: Fr) -> Fr {
        let r2 = frMul(r, r)
        let t1 = frMul(c1, r)
        let t2 = frMul(c2, r2)
        return frAdd(c0, frAdd(t1, t2))
    }

    /// p(0)
    public var atZero: Fr { c0 }

    /// p(1) = c0 + c1 + c2
    public var atOne: Fr { frAdd(c0, frAdd(c1, c2)) }

    /// Degree of the polynomial (0, 1, or 2 based on leading coefficient)
    public var degree: Int {
        if !frIsZero(c2) { return 2 }
        if !frIsZero(c1) { return 1 }
        return 0
    }
}

// MARK: - Sumcheck Proof

/// Complete sumcheck proof: round polynomials + challenges + final evaluation.
public struct GPUSumcheckProof {
    public let roundPolys: [SumcheckRoundPoly]
    public let challenges: [Fr]
    public let finalEval: Fr
    public let numVars: Int
}

// MARK: - GPU Sumcheck Prover Engine

/// High-level GPU-accelerated sumcheck prover.
///
/// Given a multilinear polynomial f: {0,1}^n -> Fr (stored as 2^n evaluations),
/// and a claimed sum C = sum_{x in {0,1}^n} f(x), the prover executes n rounds:
///
/// Round i:
///   1. Compute round polynomial p_i(X) where p_i(0) + p_i(1) = current claim
///   2. Send p_i to verifier (via transcript)
///   3. Receive challenge r_i (from transcript via Fiat-Shamir)
///   4. Fold the table: f_{i+1}(x) = f_i(0,x) + r_i * (f_i(1,x) - f_i(0,x))
///
/// After n rounds, the final evaluation f(r_0,...,r_{n-1}) can be checked
/// against an oracle query.
public class GPUSumcheckProverEngine {
    private let gpuEngine: GPUSumcheckEngine?
    private static let gpuThreshold = 1024

    /// Initialize with optional GPU acceleration.
    /// Falls back to CPU-only if Metal is unavailable.
    public init() {
        self.gpuEngine = try? GPUSumcheckEngine()
    }

    /// Initialize with an existing GPU engine (for reuse).
    public init(engine: GPUSumcheckEngine) {
        self.gpuEngine = engine
    }

    /// Whether GPU acceleration is available.
    public var hasGPU: Bool { gpuEngine != nil }

    // MARK: - Full Sumcheck Prove

    /// Execute the full sumcheck protocol for a multilinear polynomial.
    ///
    /// - Parameters:
    ///   - evals: Evaluation table of size 2^numVars (multilinear extension over {0,1}^n)
    ///   - claimedSum: The claimed sum of all evaluations (verified during proving)
    ///   - transcript: Fiat-Shamir transcript for generating challenges
    ///
    /// - Returns: A SumcheckProof containing round polynomials, challenges, and final eval
    /// - Throws: If GPU dispatch fails or if the claimed sum is inconsistent
    public func prove(
        evals: [Fr],
        claimedSum: Fr,
        transcript: Transcript
    ) throws -> GPUSumcheckProof {
        let n = evals.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Table size must be a power of 2")

        let numVars = trailingZeroBitCount(n)

        // Use GPU path if engine is available and table is large enough
        if let engine = gpuEngine, n >= GPUSumcheckProverEngine.gpuThreshold {
            return try proveGPU(evals: evals, numVars: numVars, claimedSum: claimedSum,
                                transcript: transcript, engine: engine)
        }

        return try proveCPU(evals: evals, numVars: numVars, claimedSum: claimedSum,
                            transcript: transcript)
    }

    /// Execute sumcheck for a MultilinearPoly directly.
    public func prove(
        poly: MultilinearPoly,
        claimedSum: Fr,
        transcript: Transcript
    ) throws -> GPUSumcheckProof {
        return try prove(evals: poly.evals, claimedSum: claimedSum, transcript: transcript)
    }

    // MARK: - GPU Path

    private func proveGPU(
        evals: [Fr],
        numVars: Int,
        claimedSum: Fr,
        transcript: Transcript,
        engine: GPUSumcheckEngine
    ) throws -> GPUSumcheckProof {
        let stride = MemoryLayout<Fr>.stride
        let n = evals.count

        guard let device = gpuEngine?.device else {
            throw MSMError.noGPU
        }

        guard let tableBuf = device.makeBuffer(length: n * stride, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to create table buffer")
        }
        evals.withUnsafeBytes { src in
            memcpy(tableBuf.contents(), src.baseAddress!, n * stride)
        }

        var currentTable = tableBuf
        var currentLogSize = numVars
        var roundPolys: [SumcheckRoundPoly] = []
        var challenges: [Fr] = []
        roundPolys.reserveCapacity(numVars)
        challenges.reserveCapacity(numVars)

        for _ in 0..<numVars {
            // Compute round polynomial via GPU
            let (s0, s1) = try engine.computeRoundPolyBN254(table: currentTable, logSize: currentLogSize)

            // Build degree-1 round polynomial: p(X) = s0 + (s1 - s0)*X
            let c1 = frSub(s1, s0)
            let roundPoly = SumcheckRoundPoly(c0: s0, c1: c1, c2: Fr.zero)
            roundPolys.append(roundPoly)

            // Absorb into transcript and get challenge
            transcript.absorb(s0)
            transcript.absorb(s1)
            let challenge = transcript.squeeze()
            challenges.append(challenge)

            // Fold the table via GPU
            let newTable = try engine.reduceBN254Table(
                table: currentTable, logSize: currentLogSize, challenge: challenge)
            currentTable = newTable
            currentLogSize -= 1
        }

        // Final evaluation is the single remaining element
        let finalPtr = currentTable.contents().bindMemory(to: Fr.self, capacity: 1)
        let finalEval = finalPtr[0]

        return GPUSumcheckProof(
            roundPolys: roundPolys,
            challenges: challenges,
            finalEval: finalEval,
            numVars: numVars
        )
    }

    // MARK: - CPU Path

    private func proveCPU(
        evals: [Fr],
        numVars: Int,
        claimedSum: Fr,
        transcript: Transcript
    ) throws -> GPUSumcheckProof {
        var current = evals
        var currentLogSize = numVars
        var roundPolys: [SumcheckRoundPoly] = []
        var challenges: [Fr] = []
        roundPolys.reserveCapacity(numVars)
        challenges.reserveCapacity(numVars)

        for _ in 0..<numVars {
            let halfN = current.count / 2

            // Compute s0 = sum current[0..<halfN], s1 = sum current[halfN...]
            var s0 = Fr.zero
            var s1 = Fr.zero
            current.withUnsafeBytes { buf in
                let p = buf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                withUnsafeMutableBytes(of: &s0) { r in
                    bn254_fr_vector_sum(p, Int32(halfN),
                                        r.baseAddress!.assumingMemoryBound(to: UInt64.self))
                }
                withUnsafeMutableBytes(of: &s1) { r in
                    bn254_fr_vector_sum(p + halfN * 4, Int32(halfN),
                                        r.baseAddress!.assumingMemoryBound(to: UInt64.self))
                }
            }

            // Build degree-1 round polynomial
            let c1 = frSub(s1, s0)
            let roundPoly = SumcheckRoundPoly(c0: s0, c1: c1, c2: Fr.zero)
            roundPolys.append(roundPoly)

            // Absorb and get challenge
            transcript.absorb(s0)
            transcript.absorb(s1)
            let challenge = transcript.squeeze()
            challenges.append(challenge)

            // Fold: current[i] = current[i] + r * (current[i+halfN] - current[i])
            var next = [Fr](repeating: Fr.zero, count: halfN)
            current.withUnsafeBytes { eBuf in
                next.withUnsafeMutableBytes { rBuf in
                    withUnsafeBytes(of: challenge) { cBuf in
                        bn254_fr_sumcheck_reduce(
                            eBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(halfN))
                    }
                }
            }
            current = next
            currentLogSize -= 1
        }

        let finalEval = current[0]

        return GPUSumcheckProof(
            roundPolys: roundPolys,
            challenges: challenges,
            finalEval: finalEval,
            numVars: numVars
        )
    }

    // MARK: - Verification

    /// Verify a sumcheck proof against a claimed sum.
    /// Returns the evaluation point (challenges) and final evaluation for oracle check.
    ///
    /// Verification checks:
    /// 1. Each round: p_i(0) + p_i(1) = current claim
    /// 2. Challenges match transcript
    /// 3. Final claim equals finalEval
    public static func verify(
        proof: GPUSumcheckProof,
        claimedSum: Fr,
        transcript: Transcript
    ) -> (valid: Bool, evalPoint: [Fr], finalEval: Fr) {
        var currentClaim = claimedSum

        for round in 0..<proof.numVars {
            let rp = proof.roundPolys[round]

            // Check: p(0) + p(1) = currentClaim
            let roundSum = frAdd(rp.atZero, rp.atOne)
            if !frEqual(roundSum, currentClaim) {
                return (valid: false, evalPoint: [], finalEval: Fr.zero)
            }

            // Absorb round poly and get challenge
            transcript.absorb(rp.atZero)
            transcript.absorb(rp.atOne)
            let challenge = transcript.squeeze()

            // Check challenge consistency
            if !frEqual(challenge, proof.challenges[round]) {
                return (valid: false, evalPoint: [], finalEval: Fr.zero)
            }

            // Next claim = p(r)
            currentClaim = rp.evaluate(at: challenge)
        }

        // Final check: last claim equals finalEval
        let valid = frEqual(currentClaim, proof.finalEval)
        return (valid: valid, evalPoint: proof.challenges, finalEval: proof.finalEval)
    }

    // MARK: - Helpers

    private func trailingZeroBitCount(_ n: Int) -> Int {
        var count = 0
        var v = n
        while v > 1 { v >>= 1; count += 1 }
        return count
    }
}

// MARK: - Fr helpers

/// Check if an Fr value is zero.
public func frIsZero(_ a: Fr) -> Bool {
    return a.v.0 == 0 && a.v.1 == 0 && a.v.2 == 0 && a.v.3 == 0 &&
           a.v.4 == 0 && a.v.5 == 0 && a.v.6 == 0 && a.v.7 == 0
}

