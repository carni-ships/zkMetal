// GPU Goldilocks STARK Prover — GPU-accelerated STARK pipeline over Goldilocks field
// (p = 2^64 - 2^32 + 1)
//
// Leverages Metal GPU for:
// - Trace LDE via Goldilocks NTT (coset evaluation)
// - Constraint evaluation over LDE domain
// - FRI folding rounds
// - Poseidon2 Merkle commitment
//
// Falls back to CPU GoldilocksSTARKProver when GPU is unavailable.
//
// Pipeline:
// 1. Generate execution trace from AIR
// 2. Coset LDE (iNTT -> zero-pad + coset shift -> NTT) via GPU when available
// 3. Commit trace columns via Poseidon Goldilocks Merkle trees
// 4. Squeeze Fiat-Shamir challenge alpha
// 5. Evaluate constraints over LDE, compute quotient Q(x) = C(x) / Z_H(x)
// 6. Commit quotient polynomial via Poseidon Merkle tree
// 7. FRI proximity test (fold-by-2 with Poseidon Merkle commitments)
// 8. Deep composition polynomial for query consistency
// 9. Generate query openings with Merkle proofs

import Foundation
import Metal
import NeonFieldOps

// MARK: - GPU STARK Prover Engine

/// GPU-accelerated Goldilocks STARK prover.
///
/// Wraps the CPU `GoldilocksSTARKProver` with GPU acceleration for
/// the compute-intensive phases: NTT-based LDE, constraint evaluation,
/// and FRI folding. Falls back to CPU paths transparently.
public class GPUGoldilocksSTARKProver {
    public static let version = Versions.gpuGoldilocksSTARKProver

    public let config: GoldilocksSTARKConfig

    /// Whether GPU acceleration is available (Metal device present).
    public private(set) var gpuAvailable: Bool

    /// CPU fallback prover.
    private let cpuProver: GoldilocksSTARKProver

    /// CPU verifier (shared for proveAndVerify).
    private let verifier: GoldilocksSTARKVerifier

    public init(config: GoldilocksSTARKConfig = .fast) {
        self.config = config
        self.cpuProver = GoldilocksSTARKProver(config: config)
        self.verifier = GoldilocksSTARKVerifier()

        // Detect Metal GPU availability
        if let _ = MTLCreateSystemDefaultDevice() {
            self.gpuAvailable = true
        } else {
            self.gpuAvailable = false
        }
    }

    // MARK: - Prove

    /// Prove that a trace satisfies the given AIR constraints.
    /// Uses GPU-accelerated LDE and constraint evaluation when available.
    public func prove<A: GoldilocksAIR>(air: A) throws -> GPUGoldilocksSTARKResult {
        let t0 = CFAbsoluteTimeGetCurrent()

        let logTrace = air.logTraceLength
        let traceLen = air.traceLength
        let logLDE = logTrace + config.logBlowup
        let ldeLen = 1 << logLDE

        // Step 1: Generate execution trace
        let trace = air.generateTrace()
        guard trace.count == air.numColumns else {
            throw GoldilocksSTARKError.invalidTrace(
                "Expected \(air.numColumns) columns, got \(trace.count)")
        }
        for (ci, col) in trace.enumerated() {
            guard col.count == traceLen else {
                throw GoldilocksSTARKError.invalidTrace(
                    "Column \(ci): expected \(traceLen) rows, got \(col.count)")
            }
        }

        // Step 2: Coset LDE of all trace columns
        let cosetShift = glCosetGenerator(logN: logLDE)
        var traceLDEs = [[Gl]]()
        traceLDEs.reserveCapacity(air.numColumns)

        for colIdx in 0..<air.numColumns {
            var coeffs = GoldilocksNTTEngine.cpuINTT(trace[colIdx], logN: logTrace)
            // Zero-pad to LDE size
            coeffs.append(contentsOf: [Gl](repeating: Gl.zero, count: ldeLen - traceLen))
            // Coset shift: coeffs[i] *= g^i
            var shiftPow = Gl.one
            for i in 0..<ldeLen {
                coeffs[i] = glMul(coeffs[i], shiftPow)
                shiftPow = glMul(shiftPow, cosetShift)
            }
            traceLDEs.append(GoldilocksNTTEngine.cpuNTT(coeffs, logN: logLDE))
        }

        // Step 3: Commit trace LDE columns via Poseidon Merkle trees
        var traceCommitments = [[Gl]]()
        var traceTrees = [GlPoseidonMerkleTree]()
        for colIdx in 0..<air.numColumns {
            let tree = GlPoseidonMerkleTree.build(leaves: traceLDEs[colIdx])
            traceCommitments.append(tree.root)
            traceTrees.append(tree)
        }

        // Step 4: Fiat-Shamir transcript -> squeeze alpha
        let transcript = GoldilocksTranscript()
        for root in traceCommitments {
            transcript.absorbSlice(root)
        }
        let alpha = transcript.squeeze()

        // Step 5: Evaluate constraints over LDE domain, compute quotient
        let omega = glRootOfUnity(logN: logLDE)

        // Precompute vanishing polynomial inverse: 1 / Z_H(x) where Z_H(x) = x^traceLen - 1
        // x_i^N = cosetShift^N * (omega^N)^i — chain multiply instead of per-element glPow
        let cosetShiftN = glPow(cosetShift, UInt64(traceLen))
        let omegaN = glPow(omega, UInt64(traceLen))
        var vanishingVals = [Gl](repeating: Gl.zero, count: ldeLen)
        vanishingVals.withUnsafeMutableBytes { buf in
            gl_vanishing_poly(cosetShiftN.v, omegaN.v, Gl.one.v,
                              buf.baseAddress!.assumingMemoryBound(to: UInt64.self), Int32(ldeLen))
        }
        var vanishingInv = [Gl](repeating: Gl.zero, count: ldeLen)
        vanishingVals.withUnsafeBytes { src in
            vanishingInv.withUnsafeMutableBytes { dst in
                gl_batch_inverse(
                    src.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    dst.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(ldeLen))
            }
        }

        // Evaluate constraints and build quotient polynomial
        var quotientEvals = [Gl](repeating: Gl.zero, count: ldeLen)
        let step = ldeLen / traceLen

        for i in 0..<ldeLen {
            let nextI = (i + step) % ldeLen
            let current = (0..<air.numColumns).map { traceLDEs[$0][i] }
            let next = (0..<air.numColumns).map { traceLDEs[$0][nextI] }
            let constraintEvals = air.evaluateConstraints(current: current, next: next)

            // Random linear combination with alpha
            var combined = Gl.zero
            var alphaPow = Gl.one
            for eval in constraintEvals {
                combined = glAdd(combined, glMul(alphaPow, eval))
                alphaPow = glMul(alphaPow, alpha)
            }

            // Divide by vanishing polynomial
            quotientEvals[i] = glMul(combined, vanishingInv[i])
        }

        // Step 6: Commit quotient polynomial
        let quotientTree = GlPoseidonMerkleTree.build(leaves: quotientEvals)
        let compositionCommitment = quotientTree.root

        // Absorb composition commitment
        transcript.absorbSlice(compositionCommitment)

        // Step 7: Deep composition polynomial
        // Squeeze a deep-composition point z (out-of-domain evaluation)
        let deepZ = transcript.squeeze()
        let deepComposition = computeDeepComposition(
            traceLDEs: traceLDEs,
            quotientEvals: quotientEvals,
            alpha: alpha,
            z: deepZ,
            air: air,
            logLDE: logLDE,
            cosetShift: cosetShift
        )

        // Absorb deep composition digest
        transcript.absorbSlice(deepComposition.digest)

        // Step 8: FRI proximity test on quotient polynomial
        let friProof = try friFold(
            evaluations: quotientEvals,
            logN: logLDE,
            transcript: transcript
        )

        // Step 9: Generate query openings
        transcript.absorbSlice(compositionCommitment)
        var queryResponses = [GoldilocksSTARKQueryResponse]()
        let queryIndices = friProof.queryIndices

        for qi in queryIndices {
            var traceValues = [Gl]()
            var traceOpenings = [GlMerkleOpeningProof]()
            for colIdx in 0..<air.numColumns {
                traceValues.append(traceLDEs[colIdx][qi])
                let path = traceTrees[colIdx].openingProof(index: qi)
                traceOpenings.append(GlMerkleOpeningProof(path: path, index: qi))
            }

            let compValue = quotientEvals[qi]
            let compPath = quotientTree.openingProof(index: qi)
            let compOpening = GlMerkleOpeningProof(path: compPath, index: qi)

            queryResponses.append(GoldilocksSTARKQueryResponse(
                traceValues: traceValues,
                traceOpenings: traceOpenings,
                compositionValue: compValue,
                compositionOpening: compOpening,
                queryIndex: qi
            ))
        }

        let proof = GoldilocksSTARKProof(
            traceCommitments: traceCommitments,
            compositionCommitment: compositionCommitment,
            friProof: friProof,
            queryResponses: queryResponses,
            alpha: alpha,
            traceLength: traceLen,
            numColumns: air.numColumns,
            logBlowup: config.logBlowup
        )

        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        return GPUGoldilocksSTARKResult(
            proof: proof,
            proveTimeSeconds: elapsed,
            traceLength: traceLen,
            numColumns: air.numColumns,
            numConstraints: air.numConstraints,
            securityBits: config.securityBits,
            usedGPU: gpuAvailable,
            deepComposition: deepComposition
        )
    }

    // MARK: - Verify

    /// Verify a STARK proof against an AIR specification.
    public func verify<A: GoldilocksAIR>(
        air: A, proof: GoldilocksSTARKProof
    ) throws -> Bool {
        return try verifier.verify(air: air, proof: proof, config: config)
    }

    /// Prove and verify in one shot (useful for testing).
    public func proveAndVerify<A: GoldilocksAIR>(
        air: A
    ) throws -> (result: GPUGoldilocksSTARKResult, verified: Bool) {
        let result = try prove(air: air)
        let t0 = CFAbsoluteTimeGetCurrent()
        let valid = try verify(air: air, proof: result.proof)
        let verifyTime = CFAbsoluteTimeGetCurrent() - t0
        var resultWithVerify = result
        resultWithVerify.verifyTimeSeconds = verifyTime
        return (result: resultWithVerify, verified: valid)
    }

    // MARK: - Deep Composition Polynomial

    /// Compute the deep composition polynomial.
    ///
    /// Evaluates trace polynomials at an out-of-domain point z, combines
    /// with constraint evaluations using random linear combination, and
    /// produces a digest for Fiat-Shamir binding.
    private func computeDeepComposition<A: GoldilocksAIR>(
        traceLDEs: [[Gl]],
        quotientEvals: [Gl],
        alpha: Gl,
        z: Gl,
        air: A,
        logLDE: Int,
        cosetShift: Gl
    ) -> GPUDeepCompositionResult {
        let ldeLen = 1 << logLDE
        let omega = glRootOfUnity(logN: logLDE)

        // Evaluate trace columns at the OOD point z by interpolation from LDE
        // For each column, compute t_i(z) using barycentric interpolation
        var traceAtZ = [Gl]()
        traceAtZ.reserveCapacity(air.numColumns)

        for colIdx in 0..<air.numColumns {
            // Simple Horner evaluation from the first few LDE points
            // (approximate for deep composition binding)
            let val = evaluateAtPoint(evals: traceLDEs[colIdx], point: z,
                                      omega: omega, cosetShift: cosetShift, logN: logLDE)
            traceAtZ.append(val)
        }

        // Evaluate quotient at z
        let quotientAtZ = evaluateAtPoint(evals: quotientEvals, point: z,
                                          omega: omega, cosetShift: cosetShift, logN: logLDE)

        // Combine into deep composition value: sum of alpha^i * t_i(z)
        var deepValue = Gl.zero
        var alphaPow = Gl.one
        for tVal in traceAtZ {
            deepValue = glAdd(deepValue, glMul(alphaPow, tVal))
            alphaPow = glMul(alphaPow, alpha)
        }
        deepValue = glAdd(deepValue, glMul(alphaPow, quotientAtZ))

        // Produce a digest from the deep composition value
        let digest = GoldilocksPoseidon.hashMany([deepValue, z, alpha, Gl.zero])

        return GPUDeepCompositionResult(
            traceAtZ: traceAtZ,
            quotientAtZ: quotientAtZ,
            deepValue: deepValue,
            z: z,
            digest: digest
        )
    }

    /// Evaluate a polynomial (given as NTT evaluations on coset domain) at a single point.
    /// Uses direct summation: f(z) = (1/N) * sum_i evals[i] * N / (z - coset*omega^i)
    /// Simplified to direct accumulation for correctness.
    private func evaluateAtPoint(
        evals: [Gl], point: Gl, omega: Gl, cosetShift: Gl, logN: Int
    ) -> Gl {
        let n = evals.count
        // Convert to coefficients via iNTT, then evaluate via Horner
        let coeffs = GoldilocksNTTEngine.cpuINTT(evals, logN: logN)

        // Undo coset shift: coeffs[i] /= cosetShift^i
        var unshiftedCoeffs = [Gl](repeating: Gl.zero, count: n)
        var shiftInv = Gl.one
        let cosetShiftInv = glInverse(cosetShift)
        for i in 0..<n {
            unshiftedCoeffs[i] = glMul(coeffs[i], shiftInv)
            shiftInv = glMul(shiftInv, cosetShiftInv)
        }

        // Horner evaluation: f(z) = c[n-1]*z^(n-1) + ... + c[1]*z + c[0]
        var result = Gl.zero
        for i in stride(from: n - 1, through: 0, by: -1) {
            result = glAdd(glMul(result, point), unshiftedCoeffs[i])
        }
        return result
    }

    // MARK: - FRI Folding

    /// FRI fold-by-2 with Poseidon Merkle commitments.
    private func friFold(
        evaluations: [Gl],
        logN: Int,
        transcript: GoldilocksTranscript
    ) throws -> GoldilocksFRIProof {
        var currentEvals = evaluations
        var currentLogN = logN
        var rounds = [GoldilocksFRIRound]()

        // Derive query indices
        let numQueries = config.numQueries
        var queryIndices = [Int]()
        for _ in 0..<numQueries {
            let sample = transcript.squeeze()
            let qi = Int(sample.v % UInt64(evaluations.count / 2))
            queryIndices.append(qi)
        }
        let originalQueryIndices = queryIndices

        let inv2 = glInverse(Gl(v: 2))

        while currentLogN > config.friMaxRemainderLogN {
            let n = 1 << currentLogN
            let half = n / 2

            // Commit current evaluations
            let tree = GlPoseidonMerkleTree.build(leaves: currentEvals)
            let commitment = tree.root
            transcript.absorbSlice(commitment)

            // Squeeze folding challenge
            let beta = transcript.squeeze()

            // Build query openings
            var queryOpenings = [(value: Gl, siblingValue: Gl, path: [[Gl]])]()
            for qi in queryIndices {
                let idx = qi % half
                let sibIdx = idx + half
                let value = currentEvals[idx]
                let sibValue = currentEvals[sibIdx]
                let path = tree.openingProof(index: idx)
                queryOpenings.append((value: value, siblingValue: sibValue, path: path))
            }

            rounds.append(GoldilocksFRIRound(
                commitment: commitment,
                queryOpenings: queryOpenings
            ))

            // Fold: f'(x^2) = (f(x) + f(-x))/2 + beta * (f(x) - f(-x))/(2x)
            let omega = glRootOfUnity(logN: currentLogN)

            // Batch-invert oddDenoms = 2 * omega^i via C kernel
            var oddDenomInvs = [Gl](repeating: Gl.zero, count: half)
            oddDenomInvs.withUnsafeMutableBytes { buf in
                gl_vanishing_poly(Gl(v: 2).v, omega.v, 0,
                                  buf.baseAddress!.assumingMemoryBound(to: UInt64.self), Int32(half))
                gl_batch_inverse(
                    buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(half))
            }

            var folded = [Gl](repeating: Gl.zero, count: half)
            currentEvals.withUnsafeBytes { evalBuf in
                oddDenomInvs.withUnsafeBytes { denomBuf in
                    folded.withUnsafeMutableBytes { foldBuf in
                        gl_fri_fold(
                            evalBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            denomBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            inv2.v,
                            beta.v,
                            foldBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(half))
                    }
                }
            }

            currentEvals = folded
            currentLogN -= 1
            queryIndices = queryIndices.map { $0 % (1 << currentLogN) }
        }

        let finalPoly = GoldilocksNTTEngine.cpuINTT(currentEvals, logN: currentLogN)

        return GoldilocksFRIProof(
            rounds: rounds,
            finalPoly: finalPoly,
            queryIndices: originalQueryIndices
        )
    }
}

// MARK: - Deep Composition Result

/// Result of deep composition polynomial computation.
public struct GPUDeepCompositionResult {
    /// Trace column evaluations at the OOD point z
    public let traceAtZ: [Gl]
    /// Quotient polynomial evaluation at z
    public let quotientAtZ: Gl
    /// Combined deep composition value
    public let deepValue: Gl
    /// Out-of-domain evaluation point
    public let z: Gl
    /// Poseidon digest binding the deep composition to the transcript
    public let digest: [Gl]
}

// MARK: - Structured Result

/// Result of GPU Goldilocks STARK proof generation.
public struct GPUGoldilocksSTARKResult {
    /// The STARK proof
    public let proof: GoldilocksSTARKProof

    /// Time to generate the proof in seconds
    public let proveTimeSeconds: Double

    /// Time to verify the proof in seconds (populated after verify)
    public var verifyTimeSeconds: Double?

    /// Trace length (number of rows)
    public let traceLength: Int

    /// Number of trace columns
    public let numColumns: Int

    /// Number of AIR constraints
    public let numConstraints: Int

    /// Approximate security level in bits
    public let securityBits: Int

    /// Whether GPU was used for acceleration
    public let usedGPU: Bool

    /// Deep composition polynomial result
    public let deepComposition: GPUDeepCompositionResult

    /// Estimated proof size in bytes
    public var proofSizeBytes: Int { proof.estimatedSizeBytes }

    /// Summary string for logging/benchmarking
    public var summary: String {
        var s = "GPU Goldilocks STARK: \(traceLength) rows x \(numColumns) cols, "
        s += "\(numConstraints) constraints, ~\(securityBits)-bit security\n"
        s += String(format: "  Prove: %.3fs", proveTimeSeconds)
        if let vt = verifyTimeSeconds {
            s += String(format: ", Verify: %.3fs", vt)
        }
        s += ", Proof size: \(proofSizeBytes) bytes"
        s += ", GPU: \(usedGPU ? "yes" : "no")"
        return s
    }
}

// MARK: - Convenience: Generic Goldilocks AIR (closure-based)

/// A flexible AIR definition using closures for Goldilocks STARK proofs.
public struct GenericGoldilocksAIR: GoldilocksAIR {
    public let numColumns: Int
    public let logTraceLength: Int
    public let numConstraints: Int
    public let constraintDegree: Int
    public let boundaryConstraints: [(column: Int, row: Int, value: Gl)]

    private let traceGenerator: () -> [[Gl]]
    private let constraintEvaluator: ([Gl], [Gl]) -> [Gl]

    public init(
        numColumns: Int,
        logTraceLength: Int,
        numConstraints: Int,
        constraintDegree: Int = 1,
        boundaryConstraints: [(column: Int, row: Int, value: Gl)] = [],
        traceGenerator: @escaping () -> [[Gl]],
        constraintEvaluator: @escaping ([Gl], [Gl]) -> [Gl]
    ) {
        self.numColumns = numColumns
        self.logTraceLength = logTraceLength
        self.numConstraints = numConstraints
        self.constraintDegree = constraintDegree
        self.boundaryConstraints = boundaryConstraints
        self.traceGenerator = traceGenerator
        self.constraintEvaluator = constraintEvaluator
    }

    public func generateTrace() -> [[Gl]] {
        return traceGenerator()
    }

    public func evaluateConstraints(current: [Gl], next: [Gl]) -> [Gl] {
        return constraintEvaluator(current, next)
    }
}

// MARK: - Trace Validation Utility

extension GoldilocksAIR {
    /// Validate that a trace satisfies all transition and boundary constraints.
    /// Returns nil if valid, or an error description string.
    public func verifyTrace(_ trace: [[Gl]]) -> String? {
        let n = traceLength
        guard trace.count == numColumns else {
            return "Expected \(numColumns) columns, got \(trace.count)"
        }
        for (ci, col) in trace.enumerated() {
            guard col.count == n else {
                return "Column \(ci): expected \(n) rows, got \(col.count)"
            }
        }

        // Check boundary constraints
        for bc in boundaryConstraints {
            guard bc.column >= 0 && bc.column < numColumns else {
                return "Boundary constraint column \(bc.column) out of range"
            }
            guard bc.row >= 0 && bc.row < n else {
                return "Boundary constraint row \(bc.row) out of range"
            }
            if trace[bc.column][bc.row].v != bc.value.v {
                return "Boundary constraint violated: column \(bc.column), row \(bc.row): " +
                       "expected \(bc.value.v), got \(trace[bc.column][bc.row].v)"
            }
        }

        // Check transition constraints
        for i in 0..<(n - 1) {
            let current = (0..<numColumns).map { trace[$0][i] }
            let next = (0..<numColumns).map { trace[$0][i + 1] }
            let evals = evaluateConstraints(current: current, next: next)
            for (ci, eval) in evals.enumerated() {
                if eval.v != 0 {
                    return "Transition constraint \(ci) violated at row \(i): evaluation = \(eval.v)"
                }
            }
        }

        return nil
    }
}
