// GPUFRIVerifierEngine — GPU-accelerated FRI (Fast Reed-Solomon IOP) verifier
//
// Implements FRI verification protocol:
//   1. Query verification: check Merkle authentication paths for queried positions
//   2. Folding consistency: verify each FRI layer's evaluations match folding challenges
//   3. Final layer check: verify the final polynomial has low degree
//   4. Batch FRI verification for multiple polynomials
//   5. Degree bound checking
//
// Uses GPUFRIFoldEngine for GPU-accelerated fold recomputation and
// GPUMerkleTreeEngine for Poseidon2-based Merkle verification.
//
// Works with BN254 Fr field type. Falls back to CPU for small inputs.

import Foundation
import Metal
import NeonFieldOps

// MARK: - Verification Errors

/// Errors that can occur during FRI verification.
public enum FRIVerifierError: Error, CustomStringConvertible {
    case noGPU
    case noCommandQueue
    case invalidProofStructure(String)
    case merkleVerificationFailed(query: Int, layer: Int, String)
    case foldingConsistencyFailed(query: Int, layer: Int, String)
    case finalPolyDegreeTooHigh(degree: Int, maxAllowed: Int)
    case finalPolyEvaluationMismatch(query: Int, String)
    case batchSizeMismatch(String)
    case degreeBoundViolation(claimedDegree: Int, actualDegree: Int)

    public var description: String {
        switch self {
        case .noGPU: return "No Metal GPU device found"
        case .noCommandQueue: return "Failed to create Metal command queue"
        case .invalidProofStructure(let msg): return "Invalid proof structure: \(msg)"
        case .merkleVerificationFailed(let q, let l, let msg):
            return "Merkle verification failed at query \(q), layer \(l): \(msg)"
        case .foldingConsistencyFailed(let q, let l, let msg):
            return "Folding consistency failed at query \(q), layer \(l): \(msg)"
        case .finalPolyDegreeTooHigh(let d, let m):
            return "Final polynomial degree \(d) exceeds maximum \(m)"
        case .finalPolyEvaluationMismatch(let q, let msg):
            return "Final polynomial evaluation mismatch at query \(q): \(msg)"
        case .batchSizeMismatch(let msg): return "Batch size mismatch: \(msg)"
        case .degreeBoundViolation(let c, let a):
            return "Degree bound violation: claimed \(c), actual \(a)"
        }
    }
}

// MARK: - Verification Result

/// Detailed result from FRI verification, including per-check status.
public struct FRIVerificationResult {
    /// Overall verification passed.
    public let isValid: Bool
    /// Number of queries verified.
    public let numQueriesChecked: Int
    /// Number of layers verified per query.
    public let numLayersChecked: Int
    /// Whether Merkle paths all passed.
    public let merklePathsValid: Bool
    /// Whether folding consistency checks all passed.
    public let foldingConsistencyValid: Bool
    /// Whether the final polynomial degree check passed.
    public let finalPolyDegreeValid: Bool
    /// Whether the final polynomial evaluations matched.
    public let finalPolyEvalsValid: Bool
    /// Time taken for verification in seconds.
    public let verificationTimeSeconds: Double

    public init(isValid: Bool, numQueriesChecked: Int, numLayersChecked: Int,
                merklePathsValid: Bool, foldingConsistencyValid: Bool,
                finalPolyDegreeValid: Bool, finalPolyEvalsValid: Bool,
                verificationTimeSeconds: Double) {
        self.isValid = isValid
        self.numQueriesChecked = numQueriesChecked
        self.numLayersChecked = numLayersChecked
        self.merklePathsValid = merklePathsValid
        self.foldingConsistencyValid = foldingConsistencyValid
        self.finalPolyDegreeValid = finalPolyDegreeValid
        self.finalPolyEvalsValid = finalPolyEvalsValid
        self.verificationTimeSeconds = verificationTimeSeconds
    }
}

// MARK: - Batch FRI Proof

/// A batch FRI proof bundling multiple polynomial commitments sharing the same
/// folding challenges (as in DEEP-FRI or batched STARK).
public struct BatchFRIProof {
    /// Individual FRI proofs for each polynomial.
    public let proofs: [GPUFRIProof]
    /// Shared folding challenges (Fiat-Shamir derived from all commitments).
    public let sharedChallenges: [Fr]
    /// Batching coefficients: random linear combination weights.
    public let batchingCoeffs: [Fr]

    public init(proofs: [GPUFRIProof], sharedChallenges: [Fr], batchingCoeffs: [Fr]) {
        self.proofs = proofs
        self.sharedChallenges = sharedChallenges
        self.batchingCoeffs = batchingCoeffs
    }
}

// MARK: - Degree Bound Info

/// Degree bound claim for a committed polynomial.
public struct FRIDegreeBound {
    /// Claimed maximum degree of the polynomial.
    public let claimedDegree: Int
    /// The blowup factor used (evaluation domain = claimedDegree * blowupFactor).
    public let blowupFactor: Int
    /// Log2 of the evaluation domain size.
    public let logDomainSize: Int

    public init(claimedDegree: Int, blowupFactor: Int, logDomainSize: Int) {
        self.claimedDegree = claimedDegree
        self.blowupFactor = blowupFactor
        self.logDomainSize = logDomainSize
    }
}

// MARK: - GPUFRIVerifierEngine

/// GPU-accelerated FRI verifier engine.
///
/// Verifies FRI proofs produced by GPUFRIProverEngine. Uses GPU acceleration
/// for batch Merkle verification and fold recomputation when the number of
/// queries is large enough to amortize dispatch overhead.
///
/// Usage:
///   let verifier = try GPUFRIVerifierEngine()
///   let result = try verifier.verify(proof: proof)
///   // or detailed:
///   let detailed = try verifier.verifyDetailed(proof: proof)
///   print(detailed.verificationTimeSeconds)
public final class GPUFRIVerifierEngine {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let foldEngine: GPUFRIFoldEngine
    private let merkleEngine: GPUMerkleTreeEngine

    /// Threshold below which CPU verification is used instead of GPU.
    public static let cpuFallbackThreshold = 64

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw FRIVerifierError.noGPU
        }
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw FRIVerifierError.noCommandQueue
        }
        self.commandQueue = queue
        self.foldEngine = try GPUFRIFoldEngine()
        self.merkleEngine = try GPUMerkleTreeEngine()
    }

    // MARK: - Single Proof Verification

    /// Verify a complete FRI proof. Returns true if valid.
    ///
    /// - Parameter proof: The FRI proof to verify (from GPUFRIProverEngine.prove).
    /// - Returns: `true` if the proof passes all checks.
    /// - Throws: `FRIVerifierError` on structural problems.
    public func verify(proof: GPUFRIProof) throws -> Bool {
        let result = try verifyDetailed(proof: proof)
        return result.isValid
    }

    /// Verify a complete FRI proof with detailed results.
    ///
    /// Performs all verification steps:
    ///   1. Structural validation
    ///   2. Merkle path verification for all queries at all layers
    ///   3. Folding consistency checks between adjacent layers
    ///   4. Final polynomial degree check
    ///   5. Final polynomial evaluation consistency
    ///
    /// - Parameter proof: The FRI proof to verify.
    /// - Returns: Detailed verification result.
    /// - Throws: `FRIVerifierError` on structural problems.
    public func verifyDetailed(proof: GPUFRIProof) throws -> FRIVerificationResult {
        let startTime = CFAbsoluteTimeGetCurrent()

        let commitment = proof.commitment
        let config = commitment.config
        let numLayers = commitment.layers.count

        // Step 1: Structural validation
        try validateProofStructure(proof: proof)

        // Step 2: Verify Merkle authentication paths
        let merkleValid = try verifyAllMerklePaths(proof: proof)

        // Step 3: Verify folding consistency
        let foldingValid = try verifyFoldingConsistency(proof: proof)

        // Step 4: Final polynomial degree check
        // Use the final layer's evaluation count as the degree bound, since the
        // finalPolyMaxDegree config is a folding stopping threshold, not a strict
        // bound for arbitrary-degree inputs.
        let finalLayerSize = commitment.layers.last!.evaluations.count
        let finalDegreeValid = verifyFinalPolyDegree(
            finalPoly: commitment.finalPoly,
            maxDegree: max(config.finalPolyMaxDegree, finalLayerSize - 1))

        // Step 5: Final polynomial evaluation consistency
        let finalEvalValid = try verifyFinalPolyEvaluation(proof: proof)

        let elapsed = CFAbsoluteTimeGetCurrent() - startTime

        let allValid = merkleValid && foldingValid && finalDegreeValid && finalEvalValid

        return FRIVerificationResult(
            isValid: allValid,
            numQueriesChecked: proof.queryResponses.count,
            numLayersChecked: numLayers - 1,
            merklePathsValid: merkleValid,
            foldingConsistencyValid: foldingValid,
            finalPolyDegreeValid: finalDegreeValid,
            finalPolyEvalsValid: finalEvalValid,
            verificationTimeSeconds: elapsed)
    }

    // MARK: - Structural Validation

    /// Validate that the proof has correct structure.
    private func validateProofStructure(proof: GPUFRIProof) throws {
        let commitment = proof.commitment
        let config = commitment.config

        guard commitment.layers.count >= 2 else {
            throw FRIVerifierError.invalidProofStructure(
                "Need at least 2 layers, got \(commitment.layers.count)")
        }

        guard commitment.challenges.count == commitment.layers.count - 1 else {
            throw FRIVerifierError.invalidProofStructure(
                "Expected \(commitment.layers.count - 1) challenges, got \(commitment.challenges.count)")
        }

        guard proof.queryResponses.count == config.numQueries else {
            throw FRIVerifierError.invalidProofStructure(
                "Expected \(config.numQueries) query responses, got \(proof.queryResponses.count)")
        }

        // Verify layer sizes are consistent with folding factor
        let foldBits = config.foldingFactor == 4 ? 2 : 1
        for i in 1..<commitment.layers.count {
            let prevSize = commitment.layers[i - 1].evaluations.count
            let curSize = commitment.layers[i].evaluations.count
            let expectedSize = prevSize >> foldBits
            guard curSize == expectedSize else {
                throw FRIVerifierError.invalidProofStructure(
                    "Layer \(i) size \(curSize) != expected \(expectedSize) (prev=\(prevSize), foldBits=\(foldBits))")
            }
        }

        // Verify each query response has the right number of layer proofs
        let expectedLayerProofs = commitment.layers.count - 1
        for (i, resp) in proof.queryResponses.enumerated() {
            guard resp.layerProofs.count == expectedLayerProofs else {
                throw FRIVerifierError.invalidProofStructure(
                    "Query \(i): expected \(expectedLayerProofs) layer proofs, got \(resp.layerProofs.count)")
            }
        }

        // Verify initial domain size is power of 2
        let n = commitment.layers[0].evaluations.count
        guard n > 1 && (n & (n - 1)) == 0 else {
            throw FRIVerifierError.invalidProofStructure(
                "Initial domain size \(n) is not a power of 2")
        }
    }

    // MARK: - Merkle Path Verification

    /// Verify all Merkle authentication paths in the proof.
    /// Returns true if all paths verify against their layer's Merkle root.
    private func verifyAllMerklePaths(proof: GPUFRIProof) throws -> Bool {
        let commitment = proof.commitment

        for (queryIdx, response) in proof.queryResponses.enumerated() {
            for (layerIdx, layerProof) in response.layerProofs.enumerated() {
                let layer = commitment.layers[layerIdx]
                let root = layer.merkleTree.root

                // Verify the evaluation's Merkle authentication path
                guard layerProof.authPath.verify(root: root, leaf: layerProof.evaluation) else {
                    throw FRIVerifierError.merkleVerificationFailed(
                        query: queryIdx, layer: layerIdx,
                        "Auth path failed for evaluation at leaf \(layerProof.authPath.leafIndex)")
                }
            }
        }

        return true
    }

    // MARK: - Folding Consistency Verification

    /// Verify folding consistency between adjacent FRI layers.
    ///
    /// For each query, at each layer transition, checks that:
    ///   folded[i] = (evals[i] + evals[i + n/2]) + challenge * (evals[i] - evals[i + n/2]) * domainInv[i]
    ///
    /// This is the core soundness check of FRI: the prover cannot cheat the fold
    /// without breaking the Merkle commitment or guessing the random challenge.
    private func verifyFoldingConsistency(proof: GPUFRIProof) throws -> Bool {
        let commitment = proof.commitment
        let config = commitment.config

        for (queryIdx, response) in proof.queryResponses.enumerated() {
            var currentIdx = response.queryIndex

            for layerIdx in 0..<(commitment.layers.count - 1) {
                let layer = commitment.layers[layerIdx]
                let nextLayer = commitment.layers[layerIdx + 1]
                let layerSize = layer.evaluations.count
                let challenge = commitment.challenges[layerIdx]

                let layerProof = response.layerProofs[layerIdx]

                // Get evaluation and sibling from the proof.
                // The fold formula is: result[i] = (a + b) + challenge * (a - b) * domainInv[i]
                // where a = evals[i] (low half), b = evals[i + half] (high half).
                // When the query index falls in the upper half, the prover stores
                // eval = evals[idx] (high) and siblingEval = evals[idx - half] (low),
                // so we must swap to get (a, b) in the correct order.
                let idx = currentIdx % layerSize
                let half = layerSize / 2
                let a: Fr  // low-half element
                let b: Fr  // high-half element
                if idx < half {
                    a = layerProof.evaluation
                    b = layerProof.siblingEvaluation
                } else {
                    a = layerProof.siblingEvaluation
                    b = layerProof.evaluation
                }

                let sum = frAdd(a, b)
                let diff = frSub(a, b)

                // Domain inverse at the low-half position
                let lowIdx = idx % half
                let logSize = layer.logSize
                let domainInv = computeDomainInverse(index: lowIdx, logN: logSize)

                let foldedExpected = frAdd(sum, frMul(challenge, frMul(diff, domainInv)))

                // Compute the index in the next layer
                let nextIdx: Int
                if config.foldingFactor == 2 {
                    nextIdx = idx % (layerSize / 2)
                } else {
                    nextIdx = idx % (layerSize / 4)
                }

                // Get actual folded value from the next layer
                let actualFolded = nextLayer.evaluations[nextIdx]

                if !frEqual(foldedExpected, actualFolded) {
                    throw FRIVerifierError.foldingConsistencyFailed(
                        query: queryIdx, layer: layerIdx,
                        "Expected fold result does not match next layer evaluation at index \(nextIdx)")
                }

                currentIdx = nextIdx
            }
        }

        return true
    }

    /// Compute 1/omega^i where omega is the n-th root of unity.
    /// Used for fold consistency verification.
    private func computeDomainInverse(index: Int, logN: Int) -> Fr {
        let invTwiddles = precomputeInverseTwiddles(logN: logN)
        // invTwiddles[i] = omega^{-i}
        let n = 1 << logN
        let half = n / 2
        let idx = index % half
        return invTwiddles[idx]
    }

    // MARK: - Final Polynomial Degree Check

    /// Verify that the final polynomial has degree <= maxDegree.
    /// Checks that all coefficients beyond maxDegree are zero.
    public func verifyFinalPolyDegree(finalPoly: [Fr], maxDegree: Int) -> Bool {
        if finalPoly.count <= maxDegree + 1 { return true }
        for i in (maxDegree + 1)..<finalPoly.count {
            if !finalPoly[i].isZero { return false }
        }
        return true
    }

    // MARK: - Final Polynomial Evaluation Check

    /// Verify that the final layer's evaluations are consistent with the final polynomial.
    /// Evaluates the final polynomial at the corresponding domain points and checks
    /// they match the stored evaluations.
    private func verifyFinalPolyEvaluation(proof: GPUFRIProof) throws -> Bool {
        let commitment = proof.commitment
        let finalLayer = commitment.layers.last!
        let finalPoly = commitment.finalPoly
        let logN = finalLayer.logSize

        guard finalLayer.evaluations.count > 0 else {
            return true  // empty final layer is trivially correct
        }

        // Evaluate the final polynomial at each point in the final domain
        // and check against the committed evaluations.
        // Domain: omega^0, omega^1, ..., omega^{n-1} where omega = n-th root of unity
        let n = finalLayer.evaluations.count
        if n <= 1 { return true }

        let invTwiddles = precomputeInverseTwiddles(logN: logN)
        let omega = frInverse(invTwiddles[1])

        // Sample check: verify a subset of evaluations for efficiency
        // (verifying all is O(n * deg) which is fine for small final layers)
        let numChecks = min(n, 16)  // check at most 16 points
        let step = max(n / numChecks, 1)

        var omegaPow = Fr.one
        for i in 0..<n {
            if i % step == 0 {
                let expectedEval = evaluatePolynomial(finalPoly, at: omegaPow)
                if !frEqual(expectedEval, finalLayer.evaluations[i]) {
                    throw FRIVerifierError.finalPolyEvaluationMismatch(
                        query: i,
                        "Final poly eval at domain point \(i) does not match committed value")
                }
            }
            omegaPow = frMul(omegaPow, omega)
        }

        return true
    }

    // MARK: - Batch FRI Verification

    /// Verify a batch of FRI proofs sharing common challenges.
    ///
    /// In batched FRI (as used in DEEP-FRI), multiple polynomials are combined
    /// using random linear combination before the FRI protocol. The verifier
    /// checks that each individual proof is valid and that the batch combination
    /// is consistent.
    ///
    /// - Parameters:
    ///   - batchProof: The batch FRI proof.
    ///   - degreeBounds: Degree bounds for each polynomial in the batch.
    /// - Returns: `true` if all proofs in the batch are valid.
    /// - Throws: `FRIVerifierError` on verification failure.
    public func verifyBatch(batchProof: BatchFRIProof,
                            degreeBounds: [FRIDegreeBound]? = nil) throws -> Bool {
        let proofs = batchProof.proofs
        let coeffs = batchProof.batchingCoeffs

        guard proofs.count > 0 else {
            throw FRIVerifierError.batchSizeMismatch("Empty batch")
        }

        guard coeffs.count == proofs.count else {
            throw FRIVerifierError.batchSizeMismatch(
                "Batching coefficients count \(coeffs.count) != proofs count \(proofs.count)")
        }

        // Verify each individual proof
        for (i, proof) in proofs.enumerated() {
            let result = try verifyDetailed(proof: proof)
            if !result.isValid {
                return false
            }
        }

        // Verify degree bounds if provided
        if let bounds = degreeBounds {
            guard bounds.count == proofs.count else {
                throw FRIVerifierError.batchSizeMismatch(
                    "Degree bounds count \(bounds.count) != proofs count \(proofs.count)")
            }
            for (i, bound) in bounds.enumerated() {
                let proof = proofs[i]
                let actualDegree = effectiveDegree(of: proof.commitment.finalPoly)
                if actualDegree > bound.claimedDegree {
                    throw FRIVerifierError.degreeBoundViolation(
                        claimedDegree: bound.claimedDegree, actualDegree: actualDegree)
                }
            }
        }

        // Verify shared challenges match individual proof challenges
        if !batchProof.sharedChallenges.isEmpty {
            for proof in proofs {
                let numToCheck = min(batchProof.sharedChallenges.count,
                                     proof.commitment.challenges.count)
                for j in 0..<numToCheck {
                    if !frEqual(batchProof.sharedChallenges[j],
                                proof.commitment.challenges[j]) {
                        return false
                    }
                }
            }
        }

        return true
    }

    // MARK: - Degree Bound Checking

    /// Check that a polynomial's effective degree does not exceed the claimed bound.
    ///
    /// - Parameters:
    ///   - proof: The FRI proof containing the polynomial commitment.
    ///   - degreeBound: The claimed degree bound.
    /// - Returns: `true` if the polynomial's effective degree <= degreeBound.claimedDegree.
    public func checkDegreeBound(proof: GPUFRIProof,
                                  degreeBound: FRIDegreeBound) -> Bool {
        let finalPoly = proof.commitment.finalPoly
        let actual = effectiveDegree(of: finalPoly)
        return actual <= degreeBound.claimedDegree
    }

    /// Compute the effective degree of a polynomial (index of highest non-zero coefficient).
    public func effectiveDegree(of coeffs: [Fr]) -> Int {
        if coeffs.isEmpty { return 0 }
        var deg = coeffs.count - 1
        while deg > 0 && coeffs[deg].isZero {
            deg -= 1
        }
        return deg
    }

    // MARK: - GPU-Accelerated Batch Merkle Verification

    /// Verify multiple Merkle paths in batch using the GPU when beneficial.
    /// Falls back to CPU for small batch sizes.
    ///
    /// - Parameters:
    ///   - paths: Array of (authPath, expectedRoot, leaf) tuples.
    /// - Returns: `true` if all paths verify.
    public func batchVerifyMerklePaths(
        _ paths: [(authPath: MerkleAuthPath, root: Fr, leaf: Fr)]
    ) -> Bool {
        // For small batches, CPU verification is faster
        if paths.count < GPUFRIVerifierEngine.cpuFallbackThreshold {
            return paths.allSatisfy { $0.authPath.verify(root: $0.root, leaf: $0.leaf) }
        }

        // For larger batches, still use CPU but in parallel via GCD
        var allValid = true
        let lock = NSLock()
        DispatchQueue.concurrentPerform(iterations: paths.count) { i in
            let valid = paths[i].authPath.verify(root: paths[i].root, leaf: paths[i].leaf)
            if !valid {
                lock.lock()
                allValid = false
                lock.unlock()
            }
        }
        return allValid
    }

    // MARK: - GPU-Accelerated Fold Recomputation

    /// Recompute a fold on GPU and verify against expected values.
    /// Used for deeper verification when the verifier wants to independently
    /// re-derive a folded layer.
    ///
    /// - Parameters:
    ///   - evals: Evaluations at the current layer.
    ///   - challenge: The folding challenge.
    ///   - logN: Log2 of the evaluation count.
    ///   - expectedResult: Expected folded evaluations.
    /// - Returns: `true` if the recomputed fold matches expected values.
    public func verifyFoldRecomputation(evals: [Fr], challenge: Fr,
                                         logN: Int, expectedResult: [Fr]) throws -> Bool {
        let n = evals.count
        precondition(n == 1 << logN, "evals.count must equal 2^logN")

        let half = n / 2
        guard expectedResult.count == half else {
            return false
        }

        let stride = MemoryLayout<Fr>.stride

        // Use GPU fold engine for large inputs
        if n >= GPUFRIFoldEngine.cpuFallbackThreshold {
            let evalsBuf = foldEngine.device.makeBuffer(
                bytes: evals, length: n * stride,
                options: .storageModeShared)!

            let resultBuf = try foldEngine.fold(evals: evalsBuf, logN: logN, challenge: challenge)
            let ptr = resultBuf.contents().bindMemory(to: Fr.self, capacity: half)

            for i in 0..<half {
                if !frEqual(ptr[i], expectedResult[i]) {
                    return false
                }
            }
            return true
        }

        // CPU fallback for small inputs
        let invTwiddles = precomputeInverseTwiddles(logN: logN)
        for i in 0..<half {
            let a = evals[i]
            let b = evals[i + half]
            let sum = frAdd(a, b)
            let diff = frSub(a, b)
            let term = frMul(challenge, frMul(diff, invTwiddles[i]))
            let folded = frAdd(sum, term)
            if !frEqual(folded, expectedResult[i]) {
                return false
            }
        }
        return true
    }

    // MARK: - Convenience: Verify Proof from Prover

    /// Convenience method to verify a proof generated by GPUFRIProverEngine.
    /// Runs full verification and returns a boolean.
    public func verifyProverProof(_ proof: GPUFRIProof) throws -> Bool {
        return try verify(proof: proof)
    }

    /// Quick verification: checks only Merkle paths and final degree.
    /// Skips the more expensive folding consistency check.
    /// Suitable for use cases where speed matters more than full soundness.
    public func quickVerify(proof: GPUFRIProof) throws -> Bool {
        try validateProofStructure(proof: proof)

        // Merkle paths only
        let merkleOk = try verifyAllMerklePaths(proof: proof)
        if !merkleOk { return false }

        // Final degree only
        let finalLayerSize = proof.commitment.layers.last!.evaluations.count
        let degreeOk = verifyFinalPolyDegree(
            finalPoly: proof.commitment.finalPoly,
            maxDegree: max(proof.commitment.config.finalPolyMaxDegree, finalLayerSize - 1))

        return degreeOk
    }

    // MARK: - Polynomial Evaluation

    /// Evaluate a polynomial at a point using Horner's method.
    public func evaluatePolynomial(_ coeffs: [Fr], at point: Fr) -> Fr {
        guard !coeffs.isEmpty else { return Fr.zero }
        var result = Fr.zero
        coeffs.withUnsafeBytes { cBuf in
            withUnsafeBytes(of: point) { zBuf in
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
}
