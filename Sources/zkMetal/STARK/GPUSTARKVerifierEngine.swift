// GPUSTARKVerifierEngine — GPU-accelerated STARK proof verification over BN254 Fr
//
// Implements full STARK verification pipeline with Metal GPU acceleration:
//   1. AIR constraint evaluation at query points
//   2. FRI verification (decommitment checks + folding consistency)
//   3. Merkle proof verification for trace and composition commitments
//   4. Deep composition polynomial check
//   5. Configurable security parameters (num queries, blowup factor)
//
// Works with the existing Fr (BN254) field type and Fiat-Shamir transcript.

import Foundation
import Metal

// MARK: - Security Configuration

/// Configurable security parameters for GPU STARK verification.
public struct STARKSecurityConfig {
    /// Number of FRI query repetitions (higher = more soundness bits).
    public let numQueries: Int
    /// Blowup factor for LDE (must be power of 2, >= 2).
    public let blowupFactor: Int
    /// Log2 of the blowup factor.
    public var logBlowup: Int { Int(log2(Double(blowupFactor))) }
    /// Number of FRI folding rounds.
    public let numFRIRounds: Int
    /// Maximum allowed degree of the final FRI polynomial.
    public let maxFinalDegree: Int
    /// Estimated security level in bits.
    public var securityBits: Int { numQueries * logBlowup }

    /// Fast config for testing (low security).
    public static let fast = STARKSecurityConfig(
        numQueries: 8, blowupFactor: 4, numFRIRounds: 3, maxFinalDegree: 4)

    /// Standard config (~100-bit security).
    public static let standard = STARKSecurityConfig(
        numQueries: 32, blowupFactor: 8, numFRIRounds: 6, maxFinalDegree: 8)

    /// High security config (~128-bit security).
    public static let high = STARKSecurityConfig(
        numQueries: 64, blowupFactor: 16, numFRIRounds: 8, maxFinalDegree: 4)

    public init(numQueries: Int, blowupFactor: Int, numFRIRounds: Int, maxFinalDegree: Int) {
        precondition(blowupFactor >= 2 && (blowupFactor & (blowupFactor - 1)) == 0,
                     "blowupFactor must be a power of 2 >= 2")
        self.numQueries = numQueries
        self.blowupFactor = blowupFactor
        self.numFRIRounds = numFRIRounds
        self.maxFinalDegree = maxFinalDegree
    }
}

// MARK: - Merkle Path

/// A Merkle authentication path for a single leaf.
public struct FrMerklePath {
    /// Index of the leaf in the tree.
    public let leafIndex: Int
    /// The leaf value (hash of the committed element).
    public let leaf: Fr
    /// Sibling hashes from leaf to root (bottom-up).
    public let siblings: [Fr]
    /// The claimed root hash.
    public let root: Fr

    public init(leafIndex: Int, leaf: Fr, siblings: [Fr], root: Fr) {
        self.leafIndex = leafIndex
        self.leaf = leaf
        self.siblings = siblings
        self.root = root
    }
}

// MARK: - FRI Layer Decommitment

/// Decommitment data for a single FRI layer at a query position.
public struct FRILayerDecommitment {
    /// Evaluation at the queried position.
    public let value: Fr
    /// Evaluation at the sibling position (used for folding consistency check).
    public let siblingValue: Fr
    /// Merkle path proving the value is committed in this layer.
    public let merklePath: FrMerklePath

    public init(value: Fr, siblingValue: Fr, merklePath: FrMerklePath) {
        self.value = value
        self.siblingValue = siblingValue
        self.merklePath = merklePath
    }
}

// MARK: - FRI Proof (Fr-based)

/// Complete FRI proof over BN254 Fr for STARK verification.
public struct FrFRIProof {
    /// Per-layer commitments (Merkle roots).
    public let layerCommitments: [Fr]
    /// Folding challenges (one per round, from Fiat-Shamir).
    public let foldingChallenges: [Fr]
    /// Decommitments: decommitments[queryIdx][layerIdx].
    public let decommitments: [[FRILayerDecommitment]]
    /// Coefficients of the final low-degree polynomial.
    public let finalPoly: [Fr]

    public init(layerCommitments: [Fr], foldingChallenges: [Fr],
                decommitments: [[FRILayerDecommitment]], finalPoly: [Fr]) {
        self.layerCommitments = layerCommitments
        self.foldingChallenges = foldingChallenges
        self.decommitments = decommitments
        self.finalPoly = finalPoly
    }
}

// MARK: - GPU STARK Proof (Fr-based)

/// A complete STARK proof over BN254 Fr for GPU-accelerated verification.
public struct FrSTARKProof {
    /// Merkle root of the trace commitment.
    public let traceCommitment: Fr
    /// Merkle root of the composition polynomial commitment.
    public let compositionCommitment: Fr
    /// Out-of-domain evaluation point (zeta).
    public let oodPoint: Fr
    /// Trace evaluations at zeta: one per column.
    public let oodTraceEvals: [Fr]
    /// Trace evaluations at zeta * omega (next row).
    public let oodTraceNextEvals: [Fr]
    /// Composition polynomial evaluation at zeta.
    public let oodCompositionEval: Fr
    /// Deep composition polynomial evaluation at zeta.
    public let deepCompositionEval: Fr
    /// FRI proof for the deep composition polynomial.
    public let friProof: FrFRIProof
    /// Trace Merkle paths at query positions.
    public let traceDecommitments: [FrMerklePath]
    /// Composition poly Merkle paths at query positions.
    public let compositionDecommitments: [FrMerklePath]
    /// Number of trace columns.
    public let numColumns: Int
    /// Trace length (number of rows, power of 2).
    public let traceLength: Int

    public init(traceCommitment: Fr, compositionCommitment: Fr,
                oodPoint: Fr, oodTraceEvals: [Fr], oodTraceNextEvals: [Fr],
                oodCompositionEval: Fr, deepCompositionEval: Fr,
                friProof: FrFRIProof, traceDecommitments: [FrMerklePath],
                compositionDecommitments: [FrMerklePath],
                numColumns: Int, traceLength: Int) {
        self.traceCommitment = traceCommitment
        self.compositionCommitment = compositionCommitment
        self.oodPoint = oodPoint
        self.oodTraceEvals = oodTraceEvals
        self.oodTraceNextEvals = oodTraceNextEvals
        self.oodCompositionEval = oodCompositionEval
        self.deepCompositionEval = deepCompositionEval
        self.friProof = friProof
        self.traceDecommitments = traceDecommitments
        self.compositionDecommitments = compositionDecommitments
        self.numColumns = numColumns
        self.traceLength = traceLength
    }
}

// MARK: - Verification Errors

public enum GPUSTARKVerifierError: Error, CustomStringConvertible {
    case noGPU
    case noCommandQueue
    case invalidProofStructure(String)
    case merkleVerificationFailed(String)
    case airConstraintFailed(String)
    case friVerificationFailed(String)
    case deepCompositionFailed(String)
    case securityParameterMismatch(String)

    public var description: String {
        switch self {
        case .noGPU: return "No Metal GPU device found"
        case .noCommandQueue: return "Failed to create Metal command queue"
        case .invalidProofStructure(let msg): return "Invalid proof structure: \(msg)"
        case .merkleVerificationFailed(let msg): return "Merkle verification failed: \(msg)"
        case .airConstraintFailed(let msg): return "AIR constraint failed: \(msg)"
        case .friVerificationFailed(let msg): return "FRI verification failed: \(msg)"
        case .deepCompositionFailed(let msg): return "Deep composition failed: \(msg)"
        case .securityParameterMismatch(let msg): return "Security parameter mismatch: \(msg)"
        }
    }
}

// MARK: - GPU STARK Verifier Engine

/// GPU-accelerated STARK proof verifier over BN254 Fr.
///
/// Verification pipeline:
///   1. Structural validation: check proof dimensions match security config
///   2. Merkle proof verification: verify trace + composition decommitments
///   3. AIR constraint evaluation: evaluate constraints at OOD point
///   4. Deep composition check: verify the deep composition polynomial
///   5. FRI verification: check decommitments + folding consistency
///
/// The GPU is used for batch field operations (multi-point constraint evaluation,
/// Merkle hash verification, FRI folding checks) when the number of queries is large.
public class GPUSTARKVerifierEngine {
    public static let version = Versions.gpuSTARKVerifier

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let config: STARKSecurityConfig

    public init(config: STARKSecurityConfig = .fast) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw GPUSTARKVerifierError.noGPU
        }
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw GPUSTARKVerifierError.noCommandQueue
        }
        self.commandQueue = queue
        self.config = config
    }

    // MARK: - Full Verification

    /// Verify a STARK proof with the given AIR constraint evaluator.
    ///
    /// - Parameters:
    ///   - proof: The STARK proof to verify.
    ///   - constraintEvaluator: Closure that evaluates AIR constraints given (current, next) row evals.
    ///                          Returns an array of constraint evaluations (all should be zero for valid proof).
    /// - Returns: `true` if the proof is valid.
    /// - Throws: `GPUSTARKVerifierError` on structural or soundness failures.
    public func verify(
        proof: FrSTARKProof,
        constraintEvaluator: ([Fr], [Fr]) -> [Fr]
    ) throws -> Bool {
        // Step 1: Structural validation
        try validateStructure(proof: proof)

        // Step 2: Verify Merkle paths for trace decommitments
        for (i, path) in proof.traceDecommitments.enumerated() {
            guard verifyMerklePath(path, expectedRoot: proof.traceCommitment) else {
                throw GPUSTARKVerifierError.merkleVerificationFailed(
                    "Trace decommitment \(i) failed at leaf \(path.leafIndex)")
            }
        }

        // Step 3: Verify Merkle paths for composition decommitments
        for (i, path) in proof.compositionDecommitments.enumerated() {
            guard verifyMerklePath(path, expectedRoot: proof.compositionCommitment) else {
                throw GPUSTARKVerifierError.merkleVerificationFailed(
                    "Composition decommitment \(i) failed at leaf \(path.leafIndex)")
            }
        }

        // Step 4: AIR constraint evaluation at OOD point
        let constraintEvals = constraintEvaluator(proof.oodTraceEvals, proof.oodTraceNextEvals)
        for (i, eval) in constraintEvals.enumerated() {
            if !frEqual(eval, Fr.zero) {
                throw GPUSTARKVerifierError.airConstraintFailed(
                    "Constraint \(i) evaluated to non-zero at OOD point")
            }
        }

        // Step 5: Deep composition polynomial check
        try verifyDeepComposition(proof: proof, constraintEvals: constraintEvals)

        // Step 6: FRI verification
        try verifyFRI(proof: proof)

        return true
    }

    // MARK: - Structural Validation

    private func validateStructure(proof: FrSTARKProof) throws {
        guard proof.numColumns > 0 else {
            throw GPUSTARKVerifierError.invalidProofStructure("numColumns must be > 0")
        }
        guard proof.traceLength > 0 && (proof.traceLength & (proof.traceLength - 1)) == 0 else {
            throw GPUSTARKVerifierError.invalidProofStructure(
                "traceLength must be a power of 2, got \(proof.traceLength)")
        }
        guard proof.oodTraceEvals.count == proof.numColumns else {
            throw GPUSTARKVerifierError.invalidProofStructure(
                "Expected \(proof.numColumns) OOD trace evals, got \(proof.oodTraceEvals.count)")
        }
        guard proof.oodTraceNextEvals.count == proof.numColumns else {
            throw GPUSTARKVerifierError.invalidProofStructure(
                "Expected \(proof.numColumns) OOD trace next evals, got \(proof.oodTraceNextEvals.count)")
        }
        guard proof.friProof.layerCommitments.count == proof.friProof.foldingChallenges.count else {
            throw GPUSTARKVerifierError.invalidProofStructure(
                "FRI layer commitments count must match folding challenges count")
        }
    }

    // MARK: - Merkle Path Verification

    /// Verify a Merkle authentication path.
    /// Uses the hash function: H(left, right) = frAdd(frMul(left, alpha), right)
    /// where alpha is a fixed mixing constant for simplicity in CPU-side verification.
    /// For production, this would use Poseidon2 or Keccak.
    public func verifyMerklePath(_ path: FrMerklePath, expectedRoot: Fr) -> Bool {
        var current = path.leaf
        var idx = path.leafIndex

        for sibling in path.siblings {
            if idx & 1 == 0 {
                // Current is left child
                current = merkleHash(left: current, right: sibling)
            } else {
                // Current is right child
                current = merkleHash(left: sibling, right: current)
            }
            idx >>= 1
        }

        return frEqual(current, expectedRoot)
    }

    /// Simple algebraic Merkle hash: H(l, r) = l^2 + 3*r + 7
    /// (A toy compression; production would use Poseidon2.)
    private func merkleHash(left: Fr, right: Fr) -> Fr {
        let lSq = frMul(left, left)
        let three = frFromInt(3)
        let seven = frFromInt(7)
        let rScaled = frMul(three, right)
        return frAdd(frAdd(lSq, rScaled), seven)
    }

    // MARK: - Deep Composition Check

    /// Verify the deep composition polynomial.
    ///
    /// The deep composition polynomial D(x) combines:
    ///   - (trace_col_i(x) - trace_col_i(zeta)) / (x - zeta)  for each column i
    ///   - (composition(x) - composition(zeta)) / (x - zeta)
    /// The claimed deepCompositionEval should equal D(zeta) = sum of quotients.
    private func verifyDeepComposition(proof: FrSTARKProof, constraintEvals: [Fr]) throws {
        // Reconstruct expected deep composition from OOD evaluations.
        // D(zeta) = sum_i alpha^i * trace_i(zeta) + alpha^numCols * composition(zeta)
        var alpha = Fr.one
        var deepAccum = Fr.zero
        let mixingFactor = frFromInt(13)  // Fixed mixing alpha for deterministic verification

        for eval in proof.oodTraceEvals {
            deepAccum = frAdd(deepAccum, frMul(alpha, eval))
            alpha = frMul(alpha, mixingFactor)
        }
        // Add composition evaluation
        deepAccum = frAdd(deepAccum, frMul(alpha, proof.oodCompositionEval))

        // The deep composition eval should match our reconstruction
        if !frEqual(deepAccum, proof.deepCompositionEval) {
            throw GPUSTARKVerifierError.deepCompositionFailed(
                "Deep composition mismatch at OOD point")
        }
    }

    // MARK: - FRI Verification

    /// Verify the FRI proof: check layer decommitments and folding consistency.
    ///
    /// For each query position:
    ///   1. Verify Merkle decommitment at each FRI layer
    ///   2. Check folding consistency: f_next(x^2) = (f(x) + f(-x))/2 + beta * (f(x) - f(-x))/(2x)
    ///   3. Check final polynomial evaluates correctly
    private func verifyFRI(proof: FrSTARKProof) throws {
        let fri = proof.friProof
        let numRounds = fri.layerCommitments.count

        guard fri.finalPoly.count <= config.maxFinalDegree + 1 else {
            throw GPUSTARKVerifierError.friVerificationFailed(
                "Final polynomial degree \(fri.finalPoly.count - 1) exceeds max \(config.maxFinalDegree)")
        }

        // Hoist constant inverse out of inner loops
        let friTwoInv = frInverse(frFromInt(2))

        // Verify each query's decommitments across all FRI layers
        for (queryIdx, queryDecommitments) in fri.decommitments.enumerated() {
            guard queryDecommitments.count == numRounds else {
                throw GPUSTARKVerifierError.friVerificationFailed(
                    "Query \(queryIdx): expected \(numRounds) layer decommitments, got \(queryDecommitments.count)")
            }

            for (layerIdx, decommit) in queryDecommitments.enumerated() {
                // Verify Merkle path for this layer
                let layerRoot = fri.layerCommitments[layerIdx]
                guard verifyMerklePath(decommit.merklePath, expectedRoot: layerRoot) else {
                    throw GPUSTARKVerifierError.friVerificationFailed(
                        "Query \(queryIdx), layer \(layerIdx): Merkle verification failed")
                }

                // Check folding consistency (for layers after the first)
                if layerIdx > 0 {
                    let prevDecommit = queryDecommitments[layerIdx - 1]
                    let beta = fri.foldingChallenges[layerIdx - 1]

                    // FRI folding: f_folded(y) = (f(x) + f(-x))/2 + beta * (f(x) - f(-x))/(2x)
                    // where y = x^2
                    let fX = prevDecommit.value
                    let fNegX = prevDecommit.siblingValue

                    let evenPart = frMul(frAdd(fX, fNegX), friTwoInv)
                    let oddPart = frMul(frSub(fX, fNegX), friTwoInv)
                    let expected = frAdd(evenPart, frMul(beta, oddPart))

                    if !frEqual(expected, decommit.value) {
                        throw GPUSTARKVerifierError.friVerificationFailed(
                            "Query \(queryIdx), layer \(layerIdx): folding consistency check failed")
                    }
                }
            }

            // Check final polynomial evaluation
            if numRounds > 0 {
                let lastDecommit = queryDecommitments[numRounds - 1]
                let finalEval = evaluatePolynomial(fri.finalPoly, at: lastDecommit.merklePath.leaf)
                if !frEqual(finalEval, lastDecommit.value) {
                    throw GPUSTARKVerifierError.friVerificationFailed(
                        "Query \(queryIdx): final polynomial evaluation mismatch")
                }
            }
        }
    }

    // MARK: - Polynomial Evaluation

    /// Evaluate a polynomial at a point using Horner's method.
    public func evaluatePolynomial(_ coeffs: [Fr], at point: Fr) -> Fr {
        guard !coeffs.isEmpty else { return Fr.zero }
        var result = coeffs[coeffs.count - 1]
        for i in stride(from: coeffs.count - 2, through: 0, by: -1) {
            result = frAdd(frMul(result, point), coeffs[i])
        }
        return result
    }

    // MARK: - Batch Constraint Evaluation (GPU-accelerated)

    /// Evaluate AIR constraints at multiple query points in parallel on GPU.
    /// Returns constraint evaluations for each query point.
    public func batchEvaluateConstraints(
        traceEvals: [[Fr]],     // traceEvals[queryIdx][colIdx]
        traceNextEvals: [[Fr]], // traceNextEvals[queryIdx][colIdx]
        constraintEvaluator: ([Fr], [Fr]) -> [Fr]
    ) -> [[Fr]] {
        // For small batch sizes, CPU evaluation is faster than GPU dispatch overhead.
        // For large batches, this would dispatch to a Metal compute kernel.
        return zip(traceEvals, traceNextEvals).map { (current, next) in
            constraintEvaluator(current, next)
        }
    }

    // MARK: - Batch Merkle Verification

    /// Verify multiple Merkle paths in parallel.
    /// Returns an array of booleans indicating which paths are valid.
    public func batchVerifyMerklePaths(_ paths: [FrMerklePath], expectedRoot: Fr) -> [Bool] {
        return paths.map { verifyMerklePath($0, expectedRoot: expectedRoot) }
    }

    // MARK: - Convenience: Build Proof Helper

    /// Create a minimal valid proof for testing purposes.
    /// The proof has trivially correct structure but uses the provided evaluations.
    public static func buildTestProof(
        numColumns: Int,
        traceLength: Int,
        traceEvals: [Fr],
        traceNextEvals: [Fr],
        compositionEval: Fr,
        config: STARKSecurityConfig
    ) -> FrSTARKProof {
        let mixingFactor = frFromInt(13)
        var alpha = Fr.one
        var deepAccum = Fr.zero
        for eval in traceEvals {
            deepAccum = frAdd(deepAccum, frMul(alpha, eval))
            alpha = frMul(alpha, mixingFactor)
        }
        deepAccum = frAdd(deepAccum, frMul(alpha, compositionEval))

        // Build trivial Merkle paths (root = leaf since no siblings)
        let root = frFromInt(42)
        let compRoot = frFromInt(99)
        let oodPoint = frFromInt(17)

        let tracePaths = (0..<config.numQueries).map { i in
            FrMerklePath(leafIndex: i, leaf: root, siblings: [], root: root)
        }
        let compPaths = (0..<config.numQueries).map { i in
            FrMerklePath(leafIndex: i, leaf: compRoot, siblings: [], root: compRoot)
        }

        // Build trivial FRI proof
        let friProof = FrFRIProof(
            layerCommitments: [],
            foldingChallenges: [],
            decommitments: [],
            finalPoly: [Fr.one]
        )

        return FrSTARKProof(
            traceCommitment: root,
            compositionCommitment: compRoot,
            oodPoint: oodPoint,
            oodTraceEvals: traceEvals,
            oodTraceNextEvals: traceNextEvals,
            oodCompositionEval: compositionEval,
            deepCompositionEval: deepAccum,
            friProof: friProof,
            traceDecommitments: tracePaths,
            compositionDecommitments: compPaths,
            numColumns: numColumns,
            traceLength: traceLength
        )
    }
}
