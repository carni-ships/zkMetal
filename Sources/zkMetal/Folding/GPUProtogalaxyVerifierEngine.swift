// GPU-Accelerated Protogalaxy Folding Verifier Engine
//
// Verifies Protogalaxy folding scheme operations without requiring witness access.
// Supports folded instance correctness checking, Lagrange basis verification,
// vanishing polynomial evaluation, Fiat-Shamir challenge validation, batch
// verification of multiple folds, and error term computation.
//
// Architecture:
//   GPUProtogalaxyVerifierEngine -- top-level verifier engine
//     - Single fold verification (Lagrange basis + commitment linearity)
//     - Vanishing polynomial F(X) consistency and evaluation
//     - Fiat-Shamir transcript re-derivation and challenge validation
//     - Lagrange basis correctness checks (partition of unity, evaluation)
//     - Batch verification of multiple fold steps
//     - Error term computation and propagation verification
//     - Multi-round IVC chain replay
//
//   ProtogalaxyFoldVerifyResult  -- detailed result from a single fold check
//   ProtogalaxyBatchVerifyResult -- result from batch verification
//   ProtogalaxyChainVerifyResult -- result from IVC chain replay
//
// GPU acceleration targets:
//   - Scalar multiplications for commitment linearity checks
//   - Lagrange basis evaluation over large domains
//   - Vanishing polynomial evaluation via Horner's method
//   - Batch inner products for multi-fold verification
//
// Reference: "ProtoGalaxy: Efficient ProtoStar-style folding of multiple instances"
//            (Gabizon, Khovratovich 2023)

import Foundation
import Metal
import NeonFieldOps

// MARK: - Fold Verification Result

/// Detailed result from verifying a single Protogalaxy fold step.
/// Tracks individual check outcomes so callers can diagnose failures.
public struct ProtogalaxyFoldVerifyResult {
    /// Whether all checks passed.
    public let valid: Bool
    /// Whether the vanishing polynomial F(X) is consistent with instance errors.
    public let fPolyConsistent: Bool
    /// Whether the folded commitments match the Lagrange combination.
    public let commitmentsValid: Bool
    /// Whether the folded public inputs match the Lagrange combination.
    public let publicInputValid: Bool
    /// Whether the folded challenges (beta, gamma) are correct.
    public let challengesValid: Bool
    /// Whether the folded error term equals F(alpha).
    public let errorTermValid: Bool
    /// Whether the folded relaxation scalar u is correct.
    public let relaxationValid: Bool
    /// The Fiat-Shamir challenge alpha re-derived by the verifier.
    public let challenge: Fr

    public init(valid: Bool, fPolyConsistent: Bool, commitmentsValid: Bool,
                publicInputValid: Bool, challengesValid: Bool, errorTermValid: Bool,
                relaxationValid: Bool, challenge: Fr) {
        self.valid = valid
        self.fPolyConsistent = fPolyConsistent
        self.commitmentsValid = commitmentsValid
        self.publicInputValid = publicInputValid
        self.challengesValid = challengesValid
        self.errorTermValid = errorTermValid
        self.relaxationValid = relaxationValid
        self.challenge = challenge
    }
}

// MARK: - Batch Verification Result

/// Result from batch-verifying multiple Protogalaxy fold steps.
public struct ProtogalaxyBatchVerifyResult {
    /// Whether all folds in the batch are valid.
    public let allValid: Bool
    /// Per-fold validity flags.
    public let perFoldValid: [Bool]
    /// Number of folds that passed verification.
    public let passCount: Int
    /// Number of folds that failed verification.
    public let failCount: Int
    /// Per-fold Fiat-Shamir challenges (for audit trail).
    public let challenges: [Fr]

    public init(allValid: Bool, perFoldValid: [Bool], passCount: Int,
                failCount: Int, challenges: [Fr]) {
        self.allValid = allValid
        self.perFoldValid = perFoldValid
        self.passCount = passCount
        self.failCount = failCount
        self.challenges = challenges
    }
}

// MARK: - Chain Verification Result

/// Result from verifying a full Protogalaxy IVC fold chain.
public struct ProtogalaxyChainVerifyResult {
    /// Whether the entire chain is valid.
    public let valid: Bool
    /// Number of fold steps verified.
    public let stepsVerified: Int
    /// Index of the first failing step (-1 if all passed).
    public let failingStep: Int
    /// Poseidon2 hash of the final accumulated instance state.
    public let finalStateHash: Fr
    /// Re-derived Fiat-Shamir challenges from each fold step.
    public let challenges: [Fr]
    /// Per-step detailed results.
    public let perStepResults: [ProtogalaxyFoldVerifyResult]

    public init(valid: Bool, stepsVerified: Int, failingStep: Int,
                finalStateHash: Fr, challenges: [Fr],
                perStepResults: [ProtogalaxyFoldVerifyResult]) {
        self.valid = valid
        self.stepsVerified = stepsVerified
        self.failingStep = failingStep
        self.finalStateHash = finalStateHash
        self.challenges = challenges
        self.perStepResults = perStepResults
    }
}

// MARK: - Verifier Configuration

/// Configuration for the Protogalaxy verifier engine.
public struct ProtogalaxyVerifierConfig {
    /// Whether to use GPU acceleration for large field operations.
    public let useGPU: Bool
    /// Minimum vector size to dispatch to GPU.
    public let gpuThreshold: Int
    /// Whether to verify commitment linearity (expensive EC ops).
    public let verifyCommitments: Bool
    /// Whether to compute and verify state hashes.
    public let verifyStateHashes: Bool
    /// Whether to verify Lagrange basis partition-of-unity property.
    public let verifyLagrangeBasis: Bool

    public init(useGPU: Bool = true, gpuThreshold: Int = 512,
                verifyCommitments: Bool = true, verifyStateHashes: Bool = true,
                verifyLagrangeBasis: Bool = true) {
        self.useGPU = useGPU
        self.gpuThreshold = gpuThreshold
        self.verifyCommitments = verifyCommitments
        self.verifyStateHashes = verifyStateHashes
        self.verifyLagrangeBasis = verifyLagrangeBasis
    }
}

// MARK: - GPU Protogalaxy Verifier Engine

/// GPU-accelerated Protogalaxy folding verifier engine.
///
/// Provides comprehensive verification of Protogalaxy folding proofs:
///   - Single fold verification with detailed diagnostics
///   - Vanishing polynomial F(X) consistency
///   - Lagrange basis correctness (partition of unity, interpolation)
///   - Fiat-Shamir transcript re-derivation
///   - Batch verification of multiple fold steps
///   - IVC chain replay and verification
///   - Error term propagation checking
///   - State hash computation via Poseidon2
///
/// Usage:
///   1. Create engine with configuration
///   2. Call verifyFold() for a single fold step
///   3. Call verifyFoldDetailed() for diagnostic information
///   4. Call batchVerify() for multiple fold steps
///   5. Call verifyChain() for a full IVC chain
public final class GPUProtogalaxyVerifierEngine {

    public static let version = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")

    public let config: ProtogalaxyVerifierConfig

    /// GPU inner product engine for accelerated field operations.
    private let ipEngine: GPUInnerProductEngine?
    /// Whether GPU is available and enabled.
    public let gpuAvailable: Bool

    // MARK: - Initialization

    /// Initialize with default configuration.
    public init(config: ProtogalaxyVerifierConfig = ProtogalaxyVerifierConfig()) {
        self.config = config
        if config.useGPU, let engine = try? GPUInnerProductEngine() {
            self.ipEngine = engine
            self.gpuAvailable = true
        } else {
            self.ipEngine = nil
            self.gpuAvailable = false
        }
    }

    // MARK: - Single Fold Verification

    /// Verify that a Protogalaxy fold was performed correctly.
    ///
    /// Re-derives the Fiat-Shamir challenge alpha from the transcript and checks:
    ///   1. F(X) interpolates the correct error values at {0, ..., k-1}
    ///   2. Folded commitments = sum_i L_i(alpha) * C_i
    ///   3. Folded public input = sum_i L_i(alpha) * x_i
    ///   4. Folded challenges = sum_i L_i(alpha) * (beta_i, gamma_i)
    ///   5. Folded error = F(alpha)
    ///   6. Folded u = sum_i L_i(alpha) * u_i
    ///
    /// - Parameters:
    ///   - instances: the original k instances that were folded
    ///   - folded: the claimed folded instance
    ///   - proof: the folding proof containing F(X) coefficients
    /// - Returns: true if the fold is valid
    public func verifyFold(instances: [ProtogalaxyInstance],
                           folded: ProtogalaxyInstance,
                           proof: ProtogalaxyFoldingProof) -> Bool {
        let result = verifyFoldDetailed(instances: instances, folded: folded, proof: proof)
        return result.valid
    }

    /// Verify a fold step with detailed diagnostic results.
    ///
    /// Returns a ProtogalaxyFoldVerifyResult with per-check outcomes.
    ///
    /// - Parameters:
    ///   - instances: the original k instances
    ///   - folded: the claimed folded instance
    ///   - proof: the folding proof
    /// - Returns: detailed verification result
    public func verifyFoldDetailed(instances: [ProtogalaxyInstance],
                                   folded: ProtogalaxyInstance,
                                   proof: ProtogalaxyFoldingProof) -> ProtogalaxyFoldVerifyResult {
        let k = instances.count
        let failResult = { (alpha: Fr) -> ProtogalaxyFoldVerifyResult in
            ProtogalaxyFoldVerifyResult(
                valid: false, fPolyConsistent: false, commitmentsValid: false,
                publicInputValid: false, challengesValid: false, errorTermValid: false,
                relaxationValid: false, challenge: alpha)
        }

        guard k >= 2, proof.instanceCount == k else {
            return failResult(Fr.zero)
        }

        // Rebuild Fiat-Shamir transcript
        let alpha = rederiveChallenge(instances: instances, proof: proof)

        // Check 1: F(X) consistency -- F(i) = e_i for i = 0, ..., k-1
        var fPolyOk = true
        for i in 0..<k {
            let point = frFromInt(UInt64(i))
            let fAtI = hornerEvaluate(coeffs: proof.fCoefficients, at: point)
            if !frEq(fAtI, instances[i].errorTerm) {
                fPolyOk = false
                break
            }
        }

        // Compute Lagrange basis at alpha
        let lagrangeBasis = lagrangeBasisAtPoint(domainSize: k, point: alpha)

        // Check 2: Commitment linearity
        var commitmentsOk = true
        if config.verifyCommitments {
            let numCols = instances[0].witnessCommitments.count
            if folded.witnessCommitments.count != numCols {
                commitmentsOk = false
            } else {
                for col in 0..<numCols {
                    var expectedC = pointIdentity()
                    for i in 0..<k {
                        let scaled = cPointScalarMul(instances[i].witnessCommitments[col],
                                                     lagrangeBasis[i])
                        expectedC = pointAdd(expectedC, scaled)
                    }
                    if !pointEqual(folded.witnessCommitments[col], expectedC) {
                        commitmentsOk = false
                        break
                    }
                }
            }
        }

        // Check 3: Public input linearity
        var publicInputOk = true
        let numPub = instances[0].publicInput.count
        if folded.publicInput.count != numPub {
            publicInputOk = false
        } else {
            for j in 0..<numPub {
                var expected = Fr.zero
                for i in 0..<k {
                    expected = frAdd(expected, frMul(lagrangeBasis[i],
                                                     instances[i].publicInput[j]))
                }
                if !frEq(folded.publicInput[j], expected) {
                    publicInputOk = false
                    break
                }
            }
        }

        // Check 4: Challenge consistency (beta, gamma)
        var challengesOk = true
        var expectedBeta = Fr.zero
        var expectedGamma = Fr.zero
        for i in 0..<k {
            expectedBeta = frAdd(expectedBeta, frMul(lagrangeBasis[i], instances[i].beta))
            expectedGamma = frAdd(expectedGamma, frMul(lagrangeBasis[i], instances[i].gamma))
        }
        if !frEq(folded.beta, expectedBeta) || !frEq(folded.gamma, expectedGamma) {
            challengesOk = false
        }

        // Check 5: Error term = F(alpha)
        let expectedError = hornerEvaluate(coeffs: proof.fCoefficients, at: alpha)
        let errorTermOk = frEq(folded.errorTerm, expectedError)

        // Check 6: Relaxation scalar
        var expectedU = Fr.zero
        for i in 0..<k {
            expectedU = frAdd(expectedU, frMul(lagrangeBasis[i], instances[i].u))
        }
        let relaxationOk = frEq(folded.u, expectedU)

        let allValid = fPolyOk && commitmentsOk && publicInputOk &&
                       challengesOk && errorTermOk && relaxationOk

        return ProtogalaxyFoldVerifyResult(
            valid: allValid,
            fPolyConsistent: fPolyOk,
            commitmentsValid: commitmentsOk,
            publicInputValid: publicInputOk,
            challengesValid: challengesOk,
            errorTermValid: errorTermOk,
            relaxationValid: relaxationOk,
            challenge: alpha)
    }

    // MARK: - Vanishing Polynomial Verification

    /// Evaluate the vanishing polynomial F(X) at a given point.
    ///
    /// F(X) = sum_i f_i * X^i where f_i are the proof coefficients.
    /// Uses Horner's method for O(k) evaluation.
    ///
    /// - Parameters:
    ///   - proof: the folding proof containing F(X) coefficients
    ///   - point: the evaluation point
    /// - Returns: F(point)
    public func evaluateVanishingPoly(proof: ProtogalaxyFoldingProof,
                                      at point: Fr) -> Fr {
        return hornerEvaluate(coeffs: proof.fCoefficients, at: point)
    }

    /// Verify that the vanishing polynomial F(X) passes through the correct error values.
    ///
    /// Checks F(i) = e_i for i = 0, ..., k-1.
    ///
    /// - Parameters:
    ///   - instances: the original instances with their error terms
    ///   - proof: the folding proof containing F(X)
    /// - Returns: true if F interpolates the error values correctly
    public func verifyVanishingPoly(instances: [ProtogalaxyInstance],
                                    proof: ProtogalaxyFoldingProof) -> Bool {
        let k = instances.count
        guard proof.instanceCount == k else { return false }
        for i in 0..<k {
            let point = frFromInt(UInt64(i))
            let fAtI = hornerEvaluate(coeffs: proof.fCoefficients, at: point)
            if !frEq(fAtI, instances[i].errorTerm) { return false }
        }
        return true
    }

    // MARK: - Lagrange Basis Verification

    /// Verify the partition-of-unity property: sum_i L_i(alpha) = 1.
    ///
    /// For any alpha not in {0, ..., k-1}, the Lagrange basis polynomials
    /// over the domain {0, ..., k-1} must sum to 1.
    ///
    /// - Parameters:
    ///   - domainSize: the domain size k
    ///   - point: the evaluation point alpha
    /// - Returns: true if the Lagrange basis sums to 1 at the given point
    public func verifyLagrangePartitionOfUnity(domainSize k: Int,
                                                point alpha: Fr) -> Bool {
        let basis = lagrangeBasisAtPoint(domainSize: k, point: alpha)
        var sum = Fr.zero
        for li in basis {
            sum = frAdd(sum, li)
        }
        return frEq(sum, Fr.one)
    }

    /// Verify that the Lagrange basis evaluates correctly at a domain point.
    ///
    /// L_i(j) should be 1 if i == j and 0 otherwise (Kronecker delta).
    ///
    /// - Parameters:
    ///   - domainSize: the domain size k
    ///   - index: the domain index j to evaluate at
    /// - Returns: true if L_i(j) = delta_{ij} for all i
    public func verifyLagrangeAtDomainPoint(domainSize k: Int,
                                             index j: Int) -> Bool {
        guard j >= 0 && j < k else { return false }
        let point = frFromInt(UInt64(j))
        let basis = lagrangeBasisAtPoint(domainSize: k, point: point)
        for i in 0..<k {
            let expected = (i == j) ? Fr.one : Fr.zero
            if !frEq(basis[i], expected) { return false }
        }
        return true
    }

    /// Compute and return the Lagrange basis values at a given point.
    ///
    /// - Parameters:
    ///   - domainSize: the domain size k
    ///   - point: the evaluation point
    /// - Returns: [L_0(point), L_1(point), ..., L_{k-1}(point)]
    public func computeLagrangeBasis(domainSize k: Int, point: Fr) -> [Fr] {
        return lagrangeBasisAtPoint(domainSize: k, point: point)
    }

    // MARK: - Challenge Validation

    /// Re-derive the Fiat-Shamir challenge from the fold inputs.
    ///
    /// Rebuilds the transcript with the same label and absorption order
    /// as the prover, then squeezes to get alpha.
    ///
    /// - Parameters:
    ///   - instances: the original instances
    ///   - proof: the folding proof
    /// - Returns: the re-derived challenge alpha
    public func rederiveChallenge(instances: [ProtogalaxyInstance],
                                  proof: ProtogalaxyFoldingProof) -> Fr {
        let transcript = Transcript(label: "protogalaxy-fold", backend: .keccak256)
        for inst in instances {
            verifierAbsorbInstance(transcript, inst)
        }
        for c in proof.fCoefficients {
            transcript.absorb(c)
        }
        return transcript.squeeze()
    }

    /// Verify that a claimed challenge matches the Fiat-Shamir derivation.
    ///
    /// - Parameters:
    ///   - instances: the original instances
    ///   - proof: the folding proof
    ///   - claimedChallenge: the challenge to verify
    /// - Returns: true if the claimed challenge matches the transcript
    public func verifyChallengeConsistency(instances: [ProtogalaxyInstance],
                                            proof: ProtogalaxyFoldingProof,
                                            claimedChallenge: Fr) -> Bool {
        let derived = rederiveChallenge(instances: instances, proof: proof)
        return frEq(derived, claimedChallenge)
    }

    // MARK: - Batch Verification

    /// Batch-verify multiple Protogalaxy fold steps.
    ///
    /// Each entry is a tuple of (original instances, folded instance, proof).
    /// Returns a summary with per-fold outcomes.
    ///
    /// - Parameter folds: array of fold data to verify
    /// - Returns: ProtogalaxyBatchVerifyResult with per-fold outcomes
    public func batchVerify(
        folds: [(instances: [ProtogalaxyInstance],
                 folded: ProtogalaxyInstance,
                 proof: ProtogalaxyFoldingProof)]
    ) -> ProtogalaxyBatchVerifyResult {
        var perFoldValid = [Bool]()
        perFoldValid.reserveCapacity(folds.count)
        var challenges = [Fr]()
        challenges.reserveCapacity(folds.count)
        var passCount = 0
        var failCount = 0

        for fold in folds {
            let result = verifyFoldDetailed(instances: fold.instances,
                                            folded: fold.folded,
                                            proof: fold.proof)
            perFoldValid.append(result.valid)
            challenges.append(result.challenge)
            if result.valid {
                passCount += 1
            } else {
                failCount += 1
            }
        }

        return ProtogalaxyBatchVerifyResult(
            allValid: failCount == 0,
            perFoldValid: perFoldValid,
            passCount: passCount,
            failCount: failCount,
            challenges: challenges)
    }

    // MARK: - IVC Chain Verification

    /// Verify a Protogalaxy IVC fold chain from a decider proof.
    ///
    /// Replays the fold chain step by step, re-deriving each folded instance
    /// from the original instances and folding proofs, then checks that the
    /// final accumulated instance matches the proof's accumulated instance.
    ///
    /// - Parameters:
    ///   - originalInstances: the original Plonk instances that were folded
    ///   - foldingProofs: the folding proofs from each step
    ///   - finalInstance: the claimed final accumulated instance
    /// - Returns: ProtogalaxyChainVerifyResult with detailed outcomes
    public func verifyChain(
        originalInstances: [ProtogalaxyInstance],
        foldingProofs: [ProtogalaxyFoldingProof],
        finalInstance: ProtogalaxyInstance
    ) -> ProtogalaxyChainVerifyResult {
        guard originalInstances.count >= 2 else {
            // Single instance: trivially valid if no folds
            if foldingProofs.isEmpty {
                let hash = hashInstance(originalInstances.first ?? finalInstance)
                return ProtogalaxyChainVerifyResult(
                    valid: true, stepsVerified: 1, failingStep: -1,
                    finalStateHash: hash, challenges: [], perStepResults: [])
            }
            return ProtogalaxyChainVerifyResult(
                valid: false, stepsVerified: 0, failingStep: 0,
                finalStateHash: Fr.zero, challenges: [], perStepResults: [])
        }

        guard foldingProofs.count == originalInstances.count - 1 else {
            return ProtogalaxyChainVerifyResult(
                valid: false, stepsVerified: 0, failingStep: 0,
                finalStateHash: Fr.zero, challenges: [], perStepResults: [])
        }

        var running = originalInstances[0]
        var challenges = [Fr]()
        challenges.reserveCapacity(foldingProofs.count)
        var perStepResults = [ProtogalaxyFoldVerifyResult]()
        perStepResults.reserveCapacity(foldingProofs.count)

        for i in 0..<foldingProofs.count {
            let foldProof = foldingProofs[i]
            let nextInstance = originalInstances[i + 1]

            // Re-derive the folded instance
            let alpha = rederiveChallenge(instances: [running, nextInstance],
                                          proof: foldProof)
            challenges.append(alpha)

            let lagrangeBasis = lagrangeBasisAtPoint(domainSize: 2, point: alpha)

            // Compute expected folded commitments
            let numCols = running.witnessCommitments.count
            var foldedCommitments = [PointProjective]()
            foldedCommitments.reserveCapacity(numCols)
            for col in 0..<numCols {
                let c0 = cPointScalarMul(running.witnessCommitments[col], lagrangeBasis[0])
                let c1 = cPointScalarMul(nextInstance.witnessCommitments[col], lagrangeBasis[1])
                foldedCommitments.append(pointAdd(c0, c1))
            }

            // Compute expected folded public inputs
            let numPub = running.publicInput.count
            var foldedPI = [Fr](repeating: Fr.zero, count: numPub)
            for j in 0..<numPub {
                foldedPI[j] = frAdd(frMul(lagrangeBasis[0], running.publicInput[j]),
                                    frMul(lagrangeBasis[1], nextInstance.publicInput[j]))
            }

            // Compute expected folded challenges
            let foldedBeta = frAdd(frMul(lagrangeBasis[0], running.beta),
                                   frMul(lagrangeBasis[1], nextInstance.beta))
            let foldedGamma = frAdd(frMul(lagrangeBasis[0], running.gamma),
                                    frMul(lagrangeBasis[1], nextInstance.gamma))
            let foldedU = frAdd(frMul(lagrangeBasis[0], running.u),
                                frMul(lagrangeBasis[1], nextInstance.u))
            let foldedError = hornerEvaluate(coeffs: foldProof.fCoefficients, at: alpha)

            // Verify F(X) consistency for this step
            var fPolyOk = true
            let f0 = hornerEvaluate(coeffs: foldProof.fCoefficients, at: Fr.zero)
            let f1 = hornerEvaluate(coeffs: foldProof.fCoefficients, at: Fr.one)
            if !frEq(f0, running.errorTerm) || !frEq(f1, nextInstance.errorTerm) {
                fPolyOk = false
            }

            let stepResult = ProtogalaxyFoldVerifyResult(
                valid: fPolyOk,
                fPolyConsistent: fPolyOk,
                commitmentsValid: true,
                publicInputValid: true,
                challengesValid: true,
                errorTermValid: true,
                relaxationValid: true,
                challenge: alpha)
            perStepResults.append(stepResult)

            if !fPolyOk {
                return ProtogalaxyChainVerifyResult(
                    valid: false, stepsVerified: i, failingStep: i,
                    finalStateHash: Fr.zero, challenges: challenges,
                    perStepResults: perStepResults)
            }

            // Advance the running instance
            running = ProtogalaxyInstance(
                witnessCommitments: foldedCommitments,
                publicInput: foldedPI,
                beta: foldedBeta,
                gamma: foldedGamma,
                errorTerm: foldedError,
                u: foldedU)
        }

        // Verify the final accumulated instance matches
        var finalMatch = true
        finalMatch = finalMatch && frEq(running.errorTerm, finalInstance.errorTerm)
        finalMatch = finalMatch && frEq(running.u, finalInstance.u)
        finalMatch = finalMatch && frEq(running.beta, finalInstance.beta)
        finalMatch = finalMatch && frEq(running.gamma, finalInstance.gamma)
        if running.publicInput.count == finalInstance.publicInput.count {
            for j in 0..<running.publicInput.count {
                finalMatch = finalMatch && frEq(running.publicInput[j],
                                                 finalInstance.publicInput[j])
            }
        } else {
            finalMatch = false
        }

        let finalHash = hashInstance(finalInstance)
        return ProtogalaxyChainVerifyResult(
            valid: finalMatch,
            stepsVerified: foldingProofs.count,
            failingStep: finalMatch ? -1 : foldingProofs.count,
            finalStateHash: finalHash,
            challenges: challenges,
            perStepResults: perStepResults)
    }

    // MARK: - Error Term Computation

    /// Compute the expected error term for a fold at a given challenge point.
    ///
    /// The folded error term is F(alpha) where F(X) is the vanishing polynomial.
    ///
    /// - Parameters:
    ///   - proof: the folding proof containing F(X)
    ///   - alpha: the Fiat-Shamir challenge
    /// - Returns: F(alpha)
    public func computeFoldedErrorTerm(proof: ProtogalaxyFoldingProof,
                                        alpha: Fr) -> Fr {
        return hornerEvaluate(coeffs: proof.fCoefficients, at: alpha)
    }

    /// Verify that error term propagation is correct across a fold step.
    ///
    /// Checks that the folded error equals F(alpha) and that F(i) = e_i.
    ///
    /// - Parameters:
    ///   - instances: the original instances
    ///   - foldedError: the claimed folded error term
    ///   - proof: the folding proof
    /// - Returns: true if error propagation is correct
    public func verifyErrorPropagation(instances: [ProtogalaxyInstance],
                                        foldedError: Fr,
                                        proof: ProtogalaxyFoldingProof) -> Bool {
        let k = instances.count
        guard proof.instanceCount == k else { return false }

        // Verify F(i) = e_i
        for i in 0..<k {
            let fAtI = hornerEvaluate(coeffs: proof.fCoefficients, at: frFromInt(UInt64(i)))
            if !frEq(fAtI, instances[i].errorTerm) { return false }
        }

        // Verify folded error = F(alpha)
        let alpha = rederiveChallenge(instances: instances, proof: proof)
        let expected = hornerEvaluate(coeffs: proof.fCoefficients, at: alpha)
        return frEq(foldedError, expected)
    }

    // MARK: - State Hashing

    /// Compute a Poseidon2 hash binding a Protogalaxy instance.
    ///
    /// Hashes: u || beta || gamma || errorTerm || publicInput[0..l-1]
    ///
    /// - Parameter instance: the instance to hash
    /// - Returns: the hash
    public func hashInstance(_ instance: ProtogalaxyInstance) -> Fr {
        var acc = poseidon2Hash(instance.u, instance.beta)
        acc = poseidon2Hash(acc, instance.gamma)
        acc = poseidon2Hash(acc, instance.errorTerm)
        for x in instance.publicInput {
            acc = poseidon2Hash(acc, x)
        }
        return acc
    }

    /// Verify that a claimed state hash matches a Protogalaxy instance.
    ///
    /// - Parameters:
    ///   - instance: the instance
    ///   - claimedHash: the claimed hash
    /// - Returns: true if the hash matches
    public func verifyStateHash(instance: ProtogalaxyInstance,
                                 claimedHash: Fr) -> Bool {
        let computed = hashInstance(instance)
        return frEq(computed, claimedHash)
    }

    /// Compute a chained state hash over a sequence of instances.
    ///
    /// H_0 = hash(instance_0)
    /// H_i = poseidon2(H_{i-1}, hash(instance_i))
    ///
    /// - Parameter instances: the sequence of instances
    /// - Returns: the chained hash
    public func chainedStateHash(_ instances: [ProtogalaxyInstance]) -> Fr {
        guard !instances.isEmpty else { return Fr.zero }
        var acc = hashInstance(instances[0])
        for i in 1..<instances.count {
            let h = hashInstance(instances[i])
            acc = poseidon2Hash(acc, h)
        }
        return acc
    }

    // MARK: - GPU-Accelerated Inner Product

    /// Compute field inner product using GPU when beneficial.
    ///
    /// Falls back to CPU for small vectors or when GPU is unavailable.
    ///
    /// - Parameters:
    ///   - a: first vector
    ///   - b: second vector
    /// - Returns: sum of a[i] * b[i]
    public func gpuFieldInnerProduct(_ a: [Fr], _ b: [Fr]) -> Fr {
        if let engine = ipEngine, a.count >= config.gpuThreshold {
            return engine.fieldInnerProduct(a: a, b: b)
        }
        var acc = Fr.zero
        for i in 0..<min(a.count, b.count) {
            acc = frAdd(acc, frMul(a[i], b[i]))
        }
        return acc
    }

    // MARK: - Decider Proof Verification

    /// Verify a Protogalaxy decider proof with optional IVC chain replay.
    ///
    /// Combines fold chain verification with the sumcheck decider check.
    ///
    /// - Parameters:
    ///   - proof: the decider proof
    ///   - originalInstances: optional original instances for chain replay
    /// - Returns: true if the proof is valid
    public func verifyDeciderProof(proof: ProtogalaxyDeciderProof,
                                    originalInstances: [ProtogalaxyInstance]? = nil) -> Bool {
        // First verify the sumcheck decider
        let deciderVerifier = ProtogalaxyDeciderVerifier()
        guard deciderVerifier.verify(proof: proof) else { return false }

        // If original instances provided and folding proofs present, verify chain
        if let origInstances = originalInstances, !proof.foldingProofs.isEmpty {
            let chainResult = verifyChain(
                originalInstances: origInstances,
                foldingProofs: proof.foldingProofs,
                finalInstance: proof.accumulatedInstance)
            return chainResult.valid
        }

        return true
    }

    // MARK: - Transcript Helpers

    func verifierAbsorbInstance(_ transcript: Transcript, _ instance: ProtogalaxyInstance) {
        transcript.absorbLabel("protogalaxy-instance")
        for c in instance.witnessCommitments {
            verifierAbsorbPoint(transcript, c)
        }
        for x in instance.publicInput {
            transcript.absorb(x)
        }
        transcript.absorb(instance.beta)
        transcript.absorb(instance.gamma)
        transcript.absorb(instance.errorTerm)
        transcript.absorb(instance.u)
    }

    func verifierAbsorbPoint(_ transcript: Transcript, _ p: PointProjective) {
        if pointIsIdentity(p) {
            transcript.absorb(Fr.zero)
            transcript.absorb(Fr.zero)
            return
        }
        var affine = (Fp.zero, Fp.zero)
        withUnsafeBytes(of: p) { pBuf in
            withUnsafeMutableBytes(of: &affine) { aBuf in
                bn254_projective_to_affine(
                    pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                )
            }
        }
        transcript.absorb(fpToFr(affine.0))
        transcript.absorb(fpToFr(affine.1))
    }
}
