// RecursiveSNARK — Generic recursive SNARK composition engine
//
// Provides VerifierCircuitProtocol for encoding any proof system's verifier
// as R1CS constraints, enabling recursive proof composition.
//
// Supported encoders (in VerifierCircuitEncoders/):
//   - PlonkVerifierCircuitEncoder: ~200K constraints (deferred KZG pairing)
//   - Plonky2VerifierCircuitEncoder: ~100K constraints (FRI-based)
//   - Halo2VerifierCircuitEncoder: ~300K constraints (Plonk-compiled)
//   - IPAPastaVerifierCircuitEncoder: ~50K constraints (Pasta cycle)
//
// The Groth16 recursive SNARK implementation has been moved to:
//   branch: backlog/groth16-recursive
//
// References:
//   - "Recursive Proof Composition" (Bowe et al. 2019)
//   - Nova/Folding schemes for incremental verification

import Foundation
import NeonFieldOps

// MARK: - VerifierCircuit Protocol

/// Protocol for encoding a SNARK verifier as an R1CS circuit.
///
/// Implementations translate a specific proof system's verification logic into
/// R1CS constraints, enabling recursive composition: an outer proof can attest
/// to the validity of an inner proof by proving the verifier circuit is satisfied.
public protocol VerifierCircuitProtocol {
    /// The type of proof this verifier checks.
    associatedtype ProofType
    /// The type of verification key.
    associatedtype VKType
    /// The type of public inputs to the inner proof.
    associatedtype PublicInputType

    /// Name of the inner proof system (for diagnostics).
    static var innerSystemName: String { get }

    /// Estimated number of R1CS constraints for the verifier circuit.
    /// Used for cost estimation and setup sizing.
    var estimatedConstraintCount: Int { get }

    /// Build the R1CS constraint system for verifying this proof type.
    ///
    /// Returns:
    ///   - r1cs: The constraint system encoding the verifier logic
    ///   - witnessMapper: A closure that, given a proof + VK + public inputs,
    ///     produces the full z vector satisfying the R1CS
    func buildVerifierR1CS()
        -> (r1cs: R1CSInstance,
            witnessMapper: (_ proof: ProofType, _ vk: VKType, _ publicInputs: PublicInputType) -> [Fr])

    /// Verify the inner proof natively (outside the circuit) to check soundness
    /// before attempting recursive proving.
    func nativeVerify(proof: ProofType, vk: VKType, publicInputs: PublicInputType) -> Bool
}
