// UniversalRecursiveProver — Generic recursive SNARK proving over any VerifierCircuitProtocol encoder
//
// Provides recursive proof composition where the outer proof is always Groth16 (BN254),
// but the inner proof can be any system: Plonk, Plonky2, Halo2, IPA.
//
// The prover:
//   1. Natively verifies the inner proof (soundness check)
//   2. Builds the verifier circuit R1CS for the inner proof
//   3. Maps the inner proof/VK/publicInputs to a full z vector
//   4. Runs Groth16 setup + prove on the verifier circuit
//   5. Returns the outer Groth16 proof attesting to validity
//
// Key insight: The outer Groth16 proof is BN254-compatible for chain recursion.
// Inner proof systems are decoded inside the circuit without requiring their
// full verification key material on the outer circuit.

import Foundation
import NeonFieldOps

// MARK: - Universal Recursive Proof

/// Holds the result of a recursive proving operation.
///
/// Contains the outer Groth16 proof (verifying the verifier circuit of the
/// inner proof was satisfied) plus propagated data from the inner proof.
public struct Groth16RecursiveProof {
    /// The outer Groth16 proof attesting to the inner proof's validity
    public let outerProof: Groth16Proof
    /// The verification key used for the outer Groth16 proof
    public let outerVK: Groth16VerificationKey
    /// Public inputs that were propagated from the inner proof
    public let propagatedPublicInputs: [Fr]
    /// The inner proof system type
    public let innerSystem: String
    /// Recursion depth (number of nested recursive proofs)
    public let depth: Int

    public init(outerProof: Groth16Proof, outerVK: Groth16VerificationKey,
                propagatedPublicInputs: [Fr], innerSystem: String, depth: Int) {
        self.outerProof = outerProof
        self.outerVK = outerVK
        self.propagatedPublicInputs = propagatedPublicInputs
        self.innerSystem = innerSystem
        self.depth = depth
    }
}

// MARK: - Proving Key Cache Entry

/// Cached Groth16 proving key for a verifier circuit.
/// The key is reused across proofs when the inner proof structure is identical.
private struct CachedProvingKey {
    let pk: Groth16ProvingKey
    let vk: Groth16VerificationKey
    let r1cs: R1CSInstance
    let timestamp: Date
}

// MARK: - Universal Recursive Prover

/// Generic recursive SNARK prover supporting any encoder implementing VerifierCircuitProtocol.
///
/// The prover takes an inner proof (of any supported system) and produces an outer
/// Groth16 proof that attests to the inner proof's validity. The inner proof is
/// verified inside an R1CS circuit encoding the encoder's verifier logic.
///
/// Example usage:
///   let prover = UniversalRecursiveProver(encoder: PlonkVerifierCircuitEncoder())
///   let recursiveProof = try prover.prove(innerProof: plonkProof, vk: plonkVK, publicInputs: pubInputs)
public final class UniversalRecursiveProver<Encoder: VerifierCircuitProtocol> {

    // MARK: - Properties

    /// The encoder that translates the inner proof system to R1CS constraints
    public let encoder: Encoder

    /// Groth16 prover used for generating outer proofs
    private let groth16Prover: Groth16Prover

    /// Groth16 setup engine for circuit-specific key generation
    private let groth16Setup: Groth16Setup

    /// Cache of Groth16 proving keys keyed by a circuit fingerprint.
    /// When the same verifier circuit is used repeatedly, we reuse the setup.
    private var pkCache: [String: CachedProvingKey] = [:]

    /// Maximum cache size before eviction
    private let maxCacheSize: Int = 16

    /// Whether to profile proving time
    public var profileRecursive = false

    // MARK: - Initialization

    /// Create a new universal recursive prover with the given encoder.
    ///
    /// - Parameter encoder: The encoder implementing VerifierCircuitProtocol for the inner proof system.
    public init(encoder: Encoder) throws {
        self.encoder = encoder
        self.groth16Prover = try Groth16Prover()
        self.groth16Setup = Groth16Setup()
    }

    // MARK: - Public API

    /// Prove validity of an inner proof by producing an outer Groth16 proof.
    ///
    /// The pipeline:
    ///   1. Natively verify the inner proof (soundness gate)
    ///   2. Build the verifier R1CS for the inner proof system
    ///   3. Map inner proof + VK + publicInputs to a full z vector
    ///   4. Run Groth16 setup (cached if possible) on the verifier circuit
    ///   5. Generate the outer Groth16 proof
    ///
    /// - Parameters:
    ///   - proof: The inner proof to recursively prove
    ///   - vk: The verification key of the inner proof
    ///   - publicInputs: Public inputs to the inner proof
    /// - Returns: A Groth16RecursiveProof containing the outer Groth16 proof and propagated data
    /// - Throws: If native verification fails, R1CS building fails, or Groth16 proving fails
    public func prove(proof: Encoder.ProofType, vk: Encoder.VKType,
                      publicInputs: Encoder.PublicInputType) throws -> Groth16RecursiveProof {
        var t = CFAbsoluteTimeGetCurrent()

        // Step 1: Native verification (soundness gate)
        // This catches invalid proofs before we waste time building the circuit
        if profileRecursive {
            fputs("[recursive] native verify start\n", stderr)
        }
        guard encoder.nativeVerify(proof: proof, vk: vk, publicInputs: publicInputs) else {
            throw UniversalRecursiveProverError.nativeVerificationFailed(
                system: Encoder.innerSystemName
            )
        }
        if profileRecursive {
            let te = CFAbsoluteTimeGetCurrent()
            fputs(String(format: "[recursive] native verify: %.2f ms\n", (te - t) * 1000), stderr)
            t = te
        }

        // Step 2: Build verifier R1CS
        if profileRecursive {
            fputs("[recursive] build verifier r1cs\n", stderr)
        }
        let (r1cs, witnessMapper) = encoder.buildVerifierR1CS()
        if profileRecursive {
            let te = CFAbsoluteTimeGetCurrent()
            fputs(String(format: "[recursive] build verifier r1cs: %.2f ms\n", (te - t) * 1000), stderr)
            t = te
        }

        // Step 3: Generate full z vector from proof, vk, and public inputs
        if profileRecursive {
            fputs("[recursive] witness mapping\n", stderr)
        }
        let z = witnessMapper(proof, vk, publicInputs)
        if profileRecursive {
            let te = CFAbsoluteTimeGetCurrent()
            fputs(String(format: "[recursive] witness mapping: %.2f ms\n", (te - t) * 1000), stderr)
            t = te
        }

        // Verify z satisfies the R1CS
        precondition(r1cs.isSatisfied(z: z),
                     "Verifier circuit R1CS not satisfied by witness")

        // Step 4: Get or create Groth16 proving key (cached when possible)
        let cacheKey = makeCacheKey(r1cs: r1cs)
        let (pk, outVk) = try getOrCreateKeys(r1cs: r1cs, cacheKey: cacheKey)
        if profileRecursive {
            let te = CFAbsoluteTimeGetCurrent()
            fputs(String(format: "[recursive] groth16 setup: %.2f ms\n", (te - t) * 1000), stderr)
            t = te
        }

        // Step 5: Generate outer Groth16 proof
        if profileRecursive {
            fputs("[recursive] groth16 prove\n", stderr)
        }
        let publicInputsOuter = Array(z[1..<(1 + r1cs.numPublic)])
        let proofOuter = try groth16Prover.proveWithWitnessGen(
            pk: pk,
            r1cs: r1cs,
            publicInputs: publicInputsOuter
        )
        if profileRecursive {
            let te = CFAbsoluteTimeGetCurrent()
            fputs(String(format: "[recursive] groth16 prove: %.2f ms\n", (te - t) * 1000), stderr)
        }

        return Groth16RecursiveProof(
            outerProof: proofOuter,
            outerVK: outVk,
            propagatedPublicInputs: publicInputsOuter,
            innerSystem: Encoder.innerSystemName,
            depth: 1
        )
    }

    /// Prove multiple inner proofs in sequence (for nested recursion).
    ///
    /// - Parameter proofs: Array of (proof, vk, publicInputs) tuples in order from innermost to outermost.
    /// - Returns: The outermost Groth16RecursiveProof.
    public func proveSequence(
        proofs: [(proof: Encoder.ProofType, vk: Encoder.VKType, publicInputs: Encoder.PublicInputType)]
    ) throws -> Groth16RecursiveProof {
        guard !proofs.isEmpty else {
            throw UniversalRecursiveProverError.noProofsProvided
        }

        var result: Groth16RecursiveProof?
        for (proof, vk, pubInputs) in proofs {
            result = try prove(proof: proof, vk: vk, publicInputs: pubInputs)
        }
        return result!
    }

    /// Clear the proving key cache.
    public func clearCache() {
        pkCache.removeAll()
    }

    /// Number of cached proving keys.
    public var cacheSize: Int {
        pkCache.count
    }

    // MARK: - Private Helpers

    /// Get cached keys or create new ones via Groth16 setup.
    private func getOrCreateKeys(r1cs: R1CSInstance, cacheKey: String) throws -> (Groth16ProvingKey, Groth16VerificationKey) {
        if let cached = pkCache[cacheKey] {
            if profileRecursive {
                fputs("[recursive] pk cache hit\n", stderr)
            }
            return (cached.pk, cached.vk)
        }

        if profileRecursive {
            fputs("[recursive] pk cache miss, running groth16 setup\n", stderr)
        }

        let (pk, vk) = groth16Setup.setup(r1cs: r1cs)

        // Evict oldest entry if cache is full
        if pkCache.count >= maxCacheSize {
            let oldestKey = pkCache.min(by: { $0.value.timestamp < $1.value.timestamp })!.key
            pkCache.removeValue(forKey: oldestKey)
        }

        pkCache[cacheKey] = CachedProvingKey(
            pk: pk,
            vk: vk,
            r1cs: r1cs,
            timestamp: Date()
        )

        return (pk, vk)
    }

    /// Create a cache key for the R1CS circuit.
    /// Uses constraint count and variable count as a quick fingerprint.
    /// For production, a cryptographic hash of the R1CS would be more robust.
    private func makeCacheKey(r1cs: R1CSInstance) -> String {
        return "\(r1cs.numConstraints)-\(r1cs.numVars)-\(r1cs.numPublic)-\(Encoder.innerSystemName)"
    }
}

// MARK: - Errors

public enum UniversalRecursiveProverError: Error, CustomStringConvertible {
    case nativeVerificationFailed(system: String)
    case noProofsProvided
    case r1csUnsatisfied
    case groth16ProvingFailed(String)

    public var description: String {
        switch self {
        case .nativeVerificationFailed(let system):
            return "UniversalRecursiveProver: native verification failed for \(system) proof"
        case .noProofsProvided:
            return "UniversalRecursiveProver: no proofs provided to proveSequence"
        case .r1csUnsatisfied:
            return "UniversalRecursiveProver: verifier circuit R1CS not satisfied"
        case .groth16ProvingFailed(let msg):
            return "UniversalRecursiveProver: Groth16 proving failed: \(msg)"
        }
    }
}

// MARK: - Convenience Factory

/// Create a universal recursive prover for Plonk proofs.
public func makeRecursiveProver(forPlonk: ()) throws -> UniversalRecursiveProver<PlonkVerifierCircuitEncoder> {
    return try UniversalRecursiveProver(encoder: PlonkVerifierCircuitEncoder())
}

/// Create a universal recursive prover for Halo2 proofs.
public func makeRecursiveProver(forHalo2: ()) throws -> UniversalRecursiveProver<Halo2VerifierCircuitEncoder> {
    return try UniversalRecursiveProver(encoder: Halo2VerifierCircuitEncoder())
}

/// Create a universal recursive prover for Plonky2 proofs.
public func makeRecursiveProver(forPlonky2 circuitRepr: Plonky2RecursiveCircuitRepr) throws
    -> UniversalRecursiveProver<Plonky2VerifierCircuitEncoder> {
    return try UniversalRecursiveProver(encoder: Plonky2VerifierCircuitEncoder(circuitRepr: circuitRepr))
}

/// Create a universal recursive prover for IPA proofs (Pasta cycle).
public func makeRecursiveProver(forIPA: ()) throws -> UniversalRecursiveProver<IPAPastaVerifierCircuitEncoder> {
    return try UniversalRecursiveProver(encoder: IPAPastaVerifierCircuitEncoder())
}
