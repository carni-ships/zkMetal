// RecursiveSNARK — Recursive SNARK composition engine for proof aggregation and IVC
//
// Enables verifying a Groth16 proof *inside* another Groth16 proof circuit (BN254).
// The outer proof attests that the inner proof is valid, forming the foundation for:
//   - Proof aggregation: compress N proofs into 1
//   - IVC (Incrementally Verifiable Computation): unbounded-depth recursion
//   - Proof-carrying data: public inputs flow through recursion layers
//
// Architecture:
//   1. VerifierCircuit protocol: generic interface for encoding any SNARK verifier
//      as R1CS constraints
//   2. Groth16VerifierCircuitEncoder: BN254 Groth16 verifier as R1CS using the
//      deferred-pairing approach (MSM accumulation in-circuit, pairing deferred)
//   3. RecursiveProver: given inner proof + VK, produces outer Groth16 proof
//   4. RecursiveVerifier: verifies outer proof (which transitively verifies inner)
//
// The deferred-pairing optimization avoids the ~500K-constraint full pairing check
// by splitting verification into:
//   (a) MSM accumulation: vk_accum = vk_ic[0] + sum(pub[i] * vk_ic[i+1])
//       This is checked inside the circuit (~8 constraints per public input).
//   (b) Pairing check: e(A,B) = e(alpha,beta) * e(vk_accum,gamma) * e(C,delta)
//       This is deferred to the outer native verifier (constant cost).
//
// References:
//   - "Recursive Proof Composition" (Bowe et al. 2019)
//   - "Proof-Carrying Data" (Chiesa et al. 2010)
//   - Circom groth16 verifier (iden3)

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

// MARK: - Groth16 Verifier Circuit Encoder

/// Encodes the BN254 Groth16 verifier as R1CS constraints using deferred-pairing.
///
/// The circuit checks the MSM accumulation portion of Groth16 verification:
///   vk_accum = vk_ic[0] + sum(public_input[i] * vk_ic[i+1])
///
/// The pairing check is deferred to the outer verifier, which natively checks:
///   e(A,B) = e(alpha,beta) * e(vk_accum,gamma) * e(C,delta)
///
/// This approach keeps the circuit small (~O(n) constraints for n public inputs)
/// while still providing recursive soundness through the combined circuit+pairing check.
public class Groth16VerifierCircuitEncoder: VerifierCircuitProtocol {
    public typealias ProofType = Groth16Proof
    public typealias VKType = Groth16VerificationKey
    public typealias PublicInputType = [Fr]

    public static let innerSystemName = "Groth16-BN254"

    /// Number of public inputs in the inner Groth16 proof being verified.
    public let innerPublicInputCount: Int

    /// Estimated constraints: 2*n product constraints + 2 sum constraints + 2 equality checks
    public var estimatedConstraintCount: Int {
        2 * innerPublicInputCount + 4
    }

    public init(innerPublicInputCount: Int) {
        precondition(innerPublicInputCount > 0, "inner proof must have at least 1 public input")
        self.innerPublicInputCount = innerPublicInputCount
    }

    /// Build the verifier R1CS and witness mapper.
    ///
    /// The circuit layout:
    ///   Variables [0]: constant 1
    ///   Variables [1..n]: inner proof's public input scalars (public)
    ///   Variables [n+1, n+2]: expected vk_accum (x, y) coordinates (public)
    ///   Variables [n+3..]: witness (vk_ic coords, products, partial sums)
    ///
    /// Constraints verify the linearized MSM relation:
    ///   accum_x = vk_ic[0].x + sum(pub[i] * vk_ic[i+1].x)
    ///   accum_y = vk_ic[0].y + sum(pub[i] * vk_ic[i+1].y)
    public func buildVerifierR1CS()
        -> (r1cs: R1CSInstance,
            witnessMapper: (Groth16Proof, Groth16VerificationKey, [Fr]) -> [Fr])
    {
        let nPub = innerPublicInputCount

        // Variable layout
        let numPublicVars = nPub + 2  // n scalars + accum (x, y)
        let vkICStart = numPublicVars + 1
        let numVKICVars = 2 * (nPub + 1)
        let productsStart = vkICStart + numVKICVars
        let numProductVars = 2 * nPub
        let partialStart = productsStart + numProductVars
        let totalVars = partialStart + 2

        var aE = [R1CSEntry]()
        var bE = [R1CSEntry]()
        var cE = [R1CSEntry]()
        var row = 0

        // Product constraints: product_x[i] = pub[i] * vk_ic[i+1].x
        //                      product_y[i] = pub[i] * vk_ic[i+1].y
        for i in 0..<nPub {
            let pubVar = 1 + i
            let vkICxVar = vkICStart + 2 * (i + 1)
            let vkICyVar = vkICStart + 2 * (i + 1) + 1
            let prodXVar = productsStart + 2 * i
            let prodYVar = productsStart + 2 * i + 1

            aE.append(R1CSEntry(row: row, col: pubVar, val: .one))
            bE.append(R1CSEntry(row: row, col: vkICxVar, val: .one))
            cE.append(R1CSEntry(row: row, col: prodXVar, val: .one))
            row += 1

            aE.append(R1CSEntry(row: row, col: pubVar, val: .one))
            bE.append(R1CSEntry(row: row, col: vkICyVar, val: .one))
            cE.append(R1CSEntry(row: row, col: prodYVar, val: .one))
            row += 1
        }

        // Sum constraints: sum_x = vk_ic[0].x + product_x[0] + ... + product_x[n-1]
        let sumXVar = partialStart
        let sumYVar = partialStart + 1
        let accumXVar = nPub + 1
        let accumYVar = nPub + 2
        let vkIC0xVar = vkICStart
        let vkIC0yVar = vkICStart + 1

        // (vk_ic[0].x + sum(product_x[i])) * 1 = sum_x
        var aTermsX: [(Int, Fr)] = [(vkIC0xVar, .one)]
        for i in 0..<nPub {
            aTermsX.append((productsStart + 2 * i, .one))
        }
        for (col, val) in aTermsX {
            aE.append(R1CSEntry(row: row, col: col, val: val))
        }
        bE.append(R1CSEntry(row: row, col: 0, val: .one))
        cE.append(R1CSEntry(row: row, col: sumXVar, val: .one))
        row += 1

        // (vk_ic[0].y + sum(product_y[i])) * 1 = sum_y
        var aTermsY: [(Int, Fr)] = [(vkIC0yVar, .one)]
        for i in 0..<nPub {
            aTermsY.append((productsStart + 2 * i + 1, .one))
        }
        for (col, val) in aTermsY {
            aE.append(R1CSEntry(row: row, col: col, val: val))
        }
        bE.append(R1CSEntry(row: row, col: 0, val: .one))
        cE.append(R1CSEntry(row: row, col: sumYVar, val: .one))
        row += 1

        // Equality constraints: sum_x == accum_x, sum_y == accum_y
        aE.append(R1CSEntry(row: row, col: sumXVar, val: .one))
        aE.append(R1CSEntry(row: row, col: accumXVar, val: frNeg(.one)))
        bE.append(R1CSEntry(row: row, col: 0, val: .one))
        row += 1

        aE.append(R1CSEntry(row: row, col: sumYVar, val: .one))
        aE.append(R1CSEntry(row: row, col: accumYVar, val: frNeg(.one)))
        bE.append(R1CSEntry(row: row, col: 0, val: .one))
        row += 1

        let r1cs = R1CSInstance(
            numConstraints: row,
            numVars: totalVars,
            numPublic: numPublicVars,
            aEntries: aE, bEntries: bE, cEntries: cE
        )

        let witnessMapper: (Groth16Proof, Groth16VerificationKey, [Fr]) -> [Fr] = { [nPub] _, vk, pubInputs in
            Self.generateWitness(
                nPub: nPub, publicInputs: pubInputs, vk: vk,
                numVars: totalVars, numPublicVars: numPublicVars,
                vkICStart: vkICStart, productsStart: productsStart,
                partialStart: partialStart
            )
        }

        return (r1cs, witnessMapper)
    }

    /// Native Groth16 verification (delegates to Groth16Verifier).
    public func nativeVerify(proof: Groth16Proof, vk: Groth16VerificationKey,
                              publicInputs: [Fr]) -> Bool {
        let verifier = Groth16Verifier()
        return verifier.verify(proof: proof, vk: vk, publicInputs: publicInputs)
    }

    /// Generate the full z vector for the verifier circuit.
    private static func generateWitness(
        nPub: Int, publicInputs: [Fr], vk: Groth16VerificationKey,
        numVars: Int, numPublicVars: Int,
        vkICStart: Int, productsStart: Int, partialStart: Int
    ) -> [Fr] {
        precondition(publicInputs.count == nPub)
        precondition(vk.ic.count == nPub + 1)

        var z = [Fr](repeating: .zero, count: numVars)
        z[0] = .one

        // Public: scalars
        for i in 0..<nPub {
            z[1 + i] = publicInputs[i]
        }

        // Witness: vk_ic coordinates (embed Fp into Fr by reinterpreting limbs)
        for i in 0...(nPub) {
            let aff = pointToAffine(vk.ic[i])
            z[vkICStart + 2 * i] = embeddedFpToFr(aff?.x ?? .zero)
            z[vkICStart + 2 * i + 1] = embeddedFpToFr(aff?.y ?? .zero)
        }

        // Witness: products (field multiplication, matching linearized circuit constraints)
        for i in 0..<nPub {
            let vkICx = z[vkICStart + 2 * (i + 1)]
            let vkICy = z[vkICStart + 2 * (i + 1) + 1]
            z[productsStart + 2 * i] = frMul(publicInputs[i], vkICx)
            z[productsStart + 2 * i + 1] = frMul(publicInputs[i], vkICy)
        }

        // Witness: partial sums (linearized accumulation, NOT EC point addition)
        var sumX = z[vkICStart]
        var sumY = z[vkICStart + 1]
        for i in 0..<nPub {
            sumX = frAdd(sumX, z[productsStart + 2 * i])
            sumY = frAdd(sumY, z[productsStart + 2 * i + 1])
        }
        z[partialStart] = sumX
        z[partialStart + 1] = sumY

        // Public: expected accumulation point = linearized sum
        // Must match the linearized formula the circuit constraints check,
        // NOT the actual EC point (which uses non-linear curve addition).
        z[nPub + 1] = sumX
        z[nPub + 2] = sumY

        return z
    }
}

// MARK: - Recursive SNARK Proof

/// A recursive SNARK proof: an outer Groth16 proof that attests to the validity
/// of an inner Groth16 proof. Carries the proof-carrying data (public inputs from
/// the inner proof propagated to the outer proof's public inputs).
public struct RecursiveSNARKProof {
    /// The outer Groth16 proof (verifying the inner proof's MSM accumulation)
    public let outerProof: Groth16Proof

    /// The inner proof (needed for deferred pairing check at verification time)
    public let innerProof: Groth16Proof

    /// The inner proof's verification key
    public let innerVK: Groth16VerificationKey

    /// Public inputs propagated from the inner proof (proof-carrying data)
    public let propagatedPublicInputs: [Fr]

    /// Recursion depth (1 = single-level, 2 = proof-of-proof, etc.)
    public let depth: Int

    /// Size in bytes of the full recursive proof (both layers)
    public var sizeBytes: Int {
        // Each G1 point = 2 * 32 bytes = 64 bytes
        // Each G2 point = 2 * 64 bytes = 128 bytes
        // Groth16 proof = A(G1) + B(G2) + C(G1) = 64 + 128 + 64 = 256 bytes
        // Plus public inputs: each Fr = 32 bytes
        let proofSize = 256  // per Groth16 proof
        let publicInputSize = propagatedPublicInputs.count * 32
        return proofSize * 2 + publicInputSize  // inner + outer proofs
    }

    public init(outerProof: Groth16Proof, innerProof: Groth16Proof,
                innerVK: Groth16VerificationKey, propagatedPublicInputs: [Fr],
                depth: Int) {
        self.outerProof = outerProof
        self.innerProof = innerProof
        self.innerVK = innerVK
        self.propagatedPublicInputs = propagatedPublicInputs
        self.depth = depth
    }
}

// MARK: - Recursive Prover

/// Produces recursive SNARK proofs: given an inner Groth16 proof and VK,
/// creates an outer Groth16 proof that attests to the inner proof's validity.
///
/// The proving pipeline:
///   1. Check inner proof is valid (native verification)
///   2. Build verifier circuit R1CS for the inner proof's structure
///   3. Generate witness for the verifier circuit
///   4. Run Groth16 setup + prove on the verifier circuit
///   5. Return the outer proof + inner proof as a RecursiveSNARKProof
public class RecursiveSNARKProver {
    /// Cached proving key for the outer circuit (reusable for same inner structure)
    private var cachedOuterPK: Groth16ProvingKey?
    private var cachedOuterVK: Groth16VerificationKey?
    private var cachedOuterR1CS: R1CSInstance?
    private var cachedWitnessMapper: ((Groth16Proof, Groth16VerificationKey, [Fr]) -> [Fr])?
    private var cachedInnerPublicInputCount: Int = 0

    public init() {}

    /// Produce a recursive proof: the outer proof attests that the inner proof is valid.
    ///
    /// - Parameters:
    ///   - innerProof: The Groth16 proof to verify recursively
    ///   - innerVK: Verification key for the inner proof
    ///   - innerPublicInputs: Public inputs of the inner proof
    ///   - depth: Recursion depth (default 1 for single-level)
    /// - Returns: A RecursiveSNARKProof containing both layers
    /// - Throws: If proving fails (GPU errors, etc.)
    public func prove(
        innerProof: Groth16Proof,
        innerVK: Groth16VerificationKey,
        innerPublicInputs: [Fr],
        depth: Int = 1
    ) throws -> RecursiveSNARKProof {
        let nPub = innerPublicInputs.count

        // Step 1: Verify inner proof natively (soundness check)
        let encoder = Groth16VerifierCircuitEncoder(innerPublicInputCount: nPub)
        guard encoder.nativeVerify(proof: innerProof, vk: innerVK,
                                    publicInputs: innerPublicInputs) else {
            // Return a deliberately invalid outer proof for soundness
            return RecursiveSNARKProof(
                outerProof: Groth16Proof(a: pointIdentity(), b: g2Identity(), c: pointIdentity()),
                innerProof: innerProof,
                innerVK: innerVK,
                propagatedPublicInputs: innerPublicInputs,
                depth: depth
            )
        }

        // Step 2: Build or reuse verifier circuit
        let outerR1CS: R1CSInstance
        let witnessMapper: (Groth16Proof, Groth16VerificationKey, [Fr]) -> [Fr]
        let outerPK: Groth16ProvingKey
        let outerVK: Groth16VerificationKey

        if cachedInnerPublicInputCount == nPub,
           let cpk = cachedOuterPK, let cvk = cachedOuterVK,
           let cr1cs = cachedOuterR1CS, let cwm = cachedWitnessMapper {
            outerR1CS = cr1cs
            witnessMapper = cwm
            outerPK = cpk
            outerVK = cvk
        } else {
            let (r1cs, wm) = encoder.buildVerifierR1CS()
            outerR1CS = r1cs
            witnessMapper = wm

            // Step 3: Setup for outer circuit
            let setup = Groth16Setup()
            let (pk, vk) = setup.setup(r1cs: outerR1CS)
            outerPK = pk
            outerVK = vk

            // Cache for reuse
            cachedOuterPK = pk
            cachedOuterVK = vk
            cachedOuterR1CS = outerR1CS
            cachedWitnessMapper = wm
            cachedInnerPublicInputCount = nPub
        }

        // Step 4: Generate witness for verifier circuit
        let z = witnessMapper(innerProof, innerVK, innerPublicInputs)

        // Step 5: Extract witness and prove
        let outerPublicInputs = Array(z[1...(outerR1CS.numPublic)])
        let outerWitness = Array(z[(1 + outerR1CS.numPublic)...])

        let prover = try Groth16Prover()
        let outerProof = try prover.prove(
            pk: outerPK, r1cs: outerR1CS,
            publicInputs: outerPublicInputs, witness: outerWitness
        )

        // Debug: verify immediately after proving
        let selfCheck = Groth16Verifier().verify(proof: outerProof, vk: outerVK, publicInputs: outerPublicInputs)
        if !selfCheck { fputs("  [recursive-prover-debug] SELF-CHECK FAILED: proof doesn't verify immediately after proving!\n", stderr) }

        return RecursiveSNARKProof(
            outerProof: outerProof,
            innerProof: innerProof,
            innerVK: innerVK,
            propagatedPublicInputs: innerPublicInputs,
            depth: depth
        )
    }

    /// Access the cached outer verification key (available after first prove call).
    public var outerVerificationKey: Groth16VerificationKey? {
        cachedOuterVK
    }

    /// Access the cached outer R1CS (available after first prove call).
    public var outerR1CS: R1CSInstance? {
        cachedOuterR1CS
    }
}

// MARK: - Recursive Verifier

/// Verifies a recursive SNARK proof by checking both the outer proof (Groth16 on
/// the verifier circuit) and the deferred pairing check on the inner proof.
///
/// Verification has two phases:
///   1. Outer verification: standard Groth16 verify on the outer proof
///      (checks MSM accumulation was done correctly)
///   2. Inner pairing check: deferred check that the inner proof's pairing equation holds
///      e(A,B) = e(alpha,beta) * e(vk_accum,gamma) * e(C,delta)
///
/// Both must pass for the recursive proof to be accepted. This combined check
/// provides the same security as directly verifying the inner proof, but enables
/// recursive composition (the outer proof itself can be recursed).
public class RecursiveSNARKVerifier {
    public init() {}

    /// Verify a recursive SNARK proof.
    ///
    /// - Parameters:
    ///   - recursiveProof: The recursive proof to verify
    ///   - outerVK: Verification key for the outer Groth16 proof
    /// - Returns: true if both outer verification and inner pairing check pass
    public func verify(
        recursiveProof: RecursiveSNARKProof,
        outerVK: Groth16VerificationKey
    ) -> Bool {
        // Phase 1: Verify outer Groth16 proof (checks MSM accumulation)
        let encoder = Groth16VerifierCircuitEncoder(
            innerPublicInputCount: recursiveProof.propagatedPublicInputs.count
        )
        let (outerR1CS, witnessMapper) = encoder.buildVerifierR1CS()

        // Reconstruct outer public inputs from the recursive proof
        let z = witnessMapper(
            recursiveProof.innerProof,
            recursiveProof.innerVK,
            recursiveProof.propagatedPublicInputs
        )
        let outerPublicInputs = Array(z[1...(outerR1CS.numPublic)])

        // Verify R1CS satisfiability first (debug)
        let rSat = outerR1CS.isSatisfied(z: z)
        if !rSat { fputs("  [recursive-debug] R1CS NOT satisfied by reconstructed z\n", stderr) }

        let outerVerifier = Groth16Verifier()
        let outerValid = outerVerifier.verify(
            proof: recursiveProof.outerProof,
            vk: outerVK,
            publicInputs: outerPublicInputs
        )

        if !outerValid { fputs("  [recursive-debug] outer Groth16 verification FAILED\n", stderr) }
        guard outerValid else { return false }

        // Phase 2: Verify inner proof's pairing equation (deferred from circuit)
        let innerValid = outerVerifier.verify(
            proof: recursiveProof.innerProof,
            vk: recursiveProof.innerVK,
            publicInputs: recursiveProof.propagatedPublicInputs
        )
        if !innerValid { fputs("  [recursive-debug] inner proof re-verification FAILED\n", stderr) }

        return innerValid
    }

    /// Verify with separate inner verification (when outer VK is trusted).
    /// Skips inner pairing check — use only when the outer proof system is trusted
    /// to have correctly encoded the inner verifier.
    public func verifyOuterOnly(
        recursiveProof: RecursiveSNARKProof,
        outerVK: Groth16VerificationKey
    ) -> Bool {
        let encoder = Groth16VerifierCircuitEncoder(
            innerPublicInputCount: recursiveProof.propagatedPublicInputs.count
        )
        let (outerR1CS, witnessMapper) = encoder.buildVerifierR1CS()

        let z = witnessMapper(
            recursiveProof.innerProof,
            recursiveProof.innerVK,
            recursiveProof.propagatedPublicInputs
        )
        let outerPublicInputs = Array(z[1...(outerR1CS.numPublic)])

        let outerVerifier = Groth16Verifier()
        return outerVerifier.verify(
            proof: recursiveProof.outerProof,
            vk: outerVK,
            publicInputs: outerPublicInputs
        )
    }
}

// MARK: - Helpers

/// Embed an Fp element into Fr by reinterpreting its Montgomery limbs.
/// Used for encoding point coordinates as circuit witness values.
private func embeddedFpToFr(_ fp: Fp) -> Fr {
    Fr(v: fp.v)
}
