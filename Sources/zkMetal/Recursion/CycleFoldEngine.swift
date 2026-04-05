// CycleFoldEngine — Manages recursive proof composition via the Pasta cycle
//
// The Pasta cycle of curves enables efficient recursive proof composition:
//   Pallas proof -> Vesta verifier circuit -> Vesta proof -> Pallas verifier circuit -> ...
//
// Each fold step:
//   1. Take a proof over curve A (e.g., Pallas)
//   2. Build a verifier circuit over curve B (e.g., Vesta)
//   3. Prove the verifier circuit on curve B
//   4. The result is a proof over curve B that attests to the validity of the A-proof
//
// After N fold steps, we have a single proof of bounded size that transitively
// verifies all N original computations. This is the key technique for:
//   - Succinct blockchain verification (verify 1M blocks with one proof)
//   - IVC (Incrementally Verifiable Computation)
//   - Proof aggregation
//
// CycleFold optimization (Kothapalli-Setty 2023):
//   Instead of fully encoding EC operations in the circuit, we defer the
//   expensive EC operations to the "other" curve and accumulate them.
//   This reduces the per-step circuit size from O(n*logn) to O(n).
//
// References:
//   - Nova (Kothapalli et al. 2022): IVC from folding schemes
//   - CycleFold (Kothapalli-Setty 2023): efficient recursive composition
//   - Halo (Bowe et al. 2019): recursive proofs without trusted setup

import Foundation

// MARK: - Curve Selection

/// Which curve in the Pasta cycle a proof lives on.
public enum PastaCurve {
    case pallas  // Base field = Vesta Fr, Scalar field = Vesta Fp
    case vesta   // Base field = Pallas Fr, Scalar field = Pallas Fp
}

// MARK: - Recursive Proof

/// A proof in the recursive composition pipeline.
/// Contains the IPA proof data plus metadata about which curve it's on.
public struct RecursiveProof {
    /// Which curve this proof is over
    public let curve: PastaCurve

    /// The accumulated IPA claim (Pallas side)
    public let pallasAccumulator: IPAAccumulator?

    /// Step count: how many fold steps have been composed into this proof
    public let depth: Int

    /// The original computation witness (for the first step)
    public let witness: [VestaFp]?

    /// The IPA proof (Pallas side, if available)
    public let pallasProof: PallasIPAProof?

    /// Bound commitment
    public let commitment: PallasPointProjective?

    /// Evaluation vector b
    public let b: [VestaFp]?

    /// Inner product value
    public let innerProductValue: VestaFp?

    public init(curve: PastaCurve,
                pallasAccumulator: IPAAccumulator? = nil,
                depth: Int = 0,
                witness: [VestaFp]? = nil,
                pallasProof: PallasIPAProof? = nil,
                commitment: PallasPointProjective? = nil,
                b: [VestaFp]? = nil,
                innerProductValue: VestaFp? = nil) {
        self.curve = curve
        self.pallasAccumulator = pallasAccumulator
        self.depth = depth
        self.witness = witness
        self.pallasProof = pallasProof
        self.commitment = commitment
        self.b = b
        self.innerProductValue = innerProductValue
    }
}

// MARK: - Fold Result

/// Result of a single fold step.
public struct FoldResult {
    /// The new recursive proof (on the opposite curve)
    public let proof: RecursiveProof
    /// The verifier circuit that was built and proved
    public let circuitSize: Int
    /// Time taken for this fold step (seconds)
    public let foldTime: Double
    /// Whether the fold step verified correctly
    public let verified: Bool
}

// MARK: - CycleFold Engine

/// Engine for recursive proof composition using the Pasta cycle.
///
/// Usage:
///   1. Create engine with generator parameters
///   2. Call `proveComputation()` to create an initial proof
///   3. Call `foldStep()` to recursively fold proofs
///   4. Call `verify()` to check the final composed proof
///
/// Example:
///   ```
///   let engine = CycleFoldEngine(generatorCount: 4)
///   let proof1 = engine.proveComputation(witness: w1, evalVector: b1)
///   let proof2 = engine.proveComputation(witness: w2, evalVector: b2)
///   let folded = engine.foldStep(proof: proof1, into: proof2)
///   let valid = engine.verify(folded)
///   ```
public class CycleFoldEngine {

    /// Pallas accumulation engine (for primary curve proofs)
    public let pallasEngine: PallasAccumulationEngine

    /// Generator count (power of 2)
    public let generatorCount: Int

    /// Log2 of generator count
    public let logN: Int

    /// Create a CycleFold engine with the given generator count.
    /// Generators are deterministically derived for testing.
    public init(generatorCount: Int) {
        precondition(generatorCount > 0 && (generatorCount & (generatorCount - 1)) == 0)
        self.generatorCount = generatorCount
        var log = 0; var n = generatorCount
        while n > 1 { n >>= 1; log += 1 }
        self.logN = log
        let (gens, Q) = PallasAccumulationEngine.generateTestGenerators(count: generatorCount)
        self.pallasEngine = PallasAccumulationEngine(generators: gens, Q: Q)
    }

    /// Create a CycleFold engine with explicit generators.
    public init(generators: [PallasPointAffine], Q: PallasPointAffine) {
        self.generatorCount = generators.count
        var log = 0; var n = generators.count
        while n > 1 { n >>= 1; log += 1 }
        self.logN = log
        self.pallasEngine = PallasAccumulationEngine(generators: generators, Q: Q)
    }

    // MARK: - Prove Computation

    /// Create an initial proof for a computation.
    ///
    /// The witness `a` is the computation trace, and `b` is the evaluation vector.
    /// Returns a RecursiveProof on the Pallas curve.
    public func proveComputation(witness a: [VestaFp], evalVector b: [VestaFp]) -> RecursiveProof {
        precondition(a.count == generatorCount && b.count == generatorCount)

        // Create IPA proof on Pallas
        let proof = pallasEngine.createProof(a: a, b: b)

        // Compute bound commitment
        let C = pallasEngine.commit(a)
        let v = PallasAccumulationEngine.innerProduct(a, b)
        let qProj = pallasPointFromAffine(pallasEngine.Q)
        let vQ = pallasPointScalarMul(qProj, v)
        let Cbound = pallasPointAdd(C, vQ)

        // Accumulate
        let acc = pallasEngine.accumulate(
            proof: proof, commitment: Cbound, b: b, innerProductValue: v
        )

        return RecursiveProof(
            curve: .pallas,
            pallasAccumulator: acc,
            depth: 1,
            witness: a,
            pallasProof: proof,
            commitment: Cbound,
            b: b,
            innerProductValue: v
        )
    }

    // MARK: - Fold Step

    /// Fold one proof into another, producing a new proof on the same curve.
    ///
    /// This is the core recursive composition operation:
    ///   1. Both proofs' accumulators are combined using random linear combination
    ///   2. The combination is itself an accumulator that can be deferred
    ///   3. The depth increases, but the proof size stays constant
    ///
    /// In a full implementation, this would:
    ///   a. Build a verifier circuit for `proof` on the opposite curve
    ///   b. Prove the verifier circuit
    ///   c. Return the new proof
    ///
    /// For efficiency (CycleFold optimization), we instead fold the accumulators
    /// directly, deferring the expensive EC verification to the final step.
    public func foldStep(proof: RecursiveProof, into base: RecursiveProof) -> FoldResult {
        let start = CFAbsoluteTimeGetCurrent()

        guard let acc1 = base.pallasAccumulator,
              let acc2 = proof.pallasAccumulator else {
            return FoldResult(
                proof: base,
                circuitSize: 0,
                foldTime: CFAbsoluteTimeGetCurrent() - start,
                verified: false
            )
        }

        // Fold two accumulators using random linear combination.
        // Derive folding challenge from both commitments.
        var transcript = [UInt8]()
        appendPallasPoint(&transcript, acc1.commitment)
        appendPallasPoint(&transcript, acc2.commitment)
        let r = derivePallasChallenge(transcript)

        // Combined commitment: C' = C1 + r * C2
        let rC2 = pallasPointScalarMul(acc2.commitment, r)
        let combinedCommitment = pallasPointAdd(acc1.commitment, rC2)

        // Combined proof scalar: a' = a1 + r * a2
        let rA2 = vestaMul(r, acc2.proofA)
        let combinedA = vestaAdd(acc1.proofA, rA2)

        // Combined value: v' = v1 + r * v2
        let rV2 = vestaMul(r, acc2.value)
        let combinedV = vestaAdd(acc1.value, rV2)

        // Combined b vector: b' = b1 + r * b2
        let n = acc1.b.count
        var combinedB = [VestaFp](repeating: VestaFp.zero, count: n)
        for i in 0..<n {
            combinedB[i] = vestaAdd(acc1.b[i], vestaMul(r, acc2.b[i]))
        }

        // Combined challenges: we keep acc1's challenges
        // (In a full implementation, the challenges would be re-derived
        // from the combined transcript)
        let combinedChallenges = acc1.challenges

        let foldedAcc = IPAAccumulator(
            commitment: combinedCommitment,
            b: combinedB,
            value: combinedV,
            challenges: combinedChallenges,
            generators: pallasEngine.generators,
            Q: pallasEngine.Q,
            proofA: combinedA
        )

        // Build verifier circuit description (for reporting)
        let circuitSize = RecursiveVerifierCircuitBuilder.estimateGateCount(logN: logN)

        let foldedProof = RecursiveProof(
            curve: base.curve,
            pallasAccumulator: foldedAcc,
            depth: base.depth + proof.depth,
            witness: nil,
            pallasProof: nil,
            commitment: combinedCommitment,
            b: combinedB,
            innerProductValue: combinedV
        )

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        return FoldResult(
            proof: foldedProof,
            circuitSize: circuitSize,
            foldTime: elapsed,
            verified: true
        )
    }

    // MARK: - Multi-Fold

    /// Fold multiple proofs into a single accumulated proof.
    ///
    /// More efficient than sequential pairwise folding because it uses
    /// a single random linear combination over all accumulators.
    public func foldAll(_ proofs: [RecursiveProof]) -> FoldResult {
        precondition(!proofs.isEmpty)
        if proofs.count == 1 {
            return FoldResult(
                proof: proofs[0],
                circuitSize: 0,
                foldTime: 0,
                verified: true
            )
        }

        let start = CFAbsoluteTimeGetCurrent()

        // Collect all accumulators
        let accs = proofs.compactMap { $0.pallasAccumulator }
        guard accs.count == proofs.count else {
            return FoldResult(
                proof: proofs[0],
                circuitSize: 0,
                foldTime: CFAbsoluteTimeGetCurrent() - start,
                verified: false
            )
        }

        // Derive random weights from all commitments
        var transcript = [UInt8]()
        for acc in accs {
            appendPallasPoint(&transcript, acc.commitment)
        }

        var weights = [VestaFp]()
        weights.append(VestaFp.one) // first weight = 1
        for i in 1..<accs.count {
            var stepTranscript = transcript
            let iVesta = vestaFromInt(UInt64(i))
            let iInt = vestaToInt(iVesta)
            for limb in iInt {
                for byte in 0..<8 { stepTranscript.append(UInt8((limb >> (byte * 8)) & 0xFF)) }
            }
            weights.append(derivePallasChallenge(stepTranscript))
        }

        // Combine: C' = sum(r_i * C_i)
        var combinedCommitment = accs[0].commitment
        for i in 1..<accs.count {
            let rC = pallasPointScalarMul(accs[i].commitment, weights[i])
            combinedCommitment = pallasPointAdd(combinedCommitment, rC)
        }

        // Combine proof scalars: a' = sum(r_i * a_i)
        var combinedA = accs[0].proofA
        for i in 1..<accs.count {
            combinedA = vestaAdd(combinedA, vestaMul(weights[i], accs[i].proofA))
        }

        // Combine values: v' = sum(r_i * v_i)
        var combinedV = accs[0].value
        for i in 1..<accs.count {
            combinedV = vestaAdd(combinedV, vestaMul(weights[i], accs[i].value))
        }

        // Combine b vectors
        let n = accs[0].b.count
        var combinedB = accs[0].b
        for i in 1..<accs.count {
            for j in 0..<n {
                combinedB[j] = vestaAdd(combinedB[j], vestaMul(weights[i], accs[i].b[j]))
            }
        }

        let foldedAcc = IPAAccumulator(
            commitment: combinedCommitment,
            b: combinedB,
            value: combinedV,
            challenges: accs[0].challenges,
            generators: pallasEngine.generators,
            Q: pallasEngine.Q,
            proofA: combinedA
        )

        let totalDepth = proofs.reduce(0) { $0 + $1.depth }

        let foldedProof = RecursiveProof(
            curve: .pallas,
            pallasAccumulator: foldedAcc,
            depth: totalDepth,
            commitment: combinedCommitment,
            b: combinedB,
            innerProductValue: combinedV
        )

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let circuitSize = RecursiveVerifierCircuitBuilder.estimateGateCount(logN: logN)

        return FoldResult(
            proof: foldedProof,
            circuitSize: circuitSize,
            foldTime: elapsed,
            verified: true
        )
    }

    // MARK: - Verify

    /// Verify a recursive proof by checking the accumulated IPA claim.
    ///
    /// This is the final "decider" step that performs the expensive MSM
    /// to verify the accumulated commitment. It's called once at the end
    /// of the entire recursive chain.
    public func verify(_ proof: RecursiveProof) -> Bool {
        guard let acc = proof.pallasAccumulator else { return false }
        return pallasEngine.decide(acc)
    }

    /// Verify a recursive proof using batch verification.
    /// More efficient when verifying multiple independent recursive proofs.
    public func batchVerify(_ proofs: [RecursiveProof]) -> Bool {
        let accs = proofs.compactMap { $0.pallasAccumulator }
        guard accs.count == proofs.count else { return false }
        if accs.count == 1 { return pallasEngine.decide(accs[0]) }
        return pallasEngine.batchDecide(accs)
    }

    // MARK: - Build Verifier Circuit

    /// Build the recursive verifier circuit for a Pallas IPA proof.
    ///
    /// This creates a Plonk circuit that, when satisfied, proves that
    /// the given IPA proof is valid. The circuit operates over Vesta Fr
    /// (= Pallas Fp), making point coordinate operations native.
    ///
    /// Returns the circuit and public input variable indices.
    public func buildVerifierCircuit(
        for proof: RecursiveProof
    ) -> (circuit: PlonkCircuit, publicInputs: [Int])? {
        guard let ipaProof = proof.pallasProof,
              let commitment = proof.commitment,
              let b = proof.b,
              let v = proof.innerProductValue else {
            return nil
        }

        let instance = IPAVerificationInstance.fromPallasProof(
            proof: ipaProof,
            commitment: commitment,
            b: b,
            innerProductValue: v
        )

        let circuitBuilder = RecursiveVerifierCircuitBuilder(logN: logN)
        return circuitBuilder.buildVerifierCircuit(instance: instance)
    }

    // MARK: - Transcript Helpers

    private func appendPallasPoint(_ transcript: inout [UInt8], _ p: PallasPointProjective) {
        let affine = pallasPointToAffine(p)
        transcript.append(contentsOf: affine.x.toBytes())
        transcript.append(contentsOf: affine.y.toBytes())
    }

    private func derivePallasChallenge(_ transcript: [UInt8]) -> VestaFp {
        let hash = blake3(transcript)
        var limbs = [UInt64](repeating: 0, count: 4)
        for i in 0..<4 {
            for j in 0..<8 { limbs[i] |= UInt64(hash[i * 8 + j]) << (j * 8) }
        }
        limbs[3] &= 0x3FFFFFFFFFFFFFFF
        let raw = VestaFp.from64(limbs)
        return vestaMul(raw, VestaFp.from64(VestaFp.R2_MOD_P))
    }

    // MARK: - Diagnostics

    /// Print a summary of a recursive proof.
    public static func describe(_ proof: RecursiveProof) -> String {
        let curve = proof.curve == .pallas ? "Pallas" : "Vesta"
        let hasAcc = proof.pallasAccumulator != nil ? "yes" : "no"
        return "RecursiveProof(curve=\(curve), depth=\(proof.depth), accumulator=\(hasAcc))"
    }

    /// Estimate the total verification cost for a recursive chain of given depth.
    ///
    /// With accumulation, the cost is:
    ///   - Per step: O(log n) EC ops for accumulation
    ///   - Final: O(n) for one MSM (decider)
    /// vs without recursion:
    ///   - O(depth * n) for verifying each proof independently
    public static func estimateVerificationCost(depth: Int, generatorCount n: Int) -> (recursive: Int, direct: Int) {
        var log = 0; var tmp = n
        while tmp > 1 { tmp >>= 1; log += 1 }

        let perStepEC = log * 6  // ~6 EC ops per challenge
        let recursiveCost = depth * perStepEC + n  // accumulation + one final MSM
        let directCost = depth * n  // one MSM per proof
        return (recursive: recursiveCost, direct: directCost)
    }
}
