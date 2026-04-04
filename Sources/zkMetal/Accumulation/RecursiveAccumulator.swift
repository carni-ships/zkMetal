// Recursive Proof Composition via Pasta Cycle
//
// The Pasta curves form a 2-cycle:
//   - Pallas base field (Fp) = Vesta scalar field (Fr)
//   - Pallas scalar field (Fr) = Vesta base field (Fp)
//
// This enables recursive composition:
//   - Primary: prove computation on Pallas (scalars in VestaFp)
//   - Secondary: verify Pallas proof inside a Vesta circuit
//   - The fold/accumulate step for Pallas proofs uses Pallas curve operations
//     (which are scalar-field operations from Vesta's perspective)
//
// Protocol for one recursive step:
//   1. Produce a Pallas IPA proof for current computation step
//   2. Convert to an accumulator (cheap: Fiat-Shamir + EC ops)
//   3. Collect all accumulators; at the end, batch-decide with one MSM
//
// References:
//   - Halo (Bowe et al. 2019): recursive proof without trusted setup
//   - Pasta curves (Zcash): cycle of curves enabling efficient recursion
//   - BCMS 2020: Proof-Carrying Data from accumulation schemes

import Foundation

// MARK: - Recursive Step Result

/// Result of one recursive accumulation step.
public struct RecursiveStepResult {
    /// The accumulator for this step
    public let accumulator: IPAAccumulator
    /// The IPA proof for this step
    public let proof: PallasIPAProof
    /// Bound commitment (C + v*Q) for this step
    public let boundCommitment: PallasPointProjective
    /// Inner product value for this step
    public let innerProductValue: VestaFp
    /// Step index
    public let stepIndex: Int
}

// MARK: - Recursive Prover

/// Recursive prover using the Pasta cycle.
///
/// Demonstrates IPA accumulation: instead of verifying N proofs independently,
/// accumulate them and batch-verify at the end.
///
/// The accumulation benefit:
///   - Each step: O(log n) EC ops to produce accumulator (vs O(n) for full verify)
///   - Final batch-decide: 1 MSM of size n (shared generators) for all N accumulators
///   - Total: N * O(log n) + O(n) vs N * O(n) for individual verification
///
/// Usage:
///   1. Call `step()` for each computation (produces a proof, accumulates it)
///   2. After N steps, call `finalize()` to batch-verify all accumulated claims
public class RecursiveProver {
    /// Pallas accumulation engine (primary curve)
    public let pallasEngine: PallasAccumulationEngine

    /// All accumulators collected during steps
    public private(set) var accumulators: [IPAAccumulator] = []

    /// Number of steps accumulated
    public var stepCount: Int { accumulators.count }

    /// Create a recursive prover with given Pallas generators.
    public init(generators: [PallasPointAffine], Q: PallasPointAffine) {
        self.pallasEngine = PallasAccumulationEngine(generators: generators, Q: Q)
    }

    /// Convenience: create with test generators of given size.
    public convenience init(generatorCount n: Int) {
        let (gens, Q) = PallasAccumulationEngine.generateTestGenerators(count: n)
        self.init(generators: gens, Q: Q)
    }

    /// Process one computation step:
    ///   1. Create IPA proof for the given witness vector `a` with evaluation vector `b`
    ///   2. Convert to accumulator (deferred verification claim)
    ///   3. Return the step result
    ///
    /// The witness `a` typically represents the computation trace (e.g., Poseidon2 state),
    /// and `b` is the evaluation vector (e.g., powers of a challenge point).
    public func step(a: [VestaFp], b: [VestaFp]) -> RecursiveStepResult {
        let n = a.count
        precondition(n == b.count && n == pallasEngine.generators.count)

        // Create IPA proof
        let proof = pallasEngine.createProof(a: a, b: b)

        // Compute bound commitment
        let C = pallasEngine.commit(a)
        let v = PallasAccumulationEngine.innerProduct(a, b)
        let qProj = pallasPointFromAffine(pallasEngine.Q)
        let vQ = pallasPointScalarMul(qProj, v)
        let Cbound = pallasPointAdd(C, vQ)

        // Accumulate (cheap: Fiat-Shamir + EC ops, no full verification)
        let acc = pallasEngine.accumulate(
            proof: proof,
            commitment: Cbound,
            b: b,
            innerProductValue: v
        )
        accumulators.append(acc)

        return RecursiveStepResult(
            accumulator: acc,
            proof: proof,
            boundCommitment: Cbound,
            innerProductValue: v,
            stepIndex: accumulators.count - 1
        )
    }

    /// Finalize: batch-verify all accumulated claims.
    ///
    /// Uses random linear combination to check all N accumulators
    /// with cost dominated by one MSM (shared generators).
    ///
    /// Returns true if all accumulated claims are valid.
    public func finalize() -> Bool {
        guard !accumulators.isEmpty else { return false }
        if accumulators.count == 1 {
            return pallasEngine.decide(accumulators[0])
        }
        return pallasEngine.batchDecide(accumulators)
    }

    /// Reset the prover for a fresh sequence of steps.
    public func reset() {
        accumulators = []
    }
}

// MARK: - Iterated Hash Example

/// Demonstrates recursive accumulation with iterated "hashing" (simple MiMC-like).
///
/// Each step:
///   state_{i+1} = f(state_i) where f is a degree-3 map (cube in the field)
///
/// The witness for step i is [state_i, state_i^2, state_i^3, state_{i+1}]
/// (padded to power-of-2 length matching generator count).
///
/// This is a simplified demonstration -- a real circuit would use Poseidon2 or similar.
public class IteratedHashDemo {

    /// Run N steps of iterated hashing with recursive accumulation.
    /// Returns (finalState, allValid) where allValid means the batch decider passes.
    public static func run(steps: Int, generatorCount: Int = 4) -> (VestaFp, Bool) {
        let prover = RecursiveProver(generatorCount: generatorCount)

        // Initial state
        var state = vestaFromInt(42)

        for _ in 0..<steps {
            // Compute next state: f(state) = state^3 + state + 5 (MiMC-like)
            let s2 = vestaMul(state, state)
            let s3 = vestaMul(s2, state)
            let nextState = vestaAdd(vestaAdd(s3, state), vestaFromInt(5))

            // Build witness vector (pad to generatorCount)
            var a = [VestaFp](repeating: VestaFp.zero, count: generatorCount)
            a[0] = state
            if generatorCount > 1 { a[1] = s2 }
            if generatorCount > 2 { a[2] = s3 }
            if generatorCount > 3 { a[3] = nextState }

            // Evaluation vector: simple powers [1, 2, 3, ...] for testing
            var b = [VestaFp](repeating: VestaFp.zero, count: generatorCount)
            for i in 0..<generatorCount {
                b[i] = vestaFromInt(UInt64(i + 1))
            }

            _ = prover.step(a: a, b: b)
            state = nextState
        }

        // Batch-verify all accumulated proofs at once
        let valid = prover.finalize()
        return (state, valid)
    }
}
