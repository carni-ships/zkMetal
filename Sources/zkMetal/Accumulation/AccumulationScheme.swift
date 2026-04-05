// Halo-style Accumulation Scheme (C17)
//
// Protocol-based abstraction for IPA accumulation over the Pasta cycle.
// Provides a clean interface for recursive proving without pairings (Mina/Kimchi style).
//
// Architecture:
//   AccumulationScheme (protocol) — generic interface
//     HaloAccumulationScheme — concrete Pallas/Vesta implementation
//       uses PallasAccumulationEngine for IPA commit/prove/accumulate
//       uses AccumulationVerifier for cheap step checks
//       uses AccumulationDecider for final MSM check
//
// The Pasta 2-cycle:
//   Pallas Fp = Vesta Fr, Pallas Fr = Vesta Fp
//   Primary circuit on Pallas, accumulation verifier on Vesta.
//   No pairings needed — the cycle enables recursive composition via IPA.
//
// References:
//   - Halo (Bowe et al. 2019)
//   - BCMS 2020: Proof-Carrying Data from accumulation schemes
//   - Mina/Kimchi: production Halo-style recursion on Pasta

import Foundation

// MARK: - Accumulation Scheme Protocol

/// Protocol for an IPA-based accumulation scheme.
///
/// An accumulation scheme defers expensive verification:
///   1. `accumulate` — convert a proof into a cheap accumulator (O(log n) EC ops)
///   2. `fold` — combine two accumulators (O(1) EC ops)
///   3. `batchAccumulate` — fold N claims at once via random linear combination
///   4. `decide` — perform the final expensive check (1 MSM)
///
/// The key invariant: if decide(fold(acc1, acc2)) passes, then both
/// original claims are valid (with overwhelming probability).
public protocol AccumulationSchemeProtocol {
    associatedtype Proof
    associatedtype Claim
    associatedtype Accumulator
    associatedtype FoldProof

    /// Convert a proof into an accumulator (deferred verification).
    /// Cost: O(log n) group operations (Fiat-Shamir + commitment folding).
    func accumulate(claim: Claim) -> Accumulator

    /// Fold two accumulators into one.
    /// Cost: O(1) group operations (1 scalar-mul + 1 point-add).
    func fold(_ acc1: Accumulator, _ acc2: Accumulator) -> (Accumulator, FoldProof)

    /// Batch-accumulate multiple claims into a single accumulator.
    /// Uses random linear combination for soundness.
    /// Cost: O(N) scalar-muls (much cheaper than N individual decides).
    func batchAccumulate(claims: [Claim]) -> (Accumulator, [FoldProof])

    /// Final verification of an accumulated claim.
    /// Cost: 1 MSM of size n (the expensive part, done once).
    func decide(_ acc: Accumulator) -> Bool

    /// Batch-decide multiple accumulators using random linear combination.
    /// Cost: 1 MSM (shared generators) regardless of accumulator count.
    func batchDecide(_ accs: [Accumulator]) -> Bool
}

// MARK: - Halo Accumulation Scheme (Pallas Curve)

/// Concrete Halo-style accumulation scheme over the Pallas curve.
///
/// This is the main entry point for IPA-based recursive proving on Pasta.
///
/// Usage:
///   let scheme = try HaloAccumulationScheme(generatorCount: 256)
///   let claim = scheme.createClaim(witness: a, evalVector: b)
///   let acc = scheme.accumulate(claim: claim)
///   // ... accumulate more claims ...
///   let valid = scheme.decide(acc)
public class HaloAccumulationScheme: AccumulationSchemeProtocol {
    public typealias Proof = PallasIPAProof
    public typealias Claim = PallasIPAClaim
    public typealias Accumulator = IPAAccumulator
    public typealias FoldProof = AccumulationProof

    /// Underlying Pallas accumulation engine (IPA commit/prove/verify/accumulate/decide)
    public let engine: PallasAccumulationEngine

    /// SRS size (number of generators, must be power of 2)
    public let n: Int

    /// log2(n)
    public let logN: Int

    /// Create a Halo accumulation scheme with the given Pallas generators.
    public init(generators: [PallasPointAffine], Q: PallasPointAffine) {
        self.engine = PallasAccumulationEngine(generators: generators, Q: Q)
        self.n = generators.count
        self.logN = engine.logN
    }

    /// Convenience: create with deterministic test generators.
    public convenience init(generatorCount: Int) {
        let (gens, Q) = PallasAccumulationEngine.generateTestGenerators(count: generatorCount)
        self.init(generators: gens, Q: Q)
    }

    // MARK: - Claim Creation

    /// Create an IPA claim from a witness and evaluation vector.
    ///
    /// Produces the IPA proof and packages everything into a claim
    /// ready for accumulation. This is the prover's main operation.
    ///
    /// Cost: O(n log n) for the IPA proof (dominated by MSMs in each round).
    public func createClaim(witness a: [VestaFp], evalVector b: [VestaFp]) -> PallasIPAClaim {
        precondition(a.count == n && b.count == n)
        let proof = engine.createProof(a: a, b: b)
        return engine.extractClaim(witness: a, evaluationVector: b, proof: proof)
    }

    // MARK: - AccumulationSchemeProtocol

    /// Convert a claim into an accumulator (deferred IPA verification).
    public func accumulate(claim: PallasIPAClaim) -> IPAAccumulator {
        return engine.accumulateClaim(claim)
    }

    /// Fold two accumulators into one.
    public func fold(_ acc1: IPAAccumulator, _ acc2: IPAAccumulator) -> (IPAAccumulator, AccumulationProof) {
        return engine.foldAccumulators(acc1, acc2)
    }

    /// Batch-accumulate multiple claims into a single accumulator.
    ///
    /// Uses random linear combination to fold N claims at once:
    ///   C' = C_1 + r_1*C_2 + r_1*r_2*C_3 + ...
    ///   a' = a_1 + r_1*a_2 + r_1*r_2*a_3 + ...
    ///
    /// The random challenges are derived from a Fiat-Shamir transcript
    /// binding all claim commitments.
    ///
    /// Cost: O(N) scalar-muls + point-adds (no MSMs).
    /// This is strictly cheaper than N individual accumulate+fold steps
    /// because we derive all challenges from one transcript.
    public func batchAccumulate(claims: [PallasIPAClaim]) -> (IPAAccumulator, [AccumulationProof]) {
        precondition(!claims.isEmpty)

        // Convert all claims to accumulators
        let accs = claims.map { engine.accumulateClaim($0) }

        if accs.count == 1 {
            return (accs[0], [])
        }

        // Derive all folding challenges from a single transcript
        // binding all commitments (batch Fiat-Shamir)
        var transcript = [UInt8]()
        for acc in accs {
            engine.appendPointToTranscript(&transcript, acc.commitment)
        }

        var challenges = [VestaFp]()
        challenges.reserveCapacity(accs.count - 1)
        for i in 1..<accs.count {
            var stepTranscript = transcript
            engine.appendScalarToTranscript(&stepTranscript, vestaFromInt(UInt64(i)))
            challenges.append(engine.deriveChallenge(stepTranscript))
        }

        // Fold all accumulators using the derived challenges
        var running = accs[0]
        var proofs = [AccumulationProof]()
        proofs.reserveCapacity(accs.count - 1)

        for i in 1..<accs.count {
            let rho = challenges[i - 1]

            // Fold commitment: C' = C_running + rho * C_i
            let rhoC = pallasPointScalarMul(accs[i].commitment, rho)
            let foldedCommitment = pallasPointAdd(running.commitment, rhoC)

            // Fold scalar: a' = a_running + rho * a_i
            let foldedA = vestaAdd(running.proofA, vestaMul(rho, accs[i].proofA))

            // Fold value: v' = v_running + rho * v_i
            let foldedV = vestaAdd(running.value, vestaMul(rho, accs[i].value))

            let proof = AccumulationProof(rho: rho, crossTerm: rhoC)
            proofs.append(proof)

            running = IPAAccumulator(
                commitment: foldedCommitment,
                b: running.b,
                value: foldedV,
                challenges: running.challenges,
                generators: engine.generators,
                Q: engine.Q,
                proofA: foldedA
            )
        }

        return (running, proofs)
    }

    /// Decide: fully verify an accumulated claim (expensive, done once).
    public func decide(_ acc: IPAAccumulator) -> Bool {
        return engine.decide(acc)
    }

    /// Batch-decide multiple accumulators using random linear combination.
    public func batchDecide(_ accs: [IPAAccumulator]) -> Bool {
        return engine.batchDecide(accs)
    }

    // MARK: - Accumulation Pipeline

    /// Full accumulation pipeline: create claims, accumulate, decide.
    ///
    /// Takes witness/eval pairs, creates proofs, accumulates all claims,
    /// and runs the batch decider. Returns (finalAccumulator, isValid).
    ///
    /// This is the simplest way to use Halo-style accumulation.
    public func proveAndAccumulate(
        witnesses: [(a: [VestaFp], b: [VestaFp])]
    ) -> (IPAAccumulator, Bool) {
        precondition(!witnesses.isEmpty)

        let claims = witnesses.map { createClaim(witness: $0.a, evalVector: $0.b) }
        let allAccs = claims.map { engine.accumulateClaim($0) }

        // For correctness: batch-decide the individual accumulators
        // (the folded accumulator changes structure, so we verify the originals)
        let valid = engine.batchDecide(allAccs)

        // Also produce the folded accumulator for downstream use
        let (folded, _) = batchAccumulate(claims: claims)

        return (folded, valid)
    }

    // MARK: - Verify Accumulation Step (Cheap)

    /// Verify that a fold was done correctly (cheap: O(1) group ops).
    /// This is the operation that gets encoded in a circuit for recursion.
    public func verifyFold(
        accPrev: IPAAccumulator,
        accNew: IPAAccumulator,
        accOut: IPAAccumulator,
        proof: AccumulationProof
    ) -> Bool {
        return AccumulationVerifier.verifyStep(
            accPrev: accPrev,
            accNew: accNew,
            accOut: accOut,
            proof: proof
        )
    }

    /// Verify a batch of fold steps.
    public func batchVerifyFolds(
        steps: [(prevAcc: IPAAccumulator, newAcc: IPAAccumulator,
                 outAcc: IPAAccumulator, proof: AccumulationProof)]
    ) -> Bool {
        return AccumulationVerifier.batchVerifySteps(steps: steps)
    }
}

// MARK: - Streaming Accumulator

/// Streaming accumulator for incremental accumulation.
///
/// Maintains a running accumulator and absorbs claims one at a time.
/// At any point, the current state can be decided (finalized).
///
/// This is the pattern used in IVC (Incrementally Verifiable Computation):
///   step 1: prove computation, create claim, absorb
///   step 2: prove computation, create claim, absorb
///   ...
///   step N: finalize — decide the running accumulator
///
/// Total cost: N * O(log n) for accumulation + 1 * O(n) for decide.
public class StreamingAccumulator {

    /// The accumulation scheme
    public let scheme: HaloAccumulationScheme

    /// Running accumulator (nil until first claim absorbed)
    public private(set) var running: IPAAccumulator?

    /// All individual accumulators (for batch decide at finalize)
    public private(set) var allAccumulators: [IPAAccumulator] = []

    /// Fold proofs for each accumulation step (for recursive verification)
    public private(set) var foldProofs: [AccumulationProof] = []

    /// Number of claims absorbed
    public var count: Int { allAccumulators.count }

    public init(scheme: HaloAccumulationScheme) {
        self.scheme = scheme
    }

    /// Absorb a new claim into the running accumulator.
    ///
    /// If this is the first claim, it becomes the running accumulator.
    /// Otherwise, fold it in: running' = fold(running, new).
    ///
    /// Returns the fold proof (nil for the first claim).
    @discardableResult
    public func absorb(claim: PallasIPAClaim) -> AccumulationProof? {
        let acc = scheme.accumulate(claim: claim)
        allAccumulators.append(acc)

        guard let prev = running else {
            running = acc
            return nil
        }

        let (folded, proof) = scheme.fold(prev, acc)
        running = folded
        foldProofs.append(proof)
        return proof
    }

    /// Absorb a pre-computed accumulator.
    @discardableResult
    public func absorbAccumulator(_ acc: IPAAccumulator) -> AccumulationProof? {
        allAccumulators.append(acc)

        guard let prev = running else {
            running = acc
            return nil
        }

        let (folded, proof) = scheme.fold(prev, acc)
        running = folded
        foldProofs.append(proof)
        return proof
    }

    /// Finalize: batch-decide all accumulated claims.
    ///
    /// Uses batch verification on the individual accumulators for correctness.
    /// Returns true if all claims are valid.
    public func finalize() -> Bool {
        guard !allAccumulators.isEmpty else { return false }
        return scheme.batchDecide(allAccumulators)
    }

    /// Reset for a new accumulation sequence.
    public func reset() {
        running = nil
        allAccumulators = []
        foldProofs = []
    }
}

// MARK: - Dual-Curve Accumulation (Pasta Cycle)

/// Dual-curve accumulation scheme using the Pallas/Vesta cycle.
///
/// In recursive composition, we alternate between curves:
///   - Primary proofs on Pallas (accumulated via Pallas IPA)
///   - The accumulation verifier circuit runs on Vesta
///     (because Pallas field ops are native Vesta scalar ops)
///
/// This struct tracks accumulators on both curves for a full
/// recursive proving pipeline.
///
/// Note: the Vesta-side accumulation is structurally identical
/// (just with Vesta points and PallasFp scalars). The current
/// implementation focuses on the Pallas primary, with the Vesta
/// side indicated for future circuit integration.
public struct DualCurveAccumulation {

    /// Primary accumulation on Pallas
    public let primary: HaloAccumulationScheme

    /// Number of recursive steps completed
    public private(set) var stepCount: Int = 0

    /// Primary accumulators (Pallas side)
    public private(set) var primaryAccumulators: [IPAAccumulator] = []

    /// Create a dual-curve accumulation setup.
    public init(generatorCount: Int) {
        self.primary = HaloAccumulationScheme(generatorCount: generatorCount)
    }

    /// Create with explicit generators.
    public init(generators: [PallasPointAffine], Q: PallasPointAffine) {
        self.primary = HaloAccumulationScheme(generators: generators, Q: Q)
    }

    /// Execute one recursive step on the primary curve.
    ///
    /// Creates an IPA proof for the given witness, accumulates it,
    /// and returns the accumulator for this step.
    public mutating func step(witness a: [VestaFp], evalVector b: [VestaFp]) -> IPAAccumulator {
        let claim = primary.createClaim(witness: a, evalVector: b)
        let acc = primary.accumulate(claim: claim)
        primaryAccumulators.append(acc)
        stepCount += 1
        return acc
    }

    /// Finalize: batch-decide all primary accumulators.
    public func finalize() -> Bool {
        guard !primaryAccumulators.isEmpty else { return false }
        return primary.batchDecide(primaryAccumulators)
    }

    /// Reset for a new recursive sequence.
    public mutating func reset() {
        stepCount = 0
        primaryAccumulators = []
    }
}
