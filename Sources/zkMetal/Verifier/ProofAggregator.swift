// Proof Aggregator — Aggregates multiple proofs of different types for batch verification
//
// Collects KZG, FRI, and IPA proofs, then verifies all in one batch call.
// For rollup sequencers: add proofs as they arrive, then verify all at once
// to minimize GPU dispatch overhead and amortize pairing costs.
//
// Supports mixed proof types in a single batch:
//   - KZG opening proofs (batched via random linear combination + MSM)
//   - FRI proofs (batched via shared GPU Merkle verification)
//   - IPA proofs (batched via individual verify + random weight accumulation)

import Foundation

/// Aggregates multiple proofs of possibly different types for batch verification.
///
/// Usage:
///   let agg = ProofAggregator(srs: srs, srsSecret: secret)
///   agg.addKZG(item1)
///   agg.addKZG(item2)
///   agg.addFRI(commitment: commit1, queries: queries1)
///   agg.addIPA(commitment: c, b: b, value: v, proof: p)
///   let allValid = try agg.verifyAll()
///
/// For rollup sequencers processing hundreds of proofs per block,
/// this gives near-linear speedup: N proofs verified with the cost of ~3 MSMs.
public class ProofAggregator {
    /// KZG items accumulated for batch verification
    private var kzgItems: [VerificationItem] = []

    /// FRI proofs accumulated for batch verification
    private var friProofs: [(commitment: FRICommitment, queries: [FRIQueryProof])] = []

    /// IPA proofs accumulated for batch verification
    private var ipaItems: [BatchVerifier.IPAVerificationItem] = []

    /// SRS for KZG verification
    public let srs: [PointAffine]

    /// SRS secret for testing (in production, pairings would be used instead)
    public let srsSecret: Fr

    /// IPA generators + Q point (optional, needed only if IPA proofs are added)
    public var ipaGenerators: [PointAffine]?
    public var ipaQ: PointAffine?
    public var ipaEngine: IPAEngine?

    /// Engines (lazily initialized, shared across all verifications)
    private var batchVerifier: BatchVerifier?
    private var friBatchVerifier: FRIBatchVerifier?

    /// Create an aggregator with the given SRS parameters.
    /// - Parameters:
    ///   - srs: Structured Reference String points
    ///   - srsSecret: The SRS secret (for testing; production uses pairings)
    public init(srs: [PointAffine], srsSecret: Fr) {
        self.srs = srs
        self.srsSecret = srsSecret
    }

    // MARK: - Add proofs

    /// Add a KZG opening proof to the batch.
    public func addKZG(_ item: VerificationItem) {
        kzgItems.append(item)
    }

    /// Add multiple KZG proofs at once.
    public func addKZGBatch(_ items: [VerificationItem]) {
        kzgItems.append(contentsOf: items)
    }

    /// Add a FRI proof with its commitment and query proofs to the batch.
    public func addFRI(commitment: FRICommitment, queries: [FRIQueryProof]) {
        friProofs.append((commitment: commitment, queries: queries))
    }

    /// Add an IPA proof to the batch.
    public func addIPA(commitment: PointProjective, b: [Fr],
                       innerProductValue: Fr, proof: IPAProof) {
        ipaItems.append(BatchVerifier.IPAVerificationItem(
            commitment: commitment, b: b,
            innerProductValue: innerProductValue, proof: proof))
    }

    /// Reset the aggregator, clearing all accumulated proofs.
    public func reset() {
        kzgItems.removeAll()
        friProofs.removeAll()
        ipaItems.removeAll()
    }

    // MARK: - Statistics

    /// Total number of accumulated proofs.
    public var count: Int {
        kzgItems.count + friProofs.count + ipaItems.count
    }

    /// Number of KZG proofs in the batch.
    public var kzgCount: Int { kzgItems.count }

    /// Number of FRI proofs in the batch.
    public var friCount: Int { friProofs.count }

    /// Number of IPA proofs in the batch.
    public var ipaCount: Int { ipaItems.count }

    /// Estimated savings from batch verification vs individual verification.
    public var estimatedSavings: String {
        var parts = [String]()
        if kzgItems.count > 1 {
            if kzgItems.count >= BatchVerifier.gpuMSMThreshold {
                parts.append("KZG: 3 GPU MSMs instead of \(kzgItems.count) pairings")
            } else {
                parts.append("KZG: 1 batch check instead of \(kzgItems.count) pairings")
            }
        }
        if friProofs.count > 1 {
            let totalQueries = friProofs.reduce(0) { $0 + $1.queries.count }
            parts.append("FRI: 1 GPU dispatch for \(totalQueries) Merkle paths instead of \(friProofs.count) dispatches")
        }
        if ipaItems.count > 1 {
            parts.append("IPA: batch verify \(ipaItems.count) proofs with random LC")
        }
        if parts.isEmpty {
            return "no batch savings (single proof)"
        }
        return parts.joined(separator: "; ")
    }

    // MARK: - Verify all

    /// Verify all accumulated proofs in one batch.
    /// Returns true iff ALL proofs are valid (with overwhelming probability).
    ///
    /// KZG proofs: batched via random linear combination.
    ///   - N >= 16: GPU MSM path (3 MSMs + 2 scalar muls)
    ///   - N < 16: CPU scalar multiplication path
    /// FRI proofs: batched via shared GPU Merkle verification.
    /// IPA proofs: batched via individual verify + random weight accumulation.
    public func verifyAll() throws -> Bool {
        var allValid = true

        // Batch verify KZG proofs (adaptive CPU/GPU)
        if !kzgItems.isEmpty {
            if batchVerifier == nil {
                batchVerifier = try BatchVerifier()
            }

            let kzgValid = try batchVerifier!.batchVerifyKZGAdaptive(
                items: kzgItems, srs: srs, srsSecret: srsSecret)
            allValid = allValid && kzgValid
        }

        // Batch verify FRI proofs
        if !friProofs.isEmpty {
            if friBatchVerifier == nil {
                friBatchVerifier = try FRIBatchVerifier()
            }

            let friValid = try friBatchVerifier!.batchVerify(proofs: friProofs)
            allValid = allValid && friValid
        }

        // Batch verify IPA proofs
        if !ipaItems.isEmpty {
            if batchVerifier == nil {
                batchVerifier = try BatchVerifier()
            }
            if let gens = ipaGenerators, let q = ipaQ, let eng = ipaEngine {
                let ipaValid = batchVerifier!.batchVerifyIPA(
                    items: ipaItems, generators: gens, Q: q, ipaEngine: eng)
                allValid = allValid && ipaValid
            } else {
                // No IPA engine configured -- cannot verify IPA proofs
                allValid = false
            }
        }

        return allValid
    }

    /// Verify all and return detailed results per proof type.
    public func verifyAllDetailed() throws -> (kzgValid: Bool, friValid: Bool,
                                                ipaValid: Bool, allValid: Bool) {
        var kzgValid = true
        var friValid = true
        var ipaValid = true

        if !kzgItems.isEmpty {
            if batchVerifier == nil {
                batchVerifier = try BatchVerifier()
            }
            kzgValid = try batchVerifier!.batchVerifyKZGAdaptive(
                items: kzgItems, srs: srs, srsSecret: srsSecret)
        }

        if !friProofs.isEmpty {
            if friBatchVerifier == nil {
                friBatchVerifier = try FRIBatchVerifier()
            }
            friValid = try friBatchVerifier!.batchVerify(proofs: friProofs)
        }

        if !ipaItems.isEmpty {
            if batchVerifier == nil {
                batchVerifier = try BatchVerifier()
            }
            if let gens = ipaGenerators, let q = ipaQ, let eng = ipaEngine {
                ipaValid = batchVerifier!.batchVerifyIPA(
                    items: ipaItems, generators: gens, Q: q, ipaEngine: eng)
            } else {
                ipaValid = false
            }
        }

        return (kzgValid: kzgValid, friValid: friValid,
                ipaValid: ipaValid, allValid: kzgValid && friValid && ipaValid)
    }
}
