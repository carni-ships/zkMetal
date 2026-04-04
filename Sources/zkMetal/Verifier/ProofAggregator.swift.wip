// Proof Aggregator — Aggregates multiple proofs of different types for batch verification
//
// Collects KZG and FRI proofs, then verifies all in one batch call.
// For rollup sequencers: add proofs as they arrive, then verify all at once
// to minimize GPU dispatch overhead and amortize pairing costs.

import Foundation

/// Aggregates multiple proofs of possibly different types for batch verification.
///
/// Usage:
///   let agg = try ProofAggregator(srs: srs, srsSecret: secret)
///   agg.addKZG(item1)
///   agg.addKZG(item2)
///   agg.addFRI(proof1, commitment: commit1)
///   let allValid = try agg.verifyAll()
public class ProofAggregator {
    /// KZG items accumulated for batch verification
    private var kzgItems: [VerificationItem] = []

    /// FRI proofs accumulated for batch verification
    private var friProofs: [(commitment: FRICommitment, queries: [FRIQueryProof])] = []

    /// SRS for KZG verification
    public let srs: [PointAffine]

    /// SRS secret for testing (in production, pairings would be used instead)
    public let srsSecret: Fr

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

    /// Reset the aggregator, clearing all accumulated proofs.
    public func reset() {
        kzgItems.removeAll()
        friProofs.removeAll()
    }

    // MARK: - Statistics

    /// Total number of accumulated proofs.
    public var count: Int {
        kzgItems.count + friProofs.count
    }

    /// Number of KZG proofs in the batch.
    public var kzgCount: Int { kzgItems.count }

    /// Number of FRI proofs in the batch.
    public var friCount: Int { friProofs.count }

    /// Estimated savings from batch verification vs individual verification.
    public var estimatedSavings: String {
        var parts = [String]()
        if kzgItems.count > 1 {
            // Batch: 3 MSMs + 2 scalar muls instead of N pairings
            parts.append("KZG: 3 MSMs instead of \(kzgItems.count) pairings")
        }
        if friProofs.count > 1 {
            let totalQueries = friProofs.reduce(0) { $0 + $1.queries.count }
            parts.append("FRI: 1 GPU dispatch for \(totalQueries) Merkle paths instead of \(friProofs.count) dispatches")
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
    /// KZG proofs are batch-verified using random linear combination.
    /// FRI proofs are batch-verified with shared GPU Merkle verification.
    public func verifyAll() throws -> Bool {
        var allValid = true

        // Batch verify KZG proofs
        if !kzgItems.isEmpty {
            if batchVerifier == nil {
                batchVerifier = try BatchVerifier()
            }

            let kzgValid: Bool
            if kzgItems.count >= 16 {
                // Use GPU MSM for large batches
                kzgValid = try batchVerifier!.batchVerifyKZG(
                    items: kzgItems, srs: srs, srsSecret: srsSecret)
            } else {
                // CPU scalar multiplication for small batches
                kzgValid = try batchVerifier!.batchVerifyKZG(
                    items: kzgItems, srs: srs, srsSecret: srsSecret)
            }
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

        return allValid
    }

    /// Verify all and return detailed results per proof type.
    public func verifyAllDetailed() throws -> (kzgValid: Bool, friValid: Bool, allValid: Bool) {
        var kzgValid = true
        var friValid = true

        if !kzgItems.isEmpty {
            if batchVerifier == nil {
                batchVerifier = try BatchVerifier()
            }
            kzgValid = try batchVerifier!.batchVerifyKZG(
                items: kzgItems, srs: srs, srsSecret: srsSecret)
        }

        if !friProofs.isEmpty {
            if friBatchVerifier == nil {
                friBatchVerifier = try FRIBatchVerifier()
            }
            friValid = try friBatchVerifier!.batchVerify(proofs: friProofs)
        }

        return (kzgValid: kzgValid, friValid: friValid, allValid: kzgValid && friValid)
    }
}
