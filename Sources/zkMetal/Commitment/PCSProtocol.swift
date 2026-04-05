// PCSProtocol — Unified Polynomial Commitment Scheme abstraction
//
// Allows proof systems (Plonk, Spartan, etc.) to be generic over
// the commitment scheme: KZG, IPA, FRI, Brakedown, etc.
//
// PCSProtocol: single polynomial commit/open/verify
// PCSBatchProtocol: extends with batch operations

import Foundation

// MARK: - Core PCS Protocol

/// A polynomial commitment scheme that can commit to univariate polynomials,
/// open them at arbitrary points, and verify openings.
///
/// Proof systems parameterized by `PCS: PCSProtocol` can swap between
/// KZG (pairing-based, trusted setup), IPA (discrete-log, transparent),
/// FRI (hash-based, transparent), etc.
public protocol PCSProtocol {
    /// The commitment to a polynomial (e.g. elliptic curve point, Merkle root).
    associatedtype Commitment
    /// An opening proof (e.g. KZG witness point, IPA L/R vectors, FRI query proofs).
    associatedtype Opening
    /// Public parameters (e.g. SRS for KZG, generators for IPA, domain config for FRI).
    associatedtype Params

    /// Generate public parameters supporting polynomials up to the given degree.
    func setup(maxDegree: Int) throws -> Params

    /// Commit to a polynomial given as coefficient vector.
    /// `poly[i]` is the coefficient of x^i.
    func commit(poly: [Fr], params: Params) throws -> Commitment

    /// Open a committed polynomial at a single evaluation point.
    /// Returns an opening proof that `poly(point) = evaluation`.
    func open(poly: [Fr], point: Fr, params: Params) throws -> Opening

    /// Verify that a commitment opens to the claimed evaluation at the given point.
    func verify(commitment: Commitment, point: Fr, evaluation: Fr,
                opening: Opening, params: Params) -> Bool
}

// MARK: - Batch PCS Protocol

/// Extension of PCSProtocol with batch commit/open/verify operations.
///
/// Default implementations are provided that loop over single operations,
/// but concrete adapters can override with more efficient batched algorithms
/// (e.g. random linear combination for KZG batch opening).
public protocol PCSBatchProtocol: PCSProtocol {
    /// Commit to multiple polynomials.
    func batchCommit(polys: [[Fr]], params: Params) throws -> [Commitment]

    /// Open multiple polynomials at the same evaluation point.
    func batchOpen(polys: [[Fr]], point: Fr, params: Params) throws -> Opening

    /// Verify a batch opening of multiple commitments at the same point.
    func batchVerify(commitments: [Commitment], point: Fr, evaluations: [Fr],
                     opening: Opening, params: Params) -> Bool
}

/// Default batch implementations that delegate to single operations.
extension PCSBatchProtocol {
    public func batchCommit(polys: [[Fr]], params: Params) throws -> [Commitment] {
        try polys.map { try commit(poly: $0, params: params) }
    }
}
