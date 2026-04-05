// PCSProtocol Tests — generic commit-open-verify round-trips
// for KZG, IPA, and FRI adapters.

import zkMetal
import Foundation

// MARK: - Generic test harness

/// Run commit-open-verify round-trip on any PCSProtocol conformer.
/// Creates a small polynomial, commits, opens at a point, and verifies.
private func genericCommitOpenVerify<PCS: PCSProtocol>(
    _ pcs: PCS, label: String, maxDegree: Int = 7
) where PCS.Commitment: Any, PCS.Opening: Any, PCS.Params: Any {
    do {
        let params = try pcs.setup(maxDegree: maxDegree)

        // Polynomial: p(x) = 1 + 2x + 3x^2 + 4x^3
        let poly: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]

        let commitment = try pcs.commit(poly: poly, params: params)

        // Evaluate at z = 5: p(5) = 1 + 10 + 75 + 500 = 586
        let z = frFromInt(5)
        let expectedEval = frFromInt(586)

        let opening = try pcs.open(poly: poly, point: z, params: params)

        let valid = pcs.verify(commitment: commitment, point: z,
                               evaluation: expectedEval, opening: opening, params: params)
        expect(valid, "\(label): commit-open-verify round-trip")
    } catch {
        expect(false, "\(label): error: \(error)")
    }
}

/// Test that a wrong evaluation is rejected.
private func genericWrongEvalRejected<PCS: PCSProtocol>(
    _ pcs: PCS, label: String, maxDegree: Int = 7
) where PCS.Commitment: Any, PCS.Opening: Any, PCS.Params: Any {
    do {
        let params = try pcs.setup(maxDegree: maxDegree)
        let poly: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let z = frFromInt(5)
        let wrongEval = frFromInt(999)  // p(5) = 586, not 999

        let commitment = try pcs.commit(poly: poly, params: params)
        let opening = try pcs.open(poly: poly, point: z, params: params)

        let valid = pcs.verify(commitment: commitment, point: z,
                               evaluation: wrongEval, opening: opening, params: params)
        expect(!valid, "\(label): wrong evaluation rejected")
    } catch {
        expect(false, "\(label): error: \(error)")
    }
}

// MARK: - Public test runner

public func runPCSProtocolTests() {
    suite("PCS Protocol")

    // --- KZG Adapter ---
    do {
        let kzg = KZGUnifiedPCS()
        genericCommitOpenVerify(kzg, label: "KZG")
        genericWrongEvalRejected(kzg, label: "KZG")
    }

    // --- IPA Adapter ---
    do {
        let ipa = IPAUnifiedPCS()
        genericCommitOpenVerify(ipa, label: "IPA")
        genericWrongEvalRejected(ipa, label: "IPA")
    }

    // --- FRI Adapter ---
    do {
        let fri = FRIUnifiedPCS()

        // FRI commit-open-verify round-trip
        // FRI verifies low-degree + Merkle consistency, not point evaluation directly
        do {
            let params = try fri.setup(maxDegree: 7)
            let poly: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
            let z = frFromInt(5)
            let commitment = try fri.commit(poly: poly, params: params)
            let opening = try fri.open(poly: poly, point: z, params: params)

            // FRI verifies that the committed polynomial is low-degree
            // and that the opening is consistent with the commitment
            let valid = fri.verify(commitment: commitment, point: z,
                                   evaluation: frFromInt(586), opening: opening, params: params)
            expect(valid, "FRI: commit-open-verify round-trip")
        } catch {
            expect(false, "FRI: commit-open-verify error: \(error)")
        }

        // FRI wrong commitment should fail
        do {
            let params = try fri.setup(maxDegree: 7)
            let poly1: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
            let poly2: [Fr] = [frFromInt(9), frFromInt(8), frFromInt(7), frFromInt(6)]
            let z = frFromInt(5)

            let commitment1 = try fri.commit(poly: poly1, params: params)
            let opening2 = try fri.open(poly: poly2, point: z, params: params)

            // Opening for poly2 should not verify against commitment for poly1
            let valid = fri.verify(commitment: commitment1, point: z,
                                   evaluation: frFromInt(586), opening: opening2, params: params)
            expect(!valid, "FRI: mismatched commitment/opening rejected")
        } catch {
            expect(false, "FRI: mismatch test error: \(error)")
        }
    }

    // --- KZG Batch Operations ---
    do {
        let kzg = KZGUnifiedPCS()
        do {
            let params = try kzg.setup(maxDegree: 15)

            let poly1: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3)]
            let poly2: [Fr] = [frFromInt(4), frFromInt(5), frFromInt(6)]
            let polys = [poly1, poly2]

            // Batch commit
            let commitments = try kzg.batchCommit(polys: polys, params: params)
            expect(commitments.count == 2, "KZG batch: 2 commitments")

            // Individual commits should match batch
            let c1 = try kzg.commit(poly: poly1, params: params)
            let c2 = try kzg.commit(poly: poly2, params: params)
            let c1Aff = batchToAffine([c1])
            let bc1Aff = batchToAffine([commitments[0]])
            expect(fpToInt(c1Aff[0].x) == fpToInt(bc1Aff[0].x) &&
                   fpToInt(c1Aff[0].y) == fpToInt(bc1Aff[0].y),
                   "KZG batch: commit[0] matches individual")
            let c2Aff = batchToAffine([c2])
            let bc2Aff = batchToAffine([commitments[1]])
            expect(fpToInt(c2Aff[0].x) == fpToInt(bc2Aff[0].x) &&
                   fpToInt(c2Aff[0].y) == fpToInt(bc2Aff[0].y),
                   "KZG batch: commit[1] matches individual")

            // Batch open + verify
            let z = frFromInt(3)
            let opening = try kzg.batchOpen(polys: polys, point: z, params: params)

            // Evaluate: p1(3) = 1+6+27=34, p2(3) = 4+15+54=73
            let evals = [frFromInt(34), frFromInt(73)]
            let valid = kzg.batchVerify(commitments: commitments, point: z,
                                        evaluations: evals, opening: opening, params: params)
            expect(valid, "KZG batch: open+verify")

            // Wrong evaluations should fail
            let wrongEvals = [frFromInt(34), frFromInt(999)]
            let invalid = kzg.batchVerify(commitments: commitments, point: z,
                                          evaluations: wrongEvals, opening: opening, params: params)
            expect(!invalid, "KZG batch: wrong eval rejected")
        } catch {
            expect(false, "KZG batch error: \(error)")
        }
    }

    // --- IPA Batch Operations ---
    do {
        let ipa = IPAUnifiedPCS()
        do {
            let params = try ipa.setup(maxDegree: 7)

            let poly1: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3)]
            let poly2: [Fr] = [frFromInt(4), frFromInt(5), frFromInt(6)]
            let polys = [poly1, poly2]

            // Batch commit
            let commitments = try ipa.batchCommit(polys: polys, params: params)
            expect(commitments.count == 2, "IPA batch: 2 commitments")

            // Individual commits should match batch
            let c1 = try ipa.commit(poly: poly1, params: params)
            let c2 = try ipa.commit(poly: poly2, params: params)
            let c1Aff = batchToAffine([c1])
            let bc1Aff = batchToAffine([commitments[0]])
            expect(fpToInt(c1Aff[0].x) == fpToInt(bc1Aff[0].x) &&
                   fpToInt(c1Aff[0].y) == fpToInt(bc1Aff[0].y),
                   "IPA batch: commit[0] matches individual")
            let c2Aff = batchToAffine([c2])
            let bc2Aff = batchToAffine([commitments[1]])
            expect(fpToInt(c2Aff[0].x) == fpToInt(bc2Aff[0].x) &&
                   fpToInt(c2Aff[0].y) == fpToInt(bc2Aff[0].y),
                   "IPA batch: commit[1] matches individual")

            // Batch open + verify
            let z = frFromInt(3)
            let opening = try ipa.batchOpen(polys: polys, point: z, params: params)

            // Combined eval with gamma=13: 34 + 13*73 = 34 + 949 = 983
            let combinedEval = frFromInt(983)
            // Combine commitments with gamma=13
            let gamma = frFromInt(13)
            let combinedCommitment = pointAdd(commitments[0],
                                              cPointScalarMul(commitments[1], gamma))
            let valid = ipa.verify(commitment: combinedCommitment, point: z,
                                   evaluation: combinedEval, opening: opening, params: params)
            expect(valid, "IPA batch: open+verify combined")
        } catch {
            expect(false, "IPA batch error: \(error)")
        }
    }
}
