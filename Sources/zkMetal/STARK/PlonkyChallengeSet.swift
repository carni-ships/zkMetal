// PlonkyChallengeSet — Challenge generation for Plonky3-style STARKs
//
// Contains all verifier challenges needed for a Plonky3-compatible STARK:
//   - alpha: random linear combination for constraint composition
//   - zeta: out-of-domain (OOD) evaluation point
//   - friChallenges: folding challenges for FRI protocol
//
// Challenges are derived from a Fiat-Shamir transcript (Poseidon2 duplex sponge)
// after absorbing the trace commitment, matching Plonky3's BabyBearPoseidon2 flow.

import Foundation

// MARK: - PlonkyChallengeSet

/// All verifier challenges for a Plonky3-style STARK proof.
///
/// Generated deterministically from the Fiat-Shamir transcript after the prover
/// commits to the execution trace. The challenges bind the verifier's randomness
/// to the committed trace, ensuring soundness.
///
/// Challenge derivation order (matches Plonky3):
///   1. Absorb trace commitment (Merkle root) into transcript
///   2. Squeeze alpha (constraint composition)
///   3. Squeeze zeta (OOD evaluation point)
///   4. Squeeze FRI folding challenges (one per FRI round)
public struct PlonkyChallengeSet {
    /// Random linear combination challenge for composing multiple constraints
    /// into a single quotient polynomial: Q(x) = sum_i alpha^i * C_i(x).
    public let alpha: Bb

    /// Out-of-domain evaluation point where the verifier checks the quotient.
    /// Must not be in the trace domain (i.e., not a power of the trace generator).
    public let zeta: Bb

    /// FRI folding challenges: one per FRI round.
    /// Each challenge is used to fold the polynomial by the FRI fold factor.
    /// Length = ceil(log(traceLength) / log(foldFactor)).
    public let friChallenges: [Bb]

    public init(alpha: Bb, zeta: Bb, friChallenges: [Bb]) {
        self.alpha = alpha
        self.zeta = zeta
        self.friChallenges = friChallenges
    }

    // MARK: - Generation from Plonky3 Challenger

    /// Generate a complete challenge set from a Plonky3 challenger after trace commitment.
    ///
    /// This follows Plonky3's exact protocol flow:
    ///   1. The caller has already observed all prior protocol messages
    ///   2. Observe the trace commitment (Merkle root digest)
    ///   3. Squeeze alpha, zeta, and FRI challenges in order
    ///
    /// - Parameters:
    ///   - challenger: Plonky3 duplex sponge challenger with prior state
    ///   - traceCommitment: Merkle root of the committed trace (8 BabyBear elements)
    ///   - logTraceLength: log2 of the trace length (determines number of FRI rounds)
    ///   - config: Plonky3 AIR configuration (determines FRI parameters)
    /// - Returns: Complete challenge set for the STARK proof
    public static func generateFromChallenger(
        _ challenger: Plonky3Challenger,
        traceCommitment: [Bb],
        logTraceLength: Int,
        config: Plonky3AIRConfig = .sp1Default
    ) -> PlonkyChallengeSet {
        // Step 1: Absorb trace commitment into transcript
        challenger.observeDigest(traceCommitment)

        // Step 2: Squeeze alpha (constraint composition challenge)
        let alpha = challenger.sample()

        // Step 3: Squeeze zeta (OOD evaluation point)
        let zeta = challenger.sample()

        // Step 4: Squeeze FRI folding challenges
        // Number of FRI rounds = ceil(logTraceLength / logFoldFactor)
        // With fold-by-4, each round reduces degree by factor 4
        let logFoldFactor = Plonky3AIRConfig.friLogFoldFactor
        let numFRIRounds = (logTraceLength + logFoldFactor - 1) / logFoldFactor
        let friChallenges = challenger.sampleSlice(numFRIRounds)

        return PlonkyChallengeSet(
            alpha: alpha,
            zeta: zeta,
            friChallenges: friChallenges
        )
    }

    /// Generate a challenge set from a Fiat-Shamir transcript and trace commitment bytes.
    ///
    /// This is a convenience method that wraps the Plonky3 challenger flow for callers
    /// who work with raw byte commitments rather than BabyBear digest elements.
    ///
    /// - Parameters:
    ///   - transcript: A FiatShamirTranscript (any hasher backend)
    ///   - traceCommitment: Raw bytes of the trace commitment (e.g., Merkle root)
    ///   - logTraceLength: log2 of trace length
    ///   - numFRIRounds: Number of FRI folding rounds (if nil, computed from config)
    /// - Returns: Complete challenge set
    public static func generateFromTranscript<H: TranscriptHasher>(
        _ transcript: inout FiatShamirTranscript<H>,
        traceCommitment: [UInt8],
        logTraceLength: Int,
        numFRIRounds: Int? = nil
    ) -> PlonkyChallengeSet {
        // Absorb trace commitment
        transcript.appendMessage(label: "trace-commitment", data: traceCommitment)

        // Squeeze alpha
        let alphaBytes = transcript.squeeze("alpha", byteCount: 4)
        let alphaVal = UInt32(alphaBytes[0]) |
                       (UInt32(alphaBytes[1]) << 8) |
                       (UInt32(alphaBytes[2]) << 16) |
                       (UInt32(alphaBytes[3]) << 24)
        let alpha = Bb(v: alphaVal % Bb.P)

        // Squeeze zeta
        let zetaBytes = transcript.squeeze("zeta", byteCount: 4)
        let zetaVal = UInt32(zetaBytes[0]) |
                      (UInt32(zetaBytes[1]) << 8) |
                      (UInt32(zetaBytes[2]) << 16) |
                      (UInt32(zetaBytes[3]) << 24)
        let zeta = Bb(v: zetaVal % Bb.P)

        // Squeeze FRI challenges
        let friRounds = numFRIRounds ?? ((logTraceLength + 1) / 2)
        var friChallenges = [Bb]()
        friChallenges.reserveCapacity(friRounds)
        for i in 0..<friRounds {
            let friBytes = transcript.squeeze("fri-\(i)", byteCount: 4)
            let friVal = UInt32(friBytes[0]) |
                         (UInt32(friBytes[1]) << 8) |
                         (UInt32(friBytes[2]) << 16) |
                         (UInt32(friBytes[3]) << 24)
            friChallenges.append(Bb(v: friVal % Bb.P))
        }

        return PlonkyChallengeSet(
            alpha: alpha,
            zeta: zeta,
            friChallenges: friChallenges
        )
    }
}
