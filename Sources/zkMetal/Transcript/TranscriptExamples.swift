// Transcript Integration Examples
//
// Shows how existing protocols would use the unified Fiat-Shamir transcript.
// These are illustrative patterns, not replacements for the current engines.

import Foundation

// MARK: - FRI with Transcript

/// Example: FRI commit phase using the transcript for Fiat-Shamir challenges.
///
/// Current pattern (ad-hoc in FRIEngine):
///   var betas = [Fr]()
///   for round in 0..<numRounds {
///       // beta is passed in externally or generated randomly
///       betas.append(someBeta)
///   }
///
/// With transcript:
///   Absorb Merkle root each round, squeeze a deterministic beta.
public func friCommitExample(
    initialEvals: [Fr],
    merkleRoots: [Fr],
    numRounds: Int
) -> (betas: [Fr], transcript: Transcript) {
    let t = Transcript(label: "FRI-commit", backend: .poseidon2)

    // Absorb a hash of the initial evaluations (e.g., Merkle root of LDE)
    t.absorb(merkleRoots[0])

    var betas = [Fr]()
    betas.reserveCapacity(numRounds)

    for round in 0..<numRounds {
        // Squeeze challenge for this folding round
        t.absorbLabel("fold-round")
        let beta = t.squeeze()
        betas.append(beta)

        // After folding, absorb the commitment to the folded polynomial
        if round + 1 < merkleRoots.count {
            t.absorb(merkleRoots[round + 1])
        }
    }

    return (betas: betas, transcript: t)
}

// MARK: - Sumcheck with Transcript

/// Example: Sumcheck protocol using the transcript.
///
/// Current pattern (in SumcheckEngine):
///   Challenges are passed in as an array — caller is responsible for generation.
///
/// With transcript:
///   Each round absorbs the round polynomial coefficients, then squeezes a challenge.
public func sumcheckExample(
    claimedSum: Fr,
    roundPolynomials: [[(Fr, Fr, Fr)]],  // coefficients per round
    numVars: Int
) -> (challenges: [Fr], transcript: Transcript) {
    let t = Transcript(label: "sumcheck", backend: .poseidon2)

    // Absorb the claimed sum
    t.absorb(claimedSum)

    var challenges = [Fr]()
    challenges.reserveCapacity(numVars)

    for round in 0..<numVars {
        // Absorb round polynomial coefficients (e.g., evaluations at 0, 1, 2)
        t.absorbLabel("sumcheck-round-\(round)")
        if round < roundPolynomials.count {
            let (c0, c1, c2) = roundPolynomials[round][0]
            t.absorb(c0)
            t.absorb(c1)
            t.absorb(c2)
        }

        // Squeeze challenge for this variable
        let challenge = t.squeeze()
        challenges.append(challenge)
    }

    return (challenges: challenges, transcript: t)
}

// MARK: - IPA with Transcript

/// Example: IPA (Inner Product Argument) using the transcript.
///
/// Current pattern (in IPAEngine):
///   var transcript = [UInt8]()
///   appendPoint(&transcript, Cbound)
///   appendFr(&transcript, v)
///   // each round: appendPoint L, R, then deriveChallenge via blake3
///
/// With transcript:
///   Same flow but using the unified API. Can use keccak256 for Ethereum compat.
public func ipaExample(
    commitment: Fr,      // hash of commitment point
    innerProduct: Fr,
    roundLR: [(lHash: Fr, rHash: Fr)],  // hashes of L, R points per round
    backend: Transcript.HashBackend = .keccak256  // Ethereum-compatible
) -> (challenges: [Fr], transcript: Transcript) {
    let t = Transcript(label: "IPA-opening", backend: backend)

    // Absorb initial commitment and inner product
    t.absorb(commitment)
    t.absorb(innerProduct)

    var challenges = [Fr]()
    challenges.reserveCapacity(roundLR.count)

    for (lHash, rHash) in roundLR {
        t.absorbLabel("IPA-round")
        t.absorb(lHash)
        t.absorb(rHash)
        let x = t.squeeze()
        challenges.append(x)
    }

    return (challenges: challenges, transcript: t)
}

// MARK: - Lookup (LogUp) with Transcript

/// Example: LogUp lookup argument using the transcript.
///
/// Current pattern (in LookupEngine):
///   beta is passed explicitly as a parameter to prove().
///
/// With transcript:
///   Absorb table and lookup commitments, squeeze beta.
public func lookupExample(
    tableCommitment: Fr,
    lookupCommitment: Fr
) -> (beta: Fr, transcript: Transcript) {
    let t = Transcript(label: "LogUp-lookup", backend: .poseidon2)

    // Absorb commitments
    t.absorb(tableCommitment)
    t.absorb(lookupCommitment)

    // Squeeze the LogUp challenge beta
    t.absorbLabel("beta-challenge")
    let beta = t.squeeze()

    return (beta: beta, transcript: t)
}

// MARK: - Parallel Sub-protocols with Fork

/// Example: Running parallel sub-protocols with forked transcripts.
///
/// When a protocol has independent sub-proofs that can run in parallel,
/// fork the transcript to give each sub-proof its own challenge stream
/// while maintaining security (each fork has a unique domain separator).
public func parallelSubprotocolExample(
    commitments: [Fr]
) -> [[Fr]] {
    let t = Transcript(label: "parallel-protocol", backend: .poseidon2)

    // Absorb shared commitments
    for c in commitments {
        t.absorb(c)
    }

    // Fork for each independent sub-proof
    var allChallenges = [[Fr]]()
    for i in 0..<3 {
        var child = t.fork(label: "sub-proof-\(i)")
        let challenges = child.squeezeChallenges(count: 4)
        allChallenges.append(challenges)
    }

    return allChallenges
}
