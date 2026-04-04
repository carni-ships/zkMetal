// Linear Code for Brakedown PCS
// Systematic random linear code: G = [I_k | R] where R is a seeded pseudo-random matrix.
// Encoding: codeword = [message | R^T * message], length n = k / rate.
// The GPU kernel computes R^T * message via matrix-vector multiply.

import Foundation

/// Simple random linear code for Brakedown polynomial commitments.
/// Uses a systematic code: G = [I_k | R] so the first k symbols are the message itself.
/// R is a k x (n-k) pseudo-random matrix generated deterministically from a seed.
public struct LinearCode {
    /// Message length (number of information symbols)
    public let messageLength: Int
    /// Total codeword length (message + redundancy)
    public let codewordLength: Int
    /// Redundancy length (codewordLength - messageLength)
    public let redundancyLength: Int
    /// Code rate = messageLength / codewordLength
    public let rate: Double
    /// Deterministic seed for generating the random matrix R
    public let seed: UInt32

    /// Create a linear code with given message length and rate inverse.
    /// - Parameters:
    ///   - messageLength: Number of information symbols (k)
    ///   - rateInverse: Blowup factor (n/k), typically 4-8
    ///   - seed: Deterministic seed for the random generator matrix
    public init(messageLength: Int, rateInverse: Int = 4, seed: UInt32 = 0xB4A4E) {
        precondition(messageLength > 0, "Message length must be positive")
        precondition(rateInverse >= 2, "Rate inverse must be at least 2")
        self.messageLength = messageLength
        self.codewordLength = messageLength * rateInverse
        self.redundancyLength = codewordLength - messageLength
        self.rate = 1.0 / Double(rateInverse)
        self.seed = seed
    }

    /// CPU-side encode: systematic code, codeword = [message | redundancy].
    /// The redundancy part is computed as R * message where R is the random matrix.
    public func encode(_ message: [Fr]) -> [Fr] {
        precondition(message.count == messageLength, "Message length mismatch")

        var codeword = message  // Systematic: first k symbols are the message
        var redundancy = [Fr](repeating: Fr.zero, count: redundancyLength)

        // Compute redundancy[i] = sum_j R[i][j] * message[j]
        for i in 0..<redundancyLength {
            var acc = Fr.zero
            for j in 0..<messageLength {
                let rElem = generateElement(row: UInt32(i), col: UInt32(j))
                acc = frAdd(acc, frMul(rElem, message[j]))
            }
            redundancy[i] = acc
        }

        codeword.append(contentsOf: redundancy)
        return codeword
    }

    /// Generate a pseudo-random Fr element for position (row, col) in the R matrix.
    /// Deterministic given the seed.
    public func generateElement(row: UInt32, col: UInt32) -> Fr {
        var s = seed ^ (row &* 2654435761) ^ (col &* 2246822519)
        s ^= s >> 16; s &*= 0x45d9f3b; s ^= s >> 16; s &*= 0x45d9f3b; s ^= s >> 16

        var limbs: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)
        limbs.0 = s
        limbs.1 = s ^ 0xDEADBEEF
        s ^= s >> 13; s &*= 0x5bd1e995; s ^= s >> 15
        limbs.2 = s & 0x0FFFFFFF
        s ^= s >> 13; s &*= 0x5bd1e995; s ^= s >> 15
        limbs.3 = s & 0x0FFFFFFF
        s ^= s >> 13; s &*= 0x5bd1e995; s ^= s >> 15
        limbs.4 = s & 0x0FFFFFFF
        s ^= s >> 13; s &*= 0x5bd1e995; s ^= s >> 15
        limbs.5 = s & 0x0FFFFFFF
        s ^= s >> 13; s &*= 0x5bd1e995; s ^= s >> 15
        limbs.6 = s & 0x0FFFFFFF
        s ^= s >> 13; s &*= 0x5bd1e995; s ^= s >> 15
        limbs.7 = s & 0x0FFFFFFF

        return Fr(v: limbs)
    }

    /// Minimum distance bound (for security analysis)
    /// For a random linear code over a large field, the minimum distance
    /// is at least n - k + 1 with high probability (MDS-like).
    public var minimumDistanceBound: Int {
        return redundancyLength + 1
    }
}
