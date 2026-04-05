// Expander-Code-based Linear Code for Brakedown PCS
// Reference: Golovnev, Lee, Setty, Thaler, Wahby — eprint 2021/1043
//
// The generator matrix is derived from a left-regular bipartite expander graph.
// Each output element depends on exactly D input elements (D ≈ 10),
// making encoding O(D * n) instead of O(n^2) for dense random codes.
//
// The code is systematic: G = [I_k | R] where R is a sparse matrix
// with exactly D non-zero entries per row, determined by the expander graph.

import Foundation

// MARK: - Expander Graph

/// A left-regular bipartite expander graph.
/// Left vertices: 0..<leftSize  (input / message symbols)
/// Right vertices: 0..<rightSize (redundancy symbols)
/// Each right vertex has exactly `degree` left neighbors.
/// Neighbors are chosen pseudo-randomly from a seed for reproducibility.
public struct ExpanderGraph {
    /// Number of left vertices (message length)
    public let leftSize: Int
    /// Number of right vertices (redundancy length)
    public let rightSize: Int
    /// Left-regularity degree: each right vertex connects to exactly this many left vertices
    public let degree: Int
    /// Seed for deterministic neighbor generation
    public let seed: UInt32

    /// Neighbor list: neighbors[rightIdx * degree + d] = left vertex index
    /// Stored contiguously for cache-friendly access.
    public let neighbors: [UInt32]
    /// Coefficients: coefficients[rightIdx * degree + d] = random non-zero field element
    /// Each edge has a random coefficient so the code has good distance properties.
    public let coefficients: [Fr]

    /// Build an expander graph with given parameters.
    /// Uses a hash-based PRNG seeded by (seed, rightIdx, d) for each edge.
    public init(leftSize: Int, rightSize: Int, degree: Int, seed: UInt32 = 0xB4A4E) {
        precondition(leftSize > 0 && rightSize > 0 && degree > 0)
        precondition(degree <= leftSize, "Degree cannot exceed left size")

        self.leftSize = leftSize
        self.rightSize = rightSize
        self.degree = degree
        self.seed = seed

        var nbrs = [UInt32](repeating: 0, count: rightSize * degree)
        var coeffs = [Fr](repeating: Fr.zero, count: rightSize * degree)

        for i in 0..<rightSize {
            // For each right vertex, select `degree` distinct left neighbors
            // Using rejection sampling with a small buffer to avoid duplicates
            var selected = Set<UInt32>()
            selected.reserveCapacity(degree)
            var attempt: UInt32 = 0

            for d in 0..<degree {
                // Find a unique left neighbor
                var leftIdx: UInt32
                repeat {
                    leftIdx = ExpanderGraph.prngNeighbor(
                        seed: seed, rightIdx: UInt32(i),
                        edgeIdx: UInt32(d), attempt: attempt,
                        leftSize: UInt32(leftSize)
                    )
                    attempt &+= 1
                } while selected.contains(leftIdx)
                selected.insert(leftIdx)
                nbrs[i * degree + d] = leftIdx

                // Generate a random non-zero coefficient for this edge
                coeffs[i * degree + d] = ExpanderGraph.prngCoefficient(
                    seed: seed, rightIdx: UInt32(i), edgeIdx: UInt32(d)
                )
            }
        }

        self.neighbors = nbrs
        self.coefficients = coeffs
    }

    /// Pseudo-random neighbor selection: hash (seed, rightIdx, edgeIdx, attempt) -> leftIdx
    @inline(__always)
    static func prngNeighbor(seed: UInt32, rightIdx: UInt32, edgeIdx: UInt32,
                              attempt: UInt32, leftSize: UInt32) -> UInt32 {
        var s = seed
        s ^= rightIdx &* 2654435761
        s ^= edgeIdx &* 2246822519
        s ^= attempt &* 3266489917
        s ^= s >> 16; s &*= 0x45d9f3b
        s ^= s >> 16; s &*= 0x45d9f3b
        s ^= s >> 16
        return s % leftSize
    }

    /// Pseudo-random coefficient: hash (seed, rightIdx, edgeIdx) -> small Fr element
    @inline(__always)
    static func prngCoefficient(seed: UInt32, rightIdx: UInt32, edgeIdx: UInt32) -> Fr {
        var s = seed ^ 0xCAFEBABE
        s ^= rightIdx &* 2654435761
        s ^= edgeIdx &* 2246822519
        s ^= s >> 16; s &*= 0x45d9f3b
        s ^= s >> 16; s &*= 0x45d9f3b
        s ^= s >> 16

        // Use a small non-zero coefficient (keeps it cheap, still random enough for distance)
        // Values in [1, 2^28) to stay well within Fr modulus
        let val = (s & 0x0FFFFFFF) | 1  // ensure non-zero
        return frFromInt(UInt64(val))
    }
}

// MARK: - Expander Code

/// Linear code based on expander graph adjacency.
/// Encoding: codeword = [message | R * message]
/// where R is the sparse matrix defined by the expander graph.
/// Each row of R has exactly `degree` non-zero entries.
///
/// Properties:
/// - Encoding time: O(degree * n) field multiplications (vs O(n^2) for dense codes)
/// - Rate: messageLength / codewordLength = 1 / rateInverse
/// - Distance: Ω(n) with high probability for random expanders (Sipser-Spielman)
/// - GPU-friendly: each output element is an independent sparse dot product
public struct ExpanderCode {
    /// Message length (number of information symbols)
    public let messageLength: Int
    /// Total codeword length (message + redundancy)
    public let codewordLength: Int
    /// Redundancy length = codewordLength - messageLength
    public let redundancyLength: Int
    /// Code rate = messageLength / codewordLength
    public let rate: Double
    /// Sparsity degree: each redundancy element depends on exactly this many message elements
    public let degree: Int
    /// The underlying expander graph
    public let graph: ExpanderGraph
    /// Deterministic seed
    public let seed: UInt32

    /// Create an expander code.
    /// - Parameters:
    ///   - messageLength: Number of information symbols (k)
    ///   - rateInverse: Blowup factor (n/k), typically 4-8. Default 4.
    ///   - degree: Expander degree (edges per right vertex). Default 10.
    ///             Higher degree = better distance but slower encoding.
    ///   - seed: Deterministic seed for the expander graph
    public init(messageLength: Int, rateInverse: Int = 4, degree: Int = 10, seed: UInt32 = 0xB4A4E) {
        precondition(messageLength > 0, "Message length must be positive")
        precondition(rateInverse >= 2, "Rate inverse must be at least 2")
        precondition(degree >= 2, "Degree must be at least 2")

        self.messageLength = messageLength
        self.codewordLength = messageLength * rateInverse
        self.redundancyLength = codewordLength - messageLength
        self.rate = 1.0 / Double(rateInverse)
        self.degree = min(degree, messageLength)  // Can't exceed message length
        self.seed = seed

        self.graph = ExpanderGraph(
            leftSize: messageLength,
            rightSize: redundancyLength,
            degree: self.degree,
            seed: seed
        )
    }

    /// CPU-side encode: systematic code, codeword = [message | redundancy].
    /// The redundancy part is computed via sparse matrix-vector multiply using the expander graph.
    /// Cost: O(degree * redundancyLength) field multiplications.
    public func encode(_ message: [Fr]) -> [Fr] {
        precondition(message.count == messageLength, "Message length mismatch")

        var codeword = message  // Systematic: first k symbols are the message
        var redundancy = [Fr](repeating: Fr.zero, count: redundancyLength)

        // Sparse matvec: for each right vertex i, accumulate contributions from its D neighbors
        for i in 0..<redundancyLength {
            var acc = Fr.zero
            let base = i * degree
            for d in 0..<degree {
                let leftIdx = Int(graph.neighbors[base + d])
                let coeff = graph.coefficients[base + d]
                acc = frAdd(acc, frMul(coeff, message[leftIdx]))
            }
            redundancy[i] = acc
        }

        codeword.append(contentsOf: redundancy)
        return codeword
    }

    /// Check if a codeword is valid (in the code's image).
    /// Recomputes redundancy from the message portion and compares.
    public func isValidCodeword(_ codeword: [Fr]) -> Bool {
        guard codeword.count == codewordLength else { return false }
        let message = Array(codeword.prefix(messageLength))
        let expected = encode(message)
        for i in messageLength..<codewordLength {
            if frToInt(codeword[i]) != frToInt(expected[i]) {
                return false
            }
        }
        return true
    }

    /// Minimum distance bound.
    /// For a random left-regular bipartite expander with degree D and expansion > D/2,
    /// the Sipser-Spielman bound gives minimum distance ≥ (1 - rate) * n * Ω(1).
    /// In practice, random expanders achieve near-optimal distance.
    public var minimumDistanceBound: Int {
        // Conservative bound: at least redundancyLength / 4
        return max(redundancyLength / 4, 1)
    }

    /// Return the neighbor indices and coefficients for a given redundancy row.
    /// Useful for GPU kernel dispatch (upload sparse structure).
    public func rowNeighbors(_ row: Int) -> (indices: [UInt32], coefficients: [Fr]) {
        let base = row * degree
        let indices = Array(graph.neighbors[base..<base + degree])
        let coeffs = Array(graph.coefficients[base..<base + degree])
        return (indices, coeffs)
    }
}

// MARK: - Legacy Dense Code (kept for comparison / backwards compatibility)

/// Simple random linear code using a dense pseudo-random matrix.
/// Encoding is O(k * (n-k)) — use ExpanderCode for O(D * (n-k)) instead.
public struct DenseLinearCode {
    public let messageLength: Int
    public let codewordLength: Int
    public let redundancyLength: Int
    public let rate: Double
    public let seed: UInt32

    public init(messageLength: Int, rateInverse: Int = 4, seed: UInt32 = 0xB4A4E) {
        precondition(messageLength > 0)
        precondition(rateInverse >= 2)
        self.messageLength = messageLength
        self.codewordLength = messageLength * rateInverse
        self.redundancyLength = codewordLength - messageLength
        self.rate = 1.0 / Double(rateInverse)
        self.seed = seed
    }

    public func encode(_ message: [Fr]) -> [Fr] {
        precondition(message.count == messageLength)
        var codeword = message
        var redundancy = [Fr](repeating: Fr.zero, count: redundancyLength)

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
}
