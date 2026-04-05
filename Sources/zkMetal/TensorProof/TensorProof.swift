// TensorProof — Compressed proof structure for tensor-structured sumcheck
//
// When a multilinear polynomial factors as a tensor product f(x) = f_1(x_1) ⊗ f_2(x_2) ⊗ ... ⊗ f_k(x_k),
// the sumcheck round polynomials have special structure. Instead of sending a degree-d polynomial
// per round (d+1 field elements), we send compact factor updates that reconstruct the round poly.
//
// Compression ratio: from O(n * (d+1)) field elements to O(sqrt(n) + k * d_factor) where k = number
// of tensor factors and d_factor is the degree of each factor polynomial.
//
// This is the core data structure used by TensorSumcheck (prover/verifier) and TensorCompressor
// (post-hoc compression of standard sumcheck transcripts).

import Foundation
import NeonFieldOps

// MARK: - Tensor Factor Description

/// Describes a single factor in a tensor product decomposition.
/// f(x) = f_1(x_{0..m1}) ⊗ f_2(x_{m1..m2}) ⊗ ... ⊗ f_k(x_{m_{k-1}..n})
public struct TensorFactor {
    /// Evaluations of this factor over its boolean hypercube {0,1}^m_i
    public let evaluations: [Fr]
    /// Number of variables in this factor
    public let numVars: Int

    public init(evaluations: [Fr], numVars: Int) {
        precondition(evaluations.count == (1 << numVars),
                     "Evaluations count must be 2^numVars")
        self.evaluations = evaluations
        self.numVars = numVars
    }
}

// MARK: - Compressed Round Message

/// A compressed sumcheck round message for tensor-structured polynomials.
///
/// Standard sumcheck: sends (S(0), S(1), ..., S(d)) per round -- (d+1) field elements.
/// Tensor-compressed: when the polynomial factors, each round's polynomial is determined
/// by a smaller "factor update" vector, plus the running partial products from other factors.
public struct CompressedRoundMessage {
    /// The round index (0-based)
    public let round: Int
    /// Which tensor factor this round belongs to
    public let factorIndex: Int
    /// The local round within the factor (0-based within the factor's variable block)
    public let localRound: Int
    /// Compressed data: the partial evaluation values that determine the round polynomial.
    /// For a factor with 2^m evaluations at local round j, this has 2^(m-j-1) entries
    /// representing the "half-table" after fixing variables 0..j.
    public let compressedEvals: [Fr]
    /// The round polynomial evaluated at 0, 1, ..., d (for verification compatibility)
    /// This is reconstructed from compressedEvals + factor partial products.
    public let roundPoly: [Fr]

    public init(round: Int, factorIndex: Int, localRound: Int,
                compressedEvals: [Fr], roundPoly: [Fr]) {
        self.round = round
        self.factorIndex = factorIndex
        self.localRound = localRound
        self.compressedEvals = compressedEvals
        self.roundPoly = roundPoly
    }
}

// MARK: - Tensor Sumcheck Proof (Compressed)

/// A compressed sumcheck proof that exploits tensor product structure.
///
/// Proof size comparison for n variables, k equal-sized factors:
///   Standard:    n * (d+1) field elements
///   Compressed:  k * (sqrt(N/k)) + n field elements (factor snapshots + challenges)
///   where N = 2^n
///
/// The key insight: within each factor's variable block, the round polynomials
/// are determined by the factor's evolving evaluation table (which shrinks by half each round)
/// combined with a single scalar from the other factors' partial products.
/// We only need to store the factor snapshots at boundaries + the partial product scalars.
public struct TensorSumcheckProof {
    /// The claimed sum being proved
    public let claimedSum: Fr
    /// Number of variables in the original polynomial
    public let numVars: Int
    /// Factor boundary info: (factorIndex, numVars) for each factor
    public let factorSizes: [(index: Int, numVars: Int)]
    /// Partial product scalars at factor boundaries.
    /// partialProducts[i] = product of fully-evaluated factors before factor i.
    /// partialProducts[0] = 1 (no factors evaluated yet).
    public let partialProducts: [Fr]
    /// Factor snapshots: the evaluation table of each factor at the start of its block.
    /// factorSnapshots[i] has 2^(factorSizes[i].numVars) entries.
    public let factorSnapshots: [[Fr]]
    /// Standard round polynomials for verification (computed from compressed data).
    /// rounds[i] = (S_i(0), S_i(1), S_i(2)) for degree-2 sumcheck.
    public let rounds: [(Fr, Fr, Fr)]
    /// Challenges used in each round (for transcript reconstruction)
    public let challenges: [Fr]
    /// Final evaluation after all rounds
    public let finalEval: Fr

    /// Proof size in field elements (compressed representation)
    public var compressedSize: Int {
        // Factor snapshots + partial products + challenges + final eval + claimed sum
        let snapshotSize = factorSnapshots.reduce(0) { $0 + $1.count }
        return snapshotSize + partialProducts.count + challenges.count + 2
    }

    /// Standard proof size for comparison
    public var standardSize: Int {
        return rounds.count * 3 + 1  // 3 evaluations per round + final eval
    }

    /// Compression ratio (< 1.0 means smaller)
    public var compressionRatio: Double {
        guard standardSize > 0 else { return 1.0 }
        return Double(compressedSize) / Double(standardSize)
    }
}

// MARK: - Serialization

extension TensorSumcheckProof {
    /// Serialize the compressed proof to bytes.
    /// Format: [numVars:4][numFactors:4][factorSizes...][partialProducts...][factorSnapshots...][challenges...][finalEval][claimedSum]
    public func serializeCompressed() -> [UInt8] {
        var data = [UInt8]()
        let frSize = MemoryLayout<Fr>.size

        // Header
        var nv = UInt32(numVars)
        var nf = UInt32(factorSizes.count)
        withUnsafeBytes(of: &nv) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &nf) { data.append(contentsOf: $0) }

        // Factor sizes
        for (_, m) in factorSizes {
            var mv = UInt32(m)
            withUnsafeBytes(of: &mv) { data.append(contentsOf: $0) }
        }

        // Partial products
        for pp in partialProducts {
            var v = pp
            withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
        }

        // Factor snapshots
        for snapshot in factorSnapshots {
            for e in snapshot {
                var v = e
                withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
            }
        }

        // Challenges
        for c in challenges {
            var v = c
            withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
        }

        // Final eval + claimed sum
        var fe = finalEval
        var cs = claimedSum
        withUnsafeBytes(of: &fe) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &cs) { data.append(contentsOf: $0) }

        return data
    }

    /// Deserialize from compressed bytes.
    public static func deserializeCompressed(_ data: [UInt8]) -> TensorSumcheckProof? {
        let frSize = MemoryLayout<Fr>.size
        var offset = 0

        guard data.count >= 8 else { return nil }
        let numVars = Int(data.withUnsafeBytes { $0.load(fromByteOffset: 0, as: UInt32.self) })
        let numFactors = Int(data.withUnsafeBytes { $0.load(fromByteOffset: 4, as: UInt32.self) })
        offset = 8

        // Factor sizes
        var factorSizes = [(index: Int, numVars: Int)]()
        for i in 0..<numFactors {
            guard offset + 4 <= data.count else { return nil }
            let m = Int(data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt32.self) })
            factorSizes.append((index: i, numVars: m))
            offset += 4
        }

        func readFr() -> Fr? {
            guard offset + frSize <= data.count else { return nil }
            let val = data.withUnsafeBytes { buf -> Fr in
                buf.loadUnaligned(fromByteOffset: offset, as: Fr.self)
            }
            offset += frSize
            return val
        }

        // Partial products
        var partialProducts = [Fr]()
        for _ in 0..<numFactors {
            guard let v = readFr() else { return nil }
            partialProducts.append(v)
        }

        // Factor snapshots
        var factorSnapshots = [[Fr]]()
        for (_, m) in factorSizes {
            let count = 1 << m
            var snapshot = [Fr]()
            for _ in 0..<count {
                guard let v = readFr() else { return nil }
                snapshot.append(v)
            }
            factorSnapshots.append(snapshot)
        }

        // Challenges
        var challenges = [Fr]()
        for _ in 0..<numVars {
            guard let v = readFr() else { return nil }
            challenges.append(v)
        }

        // Final eval + claimed sum
        guard let finalEval = readFr(), let claimedSum = readFr() else { return nil }

        // Rounds are not stored in compressed format; caller must reconstruct if needed
        return TensorSumcheckProof(
            claimedSum: claimedSum,
            numVars: numVars,
            factorSizes: factorSizes,
            partialProducts: partialProducts,
            factorSnapshots: factorSnapshots,
            rounds: [],  // Must be reconstructed
            challenges: challenges,
            finalEval: finalEval
        )
    }
}
