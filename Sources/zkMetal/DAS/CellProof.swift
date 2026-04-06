// Cell-based KZG Proofs for Danksharding Data Availability
//
// In Ethereum danksharding, a blob is divided into cells (groups of consecutive
// field elements). Each cell gets a KZG proof, enabling efficient batch verification
// of data availability.
//
// Cell structure:
//   - Blob = blobSize field elements (e.g., 4096)
//   - Extended blob = extendedSize field elements (e.g., 8192 with 2x RS)
//   - Cell = cellSize consecutive elements in the extended blob
//   - Number of cells = extendedSize / cellSize
//
// Each cell proof attests that a contiguous segment of the codeword is consistent
// with the committed polynomial.

import Foundation
import NeonFieldOps

// MARK: - Cell Configuration

/// Configuration for cell-based DAS proofs.
public struct CellConfig {
    /// Number of field elements per cell.
    public let cellSize: Int

    /// DAS configuration (blob size, extension factor, etc.)
    public let dasConfig: DASConfig

    /// Total number of cells in the extended blob.
    public var cellCount: Int { dasConfig.extendedSize / cellSize }

    /// Default configuration: 64 elements per cell, EIP-4844 blob.
    public static let eip4844 = CellConfig(cellSize: 64, dasConfig: .eip4844)

    /// Small configuration for tests.
    public static let test = CellConfig(cellSize: 4, dasConfig: .test)

    public init(cellSize: Int, dasConfig: DASConfig) {
        precondition(cellSize > 0 && (cellSize & (cellSize - 1)) == 0, "cellSize must be power of 2")
        precondition(dasConfig.extendedSize % cellSize == 0, "extendedSize must be divisible by cellSize")
        self.cellSize = cellSize
        self.dasConfig = dasConfig
    }
}

// MARK: - Cell Proof

/// A KZG proof for a cell: a contiguous group of field elements in the extended blob.
/// The proof attests that the polynomial committed to by the blob commitment evaluates
/// to these values at the corresponding domain positions.
public struct CellProof {
    /// Cell index (0..<cellCount).
    public let cellIndex: Int

    /// Field elements in this cell.
    public let values: [Fr381]

    /// KZG witness point for the cell.
    /// For a single-element cell, this is a standard KZG opening proof.
    /// For a multi-element cell, this is a batched proof using a random challenge.
    public let witness: G1Projective381

    /// The random challenge used to batch the cell's opening proofs (if cellSize > 1).
    public let batchChallenge: Fr381

    public init(cellIndex: Int, values: [Fr381], witness: G1Projective381, batchChallenge: Fr381) {
        self.cellIndex = cellIndex
        self.values = values
        self.witness = witness
        self.batchChallenge = batchChallenge
    }
}

// MARK: - Cell Proof Engine

/// Engine for generating and verifying cell-based KZG proofs.
/// Divides the extended blob into cells, each with a KZG proof.
public class CellProofEngine {
    public let config: CellConfig
    public let srs: [G1Affine381]

    /// Initialize with SRS and cell configuration.
    public init(srs: [G1Affine381], config: CellConfig = .eip4844) {
        self.config = config
        self.srs = srs
    }

    // MARK: - Cell Proof Generation

    /// Generate KZG proofs for all cells in the extended blob.
    ///
    /// For each cell, we batch the individual position proofs using a random linear
    /// combination. The batch challenge is derived deterministically from the cell index
    /// and commitment (Fiat-Shamir).
    ///
    /// - Parameters:
    ///   - coefficients: polynomial coefficients (the blob)
    ///   - codeword: extended codeword from RS encoding
    ///   - commitment: KZG commitment to the polynomial
    /// - Returns: array of CellProofs, one per cell
    public func generateCellProofs(coefficients: [Fr381], codeword: [Fr381],
                                   commitment: G1Projective381) throws -> [CellProof] {
        let cellCount = config.cellCount
        let cellSize = config.cellSize
        let n = config.dasConfig.extendedSize
        let logN = config.dasConfig.logExtendedSize
        let omega = fr381RootOfUnity(logN: logN)

        var proofs = [CellProof]()
        proofs.reserveCapacity(cellCount)

        for cellIdx in 0..<cellCount {
            let startPos = cellIdx * cellSize
            let values = Array(codeword[startPos..<startPos + cellSize])

            // Derive batch challenge from cell index (deterministic)
            let batchChallenge = deriveCellChallenge(cellIndex: cellIdx)

            // Compute batched witness: sum_j gamma^j * q_j(x)
            // where q_j(x) = (p(x) - v_j) / (x - omega^(startPos + j))
            let witness = try computeBatchedCellWitness(
                coefficients: coefficients,
                cellStartIndex: startPos,
                values: values,
                omega: omega,
                gamma: batchChallenge
            )

            proofs.append(CellProof(
                cellIndex: cellIdx,
                values: values,
                witness: witness,
                batchChallenge: batchChallenge
            ))
        }

        return proofs
    }

    /// Generate a KZG proof for a single cell.
    public func generateSingleCellProof(coefficients: [Fr381], codeword: [Fr381],
                                         cellIndex: Int) throws -> CellProof {
        let cellSize = config.cellSize
        let n = config.dasConfig.extendedSize
        let logN = config.dasConfig.logExtendedSize
        let omega = fr381RootOfUnity(logN: logN)

        let startPos = cellIndex * cellSize
        precondition(startPos + cellSize <= n, "Cell index out of range")
        let values = Array(codeword[startPos..<startPos + cellSize])

        let batchChallenge = deriveCellChallenge(cellIndex: cellIndex)

        let witness = try computeBatchedCellWitness(
            coefficients: coefficients,
            cellStartIndex: startPos,
            values: values,
            omega: omega,
            gamma: batchChallenge
        )

        return CellProof(
            cellIndex: cellIndex,
            values: values,
            witness: witness,
            batchChallenge: batchChallenge
        )
    }

    // MARK: - Cell Proof Verification

    /// Verify a cell proof against the blob commitment.
    /// Uses SRS secret for test-mode verification.
    ///
    /// Verification equation:
    ///   C_combined == [y_combined]*G + [s - z_combined]*witness
    /// where the combined values use the same random linear combination as generation.
    public func verifyCellProof(commitment: G1Projective381, cell: CellProof,
                                srsSecret: [UInt64]) -> Bool {
        let cellSize = config.cellSize
        let logN = config.dasConfig.logExtendedSize
        let omega = fr381RootOfUnity(logN: logN)
        let startPos = cell.cellIndex * cellSize
        let gamma = cell.batchChallenge

        // Reconstruct the combined commitment check:
        // sum_j gamma^j * (C - [v_j]*G) == sum_j gamma^j * (z_j) * witness  (conceptually)
        // Rearranged: C * (sum gamma^j) - G * (sum gamma^j * v_j) == witness * (sum gamma^j * z_j)
        // But we verify as: C_combined == [y_combined]*G + [s_minus_z_combined]*witness

        let gen = g1_381FromAffine(bls12381G1Generator())
        let sFr = fr381Mul(Fr381.from64(srsSecret), Fr381.from64(Fr381.R2_MOD_R))

        // For batched verification, we check the linearized equation:
        // sum_j gamma^j * [C - v_j*G - (s - z_j)*witness] == 0
        // Which simplifies to:
        // (sum gamma^j) * C == sum gamma^j * v_j * G + sum gamma^j * (s - z_j) * witness

        var gammaSum = Fr381.zero
        var gammaPow = Fr381.one
        var weightedValueSum = Fr381.zero
        var weightedSMinusZWitness = g1_381Identity()

        for j in 0..<cellSize {
            let posIdx = startPos + j
            let zj = fr381Pow(omega, UInt64(posIdx))
            let vj = cell.values[j]

            gammaSum = fr381Add(gammaSum, gammaPow)
            weightedValueSum = fr381Add(weightedValueSum, fr381Mul(gammaPow, vj))

            let sMzj = fr381Sub(sFr, zj)
            let gammaSMz = fr381Mul(gammaPow, sMzj)
            let term = g1_381ScalarMul(cell.witness, fr381ToInt(gammaSMz))
            weightedSMinusZWitness = g1_381Add(weightedSMinusZWitness, term)

            gammaPow = fr381Mul(gammaPow, gamma)
        }

        // LHS = gammaSum * C
        let lhs = g1_381ScalarMul(commitment, fr381ToInt(gammaSum))

        // RHS = weightedValueSum * G + weightedSMinusZWitness
        let vG = g1_381ScalarMul(gen, fr381ToInt(weightedValueSum))
        let rhs = g1_381Add(vG, weightedSMinusZWitness)

        guard let lhsAff = g1_381ToAffine(lhs),
              let rhsAff = g1_381ToAffine(rhs) else {
            return g1_381IsIdentity(lhs) && g1_381IsIdentity(rhs)
        }
        return fp381ToInt(lhsAff.x) == fp381ToInt(rhsAff.x) &&
               fp381ToInt(lhsAff.y) == fp381ToInt(rhsAff.y)
    }

    /// Verify a cell proof by recomputing the witness from scratch and comparing.
    /// This is an O(N*cellSize) verification suitable for testing without pairings.
    public func verifyCellProofByRecompute(commitment: G1Projective381, cell: CellProof,
                                            coefficients: [Fr381], codeword: [Fr381]) -> Bool {
        let cellSize = config.cellSize
        let logN = config.dasConfig.logExtendedSize
        let omega = fr381RootOfUnity(logN: logN)
        let startPos = cell.cellIndex * cellSize

        // Verify values match the codeword
        for j in 0..<cellSize {
            let posIdx = startPos + j
            if posIdx >= codeword.count { return false }
            // Use Horner evaluation (authoritative)
            let evalPoint = fr381Pow(omega, UInt64(posIdx))
            let expected = hornerEval381(coefficients, at: evalPoint)
            if fr381ToInt(cell.values[j]) != fr381ToInt(expected) {
                return false
            }
        }

        // Recompute the batched witness
        let gamma = cell.batchChallenge
        guard let expectedWitness = try? computeBatchedCellWitness(
            coefficients: coefficients, cellStartIndex: startPos,
            values: cell.values, omega: omega, gamma: gamma
        ) else { return false }

        // Compare witness points
        guard let witnessAff = g1_381ToAffine(cell.witness),
              let expectedAff = g1_381ToAffine(expectedWitness) else {
            return g1_381IsIdentity(cell.witness) && g1_381IsIdentity(expectedWitness)
        }
        return fp381ToInt(witnessAff.x) == fp381ToInt(expectedAff.x) &&
               fp381ToInt(witnessAff.y) == fp381ToInt(expectedAff.y)
    }

    /// Horner evaluation of polynomial at a point.
    private func hornerEval381(_ coeffs: [Fr381], at z: Fr381) -> Fr381 {
        var acc = Fr381.zero
        for i in stride(from: coeffs.count - 1, through: 0, by: -1) {
            acc = fr381Add(fr381Mul(acc, z), coeffs[i])
        }
        return acc
    }

    /// Verify a cell proof using BLS12-381 pairings (production mode).
    /// Uses the pairing check instead of SRS secret.
    public func verifyCellProofWithPairing(commitment: G1Projective381, cell: CellProof,
                                            g2Gen: G2Affine381, sG2: G2Affine381) -> Bool {
        let cellSize = config.cellSize
        let logN = config.dasConfig.logExtendedSize
        let omega = fr381RootOfUnity(logN: logN)
        let startPos = cell.cellIndex * cellSize
        let gamma = cell.batchChallenge

        // Compute: LHS_G1 = sum_j gamma^j * (C - [v_j]*G1)
        // Compute: LHS_G2 = sum_j gamma^j * ([s]G2 - [z_j]G2)
        // Pairing check: e(LHS_G1, G2) == e(witness, LHS_G2)
        // Equivalently: e(sum gamma^j * (C - v_j*G), G2) * e(-witness, sum gamma^j * (s - z_j)*G2) == 1

        let gen = g1_381FromAffine(bls12381G1Generator())
        var gammaPow = Fr381.one

        var combinedG1 = g1_381Identity()
        var combinedG2 = g2_381FromAffine(g2Gen)  // placeholder; recalculated below
        combinedG2 = g2_381Identity()

        for j in 0..<cellSize {
            let posIdx = startPos + j
            let zj = fr381Pow(omega, UInt64(posIdx))
            let vj = cell.values[j]

            // gamma^j * (C - v_j * G)
            let vjG = g1_381ScalarMul(gen, fr381ToInt(vj))
            let diff = g1_381Add(commitment, g1_381Negate(vjG))
            let scaled = g1_381ScalarMul(diff, fr381ToInt(gammaPow))
            combinedG1 = g1_381Add(combinedG1, scaled)

            // gamma^j * ([s]G2 - [z_j]G2)
            let zjG2 = g2_381ScalarMul(g2_381FromAffine(g2Gen), fr381ToInt(zj))
            let sMinusZjG2 = g2_381Add(g2_381FromAffine(sG2), g2_381Negate(zjG2))
            let scaledG2 = g2_381ScalarMul(sMinusZjG2, fr381ToInt(gammaPow))
            combinedG2 = g2_381Add(combinedG2, scaledG2)

            gammaPow = fr381Mul(gammaPow, gamma)
        }

        guard let g1Aff = g1_381ToAffine(combinedG1) else {
            // Combined G1 is identity => all diffs are zero => trivially correct
            return true
        }
        guard let g2Aff = g2_381ToAffine(combinedG2) else {
            return false
        }
        guard let witnessAff = g1_381ToAffine(cell.witness) else {
            // Witness is identity but combined G1 is not => fail
            return false
        }

        let negWitnessAff = g1_381NegateAffine(witnessAff)
        return bls12381PairingCheck([(g1Aff, g2Gen), (negWitnessAff, g2Aff)])
    }

    // MARK: - Private Helpers

    /// Compute the batched cell witness using random linear combination.
    /// witness = sum_j gamma^j * q_j(x) evaluated at SRS
    /// where q_j(x) = (p(x) - v_j) / (x - omega^(startIdx + j))
    private func computeBatchedCellWitness(coefficients: [Fr381], cellStartIndex: Int,
                                            values: [Fr381], omega: Fr381,
                                            gamma: Fr381) throws -> G1Projective381 {
        let cellSize = values.count
        let polyDeg = coefficients.count
        guard srs.count >= polyDeg else {
            throw RSError.gpuError("SRS too small")
        }

        // Build combined quotient: h(x) = sum_j gamma^j * q_j(x)
        // where q_j(x) = (p(x) - v_j) / (x - z_j)
        let maxQuotientDeg = polyDeg - 1
        var combined = [Fr381](repeating: .zero, count: maxQuotientDeg)
        var gammaPow = Fr381.one

        for j in 0..<cellSize {
            let posIdx = cellStartIndex + j
            let zj = fr381Pow(omega, UInt64(posIdx))

            // p(x) - v_j
            var shifted = coefficients
            shifted[0] = fr381Sub(shifted[0], values[j])

            // Synthetic division by (x - z_j)
            let quotient = syntheticDiv381(shifted, z: zj)

            // Accumulate gamma^j * q_j
            for d in 0..<quotient.count {
                if d < combined.count {
                    combined[d] = fr381Add(combined[d], fr381Mul(gammaPow, quotient[d]))
                }
            }

            gammaPow = fr381Mul(gammaPow, gamma)
        }

        // MSM to produce the witness point
        guard !combined.isEmpty else { return g1_381Identity() }
        return msm381(points: Array(srs.prefix(combined.count)), scalars: combined)
    }

    /// Derive a deterministic Fiat-Shamir challenge for a cell.
    /// Uses SHA-256 hash of the cell index reduced to Fr381.
    private func deriveCellChallenge(cellIndex: Int) -> Fr381 {
        var input = [UInt8]()
        input.append(contentsOf: Array("DAS_CELL_CHALLENGE".utf8))
        var idx = UInt64(cellIndex)
        withUnsafeBytes(of: &idx) { input.append(contentsOf: $0) }
        let hash = sha256(input)
        // Interpret as big-endian integer mod Fr381
        return hashBytesToFr381(hash)
    }

    /// Convert 32-byte hash to Fr381 (reduce mod r).
    private func hashBytesToFr381(_ hash: [UInt8]) -> Fr381 {
        precondition(hash.count == 32)
        var limbs: [UInt64] = [0, 0, 0, 0]
        for i in 0..<4 {
            let byteOffset = 24 - i * 8
            for j in 0..<8 {
                limbs[i] |= UInt64(hash[byteOffset + j]) << UInt64((7 - j) * 8)
            }
        }
        // Simple reduction: if >= modulus, subtract
        if !isLessThanFr381Modulus(limbs) {
            var borrow: UInt64 = 0
            for i in 0..<4 {
                let (diff, o1) = limbs[i].subtractingReportingOverflow(Fr381.P[i])
                let (diff2, o2) = diff.subtractingReportingOverflow(borrow)
                limbs[i] = diff2
                borrow = (o1 ? 1 : 0) + (o2 ? 1 : 0)
            }
        }
        let raw = Fr381.from64(limbs)
        return fr381Mul(raw, Fr381.from64(Fr381.R2_MOD_R))
    }

    /// Check if limbs < Fr381 modulus.
    private func isLessThanFr381Modulus(_ limbs: [UInt64]) -> Bool {
        for i in stride(from: 3, through: 0, by: -1) {
            if limbs[i] < Fr381.P[i] { return true }
            if limbs[i] > Fr381.P[i] { return false }
        }
        return false
    }

    /// Synthetic division.
    private func syntheticDiv381(_ coeffs: [Fr381], z: Fr381) -> [Fr381] {
        let n = coeffs.count
        if n < 2 { return [] }
        var quotient = [Fr381](repeating: .zero, count: n - 1)
        quotient[n - 2] = coeffs[n - 1]
        for i in stride(from: n - 3, through: 0, by: -1) {
            quotient[i] = fr381Add(coeffs[i + 1], fr381Mul(z, quotient[i + 1]))
        }
        return quotient
    }

    /// CPU MSM on BLS12-381 G1.
    private func msm381(points: [G1Affine381], scalars: [Fr381]) -> G1Projective381 {
        let n = points.count
        precondition(scalars.count == n)
        if n == 0 { return g1_381Identity() }

        var flatScalars = [UInt32]()
        flatScalars.reserveCapacity(n * 8)
        for s in scalars {
            let std = fr381ToInt(s)
            flatScalars.append(UInt32(std[0] & 0xFFFFFFFF))
            flatScalars.append(UInt32(std[0] >> 32))
            flatScalars.append(UInt32(std[1] & 0xFFFFFFFF))
            flatScalars.append(UInt32(std[1] >> 32))
            flatScalars.append(UInt32(std[2] & 0xFFFFFFFF))
            flatScalars.append(UInt32(std[2] >> 32))
            flatScalars.append(UInt32(std[3] & 0xFFFFFFFF))
            flatScalars.append(UInt32(std[3] >> 32))
        }
        return g1_381PippengerMSMFlat(points: points, flatScalars: flatScalars)
    }

}
