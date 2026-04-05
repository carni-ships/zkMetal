// Pedersen Hash / Vector Commitment Engine over BN254 G1
//
// Provides BGMW-style windowed fixed-base multi-scalar multiplication for
// efficient Pedersen commitments: C = sum(v_i * G_i) + r * H
//
// Features:
//   - Deterministic generator derivation via hash-to-curve (SHA-256 + SWU-like try-and-increment)
//   - BGMW windowed precomputation tables for fixed generators
//   - Cached generator tables for common sizes (256, 1024, 4096)
//   - Homomorphic operations on commitments
//   - Batch commitment support
//
// Used as foundation for Bulletproofs, IPA, and Verkle trees.

import Foundation
#if canImport(CryptoKit)
import CryptoKit
#endif

// MARK: - BGMW Precomputed Generator Table

/// Precomputed table for a single generator point using BGMW windowed method.
/// For window size w, stores [0*G, 1*G, 2*G, ..., (2^w - 1)*G] for each window position.
public struct BGMWTable {
    /// Window size in bits
    public let windowBits: Int
    /// Number of windows needed for 256-bit scalars
    public let numWindows: Int
    /// tables[windowIdx][digit] = digit * (2^(windowIdx * windowBits)) * G
    public let tables: [[PointProjective]]
}

/// Build a BGMW precomputed table for a single generator point.
///
/// - Parameters:
///   - generator: the base generator point (affine)
///   - windowBits: window size (default 4 for 16-entry tables)
/// - Returns: precomputed BGMW table
public func bgmwBuildTable(generator: PointAffine, windowBits: Int = 4) -> BGMWTable {
    let numWindows = (256 + windowBits - 1) / windowBits
    let tableSize = 1 << windowBits  // 2^w entries per window

    var tables = [[PointProjective]]()
    tables.reserveCapacity(numWindows)

    // base = generator in projective
    var base = pointFromAffine(generator)

    for _ in 0..<numWindows {
        var table = [PointProjective]()
        table.reserveCapacity(tableSize)
        table.append(pointIdentity())  // 0 * base
        var acc = base
        table.append(acc)  // 1 * base
        for _ in 2..<tableSize {
            acc = pointAdd(acc, base)
            table.append(acc)
        }
        tables.append(table)

        // Shift base by 2^windowBits for next window
        for _ in 0..<windowBits {
            base = pointDouble(base)
        }
    }

    return BGMWTable(windowBits: windowBits, numWindows: numWindows, tables: tables)
}

/// Fixed-base scalar multiplication using a precomputed BGMW table.
///
/// - Parameters:
///   - table: precomputed BGMW table for the generator
///   - scalar: Fr scalar to multiply by
/// - Returns: scalar * G
public func bgmwScalarMul(table: BGMWTable, scalar: Fr) -> PointProjective {
    let limbs = frToInt(scalar)
    let w = table.windowBits
    let mask = (1 << w) - 1

    var result = pointIdentity()
    for windowIdx in 0..<table.numWindows {
        let bitOffset = windowIdx * w
        let limbIdx = bitOffset / 64
        let bitInLimb = bitOffset % 64

        var digit: Int
        if limbIdx < 4 {
            digit = Int((limbs[limbIdx] >> bitInLimb) & UInt64(mask))
            // Handle window crossing limb boundary
            if bitInLimb + w > 64 && limbIdx + 1 < 4 {
                let overflow = bitInLimb + w - 64
                digit |= Int(limbs[limbIdx + 1] & ((1 << overflow) - 1)) << (64 - bitInLimb)
                digit &= mask
            }
        } else {
            digit = 0
        }

        if digit != 0 {
            result = pointAdd(result, table.tables[windowIdx][digit])
        }
    }
    return result
}

// MARK: - Pedersen Hash Engine (BN254)

/// BN254 G1 Pedersen hash / vector commitment engine with BGMW precomputation.
///
/// Usage:
///   let engine = PedersenHashEngine(size: 256)
///   let commitment = engine.commit(values: frVector, blinding: r)
///   let valid = engine.open(values: frVector, blinding: r, commitment: commitment)
public class PedersenHashEngine {
    /// Number of generators
    public let size: Int
    /// Generator points G_0, ..., G_{n-1} (affine)
    public let generators: [PointAffine]
    /// Blinding generator H (affine)
    public let blindingGenerator: PointAffine
    /// BGMW precomputed tables for each generator
    public let generatorTables: [BGMWTable]
    /// BGMW table for blinding generator
    public let blindingTable: BGMWTable
    /// Window size used for BGMW tables
    public let windowBits: Int

    /// Initialize with a given number of generators.
    ///
    /// Generators are derived deterministically via hash-to-curve from
    /// the seed "BN254_PedersenHash_{index}". The blinding generator
    /// uses seed "BN254_PedersenHash_blinding".
    ///
    /// - Parameters:
    ///   - size: number of value generators
    ///   - windowBits: BGMW window size (default 4)
    public init(size: Int, windowBits: Int = 4) {
        precondition(size > 0, "Size must be positive")
        self.size = size
        self.windowBits = windowBits

        // Derive generators deterministically
        var gens = [PointAffine]()
        gens.reserveCapacity(size)
        for i in 0..<size {
            gens.append(PedersenHashEngine.deriveGenerator(index: i))
        }
        self.generators = gens

        // Derive blinding generator
        self.blindingGenerator = PedersenHashEngine.deriveBlindingGenerator()

        // Build BGMW tables
        var tables = [BGMWTable]()
        tables.reserveCapacity(size)
        for g in gens {
            tables.append(bgmwBuildTable(generator: g, windowBits: windowBits))
        }
        self.generatorTables = tables
        self.blindingTable = bgmwBuildTable(generator: blindingGenerator, windowBits: windowBits)
    }

    // MARK: - Commit

    /// Compute Pedersen vector commitment: C = sum(v_i * G_i) + r * H
    ///
    /// Uses BGMW fixed-base scalar multiplication for each generator,
    /// which is significantly faster than variable-base MSM for fixed generators.
    ///
    /// - Parameters:
    ///   - values: vector of Fr elements to commit to
    ///   - blinding: optional blinding factor r (nil = non-hiding, zero blinding)
    /// - Returns: commitment point in projective coordinates
    public func commit(values: [Fr], blinding: Fr? = nil) -> PointProjective {
        precondition(!values.isEmpty, "Values must not be empty")
        precondition(values.count <= size,
                     "Too many values (\(values.count)) for engine of size \(size)")

        var result = pointIdentity()

        // Sum v_i * G_i using BGMW tables
        for i in 0..<values.count {
            let limbs = frToInt(values[i])
            if limbs == [0, 0, 0, 0] { continue }
            let term = bgmwScalarMul(table: generatorTables[i], scalar: values[i])
            result = pointAdd(result, term)
        }

        // Add blinding: r * H
        if let r = blinding {
            let rLimbs = frToInt(r)
            if rLimbs != [0, 0, 0, 0] {
                let blindTerm = bgmwScalarMul(table: blindingTable, scalar: r)
                result = pointAdd(result, blindTerm)
            }
        }

        return result
    }

    // MARK: - Batch Commit

    /// Batch commit multiple vectors.
    ///
    /// - Parameters:
    ///   - vectors: array of Fr vectors to commit
    ///   - blindings: optional array of blinding factors (nil entries = zero blinding)
    /// - Returns: array of commitment points
    public func batchCommit(vectors: [[Fr]], blindings: [Fr?]? = nil) -> [PointProjective] {
        if vectors.isEmpty { return [] }

        return vectors.enumerated().map { (idx, vec) in
            let r = blindings?[safe: idx] ?? nil
            return commit(values: vec, blinding: r)
        }
    }

    // MARK: - Open (Verify)

    /// Verify a commitment opening by recomputing and comparing.
    ///
    /// Checks: C == sum(v_i * G_i) + r * H
    ///
    /// - Parameters:
    ///   - values: the claimed vector values
    ///   - blinding: the claimed blinding factor r
    ///   - commitment: the commitment to verify
    /// - Returns: true if the commitment matches
    public func open(values: [Fr], blinding: Fr, commitment: PointProjective) -> Bool {
        let recomputed = commit(values: values, blinding: blinding)
        return pointEqual(commitment, recomputed)
    }

    // MARK: - Homomorphic Operations

    /// Additive homomorphism: C1 + C2.
    /// If C1 = Commit(a, r1) and C2 = Commit(b, r2),
    /// then add(C1, C2) = Commit(a+b, r1+r2).
    public static func add(_ c1: PointProjective, _ c2: PointProjective) -> PointProjective {
        return pointAdd(c1, c2)
    }

    /// Scalar multiplication of a commitment: s * C.
    /// If C = Commit(a, r), then scalarMul(s, C) = Commit(s*a, s*r).
    public static func scalarMul(_ s: Fr, _ c: PointProjective) -> PointProjective {
        return cPointScalarMul(c, s)
    }

    // MARK: - Generator Derivation

    /// Derive a BN254 G1 generator deterministically using try-and-increment hash-to-curve.
    ///
    /// Seed: "BN254_PedersenHash_{index}"
    /// Method: hash seed+counter to get x-coordinate candidate, check if x^3+3 is a QR,
    /// if so compute y and return the point (guaranteed on curve and in prime-order subgroup
    /// since BN254 G1 has cofactor 1).
    public static func deriveGenerator(index: Int) -> PointAffine {
        return hashToCurveBN254(seed: "BN254_PedersenHash_\(index)")
    }

    /// Derive the blinding generator H.
    public static func deriveBlindingGenerator() -> PointAffine {
        return hashToCurveBN254(seed: "BN254_PedersenHash_blinding")
    }

    /// Try-and-increment hash-to-curve for BN254 G1.
    ///
    /// Hashes seed || counter to get an Fp candidate x, checks if x^3 + 3 is a
    /// quadratic residue, and if so returns (x, sqrt(x^3+3)).
    /// BN254 G1 has cofactor 1, so any point on the curve is in the subgroup.
    private static func hashToCurveBN254(seed: String) -> PointAffine {
        var counter: UInt32 = 0

        while true {
            var input = Array(seed.utf8)
            input.append(contentsOf: withUnsafeBytes(of: counter) { Array($0) })

            let hash = pedersenSHA256(input)

            // Interpret hash as Fp element
            var limbs: [UInt64] = [0, 0, 0, 0]
            for i in 0..<4 {
                for j in 0..<8 {
                    let byteIdx = i * 8 + j
                    if byteIdx < hash.count {
                        limbs[i] |= UInt64(hash[byteIdx]) << (j * 8)
                    }
                }
            }

            // Reduce mod p and convert to Montgomery form
            let raw = Fp.from64(limbs)
            let xCandidate = fpMul(raw, Fp.from64(Fp.R2_MOD_P))

            // BN254: y^2 = x^3 + 3
            let x2 = fpSqr(xCandidate)
            let x3 = fpMul(x2, xCandidate)
            let b = fpFromInt(3)
            let rhs = fpAdd(x3, b)

            // Try to compute sqrt
            if let y = fpSqrt(rhs) {
                return PointAffine(x: xCandidate, y: y)
            }

            counter += 1
        }
    }
}

// MARK: - Cached Engine Factory

/// Cache of PedersenHashEngine instances for common sizes.
/// Thread-safe via serial access (not concurrent-safe; use external locking if needed).
private var _cachedEngines: [Int: PedersenHashEngine] = [:]

/// Get or create a PedersenHashEngine for the given size.
/// Caches engines for sizes 256, 1024, and 4096.
public func getPedersenHashEngine(size: Int) -> PedersenHashEngine {
    if let cached = _cachedEngines[size] {
        return cached
    }
    let engine = PedersenHashEngine(size: size)
    // Cache common sizes
    if size == 256 || size == 1024 || size == 4096 {
        _cachedEngines[size] = engine
    }
    return engine
}

// MARK: - SHA-256 helper (avoids name collision with existing sha256Bytes)

private func pedersenSHA256(_ data: [UInt8]) -> [UInt8] {
    #if canImport(CryptoKit)
    let digest = SHA256.hash(data: data)
    return Array(digest)
    #else
    fatalError("CryptoKit not available")
    #endif
}

// MARK: - Collection safe subscript

private extension Collection {
    subscript(safe index: Index) -> Element? {
        return indices.contains(index) ? self[index] : nil
    }
}
