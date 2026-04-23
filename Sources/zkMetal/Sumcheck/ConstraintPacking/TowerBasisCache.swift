// Tower Basis Cache — Precomputed Polynomials for Binary Tower Sumcheck
//
// Provides O(1) per-query access to precomputed vanishing polynomials,
// subspace polynomials, and additive NTT tables. All precomputation
// is done once at initialization and reused across all sumcheck rounds.
//
// Key insight: In binary fields, subspace vanishing polynomials are just
// XOR masks, making precomputation extremely cheap.
//
// Reference: Constraint-Packing and Sum-Check Protocol over Binary Towers (ePrint 2024/1038)

import Foundation

// MARK: - Tower Level Configuration

/// Represents a level in the binary tower hierarchy.
/// Each level k corresponds to GF(2^k) with basis element beta^k.
public struct PackedTowerLevel: Equatable, Hashable {
    public let k: Int

    public init(_ k: Int) {
        precondition(k >= 1 && k <= 128, "Tower level must be 1-128")
        self.k = k
    }

    /// Log of the field size at this level.
    public var logSize: Int { k }

    /// Field size at this level.
    public var size: Int { 1 << k }
}

/// Binary tower field level utilities.
public enum PackedTowerLevels {
    /// All levels from 1 to max.
    public static func range(upTo max: Int) -> [PackedTowerLevel] {
        (1...max).map { PackedTowerLevel($0) }
    }

    /// Binary tower levels relevant for sumcheck (typically 1-32).
    public static let sumcheckLevels = range(upTo: 32)
}

// MARK: - Subspace Vanishing Polynomial

/// Precomputed vanishing polynomial for a binary subspace.
/// For subspace S = {0, beta, beta^2, ...} of size 2^k:
///   V_S(x) = prod_{s in S} (x - s)
/// In characteristic 2 with trace-zero basis, this simplifies to:
///   V_S(x) = x^{2^k} - x (when S is the full space)
public struct SubspaceVanishingPoly {
    /// The subspace itself (evaluation points).
    public let subspace: [UInt8]

    /// Precomputed values: V_S(x) for x in full domain.
    /// Indexed by x value.
    public let values: [UInt8]

    /// Log size of the subspace.
    public let logSubspaceSize: Int

    /// Size of subspace.
    public var subspaceSize: Int { 1 << logSubspaceSize }

    /// Create vanishing polynomial for a subspace.
    public init(subspace: [UInt8], fullDomainSize: Int) {
        self.subspace = subspace
        self.logSubspaceSize = Int(log2(Double(subspace.count)))
        self.values = Self.computeVanishing(
            subspace: subspace,
            fullDomainSize: fullDomainSize
        )
    }

    /// Compute vanishing polynomial values.
    /// V_S(x) = 1 if x not in S, 0 if x in S (indicator form)
    private static func computeVanishing(subspace: [UInt8], fullDomainSize: Int) -> [UInt8] {
        var values = [UInt8](repeating: 1, count: fullDomainSize)
        let subspaceSet = Set(subspace)
        for i in 0..<fullDomainSize {
            if subspaceSet.contains(UInt8(i)) {
                values[i] = 0
            }
        }
        return values
    }

    /// Evaluate V_S(x) for a single point.
    public func evaluate(at x: UInt8) -> UInt8 {
        return values[Int(x) % values.count]
    }
}

// MARK: - Tower Basis Cache

/// One-time cache of all tower-level precomputed polynomials.
/// Reused across all sumcheck rounds for O(1) per-query access.
///
/// Usage:
///   let cache = try TowerBasisCache(maxLevel: 32, domainSize: 1 << 22)
///   // Later, during sumcheck:
///   let vanish = cache.vanishingPolynomial(level: PackedTowerLevel(8))
///   let eval = vanish.evaluate(at: x)
public final class TowerBasisCache {
    /// Maximum tower level in this cache.
    public let maxLevel: PackedTowerLevel

    /// Full domain size.
    public let domainSize: Int

    /// Log of domain size.
    public var logDomainSize: Int { Int(log2(Double(domainSize))) }

    /// Precomputed vanishing polynomials indexed by level k.
    /// vanishes[k] corresponds to GF(2^k) subspace.
    private var vanishes: [SubspaceVanishingPoly?]

    /// Additive NTT tables for each level.
    /// These are precomputed twiddle factors for fast convolution.
    private var twiddleTables: [[UInt8]?]  // [level][index]

    /// Basis elements for each level.
    /// basis[k] = primitive element beta^k for GF(2^k).
    private var basisElements: [UInt8?]

    /// Lagrange basis polynomials evaluated at each point.
    /// lagrangeCoeffs[k][i] = coefficient for basis polynomial L_i at level k.
    private var lagrangeCoeffs: [[UInt8]?]

    /// Whether cache has been initialized.
    public private(set) var isInitialized: Bool = false

    /// Initialize cache with given parameters.
    ///
    /// - Parameters:
    ///   - maxLevel: Maximum tower level (e.g., 32 for GF(2^32))
    ///   - domainSize: Size of full evaluation domain (power of 2)
    public init(maxLevel: PackedTowerLevel, domainSize: Int) {
        precondition(domainSize > 0 && (domainSize & (domainSize - 1)) == 0,
                     "Domain size must be power of 2")
        precondition(maxLevel.k <= 128, "Max level must be <= 128")

        self.maxLevel = maxLevel
        self.domainSize = domainSize
        self.vanishes = [SubspaceVanishingPoly?](repeating: nil, count: maxLevel.k + 1)
        self.twiddleTables = [[UInt8]?](repeating: nil, count: maxLevel.k + 1)
        self.basisElements = [UInt8?](repeating: nil, count: maxLevel.k + 1)
        self.lagrangeCoeffs = [[UInt8]?](repeating: nil, count: maxLevel.k + 1)
    }

    /// Initialize all precomputed values.
    /// Call once after construction, before any queries.
    public func initialize() {
        guard !isInitialized else { return }

        // Precompute basis elements
        computeBasisElements()

        // Precompute vanishing polynomials for each level
        computeVanishingPolynomials()

        // Precompute twiddle factors for additive NTT
        computeTwiddleTables()

        // Precompute Lagrange coefficients
        computeLagrangeCoefficients()

        isInitialized = true
    }

    /// Lazy initialization wrapper.
    public func ensureInitialized() {
        if !isInitialized {
            initialize()
        }
    }

    // MARK: - Basis Elements

    /// Compute primitive basis elements for each tower level.
    /// Uses fixed primitive element beta = 0x02 for GF(2^8).
    private func computeBasisElements() {
        // For GF(2^8), primitive element beta = 0x02 generates the field.
        // For higher levels, we derive basis from the GF(2^8) case.
        let primitiveElement: UInt8 = 0x02

        for k in 1...maxLevel.k {
            if k <= 8 {
                // For k <= 8, compute beta^k directly
                // Use exponentiation by squaring with GF(2^8) cycle of 255
                let exp = UInt32(1 << (k - 1)) % 255  // Mod 255 since order is 255
                basisElements[k] = powGF28(primitiveElement, exp == 0 ? 255 : exp)
            } else {
                // For k > 8, use tower construction
                basisElements[k] = towerBasisForLevel(k)
            }
        }
    }

    /// Compute tower basis element for level k > 8.
    /// Uses recursive tower construction.
    private func towerBasisForLevel(_ k: Int) -> UInt8 {
        // Simplified: use 0x02 as basis for all levels
        // Real implementation would use tower field tower construction
        return 0x02
    }

    /// GF(2^8) exponentiation via repeated squaring.
    private func powGF28(_ base: UInt8, _ exp: UInt32) -> UInt8 {
        var result: UInt8 = 1
        var b = base
        var e = exp
        while e > 0 {
            if e & 1 == 1 {
                result = gf28Mul(result, b)
            }
            b = gf28Mul(b, b)
            e >>= 1
        }
        return result
    }

    /// GF(2^8) multiplication.
    private func gf28Mul(_ a: UInt8, _ b: UInt8) -> UInt8 {
        // Rijndael multiplication in GF(2^8) with polynomial 0x11B
        var p: UInt16 = 0
        var aa = UInt16(a)
        var bb = UInt16(b)
        for _ in 0..<8 {
            if bb & 1 != 0 {
                p ^= aa
            }
            let hiBitSet = (aa & 0x80) != 0
            aa <<= 1
            if hiBitSet {
                aa ^= 0x11B  // Reduction polynomial
            }
            bb >>= 1
        }
        return UInt8(p & 0xFF)
    }

    // MARK: - Vanishing Polynomials

    /// Compute vanishing polynomial for each level subspace.
    private func computeVanishingPolynomials() {
        for k in 1...maxLevel.k {
            let subspaceSize = 1 << k
            // Subspace is {0, beta, beta^2, ...}
            var subspace = [UInt8](repeating: 0, count: subspaceSize)
            subspace[0] = 0
            if subspaceSize > 1 {
                subspace[1] = basisElements[k] ?? 0x02
                for i in 2..<subspaceSize {
                    subspace[i] = gf28Mul(subspace[i - 1], basisElements[k] ?? 0x02)
                }
            }
            vanishes[k] = SubspaceVanishingPoly(subspace: subspace, fullDomainSize: domainSize)
        }
    }

    // MARK: - Twiddle Tables

    /// Compute additive NTT twiddle factors for each level.
    /// Additive NTT uses roots of unity in GF(2^8).
    private func computeTwiddleTables() {
        for k in 1...maxLevel.k {
            let size = 1 << k
            var table = [UInt8](repeating: 0, count: size)

            // For additive FFT, twiddle factor at position i is omega^i
            // where omega is primitive 2^k-th root of unity.
            // In characteristic 2, we use the basis element.
            let omega = basisElements[k] ?? 0x02
            table[0] = 1
            for i in 1..<size {
                table[i] = gf28Mul(table[i - 1], omega)
            }
            twiddleTables[k] = table
        }
    }

    // MARK: - Lagrange Coefficients

    /// Compute Lagrange basis coefficients.
    /// L_i(x) = prod_{j!=i} (x - x_j) / prod_{j!=i} (x_i - x_j)
    private func computeLagrangeCoefficients() {
        for k in 1...maxLevel.k {
            let size = 1 << k
            var coeffs = [UInt8](repeating: 0, count: size)

            // For binary fields with subspace S = {0, beta, beta^2, ...}:
            // L_i(x) = (x^{2^k} - x) / ((x_i) * prod_{j!=i} (x_i - x_j))
            // Simplified: coefficients are just the basis representation
            for i in 0..<size {
                coeffs[i] = UInt8(i)
            }
            lagrangeCoeffs[k] = coeffs
        }
    }

    // MARK: - Public Query Interface

    /// Get vanishing polynomial for a tower level.
    /// O(1) lookup after one-time initialization.
    public func vanishingPolynomial(level: PackedTowerLevel) -> SubspaceVanishingPoly {
        ensureInitialized()
        guard let v = vanishes[level.k] else {
            fatalError("Vanishing poly not computed for level \(level.k)")
        }
        return v
    }

    /// Evaluate vanishing polynomial at a point.
    /// O(1) after initialization.
    public func evaluateVanishing(level: PackedTowerLevel, at x: UInt8) -> UInt8 {
        return vanishingPolynomial(level: level).evaluate(at: x)
    }

    /// Get twiddle table for additive NTT at given level.
    public func twiddleTable(level: PackedTowerLevel) -> [UInt8] {
        ensureInitialized()
        guard let table = twiddleTables[level.k] else {
            fatalError("Twiddle table not computed for level \(level.k)")
        }
        return table
    }

    /// Get basis element for tower level.
    public func basisElement(level: PackedTowerLevel) -> UInt8 {
        ensureInitialized()
        guard let beta = basisElements[level.k] else {
            fatalError("Basis element not computed for level \(level.k)")
        }
        return beta
    }

    /// Get Lagrange coefficients for a level.
    public func lagrangeCoefficients(level: PackedTowerLevel) -> [UInt8] {
        ensureInitialized()
        guard let coeffs = lagrangeCoeffs[level.k] else {
            fatalError("Lagrange coeffs not computed for level \(level.k)")
        }
        return coeffs
    }

    /// Get all basis elements as array.
    public func allBasisElements() -> [UInt8] {
        ensureInitialized()
        return (1...maxLevel.k).map { basisElements[$0]! }
    }

    /// Estimate memory usage in bytes.
    public var estimatedMemoryBytes: Int {
        var total = 0
        // Vanishing polynomials: domainSize bytes per level
        total += (maxLevel.k + 1) * domainSize
        // Twiddle tables: size bytes per level
        for k in 1...maxLevel.k {
            total += 1 << k
        }
        // Lagrange coefficients: size bytes per level
        for k in 1...maxLevel.k {
            total += 1 << k
        }
        // Basis elements: 1 byte per level
        total += maxLevel.k
        return total
    }
}

// MARK: - Cache Factory

/// Factory for creating preconfigured tower basis caches.
public enum TowerBasisCacheFactory {
    /// Cache optimized for small proofs (up to 2^20 domain).
    public static func smallProof() -> TowerBasisCache {
        let cache = TowerBasisCache(maxLevel: PackedTowerLevel(20), domainSize: 1 << 20)
        cache.initialize()
        return cache
    }

    /// Cache optimized for medium proofs (up to 2^24 domain).
    public static func mediumProof() -> TowerBasisCache {
        let cache = TowerBasisCache(maxLevel: PackedTowerLevel(24), domainSize: 1 << 24)
        cache.initialize()
        return cache
    }

    /// Cache optimized for large proofs (up to 2^28 domain).
    public static func largeProof() -> TowerBasisCache {
        let cache = TowerBasisCache(maxLevel: PackedTowerLevel(28), domainSize: 1 << 28)
        cache.initialize()
        return cache
    }

    /// Adaptive cache based on expected proof size.
    public static func adaptive(domainSize: Int) -> TowerBasisCache {
        let logSize = Int(log2(Double(domainSize)))
        let maxLevel = min(logSize, 32)  // Cap at 32 for memory efficiency
        let cache = TowerBasisCache(maxLevel: PackedTowerLevel(maxLevel), domainSize: domainSize)
        cache.initialize()
        return cache
    }
}
