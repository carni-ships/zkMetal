// Constraint Packer — Pack Multiple Constraints into Shared Multilinear Polynomials
//
// Packs multiple constraints into shared multilinear polynomials using the tower's
// graded structure. Each tower level k contributes a separate "slice" of the
// multilinear extension, enabling efficient batch verification.
//
// Key insight: Instead of one polynomial per constraint, we interleave constraints
// into the same polynomial using tower level as an additional dimension.
//
// Reference: Constraint-Packing and Sum-Check Protocol over Binary Towers (ePrint 2024/1038)

import Foundation

// MARK: - Constraint Types

/// A single arithmetic constraint in R1CS form.
/// Constraints are of the form: (A_i * z) · (B_i * z) = C_i * z
public struct PackedR1CSConstraint: Equatable {
    /// A matrix row (sparse representation as (index, value) pairs).
    public let a: [(Int, UInt8)]

    /// B matrix row (sparse representation as (index, value) pairs).
    public let b: [(Int, UInt8)]

    /// C matrix row (sparse representation as (index, value) pairs).
    public let c: [(Int, UInt8)]

    public init(a: [(Int, UInt8)], b: [(Int, UInt8)], c: [(Int, UInt8)]) {
        self.a = a
        self.b = b
        self.c = c
    }

    public static func == (lhs: PackedR1CSConstraint, rhs: PackedR1CSConstraint) -> Bool {
        guard lhs.a.count == rhs.a.count,
              lhs.b.count == rhs.b.count,
              lhs.c.count == rhs.c.count else { return false }
        for i in 0..<lhs.a.count {
            if lhs.a[i].0 != rhs.a[i].0 || lhs.a[i].1 != rhs.a[i].1 { return false }
        }
        for i in 0..<lhs.b.count {
            if lhs.b[i].0 != rhs.b[i].0 || lhs.b[i].1 != rhs.b[i].1 { return false }
        }
        for i in 0..<lhs.c.count {
            if lhs.c[i].0 != rhs.c[i].0 || lhs.c[i].1 != rhs.c[i].1 { return false }
        }
        return true
    }

    /// Create a simple linear constraint: a * x = b.
    public static func linear(a: [(Int, UInt8)], b: [(Int, UInt8)]) -> PackedR1CSConstraint {
        return PackedR1CSConstraint(a: a, b: [(0, 1)], c: b)
    }

    /// Create a multiplication constraint: a * b = c.
    public static func multiplication(a: [(Int, UInt8)], b: [(Int, UInt8)], c: [(Int, UInt8)]) -> PackedR1CSConstraint {
        return PackedR1CSConstraint(a: a, b: b, c: c)
    }
}

/// A packed constraint that combines multiple R1CS constraints.
public struct AmortizedPackedConstraint {
    /// Original constraints that were packed.
    public let originalConstraints: [PackedR1CSConstraint]

    /// Tower level for this constraint's evaluation.
    public let towerLevel: PackedTowerLevel

    /// Evaluated coefficients at each variable position.
    /// Length equals number of variables at this tower level.
    public let aCoeffs: [UInt8]
    public let bCoeffs: [UInt8]
    public let cCoeffs: [UInt8]

    /// Number of constraints packed together.
    public var packingFactor: Int { originalConstraints.count }

    public init(
        originalConstraints: [PackedR1CSConstraint],
        towerLevel: PackedTowerLevel,
        aCoeffs: [UInt8],
        bCoeffs: [UInt8],
        cCoeffs: [UInt8]
    ) {
        self.originalConstraints = originalConstraints
        self.towerLevel = towerLevel
        self.aCoeffs = aCoeffs
        self.bCoeffs = bCoeffs
        self.cCoeffs = cCoeffs
    }
}

// MARK: - Packing Strategy

/// Strategy for how constraints are packed into tower levels.
public enum PackingStrategy: Equatable {
    /// One constraint per tower level (maximum parallelism).
    case onePerLevel

    /// Pack as many constraints as fit in one tower level.
    case maximizeDensity

    /// Hybrid: use both onePerLevel and maximizeDensity.
    case adaptive(maxConstraintsPerLevel: Int)

    /// Custom packing with specified batch sizes per level.
    case custom([Int])  // [numConstraints for level 1, level 2, ...]
}

/// Configuration for constraint packing.
public struct PackingConfig: Equatable {
    public let strategy: PackingStrategy
    public let maxPackedTowerLevel: PackedTowerLevel
    public let enableConstraintReuse: Bool
    public let enableSliceOptimization: Bool

    public init(
        strategy: PackingStrategy = .maximizeDensity,
        maxPackedTowerLevel: PackedTowerLevel = PackedTowerLevel(16),
        enableConstraintReuse: Bool = true,
        enableSliceOptimization: Bool = true
    ) {
        self.strategy = strategy
        self.maxPackedTowerLevel = maxPackedTowerLevel
        self.enableConstraintReuse = enableConstraintReuse
        self.enableSliceOptimization = enableSliceOptimization
    }

    /// Default configuration for sumcheck.
    public static let sumcheckDefault = PackingConfig(
        strategy: .maximizeDensity,
        maxPackedTowerLevel: PackedTowerLevel(16),
        enableConstraintReuse: true,
        enableSliceOptimization: true
    )
}

// MARK: - Constraint Packer

/// Packs multiple R1CS constraints into shared multilinear polynomials.
public final class ConstraintPacker {
    /// Configuration for packing.
    public let config: PackingConfig

    /// Tower basis cache for fast evaluation.
    public let basisCache: TowerBasisCache

    /// All packed constraints organized by tower level.
    /// packedByLevel[k] contains constraints at tower level k.
    private var packedByLevel: [[AmortizedPackedConstraint]]

    /// Total number of original constraints.
    public private(set) var totalConstraints: Int = 0

    /// Packing efficiency: original constraints / packed polynomials.
    public var packingEfficiency: Double {
        guard totalConstraints > 0 else { return 0 }
        let packedCount = packedByLevel.reduce(0) { $0 + $1.count }
        return Double(totalConstraints) / Double(max(1, packedCount))
    }

    public init(config: PackingConfig, basisCache: TowerBasisCache) {
        self.config = config
        self.basisCache = basisCache
        self.packedByLevel = [[AmortizedPackedConstraint]](repeating: [], count: config.maxPackedTowerLevel.k + 1)
    }

    // MARK: - Packing

    /// Pack a batch of constraints using the configured strategy.
    ///
    /// - Parameters:
    ///   - constraints: R1CS constraints to pack
    ///   - variableCount: Number of variables in the circuit
    /// - Returns: Number of packed constraints
    @discardableResult
    public func pack(constraints: [PackedR1CSConstraint], variableCount: Int) -> Int {
        totalConstraints += constraints.count

        switch config.strategy {
        case .onePerLevel:
            return packOnePerLevel(constraints, variableCount: variableCount)

        case .maximizeDensity:
            return packMaximizingDensity(constraints, variableCount: variableCount)

        case .adaptive(let maxPerLevel):
            return packAdaptive(constraints, variableCount: variableCount, maxPerLevel: maxPerLevel)

        case .custom(let batchSizes):
            return packCustom(constraints, variableCount: variableCount, batchSizes: batchSizes)
        }
    }

    /// Pack one constraint per tower level.
    private func packOnePerLevel(_ constraints: [PackedR1CSConstraint], variableCount: Int) -> Int {
        var packed = 0
        for (i, constraint) in constraints.enumerated() {
            let level = PackedTowerLevel(min(i + 1, config.maxPackedTowerLevel.k))
            let packedConstraint = createPackedConstraint(
                [constraint],
                level: level,
                variableCount: variableCount
            )
            packedByLevel[level.k].append(packedConstraint)
            packed += 1
        }
        return packed
    }

    /// Pack constraints to maximize density at each level.
    private func packMaximizingDensity(_ constraints: [PackedR1CSConstraint], variableCount: Int) -> Int {
        // Determine how many constraints fit per level
        let constraintsPerLevel = variableCount  // One coefficient per variable position

        var remaining = constraints
        var packed = 0

        for levelK in 1...config.maxPackedTowerLevel.k {
            if remaining.isEmpty { break }

            let batchSize = min(constraintsPerLevel, remaining.count)
            let batch = Array(remaining.prefix(batchSize))
            remaining = Array(remaining.dropFirst(batchSize))

            let packedConstraint = createPackedConstraint(
                batch,
                level: PackedTowerLevel(levelK),
                variableCount: variableCount
            )
            packedByLevel[levelK].append(packedConstraint)
            packed += batch.count
        }

        return packed
    }

    /// Pack with adaptive batch sizes.
    private func packAdaptive(_ constraints: [PackedR1CSConstraint], variableCount: Int, maxPerLevel: Int) -> Int {
        var remaining = constraints
        var packed = 0

        for levelK in 1...config.maxPackedTowerLevel.k {
            if remaining.isEmpty { break }

            let batchSize = min(maxPerLevel, remaining.count)
            let batch = Array(remaining.prefix(batchSize))
            remaining = Array(remaining.dropFirst(batchSize))

            let packedConstraint = createPackedConstraint(
                batch,
                level: PackedTowerLevel(levelK),
                variableCount: variableCount
            )
            packedByLevel[levelK].append(packedConstraint)
            packed += batch.count
        }

        return packed
    }

    /// Pack with custom batch sizes per level.
    private func packCustom(_ constraints: [PackedR1CSConstraint], variableCount: Int, batchSizes: [Int]) -> Int {
        var remaining = constraints
        var packed = 0

        for (levelK, batchSize) in batchSizes.enumerated() {
            if remaining.isEmpty || levelK > config.maxPackedTowerLevel.k { break }

            let actualSize = min(batchSize, remaining.count)
            let batch = Array(remaining.prefix(actualSize))
            remaining = Array(remaining.dropFirst(actualSize))

            let packedConstraint = createPackedConstraint(
                batch,
                level: PackedTowerLevel(levelK),
                variableCount: variableCount
            )
            packedByLevel[levelK].append(packedConstraint)
            packed += batch.count
        }

        // Pack any remaining constraints using maximizeDensity
        if !remaining.isEmpty {
            packed += packMaximizingDensity(remaining, variableCount: variableCount)
        }

        return packed
    }

    /// Create a packed constraint from multiple R1CS constraints.
    private func createPackedConstraint(
        _ constraints: [PackedR1CSConstraint],
        level: PackedTowerLevel,
        variableCount: Int
    ) -> AmortizedPackedConstraint {
        // Combine A, B, C coefficients from all constraints
        var aCoeffs = [UInt8](repeating: 0, count: variableCount)
        var bCoeffs = [UInt8](repeating: 0, count: variableCount)
        var cCoeffs = [UInt8](repeating: 0, count: variableCount)

        for constraint in constraints {
            for (idx, val) in constraint.a {
                if idx < variableCount {
                    aCoeffs[idx] ^= val  // XOR for GF(2)
                }
            }
            for (idx, val) in constraint.b {
                if idx < variableCount {
                    bCoeffs[idx] ^= val
                }
            }
            for (idx, val) in constraint.c {
                if idx < variableCount {
                    cCoeffs[idx] ^= val
                }
            }
        }

        return AmortizedPackedConstraint(
            originalConstraints: constraints,
            towerLevel: level,
            aCoeffs: aCoeffs,
            bCoeffs: bCoeffs,
            cCoeffs: cCoeffs
        )
    }

    // MARK: - Query Interface

    /// Get all packed constraints at a tower level.
    public func packedConstraints(at level: PackedTowerLevel) -> [AmortizedPackedConstraint] {
        guard level.k <= config.maxPackedTowerLevel.k else { return [] }
        return packedByLevel[level.k]
    }

    /// Get all non-empty tower levels.
    public var activeLevels: [PackedTowerLevel] {
        (1...config.maxPackedTowerLevel.k).compactMap { k in
            packedByLevel[k].isEmpty ? nil : PackedTowerLevel(k)
        }
    }

    /// Evaluate packed constraint at a point using tower basis.
    ///
    /// - Parameters:
    ///   - constraint: Packed constraint to evaluate
    ///   - point: Evaluation point x
    ///   - witness: Witness values z
    /// - Returns: Triple (A(x)*z, B(x)*z, C(x)*z)
    public func evaluate(
        _ constraint: AmortizedPackedConstraint,
        at point: UInt8,
        witness: [UInt8]
    ) -> (a: UInt8, b: UInt8, c: UInt8) {
        // Get vanishing polynomial for this level
        let vanish = basisCache.evaluateVanishing(level: constraint.towerLevel, at: point)

        // Evaluate A, B, C using the witness
        var aEval: UInt8 = 0
        var bEval: UInt8 = 0
        var cEval: UInt8 = 0

        for (idx, coeff) in constraint.aCoeffs.enumerated() {
            if idx < witness.count {
                aEval ^= gf28Mul(coeff, witness[idx])
            }
        }
        for (idx, coeff) in constraint.bCoeffs.enumerated() {
            if idx < witness.count {
                bEval ^= gf28Mul(coeff, witness[idx])
            }
        }
        for (idx, coeff) in constraint.cCoeffs.enumerated() {
            if idx < witness.count {
                cEval ^= gf28Mul(coeff, witness[idx])
            }
        }

        // Multiply by vanishing polynomial
        aEval = gf28Mul(aEval, vanish)
        bEval = gf28Mul(bEval, vanish)
        cEval = gf28Mul(cEval, vanish)

        return (aEval, bEval, cEval)
    }

    /// GF(2^8) multiplication.
    private func gf28Mul(_ a: UInt8, _ b: UInt8) -> UInt8 {
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
                aa ^= 0x11B
            }
            bb >>= 1
        }
        return UInt8(p & 0xFF)
    }
}

// MARK: - Packing Efficiency Analyzer

/// Analyzes packing efficiency and provides optimization suggestions.
public struct PackingEfficiencyAnalyzer {
    public let packer: ConstraintPacker

    public init(packer: ConstraintPacker) {
        self.packer = packer
    }

    /// Overall packing efficiency.
    public var efficiency: Double { packer.packingEfficiency }

    /// Efficiency per tower level.
    public func efficiency(at level: PackedTowerLevel) -> Double {
        let packed = packer.packedConstraints(at: level)
        guard !packed.isEmpty else { return 0 }
        let totalConstraints = packed.reduce(0) { $0 + $1.originalConstraints.count }
        return Double(totalConstraints) / Double(packed.count)
    }

    /// Identify levels with low packing density.
    public var lowDensityLevels: [PackedTowerLevel] {
        packer.activeLevels.filter { efficiency(at: $0) < 2.0 }
    }

    /// Suggest improved packing configuration.
    public func suggestOptimizedConfig() -> PackingConfig {
        let current = packer.config

        // If low density levels exist, suggest onePerLevel
        if !lowDensityLevels.isEmpty && efficiency < 1.5 {
            return PackingConfig(
                strategy: .onePerLevel,
                maxPackedTowerLevel: current.maxPackedTowerLevel,
                enableConstraintReuse: current.enableConstraintReuse,
                enableSliceOptimization: current.enableSliceOptimization
            )
        }

        return current
    }
}
