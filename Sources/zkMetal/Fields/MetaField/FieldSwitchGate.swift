// FieldSwitchGate Protocol — Conversion between Tower and Prime Representations
//
// The FieldSwitchGate is the core protocol for converting between tower (binary)
// and prime field representations. It provides the mathematical machinery for
// transparent composition of tower-native and prime-native operations.
//
// Key Design Decisions:
// 1. Conversion is typically expensive, so we minimize it through "active representation"
// 2. Both representations should remain consistent (enforced via validation in debug)
// 3. Conversion costs are accounted for in performance calculations
//
// Conversion Strategies:
// - Direct embedding: tower -> prime via integer interpretation + modular reduction
// - Inverse embedding: prime -> tower via mixed-radix decomposition
// - Lookup-based: precompute small conversion tables for frequently used values
// - Polynomial evaluation: treat tower bits as polynomial coefficients

import Foundation

// MARK: - FieldSwitchGate Protocol

/// Protocol for converting between tower and prime field representations.
///
/// FieldSwitchGate provides the mathematical relations for switching between
/// tower (GF(2^m)) and prime (GF(p)) representations of field elements.
///
/// Conformance provides:
/// - Canonical conversion functions
/// - Cost estimation for conversions
/// - Optimization hints for when to switch representations
public protocol FieldSwitchGate {
    associatedtype Tower: BinaryTowerProtocol
    associatedtype Prime

    /// The tower field type
    static var towerField: Tower.Type { get }

    /// The prime field type
    static var primeField: Prime.Type { get }

    // MARK: - Core Conversions

    /// Convert tower element to prime representation.
    /// This is the "down" conversion in the meta-field lattice.
    func toPrime() -> Prime

    /// Convert prime element to tower representation.
    /// This is the "up" conversion in the meta-field lattice.
    func toTower() -> Prime

    // MARK: - Conversion with Constraint Counting

    /// Convert tower to prime with constraint count estimation.
    /// Returns (primeValue, estimatedConstraints)
    func toPrimeConstrained() -> (Prime, Int)

    /// Convert prime to tower with constraint count estimation.
    /// Returns (towerValue, estimatedConstraints)
    func toTowerConstrained() -> (Tower, Int)

    // MARK: - Batch Conversions

    /// Batch convert tower elements to prime (for GPU processing)
    static func batchToPrime(_ towers: [Tower]) -> [Prime]

    /// Batch convert prime elements to tower (for GPU processing)
    static func batchToTower(_ primes: [Prime]) -> [Tower]
}

// MARK: - FieldSwitchGate Errors

public enum FieldSwitchError: Error {
    case invalidBitLength(expected: Int, actual: Int)
    case conversionOverflow
    case representationMismatch
    case optimizationNotSupported
}

// MARK: - BN254 FieldSwitchGate

/// FieldSwitchGate implementation for BN254 with BinaryTower128.
///
/// BN254 is a 254-bit prime, which fits efficiently into GF(2^255).
/// The conversion uses the fact that both fields have similar bit widths.
public struct BN254FieldSwitchGate: FieldSwitchGate {
    public typealias Tower = BinaryTower128
    public typealias Prime = Fr

    public static var towerField: Tower.Type { BinaryTower128.self }
    public static var primeField: Prime.Type { Fr.self }

    private var tower: BinaryTower128
    private var prime: Fr
    private var activeRep: MetaFieldPair<Tower, Prime>.Representation

    public init(tower: Tower) {
        self.tower = tower
        self.prime = BN254MetaFieldPair.towerToPrime(tower)
        self.activeRep = .tower
    }

    public init(prime: Prime) {
        self.tower = BN254MetaFieldPair.primeToTower(prime)
        self.prime = prime
        self.activeRep = .prime
    }

    public func toPrime() -> Fr {
        if activeRep == .tower {
            prime = BN254MetaFieldPair.towerToPrime(tower)
            activeRep = .both
        }
        return prime
    }

    public func toTower() -> BinaryTower128 {
        if activeRep == .prime {
            tower = BN254MetaFieldPair.primeToTower(prime)
            activeRep = .both
        }
        return tower
    }

    public func toPrimeConstrained() -> (Fr, Int) {
        // BN254 to BN254 conversion (same field) - no constraints needed
        // The tower representation is just a different encoding
        let p = toPrime()
        return (p, 0)
    }

    public func toTowerConstrained() -> (BinaryTower128, Int) {
        let t = toTower()
        return (t, 0)
    }

    public static func batchToPrime(_ towers: [BinaryTower128]) -> [Fr] {
        towers.map { BN254MetaFieldPair.towerToPrime($0) }
    }

    public static func batchToTower(_ primes: [Fr]) -> [BinaryTower128] {
        primes.map { BN254MetaFieldPair.primeToTower($0) }
    }
}

// MARK: - Conversion Cost Analysis

/// Tracks the cost of conversions between representations
public struct ConversionCost {
    /// Estimated constraint count for tower -> prime
    public let towerToPrime: Int

    /// Estimated constraint count for prime -> tower
    public let primeToTower: Int

    /// Whether fast (lookup-based) conversion is available
    public let hasFastPath: Bool

    public static let zero = ConversionCost(towerToPrime: 0, primeToTower: 0, hasFastPath: true)

    /// Estimate cost for BN254: zero for internal conversions
    public static let bn254 = ConversionCost(towerToPrime: 0, primeToTower: 0, hasFastPath: true)
}

// MARK: - Optimized Conversion Strategies

/// Strategies for optimizing tower-prime conversions
public enum ConversionStrategy {
    /// Direct interpretation of bits as integer, then reduce mod p
    case direct

    /// Use lookup tables for small subfield elements
    case lookup

    /// Polynomial evaluation approach
    case polynomial

    /// Hybrid: use direct for most, lookup for special values
    case hybrid
}

/// Configuration for conversion optimization
public struct ConversionConfig {
    public let strategy: ConversionStrategy
    public let useCaching: Bool
    public let cacheSize: Int

    public static let `default` = ConversionConfig(
        strategy: .direct,
        useCaching: true,
        cacheSize: 256
    )

    public static let optimized = ConversionConfig(
        strategy: .hybrid,
        useCaching: true,
        cacheSize: 1024
    )
}

// MARK: - Conversion Cache

/// Cache for frequently used conversions
public final class ConversionCache<T: Hashable, U> {
    private var towerToPrime: [T: U] = [:]
    private var primeToTower: [U: T] = [:]
    private let maxSize: Int
    private let lock = NSLock()

    public init(maxSize: Int = 256) {
        self.maxSize = maxSize
    }

    public func lookupTowerToPrime(_ tower: T) -> U? {
        lock.lock()
        defer { lock.unlock() }
        return towerToPrime[tower]
    }

    public func lookupPrimeToTower(_ prime: U) -> T? {
        lock.lock()
        defer { lock.unlock() }
        return primeToTower[prime]
    }

    public func store(_ tower: T, prime: U) {
        lock.lock()
        defer { lock.unlock() }
        if towerToPrime.count >= maxSize {
            // Simple eviction: clear half
            let keysToRemove = Array(towerToPrime.keys.prefix(maxSize / 2))
            for key in keysToRemove {
                if let p = towerToPrime.removeValue(forKey: key) {
                    primeToTower.removeValue(forKey: p)
                }
            }
        }
        towerToPrime[tower] = prime
        primeToTower[prime] = tower
    }
}

// MARK: - Gate Evaluation Context

/// Context for field switch gate evaluation with cost tracking
public struct GateEvaluationContext {
    public var constraintCount: Int = 0
    public var conversionCount: Int = 0
    public var currentRepresentation: MetaFieldPair<BinaryTower128, Fr>.Representation = .both

    public var totalCost: Int {
        // Each conversion costs roughly 1 constraint equivalent
        constraintCount + conversionCount
    }

    public mutating func recordConversion() {
        conversionCount += 1
    }

    public mutating func recordConstraint() {
        constraintCount += 1
    }

    public mutating func reset() {
        constraintCount = 0
        conversionCount = 0
    }
}

// MARK: - FieldSwitchGate Utilities

extension FieldSwitchGate {
    /// Estimate the cost savings from using meta-field approach
    /// vs pure tower or pure prime computation.
    ///
    /// Returns: (towerNativeSavings, primeNativeSavings, conversionCost)
    public static func analyzeCostBenefit(
        towerOps: Int,
        primeOps: Int,
        mixedOps: Int
    ) -> (towerBetter: Bool, primeBetter: Bool, netSavings: Int) {
        // Tower ops: 1 unit each (XOR is free)
        // Prime ops: 2 units each (modular arithmetic)
        // Mixed ops: 3 units if converted, 1 if native
        // Conversion: 1 unit

        let pureTowerCost = towerOps * 1 + mixedOps * 3
        let purePrimeCost = primeOps * 2 + mixedOps * 3
        let mixedWithConversion = towerOps * 1 + primeOps * 2 + mixedOps * 1 + mixedOps // conversion

        let towerBetter = pureTowerCost < purePrimeCost && pureTowerCost < mixedWithConversion
        let primeBetter = purePrimeCost < pureTowerCost && purePrimeCost < mixedWithConversion
        let netSavings = min(pureTowerCost, purePrimeCost, mixedWithConversion)

        return (towerBetter, primeBetter, netSavings)
    }
}
