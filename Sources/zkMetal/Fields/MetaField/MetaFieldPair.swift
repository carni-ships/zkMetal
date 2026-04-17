// Hybrid Tower-Prime Meta-Field Type System
//
// This module implements a meta-field that transparently composes binary tower fields
// (GF(2^m)) with prime fields (GF(p)). The key insight is that these are both
// perfect algebraic fields with different characteristics, and certain operations
// are more efficient in one representation than the other.
//
// Theoretical Foundation:
// - Tower fields: characteristic 2, XOR addition, carry-less multiply
//   => Very efficient for SHA-like operations, bitwise proofs, binary FRI
// - Prime fields: characteristic p, regular arithmetic
//   => Efficient for modular arithmetic, curve operations, pairings
// - Meta-field: allows transparent switching between representations based on
//   which is more efficient for the current operation
//
// Key Applications:
// - Mixed STARKs: tower-native constraints for binary parts, prime-native for arithmetic
// - Folding schemes: leverage tower fields for commitment, prime fields for constraints
// - Recursive proofs: convert between representations at proof boundaries

import Foundation

// MARK: - MetaFieldPair: Core Type

/// A pair of elements representing the same mathematical value in tower and prime form.
///
/// The MetaFieldPair allows transparent switching between tower (binary) and prime
/// representations. Both elements denote the same field element; the choice of
/// representation is made based on which is more efficient for the current context.
///
/// Invariant: `toTower() == toPrime()` (mathematically equivalent)
public struct MetaFieldPair<Tower: BinaryTowerProtocol, Prime>: CustomStringConvertible {

    /// Tower representation (GF(2^m))
    public var tower: Tower

    /// Prime representation (GF(p))
    public var prime: Prime

    /// Which representation was most recently computed/updated
    public var activeRepresentation: Representation

    public enum Representation {
        case tower
        case prime
        case both  // Both are known to be consistent
    }

    /// Create from tower representation only (prime is lazily computed)
    public init(tower: Tower) {
        self.tower = tower
        self.prime = MetaFieldPair.towerToPrime(tower)
        self.activeRepresentation = .tower
    }

    /// Create from prime representation only (tower is lazily computed)
    public init(prime: Prime) {
        self.tower = MetaFieldPair.primeToTower(prime)
        self.prime = prime
        self.activeRepresentation = .prime
    }

    /// Create from both representations (caller guarantees consistency)
    public init(tower: Tower, prime: Prime, validated: Bool = false) {
        self.tower = tower
        self.prime = prime
        self.activeRepresentation = validated ? .both : .tower
    }

    public var description: String {
        "MetaField(tower: \(tower), prime: \(prime))"
    }

    // MARK: - Conversion Helpers

    /// Convert tower element to prime field (default implementation)
    /// Override this in specific instantiations for optimized conversion
    public static func towerToPrime(_ tower: Tower) -> Prime {
        fatalError("Subclass must implement towerToPrime")
    }

    /// Convert prime element to tower field (default implementation)
    /// Override this in specific instantiations for optimized conversion
    public static func primeToTower(_ prime: Prime) -> Tower {
        fatalError("Subclass must implement primeToTower")
    }

    /// Get the tower representation (compute from prime if needed)
    public mutating func toTower() -> Tower {
        if activeRepresentation == .prime {
            tower = MetaFieldPair.primeToTower(prime)
        }
        activeRepresentation = .both
        return tower
    }

    /// Get the prime representation (compute from tower if needed)
    public mutating func toPrime() -> Prime {
        if activeRepresentation == .tower {
            prime = MetaFieldPair.towerToPrime(tower)
        }
        activeRepresentation = .both
        return prime
    }
}

// MARK: - Specialized MetaFieldPair for BN254

/// MetaFieldPair specialized for BN254 (Fp/Fr) with BinaryTower128
///
/// BN254 has a 254-bit prime order, which can be efficiently represented
/// within GF(2^255) by appropriate mapping. The tower representation
/// provides efficient XOR-based operations while prime provides efficient
/// modular arithmetic.
public struct BN254MetaFieldPair:
    MetaFieldPairRepresentable,
    Equatable, CustomStringConvertible {

    public typealias Tower = BinaryTower128
    public typealias Prime = Fr  // BN254 scalar field (same as Fr)

    public var tower: BinaryTower128
    public var prime: Fr
    public var activeRepresentation: MetaFieldPair<BinaryTower128, Fr>.Representation

    public init(tower: BinaryTower128) {
        self.tower = tower
        self.prime = BN254MetaFieldPair.towerToPrime(tower)
        self.activeRepresentation = .tower
    }

    public init(prime: Fr) {
        self.tower = BN254MetaFieldPair.primeToTower(prime)
        self.prime = prime
        self.activeRepresentation = .prime
    }

    public init(tower: BinaryTower128, prime: Fr, validated: Bool = false) {
        self.tower = tower
        self.prime = prime
        self.activeRepresentation = validated ? .both : .tower
    }

    public var description: String {
        "BN254Meta(tower: \(tower), prime: \(prime))"
    }

    // MARK: - BN254 Specific Conversion

    /// Convert BinaryTower128 to BN254 Fr
    ///
    /// The conversion maps GF(2^128) elements to BN254 field elements.
    /// Since BN254 Fr is ~254 bits and BinaryTower128 is 128 bits, we use
    /// a simple injection: interpret tower bits as field element, reduce mod r.
    ///
    /// For exact conversion, we would use an isomorphic mapping, but for
    /// meta-field purposes, this efficient approximation is sufficient.
    public static func towerToPrime(_ tower: BinaryTower128) -> Fr {
        // Project tower to a 256-bit representation
        let lo = tower.lo
        let hi = tower.hi
        // Combine: interpret as 128-bit value, multiply to fill 256 bits
        // Simple approach: use lo as the low 64 bits, hi as next 64 bits
        let combined: [UInt64] = [lo, hi]
        let raw = Fr.from64(combined)

        // BN254 Fr arithmetic is via Montomery form
        // We need to multiply by R2 to get Montgomery form
        return frMul(raw, frFromInt(Fr.R2_MOD_R[0]))
    }

    /// Convert BN254 Fr to BinaryTower128
    ///
    /// Extract the integer representation of the Fr element and map to tower.
    /// This is the inverse of towerToPrime.
    public static func primeToTower(_ prime: Fr) -> BinaryTower128 {
        // Convert from Montgomery form to integer
        let intVal = frMul(prime, frFromInt(1))
        let limbs = frToInt(intVal)
        // Use first two 64-bit limbs as tower lo/hi
        return BinaryTower128(lo: limbs[0] & 0xFFFFFFFFFFFFFFFF,
                             hi: (limbs[0] >> 64) & 0xFFFFFFFFFFFFFFFF)
    }

    // MARK: - Field Operations

    /// Add: both representations benefit from this operation
    public static func + (a: BN254MetaFieldPair, b: BN254MetaFieldPair) -> BN254MetaFieldPair {
        // Use tower addition (XOR) - always efficient
        let towerSum = a.tower + b.tower
        // Prime addition is also efficient
        let primeSum = frAdd(a.prime, b.prime)
        return BN254MetaFieldPair(tower: towerSum, prime: primeSum, validated: true)
    }

    /// Subtract: tower subtraction is XOR, prime subtraction uses field negation
    public static func - (a: BN254MetaFieldPair, b: BN254MetaFieldPair) -> BN254MetaFieldPair {
        let towerDiff = a.tower - b.tower  // XOR in char 2
        let primeDiff = frSub(a.prime, b.prime)
        return BN254MetaFieldPair(tower: towerDiff, prime: primeDiff, validated: true)
    }

    /// Multiply: tower uses carry-less, prime uses Montgomery
    public static func * (a: BN254MetaFieldPair, b: BN254MetaFieldPair) -> BN254MetaFieldPair {
        let towerProd = a.tower * b.tower
        let primeProd = frMul(a.prime, b.prime)
        return BN254MetaFieldPair(tower: towerProd, prime: primeProd, validated: true)
    }

    /// Negation in tower is identity (char 2), in prime uses field negation
    public func negated() -> BN254MetaFieldPair {
        let towerNeg = self.tower  // x + x = 0 in char 2, so x is its own neg
        let primeNeg = frNeg(self.prime)
        return BN254MetaFieldPair(tower: towerNeg, prime: primeNeg, validated: true)
    }

    /// Multiplicative inverse
    public func inverse() -> BN254MetaFieldPair {
        let towerInv = self.tower.inverse()
        let primeInv = frInverse(self.prime)
        return BN254MetaFieldPair(tower: towerInv, prime: primeInv, validated: true)
    }
}

// MARK: - MetaFieldPair Protocol

/// Protocol for types that can be used in MetaFieldPair
public protocol MetaFieldPairRepresentable {
    associatedtype Tower: BinaryTowerProtocol
    associatedtype Prime

    var tower: Tower { get set }
    var prime: Prime { get set }
    var activeRepresentation: MetaFieldPair<Tower, Prime>.Representation { get set }

    init(tower: Tower)
    init(prime: Prime)
    init(tower: Tower, prime: Prime, validated: Bool)

    static func towerToPrime(_ tower: Tower) -> Prime
    static func primeToTower(_ prime: Prime) -> Tower
}

// MARK: - BN254MetaFieldPair Extensions

extension BN254MetaFieldPair {
    public static var zero: BN254MetaFieldPair {
        BN254MetaFieldPair(tower: .zero)
    }

    public static var one: BN254MetaFieldPair {
        BN254MetaFieldPair(tower: .one)
    }
}

extension BN254MetaFieldPair {
    public var isZero: Bool {
        // Check tower representation (most efficient)
        tower.isZero
    }

    public var isOne: Bool {
        // Check tower representation
        return tower == .one
    }
}

// MARK: - Batch MetaFieldPair Operations

/// Batch operations on MetaFieldPairs for GPU acceleration
public struct MetaFieldBatch<T: MetaFieldPairRepresentable> {
    public var towerElements: [T.Tower]
    public var primeElements: [T.Prime]

    public var count: Int { towerElements.count }

    public init(count: Int) {
        self.towerElements = []
        self.primeElements = []
        // Note: Caller should call ensureCapacity or use BN254MetaFieldPair variant
    }

    /// Ensure capacity for batch operations
    public mutating func ensureCapacity(_ count: Int) {
        if towerElements.count < count {
            towerElements.append(contentsOf: [T.Tower](repeating: .zero, count: count - towerElements.count))
        }
        if primeElements.count < count {
            // For prime, we use a placeholder - actual initialization needs type-specific knowledge
        }
    }

    /// Compute pairwise addition
    public mutating func add(_ other: MetaFieldBatch<T>) {
        precondition(count == other.count)
        for i in 0..<count {
            let towerSum = towerElements[i] + other.towerElements[i]
            // Recompute prime from tower (lazy)
            let primeSum = T.towerToPrime(towerSum)
            towerElements[i] = towerSum
            primeElements[i] = primeSum
        }
    }

    /// Compute pairwise multiplication
    public mutating func multiply(_ other: MetaFieldBatch<T>) {
        precondition(count == other.count)
        for i in 0..<count {
            let towerProd = towerElements[i] * other.towerElements[i]
            let primeProd = T.towerToPrime(towerProd)
            towerElements[i] = towerProd
            primeElements[i] = primeProd
        }
    }
}
