// MetaField Module — Hybrid Tower-Prime Field Type System
//
// This module provides the complete MetaField type system for zkMetal, enabling
// transparent composition of binary tower fields (GF(2^m)) and prime fields (GF(p)).
//
// Components:
// - MetaFieldPair: Core paired representation type
// - FieldSwitchGate: Protocol for conversions
// - EncodingRelations: Mathematical relations for transparent composition
// - MetaFieldCircuit: Integration with constraint system
//
// Usage:
//   let tower = BinaryTower128(lo: 0xDEAD, hi: 0xBEEF)
//   let meta = BN254MetaFieldPair(tower: tower)
//
//   // Operate in either representation
//   let sum = meta + meta  // uses tower (XOR) then converts to prime
//   let prod = meta * meta // uses tower Karatsuba
//
//   // Explicit conversion when needed
//   let prime = meta.toPrime()
//   let tower = meta.toTower()

import Foundation

// MARK: - Public Types

/// Type alias for tower field (currently BinaryTower128)
public typealias TowerField = BinaryTower128

/// Type alias for prime field (currently BN254 Fr)
public typealias PrimeField = Fr

// MARK: - Version

/// MetaField module version
public enum MetaFieldVersion {
    public static let major = 1
    public static let minor = 0
    public static let patch = 0
    public static let string = "1.0.0"
}
