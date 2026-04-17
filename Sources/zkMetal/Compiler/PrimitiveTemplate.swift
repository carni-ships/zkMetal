// PrimitiveTemplate.swift — Template system for cryptographic primitives
//
// Defines generic templates for cryptographic primitives that can be instantiated
// for any tower field. Templates specify operations in generic field terms,
// and the compiler generates optimized tower-native implementations.

import Foundation

// MARK: - Field-Generic Operation Protocols

/// Represents a field element that supports the basic arithmetic operations
/// needed for cryptographic primitive templates.
public protocol FieldElement: Equatable, CustomStringConvertible {
    associatedtype Field: FieldLike

    static var zero: Self { get }
    static var one: Self { get }

    var isZero: Bool { get }

    static func + (lhs: Self, rhs: Self) -> Self
    static func - (lhs: Self, rhs: Self) -> Self
    static func * (lhs: Self, rhs: Self) -> Self

    func inverse() -> Self
    func squared() -> Self
}

/// Protocol for field types that provide arithmetic operations.
public protocol FieldLike {
    associatedtype Element: FieldElement

    static var zero: Element { get }
    static var one: Element { get }

    static func add(_ a: Element, _ b: Element) -> Element
    static func multiply(_ a: Element, _ b: Element) -> Element
    static func inverse(_ a: Element) -> Element
}

// MARK: - Primitive Template Protocol

/// A template for a cryptographic primitive that can be instantiated
/// for any tower field level.
///
/// Templates define the generic algorithm in field-agnostic terms.
/// The compiler uses these templates to generate optimized implementations.
public protocol PrimitiveTemplate {
    /// The name of this primitive (e.g., "Poseidon2", "MerkleTree", "FRI")
    static var name: String { get }

    /// The tower field levels this primitive can be instantiated for
    static var supportedFieldLevels: [TowerLevel] { get }

    /// Generate the type declarations for this primitive at the given field level.
    /// Returns Swift source code as a string.
    static func generateTypeDeclaration(fieldLevel: TowerLevel) -> String

    /// Generate the implementation for this primitive at the given field level.
    /// Returns Swift source code as a string.
    static func generateImplementation(fieldLevel: TowerLevel) -> String
}

// MARK: - Tower Level Specification

/// Represents a specific level in the binary tower field hierarchy.
/// The tower: GF(2) -> GF(2^2) -> GF(2^4) -> GF(2^8) -> GF(2^16) -> GF(2^32) -> GF(2^64) -> GF(2^128)
public enum TowerLevel: Int, CaseIterable {
    case gf2_1 = 1      // GF(2) — base field
    case gf2_2 = 2      // GF(2^2)
    case gf2_4 = 4      // GF(2^4)
    case gf2_8 = 8      // GF(2^8) — AES field
    case gf2_16 = 16    // GF(2^16)
    case gf2_32 = 32    // GF(2^32)
    case gf2_64 = 64    // GF(2^64)
    case gf2_128 = 128  // GF(2^128)

    /// The Swift type name for this tower level
    public var swiftTypeName: String {
        switch self {
        case .gf2_1: return "BinaryTower1"
        case .gf2_2: return "BinaryTower2"
        case .gf2_4: return "BinaryTower4"
        case .gf2_8: return "BinaryTower8"
        case .gf2_16: return "BinaryTower16"
        case .gf2_32: return "BinaryTower32"
        case .gf2_64: return "BinaryTower64"
        case .gf2_128: return "BinaryTower128"
        }
    }

    /// Native Swift integer type for storage
    public var nativeIntegerType: String {
        switch self {
        case .gf2_1: return "UInt8"  // Use UInt8 for single bit
        case .gf2_2: return "UInt8"
        case .gf2_4: return "UInt8"
        case .gf2_8: return "UInt8"
        case .gf2_16: return "UInt16"
        case .gf2_32: return "UInt32"
        case .gf2_64: return "UInt64"
        case .gf2_128: return "(UInt64, UInt64)"
        }
    }

    /// The underlying bit width of a single element
    public var bitWidth: Int { rawValue }

    /// The extension degree relative to GF(2)
    public var extensionDegree: Int { rawValue }
}

// MARK: - Template Instantiation Context

/// Context for template instantiation, capturing the field level
/// and any additional configuration.
public struct TemplateInstantiationContext {
    public let fieldLevel: TowerLevel
    public let config: PrimitiveConfig

    public init(fieldLevel: TowerLevel, config: PrimitiveConfig = PrimitiveConfig()) {
        self.fieldLevel = fieldLevel
        self.config = config
    }
}

/// Configuration for primitive generation.
public struct PrimitiveConfig {
    /// Whether to enable SIMD vectorization
    public var vectorize: Bool = false

    /// Target architecture (cpu, gpu, or both)
    public var targetArch: TargetArchitecture = .cpu

    /// Unroll factor for loops (0 = auto)
    public var unrollFactor: Int = 0

    /// Enable experimental optimizations
    public var experimentalOptimizations: Bool = false

    public init(
        vectorize: Bool = false,
        targetArch: TargetArchitecture = .cpu,
        unrollFactor: Int = 0,
        experimentalOptimizations: Bool = false
    ) {
        self.vectorize = vectorize
        self.targetArch = targetArch
        self.unrollFactor = unrollFactor
        self.experimentalOptimizations = experimentalOptimizations
    }
}

/// Target architecture for code generation.
public enum TargetArchitecture {
    case cpu
    case gpu
    case both
}

// MARK: - Primitive Registry

/// Registry for all available primitive templates.
public enum PrimitiveRegistry {
    private static var templates: [String: PrimitiveTemplate.Type] = [:]

    /// Register a primitive template.
    public static func register(_ template: PrimitiveTemplate.Type) {
        templates[template.name] = template
    }

    /// Get a template by name.
    public static func get(_ name: String) -> PrimitiveTemplate.Type? {
        templates[name]
    }

    /// Get all registered template names.
    public static var allTemplateNames: [String] {
        Array(templates.keys).sorted()
    }
}

// MARK: - Convenience Extensions

extension FieldElement {
    /// Default pow implementation via square-and-multiply.
    public func pow(_ exponent: Int) -> Self {
        if exponent == 0 { return Self.one }
        var result = Self.one
        var base = self
        var exp = exponent
        while exp > 0 {
            if exp & 1 == 1 { result = result * base }
            base = base.squared()
            exp >>= 1
        }
        return result
    }
}
