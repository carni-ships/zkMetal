// Compiler.swift — Tower-Native Primitive Compiler for zkMetal
//
// This module provides a compiler that automatically generates tower-field-native
// implementations of cryptographic primitives.
//
// # Overview
//
// The compiler takes generic primitive templates and generates optimized Swift code
// for specific binary tower field levels (GF(2^8), GF(2^16), GF(2^32), GF(2^64), GF(2^128)).
//
// # Usage
//
// ```swift
// import Compiler
//
// let compiler = PrimitiveCompiler()
// let primitives = compiler.generatePrimitives(for: .gf2_8)
// ```
//
// # Components
//
// - **PrimitiveTemplate**: Protocol for defining cryptographic primitives
// - **TowerFieldCodeGenerator**: Generates field arithmetic implementations
// - **OptimizationFramework**: Analyzes and applies optimizations
// - **GeneratedPrimitiveFactory**: Creates instantiated primitives
// - **PrimitiveCompiler**: Main entry point

// MARK: - Module Exports

@_exported import Foundation

// Re-export key types for convenience
public typealias TowerLevel = TowerLevel

// MARK: - Version

public enum CompilerVersion {
    public static let major = 1
    public static let minor = 0
    public static let patch = 0
    public static let version = "\(major).\(minor).\(patch)"
    public static let fullName = "Tower-Native Primitive Compiler v\(version)"
}

// MARK: - Quick Start

/// Generate all primitives for a field level with default settings.
public func generatePrimitives(for level: TowerLevel) -> [GeneratedPrimitive] {
    let compiler = PrimitiveCompiler()
    return compiler.generatePrimitives(for: level)
}

/// Generate a specific primitive template for a field level.
public func generatePrimitive(
    template: String,
    level: TowerLevel
) -> GeneratedPrimitive? {
    let compiler = PrimitiveCompiler()
    return compiler.generatePrimitive(templateName: template, level: level)
}

/// Generate field implementation for a tower level.
public func generateField(level: TowerLevel) -> String {
    let compiler = PrimitiveCompiler()
    return compiler.generateField(level: level)
}
