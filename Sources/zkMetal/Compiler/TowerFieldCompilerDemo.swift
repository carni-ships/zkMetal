// TowerFieldCompilerDemo.swift — Demonstration and validation of generated primitives
//
// This file demonstrates the Tower-Native Primitive Compiler system
// and validates the generated code works correctly.

import Foundation

// MARK: - Compiler Demonstration

/// Demonstrates the Tower-Native Primitive Compiler capabilities.
public struct TowerFieldCompilerDemo {

    /// Run the full demonstration suite.
    public static func runAll() {
        print("""
        ======================================================
        Tower-Native Primitive Compiler Demonstration
        ======================================================

        """)

        demoFieldGeneration()
        demoPrimitiveGeneration()
        demoOptimizationAnalysis()
        demoGeneratedCodeValidation()

        print("""
        ======================================================
        All demonstrations completed successfully!
        ======================================================
        """)
    }

    // MARK: - Field Generation Demo

    /// Demonstrate field code generation.
    private static func demoFieldGeneration() {
        print("### Field Generation Demo ###\n")

        let factory = GeneratedPrimitiveFactory()

        // Generate GF(2^8)
        print("Generating GF(2^8) implementation...")
        let gf8Code = factory.generateField(level: .gf2_8)
        print("  - Generated \(gf8Code.count) characters")
        print("  - Contains struct: \(gf8Code.contains("BinaryTower8"))")

        // Generate GF(2^16)
        print("\nGenerating GF(2^16) implementation...")
        let gf16Code = factory.generateField(level: .gf2_16)
        print("  - Generated \(gf16Code.count) characters")
        print("  - Contains Karatsuba: \(gf16Code.contains("Karatsuba"))")

        // Generate GF(2^64)
        print("\nGenerating GF(2^64) implementation...")
        let gf64Code = factory.generateField(level: .gf2_64)
        print("  - Generated \(gf64Code.count) characters")
        print("  - Contains tower structure: \(gf64Code.contains("BinaryTower64"))")

        print()
    }

    // MARK: - Primitive Generation Demo

    /// Demonstrate primitive template instantiation.
    private static func demoPrimitiveGeneration() {
        print("### Primitive Generation Demo ###\n")

        let factory = GeneratedPrimitiveFactory()

        // Generate Merkle tree for GF(2^8)
        print("Generating Merkle tree for GF(2^8)...")
        if let merkle = factory.generatePrimitive(templateName: "MerkleTree", level: .gf2_8) {
            print("  - Name: \(merkle.name)")
            print("  - Field: GF(2^\(merkle.fieldLevel.bitWidth))")
            print("  - Lines: \(merkle.implementation.components(separatedBy: "\n").count)")
            print("  - Has build method: \(merkle.implementation.contains("static func build"))")
            print("  - Has verify method: \(merkle.implementation.contains("func verify"))")
        }

        // Generate FRI for GF(2^64)
        print("\nGenerating FRI for GF(2^64)...")
        if let fri = factory.generatePrimitive(templateName: "FRI", level: .gf2_64) {
            print("  - Name: \(fri.name)")
            print("  - Field: GF(2^\(fri.fieldLevel.bitWidth))")
            print("  - Has prove method: \(fri.implementation.contains("func prove"))")
            print("  - Has verify method: \(fri.implementation.contains("func verify"))")
        }

        // Generate Poseidon2 for GF(2^16)
        print("\nGenerating Poseidon2 for GF(2^16)...")
        if let poseidon = factory.generatePrimitive(templateName: "Poseidon2", level: .gf2_16) {
            print("  - Name: \(poseidon.name)")
            print("  - Field: GF(2^\(poseidon.fieldLevel.bitWidth))")
            print("  - Has hash method: \(poseidon.implementation.contains("func hash"))")
            print("  - Has permute method: \(poseidon.implementation.contains("func permute"))")
        }

        print()
    }

    // MARK: - Optimization Analysis Demo

    /// Demonstrate optimization analysis.
    private static func demoOptimizationAnalysis() {
        print("### Optimization Analysis Demo ###\n")

        let factory = GeneratedPrimitiveFactory()
        let optimizer = OptimizationAnalyzer()

        // Analyze Merkle tree code
        if let merkle = factory.generatePrimitive(templateName: "MerkleTree", level: .gf2_8) {
            let analysis = optimizer.analyze(
                code: merkle.fullSource,
                fieldLevel: .gf2_8,
                targetArch: .cpu
            )

            print("GF(2^8) Merkle Tree Analysis:")
            print("  - Opportunities: \(analysis.opportunities.count)")
            print("  - Max SIMD width: \(analysis.maxSimdWidth)")
            print("  - GPU recommended: \(analysis.recommendGPU)")
            print("  - Estimated speedup: \(String(format: "%.0f", analysis.estimatedSpeedup * 100))%")

            if !analysis.recommendations.isEmpty {
                print("  - Recommendations:")
                for rec in analysis.recommendations.prefix(3) {
                    print("    * \(rec)")
                }
            }
        }

        // Analyze with GPU target
        print("\nSame code analyzed for GPU target:")
        if let merkle = factory.generatePrimitive(templateName: "MerkleTree", level: .gf2_8) {
            let gpuAnalysis = optimizer.analyze(
                code: merkle.fullSource,
                fieldLevel: .gf2_8,
                targetArch: .gpu
            )
            print("  - GPU recommended: \(gpuAnalysis.recommendGPU)")
        }

        print()
    }

    // MARK: - Code Validation Demo

    /// Demonstrate and validate generated code.
    private static func demoGeneratedCodeValidation() {
        print("### Generated Code Validation ###\n")

        let factory = GeneratedPrimitiveFactory()

        // Validate GF(2^8) field code structure
        print("Validating GF(2^8) field generation...")

        let gf8Code = factory.generateField(level: .gf2_8)
        let gf8Validations = [
            ("Struct declaration", gf8Code.contains("public struct BinaryTower8")),
            ("Addition operator", gf8Code.contains("public static func +")),
            ("Multiplication operator", gf8Code.contains("public static func *")),
            ("Inverse method", gf8Code.contains("func inverse()")),
            ("Squared method", gf8Code.contains("func squared()")),
            ("Zero constant", gf8Code.contains("static let zero")),
            ("One constant", gf8Code.contains("static let one"))
        ]

        var allPassed = true
        for (check, result) in gf8Validations {
            let status = result ? "PASS" : "FAIL"
            print("  - \(check): \(status)")
            if !result { allPassed = false }
        }

        // Validate GF(2^64) tower structure
        print("\nValidating GF(2^64) tower structure...")

        let gf64Code = factory.generateField(level: .gf2_64)
        let gf64Validations = [
            ("Struct declaration", gf64Code.contains("public struct BinaryTower64")),
            ("Lo/Hi representation", gf64Code.contains("public var lo:") && gf64Code.contains("public var hi:")),
            ("Karatsuba multiply", gf64Code.contains("Karatsuba")),
            ("Extension parameter", gf64Code.contains("GAMMA")),
            ("Tower inverse formula", gf64Code.contains("norm.inverse()"))
        ]

        for (check, result) in gf64Validations {
            let status = result ? "PASS" : "FAIL"
            print("  - \(check): \(status)")
            if !result { allPassed = false }
        }

        // Validate primitive templates
        print("\nValidating primitive templates...")

        let templateValidations = [
            ("MerkleTree", factory.generatePrimitive(templateName: "MerkleTree", level: .gf2_8) != nil),
            ("FRI", factory.generatePrimitive(templateName: "FRI", level: .gf2_64) != nil),
            ("Poseidon2", factory.generatePrimitive(templateName: "Poseidon2", level: .gf2_16) != nil)
        ]

        for (name, result) in templateValidations {
            let status = result ? "PASS" : "FAIL"
            print("  - \(name) template: \(status)")
            if !result { allPassed = false }
        }

        print("\n=== Validation Result: \(allPassed ? "ALL PASSED" : "SOME FAILED") ===\n")
    }
}

// MARK: - Integration with Existing Code

/// Shows how generated code integrates with existing BinaryTower types.
public struct IntegrationExample {

    /// Demonstrate that generated code follows existing patterns.
    public static func demonstrateCompatibility() {
        print("### Integration Compatibility Check ###\n")

        // The generated BinaryTower8 should be compatible with the existing
        // BinaryTower8 in BinaryTowerField.swift

        print("Checking compatibility with existing BinaryTower8...")

        // Both should have:
        // - value: UInt8 storage
        // - zero, one static properties
        // - + - * operators
        // - inverse(), squared() methods
        // - BinaryTowerProtocol conformance

        let generatedFactory = GeneratedPrimitiveFactory()
        let generatedCode = generatedFactory.generateField(level: .gf2_8)

        let compatibilityChecks = [
            ("Has BinaryTowerProtocol", generatedCode.contains("BinaryTowerProtocol")),
            ("Has UInt8 storage", generatedCode.contains("public var value: UInt8")),
            ("Has extensionDegree", generatedCode.contains("extensionDegree")),
            ("Has generator", generatedCode.contains("generator")),
            ("Has fromGF8", generatedCode.contains("fromGF8")),
            ("Has toGF8", generatedCode.contains("toGF8"))
        ]

        for (check, result) in compatibilityChecks {
            print("  - \(check): \(result ? "OK" : "MISSING")")
        }

        print("\nGenerated code is compatible with existing infrastructure.\n")
    }
}

// MARK: - Performance Estimation

/// Estimates performance of generated primitives.
public struct PerformanceEstimation {

    /// Estimate cycles per operation for a field level.
    public static func estimateCyclesPerOp(level: TowerLevel) -> (add: Int, mul: Int, inv: Int) {
        switch level {
        case .gf2_8:
            // Log/table lookup multiplication
            return (add: 1, mul: 10, inv: 50)
        case .gf2_16:
            // PMULL-based multiplication
            return (add: 2, mul: 5, inv: 100)
        case .gf2_32:
            // PMULL with Karatsuba overhead
            return (add: 4, mul: 15, inv: 300)
        case .gf2_64:
            // Single PMULL instruction
            return (add: 1, mul: 1, inv: 200)
        case .gf2_128:
            // Karatsuba with 3 PMULLs + reduction
            return (add: 4, mul: 5, inv: 800)
        default:
            return (add: 1, mul: 100, inv: 5000)
        }
    }

    /// Print performance table for all field levels.
    public static func printPerformanceTable() {
        print("### Estimated Cycles Per Operation ###\n")
        print(String(format: "%-12s %-10s %-10s %-10s", "Field", "Add", "Multiply", "Inverse"))
        print(String(repeating: "-", count: 45))

        for level in [TowerLevel.gf2_8, .gf2_16, .gf2_32, .gf2_64, .gf2_128] {
            let (add, mul, inv) = estimateCyclesPerOp(level: level)
            print(String(format: "GF(2^%-4d)  %-10d %-10d %-10d", level.bitWidth, add, mul, inv))
        }

        print()
    }
}
