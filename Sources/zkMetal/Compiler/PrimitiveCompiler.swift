// PrimitiveCompiler.swift — Main entry point for the Tower-Native Primitive Compiler
//
// Usage:
//   let compiler = PrimitiveCompiler()
//   compiler.generateAll()

import Foundation

// MARK: - Primitive Compiler

/// Main compiler class that orchestrates primitive generation.
public final class PrimitiveCompiler {

    /// The factory for generating primitives
    public let factory: GeneratedPrimitiveFactory

    /// Output directory for generated code
    public var outputDirectory: String?

    /// Current configuration
    public var config: PrimitiveConfig

    public init(
        outputDirectory: String? = nil,
        config: PrimitiveConfig = PrimitiveConfig()
    ) {
        self.factory = GeneratedPrimitiveFactory()
        self.outputDirectory = outputDirectory
        self.config = config
    }

    // MARK: - Field Generation

    /// Generate all tower field implementations.
    public func generateAllFields() -> [TowerLevel: String] {
        factory.generateAllFields()
    }

    /// Generate a specific field level.
    public func generateField(level: TowerLevel) -> String {
        factory.generateField(level: level)
    }

    // MARK: - Primitive Generation

    /// Generate all supported primitives for a specific field level.
    public func generatePrimitives(for level: TowerLevel) -> [GeneratedPrimitive] {
        let templateNames = factory.registeredTemplates
        return factory.generatePrimitives(templateNames: templateNames, levels: [level], config: config)
    }

    /// Generate a specific primitive for a field level.
    public func generatePrimitive(templateName: String, level: TowerLevel) -> GeneratedPrimitive? {
        factory.generatePrimitive(templateName: templateName, level: level, config: config)
    }

    // MARK: - Code Output

    /// Generate and optionally write to disk.
    public func generateAndWrite(
        primitives: [GeneratedPrimitive],
        to directory: String? = nil,
        includeAnalysis: Bool = true
    ) throws -> String {
        let outputDir = directory ?? outputDirectory ?? "./Generated"

        try factory.writePrimitives(primitives, to: outputDir, includeAnalysis: includeAnalysis)

        return factory.generateModule(primitives: primitives, moduleName: "GeneratedPrimitives")
    }

    // MARK: - Analysis

    /// Analyze optimization opportunities for a field level.
    public func analyzeOptimizations(
        code: String,
        fieldLevel: TowerLevel
    ) -> OptimizationAnalysis {
        let analyzer = OptimizationAnalyzer()
        return analyzer.analyze(code: code, fieldLevel: fieldLevel, targetArch: config.targetArch)
    }

    /// Print analysis results.
    public func printAnalysis(_ analysis: OptimizationAnalysis) {
        print("\n=== Optimization Analysis ===")
        print("Max SIMD Width: \(analysis.maxSimdWidth)")
        print("GPU Recommended: \(analysis.recommendGPU)")
        print("Estimated Speedup: \(String(format: "%.0f", analysis.estimatedSpeedup * 100))%")
        print("\nOpportunities Found: \(analysis.opportunities.count)")
        for opp in analysis.opportunities {
            print("  - \(opp.description) (gain: \(String(format: "%.0f", opp.estimatedGain * 100))%)")
        }
        print("\nRecommendations:")
        for rec in analysis.recommendations {
            print("  - \(rec)")
        }
    }
}

// MARK: - Standalone Generation Example

/// Demonstrates generating GF(2^8) arithmetic as a first step.
public func demonstrateGF8Generation() {
    print("=== Tower-Native Primitive Compiler Demo ===\n")

    let compiler = PrimitiveCompiler()

    // 1. Generate GF(2^8) field
    print("1. Generating GF(2^8) field implementation...")
    let gf8Code = compiler.generateField(level: .gf2_8)
    print("   Generated \(gf8Code.components(separatedBy: "\n").count) lines of code")

    // 2. Generate primitives for GF(2^8)
    print("\n2. Generating primitives for GF(2^8)...")
    let primitives = compiler.generatePrimitives(for: .gf2_8)
    for primitive in primitives {
        print("   - \(primitive.name): \(primitive.implementation.components(separatedBy: "\n").count) lines")
        if let analysis = primitive.optimizationAnalysis {
            print("     Estimated speedup: \(String(format: "%.0f", analysis.estimatedSpeedup * 100))%")
        }
    }

    // 3. Analyze the generated code
    print("\n3. Analyzing GF(2^8) Merkle tree for optimizations...")
    if let merkle = primitives.first(where: { $0.name == "MerkleTree" }) {
        compiler.printAnalysis(compiler.analyzeOptimizations(
            code: merkle.fullSource,
            fieldLevel: .gf2_8
        ))
    }

    print("\n=== Demo Complete ===")
}

// MARK: - Full Generation Workflow

/// Full workflow for generating all primitives.
public func generateAllPrimitives(outputDir: String = "./Sources/zkMetal/Compiler/Generated") {
    let compiler = PrimitiveCompiler()

    // Generate for key field levels
    let levels: [TowerLevel] = [.gf2_8, .gf2_16, .gf2_32, .gf2_64]

    var allPrimitives = [GeneratedPrimitive]()

    for level in levels {
        print("Generating primitives for GF(2^\(level.bitWidth))...")
        let primitives = compiler.generatePrimitives(for: level)
        allPrimitives.append(contentsOf: primitives)
    }

    // Write to disk
    do {
        print("\nWriting generated code to \(outputDir)...")
        _ = try compiler.generateAndWrite(primitives: allPrimitives, to: outputDir)
        print("Generation complete!")
    } catch {
        print("Error writing generated code: \(error)")
    }
}
