// GeneratedPrimitiveFactory.swift — Factory for generating and instantiating primitives
//
// Provides a unified interface for generating cryptographic primitives
// optimized for specific tower field levels.

import Foundation

// MARK: - Generated Primitive Types

/// Represents a generated primitive with its type and implementation.
public struct GeneratedPrimitive {
    /// The name of the primitive
    public let name: String

    /// The tower field level
    public let fieldLevel: TowerLevel

    /// The generated type declaration code
    public let typeDeclaration: String

    /// The generated implementation code
    public let implementation: String

    /// Optimization analysis results
    public let optimizationAnalysis: OptimizationAnalysis?

    /// Combined Swift source code (type + implementation)
    public var fullSource: String {
        typeDeclaration + "\n" + implementation
    }

    public init(
        name: String,
        fieldLevel: TowerLevel,
        typeDeclaration: String,
        implementation: String,
        optimizationAnalysis: OptimizationAnalysis? = nil
    ) {
        self.name = name
        self.fieldLevel = fieldLevel
        self.typeDeclaration = typeDeclaration
        self.implementation = implementation
        self.optimizationAnalysis = optimizationAnalysis
    }
}

// MARK: - Primitive Factory

/// Factory for generating tower-native cryptographic primitives.
public final class GeneratedPrimitiveFactory {

    /// Code generator for tower fields
    public let fieldGenerator = TowerFieldCodeGenerator()

    /// Batch operations generator
    public let batchGenerator = BatchOperationsGenerator()

    /// Optimization analyzer
    public let optimizer = OptimizationAnalyzer()

    /// Registered templates
    private var templates: [String: PrimitiveTemplate.Type] = [:]

    public init() {
        // Register all built-in templates
        templates[Poseidon2Template.name] = Poseidon2Template.self
        templates[MerkleTreeTemplate.name] = MerkleTreeTemplate.self
        templates[FRITemplate.name] = FRITemplate.self
    }

    // MARK: - Field Generation

    /// Generate a tower field implementation for the given level.
    /// - Parameter level: The tower field level to generate
    /// - Returns: Generated Swift source code
    public func generateField(level: TowerLevel) -> String {
        fieldGenerator.generateField(level: level)
    }

    /// Generate all tower fields from GF(2^8) to GF(2^128).
    /// - Returns: Dictionary mapping level to generated source code
    public func generateAllFields() -> [TowerLevel: String] {
        var results = [TowerLevel: String]()
        for level in [TowerLevel.gf2_8, .gf2_16, .gf2_32, .gf2_64, .gf2_128] {
            results[level] = generateField(level: level)
        }
        return results
    }

    // MARK: - Primitive Generation

    /// Generate a primitive for a specific tower field level.
    /// - Parameters:
    ///   - templateName: The name of the template to instantiate
    ///   - level: The target tower field level
    ///   - config: Generation configuration
    /// - Returns: Generated primitive with source code and analysis
    public func generatePrimitive(
        templateName: String,
        level: TowerLevel,
        config: PrimitiveConfig = PrimitiveConfig()
    ) -> GeneratedPrimitive? {
        guard let template = templates[templateName] else {
            print("Unknown template: \(templateName)")
            return nil
        }

        // Check if template supports this field level
        guard template.supportedFieldLevels.contains(level) else {
            print("\(templateName) does not support GF(2^\(level.bitWidth))")
            return nil
        }

        // Generate code
        let typeDecl = template.generateTypeDeclaration(fieldLevel: level)
        let impl = template.generateImplementation(fieldLevel: level)

        // Analyze for optimizations
        let fullCode = typeDecl + impl
        let analysis = optimizer.analyze(
            code: fullCode,
            fieldLevel: level,
            targetArch: config.targetArch
        )

        // Apply optimizations if requested
        var finalImpl = impl
        if config.vectorize {
            let pipeline = OptimizationPipeline.cpuDefault
            var codeCopy = impl
            _ = pipeline.run(code: &codeCopy, context: OptimizationContext(
                targetArch: config.targetArch,
                fieldLevel: level,
                config: config
            ))
            finalImpl = codeCopy
        }

        return GeneratedPrimitive(
            name: templateName,
            fieldLevel: level,
            typeDeclaration: typeDecl,
            implementation: finalImpl,
            optimizationAnalysis: analysis
        )
    }

    /// Generate multiple primitives at once.
    /// - Parameters:
    ///   - templateNames: Template names to instantiate
    ///   - levels: Target field levels
    ///   - config: Generation configuration
    /// - Returns: Array of generated primitives
    public func generatePrimitives(
        templateNames: [String],
        levels: [TowerLevel],
        config: PrimitiveConfig = PrimitiveConfig()
    ) -> [GeneratedPrimitive] {
        var results = [GeneratedPrimitive]()

        for templateName in templateNames {
            for level in levels {
                if let primitive = generatePrimitive(
                    templateName: templateName,
                    level: level,
                    config: config
                ) {
                    results.append(primitive)
                }
            }
        }

        return results
    }

    /// Register a custom template.
    public func registerTemplate(_ template: PrimitiveTemplate.Type) {
        templates[template.name] = template
    }

    /// Get all registered template names.
    public var registeredTemplates: [String] {
        Array(templates.keys).sorted()
    }
}

// MARK: - Code Output

extension GeneratedPrimitiveFactory {

    /// Write generated primitives to a directory.
    /// - Parameters:
    ///   - primitives: Primitives to write
    ///   - directory: Target directory path
    ///   - includeAnalysis: Whether to include optimization analysis
    public func writePrimitives(
        _ primitives: [GeneratedPrimitive],
        to directory: String,
        includeAnalysis: Bool = true
    ) throws {
        let fileManager = FileManager.default

        // Create directory if needed
        if !fileManager.fileExists(atPath: directory) {
            try fileManager.createDirectory(atPath: directory, withIntermediateDirectories: true)
        }

        for primitive in primitives {
            let filename = "\(primitive.name)GF\(primitive.fieldLevel.rawValue).swift"
            let filepath = (directory as NSString).appendingPathComponent(filename)

            var content = primitive.fullSource

            if includeAnalysis, let analysis = primitive.optimizationAnalysis {
                content += "\n\n// MARK: - Optimization Analysis\n"
                content += "// Estimated speedup: \(String(format: "%.1f", analysis.estimatedSpeedup * 100))%\n"
                content += "// Max SIMD width: \(analysis.maxSimdWidth)\n"
                content += "// GPU recommended: \(analysis.recommendGPU)\n"
                for rec in analysis.recommendations {
                    content += "// - \(rec)\n"
                }
            }

            try content.write(toFile: filepath, atomically: true, encoding: .utf8)
            print("Generated: \(filename)")
        }
    }

    /// Generate a complete primitive module with header and exports.
    /// - Parameters:
    ///   - primitives: Primitives to include
    ///   - moduleName: Name of the module
    /// - Returns: Complete Swift module source
    public func generateModule(
        primitives: [GeneratedPrimitive],
        moduleName: String
    ) -> String {
        var output = """
        // \(moduleName).swift — Auto-generated tower-native primitives
        // Generated by zkMetal Primitive Compiler
        // DO NOT EDIT - Regenerate with PrimitiveCompiler

        import Foundation

        """

        for primitive in primitives {
            output += "// ============================================================\n"
            output += "// \(primitive.name) for GF(2^\(primitive.fieldLevel.bitWidth))\n"
            output += "// ============================================================\n\n"
            output += primitive.fullSource
            output += "\n\n"
        }

        return output
    }
}

// MARK: - Convenience Extensions

extension TowerLevel {
    /// Generate field implementation using the factory.
    public func generateFieldImplementation() -> String {
        let factory = GeneratedPrimitiveFactory()
        return factory.generateField(level: self)
    }

    /// Get all primitives available at this field level.
    public func availablePrimitives() -> [String] {
        let factory = GeneratedPrimitiveFactory()
        return factory.registeredTemplates.filter { name in
            if let template = factory.templates[name] {
                return template.supportedFieldLevels.contains(self)
            }
            return false
        }
    }
}

extension GeneratedPrimitiveFactory {
    /// Default factory instance
    public static let `default` = GeneratedPrimitiveFactory()
}
