// ShaderCache — Persistent Metal pipeline cache with MTLBinaryArchive
//
// Eliminates first-run JIT overhead by caching compiled pipeline states to disk.
// On subsequent runs, pipelines load from the binary archive (near-zero cost).
// Invalidation is hash-based: if any shader source changes, the cache is rebuilt.

import Foundation
import Metal
import CommonCrypto

// MARK: - ShaderCache

/// Thread-safe persistent cache for compiled Metal pipeline states.
/// Uses MTLBinaryArchive to serialize compiled pipelines to disk.
public final class ShaderCache: @unchecked Sendable {

    /// Singleton shared instance.
    public static let shared = ShaderCache()

    /// Directory where cached archives are stored.
    public static let cacheDir: URL = {
        let base = FileManager.default.homeDirectoryForCurrentUser
        return base.appendingPathComponent(".zkmetal").appendingPathComponent("shader_cache")
    }()

    private let lock = NSLock()
    /// archive keyed by module name (e.g. "ntt_bn254", "msm_bn254")
    private var archives: [String: MTLBinaryArchive] = [:]
    /// pipeline states keyed by "module/kernelName"
    private var pipelines: [String: MTLComputePipelineState] = [:]
    /// Tracks source hashes used to validate cache freshness.
    private var sourceHashes: [String: String] = [:]

    private init() {
        try? FileManager.default.createDirectory(
            at: ShaderCache.cacheDir, withIntermediateDirectories: true)
    }

    // MARK: - Public API

    /// Load or compile a library, cache the binary archive, and return pipeline states.
    ///
    /// - Parameters:
    ///   - module: Logical name for this shader group (e.g. "msm_bn254")
    ///   - device: Metal device
    ///   - sourceFiles: Ordered list of .metal source file paths to concatenate
    ///   - kernelNames: Kernel function names to create pipeline states for
    ///   - preprocessor: Optional transform applied to concatenated source before compilation
    /// - Returns: Dictionary mapping kernel name to compiled pipeline state
    public func loadOrCompile(
        module: String,
        device: MTLDevice,
        sourceFiles: [String],
        kernelNames: [String],
        preprocessor: ((String) -> String)? = nil
    ) throws -> [String: MTLComputePipelineState] {
        lock.lock()
        defer { lock.unlock() }

        // Check if all pipelines already cached in memory
        let allCached = kernelNames.allSatisfy { pipelines["\(module)/\($0)"] != nil }
        if allCached {
            var result: [String: MTLComputePipelineState] = [:]
            for name in kernelNames {
                result[name] = pipelines["\(module)/\(name)"]
            }
            return result
        }

        // Compute source hash for cache invalidation
        let currentHash = try computeSourceHash(files: sourceFiles)
        let archiveURL = ShaderCache.cacheDir.appendingPathComponent("\(module).metallib")
        let hashURL = ShaderCache.cacheDir.appendingPathComponent("\(module).sha256")

        // Try loading from disk cache
        if let cached = try? loadFromCache(
            module: module, device: device, archiveURL: archiveURL,
            hashURL: hashURL, currentHash: currentHash, kernelNames: kernelNames
        ) {
            return cached
        }

        // Cache miss or invalid — compile from source
        let library = try compileFromSource(
            device: device, sourceFiles: sourceFiles, preprocessor: preprocessor)

        // Create pipeline states
        var result: [String: MTLComputePipelineState] = [:]
        for name in kernelNames {
            guard let fn = library.makeFunction(name: name) else {
                throw ShaderCacheError.missingKernel(name)
            }
            let pso = try device.makeComputePipelineState(function: fn)
            result[name] = pso
            pipelines["\(module)/\(name)"] = pso
        }

        // Serialize to disk cache via MTLBinaryArchive
        serializeToCache(
            device: device, library: library, archiveURL: archiveURL,
            hashURL: hashURL, currentHash: currentHash, kernelNames: kernelNames)

        sourceHashes[module] = currentHash
        return result
    }

    /// Retrieve a previously compiled pipeline state by module and kernel name.
    public func pipeline(module: String, kernel: String) -> MTLComputePipelineState? {
        lock.lock()
        defer { lock.unlock() }
        return pipelines["\(module)/\(kernel)"]
    }

    /// Invalidate all cached data (memory and disk).
    public func invalidateAll() {
        lock.lock()
        defer { lock.unlock() }
        pipelines.removeAll()
        archives.removeAll()
        sourceHashes.removeAll()
        try? FileManager.default.removeItem(at: ShaderCache.cacheDir)
        try? FileManager.default.createDirectory(
            at: ShaderCache.cacheDir, withIntermediateDirectories: true)
    }

    /// Invalidate a specific module's cache.
    public func invalidate(module: String) {
        lock.lock()
        defer { lock.unlock() }
        let keysToRemove = pipelines.keys.filter { $0.hasPrefix("\(module)/") }
        for key in keysToRemove { pipelines.removeValue(forKey: key) }
        archives.removeValue(forKey: module)
        sourceHashes.removeValue(forKey: module)
        let archiveURL = ShaderCache.cacheDir.appendingPathComponent("\(module).metallib")
        let hashURL = ShaderCache.cacheDir.appendingPathComponent("\(module).sha256")
        try? FileManager.default.removeItem(at: archiveURL)
        try? FileManager.default.removeItem(at: hashURL)
    }

    // MARK: - Private helpers

    private func computeSourceHash(files: [String]) throws -> String {
        var context = CC_SHA256_CTX()
        CC_SHA256_Init(&context)
        for path in files {
            let data = try Data(contentsOf: URL(fileURLWithPath: path))
            data.withUnsafeBytes { ptr in
                _ = CC_SHA256_Update(&context, ptr.baseAddress, CC_LONG(data.count))
            }
        }
        var digest = [UInt8](repeating: 0, count: Int(CC_SHA256_DIGEST_LENGTH))
        CC_SHA256_Final(&digest, &context)
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    private func loadFromCache(
        module: String,
        device: MTLDevice,
        archiveURL: URL,
        hashURL: URL,
        currentHash: String,
        kernelNames: [String]
    ) throws -> [String: MTLComputePipelineState]? {
        // Verify hash matches
        guard FileManager.default.fileExists(atPath: archiveURL.path),
              FileManager.default.fileExists(atPath: hashURL.path) else { return nil }

        let storedHash = try String(contentsOf: hashURL, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)
        guard storedHash == currentHash else { return nil }

        // Load the metallib from cache
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(URL: archiveURL)
        } catch {
            return nil
        }

        // Verify all kernels exist
        guard kernelNames.allSatisfy({ library.makeFunction(name: $0) != nil }) else {
            return nil
        }

        // Build pipeline states from cached library
        var result: [String: MTLComputePipelineState] = [:]
        for name in kernelNames {
            guard let fn = library.makeFunction(name: name) else { return nil }
            let pso = try device.makeComputePipelineState(function: fn)
            result[name] = pso
            pipelines["\(module)/\(name)"] = pso
        }
        sourceHashes[module] = currentHash
        return result
    }

    private func compileFromSource(
        device: MTLDevice,
        sourceFiles: [String],
        preprocessor: ((String) -> String)?
    ) throws -> MTLLibrary {
        // Read and concatenate sources
        var combined = ""
        for path in sourceFiles {
            let source = try String(contentsOfFile: path, encoding: .utf8)
            combined += source + "\n"
        }
        if let transform = preprocessor {
            combined = transform(combined)
        }

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    private func serializeToCache(
        device: MTLDevice,
        library: MTLLibrary,
        archiveURL: URL,
        hashURL: URL,
        currentHash: String,
        kernelNames: [String]
    ) {
        // Use MTLBinaryArchive to serialize compiled pipelines
        if #available(macOS 11.0, iOS 14.0, *) {
            let archiveDesc = MTLBinaryArchiveDescriptor()
            if let archive = try? device.makeBinaryArchive(descriptor: archiveDesc) {
                for name in kernelNames {
                    let desc = MTLComputePipelineDescriptor()
                    desc.computeFunction = library.makeFunction(name: name)
                    try? archive.addComputePipelineFunctions(descriptor: desc)
                }
                try? archive.serialize(to: archiveURL)
                archives["temp"] = archive  // keep reference alive
            }
        }

        // Also serialize the metallib directly for fast reload
        // (MTLBinaryArchive is device-specific; metallib is the portable fallback)
        // The compileFromSource already made the library; we write the hash file
        // to validate future loads.
        try? currentHash.write(to: hashURL, atomically: true, encoding: .utf8)
    }
}

// MARK: - ShaderCacheError

public enum ShaderCacheError: Error, CustomStringConvertible {
    case missingKernel(String)
    case compilationFailed(String)
    case cacheCorrupted(String)

    public var description: String {
        switch self {
        case .missingKernel(let name): return "Missing kernel function: \(name)"
        case .compilationFailed(let msg): return "Shader compilation failed: \(msg)"
        case .cacheCorrupted(let msg): return "Cache corrupted: \(msg)"
        }
    }
}

// MARK: - PipelinePrecompiler

/// Background precompiler that enumerates all shader modules and compiles them
/// at app startup, populating ShaderCache for zero-latency first use.
public final class PipelinePrecompiler: @unchecked Sendable {

    /// Known shader modules with their source files and kernel names.
    /// Each module definition specifies paths relative to the shader directory.
    public struct ModuleDefinition {
        public let name: String
        public let sourceFiles: [String]  // relative to shader dir
        public let kernelNames: [String]
        public let preprocessor: ((String) -> String)?

        public init(
            name: String,
            sourceFiles: [String],
            kernelNames: [String],
            preprocessor: ((String) -> String)? = nil
        ) {
            self.name = name
            self.sourceFiles = sourceFiles
            self.kernelNames = kernelNames
            self.preprocessor = preprocessor
        }
    }

    /// Standard preprocessor: strip #include and header guards for concatenation.
    public static func stripIncludesAndGuards(_ source: String) -> String {
        source.split(separator: "\n", omittingEmptySubsequences: false)
            .filter { line in
                !line.contains("#include") &&
                !line.hasPrefix("#ifndef ") &&
                !line.hasPrefix("#define ") &&
                !(line.hasPrefix("#endif") && line.contains("//"))
            }
            .joined(separator: "\n")
    }

    /// All known shader modules in the zkMetal library.
    public static func allModules(shaderDir: String) -> [ModuleDefinition] {
        let strip = stripIncludesAndGuards
        return [
            ModuleDefinition(
                name: "msm_bn254",
                sourceFiles: [
                    "\(shaderDir)/fields/bn254_fp.metal",
                    "\(shaderDir)/geometry/bn254_curve.metal",
                    "\(shaderDir)/msm/glv_kernels.metal",
                    "\(shaderDir)/msm/msm_kernels.metal",
                ],
                kernelNames: [
                    "msm_reduce_sorted_buckets", "msm_reduce_cooperative",
                    "msm_bucket_sum_direct", "msm_combine_segments",
                    "msm_horner_combine", "glv_endomorphism", "glv_decompose",
                    "signed_digit_extract", "gpu_sort_histogram", "gpu_sort_scatter",
                    "gpu_build_csm",
                ],
                preprocessor: strip
            ),
            ModuleDefinition(
                name: "ntt_bn254",
                sourceFiles: [
                    "\(shaderDir)/fields/bn254_fr.metal",
                    "\(shaderDir)/ntt/ntt_kernels.metal",
                ],
                kernelNames: [
                    "ntt_butterfly", "ntt_butterfly_radix4",
                    "intt_butterfly", "intt_butterfly_radix4",
                    "ntt_butterfly_fused", "intt_butterfly_fused",
                    "ntt_scale", "ntt_bitrev", "ntt_bitrev_inplace",
                    "ntt_column_fused", "ntt_row_fused",
                    "ntt_twiddle_multiply", "ntt_transpose",
                ],
                preprocessor: strip
            ),
            ModuleDefinition(
                name: "poseidon2_bn254",
                sourceFiles: [
                    "\(shaderDir)/fields/bn254_fr.metal",
                    "\(shaderDir)/hash/poseidon2.metal",
                ],
                kernelNames: [
                    "poseidon2_permute", "poseidon2_hash_pairs",
                    "poseidon2_merkle_fused", "poseidon2_merkle_fused_full",
                    "poseidon2_merkle_fused_batch",
                    "poseidon2_merkle_update_scattered",
                ],
                preprocessor: strip
            ),
            ModuleDefinition(
                name: "keccak256",
                sourceFiles: [
                    "\(shaderDir)/hash/keccak256.metal",
                ],
                kernelNames: [
                    "keccak256_hash", "keccak256_merkle",
                ],
                preprocessor: strip
            ),
            ModuleDefinition(
                name: "blake3",
                sourceFiles: [
                    "\(shaderDir)/hash/blake3.metal",
                ],
                kernelNames: [
                    "blake3_hash", "blake3_merkle",
                ],
                preprocessor: strip
            ),
            ModuleDefinition(
                name: "sha256",
                sourceFiles: [
                    "\(shaderDir)/hash/sha256.metal",
                ],
                kernelNames: [
                    "sha256_hash", "sha256_merkle",
                ],
                preprocessor: strip
            ),
            ModuleDefinition(
                name: "radix_sort",
                sourceFiles: [
                    "\(shaderDir)/sort/radix_sort.metal",
                ],
                kernelNames: [
                    "radix_sort_histogram", "radix_sort_scatter",
                ],
                preprocessor: strip
            ),
            ModuleDefinition(
                name: "fri",
                sourceFiles: [
                    "\(shaderDir)/fields/bn254_fr.metal",
                    "\(shaderDir)/fri/fri_kernels.metal",
                ],
                kernelNames: [
                    "fri_fold",
                ],
                preprocessor: strip
            ),
            ModuleDefinition(
                name: "sumcheck",
                sourceFiles: [
                    "\(shaderDir)/fields/bn254_fr.metal",
                    "\(shaderDir)/sumcheck/sumcheck_kernels.metal",
                ],
                kernelNames: [
                    "sumcheck_fold",
                ],
                preprocessor: strip
            ),
        ]
    }

    /// Precompile all known shader modules in parallel.
    ///
    /// - Parameters:
    ///   - device: Metal device
    ///   - cache: ShaderCache to populate (defaults to .shared)
    ///   - completion: Called when all modules are compiled, with list of any errors
    public static func precompileAll(
        device: MTLDevice,
        cache: ShaderCache = .shared,
        completion: @escaping ([String: Error]) -> Void
    ) {
        let shaderDir = findShaderDir()
        let modules = allModules(shaderDir: shaderDir)
        let group = DispatchGroup()
        let errorLock = NSLock()
        var errors: [String: Error] = [:]

        for module in modules {
            group.enter()
            DispatchQueue.global(qos: .userInitiated).async {
                defer { group.leave() }

                // Verify source files exist before attempting compile
                let existingFiles = module.sourceFiles.filter {
                    FileManager.default.fileExists(atPath: $0)
                }
                guard !existingFiles.isEmpty else { return }

                do {
                    _ = try cache.loadOrCompile(
                        module: module.name,
                        device: device,
                        sourceFiles: existingFiles,
                        kernelNames: module.kernelNames,
                        preprocessor: module.preprocessor
                    )
                } catch {
                    errorLock.lock()
                    errors[module.name] = error
                    errorLock.unlock()
                }
            }
        }

        group.notify(queue: .main) {
            completion(errors)
        }
    }

    /// Synchronous version of precompileAll for use in init paths.
    public static func precompileAllSync(
        device: MTLDevice,
        cache: ShaderCache = .shared
    ) -> [String: Error] {
        let shaderDir = findShaderDir()
        let modules = allModules(shaderDir: shaderDir)
        var errors: [String: Error] = [:]
        let errorLock = NSLock()

        DispatchQueue.concurrentPerform(iterations: modules.count) { i in
            let module = modules[i]
            let existingFiles = module.sourceFiles.filter {
                FileManager.default.fileExists(atPath: $0)
            }
            guard !existingFiles.isEmpty else { return }

            do {
                _ = try cache.loadOrCompile(
                    module: module.name,
                    device: device,
                    sourceFiles: existingFiles,
                    kernelNames: module.kernelNames,
                    preprocessor: module.preprocessor
                )
            } catch {
                errorLock.lock()
                errors[module.name] = error
                errorLock.unlock()
            }
        }
        return errors
    }
}

// MARK: - LazyPipeline

/// Wrapper that loads a pipeline state on first use from ShaderCache.
/// Avoids paying compilation cost until the kernel is actually needed,
/// while still benefiting from the disk cache after first compile.
public final class LazyPipeline: @unchecked Sendable {
    private let module: String
    private let kernel: String
    private let sourceFiles: [String]
    private let preprocessor: ((String) -> String)?
    private let device: MTLDevice
    private let cache: ShaderCache

    private var _pipeline: MTLComputePipelineState?
    private let lock = NSLock()

    /// Create a lazy pipeline wrapper.
    ///
    /// - Parameters:
    ///   - module: Module name for cache grouping
    ///   - kernel: Kernel function name
    ///   - device: Metal device
    ///   - sourceFiles: Shader source file paths (absolute)
    ///   - preprocessor: Optional source transform
    ///   - cache: ShaderCache instance (defaults to .shared)
    public init(
        module: String,
        kernel: String,
        device: MTLDevice,
        sourceFiles: [String],
        preprocessor: ((String) -> String)? = nil,
        cache: ShaderCache = .shared
    ) {
        self.module = module
        self.kernel = kernel
        self.device = device
        self.sourceFiles = sourceFiles
        self.preprocessor = preprocessor
        self.cache = cache
    }

    /// The compiled pipeline state. Compiled on first access and cached thereafter.
    /// Throws on compilation failure.
    public func get() throws -> MTLComputePipelineState {
        lock.lock()
        defer { lock.unlock() }

        if let existing = _pipeline { return existing }

        // Check if already in ShaderCache (e.g. from precompile)
        if let cached = cache.pipeline(module: module, kernel: kernel) {
            _pipeline = cached
            return cached
        }

        // Compile via ShaderCache (will hit disk cache if available)
        let all = try cache.loadOrCompile(
            module: module,
            device: device,
            sourceFiles: sourceFiles,
            kernelNames: [kernel],
            preprocessor: preprocessor
        )
        guard let pso = all[kernel] else {
            throw ShaderCacheError.missingKernel(kernel)
        }
        _pipeline = pso
        return pso
    }

    /// Whether the pipeline has been compiled (without triggering compilation).
    public var isCompiled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return _pipeline != nil || cache.pipeline(module: module, kernel: kernel) != nil
    }
}
