// Runtime Compiler — JIT compilation of constraint systems to Metal GPU pipelines
// Caches compiled pipelines by constraint system hash for amortized compilation cost.
// Uses MTLDevice.makeLibrary(source:options:) for runtime MSL compilation.

import Foundation
import Metal

/// Cached compiled pipeline entry
private struct CachedPipeline {
    let compiled: CompiledConstraints
    let lastAccess: CFAbsoluteTime
}

/// Runtime compiler that JIT-compiles constraint systems to Metal compute pipelines.
/// Thread-safe pipeline cache keyed by constraint system structural hash.
public class RuntimeCompiler {
    public static let shared = RuntimeCompiler()

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    private let frSource: String
    private let codegen = MetalCodegen()

    /// Pipeline cache: stableHash -> compiled pipeline
    private var cache: [Int: CachedPipeline] = [:]
    private let lock = NSLock()

    /// Maximum number of cached pipelines before eviction
    public var maxCacheSize: Int = 64

    /// Statistics
    public private(set) var cacheHits: Int = 0
    public private(set) var cacheMisses: Int = 0
    public private(set) var totalCompileTimeMs: Double = 0

    public init?() {
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }
        self.device = device
        guard let queue = device.makeCommandQueue() else { return nil }
        self.commandQueue = queue

        // Load field arithmetic source
        let shaderDir = findShaderDir()
        guard let rawFr = try? String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8) else {
            return nil
        }
        self.frSource = rawFr
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")
    }

    // For testing with a specific device
    public init(device: MTLDevice, commandQueue: MTLCommandQueue, frSource: String) {
        self.device = device
        self.commandQueue = commandQueue
        self.frSource = frSource
    }

    // MARK: - Compile (cached)

    /// Compile a constraint system, returning a cached pipeline if available.
    /// Thread-safe: multiple threads can compile different systems concurrently.
    public func compile(_ system: ConstraintSystem, includeQuotient: Bool = false) throws -> CompiledConstraints {
        let hash = system.stableHash

        lock.lock()
        if let entry = cache[hash] {
            // Update access time
            cache[hash] = CachedPipeline(compiled: entry.compiled, lastAccess: CFAbsoluteTimeGetCurrent())
            cacheHits += 1
            lock.unlock()
            return entry.compiled
        }
        cacheMisses += 1
        lock.unlock()

        // Compile outside the lock (Metal compilation can be slow)
        let compiled = try compileUncached(system, includeQuotient: includeQuotient)

        lock.lock()
        // Evict LRU if cache is full
        if cache.count >= maxCacheSize {
            evictLRU()
        }
        cache[hash] = CachedPipeline(compiled: compiled, lastAccess: CFAbsoluteTimeGetCurrent())
        lock.unlock()

        return compiled
    }

    /// Compile without caching. Useful for benchmarking compilation time.
    public func compileUncached(_ system: ConstraintSystem, includeQuotient: Bool = false) throws -> CompiledConstraints {
        let t0 = CFAbsoluteTimeGetCurrent()

        // Generate Metal source with constant folding + CSE
        let evalSource = codegen.generateConstraintEval(system: system)
        let cleanEval = stripMetalHeaders(evalSource)

        var fullSource = frSource + "\n" + cleanEval

        if includeQuotient {
            let quotientSource = codegen.generateQuotientEval(system: system)
            let cleanQuotient = stripMetalHeaders(quotientSource)
            fullSource += "\n" + cleanQuotient
        }

        // JIT compile Metal library
        let options = MTLCompileOptions()
        options.fastMathEnabled = true

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: fullSource, options: options)
        } catch {
            fputs("RuntimeCompiler: Metal compilation failed.\nGenerated source:\n\(fullSource)\n", stderr)
            throw MSMError.gpuError("Metal compile error: \(error.localizedDescription)")
        }

        guard let evalFn = library.makeFunction(name: "eval_constraints") else {
            throw MSMError.missingKernel
        }
        let evalPipeline = try device.makeComputePipelineState(function: evalFn)

        var quotientPipeline: MTLComputePipelineState? = nil
        if includeQuotient {
            if let quotientFn = library.makeFunction(name: "eval_quotient") {
                quotientPipeline = try device.makeComputePipelineState(function: quotientFn)
            }
        }

        let compileTime = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0

        lock.lock()
        totalCompileTimeMs += compileTime
        lock.unlock()

        return CompiledConstraints(
            evalPipeline: evalPipeline,
            quotientPipeline: quotientPipeline,
            system: system,
            shaderSource: fullSource,
            compileTimeMs: compileTime
        )
    }

    // MARK: - Compile + Evaluate (convenience)

    /// Compile (cached) and evaluate constraints on a trace in one call.
    /// Returns output buffer: numRows * numConstraints Fr values.
    public func evaluate(system: ConstraintSystem,
                         trace: MTLBuffer,
                         numRows: Int) throws -> MTLBuffer {
        let compiled = try compile(system)
        let numConstraints = system.constraints.count
        let numCols = system.numWires
        let outputSize = numRows * numConstraints * MemoryLayout<Fr>.stride

        guard let outputBuf = device.makeBuffer(length: outputSize, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate constraint output buffer")
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(compiled.evalPipeline)
        enc.setBuffer(trace, offset: 0, index: 0)
        enc.setBuffer(outputBuf, offset: 0, index: 1)
        var cols = UInt32(numCols)
        var rows = UInt32(numRows)
        enc.setBytes(&cols, length: 4, index: 2)
        enc.setBytes(&rows, length: 4, index: 3)

        let tg = min(256, Int(compiled.evalPipeline.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: numRows, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw MSMError.gpuError("Constraint eval GPU error: \(error.localizedDescription)")
        }

        return outputBuf
    }

    /// Compile (cached) and verify that all constraints are satisfied.
    public func verify(system: ConstraintSystem,
                       trace: MTLBuffer,
                       numRows: Int) throws -> Bool {
        let outputBuf = try evaluate(system: system, trace: trace, numRows: numRows)
        let numConstraints = system.constraints.count
        let count = numRows * numConstraints
        let ptr = outputBuf.contents().bindMemory(to: Fr.self, capacity: count)
        for i in 0..<count {
            if !ptr[i].isZero { return false }
        }
        return true
    }

    // MARK: - Cache Management

    /// Clear the pipeline cache.
    public func clearCache() {
        lock.lock()
        cache.removeAll()
        cacheHits = 0
        cacheMisses = 0
        totalCompileTimeMs = 0
        lock.unlock()
    }

    /// Number of cached pipelines.
    public var cacheCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return cache.count
    }

    /// Cache hit rate (0.0 to 1.0).
    public var hitRate: Double {
        let total = cacheHits + cacheMisses
        return total > 0 ? Double(cacheHits) / Double(total) : 0.0
    }

    // MARK: - Internal

    private func stripMetalHeaders(_ source: String) -> String {
        source.split(separator: "\n")
            .filter { !$0.contains("#include") && !$0.contains("using namespace metal") }
            .joined(separator: "\n")
    }

    /// Evict the least-recently-used cache entry.
    private func evictLRU() {
        guard let oldest = cache.min(by: { $0.value.lastAccess < $1.value.lastAccess }) else { return }
        cache.removeValue(forKey: oldest.key)
    }
}
