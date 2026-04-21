// MerkleTreeCache — GPU buffer caching for repeated Merkle tree builds
//
// Problem: For repeated proofs with the same LDE domain, trees are rebuilt
// from scratch even though the tree structure is identical.
//
// Solution: Cache pre-allocated GPU buffers and intermediate tree nodes by domain size.
// This eliminates repeated buffer allocation and reduces GPU memory fragmentation.
//
// Cache strategy:
//   - Key by (evalLen, numColumns) - tree structure is deterministic based on these
//   - Cache internal node buffers (tree nodes excluding leaves, which vary per proof)
//   - Pre-warm cache for common domain sizes (2^18, 2^20)
//
// Thread safety: All cache access is synchronized via NSLock.

import Foundation
import Metal

// MARK: - Cache Key

/// Cache key for Merkle tree buffers.
/// Combines evalLen and numColumns since tree structure is deterministic based on these.
public struct MerkleTreeCacheKey: Hashable {
    /// Evaluation length (number of leaves = evalLen / 8 for Poseidon2-M31)
    public let evalLen: Int

    /// Number of columns being committed
    public let numColumns: Int

    /// Log of evaluation length for faster lookups
    public var logEvalLen: Int { Int(log2(Double(evalLen))) }

    public init(evalLen: Int, numColumns: Int) {
        precondition(evalLen > 0 && (evalLen & (evalLen - 1)) == 0, "evalLen must be power of 2")
        precondition(numColumns > 0)
        self.evalLen = evalLen
        self.numColumns = numColumns
    }

    /// Hash combination for efficient dictionary lookups
    public func hash(into hasher: inout Hasher) {
        hasher.combine(evalLen)
        hasher.combine(numColumns)
    }
}

// MARK: - Cached Tree State

/// Cached GPU buffers for a Merkle tree structure.
/// These buffers are reused across proofs with the same domain size.
public final class CachedTreeState {
    /// Device this cache entry belongs to
    public let device: MTLDevice

    /// Cache key this entry was created for
    public let key: MerkleTreeCacheKey

    /// Pre-allocated buffer for leaf hashes (8 M31 per leaf).
    /// Sized for max(evalLen) across all cached entries.
    public let leafHashBuffer: MTLBuffer

    /// Pre-allocated buffer for internal tree nodes.
    /// Size: evalLen - numLeaves (internal nodes)
    public let internalNodesBuffer: MTLBuffer

    /// Buffer size for leaf hashes in elements (M31 count)
    public let leafBufferElements: Int

    /// Buffer size for internal nodes in elements (M31 count)
    public let internalBufferElements: Int

    /// Number of leaves
    public let numLeaves: Int

    /// Tree depth (log2 of number of leaves)
    public let depth: Int

    /// Last access timestamp for LRU eviction
    public var lastAccessTime: CFAbsoluteTime

    /// Number of times this cache entry was reused
    public var hitCount: Int = 0

    init(device: MTLDevice, key: MerkleTreeCacheKey) throws {
        self.device = device
        self.key = key
        self.lastAccessTime = CFAbsoluteTimeGetCurrent()

        let nodeSize = Poseidon2M31Engine.nodeSize  // 8 M31 per node
        let numLeaves = key.evalLen / nodeSize
        self.numLeaves = numLeaves
        self.depth = numLeaves.trailingZeroBitCount

        // Total tree nodes = 2 * numLeaves - 1
        // Leaves take first numLeaves * nodeSize elements
        // Internal nodes take (numLeaves - 1) * nodeSize elements
        let totalNodes = 2 * numLeaves - 1
        let internalNodes = totalNodes - numLeaves
        self.leafBufferElements = numLeaves * nodeSize
        self.internalBufferElements = internalNodes * nodeSize

        // Allocate leaf hash buffer
        let leafBytes = leafBufferElements * MemoryLayout<UInt32>.stride
        guard let leafBuf = device.makeBuffer(length: leafBytes, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate cached leaf buffer")
        }
        self.leafHashBuffer = leafBuf

        // Allocate internal nodes buffer
        let internalBytes = internalBufferElements * MemoryLayout<UInt32>.stride
        guard let internalBuf = device.makeBuffer(length: internalBytes, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate cached internal buffer")
        }
        self.internalNodesBuffer = internalBuf
    }

    /// Update access time and hit count
    func recordAccess() {
        lastAccessTime = CFAbsoluteTimeGetCurrent()
        hitCount += 1
    }

    /// Estimated memory usage in bytes
    var estimatedMemoryBytes: Int {
        let stride = MemoryLayout<UInt32>.stride
        return leafBufferElements * stride + internalBufferElements * stride
    }
}

// MARK: - Merkle Tree Cache

/// Thread-safe cache for Merkle tree GPU buffers.
///
/// For repeated proofs with the same LDE domain, this cache:
///   1. Pre-allocates GPU buffers once
///   2. Reuses buffers across proofs
///   3. Tracks hit rate and memory usage
///
/// Usage:
///   let cache = MerkleTreeCache(device: device)
///   let state = try cache.getOrCreate(evalLen: 65536, numColumns: 180)
///   // Use cached buffers for tree building
public final class MerkleTreeCache {
    /// Maximum number of cache entries
    public static let maxCacheEntries = 8

    /// Maximum total memory (bytes) for cached entries
    public static let maxCacheMemoryBytes = 256 * 1024 * 1024  // 256 MB

    /// Device this cache operates on
    public let device: MTLDevice

    /// Cache entries keyed by (evalLen, numColumns)
    private var entries: [MerkleTreeCacheKey: CachedTreeState] = [:]

    /// Lock for thread-safe access
    private let lock = NSLock()

    /// Total cached memory
    private var totalMemoryBytes: Int = 0

    /// Statistics
    public private(set) var hitCount: Int = 0
    public private(set) var missCount: Int = 0

    /// Common domain sizes to pre-warm (2^18, 2^20)
    public static let commonDomainSizes: [(evalLen: Int, numColumns: Int)] = [
        (1 << 18, 180),  // EVM trace: 2^18 eval
        (1 << 20, 180),  // Large trace: 2^20 eval
        (1 << 18, 64),   // Small trace
        (1 << 20, 64),   // Medium trace
    ]

    public init(device: MTLDevice) {
        self.device = device
    }

    // MARK: - Public Interface

    /// Get or create cached tree state for the given domain.
    ///
    /// - Parameters:
    ///   - evalLen: Evaluation length (must be power of 2)
    ///   - numColumns: Number of columns
    /// - Returns: Cached tree state with pre-allocated buffers
    public func getOrCreate(evalLen: Int, numColumns: Int) throws -> CachedTreeState {
        let key = MerkleTreeCacheKey(evalLen: evalLen, numColumns: numColumns)

        lock.lock()
        defer { lock.unlock() }

        // Check cache
        if let cached = entries[key] {
            cached.recordAccess()
            hitCount += 1
            return cached
        }

        missCount += 1

        // Create new entry
        let state = try CachedTreeState(device: device, key: key)
        let entryMemory = state.estimatedMemoryBytes

        // Evict if needed
        evictIfNeeded(newEntryMemory: entryMemory)

        // Store entry
        entries[key] = state
        totalMemoryBytes += entryMemory

        return state
    }

    /// Get existing cache entry without creating new one.
    public func get(evalLen: Int, numColumns: Int) -> CachedTreeState? {
        let key = MerkleTreeCacheKey(evalLen: evalLen, numColumns: numColumns)

        lock.lock()
        defer { lock.unlock() }

        if let cached = entries[key] {
            cached.recordAccess()
            hitCount += 1
            return cached
        }

        return nil
    }

    /// Pre-warm cache for common domain sizes.
    ///
    /// Call this during initialization to avoid first-proof cold start.
    public func prewarm() {
        lock.lock()
        defer { lock.unlock() }

        for (evalLen, numColumns) in Self.commonDomainSizes {
            let key = MerkleTreeCacheKey(evalLen: evalLen, numColumns: numColumns)

            // Skip if already cached
            if entries[key] != nil { continue }

            // Evict if needed
            let estimatedMem = estimateMemory(evalLen: evalLen, numColumns: numColumns)
            evictIfNeeded(newEntryMemory: estimatedMem)

            do {
                let state = try CachedTreeState(device: device, key: key)
                entries[key] = state
                totalMemoryBytes += state.estimatedMemoryBytes
            } catch {
                // Skip prewarm failures silently
            }
        }
    }

    /// Clear all cache entries.
    public func clear() {
        lock.lock()
        defer { lock.unlock() }

        entries.removeAll()
        totalMemoryBytes = 0
        hitCount = 0
        missCount = 0
    }

    /// Remove entry for specific domain.
    public func remove(evalLen: Int, numColumns: Int) {
        let key = MerkleTreeCacheKey(evalLen: evalLen, numColumns: numColumns)

        lock.lock()
        defer { lock.unlock() }

        if let entry = entries.removeValue(forKey: key) {
            totalMemoryBytes -= entry.estimatedMemoryBytes
        }
    }

    // MARK: - Statistics

    /// Cache hit rate (0.0 to 1.0)
    public var hitRate: Double {
        let total = hitCount + missCount
        guard total > 0 else { return 0.0 }
        return Double(hitCount) / Double(total)
    }

    /// Current number of cache entries
    public var entryCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return entries.count
    }

    /// Current total cached memory in bytes
    public var memoryUsageBytes: Int {
        lock.lock()
        defer { lock.unlock() }
        return totalMemoryBytes
    }

    /// Human-readable cache statistics
    public var statsDescription: String {
        lock.lock()
        defer { lock.unlock() }

        let total = hitCount + missCount
        let rate = total > 0 ? Double(hitCount) / Double(total) * 100 : 0

        var lines = [
            "MerkleTreeCache:",
            "  Entries: \(entries.count)/\(Self.maxCacheEntries)",
            "  Memory: \(totalMemoryBytes / 1024 / 1024) MB / \(Self.maxCacheMemoryBytes / 1024 / 1024) MB",
            "  Hits: \(hitCount), Misses: \(missCount)",
            "  Hit rate: \(String(format: "%.1f", rate))%",
        ]

        // Add per-entry stats
        for (key, state) in entries {
            lines.append("    [\(key.evalLen), \(key.numColumns)] hits=\(state.hitCount) mem=\(state.estimatedMemoryBytes / 1024) KB")
        }

        return lines.joined(separator: "\n")
    }

    // MARK: - Private Helpers

    /// Evict LRU entries if cache limits exceeded.
    private func evictIfNeeded(newEntryMemory: Int) {
        // Check entry count limit
        while entries.count >= Self.maxCacheEntries {
            evictLRU()
        }

        // Check memory limit
        while totalMemoryBytes + newEntryMemory > Self.maxCacheMemoryBytes && !entries.isEmpty {
            evictLRU()
        }
    }

    /// Evict least recently used entry.
    private func evictLRU() {
        guard let lruKey = entries.min(by: { $0.value.lastAccessTime < $1.value.lastAccessTime })?.key else {
            return
        }

        if let entry = entries.removeValue(forKey: lruKey) {
            totalMemoryBytes -= entry.estimatedMemoryBytes
        }
    }

    /// Estimate memory for a domain without creating cache entry.
    private func estimateMemory(evalLen: Int, numColumns: Int) -> Int {
        let nodeSize = Poseidon2M31Engine.nodeSize
        let numLeaves = evalLen / nodeSize
        let totalNodes = 2 * numLeaves - 1
        let internalNodes = totalNodes - numLeaves
        let leafElements = numLeaves * nodeSize
        let internalElements = internalNodes * nodeSize
        let stride = MemoryLayout<UInt32>.stride
        return (leafElements + internalElements) * stride
    }
}

// MARK: - Global Cache Instance

/// Global Merkle tree cache for the default GPU device.
/// Use this for single-GPU systems.
public final class GlobalMerkleTreeCache {
    /// Shared cache instance
    private static var instance: MerkleTreeCache?

    /// Device the cache belongs to
    private static var cachedDevice: MTLDevice?

    /// Lock for lazy initialization
    private static let initLock = NSLock()

    /// Get or create global cache instance.
    public static func get() -> MerkleTreeCache? {
        initLock.lock()
        defer { initLock.unlock() }

        guard let device = MTLCreateSystemDefaultDevice() else {
            return nil
        }

        // Create new instance if device changed or doesn't exist
        if instance == nil || cachedDevice !== device {
            instance = MerkleTreeCache(device: device)
            cachedDevice = device
            instance?.prewarm()
        }

        return instance
    }

    /// Clear global cache.
    public static func clear() {
        initLock.lock()
        defer { initLock.unlock() }

        instance?.clear()
        instance = nil
        cachedDevice = nil
    }
}
