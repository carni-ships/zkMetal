// GPUBufferPool — Thread-safe Metal buffer recycling pool
//
// Reduces MTLBuffer allocation overhead by pre-allocating and recycling buffers.
// Buffers are bucketed by power-of-2 size classes for better reuse.
// Maximum pool size cap prevents unbounded memory growth.

import Foundation
import Metal

// MARK: - GPUBufferPool

/// Thread-safe pool that recycles Metal buffers to avoid repeated allocation overhead.
///
/// Buffers are grouped into power-of-2 size classes. When a buffer is requested,
/// the pool first checks for a recycled buffer in the matching (or next larger)
/// size class. If none is available, a new buffer is allocated from the device.
///
/// Usage:
/// ```swift
/// let buf = pool.allocate(size: 65536)
/// // ... use buf ...
/// pool.release(buffer: buf)
/// ```
///
/// Or use the scoped helper that auto-releases:
/// ```swift
/// let result = pool.withBuffer(size: 65536) { buf in
///     // buf is valid only within this closure
///     return doWork(buf)
/// }
/// ```
public final class GPUBufferPool: @unchecked Sendable {

    // MARK: - Configuration

    /// Default maximum pool size in bytes (512 MB).
    public static let defaultMaxPoolBytes: Int = 512 * 1024 * 1024

    /// Minimum allocation size (256 bytes). Smaller requests are rounded up.
    public static let minimumAllocationSize: Int = 256

    /// Singleton shared instance. Created lazily on first access.
    /// Returns nil if no Metal device is available.
    public static let shared: GPUBufferPool? = {
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }
        return GPUBufferPool(device: device)
    }()

    // MARK: - Pool statistics

    /// Snapshot of pool statistics at a point in time.
    public struct Stats {
        /// Number of allocate() calls that returned a recycled buffer.
        public let hits: Int
        /// Number of allocate() calls that created a new buffer.
        public let misses: Int
        /// Hit rate as a fraction [0, 1].
        public var hitRate: Double {
            let total = hits + misses
            return total == 0 ? 0 : Double(hits) / Double(total)
        }
        /// Total bytes currently allocated (in-use + pooled).
        public let totalAllocatedBytes: Int
        /// Bytes currently in use by callers (not returned to pool).
        public let inUseBytes: Int
        /// Bytes sitting idle in the pool, available for reuse.
        public let pooledBytes: Int
        /// Peak in-use bytes observed.
        public let peakInUseBytes: Int
        /// Number of buffers currently in use.
        public let inUseCount: Int
        /// Number of buffers sitting in the pool.
        public let pooledCount: Int
    }

    // MARK: - Private state

    private let device: MTLDevice
    private let maxPoolBytes: Int
    private let lock = NSLock()

    /// Pooled (idle) buffers keyed by size class (power-of-2 exponent).
    /// Each bucket is a stack (LIFO) for cache locality.
    private var buckets: [Int: [MTLBuffer]] = [:]

    /// Track which buffers are currently in use (by their unmanaged pointer identity).
    private var inUseSet: Set<ObjectIdentifier> = []

    // Stats counters
    private var _hits: Int = 0
    private var _misses: Int = 0
    private var _totalAllocatedBytes: Int = 0
    private var _inUseBytes: Int = 0
    private var _pooledBytes: Int = 0
    private var _peakInUseBytes: Int = 0
    private var _inUseCount: Int = 0
    private var _pooledCount: Int = 0

    // MARK: - Init

    /// Create a buffer pool for the given Metal device.
    ///
    /// - Parameters:
    ///   - device: The Metal device to allocate buffers from.
    ///   - maxPoolBytes: Maximum bytes to keep in the idle pool. Defaults to 512 MB.
    public init(device: MTLDevice, maxPoolBytes: Int = GPUBufferPool.defaultMaxPoolBytes) {
        self.device = device
        self.maxPoolBytes = maxPoolBytes
    }

    // MARK: - Public API

    /// Allocate a buffer of at least `size` bytes.
    ///
    /// Returns a recycled buffer from the pool if one is available in the
    /// matching size class; otherwise allocates a new buffer from the device.
    ///
    /// - Parameters:
    ///   - size: Minimum buffer size in bytes.
    ///   - options: Metal resource options. Defaults to `.storageModeShared`.
    /// - Returns: An MTLBuffer of at least `size` bytes.
    public func allocate(size: Int, options: MTLResourceOptions = .storageModeShared) -> MTLBuffer? {
        let bucketSize = Self.roundUpToPowerOf2(Swift.max(size, Self.minimumAllocationSize))
        let sizeClass = Self.sizeClassExponent(bucketSize)

        lock.lock()
        defer { lock.unlock() }

        // Try to find a recycled buffer in this size class
        if var stack = buckets[sizeClass], !stack.isEmpty {
            let buffer = stack.removeLast()
            buckets[sizeClass] = stack

            _hits += 1
            _pooledBytes -= buffer.length
            _pooledCount -= 1
            _inUseBytes += buffer.length
            _inUseCount += 1
            _peakInUseBytes = Swift.max(_peakInUseBytes, _inUseBytes)

            inUseSet.insert(ObjectIdentifier(buffer))
            return buffer
        }

        // Cache miss: allocate a new buffer
        guard let buffer = device.makeBuffer(length: bucketSize, options: options) else {
            return nil
        }

        _misses += 1
        _totalAllocatedBytes += bucketSize
        _inUseBytes += bucketSize
        _inUseCount += 1
        _peakInUseBytes = Swift.max(_peakInUseBytes, _inUseBytes)

        inUseSet.insert(ObjectIdentifier(buffer))
        return buffer
    }

    /// Return a buffer to the pool for future reuse.
    ///
    /// The buffer must have been obtained from this pool via `allocate()`.
    /// If the pool is at capacity, the buffer is discarded instead.
    ///
    /// - Parameter buffer: The buffer to return to the pool.
    public func release(buffer: MTLBuffer) {
        let sizeClass = Self.sizeClassExponent(buffer.length)

        lock.lock()
        defer { lock.unlock() }

        let id = ObjectIdentifier(buffer)
        guard inUseSet.remove(id) != nil else {
            // Buffer was not allocated by this pool or already released; ignore.
            return
        }

        _inUseBytes -= buffer.length
        _inUseCount -= 1

        // Check if adding this buffer would exceed pool cap
        if _pooledBytes + buffer.length > maxPoolBytes {
            // Discard the buffer (let ARC deallocate it)
            _totalAllocatedBytes -= buffer.length
            return
        }

        // Return to the appropriate bucket
        if buckets[sizeClass] == nil {
            buckets[sizeClass] = []
        }
        buckets[sizeClass]!.append(buffer)
        _pooledBytes += buffer.length
        _pooledCount += 1
    }

    /// Scoped allocation that auto-releases the buffer when the closure returns.
    ///
    /// - Parameters:
    ///   - size: Minimum buffer size in bytes.
    ///   - options: Metal resource options. Defaults to `.storageModeShared`.
    ///   - body: Closure that receives the allocated buffer and returns a result.
    /// - Returns: The value returned by `body`, or nil if allocation failed.
    @discardableResult
    public func withBuffer<T>(
        size: Int,
        options: MTLResourceOptions = .storageModeShared,
        body: (MTLBuffer) throws -> T
    ) rethrows -> T? {
        guard let buffer = allocate(size: size, options: options) else {
            return nil
        }
        defer { release(buffer: buffer) }
        return try body(buffer)
    }

    /// Release all idle buffers back to the system.
    ///
    /// Buffers currently in use are unaffected. Only pooled (idle) buffers
    /// are discarded. This is useful to reduce memory pressure.
    public func trim() {
        lock.lock()
        defer { lock.unlock() }

        _totalAllocatedBytes -= _pooledBytes
        _pooledBytes = 0
        _pooledCount = 0
        buckets.removeAll()
    }

    /// Trim buffers in the pool that have not been used recently,
    /// keeping at most `keepBytes` of idle capacity.
    ///
    /// - Parameter keepBytes: Maximum idle bytes to retain. Defaults to 0 (trim all).
    public func trim(keepBytes: Int) {
        lock.lock()
        defer { lock.unlock() }

        if _pooledBytes <= keepBytes { return }

        // Remove from largest buckets first (they free the most memory)
        let sortedKeys = buckets.keys.sorted(by: >)
        for key in sortedKeys {
            guard _pooledBytes > keepBytes else { break }
            guard var stack = buckets[key], !stack.isEmpty else { continue }

            while !stack.isEmpty && _pooledBytes > keepBytes {
                let buf = stack.removeLast()
                _pooledBytes -= buf.length
                _pooledCount -= 1
                _totalAllocatedBytes -= buf.length
            }

            if stack.isEmpty {
                buckets.removeValue(forKey: key)
            } else {
                buckets[key] = stack
            }
        }
    }

    /// Current pool statistics snapshot.
    public var stats: Stats {
        lock.lock()
        defer { lock.unlock() }
        return Stats(
            hits: _hits,
            misses: _misses,
            totalAllocatedBytes: _totalAllocatedBytes,
            inUseBytes: _inUseBytes,
            pooledBytes: _pooledBytes,
            peakInUseBytes: _peakInUseBytes,
            inUseCount: _inUseCount,
            pooledCount: _pooledCount
        )
    }

    /// Reset statistics counters (does not affect pooled buffers).
    public func resetStats() {
        lock.lock()
        defer { lock.unlock() }
        _hits = 0
        _misses = 0
        _peakInUseBytes = _inUseBytes
    }

    // MARK: - Size class helpers

    /// Round up to the next power of 2.
    static func roundUpToPowerOf2(_ n: Int) -> Int {
        guard n > 0 else { return 1 }
        // If already a power of 2, return as-is
        if n & (n - 1) == 0 { return n }
        // Find the next power of 2
        var v = n - 1
        v |= v >> 1
        v |= v >> 2
        v |= v >> 4
        v |= v >> 8
        v |= v >> 16
        v |= v >> 32
        return v + 1
    }

    /// Compute the exponent (log2) for a power-of-2 value.
    static func sizeClassExponent(_ n: Int) -> Int {
        guard n > 0 else { return 0 }
        // n is assumed to be a power of 2
        var v = n
        var exp = 0
        while v > 1 {
            v >>= 1
            exp += 1
        }
        return exp
    }
}
