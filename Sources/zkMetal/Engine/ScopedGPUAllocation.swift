// ScopedGPUAllocation — RAII-style scoped buffer management for proving operations
//
// Manages all Metal buffers for a single proving operation. When the context
// is deallocated (or explicitly released), all buffers are returned to the pool.
// Tracks watermark (peak memory usage) for profiling.

import Foundation
import Metal

// MARK: - ScopedGPUContext

/// Manages all Metal buffers for a single proving operation.
///
/// All buffers allocated through this context are automatically returned
/// to the pool when the context is deallocated or `releaseAll()` is called.
///
/// Usage:
/// ```swift
/// let ctx = ScopedGPUContext(pool: pool)
/// let a = ctx.allocate(size: 4096)!
/// let b = ctx.allocate(size: 8192)!
/// // ... use a, b ...
/// // All buffers released when ctx goes out of scope
/// ```
public final class ScopedGPUContext: @unchecked Sendable {

    private let pool: GPUBufferPool
    private let lock = NSLock()

    /// All buffers currently owned by this context.
    private var ownedBuffers: [MTLBuffer] = []

    /// Running total of bytes currently allocated through this context.
    private var _currentBytes: Int = 0

    /// Peak bytes observed (watermark).
    private var _peakBytes: Int = 0

    /// Total number of allocations made through this context.
    private var _allocationCount: Int = 0

    /// Create a scoped context backed by the given buffer pool.
    ///
    /// - Parameter pool: The buffer pool to allocate from and return to.
    public init(pool: GPUBufferPool) {
        self.pool = pool
    }

    /// Convenience initializer using the shared pool.
    /// Returns nil if no Metal device is available.
    public convenience init?() {
        guard let shared = GPUBufferPool.shared else { return nil }
        self.init(pool: shared)
    }

    deinit {
        releaseAll()
    }

    // MARK: - Allocation

    /// Allocate a buffer of at least `size` bytes, tracked by this context.
    ///
    /// The buffer is automatically released when this context is deallocated.
    ///
    /// - Parameters:
    ///   - size: Minimum buffer size in bytes.
    ///   - options: Metal resource options. Defaults to `.storageModeShared`.
    /// - Returns: An MTLBuffer, or nil if allocation failed.
    public func allocate(size: Int, options: MTLResourceOptions = .storageModeShared) -> MTLBuffer? {
        guard let buffer = pool.allocate(size: size, options: options) else {
            return nil
        }

        lock.lock()
        ownedBuffers.append(buffer)
        _currentBytes += buffer.length
        _allocationCount += 1
        _peakBytes = Swift.max(_peakBytes, _currentBytes)
        lock.unlock()

        return buffer
    }

    /// Release a specific buffer back to the pool early.
    ///
    /// If the buffer was not allocated through this context, this is a no-op.
    ///
    /// - Parameter buffer: The buffer to release.
    public func release(buffer: MTLBuffer) {
        lock.lock()
        if let idx = ownedBuffers.firstIndex(where: { $0 === buffer }) {
            ownedBuffers.remove(at: idx)
            _currentBytes -= buffer.length
            lock.unlock()
            pool.release(buffer: buffer)
        } else {
            lock.unlock()
        }
    }

    /// Release all buffers back to the pool.
    public func releaseAll() {
        lock.lock()
        let buffersToRelease = ownedBuffers
        ownedBuffers.removeAll()
        _currentBytes = 0
        lock.unlock()

        for buf in buffersToRelease {
            pool.release(buffer: buf)
        }
    }

    // MARK: - Scoped allocation

    /// Allocate a buffer for the duration of the closure, then release it.
    ///
    /// - Parameters:
    ///   - size: Minimum buffer size in bytes.
    ///   - options: Metal resource options.
    ///   - body: Closure that uses the buffer.
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

    // MARK: - Statistics

    /// Bytes currently allocated through this context.
    public var currentBytes: Int {
        lock.lock()
        defer { lock.unlock() }
        return _currentBytes
    }

    /// Peak bytes observed (watermark) since creation.
    public var peakBytes: Int {
        lock.lock()
        defer { lock.unlock() }
        return _peakBytes
    }

    /// Number of buffers currently owned by this context.
    public var bufferCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return ownedBuffers.count
    }

    /// Total allocations made through this context (including released ones).
    public var allocationCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return _allocationCount
    }
}
