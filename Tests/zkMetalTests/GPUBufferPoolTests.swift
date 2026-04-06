// GPUBufferPoolTests — Tests for GPU buffer memory pool and scoped allocation

import Foundation
import Metal
import zkMetal

func runGPUBufferPoolTests() {
    suite("GPUBufferPool")

    guard let device = MTLCreateSystemDefaultDevice() else {
        print("  [SKIP] No Metal device available")
        return
    }

    // MARK: - Size class helpers

    // Test power-of-2 rounding
    expectEqual(GPUBufferPool.roundUpToPowerOf2(1), 1, "roundUp(1)")
    expectEqual(GPUBufferPool.roundUpToPowerOf2(2), 2, "roundUp(2)")
    expectEqual(GPUBufferPool.roundUpToPowerOf2(3), 4, "roundUp(3)")
    expectEqual(GPUBufferPool.roundUpToPowerOf2(5), 8, "roundUp(5)")
    expectEqual(GPUBufferPool.roundUpToPowerOf2(1023), 1024, "roundUp(1023)")
    expectEqual(GPUBufferPool.roundUpToPowerOf2(1024), 1024, "roundUp(1024)")
    expectEqual(GPUBufferPool.roundUpToPowerOf2(1025), 2048, "roundUp(1025)")
    expectEqual(GPUBufferPool.roundUpToPowerOf2(65536), 65536, "roundUp(65536)")

    // Test size class exponent
    expectEqual(GPUBufferPool.sizeClassExponent(1), 0, "sizeClass(1)")
    expectEqual(GPUBufferPool.sizeClassExponent(256), 8, "sizeClass(256)")
    expectEqual(GPUBufferPool.sizeClassExponent(1024), 10, "sizeClass(1024)")
    expectEqual(GPUBufferPool.sizeClassExponent(65536), 16, "sizeClass(65536)")

    // MARK: - Basic allocate and release

    do {
        let pool = GPUBufferPool(device: device)

        // First allocation should be a miss
        guard let buf1 = pool.allocate(size: 4096) else {
            expect(false, "allocate(4096) returned nil")
            return
        }
        expect(buf1.length >= 4096, "buffer size >= requested")
        expectEqual(pool.stats.misses, 1, "first alloc is a miss")
        expectEqual(pool.stats.hits, 0, "no hits yet")
        expectEqual(pool.stats.inUseCount, 1, "1 buffer in use")

        // Release it
        pool.release(buffer: buf1)
        expectEqual(pool.stats.pooledCount, 1, "1 buffer pooled after release")
        expectEqual(pool.stats.inUseCount, 0, "0 in use after release")

        // Second allocation of same size should be a hit
        guard let buf2 = pool.allocate(size: 4096) else {
            expect(false, "second allocate(4096) returned nil")
            return
        }
        expectEqual(pool.stats.hits, 1, "second alloc is a hit")
        expectEqual(pool.stats.pooledCount, 0, "pool is empty after reuse")
        expect(buf2.length >= 4096, "recycled buffer size >= requested")

        pool.release(buffer: buf2)
    }

    // MARK: - Size bucketing

    do {
        let pool = GPUBufferPool(device: device)

        // Allocate a 4096-byte buffer (which rounds up to 4096)
        guard let buf = pool.allocate(size: 3000) else {
            expect(false, "allocate(3000) returned nil")
            return
        }
        // Should be rounded up to next power of 2
        expectEqual(buf.length, 4096, "3000 rounds up to 4096")
        pool.release(buffer: buf)

        // Requesting 3500 should reuse the 4096 buffer
        guard let buf2 = pool.allocate(size: 3500) else {
            expect(false, "allocate(3500) returned nil")
            return
        }
        expectEqual(pool.stats.hits, 1, "3500 reuses 4096 bucket")
        expectEqual(buf2.length, 4096, "reused buffer is 4096")
        pool.release(buffer: buf2)
    }

    // MARK: - Different size classes don't interfere

    do {
        let pool = GPUBufferPool(device: device)

        guard let small = pool.allocate(size: 256) else {
            expect(false, "allocate(256) returned nil")
            return
        }
        guard let large = pool.allocate(size: 65536) else {
            expect(false, "allocate(65536) returned nil")
            return
        }
        pool.release(buffer: small)
        pool.release(buffer: large)
        expectEqual(pool.stats.pooledCount, 2, "two pooled buffers")

        // Requesting 256 should not return the 65536 buffer
        guard let reused = pool.allocate(size: 256) else {
            expect(false, "re-allocate(256) returned nil")
            return
        }
        expectEqual(reused.length, 256, "got 256 buffer, not 65536")
        pool.release(buffer: reused)
    }

    // MARK: - withBuffer scoped allocation

    do {
        let pool = GPUBufferPool(device: device)

        let result = pool.withBuffer(size: 8192) { buf -> Int in
            expect(buf.length >= 8192, "scoped buffer has correct size")
            expectEqual(pool.stats.inUseCount, 1, "buffer in use during closure")
            return 42
        }
        expectEqual(result, 42, "withBuffer returns closure result")
        expectEqual(pool.stats.inUseCount, 0, "buffer released after closure")
        expectEqual(pool.stats.pooledCount, 1, "buffer returned to pool")
    }

    // MARK: - Trim

    do {
        let pool = GPUBufferPool(device: device)

        // Fill the pool with some buffers
        var buffers: [MTLBuffer] = []
        for _ in 0..<5 {
            if let buf = pool.allocate(size: 4096) {
                buffers.append(buf)
            }
        }
        for buf in buffers {
            pool.release(buffer: buf)
        }
        expectEqual(pool.stats.pooledCount, 5, "5 buffers pooled")
        expect(pool.stats.pooledBytes > 0, "pooled bytes > 0")

        // Trim all
        pool.trim()
        expectEqual(pool.stats.pooledCount, 0, "pool empty after trim")
        expectEqual(pool.stats.pooledBytes, 0, "0 pooled bytes after trim")
    }

    // MARK: - Trim with keepBytes

    do {
        let pool = GPUBufferPool(device: device)

        var buffers: [MTLBuffer] = []
        // Allocate 4 x 4096 = 16384 bytes
        for _ in 0..<4 {
            if let buf = pool.allocate(size: 4096) {
                buffers.append(buf)
            }
        }
        for buf in buffers {
            pool.release(buffer: buf)
        }
        let pooledBefore = pool.stats.pooledBytes
        expect(pooledBefore == 4 * 4096, "16384 bytes pooled")

        // Trim to keep 8192
        pool.trim(keepBytes: 8192)
        expect(pool.stats.pooledBytes <= 8192, "pooled bytes <= 8192 after trim")
        expect(pool.stats.pooledCount >= 1, "at least 1 buffer kept")
    }

    // MARK: - Pool cap enforcement

    do {
        // Create a pool with a very small cap (16 KB)
        let pool = GPUBufferPool(device: device, maxPoolBytes: 16384)

        // Allocate and release 64 KB buffer
        if let bigBuf = pool.allocate(size: 65536) {
            pool.release(buffer: bigBuf)
            // The buffer should be discarded (exceeds pool cap)
            expectEqual(pool.stats.pooledCount, 0, "big buffer discarded when exceeding cap")
        }

        // Small buffers should still be pooled
        if let smallBuf = pool.allocate(size: 4096) {
            pool.release(buffer: smallBuf)
            expectEqual(pool.stats.pooledCount, 1, "small buffer fits in capped pool")
        }
    }

    // MARK: - Stats

    do {
        let pool = GPUBufferPool(device: device)

        expectEqual(pool.stats.hits, 0, "initial hits = 0")
        expectEqual(pool.stats.misses, 0, "initial misses = 0")
        expect(pool.stats.hitRate == 0, "initial hit rate = 0")

        guard let buf = pool.allocate(size: 1024) else {
            expect(false, "allocation failed")
            return
        }
        expect(pool.stats.peakInUseBytes >= 1024, "peak tracks allocation")
        pool.release(buffer: buf)

        _ = pool.allocate(size: 1024) // hit
        expectEqual(pool.stats.hits, 1, "1 hit")
        expectEqual(pool.stats.misses, 1, "1 miss")
        expect(pool.stats.hitRate == 0.5, "50% hit rate")

        pool.resetStats()
        expectEqual(pool.stats.hits, 0, "hits reset")
        expectEqual(pool.stats.misses, 0, "misses reset")
    }

    // MARK: - Double release is a no-op

    do {
        let pool = GPUBufferPool(device: device)

        if let buf = pool.allocate(size: 4096) {
            pool.release(buffer: buf)
            let countBefore = pool.stats.pooledCount
            pool.release(buffer: buf) // should be ignored
            expectEqual(pool.stats.pooledCount, countBefore, "double release is no-op")
        }
    }

    // MARK: - ScopedGPUContext

    suite("ScopedGPUContext")

    // Basic scoped allocation
    do {
        let pool = GPUBufferPool(device: device)
        let ctx = ScopedGPUContext(pool: pool)

        guard let buf1 = ctx.allocate(size: 4096) else {
            expect(false, "scoped allocate failed")
            return
        }
        guard let buf2 = ctx.allocate(size: 8192) else {
            expect(false, "scoped allocate 2 failed")
            return
        }
        _ = buf1; _ = buf2

        expectEqual(ctx.bufferCount, 2, "2 buffers owned")
        expect(ctx.currentBytes >= 4096 + 8192, "current bytes tracks both")
        expect(ctx.peakBytes >= 4096 + 8192, "peak bytes tracks both")
        expectEqual(ctx.allocationCount, 2, "2 allocations made")

        // Release all
        ctx.releaseAll()
        expectEqual(ctx.bufferCount, 0, "0 buffers after releaseAll")
        expectEqual(ctx.currentBytes, 0, "0 bytes after releaseAll")
        expect(ctx.peakBytes >= 4096 + 8192, "peak preserved after releaseAll")
        expectEqual(pool.stats.pooledCount, 2, "buffers returned to pool")
    }

    // Early release of individual buffer
    do {
        let pool = GPUBufferPool(device: device)
        let ctx = ScopedGPUContext(pool: pool)

        guard let buf1 = ctx.allocate(size: 4096) else {
            expect(false, "scoped allocate failed")
            return
        }
        guard let _ = ctx.allocate(size: 8192) else {
            expect(false, "scoped allocate 2 failed")
            return
        }

        ctx.release(buffer: buf1)
        expectEqual(ctx.bufferCount, 1, "1 buffer after early release")
        expect(pool.stats.pooledCount >= 1, "early-released buffer returned to pool")
    }

    // Auto-release on dealloc
    do {
        let pool = GPUBufferPool(device: device)
        do {
            let ctx = ScopedGPUContext(pool: pool)
            _ = ctx.allocate(size: 4096)
            _ = ctx.allocate(size: 8192)
            // ctx goes out of scope here
        }
        // Buffers should be returned to pool
        expectEqual(pool.stats.pooledCount, 2, "buffers auto-released on dealloc")
    }

    // Scoped withBuffer
    do {
        let pool = GPUBufferPool(device: device)
        let ctx = ScopedGPUContext(pool: pool)

        let val = ctx.withBuffer(size: 4096) { buf -> Int in
            expectEqual(ctx.bufferCount, 1, "1 buffer during withBuffer")
            return 99
        }
        expectEqual(val, 99, "withBuffer returns closure result")
        expectEqual(ctx.bufferCount, 0, "buffer released after withBuffer")
    }

    // Watermark tracking
    do {
        let pool = GPUBufferPool(device: device)
        let ctx = ScopedGPUContext(pool: pool)

        let buf1 = ctx.allocate(size: 4096) // 4096 in use
        let buf2 = ctx.allocate(size: 8192) // 4096+8192 = 12288 in use (peak)
        if let b = buf1 { ctx.release(buffer: b) } // 8192 in use
        let _ = ctx.allocate(size: 1024) // 8192+1024 = 9216 in use
        _ = buf2

        // Peak should be from when buf1+buf2 were both alive
        // Actual peak depends on rounded sizes: 4096+8192 = 12288
        expect(ctx.peakBytes >= 12288, "watermark captures peak: \(ctx.peakBytes)")
        expectEqual(ctx.allocationCount, 3, "3 total allocations")
    }

    // Concurrent allocation stress test
    do {
        let pool = GPUBufferPool(device: device)
        let group = DispatchGroup()
        let iterations = 100

        for _ in 0..<iterations {
            group.enter()
            DispatchQueue.global().async {
                if let buf = pool.allocate(size: 1024) {
                    // Simulate brief use
                    pool.release(buffer: buf)
                }
                group.leave()
            }
        }

        group.wait()
        let s = pool.stats
        expectEqual(s.hits + s.misses, iterations, "all \(iterations) allocations accounted for")
        expectEqual(s.inUseCount, 0, "no buffers in use after concurrent test")
        expect(s.hits > 0, "some hits in concurrent test (got \(s.hits))")
    }

    print("  GPUBufferPool: \(GPUBufferPool.roundUpToPowerOf2(100)) == 128, " +
          "\(GPUBufferPool.sizeClassExponent(128)) == 7")
}
