// MetalPipelineChain — Speculative command buffer chaining utility
// Eliminates CPU-GPU round-trips by encoding multiple compute passes
// into a single MTLCommandBuffer with encoder boundaries as memory barriers.
//
// Usage:
//   let chain = MetalPipelineChain(queue: commandQueue)
//   chain.addPass { enc in  /* encode NTT */ }
//   chain.addPass { enc in  /* encode pointwise mul */ }
//   chain.addPass { enc in  /* encode INTT */ }
//   try chain.execute()  // single commit + wait
//
// Between passes, the previous encoder is ended and a new one begins,
// which acts as an implicit memory barrier on Metal.

import Metal

/// Lightweight wrapper for chaining multiple GPU compute dispatches into a
/// single command buffer submission, eliminating CPU-GPU round-trip overhead.
public final class MetalPipelineChain {
    public let commandQueue: MTLCommandQueue

    private var commandBuffer: MTLCommandBuffer?
    private var currentEncoder: MTLComputeCommandEncoder?

    /// Number of compute passes encoded so far.
    public private(set) var passCount: Int = 0

    public init(queue: MTLCommandQueue) {
        self.commandQueue = queue
    }

    /// Lazily create the command buffer on first use.
    private func ensureCommandBuffer() throws -> MTLCommandBuffer {
        if let cb = commandBuffer { return cb }
        guard let cb = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        commandBuffer = cb
        return cb
    }

    /// Add a compute pass. The closure receives a compute command encoder.
    /// Each call ends the previous encoder (implicit memory barrier) and starts a new one.
    /// The encoder is ended automatically after the closure returns.
    @discardableResult
    public func addPass(_ body: (MTLComputeCommandEncoder) throws -> Void) throws -> MetalPipelineChain {
        let cb = try ensureCommandBuffer()

        // End previous encoder if any (acts as memory barrier)
        if let enc = currentEncoder {
            enc.endEncoding()
            currentEncoder = nil
        }

        let enc = cb.makeComputeCommandEncoder()!
        try body(enc)
        enc.endEncoding()
        passCount += 1
        return self
    }

    /// Add a compute pass that keeps the encoder open for the next pass.
    /// Use when multiple dispatches can share one encoder (no barrier needed between them,
    /// e.g., independent buffers). Call `barrier()` or `addPass` to close it.
    @discardableResult
    public func addContinuousPass(_ body: (MTLComputeCommandEncoder) throws -> Void) throws -> MetalPipelineChain {
        let cb = try ensureCommandBuffer()

        if currentEncoder == nil {
            currentEncoder = cb.makeComputeCommandEncoder()!
        }
        try body(currentEncoder!)
        passCount += 1
        return self
    }

    /// Insert an explicit memory barrier within a continuous encoder.
    /// Use between dispatches that have data dependencies but share an encoder.
    @discardableResult
    public func barrier() -> MetalPipelineChain {
        currentEncoder?.memoryBarrier(scope: .buffers)
        return self
    }

    /// Add a blit pass (for copies, fills, etc.).
    @discardableResult
    public func addBlitPass(_ body: (MTLBlitCommandEncoder) throws -> Void) throws -> MetalPipelineChain {
        let cb = try ensureCommandBuffer()

        if let enc = currentEncoder {
            enc.endEncoding()
            currentEncoder = nil
        }

        let blit = cb.makeBlitCommandEncoder()!
        try body(blit)
        blit.endEncoding()
        return self
    }

    /// Encode NTT operations directly into the chain's command buffer.
    /// Returns the command buffer for use with NTTEngine.encodeNTT(data:logN:cmdBuf:).
    public func getCommandBuffer() throws -> MTLCommandBuffer {
        // Close any open encoder first
        if let enc = currentEncoder {
            enc.endEncoding()
            currentEncoder = nil
        }
        return try ensureCommandBuffer()
    }

    /// Commit the command buffer and wait for GPU completion. Single round-trip.
    public func execute() throws {
        guard let cb = commandBuffer else { return }  // nothing encoded

        if let enc = currentEncoder {
            enc.endEncoding()
            currentEncoder = nil
        }

        cb.commit()
        cb.waitUntilCompleted()

        if let error = cb.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        commandBuffer = nil
        passCount = 0
    }

    /// Commit without waiting — for overlapping CPU work with GPU.
    /// Returns the command buffer for later waitUntilCompleted().
    @discardableResult
    public func commit() throws -> MTLCommandBuffer? {
        guard let cb = commandBuffer else { return nil }

        if let enc = currentEncoder {
            enc.endEncoding()
            currentEncoder = nil
        }

        cb.commit()
        let result = cb
        commandBuffer = nil
        passCount = 0
        return result
    }

    /// Check if any passes have been encoded.
    public var isEmpty: Bool { passCount == 0 && commandBuffer == nil }
}
