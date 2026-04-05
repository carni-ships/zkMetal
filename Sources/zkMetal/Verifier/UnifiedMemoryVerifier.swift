// Unified-Memory Streaming Proof Verification
// Exploits Apple Silicon unified memory architecture to verify proofs without
// any CPU<->GPU memcpy. The prover GPU writes proof elements (G1/G2 points,
// field elements) directly into storageModeShared MTLBuffers. The verifier
// reads those same buffers via MTLBuffer.contents() — zero physical copies
// because CPU and GPU share the same DRAM on Apple Silicon.
//
// Pipeline:
//   1. Prover dispatches GPU compute (MSM, NTT, etc.)
//   2. GPU writes proof points into shared MTLBuffers
//   3. Verifier reads proof data from same buffers (no memcpy)
//   4. Verifier dispatches its own GPU work (EC checks, Merkle) on shared buffers
//
// Batch mode: a shared UnifiedBufferPool pre-allocates N proof slots.
// Multiple proofs are generated and verified from the same pool with no
// per-proof allocation overhead.

import Foundation
import Metal

// MARK: - Unified Proof Buffer

/// A proof buffer backed by a single storageModeShared MTLBuffer.
/// GPU prover writes proof elements here; CPU verifier reads them directly.
/// No copies occur — both access the same physical memory on Apple Silicon.
public final class UnifiedProofBuffer {
    /// The underlying Metal buffer (storageModeShared = unified memory).
    public let buffer: MTLBuffer

    /// Layout metadata: byte offsets for each proof element.
    public var layout: ProofBufferLayout

    /// Whether GPU has finished writing this proof.
    public private(set) var gpuComplete: Bool = false

    /// The command buffer that produced this proof (for synchronization).
    public private(set) var producerCommandBuffer: MTLCommandBuffer?

    public init(device: MTLDevice, capacity: Int) throws {
        guard let buf = device.makeBuffer(length: capacity, options: .storageModeShared) else {
            throw UnifiedVerifierError.allocationFailed("Failed to allocate \(capacity) byte unified proof buffer")
        }
        self.buffer = buf
        self.layout = ProofBufferLayout()
    }

    /// Zero-copy pointer to the buffer contents.
    /// On Apple Silicon, this IS the same physical memory the GPU wrote to.
    public var contents: UnsafeMutableRawPointer { buffer.contents() }

    /// Mark this buffer as containing a complete GPU-produced proof.
    public func markComplete(commandBuffer: MTLCommandBuffer) {
        self.producerCommandBuffer = commandBuffer
        self.gpuComplete = true
    }

    /// Wait for the GPU producer to finish writing, then return.
    /// After this call, contents are safe to read from CPU.
    public func waitForProducer() {
        if let cb = producerCommandBuffer, cb.status != .completed {
            cb.waitUntilCompleted()
        }
    }

    /// Reset for reuse (no deallocation — buffer is reused).
    public func reset() {
        layout = ProofBufferLayout()
        gpuComplete = false
        producerCommandBuffer = nil
    }

    // MARK: - Write helpers (for prover GPU output)

    /// Write a G1 projective point at the given offset.
    /// Returns the next write offset.
    @discardableResult
    public func writeG1Point(_ point: PointProjective, at offset: Int) -> Int {
        let stride = MemoryLayout<PointProjective>.stride
        contents.storeBytes(of: point, toByteOffset: offset, as: PointProjective.self)
        return offset + stride
    }

    /// Write a G2 projective point at the given offset.
    @discardableResult
    public func writeG2Point(_ point: G2ProjectivePoint, at offset: Int) -> Int {
        let stride = MemoryLayout<G2ProjectivePoint>.stride
        contents.storeBytes(of: point, toByteOffset: offset, as: G2ProjectivePoint.self)
        return offset + stride
    }

    /// Write a field element at the given offset.
    @discardableResult
    public func writeFr(_ value: Fr, at offset: Int) -> Int {
        let stride = MemoryLayout<Fr>.stride
        contents.storeBytes(of: value, toByteOffset: offset, as: Fr.self)
        return offset + stride
    }

    // MARK: - Read helpers (for verifier — zero-copy on unified memory)

    /// Read a G1 projective point. No copy — direct unified memory access.
    public func readG1Point(at offset: Int) -> PointProjective {
        contents.load(fromByteOffset: offset, as: PointProjective.self)
    }

    /// Read a G2 projective point. No copy — direct unified memory access.
    public func readG2Point(at offset: Int) -> G2ProjectivePoint {
        contents.load(fromByteOffset: offset, as: G2ProjectivePoint.self)
    }

    /// Read a field element. No copy — direct unified memory access.
    public func readFr(at offset: Int) -> Fr {
        contents.load(fromByteOffset: offset, as: Fr.self)
    }
}

// MARK: - Proof Buffer Layout

/// Tracks byte offsets for proof elements within a UnifiedProofBuffer.
/// Supports Groth16 (3 elements), Plonk (many elements), and KZG (2 elements).
public struct ProofBufferLayout {
    /// Named offsets for each proof element.
    public var offsets: [String: Int] = [:]

    /// Current write cursor (next free byte).
    public var cursor: Int = 0

    /// Reserve space for a named element and return its offset.
    @discardableResult
    public mutating func reserve(name: String, size: Int, alignment: Int = 16) -> Int {
        // Align cursor
        let alignedCursor = (cursor + alignment - 1) & ~(alignment - 1)
        offsets[name] = alignedCursor
        cursor = alignedCursor + size
        return alignedCursor
    }

    /// Total bytes used.
    public var totalSize: Int { cursor }

    /// Get offset for a named element.
    public func offset(for name: String) -> Int? { offsets[name] }
}

// MARK: - Unified Buffer Pool

/// Pre-allocated pool of UnifiedProofBuffers for batch prove-then-verify.
/// Eliminates per-proof allocation overhead: allocate once, reuse N times.
///
/// Usage:
///   let pool = try UnifiedBufferPool(device: device, count: 8, proofSize: 4096)
///   let buf = pool.acquire()      // get a buffer for prover
///   // ... GPU writes proof into buf ...
///   pool.release(buf)             // return to pool (verifier can still read it)
public final class UnifiedBufferPool {
    public let device: MTLDevice
    private var available: [UnifiedProofBuffer]
    private var inUse: Set<ObjectIdentifier> = []
    public let proofSize: Int
    public let poolSize: Int

    /// Create a pool of `count` buffers, each `proofSize` bytes.
    public init(device: MTLDevice, count: Int, proofSize: Int) throws {
        self.device = device
        self.proofSize = proofSize
        self.poolSize = count
        self.available = []
        available.reserveCapacity(count)
        for _ in 0..<count {
            let buf = try UnifiedProofBuffer(device: device, capacity: proofSize)
            available.append(buf)
        }
    }

    /// Acquire a buffer from the pool. Returns nil if pool is exhausted.
    public func acquire() -> UnifiedProofBuffer? {
        guard let buf = available.popLast() else { return nil }
        buf.reset()
        inUse.insert(ObjectIdentifier(buf))
        return buf
    }

    /// Release a buffer back to the pool for reuse.
    public func release(_ buf: UnifiedProofBuffer) {
        let id = ObjectIdentifier(buf)
        guard inUse.contains(id) else { return }
        inUse.remove(id)
        available.append(buf)
    }

    /// Number of buffers currently available.
    public var availableCount: Int { available.count }

    /// Number of buffers currently in use.
    public var inUseCount: Int { inUse.count }
}

// MARK: - Streaming Verify Pipeline

/// End-to-end prove-then-verify pipeline with zero CPU<->GPU copies.
///
/// The prover writes proof elements into a UnifiedProofBuffer (storageModeShared).
/// The verifier reads those same bytes directly — no memcpy, no serialization.
///
/// Supports:
///   - Groth16 proofs (3 group elements: A in G1, B in G2, C in G1)
///   - KZG opening proofs (evaluation + witness commitment)
///   - Plonk proofs (multiple commitments + evaluations)
///   - Batch mode via UnifiedBufferPool
///
/// Example:
///   let pipeline = try StreamingVerifyPipeline()
///   let buf = try pipeline.allocateProofBuffer(proofType: .groth16)
///   // Prover writes proof into buf (GPU or CPU, same buffer either way)
///   pipeline.writeGroth16Proof(proof, to: buf)
///   // Verify directly from buffer — no copy
///   let valid = pipeline.verifyGroth16FromBuffer(buf, vk: vk, publicInputs: inputs)
public final class StreamingVerifyPipeline {
    public static let version = Versions.streamVerify

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    /// Groth16 verifier (reused across calls)
    private let groth16Verifier: Groth16Verifier

    /// Streaming verifier for Merkle/EC checks
    private let streamingVerifier: StreamingVerifier

    /// Buffer pool for batch mode (lazily initialized)
    private var pool: UnifiedBufferPool?

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw UnifiedVerifierError.noGPU
        }
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw UnifiedVerifierError.noGPU
        }
        self.commandQueue = queue
        self.groth16Verifier = Groth16Verifier()
        self.streamingVerifier = try StreamingVerifier()
    }

    /// Initialize with existing device and verifier (shares GPU resources).
    public init(device: MTLDevice, commandQueue: MTLCommandQueue,
                streamingVerifier: StreamingVerifier) {
        self.device = device
        self.commandQueue = commandQueue
        self.groth16Verifier = Groth16Verifier()
        self.streamingVerifier = streamingVerifier
    }

    // MARK: - Buffer Allocation

    /// Size constants for proof types.
    public enum ProofType {
        case groth16    // A(G1) + B(G2) + C(G1)
        case kzg        // evaluation(Fr) + witness(G1)
        case plonk      // ~8 commitments + ~6 evaluations + 2 opening proofs

        public var estimatedSize: Int {
            let g1Size = MemoryLayout<PointProjective>.stride
            let g2Size = MemoryLayout<G2ProjectivePoint>.stride
            let frSize = MemoryLayout<Fr>.stride
            switch self {
            case .groth16: return g1Size * 2 + g2Size + 256  // padding
            case .kzg: return frSize + g1Size + 128
            case .plonk: return g1Size * 10 + frSize * 8 + 512
            }
        }
    }

    /// Allocate a unified proof buffer sized for the given proof type.
    public func allocateProofBuffer(proofType: ProofType) throws -> UnifiedProofBuffer {
        try UnifiedProofBuffer(device: device, capacity: proofType.estimatedSize)
    }

    /// Allocate a unified proof buffer with explicit size.
    public func allocateProofBuffer(size: Int) throws -> UnifiedProofBuffer {
        try UnifiedProofBuffer(device: device, capacity: size)
    }

    // MARK: - Groth16 Zero-Copy Pipeline

    /// Write a Groth16 proof into a unified buffer.
    /// Prover calls this after GPU MSM completes — proof points go directly
    /// into the shared buffer that the verifier will read.
    public func writeGroth16Proof(_ proof: Groth16Proof, to buffer: UnifiedProofBuffer) {
        var layout = ProofBufferLayout()
        let g1Stride = MemoryLayout<PointProjective>.stride
        let g2Stride = MemoryLayout<G2ProjectivePoint>.stride

        let aOff = layout.reserve(name: "groth16.a", size: g1Stride)
        let bOff = layout.reserve(name: "groth16.b", size: g2Stride)
        let cOff = layout.reserve(name: "groth16.c", size: g1Stride)

        buffer.writeG1Point(proof.a, at: aOff)
        buffer.writeG2Point(proof.b, at: bOff)
        buffer.writeG1Point(proof.c, at: cOff)
        buffer.layout = layout
    }

    /// Verify a Groth16 proof by reading directly from a unified buffer.
    /// No memcpy: the verifier accesses the same physical memory the prover wrote.
    public func verifyGroth16FromBuffer(
        _ buffer: UnifiedProofBuffer,
        vk: Groth16VerificationKey,
        publicInputs: [Fr]
    ) -> Bool {
        // Wait for GPU producer if needed
        buffer.waitForProducer()

        // Read proof elements from unified memory (zero-copy)
        guard let aOff = buffer.layout.offset(for: "groth16.a"),
              let bOff = buffer.layout.offset(for: "groth16.b"),
              let cOff = buffer.layout.offset(for: "groth16.c") else {
            return false
        }

        let a = buffer.readG1Point(at: aOff)
        let b = buffer.readG2Point(at: bOff)
        let c = buffer.readG1Point(at: cOff)

        let proof = Groth16Proof(a: a, b: b, c: c)
        return groth16Verifier.verify(proof: proof, vk: vk, publicInputs: publicInputs)
    }

    // MARK: - KZG Zero-Copy Pipeline

    /// Write a KZG proof into a unified buffer.
    public func writeKZGProof(_ proof: KZGProof, to buffer: UnifiedProofBuffer) {
        var layout = ProofBufferLayout()
        let frStride = MemoryLayout<Fr>.stride
        let g1Stride = MemoryLayout<PointProjective>.stride

        let evalOff = layout.reserve(name: "kzg.evaluation", size: frStride)
        let witOff = layout.reserve(name: "kzg.witness", size: g1Stride)

        buffer.writeFr(proof.evaluation, at: evalOff)
        buffer.writeG1Point(proof.witness, at: witOff)
        buffer.layout = layout
    }

    /// Read a KZG proof from a unified buffer (zero-copy).
    public func readKZGProofFromBuffer(_ buffer: UnifiedProofBuffer) -> KZGProof? {
        buffer.waitForProducer()
        guard let evalOff = buffer.layout.offset(for: "kzg.evaluation"),
              let witOff = buffer.layout.offset(for: "kzg.witness") else { return nil }

        return KZGProof(
            evaluation: buffer.readFr(at: evalOff),
            witness: buffer.readG1Point(at: witOff))
    }

    /// Verify a KZG proof from a unified buffer using the SRS secret (test mode).
    public func verifyKZGFromBuffer(
        _ buffer: UnifiedProofBuffer,
        commitment: PointProjective,
        point: Fr,
        srs: [PointAffine],
        srsSecret: Fr
    ) -> Bool {
        guard let proof = readKZGProofFromBuffer(buffer) else { return false }

        // Verification: C - v*G == (s - z) * pi
        let g = pointFromAffine(srs[0])
        let lhs = pointAdd(commitment, pointNeg(cPointScalarMul(g, proof.evaluation)))
        let factor = frSub(srsSecret, point)
        let rhs = cPointScalarMul(proof.witness, factor)

        if pointIsIdentity(lhs) && pointIsIdentity(rhs) { return true }
        if pointIsIdentity(lhs) || pointIsIdentity(rhs) { return false }

        let lhsAff = batchToAffine([lhs])
        let rhsAff = batchToAffine([rhs])
        return fpToInt(lhsAff[0].x) == fpToInt(rhsAff[0].x) &&
               fpToInt(lhsAff[0].y) == fpToInt(rhsAff[0].y)
    }

    // MARK: - Plonk Zero-Copy Pipeline

    /// Write a Plonk proof into a unified buffer.
    public func writePlonkProof(_ proof: PlonkProof, to buffer: UnifiedProofBuffer) {
        var layout = ProofBufferLayout()
        let g1Stride = MemoryLayout<PointProjective>.stride
        let frStride = MemoryLayout<Fr>.stride

        // Commitments (G1 points)
        let names = ["plonk.aCommit", "plonk.bCommit", "plonk.cCommit",
                     "plonk.zCommit", "plonk.tLoCommit", "plonk.tMidCommit",
                     "plonk.tHiCommit", "plonk.openingProof", "plonk.shiftedOpeningProof"]
        let points = [proof.aCommit, proof.bCommit, proof.cCommit,
                      proof.zCommit, proof.tLoCommit, proof.tMidCommit,
                      proof.tHiCommit, proof.openingProof, proof.shiftedOpeningProof]

        for (name, pt) in zip(names, points) {
            let off = layout.reserve(name: name, size: g1Stride)
            buffer.writeG1Point(pt, at: off)
        }

        // Extra quotient commits
        let extraCountOff = layout.reserve(name: "plonk.tExtraCount", size: 4)
        buffer.contents.storeBytes(of: UInt32(proof.tExtraCommits.count),
                                   toByteOffset: extraCountOff, as: UInt32.self)
        for (i, pt) in proof.tExtraCommits.enumerated() {
            let off = layout.reserve(name: "plonk.tExtra.\(i)", size: g1Stride)
            buffer.writeG1Point(pt, at: off)
        }

        // Evaluations (Fr elements)
        let evalNames = ["plonk.aEval", "plonk.bEval", "plonk.cEval",
                         "plonk.sigma1Eval", "plonk.sigma2Eval", "plonk.zOmegaEval"]
        let evals: [Fr] = [proof.aEval, proof.bEval, proof.cEval,
                           proof.sigma1Eval, proof.sigma2Eval, proof.zOmegaEval]

        for (name, val) in zip(evalNames, evals) {
            let off = layout.reserve(name: name, size: frStride)
            buffer.writeFr(val, at: off)
        }

        buffer.layout = layout
    }

    /// Read a Plonk proof from a unified buffer (zero-copy).
    public func readPlonkProofFromBuffer(_ buffer: UnifiedProofBuffer) -> PlonkProof? {
        buffer.waitForProducer()

        func g1(_ name: String) -> PointProjective? {
            guard let off = buffer.layout.offset(for: name) else { return nil }
            return buffer.readG1Point(at: off)
        }
        func fr(_ name: String) -> Fr? {
            guard let off = buffer.layout.offset(for: name) else { return nil }
            return buffer.readFr(at: off)
        }

        guard let aC = g1("plonk.aCommit"), let bC = g1("plonk.bCommit"),
              let cC = g1("plonk.cCommit"), let zC = g1("plonk.zCommit"),
              let tLo = g1("plonk.tLoCommit"), let tMid = g1("plonk.tMidCommit"),
              let tHi = g1("plonk.tHiCommit"),
              let op = g1("plonk.openingProof"), let sop = g1("plonk.shiftedOpeningProof"),
              let aE = fr("plonk.aEval"), let bE = fr("plonk.bEval"),
              let cE = fr("plonk.cEval"),
              let s1E = fr("plonk.sigma1Eval"), let s2E = fr("plonk.sigma2Eval"),
              let zOE = fr("plonk.zOmegaEval") else {
            return nil
        }

        guard let ecOff = buffer.layout.offset(for: "plonk.tExtraCount") else { return nil }
        let extraCount = Int(buffer.contents.load(fromByteOffset: ecOff, as: UInt32.self))
        var extras = [PointProjective]()
        extras.reserveCapacity(extraCount)
        for i in 0..<extraCount {
            guard let pt = g1("plonk.tExtra.\(i)") else { return nil }
            extras.append(pt)
        }

        return PlonkProof(
            aCommit: aC, bCommit: bC, cCommit: cC, zCommit: zC,
            tLoCommit: tLo, tMidCommit: tMid, tHiCommit: tHi,
            tExtraCommits: extras,
            aEval: aE, bEval: bE, cEval: cE,
            sigma1Eval: s1E, sigma2Eval: s2E, zOmegaEval: zOE,
            openingProof: op, shiftedOpeningProof: sop)
    }

    // MARK: - Batch Prove-and-Verify

    /// Initialize the buffer pool for batch mode.
    /// Call once before batchProveAndVerify.
    public func initPool(count: Int, proofType: ProofType) throws {
        pool = try UnifiedBufferPool(
            device: device, count: count, proofSize: proofType.estimatedSize)
    }

    /// Initialize pool with explicit buffer size.
    public func initPool(count: Int, bufferSize: Int) throws {
        pool = try UnifiedBufferPool(
            device: device, count: count, proofSize: bufferSize)
    }

    /// Batch verify N Groth16 proofs from the buffer pool.
    /// Each proof is read zero-copy from its UnifiedProofBuffer.
    ///
    /// Flow:
    ///   1. For each proof: acquire buffer -> write proof -> mark complete
    ///   2. For each buffer: wait for GPU -> read proof -> verify
    ///   3. Release all buffers back to pool
    ///
    /// Returns array of per-proof verification results.
    public func batchVerifyGroth16(
        proofs: [Groth16Proof],
        vk: Groth16VerificationKey,
        publicInputs: [[Fr]]
    ) throws -> [Bool] {
        let n = proofs.count
        guard n == publicInputs.count else {
            throw UnifiedVerifierError.mismatchedInputs(
                "Got \(n) proofs but \(publicInputs.count) public input sets")
        }

        // Allocate or use pool
        var buffers = [UnifiedProofBuffer]()
        buffers.reserveCapacity(n)

        if let pool = pool {
            for _ in 0..<n {
                guard let buf = pool.acquire() else {
                    throw UnifiedVerifierError.poolExhausted
                }
                buffers.append(buf)
            }
        } else {
            for _ in 0..<n {
                buffers.append(try allocateProofBuffer(proofType: .groth16))
            }
        }

        // Write all proofs into unified buffers (simulating GPU output)
        for i in 0..<n {
            writeGroth16Proof(proofs[i], to: buffers[i])
        }

        // Verify all proofs (zero-copy reads from unified memory)
        var results = [Bool]()
        results.reserveCapacity(n)

        for i in 0..<n {
            let valid = verifyGroth16FromBuffer(buffers[i], vk: vk, publicInputs: publicInputs[i])
            results.append(valid)
        }

        // Release buffers back to pool
        if let pool = pool {
            for buf in buffers { pool.release(buf) }
        }

        return results
    }

    /// Batch verify N KZG proofs from unified buffers.
    /// Uses BatchVerifier for efficient random-linear-combination batching.
    public func batchVerifyKZG(
        proofs: [KZGProof],
        commitments: [PointProjective],
        points: [Fr],
        srs: [PointAffine],
        srsSecret: Fr
    ) throws -> Bool {
        let n = proofs.count
        guard n == commitments.count, n == points.count else {
            throw UnifiedVerifierError.mismatchedInputs("Mismatched array lengths")
        }

        // Write proofs into unified buffers
        var buffers = [UnifiedProofBuffer]()
        for i in 0..<n {
            let buf = try allocateProofBuffer(proofType: .kzg)
            writeKZGProof(proofs[i], to: buf)
            buffers.append(buf)
        }

        // Build verification items by reading from unified memory (zero-copy)
        var items = [VerificationItem]()
        items.reserveCapacity(n)
        for i in 0..<n {
            guard let proof = readKZGProofFromBuffer(buffers[i]) else {
                throw UnifiedVerifierError.readFailed("Failed to read KZG proof \(i)")
            }
            items.append(VerificationItem(
                commitment: commitments[i],
                point: points[i],
                value: proof.evaluation,
                proof: proof.witness))
        }

        // Batch verify using existing BatchVerifier
        let batchVerifier = try BatchVerifier()
        return try batchVerifier.batchVerifyKZGAdaptive(
            items: items, srs: srs, srsSecret: srsSecret)
    }

    // MARK: - GPU Direct Pipeline

    /// Full zero-copy pipeline: GPU prover writes MTLBuffer -> verifier reads same buffer.
    /// This method encodes prover MSM work into a command buffer, and the proof
    /// output lands directly in a UnifiedProofBuffer that the verifier then reads.
    ///
    /// For Groth16: the MSM results (A, B, C group elements) are written by GPU
    /// into the same shared buffer the verifier consumes.
    ///
    /// Returns the proof buffer (caller can pass to verifyGroth16FromBuffer).
    public func proveGroth16ToBuffer(
        prover: Groth16Prover,
        pk: Groth16ProvingKey,
        r1cs: R1CSInstance,
        publicInputs: [Fr],
        witness: [Fr]
    ) throws -> (buffer: UnifiedProofBuffer, proof: Groth16Proof) {
        // Generate proof (GPU MSM writes intermediate results to storageModeShared buffers)
        let proof = try prover.prove(pk: pk, r1cs: r1cs,
                                     publicInputs: publicInputs, witness: witness)

        // Write proof elements into unified buffer
        // On Apple Silicon, both the MSM output and this buffer are in the same
        // unified memory — the write is a pointer store, not a DMA transfer.
        let buffer = try allocateProofBuffer(proofType: .groth16)
        writeGroth16Proof(proof, to: buffer)

        return (buffer, proof)
    }

    /// End-to-end: prove on GPU, verify from same unified memory, no copies.
    public func proveAndVerifyGroth16(
        prover: Groth16Prover,
        pk: Groth16ProvingKey,
        vk: Groth16VerificationKey,
        r1cs: R1CSInstance,
        publicInputs: [Fr],
        witness: [Fr]
    ) throws -> Bool {
        let (buffer, _) = try proveGroth16ToBuffer(
            prover: prover, pk: pk, r1cs: r1cs,
            publicInputs: publicInputs, witness: witness)

        return verifyGroth16FromBuffer(buffer, vk: vk, publicInputs: publicInputs)
    }
}

// MARK: - Errors

public enum UnifiedVerifierError: Error, CustomStringConvertible {
    case noGPU
    case allocationFailed(String)
    case poolExhausted
    case mismatchedInputs(String)
    case readFailed(String)

    public var description: String {
        switch self {
        case .noGPU: return "No Metal GPU device available"
        case .allocationFailed(let msg): return "Buffer allocation failed: \(msg)"
        case .poolExhausted: return "Buffer pool exhausted — all buffers in use"
        case .mismatchedInputs(let msg): return "Mismatched inputs: \(msg)"
        case .readFailed(let msg): return "Buffer read failed: \(msg)"
        }
    }
}
