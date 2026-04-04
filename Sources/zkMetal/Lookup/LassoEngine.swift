// Lasso Structured Lookup Engine
// Implements the Lasso lookup argument (Setty et al. 2023, Jolt paper) which decomposes
// large lookup tables into small subtables using tensor structure.
//
// Key idea: if T = T1 ⊗ T2 ⊗ ... ⊗ Tc (tensor product of subtables),
// then proving v[i] ∈ T reduces to proving each component v_k[i] ∈ T_k.
// This dramatically reduces prover memory and work for structured tables.
//
// Example: range check [0, 2^32) → 4 subtables of 256 entries each (byte-level).
//
// Memory checking per subtable:
//   For each subtable Tk of size Sk:
//     read_counts[j] = number of times index j is used
//     Prove via sumcheck: Σ_j read_counts[j]/(β - Tk[j]) = Σ_i 1/(β - v_k[i])
//
// References: Lasso (Setty et al. 2023), Jolt (Arun et al. 2024)

import Foundation
import Metal
import NeonFieldOps

// MARK: - Lasso Table Definition

/// Defines a structured lookup table decomposed into subtables.
/// The full table T is the tensor product/composition of subtables.
public struct LassoTable {
    /// Small subtables: subtables[k] is the k-th subtable
    public let subtables: [[Fr]]
    /// How to combine subtable entries into a full table value:
    /// compose([Tk1[idx1], Tk2[idx2], ...]) -> T[i]
    public let compose: ([Fr]) -> Fr
    /// Decompose a lookup value into subtable indices:
    /// decompose(v) -> [idx1, idx2, ...] such that compose(Tk[idxk]) = v
    public let decompose: (Fr) -> [Int]
    /// Number of chunks (subtables)
    public let numChunks: Int
    /// Optional fast batch decomposition: avoids per-element closure + heap allocation.
    /// Takes (lookups, m) and returns [[Int]] of size [numChunks][m].
    public let batchDecompose: (([Fr], Int) -> [[Int]])?

    /// Range check table: proves values are in [0, 2^bits).
    /// Decomposes into `chunks` byte-sized subtables.
    /// Each subtable is [0, 1, ..., 2^(bits/chunks) - 1].
    public static func rangeCheck(bits: Int, chunks: Int) -> LassoTable {
        precondition(bits > 0 && bits % chunks == 0, "bits must be divisible by chunks")
        let bitsPerChunk = bits / chunks
        let chunkSize = 1 << bitsPerChunk
        let chunkMask = UInt64(chunkSize - 1)

        let subtable: [Fr] = (0..<chunkSize).map { frFromInt(UInt64($0)) }
        let subtables = Array(repeating: subtable, count: chunks)

        let compose: ([Fr]) -> Fr = { components in
            // Reconstruct: v = c0 + c1 * 2^bpc + c2 * 2^(2*bpc) + ...
            var result = Fr.zero
            var shift = Fr.one
            let base = frFromInt(UInt64(chunkSize))
            for c in components {
                result = frAdd(result, frMul(c, shift))
                shift = frMul(shift, base)
            }
            return result
        }

        let decompose: (Fr) -> [Int] = { value in
            let limbs = frToInt(value)
            let v = limbs[0]  // Assumes value fits in 64 bits
            var indices = [Int]()
            indices.reserveCapacity(chunks)
            for k in 0..<chunks {
                let shift = k * bitsPerChunk
                let idx = Int((v >> shift) & chunkMask)
                indices.append(idx)
            }
            return indices
        }

        // Fast batch decompose using C: Montgomery reduction + bit extraction in one pass,
        // no per-element heap allocation.
        let bpc = bitsPerChunk
        let nc = chunks
        let batchDec: ([Fr], Int) -> [[Int]] = { lookups, m in
            // C function writes flat array indices[k*m + i]
            var flat = [Int32](repeating: 0, count: nc * m)
            lookups.withUnsafeBytes { rawPtr in
                flat.withUnsafeMutableBufferPointer { idxPtr in
                    bn254_fr_batch_decompose(
                        rawPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(m), Int32(nc), Int32(bpc),
                        idxPtr.baseAddress!
                    )
                }
            }
            // Reshape flat[k*m + i] into [[Int]] of size [numChunks][m]
            var indices = [[Int]](repeating: [Int](repeating: 0, count: m), count: nc)
            for k in 0..<nc {
                let offset = k * m
                for i in 0..<m {
                    indices[k][i] = Int(flat[offset + i])
                }
            }
            return indices
        }

        return LassoTable(subtables: subtables, compose: compose,
                          decompose: decompose, numChunks: chunks,
                          batchDecompose: batchDec)
    }

    /// XOR table: proves z = x XOR y for `bits`-wide values.
    /// Decomposes into byte-level XOR tables.
    /// Each subtable has 256 entries: xor_table[i*16 + j] = i XOR j for 4-bit nibbles,
    /// or for 8-bit: subtable[i*256 + j] = i XOR j.
    /// For simplicity, uses full 2^(2*bpc) entry table per chunk.
    public static func xor(bits: Int) -> LassoTable {
        precondition(bits > 0 && bits % 2 == 0, "bits must be even")
        let bitsPerChunk = min(bits, 8)  // 8-bit chunks max
        let chunks = bits / bitsPerChunk
        let chunkRange = 1 << bitsPerChunk
        let chunkMask = UInt64(chunkRange - 1)

        // Each subtable: for inputs (a, b) ∈ [0, chunkRange)^2, entry = a XOR b
        // Indexed as a * chunkRange + b
        let tableSize = chunkRange * chunkRange
        let subtable: [Fr] = (0..<tableSize).map { idx in
            let a = idx / chunkRange
            let b = idx % chunkRange
            return frFromInt(UInt64(a ^ b))
        }
        let subtables = Array(repeating: subtable, count: chunks)

        let compose: ([Fr]) -> Fr = { components in
            var result = Fr.zero
            var shift = Fr.one
            let base = frFromInt(UInt64(chunkRange))
            for c in components {
                result = frAdd(result, frMul(c, shift))
                shift = frMul(shift, base)
            }
            return result
        }

        // decompose takes a packed (x, y, z) and returns subtable indices
        // For XOR: the lookup value encodes (x_k, y_k) per chunk as x_k * chunkRange + y_k
        let decompose: (Fr) -> [Int] = { value in
            let limbs = frToInt(value)
            let v = limbs[0]
            var indices = [Int]()
            indices.reserveCapacity(chunks)
            for k in 0..<chunks {
                let shift = k * bitsPerChunk
                let idx = Int((v >> shift) & chunkMask)
                // For XOR, the index into the subtable packs the two operand chunks
                indices.append(idx)
            }
            return indices
        }

        return LassoTable(subtables: subtables, compose: compose,
                          decompose: decompose, numChunks: chunks,
                          batchDecompose: nil)
    }

    /// AND table: proves z = x AND y for `bits`-wide values.
    /// Same structure as XOR but with AND operation.
    public static func and(bits: Int) -> LassoTable {
        precondition(bits > 0 && bits % 2 == 0, "bits must be even")
        let bitsPerChunk = min(bits, 8)
        let chunks = bits / bitsPerChunk
        let chunkRange = 1 << bitsPerChunk
        let chunkMask = UInt64(chunkRange - 1)

        let tableSize = chunkRange * chunkRange
        let subtable: [Fr] = (0..<tableSize).map { idx in
            let a = idx / chunkRange
            let b = idx % chunkRange
            return frFromInt(UInt64(a & b))
        }
        let subtables = Array(repeating: subtable, count: chunks)

        let compose: ([Fr]) -> Fr = { components in
            var result = Fr.zero
            var shift = Fr.one
            let base = frFromInt(UInt64(chunkRange))
            for c in components {
                result = frAdd(result, frMul(c, shift))
                shift = frMul(shift, base)
            }
            return result
        }

        let decompose: (Fr) -> [Int] = { value in
            let limbs = frToInt(value)
            let v = limbs[0]
            var indices = [Int]()
            indices.reserveCapacity(chunks)
            for k in 0..<chunks {
                let shift = k * bitsPerChunk
                let idx = Int((v >> shift) & chunkMask)
                indices.append(idx)
            }
            return indices
        }

        return LassoTable(subtables: subtables, compose: compose,
                          decompose: decompose, numChunks: chunks,
                          batchDecompose: nil)
    }
}

// MARK: - Lasso Proof

/// Proof for one subtable's memory checking argument
public struct SubtableProof {
    /// Subtable index (which chunk)
    public let chunkIndex: Int
    /// Read counts: read_counts[j] = number of times subtable[j] was accessed
    public let readCounts: [Fr]
    /// Challenge beta used for this subtable
    public let beta: Fr
    /// Sumcheck rounds for the read side: Σ_i 1/(β + v_k[i])
    public let readSumcheckRounds: [(Fr, Fr, Fr)]
    /// Sumcheck rounds for the table side: Σ_j read_counts[j]/(β + Tk[j])
    public let tableSumcheckRounds: [(Fr, Fr, Fr)]
    /// Claimed sum S = Σ_i 1/(β + v_k[i])
    public let claimedSum: Fr
    /// Final evaluation of read inverse polynomial at random point
    public let readFinalEval: Fr
    /// Final evaluation of table inverse polynomial at random point
    public let tableFinalEval: Fr

    public init(chunkIndex: Int, readCounts: [Fr], beta: Fr,
                readSumcheckRounds: [(Fr, Fr, Fr)],
                tableSumcheckRounds: [(Fr, Fr, Fr)],
                claimedSum: Fr, readFinalEval: Fr, tableFinalEval: Fr) {
        self.chunkIndex = chunkIndex
        self.readCounts = readCounts
        self.beta = beta
        self.readSumcheckRounds = readSumcheckRounds
        self.tableSumcheckRounds = tableSumcheckRounds
        self.claimedSum = claimedSum
        self.readFinalEval = readFinalEval
        self.tableFinalEval = tableFinalEval
    }
}

/// Complete Lasso proof: one SubtableProof per chunk plus decomposition witnesses
public struct LassoProof {
    /// Number of chunks (subtables)
    public let numChunks: Int
    /// Per-subtable proofs
    public let subtableProofs: [SubtableProof]
    /// Decomposed indices: indices[k][i] = index into subtable k for lookup i
    public let indices: [[Int]]

    public init(numChunks: Int, subtableProofs: [SubtableProof], indices: [[Int]]) {
        self.numChunks = numChunks
        self.subtableProofs = subtableProofs
        self.indices = indices
    }
}

// MARK: - Lasso Engine

public class LassoEngine {
    public static let version = Versions.lasso
    public let polyEngine: PolyEngine
    public let sumcheckEngine: SumcheckEngine

    /// Enable profiling output to identify which phases dominate.
    public var profileLasso = false

    // Grow-only buffer cache: avoids per-call GPU allocation
    private var cachedReadBuf: (MTLBuffer, Int)?      // for betaPlusRead (size m)
    private var cachedReadOutBuf: (MTLBuffer, Int)?    // for hRead output (size m)
    private var cachedTableBuf: (MTLBuffer, Int)?      // for betaPlusT (size S)
    private var cachedTableOutBuf: (MTLBuffer, Int)?   // for invBetaPlusT output (size S)
    private var cachedHadamardInBuf: (MTLBuffer, Int)? // for readCounts input (size S)
    private var cachedHadamardOutBuf: (MTLBuffer, Int)? // for hTable output (size S)

    /// Get or grow a cached buffer. Returns the buffer (capacity >= minBytes).
    private func getCachedBuffer(_ cached: inout (MTLBuffer, Int)?, minBytes: Int) -> MTLBuffer {
        if let (buf, cap) = cached, cap >= minBytes {
            return buf
        }
        let buf = polyEngine.device.makeBuffer(length: minBytes, options: .storageModeShared)!
        cached = (buf, minBytes)
        return buf
    }

    public init() throws {
        self.polyEngine = try PolyEngine()
        self.sumcheckEngine = try SumcheckEngine()
    }

    /// Prove that every element in `lookups` exists in the structured table.
    /// The table must have tensor/decomposable structure defined by LassoTable.
    public func prove(lookups: [Fr], table: LassoTable) throws -> LassoProof {
        let _tTotal = profileLasso ? CFAbsoluteTimeGetCurrent() : 0
        let m = lookups.count
        precondition(m > 0 && (m & (m - 1)) == 0, "Lookup count must be power of 2")

        // Step 1: Decompose each lookup value into subtable indices
        var _tPhase = profileLasso ? CFAbsoluteTimeGetCurrent() : 0

        let indices: [[Int]]
        if let batchDec = table.batchDecompose {
            // Fast C path: Montgomery reduction + bit extraction in one pass
            indices = batchDec(lookups, m)
        } else {
            // Fallback: per-element closure decomposition
            var idx = [[Int]](repeating: [Int](repeating: 0, count: m), count: table.numChunks)
            for i in 0..<m {
                let decomposed = table.decompose(lookups[i])
                precondition(decomposed.count == table.numChunks,
                             "Decomposition must produce \(table.numChunks) indices")
                for k in 0..<table.numChunks {
                    idx[k][i] = decomposed[k]
                }
            }
            indices = idx
        }

        if profileLasso { let _t = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [lasso] decompose: %.2f ms\n", (_t - _tPhase) * 1000), stderr); _tPhase = _t }

        // Step 2: For each subtable, build read_counts and run memory checking sumcheck
        var subtableProofs = [SubtableProof]()
        subtableProofs.reserveCapacity(table.numChunks)

        var transcript = [UInt8]()
        transcript.reserveCapacity(1024)
        // Seed transcript with lookup count and number of chunks
        appendUInt64(&transcript, UInt64(m))
        appendUInt64(&transcript, UInt64(table.numChunks))

        let logM = Int(log2(Double(m)))
        let frStride = MemoryLayout<Fr>.stride

        for k in 0..<table.numChunks {
            let subtable = table.subtables[k]
            let S = subtable.count
            precondition(S > 0 && (S & (S - 1)) == 0, "Subtable size must be power of 2")

            if profileLasso { _tPhase = CFAbsoluteTimeGetCurrent() }

            // Compute read_counts[j] = how many times index j is used
            var countRaw = [UInt64](repeating: 0, count: S)
            for idx in indices[k] {
                precondition(idx >= 0 && idx < S, "Index \(idx) out of subtable range [0, \(S))")
                countRaw[idx] &+= 1
            }
            let readCounts: [Fr] = countRaw.map { frFromInt($0) }

            // Derive beta challenge from transcript
            let beta = deriveChallenge(transcript)
            appendFr(&transcript, beta)

            if profileLasso { let _t = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [lasso] chunk %d counts+beta: %.2f ms\n", k, (_t - _tPhase) * 1000), stderr); _tPhase = _t }

            // Compute read-side evaluations: h_read[i] = 1/(β + subtable[indices[k][i]])
            // Use C batch_beta_add for cache-friendly gather+add (avoids Swift frAdd overhead)
            let readBytes = m * frStride
            let readBuf = getCachedBuffer(&cachedReadBuf, minBytes: readBytes)
            let chunkIndices32: [Int32] = indices[k].map { Int32($0) }
            withUnsafeBytes(of: beta) { betaPtr in
                subtable.withUnsafeBytes { valPtr in
                    chunkIndices32.withUnsafeBufferPointer { idxPtr in
                        bn254_fr_batch_beta_add(
                            betaPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            valPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            idxPtr.baseAddress!,
                            Int32(m),
                            readBuf.contents().assumingMemoryBound(to: UInt64.self)
                        )
                    }
                }
            }

            // Prepare table-side: compute β + Tk[j] for all j, write to GPU buffer
            let tableBytes = S * frStride
            let tableBuf = getCachedBuffer(&cachedTableBuf, minBytes: tableBytes)
            let seqIndices: [Int32] = (0..<S).map { Int32($0) }
            withUnsafeBytes(of: beta) { betaPtr in
                subtable.withUnsafeBytes { valPtr in
                    seqIndices.withUnsafeBufferPointer { idxPtr in
                        bn254_fr_batch_beta_add(
                            betaPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            valPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            idxPtr.baseAddress!,
                            Int32(S),
                            tableBuf.contents().assumingMemoryBound(to: UInt64.self)
                        )
                    }
                }
            }

            // Prepare hadamard input buffer (readCounts)
            let hadInBuf = getCachedBuffer(&cachedHadamardInBuf, minBytes: tableBytes)
            readCounts.withUnsafeBytes { src in memcpy(hadInBuf.contents(), src.baseAddress!, tableBytes) }

            // Single command buffer: batchInverse(read) + batchInverse(table) + hadamard(table)
            let readOutBuf = getCachedBuffer(&cachedReadOutBuf, minBytes: readBytes)
            let tableOutBuf = getCachedBuffer(&cachedTableOutBuf, minBytes: tableBytes)
            let hadOutBuf = getCachedBuffer(&cachedHadamardOutBuf, minBytes: tableBytes)
            try encodeFusedInverseAndHadamard(
                readIn: readBuf, readOut: readOutBuf, readN: m,
                tableIn: tableBuf, tableOut: tableOutBuf, tableN: S,
                hadA: hadInBuf, hadB: tableOutBuf, hadOut: hadOutBuf, hadN: S
            )

            let hRead = Array(UnsafeBufferPointer(start: readOutBuf.contents().bindMemory(to: Fr.self, capacity: m), count: m))
            let hTable = Array(UnsafeBufferPointer(start: hadOutBuf.contents().bindMemory(to: Fr.self, capacity: S), count: S))

            if profileLasso { let _t = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [lasso] chunk %d fused GPU (%d+%d): %.2f ms\n", k, m, S, (_t - _tPhase) * 1000), stderr); _tPhase = _t }

            // Compute claimed sum S = Σ h_read[i] using fast C vector sum
            var sum: Fr = Fr.zero
            var sumLimbs = [UInt64](repeating: 0, count: 4)
            bn254_fr_vector_sum(
                readOutBuf.contents().assumingMemoryBound(to: UInt64.self),
                Int32(m),
                &sumLimbs
            )
            sum = Fr.from64(sumLimbs)

            // Sanity check: table side should match (small array, use C sum too)
            var tableSumLimbs = [UInt64](repeating: 0, count: 4)
            hTable.withUnsafeBytes { ptr in
                bn254_fr_vector_sum(
                    ptr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(S),
                    &tableSumLimbs
                )
            }
            let tableSum = Fr.from64(tableSumLimbs)
            precondition(frEqual(sum, tableSum),
                         "Lasso sum mismatch for subtable \(k) — decomposition error")

            if profileLasso { let _t = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [lasso] chunk %d sums: %.2f ms\n", k, (_t - _tPhase) * 1000), stderr); _tPhase = _t }

            // Run sumcheck on h_read (read side)
            appendFr(&transcript, sum)

            let (readRounds, readFinalEval, readChallenges) = try runSumcheck(
                evals: hRead, numVars: logM, transcript: &transcript)

            if profileLasso { let _t = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [lasso] chunk %d sumcheck read (2^%d): %.2f ms\n", k, logM, (_t - _tPhase) * 1000), stderr); _tPhase = _t }

            // Run sumcheck on h_table (table side)
            let logS = Int(log2(Double(S)))
            let (tableRounds, tableFinalEval, _) = try runSumcheck(
                evals: hTable, numVars: logS, transcript: &transcript)

            if profileLasso { let _t = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [lasso] chunk %d sumcheck table (2^%d): %.2f ms\n", k, logS, (_t - _tPhase) * 1000), stderr); _tPhase = _t }

            subtableProofs.append(SubtableProof(
                chunkIndex: k,
                readCounts: readCounts,
                beta: beta,
                readSumcheckRounds: readRounds,
                tableSumcheckRounds: tableRounds,
                claimedSum: sum,
                readFinalEval: readFinalEval,
                tableFinalEval: tableFinalEval
            ))
        }

        if profileLasso { fputs(String(format: "  [lasso] total prove: %.2f ms\n", (CFAbsoluteTimeGetCurrent() - _tTotal) * 1000), stderr) }

        return LassoProof(numChunks: table.numChunks,
                          subtableProofs: subtableProofs,
                          indices: indices)
    }

    // MARK: - Cached GPU dispatch helpers

    /// Fused GPU dispatch: batchInverse(read) + batchInverse(table) + hadamard(table)
    /// in a single command buffer, saving 2 command buffer creation/commit/wait cycles.
    private func encodeFusedInverseAndHadamard(
        readIn: MTLBuffer, readOut: MTLBuffer, readN: Int,
        tableIn: MTLBuffer, tableOut: MTLBuffer, tableN: Int,
        hadA: MTLBuffer, hadB: MTLBuffer, hadOut: MTLBuffer, hadN: Int
    ) throws {
        guard let cmdBuf = polyEngine.commandQueue.makeCommandBuffer() else { throw MSMError.noCommandBuffer }

        // Encode batchInverse for read-side (large: 262144)
        let enc1 = cmdBuf.makeComputeCommandEncoder()!
        enc1.setComputePipelineState(polyEngine.batchInverseFunction)
        enc1.setBuffer(readIn, offset: 0, index: 0)
        enc1.setBuffer(readOut, offset: 0, index: 1)
        var readNVal = UInt32(readN)
        enc1.setBytes(&readNVal, length: 4, index: 2)
        let chunkSize = 512
        let readGroups = (readN + chunkSize - 1) / chunkSize
        let biTG = min(64, Int(polyEngine.batchInverseFunction.maxTotalThreadsPerThreadgroup))
        enc1.dispatchThreadgroups(MTLSize(width: readGroups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: biTG, height: 1, depth: 1))
        enc1.endEncoding()

        // Encode batchInverse for table-side (small: 256)
        let enc2 = cmdBuf.makeComputeCommandEncoder()!
        enc2.setComputePipelineState(polyEngine.batchInverseFunction)
        enc2.setBuffer(tableIn, offset: 0, index: 0)
        enc2.setBuffer(tableOut, offset: 0, index: 1)
        var tableNVal = UInt32(tableN)
        enc2.setBytes(&tableNVal, length: 4, index: 2)
        let tableGroups = (tableN + chunkSize - 1) / chunkSize
        enc2.dispatchThreadgroups(MTLSize(width: tableGroups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: biTG, height: 1, depth: 1))
        enc2.endEncoding()

        // Encode hadamard: hadOut = hadA * hadB (uses tableOut from previous encoder)
        let enc3 = cmdBuf.makeComputeCommandEncoder()!
        enc3.setComputePipelineState(polyEngine.hadamardFunction)
        enc3.setBuffer(hadA, offset: 0, index: 0)
        enc3.setBuffer(hadB, offset: 0, index: 1)
        enc3.setBuffer(hadOut, offset: 0, index: 2)
        var hadNVal = UInt32(hadN)
        enc3.setBytes(&hadNVal, length: 4, index: 3)
        let hadTG = min(256, Int(polyEngine.hadamardFunction.maxTotalThreadsPerThreadgroup))
        enc3.dispatchThreads(MTLSize(width: hadN, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: hadTG, height: 1, depth: 1))
        enc3.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error { throw MSMError.gpuError(error.localizedDescription) }
    }

    /// Verify a Lasso proof.
    /// Optimized: C batch inverse (Montgomery's trick), C MLE eval, precomputed constants.
    public func verify(proof: LassoProof, lookups: [Fr], table: LassoTable) throws -> Bool {
        let _tTotal = profileLasso ? CFAbsoluteTimeGetCurrent() : 0
        var _tPhase = profileLasso ? CFAbsoluteTimeGetCurrent() : 0
        let m = lookups.count
        guard proof.numChunks == table.numChunks else { return false }
        guard proof.subtableProofs.count == table.numChunks else { return false }
        guard proof.indices.count == table.numChunks else { return false }

        // Verify decomposition: compose(subtable[k][indices[k][i]]) == lookups[i]
        // Optimized: for range-check tables, use batchDecompose to recompute indices
        // and compare directly, avoiding per-element compose() closure calls.
        if let batchDec = table.batchDecompose {
            // Fast path: recompute decomposition and compare index arrays
            let recomputed = batchDec(lookups, m)
            for k in 0..<table.numChunks {
                let S = table.subtables[k].count
                for i in 0..<m {
                    let idx = proof.indices[k][i]
                    guard idx >= 0 && idx < S else { return false }
                    guard idx == recomputed[k][i] else { return false }
                }
            }
        } else {
            // Fallback: per-element compose check
            for i in 0..<m {
                var components = [Fr]()
                components.reserveCapacity(table.numChunks)
                for k in 0..<table.numChunks {
                    guard proof.indices[k][i] >= 0 && proof.indices[k][i] < table.subtables[k].count else {
                        return false
                    }
                    components.append(table.subtables[k][proof.indices[k][i]])
                }
                let reconstructed = table.compose(components)
                if !frEqual(reconstructed, lookups[i]) { return false }
            }
        }

        if profileLasso { let _t = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [verify] decomposition check: %.2f ms\n", (_t - _tPhase) * 1000), stderr); _tPhase = _t }

        // Reconstruct transcript (must match prover)
        var transcript = [UInt8]()
        transcript.reserveCapacity(1024)
        appendUInt64(&transcript, UInt64(m))
        appendUInt64(&transcript, UInt64(table.numChunks))

        // Precompute inv2 once (used in evaluateQuadratic, avoids repeated field inversion)
        let two = frAdd(Fr.one, Fr.one)
        let inv2 = frInverse(two)
        let negOne = frSub(Fr.zero, Fr.one)

        let logM = Int(log2(Double(m)))

        // Phase 1: Sequential transcript reconstruction + lightweight checks.
        // Derive all challenges, verify sumcheck round consistency.
        // This is fast (<2ms) since it only involves hashing + field adds.
        var allReadChallenges = [[Fr]]()
        var allTableChallenges = [[Fr]]()
        allReadChallenges.reserveCapacity(table.numChunks)
        allTableChallenges.reserveCapacity(table.numChunks)

        for k in 0..<table.numChunks {
            let sp = proof.subtableProofs[k]
            let subtable = table.subtables[k]
            let S = subtable.count
            let logS = Int(log2(Double(S)))

            guard sp.readSumcheckRounds.count == logM else { return false }
            guard sp.tableSumcheckRounds.count == logS else { return false }
            guard sp.readCounts.count == S else { return false }

            // Verify beta derivation
            let expectedBeta = deriveChallenge(transcript)
            guard frEqual(sp.beta, expectedBeta) else { return false }
            appendFr(&transcript, sp.beta)

            // Verify read_counts consistency: sum must equal m
            var totalReadsLimbs = [UInt64](repeating: 0, count: 4)
            sp.readCounts.withUnsafeBytes { ptr in
                bn254_fr_vector_sum(
                    ptr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(S),
                    &totalReadsLimbs
                )
            }
            let totalReads = Fr.from64(totalReadsLimbs)
            if !frEqual(totalReads, frFromInt(UInt64(m))) { return false }

            // Verify read_counts match the claimed indices
            var expectedCounts = [UInt64](repeating: 0, count: S)
            for idx in proof.indices[k] {
                guard idx >= 0 && idx < S else { return false }
                expectedCounts[idx] &+= 1
            }
            for j in 0..<S {
                if !frEqual(sp.readCounts[j], frFromInt(expectedCounts[j])) { return false }
            }

            appendFr(&transcript, sp.claimedSum)

            // Derive read-side challenges
            let useGPURead = m >= 256
            var rc = [Fr]()
            rc.reserveCapacity(logM)
            if useGPURead {
                for _ in 0..<logM {
                    let c = deriveChallenge(transcript)
                    rc.append(c)
                    appendFr(&transcript, c)
                }
            } else {
                for round in 0..<logM {
                    let (s0, s1, s2) = sp.readSumcheckRounds[round]
                    appendFr(&transcript, s0)
                    appendFr(&transcript, s1)
                    appendFr(&transcript, s2)
                    let c = deriveChallenge(transcript)
                    rc.append(c)
                    appendFr(&transcript, c)
                }
            }
            allReadChallenges.append(rc)

            // Check sumcheck read rounds
            let (rs0, rs1, _) = sp.readSumcheckRounds[0]
            if !frEqual(frAdd(rs0, rs1), sp.claimedSum) { return false }
            for round in 1..<logM {
                let (s0, s1, _) = sp.readSumcheckRounds[round]
                let prevEval = evaluateQuadraticFast(sp.readSumcheckRounds[round - 1], at: rc[round - 1], inv2: inv2, negOne: negOne)
                if !frEqual(frAdd(s0, s1), prevEval) { return false }
            }
            let lastReadEval = evaluateQuadraticFast(
                sp.readSumcheckRounds[logM - 1], at: rc[logM - 1], inv2: inv2, negOne: negOne)
            if !frEqual(lastReadEval, sp.readFinalEval) { return false }

            // Derive table-side challenges
            let useGPUTable = S >= 256
            var tc = [Fr]()
            tc.reserveCapacity(logS)
            if useGPUTable {
                for _ in 0..<logS {
                    let c = deriveChallenge(transcript)
                    tc.append(c)
                    appendFr(&transcript, c)
                }
            } else {
                for round in 0..<logS {
                    let (s0, s1, s2) = sp.tableSumcheckRounds[round]
                    appendFr(&transcript, s0)
                    appendFr(&transcript, s1)
                    appendFr(&transcript, s2)
                    let c = deriveChallenge(transcript)
                    tc.append(c)
                    appendFr(&transcript, c)
                }
            }
            allTableChallenges.append(tc)

            // Check sumcheck table rounds
            let (ts0, ts1, _) = sp.tableSumcheckRounds[0]
            if !frEqual(frAdd(ts0, ts1), sp.claimedSum) { return false }
            for round in 1..<logS {
                let (s0, s1, _) = sp.tableSumcheckRounds[round]
                let prevEval = evaluateQuadraticFast(sp.tableSumcheckRounds[round - 1], at: tc[round - 1], inv2: inv2, negOne: negOne)
                if !frEqual(frAdd(s0, s1), prevEval) { return false }
            }
            let lastTableEval = evaluateQuadraticFast(
                sp.tableSumcheckRounds[logS - 1], at: tc[logS - 1], inv2: inv2, negOne: negOne)
            if !frEqual(lastTableEval, sp.tableFinalEval) { return false }
        }

        if profileLasso { let _t = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [verify] phase 1 (transcript+sumcheck): %.2f ms\n", (_t - _tPhase) * 1000), stderr); _tPhase = _t }

        // Phase 2: Parallel polynomial evaluation across all chunks.
        // Each chunk's inverse_evals + MLE is independent and compute-intensive.
        var chunkResults = [Bool](repeating: true, count: table.numChunks)

        DispatchQueue.concurrentPerform(iterations: table.numChunks) { k in
            let sp = proof.subtableProofs[k]
            let subtable = table.subtables[k]
            let S = subtable.count
            let logS = Int(log2(Double(S)))
            let rc = allReadChallenges[k]
            let tc = allTableChallenges[k]

            // Read-side: gather + batch inverse + MLE
            var hReadEvals = [Fr](repeating: Fr.zero, count: m)
            let idx32: [Int32] = proof.indices[k].map { Int32($0) }
            subtable.withUnsafeBytes { stPtr in
                withUnsafeBytes(of: sp.beta) { betaPtr in
                    idx32.withUnsafeBufferPointer { idxPtr in
                        hReadEvals.withUnsafeMutableBytes { outPtr in
                            bn254_fr_inverse_evals_indexed(
                                betaPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                stPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                UnsafePointer<Int32>(OpaquePointer(idxPtr.baseAddress!)),
                                Int32(m),
                                outPtr.baseAddress!.assumingMemoryBound(to: UInt64.self)
                            )
                        }
                    }
                }
            }
            var hReadResult = [UInt64](repeating: 0, count: 4)
            hReadEvals.withUnsafeBytes { evPtr in
                rc.withUnsafeBytes { ptPtr in
                    bn254_fr_mle_eval(
                        evPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(logM),
                        ptPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        &hReadResult
                    )
                }
            }
            let hReadAtR = Fr.from64(hReadResult)
            if !frEqual(hReadAtR, sp.readFinalEval) { chunkResults[k] = false; return }

            // Table-side: weighted inverse + MLE
            var hTableEvals = [Fr](repeating: Fr.zero, count: S)
            subtable.withUnsafeBytes { valPtr in
                withUnsafeBytes(of: sp.beta) { betaPtr in
                    sp.readCounts.withUnsafeBytes { wPtr in
                        hTableEvals.withUnsafeMutableBytes { outPtr in
                            bn254_fr_weighted_inverse_evals(
                                betaPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                valPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                wPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                Int32(S),
                                outPtr.baseAddress!.assumingMemoryBound(to: UInt64.self)
                            )
                        }
                    }
                }
            }
            var hTableResult = [UInt64](repeating: 0, count: 4)
            hTableEvals.withUnsafeBytes { evPtr in
                tc.withUnsafeBytes { ptPtr in
                    bn254_fr_mle_eval(
                        evPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(logS),
                        ptPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        &hTableResult
                    )
                }
            }
            let hTableAtR = Fr.from64(hTableResult)
            if !frEqual(hTableAtR, sp.tableFinalEval) { chunkResults[k] = false }
        }

        // Check all chunks passed
        for k in 0..<table.numChunks {
            if !chunkResults[k] { return false }
        }

        if profileLasso { let _t = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [verify] phase 2 (parallel poly eval): %.2f ms\n", (_t - _tPhase) * 1000), stderr); _tPhase = _t }

        if profileLasso { fputs(String(format: "  [verify] total: %.2f ms\n", (CFAbsoluteTimeGetCurrent() - _tTotal) * 1000), stderr) }

        return true
    }

    // MARK: - Sumcheck execution

    /// Run sumcheck, matching the pattern from LookupEngine.
    private func runSumcheck(evals: [Fr], numVars: Int,
                             transcript: inout [UInt8]) throws
        -> (rounds: [(Fr, Fr, Fr)], finalEval: Fr, challenges: [Fr])
    {
        let useGPU = evals.count >= 256
        var rounds = [(Fr, Fr, Fr)]()
        var challenges = [Fr]()
        rounds.reserveCapacity(numVars)
        challenges.reserveCapacity(numVars)

        if useGPU {
            for _ in 0..<numVars {
                let c = deriveChallenge(transcript)
                challenges.append(c)
                appendFr(&transcript, c)
            }
            let (gpuRounds, finalEval) = try sumcheckEngine.fullSumcheck(
                evals: evals, challenges: challenges)
            return (gpuRounds, finalEval, challenges)
        } else {
            var current = evals
            for _ in 0..<numVars {
                let roundPoly = SumcheckEngine.cpuRoundPoly(evals: current)
                rounds.append(roundPoly)

                appendFr(&transcript, roundPoly.0)
                appendFr(&transcript, roundPoly.1)
                appendFr(&transcript, roundPoly.2)
                let challenge = deriveChallenge(transcript)
                challenges.append(challenge)
                appendFr(&transcript, challenge)

                current = SumcheckEngine.cpuReduce(evals: current, challenge: challenge)
            }
            precondition(current.count == 1)
            return (rounds, current[0], challenges)
        }
    }

    // MARK: - Helpers

    /// Compute h[i] = 1/(β + values[i])
    private func computeInverseEvals(values: [Fr], beta: Fr) throws -> [Fr] {
        var betaPlusV = [Fr](repeating: Fr.zero, count: values.count)
        for i in 0..<values.count {
            betaPlusV[i] = frAdd(beta, values[i])
        }
        return try polyEngine.batchInverse(betaPlusV)
    }

    /// Compute h[j] = weights[j]/(β + values[j])
    private func computeWeightedInverseEvals(values: [Fr], weights: [Fr], beta: Fr) throws -> [Fr] {
        var betaPlusV = [Fr](repeating: Fr.zero, count: values.count)
        for i in 0..<values.count {
            betaPlusV[i] = frAdd(beta, values[i])
        }
        let inv = try polyEngine.batchInverse(betaPlusV)
        return try polyEngine.hadamard(weights, inv)
    }

    /// Evaluate a degree-2 polynomial given by (S(0), S(1), S(2)) at point x.
    private func evaluateQuadratic(_ triple: (Fr, Fr, Fr), at x: Fr) -> Fr {
        let (s0, s1, s2) = triple
        let one = Fr.one
        let two = frAdd(one, one)
        let xm1 = frSub(x, one)
        let xm2 = frSub(x, two)
        let inv2 = frInverse(two)
        let negOne = frSub(Fr.zero, one)

        let l0 = frMul(frMul(xm1, xm2), inv2)
        let l1 = frMul(frMul(x, xm2), negOne)
        let l2 = frMul(frMul(x, xm1), inv2)

        return frAdd(frAdd(frMul(s0, l0), frMul(s1, l1)), frMul(s2, l2))
    }

    /// Fast evaluateQuadratic with precomputed constants (avoids frInverse per call).
    private func evaluateQuadraticFast(_ triple: (Fr, Fr, Fr), at x: Fr, inv2: Fr, negOne: Fr) -> Fr {
        let (s0, s1, s2) = triple
        let xm1 = frSub(x, Fr.one)
        let two = frAdd(Fr.one, Fr.one)
        let xm2 = frSub(x, two)

        let l0 = frMul(frMul(xm1, xm2), inv2)
        let l1 = frMul(frMul(x, xm2), negOne)
        let l2 = frMul(frMul(x, xm1), inv2)

        return frAdd(frAdd(frMul(s0, l0), frMul(s1, l1)), frMul(s2, l2))
    }

    /// Evaluate multilinear extension at a point.
    private func evaluateMLE(_ evals: [Fr], at point: [Fr]) -> Fr {
        var current = evals
        for r in point {
            let half = current.count / 2
            var next = [Fr](repeating: Fr.zero, count: half)
            let oneMinusR = frSub(Fr.one, r)
            for i in 0..<half {
                next[i] = frAdd(frMul(oneMinusR, current[i]), frMul(r, current[half + i]))
            }
            current = next
        }
        precondition(current.count == 1)
        return current[0]
    }

    // MARK: - Fiat-Shamir

    private func appendFr(_ transcript: inout [UInt8], _ v: Fr) {
        let vInt = frToInt(v)
        for limb in vInt {
            for byte in 0..<8 {
                transcript.append(UInt8((limb >> (byte * 8)) & 0xFF))
            }
        }
    }

    private func appendUInt64(_ transcript: inout [UInt8], _ v: UInt64) {
        var val = v
        for _ in 0..<8 {
            transcript.append(UInt8(val & 0xFF))
            val >>= 8
        }
    }

    private func deriveChallenge(_ transcript: [UInt8]) -> Fr {
        let hash = blake3(transcript)
        var limbs = [UInt64](repeating: 0, count: 4)
        for i in 0..<4 {
            for j in 0..<8 {
                limbs[i] |= UInt64(hash[i * 8 + j]) << (j * 8)
            }
        }
        let raw = Fr.from64(limbs)
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))
    }
}
