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

    // Reusable C-allocated buffers for inverse evals (avoids Swift array zero-init)
    private var hReadBuf: UnsafeMutablePointer<UInt64>?
    private var hReadBufCount = 0
    private var hTableBuf: UnsafeMutablePointer<UInt64>?
    private var hTableBufCount = 0

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

        // indices32Flat: contiguous Int32 array [numChunks * m], indices32Flat[k*m+i]
        // The [[Int]] reshape is deferred to proof construction to avoid O(numChunks*m) copy.
        let indices32Flat: [Int32]
        if table.batchDecompose != nil {
            // Fast C path: Montgomery reduction + bit extraction in one pass
            var flat = [Int32](repeating: 0, count: table.numChunks * m)
            lookups.withUnsafeBytes { rawPtr in
                flat.withUnsafeMutableBufferPointer { idxPtr in
                    bn254_fr_batch_decompose(
                        rawPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(m), Int32(table.numChunks), Int32(table.subtables[0].count.trailingZeroBitCount),
                        idxPtr.baseAddress!
                    )
                }
            }
            indices32Flat = flat
        } else {
            // Fallback: per-element closure decomposition
            var flat = [Int32](repeating: 0, count: table.numChunks * m)
            for i in 0..<m {
                let decomposed = table.decompose(lookups[i])
                precondition(decomposed.count == table.numChunks,
                             "Decomposition must produce \(table.numChunks) indices")
                for k in 0..<table.numChunks {
                    flat[k * m + i] = Int32(decomposed[k])
                }
            }
            indices32Flat = flat
        }

        if profileLasso { let _t = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [lasso] decompose: %.2f ms\n", (_t - _tPhase) * 1000), stderr); _tPhase = _t }

        // Step 2: For each subtable, build read_counts and run memory checking sumcheck
        var transcript = [UInt8]()
        transcript.reserveCapacity(1024)
        // Seed transcript with lookup count and number of chunks
        appendUInt64(&transcript, UInt64(m))
        appendUInt64(&transcript, UInt64(table.numChunks))

        let logM = Int(log2(Double(m)))
        let frStride = MemoryLayout<Fr>.stride

        // Two-phase approach: Phase A computes inverse evals + transcript, Phase B runs GPU sumchecks.
        var subtableProofs = [SubtableProof]()
        subtableProofs.reserveCapacity(table.numChunks)

        // Per-chunk reusable C buffers for inverse evals
        let firstS = table.subtables[0].count
        if hReadBufCount < m {
            hReadBuf?.deallocate()
            hReadBuf = .allocate(capacity: m * 4)
            hReadBufCount = m
        }
        if hTableBufCount < firstS {
            hTableBuf?.deallocate()
            hTableBuf = .allocate(capacity: firstS * 4)
            hTableBufCount = firstS
        }

        // Per-chunk pre-computed data for Phase B
        struct ChunkData {
            var readCounts: [Fr]
            var beta: Fr
            var sum: Fr
            var hRead: [Fr]
            var hTable: [Fr]
            var readChallenges: [Fr]
            var tableChallenges: [Fr]
        }
        var chunkDataArr = [ChunkData]()
        chunkDataArr.reserveCapacity(table.numChunks)

        // Phase A: For each chunk, compute inverse evals, sum, pre-derive challenges
        for k in 0..<table.numChunks {
            let subtable = table.subtables[k]
            let S = subtable.count
            precondition(S > 0 && (S & (S - 1)) == 0, "Subtable size must be power of 2")

            if profileLasso { _tPhase = CFAbsoluteTimeGetCurrent() }

            // Compute read_counts[j] = how many times index j is used
            var countRaw = [UInt64](repeating: 0, count: S)
            let chunkBase = k * m
            indices32Flat.withUnsafeBufferPointer { flatPtr in
                for i in 0..<m {
                    countRaw[Int(flatPtr[chunkBase + i])] &+= 1
                }
            }
            let readCounts: [Fr] = countRaw.map { frFromInt($0) }

            // Derive beta challenge from transcript
            let beta = deriveChallenge(transcript)
            appendFr(&transcript, beta)

            if profileLasso { let _t = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [lasso] chunk %d counts+beta: %.2f ms\n", k, (_t - _tPhase) * 1000), stderr); _tPhase = _t }

            // Compute read-side and table-side inverse evaluations
            let chunkOffset = k * m
            let hReadPtr = hReadBuf!
            let hTablePtr = hTableBuf!

            // Read-side: gather + beta_add + parallel batch_inverse (262144 elements)
            withUnsafeBytes(of: beta) { betaPtr in
                subtable.withUnsafeBytes { stPtr in
                    indices32Flat.withUnsafeBufferPointer { flatPtr in
                        bn254_fr_inverse_evals_indexed(
                            betaPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            stPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            flatPtr.baseAddress! + chunkOffset,
                            Int32(m),
                            hReadPtr
                        )
                    }
                }
            }
            // Table-side: beta_add + batch_inverse + hadamard with readCounts (256 elements)
            withUnsafeBytes(of: beta) { betaPtr in
                subtable.withUnsafeBytes { stPtr in
                    readCounts.withUnsafeBytes { wPtr in
                        bn254_fr_weighted_inverse_evals(
                            betaPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            stPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            wPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(S),
                            hTablePtr
                        )
                    }
                }
            }

            // Copy results to Swift arrays (needed for GPU sumcheck input)
            let hRead = [Fr](unsafeUninitializedCapacity: m) { buf, count in
                memcpy(buf.baseAddress!, hReadPtr, m * frStride)
                count = m
            }
            let hTable = [Fr](unsafeUninitializedCapacity: S) { buf, count in
                memcpy(buf.baseAddress!, hTablePtr, S * frStride)
                count = S
            }

            if profileLasso { let _t = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [lasso] chunk %d inverse_evals (%d+%d): %.2f ms\n", k, m, S, (_t - _tPhase) * 1000), stderr); _tPhase = _t }

            // Compute claimed sum
            var sumLimbs = [UInt64](repeating: 0, count: 4)
            bn254_fr_vector_sum(hReadPtr, Int32(m), &sumLimbs)
            let sum = Fr.from64(sumLimbs)

            appendFr(&transcript, sum)

            let useGPURead = m >= 256
            let logS = Int(log2(Double(S)))
            let useGPUTable = S >= 256

            if useGPURead && useGPUTable {
                // Pre-derive all sumcheck challenges (transcript-only, no GPU blocking)
                var readChallenges = [Fr]()
                readChallenges.reserveCapacity(logM)
                for _ in 0..<logM {
                    let c = deriveChallenge(transcript)
                    readChallenges.append(c)
                    appendFr(&transcript, c)
                }
                var tableChallenges = [Fr]()
                tableChallenges.reserveCapacity(logS)
                for _ in 0..<logS {
                    let c = deriveChallenge(transcript)
                    tableChallenges.append(c)
                    appendFr(&transcript, c)
                }

                if profileLasso { let _t = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [lasso] chunk %d transcript: %.2f ms\n", k, (_t - _tPhase) * 1000), stderr); _tPhase = _t }

                chunkDataArr.append(ChunkData(
                    readCounts: readCounts, beta: beta, sum: sum,
                    hRead: hRead, hTable: hTable,
                    readChallenges: readChallenges, tableChallenges: tableChallenges
                ))
            } else {
                // CPU path: run sumcheck inline with round polys in transcript
                let (readRounds, readFinalEval, readChallenges) = try runSumcheck(
                    evals: hRead, numVars: logM, transcript: &transcript)
                let (tableRounds, tableFinalEval, tableChallenges) = try runSumcheck(
                    evals: hTable, numVars: logS, transcript: &transcript)

                chunkDataArr.append(ChunkData(
                    readCounts: readCounts, beta: beta, sum: sum,
                    hRead: hRead, hTable: hTable,
                    readChallenges: readChallenges, tableChallenges: tableChallenges
                ))
                // For CPU path, add proof directly
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
        }

        if profileLasso { let _t = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [lasso] phase A total: %.2f ms\n", (_t - _tTotal) * 1000), stderr); _tPhase = _t }

        // Phase B: Run sumchecks for chunks that pre-derived challenges.
        // GPU for large read-side (m = 2^18), C CPU for small table-side (S = 256).
        for k in 0..<table.numChunks {
            if subtableProofs.count > k { continue }

            if profileLasso { _tPhase = CFAbsoluteTimeGetCurrent() }
            let cd = chunkDataArr[k]
            let S = table.subtables[k].count
            let logS = Int(log2(Double(S)))

            // Read-side: GPU sumcheck (2^18 elements)
            let (readRounds, readFinalEval) = try sumcheckEngine.fullSumcheck(
                evals: cd.hRead, challenges: cd.readChallenges)

            // Table-side: C CPU sumcheck (256 elements — GPU dispatch overhead not worthwhile)
            let tableRounds: [(Fr, Fr, Fr)]
            let tableFinalEval: Fr
            if S <= 1024 {
                // C path: faster for small inputs, avoids GPU command buffer overhead
                let tRoundsCount = logS * 12  // logS rounds, 3 Fr per round, 4 uint64 per Fr
                var tRoundsRaw = [UInt64](repeating: 0, count: tRoundsCount)
                var tFinalRaw = [UInt64](repeating: 0, count: 4)
                cd.hTable.withUnsafeBytes { evPtr in
                    cd.tableChallenges.withUnsafeBytes { chPtr in
                        bn254_fr_full_sumcheck(
                            evPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(logS),
                            chPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            &tRoundsRaw,
                            &tFinalRaw
                        )
                    }
                }
                var rounds = [(Fr, Fr, Fr)]()
                rounds.reserveCapacity(logS)
                for r in 0..<logS {
                    let b = r * 12
                    rounds.append((
                        Fr.from64([tRoundsRaw[b], tRoundsRaw[b+1], tRoundsRaw[b+2], tRoundsRaw[b+3]]),
                        Fr.from64([tRoundsRaw[b+4], tRoundsRaw[b+5], tRoundsRaw[b+6], tRoundsRaw[b+7]]),
                        Fr.from64([tRoundsRaw[b+8], tRoundsRaw[b+9], tRoundsRaw[b+10], tRoundsRaw[b+11]])
                    ))
                }
                tableRounds = rounds
                tableFinalEval = Fr.from64(tFinalRaw)
            } else {
                let (tr, tf) = try sumcheckEngine.fullSumcheck(
                    evals: cd.hTable, challenges: cd.tableChallenges)
                tableRounds = tr
                tableFinalEval = tf
            }

            if profileLasso { let _t = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [lasso] chunk %d sumchecks (2^%d+2^%d): %.2f ms\n", k, logM, logS, (_t - _tPhase) * 1000), stderr); _tPhase = _t }

            subtableProofs.append(SubtableProof(
                chunkIndex: k,
                readCounts: cd.readCounts,
                beta: cd.beta,
                readSumcheckRounds: readRounds,
                tableSumcheckRounds: tableRounds,
                claimedSum: cd.sum,
                readFinalEval: readFinalEval,
                tableFinalEval: tableFinalEval
            ))
        }

        if profileLasso { fputs(String(format: "  [lasso] total prove: %.2f ms\n", (CFAbsoluteTimeGetCurrent() - _tTotal) * 1000), stderr) }

        // Build [[Int]] indices for proof (deferred from decomposition to avoid hot-path copy)
        let indices: [[Int]] = (0..<table.numChunks).map { k in
            let offset = k * m
            return (0..<m).map { i in Int(indices32Flat[offset + i]) }
        }

        return LassoProof(numChunks: table.numChunks,
                          subtableProofs: subtableProofs,
                          indices: indices)
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
            current.withUnsafeBytes { cBuf in
                withUnsafeBytes(of: r) { rBuf in
                    next.withUnsafeMutableBytes { outBuf in
                        bn254_fr_fold_halves(
                            cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            outBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(half))
                    }
                }
            }
            current = next
        }
        precondition(current.count == 1)
        return current[0]
    }

    // MARK: - Fiat-Shamir

    private func appendFr(_ transcript: inout [UInt8], _ v: Fr) {
        // Direct memcpy of Montgomery representation — faster than frToInt + byte loop.
        // Both prover and verifier use the same representation, so this is safe.
        withUnsafeBytes(of: v) { raw in
            transcript.append(contentsOf: UnsafeBufferPointer(
                start: raw.baseAddress!.assumingMemoryBound(to: UInt8.self),
                count: MemoryLayout<Fr>.size
            ))
        }
    }

    private func appendUInt64(_ transcript: inout [UInt8], _ v: UInt64) {
        var val = v
        withUnsafeBytes(of: &val) { raw in
            transcript.append(contentsOf: UnsafeBufferPointer(
                start: raw.baseAddress!.assumingMemoryBound(to: UInt8.self),
                count: 8
            ))
        }
    }

    private func deriveChallenge(_ transcript: [UInt8]) -> Fr {
        var hash = [UInt8](repeating: 0, count: 32)
        transcript.withUnsafeBufferPointer { ptr in
            blake3_hash_neon(ptr.baseAddress!, ptr.count, &hash)
        }
        var limbs = [UInt64](repeating: 0, count: 4)
        memcpy(&limbs, &hash, 32)
        let raw = Fr.from64(limbs)
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))
    }

}
